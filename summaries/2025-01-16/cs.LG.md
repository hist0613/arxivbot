New uploads on arXiv(cs.CL)

### Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails (https://arxiv.org/abs/2501.09004)
Comments:
          arXiv admin note: text overlap with arXiv:2404.05993

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 상업적 사용을 위한 안전성 관련 데이터셋의 필요성을 강조하고, Aegis 2.0이라는 포괄적이며 적응 가능한 위험 분류체계를 제안합니다. 이 체계는 12개의 주요 위험 카테고리와 9개의 세부 카테고리로 구성되어 있으며, 이는 다운스트림 사용자에게 다양한 위험 유형을 관리할 수 있는 유연한 도구를 제공합니다. 연구팀은 사람의 주석이 포함된 고품질의 데이터셋을 통해 LLM 반응의 안전성을 평가하는 새로운 방법론을 사용합니다.

- **Technical Details**: Aegis 2.0은 34,248개의 샘플을 포함하고 있으며, 이는 각 LLM과의 상호작용을 기반으로 한 것입니다. 이 데이터셋의 고유한 점은 각 데이터가 특정 위험 카테고리에 맞춰 주석이 달려 있다는 것입니다. 또한, 인간 주석자들이 비정형 데이터를 입력하면, 이를 통해 새로운 위험을 식별하고 유기적으로 분류할 수 있는 시스템이 구현됩니다. 이러한 접근 방식이 안전 모델의 유연성을 크게 증가시킵니다.

- **Performance Highlights**: 연구 결과, Aegis 2.0 데이터를 사용하여 파라미터 효율적인 방식으로 훈련된 모델들이 최신 LLM 안전 모델과 경쟁할 만한 성능을 발휘함을 보여주었습니다. 또한, 주제 따르기와 안전성 데이터를 결합한 새로운 훈련 방식이 개발되어 모델의 적응성과 일반화 능력을 향상시켰습니다. 연구팀은 Aegis 2.0 데이터와 모델을 오픈소스로 공개할 계획이며, 이를 통해 LLM 안전성 확보를 위한 연구 커뮤니티의 기여를 촉진하고자 합니다.



### Personality Modeling for Persuasion of Misinformation using AI Agen (https://arxiv.org/abs/2501.08985)
- **What's New**: 이 연구는 개인의 성격 특성이 잘못된 정보에 대한 취약성과 전파에 미치는 영향을 이해하기 위해 혁신적인 에이전트 기반 모델링 접근법을 사용했습니다. 여섯 가지 AI 에이전트가 서로 다른 다섯 가지 성격 차원을 시뮬레이트하며, 실제와 유사한 상호작용을 통해 동적인 잘못된 정보 논의를 분석했습니다. 이 연구의 주요 발견은 분석적인 성격 특성이 증거 기반 논의의 효과를 높이며, 비공격적인 설득 전략이 놀라운 성공을 거둘 수 있다는 것입니다.

- **Technical Details**: 연구에서는 Big Five 성격 특성 모델에서 추출된 성격 차원을 반영한 여섯 개의 특성 에이전트를 설계했습니다. 이 에이전트들은 AgentScope 프레임워크를 통해 여섯 가지 잘못된 정보 주제에 대해 상호작용하며, 각 에이전트는 성격 프로필에 따라 정보의 설득 효과성과 저항 능력을 평가했습니다. 전체적으로 90개의 개별 상호작용이 발생했으며, 에이전트의 결정 과정은 지정된 성격 프로필에 의해 통제되었습니다.

- **Performance Highlights**: 에이전트 간의 상호작용 분석 결과, 성격 프로필과 설득 효과 간에 유의미한 상관 관계가 드러났습니다. 특히 비판적이고 도전적인 특성을 지닌 에이전트 4는 HIV 관련 잘못된 정보 논의에서 59.4%의 성공률을 기록했습니다. 이 결과는 성격 특성이 잘못된 정보에 대한 저항 및 설득에 미치는 복잡한 영향을 강조하며, 개인 고유의 감정적 연결과 신뢰 구축을 우선시하는 효과적인 잘못된 정보 방지 전략의 필요성을 제시합니다.



### Learning to Extract Cross-Domain Aspects and Understanding Sentiments Using Large Language Models (https://arxiv.org/abs/2501.08974)
- **What's New**: 이번 연구에서 우리는 교차 도메인 기반의 감성 분석을 위한 새로운 프레임워크를 제안합니다. 우리는 SemEval-2015 Task 12의 기존 데이터셋을 수정하여 외부 지식 소스와 데이터를 활용하여 교차 도메인 감성 분석에 적용 가능하도록 하였습니다. 이러한 접근 방식은 기업들에게 데이터 과학 기술에 대한 높은 전문 지식 없이도 쉽게 사용할 수 있는 비즈니스 친화적인 모델을 제공합니다.

- **Technical Details**: 제안된 아키텍처는 LLMs (Large Language Models)를 활용하여 제품 및 도메인에 따라 측면을 이해하고 분석하는 기능을 갖추고 있습니다. BERT 모델은 Transformer 구조를 기반으로 하여 문맥 관계를 이해하기 위해 양방향 자기 주의를 사용합니다. 이 연구에서는 특정 데이터셋에 대해 훈련된 BERT 모델을 사용하여 교차 도메인 분석을 수행하고, 이는 기존의 모델들처럼 비싼 파인 튜닝 없이 가능함을 보여주고 있습니다.

- **Performance Highlights**: 우리는 SemEval-2015 Task 12 데이터셋에 대해 92%의 정확도로 Aspect-Based Sentiment Analysis 성능을 달성했습니다. ABSA를 통해 특정 측면에 대한 고객의 의견을 명확히 파악할 수 있을 뿐 아니라, 이를 통해 기업이 고객 기대와 맞춤형 마케팅 전략을 개선하는 데 성장 가능성을 지니고 있습니다.



### Applying General Turn-taking Models to Conversational Human-Robot Interaction (https://arxiv.org/abs/2501.08946)
Comments:
          Accepted at HRI 2025 (the IEEE/ACM International Conference on Human-Robot Interaction)

- **What's New**: 이 논문은 대화에서 턴 테이킹(turn-taking)의 중요성을 강조하고, 기존 HRI(Human-Robot Interaction) 시스템이 자주 간단한 침묵 기반 모델에 의존함으로써 비자연스러운 정지와 인터럽션을 야기했음을 설명합니다. 저자들은 TurnGPT와 Voice Activity Projection(VAP)라는 일반 턴 테이킹 모델을 최초로 HRI에 적용해 대화 역동성을 개선하는 방안을 제시합니다. 이들 모델은 기존 데이터를 통해 학습되어 특정 도메인에 대한 세부 조정 없이도 널리 적용이 가능합니다.

- **Technical Details**: 이 논문에서 소개된 두 가지 일반 턴 테이킹 모델, TurnGPT와 VAP는 서로 보완적인 특성을 가지고 있습니다. TurnGPT는 대화의 구문적 및 의미적 측면을 고려하여 긴급한 의존성(pragmatic dependencies)을 반영하고, VAP는 오디오 데이터를 사용하여 대화의 동태를 지속적으로 예측합니다. 연구자들은 이들 모델을 결합하여 로봇이 반응 준비를 시작하고 턴을 교환하며 잠재적 인터럽션을 처리할 시점을 예측하는 방법을 제안합니다.

- **Performance Highlights**: 39명의 성인을 대상으로 한 실험 결과, 참가자들은 제안된 시스템을 기존 시스템보다 선호했으며, 이는 응답 지연과 인터럽션을 크게 줄였습니다. 효과적인 턴 테이킹 모델이 HRI 설정에서 유용하다는 것을 입증한 이 연구는 로봇과의 자연스러운 상호작용을 위한 중요한 첫걸음으로 평가됩니다. 이러한 발전은 로봇 커뮤니케이션 기술의 향상을 통해 전반적인 사용자 경험을 증진시키는 데 기여할 것으로 기대됩니다.



### GenAI Content Detection Task 3: Cross-Domain Machine-Generated Text Detection Challeng (https://arxiv.org/abs/2501.08913)
Comments:
          COLING 2025

- **What's New**: 이 연구는 새로운 RAID 벤치마크를 사용하여 대규모 언어 모델(LLM)로 생성된 텍스트를 감지할 수 있는 단일 모델의 가능성을 탐구합니다. 이전의 공유 과제가 특정 도메인에 한정된 반면, 본 연구에서는 여러 도메인 및 모델을 사용하여 훈련 중에 모든 정보를 제공하고 탐지기의 경계를 이해하려고 했습니다. 9개의 팀이 참여하여 23개의 탐지기 제출물을 통해 99% 이상의 높은 정확도를 기록하여 여러 모델과 도메인의 텍스트를 효과적으로 탐지할 수 있음을 보여주었습니다.

- **Technical Details**: RAID는 11개의 생성 모델, 8개의 텍스트 도메인 및 11개의 적대적 공격이 포함된 1천만 개 이상의 문서로 구성된 데이터셋입니다. 연구 질문(RQ1 및 RQ2)을 통해 대규모로 텍스트를 생성하는 모델과 도메인이 존재하는 상황에서 탐지력이 어떻게 발휘되는지를 살펴보았습니다. 각 참가 팀은 주어진 도메인에 대한 탐지기를 제출하고, 적대적 공격이 포함된 서브태스크에서 성능을 평가했습니다.

- **Performance Highlights**: 두 팀(Pangram과 Leidos)은 적대적 공격이 없는 상황에서 99.3%의 정확도를 기록하고, 적대적 공격이 있는 경우에도 97.7%의 높은 성능을 보였습니다. 불균형 데이터셋을 고려할 때, 이러한 결과는 탐지기가 여러 도메인과 모델에서 동시에 텍스트를 강력하게 탐지할 수 있는 능력을 입증합니다. 평가 결과는 공개적으로 이용 가능하며, 향후 계획된 벤치마킹 방향이 함께 논의될 것입니다.



### ToMATO: Verbalizing the Mental States of Role-Playing LLMs for Benchmarking Theory of Mind (https://arxiv.org/abs/2501.08838)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 입력된 기존 ToM(Theory of Mind) 벤치마크가 현실과 어긋나는 세 가지 측면을 지적하고, ToMATO라는 새로운 벤치마크를 소개합니다. ToMATO는 대화에 기반한 다중 선택 퀴즈(Multiple-choice QA) 형식으로 구성되어 있습니다. 이 벤치마크는 정보 비대칭을 이용한 LLM-LLM 간의 대화를 통해 생성됩니다.

- **Technical Details**: ToMATO는 믿음(belief), 의도(intention), 욕구(desire), 감정(emotion), 그리고 지식(knowledge)의 다섯 가지 범주를 아우르는 1차 및 2차 정신 상태(mental states)를 포착합니다. 각 발화 전에 역할극(role-playing) LLM이 자신의 생각을 구술해야 하는 프롬프팅 방법을 사용하여, 대화 중 캐릭터들의 정신 상태를 평가하는 질문에 대한 답변을 생성합니다. 정보 비대칭을 통해 다양한 정신 상태에 대한 잘못된 믿음(false beliefs)이 생성될 수 있도록 설계되었습니다.

- **Performance Highlights**: 아홉 개의 LLM을 ToMATO에서 평가한 결과, 특히 잘못된 믿음을 이해하는 데 있어 GPT-4o mini조차도 인간 성능에 미치지 못하는 것으로 나타났습니다. 이 연구는 역할극 LLM 간의 정보 비대칭이 잘못된 믿음을 자주 생성하게 함을 보여주며, 다채로운 인격 특성(personality traits)의 반영 또한 효과적으로 이루어지고 있음을 입증합니다.



### Enhanced Large Language Models for Effective Screening of Depression and Anxiety (https://arxiv.org/abs/2501.08769)
- **What's New**: 이번 논문은 우울증 및 불안 장애와 같은 정신 건강 문제의 조기 식별과 관리를 위한 새로운 접근 방식을 제안합니다. 연구에서는 1,157개의 대화로 구성된 임상 면접의 합성 파이프라인인 PsyInterview를 소개하고, 감정 장애 평가를 위한 LLM 기반 시스템인 EmoScan을 개발했습니다. 이 시스템은 정신 장애를 크게 구분하고 세부적으로 평가하는 기능을 갖추고 있으며, 고품질의 인터뷰를 수행할 수 있도록 설계되었습니다.

- **Technical Details**: EmoScan은 일반 장애(예: 불안 또는 우울 장애)에서 세부 장애(예: 주요 우울 장애)까지 구분할 수 있는 기능이 있습니다. 본 연구에서는 EmoScan의 성능이 기존 모델 및 GPT-4와 같은 다른 LLM보다 우수하다는 것을 보여주었습니다. 평가 결과 EmoScan은 감정 장애를 스크리닝하는 데 있어 F1-score가 0.7467에 달하며, BERTScore는 0.9408로 우수한 설명 능력을 보였습니다.

- **Performance Highlights**: EmoScan은 인터뷰 기술에서 비율된 평가 및 인간 평가 모두에서 탁월한 결과를 기록하며 다른 기준 모델들을 초월했습니다. 또한 외부 데이터셋에 대한 F1-score도 0.67로 견고한 일반화를 자랑합니다. 이 연구는 효과적인 정신 건강 LLM 도구의 개발을 위한 데이터 기반 생성 파이프라인의 중요성을 강조하고 있습니다.



### Expanding Vietnamese SentiWordNet to Improve Performance of Vietnamese Sentiment Analysis Models (https://arxiv.org/abs/2501.08758)
- **What's New**: 이번 논문에서는 베트남어 리뷰의 감정 분석을 위한 새로운 접근 방식을 제안합니다. 이 접근 방식은 PhoBERT-V2와 SentiWordNet을 결합하여 감정 분석 작업의 성능을 개선합니다. 이러한 조합은 언어적 맥락에 최적화된 BERT 방법론을 기반으로 하며, 특정 응용 분야에서의 효과를 극대화합니다.

- **Technical Details**: PhoBERT-V2는 RoBERTa 기반으로 개발되어 BERT의 전훈련 방법을 최적화하여 성능을 개선합니다. 본 연구에서는 SentiWordNet이라는 감정 분류 지원을 위한 어휘 자원을 활용하여, 감정 분석에 필요한 텍스트의 극성을 탐지하고 분석합니다. 이러한 기법들은 베트남어의 고유성을 반영하여 보다 정확한 결과를 도출하는 데 기여합니다.

- **Performance Highlights**: 실험 결과는 VLSP 2016 및 AIVIVN 2019 데이터셋에서 우수한 성과를 보여주었습니다. 우리의 감정 분석 시스템은 기존 모델들과 비교했을 때 현저한 성능 향상을 보였습니다. 이는 제안한 모델의 유효성과 실용성에 대한 강력한 증거로 작용합니다.



### The Inherent Limits of Pretrained LLMs: The Unexpected Convergence of Instruction Tuning and In-Context Learning Capabilities (https://arxiv.org/abs/2501.08716)
Comments:
          The code for this paper is available at: this https URL

- **What's New**: 최근 연구에 따르면, 대형 언어 모델(LLMs)은 다양한 작업을 수행할 수 있는 놀라운 능력을 보여주지만, 인간이 해결할 수 있는 간단한 문제조차도 실패하는 경우가 있다. 본 논문에서는 LLM의 성능에 영향을 미치는 요소들을 분리하여 조사하며, ‘instruction-tuning’이 기초 모델(base models)과는 다른 근본적인 능력을 제공하는지 여부를 살펴본다. 이를 위해 다양한 모델 가족과 작업 유형에서 90개의 LLM에 대한 실험을 수행하여 두 모델 간의 성능 상관관계를 입증한다.

- **Technical Details**: 본 연구에서 ‘instruction tuning’은 LLM들이 사용자 지시를 이해하도록 개선하는 과정을 말한다. 실험을 통해 지시 조정된 모델들이 기본 모델의 성능과 유사하게 제한된 경향을 보인다는 것을 발견하였다. 이는 LLM이 사전 학습 데이터에서의 선험적(prior) 정보에 의존하고 있다는 것을 나타내며, 이러한 경향은 주어진 작업의 성격에 따라 다소 변동적일 수 있다.

- **Performance Highlights**: 실험 결과, instruction-tuned 모델의 성능은 해당 기본 모델의 성능과 상당히 강한 상관관계를 유지하고 있음을 확인했다. 이는 instruction tuning이 새로운 능력을 부여하는 것이 아니라, 주어진 작업에 대한 해석 능력을 개선함으로써 성능을 높인다는 점을 시사한다. 이러한 결과는 향후 LLM의 능력 및 활용 가능성을 연구하는 데 중요한 기초 자료가 될 것으로 기대된다.



### Deep Learning-Based Feature Fusion for Emotion Analysis and Suicide Risk Differentiation in Chinese Psychological Support Hotlines (https://arxiv.org/abs/2501.08696)
- **What's New**: 정신 건강 문제는 전 세계적으로 중요한 공공 건강 문제로, 심리 지원 핫라인은 조기 자살 위험을 식별하고 지원을 제공하는 데 중요한 역할을 합니다. 이 연구는 음성 신호 분석을 통한 감정 표현을 이해하기 위한 새로운 방법을 제안합니다. 벤치마크 데이터셋에서 79.13%의 F1-score를 달성하면서 최신 기술과 비교하여 우수한 성능을 보였습니다.

- **Technical Details**: 본 연구에서는 중국 최대의 심리 지원 핫라인의 데이터를 활용하여 피치(pitch)와 MFCC(Mel-frequency cepstral coefficients) 특징을 결합한 딥 러닝 기반의 특성 추출 방법을 사용하였습니다. 이 통합된 특성 접근법은 감정분류에서 실험적으로 우수한 결과를 나타내었으며, Transformer 아키텍처와 함께 사용되어 감정 인식을 개선했습니다. 이 기법은 음성 신호의 다채로운 음향 정보를 효과적으로 탐색하여 자살 위험 탐지의 정확성을 높이는 목표를 갖고 있습니다.

- **Performance Highlights**: 자살 행동이 있는 46명의 피험자와 그렇지 않은 군을 비교한 결과, 자살 군이 더 잦은 정서적 변화를 보였으나 통계적으로 유의미하지 않았습니다. 연구 결과 도출에 따라 감정의 변동 빈도와 강도가 자살 위험 평가를 위한 새로운 바이오마커가 될 수 있음을 제안합니다. 감정 추세 분석은 전통적인 정적 척도 점수보다 더 동적인 통찰력을 제공하여 정신적 상태의 변화를 실시간으로 포착할 수 있습니다.



### MAGNET: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities (https://arxiv.org/abs/2501.08648)
- **What's New**: 이번 연구에서는 MAGNET(Modified Attention for Generation and Encoding of Text)를 소개합니다. MAGNET는 단방향 디코더 전용 대형 언어 모델(LLMs)을 양방향 모델로 적응시켜 강력한 텍스트 표현을 생성하고, 누락된 텍스트를 채우는 기능을 향상시킵니다. 이는 세 가지 자기 지도 학습(Self-Supervised Learning) 목표와 결합된 주의(attention) 메커니즘을 사용하여 통합 훈련을 가능하게 합니다.

- **Technical Details**: MAGNET는(1) 토큰 수준 및 문장 수준의 표현을 학습하기 위한 마스킹 모델링 목표,(2) 문장 수준 표현을 위한 대조 목표, (3) 누락된 텍스트를 채우기 위한 목표를 설정합니다. 또한, 특별히 설계된 주의 마스크를 통해 양방향 주의과 인과 주의를 결합하여 동시에 훈련할 수 있게 합니다. 이 방식은 LLaMA-2-7B 모델에 적용되며 쉽게 수정 가능함을 보여줍니다.

- **Performance Highlights**: MAGNET로 조정된 LLaMA-2-7B는 토큰 수준 및 문장 수준의 표현 학습 작업에서 기존의 강력한 텍스트 인코더를 초과 달성합니다. 또한, 문맥적으로 적절한 텍스트를 효율적으로 생성할 수 있으며, 반복 문제 없이 개방형 텍스트 생성 능력을 유지합니다. 마지막으로, MAGNET으로 조정된 LLM은 사전 훈련 동안 습득한 지식을 보존합니다.



### Reassessing the Role of Chain-of-Thought in Sentiment Analysis: Insights and Limitations (https://arxiv.org/abs/2501.08641)
- **What's New**: 이 논문은 대형 언어 모델에서 의미적 이해의 관계를 탐구합니다. 기존의 언어와 사고에 대한 두 가지 입장에서, 사고가 언어 모델의 의미적 이해에 미치는 영향을 평가하기 위해 체인 오브 사고(Chain-of-Thought) 방법을 사용한 실험을 수행했습니다. 실험 결과는 이 접근 방법이 감정 분석(sentiment analysis) 과제에서 미미한 영향을 미친다고 나타났습니다.

- **Technical Details**: 연구팀은 언어와 사고의 관계를 탐구하기 위해 체인 오브 사고(CoT) 기법을 감정 기반 분석(aspect-based sentiment analysis) 과제에 적용했습니다. CoT는 모델의 추론 능력을 촉진하기 위해 단계별 추론 과정을 제공하며, 실험에서는 다양한 데이터 세트(Dataset)와 모델을 활용했습니다. 새로운 감정 데이터 세트를 수동으로 구축하여 감정의 복잡성과 변화를 평가하고, CoT가 감정 이해에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과에 따르면, CoT 방법은 가장 작은 모델(Gemma-2, 2B)에서 및 1-shot 설정에서 성능을 개선하는 것으로 나타났습니다. 그러나 전반적으로 CoT가 감정 중심의 의미적 과제에서 미미한 영향을 미친 것으로 결론지어졌습니다. 감정 분석 태스크에서 모델의 성과는 주로 시연에서 제공된 정보에 의해 좌우된다는 사실이 밝혀졌습니다.



### ViBidirectionMT-Eval: Machine Translation for Vietnamese-Chinese and Vietnamese-Lao language pair (https://arxiv.org/abs/2501.08621)
- **What's New**: 이 논문은 VLSP 2022-2023 기계 번역 공유 작업의 결과를 발표하며, 베트남어-중국어 및 베트남어-라오어 기계 번역을 초점으로 하고 있습니다. 이 작업은 베트남어 및 음성 처리(VLSP) 워크숍의 일환으로 진행되었으며, 기계 번역 시스템의 구축을 목표로 하였습니다. 제출된 모델들은 1,000 쌍의 테스트 쌍을 기반으로 평가되었으며, BLEU와 SacreBLEU와 같은 기존 지표들을 사용했습니다.

- **Technical Details**: 기계 번역(Neural Machine Translation, NMT) 시스템은 여전히 번역 품질에서 많은 도전 과제를 안고 있습니다. 이번 공유 작업은 데이터셋을 제공하고, 기계 번역 모델을 테스트할 수 있는 공적인 평가 세트를 포함했습니다. VLSP 2022와 2023에서 각각 베트남어-중국어 및 베트남어-라오어 번역을 위해 다양한 양의 이중 언어 문장 쌍을 포함하여 훈련 데이터를 구성했습니다.

- **Performance Highlights**: VLSP 2022와 2023 기계 번역 작업에 등록된 팀 수는 각각 25개 및 26개 팀이었으며, 높은 품질의 결과물을 제출한 팀들이 평가되었습니다. 기계 번역 작업에서 가장 효과적인 세 가지 접근 방식을 선정하여 그 내용과 기여도를 문서화했습니다. 최종 모델을 구성하는 데 있어 팀들은 데이터 합성을 활용하는 기법을 통해 훈련 세트를 확대하여 정확하고 자연스러운 번역을 달성하였습니다.



### Disjoint Processing Mechanisms of Hierarchical and Linear Grammars in Large Language Models (https://arxiv.org/abs/2501.08618)
- **What's New**: 이 논문은 자연어 처리에서의 계층적 구조에 대한 대형 언어 모델(LLMs)의 반응을 조사합니다. 연구진은 LLMs가 계층적인 문法(hierarchical grammar)과 비계층적인 문법(linear grammar)에서 어떻게 다르게 작용하는지를 실험을 통해 확인하였습니다. 특히, 계층적 문법을 처리하는 메커니즘이 비계층적 문법과는 구별되는 특성을 가진다는 것을 보여주었습니다.

- **Technical Details**: 이 연구에서는 Mistral-v0.3, QWen 2, Llama 2 및 Llama 3.1 모델을 사용하여 실험을 진행하였습니다. 각각의 문법 구조는 영어, 이탈리아어, 일본어를 기반으로 생성되었으며, 총 18개의 문법이 실험되었습니다. LLMs의 계층적 및 비계층적 문법 처리를 비교하기 위해 다양한 입력 패턴과 문법을 통해 실험이 설계되었습니다.

- **Performance Highlights**: 실험 결과, LLMs는 계층적 문법과 비계층적 문법의 입력에 대해 뚜렷한 차이를 보이며, 이들 문법을 처리하는 특정 구성 요소의 역할이 다르다는 것을 확인했습니다. 또한, 새로운(nonce) 문법에서도 계층적 언어 구조에 대한 감수성이 드러나, 이는 의미와 관계없이 언어 데이터에 대한 노출만으로도 기능적 전문화가 가능하다는 결론에 이르게 하였습니다.



### Assessing the Alignment of FOL Closeness Metrics with Human Judgemen (https://arxiv.org/abs/2501.08613)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 First-Order Logic (FOL) 번역의 정확성을 평가하는 다양한 메트릭의 효과를 탐구합니다. 특히 기존 메트릭의 민감도를 조사하여, FOL을 생성할 때 발생하는 이상 현상에 적절히 대응하지 못하는 지점을 확인했습니다. 연구 결과, BLEU와 Logical Equivalency와 같은 일반적으로 사용되는 메트릭들이 FOL 생성의 변형에 대한 대응이 미흡하다는 것을 발견했습니다.

- **Technical Details**: 실험에서는 FOLIO 데이터셋의 훈련 세트를 사용하여, 각 자연어 문장을 FOL로 변환한 결과를 여러 메트릭으로 평가했습니다. 사용한 메트릭에는 BLEU, ROUGE, METEOR, BERTScore, Smatch++ 등이 포함되며, 각 메트릭의 민감도를 분석하기 위해 다양한 변형을 도입했습니다. 결과적으로, BERTScore가 인간의 판단과 더 높은 일치를 보이며, Smatch++가 양화사 변형을 감지하는 데 보다 민감한 성능을 보여주었습니다.

- **Performance Highlights**: 연구 결과는 여러 변형에 대해 메트릭의 응답을 비교하며, 특히 부정 연산이나 연산자 교환에 대한 메트릭의 민감도가 두드러진다는 것을 보여줍니다. BERTScore는 인간 순위와의 상관관계가 강한 반면, Logical Equivalency 점수는 최소한의 일치를 보여주어 FOL 문장을 평가하는 데 있어 의미의 중요성을 강조합니다. 마지막으로, 여러 메트릭을 결합하여 사용할 경우 메트릭의 정렬성과 민감도를 향상시킬 수 있는 가능성을 시사합니다.



### Dynamic Knowledge Integration for Enhanced Vision-Language Reasoning (https://arxiv.org/abs/2501.08597)
- **What's New**: 본 논문에서는 대형 비전-언어 모델(LVLM)의 지식 집합을 동적으로 통합하는 새로운 방법인 Adaptive Knowledge-Guided Pretraining for Large Vision-Language Models (AKGP-LVLM)을 제안합니다. 이 방법은 사전 학습(pretraining)과 미세 조정(fine-tuning) 단계 동안 구조화된 지식과 비구조화된 지식을 LVLM에 통합하여, 시각적 질문 답변 및 추론과 같은 지식 집약적 작업의 성능을 크게 향상시킬 수 있습니다. 특히, knowledge encoder와 retrieval 메커니즘을 활용하여 작업에 관련된 정보를 효과적으로 선택하고 다중 모달과 지식 표현을 정렬할 수 있는 동적인 어댑터를 도입합니다.

- **Technical Details**: AKGP-LVLM의 주요 구성 요소는 두 가지입니다: Knowledge-Guided Multimodal Pretraining과 Dynamic Knowledge Adaptor입니다. 이 단계에서는 외부 지식 베이스를 활용하여 시각적 및 텍스트 입력에 추가적인 의미적 맥락을 제공하며, 작업 별 추론을 최적화하기 위해 모델을 미세 조정합니다. Knowledge encoder는 그래프 기반 및 텍스트 기반 표현을 처리하고, dynamic retrieval 메커니즘이 관련 있는 지식만을 통합하도록 보장하여 모델이 부정확한 정보로 혼란스럽지 않도록 합니다.

- **Performance Highlights**: AKGP-LVLM은 OK-VQA, FVQA, SNLI-VE, NLVR2와 같은 널리 인정받은 벤치마크 데이터셋에서 평가되었으며, 기존의 최첨단 모델들에 비해 성능이 크게 향상되었습니다. 예를 들어, AKGP-LVLM은 OK-VQA에서 이전 방법에 비해 4.56%, NLVR2에서 3.34% 개선된 성적을 달성하였습니다. 또한, 인간 평가 결과 모델의 출력이 더 높은 정확성과 관련성을 가지고 있음을 확인했습니다.



### LoRS: Efficient Low-Rank Adaptation for Sparse Large Language Mod (https://arxiv.org/abs/2501.08582)
Comments:
          12 pages, 4 figures

- **What's New**: 본 논문은 기존의 LoRA 방법이 희소한 대형 언어 모델에서 발생하는 메모리 및 계산 효율성 문제를 극복하기 위한 새로운 접근법인 LoRS를 소개합니다. LoRS는 스팽 있는 모델의 성능을 유지하면서 메모리 및 계산 오버헤드를 최소화하는 혁신적인 방법으로, 웨이트 재계산(weight recompute)과 계산 그래프 재배열(computational graph rearrangement) 기법을 사용합니다. 또한, 더 나은 어댑터 초기화를 통해 효율성을 향상시키고 있습니다.

- **Technical Details**: LoRS 방법은 희소한 LLM을 튜닝하는 과정에서 웨이트 재계산과 계산 그래프 재배열 전략을 통합하여 메모리와 계산 오버헤드를 줄입니다. 이 방법은 포워드 패스 중 적합성을 위한 웨이트를 폐기하고 백워드 패스 중에 이를 재계산하여 메모리 사용량을 크게 줄입니다. 추가적으로, 계산 그래프의 재배열을 통해 SQFT 및 SPP 보다도 계산 오버헤드를 더욱 감소시킵니다.

- **Performance Highlights**: 다양한 희소성 패턴을 가진 LLM에 대해 실험한 결과, LoRS는 기존 SP-LoRA 방법들보다 메모리 사용량과 계산 효율성에서 우수한 성능을 보여주었습니다. 영어 자연어 처리 작업에서 조정된 희소 LLM의 제로샷(zero-shot) 성능이 여러 벤치마크 과제에서 큰 개선을 나타냈습니다. 이를 통해 LoRS는 희소한 모델의 실질적인 유용성을 높이는 중요한 방법으로 자리잡을 가능성이 있습니다.



### What Limits LLM-based Human Simulation: LLMs or Our Design? (https://arxiv.org/abs/2501.08579)
- **What's New**: 이번 연구는 LLM(대규모 언어 모델)을 기반으로 한 인간 시뮬레이션의 한계를 극복하기 위한 프레임워크 설계의 과제를 다루고 있습니다. 기존 연구들은 LLM 기반의 인간 시뮬레이션과 실제 관찰 사이의 간극을 드러내며, 이러한 문제를 해결하기 위한 포괄적인 분석과 제안된 솔루션을 제공합니다. 특히, 데이터 수집 및 평가 방법에서의 미래 방향성을 탐구하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 LLM 기반의 인간 시뮬레이션을 위한 새로운 프레임워크를 제안하고 있습니다. 이 프레임워크는 세 가지 핵심 요소로 구성됩니다: 시뮬레이션 환경(ℰ), 시뮬레이션 에이전트(ℱ), 그리고 시뮬레이션 규칙(ℛ)입니다. LLM은 에이전트의 역할을 하며, 인간 데이터는 평가 기준으로 사용됩니다.

- **Performance Highlights**: 시뮬레이션은 사회적 상호작용, 경제적 행동, 정책 이행 등 다양한 분야에서 최초의 성공을 보여주었습니다. LLM 기반의 시뮬레이션은 고품질의 데이터 생성과 평가 기능을 통해 LLM의 사전 학습 및 시뮬레이션 능력을 향상시키는 데 기여할 수 있습니다. 이러한 기술들은 미래의 연구와 실용적 적용에 있어 중요한 기초를 제공합니다.



### Information Entropy Invariance: Enhancing Length Extrapolation in Attention Mechanisms (https://arxiv.org/abs/2501.08570)
- **What's New**: 이 논문은 정보 엔트로피 불변성(information entropy invariance) 관점에서 길이 외삽(length extrapolation) 능력을 개선하는 새로운 접근 방식을 제안합니다. 기존 연구에서 적용된 스케일링 온도(scaled temperature) 개념에 수학적 정당성을 부여하며, 두 가지 새로운 스케일링 온도인 InfoScale과 CosScale을 도입했습니다. 이 방법론은 LLM의 성능을 개선하고 특히 기존 방법론들보다 우수한 결과를 도출합니다.

- **Technical Details**: InfoScale은 훈련 과정을 필요로 하지 않는 방법으로, 길이 외삽 과정에서 원래 토큰에 대한 집중을 유지합니다. 이어서 CosScale을 도입하여 코사인 주의(cosine attention)에 대한 영향을 이론적으로 분석하였고, InfoScale과 CosScale의 결합이 GAU-α 모델에서 최첨단 성능을 달성하는 것을 확인했습니다. 스케일 증가 시 주의 메커니즘이 창(windowed attention)을 근사하는 결과도 보였습니다.

- **Performance Highlights**: 실험 결과, InfoScale과 CosScale을 결합했을 때 64배의 훈련 길이에 해당하는 컨텍스트 창(context window)에서 성능이 극대화되었습니다. 이 논문은 기존의 RoPE 기반 방법들 및 다양한 길이 외삽 기술들과 비교해 뚜렷한 성과를 보이며, 주의 점수의 희석(attention score dilution)이 긴 범위 контекст 처리를 위한 주요 도전 과제가 됨을 시사합니다.



### Knowledge prompt chaining for semantic modeling (https://arxiv.org/abs/2501.08540)
- **What's New**: 이 논문은 Knowledge Prompt Chaining이라는 새로운 자동 의미 모델링 프레임워크를 제안합니다. 이 프레임워크는 구조화된 데이터와 관련된 도메인 온톨로지를 체계적으로 직렬화하여 Long In-context Learning (LongICL) 시스템 프롬프트에 주입합니다. 이전 연구들이 통합된 그래프를 구성하는 방식과는 달리, 우리는 데이터의 소수 포인트만으로 프롬프트를 적응하여 더욱 효율적인 모델을 구현합니다. 이렇게 하면 사용자는 새로운 소스를 입력할 때마다 의미 레이블과 의미 모델을 자연스럽게 생성할 수 있습니다.

- **Technical Details**: 우리는 주어진 구조화된 데이터와 온톨로지를 활용해 두 단계로 의미 모델링을 수행합니다. 첫 번째 단계는 의미 라벨링으로, 이는 도메인 온톨로지의 클래스와 속성을 사용하여 데이터 속성 열에 주석을 달아 의미 타입을 도출하는 과정입니다. 두 번째 단계는 이러한 주석을 바탕으로 속성 간의 의미 관계를 파악하는 것입니다. 이 과정에서 효율적으로 구조 정보를 보존하고 그라프의 잠재적 표현을 학습하여 사용자 요구에 맞춘 맞춤화된 모델을 생성합니다.

- **Performance Highlights**: 우리의 방법은 세 가지 실제 데이터 세트에서 평가되었으며, 최신 기술들과 비교했을 때 더 나은 성능을 보였습니다. 프롬프트 체인과 가지치기를 적용함으로써, 우리는 더 적은 토큰을 사용해도 의미 모델을 생성할 수 있음을 입증했습니다. 이로 인해 사용자는 전체 테이블이나 JSON, XML 파일을 처리할 필요 없이 작은 데이터 포인트만으로도 의미를 생성할 수 있어 효율성이 크게 향상되었습니다.



### Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers (https://arxiv.org/abs/2501.08537)
Comments:
          Mistakenly submitted as a replacement to 2405.05409v4

- **What's New**: 이 연구는 transformers의 구성적(reasoning-oriented) 문제 해결 메커니즘에 대한 심층 분석을 제공합니다. 특히, 복잡성 제어 전략이 모델이 규칙을 학습하고 일반화하는 방식에 미치는 영향을 조사하였습니다. 기존의 제약을 넘어서, 이 연구에서는 보다 구체적이고 통제된 환경에서 모델의 사고 과정을 분석했습니다.

- **Technical Details**: 연구는 synthetic experimental approach를 사용하여 transformers의 내부 메커니즘을 조사합니다. 데이터 생성 프레임워크를 통해 훈련 데이터, in-distribution (ID) 테스트 데이터, 그리고 out-of-distribution (OOD) 테스트 데이터 사이의 명확한 구분을 도모했습니다. 복잡성 제어는 파라미터 초기화와 가중치 감소 계수를 통해 모델의 복잡성과 추론 능력에 영향을 미치는 전략을 의미합니다.

- **Performance Highlights**: 복잡성 제어 프레임워크를 적용함으로써 OOD 일반화 성능이 눈에 띄게 개선되는 것을 관찰했습니다. 이 연구 결과는 transformers가 다양한 상황에서 고차원 개념을 조합해 새로운 객체를 생성할 수 있는 능력을 향상시키는 방법에 대한 통찰을 제공합니다. 모델의 행동 예측 또한 가능해져, 훈련 과정에서의 모델 동작을 예측할 수 있는 기반을 마련했습니다.



### Doc-Guided Sent2Sent++: A Sent2Sent++ Agent with Doc-Guided memory for Document-level Machine Translation (https://arxiv.org/abs/2501.08523)
- **What's New**: 이 논문에서는 문서 수준 기계 번역(Document-level Machine Translation, DocMT)에 관한 새로운 접근법인 Doc-Guided Sent2Sent++를 제안합니다. 이 모델은 인크리멘탈 문장 수준 강제 디코딩 전략(incremental sentence-level forced decoding strategy)을 사용하여 문장 누락을 방지하고 인접 문장 간의 유창성을 높입니다. 기존의 Doc2Doc 및 Doc2Sent 방법과 차별화된 점은 summary(요약) 정보만을 메모리에 포함하여 번역의 일관성을 유지한다는 것입니다.

- **Technical Details**: Sent2Sent++는 두 개의 인접 문장을 함께 디코딩하는 방식으로 작동합니다. 이 때 이전에 번역된 문장이 현재 디코딩에서 접두사로 사용되며, 이는 문서의 전체적인 번역 유창성을 높여줍니다. 또, 이 모델은 Doc-Guided Memory라 불리는 메모리 구조를 도입하여 전체 문서의 요약과 번역만을 포함하여 효율적인 정보 관리를 실현합니다. 이는 개인 문장 번역의 일관성을 보장하면서도 문서 전반에 걸쳐 번역의 품질을 향상시킵니다.

- **Performance Highlights**: 다양한 언어와 도메인에서 광범위한 테스트를 진행한 결과, Sent2Sent++는 기존 방법들보다 품질, 일관성 및 유창성에서 우수한 성능을 보였습니다. 특히, s-COMET, d-COMET, LTCR-$1_f$, 및 document-level perplexity (d-ppl) 같은 지표에서 현저한 개선이 확인되었습니다. 이 결과는 Sent2Sent++가 다국어 및 다양한 도메인에서 종합적으로 뛰어난 성능을 발휘하는 효율적인 접근법임을 뒷받침합니다.



### Adapting Whisper for Regional Dialects: Enhancing Public Services for Vulnerable Populations in the United Kingdom (https://arxiv.org/abs/2501.08502)
- **What's New**: 본 연구에서는 영국에서의 지역 어악을 포착할 수 있는 최신 자동 음성 인식(ASR) 모델의 성능을 평가하기 위해 공공 서비스 분야에서 새로운 데이터를 수집하였습니다. 스코틀랜드의 두 가지 독특한 방언을 사용하는 지역을 집중 조사하였으며, 이러한 연구는 사회적 취약집단에 속하는 개인들이 겪는 정보 전달의 장애를 해결하는 데 초점을 맞추고 있습니다. WhisperLarge-v3와 같은 최신 ASR 모델을 기반으로 성능을 세밀히 분석하고 있으며, 이는 실질적인 공공 서비스 제공에 있어 중요합니다.

- **Technical Details**: 자동 음성 인식 시스템은 다양한 상황에서 사용되고 있으며, 특히 사회적 언어 편향으로 인한 문제를 다루는 것이 중요하습니다. 연구에서는 Whisper 모델이 복잡한 오디오 환경에서도 낮은 단어 오류율(Word Error Rate, WER)을 보여주며, 다양한 언어 조합에 대한 인식 능력을 갖추고 있음을 강조합니다. 분류 작업에서 WER외에도 인간 검토를 통한 모델 오류 분석을 통해 평가 기법의 장단점을 점검하고 있습니다.

- **Performance Highlights**: Whisper 모델은 수집된 데이터에 대해 기존의 기초 데이터보다 높은 WER를 보였으나, 특정 지역에 대해 세밀하게 조정된 후 성능이 향상된 것으로 나타났습니다. Fine-tuning을 통해 스코틀랜드의 두 지역에 특화된 성능 개선이 가능함을 보였으며, 이는 다른 지역에 있는 데이터세트에도 적용될 수 있음을 시사합니다. 그러나 WER를 평가 기준으로 사용할 때의 한계도 동시에 분석되어, 평가 메트릭의 활용에 대한 논의가 이어집니다.



### Quantifying the Importance of Data Alignment in Downstream Model Performanc (https://arxiv.org/abs/2501.08496)
- **What's New**: 이 논문은 전통적으로 강조되어온 데이터셋 크기 대신, 데이터 정렬(data alignment)의 역할에 주목합니다. 이 정렬은 데이터 품질(data quality)에서 간과되기 쉬운 측면으로, 대규모 언어 모델(Large Language Models, LLM)의 성능 개선에 미치는 영향을 정량화하기 위해 Task2Vec 기반의 정렬 계수(alignment coefficient)를 사용했습니다.

- **Technical Details**: 연구는 두 가지 설정에서 제어된 개입 실험(interventional experiments)을 실시했습니다. 첫 번째는 다양한 프리 트레인(pre-training) 데이터셋과 평가 데이터셋 간의 정렬 계수 증가가 미치는 영향을, 두 번째는 도메인 특화 파인 튜닝(fine-tuning) 데이터셋과 도메인 특화 평가 간의 정렬 계수 증가가 미치는 영향을 분석했습니다. 이러한 설정 중 Autoformalization이라는 특정 도메인 과제를 통해 데이터를 평가하였습니다.

- **Performance Highlights**: 연구 결과 모델의 학습 데이터와 평가 데이터 간의 정렬 계수는 모델의 손실(Loss) 및 혼란도(Perplexity)와 강하게 부정적인 상관관계를 가지며, 이는 각 다운스트림 작업에 대한 모델의 성능에 직접적인 영향을 미친다는 것을 보여줍니다. 특히 데이터가 평가 작업과 잘 정렬된 경우 낮은 혼란도 점수가 나타나며, 이는 LLM 훈련 접근 방식을 재평가할 필요성을 시사합니다.



### The Theater Stage as Laboratory: Review of Real-Time Comedy LLM Systems for Live Performanc (https://arxiv.org/abs/2501.08474)
Comments:
          8 pages, 1st Workshop on Computational Humor (CHum), COLING 2025

- **What's New**: 최근의 코미디 생성과 관련한 연구는 주로 AI 시스템의 실시간 평가에 중점을 두고 있으며, 특히 생중계 환경에서의 공연이 주요한 관심사로 자리잡고 있습니다. 이 논문에서는 생중계 코미디의 도전 과제와 기회를 분석하고, AI가 효율적으로 참여할 수 있는 이상적인 조건을 제시합니다. AI와 인간 간의 협업적 관계를 규명하는 것이 이 연구의 주요 초점입니다.

- **Technical Details**: 이 연구는 'live performance'와 'improvised theater'라는 개념을 바탕으로 하여, AI가 실시간 피드백을 통해 코미디를 생성할 수 있는 가능성을 탐구합니다. 특히, 인간과 로봇 간의 대화 유형, 코미디 타이밍, 관객 반응과 같은 기술적 요소들을 중요한 평가 기준으로 설정하고 있습니다. AI 코미디 생성의 생물학적 근거를 انسانی적 요소와 결합하여 분석하는 것은 이 분야의 주요 연구 질문입니다.

- **Performance Highlights**: 실제 공연 환경에서 AI와 인간 코미디언 간의 상호작용은 코미디 생성에 있어서 중요한 평가 기준으로 작용합니다. 또한, 공연 중 관객의 피드백을 즉각적으로 활용하여 코미디의 질적 향상을 도모할 수 있는 기회를 제시합니다. 과거의 연구들과 비교하여, AI가 코미디 공연에 어떻게 통합되는지에 대한 새로운 통찰력을 제공합니다.



### Selective Attention Merging for low resource tasks: A case study of Child ASR (https://arxiv.org/abs/2501.08468)
Comments:
          To appear in ICASSP 2025

- **What's New**: 본 논문에서는 Speech Foundation Models (SFMs)의 모델 병합 기법을 탐구하여 저자원 환경에서의 아동용 자동 음성 인식(ASR) 성능을 개선하는 방법을 제시합니다. 새로운 Selective Attention (SA) Merge 기법을 도입하여 주의(attention) 행렬에서 작업 벡터(task vectors)를 선택적으로 병합합니다. MyST 데이터베이스에서 수행된 실험을 통해 기존 모델 병합 및 데이터 증대 기법보다 월등한 성능을 보였습니다.

- **Technical Details**: 모델 병합은 특정 도메인에 특화된 여러 모델들을 결합하는 기존의 연구 영역입니다. 본 논문에서 제안하는 SA Merge 기법은 두 모델의 주의 행렬에서 작업 벡터를 병합하는 데 집중합니다. 다양한 병합 기법으로는 Linear Interpolation (Lerp), Spherical Linear Interpolation (Slerp), Task Arithmetic (TA) 등이 있으며, 이들 방법은 서로 다른 방식으로 모델의 파라미터를 결합하고 최적화합니다.

- **Performance Highlights**: 실험 결과, SA Merge 기법이 MyST 데이터베이스에서 상대적인 단어 오류율(Word Error Rate, WER)을 최대 14% 감소시킨 것으로 나타났습니다. Whisper-small 모델에서는 새로운 상태의 WER 8.69를 기록하여 아동 ASR의 성능을 크게 향상시킴을 보여주었습니다. 데이터 증대 기법과 SA Merge의 결합을 통해 저자원 환경에서 ASR 성능을 개선할 수 있는 가능성을 밝히고 있습니다.



### Large Language Models For Text Classification: Case Study And Comprehensive Review (https://arxiv.org/abs/2501.08457)
- **What's New**: 이번 연구에서는 데이터 분류에서의 대규모 언어 모델(LLMs)의 가능성을 탐구하고, 다양한 LLM의 성능을 최신 딥러닝(deep learning) 및 머신러닝(machine learning) 모델과 비교합니다. 특히, 직원의 근무지 분류와 가짜 뉴스 탐지의 두 가지 분류 시나리오에서 LLM의 성과를 평가하며, 각 모델의 성능과 시간 효율성을 종합적으로 분석합니다. 연구 결과, LLM이 복잡한 분류 작업에서 전통적인 방법보다 우수하지만, 추론 시간의 비용이 발생함을 보여줍니다.

- **Technical Details**: 이 연구에서는 여러 LLM과 머신러닝 알고리즘을 비교하여 다중 클래스 분류 문제와 이진 분류 문제에서의 성능을 분석합니다. 모델 크기와 양자화(quantization), 프롬프트 기술이 성능에 미치는 영향을 탐구합니다. 특히, 다양한 프롬프트 기법들이 LLM 성능에 미치는 영향을 분석하고, 적절한 최적화가 결과를 개선할 수 있음을 보여줍니다.

- **Performance Highlights**: 결과적으로, Llama3와 GPT-4 모델은 다중 클래스 분류와 같은 복잡한 작업에서 전통적인 방법보다 뛰어난 성능을 발휘하는 것으로 나타났습니다. 반면, 간단한 이진 분류 작업에서는 보다 전통적인 머신러닝 모델이 성능 대비 시간 효율성에서 우위를 보였습니다. 이 연구는 LLM의 효과적인 사용 사례 및 프롬프트 기술의 중요성을 강조하며, 실제 적용 가능성에 대한 통찰을 제공합니다.



### Jochre 3 and the Yiddish OCR corpus (https://arxiv.org/abs/2501.08442)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구에서는 공개된 Yiddish OCR 코퍼스를 구축하고, 오픈 소스 OCR 도구인 Jochre 3를 평가합니다. 이 도구는 페이지 레이아웃 분석을 위한 YOLOv8 모델과 글리프 인식을 위한 CNN 네트워크를 포함합니다. Jochre 3는 Yiddish 텍스트에서 1.5%의 CER로 다른 기존 모델들보다 훨씬 더 우수한 성능을 보이고 있습니다.

- **Technical Details**: Yiddish OCR 코퍼스는 658 페이지, 186K 토큰, 840K 글리프로 구성되어 있습니다. Jochre 3는 데이터 증강을 통한 훈련을 위해 의도적으로 구축된 코퍼스를 활용하며, 여러 철자 변형과 알파벳을 처리해야 하는 도전 과제를 맞닥뜨리게 됩니다. OCR 과정에서 디아크리틱스를 무시하는 기능과 레이아웃 학습 기능이 포함되어 있습니다.

- **Performance Highlights**: Jochre 3는 테스트 코퍼스에서 1.5%의 CER를 달성하여 Yiddish 분야의 기존 공개 모델층을 능가합니다. 전체 660M 단어 Yiddish Book Center를 분석하여, 새로운 OCR은 Yiddish Book Center OCR 검색 엔진을 통해 검색 가능합니다. 이러한 성능 향상은 역사적인 Yiddish 문서를 디지털화하고 접근할 수 있도록 돕습니다.



### Religious Bias Landscape in Language and Text-to-Image Models: Analysis, Detection, and Debiasing Strategies (https://arxiv.org/abs/2501.08441)
- **What's New**: 이 연구는 종교적 편향(religious bias)에 대한 비판적 검토가 필요한 텍스트 및 이미지 생성 모델(text-to-image generation models)의 편향을 체계적으로 조사합니다. 약 400개의 고유한 프롬프트(prompts)를 사용하여 다양한 작업에서 언어 모델의 편향을 탐색하고, 특정 종교와 관련된 고정관념과 편향의 존재를 밝혀냅니다. 또한, 성별, 나이, 국적과 같은 인구통계적 요인과의 교차-domain 편향(cross-domain biases)도 검토합니다. 이는 이전의 연구와는 달리 종교적 편향을 포괄하고 있습니다.

- **Technical Details**: 이 논문은 100개의 고유한 마스크 채우기(mask filling) 및 프롬프트 완료(prompt completion) 프롬프트를 정교하게 제작하여 여러 pretrained language models와 대형 언어 모델(LLMs)의 편향을 평가합니다. 또한, 각 텍스트-이미지 생성 모델에서 부정적 의미를 가진 형용사에 대해 50개의 편향된 이미지를 생성하여 편향의 정도를 평가하는 등, 언어 모델과 이미지 생성 모델에서의 편향을 상세히 분석합니다. 아래 링크에서 데이터셋과 리소스를 공개하여 후속 연구를 지원합니다: https://github.com/ajwad-abrar/Religious-Bias.

- **Performance Highlights**: 연구 결과, 언어 모델은 텍스트와 이미지 생성 작업 모두에서 여전히 상당한 편향을 보이며, 이러한 편향은 사회적 및 문화적 불평등을 반영합니다. 모델을 통한 종교적 편향 분석은 특정 종교에 대해 편향된 원칙이 어떻게 강화되는지를 보여줍니다. 연구팀은 편향을 줄이기 위해 긍정적 용어 증강(positive term augmentation) 및 편향 완화 지침(bias mitigation instructions) 등의 기술을 사용하고, 이러한 기술이 편향을 줄이는 데 효과적임을 평가합니다.



### Ensemble of Large Language Models for Curated Labeling and Rating of Free-text Data (https://arxiv.org/abs/2501.08413)
- **What's New**: 이 연구는 자유 텍스트 데이터의 라벨링 (labeling) 작업을 개선하기 위해 로컬에서 배포 가능한 작은 규모의 오픈 소스 대형 언어 모델 (LLM)을 사용한 앙상블 (ensemble) 프레임워크를 제안합니다. 기존의 라벨링 방법론은 시간이 많이 걸리고 노동 집약적인 반면, proposed framework는 여러 모델의 다양성을 활용하여 라벨의 일관성을 높이는 방법을 모색합니다. 이를 통해 라벨링 과정의 정확성과 신뢰도를 높이고자 하는 새로운 시도를 보여줍니다.

- **Technical Details**: 연구에서는 7-8억 개의 파라미터를 가진 소형 LLM의 앙상블 기법을 사용하여 기존의 라벨링 방식의 한계를 극복하려고 합니다. 각각 다른 LLM의 라벨 간의 일치도와 불일치도를 비교하는 relevancy scoring methodology를 도입하여, 다양한 LLM의 성과를 조화롭게 통합하는 방식으로 라벨 작업을 수행합니다. 이러한 접근 방식은 인간 평가자에게 각 텍스트에 대한 초기 라벨과 relevancy 점수를 제공하여 라벨링의 노동 부담을 줄이는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 앙상블 방식의 LLM들은 개별 LLM보다 더 높은 정확도로 인간의 라벨링과 일치하는 결과를 보였습니다. 특히, 같은 크기의 LLM들 사이에서도 성능의 이질성이 존재하며, 어떤 모델은 낮은 감수성(高 感数性) 대신 높은 정밀도(高 精密度)를 보이고, 또 다른 모델은 그 반대의 결과를 나타냈습니다. 이러한 발견은 relevancy scoring 방식이 LLM 간의 라벨링에서의 다양성을 효과적으로 완화하였다는 것을 시사합니다.



### MERaLiON-TextLLM: Cross-Lingual Understanding of Large Language Models in Chinese, Indonesian, Malay, and Singlish (https://arxiv.org/abs/2501.08335)
- **What's New**: 이번 연구에서는 MERaLiON-TextLLM이라는 오픈 소스 멀티링구얼 대형 언어 모델이 소개됩니다. 이 모델은 중국어, 인도네시아어, 말레이어 및 싱글리쉬의 이해 및 생성 능력을 개선하기 위해 설계되었습니다. 초기 모델은 Llama-3-8B-Base 기반으로, 지속적인 프리트레이닝(pre-training)과 가중치 병합(weight merging) 과정을 통해 성능을 최적화하였습니다. 이 모델은 공식 Llama-3 모델의 성능을 초과하는 결과를 보여주며, 추가 연구를 위한 자원으로 체크포인트를 제공합니다.

- **Technical Details**: MERaLiON-LLaMA-3-8B-Instruct 모델은 영어, 중국어, 인도네시아어를 중심으로 광범위하게 프리트레이닝되었습니다. 모델 훈련은 MaxText AI-Hypercomputer 플랫폼에서 진행되었으며, NVIDIA H100 GPU와 TPU v4-128 칩스를 사용하여 약 400 TFLOPS의 성능을 달성했습니다. 훈련 데이터는 영어에 38억 개의 토큰, 인도네시아어에 45억, 중국어에 42억 개가 할당되어 있으며, 이는 각 언어 간의 균형 잡힌 성능을 도모합니다.

- **Performance Highlights**: MERaLiON-LLaMA-3-8B-Instruct 모델은 Cross-MMLU 및 Cross-LogiQA 벤치마크에서 기존 Llama-3.1-8B-Instruct 모델을 뛰어넘는 성능을 보였습니다. 영어, 중국어, 인도네시아어에서의 점수는 각각 0.85, 0.69, 0.71로, 최신 다국어 모델 간의 성능 비교에서 우수한 결과를 확인했습니다. 다양한 벤치마크를 통해 MERaLiON 모델의 다국어 및 지침 이행(instruction-following) 능력이 입증되고 있습니다.



### Multimodal LLMs Can Reason about Aesthetics in Zero-Sho (https://arxiv.org/abs/2501.09012)
Comments:
          WIP, Homepage this https URL

- **What's New**: 본 연구에서는 다중모드 LLM(MLLM)의 추론 능력을 활용하여 예술 작품의 미적 평가를 수행하는 최초의 연구를 제시합니다. 이를 위해 MM-StyleBench라는 새로운 고품질 데이터셋을 구축하고, 인간 선호도를 수학적으로 모델링하는 방법을 개발하였습니다. 실험 결과, MLLM이 예술 평가에서 내재적인 환각 문제를 겪으며, ArtCoT라는 방법을 제안하여 MLLM의 미적 추론 능력을 향상할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 MM-StyleBench라는 대규모 주석 데이터셋을 통해 다양한 콘텐츠와 스타일 인스턴스를 평가하였습니다. MLLM의 응답과 인간 선호도의 상관관계를 분석하며, MLLM의 출력이 인간의 선호와 일치하지 않는 주된 문제를 확인했습니다. ArtCoT는 예술 평가의 명확한 하위 과제를 정의하여 환각을 줄이고 MLLM의 추론 능력을 향상시키는 방법입니다.

- **Performance Highlights**: ArtCoT 방법을 적용한 결과, MLLM의 미적 정렬이 일관되게 향상되었음을 보여주었습니다. 특히, 예술 특정 작업의 분해가 MLLM의 추론 능력을 촉진하고 더 객관적인 사고 과정을 이끌어내는 데 기여했습니다. 연구의 결과는 MLLM이 예술 평가 작업에 어떻게 적용될 수 있는지에 대한 귀중한 통찰을 제공하며, 강화 학습을 통한 스타일 전송 및 이미지 생성과 같은 다양한 응용 분야에 도움이 될 수 있습니다.



### Disentangling Exploration of Large Language Models by Optimal Exploitation (https://arxiv.org/abs/2501.08925)
- **What's New**: 이 논문은 탐험(exploration)이 미래의 수익을 증가시킬 수 있도록 정보를 전달하는 것을 목표로 하는 새로운 평가 틀을 제안합니다. 기존 연구들은 주로 탐험과 활용(exploitation) 간의 균형을 중점적으로 다루어왔습니다. 그러나 이 연구는 탐험을 독립적인 목표로 분리하고, 여러 대형 언어 모델(LLM)이 상태 공간(state-space)을 탐색할 때의 성능을 평가합니다.

- **Technical Details**: 탐험 성능을 평가하기 위해, 연구진은 기존 보상의 결여를 탐험과 활용 구성요소로 분해하여 측정하는 방법을 도입했습니다. 이 접근법은 LLM의 탐험 전략을 체계적으로 검토할 수 있는 기반을 제공합니다. 실험 결과, 대부분의 LLM 모델들이 상태 공간을 충분히 탐색하는 데 어려움을 겪고 있으며, 모델 크기와 탐험 성능 사이에 긍정적인 상관관계가 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과는 대부분의 LLM이 독립된 탐험 작업에서 성과를 내는데 한계를 가지며, 이는 탐험과 활용의 트레이드오프를 강조하고 있습니다. 또한, 연구팀은 더 큰 모델이 더 나은 탐험 성능을 보여주는 경향이 있다는 점을 확인했습니다. 이 연구는 LLM의 탐험능력을 평가하고 향상시키기 위한 훌륭한 도구로 작용할 수 있는 분해 방식을 제공합니다.



### MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents (https://arxiv.org/abs/2501.08828)
Comments:
this https URL

- **What's New**: 이번 연구는 Multi-Modal Document Retrieval을 위한 새로운 벤치마크인 MMDocIR을 소개합니다. MMDocIR은 페이지 레벨(page-level)과 레이아웃 레벨(layout-level) 검색의 두 가지 주요 작업으로 구성되어 있으며, 이를 통해 사용자 질문에 대한 더욱 세분화된 답변을 제공할 수 있습니다. 기존의 벤치마크들에서는 미비했던 요소를 보완하여, 문서 내에서의 정확한 검색 성능 평가를 가능하게 합니다.

- **Technical Details**: MMDocIR은 313개의 문서와 1,685개의 질문, 그리고 73,843개의 질문 응답 쌍으로 구성된 훈련 세트를 포함합니다. 본 연구에서는 특히 레이아웃을 정밀하게 표시하기 위한 주석(annotation) 작업을 수행하였으며, 각 페이지에 대한 증거를 포함하는 레이블을 제공합니다. 또한, 비주얼 기반의 검색 시스템과 텍스트 기반 시스템의 성능 차이를 분석하여 비주얼 요소의 중요성을 강조합니다.

- **Performance Highlights**: 엄격한 실험을 통해 비주얼 검색기가 텍스트 검색기보다 상당히 뛰어난 성능을 보인다는 사실을 확인했습니다. 최신 실험 결과는 MMDocIR 훈련 세트가 multi-modal document retrieval 과정에 긍정적인 영향을 미친다는 것을 보여줍니다. 이러한 결과는 비주얼 요소를 통합하는 것이 multi-modal document retrieval를 향상시키는 데 중요한 역할을 한다는 것을 강조합니다.



### SAIF: A Comprehensive Framework for Evaluating the Risks of Generative AI in the Public Sector (https://arxiv.org/abs/2501.08814)
Comments:
          6 pages, 2 figures, 1 tables. AI for Public Missions (AIPM) Workshop at the 39th AAAI Conference on Artificial Intelligence (AAAI 2025)

- **What's New**: 이번 연구에서는 공공 부문에서의 생성 AI의 위험 평가를 위한 체계적인 데이터 생성 프레임워크(SAIF)를 제안합니다. 이 프레임워크는 위험을 분해하고 다양한 시나리오를 설계하는 4단계로 구성되어 있으며, 이를 통해 위험을 체계적이고 일관되게 평가할 수 있도록 합니다. 생성 AI의 다중 모달 기능을 포함한 리스크 분류를 확장하여, 공공 서비스에서의 안전하고 책임 있는 통합을 위한 기초를 제공합니다.

- **Technical Details**: 생성 AI의 리스크는 시스템적 및 운영적 리스크, 콘텐츠 안전 리스크, 사회적 리스크, 법적 및 인권 관련 리스크로 구분됩니다. 시스템 리스크는 AI 시스템의 보안 취약점에서 발생하여 개인 정보 유출의 위험이 있으며, 운영적 리스크는 AI가 본래의 용도에서 벗어나 불공정한 결정을 내릴 수 있는 가능성을 포함합니다. 콘텐츠 안전 리스크는 생성된 콘텐츠가 해롭거나 부적절할 때 발생하며, 사회적 리스크는 개인 데이터의 불법적인 수집이나 보관이 우려됩니다.

- **Performance Highlights**: 생성 AI는 공공 부문에서 행정 효율성을 개선하고 복잡한 의사결정을 지원하는 잠재력을 보여주고 있습니다. 예를 들어, 미국의 이민 관련 질문을 처리하는 챗봇과 같은 성공 사례는 공공 서비스 접근성을 향상시킵니다. 그러나 이러한 기술의 통합은 잘못된 정보나 신뢰 저하 등의 리스크를 초래하므로 신중한 평가가 필요합니다.



### Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching (https://arxiv.org/abs/2501.08686)
Comments:
          Under Review

- **What's New**: KG-RAG4SM 모델은 지식 그래프 기반 검색 증강 생성(rerieval-augmented generation) 방법을 사용하여 스키마 매칭 및 데이터 통합을 수행합니다. 이 모델은 대규모 지식 그래프에서 관련 서브그래프를 식별하는 새로운 벡터 기반, 쿼리 기반 및 그래프 탐색 기반의 검색 방법을 도입하였습니다. KG-RAG4SM은 외부 지식 그래프에서 관련 정보를 수집하여 기존 LLM(large language models)의 성능을 향상시키며, 재훈련 없이 복잡한 매칭 작업을 해결할 수 있습니다.

- **Technical Details**: 이 모델은 벡터 기반 검색, 쿼리 기반 검색 및 그래프 탐색 방법을 통합하여 대규모 KG에서 관련 서브그래프를 수집하고, 순위 체계를 통해 불필요한 지식을 제거합니다. KG-RAG4SM은 LLM의 프롬프트를 강화하여 최종 매칭 결과를 생성하는 데 도움을 주며, 그 과정에서 기존의 LLM 기반 방법들과 비교하여 해상도 및 성능이 월등히 향상된 결과를 보여줍니다. 특히, 헬스케어 데이터베이스의 복잡한 매칭 사례를 처리하는 데 효과적이며, 외부 지식 기초를 활용하여 LLM의 맥락과 의미를 확장합니다.

- **Performance Highlights**: 실험 결과 KG-RAG4SM은 MIMIC 데이터셋에서 LLM 기반 최신 방법(Jellyfish-8B)보다 35.89% 및 30.50% 더 나은 정밀도와 F1 점수를 기록하며, Synthea 데이터셋에서도 PLM 기반 최신 방법(SMAT)을 69.20% 및 21.97% 상회하는 성능을 나타냈습니다. KG-RAG4SM은 대규모 지식 그래프에서 정보를 효율적으로 검색하고, 실제 상황에서의 스키마 매칭 문제를 점진적으로 해결할 수 있는 가능성을 보여줍니다. 이 결과는 복잡한 매칭 상황에서도 LLM의 환각 문제를 효과적으로 완화하는 데 성공했습니다.



### SWSC: Shared Weight for Similar Channel in LLM (https://arxiv.org/abs/2501.08631)
Comments:
          5pages, 3 figures, work in progress

- **What's New**: 본 연구에서는 SWSC라는 새로운 LLM 압축 기법을 제안합니다. 이 방법은 유사 채널의 가중치를 공유하는 개념을 바탕으로 하여 모델의 파라미터 수를 획기적으로 줄입니다. K-Means 클러스터링 알고리즘을 사용하여 유사 벡터들을 군집화하고, 각 군집에서 대표 벡터를 선택하여 여러 벡터를 대체합니다. 이렇게 하여 LLM의 압축 및 배포 효율성을 높이며, 다양한 장치에서 모델을 쉽게 사용할 수 있도록 합니다.

- **Technical Details**: SWSC는 K-Means 클러스터링을 기반으로 하여 모델 가중치를 채널별로 군집화하는 방식으로 작동합니다. 각 군집에서 유사한 벡터를 그룹화하고, 대표 벡터를 선택하여 군집 내 모든 벡터를 대체합니다. 압축 전후의 가중치 오류 값을 특이값 분해(Singular Value Decomposition, SVD)를 통해 계산하고, 중요 특이값 및 벡터를 유지하여 정확도를 보정합니다. 이러한 기술들은 압축된 LLM의 성능을 보장하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 SWSC 방법은 저정밀 조건에서도 LLM의 성능이 크게 저하되지 않는다는 것을 보여줍니다. 기존의 압축 기법과 비교했을 때, SWSC는 모델의 압축 효율성을 대폭 향상시키면서도 성능을 유지하는 데 성공했습니다. 이 새로운 접근법은 LLM의 배포를 용이하게 하여 인공지능 기술의 발전과 보급에 기여할 것으로 기대됩니다.



### RLHS: Mitigating Misalignment in RLHF with Hindsight Simulation (https://arxiv.org/abs/2501.08617)
- **What's New**: 이번 연구는 기존의 Reinforcement Learning from Human Feedback (RLHF) 방법의 한계를 극복하기 위해 Hindsight Feedback에 기초한 새로운 알고리즘인 Reinforcement Learning from Hindsight Simulation (RLHS)를 소개합니다. RLHF는 즉각적인 피드백을 통해 모델을 최적화하려 하였으나, 이로 인해 모델의 행동이 인간의 가치와 잘 align되지 않을 수 있다는 문제를 지적합니다. RLHS는 시뮬레이션된 결과를 통해 피드백을 생성하여 이러한 미스알라인먼트를 효과적으로 완화할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 Partial Observable Markov Decision Process (POMDP)로 모델링된 인간 의사결정 문제를 다릅니다. RLHS 방법은 AI 시스템이 가능한 인간 행동을 시뮬레이션하고 그 결과로부터 피드백을 받는 구조로, 이는 기존의 RLHF와는 다른 접근 방식입니다. RLHS는 Proximal Policy Optimization (PPO)와 Direct Preference Optimization (DPO)과 같은 선호 최적화 방법에 적용되어, 모델의 misalignment를 줄이는 데 효과적임을 실증적으로 입증했습니다.

- **Performance Highlights**: 인간 사용자 연구 결과, RLHS는 RLHF보다 사용자 목표 달성 및 만족도에서 일관되게 우수한 성능을 보였습니다. 시뮬레이션된 피드백만을 사용하였음에도 불구하고, RLHS는 사용자의 진정한 유틸리티를 향상시키며 잘못된 정보를 기반으로 한 결정을 줄이는 데 기여했습니다. 이 결과들은 장기적인 결과에 초점을 맞추는 것이 RLHF의 misalignment을 완화하는 데 중요함을 강조합니다.



### Towards Zero-Shot & Explainable Video Description by Reasoning over Graphs of Events in Space and Tim (https://arxiv.org/abs/2501.08460)
- **What's New**: 최근 머신러닝의 발전 속에서, Transformer가 컴퓨터 비전과 자연어 처리와 같은 다양한 분야에서 주목받고 있습니다. 이 연구는 비전(vision)과 언어(language) 간의 관계를 이해하는 문제에 도전하며, 비전과 언어 모델 간의 연계성을 설명 가능하고 체계적으로 연결하기 위한 새로운 접근법을 제안합니다. 이러한 접근법은 시간과 공간의 사건(event)을 기반으로 하여 자연어로 비디오를 설명하는 문제를 해결하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 비디오의 프레임에서 발생하는 다양한 고수준 정보를 활용하여 Graph of Events in Space and Time (GEST)를 구축합니다. GEST는 비디오의 물리적, 시간적, 의미적 요소를 나타내는 노드(nodes)와 엣지(edges)로 구성되어 있습니다. 이 표현을 통해 비디오는 명확하고 체계적으로 분석되며, proto-language를 생성한 후 이를 자연어 설명으로 변환하는 두 단계의 과정을 거칩니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 데이터셋에서 수집한 비디오에 대해 풍부하고 관련성 높은 텍스트 설명을 생성할 수 있음을 검증했습니다. 기존의 비디오 설명 모델들이 짧은 캡션(caption)만을 생성하는 것과 달리, VLLMs과의 조합을 통해 더 긴 설명을 가능하게 하였습니다. 본 연구에서는 Bleu와 ROUGE 같은 표준 메트릭(Standard metric)을 활용하여 성과를 평가하고 효율성을 입증했습니다.



### Tag&Tab: Pretraining Data Detection in Large Language Models Using Keyword-Based Membership Inference Attack (https://arxiv.org/abs/2501.08454)
- **What's New**: 최근 대규모 언어 모델(LLMs)에서 데이터 유출 감지를 위한 새로운 접근 방식인 Tag&Tab이 제안되었습니다. 기존의 문장 수준 또는 단락 수준 회원 추론 공격(MIAs)은 기초적인 확률 분석에 의존했으나, 이 방법은 텍스트 내용의 의미적 중요성과 단어의 중요성을 고려하지 않았습니다. Tag&Tab은 자연어 처리(NLP) 기술을 활용하여 입력 텍스트에서 키워드를 태그(tagging)하고, 그 확률을 기반으로 평균 로그 우도를 계산하여 텍스트 회원성을 확인합니다.

- **Technical Details**: Tag&Tab 접근법은 다음의 세 가지 주요 단계로 구성됩니다. 첫 번째 단계인 Preprocessing에서는 단어 엔트로피 맵을 구축하여 최적의 키워드 선택을 위해 특정 문장을 필터링합니다. 두 번째 단계인 Tagging에서는 텍스트 내에서 높은 엔트로피를 가진 K개의 단어를 선택하고, 마지막 Tabbing 단계에서는 전체 텍스트를 LLM에 전달하여 이 K개의 키워드의 평균 로그 우도를 계산합니다.

- **Performance Highlights**: Tag&Tab은 세 가지 벤치마크 데이터셋(BookMIA, MIMIR, The Pile)과 여러 개의 오픈소스 LLM을 대상으로 실험을 수행한 결과, 기존의 최첨단 기술보다 AUC 점수가 평균 4.1%에서 12.1% 향상된 것으로 나타났습니다. 이 연구는 LLM의 사전학습 데이터 유출 탐지에 대한 새로운 기준을 제시하며, 단어의 중요성이 MIAs에서 결정적인 역할을 한다는 사실을 입증했습니다.



### SEAL: Speaker Error Correction using Acoustic-conditioned Large Language Models (https://arxiv.org/abs/2501.08421)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 이 논문은 Speaker Diarization(SD) 시스템의 성능 향상을 위해 Acoustic-conditioned Large Language Models(LLMs)를 활용하는 새로운 접근 방식을 제안합니다. 기존 SD 시스템에서 발생하는 화자 오류를 줄이기 위해 음향 정보를 LLM과 결합합니다. 또한, 복잡한 후처리 없이 LLM의 환각을 줄일 수 있는 단순한 제한 디코딩 전략을 도입하여 더 정확한 화자 레이블을 할당합니다.

- **Technical Details**: 메인 아이디어는 SEAL이라는 프레임워크를 사용하여 EEND(End-to-End Neural Diarization) 네트워크에서 제공하는 음향적 후행 확률을 활용하는 것입니다. 각 단어에 대한 화자 포스터리어 성능을 네트워크를 통해 포착하고, 이를 바탕으로 LLM이 보다 정확한 결과를 도출하도록 합니다. 또한, 화자 확률을 직관적으로 이해하기 쉬운 레이블로 변환하고, 이를 통해 LLM이 더 나은 성능을 발휘하도록 해줍니다.

- **Performance Highlights**: 제안된 접근 방식은 Fisher, Callhome, RT03-CTS 데이터셋을 통해 기존 Acoustic SD와 비교할 때 화자 오류율을 24-43%까지 감소시키는 효과를 보여주었습니다. 이러한 결과는 SEAL이 LLM의 내재된 어휘 지식을 음향 정보와 통합함으로써 발생하는 큰 개선을 반영합니다. 연구 결과는 다양한 다중 화자 전사 응용 프로그램에 이 혁신적인 방법이 실질적인 기여를 할 수 있음을 보여줍니다.



### OptiChat: Bridging Optimization Models and Practitioners with Large Language Models (https://arxiv.org/abs/2501.08406)
- **What's New**: 본 논문은 최적화 모델을 해석하고 설명하기 위해 자연어 대화 시스템인 OptiChat을 소개합니다. OptiChat은 최적화 전문가의 도움 없이 실제 적용 분야에서 작업하는 사용자들이 모델을 이해하고 진단할 수 있도록 돕기 위해 설계되었습니다. 이를 통해 최적화 모델과 사용자 간의 상호작용을 원활하게 하고, 사용자가 직면한 문제를 보다 잘 해결할 수 있는 것을 목표로 합니다.

- **Technical Details**: OptiChat은 다양한 사전 정의된 함수와 코드 생성을 결합하여 최적화 모델의 해석을 지원합니다. 사용자가 입력한 최적화 코드에는 Pyomo/Python 언어가 사용되며, 쿼리를 효과적으로 처리하기 위해 각 사용자 쿼리 유형에 따라 질의 전략을 구현합니다. 사용자는 진단, 검색, 민감도, 가정 질문(what-if) 및 비가정 질문(why-not) 등 다양한 쿼리를 통해 모델에 대한 정보를 요청할 수 있습니다.

- **Performance Highlights**: OptiChat은 24개의 최적화 모델을 테스트한 결과, 전문가와 비교해 짧은 시간 내에 정확한 모델 설명과 사용자 쿼리에 대한 응답을 제공하는 것으로 나타났습니다. 사용자가 모델을 독립적으로 해석하지 못할 경우 OptiChat이 그 부담을 덜어주며, 신뢰할 수 있는 자동화된 모델 설명을 통해 최적화 전문가의 시간을 절약할 수 있습니다. 최종적으로 OptiChat의 설명 품질은 전문가가 제공하는 설명과 유사한 수준으로 평가되었습니다.



### Towards Best Practices for Open Datasets for LLM Training (https://arxiv.org/abs/2501.08365)
- **What's New**: 이번 논문에서는 인공지능(AI) 기업들이 저작권 소유자의 동의 없이 대규모 언어 모델(LLM)을 훈련시키고 있는 현황을 다룹니다. 이러한 행위의 허용 여부는 지역에 따라 다르며, EU와 일본 같은 국가에서는 특정 제한 하에 허용되고 있지만, 미국에서는 법적 경관이 모호합니다. 이로 인해 창작자들이 제기하는 저작권 소송이 증가하고 있으며, 이는 기업 및 공공기관이 훈련 데이터셋에 대한 정보를 축소하는 최근의 경향에 영향을 미치고 있습니다.

- **Technical Details**: 저작권 논란에도 불구하고, AI 모델의 투명성과 책임성을 저해하는 정보 공유 제한은 연구자, 감사인 및 피해자들이 AI 모델을 이해하는 데 필요한 정보에 접근하는 데 문제를 일으킵니다. 이러한 문제는 공개 접근(open access) 및 공공 도메인(public domain) 데이터를 기반으로 하는 언어 모델의 훈련으로 완화될 수 있지만, 현재 이에 대한 기술적 및 사회적 도전으로 인해 의미 있는 규모로 훈련된 모델은 없습니다. 데이터 조합에 필요한 부정확하고 불완전한 메타데이터, 물리적 기록의 디지털화(digitization) 비용과 복잡성, 빠르게 변화하는 환경에서 관련성과 책임성을 보장하기 위한 다양한 법적 및 기술적 기술들이 그 장애 요인입니다.

- **Performance Highlights**: AI 시스템이 책임감 있게 관리되고 큐레이션된 공개 라이센스 데이터로 훈련될 수 있는 미래를 구축하기 위해서는 법적, 기술적, 정책적 분야 간의 협력이 필수적입니다. 메타데이터 표준, 디지털화 및 개방성 문화 촉진에 대한 투자도 중요합니다. 이러한 통합적인 접근이 이루어져야만 AI 모델의 사용과 관련된 데이터 안전성과 책임이 보장될 수 있습니다.



New uploads on arXiv(cs.IR)

### MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents (https://arxiv.org/abs/2501.08828)
Comments:
this https URL

- **What's New**: 이번 연구는 Multi-Modal Document Retrieval을 위한 새로운 벤치마크인 MMDocIR을 소개합니다. MMDocIR은 페이지 레벨(page-level)과 레이아웃 레벨(layout-level) 검색의 두 가지 주요 작업으로 구성되어 있으며, 이를 통해 사용자 질문에 대한 더욱 세분화된 답변을 제공할 수 있습니다. 기존의 벤치마크들에서는 미비했던 요소를 보완하여, 문서 내에서의 정확한 검색 성능 평가를 가능하게 합니다.

- **Technical Details**: MMDocIR은 313개의 문서와 1,685개의 질문, 그리고 73,843개의 질문 응답 쌍으로 구성된 훈련 세트를 포함합니다. 본 연구에서는 특히 레이아웃을 정밀하게 표시하기 위한 주석(annotation) 작업을 수행하였으며, 각 페이지에 대한 증거를 포함하는 레이블을 제공합니다. 또한, 비주얼 기반의 검색 시스템과 텍스트 기반 시스템의 성능 차이를 분석하여 비주얼 요소의 중요성을 강조합니다.

- **Performance Highlights**: 엄격한 실험을 통해 비주얼 검색기가 텍스트 검색기보다 상당히 뛰어난 성능을 보인다는 사실을 확인했습니다. 최신 실험 결과는 MMDocIR 훈련 세트가 multi-modal document retrieval 과정에 긍정적인 영향을 미친다는 것을 보여줍니다. 이러한 결과는 비주얼 요소를 통합하는 것이 multi-modal document retrieval를 향상시키는 데 중요한 역할을 한다는 것을 강조합니다.



### $\texttt{InfoHier}$: Hierarchical Information Extraction via Encoding and Embedding (https://arxiv.org/abs/2501.08717)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문은 Self-Supervised Learning(SSL)과 Hierarchical Clustering(HC)을 통합한 InfoHier라는 새로운 프레임워크를 제안합니다. 기존 SSL 방법은 비층적 구조에 집중하며, 다차원 데이터의 복잡한 관계를 반영하지 못했습니다. HC는 계층적 데이터를 이해하는 데 유리하지만, 엄격한 유사도 기반 메트릭에 의존해 한계가 있었습니다.

- **Technical Details**: InfoHier는 SSL을 통해 적응형 표현을 제공하고, HC가 복잡한 패턴을 캐치할 수 있도록 도와줍니다. 이를 위해 각 데이터 포인트에 대해 루트 이진 트리 형태의 계층적 구조를 추출하며, 이는 잠재 표현(latent representation)을 통해 이루어집니다. 이 과정에서 Dasgupta 손실을 사용하여 클러스터 성능을 향상시키고, 대조 손실(contrastive loss)을 결합하여 Encoder 네트워크를 더욱 견고하게 조정합니다.

- **Performance Highlights**: InfoHier는 클러스터링과 표현 학습 모두에서 성능을 향상시킬 수 있는 가능성을 지닙니다. 이번 연구에서 제안하는 방법은 복잡한 데이터셋에서 정보 계층 구조를 반영한 보다 효율적인 분석을 가능하게 하여 정보 검색 및 데이터 관리를 효과적으로 수행할 수 있습니다. 이 연구는 대규모 무라벨 데이터에 대한 정보 추출의 현재 한계를 극복하는 데 기여할 것입니다.



### Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever (https://arxiv.org/abs/2501.08695)
- **What's New**: 이 논문에서는 새로운 검색 패러다임으로서 스트리밍 벡터 양자화 모델(streaming Vector Quantization model)을 제안합니다. 이 모델은 실시간으로 항목을 색인에 연결하여 유연성과 즉각성을 제공합니다. 그 결과, 기존의 복잡한 랭킹 모델을 지원하면서도 색인 균형(index balancing) 및 수리 가능성(reparability)을 확보합니다. 특히, 이 접근법은 Douyin과 Douyin Lite에서 기존의 주요 검색 모델을 대체하여 사용자 참여도를 높였습니다.

- **Technical Details**: 스트리밍 벡터 양자화 모델은 실시간으로 항목을 색인에 적절하게 배치하는 기능을 가지고 있습니다. 기존의 HNSW 구조는 주기적으로 재구성해야 하며, 이는 새로운 항목과 클러스터 의미의 변화를 포착하는 데 한계를 가지고 있습니다. 또한, 이 모델은 모든 클러스터가 추천 프로세스에 참여할 수 있도록 하는 병합 정렬(merge-sort) 수정을 통해 균형 잡힌 색인을 제공합니다. 이에 따라, 복잡한 모델 및 다중 작업 학습(multi-task learning)과도 뛰어난 호환성을 보장합니다.

- **Performance Highlights**: 스트리밍 VQ는 Douyin과 Douyin Lite에서 모든 주요 검색기를 대체하였으며, 그 결과 사용자 참여도가 크게 향상되었습니다. 기존의 검색기들보다 더 나은 성능을 발휘하며, 구현이 용이한 아키텍처를 통해 대규모 시스템에서 손쉽게 배포할 수 있습니다. 이 모델은 색인 초기화 및 수리를 필요로 하지 않으며, 실시간으로 업데이트할 수 있는 강점을 가지고 있습니다.



### Continuous Approach to Phase (Norm) Retrieval Frames (https://arxiv.org/abs/2501.08927)
- **What's New**: 이번 논문은 Hilbert 공간에서의 phase retrieval와 norm retrieval에 초점을 맞춰 continuous frames의 특성을 조사합니다. 새로운 개념인 continuous near-Riesz bases를 도입하고, invertible operators에 대한 불변성을 입증합니다. 결과적으로, continuous frames의 phase 및 norm retrieval 속성에 대한 동등한 조건들도 제시됩니다.

- **Technical Details**: continuous frames는 일반화된 프레임으로, Radon measures를 갖춘 지역적으로 콤팩트한 공간에 확장된 인덱스 집합을 가지고 있습니다. 논문은 Hilbert 공간의 측정 공간에서 Bessel mappings의 norm bound를 설정하고, continuous near-Riesz bases와 그들의 불변성을 탐구합니다. 특히, perturbation 아래 phase retrieval의 안정성에 대한 연구가 포함되어 있습니다.

- **Performance Highlights**: 연구 결과는 tensor product Hilbert 공간의 tensor product frames에 대해 phase와 norm retrieval 속성이 서로 어떻게 관련되어 있는지를 밝힙니다. 이러한 발견은 신호 처리와 양자 역학 분야에서 중요한 통찰력을 제공합니다. 이 논문은 continuous frames 및 그들의 retrieval 속성에 대한 포괄적인 프레임워크를 제공하여, 실제 도전 과제를 해결하는 데 있어 새로운 통찰력과 도구를 제공합니다.



### Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching (https://arxiv.org/abs/2501.08686)
Comments:
          Under Review

- **What's New**: KG-RAG4SM 모델은 지식 그래프 기반 검색 증강 생성(rerieval-augmented generation) 방법을 사용하여 스키마 매칭 및 데이터 통합을 수행합니다. 이 모델은 대규모 지식 그래프에서 관련 서브그래프를 식별하는 새로운 벡터 기반, 쿼리 기반 및 그래프 탐색 기반의 검색 방법을 도입하였습니다. KG-RAG4SM은 외부 지식 그래프에서 관련 정보를 수집하여 기존 LLM(large language models)의 성능을 향상시키며, 재훈련 없이 복잡한 매칭 작업을 해결할 수 있습니다.

- **Technical Details**: 이 모델은 벡터 기반 검색, 쿼리 기반 검색 및 그래프 탐색 방법을 통합하여 대규모 KG에서 관련 서브그래프를 수집하고, 순위 체계를 통해 불필요한 지식을 제거합니다. KG-RAG4SM은 LLM의 프롬프트를 강화하여 최종 매칭 결과를 생성하는 데 도움을 주며, 그 과정에서 기존의 LLM 기반 방법들과 비교하여 해상도 및 성능이 월등히 향상된 결과를 보여줍니다. 특히, 헬스케어 데이터베이스의 복잡한 매칭 사례를 처리하는 데 효과적이며, 외부 지식 기초를 활용하여 LLM의 맥락과 의미를 확장합니다.

- **Performance Highlights**: 실험 결과 KG-RAG4SM은 MIMIC 데이터셋에서 LLM 기반 최신 방법(Jellyfish-8B)보다 35.89% 및 30.50% 더 나은 정밀도와 F1 점수를 기록하며, Synthea 데이터셋에서도 PLM 기반 최신 방법(SMAT)을 69.20% 및 21.97% 상회하는 성능을 나타냈습니다. KG-RAG4SM은 대규모 지식 그래프에서 정보를 효율적으로 검색하고, 실제 상황에서의 스키마 매칭 문제를 점진적으로 해결할 수 있는 가능성을 보여줍니다. 이 결과는 복잡한 매칭 상황에서도 LLM의 환각 문제를 효과적으로 완화하는 데 성공했습니다.



### DNMDR: Dynamic Networks and Multi-view Drug Representations for Safe Medication Recommendation (https://arxiv.org/abs/2501.08572)
- **What's New**: 이 논문은 동적 네트워크와 다중 뷰 약물 표현을 통합한 새로운 약물 추천(Medication Recommendation, MR) 방법인 DNMDR를 제안합니다. 기존 MR 시스템의 한계를 극복하기 위해, 시간적 EHR(전자 건강 기록) 데이터를 기반으로 한 동적 이질 네트워크의 가중 스냅샷 시퀀스를 구성하였고, 모든 동적 네트워크는 환자의 다양한 의료 사건에서 구조적 상관관계와 시간적 의존성을 학습하도록 공동 훈련되었습니다. 이렇게 구성된 방법론은 환자 표현을 개선하는 데 기여하며, 약물 간 상호 작용을 줄이고 보다 안전한 약물 추천을 가능하게 합니다.

- **Technical Details**: 연구에 사용된 DNMDR 모델은 동적 네트워크 및 다중 뷰 약물 그래프를 기반으로 하여, 임상 이벤트의 구조적 관계와 역사적 건강 상태의 시간적 의존성을 동시적으로 탐색합니다. 또한, 약물의 화학적 구조와 상호작용 정보를 통합하여 정확하고 안전한 약물 조합 추천을 지원합니다. 이전의 의료 경험 및 기존 데이터베이스에서 얻은 알려진 약물 간 상호작용 정보를 활용하여, 약물 이체의 안전성을 보장합니다.

- **Performance Highlights**: 실제 EHR 데이터셋에서 실시된 광범위한 실험 결과, DNMDR 방법은 다양한 평가 지표(예: PRAUC, Jaccard, DDI 비율)에서 기존의 최신 기준 모델보다 큰 폭으로 성능을 개선하는 것으로 나타났습니다. 이로 인해, DNMDR은 의료 분야에서 약물 추천 시스템의 실제 적용 가능성을 높이는 결과를 도출하였습니다.



New uploads on arXiv(cs.CV)

### Ouroboros-Diffusion: Exploring Consistent Content Generation in Tuning-free Long Video Diffusion (https://arxiv.org/abs/2501.09019)
- **What's New**: 최근 FIFO(Fast In Fast Out) 비디오 확산(video diffusion) 기술은 사전 훈련된 텍스트-비디오 모델을 기반으로 하여 튜닝 없이 긴 비디오 생성에 효과적인 접근 방식으로 발전했습니다. 본 논문에서는 구조적 및 내용 일관성(content consistency)을 개선하기 위한 새로운 비디오 노이즈 제거 프레임워크인 Ouroboros-Diffusion을 제안합니다. 이 프레임워크는 긴 비디오 생성에서 콘텐츠 일관성을 향상시키기 위해 새로운 잠재 샘플링 기법을 도입하며, 구조적 일관성을 유지하는 데 중점을 두고 있습니다.

- **Technical Details**: Ouroboros-Diffusion에서 고안된 Self-Aware Cross-Frame Attention(SACFA) 메커니즘은 짧은 구간 내에서 프레임 간 주제를 정렬하여 시각적 일관성을 개선합니다. 이는 각 프레임에서 주제 토큰을 추출하여 주제를 일관되게 유지하는 데 도움을 줍니다. 또한, 자가 회귀 안내(self-recurrent guidance) 기술을 활용하여 대기열 앞쪽의 모든 이전 고화질 프레임의 정보를 수집함으로써 후속 프레임의 노이즈 제거를 안내합니다.

- **Performance Highlights**: VBench 벤치마크에서 긴 비디오 생성을 위한 광범위한 실험을 통해 Ouroboros-Diffusion의 우수성을 검증했습니다. 특히, 주제 일관성(subject consistency), 움직임의 부드러움(motion smoothness), 그리고 시간적 일관성(temporal consistency) 면에서 뛰어난 성능을 보여주었습니다. 결과적으로, 이 접근 방식은 긴 비디오 생성 과정에서 시각적 및 동작 품질을 향상시키는 데 기여합니다.



### Multimodal LLMs Can Reason about Aesthetics in Zero-Sho (https://arxiv.org/abs/2501.09012)
Comments:
          WIP, Homepage this https URL

- **What's New**: 본 연구에서는 다중모드 LLM(MLLM)의 추론 능력을 활용하여 예술 작품의 미적 평가를 수행하는 최초의 연구를 제시합니다. 이를 위해 MM-StyleBench라는 새로운 고품질 데이터셋을 구축하고, 인간 선호도를 수학적으로 모델링하는 방법을 개발하였습니다. 실험 결과, MLLM이 예술 평가에서 내재적인 환각 문제를 겪으며, ArtCoT라는 방법을 제안하여 MLLM의 미적 추론 능력을 향상할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 MM-StyleBench라는 대규모 주석 데이터셋을 통해 다양한 콘텐츠와 스타일 인스턴스를 평가하였습니다. MLLM의 응답과 인간 선호도의 상관관계를 분석하며, MLLM의 출력이 인간의 선호와 일치하지 않는 주된 문제를 확인했습니다. ArtCoT는 예술 평가의 명확한 하위 과제를 정의하여 환각을 줄이고 MLLM의 추론 능력을 향상시키는 방법입니다.

- **Performance Highlights**: ArtCoT 방법을 적용한 결과, MLLM의 미적 정렬이 일관되게 향상되었음을 보여주었습니다. 특히, 예술 특정 작업의 분해가 MLLM의 추론 능력을 촉진하고 더 객관적인 사고 과정을 이끌어내는 데 기여했습니다. 연구의 결과는 MLLM이 예술 평가 작업에 어떻게 적용될 수 있는지에 대한 귀중한 통찰을 제공하며, 강화 학습을 통한 스타일 전송 및 이미지 생성과 같은 다양한 응용 분야에 도움이 될 수 있습니다.



### SimGen: A Diffusion-Based Framework for Simultaneous Surgical Image and Segmentation Mask Generation (https://arxiv.org/abs/2501.09008)
Comments:
          12 pages, 17 figures, 4 tables, project page at this https URL

- **What's New**: 이 연구에서는 SimGen이라는 새로운 작업과 방법을 제안합니다. SimGen은 동시 이미지 및 마스크 생성을 위한 확산 모델로, 고품질 외과 이미지를 생성하며 해당하는 분할 마스크(예: segmentation masks)도 함께 생성합니다.

- **Technical Details**: SimGen은 Denoising Diffusion Probabilistic Models (DDPM) 기반으로 구축되며, 잔여 U-Net 아키텍처를 채택하여 그래디언트 흐름을 개선하고 모델 수렴을 안정화시킵니다. 크로스-상관(즉, cross-correlation) 우선순위를 활용하여 연속 이미지 데이터와 이산 마스크 분포 간의 의존성을 캡처하는 방식으로 관련성을 유지합니다.

- **Performance Highlights**: SimGen은 여섯 개의 공공 데이터셋을 사용하여 평가된 결과, 기존 모델 대비 이미지 품질 및 마스크 정확성에서 우수한 성능을 보여주었습니다. 특히, 생성된 이미지-마스크 쌍은 윤곽 정렬을 기초로 한 경계 인식이 가능하며, 실제 데이터가 제한된 상황에서도 유용하게 활용될 수 있습니다.



### RepVideo: Rethinking Cross-Layer Representation for Video Generation (https://arxiv.org/abs/2501.08994)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 비디오 생성 분야에서 변별력 있는 발전을 가져온 새로운 접근 방식인 RepVideo를 제안합니다. 이 프레임워크는 텍스트-비디오 확산 모델을 위해 중간 레이어에서의 특징을 집계하여 더욱 안정적인 의미 표현을 생성합니다. RepVideo는 인접한 프레임 간의 일관성을 유지하면서도 고차원적인 공간적 표현을 향상시키는 데 기여합니다. 이를 통해 비디오의 품질과 동적 일관성을 동시에 개선하고자 합니다.

- **Technical Details**: RepVideo 구조는 여러 인접 변환기 레이어의 특징을 집계하고 평균 집계 전략을 통해 안정적인 의미 표현을 달성합니다. 이 과정에서 게이팅 메커니즘을 사용하여 집계된 표현을 원래의 변환기 입력과 결합함으로써 각 변환기 레이어에 대해 향상된 특징 입력을 생성하게 됩니다. 이러한 방식으로 RepVideo는 인접한 프레임 간의 세부 특징을 일관되게 유지하며, 비디오 생성에서 중요한 시간적 일관성을 개선할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 RepVideo의 효과가 입증되었습니다. 실험 결과 RepVideo는 시간적 일관성과 공간적 세부 표현 생성 모두에서 경쟁력 있는 성능을 기록하며, 고품질의 비디오 생성이 가능하다는 것을 보여주었습니다. 이러한 성능 향상은 비디오 생성 기술 발전에 중요한 기여를 할 것으로 기대됩니다.



### CityDreamer4D: Compositional Generative Model of Unbounded 4D Cities (https://arxiv.org/abs/2501.08983)
- **What's New**: 본 연구에서는 CityDreamer4D라는 독창적인 생성 모델을 제안합니다. 이 모델은 동적 객체(예: 차량)와 정적 장면(예: 건물, 도로)을 분리하여 넓은 범위의 4D 도시를 생성하는 데 중점을 둡니다. 또한, 다양한 물체를 구성하는 데 필요한 다양한 신경 필드를 응용하여 높은 품질의 4D 도시 이미지를 생성합니다.

- **Technical Details**: CityDreamer4D는 다수의 생성 모듈을 포함하여 복잡한 도시 장면을 처리합니다. Traffic Scenario Generator와 Unbounded Layout Generator를 통해 동적 교통 시나리오와 정적 도시 구성을 생성하고, BEV(bird's-eye view)를 활용하여 효율적인 데이터 표현을 제공합니다. 또한, 각 객체의 특징에 맞춘 커스터마이즈된 생성 해시 격자(generative hash grids)와 주기적 위치 임베딩을 적용하여 고유한 장면 파라미터화를 가지고 있습니다.

- **Performance Highlights**: 이 모델은 OSM, GoogleEarth 및 CityTopia와 같은 다양한 데이터세트를 통해 훈련되어 있으며, 사실적인 4D 도시 생성에서 최첨단 성능을 입증합니다. CityDreamer4D는 인스턴스 편집, 도시 스타일화 및 도시 시뮬레이션과 같은 다양한 후속 응용 프로그램을 지원하여 도시 생성의 범위를 확장합니다.



### CityLoc: 6 DoF Localization of Text Descriptions in Large-Scale Scenes with Gaussian Representation (https://arxiv.org/abs/2501.08982)
- **What's New**: 이번 논문에서 우리는 대규모 3D 장면에서 텍스트 설명에 따른 카메라 포즈 분포를 생성하는 방법을 제안합니다. 기존의 표준 검색 방법 및 학습 기반 접근 방식과 비교하여, 우리의 방법은 이들을 뛰어넘는 성능을 보여줍니다. 특히, Gaussian splatting을 통해 포즈 후보를 개선하고, 텍스트 기반 6DoF 로컬라이제이션의 정확도를 크게 향상시킵니다.

- **Technical Details**: 우리는 diffusion 기반 아키텍처를 사용하여 텍스트 설명을 조건으로 일반화된 카메라 포즈 분포를 생성합니다. 여기서 텍스트 피처를 기반으로 하여 노이즈가 있는 6DoF 카메라 포즈를 반복적으로 업데이트합니다. 이러한 과정에서 우리는 미리 훈련된 Vision-Language Model인 CLIP을 활용해 텍스트 설명과 포즈 분포 간의 관계를 설정합니다.

- **Performance Highlights**: 우리의 방법은 다섯 개의 대규모 데이터셋 전반에서 일관되게 높은 성능을 보여줍니다. 특히, 높은 세부 묘사 수준의 텍스트로 테스트할 때, 로컬라이제이션 정확도가 크게 향상되는 것을 확인했습니다. 실험 결과, Pose Diffusion 및 Text2Pose 등 기존의 방법들보다 우수한 성능을 기록하였습니다.



### An analysis of data variation and bias in image-based dermatological datasets for machine learning classification (https://arxiv.org/abs/2501.08962)
Comments:
          10 pages, 1 figure

- **What's New**: 이번 연구는 피부암 진단을 위해 실무 환경에서 사용되는 임상 이미지를 대상으로 한 새로운 데이터 분포 분석을 제안한다. 기존의 dermoscopic 데이터셋으로 훈련된 모델들이 임상 환경에서 효과적으로 일반화되지 못하는 문제에 주목하고, 이러한 챌린지를 해결하기 위한 접근법을 모색한다. 연구는 임상 이미지에서 발생하는 데이터 불균형과 잡음을 해결하기 위해 transfer learning을 고려한다.

- **Technical Details**: 연구는 다양한 deep learning 아키텍처를 활용하여 실제 임상 및 dermoscopic 이미지의 차이를 분석한다. 실험에 사용된 모델 아키텍처에는 ConvNext, DenseNet, ResNet, EfficientNet 등이 포함된다. 또한, 데이터 손실을 줄이기 위해 augmentations 기술을 적용하여 모델의 훈련 과정에서 발생하는 잡음 및 클래스 불균형의 영향을 완화하고자 하였다.

- **Performance Highlights**: 모델의 임상 분류 성능은 fine-tuning 과정을 통해 향상되었으며, 다양한 데이터셋 구성을 실험하여 임상 시나리오에서의 모델 비교 가능성을 평가하였다. 연구 결과는 데이터 배포의 중요성과 이를 통해 예측 정확성을 강화할 수 있는 방법론에 대한 통찰을 제공한다. 최종적으로, 서로 다른 분포 간의 데이터 결합 방안을 모색하여 모델의 정확도 저하를 줄이는 전략을 제시하였다.



### Visual WetlandBirds Dataset: Bird Species Identification and Behavior Recognition in Videos (https://arxiv.org/abs/2501.08931)
- **What's New**: 현재의 생물다양성 손실 위기는 동물 모니터링의 필요성을 더욱 부각시키고 있습니다. 이에 따라, 본 연구는 조류 행동 탐지와 종 분류를 위해 특별히 설계된 첫 번째 정밀 비디오 데이터셋을 소개합니다. 이 데이터셋은 178개의 비디오로 구성되어 있으며, 스페인 알리칸테 지역의 습지에서 촬영되었습니다.

- **Technical Details**: 제안된 데이터셋은 13종의 조류가 7가지의 행동을 수행하는 장면을 포함하고 있습니다. 각 비디오는 행동이 발생하는 위치(바운딩 박스)와 함께 프레임 단위로 주석이 달려 있어 기존의 데이터셋과 차별화됩니다. 데이터 수집과 주석 작업은 전문 생태학자와 협력하여 진행되었습니다.

- **Performance Highlights**: 최신 모델을 사용한 기초 성능 결과도 제공되어 조류 행동 인식과 종 분류 작업의 효과성을 평가합니다. 데이터셋은 다양한 행동 클래스에서 샘플 비디오를 포함하고 있으며, 이는 조류 모니터링과 보존 전략 수립에 중요한 기초 자료를 제공합니다.



### Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Datas (https://arxiv.org/abs/2501.08924)
- **What's New**: 이번 논문은 다양한 카메라 센서와 이미지 개발 작업 흐름에 걸쳐 일반화 가능한 denoising 모델 개발을 지원하기 위해 설계된 Raw Natural Image Noise Dataset (RawNIND)을 소개합니다. 두 가지 denoising 접근 방식을 제안하며, 첫 번째 방법은 원시 Bayer 데이터를 직접 처리하여 계산 효율성을 극대화하고, 두 번째 방법은 선형 RGB 이미지를 사용하여 다른 센서에 대한 일반화를 개선합니다. 이 모든 방법은 전통적인 처리 방식보다 우수한 성능을 보여줍니다.

- **Technical Details**: 이 연구에서 제안한 방법은 원시 센서 데이터에 직접 작용하는 denoising 및 압축 통합 모델을 포함합니다. 또한 RawNIND 데이터셋을 통해 다양한 카메라 센서와 작업 흐름으로 일반화된 denoising 모델을 개발할 수 있는 가능성을 제시합니다. Raw Bayer 이미지 denoising은 한 차원 적게 처리하여 효율성을 높이며, 선형 RGB denoising은 고품질 이미지를 유지하면서 다양한 색상 필터 배열에 대한 일반화를 가능하게 합니다.

- **Performance Highlights**: 제안된 두 가지 denoising 방법 모두 개발된 이미지를 입력으로 사용하는 기존 접근 방식보다 월등한 성능을 발휘하며, 실제 애플리케이션에 유연하게 통합될 수 있습니다. 이 연구는 원시 데이터 수준에서 denoising과 압축을 통합함으로써 비율-왜곡 성능과 계산 효율성을 크게 향상시켜 이미지 처리에서의 새로운 패러다임 전환을 제안합니다.



### Empowering Agricultural Insights: RiceLeafBD - A Novel Dataset and Optimal Model Selection for Rice Leaf Disease Diagnosis through Transfer Learning Techniqu (https://arxiv.org/abs/2501.08912)
- **What's New**: 이 연구는 방글라데시의 농장에서 수집한 새로운 데이터셋인 RiceLeafBD를 도입하여 쌀 잎 질병을 조기에 진단하고 모니터링하는 데 기여함을 강조합니다. 이 데이터셋은 네 가지 유형의 잎 스트레스를 나타내는 1,555장의 이미지를 포함하고 있으며, 실제 필드 조건에서 촬영된 이미지로 구성되어 있습니다. 따라서 농업 분야에서 인공지능(Artificial Intelligence) 기술의 활용을 통해 기초적인 데이터 부족 문제를 해결할 수 있습니다.

- **Technical Details**: 연구는 다양한 딥러닝(DL) 모델을 이행하며, 특히 Light CNN, MobileNet-v2 및 InceptionNet-v2를 적용하여 모델의 성능을 평가했습니다. 이러한 모델은 쌀 잎 질병을 정확하고 신속하게 진단할 수 있는 가능성을 보여주며, EfficientNet-V2는 91.5%의 성능을 기록했습니다. 데이터셋의 객관성을 통해, 이 연구는 수확량 감소에 대한 갈망에 대응하기 위한 실증적인 접근법을 제공합니다.

- **Performance Highlights**: 제안된 데이터셋과 모델은 다른 전통적인 방법들과 비교하여 높은 정확도를 기록하여 그 효과적으로 쌀 잎 질병을 식별할 수 있는 능력을 입증했습니다. 연구 결과는 특히 농업 생산성을 높이고 식량 안보를 강화할 수 있는 방안으로 작용할 수 있습니다. 이에 따라, 방글라데시의 독특한 농업 환경에서 쌀 작물의 질병 조기 탐지가 더욱 중요하다는 점을 강조합니다.



### Lights, Camera, Matching: The Role of Image Illumination in Fair Face Recognition (https://arxiv.org/abs/2501.08910)
Comments:
          14 pages, 11 figures, Conference submission

- **What's New**: 이 연구에서는 인종 간의 얼굴 인식 정확도 차이를 줄이는 것을 목표로 하며, 특히 백인(Caucasian)과 아프리칸 아메리칸(African American) 여성의 이미지 쌍에서 얼굴 밝기(birghtness)가 미치는 영향을 분석합니다. 얼굴 이미지의 밝기를 중앙 픽셀 값(median pixel value)과 픽셀 분포로 해석하여 조정하는 세 가지 실험을 수행했습니다. 이 연구의 결과는 밝기 조정을 통해 정확도 차이를 최대 57.6%까지 감소시키는 데 성공했음을 보여줍니다.

- **Technical Details**: 연구의 주요 실험은 얼굴 영역의 밝기 균형을 목표로 하며, 서로 다른 인종의 얼굴 이미지 쌍에서 유사도 점수(similarity score)의 차이를 분석합니다. 첫 번째 실험에서는 두 이미지 간의 밝기 차이가 유사도 점수에 미치는 영향을 측정하고, 이후 두 번째와 세 번째 실험에서는 밝기 값의 분포를 고려하여 조정합니다. 연구는 MORPH 데이터셋을 활용해 다양한 밝기 패턴을 분석하였으며, ArcFace 손실 함수를 기반한 조합 마진 모델(margin model)을 사용하여 유사도 점수를 계산했습니다.

- **Performance Highlights**: 실험 결과, 백인 여성의 유사도 점수는 최대 5.9%, 아프리칸 아메리칸 여성의 유사도 점수는 3.7% 개선되었습니다. DESIRED 결과는 각 인종의 이미지 쌍에서 d'가 감소하여 두 분포의 유사도가 더 가까워진다는 것이며, 이는 이미지 간 밝기 조정이 정확도 개선에 긍정적인 영향을 미친다는 것을 시사합니다. 최종적으로, 밝기 기반 균형 조정이 두 인종 간의 얼굴 인식의 정확도 격차를 줄이는 데 효과적임을 보여주었습니다.



### Enhanced Multi-Scale Cross-Attention for Person Image Generation (https://arxiv.org/abs/2501.08900)
Comments:
          Accepted to TPAMI, an extended version of a paper published in ECCV2020. arXiv admin note: substantial text overlap with arXiv:2007.09278

- **What's New**: 이번 논문에서는 사람 이미지 생성을 위한 새로운 크로스 어텐션 기반 생성적 적대 신경망(XingGAN 및 XingGAN++)을 제안합니다. 이 방식은 서로 다른 모달리티(appearance와 shape) 간의 상관 행렬을 계산하여 모달리티를 통합하는 방식으로 작동하며, 성능 개선을 위한 상호 작용을 통해 두 개의 생성 분기를 갖습니다. 또한, 두 가지 새로운 multi-scale cross-attention 블록을 제안하여 다양한 스케일 내에서 사람 포즈 간의 장기적 상관관계를 효과적으로 학습합니다.

- **Technical Details**: XingGAN은 shape-guided appearance-based generation과 appearance-guided shape-based generation을 포함하는 두 개의 분기를 갖습니다. 각 분기는 SA(Shape-guided Appearance) 및 AS(Appearance-guided Shape) 블록의 순서로 구성되어, 각각 appearance와 shape 표현을 점진적으로 업데이트합니다. 또한, Enhanced Attention (EA) 모듈을 통해 상관관계의 노이즈와 모호함을 줄여 생성 성능을 향상시키고, Densely Connected Co-attention Fusion 모듈을 통해 다양한 단계에서 appearance와 shape 특성을 효과적으로 융합합니다.

- **Performance Highlights**: 두 개의 공개 데이터셋(Market-1501 및 DeepFashion)에서 수행된 실험 결과, 제안된 방법이 기존 GAN 기반 방법을 초과하고 확산 기반 방법과 유사한 성능을 기록했습니다. 하지만 XingGAN은 훈련 및 추론 과정에서 확산 기반 방법보다 상당히 빠른 속도를 보입니다. 특히, 훈련 과정에서 PIDM보다 69.33배, 추론 과정에서 18.04배 빠른 성능을 발휘하였습니다.



### Feature-based One-For-All: A Universal Framework for Heterogeneous Knowledge Distillation (https://arxiv.org/abs/2501.08885)
- **What's New**: 이번 연구에서 제안한 FOFA(Knowledge Distillation) 프레임워크는 다양한 아키텍처 간의 기능 증류(feature distillation)을 가능하게 합니다. 기존의 지식 증류 방법들은 주로 동질적인 아키텍처에 초점을 맞췄지만, FOFA는 이를 개선하여 이질적인 아키텍처에서도 효과적으로 작동합니다. FOFA는 두 가지 주요 모듈, 즉 Adaptive Feedback Prompt (AFP)와 Region-Aware Attention (RAA)을 통해 다양한 모델 간 지식 전이를 실현하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: FOFA 프레임워크는 지역 인식 주의(region-aware attention)를 활용하여 서로 다른 아키텍처 간의 관점을 일치시키는 데 중점을 둡니다. AFP 모듈은 학생 모델의 학습 과정에 맞춰 교사 모델이 동적으로 기능을 조정할 수 있도록 돕습니다. 이러한 구성 요소들은 중간 레이어의 기능을 효과적으로 증류할 수 있게 하며, 다양한 아키텍처에서의 중간 기능을 충분히 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: FOFA 프레임워크는 CIFAR-100에서 최대 16.94%, ImageNet-1K에서 3.35%, COCO에서는 3.50%의 성능 향상을 보여주며, 학생 모델의 전반적인 성능을 크게 개선합니다. 이를 통해 FOFA는 학습 경과에 따라 동적으로 조정되는 교사 모델의 가능성을 제시하며, 이질적인 아키텍처에서의 지식 전이가 더욱 효과적임을 입증하였습니다.



### Generative Planning with 3D-vision Language Pre-training for End-to-End Autonomous Driving (https://arxiv.org/abs/2501.08861)
- **What's New**: 자율주행 시스템을 위한 새로운 GPVL(Generative Planning with 3D-vision Language Pre-training) 방법론이 제안되었습니다. 이 방법론은 3D-비전 언어 사전 훈련 모듈을 통해 시각적 인식과 언어적 이해의 간극을 메우고, 크로스 모달 언어 모델을 통해 종합적인 주행 결정 및 세부 경로 생성 기능을 제공합니다. 실험 결과, GPVL은 최신 연구와 비교하여 우수한 성능을 보여주며, 다양한 시나리오에서도 강력한 일반화 능력과 실시간 가능성을 지니고 있습니다.

- **Technical Details**: GPVL은 3D-비전 언어 사전 훈련 모듈을 통해 비전과 언어 특성 간의 그룹 간 상관 관계를 확립합니다. 이는 BEV(새의 눈 전망) 특징 맵을 추출하여 주행 환경에 대한 포괄적인 이해를 가능하게 하며, 최종적으로 2D 장면 캡셔닝 모델을 기반으로 하여 전체적인 주행 결정을 생성합니다. 이 방법론은 텍스트 기반의 추론 능력을 갖추고 있으며, 시각 인식과 네비게이션 정보에 기반한 경로 생성을 지원합니다.

- **Performance Highlights**: GPVL은 nuScenes 데이터셋에서 수행된 실험을 통해 탁월한 성능을 발휘하며, 기존의 방법들과 비교하여 해석 가능성과 안전성을 크게 향상시킵니다. 이 시스템은 개별 모듈의 누적 오류를 줄이고, 자율주행 시스템의 전반적인 안전성을 높이는 방향으로 설계되었습니다. 특히, 3D 공간 관계를 효과적으로 이해하고, 복잡한 주행 환경에서도 합리적인 분석 및 맥락에 맞는 경로 생성을 가능하게 하는 점이 강조됩니다.



### MANTA: Diffusion Mamba for Efficient and Effective Stochastic Long-Term Dense Anticipation (https://arxiv.org/abs/2501.08837)
- **What's New**: 이번 연구에서는 확률적(long-term) 밀집 예측(stochastic long-term dense anticipation)의 문제를 다루고 있습니다. 이 작업의 목표는 제공된 비디오 관찰을 바탕으로 몇 분간의 미래에서 동작과 그 지속 시간을 예측하는 것입니다. 제안된 MANTA(MAmba for ANTicipation) 네트워크를 통해 긴 시퀀스에 대한 효율적이고 효과적인 시간적 모델링이 가능해지며, 기존의 방법보다 계산 및 메모리 효율성을 크게 향상시킵니다.

- **Technical Details**: 우리는 MANTA 네트워크를 통해 긴 시퀀스에 대해 글로벌 수용 영역(global receptive field)을 유지하면서도 선형 복잡성을 유지합니다. 이를 통해 관찰된 정보가 전체 마스크 처리된 입력 시퀀스에 조기에 전파될 수 있도록 합니다. MANTA 모델은 과거 관찰된 프레임과 미래 마스킹된 프레임을 처리하기 위한 내재적인 데이터 의존적 게이팅 메커니즘을 활용하고 있습니다.

- **Performance Highlights**: 제안된 모델은 Breakfast, 50Salads, Assembly101의 세 가지 데이터세트에서 최첨단 성능을 보여줍니다. 특히, 이전의 최고 성능 모델인 GTDA에 비해 추론 및 훈련 시간에서 각각 최대 65.3배 및 6.6배의 속도 향상을 달성했습니다. 따라서 우리의 접근 방식은 효율성과 성능 두 가지 측면에서 모두 우수한 결과를 보여줍니다.



### IDEA: Image Description Enhanced CLIP-Adapter (https://arxiv.org/abs/2501.08816)
- **What's New**: 이번 논문은 CLIP (Contrastive Language-Image Pre-training)을 기반으로 한 Image Description Enhanced CLIP-Adapter (IDEA) 방법을 제안합니다. 이 방법은 이미지와 텍스트의 상호작용을 활용하여 정밀한 특성을 포착함으로써, 적은 샘플로 이미지 분류 작업을 수행할 수 있도록 돕습니다. 또한, Trainable-IDEA (T-IDEA)를 도입하여 경량의 학습 가능한 컴포넌트를 추가하고, 11개의 데이터셋에서 SOTA (State-Of-The-Art) 성능을 달성했습니다.

- **Technical Details**: IDEA는 훈련이 필요 없는 방법으로, 이미지-텍스트 쌍의 보완적인 관계를 활용하여 다중 모달리티(multi-modality) 간의 의미적 연관성을 탐구합니다. T-IDEA는 경량 프로젝터와 학습 가능한 잠재 공간(learnable latent space)을 통합하여 IDEA의 성능을 더욱 향상시킵니다. 이러한 방식은 기존의 방법과는 다르게 강력한 성능을 발휘하며, 11개의 공개 이미지 데이터셋에서 실험적으로 검증되었습니다.

- **Performance Highlights**: IDEA와 T-IDEA는 훈련이 필요 없는 설정과 훈련이 요구되는 설정 모두에서 기존의 SOTA 방법들을 초월하는 성능을 보여주었습니다. 새로운 데이터셋인 'IMD-11'을 생성하여 총 1,637,795개의 이미지-텍스트 쌍을 제공하며, 이는 연구에 중요한 기여를 하게 됩니다. 이러한 성과는 제한된 학습 데이터로도 뛰어난 성능을 달성할 수 있음을 시사합니다.



### Human Pose-Constrained UV Map Estimation (https://arxiv.org/abs/2501.08815)
- **What's New**: 이번 논문에서는 UV map estimation에 있어 새로운 접근법, Pose-Constrained Continuous Surface Embeddings (PC-CSE)를 제안합니다. 이 기법은 기존의 방법들이 개별적으로 픽셀(descriptor) 간의 비교만으로 이루어졌음을 지적하며, 전체적인 일관성을 유지하면서도 세부적인 정밀도를 보장하는 방식입니다. 특히, 인간의 2D 포즈를 픽셀-정점 할당(assignment) 과정에 통합함으로써 구조적 제약을 강화했습니다.

- **Technical Details**: PC-CSE는 2D 인간 포즈를 통해 제공된 해부학적 제약(anatomical constraints)을 사용하여 UV 맵의 일관성을 높입니다. 기존 방법들과 달리, 이 접근법은 특히 전체 신체 포즈(whole-body poses)에서 손과 발에 대한 추가적인 세부 정보를 포함하여 더 나은 제약을 제공합니다. 이러한 조건을 바탕으로 UV 맵이 보다 유효하며 해부학적으로 타당성을 높여줍니다.

- **Performance Highlights**: DensePose COCO 데이터셋에서의 평가 결과, 제안된 방법은 선택된 2D 인간 포즈 모델에 관계없이 일관된 성능 향상을 보여줍니다. UV 맵을 인간 포즈로 조건화하여 잘못된 매핑(invalid mappings)을 줄이고 해부학적 타당성을 향상시키는 데 기여하고 있습니다. 또한, 논문에서는 ground-truth annotations의 불일치를 강조하고 있습니다.



### Multi-visual modality micro drone-based structural damage detection (https://arxiv.org/abs/2501.08807)
- **What's New**: 이번 연구에서는 구조물 손상 탐지를 위한 강력한 프레임워크인 DetectorX를 제안합니다. DetectorX는 마이크로 드론과 결합되어 손상 탐지의 정확한 검출 및 회복력 확보를 목표로 합니다. 이 프레임워크는 두 가지 혁신적인 모듈인 stem block과 spiral pooling 기법을 도입하여 객체 탐지기의 강인성을 보장합니다.

- **Technical Details**: DetectorX는 두 개의 Deep Convolutional Neural Network (DCNN) 모델의 출력을 활용하여 동적인 시각 모달리티를 생성하는 stem block을 포함합니다. 또한, event-based reward reinforcement learning 방법을 활용하여 parent와 child DCNN 모델의 행동을 제약하여 보상 결과를 도출합니다. spiral pooling 기법은 온라인 이미지 증강 방법으로, 나선형과 평균/최대 풀링된 특징을 결합하여 특징 표현을 강화합니다.

- **Performance Highlights**: 세 가지의 광범위한 실험을 통해, DetectorX는 여러 메트릭에서 경쟁하는 탐지기들보다 우수한 성능을 보였습니다. 특히, precision(0.88), recall(0.84), average precision(0.91), mean average precision(0.76), mean average recall(0.73)의 결과를 제공합니다. 이 연구의 결과는 DetectorX가 도전적인 환경에서도 만족스러운 결과를 제공하고 회복력이 뛰어난 것으로 나타났음을 보여줍니다.



### Exploring ChatGPT for Face Presentation Attack Detection in Zero and Few-Shot in-Context Learning (https://arxiv.org/abs/2501.08799)
Comments:
          Accepted in WACV workshop 2025

- **What's New**: 이번 연구는 ChatGPT의 최신 버전인 GPT-4o가 Face Presentation Attack Detection (PAD) 분야에서 상업 솔루션들을 포함하여 여러 모델들보다 우수한 성능을 발휘할 수 있는 가능성을 강조합니다. 특히 몇 가지 특정 시나리오에서 높은 일관성을 보이며, 추가 제공되는 예제에 따라 성능이 향상되는 few-shot in-context learning(컨텍스트 내 소수 샷 학습)이 두드러졌습니다. 또한, 상세한 프롬프트를 사용하면 모델이 신뢰할 수 있는 점수를 제공하는 것으로 나타났습니다.

- **Technical Details**: 연구는 SOTERIA 데이터 세트의 하위 집합을 사용하여 진행되었으며, 데이터 프라이버시 규정을 준수하기 위해 동의한 개인의 데이터만 사용했습니다. GPT-4o는 공격 유형(인쇄 또는 재생)을 분류하도록 명시적으로 지시받지 않았음에도 불구하고, 높은 정확도로 공격 유형을 올바르게 예측하는 emergent reasoning(출현적 추론) 능력을 발휘했습니다. 반면, zero-shot(제로 샷) 작업에서는 성능이 제한되며, 전문적인 PAD 시스템에 비해 다소 약점이 있습니다.

- **Performance Highlights**: GPT-4o는 특히 few-shot 환경에서 뛰어난 인식 능력을 보여주었으며, 모델의 해석 가능성을 개선하는 설명 요청 프롬프트가 성능을 약간 향상시켰습니다. 이러한 연구 결과는 GPT-4o가 PAD 애플리케이션에서 유망하다는 것을 보여주며, 데이터 프라이버시 문제를 해결하고 다양한 데이터 세트에서의 일반화를 향상시킬 수 있는 미래 연구의 기초를 다짐하였습니다.



### Admitting Ignorance Helps the Video Question Answering Models to Answer (https://arxiv.org/abs/2501.08771)
- **What's New**: 이번 논문은 비디오 질문 응답(VideoQA) 시스템에서 발생할 수 있는 스푸리어스 상관관계(spurious correlation)를 해결하기 위한 혁신적인 학습 프레임워크를 제안합니다. 기존의 많은 모델들이 질문과 답변 간의 상관관계를 단순히 최대화하려는 경향이 있었으나, 이는 종종 비효율적인 패턴에 의존하게 만듭니다. 본 연구는 모델이 개입된 질문에 대한 무지를 인정하도록 강요하여, 보다 정교한 비디오-질문 정렬(video-question alignment)과 멀티모달 표현(multimodal representation)을 학습하도록 유도합니다.

- **Technical Details**: 제안하는 프레임워크에서는 질문에 대해 두 가지 유형의 개입(interventions), 즉 displacement(질문 교체)와 perturbation(단어 변경)을 수행합니다. 이러한 기법들은 모델이 질문과 비디오 간의 쉽게 파악할 수 있는 글로벌 상관관계(global correspondence)와 더 어려운 로컬 상관관계를 모두 학습할 수 있게 도와주며, 커리큘럼 학습(curriculum learning) 전략을 도입하여 무지를 줄이는 방향으로 시스템을 점진적으로 훈련합니다. 이를 통해 모델은 개입된 질문-비디오 쌍에도 올바른 답변을 제공할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 실험 결과는 제안된 프레임워크가 비디오QA 모델의 성능을 획기적으로 향상시킬 수 있음을 보여주며, 모델의 구조적 변경을 최소화하면서도 우수한 결과를 달성합니다. 이 연구는 기존의 다양한 비디오QA 모델에 대해 적용 가능하므로, 보다 넓은 범위에서의 일반화가 가능하다는 장점이 있습니다. 전반적으로, 본 논문은 비디오-질문 정렬을 개선하고 멀티모달 표현의 강건성을 높이는 데 핵심적인 기여를 하고 있습니다.



### Few-Shot Learner Generalizes Across AI-Generated Image Detection (https://arxiv.org/abs/2501.08763)
Comments:
          11 pages, 5 figures

- **What's New**: 최근의 AI 생성 이미지 탐지기는 기존의 제한적인 생성 모델에서는 효율적으로 작동하지만, 한정되지 않은 새로운 모델에서는 성능 저하를 겪습니다. 본 연구에서는 Few-Shot Detector (FSD)라는 혁신적인 탐지기를 제안하여 매우 소수의 샘플만으로도 새로운 가짜 이미지를 효과적으로 구분할 수 있습니다. 실험을 통해 FSD는 GenImage 데이터셋에서 평균 정확도(ACC)를 7.4% 향상시켜 최첨단 성능을 기록했습니다.

- **Technical Details**: FSD는 몇 샘플을 받아들이는 프로토타입 네트워크(prototypical network)를 활용하여 새로운 데이터에 대한 메트릭 공간(metric space)을 학습합니다. 기존의 엔드 투 엔드 분류 방식과는 달리, FSD는 여러 클래스의 몇 가지 샘플을 받아 해당 프로토타입 표현을 계산하며, 이 정보를 바탕으로 최근접 이웃 방법(nearest-neighbor method)으로 테스트 이미지를 분류합니다. 이러한 방식을 통해 교육 데이터가 부족한 경우에도 오버피팅을 피하며 분류를 수행할 수 있습니다.

- **Performance Highlights**: FSD는 기존 최첨단 방법보다 눈에 띄는 개선을 보여주었으며, 단 10개의 샘플로도 탐지 성능을 크게 향상시킬 수 있음을 확인했습니다. 더불어, 생성 모델의 빠른 발전에 발맞춰 AI 생성 이미지를 소스에 따라 구분할 수 있도록 하여, 현실 세계의 적용 가능성을 높였습니다. 따라서 FSD는 새로운 생성 모델에 대해 추가 학습 없이도 탁월한 일반화 능력을 발휘하여 가짜 이미지 탐지의 새로운 기준을 마련합니다.



### Self-supervised Transformation Learning for Equivariant Representations (https://arxiv.org/abs/2501.08712)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 자기지도 변환 학습(Self-supervised Transformation Learning, STL) 방법을 제안합니다. 기존의 변환 레이블을 이미지 쌍에서 유도된 변환 표현으로 대체함으로써, 변환의 내성을 높이고 성능을 개선하는 것을 목표로 합니다. STL은 이전 방법들과 같은 배치 복잡성을 유지하면서도 변환 표현을 학습할 수 있는 방식으로, 다양한 분류와 탐지 작업에서 효과를 입증했습니다.

- **Technical Details**: STL에서 사용하는 변환 표현은 이미지에 적용된 변환에 불변성을 가지고 있습니다. 이 표현을 학습함으로써, STL은 정확하고 복잡한 변환 정보를 캡처할 수 있습니다. 추가로 AugMix와 같은 복잡한 변환을 통합하여 과거의 동등한 방법에서 불가능했던 성능 향상을 달성했습니다. 이러한 방식은 다양한 기초 모델과도 호환 가능하여 폭넓은 적용성을 지니고 있습니다.

- **Performance Highlights**: STL은 여러 데이터셋과 작업에서 매우 경쟁력 있는 성능을 발휘했습니다. 11개의 벤치마크 중 7개에서 기존 방법들을 초과하는 성과를 보여주었으며, 탐지 작업에서 특히 두드러진 성과를 올렸습니다. 변환 표현의 상호 의존성을 효과적으로 캡처함으로써, STL은 기존의 변환 학습 방법들보다 우수한 변환 예측 성능을 자랑합니다.



### RealVVT: Towards Photorealistic Video Virtual Try-on via Spatio-Temporal Consistency (https://arxiv.org/abs/2501.08682)
Comments:
          10 pages (8 pages main text, 2 pages references), 5 figures in the main text, and 4 pages supplementary materials with 3 additional figures

- **What's New**: 이 논문에서는 RealVVT라는 새로운 Virtual Try-on 프레임워크를 소개합니다. 기존의 비디오 기반 Virtual Try-on 방법론이 직면한 문제를 해결하기 위해, 의복의 일관성과 사실감을 보장하는 혁신적인 접근 방식을 채택했습니다.

- **Technical Details**: RealVVT 프레임워크는 Clothing & Temporal Consistency 전략과 Agnostic-guided Attention Focus Loss 메커니즘을 포함하여 공간적 일관성을 확보합니다. Pose-guided Long Video VTO 기법을 통해 긴 비디오 환경에서도 의복의 특성을 효과적으로 유지할 수 있습니다.

- **Performance Highlights**: 다양한 데이터셋을 통한 실험에서 RealVVT는 단일 이미지 및 비디오 Virtual Try-on 작업 모두에서 기존의 최첨단 모델들을 초월하는 성능을 보여주었습니다. 이 접근 방식은 패션 전자 상거래 및 가상 피팅 환경 내에서 실용적인 응용 가능성을 제시합니다.



### FlexiClip: Locality-Preserving Free-Form Character Animation (https://arxiv.org/abs/2501.08676)
Comments:
          13 pages, 4 figures, 7 tables

- **What's New**: 이 논문에서는 FlexiClip이라는 새로운 접근법을 소개하여 정적 클립 아트 이미지를 애니메이션화 하는 문제를 해결합니다. FlexiClip은 기존의 방법들이 직면한 시간적 일관성과 기하학적 정합성 문제를 동시에 해결하고자 합니다. 이 방법은 전통적인 Bézier 곡선 기반의 궤적 모델링을 확장하여, 시간적 Jacobian을 사용해 점진적으로 동작 역학을 수정합니다.

- **Technical Details**: FlexiClip은 pfODE (Probability Flow ODE)를 사용하여 시간적 교정을 연속적인 시간 함수로 모델링합니다. 이 접근법은 이전의 이산적인 최적화 방법보다 더 효과적으로 시간적 노이즈 문제를 해결합니다. 또한, GFlowNet (Generative Flow Network)의 원칙에서 영감을 받아 흐름 매칭 손실(flow matching loss)을 도입하여 매끄러운 애니메이션 전환을 최적화합니다.

- **Performance Highlights**: 광범위한 실험과 절단 연구(ablations) 결과, FlexiClip은 다양한 클립 아트 유형(인물 및 동물)을 포함한 애니메이션 생성에서 부드럽고 자연스럽고 시간적으로 일관된 결과를 산출하는 능력을 입증했습니다. 이 논문은 복잡한 비틀림이나 비강체 변형을 포함하는 애니메이션을 처리할 수 있는 FlexiClip의 성능을 강조하고 있습니다.



### A Survey on Facial Image Privacy Preservation in Cloud-Based Services (https://arxiv.org/abs/2501.08665)
- **What's New**: 본 논문에서는 클라우드 기반 서비스에서 얼굴 이미지의 개인 정보를 보호하기 위한 현재 방법에 대한 종합적인 리뷰를 제공합니다. 이러한 방법들은 주로 이미지 난독화 기반 보호(image obfuscation-based protection) 및 적대적 섭동 기반 보호(adversarial perturbation-based protection)로 나뉘어지며, 각 카테고리의 효과성을 질적 및 양적으로 비교합니다. 또한, 해결되지 않은 도전 과제를 강조하고, 클라우드 컴퓨팅 환경에서 개인 정보 보호를 개선하기 위한 미래 연구 방향을 제안합니다.

- **Technical Details**: 얼굴 인식은 고유한 얼굴 특징을 분석하여 개인을 식별하는 생체 인식 기술입니다. 얼굴 인식 모델들은 CNN(convolutional neural network) 및 FaceNet 기반 모델과 같은 여러 유형으로 나뉘어지며, 각각의 사용에 있어 속도와 정확성, 강건성 측면에서 장점을 제공합니다. 클라우드 기반 서비스에서는 대규모 사용자 데이터를 처리하기 위해 다수의 알고리즘을 통합하여 정확성을 최적화하며, 이로 인해 개인 정보 보호 문제에 대한 연구가 활발히 진행되고 있습니다.

- **Performance Highlights**: 개인 정보 보호를 위한 최신 기술은 얼굴 이미지의 인식 가능성을 줄이고 유틸리티를 유지하는 것을 목표로 하고 있습니다. AMT-GAN과 Gender-AN 같은 기술들은 개인의 얼굴을 변형하여 인식 저항력을 높이며, DeepBlur와 Diff-Privacy는 시각적 품질을 유지하면서도 인식이 불가능하도록 이미지를 조작합니다. 하드웨어 기반 접근법인 광학 인코더를 사용하는 방법도 제안되었으며, 이는 소프트웨어 기반 솔루션이 간과할 수 있는 보안 취약점에 강력한 방어를 제공합니다.



### BRIGHT-VO: Brightness-Guided Hybrid Transformer for Visual Odometry with Multi-modality Refinement Modu (https://arxiv.org/abs/2501.08659)
Comments:
          9 pages, 7 figures

- **What's New**: BrightVO는 Transformer 아키텍처를 기반으로 하는 새로운 Visual Odometry (VO) 모델로, 저조도 환경에서의 카메라 위치 추정 문제를 해결하는 데 초점을 맞추고 있습니다. 이 모델은 이미지의 시각적 특징을 추출하는 프런트 엔드 기능과 관성 측정 장치(Inertial Measurement Unit, IMU) 데이터를 통합하는 멀티 모달리티 정제 모듈을 후면에 포함하여 정확성과 강건성을 높입니다. 또한, 새로운 저조도 데이터셋 KiC4R을 생성하여 다양한 조명 조건에서 VO 프레임워크의 훈련과 평가를 지원합니다.

- **Technical Details**: BrightVO는 최첨단 딥러닝 접근 방식인 Transformer 아키텍처를 활용하여 긴 시퀀스에서도 복잡한 시각적 특징을 모델링할 수 있습니다. 이 모델은 자가 주의(self-attention) 메커니즘을 통해 저조도 조건에서도 가장 관련성 높은 특징에 초점을 맞춰 pose estimation을 수행합니다. 더불어, IMU 데이터를 통합하는 그래프 최적화를 기반으로 하는 후면 정제 블록을 통해 위치 추정의 오류를 줄이고 정확도를 향상시킵니다.

- **Performance Highlights**: BrightVO는 KiC4R 데이터셋과 KITTI 벤치마크에서 기존의 방법들보다 20% 향상된 위치 추정 정확도를 보여 주며, 저조도 조건에서는 무려 259%의 성능 향상을 기록했습니다. 이러한 결과는 BrightVO가 저조도 환경에서의 로컬라이제이션 성능을 크게 개선했음을 보여줍니다. 이러한 연구는 연구 목적으로 전혀 무료로 공개되어 있습니다.



### StereoGen: High-quality Stereo Image Generation from a Single Imag (https://arxiv.org/abs/2501.08654)
- **What's New**: 본 논문에서는 StereoGen이라는 새로운 파이프라인을 제안하여 고품질 스테레오 이미지 생성을 가능하게 합니다. 기존의 방법들과는 달리, 이 파이프라인은 모노큘러 깊이 추정 모델이 생성한 가상의 disparity를 이용하여 단일 이미지를 좌측 이미지로 사용하고 우측 이미지를 합성합니다. 또한, Diffusion Inpainting 모델을 미세 조정하여 합성된 우측 이미지 내의 가려진 영역을 복구함으로써 더 나은 세부 사항과 손상되지 않은 의미 구조를 제공합니다.

- **Technical Details**: StereoGen 파이프라인은 단일 이미지를 좌측 이미지로 사용하고, 이에 대한 정규화된 역 깊이를 추정하여 adaptive disparity selection(ADS) 모듈을 통해 corresponding disparity를 생성합니다. 이후, forward warping 기법을 사용하여 전반적인 이미지 합성과 마스크를 추출하며, 복구해야 할 가려진 영역에 대한 인페인팅 마스크를 설정합니다. 이 과정을 통해 합성된 이미지의 품질을 크게 향상시키고 모델의 훈련을 더욱 견고하게 만듭니다.

- **Performance Highlights**: 실험 결과, 제안된 파이프라인 하에서 훈련된 모델들은 모든 공개된 방법들 중에서 최첨단의 제로샷(zero-shot) 일반화 성능을 달성합니다. Training-Free Confidence Generation과 Adaptive Disparity Selection을 통해 가상의 disparity 및 이미지 품질의 신뢰도를 높이는 등 성능 개선을 이룩했습니다. 최종적으로, Scene Flow와 유사한 양의 합성 데이터만으로도 뛰어난 일반화 성능을 보장합니다.



### Joint Learning of Depth and Appearance for Portrait Image Animation (https://arxiv.org/abs/2501.08649)
- **What's New**: 본 연구는 2D 초상화 애니메이션의 새로운 접근 방식을 소개합니다. 기존의 이미지 생성 방법들이 RGB 이미지를 생성하는 데 집중했던 반면, 우리의 방법은 깊이(depth) 정보도 함께 학습하여 더욱 풍부한 시각적 출력을 가능하게 합니다. 우리는 시각적 외관과 깊이를 동시에 학습할 수 있는 새로운 확산 기반 초상화 이미지 생성기를 개발하였습니다. 이 프레임워크는 얼굴 깊이 추정, 이미지 리라이트 및 오디오 기반의 애니메이션 생성과 같은 다양한 다운스트림 애플리케이션에 효율적으로 적응할 수 있습니다.

- **Technical Details**: 우리는 Stable Diffusion 아키텍처를 기반으로 한 새로운 초상화 생성기를 소개합니다. 이 모델은 RGB 및 깊이(latent images)를 분리하여 추가 잡음을 제거하는 6채널 입력 이미지를 처리하며, 함께 사용되는 참조 네트워크(reference network)는 RGB 참조 이미지를 통해 이미지 확산 과정을 안내합니다. 이를 통해 생성된 깊이 맵은 생성된 얼굴 이미지와 잘 일치하며, 두 개의 채널 간의 강한 상관관계를 보장합니다. 학습 데이터는 스튜디오에서 캡처한 얼굴 이미지와 3D 기하학적 정보를 포함하여, 야외 환경에서도 잘 작동하는 모델로 일반화될 수 있도록 합니다.

- **Performance Highlights**: 훈련이 끝난 후, 본 모델은 RGB 및 깊이 이미지를 커플링한 상태로 샘플링할 수 있으며, 주어진 이미지를 기반으로 깊이 채널을 인페인팅(inpainting)하거나 반대로 깊이 이미지를 사용하여 RGB 채널을 인페인팅하는 등의 작업을 수행할 수 있습니다. 또한, 깊이 정보로부터 얼굴 이미지의 조명을 변경하는 후처리도 가능하며, 마지막으로 오디오 입력에 기반한 talking head 애니메이션 생성이 가능합니다. 이러한 다양한 기능들은 초상화 조작 분야에서 새로운 가능성을 열어줍니다.



### MonSter: Marry Monodepth to Stereo Unleashes Power (https://arxiv.org/abs/2501.08643)
- **What's New**: 이번 논문에서는 stereo matching을 개선하기 위한 새로운 접근법인 MonSter를 제안합니다. MonSter는 monocular depth estimation과 stereo matching의 강점을 결합하여 서로를 지속적으로 개선하는 이중 브랜치 아키텍처를 채택합니다. 이 방법은 특히 occlusions나 textureless 영역과 같은 ill-posed 지역에서 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: MonSter는 두 개의 주요 브랜치로 구성됩니다: monocular depth branch와 stereo matching branch입니다. 이들 브랜치는 초기 monocular depth와 stereo disparity를 추정한 후, 상호 개선 모듈을 통해 서로를 반복적으로 개선합니다. Stereo Guided Alignment(SGA)와 Mono Guided Refinement(MGR) 모듈을 통해 두 브랜치의 노이즈를 줄이고, 신뢰할 수 있는 stereo cue를 선택하여 monocular disparity의 scale과 shift를 보정합니다.

- **Performance Highlights**: MonSter는 KITTI 2012, KITTI 2015, Scene Flow, Middlebury, ETH3D 등 5개의 주요 벤치마크에서 1위를 차지했습니다. 기존 방법에 비해 깊이 인식 성능을 최대 49.5% 향상시킨 결과를 보여주며, 제로샷(Zero-shot) 일반화에서도 탁월한 성능을 기록했습니다. 이러한 성능은 MonSter가 synthetic 데이터로만 훈련되어도 다양한 실제 데이터셋에서 강력한 성능을 보임을 증명합니다.



### Detecting Wildfire Flame and Smoke through Edge Computing using Transfer Learning Enhanced Deep Learning Models (https://arxiv.org/abs/2501.08639)
Comments:
          11 pages, 7 figures

- **What's New**: 본 연구에서는 경량화된 자율 비행체(UAV)와 엣지 컴퓨팅(edge computing) 기술을 통합하여 실시간 데이터 처리를 가능하게 하는 점을 강조하고 있습니다. 특히, 제한된 데이터셋을 기반으로 한 산불 감지(object detection) 성능 향상에서 Transfer Learning(TL)의 중요성을 연구하였습니다. 이 연구는 YOLO(You Only Look Once) 모델의 TL 적용이 어떤 영향을 미치는지를 탐구합니다.

- **Technical Details**: 본 연구에서는 AFSE(Aerial Fire and Smoke Essential) 데이터셋을 타겟으로 하고, FASDD(Flame and Smoke Detection Dataset) 및 COCO(Microsoft Common Objects in Context) 데이터셋을 출처 데이터셋으로 사용합니다. 두 단계로 구성된 TL 방법을 통해 D-Fire 또는 FASDD를 초기 대상 데이터셋으로 사용하고, AFSE를 후속 단계로 설정하여 성능을 평가했습니다. 최적화 후 TL은 탐지 정밀도를 높이고 훈련 시간을 줄였으나, 카스케이드(cascaded) TL로는 뚜렷한 성과를 보고하지 못했습니다.

- **Performance Highlights**: TL을 적용한 YOLOv5n 모델은 평균 79.2%의 mean Average Precision (mAP@0.5)에 도달하며, 모델의 일반화 능력이 향상되었습니다. 또한 TL이 없는 상태에서도 YOLOv5n는 하드웨어 가속 없이도 새로운 YOLO11n보다 거의 두 배 빠른 속도로 이미지를 처리하는 결과를 보였습니다. 전체적으로, TL이 물체 탐지의 정확성을 증가시키는 데 기여한다는 결과를 확인하였으나, 엣지 컴퓨팅 성능 개선을 위한 추가적인 개발이 필요함을 시사합니다.



### Computerized Assessment of Motor Imitation for Distinguishing Autism in Video (CAMI-2DNet) (https://arxiv.org/abs/2501.08609)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: CAMI-2DNet는 비디오 데이터에서 모터 모방(motor imitation)을 평가하는 새로운 딥러닝 기반 접근법으로, 데이터 정규화(normalization), 정리(cleaning), 주석(annotation) 작업들을 제거하여 유연성과 해석 가능성을 높입니다. 이 시스템은 인코더-디코더(encoder-decoder) 아키텍처를 사용하여 비디오를 모션 인코딩(motion encoding)으로 매핑하며, 이는 신체 형태나 카메라 시점 등의 불필요한 요소들과 분리된 특징을 가집니다. CAMI-2DNet은 기존의 HOC 방식의 한계와 수동적인 CAMI-2D 및 CAMI-3D 접근법의 문제들을 해결하여 빠르고 신뢰할 수 있는 모방 평가 도구로서의 가능성을 보여줍니다.

- **Technical Details**: CAMI-2DNet는 대규모 합성 데이터(synthetic data)를 사용하여 가상 캐릭터의 모션을 리타겟팅(motion retargeting)하여 생성한 다양한 포즈와 뷰의 조합을 학습합니다. 그런 다음, 이러한 데이터와 실제 ASD 및 신경전형적인(NT) 참가자들로부터 수집한 데이터로부터 고립된 모션 특징을 학습하여, 무작위적인 요인들에 영향을 받지 않는 분리(disentangled)된 모션 표현을 효과적으로 인코딩합니다. 이 방식은 단순히 주관적인 평가가 아닌, 정량적인 유사성 점수(similarity score) 산출을 통해 모방 능력을 객관적으로 평가하는 데에 중점을 두고 있습니다.

- **Performance Highlights**: CAMI-2DNet는 기존의 HOC 점수와 강한 상관관계를 보이면서도 ASC 아동과 NT 아동을 구별하는 데에서 CAMI-2D보다 뛰어난 성능을 발휘합니다. 또한 CAMI-3D와 비교할 때, 비디오 데이터로 직접 작동하며 필요하지 않은 정규화 및 주석 작업이 없기 때문에 보다 높은 실용성을 제공합니다. 이는 가능한 한 빠르고 상세한 평가를 가능하게 하여 조기 및 정확한 진단을 촉진할 수 있는 잠재력을 가집니다.



### PACF: Prototype Augmented Compact Features for Improving Domain Adaptive Object Detection (https://arxiv.org/abs/2501.08605)
- **What's New**: 이번 연구는 객체 탐지(Object Detection) 분야에서 도메인 차이(domain gap)로 인한 성능 저하 문제를 해결하기 위해 Prototype Augmented Compact Features (PACF) 프레임워크를 제안합니다. 이 프레임워크는 클래스 내 피처의 분포를 정규화하고, 타겟 RoI 피처의 분포를 보정하기 위한 프로토타입 크로스 엔트로피 손실(prototype cross entropy loss)을 도출합니다. 또한, 서로 다른 분류기가 상호 학습할 수 있도록 하는 상호 정규화 전략(mutual regularization strategy)을 도입하여 모델의 성능을 향상시킵니다.

- **Technical Details**: PACF 프레임워크는 클래스 별 도메인 특화 프로토타입(domain-specific class prototypes)을 도입하여, 소스와 타겟 프로토타입과의 유사성을 기반으로 하는 ‘cosine-softmax’ 연산을 통해 특징의 분포를 정규화합니다. 타겟 특성이 동일 클래스의 소스 및 타겟 프로토타입에 가깝고, 다른 클래스의 프로토타입에서 멀어지도록 하는 것이 목표입니다. 이를 통해 클래스 조건부 분포(class-conditional distribution)의 평균 이동(mean shift)을 줄이고, 더 컴팩트한 피처를 추출할 수 있습니다.

- **Performance Highlights**: 실험 결과, PACF 프레임워크는 다양한 적응 설정에서 최신 기술(state-of-the-art) 대비 크게 개선된 성능을 보여주었습니다. 예를 들어, Cityscapes→Foggy Cityscapes 변환에서 성능이 50.3%에서 52.3%로 향상되었습니다. 이러한 성능 향상은 제안된 접근 방식의 범용성과 효과성을 입증합니다.



### Watermarking in Diffusion Model: Gaussian Shading with Exact Diffusion Inversion via Coupled Transformations (EDICT) (https://arxiv.org/abs/2501.08604)
Comments:
          5 pages

- **What's New**: 이 연구는 유명한 워터마킹 기법인 Gaussian Shading의 성능을 향상시키기 위해 Exact Diffusion Inversion via Coupled Transformations(EDICT) 프레임워크를 통합하는 새로운 접근 방식을 제안합니다. 기존의 Gaussian Shading은 노이즈 잠재 공간에 워터마크를 삽입하고 이미지 생성을 위해 반복적인 노이즈 제거 과정을 포함하는 반면, 그 반전 과정이 정확하지 않아 워터마크 왜곡이 발생할 수 있습니다. 하지만 EDICT는 정확한 역 매핑을 도출할 수 있는 기능을 제공하여 이 과정을 정제함으로써 이미지와 삽입된 워터마크의 보다 정확한 재구성을 가능하게 합니다.

- **Technical Details**: 제안하는 방법은 워터마크가 포함된 노이즈 잠재를 중복시켜 EDICT를 통해 두 잠재 사이의 상호보완적이며 교차적인 노이즈 제거 및 추가 방식으로 처리합니다. EDICT는 서로를 번갈아가며 반전하는 두 개의 연결된 노이즈 벡터를 유지함으로써 수학적으로 정확한 반전을 가능하게 합니다. 실험 결과는 이 통합 접근 방식이 워터마크 회복 충실도에서 약간의, 그러나 통계적으로 유의미한 개선이 있음을 보여줍니다.

- **Performance Highlights**: 본 연구 결과는 EDICT가 기존 확산 기반 워터마킹 기술의 정확하고 강력한 반전 메커니즘을 개선할 수 있는 잠재력을 가지고 있음을 강조합니다. 그리고 EDICT와 Gaussian Shading의 결합이 디지털 워터마킹 분야에서 첫 번째로 시행된 작업임을 밝히며, 신뢰할 수 있고 높은 충실도의 워터마크 삽입 및 추출에 대한 새로운 연구 영역을 열었습니다. 우리 연구는 기존 방법에 비해 약간의 성능 개선을 보이며, 향후 더 많은 연구를 위한 기반이 될 것입니다.



### Densely Connected Parameter-Efficient Tuning for Referring Image Segmentation (https://arxiv.org/abs/2501.08580)
Comments:
          Accepted by AAAI2025

- **What's New**: 이번 논문에서는 Parameter-Efficient Tuning (PET) 접근법을 통해 전통적인 모델 파인튜닝 방법을 대체할 수 있는 DETRIS라는 새로운 프레임워크를 소개합니다. DETRIS는 각 층 간의 밀접한 연결을 통해 저랭크 시각 기능 전달을 향상시켜 비정렬된 인코더에 대한 효과적인 교차 모달 기능 상호작용을 지원합니다. 또한, 텍스트 어댑터를 사용하여 텍스트 기능을 개선하는 방법을 제시합니다.

- **Technical Details**: DETRIS는 Dense Aligner라는 어댑터를 통해 사전 훈련된 모델에 간편하게 통합될 수 있는 구성 요소로, 중간 단계에서 다중 스케일 의미 특성을 포착하기 위한 밀집 혼합 합성곱 모듈과 시각적 및 텍스트 기능 간의 정보 교환을 촉진하는 크로스 얼라이너 모듈로 구성됩니다. 이 프레임워크는 DINO(Oquab et al. 2023)를 시각 인코더로 사용하며, 주요 기여는 효율적인 파라미터 조정 방법론을 통해 RIS 작업에서 강력한 성과를 달성하는 것입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 완전 파인튜닝에 비해 0.9%에서 1.8%의 백본 파라미터 업데이트만으로 요구되는 성능을 초과 뛰어넘는 것으로 나타났습니다. 이는 시각-언어 정렬 및 밀집 예측 작업 수행에 있어 DINO의 가능성을 높이는 데 기여하였습니다. 이러한 결과는 대규모 모델을 효과적으로 적응시키는 새로운 길을 제시합니다.



### Scalable and High-Quality Neural Implicit Representation for 3D Reconstruction (https://arxiv.org/abs/2501.08577)
- **What's New**: 본 논문에서는 최근 제안된 다양한 SDF 기반의 신경 암시적 표면 재구성 방법의 한계를 극복하기 위한 새로운 접근 방식을 소개합니다. 제안된 방법은 여러 개의 독립적인 로컬 신경 SDF를 융합하여 하나의 객체나 장면을 모델링하는 '분할 및 정복' 방법론을 통합하고 있습니다. 이를 통해 고품질의 표면 재구성을 가능하게 하며, 대규모 씬 재구성 또한 지원합니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 단계로 구성됩니다: (1) 객체 구조나 데이터 분포를 기반으로 로컬 복사 필드의 분포 및 중첩 관계 구성, (2) 인접 로컬 SDF 간의 상대 포즈 등록, (3) SDF 블렌딩을 포함합니다. 이 방식은 각 로컬 영역의 독립적인 표현을 통해 고충실도의 표면 재구성을 실현할 수 있게 합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법이 기존의 단일 신경망 기반 방법들보다 3D 재구성의 품질과 확장성이 뛰어난 것을 보여줍니다. 특히, 여덟 개의 로컬 SDF로 재구성할 때 실험 결과에서 Lego의 Chamfer 거리 정확성이 45.6% 향상되었으며, 최대 1200m x 800m 면적을 커버하는 고충실도 텍스처 맵을 이용한 인상 깊은 성과를 얻었습니다.



### MIAFEx: An Attention-based Feature Extraction Method for Medical Image Classification (https://arxiv.org/abs/2501.08562)
Comments:
          In preparation for Journal Submission

- **What's New**: 이 논문은 기존의 특징 추출 기법과 머신 러닝 분류기의 한계를 극복하기 위해 Medical Image Attention-based Feature Extractor (MIAFEx)를 제안합니다. MIAFEx는 Transformer 인코더 아키텍처 내에서 분류 토큰을 향상시키기 위해 학습 가능한 정제 메커니즘을 활용하여 눈에 띄는 특징을 효과적으로 추출합니다. 이는 의료 이미징 데이터의 특성에 적응하고 기능의 품질을 개선하며, 특히 제한된 훈련 데이터 상황에서 높은 정확성을 보여줍니다.

- **Technical Details**: MIAFEx는 Transformer 모델의 [CLS] 토큰을 정제하여 준비됩니다. 이 정제 과정은 학습 가능한 가중치를 통해 동적으로 조절되어, 모델이 분류에 가장 중요한 특징에 집중할 수 있도록 돕습니다. 이러한 접근 방식은 전통적인 특징 추출 기법과 혼합 분류기와 비교하여 뛰어난 성능을 발휘하며, 다양한 의료 이미징 데이터셋에서 효과성을 확인했습니다.

- **Performance Highlights**: MIAFEx는 CNN 및 ViT와 같은 현대적인 딥 러닝 모델과 비교하여 특히 작은 데이터셋에서 성능이 우수하다는 것을 나타냅니다. 실험 결과, MIAFEx는 다양한 복잡한 의료 이미징 데이터셋에서 높은 정확성과 강인성을 기록하였으며, 이는 전통적인 모델들이 일반화하는 데 어려움을 겪는 상황에서도 효과적으로 기능하는 것으로 확인되었습니다.



### DynamicFace: High-Quality and Consistent Video Face Swapping using Composable 3D Facial Priors (https://arxiv.org/abs/2501.08553)
- **What's New**: 이번 논문에서는 비디오 얼굴 스와핑을 위한 새로운 방법인 DynamicFace를 제안합니다. 이 방법은 Stable Diffusion 모델의 강력함과 플러그 앤 플레이(plug-and-play) 시간적 레이어를 이용하여, 정밀하고 분리된 얼굴 조건을 통해 얼굴을 스와핑하게 됩니다. 특히, 3D 얼굴 사전 정보를 활용하여 얼굴의 정체성을 보존하고 복잡한 동작 정보를 효과적으로 처리하는 데 중점을 두었습니다.

- **Technical Details**: DynamicFace의 핵심은 네 가지 세밀한 얼굴 조건을 도입하는 것입니다: 배경, 형태 인식 정규 맵(shape-aware normal map), 표정 관련 랜드마크, 정체성 제거 UV 텍스처 맵입니다. 이러한 조건들은 서로 분리되어 있으며, 3D 얼굴 사전(prior)와의 정렬 전략을 통해 비디오에서의 시간적 일관성을 보장합니다. 이는 3D 얼굴 재구성 모델인 3DDFA-V3로부터 추출된 pose 및 expression 매개변수를 사용하여 이루어집니다.

- **Performance Highlights**: 실험을 통해, DynamicFace는 FF++ 데이터셋에서 최신 기술의 성과를 능가하는 성능을 보여줍니다. 특히, 이미지 품질, 정체성 보존, 표정 정확도 측면에서 우수한 결과를 기록하였으며, 기존의 방법들이 직면했던 문제들을 해결합니다. 이 방법은 비디오 도메인으로 쉽게 전이될 수 있어, 다양한 응용 가능성을 제공합니다.



### The Devil is in Temporal Token: High Quality Video Reasoning Segmentation (https://arxiv.org/abs/2501.08549)
- **What's New**: VRS-HQ는 기존 비디오 추론 분할(Video Reasoning Segmentation, VRS) 방법의 한계를 보완하기 위해 고안된 새로운 접근 방식입니다. 이 모델은 Multimodal Large Language Models (MLLMs)를 활용하여 시간적(dynamic) 특징을 계층적으로 통합합니다. 주요 혁신으로는 Temporal Dynamic Aggregation (TDA)와 Token-driven Keyframe Selection (TKS)가 있습니다.

- **Technical Details**: VRS-HQ는 프레임 수준 및 시간 수준의 토큰을 사용하여 MLLM의 자기 회귀 학습(autoregressive learning)을 통해 지역 및 전역 정보를 효과적으로 캡처합니다. Temporal Dynamic Aggregation(TDA)을 통해 cosin similarity 기반의 가중 합성을 적용하여 프레임 수준의 가시적 특징을 시간 수준의 특징으로 통합합니다. Token-driven Keyframe Selection(TKS) 방식으로, 각 샘플링한 프레임에 대해 SAM2를 통해 occlusion score를 계산하여 키프레임을 선택하는 신뢰성을 강화합니다.

- **Performance Highlights**: VRS-HQ는 ReVOS 데이터셋에서 VISA보다 5.9%, 12.5%, 9.1%의 J&F 점수 개선을 달성하며 최첨단 성능을 입증하였습니다. 또한 여러 비디오 분할 기준에서 이전 방법들을 초월하여 7.3%, 5.6%, 6.5%의 성능 향상을 나타냈습니다. 이러한 결과는 VRS-HQ의 강력한 시간적 추론 및 분할 능력을 강조합니다.



### Comprehensive Subjective and Objective Evaluation Method for Text-generated Video (https://arxiv.org/abs/2501.08545)
- **What's New**: 최근 텍스트-비디오 생성(T2V) 기술의 발전은 Gen3, Pika 및 Sora와 같은 모델들을 통해 그 적용성과 인기를 크게 넓혔습니다. 이러한 발전은 텍스트로 생성된 비디오의 지각 품질을 평가하고, 비디오 생성 모델을 최적화하는 정확한 품질 평가 메트릭의 필요성을 증가시켰습니다. 하지만 생성된 비디오의 품질을 평가하는 데는 복잡한 왜곡이 존재하여 여전히 도전 과제가 많습니다.

- **Technical Details**: 이 연구에서는 텍스트 생성 비디오 평가를 위한 대규모 벤치마크 데이터셋인 T2VEval-Bench를 구축하였습니다. 이 데이터셋은 148개의 텍스트와 12개 모델에 의해 생성된 1,783개의 비디오를 포함하고 있으며, 주관적 평가를 통해 비디오 품질, 미학 품질, 실제성, 텍스트-비디오 일관성 등 다섯 가지 주요 점수를 수집하였습니다. 또한 T2VEval 모델을 개발하여 비디오의 품질, 진위성, 일관성을 평가하는 세 가지 지점을 평가합니다.

- **Performance Highlights**: T2VEval은 여러 메트릭에서 최첨단 성능을 달성하며, 강력한 객관적 평가 모델로 자리 잡았습니다. 이 모델은 각 가지 평가 지점을 독립적으로 학습하도록 하는 프로그레시브 트레이닝 전략을 통해 설계되었습니다. 실험 결과는 T2VEval이 다양한 T2V 모델의 성능을 효과적으로 평가하고, 향후 연구에 넓은 응용 가능성을 제공함을 보여줍니다.



### Multimodal Fake News Video Explanation Generation (https://arxiv.org/abs/2501.08514)
- **What's New**: 이번 논문에서는 Fake News Video Explanation (FNVE)이라는 새로운 문제를 제안합니다. 이 과제는 비디오와 캡션 텍스트가 포함된 다중 모달 뉴스에서 자연어 설명을 생성하여 예측의 진실성을 드러내는 것을 목표로 합니다. 이를 위해, 각각의 설명이 뉴스 스레드의 속성을 설명하는 자연어 문장으로 구성된 FakeNVE 데이터셋을 개발하고, 이를 벤치마킹 하기 위해 다중 모달 트랜스포머 기반 아키텍처를 사용했습니다.

- **Technical Details**: 본 연구에서는 Multimodal Relational Graph Transformer (MRGT) 모델을 사용하여 FakeNVE 데이터셋을 평가합니다. MRGT는 제목, 비디오 프레임, OCR 텍스트 및 관련 뉴스 컨텍스트를 포괄하는 다중 모달 맥락 관계를 종합적으로 나타냅니다. 또한, 설명 생성을 위한 생성기로 BART 기반의 자가 회귀 디코더를 활용합니다.

- **Performance Highlights**: 경험적 결과에 따르면, 다양한 기준선 모델에서 ​​긍정적인 결과를 도출하였으며, 여러 평가 메트릭에 걸쳐 강력한 성능을 보였습니다. 설명 생성에 대한 인간 평가도 수행하여, 적합성과 유창성 모두에서 높은 점수를 달성했습니다. 이 연구에서 제안한 방법의 우수성을 검증하기 위해 광범위한 질적 및 양적 실험을 실시했으며, 코드를 커뮤니티에 공개하여 연구 발전에 기여할 계획입니다.



### Yuan: Yielding Unblemished Aesthetics Through A Unified Network for Visual Imperfections Removal in Generated Images (https://arxiv.org/abs/2501.08505)
- **What's New**: 이번 논문에서는 새로운 프레임워크인 Yuan을 소개합니다. Yuan은 텍스트에서 이미지 생성 과정에서 발생하는 시각적 결함을 자동으로 수정하는 기능을 갖추고 있습니다. 이 프레임워크는 텍스트 프롬프트와 분할된 이미지에 조건을 부여하여 정밀한 마스크를 생성하며, 이전 방법론에서는 수작업이 필요했던 과정을 자동화합니다.

- **Technical Details**: Yuan은 이미지 내 특정 결함을 자동으로 탐지하고 수정하는 두 가지 모듈로 구성됩니다. 첫 번째 모듈인 grounded segmentation은 사전 정의된 마스크 없이 결함을 식별합니다. 이후 inpainting 모듈은 식별된 영역에 대해 맥락에 일치하는 내용을 통합하여 원본 이미지의 완전성과 품질을 유지합니다.

- **Performance Highlights**: Yuan은 ImageNet100 및 Stanford Dogs와 같은 공개 데이터셋과 사용자 맞춤형 데이터셋에서 우수한 성능을 입증했습니다. 자동화된 결함 탐지와 맥락 인식 재구성을 통해 이 프레임워크는 정량적 메트릭에서 우수한 점수를 달성하며, 품질 평가에서도 긍정적인 결과를 보여줍니다. 이러한 결과는 다양한 분야에서 AI 생성 이미지의 품질과 적용 가능성을 크게 향상시킬 수 있음을 시사합니다.



### SuperSAM: Crafting a SAM Supernetwork via Structured Pruning and Unstructured Parameter Prioritization (https://arxiv.org/abs/2501.08504)
- **What's New**: 이 논문에서는 Vision Transformer (ViT) 기반의 Neural Architecture Search (NAS)를 위한 새로운 검색 공간 설계 전략을 제안합니다. 특히, Segment Anything Model (SAM)을 가중치 공유(supernetwork)인 SuperSAM으로 변환하여 다양한 하위 네트워크를 효율적으로 탐색합니다. 이 접근 방식은 계층별로 구조적 가지치기와 매개변수 우선 순위를 자동화하여 최적의 하위 네트워크를 제안합니다.

- **Technical Details**: SuperSAM은 SAM 모델을 변형하여 다양한 자원 제약에 맞는 하위 네트워크를 생성할 수 있는 "탄력적인" 구조로 설계되었습니다. 논문에서 제시된 방법은 프루닝(pruning) 기법을 사용하여 각 transformer 계층의 중요도를 평가하고 중요하지 않은 계층에 대해 확률적 가지치기를 실시합니다. 또한, MLP 블록의 매개변수 우선 순위 지정과 함께 각 계층에서 중요한 매개변수를 유지하여 결과적으로 연결된 하위 네트워크를 생성합니다.

- **Performance Highlights**: SuperSAM에서 파생된 하위 네트워크는 기존의 사전 훈련된 SAM ViT-B 모델에 비해 30-70% 더 작은 크기를 가지고 있지만, 성능에서 실제로 우수한 성과를 보입니다. 이를 통해 기존 모델보다 적은 자원이 소요되면서도 동일한 수준의 성능을 유지하는 효율적인 구조를 달성했습니다. 이 연구는 ViT NAS의 검색 공간 설계에 대한 새로운 접근 방식을 제시합니다.



### FLAVARS: A Multimodal Foundational Language and Vision Alignment Model for Remote Sensing (https://arxiv.org/abs/2501.08490)
- **What's New**: 이번 연구에서는 FLAVARS라는 새로운 사전 훈련(pretraining) 방법을 제안합니다. 이는 대비 학습(contrastive learning)과 마스크 모델링(masked modeling)의 장점을 결합하고, 지리적 정렬(geospatial alignment)을 위한 대비 위치 인코딩(contrastive location encoding)을 추가합니다. FLAVARS 모델은 SkyCLIP을 기준으로 KNN 분류 및 의미 분할(semantic segmentation) 같은 비전 전용(task-specific) 작업에서 현저한 성능 향상을 보여주었습니다.

- **Technical Details**: FLAVARS는 대규모 원거리 센싱 이미지 및 텍스트 설명 데이터셋인 SkyScript에서 훈련되었습니다. 기존의 다중 모달 사전 훈련 방식이 성능 저하를 초래한 반면, FLAVARS는 비전 전용 작업에서 성능을 크게 향상시키면서 제로 샷 분류(zero-shot classification) 능력을 유지합니다. 특히, 지리적 이미지-위치 정렬을 통해 사전 훈련 성능이 개선되었음을 발견했습니다.

- **Performance Highlights**: FLAVARS 사전 훈련 방식은 SpaceNet1 데이터셋에서 +6% mIOU 향상을 기록하며, KNN 분류 및 의미 분할 작업에서 SkyCLIP보다 월등한 성능을 보였습니다. 결과적으로, FLAVARS는 기존 CLIP 기반 사전 훈련에 비해 비전-언어 정렬(vision-language alignment) 능력을 자랑하지만, 데이터 세트 간 데이터 분할 기준이 다르기 때문에 보다 일반화된 성능 향상은 추가적인 연구가 필요합니다.



### Benchmarking Classical, Deep, and Generative Models for Human Activity Recognition (https://arxiv.org/abs/2501.08471)
Comments:
          48 pages, 21 Figures

- **What's New**: 이번 연구는 Human Activity Recognition (HAR)의 다양한 모델, 즉 전통 기계 학습, 딥 러닝 아키텍처 및 Restricted Boltzmann Machines (RBMs)의 성능을 비교합니다. 연구진은 UCI-HAR, OPPORTUNITY, PAMAP2, WISDM, Berkeley MHAD라는 5개의 주요 벤치마크 데이터 세트를 사용하여 다양한 모델의 정확성, 정밀도, 재현율, F1-score 등 다양한 지표를 평가하였습니다. CNN 모델이 전 데이터 세트에서 우수한 성능을 보인다는 것을 발견하였으며, 특히 Berkeley MHAD에서 두드러진 성과를 나타냈습니다.

- **Technical Details**: 본 논문은 HAR을 위한 다양한 모델들이 각기 다른 강점과 한계를 가지고 있음을 강조합니다. 전통적인 모델인 Decision Trees, Random Forests가 소규모 데이터 세트에서 좋은 성능을 보이나, 대규모 복잡한 데이터에서는 어려움을 겪는 반면, RBM 기반 모델은 주로 feature learning에 유용한 잠재력을 보였습니다. 딥 러닝 아키텍처는 복잡한 시공간 패턴을 학습할 수 있지만, 레이블이 있는 데이터와 컴퓨팅 자원을 많이 요구하는 경향이 있습니다.

- **Performance Highlights**: CNN 모델이 모든 데이터 세트에서 최고 성능을 보여주었으며, 이는 다양한 환경에서 HAR의 신뢰성을 높이는 데 기여할 수 있습니다. 기계 학습 및 딥 러닝을 포함한 여러 모델이 다양한 상황에서 어떻게 성능이 달라지는지를 분석하여 연구자들이 필요한 모델을 선택하는 데 도움을 줄 수 있습니다. 결과적으로, 이 연구는 의료, 스마트 환경 및 보안 분야와 같은 실제 응용 프로그램에 대한 귀중한 통찰력을 제공합니다.



### Detecting Contextual Anomalies by Discovering Consistent Spatial Regions (https://arxiv.org/abs/2501.08470)
- **What's New**: 이 논문에서는 비디오 이상 감지를 위한 공간적 맥락 모델링 방법을 제안합니다. 주된 아이디어는 가우시안 혼합 모델(Gaussian mixture models)을 사용하여 객체 수준 활동이 유사한 영역을 클러스터링하는 것입니다. 이러한 간단한 접근법은 경쟁 모델에 비해 수량적으로 더 적은 매개변수를 사용하며, 복잡한 공간 맥락 의존 Street Scene 데이터셋에서 최첨단 성능을 달성합니다.

- **Technical Details**: 논문에서는 '정상' 객체 행동을 학습하기 위해 비디오 내의 다양한 위치에서 객체 행동을 기반으로 하는 공간적 영역을 자연스럽게 발견하는 방법을 설명합니다. 고해상도 지역 발견을 통해, 간단한 가우시안 혼합 모델을 학습하여 공간 맥락 의존 이상을 감지할 수 있음을 보여줍니다. 이 접근방법은 필요한 모델 수를 크게 줄이고 사용자에게 더 나은 해석 가능성을 제공합니다.

- **Performance Highlights**: 제안된 방법은 사용자가 지정한 소수의 활동 지역을 발견하여 Street Scene 데이터셋에서 최첨단 성능을 달성합니다. 이 지역들은 일반적으로 다음과 같은 공간 컨텍스트에서 의미 있는 영역: 교통 차선, 자전거 도로, 보도 등을 포함합니다. 새 객체가 주어진 지역 모델에 따라 정상 또는 비정상으로 분류될 수 있으며, 예를 들어 도로에 있는 보행자나 불법 주정차된 차량과 같은 공간 맥락 의존 이상을 발견할 수 있습니다.



### Predicting Performance of Object Detection Models in Electron Microscopy Using Random Forests (https://arxiv.org/abs/2501.08465)
Comments:
          14 pages, 9 figures, 3 tables

- **What's New**: 이 연구는 새로운 비표시 데이터셋에 객체 탐지 모델을 적용할 때 예측 불확실성을 정량화하는 접근 방식을 제안합니다. 특히, 금속 합금의 TEM 이미지에서 방사선 유도 구멍을 감지하는 데 초점을 맞추고 있습니다. 무작위 숲 회귀 모델을 개발하여 객체 탐지 F1 점수를 예측하며, 이는 객체의 정확한 위치와 분류 능력을 평가하는 통계적 지표입니다.

- **Technical Details**: 연구에서는 먼저 Mask R-CNN 모델을 통한 예측의 특징을 사용하여 랜덤 포레스트 회귀 모델을 생성했습니다. 이 모델은 F1 점수를 예측하고, 훈련된 모델의 평균 절대 오차(MAE)는 0.09, $R^2$ 점수는 0.77로 통계적으로 유의미한 상관관계를 보여줍니다. 이 방법은 다양한 이미징 및 재료 도메인을 가진 세 가지 TEM 이미지 데이터셋에서 강건함을 입증했습니다.

- **Performance Highlights**: 연구 결과, 새로운 비표시 이미지에 대한 빠른 예측이 가능하며, 결함 탐지 및 세분화 모델 예측의 신뢰성을 평가하는 데 유용합니다. 이 접근 방식은 특정 데이터셋에 대한 모델 적용 가능성을 파악하고, 도메인 이동의 가능성에 대한 정보를 제공하여 사용자가 최적의 결과를 위해 추가 데이터로 모델을 조정할 필요성이 있는지 평가할 수 있게 합니다.



### Towards Zero-Shot & Explainable Video Description by Reasoning over Graphs of Events in Space and Tim (https://arxiv.org/abs/2501.08460)
- **What's New**: 최근 머신러닝의 발전 속에서, Transformer가 컴퓨터 비전과 자연어 처리와 같은 다양한 분야에서 주목받고 있습니다. 이 연구는 비전(vision)과 언어(language) 간의 관계를 이해하는 문제에 도전하며, 비전과 언어 모델 간의 연계성을 설명 가능하고 체계적으로 연결하기 위한 새로운 접근법을 제안합니다. 이러한 접근법은 시간과 공간의 사건(event)을 기반으로 하여 자연어로 비디오를 설명하는 문제를 해결하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 비디오의 프레임에서 발생하는 다양한 고수준 정보를 활용하여 Graph of Events in Space and Time (GEST)를 구축합니다. GEST는 비디오의 물리적, 시간적, 의미적 요소를 나타내는 노드(nodes)와 엣지(edges)로 구성되어 있습니다. 이 표현을 통해 비디오는 명확하고 체계적으로 분석되며, proto-language를 생성한 후 이를 자연어 설명으로 변환하는 두 단계의 과정을 거칩니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 데이터셋에서 수집한 비디오에 대해 풍부하고 관련성 높은 텍스트 설명을 생성할 수 있음을 검증했습니다. 기존의 비디오 설명 모델들이 짧은 캡션(caption)만을 생성하는 것과 달리, VLLMs과의 조합을 통해 더 긴 설명을 가능하게 하였습니다. 본 연구에서는 Bleu와 ROUGE 같은 표준 메트릭(Standard metric)을 활용하여 성과를 평가하고 효율성을 입증했습니다.



### Vchitect-2.0: Parallel Transformer for Scaling Up Video Diffusion Models (https://arxiv.org/abs/2501.08453)
- **What's New**: Vchitect-2.0는 대규모 텍스트-비디오 생성 을 위한 비디오 확산 모델을 확장하기 위해 설계된 병렬 트랜스포머 아키텍처입니다. 이 시스템은 텍스트 설명과 생성된 비디오 프레임 간의 일관된 정렬을 이루기 위해 새로운 Multimodal Diffusion Block을 도입했습니다. 또한, 메모리 효율적인 학습 프레임워크를 통해 장기간 비디오 시퀀스를 효율적으로 훈련할 수 있습니다.

- **Technical Details**: Vchitect-2.0은 텍스트 프롬프트와 프레임 간의 기능 정렬을 보장하는 멀티모달 확산 블록을 특징으로 합니다. 혼합 병렬성(hybrid parallelism) 프레임워크를 통해 메모리 최적화 기술과 결합하여 확장성과 효율성을 제공하며, 분산 시스템에서의 고해상도 비디오의 효율적인 생성을 가능하게 합니다. 이 시스템은 여러 가지 메모리 저감 기술을 포함하며, 장시간 비디오 시퀀스의 훈련을 지원합니다.

- **Performance Highlights**: 광범위한 벤치마크 평가에서 Vchitect-2.0은 주요 메트릭에서 기존 기술을 일관되게 능가하는 성능을 보였습니다. 기술적 분석을 통해 더욱 부드러운 프레임 전환을 이룩하며, 모션 아티팩트를 효과적으로 줄였습니다. 하이브리드 병렬성 프레임워크는 훈련 확장성을 증진시키고 메모리 소모를 줄이는 데 기여하였으며, ablation 연구에서도 멀티모달 확산 블록과 병렬 전략의 중요성이 부각되었습니다.



### Poseidon: A ViT-based Architecture for Multi-Frame Pose Estimation with Adaptive Frame Weighting and Multi-Scale Feature Fusion (https://arxiv.org/abs/2501.08446)
- **What's New**: 이번 논문에서는 ViTPose 모델을 기반으로 하는 새로운 다중 프레임 포즈 추정 아키텍처인 Poseidon을 제안합니다. 이 모델은 시간 정보를 통합하여 복잡한 동작을 보다 정확하고 강인하게 이해할 수 있도록 설계되었습니다. Poseidon은 시간을 고려한 적응형 프레임 가중치(AFW) 메커니즘, 다중 스케일 특징 융합(MSFF) 모듈, 그리고 중앙 프레임과 맥락 프레임 간의 효과적인 정보 교환을 위한 크로스 어텐션 모듈 등을 도입합니다.

- **Technical Details**: Poseidon 아키텍처는 ViTPose 모델을 활용하여 다중 프레임 포즈 추정에 최적화되었습니다. 이 과정에서 AFW 메커니즘을 통해 특정 프레임의 중요성을 동적으로 평가하고, MSFF 모듈을 통해 서로 다른 레이어의 특징을 결합하여 세부정보와 높은 수준의 의미 정보를 동시에 포착합니다. 또한, 크로스 어텐션 모듈은 중앙 및 맥락 프레임 간의 원활한 정보 교환을 통해 시간적 일관성을 향상시킵니다.

- **Performance Highlights**: Poseidon은 PoseTrack21 및 PoseTrack18 데이터셋에서 mAP 점수 88.3 및 87.8을 기록하며 기존의 방법들을 능가하는 성능을 보였습니다. 복잡한 비디오 시나리오에서 향상된 포즈 추정 정확도를 제공하며, 실제 애플리케이션에 적합한 확장성 및 계산 효율성을 지니고 있습니다. 이를 통해 다중 프레임 포즈 추정의 최첨단 성능을 달성하였습니다.



### Instruction-Guided Fusion of Multi-Layer Visual Features in Large Vision-Language Models (https://arxiv.org/abs/2501.08443)
- **What's New**: 이 논문에서는 다양한 인코더 레이어에서 추출한 시각적 특징이 LVLM(대규모 비전-언어 모델)의 성능에 미치는 영향을 체계적으로 조사하였습니다. 다양한 작업 카테고리에서 18개의 벤치마크를 대상으로 분석한 결과, 각기 다른 레이어의 시각적 특징들이 상호 보완적이며, 기존의 균일한 융합 방식이 최적의 성능을 발휘하지 못함을 발견하였습니다.

- **Technical Details**: 제안된 지침 기반 비전 집합기를 통해, LVLM이 입력된 텍스트 지침에 따라 동적으로 다중 레이어 특징을 통합할 수 있도록 하였습니다. 이는 시각적 토큰의 수를 증가시키지 않으면서도 작업 특화된 특징 통합을 가능하게 합니다. 제안된 모듈은 LLaVA-v1.5 프레임워크에 통합되어 우수한 성능 개선을 이끌어냈습니다.

- **Performance Highlights**: 실험 결과, 미드-투-하이 레벨의 특징이 의미적 작업에서 우위를 점하며, 저수준 특징이 세밀한 인식 작업에 필수적이라는 것을 확인하였습니다. 제안된 모듈은 기존의 태스크-비분리적인 융합 방법들보다도 더 나은 성능을 보여주며, 레이어별 특징의 중요성을 강조하는 귀중한 통찰력을 제공합니다.



### FARE: A Deep Learning-Based Framework for Radar-based Face Recognition and Out-of-distribution Detection (https://arxiv.org/abs/2501.08440)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 이 연구에서는 단거리 FMCW 레이더를 활용한 얼굴 인식 및 OOD (out-of-distribution) 탐지 시스템을 제안합니다. 이 시스템은 Range-Doppler 및 micro Range-Doppler 이미지를 사용하며, 인식 모델의 두 개의 경로를 통해 ID 얼굴 분류와 OOD 탐지를 동시에 수행합니다. 두 단계로 훈련이 진행되며, 첫 단계에서는 triplet loss를 사용하여 ID 얼굴의 분류 정확도를 최적화합니다.

- **Technical Details**: 제안하는 아키텍처는 두 가지 경로, 즉 기본 경로(PP)와 중간 경로(IP)를 통해 구성되어 있습니다. PP는 ID 얼굴의 정확한 분류를 담당하고, IP는 OOD 탐지를 위한 구조입니다. 첫 번째 단계에서는 PP를 훈련시키고, 두 번째 단계에서는 PP를 고정하여 IPs를 훈련하여 OOD 탐지를 수행합니다.

- **Performance Highlights**: 제안된 방식을 통해 60 GHz FMCW 레이더로 생성한 데이터셋에서 ID 얼굴 분류 정확도 99.30%와 OOD 탐지 AUROC 96.91%를 달성했습니다. 또한, FARE는 기존 OOD 탐지 방법들에 비해 뛰어난 성능을 보이며, 이는 보안과 신뢰성을 위한 중요한 발전을 의미합니다.



### Cross-Modal Transferable Image-to-Video Attack on Video Quality Metrics (https://arxiv.org/abs/2501.08415)
Comments:
          Accepted for VISAPP 2025

- **What's New**: 최근 연구에서 현대 영상 및 비디오 품질 평가(IQA/VQA) 지표가 적대적 공격에 취약하다는 것이 밝혀졌습니다. 이는 공공 벤치마크와 자율주행 같은 보다 중대한 상황에서 이러한 지표에 의존하는 것의 안전성에 대한 우려를 불러일으킵니다. 본 논문에서는 이미지 품질 지표(IQA)와 CLIP 모듈을 활용하여 비디오 품질 평가(VQA) 모델에 대한 적대적 공격의 취약성을 탐구하는 새로운 방법, IC2VQA를 제안합니다.

- **Technical Details**: IC2VQA 접근법은 여러 고해상도 비디오와 3개의 대상 VQA 모델을 사용하여 포괄적인 실험을 수행한 결과 기존 방법들에 비해 뛰어난 성능을 보여주었습니다. 이 방법은 이미지와 비디오의 저수준 특징 공간의 유사성에 의해 동기가 부여되었으며, 저수준 세멘틱을 효과적으로 포착하는 CLIP 모델의 추가를 통해 전이 가능성을 높였습니다. 논문에서는 IQA와 VQA 지표의 심층 특성들 간의 상관관계를 분석하였습니다.

- **Performance Highlights**: 실험 결과 IC2VQA는 세 개의 블랙박스 VQA 모델에 대한 공격에서 높은 성공률을 달성하였습니다. 또한, 기존의 블랙박스 공격 전략들과 비교하여 같은 반복 횟수 및 공격 강도에서 공격 성공률에서의 우수성을 강조했습니다. 이 방법은 강력한 VQA 지표에 대한 심층 분석에 기여할 것으로 예상됩니다.



### Leveraging 2D Masked Reconstruction for Domain Adaptation of 3D Pose Estimation (https://arxiv.org/abs/2501.08408)
Comments:
          16 pages, 7 figures

- **What's New**: 이 연구는 RGB 기반 3D 포즈 추정 방법의 효율성을 높이기 위해 비지도 학습 도메인 적응 프레임워크를 소개합니다. 기존의 방법들은 훈련 데이터와 분포가 다른 테스트 이미지에는 잘 작동하지 않았으나, 이 연구에서는 마스킹된 이미지 모델링(Masked Image Modeling, MIM)을 활용하여 라벨이 없는 데이터를 활용하는 방안을 제안합니다. 이를 통해 훈련 과정에서 다양한 데이터를 효과적으로 사용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서 제안하는 기술은 전경 중심 재구성과 주의 정규화(attention regularization)입니다. 전경 중심 재구성은 주어진 데이터에서 독특한 정보나 특징을 부각시켜, 라벨이 없는 데이터의 활용 효율성을 높이는 데 도움을 줍니다. 주의 정규화는 모델이 중요한 요소에 더 집중할 수 있도록 학습을 유도합니다. 이러한 방법은 다양한 데이터셋을 통한 실험을 통해 검증됩니다.

- **Performance Highlights**: 본 연구는 인간 및 손 포즈 추정 작업에서 여러 데이터셋을 대상으로 성능 평가를 진행했습니다. 특히 교차 도메인(cross-domain) 시나리오를 통해 검증한 결과, 제안된 방법은 모든 데이터셋에서 최첨단 정확도를 달성했습니다. 이는 제안된 비지도 학습 접근 방식이 실제 데이터 환경에서 효과적임을 입증하는 사례로 평가됩니다.



### Weight Averaging for Out-of-Distribution Generalization and Few-Shot Domain Adaptation (https://arxiv.org/abs/2501.08361)
Comments:
          Master Thesis

- **What's New**: 이번 논문에서는 out-of-distribution generalization (OOD 일반화) 문제를 해결하기 위해 weight averaging (WA)와 sharpness-aware minimization (SAM)이라는 두 가지 기법을 소개합니다. WA는 서로 다른 하이퍼파라미터로 훈련된 여러 모델의 가중치를 평균내어 OOD 일반화 성능을 향상시키며, SAM은 평평한 영역에서의 미니마를 찾아 네트워크를 최적화하여 분포 변화에 대한 성능을 개선합니다. 새로운 접근으로, WA에서 기울기 유사도를 손실 정규화 기법으로 도입하여 모델의 다양성을 높이고, WA와 SAM을 결합하여 few-shot domain adaptation (소수 샷 도메인 적응)의 문제를 해결하고자 합니다.

- **Technical Details**: 논문에서 제안하는 방법은 MNIST, SVHN, USPS, MNIST-M과 같은 숫자 데이터셋 및 VLCS, PACS 같은 도메인 적응 데이터셋에서 실험했습니다. 신경망 학습 과정에서 기울기 유사도를 정규화하여 모델 다양성을 명시적으로 증가시키는 새로운 손실 함수를 도입했습니다. WA와 SAM의 결합을 통해 소수의 샷에서도 더 나은 OOD 일반화 성능을 발휘하도록 하였습니다.

- **Performance Highlights**: 실험 결과 WA와 SAM을 결합함으로써 OOD 일반화 성능이 크게 향상되었고, 소수 샷 도메인 적응 정확성이 상당히 증가했습니다. 이 방법들은 기존의 기법보다 다양한 상황에서 더 나은 성능을 보였으며, 향후 연구 방향에 대한 기초를 마련하고 있습니다. 이처럼 성능 개선을 통해 OOD 문제와 소수 샷 적응에 대한 새로운 해결책을 제공하는 점이 주목받고 있습니다.



### SCOT: Self-Supervised Contrastive Pretraining For Zero-Shot Compositional Retrieva (https://arxiv.org/abs/2501.08347)
Comments:
          Paper accepted at WACV 2025 in round 1

- **What's New**: 이 논문에서는 SCOT(Self-supervised COmpositional Training)이라는 새로운 zero-shot compositional pretraining 전략을 제안합니다. 이 방법은 기존의 레이블이 달린 triplet 데이터셋을 필요로 하지 않으며, 다양한 캡션이 달린 이미지 데이터셋을 활용하여 open-world 일반화 능력을 보여줍니다. SCOT는 성능 향상을 위해 대규모 contrastively-pretrained 비전-언어 모델의 시각적 및 텍스트 표현의 근접성을 활용합니다.

- **Technical Details**: SCOT는 입력으로 제공된 이미지와 그에 해당하는 캡션을 바탕으로 modification 텍스트를 생성하고, 이를 통해 CIR 모델을 훈련시킵니다. 이 과정에서 생성된 modification 텍스트가 참조 이미지와 조합되어 contrastive image retrieval loss를 최적화하는 방식입니다. 이러한 접근법은 기존의 inversion 기반 기법과는 달리 compositional 모델을 직접 훈련시키며 점차 업그레이드가 가능하여 다양한 도메인에 쉽게 적응할 수 있습니다.

- **Performance Highlights**: SCOT는 FashionIQ와 CIRR와 같은 표준 벤치마크에서 기존의 zero-shot compositional retrieval 방법들 뿐만 아니라 많은 fully-supervised 방법들보다 뛰어난 성능을 보여주었습니다. 이 접근법은 0-shot 조건에서 좋은 일반화 능력을 가지며, 자동 생성된 데이터로 인해 수작업으로 구성된 기존 데이터셋에 대한 의존도를 크게 줄일 수 있습니다.



### Vision Foundation Models for Computed Tomography (https://arxiv.org/abs/2501.09001)
Comments:
          6 figures, followed by 9 Extended Data Figures and a Supplementary Information document

- **What's New**: 이번 연구에서는 CT-FM이라고 하는 대규모 3D 이미지 기반 사전 훈련 모델을 개발하여 방사선학(radiology) 분야의 다양한 복합 작업을 수행할 수 있도록 했습니다. CT-FM은 148,000개의 CT 스캔을 통해 label-agnostic contrastive learning 기법을 이용해 사전 훈련되었습니다. 이는 다른 기존 모델들에 비해 월등한 성능을 보이는 것으로 평가됩니다.

- **Technical Details**: CT-FM은 전체 신체 및 종양 분할(whole-body and tumor segmentation), 머리 CT triage, 의료 이미지 검색(medical image retrieval), 의미적 이해(semantic understanding) 등 네 가지 작업 범주에서 평가되었습니다. 이 모델은 해부학적으로 지역을 클러스터링하고 스캔 간 유사한 해부학적 및 구조적 개념을 인식할 수 있는 능력을 보여주었습니다. 또한, 테스트-재시험(test-retest) 설정에서도 강건함을 유지하며, 임베딩(embedding)에 부착된 합리적인 주요 영역(salient regions)을 표시했습니다.

- **Performance Highlights**: CT-FM은 정량적 성공을 넘어 대규모 의료 영상 기초 모델의 가치를 입증하였습니다. 또한, 모델 가중치(weights), 코드(code), 데이터(data)를 오픈 소싱(open-source)하여 방사선학 분야에서 더욱 적응적이고 신뢰할 수 있으며 해석 가능한 AI 솔루션을 지원하고자 합니다.



### Multi-View Transformers for Airway-To-Lung Ratio Inference on Cardiac CT Scans: The C4R Study (https://arxiv.org/abs/2501.08902)
Comments:
          Accepted to appear in Proceedings of International Symposium on Biomedical Imaging (ISBI), 2025

- **What's New**: 이 연구에서는 관상 CT(CT) 이미지를 사용하여 폐 크기 대비 기도 나무의 내경 비율(airway tree lumen to lung size ratio, ALR)을 추정하는 새로운 방법을 발표합니다. 이러한 ALR 값은 만성 폐쇄성 폐질환(Chronic Obstructive Pulmonary Disease, COPD)과 COVID-19의 중증도 및 후속 증상(Post-acute Sequelae of SARS-CoV-2 Infection, PASC)과의 관계를 연구하는 데 중요한 역할을 합니다. 연구 팀은 다각적(Multi-view) 접근 방식을 통해 ALR 값을 추론할 수 있는 Multi-view Swin Transformer라는 혁신적인 모델을 도입했습니다.

- **Technical Details**: 이 모델은 다중 뷰의 관상 CT 이미지를 이용하여 폐의 ALR 값을 구하고, Multi-Ethnic Study of Atherosclerosis (MESA)에서 수집된 쌍(pair) 형태의 전체 폐(full-lung) 및 관상 CT 데이터셋을 사용하여 지도 학습(supervised training)을 진행하였습니다. 연구 결과, 제안된 네트워크는 세분화된 관상 CT 이미지를 통한 직접적인 ALR 추정 방법보다 훨씬 높은 성능을 보여주었고, 전체 폐 ALR의 진실값(ground truth)과 유사한 정확도와 재현성을 달성했습니다.

- **Performance Highlights**: 이 연구에서 제안된 모델은 ALR 값을 추론할 때 높은 정확도와 재현성을 보여주었습니다. 특정 알림 기준을 이용한 반복 측정(re-scan)의 결과와 비교했을 때, 본 모델의 성능은 기존 방법에 비해 현저히 우수한 것으로 나타났습니다. 또한, 이를 통해 광범위한 역학 연구에서 사용 가능한 관상 CT 이미지를 활용하여 COPD와 COVID-19의 잠재적 관계를 더 깊이 분석할 수 있는 가능성을 제시합니다.



### Exploring Task-Level Optimal Prompts for Visual In-Context Learning (https://arxiv.org/abs/2501.08841)
- **What's New**: 이 논문에서는 Visual In-Context Learning (VICL) 접근법의 새로운 가능성을 발견했습니다. 기존의 샘플별 프롬프트(search strategy)는 높은 비용과 과적합(overfitting)의 위험을 동반하는 반면, 기존의 여러 샘플에서 동일한 프롬프트가 최적 성능을 달성한다는 점을 지적하고 있습니다. 이에 따라, 새로운 태스크 레벨 프롬프트(task-level prompting) 접근법을 제안하여 비용을 절감하고 성능을 유지할 수 있음을 보였습니다.

- **Technical Details**: 논문에서 제안하는 두 가지 새로운 프롬프트 검색 전략은 Top-K 프롬프트(selection) 방법과 Greedy 검색(search) 방법입니다. Top-K 방법은 개별 데모의 성능을 측정한 후 가장 우수한 K개의 데모를 선택하여 최종 프롬프트를 생성합니다. Greedy 검색 방법은 각 단계에서 성능을 극대화할 수 있는 가장 좋은 데모를 선택하는 방식으로, 알고리즘의 각 단계에서 지역 최적 해를 찾습니다.

- **Performance Highlights**: 제안된 두 가지 전략은 다양한 다운스트림 태스크에서 효과적인 성과를 보여주었습니다. 특히, 98% 이상의 프롬프트 검색 시간을 절약하면서도 기존 방법들에 비해 6.2% 이상의 상대적 성능 향상을 달성했습니다. 이러한 결과는 태스크 레벨 프롬프트의 효과성과 효율성을 입증하며, 새로운 접근법이 과거의 방법들을 초월할 수 있음을 강조하고 있습니다.



### MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents (https://arxiv.org/abs/2501.08828)
Comments:
this https URL

- **What's New**: 이번 연구는 Multi-Modal Document Retrieval을 위한 새로운 벤치마크인 MMDocIR을 소개합니다. MMDocIR은 페이지 레벨(page-level)과 레이아웃 레벨(layout-level) 검색의 두 가지 주요 작업으로 구성되어 있으며, 이를 통해 사용자 질문에 대한 더욱 세분화된 답변을 제공할 수 있습니다. 기존의 벤치마크들에서는 미비했던 요소를 보완하여, 문서 내에서의 정확한 검색 성능 평가를 가능하게 합니다.

- **Technical Details**: MMDocIR은 313개의 문서와 1,685개의 질문, 그리고 73,843개의 질문 응답 쌍으로 구성된 훈련 세트를 포함합니다. 본 연구에서는 특히 레이아웃을 정밀하게 표시하기 위한 주석(annotation) 작업을 수행하였으며, 각 페이지에 대한 증거를 포함하는 레이블을 제공합니다. 또한, 비주얼 기반의 검색 시스템과 텍스트 기반 시스템의 성능 차이를 분석하여 비주얼 요소의 중요성을 강조합니다.

- **Performance Highlights**: 엄격한 실험을 통해 비주얼 검색기가 텍스트 검색기보다 상당히 뛰어난 성능을 보인다는 사실을 확인했습니다. 최신 실험 결과는 MMDocIR 훈련 세트가 multi-modal document retrieval 과정에 긍정적인 영향을 미친다는 것을 보여줍니다. 이러한 결과는 비주얼 요소를 통합하는 것이 multi-modal document retrieval를 향상시키는 데 중요한 역할을 한다는 것을 강조합니다.



### Boosting Diffusion Guidance via Learning Degradation-Aware Models for Blind Super Resolution (https://arxiv.org/abs/2501.08819)
Comments:
          To appear in WACV 2025. Code is available at: this https URL

- **What's New**: 최근 확산 기반의 블라인드 슈퍼 해상도(SR) 방법들이 고해상도 이미지를 생성하는 데 뛰어난 능력을 보여주었지만, 세부사항이 종종 충실도(fidelity)를 희생하면서 얻어졌습니다. 본 논문에서는 경량화가 필요 없는 손상 인식 모델(degradation-aware models)을 제안하며, 이는 기존의 확산 가이드(diffusion guidance) 프레임워크와 통합될 수 있습니다. 또한, 성능을 더욱 향상시키기 위해 입력 교란(input perturbation)과 가이드 스칼라(guidance scalar)라는 두 가지 새로운 기술을 제안합니다.

- **Technical Details**: 우리는 DIV2K의 미지의 손상 데이터셋, CelebA-HQ 데이터셋, 그리고 ImageNet 데이터셋의 세 가지 데이터셋에서 우리의 방법을 평가했습니다. 이를 위해 LR-HR(Low-Resolution to High-Resolution) 페어 데이터를 생성하고, DIV2K의 검증 세트로부터 랜덤하게 1K의 HR 패치와 해당하는 LR 패치를 잘라내어 DIV2K-Val로 명명했습니다. 모든 실험에서 DDPM(Denoising Diffusion Probabilistic Models)을 사용하고, 가이드 스칼라는 0.3으로 설정했습니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 기존의 최첨단 방법들보다 더 우수한 성능을 보여주었습니다. PSNR, SSIM, LPIPS, DISTS 및 FID와 같은 다양한 평가 메트릭으로 비교한 결과, 우리 모델은 특히 LPIPS, DISTS, 그리고 FID에서 최상의 점수를 달성했습니다. 이러한 결과는 우리의 방법이 기존의 생성 기반 블라인드 SR 방법들보다 이미지 품질을 효과적으로 개선할 수 있음을 시사합니다.



### $\texttt{InfoHier}$: Hierarchical Information Extraction via Encoding and Embedding (https://arxiv.org/abs/2501.08717)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문은 Self-Supervised Learning(SSL)과 Hierarchical Clustering(HC)을 통합한 InfoHier라는 새로운 프레임워크를 제안합니다. 기존 SSL 방법은 비층적 구조에 집중하며, 다차원 데이터의 복잡한 관계를 반영하지 못했습니다. HC는 계층적 데이터를 이해하는 데 유리하지만, 엄격한 유사도 기반 메트릭에 의존해 한계가 있었습니다.

- **Technical Details**: InfoHier는 SSL을 통해 적응형 표현을 제공하고, HC가 복잡한 패턴을 캐치할 수 있도록 도와줍니다. 이를 위해 각 데이터 포인트에 대해 루트 이진 트리 형태의 계층적 구조를 추출하며, 이는 잠재 표현(latent representation)을 통해 이루어집니다. 이 과정에서 Dasgupta 손실을 사용하여 클러스터 성능을 향상시키고, 대조 손실(contrastive loss)을 결합하여 Encoder 네트워크를 더욱 견고하게 조정합니다.

- **Performance Highlights**: InfoHier는 클러스터링과 표현 학습 모두에서 성능을 향상시킬 수 있는 가능성을 지닙니다. 이번 연구에서 제안하는 방법은 복잡한 데이터셋에서 정보 계층 구조를 반영한 보다 효율적인 분석을 가능하게 하여 정보 검색 및 데이터 관리를 효과적으로 수행할 수 있습니다. 이 연구는 대규모 무라벨 데이터에 대한 정보 추출의 현재 한계를 극복하는 데 기여할 것입니다.



### GS-LIVO: Real-Time LiDAR, Inertial, and Visual Multi-sensor Fused Odometry with Gaussian Mapping (https://arxiv.org/abs/2501.08672)
- **What's New**: 최근 3D Gaussian splatting (3D-GS) 기술이 새로운 장면 표현 방식으로 주목받고 있습니다. 기존의 비전 기반 3D-GS 방법은 수동으로 설계된 휴리스틱에 의존하여 점 구름 밀도를 높이고, 차폐(occlusion) 처리 및 높은 GPU 메모리 소비 문제에 직면해 있습니다. 이 연구에서는 LiDAR(Inertial-Visual) 센서를 통합하여 실시간 Gaussian 기반의 SLAM 시스템을 제안합니다.

- **Technical Details**: 제안된 시스템은 글로벌 Gaussian 맵과 슬라이딩 윈도우를 포함한 Gaussians의 구조로 되어 있으며, IESKF 기반의 궤적 추적 기술을 활용합니다. 글로벌 Gaussian 맵은 재귀적 옥트리 구조로 해시 인덱스 된 복셀을 포함하여 공간 효율성을 높이고 다양한 세부 수준(LoD)에 적응할 수 있습니다. Gaussian 맵은 다중 센서 융합을 통해 초기화되며, 사진 기하학적 경량 그래디언트를 통해 최적화됩니다.

- **Performance Highlights**: 이 시스템은 슬라이딩 윈도우 내에서 맵을 최적화하여 GPU 계산과 메모리 소비를 현저히 줄입니다. 실시간으로 업데이트하고 렌더링할 수 있는 기능을 강조하며, NVIDIA Jetson Orin NX 플랫폼에서 자원 제약이 있는 임베디드 시스템에 구현되었습니다. 실제 실험 결과, 우리의 방법은 메모리 사용량을 상당히 줄이고 Gaussians의 최적화를 가속화하여 높은 렌더링 품질을 유지하면서 경쟁력 있는 궤적 추적 정확도를 달성했습니다.



### TimeFlow: Longitudinal Brain Image Registration and Aging Progression Analysis (https://arxiv.org/abs/2501.08667)
- **What's New**: 이번 연구에서는 TimeFlow라는 새로운 프레임워크를 제안합니다. TimeFlow는 기존의 긴 데이터 시퀀스나 시간 연속성의 요구 없이도 정확한 장기 뇌 MRI 등록 및 미래 이미지 예측을 가능하게 합니다. 이는 신경퇴행성 질환 분석과 건강한 노화 연구에 기여할 수 있는 혁신적인 접근 방식입니다.

- **Technical Details**: TimeFlow는 U-Net 아키텍처를 기반으로 하며, 확산 모델의 영감을 받은 시간 조건화(Temporal Conditioning)를 적용합니다. 이 프레임워크는 생물학적 노화를 반영하는 시간 파라미터에 조건화된 흐름 장(field)을 생성하여, 두 개의 이미지만으로도 미래의 시간을 예측하고 장기 등록을 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, TimeFlow는 기존의 최첨단 방법들과 비교했을 때 미래 시간점 예측 및 등록 정확성에서 우수한 성능을 보였습니다. 또한, 이 방법은 병적 노화와 건강한 노화를 효과적으로 구분할 수 있어, 비계층화된 데이터로도 뇌 노화 및 만성 질환에 대한 정확하고 효율적인 분석을 지원합니다.



### Product of Gaussian Mixture Diffusion Model for non-linear MRI Inversion (https://arxiv.org/abs/2501.08662)
- **What's New**: 본 논문에서는 최근의 확산 모델(Diffusion models)이 자기공명영상(MRI) 재구성에서 뛰어난 성능을 보이는 점에 주목하였습니다. 다수의 파라미터로 구성된 기존의 블랙박스(blakc-box) 네트워크가 해석 가능성 및 재구성 속도를 저해하는 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 경량(lean)하고 파라미터 효율적(parameter-efficient)이며, 해석 가능한 가우시안 혼합 모델을 이미지 프라이어(image prior)로 사용하여 이미지와 코일 민감도(coil sensitivities)를 공동으로 재구성합니다. 이 과정에서 고전적인 매끄러움 프라이어(smoothness priors)를 코일 민감도에 적용했습니다.

- **Performance Highlights**: 제안된 방법은 기존의 고전적인 변이학적 패널티(total variation)와 유사한 수준의 성능을 보여주며, 빠른 추론(fast inference)과 샘플링 경로(sampling trajectories)에 대한 강인성(robustness)을 입증하였습니다. 또한, 확률적(formulation) 구성을 통해 후방 기대값(posterior expectation)과 화소별 분산(pixel-wise variance) 계산이 가능해졌습니다.



### Self-Organizing Edge Computing Distribution Framework for Visual SLAM (https://arxiv.org/abs/2501.08629)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문에서는 이동 로봇을 위한 에지 지원 분산 SLAM(Distributed SLAM) 프레임워크를 제안합니다. 이 프레임워크는 특정 서버에 의존하지 않고, 네트워크 상의 여러 장치 간에 SLAM 실행을 자율적으로 분산하고 계층화합니다. 이를 통해 기존 SLAM 시스템의 정확성과 효율성을 유지하면서도 네트워크와 장치 실패에 대한 복원력을 강화했습니다. 또한, 모노큘러 ORB SLAM3를 기반으로 구현하여 실험을 진행했습니다.

- **Technical Details**: 제안된 구조는 세 가지 계층으로 구성되어 있으며, 장치에 구애받지 않습니다. 코어 계층은 모노큘러 ORB SLAM3를 포함하고, 분산 계층은 SLAM 실행을 조정하며 통신 계층은 ROS2와 FastDDS 미들웨어를 사용하여 다른 네트워크 노드와 상호작용합니다. 시스템의 상태 관리를 통해 최종 일관성을 보장하고, 다중 노드 또는 단일 노드 환경 모두에서 SLAM을 실행할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 ATE(평균 궤적 오차)와 CPU 사용량을 기준으로 ORB SLAM3와 유사한 성능을 보여줍니다. 또한, 모든 실험에서 네트워크를 통한 자원 분배가 가능함을 입증했습니다. 끝으로, 단일 장치 구성일 경우에는 독립적으로 실행할 수 있는 능력도 갖추고 있어 다양한 상황에 적응할 수 있는 장점을 보여줍니다.



### Image-to-Force Estimation for Soft Tissue Interaction in Robotic-Assisted Surgery Using Structured Ligh (https://arxiv.org/abs/2501.08593)
- **What's New**: 최근 연구에서는 Minimally Invasive Surgical (MIS) 로봇에 있어 정확한 haptic 인식이 중요하다고 강조하고 있습니다. 기존 시스템의 제한된 공간으로 인해 센서를 사용한 직접적인 힘 측정이 어려웠지만, 본 논문에서는 3D 재구성을 통한 효과적인 비전 기반 힘 추정 시스템을 제안합니다. 이를 통해 소프트 티슈와의 상호작용 시 안전성을 높일 수 있습니다.

- **Technical Details**: 본 연구는 One-Shot 구조광 투사 및 이미지-힘 신경망 처리 방식을 통해 소프트 티슈의 3D 점군(3D point clouds)을 재구성하는 방법론을 소개합니다. 표면의 유형이 부족한 티슈에 대해서도 효과적으로 적용할 수 있으며, 이는 PointNet 기반의 힘 추정 방법을 수정하여 복잡한 기계적 성질을 대표하는 데 뛰어난 성능을 보입니다. 또한, 실험 결과를 통해 제안된 방법의 유효성이 입증되었습니다.

- **Performance Highlights**: 실험에서 서로 다른 강성을 가진 세 가지 실리콘 재료에 대해 수치적 힘 상호작용 실험을 진행하였습니다. 이를 통해 제안된 비전 기반 힘 추정 시스템이 전통적인 방법보다 더 높은 정확도와 일반화 능력을 갖추었음을 확인하였습니다. 마지막으로 본 연구는 Toumai 복강경 수술 로봇 플랫폼에서 적용 가능성을 시연하며 실용성을 더욱 보강합니다.



### A Systematic Review of Machine Learning Methods for Multimodal EEG Data in Clinical Application (https://arxiv.org/abs/2501.08585)
Comments:
          This paper includes 4 figures, 6 tables, and totals 18 pages

- **What's New**: 이번 연구에서는 다중 모달 데이터(multimodal data)를 EEG 자료에 통합하여 기계 학습(machine learning)과 딥 러닝(deep learning) 모델의 정확성을 높이는 방법을 탐구했습니다. 여러 임상 응용 분야에서의 EEG 데이터의 새롭고 효과적인 활용을 제시하며, 신경정신적 장애와 같은 복잡한 임상 문제를 해결하는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: 문헌 검색은 PubMed, Web of Science, Google Scholar를 통해 수행되었으며, 총 16개의 연구가 최종적으로 선정되었습니다. 데이터 융합(data fusion)은 신호(signal), 특징(feature), 결정(decision) 수준에서 이루어졌고, 가장 많이 사용된 기계 학습 모델은 서포트 벡터 머신(support vector machines, SVM)과 결정 트리(decision trees)였습니다.

- **Performance Highlights**: 16개의 연구 중 11개에서 다중 모달 EEG 데이터를 사용했을 때 모델의 정확도가 향상되었다고 보고되었습니다. 이 리뷰는 다중 모달 EEG 기반 기계 학습 모델이 임상 진단 및 문제 해결에 중요한 가능성을 갖고 있다는 점을 강조합니다.



### GOTLoc: General Outdoor Text-based Localization Using Scene Graph Retrieval with OpenStreetMap (https://arxiv.org/abs/2501.08575)
- **What's New**: GOTLoc은 GPS 신호가 없는 야외 환경에서도 작동할 수 있는 강력한 로컬라이제이션 방법을 제안합니다. 기존의 텍스트 기반 로컬라이제이션 연구들은 일반적으로 맵을 포인트 클라우드(point cloud)로 표현했지만, GOTLoc은 씬 그래프(scene graph)를 활용하여 공간 정보를 저장하며 로봇이 큰 양의 맵 데이터를 처리할 수 있도록 지원합니다. 또한, OpenStreetMap와 같은 공공 데이터 소스를 활용하여 별도로 커스텀 맵 데이터를 만들 필요성을 없앴습니다.

- **Technical Details**: GOTLoc은 씬 그래프를 통해 맵 데이터의 저장 요구 사항을 줄이고, 프레임 수가 증가하더라도 알고리즘 속도를 일정하게 유지합니다. 이 방법은 텍스트 임베딩을 사용하여 벡터DB에서 일치하는 후보를 추출한 후, 이를 기반으로 공동 임베딩(joint embedding)을 계산하는 방식으로 전체 처리 시간을 단축시킵니다. 텍스트와 OSM 데이터를 입력으로 받아들여 가장 높은 일치를 보이는 씬 ID를 반환하는 구조입니다.

- **Performance Highlights**: 시티 스케일 환경에서의 성과 평가를 통해 GOTLoc이 기존 포인트 클라우드 기반 접근 방식에 비해 훨씬 적은 저장 공간을 필요로 하며, 몇 초 이내에 전반적인 처리를 완료함을 입증했습니다. GOTLoc의 정확도는 포인트 클라우드 맵에 의존하는 알고리즘과 비교했을 때 유사한 수준의 성능을 보였으며, 실제 로봇 시스템에 유용하게 적용할 수 있음을 확증하였습니다.



### Exploring the Efficacy of Meta-Learning: Unveiling Superior Data Diversity Utilization of MAML Over Pre-training (https://arxiv.org/abs/2501.08506)
- **What's New**: 이 연구는 대규모 비전 모델의 성능을 좌우하는 데이터셋의 다양한 특성 중에서도 특히 데이터 다양성(data diversity)이 모델 성능에 미치는 영향을 탐구합니다. 기존 연구들이 주로 데이터 양(data size)이나 모델의 크기와 복잡성에 집중했던 것과는 달리, 본 연구에서는 데이터 다양성을 중요한 요소로 제시하고 있습니다.

- **Technical Details**: 연구에서는 Task2Vec라는 메트릭을 사용하여 데이터셋의 내재적인 다양성을 측정합니다. 이 메트릭은 작업(task)을 확률 분포로 간주하고, 서로 다른 작업의 Task2Vec 임베딩(task embedding) 간의 평균 거리 평균을 계산하는 방식으로 작동합니다. 우리의 분석은 12개의 인기 있는 시각 데이터셋과 다양한 모델 구성에서 메타 러닝 (meta-learning) 기법을 연구하였습니다.

- **Performance Highlights**: 실험 결과, 데이터셋 다양성과 모델 성능 사이에 양의 상관관계가 존재하는 것으로 나타났습니다. 특히 higher-order MAML 모델은 데이터 다양성과 모델 성능 간의 더 강력한 상관관계를 보였으며, 연구에서 관찰된 R-squared 값은 0.4에 이르렀습니다. 이는 데이터 다양성이 모델의 성능을 향상시키는 중요한 요소임을 시사합니다.



### Automotive Elevation Mapping with Interferometric Synthetic Aperture Radar (https://arxiv.org/abs/2501.08495)
Comments:
          9 pages, 6 figures

- **What's New**: 이 연구는 Interferometric Synthetic Aperture Radar (InSAR)를 활용하여 저해상도 레이더 배열을 사용해 3D 공간에서의 정확한 검출 위치 파악을 가능하게 하는 새로운 방법을 제시합니다. 이 기술은 도시 및 농업 환경 모두에 적용될 수 있으며, 안전한 자율 주행을 위한 3D 매핑을 가능하게 합니다. 더불어, 이 작업은 자동차 환경에서의 고도 맵핑을 위한 자동차용 InSAR의 첫 번째 연구로 자리잡고 있습니다.

- **Technical Details**: InSAR는 SAR 이미지에서 생성된 위상 측정을 활용하여 고도 정보를 추출하는 기술입니다. 저비용 밀리미터파(mmWave) 레이더 센서를 이용해, 경량화된 확장성과 정밀함을 제공하는 신호 모델링을 통해 HD 이미지를 생성합니다. 두 개의 레이더 센서를 사용한 간섭법을 통해 고도 정보를 측정하며, 주행 속도에 따라 아지메스 해상도를 최적화함으로써 레이더 시스템의 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 저해상도 레이더 배열을 통해 밀집한 농업 및 도시 환경에서 적합한 3D 매핑이 가능함을 입증하였습니다. 이 연구의 접근 방식은 고도의 정밀성과 민감성을 바탕으로 안전한 자율주행 결정을 위한 높은 신뢰성을 제공합니다. 이러한 기술적 진보는 자율주행차의 지도 작성 프로세스를 혁신적으로 변화시킬 잠재력을 지닙니다.



### RWKV-UNet: Improving UNet with Long-Range Cooperation for Effective Medical Image Segmentation (https://arxiv.org/abs/2501.08458)
- **What's New**: 본 논문에서는 RWKV-UNet이라는 새로운 모델을 제안합니다. 이는 U-Net 아키텍처에 RWKV(수용 임계값 가중치 키 값) 구조를 통합하여 긴 범위 의존성을 캡처하고 맥락 인식을 개선합니다. 이러한 개선은 정확한 의료 이미지 분할에 매우 중요하며, 실험을 통해 다양한 의료 이미지 데이터셋에서 최첨단 성능을 보여주고 있습니다.

- **Technical Details**: RWKV-UNet의 구조는 인코더, 디코더, 스킵 커넥션 및 Cross-Channel Mix (CCM) 모듈로 구성됩니다. 인버티드 잔여 RWKV(IR-RWKV) 블록은 CNN과 RWKV를 결합하여 강력한 인코더를 구축하며, 세부 정보를 개선하기 위해 스페이셜 믹스를 사용하는 방법론이 포함되어 있습니다. 이 모델은 고급 기능 추출을 위한 다중 규모 기능 융합을 통해 성능 향상을 이룹니다.

- **Performance Highlights**: RWKV-UNet은 Synapse, ACDC, BUSI, CVC-ClinicDB 등 다양한 벤치마크 데이터셋에서 SOTA 성능을 달성합니다. 또한 RWKV-UNet-S와 RWKV-UNet-T라는 소형 변형 모델을 제안하여 정확성과 계산 효율성을 균형 있게 조정하였습니다. 이는 넓은 임상 애플리케이션에 적합한 모델로 자리잡을 수 있음을 보여줍니다.



### BiDepth Multimodal Neural Network: Bidirectional Depth Deep Learning Arcitecture for Spatial-Temporal Prediction (https://arxiv.org/abs/2501.08411)
Comments:
          This paper has been submitted to Applied Intelligence for review

- **What's New**: 이 논문은 동적 시스템에서 공간-시간(Spatial-Temporal, ST) 정보를 정확하게 예측하는 데 중점을 둡니다. 제안된 BiDepth Multimodal Neural Network (BDMNN)은 쌍방향 깊이 조정을 사용하여 장기적 계절성(long-term seasonality)과 단기적인 변동(short-term fluctuations)을 모두 이해할 수 있게 합니다. 이는 기존의 통계적 접근 방식 및 전통적인 신경망(neural networks)의 한계를 극복하는 데 도움을 줍니다.

- **Technical Details**: BDMNN은 다양한 시간적 깊이(variable temporal depths)에서의 정보를 효과적으로 통합하며 공간적 맥락(spatial context)을 유지합니다. 이 모델은 장기적인 역사적 분석(comprehensive long-term historical analysis)과 단기적인 새로운 정보에 대한 빠른 반응(responsiveness) 간의 균형을 이루도록 설계되었습니다. 이러한 복잡한 ST 맥락에 적응하여 예측 정확성을 높입니다.

- **Performance Highlights**: 실제 공공 데이터를 사용한 사례 연구에서는 도시 교통 예측(urban traffic prediction)의 평균 제곱 오차(Mean Squared Error)를 12% 줄이는 등 예측 정확성이 크게 향상되었습니다. 또한, 강수 예측(rain precipitation forecasting)의 경우, 최신 기준(line benchmarks)에 비해 15%의 개선을 이루어냈습니다. 이러한 성과는 추가적인 계산 자원(computational resources)을 요구하지 않습니다.



### 3D Gaussian Splatting with Normal Information for Mesh Extraction and Improved Rendering (https://arxiv.org/abs/2501.08370)
Comments:
          ICASSP 2025: Workshop on Generative Data Augmentation for Real-World Signal Processing Applications

- **What's New**: 이번 연구는 Differentiable 3D Gaussian splatting 기법을 통해 복잡한 장면을 2D 뷰의 집합으로 표현하고, 고품질의 실시간 새로운 뷰 합성을 가능하게 하는 접근 방식을 제시합니다. 제안된 방법은 기상 모델에서 유래된 signed distance function의 그래디언트를 활용한 정규화 방식을 통해 렌더링 품질을 높입니다. 이 과정에서 복잡한 형상을 더 정확하게 재구성함으로써, 비디오 생성, 애니메이션, AR-VR 및 게임과 같은 다운스트림 애플리케이션에 essential한 기반을 마련합니다.

- **Technical Details**: 연구에서는 고곡률 및 섬세한 세부 사항이 있는 영역에서 발생할 수 있는 기하학 재구성의 부정확성을 다루기 위해, Gaussian에서 추정된 signed distance function의 그래디언트를 사용하는 새로운 정규화 방법을 도입합니다. 이 방법은 렌더링과 메쉬 재구성을 개선하는 동시에, 고품질 메쉬를 추출하는 것을 중점적으로 다룹니다. 이를 통해 각기 다른 데이터셋(Mip-NeRF360, Tanks and Temples, Deep-Blending)에서 실험을 통해 균형 잡힌 성능을 보임을 입증하고 있습니다.

- **Performance Highlights**: 제안된 접근법은 photorealism(metrics) 평가에서 기존의 메쉬 추출 렌더링 기법에 비해 높은 점수를 기록하며, 메쉬 품질을 저하시키지 않습니다. 이는 복잡한 장면을 더 현실감 있게 렌더링 할 수 있도록 도와줍니다. 데이터셋의 다양한 테스트를 통해 본 연구의 유효성을 확인하였으며, 향후 기술적인 발전 가능성을 암시합니다.



### A Preliminary Survey of Semantic Descriptive Model for Images (https://arxiv.org/abs/2501.08352)
Comments:
          3 pages, 2 figures

- **What's New**: 이번 연구는 고대 중국 회화(Ancient Chinese Paintings, ACP) 분야에서 이미지 설명과 심층 문화 분석을 위한 통합된 프레임워크가 부족하다는 문제를 해결하고자 합니다. 베이징 궁전 박물관(Beijing Palace Museum)의 ACP 소장품을 활용하여 아이코노로기(Iconological) 이론과 새로운 용어 추출 및 매핑 작업 프로세스를 통합한 의미론적 모델(semantic model)을 개발하였습니다. 이 모델은 기초적인 데이터 구조의 개선뿐만 아니라 예술의 지식 조직(knowledge organization)에 기여할 수 있음을 보여주고 있습니다.

- **Technical Details**: 연구에서 개발된 모델은 아이코노로기 이론을 기반으로 하여 고대 중국 회화의 주제 수준에서 데이터를 분석합니다. 새로운 용어 추출(term extraction) 및 매핑(mapping) 프로세스를 통해 ACP의 통합된 이해를 촉진합니다. 이 모델은 기존의 예술 분석 기법과는 다르게 체계적이고 구조적으로 접근할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 모델의 효과성이 강조되며, 고대 중국 회화에 대한 문화적 탐색(cultural exploration)과 예술 관련 지식 조직을 지원할 수 있는 가능성이 제시됩니다. 연구 결과는 ACP 관련 데이터베이스 구축이나 예술 연구자들의 작업에 중요한 기초 자료로 활용될 수 있습니다. 향후 연구는 이 모델을 통해 더 많은 데이터를 수집하고 분석하는 방향으로 나아갈 것입니다.



### High-throughput digital twin framework for predicting neurite deterioration using MetaFormer attention (https://arxiv.org/abs/2501.08334)
Comments:
          17 pages, 8 figures

- **What's New**: 본 논문에서는 신경 발달 장애(NDD, Neurodevelopmental Disorders)와 관련된 신경가소성 저하를 모델링하기 위한 새로운 하이퍼루프 디지털 트윈 프레임워크를 소개합니다. 이 프레임워크는 합성 데이터 생성, 실험 이미지 및 머신러닝(ML) 모델을 통합하여 다양한 신경가소성 저하 패턴을 포착하는 데 유용합니다. 유전적 요인 및 신경생물학적 복잡성을 고려하여, 시뮬레이션과 실험 데이터를 통합하여 연구자들이 실험 결과를 예측할 수 있도록 돕습니다.

- **Technical Details**: 디지털 트윈 프레임워크는 세 가지 모듈로 구성됩니다: 합성 데이터 생성기, 실험 데이터 세트, 그리고 MetaFormer 기반 ML 모델입니다. 합성 데이터 생성기는 이소기하학적 분석(IGA) 기반의 단계 필드 모델을 사용하여 신경가소성 저하를 시뮬레이션합니다. MetaFormer 모델은 공간적 및 시간적 의존성을 효과적으로 캡처하고, 평균 오차를 각각 1.9641%와 6.0339%로 줄여주는 예측 능력을 가지고 있습니다.

- **Performance Highlights**: 본 프레임워크는 실험 및 합성 데이터의 조합을 통해 신경가소성의 동적 변화를 이해하고 예측하는 데 도움이 됩니다. 합성 데이터 세트는 다양한 신경가소성 저하 패턴을 시뮬레이션하여 연구자들에게 풍부한 학습 자료를 제공합니다. 이 연구는 치료 개발을 위한 통찰력을 제공하고, 비용을 절감하며, 실험 설계를 개선하여 NDDs 연구의 진행을 가속화합니다.



New uploads on arXiv(cs.AI)

### AI-RAN: Transforming RAN with AI-driven Computing Infrastructur (https://arxiv.org/abs/2501.09007)
Comments:
          7 pages, 5 figures

- **What's New**: 이번 논문은 인공지능(AI)과 무선 접속 네트워크(RAN)의 융합을 다룬 AI-RAN 개념을 소개합니다. 전통적인 통신 중심 인프라에서 통합된 컴퓨트-커뮤니케이션 플랫폼으로의 변화와 함께 AI-RAN은 미래 네트워크의 성능 요구사항을 충족시키고 자산 활용도를 향상시키는 방안을 제시합니다. RAN의 진화를 통해 AI-RAN의 세 가지 형태를 정의하고, 각 형태의 구현을 설명합니다.

- **Technical Details**: AI-RAN은 RAN과 AI 워크로드를 동일한 인프라에서 통합하여 처리할 수 있는 구조를 가지고 있습니다. 논문에서는 AI-RAN의 통합을 위한 핵심 요구사항 및 가능 요소를 파악하고, 이를 바탕으로 AI-RAN을 실현하기 위한 참조 아키텍처(reference architecture)를 제시합니다. NVIDIA Grace-Hopper GH200 서버를 활용한 개념 증명(proof-of-concept)을 통해 RAN과 AI 워크로드를 동시 처리하는 사례를 보여줍니다.

- **Performance Highlights**: AI-RAN의 도입으로 향후 네트워크의 성능이 개선될 것으로 기대되며, 자원의 효율적인 사용이 가능해집니다. 논문에서는 AI-RAN의 가능성을 실제 구현을 통해 입증하고, 앞으로의 개발 방향에 대한 제안을 통해 앞으로의 연구에 대한 비전을 제시합니다. 이를 통해 업계에서 AI-RAN의 채택을 강화할 수 있는 기초 작업이 될 것입니다.



### Development and Validation of the Provider Documentation Summarization Quality Instrument for Large Language Models (https://arxiv.org/abs/2501.08977)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)으로 생성된 임상 요약을 평가하기 위해 PDSQI-9(Provider Documentation Summarization Quality Instrument)를 개발하였습니다. 기존의 문서화 품질 평가 도구들은 LLM이 생성한 복잡한 텍스트의 특성을 고려하지 않아 문제점을 가지고 있었습니다. PDSQI-9는 실제 EHR(전자 건강 기록) 데이터로부터의 다문서 요약을 바탕으로 한 연구를 통해 검증되었습니다.

- **Technical Details**: 이 연구에서 LLM(예: GPT-4o, Mixtral 8x7b, Llama 3-8b)을 사용하여 다양한 전문 분야의 실제 EHR 데이터를 기반으로 다문서 요약을 생성하였습니다. 평가 방법으로는 Pearson 상관분석, 요인 분석, Cronbach's alpha를 통한 구조적 타당성 검증이 포함되었습니다. 7명의 의사 평가자가 779개의 요약을 평가하며, 8,329개의 질문에 응답하여 높은 평가자 간 신뢰도를 달성했습니다.

- **Performance Highlights**: PDSQI-9는 강력한 내부 일관성을 보여주었고(Cronbach's alpha = 0.879), 높은 평가자 간 신뢰도(ICC = 0.867)를 기록하며 구조적 타당성을 입증했습니다. 요인 분석 결과는 4가지 요인(조직화, 명확성, 정확성, 유용성)이 전체 분산의 58%를 설명하는 것으로 나타났습니다. 이 도구는 LLM으로 생성된 임상 요약을 평가하고, 헬스케어 작업 흐름에 LLM을 보다 안전하게 통합하는 데 기여할 수 있습니다.



### Analyzing the Ethical Logic of Six Large Language Models (https://arxiv.org/abs/2501.08951)
- **What's New**: 본 연구는 OpenAI GPT-4o, Meta LLaMA 3.1, Perplexity, Anthropic Claude 3.5 Sonnet, Google Gemini, Mistral 7B와 같은 여섯 개의 주요 생성적 대형 언어 모델(generative large language models)의 윤리적 추론을 조사합니다. 기존의 정렬(alignment) 연구와는 달리, 모델들이 윤리적 논리를 설명하고 적용하는 방식을 분석합니다.

- **Technical Details**: 이 연구는 세 가지 확립된 윤리 유형론인 결과주의-의무론 분석(consequentialist-deontological analytic), 도덕적 기초 이론(Moral Foundations Theory), 콜버그 도덕 발달 단계 모델(Kohlberg Stages of Moral Development Model)을 통해 진행됩니다. 연구 결과, LLMs는 합리주의적(consequentialist) 강조가 두드러진 유사한 윤리적 논리를 보여 주며, 결정 시 피해 최소화(harm minimization)와 공정성(fairness)을 우선시하는 경향이 있습니다.

- **Performance Highlights**: 모델들은 정교함, 신중함, 자기 인식을 지속적으로 나타내며 도덕 철학(moral philosophy)에서 대학원 수준의 담론에 유사한 윤리적 추론을 제공합니다. 또한, 이 시스템들은 모든 모델이 전형적인 인간의 도덕 논리보다 더 정교한 윤리적 추론을 설명한다고 말하는 점에서 놀라운 일관성을 보여줍니다.



### Leveraging Large Language Models as Knowledge-Driven Agents for Reliable Retrosynthesis Planning (https://arxiv.org/abs/2501.08897)
- **What's New**: 이 연구는 맥로분자(예: 폴리머)의 레트로합성(retrosynthesis) 계획을 완전히 자동화하기 위한 최초의 시도를 소개합니다. 저자들은 대형 언어 모델(LLMs)과 지식 그래프(KGs)를 결합한 에이전트 시스템을 제안하여 화학 반응 정보를 자동으로 검색하고 구조화된 지식 그래프를 생성합니다. 이 시스템은 혁신적인 Multi-branched Reaction Pathway Search (MBRPS) 알고리즘을 도입하여 복잡한 다중 분기 경로 탐색에 중점을 둡니다.

- **Technical Details**: 이 에이전트는 Google Scholar API를 통해 관련 논문 제목을 검색하고 웹 스크래핑을 통해 이를 다운로드합니다. PDF에서 추출된 데이터는 LLMs가 이해하기 쉽게 정리된 형태로 클리닝됩니다. 마지막으로, Memoized Depth-first Search (MDFS) 알고리즘을 사용하여 레트로합성 경로 트리를 구성하며, 이 트리는 시작 화합물으로부터 타겟 화합물까지의 모든 합성 경로를 따릅니다.

- **Performance Highlights**: 프리젠테이션과 데모 비디오에서 시연된 바와 같이, 이 새로운 접근법은 폴리이미드 합성에 적용되어 수백 가지 경로를 갖는 레트로합성 경로 트리를 생성하고, 최적의 경로를 제안합니다. 기존의 경로 및 새로운 경로를 모두 포함해 유망한 결과를 보여주며, 이는 더 넓은 응용 프로그램 가능성을 암시합니다.



### Exploring Task-Level Optimal Prompts for Visual In-Context Learning (https://arxiv.org/abs/2501.08841)
- **What's New**: 이 논문에서는 Visual In-Context Learning (VICL) 접근법의 새로운 가능성을 발견했습니다. 기존의 샘플별 프롬프트(search strategy)는 높은 비용과 과적합(overfitting)의 위험을 동반하는 반면, 기존의 여러 샘플에서 동일한 프롬프트가 최적 성능을 달성한다는 점을 지적하고 있습니다. 이에 따라, 새로운 태스크 레벨 프롬프트(task-level prompting) 접근법을 제안하여 비용을 절감하고 성능을 유지할 수 있음을 보였습니다.

- **Technical Details**: 논문에서 제안하는 두 가지 새로운 프롬프트 검색 전략은 Top-K 프롬프트(selection) 방법과 Greedy 검색(search) 방법입니다. Top-K 방법은 개별 데모의 성능을 측정한 후 가장 우수한 K개의 데모를 선택하여 최종 프롬프트를 생성합니다. Greedy 검색 방법은 각 단계에서 성능을 극대화할 수 있는 가장 좋은 데모를 선택하는 방식으로, 알고리즘의 각 단계에서 지역 최적 해를 찾습니다.

- **Performance Highlights**: 제안된 두 가지 전략은 다양한 다운스트림 태스크에서 효과적인 성과를 보여주었습니다. 특히, 98% 이상의 프롬프트 검색 시간을 절약하면서도 기존 방법들에 비해 6.2% 이상의 상대적 성능 향상을 달성했습니다. 이러한 결과는 태스크 레벨 프롬프트의 효과성과 효율성을 입증하며, 새로운 접근법이 과거의 방법들을 초월할 수 있음을 강조하고 있습니다.



### SAIF: A Comprehensive Framework for Evaluating the Risks of Generative AI in the Public Sector (https://arxiv.org/abs/2501.08814)
Comments:
          6 pages, 2 figures, 1 tables. AI for Public Missions (AIPM) Workshop at the 39th AAAI Conference on Artificial Intelligence (AAAI 2025)

- **What's New**: 이번 연구에서는 공공 부문에서의 생성 AI의 위험 평가를 위한 체계적인 데이터 생성 프레임워크(SAIF)를 제안합니다. 이 프레임워크는 위험을 분해하고 다양한 시나리오를 설계하는 4단계로 구성되어 있으며, 이를 통해 위험을 체계적이고 일관되게 평가할 수 있도록 합니다. 생성 AI의 다중 모달 기능을 포함한 리스크 분류를 확장하여, 공공 서비스에서의 안전하고 책임 있는 통합을 위한 기초를 제공합니다.

- **Technical Details**: 생성 AI의 리스크는 시스템적 및 운영적 리스크, 콘텐츠 안전 리스크, 사회적 리스크, 법적 및 인권 관련 리스크로 구분됩니다. 시스템 리스크는 AI 시스템의 보안 취약점에서 발생하여 개인 정보 유출의 위험이 있으며, 운영적 리스크는 AI가 본래의 용도에서 벗어나 불공정한 결정을 내릴 수 있는 가능성을 포함합니다. 콘텐츠 안전 리스크는 생성된 콘텐츠가 해롭거나 부적절할 때 발생하며, 사회적 리스크는 개인 데이터의 불법적인 수집이나 보관이 우려됩니다.

- **Performance Highlights**: 생성 AI는 공공 부문에서 행정 효율성을 개선하고 복잡한 의사결정을 지원하는 잠재력을 보여주고 있습니다. 예를 들어, 미국의 이민 관련 질문을 처리하는 챗봇과 같은 성공 사례는 공공 서비스 접근성을 향상시킵니다. 그러나 이러한 기술의 통합은 잘못된 정보나 신뢰 저하 등의 리스크를 초래하므로 신중한 평가가 필요합니다.



### Application of Deep Reinforcement Learning to UAV Swarming for Ground Surveillanc (https://arxiv.org/abs/2501.08655)
- **What's New**: 이번 논문에서는 항공 스와프(air swarm)의 최신 진행 상황을 정리하고, 고전적 방법과 새로운 강화 학습 기반 접근 방식을 모두 다룹니다. 특히, 심층 강화 학습(deep reinforcement learning)과 다중 에이전트 중앙 집중형 스와프 아키텍처(multi-agent centralized swarm architecture)를 통합한 하이브리드 AI 시스템을 제안합니다. 이 시스템은 특정 지역의 감시(surveillance)에 특화되어 있으며, 보안(security) 및 법 집행(law enforcement) 응용 프로그램을 위한 지상 목표(target) 탐색 및 추적을 수행합니다.

- **Technical Details**: 제안된 시스템은 협력하는 UAV(무인 항공기) 간의 작업을 분배하는 중앙 스와프 제어기(central swarm controller)에 의해 관리됩니다. 각 UAV 에이전트는 스와프 제어기가 제안한 다양한 작업 유형에 맞춰, 서로 협력하는 하위 에이전트(sub-agent)들의 집합에 의해 제어됩니다. 특히, 에이전트의 행동은 근접 정책 최적화(proximal policy optimization, PPO) 알고리즘을 사용하여 훈련되었습니다.

- **Performance Highlights**: 시뮬레이션을 통한 결과에 따르면, 본 시스템은 작업 지역을 효과적으로 탐색하고, 목표를 합리적인 시간 내에 확보하며, 지속적이고 일관된 방식으로 추적할 수 있는 능력을 보여줍니다. 여러 성능 측정(metrics)을 통해 스와프의 성능을 평가하는 기준도 정의되었습니다.



### Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design (https://arxiv.org/abs/2501.08603)
- **What's New**: 최근 Large Language Model (LLM)을 기반으로 한 자동 휴리스틱 설계(AHD) 방법이 등장하였습니다. 기존의 휴리스틱 설계 방식은 도메인 지식이 많이 필요했던 반면, LLM을 활용하면 수동 개입 없이도 고품질의 휴리스틱을 생성할 수 있습니다. 그러나 기존의 방법은 지역 최적점(local optima)에 수렴하는 경향이 있어 문제를 극복하기 위해 Monte Carlo Tree Search (MCTS)를 사용하여 더욱 효과적인 탐색을 제안하고 있습니다.

- **Technical Details**: MCTS-AHD 방법론은 LLM이 생성한 모든 휴리스틱을 트리 구조에 저장하면서 연구합니다. 이 과정에서는 새로운 사고 정렬(thought-alignment) 프로세스와 탐색 감쇠(exploration-decay) 기법을 도입하여 더욱 다양하고 높은 품질의 휴리스틱 생성을 목표로 합니다. 이는 기존의 인구 기반 절차가 가진 탐색의 한계를 극복하며, 더 나은 결과를 제공합니다.

- **Performance Highlights**: 제안된 MCTS-AHD 방법론은 다양한 복잡한 작업에서 significantly higher-quality heuristics를 생성하는 데 성공하였습니다. 실험 결과, 전통적인 인구 기반 접근 방식보다 더 나은 성능을 보였으며, 이는 복잡한 계획 작업 해결에 있어 새로운 가능성을 열어줍니다. 본 연구의 코드는 공개되어 연구자들이 직접 활용할 수 있도록 지원합니다.



### Sound Scene Synthesis at the DCASE 2024 Challeng (https://arxiv.org/abs/2501.08587)
- **What's New**: 본 논문은 DCASE 2024 챌린지의 Task 7인 sound scene synthesis(음향 장면 합성)를 소개합니다. 최근의 generative models(생성 모델)의 발전으로 실제적이고 다양한 오디오 콘텐츠 생성이 가능해졌습니다. 이 챌린지는 서로 다른 사운드 장면 합성 시스템을 비교할 수 있는 표준화된 평가 프레임워크를 제시하며, 객관적 및 주관적 메트릭을 포함하고 있습니다.

- **Technical Details**: 챌린지는 텍스트-투-사운드 생성 작업으로 정의되며, 시스템은 텍스트 설명에 따른 현실적인 환경 오디오를 생성해야 합니다. 각 프롬프트는 "Foreground with Background in the background" 형식을 따라 주요 소리 원천과 음향적 맥락을 구분합니다. 생성된 오디오는 4초 길이의 16비트 모노 오디오 클립으로, 32 kHz 샘플링 레이트를 사용하며 음악이나 이해 가능한 음성은 금지됩니다.

- **Performance Highlights**: 평가는 Fréchet Audio Distance (FAD) 지표를 사용하여 이루어지며, 주관적 평가에서는 Foreground Fit, Background Fit, Overall Audio Quality와 같은 세 가지 측면이 0-10 스케일로 평가되었습니다. 제출된 네 개의 시스템에 대한 평가 결과, 음향 기술자 레퍼런스와 제출된 시스템 간에는 36%의 성능 차이가 나 있으며, 이는 합성 오디오 품질 개선의 필요성을 강조합니다.



### Evaluating SAT and SMT Solvers on Large-Scale Sudoku Puzzles (https://arxiv.org/abs/2501.08569)
- **What's New**: 최근 SMT (Satisfiability Modulo Theories) 솔버의 발전은 제약 만족 문제를 해결하는 접근 방식에 혁신을 가져왔습니다. 본 연구에서는 Z3, CVC5, DPLL(T)와 같은 최신 SMT 솔버의 성능을 전통적인 SAT 솔버인 DPLL 알고리즘과 비교하였습니다. 개선된 스도쿠 생성기를 통해 생성된 다양한 난이도의 25x25 스도쿠 퍼즐을 벤치마킹함으로써, SMT 솔버가 기존 SAT 솔버에 비해 매우 뛰어난 성능을 보임을 확인하였습니다.

- **Technical Details**: 이 연구에서는 25x25 스도쿠 퍼즐을 인코딩하고 해결하기 위한 기반 구조를 제공하였습니다. Input Validation, Propagation Statistics와 같은 주요 구성 요소들은 솔버들이 입력을 검증하고 성능에 대한 통계를 유지함을 보장합니다. 또한, DPLL 알고리즘을 활용해 Sudoku 퍼즐을 CNF (Conjunctive Normal Form)로 인코딩하고, PySAT 파이썬 패키지를 통해 체계적인 해결 방식을 적용했습니다.

- **Performance Highlights**: 결과적으로, DPLL(T) 및 Z3 솔서는 고급 이론적 추론과 인코딩 기법을 통해 전통적인 SAT 솔버보다 효율적이고 확장성 있는 문제 해결 능력을 보여주었습니다. 이 연구를 통해 스마트한 SMT 솔버의 효용성이 더욱 강조되었으며, 실제 애플리케이션에서 발생하는 대규모 제약 만족 문제를 처리하는 데 큰 장점이 있음을 입증하였습니다.



### DualOpt: A Dual Divide-and-Optimize Algorithm for the Large-scale Traveling Salesman Problem (https://arxiv.org/abs/2501.08565)
Comments:
          Accepted by AAAI-25, February 2025

- **What's New**: 본 논문은 대규모 여행 판매원 문제(TSP)를 해결하기 위한 이중 분할 및 최적화 알고리즘(DualOpt)을 제안합니다. DualOpt는 두 가지 상보적 전략을 결합하여 해결 품질과 계산 효율성을 향상시킵니다. 첫 번째 전략은 격자 기반의 분할 정복(divide-and-conquer) 절차로, TSP를 더 작은 하위 문제로 나누고 이를 병렬로 해결합니다.

- **Technical Details**: DualOpt는 각 지점을 M×M 격자로 분할하고, 각 격자 내에서 LKH3 솔버를 이용하여 해결한 후, 경로를 부분 경로와 노드로 분해하는 엣지 브레이킹 전략을 도입합니다. 이후 경로를 비겹치는 하위 경로로 나누어 각각을 신경망 솔버(neural solver)로 최적화하여 최종 경로를 개선합니다. 이 두 가지 절차는 TSP의 계산 복잡성을 감소시키고 더 나은 솔루션을 제공하는 데 기여합니다.

- **Performance Highlights**: 광범위한 실험 결과, DualOpt는 100,000개의 노드를 포함한 대규모 사례에서 LKH3보다 104배 빠른 속도로 1.40%의 개선 차이를 달성하며, 10개의 최첨단 알고리즘과 비교하여 경쟁력 있는 결과를 얻었습니다. TSPLIB 기준에서 강력한 일반화를 보여주며, 다양한 실제 TSP 응용 프로그램을 처리할 수 있는 능력을 입증했습니다.



### ANSR-DT: An Adaptive Neuro-Symbolic Learning and Reasoning Framework for Digital Twins (https://arxiv.org/abs/2501.08561)
- **What's New**: 이 논문에서는 디지털 트윈 기술을 위한 적응형 신경-상징적 학습 프레임워크인 ANSR-DT를 제안합니다. ANSR-DT는 패턴 인식 알고리즘과 강화 학습, 상징적 추론을 결합하여 실시간 학습과 적응형 지능을 가능하게 합니다. 이 프레임워크는 인간-기계 협업을 필요로 하는 응용 프로그램에서 보다 나은 의사 결정을 지원하며, 기존의 최첨단 방법들과 비교해 의사 결정의 정확성, 신뢰성 및 해석 가능성에서 유의미한 개선을 경험했습니다.

- **Technical Details**: ANSR-DT 프레임워크는 Proximal Policy Optimization(PPO) 알고리즘을 활용하여 CNN-LSTM 및 주의(attention) 기술과 연계하여 상징적 추론의 논리적 명확성을 확보합니다. 이 프레임워크의 구조는 물리적 레이어, 처리 레이어, 적응 레이어로 구성되어 있습니다. 센서 데이터가 시스템을 통해 흐르고 의사 결정을 내리는 과정에서 확보된 해석 가능한 결과들은 CNN-LSTM 알고리즘을 통해 생성됩니다.

- **Performance Highlights**: ANSR-DT는 경쟁 프레임워크와 비교하여 실질적인 성능 향상을 보여줍니다. 특히, 다양한 산업 환경에서 실시간 적응 및 지속적인 학습을 보장하여 사용자 선호와 환경 변화에 능동적으로 대응할 수 있도록 돕습니다. 또한, 기호적 추론을 통한 알고리즘 개발이 가능하여 더 나은 운영과 성과를 위한 논리적인 결과를 제공합니다.



### Reinforcement Learning-Enhanced Procedural Generation for Dynamic Narrative-Driven AR Experiences (https://arxiv.org/abs/2501.08552)
Comments:
          Number of pages: 13, Number of figures: 4. Accepted for presentation at GRAPP 2025 - 20th International Conference on Computer Graphics Theory and Applications (for additional details on the conference visit this https URL). Disclaimer: This preprint may differ from the final version published in the conference proceedings

- **What's New**: 이 연구는 아르 AR 환경을 위한 강화 학습 기반의 Wave Function Collapse (WFC) 프레임워크를 제안합니다. 이 방식은 특정 환경 규칙과 동적 타일 가중치 조정을 통해 맵을 생성하여, 게임 플레이 요구에 맞춰 맵의 응답성을 향상시킵니다. 이를 통해 만들어진 맵은 내러티브 기반 AR 게임에 적합할 뿐만 아니라, 교육 및 시뮬레이션 훈련 등 다양한 분야에도 적용 가능성이 있습니다.

- **Technical Details**: 제안된 WFC 알고리즘은 도시, 사막, 숲의 세 가지 독특한 생태계를 포함하여 맵을 생성합니다. 각 생태계는 고유한 레이아웃과 예술적 스타일을 가지며, RL을 통해 최적화된 생태계 일관성과 경로 레이아웃을 보장합니다. 이 알고리즘은 셀과 타일의 개념을 사용하여 그리드 기반의 3D 맵을 동적으로 생성하며, 실시간으로 던전 마스터가 이 맵을 조정할 수 있도록 지원합니다.

- **Performance Highlights**: 비교 평가 및 사용자 연구를 통해 제안된 프레임워크는 맵 품질에서 우수한 성과를 보였으며, 몰입감 있는 경험을 제공합니다. 이러한 특성 덕분에 내러티브 중심의 AR 게임에 적합하게 설계되어 있으며, 기존 방법에 비해 더 높은 사용자 경험을 제공합니다. 또한, 보고된 성능은 교육 및 XR 경험과 같은 더 넓은 응용 분야에서도 큰 가능성을 보여줍니다.



### Active Sampling for Node Attribute Completion on Graphs (https://arxiv.org/abs/2501.08450)
- **What's New**: 본 논문은 노드 속성 복원을 위한 새로운 AcTive Sampling 알고리즘(ATS)을 제안합니다. ATS는 그래프 구조, 표현 유사성 및 학습 편향을 고려하여 각 노드의 대표성(representativeness)과 불확실성(uncertainty)을 측정합니다. 이를 통해 학습 과정에서 중요성이 높은 노드를 서서히 선택하여 모델이 노드 속성을 효과적으로 학습하도록 유도합니다. 기존의 Structure-attribute Transformer(SAT) 모델과 결합될 수 있으며, 노드 속성 완성을 위한 차별화된 방식을 제공합니다.

- **Technical Details**: ATS는 각 노드의 정보를 그래프 구조에 기반하여 대표성과 불확실성을 통해 측정합니다. 노드 선택 과정에서는 Beta 분포에 의해 조절되는 가중치를 사용하여 이 두 속성의 선형 결합을 통해 최종 점수를 산출합니다. 이러한 접근은 모델이 훈련할 때 초기에 높은 대표성을 가진 노드를 우선적으로 고려하고 시간이 지남에 따라 불확실성을 반영할 수 있도록 합니다. 또한, ATS는 SAT와 함께 반복적으로 학습하여 모델이 수렴될 때까지 작동할 수 있습니다.

- **Performance Highlights**: 네 개의 공공 벤치마크 데이터 세트와 두 개의 다운스트림 작업에서의 실험 결과, ATS는 노드 속성 완성 모델이 더 나은 최적값에 도달하도록 돕고, 더 높은 품질의 노드 속성을 복원하여 노드 분류 및 프로필 작업에 긍정적인 영향을 미친다는 사실이 입증되었습니다. ATS는 다양한 기본 모델에 유연하게 적용될 수 있으며 작업 효율성을 향상시키는 데 기여할 수 있습니다.



### How Do Generative Models Draw a Software Engineer? A Case Study on Stable Diffusion Bias (https://arxiv.org/abs/2501.09014)
- **What's New**: 본 논문에서는 소프트웨어 공학(Software Engineering, SE) 관련 작업에 대해 세 가지 버전의 Stable Diffusion 모델이 생성한 이미지의 성별 및 인종 편향을 심층적으로 분석했습니다. 이전 연구들에서 SE 커뮤니티 내 성별 및 인종 불균형이 존재하는 것으로 나타났으므로, 이러한 모델을 사용할 때 주의가 필요하다는 점을 강조합니다. 연구 결과 모든 모델이 소프트웨어 엔지니어를 표현할 때 남성을 지배적으로 나타내며 여성을 심각하게 저대표하는 경향이 있음을 보여주었습니다.

- **Technical Details**: Stable Diffusion(현재 SD 2, SD XL, SD 3) 모델은 텍스트 프롬프트를 기반으로 이미지를 생성하는 방법을 사용하며, 이 연구에서는 56개의 소프트웨어 관련 작업을 다뤘습니다. 6,720개의 이미지를 생성하기 위해 각 모델에 두 가지 서로 다른 프롬프트 스타일을 입력했습니다. 결과적으로, SD 모델들은 '소프트웨어 엔지니어' 키워드를 포함할 때 남성 편향이 강화되는 경향을 보였습니다.

- **Performance Highlights**: 연구에서 관찰된 주요 결과는 SD 2와 SD XL이 백인 이미지를 우선적으로 생성하는 반면, SD 3는 아시아인 이미지를 다소 더 많이 나타낸다는 것입니다. 하지만 모든 모델은 흑인 및 아랍 인물의 이미지 생성에서는 심각하게 저대표하는 문제를 가지고 있습니다. 따라서 이러한 편향을 저감할 수 있는 방법에 대한 후속 연구가 필요하다고 강조합니다.



### Multimodal LLMs Can Reason about Aesthetics in Zero-Sho (https://arxiv.org/abs/2501.09012)
Comments:
          WIP, Homepage this https URL

- **What's New**: 본 연구에서는 다중모드 LLM(MLLM)의 추론 능력을 활용하여 예술 작품의 미적 평가를 수행하는 최초의 연구를 제시합니다. 이를 위해 MM-StyleBench라는 새로운 고품질 데이터셋을 구축하고, 인간 선호도를 수학적으로 모델링하는 방법을 개발하였습니다. 실험 결과, MLLM이 예술 평가에서 내재적인 환각 문제를 겪으며, ArtCoT라는 방법을 제안하여 MLLM의 미적 추론 능력을 향상할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 MM-StyleBench라는 대규모 주석 데이터셋을 통해 다양한 콘텐츠와 스타일 인스턴스를 평가하였습니다. MLLM의 응답과 인간 선호도의 상관관계를 분석하며, MLLM의 출력이 인간의 선호와 일치하지 않는 주된 문제를 확인했습니다. ArtCoT는 예술 평가의 명확한 하위 과제를 정의하여 환각을 줄이고 MLLM의 추론 능력을 향상시키는 방법입니다.

- **Performance Highlights**: ArtCoT 방법을 적용한 결과, MLLM의 미적 정렬이 일관되게 향상되었음을 보여주었습니다. 특히, 예술 특정 작업의 분해가 MLLM의 추론 능력을 촉진하고 더 객관적인 사고 과정을 이끌어내는 데 기여했습니다. 연구의 결과는 MLLM이 예술 평가 작업에 어떻게 적용될 수 있는지에 대한 귀중한 통찰을 제공하며, 강화 학습을 통한 스타일 전송 및 이미지 생성과 같은 다양한 응용 분야에 도움이 될 수 있습니다.



### Personality Modeling for Persuasion of Misinformation using AI Agen (https://arxiv.org/abs/2501.08985)
- **What's New**: 이 연구는 개인의 성격 특성이 잘못된 정보에 대한 취약성과 전파에 미치는 영향을 이해하기 위해 혁신적인 에이전트 기반 모델링 접근법을 사용했습니다. 여섯 가지 AI 에이전트가 서로 다른 다섯 가지 성격 차원을 시뮬레이트하며, 실제와 유사한 상호작용을 통해 동적인 잘못된 정보 논의를 분석했습니다. 이 연구의 주요 발견은 분석적인 성격 특성이 증거 기반 논의의 효과를 높이며, 비공격적인 설득 전략이 놀라운 성공을 거둘 수 있다는 것입니다.

- **Technical Details**: 연구에서는 Big Five 성격 특성 모델에서 추출된 성격 차원을 반영한 여섯 개의 특성 에이전트를 설계했습니다. 이 에이전트들은 AgentScope 프레임워크를 통해 여섯 가지 잘못된 정보 주제에 대해 상호작용하며, 각 에이전트는 성격 프로필에 따라 정보의 설득 효과성과 저항 능력을 평가했습니다. 전체적으로 90개의 개별 상호작용이 발생했으며, 에이전트의 결정 과정은 지정된 성격 프로필에 의해 통제되었습니다.

- **Performance Highlights**: 에이전트 간의 상호작용 분석 결과, 성격 프로필과 설득 효과 간에 유의미한 상관 관계가 드러났습니다. 특히 비판적이고 도전적인 특성을 지닌 에이전트 4는 HIV 관련 잘못된 정보 논의에서 59.4%의 성공률을 기록했습니다. 이 결과는 성격 특성이 잘못된 정보에 대한 저항 및 설득에 미치는 복잡한 영향을 강조하며, 개인 고유의 감정적 연결과 신뢰 구축을 우선시하는 효과적인 잘못된 정보 방지 전략의 필요성을 제시합니다.



### Trusted Machine Learning Models Unlock Private Inference for Problems Currently Infeasible with Cryptography (https://arxiv.org/abs/2501.08970)
- **What's New**: 최근 기계 학습의 발전이 개인 정보를 보호하는 새로운 패러다임을 가능하게 만들었습니다. 이 연구에서는 Trusted Capable Model Environments (TCMEs)가 신뢰할 수 있는 제3자의 역할을 할 수 있는 기계 학습 모델을 제안합니다. TCMEs는 입력 및 출력 제약 조건 하에서 작동하는 모델과의 상호작용을 통해, 민감한 데이터를 보호하면서도 안전한 계산을 수행할 수 있도록 합니다.

- **Technical Details**: TCMEs의 주요 속성은 상태가 없음(statelessness), 정보 흐름 제어(explicit information flow control), 및 검증 가능성(verifiability)입니다. 이들 속성은 모델이 이전 상호작용에 대한 상태를 유지하지 않도록 하고, 신뢰성을 보장할 수 있도록 하는 데 필수적입니다. 특히, TCME 환경에서는 기계 학습 모델이 주어진 입력을 기반으로 올바른 출력을 생성하고, 모든 참여자가 합의한 정보 흐름 정책을 준수해야 합니다.

- **Performance Highlights**: TCME는 기존의 암호화 솔루션이 비현실적인 문제를 해결할 수 있는 대안을 제공합니다. 연구에서는 TCME가 단순한 암호 학습 문제를 포함한 여러 사용 사례를 소개하며, 민감한 데이터가 모델에 제공되더라도 이를 기록하거나 노출하지 않도록 설계되었습니다. TCMEs는 기계 학습 모델의 능력을 활용하여 안전한 계산을 수행하게 하며, 결국 개인 정보 보호와 계산 효율성을 동시에 달성하는 것을 목표로 합니다.



### An analysis of data variation and bias in image-based dermatological datasets for machine learning classification (https://arxiv.org/abs/2501.08962)
Comments:
          10 pages, 1 figure

- **What's New**: 이번 연구는 피부암 진단을 위해 실무 환경에서 사용되는 임상 이미지를 대상으로 한 새로운 데이터 분포 분석을 제안한다. 기존의 dermoscopic 데이터셋으로 훈련된 모델들이 임상 환경에서 효과적으로 일반화되지 못하는 문제에 주목하고, 이러한 챌린지를 해결하기 위한 접근법을 모색한다. 연구는 임상 이미지에서 발생하는 데이터 불균형과 잡음을 해결하기 위해 transfer learning을 고려한다.

- **Technical Details**: 연구는 다양한 deep learning 아키텍처를 활용하여 실제 임상 및 dermoscopic 이미지의 차이를 분석한다. 실험에 사용된 모델 아키텍처에는 ConvNext, DenseNet, ResNet, EfficientNet 등이 포함된다. 또한, 데이터 손실을 줄이기 위해 augmentations 기술을 적용하여 모델의 훈련 과정에서 발생하는 잡음 및 클래스 불균형의 영향을 완화하고자 하였다.

- **Performance Highlights**: 모델의 임상 분류 성능은 fine-tuning 과정을 통해 향상되었으며, 다양한 데이터셋 구성을 실험하여 임상 시나리오에서의 모델 비교 가능성을 평가하였다. 연구 결과는 데이터 배포의 중요성과 이를 통해 예측 정확성을 강화할 수 있는 방법론에 대한 통찰을 제공한다. 최종적으로, 서로 다른 분포 간의 데이터 결합 방안을 모색하여 모델의 정확도 저하를 줄이는 전략을 제시하였다.



### Kolmogorov-Arnold Networks for Time Series Granger Causality Inferenc (https://arxiv.org/abs/2501.08958)
- **What's New**: 이번 논문에서는 Granger Causality Kolmogorov-Arnold Networks(GCKAN)라는 혁신적인 아키텍처를 소개합니다. GCKAN은 최근에 제안된 Kolmogorov-Arnold Networks(KAN)를 인과 추론(causal inference) 분야로 확장한 모델입니다. 이 모델은 KAN 계층에서 추출된 기본 가중치(base weights)를 활용하고, 희소성 유도 패널티(sparsity-inducing penalty)와 Ridge 정규화(ridge regularization)를 결합하여 시간 시계열로부터 Granger 인과관계를 추론합니다.

- **Technical Details**: GCKAN은 자동 시간 지연 선택(automatic time lag selection)을 가능하게 하여 시계열 데이터에서 Granger 인과관계를 추론하는 데 도움을 줍니다. 또한, 시간 반전된 Granger 인과관계를 활용하는 알고리즘을 제안하여 추론 정확도를 향상시키고자 합니다. 이 알고리즘은 원본 및 시간 반전된 시리즈에서 파생된 예측 및 희소성 유도 손실을 비교하여 더 높은 점수를 가진 인과관계를 자동으로 선택하거나 두 결과를 통합하여 잘못된 연결(spurious connectivity)을 완화합니다.

- **Performance Highlights**: 다양한 Lorenz-96, 유전자 조절 네트워크(gene regulatory networks), fMRI BOLD 신호 및 VAR 데이터 세트를 통해 진행된 광범위한 실험에서, 제안된 모델이 비선형(nonlinear), 고차원(high-dimensional), 제한된 샘플(lemon-sample) 시간 시계열에서의 Granger 인과관력 추론에서 최신 방법들(state-of-the-art methods)에 비해 경쟁력 있는 성능을 달성하는 것으로 나타났습니다.



### Visual WetlandBirds Dataset: Bird Species Identification and Behavior Recognition in Videos (https://arxiv.org/abs/2501.08931)
- **What's New**: 현재의 생물다양성 손실 위기는 동물 모니터링의 필요성을 더욱 부각시키고 있습니다. 이에 따라, 본 연구는 조류 행동 탐지와 종 분류를 위해 특별히 설계된 첫 번째 정밀 비디오 데이터셋을 소개합니다. 이 데이터셋은 178개의 비디오로 구성되어 있으며, 스페인 알리칸테 지역의 습지에서 촬영되었습니다.

- **Technical Details**: 제안된 데이터셋은 13종의 조류가 7가지의 행동을 수행하는 장면을 포함하고 있습니다. 각 비디오는 행동이 발생하는 위치(바운딩 박스)와 함께 프레임 단위로 주석이 달려 있어 기존의 데이터셋과 차별화됩니다. 데이터 수집과 주석 작업은 전문 생태학자와 협력하여 진행되었습니다.

- **Performance Highlights**: 최신 모델을 사용한 기초 성능 결과도 제공되어 조류 행동 인식과 종 분류 작업의 효과성을 평가합니다. 데이터셋은 다양한 행동 클래스에서 샘플 비디오를 포함하고 있으며, 이는 조류 모니터링과 보존 전략 수립에 중요한 기초 자료를 제공합니다.



### Disentangling Exploration of Large Language Models by Optimal Exploitation (https://arxiv.org/abs/2501.08925)
- **What's New**: 이 논문은 탐험(exploration)이 미래의 수익을 증가시킬 수 있도록 정보를 전달하는 것을 목표로 하는 새로운 평가 틀을 제안합니다. 기존 연구들은 주로 탐험과 활용(exploitation) 간의 균형을 중점적으로 다루어왔습니다. 그러나 이 연구는 탐험을 독립적인 목표로 분리하고, 여러 대형 언어 모델(LLM)이 상태 공간(state-space)을 탐색할 때의 성능을 평가합니다.

- **Technical Details**: 탐험 성능을 평가하기 위해, 연구진은 기존 보상의 결여를 탐험과 활용 구성요소로 분해하여 측정하는 방법을 도입했습니다. 이 접근법은 LLM의 탐험 전략을 체계적으로 검토할 수 있는 기반을 제공합니다. 실험 결과, 대부분의 LLM 모델들이 상태 공간을 충분히 탐색하는 데 어려움을 겪고 있으며, 모델 크기와 탐험 성능 사이에 긍정적인 상관관계가 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과는 대부분의 LLM이 독립된 탐험 작업에서 성과를 내는데 한계를 가지며, 이는 탐험과 활용의 트레이드오프를 강조하고 있습니다. 또한, 연구팀은 더 큰 모델이 더 나은 탐험 성능을 보여주는 경향이 있다는 점을 확인했습니다. 이 연구는 LLM의 탐험능력을 평가하고 향상시키기 위한 훌륭한 도구로 작용할 수 있는 분해 방식을 제공합니다.



### Modeling Melt Pool Features and Spatter Using Symbolic Regression and Machine Learning (https://arxiv.org/abs/2501.08922)
- **What's New**: 본 연구에서는 기계 학습(Machine Learning, ML)과 다항식 기호 회귀(Polynomial Symbolic Regression) 모델을 활용하여 레이저 파우더 베드 융합(LPBF) 공정에서 녹는 풀(melt pool)의 변화를 효과적으로 포착하고 통제할 수 있는 프레임워크를 개발했습니다. 이는 품질 관리를 개선하고 결함을 최소화하는 데 도움을 주며, 과거의 데이터셋을 기반으로 한 예측 모델을 통해 인쇄 품질의 일관성을 높이는데 기여하는 기술입니다.

- **Technical Details**: 연구에서는 281개의 프로세스 조건을 위한 데이터셋을 사용하여 녹는 풀의 치수(길이, 너비, 깊이) 및 형상(면적, 부피)과 같은 파라미터를 추출했습니다. 기계 학습 모델을 통해 95% 이상의 높은 결정 계수(R²)를 달성하였으며, 특히 ExtraTree 모델이 가장 높은 R² 값을 기록하였습니다. 로그 변환을 통해 스패터(spatter) 관련 변수의 예측 성능을 향상시킬 수 있었습니다.

- **Performance Highlights**: 이 연구는 LPBF 공정의 품질을 설계 최적화와 결함 예방을 통해 향상시키기 위한 혁신적인 접근 방식을 제시합니다. 특히 기계 학습을 통해 프로세스 조건과 녹는 풀의 차원 및 형상 간의 상관관계를 이해하고 해석 가능한 수학적 표현을 제공하며, 품질 보증 및 불량률 감소에 큰 도움을 줄 수 있습니다. 이러한 연구 결과는 여러 산업 분야에서 기계 학습이 품질 관리에 미치는 영향을 명확하게 드러냅니다.



### Projection Implicit Q-Learning with Support Constraint for Offline Reinforcement Learning (https://arxiv.org/abs/2501.08907)
- **What's New**: Proj-IQL은 Offline Reinforcement Learning에서 발생할 수 있는 OOD(Out-Of-Distribution) 행동으로 인한 과외삽(E extrapolation) 오류를 해결하기 위해 제안된 새로운 알고리즘입니다. 본 알고리즘은 정책 평가(Policy Evaluation)와 정책 개선(Policy Improvement) 단계에서 장기적인 관점에서 효과적인 학습을 구현하기 위해 다단계 적용과 지원 제약(Support Constraint)을 포함하고 있습니다. Proj-IQL은 이론적으로 정책 개선을 보장하고 더 엄격한 기준을 통해 우수한 행동을 보장합니다.

- **Technical Details**: Proj-IQL은 정책 평가 단계에서 벡터 투영(Vector Projection)을 통해 단일 단계 접근법을 다단계 접근법으로 일반화합니다. 이 단계에서 기존의 고정된 보수성 매개변수(Conservatism Parameter)를 대체하여 데이터셋에 특화된 조정 없이도 동작할 수 있도록 합니다. 또한 정책 개선 단계에서는 정책 평가 접근법과 더 잘 조화되는 지원 제약을 도입하여 더욱 효율적인 정책 개선이 이루어집니다.

- **Performance Highlights**: Proj-IQL은 D4RL 벤치마크에서 최첨단(State-of-the-art) 성능을 기록하였으며, 특히 어려운 탐색(구현) 작업에서 뛰어난 성과를 보여주었습니다. 알고리즘은 로코모션 작업과 같은 다양한 Gym-MuJoCo-v2 환경에서 우수한 결과를 도출하며, 기존 알고리즘들과 비교해 더 나은 성능을 입증했습니다. 이러한 결과는 Proj-IQL이 실제 환경에서의 적용 가능성을 더욱 높여주는 연구로 평가됩니다.



### Computing Game Symmetries and Equilibria That Respect Them (https://arxiv.org/abs/2501.08905)
Comments:
          Long and updated version to the published paper in the Proceedings of the 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025). 24 pages, 2 figures, 1 table

- **What's New**: 이번 연구에서는 다중 에이전트 시스템 내의 대칭성을 인식하여 전략적 상호작용을 더욱 간결하게 표현하고 효율적으로 분석 및 해결할 수 있음을 보여줍니다. 대칭성은 게임 이론에서의 균형 선택과 관련된 개념적 함의도 지니고 있습니다. 우리는 정상형 게임(normal-form games)의 고전적인 틀을 사용하여 게임 대칭성을 분석하며, 이를 그래프 자동동형(graph automorphisms)과 연계하여 게임 내 대칭성의 특성을 나타내는 완전성을 증명합니다.

- **Technical Details**: 논문에서는 게임의 대칭성이 일부 선수(player) 또는 행동(actions) 간에 존재할 수 있음을 보여주며, 대칭성을 이용하여 내시 균형(Nash equilibrium)을 계산하는 문제의 계산 복잡성을 검토합니다. 특히, 주어진 대칭성 집합을 존중하는 내시 균형을 찾는 문제는 일반합(general-sum) 게임이나 팀 게임에서 PPAD 및 CLS 완전성을 갖는 것으로 나타났습니다. 이러한 이론적 배경을 바탕으로 다수의 대칭성을 알고 있는 경우 혹은 2인 제로섬 게임(zero-sum game)에서 대칭성을 모르는 경우에 대해 다항식 시간 방법을 제시합니다.

- **Performance Highlights**: 다양한 대칭을 활용하는 기존의 접근방법과 비교했을 때, 제시된 알고리즘은 Nash 균형을 찾는 데 더욱 효율적임을 보입니다. 예를 들어, 적절한 대칭성을 권장하는 전략 프로필을 선택하면 플레이어들이 반복적으로 같은 전략을 사용하여 장기적으로 더 높은 점수를 얻을 수 있습니다. 특히, 다수의 대칭성을 인식하고 적용함으로써 플레이어들은 더 낮은 리워드를 추구하는 것이 일반적이라는 점도 강조됩니다.



### Karatsuba Matrix Multiplication and its Efficient Custom Hardware Implementations (https://arxiv.org/abs/2501.08889)
Comments:
          Accepted for publication in IEEE Transactions on Computers; Associated source code available on github at this https URL

- **What's New**: 이 연구에서는 Karatsuba 알고리즘을 확장하여 정수 행렬 곱셈에 적용하는 새로운 방법을 제안합니다. 기존의 Karatsuba 알고리즘이 갖는 곱셈 복잡도 감소 효과를 유지하면서, 추가적인 덧셈의 복잡도를 줄이는 데 중점을 두었습니다. 이 외에도 Karatsuba 행렬 곱셈(Karatsuba Matrix Multiplication, KMM)을 효율적으로 활용할 수 있는 새로운 하드웨어 아키텍처를 개발하였습니다.

- **Technical Details**: 이 논문에서는 사용된 표기법을 통해 Karatsuba 알고리즘의 복잡성을 정량적으로 분석합니다. 제안된 KMM 알고리즘은 기존의 스칼라 Karatsuba 및 행렬 곱셈 알고리즘에 비해 실제 면적 또는 수행 시간에서 개선을 기대할 수 있습니다. 또한 제안된 하드웨어 아키텍처는 기존의 systolic array(합성 배열)와 전통적인 곱셈기 아키텍처를 기반으로 하여 구현될 수 있습니다.

- **Performance Highlights**: KMM 및 관련 하드웨어 아키텍처는 다른 알고리즘과 비교했을 때 행렬 곱셈 하드웨어에서 성능 개선과 면적 효율성을 제공하는 것으로 입증되었습니다. 이 연구는 기존의 최첨단 기술과 동일한 컴퓨팅 플랫폼에서 구현된 각각의 baseline 설계와 비교하여 KMM 아키텍처의 이점을 평가하였습니다. 이러한 개선된 성능은 정수 행렬 곱셈의 가속에 실질적인 기여를 할 수 있는 가능성을 제시합니다.



### Incrementally Learning Multiple Diverse Data Domains via Multi-Source Dynamic Expansion Mod (https://arxiv.org/abs/2501.08878)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 논문에서는 다양한 데이터 도메인에서의 지속적인 학습(Continual Learning) 문제를 다루기 위해 Multi-Source Dynamic Expansion Model (MSDEM)이라는 새로운 방법론을 소개합니다. 이 모델은 여러 개의 사전 훈련된(backbone) 모델을 기반으로 점진적으로 새로운 전문가를 구축하여 새로운 작업에 적응하게 설계되었습니다. 또한, 동적 확장 가능한 주의 메커니즘과 동적 그래프 가중 라우터를 제안하여 지식 전이를 극대화하고 일반화 성능을 향상시킵니다.

- **Technical Details**: MSDEM은 여러 출처에서 훈련된 백본 모델의 지식을 통합하여 강력한 일반화 표현을 제공하는 것을 목표로 합니다. 특히, 동적 확장 가능한 주의 메커니즘(Dynamic Expandable Attention Mechanism, DEAM)을 통해 여러 백본에서 추출한 표현의 중요성을 동적으로 평가하여 효과적으로 전이학습을 수행합니다. 또한, 동적 그래프 가중 라우터(Dynamic Graph Weight Router, DGWR) 전략을 통해 이전에 학습된 파라미터를 재사용하여 새로운 작업 학습을 돕습니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 방법론이 다양한 복잡한 데이터셋에서 최첨단 성능을 달성했음을 입증하였습니다. MSDEM은 데이터 도메인이 다양할 때에도 뛰어난 일반화 성능을 보이며, 적은 파라미터로도 효과적으로 학습할 수 있음을 보여줍니다. 마지막으로, 논문은 MSDEM 프레임워크와 함께 여러 중요한 기여를 통해 지속적인 학습의 새로운 가능성을 제시하고 있습니다.



### Silent Abandonment in Text-Based Contact Centers: Identifying, Quantifying, and Mitigating its Operational Impacts (https://arxiv.org/abs/2501.08869)
Comments:
          arXiv admin note: text overlap with arXiv:2304.11754

- **What's New**: 이번 논문은 고객 서비스 개선을 위한 새로운 접근법을 제시합니다. 텍스트를 통한 고객 상담에서 발생하는 'silent abandonment'(침묵적 이탈)의 실태를 측정하고 그 영향을 완화할 방법을 모색합니다. 이탈한 고객을 파악하는 데 있어 기존의 불확실성을 해결하기 위한 분류 모델 및 EM 알고리즘을 개발했습니다.

- **Technical Details**: 연구에 따르면, 17개 기업에서 조사한 결과, 고객의 3%에서 70%가 침묵적으로 이탈했으며, 71.3%의 이탈 고객이 이러한 침묵적인 이탈을 선택했습니다. 이로 인해 상담원의 효율성이 3.2% 감소하고 시스템 용량이 15.3% 줄어들며, 연간 상담원당 $5,457의 비용이 발생하는 것으로 나타났습니다. EM 알고리즘을 통해 고객의 인내도를 추정하고 영향을 미치는 요인을 파악했습니다.

- **Performance Highlights**: 고객이 대기하는 동안 메시지를 작성할 수 있도록 허용하면 데이터 손실 문제는 발생하지만, 고객의 인내도가 크게 증가하고 서비스 시간은 줄어드는 긍정적인 효과가 있음을 입증했습니다. 이러한 접근은 고객 이탈을 줄이고 필요한 인력을 감소시키는 결과를 가져왔습니다.



### ARMOR: Shielding Unlearnable Examples against Data Augmentation (https://arxiv.org/abs/2501.08862)
- **What's New**: 이 논문에서는 데이터 증강(data augmentation)이라는 일반적인 데이터 전처리 기술이 개인 정보 보호(data privacy)에 미치는 잠재적인 위협을 밝힙니다. 특히, 데이터 증강이 적용될 경우, 학습 불가능한 예제(unlearnable examples)의 정확도가 21.3%에서 66.1%로 증가할 수 있다는 점을 보여줍니다. 이를 통해 데이터 증강이 개인 데이터 보호 샘플의 유용성을 심각하게 손상시킬 수 있음을 지적합니다.

- **Technical Details**: ARMOR라는 데이터 개인 정보 보호를 위한 방어 프레임워크를 제안합니다. 이 프레임워크는 공격자가 사용하는 데이터 증강 전략에 대한 정보가 없는 상황에서도 효과적으로 작동하도록 설계되었습니다. 비국소 모듈(non-local module)을 활용하여 서그잇 모델(surrogate model)을 구축하고, 각 클래스에 최적화된 데이터 증강 전략을 선택할 수 있는 서그잇 증강 선택 전략을 개발합니다.

- **Performance Highlights**: ARMOR는 4개의 데이터셋(CIFAR-10, CIFAR-100, Mini-ImageNet, VGG-Face)과 5개의 데이터 증강 방법을 사용한 실험에서 뛰어난 성능을 입증합니다. ARMOR는 기존 6가지 최첨단 방어 방법보다 효과적으로 개인 데이터의 학습 불가능성을 유지하여, 증강된 샘플로 훈련된 모델의 정확도를 최대 60%까지 감소시킬 수 있음을 확인했습니다. 또한, 다양한 데이터 증강 방법에 대해서도 견고한 방어 성능을 보여줍니다.



### Digital Phenotyping for Adolescent Mental Health: A Feasibility Study Employing Machine Learning to Predict Mental Health Risk From Active and Passive Smartphone Data (https://arxiv.org/abs/2501.08851)
- **What's New**: 이번 연구는 25세 이전에 대부분 발생하는 정신장애에 대한 예측 모델을 개발하기 위해, 스마트폰 데이터를 통합하여 활용하는 새로운 기계 학습 프레임워크를 사용하였습니다. 이는 청소년의 정신 건강 위험을 일찍 검출할 수 있는 가능성을 제시합니다. 특히, Mindcraft 앱을 통해 비임상 청소년의 다양한 정신장애에 대한 리스크를 예측하고자 하였습니다.

- **Technical Details**: 연구 참가자들은 런던의 세 개 학교에서 모집된 103명이며, 평균 연령은 16.1세입니다. 참가자들은 여러 설문지를 완료하고 14일 동안 Mindcraft 앱을 사용하여 능동적(active) 및 수동적(passive) 데이터를 수집했습니다. 기계 학습 모델은 contrastive pretraining을 통해 사용자 특성을 안정화하고, supervised fine-tuning을 통해 성능을 향상시켰습니다.

- **Performance Highlights**: 능동적 및 수동적 데이터를 통합한 결과, 단일 데이터 소스보다 우수한 성능을 보였습니다. SDQ-High 위험도에 대한 평균 균형 정확도는 0.71, 불면증에 대해서는 0.67, 자살 사고에 대해 0.77, 섭식 장애에 대해서는 0.70을 기록했습니다. 이러한 결과는 고급 기계 학습 기법과 스마트폰 데이터를 통합하여 정신 건강 위험을 예측할 수 있는 가능성을 보여줍니다.



### Graph Counterfactual Explainable AI via Latent Space Traversa (https://arxiv.org/abs/2501.08850)
Comments:
          Published at Northern Lights Deep Learning Conference 2025

- **What's New**: 이번 논문은 깊은 신경망의 예측을 설명하는 데 있어 카운터팩추얼 설명(counterfactual explanations)을 생성하는 새로운 방법을 제안합니다. 특히 구별되는 노드 구조와 그래프 분류기의 연속적 속성을 모두 고려하여 그래프 데이터에 대한 설명 가능성을 높이고자 했습니다. 이러한 방법은 케이스 별로 순열 동치(permutation equivariance) 그래프 변분 오토인코더(variational autoencoder)를 활용하여, 예측 결과를 변경하는 최적의 대안 입력을 찾는 데 중점을 두고 있습니다.

- **Technical Details**: 카운터팩추얼 설명을 생성하기 위해, 그래프 분류기(classifier)의 분류 경계를 횡단하면서 잠재 공간(latent space)을 탐색하는 방법을 채택했습니다. 이 과정에서, 순열 동치 그래프 변분 오토인코더(PEGVAE)를 통해 그래프의 의미론적 잠재 표현(semantic latent representation)을 구축하며, 이는 입력 그래프와 의미상 유사한 그래프를 생성할 수 있게 해줍니다. 이러한 접근 방식은 주어진 그래프의 클래스 라벨을 최소한으로 수정하는 그래프를 반환하는 데 초점을 두고 있습니다.

- **Performance Highlights**: 세 가지 그래프 데이터셋에 대한 실험 결과, 제안된 모델은 기존 방법들(base lines)보다 일관되게 높은 성능을 보여주었습니다. 또한, 성능면에서도 더 강건한 결과를 나타냈으며, 다양한 데이터에 대해 효과적으로 카운터팩추얼 설명을 생성할 수 있는 능력을 입증하였습니다. 이로 인해 AI 시스템의 설명 가능성을 크게 향상시켰다는 점에서 의미가 큽니다.



### RouteNet-Gauss: Hardware-Enhanced Network Modeling with Machine Learning (https://arxiv.org/abs/2501.08848)
Comments:
          13 pages, 11 figures

- **What's New**: 본 논문은 RouteNet-Gauss라는 새로운 접근 방식을 소개합니다. RouteNet-Gauss는 네트워크 테스트베드와 머신러닝(ML) 모델을 통합하여 기존의 Discrete Event Simulation(DES)의 한계를 극복하려고 합니다. 이 모델은 테스트베드를 하드웨어 가속기로 활용하여 훈련 데이터셋을 빠르게 생성하고 실제 환경에 대한 높은 충실도로 네트워크 시나리오를 시뮬레이션합니다. RouteNet-Gauss는 예측 오류를 최대 95%까지 줄이고, 추론 시간에서 488배의 속도 향상을 보여줍니다.

- **Technical Details**: RouteNet-Gauss는 모듈화된 아키텍처로, 네트워크 시나리오의 특징에 따라 동적으로 구성됩니다. 이 시스템은 토폴로지나 라우팅과 같은 다양한 네트워크 구성을 이해하고 일반화할 수 있으며, 훈련 중 보지 못한 네트워크에도 적응할 수 있습니다. 또한, Temporal Aggregated Performance Estimation(TAPE)을 지원하여 시간적 세분성을 조정하고 흐름 성능 지표에서 높은 정확도를 유지합니다. 이 접근 방식은 시뮬레이션의 효율성과 정확성을 개선하는 데 유망한 도구를 제공합니다.

- **Performance Highlights**: 실험 결과, RouteNet-Gauss는 학습 중에 보지 못한 시나리오의 모델링에서 평균 절대 백분율 오차가 2.289%에 불과할 정도로 뛰어난 정확도를 보여줍니다. 이 모델은 특정 최첨단 솔루션에 비해 최대 488배 빠르게 추론할 수 있는 성능을 발휘하고 있습니다. RouteNet-Gauss는 하드웨어 테스트베드를 활용하여 다양한 네트워크 구성을 동적으로 생성하고, 흐름 수준 메트릭스를 제공하여 실용적인 응용에 적합합니다.



### Automatic tuning of communication protocols for vehicular ad hoc networks using metaheuristics (https://arxiv.org/abs/2501.08847)
- **What's New**: 이 논문은 차량 애드혹 네트워크(VANETs)의 새로운 개발 방향을 제시합니다. 차량들은 사전 구축된 인프라 없이도 자발적으로 상호 연결될 수 있으며, 이를 위해 최적의 통신 프로토콜 구성이 필수적입니다. 최적의 QoS (Quality of Service)를 미리 확보하기 위해 파일 전송 프로토콜 구성(FTC)을 최적화하는 문제를 다룹니다.

- **Technical Details**: 이 연구는 파일 전송 프로토콜 구성(FTC)을 최적화하여 전송 시간, 잃어버린 패킷 수, 전송된 데이터 양을 최소화하는 것을 목표로 합니다. 다섯 가지 최첨단 최적화 기법인 Particle Swarm Optimization (PSO), Differential Evolution (DE), Genetic Algorithm (GA), Evolutionary Strategy (ES), Simulated Annealing (SA)을 사용하여 FTC 문제에 접근합니다. 연구에서는 ns-2라는 유명한 VANET 시뮬레이터를 사용하여 도시와 고속도로 시나리오에 대한 테스트 환경을 정의합니다.

- **Performance Highlights**: 실험 결과, PSO가 두 가지 VANET 환경 사례(도시 및 고속도로 시나리오) 모두에서 다른 알고리즘보다 우수한 성능을 보였습니다. PSO는 전송 시간, 패킷 손실, 전송 데이터 양 향상에서 가장 효과적이며, 이러한 결과는 VANET의 실용적인 적용 가능성을 높이는 데 기여합니다.



### ToMATO: Verbalizing the Mental States of Role-Playing LLMs for Benchmarking Theory of Mind (https://arxiv.org/abs/2501.08838)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 입력된 기존 ToM(Theory of Mind) 벤치마크가 현실과 어긋나는 세 가지 측면을 지적하고, ToMATO라는 새로운 벤치마크를 소개합니다. ToMATO는 대화에 기반한 다중 선택 퀴즈(Multiple-choice QA) 형식으로 구성되어 있습니다. 이 벤치마크는 정보 비대칭을 이용한 LLM-LLM 간의 대화를 통해 생성됩니다.

- **Technical Details**: ToMATO는 믿음(belief), 의도(intention), 욕구(desire), 감정(emotion), 그리고 지식(knowledge)의 다섯 가지 범주를 아우르는 1차 및 2차 정신 상태(mental states)를 포착합니다. 각 발화 전에 역할극(role-playing) LLM이 자신의 생각을 구술해야 하는 프롬프팅 방법을 사용하여, 대화 중 캐릭터들의 정신 상태를 평가하는 질문에 대한 답변을 생성합니다. 정보 비대칭을 통해 다양한 정신 상태에 대한 잘못된 믿음(false beliefs)이 생성될 수 있도록 설계되었습니다.

- **Performance Highlights**: 아홉 개의 LLM을 ToMATO에서 평가한 결과, 특히 잘못된 믿음을 이해하는 데 있어 GPT-4o mini조차도 인간 성능에 미치지 못하는 것으로 나타났습니다. 이 연구는 역할극 LLM 간의 정보 비대칭이 잘못된 믿음을 자주 생성하게 함을 보여주며, 다채로운 인격 특성(personality traits)의 반영 또한 효과적으로 이루어지고 있음을 입증합니다.



### MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents (https://arxiv.org/abs/2501.08828)
Comments:
this https URL

- **What's New**: 이번 연구는 Multi-Modal Document Retrieval을 위한 새로운 벤치마크인 MMDocIR을 소개합니다. MMDocIR은 페이지 레벨(page-level)과 레이아웃 레벨(layout-level) 검색의 두 가지 주요 작업으로 구성되어 있으며, 이를 통해 사용자 질문에 대한 더욱 세분화된 답변을 제공할 수 있습니다. 기존의 벤치마크들에서는 미비했던 요소를 보완하여, 문서 내에서의 정확한 검색 성능 평가를 가능하게 합니다.

- **Technical Details**: MMDocIR은 313개의 문서와 1,685개의 질문, 그리고 73,843개의 질문 응답 쌍으로 구성된 훈련 세트를 포함합니다. 본 연구에서는 특히 레이아웃을 정밀하게 표시하기 위한 주석(annotation) 작업을 수행하였으며, 각 페이지에 대한 증거를 포함하는 레이블을 제공합니다. 또한, 비주얼 기반의 검색 시스템과 텍스트 기반 시스템의 성능 차이를 분석하여 비주얼 요소의 중요성을 강조합니다.

- **Performance Highlights**: 엄격한 실험을 통해 비주얼 검색기가 텍스트 검색기보다 상당히 뛰어난 성능을 보인다는 사실을 확인했습니다. 최신 실험 결과는 MMDocIR 훈련 세트가 multi-modal document retrieval 과정에 긍정적인 영향을 미친다는 것을 보여줍니다. 이러한 결과는 비주얼 요소를 통합하는 것이 multi-modal document retrieval를 향상시키는 데 중요한 역할을 한다는 것을 강조합니다.



### IDEA: Image Description Enhanced CLIP-Adapter (https://arxiv.org/abs/2501.08816)
- **What's New**: 이번 논문은 CLIP (Contrastive Language-Image Pre-training)을 기반으로 한 Image Description Enhanced CLIP-Adapter (IDEA) 방법을 제안합니다. 이 방법은 이미지와 텍스트의 상호작용을 활용하여 정밀한 특성을 포착함으로써, 적은 샘플로 이미지 분류 작업을 수행할 수 있도록 돕습니다. 또한, Trainable-IDEA (T-IDEA)를 도입하여 경량의 학습 가능한 컴포넌트를 추가하고, 11개의 데이터셋에서 SOTA (State-Of-The-Art) 성능을 달성했습니다.

- **Technical Details**: IDEA는 훈련이 필요 없는 방법으로, 이미지-텍스트 쌍의 보완적인 관계를 활용하여 다중 모달리티(multi-modality) 간의 의미적 연관성을 탐구합니다. T-IDEA는 경량 프로젝터와 학습 가능한 잠재 공간(learnable latent space)을 통합하여 IDEA의 성능을 더욱 향상시킵니다. 이러한 방식은 기존의 방법과는 다르게 강력한 성능을 발휘하며, 11개의 공개 이미지 데이터셋에서 실험적으로 검증되었습니다.

- **Performance Highlights**: IDEA와 T-IDEA는 훈련이 필요 없는 설정과 훈련이 요구되는 설정 모두에서 기존의 SOTA 방법들을 초월하는 성능을 보여주었습니다. 새로운 데이터셋인 'IMD-11'을 생성하여 총 1,637,795개의 이미지-텍스트 쌍을 제공하며, 이는 연구에 중요한 기여를 하게 됩니다. 이러한 성과는 제한된 학습 데이터로도 뛰어난 성능을 달성할 수 있음을 시사합니다.



### XMusic: Towards a Generalized and Controllable Symbolic Music Generation Framework (https://arxiv.org/abs/2501.08809)
Comments:
          accepted by TMM

- **What's New**: 최근 인공지능 생성 콘텐츠(AIGC) 분야에서 주목할 만한 발전이 이루어졌지만, AI가 생성한 음악의 질은 아직 이러한 기준에 도달하지 못했습니다. 본 논문은 XMusic이라는 일반화된 기호 음악 생성 프레임워크를 제시하며, 다양한 프롬프트(이미지, 비디오, 텍스트, 태그 및 허밍)를 지원하여 감정적으로 제어 가능한 고품질 기호 음악을 생성할 수 있도록 합니다. 또한, 저자는 XMIDI라는 대규모 기호 음악 데이터셋을 구축하였으며, 이는 108,023개의 MIDI 파일로 구성되어 있습니다.

- **Technical Details**: XMusic은 XProjector와 XComposer라는 두 가지 핵심 구성 요소로 이루어져 있습니다. XProjector는 다양한 모달리티의 프롬프트를 기호 음악 요소(감정, 장르, 리듬 및 음표)로 파싱하며, XComposer는 생성기(Generator)와 선택기(Selector)로 구성되어 있습니다. 생성기는 혁신적인 기호 음악 표현을 기반으로 감정적으로 제어 가능한 음악을 생성하고, 선택기는 품질 평가, 감정 인식 및 장르 인식 작업을 포함한 다중 작업 학습 스킴을 통해 고품질 기호 음악을 식별합니다.

- **Performance Highlights**: XMusic은 종합적인 평가에서 현재의 최첨단 방법을 뛰어넘으며, 탁월한 음악 품질을 보여주었습니다. 객관적이고 주관적인 평가 결과, XMusic이 상대적으로 더 높은 성능을 유지함을 확인할 수 있었습니다. 또한, XMusic은 WAIC 2023에서 아홉 가지 하이라이트 중 하나로 선정되었습니다.



### Networked Agents in the Dark: Team Value Learning under Partial Observability (https://arxiv.org/abs/2501.08778)
Comments:
          18 pages, 7 figures, 5 tables. Accepted as supplemental material at Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025), Detroit, Michigan, USA, May 19 - 23, 2025, IFAAMAS

- **What's New**: 이 논문은 새로운 협력적 다중 에이전트 강화 학습(Cooperative Multi-Agent Reinforcement Learning, MARL) 접근법인 DNA-MARL을 제안합니다. 기존 방법들은 전체 상태 정보나 공동 관측에 의존했으나, 본 연구에서는 제한된 관측(Partial Observability) 하에 상호 목표를 달성하는 방법을 학습해야 합니다. 또한, 에이전트들은 개별 보상을 수집하고, 지역 통신을 통해 팀 가치 함수를 근사하는 방법을 사용하여 협력적인 행동을 구현합니다.

- **Technical Details**: 우리는 네트워크 동적 부분 관찰 마르코프 게임(Networked Dynamic Partially Observable Markov Game, ND-POMG) 프레임워크를 도입하여 에이전트가 스위칭 토폴로지 통신망에서 소통하는 방법을 설명합니다. DNA-MARL은 지역 통신을 위한 합의 메커니즘(consensus mechanism)과 지역 계산을 위한 경량 하강(Gradient Descent) 방법을 사용합니다. 이 접근법은 에이전트들이 정보를 공유하여 팀 가치를 합의하는 것을 통해 협력적인 가치 함수 학습을 촉진합니다.

- **Performance Highlights**: DNA-MARL은 기존의 협력적 학습 방법보다 뛰어난 성과를 보여주었으며, 다양한 벤치마크 MARL 시나리오를 통해 평가되었습니다. 이 방식은 개인의 데이터 프라이버시를 보호하면서도 효과적인 협력을 통해 더 나은 결과를 이끌어낼 수 있도록 설계되었습니다. 본 논문은 DNA-MARL이 다른 비중심화 훈련 및 실행 시스템보다 우수한 성능을 보임을 입증합니다.



### How Developers Interact with AI: A Taxonomy of Human-AI Collaboration in Software Engineering (https://arxiv.org/abs/2501.08774)
Comments:
          Accepted at 2nd ACM International Conference on AI Foundation Models and Software Engineering (FORGE 2025)

- **What's New**: 이 논문은 소프트웨어 개발에서 AI 도구와 개발자 간의 상호작용 유형에 대한 최초의 분류법(taxonomy)을 제안합니다. 11가지 구체적인 상호작용 유형이 정의되어 있으며, 이러한 다양성은 AI 도구 사용의 복잡성을 높입니다. 연구자와 도구 설계자는 개발자-AI 인터페이스 간의 공통된 언어와 모델이 부족하여 최적화를 어렵게 하는데, 이 분류법이 이러한 격차를 해소할 수 있습니다.

- **Technical Details**: 각 상호작용 유형은 트리거(trigger), AI 반응(AI response), 개발자 반응(developer response), 생성된 출력(output), 구체적인 예로 나뉩니다. 예를 들어 auto-complete 코드 제안은 자동으로 실행되어 개발 워크플로우에 통합되며 신속한 생산성을 제공합니다. 명령 기반 행동(command-driven actions)이나 대화형 지원(conversational assistance)은 특정 작업을 동적으로 수행할 수 있게 해줍니다.

- **Performance Highlights**: 이 연구는 개발자가 AI 도구를 사용하면서 겪는 상호작용의 다양한 유형과 그에 따른 효과를 분석합니다. 또한, 각 상호작용 유형이 생산성과 코드 품질에 미치는 영향을 논의하여, 초보자와 숙련자의 요구에 맞춘 최적의 도구 탐색에 기여하고자 합니다. 이를 통해 AI 도구가 실제 개발자의 필요와 작업 흐름에 부합하도록 발전시킬 수 있는 기회를 모색합니다.



### Leveraging LLM Agents for Translating Network Configurations (https://arxiv.org/abs/2501.08760)
- **What's New**: 이번 논문에서는 네트워크 설정 번역(configuration translation)의 필요성을 강조하며, 대규모 언어 모델(LLM) 에이전트를 활용한 새로운 프레임워크를 제안합니다. 이 프레임워크는 네트워크 장비의 이질성을 극복하기 위해 의도 기반의 접근 방법을 사용하여, 보다 정교하고 자동화된 설정 번역을 가능하게 합니다. 연구 결과에 따르면, 제안한 방법은 97.74%의 구문(correctness) 정확성을 달성하며 기존의 방법들보다 뛰어난 번역 성능을 보였습니다.

- **Technical Details**: 제안된 의도 기반 검색 보강 생성(IGAG) 모듈은 설정 파일을 조각으로 나누고, 의도를 추출한 후 정확한 번역을 생성하는 시스템입니다. IRAG 모듈은 셋의 주요 구성 요소로 이루어져 있습니다: (a) 설정 분할 및 의도 추출을 위한 설계된 프롬프트 시스템, (b) 필터링과 투표 전략을 결합한 수동 검색 메커니즘, (c) 맥락 종속성을 유지하는 점진적 번역 과정입니다. 또한, 번역의 정확성을 높이기 위해 이중 검증 모듈을 설계했습니다.

- **Performance Highlights**: 실험을 통해 제안한 방법은 97.74%의 구문 정확성을 달성하였으며, 기존의 최신 방법들과 비교해도 번역의 정확성에서 뛰어난 성과를 보이고 있습니다. 이 논문은 다양한 벤더 간의 네트워크 설정 번역의 어려움을 분석하고, 이를 해결하기 위한 LLM 에이전트를 활용한 프레임워크를 구현하여 실제 데이터셋에서 성능을 평가했습니다. 향후 코드 또한 리뷰 프로세스 이후 오픈 소스로 제공될 예정입니다.



### Self-supervised Transformation Learning for Equivariant Representations (https://arxiv.org/abs/2501.08712)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 자기지도 변환 학습(Self-supervised Transformation Learning, STL) 방법을 제안합니다. 기존의 변환 레이블을 이미지 쌍에서 유도된 변환 표현으로 대체함으로써, 변환의 내성을 높이고 성능을 개선하는 것을 목표로 합니다. STL은 이전 방법들과 같은 배치 복잡성을 유지하면서도 변환 표현을 학습할 수 있는 방식으로, 다양한 분류와 탐지 작업에서 효과를 입증했습니다.

- **Technical Details**: STL에서 사용하는 변환 표현은 이미지에 적용된 변환에 불변성을 가지고 있습니다. 이 표현을 학습함으로써, STL은 정확하고 복잡한 변환 정보를 캡처할 수 있습니다. 추가로 AugMix와 같은 복잡한 변환을 통합하여 과거의 동등한 방법에서 불가능했던 성능 향상을 달성했습니다. 이러한 방식은 다양한 기초 모델과도 호환 가능하여 폭넓은 적용성을 지니고 있습니다.

- **Performance Highlights**: STL은 여러 데이터셋과 작업에서 매우 경쟁력 있는 성능을 발휘했습니다. 11개의 벤치마크 중 7개에서 기존 방법들을 초과하는 성과를 보여주었으며, 탐지 작업에서 특히 두드러진 성과를 올렸습니다. 변환 표현의 상호 의존성을 효과적으로 캡처함으로써, STL은 기존의 변환 학습 방법들보다 우수한 변환 예측 성능을 자랑합니다.



### SPEQ: Stabilization Phases for Efficient Q-Learning in High Update-To-Data Ratio Reinforcement Learning (https://arxiv.org/abs/2501.08669)
- **What's New**: 이번 논문에서는 Deep Reinforcement Learning (심층 강화 학습)에서 샘플 효율성을 향상시키기 위한 새로운 접근 방식을 소개합니다. 특히, Update-To-Data (UTD) 비율을 최적화하여 컴퓨팅 효율성을 개선하는 방법론을 설명합니다. 이를 통해 기존의 DroQ 알고리즘보다 56% 적은 그래디언트 업데이트 수와 50%적은 훈련 시간을 요구하면서도 비슷한 성능을 달성하였습니다.

- **Technical Details**: 제안된 SPEQ (Stabilization Phases for Efficient Q-Learning) 방법은 두 가지 단계로 나뉘어 있습니다: 낮은 UTD 비율의 온라인 훈련 단계와 높은 UTD 비율의 오프라인 안정화 단계. 안정화 단계에서는 새로운 환경 상호작용을 수집하지 않고도 Q-함수를 튜닝하여 리플레이 버퍼의 효율성을 향상시킵니다. 이 방법은 Q-값의 편향(bias)을 줄이고, 더 효과적으로 저장된 샘플을 활용할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과에 따르면, SPEQ는 최신 고 UTD 비율 알고리즘과 유사한 성능을 발휘하면서도 적은 수의 그래디언트 단계와 낮은 컴퓨팅 자원을 요구합니다. 이는 SPEQ가 더 나은 샘플 효율성을 유지하면서도 실용적인 컴퓨터 비용을 제공하는 것을 의미합니다. 이러한 결과는 SPEQ의 전반적인 컴퓨팅 효율성이 기존 기술보다 뛰어남을 입증합니다.



### Fine-grained Spatio-temporal Event Prediction with Self-adaptive Anchor Graph (https://arxiv.org/abs/2501.08653)
Comments:
          Accepted to SIAM International Conference on Data Mining 2025 (SDM'25)

- **What's New**: 이번 연구에서는 정교한 이벤트 예측을 위한 새로운 Graph Spatio-Temporal Point Process (GSTPP) 모델을 제안합니다. GSTPP는 neural Ordinary Differential Equations (ODEs)를 사용하여 공간 지역의 동적 상태를 공동으로 모델링하는 인코더-디코더 아키텍처를 채택합니다. Self-Adaptive Anchor Graph (SAAG)를 기반으로 한 이 모델은 복잡한 공간 이벤트 패턴을 학습하는 능력을 향상시킵니다.

- **Technical Details**: GSTPP 모델은 광범위한 지역에서 발생하는 이벤트의 동적을 반영하기 위해 지역별 상태와 전역 상태를 공동으로 모델링합니다. SAAG를 사용하여 공간 의존성과 지역 간의 상호작용을 효과적으로 캡처하며, 이는 모델의 지역적 특성을 학습하는 과정을 돕습니다. 또한 Location-aware Graph Convolutional Network (L-GCN)와 Relative Location Encoder (RLE)와 같은 여러 서브모듈을 통해 성능을 더욱 강화합니다.

- **Performance Highlights**: 제안된 GSTPP 모델은 기존의 spatio-temporal event prediction 접근 방식에 비해 정확도를 상당히 향상시킵니다. 실험 결과는 GSTPP가 복잡한 이벤트 동적을 더욱 효과적으로 모델링할 수 있음을 보여줍니다. 이러한 결과는 GSTPP가 spatio-temporal 이벤트 예측 분야에서 최신 기술의 중요한 발전을 의미합니다.



### MAGNET: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities (https://arxiv.org/abs/2501.08648)
- **What's New**: 이번 연구에서는 MAGNET(Modified Attention for Generation and Encoding of Text)를 소개합니다. MAGNET는 단방향 디코더 전용 대형 언어 모델(LLMs)을 양방향 모델로 적응시켜 강력한 텍스트 표현을 생성하고, 누락된 텍스트를 채우는 기능을 향상시킵니다. 이는 세 가지 자기 지도 학습(Self-Supervised Learning) 목표와 결합된 주의(attention) 메커니즘을 사용하여 통합 훈련을 가능하게 합니다.

- **Technical Details**: MAGNET는(1) 토큰 수준 및 문장 수준의 표현을 학습하기 위한 마스킹 모델링 목표,(2) 문장 수준 표현을 위한 대조 목표, (3) 누락된 텍스트를 채우기 위한 목표를 설정합니다. 또한, 특별히 설계된 주의 마스크를 통해 양방향 주의과 인과 주의를 결합하여 동시에 훈련할 수 있게 합니다. 이 방식은 LLaMA-2-7B 모델에 적용되며 쉽게 수정 가능함을 보여줍니다.

- **Performance Highlights**: MAGNET로 조정된 LLaMA-2-7B는 토큰 수준 및 문장 수준의 표현 학습 작업에서 기존의 강력한 텍스트 인코더를 초과 달성합니다. 또한, 문맥적으로 적절한 텍스트를 효율적으로 생성할 수 있으며, 반복 문제 없이 개방형 텍스트 생성 능력을 유지합니다. 마지막으로, MAGNET으로 조정된 LLM은 사전 훈련 동안 습득한 지식을 보존합니다.



### Reassessing the Role of Chain-of-Thought in Sentiment Analysis: Insights and Limitations (https://arxiv.org/abs/2501.08641)
- **What's New**: 이 논문은 대형 언어 모델에서 의미적 이해의 관계를 탐구합니다. 기존의 언어와 사고에 대한 두 가지 입장에서, 사고가 언어 모델의 의미적 이해에 미치는 영향을 평가하기 위해 체인 오브 사고(Chain-of-Thought) 방법을 사용한 실험을 수행했습니다. 실험 결과는 이 접근 방법이 감정 분석(sentiment analysis) 과제에서 미미한 영향을 미친다고 나타났습니다.

- **Technical Details**: 연구팀은 언어와 사고의 관계를 탐구하기 위해 체인 오브 사고(CoT) 기법을 감정 기반 분석(aspect-based sentiment analysis) 과제에 적용했습니다. CoT는 모델의 추론 능력을 촉진하기 위해 단계별 추론 과정을 제공하며, 실험에서는 다양한 데이터 세트(Dataset)와 모델을 활용했습니다. 새로운 감정 데이터 세트를 수동으로 구축하여 감정의 복잡성과 변화를 평가하고, CoT가 감정 이해에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과에 따르면, CoT 방법은 가장 작은 모델(Gemma-2, 2B)에서 및 1-shot 설정에서 성능을 개선하는 것으로 나타났습니다. 그러나 전반적으로 CoT가 감정 중심의 의미적 과제에서 미미한 영향을 미친 것으로 결론지어졌습니다. 감정 분석 태스크에서 모델의 성과는 주로 시연에서 제공된 정보에 의해 좌우된다는 사실이 밝혀졌습니다.



### ViBidirectionMT-Eval: Machine Translation for Vietnamese-Chinese and Vietnamese-Lao language pair (https://arxiv.org/abs/2501.08621)
- **What's New**: 이 논문은 VLSP 2022-2023 기계 번역 공유 작업의 결과를 발표하며, 베트남어-중국어 및 베트남어-라오어 기계 번역을 초점으로 하고 있습니다. 이 작업은 베트남어 및 음성 처리(VLSP) 워크숍의 일환으로 진행되었으며, 기계 번역 시스템의 구축을 목표로 하였습니다. 제출된 모델들은 1,000 쌍의 테스트 쌍을 기반으로 평가되었으며, BLEU와 SacreBLEU와 같은 기존 지표들을 사용했습니다.

- **Technical Details**: 기계 번역(Neural Machine Translation, NMT) 시스템은 여전히 번역 품질에서 많은 도전 과제를 안고 있습니다. 이번 공유 작업은 데이터셋을 제공하고, 기계 번역 모델을 테스트할 수 있는 공적인 평가 세트를 포함했습니다. VLSP 2022와 2023에서 각각 베트남어-중국어 및 베트남어-라오어 번역을 위해 다양한 양의 이중 언어 문장 쌍을 포함하여 훈련 데이터를 구성했습니다.

- **Performance Highlights**: VLSP 2022와 2023 기계 번역 작업에 등록된 팀 수는 각각 25개 및 26개 팀이었으며, 높은 품질의 결과물을 제출한 팀들이 평가되었습니다. 기계 번역 작업에서 가장 효과적인 세 가지 접근 방식을 선정하여 그 내용과 기여도를 문서화했습니다. 최종 모델을 구성하는 데 있어 팀들은 데이터 합성을 활용하는 기법을 통해 훈련 세트를 확대하여 정확하고 자연스러운 번역을 달성하였습니다.



### Disjoint Processing Mechanisms of Hierarchical and Linear Grammars in Large Language Models (https://arxiv.org/abs/2501.08618)
- **What's New**: 이 논문은 자연어 처리에서의 계층적 구조에 대한 대형 언어 모델(LLMs)의 반응을 조사합니다. 연구진은 LLMs가 계층적인 문法(hierarchical grammar)과 비계층적인 문법(linear grammar)에서 어떻게 다르게 작용하는지를 실험을 통해 확인하였습니다. 특히, 계층적 문법을 처리하는 메커니즘이 비계층적 문법과는 구별되는 특성을 가진다는 것을 보여주었습니다.

- **Technical Details**: 이 연구에서는 Mistral-v0.3, QWen 2, Llama 2 및 Llama 3.1 모델을 사용하여 실험을 진행하였습니다. 각각의 문법 구조는 영어, 이탈리아어, 일본어를 기반으로 생성되었으며, 총 18개의 문법이 실험되었습니다. LLMs의 계층적 및 비계층적 문법 처리를 비교하기 위해 다양한 입력 패턴과 문법을 통해 실험이 설계되었습니다.

- **Performance Highlights**: 실험 결과, LLMs는 계층적 문법과 비계층적 문법의 입력에 대해 뚜렷한 차이를 보이며, 이들 문법을 처리하는 특정 구성 요소의 역할이 다르다는 것을 확인했습니다. 또한, 새로운(nonce) 문법에서도 계층적 언어 구조에 대한 감수성이 드러나, 이는 의미와 관계없이 언어 데이터에 대한 노출만으로도 기능적 전문화가 가능하다는 결론에 이르게 하였습니다.



### RLHS: Mitigating Misalignment in RLHF with Hindsight Simulation (https://arxiv.org/abs/2501.08617)
- **What's New**: 이번 연구는 기존의 Reinforcement Learning from Human Feedback (RLHF) 방법의 한계를 극복하기 위해 Hindsight Feedback에 기초한 새로운 알고리즘인 Reinforcement Learning from Hindsight Simulation (RLHS)를 소개합니다. RLHF는 즉각적인 피드백을 통해 모델을 최적화하려 하였으나, 이로 인해 모델의 행동이 인간의 가치와 잘 align되지 않을 수 있다는 문제를 지적합니다. RLHS는 시뮬레이션된 결과를 통해 피드백을 생성하여 이러한 미스알라인먼트를 효과적으로 완화할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 Partial Observable Markov Decision Process (POMDP)로 모델링된 인간 의사결정 문제를 다릅니다. RLHS 방법은 AI 시스템이 가능한 인간 행동을 시뮬레이션하고 그 결과로부터 피드백을 받는 구조로, 이는 기존의 RLHF와는 다른 접근 방식입니다. RLHS는 Proximal Policy Optimization (PPO)와 Direct Preference Optimization (DPO)과 같은 선호 최적화 방법에 적용되어, 모델의 misalignment를 줄이는 데 효과적임을 실증적으로 입증했습니다.

- **Performance Highlights**: 인간 사용자 연구 결과, RLHS는 RLHF보다 사용자 목표 달성 및 만족도에서 일관되게 우수한 성능을 보였습니다. 시뮬레이션된 피드백만을 사용하였음에도 불구하고, RLHS는 사용자의 진정한 유틸리티를 향상시키며 잘못된 정보를 기반으로 한 결정을 줄이는 데 기여했습니다. 이 결과들은 장기적인 결과에 초점을 맞추는 것이 RLHF의 misalignment을 완화하는 데 중요함을 강조합니다.



### AutoRestTest: A Tool for Automated REST API Testing Using LLMs and MARL (https://arxiv.org/abs/2501.08600)
Comments:
          To be published in the 47th IEEE/ACM International Conference on Software Engineering - Demonstration Track (ICSE-Demo 2025)

- **What's New**: 본 논문에서는 REST API 테스트를 위한 새로운 도구 AutoRestTest를 소개합니다. 이 도구는 Semantic Operation Dependency Graph (SODG)와 Multi-Agent Reinforcement Learning (MARL), 대형 언어 모델(LLM)을 통합하여 API 테스트의 효과성을 극대화합니다. AutoRestTest는 작업 종속 매개변수를 파악하고, 다섯 개의 전문 에이전트를 활용해 작업 시퀀스와 매개변수 조합을 생성합니다.

- **Technical Details**: AutoRestTest는 REST API 테스트의 다양한 복잡성을 다루기 위해 SODG를 사용하여 작업 간의 종속성을 분석합니다. 각 에이전트는 특정 작업을 수행하고, 다양한 매개변수 및 값과의 조합을 통해 최적의 테스트 경로를 생성합니다. 이 도구는 CLI(Command-Line Interface)를 제공하며, 성공적인 작업 횟수, 고유 서버 오류 감지 및 경과 시간을 지속적으로 기록합니다.

- **Performance Highlights**: AutoRestTest는 기존의 REST API 테스트 도구들과 비교하여 높은 코드 커버리지와 우수한 결함 탐지 능력을 입증하였습니다. 실험 결과, AutoRestTest는 다양한 메트릭에서 상위 도구들을 초과하는 결과를 보여주었으며, 특히 119.17%, 59.83%, 52.42% 더 많은 분기, 코드 라인 및 메서드를 커버했습니다. 또한, 이 도구는 9.2배, 2.5배, 2.4배 더 많은 버그를 발견한 것으로 나타났습니다.



### LlamaRestTest: Effective REST API Testing with Small Language Models (https://arxiv.org/abs/2501.08598)
Comments:
          To be published in the ACM International Conference on the Foundations of Software Engineering (FSE 2025)

- **What's New**: LlamaRestTest는 REST API 테스트를 위한 새로운 접근 방식으로, 서버 응답을 포함한 두 개의 맞춤형 LLM을 활용하여 신뢰성 있는 테스트 입력을 생성하고 매개변수 간의 의존성을 발견하는 기능을 제공합니다. 이 연구는 Llama3-8b 모델을 기반으로 한 LlamaREST-IPD와 LlamaREST-EX를 사용하여 동적 서버 피드백에 따른 지속적인 개선의 필요성을 강조합니다. 특히 해당 도구는 서버 응답을 통해 테스트 중에 테스트 입력과 규칙을 동적으로 정제하는 혁신을 선보입니다.

- **Technical Details**: LlamaRestTest는 Llama3-8b 모델을 미세 조정하여 생성된 두 개의 LLM을 포함하고 있으며, 각각 매개변수 의존성을 식별하고 입력 값을 생성하는 데 적합합니다. LlamaREST-IPD는 각 API에 대해 어떤 매개변수를 선택할지를 결정하고, LlamaREST-EX는 매개변수에 할당할 값을 정합니다. 이 모델들은 2-bit, 4-bit, 8-bit 양자화로 최적화되어 더 작고 빠른 성능을 유지하며, ARAT-RL 프레임워크와 통합되어 REST API 테스트에 활용됩니다.

- **Performance Highlights**: LlamaRestTest는 RESTGPT 및 여러 최신 REST API 테스트 도구들과 비교하여 코드 커버리지와 에러 감지에서 우수한 성과를 보였습니다. 특히 12개의 실제 RESTful 서비스에서 측정된 성능 실험을 통해, 미세 조정된 LlamaREST-EX는 입력 생성의 정확성을 크게 향상시켰으며, 4-bit 및 8-bit 양자화 버전도 합리적인 성능을 유지하는 것으로 나타났습니다. 이 연구 결과는 LlamaRestTest가 REST API 테스트의 동적 피드백을 효과적으로 활용하면서 주목할 만한 결과를 도출해냈음을 보여줍니다.



### OpenMLDB: A Real-Time Relational Data Feature Computation System for Online ML (https://arxiv.org/abs/2501.08591)
- **What's New**: OpenMLDB는 오프라인과 온라인에서 일관된 특성 계산을 보장하는 통합 쿼리 계획 생성기를 제공하여 특성 배포 시간을 크게 단축합니다. 이 시스템은 이전에 각기 다른 팀과 시스템이 처리하던 오프라인 및 온라인 프로세스를 통합하여 데이터 일관성을 높입니다. 또한, OpenMLDB는 클라우드 환경에서 높은 동시성을 유지하며 실시간으로 안정적인 특성 업데이트를 지원합니다.

- **Technical Details**: OpenMLDB는 오프라인 및 온라인 계산을 위한 최적화 기술을 통해 초고속 특성 계산 성능을 실현합니다. 온라인 계산의 경우, 긴 윈도우 집계를 개선하기 위해 사전 집계 결과를 재사용하며, 다중 테이블 윈도우 유니온의 성능 병목을 해결하는 자가 조정 전략을 사용합니다. 오프라인 계산의 경우, 다중 윈도우 병렬 최적화를 지원하며 데이터 스큐를 줄이기 위해 튜플을 재분배합니다.

- **Performance Highlights**: OpenMLDB는 Flink 및 DuckDB보다 10배~20배 높은 온라인 성능을 제공하며, Spark 및 GreenPlum과 같은 MPP 데이터베이스에 비해 6배 더 빠른 오프라인 성능을 자랑합니다. 또한, Redis와 같은 인메모리 데이터베이스에 비해 40% 낮은 메모리 사용량을 기록하여 리소스 절약에 기여합니다. 현재 OpenMLDB는 150명 이상의 기여자가 참여하고 있으며, GitHub에서 1.6k 스타를 얻었습니다.



### A Systematic Review of Machine Learning Methods for Multimodal EEG Data in Clinical Application (https://arxiv.org/abs/2501.08585)
Comments:
          This paper includes 4 figures, 6 tables, and totals 18 pages

- **What's New**: 이번 연구에서는 다중 모달 데이터(multimodal data)를 EEG 자료에 통합하여 기계 학습(machine learning)과 딥 러닝(deep learning) 모델의 정확성을 높이는 방법을 탐구했습니다. 여러 임상 응용 분야에서의 EEG 데이터의 새롭고 효과적인 활용을 제시하며, 신경정신적 장애와 같은 복잡한 임상 문제를 해결하는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: 문헌 검색은 PubMed, Web of Science, Google Scholar를 통해 수행되었으며, 총 16개의 연구가 최종적으로 선정되었습니다. 데이터 융합(data fusion)은 신호(signal), 특징(feature), 결정(decision) 수준에서 이루어졌고, 가장 많이 사용된 기계 학습 모델은 서포트 벡터 머신(support vector machines, SVM)과 결정 트리(decision trees)였습니다.

- **Performance Highlights**: 16개의 연구 중 11개에서 다중 모달 EEG 데이터를 사용했을 때 모델의 정확도가 향상되었다고 보고되었습니다. 이 리뷰는 다중 모달 EEG 기반 기계 학습 모델이 임상 진단 및 문제 해결에 중요한 가능성을 갖고 있다는 점을 강조합니다.



### Towards Lightweight and Stable Zero-shot TTS with Self-distilled Representation Disentanglemen (https://arxiv.org/abs/2501.08566)
Comments:
          5 pages,4 figures

- **What's New**: 이 논문에서는 새로운 경량의 제로샷 텍스트-투-스피치(Zero-shot Text-To-Speech, TTS) 시스템을 소개합니다. 기존의 제로샷 TTS 시스템은 대규모 데이터와 모델에 의존하는 반면, 본 연구에서는 언어 내용과 화자 속성을 효과적으로 모델링하는 새로운 아키텍처를 도입합니다. 또한, 객관적인 데이터 쌍을 생성하는 자기 증류(self-distillation) 프레임워크를 통해 모델의 훈련 능력을 향상시킵니다.

- **Technical Details**: 제안된 시스템은 콘텐츠 추출(content extraction)과 화자 적응(speaker adaptation)의 두 가지 주요 구성 요소로 이루어져 있습니다. Mel Variational Autoencoder(VAE)를 사용하여 텍스트와 멜 스펙트로그램에서 화자 독립적인 콘텐츠 표현을 추출합니다. 이를 통해 생성된 콘텐츠 표현은 다중 수준의 화자 특성과 결합하여 목표 음성을 생성하는 데 필요한 조건을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안하는 시스템은 콘텐츠 무결성(content integrity)에서 기존 모델을 초월하며 화자 유사성에서도 좋은 결과를 보였습니다. 컴퓨테이셔널 효율이 매우 뛰어나 RTF(Real-Time Factor)는 CPU에서 0.13, GPU에서 0.012를 기록하여 자원이 제한된 환경에서도 효과적으로 동작할 수 있습니다. 이를 통해 본 연구는 실시간 응용 프로그램에도 적합한 제로샷 TTS 시스템의 가능성을 제시합니다.



### LAMS: LLM-Driven Automatic Mode Switching for Assistive Teleoperation (https://arxiv.org/abs/2501.08558)
- **What's New**: 본 논문은 LLM-Driven Automatic Mode Switching (LAMS)라는 새로운 접근 방식을 소개합니다. LAMS는 사용자가 이전에 수행한 작업에 대한 예시 없이도 작업 맥락에 따라 자동으로 제어 모드를 전환할 수 있도록 Large Language Models (LLMs)를 활용합니다. 기존의 자동 모드 스위칭 방법들이 특정 작업에 의존하는 것과는 달리, LAMS는 일반화 가능성을 강조합니다.

- **Technical Details**: LAMS는 사용자가 조작하는 작업의 맥락을 자연어 지시로 변환하고, 이를 LLM에 입력하여 조이스틱의 방향 이동을 특정 로봇 행동으로 맵핑합니다. 이 과정에서 사용자 상호작용을 통해 생성된 모드 전환 예시를 통합하여 성능을 점진적으로 개선할 수 있습니다. 또한, LAMS는 사용자 연구에서 10명의 참가자를 대상으로 복잡한 작업을 수행함으로써 그 효과를 입증했습니다.

- **Performance Highlights**: LAMS는 사용자가 복잡한 여러 단계를 포함하는 작업을 수행할 때 필요한 수동 모드 전환의 수를 줄이는 데 효과적임을 보여주었습니다. 연구 결과, 사용자는 LAMS를 기존의 대안 방법보다 선호하며, LAMS는 반복적인 작업 수행을 통해 자동 모드 전환 능력을 개선하는 것으로 나타났습니다. 이로 인해 사용자의 인지 부하가 줄어들고 작업 효율성이 향상되었습니다.



### The Devil is in Temporal Token: High Quality Video Reasoning Segmentation (https://arxiv.org/abs/2501.08549)
- **What's New**: VRS-HQ는 기존 비디오 추론 분할(Video Reasoning Segmentation, VRS) 방법의 한계를 보완하기 위해 고안된 새로운 접근 방식입니다. 이 모델은 Multimodal Large Language Models (MLLMs)를 활용하여 시간적(dynamic) 특징을 계층적으로 통합합니다. 주요 혁신으로는 Temporal Dynamic Aggregation (TDA)와 Token-driven Keyframe Selection (TKS)가 있습니다.

- **Technical Details**: VRS-HQ는 프레임 수준 및 시간 수준의 토큰을 사용하여 MLLM의 자기 회귀 학습(autoregressive learning)을 통해 지역 및 전역 정보를 효과적으로 캡처합니다. Temporal Dynamic Aggregation(TDA)을 통해 cosin similarity 기반의 가중 합성을 적용하여 프레임 수준의 가시적 특징을 시간 수준의 특징으로 통합합니다. Token-driven Keyframe Selection(TKS) 방식으로, 각 샘플링한 프레임에 대해 SAM2를 통해 occlusion score를 계산하여 키프레임을 선택하는 신뢰성을 강화합니다.

- **Performance Highlights**: VRS-HQ는 ReVOS 데이터셋에서 VISA보다 5.9%, 12.5%, 9.1%의 J&F 점수 개선을 달성하며 최첨단 성능을 입증하였습니다. 또한 여러 비디오 분할 기준에서 이전 방법들을 초월하여 7.3%, 5.6%, 6.5%의 성능 향상을 나타냈습니다. 이러한 결과는 VRS-HQ의 강력한 시간적 추론 및 분할 능력을 강조합니다.



### Knowledge prompt chaining for semantic modeling (https://arxiv.org/abs/2501.08540)
- **What's New**: 이 논문은 Knowledge Prompt Chaining이라는 새로운 자동 의미 모델링 프레임워크를 제안합니다. 이 프레임워크는 구조화된 데이터와 관련된 도메인 온톨로지를 체계적으로 직렬화하여 Long In-context Learning (LongICL) 시스템 프롬프트에 주입합니다. 이전 연구들이 통합된 그래프를 구성하는 방식과는 달리, 우리는 데이터의 소수 포인트만으로 프롬프트를 적응하여 더욱 효율적인 모델을 구현합니다. 이렇게 하면 사용자는 새로운 소스를 입력할 때마다 의미 레이블과 의미 모델을 자연스럽게 생성할 수 있습니다.

- **Technical Details**: 우리는 주어진 구조화된 데이터와 온톨로지를 활용해 두 단계로 의미 모델링을 수행합니다. 첫 번째 단계는 의미 라벨링으로, 이는 도메인 온톨로지의 클래스와 속성을 사용하여 데이터 속성 열에 주석을 달아 의미 타입을 도출하는 과정입니다. 두 번째 단계는 이러한 주석을 바탕으로 속성 간의 의미 관계를 파악하는 것입니다. 이 과정에서 효율적으로 구조 정보를 보존하고 그라프의 잠재적 표현을 학습하여 사용자 요구에 맞춘 맞춤화된 모델을 생성합니다.

- **Performance Highlights**: 우리의 방법은 세 가지 실제 데이터 세트에서 평가되었으며, 최신 기술들과 비교했을 때 더 나은 성능을 보였습니다. 프롬프트 체인과 가지치기를 적용함으로써, 우리는 더 적은 토큰을 사용해도 의미 모델을 생성할 수 있음을 입증했습니다. 이로 인해 사용자는 전체 테이블이나 JSON, XML 파일을 처리할 필요 없이 작은 데이터 포인트만으로도 의미를 생성할 수 있어 효율성이 크게 향상되었습니다.



### Dynamic Portfolio Optimization via Augmented DDPG with Quantum Price Levels-Based Trading Strategy (https://arxiv.org/abs/2501.08528)
Comments:
          8 pages

- **What's New**: 이 연구에서 제안된 새로운 모델인 Augmented DDPG는 Deep Deterministic Policy Gradient (DDPG)를 기반으로 하여, DRL 알고리즘의 느린 학습 속도와 높은 샘플 복잡성 문제를 해결하고자 하였습니다. 또한 Quantum Finance Theory (QFT)에서 파생된 Quantum Price Levels (QPLs)를 기반으로 하는 혁신적인 리스크 관리 전략을 도입하여, 기존 모델들에 비해 우수한 수익성과 리스크 관리 능력을 보여주었습니다.

- **Technical Details**: 동적 포트폴리오 최적화(DPO)는 금융 시장의 변동에 따라 자산 배분을 조정하는 과정으로, Augmented DDPG 모델은 드리븐러닝(reinforcement learning)과 딥러닝(deep learning)의 조합을 통해 효율성을 증대시키고자 하였습니다. 이를 통해 초과적인 거래 전략의 학습 및 리스크 제어가 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 Augmented DDPG 모델은 기존의 기본 모델들과 비교했을 때, 더 나은 수익성과 리스크 관리 능력을 보이며, 샘플 복잡성 또한 줄여 효과적인 DPO 문제 해결에 기여하였습니다. 연구진은 이 모델을 통해 DPO 문제에 대한 보다 안정적인 솔루션을 제시하고 있습니다.



### Doc-Guided Sent2Sent++: A Sent2Sent++ Agent with Doc-Guided memory for Document-level Machine Translation (https://arxiv.org/abs/2501.08523)
- **What's New**: 이 논문에서는 문서 수준 기계 번역(Document-level Machine Translation, DocMT)에 관한 새로운 접근법인 Doc-Guided Sent2Sent++를 제안합니다. 이 모델은 인크리멘탈 문장 수준 강제 디코딩 전략(incremental sentence-level forced decoding strategy)을 사용하여 문장 누락을 방지하고 인접 문장 간의 유창성을 높입니다. 기존의 Doc2Doc 및 Doc2Sent 방법과 차별화된 점은 summary(요약) 정보만을 메모리에 포함하여 번역의 일관성을 유지한다는 것입니다.

- **Technical Details**: Sent2Sent++는 두 개의 인접 문장을 함께 디코딩하는 방식으로 작동합니다. 이 때 이전에 번역된 문장이 현재 디코딩에서 접두사로 사용되며, 이는 문서의 전체적인 번역 유창성을 높여줍니다. 또, 이 모델은 Doc-Guided Memory라 불리는 메모리 구조를 도입하여 전체 문서의 요약과 번역만을 포함하여 효율적인 정보 관리를 실현합니다. 이는 개인 문장 번역의 일관성을 보장하면서도 문서 전반에 걸쳐 번역의 품질을 향상시킵니다.

- **Performance Highlights**: 다양한 언어와 도메인에서 광범위한 테스트를 진행한 결과, Sent2Sent++는 기존 방법들보다 품질, 일관성 및 유창성에서 우수한 성능을 보였습니다. 특히, s-COMET, d-COMET, LTCR-$1_f$, 및 document-level perplexity (d-ppl) 같은 지표에서 현저한 개선이 확인되었습니다. 이 결과는 Sent2Sent++가 다국어 및 다양한 도메인에서 종합적으로 뛰어난 성능을 발휘하는 효율적인 접근법임을 뒷받침합니다.



### Mitigating Domain Shift in Federated Learning via Intra- and Inter-Domain Prototypes (https://arxiv.org/abs/2501.08521)
Comments:
          13 pages, 9 figures, 10 tables

- **What's New**: 이 논문에서는 기존의 federated prototype learning 방법이 intra-domain (내부 도메인) 특성을 무시했다는 점을 지적하고, 새로운 방법론인 I²PFL를 제안합니다. I²PFL는 intra-domain (내부 도메인)과 inter-domain (외부 도메인) 프로토타입을 통합하여 도메인 변화(domain shift)를 완화하고, 여러 도메인에서 일반화된 글로벌 모델을 학습할 수 있도록 합니다. 이는 실제 문제에서 분포가 다른 클라이언트들이 협력할 수 있는 환경을 제공합니다.

- **Technical Details**: 제안된 I²PFL 방법에서는 MixUp 기반의 증강된 프로토타입을 통해 타 지역(domain) 내 특성을 aligning (정렬)하는 방식을 사용합니다. 이를 통해 각 지역의 다양성을 포착하고, 로컬 특징의 일반화가 강화됩니다. 또한, 클라이언트 간 도메인 왜곡(domain skew)을 감소시키기 위해 inter-domain 프로토타입을 재가중치하는 메커니즘을 도입하여, 보다 일반화된 프로토타입을 생성합니다.

- **Performance Highlights**: I²PFL 방법은 Digits, Office-10, PACS 데이터셋에 대한 광범위한 실험을 통해 다른 기존 방법론에 비해 우수한 성능을 보였습니다. 특히, 이 방법은 도메인이 이질적인 환경에서도 효과적으로 작동하여 federated learning의 응용 가능성을 확대합니다. 이러한 결과는 I²PFL가 다양한 실제 응용 분야에서도 신뢰할 수 있는 선택이 될 수 있음을 나타냅니다.



### Easing Seasickness through Attention Redirection with a Mindfulness-Based Brain--Computer Interfac (https://arxiv.org/abs/2501.08518)
- **What's New**: 이번 연구에서는 해상에서의 멀미 증상을 완화하기 위한 마인드풀니스 기반의 뇌-컴퓨터 인터페이스(BCI) 시스템을 제안합니다. 기존의 주의 전환 기법을 해양 환경에 맞춰 적용한 점이 특징입니다. 이 시스템은 단일 채널의 헤드밴드를 통해 전두엽의 EEG 신호를 수집하여 실시간으로 마인드풀니스 상태를 평가하고 피드백을 제공합니다.

- **Technical Details**: 시스템은 43명의 참가자가 3개의 세션(실제 피드백 마인드풀니스 세션, 휴식 세션, 유사 피드백 마인드풀니스 세션)에 참여하여 실험이 진행되었습니다. EEG 신호를 수집하고, 그에 따라 실시간으로 마인드풀니스 점수를 제공하여 신체적 불편감에서 마인드풀니스 연습으로 주의를 전환하도록 돕습니다. EEG 분석 결과, 마인드풀니스 BCI의 사용으로 멀미 증상의 심각성이 감소했음을 확인했습니다.

- **Performance Highlights**: 참가자의 81.39%가 마인드풀니스 BCI 개입이 효과적이었다고 보고하였고, MISC(불행 지수) 점수로 측정한 멀미 증상이 유의미하게 감소했습니다. 또한, 실시간 피드백 마인드풀니스 세션 중 전체 EEG 밴드 전력이 감소하여 뇌 활동의 부드러움과 조절된 상태를 조성하는 것으로 나타났습니다. 이 연구는 해상에서의 멀미 개입을 위한 새로운 비약물적이고 효과적인 접근 방식을 제안합니다.



### Exploring the Efficacy of Meta-Learning: Unveiling Superior Data Diversity Utilization of MAML Over Pre-training (https://arxiv.org/abs/2501.08506)
- **What's New**: 이 연구는 대규모 비전 모델의 성능을 좌우하는 데이터셋의 다양한 특성 중에서도 특히 데이터 다양성(data diversity)이 모델 성능에 미치는 영향을 탐구합니다. 기존 연구들이 주로 데이터 양(data size)이나 모델의 크기와 복잡성에 집중했던 것과는 달리, 본 연구에서는 데이터 다양성을 중요한 요소로 제시하고 있습니다.

- **Technical Details**: 연구에서는 Task2Vec라는 메트릭을 사용하여 데이터셋의 내재적인 다양성을 측정합니다. 이 메트릭은 작업(task)을 확률 분포로 간주하고, 서로 다른 작업의 Task2Vec 임베딩(task embedding) 간의 평균 거리 평균을 계산하는 방식으로 작동합니다. 우리의 분석은 12개의 인기 있는 시각 데이터셋과 다양한 모델 구성에서 메타 러닝 (meta-learning) 기법을 연구하였습니다.

- **Performance Highlights**: 실험 결과, 데이터셋 다양성과 모델 성능 사이에 양의 상관관계가 존재하는 것으로 나타났습니다. 특히 higher-order MAML 모델은 데이터 다양성과 모델 성능 간의 더 강력한 상관관계를 보였으며, 연구에서 관찰된 R-squared 값은 0.4에 이르렀습니다. 이는 데이터 다양성이 모델의 성능을 향상시키는 중요한 요소임을 시사합니다.



### Adapting Whisper for Regional Dialects: Enhancing Public Services for Vulnerable Populations in the United Kingdom (https://arxiv.org/abs/2501.08502)
- **What's New**: 본 연구에서는 영국에서의 지역 어악을 포착할 수 있는 최신 자동 음성 인식(ASR) 모델의 성능을 평가하기 위해 공공 서비스 분야에서 새로운 데이터를 수집하였습니다. 스코틀랜드의 두 가지 독특한 방언을 사용하는 지역을 집중 조사하였으며, 이러한 연구는 사회적 취약집단에 속하는 개인들이 겪는 정보 전달의 장애를 해결하는 데 초점을 맞추고 있습니다. WhisperLarge-v3와 같은 최신 ASR 모델을 기반으로 성능을 세밀히 분석하고 있으며, 이는 실질적인 공공 서비스 제공에 있어 중요합니다.

- **Technical Details**: 자동 음성 인식 시스템은 다양한 상황에서 사용되고 있으며, 특히 사회적 언어 편향으로 인한 문제를 다루는 것이 중요하습니다. 연구에서는 Whisper 모델이 복잡한 오디오 환경에서도 낮은 단어 오류율(Word Error Rate, WER)을 보여주며, 다양한 언어 조합에 대한 인식 능력을 갖추고 있음을 강조합니다. 분류 작업에서 WER외에도 인간 검토를 통한 모델 오류 분석을 통해 평가 기법의 장단점을 점검하고 있습니다.

- **Performance Highlights**: Whisper 모델은 수집된 데이터에 대해 기존의 기초 데이터보다 높은 WER를 보였으나, 특정 지역에 대해 세밀하게 조정된 후 성능이 향상된 것으로 나타났습니다. Fine-tuning을 통해 스코틀랜드의 두 지역에 특화된 성능 개선이 가능함을 보였으며, 이는 다른 지역에 있는 데이터세트에도 적용될 수 있음을 시사합니다. 그러나 WER를 평가 기준으로 사용할 때의 한계도 동시에 분석되어, 평가 메트릭의 활용에 대한 논의가 이어집니다.



### Quantifying the Importance of Data Alignment in Downstream Model Performanc (https://arxiv.org/abs/2501.08496)
- **What's New**: 이 논문은 전통적으로 강조되어온 데이터셋 크기 대신, 데이터 정렬(data alignment)의 역할에 주목합니다. 이 정렬은 데이터 품질(data quality)에서 간과되기 쉬운 측면으로, 대규모 언어 모델(Large Language Models, LLM)의 성능 개선에 미치는 영향을 정량화하기 위해 Task2Vec 기반의 정렬 계수(alignment coefficient)를 사용했습니다.

- **Technical Details**: 연구는 두 가지 설정에서 제어된 개입 실험(interventional experiments)을 실시했습니다. 첫 번째는 다양한 프리 트레인(pre-training) 데이터셋과 평가 데이터셋 간의 정렬 계수 증가가 미치는 영향을, 두 번째는 도메인 특화 파인 튜닝(fine-tuning) 데이터셋과 도메인 특화 평가 간의 정렬 계수 증가가 미치는 영향을 분석했습니다. 이러한 설정 중 Autoformalization이라는 특정 도메인 과제를 통해 데이터를 평가하였습니다.

- **Performance Highlights**: 연구 결과 모델의 학습 데이터와 평가 데이터 간의 정렬 계수는 모델의 손실(Loss) 및 혼란도(Perplexity)와 강하게 부정적인 상관관계를 가지며, 이는 각 다운스트림 작업에 대한 모델의 성능에 직접적인 영향을 미친다는 것을 보여줍니다. 특히 데이터가 평가 작업과 잘 정렬된 경우 낮은 혼란도 점수가 나타나며, 이는 LLM 훈련 접근 방식을 재평가할 필요성을 시사합니다.



### Benchmarking Classical, Deep, and Generative Models for Human Activity Recognition (https://arxiv.org/abs/2501.08471)
Comments:
          48 pages, 21 Figures

- **What's New**: 이번 연구는 Human Activity Recognition (HAR)의 다양한 모델, 즉 전통 기계 학습, 딥 러닝 아키텍처 및 Restricted Boltzmann Machines (RBMs)의 성능을 비교합니다. 연구진은 UCI-HAR, OPPORTUNITY, PAMAP2, WISDM, Berkeley MHAD라는 5개의 주요 벤치마크 데이터 세트를 사용하여 다양한 모델의 정확성, 정밀도, 재현율, F1-score 등 다양한 지표를 평가하였습니다. CNN 모델이 전 데이터 세트에서 우수한 성능을 보인다는 것을 발견하였으며, 특히 Berkeley MHAD에서 두드러진 성과를 나타냈습니다.

- **Technical Details**: 본 논문은 HAR을 위한 다양한 모델들이 각기 다른 강점과 한계를 가지고 있음을 강조합니다. 전통적인 모델인 Decision Trees, Random Forests가 소규모 데이터 세트에서 좋은 성능을 보이나, 대규모 복잡한 데이터에서는 어려움을 겪는 반면, RBM 기반 모델은 주로 feature learning에 유용한 잠재력을 보였습니다. 딥 러닝 아키텍처는 복잡한 시공간 패턴을 학습할 수 있지만, 레이블이 있는 데이터와 컴퓨팅 자원을 많이 요구하는 경향이 있습니다.

- **Performance Highlights**: CNN 모델이 모든 데이터 세트에서 최고 성능을 보여주었으며, 이는 다양한 환경에서 HAR의 신뢰성을 높이는 데 기여할 수 있습니다. 기계 학습 및 딥 러닝을 포함한 여러 모델이 다양한 상황에서 어떻게 성능이 달라지는지를 분석하여 연구자들이 필요한 모델을 선택하는 데 도움을 줄 수 있습니다. 결과적으로, 이 연구는 의료, 스마트 환경 및 보안 분야와 같은 실제 응용 프로그램에 대한 귀중한 통찰력을 제공합니다.



### Detecting Contextual Anomalies by Discovering Consistent Spatial Regions (https://arxiv.org/abs/2501.08470)
- **What's New**: 이 논문에서는 비디오 이상 감지를 위한 공간적 맥락 모델링 방법을 제안합니다. 주된 아이디어는 가우시안 혼합 모델(Gaussian mixture models)을 사용하여 객체 수준 활동이 유사한 영역을 클러스터링하는 것입니다. 이러한 간단한 접근법은 경쟁 모델에 비해 수량적으로 더 적은 매개변수를 사용하며, 복잡한 공간 맥락 의존 Street Scene 데이터셋에서 최첨단 성능을 달성합니다.

- **Technical Details**: 논문에서는 '정상' 객체 행동을 학습하기 위해 비디오 내의 다양한 위치에서 객체 행동을 기반으로 하는 공간적 영역을 자연스럽게 발견하는 방법을 설명합니다. 고해상도 지역 발견을 통해, 간단한 가우시안 혼합 모델을 학습하여 공간 맥락 의존 이상을 감지할 수 있음을 보여줍니다. 이 접근방법은 필요한 모델 수를 크게 줄이고 사용자에게 더 나은 해석 가능성을 제공합니다.

- **Performance Highlights**: 제안된 방법은 사용자가 지정한 소수의 활동 지역을 발견하여 Street Scene 데이터셋에서 최첨단 성능을 달성합니다. 이 지역들은 일반적으로 다음과 같은 공간 컨텍스트에서 의미 있는 영역: 교통 차선, 자전거 도로, 보도 등을 포함합니다. 새 객체가 주어진 지역 모델에 따라 정상 또는 비정상으로 분류될 수 있으며, 예를 들어 도로에 있는 보행자나 불법 주정차된 차량과 같은 공간 맥락 의존 이상을 발견할 수 있습니다.



### A Short-Term Predict-Then-Cluster Framework for Meal Delivery Services (https://arxiv.org/abs/2501.08466)
- **What's New**: 본 연구는 주문형 음식 배달 서비스의 수요 예측과 동적 클러스터링을 결합한 단기 예측-클러스터 프레임워크를 제안합니다. 이 프레임워크는 근접한 수요 예측에 기반하여 사용자 정의 운영 제약에 맞는 동적 클러스터를 생성하도록 설계되었습니다. 제안된 방법은 전통적인 시계열 접근법보다 정확성과 계산 효율성에서 우수한 성과를 보입니다.

- **Technical Details**: 프레임워크는 수요 예측 단계와 클러스터링 단계로 구성되어 있으며, 앙상블 학습 방법(ensemble-learning methods)을 활용하여 다변량 특성을 가진 점치기(point) 및 분포 예측(distributional forecasting)을 수행합니다. 특히, 제약이 있는 K-평균 클러스터링(Constrained K-Means Clustering, CKMC)과 연속성 제약이 있는 계층적 클러스터링(Contiguity Constrained Hierarchical Clustering, CCHC-ICE)을 도입하여 수요 예측에 기반한 동적 클러스터를 생성합니다.

- **Performance Highlights**: 유럽 및 대만의 사례 연구 평가 결과, 제안된 방법이 기존의 전통적 시계열 방법들에 비해 더 높은 정확성을 보이며, 계산 효율성 또한 크게 향상되었습니다. 예측된 수요 정보를 통합함으로써 부가적인 운영 통찰력을 제공하고, 이를 통해 배달 효율성이 크게 개선됨을 보여줍니다. 이 연구는 주문형 플랫폼 기반의 도시 물류 및 승객 이동 서비스에 적용될 수 있는 가능성을 제시하며, 지속 가능하고 효율적인 도시 운영을 촉진합니다.



### Towards Zero-Shot & Explainable Video Description by Reasoning over Graphs of Events in Space and Tim (https://arxiv.org/abs/2501.08460)
- **What's New**: 최근 머신러닝의 발전 속에서, Transformer가 컴퓨터 비전과 자연어 처리와 같은 다양한 분야에서 주목받고 있습니다. 이 연구는 비전(vision)과 언어(language) 간의 관계를 이해하는 문제에 도전하며, 비전과 언어 모델 간의 연계성을 설명 가능하고 체계적으로 연결하기 위한 새로운 접근법을 제안합니다. 이러한 접근법은 시간과 공간의 사건(event)을 기반으로 하여 자연어로 비디오를 설명하는 문제를 해결하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 비디오의 프레임에서 발생하는 다양한 고수준 정보를 활용하여 Graph of Events in Space and Time (GEST)를 구축합니다. GEST는 비디오의 물리적, 시간적, 의미적 요소를 나타내는 노드(nodes)와 엣지(edges)로 구성되어 있습니다. 이 표현을 통해 비디오는 명확하고 체계적으로 분석되며, proto-language를 생성한 후 이를 자연어 설명으로 변환하는 두 단계의 과정을 거칩니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 데이터셋에서 수집한 비디오에 대해 풍부하고 관련성 높은 텍스트 설명을 생성할 수 있음을 검증했습니다. 기존의 비디오 설명 모델들이 짧은 캡션(caption)만을 생성하는 것과 달리, VLLMs과의 조합을 통해 더 긴 설명을 가능하게 하였습니다. 본 연구에서는 Bleu와 ROUGE 같은 표준 메트릭(Standard metric)을 활용하여 성과를 평가하고 효율성을 입증했습니다.



### FARE: A Deep Learning-Based Framework for Radar-based Face Recognition and Out-of-distribution Detection (https://arxiv.org/abs/2501.08440)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 이 연구에서는 단거리 FMCW 레이더를 활용한 얼굴 인식 및 OOD (out-of-distribution) 탐지 시스템을 제안합니다. 이 시스템은 Range-Doppler 및 micro Range-Doppler 이미지를 사용하며, 인식 모델의 두 개의 경로를 통해 ID 얼굴 분류와 OOD 탐지를 동시에 수행합니다. 두 단계로 훈련이 진행되며, 첫 단계에서는 triplet loss를 사용하여 ID 얼굴의 분류 정확도를 최적화합니다.

- **Technical Details**: 제안하는 아키텍처는 두 가지 경로, 즉 기본 경로(PP)와 중간 경로(IP)를 통해 구성되어 있습니다. PP는 ID 얼굴의 정확한 분류를 담당하고, IP는 OOD 탐지를 위한 구조입니다. 첫 번째 단계에서는 PP를 훈련시키고, 두 번째 단계에서는 PP를 고정하여 IPs를 훈련하여 OOD 탐지를 수행합니다.

- **Performance Highlights**: 제안된 방식을 통해 60 GHz FMCW 레이더로 생성한 데이터셋에서 ID 얼굴 분류 정확도 99.30%와 OOD 탐지 AUROC 96.91%를 달성했습니다. 또한, FARE는 기존 OOD 탐지 방법들에 비해 뛰어난 성능을 보이며, 이는 보안과 신뢰성을 위한 중요한 발전을 의미합니다.



### Modeling Discrimination with Causal Abstraction (https://arxiv.org/abs/2501.08429)
- **What's New**: 이 논문에서는 인종 차별의 개념을 새롭게 정의하는 프레임워크를 제안합니다. 저자들은 인종이 다른 특성의 고급 추상화라는 관점에서 인종 차별을 재구성하며, 이는 인종이 더 나쁜 대우를 초래할 수 있음을 모델링할 수 있음을 강조합니다. 이 접근법은 인종이 사회적으로 구축된 개념이라는 가정과 인종의 구성 요소 간의 정렬을 통해 인종 차별의 인과적 역할을 명확히 할 수 있는 가능성을 제공합니다.

- **Technical Details**: 저자들은 인종 차별의 인과적 모델링이 불충분하며, 인종이 사회적 맥락에서 구성된 특성이기 때문에 인과적 분석에 대한 모듈성 요구가 충족되지 않는다고 주장합니다. 논문은 인종이 다른 속성과의 상호작용 속에서 사회적으로 어떻게 형성되는지를 탐구하며, 이런 식으로 인종 차별이 발생하는 과정을 명확히 이해하려는 노력을 합니다. 이러한 관점은 인종 차별의 기존 문헌에서의 불일치를 지적하며, 인과적 설명의 정밀성을 유지합니다.

- **Performance Highlights**: 이 연구는 기존 인종 차별 연구에서의 개념적 한계와 모듈성 문제를 해결하기 위한 새로운 패러다임을 제시합니다. 이를 통해 저자들은 사회적 구성 요소와 인종 간의 인과적 관계를 더 깊이 탐구할 수 있음을 보여줍니다. 이론적으로, 이는 알고리즘 공정성 및 사회 과학에서 인종 차별 탐지의 기준을 강화하는 데 기여할 수 있습니다.



### Causal vs. Anticausal merging of predictors (https://arxiv.org/abs/2501.08426)
Comments:
          Presented at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문에서는 동일한 데이터를 사용하여 인과적(causal)과 비인과적(anticausal) 방향에서 예측기를 병합하는 과정에서 발생하는 차이를 연구합니다. 특히 두 개의 연속 변수를 사용하여 하나의 이진 변수를 목표로 삼는 간단한 모델을 통해 비대칭(asymmetries) 현상을 조사합니다. Causal Maximum Entropy (CMAXENT)를 통해 예측기를 병합하는데, 다른 병합 방법을 사용하더라도 이와 유사한 차이가 발생할 것으로 기대하고 있습니다.

- **Technical Details**: CMAXENT 솔루션은 모든 쌍변량 분포를 관찰할 경우 인과적 방향에서는 로지스틱 회귀(logistic regression)로, 비인과적 방향에서는 선형 판별 분석(Linear Discriminant Analysis, LDA)으로 축소됩니다. 이 연구는 예측기를 병합할 때 인과적인 가정이 어떻게 결과에 영향을 미치는지를 분석하며, 특히 모든 변수를 관찰하지 않을 때의 차이를 강조합니다. 의료 분야에서 질병의 유무를 예측하는 데 중요한 인과적 가정의 포함 필요성을 설명합니다.

- **Performance Highlights**: 연구 결과, CMAXENT가 이진 목표 변수와 연속 공변수를 사용할 때 인과적과 비인과적 방향에서 각각 로지스틱 회귀와 LDA로 패러데스가 내려집니다. 또한, OOV(Out-Of-Variable) 일반화에 대한 함의도 분석하여 모든 순간을 관찰하지 못할 경우의 결정 경계(decision boundary) 차이에 대해 논의합니다. 본 논문은 예측기의 병합 과정에서 인과적 가정이 미치는 영향을 규명하여, 머신러닝 및 통계 분야에서 중요한 기여를 하고자 합니다.



### SEAL: Speaker Error Correction using Acoustic-conditioned Large Language Models (https://arxiv.org/abs/2501.08421)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 이 논문은 Speaker Diarization(SD) 시스템의 성능 향상을 위해 Acoustic-conditioned Large Language Models(LLMs)를 활용하는 새로운 접근 방식을 제안합니다. 기존 SD 시스템에서 발생하는 화자 오류를 줄이기 위해 음향 정보를 LLM과 결합합니다. 또한, 복잡한 후처리 없이 LLM의 환각을 줄일 수 있는 단순한 제한 디코딩 전략을 도입하여 더 정확한 화자 레이블을 할당합니다.

- **Technical Details**: 메인 아이디어는 SEAL이라는 프레임워크를 사용하여 EEND(End-to-End Neural Diarization) 네트워크에서 제공하는 음향적 후행 확률을 활용하는 것입니다. 각 단어에 대한 화자 포스터리어 성능을 네트워크를 통해 포착하고, 이를 바탕으로 LLM이 보다 정확한 결과를 도출하도록 합니다. 또한, 화자 확률을 직관적으로 이해하기 쉬운 레이블로 변환하고, 이를 통해 LLM이 더 나은 성능을 발휘하도록 해줍니다.

- **Performance Highlights**: 제안된 접근 방식은 Fisher, Callhome, RT03-CTS 데이터셋을 통해 기존 Acoustic SD와 비교할 때 화자 오류율을 24-43%까지 감소시키는 효과를 보여주었습니다. 이러한 결과는 SEAL이 LLM의 내재된 어휘 지식을 음향 정보와 통합함으로써 발생하는 큰 개선을 반영합니다. 연구 결과는 다양한 다중 화자 전사 응용 프로그램에 이 혁신적인 방법이 실질적인 기여를 할 수 있음을 보여줍니다.



### CVaR-Based Variational Quantum Optimization for User Association in Handoff-Aware Vehicular Networks (https://arxiv.org/abs/2501.08418)
Comments:
          Accepted in IEEE International Conference on Communications (ICC 2025)

- **What's New**: 이번 연구에서는 차량 네트워크(Vehicular Networks, VNet)의 일반화된 할당 문제(Generalized Assignment Problem, GAP)를 해결하기 위해 새로운 Conditional Value at Risk (CVaR) 기반의 변분 양자 고유해법(Variational Quantum Eigensolver, VQE) 프레임워크를 제안합니다. 기존의 선형 합 산술 문제에서 비롯된 GAP는 여러 제약 조건들로 인해 발생하는 계산적 도전 과제가 있습니다.

- **Technical Details**: 제안된 방법은 하이브리드 양자-고전적 구조(hybrid quantum-classical structure)를 채택하여, 목적 함수(objective function)와 제약 조건-specific penalties를 균형 있게 통합한 맞춤형 비용 함수(tailored cost function)를 활용합니다. CVaR-VQE 모델을 사용하여 솔루션 공간의 하위 구간(lower tail)에서 최적화를 집중함으로써, 노이즈가 있는 중간 규모 양자(NISQ) 장치에서의 수렴(convergence)과 회복력(resilience)을 향상시킵니다.

- **Performance Highlights**: 우리의 접근법은 차량 네트워크에서의 사용자 연관 문제(user-association problem)에 적용되었으며, 딥 뉴럴 네트워크(Deep Neural Network, DNN) 접근법에 비해 23.5%의 성능 향상을 달성하였습니다. 이 연구는 차량 네트워크에서 효율적인 자원 할당을 위한 새로운 범위의 가능성을 보여줍니다.



### A Survey on Recent Advances in Self-Organizing Maps (https://arxiv.org/abs/2501.08416)
Comments:
          36 pages

- **What's New**: 이 논문의 주요 내용은 Self-Organising Map(SOM) 알고리즘의 최근 10년간 발전과 응용에 대한 포괄적인 리뷰입니다. Kohonen의 개념을 기초로 한 SOM 알고리즘은 고차원 데이터셋을 2차원으로 시각화하여 클러스터링을 수행하는 데 효과적인 도구입니다. 최근 연구들은 SOM의 효율성과 다양성을 높이기 위해 알고리즘 개선에 중점을 두고 있습니다.

- **Technical Details**: SOM 알고리즘의 주요 단계는 초기화, 훈련, 최적화이며, 각 뉴런의 가중치 벡터를 초기화하고 입력 벡터에 맞춰 뉴런의 가중치를 조정합니다. 가중치는 입력 데이터와의 거리 측정(주로 유클리드 거리)에 기초하여 업데이트됩니다. SOM은 두 가지 모드(훈련 및 매핑)로 작동하며, 고차원 데이터를 2차원 맵으로 변환합니다.

- **Performance Highlights**: SOM은 경제 분석, 소프트웨어 공학, 감정 분류, 의료 연구 등 다양한 분야에서 활발하게 사용되고 있습니다. 특히, 최근 연구들은 SOM 알고리즘의 계산 비용을 줄이고, 데이터 시각화의 품질을 향상시키기 위해 여러 기술적 접근 방안을 모색하고 있습니다. SOM의 진화 과정은 데이터 관리, 토폴로지 및 메트릭, 학습 기술, 시각화 및 성능 개선을 포함합니다.



### Cross-Modal Transferable Image-to-Video Attack on Video Quality Metrics (https://arxiv.org/abs/2501.08415)
Comments:
          Accepted for VISAPP 2025

- **What's New**: 최근 연구에서 현대 영상 및 비디오 품질 평가(IQA/VQA) 지표가 적대적 공격에 취약하다는 것이 밝혀졌습니다. 이는 공공 벤치마크와 자율주행 같은 보다 중대한 상황에서 이러한 지표에 의존하는 것의 안전성에 대한 우려를 불러일으킵니다. 본 논문에서는 이미지 품질 지표(IQA)와 CLIP 모듈을 활용하여 비디오 품질 평가(VQA) 모델에 대한 적대적 공격의 취약성을 탐구하는 새로운 방법, IC2VQA를 제안합니다.

- **Technical Details**: IC2VQA 접근법은 여러 고해상도 비디오와 3개의 대상 VQA 모델을 사용하여 포괄적인 실험을 수행한 결과 기존 방법들에 비해 뛰어난 성능을 보여주었습니다. 이 방법은 이미지와 비디오의 저수준 특징 공간의 유사성에 의해 동기가 부여되었으며, 저수준 세멘틱을 효과적으로 포착하는 CLIP 모델의 추가를 통해 전이 가능성을 높였습니다. 논문에서는 IQA와 VQA 지표의 심층 특성들 간의 상관관계를 분석하였습니다.

- **Performance Highlights**: 실험 결과 IC2VQA는 세 개의 블랙박스 VQA 모델에 대한 공격에서 높은 성공률을 달성하였습니다. 또한, 기존의 블랙박스 공격 전략들과 비교하여 같은 반복 횟수 및 공격 강도에서 공격 성공률에서의 우수성을 강조했습니다. 이 방법은 강력한 VQA 지표에 대한 심층 분석에 기여할 것으로 예상됩니다.



### BiDepth Multimodal Neural Network: Bidirectional Depth Deep Learning Arcitecture for Spatial-Temporal Prediction (https://arxiv.org/abs/2501.08411)
Comments:
          This paper has been submitted to Applied Intelligence for review

- **What's New**: 이 논문은 동적 시스템에서 공간-시간(Spatial-Temporal, ST) 정보를 정확하게 예측하는 데 중점을 둡니다. 제안된 BiDepth Multimodal Neural Network (BDMNN)은 쌍방향 깊이 조정을 사용하여 장기적 계절성(long-term seasonality)과 단기적인 변동(short-term fluctuations)을 모두 이해할 수 있게 합니다. 이는 기존의 통계적 접근 방식 및 전통적인 신경망(neural networks)의 한계를 극복하는 데 도움을 줍니다.

- **Technical Details**: BDMNN은 다양한 시간적 깊이(variable temporal depths)에서의 정보를 효과적으로 통합하며 공간적 맥락(spatial context)을 유지합니다. 이 모델은 장기적인 역사적 분석(comprehensive long-term historical analysis)과 단기적인 새로운 정보에 대한 빠른 반응(responsiveness) 간의 균형을 이루도록 설계되었습니다. 이러한 복잡한 ST 맥락에 적응하여 예측 정확성을 높입니다.

- **Performance Highlights**: 실제 공공 데이터를 사용한 사례 연구에서는 도시 교통 예측(urban traffic prediction)의 평균 제곱 오차(Mean Squared Error)를 12% 줄이는 등 예측 정확성이 크게 향상되었습니다. 또한, 강수 예측(rain precipitation forecasting)의 경우, 최신 기준(line benchmarks)에 비해 15%의 개선을 이루어냈습니다. 이러한 성과는 추가적인 계산 자원(computational resources)을 요구하지 않습니다.



### Addressing Quality Challenges in Deep Learning: The Role of MLOps and Domain Knowledg (https://arxiv.org/abs/2501.08402)
Comments:
          6 pages, 1 figure, accepted to the 4th International Conference on AI Engineering - Software Engineering for AI (CAIN)

- **What's New**: 이 연구는 딥러닝(DL) 시스템의 품질 속성, 특히 정확도(correctness)와 자원 효율성(resource efficiency) 향상을 위해 MLOps 실천 및 도메인 지식(domain knowledge)의 역할을 조명합니다. MLOps는 실험 추적(experiment tracking)과 자동 성능 모니터링(automatic performance monitoring)과 같은 요소들이 포함된 ML 주기를 지원하여 투명성과 재현성을 제공합니다. 본 연구는 DL 시스템에서 도메인 지식 통합의 필요성을 강조하며, 이를 통해 시스템의 질적 속성을 어떻게 개선할 수 있는지 구체적인 사례를 통해 설명합니다.

- **Technical Details**: 이 연구에서는 딥러닝 기반 시스템의 품질 속성을 개선하기 위해 Goal Question Metric (GQM) 접근법을 채택하고 MLOps와 도메인 지식의 활용을 분석합니다. 연구 질문(RQ)으로는 MLOps의 실천이 시스템 품질 속성에 미치는 영향과 도메인 지식의 적용이 DL 시스템의 정확도, 시간 효율성, 자원 활용도에 미치는 영향을 다룹니다. 또한, 체크해야 할 메트릭스(accuracy, latency, energy consumption)를 제시하며, 이를 통해 수집한 데이터에 대한 통계 분석을 수행합니다.

- **Performance Highlights**: 운영 중 발생한 시스템 성능 저하를 개선하기 위해 MLflow를 활용하여 실험 배포(experimental deployment)에서 수집한 데이터를 분석하고 피드백을 제공하였습니다. DL 모델링 단계에서의 변화가 품질 메트릭에 미치는 영향을 추적하는 데 성공하여, 자동 로그 기능을 통해 성능 개선을 위한 인사이트를 도출했습니다. 본 연구에서는 DL 모델을 최적화하고, 도메인 지식을 통합한 알고리즘 구현으로 품질 속성을 향상시키는 전략적 접근법을 제시합니다.



### Towards Best Practices for Open Datasets for LLM Training (https://arxiv.org/abs/2501.08365)
- **What's New**: 이번 논문에서는 인공지능(AI) 기업들이 저작권 소유자의 동의 없이 대규모 언어 모델(LLM)을 훈련시키고 있는 현황을 다룹니다. 이러한 행위의 허용 여부는 지역에 따라 다르며, EU와 일본 같은 국가에서는 특정 제한 하에 허용되고 있지만, 미국에서는 법적 경관이 모호합니다. 이로 인해 창작자들이 제기하는 저작권 소송이 증가하고 있으며, 이는 기업 및 공공기관이 훈련 데이터셋에 대한 정보를 축소하는 최근의 경향에 영향을 미치고 있습니다.

- **Technical Details**: 저작권 논란에도 불구하고, AI 모델의 투명성과 책임성을 저해하는 정보 공유 제한은 연구자, 감사인 및 피해자들이 AI 모델을 이해하는 데 필요한 정보에 접근하는 데 문제를 일으킵니다. 이러한 문제는 공개 접근(open access) 및 공공 도메인(public domain) 데이터를 기반으로 하는 언어 모델의 훈련으로 완화될 수 있지만, 현재 이에 대한 기술적 및 사회적 도전으로 인해 의미 있는 규모로 훈련된 모델은 없습니다. 데이터 조합에 필요한 부정확하고 불완전한 메타데이터, 물리적 기록의 디지털화(digitization) 비용과 복잡성, 빠르게 변화하는 환경에서 관련성과 책임성을 보장하기 위한 다양한 법적 및 기술적 기술들이 그 장애 요인입니다.

- **Performance Highlights**: AI 시스템이 책임감 있게 관리되고 큐레이션된 공개 라이센스 데이터로 훈련될 수 있는 미래를 구축하기 위해서는 법적, 기술적, 정책적 분야 간의 협력이 필수적입니다. 메타데이터 표준, 디지털화 및 개방성 문화 촉진에 대한 투자도 중요합니다. 이러한 통합적인 접근이 이루어져야만 AI 모델의 사용과 관련된 데이터 안전성과 책임이 보장될 수 있습니다.



### SCOT: Self-Supervised Contrastive Pretraining For Zero-Shot Compositional Retrieva (https://arxiv.org/abs/2501.08347)
Comments:
          Paper accepted at WACV 2025 in round 1

- **What's New**: 이 논문에서는 SCOT(Self-supervised COmpositional Training)이라는 새로운 zero-shot compositional pretraining 전략을 제안합니다. 이 방법은 기존의 레이블이 달린 triplet 데이터셋을 필요로 하지 않으며, 다양한 캡션이 달린 이미지 데이터셋을 활용하여 open-world 일반화 능력을 보여줍니다. SCOT는 성능 향상을 위해 대규모 contrastively-pretrained 비전-언어 모델의 시각적 및 텍스트 표현의 근접성을 활용합니다.

- **Technical Details**: SCOT는 입력으로 제공된 이미지와 그에 해당하는 캡션을 바탕으로 modification 텍스트를 생성하고, 이를 통해 CIR 모델을 훈련시킵니다. 이 과정에서 생성된 modification 텍스트가 참조 이미지와 조합되어 contrastive image retrieval loss를 최적화하는 방식입니다. 이러한 접근법은 기존의 inversion 기반 기법과는 달리 compositional 모델을 직접 훈련시키며 점차 업그레이드가 가능하여 다양한 도메인에 쉽게 적응할 수 있습니다.

- **Performance Highlights**: SCOT는 FashionIQ와 CIRR와 같은 표준 벤치마크에서 기존의 zero-shot compositional retrieval 방법들 뿐만 아니라 많은 fully-supervised 방법들보다 뛰어난 성능을 보여주었습니다. 이 접근법은 0-shot 조건에서 좋은 일반화 능력을 가지며, 자동 생성된 데이터로 인해 수작업으로 구성된 기존 데이터셋에 대한 의존도를 크게 줄일 수 있습니다.



### Operator Learning for Reconstructing Flow Fields from Sparse Measurements: an Energy Transformer Approach (https://arxiv.org/abs/2501.08339)
- **What's New**: 이번 논문에서는 유체 역학 분야에서 관측 데이터의 일부만으로 완전한 속도 필드를 복구하는 복원 문제를 해결하기 위해 Energy Transformer(ET)라는 새로운 operator learning(오퍼레이터 학습) 프레임워크를 제안합니다. 전통적인 알고리즘에 비해 ET는 더 빠른 학습 및 추론 속도를 제공하며, 노이즈가 있는 실험적인 측정에서도 높은 정확도를 자랑합니다. ET는 고전적인 Hopfield 네트워크에서 영감을 받아 개발된 Dense Associative Memory 구조를 기반으로 하고 있습니다.

- **Technical Details**: Energy Transformer는 주어진 관측 데이터를 바탕으로 에너지 함수를 최소화하여 모든 속성 데이터를 추정합니다. 여기서 관측 데이터는 x와 v의 쌍으로 구성되며, 우리는 관측 데이터와 전체 데이터 간의 맵을 학습하여 복원 문제를 해결합니다. 다양한 고유한 구조를 가진 operator learning 방법들과 비교하여 ET는 데이터가 불완전하고 노이즈가 섞인 상황에서도 잘 작동하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, ET는 불완전한 데이터(최대 90% 누락)에서도 복잡한 유동장(flow field)을 정확히 재구성할 수 있는 능력을 보여주었습니다. 다양한 유체 메커니즘 샘플을 이용한 검증에서, ET는 빠른 학습 속도와 GPU 상에서의 효율적인 추론을 제공하여 새로운 유체 역학 및 기계적 응용 방향을 제시합니다. 이 방법은 지질 물리학, 기상 예측 등 다양한 분야로 확장될 수 있는 가능성을 가집니다.



### MERaLiON-TextLLM: Cross-Lingual Understanding of Large Language Models in Chinese, Indonesian, Malay, and Singlish (https://arxiv.org/abs/2501.08335)
- **What's New**: 이번 연구에서는 MERaLiON-TextLLM이라는 오픈 소스 멀티링구얼 대형 언어 모델이 소개됩니다. 이 모델은 중국어, 인도네시아어, 말레이어 및 싱글리쉬의 이해 및 생성 능력을 개선하기 위해 설계되었습니다. 초기 모델은 Llama-3-8B-Base 기반으로, 지속적인 프리트레이닝(pre-training)과 가중치 병합(weight merging) 과정을 통해 성능을 최적화하였습니다. 이 모델은 공식 Llama-3 모델의 성능을 초과하는 결과를 보여주며, 추가 연구를 위한 자원으로 체크포인트를 제공합니다.

- **Technical Details**: MERaLiON-LLaMA-3-8B-Instruct 모델은 영어, 중국어, 인도네시아어를 중심으로 광범위하게 프리트레이닝되었습니다. 모델 훈련은 MaxText AI-Hypercomputer 플랫폼에서 진행되었으며, NVIDIA H100 GPU와 TPU v4-128 칩스를 사용하여 약 400 TFLOPS의 성능을 달성했습니다. 훈련 데이터는 영어에 38억 개의 토큰, 인도네시아어에 45억, 중국어에 42억 개가 할당되어 있으며, 이는 각 언어 간의 균형 잡힌 성능을 도모합니다.

- **Performance Highlights**: MERaLiON-LLaMA-3-8B-Instruct 모델은 Cross-MMLU 및 Cross-LogiQA 벤치마크에서 기존 Llama-3.1-8B-Instruct 모델을 뛰어넘는 성능을 보였습니다. 영어, 중국어, 인도네시아어에서의 점수는 각각 0.85, 0.69, 0.71로, 최신 다국어 모델 간의 성능 비교에서 우수한 결과를 확인했습니다. 다양한 벤치마크를 통해 MERaLiON 모델의 다국어 및 지침 이행(instruction-following) 능력이 입증되고 있습니다.



New uploads on arXiv(cs.LG)

### Improving Stability Estimates in Adversarial Explainable AI through Alternate Search Methods (https://arxiv.org/abs/2501.09006)
Comments:
          9 pages, 3 figures, 5 tables. arXiv admin note: text overlap with arXiv:2406.15839

- **What's New**: 이 논문은 머신 러닝 모델이 복잡해지면서 발생하는 불안정성의 정도를 탐구합니다. 기존 연구들이 불안정성의 존재를 확인했던 반면, 본 연구는 최소한의 변화(minimum viable perturbations)를 찾아내어 설명의 품질 변화를 측정할 수 있는 방법론을 제안합니다. 이 방식은 특정 임계값(threshold)을 설정하고 불안정성을 정량적으로 분석하게 해줍니다.

- **Technical Details**: 연구는 설명 가능한 인공지능(XAI) 모델에 대해 텍스트 기반 데이터에 집중합니다. 이로 인해 안정성(robustness)의 정의와 신뢰성 있는 해석 가능 여부를 평가합니다. 또한, 텍스트 데이터의 perturbations에 대해 유전자 알고리즘(genetic algorithm)을 활용해 최소 변화 수를 찾는 탐색 과정을 도입하여 더 넓은 탐색 공간을 조사합니다.

- **Performance Highlights**: 이 연구는 설명의 안정성을 분석하는 데 있어 효과적인 방법론을 제시합니다. 예를 들어, 두 개의 다른 perturbation을 비교했을 때, 하나는 더 적은 수의 변화로 유의미한 설명의 변화를 보였으며, 이는 불안정성이 심각하다는 것을 의미합니다. 이러한 비교는 XAI 방법들이 서로 어떻게 다르게 작용하는지에 대한 통찰을 제공합니다.



### VECT-GAN: A variationally encoded generative model for overcoming data scarcity in pharmaceutical scienc (https://arxiv.org/abs/2501.08995)
Comments:
          30 pages, 6 primary figures, 3 supplementary figures

- **What's New**: 이번 연구에서는 의약품 연구에서 데이터 부족 문제를 해결하기 위한 새로운 접근 방식으로 Variationally Encoded Conditional Tabular Generative Adversarial Network (VECT GAN) 모델을 개발했습니다. 이 모델은 작은 크기와 잡음이 많은 데이터셋을 증강하기 위해 설계되어, 기존의 데이터 기반 방법보다 더 효율적인 개발을 가능하게 합니다. 또한, 이 모델은 약물과 같은 소분자가 포함된 데이터셋에서도 쉽게 활용할 수 있도록 제작되었습니다.

- **Technical Details**: VECT GAN은 데이터 증강을 위한 새로운 생성 모델로, 작은 데이터셋에 대한 회귀 모델 개발 이전의 파이프라인을 통해 성능을 높입니다. 연구에서는 여섯 개의 제약 데이터셋에 이 파이프라인을 적용하였고, 그것이 다른 최첨단 테이블 생성 모델보다 안정적이고 심각한 성능 향상을 가져온다는 것을 입증했습니다. 이 모델은 또한 ChEMBL 데이터베이스에서 사전 훈련되어, 약물 유사 분자에 대한 일반화 능력을 향상시키는 지식 증류(knowledge distillation) 기법을 활용합니다.

- **Performance Highlights**: VECT GAN을 이용해 소규모 테이블 데이터셋에 대한 정규화의 힘을 시연하였으며, 그 결과 의약품 모델 개발에서 표준 관행으로 자리 잡을 가능성을 강조했습니다. 실험적 특성을 가진 새로운 의약용 점착성 폴리머를 개발하여 실제 적용 가능성을 보여주었습니다. 최종적으로, VECT GAN은 pip 패키지로도 제공되어 연구자들이 쉽게 접근할 수 있도록 하고 있습니다.



### Training-Aware Risk Control for Intensity Modulated Radiation Therapies Quality Assurance with Conformal Prediction (https://arxiv.org/abs/2501.08963)
Comments:
          2024 Machine Learning for Health Symposium

- **What's New**: 이 연구에서는 Intensity Modulated Radiation Therapy (IMRT) 품질 보증(QA) 과정을 개선하기 위해 새로운 접근법인 conformal prediction 방법론을 도입했습니다. 이를 통해 IMRT 계획의 필요성에 따라 측정할 계획을 잘 분류할 수 있는 메커니즘을 제공합니다. 연구진은 기존의 conformal risk control과 conformal training의 이점을 결합한 새로운 방법을 제안하였으며, 이를 통해 환자 안전성을 유지하면서 IMRT QA의 효율성을 높이는 것을 목표로 하였습니다.

- **Technical Details**: 제안된 방법은 gamma passing rate(GPR)에 기반한 의사결정 임계값을 설정하고, 임상 평가에 사용된 위험 함수들을 포함하여률, 완성도 향상 하는 데 중점을 두고 있습니다. 이를 통해 QA 과정에서 불필요한 측정의 수를 줄이면서도 높은 민감성(sensitivity)과 특이도(specificity)를 보장합니다. 또한, 제안된 방법은 기존 baselines와의 비교에서도 더 나은 성능을 보여주며 큰 신뢰 구간(confidence interval)을 생성하지 않는 장점을 가지고 있습니다.

- **Performance Highlights**: 연구의 결과는 실세계의 두 가지 IMRT 치료 계획 데이터 세트에 대한 평가를 통해 제안된 방법이 높은 민감성과 특이도를 달성할 수 있음을 보여줍니다. 이는 기존의 머신 러닝 모델들이 다루지 못했던 IMRT QA의 한계를 극복하는 데 기여할 것으로 기대됩니다. 결국, 연구진은 정확한 임상 의사결정을 지원하고 IMRT QA 과정의 업무 부담을 줄일 수 있는 잠재력을 가진 방법론을 개발하였음을 증명했습니다.



### Kolmogorov-Arnold Networks for Time Series Granger Causality Inferenc (https://arxiv.org/abs/2501.08958)
- **What's New**: 이번 논문에서는 Granger Causality Kolmogorov-Arnold Networks(GCKAN)라는 혁신적인 아키텍처를 소개합니다. GCKAN은 최근에 제안된 Kolmogorov-Arnold Networks(KAN)를 인과 추론(causal inference) 분야로 확장한 모델입니다. 이 모델은 KAN 계층에서 추출된 기본 가중치(base weights)를 활용하고, 희소성 유도 패널티(sparsity-inducing penalty)와 Ridge 정규화(ridge regularization)를 결합하여 시간 시계열로부터 Granger 인과관계를 추론합니다.

- **Technical Details**: GCKAN은 자동 시간 지연 선택(automatic time lag selection)을 가능하게 하여 시계열 데이터에서 Granger 인과관계를 추론하는 데 도움을 줍니다. 또한, 시간 반전된 Granger 인과관계를 활용하는 알고리즘을 제안하여 추론 정확도를 향상시키고자 합니다. 이 알고리즘은 원본 및 시간 반전된 시리즈에서 파생된 예측 및 희소성 유도 손실을 비교하여 더 높은 점수를 가진 인과관계를 자동으로 선택하거나 두 결과를 통합하여 잘못된 연결(spurious connectivity)을 완화합니다.

- **Performance Highlights**: 다양한 Lorenz-96, 유전자 조절 네트워크(gene regulatory networks), fMRI BOLD 신호 및 VAR 데이터 세트를 통해 진행된 광범위한 실험에서, 제안된 모델이 비선형(nonlinear), 고차원(high-dimensional), 제한된 샘플(lemon-sample) 시간 시계열에서의 Granger 인과관력 추론에서 최신 방법들(state-of-the-art methods)에 비해 경쟁력 있는 성능을 달성하는 것으로 나타났습니다.



### Disentangling Exploration of Large Language Models by Optimal Exploitation (https://arxiv.org/abs/2501.08925)
- **What's New**: 이 논문은 탐험(exploration)이 미래의 수익을 증가시킬 수 있도록 정보를 전달하는 것을 목표로 하는 새로운 평가 틀을 제안합니다. 기존 연구들은 주로 탐험과 활용(exploitation) 간의 균형을 중점적으로 다루어왔습니다. 그러나 이 연구는 탐험을 독립적인 목표로 분리하고, 여러 대형 언어 모델(LLM)이 상태 공간(state-space)을 탐색할 때의 성능을 평가합니다.

- **Technical Details**: 탐험 성능을 평가하기 위해, 연구진은 기존 보상의 결여를 탐험과 활용 구성요소로 분해하여 측정하는 방법을 도입했습니다. 이 접근법은 LLM의 탐험 전략을 체계적으로 검토할 수 있는 기반을 제공합니다. 실험 결과, 대부분의 LLM 모델들이 상태 공간을 충분히 탐색하는 데 어려움을 겪고 있으며, 모델 크기와 탐험 성능 사이에 긍정적인 상관관계가 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과는 대부분의 LLM이 독립된 탐험 작업에서 성과를 내는데 한계를 가지며, 이는 탐험과 활용의 트레이드오프를 강조하고 있습니다. 또한, 연구팀은 더 큰 모델이 더 나은 탐험 성능을 보여주는 경향이 있다는 점을 확인했습니다. 이 연구는 LLM의 탐험능력을 평가하고 향상시키기 위한 훌륭한 도구로 작용할 수 있는 분해 방식을 제공합니다.



### Modeling Melt Pool Features and Spatter Using Symbolic Regression and Machine Learning (https://arxiv.org/abs/2501.08922)
- **What's New**: 본 연구에서는 기계 학습(Machine Learning, ML)과 다항식 기호 회귀(Polynomial Symbolic Regression) 모델을 활용하여 레이저 파우더 베드 융합(LPBF) 공정에서 녹는 풀(melt pool)의 변화를 효과적으로 포착하고 통제할 수 있는 프레임워크를 개발했습니다. 이는 품질 관리를 개선하고 결함을 최소화하는 데 도움을 주며, 과거의 데이터셋을 기반으로 한 예측 모델을 통해 인쇄 품질의 일관성을 높이는데 기여하는 기술입니다.

- **Technical Details**: 연구에서는 281개의 프로세스 조건을 위한 데이터셋을 사용하여 녹는 풀의 치수(길이, 너비, 깊이) 및 형상(면적, 부피)과 같은 파라미터를 추출했습니다. 기계 학습 모델을 통해 95% 이상의 높은 결정 계수(R²)를 달성하였으며, 특히 ExtraTree 모델이 가장 높은 R² 값을 기록하였습니다. 로그 변환을 통해 스패터(spatter) 관련 변수의 예측 성능을 향상시킬 수 있었습니다.

- **Performance Highlights**: 이 연구는 LPBF 공정의 품질을 설계 최적화와 결함 예방을 통해 향상시키기 위한 혁신적인 접근 방식을 제시합니다. 특히 기계 학습을 통해 프로세스 조건과 녹는 풀의 차원 및 형상 간의 상관관계를 이해하고 해석 가능한 수학적 표현을 제공하며, 품질 보증 및 불량률 감소에 큰 도움을 줄 수 있습니다. 이러한 연구 결과는 여러 산업 분야에서 기계 학습이 품질 관리에 미치는 영향을 명확하게 드러냅니다.



### Projection Implicit Q-Learning with Support Constraint for Offline Reinforcement Learning (https://arxiv.org/abs/2501.08907)
- **What's New**: Proj-IQL은 Offline Reinforcement Learning에서 발생할 수 있는 OOD(Out-Of-Distribution) 행동으로 인한 과외삽(E extrapolation) 오류를 해결하기 위해 제안된 새로운 알고리즘입니다. 본 알고리즘은 정책 평가(Policy Evaluation)와 정책 개선(Policy Improvement) 단계에서 장기적인 관점에서 효과적인 학습을 구현하기 위해 다단계 적용과 지원 제약(Support Constraint)을 포함하고 있습니다. Proj-IQL은 이론적으로 정책 개선을 보장하고 더 엄격한 기준을 통해 우수한 행동을 보장합니다.

- **Technical Details**: Proj-IQL은 정책 평가 단계에서 벡터 투영(Vector Projection)을 통해 단일 단계 접근법을 다단계 접근법으로 일반화합니다. 이 단계에서 기존의 고정된 보수성 매개변수(Conservatism Parameter)를 대체하여 데이터셋에 특화된 조정 없이도 동작할 수 있도록 합니다. 또한 정책 개선 단계에서는 정책 평가 접근법과 더 잘 조화되는 지원 제약을 도입하여 더욱 효율적인 정책 개선이 이루어집니다.

- **Performance Highlights**: Proj-IQL은 D4RL 벤치마크에서 최첨단(State-of-the-art) 성능을 기록하였으며, 특히 어려운 탐색(구현) 작업에서 뛰어난 성과를 보여주었습니다. 알고리즘은 로코모션 작업과 같은 다양한 Gym-MuJoCo-v2 환경에서 우수한 결과를 도출하며, 기존 알고리즘들과 비교해 더 나은 성능을 입증했습니다. 이러한 결과는 Proj-IQL이 실제 환경에서의 적용 가능성을 더욱 높여주는 연구로 평가됩니다.



### A Two-Stage Pretraining-Finetuning Framework for Treatment Effect Estimation with Unmeasured Confounding (https://arxiv.org/abs/2501.08888)
Comments:
          KDD 25 Research Track

- **What's New**: 본 연구는 관측 데이터(observational data)와 소규모 무작위 대조 실험(randomized controlled trials, RCT) 데이터를 모두 활용하여 조건부 평균 처치 효과(conditional average treatment effect, CATE)를 추정하는 이단계(pretraining-finetuning) 프레임워크, TSPF를 제안합니다. 기존 연구들이 충족하기 어려운 강력한 무시 가능성 가정(strong ignorability assumption)을 벗어나, 실제 세계에서 자주 존재하는 측정되지 않은 혼란 변수(unmeasured confounding)에 대처하는 혁신적인 접근입니다.

- **Technical Details**: 제안된 접근 방식은 첫 번째 단계에서 대규모 관측 데이터를 통해 공변량(covariates)의 기초적 표현을 학습하고, 두 번째 단계에서 소규모 RCT 데이터를 사용하여 이 표현을 조정합니다. 이 과정에서 과적합(overfitting)을 방지하기 위해 모델 매개변수의 부분 초기화(partial parameter initialization) 방식을 도입해 소규모 RCT 데이터로 인한 문제를 최소화합니다.

- **Performance Highlights**: IHDP 및 Jobs 데이터셋에서의 광범위한 실험을 통해 우리의 방법론이 기존의 접근 방식보다 뛰어난 성능을 보임을 입증했습니다. 제안된 TSPF 프레임워크는 모델의 구조를 유연하게 조정할 수 있어, 제한된 RCT 데이터로 인한 과적합 문제를 완화하면서 CATE 추정의 정확성을 높이는 데 기여합니다.



### PAC Learnability of Scenario Decision-Making Algorithms: Necessary and Sufficient Conditions (https://arxiv.org/abs/2501.08887)
- **What's New**: 본 논문에서는 scenario decision-making 알고리즘의 PAC(Probably Approximately Correct) 속성을 연구합니다. 이 알고리즘은 안전 제약 조건을 위반할 위험이 매우 낮은 결정을 내릴 수 있는 능력을 갖추어야 합니다. 연구 결과, VC 차원(VC dimension)과 압축 크기(compression size)와 같은 기존에 제시된 충분 조건이 필수 조건이 아닐 수 있음을 보여줍니다.

- **Technical Details**: 리스크가 있는 결정making 문제를 다루면서, 논문에서는 결정 집합과 제약 집합을 정의합니다. 주어진 결정과 제약 조건에 대해, 확률 분포에 따라 결정의 위반 확률이 정의됩니다. 이러한 연구는 Pac 조건의 필요성을 탐구하고, VC 차원과 'no-free-lunch' 정리에 영감을 받아 새로운 수량을 도입하여 PAC 조건의 필요성을 논의합니다.

- **Performance Highlights**: 결정making 알고리즘의 보다 일반적인 구성요소와 경우의 수를 조사하며, 기존의 충분 조건들이 상황에 따라 반드시 필요한 것은 아니라고 결론짓습니다. 이 연구는 위험 관련 결정making 분야에 새로운 통찰을 제공하며, 특정 클래스의 문제들에 대한 PAC 조건의 강도와 연결될 수 있습니다. 최종적으로, 이를 통해 리스크를 줄이고 의사결정 프로세스를 향상시키기 위한 기초 연구를 제시합니다.



### Increasing Batch Size Improves Convergence of Stochastic Gradient Descent with Momentum (https://arxiv.org/abs/2501.08883)
Comments:
          22 pages

- **What's New**: 이번 논문에서는 Stochastic Gradient Descent with Momentum (SGDM)의 배치 크기 설정이 미치는 영향을 분석합니다. SGDM은 SGD에 모멘텀 항을 추가하여 정의되며, 딥 뉴럴 네트워크 훈련에 널리 사용됩니다. 연구 결과, 일정한 배치 크기를 사용하는 경우에는 항상 전체 그래디언트 노름의 기대값을 최소화하지 않는다는 것을 이론적으로 증명했습니다. 반면, 증가하는 배치 크기를 사용할 경우, 이 기대값을 최소화하며 수렴 성능이 향상된다는 점이 강조되었습니다.

- **Technical Details**: SGDM은 배치 크기와 관련된 성능에 강한 관계를 가지며, 모멘텀 버퍼를 포함하여 정의됩니다. 여기서 t-번째 SGDM 근사치인 𝜽t는 다음과 같이 표현됩니다: 𝜽t+1=𝜽t−ηt•𝒎t. 또한, 가변 모멘텀을 사용하는 다양한 SGDM 변종들이 소개되고 있으며, 이들 각각은 비결정적(nonconvex) 최적화에서의 성능을 개선하기 위해 설계되었습니다. 논문에서는 이러한 모멘텀 기법들이 실제 사용에서 형성되는 데이터를 기반으로 하는 경과들을 보여줍니다.

- **Performance Highlights**: 연구에서 제시된 수치 결과들은 증가하는 배치 크기를 가진 mini-batch SGDM이 일정한 배치 크기로 사용했을 때보다 더 빠른 수렴을 나타냄을 보여줍니다. 이는 딥 뉴럴 네트워크 훈련 시 성능 향상에 기여할 수 있는 중요한 통찰을 제공합니다. 또한, 여러 최적화 알고리즘의 파이썬 구현체(implementations)가 제공되어 연구의 실용성을 더욱 높이고 있습니다.



### Incrementally Learning Multiple Diverse Data Domains via Multi-Source Dynamic Expansion Mod (https://arxiv.org/abs/2501.08878)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 논문에서는 다양한 데이터 도메인에서의 지속적인 학습(Continual Learning) 문제를 다루기 위해 Multi-Source Dynamic Expansion Model (MSDEM)이라는 새로운 방법론을 소개합니다. 이 모델은 여러 개의 사전 훈련된(backbone) 모델을 기반으로 점진적으로 새로운 전문가를 구축하여 새로운 작업에 적응하게 설계되었습니다. 또한, 동적 확장 가능한 주의 메커니즘과 동적 그래프 가중 라우터를 제안하여 지식 전이를 극대화하고 일반화 성능을 향상시킵니다.

- **Technical Details**: MSDEM은 여러 출처에서 훈련된 백본 모델의 지식을 통합하여 강력한 일반화 표현을 제공하는 것을 목표로 합니다. 특히, 동적 확장 가능한 주의 메커니즘(Dynamic Expandable Attention Mechanism, DEAM)을 통해 여러 백본에서 추출한 표현의 중요성을 동적으로 평가하여 효과적으로 전이학습을 수행합니다. 또한, 동적 그래프 가중 라우터(Dynamic Graph Weight Router, DGWR) 전략을 통해 이전에 학습된 파라미터를 재사용하여 새로운 작업 학습을 돕습니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 방법론이 다양한 복잡한 데이터셋에서 최첨단 성능을 달성했음을 입증하였습니다. MSDEM은 데이터 도메인이 다양할 때에도 뛰어난 일반화 성능을 보이며, 적은 파라미터로도 효과적으로 학습할 수 있음을 보여줍니다. 마지막으로, 논문은 MSDEM 프레임워크와 함께 여러 중요한 기여를 통해 지속적인 학습의 새로운 가능성을 제시하고 있습니다.



### ARMOR: Shielding Unlearnable Examples against Data Augmentation (https://arxiv.org/abs/2501.08862)
- **What's New**: 이 논문에서는 데이터 증강(data augmentation)이라는 일반적인 데이터 전처리 기술이 개인 정보 보호(data privacy)에 미치는 잠재적인 위협을 밝힙니다. 특히, 데이터 증강이 적용될 경우, 학습 불가능한 예제(unlearnable examples)의 정확도가 21.3%에서 66.1%로 증가할 수 있다는 점을 보여줍니다. 이를 통해 데이터 증강이 개인 데이터 보호 샘플의 유용성을 심각하게 손상시킬 수 있음을 지적합니다.

- **Technical Details**: ARMOR라는 데이터 개인 정보 보호를 위한 방어 프레임워크를 제안합니다. 이 프레임워크는 공격자가 사용하는 데이터 증강 전략에 대한 정보가 없는 상황에서도 효과적으로 작동하도록 설계되었습니다. 비국소 모듈(non-local module)을 활용하여 서그잇 모델(surrogate model)을 구축하고, 각 클래스에 최적화된 데이터 증강 전략을 선택할 수 있는 서그잇 증강 선택 전략을 개발합니다.

- **Performance Highlights**: ARMOR는 4개의 데이터셋(CIFAR-10, CIFAR-100, Mini-ImageNet, VGG-Face)과 5개의 데이터 증강 방법을 사용한 실험에서 뛰어난 성능을 입증합니다. ARMOR는 기존 6가지 최첨단 방어 방법보다 효과적으로 개인 데이터의 학습 불가능성을 유지하여, 증강된 샘플로 훈련된 모델의 정확도를 최대 60%까지 감소시킬 수 있음을 확인했습니다. 또한, 다양한 데이터 증강 방법에 대해서도 견고한 방어 성능을 보여줍니다.



### Digital Phenotyping for Adolescent Mental Health: A Feasibility Study Employing Machine Learning to Predict Mental Health Risk From Active and Passive Smartphone Data (https://arxiv.org/abs/2501.08851)
- **What's New**: 이번 연구는 25세 이전에 대부분 발생하는 정신장애에 대한 예측 모델을 개발하기 위해, 스마트폰 데이터를 통합하여 활용하는 새로운 기계 학습 프레임워크를 사용하였습니다. 이는 청소년의 정신 건강 위험을 일찍 검출할 수 있는 가능성을 제시합니다. 특히, Mindcraft 앱을 통해 비임상 청소년의 다양한 정신장애에 대한 리스크를 예측하고자 하였습니다.

- **Technical Details**: 연구 참가자들은 런던의 세 개 학교에서 모집된 103명이며, 평균 연령은 16.1세입니다. 참가자들은 여러 설문지를 완료하고 14일 동안 Mindcraft 앱을 사용하여 능동적(active) 및 수동적(passive) 데이터를 수집했습니다. 기계 학습 모델은 contrastive pretraining을 통해 사용자 특성을 안정화하고, supervised fine-tuning을 통해 성능을 향상시켰습니다.

- **Performance Highlights**: 능동적 및 수동적 데이터를 통합한 결과, 단일 데이터 소스보다 우수한 성능을 보였습니다. SDQ-High 위험도에 대한 평균 균형 정확도는 0.71, 불면증에 대해서는 0.67, 자살 사고에 대해 0.77, 섭식 장애에 대해서는 0.70을 기록했습니다. 이러한 결과는 고급 기계 학습 기법과 스마트폰 데이터를 통합하여 정신 건강 위험을 예측할 수 있는 가능성을 보여줍니다.



### Graph Counterfactual Explainable AI via Latent Space Traversa (https://arxiv.org/abs/2501.08850)
Comments:
          Published at Northern Lights Deep Learning Conference 2025

- **What's New**: 이번 논문은 깊은 신경망의 예측을 설명하는 데 있어 카운터팩추얼 설명(counterfactual explanations)을 생성하는 새로운 방법을 제안합니다. 특히 구별되는 노드 구조와 그래프 분류기의 연속적 속성을 모두 고려하여 그래프 데이터에 대한 설명 가능성을 높이고자 했습니다. 이러한 방법은 케이스 별로 순열 동치(permutation equivariance) 그래프 변분 오토인코더(variational autoencoder)를 활용하여, 예측 결과를 변경하는 최적의 대안 입력을 찾는 데 중점을 두고 있습니다.

- **Technical Details**: 카운터팩추얼 설명을 생성하기 위해, 그래프 분류기(classifier)의 분류 경계를 횡단하면서 잠재 공간(latent space)을 탐색하는 방법을 채택했습니다. 이 과정에서, 순열 동치 그래프 변분 오토인코더(PEGVAE)를 통해 그래프의 의미론적 잠재 표현(semantic latent representation)을 구축하며, 이는 입력 그래프와 의미상 유사한 그래프를 생성할 수 있게 해줍니다. 이러한 접근 방식은 주어진 그래프의 클래스 라벨을 최소한으로 수정하는 그래프를 반환하는 데 초점을 두고 있습니다.

- **Performance Highlights**: 세 가지 그래프 데이터셋에 대한 실험 결과, 제안된 모델은 기존 방법들(base lines)보다 일관되게 높은 성능을 보여주었습니다. 또한, 성능면에서도 더 강건한 결과를 나타냈으며, 다양한 데이터에 대해 효과적으로 카운터팩추얼 설명을 생성할 수 있는 능력을 입증하였습니다. 이로 인해 AI 시스템의 설명 가능성을 크게 향상시켰다는 점에서 의미가 큽니다.



### A Closer Look at the Learnability of Out-of-Distribution (OOD) Detection (https://arxiv.org/abs/2501.08821)
- **What's New**: 이번 논문에서는 머신러닝 알고리즘이 배포 시 'out-of-distribution' (OOD) 데이터에 직면하는 문제를 고찰하며, 기존 OOD 탐지 이론의 비관적인 결과와 실제 성능 간의 간극을 메우려 합니다. PAC 학습 이론을 바탕으로 균일한 학습 가능성과 비균일한 학습 가능성을 구분하고, OOD 탐지가 가능할 조건을 명확히 하였습니다. 이와 함께, 비균일 학습 가능성이 여러 부정적인 결과를 긍정적으로 전환할 수 있다는 점을 증명합니다.

- **Technical Details**: 논문에서는 OOD 탐지의 학습 가능성을 두 가지로 나누어 분석합니다: 균일 학습 가능성(uniform learnability)과 비균일 학습 가능성(non-uniform learnability)입니다. OOD 탐지를 위한 구체적인 학습 알고리즘을 제공하고, 샘플 복잡성(sample complexity)에 대한 분석도 포함되어 있습니다. 또한, OOD 및 ID 분포의 서포트(support) 조건에 따라 OOD 탐지가 학습 가능성이 어떻게 달라지는지 확인합니다.

- **Performance Highlights**: 결과적으로, OOD 탐지는 이론적으로 불가능하지 않으며, 실제 방법들이 성공적인 이유는 데이터 분포와 OOD 탐지 벤치마크가 특정한 좋은 속성을 지니기 때문일 가능성이 있음을 제시합니다. 논문에서는 ID 및 OOD 분포의 서포트 간 거리가 일정 이상일 경우를 다루는 비균일 OOD 탐지기를, 모든 ID 분포가 볼록한 서포트를 가질 경우 비균일 학습기를 제시합니다. 이처럼, OOD 탐지 이론에서는 더욱 깊은 연구가 필요함을 강조하고 있습니다.



### Deep learning for temporal super-resolution 4D Flow MRI (https://arxiv.org/abs/2501.08780)
Comments:
          12 pages, 8 figures

- **What's New**: 본 논문에서는 4D 흐름 자기공명영상(4D Flow MRI)에서 시간 해상도를 향상시키기 위한 잔여 신경망(residual network)을 구현하고 평가하는 내용을 다루고 있습니다. 특히, 기존의 공간 초해상도 네트워크인 4DFlowNet을 재설계하여 시간 업샘플링을 위한 새로운 아키텍처로 발전시켰습니다. 이를 통해 다양한 환자 지형에서 환자 맞춤형 재훈련 없이도 빠른 흐름 현상을 정확하게 포착할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 시간 초해상도 네트워크는 주어진 영역에서 발생하는 흐름 벡터의 정밀한 복원을 목표로 하며, 입력 및 출력 레이어와 중심 업샘플링 레이어의 주요 구성이 수정되었습니다. 원래의 공간 네트워크는 정적 시간 지점에서 3D 패치를 처리하며, 3D 합성곱(convolution)을 사용하여 정보를 전파합니다. 개선된 네트워크는 선형 업샘플링을 통해 시간 차원에서만 해상도를 향상시키도록 설계되었습니다.

- **Performance Highlights**: 테스트 결과, unseen in-silico 환경에서 평균 절대 오차(MAE)가 1.0 cm/s로 뛰어난 성능을 보였으며, 이는 전통적인 선형 및 sinc 보간법 대비 개선된 결과입니다. 더욱이, low-resolution in-vivo 데이터로부터 높은 해상도의 시간 정보가 합성되었으며, 고유한 흐름 프레임에서 강한 상관관계가 관찰되었습니다. 이러한 결과는 데이터 기반 신경망을 활용하여 4D Flow MRI의 시간 초해상도를 활용할 수 있는 잠재력을 강조하고 있습니다.



### Networked Agents in the Dark: Team Value Learning under Partial Observability (https://arxiv.org/abs/2501.08778)
Comments:
          18 pages, 7 figures, 5 tables. Accepted as supplemental material at Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025), Detroit, Michigan, USA, May 19 - 23, 2025, IFAAMAS

- **What's New**: 이 논문은 새로운 협력적 다중 에이전트 강화 학습(Cooperative Multi-Agent Reinforcement Learning, MARL) 접근법인 DNA-MARL을 제안합니다. 기존 방법들은 전체 상태 정보나 공동 관측에 의존했으나, 본 연구에서는 제한된 관측(Partial Observability) 하에 상호 목표를 달성하는 방법을 학습해야 합니다. 또한, 에이전트들은 개별 보상을 수집하고, 지역 통신을 통해 팀 가치 함수를 근사하는 방법을 사용하여 협력적인 행동을 구현합니다.

- **Technical Details**: 우리는 네트워크 동적 부분 관찰 마르코프 게임(Networked Dynamic Partially Observable Markov Game, ND-POMG) 프레임워크를 도입하여 에이전트가 스위칭 토폴로지 통신망에서 소통하는 방법을 설명합니다. DNA-MARL은 지역 통신을 위한 합의 메커니즘(consensus mechanism)과 지역 계산을 위한 경량 하강(Gradient Descent) 방법을 사용합니다. 이 접근법은 에이전트들이 정보를 공유하여 팀 가치를 합의하는 것을 통해 협력적인 가치 함수 학습을 촉진합니다.

- **Performance Highlights**: DNA-MARL은 기존의 협력적 학습 방법보다 뛰어난 성과를 보여주었으며, 다양한 벤치마크 MARL 시나리오를 통해 평가되었습니다. 이 방식은 개인의 데이터 프라이버시를 보호하면서도 효과적인 협력을 통해 더 나은 결과를 이끌어낼 수 있도록 설계되었습니다. 본 논문은 DNA-MARL이 다른 비중심화 훈련 및 실행 시스템보다 우수한 성능을 보임을 입증합니다.



### MeshMask: Physics-Based Simulations with Masked Graph Neural Networks (https://arxiv.org/abs/2501.08738)
- **What's New**: 본 논문에서는 그래프 신경망(GNN)을 위한 새로운 마스킹 프리트레이닝 기법을 소개합니다. 이 기법은 컴퓨터 유체 역학(CFD) 문제에 적용되어, 훈련 중 최대 40%의 입력 메시 노드를 무작위로 마스킹하여 복잡한 유체 역학에 대한 강건한 표현을 학습하도록 모델을 유도합니다. 이를 비대칭 인코더-디코더 아키텍처 및 게이티드 멀티 레이어 퍼셉트론과 결합하여 성능을 향상시킵니다.

- **Technical Details**: 이론적 배경에서는 메시를 무방향 그래프 G=(V,E)로 고려하며, 각각의 노드(V)와 엣지(E)의 속성이 정의됩니다. 제안된 인코더-디코더 아키텍처는 두 개의 GNN을 차례로 쌓아 AutoEncoder 형태로 구성되며, 이들 각각은 인코드-프로세스-디코드 방식의 다중 그리드 GNN입니다. 또한, 특성 벡터는 압력, 속도와 같은 노드 속성을 포함하고, 경계 조건을 나타내기 위한 원-핫 벡터 및 전체적인 정보로 결합됩니다.

- **Performance Highlights**: 제안된 기법은 250,000개 이상의 노드를 가진 3D intracranial aneurysm 시뮬레이션을 포함하는 7개의 CFD 데이터셋에서 최첨단 성능을 달성합니다. 이전 최고 모델과 비교하여 장기 예측 정확도를 최대 60% 개선함으로써 훈련 효율성도 증가시킵니다. 또한, 다양한 데이터셋에서 동시에 효과적인 프리트레이닝을 가능하게 하여 새로운 작업에서 높은 성능을 달성하는 데 필요한 시간과 데이터를 크게 줄이는 방법을 제공합니다.



### Resource-Constrained Federated Continual Learning: What Does Matter? (https://arxiv.org/abs/2501.08737)
Comments:
          arXiv admin note: text overlap with arXiv:2303.11165 by other authors

- **What's New**: 이번 논문은 연속적인 데이터 스트림에서 프라이버시를 유지하며 모델을 훈련하는 연합 지속 학습(Federated Continual Learning, FCL)의 실제 적용 가능성을 분석합니다. 기존의 FCL 연구는 데이터 프라이버시와 이전 데이터 접근에 초점을 맞추었으나, 실제 시나리오에서는 장치의 메모리, 계산 리소스, 레이블 비율과 같은 제약이 존재합니다. 저자들은 다양한 리소스 제약 조건에서 기존 FCL 방법의 성능을 대규모 벤치마크를 통해 평가하였습니다.

- **Technical Details**: FCL을 연구하기 위해 연구팀은 클래스 증분 학습(Class-IL) 및 도메인 증분 학습(Domain-IL) 두 가지 시나리오에서 6개의 대규모 데이터셋을 이용하여 1,000시간 이상의 GPU 리소스를 소모하며 실험을 진행하였습니다. 실험 결과, 기존의 FCL 방법들이 제한된 리소스 환경에서는 기대하는 성능을 전혀 얻지 못함을 발견했습니다. 특히, 메모리 버퍼, 계산 예산, 레이블 비율의 세 가지 주요 리소스 제약이 성능에 큰 영향을 끼치는 것으로 나타났습니다.

- **Performance Highlights**: 저자들은 각기 다른 리소스 제약 조건 하에서도 기존의 FCL 방법들을 철저히 평가했습니다. 이 분석 결과, 기존 방법들은 메모리 제약이 있는 환경에서는 잘 동작하지만, 계산 예산과 희소 레이블 데이터가 제한된 시나리오에서는 실용성이 떨어짐을 확인했습니다. 이러한 발견은 FCL의 미래 연구 방향에 중요한 통찰을 제공하며, 리소스 제약이 있는 설정에서의 FCL 기술 향상의 필요성을 강조합니다.



### GRAPPA - A Hybrid Graph Neural Network for Predicting Pure Component Vapor Pressures (https://arxiv.org/abs/2501.08729)
Comments:
          38 pages, 12 figures

- **What's New**: 이 논문에서는 GRAPPA라는 하이브리드 그래프 신경망을 개발하여 순수 화합물의 증기 압력을 예측하는 방법을 제안합니다. GRAPPA는 오직 분자 구조만을 입력으로 받아 어떤 유기 화합물의 증기 압력 곡선을 예측할 수 있습니다. 이 모델은 메시지 패싱 단계에 대한 그래프 주의 네트워크, 장거리 상호작용을 캡처하는 풀링 함수, Antoine 방정식의 컴포넌트 별 파라미터를 제공하는 예측 헤드로 구성됩니다.

- **Technical Details**: 이 모델은 약 25,000개의 실험적 증기 압력 데이터에 대해 훈련되고 평가되었습니다. GRAPPA는 실험적 데이터가 없는 미지의 컴포넌트에 대해서도 우수한 예측 정확도를 보이며, 기존의 그룹 기여 방법(Group Contribution Methods)이나 다른 머신러닝 접근법들에 비해 뛰어난 정확성과 적용 가능성을 자랑합니다. 모든 모델 및 코드는 완전히 공개되었으며, GRAPPA는 인터랙티브 웹사이트를 통해 직접 사용할 수 있습니다.

- **Performance Highlights**: GRAPPA는 미지의 컴포넌트에 대한 예측 정확성에서 우수한 성능을 나타내며, 현재까지의 최첨단 방법들을 초월한 결과를 보여줍니다. 특히, GRAPPA는 기존의 수작업으로 데이터가 필요한 방법들과 달리, 데이터 기반의 예측을 가능하게 하여 실험적인 어려움을 극복합니다. 이로 인해 화학 과정 설계에 있어 중요한 역할을 할 것으로 예상됩니다.



### Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models (https://arxiv.org/abs/2501.08727)
- **What's New**: 본 논문에서는 텍스트-이미지 모델의 파라미터 효율적인 파인튜닝(PEFT) 기법에 대한 새로운 접근법인 변형 저랭크 적응(Transformed Low-Rank Adaptation, TLoRA)을 제안합니다. TLoRA는 사전 훈련된 가중치를 저랭크 목표 가중치에 가깝게 정렬하기 위해 학습 가능한 변형을 사용합니다. 이를 통해 남은 가중치는 더 작은 근사 오차를 가진 compact하고 parameter-efficient한 구조로 근사될 수 있습니다.

- **Technical Details**: 제안하는 TLoRA 방법은 변형(Transform) 및 잔여(Residual) 적응의 두 부분으로 구성되어 있습니다. 변형 적응은 사전 훈련된 가중치에 선형 변형을 적용하여 더 낮은 랭크의 파인튜닝 프로세스로 투영합니다. 잔여 적응은 컴팩트한 구조를 사용하여 잔여 부분을 효율적으로 근사하도록 설계되었습니다. 또한, 이 적응을 효과적으로 모수화하기 위해 텐서 분해(tensor decomposition) 기법을 채택하였습니다.

- **Performance Highlights**: 실험 결과, TLoRA는 주제 주도 생성(subject-driven generation) 및 제어 가능한 생성(controllable generation) 작업에서 LoRA 및 여러 기준선 모델에 비해 더 나은 성능과 파라미터 효율성을 보여주었습니다. 특히, SDXL 모델을 fine-tuning 하는 과정에서 단 0.4M의 파라미터로도 만족스러운 성능을 달성할 수 있었습니다. 이와 같은 결과는 변형 및 잔여 적응의 결합이 실제로 성능을 향상시킬 수 있음을 입증합니다.



### Disentangled Interleaving Variational Encoding (https://arxiv.org/abs/2501.08710)
- **What's New**: 이 논문에서는 여러 작업을 동시에 수행하는 다중 작업 학습(multi-task learning)에서 나타나는 상충하는 목표를 해결하기 위한 Deep Disentangled Interleaving Variational Encoding (DeepDIVE) 모델을 제안합니다. 이 연구는 잠재 공간(latent space)에서 마진(marginal) 및 조건부(conditional) 확률 분포를 분리하는 방법론을 기반으로 하며, 이를 통해 모델의 해석력을 높이고 재구성 및 예측 목표를 결합하여 효율성을 증명합니다. 또한, Naïve Bayes를 사용하여 분리(disentanglement)를 위한 손실 함수(loss function)를 수학적으로 도출하였습니다.

- **Technical Details**: DeepDIVE는 변형 오토인코더(Variational Autoencoder, VAE) 아키텍처를 기반으로 하여 원래 입력을 마진 및 조건부 확률 분포로 분리하는 접근 방식을 개발하였습니다. 이 과정에서 베이즈 정리를 적용하여 입력을 보다 관리하기 쉬운 구성 요소로 분해하고, 크로스 어텐션(cross-attention) 메커니즘을 통해 이러한 특성을 통합하는 방법을 제안합니다. 또한, 교차 엔트로피 손실(cross entropy loss) 최소화를 통해 Kullback-Leibler divergence에 대한 상한을 설정하여 DeepDIVE의 수렴(convergence) 근거를 제공합니다.

- **Performance Highlights**: 실험 결과, DeepDIVE는 원래의 VAE보다 더 나은 예측 정확도를 보여주었으며, 최신 기법들과 비슷한 성과를 냈습니다. 이 모델은 전통적인 예측 방식에서의 불확실성과 변동성을 더 잘 반영할 수 있는 가능성을 보여줍니다. 공공 데이터셋을 사용한 검증에서도 원본 입력의 분리가 효과적으로 이루어졌으며, 다양한 예측 작업에서 우수한 성능을 발휘했습니다.



### Diagonal Over-parameterization in Reproducing Kernel Hilbert Spaces as an Adaptive Feature Model: Generalization and Adaptivity (https://arxiv.org/abs/2501.08679)
Comments:
          arXiv admin note: text overlap with arXiv:2409.00894

- **What's New**: 본 논문은 훈련 중에 커널 고유값(eigenvalues)과 출력 계수(output coefficients)를 동시에 학습하는 대각형 적응 커널(diagonal adaptive kernel) 모델을 소개합니다. 이 모델은 기존의 고정 커널(fixed-kernel) 방법들과 달리 진리 함수(truth function)의 구조에 적응하여 일반화를 현저히 개선합니다. 특히 초기 커널이 목표와 일치하지 않는 경우에도 효과적입니다.

- **Technical Details**: 저자들은 훈련 중 키와 진리 함수의 계수 간의 정렬(alignment)을 중점적으로 논의하며, 특징 맵(feature map)의 대각형 매개변수화(parameterization)를 도입하여 특징 매개변수(feature parameters)와 출력 계수에 대해 경량화된 경량화로 해결할 수 있는 기회를 제시합니다. 특히, eigenvalues를 동적으로 조정함으로써 모델이 향상된 일반화 능력을 갖는다는 사실을 증명하고 있습니다.

- **Performance Highlights**: 연구 결과, 대각형 적응 커널 방법은 초기 고유값 선택의 영향을 줄이고 압도적으로 우수한 일반화 성능을 보임을 입증했습니다. 특히, 올바른 조기 중단(early-stopping)을 통해 오라클 수렴률(oracle convergence rate)에 근접하는 성과를 도출하였습니다. 추가적인 깊이를 고려할 때, 깊이를 늘리는 것이 일반화 능력을 더 향상시킬 수 있음을 보여주었습니다.



### Investigating Parameter-Efficiency of Hybrid QuGANs Based on Geometric Properties of Generated Sea Route Graphs (https://arxiv.org/abs/2501.08678)
- **What's New**: 이 연구에서는 QuGANs(Quantum-classical Hybrid Generative Adversarial Networks)를 사용하여 실제 세계의 선박 경로 데이터를 기반으로 인공적으로 경로 그래프를 생성합니다. 연구의 초점은 QuGANs가 이 데이터를 얼마나 잘 학습하고 재현할 수 있는지를 비교하는 것입니다. 또한 이 연구는 전통적인 GANs(Generative Adversarial Networks)와의 효율성 비교를 통해 QuGANs의 잠재력을 조명합니다.

- **Technical Details**: QuGAN는 GAN의 구조를 기반으로 하여 적어도 하나의 구성 요소가 양자 컴퓨팅(QC)을 사용하여 구현됩니다. 본 연구에서는 GAN의 생성기(generator)를 QC 시뮬레이터에서 구현하고, 그것이 고전적인 차별자(discriminator)와 상호작용하는 방식으로 진행됩니다. 이렇게 함으로써 선박 경로가 가지는 기본적인 기하학적 구조를 인식하고 이를 생성 데이터에 재현할 수 있는 효율성을 평가합니다.

- **Performance Highlights**: 연구 결과, QuGANs는 기본적인 기하학적 속성과 분포를 빠르게 학습하고 표현할 수 있지만, 샘플링된 데이터에 변화를 도입하는 데 어려움을 겪는 것으로 나타났습니다. 특정 파라미터 수를 기준으로 볼 때, 일부 QuGANs는 고전적인 GANs와 유사한 품질의 결과를 보였으며, 이는 QC가 다양한 생성적 인공지능 분야에서 활용될 가능성을 시사합니다.



### SPEQ: Stabilization Phases for Efficient Q-Learning in High Update-To-Data Ratio Reinforcement Learning (https://arxiv.org/abs/2501.08669)
- **What's New**: 이번 논문에서는 Deep Reinforcement Learning (심층 강화 학습)에서 샘플 효율성을 향상시키기 위한 새로운 접근 방식을 소개합니다. 특히, Update-To-Data (UTD) 비율을 최적화하여 컴퓨팅 효율성을 개선하는 방법론을 설명합니다. 이를 통해 기존의 DroQ 알고리즘보다 56% 적은 그래디언트 업데이트 수와 50%적은 훈련 시간을 요구하면서도 비슷한 성능을 달성하였습니다.

- **Technical Details**: 제안된 SPEQ (Stabilization Phases for Efficient Q-Learning) 방법은 두 가지 단계로 나뉘어 있습니다: 낮은 UTD 비율의 온라인 훈련 단계와 높은 UTD 비율의 오프라인 안정화 단계. 안정화 단계에서는 새로운 환경 상호작용을 수집하지 않고도 Q-함수를 튜닝하여 리플레이 버퍼의 효율성을 향상시킵니다. 이 방법은 Q-값의 편향(bias)을 줄이고, 더 효과적으로 저장된 샘플을 활용할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과에 따르면, SPEQ는 최신 고 UTD 비율 알고리즘과 유사한 성능을 발휘하면서도 적은 수의 그래디언트 단계와 낮은 컴퓨팅 자원을 요구합니다. 이는 SPEQ가 더 나은 샘플 효율성을 유지하면서도 실용적인 컴퓨터 비용을 제공하는 것을 의미합니다. 이러한 결과는 SPEQ의 전반적인 컴퓨팅 효율성이 기존 기술보다 뛰어남을 입증합니다.



### Fine-grained Spatio-temporal Event Prediction with Self-adaptive Anchor Graph (https://arxiv.org/abs/2501.08653)
Comments:
          Accepted to SIAM International Conference on Data Mining 2025 (SDM'25)

- **What's New**: 이번 연구에서는 정교한 이벤트 예측을 위한 새로운 Graph Spatio-Temporal Point Process (GSTPP) 모델을 제안합니다. GSTPP는 neural Ordinary Differential Equations (ODEs)를 사용하여 공간 지역의 동적 상태를 공동으로 모델링하는 인코더-디코더 아키텍처를 채택합니다. Self-Adaptive Anchor Graph (SAAG)를 기반으로 한 이 모델은 복잡한 공간 이벤트 패턴을 학습하는 능력을 향상시킵니다.

- **Technical Details**: GSTPP 모델은 광범위한 지역에서 발생하는 이벤트의 동적을 반영하기 위해 지역별 상태와 전역 상태를 공동으로 모델링합니다. SAAG를 사용하여 공간 의존성과 지역 간의 상호작용을 효과적으로 캡처하며, 이는 모델의 지역적 특성을 학습하는 과정을 돕습니다. 또한 Location-aware Graph Convolutional Network (L-GCN)와 Relative Location Encoder (RLE)와 같은 여러 서브모듈을 통해 성능을 더욱 강화합니다.

- **Performance Highlights**: 제안된 GSTPP 모델은 기존의 spatio-temporal event prediction 접근 방식에 비해 정확도를 상당히 향상시킵니다. 실험 결과는 GSTPP가 복잡한 이벤트 동적을 더욱 효과적으로 모델링할 수 있음을 보여줍니다. 이러한 결과는 GSTPP가 spatio-temporal 이벤트 예측 분야에서 최신 기술의 중요한 발전을 의미합니다.



### Quantum Reservoir Computing and Risk Bounds (https://arxiv.org/abs/2501.08640)
- **What's New**: 본 연구에서는 Rademacher complexity를 사용하여 여러 종류의 양자 저수지(quantum reservoirs)에 대한 일반화 오류(generalisation errors)를 경계할 수 있는 방법을 제안합니다. 특히 두 가지 양자 저수지 클래스에 대한 구체적이고 매개변수에 의존적인 경계를 제공합니다. 이를 통해 경계가 큐비트(qubit)의 수가 증가함에 따라 어떻게 변화하는지를 분석했습니다.

- **Technical Details**: 양자 저수지와 읽기 함수(readout function)에 대한 매개변수의 명시적 의존성을 통해 어느 정도 일반화 오류를 조절할 수 있는 가능성을 찾아냈습니다. 일반화 경계는 훈련 샘플 수의 증가에 따라 수렴(converge)하는 경향이 있으며, 이는 다항적인 특성을 가진 읽기 함수가 포함된 클래스에 적용됩니다. 그러나 큐비트의 수(n)가 증가함에 따라 경계는 지수적으로(scale exponentially) 증가한다는 점이 주목할 만합니다.

- **Performance Highlights**: 제안된 경계는 양자 동역학(quantum dynamics)과 읽기 함수에 대한 몇 가지 가설을 충족하는 다른 저수지 클래스에도 적용 가능한 장점을 가지고 있습니다. 이 연구 결과는 양자 시스템에서 일반화 오류를 보다 효과적으로 통제할 수 있는 방향성을 제시합니다.



### SWSC: Shared Weight for Similar Channel in LLM (https://arxiv.org/abs/2501.08631)
Comments:
          5pages, 3 figures, work in progress

- **What's New**: 본 연구에서는 SWSC라는 새로운 LLM 압축 기법을 제안합니다. 이 방법은 유사 채널의 가중치를 공유하는 개념을 바탕으로 하여 모델의 파라미터 수를 획기적으로 줄입니다. K-Means 클러스터링 알고리즘을 사용하여 유사 벡터들을 군집화하고, 각 군집에서 대표 벡터를 선택하여 여러 벡터를 대체합니다. 이렇게 하여 LLM의 압축 및 배포 효율성을 높이며, 다양한 장치에서 모델을 쉽게 사용할 수 있도록 합니다.

- **Technical Details**: SWSC는 K-Means 클러스터링을 기반으로 하여 모델 가중치를 채널별로 군집화하는 방식으로 작동합니다. 각 군집에서 유사한 벡터를 그룹화하고, 대표 벡터를 선택하여 군집 내 모든 벡터를 대체합니다. 압축 전후의 가중치 오류 값을 특이값 분해(Singular Value Decomposition, SVD)를 통해 계산하고, 중요 특이값 및 벡터를 유지하여 정확도를 보정합니다. 이러한 기술들은 압축된 LLM의 성능을 보장하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 SWSC 방법은 저정밀 조건에서도 LLM의 성능이 크게 저하되지 않는다는 것을 보여줍니다. 기존의 압축 기법과 비교했을 때, SWSC는 모델의 압축 효율성을 대폭 향상시키면서도 성능을 유지하는 데 성공했습니다. 이 새로운 접근법은 LLM의 배포를 용이하게 하여 인공지능 기술의 발전과 보급에 기여할 것으로 기대됩니다.



### Transformer-based Multivariate Time Series Anomaly Localization (https://arxiv.org/abs/2501.08628)
- **What's New**: 본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터의 비지도 이상 진단에 대한 새로운 Transformer 기반 모델을 소개합니다. 특히, 자가 주의(self-attention) 메커니즘의 학습 행동을 분석하여 이상 탐지와 더불어 위치 추적(localization) 성능을 향상시키는 데 중점을 두고 있습니다. 논문의 주요 기여는 이상 위치 추적 문제를 시간 단계, 윈도우, 세그먼트 기반의 세 가지 단계로 구성하는 것입니다. 이러한 접근 방식은 Space-Time Anomaly Score (STAS)라는 새로운 메트릭을 개발하여 개별 이상 행동과 시계열 간의 의존성을 포착하는 데 기여합니다.

- **Technical Details**: 이 논문은 Multivariate Time Series (MTS)의 이상 진단을 위한 방법론을 제시하며, 이를 통해 MTS의 복잡한 동역학을 포착합니다. STAS는 Transformer의 잠재 표현(latent representations)과 시공간 통계 모델 간의 연결에서 영감을 받아 설계되었습니다. 또한, 이상을 중심으로 한 통계적 특징을 분석하는 Statistical Feature Anomaly Score (SFAS)가 STAS를 보완하며, 두 메트릭의 결합은 잘못된 경고를 줄이는 데 도움을 줄 수 있습니다. 이 모델은 실제 및 합성 데이터셋에 대한 실험을 통해 기존의 최첨단 방법들에 대한 우수성을 입증하고 있습니다.

- **Performance Highlights**: 실제 및 합성 데이터셋을 통한 실험 결과, 제안된 모델은 이상 탐지 및 위치 추적 작업 모두에서 기존의 최신 방법들보다 뛰어난 성능을 보였습니다. 특히, STAS와 SFAS를 통한 이상 진단 성능은 높은 신뢰성을 지니며, 복잡한 CPS 환경에서 운영의 안전성을 증대시키는 데 기여할 것으로 기대됩니다. 다변량 시계열의 복잡성을 다룰 수 있는 이 모델은 향후 관련 연구 및 실제 응용 분야에서 중요한 역할을 할 것입니다.



### CT-PatchTST: Channel-Time Patch Time-Series Transformer for Long-Term Renewable Energy Forecasting (https://arxiv.org/abs/2501.08620)
- **What's New**: 이 연구에서는 Channel-Time Patch Time-Series Transformer (CT-PatchTST)라는 고급 딥러닝 모델을 개발하여 태양광 및 풍력 에너지 시스템의 전력 출력을 예측합니다. 기존의 Patch Time-Series Transformer(PatchTST) 모델이 채널 독립적 접근 방식을 사용하였으나, 채널 간 관계를 간과하는 문제를 해결하기 위해 CT-PatchTST를 제안하였습니다. 이 모델은 채널 간 정보를 처리하는 방식을 개선하여 더 정밀한 에너지 예측을 제공합니다.

- **Technical Details**: CT-PatchTST는 두 가지 주의 메커니즘인 채널 주의 및 시간 주의를 통합하여 채널 간 의존성을 효과적으로 포착합니다. 이 모델은 복잡한 다변량 시계열 데이터를 처리할 수 있어 기존의 형태보다 뛰어난 예측 성능을 보입니다. 연구는 덴마크의 연안 및 해상 풍력 데이터와 태양광 데이터에 대한 실험을 통해 CT-PatchTST의 성능을 철저히 분석하였습니다.

- **Performance Highlights**: CT-PatchTST는 여러 예측 시나리오에서 기존의 최첨단 모델보다 우수한 성능을 발휘하여 신뢰성과 견고성을 강조합니다. 이 연구는 풍력과 태양광 시스템의 예측 가능성을 개선하여 에너지 그리드와의 원활한 통합을 지원하고, 재생 가능 에너지 기술의 널리 보급을 위한 중요한 기여를 합니다.



### RLHS: Mitigating Misalignment in RLHF with Hindsight Simulation (https://arxiv.org/abs/2501.08617)
- **What's New**: 이번 연구는 기존의 Reinforcement Learning from Human Feedback (RLHF) 방법의 한계를 극복하기 위해 Hindsight Feedback에 기초한 새로운 알고리즘인 Reinforcement Learning from Hindsight Simulation (RLHS)를 소개합니다. RLHF는 즉각적인 피드백을 통해 모델을 최적화하려 하였으나, 이로 인해 모델의 행동이 인간의 가치와 잘 align되지 않을 수 있다는 문제를 지적합니다. RLHS는 시뮬레이션된 결과를 통해 피드백을 생성하여 이러한 미스알라인먼트를 효과적으로 완화할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 Partial Observable Markov Decision Process (POMDP)로 모델링된 인간 의사결정 문제를 다릅니다. RLHS 방법은 AI 시스템이 가능한 인간 행동을 시뮬레이션하고 그 결과로부터 피드백을 받는 구조로, 이는 기존의 RLHF와는 다른 접근 방식입니다. RLHS는 Proximal Policy Optimization (PPO)와 Direct Preference Optimization (DPO)과 같은 선호 최적화 방법에 적용되어, 모델의 misalignment를 줄이는 데 효과적임을 실증적으로 입증했습니다.

- **Performance Highlights**: 인간 사용자 연구 결과, RLHS는 RLHF보다 사용자 목표 달성 및 만족도에서 일관되게 우수한 성능을 보였습니다. 시뮬레이션된 피드백만을 사용하였음에도 불구하고, RLHS는 사용자의 진정한 유틸리티를 향상시키며 잘못된 정보를 기반으로 한 결정을 줄이는 데 기여했습니다. 이 결과들은 장기적인 결과에 초점을 맞추는 것이 RLHF의 misalignment을 완화하는 데 중요함을 강조합니다.



### Towards Aligned Data Forgetting via Twin Machine Unlearning (https://arxiv.org/abs/2501.08615)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2408.11433

- **What's New**: 최근의 개인정보 보호 규정들은 기계의 비학습(machinery unlearning) 기술의 발전을 이끌어 내고 있습니다. 이 기술은 훈련된 모델이 특정 훈련 데이터를 효율적으로 잊을 수 있도록 하여 데이터의 삭제 요청을 충족시킵니다. 기존의 비학습 방법들은 데이터 잊기의 개념을 정확도를 0으로 만드는 것으로 이해했지만, 이 논문은 비학습된 모델이 원래 모델과 정렬되도록 하는 것을 목표로 설정합니다.

- **Technical Details**: 본 연구에서 제안하는 방법은 Twin Machine Unlearning (TMU)으로, 이는 원래 비학습 문제에 대해 쌍둥이 비학습 문제를 정의합니다. 이를 통해 원래 문제에 대해 훈련된 일반화-레이블 예측기를 이용하여 데이터를 정렬하는 방식으로 작동합니다. 두 개의 서브문제를 설정하여 각각 정답 레이블과 차별화된 특징을 확보하며, 이를 통해 필요한 비학습 데이터를 정렬합니다.

- **Performance Highlights**: Empirical experiments에 따르면, 제안된 TMU 접근법은 비학습 모델과 금 모델 간의 정렬을 크게 개선하였습니다. 모델의 전반적인 성능을 유지하면서도 삭제해야 할 데이터셋에 대한 정확도를 일부만 낮추는 방식으로, 전체적인 정확도를 유지하는 성과를 보여주었습니다. 이와 같은 접근은 데이터 삭제 요청을 보다 효율적으로 처리할 수 있도록 합니다.



### Neural Risk-sensitive Satisficing in Contextual Bandits (https://arxiv.org/abs/2501.08612)
Comments:
          Accepted by AROB-ISBC 2025

- **What's New**: 이 논문에서는 Neural Risk-sensitive Satisficing (NeuralRS) 알고리즘을 제안하여 기존의 Regional Linear Risk-sensitive Satisficing (RegLinRS) 알고리즘의 한계를 극복하고자 합니다. NeuralRS는 Neural Network (NN)을 기능 근사기로 사용하여 비선형 관계를 처리할 수 있게 합니다. 이로 인해 맞춤형 추천 시스템의 성능을 크게 향상시킬 것으로 기대됩니다.

- **Technical Details**: Contextual bandit 문제는 행동의 기대 보상을 추정하는 작업을 추상화하는 프레임워크로, 각 Timestep에서 d차원 특징 벡터를 받아 행동을 선택합니다. RegLinRS는 선형 관계를 기반으로 보상을 예측하지만, NeuralRS는 신경망을 통해 비선형 관계를 적절히 처리하여 다양한 데이터 환경에서 더 나은 적응성을 보여줍니다. 이를 통해 각 특성과 기대 보상 간의 복잡한 기능을 표현할 수 있게 됩니다.

- **Performance Highlights**: NeuralRS 알고리즘은 RegLinRS보다 실세계 데이터셋에서 유연하게 적응할 수 있는 능력을 보여줄 예정입니다. 또한 NeuralRS는 전통적인 NN 기반 알고리즘들과 비교했을 때 동등하거나 더 높은 성능을 달성할 것으로 기대됩니다. 이러한 멀티모달 환경에서의 성능 비교는 특히 추천 시스템의 문제 해결에 중요한 기여를 할 것입니다.



### Molecular Graph Contrastive Learning with Line Graph (https://arxiv.org/abs/2501.08589)
- **What's New**: 이 논문은 그래프 대비 학습(graph contrastive learning, GCL)에서의 분자 속성 예측 및 약물 설계의 레이블 부족 문제를 해결하기 위해 새로운 방법인 LEMON을 제안합니다. 특히, 기존의 데이터 변형 기반 방법은 분자 의미를 왜곡하는 문제를 가지고 있으며, 도메인 지식 기반 방법은 일반화 능력이 제한적입니다. LEMON은 선형 그래프(line graph)와 화합물의 그래프를 비유하고, 이러한 구조를 통해 분자 의미를 온전히 인코딩할 수 있도록 설계되었습니다.

- **Technical Details**: LEMON은 분자 그래프를 해당 선형 그래프와 대조하여 분자 의미를 잃지 않고 인코딩할 수 있도록 돕습니다. 이 방법은 이중 나선 그래프 인코더를 활용하여, 두 그래프 간의 정보를 일관되게 유지하고 과도한 평활화(over-smoothing) 문제를 해결합니다. 또한, 엣지 속성 융합(edge attribute fusion) 및 새로운 로컬 대조 손실(local contrastive loss)을 도입하여 정보 전송을 강화하고 어려운 네거티브 샘플(hard negative samples)에 효과적으로 대응합니다.

- **Performance Highlights**: LEMON은 ZINC15에서 200만 개의 분자 그래프에 대한 사전 학습을 통해 분자 속성 예측에 관한 8개의 벤치마크에서 뛰어난 성능을 보였습니다. 특히, 평균 ROC-AUC와 평균 순위에서 최고 성과를 기록하며, 기존의 최첨단(State-of-the-art) 방법들과 비교했을 때 그 우수성을 입증하였습니다. 이 연구는 분자 의미를 완전하게 탐구하는 데 있어 최초의 시도를 나타내며, 어려운 네거티브 샘플을 다룰 수 있는 능력을 추가하여 LEMON을 더욱 향상시켰습니다.



### Normalize Then Propagate: Efficient Homophilous Regularization for Few-shot Semi-Supervised Node Classification (https://arxiv.org/abs/2501.08581)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문은 Graph Neural Networks (GNNs)의 몇 가지 제한점을 분석하고, 라벨 부족 문제를 해결하기 위한 새로운 알고리즘 NormProp을 제안합니다. NormProp은 레이블이 없는 노드의 동질성(homophily) 가정을 이용하여 추가적인 감독 신호(supersvision signals)를 생성함으로써 일반화(generalization) 능력을 향상시킵니다. 이 방법은 GNN을 훈련시키는 데 필요한 라벨 수를 줄일 수 있는 효율적인 방법입니다.

- **Technical Details**: NormProp 모델은 노드 표현 벡터의 방향과 유클리디안 노름(Euclidean norm)을 분리하여 메시지 전파(message passing) 과정에서 클래스 정보와 집합의 일관성을 효과적으로 캡처합니다. 여기서 주요 작업은 '정규화 후 전파(Normalize then Propagate)'로, 노드 특징을 단위 초구(unit hypersphere)로 매핑하고 저역필터(low-pass filter)를 적용합니다. 또한, 노드의 표현 벡터에 대한 유클리디안 노름의 이론적 상한을 분석하고, 레이블이 없는 노드의 이웃 일관성을 제약하는 동질적 정규화(homophilous regularization)를 제안합니다.

- **Performance Highlights**: NormProp은 라벨이 적은 노드 분류 작업에서 주목할 만한 성능을 보이며, 낮은 계산 복잡성으로 최신 방법들에 비해 우수한 결과를 제공합니다. 광범위한 실험을 통해 NormProp의 효과성과 효율성이 검증되었으며, 특히 레이블이 적은 상황에서도 뛰어난 일반화 능력을 보여줍니다. 이러한 성과는 기존 GNN 모델들이 겪는 오버피팅(overfitting) 문제를 해결하는 데 기여합니다.



### DNMDR: Dynamic Networks and Multi-view Drug Representations for Safe Medication Recommendation (https://arxiv.org/abs/2501.08572)
- **What's New**: 이 논문은 동적 네트워크와 다중 뷰 약물 표현을 통합한 새로운 약물 추천(Medication Recommendation, MR) 방법인 DNMDR를 제안합니다. 기존 MR 시스템의 한계를 극복하기 위해, 시간적 EHR(전자 건강 기록) 데이터를 기반으로 한 동적 이질 네트워크의 가중 스냅샷 시퀀스를 구성하였고, 모든 동적 네트워크는 환자의 다양한 의료 사건에서 구조적 상관관계와 시간적 의존성을 학습하도록 공동 훈련되었습니다. 이렇게 구성된 방법론은 환자 표현을 개선하는 데 기여하며, 약물 간 상호 작용을 줄이고 보다 안전한 약물 추천을 가능하게 합니다.

- **Technical Details**: 연구에 사용된 DNMDR 모델은 동적 네트워크 및 다중 뷰 약물 그래프를 기반으로 하여, 임상 이벤트의 구조적 관계와 역사적 건강 상태의 시간적 의존성을 동시적으로 탐색합니다. 또한, 약물의 화학적 구조와 상호작용 정보를 통합하여 정확하고 안전한 약물 조합 추천을 지원합니다. 이전의 의료 경험 및 기존 데이터베이스에서 얻은 알려진 약물 간 상호작용 정보를 활용하여, 약물 이체의 안전성을 보장합니다.

- **Performance Highlights**: 실제 EHR 데이터셋에서 실시된 광범위한 실험 결과, DNMDR 방법은 다양한 평가 지표(예: PRAUC, Jaccard, DDI 비율)에서 기존의 최신 기준 모델보다 큰 폭으로 성능을 개선하는 것으로 나타났습니다. 이로 인해, DNMDR은 의료 분야에서 약물 추천 시스템의 실제 적용 가능성을 높이는 결과를 도출하였습니다.



### Adaptive Sampled Softmax with Inverted Multi-Index: Methods, Theory and Applications (https://arxiv.org/abs/2501.08563)
Comments:
          40 pages

- **What's New**: 이번 연구에서 저자들은 MIDX Sampler라는 새로운 샘플링 전략을 제안하여 다중 클래스 분류에서 softmax 함수의 계산 비용을 줄이는 동시에 정확도와 샘플링 효율성을 향상시킵니다. 이 방법은 특히 수백만 또는 수십억 개의 클래스가 있는 대규모 문제 영역에서 유용합니다. MIDX Sampler는 세분화된 다항 확률에 대해 softmax 확률을 분해하여 효율성을 높이고, uniform distribution을 통해 더욱 간소화된 계산을 제공합니다.

- **Technical Details**: MIDX Sampler는 Inverted Multi-Index 접근법 기반의 적응형 샘플링 전략입니다. 이를 통해 softmax 확률을 여러 다항 확률로 나누어 클래스 수가 아닌 코드워드 수에 대한 시간 복잡도를 줄이고, 마지막 단계에서는 쿼리 특화된 잔여 확률을 단순한 균등 분포로 대체하여 효율성을 한층 강화합니다. 이 접근법은 샘플링 편향, 기울기 편향, 수렴 속도 및 일반화 오류 경계와 같은 중요한 이슈들을 체계적으로 해결할 수 있는 기반이 됩니다.

- **Performance Highlights**: 실험 결과, MIDX Sampler는 기존 방법보다 뛰어난 성능과 효율성을 제공함을 입증했습니다. 다양한 대규모 언어 모델, 순차 추천 시스템 및 극단적인 다중 클래스 분류 과제에서 성능이 향상되었습니다. 저자들은 이 샘플러의 구현을 GitHub에서 공개하고 있으며, 이는 실용적인 응용에서의 효과성과 확장성을 보여줍니다.



### Homophily-aware Heterogeneous Graph Contrastive Learning (https://arxiv.org/abs/2501.08538)
- **What's New**: 이번 연구에서는 이분화 그래프 비지도 학습의 새로운 접근법인 HGMS(Heterogeneous Graph Contrastive Learning Framework)를 제안합니다. 이 프레임워크는 연결 강도와 다각적 자기 표현을 활용하여 동질적 노드 표현을 학습합니다. 특히, 이 연구는 데이터 내의 동질성을 높이고 비동질?(heterophily) 구조의 영향을 줄이기 위한 방법에 초점을 두고 있습니다.

- **Technical Details**: HGMS는 이분화된 그래프에서 노드 연결의 강도를 활용하고, 메타 경로 기반의 이웃 노드 쌍 간에 관찰되는 동질성을 분석합니다. 연구진은 이 문맥을 기반으로 한 이분화 엣지 드롭 전략을 통해 강화된 시각에서 동질성을 확보하고, 다각적 자기 표현 학습 방법을 도입해 노드 간의 동질성을 추론합니다. 해결된 자기 표현 행렬은 추가적인 강화된 시각을 제공하며, 대조 손실에서 가짜 부정 예시를 식별하는 데 사용됩니다.

- **Performance Highlights**: 여섯 개의 공개 데이터셋을 대상으로 한 폭넓은 실험 결과는 HGMS의 우수성을 뒷받침합니다. 이 연구는 동질성이 HG 프리 트레이닝(HGP) 방법의 성능에 미치는 영향을 입증하며, 제안된 모델이 다양한 하위 작업에서 뛰어난 성능을 보여줍니다. 연구의 주요 발견은 HGP 모델의 표현력을 높이는 것과 관련이 있습니다.



### Mitigating Domain Shift in Federated Learning via Intra- and Inter-Domain Prototypes (https://arxiv.org/abs/2501.08521)
Comments:
          13 pages, 9 figures, 10 tables

- **What's New**: 이 논문에서는 기존의 federated prototype learning 방법이 intra-domain (내부 도메인) 특성을 무시했다는 점을 지적하고, 새로운 방법론인 I²PFL를 제안합니다. I²PFL는 intra-domain (내부 도메인)과 inter-domain (외부 도메인) 프로토타입을 통합하여 도메인 변화(domain shift)를 완화하고, 여러 도메인에서 일반화된 글로벌 모델을 학습할 수 있도록 합니다. 이는 실제 문제에서 분포가 다른 클라이언트들이 협력할 수 있는 환경을 제공합니다.

- **Technical Details**: 제안된 I²PFL 방법에서는 MixUp 기반의 증강된 프로토타입을 통해 타 지역(domain) 내 특성을 aligning (정렬)하는 방식을 사용합니다. 이를 통해 각 지역의 다양성을 포착하고, 로컬 특징의 일반화가 강화됩니다. 또한, 클라이언트 간 도메인 왜곡(domain skew)을 감소시키기 위해 inter-domain 프로토타입을 재가중치하는 메커니즘을 도입하여, 보다 일반화된 프로토타입을 생성합니다.

- **Performance Highlights**: I²PFL 방법은 Digits, Office-10, PACS 데이터셋에 대한 광범위한 실험을 통해 다른 기존 방법론에 비해 우수한 성능을 보였습니다. 특히, 이 방법은 도메인이 이질적인 환경에서도 효과적으로 작동하여 federated learning의 응용 가능성을 확대합니다. 이러한 결과는 I²PFL가 다양한 실제 응용 분야에서도 신뢰할 수 있는 선택이 될 수 있음을 나타냅니다.



### Learning Hyperplane Tree: A Piecewise Linear and Fully Interpretable Decision-making Framework (https://arxiv.org/abs/2501.08515)
- **What's New**: 이번 논문에서는 Learning Hyperplane Tree (LHT)라는 새로운 트리 기반 모델을 소개합니다. LHT는 여러 공공 데이터셋에서 기존의 트리 모델보다 우수한 분류 성능을 보여줍니다. 이 모델은 데이터를 여러 개의 hyperplane(하이퍼플레인)을 사용하여 나누는 간단하고 효율적인 구조를 가지고 있습니다. 각 단계에서의 분리가 완벽하지는 않지만, LHT는 점진적으로 샘플의 구분을 개선합니다.

- **Technical Details**: LHT는 이진 분류 문제에 대해 설계되며, 각 클래스에 대해 별도의 LHT를 구성할 수 있습니다. 이를 통해 타겟 클래스의 샘플과 비타겟 클래스의 샘플을 구분하며, 각 분기 블록은 hyperplane과 membership function(멤버십 함수)을 정의하여 데이터를 분리합니다. LHT는 피처 선택(feature selection) 및 블록 분할(block splitting) 두 단계를 통해 작동하며, 각 피처의 유용성을 평가하여 낮은 유용성을 가진 피처는 필터링합니다.

- **Performance Highlights**: LHT는 피스와이즈(linear piecewise) 구조 덕분에 고속 추론 속도를 자랑합니다. 또한, 공공 데이터셋에서 기존의 SOTA 방법보다 높은 테스트 정확도를 보여줍니다. LHT의 투명한 구조 덕분에 각 피처의 결정 과정에서의 기여도를 명확하게 관찰할 수 있어, 높은 해석 가능성을 제공합니다.



### Score-based 3D molecule generation with neural fields (https://arxiv.org/abs/2501.08508)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 3D 분자의 새로운 표현을 제안합니다. 이는 연속적인 원자 밀도 필드를 기반으로 하며, 이를 통해 무작위 3D 분자 생성을 위한 새로운 모델인 FuncMol을 제안합니다. FuncMol은 조건부 신경망을 사용하여 분자 필드를 잠재 코드로 인코딩하고, Langevin MCMC (마르코프 체인 몬테 카를로) 방식을 활용하여 노이즈가 있는 코드를 샘플링합니다. 이 모델은 구조에 대한 가정을 하지 않고 모든 원자를 포함하는 3D 분자 생성을 효과적으로 수행합니다.

- **Technical Details**: 이 연구는 분자를 원자 점유율을 인코딩하는 연속 함수로 표현하는 새로운 방법을 제시합니다. 이 방법은 분자를 나타내는데 더 자연스럽고, 신경망을 파라미터로 하는 분자 점유율 필드를 사용합니다. FuncMol은 세 단계를 통해 샘플링을 수행하며, 이는 노이즈 코드를 샘플링하는 'walk' 단계, 'jump' 단계에서 청정 코드를 추정하고, 마지막으로 이를 분자로 변환하는 'decode' 단계로 구성됩니다. 이러한 접근 방식은 특히 메모리 사용량이 적고 스케일 조정이 용이합니다.

- **Performance Highlights**: FuncMol은 drug-like 분자 데이터셋 GEOM-drugs에서 경쟁력 있는 성능을 보입니다. 샘플링 속도는 기존 방법보다 최소한 한 자리 수 폭으로 빨라졌으며, 다양한 표준과 새로운 메트릭을 사용하여 생성 품질을 측정합니다. 또한, FuncMol은 CREMP 데이터셋에서 더 큰 3D 분자로의 확장성도 입증하였습니다. 이 모델은 다양한 분자 설계 문제에 적용할 수 있으며, 다양한 원자 밀도, 표면, 약리작용기, 분자 오르빗 등 여러 가지 파라미터를 표현할 수 있습니다.



### Exploring the Efficacy of Meta-Learning: Unveiling Superior Data Diversity Utilization of MAML Over Pre-training (https://arxiv.org/abs/2501.08506)
- **What's New**: 이 연구는 대규모 비전 모델의 성능을 좌우하는 데이터셋의 다양한 특성 중에서도 특히 데이터 다양성(data diversity)이 모델 성능에 미치는 영향을 탐구합니다. 기존 연구들이 주로 데이터 양(data size)이나 모델의 크기와 복잡성에 집중했던 것과는 달리, 본 연구에서는 데이터 다양성을 중요한 요소로 제시하고 있습니다.

- **Technical Details**: 연구에서는 Task2Vec라는 메트릭을 사용하여 데이터셋의 내재적인 다양성을 측정합니다. 이 메트릭은 작업(task)을 확률 분포로 간주하고, 서로 다른 작업의 Task2Vec 임베딩(task embedding) 간의 평균 거리 평균을 계산하는 방식으로 작동합니다. 우리의 분석은 12개의 인기 있는 시각 데이터셋과 다양한 모델 구성에서 메타 러닝 (meta-learning) 기법을 연구하였습니다.

- **Performance Highlights**: 실험 결과, 데이터셋 다양성과 모델 성능 사이에 양의 상관관계가 존재하는 것으로 나타났습니다. 특히 higher-order MAML 모델은 데이터 다양성과 모델 성능 간의 더 강력한 상관관계를 보였으며, 연구에서 관찰된 R-squared 값은 0.4에 이르렀습니다. 이는 데이터 다양성이 모델의 성능을 향상시키는 중요한 요소임을 시사합니다.



### Time series forecasting for multidimensional telemetry data using GAN and BiLSTM in a Digital Twin (https://arxiv.org/abs/2501.08464)
- **What's New**: 최근 디지털 트윈(Digital Twin) 분야에서 예측 가능한 서비스의 필요성이 증가하고 있습니다. 제안된 연구에서는 생성적 적대 신경망(Generative Adversarial Networks, GAN)과 양방향 장기 단기 기억 네트워크(BiLSTM)의 결합을 통해 시계열 예측을 개선하고자 합니다. 이를 통해 디지털 트윈의 데이터와 통합하여 더 나은 성능 예측이 가능하도록 하는 방법이 제안됩니다.

- **Technical Details**: 본 연구에서는 GAN을 통해 데이터 분포를 추출하고, BiLSTM을 활용하여 시간적 행동을 학습하여 시계열 데이터를 예측하는 데이터 기반 아키텍처를 제안했습니다. GAN은 데이터 시퀀스의 생성과 확장을 가능하게 하며, BiLSTM은 여러 특성(feature)을 반영하여 예측의 정확성을 높입니다. 이 방법은 기존의 GAN 기반 시계열 연구에서 발견된 한계를 극복하기 위한 것입니다.

- **Performance Highlights**: 연구 결과, 제안된 GAN + BiLSTM 접근법이 기존의 시간 시리즈 예측 방법들보다 더 정확한 결과를 도출할 것으로 기대됩니다. 이 방법은 데이터 세트의 환경적 행동을 지속적으로 업데이트하여 실시간 예측이 가능하게 합니다. 또한, 다양한 비선형 환경에서도 유연하게 적용될 수 있어 디지털 트윈 생성에 효과적입니다.



### Keras Sig: Efficient Path Signature Computation on GPU in Keras 3 (https://arxiv.org/abs/2501.08455)
- **What's New**: 이 논문에서는 심층 학습 애플리케이션을 위해 경로 서명을 계산하도록 설계된 고성능 Python 라이브러리인 Keras Sig를 소개합니다. Keras 3로 완전히 구축된 Keras Sig는 PyTorch, JAX 및 TensorFlow와 같은 널리 사용되는 심층 학습 백엔드와의 매끄러운 통합을 활용합니다. Keras Sig는 최신 기술들을 활용하여 GPU 병렬 처리 이점을 극대화하여 기존 방법보다 훈련 시간을 55% 단축하고, 직접 서명 계산에서 5배에서 10배의 성능 향상을 이루어냈습니다.

- **Technical Details**: Keras Sig는 고수준 텐서 연산을 활용하여 저수준 C++ 코드 대신 높은 성능을 발휘합니다. 이로 인해 딥러닝 라이브러리에서 일반적으로 발생하는 버전 관리 및 호환성 문제를 크게 줄이고, 다양한 하드웨어 구성에서 우수하거나 동 등급의 성능을 발휘합니다. 본 라이브러리는 GPU 최적화 구현을 통해 연산을 재구성하여 병렬성을 극대화하며, 이는 특히 대규모 입력 시퀀스에서 효율적으로 확장됩니다.

- **Performance Highlights**: Keras Sig는 GPU 하드웨어에서 기존 라이브러리보다 뛰어난 성능을 발휘하며, CPU에서도 비슷한 성능을 유지합니다. 기존의 CPU 기반 서명 변환 라이브러리에 비해 GPU 지원을 제공하고, 저수준 최적화를 필요로 하지 않으므로 머신러닝 워크플로우에 서명 변환을 통합하는 데 강력하고 미래지향적인 솔루션을 제공합니다. 본 연구에서 Keras Sig는 TensorFlow 래퍼를 성공적으로 대체하여 성능을 향상시켰습니다.



### Physics-informed neural networks for phase-resolved data assimilation and prediction of nonlinear ocean waves (https://arxiv.org/abs/2501.08430)
Comments:
          22 pages, 12 Figures, preprint

- **What's New**: 본 연구에서는 전통적인 파 예측 방법의 한계를 극복하기 위해 물리 정보 기반 신경망(Physics-Informed Neural Networks, PINNs)을 활용한 새로운 해결 방법을 제안합니다. 이 방법은 잠재 흐름 이론(Potential Flow Theory, PFT) 솔루션을 신경망으로 파라미터화하여 파 데이터의 동화 및 예측을 수치적으로 효율적으로 수행할 수 있도록 합니다. PINN 프레임워크는 실험적인 데이터와의 비교를 통해 검증되었습니다.

- **Technical Details**: 이 연구에서는 PINNs를 사용하여 비선형 파의 동작을 모델링하고 예측하는 방식으로, 전통적인 PFT 솔버의 속도 제한 문제를 해결합니다. PINN을 통해 표면 높이 측정치만으로도 유체의 전체 비선형 속도 잠재력을 추론할 수 있어 실험적으로 측정하기 어려운 유체 속도를 계산할 수 있습니다. 이 접근법은 단순화된 모델이 포착하기 어려운 강한 비선형성을 잘 포착할 수 있다는 장점이 있습니다.

- **Performance Highlights**: 제안된 방법은 실험적인 파 실험 데이터와의 비교를 통해 정확하게 불규칙하고 비선형적인 파의 표면 역학을 예측하는 성능을 입증하였습니다. 결과적으로 PINNs를 활용하여 구축된 모델은 기존 방법에 비해 계산적으로 저렴하면서도 효과적인 데이터 동화 기술 개발에 기여할 수 있는 가능성을 보여줍니다.



### Physics-Informed Latent Neural Operator for Real-time Predictions of Complex Physical Systems (https://arxiv.org/abs/2501.08428)
- **What's New**: 본 논문에서는 PI-Latent-NO라는 물리 기반의 잠재 연산자 학습 프레임워크를 소개합니다. 이 프레임워크는 두 개의 DeepONet을 결합하여 높은 차원의 문제를 효율적으로 처리하며, 물리 법칙을 학습 과정에 직접 통합함으로써 데이터 샘플 수를 대폭 감소시킬 수 있습니다. 기존 방법들과 달리 이 접근 방식은 엔드-투-엔드 최적화를 가능하게 하여 데이터 부족 문제를 해결합니다.

- **Technical Details**: PI-Latent-NO는 두 개의 연결된 DeepONet으로 구성되어 있습니다. 첫 번째 네트워크는 운동의 저차원 잠재 표현을 학습하는 Latent-DeepONet이며, 두 번째 네트워크는 이 잠재 표현을 원래의 물리적 공간으로 재구성하는 Reconstruction-DeepONet입니다. 이 구조는 문제의 차원 확장에 대해 약선형 스케일링을 가능하게 하여 기존 방법들보다 뛰어난 성능을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 고차원 매개변수 PDE 문제에 대한 효과를 입증했으며, 기존 물리 기반 DeepONet 모델에 비해 예측 정확도 및 계산 효율성이 크게 향상되었습니다. 또한, 이 접근 방식은 복잡한 물리 시스템의 실시간 예측 가능성을 제시하며, 기후 모델링과 공학 설계 최적화 등 다양한 분야에 응용될 수 있는 잠재력을 보여줍니다.



### Causal vs. Anticausal merging of predictors (https://arxiv.org/abs/2501.08426)
Comments:
          Presented at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문에서는 동일한 데이터를 사용하여 인과적(causal)과 비인과적(anticausal) 방향에서 예측기를 병합하는 과정에서 발생하는 차이를 연구합니다. 특히 두 개의 연속 변수를 사용하여 하나의 이진 변수를 목표로 삼는 간단한 모델을 통해 비대칭(asymmetries) 현상을 조사합니다. Causal Maximum Entropy (CMAXENT)를 통해 예측기를 병합하는데, 다른 병합 방법을 사용하더라도 이와 유사한 차이가 발생할 것으로 기대하고 있습니다.

- **Technical Details**: CMAXENT 솔루션은 모든 쌍변량 분포를 관찰할 경우 인과적 방향에서는 로지스틱 회귀(logistic regression)로, 비인과적 방향에서는 선형 판별 분석(Linear Discriminant Analysis, LDA)으로 축소됩니다. 이 연구는 예측기를 병합할 때 인과적인 가정이 어떻게 결과에 영향을 미치는지를 분석하며, 특히 모든 변수를 관찰하지 않을 때의 차이를 강조합니다. 의료 분야에서 질병의 유무를 예측하는 데 중요한 인과적 가정의 포함 필요성을 설명합니다.

- **Performance Highlights**: 연구 결과, CMAXENT가 이진 목표 변수와 연속 공변수를 사용할 때 인과적과 비인과적 방향에서 각각 로지스틱 회귀와 LDA로 패러데스가 내려집니다. 또한, OOV(Out-Of-Variable) 일반화에 대한 함의도 분석하여 모든 순간을 관찰하지 못할 경우의 결정 경계(decision boundary) 차이에 대해 논의합니다. 본 논문은 예측기의 병합 과정에서 인과적 가정이 미치는 영향을 규명하여, 머신러닝 및 통계 분야에서 중요한 기여를 하고자 합니다.



### Is Stochastic Gradient Descent Effective? A PDE Perspective on Machine Learning processes (https://arxiv.org/abs/2501.08425)
- **What's New**: 이 논문에서는 비볼록 손실 함수(non-convex loss functions)를 최소화하기 위해 신경망 가중치를 최적화하는 데 널리 사용되는 방법인 확률적 경사 하강법(Stochastic Gradient Descent, SGD)의 동작을 분석합니다. 특히, 기존의 방법으로는 대응할 수 없는 비볼록 잠재력(non-convex potential)과 퇴화 확산 행렬(degenerate diffusion matrix) 아래에서의 동작을 다루고 있습니다. 이 연구는 SGD에 대한 새로운 방법론적 통찰력을 제공합니다.

- **Technical Details**: SGD의 초기 단계에서 손실 함수(loss function)는 가중치를 가장 가까운 지역 최소값(local minimum) 주변으로 집중시키며, 이 단계를 "드리프트(Dift) 영역"이라고 명명합니다. 이후에는 확률적(fluctuations) 변화가 서브 최적 지역 최소값(suboptimal local minima)에서 벗어나는 데 도움을 주는 "확산(Diffusion) 영역"을 소개합니다. 이를 위해 평균 탈출 시간(Mean Exit Time, MET) 분석을 수행하고 MET의 상한 및 하한을 증명합니다.

- **Performance Highlights**: 이 연구는 비볼록 비용 함수(non-convex cost function)와 퇴화 확산 행렬 하에서 SGD의 점근적 수렴(asymptotic convergence)을 다룹니다. 우리는 이론적 효율성과 동적 특성을 설명하기 위해 이중성(duality) 및 엔트로피(entropy) 방법을 활용합니다. 결과적으로, SGD의 성능과 동작에 대한 새로운 관점을 제시하며, 머신 러닝 과정에서 "SGD가 나쁜 최소값에서 벗어나는 데 얼마나 시간이 걸리는가?"와 같은 기본 질문에 대한 통찰을 제공합니다.



### CVaR-Based Variational Quantum Optimization for User Association in Handoff-Aware Vehicular Networks (https://arxiv.org/abs/2501.08418)
Comments:
          Accepted in IEEE International Conference on Communications (ICC 2025)

- **What's New**: 이번 연구에서는 차량 네트워크(Vehicular Networks, VNet)의 일반화된 할당 문제(Generalized Assignment Problem, GAP)를 해결하기 위해 새로운 Conditional Value at Risk (CVaR) 기반의 변분 양자 고유해법(Variational Quantum Eigensolver, VQE) 프레임워크를 제안합니다. 기존의 선형 합 산술 문제에서 비롯된 GAP는 여러 제약 조건들로 인해 발생하는 계산적 도전 과제가 있습니다.

- **Technical Details**: 제안된 방법은 하이브리드 양자-고전적 구조(hybrid quantum-classical structure)를 채택하여, 목적 함수(objective function)와 제약 조건-specific penalties를 균형 있게 통합한 맞춤형 비용 함수(tailored cost function)를 활용합니다. CVaR-VQE 모델을 사용하여 솔루션 공간의 하위 구간(lower tail)에서 최적화를 집중함으로써, 노이즈가 있는 중간 규모 양자(NISQ) 장치에서의 수렴(convergence)과 회복력(resilience)을 향상시킵니다.

- **Performance Highlights**: 우리의 접근법은 차량 네트워크에서의 사용자 연관 문제(user-association problem)에 적용되었으며, 딥 뉴럴 네트워크(Deep Neural Network, DNN) 접근법에 비해 23.5%의 성능 향상을 달성하였습니다. 이 연구는 차량 네트워크에서 효율적인 자원 할당을 위한 새로운 범위의 가능성을 보여줍니다.



### BiDepth Multimodal Neural Network: Bidirectional Depth Deep Learning Arcitecture for Spatial-Temporal Prediction (https://arxiv.org/abs/2501.08411)
Comments:
          This paper has been submitted to Applied Intelligence for review

- **What's New**: 이 논문은 동적 시스템에서 공간-시간(Spatial-Temporal, ST) 정보를 정확하게 예측하는 데 중점을 둡니다. 제안된 BiDepth Multimodal Neural Network (BDMNN)은 쌍방향 깊이 조정을 사용하여 장기적 계절성(long-term seasonality)과 단기적인 변동(short-term fluctuations)을 모두 이해할 수 있게 합니다. 이는 기존의 통계적 접근 방식 및 전통적인 신경망(neural networks)의 한계를 극복하는 데 도움을 줍니다.

- **Technical Details**: BDMNN은 다양한 시간적 깊이(variable temporal depths)에서의 정보를 효과적으로 통합하며 공간적 맥락(spatial context)을 유지합니다. 이 모델은 장기적인 역사적 분석(comprehensive long-term historical analysis)과 단기적인 새로운 정보에 대한 빠른 반응(responsiveness) 간의 균형을 이루도록 설계되었습니다. 이러한 복잡한 ST 맥락에 적응하여 예측 정확성을 높입니다.

- **Performance Highlights**: 실제 공공 데이터를 사용한 사례 연구에서는 도시 교통 예측(urban traffic prediction)의 평균 제곱 오차(Mean Squared Error)를 12% 줄이는 등 예측 정확성이 크게 향상되었습니다. 또한, 강수 예측(rain precipitation forecasting)의 경우, 최신 기준(line benchmarks)에 비해 15%의 개선을 이루어냈습니다. 이러한 성과는 추가적인 계산 자원(computational resources)을 요구하지 않습니다.



### Predict Confidently, Predict Right: Abstention in Dynamic Graph Learning (https://arxiv.org/abs/2501.08397)
- **What's New**: 이 논문은 연속 시간 동적 그래프(CTDGs)에서 그래프 신경망(GNNs)의 프레임워크 내에 거부 옵션(reject option) 전략을 처음으로 통합하여 불확실성이 높고 자신감이 낮을 때 예측을 전략적으로 중단할 수 있도록 합니다. 이를 통해 중요한 잘못 분류의 위험을 최소화하고 결과와 신뢰성을 향상시킵니다. 또한, 이 과정에서 특정 범위(coverage) 내에서 예측을 최적화하는 커버리지 기반의 중단 예측 모델을 제안합니다.

- **Technical Details**: 기존의 GNNs는 동적 그래프 내의 복잡한 상호작용과 진화 구조를 포착하는 데 유망했지만, 잘못 분류의 비용이 높은 경우에서는 한계를 보였습니다. 본 논문에서는 거부 옵션을 통한 불확실성 관리와 극단적인 클래스 불균형 문제를 해결하기 위한 새로운 프레임워크를 제안하며, 이를 통해 라이징 태스크에 대한 예측 성능을 크게 향상시킵니다. 또한, 우리의 접근 방식은 엔드 투 엔드의 신경망 아키텍처를 설계하여 동적 그래프의 구조와 노드 상호작용에 따라 신뢰 수준을 동적으로 조정합니다.

- **Performance Highlights**: 다양한 동적 그래프 데이터세트를 대상으로 한 실험 결과, 우리 모델은 예측의 신뢰도와 커버리지 간의 트레이드오프를 효과적으로 관리하면서 성능 지표를 크게 개선했습니다. 특히 링크 예측 및 노드 분류 작업에서 AUC(곡선 아래 면적)와 AP(평균 정밀도) 점수를 향상시켜 위험 민감 애플리케이션에서의 신뢰성과 효용성을 극대화했습니다. 이러한 성과는 동적 및 불확실한 환경에서의 고정밀 적용을 위한 믿을 수 있는 솔루션으로서의 가능성을 입증합니다.



### Towards Fast, Specialized Machine Learning Force Fields: Distilling Foundation Models via Energy Hessians (https://arxiv.org/abs/2501.09009)
Comments:
          Under Review at ICLR 2025

- **What's New**: 이 논문은 머신 러닝 기반 포스 필드(MLFF)를 더욱 전문화된 MLFF로 전환하는 새로운 방법론을 제시합니다. 특히, 기존의 일반-purpose 모델에서 도출된 지식을 소형, 고속 모델에 전이하여 특정 화학 공간의 특성에 맞춘 모델을 생성합니다. 이 방식은 교육(teacher) 모델의 에너지 예측 헤시안(Hessian)을 작동하여 학생(student) 모델의 훈련 기초를 제공합니다.

- **Technical Details**: MLFF는 분자 구성에 기반하여 잠재적 에너지와 원자당 힘을 매핑하는 학습 가능한 함수 근사기입니다. 이 연구에서는 Knowledge Distillation (KD)을 기반으로 하는 접근 방식을 사용하여 원본 모델에서 작은 특화 모델로 지식을 효율적으로 전이합니다. 이를 통해 모델 아키텍처와 유도 편향에 구애받지 않고 학생-교사 모델 쌍에서 사용할 수 있는 간단하고 효율적인 방법을 제시합니다.

- **Performance Highlights**: 제안된 방법은 MACE-OFF, MACE-MP-0, JMP의 세 가지 MLFF FMs를 대상으로 시험하여, 특정 화학 분포에 특화된 학생 MLFF 모델이 최대 20배 빠른 추론 속도를 달성하는 것을 보여주었습니다. 또한, 에너지 및 힘 오차, MD 시뮬레이션의 안정성 및 에너지 보존에서의 상당한 향상을 달성하며, 대부분의 경우 학생 모델이 원본 모델보다 더 뛰어난 성능을 보였습니다.



### CrystalGRW: Generative Modeling of Crystal Structures with Targeted Properties via Geodesic Random Walks (https://arxiv.org/abs/2501.08998)
Comments:
          10+12 pages, 10 figures

- **What's New**: CrystalGRW라는 새로운 확산 기반 생성 모델이 도입되어 기존 결정 구조 예측 방법보다 효율적이고 정확한 결정 구조 생성을 가능하게 합니다. 이 모델은 Riemannian manifold 상에서 작동하며, 주기적인 결정 구조의 특성을 유지하는 새로운 결정 구성을 예측할 수 있습니다. 또한 CrystalGRW는 조건부 제어 기능을 통해 특정 결정학적 특성을 지정함으로써, 목표에 맞춘 물질 발견을 도와줍니다.

- **Technical Details**: CrystalGRW는 Riemannian score-based generative model(RSGM)과 기존에 개발된 공간 균형 잡힌 그래프 신경망 EquiformerV2를 결합하여 결정 구조 생성을 수행합니다. 이 모델은 원자의 좌표, 원자 유형 및 격자 매트릭스와 같은 세 가지 결정 특성을 생성하며, 학습된 데이터 분포를 바탕으로 새로운 구조 데이터를 효율적으로 샘플링합니다. 새로운 결정은 geodesic path를 따라 이동하는 랜덤 워크를 통해 생성됩니다.

- **Performance Highlights**: CrystalGRW는 기존 모델과 비교했을 때 그라운드 스테이트에 가까운 현실적인 결정 구조를 생성하는 능력을 보여줍니다. 실험적으로 검증된 구조에 대한 정확도를 제공하며, DFT(밀도 범함수 이론) 최적화된 구조와 유사한 속성을 가진 구조를 생성하여 후속 DFT 이완 시간을 줄여 줍니다. 이러한 특성 덕분에 CrystalGRW는 물질 발견과 역설계를 가속화하는 데 기여할 수 있습니다.



### Trusted Machine Learning Models Unlock Private Inference for Problems Currently Infeasible with Cryptography (https://arxiv.org/abs/2501.08970)
- **What's New**: 최근 기계 학습의 발전이 개인 정보를 보호하는 새로운 패러다임을 가능하게 만들었습니다. 이 연구에서는 Trusted Capable Model Environments (TCMEs)가 신뢰할 수 있는 제3자의 역할을 할 수 있는 기계 학습 모델을 제안합니다. TCMEs는 입력 및 출력 제약 조건 하에서 작동하는 모델과의 상호작용을 통해, 민감한 데이터를 보호하면서도 안전한 계산을 수행할 수 있도록 합니다.

- **Technical Details**: TCMEs의 주요 속성은 상태가 없음(statelessness), 정보 흐름 제어(explicit information flow control), 및 검증 가능성(verifiability)입니다. 이들 속성은 모델이 이전 상호작용에 대한 상태를 유지하지 않도록 하고, 신뢰성을 보장할 수 있도록 하는 데 필수적입니다. 특히, TCME 환경에서는 기계 학습 모델이 주어진 입력을 기반으로 올바른 출력을 생성하고, 모든 참여자가 합의한 정보 흐름 정책을 준수해야 합니다.

- **Performance Highlights**: TCME는 기존의 암호화 솔루션이 비현실적인 문제를 해결할 수 있는 대안을 제공합니다. 연구에서는 TCME가 단순한 암호 학습 문제를 포함한 여러 사용 사례를 소개하며, 민감한 데이터가 모델에 제공되더라도 이를 기록하거나 노출하지 않도록 설계되었습니다. TCMEs는 기계 학습 모델의 능력을 활용하여 안전한 계산을 수행하게 하며, 결국 개인 정보 보호와 계산 효율성을 동시에 달성하는 것을 목표로 합니다.



### Computing Approximated Fixpoints via Dampened Mann Iteration (https://arxiv.org/abs/2501.08950)
- **What's New**: 이번 연구에서는 비정확하게 알려진 함수의 최소 고정점(least fixpoint)을 근사하는 방법을 제시합니다. 여기서 우리는 함수가 근사 함수의 시퀀스로 표현될 때, 이들이 그 함수의 최소 고정점으로 수렴하도록 하는 새로운 접근 방식을 사용합니다. 특히, 단조(monotone) 및 비확장(non-expansive) 함수에 대해 논의하며, 이러한 함수에서의 고정점의 유일성이 보장되지 않는 점을 강조합니다.

- **Technical Details**: 우리의 주요 기여는 댐핑 계수(dampening factor)를 가진 Mann 반복(iteration) 방식의 변형을 제안하는 것입니다. 이 반복 방식은 특정한 조건하에 함수의 최소 고정점으로 수렴하는 것을 보장합니다. 이러한 수렴성은 Markov 의사결정 과정(Markov decision processes, MDPs)에서의 모델 기반 강화학습(context of model-based reinforcement learning)에서도 활용되므로, 이 연구의 결과는 상당한 응용 가능성을 지니고 있습니다.

- **Performance Highlights**: 제안된 반복 방식은 MDPs에 적용하여 최적의 예상 수익(optimal expected return)으로의 수렴을 이끌어 낼 수 있음을 보여줍니다. 또한, 확률적 오류 경계(probabilistic error bounds)를 통해 주어진 함수가 근사 가능한 경우에도 최소 고정점으로 거의 확실하게(iterate to the least fixpoint almost surely) 수렴할 수 있는 기회를 제공합니다. 이는 단순 확률 게임(simple stochastic games)과 같은 샘플링을 통해 탐색할 수 있는 확률적 시스템에서도 유용하게 적용될 수 있습니다.



### A Reinforcement Learning Approach to Quiet and Safe UAM Traffic Managemen (https://arxiv.org/abs/2501.08941)
Comments:
          Paper presented at SciTech 2025

- **What's New**: 이 논문은 도시 항공 이동성(UAM) 시스템의 통합에 따른 문제를 다룹니다. 특히, 소음과 안전을 함께 고려하는 다중 에이전트 강화 학습(multi-agent reinforcement learning) 접근 방식을 제안하며, 고도 조정을 통해 이 두 가지 목표를 균형 있게 달성하고자 합니다. 이는 기존 항공 시스템에서 발생하는 소음을 줄이고, 안전한 분리를 유지하는 방법에 대한 새로운 시각을 제공하고 있습니다.

- **Technical Details**: 연구는 UAM 환경을 마르코프 결정 과정(MDP)으로 모델링하여 소음 완화와 수직 분리를 보장하는 목표를 보상 함수에 정의합니다. 강화 학습 모델은 여러 층의 UAM 네트워크에서 고도를 조정함으로써 소음과 혼잡도 간의 트레이드오프를 학습합니다. 이 접근 방식은 고도가 낮을수록 소음은 증가하지만 사고의 위험이 줄어드는 상관관계를 설명합니다.

- **Performance Highlights**: 결과는 UAM 작전에서 소음 감소와 수직 분리 간의 상충 관계를 보여줍니다. 고도를 높이고자 할 때는 공중 교통 혼잡도가 증가하고 충돌 분리 사건의 발생 빈도가 높아지는 반면, 다양한 고도를 분산시키면 소음을 증가시키는 현상을 관찰했습니다. 이러한 발견은 환경적 지속 가능성과 운영 안전성을 동시에 달성하기 위한 균형 잡힌 정책의 필요성을 강조합니다.



### GenAI Content Detection Task 3: Cross-Domain Machine-Generated Text Detection Challeng (https://arxiv.org/abs/2501.08913)
Comments:
          COLING 2025

- **What's New**: 이 연구는 새로운 RAID 벤치마크를 사용하여 대규모 언어 모델(LLM)로 생성된 텍스트를 감지할 수 있는 단일 모델의 가능성을 탐구합니다. 이전의 공유 과제가 특정 도메인에 한정된 반면, 본 연구에서는 여러 도메인 및 모델을 사용하여 훈련 중에 모든 정보를 제공하고 탐지기의 경계를 이해하려고 했습니다. 9개의 팀이 참여하여 23개의 탐지기 제출물을 통해 99% 이상의 높은 정확도를 기록하여 여러 모델과 도메인의 텍스트를 효과적으로 탐지할 수 있음을 보여주었습니다.

- **Technical Details**: RAID는 11개의 생성 모델, 8개의 텍스트 도메인 및 11개의 적대적 공격이 포함된 1천만 개 이상의 문서로 구성된 데이터셋입니다. 연구 질문(RQ1 및 RQ2)을 통해 대규모로 텍스트를 생성하는 모델과 도메인이 존재하는 상황에서 탐지력이 어떻게 발휘되는지를 살펴보았습니다. 각 참가 팀은 주어진 도메인에 대한 탐지기를 제출하고, 적대적 공격이 포함된 서브태스크에서 성능을 평가했습니다.

- **Performance Highlights**: 두 팀(Pangram과 Leidos)은 적대적 공격이 없는 상황에서 99.3%의 정확도를 기록하고, 적대적 공격이 있는 경우에도 97.7%의 높은 성능을 보였습니다. 불균형 데이터셋을 고려할 때, 이러한 결과는 탐지기가 여러 도메인과 모델에서 동시에 텍스트를 강력하게 탐지할 수 있는 능력을 입증합니다. 평가 결과는 공개적으로 이용 가능하며, 향후 계획된 벤치마킹 방향이 함께 논의될 것입니다.



### Multi-View Transformers for Airway-To-Lung Ratio Inference on Cardiac CT Scans: The C4R Study (https://arxiv.org/abs/2501.08902)
Comments:
          Accepted to appear in Proceedings of International Symposium on Biomedical Imaging (ISBI), 2025

- **What's New**: 이 연구에서는 관상 CT(CT) 이미지를 사용하여 폐 크기 대비 기도 나무의 내경 비율(airway tree lumen to lung size ratio, ALR)을 추정하는 새로운 방법을 발표합니다. 이러한 ALR 값은 만성 폐쇄성 폐질환(Chronic Obstructive Pulmonary Disease, COPD)과 COVID-19의 중증도 및 후속 증상(Post-acute Sequelae of SARS-CoV-2 Infection, PASC)과의 관계를 연구하는 데 중요한 역할을 합니다. 연구 팀은 다각적(Multi-view) 접근 방식을 통해 ALR 값을 추론할 수 있는 Multi-view Swin Transformer라는 혁신적인 모델을 도입했습니다.

- **Technical Details**: 이 모델은 다중 뷰의 관상 CT 이미지를 이용하여 폐의 ALR 값을 구하고, Multi-Ethnic Study of Atherosclerosis (MESA)에서 수집된 쌍(pair) 형태의 전체 폐(full-lung) 및 관상 CT 데이터셋을 사용하여 지도 학습(supervised training)을 진행하였습니다. 연구 결과, 제안된 네트워크는 세분화된 관상 CT 이미지를 통한 직접적인 ALR 추정 방법보다 훨씬 높은 성능을 보여주었고, 전체 폐 ALR의 진실값(ground truth)과 유사한 정확도와 재현성을 달성했습니다.

- **Performance Highlights**: 이 연구에서 제안된 모델은 ALR 값을 추론할 때 높은 정확도와 재현성을 보여주었습니다. 특정 알림 기준을 이용한 반복 측정(re-scan)의 결과와 비교했을 때, 본 모델의 성능은 기존 방법에 비해 현저히 우수한 것으로 나타났습니다. 또한, 이를 통해 광범위한 역학 연구에서 사용 가능한 관상 CT 이미지를 활용하여 COPD와 COVID-19의 잠재적 관계를 더 깊이 분석할 수 있는 가능성을 제시합니다.



### Improved Compression Bounds for Scenario Decision Making (https://arxiv.org/abs/2501.08884)
- **What's New**: 이번 논문에서는 불확실한 환경에서 결정 내리는 시나리오 결정 방식의 새로운 경계를 제안합니다. 기존의 방식들이 제시한 경계는 다양한 가정 하에 만들어졌다면, 본 논문은 이러한 가정보다 약한 조건에서 더 나은 경계를 제시합니다. 이를 통해 의사 결정 과정에서 신뢰성을 더욱 높일 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 이 연구는 여러 샘플을 통해 불확실성을 평가하고, 이 샘플들을 기반으로 한 결정이 특정 위험 수준을 초과할 확률에 대한 경계를 제시합니다. 각 샘플 시나리오의 수, 최대 허용 위험, 그리고 문제의 고유한 속성인 'compression size'와 같은 요소들이 이러한 경계에 영향을 미칩니다. 저자들은 더욱 강력한 가정 없이도 기존의 경계를 개선할 수 있는 새로운 수식들을 개발했습니다.

- **Performance Highlights**: 제안된 경계는 실질적인 상황에서의 의사 결정 효율성을 높이는 데 기여할 것으로 보입니다. 여러 기존 방법들과의 비교를 통해, 새로운 경계가 실제로 어떻게 개선되는지를 보여줄 것입니다. 이 연구는 의사 결정 이론 분야에 중요한 기여를 할 것으로 예상됩니다.



### RouteNet-Gauss: Hardware-Enhanced Network Modeling with Machine Learning (https://arxiv.org/abs/2501.08848)
Comments:
          13 pages, 11 figures

- **What's New**: 본 논문은 RouteNet-Gauss라는 새로운 접근 방식을 소개합니다. RouteNet-Gauss는 네트워크 테스트베드와 머신러닝(ML) 모델을 통합하여 기존의 Discrete Event Simulation(DES)의 한계를 극복하려고 합니다. 이 모델은 테스트베드를 하드웨어 가속기로 활용하여 훈련 데이터셋을 빠르게 생성하고 실제 환경에 대한 높은 충실도로 네트워크 시나리오를 시뮬레이션합니다. RouteNet-Gauss는 예측 오류를 최대 95%까지 줄이고, 추론 시간에서 488배의 속도 향상을 보여줍니다.

- **Technical Details**: RouteNet-Gauss는 모듈화된 아키텍처로, 네트워크 시나리오의 특징에 따라 동적으로 구성됩니다. 이 시스템은 토폴로지나 라우팅과 같은 다양한 네트워크 구성을 이해하고 일반화할 수 있으며, 훈련 중 보지 못한 네트워크에도 적응할 수 있습니다. 또한, Temporal Aggregated Performance Estimation(TAPE)을 지원하여 시간적 세분성을 조정하고 흐름 성능 지표에서 높은 정확도를 유지합니다. 이 접근 방식은 시뮬레이션의 효율성과 정확성을 개선하는 데 유망한 도구를 제공합니다.

- **Performance Highlights**: 실험 결과, RouteNet-Gauss는 학습 중에 보지 못한 시나리오의 모델링에서 평균 절대 백분율 오차가 2.289%에 불과할 정도로 뛰어난 정확도를 보여줍니다. 이 모델은 특정 최첨단 솔루션에 비해 최대 488배 빠르게 추론할 수 있는 성능을 발휘하고 있습니다. RouteNet-Gauss는 하드웨어 테스트베드를 활용하여 다양한 네트워크 구성을 동적으로 생성하고, 흐름 수준 메트릭스를 제공하여 실용적인 응용에 적합합니다.



### Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation (https://arxiv.org/abs/2501.08822)
- **What's New**: 이 논문에서는 Huang 외(2015)가 제안한 Queue-Reactive 모델을 기반으로 한 Multidimensional Deep Queue-Reactive (MDQR) 모델을 소개합니다. MDQR 모델은 대기열 독립성 가정을 완화하고, 시장 특징을 통해 상태 공간을 확장하며, 주문 크기의 분포를 모델링합니다. 이러한 확장은 복잡한 가격 수준 간의 의존성을 학습하고 다양한 시장 상황에 적응할 수 있는 신경망(Neural Network) 아키텍처를 활용합니다.

- **Technical Details**: MDQR 모델은 주문 장부(order book)의 동역학을 모델링하기 위해 Queue-Reactive 모델의 신경망 확장을 적용합니다. 이 모델은 가격 수준 간의 상관관계를 반영하고, 실제 주문 사이즈를 잘 모델링하는 데 중점을 두고 있습니다. 또한, 모델은 'market impact' 패턴과 주요 스타일화 사실을 재현할 수 있는 능력을 검증하는 경험적 데이터를 사용합니다.

- **Performance Highlights**: Bund 선물 시장의 데이터를 활용하여, MDQR 모델이 시장 영향의 제곱근 법칙(square-root law), 대기열 간의 상관관계(cross-queue correlations), 그리고 현실적인 주문 크기 패턴을 잘 포착하는 것을 보여줍니다. 특히 조건부 및 정상 분포를 재현하는 데 강점을 보이며, 효율적인 계산 성능을 유지하여 강화학습(reinforcement learning) 또는 현실적인 백테스트(backtesting)와 같은 실용적 응용에서도 적합합니다.



### IDEA: Image Description Enhanced CLIP-Adapter (https://arxiv.org/abs/2501.08816)
- **What's New**: 이번 논문은 CLIP (Contrastive Language-Image Pre-training)을 기반으로 한 Image Description Enhanced CLIP-Adapter (IDEA) 방법을 제안합니다. 이 방법은 이미지와 텍스트의 상호작용을 활용하여 정밀한 특성을 포착함으로써, 적은 샘플로 이미지 분류 작업을 수행할 수 있도록 돕습니다. 또한, Trainable-IDEA (T-IDEA)를 도입하여 경량의 학습 가능한 컴포넌트를 추가하고, 11개의 데이터셋에서 SOTA (State-Of-The-Art) 성능을 달성했습니다.

- **Technical Details**: IDEA는 훈련이 필요 없는 방법으로, 이미지-텍스트 쌍의 보완적인 관계를 활용하여 다중 모달리티(multi-modality) 간의 의미적 연관성을 탐구합니다. T-IDEA는 경량 프로젝터와 학습 가능한 잠재 공간(learnable latent space)을 통합하여 IDEA의 성능을 더욱 향상시킵니다. 이러한 방식은 기존의 방법과는 다르게 강력한 성능을 발휘하며, 11개의 공개 이미지 데이터셋에서 실험적으로 검증되었습니다.

- **Performance Highlights**: IDEA와 T-IDEA는 훈련이 필요 없는 설정과 훈련이 요구되는 설정 모두에서 기존의 SOTA 방법들을 초월하는 성능을 보여주었습니다. 새로운 데이터셋인 'IMD-11'을 생성하여 총 1,637,795개의 이미지-텍스트 쌍을 제공하며, 이는 연구에 중요한 기여를 하게 됩니다. 이러한 성과는 제한된 학습 데이터로도 뛰어난 성능을 달성할 수 있음을 시사합니다.



### Nesterov Acceleration for Ensemble Kalman Inversion and Variants (https://arxiv.org/abs/2501.08779)
- **What's New**: 이번 연구는 Ensemble Kalman Inversion (EKI)에서 Nesterov acceleration 기법을 도입하여 다양한 역문제에 대한 EKI 비용 함수의 감소 속도를 높이는 효과를 보였습니다. Nesterov acceleration은 EKI의 몇 가지 변형에도 적용되어 효율성을 극대화할 수 있는 방법론으로 자리 잡혔습니다. 연구 결과는 기계 학습 및 최적화 알고리즘의 발전이 Kalman 최적화 분야에 어떻게 전이될 수 있는지를 보여줍니다.

- **Technical Details**: 연구에서는 EKI의 연속 시간 버전에서 시작하여, 모델 파라미터와 관측치 사이의 관계를 표현하기 위해 Gaussian 노이즈를 가진 비용 함수를 정의합니다. EKI는 확률 분포를 바탕으로 파라미터의 점 추정치를 계산하는 데 집중하며, Nesterov acceleration을 적용하여 이 계산 과정을 가속화합니다. 이를 위해, 연구는 Monte Carlo 근사를 통해 파라미터의 타임 스텝을 명시적으로 이산화하고, EKI의 일반적인 형태로 변환합니다.

- **Performance Highlights**: Nesterov acceleration을 통합한 EKI는 기존 EKI 변형들과 비교했을 때 더 빠른 수렴 속도를 보여 주목받고 있습니다. 특히, unscented Kalman inversion과 ensemble transform Kalman inversion에 대한 적용을 통해 계산 복잡성을 줄이면서도 성능을 강화하였습니다. 이 접근법은 또한 추가적인 계산 비용 없이 기존 알고리즘과 쉽게 결합할 수 있는 장점이 있습니다.



### Leveraging LLM Agents for Translating Network Configurations (https://arxiv.org/abs/2501.08760)
- **What's New**: 이번 논문에서는 네트워크 설정 번역(configuration translation)의 필요성을 강조하며, 대규모 언어 모델(LLM) 에이전트를 활용한 새로운 프레임워크를 제안합니다. 이 프레임워크는 네트워크 장비의 이질성을 극복하기 위해 의도 기반의 접근 방법을 사용하여, 보다 정교하고 자동화된 설정 번역을 가능하게 합니다. 연구 결과에 따르면, 제안한 방법은 97.74%의 구문(correctness) 정확성을 달성하며 기존의 방법들보다 뛰어난 번역 성능을 보였습니다.

- **Technical Details**: 제안된 의도 기반 검색 보강 생성(IGAG) 모듈은 설정 파일을 조각으로 나누고, 의도를 추출한 후 정확한 번역을 생성하는 시스템입니다. IRAG 모듈은 셋의 주요 구성 요소로 이루어져 있습니다: (a) 설정 분할 및 의도 추출을 위한 설계된 프롬프트 시스템, (b) 필터링과 투표 전략을 결합한 수동 검색 메커니즘, (c) 맥락 종속성을 유지하는 점진적 번역 과정입니다. 또한, 번역의 정확성을 높이기 위해 이중 검증 모듈을 설계했습니다.

- **Performance Highlights**: 실험을 통해 제안한 방법은 97.74%의 구문 정확성을 달성하였으며, 기존의 최신 방법들과 비교해도 번역의 정확성에서 뛰어난 성과를 보이고 있습니다. 이 논문은 다양한 벤더 간의 네트워크 설정 번역의 어려움을 분석하고, 이를 해결하기 위한 LLM 에이전트를 활용한 프레임워크를 구현하여 실제 데이터셋에서 성능을 평가했습니다. 향후 코드 또한 리뷰 프로세스 이후 오픈 소스로 제공될 예정입니다.



### $\texttt{InfoHier}$: Hierarchical Information Extraction via Encoding and Embedding (https://arxiv.org/abs/2501.08717)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문은 Self-Supervised Learning(SSL)과 Hierarchical Clustering(HC)을 통합한 InfoHier라는 새로운 프레임워크를 제안합니다. 기존 SSL 방법은 비층적 구조에 집중하며, 다차원 데이터의 복잡한 관계를 반영하지 못했습니다. HC는 계층적 데이터를 이해하는 데 유리하지만, 엄격한 유사도 기반 메트릭에 의존해 한계가 있었습니다.

- **Technical Details**: InfoHier는 SSL을 통해 적응형 표현을 제공하고, HC가 복잡한 패턴을 캐치할 수 있도록 도와줍니다. 이를 위해 각 데이터 포인트에 대해 루트 이진 트리 형태의 계층적 구조를 추출하며, 이는 잠재 표현(latent representation)을 통해 이루어집니다. 이 과정에서 Dasgupta 손실을 사용하여 클러스터 성능을 향상시키고, 대조 손실(contrastive loss)을 결합하여 Encoder 네트워크를 더욱 견고하게 조정합니다.

- **Performance Highlights**: InfoHier는 클러스터링과 표현 학습 모두에서 성능을 향상시킬 수 있는 가능성을 지닙니다. 이번 연구에서 제안하는 방법은 복잡한 데이터셋에서 정보 계층 구조를 반영한 보다 효율적인 분석을 가능하게 하여 정보 검색 및 데이터 관리를 효과적으로 수행할 수 있습니다. 이 연구는 대규모 무라벨 데이터에 대한 정보 추출의 현재 한계를 극복하는 데 기여할 것입니다.



### Self-supervised Transformation Learning for Equivariant Representations (https://arxiv.org/abs/2501.08712)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 자기지도 변환 학습(Self-supervised Transformation Learning, STL) 방법을 제안합니다. 기존의 변환 레이블을 이미지 쌍에서 유도된 변환 표현으로 대체함으로써, 변환의 내성을 높이고 성능을 개선하는 것을 목표로 합니다. STL은 이전 방법들과 같은 배치 복잡성을 유지하면서도 변환 표현을 학습할 수 있는 방식으로, 다양한 분류와 탐지 작업에서 효과를 입증했습니다.

- **Technical Details**: STL에서 사용하는 변환 표현은 이미지에 적용된 변환에 불변성을 가지고 있습니다. 이 표현을 학습함으로써, STL은 정확하고 복잡한 변환 정보를 캡처할 수 있습니다. 추가로 AugMix와 같은 복잡한 변환을 통합하여 과거의 동등한 방법에서 불가능했던 성능 향상을 달성했습니다. 이러한 방식은 다양한 기초 모델과도 호환 가능하여 폭넓은 적용성을 지니고 있습니다.

- **Performance Highlights**: STL은 여러 데이터셋과 작업에서 매우 경쟁력 있는 성능을 발휘했습니다. 11개의 벤치마크 중 7개에서 기존 방법들을 초과하는 성과를 보여주었으며, 탐지 작업에서 특히 두드러진 성과를 올렸습니다. 변환 표현의 상호 의존성을 효과적으로 캡처함으로써, STL은 기존의 변환 학습 방법들보다 우수한 변환 예측 성능을 자랑합니다.



### Product of Gaussian Mixture Diffusion Model for non-linear MRI Inversion (https://arxiv.org/abs/2501.08662)
- **What's New**: 본 논문에서는 최근의 확산 모델(Diffusion models)이 자기공명영상(MRI) 재구성에서 뛰어난 성능을 보이는 점에 주목하였습니다. 다수의 파라미터로 구성된 기존의 블랙박스(blakc-box) 네트워크가 해석 가능성 및 재구성 속도를 저해하는 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 경량(lean)하고 파라미터 효율적(parameter-efficient)이며, 해석 가능한 가우시안 혼합 모델을 이미지 프라이어(image prior)로 사용하여 이미지와 코일 민감도(coil sensitivities)를 공동으로 재구성합니다. 이 과정에서 고전적인 매끄러움 프라이어(smoothness priors)를 코일 민감도에 적용했습니다.

- **Performance Highlights**: 제안된 방법은 기존의 고전적인 변이학적 패널티(total variation)와 유사한 수준의 성능을 보여주며, 빠른 추론(fast inference)과 샘플링 경로(sampling trajectories)에 대한 강인성(robustness)을 입증하였습니다. 또한, 확률적(formulation) 구성을 통해 후방 기대값(posterior expectation)과 화소별 분산(pixel-wise variance) 계산이 가능해졌습니다.



### Joint Learning of Depth and Appearance for Portrait Image Animation (https://arxiv.org/abs/2501.08649)
- **What's New**: 본 연구는 2D 초상화 애니메이션의 새로운 접근 방식을 소개합니다. 기존의 이미지 생성 방법들이 RGB 이미지를 생성하는 데 집중했던 반면, 우리의 방법은 깊이(depth) 정보도 함께 학습하여 더욱 풍부한 시각적 출력을 가능하게 합니다. 우리는 시각적 외관과 깊이를 동시에 학습할 수 있는 새로운 확산 기반 초상화 이미지 생성기를 개발하였습니다. 이 프레임워크는 얼굴 깊이 추정, 이미지 리라이트 및 오디오 기반의 애니메이션 생성과 같은 다양한 다운스트림 애플리케이션에 효율적으로 적응할 수 있습니다.

- **Technical Details**: 우리는 Stable Diffusion 아키텍처를 기반으로 한 새로운 초상화 생성기를 소개합니다. 이 모델은 RGB 및 깊이(latent images)를 분리하여 추가 잡음을 제거하는 6채널 입력 이미지를 처리하며, 함께 사용되는 참조 네트워크(reference network)는 RGB 참조 이미지를 통해 이미지 확산 과정을 안내합니다. 이를 통해 생성된 깊이 맵은 생성된 얼굴 이미지와 잘 일치하며, 두 개의 채널 간의 강한 상관관계를 보장합니다. 학습 데이터는 스튜디오에서 캡처한 얼굴 이미지와 3D 기하학적 정보를 포함하여, 야외 환경에서도 잘 작동하는 모델로 일반화될 수 있도록 합니다.

- **Performance Highlights**: 훈련이 끝난 후, 본 모델은 RGB 및 깊이 이미지를 커플링한 상태로 샘플링할 수 있으며, 주어진 이미지를 기반으로 깊이 채널을 인페인팅(inpainting)하거나 반대로 깊이 이미지를 사용하여 RGB 채널을 인페인팅하는 등의 작업을 수행할 수 있습니다. 또한, 깊이 정보로부터 얼굴 이미지의 조명을 변경하는 후처리도 가능하며, 마지막으로 오디오 입력에 기반한 talking head 애니메이션 생성이 가능합니다. 이러한 다양한 기능들은 초상화 조작 분야에서 새로운 가능성을 열어줍니다.



### A Learning Algorithm That Attains the Human Optimum in a Repeated Human-Machine Interaction Gam (https://arxiv.org/abs/2501.08626)
- **What's New**: 이번 연구에서는 인간의 행동을 통해 학습 기반 제어 시스템의 비용 함수를 최소화하는 게임 이론 기반 학습 알고리즘을 제안합니다. 기존의 역문제를 해결해야 했던 접근 방식과는 달리, 새로운 알고리즘은 인간의 행동 관찰만으로 최적의 비용을 찾을 수 있습니다. 이러한 기법은 인간-로봇 상호작용에서 특히 유용할 것으로 보입니다.

- **Technical Details**: 제시된 알고리즘은 반복 게임(Repeated Game) 구조에서 인간-기계 상호작용을 모델링합니다. 기존의 방법들과 달리 알고리즘은 인간의 비용 함수에 대한 사전 지식이 없으며, 단순히 인간의 반응을 관찰하면서 최적의 비용을 찾아냅니다. 본 알고리즘은 비대칭 정보 상황에서 두 에이전트가 공유 비용을 최소화하려고 할 때 유용하게 작동합니다.

- **Performance Highlights**: 광범위한 인간 실험에서 이 알고리즘의 성능을 평가한 결과, 규정된 인간 비용 함수의 최소값으로 일관되게 수렴함을 확인했습니다. 이 연구는 의수 및 외골격 장치의 최적 보조 제공에 중요한 기여를 할 것으로 기대되며, 미래의 이론적 및 경험적 확장 방향에 대해 논의합니다.



### OpenMLDB: A Real-Time Relational Data Feature Computation System for Online ML (https://arxiv.org/abs/2501.08591)
- **What's New**: OpenMLDB는 오프라인과 온라인에서 일관된 특성 계산을 보장하는 통합 쿼리 계획 생성기를 제공하여 특성 배포 시간을 크게 단축합니다. 이 시스템은 이전에 각기 다른 팀과 시스템이 처리하던 오프라인 및 온라인 프로세스를 통합하여 데이터 일관성을 높입니다. 또한, OpenMLDB는 클라우드 환경에서 높은 동시성을 유지하며 실시간으로 안정적인 특성 업데이트를 지원합니다.

- **Technical Details**: OpenMLDB는 오프라인 및 온라인 계산을 위한 최적화 기술을 통해 초고속 특성 계산 성능을 실현합니다. 온라인 계산의 경우, 긴 윈도우 집계를 개선하기 위해 사전 집계 결과를 재사용하며, 다중 테이블 윈도우 유니온의 성능 병목을 해결하는 자가 조정 전략을 사용합니다. 오프라인 계산의 경우, 다중 윈도우 병렬 최적화를 지원하며 데이터 스큐를 줄이기 위해 튜플을 재분배합니다.

- **Performance Highlights**: OpenMLDB는 Flink 및 DuckDB보다 10배~20배 높은 온라인 성능을 제공하며, Spark 및 GreenPlum과 같은 MPP 데이터베이스에 비해 6배 더 빠른 오프라인 성능을 자랑합니다. 또한, Redis와 같은 인메모리 데이터베이스에 비해 40% 낮은 메모리 사용량을 기록하여 리소스 절약에 기여합니다. 현재 OpenMLDB는 150명 이상의 기여자가 참여하고 있으며, GitHub에서 1.6k 스타를 얻었습니다.



### A Systematic Review of Machine Learning Methods for Multimodal EEG Data in Clinical Application (https://arxiv.org/abs/2501.08585)
Comments:
          This paper includes 4 figures, 6 tables, and totals 18 pages

- **What's New**: 이번 연구에서는 다중 모달 데이터(multimodal data)를 EEG 자료에 통합하여 기계 학습(machine learning)과 딥 러닝(deep learning) 모델의 정확성을 높이는 방법을 탐구했습니다. 여러 임상 응용 분야에서의 EEG 데이터의 새롭고 효과적인 활용을 제시하며, 신경정신적 장애와 같은 복잡한 임상 문제를 해결하는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: 문헌 검색은 PubMed, Web of Science, Google Scholar를 통해 수행되었으며, 총 16개의 연구가 최종적으로 선정되었습니다. 데이터 융합(data fusion)은 신호(signal), 특징(feature), 결정(decision) 수준에서 이루어졌고, 가장 많이 사용된 기계 학습 모델은 서포트 벡터 머신(support vector machines, SVM)과 결정 트리(decision trees)였습니다.

- **Performance Highlights**: 16개의 연구 중 11개에서 다중 모달 EEG 데이터를 사용했을 때 모델의 정확도가 향상되었다고 보고되었습니다. 이 리뷰는 다중 모달 EEG 기반 기계 학습 모델이 임상 진단 및 문제 해결에 중요한 가능성을 갖고 있다는 점을 강조합니다.



### MIAFEx: An Attention-based Feature Extraction Method for Medical Image Classification (https://arxiv.org/abs/2501.08562)
Comments:
          In preparation for Journal Submission

- **What's New**: 이 논문은 기존의 특징 추출 기법과 머신 러닝 분류기의 한계를 극복하기 위해 Medical Image Attention-based Feature Extractor (MIAFEx)를 제안합니다. MIAFEx는 Transformer 인코더 아키텍처 내에서 분류 토큰을 향상시키기 위해 학습 가능한 정제 메커니즘을 활용하여 눈에 띄는 특징을 효과적으로 추출합니다. 이는 의료 이미징 데이터의 특성에 적응하고 기능의 품질을 개선하며, 특히 제한된 훈련 데이터 상황에서 높은 정확성을 보여줍니다.

- **Technical Details**: MIAFEx는 Transformer 모델의 [CLS] 토큰을 정제하여 준비됩니다. 이 정제 과정은 학습 가능한 가중치를 통해 동적으로 조절되어, 모델이 분류에 가장 중요한 특징에 집중할 수 있도록 돕습니다. 이러한 접근 방식은 전통적인 특징 추출 기법과 혼합 분류기와 비교하여 뛰어난 성능을 발휘하며, 다양한 의료 이미징 데이터셋에서 효과성을 확인했습니다.

- **Performance Highlights**: MIAFEx는 CNN 및 ViT와 같은 현대적인 딥 러닝 모델과 비교하여 특히 작은 데이터셋에서 성능이 우수하다는 것을 나타냅니다. 실험 결과, MIAFEx는 다양한 복잡한 의료 이미징 데이터셋에서 높은 정확성과 강인성을 기록하였으며, 이는 전통적인 모델들이 일반화하는 데 어려움을 겪는 상황에서도 효과적으로 기능하는 것으로 확인되었습니다.



### ANSR-DT: An Adaptive Neuro-Symbolic Learning and Reasoning Framework for Digital Twins (https://arxiv.org/abs/2501.08561)
- **What's New**: 이 논문에서는 디지털 트윈 기술을 위한 적응형 신경-상징적 학습 프레임워크인 ANSR-DT를 제안합니다. ANSR-DT는 패턴 인식 알고리즘과 강화 학습, 상징적 추론을 결합하여 실시간 학습과 적응형 지능을 가능하게 합니다. 이 프레임워크는 인간-기계 협업을 필요로 하는 응용 프로그램에서 보다 나은 의사 결정을 지원하며, 기존의 최첨단 방법들과 비교해 의사 결정의 정확성, 신뢰성 및 해석 가능성에서 유의미한 개선을 경험했습니다.

- **Technical Details**: ANSR-DT 프레임워크는 Proximal Policy Optimization(PPO) 알고리즘을 활용하여 CNN-LSTM 및 주의(attention) 기술과 연계하여 상징적 추론의 논리적 명확성을 확보합니다. 이 프레임워크의 구조는 물리적 레이어, 처리 레이어, 적응 레이어로 구성되어 있습니다. 센서 데이터가 시스템을 통해 흐르고 의사 결정을 내리는 과정에서 확보된 해석 가능한 결과들은 CNN-LSTM 알고리즘을 통해 생성됩니다.

- **Performance Highlights**: ANSR-DT는 경쟁 프레임워크와 비교하여 실질적인 성능 향상을 보여줍니다. 특히, 다양한 산업 환경에서 실시간 적응 및 지속적인 학습을 보장하여 사용자 선호와 환경 변화에 능동적으로 대응할 수 있도록 돕습니다. 또한, 기호적 추론을 통한 알고리즘 개발이 가능하여 더 나은 운영과 성과를 위한 논리적인 결과를 제공합니다.



### LAMS: LLM-Driven Automatic Mode Switching for Assistive Teleoperation (https://arxiv.org/abs/2501.08558)
- **What's New**: 본 논문은 LLM-Driven Automatic Mode Switching (LAMS)라는 새로운 접근 방식을 소개합니다. LAMS는 사용자가 이전에 수행한 작업에 대한 예시 없이도 작업 맥락에 따라 자동으로 제어 모드를 전환할 수 있도록 Large Language Models (LLMs)를 활용합니다. 기존의 자동 모드 스위칭 방법들이 특정 작업에 의존하는 것과는 달리, LAMS는 일반화 가능성을 강조합니다.

- **Technical Details**: LAMS는 사용자가 조작하는 작업의 맥락을 자연어 지시로 변환하고, 이를 LLM에 입력하여 조이스틱의 방향 이동을 특정 로봇 행동으로 맵핑합니다. 이 과정에서 사용자 상호작용을 통해 생성된 모드 전환 예시를 통합하여 성능을 점진적으로 개선할 수 있습니다. 또한, LAMS는 사용자 연구에서 10명의 참가자를 대상으로 복잡한 작업을 수행함으로써 그 효과를 입증했습니다.

- **Performance Highlights**: LAMS는 사용자가 복잡한 여러 단계를 포함하는 작업을 수행할 때 필요한 수동 모드 전환의 수를 줄이는 데 효과적임을 보여주었습니다. 연구 결과, 사용자는 LAMS를 기존의 대안 방법보다 선호하며, LAMS는 반복적인 작업 수행을 통해 자동 모드 전환 능력을 개선하는 것으로 나타났습니다. 이로 인해 사용자의 인지 부하가 줄어들고 작업 효율성이 향상되었습니다.



### Reinforcement Learning-Enhanced Procedural Generation for Dynamic Narrative-Driven AR Experiences (https://arxiv.org/abs/2501.08552)
Comments:
          Number of pages: 13, Number of figures: 4. Accepted for presentation at GRAPP 2025 - 20th International Conference on Computer Graphics Theory and Applications (for additional details on the conference visit this https URL). Disclaimer: This preprint may differ from the final version published in the conference proceedings

- **What's New**: 이 연구는 아르 AR 환경을 위한 강화 학습 기반의 Wave Function Collapse (WFC) 프레임워크를 제안합니다. 이 방식은 특정 환경 규칙과 동적 타일 가중치 조정을 통해 맵을 생성하여, 게임 플레이 요구에 맞춰 맵의 응답성을 향상시킵니다. 이를 통해 만들어진 맵은 내러티브 기반 AR 게임에 적합할 뿐만 아니라, 교육 및 시뮬레이션 훈련 등 다양한 분야에도 적용 가능성이 있습니다.

- **Technical Details**: 제안된 WFC 알고리즘은 도시, 사막, 숲의 세 가지 독특한 생태계를 포함하여 맵을 생성합니다. 각 생태계는 고유한 레이아웃과 예술적 스타일을 가지며, RL을 통해 최적화된 생태계 일관성과 경로 레이아웃을 보장합니다. 이 알고리즘은 셀과 타일의 개념을 사용하여 그리드 기반의 3D 맵을 동적으로 생성하며, 실시간으로 던전 마스터가 이 맵을 조정할 수 있도록 지원합니다.

- **Performance Highlights**: 비교 평가 및 사용자 연구를 통해 제안된 프레임워크는 맵 품질에서 우수한 성과를 보였으며, 몰입감 있는 경험을 제공합니다. 이러한 특성 덕분에 내러티브 중심의 AR 게임에 적합하게 설계되어 있으며, 기존 방법에 비해 더 높은 사용자 경험을 제공합니다. 또한, 보고된 성능은 교육 및 XR 경험과 같은 더 넓은 응용 분야에서도 큰 가능성을 보여줍니다.



### A Theory of Optimistically Universal Online Learnability for General Concept Classes (https://arxiv.org/abs/2501.08551)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 \\{0, 1\\} 라벨을 가진 개념 클래스의 낙관적으로 보편적인 온라인 학습 가능성에 대해 완전한 특성을 제공합니다. 저자들은 온라인 학습을 최소한의 가정 하에서 배우는 방법을 이해하기 위해 낙관적으로 보편적인 온라인 학습을 정의하고, 이를 통해 각 개념 클래스에 대한 학습 가능성을 탐구합니다. 이는 기계 학습 및 통계적 학습에서 중요한 발전으로, 알고리즘의 일반성을 강조합니다.

- **Technical Details**: 본 연구에서 저자들은 데이터 프로세스의 특성과 가정에 대해 두 가지 질문을 제기합니다. 첫째는 데이터 프로세스에 대한 최소한의 가정이 무엇인지, 둘째는 이러한 최소한의 가정을 만족하는 모든 데이터 프로세스에서 성공할 수 있는 학습 알고리즘이 존재하는지입니다. 이러한 알고리즘은 주어진 개념 클래스에 대해 낙관적으로 보편적이라고 불립니다.

- **Performance Highlights**: 저자들은 모든 개념 클래스에 대해 두 가지 질문을 해결하였고, 각 경우에 대한 일반 학습 알고리즘을 설계했습니다. 또한 이러한 알고리즘과 결과를 불확실한 경우로 확장하여, 불확실한 경우와 실현 가능한 경우의 학습 가능성에 대한 데이터 프로세스의 최소 가정 간의 동등성을 보여주었습니다.



### OMEGA: A Low-Latency GNN Serving System for Large Graphs (https://arxiv.org/abs/2501.08547)
- **What's New**: 이번 논문에서는 OMEGA라는 시스템을 제안하여 대규모 그래프에서 GNN(그래프 신경망)의 낮은 대기 시간과 최소한의 정확도 손실을 달성하는 방법을 소개합니다. 기존의 근사 기술이 훈련 단계에서 오버헤드를 줄일 수 있지만 서비스 단계에서는 여전히 높은 지연 시간과 정확도 손실을 초래할 수 있다는 문제를 지적하고 있습니다. OMEGA는 두 가지 핵심 아이디어인 선택적 재계산(selective recomputation)과 컴퓨테이션 그래프 병렬화(computation graph parallelism)를 활용하여 이러한 문제를 해결합니다.

- **Technical Details**: OMEGA 시스템은 재계산을 최소화하면서 기존의 임베딩(embeddings)을 재사용하여 정확도를 유지하는 방법을 채택합니다. 이 과정에서 Selective Recomputation of Precomputed Embeddings(SRPE)를 통해 데이터 의존성을 고려하여 재계산이 필요한 소수의 이웃 노드를 식별하고, Computation Graph Parallelism(CGP)을 통해 통신 오버헤드를 줄입니다. 각 머신에서 로컬 파티션의 데이터를 사용하여 파티셔닝된 계산 그래프를 구축하고, 이를 효율적으로 집계하여 최종 출력을 계산하는 방식입니다.

- **Performance Highlights**: 시험 결과는 SRPE와 CGP의 조합이 OMEGA가 DGL 기반의 전체 그래프 및 근사 기반 서비스 시스템에 비해 최대 159배 및 10.8배 더 낮은 대기 시간을 기록하면서도, 최소한의 정확도 손실을 유지함을 보여줍니다. OMEGA는 대규모 데이터셋과 GNN 모델에서 기존의 최첨단 기술을 능가하는 성능을 나타냅니다. 이러한 결과는 실제 GNN 사용 사례에서의 적용 가능성을 더욱 높여 줄 것입니다.



### Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers (https://arxiv.org/abs/2501.08537)
Comments:
          Mistakenly submitted as a replacement to 2405.05409v4

- **What's New**: 이 연구는 transformers의 구성적(reasoning-oriented) 문제 해결 메커니즘에 대한 심층 분석을 제공합니다. 특히, 복잡성 제어 전략이 모델이 규칙을 학습하고 일반화하는 방식에 미치는 영향을 조사하였습니다. 기존의 제약을 넘어서, 이 연구에서는 보다 구체적이고 통제된 환경에서 모델의 사고 과정을 분석했습니다.

- **Technical Details**: 연구는 synthetic experimental approach를 사용하여 transformers의 내부 메커니즘을 조사합니다. 데이터 생성 프레임워크를 통해 훈련 데이터, in-distribution (ID) 테스트 데이터, 그리고 out-of-distribution (OOD) 테스트 데이터 사이의 명확한 구분을 도모했습니다. 복잡성 제어는 파라미터 초기화와 가중치 감소 계수를 통해 모델의 복잡성과 추론 능력에 영향을 미치는 전략을 의미합니다.

- **Performance Highlights**: 복잡성 제어 프레임워크를 적용함으로써 OOD 일반화 성능이 눈에 띄게 개선되는 것을 관찰했습니다. 이 연구 결과는 transformers가 다양한 상황에서 고차원 개념을 조합해 새로운 객체를 생성할 수 있는 능력을 향상시키는 방법에 대한 통찰을 제공합니다. 모델의 행동 예측 또한 가능해져, 훈련 과정에서의 모델 동작을 예측할 수 있는 기반을 마련했습니다.



### SuperSAM: Crafting a SAM Supernetwork via Structured Pruning and Unstructured Parameter Prioritization (https://arxiv.org/abs/2501.08504)
- **What's New**: 이 논문에서는 Vision Transformer (ViT) 기반의 Neural Architecture Search (NAS)를 위한 새로운 검색 공간 설계 전략을 제안합니다. 특히, Segment Anything Model (SAM)을 가중치 공유(supernetwork)인 SuperSAM으로 변환하여 다양한 하위 네트워크를 효율적으로 탐색합니다. 이 접근 방식은 계층별로 구조적 가지치기와 매개변수 우선 순위를 자동화하여 최적의 하위 네트워크를 제안합니다.

- **Technical Details**: SuperSAM은 SAM 모델을 변형하여 다양한 자원 제약에 맞는 하위 네트워크를 생성할 수 있는 "탄력적인" 구조로 설계되었습니다. 논문에서 제시된 방법은 프루닝(pruning) 기법을 사용하여 각 transformer 계층의 중요도를 평가하고 중요하지 않은 계층에 대해 확률적 가지치기를 실시합니다. 또한, MLP 블록의 매개변수 우선 순위 지정과 함께 각 계층에서 중요한 매개변수를 유지하여 결과적으로 연결된 하위 네트워크를 생성합니다.

- **Performance Highlights**: SuperSAM에서 파생된 하위 네트워크는 기존의 사전 훈련된 SAM ViT-B 모델에 비해 30-70% 더 작은 크기를 가지고 있지만, 성능에서 실제로 우수한 성과를 보입니다. 이를 통해 기존 모델보다 적은 자원이 소요되면서도 동일한 수준의 성능을 유지하는 효율적인 구조를 달성했습니다. 이 연구는 ViT NAS의 검색 공간 설계에 대한 새로운 접근 방식을 제시합니다.



### Scalable Bayesian Physics-Informed Kolmogorov-Arnold Networks (https://arxiv.org/abs/2501.08501)
- **What's New**: 본 논문에서는 불확실성 정량화(Uncertainty Quantification, UQ)의 새로운 접근 방식을 제안합니다. 전통적인 다층 퍼셉트론(Multilayer Perceptions, MLPs) 대신 파라미터가 적은 Kolmogorov-Arnold Networks (KANs)를 사용합니다. 특히, 이 논문은 dropout Tikhonov ensemble Kalman inversion (DTEKI) 방법을 Chebyshev KANs와 결합하여 효율적인 불확실성 정량화를 달성하고자 합니다. 이를 통해 기존의 Hamiltonian Monte Carlo (HMC) 방법보다 높은 효율성과 안정성을 보입니다.

- **Technical Details**: 본 연구는 Bayesian Physics-Informed Neural Networks (B-PINNs)와 Chebyshev KANs를 활용하여 고차원 및 대규모 데이터 집합에서 발생하는 계산의 비효율성을 해결하고자 합니다. 새로운 DTEKI 방법은 그라디언트 계산이 필요 없는 방식으로, 모델의 파라미터 공간 차원을 줄이는 active subspace 방법을 적용하여 더 신뢰할 수 있는 UQ 결과를 제공합니다. 특히, Tikhonov 정규화와 dropout 제약을 통해 오버피팅(Overfitting) 문제를 완화합니다.

- **Performance Highlights**: 여러 실험 결과에서 제안된 방법은 작은 노이즈를 가진 문제에서 HMC와 비교해 동등한 성능을 보였으며, 또한 고차원 문제에 대해서도 뛰어난 효율성을 나타냈습니다. 새로운 접근 방법은 예측 정확성을 유지하면서 계산 비용을 상당히 낮추어 주며, 다양한 환경에서의 테스트에서도 긍정적인 결과를 보여줍니다. 최종적으로, 이 방법은 기존 모델보다 확장성 있는 솔루션을 제공하며, 여러 테스트 케이스에서 그 효과를 입증하였습니다.



### Quantifying the Importance of Data Alignment in Downstream Model Performanc (https://arxiv.org/abs/2501.08496)
- **What's New**: 이 논문은 전통적으로 강조되어온 데이터셋 크기 대신, 데이터 정렬(data alignment)의 역할에 주목합니다. 이 정렬은 데이터 품질(data quality)에서 간과되기 쉬운 측면으로, 대규모 언어 모델(Large Language Models, LLM)의 성능 개선에 미치는 영향을 정량화하기 위해 Task2Vec 기반의 정렬 계수(alignment coefficient)를 사용했습니다.

- **Technical Details**: 연구는 두 가지 설정에서 제어된 개입 실험(interventional experiments)을 실시했습니다. 첫 번째는 다양한 프리 트레인(pre-training) 데이터셋과 평가 데이터셋 간의 정렬 계수 증가가 미치는 영향을, 두 번째는 도메인 특화 파인 튜닝(fine-tuning) 데이터셋과 도메인 특화 평가 간의 정렬 계수 증가가 미치는 영향을 분석했습니다. 이러한 설정 중 Autoformalization이라는 특정 도메인 과제를 통해 데이터를 평가하였습니다.

- **Performance Highlights**: 연구 결과 모델의 학습 데이터와 평가 데이터 간의 정렬 계수는 모델의 손실(Loss) 및 혼란도(Perplexity)와 강하게 부정적인 상관관계를 가지며, 이는 각 다운스트림 작업에 대한 모델의 성능에 직접적인 영향을 미친다는 것을 보여줍니다. 특히 데이터가 평가 작업과 잘 정렬된 경우 낮은 혼란도 점수가 나타나며, 이는 LLM 훈련 접근 방식을 재평가할 필요성을 시사합니다.



### FLAVARS: A Multimodal Foundational Language and Vision Alignment Model for Remote Sensing (https://arxiv.org/abs/2501.08490)
- **What's New**: 이번 연구에서는 FLAVARS라는 새로운 사전 훈련(pretraining) 방법을 제안합니다. 이는 대비 학습(contrastive learning)과 마스크 모델링(masked modeling)의 장점을 결합하고, 지리적 정렬(geospatial alignment)을 위한 대비 위치 인코딩(contrastive location encoding)을 추가합니다. FLAVARS 모델은 SkyCLIP을 기준으로 KNN 분류 및 의미 분할(semantic segmentation) 같은 비전 전용(task-specific) 작업에서 현저한 성능 향상을 보여주었습니다.

- **Technical Details**: FLAVARS는 대규모 원거리 센싱 이미지 및 텍스트 설명 데이터셋인 SkyScript에서 훈련되었습니다. 기존의 다중 모달 사전 훈련 방식이 성능 저하를 초래한 반면, FLAVARS는 비전 전용 작업에서 성능을 크게 향상시키면서 제로 샷 분류(zero-shot classification) 능력을 유지합니다. 특히, 지리적 이미지-위치 정렬을 통해 사전 훈련 성능이 개선되었음을 발견했습니다.

- **Performance Highlights**: FLAVARS 사전 훈련 방식은 SpaceNet1 데이터셋에서 +6% mIOU 향상을 기록하며, KNN 분류 및 의미 분할 작업에서 SkyCLIP보다 월등한 성능을 보였습니다. 결과적으로, FLAVARS는 기존 CLIP 기반 사전 훈련에 비해 비전-언어 정렬(vision-language alignment) 능력을 자랑하지만, 데이터 세트 간 데이터 분할 기준이 다르기 때문에 보다 일반화된 성능 향상은 추가적인 연구가 필요합니다.



### Head Motion Degrades Machine Learning Classification of Alzheimer's Disease from Positron Emission Tomography (https://arxiv.org/abs/2501.08459)
Comments:
          5 pages

- **What's New**: 본 연구는 알츠하이머 질병(Alzheimer’s Disease, AD) 진단을 위해 PET 이미지를 활용한 머신 러닝(ML) 기반 분류 알고리즘의 성능이, 환자의 머리 움직임 수정 여부에 따라 어떻게 달라지는지를 보여줍니다. 특히 움직임 보정(motion correction) 없이 이미지를 분류할 경우 정확도가 크게 저하된다는 중요한 점을 발견했습니다. 이는 AD 진단에서 PET 기반 머신 러닝 알고리즘의 유효성을 제한할 수 있습니다.

- **Technical Details**: 연구에서는 128개의 $^{11}$C-UCB-J과 173개의 $^{18}$F-FDG 스캔 데이터를 사용했습니다. 각 스캔은 움직임 수정이 적용된 PET 이미지와 적용되지 않은 이미지를 바탕으로 두 가지 분류 방법으로 실험이 진행되었습니다. 또한, ResNet10 아키텍처를 이용하여 PET 이미지 특징을 추출하고 이 특징을 기반으로 Support Vector Machine (SVM) 분류기를 이용해 이진 분류를 수행했습니다.

- **Performance Highlights**: 테스트 결과, 움직임 보정이 적용되지 않은 이미지에서의 분류 정확도가 $^{18}$F-FDG 스캔에서 10% 감소, $^{11}$C-UCB-J 스캔에서 5% 감소함을 확인했습니다. 이 연구는 PET 이미지를 최적 활용하기 위해 효율적인 움직임 보정 방법이 필요하다는 것을 강조하고 있으며, 머신 러닝 분석의 유효성을 향상시키기 위한 기초 자료를 제공합니다.



### Large Language Models For Text Classification: Case Study And Comprehensive Review (https://arxiv.org/abs/2501.08457)
- **What's New**: 이번 연구에서는 데이터 분류에서의 대규모 언어 모델(LLMs)의 가능성을 탐구하고, 다양한 LLM의 성능을 최신 딥러닝(deep learning) 및 머신러닝(machine learning) 모델과 비교합니다. 특히, 직원의 근무지 분류와 가짜 뉴스 탐지의 두 가지 분류 시나리오에서 LLM의 성과를 평가하며, 각 모델의 성능과 시간 효율성을 종합적으로 분석합니다. 연구 결과, LLM이 복잡한 분류 작업에서 전통적인 방법보다 우수하지만, 추론 시간의 비용이 발생함을 보여줍니다.

- **Technical Details**: 이 연구에서는 여러 LLM과 머신러닝 알고리즘을 비교하여 다중 클래스 분류 문제와 이진 분류 문제에서의 성능을 분석합니다. 모델 크기와 양자화(quantization), 프롬프트 기술이 성능에 미치는 영향을 탐구합니다. 특히, 다양한 프롬프트 기법들이 LLM 성능에 미치는 영향을 분석하고, 적절한 최적화가 결과를 개선할 수 있음을 보여줍니다.

- **Performance Highlights**: 결과적으로, Llama3와 GPT-4 모델은 다중 클래스 분류와 같은 복잡한 작업에서 전통적인 방법보다 뛰어난 성능을 발휘하는 것으로 나타났습니다. 반면, 간단한 이진 분류 작업에서는 보다 전통적인 머신러닝 모델이 성능 대비 시간 효율성에서 우위를 보였습니다. 이 연구는 LLM의 효과적인 사용 사례 및 프롬프트 기술의 중요성을 강조하며, 실제 적용 가능성에 대한 통찰을 제공합니다.



### Vchitect-2.0: Parallel Transformer for Scaling Up Video Diffusion Models (https://arxiv.org/abs/2501.08453)
- **What's New**: Vchitect-2.0는 대규모 텍스트-비디오 생성 을 위한 비디오 확산 모델을 확장하기 위해 설계된 병렬 트랜스포머 아키텍처입니다. 이 시스템은 텍스트 설명과 생성된 비디오 프레임 간의 일관된 정렬을 이루기 위해 새로운 Multimodal Diffusion Block을 도입했습니다. 또한, 메모리 효율적인 학습 프레임워크를 통해 장기간 비디오 시퀀스를 효율적으로 훈련할 수 있습니다.

- **Technical Details**: Vchitect-2.0은 텍스트 프롬프트와 프레임 간의 기능 정렬을 보장하는 멀티모달 확산 블록을 특징으로 합니다. 혼합 병렬성(hybrid parallelism) 프레임워크를 통해 메모리 최적화 기술과 결합하여 확장성과 효율성을 제공하며, 분산 시스템에서의 고해상도 비디오의 효율적인 생성을 가능하게 합니다. 이 시스템은 여러 가지 메모리 저감 기술을 포함하며, 장시간 비디오 시퀀스의 훈련을 지원합니다.

- **Performance Highlights**: 광범위한 벤치마크 평가에서 Vchitect-2.0은 주요 메트릭에서 기존 기술을 일관되게 능가하는 성능을 보였습니다. 기술적 분석을 통해 더욱 부드러운 프레임 전환을 이룩하며, 모션 아티팩트를 효과적으로 줄였습니다. 하이브리드 병렬성 프레임워크는 훈련 확장성을 증진시키고 메모리 소모를 줄이는 데 기여하였으며, ablation 연구에서도 멀티모달 확산 블록과 병렬 전략의 중요성이 부각되었습니다.



### Instruction-Guided Fusion of Multi-Layer Visual Features in Large Vision-Language Models (https://arxiv.org/abs/2501.08443)
- **What's New**: 이 논문에서는 다양한 인코더 레이어에서 추출한 시각적 특징이 LVLM(대규모 비전-언어 모델)의 성능에 미치는 영향을 체계적으로 조사하였습니다. 다양한 작업 카테고리에서 18개의 벤치마크를 대상으로 분석한 결과, 각기 다른 레이어의 시각적 특징들이 상호 보완적이며, 기존의 균일한 융합 방식이 최적의 성능을 발휘하지 못함을 발견하였습니다.

- **Technical Details**: 제안된 지침 기반 비전 집합기를 통해, LVLM이 입력된 텍스트 지침에 따라 동적으로 다중 레이어 특징을 통합할 수 있도록 하였습니다. 이는 시각적 토큰의 수를 증가시키지 않으면서도 작업 특화된 특징 통합을 가능하게 합니다. 제안된 모듈은 LLaVA-v1.5 프레임워크에 통합되어 우수한 성능 개선을 이끌어냈습니다.

- **Performance Highlights**: 실험 결과, 미드-투-하이 레벨의 특징이 의미적 작업에서 우위를 점하며, 저수준 특징이 세밀한 인식 작업에 필수적이라는 것을 확인하였습니다. 제안된 모듈은 기존의 태스크-비분리적인 융합 방법들보다도 더 나은 성능을 보여주며, 레이어별 특징의 중요성을 강조하는 귀중한 통찰력을 제공합니다.



### FARE: A Deep Learning-Based Framework for Radar-based Face Recognition and Out-of-distribution Detection (https://arxiv.org/abs/2501.08440)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 이 연구에서는 단거리 FMCW 레이더를 활용한 얼굴 인식 및 OOD (out-of-distribution) 탐지 시스템을 제안합니다. 이 시스템은 Range-Doppler 및 micro Range-Doppler 이미지를 사용하며, 인식 모델의 두 개의 경로를 통해 ID 얼굴 분류와 OOD 탐지를 동시에 수행합니다. 두 단계로 훈련이 진행되며, 첫 단계에서는 triplet loss를 사용하여 ID 얼굴의 분류 정확도를 최적화합니다.

- **Technical Details**: 제안하는 아키텍처는 두 가지 경로, 즉 기본 경로(PP)와 중간 경로(IP)를 통해 구성되어 있습니다. PP는 ID 얼굴의 정확한 분류를 담당하고, IP는 OOD 탐지를 위한 구조입니다. 첫 번째 단계에서는 PP를 훈련시키고, 두 번째 단계에서는 PP를 고정하여 IPs를 훈련하여 OOD 탐지를 수행합니다.

- **Performance Highlights**: 제안된 방식을 통해 60 GHz FMCW 레이더로 생성한 데이터셋에서 ID 얼굴 분류 정확도 99.30%와 OOD 탐지 AUROC 96.91%를 달성했습니다. 또한, FARE는 기존 OOD 탐지 방법들에 비해 뛰어난 성능을 보이며, 이는 보안과 신뢰성을 위한 중요한 발전을 의미합니다.



### A Constant Velocity Latent Dynamics Approach for Accelerating Simulation of Stiff Nonlinear Systems (https://arxiv.org/abs/2501.08423)
- **What's New**: 이번 연구에서는 StODEs(강건 상미분 방정식)을 해결하기 위해 새로운 접근 방식을 제안합니다. 기존의 수치적 통합 방법을 피하고, 상태 공간 내에서 잠재 동역학( latent dynamics)을 학습하는 방식으로, 시간 통합 없이도 솔루션을 구할 수 있습니다. 이는 StODEs의 특성을 효과적으로 반영하면서도 기존의 머신 러닝 방법에서는 발생할 수 있는 시간 스케일 문제를 해결합니다.

- **Technical Details**: 우리는 일정한 속도를 가진 잠재 동역학 시스템을 고려하여, 이 시스템의 솔루션이 직선의 연속형으로 표현됩니다. 인코더 네트워크가 경사(기울기)와 초기 조건을 학습하여 잠재 동역학을 형성합니다. 또한, 비선형 시간 변환(nonlinear transformation of time)을 사용하여 잠재 공간(latent space)에서 시간을 늘리거나 줄이는 역할을 하며, 이로 인해 솔루션의 서로 다른 시간 영역에 다양한 주의를 기울일 수 있습니다.

- **Performance Highlights**: 우리는 우리의 접근 방식이 StODEs의 강건 비선형 시스템의 솔루션을 임의의 정확도 {
\epsilon}에 맞춰 근사할 수 있다는 보편적 근사화 증명을 제공합니다. 실험 결과에 따르면, 제안한 방법이 StODEs를 다룰 때 기존의 최첨단 머신 러닝 방법을 능가하는 성능을 보여줍니다.



### SEAL: Speaker Error Correction using Acoustic-conditioned Large Language Models (https://arxiv.org/abs/2501.08421)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 이 논문은 Speaker Diarization(SD) 시스템의 성능 향상을 위해 Acoustic-conditioned Large Language Models(LLMs)를 활용하는 새로운 접근 방식을 제안합니다. 기존 SD 시스템에서 발생하는 화자 오류를 줄이기 위해 음향 정보를 LLM과 결합합니다. 또한, 복잡한 후처리 없이 LLM의 환각을 줄일 수 있는 단순한 제한 디코딩 전략을 도입하여 더 정확한 화자 레이블을 할당합니다.

- **Technical Details**: 메인 아이디어는 SEAL이라는 프레임워크를 사용하여 EEND(End-to-End Neural Diarization) 네트워크에서 제공하는 음향적 후행 확률을 활용하는 것입니다. 각 단어에 대한 화자 포스터리어 성능을 네트워크를 통해 포착하고, 이를 바탕으로 LLM이 보다 정확한 결과를 도출하도록 합니다. 또한, 화자 확률을 직관적으로 이해하기 쉬운 레이블로 변환하고, 이를 통해 LLM이 더 나은 성능을 발휘하도록 해줍니다.

- **Performance Highlights**: 제안된 접근 방식은 Fisher, Callhome, RT03-CTS 데이터셋을 통해 기존 Acoustic SD와 비교할 때 화자 오류율을 24-43%까지 감소시키는 효과를 보여주었습니다. 이러한 결과는 SEAL이 LLM의 내재된 어휘 지식을 음향 정보와 통합함으로써 발생하는 큰 개선을 반영합니다. 연구 결과는 다양한 다중 화자 전사 응용 프로그램에 이 혁신적인 방법이 실질적인 기여를 할 수 있음을 보여줍니다.



### Leveraging 2D Masked Reconstruction for Domain Adaptation of 3D Pose Estimation (https://arxiv.org/abs/2501.08408)
Comments:
          16 pages, 7 figures

- **What's New**: 이 연구는 RGB 기반 3D 포즈 추정 방법의 효율성을 높이기 위해 비지도 학습 도메인 적응 프레임워크를 소개합니다. 기존의 방법들은 훈련 데이터와 분포가 다른 테스트 이미지에는 잘 작동하지 않았으나, 이 연구에서는 마스킹된 이미지 모델링(Masked Image Modeling, MIM)을 활용하여 라벨이 없는 데이터를 활용하는 방안을 제안합니다. 이를 통해 훈련 과정에서 다양한 데이터를 효과적으로 사용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서 제안하는 기술은 전경 중심 재구성과 주의 정규화(attention regularization)입니다. 전경 중심 재구성은 주어진 데이터에서 독특한 정보나 특징을 부각시켜, 라벨이 없는 데이터의 활용 효율성을 높이는 데 도움을 줍니다. 주의 정규화는 모델이 중요한 요소에 더 집중할 수 있도록 학습을 유도합니다. 이러한 방법은 다양한 데이터셋을 통한 실험을 통해 검증됩니다.

- **Performance Highlights**: 본 연구는 인간 및 손 포즈 추정 작업에서 여러 데이터셋을 대상으로 성능 평가를 진행했습니다. 특히 교차 도메인(cross-domain) 시나리오를 통해 검증한 결과, 제안된 방법은 모든 데이터셋에서 최첨단 정확도를 달성했습니다. 이는 제안된 비지도 학습 접근 방식이 실제 데이터 환경에서 효과적임을 입증하는 사례로 평가됩니다.



### OptiChat: Bridging Optimization Models and Practitioners with Large Language Models (https://arxiv.org/abs/2501.08406)
- **What's New**: 본 논문은 최적화 모델을 해석하고 설명하기 위해 자연어 대화 시스템인 OptiChat을 소개합니다. OptiChat은 최적화 전문가의 도움 없이 실제 적용 분야에서 작업하는 사용자들이 모델을 이해하고 진단할 수 있도록 돕기 위해 설계되었습니다. 이를 통해 최적화 모델과 사용자 간의 상호작용을 원활하게 하고, 사용자가 직면한 문제를 보다 잘 해결할 수 있는 것을 목표로 합니다.

- **Technical Details**: OptiChat은 다양한 사전 정의된 함수와 코드 생성을 결합하여 최적화 모델의 해석을 지원합니다. 사용자가 입력한 최적화 코드에는 Pyomo/Python 언어가 사용되며, 쿼리를 효과적으로 처리하기 위해 각 사용자 쿼리 유형에 따라 질의 전략을 구현합니다. 사용자는 진단, 검색, 민감도, 가정 질문(what-if) 및 비가정 질문(why-not) 등 다양한 쿼리를 통해 모델에 대한 정보를 요청할 수 있습니다.

- **Performance Highlights**: OptiChat은 24개의 최적화 모델을 테스트한 결과, 전문가와 비교해 짧은 시간 내에 정확한 모델 설명과 사용자 쿼리에 대한 응답을 제공하는 것으로 나타났습니다. 사용자가 모델을 독립적으로 해석하지 못할 경우 OptiChat이 그 부담을 덜어주며, 신뢰할 수 있는 자동화된 모델 설명을 통해 최적화 전문가의 시간을 절약할 수 있습니다. 최종적으로 OptiChat의 설명 품질은 전문가가 제공하는 설명과 유사한 수준으로 평가되었습니다.



### Empathetic Conversational Agents: Utilizing Neural and Physiological Signals for Enhanced Empathetic Interactions (https://arxiv.org/abs/2501.08393)
- **What's New**: 본 논문은 대화형 에이전트(Conversational Agents, CAs)가 사용자와의 공감 있는 상호작용을 향상시키기 위해 신경적 및 생리적 신호를 통합하는 방법을 탐구합니다. 특히, 신경 신호는 소음이 많거나 어두운 환경에서도 효과적으로 감정을 인식할 수 있는 잠재력을 지니고 있습니다. 본 연구에서는 사용자 연구를 통해 이러한 통합이 실시간으로 감정을 이해하고 상호작용의 질을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 대화형 에이전트는 신경적 및 생리적 신호를 감지하고 이를 기반으로 적절한 감정 표현을 구현할 수 있도록 설계되었습니다. 이 신호들은 EEG(뇌파 검사), GSR(피부 전도도), HRV(심박 변이나), BVP(혈액량 압력)와 같은 다양한 생리적 입력을 포함하여 실시간 감정 인식을 지원합니다. 연구에서는 다중 모달(multiple modalities) 데이터를 활용하여 공감(digital humans, DHs)할 수 있는 에이전트를 개발하는 방법을 다룹니다.

- **Performance Highlights**: 사용자 연구 결과, 공감 대화형 에이전트와의 상호작용 동안 참가자들은 더 강한 감정을 느끼고 높은 참여도를 경험하였습니다. 이는 신경적 및 생리적 신호를 통한 감정 인식의 효과를 입증하며, 대화형 에이전트의 공감 능력이 실질적으로 강화될 수 있음을 보여줍니다. 그러나 인식 정확도, 감정 전이 속도, 개인적 성격의 영향, 그리고 음성 톤 조절의 한계와 같은 여러 도전 과제가 여전히 남아 있습니다.



### Towards Best Practices for Open Datasets for LLM Training (https://arxiv.org/abs/2501.08365)
- **What's New**: 이번 논문에서는 인공지능(AI) 기업들이 저작권 소유자의 동의 없이 대규모 언어 모델(LLM)을 훈련시키고 있는 현황을 다룹니다. 이러한 행위의 허용 여부는 지역에 따라 다르며, EU와 일본 같은 국가에서는 특정 제한 하에 허용되고 있지만, 미국에서는 법적 경관이 모호합니다. 이로 인해 창작자들이 제기하는 저작권 소송이 증가하고 있으며, 이는 기업 및 공공기관이 훈련 데이터셋에 대한 정보를 축소하는 최근의 경향에 영향을 미치고 있습니다.

- **Technical Details**: 저작권 논란에도 불구하고, AI 모델의 투명성과 책임성을 저해하는 정보 공유 제한은 연구자, 감사인 및 피해자들이 AI 모델을 이해하는 데 필요한 정보에 접근하는 데 문제를 일으킵니다. 이러한 문제는 공개 접근(open access) 및 공공 도메인(public domain) 데이터를 기반으로 하는 언어 모델의 훈련으로 완화될 수 있지만, 현재 이에 대한 기술적 및 사회적 도전으로 인해 의미 있는 규모로 훈련된 모델은 없습니다. 데이터 조합에 필요한 부정확하고 불완전한 메타데이터, 물리적 기록의 디지털화(digitization) 비용과 복잡성, 빠르게 변화하는 환경에서 관련성과 책임성을 보장하기 위한 다양한 법적 및 기술적 기술들이 그 장애 요인입니다.

- **Performance Highlights**: AI 시스템이 책임감 있게 관리되고 큐레이션된 공개 라이센스 데이터로 훈련될 수 있는 미래를 구축하기 위해서는 법적, 기술적, 정책적 분야 간의 협력이 필수적입니다. 메타데이터 표준, 디지털화 및 개방성 문화 촉진에 대한 투자도 중요합니다. 이러한 통합적인 접근이 이루어져야만 AI 모델의 사용과 관련된 데이터 안전성과 책임이 보장될 수 있습니다.



### Weight Averaging for Out-of-Distribution Generalization and Few-Shot Domain Adaptation (https://arxiv.org/abs/2501.08361)
Comments:
          Master Thesis

- **What's New**: 이번 논문에서는 out-of-distribution generalization (OOD 일반화) 문제를 해결하기 위해 weight averaging (WA)와 sharpness-aware minimization (SAM)이라는 두 가지 기법을 소개합니다. WA는 서로 다른 하이퍼파라미터로 훈련된 여러 모델의 가중치를 평균내어 OOD 일반화 성능을 향상시키며, SAM은 평평한 영역에서의 미니마를 찾아 네트워크를 최적화하여 분포 변화에 대한 성능을 개선합니다. 새로운 접근으로, WA에서 기울기 유사도를 손실 정규화 기법으로 도입하여 모델의 다양성을 높이고, WA와 SAM을 결합하여 few-shot domain adaptation (소수 샷 도메인 적응)의 문제를 해결하고자 합니다.

- **Technical Details**: 논문에서 제안하는 방법은 MNIST, SVHN, USPS, MNIST-M과 같은 숫자 데이터셋 및 VLCS, PACS 같은 도메인 적응 데이터셋에서 실험했습니다. 신경망 학습 과정에서 기울기 유사도를 정규화하여 모델 다양성을 명시적으로 증가시키는 새로운 손실 함수를 도입했습니다. WA와 SAM의 결합을 통해 소수의 샷에서도 더 나은 OOD 일반화 성능을 발휘하도록 하였습니다.

- **Performance Highlights**: 실험 결과 WA와 SAM을 결합함으로써 OOD 일반화 성능이 크게 향상되었고, 소수 샷 도메인 적응 정확성이 상당히 증가했습니다. 이 방법들은 기존의 기법보다 다양한 상황에서 더 나은 성능을 보였으며, 향후 연구 방향에 대한 기초를 마련하고 있습니다. 이처럼 성능 개선을 통해 OOD 문제와 소수 샷 적응에 대한 새로운 해결책을 제공하는 점이 주목받고 있습니다.



### Dissecting a Small Artificial Neural Network (https://arxiv.org/abs/2501.08341)
Comments:
          12 pages, 8 figures, and 2 tables

- **What's New**: 이번 연구는 XOR 게이트를 나타내는 가장 간단한 인공 신경망의 손실 경관(loss landscape)과 역전파(backpropagation) 수렴 동역학을 조사합니다. 손실 경관의 9차원 파라미터 공간에서의 단면(cross-sections)이 명확한 특징을 보이며, 이는 역전파가 높은 효율로 제로 손실에 수렴하는 이유를 이해하는 데 기여합니다. 연구는 또한 통계 물리학(statistical physics) 관점에서 마이크로캐노니컬 엔트로피(microcanonical entropy)를 도입하여 네트워크의 상태 행동을 특성화합니다.

- **Technical Details**: 연구에서는 XOR 문제를 해결하기 위해 간단한 인공 신경망을 사용하였습니다. 이 신경망의 구조는 입력층에 두 개의 뉴런과 단일 은닉층(hidden layer)에 두 개의 뉴런으로 구성되며, 총 6개의 가중치(weight) 링크가 연결되어 있습니다. 역전파를 통해 네트워크의 최적화를 분석하며, 학습 과정에서의 다양한 차원을 고려하여 경량화된 접근 방식을 사용합니다.

- **Performance Highlights**: XOR 문제를 해결하는 신경망의 학습 과정에서 손실 경관은 가중치와 바이어스가 계속 드리프트하면서도 효율적으로 수렴하는 특성을 보입니다. 은닉 뉴런이 추가됨에 따라 손실 경관이 단순화되고, 이는 엔트로피 장벽(entropic barriers)을 제거하는 결과를 가져옵니다. 본 연구는 신경망의 복잡한 학습 구조와 그 동역학의 관계를 이해하는 데 기여하고 있습니다.



### High-throughput digital twin framework for predicting neurite deterioration using MetaFormer attention (https://arxiv.org/abs/2501.08334)
Comments:
          17 pages, 8 figures

- **What's New**: 본 논문에서는 신경 발달 장애(NDD, Neurodevelopmental Disorders)와 관련된 신경가소성 저하를 모델링하기 위한 새로운 하이퍼루프 디지털 트윈 프레임워크를 소개합니다. 이 프레임워크는 합성 데이터 생성, 실험 이미지 및 머신러닝(ML) 모델을 통합하여 다양한 신경가소성 저하 패턴을 포착하는 데 유용합니다. 유전적 요인 및 신경생물학적 복잡성을 고려하여, 시뮬레이션과 실험 데이터를 통합하여 연구자들이 실험 결과를 예측할 수 있도록 돕습니다.

- **Technical Details**: 디지털 트윈 프레임워크는 세 가지 모듈로 구성됩니다: 합성 데이터 생성기, 실험 데이터 세트, 그리고 MetaFormer 기반 ML 모델입니다. 합성 데이터 생성기는 이소기하학적 분석(IGA) 기반의 단계 필드 모델을 사용하여 신경가소성 저하를 시뮬레이션합니다. MetaFormer 모델은 공간적 및 시간적 의존성을 효과적으로 캡처하고, 평균 오차를 각각 1.9641%와 6.0339%로 줄여주는 예측 능력을 가지고 있습니다.

- **Performance Highlights**: 본 프레임워크는 실험 및 합성 데이터의 조합을 통해 신경가소성의 동적 변화를 이해하고 예측하는 데 도움이 됩니다. 합성 데이터 세트는 다양한 신경가소성 저하 패턴을 시뮬레이션하여 연구자들에게 풍부한 학습 자료를 제공합니다. 이 연구는 치료 개발을 위한 통찰력을 제공하고, 비용을 절감하며, 실험 설계를 개선하여 NDDs 연구의 진행을 가속화합니다.



### Customizable LLM-Powered Chatbot for Behavioral Science Research (https://arxiv.org/abs/2501.05541)
- **What's New**: 본 연구에서는 행동 과학 연구를 보조하기 위해 설계된 웹 기반의 사용자 정의 LLM 기반 챗봇 시스템인 CLPC(Customizable LLM-Powered Chatbot)를 소개합니다. 이 시스템은 전통적인 챗봇의 개념을 넘어 실험 도구로서 기능하며, 연구에 필요한 데이터 교차 검증을 용이하게 합니다. 또한 CLPC는 연구자가 자신의 로깅 이벤트를 쉽게 통합할 수 있도록 설계되어 있습니다.

- **Technical Details**: CLPC 시스템은 React 프레임워크로 구축되어 웹과 모바일 플랫폼 모두에서 사용할 수 있으며, 다양한 화면 크기에 적응하는 반응형 사용자 인터페이스를 제공합니다. 이 시스템은 사용자가 사용자 이름과 실험 코드를 입력함으로써 데이터 관리가 용이하게 설계되었습니다. 사용자는 LLM(대규모 언어 모델)의 선택, 결과 표시 방식 및 단어 간 지연 시간 등을 조정할 수 있는 4개의 매개변수를 설정할 수 있습니다.

- **Performance Highlights**: CLPC는 연구자들이 코드를 변경하지 않고도 여러 매개변수를 조정할 수 있는 유연성을 제공합니다. 특히, 새로운 모델이나 사용자 정의 모델을 용이하게 통합할 수 있도록 설계된 백엔드는 다양한 연구 환경에서 CLPC의 적응성을 높여줍니다. 전체적인 데이터 로깅 메커니즘은 사용자 상호작용을 포괄적으로 기록할 수 있도록 하여 연구에서 필요한 데이터 수집을 간소화합니다.




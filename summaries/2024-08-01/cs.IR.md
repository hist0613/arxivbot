New uploads on arXiv(cs.CL)

### ShieldGemma: Generative AI Content Moderation Based on Gemma (https://arxiv.org/abs/2407.21772)
- **What's New**: 자동으로 MCQ(Multiple Choice Questions)을 생성하는 기존 메트릭들은 교육적 가치를 고려하지 않고 단순히 단어의 유사성만 평가해왔습니다. 이를 해결하기 위해 새로운 평가 메트릭, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)을 제안합니다. 이 메트릭은 MCQ가 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되고, 이를 자동 평가할 수 있는 두 가지 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 다양한 전문가 평가 MCQ 품질 지표에 대해 강력한 예측력을 보입니다.



### Adaptive Retrieval-Augmented Generation for Conversational Systems (https://arxiv.org/abs/2407.21712)
Comments:
          12 pages, under review

- **What's New**: 자동 다지선다형 질문(MCQ) 생성을 위한 새로운 자동 평가 메트릭 Knowledge Dependent Answerability(KDA)을 제안. KDA는 학생의 지식을 평가할 수 있도록 MCQ의 대답 가능성을 측정하는 새로운 접근법을 제시합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한 최근의 연구는 특정 순간에 외부 지식을 필요로 하는 시스템 응답을 예측하는 RAGate 모델을 제안하여, 불필요한 지식 증가를 방지하고 대화형 시스템의 반응성을 개선하는 방법을 소개하고 있습니다.

- **Technical Details**: KDA는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 교육적 가치를 반영하지 못한다는 한계를 극복하기 위해 개발되었습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 통해 학생의 문제 해결 행동을 모방하여 KDA를 자동으로 추정하는 메트릭입니다. 또한, 최근의 연구는 RAG는 무조건적으로 필요하지 않음을 지적하며, 각 회차에 어떤 외부 지식을 증가시킬 것인지 예측하는 RAGate라는 게이트 기능을 가진 모델을 제안합니다. 이 방법은 대화 시나리오에서 사용된 외부 지식의 관련성과 시스템 반응의 자신감 수준 사이의 상관 관계를 보고 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 상황에서의 사용성과 강한 상관관계를 나타냈으며, 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 예측력이 높았습니다. RAGate 모델은 대화형 시스템이 적절한 대화 회차에 외부 지식을 효율적으로 사용하도록 하여, 고품질의 시스템 반응을 생성할 수 있음을 입증했습니다.



### Synth-Empathy: Towards High-Quality Synthetic Empathy Data (https://arxiv.org/abs/2407.21669)
Comments:
          arXiv admin note: text overlap with arXiv:2407.01937

- **What's New**: 이 논문들에서는 여러 영역에서 새로운 자동 평가 메트릭과 데이터 생성 방법을 제안합니다. 첫 번째 논문은 자동으로 생성된 다지선다형 질문(MCQ)의 교육적 가치를 평가할 수 있는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 두 번째 논문은 contrastive learning과 counterfactual augmentation을 사용하여 NLP 모델의 robustness를 개선할 수 있는 방법을 탐구합니다. 세 번째 논문은 대규모 언어 모델(LLM)을 활용한 새로운 감정 데이터 생성 및 품질 관리 파이프라인, Synth-Empathy를 소개하며, 이는 인간 노동 없이 고품질 감정 데이터 생성을 목표로 합니다.

- **Technical Details**: 첫 번째 논문에서는 KDA라는 새로운 메트릭을 통해 MCQ의 대답 가능성을 측정하며, 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 두 번째 논문은 여러 개의 counterfactual을 생성 및 집합적 의사결정을 통해 모델의 인과 관계 이해도를 높이는 것을 목표로 합니다. 세 번째 논문인 Synth-Empathy는 프롬프트를 이용한 감정 데이터 생성, 도메인 지식을 사용한 품질 선정, 다양성 선정 등 세 가지 단계로 데이터를 생성 및 관리합니다.

- **Performance Highlights**: 첫 번째 논문의 KDA_disc와 KDA_cont는 사용성 측면에서 MCQ 평가와 강한 상관 관계를 보여주며, 이를 통해 교육적 평가에 유용한 도구임이 입증되었습니다. 두 번째 논문은 대조 학습과 집합적 의사결정을 통해 다양한 차원에서 기존 방법들에 비해 현저한 성능 향상을 이뤄냅니다. 세 번째 논문에서는 인간 노동 없이 고품질 감정 데이터를 생성하여, 여러 벤치마크에서 최첨단(SOTA) 성능을 달성하였습니다.



### Defending Jailbreak Attack in VLMs via Cross-modality Information Detector (https://arxiv.org/abs/2407.21659)
Comments:
          12 pages, 9 figures, ACL ARR 2024 June Submission

- **MCQ Generation Paper**: [{"What's New": '자동 MCQ 생성의 교육적 가치를 평가하기 위한 새로운 메트릭, 지식 종속 가능성(KDA)를 제안합니다. KDA는 특정 사실에 대한 학생의 지식을 평가할 수 있는 MCQ의 능력을 평가합니다.'}, {'Technical Details': 'KDA는 인간 설문조사에서 학생 응답을 기반으로 측정되며, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.'}, {'Performance Highlights': 'KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 가지며, 여러 전문가가 라벨링한 MCQ 품질 지표를 예측하는 강력한 예측력을 보였습니다.'}]

- **NLP Robustness Paper**: [{"What's New": '대비 학습과 반사실적 증대를 활용하여 NLP 모델의 강건성을 향상시키는 새로운 접근법을 제안합니다.'}, {'Technical Details': '기존 방법과 달리, 여러 개의 반사실적(counterfactual) 데이터셋을 생성하고 집합적 의사결정(collective decisions)을 통해 각 용어의 인과관계를 확인합니다.'}, {'Performance Highlights': '우리의 접근법은 다양한 차원에서 1) 반사실적 강건성, 2) 도메인 간 일반화, 3) 희소 데이터의 일반화에서 상당한 개선을 이루었습니다.'}]

- **Vision Language Models (VLMs) Security Paper**: [{"What's New": 'VLMs의 탈옥 공격(jailbreak attacks)을 탐지하기 위한 플러그 앤 플레이 방식의 감지기인 CIDER를 제안합니다.'}, {'Technical Details': 'CIDER는 선형 사전-탐지 모듈로, 학습된 디노이저를 사용하여 텍스트 및 이미지의 시맨틱 유사성을 기반으로 악의적으로 왜곡된 입력을 식별합니다.'}, {'Performance Highlights': 'CIDER는 백서(white-box) 및 흑서(black-box) VLMs와 공격 방법 모두에 대한 높은 탐지 성공률과 낮은 계산 비용을 보였습니다.'}]



### Towards Achieving Human Parity on End-to-end Simultaneous Speech Translation via LLM Agen (https://arxiv.org/abs/2407.21646)
Comments:
          Authors are listed in alphabetical order by last name. Demonstrations and human-annotated test sets are available at this https URL

- **What's New**: 새로운 MCQ 생성 평가 메트릭으로 '지식 종속 가능성(KDA)'를 제안하여 교육적 가치를 고려한 자동 평가를 한다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 메트릭은 단어 유사성만을 고려하지만 KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 중심으로 설계되었다. KDA_disc와 KDA_cont는 인간 답변 데이터를 기반으로 하여 KDA를 측정하며, 언어 모델을 사용하여 자동 평가를 수행한다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실 세팅에서 사용성과 강한 상관관계를 나타냈다. 기존 n-gram 기반 메트릭과 결합할 때 다양한 전문가가 평가한 MCQ 품질 측정에서 높은 예측력을 보여준다.



### Zero-Shot Cross-Domain Dialogue State Tracking via Dual Low-Rank Adaptation (https://arxiv.org/abs/2407.21633)
Comments:
          Accepted by ACL 2024

- **What's New**: 이 논문에서는 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안하여, MCQ의 대답 가능성(answerability)을 측정하고, 그 교육적 가치를 평가합니다. 또한, 제로샷(dialogue state tracking)을 위한 Dual Low-Rank Adaptation (DualLoRA)을 소개하며, 이를 통해 대화 시스템이 새로운 도메인으로의 전환을 쉽게 할 수 있도록 한다는 점을 다룹니다.

- **Technical Details**: 기존 MCQ 생성 평가 메트릭들(BLEU, ROUGE, METEOR)은 텍스트 유사성에만 집중하지만, KDA는 학생의 문제 해결 능력을 기반으로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용하여 학생의 행동을 모방함으로써 KDA를 근사합니다. 대화 상태 추적(DST)에서는 DualLoRA를 도입하여, 두 개의 LoRA 구성을 통해 대화 컨텍스트 처리와 프롬프트 최적화를 각각 처리, 모델의 전체 레이어에서 프롬프트의 영향을 유지할 수 있게 합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 다양한 전문가-라벨 MCQ 품질 측정 지표에서 예측력을 보여주었습니다. 또한, DualLoRA는 MultiWOZ와 SGD 데이터셋에서 기존의 제로샷 모델보다 높은 Joint Goal Accuracy (JGA)를 기록하며 뛰어난 성능을 입증했습니다.



### TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods (https://arxiv.org/abs/2407.21630)
- **What's New**: 이번 달에는 교사들이 학생 평가를 위한 새로운 MCQ 자동 생성 평가 메트릭이 제안되었습니다. '지식 종속 가능성(KDA)'이라는 이 메트릭은 기존 BLEU, ROUGE, METEOR의 한계를 극복하며, MCQ가 학생의 지식을 얼마나 잘 평가할 수 있는지를 측정합니다. 또한, 딥 러닝 모델의 강건성을 높이기 위한 새로운 접근법과 텍스트 저작권 은폐(Obfuscation)를 위한 혁신적 방법이 소개되었습니다.

- **Technical Details**: MCQ 자동 생성 평가에서는 KDA와 이를 기반으로 한 KDA_disc와 KDA_cont 메트릭을 소개했습니다. 이 메트릭들은 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방하여 평가합니다. 딥러닝 모델의 강건성을 높이기 위한 연구에서는 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 결합해 단어 인과관계를 규명하는 방법을 제시했습니다. 저작권 은폐 연구에서는 'TAROT'이라는 새로운 모델을 제안하며, 이는 정책 최적화(policy optimization)를 이용해 텍스트의 저자 정체성을 감추면서도 유용성을 최적화합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가 결과에 의해 실제 강의실 설정에서도 높은 사용성을 보였습니다. 대조 학습을 이용한 접근법은 반사실적 강건성, 크로스 도메인 일반화, 희소 데이터에서의 일반화에서 상당한 성능 향상을 달성했습니다. TAROT 모델은 영화 리뷰, 블로그 글, 학술 문서 등 다양한 데이터셋에 적용하여 저자 식별의 정확성을 크게 줄이면서도 유용성을 유지했습니다.



### PMoE: Progressive Mixture of Experts with Asymmetric Transformer for Continual Learning (https://arxiv.org/abs/2407.21571)
- **What's New**: 이번 주 AI 뉴스레터에서는 자동 생성된 다지선다형 질문(MCQ) 평가 메트릭, 자연어 처리(NLP) 모델의 강인성 확보를 위한 새로운 접근법, 그리고 대형 언어 모델(LLM)에서 발생하는 망각을 최소화하는 방법에 대해 다룹니다.

- **Technical Details**: [{"What's New": '자동 MCQ 생성 평가를 위한 새로운 메트릭, KDA(Knowledge Dependent Answerability)를 제안합니다.', 'Technical Details': '기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에 중점을 두어 교육적 가치를 반영하지 못합니다. 그러나 KDA는 학생의 지식 평가 능력을 측정할 수 있도록 설계되었습니다. 또한 KDA를 자동으로 측정할 수 있는 KDA_disc와 KDA_cont를 소개합니다. 이는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성에 대해 강한 상관관계를 가짐을 보여주었습니다. 또한, 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 높은 예측력을 보였습니다.'}, {"What's New": 'NLP 모델의 강인성을 확보하기 위한 대립 학습(contrastive learning)과 반사실 증대법(counterfactual augmentation)을 결합한 새로운 접근법을 제안합니다.', 'Technical Details': '기존의 증대법은 사람이나 기계가 반사실을 추가하는 방식인데, 이는 여전히 허위 패턴(spurious patterns)에 영향을 받습니다. 본 논문에서는 여러 개의 반사실을 생성하고 집합적 의사 결정(collective decisions)을 통해 더 강인하게 단어들의 인과관계를 파악하는 방법을 제안합니다.', 'Performance Highlights': '이 접근법은 반사실 강인성, 도메인 간 일반화, 데이터 부족 상황에서의 일반화 등 다양한 차원에서 유의미한 성능 향상을 달성했습니다.'}, {"What's New": '대형 언어 모델(LLM)의 지속학습(continual learning) 문제를 해결하기 위해 PMoE (Progressive Mixture of Experts with Asymmetric Transformer)를 도입합니다.', 'Technical Details': 'PMoE는 비대칭적인 설계로 얕은 레이어는 일반적인 지식을 유지하고, 깊은 레이어는 새로운 지식을 학습하도록 설계되었습니다. 새로운 지식을 효율적으로 할당하는 라우터(router)와 점진적으로 추가되는 전문가들로 구성됩니다.', 'Performance Highlights': 'TRACE 데이터셋과 일반 언어 이해 데이터셋에서 PMoE는 이전 최첨단 방법보다 우수한 성능을 보였습니다. 특히, LoRA와 비교하여 지속학습에서의 유연성과 효율성 면에서 매우 효과적이었습니다.'}]



### Generative Sentiment Analysis via Latent Category Distribution and Constrained Decoding (https://arxiv.org/abs/2407.21560)
- **What's New**: 이 논문에서는 MCQ (Multiple Choice Questions; 객관식 문제) 생성의 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭인 KDA (Knowledge Dependent Answerability)를 제안합니다. 또한, MCQ의 대답 가능성을 학생의 지식을 기반으로 평가하는 자동화된 방법인 KDA_disc와 KDA_cont를 소개합니다.

- **Technical Details**: KDA 메트릭은 학생들이 특정 목표 사실(target fact)에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성을 측정하는 것입니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 이 방법은 인간 설문 조사 결과를 기반으로 한 측정 방법을 포함하며, 실제 교실 환경에서의 사용성과 높은 상관관계를 보였습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었으며, 이들 방법들은 전문가가 라벨링한 다양한 MCQ 품질 측정 지표들에 대해 높은 예측 능력을 가집니다.



### Tracing Intricate Cues in Dialogue: Joint Graph Structure and Sentiment Dynamics for Multimodal Emotion Recognition (https://arxiv.org/abs/2407.21536)
Comments:
          Submitted

- **What's New**: [{'Multiple Choice Question Generation': '자동 MCQ 생성은 교육자의 평가 작업 시간을 줄일 수 있지만, 기존의 평가 메트릭은 교육적 가치를 고려하지 않습니다. 이를 해결하기 위해 우리는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다.'}, {'Robustness in NLP Models': '최근 딥 러닝 모델이 NLP 작업에서 높은 정확성을 보였지만, spurious 패턴에 의존해 robustness가 제한됩니다. 이 문제를 해결하기 위해 대조 학습과 counterfactual 증강을 활용한 방법을 제안합니다.'}, {'Multimodal Emotion Recognition': '대화에서의 다중 모달 감정 인식(MERC)은 다양한 모달 정보를 통합하여 감정 상태를 정확하게 인식하는 작업입니다. 이를 위해 우리는 GraphSmile이라는 새로운 접근 방식을 제안하여 감정 인식의 정확성을 높였습니다.'}]

- **Technical Details**: [{'Multiple Choice Question Generation': 'KDA 메트릭은 대상 사실에 대한 학생의 지식을 평가하는 능력을 기반으로 MCQ의 대답 가능성을 측정합니다. 또한, KDA 메트릭을 모방한 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안하여 인간의 설문 조사를 기반으로 측정합니다.'}, {'Robustness in NLP Models': "기존의 증강 방법과 달리, 우리의 접근 방식은 하나의 counterfactual이 아닌 '여러 개의' counterfactual을 생성하여 집합적 의사 결정을 통해 각 용어의 인과 관계를 robust하게 감독합니다."}, {'Multimodal Emotion Recognition': 'GraphSmile은 GSF와 SDP 모듈로 구성되어 있습니다. GSF 모듈은 그래프 구조를 활용하여 크로스 모달 및 내부 모달 감정 의존성을 계층별로 통합합니다. SDP 모듈은 발화 간 감정 변화의 동적 탐지를 통해 모델의 감정 변화 인식 능력을 향상시킵니다.'}]

- **Performance Highlights**: [{'Multiple Choice Question Generation': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, 이 메트릭들은 n-gram 기반의 유사성 메트릭과 결합될 때 더욱 강력한 예측력을 나타냅니다.'}, {'Robustness in NLP Models': '우리의 접근 방식은 다양한 차원에서 significant improvements를 달성했습니다: counterfactual robustness, cross-domain generalization, scarce data에서의 generalization.'}, {'Multimodal Emotion Recognition': 'GraphSmile은 다중 벤치마크 데이터셋에서 뛰어난 성능을 발휘하며, 기존 모델들보다 복잡한 감정과 감정 패턴을 더 잘 처리합니다.'}]



### Data Contamination Report from the 2024 CONDA Shared Task (https://arxiv.org/abs/2407.21530)
Comments:
this https URL

- **What's New**: 이 논문은 자동 출제 MCQ 생성의 평가 방법이 교육적 가치를 고려하지 않는다는 문제점을 지적하고, 이를 해결하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안합니다.

- **Technical Details**: 제안된 KDA 메트릭은 학생의 지식 상태를 반영하여 MCQ의 응답 가능성을 측정합니다. 이에 따라 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표가 도입되었으며, 이는 사전 훈련된 언어 모델(pre-trained language models)을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc 및 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 보여줍니다. 특히, 이 메트릭들을 n-gram 기반 유사성 메트릭과 결합하면 여러 전문가가 라벨링한 MCQ 품질 지표에 대해 강한 예측력이 있음이 증명되었습니다.



### Generative Expressive Conversational Speech Synthesis (https://arxiv.org/abs/2407.21491)
Comments:
          14 pages, 6 figures, 8 tables. Accepted by ACM MM 2024

- **MCQ Evaluation Metric Improvement**: [{"What's New": '자동으로 생성된 Multiple Choice Questions(MCQ)에 대한 평가 메트릭을 개선하기 위해 새로운 자동 평가 메트릭, KDA(Knowledge Dependent Answerability)를 제안했습니다. 이는 기존 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사성 메트릭이 교육적 가치를 간과하는 문제를 해결합니다.', 'Technical Details': 'KDA는 학생들이 주어진 목표 사실(target fact)에 대한 지식을 바탕으로 문제를 해결할 수 있는지를 측정합니다. 이를 위해 인간 설문조사를 기반으로 KDA 측정 방법을 먼저 제시한 후, 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.', 'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 입증했습니다. n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링한 MCQ 품질 척도에 대해 강력한 예측력을 보여줍니다.'}]

- **Robust NLP Models Using Counterfactuals**: [{"What's New": '최근 NLP 태스크에서 인간보다 뛰어난 정확성을 보이는 딥 모델들이 심사 인과관계(spurious correlation)에 의존하는 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 방법을 제안했습니다.', 'Technical Details': '기존의 증강 방법들은 사람이 반사실적 사례를 직접 추가해야 하거나 데이터셋 내에 반사실적 사례를 찾는 방식을 사용하였으나, 제안된 방법은 여러 개의 반사실적 사례를 합성하고, 이러한 사례들에 대한 예측 분포의 집합적 의사 결정을 통해 각 용어의 인과성을 감소시킵니다.', 'Performance Highlights': '실증 결과, 제안된 방법은 속성 기반 합성의 태스크 모델 편향에 덜 민감하여 반사실적 robustness, 도메인 간 일반화(cross-domain generalization), 그리고 부족한 데이터에서의 일반화에서 유의미한 개선을 보여줍니다.'}]

- **Expressive Conversational Speech Synthesis**: [{"What's New": '대화형 음성 합성(conversational speech synthesis)을 개선하기 위해 GPT-Talker라는 새로운 생성 표현 CSS 시스템을 제안했습니다. 이는 다중 회선 대화의 멀티모달 정보를 불연속 토큰 시퀀스로 변환하고 GPT로 예측하여 사용자 에이전트 대화 컨텍스트를 포괄적으로 통합합니다.', 'Technical Details': 'GPT-Talker 시스템은 다중 회선 대화 히스토리의 멀티모달 정보를 불연속 토큰 시퀀스로 변환하고 이를 무결하게 통합하여 사용자와 에이전트의 대화 컨텍스트를 형성합니다. GPT를 통해 에이전트의 응답의 의미론 및 스타일 지식을 포함한 토큰 시퀀스를 예측하고, 사용자에게 피드백을 전달하는 대화식 음성을 VITS로 합성합니다.', 'Performance Highlights': '새로운 대규모 Natural CSS Dataset(NCSSD)은 자연스럽게 기록된 대화식 음성과 TV 쇼에서 추출된 대화를 포함하며 중국어와 영어로 구성되어 총 236시간의 녹음을 포함합니다. 주관적 및 객관적 평가 모두에서 GPT-Talker 모델이 자연스러운 표현력 측면에서 다른 최신 CSS 시스템보다 현저한 성능 향상을 보여줍니다.'}]



### Maverick: Efficient and Accurate Coreference Resolution Defying Recent Trends (https://arxiv.org/abs/2407.21489)
Comments:
          Accepted at main conference of ACL 2024. 15 pages

- **What's New**: 이 논문에서는 새로운 자동 평가 메트릭, 지식 종속 가능성(KDA), 를 제안하여 MCQ(선다형 질문)의 교육적 가치를 평가하고 학생의 지식 평가 능력을 측정합니다. 또한, 여러 개의 counterfactual을 생성하여 robustness를 향상시키는 방법과 적은 파라미터로도 높은 성능을 발휘하는 Coreference Resolution 시스템 'Maverick'을 소개합니다.

- **Technical Details**: MCQ 평가 메트릭으로는 BLEU, ROUGE, METEOR 대신 KDA를 사용하여 학생의 대답 가능성을 측정합니다. 또한, KDA를 human survey 기반으로 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 그 외, contrastive learning과 counterfactual augmentation을 활용한 복수의 counterfactual을 생성하여 보다 robust하게 인과관계를 파악하는 방법을 제시합니다. 마지막으로, Maverick은 특정 태스크에 최적화된 모델로서, 0.006 배의 메모리만 사용하고 170 배 빠른 추론 속도로 CoNLL-2012 benchmark에서의 state-of-the-art 성능을 자랑합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 높은 상관관계를 보이며 n-gram 기반 유사성 메트릭들과 결합 시 강한 예측력을 가지고 있습니다. 새로운 방법론은 다양한 차원에서 큰 성능 향상을 보였으며, Maverick은 적은 자원으로도 기존의 state-of-the-art 모델을 능가하는 성능을 발휘합니다.



### On the Problem of Text-To-Speech Model Selection for Synthetic Data Generation in Automatic Speech Recognition (https://arxiv.org/abs/2407.21476)
Comments:
          Accepted at the SynData4GenAI 2024 workshop

- **What's New**: 이번 연구에서는 자동 다중선택 문제(MCQ) 생성을 평가할 새로운 메트릭인 지식 종속 대답 가능성(KDA)를 제안하였습니다. 이 메트릭은 기존의 BLEU, ROUGE, METEOR 등의 비슷한 단어 기반의 평가와 달리, 생성된 MCQ가 학생의 지식을 얼마나 잘 평가할 수 있는지를 중점적으로 평가합니다. 또한, 심리 불변(decoupling)의 견고성을 높이기 위해 대조 학습과 반사실적 데이터 증강(augmentation)의 새로운 접근 방식을 도입했습니다.

- **Technical Details**: KDA는 학생 응답 기반으로 측정됩니다. 이를 기반으로 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하였으며, 이는 사전 학습된 언어 모델을 통해 학생의 문제 해결 행동을 흉내냅니다. 또한, 다수의 counterfactual을 생성하여 집단적 의사 결정(collective decision)을 통해 보다 견고하게 인과 관계를 감독하는 방법을 제안했습니다.

- **Performance Highlights**: 연구 결과, KDA_disc와 KDA_cont는 사람의 평가와 강한 상관 관계를 가지며, n-gram 기반의 평가 메트릭과 결합 시 전문가가 라벨링한 여러 MCQ 품질 지표에서 강력한 예측력을 보였습니다. 또한, 반사실적 견고성, 크로스 도메인 일반화, 희소 데이터에서의 일반화에서 유의미한 성능 향상을 이루었습니다.



### Improving Faithfulness of Large Language Models in Summarization via Sliding Generation and Self-Consistency (https://arxiv.org/abs/2407.21443)
Comments:
          Long paper accepted at LREC-COLING 2024 (oral)

- **What's New**: 최근 아카이브에 게재된 여러 논문들을 요약했습니다. 이 논문들은 자동 MCQ 생성 평가 메트릭, NLP 태스크 모델의 robustness 향상 및 LLM을 활용한 텍스트 요약의 신뢰성 향상에 중점을 두고 있습니다.

- **Technical Details**: [{'paper_title': '자동 MCQ 생성 평가 메트릭', 'details': 'BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 학생 설문조사를 바탕으로 KDA를 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.'}, {'paper_title': 'NLP 태스크 모델의 Robustness 향상', 'details': '최근 deep model들이 spurious pattern에 의존하는 문제를 개선하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 방법을 제안합니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어나 문장의 인과관계를 좀 더 robust하게 파악합니다.'}, {'paper_title': 'LLM을 활용한 텍스트 요약 신뢰성 향상', 'details': 'LLM의 hallucination 문제를 해결하기 위해 SliSum이라는 새로운 요약 생성 전략을 제안합니다. 이 전략은 슬라이딩 윈도우(sliding windows)와 self-consistency를 활용하여 LLM이 텍스트 전체를 더 공정하고 신뢰성 있게 처리하도록 합니다. SliSum은 LLM을 통해 겹치는 윈도우로 나눠진 로컬 요약을 생성하고, 클러스터링과 다수결 알고리즘을 통해 전체 문서의 신뢰성 있는 요약을 만듭니다.'}]

- **Performance Highlights**: [{'paper_title': '자동 MCQ 생성 평가 메트릭', 'highlights': 'KDA_disc와 KDA_cont는 인간 전문가가 평가한 실제 강의실 세팅에서의 사용성과 높은 상관관계를 보였습니다. 또한, 이 두 메트릭을 n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대한 강력한 예측력을 가집니다.'}, {'paper_title': 'NLP 태스크 모델의 Robustness 향상', 'highlights': '대규모 실험에서 제안한 방법은 counterfactual robustness, cross-domain generalization, generalization from scarce data와 같은 다양한 측면에서 상당한 향상을 달성했습니다.'}, {'paper_title': 'LLM을 활용한 텍스트 요약 신뢰성 향상', 'highlights': 'SliSum은 LLaMA-2, Claude-2, GPT-3.5와 같은 다양한 LLM에서 단기 및 장기 텍스트 요약의 신뢰성을 크게 향상시켰습니다. 추가적인 fine-tuning이나 리소스 없이도 유창성과 정보량을 유지하며 신뢰성을 보장하는 결과를 보였습니다.'}]



### QuestGen: Effectiveness of Question Generation Methods for Fact-Checking Applications (https://arxiv.org/abs/2407.21441)
Comments:
          Accepted in CIKM 2024 as a short paper 4 pages and 1 page references

- **What's New**: 이 논문에서는 교육적 가치를 평가하지 못하는 기존 MCQ 생성 평가 메트릭(BLEU, ROUGE, METEOR)의 한계를 극복하기 위해 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭을 제안합니다. KDA는 MCQ의 대답 가능성(answerability)을 측정하여 학생의 지식 평가 능력을 평가합니다.

- **Technical Details**: KDA는 학생의 응답을 바탕으로 측정되며, 이를 기반으로 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다. 논문에서는 교육 전문가가 수기로 평가한 데이터를 사용하여 이 방법들이 실제 강의실에서의 사용성과 강한 상관관계를 가지는 것을 입증합니다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측력을 보였습니다. 이를 통해 이 새로운 평가 메트릭이 교육적 가치를 잘 반영할 수 있음을 보여줍니다.



### Cost-Effective Hallucination Detection for LLMs (https://arxiv.org/abs/2407.21424)
- **What's New**: 교사들이 학생 평가에 소비하는 시간을 크게 줄일 수 있는 자동 다지선다형 질문 (MCQ) 생성은 여전히 교육적 가치를 고려하지 않는 기존 평가 기준 때문에 제한되고 있습니다. 이를 해결하기 위해 우리는 목표 사실(Target Fact)에 대한 학생의 지식을 평가하는 능력을 측정하는 새로운 자동 평가 메트릭인 KDA (Knowledge Dependent Answerability)를 제안합니다.

- **Technical Details**: KDA는 주어진 목표 사실에 대한 지식을 바탕으로 MCQ의 답변 가능성을 측정합니다. 우리는 먼저 학생 응답을 이용한 인간 조사로 KDA를 측정하는 방법을 보여줍니다. 그 다음, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 가짐을 입증했습니다. 또한, 이를 n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대한 강력한 예측력을 보입니다.



### Dancing in Chains: Reconciling Instruction Following and Faithfulness in Language Models (https://arxiv.org/abs/2407.21417)
Comments:
          preprint

- **What's New**: 최근 논문에서는 Multiple Choice Questions(MCQ)의 자동 생성 평가 메트릭이 주로 n-gram 유사성에 기반하고 있어 교육적 가치인 학생의 지식을 평가하는 능력을 고려하지 못한다고 지적하고 있다. 이를 해결하기 위해 지식 종속 응답 가능성(KDA)이라는 새로운 자동 평가 메트릭을 도입했다. 또 다른 논문에서는 NLP 모델의 강건성이 spurious pattern에 의해 제한되는 문제를 해결하기 위해 대조 학습과 counterfactual augmentation을 제안한다. 마지막 논문에서는 언어 모델(LM)이 인간의 지시를 따르는 것과 정확한 응답을 하는 것 사이의 상충관계를 연구하고, 이를 해결하기 위해 Rejection Sampling for Continued Self-instruction Tuning (ReSet)이라는 방법을 제안한다.

- **Technical Details**: 지식 종속 응답 가능성(KDA)은 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 새로운 메트릭이다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했다. NLP 모델의 강건성 문제는 대조 학습과 집합적 의사 결정(collective decisions)을 통해 각 단어의 인과관계를 더 robust하게 파악하는 방법으로 해결하고자 했다. ReSet 방법은 Rejection Sampling을 이용해 높은 품질의 예시를 선택하고 다시 학습 데이터로 사용하여 모델의 성능을 향상시키는 접근 방식이다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보여주는 것으로 입증되었다. 또한 대조 학습을 통한 접근 방식은 다양한 차원에서의 성능 향상을 보여주었다: 1) counterfactual robustness, 2) cross-domain generalization, 3) generalization from scarce data. ReSet 방법을 통해 8,000개의 추가 학습 예시로 학습된 LM은 기존의 MTL 대비 성능 지표에서 최대 +18.8%까지 향상되었다. 데이터 양을 줄이면서도 더 높은 품질의 데이터로 학습하면 faithfulness에서 최대 31.3% 향상된 결과를 보였다.



### GEGA: Graph Convolutional Networks and Evidence Retrieval Guided Attention for Enhanced Document-level Relation Extraction (https://arxiv.org/abs/2407.21384)
- **What's New**: MCQ 생성 자동화는 교육자가 평가하는 시간을 절약하는데 중점이 있습니다. 기존의 n-그램 기반 평가 메트릭은 교육적 가치를 간과하므로, 새로운 평가 기준 Knowledge Dependent Answerability (KDA)를 제안하여 MCQ의 학습 평가 능력을 측정합니다. 또한, 자연어 처리(NLP) 모델이 spurious patterns에 의존해 빈약한 신뢰성을 보이는 상황을 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용합니다. 마지막으로, 문서 수준 관계 추출 (DocRE) 문제를 해결하기 위해 GEGA 모델을 제안하여 다중 엔티티 간의 복잡한 관계 추출을 개선합니다.

- **Technical Details**: KDA는 학생들의 지식에 기반한 MCQ의 대답 가능성을 측정합니다. 이를 위해 학생 응답 데이터를 사용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되었습니다. NLP 모델의 신뢰성 강화를 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust한 인과 관계 파악을 합니다. DocRE 문제 해결을 위해 GEGA 모델이 그래프 신경망과 다중 스케일 표현 집약을 사용하여 문서의 증거 문장을 추출하고 관계 추출 성능을 향상시킵니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 학습 환경에서 강한 상관관계를 보였으며, 다양한 전문가 평가 MCQ 품질 측정에서 예측력이 높다는 것이 입증되었습니다. 새로운 counterfactual augmentation 접근법이 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터 환경에서도 탁월한 성능 향상을 보여주었습니다. GEGA 모델은 DocRED 및 Re-DocRED 등 세 가지 벤치마크 데이터셋에서 기존 SOTA 모델 대비 전반적인 성능 향상을 기록했습니다.



### Performance of Recent Large Language Models for a Low-Resourced Languag (https://arxiv.org/abs/2407.21330)
- **What's New**: 몇 가지 새로운 연구 발표가 있었습니다. 첫 번째 연구는 중추적인 평가 메트릭을 개선하고자 MCQ 생성의 평가 방식을 혁신적으로 제안했습니다. 두 번째 연구는 NLP 모델들의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 이용하는 방법을 고안했습니다. 세 번째 연구는 최신 대형 언어 모델(LLM)들의 저자원 언어 성능을 평가하였습니다.

- **Technical Details**: [{'MCQ Generation Evaluation': '기존의 평가 메트릭 BLEU, ROUGE, METEOR가 MCQ의 교육적 가치를 평가하지 못한다는 문제점을 지적했습니다. 이에 대응하기 위해 Knowledge Dependent Answerability (KDA)라는 자동 평가 메트릭을 제안했습니다. KDA는 학생의 지식을 평가할 수 있는 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다.'}, {'NLP Robustness Improvement': '최근 deep models의 정확성이 뛰어나지만 spurious patterns에 의존하여 robustness가 제한적이라는 점을 지적했습니다. 기존은 사람이 counterfactual을 만들거나 유사-counterfactual을 찾는 방법이었으나, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust하게 인과관계를 파악하는 방법을 제안했습니다.'}, {'LLM Performance on Low-Resourced Languages': '최신 대형 언어 모델(LLM)들이 저자원 언어인 Sinhala에서 성능이 낮다는 문제점을 평가했습니다. Claude와 GPT 4는 바로 사용 가능한 상태에서 좋은 성능을 보였고, Llama와 Mistral은 성능이 낮았지만 몇 가지 가능성을 보여줬습니다.'}]

- **Performance Highlights**: [{'MCQ Generation Evaluation': 'KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성에서 강한 상관관계를 보여줬습니다. 또한, n-gram 기반의 유사성 메트릭과 결합하면 다양한 전문가-라벨링 MCQ 품질 측정에 대한 강력한 예측력을 가집니다.'}, {'NLP Robustness Improvement': '제안된 방법은 다양한 차원에서 significant improvements를 보여줬습니다: counterfactual robustness, cross-domain generalization, scarce data에서의 generalization.'}, {'LLM Performance on Low-Resourced Languages': 'Claude와 GPT 4가 이전 버전보다 훨씬 더 좋은 성능을 보였으며, Llama와 Mistral은 약간의 fine-tuning을 통해 개선 가능성이 보였습니다.'}]



### Beyond Silent Letters: Amplifying LLMs in Emotion Recognition with Vocal Nuances (https://arxiv.org/abs/2407.21315)
- **What's New**: 이번 뉴스레터에서는 출판된 최근 논문들을 소개합니다. 이 논문들은 자동 MCQ 생성, NLP의 강인성 개선, 음성 감정 인식 등 다양한 주제에 집중하고 있습니다.

- **Papers**: [{'title': 'Automatic Generation of Multiple Choice Questions with Knowledge Dependent Answerability (KDA)', "What's New": '이 논문은 기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 평가하지 못하는 문제를 해결하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.', 'Technical Details': 'KDA는 학생들이 특정 사실에 대한 지식을 가지고 있는 경우, 질문의 답변 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human study에서 KDA_disc와 KDA_cont는 실제 강의실에서의 사용성 및 전문가가 라벨링한 평가 메트릭 품질 측정값과 강한 상관관계를 보였습니다.'}, {'title': 'Robust NLP Models by Counterfactual Augmentation Using Contrastive Learning', "What's New": '최근 NLP 모델들의 정확성에도 불구하고, spurious pattern에 의존하여 강인성이 부족한 문제를 해결하기 위한 새로운 방법을 제안합니다.', 'Technical Details': '대조 학습 (contrastive learning)과 반사실적 증강법 (counterfactual augmentation)을 사용하여 여러 개의 counterfactual을 생성하고, 모델이 집합적 의사 결정을 통해 용어 각각의 인과성을 평가합니다.', 'Performance Highlights': '이 접근 방법은 다양한 차원에서 significant improvements을 달성했으며, 특히 반사실적 강인성, 도메인 간 일반화, 제한된 데이터에서의 일반화에서 뛰어난 성과를 보였습니다.'}, {'title': 'Emotion Detection in Speech Using Large Language Models (LLMs)', "What's New": 'LLM의 텍스트 처리 능력을 활용하고 오디오 입력을 처리하는 데 있어 LLM의 제약을 해결하기 위해 새로운 접근 방식을 제안합니다.', 'Technical Details': '음성 특징을 자연어 설명으로 번역하여 LLM 프롬프트에 통합함으로써, 아키텍처 수정 없이 멀티모달 감정 분석을 수행할 수 있습니다. 이 방법은 IEMOCAP 및 MELD 데이터셋에서 평가되었으며, 감정 인식 정확도에서 유의미한 개선을 보였습니다.', 'Performance Highlights': 'IEMOCAP 데이터셋에서 weighted F1 score가 70.111%에서 72.596%로 2 퍼센트 포인트 증가하는 성과를 기록했습니다.'}]



### Model Attribution in Machine-Generated Disinformation: A Domain Generalization Approach with Supervised Contrastive Learning (https://arxiv.org/abs/2407.21264)
Comments:
          10 pages, 2 figures, accepted at DSAA 2024

- **What's New**: 이 논문에서는 기계-생성 왜곡 정보 (machine-generated disinformation)의 출처 모델을 특정하는 문제를 다루고 있습니다. 이는 최근 LLM(대형 언어 모델: Large Language Models)들이 생성하는 텍스트가 사람의 글과 구분하기 어려운 수준이기 때문입니다. 이에 저자들은 Supervised Contrastive Learning을 활용하여 도메인 일반화 문제(Domain Generalization Problem)로 모델 귀속 문제를 해결하려고 합니다.

- **Technical Details**: 도메인 일반화(Domain Generalization)는 예측 모델이 여러 도메인에서 도메인-고유한(feature invariant features) 특징들을 무시하고 일관성 있는 특징을 학습하도록 돕는 기법입니다. 본 연구에서는 『open-ended』, 『rewriting』, 『paraphrasing』의 세 가지 일반적인 프롬프트 방법(prompting methods)과 『LLaMA-2』, 『ChatGPT』, 『Vicuna』의 세 가지 최신 LLM을 활용하여 모델을 평가합니다. Supervised Contrastive Learning (SCL)은 같은 클래스의 인스턴스를 가까이 두고, 다른 클래스의 인스턴스를 멀리 두도록 학습하는 기법으로, 메모리 절약 기능을 통해 많은 대조 예시(contrasting examples)를 효율적으로 처리할 수 있습니다.

- **Performance Highlights**: 결과는 제안한 모델이 다양한 프롬프트 조건에서 뛰어난 성능을 발휘하여, LLM 간 텍스트 출처를 정확하게 구분할 수 있음을 보여줍니다. 특히 『ChatGPT-4』의 문맥 학습 능력(in-context learning)을 활용하여 특정 LLM의 글쓰기 스타일을 인식하고 적용하는 과정에서도 뛰어난 적응력을 보여주었습니다. 그러나 이러한 접근 방식은 제공된 학습 예시의 품질과 관련성에 크게 의존하는 한계도 드러났습니다.



### Adaptive Pre-training Data Detection for Large Language Models via Surprising Tokens (https://arxiv.org/abs/2407.21248)
- **What's New**: 최근의 두 논문에서 제안한 새로운 평가 메트릭과 강화된 데이터 탐지 방법은 AI의 다양한 응용 분야에서 교육 가치와 모델의 데이터 안전성을 크게 향상시킵니다.

- **Technical Details**: {'MCQ 생성': '- KDA (Knowledge Dependent Answerability)라는 새로운 평가 메트릭을 제안하여 자동으로 생성된 MCQ의 교육적 가치를 평가합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', '데이터 탐지': "- LLM(pre-trained Large Language Models)의 데이터 탐지를 위한 새로운 방법인 SURP를 제안합니다. 이 방법은 Shannon 엔트로피를 사용하여 '놀라운 토큰(surprising tokens)'을 식별하고, 이 확인을 통해 모델이 보지 않은 데이터와 훈련 데이터 간의 차이를 증폭시킵니다."}

- **Performance Highlights**: {'MCQ 평가': '- Human evaluation을 통해 KDA_disc와 KDA_cont는 실제 학습 환경에서 강한 상관관계를 가졌으며, 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 높은 예측 능력을 보였습니다.', '데이터 탐지 성능': '- SURP는 다양한 벤치마크와 LLM에서 기존 방법보다 최대 29.5% 개선된 성능을 나타냈습니다.'}



### Decomposed Prompting to Answer Questions on a Course Discussion Board (https://arxiv.org/abs/2407.21170)
Comments:
          6 pages. Published at International Conference on Artificial Intelligence in Education 2023. Code repository: this https URL

- **What's New**: 최근 연구는 오토매틱 단답형 생성과 관련된 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 기존의 BLEU, ROUGE, METEOR가 교육적 가치를 반영하지 못하는 문제를 해결하고자 KDA는 학생들이 특정 사실에 대해 지식을 가지고 답할 수 있는 능력을 측정합니다.

- **Technical Details**: KDA는 학생 응답 기반의 human survey를 통해 측정됩니다. 또한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 language models를 사용해 학생들의 문제 해결 행위를 모방합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 내 사용성과 높은 상관관계를 가짐을 보여주었습니다. 또한, n-gram 기반의 유사성 메트릭과 결합할 때, 다양한 전문가가 라벨링한 MCQ 품질 측정에 강한 예측력을 보였습니다.



### Event-Arguments Extraction Corpus and Modeling using BERT for Arabic (https://arxiv.org/abs/2407.21153)
- **What's New**: [{'title': 'New Metric for Evaluating MCQ Quality', 'content': '교사들을 위해 자동으로 다지선다형 질문 (MCQ)를 생성하는데 도움이 되는 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)가 제안되었습니다. 이 새로운 메트릭은 기존의 BLEU, ROUGE, METEOR와 달리 교육적 가치를 평가합니다.'}, {'title': 'Robustness in NLP with Collective Counterfactuals', 'content': '최근 NLP 태스크에서 deep models의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 함께 여러 개의 역사실(counterfactuals)을 생성해 집합적으로 (collective decisions) 평가하는 방법이 제안되었습니다.'}, {'title': 'Arabic Event-Argument Extraction Corpus', 'content': '아랍어 언어 자원 부족 문제를 해결하기 위해 이벤트-서술어 추출(event-argument extraction)을 위한 새로운 코퍼스가 생성되었습니다. 이 코퍼스는 BERT 기반 방법을 사용하여 높은 성능을 보였습니다.'}]

- **Technical Details**: [{'title': 'KDA Metric', 'content': 'KDA는 학생 반응을 기반으로 MCQ의 정답 가능성을 측정합니다. KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭이, 사전 학습된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방하여 KDA를 예측합니다.'}, {'title': 'Collective Counterfactuals for Robustness', 'content': '역사실 셋을 생성하여 해당 셋의 예측 분포에 대해 집합적으로 결정하는 방법을 통해 단어들의 인과관계를 더 robust하게 감독하는 방식이 제안되었습니다. 이를 통해 형용 접근법이 임의의 상관관계에 덜 민감하게 만들고, 다양한 조건에서 성능을 향상시킵니다.'}, {'title': 'Arabic Event-Argument Extraction with BERT', 'content': 'BERT를 사용해 이벤트-서술어 추출 작업을 텍스트 함의(text entailment) 문제로 처리하는 새로운 방법론이 제안되었습니다. 이 방법은 94.01%의 F1 점수를 달성했습니다. 또한, 도메인 외 데이터셋으로 추가 평가하여 83.59%의 F1 점수를 기록했습니다.'}]

- **Performance Highlights**: [{'title': 'KDA Metric', 'content': '교사와 전문가가 실제 수업에서 사용할 수 있는 질문들에 대해 KDA_disc와 KDA_soft 메트릭이 강한 상관관계를 보였습니다.'}, {'title': 'Robustness in NLP', 'content': '새로운 역사실 생성 및 집합적 의사결정 접근방식은 1) 반증적인 robustness, 2) 크로스 도메인 일반화, 3) 희소한 데이터에서의 일반화 측면에서 중요한 향상을 달성했습니다.'}, {'title': 'Arabic Corpora for Event-Argument Extraction', 'content': '새로운 워조드-하다스(Wojood-Hadath) 코퍼스는 82.23%의 Kappa 점수와 87.2%의 F1 점수를 기록했습니다. BERT 기반 방법은 94.01%의 F1 점수를 달성했으며 추가 데이터셋에서도 83.59%의 F1 점수를 기록했습니다.'}]



### Enhancing Semantic Similarity Understanding in Arabic NLP with Nested Embedding Learning (https://arxiv.org/abs/2407.21139)
- **What's New**: 새로운 프레임워크인 Matryoshka Embedding Learning을 통해 Arabic nested embedding models를 훈련, 아랍어 NLP 다운스트림 작업에서 nested embedding 모델의 우수성을 강조했습니다.

- **Technical Details**: 다양한 문장 유사성 데이터셋을 아랍어로 번역하여 포괄적인 평가 프레임워크를 구성하고, 아랍어 자연 언어 추론 데이터셋을 기반으로 몇 가지 nested embedding 모델을 훈련했습니다.

- **Performance Highlights**: Matryoshka embedding 모델이 아랍어의 의미적 미세한 부분을 잘 포착하여 기존 모델을 최대 20-25%까지 뛰어넘는 성능을 보였고, Hugging Face를 통해 데이터셋과 모델을 공개하여 연구 및 적용을 촉진했습니다.



### Entropy, Thermodynamics and the Geometrization of the Language Mod (https://arxiv.org/abs/2407.21092)
Comments:
          18 pages

- **What's New**: 자동화된 다지 선다형 문제(Multiple Choice Questions, MCQ) 생성 및 평가에서 교육적 가치를 측정하는 새로운 메트릭으로 'Knowledge Dependent Answerability (KDA)'를 제안합니다. KDA는 학생이 특정 대상 사실에 대해 문제를 대답할 수 있는 능력을 측정합니다. KDA_disc와 KDA_cont라는 새로운 자동 평가 메트릭도 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Technical Details**: 기존의 MCQ 생성 평가 메트릭(BLEU, ROUGE, METEOR)은 단어의 n-gram 유사성만을 측정해 교육적 가치가 반영되지 않습니다. KDA는 인간 설문조사를 통해 측정되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용해 KDA를 근사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc 및 KDA_cont는 실제 강의실 환경에서의 사용성 및 여러 전문가가 레이블링한 MCQ 품질 측정치와 강한 상관관계가 있음을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합 시 여러 전문가가 평가한 MCQ 품질 측정치에 대해 강력한 예측력을 보였습니다.

- **What's New 2**: 대조 학습(contrastive learning) 및 반사실적 증강(counterfactual augmentation)을 통해 NLP 모델의 강인성을 향상시키는 방법을 제안합니다.

- **Technical Details 2**: 기존 방법들은 인간이 반사실적 데이터를 추가하거나 데이터셋에서 유사 반사실적 데이터를 찾는 방식이나, 여전히 spurious correlation에 영향을 받습니다. 이 논문에서는 여러 개의 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 단어 간의 인과 관계를 더 강인하게 파악하는 방법을 제안합니다.

- **Performance Highlights 2**: 적은 데이터 및 다양한 차원에서 대조 학습을 통해 얻은 반사실적 강건성(counterfactual robustness), 도메인 간 일반화, 희소 데이터에서의 일반화 능력에서 중요한 향상을 달성했습니다.



### Accelerating Large Language Model Inference with Self-Supervised Early Exits (https://arxiv.org/abs/2407.21082)
- **What's New**: 이번 논문에서는 MCQ (Multiple Choice Questions)의 자동 생성 평가를 위해 새로운 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 메트릭을 제안했습니다. 기존의 BLEU, ROUGE, METEOR 메트릭은 단순한 n-gram 기반의 유사성 평가에 그치고 있어 교육적 가치를 정확히 반영하지 못하는 문제를 해결합니다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 이를 위해 학생의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다. 이 메트릭들은 인간 조사에서 얻은 데이터를 바탕으로 학생 응답을 기반으로 KDA를 측정합니다. 또한, 이 새로운 메트릭들은 실제 강의실 세팅에서의 사용성에 대해 전문가들이 라벨링한 데이터와 강한 상관관계를 보입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합될 때 다양한 전문가 지정 MCQ 품질 측정에 대해 강력한 예측력을 보입니다. 이를 통해 MCQ의 교육적 유용성을 더 정확하게 평가하고, 궁극적으로는 학습 평가의 질을 향상시킬 수 있습니다.



### Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models (https://arxiv.org/abs/2407.21077)
- **MCQ Generation Paper**: [{"What's New": '이 논문은 자동 단답형 문제 생성(MCQ) 평가를 위한 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 기존의 평가 메트릭은 교육적인 가치를 고려하지 못하고, 단순히 n-gram 유사성에 의존하고 있습니다.'}, {'Technical Details': "KDA 메트릭은 MCQ의 답변 가능성을 측정하고, 학생의 지식 평가 능력을 평가합니다. 이를 위해 KDA를 student's responses를 기반으로 측정하고, 사전 훈련된 언어 모델을 사용하여 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안합니다."}, {'Performance Highlights': 'Human study 결과, KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 메트릭과 결합되었을 때, 전문가가 라벨링한 MCQ 품질 측정에 대해 강력한 예측 성능을 나타냅니다.'}]

- **NLP Robustness Paper**: [{"What's New": 'Contrastive learning과 counterfactual augmentation을 활용하여 NLP 태스크에서의 모델 robust성을 향상시키는 방법을 제안합니다.'}, {'Technical Details': '기존 방법들은 human이 직접 counterfactual을 추가하거나 모델이 데이터 내에서 유사한 것들을 찾는 방식을 사용했으나, 이는 spurious correlation에 영향을 받을 수 있습니다. 제안된 방법은 다수의 counterfactual을 생성하고, collective decision-making을 통해 단어의 인과관계를 robust하게 평가합니다.'}, {'Performance Highlights': '제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization 측면에서 기존 방법보다 우수한 성능을 보였습니다.'}]

- **Code Generation with Genetic-Instruct Paper**: [{"What's New": '코드 생성 능력을 갖춘 LLM을 위한 scalable synthetic instruction generation 방법인 Genetic-Instruct를 소개합니다. 이 알고리즘은 self-instruction을 이용하여 한정된 seed 샘플로부터 여러 개의 synthetic 샘플을 생성합니다.'}, {'Technical Details': 'Genetic-Instruct는 진화 알고리즘에서 영감을 받아, mutation과 crossover를 사용하여 새로운 instruction을 생성합니다. Instructor-LLM이 새로운 instruction을 생성하고, Coder-LLM이 코드를 생성합니다. Judge-LLM은 생성된 instruction과 코드를 평가하여 품질 검사를 통과한 샘플만을 상위 풀에 추가합니다.'}, {'Performance Highlights': '생성된 synthetic 데이터셋을 사용하여 여러 open-source software coding LLMs를 fine-tuning한 결과, 기존 베이스라인보다 향상된 코드 생성 정확도를 보여주었습니다.'}]



### Occam's Razor and Bender and Koller's Octopus (https://arxiv.org/abs/2407.21070)
Comments:
          In ACL 2024 Workshop on Teaching NLP (TeachNLP 2024)

- **What's New**: 교육자가 학생 평가에 소비하는 시간을 획기적으로 줄일 수 있는 자동 생성 Multiple Choice Questions(MCQ)의 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안했습니다. 기존의 BLEU, ROUGE, METEOR 메트릭들은 단순히 데이터셋 내의 샘플과의 단어 유사성만을 평가했으나, KDA 메트릭은 MCQ가 학생의 지식을 어떻게 측정하는지를 평가합니다. 또한, 최근 NLP 태스크에서 높은 정확성을 자랑하는 deep model의 robustness(강인성) 향상을 위해 contrastive learning과 counterfactual augmentation 기법을 활용하는 연구도 있습니다.

- **Technical Details**: KDA는 학생의 대상 사실에 대한 지식을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했으며, 이는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 방식을 모사합니다. 또한, deep model의 robustness 문제를 해결하기 위해 여러 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 오류 패턴을 줄이는 방법을 탐구합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 보임을 확인했습니다. 또한, 기존의 n-gram 유사성 메트릭과 결합했을 때, 다양한 전문가가 라벨링한 MCQ 품질 측정치에서 강력한 예측 능력을 보였습니다. deep model의 robustness 연구에서는 counterfactual augmentation 방식이 다양한 측면에서 모델의 정확도를 향상시켰음을 실험적으로 확인했습니다.

- **Additional Information**: 이 외에도 Bender와 Koller의 논문 'Climbing toward NLU: on meaning form, and understanding in the age of data'에 대한 학생 교육 자료를 제공합니다. 학생들이 논문의 주장을 비판적으로 분석할 수 있도록 다양한 반론과 함께 학습 자료를 포함시키며, 실제 강의에서도 활발한 토론을 유도합니다.



### Exploring Genre and Success Classification through Song Lyrics using DistilBERT: A Fun NLP Ventur (https://arxiv.org/abs/2407.21068)
- **Automatic MCQ Generation**: [{"What's New": '자동 다중 선택 질문 생성 (MCQ generation)을 위한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)을 제안하였습니다. 이는 학생이 해당 사실에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성을 평가합니다.'}, {'Technical Details': '우리는 학생의 응답을 바탕으로 KDA를 측정하고, 사전에 훈련된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭은 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.'}, {'Performance Highlights': '본 연구는 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합되었을 때, 다양한 전문가 레이블 MCQ 품질 측정에 대한 예측력이 강함을 입증하였습니다.'}]

- **NLP Model Robustness**: [{"What's New": '최근 NLP 태스크에서 사람보다 나은 정확성을 보이는 딥 모델들의 robustness 문제를 해결하기 위해, 대비 학습 (contrastive learning)과 counterfactual augmentation을 이용한 새로운 접근법을 제안하였습니다.'}, {'Technical Details': "기존의 방법과 달리, 우리는 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 각 용어의 인과 관계를 견고하게 감독하는 방법을 제안합니다."}, {'Performance Highlights': '실험 결과, 제안된 접근법은 반대사실 (counterfactual) robustness, 도메인 간 일반화 (cross-domain generalization), 그리고 데이터를 적게 이용한 일반화 (generalization from scarce data)에서 유의미한 성능 향상을 보여주었습니다.'}]

- **Song Lyrics Comprehension**: [{"What's New": '노래 가사를 종합적으로 이해하는 문제에 대해, 장르 분류 (genre classification), 뷰 기반 성공 예측 (view-based success prediction), 그리고 발매 연도 예측 (release year prediction)에 중점을 둔 NLP 접근법을 제시하였습니다.'}, {'Technical Details': '장르 분류를 위해 DistilBERT 모델을 사용하고, BERT 임베딩을 활용하여 발매 연도를 예측하였습니다. 지원 벡터 머신 (Support Vector Machines)은 발매 연도를 예측하는 데에 있어 가장 낮은 RMSE를 기록하며 다른 모델들을 능가하였습니다.'}, {'Performance Highlights': '장르 분류에서 65%의 정확도와 성공 예측에서 79%의 정확도를 달성하였으며, 발매 연도 예측에서는 14.18이라는 최저 RMSE를 기록하였습니다.'}]



### ELP-Adapters: Parameter Efficient Adapter Tuning for Various Speech Processing Tasks (https://arxiv.org/abs/2407.21066)
- **What's New**: 최근 발표된 논문에서는 자동 다지선다형 문항 생성(MCQ) 평가 도구로, 신뢰성 있는 평가 메트릭 Knowledge Dependent Answerability(KDA)를 제안했습니다. 또한, 데이터의 잠재적 인과관계 및 여러 NLP 태스크에 대해 Contrastive Learning과 Counterfactual Augmentation을 활용한 견고성 향상 방법도 제시되었습니다. 더불어, ELP-adapter 튜닝을 활용하여 음성 처리 태스크에서의 파라미터 효율성을 높은 수준으로 유지하면서도 최적의 성능을 달성하는 방법도 논의되었습니다.

- **Technical Details**: KDA 메트릭은 특정 목표 사실에 대한 학생의 지식을 평가할 수 있도록 MCQ의 대답 가능성(answerability)을 측정합니다. 자동 평가 메트릭 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다. Contrastive Learning과 Counterfactual Augmentation을 사용하는 방법은 자동으로 counterfactual을 생성하고 다수의 결정에 따라 단어들의 인과관계를 감시하는 방식입니다. ELP-adapter 튜닝은 세 가지 종류의 어댑터(E-adapter, L-adapter, P-adapter)를 통해 세부적인 음성 표현을 학습하고, 각 어댑터의 역할에 따라 후속 태스크에서의 성능을 향상시키는 방식입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트에서 사용성과 강한 상관관계를 보였습니다. 제안된 Robustness 방법은 여러 NLP 태스크에서 다양한 차원 (예: counterfactual robustness, cross-domain generalization, scarce data generalization)에서 성능 향상을 달성했습니다. ELP-adapter 튜닝은 ASR, ASV, SER, SIC와 같은 음성 처리 태스크에서 성능이 전체 파라미터를 튜닝한 것과 대등하거나 더 나은 성능을 보였으며, 학습 파라미터 수를 90% 줄이는 데 성공했습니다.



### LawLLM: Law Large Language Model for the US Legal System (https://arxiv.org/abs/2407.21065)
Comments:
          21 pages, 2 figures, accepted at the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024) for the Applied Research Paper track

- **What's New**: 최근 발표된 아카이브(arXiv) 논문들은 다양한 AI 모델과 평가 메트릭의 새로운 접근법을 제시하고 있습니다. 법률 분석, MCQ 생성, NLP의 강건성(robustness) 개선 등 다양한 분야에서의 혁신적인 연구들이 포함되어 있습니다.

- **Technical Details**: [{'MCQ Generation': "기존의 MCQ 자동 생성 평가 메트릭(BLEU, ROUGE, METEOR)들이 n-gram 기반 유사도에 초점을 맞추어 교육적 가치를 무시했다는 문제를 해결하기 위해, 'Knowledge Dependent Answerability(KDA)'라는 새로운 메트릭을 제안합니다. KDA는 학생이 대상 사실(target fact)을 알고 있을 경우 MCQ의 답변 가능성을 측정합니다."}, {'NLP Robustness': "최근 NLP 태스크에서 인간 이상의 정확성을 보이는 딥 모델이 많은데, 이 모델들이 'spurious pattern'에 의존하여 강건성이 제한됩니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 'counterfactual augmentation'을 통해 모델의 강건성을 개선하는 방법을 제안합니다."}, {'Legal Large Language Model (LawLLM)': '법률 언어의 복잡성과 특수 용어 때문에 사례 찾기와 판례 예측이 어려운 문제를 해결하기 위해 LawLLM을 제안합니다. LawLLM은 Similar Case Retrieval(SCR), Precedent Case Recommendation(PCR), Legal Judgment Prediction(LJP) 등의 다양한 법률 태스크를 동시에 수행할 수 있습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성에서 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 레이블 MCQ 품질 측정에서 강한 예측력을 보였습니다.'}, {'NLP Robustness': '제안된 접근법을 사용하면 집단적 의사 결정을 통해 단어들의 인과관계를 더 robust하게 파악할 수 있어, 다양한 측면에서(Bias, Cross-domain Generalization, 데이터가 부족한 상황에서의 Generalization) 의미 있는 개선을 보였습니다.'}, {'LawLLM': 'LawLLM은 zero-shot 및 few-shot 시나리오에서 기존 기준 모델보다 일관되게 우수한 성능을 보였습니다. 이 모델은 특히 유사한 사례와 판례를 명확히 구분함으로써 법률 분야의 다양한 요구를 충족시킬 수 있습니다.'}]



### Improving noisy student training for low-resource languages in End-to-End ASR using CycleGAN and inter-domain losses (https://arxiv.org/abs/2407.21061)
Comments:
          10 pages (2 for references), 4 figures, published in SIGUL2024@LREC-COLING 2024

- **What's New**: 다양한 연구들이 교육과 언어 처리 분야의 최신 기법들을 활용하여 교육적 가치를 증대하거나 언어 모델의 robustness를 강화하려는 시도를 보이고 있습니다. 특히, Multiple Choice Question (MCQ) 자동 생성과 관련하여 교육적인 평가 지표를 도입하거나, spurious pattern 의존성을 줄이기 위해 counterfactual augmentation을 사용하는 새로운 방법이 제안되었습니다. 또한, 저자들은 저자원 환경에서도 성능을 극대화하기 위해 CycleGAN과 hyperparameter tuning을 통해 강화된 새로운 자동 음성 인식 방법을 개발했습니다.

- **Technical Details**: MCQ 자동 생성과 관련하여, 기존의 평가 메트릭(BLEU, ROUGE, METEOR)은 데이터셋 내의 gold sample과의 유사성만 평가하여 교육적 가치를 간과한다는 문제가 있습니다. 이를 해결하기 위해, 저자들은 Knowledge Dependent Answerability (KDA)를 새롭게 제안하여 학생들이 특정 사실에 대해 가지고 있는 지식을 평가하는 능력을 측정합니다. 이는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 구현되며, pretrained language models를 활용하여 학생들의 문제 해결 행동을 모방합니다. 언어 모델의 robustness 강화와 관련해서는, contrastive learning와 counterfactual augmentation을 활용하여 spurious pattern 의존성을 줄이는 새로운 방법이 제안되었습니다. 따라서 여러 개의 counterfactual을 생성하고 집합적인 의사 결정을 통해 각 term의 인과관계를 robust하게 감독하는 방법을 사용합니다. 저자원 언어 환경의 음성 인식에서는, paired speech-text와 unlabeled speech가 매우 제한된 상황에서 외부 텍스트를 활용하여 모델 성능을 향상시키는 방법을 제안합니다. 이를 위해 CycleGAN과 inter-domain losses를 강화한 자동 하이퍼파라미터 튜닝 기법을 사용합니다.

- **Performance Highlights**: MCQ 자동 생성 연구에서, KDA_disc와 KDA_cont는 실제 강의실 세팅에서의 사용성과 높은 상관관계를 가지며, 우수한 예측력을 보입니다. 또한, 언어 모델의 robustness를 강화하는 방법에서는 counterfactual robustness, cross-domain generalization, 그리고 scarce data generalization에서 상당한 개선 효과를 보였습니다. 저자원 언어 음성 인식 연구에서는, 제안된 성능 강화 방법을 통해 기존 baseline model 대비 최대 20%의 Word Error Rate (WER) 감소를 달성했습니다.



### Using Large Language Models for the Interpretation of Building Regulations (https://arxiv.org/abs/2407.21060)
Comments:
          Presented at the 13th Conference on Engineering, Project and Production Management

- **What's New**: 최근 건설 프로젝트에서 준수 확인(compliance checking)의 중요성을 높이기 위하여 우리는 대형 언어 모델(GPT)을 사용하여 건물 규정을 자동으로 번역하는 방안을 제안합니다. 이를 통해 상당한 시간과 비용을 절감할 수 있을 것으로 기대됩니다.

- **Technical Details**: BIM(Building Information Models)을 활용하여 디지털 건물 설계 데이터를 사용했던 기존 방법과 달리, GPT-3.5 및 GPT-4와 같은 대형 언어 모델(LLMs)은 건물 규정을 LegalRuleML로 번역하는 데 활용됩니다. 몇 샷 학습(few-shot learning) 설정에서 GPT-3.5에게 몇 가지 예제 번역을 제공하여 기본 형식을 학습하게 하고, 시스템 프롬프트를 사용하여 LegalRuleML 표현을 명확히 합니다.

- **Performance Highlights**: GPT-3.5는 광범위한 사전 학습을 통해 전문가 도메인 지식을 내포하고 있는 것을 확인했습니다. 또한, 체인 오브 간단한 추론(chain-of-thought reasoning) 및 자기 일관성(self-consistency) 전략이 이 사례에 적용될 수 있는지를 조사하였으며, 이는 ACC(자동 준수 확인)를 더 효율적이고 효과적으로 지원할 수 있습니다.



### Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks (https://arxiv.org/abs/2407.21059)
- **What's New**: 이번 AI 뉴스레터에서는 여러 최신 연구의 핵심 내용을 다룹니다. 첫째, 자동 Multiple Choice Questions (MCQ) 생성의 평가 메트릭에서 교육적 가치를 반영하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 둘째, 자연어 처리(NLP) 모델의 robustness 향상을 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법이 소개됩니다. 셋째, Retrieval-augmented Generation (RAG) 시스템의 모듈화된 framework인 Modular RAG가 제안되어, 복잡한 RAG 시스템의 유연성 및 확장성을 높이는 방법을 설명합니다.

- **Technical Details**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Details': 'BLEU, ROUGE, METEOR와 같은 전통적 평가 메트릭이 아니라, MCQ의 대답 가능성 및 교육적 가치를 평가하는 Knowledge Dependent Answerability (KDA) 메트릭을 제안합니다. KDA_disc와 KDA_cont를 사용해 사람의 문제 해결 행동을 모방하여 자동 평가를 수행합니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': '최근 모델들이 spurious patterns에 의존하는 문제를 해결하고자 counterfactual augmentation과 contrastive learning 기법을 사용합니다. 다양한 counterfactual 생성과 집합적 의사 결정을 통해 인과관계를 더 robust하게 파악합니다.'}, {'Paper': 'Modular RAG Framework', 'Details': 'Retrieval-augmented Generation (RAG) 시스템의 복잡성을 해결하기 위해 모듈화된 RAG framework를 제안합니다. 모듈, 서브모듈, 연산자로 나누어 시스템 유지 보수성과 유연성을 강화하고, RAG Flow를 통해 다양한 RAG 방법론을 통합합니다.'}]

- **Performance Highlights**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Highlights': 'KDA_disc와 KDA_cont는 KDA와 실제 강의실 세팅에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 메트릭과 결합 시 전문가가 레이블한 다양한 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다.'}, {'Paper': 'Robustness in NLP Models', 'Highlights': '집단적 결정 방식을 통해 기존 모델들의 진귀 데이터에서의 일반화 성능과 cross-domain generalization, counterfactual robustness에서 의미 있는 향상을 달성했습니다.'}, {'Paper': 'Modular RAG Framework', 'Highlights': '모듈화된 RAG 시스템은 유연성과 확장성이 뛰어나, 다양한 데이터 소스 및 작업 시나리오에 맞게 모듈과 연산자를 조합할 수 있습니다. 이는 RAG 기술의 실용적 배치와 연구 방향에 새로운 기회를 제공합니다.'}]



### Understanding the Interplay of Scale, Data, and Bias in Language Models: A Case Study with BER (https://arxiv.org/abs/2407.21058)
- **What's New**: 이 연구는 Multiple Choice Questions (MCQ) 자동 생성 평가 메트릭의 교육적 가치를 개선하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, 모델 스케일과 사전 학습 데이터가 사회적 편견에 미치는 영향을 탐구하였습니다.

- **Technical Details**: [{'MCQ Generation': '기존의 MCQ 생성 평가 메트릭(BLEU, ROUGE, METEOR)은 단순히 생성된 MCQ와 데이터셋의 골드 샘플 간의 n-gram 기반 유사성에 집중하는데, 이는 교육적 가치를 무시합니다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였고, 이는 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가합니다.', 'Methodology': '학생 설문조사를 통해 KDA를 측정하는 방법을 먼저 제시한 후, 이를 모방하기 위해 사전 학습된 언어 모델을 활용한 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안했습니다.'}, {'Robustness in NLP': '최근 딥 모델들이 NLP 태스크에서 높은 정확성을 보이지만, 여전히 spurious pattern에 의존하여 robustness가 떨어지는 문제가 있습니다. 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 사용하여 모델의 robustness를 향상시키고, 여러 개의 반사실을 생성해 집합적 의사 결정을 통해 단어들의 인과관계를 평가하는 방법을 제안했습니다.'}, {'Social Biases in LLMs': '대형 언어 모델의 크기와 사전 학습 데이터가 사회적 편견에 미치는 영향을 연구했습니다. BERT 모델을 사례로, 모델 크기와 사전 학습 데이터(Wikipedia, Common Crawl)에 따라 나타나는 사회적 편견을 분석했습니다.'}]

- **Performance Highlights**: [{'MCQ Evaluation': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. n-gram 기반의 유사성 메트릭과 결합될 때, KDA_disc와 KDA_cont가 전문가가 라벨링한 다양한 MCQ 품질 척도를 예측하는 데 강력합니다.'}, {'NLP Robustness': '집합적 의사 결정을 통해 모델의 할당 편향(task model bias)을 줄여주며, counterfactual robustness, cross-domain generalization, scarce data 상황에서의 일반화 등 여러 면에서 성능 향상을 이루었습니다.'}, {'Bias Analysis': '모델 크기가 증가함에 따라 Wikipedia로 사전 학습된 모델은 성별 고정관념이 증가하고, Common Crawl로 사전 학습된 모델은 독성(toxicity)이 증가함을 발견했습니다. 그러나 사후 학습(fine-tuning)에서는 모델 크기가 증가할수록 편견이 감소하는 경향을 보였습니다.'}]



### Multi-group Uncertainty Quantification for Long-form Text Generation (https://arxiv.org/abs/2407.21057)
- **What's New**: 이번 주에는 여러 중요한 논문들이 발표되었습니다. 첫 번째로는 자동 MCQ 생성에 대한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안한 연구가 있습니다. 두 번째로는 대조 학습과 counterfactual augmentation을 결합하여 언어 모델의 robust를 향상시키려는 연구가 있었습니다. 마지막으로는 대형 언어 모델(LLM)의 사실적 정확성을 개선하고 이에 대한 불확실성을 정량화하려는 연구입니다.

- **Multiple Choice Question Generation**: [{"What's New": '기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 평가하지 못한다는 문제를 해결하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)가 제안되었습니다.', 'Technical Details': 'KDA는 학생들이 대상 사실을 알고 있는 경우, MCQ(Multiple Choice Question)의 대답 가능성을 측정합니다. 이 메트릭은 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 포함하며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지고 있음이 입증되었습니다. 또한, n-gram 기반의 유사성 메트릭과 결합된 KDA_disc와 KDA_cont가 전문가가 평가한 MCQ 품질 측정치와 강한 예측력을 가진다는 결과가 나왔습니다.'}]

- **Robust NLP Models**: [{"What's New": '최근 NLP 태스크에서 super-human 정확성을 나타내지만, spurious 패턴에 의존하는 문제를 해결하기 위해 새로운 방법이 제안되었습니다.', 'Technical Details': '기존의 방법들은 인간이 counterfactual을 추가하거나 자동으로 유사한 것을 찾는 방식이었으나, 새로운 방법은 여러 개의 counterfactual을 생성하고 collective decisions으로 단어들의 인과관계를 파악하는 것입니다.', 'Performance Highlights': '이 접근법은 attribution-based synthesis의 task model bias에 덜 민감하여, 다양한 차원에서 significant improvements를 달성하였습니다. 예를 들어, counterfactual robustness, cross-domain generalization, 스카스 데이터로부터 일반화 등이 있습니다.'}]

- **Uncertainty Quantification in LLMs**: [{"What's New": '대형 언어 모델(LLM)의 사실적 오류와 환각(hallucinations)을 줄이기 위한 불확실성 정량화 방법을 연구하였습니다.', 'Technical Details': '이 연구는 구체적으로 개별 클레임 수준의 보정(calibration)을 통해 불확실성을 평가하고, 전체 출력에 대한 conformal prediction을 통해 불확실성을 평가합니다. 또, 여러 하위 그룹에 대해서도 이러한 불확실성 보장(multi-group)을 제공하기 위해 multicalibration과 multivalid conformal prediction을 사용합니다.', 'Performance Highlights': '전기의 실행 결과에 따르면 멀티 그룹 속성들을 사용하는 것이 전반적인 성능과 그룹별 성능에서 모두 개선 효과가 있음을 보여줍니다. 특히, standard (marginal) calibration과 conformal prediction 기법들보다 우수한 성과를 보였습니다.'}]



### What Matters in Explanations: Towards Explainable Fake Review Detection Focusing on Transformers (https://arxiv.org/abs/2407.21056)
- **What's New**: 이번 논문에서 자동으로 다중 선택 질문(MCQ)을 생성하고 평가하는 새로운 방법을 제안했습니다. BLEU, ROUGE, METEOR와 같은 기존 메트릭은 교육적 가치를 평가하지 못합니다. 본 논문에서는 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 도입하여 MCQ의 대답 가능성을 측정하고, 학생의 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: KDA는 특정 대상 사실(target fact)에 대한 MCQ의 답변 가능성을 측정합니다. 이를 위해 학생 응답을 기준으로 KDA를 측정하는 방법을 제시하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: 인간 평가 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 세트에서 전문가가 라벨링한 사용성과 강한 상관관계를 가짐을 확인했습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, 이 메트릭들은 다양한 전문가가 라벨링한 MCQ 품질 측정에 대해 높은 예측력을 보여줍니다.



### Bailicai: A Domain-Optimized Retrieval-Augmented Generation Framework for Medical Applications (https://arxiv.org/abs/2407.21055)
- **MCQ Generation and Evaluation**: [{"What's New": '최근 자동으로 객관식 문제(MCQ)를 생성하는 방법이 제안되었습니다. 하지만 기존의 평가 메트릭(BLEU, ROUGE, METEOR)은 생성된 문항의 교육적 가치를 고려하지 않습니다. 본 논문에서는 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 제안합니다.'}, {'Technical Details': 'KDA는 학생의 지식을 기반으로 MCQ의 답변 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 추가로 제안하며, 사전 훈련된 언어 모델(pre-trained language models)을 활용하여 학생의 문제 해결 행동을 모방합니다.'}, {'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 상황에서도 강한 상관관계를 가지면서 높은 예측력을 보이는 것으로 나타났습니다.'}]

- **Robust NLP Models via Counterfactuals**: [{"What's New": '최근의 deep model들은 NLP 태스크에서 높은 정확도를 보였지만, spurious 패턴에 의존하여 robustness가 낮은 문제를 보였습니다. 본 논문에서는 contrastive learning과 counterfactual augmentation을 활용한 새로운 접근 방법을 제안합니다.'}, {'Technical Details': "기존의 데이터 증강 방법은 사람의 개입이나 자동화를 통해 counterfactual을 추가하는 방식이었습니다. 그러나 이 논문에서는 '여러 개의' counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 더 robust하게 인과관계를 파악하는 방법을 도입합니다."}, {'Performance Highlights': '집합적 의사 결정을 통해 모델의 바이어스 민감성을 줄이고, 다방면에서 유의한 성능 개선을 보았습니다. 특히 counterfactual robustness, cross-domain generalization, 그리고 소규모 데이터 학습에서 두드러진 성과를 보였습니다.'}]

- **Bailicai Framework for Medical LLMs**: [{"What's New": '의학 분야에서는 기존의 large language models(LLMs)가 적절한 성능을 보이며 활용되고 있지만, 여전히 한계가 존재합니다. 본 논문에서는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 의학 분야에서의 성능을 개선한 Bailicai 프레임워크를 제안합니다.'}, {'Technical Details': 'Bailicai는 네 가지 서브 모듈로 구성됩니다: Medical Knowledge Injection, Self-Knowledge Boundary Identification, Directed Acyclic Graph Task Decomposition, 및 Retrieval-Augmented Generation입니다. 이를 통해 입력 쿼리가 외부 지식이 필요한지를 판별하고, 필요시 RAG를 실행합니다.'}, {'Performance Highlights': 'Bailicai는 기존의 의학 분야 LLM과 RAG 접근법을 넘어서는 성능을 보였습니다. 여러 의학적 벤치마크에서 GPT-3.5의 성능을 초과했으며, hallucinations 문제를 완화했습니다.'}]



### Sentiment Reasoning for Healthcar (https://arxiv.org/abs/2407.21054)
Comments:
          Preprint, 18 pages

- **MCQ Generation**: [{"What's New": '자동 MCQ 생성 평가를 위한 새로운 메트릭인 지식 종속 가능성(KDA, Knowledge Dependent Answerability)을 제안. 이는 학생이 대상 사실에 대한 지식을 가지고 답할 수 있는지 평가하는 능력을 측정합니다.'}, {'Technical Details': 'KDA는 학생의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Performance Highlights': 'Human evaluation에서 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합 시 여러 전문가 레이블 MCQ 품질 측정에 대한 강한 예측력을 보였습니다.'}]

- **NLP Robustness**: [{"What's New": '대조 학습과 반사실적 증강을 통해 NLP 모델의 robustness(견고성)를 향상시키는 방법을 제안합니다.'}, {'Technical Details': '기존 방법들은 데이터셋에서 반사실적인 예제를 사람이 추가하거나 모델이 자동으로 찾아내야 하지만, 새로운 접근법은 여러 개의 반사실적 예제를 생성하여 집합적 결정을 통해 단어들의 인과관계를 더욱 견고하게 감독합니다.'}, {'Performance Highlights': '제안된 방법은 반사실적 견고성, 도메인 간 일반화, 그리고 적은 데이터에서의 일반화에서 상당한 향상을 달성했습니다.'}]

- **Sentiment Analysis**: [{"What's New": '새로운 태스크인 Sentiment Reasoning을 제안하며, 음성과 텍스트 모달리티를 모두 포함하는 멀티모달 멀티태스크 프레임워크와 데이터를 소개합니다.'}, {'Technical Details': '여러 감정 카테고리에 대해 AI가 감정의 맥락을 이해하고 복잡한 인간의 의사소통 요소를 고려하도록 Reasoning 능력을 포함합니다. 이를 통해 더 신뢰할 수 있는 감정 분석을 달성합니다.'}, {'Performance Highlights': '합리성(Reasoning)이 추가된 훈련이 텍스트와 ASR 설정 모두에서 모델 성능을 향상시키며, 생성된 합리성들이 인간이 생성한 합리성과 유사한 의미론을 유지하는 것을 발견했습니다.'}]



### Knowledge Models for Cancer Clinical Practice Guidelines : Construction, Management and Usage in Question Answering (https://arxiv.org/abs/2407.21053)
- **What's New**: 최근 MCQ(Multiple Choice Question) 자동 생성 기술이 교사들이 학생 평가에 소요되는 시간을 대폭 줄일 수 있도록 지원하는 연구가 발표되었습니다. 그러나 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ의 교육적 가치를 충분히 평가하지 못하는 문제를 가지고 있습니다. 이에 따라, 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안하여 MCQ의 대답 가능성을 측정하고, 이로써 학생의 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: KDA는 Human Survey를 통해 학생들의 응답을 기반으로 측정하며, 이를 자동화하기 위해 사전 훈련된 언어 모델을 활용한 KDA_disc, KDA_cont 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 학생들의 문제 해결 행동을 모방하여 KDA를 근사화합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 확인하였습니다. 또한, n-gram 기반의 유사성 메트릭과 결합했을 때 다양한 전문가 레이블 MCQ 품질 척도를 예측하는 데 있어 강력한 예측력을 가지고 있음을 확인하였습니다.



### Table-Filling via Mean Teacher for Cross-domain Aspect Sentiment Triplet Extraction (https://arxiv.org/abs/2407.21052)
Comments:
          Accepted by CIKM2024

- **What's New**: 자동 MCQ 생성의 평가메트릭을 개선하기 위해 지식 종속 가능성(KDA)이라는 새로운 메트릭을 도입했습니다. 이는 MCQ가 특정 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: 기존의 n-gram 기반 메트릭은 BLEU, ROUGE, METEOR 등을 사용하나 교육적 가치를 평가하지 않습니다. 새로운 KDA 메트릭은 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont는 pretrained language models를 사용하여 학생의 문제 해결 행동을 모사하여 KDA를 근사합니다.

- **Performance Highlights**: Human evaluation에서 KDA_disc와 KDA_cont는 실제 교실 세트에서의 사용성과 강한 상관관계를 보였습니다. 또한 n-gram 기반 메트릭과 결합할 경우, 다양한 전문가가 라벨링한 MCQ의 품질 측정에 대해 높은 예측력을 보여줍니다.



### An Active Inference Strategy for Prompting Reliable Responses from Large Language Models in Medical Practic (https://arxiv.org/abs/2407.21051)
Comments:
          25 pages, 4 figures

- **What's New**: 이 논문에서는 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 또한, 기존의 데이터셋에서 counterfactual을 생성하는 방식과 달리 여러 개의 counterfactual을 집합적으로 평가하여 단어들 간의 인과관계를 더 robust하게 평가하는 방법도 소개됩니다. 마지막으로, 의료 분야에서 대규모 언어 모델(LLMs)의 응용 가능성을 논의하며, 특정 도메인에 검증된 데이터셋을 사용하고 액터-크리틱 프롬프트 프로토콜을 도입하는 프레임워크를 제안합니다.

- **Technical Details**: 첫 번째 논문에서는 MCQ 평가를 위한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하고, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다. 두 번째 논문에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 모델의 인과 관계 이해를 평가합니다. 세 번째 논문은 LLM의 지식 베이스를 검증된 의료 정보로 제한하고, 액터-크리틱 프롬프트 프로토콜을 도입하여 정확성과 신뢰성을 높이는 방법을 제안합니다.

- **Performance Highlights**: 첫 번째 논문의 경우, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 가집니다. 두 번째 논문은 counterfactual robustness, cross-domain generalization, and generalization from scarce data에서 유의미한 성능 향상을 달성했습니다. 세 번째 논문에서는 CBT-I 전문가들이 LLM의 응답을 블라인드 형식으로 평가했으며, LLM의 응답이 종종 인간 치료사의 적절한 응답을 능가하는 높은 평점을 받았습니다.



### Artificial Intelligence in Extracting Diagnostic Data from Dental Records (https://arxiv.org/abs/2407.21050)
Comments:
          11 pages, 2 tables, 3 figures, under review

- **What's New**: 자동 MCQ(다중 선택 질문) 생성 및 평가를 위한 새로운 방법론이 제시되었고, 최신 NLP 기법을 활용해 데이터의 신뢰성 문제를 개선하려는 시도가 이루어졌습니다. 또한, 치과 기록에서 구조화되지 않은 데이터를 구조화된 진단 정보로 변환하는 AI 및 NLP 기반 방법론이 제안되었습니다.

- **Technical Details**: 첫 번째 연구에서는 MCQ의 교육적 가치를 측정하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안했습니다. KDA는 학생의 지식을 평가하는 MCQ의 대답 가능성(answerability)을 측정합니다. 두 번째 연구에서는 대조 학습(contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용해 NLP 모델의 강인성을 향상시키는 방법을 탐구했습니다. 마지막 연구에서는 GPT-4를 활용해 치과 기록의 구조화되지 않은 텍스트에서 진단 정보를 추출하는 RoBERTa 모델을 미세 조정했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 지표는 실제 강의실에서의 사용성과 강한 상관관계를 보였습니다. 반사실적 증가 기반 방법론은 대조 학습과 함께 다양한 차원에서 강한 일반화 성능을 보였습니다. 또한, RoBERTa 모델의 진단 정확도는 두 사이트에서 각각 0.99와 0.98의 점수를 기록했으며, 세부 유형(category)에 대한 정확도도 매우 높았습니다.



### Evaluating Long Range Dependency Handling in Code Generation Models using Multi-Step Key Retrieva (https://arxiv.org/abs/2407.21049)
Comments:
          29 pages, 18 figures

- **What's New**: 교사의 학습 평가 시간을 절감하기 위해 자동 생성 다지선다형 문제(MCQ)를 활용한 연구가 활발하다. 새로운 측정 기준인 Knowledge Dependent Answerability(KDA)를 제안해 학생의 지식 평가를 보다 정확하게 측정할 수 있다.

- **Technical Details**: 기존 BLEU, ROUGE, METEOR 같은 n-gram 기반의 유사성 측정 지표는 교육적 가치를 판단하지 않는다. 새로운 KDA 메트릭은 인간 조사에서 측정한 학생 응답을 기반으로 하며, KDA_disc 및 KDA_cont 자동 측정 지표를 통해 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서 전문가가 평가한 사용성 측정과 강한 상관관계를 보였다. 또한, n-gram 기반 유사성 지표와 결합했을 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강력한 예측력을 보여준다.



### APTNESS: Incorporating Appraisal Theory and Emotion Support Strategies for Empathetic Response Generation (https://arxiv.org/abs/2407.21048)
Comments:
          Appectped to CIKM2024

- **What's New**: 최근 연구는 자동 MCQ (Multiple Choice Questions) 생성과 NLP 모델의 강건성 및 공감 반응 생성 향상에 초점을 맞추고 있습니다. 특히, MCQ 생성에서 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭이 제안되었으며, NLP 모델의 강건성을 위한 대조 학습 및 반증 증강 방법이 소개되었습니다. 또한, 공감 반응 생성 분야에서는 감동적 반응 생성 프레임워크인 APTNESS가 소개되었습니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 평가 메트릭은 문장의 n-gram 기반 유사성만을 평가하지만, 교육적 가치를 고려하지 않습니다. 새로운 지식 종속 가능성(KDA) 메트릭은 MCQ의 대답 가능성을 중점으로 삼아 평가합니다. 특히, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭이 제안되었습니다. 이들은 사전 훈련된 언어 모델을 사용해 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.', 'NLP Model Robustness': '기존 대조 학습 및 반증 증강 방법은 사람의 개입이나 데이터셋 내 반증을 찾는 방식이었으나, 이 논문에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과 관계를 더 robust하게 평가하는 방법을 제안했습니다. 이 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization을 목표로 합니다.', 'Empathetic Response Generation': '공감 반응을 생성하는 데 있어서 APTNESS라는 프레임워크는 감정 팔레트(emotional palette)와 감정 지원 전략(emotional support strategy)을 통합하여 공감 능력을 향상시킵니다. 감정 팔레트는 7개의 주요 감정과 23개의 하위 감정으로 구성되어 있으며, 평가 이론(appraisal theory)을 사용해 이 팔레트를 세분화합니다. 이 프레임워크는 의미 검색 기법을 통해 외부 리소스를 활용하여 LLM의 공감 능력을 강화합니다.'}]

- **Performance Highlights**: [{'MCQ Generation Performance': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반의 유사성 메트릭과 결합했을 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대한 예측력이 뛰어났습니다.', 'NLP Model Robustness Performance': '제안된 방법은 다양한 차원에서 현저한 성능 향상을 보였습니다. 특히, counterfactual robustness, 도메인 간 일반화(cross-domain generalization), 그리고 제한된 데이터에서의 일반화(generalization from scarce data)에서의 성능이 탁월했습니다.', 'Empathetic Response Generation Performance': 'APTNESS 프레임워크는 ED와 ET라는 감정 대화 데이터셋을 사용해 자동 평가를 진행한 결과, 인지적 공감(cognitive empathy)과 감정적 공감(affective empathy) 모두에서 모델의 공감 능력을 크게 향상시켰습니다.'}]



### Promises and Pitfalls of Generative Masked Language Modeling: Theoretical Framework and Practical Guidelines (https://arxiv.org/abs/2407.21046)
Comments:
          ICML 2024

- **What's New**: 최근 제안된 논문에서 MCQ(다지선다형 질문) 자동 생성의 새로운 평가 메트릭과, NLP 태스크에서의 연관 학습 및 반사실적 증강 방법, 그리고 생성 마스킹 언어 모델(GMLMs)에 대한 이론적 분석 및 최적화 전략들이 소개되었습니다.

- **Technical Details**: [{'title': 'MCQ 자동 생성 평가', 'details': '기존의 BLEU, ROUGE 및 METEOR 평가 메트릭이 교육적 가치를 고려하지 않는 한계를 극복하기 위해 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭이 제안되었습니다. KDA는 학생의 지식을 평가하는 능력을 측정하며, 학생 응답을 모방한 언어 모델을 사용하여 이를 자동화합니다.'}, {'title': 'NLP 태스크에서의 반사실적 증강', 'details': '기존 방법들이 흩어진 상관관계(spurious correlations)에 영향을 받는 문제를 해결하기 위해 대조 학습과 반사실적 증강을 결합한 새로운 접근법이 제안되었습니다. 다수의 반사실적 항목을 생성하고 집합적인 의사결정을 통해 각 항목의 인과성을 보다 robust하게 감독할 수 있게 합니다.'}, {'title': 'GMLMs의 이론적 프레임워크 개발', 'details': '순차적이고 단방향적인 생성 한계를 가진 현재의 AR 모델을 대신해, 조건부 확률을 통해 데이터를 생성하는 GMLMs에 대한 수학적 프레임워크가 개발되었습니다. 이를 통해 추론 속도와 샘플 복잡성의 균형을 맞추고, 기계 번역과 같은 특정 태스크에서 성능 향상을 증명했습니다.'}]

- **Performance Highlights**: [{'title': 'MCQ 평가 메트릭의 성능', 'details': 'KDA_disc와 KDA_cont가 실제 클래스룸 환경에서 전문가들이 라벨링한 사용성 평가와 높은 상관관계를 보였으며, n-그램 기반 유사성 메트릭과 결합했을 때 다양한 전문가 라벨링 MCQ 품질 측정에서 강한 예측력을 보였습니다.'}, {'title': 'NLP 태스크에서의 성능 향상', 'details': '새로운 접근법을 통해 여러 차원에서 향상된 결과를 확인했습니다. 특히 counterfactual robustness, cross-domain generalization, 그리고 희소 데이터 일반화에서 매우 좋은 성과를 보였습니다.'}, {'title': 'GMLMs의 속도 및 품질 개선', 'details': 'GMLMs는 기계 번역 태스크에서 기존 AR 모델 대비 2-3배의 속도 향상을 보였으며, 품질 저하가 거의 없었습니다. 이론적으로 증명된 큰 마스킹 비율, 사용자 지정 어휘, AR 모델로부터의 distillation과 같은 주요 구성 요소들의 중요성을 실험을 통해 확인했습니다.'}]



### Unlocking the Potential: Benchmarking Large Language Models in Water Engineering and Research (https://arxiv.org/abs/2407.21045)
- **What's New**: 이 논문은 최근 대형 언어 모델(LLMs)들이 다양한 분야에서의 응용 가능성에 대한 관심이 증가하고 있는 가운데, LLMs가 수자원 공학 및 연구 작업에서 '수자원 전문가 모델'로 효과적으로 기능할 수 있는지를 탐구합니다. 이를 위해 WaterER이라는 도메인 특화 벤치마크 스위트를 설정하여 983개의 수자원 공학 및 연구 관련 작업을 평가합니다.

- **Technical Details**: WaterER 벤치마크 스위트는 '폐수 처리', '환경 복원', '음료수 처리 및 배급', '위생', '혐기성 소화' 및 '오염물 평가' 등 여섯 가지 카테고리로 구분된 과제를 포함합니다. 평가된 모델은 GPT-4, GPT-3.5, Gemini, GLM-4, ERNIE, QWEN 및 Llama3 등 총 7개 입니다. 이 논문에서는 특히 GPT-4가 다양한 복잡한 수자원 공학 및 연구 작업을 처리하는 데 강점을 보였고, Gemini는 학문적 맥락에서 특화된 능력을 보여주었으며, Llama3는 중국어 수자원 공학 질문에 가장 강력한 답변 능력을 보인다고 강조합니다.

- **Performance Highlights**: 현재 LLMs는 '오염물 및 관련 수질 모니터링 및 평가'에 대한 논문의 연구 격차를 정확하게 생성하는 데 뛰어난 성능을 보였고, '폐수 처리 프로세스', '환경 복원', '음료수 처리'와 관련된 논문 제목을 작성하는 데 더 적합했습니다. 이 연구는 WaterER 벤치마크를 소개하여 수자원 공학 및 연구에서 LLM의 예측 신뢰성을 평가하는 첫 시도입니다. 이 표준화된 평가 프레임워크는 LLM 기술의 미래 발전을 촉진하고, 이러한 모델들이 진정한 '수자원 전문가'로 발전하는데 중요한 역할을 할 것입니다.



### CP-Prompt: Composition-Based Cross-modal Prompting for Domain-Incremental Continual Learning (https://arxiv.org/abs/2407.21043)
- **What's New**: 최근 NLP 태스크에서 정확도 향상을 목표로 한 다양한 접근법이 소개되었습니다. 자동 다지선다형 질문 생성(MCQ), 대조 학습 기반 개선 및 도메인 증가 학습(DIL)에서 단일 모델이 시퀀스 도메인에서 새로운 데이터 학습을 지속할 수 있도록 하는 방법들이 주목받고 있습니다.

- **Technical Details**: {'MCQ Generation': "BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭은 교육적 가치를 충분히 반영하지 않기 때문에, 우리는 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안합니다. KDA는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하며 MCQ의 교육적 유효성을 측정합니다.", 'Robustness': "최근 모델들이 spurious pattern을 이용함에 따라 난제들이 보고되었습니다. 이를 해결하기 위해, 우리는 대조 학습과 counterfactual augmentation을 활용하여 보다 robust한 성능을 발휘하는 방법을 제안합니다. 이 방법은 '여러 개의' counterfactual을 생성하여 단어들의 인과관계를 파악해 모델의 편향성을 줄이는 것을 목표로 합니다.", 'Domain-Incremental Learning (DIL)': 'CP-Prompt라는 새로운 프레임워크를 제안하여, 모델이 계속해서 다양한 데이터 스타일에서 학습하도록 합니다. 이 프레임워크는 공통 프로프트 전략과 특정 도메인 인식을 위한 Prefix-One 프롬프트를 결합하여 인과 관계 기억 상실을 줄입니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 강의실 사용성과 강한 상관관계를 가지며, 기존의 n-gram 기반 메트릭과 결합하여 다양한 평가 지표에서 뛰어난 예측력을 보여줍니다.', 'Robustness': '우리의 접근법은 대조 학습을 통해 과제 모델 편향에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 측면에서 뛰어난 성능을 보입니다.', 'DIL': 'CP-Prompt는 0.22%의 추가 파라미터만 조정하여, 기존 최첨단 샘플 프리 베이스라인보다 크게 개선된 2.3%의 성능 향상을 보여줍니다.'}



### They Look Like Each Other: Case-based Reasoning for Explainable Depression Detection on Twitter using Large Language Models (https://arxiv.org/abs/2407.21041)
- **What's New**: MCQ 자동 생성의 효율성이 향상될 것으로 기대되며, 기존 평가 메트릭들이 교육적 가치를 충분히 반영하지 못하는 문제점을 해결하기 위해 지식 종속 가능성(KDA)를 도입하였습니다. 또한, 최근 NLP 과제에서 높은 정확도를 보이는 deep model들이 여전히 spurious pattern에 의존하는 한계를 극복하기 위해 대조 학습(contrastive learning) 및 counterfactual augmentation을 제안하였습니다. 마지막으로, 트위터 기반 우울증 감지 프레임워크 ProtoDep을 새롭게 도입하여 설명 가능하고 투명한 진단 방법을 제시하였습니다.

- **Technical Details**: MCQ 평가 메트릭 KDA는 학생들의 인간 설문 응답을 기반으로 측정하며, 이를 통해 기계 학습 모델이 학생의 문제 해결 행위를 모방하도록 합니다. KDA_disc와 KDA_cont 같은 자동 평가 메트릭은 실제 교실 환경에서 전문가가 라벨링한 usability와 강한 상관관계를 보였습니다. 대조 학습과 counterfactual augmentation을 통해 model bias로부터의 민감도를 줄이고 다양한 차원에서 성능을 향상시켰습니다. ProtoDep은 prototype learning과 대형 언어 모델(LLMs)의 생성 능력을 활용하여, 트윗 및 사용자 수준의 증상 설명, 비슷한 개인과의 사례 비교, 그리고 투명한 의사 결정을 제공하는 다층적 설명을 제공합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합할 때 여러 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 보입니다. 제안된 대조 학습 접근 방식은 counterfactual robustness, cross-domain generalization, 그리고 부족한 데이터로부터의 일반화에서 유의미한 성능 향상을 달성했습니다. ProtoDep는 다섯 가지 기준 데이터셋에서 최첨단 성능을 가까스로 유지하면서도 의미 있는 prototype을 학습하여, 우울증 감지에서의 신뢰성과 투명성을 크게 높일 수 있는 잠재력을 보였습니다.



### Mapping Patient Trajectories: Understanding and Visualizing Sepsis Prognostic Pathways from Patients Clinical Narratives (https://arxiv.org/abs/2407.21039)
Comments:
          preprint, 8 pages, 6 figures

- **What's New**: 새로운 영향력 평가 메트릭 KDA(지식 종속 가능성)을 제안하여 자동 생성된 MCQ의 답변 가능성과 교육적 가치를 측정합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 메트릭이 아닌, KDA는 학생 응답으로부터 측정되며 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 통해 학생의 문제 해결 행동을 흉내내도록 합니다. 이를 통해 KDA와 교실 환경에서의 사용성과 강한 상관관계를 보여줍니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측 능력을 보입니다.



### Advancing Chart Question Answering with Robust Chart Component Recognition (https://arxiv.org/abs/2407.21038)
- **What's New**: 이 논문에서는 Multiple Choice Questions(MCQ) 생성의 자동 평가 메트릭인 BLEU, ROUGE, METEOR가 교육적 가치를 반영하지 못한다는 문제를 해결하기 위해 Knowledge Dependent Answerability(KDA)라는 새로운 메트릭을 제안합니다. 이는 학생들이 특정 사실에 대한 지식을 평가하는 능력을 기준으로 MCQ의 답변 가능성을 측정합니다.

- **Technical Details**: 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하여 KDA를 대체하는 방법을 소개합니다. 이는 사전 학습된 언어 모델을 통해 학생들의 문제 해결 경향을 모방합니다. 또한, 다양하고 복잡한 차트 해석을 개선하기 위해 Chartformer라는 프레임워크와 Question-guided Deformable Co-Attention(QDChart) 메커니즘을 소개합니다.

- **Performance Highlights**: KDA 기반 메트릭은 n-gram 기반 메트릭과 결합하여 MCQ 품질을 예측하는 데 강력한 예측력을 보였습니다. Chartformer는 mAP에서 3.2%, ChartQA에서 15.4% 향상을 보여주었으며, 기존 모델보다 뛰어난 성능을 입증했습니다.



### An Application of Large Language Models to Coding Negotiation Transcripts (https://arxiv.org/abs/2407.21037)
- **What's New**: 이 논문에서는 기존 MCQ 평가 메트릭의 한계를 극복하기 위해 새로운 평가 방법론, 지식 종속 가능성 (KDA, Knowledge Dependent Answerability)를 제안했습니다. 또한 최근 NLP 태스크에서 발생하는 spurious pattern 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 결합한 방법론을 제시합니다. 마지막으로 Vanderbilt AI Negotiation Lab에서 LLMs를 활용한 협상 대화 분석 연구를 수행했습니다.

- **Technical Details**: MCQ 생성 평가를 위해 지식 종속 가능성을 측정하는 KDA 메트릭을 도입했으며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 방식을 모방하여 KDA를 근사합니다. Robustness 문제를 해결하기 위해 대조 학습과 반사실적 (counterfactual) 데이터 증강을 결합한 방법론을 사용했습니다. 협상 대화 분석 연구에서는 initial zero-shot, fine-tuning, in-context learning을 포함한 세 가지 LLM 모델 코딩 전략을 테스트했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 때 예측력이 높아집니다. 여러 개의 counterfactual을 활용한 방법론은 다양한 차원에서 큰 개선을 이뤘으며, 특히 counterfactual robustness, cross-domain generalization, 및 scarce data로부터의 일반화에서 탁월한 성과를 보였습니다. 협상 대화 분석 연구에서는 fine-tuned BERT 모델인 BERT-NegCodingJäckel이 논문 제출 당시 최고 성능을 달성했습니다.



### Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress? (https://arxiv.org/abs/2407.21792)
- **What's New**: [{'Paper Title': 'Automatic Evaluation Metrics for Multiple Choice Question Generation', 'Summary': '이 논문은 MCQ(Multiple Choice Question) 생성의 평가 메트릭으로서 지식 종속 가능성(KDA)을 제안합니다. BLEU, ROUGE, METEOR 등의 기존 평가 메트릭이 단순한 n-gram 유사성에만 의존하여 학습적인 가치를 반영하지 않는 문제를 해결하고자, KDA는 대상 사실에 대한 학생의 지식 홍보 여부를 측정합니다.'}, {'Paper Title': 'Counterfactual-based Robustness in NLP through Collective Decision-Making', 'Summary': 'NLP 과업에서 spurious pattern에 의존하는 문제를 해결하기 위해 counterfactual augmentation을 활용한 contrastive learning 방법을 제안합니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 보다 신뢰성 있게 파악합니다.'}, {'Paper Title': 'Comprehensive Meta-Analysis of AI Safety Benchmarks', 'Summary': "AI 안전 연구의 불명확성과 기존 벤치마크의 모호한 목표를 해결하기 위해 포괄적인 메타 분석을 수행합니다. AI 안전 벤치마크와 일반적 능력 (예: 일반 지식 및 추론) 사이의 높은 상관관계를 밝혀내어 'safetywashing'의 가능성을 제기합니다."}]

- **Technical Details**: [{'Paper Title': 'Automatic Evaluation Metrics for Multiple Choice Question Generation', 'Details': 'KDA는 학생들의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 사용하여 학습 모델이 학생들의 문제 해결 행동을 모방하여 대답 가능성을 평가합니다.'}, {'Paper Title': 'Counterfactual-based Robustness in NLP through Collective Decision-Making', 'Details': '기존의 counterfactual augmentation과 달리, 이 방법은 여러 개의 counterfactual을 생성하고 collective decisions을 통해 모델의 편향에 덜 민감한 robust한 인과관계 판단을 수행합니다.'}, {'Paper Title': 'Comprehensive Meta-Analysis of AI Safety Benchmarks', 'Details': '다양한 모델을 대상으로 AI 안전 벤치마크와 일반적 능력 간의 상관관계를 실증적으로 분석하였습니다. 이를 통해 AI 안전 연구의 목표를 명확히 구분하고 의미 있는 안전 메트릭 개발의 기초를 제시합니다.'}]

- **Performance Highlights**: [{'Paper Title': 'Automatic Evaluation Metrics for Multiple Choice Question Generation', 'Highlights': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세팅에서의 사용성과 강한 상관관계를 보였으며, n-gram 유사성 메트릭과 결합하여 다양한 MCQ 품질 척도를 예측하는 데 강력한 예측력을 가집니다.'}, {'Paper Title': 'Counterfactual-based Robustness in NLP through Collective Decision-Making', 'Highlights': '이 접근법은 counterfactual robustness, cross-domain generalization, 그리고 부족한 데이터에서의 일반화 측면에서 현저한 성능 향상을 달성하였습니다.'}, {'Paper Title': 'Comprehensive Meta-Analysis of AI Safety Benchmarks', 'Highlights': "많은 AI 안전 벤치마크가 일반 모델 역량과 높은 상관관계를 보이며, 'safetywashing' 문제를 드러냈습니다. 이는 AI 안전 연구의 방향을 명확히 하는 데 기여할 것으로 기대됩니다."}]



### Vision-Language Model Based Handwriting Verification (https://arxiv.org/abs/2407.21788)
Comments:
          4 Pages, 1 Figure, 1 Table, Accepted as Short paper at Irish Machine Vision and Image Processing (IMVIP) Conference

- **What's New**: 이번 아카이브 논문에서는 광범위하게 사용되고 있는 MCQ(Multiple Choice Questions) 생성 분야의 평가 메트릭에 대한 혁신적인 접근법을 제안합니다. 또한 새로운 학습 데이터 보강 및 대조 학습 방법론과 필적 검증을 위한 시각 언어 모델(Vision Language Models, VLMs)을 다룹니다.

- **Technical Details**: [{'MCQ Generation': '기존 평가는 BLEU, ROUGE, METEOR 등 단순한 n-그램 유사성 분석에 의존했으나 이는 교육적 가치를 반영하지 못합니다. 이에 따라, 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭을 제안하여 MCQ의 대답 가능성(answerability)에 기반한 학생 지식 평가를 강화하였습니다. KDA_disc와 KDA_cont라는 자동 평가 메트릭도 소개하였습니다.', 'Counterfactual Augmentation': '최근 NLP 태스크에서의 deep model들이 spurious pattern에 의존하는 경향이 있어, 대조 학습(contrastive learning) 및 Counterfactual Augmentation을 통한 와생성법(synthesis)이 중요합니다. 여러 counterfactual을 생성하고 집합적 의사 결정(collective decision)을 하여 단어 인과 관계(각 term들의 causality)를 강화하는 방식을 제안합니다.'}, {'Handwriting Verification': '문서 검증에서 필적 검증은 매우 중요합니다. 이 연구에서는 시각 언어 모델 (Vision Language Models, VLMs)을 활용하여 GPT-4o와 PaliGemma를 사용해 필적 검증에 대한 인간 이해 가능한 설명을 제공하는 방법을 탐구합니다. CEDAR 필적 데이터셋을 사용한 실험에서는 VLM이 더 높은 해석 가능성을 제공함을 보여주었으나, ResNet-18 기반 CNN 모델이 여전히 최고 성능을 기록하였습니다 (정확도: 84%).'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont가 KDA와 강한 상관관계를 보였고, n-그램 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 평가한 MCQ 품질 측정에 대해 높은 예측력을 보여줌', 'Counterfactual Augmentation': '다양한 차원에서 현저한 개선; 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 일반화 능력 향상', 'Handwriting Verification': '필적 검증의 경우 GPT-4o와 몇몇 다른 VLM이 인간해석 가능성을 높였으나, ResNet-18 모델이 CEDAR AND 데이터셋에서 최고 정확도 (정확도: 84%)를 기록함'}]



### The Llama 3 Herd of Models (https://arxiv.org/abs/2407.21783)
- **What's New**: 이번 주요 발표는 새로운 자동 평가 메트릭제 사용하여 학생들이 해당 사실을 이해하는 능력을 평가할 수 있는 지식 종속 대답 가능성(KDA)의 도입입니다. 또한, Llama 3 모델의 공개와 그 성능에 대한 광범위한 실험 결과도 포함되어 있습니다.

- **Technical Details**: MCQ 생성 평가를 위한 새로운 지표인 KDA(Knowledge Dependent Answerability)는 학생들이 해당 사실을 이해하는 능력을 평가하는 데 중점을 둡니다. 특히, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행위를 모방하고, 이를 통해 KDA를 근사합니다. 또한, deep learning 모델의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 counterfactual 증대(augmentation)를 통합하여 인과관계 분석의 안정성을 높였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 교실 사용성 모두에 강한 상관관계를 나타냈으며, 전문가가 라벨을 부여한 MCQ 품질 지표에 대한 강력한 예측력을 보였습니다. Llama 3 모델은 GPT-4와 비슷한 성능을 보여주고 있어 다양한 작업에서 경쟁력을 입증했으며, 특히 이미지, 비디오, 음성 인식 작업에서도 최신 기술과 견줄 만한 성과를 보였습니다.



### Can LLMs "Reason" in Music? An Evaluation of LLMs' Capability of Music Understanding and Generation (https://arxiv.org/abs/2407.21531)
Comments:
          Accepted by ISMIR2024

- **What's New**: MCQ 자동 생성 및 평가 메트릭 개선을 향한 새로운 접근법을 소개합니다. 기존에는 BLEU, ROUGE, METEOR 등과 같은 메트릭이 사용되었으나, 이들 메트릭은 생성된 MCQ의 교육적 가치를 평가하지 못했습니다. 이를 해결하기 위해, 우리는 지식 종속 가능성(KDA) 메트릭을 제안하여 MCQ의 대답 가능성 및 학생 지식 평가 능력을 측정합니다. 이와 함께, NLP 태스크에서 건전성 (robustness) 문제를 다루기 위해 대비 학습 (contrastive learning) 및 반사실 증강 (counterfactual augmentation)을 사용하는 새로운 방식을 제시합니다. 또한 Symbolic Music 처리에 대한 GPT-4 및 Llama2 등의 모델 성능을 평가하여, 다단계 추론에서 LLM (Large Language Models)의 한계와 가능성을 탐구합니다.

- **Technical Details**: MCQ 평가에서 제안한 KDA 메트릭은 학생 응답 기반으로 대답 가능성을 측정하며, 이를 통해 KDA_disc, KDA_cont 두 가지 자동 평가 메트릭을 개발했습니다. NLP robustness를 위해서는, 여러 개의 반사실 (counterfactual) 생성을 통한 집합적 의사 결정 방법을 제안했습니다. Symbolic Music 연구에서는, GPT-4, Gemma-7B-it, Llama2-7B-chat, Qwen-7B-chat 모델들의 음악 이해 및 생성 능력을 평가했습니다. 특히, 음악 이론, 모티브 추출, 폼 추출 등의 과제를 통해 모델의 성능을 분석했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 두 메트릭이 실제 강의실 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 MCQ 품질 기준과도 높은 예측 능력을 가졌습니다. 반사실 증강 기법을 통해 다양한 영역에서 성능 향상을 이뤘으며, cross-domain generalization 및 제한된 데이터 환경에서도 우수한 결과를 나타냈습니다. Symbolic Music 작업에서 GPT-4와 Llama2 등의 모델은 다단계 음악 추론에서는 미흡함을 보였으며, 특히 문자열 조건에 맞춘 생성 작업에서 한계를 드러냈습니다.



### Interpreting and learning voice commands with a Large Language Model for a robot system (https://arxiv.org/abs/2407.21512)
Comments:
          PP-RAI 2024, 5th Polish Conference on Artificial Intelligence, 18-20.04.2024 Warsaw, Poland

- **What's New**: 본 연구는 대형 언어 모델(LLMs)인 GPT-4와 데이터베이스를 결합하여 로봇의 의사 결정 능력과 적응성을 향상시키는 새로운 시스템을 제안했다. 이를 통해 요청 해석 문제를 해결하고, 실시간 상호작용 및 학습을 가능하게 한다.

- **Technical Details**: 본 시스템은 ROS 기반 아키텍처와 OpenAI의 GPT-4 API를 사용하여 음성을 텍스트로 변환하고 텍스트를 음성으로 변환한다. LangProc 모듈은 자연어 처리를 담당하며, TaskER 모듈은 로봇의 태스크 관리를 담당한다. TaskDatabase는 PostgreSQL 데이터베이스를 통해 의도(intent)와 슬롯(slot)을 저장하고 매핑한다.

- **Performance Highlights**: 이 시스템은 노인 요양 시설에서 물품 전달 시나리오에서 효과적으로 작동했다. 또한 GPT-4를 통해 예기치 않은 질문에도 적절히 대처하며 '설탕', '레몬'과 같은 속성을 기억하는 능력을 보여주었다. 다만, 5개 이상의 파라미터를 다룰 때는 성능이 떨어질 수 있어 향후 개선이 필요하다.



### Navigating Beyond Instructions: Vision-and-Language Navigation in Obstructed Environments (https://arxiv.org/abs/2407.21452)
Comments:
          Accepted to MM 2024

- **What's New**: 새로운 연구는 기존의 Vision-and-Language Navigation (VLN) 태스크에서 발생하는 실제 환경과의 불일치 문제를 해결하기 위해 R2R-UNO 데이터셋을 도입했습니다. 이 데이터셋은 다양한 예기치 못한 방해물(obstructions)을 포함하여 실내외 에이전트의 성능을 테스트합니다.

- **Technical Details**: R2R-UNO 데이터셋은 기존 R2R 데이터셋을 기반으로 하며, 경로에 있는 다양한 종류의 방해물을 통합하여 지침과 실제 환경 사이의 불일치를 생성합니다. 이를 위해 텍스트-이미지 인페인팅(inpainting) 기술을 사용하여 다양한 방해물을 시각적으로 삽입하고, 필터링 모듈을 통해 높은 품질의 이미지만 선택합니다. 또한, 'ObVLN'이라는 새로운 방법을 제안하여 커리큘럼 학습 전략과 가상 그래프 생성 메커니즘을 사용해 에이전트가 방해물 환경에 적응할 수 있도록 합니다.

- **Performance Highlights**: 초기 실험 결과, 기존 최첨단 VLN 방법들은 방해물이 있는 환경에서 성능이 크게 저하되는 것으로 나타났습니다. 그러나 ObVLN을 사용하면 원래 환경에서도 강력한 성능을 유지하면서도 불일치가 있는 시나리오에서 놀라운 67%의 성공률(Success Rate)을 기록했습니다. 이는 기존 방법들에 비해 30% 이상의 성능 향상을 보여줍니다.



### MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training (https://arxiv.org/abs/2407.21439)
- **What's New**: 자동 MCQ(객관식 질문) 생성 및 평가에 있어, 학습자의 지식을 평가하는 능력을 고려한 새로운 평가 메트릭(Knowledge Dependent Answerability, KDA)을 제안합니다. 또한, 다양한 노이즈(잡음)를 처리하여 고도의 정확성과 연관성을 유지하는 멀티모달(Multimodal) 학습 프레임워크, RagLLaVA를 발표했습니다.

- **Technical Details**: 기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 단어의 유사성에 기반하여 결과를 평가했으나, 교육적 가치를 반영하지 못했습니다. 새로운 메트릭인 KDA는 학생 지식의 가치를 평가하며, 인간 설문조사를 기반으로 한 KDA 측정 방법과 미리 학습된 언어 모델을 활용한 자동 측정 방법(KDA_disc, KDA_cont)을 제안합니다. 또한, RagLLaVA는 멀티모달 데이터 수집 시 발생하는 코스 그레인드(coarse-grained) 및 파인 그레인드(fine-grained) 잡음을 줄이기 위해 노이즈 주입 학습과 지식 향상 재랭킹(knowledge-enhanced reranking)을 사용합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실험을 통해 실제 강의실에서의 사용성 및 전문가 평가 기준과 강한 상관관계를 보여주었습니다. RagLLaVA는 다양한 데이터셋에서 정확하고 강력한 결과물을 생성하며, 멀티모달 질문 응답 작업에서 탁월한 성능을 입증했습니다.



### Towards interfacing large language models with ASR systems using confidence measures and prompting (https://arxiv.org/abs/2407.21414)
Comments:
          5 pages, 3 figures, 5 tables. Accepted to Interspeech 2024

- **What's New**: 자동화된 선택형 질문(Multiple Choice Questions, MCQ) 생성을 더욱 효율적이고 교육적으로 가치 있게 평가하기 위한 새로운 메트릭이 제안되었습니다. 또한, 최근 자연어 처리(NLP) 태스크에서 모델의 루버스트성(Robustness)을 향상시키기 위해 대조적 학습과 반사실적 증강을 결합한 새로운 접근법이 소개되었습니다. 마지막으로, 대형 언어 모델(LLMs)을 활용한 자동 음성 인식(ASR) 트랜스크립트의 사후 교정을 탐구하는 연구가 발표되었습니다.

- **Technical Details**: [{'paper': 'Automatic Generation of Multiple Choice Questions', 'summary': '기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭은 MCQ 생성에서 단순한 n-gram 기반의 유사성만을 평가하여 교육적 가치를 고려하지 못한다는 문제를 해결하기 위해, 대상 사실(Target Fact)에 대한 학생의 지식을 평가할 수 있는 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 제안. KDA는 학생 응답 기반의 인간 설문조사를 통해 측정되고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭으로 근사화.'}, {'paper': 'Robustness in NLP through Contrastive Learning and Counterfactual Augmentation', 'summary': '최근 NLP 태스크에서 인간 이상의 정확성을 보이는 심층 모델(deep models)의 루버스트성을 향상시키기 위해 대조적 학습(Contrastive Learning)과 반사실적 증강(Counterfactual Augmentation)을 결합한 새로운 접근법을 제안. 기존 방법들과 달리, 여러 개의 반사실적 데이터를 생성하고 이를 집합적 의사결정(Collective Decision)을 통해 분석하여 더 강력한 인과관계 파악. 결과적으로 반사실적 루버스트성, 교차 도메인 일반화, 희소 데이터에서의 일반화 등 다각도에서 의미 있는 개선을 달성.'}, {'paper': 'Post-hoc Correction of ASR Transcripts with Large Language Models', 'summary': '대형 언어 모델(LLMs)를 활용하여 자동 음성 인식(ASR) 트랜스크립트의 오류를 교정하는 새로운 방법을 제안. 정확도가 높은 트랜스크립트에 오류를 도입하지 않기 위해 자신감 기반 필터링 방법을 여러 가지 제안. 구체적으로는 문장 또는 단어의 자신감 점수가 특정 기준 이하인 경우에만 LLM이 교정을 수행하도록 설계함.'}]

- **Performance Highlights**: [{'paper': 'Automatic Generation of Multiple Choice Questions', 'summary': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 평가한 MCQ 품질 측정에서 강한 예측 능력을 보임.'}, {'paper': 'Robustness in NLP through Contrastive Learning and Counterfactual Augmentation', 'summary': '제안된 방법은 반사실적 루버스트성, 교차 도메인 일반화 및 희소 데이터에서의 일반화 등 다양한 측면에서 기존 방법에 비해 의미 있는 성능 향상을 달성하였음.'}, {'paper': 'Post-hoc Correction of ASR Transcripts with Large Language Models', 'summary': '자신감 점수 기반 필터링 방법을 사용하여 정확도가 낮은 문장과 단어를 교정한 결과, ASR 시스템의 성능을 향상시킬 수 있었음. 그러나 LLM이 오류를 도입할 가능성을 최소화하기 위해 필터링 방법의 중요성이 강조됨.'}]



### Prompting Medical Large Vision-Language Models to Diagnose Pathologies by Visual Question Answering (https://arxiv.org/abs/2407.21368)
- **What's New**: 기존의 MCQ 생성 평가 메트릭이 교육적 가치를 평가하지 못한다는 문제를 해결하기 위해, 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안했습니다. 이 메트릭은 학생이 대상 사실에 대한 지식을 기반으로 MCQ에 답할 수 있는 능력을 측정합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 방식을 통해 프리트레인된 언어 모델이 학생의 문제 해결 행동을 모방합니다. 또한, 우리는 다양한 전문가 라벨 MCQ 품질 측정에서 strong predictive power를 보여줍니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 설정에서 사용성 및 KDA와 높은 상관관계를 가지고 있음을 입증했습니다. 또한, n-gram 기반 유사성 메트릭과 결합될 때 이들 메트릭이 전문가 라벨 품질 측정치에 강한 예측력을 가짐을 보여주었습니다.



### Multi-Level Querying using A Knowledge Pyramid (https://arxiv.org/abs/2407.21276)
- **What's New**: {'MCQ Generation': "기존의 n-gram 기반 메트릭이 MCQ의 교육적 가치를 평가하지 못하는 문제를 해결하기 위해, 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안합니다.", 'Model Robustness': '최근 심층 모델이 높은 정확성을 보이지만 spurious patterns에 의존하여 robustness가 제한된 문제를 해결하고자, 대조 학습 (contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용하는 방법을 제안합니다.', 'RAG Enhancement': '기존 RAG 방법이 주로 리콜(Recall)을 향상시키는 데 초점을 맞추고 있는 반면, 정확도(Precision)와 리콜의 균형을 맞추기 위해 다층 지식 피라미드 접근 방식을 제안합니다.'}

- **Technical Details**: {'MCQ Generation': 'KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하는 메트릭입니다. 특히, 인간 조사를 통해 측정한 이후, 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방한 KDA_disc와 KDA_cont를 제안합니다.', 'Model Robustness': '기존의 사람 또는 기계가 데이터에서 유사한 반사실적 데이터를 추가하는 방법과 다르게, 이 논문에서는 여러 개의 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 각 용어의 인과 관계를 보다 robust하게 평가하는 방법을 제안합니다.', 'RAG Enhancement': '다층 지식 피라미드 접근법은 온톨로지, 지식 그래프(KGs), 청크 기반의 원시 텍스트의 세 가지 계층으로 구성되며, 크로스-레이어 증강 기술을 사용하여 포괄적인 지식 커버리지와 동적 업데이트를 수행합니다. 온톨로지와 사례의 압축성을 유지하기 위해 크로스-레이어 필터링 방법을 사용합니다.'}

- **Performance Highlights**: {'MCQ Generation': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지고 있음을 입증했습니다. n-gram 기반 유사성 메트릭과 결합했을 때, 여러 전문가가 라벨링한 MCQ 품질 측정치에 대해 강한 예측력을 보여줍니다.', 'Model Robustness': '집합적 의사 결정 접근 방식으로 모델의 편향에 민감하지 않고, 반사실적 robustness, 교차 도메인 일반화, 그리고 희소한 데이터로부터의 일반화 등 다양한 차원에서 큰 개선을 이룩했습니다.', 'RAG Enhancement': 'PolyRAG는 19개의 최첨단 방법을 능가하며, GPT-4의 성능을 0.1636에서 0.8109로 395%의 F1 향상을 이끌어냈습니다.'}



### Advancing Vietnamese Visual Question Answering with Transformer and Convolutional Integration (https://arxiv.org/abs/2407.21229)
Comments:
          Accepted at the journal of Computers & Electrical Engineering (Received 8 March 2024, Revised 8 June 2024, Accepted 10 July 2024)

- **What's New**: 최근 논문들에서는 다양한 AI와 컴퓨터 비전 연구와 관련된 새로운 접근법을 제안하고 있습니다. 본 논문들에서는 자동 MCQ(객관식 문제) 생성의 평가 메트릭, NLP 태스크에서의 로버스트니스 향상 방법, 그리고 베트남어 비주얼 질문 응답(ViVQA) 시스템에 대한 혁신적인 접근법을 소개하고 있습니다.

- **Technical Details**: {'MCQ Generation': '기존의 n-gram 기반 평가 메트릭, 예를 들어 BLEU, ROUGE, METEOR는 교육적 가치를 평가하지 못합니다. 이를 개선하기 위해 지식 종속 가능성(KDA)이라는 새로운 메트릭을 제안합니다. 이는 학생들이 특정 사실에 대한 지식을 평가할 수 있는 능력을 측정합니다. KDA_exp와 KDA_cont라는 두 가지 자동화 메트릭이 제안되었습니다.', 'NLP Robustness': "최근의 deep models가 NLP 태스크에서 사람보다 나은 정확성을 보였으나, 이는 스퓨리어스 패턴(spurious pattern)에 의존하는 문제로 인해 로버스트니스가 제한적입니다. 이에 대응하기 위해 대조 학습과 반사실적 증강(counterfactual augmentation)을 이용한 방법을 제안합니다. '여러 개의' 반사실을 생성하고 집합적 의사 결정을 통해 단어들의 인과 관계를 더 robust하게 파악하는 방법을 도입했습니다.", 'ViVQA': 'ViVQA 시스템에서 BLIP-2와 EfficientNet을 통합하여 이미지의 로컬 및 글로벌 특징을 처리합니다. 그 후, 멀티모달 융합 모듈을 통해 시각적 및 텍스트 특징을 결합합니다. 제안된 방법은 베트남어 VQA 데이터셋에서 기존 방법론보다 성능이 뛰어남을 보여줍니다.'}

- **Performance Highlights**: {'MCQ Generation': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 메트릭과 결합했을 때 전문 라벨링된 다양한 MCQ 품질 지표에 대해 강력한 예측 능력을 가집니다.', 'NLP Robustness': '대조 학습과 반사실적 증강을 통해 스퓨리어스 상관성의 영향을 적게 받고, 다양한 차원에서 로버스트니스를 향상시켜 반사실 강인성, 크로스 도메인 일반화, 데이터 부족 상황에서도 우수한 성능을 나타냅니다.', 'ViVQA': 'ViVQA 데이터셋의 테스트 셋에서 $71.04\\%$의 정확도를 달성하며, 기존의 경쟁 베이스라인을 모두 능가합니다. 특히, 이미지 표현의 개선과 멀티뷰 특징 표현을 활용한 성능 향상이 두드러집니다.'}



### GenRec: Generative Personalized Sequential Recommendation (https://arxiv.org/abs/2407.21191)
- **1**: {"What's New": 'Automatic MCQ(객관식 질문) 생성의 평가 메트릭으로 Knowledge Dependent Answerability (KDA) 제안.', 'Technical Details': '기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에 의존하며 교육적 가치를 평가하지 않음. KDA는 목표 사실에 대한 학생의 응답 가능성을 측정함.', 'Performance Highlights': '인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 나타냄.'}

- **2**: {"What's New": '대조 학습(contrastive learning)과 반사실적 데이터 증가(counterfactual augmentation)로 NLP 태스크에서의 모델 강건성(robustness) 향상.', 'Technical Details': '기존 방법은 인간이 직접 반사실적 데이터를 추가하거나 데이터를 통해 자동 생성함. 제안된 방법은 다수의 반사실적 데이터를 합성하고 집합적 의사 결정을 통해 인과관계를 더욱 강건하게 평가함.', 'Performance Highlights': '다양한 차원에서 유의미한 성능 개선 달성: 반사실적 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 한정된 데이터에서의 일반화(generalization from scarce data).'}

- **3**: {"What's New": "Generative Recommendation (GenRec) 모델 제안으로, 'pretrain, prompt and predict' 패러다임을 사용하여 시퀀스 추천을 생성(task)으로 전환.", 'Technical Details': 'Transformer 기반의 시퀀스 modeling 활용, 마스킹된 아이템 예측 목표 사용. 기존의 생성 모델과 달리 수동으로 설계된 hard prompt에 의존하지 않음.', 'Performance Highlights': 'GenRec은 광범위한 공개 실세계 데이터셋에서 state-of-the-art 성능 달성. 제안된 마스킹 아이템 예측 목표는 모델 성능을 크게 향상시킴.'}



### Apple Intelligence Foundation Language Models (https://arxiv.org/abs/2407.21075)
- **What's New**: MCQ 자동 생성 평가를 위한 새로운 메트릭 'Knowledge Dependent Answerability (KDA)'가 제안되었습니다. 이 메트릭은 BLEU, ROUGE, METEOR 등 기존 n-gram 기반의 유사성 평가 메트릭이 간과한 교육적 가치를 평가합니다.

- **Technical Details**: KDA는 대상 사실(target fact)에 대한 MCQ의 대답 가능성을 측정하는 새로운 자동 평가 메트릭입니다. 인간 응답으로 기반한 KDA 측정 방법을 먼저 보여주고, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: Human studies 결과, KDA_disc와 KDA_cont는 실제 강의실 사용성과 강한 상관관계를 보였습니다. 추가로, 이 메트릭들은 n-gram 기반 유사성 메트릭과 결합되었을 때 전문가가 라벨링한 MCQ 품질 지표를 예측하는 데 강력한 효력을 나타냈습니다.



### Enhancing Adversarial Text Attacks on BERT Models with Projected Gradient Descen (https://arxiv.org/abs/2407.21073)
Comments:
          This paper is the pre-reviewed version of our paper that has been accepted for oral presentation and publication in the 4th IEEE ASIANCON. The conference will be organized in Pune, INDIA from August 23 to 25, 2024. The paper consists of 8 pages and it contains 10 tables. It is NOT the final camera-ready version that will be in IEEE Xplore

- **What's New**: MCQ 자동 생성의 평가 메트릭으로 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)을 제안하여, MCQ가 학생의 지식을 평가하는 능력을 측정합니다. 또한, NLP 모델의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 사용하는 새로운 방법을 소개합니다. 마지막으로, BERT-Attack의 한계를 개선한 PGD-BERT-Attack을 제안합니다.

- **Technical Details**: 첫 번째 논문은 KDA를 기반으로 한 평가 방식으로, 인간 설문 조사 데이터를 사용해 학생들의 문제 해결 능력을 모방하여 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안합니다. 두 번째 논문은 여러 개의 counterfactual을 생성하고, collective decision을 통해 더 견고한 인과관계 분석을 수행합니다. 마지막으로, PGD-BERT-Attack은 기존 BERT-Attack의 한계를 극복하기 위해 Projected Gradient Descent (PGD)을 사용해, 보다 효과적이고 robustness를 갖춘 adversarial 예제를 생성합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 전문가가 평가한 실제 강의실 세트의 usability와 강한 상관관계를 보였습니다. 새로운 NLP 모델은 1) counterfactual robustness, 2) cross-domain generalization, 3) generalization from scarce data 측면에서 개선을 확인했습니다. 또한, PGD-BERT-Attack은 misclassification 유발률을 높이면서도 원본 입력과의 의미적 유사성을 유지하여, 실제 적용 가능성을 높였습니다.



### Beyond Metrics: A Critical Analysis of the Variability in Large Language Model Evaluation Frameworks (https://arxiv.org/abs/2407.21072)
Comments:
          15 pages, 3 figures

- **What's New**: 자동 다중 선택 질문(MCQ) 생성의 교육적 가치를 평가하기 위해 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안했습니다. KDA는 생성된 MCQ가 학생의 목표 지식에 비추어 얼마나 답변 가능한지를 측정합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정됩니다. 이를 자동화하기 위해 사전 훈련된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 이 메트릭들은 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 전문가가 라벨링한 다양한 MCQ 품질 측정 기준에 대해 강력한 예측 능력을 보임을 확인했습니다.



### Towards Automated Data Sciences with Natural Language and SageCopilot: Practices and Lessons Learned (https://arxiv.org/abs/2407.21040)
- **Multiple Choice Question (MCQ) Generation**: [{"What's New": '기존의 BLEU, ROUGE, METEOR 평가 메트릭이 교육적 가치를 고려하지 않고 단순히 n-gram 유사성만을 측정하는 문제를 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭을 제안했습니다. KDA는 MCQ의 대답 가능성을 측정하며 학생의 지식을 평가하는 능력을 평가합니다.'}, {'Technical Details': 'KDA의 측정을 위해 학생 응답을 기반으로 한 human survey를 이용하며, 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하여 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont는 실제 교실 세트에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가가 표시한 MCQ 품질 측정을 예측하는 능력이 뛰어남이 확인되었습니다.'}]

- **Robustness in NLP Models**: [{"What's New": '최근 딥 모델들이 NLP 작업에서 높은 정확성을 보였으나, spurious pattern에 의존하여 robustness가 제한된다는 문제를 해결하기 위해 대비 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다.'}, {'Technical Details': "반사실 증강을 위해 '여러 개의' counterfactual을 생성하고 이들을 기반으로 집합적 의사 결정을 내림으로써, 각 단어의 인과관계를 강건하게 감독하는 방법을 제안합니다."}, {'Performance Highlights': '실험 결과, 집합적 의사 결정을 통해 태스크 모델 편향에 덜 민감해지며, 1) 반사실 robust성, 2) 도메인 간 일반화, 3) 부족한 데이터에서의 일반화에서 큰 개선을 이루었습니다.'}]

- **Automating Data Science Pipeline**: [{"What's New": 'NL2SQL 분야가 자연어 명령을 실행 가능한 SQL 스크립트로 번역하는 데 있어 많은 발전을 이루었으나, 데이터 쿼리, 분석, 시각화 및 보고를 포함하는 더 넓은 데이터 과학 파이프라인의 완전한 자동화는 여전히 복잡한 문제입니다. 이 연구에서는 LLMs, AutoAgents, LUIs를 통합하여 데이터 과학 파이프라인을 자동화하는 고급 시스템 SageCopilot을 소개합니다.'}, {'Technical Details': 'SageCopilot은 두 단계 디자인을 채택하여 온라인 컴포넌트에서는 사용자의 입력을 실행 가능한 스크립트로 변환하고 결과를 보고 및 시각화하며, 오프라인 준비 단계에서는 ICL에서 요청한 데모를 준비합니다. Chain-of-Thought 및 prompt-tuning과 같은 최첨단 전략도 SageCopilot의 성능을 향상시키기 위해 사용되었습니다.'}, {'Performance Highlights': '엄격한 테스트와 비교 분석을 통해 SageCopilot은 스크립트 생성 및 실행, 시각화 결과 제공에서 높은 성능을 입증했습니다. 세부적인 소성 연구는 SageCopilot의 다양한 구성 요소와 전략이 데이터 과학의 끝에서 끝까지 정확성에 기여하는 바를 강조했습니다.'}]



### Multi-Grained Query-Guided Set Prediction Network for Grounded Multimodal Named Entity Recognition (https://arxiv.org/abs/2407.21033)
Comments:
          11 pages, 5 figures

- **Multiple Choice Questions Generation**: [{"What's New": '자동 객관식 문제(MCQ) 생성 평가에 대한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)가 소개되었습니다. KDA는 기존의 BLEU, ROUGE, METEOR 지표들이 간과한 교육적 가치를 고려하며, 학생의 지식 평가 능력을 측정합니다.'}, {'Technical Details': 'KDA 측정은 학생 응답 기반의 설문조사를 통해 이루어지며, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하는 방법으로 KDA를 근사화합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont는 KDA 및 실제 교실 환경에서의 사용성과 강한 상관관계를 보였습니다. n-gram 기반의 유사성 메트릭과 결합할 경우, 다양한 전문가 레이블 MCQ 품질 측정에서도 강력한 예측력을 나타냅니다.'}]

- **Deep Models Robustness**: [{"What's New": '딥러닝 모델의 spurious pattern 의존성을 극복하기 위한 contrastive learning 및 counterfactual augmentation 접근법이 제안되었습니다.'}, {'Technical Details': "기존 방법들은 사람이 counterfactual을 추가하거나 모델이 데이터셋에서 자동으로 근접한 counterfactual을 맞추어야 했으나, 새로운 접근법은 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 단어 인과관계를 더욱 견고하게 감독합니다."}, {'Performance Highlights': '새로운 접근법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data 상황에서의 일반화 성능에서 유의미한 성장을 보였습니다.'}]

- **Grounded Multimodal Named Entity Recognition (GMNER)**: [{"What's New": '기존의 MRC 기반 또는 sequence generation 기반 모델의 제약을 극복하기 위해 Multi-grained Query-guided Set Prediction Network (MQSPN)라는 새로운 통합 프레임워크가 제안되었습니다.'}, {'Technical Details': 'MQSPN은 Multi-grained Query Set (MQS)와 Multimodal Set Prediction Network (MSP), 그리고 Query-guided Fusion Net (QFNet)을 포함합니다. 이것들은 intra-entity와 inter-entity 레벨에서 적절한 관계들을 학습하도록 설계되었습니다.'}, {'Performance Highlights': '광범위한 실험 결과, 제안된 방법이 다양한 벤치마크에서 SOTA 성능을 달성했으며, 특히 세부적인 GMNER 벤치마크에서 2.83%의 F1 점수 향상을 이뤄냈습니다.'}]



### LLM-Find: An Autonomous GIS Agent Framework for Geospatial Data Retrieva (https://arxiv.org/abs/2407.21024)
- **What's New**: 이번 연구에서는 학습 평가 시간을 줄이기 위한 자동 Multiple Choice Questions (MCQ) 생성에 관한 평가 메트릭과 딥러닝 모델의 robustness 향상을 위한 대조 학습(constrative learning) 및 counterfactual augmentation 방법, 그리고 자율 공간 정보 시스템(GIS) 에이전트 프레임워크를 제안합니다.

- **Technical Details**: [{'MCQ Generation': '기존 MCQ 생성의 평가 메트릭이 n-gram 중심의 유사성(BLEU, ROUGE, METEOR)을 고려한다면, 우리는 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)를 도입했습니다. KDA를 인간 설문조사를 통해 측정하고, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다.', 'Robustness in NLP': '최근 딥러닝 모델들이 사람보다 높은 정확도를 보였지만, 여전히 spurious pattern에 의존하는 문제로 robustness가 제한됩니다. 이를 해결하기 위해, 우리는 여러 개의 counterfactual을 생성하고 집단적 의사 결정을 통해 더 robust하게 인과관계를 파악하는 방법을 제안합니다.', 'GIS Agent': '자율 GIS 에이전트 프레임워크 LLM-Find를 제안했습니다. 이는 LLM을 의사 결정자로 활용하여 필요한 공간 데이터를 선택하고 가져오며, 플러그 앤 플레이(plug-and-play) 방식으로 설계되었습니다. 다양한 데이터 소스(예: OpenStreetMap, US Census Bureau, ESRI World Imagery 등)에서 데이터를 검색할 수 있는 프로토타입 에이전트를 개발하여 그 가능성을 실험적으로 검증했습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 상황에서 높은 usability와 강한 상관관계를 가지는 것을 보였습니다. 또한, KDA_disc와 KDA_cont를 n-gram 기반 메트릭과 결합할 경우, 전문가가 라벨링한 다양한 MCQ 품질 측정 기준에 대해 강력한 예측력을 보였습니다.', 'Robustness in NLP': '우리 접근법은 다차원에서 상당한 성능 향상을 이루었는데, 그 예로 counterfactual robustness, cross-domain generalization 및 부족한 데이터에서의 일반화를 들 수 있습니다.', 'GIS Agent': '프레임워크는 다양한 데이터 소스에서 데이터를 성공적으로 검색할 수 있음을 입증했으며, 새로운 데이터 소스를 추가하는 데 유연성과 확장성이 뛰어났습니다.'}]



New uploads on arXiv(cs.IR)

### MOSAIC: Multimodal Multistakeholder-aware Visual Art Recommendation (https://arxiv.org/abs/2407.21758)
- **What's New**: 최근 MCQ 자동 생성 평가에서, 기존의 평가 메트릭이 교육적 가치를 제대로 반영하지 못하고 있어 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안했습니다. 또한, 최근 NLP 태스크에서 사용되는 deep model의 robustness를 강화하기 위해 대조 학습(contrastive learning) 및 반사실 증강(counterfactual augmentation)을 활용하는 방안에 대해 연구했습니다. 마지막으로, 시각 예술(VA) 추천 시스템에서 다중 이해관계자를 고려한 새로운 접근법 MOSAIC을 소개했습니다.

- **Technical Details**: KDA는 학생의 문제 해결 능력을 모사하기 위해 사전 학습된 언어 모델을 활용해 학생 응답 데이터를 기반으로 KDA_disc, KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다. Deep model의 robustness를 위해 기존의 반사실 증강 방식 대신 다수의 반사실을 생성하여 범위 내에서 예측 분포를 조정하는 신규 방법론을 제안했습니다. MOSAIC은 CLIP 및 BLIP 백본 아키텍처, 다양한 최적화 목표를 통해 사용자 중심의 새로운 콘텐츠 추천을 달성하려 합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교육 환경에서 사람 평가와 강하게 상관관계를 나타내며, n-gram 기반의 유사성 메트릭과 결합 시 높은 예측력을 보였습니다. 대조 학습 및 반사실 증강을 통해 다양한 기준에서 강력한 성능 향상을 달성했으며, MOSAIC은 사용자 테스트에서 인기도와 대표성 측면에서 긍정적인 평가를 받았습니다.



### Learning Effective Representations for Retrieval Using Self-Distillation with Adaptive Relevance Margins (https://arxiv.org/abs/2407.21515)
Comments:
          9 Pages, 4 Tables, 6 Figures

- **What's New**: 이번 연구에서 자동으로 생성된 다지선다형 질문(MCQ)에 대한 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하였습니다. KDA는 기존의 BLEU, ROUGE, METEOR 메트릭이 간과한 교육적 가치를 평가하기 위해 만들었습니다. 또한, 새로운 bootstrap 방식의 loss function을 통한 self-distillation 방법을 개발하여 정보 검색 모델의 효율성을 크게 향상시켰습니다.

- **Technical Details**: KDA는 target fact에 대한 학생의 지식을 평가할 수 있는 MCQ의 대답 가능성(answerability)을 측정하고, 이를 인간 평가를 통해 검증한 후, 사전 학습된 언어 모델을 이용해 KDA_disc와 KDA_cont의 두 가지 자동 평가 메트릭을 제안합니다. 또한, 기존의 지식 증류(knowledge distillation) 방식을 사용하지 않고, 교육 없이도 가능한 새로운 parameter-free loss function을 제안하여 bi-encoder를 효율적으로 훈련할 수 있게 합니다. 이 방법은 implicit hard negative mining을 통해 배치 샘플링(batch sampling) 절차를 제거합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가 기준과 강한 상관관계를 보였고, 기존 n-gram 기반 메트릭과 함께 사용되었을 때 전문가가 레이블한 다양한 MCQ 품질 측정 기준을 예측하는 데 큰 효율성을 보여주었습니다. 새로운 self-distillation 방식은 교육 데이터의 13.5%만을 사용하면서도 교사 모델 기반 증류와 동등한 성능을 보여주었고, 학습 시간에서는 최대 15배의 속도 향상이 이루어졌습니다.



### Breaking the Hourglass Phenomenon of Residual Quantization: Enhancing the Upper Bound of Generative Retrieva (https://arxiv.org/abs/2407.21488)
- **What's New**: 최근 MCQ 생성 평가 메트릭의 한계를 극복하기 위해 제안된 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'에 대해 기술합니다. 또한, 문장 생성을 더 견고하게 하기 위해 대조 학습과 반사실적 증대를 사용하는 접근 방식과 e-commerce 검색 및 추천 시스템에서 generative retrieval 방법의 제약을 해결하고자 하는 새로운 방법들을 소개합니다.

- **Technical Details**: 첫 번째 논문에서는, BLEU, ROUGE, METEOR와 같은 기존 메트릭의 한계를 지적하고 'Knowledge Dependent Answerability (KDA)'라는 새로운 평가 방법을 제안합니다. KDA는 학생들이 해당 MCQ에 대해 답변을 제공할 때의 '지식 기반 대답 가능성'을 측정합니다. 두 번째 논문에서는, 대조 학습(contrastive learning)과 반사실적 증대(counterfactual augmentation) 기법을 사용하여 NLP 모델의 robustness를 향상시키는 방법을 제시합니다. 세 번째 논문에서는, Residual Quantization 기반의 Semantic Identifiers (RQ-SID)가 겪는 'Hourglass' 문제를 해결하기 위한 접근 방안을 논의하고 있습니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 실제 강의실 세트에서의 활용도와 강한 상관관계를 나타내는 것으로 확인되었습니다. 대조적 학습과 반사실적 증대를 사용한 방법은 다양한 측면에서 significant한 성능 향상을 보여주었으며, e-commerce 시나리오에서 RQ-SID의 'Hourglass' 문제를 해결한 방법은 codebook 이용률과 데이터 분포의 최적화를 통해 추천 시스템 성능을 크게 향상시켰습니다.



### ABCDE: Application-Based Cluster Diff Evals (https://arxiv.org/abs/2407.21430)
- **MCQ Generation**: [{"What's New": '자동 MCQ 생성의 평가를 위해 기존의 BLEU, ROUGE, METEOR 같은 n-gram 기반의 메트릭이 아닌, 교육적 가치를 평가하는 새로운 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 메트릭 제안.', 'Technical Details': 'KDA는 학생들이 해당 사실을 이해하는지 평가하기 위해 MCQ의 대답 가능성을 측정한다. KDA_disc와 KDA_cont 자동 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 대략적인 KDA를 산출한다.', 'Performance Highlights': '인간 평가를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성 및 KDA와 강한 상관관계를 가진다는 것이 증명되었다.'}]

- **NLP Model Robustness**: [{"What's New": '최근 NLP 태스크에서의 딥러닝 모델의 높은 정확성에도 불구하고 spurious pattern에 의존하여 robustness가 제한되는 문제를 해결하기 위해 대비 학습 (Contrastive Learning)과 counterfactual augmentation라는 새로운 강화 방법 제안.', 'Technical Details': '기존 counterfactual augmentation 기법과 달리, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 강력하게 단어들의 인과관계를 감독한다.', 'Performance Highlights': '이 접근 방식은 다양한 차원(대비 (Counterfactual) 견고성, 도메인 간 일반화, 드문 데이터로부터의 일반화)에서 상당한 성능 향상을 보였다.'}]

- **Cluster Evaluation**: [{"What's New": '매우 큰 인구의 아이템을 클러스터링하는 문제를 평가하기 위한 새로운 평가 기법 ABCDE 제안. Baseline 클러스터링과 Experiment 클러스터링 간의 차이점을 특징짓고, 어떤 클러스터링이 더 나은지 결정한다.', 'Technical Details': 'ABCDE는 아이템의 애플리케이션별 중대한 값을 반영할 수 있으며, 인간 판단을 최소화하여 비용을 절감하고, 아이템 데이터의 임의 슬라이스에 대한 메트릭을 보고할 수 있다. 클러스터링 품질의 차이를 측정하기 위해 미리 고비용의 그라운드 트루쓰 (ground truth)를 구축하는 대신, 실제 클러스터링 간의 차이에 기반하여 판단용 질문을 샘플링한다.', 'Performance Highlights': 'ABCDE는 포인트 와이즈 메트릭을 활용하여 직관적이고 이해하기 쉬운 품질 평가를 제공하며, 클러스터링 알고리즘의 개선에도 기여할 수 있다.'}]



### Personalized Multi-task Training for Recommender System (https://arxiv.org/abs/2407.21364)
Comments:
          11 pages

- **What's New**: 이 논문들은 각각 새로운 교육 평가 메트릭, 새로운 강건화 방법, 그리고 개인화된 멀티태스크 학습 알고리즘을 제안합니다. 첫 번째 논문은 Multiple Choice Questions(MCQ)을 평가하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)를 소개합니다. 두 번째 논문은 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 통해 NLP 모델의 강건성을 높이는 방법을 다룹니다. 세 번째 논문은 여러 정보 소스로부터 사용자의 선호도를 더 잘 파악하기 위해 처음으로 개인화된 멀티태스크 학습 알고리즘을 제안합니다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭이 교육적 가치를 충분히 반영하지 못한다는 문제를 해결하기 위해 KDA를 제안하고, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 개발하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 두 번째 논문에서는 머신이 데이터셋에서 반사실적 데이터를 자동으로 매칭하는 대신, '여러 개의' 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 감독하는 방법을 제안합니다. 세 번째 논문은 사용자/항목 임베딩을 다양한 정보 소스로부터 얻기 위해 PMTRec를 제안하며, 개인화된 태스크 가중치와 다양한 태스크 방향성을 다루고, 태스크 간 그래디언트 크기 차이를 조절하는 모듈들을 개발합니다.

- **Performance Highlights**: 첫 번째 논문의 Human evaluation 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 나타냈으며, 전문가가 라벨링한 MCQ 품질 지표에 대해 예측력이 높은 것으로 나타났습니다. 두 번째 논문은 대조 학습 기반 접근법이 반사실적 강건성, 도메인 간 일반화, 희소 데이터 일반화 측면에서 의미 있는 개선을 가져왔습니다. 세 번째 논문은 세 가지 실제 데이터셋에서 기존의 멀티태스크 학습 방법들을 능가하는 성능을 보였으며, 다양한 태스크를 동시에 활용해 개인화된 추천 정확도를 크게 향상시켰음을 입증했습니다.



### Implementing Streaming algorithm and k-means clusters to RAG (https://arxiv.org/abs/2407.21300)
- **What's New**: 연구진은 자동 다중 선택 질문(MCQ) 생성에서 BLEU, ROUGE, METEOR 등 기존 메트릭이 교육적 가치를 고려하지 못하는 문제를 해결하기 위해 'Knowledge Dependent Answerability (KDA)'라는 새로운 자동 평가 메트릭을 제안하였다. 또한, NLP 태스크에서 최근 딥 모델들이 겪는 강건성 문제를 해결하기 위해 대조 학습과 반사실적 증강(counterfactual augmentation)을 사용하는 방법을 연구하였다. 마지막으로, 정보 검색에 사용되는 RAG 모델에서 메모리 소비 문제를 해결하기 위해 스트리밍 알고리즘과 k-평균 클러스터링을 결합한 새로운 방법을 제안하였다.

- **Technical Details**: KDA는 학생의 반응을 바탕으로 MCQ의 대답 가능성을 측정하고, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다. 대조 학습과 반사실적 증강을 통해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안한다. RAG 모델에서는 스트리밍 알고리즘을 사용해 인덱스를 업데이트하고, k-means 알고리즘을 사용해 유사한 문서를 클러스터링하여 메모리 소비를 줄이고 쿼리 시간을 단축하는 방법을 제안하였다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 사용성과 강한 상관관계를 보였으며, 기존의 n-gram 기반 메트릭과 결합했을 때 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에 대한 예측력이 높았다. 제안된 대조 학습 및 반사실적 증강 방법은 counterfactual robust성, 도메인간 일반화, 적은 데이터로부터의 일반화 등 다양한 측면에서 유의미한 개선을 보여주었다. RAG 모델에서는 정확도와 메모리 사용량 면에서 스트리밍 알고리즘과 k-means 클러스터링을 결합한 새로운 방법이 우수한 성능을 보였다.



### GenRec: Generative Personalized Sequential Recommendation (https://arxiv.org/abs/2407.21191)
- **1**: {"What's New": 'Automatic MCQ(객관식 질문) 생성의 평가 메트릭으로 Knowledge Dependent Answerability (KDA) 제안.', 'Technical Details': '기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에 의존하며 교육적 가치를 평가하지 않음. KDA는 목표 사실에 대한 학생의 응답 가능성을 측정함.', 'Performance Highlights': '인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 나타냄.'}

- **2**: {"What's New": '대조 학습(contrastive learning)과 반사실적 데이터 증가(counterfactual augmentation)로 NLP 태스크에서의 모델 강건성(robustness) 향상.', 'Technical Details': '기존 방법은 인간이 직접 반사실적 데이터를 추가하거나 데이터를 통해 자동 생성함. 제안된 방법은 다수의 반사실적 데이터를 합성하고 집합적 의사 결정을 통해 인과관계를 더욱 강건하게 평가함.', 'Performance Highlights': '다양한 차원에서 유의미한 성능 개선 달성: 반사실적 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 한정된 데이터에서의 일반화(generalization from scarce data).'}

- **3**: {"What's New": "Generative Recommendation (GenRec) 모델 제안으로, 'pretrain, prompt and predict' 패러다임을 사용하여 시퀀스 추천을 생성(task)으로 전환.", 'Technical Details': 'Transformer 기반의 시퀀스 modeling 활용, 마스킹된 아이템 예측 목표 사용. 기존의 생성 모델과 달리 수동으로 설계된 hard prompt에 의존하지 않음.', 'Performance Highlights': 'GenRec은 광범위한 공개 실세계 데이터셋에서 state-of-the-art 성능 달성. 제안된 마스킹 아이템 예측 목표는 모델 성능을 크게 향상시킴.'}



### Watermarking Recommender Systems (https://arxiv.org/abs/2407.21034)
- **What's New**: 이번 주 아카이브 논문에서는 세 가지 최신 연구 결과를 다룹니다. 첫 번째는 자동 다지선다 문제(MCQ) 생성의 교육적 가치를 평가하기 위한 새로운 메트릭, Knowledge Dependent Answerability (KDA)입니다. 두 번째 논문에서는 NLP 모델의 강건성(robustness)을 높이기 위해 대조 학습과 반사실적 데이터 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다. 마지막 논문은 추천 시스템(recommender systems)에서 모델 도난을 방지하기 위한 자동 회귀 분포 외 워터마킹(Autoregressive Out-of-distribution Watermarking, AOW) 기술을 소개합니다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 MCQ의 교육적 가치를 제대로 평가하지 못하는 문제를 지적하고, KDA 메트릭을 제안했습니다. 이 메트릭은 학습자의 답변 가능성을 기준으로 MCQ를 평가합니다. 두 번째 논문에서는 여러 개의 반사실적 데이터(counterfactuals)를 생성하고 집합적 의사 결정(collective decisions)을 통해 모델의 인과 관계를 강화하는 방법을 제안했습니다. 세 번째 논문에서는 추천 시스템의 모델을 보호하기 위한 AOW 방법을 제안하고, 모델의 예측 점수를 기반으로 워터마크 시퀀스를 자동 회귀적으로 생성하는 방법을 설명했습니다.

- **Performance Highlights**: 첫 번째 논문에서는 KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, 기존의 n-gram 기반 메트릭과 결합 시 높은 예측 성능을 나타낸다고 보고했습니다. 두 번째 논문은 기존보다 더 높은 반사실적 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 그리고 희귀 데이터로부터의 일반화 성능을 달성했습니다. 마지막으로, AOW 기술은 비틀어짐(distillation)과 미세 조정(fine-tuning) 등 다양한 공격에 저항할 수 있으며, 높은 신뢰도의 워터마크 추출 성능을 보입니다.



### Multi-Grained Query-Guided Set Prediction Network for Grounded Multimodal Named Entity Recognition (https://arxiv.org/abs/2407.21033)
Comments:
          11 pages, 5 figures

- **Multiple Choice Questions Generation**: [{"What's New": '자동 객관식 문제(MCQ) 생성 평가에 대한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)가 소개되었습니다. KDA는 기존의 BLEU, ROUGE, METEOR 지표들이 간과한 교육적 가치를 고려하며, 학생의 지식 평가 능력을 측정합니다.'}, {'Technical Details': 'KDA 측정은 학생 응답 기반의 설문조사를 통해 이루어지며, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하는 방법으로 KDA를 근사화합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont는 KDA 및 실제 교실 환경에서의 사용성과 강한 상관관계를 보였습니다. n-gram 기반의 유사성 메트릭과 결합할 경우, 다양한 전문가 레이블 MCQ 품질 측정에서도 강력한 예측력을 나타냅니다.'}]

- **Deep Models Robustness**: [{"What's New": '딥러닝 모델의 spurious pattern 의존성을 극복하기 위한 contrastive learning 및 counterfactual augmentation 접근법이 제안되었습니다.'}, {'Technical Details': "기존 방법들은 사람이 counterfactual을 추가하거나 모델이 데이터셋에서 자동으로 근접한 counterfactual을 맞추어야 했으나, 새로운 접근법은 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 단어 인과관계를 더욱 견고하게 감독합니다."}, {'Performance Highlights': '새로운 접근법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data 상황에서의 일반화 성능에서 유의미한 성장을 보였습니다.'}]

- **Grounded Multimodal Named Entity Recognition (GMNER)**: [{"What's New": '기존의 MRC 기반 또는 sequence generation 기반 모델의 제약을 극복하기 위해 Multi-grained Query-guided Set Prediction Network (MQSPN)라는 새로운 통합 프레임워크가 제안되었습니다.'}, {'Technical Details': 'MQSPN은 Multi-grained Query Set (MQS)와 Multimodal Set Prediction Network (MSP), 그리고 Query-guided Fusion Net (QFNet)을 포함합니다. 이것들은 intra-entity와 inter-entity 레벨에서 적절한 관계들을 학습하도록 설계되었습니다.'}, {'Performance Highlights': '광범위한 실험 결과, 제안된 방법이 다양한 벤치마크에서 SOTA 성능을 달성했으며, 특히 세부적인 GMNER 벤치마크에서 2.83%의 F1 점수 향상을 이뤄냈습니다.'}]



### E-Commerce Product Recommendation System based on ML Algorithms (https://arxiv.org/abs/2407.21026)
- **Whats New**: 이번 뉴스레터에서는 최신 arXiv 논문에서 제안된 혁신적인 기술과 방법론을 소개합니다. 이들 연구는 자동 MCQ 생성, NLP의 robustness 향상, 그리고 eCommerce 추천 시스템의 개인 맞춤화를 다루고 있습니다.

- **Technical Details**: [{'Title': '자동 MCQ 생성의 새로운 평가 메트릭', 'Summary': '기존 평가 메트릭인 BLEU, ROUGE, METEOR가 가진 한계를 극복하기 위해, 본 연구에서는 새로운 평가 메트릭 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 학생들의 대상 사실(target fact)에 대한 지식을 평가하는 능력을 중점적으로 다룹니다. 또한, KDA_disc와 KDA_cont와 같은 자동 평가 메트릭을 사용하여 학생들이 문제를 해결하는 행동을 모방하는 방법을 제시합니다.'}, {'Title': 'NLP태스크의 robustness 개선', 'Summary': '최근의 딥러닝 모델들이 높은 정확도를 보였음에도 불구하고, spurious pattern에 의존하는 문제가 있다. 이 논문에서는 대조 학습(contrastive learning)과 반사실적인 증강(counterfactual augmentation)을 결합하여, 여러 개의 counterfactual들을 생성하고 집합적 의사결정을 통해 robustness를 개선하는 방법을 제안합니다.'}, {'Title': 'eCommerce 제품 추천 시스템의 개인 맞춤화', 'Summary': '본 연구에서는 최신 머신러닝 기법을 활용하여 각 고객에게 맞춤형 제품 추천을 제공하는 모델을 개발했습니다. PCA를 사용하여 feature를 축소하고, Gaussian Naive Bayes (GNB), Random Forest (RF), Logistic Regression (LR), Decision Tree (DT)와 같은 네 가지 머신러닝 알고리즘을 비교했습니다. 이 중, Random Forest 알고리즘이 99.6%의 최고 정확도를 달성했습니다.'}]

- **Performance Highlights**: [{'Title': '자동 MCQ 평가 메트릭 성능', 'Summary': 'Human studies에 따르면, KDA_disc과 KDA_cont는 실제 강의실에서의 usability와 강한 상관관계를 가지고 있으며, n-gram 기반 평가와 결합 시 전문가가 라벨링한 MCQ 품질 측정에 대해 강력한 예측력을 가집니다.'}, {'Title': 'NLP 모델의 robustness 향상 성능', 'Summary': '제안된 방법은 다양한 측면에서 상당한 향상을 이루었습니다. 특히, counterfactual robustness, cross-domain generalization, 그리고 부족한 데이터에서의 일반화 성능에서 눈에 띄는 개선 효과를 보여주었습니다.'}, {'Title': 'eCommerce 추천 시스템 성능', 'Summary': 'Random Forest 알고리즘이 99.6%의 정확도와 96.99 r square score, 1.92% MSE score, 0.087 MAE score를 달성했습니다. 이는 고객과 기업 모두에게 이점을 제공하는 결과입니다.'}]



### LLM-Find: An Autonomous GIS Agent Framework for Geospatial Data Retrieva (https://arxiv.org/abs/2407.21024)
- **What's New**: 이번 연구에서는 학습 평가 시간을 줄이기 위한 자동 Multiple Choice Questions (MCQ) 생성에 관한 평가 메트릭과 딥러닝 모델의 robustness 향상을 위한 대조 학습(constrative learning) 및 counterfactual augmentation 방법, 그리고 자율 공간 정보 시스템(GIS) 에이전트 프레임워크를 제안합니다.

- **Technical Details**: [{'MCQ Generation': '기존 MCQ 생성의 평가 메트릭이 n-gram 중심의 유사성(BLEU, ROUGE, METEOR)을 고려한다면, 우리는 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)를 도입했습니다. KDA를 인간 설문조사를 통해 측정하고, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다.', 'Robustness in NLP': '최근 딥러닝 모델들이 사람보다 높은 정확도를 보였지만, 여전히 spurious pattern에 의존하는 문제로 robustness가 제한됩니다. 이를 해결하기 위해, 우리는 여러 개의 counterfactual을 생성하고 집단적 의사 결정을 통해 더 robust하게 인과관계를 파악하는 방법을 제안합니다.', 'GIS Agent': '자율 GIS 에이전트 프레임워크 LLM-Find를 제안했습니다. 이는 LLM을 의사 결정자로 활용하여 필요한 공간 데이터를 선택하고 가져오며, 플러그 앤 플레이(plug-and-play) 방식으로 설계되었습니다. 다양한 데이터 소스(예: OpenStreetMap, US Census Bureau, ESRI World Imagery 등)에서 데이터를 검색할 수 있는 프로토타입 에이전트를 개발하여 그 가능성을 실험적으로 검증했습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 상황에서 높은 usability와 강한 상관관계를 가지는 것을 보였습니다. 또한, KDA_disc와 KDA_cont를 n-gram 기반 메트릭과 결합할 경우, 전문가가 라벨링한 다양한 MCQ 품질 측정 기준에 대해 강력한 예측력을 보였습니다.', 'Robustness in NLP': '우리 접근법은 다차원에서 상당한 성능 향상을 이루었는데, 그 예로 counterfactual robustness, cross-domain generalization 및 부족한 데이터에서의 일반화를 들 수 있습니다.', 'GIS Agent': '프레임워크는 다양한 데이터 소스에서 데이터를 성공적으로 검색할 수 있음을 입증했으며, 새로운 데이터 소스를 추가하는 데 유연성과 확장성이 뛰어났습니다.'}]



### A Comprehensive Survey on Retrieval Methods in Recommender Systems (https://arxiv.org/abs/2407.21022)
Comments:
          38 pages

- **What's New**: 새로운 'Knowledge Dependent Answerability (KDA)' 메트릭을 제안하여 기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 고려하지 않는다는 문제를 해결합니다. 또한 다양한 대응 문장을 생성하고 집합적 의사 결정을 통해 학습 모델의 편향 문제를 해결하고자 합니다.

- **Technical Details**: KDA는 학생들이 특정 사실에 대해 알고 있는지 평가하는 능력을 측정하는 메트릭입니다. 이 메트릭은 학생들의 응답을 기반으로 측정합니다. KDA_disc와 KDA_cont 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모사합니다. 또한 여러 개의 대응 문장을 생성하고 이를 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont 메트릭이 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며 전문가가 레이블링한 다양한 MCQ 품질 지표에 대해 강한 예측력을 가지고 있음을 보여주었습니다. 대응 문장을 생성하고 집합적 의사 결정을 통해 모델의 다양한 편향 문제를 해결하며, 다양한 차원에서 중요한 성능 향상을 달성했습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization.



### Adaptive Retrieval-Augmented Generation for Conversational Systems (https://arxiv.org/abs/2407.21712)
Comments:
          12 pages, under review

- **What's New**: 자동 다지선다형 질문(MCQ) 생성을 위한 새로운 자동 평가 메트릭 Knowledge Dependent Answerability(KDA)을 제안. KDA는 학생의 지식을 평가할 수 있도록 MCQ의 대답 가능성을 측정하는 새로운 접근법을 제시합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한 최근의 연구는 특정 순간에 외부 지식을 필요로 하는 시스템 응답을 예측하는 RAGate 모델을 제안하여, 불필요한 지식 증가를 방지하고 대화형 시스템의 반응성을 개선하는 방법을 소개하고 있습니다.

- **Technical Details**: KDA는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 교육적 가치를 반영하지 못한다는 한계를 극복하기 위해 개발되었습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 통해 학생의 문제 해결 행동을 모방하여 KDA를 자동으로 추정하는 메트릭입니다. 또한, 최근의 연구는 RAG는 무조건적으로 필요하지 않음을 지적하며, 각 회차에 어떤 외부 지식을 증가시킬 것인지 예측하는 RAGate라는 게이트 기능을 가진 모델을 제안합니다. 이 방법은 대화 시나리오에서 사용된 외부 지식의 관련성과 시스템 반응의 자신감 수준 사이의 상관 관계를 보고 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 상황에서의 사용성과 강한 상관관계를 나타냈으며, 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 예측력이 높았습니다. RAGate 모델은 대화형 시스템이 적절한 대화 회차에 외부 지식을 효율적으로 사용하도록 하여, 고품질의 시스템 반응을 생성할 수 있음을 입증했습니다.



### ProSpec RL: Plan Ahead, then Execu (https://arxiv.org/abs/2407.21359)
- **What's New**: 최근의 논문은 Neural Processing Language (NLP)와 Reinforcement Learning (RL) 분야에서 새로운 접근 방식을 제안합니다. MCQ 생성에서는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 교육적 가치를 평가하지 못하는 문제를 해결하기 위해 KDA (Knowledge Dependent Answerability)라는 새로운 메트릭을 제안합니다. 또한, RL에서 기존의 모델 기반 및 모델 프리방식의 단점을 극복하기 위해 'Prospective RL'과 'cycle consistency'를 통한 data efficiency 향상을 제안합니다.

- **Technical Details**: MCQ 관련 논문에서는 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하여, 사전 학습된 언어 모델을 활용해 KDA를 근접하게 측정합니다. RL 논문에서는 역동 모델을 이용해 미래 상태를 예측하고, Model Predictive Control (MPC)와 cycle consistency를 적용해 낮은 리스크와 높은 데이터를 효율적으로 활용하는 방법을 소개합니다.

- **Performance Highlights**: MCQ 생성에서 KDA_disc와 KDA_cont가 실제 수업 환경에서 강한 상관관계를 보여주었으며, 전문가가 라벨링한 MCQ 품질 지표와 예측력이 강하게 일치하였습니다. Prospective RL의 성능은 DeepMind 벤치마크에서 유의미한 성능 향상을 이뤄냈으며, 데이터 효율성을 크게 개선하였습니다.



### LawLLM: Law Large Language Model for the US Legal System (https://arxiv.org/abs/2407.21065)
Comments:
          21 pages, 2 figures, accepted at the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024) for the Applied Research Paper track

- **What's New**: 최근 발표된 아카이브(arXiv) 논문들은 다양한 AI 모델과 평가 메트릭의 새로운 접근법을 제시하고 있습니다. 법률 분석, MCQ 생성, NLP의 강건성(robustness) 개선 등 다양한 분야에서의 혁신적인 연구들이 포함되어 있습니다.

- **Technical Details**: [{'MCQ Generation': "기존의 MCQ 자동 생성 평가 메트릭(BLEU, ROUGE, METEOR)들이 n-gram 기반 유사도에 초점을 맞추어 교육적 가치를 무시했다는 문제를 해결하기 위해, 'Knowledge Dependent Answerability(KDA)'라는 새로운 메트릭을 제안합니다. KDA는 학생이 대상 사실(target fact)을 알고 있을 경우 MCQ의 답변 가능성을 측정합니다."}, {'NLP Robustness': "최근 NLP 태스크에서 인간 이상의 정확성을 보이는 딥 모델이 많은데, 이 모델들이 'spurious pattern'에 의존하여 강건성이 제한됩니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 'counterfactual augmentation'을 통해 모델의 강건성을 개선하는 방법을 제안합니다."}, {'Legal Large Language Model (LawLLM)': '법률 언어의 복잡성과 특수 용어 때문에 사례 찾기와 판례 예측이 어려운 문제를 해결하기 위해 LawLLM을 제안합니다. LawLLM은 Similar Case Retrieval(SCR), Precedent Case Recommendation(PCR), Legal Judgment Prediction(LJP) 등의 다양한 법률 태스크를 동시에 수행할 수 있습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성에서 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 레이블 MCQ 품질 측정에서 강한 예측력을 보였습니다.'}, {'NLP Robustness': '제안된 접근법을 사용하면 집단적 의사 결정을 통해 단어들의 인과관계를 더 robust하게 파악할 수 있어, 다양한 측면에서(Bias, Cross-domain Generalization, 데이터가 부족한 상황에서의 Generalization) 의미 있는 개선을 보였습니다.'}, {'LawLLM': 'LawLLM은 zero-shot 및 few-shot 시나리오에서 기존 기준 모델보다 일관되게 우수한 성능을 보였습니다. 이 모델은 특히 유사한 사례와 판례를 명확히 구분함으로써 법률 분야의 다양한 요구를 충족시킬 수 있습니다.'}]



### Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks (https://arxiv.org/abs/2407.21059)
- **What's New**: 이번 AI 뉴스레터에서는 여러 최신 연구의 핵심 내용을 다룹니다. 첫째, 자동 Multiple Choice Questions (MCQ) 생성의 평가 메트릭에서 교육적 가치를 반영하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 둘째, 자연어 처리(NLP) 모델의 robustness 향상을 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법이 소개됩니다. 셋째, Retrieval-augmented Generation (RAG) 시스템의 모듈화된 framework인 Modular RAG가 제안되어, 복잡한 RAG 시스템의 유연성 및 확장성을 높이는 방법을 설명합니다.

- **Technical Details**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Details': 'BLEU, ROUGE, METEOR와 같은 전통적 평가 메트릭이 아니라, MCQ의 대답 가능성 및 교육적 가치를 평가하는 Knowledge Dependent Answerability (KDA) 메트릭을 제안합니다. KDA_disc와 KDA_cont를 사용해 사람의 문제 해결 행동을 모방하여 자동 평가를 수행합니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': '최근 모델들이 spurious patterns에 의존하는 문제를 해결하고자 counterfactual augmentation과 contrastive learning 기법을 사용합니다. 다양한 counterfactual 생성과 집합적 의사 결정을 통해 인과관계를 더 robust하게 파악합니다.'}, {'Paper': 'Modular RAG Framework', 'Details': 'Retrieval-augmented Generation (RAG) 시스템의 복잡성을 해결하기 위해 모듈화된 RAG framework를 제안합니다. 모듈, 서브모듈, 연산자로 나누어 시스템 유지 보수성과 유연성을 강화하고, RAG Flow를 통해 다양한 RAG 방법론을 통합합니다.'}]

- **Performance Highlights**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Highlights': 'KDA_disc와 KDA_cont는 KDA와 실제 강의실 세팅에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 메트릭과 결합 시 전문가가 레이블한 다양한 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다.'}, {'Paper': 'Robustness in NLP Models', 'Highlights': '집단적 결정 방식을 통해 기존 모델들의 진귀 데이터에서의 일반화 성능과 cross-domain generalization, counterfactual robustness에서 의미 있는 향상을 달성했습니다.'}, {'Paper': 'Modular RAG Framework', 'Highlights': '모듈화된 RAG 시스템은 유연성과 확장성이 뛰어나, 다양한 데이터 소스 및 작업 시나리오에 맞게 모듈과 연산자를 조합할 수 있습니다. 이는 RAG 기술의 실용적 배치와 연구 방향에 새로운 기회를 제공합니다.'}]



### What Matters in Explanations: Towards Explainable Fake Review Detection Focusing on Transformers (https://arxiv.org/abs/2407.21056)
- **What's New**: 이번 논문에서 자동으로 다중 선택 질문(MCQ)을 생성하고 평가하는 새로운 방법을 제안했습니다. BLEU, ROUGE, METEOR와 같은 기존 메트릭은 교육적 가치를 평가하지 못합니다. 본 논문에서는 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 도입하여 MCQ의 대답 가능성을 측정하고, 학생의 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: KDA는 특정 대상 사실(target fact)에 대한 MCQ의 답변 가능성을 측정합니다. 이를 위해 학생 응답을 기준으로 KDA를 측정하는 방법을 제시하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: 인간 평가 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 세트에서 전문가가 라벨링한 사용성과 강한 상관관계를 가짐을 확인했습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, 이 메트릭들은 다양한 전문가가 라벨링한 MCQ 품질 측정에 대해 높은 예측력을 보여줍니다.



### Advancing Chart Question Answering with Robust Chart Component Recognition (https://arxiv.org/abs/2407.21038)
- **What's New**: 이 논문에서는 Multiple Choice Questions(MCQ) 생성의 자동 평가 메트릭인 BLEU, ROUGE, METEOR가 교육적 가치를 반영하지 못한다는 문제를 해결하기 위해 Knowledge Dependent Answerability(KDA)라는 새로운 메트릭을 제안합니다. 이는 학생들이 특정 사실에 대한 지식을 평가하는 능력을 기준으로 MCQ의 답변 가능성을 측정합니다.

- **Technical Details**: 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하여 KDA를 대체하는 방법을 소개합니다. 이는 사전 학습된 언어 모델을 통해 학생들의 문제 해결 경향을 모방합니다. 또한, 다양하고 복잡한 차트 해석을 개선하기 위해 Chartformer라는 프레임워크와 Question-guided Deformable Co-Attention(QDChart) 메커니즘을 소개합니다.

- **Performance Highlights**: KDA 기반 메트릭은 n-gram 기반 메트릭과 결합하여 MCQ 품질을 예측하는 데 강력한 예측력을 보였습니다. Chartformer는 mAP에서 3.2%, ChartQA에서 15.4% 향상을 보여주었으며, 기존 모델보다 뛰어난 성능을 입증했습니다.




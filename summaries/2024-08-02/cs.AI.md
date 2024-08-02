New uploads on arXiv(cs.CL)

### AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation (https://arxiv.org/abs/2408.00764)
- **What's New**: 이번 뉴스레터는 자동 Multiple Choice Questions (MCQ) 평가와 대규모 언어 모델(LLM) 기반 에이전트의 계획 능력 향상에 대한 최신 연구를 다룹니다. 또한, deep model의 robustness 문제를 해결하는 새로운 접근법에 대해서도 소개합니다.

- **Technical Details**: {'MCQ Generation': 'MCQ 생성의 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 고려하지 않습니다. 새로운 평가 방법인 Knowledge Dependent Answerability(KDA)는 학생의 지식을 평가하는 MCQ의 답변 가능성(answerability)을 측정합니다. 이 메트릭은 학생 반응 설문조사를 기반으로 하며, KDA_disc와 KDA_cont를 제안합니다.', 'Deep Model Robustness': 'Deep model의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 이용합니다. 여러 개의 counterfactual을 생성하고 collective decision을 통해 단어들의 인과관계를 더 견고하게 파악하는 접근법을 제안합니다.', 'LLM-based Agent Training': 'LLM 기반 에이전트의 계획 능력을 향상시키기 위해 instruction tuning을 이용합니다. AgentGen이라는 프레임워크를 통해 다양한 환경과 난이도가 점진적으로 상승하는 계획 태스크를 자동으로 생성합니다. 이를 통해 LLM의 계획 능력을 향상시킵니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 사용성과 강한 상관관계를 보이며, n-gram 기반 메트릭과 결합하여 전문가가 평가한 MCQ 품질을 잘 예측합니다.', 'Deep Model Robustness': '여러 실험 결과, 제안된 방법이 counterfactual robustness, 교차 도메인 일반화(cross-domain generalization), 및 희소 데이터에서의 일반화(generalization from scarce data) 등 다양한 측면에서 개선되었습니다.', 'LLM-based Agent Training': 'AgentGen 프레임워크로 instruction-tuning된 Llama-3 8B 모델이 GPT-3.5와 GPT-4를 능가하는 성능을 보였습니다.'}



### CERT-ED: Certifiably Robust Text Classification for Edit Distanc (https://arxiv.org/abs/2408.00728)
Comments:
          22 pages, 3 figures, 12 tables. Include 11 pages of appendices

- **What's New**: 기존의 자동생성 MCQ 평가 메트릭이 교육적 가치를 고려하지 못하는 문제를 해결하기 위해, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 반면, NLP 태스크에서 deep model들이 spurious pattern에 의존해 robust하지 못한 문제를 해결하기 위해 counterfactual augmentation을 도입한 contrastive learning을 활용합니다. 또한, 텍스트 데이터에서 edit distance를 사용한 attacks에 대해 robust한 모델을 구현하기 위해 CERTified Edit Distance (CERT-ED) 방어 메트릭을 제안했습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 평가합니다. KDA_disc와 KDA_cont는 pretrained language models를 사용해 학생의 문제 해결 행동을 모방합니다. 이에 반해, 새로운 NLP 모델은 여러 개의 counterfactual을 생성하여 collective decision 방식으로 각 단어의 인과관계를 더 robust하게 파악합니다. CERT-ED는 Randomized Deletion을 확장하여 edit distance 반경 내의 임의의 attacks에 대해 NLP 분류 태스크에 대한 multi-class 예측을 인증합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 설정에서의 사용성과 강한 상관관계를 보였습니다. 새로운 NLP 접근 방식은 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터에서의 generalization 측면에서 상당한 개선을 보였습니다. CERT-ED는 기존 방법보다 5개 데이터셋 중 4개에서 더 높은 정확성과 인증된 cardianlity를 보였으며, 다양한 공격 설정에서 empiral robustness를 개선했습니다.



### Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions (https://arxiv.org/abs/2408.00727)
- **papers**: [{"What's New": '자동 MCQ생성을 위한 새로운 평가 메트릭 KDA가 제안되었습니다. KDA는 MCQ의 교육적 가치를 평가하는 데 초점을 맞추며 학습자의 목표 사실에 대한 지식 평가 능력을 측정합니다.', 'Technical Details': '기존 메트릭 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에 초점을 맞추지만, KDA는 학생의 응답을 기반으로 한 KDA 측정 방법을 도입합니다. 이 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA를 근사합니다.', 'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_soft가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 확인했습니다. KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합하여 다양한 전문가 레이블 MCQ 품질 측정에 대한 강력한 예측 성능을 보였습니다.'}, {"What's New": '대조 학습(contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용하여 NLP 모델의 robustness를 개선한 새로운 접근 방식이 제안되었습니다.', 'Technical Details': '기존의 반사실적 증강 방법은 인간이나 기계가 데이터셋 내에서 반사실적 요소를 추가하거나 자동으로 찾아야 하는데, 이는 여전히 spurious correlation에 영향을 받습니다. 본 논문에서는 여러 개의 반사실적 사례를 생성하고 집합적 의사 결정을 통해 단어의 인과성을 robust하게 감독하는 방식을 제안합니다.', 'Performance Highlights': '다양한 차원에서의 성능 개선을 통해 해당 접근 방식이 반사실적 robustness, 도메인 간 일반화, 그리고 희소한 데이터로부터의 일반화를 달성함을 입증했습니다.'}, {"What's New": '의학 질문에 대한 성능을 향상시키기 위해 i-MedRAG라는 반복적 RAG 프레임워크가 제안되었습니다. 이 모델은 이전 정보 검색 시도에 기반한 추가 질문을 반복적으로 할 수 있도록 설계되었습니다.', 'Technical Details': 'i-MedRAG는 기본 RAG 시스템에서 한 번만 수행되는 정보 검색 단계를 여러 단계로 확장하여 복잡한 임상 논리를 해결합니다. 이는 LLM이 반복적으로 후속 질문을 생성하고 이를 바탕으로 원래 질문에 대한 더 나은 답변을 생성할 수 있도록 합니다.', 'Performance Highlights': 'i-MedRAG는 다양한 LLM에서 미국 의사 면허 시험(USMLE) 하위 집합 및 Massive Multitask Language Understanding (MMLU) 데이터셋의 복잡한 질문에 대해 향상된 성능을 보였습니다. 특히, i-MedRAG는 GPT-3.5에 대한 zero-shot 설정에서 모든 이전 프롬프트 엔지니어링 및 미세 조정 방법을 능가하여 MedQA 데이터셋에서 69.68%의 정확도를 달성했습니다.'}]



### Improving Text Embeddings for Smaller Language Models Using Contrastive Fine-tuning (https://arxiv.org/abs/2408.00690)
Comments:
          Code: this https URL, Huggingface: this https URL

- **What's New**: MCQ 생성을 위한 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 또한, 작은 언어 모델의 텍스트 임베딩 품질을 향상시키기 위한 방법으로 contrastive fine-tuning을 소개했습니다.

- **Technical Details**: MCQ 자동 생성에서는 KDA 메트릭을 통해 학생들의 반응을 바탕으로 대답 가능성을 측정했습니다. 두 개의 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안하여 사전 학습된 언어 모델을 사용해 학생들의 문제 해결 행동을 모사했습니다. 작은 언어 모델('MiniCPM', 'Phi-2', 'Gemma')의 텍스트 임베딩 품질 향상을 위해 contrastive fine-tuning을 적용했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 메트릭과 결합 시 전문가가 평가한 MCQ 품질에 대한 예측력을 높였습니다. MiniCPM은 contrastive fine-tuning을 통하여 평균 56.33%의 성능 향상을 보여주었습니다.



### Assessing the Variety of a Concept Space Using an Unbiased Estimate of Rao's Quadratic Index (https://arxiv.org/abs/2408.00684)
- **What's New**: 최신 연구는 자동 MCQ 생성의 평가 메트릭으로 BLEU, ROUGE, METEOR와 같은 기존 지표가 교육적 가치를 반영하지 못한다는 문제점을 지적하고, 지식 종속 가능성(KDA)이라는 새로운 메트릭을 제안하였습니다. 또한, NLP 태스크에서 deep model들의 robustness를 향상시키기 위해 대조 학습(contrastive learning) 및 반사실적 증가(counterfactual augmentation)를 활용하는 방법이 제안되었습니다. 마지막으로, 설계 창의성과 다양한 개념 공간 탐색을 측정하는 새로운 거리 기반 다양성 메트릭을 소개하였습니다.

- **Technical Details**: MCQ 평가에서, KDA는 학생들이 해당 사실에 관한 지식을 가지고 있는 상황에서 MCQ의 대답 가능성을 측정하는 메트릭입니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 자동으로 학생들의 문제 해결 행동을 모방하면서 KDA를 근사합니다. NLP 모델에서, 여러 개의 반사실적(counterfactual) 예시를 생성하고 집합적인 의사 결정(collective decisions)을 통해 단어들의 인과관계를 평가하는 방법이 제안되었습니다. 이 방법은 스푸리어스 상관관계(spurious correlations)에 덜 민감하여 나은 robustness를 제공합니다. 설계 창의성 측정에서, SAPPhIRE 모델을 통해 두 개념 간의 실수 거리(real-valued distance)를 측정하는 새로운 다양성 메트릭을 제안하고, 이 프레임워크는 VariAnT라는 소프트웨어 도구에 구현되었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보여주었으며, 전문가들이 레이블링한 MCQ 품질 지표들에 대한 예측력도 높았습니다. NLP 분야에서, 제안된 방법은 대조 학습 및 반사실적 증가를 통해 반사실적 robust성, 도메인 간 일반화, 그리고 희소 데이터로부터의 일반화 성능에서 상당한 개선을 보였습니다. 설계 도구 VariAnT는 설계 초기에 다양한 개념을 탐색하고 평가하는데 유용성을 입증했습니다.



### Leveraging Entailment Judgements in Cross-Lingual Summarisation (https://arxiv.org/abs/2408.00675)
- **What's New**: 이번 논문에서는 MCQ(객관식 질문) 생성의 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 학생이 특정 사실에 대한 지식을 가지고 있을 때 MCQ의 답변 가능성을 측정합니다. 또한, 논문은 NLP 태스크에서 recent deep model들의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 접근법을 소개합니다. 마지막으로, Cross-Lingual Summarisation (CLS)의 데이터 내 기계적 신뢰성을 평가하고 향상시키기 위한 새로운 방법론을 제안합니다.

- **Technical Details**: MCQ 평가에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다. 약점으로 지적된 기존 BLEU, ROUGE, METEOR 메트릭의 한계를 극복합니다. NLP 부문에서는 다수의 counterfactual을 생성하고 collective decision making을 통해 각 용어의 causality를 강화하는 방법을 제안합니다. CLS 부문에서는 cross-lingual Natural Language Inference (X-NLI)를 활용해 올바른 training data를 사용하고 unlikelihood loss를 사용해 모델을 학습합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성과 높은 상관관계를 보였으며, 전문가 라벨링 기준에서도 높은 예측력을 기록했습니다. NLP task에서는 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization에서 상당한 성능 향상을 달성했습니다. CLS에서는, 단순히 더 신뢰할 수 있는 smaller dataset으로 fine-tuning하여 정보 전달력을 유지하면서도 더 높은 신뢰도의 요약을 생성할 수 있음을 실험으로 입증했습니다.



### Aligning Multiple Knowledge Graphs in a Single Pass (https://arxiv.org/abs/2408.00662)
- **What's New**: {'MCQ Generation': '자동으로 Multiple Choice Questions(MCQ)를 생성하는 것은 교사의 학습 평가 시간을 크게 줄일 수 있습니다. 그러나 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 고려하지 않고 있습니다. 이를 해결하기 위해 우리는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다.', 'Robustness in NLP': '최근 딥러닝 모델이 NLP 태스크에서 높은 정확성을 보였으나, spurious patterns에 의존하는 문제로 인해 robustness가 제한되고 있습니다. 이 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 데이터 증순(counterfactual augmentation)을 활용하는 방법을 제안합니다.', 'Entity Alignment': '기존 Entity Alignment(EA) 방법은 두 개의 서로 다른 지식 그래프(KG)를 맞추는 것에 초점을 맞추고 있으나, 다중 KG를 맞추는 방법은 연구되지 않았습니다. 이를 해결하기 위해 MultiEA라는 새로운 프레임워크를 제안합니다.'}

- **Technical Details**: {'MCQ Generation': 'KDA는 MCQ의 대답 가능성(answerability)을 측정하여 대상 사실에 대한 학생의 지식을 평가합니다. 우리는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 KDA를 근사합니다.', 'Robustness in NLP': '제안된 방법은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과관계를 robust하게 감독합니다. 이를 통해 기존 방법보다 높은 robustness와 일반화(generalization)를 달성합니다.', 'Entity Alignment': 'MultiEA는 모든 후보 KG를 공통 특성 공간에 임베딩하고, 고차 유사성을 도입하여 정렬 성능을 향상시키는 혁신적인 추론 강화 기술을 제안합니다. 또한, 우리는 다중 KG를 포함하는 두 개의 새로운 벤치마크 데이터셋을 구축하여 MultiEA의 효과성을 검증합니다.'}

- **Performance Highlights**: {'MCQ Generation': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서 높은 사용성을 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 높은 예측력을 보였습니다.', 'Robustness in NLP': '집합적 의사 결정 방식을 사용함으로써 속성 기반 합성(attribution-based synthesis)의 편향에 덜 민감하여 다양한 차원에서 의미 있는 개선을 달성했습니다.', 'Entity Alignment': '실험 결과, MultiEA는 효율적이고 효과적으로 다중 KG를 단일 패스로 정렬할 수 있었으며, 더 나은 정렬 성능을 달성했습니다.'}



### Downstream bias mitigation is all you need (https://arxiv.org/abs/2408.00612)
Comments:
          21 pages, 11 figures, 2 tables

- **What's New**: 이 논문은 자동 생성 다중 선택 질문(MCQ) 평가에 있어서 기존의 n-gram 기반 메트릭인 BLEU, ROUGE, 그리고 METEOR가 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)을 제안합니다. KDA는 생성된 MCQ가 특정 지식에 대한 학생의 이해도를 평가하는 능력을 측정합니다.

- **Technical Details**: KDA는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 통해 측정되며, pretrained language models를 활용하여 학생의 문제 해결 행동을 모방합니다. 이 메트릭은 인간 설문 조사를 기반으로 KDA를 측정하는 방식에서 출발하여, 자동 평가로 가는 방법을 제시합니다. 또한, KDA_disc와 KDA_cont는 실제 강의실 환경에서 전문가들이 라벨링한 사용성과 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 메트릭과 결합할 때 다양한 전문가 라벨링 MCQ 품질 측정치에 대해 강력한 예측 능력을 보였습니다. 이는 MCQ 평가 메트릭의 교육적 가치를 높이는데 중요한 기여를 합니다.



### Closing the gap between open-source and commercial large language models for medical evidence summarization (https://arxiv.org/abs/2408.00588)
- **What's New**: 이 논문은 MCQ(Multiple Choice Questions) 자동 생성 및 평가를 위한 새로운 지표(Knowledge Dependent Answerability, KDA)를 제안하였습니다. 기존의 지표들이 문장 유사성에 초점을 맞추는 반면, KDA는 학생의 지식 평가 능력을 중점적으로 측정합니다. 다른 논문에서는 최신 NLP 모델들의 강력한 성능에도 불구하고, 제한된 robustness 문제를 해결하기 위해 대조 학습(Contrastive Learning)과 반사실적 증강(Counterfactual Augmentation)을 도입합니다. 또한, 의료 분야에서 열린 소스 LLMs의 성능을 향상시키기 위해 세부적 조정을 시도한 연구도 포함되어 있습니다.

- **Technical Details**: MCQ 평가를 위해 KDA_disc와 KDA_cont라는 지표를 제안했습니다. 이는 사전학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방합니다. 대조 학습을 위해 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 spurious correlations를 제거하는 방법을 제안합니다. 의료 증거 요약을 위해 PRIMERA, LongT5, Llama-2 모델을 세부 조정하여 성능 향상을 목표로 했으며, Low-Rank Adaptation (LoRA) 방법을 사용해 세부 조정을 최적화했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 다른 n-gram 기반 유사성 지표와 결합하여 높은 예측력을 나타냈습니다. 대조 학습 접근 방식은 다양한 측면에서 개선된 robustness, cross-domain generalization, 그리고 데이터가 부족한 상황에서도 유의미한 성능 향상을 보였습니다. MedReview 데이터셋을 사용한 실험에서는 세부 조정된 모델들이 ROUGE-L, METEOR, CHRF 점수에서 유의미한 향상을 보였으며, 특히 LongT5 모델은 GPT-3.5와 비슷한 성능을 보여주었습니다.



### Non Verbis, Sed Rebus: Large Language Models are Weak Solvers of Italian Rebuses (https://arxiv.org/abs/2408.00584)
Comments:
          Code: this https URL. Artifacts: this https URL

- **What's New**: 자동 MCQ(Multiple Choice Question) 생성 알고리즘에서 n-gram 기반 유사성 평가 메트릭이 교육적 가치를 무시하는 문제를 해결하기 위해, 지식 종속 가능성(KDA, Knowledge Dependent Answerability)이라는 새로운 평가 지표를 제안했습니다.

- **Technical Details**: KDA는 대상 사실(target fact)에 대한 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. 구체적으로, 학생 응답을 기반으로 KDA를 측정하는 방법을 제시한 후, 사전 훈련된 언어 모델을 활용하여 KDA를 근사하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안했습니다.

- **Performance Highlights**: Human evaluation을 통해, KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 가지는 것을 보여주었습니다. 그리고 n-gram 기반 유사성 메트릭과 결합하면 전문가가 라벨링한 다양한 MCQ 품질 지표들에 대해 높은 예측력을 가집니다.



### Intermittent Semi-working Mask: A New Masking Paradigm for LLMs (https://arxiv.org/abs/2408.00539)
- **What's New**: 최근 자동 MCQ (Multiple Choice Questions) 생성 및 LLM (Large Language Models) 대화 시스템에서의 향상된 메트릭과 방법론이 제안되었습니다. MCQ 생성의 경우, 새로운 평가 메트릭인 지식 종속 가능성 (KDA)을 제안하여 질문의 교육적 가치를 평가합니다. 반면에 LLM 대화 시스템에서는 Intermittent Semi-working Mask (ISM)이라는 새로운 마스킹 기법을 도입하여 다중 회전 대화 중의 응답 생성 지연을 줄입니다.

- **Technical Details**: [{'MCQ Generation': '기존 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR은 n-gram 기반의 유사성만을 고려하여 교육적 가치를 평가하는 데 한계를 가집니다. 새로운 평가 메트릭 KDA는 학생의 대답 가능성을 측정하고, 인간 설문으로부터 얻은 학생 반응을 통해 이를 평가합니다.'}, {'LLM Dialogue System': '기존 LLM 모델들은 인과적 LLM(causal LLM)과 접두사 LLM(prefix LLM)으로 나뉘며, 접두사 LLM은 다중 회전 대화에서 역사적 문맥을 잘 활용하지만, 높은 생성 지연의 문제를 가집니다. 새로운 마스킹 기법인 ISM은 쿼리와 응답에 대해 교차적인 양방향 및 단방향 어텐션을 적용하여 이 문제를 개선합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 KDA와 강한 상관관계를 보이며, 전문가에 의해 라벨링된 실제 강의실 세트에서의 사용성과도 매우 높은 상관관계를 나타냈습니다.'}, {'LLM Dialogue System': 'ISM은 GPT-4를 사용하여 평가한 다중 회전 대화 벤치마크 데이터셋에서 기존 최첨단 생성 품질을 유지하면서도 낮은 대화 생성 지연을 달성했습니다.'}]



### GalleryGPT: Analyzing Paintings with Large Multimodal Models (https://arxiv.org/abs/2408.00491)
Comments:
          Accepted as Oral Presentation at ACM Multimedia 2024

- **What's New**: 자동 MCQ 생성을 위한 새로운 평가 메트릭 Knowledge Dependent Answerability (KDA)을 제안하며, 이는 MCQ의 대답 가능성(answerability)을 측정하여 학생의 지식 평가를 가능하게 합니다. 또한, 딥 러닝 모델의 취약성을 극복하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 접근 방식을 소개합니다. 마지막으로, 예술 작품 분석을 위한 포괄적인 형식 분석을 제안하고 이를 위해 커다란 멀티 모달 모델 GalleryGPT를 개발하였습니다.

- **Technical Details**: KDA는 학생의 문제 해결 행동을 모방하기 위한 사전 학습된 언어 모델을 이용하여 자동 평가 메트릭(KDA_disc, KDA_cont)을 도입하였습니다. 딥 러닝 모델의 경우, '여러 개의' 반사실(counterfactual)을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 보장할 수 있는 robust한 방법을 제안합니다. 예술 작품 분석에서는 PaintingForm이라는 대규모 데이터셋과 LLaVA 아키텍처를 기반으로 한 GalleryGPT 모델을 소개하며, 이는 시각적 요소에 초점을 맞춘 형식 분석을 생성합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅과 강한 상관관계를 가지며, n-gram 기반 메트릭과 결합하여 전문가 레이블이 붙은 다양한 MCQ 품질 측정 지표에 강력한 예측력을 보입니다. 딥 러닝 모델의 robust 접근 방식은 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터로부터의 일반화 능력에서 상당한 성능 향상을 보여주었습니다. GalleryGPT는 다양한 예술 분석 작업에서 우수한 성능을 나타내며, 특히 수집된 데이터 셋의 우수성과 모델의 시각적 이해 및 분석 능력을 입증했습니다.



### In-Context Example Selection via Similarity Search Improves Low-Resource Machine Translation (https://arxiv.org/abs/2408.00397)
- **What's New**: 이 논문은 자동으로 다지선다형 질문(MCQ)를 생성하는 기술의 평가 메트릭스를 제안하며, 기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반의 메트릭이 교육적 가치를 평가하지 못하는 문제를 해결합니다. 새로운 메트릭, Knowledge Dependent Answerability(KDA),는 대상 지식에 기반한 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 갖춘 질문을 자동으로 생성합니다.

- **Technical Details**: KDA는 학생의 응답 데이터를 사용한 인간 설문 조사를 통해 측정되며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이를 통해 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA의 근사치를 계산합니다. 기존 n-gram 기반 유사성 메트릭과 결합하면, KDA_disc와 KDA_cont는 전문가들이 라벨링한 다양한 MCQ 품질 척도를 강력하게 예측할 수 있습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 입증했습니다. KDA 기반의 자동 평가 메트릭은 다양한 전문가 라벨링 MCQ 품질 척도에서도 강력한 예측력을 보였습니다.



### DeliLaw: A Chinese Legal Counselling System Based on a Large Language Mod (https://arxiv.org/abs/2408.00357)
Comments:
          CIKM 2024, 5 pages with 3 figures

- **What's New**: 이번 소식에서는 세 가지 주요 논문을 다뤘습니다. 첫 번째 논문은 자동 Multiple Choice Questions(MCQ) 생성에서 교육적 가치를 평가하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 두 번째 논문은 대조 학습과 반사실적 데이터(E.counterfactual data)를 통해 NLP 태스크에서 모델의 강건성(robustness)을 높이는 방법을 제안합니다. 마지막으로, 중국 법률 도메인에서의 법률 검색 시스템 DeliLaw에 대해 소개합니다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭의 한계를 지적하고, KDA metic을 통해 MCQ의 지식 종속 대답 가능성을 평가하는 방식을 제안합니다. 두 번째 논문은 여러 개의 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다. 마지막 논문은 RoBERTa-large 아키텍처로 의도 분류(intent classification) 모델을 구축하고, BGE 임베딩 모델을 사용하여 법률 검색 시스템을 개선하는 방법을 제시합니다.

- **Performance Highlights**: 첫 번째 논문에서 제안한 KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, 기존 n-gram 기반 유사성 메트릭과 결합 시 더 높은 예측력을 보여주었습니다. 두 번째 논문에서는 대조 학습을 통해 반사실적 강건성, 교차 도메인 일반화, 희소 데이터 일반화에서 중요한 개선을 달성했습니다. 세 번째 논문에서는 법률 검색 시스템의 정확도와 재현율 향상을 위해 두 단계의 미세 조정을 통해 법률 문서를 효과적으로 검색할 수 있음을 실험적으로 입증하였습니다.



### Bailing-TTS: Chinese Dialectal Speech Synthesis Towards Human-like Spontaneous Representation (https://arxiv.org/abs/2408.00284)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 논문에서는 여러 개의 counterfactuals를 생성하여 스퍼리어스 패턴(spurious patterns) 문제를 해결하는 방법을 제안합니다. 또한 자동 MCQ(Multiple Choice Questions) 생성의 새로운 평가 메트릭으로 KDA(Knowledge Dependent Answerability)를 소개하고, Chinese 방언 음성을 생성할 수 있는 대규모 TTS 모델인 Bailing-TTS를 제안합니다.

- **Technical Details**: 1. 새로운 KDA 메트릭은 학생의 목표 사실 대상 지식을 평가하기 위해 설계되었으며, BLEU, ROUGE, METEOR와 같은 기존의 n-gram 기반 평가 메트릭의 한계를 극복합니다. 
2. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 강건한 모델을 만들고, 모델의 편향성을 줄이는 방법을 제안합니다. 
3. Bailing-TTS 모델은 연속적인 반감독 학습 프레임워크와 다단계 훈련 프로세스를 사용하여 텍스트와 음성 토큰의 정렬을 촉진하고, 특정 트랜스포머 아키텍처를 사용하여 다양한 중국 방언 데이터를 처리할 수 있습니다.

- **Performance Highlights**: 1. KDA_disc 및 KDA_cont는 KDA와 실제 강의실에서의 사용성과 강한 상관관계를 갖고 있음이 인간 연구를 통해 입증되었습니다.
2. 제안된 방법으로 구성된 counterfactual 기반 모델은 여러 차원에서 성능 향상을 보여주었고, 특히 반사례 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 그리고 소량 데이터로부터의 일반화(generalization from scarce data)에서 знач来한 성과를 올렸습니다.
3. Bailing-TTS는 중국 방언 음성 생성에서 인간 수준의 자연스러움을 가지며, WER 및 MOS 평가 지표에서 뛰어난 성능을 보였습니다. 예를 들어, Mandarin 음성 평가에서 WER 1.86, MOS 4.32를 기록하여 인간 스피커와 비슷한 성능을 보여주었습니다.



### Navigating Text-to-Image Generative Bias across Indic Languages (https://arxiv.org/abs/2408.00283)
Comments:
          Accepted in ECCV 2024

- **What's New**: 다양한 언어 및 문화적 요소를 반영하여 텍스트-이미지 생성 모델(TTI)을 평가하는 새로운 벤치마크 'IndicTTI'가 제안되었습니다. 이 벤치마크는 인도 전역에서 사용되는 인도 언어에 대한 TTI 모델의 생성을 평가하고, 이를 통해 영어 외 언어에서도 질 높은 이미지 생성을 목표로 합니다.

- **Technical Details**: IndicTTI 벤치마크는 30개의 인도 언어와 영어를 포함하여 다소 복잡한 프롬프트를 사용하여 이미지를 생성합니다. COCO-NLLB 데이터셋을 샘플로 사용하여 1000개의 다양한 이미지-캡션 쌍을 선정하고, 이를 통해 모델의 생성 성능 및 문화적 적합성을 평가합니다. 평가는 두 가지 측면에서 이루어지며, 하나는 생성 결과의 정확성(correctness)을, 다른 하나는 표현 다양성(representation)을 측정합니다.

- **Performance Highlights**: 본 연구에서는 Stable Diffusion, Alt Diffusion, Midjourney, 그리고 Dalle3와 같은 모델을 이용하여 각기 다른 언어 및 스크립트의 문화적 표현을 평가합니다. 특히, 실험에서 특정 언어 스크립트가 인도 신들에 대한 이미지 생성에 어떻게 영향을 미치는지, 아시아 여성과 커플의 생성 빈도, Dalle3 모델의 높은 정확도 등을 관찰하였습니다. 이를 통해 각 TTI 모델의 강점과 개선이 필요한 부분을 밝혀냈습니다.



### QUITO: Accelerating Long-Context Reasoning through Query-Guided Context Compression (https://arxiv.org/abs/2408.00274)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 추론 복잡성과 계산 비용을 크게 줄여주는 새로운 Query-gUIded aTtention cOmpression (QUITO) 방법을 소개합니다. QUITO는 질문에 대한 컨텍스트의 중요도를 평가하여 불필요한 정보를 걸러내고, 컨텍스트 길이 제한을 충족시키는 세 가지 필터링 방법을 제안합니다.

- **Technical Details**: QUITO는 self-attention 메커니즘을 사용하여 토큰의 중요도를 평가합니다. 이 방법은 상대적으로 작은 크기의 LLM (0.5 billion parameters)을 활용하여 기존의 대형 모델(7 billion 또는 13 billion parameters)보다 효율적으로 작동합니다. QUITO는 두 개의 대표적인 데이터셋(NaturalQuestions, ASQA)에서 평가되었으며, 다양한 데이터셋과 downstream LLM에서 뛰어난 성능을 입증했습니다.

- **Performance Highlights**: 실험 결과, QUITO는 기존의 강력한 기준선 모델을 20.2%의 정확도 향상으로 능가했습니다. 이는 QUITO가 LLM의 계산 효율성과 정확도를 동시에 개선함을 보여줍니다.



### Clover-2: Accurate Inference for Regressive Lightweight Speculative Decoding (https://arxiv.org/abs/2408.00264)
- **What's New**: 다양한 NLP 연구에서 새로운 자동 평가 메트릭과 반사실적 데이터 증강 및 강력한 텍스트 생성 모델에 대한 최신 발전이 제안되었습니다.

- **Technical Details**: [{'Title': 'MCQ 자동 생성 평가를 위한 지식 종속 가능성(KDA)', 'Content': '기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 평가 메트릭의 한계를 극복하기 위해, 새로운 평가 메트릭인 KDA를 제안. KDA는 MCQ의 대답 가능성을 측정하며, 대상 사실에 대한 학생의 지식 평가 능력을 분석.'}, {'Title': 'NLP 태스크의 강건성을 위한 반사실적 학습', 'Content': '최근 deep model들이 spurious pattern에 의존하는 문제 때문에 robustness가 제한됨. 이 논문에서는 대조 학습과 반사실적 데이터 증강을 통해 모델의 강건성을 높이는 방법을 제안. 다양한 태스크와 데이터 도메인에서 성능 향상 확인.'}, {'Title': '효율적인 텍스트 생성 모델, Clover-2', 'Content': 'Clover의 업그레이드 버전인 Clover-2는 RNN 기반 초안 모델의 성능을 높이기 위해 지식 증류(knowledge distillation)와 개선된 모델 구조를 사용. Vicuan 7B와 LLaMA3-Instruct 8B 모델 테스트에서 기존 방법들보다 큰 성능 향상을 보임.'}]

- **Performance Highlights**: [{'Title': 'MCQ 자동 생성의 KDA 혁신', 'Content': 'human evaluation에서 KDA_disc와 KDA_cont가 높은 상관관계를 나타내고, n-gram 기반 메트릭과 결합 시 더 강력한 예측 성능을 가짐.'}, {'Title': '강건한 NLP 모델', 'Content': '다양한 데이터 도메인에서 counterfactual을 통해 유의미한 성능 향상; 1) 반사실적 강건성, 2) 크로스 도메인 일반화, 3) 데이터 부족 상황에서의 일반화.'}, {'Title': 'Clover-2의 뛰어난 성능', 'Content': 'Vicuan 7B와 LLaMA3-Instruct 8B 실험에서 standard decoding 대비 최대 3.00배의 처리 속도 향상. EAGLE보다 최대 9.3% 빠른 속도로 speculative token 생성, 특히 RNN 구조를 유지하면서도 효율성 증대.'}]



### Enhanced Structured State Space Models via Grouped FIR Filtering and Attention Sink Mechanisms (https://arxiv.org/abs/2408.00244)
- **What's New**: 최근 연구들은 자동 다중 선택 문제(MCQ)의 생성에 있어 교사들의 평가 시간을 현저히 줄일 수 있는 가능성을 보여주고 있습니다. 그러나 현재의 평가 메트릭들은 생성된 MCQ의 교육적 가치를 충분히 반영하지 못하고 있습니다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 '지식 종속 가능성'(Knowledge Dependent Answerability, KDA)이 제안되었습니다. KDA는 학생들의 응답을 통해 MCQ의 대답 가능성을 측정하며, 두 개의 자동 평가 메트릭(KDA_disc와 KDA_cont)을 도입해 학생들의 문제 해결 행동을 모방하도록 합니다.

- **Technical Details**: KDA는 학생들이 특정 사실에 대한 지식을 기반으로 MCQ의 대답 가능성을 평가합니다. 이를 위해 두 개의 자동 평가 메트릭, KDA_disc 및 KDA_cont가 도입되었습니다. 이들은 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. 또한 이러한 메트릭들은 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합될 때 다양한 전문가가 라벨링한 MCQ 품질 측정에 대해 강력한 예측력을 가집니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 KDA와 강한 상관관계를 가짐과 동시에 실제 강의실 세트에서도 유용성이 확인되었습니다. 두 메트릭은 n-gram 기반 유사성 메트릭과 결합될 때 전문가가 라벨링한 MCQ 품질 측정의 예측력에 있어 높은 성과를 보였습니다.



### Sentence-wise Speech Summarization: Task, Datasets, and End-to-End Modeling with LM Knowledge Distillation (https://arxiv.org/abs/2408.00205)
Comments:
          Accepted to Interspeech2024. Dataset: this https URL

- **What's New**: 새로운 MCQ 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하고, 최근 자연어 처리 태스크에서의 deep models의 robustness 한계 문제를 contrastive learning과 counterfactual augmentation로 해결하는 방안을 제시했으며, 문장별 음성 요약을 위한 새로운 접근 방식인 Sen-SSum을 소개했습니다.

- **Technical Details**: [{'MCQ Generation': '기존 BLEU, ROUGE, METEOR 메트릭은 유사 단어만 비교하여 교육적 가치를 평가하지 못합니다. 제안한 KDA는 학생의 대상 사실 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. 자동 평가 메트릭 KDA_disc와 KDA_cont가 강한 상관관계를 보여, 다른 n-gram 기반 유사성 메트릭과 결합하여 예측력을 향상시킵니다.'}, {'NLP Robustness': 'deep model들이 spurious pattern에 의존하여 robustness가 제한된다는 문제를 보고, 여러 개의 counterfactual을 합성하고 집합적 의사 결정을 통해 more robust하게 인과관계를 파악하는 방법을 제안했습니다. 이 방법은 counterfactual robustness, cross-domain generalization, and scarce data generalization에 큰 개선을 보였습니다.'}, {'Sen-SSum': 'Sen-SSum은 자동 음성 인식(ASR)과 음성 요약의 실시간 처리를 결합하여 문장별로 요약을 생성합니다. Mega-SSum과 CSJ-SSum 두 개의 데이터셋으로 Transformer-based 모델을 평가했습니다. Cascade 모델은 ASR과 텍스트 요약(TSum) 모델을 결합한 방식이고, End-to-End(E2E) 모델은 직접 음성에서 텍스트 요약을 생성하는 방식입니다. E2E 모델 성능 향상을 위해 지식 증류(knowledge distillation)를 제안했습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'human studies에서 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가졌으며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 레이블링한 MCQ 품질 측정치에 대해 높은 예측력을 보였습니다.'}, {'NLP Robustness': '우리의 접근법은 집합적 의사 결정을 통해 더 적은 모델 바이어스에 민감하게 반응해, 다양한 차원에서 유의한 성능 향상을 달성하였습니다.'}, {'Sen-SSum': '제안된 지식 증류 방법이 E2E 모델의 성능을 Mega-SSum과 CSJ-SSum 데이터셋 모두에서 효과적으로 향상시켰습니다. 특정 조건에서 pseudo-summaries가 수동 요약보다 더 나은 요약 정확도를 보였습니다.'}]



### Automatic Generation of Behavioral Test Cases For Natural Language Processing Using Clustering and Prompting (https://arxiv.org/abs/2408.00161)
- **What's New**: 최근의 딥 러닝 모델들이 사람보다 높은 정확성을 보이는 NLP 태스크에서 롭스터스 신뢰도(robustness) 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적인 증강(counterfactual augmentation)을 활용. 기존 평가 메트릭은 n-그램 기반 유사성에 집중하여 교육적 가치를 평가하지 못함. 우리는 MCQ 생성에서 학생의 지식 평가 능력을 측정하는 새로운 메트릭을 제안.

- **Technical Details**: 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 사용하여 MCQ 대답 가능성을 측정. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 흉내내는 방식으로 KDA를 근사. 또한, 반사실적 데이터를 '여러 개' 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어 인과 관계를 robust하게 학습. 대규모 언어 모델을 사용하여 의미 있는 그룹을 만들고, MFT(Minimal Functionality Tests)를 자동 생성.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 사용성과 강한 상관관계를 보임. 집합적 의사 결정을 통해 다양한 측면에서 높은 성능을 보였으며, 반사실적 robust성, 크로스 도메인 일반화, 적은 데이터에서의 일반화 능력에서 큰 개선을 달성. Amazon 리뷰 코퍼스를 사용하여 자동 테스트 케이스 생성 방법을 시연하고, 네 가지 분류 알고리즘의 행동 테스트 프로필을 분석.



### Distributed In-Context Learning under Non-IID Among Clients (https://arxiv.org/abs/2408.00144)
Comments:
          12 pages

- **What’s New**: 이 논문은 새로운 자동 평가 메트릭인 '지식 종속 가능성' (Knowledge Dependent Answerability, KDA)을 제안합니다. 기존의 MCQ 생성 평가 메트릭(BLEU, ROUGE, METEOR)이 교육적 가치를 무시하는 문제를 해결하기 위해 설계되었습니다. 또한, 대규모 언어 모델의 적응 문제를 다루며, 대리 데이터셋을 사용한 분산형 non-IID ICL 문제 해결 방법도 제시합니다.

- **Technical Details**: KDA는 학생의 답변 가능성을 기준으로 MCQ의 교육적 가치를 측정합니다. 이를 위해, 인간 설문조사를 통해 학생 응답 데이터를 사용해 KDA를 측정하는 방법을 제시하고, pretrained language models를 활용해 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 또한, in-context learning (ICL)을 분산형 non-IID 데이터 상황에 맞게 개선하는 방법을 논의하며, 서버가 클라이언트별로 다르게 ICE 예산을 할당하는 새로운 프레임워크를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가와 전문가가 선정한 범주에서 강한 상관관계를 보여줍니다. 또한, 논문의 접근법이 부정적인 패턴(spurious patterns)에 덜 민감하며, counterfactual robustness, cross-domain generalization 및 scarce data의 일반화에서 유의미한 성과를 얻었습니다. 분산형 non-IID ICL 상황에서도 다양한 데이터셋에서 경쟁 방법들보다 우수한 성능을 입증했습니다.



### Correcting Negative Bias in Large Language Models through Negative Attention Score Alignmen (https://arxiv.org/abs/2408.00137)
- **What's New**: 본 연구는 자동으로 다지 선다형 질문(MCQ)을 생성할 때 사용할 수 있는 새로운 평가 메트릭인 지식 종속 가능성(KDA; Knowledge Dependent Answerability)을 제안합니다. 이는 기존의 BLEU, ROUGE, METEOR와 달리 생성된 MCQ가 학생의 지식 평가 능력을 측정하는데 초점을 맞추고 있습니다.

- **Technical Details**: 지식 종속 가능성(KDA)은 '목표 사실(target fact)'에 대한 학생의 지식을 바탕으로 MCQ의 답변 가능성을 평가합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용해 학생들의 문제 해결 행위를 시뮬레이션함으로써 KDA를 자동으로 측정하게 됩니다. 이 과정에서 인간의 응답을 통해 KDA를 평가하는 방법을 먼저 제시하고, 이를 바탕으로 두 개의 자동 평가 메트릭을 개발했습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합될 때, KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측력을 보여줍니다.



### A Course Shared Task on Evaluating LLM Output for Clinical Questions (https://arxiv.org/abs/2408.00122)
Comments:
          accepted at the sixth Workshop on Teaching NLP (co-located with ACL 2024)

- **What's New**: 이 논문은 새로운 자동 평가 메트릭, 지식 종속 가능성(KDA: Knowledge Dependent Answerability)을 제안하여 MCQ(선택형 문제)의 교육적 가치를 측정합니다. 또한, 최근의 딥 러닝 모델들이 NLP 작업에서 초인적인 정확성을 보였음에도 불구하고, 오히려 허위 패턴(spurious patterns)에 의존하여 강건성(robustness)이 제한되는 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 확장(counterfactual augmentation)을 활용한 방안을 소개합니다. 마지막으로, 독일 다름슈타트 공과대학에서 2023/2024 학년도 Foundations of Language Technology(FoLT) 강의에서 학생들에게 LLM(대형 언어 모델)의 유해한 답변을 평가하는 과제를 시행했음을 보고합니다.

- **Technical Details**: 첫 번째 논문에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행위를 모방합니다. 두 번째 논문에서는 '여러 개의' 반사실적(counterfactual) 예시들을 생성하고, 집합적인 의사결정을 통해 모델의 인과 관계를 더욱 강건하게 판단하는 방식을 제안합니다. 마지막 논문에서 LLM의 출력물이 건강 관련 임상 질문에 대해 유해한 정보를 포함하는지를 검사하는 과제를 설계하고, 이를 구현하고 평가합니다.

- **Performance Highlights**: 첫 번째 논문에서는 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서 사용 가능성과 강한 상관관계를 가지며, 기존의 n-gram 기반 유사성 메트릭과 결합할 때 높은 예측력을 발휘한다고 보고합니다. 두 번째 논문에서는 제안된 방법이 반사실적 강건성, 도메인 간 일반화, 부족한 데이터로부터의 일반화 측면에서 큰 개선을 이루었다고 강조합니다. 마지막으로, 임상 관련 과제에서는 500개 이상의 CCAs를 이용해 학생들이 데이터 주석, 모델 개발 및 평가 작업을 수행하며, 참여 학생들로부터 다양한 피드백을 수렴하고 이를 바탕으로 과제의 개선 방안을 논의합니다.



### Gemma 2: Improving Open Language Models at a Practical Siz (https://arxiv.org/abs/2408.00118)
- **What's New**: 자동 다중 선택 질문(MCQ)의 평가를 위해 새로운 자동 평가 메트릭인 지식 종속 대답 가능성(KDA)을 제안했으며, 이는 학생이 특정 사실에 대해 가진 지식을 평가할 수 있는 MCQ의 대답 가능성을 측정하는 것을 목표로 합니다. 또한, 깊은 학습 모델의 강건성 문제를 해결하기 위해 대조 학습과 반사실적 증강(contrastive learning and counterfactual augmentation)을 활용하는 새로운 방법이 제안되었습니다. 마지막으로, 새로운 경량 언어 모델인 Gemma 2를 소개하며, 이는 2억에서 270억 매개변수까지의 규모로 제공됩니다.

- **Technical Details**: 기존 평가 메트릭들은 MCQ의 교육적 가치를 반영하지 못하는 문제가 있었으며, 이에 대응하기 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭들이 제안되었습니다. 이는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 대조 학습과 반사실적 증강 방법을 통해 여러 개의 반사실적(synthetic) 데이터를 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계(causality)를 더 강건하게 평가합니다. Gemma 2는 Transformer 아키텍처에서 로컬 및 글로벌 주의층(local-global attentions)과 그룹-질의 주의(group-query attention) 메커니즘을 적용받았으며, 지식 증류(knowledge distillation)를 통해 훈련되었습니다.

- **Performance Highlights**: Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실(set)에서의 사용성 측면에서 강한 상관관계를 가짐을 입증했습니다. 제안된 대조 학습 및 반사실적 증강 접근 방식은 다양한 차원에서 강건성, 도메인 간 일반화, 희소한 데이터로부터의 일반화에 있어서 큰 개선을 이루었음을 경험적으로 보여주었습니다. Gemma 2 모델은 여러 자동화 및 인간 평가에서 경쟁력 있는 성능을 보여주었고, 특정 질문 응답, 상식 추론, 수학 및 과학, 코딩 등의 도메인에서 두드러진 성능을 보였습니다.



### ReLiK: Retrieve and LinK, Fast and Accurate Entity Linking and Relation Extraction on an Academic Budg (https://arxiv.org/abs/2408.00103)
Comments:
          To be presented at ACL 2024

- **What's New**: 새롭게 제안된 논문에서는 자동 MCQ 생성 평가를 위한 지식 종속 가능성(KDA) 메트릭을 도입했습니다. 이는 기존의 n-gram 기반 유사성을 넘어 학생의 목표 사실에 대한 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: 우리는 먼저 human survey 데이터를 통해 KDA를 측정하고, 이를 근거로 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 통해 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 지표에 대한 예측력이 높음을 보여주었습니다.



### MM-Vet v2: A Challenging Benchmark to Evaluate Large Multimodal Models for Integrated Capabilities (https://arxiv.org/abs/2408.00765)
Comments:
          Extension of MM-Vet: arXiv:2308.02490

- **What's New**: 이 논문에서는 기존의 MM-Vet의 단점을 보완하여 새로운 평가 메트릭인 MM-Vet v2를 소개합니다. MM-Vet v2는 '이미지-텍스트 시퀀스 이해'라는 새로운 비전-언어(Vision-Language) 능력을 평가하여 모델이 이미지-텍스트 시퀀스를 처리하는 능력을 포함합니다.

- **Technical Details**: MM-Vet v2는 기존 MM-Vet의 여섯 가지 핵심 VL 능력 (Recognition, Knowledge, Spatial Awareness, Language Generation, OCR, Math) 외에 추가로 '이미지-텍스트 시퀀스 이해'를 포함합니다. 이를 위해 연구팀은 다양한 시나리오에서 수집한 517개의 질문을 설계하고, 긴 답변을 필요로 하는 질문은 GPT-4V를 사용해 초안을 작성한 뒤 전문가가 검토합니다.

- **Performance Highlights**: MM-Vet v2로 다양한 대형 멀티모달 모델(LMMs)을 평가한 결과, Claude 3.5 Sonnet이 71.8점을 기록하며 가장 높은 성능을 보였습니다. GPT-4o는 71.0점을 기록했고, 오픈 소스 모델 중에서는 InternVL2-Llama3-76B가 68.4점으로 가장 뛰어난 성능을 나타냈습니다.



### Tamper-Resistant Safeguards for Open-Weight LLMs (https://arxiv.org/abs/2408.00761)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 자동으로 생성된 객관식 질문(MCQ)의 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안합니다. 또한, 대규모 언어 모델(LLM)의 악성 사용을 방지하기 위한 새로운 보호 기법, Tamper-Resistant Safeguards (TAR)을 도입했습니다.

- **Technical Details**: KDA 메트릭은 학생들의 실제 응답 데이터를 활용해 자동으로 MCQ의 대답 가능성을 평가합니다. 이를 통해 BLEU, ROUGE, METEOR와 같은 기존의 n-그램 기반 메트릭들이 간과하는 교육적 가치를 파악할 수 있습니다. TAR 기법은 모델 가중치를 조작하는 공격에도 견딜 수 있는 안전 장치를 언어 모델에 포함시키는 방법입니다. 이 방법은 adversarial training을 활용해 모델 실패를 방지합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 메트릭과 결합하여 전문가들이 라벨링한 다양한 MCQ 품질 측정에 대해 높은 예측력을 보여줍니다. 또한, TAR 기법은 5,000 스텝 이상의 파인 튜닝 공격에도 견딜 수 있는 강력한 저항성을 보였습니다. 이를 통해 LLM의 안전성과 보안성을 크게 개선할 수 있습니다.



### SentenceVAE: Faster, Longer and More Accurate Inference with Next-sentence Prediction for Large Language Models (https://arxiv.org/abs/2408.00655)
Comments:
          First preview version

- **What's New**: {'Paper1': '이번 연구에서는 자동 MCQ 생성의 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다. 이 메트릭은 학생들이 특정 지식을 바탕으로 MCQ에 답할 수 있는 능력을 측정합니다.', 'Paper2': 'NLP 작업에서 깊은 모델의 강인성(robustness) 향상을 위해 대조학습(contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용하는 새로운 접근법을 제안합니다. 여러 반사실적 사례를 생성하고 집합적 의사 결정(collective decisions)을 통해 강인성을 확보합니다.', 'Paper3': '대규모 언어 모델(LLMs)의 추론 속도를 높이기 위해 새로운 SentenceVAE 모델을 도입하여 다음 문장 예측 방법을 제안합니다. 이는 LLM의 입력과 출력 레이어에 SentenceVAE를 통합하여 문장 단위 추론 방식을 사용합니다.'}

- **Technical Details**: {'Paper1': 'KDA를 측정하기 위해 학생 응답을 바탕으로 한 인간 설문을 활용하는 방법을 제안하고, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안합니다. 이들은 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방합니다.', 'Paper2': '기존의 반사실적 증강 방식은 인간이 직접 생성하거나 데이터셋에서 반사실적 사례를 찾아내는 방식이었습니다. 제안된 방법은 여러 반사실적 사례를 합성하고, 이를 기반으로 한 집합적 의사 결정으로 각 용어의 인과성을 평가합니다.', 'Paper3': 'SentenceVAE는 인코더와 디코더로 구성되어 있으며, 문장의 정보를 하나의 토큰으로 압축하고 이를 다시 문장 형태로 복원합니다. LLM의 추론 과정에서 자가 주의 계산의 메모리 요구를 줄이고, 같은 문맥 길이에서도 메모리 오버헤드를 줄입니다.'}

- **Performance Highlights**: {'Paper1': 'KDA_disc와 KDA_cont는 인간 설문을 통한 실험에서 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가 레이블 MCQ 품질 측정에 대한 예측력을 보여주었습니다.', 'Paper2': '제안된 방법은 반사실적 강인성, 도메인 간 일반화, 그리고 희소 데이터에서의 일반화 등 여러 측면에서 성능 향상을 보여주었습니다.', 'Paper3': 'SentenceVAE를 이용한 새로운 추론 방법은 추론 속도를 204~365% 증가시키고, 동일한 문맥 길이에 대해 메모리 오버헤드를 86~91% 감소시켰습니다. 모델 규모가 커질수록 이점이 더욱 증대되었습니다.'}



### SynesLM: A Unified Approach for Audio-visual Speech Recognition and Translation via Language Model and Synthetic Data (https://arxiv.org/abs/2408.00624)
- **What's New**: 자동 MCQ 생성의 평가 문제를 해결하기 위해 새로운 지식 종속 대답 가능성(KDA) 메트릭을 제안했습니다. 기존에는 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 주로 사용되었지만, 이는 교육적 가치를 평가하지 못했습니다. 또한, NLP 태스크에서 모델의 강인함을 향상시키기 위해 대조 학습과 반사실적 증가(counterfactual augmentation) 방법을 제안했습니다. 마지막으로 SynesLM이라는 모델을 통해 오디오-비주얼 자동 음성 인식(AV-ASR), 비주얼 보조 음성/기계 번역(VST/VMT) 등 다양한 멀티모달 언어 이해 태스크를 수행할 수 있는 통합 모델을 소개했습니다.

- **Technical Details**: MCQ 생성 평가를 위해 KDA라는 새로운 메트릭을 도입하였으며, 이는 학생의 반응을 기반으로 대답 가능성을 측정합니다. 대조 학습과 반사실적 증가 방법을 통해 강인함이 향상된 모델을 설계했습니다. SynesLM 모델은 전체 프레임의 시각 정보를 활용하고, 시각-언어 연결 레이어를 통해 음성 및 텍스트 토큰을 동일한 임베딩 공간에 맵핑합니다. 이를 통해 시각적 정보와 텍스트 정보 사이의 통합을 가능케 했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 활용도와 강한 상관관계를 가지며, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 예측력이 높다는 것을 보여주었습니다. 대조 학습을 통해 모델이 다양한 차원에서 강인함, 크로스 도메인 일반화 및 희소한 데이터로부터의 일반화에서 현저한 개선을 보였습니다. SynesLM 모델은 VisSpeech 데이터셋에서 zero-shot AV-ASR의 Word Error Rate (WER)을 43.4%에서 39.4%로 낮추며 SOTA 성능을 달성했고, VST에서는 BLEU 점수가 37.2에서 43.5로, VMT에서는 54.4에서 54.8로 향상되었습니다.



### Are Bigger Encoders Always Better in Vision Large Models? (https://arxiv.org/abs/2408.00620)
- **What's New**: 이번 연구에서는 기존의 자동 MCQ(Multiple Choice Questions) 생성 평가 메트릭의 한계를 극복하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안합니다. 또한, 새로운 KDA_disc와 KDA_cont 메트릭을 통해 실제 강의실 세트에서의 사용성과 교육적 가치를 평가합니다.

- **Technical Details**: BLEU, ROUGE, METEOR과 같은 기존의 MCQ 평가 메트릭은 단순히 n-gram 유사성에 의존하여 교육적 가치를 충분히 반영하지 못합니다. 반면, KDA는 학생의 지식 기반으로 MCQ의 답변 가능성을 평가하며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 통해 이를 자동화하려는 접근 방식입니다.

- **Performance Highlights**: Human evaluation 연구를 통해 KDA_disc와 KDA_cont 메트릭이 실제 전문가들이 평가한 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 이를 통해 n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가가 평가한 MCQ 품질 측정치에 대한 예측력이 높음을 확인하였습니다.



### Alleviating Hallucination in Large Vision-Language Models with Active Retrieval Augmentation (https://arxiv.org/abs/2408.00555)
- **What's New**: 최근 논문에서 여러 가지 혁신적인 새로운 평가 및 학습 방법론을 소개했습니다. 첫 번째로, 자동화된 다지 선다형 질문(MCQ) 생성의 평가를 위해 기존 BLEU, ROUGE, METEOR와 같은 메트릭이 교육적 가치를 무시한다는 문제를 해결하기 위해 새로운 지식 종속 가능성(KDA) 메트릭을 제안했습니다. 두 번째로, NLP 태스크에서의 deep model의 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용했습니다. 마지막으로, 대규모 비전-언어 모델(LVLMs)의 환각현상(hallucination)을 줄이기 위해 외부 지식 자원을 통한 액티브 리트리벌-보강 모델(ARA)을 제안했습니다.

- **Technical Details**: 지식 종속 가능성(KDA)를 통해 자동 MCQ의 대답 가능성을 평가했습니다. KDA를 불러오는 과정에서는 인간의 설문 응답을 기초로 합니다. 이후 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 개발해 미리 학습된 언어 모델을 통해 학생들의 문제해결 행동을 모방했습니다. NLP 퍼포먼스 개선을 위해서는 counterfactual을 생성하여 튜닝된 모델의 스푸리어스 패턴을 줄였습니다. 액티브 리트리벌-보강 모델에서는 LVLMs의 환각현상을 줄이기 위해 (i) 이미지 계층 구조를 분해하고 (ii) 효과적인 리트리벌 방법을 선정하며 (iii) 리트리벌 시간을 조정하는 세 가지 주요 차원을 고려한 접근을 취했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 세팅과 강한 상관관계를 보였으며, 전문가가 표기한 MCQ 품질 지표와도 높은 예측력을 보였습니다. 새로운 대조 학습 방법은 이종 도메인 일반화 및 드문 데이터에서의 일반화 측면에서 크게 향상된 성능을 달성했습니다. ARA 모델은 세 가지 널리 사용되는 LVLM(LLaVA-1.5, Qwen-VL, mPLUG-Owl2)을 사용하여 네 가지 벤치마크에서 환각현상을 효과적으로 줄이는 성과를 보였습니다.



### Mitigating Multilingual Hallucination in Large Vision-Language Models (https://arxiv.org/abs/2408.00550)
- **What's New**: 최근 자동 생성 다지선다형 문제(MCQ) 평가에 있어서 교육적 가치를 측정하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)가 제안되었습니다. 이와 더불어, 대조 학습과 반사실적 증강을 활용한 NLP 딥 모델의 강건성 향상 방법과 LVLM(대형 비전-언어 모델)의 다중언어 환각 문제를 완화하는 새로운 프레임워크가 소개되었습니다.

- **Technical Details**: KDA는 학생의 지식 기반으로 MCQ의 답변 가능성을 평가합니다. 대조 학습 및 반사실적 증강을 통한 NLP 모델의 반사실적 강건성과 도메인 간 일반화를 목표로 합니다. 다중언어 환각 제거(MHR) 프레임워크는 다양한 언어에서의 환각을 줄이기 위해 다단계 접합 방법과 직접 선호 최적화를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭이 실제 강의실 세트에서의 사용성과 높은 상관성을 보였습니다. LVLMs의 MHR 프레임워크는 POPE 벤치마크에서 평균 19.0%의 정확도 향상을 달성했습니다. 다양한 언어에서의 강건성과 일반화 능력도 현저히 개선되었습니다.



### The Monetisation of Toxicity: Analysing YouTube Content Creators and Controversy-Driven Engagemen (https://arxiv.org/abs/2408.00534)
Comments:
          Accept for publication at the 4th International Workshop on Open Challenges in Online Social Networks (OASIS) held in conjunction with 35th ACM Conference on Hypertext and Social Media (HT24)

- **What's New**: 최근 MCQ 생성의 평가 메트릭으로 기존 BLEU, ROUGE, METEOR와 같은 n-gram 기반 메트릭이 교육적 가치를 반영하지 못한다는 문제를 해결하기 위해 'Knowledge Dependent Answerability (KDA)'를 제안했습니다. 이는 학생의 지식 기반으로 MCQ의 대답 가능성을 측정합니다. 또한 최근 딥러닝 모델의 한계로 보고된 spurious pattern 의존성 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 새 접근법이 제안되었습니다. 마지막으로, YouTube의 논란 콘텐츠와 독성(toxicity), 수익화(monetisation)의 관계를 분석한 연구가 발표되었습니다.

- **Technical Details**: [{'Original Paper': 'MCQ 생성 평가에 대한 새로운 메트릭인 KDA는 목표 사실(target fact)에 대한 학생의 지식을 기반으로 MCQ의 대답 가능성을 평가합니다. KDA는 학생의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생들이 문제를 푸는 행동을 모방합니다.', 'New Approach': '대조 학습과 반사실적 증강을 활용해 스퍼리어스 패턴(spurious pattern)에 덜 민감한 모델을 제안하며, 이는 다양한 차원에서 상당한 성능 향상을 달성합니다.'}, {'Original Paper': 'YouTube 논란 콘텐츠 연구에서는 Reddit의 논란이 되는 YouTube 채널 20개를 분석하여 논란, 독성 및 수익화 간의 관계를 조사했습니다. 16,349개의 비디오와 1억 500만 개 이상의 댓글을 포함하는 데이터셋을 구축했으며, 머신 러닝 모델을 훈련하여 댓글 독성을 측정했습니다.'}]

- **Performance Highlights**: [{'Paper': 'MCQ 생성 평가 연구에서는 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가짐을 확인했습니다. 또한 전문가가 평가한 다양한 MCQ 질 평가 측정에 대해 n-gram 기반 유사성 메트릭과 결합하여 강한 예측력을 가짐을 보여주었습니다.'}, {'Paper': '대조 학습과 반사실적 증강을 위한 새로운 접근법은 1) 반사실적 견고성, 2) 도메인 간 일반화, 3) 희소한 데이터에서의 일반화에서 상당한 성능 향상을 보였습니다.'}, {'Paper': 'YouTube 연구에서는 논란이 높은 댓글이 참여도를 높일 수 있지만 수익화에는 부정적인 영향을 미친다는 것을 발견했습니다. 일부 채널은 높은 독성에도 불구하고 수익화 전략에서 더 성공적임을 보여주었습니다.'}]



### ABC Align: Large Language Model Alignment for Safety & Accuracy (https://arxiv.org/abs/2408.00307)
Comments:
          23 pages, 4 figures

- **What's New**: 이번 주 Arxiv 논문에서는 대규모 언어 모델(LLMs)의 맞춤화에 대한 새로운 방법론인 'ABC Align'을 소개했습니다. ABC Align은 LLMs에 대한 인공지능 원칙과 조직의 선호도를 통합하는 새로운 정렬 방법론입니다. 이를 통해 모델의 편향을 줄이고 정확성을 높이며, 표준 벤치마크 대비 성능을 개선합니다.

- **Technical Details**: ABC Align은 기존의 모델 미세조정(fine-tuning) 및 내부 문맥 학습(In-Context Learning, ICL) 기술을 결합하여 수행됩니다. 최근의 합성 데이터 생성(synthetic data generation), 선호 최적화(preference optimization), 그리고 훈련 후 모델 양자화(post-training model quantisation) 기술을 활용하여 다수의 오픈 소스 모델과 폐쇄 소스 모델에서 성능 상승을 보여줍니다. 다양한 데이터 세트—뉴스 기사, AI 원칙(ABC AI Principles), 그리고 내부 검색-강화 생성 도구(RAG)에서 제공된 Q&A 쌍—를 사용해 데이터 세트를 생성하고 모델을 정렬합니다.

- **Performance Highlights**: 대표적인 성과는 Meta의 Llama3-8B 모델에서 TruthfulQA 벤치마크에서 23.51%의 상대적 성능 향상을 보였다는 것입니다. 또한 OpenAI의 GPT4-turbo 모델에서는 Bias Benchmark for Question Answering (BBQ)에서 기본 라인 대비 성능이 77.54% 향상되었습니다. 이는 소규모 데이터 세트(897 샘플)을 사용하여 달성한 결과로, 메타에서 자체적으로 조정한 버전보다 훨씬 나은 성능을 보였습니다.



### Lost in Translation: Latent Concept Misalignment in Text-to-Image Diffusion Models (https://arxiv.org/abs/2408.00230)
Comments:
          33 pages, 19 figures

- **What's New**: MCQ 자동 생성의 한계를 보완하고 교육적 가치를 높이는 새로운 자동 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안했습니다. 이는 학생이 질문에 대답할 수 있는 능력을 측정하여 교육적 유효성을 평가합니다. 또한, 텍스트에서 이미지로 변환하는 확산 모델의 개념 불일치 문제(Latent Concept Misalignment, LC-Mis)를 조사하고 해결하기 위한 새로운 자동화된 파이프라인을 개발했습니다.

- **Technical Details**: MCQ의 교육적 가치를 평가하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. LC-Mis 문제를 해결하기 위해, LLM(대형 언어 모델)을 사용하여 개념의 일치성을 강화하는 파이프라인을 설계했습니다. 이 파이프라인은 개념 쌍을 분할하여 두 단계로 이미지 생성을 수행합니다.

- **Performance Highlights**: 연구 결과, KDA_disc와 KDA_cont가 실제 강의실 환경에서 높은 사용성과 강한 상관관계를 나타냈습니다. LC-Mis 문제 해결을 위한 새로운 접근법은 기존 모델보다 텍스트와 이미지의 일치를 크게 개선했으며, 다양한 분야에서의 적용성과 유연성을 증대시키는 데 기여했습니다.



### OmniParser for Pure Vision Based GUI Agen (https://arxiv.org/abs/2408.00203)
- **What's New**: 이 논문에서는 자동 MCQ(객관식 질문) 생성의 교육적 가치를 평가하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)을 제안합니다. 또한, GPT-4V와 같은 대형 비전-언어 모델이 다양한 사용자 인터페이스에서 동작하는 것을 돕기 위해 OmniParser를 도입하여 화면 상의 상호작용 가능한 요소와 그 기능적 의미를 보다 정확하게 이해하도록 합니다.

- **Technical Details**: [{'MCQ Generation': '기존 MCQ 생성의 평가 메트릭들은 BLEU, ROUGE, METEOR처럼 n-gram 기반의 유사성에만 집중되어 있었으나, 교육적 가치를 평가하지 않았습니다. 이를 해결하기 위해, KDA라는 새로운 메트릭을 통해 학생이 해당 정보를 알고 있을 때 MCQ가 답할 수 있는지 여부를 평가합니다. KDA는 Human survey를 기반으로 측정되며, KDA_disc와 KDA_cont라는 자동화된 평가 메트릭을 제안합니다.'}, {'Model Robustness': '최근의 deep model들이 NLP 태스크에서 높은 정확성을 보였으나, spurious pattern에 의존하여 robustness가 제한되었습니다. 이 논문에서는 contrastive learning과 counterfactual augmentation을 활용하여 단어들의 인과관계를 판단하는 방법을 제안합니다. 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 이전의 spurious correlation 문제를 극복합니다.'}, {'Screen Parsing': 'OmniParser는 사용자의 인터페이스 스크린샷을 파싱하여 구조화된 요소들로 변환하는 종합적인 방법입니다. 이를 통해 GPT-4V가 다양한 애플리케이션에서 예측한 행동을 정확한 화면 영역에 결합할 수 있습니다. 이 시스템은 상호작용 가능한 요소를 탐지하고 기능적 의미를 추출하기 위해 특별히 학습된 모델을 활용합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 함께 사용할 때, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였습니다.'}, {'Model Robustness': '제안된 방법은 단어들의 인과관계 파악에서 bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, and generalization from scarce data에서 놀라운 성능 향상을 보였습니다.'}, {'Screen Parsing': 'OmniParser는 ScreenSpot, Mind2Web, AITW 벤치마크에서 기존 GPT-4V 기반 모델을 능가했습니다. 특히 추가 정보 없이 스크린샷만으로도 뛰어난 성능을 보여주었습니다.'}]



### Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models (https://arxiv.org/abs/2408.00197)
- **What's New**: 이번 연구에서는 자동 객관식 문제(MCQ) 생성을 위한 새로운 평가 메트릭, KDA(Knowledge Dependent Answerability)를 제안했습니다. 이는 기존 BLEU, ROUGE, METEOR 등의 n-gram 기반 메트릭이 학습 평가의 교육적 가치를 고려하지 못하는 문제를 해결하기 위한 것입니다.

- **Technical Details**: KDA는 목표 사실에 대한 학생의 지식을 바탕으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 학생 응답 기반의 KDA 측정 방법을 먼저 제안하고, 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안했습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 (1) 실제 강의실 환경에서의 사용성, (2) 전문가가 라벨링한 MCQ 품질 측정치와 강한 상관관계가 있음을 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합하여 KDA_disc와 KDA_cont는 다양한 전문가 라벨링된 MCQ 품질 측정치에 대한 예측력이 뛰어났습니다.



### A Taxonomy of Stereotype Content in Large Language Models (https://arxiv.org/abs/2408.00162)
- **What's New**: 우리는 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'의 도입을 통해 MCQ의 교육적 평가를 강화하려고 합니다. 또한, 딥러닝 모델의 강인성을 높이기 위해 대조 학습 (contrastive learning)과 반사실적 증대(counterfactual augmentation) 기법을 활용하는 연구도 소개합니다. 마지막으로, 최신 대형 언어 모델 (LLM)에서의 고정관념 내용을 분류하는 새로운 방식의 연구를 발표합니다.

- **Technical Details**: 첫 번째 연구에서는 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭의 한계를 극복하기 위해 KDA를 제안하고, 학생의 문제 해결 행동을 사전 학습된 언어 모델이 모방하게 하여 KDA_disc와 KDA_cont 자동 평가 메트릭을 개발했습니다. 두 번째 연구에서는 '집합적 의사 결정 (collective decisions)'을 통해 단어들의 인과관계를 파악하는 방법을 제안하고, counterfactual을 '여러 개' 생성하는 접근법을 사용했습니다. 세 번째 연구는 ChatGPT 3.5, Llama 3, Mixtral 8x7B 모델에서 87개의 사회 카테고리를 기반으로 한 고정관념 차원을 분류하고, 이를 통해 모델의 내부 평가를 예측하는 다차원 분류법을 제안했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 강한 상관관계를 보였으며, MCQ 품질 평가에서도 예측 능력을 높였습니다. 두 번째 연구에서는 다양한 차원에서 강인성이 크게 개선되었으며, 특히 counterfactual robustness 및 cross-domain generalization에서 두각을 나타냈습니다. 세 번째 연구에서는 LLM이 인간보다 더 긍정적인 고정관념을 보이지만, 카테고리와 차원에 따라 큰 변동성을 보였습니다.



### Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models (https://arxiv.org/abs/2408.00113)
Comments:
          Oral paper (top 5%) at the ICML 2024 Mechanistic Interpretability Workshop

- **What's New**: 새로운 자동 MCQ 평가 메트릭인 지식 종속 가능성(KDA)이 제안되었습니다. 또한, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭이 소개되었습니다. 한편, NLP 태스크에서 강력한 정확성을 보이는 최신 deep models가 spurious patterns에 의존하는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용한 새로운 방법론이 제시되었습니다. 마지막으로, Sparse Autoencoders(SAEs)의 성능을 측정하는 새로운 기법 p-annealing이 제안되었습니다.

- **Technical Details**: ['MCQ 평가에서 BLEU, ROUGE, METEOR 같은 기존 메트릭은 교육적 가치를 고려하지 않기 때문에 KDA라는 새로운 메트릭이 제안되었습니다. KDA는 학생들의 반응을 토대로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다.', 'NLP 모델의 robustness를 높이기 위해 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 강화하는 방법이 제안되었습니다. 이 방법은 기존 spurious correlation 문제를 해결하고 여러 차원에서 성능을 향상시킵니다.', 'Sparse Autoencoders(SAEs)에서 interpretable features를 분리하기 위해 chess와 Othello 게임 transcript를 사용하는 새로운 SAE 평가 기법이 도입되었습니다. p-annealing이라는 새로운 SAE 훈련 기법도 소개되었습니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont 메트릭은 실제 강의실 세트에서 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 경우 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 높은 예측력을 보였습니다.', '새로운 counterfactual augmentation 접근법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등 다양한 차원에서 기존 방법을 능가하는 성능을 보여주었습니다.', 'p-annealing 기법을 사용한 SAEs는 기존의 비지도 학습 메트릭과 새로운 메트릭 모두에서 높은 성능을 기록하며, 이전 메소드와 비교해 개선된 결과를 보여주었습니다.']



### Towards a Universal Method for Meaningful Signal Detection (https://arxiv.org/abs/2408.00016)
- **What's New**: 자동 MCQ 생성 평가 방식에서 기존의 n-gram 기반 메트릭이 교육적 가치를 반영하지 않는 문제를 해결하기 위해 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안했습니다. 이 메트릭은 MCQ의 대답 가능성(answerability)을 측정하고 학생 지식을 평가하는 능력을 확인합니다.

- **Technical Details**: KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 이를 통해 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. Human evaluation을 통해 이 메트릭들이 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지는 것을 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 메트릭과 결합되었을 때 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측 능력을 보였습니다.



### Framework for Curating Speech Datasets and Evaluating ASR Systems: A Case Study for Polish (https://arxiv.org/abs/2408.00005)
Comments:
          Submitted to NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 자동으로 주관식 질문을 생성하지만, 현재의 평가 지표들은 교육적 가치를 반영하지 못하는 문제를 극복하기 위해 새로운 평가 메트릭을 제안했습니다. 또한, NLP 태스크에서 스퍼리어스 패턴에 의존하는 문제를 해결하기 위해 대조 학습과 counterfactual 증강 방법을 소개했습니다. 폴란드어 음성 인식 시스템의 평가를 위해 포괄적인 프레임워크를 마련하고, 다양한 ASR 시스템과 모델을 비교하는 사례 연구를 진행했습니다.

- **Technical Details**: 첫 번째 연구에서는 지식 종속 가능성(KDA)라는 새로운 평가 메트릭을 제안하여, 학생들이 대상 사실에 대한 지식으로 주관식 질문을 답할 수 있는지를 평가합니다. 두 번째 연구에서는 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 세 번째 연구에서는 24개의 공개 데이터셋을 큐레이팅하여 폴란드어 ASR 시스템 간 성능을 비교하는 프레임워크를 구축했습니다.

- **Performance Highlights**: 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 실제 강의실 사용성과 강한 상관관계를 보였습니다. 집합적 의사 결정을 통한 새로운 증강 방법으로 다양한 차원에서의 성능 향상을 달성했습니다. 폴란드어 ASR 시스템 평가에서 25개의 시스템 및 모델을 비교하여 다양한 데이터셋과 화자 인구통계에 따른 성능 변화를 발견했습니다.



### Handling Numeric Expressions in Automatic Speech Recognition (https://arxiv.org/abs/2408.00004)
- **What's New**: MCQ(주관식 문제)의 자동 생성 기술이 발전함에 따라, 교사들이 학습 평가에 소비하는 시간 절감 가능성이 크게 부각되고 있습니다. 그러나 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 n-gram 기반의 유사성만을 평가해 교육적 가치를 무시하고 있습니다. 이를 해결하기 위하여, 지식 종속 대답 가능성(KDA)이라는 새로운 자동 평가 메트릭이 제안되었습니다. 또한, 논문은 NLP 태스크에서 최근의 deep model들이 사람을 능가하는 정확성을 보였으나, spurious pattern에 의존해 robustness가 제한된다는 문제를 제기하고 있습니다. 이에 대처하기 위해, contrastive learning과 counterfactual augmentation을 활용한 방법을 제안합니다. 마지막으로, 자동 음성 인식(ASR) 시스템에서 숫자 표현을 올바르게 포맷하는 문제를 다루고, 이를 위해 LLM과 TTS 모델을 사용한 데이터 생성 전략을 비교합니다.

- **Technical Details**: 첫 번째 논문에서는 MCQ의 대답 가능성을 평가하는 KDA 메트릭을 제안하며, 인간 설문을 통한 KDA 측정 방법을 소개합니다. 두 번째 논문은 여러 counterfactual을 생성하여 집합적 의사 결정을 통해 단어들의 인과관계를 파악하는 방법을 설명합니다. 세 번째 논문에서는 numeric expressions (숫자 표현)을 올바르게 포맷하기 위해, LLM과 TTS 모델을 사용한 데이터 생성 전략을 사용하며, cascaded와 end-to-end 접근 방식을 비교합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였고, n-gram 기반 메트릭과 함께 사용할 때 다양한 전문가 라벨링된 MCQ 품질 지표에 대한 예측력이 뛰어난 것으로 나타났습니다. counterfactual 합성 방법은 여러 차원에서 significant improvement를 달성했으며, 특히 counterfactual robustness와 교차 도메인 일반화에서 두드러진 성과를 보였습니다. ASR 숫자 표현 포맷팅에서 end-to-end 모델은 낮은 지연시간과 추론 비용이라는 장점과 함께 competitive한 성능을 보였습니다.



New uploads on arXiv(cs.IR)

### Adversarial Text Rewriting for Text-aware Recommender Systems (https://arxiv.org/abs/2408.00312)
Comments:
          Accepted for publication at: 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024). Code and data at: this https URL

- **What's New**: 최근 연구들은 자동 MCQ 생성 효율성을 높이기 위해 새로운 평가 메트릭인 'Knowledge Dependent Answerability (KDA)'를 제안하면서, 기존 BLEU, ROUGE, METEOR 메트릭의 한계를 논의하고 있습니다. 그 외 여러 연구에서는 최신 심층 모델의 강점을 활용해 NLP 태스크에서 루버스트성을 향상시키기 위해 '대조 학습(contrastive learning)'과 '반사실적 증강(counterfactual augmentation)' 방법을 도입하며, 텍스트 재작성 공격(text rewriting attack)이 텍스트 인식 추천 시스템(text-aware recommender systems)의 취약성을 높이는 방법에 대해 연구하고 있습니다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 지식을 평가하는 새로운 메트릭으로, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 통해 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모사합니다. 또한, 반사실적 증강을 통한 NLP 모델의 루버스트성 개선은 여러 반사실적(counterfactual)을 생성하고 집단적 의사 결정을 통해 단어들의 인과관계를 더 견고하게 파악합니다. 텍스트 인식 추천 시스템의 경우, ATR 알고리즘은 두 가지 재작성 전략을 통해 공격 효율성을 극대화하며, 공격 자원에 따라 '이중 단계 미세 조정(two-phase fine-tuning)'과 '문맥 학습(in-context learning)' 방식을 사용합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 교실 환경에서의 사용성을 높이는 데 강한 상관관계를 보였습니다. 반사실적 증강을 통해 루버스트성, 교차 도메인 일반화(cross-domain generalization), 희소 데이터 일반화(generalization from scarce data)에서 상당한 개선을 이룰 수 있었습니다. ATR 알고리즘은 3개의 데이터셋과 4개의 텍스트 인식 추천 시스템에서 테스트한 결과, 기존 순위를 크게 향상시키며, 어드버서리 리라이트 공격(adversarial text rewriting attack)에 대한 높은 취약성을 보였습니다.



### Simple but Efficient: A Multi-Scenario Nearline Retrieval Framework for Recommendation on Taobao (https://arxiv.org/abs/2408.00247)
- **What's New**: 교사들이 학생 평가에 드는 시간을 줄이기 위해 자동 MCQ 생성에 대한 연구가 진행되고 있다. BLEU, ROUGE, METEOR 등의 기존 평가 메트릭은 단순히 단어 유사성만을 평가하여 교육적 가치를 고려하지 않는다. 이를 해결하기 위해, 우리는 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안한다.

- **Technical Details**: KDA는 MCQ의 대답 가능성(answerability)을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다. 우리는 먼저 학생 응답을 활용한 인간 평가를 통해 KDA를 측정하는 방법을 보여주었다. 그 후, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 사용성 측면에서 강한 상관관계를 가지고 있음을 보였다. 또한, n-gram 기반 유사성 메트릭과 결합할 때 KDA_disc와 KDA_cont는 다양한 전문가가 평가한 MCQ 품질 지표를 강력하게 예측할 수 있음을 증명했다.



### Review of Explainable Graph-Based Recommender Systems (https://arxiv.org/abs/2408.00166)
- **What's New**: 새로운 논문들은 MCQ 자동 생성의 교육적 평가를 위한 새로운 메트릭(KDA)을 제안하고, NLP 태스크의 robust하도를 향상시키기 위해 집합적 의사결정(collective decisions)을 활용하며, 그래프 기반 추천 시스템의 설명 가능성을 탐구한다.

- **Technical Details**: 첫 번째 논문에서는 지식 종속 가능성(KDA)을 측정하여 MCQ의 답변 가능성을 평가하고, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안한다. 두 번째 논문에서는 '여러 개의' counterfactual을 생성하여 단어들의 인과관계를 robust하게 파악하는 방법을 설명한다. 세 번째 논문에서는 그래프 기반 추천 시스템의 설명 가능성을 학습 방법, 설명 방법, 설명 유형 등의 세 가지 측면으로 분류하여 분석한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, 전문가가 라벨링한 MCQ 품질 측정치와 높은 예측력을 보인다. 집합적 의사결정을 통한 접근 방식은 counterfactual robustness, cross-domain generalization, 및 적은 데이터에서의 일반화 성능에서 중요한 향상을 보여준다.



### Semantic Codebook Learning for Dynamic Recommendation Models (https://arxiv.org/abs/2408.00123)
- **What's New**: 자동 다중 선택 질문(MCQ) 생성의 첨단 평가 메트릭 Knowledge Dependent Answerability (KDA)를 소개합니다. 또한, 자연어 처리(NLP)에서의 강력한 대립 학습(contrastive learning) 및 반사실 강화(counterfactual augmentation) 방법을 제안합니다. 마지막으로, 동적 순차 추천 모델(DSR)에서의 고도화된 프레임워크인 SOLID를 발표합니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 단순한 단어 유사성만을 측정하는 데 반해, KDA는 MCQ의 답변 가능성을 고려하여 학생의 학습 성과를 더 정확하게 평가함.'}, {'Robustness in NLP': 'spurious 패턴에 의존하는 기존 방법의 문제를 해결하기 위해 대립 학습(contrastive learning) 및 반사실 데이터 강화(counterfactual augmentation)를 사용하여 모델의 강력함을 높입니다. 여러 개의 counterfactual을 생성하여 집합적 의사 결정을 통해 인과관계를 더 명확히 파악합니다.'}, {'Dynamic Sequential Recommendation': 'SOLID 프레임워크는 아이템 시퀀스를 의미 시퀀스로 변환시키고 dual parameter model을 사용하여 검색 공간을 축소하고 추천 시스템의 동질성을 활용합니다. 또한, semantic metacode와 semantic codebook을 사용하여 분리된 아이템 표현을 저장하여 정확한 파라미터 생성을 보장합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': '인간 설문을 통한 평가에서 KDA_disc와 KDA_cont는 실제 강의실에서 사용 가능성이 높고 전문가들이 라벨링한 다양한 MCQ 품질 지표와 강한 상관관계를 보였습니다.'}, {'Robustness in NLP': '우리의 접근 방식은 대립 학습 및 반사실 강화 기술을 통해 1) 반사실 robustness, 2) 교차 도메인 일반화, 3) 희소한 데이터로부터의 일반화에서 현저한 개선을 달성했습니다.'}, {'Dynamic Sequential Recommendation': 'SOLID는 기존의 DSR을 일관되게 능가하며 더 정확하고 안정적이며 강력한 추천을 제공합니다. 실험 결과, SOLID는 다양한 평가 지표에서 탁월한 성능을 입증했습니다.'}]



### MIMNet: Multi-Interest Meta Network with Multi-Granularity Target-Guided Attention for Cross-domain Recommendation (https://arxiv.org/abs/2408.00038)
- **What's New**: 이미 존재하는 평가 메트릭들이 자동 MCQ 생성의 교육적 가치를 평가하지 못하는 문제점을 해결하기 위해, Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭이 제안되었습니다. 또한, 자연어 처리(NLP)에서 deep model의 robust를 보장하기 위해서 대비 학습(contrastive learning)과 counterfactual augmentation을 이용한 새로운 방법이 제안되었습니다. 마지막으로 cross-domain recommendation(CDR)에서 유저의 다양한 관심사를 반영하는 Multi-interest Meta Network with Multi-granularity Target-guided Attention (MIMNet) 모델이 제안되었습니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성과 학생의 지식 평가 능력을 측정하는 자동 평가 메트릭으로, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 사용해 pretrained language models를 활용하여 학생의 문제 해결 행동을 모방합니다. NLP 태스크에서 대비 학습과 multiple counterfactual 생성 및 집합적 의사 결정 (collective decisions)을 통한 robustness를 강화하는 방법이 제안되었습니다. 또한, MIMNet 모델은 capsule network를 이용해 유저의 다양한 관심사를 학습하고, meta network를 통해 여러 관심사 수준의 preference bridge를 생성하여 소스 도메인에서 타겟 도메인으로 유저의 선호도를 전이시킵니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보여줬으며, n-gram based similarity metrics와 결합 시 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강한 예측력을 보였습니다. NLP 태스크에서 새로운 robustness 방법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization을 포함한 다양한 차원에서 상당한 성능 향상을 달성했습니다. MIMNet 모델은 세 가지 실제 CDR 태스크에서 모든 baseline 방법보다 일관되게 우수한 성능을 보여줬습니다.



### Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation (https://arxiv.org/abs/2408.00490)
Comments:
          14 pages

- **What's New**: 세 개의 연구 논문에서 새로운 방법론을 제안했습니다. 첫 번째 논문은 Multiple Choice Questions (MCQ)의 새로운 자동 평가 메트릭(Knowledge Dependent Answerability, KDA)을 소개하고, 학생의 답변 가능성을 평가하는 데 초점을 맞췄습니다. 두 번째 논문은 NLP 태스크에서 깊이 모델의 강인성을 개선하기 위해 대조 학습과 반사실적(contrafactual) 증강을 활용하는 새로운 방법을 제안했습니다. 세 번째 논문은 그래프 신경망(Graph Neural Networks, GNNs)을 사용한 추천 시스템의 Out-of-Distribution (OOD) 데이터에서의 일반화 성능을 향상시키기 위한 새로운 접근법(CausalDiffRec)을 제안했습니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 고려하지 않기 때문에 새로운 평가 메트릭인 KDA를 제안하였습니다. KDA는 학생의 대상 사실에 대한 지식에 기반한 MCQ의 답변 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용해 학생의 문제 해결 능력을 모방하여 자동 평가합니다.', 'NLP Robustness': "대조 학습과 반사실적 증강을 활용하여 깊이 모델의 강인성을 개선하는 새로운 접근법을 제안합니다. '여러 개의' 반사실적을 생성하고 집합적 의사 결정(collective decisions)을 통해 인과 관계를 더 강하게 파악합니다.", 'GNN-based Recommendations': '구조적 인과 모델(Structural Causal Model, SCM)을 구성하여 상호 작용 데이터를 분석하고 환경 혼동 변수(environmental confounders)를 제거하여 GNN 모델의 OOD 데이터에 대한 일반화 성능을 향상시키기 위해 Causal Diffusion 기반의 그래프 표현 학습(CausalDiffRec) 방법을 제안합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'Human studies에 따르면, KDA_disc와 KDA_cont는 전문가들이 레이블링한 실제 강의실에서의 사용성과 강한 상관관계를 보였습니다. 이 메트릭을 기존의 n-그램 기반 유사성 메트릭과 결합할 경우 다양한 MCQ 품질 측정에 대해 강력한 예측력을 가지게 됩니다.', 'NLP Robustness': '우리의 접근법은 반사실적 강인성, 도메인 간 일반화, 희소 데이터에서의 일반화 등 다양한 차원에서 기존 방법보다 더 높은 성능을 보였습니다.', 'GNN-based Recommendations': 'CausalDiffRec은 다양한 OOD 데이터셋에서의 일반화 성능을 최대 36.73% 향상시켰으며, Food에서 10.69%, KuaiRec에서 18.83%, Yelp2018에서 22.41%, Douban에서 11.65%의 평균 성능 향상을 보였습니다.'}



### Towards Explainable and Interpretable Musical Difficulty Estimation: A Parameter-efficient Approach (https://arxiv.org/abs/2408.00473)
- **What's New**: 최근 논문에서는 자동 Multiple Choice Questions(MCQ) 생성의 교육적 가치를 평가하기 위해 novel한 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. 또한, NLP 모델의 robustness 향상을 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안합니다. 마지막으로, 음악 교육에서의 난이도 추정을 위한 투명하고 이해 가능한 방법을 제시하며, 특히 피아노 작품의 난이도 추정을 위한 새로운 white-box 모델인 RubricNet을 소개합니다.

- **Technical Details**: [{'paper': 'Automatic MCQ Generation', 'summary': '기존의 자동 MCQ 생성 메트릭은 BLEU, ROUGE, METEOR와 같은 n-gram 유사성에 초점을 맞추고 있지만, 이는 교육적 가치를 간과하고 있습니다. 이를 해결하기 위해 제안된 KDA는 MCQ의 대답 가능성을 측정하고, 학생의 지식을 평가하는 능력을 분석합니다. 이 메트릭은 학생의 응답을 기반으로 한 KDA와, 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont로 구성됩니다.'}, {'paper': 'NLP Robustness', 'summary': '최근 NLP 모델들이 spurious pattern에 의존해 robustness가 제한된다는 문제를 다룹니다. 인간의 도움 없이 여러 개의 counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과관계를 분석하는 방법을 제안합니다. 이를 통해 다양한 측면에서 성능 향상이 이루어졌음을 실험적 결과로 증명했습니다.'}, {'paper': 'Music Difficulty Estimation', 'summary': '음악 작품의 난이도를 추정하기 위해 이해 가능한 설명을 제공하는 새로운 white-box 모델인 RubricNet을 소개합니다. 이 모델은 음악 교육에서 흔히 사용되는 루브릭(rubric) 개념을 활용하여 투명성과 해석 가능성을 높였습니다. 평가 결과는 피아노 레퍼토리의 9개 등급을 기준으로 큰 정확도와 낮은 평균 제곱 오차(MSE)를 보여주었습니다.'}]

- **Performance Highlights**: [{'paper': 'Automatic MCQ Generation', 'summary': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가가 라벨한 MCQ 품질 측정에 강한 예측 능력을 가집니다.'}, {'paper': 'NLP Robustness', 'summary': '제안한 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization에서 유의미한 성능 향상을 보였습니다.'}, {'paper': 'Music Difficulty Estimation', 'summary': 'RubricNet 모델은 9개의 피아노 레벨을 기준으로 41.4%의 독립적인 정확도와 1.7의 평균 제곱 오차(MSE)를 기록하며, 해석 가능하고 높은 정확도의 결과를 제공했습니다.'}]



### DiscipLink: Unfolding Interdisciplinary Information Seeking Process via Human-AI Co-Exploration (https://arxiv.org/abs/2408.00447)
- **What's New**: 자동 MCQ 생성을 위한 기존 평가 메트릭은 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)이라는 새로운 메트릭을 제안. 또한, 최근 NLP deep models의 robustness 향상을 위해 종속 패턴에 의존하지 않는 대조적 학습(contrastive learning) 및 counterfactual augmentation 방법을 도입. 마지막으로, DiscipLink이라는 새로운 인터랙티브 시스템을 소개, 연구자들이 대형 언어 모델(LLMs)과 협력하여 다양한 학문 분야에서 정보를 탐색할 수 있도록 지원.

- **Technical Details**: KDA 측정은 학생의 응답을 기반으로 학습된 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 pre-trained language models을 활용해 학습자의 문제 해결 행동을 모방해 측정됨. 대조적 학습 및 counterfactual augmentation 방법은 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어의 인과관계를 robust하게 지도를 제안. DiscipLink 시스템은 사용자의 관심 주제를 바탕으로 탐색 질문을 생성하고 분야별 용어를 사용해 자동으로 쿼리를 확장해 논문을 검색.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 실제 강의실 사용성과 강한 상관관계가 있음이 확인됨. 여러 차원에서 robustness, cross-domain generalization 및 일반화가 매우 향상됨. DiscipLink 시스템은 기존 문헌 검색 도구들보다 더 효율적이고 포괄적인 지식 범위를 탐색 가능하게 함. DiscipLink를 사용한 경험 많은 대화 연구자들은 시스템의 공동 탐색 워크플로우에 크게 만족했으며 일부 제한점을 지적함.



### DistillGrasp: Integrating Features Correlation with Knowledge Distillation for Depth Completion of Transparent Objects (https://arxiv.org/abs/2408.00337)
Comments:
          10 pages, 5 figures

- **What's New**: 기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR가 단순히 단어단위의 유사성만을 측정하고 교육적 가치를 평가하지 못하는 문제점을 해결하기 위해, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, 최근 NLP 태스크에서 높은 정확성을 보이는 모델들도 spurious pattern에 의존하는 문제로 인해 robustness가 부족한 문제를 contrastive learning (대조 학습) 및 counterfactual augmentation을 통해 개선하려는 접근을 소개합니다. 투명한 객체의 깊이 데이터를 정확히 캡처하지 못하는 RGB-D 카메라 문제를 해결하기 위해, DistillGrasp라는 효율적인 depth completion 네트워크를 제안했습니다.

- **Technical Details**: 새롭게 제안된 KDA 메트릭은 학생의 목표 사실(knowledge of the target fact)에 대한 지식을 평가하기 위해 최초로 제안되었습니다. KDA를 측정하기 위해 human survey 응답을 바탕으로 하는 방법을 제시했으며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 이를 자동으로 approximation하는 방법을 제안했습니다. NLP robustness를 향상시키기 위해 여러 개의 counterfactual을 생성하여 collective decision을 기반으로 각 용어의 인과관계를 robust하게 감독하는 방법을 제안했습니다. DistillGrasp는 teacher branch와 student branch로 나누어 지식 증류(knowledge distillation)을 사용하여 모델의 효율성을 높이는 방법입니다. 특히, teacher branch에서는 position correlation block (PCB)을 사용하고, student branch에서는 consistent feature correlation module (CFCM)를 사용합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서 usability와 매우 높은 상관관계를 보여줬습니다. 또한 n-gram 기반의 유사성 메트릭과 결합했을 때, 다양한 전문가가 라벨링한 MCQ 품질 지표에 대한 예측력이 뛰어났습니다. NLP 모델의 robustness를 높이기 위해 제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data로부터의 일반화에서 중요한 향상을 보였습니다. DistillGrasp는 ClearGrasp 데이터셋에서 최신 방법들보다 정확도와 일반화에서 뛰어난 성능을 보였으며, UR10e 로봇에 적용하여 실제 환경에서도 효과성과 robustness를 입증받았습니다.



### Exploiting Preferences in Loss Functions for Sequential Recommendation via Weak Transitivity (https://arxiv.org/abs/2408.00326)
Comments:
          Accepted to CIKM 2024, Short Research Paper Track

- **What’s New**: 자동으로 객관식 문제(Multiple Choice Questions, MCQ)를 생성하는 것은 교육자들의 평가 시간을 상당히 줄일 수 있지만, 기존의 평가 지표(BLEU, ROUGE, METEOR)는 단순히 생성된 MCQ와 데이터셋의 정답 샘플 간의 n-gram 유사성만을 측정하여 교육적 가치를 평가하지 않습니다. 이를 해결하기 위해 대상 사실(Target Fact)에 대한 지식을 고려한 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성(Answerability)을 측정하여 대상 사실에 대한 학생의 지식을 평가합니다. 이를 위해 학생 설문조사로부터 KDA를 측정하는 방법을 제시하고, 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 지표(KDA_disc와 KDA_cont)를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA 및 실제 교실 환경에서의 사용성과 강한 상관관계를 가짐을 입증했습니다. 또한, n-gram 기반 유사성 지표와 결합되었을 때, KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정 지표에 강한 예측력을 가졌습니다.



### Leveraging Weak Cross-Modal Guidance for Coherence Modelling via Iterative Learning (https://arxiv.org/abs/2408.00305)
Comments:
          Accepted by ACM Multimedia 2024

- **What's New**: 세 가지 주요 연구들이 소개되었습니다. 첫 번째 연구는 자동 MCQ 생성의 평가 메트릭으로서 지식 종속 대답 가능성 (Knowledge Dependent Answerability, KDA)을 제안하여 BLEU, ROUGE, METEOR과 같은 기존 메트릭의 한계를 극복하고자 합니다. 두 번째 연구는 NLP 태스크에서 deep models의 robustness 향상을 위한 대안으로 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안합니다. 세 번째 연구는 cross-modal coherence modeling을 위해 약한 교차 모달 참고 주문 방법 (WeGO)을 제시하여 더 유연하고 비용 효율적인 데이터 활용을 목표로 합니다.

- **Technical Details**: 첫 번째 연구에서는 KDA를 측정하기 위한 인간 설문조사 기반 방법과 pretrained language model을 활용한 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안하였습니다. 두 번째 연구는 집합적 의사 결정 (collective decisions)을 통해 spurious correlation의 영향을 최소화하고 학습을 진행하는 접근 방식을 소개합니다. 세 번째 연구에서는 Iterative Learning paradigm을 적용하여 두 모달리티의 order 모델을 공동 최적화하는 방법을 제안하며, 이는 inferencing 단계에서 교차 모달리티 간의 주문 가이드를 더 효율적으로 사용할 수 있게 합니다.

- **Performance Highlights**: 첫 번째 연구에서는 KDA_disc와 KDA_cont가 실제 강의실 세트에서 높은 유용성과 상관관계를 보였습니다. 두 번째 연구는 counterfactual robustness, cross-domain generalization, 그리고 limited data environment에서 일반화된 성능을 향상시키는 결과를 거두었습니다. 세 번째 연구에서는 두 가지 공공 데이터셋에서 기존 방법들보다 뛰어난 성능을 입증하였으며, 주요 기술 모듈들은 ablation studies를 통해 효과적으로 평가되었습니다.



### RDP: Ranked Differential Privacy for Facial Feature Protection in Multiscale Sparsified Subspac (https://arxiv.org/abs/2408.00294)
Comments:
          13 pages, 6 figures

- **What's New**: 최근 자동 MCQ(객관식 질문) 생성 방법이 교육자의 평가 시간을 줄일 수 있는 가능성을 보여주고 있습니다. 하지만 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 교육적 가치를 반영하지 못하여, 학생의 지식을 평가하는 능력을 적절히 평가하지 못합니다. 이를 해결하기 위해, 대상 사실에 대한 학생의 지식을 평가하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다.

- **Technical Details**: KDA는 대상 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성(answerability)을 측정합니다. 우리는 먼저 인간 설문조사에서 수집한 학생의 응답을 통해 KDA를 측정하는 방법을 제시하고, 이를 모사하는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭은 사전 학습된 언어 모델(pre-trained language models)을 활용하여 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont은 (1) KDA와 (2) 실제 강의실 세트에서 전문가가 라벨링한 사용성과 높은 상관관계를 가지며, n-gram 유사성 메트릭과 결합했을 때 다양한 전문가 라벨 MCQ 품질 측정치에 대해 강한 예측력을 가짐을 보여주었습니다.



### Temporal Subspace Clustering for Molecular Dynamics Data (https://arxiv.org/abs/2408.00056)
Comments:
          Accepted as a research paper at BIOKDD 2024

- **MCQ Generation**: {"What's New": '기존의 MCQ 생성 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해, MCQ의 대답 가능성(answerability)을 측정하고 학생의 지식을 평가할 수 있는 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다.', 'Technical Details': 'KDA는 사람들의 반응을 기반으로 하고, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. KDA_disc와 KDA_cont 같은 두 가지 자동 평가 메트릭을 제안합니다.', 'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강하게 상관관계가 있음을 확인하였습니다. 또한 n-gram 기반 유사성 메트릭을 결합했을 때, 다양한 전문가 라벨링 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.'}

- **Contrastive Learning for NLP Robustness**: {"What's New": '최근 NLP 태스크에서 높은 정확성을 보이는 deep model들의 robustness 문제를 해결하기 위해, 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 단어의 인과관계를 robust하게 파악하는 방법을 제안합니다.', 'Technical Details': '기존 방법들과 달리, 우리의 접근법은 여러 개의 반사실적(counterfactual)을 생성하고 집합적 의사 결정(collective decisions)을 통해 더 robust하게 단어들의 인과관계를 파악합니다.', 'Performance Highlights': '제안한 방법이 반사실적 견고성(counterfactual robustness), 크로스-도메인 일반화(cross-domain generalization) 및 희소 데이터로부터의 일반화(generalization from scarce data)에 있어 중요한 개선을 달성했음을 실험적으로 확인했습니다.'}

- **MOSCITO for Molecular Dynamics**: {"What's New": '분자 동역학 데이터 분석을 위한 일단계(One-step) 서브스페이스 군집화(subspace clustering) 방법인 MOSCITO를 소개합니다.', 'Technical Details': 'MOSCITO는 시간 시퀀스 데이터에서의 순차적 관계를 이용하여 분자 동역학 궤적(molecular dynamics trajectory)을 클러스터링 합니다. 기존 두 단계 클러스터링 방법과 달리, 절차의 단순함과 효율성을 제공합니다.', 'Performance Highlights': 'MOSCITO는 4개의 서로 다른 단백질 실험에서 state-of-the-art 성능을 달성하였으며, 특히 클러스터 수가 적을 때 궤적의 분할을 더욱 잘 수행했습니다.'}



New uploads on arXiv(cs.CV)

### Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation (https://arxiv.org/abs/2408.00766)
Comments:
          30 pages, 20 figures, Accepted to ECCV 2024

- **What's New**: 이 논문에서는 자동 MCQ(다지선다형 질문: Multiple Choice Questions) 생성은 교사의 학습 평가 시간을 줄일 수 있다는 점에 주목하고 있습니다. 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 생성된 MCQ의 교육적 가치나 학생의 지식을 평가하는 능력을 충분히 반영하지 못하고 있습니다. 이를 해결하기 위해, 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다.

- **Technical Details**: KDA는 학생이 해당 사실(target fact)에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성(Answerability)을 측정합니다. 이 연구에서는 먼저 학생 응답을 기반으로 KDA를 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방함으로써 KDA를 근사합니다.

- **Performance Highlights**: Human studies 결과, KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 추가로, n-gram 기반 유사성 메트릭과 결합했을 때, 전문가가 라벨한 다양한 MCQ 품질 메트릭에 대한 예측 능력이 강력한 것으로 나타났습니다.



### MM-Vet v2: A Challenging Benchmark to Evaluate Large Multimodal Models for Integrated Capabilities (https://arxiv.org/abs/2408.00765)
Comments:
          Extension of MM-Vet: arXiv:2308.02490

- **What's New**: 이 논문에서는 기존의 MM-Vet의 단점을 보완하여 새로운 평가 메트릭인 MM-Vet v2를 소개합니다. MM-Vet v2는 '이미지-텍스트 시퀀스 이해'라는 새로운 비전-언어(Vision-Language) 능력을 평가하여 모델이 이미지-텍스트 시퀀스를 처리하는 능력을 포함합니다.

- **Technical Details**: MM-Vet v2는 기존 MM-Vet의 여섯 가지 핵심 VL 능력 (Recognition, Knowledge, Spatial Awareness, Language Generation, OCR, Math) 외에 추가로 '이미지-텍스트 시퀀스 이해'를 포함합니다. 이를 위해 연구팀은 다양한 시나리오에서 수집한 517개의 질문을 설계하고, 긴 답변을 필요로 하는 질문은 GPT-4V를 사용해 초안을 작성한 뒤 전문가가 검토합니다.

- **Performance Highlights**: MM-Vet v2로 다양한 대형 멀티모달 모델(LMMs)을 평가한 결과, Claude 3.5 Sonnet이 71.8점을 기록하며 가장 높은 성능을 보였습니다. GPT-4o는 71.0점을 기록했고, 오픈 소스 모델 중에서는 InternVL2-Llama3-76B가 68.4점으로 가장 뛰어난 성능을 나타냈습니다.



### UniTalker: Scaling up Audio-Driven 3D Facial Animation through A Unified Mod (https://arxiv.org/abs/2408.00762)
- **What's New**: UniTalker라는 통합 모델을 소개하여 다양한 3D 얼굴 애니메이션 데이터셋과 다양한 주석(annotation) 타입을 효과적으로 활용할 수 있게 하였습니다. 또한 A2F-Bench라는 새로운 대규모 데이터셋을 구축하여 18.5시간의 학습 데이터를 제공합니다.

- **Technical Details**: UniTalker는 멀티헤드 아키텍처(Multi-Head Architecture)를 특징으로 하며, 주성분 분석(PCA), 모델 워밍업(Model Warm-Up), 그리고 Pivot Identity Embedding과 같은 세 가지 훈련 전략을 활용하여 훈련의 안정성을 높이고 멀티헤드 출력 간의 일관성을 유지합니다. 이를 통해 다른 주석 타입을 가진 데이터셋들을 통합하여 학습할 수 있습니다.

- **Performance Highlights**: [{'dataset': 'BIWI', 'lip_vertex_error': '4.25e-4에서 3.86e-4로 9.2% 감소'}, {'dataset': 'Vocaset', 'lip_vertex_error': '9.63e-6에서 8.30e-6로 13.7% 감소'}, {'additional': 'A2F-Bench 데이터셋에서 평균 6.3%의 에러 감소'}, {'generalization': '미보유 데이터셋에 대해서도 소량의 데이터만으로도 기존 모델들보다 더 나은 성능을 보여줌'}]



### Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention (https://arxiv.org/abs/2408.00760)
- **Multiple Choice Questions (MCQ) Generation**: {"What's New": '새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 도입하여, MCQ의 교육적 가치를 평가하는 능력을 향상시켰습니다.', 'Technical Details': 'KDA는 학생의 대상 사실(target fact)에 대한 지식을 기준으로 MCQ의 답변 가능성을 측정합니다. 주어진 대상 사실에 기반하여 인간 설문조사를 통해 KDA를 측정하고, 이를 근사하기 위해 미리 훈련된 언어 모델을 활용한 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안합니다.', 'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 강의실 세팅에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 레이블 MCQ 품질 측정에 대한 예측력이 높아짐을 확인했습니다.'}

- **Robust NLP Models**: {"What's New": '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 NLP 모델의 robustness를 향상시키는 새로운 접근법이 제안되었습니다.', 'Technical Details': '기존 증강 방식이 spurious correlation에 영향을 받는 반면, 이 접근법은 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 각 단어의 인과관계를 robust하게 감독합니다.', 'Performance Highlights': '테스트 결과, 이 접근법은 counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 측면에서 성능 개선을 보여주었습니다.'}

- **Smoothed Energy Guidance (SEG) for Image Generation**: {"What's New": 'Self-attention 메커니즘의 에너지 기반 관점을 활용하여 이미지 생성 품질을 향상시키는 Smoothed Energy Guidance (SEG)라는 새로운 방법론이 제안되었습니다.', 'Technical Details': 'SEG는 Gaussian 커널 매개변수를 조정하여 에너지 풍경의 곡률을 제어하고, query blurring을 통해 복잡도를 낮추며, 본래의 예측에서 구조적 변경을 최소화합니다.', 'Performance Highlights': '텍스트 조건 유무에 관계없이 다양한 실험에서 SEG가 이전 방법들보다 더 나은 샘플 품질을 제공하고 부작용을 줄이는 Pareto 개선을 달성했습니다.'}



### Text-Guided Video Masked Autoencoder (https://arxiv.org/abs/2408.00759)
Comments:
          Accepted to ECCV 2024

- **What's New**: 머신러닝 및 자연어 처리(NLP)에서의 발전에도 불구하고 기존 MCQ(객관식 문제) 생성 평가 방식에는 교육적 가치와 관련된 측면에서 한계가 있었습니다. 이 논문에서는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 소개하여, MCQ의 교육적 유효성을 더 잘 평가할 수 있도록 하였습니다. 두 번째 연구는 반대 사례(counterfactual)를 이용한 학습을 통해 모델의 견고성을 강화하는 방식을 제안하고 있으며, 마지막 논문에서는 텍스트 지도를 이용한 마스킹 알고리즘(TGM)을 도입하여 비디오 인식 성능을 향상시키는 내용을 다루고 있습니다.

- **Technical Details**: 첫 번째 논문에서는 KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하였습니다. 두 번째 논문에서는 여러 개의 counterfactual을 생성하고 collective decisions를 통해 견고성을 강화하는 방법을 제안하였습니다. 마지막 논문에서는 text-guided masking(TGM) 알고리즘을 도입하여 비디오와 텍스트 간 대비 학습을 결합한 새로운 프레임워크를 소개하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실에서 전문가들이 언급한 사용성과 강한 상관관계를 보였습니다. 두 번째 연구에서는 counterfactual robustness, cross-domain generalization 및 scarce data에서의 generalization 성능이 눈에 띄게 향상되었습니다. 마지막 연구에서는 텍스트 지도를 이용한 마스킹이 Kinetics-400(K400)과 Something-Something V2(SSv2) 데이터셋에서 최적의 성능을 기록하였으며, 5개의 액션 인식 데이터셋과 1개의 주관적 이해 데이터셋에서 가장 높은 성능을 달성했습니다.



### Segment anything model 2: an application to 2D and 3D medical images (https://arxiv.org/abs/2408.00756)
Comments:
          11 pages, 7 figures. A first attempt on evaluating SAM 2 on medical images

- **What's New**: 기존의 여러 평가 메트릭들이 다루기 부족했던, 교육적 가치뿐만 아니라 답변 가능성(knowledge dependent answerability, KDA)를 측정하는 새로운 자동 평가 메트릭을 제안합니다. 이를 통해 학생의 대상 사실(target fact)에 대한 지식을 직접적으로 평가할 수 있게 되었습니다. 또한, KDA를 근사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 소개하였습니다.

- **Technical Details**: KDA 측정을 위해 학생 응답을 대상으로 한 human survey를 기반으로 한 KDA 산출 방법을 먼저 보여준 다음, pre-trained language models을 활용해, student의 문제 해결 행동을 모사하여 KDA를 근사하는 KDA_disc와 KDA_cont를 제안하였습니다. 또한, 두 자동 평가 메트릭은 전문가가 labeled한 MCQ 품질 메트릭을 예측하는 데 강한 예측력을 가졌다는 것을 나타냈습니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계가 있음을 확인하였습니다. 이들 메트릭이 기존 n-gram 기반 유사성 평가 메트릭과 결합되는 경우 MCQ 품질 측정에서 더욱 강력한 예측력을 보였습니다.



### Coarse Correspondence Elicit 3D Spacetime Understanding in Multimodal Language Mod (https://arxiv.org/abs/2408.00754)
Comments:
          project page: this https URL

- **What's New**: 이번 연구에서는 대답 가능성(answerability)을 평가하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 이는 기존의 BLEU, ROUGE, METEOR 등의 평가 메트릭들이 갖는 교육적 가치를 무시하는 문제를 해결하고, 학생들이 해당 사실에 대해 얼마나 잘 알고 있는지를 평가하는 능력을 측정하는 데 중점을 둡니다.

- **Technical Details**: 새로운 KDA 메트릭은 학생들의 응답을 기반으로 KDA를 측정한 다음, 사전 학습된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 상관 관계를 통해 KDA와 실제 강의실에서의 사용성을 평가합니다.

- **Performance Highlights**: Human evaluation을 통해 두 가지 새로운 메트릭(KDA_disc와 KDA_cont)이 KDA 및 전문가들이 라벨링한 다양한 MCQ 품질 측정치와 강한 상관관계를 가지는 것을 확인했습니다. 기존의 n-gram 기반의 유사성 메트릭과 결합할 경우, 전문가 라벨링된 MCQ 품질 측정치를 예측하는 데 높은 성능을 보였습니다.



### Leaf Angle Estimation using Mask R-CNN and LETR Vision Transformer (https://arxiv.org/abs/2408.00749)
- **What's New**: 이 논문은 다중 선택 질문 (Multiple Choice Questions, MCQ)의 자동 생성을 위한 평가 메트릭이 기존의 BLEU, ROUGE, METEOR와 같은 n-그램 기반의 유사성만을 비교하여 교육적 가치를 무시하는 문제를 해결하기 위해 제안되었다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하였다.

- **Technical Details**: KDA는 생성된 MCQ의 대답 가능성 (answerability)를 측정하고, 대상 사실에 대한 학생의 지식 평가 능력을 평가한다. KDA를 측정하기 위해 인간 평가단의 학생 응답을 사용한 후, 사전 학습된 언어 모델들을 활용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하였다.

- **Performance Highlights**: 인간 연구를 통해, KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었으며, n-그램 기반의 유사성 메트릭들과 결합했을 때 전문가가 레이블링한 MCQ 품질 측정에 대한 예측력이 강함을 보여주었다.



### Collaborative Vision-Text Representation Optimizing for Open-Vocabulary Segmentation (https://arxiv.org/abs/2408.00744)
Comments:
          ECCV 2024

- **What's New**: 자동 생성된 Multiple Choice Questions (MCQ)에 대한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. CLIP 모델 기반의 Open-Vocabulary Segmentation (OVS) 태스크의 성능을 강화하는 새로운 접근 방식 MAFT+도 도입되었습니다.

- **Technical Details**: [{'자동 MCQ 생성': '기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 무시하고 단순히 n-gram 기반 유사성에 의존하여 MCQ를 평가합니다. 이를 해결하기 위해, 우리는 KDA라는 새로운 메트릭을 제안하여 MCQ의 대답 가능성과 해당 사실에 대한 학생의 지식 평가 능력을 측정합니다. 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하며, 이는 미리 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다.'}, {'OVS 강화': '기존 OVS 모델은 CLIP 모델을 동결하거나 특정 방식으로 조정하여 상호 협력 최적화를 진행하지 않습니다. 우리는 Content-Dependent Transfer (CDT)와 Representation Compensation (RC) 전략을 도입하여 CLIP-T와 CLIP-V의 동시 최적화 및 제로 샷 성능을 유지하는 방법을 제안했습니다.'}]

- **Performance Highlights**: [{'자동 MCQ 생성': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 KDA 메트릭 사이에 강한 상관관계를 가지고 있으며, 전문가가 레이블링한 MCQ 품질 지표에 대한 예측력도 높습니다.'}, {'OVS 강화': 'MAFT+는 기존의 OVS와 비교하여 A-847, A-150, PC-459, PC-59, PAS-20 데이터셋에서 각각 +0.5, +2.3, +3.4, +0.4, +1.1 mIoU로 성능이 향상되었습니다. 또한, ADE20K 데이터셋에서는 27.1 PQ, 73.5 SQ, 32.9 RQ의 성능을 기록했습니다.'}]



### Virchow 2: Scaling Self-Supervised Mixed Magnification Models in Pathology (https://arxiv.org/abs/2408.00738)
- **What's New**: 현재의 자동 MCQ(Multiple Choice Questions, 객관식 질문) 생성에 대한 평가 메트릭은 BLEU, ROUGE, METEOR 등의 n-gram 기반 유사성에 집중해 교육적 가치를 무시하고 있다. 이를 해결하기 위해 본 논문에서는 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability, 지식 종속 가능성)를 제안했다.

- **Technical Details**: KDA는 학생이 목표 사실(target fact)에 대한 지식을 가지고 있을 때 MCQ의 답변 가능성을 측정하는 방법이다. Human survey를 통해 학생 반응 기반으로 KDA를 측정한 후, 사전 학습된 언어 모델(pre-trained language models)을 사용해 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안한다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성(usability)와 강한 상관관계를 가지고 있음을 확인했다. 또한, n-gram 기반 유사성 메트릭과 결합할 때 KDA_disc와 KDA_cont가 다양한 전문가가 라벨링한 MCQ 품질 측정에 강력한 예측력을 가짐을 보여주었다.



### TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models (https://arxiv.org/abs/2408.00735)
Comments:
          Project page: this https URL

- **What's New**: 최신 논문은 다양한 학습 평가, 강인성 개선, 그리고 텍스트 기반 이미지 편집과 관련된 새로운 기술을 제안합니다. 자동 생성 MCQ 평가를 위해 Knowledge Dependent Answerability(KDA)를 도입하고, 텍스트 기반 이미지 편집을 위해 새로운 빠른 샘플링 방법을 소개합니다.

- **Technical Details**: 첫 번째 연구에서는 교육적 가치를 평가하기 위해 BLEU, ROUGE, METEOR 대신 KDA (Knowledge Dependent Answerability)라는 새로운 평가 메트릭을 제안합니다. 두 번째 연구에서는 contrastive learning과 counterfactual augmentation을 활용하여 강인성을 향상시키는 방법을 설명합니다. 세 번째 연구는 'edit-friendly' DDPM-noise inversion 접근 방식을 분석하고, 시각적 아티팩트를 줄이며 편집 강도를 높이는 방식으로 향상된 텍스트 기반 이미지 편집을 제공합니다.

- **Performance Highlights**: 첫 번째 논문에서는 KDA_disc와 KDA_cont 메트릭의 높은 상관성과 예측력을 확인했습니다. 두 번째 연구에서는 counterfactual robustness, cross-domain generalization, 그리고 희소 데이터에서의 일반화에서 유의한 성능 향상을 이루었습니다. 세 번째 연구에서는 기존 편집 방법보다 5배 빠른 속도를 선보이며, 시각적 아티팩트를 최소화하고 높은 편집 품질을 유지할 수 있었습니다.



### SAM 2: Segment Anything in Images and Videos (https://arxiv.org/abs/2408.00714)
Comments:
          Website: this https URL

- **What's New**: 강력한 성능과 사용성을 갖춘 새로운 자동 평가 메트릭 지식 종속 가능성(KDA)을 제안합니다. 최근 NLP 태스크에서의 deep models의 성능이 높지만 spurious patterns에 의존하는 문제를 해결하기 위해 대조학습과 반사실적 증강법을 사용했습니다. 새롭게 공개된 Segment Anything Model 2 (SAM 2)는 이미지와 비디오에서 프롬프트 기반 분할을 수행하며, 특히 비디오 분할 성능이 크게 향상되었습니다.

- **Technical Details**: MCQ 생성 모델의 교육적 가치를 평가하기 위해 KDA를 도입했으며, 이는 인간 평가를 통해 확인되었습니다. NLP의 robustness 향상을 위해 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어의 인과관계를 파악했습니다. SAM 2는 Streaming Memory를 갖춘 Transformer 구조를 사용해 실시간 비디오 처리 능력을 강화하였으며, 데이터 엔진을 통해 세계 최대의 비디오 분할 데이터셋을 구축했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont가 실제 강의실 사용성에서 높은 상관관계를 나타냈습니다. 우리의 접근 방식은 task model의 바이어스에 덜 민감하며, 반사실적 증강에서 tf robustness, cross-domain generalization, 데이터 부족 상황에서의 generalization 등 여러 측면에서 성능 개선을 보였습니다. SAM 2는 이전 방법들에 비해 3배 적은 상호작용으로 더 높은 정확도를 달성했으며, 이미지 분할에서도 6배 빠른 성능을 보였습니다.



### MotionFix: Text-Driven 3D Human Motion Editing (https://arxiv.org/abs/2408.00712)
Comments:
          arXiv v1

- **What's New**: 이 논문은 3D 모션 편집에 대해 다루고 있으며, 텍스트 설명을 기반으로 3D 인간 모션을 원하는 방식으로 편집하는 모델을 제안합니다. 기존 모션 캡처 데이터셋을 활용하여 반자동으로 모션-모션-텍스트 트리플렛 형태의 데이터를 수집함으로써 새로운 MotionFix 데이터셋을 구축하였습니다. 이를 통해 조건부 확산 모델 (TMED)을 훈련하여 텍스트 지시사항에 따라 모션을 편집할 수 있도록 했습니다.

- **Technical Details**: 제안된 방법론은 기존의 텍스트-이미지 생성 모델과는 달리, 3D 인간 모션 생성 및 편집에 중점을 두고 있습니다. MotionFix 데이터셋은 소스 모션, 목표 모션, 편집 텍스트의 트리플렛 데이터로 구성되어 있으며, 최근 TMR 모션 임베딩(space)을 활용하여 적절한 모션 쌍을 자동으로 매칭하고, 이를 통해 텍스트 설명을 수집합니다. 이에 기반한 Text-based Motion Editing Diffusion 모델(TMED)을 활용하여 소스 모션과 편집 텍스트를 입력으로 받아 새로운 모션을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 TMED 모델은 텍스트 기반의 모션 생성 방법들보다 우수한 성능을 보였습니다. 특히 모션-모션-텍스트 트리플렛 데이터를 사용했을 때 목표 모션과 더 가깝게 생성되었고, 편집 관련 바디 파트 탐지에서도 향상된 성능을 나타냈습니다. TMED 모델은 다양한 편집 작업에 대해 질적 및 양적으로 뛰어난 성능을 입증했습니다.



### Synthetic dual image generation for reduction of labeling efforts in semantic segmentation of micrographs with a customized metric function (https://arxiv.org/abs/2408.00707)
- **What's New**: 이번 연구에서는 자동 MCQ 생성의 교육적 가치를 측정하기 위한 새로운 지식 종속 가능성(KDA) 메트릭을 제안했습니다. 또한, 다양한 deep learning 모델의 NLP 태스크에서 spurious 패턴에 의존하여 강인성이 제한되는 문제를 해결하기 위해 대조 학습과 반사실 증강을 활용한 새로운 접근 방식을 제안했습니다. 마지막으로, 재료 분석을 위한 semantic segmentation 모델의 성능을 향상시키기 위해 합성 미세 구조 이미지를 생성하는 워크플로우를 소개했습니다.

- **Technical Details**: 첫 번째 연구에서는 MCQ의 대답 가능성을 평가하는 KDA 메트릭을 제안하고, 이를 기반으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 개발했습니다. 두 번째 연구에서는 여러 개의 반사실을 생성하고 집합적 의사 결정을 통해 단어의 인과관계를 robust하게 파악하는 방법을 제안했습니다. 세 번째 연구에서는 VQ-VAE 모델을 사용하여 합성 마이크로그래프와 마스크를 생성하고, 이를 통해 semantic segmentation 모델을 학습하는 방법을 제시했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 두 번째 연구에서는 제안된 접근 방식이 반사실 강인성, 도메인 간 일반화, 소수 데이터의 일반화에서 상당한 개선을 이루었습니다. 세 번째 연구에서는 합성 데이터를 사용하여 U-Net 모델을 학습한 결과, 샘플 준비와 획득 시간 및 이미지 처리와 라벨링 작업의 노력이 감소했음을 확인했습니다.



### Point-supervised Brain Tumor Segmentation with Box-prompted MedSAM (https://arxiv.org/abs/2408.00706)
Comments:
          2024 IEEE Nuclear Science Symposium and Medical Imaging Conference

- **What's New**: 이번 연구는 MCQ (Multiple Choice Questions) 생성의 자동 평가 메트릭의 한계를 극복하기 위해 제안된 새로운 Knowledge Dependent Answerability (KDA) 지표를 소개합니다. KDA는 학생이 해당 사실(target fact)에 대한 지식을 기반으로 MCQ의 대답 가능성을 측정하며, 이는 기존의 n-gram 유사성 메트릭이 지니고 있던 교육적 가치 측면의 한계를 극복합니다.

- **Technical Details**: KDA 측정 방식은 먼저 학생들의 응답을 통해 휴먼 설문조사를 기반으로 한 평가 지표를 측정합니다. 이후, 미리 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행위를 모방함으로써 KDA를 근사화하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이는 약어화된 'discrete'와 'continuous'에서 각각 유래됐습니다. 인간 연구를 통해 이 두 메트릭은 실제 교실 환경에서의 사용성과 강한 상관관계를 지니고 있음을 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가들이 라벨링한 다양한 MCQ 품질 기준을 예측하는 데 있어 높은 예측력을 보였습니다. 특히, 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 더욱 강력한 예측력을 보여주었습니다.



### Joint Neural Networks for One-shot Object Recognition and Detection (https://arxiv.org/abs/2408.00701)
Comments:
          published as part of the PhD thesis: this https URL

- **What's New**: [{'title': 'Enhancing Multiple Choice Questions (MCQ) Generation with Knowledge Dependent Answerability (KDA)', 'summary': '기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치 평가에 한계가 있다는 문제를 해결하고자, 우리는 MCQ의 대답 가능성(Answerability)과 교육적 가치를 평가하는 새로운 자동 평가 메트릭인 KDA를 제안합니다. 인간 설문조사에서 측정된 학생 반응을 기반으로 KDA를 측정하는 방법을 보여주고, 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다.'}]

- **Technical Details**: [{'title': 'Robustness Improvements in Deep Learning Models Through Contrastive Learning and Counterfactual Augmentation', 'summary': '본 논문에서는 최신 deep model들이 NLP 태스크에서 사람보다 나은 정확성을 보였으나 spurious pattern에 의존해 robustness가 제한된다는 문제를 해결하기 위해, 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 도입했습니다. 우리는 여러 개의 반사실적 예제를 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 보다 견고하게 파악하는 방법을 제안합니다.'}]

- **Performance Highlights**: [{'title': 'Achieving Superior Results in One-Shot Object Recognition and Detection using Joint Neural Networks', 'summary': 'Siamese Neural Networks와 state-of-art multi-box detection을 바탕으로 한 새로운 Joint Neural Networks가 도입되었습니다. 이 접근법은 외부 데이터를 걸치지 않고도 unseen 클래스에서 높은 성능을 발휘하며, MiniImageNet 데이터셋에서 61.41%의 정확도와 COCO 데이터셋에서 학습하여 Pascal VOC 데이터셀에서 47.1%의 mAP를 달성했습니다.'}]



### Scaling Backwards: Minimal Synthetic Pre-training? (https://arxiv.org/abs/2408.00677)
Comments:
          Accepted to ECCV2024

- **What's New**: 최근 MCQ(Multiple Choice Questions) 자동 생성의 교육적 가치를 평가하기 위한 새로운 메트릭, 지식 종속 가능성(KDA, Knowledge Dependent Answerability)이 제안되었습니다. 또한, 딥러닝 모델의 강건성을 강화하는데 유용한 대조적 학습과 반사실적 증강(counterfactual augmentation) 방법론이 소개되었습니다. 마지막으로, 최소한의 합성 데이터셋을 활용한 사전 학습의 가능성이 탐구되어, 단일 프랙탈 이미지를 통한 효과적인 사전 학습 방법이 제시되었습니다.

- **Technical Details**: MCQ 생성을 위한 기존 메트릭 (BLEU, ROUGE, METEOR)은 교육적 가치를 고려하지 않습니다. 이를 해결하기 위해 KDA, KDA_disc, KDA_cont와 같은 새로운 평가 메트릭을 제안하며, 사전 훈련된 언어 모델을 활용해 평가합니다. NLP 태스크에서 대조적 학습과 반사실적 증강을 통해 모델의 강건성을 높이는 방법이 연구되었으며, 이는 분산된 대안들을 집합적 의사결정(collective decision)을 통해 처리합니다. 마지막으로, 단일 프랙탈 이미지를 통한 최소한의 사전 학습 데이터셋(1-parameter Fractal as Data, 1p-frac)으로도 대규모 데이터셋과 유사한 성능을 달성할 수 있음을 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성을 기준으로 전문가가 평가한 결과와 높은 상관관계를 보였습니다. 또한, 반사실적 증강을 통한 모델의 강건성은 다양한 조건에서 일반화 성능을 향상시켰습니다. 단일 프랙탈 이미지를 이용한 사전 학습은 기존 대규모 데이터셋(ImageNet-1k)과 유사한 성능을 발휘하며, 이는 사전 학습 데이터셋의 스케일을 '역으로 확장'할 가능성을 시사합니다.



### ExpertAF: Expert Actionable Feedback from Video (https://arxiv.org/abs/2408.00672)
Comments:
          Technical report

- **What's New**: 이번 연구는 자동으로 다지선다형 문제(MCQ)를 생성하고 평가할 수 있는 새로운 지표인 Knowledge Dependent Answerability(KDA)를 제안합니다. 기존 평가 지표 BLEU, ROUGE, METEOR가 교육적 가치를 겨냥하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: KDA는 학생의 응답을 바탕으로 문제에 대한 답변 가능성을 측정합니다. 이를 위해 인적 조사를 통해 KDA를 측정하고, 사전 학습된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였습니다. 이는 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인적 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 설정에서 사용성 및 전문가가 라벨링한 MCQ 품질 측정에 있어 강한 상관관계를 보였습니다. n-그램 기반의 유사성 메트릭과 결합했을 때 다양한 전문가 라벨링된 MCQ 품질 측정에 대해 강력한 예측력을 보여주었습니다.



### SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglemen (https://arxiv.org/abs/2408.00653)
- **What’s New**: 이번 뉴스레터에서는 자동 MCQ 생성에서 교육적 가치를 평가하는 새로운 메트릭, NLP 태스크에서 모델의 robustness를 강화하기 위한 대조 학습과 counterfactual augmentation, 그리고 단일 이미지에서 빠르고 고품질의 3D 메시 복원을 할 수 있는 새로운 방법인 SF3D에 대해 소개합니다.

- **Technical Details**: 기존 자동 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 교육적 가치를 측정하지 못한다는 문제가 있습니다. 이러한 문제를 해결하기 위해, 우리는 목표 사실에 대한 학생의 지식을 측정하는 능력을 평가하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. KDA_disc와 KDA_cont를 통해 학생들의 문제 해결 행동을 모방함으로써 KDA를 자동으로 근사할 수 있습니다. 또한, 최근의 deep modeller들이 NLP 태스크에서 높은 정확도를 보였지만 spurious pattern에 영향을 받아 robustness가 제한된다는 문제를 해결하기 위해, 우리는 대조 학습과 counterfactual augmentation를 활용하여 robustness를 높이는 방법을 제안했습니다. 우리 방법은 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더욱 robust하게 파악합니다. 마지막으로, SF3D는 단일 이미지에서 빠르고 고품질의 3D 메시를 복원할 수 있는 새로운 방법으로, fast UV unwrapping 기술을 통합하여 직각의 색상을 사용하는 것보다 빠르게 텍스처를 생성합니다. Material parameter와 normal maps를 예측하여 시각적 품질을 향상시키며, delighting 단계를 통해 저주파 조명 효과를 효과적으로 제거합니다. SF3D는 0.5초 만에 고품질의 텍스처드를 생성할 수 있으며, 다양한 조명 환경에서도 쉽게 사용할 수 있습니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 제안된 방법은 집합적 의사 결정을 통해 attribution-based synthesis의 task model bias에 덜 민감하게 반응하며, counterfactual robustness, cross-domain generalization, 그리고 희소 데이터로부터의 generalization 측면에서 의미 있는 개선을 이루었습니다. SF3D는 기존 기술들보다 우수한 성능을 입증했으며, 0.5초 만에 고품질의 3D 텍스처드를 재구성합니다. 두꺼운 resolution을 위한 transformer 구조와 pixel shuffling operation를 통해 높은 해상도와 우수한 시각적 품질을 달성했습니다.



### Towards End-to-End Explainable Facial Action Unit Recognition via Vision-Language Joint Learning (https://arxiv.org/abs/2408.00644)
Comments:
          10 pages, 5 figures, 4 tables

- **What's New**: 이 논문들은 다양한 주제를 다루고 있지만, 모두 각각 새로운 접근 방식을 제안하고 있습니다. 첫 번째 논문은 MCQ(Multiple Choice Questions)의 자동 생성과 평가를 위한 새로운 메트릭인 지식 종속 가능성(KDA)을 도입했습니다. 두 번째 논문은 대조 학습(contrastive learning)과 대안적 증강(counterfactual augmentation)을 활용해 NLP 모델의 강건함을 개선하는 방법을 연구했습니다. 세 번째 논문은 얼굴 행동 단위(Facial Action Units, FAU)를 더 설명 가능한 방법으로 인식하는 VL-FAU라는 새로운 비전-언어 공동 학습 네트워크를 제안했습니다.

- **Technical Details**: [{'Paper': '첫 번째 논문', 'Details': '기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에 초점을 맞추지만, 교육적 가치를 평가하지 않음. 지식 종속 가능성(KDA) 메트릭을 통해 MCQ가 학생의 지식을 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방하여 자동으로 평가합니다.'}, {'Paper': '두 번째 논문', 'Details': '대조 학습과 대안적 증강을 활용하여 모델의 강건성을 개선하며, 특히 인과 관계를 더 잘 감지할 수 있도록 여러 개의 대안적 사실(counterfactuals)을 생성합니다. 이러한 접근 방식은 spurious correlation에 덜 민감하고 다양한 차원에서 성능을 향상시킵니다.'}, {'Paper': '세 번째 논문', 'Details': 'VL-FAU는 비전과 언어의 공동 학습을 통해 설명 가능한 FAU 인식을 제공합니다. 이 모델은 다양한 AU 특징에 대한 세밀한 언어 해석을 생성하여 각 개별 AU의 표현력을 강화하고 AU 사이의 구분을 명확히 합니다.'}]

- **Performance Highlights**: [{'Paper': '첫 번째 논문', 'Details': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서 사용성과 강한 상관관계를 가지며, 여러 전문가들이 라벨링한 MCQ 품질 측정 지표와 결합될 때 강력한 예측력을 가집니다.'}, {'Paper': '두 번째 논문', 'Details': '제안된 방법은 대안적 사실의 집합에 대한 집합적 의사 결정을 통해 서로 다른 도메인에서의 일반화와 희소한 데이터로부터의 일반화에서 상당한 개선을 이룹니다.'}, {'Paper': '세 번째 논문', 'Details': 'DISFA와 BP4D AU 데이터셋에서 기존 최첨단 방법보다 뛰어난 성능을 나타내며, AU 예측에 대한 지역 및 전역 수준의 언어 해석을 제공합니다.'}]



### Deep Learning in Medical Image Classification from MRI-based Brain Tumor Images (https://arxiv.org/abs/2408.00636)
- **What's New**: 새로운 평가 메트릭 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 및 복수 반사적 학습(contrastive learning)을 통해 자동 MCQ 생성 및 NLP 모델의 강건성을 개선하는 연구가 발표되었습니다. 또한, 새로운 MobileNet-BT 모델을 활용한 뇌종양 MRI 이미지 분류 연구도 포함되어 있습니다.

- **Technical Details**: 기존의 n-gram 기반 평가 메트릭이 MCQ의 교육적 가치를 충분히 평가하지 못하는 문제를 해결하기 위해, KDA는 타겟 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. contrastive learning과 counterfactual augmentation 접근법을 통해 NLP 모델의 강건성을 개선하며, 여러 개의 기후적 대조 사실(counterfactuals)을 생성하는 방식을 채택하여 보다 강력한 인과관계 파악을 지원합니다. MRI 이미지 분류에서 MobileNetV2, EfficientNet-B0, ResNet-18, VGG16을 포함한 사전 학습된 모델 및 새로운 MobileNet-BT 모델을 사용하여 뇌종양 이미지를 분류합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. NLP 모델에서는 기후적 대조 사실을 활용하여 반사적 학습을 통해 강건성을 대폭 개선하였습니다. 뇌종양 MRI 이미지 분류에서는 MobileNet-BT 모델이 더 높은 정확도와 F1 점수를 달성하였습니다.



### Empowering Snapshot Compressive Imaging: Spatial-Spectral State Space Model with Across-Scanning and Local Enhancemen (https://arxiv.org/abs/2408.00629)
Comments:
          12 pages,6 figures

- **What's New**: MCQ(Multiple Choice Questions) 생성을 평가하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 도입했습니다. KDA는 기존 BLEU, ROUGE, METEOR와 달리 학생의 대상 사실에 대한 이해도를 평가하는 기능을 가지고 있습니다.

- **Technical Details**: KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 방식으로 KDA를 근사화합니다. Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성 및 KDA와 강한 상관관계를 가지고 있음을 확인했습니다. 또한 이 메트릭들과 n-gram 기반 유사성 메트릭을 결합하면 전문가가 평가한 다양한 MCQ 품질 측정치에 대한 예측력이 강화됨을 보여줍니다.

- **Performance Highlights**: ASLE-SSM(State Space Model with Across-Scanning and Local Enhancement)은 HSI(Hyperspectral Image) 복원에 대해 현존하는 최신 방법들보다 탁월한 성능을 보입니다. ASLE-SSM은 Transformer 기반의 MST보다 2.4배 빠른 추론 속도를 가지며, 0.12M의 파라미터를 절약하여 최저 계산 비용과 파라미터 개수를 달성합니다.



### Are Bigger Encoders Always Better in Vision Large Models? (https://arxiv.org/abs/2408.00620)
- **What's New**: 이번 연구에서는 기존의 자동 MCQ(Multiple Choice Questions) 생성 평가 메트릭의 한계를 극복하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안합니다. 또한, 새로운 KDA_disc와 KDA_cont 메트릭을 통해 실제 강의실 세트에서의 사용성과 교육적 가치를 평가합니다.

- **Technical Details**: BLEU, ROUGE, METEOR과 같은 기존의 MCQ 평가 메트릭은 단순히 n-gram 유사성에 의존하여 교육적 가치를 충분히 반영하지 못합니다. 반면, KDA는 학생의 지식 기반으로 MCQ의 답변 가능성을 평가하며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 통해 이를 자동화하려는 접근 방식입니다.

- **Performance Highlights**: Human evaluation 연구를 통해 KDA_disc와 KDA_cont 메트릭이 실제 전문가들이 평가한 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 이를 통해 n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가가 평가한 MCQ 품질 측정치에 대한 예측력이 높음을 확인하였습니다.



### Harnessing Uncertainty-aware Bounding Boxes for Unsupervised 3D Object Detection (https://arxiv.org/abs/2408.00619)
Comments:
          Preprint, 14 pages, 4 figures, 4 tables

- **What's New**: 이번 연구에서는 자동으로 생성된 다지선다형 문항(MCQ)의 평가를 위한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. KDA는 대상 사실에 대한 학생의 지식 기반으로 문항의 답변 가능성을 측정합니다.

- **Technical Details**: KDA를 측정하기 위해 먼저 학생 응답 데이터를 기반으로 KDA를 정의한 후, 이를 모방하기 위해 사전 학습된 언어 모델을 활용한 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안합니다. 또한, 인간 설문 조사를 통해 이러한 메트릭이 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 입증했습니다.

- **Performance Highlights**: 연구 결과, KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합할 때 여러 전문가가 라벨링한 MCQ 품질 측정에 대해 높은 예측력을 보였습니다.



### Learned Compression of Point Cloud Geometry and Attributes in a Single Model through Multimodal Rate-Contro (https://arxiv.org/abs/2408.00599)
Comments:
          20 pages, 13 figures

- **What's New**: 이 논문은 자동 MCQ(객관식 문제) 생성 평가지표의 한계를 극복하고자 새로운 자동 평가지표인 지식 종속 가능성(KDA)을 제안합니다. KDA는 실제 교육적 가치를 평가하는 데 중점을 두고 있으며, 기존의 n-gram 기반 유사성 평가를 넘어 학생의 문제 해결 행동을 모방하는 방법을 사용하여 평가의 정확도를 높입니다.

- **Technical Details**: 새롭게 제안된 KDA는 학생 응답 기반의 평가를 통해 MCQ의 대답 가능성을 측정합니다. 구체적으로는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방함으로써 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이를 통해 더 현실적이고 교육적인 평가를 가능하게 합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가지는 것으로 나타났습니다. 또한, 이 두 메트릭이 n-gram 기반 유사성 메트릭과 결합되었을 때 다양한 전문가가 라벨을 부여한 MCQ 품질 척도들에 대해 강한 예측력을 보였습니다.



### MUFASA: Multi-View Fusion and Adaptation Network with Spatial Awareness for Radar Object Detection (https://arxiv.org/abs/2408.00565)
Comments:
          Accepted by ICANN 2024

- **What's New**: 이 논문에서는 MCQ(Multiple Choice Questions) 생성의 새로운 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안합니다. 또한 이 논문에서는 NLP 태스크를 위한 대조 학습(contrastive learning) 및 반사실적 증강(counterfactual augmentation)을 활용한 견고한 방법론을 제안하며, 레이더 점군 데이터의 객체 탐지를 개선하는 MUFASA라는 새로운 방법을 소개합니다.

- **Technical Details**: 자동 MCQ 생성에서는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해 KDA를 도입하였습니다. NLP 태스크에서는 반사실적들을 집합적으로 생성하고, 이에 기반한 의사 결정을 통해 인과관계를 보다 견고하게 파악하는 방법이 제안되었습니다. 또 다른 연구에서는 GeoSPA와 DEMVA 모듈을 포함한 MUFASA라는 레이더 점군에 대한 특징 추출 방법이 소개되었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 세팅에서 우수한 사용성을 보였으며, 다양한 전문가가 라벨링한 MCQ 품질 메트릭을 예측하는 데 강력한 예측력을 보였습니다. 대조 학습과 반사실적 증강 접근법은 반사실적 견고성, 도메인 간 일반화, 드문 데이터 일반화 측면에서 상당한 개선을 이루었습니다. MUFASA는 VoD 데이터셋에서 50.24%의 mAP로 레이더 기반 방법 중 최고 성능을 기록했습니다.

- **What's New 3**: 이 논문에서는 MUFASA라는 새로운 방법이 제안되었으며, 이 방법은 GeoSPA와 DEMVA 모듈을 통해 레이더 점군 데이터의 객체 탐지를 개선하는 특징을 지니고 있습니다. 이러한 접근 방식을 통해 매우 희박한 레이더 점군 데이터에서도 높은 탐지 성능을 달성할 수 있음을 보여줍니다.



### Alleviating Hallucination in Large Vision-Language Models with Active Retrieval Augmentation (https://arxiv.org/abs/2408.00555)
- **What's New**: 최근 논문에서 여러 가지 혁신적인 새로운 평가 및 학습 방법론을 소개했습니다. 첫 번째로, 자동화된 다지 선다형 질문(MCQ) 생성의 평가를 위해 기존 BLEU, ROUGE, METEOR와 같은 메트릭이 교육적 가치를 무시한다는 문제를 해결하기 위해 새로운 지식 종속 가능성(KDA) 메트릭을 제안했습니다. 두 번째로, NLP 태스크에서의 deep model의 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용했습니다. 마지막으로, 대규모 비전-언어 모델(LVLMs)의 환각현상(hallucination)을 줄이기 위해 외부 지식 자원을 통한 액티브 리트리벌-보강 모델(ARA)을 제안했습니다.

- **Technical Details**: 지식 종속 가능성(KDA)를 통해 자동 MCQ의 대답 가능성을 평가했습니다. KDA를 불러오는 과정에서는 인간의 설문 응답을 기초로 합니다. 이후 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 개발해 미리 학습된 언어 모델을 통해 학생들의 문제해결 행동을 모방했습니다. NLP 퍼포먼스 개선을 위해서는 counterfactual을 생성하여 튜닝된 모델의 스푸리어스 패턴을 줄였습니다. 액티브 리트리벌-보강 모델에서는 LVLMs의 환각현상을 줄이기 위해 (i) 이미지 계층 구조를 분해하고 (ii) 효과적인 리트리벌 방법을 선정하며 (iii) 리트리벌 시간을 조정하는 세 가지 주요 차원을 고려한 접근을 취했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 세팅과 강한 상관관계를 보였으며, 전문가가 표기한 MCQ 품질 지표와도 높은 예측력을 보였습니다. 새로운 대조 학습 방법은 이종 도메인 일반화 및 드문 데이터에서의 일반화 측면에서 크게 향상된 성능을 달성했습니다. ARA 모델은 세 가지 널리 사용되는 LVLM(LLaVA-1.5, Qwen-VL, mPLUG-Owl2)을 사용하여 네 가지 벤치마크에서 환각현상을 효과적으로 줄이는 성과를 보였습니다.



### Mitigating Multilingual Hallucination in Large Vision-Language Models (https://arxiv.org/abs/2408.00550)
- **What's New**: 최근 자동 생성 다지선다형 문제(MCQ) 평가에 있어서 교육적 가치를 측정하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)가 제안되었습니다. 이와 더불어, 대조 학습과 반사실적 증강을 활용한 NLP 딥 모델의 강건성 향상 방법과 LVLM(대형 비전-언어 모델)의 다중언어 환각 문제를 완화하는 새로운 프레임워크가 소개되었습니다.

- **Technical Details**: KDA는 학생의 지식 기반으로 MCQ의 답변 가능성을 평가합니다. 대조 학습 및 반사실적 증강을 통한 NLP 모델의 반사실적 강건성과 도메인 간 일반화를 목표로 합니다. 다중언어 환각 제거(MHR) 프레임워크는 다양한 언어에서의 환각을 줄이기 위해 다단계 접합 방법과 직접 선호 최적화를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭이 실제 강의실 세트에서의 사용성과 높은 상관성을 보였습니다. LVLMs의 MHR 프레임워크는 POPE 벤치마크에서 평균 19.0%의 정확도 향상을 달성했습니다. 다양한 언어에서의 강건성과 일반화 능력도 현저히 개선되었습니다.



### How Effective are Self-Supervised Models for Contact Identification in Videos (https://arxiv.org/abs/2408.00498)
Comments:
          15 pages, 6 figures

- **What's New**: 최근 NLP 태스크에서 deep model들이 인공 지능 성능 측면에서 사람을 능가했음에도 불구하고, 비정상적 패턴에 의존하여 안정성이 부족한 문제를 보완하기 위해 대조적 학습 (Contrastive Learning) 및 비대상 강화 (Counterfactual Augmentation) 방법을 적용했습니다. 또한, 새로운 MCQ 자동 생성 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. 이는 학생의 학습 평가 요소를 고려한 새로운 접근입니다.

- **Technical Details**: 대조적 학습과 비대상 강화 방법을 통해 데이터의 비정상적 상관관계 (Spurious Correlations)를 해결하고자 했습니다. 또한, 여러 비대상 데이터 집합을 생성하여 이 데이터 분포 기반 의사 결정을 통해 각 용어의 인과 관계를 더 robust하게 감독했습니다. KDA 메트릭의 경우, 학생 응답 기반의 인간 설문조사를 통해 지식 종속 대답 가능성을 측정하였으며, 이를 위하여 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하였습니다. 이는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 1) 비대상 견고성 (counterfactual robustness), 2) 도메인 간 일반화 (cross-domain generalization), 3) 희소 데이터 (scarce data)로부터의 일반화 측면에서 상당한 성능 향상을 나타냈습니다. KDA_disc와 KDA_cont는 실제 교실 세팅에서의 사용성 및 다양한 전문가가 라벨링한 MCQ 품질 지표를 예측하는 강력한 기능을 보여주었습니다.



### SegStitch: Multidimensional Transformer for Robust and Efficient Medical Imaging Segmentation (https://arxiv.org/abs/2408.00496)
- **What's New**: [{'MCQ Generation': '자동으로 Multiple Choice Questions (MCQ)를 생성하는 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안했습니다. 이는 학생이 목표 사실에 대해 지식을 평가하는 능력을 측정합니다.'}, {'Robust NLP Models': '대조 학습과 반사실적 증강을 활용하여 NLP 모델의 견고성을 강화하는 새로운 방법을 제안했습니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악합니다.'}, {'Medical Imaging Segmentation': '트랜스포머와 denoising ODE 블록을 통합한 새로운 의료 영상 분할 아키텍처 SegStitch를 소개했습니다. 이는 mDSC 메트릭에서 최대 11.48%까지 향상되었으며, 모델 파라미터 수와 FLOPs를 각각 36.7%와 10.7% 줄였습니다.'}]

- **Technical Details**: [{'MCQ Generation': 'KDA는 학생 응답 기반으로 평가되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 pretrained language models를 활용해 학생의 문제 해결 행동을 모방합니다. 이 메트릭은 데이터셋의 n-gram 유사성과 결합될 때 강력한 예측 능력을 보입니다.'}, {'Robust NLP Models': '단순한 intristic spurious correlations를 피하기 위해 여러 개의 counterfactual을 생성하고, collective decision-making 방식을 도입해 더 robust한 인과 관계 추론을 수행합니다.'}, {'Medical Imaging Segmentation': 'SegStitch는 3D 볼륨을 전체 입력으로 사용하는 대신 축상 패치와 패치 별 쿼리를 사용하여 의미 일관성을 보장합니다. BTCV 및 ACDC 데이터셋에서 extensive experiments를 통해 각각 11.48%와 6.71%의 향상을 달성했습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont 메트릭은 실제 교실 환경에서의 사용성과 인간 평가에서 강한 상관관계를 보였으며, 다양한 전문가가 표기한 MCQ 품질 측정 기준을 예측하는 데 강력한 predictive power를 보였습니다.'}, {'Robust NLP Models': '집합적 의사 결정 방법은 attribution-based synthesis에서 발생하는 task model 편향에 덜 민감하고, 대조적 강건성, cross-domain generalization, 희소 데이터 일반화에서 유의미한 성능 향상을 달성했습니다.'}, {'Medical Imaging Segmentation': 'SegStitch는 UNETR에 비해 파라미터 수를 36.7%, FLOPs를 10.7% 감소시키면서 효율성을 입증했고, 실질적인 임상 환경에서 적용될 가능성이 높습니다.'}]



### Explainable Emotion Decoding for Human and Computer Vision (https://arxiv.org/abs/2408.00493)
Comments:
          This work has been accepted to be presented to The 2nd World Conference on eXplainable Artificial Intelligence (xAI 2024), July 17-19, 2024 - Malta

- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하기 위한 새롭고 혁신적인 메트릭 'Knowledge Dependent Answerability (KDA)'를 도입했습니다. 또한, 최근 NLP 태스크에서 deep models의 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법론을 제안하였습니다. 마지막으로, 신경과학 분야에서 Explainable AI (XAI) 기술을 활용하여 인간과 컴퓨터 비전 시스템을 병렬로 분석하는 연구를 발표하였습니다.

- **Technical Details**: [{'paper': 'MCQ Generation', 'details': '기존 BLEU, ROUGE, METEOR 메트릭이 데이터셋의 골드 샘플과의 n-gram 유사성에 초점을 맞추는 한계를 극복하기 위해 KDA 메트릭을 제안했습니다. KDA 메트릭은 MCQ의 대답 가능성을 측정하고 학생이 특정 사실에 대한 지식을 평가하는 능력을 평가합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'paper': 'NLP Robustness', 'details': "최근 deep models의 spurious patterns 의존성을 극복하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안했습니다. 'Set of counterfactuals'을 생성하고 집합적 의사 결정을 통해 인과관계를 파악하는 방법으로, 기존 방법의 spurious correlation 문제를 해결합니다."}, {'paper': 'Explainable AI in Neuroscience', 'details': '신경과학 분야에서 Explainable AI (XAI) 기술을 활용해 인간과 컴퓨터 비전 시스템을 병렬로 분석했습니다. fMRI 데이터와 영화 프레임을 기반으로 두 ML 모델을 훈련하고 설명하였습니다. 인간 비전의 경우 fMRI 데이터를 감정 레이블과 연결하고, 컴퓨터 비전의 경우 영화 프레임의 픽셀 수준 설명을 제공합니다.'}]

- **Performance Highlights**: [{'paper': 'MCQ Generation', 'highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있다는 것을 증명했습니다. 전문가가 라벨링한 MCQ 품질 측정치에서 높은 예측 능력을 보였습니다.'}, {'paper': 'NLP Robustness', 'highlights': '제안된 접근 방식이 attribution-based synthesis에 대한 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 scarcity한 데이터로부터의 일반화에서 significant한 개선을 달성했습니다.'}, {'paper': 'Explainable AI in Neuroscience', 'highlights': '인간 주의 데이터와 CNNs의 주의 매칭도에 기반한 신경 패턴을 발견함으로써 CV 모델과 인간 시각 시스템을 연결하는 데 유용한 정보를 제공했습니다.'}]



### Multi-label Sewer Pipe Defect Recognition with Mask Attention Feature Enhancement and Label Correlation Learning (https://arxiv.org/abs/2408.00489)
Comments:
          Accepted by the Journal of Computing in Civil Engineering

- **What's New**: MCQ와 하수관 파이프 결함 인식을 위한 새로운 평가 메트릭 및 모델이 제안되었습니다. 이 논문들은 기존 메트릭의 단점을 극복하고 교육적 가치와 모델의 성능을 높이기 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: [{'title': 'Multiple Choice Question (MCQ) 생성 평가 메트릭', 'contents': ['기존 BLEU, ROUGE, METEOR 메트릭은 MCQ의 교육적 가치를 반영하지 못합니다.', '새로운 메트릭, KDA (Knowledge Dependent Answerability)를 제안하여 MCQ의 대답 가능성을 측정합니다.', 'KDA_disc와 KDA_cont라는 자동 평가 메트릭을 도입하여, 사전 훈련된 언어 모델을 사용해 학생의 문제 해결 행동을 모사합니다.']}, {'title': '강력한 NLP 모델의 Robustness 강화', 'contents': ['기존 deep model들은 spurious pattern에 의해 robustness가 제한됩니다.', '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 통해 모델의 인과성을 강화합니다.', '여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 비편향적인 인과관계 학습을 유도합니다.']}, {'title': '하수관 파이프 결함 인식을 위한 모델', 'contents': ['Mask attention 기법을 사용하여 여러 결함을 동시에 인식할 수 있는 다중 레이블 파이프 결함 인식 방법을 제안합니다.', '제안된 MA-Q2L (Mask Attention guided Q2L)은 이미지의 국소 공간 정보에 집중하여 분류 성능을 향상시킵니다.', '비대칭 손실 함수(asymmetric loss function)를 이용해 클래스 불균형(long-tail problem) 문제를 해결합니다.']}]

- **Performance Highlights**: [{'title': 'MCQ 평가 메트릭', 'contents': ['KDA_disc와 KDA_cont 메트릭은 실제 강의실 세팅에서 전문가가 라벨링한 사용성과 강한 상관관계를 보였습니다.', 'n-gram 기반 유사성 메트릭과 결합할 경우, 다양한 전문가 라벨 MCQ 품질 측정에 높은 예측 능력을 보였습니다.']}, {'title': 'NLP 모델의 Robustness', 'contents': ['대조적 학습과 반사실적 증강을 적용한 결과, 다양한 차원에서 유의미한 성능 향상을 달성했습니다.', '임무 모델의 편향에 덜 민감한 결과를 보였습니다.']}, {'title': '하수관 파이프 결함 인식 모델', 'contents': ['제안된 MA-Q2L 모델은 Sewer-ML 전체 데이터셋에서 11.87%의 F2 metric 향상을 보였습니다.', '1/16의 데이터셋만으로도 상태 검색 성능이 우수한 결과를 도출했습니다.']}]



### Image Super-Resolution with Taylor Expansion Approximation and Large Field Reception (https://arxiv.org/abs/2408.00470)
- **What's New**: 이 논문은 MCQ (Multiple Choice Questions)의 자동 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(KDA; Knowledge Dependent Answerability)을 제안합니다. 이는 학생이 관련된 목표 사실에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 메트릭은 생성된 MCQ와 데이터셋의 골드 샘플간의 n-gram 기반 유사성에 집중하여 교육적 가치를 간과합니다. 반면, 우리가 제안한 KDA는 학생 반응을 통해 MCQ의 효과를 측정하며, 이를 바탕으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입합니다. 이는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또, Blind Super-Resolution(SR)과 관련된 논문은 자가 유사성 계산의 높은 연산 복잡성을 다루기 위한 STEA(Second-order Taylor Expansion Approximation)와 MLFR(Multi-Scale Large Field Reception) 설계를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 높은 상관관계를 나타냈습니다. 추가로 n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가 라벨링 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다. Blind SR에서 제안된 LabNet과 RealNet은 각각 실험실과 실제 환경에서 뛰어난 성능을 발휘합니다.



### Reenact Anything: Semantic Video Motion Transfer Using Motion-Textual Inversion (https://arxiv.org/abs/2408.00458)
Comments:
          Preprint. All videos in this paper are best viewed as animations with Acrobat Reader by pressing the highlighted frame of each video

- **What's New**: 새로운 MCQ 자동 생성 평가 메트릭(KDA), 더 robust한 NLP 모델을 위한 contrastive learning 및 counterfactual augmentation 방법, 그리고 pre-trained image-to-video 모델을 이용한 high temporal motion granularity 비디오 생성 기법이 소개되었습니다.

- **Technical Details**: [{'Automatic MCQ Generation': 'KDA(Knowledge Dependent Answerability) 메트릭을 제안하여, 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 pre-trained language model을 이용하여 학생들의 문제 해결 행동을 모사합니다.'}, {'NLP Model Robustness': '기존 방법이 spurious correlation에 제한된 것과 다르게, 다양한 counterfactual을 생성하고, 이들의 집합적 의사 결정(collective decisions)을 통해 단어 간의 인과관계를 더 robust하게 supervise하는 방법을 제안합니다.'}, {'Video Generation and Editing': '텍스트 기반이 아닌 사전 학습된 이미지-비디오 모델을 사용하여 목표 객체나 장면의 외형을 정확히 보존하면서 모션을 디스엔텐글하는 방법을 제안합니다. 모션 레퍼런스 비디오를 기반으로 최적화된 embedding을 다양한 타겟 이미지에 적용하여 의미적으로 유사한 모션을 가지는 비디오를 생성합니다.'}]

- **Performance Highlights**: [{'Automatic MCQ Generation': 'KDA_disc와 KDA_cont 메트릭이 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지고 있으며, 기존 n-gram 기반 평가 메트릭과 결합될 때 높은 예측 성능을 보여주었습니다.'}, {'NLP Model Robustness': '이 방법론은 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터를 통한 일반화에서 상당한 성능 향상을 보여주었습니다.'}, {'Video Generation and Editing': 'semantic video motion transfer 작업에서 기존 메서드 대비 높은 성과를 보였습니다.'}]



### Focus, Distinguish, and Prompt: Unleashing CLIP for Efficient and Flexible Scene Text Retrieva (https://arxiv.org/abs/2408.00441)
Comments:
          Accepted by ACM MM 2024

- **What's New**: We have explored three recent papers that bring forth novel approaches to different AI challenges. The first paper addresses the limitations of existing Multiple Choice Questions (MCQ) generation evaluation metrics by proposing a new Knowledge Dependent Answerability (KDA) metric. The second paper enhances deep model robustness in Natural Language Processing (NLP) tasks through contrastive learning and counterfactual augmentation. Finally, the third paper introduces an Optical Character Recognition (OCR)-free model for Scene Text Retrieval (STR), leveraging a combination of Contrastive Language-Image Pre-training (CLIP) and a novel 'Focus, Distinguish, and Prompt' (FDP) approach.

- **Technical Details**: 1. **Knowledge Dependent Answerability (KDA)**: This new metric measures the answerability of an MCQ given knowledge of the target fact, using KDA_disc and KDA_cont based on pre-trained language models to simulate student behavior.
2. **Contrastive Learning and Counterfactual Augmentation**: This approach creates multiple counterfactual examples and uses collective decision-making to reduce the sensitivity to spurious patterns, improving robustness in diverse dimensions like counterfactual robustness, cross-domain generalization, and generalization from scarce data.
3. **CLIP and FDP for Scene Text Retrieval**: FDP focuses on text within images by applying rough text localization and distinguishing content words from function words. A semantic-aware prompting scheme converts query text into a learnable prompt for efficient and flexible text retrieval.

- **Performance Highlights**: 1. **KDA Metrics**: Human studies showed that KDA_disc and KDA_cont metrics have strong correlations with KDA and classroom usability, providing a significant predictive power for various expert-labeled MCQ quality measures.
2. **Counterfactual Augmentation Approach**: Empirical results indicate significant improvements in robustness and generalization over traditional augmentation methods.
3. **FDP Model**: Compared to existing OCR-based methods, FDP enhances the inference speed by 4 times and improves retrieval accuracy by 4.37% in the IIIT-STR benchmark. Additional experiments validate its effectiveness in handling diverse forms of query text.



### MonoMM: A Multi-scale Mamba-Enhanced Network for Real-time Monocular 3D Object Detection (https://arxiv.org/abs/2408.00438)
- **What's New**: 이번 논문에서는 MCQ 생성에서 형식적 유사성만을 평가하는 기존의 BLEU, ROUGE, METEOR 같은 메트릭 대신, 학습자의 대상 사실에 대한 지식을 평가할 수 있는 새로운 평가 메트릭, '지식 종속 가능성 (Knowledge Dependent Answerability; KDA)'을 제안합니다.

- **Technical Details**: KDA는 학생의 대상 사실에 대한 지식을 기준으로 MCQ의 답변 가능성을 측정합니다. 구체적으로는, 사람들의 응답을 기반으로 KDA를 측정한 후, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델 (Pre-trained Language Model)을 활용하여 학생들의 문제 해결 행동을 모방합니다. 이를 통해 전통적인 n-gram 기반의 메트릭과 KDA를 결합하여 MCQ의 질을 예측할 수 있도록 했습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실에서 사용성과 강한 상관관계를 가지며, 전문가가 평가한 다양한 MCQ 품질 지표에 예측 능력이 뛰어나다는 것을 보였습니다.



### CARMIL: Context-Aware Regularization on Multiple Instance Learning models for Whole Slide Images (https://arxiv.org/abs/2408.00427)
- **What's New**: 최근 딥러닝 모델이 NLP에서 사람보다 높은 정확도를 보여주고 있음에도 불구하고, spurious 패턴에 의존하여 robustness에 한계가 있는 것으로 보고되고 있습니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 새로운 방법을 제안합니다. 또한, 종양학적 Whole Slide Images(WSIs)를 조금 더 정밀하게 분석할 수 있도록 Context-Aware Regularization (CARMIL)을 도입하고, 이를 통해 암 예후 예측 모델의 성능을 강화하는 연구도 소개되었습니다.

- **Technical Details**: [{'Automatic MCQ Generation': '기존의 MCQ 평가 메트릭은 BLEU, ROUGE, METEOR와 같이 n-gram 기반 유사성을 측정하여 교육적 가치를 고려하지 않습니다. 이를 극복하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안하며, 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방함으로써 KDA를 approximating 하는 KDA_disc와 KDA_cont를 소개합니다.'}, {'Robustness in NLP': '기존의 counterfactual 기반 방법들은 spurious 패턴에 의해 영향을 받는 반면, 제안된 방법은 여러 개의 counterfactual을 생성하고 하나의 집합적인 의사 결정을 통해, 각 용어의 인과관계를 보다 robust하게 파악합니다.'}, {'Cancer Prognosis from WSIs': '기존 MIL 모델은 이미지 패치를 독립적으로 가정하여 공간적 컨텍스트를 잃습니다. 이에 대해 CARMIL이라는 정규화 기법을 통해 공간적 지식을 결합하고, 새로운 정량적 평가 지표인 DeltaCon을 도입하여 모델의 Context-Awareness를 측정합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 사용성이 높음을 증명했으며, 전문가가 라벨링한 MCQ 품질 측정에서도 예측력이 강함을 보여주었습니다.'}, {'NLP Task Robustness': '대조 학습과 counterfactual augmentation을 통해 다양한 차원에서 큰 성능 향상, 특히 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터로부터의 일반화 성능이 개선되었습니다.'}, {'Context-Aware Regularization': 'Glioblastoma와 대장암 데이터셋에 대한 생존 분석에서 CARMIL이 원래 context-independent한 MIL 모델들을 Context-Aware 모델로 전환하고 C-index를 향상시키는 것을 입증했습니다.'}]



### MPT-PAR:Mix-Parameters Transformer for Panoramic Activity Recognition (https://arxiv.org/abs/2408.00420)
- **What's New**: 새로운 자동 평가 메트릭, Knowledge Dependent Answerability (KDA),를 제안하여 지문 기반의 MCQ 생성 평가 방법을 개선했습니다. 이 메트릭은 학생의 대상 사실(target fact)에 대한 지식을 기반으로 MCQ의 대답 가능성(answerability)을 평가합니다. 또한, KDA를 측정할 수 있는 두 가지 자동화된 평가 메트릭, KDA_disc와 KDA_cont를 도입했습니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 한 human survey를 통해 측정됩니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델(pre-trained language models)을 사용하여 학생들의 문제 해결 행동을 모방합니다. 이를 통해 학습 세트에서 대답 가능성을 자동으로 평가할 수 있습니다. 또한, 기존의 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링 한 MCQ 품질 측정치를 예측하는 데 강력한 예측력을 가집니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 (1) KDA와의 강한 상관관계, (2) 실제 강의 세트에서 전문가가 라벨링한 사용성과의 강한 상관관계를 보였습니다. 이 메트릭들은 MCQ의 대답 가능성과 교육적 유효성을 평가하는 데 유용하게 사용될 수 있습니다.



### Towards Reliable Advertising Image Generation Using Human Feedback (https://arxiv.org/abs/2408.00418)
Comments:
          ECCV2024

- **What's New**: 교사들이 학생 평가에 사용하는 시간을 줄이기 위해 자동 Multiple Choice Questions (MCQ) 생성이 가능한 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안합니다. 또한 강화된 로버스트 모델을 위해 대조 학습(contrastive learning)과 반팩추얼 증강(counterfactual augmentation)을 활용하는 방안을 제시하고, 이커머스 환경에서 안정적인 광고 이미지 생성 솔루션을 개발했습니다.

- **Technical Details**: [{'MCQ Generation': '기존 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에만 집중해 교육적 가치를 평가하지 못했습니다. 이를 해결하기 위해 제안된 KDA는 학생의 답변 가능성을 측정하며 새로운 KDA_disc와 KDA_cont 메트릭을 통해 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.'}, {'Robustness in NLP': '최근 딥러닝 모델들은 NLP 과제에서 높은 정확성을 보였으나, spurious 패턴에 의존해 로버스트성이 제한되었습니다. 반팩추얼을 자동으로 생성하고 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과관계를 파악하는 방법을 제안합니다.'}, {'E-commerce Advertising': '광고 이미지 생성에서 Reliable Feedback Network (RFNet)을 활용해 생성된 이미지를 검사하고, Recurrent Generation 과정과 일관된 조건 정규화(Consistent Condition regularization)를 통한 미세 조정으로 생성된 이미지의 가용률을 높였습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 레이블한 다양한 MCQ 품질 지표에 대한 예측력이 뛰어남을 확인했습니다.'}, {'Robustness in NLP': '제안된 방법론은 다양한 차원에서의 성능 향상을 보였으며, 반팩추얼 로버스트성, 교차 도메인 일반화, 적은 데이터로부터의 일반화에서 의미 있는 성능 향상을 달성했습니다.'}, {'E-commerce Advertising': 'Recurrent Generation과 RFNet의 결합은 가용한 광고 이미지의 수를 크게 증가시켰고, 일관된 조건 정규화를 통해 미세 조정한 후에도 시각적 매력을 유지하면서 생성 프로세스의 효율성이 크게 향상되었습니다.'}]



### Deepfake Media Forensics: State of the Art and Challenges Ahead (https://arxiv.org/abs/2408.00388)
- **What's New**: 최신 연구는 지식 종속 가능성 (KDA)이라는 새로운 자동 평가 메트릭을 도입하여 다지선다형 질문(MCQ)의 교육적 유효성을 평가하는 방법을 제안했다. 기존 평가 메트릭은 교육적 가치를 반영하지 못했으며, 이 새로운 메트릭은 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정한다.

- **Technical Details**: KDA는 학생 응답을 통해 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 이를 근사화한다. 이 메트릭은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: Human studies를 통해 KDA_disc 및 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, n-gram 기반의 유사성 메트릭과 결합되었을 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강력한 예측력을 갖고 있음을 보였다.



### Few-shot Defect Image Generation based on Consistency Modeling (https://arxiv.org/abs/2408.00372)
- **What's New**: 이번 논문에서는 기존의 데이터셋 내에서 단어 유사성만 측정하던 BLEU, ROUGE, METEOR 메트릭을 대체할 수 있는 새로운 평가 방법, 지식 종속 가능성(KDA, Knowledge Dependent Answerability)을 제안합니다. 이 접근법은 학생들의 문제 해결 행동을 모방하기 위해 사전 훈련된 언어 모델을 활용하여 MCQ(Multiple Choice Questions)의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생 설문 조사를 통해 획득한 데이터를 바탕으로 MCQ의 대답 가능성을 측정합니다. 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하며, 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. 이를 통해 MCQ의 교육적 효용성을 평가합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계가 있음을 입증했습니다. 두 메트릭은 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강한 예측력을 보였습니다.



### High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior (https://arxiv.org/abs/2408.00361)
Comments:
          ECCV2024

- **What's New**: 이번 연구에서는 여러 프레임과 고해상도 등의 '풍부한 자원' 입력 없이도 단일 저해상도 영상만으로도 높은 정밀도의 깊이 추정을 가능하게 하는 새로운 자기 지도(self-supervised) 단안 카메라(depth estimation) 기법(RPrDepth)을 제안합니다. 이 접근법은 학습 단계에서 풍부한 자원 데이터를 사전 정보로 활용해 깊이 추정을 수행합니다. 이를 통해 실제 환경에서의 적용 가능성을 높였습니다.

- **Technical Details**: RPrDepth 모델은 고해상도 및 다중 프레임 입력을 훈련 단계에서 사전 정보로 사용합니다. 이 정보는 오프라인에서 참조 특징(reference features)으로 추출됩니다. 추론 단계에서는 단일 저해상도 이미지의 유사한 픽셀을 참조 데이터셋에서 찾아 깊이 추정을 수행합니다. 또한, 풍부한 자원 입력에서 얻은 깊이 예측을 유사한 조건의 입력에 대해 의사 레이블(pseudo label)로 사용하여 Rich-resource Guided Loss를 도입했습니다. 이 방법은 집합적 의사 결정(collective decisions)을 통한 일관성(consistency) 유지를 강조합니다.

- **Performance Highlights**: 실험 결과, 제안된 RPrDepth 모델은 단일 저해상도 입력만으로도 다른 단일 이미지 모델을 능가했으며, 심지어 풍부한 자원 기반 모델과 비교해도 유사하거나 더 나은 성능을 달성했습니다. 이는 실제 응용에서의 깊이 추정 방법의 실용성을 크게 높입니다.



### DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training (https://arxiv.org/abs/2408.00355)
Comments:
          Accepted by ACMMM2024

- **What's New**: 이번 연구는 Transformer 기반의 자동 텍스트 스포팅(text spotting) 아키텍처에서 바이어파트 그래프 매칭(bipartite graph matching)에 기인한 불안정성을 해결하기 위해 새로운 디노이즈 트레이닝(denoising training) 방법인 DNTextSpotter를 제안합니다. 이 방법은 임의의 형상 텍스트 스포팅에 적합하며, 기존 방식의 단점을 극복하도록 고안되었습니다.

- **Technical Details**: DNTextSpotter는 비지어 제어점(Bézier control points)과 텍스트 문자를 사용하여 노이즈가 추가된 쿼리를 초기화합니다. 이를 위해 텍스트 내용을 포지션과 맞추기 위해 마스크된 문자 슬라이딩 방법(masked character sliding method)을 적용합니다. 또한, 백그라운드 문자 분류를 위한 추가 손실 함수(loss function)를 사용하여 모델의 인식 능력을 향상시킵니다.

- **Performance Highlights**: 이 새로운 방법은 여러 벤치마크에서 뛰어난 성능을 보였으며, 특히 Inverse-Text 데이터셋에서 11.3%의 성능 향상을 달성했습니다. Total-Text 및 SCUT-CTW1500에서는 각각 2.0% 및 2.1% 향상을 보여주었습니다. ViTAEv2-S 백본을 사용할 때는 모든 메트릭에서 성능이 더욱 향상되었습니다.



### Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion (https://arxiv.org/abs/2408.00352)
- **What's New**: ['이 논문은 Multiple Choice Questions (MCQ)의 자동 생성에서 기존 평가 메트릭의 한계를 언급하며, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 평가 기준을 제안하였습니다. 이 메트릭은 MCQ의 답변 가능성을 측정하여 교육적 가치를 평가합니다.', 'T2M 모델의 악의적인 공격을 위한 ALERT-Motion 프레임워크를 소개합니다. 이 방법은 LLM(Large Language Model)을 활용하여 자연스러운 최적의 공격 텍스트를 생성하며, 기존 방법보다 높은 공격 성공률과 은밀한 공격 프롬프트를 달성합니다.']

- **Technical Details**: ['KDA 메트릭은 학생들의 응답을 바탕으로 MCQ의 답변 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안합니다. 이들은 미리 훈련된 언어 모델(Pre-trained Language Models)을 이용하여 학생의 문제 해결 행동을 모방합니다.', 'ALERT-Motion 프레임워크는 두 가지 주요 모듈로 구성됩니다. 1) 적응형 디스패칭 모듈 (Adaptive Dispatching Module)은 LLM 기반 에이전트가 공격 프롬프트를 자동으로 생성 및 수정합니다. 2) 멀티모달 정보 대조 모듈 (Multimodal Information Contrastive Module)은 의미론적으로 관련된 움직임 정보를 추출하여 공격 프롬프트 생성에 도움을 줍니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont은 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합될 때 전문가가 레이블링한 다양한 MCQ 품질 측정에 대해 높은 예측력을 가집니다.', 'ALERT-Motion은 두 개의 인기 T2M 모델에서 실행되어 더 높은 공격 성공률과 더 자연스럽고 탐지하기 어려운 공격 프롬프트를 생성했습니다.']



### Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos (https://arxiv.org/abs/2408.00351)
Comments:
          ECCV 2024 accepted

- **What's New**: 이 논문에서는 자동 주관식 문항(MCQ) 생성의 교육적 가치를 평가하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 생성된 MCQ와 골드 샘플 간의 단어 유사도만 측정하여 교육적 가치를 평가하지 못합니다. KDA는 대상 사실에 대한 학생의 지식을 통해 MCQ의 대답 가능성(answerability)을 평가합니다.

- **Technical Details**: KDA는 학생 설문조사를 기반으로 측정하고, 이를 근사하기 위해 사전 학습된 언어 모델을 활용한 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안합니다. 이 모델들은 학생들의 문제 해결 행동을 모방하여 MCQ의 대답 가능성을 평가합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 실제 교실 환경에서 사용성과 강한 상관관계를 가지는 것으로 나타났습니다. 또한 이 메트릭들은 n-gram 기반 유사성 메트릭과 결합될 때 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대해 강한 예측력을 보여줬습니다.



### A Simple Background Augmentation Method for Object Detection with Diffusion Mod (https://arxiv.org/abs/2408.00350)
- **What's New**: 새로운 자동 평가 지표인 Knowledge Dependent Answerability (KDA)을 제안하여, MCQ의 대답 가능성(answerability)과 학생의 해당 사실에 대한 지식을 평가할 수 있게 하였습니다. 이 지표는 기존의 평가 메트릭들이 무시했던 교육적 가치를 반영합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정하며, 핵심 사실의 학습 평가 성능을 나타냅니다. 또한, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대한 예측력이 강력함을 입증했습니다.



### Advancing Medical Image Segmentation: Morphology-Driven Learning with Diffusion Transformer (https://arxiv.org/abs/2408.00347)
Comments:
          Accepted in BMVC 2024

- **What's New**: 새로운 MCQ 생성 평가 메트릭인 Knowledge Dependent Answerability (KDA) 제안. 이는 MCQ가 학생의 대상 지식(fact)에 대한 이해를 평가하는 능력을 측정합니다.

- **Technical Details**: 기존 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에만 초점을 맞추어 교육적 가치를 반영하지 못하고 있습니다. KDA는 학생 응답을 바탕으로 MCQ의 대답 가능성을 측정하며, 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 해당 메트릭들은 예비 학습된 언어 모델을 활용하여 학생의 문제 해결 습관을 모방합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 KDA 및 실제 교실 환경에서의 사용성과 높은 상관관계를 보였습니다. n-gram 기반 유사성 메트릭과 결합될 때 다양한 전문가가 평가한 MCQ의 품질 측정에 대해 강력한 예측력을 나타냈습니다.



### DistillGrasp: Integrating Features Correlation with Knowledge Distillation for Depth Completion of Transparent Objects (https://arxiv.org/abs/2408.00337)
Comments:
          10 pages, 5 figures

- **What's New**: 기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR가 단순히 단어단위의 유사성만을 측정하고 교육적 가치를 평가하지 못하는 문제점을 해결하기 위해, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, 최근 NLP 태스크에서 높은 정확성을 보이는 모델들도 spurious pattern에 의존하는 문제로 인해 robustness가 부족한 문제를 contrastive learning (대조 학습) 및 counterfactual augmentation을 통해 개선하려는 접근을 소개합니다. 투명한 객체의 깊이 데이터를 정확히 캡처하지 못하는 RGB-D 카메라 문제를 해결하기 위해, DistillGrasp라는 효율적인 depth completion 네트워크를 제안했습니다.

- **Technical Details**: 새롭게 제안된 KDA 메트릭은 학생의 목표 사실(knowledge of the target fact)에 대한 지식을 평가하기 위해 최초로 제안되었습니다. KDA를 측정하기 위해 human survey 응답을 바탕으로 하는 방법을 제시했으며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 이를 자동으로 approximation하는 방법을 제안했습니다. NLP robustness를 향상시키기 위해 여러 개의 counterfactual을 생성하여 collective decision을 기반으로 각 용어의 인과관계를 robust하게 감독하는 방법을 제안했습니다. DistillGrasp는 teacher branch와 student branch로 나누어 지식 증류(knowledge distillation)을 사용하여 모델의 효율성을 높이는 방법입니다. 특히, teacher branch에서는 position correlation block (PCB)을 사용하고, student branch에서는 consistent feature correlation module (CFCM)를 사용합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서 usability와 매우 높은 상관관계를 보여줬습니다. 또한 n-gram 기반의 유사성 메트릭과 결합했을 때, 다양한 전문가가 라벨링한 MCQ 품질 지표에 대한 예측력이 뛰어났습니다. NLP 모델의 robustness를 높이기 위해 제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data로부터의 일반화에서 중요한 향상을 보였습니다. DistillGrasp는 ClearGrasp 데이터셋에서 최신 방법들보다 정확도와 일반화에서 뛰어난 성능을 보였으며, UR10e 로봇에 적용하여 실제 환경에서도 효과성과 robustness를 입증받았습니다.



### Vision-based Wearable Steering Assistance for People with Impaired Vision in Jogging (https://arxiv.org/abs/2408.00332)
Comments:
          Accepted to ICRA 2024

- **What's New**: 최근 여러 논문이 NLP 태스크에서 사람보다 더 높은 정확도로 성능을 보이나, spurious pattern에 의존하여 robustness가 제한된다는 문제가 제기되었다. 본 연구는 이러한 문제를 해결하기 위해 대조 학습 (contrastive learning)과 반사례 증강 (counterfactual augmentation)을 활용한 새로운 접근 방식을 제안했다. 또한, 시각 장애인이 야외 스포츠를 안전하게 즐길 수 있도록 경량 멀티태스크 네트워크를 설계하여 트랙 라인과 장애물을 동시에 감지하는 웨어러블 디바이스를 개발하였다.

- **Technical Details**: MCQ 생성을 위한 새로운 평가 메트릭 Knowledge Dependent Answerability (KDA)를 도입하여 학생의 지식 평가 능력을 측정한다. 또한, 반사례 증강을 활용한 대조 학습 방법을 통해 여러 반사례 집합을 생성하고, 집합적 의사 결정을 통해 각 단어의 인과관계를 robust하게 감독하는 방식을 시도하였다. 시각 장애인을 위한 웨어러블 디바이스에서는 경량 멀티태스크 네트워크를 통해 경로 계획의 실제 적용성을 향상시켰으며, RealSense D435i 깊이 카메라와 Jetson Orin NX 임베디드 장치를 사용하여 실시간 인식과 내비게이션 기능을 제공한다.

- **Performance Highlights**: 제안된 MCQ 평가 메트릭 KDA_disc와 KDA_cont는 기존 메트릭과 결합 시 전문가 레이블이 붙은 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였다. 반사례 기반 접근 방식은 다양한 차원에서 성능을 크게 향상시켰으며, 특히 counterfactual robustness, cross-domain generalization, scarce data generalization에서 우수한 성능을 나타냈다. 시각 장애인을 위해 개발된 웨어러블 디바이스는 다양한 운동 시나리오에서 성공적으로 400미터를 평균 속도 1.34 m/s로 자유롭게 이동할 수 있도록 도와 정상인 수준의 조깅 속도를 충족시켰다.



### DECIDER: Leveraging Foundation Model Priors for Improved Model Failure Detection and Explanation (https://arxiv.org/abs/2408.00331)
Comments:
          Accepted at ECCV (European Conference on Computer Vision) 2024

- **What's New**: 이 논문들은 새로운 평가 메트릭 및 방법론으로 자동 MCQ 생성, NLP 적용 강인성 개선, 이미지 분류 모델의 오류 감지에 대한 접근을 소개합니다.

- **Technical Details**: ['MCQ 생성: 기존 BLEU, ROUGE, METEOR 메트릭의 한계를 극복하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안. KDA_disc와 KDA_cont 같은 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방함.', 'NLP 강인성: Spurious pattern에 대한 의존도를 줄이고 모델의 robust를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 활용. 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 robustness를 향상시킴.', '오류 감지: 기계 학습 모델의 오류를 신뢰성 있게 감지하기 위해 DECIDER를 제안. 이는 대규모 언어 모델(LLMs)과 비전-언어 모델(VLMs)을 활용하여 원본 모델과 디바이스된 모델 간 불일치를 측정함으로써 오류를 감지함.']

- **Performance Highlights**: ['MCQ 생성: KDA_disc와 KDA_cont는 인간 평가와 강한 상관관계를 가지며, 교육의 실제 사용성 측면에서도 높은 예측력을 보임.', 'NLP 강인성: 다양한 평가 지표 측면에서 counterfactual robustness, cross-domain generalization, 적은 데이터의 일반화에 있어 상당한 향상을 이룩함.', '오류 감지: 다양한 벤치마크 실험에서 DECIDER는 성능의 전반적인 Matthews 상관 계수와 실패 및 성공 리콜 측면에서 기존의 기준점을 크게 초과하는 결과를 나타냄.']



### Translating Imaging to Genomics: Leveraging Transformers for Predictive Modeling (https://arxiv.org/abs/2408.00311)
- **What's New**: 이번 연구는 새로운 문자열을 기반으로 이미지와 유전 정보를 예측하는 Transformer 기반 모델을 소개합니다. 이 접근법은 기존의 침습적 전 슬라이드 이미지 (WSI) 대신 CT/MRI 이미지를 활용하여 유전자 프로파일을 예측합니다.

- **Technical Details**: 이번 연구에서는 TCIA 데이터베이스의 CT/MRI 이미지와 TCGA 포털의 유전자 데이터를 사용하여 모델을 개발했습니다. 모델은 TransUNet을 사용해 이미지를 인코딩하고, 드롭아웃 레이어와 1D 컨볼루션 레이어로 유전자 예측을 진행합니다. 사용된 손실 함수는 Mean Squared Error (MSE)입니다.

- **Performance Highlights**: 모델 평가 결과, 여러 암 유형에 대한 유전자 연관성을 성공적으로 예측하였습니다. 특히, 폐암의 BRAF, ALK, KRAS, 및 글리오블라스토마의 IDH, 유방암의 CHEK와 같은 특정 유전자와의 연관성이 강하게 나타났습니다. 이는 비침습적 영상 기반 유전자 예측의 가능성을 강하게 시사합니다.



### Neural Octahedral Field: Octahedral prior for simultaneous smoothing and sharp edge regularization (https://arxiv.org/abs/2408.00303)
Comments:
          project page: this https URL

- **What's New**: 최근 MCQ(객관식 질문)의 자동 생성이 평가 메트릭의 한계를 극복하기 위해, 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안하였습니다. 이와 더불어, 신규 neural implicit representation 변화와 3D 스캐닝 기술 발전에 맞춘 새로운 방안도 제시되었습니다.

- **Technical Details**: ['MCQ 자동 생성에서는 기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 평가하지 못한다는 한계를 지적하며, 새로운 평가 메트릭인 KDA를 제안했습니다. 이는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다.', 'MCQ 평가에서 KDA_disc와 KDA_cont는 사전 학습된 언어 모델(pre-trained language models)을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'NLP의 robustness를 위해 contrastive learning과 counterfactual augmentation을 활용한 새로운 방법이 제안되었습니다. 이는 기존 방법보다 spurious correlations의 영향을 덜 받으며, 다양한 차원에서 성능 향상을 보여줍니다.', '3D 표면 재구성을 위해 octahedral field와 같은 새로운 neural implicit representation 방식을 제안하였습니다. 이 방식은 중점적으로 geometry features와 sharp edges를 자동적으로 유지할 수 있게 도와줍니다.']

- **Performance Highlights**: ['MCQ 평가에서 KDA_disc와 KDA_cont은 실제 강의실 세트에서의 사용성과 강한 상관관계를 나타냈습니다. 또한, 전문가가 라벨링한 MCQ 품질 측정에서 예측력이 높았습니다.', 'NLP robustness 연구에서는 식별된 문제들의 과대 적합에 덜 영향을 받아서 다양한 차원에서 향상된 성능을 보였습니다. 특히 counterfactual robustness, cross-domain generalization, scarce data generalization에서 유의미한 성과를 보였습니다.', '3D 표면 재구성 작업에서는 제안된 방식이 전통적 및 다른 neural approaches와 비교하여 포괄적인 실험에서 우수한 결과를 나타냈습니다.']



### Towards Flexible Evaluation for Generative Visual Question Answering (https://arxiv.org/abs/2408.00300)
- **What's New**: 최근 발표된 논문에서는 여러 가지 주요 AI 및 NLP 분야에서 혁신적인 기술과 평가 메트릭을 제안하였습니다. 이 논문들은 자동 MCQ 생성, NLP 태스크의 robustification, 그리고 Visual Question Answering (VQA) 평가를 다룹니다.

- **Technical Details**: [{'paper': 'Automatic Generation of Multiple Choice Questions', 'details': '기존의 BLEU, ROUGE, METEOR와 같은 MCQ 생성 평가 메트릭은 교육적 가치를 고려하지 않기 때문에 새로운 지식 종속 가능성 (KDA) 메트릭을 제안하였습니다. 이 메트릭은 학생의 문제 해결 행동을 모방하는 사전 훈련된 언어 모델을 사용하여 대답 가능성 (answerability)을 측정합니다.'}, {'paper': 'Counterfactual Robustness in NLP Models', 'details': '최근 NLP 모델들의 spurious pattern에 의한 robustness 제한 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용하는 새로운 방법을 제안합니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더 robust하게 파악합니다.'}, {'paper': 'Semantics-based Evaluation for VQA', 'details': 'VQA 평가에서 MLLMs의 다양한 응답을 적절하게 평가하기 위해, 기존의 Exact Match 메트릭의 한계를 극복하는 semantics-based 평가 메트릭을 제안합니다. 이 새로운 메트릭은 응답의 다양한 형태를 포함하여 평가의 일관성을 유지하며 인간 판단과 잘 맞도록 설계되었습니다. '}]

- **Performance Highlights**: [{'paper': 'Automatic Generation of Multiple Choice Questions', 'highlights': 'KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 확인했습니다. 또한, n-gram기반 유사성 메트릭과 결합할 경우, 전문가가 라벨링한 MCQ 품질 측정에 강한 예측력을 가집니다.'}, {'paper': 'Counterfactual Robustness in NLP Models', 'highlights': '집합적 의사 결정 방법을 사용하면 attribution-based synthesis 모델의 편향에 덜 민감하며, counterfactual robustness, cross-domain generalization 및 data 부족 상황에서의 일반화 측면에서 유의미한 개선을 달성합니다.'}, {'paper': 'Semantics-based Evaluation for VQA', 'highlights': '제안된 semantics-based VQA 평가 메트릭이 기존의 BLEU, ROUGE, METEOR와 같은 메트릭보다 성능이 우수합니다. 실험 결과, 이 새 메트릭이 인간 판단과 훨씬 더 잘 일치하며, 다양한 응답 형태에 대해 보다 정확한 평가를 제공합니다.'}]



### Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names (https://arxiv.org/abs/2408.00298)
- **What's New**: 이 논문들은 교육, 텍스트 생성 및 시각 장애인을 위한 만화 접근성 개선과 같은 다양한 주제를 다루고 있습니다. 첫 번째 논문은 자동 MCQ 생성의 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안합니다. 두 번째 논문은 대조 학습과 반사실적 증강을 활용하여 NLP 모델의 정교함을 개선합니다. 세 번째 논문은 시각 장애인을 위한 만화 접근성을 개선하기 위해 만화 챕터 전체의 대화 기록을 생성하는 모델 'Magiv2'를 소개합니다.

- **Technical Details**: [{'Paper': '자동 MCQ 생성', 'Details': 'KDA는 학생의 목표 사실 지식을 평가하는 능력을 기반으로 MCQ의 답변 가능성을 측정합니다. 인간 평가 및 사전 학습된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.'}, {'Paper': 'NLP 모델의 Robustness 개선', 'Details': '본 논문은 대조 학습과 반사실적 증강을 활용하여 NLP 모델의 Robustness를 개선합니다. 기존 방법과 달리 다수의 반사실적을 생성하여 집합적 의사 결정을 통해 인과 관계를 더 견고하게 분석합니다.'}, {'Paper': '시각 장애인을 위한 만화 접근성', 'Details': 'Magiv2는 챕터 전체의 대화 기록을 일관되게 생성하는 모델입니다. 캐릭터 이름과 텍스트를 정확하게 연관시키는 강력한 기능을 갖추고 있습니다. PopManga 평가 데이터 세트를 확장하고 새로운 캐릭터 데이터 은행을 구축했습니다.'}]

- **Performance Highlights**: [{'Paper': '자동 MCQ 생성', 'Highlights': 'KDA_disc와 KDA_cont가 인간 평가와 실제 교실에서의 사용성에서 강한 상관관계를 보였습니다. 여러 전문적으로 라벨된 MCQ 품질 측정값에 대해 예측력을 높였습니다.'}, {'Paper': 'NLP 모델의 Robustness 개선', 'Highlights': '집합적 의사 결정 방식을 통해 모델 편향에 덜 민감하게 되어 반사실적 Robustness, 크로스 도메인 일반화, 희소 데이터 일반화 측면에서 중요한 개선을 달성했습니다.'}, {'Paper': '시각 장애인을 위한 만화 접근성', 'Highlights': 'Magiv2는 챕터 전체의 대화 기록을 높은 정확도로 생성하고, 캐릭터 이름과 대화 상자 관계를 일관되게 유지합니다. 새로운 데이터 세트는 76개의 만화 시리즈에서 11K 이상의 주요 캐릭터를 포함하고 있습니다.'}]



### EmoTalk3D: High-Fidelity Free-View Synthesis of Emotional 3D Talking Head (https://arxiv.org/abs/2408.00297)
Comments:
          ECCV 2024

- **Automatic MCQ Generation**: [{"What's New": '기존 메트릭들이 MCQ의 교육적 가치를 평가하지 못하는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안했습니다.'}, {'Technical Details': 'KDA는 특정 사실에 대한 학생 응답을 통해 MCQ의 대답 가능성(answerability)을 측정하며, 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다.'}, {'Performance Highlights': '인간 연구를 통해, KDA_disc와 KDA_cont가 실제 강의실 환경에서 사용성과 강한 상관관계를 가지며,기존 n-gram기반 평가 메트릭과 결합 시 다양한 전문가가 라벨링한 MCQ 품질 측정에 대해 강력한 예측력을 보입니다.'}]

- **Deep Models' Robustness**: [{"What's New": '최근 NLP에서 인간을 능가하는 정확성을 보이는 deep model들의 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용하는 방법을 제안합니다.'}, {'Technical Details': "'Multiple counterfactuals'을 생성하고 집합적 의사 결정을 통해 각 용어의 인과관계를 더욱 robust하게 감독하는 방법입니다."}, {'Performance Highlights': '제안된 접근 방식은 기존 방법들에 비해 태스크 모델의 bias에 덜 민감하여, 다양한 차원에서의 성능 개선을 보여줍니다: 1) counterfactual robustness, 2) 교차 도메인 일반화(cross-domain generalization), 3) 부족한 데이터에서의 일반화.'}]

- **3D Talking Heads Synthesis**: [{"What's New": '조절 가능한 감정 표현과 향상된 입술 동기화를 특징으로 하는 새로운 3D 말하는 머리(talking heads) 생성 방법을 제안했습니다.'}, {'Technical Details': "이 방법은 'Speech-to-Geometry-to-Appearance' 매핑 프레임워크를 사용하여, 오디오 신호에서 4D 점구름을 예측한 뒤, 이 점들을 기반으로 4D 가우시안 모델을 사용해 얼굴 외관을 효율적으로 나타내는 방식을 취합니다."}, {'Performance Highlights': '제안된 방법은 입술 움직임 생성의 안정성과 렌더링 품질을 향상시키며, 주름과 미묘한 표정과 같은 얼굴의 동적 세부 사항을 포착하는 데 뛰어난 성능을 보입니다.'}]



### Head360: Learning a Parametric 3D Full-Head for Free-View Synthesis in 360{\deg} (https://arxiv.org/abs/2408.00296)
Comments:
          ECCV 2024

- **Multiple Choice Question (MCQ) 자동 생성**: {"What's New": '자동 MCQ 생성 평가를 위한 새로운 메트릭, Knowledge Dependent Answerability (KDA)을 제안합니다. 이는 학생의 답변 가능성을 측정하여 교육적 가치를 평가합니다.', 'Technical Details': '우리는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안하며, 이는 사전 학습된 언어 모델 (pre-trained language model)을 사용해 학생의 문제해결 행동을 모방합니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 실제 학급 환경에서의 사용성과 강한 상관관계를 보였으며, 기존의 n-gram 기반 메트릭과 결합했을 때 전문가가 평가한 다양한 MCQ 품질에 대한 예측력이 향상됩니다.'}

- **NLP 모델의 Robustness 향상**: {"What's New": '대조 학습 (contrastive learning) 및 실험적 데이터 증강 (counterfactual augmentation)을 통해 NLP 모델의 robustness를 강화하려는 새로운 접근 방안을 제안합니다.', 'Technical Details': '기존의 증강 방법과 달리 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 각 단어의 인과관계를 강화합니다.', 'Performance Highlights': '이 접근 방식은 1) 반실재적 강건성 (counterfactual robustness), 2) 분야 간 일반화 (cross-domain generalization), 3) 부족한 데이터에서의 일반화 면에서 유의미한 개선을 보였습니다.'}

- **360° 파라메트릭 인간 머리 모델**: {"What's New": '새롭게 제작된 고화질 인간 머리를 기반으로 360° 파라메트릭 머리 모델을 제안합니다. 이는 여러 각도에서 렌더링 가능하고 텍스트 기반 수정과 애니메이션 기능을 제공합니다.', 'Technical Details': '우리는 얼굴 모양/모션과 외형을 각각 고전적인 파라메트릭 3D 메쉬 모델과 신경 텍스처 (neural texture)로 분리하여 표현합니다.', 'Performance Highlights': "제안된 모델은 고화질 3D 머리 모델 생성 및 텍스트 기반 외형 편집, 애니메이션 기능에서 이전 모델을 초과하는 성능을 보여줍니다. 고화질 3D 머리 데이터셋인 'SynHead100'도 함께 공개됩니다."}



### RDP: Ranked Differential Privacy for Facial Feature Protection in Multiscale Sparsified Subspac (https://arxiv.org/abs/2408.00294)
Comments:
          13 pages, 6 figures

- **What's New**: 최근 자동 MCQ(객관식 질문) 생성 방법이 교육자의 평가 시간을 줄일 수 있는 가능성을 보여주고 있습니다. 하지만 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 교육적 가치를 반영하지 못하여, 학생의 지식을 평가하는 능력을 적절히 평가하지 못합니다. 이를 해결하기 위해, 대상 사실에 대한 학생의 지식을 평가하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다.

- **Technical Details**: KDA는 대상 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성(answerability)을 측정합니다. 우리는 먼저 인간 설문조사에서 수집한 학생의 응답을 통해 KDA를 측정하는 방법을 제시하고, 이를 모사하는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭은 사전 학습된 언어 모델(pre-trained language models)을 활용하여 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont은 (1) KDA와 (2) 실제 강의실 세트에서 전문가가 라벨링한 사용성과 높은 상관관계를 가지며, n-gram 유사성 메트릭과 결합했을 때 다양한 전문가 라벨 MCQ 품질 측정치에 대해 강한 예측력을 가짐을 보여주었습니다.



### Multi-Modal Parameter-Efficient Fine-tuning via Graph Neural Network (https://arxiv.org/abs/2408.00290)
- **What's New**: 최근의 딥 모델들이 NLP 태스크에서 뛰어난 성능을 보였지만, 자주 쓰이는 패턴에 의존하여 robustness가 제한되는 문제가 있습니다. 이를 해결하기 위해, 우리는 대조학습 (contrastive learning)과 반사실적 증강 (counterfactual augmentation) 방법을 제안합니다. 또한, 교육적 가치를 평가하기 어려운 기존 MCQ 생성 메트릭 문제를 해결하려는 새로운 메트릭, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)을 제안합니다.

- **Technical Details**: 제안된 MCQ 메트릭, KDA는 학생의 반응을 기반으로 MCQ의 대답 가능성을 측정하며, 이는 미리 학습된 언어 모델을 이용해 학생들이 문제를 풀어가는 과정을 모방합니다. 또한 반사실적 증강 기존 방법들과 달리, 다양한 반사실적 세트를 만들어 통합적인 결정을 내리고, 각 용어의 인과성을 보다 더 robust하게 감독하는 방법을 제시합니다.

- **Performance Highlights**: Human studies를 통해 우리가 제안한 KDA_disc와 KDA_cont는 실제 강의실 사용성과 강하게 상관성이 있음을 밝혔습니다. 또한 대조학습과 반사실적 증강 방법이 기존 모델들보다 다양한 차원에서 상당한 성능 향상을 보여주었으며, 특히 여러 데이터셋에서 4.45%, 2.92%, 0.23%의 정확도 향상을 보였습니다.



### Gradient Harmonization in Unsupervised Domain Adaptation (https://arxiv.org/abs/2408.00288)
Comments:
          IEEE TPAMI 2024

- **What's New**: 이번 논문에서는 자동 MCQ (Multiple Choice Questions) 생성의 평가 메트릭이 기존의 BLEU, ROUGE, METEOR와 같이 n-gram 기반의 유사성을 평가하는 방식에서 벗어나 학생의 지식을 평가할 수 있는 능력을 고려하지 않는 문제를 해결하고자 합니다. 새로운 자동 평가 메트릭인 지식 종속 가능성 (KDA, Knowledge Dependent Answerability)을 제안하여, MCQ가 주어진 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: KDA는 학생의 응답 데이터를 기반으로 MCQ의 대답 가능성을 평가합니다. KDA를 근사하기 위해 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안했습니다.

- **Performance Highlights**: Human evaluation을 통해, KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성 및 KDA 자체와 강한 상관관계를 가지고 있음을 발견했습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정치를 예측하는 능력이 뛰어남을 확인했습니다.



### Diff3DETR:Agent-based Diffusion Model for Semi-supervised 3D Object Detection (https://arxiv.org/abs/2408.00286)
Comments:
          Accepted to ECCV2024

- **What's New**: 이번 뉴스레터에서는 세 가지 최신 arXiv 논문을 소개합니다. 첫째, 자동화된 객관식 질문(MCQ) 생성의 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안한 연구입니다. 둘째, 대조 학습과 반사실적 증강(counterfactual augmentation)을 통해 NLP 모델의 robust성 개선을 목표로 하는 연구입니다. 셋째, 반사실적(diffusion) 모델을 사용한 반지도 학습 3D 객체 탐지를 제안한 연구입니다. 각 연구의 주요 내용을 보시겠습니다.

- **Technical Details**: [{'title': '객관식 질문(MCQ) 자동 생성의 새로운 평가 메트릭', 'description': '기존의 BLEU, ROUGE, METEOR과 같은 평가 메트릭은 단순히 데이터셋의 골드 샘플과 n-gram 유사성을 비교할 뿐, 교육적 가치를 평가하지 못합니다. 본 연구에서는 학생의 대상 사실(target fact)에 대한 지식을 평가하는 능력을 측정하는 KDA를 제안합니다. KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 통해 KDA를 근사하고, 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, {'title': '대조 학습과 반사실적 증강을 통한 NLP 모델의 robust성 개선', 'description': '본 연구는 반사실적 증강을 통해 모델의 인과성을 강력하게 감독(supervise)하는 방법을 제안합니다. 기존 방법들은 spurious correlation에 영향을 받지만, 여러 개의 반사실적을 생성하고 집합적 의사 결정을 통해 더 robust하게 모델의 성능을 개선할 수 있습니다.'}, {'title': '반사실적 모델을 사용한 반지도 학습 3D 객체 탐지', 'description': 'Diff3DETR, agent-based object query generator와 box-aware denoising 모듈을 통합한 새로운 DETR 프레임워크를 제안합니다. 이는 동적 장면에 적응하며 샘플링 위치와 콘텐츠 임베딩의 균형을 맞추고, DDIM denoising 프로세스와 transformer 디코더를 통해 초기 박스를 점진적으로 개선합니다. 이 방법은 ScanNet과 SUN RGB-D 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다.'}]



### DMESA: Densely Matching Everything by Segmenting Anything (https://arxiv.org/abs/2408.00279)
- **Proposal**: [{"What's New": 'Multiple Choice Questions (MCQ) 자동 생성을 위한 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안했습니다. 이는 기존 BLEU, ROUGE, METEOR와 다르게 교육적 가치를 고려하여 MCQ의 대답 가능성을 평가합니다.'}, {'Technical Details': '우리는 Human survey를 통해 KDA를 측정하는 방법을 보여주고, 사전 교육된 language models을 활용하여 KDA를 근사하는 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안했습니다.'}, {'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 높은 사용성을 가지고 있음을 입증했습니다. 기존의 n-gram 기반 유사성 메트릭과 함께 사용했을 때, 다양한 전문가가 라벨링한 MCQ 품질 척도에 대한 강력한 예측력을 보였습니다.'}]

- **Robustness**: [{"What's New": '대조 학습 (contrastive learning) 과 반사기보강 (counterfactual augmentation)을 활용하여 신뢰성을 높이는 접근법을 제안합니다.'}, {'Technical Details': '기존 방법과 달리, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 예측 분포를 검토하여 인과관계를 보다 robust하게 감독합니다.'}, {'Performance Highlights': '우리 접근법은 counterfactual robustness, 교차 도메인 일반화, 데이터 부족 환경에서의 일반화 등 다양한 측면에서 상당한 성능 향상을 보였습니다.'}]

- **Feature Matching**: [{"What's New": 'Segment Anything Model (SAM) 을 활용하는 새로운 feature matching 방법인 MESA와 DMESA를 제안했습니다. 이는 매칭 중복성을 효과적으로 완화하는 데 중점을 둡니다.'}, {'Technical Details': 'MESA는 sparse matching 프레임워크를 채택하여 Area Graph (AG)를 통해 SAM 결과에서 후보 영역을 얻고 그래프 에너지 최소화 방식으로 area matching을 해결합니다. DMESA는 dense matching 프레임워크를 적용하여 효율성을 높이고 Gaussian Mixture Model과 Expectation Maximization을 활용하여 매칭 분포를 생성 및 정제합니다.'}, {'Performance Highlights': 'DMESA는 MESA 대비 거의 5배의 속도 향상을 보였으며, 경쟁력 있는 정확성을 유지합니다. 다섯 개의 데이터셋에서 다양한 point matching baseline에서 일관된 성능 개선을 보여주었으며, 이미지 해상도 변화에 대한 높은 신뢰성을 입증했습니다.'}]



### Improving Image De-raining Using Reference-Guided Transformers (https://arxiv.org/abs/2408.00258)
- **What’s New 1**: 기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ의 학습적 가치를 고려하지 못하고 있다. 이를 해결하기 위해 우리는 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했다.

- **Technical Details 1**: KDA는 MCQ의 답변 가능성을 측정하며, 학생의 대상 사실(target fact) 지식을 평가하는 능력을 가진다. 제안된 모델은 두 종류의 자동 평가 메트릭, KDA_disc와 KDA_cont를 포함하며, 이는 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights 1**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 나타내는 것을 확인했다. 또한 이 메트릭들은 기존의 n-gram 기반 유사도 메트릭과 결합하여 다양한 전문가가 라벨링한 MCQ의 질적 측정을 예측하는 데 강한 예측력을 보였다.



### LoopSparseGS: Loop Based Sparse-View Friendly Gaussian Splatting (https://arxiv.org/abs/2408.00254)
Comments:
          13 pages, 10 figures

- **What's New**: 이 논문은 MCQ(다지 선다형 질문) 생성을 평가하기 위해 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 기존의 BLEU, ROUGE, METEOR와 달리 MCQ의 대답 가능성과 교육적 가치를 평가합니다.

- **Technical Details**: 제안된 KDA는 학생의 문제 해결 행동을 모방하기 위해 사전 학습된 언어 모델을 사용하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 포함합니다. 이는 인간 설문 조사를 기반으로 KDA를 측정하는 방법을 제시합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 KDA와 전문가가 라벨링한 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 측정에 대한 예측력이 높아짐을 확인했습니다.



### Task-Adapter: Task-specific Adaptation of Image Models for Few-shot Action Recognition (https://arxiv.org/abs/2408.00249)
Comments:
          Accepted by ACM MM2024

- **What's New**: 최근 자동 다지선다형 질문(MCQ) 생성 및 평가에서 BLEU, ROUGE, METEOR와 같은 전통적인 n-gram 기반 메트릭이 교육적 가치를 적절히 평가하지 못한다는 문제를 해결하기 위해, 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 또한, 최근 딥러닝 모델들이 NLP 태스크에서 높은 정확도를 보였으나 spurious pattern에 의존하는 문제로 인해 강건성(robustness)이 제한된다는 점을 해결하기 위해 대조 학습 및 반사경(Counterfactual) 증대를 활용한 새로운 방법론을 제안했습니다. 마지막으로, 몇 샷 학습(few-shot learning) 액션 인식에서 과도한 튜닝으로 발생하는 overfitting 문제를 해결하고자 Task-Adapter를 도입하여 효율적으로 task-specific된 적응 메커니즘을 포함한 방법을 제안했습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 지식 의존형 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 통해 이를 자동화합니다. 대조 학습 및 반사경 증대를 통해 더 robust하게 단어들의 인과관계를 파악하였고, 여러 개의 반사경을 생성하여 집합적 의사 결정(collective decisions)을 통해 강건성을 증진하였습니다. Task-Adapter를 통해 비디오 데이터셋에서 Pre-trained 모델의 일부만 미세 조정하여 과적합 문제를 방지하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가와 높은 상관관계를 가지며, 교육적 측면에서도 유용한 평가 메트릭으로 인정받았습니다. 제안된 반사경 증대 방법은 다양한 차원에서 의미 있는 성능 향상을 달성했으며, 특히 대조적 robust성과 cross-domain 일반화에서 뛰어난 성능을 보였습니다. Task-Adapter를 사용한 몇 샷 액션 인식에서는 기존 최첨단 방법들을 크게 능가하는 성과를 보이며, 특히 SSv2 dataset에서 큰 차이를 보였습니다.



### A Prior Embedding-Driven Architecture for Long Distance Blind Iris Recognition (https://arxiv.org/abs/2408.00210)
- **What's New**: 이 논문은 장거리에서의 눈동자 인식 (iris recognition) 시, 미지의 손상으로 인해 발생하는 '블라인드 아이리스 이미지' 문제를 해결하기 위한 새로운 접근법을 제안합니다. 구체적으로, 블라인드 아이리스 이미지를 효과적으로 복원하기 위한 Iris-PPRGAN 네트워크와 더 나은 아이리스 특징 추출을 위한 Insight-Iris 분류기를 소개합니다.

- **Technical Details**: Iris-PPRGAN 네트워크는 생성적 적대 신경망 (GAN)을 Prior Decoder로 사용하고, 심층신경망 (DNN)을 인코더로 사용하여 블라인드 아이리스의 텍스처를 복원합니다. 또한 InsightFace의 병목 모듈을 수정하여 Insight-Iris 분류기를 개발하였고, 이는 복원된 아이리스 이미지를 인식하는 데 사용됩니다.

- **Performance Highlights**: 공개된 CASIA-Iris-distance 데이터셋에서, 제안된 방법은 정성적 및 정량적으로 최첨단 블라인드 아이리스 복원 방법보다 우수한 성능을 보였습니다. 특히, 복원 후의 장거리 블라인드 아이리스 이미지의 인식률이 90%에 달했으며, 이는 복원 전 이미지와 비교하여 약 10% 포인트가 개선된 결과입니다.



### OmniParser for Pure Vision Based GUI Agen (https://arxiv.org/abs/2408.00203)
- **What's New**: 이 논문에서는 자동 MCQ(객관식 질문) 생성의 교육적 가치를 평가하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)을 제안합니다. 또한, GPT-4V와 같은 대형 비전-언어 모델이 다양한 사용자 인터페이스에서 동작하는 것을 돕기 위해 OmniParser를 도입하여 화면 상의 상호작용 가능한 요소와 그 기능적 의미를 보다 정확하게 이해하도록 합니다.

- **Technical Details**: [{'MCQ Generation': '기존 MCQ 생성의 평가 메트릭들은 BLEU, ROUGE, METEOR처럼 n-gram 기반의 유사성에만 집중되어 있었으나, 교육적 가치를 평가하지 않았습니다. 이를 해결하기 위해, KDA라는 새로운 메트릭을 통해 학생이 해당 정보를 알고 있을 때 MCQ가 답할 수 있는지 여부를 평가합니다. KDA는 Human survey를 기반으로 측정되며, KDA_disc와 KDA_cont라는 자동화된 평가 메트릭을 제안합니다.'}, {'Model Robustness': '최근의 deep model들이 NLP 태스크에서 높은 정확성을 보였으나, spurious pattern에 의존하여 robustness가 제한되었습니다. 이 논문에서는 contrastive learning과 counterfactual augmentation을 활용하여 단어들의 인과관계를 판단하는 방법을 제안합니다. 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 이전의 spurious correlation 문제를 극복합니다.'}, {'Screen Parsing': 'OmniParser는 사용자의 인터페이스 스크린샷을 파싱하여 구조화된 요소들로 변환하는 종합적인 방법입니다. 이를 통해 GPT-4V가 다양한 애플리케이션에서 예측한 행동을 정확한 화면 영역에 결합할 수 있습니다. 이 시스템은 상호작용 가능한 요소를 탐지하고 기능적 의미를 추출하기 위해 특별히 학습된 모델을 활용합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 함께 사용할 때, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였습니다.'}, {'Model Robustness': '제안된 방법은 단어들의 인과관계 파악에서 bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, and generalization from scarce data에서 놀라운 성능 향상을 보였습니다.'}, {'Screen Parsing': 'OmniParser는 ScreenSpot, Mind2Web, AITW 벤치마크에서 기존 GPT-4V 기반 모델을 능가했습니다. 특히 추가 정보 없이 스크린샷만으로도 뛰어난 성능을 보여주었습니다.'}]



### S-SYNTH: Knowledge-Based, Synthetic Generation of Skin Images (https://arxiv.org/abs/2408.00191)
Comments:
          Accepted to the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024

- **What's New**: 최근 문서에서 다룬 여러 혁신적인 연구를 요약했습니다. 첫 번째 연구는 자동 MCQ(Multiple Choice Questions) 생성의 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 두 번째 연구는 NLP(자연어 처리) 모델의 강인성(robustness)을 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용하는 방법에 대해 다룹니다. 마지막으로, S-SYNTH라는 피부 시뮬레이션 프레임워크를 소개하여 피부 병변(segmenting) 세분화 성능을 개선하는 거대한 피부 데이터를 생성하는 방법을 설명합니다.

- **Technical Details**: 첫 번째 연구에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 연구는 반사실적 데이터 세트를 통해 인과 관계(causal relationship)를 강화하고, 이를 통해 다양한 차원에서 모델의 성능을 향상시키는 방법을 제안합니다. 마지막 연구에서는 S-SYNTH를 통해 3D 피부 모델과 디지털로 렌더링된 이미지를 생성하며, 이를 통해 실제 피부 이미지의 성능 추세를 모방하고, 기존 데이터 세트의 한계를 보완합니다.

- **Performance Highlights**: 첫 번째 연구에서는 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 보였습니다. 두 번째 연구에서는 다양한 차원에서 모델의 강인성, 도메인 간 일반화(cross-domain generalization), 제한된 데이터에서의 일반화 성능이 크게 향상되었습니다. 마지막 연구에서는 S-SYNTH를 통해 생성된 합성 이미지가 제한된 실제 이미지 세트를 사용할 때 향상된 분할 성능(segmentation performance)을 보여주었고, 피부 색상과 병변 크기 측면에서 실제 환자 데이터 세트와 유사한 비교 성능 추세를 나타냈습니다.



### CC-SAM: SAM with Cross-feature Attention and Context for Ultrasound Image Segmentation (https://arxiv.org/abs/2408.00181)
Comments:
          Accepted to ECCV 2024

- **What's New**: 이번 연구에서는 교육적 가치를 평가하는 새로운 MCQ 생성 자동 평가 메트릭, Knowledge Dependent Answerability(KDA)를 제안했습니다. 또한, NLP 태스크의 robust성을 높이기 위해 대조 학습 및 counterfactual augmentation을 활용한 새로운 방법을 제안하였습니다. 마지막으로, SAM 모델의 의료 영상 세그멘테이션 성능을 향상시키기 위해 CNN과 ViT를 활용한 CC-SAM 모델을 도입했습니다.

- **Technical Details**: KDA는 학생의 응답 기반으로 MCQ의 대답 가능성을 측정하며, 이를 자동 평가 메트릭(KDA_disc, KDA_cont)으로 구현하여 사전 학습된 언어 모델을 통해 학생의 문제 해결 행동을 모방합니다. 또 다른 연구에서는 여러 개의 counterfactual을 생성하여 더 robust하게 인과 관계를 평가하는 방법을 제안했습니다. CC-SAM에서는 정지된 CNN 지점과 ViT 인코더를 통합하고, variational attention fusion 모듈을 도입하여 의료 영상에서의 성능을 개선하였습니다.

- **Performance Highlights**: KDA 기반 자동 평가 메트릭은 학생 및 전문가 평가와 강한 상관관계를 가지며, 기존 단어 기반 유사성 메트릭과 결합 시 MCQ의 품질을 예측하는 강력한 성능을 보였습니다. NLP 모델의 robust성을 평가한 연구에서는 counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 등 여러 차원에서 상당한 향상을 나타냈습니다. CC-SAM은 ChatGPT 기반 텍스트 프롬프트를 도입하여 의료 영상 세그멘테이션의 성능을 크게 향상시켰습니다.



### Strike the Balance: On-the-Fly Uncertainty based User Interactions for Long-Term Video Object Segmentation (https://arxiv.org/abs/2408.00169)
- **November 2023**: [{"What's New": '새로운 MCQ 자동 평가 메트릭 KDA(지식 종속 가능성)를 제안하였습니다. 기존의 BLEU, ROUGE, METEOR와 같은 메트릭은 MCQ의 교육적 가치를 고려하지 않은 한계가 있었습니다.', 'Technical Details': 'KDA는 대상 사실(target fact)에 대한 학생의 지식을 평가하는 능력을 측정합니다. 우리는 학생 응답을 기반으로 KDA를 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다. 이들은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, 다양한 전문 평가 고급 MCQ 품질 측정에 대한 강력한 예측력을 보였습니다.'}, {"What's New": '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 NLP 태스크의 강인성 개선을 제안하였습니다.', 'Technical Details': "기존의 증강 방법들은 spurious correlation의 영향을 받는 한계가 있었습니다. 우리는 '여러 개의' counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과관계를 더 robust하게 감독하는 방법을 제안합니다.", 'Performance Highlights': '우리의 접근 방식은 대조 학습에 의해 더욱 강인해졌으며, 다양한 차원에서 중요한 개선을 달성했습니다: 1) 반사실적 강인성, 2) 크로스-도메인 일반화, 3) 드문 데이터에서의 일반화.'}, {"What's New": '영상 객체 분할(VOS)의 새로운 버전, Lazy Interactive Video Object Segmentation (ziVOS)을 소개합니다. 이는 반자동(sem-automatic) 및 인터랙티브(Interactive) 접근 방식을 결합합니다.', 'Technical Details': 'ziVOS는 온라인(실시간)으로 녹화된 시퀀스를 타겟으로 하며, 사용자 피드백을 실시간으로 받아 객체 추적 기간을 최대화하는 것을 목표로 합니다. 한 프레임당 하나의 상호작용을 허용하여 최소한의 사용자 수정만으로 객체 추적을 유지합니다.', 'Performance Highlights': '우리는 ziVOS를 위한 경쟁력 있는 기준선인 Lazy-XMem를 제안했으며, 추적 상태의 불확실성을 평가하여 필요할 때마다 모델의 예측을 수정합니다. 우리의 방법론은 LVOS 데이터셋을 통해 평가되었으며, 최신 평가 메트릭 및 사용자 작업량 평가를 통해 높은 성능을 입증했습니다.'}]



### Certifying Robustness of Learning-Based Keypoint Detection and Pose Estimation Methods (https://arxiv.org/abs/2408.00117)
Comments:
          25 pages, 10 figures, 5 tables

- **What's New**: 최근 NLP 태스크에서 인간보다 높은 정확성을 보이는 딥 모델이 spurious 패턴에 의존해 robustness가 제한된다는 문제를 해결하기 위해, 대조 학습(contrastive learning)과 반사실 확장(counterfactual augmentation)을 활용하는 새로운 방법을 제안했습니다. 이 연구는 여러 개의 반사실 세트를 생성하고, 집합적 의사 결정을 통해 각 용어의 인과 관계를 보다 robust하게 감독하는 접근 방식을 채택합니다.

- **Technical Details**: 기존 방법은 사람이 직접 반사실을 추가하거나, 모델이 데이터셋에서 유사한 반사실을 자동으로 찾아야 했습니다. 그러나 이는 여전히 spurious 상관관계에 영향을 받는다는 문제가 있었습니다. 제안된 방법은 여러 개의 반사실을 생성하고, 이를 통해 집합적 의사 결정을 내리며, 각 용어의 인과 관계를 robust하게 감독할 수 있습니다. 특히, 이 방법은 특정 딥러닝 모델의 편향에 덜 민감하게끔 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 크게 세 가지 면에서 성능이 향상되었습니다: 1) 반사실 robustness, 2) 도메인 간 일반화(cross-domain generalization), 3) 희소 데이터로부터의 일반화. 이는 다양한 평가에서 기존 방법들보다 우수한 성능을 나타냈습니다.



### Automated Sperm Morphology Analysis Based on Instance-Aware Part Segmentation (https://arxiv.org/abs/2408.00112)
Comments:
          Accepted to ICRA 2024

- **What's New**: 이 논문에서는 다중 선택 질문(MCQ)의 자동 생성 및 평가를 개선하기 위해 새로운 자동 평가 메트릭 '지식 종속 가능성(KDA)'을 제안했습니다. 또한, NLP 태스크의 robustness를 강화하기 위해 대조 학습과 반사실적 데이터 증강 기법을 도입하는 방법과 정자 형태 분석의 자동화 기술이 제시되었습니다.

- **Technical Details**: MCQ 평가에서는 KDA_disc와 KDA_cont라는 두 가지 새로운 자동 평가 지표를 정의하여 사전 훈련된 언어 모델을 활용했습니다. 반사실적 증강 기법에서는 여러 개의 반사실적 데이터를 생성하고, 집합적 의사 결정을 통해 각 용어의 인과관계를 더 강력하게 관리합니다. 정자 형태 분석에서는 주의(attention) 기반 인스턴스 인식 부품 분할 네트워크와 중심선(centerline) 기반 꼬리 형태 측정 방법을 개발했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의 설정에서의 사용성과 강하게 상관되어 있음을 확인했습니다. 반사실적 데이터 증강 기법으로 다양한 차원에서 성능이 향상되었으며, 새로운 정자 형태 분석 네트워크가 기존 RP-R-CNN을 9.2% 초과하여 57.2% APvol^p를 달성했습니다. 꼬리 형태 측정 정확도는 길이, 폭, 곡률에서 각각 95.34%, 96.39%, 91.2%를 기록했습니다.



### WAS: Dataset and Methods for Artistic Text Segmentation (https://arxiv.org/abs/2408.00106)
Comments:
          Accepted by ECCV 2024

- **What's New**: 자동 다중 선택 질문(MCQ) 생성의 효율성을 개선하기 위해 새로운 평가 메트릭, Knowledge Dependent Answerability(KDA)를 도입했습니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 메트릭의 한계를 극복하고 교육적 가치를 평가합니다.

- **Technical Details**: KDA는 학생 설문을 통해 MCQ의 대답 가능성(answerability)을 측정합니다. 이를 기반으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실에서의 사용성과 강한 상관관계를 보여주었고, n-gram 기반 메트릭과 결합해 전문가가 라벨링한 MCQ 품질을 예측하는 데 높은 정확도를 보였습니다.



### From Attributes to Natural Language: A Survey and Foresight on Text-based Person Re-identification (https://arxiv.org/abs/2408.00096)
- **What's New**: 이번 논문은 교육자들이 학생 평가에 소요하는 시간을 크게 줄일 수 있는 Multiple Choice Questions (MCQ) 자동 생성 분야의 혁신적인 연구를 소개합니다. 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 데이터셋 내 금 샘플과의 n-gram 기반 유사성에만 초점을 두어 교육적 가치를 간과하고 있습니다. 이러한 문제를 해결하기 위해, 우리는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 측정하고 해당 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 특히, 학생 응답을 분석하여 KDA를 측정하는 방법을 먼저 제시하고, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사화하는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대한 예측 정확도를 높입니다.

- **Performance Highlights**: 인간 연구를 통해 우리는 KDA_disc와 KDA_cont가 실제 강의실 설정에서 사용 가능성과 강한 상관관계를 가지고 있음을 보였습니다. 또한, 이 메트릭들은 전문가가 라벨링한 다양한 MCQ 품질 측정치와 강한 예측력을 보여주었습니다.



### Localized Gaussian Splatting Editing with Contextual Awareness (https://arxiv.org/abs/2408.00083)
- **What's New**: 이 연구는 자동으로 다중 선택 질문(MCQ)을 생성할 때 기존의 평가 지표인 BLEU, ROUGE, METEOR가 데이터셋의 골드 샘플과의 유사성에만 초점을 맞추고 교육적 가치를 고려하지 않는 문제점을 지적합니다. 새로운 자동 평가 메트릭으로 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하여 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: 지식 종속 가능성(KDA)를 측정하는 방법으로, 인적 조사(human survey)에 기반한 학생의 응답을 활용합니다. 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안하여 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 실제 교실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, 다양한 전문가 레이블 MCQ 품질 측정에 대해 강력한 예측력을 가집니다.



### Evaluating Transfer Learning in Deep Learning Models for Classification on a Custom Wildlife Dataset: Can YOLOv8 Surpass Other Architectures? (https://arxiv.org/abs/2408.00002)
Comments:
          This paper is being reviewed by SN Computer Science (springer journal)

- **What's New**: 이 논문은 자동으로 Multiple Choice Questions(MCQ)를 생성할 때, 기존의 평가 메트릭이 교육적 가치를 충분히 반영하지 않는 문제를 제기하고 있습니다. 이를 해결하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)을 제안합니다. 또한, 우리가 제안한 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭이 실제 강의실에서의 사용성 및 기존 메트릭과 강한 상관관계를 가지고 있음을 확인했습니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성(answerability)을 측정하고, 학생의 지식을 평가하는 능력을 평가합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델(pre-trained language models)을 사용해 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 우리는 다양한 인간 연구를 통해 이 메트릭이 강의실에서도 유용하다는 것을 보여주었으며, n-gram 기반의 유사성 메트릭과 결합하여 MCQ 품질을 예측하는 강력한 예측 능력을 가집니다.

- **Performance Highlights**: {'MCQ Generation': {'KDA_disc': '강의실 사용성 및 KDA와 높은 상관관계', 'KDA_cont': '유사하게 높은 강의실 사용성 및 KDA와의 상관관계'}, 'NLP Robustness': '이전 연구들이 spurious patterns에 의존하여 robustness가 제한되는 문제를 지적하고, 대비 학습(contrastive learning) 및 반사실적 증가(counterfactual augmentation) 방법을 활용해 이를 개선', 'Wildlife Monitoring': '딥러닝 모델(CNNs)과 전이 학습(transfer learning)을 통해 멸종 위기 종의 모니터링을 자동화하는 방법 연구. YOLOv8 모델이 97.39%의 훈련 정확도와 96.50%의 F1 점수를 기록하며, 다른 모델보다 높은 성능을 보임'}



### Replication in Visual Diffusion Models: A Survey and Outlook (https://arxiv.org/abs/2408.00001)
Comments:
          The first survey focuses on replication in visual diffusion models. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: {'MCQ Generation': "이 논문은 기존 MCQ(Multiple Choice Questions) 생성 평가 메트릭이 교육적 가치를 충분히 반영하지 못하는 문제를 다루고, 새롭게 'Knowledge Dependent Answerability (KDA)'라는 평가 메트릭을 제안합니다. 이 메트릭은 질문의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 평가합니다.", 'Contrastive Learning in NLP': '최근 NLP 태스크에서 높은 정확성을 보이는 deep models의 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 결합한 새로운 접근법을 제안합니다. 여기에 추가로 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 강화된 인과관계를 파악합니다.', 'Visual Diffusion Models': 'Visual diffusion models에서 훈련 데이터 복제를 탐구하고 이와 관련된 프라이버시, 보안, 저작권 문제를 다루는 첫 종합적인 리뷰를 제공합니다. 이 리뷰는 복제를 탐지하고 이해하며 완화하는 방법을 카테고리화합니다.'}

- **Technical Details**: {'MCQ Generation': 'KDA 측정을 위해 인간 설문조사를 기반으로 한 방법을 제시하고, 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont를 제안합니다. 이를 통해 학생의 문제 해결 행동을 모방하여 자동 평가 지표를 제공합니다.', 'Contrastive Learning in NLP': '여러 개의 counterfactual을 생성하여 집합적 의사 결정을 통해 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 이는 기존의 spurious correlation 문제를 해결할 수 있는 것으로 나타났습니다.', 'Visual Diffusion Models': '복제를 탐지하기 위한 메커니즘, 이 현상을 유발하는 요인 분석, 복제를 완화하는 전략을 설명합니다. 또한, 실세계에 미치는 영향을 검토하고, 특히 헬스케어 분야에서의 프라이버시 문제를 강조합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 교실 세트에서의 사용성 및 전문가가 라벨링한 다양한 MCQ 품질 측정 항목에 대해 강한 예측력(Strong predictive power)을 보였습니다.', 'Contrastive Learning in NLP': '이 접근법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등에서 유의미한 성능 향상을 보였습니다.', 'Visual Diffusion Models': '이 리뷰 논문은 복제 탐지와 벤치마킹에서의 어려움, 더욱 견고한 완화 기술의 개발 방향 등을 논의하여 연구자 및 실무자에게 유익한 통찰을 제공합니다.'}



### AMAES: Augmented Masked Autoencoder Pretraining on Public Brain MRI Data for 3D-Native Segmentation (https://arxiv.org/abs/2408.00640)
- **Multiple Choice Questions Generation Evaluation**: [{"What's New": '기존의 MCQ 생성 평가 메트릭이 교육적 가치를 무시한다는 문제를 해결하기 위해, 지식 종속 가능성(KDA)을 도입. 이는 MCQ의 학생 지식 평가 능력과 대답 가능성을 측정하는 새로운 자동 평가 메트릭입니다.'}, {'Technical Details': 'KDA는 학생 응답을 바탕으로 측정되며, 두 개의 자동 평가 메트릭 KDA_disc와 KDA_cont가 도입되었습니다. 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA 값을 근사합니다. KDA_disc와 KDA_cont는 n-gram 기반의 유사성 메트릭과 결합하여 높은 예측력을 지닙니다.'}, {'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성 (usability) 과 강한 상관관계를 가진다는 점을 입증했습니다.'}]

- **NLP Model Robustness**: [{"What's New": '최근 딥러닝 모델이 NLP 태스크에서 높은 정확성을 보였지만, spurious pattern에 의존함으로 인해 robust하지 않은 문제가 보고되었습니다. 이에 대해 본 연구는 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 새로운 접근을 제안합니다.'}, {'Technical Details': '기존의 augmentation 방법들이 spurious correlation에 영향을 받는 것과 달리, 우리의 방법은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 robust하게 단어들의 인과관계를 파악합니다. 이는 모델의 편향에 덜 민감하게 만들며 다양한 차원에서 성능을 개선합니다.'}, {'Performance Highlights': '감소된 task 모델의 편향과 함께 다양한 차원에서 반사실적 강건성, 도메인 간 일반화, 적은 데이터로부터의 일반화 성능에서 크게 향상된 결과를 보였습니다.'}]

- **3D Semantic Segmentation Pretraining**: [{"What's New": '대규모, 도메인 특화 데이터셋을 이용한 3D semantic segmentation 모델의 자가 지도 사전훈련(self-supervised pretraining)이 성능에 미치는 영향을 조사했습니다. 이를 위해 44,756개 브레인 MRI 볼륨을 포함하는 BRAINS-45K 데이터셋을 소개합니다.'}, {'Technical Details': 'AMAES 프레임워크는 Masked Image Modeling (MIM)과 intensity-based augmentation reversal을 결합하여 메모리 사용, 실행 시간, 그리고 finetuning 성능의 균형을 잡습니다. U-Net과 MedNeXt라는 두 가지 백본 아키텍처를 사용하여 세 가지 다운스트림 작업에서 사전 훈련의 효과를 평가했습니다.'}, {'Performance Highlights': '사전 훈련된 AMAES 모델은 대부분의 평가된 경우에서 분할 성능을 크게 향상시켰으며, augmentation을 사용한 사전 훈련이 대형 데이터셋에서도 유익하다는 점을 확인했습니다.'}]



### Privacy-preserving datasets by capturing feature distributions with Conditional VAEs (https://arxiv.org/abs/2408.00639)
Comments:
          Accepted at BMVC 2024

- **What's New**: 최근 MCQ 자동 생성의 평가 메트릭이 educational value를 간과한다는 문제를 해결하기 위해, 새로운 자동 평가 메트릭 Knowledge Dependent Answerability (KDA)를 제안하였습니다. 또한, 기존 deep model들이 spurious pattern에 의존하여 robustness가 제한된다는 점을 해결하기 위해, contrastive learning과 counterfactual augmentation을 활용한 새로운 방법을 제시했습니다. 마지막으로, 데이터 보안 문제를 해결하기 위해 Conditional Variational Autoencoders (CVAEs)를 사용한 새로운 익명화 접근법을 소개했습니다.

- **Technical Details**: 우리는 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안하여, MCQ의 실제 교육적 가치를 측정하고자 하였습니다. 또한, pre-trained language models를 활용하여 KDA를 근사하는 KDA_disc와 KDA_cont를 제안하였습니다. spurious pattern에 의존하는 문제를 해결하기 위해 여러 개의 counterfactual을 생성하고, collective decision을 통해 robust하게 인과관계를 파악하는 방법을 사용하였습니다. 마지막으로, vision foundation models를 활용하여 Conditional Variational Autoencoders (CVAEs)를 통해 특징 벡터 공간에서 다양한 가상 데이터를 생성하는 방식을 도입했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont가 실제 강의실 세트에서 usability와 강한 상관관계를 가짐을 human evaluation을 통해 확인하였습니다. 새로운 counterfactual 방법론은 다양한 태스크에서 significant improvements을 달성했으며, 특히 counterfactual robustness, cross-domain generalization, 그리고 generalization from scarce data 측면에서 큰 성과를 보였습니다. CVAEs를 사용한 새 익명화 접근법은 전통적인 방법들보다 데이터셋 다양성 및 모델의 robustness 측면에서 뛰어난 성능을 보였습니다.



### SynesLM: A Unified Approach for Audio-visual Speech Recognition and Translation via Language Model and Synthetic Data (https://arxiv.org/abs/2408.00624)
- **What's New**: 자동 MCQ 생성의 평가 문제를 해결하기 위해 새로운 지식 종속 대답 가능성(KDA) 메트릭을 제안했습니다. 기존에는 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 주로 사용되었지만, 이는 교육적 가치를 평가하지 못했습니다. 또한, NLP 태스크에서 모델의 강인함을 향상시키기 위해 대조 학습과 반사실적 증가(counterfactual augmentation) 방법을 제안했습니다. 마지막으로 SynesLM이라는 모델을 통해 오디오-비주얼 자동 음성 인식(AV-ASR), 비주얼 보조 음성/기계 번역(VST/VMT) 등 다양한 멀티모달 언어 이해 태스크를 수행할 수 있는 통합 모델을 소개했습니다.

- **Technical Details**: MCQ 생성 평가를 위해 KDA라는 새로운 메트릭을 도입하였으며, 이는 학생의 반응을 기반으로 대답 가능성을 측정합니다. 대조 학습과 반사실적 증가 방법을 통해 강인함이 향상된 모델을 설계했습니다. SynesLM 모델은 전체 프레임의 시각 정보를 활용하고, 시각-언어 연결 레이어를 통해 음성 및 텍스트 토큰을 동일한 임베딩 공간에 맵핑합니다. 이를 통해 시각적 정보와 텍스트 정보 사이의 통합을 가능케 했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 활용도와 강한 상관관계를 가지며, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 예측력이 높다는 것을 보여주었습니다. 대조 학습을 통해 모델이 다양한 차원에서 강인함, 크로스 도메인 일반화 및 희소한 데이터로부터의 일반화에서 현저한 개선을 보였습니다. SynesLM 모델은 VisSpeech 데이터셋에서 zero-shot AV-ASR의 Word Error Rate (WER)을 43.4%에서 39.4%로 낮추며 SOTA 성능을 달성했고, VST에서는 BLEU 점수가 37.2에서 43.5로, VMT에서는 54.4에서 54.8로 향상되었습니다.



### Regional quality estimation for echocardiography using deep learning (https://arxiv.org/abs/2408.00591)
- **What's New**: 이번 뉴스레터에서는 최신 AI 연구 논문 세 편을 소개합니다. 첫 번째 논문은 다중 선택형 질문(MCQ) 자동 생성을 개선하기 위한 새로운 평가 메트릭을 제안합니다. 두 번째 논문은 NLP 태스크에서의 모델 강건성(robustness)을 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 연구합니다. 세 번째 논문은 심장 초음파 이미지의 품질 평가를 자동화하는 방법을 제안합니다.

- **Technical Details**: 첫 번째 논문에서는 기존 BLEU, ROUGE, METEOR 메트릭의 한계를 지적하고, 지식 종속 대답 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 메트릭을 제안합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 자동으로 학생의 문제 해결 행동을 모방합니다. 두 번째 논문에서는 반사실적 증강을 통해 여러 개의 반사실적(counterfactual)을 생성하고, 집합적 의사 결정(collective decision)을 통해 모델의 인과 관계를 더 정확하게 파악하는 방법을 제안합니다. 세 번째 논문에서는 세 가지 방법으로 심장 초음파 이미지의 품질을 평가하는 방식을 비교했습니다: 1) 전통적인 픽셀 기반 메트릭(gCNR), 2) U-Net을 사용한 지역 이미지 일관성, 3) 엔드 투 엔드(End-to-end) 딥 러닝 모델.

- **Performance Highlights**: 첫 번째 논문에서 KDA_disc와 KDA_cont는 실제 교실 환경에서 전문가들이 라벨링한 사용성과 강한 상관관계를 보였습니다. 두 번째 논문에서는 집합적 의사 결정을 통해 반사실적 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희소 데이터 일반화에서 유의미한 향상을 이루었습니다. 세 번째 논문에서 엔드 투 엔드 모델은 가장 높은 성과를 보여주었으며, 전문가들의 수동 라벨링과 우수한 상관관계(Spearman correlation, ρ = 0.69)를 보였습니다.



### High-Quality, ROS Compatible Video Encoding and Decoding for High-Definition Datasets (https://arxiv.org/abs/2408.00538)
- **What's New**: 우리 팀은 자동 MCQ 생성의 평가 메트릭을 개선하기 위해 새로운 메트릭인 Knowledge Dependent Answerability (KDA)을 도입했습니다. 이 메트릭은 MCQ의 답변 가능성(answerability)과 학생의 지식 평가 능력을 측정합니다. 또한, NLP 태스크에서 모델의 robustness를 강화하기 위해 대조 학습(contrastive learning)과 counterfactual 증강을 활용하는 접근 방식을 제안했습니다. 마지막으로, 로봇 데이터셋의 비디오 데이터를 압축하기 위해 최신 비디오 인코더를 사용한 연구 결과를 소개합니다.

- **Technical Details**: MCQ 생성 평가 메트릭으로 KDA를 도입하여 학생의 문제 해결 행동을 모방하는 pretrained language model을 활용했습니다. NLP 모델의 robustness를 향상시키기 위해 여러 counterfactual을 생성하고 집합적 의사 결정을 통해 단어의 인과 관계를 강화하는 방법을 제안했습니다. SLAM(동시 위치 추정 및 지도 작성) 알고리즘의 평가를 위한 고해상도 비디오 데이터를 압축하기 위해 H.264, H.265, AV1 코덱을 평가했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 사용성과 전문가 레이블 MCQ 품질 측정치와 강한 상관관계를 보였습니다. NLP 모델에서 대조 학습을 적용한 결과, 기존 방법보다 반목되지 않는 패턴을 줄이는 데 효과적인 것으로 나타났습니다. 로봇 데이터셋 비디오 인코딩에서는 높은 화질을 유지하면서도 적절한 저장 크기를 확보할 수 있음을 실험적으로 입증했습니다.



### GalleryGPT: Analyzing Paintings with Large Multimodal Models (https://arxiv.org/abs/2408.00491)
Comments:
          Accepted as Oral Presentation at ACM Multimedia 2024

- **What's New**: 자동 MCQ 생성을 위한 새로운 평가 메트릭 Knowledge Dependent Answerability (KDA)을 제안하며, 이는 MCQ의 대답 가능성(answerability)을 측정하여 학생의 지식 평가를 가능하게 합니다. 또한, 딥 러닝 모델의 취약성을 극복하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 접근 방식을 소개합니다. 마지막으로, 예술 작품 분석을 위한 포괄적인 형식 분석을 제안하고 이를 위해 커다란 멀티 모달 모델 GalleryGPT를 개발하였습니다.

- **Technical Details**: KDA는 학생의 문제 해결 행동을 모방하기 위한 사전 학습된 언어 모델을 이용하여 자동 평가 메트릭(KDA_disc, KDA_cont)을 도입하였습니다. 딥 러닝 모델의 경우, '여러 개의' 반사실(counterfactual)을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 보장할 수 있는 robust한 방법을 제안합니다. 예술 작품 분석에서는 PaintingForm이라는 대규모 데이터셋과 LLaVA 아키텍처를 기반으로 한 GalleryGPT 모델을 소개하며, 이는 시각적 요소에 초점을 맞춘 형식 분석을 생성합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅과 강한 상관관계를 가지며, n-gram 기반 메트릭과 결합하여 전문가 레이블이 붙은 다양한 MCQ 품질 측정 지표에 강력한 예측력을 보입니다. 딥 러닝 모델의 robust 접근 방식은 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터로부터의 일반화 능력에서 상당한 성능 향상을 보여주었습니다. GalleryGPT는 다양한 예술 분석 작업에서 우수한 성능을 나타내며, 특히 수집된 데이터 셋의 우수성과 모델의 시각적 이해 및 분석 능력을 입증했습니다.



### A Systematic Review on Long-Tailed Learning (https://arxiv.org/abs/2408.00483)
Comments:
          Current Under Revision at IEEE TNNLS. [This is the long/Full-length version of our Long-Tailed Learning Survey paper]

- **What's New**: 자동 MCQ(다지선다 질문) 생성의 평가 메트릭에 대한 새로운 접근인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었습니다. 또한 최신 딥러닝 모델의 강건성을 향상하기 위해 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 이용하는 새로운 방법론이 소개되었습니다. 마지막으로, 롱테일 학습의 최근 발전을 포괄적으로 조사하고 이를 8가지 차원으로 분류하는 새로운 분류 체계가 제안되었습니다.

- **Technical Details**: MCQ 생성의 새로운 평가 메트릭인 KDA는 학생이 대상 지식에 대한 이해를 바탕으로 대답 가능성을 평가합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하며, 이들은 실제 강의실 세팅의 사용성과 높은 상관관계를 가지고 있습니다. 딥러닝 모델의 강건성 개선을 위한 새로운 접근은 반사실을 집합적으로 생성하여 각 용어의 인과 관계를 더 강건하게 감독할 수 있도록 합니다. 롱테일 학습에 대한 포괄적인 조사는 데이터 균형 조정, 신경망 아키텍처, 특징 풍부화, 손실 함수 조정 등 8가지 차원을 포함합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 강한 상관관계를 보이며, 전문가가 라벨링한 다양한 MCQ 품질 메트릭에 대해 높은 예측력을 보입니다. 반사실 증강 방법론은 반사실 강건성, 도메인 간 일반화, 소량 데이터 일반화 등 다양한 측면에서 상당한 개선을 달성했습니다. 롱테일 학습 방법들은 객체 인식, 객체 탐지, 이미지 분할 등의 하위 작업에서 꼬리 클래스의 인식 정확도를 크게 향상시킵니다.



### DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving (https://arxiv.org/abs/2408.00415)
Comments:
          19 pages, 9 figures

- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 도입했습니다. 또한, 최근 NLP 태스크에서의 deep 모델의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 적용한 연구가 발표되었습니다. 마침내, DriveArena라는 고충실도 폐쇄 루프 시뮬레이션 시스템이 자율 주행 에이전트를 위한 새로운 플랫폼으로 소개되었습니다.

- **Technical Details**: MCQ 생성 평가를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였습니다. 이들은 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. NLP 모델의 robustness를 위해 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 용어의 인과 관계에 대해 robust하게 감독하는 방법을 제안했습니다. DriveArena는 Traffic Manager와 World Dreamer로 구성된 모듈형 설계를 채택하였으며, 다양한 도시의 로드맵을 바탕으로 현실적인 교통 흐름을 생성할 수 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가들이 평가한 실제 강의실에서의 사용성과 강한 상관관계를 나타냈으며, MCQ 품질 측정의 예측력도 높였습니다. DriveArena는 기존 방식보다 덜 감각적으로 편향된 결과를 보여주며 다양하고 복잡한 시나리오에서 자율 주행 에이전트의 학습과 진화를 가능하게 합니다.



### Enhancing Whole Slide Pathology Foundation Models through Stain Normalization (https://arxiv.org/abs/2408.00380)
Comments:
          13 pages, 8 figures

- **What's New**: 자동으로 Multiple Choice Questions(MCQ)을 생성하는 새로운 평가 메트릭을 도입했습니다. 지식 종속 가능성(KDA)이라고 불리는 이 메트릭은 학생들이 목표 사실에 대한 지식을 평가할 수 있는 MCQ의 대답 가능성을 측정합니다. 또한, 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안하여 사람의 응답을 통해 KDA를 근사합니다.

- **Technical Details**: 제안된 KDA는 기존 평가 메트릭(BLEU, ROUGE, METEOR)이 놓치고 있는 교육적 가치를 반영합니다. KDA 기반 메트릭은 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모사합니다. 또한, Human study를 통해 이러한 메트릭이 실제 교육 현장에서 얼마나 유용한지 검증됐습니다.

- **Performance Highlights**: Human study 결과, KDA_disc와 KDA_cont는 KDA 및 전문가가 지정한 실제 교실에서의 사용성에 대한 강한 상관관계를 보였습니다. KDA_disc와 KDA_cont를 n-gram 기반 유사성 메트릭과 결합할 때, 이는 전문가가 지정한 다양한 MCQ 품질 측정항목에 대한 예측력을 높이는 데 큰 기여를 했습니다.



### Conformal Trajectory Prediction with Multi-View Data Integration in Cooperative Driving (https://arxiv.org/abs/2408.00374)
- **What's New**: 이번 뉴스레터에서는 최신 연구 논문의 발전 사항을 살펴봅니다. 각 연구들은 교육, 자연어 처리, 그리고 자율주행과 관련된 다양한 주제들을 다룹니다.

- **Technical Details**: [{'Title': 'Knowledge Dependent Answerability (KDA) 메트릭을 사용한 자동 Multiple Choice Question (MCQ) 생성 평가', 'Content': '기존의 MCQ 생성 평가 방법은 교육적 가치를 고려하지 않고 n-gram 유사성에만 의존하고 있습니다. 이를 해결하기 위해 우리는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 대상 사실에 대한 학생의 지식을 측정하면서 MCQ의 대답 가능성을 평가합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강하게 연관되어 있음을 입증했습니다.'}, {'Title': 'Robustness 강화 위한 Contrastive Learning 및 Counterfactual Augmentation', 'Content': '최신 deep model들이 자연어 처리(NLP) 태스크에서 높은 정확성을 보이지만, spurious patterns에 의존해 robustness가 제한된다는 문제점이 있습니다. 이 논문에서는 contrastive learning 및 counterfactual augmentation을 활용하여 robustness를 강화하는 방법을 제안합니다. 기존의 augmentation 방법과 달리, 여러 개의 counterfactual을 생성하고 collective decisions를 통해 단어의 인과관계를 파악하는 방식입니다.'}, {'Title': 'V2INet: Multi-view 데이터를 통한 자율주행 경로 예측 프레임워크', 'Content': 'V2INet이라는 새로운 경로 예측 프레임워크를 소개합니다. 이 모델은 vehicle-to-vehicle (V2V) 및 vehicle-to-infrastructure (V2I) 통신기술을 활용해 멀티뷰 데이터를 모델링합니다. 기존에는 멀티뷰 데이터를 수동으로 결합하거나 별도의 훈련 단계를 필요로 했지만, V2INet은 end-to-end 훈련을 지원하여 성능과 유연성을 높입니다. 또한, conformal prediction 모듈을 사용하여 예측 결과의 신뢰 구간을 제공하여 예측의 신뢰성을 크게 높였습니다.'}]

- **Performance Highlights**: [{'Title': 'KDA_disc 및 KDA_cont의 예측 성능', 'Content': 'KDA_disc와 KDA_cont가 다양한 전문가가 라벨링한 MCQ 품질 척도에서 강력한 예측력을 보였습니다.'}, {'Title': '강화된 robustness', 'Content': '우리의 접근 방식은 여러 측면에서 significant한 향상을 이루었으며, 특히 counterfactual robustness, cross-domain generalization, 그리고 데이터가 제한된 환경에서의 일반화 성과에서 두드러졌습니다.'}, {'Title': 'V2INet의 경로 예측 성능', 'Content': 'V2INet은 실험을 통해 Final Displacement Error (FDE)와 Miss Rate (MR) 측면에서 뛰어난 성능을 보여주었습니다. 이 시스템은 단일 GPU에서 실행되었습니다.'}]



### Multimodal Fusion and Coherence Modeling for Video Topic Segmentation (https://arxiv.org/abs/2408.00365)
- **What's New**: 이 연구는 비디오 주제 분할(Video Topic Segmentation, VTS)을 다루며, 비디오를 명확한 주제로 나누는 과제를 향상시키는 새로운 기법을 제안합니다. 특히, 교육 비디오를 대상으로 다루어 학습 경험을 증대시키는 데 초점을 맞추고 있습니다. 또한, 새로운 대규모 중국어 강의 비디오 데이터셋을 소개해 연구 확대를 도모합니다.

- **Technical Details**: 제안된 방법은 크로스-어텐션(cross-attention)과 전문가 혼합(Mixture-of-Experts, MoE) 아키텍처를 활용한 멀티모달 융합을 탐구하고, 멀티모달 대조 학습(multimodal contrastive learning)을 통해 모델을 사전 학습 및 미세 조정(fine-tuning)합니다. 또한, VTS 작업에 맞춘 새로운 사전 학습 과제와 미세 조정 작업을 제안합니다. 이 접근 방식은 강의 비디오와 같은 교육 비디오에서 평가되었으며, 중국어 및 영어 데이터셋에서 최고의 성능을 보였습니다.

- **Performance Highlights**: 제안된 모델은 다양한 교육 비디오 데이터셋에서 기존의 비감독 및 감독 기반 모델을 능가하는 SOTA (state-of-the-art) 성능을 달성했습니다. 또한 세밀한 분해 연구를 통해 이 접근 방식의 효과를 추가로 검증했습니다.



### IN-Sight: Interactive Navigation through Sigh (https://arxiv.org/abs/2408.00343)
Comments:
          The 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 자동 MCQ 생성의 평가 메트릭으로 지식 종속 가능성(KDA)을 제안하며, 딥 러닝 모델의 강건성을 위해 대조 학습 및 반사실적 증강(counterfactual augmentation) 방법을 도입했습니다. 또한, IN-Sight라는 자가 지도 학습 기반 경로 계획 시스템을 제안하여 장애물 상호작용을 통해 더 나은 내비게이션 전략을 구현합니다.

- **Technical Details**: 1. 자동 MCQ 생성: BLEU, ROUGE, METEOR 등의 기존 메트릭의 한계를 해결하기 위해 KDA를 제안. KDA_disc와 KDA_cont라는 자동 평가 메트릭을 통해 KDA를 근사하여 학생의 문제 해결 행동을 모방함.
2. 딥 러닝 모델의 강건성: 여러 개의 counterfactual을 생성하고 집합적 의사결정(collective decisions) 방식을 통해 단어 간 인과관계를 파악. 다양한 차원에서 강건성을 크게 향상시킴.
3. IN-Sight: RGB-D 관찰을 통해 이동 가능 점수(traversability scores)를 계산하고, 이를 이용해 복잡한 환경에서 장거리 경로 계획 수행. 로컬 플래너는 표현 학습기법을 사용하여 임퍼러티브 차별 가능 지도(differentiable costmap)로 훈련됨.

- **Performance Highlights**: 1. 자동 MCQ 생성 평가 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 보임. 
2. 다양한 평가를 통해 반사실적 강건성 및 크로스 도메인 일반화 성능이 향상됨.
3. IN-Sight는 Intel SPEAR 시뮬레이터에서의 훈련을 통해 강력한 경로 계획 능력을 입증, 실제 환경에서도 바로 적용 가능함을 보여줌.



### ADBM: Adversarial diffusion bridge model for reliable adversarial purification (https://arxiv.org/abs/2408.00315)
Comments:
          20 pages

- **What's New**: MCQ 자동 생성을 위한 새로운 평가 메트릭으로 KDA(Knowledge Dependent Answerability)를 제안합니다. 기존 평가 메트릭들은 n-gram 유사성에만 치중해 교육적 가치를 평가하지 못하였으나, KDA는 MCQ의 대답 가능성을 통해 학생의 지식을 평가합니다.

- **Technical Details**: KDA는 학생의 지식에 따라 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 자동 평가 지표를 제안하여, 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 높은 상관관계를 가짐을 확인하였습니다. n-gram 기반 유사성 메트릭과 결합할 경우, 다양한 전문가 평가 MCQ 품질 지표에 대해 높은 예측력을 보였습니다.



### Navigating Text-to-Image Generative Bias across Indic Languages (https://arxiv.org/abs/2408.00283)
Comments:
          Accepted in ECCV 2024

- **What's New**: 다양한 언어 및 문화적 요소를 반영하여 텍스트-이미지 생성 모델(TTI)을 평가하는 새로운 벤치마크 'IndicTTI'가 제안되었습니다. 이 벤치마크는 인도 전역에서 사용되는 인도 언어에 대한 TTI 모델의 생성을 평가하고, 이를 통해 영어 외 언어에서도 질 높은 이미지 생성을 목표로 합니다.

- **Technical Details**: IndicTTI 벤치마크는 30개의 인도 언어와 영어를 포함하여 다소 복잡한 프롬프트를 사용하여 이미지를 생성합니다. COCO-NLLB 데이터셋을 샘플로 사용하여 1000개의 다양한 이미지-캡션 쌍을 선정하고, 이를 통해 모델의 생성 성능 및 문화적 적합성을 평가합니다. 평가는 두 가지 측면에서 이루어지며, 하나는 생성 결과의 정확성(correctness)을, 다른 하나는 표현 다양성(representation)을 측정합니다.

- **Performance Highlights**: 본 연구에서는 Stable Diffusion, Alt Diffusion, Midjourney, 그리고 Dalle3와 같은 모델을 이용하여 각기 다른 언어 및 스크립트의 문화적 표현을 평가합니다. 특히, 실험에서 특정 언어 스크립트가 인도 신들에 대한 이미지 생성에 어떻게 영향을 미치는지, 아시아 여성과 커플의 생성 빈도, Dalle3 모델의 높은 정확도 등을 관찰하였습니다. 이를 통해 각 TTI 모델의 강점과 개선이 필요한 부분을 밝혀냈습니다.



### 3D U-KAN Implementation for Multi-modal MRI Brain Tumor Segmentation (https://arxiv.org/abs/2408.00273)
- **What's New**: 교육적 가치를 더 잘 반영하기 위해 새로운 MCQ(auto generated Multiple Choice Questions) 평가 메트릭을 제안했습니다. 또한 NLP 태스크의 강화된 robust 성능을 위한 contrastive learning과 counterfactual augmentation 기법이 제안되었으며, 3D 뇌종양 세분화(Brain Tumor Segmentation)를 위해 U-Net 기반 네트워크를 Kolmogorov-Arnold Network(KAN) 레이어로 강화한 U-KAN 및 새로운 UKAN-SE를 도입했습니다.

- **Technical Details**: 지식 종속 가능성(KDA)을 통해 MCQ의 대답 가능성을 측정하고, human survey로부터 학생의 응답을 바탕으로 KDA를 측정했습니다. NLP 태스크의 robustness를 높이기 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 spurious correlation을 줄이는 방법을 제안했습니다. U-KAN과 UKAN-SE는 다중 모드 MRI 데이터를 사용하여 3D 뇌종양 세분화를 실시하며, Squeeze-and-Excitation 모듈을 통해 global attention을 구현했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 제안된 NLP 기법은 반사실적 robustness, cross-domain generalization 및 희소 데이터에서의 generalization에서 유의미한 개선을 보였습니다. U-KAN과 UKAN-SE는 U-Net 등 기존 모델보다 뛰어난 성능을 보여주었으며, UKAN-SE는 약간 더 우수한 성과를 나타냈습니다. 이 모델들은 약 10.6백만개의 파라미터만을 사용하면서도 훈련 시간 측면에서 U-Net과 Attention U-Net 대비 1/4, Swin UNETR 대비 1/6의 시간이 소요되었습니다.



### Revocable Backdoor for Deep Model Trading (https://arxiv.org/abs/2408.00255)
Comments:
          to appear in ECAI 2024

- **What's New**: 최근 연구에서는 자동 MCQ 생성의 평가 메트릭이 교육적인 가치를 충분히 반영하지 못한다는 문제를 지적하고 있습니다. 이에 따라, 우리는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하여 MCQ의 교육적 가치를 평가하는 방법을 제시했습니다. 또한 최근의 NLP 태스크에서 deep models의 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 방법을 제안했습니다. 마지막으로, deep learning 모델에서 backdoor 공격은 일반적으로 보안 취약점으로 간주되지만, 이를 긍정적으로 활용하는 'Revocable Backdoor'라는 새로운 개념을 도입하고, 모델 거래 시나리오에서의 적용 가능성을 논의했습니다.

- **Technical Details**: 첫 번째 논문의 새로운 메트릭 KDA는 학생들이 실제로 문제를 풀면서 제공하는 응답을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하였습니다. 두 번째 논문에서는 다중 counterfactual을 생성하고 집합적 의사 결정을 통해 모델의 인과관계를 보다 robust하게 파악하는 방법을 제안하였습니다. 마지막 논문에서는, 'Revocable Backdoor'를 통해 백도어가 심어졌으나 성능 저하 없이 쉽게 제거할 수 있는 새로운 deep model 거래 시나리오를 제안했습니다. 이 개념을 위해 특정 마스크 행렬을 디자인하여 내부 피처 맵을 관리합니다.

- **Performance Highlights**: 첫 번째 논문에서는 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 보였고, n-gram 기반의 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측 능력을 보였습니다. 두 번째 논문에서는 대조 학습과 counterfactual augmentation을 통해 기존 방법보다 다양한 차원에서 의미 있는 성능 향상을 이뤄냈습니다. 마지막 논문에서는, Revocable Backdoor가 다양한 데이터셋과 네트워크 아키텍처에서의 실험을 통해 그 실용성과 robust함을 확인했습니다.



### multiGradICON: A Foundation Model for Multimodal Medical Image Registration (https://arxiv.org/abs/2408.00221)
- **What's New**: 이번 연구는 자동 MCQ 생성(multiple choice question generation)을 위한 새로운 평가 메트릭인 지식 종속 가능성(knowledge dependent answerability, KDA)을 제안합니다. 이는 기존 평가 메트릭들이 단순히 n-gram 기반 유사성을 평가하는 데 그쳐 교육적 가치를 배제하였음을 해결하고자 하였습니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성(answerability)을 측정하여 대상 사실에 대한 학생의 지식을 평가합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이는 사전훈련된 언어 모델을 이용해 학생들의 문제 해결 행동을 모사함으로써 KDA를 근사합니다.

- **Performance Highlights**: 인간 연구 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 KDA 간에 강한 상관관계를 보여줍니다. 또한, n-gram 기반 유사성 메트릭과 결합될 때 다양한 전문가 라벨링 MCQ 품질 측정에 대해 강한 예측력을 보였습니다.



### Hierarchical Conditioning of Diffusion Models Using Tree-of-Life for Studying Species Evolution (https://arxiv.org/abs/2408.00160)
- **What's New**: 우리는 자동 Multiple Choice Question(MCQ) 생성의 교육적 가치를 평가하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. 또한 Counterfactual Augmentation을 이용하여 NLP 모델의 Robustness를 향상시키는 새로운 접근법과 Phylogenetic Knowledge로 생물 진화 특성을 자동으로 발견하는 Phylo-Diffusion 프레임워크를 소개합니다.

- **Technical Details**: KDA는 학생의 지식을 평가하기 위한 MCQ의 대답 가능성(Answerability)을 측정하며, 인간 평가를 통해 실험되었습니다. 우리는 또한 프리트레인된 언어 모델을 사용하여 이를 자동으로 평가하는 KDA_disc와 KDA_cont 메트릭을 제안합니다. 또한, Counterfactual Augmentation을 통해 모델 편향에 덜 민감한 방법을 제안합니다. Phylo-Diffusion에서는 Phylogenetic Tree의 정보를 담은 HIER-Embed를 이용하여 생성 모델의 임베딩 공간을 구조화하고, Trait Masking과 Trait Swapping 실험을 소개합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 내 사용성과 강한 상관관계를 보였습니다. 또한, 새로운 Counterfactual Augmentation 방법은 Counterfactual Robustness, Cross-Domain Generalization, 그리고 소량 데이터 일반화에 있어 큰 개선을 보였습니다. Phylo-Diffusion은 어류와 조류의 진화적 특성을 의미 있게 포착하며 생물 진화 연구에 새로운 통찰을 제공하였습니다.



### StyleRF-VolVis: Style Transfer of Neural Radiance Fields for Expressive Volume Visualization (https://arxiv.org/abs/2408.00150)
Comments:
          Accepted by IEEE VIS 2024

- **What's New**: 최근 교육자들의 시간 절감과 학생 평가 방안을 위해 새로운 자동 MCQ (Multiple Choice Questions) 생성 방법이 제안되었습니다. 또한, NLP (Natural Language Processing) 태스크에서 더욱 견고한 모델 학습을 위해 contrastive learning과 counterfactual augmentation을 결합한 방안이 연구되었습니다. 마지막으로, 볼륨 비주얼라이제이션 (VolVis)에서의 새로운 스타일 전달 프레임워크 'StyleRF-VolVis'가 소개되었습니다.

- **Technical Details**: 첫 번째 논문에서는 MCQ의 교육적 가치를 평가하기 위해 지식 종속 대답 가능성(KDA)을 측정하는 새로운 자동 평가 메트릭을 제안했습니다. 두 번째 논문에서는 spurious correlations에 영향을 받지 않기 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안했습니다. 마지막 논문에서는 NeRF (neural radiance field) 모델을 활용하여 DVR (Direct Volume Rendering) 장면의 스타일을 변경하면서도 원본 콘텐츠 정보를 유지하는 'StyleRF-VolVis' 기술을 개발했습니다.

- **Performance Highlights**: 첫 번째 연구에서는 KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 높은 상관관계를 보여 주었습니다. 두 번째 연구는 실험 결과, 집합적 의사 결정 방식이 할당 기반 합성의 task model bias에 덜 민감하여 다양한 측면에서 성능이 향상됨을 보였습니다. 세 번째 연구는 StyleRF-VolVis가 다양한 볼륨 렌더링 장면에 대해 기존 AdaIN, ReReVST, ARF, SNeRF 스타일 렌더링 솔루션보다 우수한 품질, 일관성, 유연성을 보였습니다.



### DDU-Net: A Domain Decomposition-based CNN for High-Resolution Image Segmentation on Multiple GPUs (https://arxiv.org/abs/2407.21266)
- **What's New**: 최근 논문에서는 자동 MCQ(Multiple Choice Questions) 생성의 평가 메트릭과 deep learning 모델의 robustness, 그리고 초고해상도 이미지 분할(segment)에 관한 새로운 접근 방법을 제안했습니다.

- **Technical Details**: {'mcq_generation': '기존 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 고려하지 않기 때문에 새로운 지식 종속 가능성(KDA) 메트릭을 제안했습니다. KDA는 학생이 대상 사실에 대한 지식을 가지고 MCQ에 대답할 수 있는 능력을 평가합니다. 이를 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 소개했습니다.', 'robust_nlp_models': 'NLP 태스크에서 deep 모델의 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 반실제 데이터 증가(counterfactual augmentation)를 활용한 방법을 제안했습니다. 기존 방법과 달리, 여러 개의 반실제 데이터를 생성하고 집합적 의사 결정을 통해 인과관계를 더 robust하게 파악했습니다.', 'high_res_segmentation': '초고해상도 이미지의 분할 문제를 해결하기 위해 도메인 분해 기반 U-Net(DDU-Net) 아키텍처를 도입했습니다. 입력 이미지를 독립적인 패치(patch)로 분할해 처리하고, 패치 간 정보를 교환하는 네트워크를 추가해 공간적 맥락을 향상시켰습니다.'}

- **Performance Highlights**: {'mcq_generation': 'Human studies를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 평가한 실제 강의실에서의 사용성과 높은 상관관계를 가짐을 보여주었습니다.', 'robust_nlp_models': '집합적 의사 결정을 통해 다양한 측면에서 significant한 개선을 이루었으며, 예를 들어 반실제 robustness, 크로스 도메인 일반화(cross-domain generalization), 소량 데이터로부터의 일반화(generalization from scarce data) 등에서 뚜렷한 성과를 보였습니다.', 'high_res_segmentation': '실험 결과, 패치 간 통신을 포함한 네트워크가 동일한 네트워크와 비슷한 성능을 보였으며, 교차 네트워크가 포함된 모델이 2-3% 높은 IoU 점수를 기록했습니다.'}



### Perm: A Parametric Representation for Multi-Style 3D Hair Modeling (https://arxiv.org/abs/2407.19451)
Comments:
          Project page: this https URL

- **What's New**: 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 소개합니다. 이 메트릭은 생성된 MCQ(Multiple Choice Questions)의 답변 가능성을 평가하여 교육적 가치를 반영합니다. 또한, 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 n-gram 기반의 유사성만을 평가하는 반면, KDA는 학생의 실제 지식 평가 능력을 고려합니다.

- **Technical Details**: KDA는 인간 설문 조사를 통해 학생의 답변을 기반으로 측정됩니다. 또한, 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안하여 이를 흉내내는 사전 훈련된 언어 모델을 활용합니다. 이를 통해 MCQ의 교육적 유용성과 강한 상관관계를 가진 결과를 도출합니다.

- **Performance Highlights**: 우리의 연구는 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 전문가가 라벨링한 MCQ 품질 측정에서 강한 예측 능력을 가지고 있음을 보여줍니다.



New uploads on arXiv(cs.AI)

### DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency (https://arxiv.org/abs/2408.00741)
- **What's New**: 자동 다지선다형 질문(MCQ) 생성을 위한 새로운 평가 기준인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)를 제안합니다. 또한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 KDA를 근사하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모사합니다.

- **Technical Details**: 기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, 그리고 METEOR는 데이터셋 내의 골드 샘플과의 n-gram 유사성을 중점적으로 평가해 교육적 가치를 무시합니다. KDA는 대상 사실에 대한 학생의 지식을 평가하여 MCQ의 대답 가능성을 측정합니다. Human survey를 통해 KDA를 측정하고, KDA_disc와 KDA_cont는 예측 모델을 통해 이를 자동으로 수행합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 전문가가 라벨링한 다양한 MCQ 품질 측정에서 큰 예측 능력을 보였습니다.



### An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models (https://arxiv.org/abs/2408.00724)
- **What's New**: 새로운 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭을 제안하여 MCQ(Multiple Choice Questions)의 교육적 가치를 평가하는 것을 제안했습니다. 이는 기존의 BLEU, ROUGE, METEOR 메트릭과 달리 학생이 특정 사실에 대한 지식을 정확하게 평가할 수 있는지 측정합니다. 또한, LLM(Large Language Models)의 추론 시간 동안의 컴퓨팅 최적화에 관한 새로운 연구가 진행되었습니다. 다양한 추론 전략을 평가하여, 소규모 모델이 더 복잡한 알고리즘과 결합할 때 효율적일 수 있다는 결과를 도출했습니다.

- **Technical Details**: MCQ 평가를 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안했습니다. 이들은 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다. 추론 최적화를 위해 Greedy Search, Majority Voting, Best-of-N, Weighted Voting 등의 추론 전략을 사용했습니다. 새로운 REBASE (REward BAlanced SEarch) 알고리즘도 제안되어, 더 작은 모델에서도 높은 성능을 유지할 수 있도록 했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 강한 상관관계를 보였습니다. 추론 최적화 실험 결과에서는 소규모 Llemma-7B 모델이 Llemma-34B 모델과 비교하여 2배 적은 FLOPs를 사용하면서도 유사한 정확도를 달성했습니다. REBASE 알고리즘을 통해 소규모 모델의 성능이 향상되었으며, 실험 결과 모든 설정과 태스크에서 우수한 성과를 보였습니다.



### Can Developers Prompt? A Controlled Experiment for Code Documentation Generation (https://arxiv.org/abs/2408.00686)
Comments:
          Accepted at the 40th IEEE International Conference on Software Maintenance and Evolution (ICSME)

- **What's New**: 이번 연구에서는 기존의 MCQ 생성 평가 메트릭들이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안했습니다. 이 메트릭은 학생이 대상 사실(target fact)을 알고 있을 때 MCQ의 답변 가능성(answerability)을 측정합니다. 또한, NLP 태스크에서 deep model의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 결합한 새로운 방법론이 제안되었습니다. 마지막으로, 대형 언어 모델(LLM)을 활용한 코드 문서화 자동화 가능성을 연구했으며, 전문가와 학생들이 LLM을 효과적으로 활용할 수 있는지에 대한 실험 결과를 보고했습니다.

- **Technical Details**: MCQ 생성을 위한 새로운 평가 메트릭인 KDA는 학생 응답을 기반으로 측정된 KDA와 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한 KDA_disc, KDA_cont 두 가지 자동 평가 메트릭을 사용합니다. NLP 태스크에서의 robustness 향상을 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악합니다. 코드 문서화 자동화를 위한 연구는 ChatGPT와 유사한 Visual Studio Code 확장을 사용하여 ad-hoc 프롬프트와 준비된 few-shot 프롬프트 간의 성능을 비교했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가 결과와 실제 강의실에서의 사용성에서 강한 상관관계를 나타내며, n-gram 기반 메트릭과 결합했을 때 다양한 전문가 평가 MCQ 품질 측정에 대한 예측력이 뛰어납니다. 제안된 NLP 모델은 counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 측면에서 상당한 개선을 보여주었습니다. LLM을 활용한 코드 문서화 연구에서는 전문가가 'Docstring' 키워드를 포함한 ad-hoc 프롬프트로 더 높은 품질의 문서화를 생성했으며, 학생들은 즉각적인 프롬프트 작성 지원을 원했습니다.



### SentenceVAE: Faster, Longer and More Accurate Inference with Next-sentence Prediction for Large Language Models (https://arxiv.org/abs/2408.00655)
Comments:
          First preview version

- **What's New**: {'Paper1': '이번 연구에서는 자동 MCQ 생성의 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다. 이 메트릭은 학생들이 특정 지식을 바탕으로 MCQ에 답할 수 있는 능력을 측정합니다.', 'Paper2': 'NLP 작업에서 깊은 모델의 강인성(robustness) 향상을 위해 대조학습(contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용하는 새로운 접근법을 제안합니다. 여러 반사실적 사례를 생성하고 집합적 의사 결정(collective decisions)을 통해 강인성을 확보합니다.', 'Paper3': '대규모 언어 모델(LLMs)의 추론 속도를 높이기 위해 새로운 SentenceVAE 모델을 도입하여 다음 문장 예측 방법을 제안합니다. 이는 LLM의 입력과 출력 레이어에 SentenceVAE를 통합하여 문장 단위 추론 방식을 사용합니다.'}

- **Technical Details**: {'Paper1': 'KDA를 측정하기 위해 학생 응답을 바탕으로 한 인간 설문을 활용하는 방법을 제안하고, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안합니다. 이들은 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방합니다.', 'Paper2': '기존의 반사실적 증강 방식은 인간이 직접 생성하거나 데이터셋에서 반사실적 사례를 찾아내는 방식이었습니다. 제안된 방법은 여러 반사실적 사례를 합성하고, 이를 기반으로 한 집합적 의사 결정으로 각 용어의 인과성을 평가합니다.', 'Paper3': 'SentenceVAE는 인코더와 디코더로 구성되어 있으며, 문장의 정보를 하나의 토큰으로 압축하고 이를 다시 문장 형태로 복원합니다. LLM의 추론 과정에서 자가 주의 계산의 메모리 요구를 줄이고, 같은 문맥 길이에서도 메모리 오버헤드를 줄입니다.'}

- **Performance Highlights**: {'Paper1': 'KDA_disc와 KDA_cont는 인간 설문을 통한 실험에서 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가 레이블 MCQ 품질 측정에 대한 예측력을 보여주었습니다.', 'Paper2': '제안된 방법은 반사실적 강인성, 도메인 간 일반화, 그리고 희소 데이터에서의 일반화 등 여러 측면에서 성능 향상을 보여주었습니다.', 'Paper3': 'SentenceVAE를 이용한 새로운 추론 방법은 추론 속도를 204~365% 증가시키고, 동일한 문맥 길이에 대해 메모리 오버헤드를 86~91% 감소시켰습니다. 모델 규모가 커질수록 이점이 더욱 증대되었습니다.'}



### Unlocking Fair Use in the Generative AI Supply Chain: A Systematized Literature Review (https://arxiv.org/abs/2408.00613)
- **What's New**: 새로운 평가 메트릭, KDA(Knowledge Dependent Answerability),를 제안하여 자동 MCQ 생성에서 교육적 가치를 반영합니다. 또한, 대조 학습과 역조건 학습(counterfactual augmentation)을 활용하여 NLP 태스크의 robustness를 향상시키는 방법이 제안되었습니다. 마지막으로, Generative AI(GenAI) 공급 체인에서 각 이해관계자의 기대와 목표를 체계화하여 공정 사용 논쟁과 관련된 연구의 틈을 밝히고자 합니다.

- **Technical Details**: 1) BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭은 n-gram 기반의 유사성에 치중하였으나, KDA는 대상 사실에 대한 지식이 주어진 상황에서 MCQ의 대답 가능성을 평가합니다. 2) 대조 학습 및 역조건 학습을 통해 여러 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 사용하여 단어의 인과 관계를 robust하게 판단합니다. 3) PRISM 데이터 구조에 따라 문헌 리뷰를 체계적으로 진행하여 GenAI 공급 체인 내의 다양한 이해관계자의 기대와 목표를 분석하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서 전문가가 명명한 사용성과 강한 상관관계를 보였으며, n-gram 기반 평가 메트릭과 결합되었을 때 예측력이 높아졌습니다. 대조 학습을 활용한 방법은 counterfactual robustness, cross-domain generalization 및 적은 데이터 상황에서의 성능 향상이 있었습니다. GenAI의 공정 사용 논쟁과 관련된 문헌 리뷰를 통해 새로운 연구 틈과 향후 연구 방향을 제시했습니다.



### Illustrating Classic Brazilian Books using a Text-To-Image Diffusion Mod (https://arxiv.org/abs/2408.00544)
Comments:
          7 pages, 2 figures

- **What's New**: 최근에 자동 다중선택질문(MCQ)의 생성은 교사들이 학생 평가에 드는 시간을 크게 줄일 수 있는 잠재력을 가지고 있습니다. 그러나 기존의 평가 지표인 BLEU, ROUGE, METEOR는 교육적 가치를 반영하지 못하고 단어 유사성에만 집중합니다. 이에 우리는 새로운 자동 평가 메트릭인 지식 종속 대답 가능성(KDA)을 제안하여, MCQ가 학생의 지식을 평가하는 능력을 측정하려 합니다.

- **Technical Details**: 우리는 KDA를 측정하는 방법으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이는 사전 학습된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방합니다. 우리는 인간 연구를 통해 이 두 메트릭이 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지는 것을 확인했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont를 사용한 결과, 이러한 메트릭이 전문가가 레이블링한 다양한 MCQ 품질 측정 항목에 대해 강한 예측력을 가지는 것을 보여주었습니다.



### A new approach for encoding code and assisting code understanding (https://arxiv.org/abs/2408.00521)
Comments:
          10 page, 14 figures

- **What's New**: 본 논문은 자동으로 Multiple Choice Questions (MCQ)를 생성하는 과정의 평가 메트릭의 한계를 극복하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭은 생성된 MCQ가 데이터셋의 골드 샘플과 n-gram 유사성만을 고려하여 교육적 가치를 무시하는 문제점이 있습니다. 이를 해결하기 위해 KDA는 MCQ의 '답변 가능성'을 측정하고 학생의 대상 사실에 대한 지식 평가 능력을 평가합니다. 또 다른 연구는 GPT 모델의 자동 회귀(next-word prediction) 패러다임의 한계를 극복하기 위해 대조 학습(contrastive learning) 및 반사실적 증강(counterfactual augmentation)을 활용하여 강건성을 향상시키는 방법을 제안합니다. 추가적으로 코드 이해를 위한 새로운 패러다임과 Diffusion 기술을 적용한 코드 생성을 제안합니다.

- **Technical Details**: KDA는 사람의 응답을 기반으로 KDA를 측정하는 방법을 제안하며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 KDA_disc 와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다. 또한, Diffusion 기술을 활용하여 자연어가 아닌 이질적인 이미지 형식으로 코드를 인코딩하는 새로운 방법을 제안합니다. 이를 통해 코드 이해와 생성을 위한 새로운 텍스트-코드 인코더 모델을 설계하였습니다.

- **Performance Highlights**: 인간 설문조사를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 레이블링 한 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 확인하였습니다. 또한, 대조 학습과 반사실적 증강을 통해 강건성, 도메인 간 일반화(cross-domain generalization), 희소 데이터로부터의 일반화에서 유의미한 성과를 보였습니다. 코드 이해 뿐만 아니라 고품질 코드 생성을 위한 베이스가 되는 Diffusion 기술 기반의 새로운 패러다임을 제안합니다.



### HBot: A Chatbot for Healthcare Applications in Traditional Chinese Medicine Based on Human Body 3D Visualization (https://arxiv.org/abs/2408.00481)
Comments:
          System Demonstration

- **What's New**: 기존의 평가 메트릭은 MCQ의 교육적 가치를 고려하지 않아서, 이 논문에서는 학생의 지식을 평가하는 능력을 측정하기 위해서 지식 종속 가능성(KDA; Knowledge Dependent Answerability)이라는 새로운 메트릭을 제안하였습니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 MCQ의 대답 가능성을 측정하며, pre-trained 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이는 BLEU, ROUGE, METEOR 등의 n-gram 기반 메트릭과 결합하여 더 정교한 평가를 가능하게 합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 전문가가 레이블링한 다양한 MCQ 품질 측정값에 대해 강한 예측력을 가집니다.



### Ontological Relations from Word Embeddings (https://arxiv.org/abs/2408.00444)
- **What's New**: 이 논문에서는 기존 n-gram 기반 평가 메트릭이 Multiple Choice Questions(MCQ)의 학습 평가 능력을 충분히 반영하지 못하는 문제를 해결하기 위해, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했다. KDA는 MCQ가 특정 목표 사실에 대한 지식을 평가하는 능력을 자동으로 측정한다.

- **Technical Details**: KDA를 계산하기 위해 우선 학생의 응답을 기반으로 한 인간 설문조사를 사용하여 KDA를 측정하는 방법을 보여준다. 또한, 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안한다.

- **Performance Highlights**: 인간 연구를 통해, KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성 및 다양한 전문가가 라벨한 MCQ 품질 측정에서 높은 예측력을 가짐을 입증하였다. n-gram 기반 유사성 메트릭과 결합 시, KDA_disc와 KDA_cont는 강력한 예측력을 보여주었다.



### Unsupervised Pairwise Causal Discovery on Heterogeneous Data using Mutual Information Measures (https://arxiv.org/abs/2408.00399)
Comments:
          26th International Conference of the Catalan Association for Artificial Intelligence

- **What's New**: 최근 자동 여러 가지 선택 문제 (MCQ) 생성을 위한 새로운 평가 메트릭이 제안되었습니다. BLEU, ROUGE, METEOR와 같은 기존 메트릭은 MCQ의 교육적 가치를 무시하고 단순히 n-gram 기반 유사성에 초점을 맞췄습니다. 이에 대비해, 논문은 'Knowledge Dependent Answerability (KDA)'라는 새로운 자동 평가 메트릭을 도입하여 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가하는 능력을 강조합니다.

- **Technical Details**: KDA는 인간 설문조사를 통해 학생의 응답을 바탕으로 측정됩니다. 추가적으로, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont가 제안되었습니다. 이는 학생들이 실제 문제를 풀 때의 행동을 모방하여 MCQ의 유용성을 평가합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 상황에서의 사용성과 높은 상관관계를 가지는 것이 증명되었습니다. KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합될 때 전문가가 라벨링한 MCQ 품질 측정에서 강력한 예측력을 보여줍니다.



### Conformal Trajectory Prediction with Multi-View Data Integration in Cooperative Driving (https://arxiv.org/abs/2408.00374)
- **What's New**: 이번 뉴스레터에서는 최신 연구 논문의 발전 사항을 살펴봅니다. 각 연구들은 교육, 자연어 처리, 그리고 자율주행과 관련된 다양한 주제들을 다룹니다.

- **Technical Details**: [{'Title': 'Knowledge Dependent Answerability (KDA) 메트릭을 사용한 자동 Multiple Choice Question (MCQ) 생성 평가', 'Content': '기존의 MCQ 생성 평가 방법은 교육적 가치를 고려하지 않고 n-gram 유사성에만 의존하고 있습니다. 이를 해결하기 위해 우리는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 대상 사실에 대한 학생의 지식을 측정하면서 MCQ의 대답 가능성을 평가합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강하게 연관되어 있음을 입증했습니다.'}, {'Title': 'Robustness 강화 위한 Contrastive Learning 및 Counterfactual Augmentation', 'Content': '최신 deep model들이 자연어 처리(NLP) 태스크에서 높은 정확성을 보이지만, spurious patterns에 의존해 robustness가 제한된다는 문제점이 있습니다. 이 논문에서는 contrastive learning 및 counterfactual augmentation을 활용하여 robustness를 강화하는 방법을 제안합니다. 기존의 augmentation 방법과 달리, 여러 개의 counterfactual을 생성하고 collective decisions를 통해 단어의 인과관계를 파악하는 방식입니다.'}, {'Title': 'V2INet: Multi-view 데이터를 통한 자율주행 경로 예측 프레임워크', 'Content': 'V2INet이라는 새로운 경로 예측 프레임워크를 소개합니다. 이 모델은 vehicle-to-vehicle (V2V) 및 vehicle-to-infrastructure (V2I) 통신기술을 활용해 멀티뷰 데이터를 모델링합니다. 기존에는 멀티뷰 데이터를 수동으로 결합하거나 별도의 훈련 단계를 필요로 했지만, V2INet은 end-to-end 훈련을 지원하여 성능과 유연성을 높입니다. 또한, conformal prediction 모듈을 사용하여 예측 결과의 신뢰 구간을 제공하여 예측의 신뢰성을 크게 높였습니다.'}]

- **Performance Highlights**: [{'Title': 'KDA_disc 및 KDA_cont의 예측 성능', 'Content': 'KDA_disc와 KDA_cont가 다양한 전문가가 라벨링한 MCQ 품질 척도에서 강력한 예측력을 보였습니다.'}, {'Title': '강화된 robustness', 'Content': '우리의 접근 방식은 여러 측면에서 significant한 향상을 이루었으며, 특히 counterfactual robustness, cross-domain generalization, 그리고 데이터가 제한된 환경에서의 일반화 성과에서 두드러졌습니다.'}, {'Title': 'V2INet의 경로 예측 성능', 'Content': 'V2INet은 실험을 통해 Final Displacement Error (FDE)와 Miss Rate (MR) 측면에서 뛰어난 성능을 보여주었습니다. 이 시스템은 단일 GPU에서 실행되었습니다.'}]



### Multimodal Fusion and Coherence Modeling for Video Topic Segmentation (https://arxiv.org/abs/2408.00365)
- **What's New**: 이 연구는 비디오 주제 분할(Video Topic Segmentation, VTS)을 다루며, 비디오를 명확한 주제로 나누는 과제를 향상시키는 새로운 기법을 제안합니다. 특히, 교육 비디오를 대상으로 다루어 학습 경험을 증대시키는 데 초점을 맞추고 있습니다. 또한, 새로운 대규모 중국어 강의 비디오 데이터셋을 소개해 연구 확대를 도모합니다.

- **Technical Details**: 제안된 방법은 크로스-어텐션(cross-attention)과 전문가 혼합(Mixture-of-Experts, MoE) 아키텍처를 활용한 멀티모달 융합을 탐구하고, 멀티모달 대조 학습(multimodal contrastive learning)을 통해 모델을 사전 학습 및 미세 조정(fine-tuning)합니다. 또한, VTS 작업에 맞춘 새로운 사전 학습 과제와 미세 조정 작업을 제안합니다. 이 접근 방식은 강의 비디오와 같은 교육 비디오에서 평가되었으며, 중국어 및 영어 데이터셋에서 최고의 성능을 보였습니다.

- **Performance Highlights**: 제안된 모델은 다양한 교육 비디오 데이터셋에서 기존의 비감독 및 감독 기반 모델을 능가하는 SOTA (state-of-the-art) 성능을 달성했습니다. 또한 세밀한 분해 연구를 통해 이 접근 방식의 효과를 추가로 검증했습니다.



### Towards Scalable GPU-Accelerated SNN Training via Temporal Fusion (https://arxiv.org/abs/2408.00280)
Comments:
          International Conference on Artificial Neural Networks (ICANN) 2024

- **What's New**: 새로운 자동 평가 메트릭 Knowledge Dependent Answerability (KDA)을 소개합니다. KDA는 기존 평가 지표들이 간과하는 교육적 가치를 평가하고, MCQ의 대답 가능성(Answerability)을 측정합니다.

- **Technical Details**: KDA는 학생의 문제 해결 행동을 모방하는 사전 훈련된 언어 모델을 활용합니다. 이를 통해 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이들 메트릭은 인간 실험에서 사용성과 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합될 때 다양한 전문가 레이블 MCQ 품질 측정에서 강력한 예측력을 보였습니다.



### RoCo:Robust Collaborative Perception By Iterative Object Matching and Pose Adjustmen (https://arxiv.org/abs/2408.00257)
Comments:
          ACM MM2024

- **What's New**: 이번 연구에서는 Multiple Choice Questions (MCQ) 생성을 위한 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)를 소개합니다. 이 메트릭은 기존의 BLEU, ROUGE, METEOR과 달리 MCQ의 교육적 가치를 평가하며, 학생의 지식 평가 능력을 측정합니다. 또한, 우리는 최근 NLP 태스크에서 deep model의 robustness를 강화하기 위한 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안했습니다. 마지막으로, 여러 대의 차량 간 협력적인 인식 성능을 강화하기 위한 RoCo라는 새로운 비감독 학습 프레임워크를 제시했습니다.

- **Technical Details**: [{'MCQ Generation': 'KDA_disc 와 KDA_cont라는 두 가지 새로운 자동 평가 메트릭을 제안했으며, 각각 학생의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 이용해 KDA를 근사합니다. 인간 설문 조사에서 얻은 학생 응답을 바탕으로 KDA를 측정하는 방법도 제시되었습니다.', 'NLP Robustness': 'counterfactual을 데이터셋에 추가하기 위해 contrastive learning 방식과 collective decision 방식을 활용하여, task model bias에 대한 민감도를 줄이는 접근법을 제안했습니다. 이를 통해 다양한 차원에서 significant improvements를 달성했습니다.', 'Collaborative Perception': 'RoCo는 객체 매칭과 agent 포즈 조정을 반복적으로 수행하는 새롭고 비감독 학습된 프레임워크입니다. distance와 local graph의 구조적 일관성을 바탕으로 객체 매칭 관계를 설정하고, global observation consistency를 통해 agent 포즈를 최적화하여 더 정확한 객체 매칭을 달성합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, n-gram 기반의 유사성 메트릭과 결합될 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강력한 예측 능력을 보였습니다.', 'NLP Robustness': 'collective decision을 통해 NLP 모델의 robustness를 높이는 데 성공하였으며, counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization에서 중요한 개선을 이루었습니다.', 'Collaborative Perception': 'RoCo는 기존 방법들보다 3D 객체 감지 성능에서 일관되게 우수한 성능을 보였으며, agent 포즈 정보에 대한 높은 수준의 노이즈에도 매우 높은 robust성을 나타냈습니다. V2XSet과 DAIR-V2X와 같은 실제 및 시뮬레이션 데이터셋에서 뛰어난 성능을 보여주었습니다.'}]



### Multiple Greedy Quasi-Newton Methods for Saddle Point Problems (https://arxiv.org/abs/2408.00241)
Comments:
          Submitted to DOCS 2024

- **What's New**: 자동 다지 선택 질문(MCQ)을 생성하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)을 도입했습니다. 이 메트릭은 데이터셋의 유사성 기준을 넘어 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: KDA의 측정 방식은 두 가지로 구분됩니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 인문 평가를 통해 이 메트릭이 실제 강의실 세팅의 사용성 및 골드 표본과의 유사성 기준과 강한 상관관계를 가지는 것으로 확인되었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ 품질 측정 값에 대해 강력한 예측력을 보였으며, 이는 n-그램 기반 유사성 메트릭과 결합했을 때 더욱 효과적입니다.



### Lost in Translation: Latent Concept Misalignment in Text-to-Image Diffusion Models (https://arxiv.org/abs/2408.00230)
Comments:
          33 pages, 19 figures

- **What's New**: MCQ 자동 생성의 한계를 보완하고 교육적 가치를 높이는 새로운 자동 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안했습니다. 이는 학생이 질문에 대답할 수 있는 능력을 측정하여 교육적 유효성을 평가합니다. 또한, 텍스트에서 이미지로 변환하는 확산 모델의 개념 불일치 문제(Latent Concept Misalignment, LC-Mis)를 조사하고 해결하기 위한 새로운 자동화된 파이프라인을 개발했습니다.

- **Technical Details**: MCQ의 교육적 가치를 평가하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. LC-Mis 문제를 해결하기 위해, LLM(대형 언어 모델)을 사용하여 개념의 일치성을 강화하는 파이프라인을 설계했습니다. 이 파이프라인은 개념 쌍을 분할하여 두 단계로 이미지 생성을 수행합니다.

- **Performance Highlights**: 연구 결과, KDA_disc와 KDA_cont가 실제 강의실 환경에서 높은 사용성과 강한 상관관계를 나타냈습니다. LC-Mis 문제 해결을 위한 새로운 접근법은 기존 모델보다 텍스트와 이미지의 일치를 크게 개선했으며, 다양한 분야에서의 적용성과 유연성을 증대시키는 데 기여했습니다.



### Finch: Prompt-guided Key-Value Cache Compression (https://arxiv.org/abs/2408.00167)
- **Multiple Choice Questions Generation**: [{"What's New": 'MCQ 생성에서 기존의 BLEU, ROUGE, METEOR 메트릭이 단순히 생성된 질문과 데이터셋의 골드 샘플과의 n-gram 유사성에 초점을 맞추는데, 이것이 교육적 가치를 반영하지 못한다는 문제를 지적하고 새로운 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안했습니다.'}, {'Technical Details': 'KDA는 학생의 대상 사실(target fact)에 대한 지식을 측정하는 능력에 기반하여 MCQ의 답변 가능성을 평가합니다. 이를 위해 처음에는 인간 설문조사를 통해 KDA를 측정하고, 이후 KDA_disc 및 KDA_cont라는 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측력을 가집니다.'}]

- **Robust NLP Models**: [{"What's New": '최근 딥러닝 모델들이 NLP 작업에서 초인간적 정확성을 보여주었지만, spurious pattern에 의존하는 문제로 인해 그 견고함이 제한된다고 보고되었습니다. 이에 따라 본 논문에서는 견고성을 위해 대조 학습(contrastive learning)과 반사실 강화(counterfactual augmentation)를 활용하는 방법을 제안합니다.'}, {'Technical Details': '기존의 반사실 강화 방법이 spurious correlation에 영향을 받는 문제를 해결하기 위해, 여러 개의 반사실을 생성하고 이들의 예측 분포에 대한 집합적 결정을 통해 각 용어의 인과 관계를 견고하게 감독하는 방식을 소개합니다.'}, {'Performance Highlights': '집합적 결정 방식을 통해 과제 모델의 편향에 덜 민감하게 되고, 반사실 견고성, 도메인 간 일반화 및 데이터 부족 상황에서도 상당한 성능 개선을 이룹니다.'}]

- **Finch Approach to Compression in Large Language Models**: [{"What's New": '최근의 대형 언어 모델(LLMs) 응용 프로그램, 예를 들어 검색-증강 생성(Retrieval-Augmented Generation)과 챗봇,에서는 더 많은 입력 컨텍스트를 처리할 필요성이 증가했습니다. 하지만 이를 위해서는 대규모 GPU 메모리가 필요합니다. Finch라는 새로운 접근 방식을 통해 입력 컨텍스트를 압축하고, 컨텍스트 윈도우 내에서 긴 문서를 압축된 버전으로 저장하여 대형 입력을 처리할 수 있게 합니다.'}, {'Technical Details': 'Finch는 프리트레인드 된 모델의 self-attention 가중치를 활용하여 긴 텍스트의 가장 관련성이 높은 Key (K) 및 Value (V) 쌍을 식별하여 KV 캐시에 저장합니다. 이 방법은 고도의 압축률(최대 93배)에도 의미적 완전성을 유지합니다.'}, {'Performance Highlights': 'SQuAD v2 벤치마크에서 Finch는 원래의 LLM 대비 2.35배 압축률에서 비교 가능한 생성 품질을 유지하고, 3.76배 압축률에서는 90% 정확도를 유지합니다. 또, LongLLM과 비교해 대부분의 태스크에서 최상의 품질 점수를 기록했습니다.'}]



### Formal Ethical Obligations in Reinforcement Learning Agents: Verification and Policy Updates (https://arxiv.org/abs/2408.00147)
- **What's New**: 본 논문은 다중 선택 질문(MCQ) 자동 생성을 위한 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안한다. KDA는 학생이 특정 사실에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성을 측정하여, 기존의 n-gram 기반의 유사성 평가 메트릭이 놓치고 있는 교육적 가치를 반영한다.

- **Technical Details**: KDA는 인간 설문조사를 통해 측정되며, 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방함으로써 자동 평가 메트릭(KDA_disc, KDA_cont)으로 근사화되었다. 이는 이전 메트릭과는 달리 학생의 지식 평가 능력을 정확하게 반영한다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강하게 상관관계가 있음을 입증하였다. 또한, n-gram 기반 유사성 메트릭과 결합할 때, 전문가가 레이블링한 MCQ 품질 기준에 대해 높은 예측력을 가진다.



### Inductive or Deductive? Rethinking the Fundamental Reasoning Abilities of LLMs (https://arxiv.org/abs/2408.00114)
- **What's New**: 최근의 deep model들이 자연어처리(NLP) 작업에서 초인적인 정확성을 보여주었으나, 스퓨리어스 패턴(spurious patterns)에 의존하여 강인함(robustness)이 제한된 상태임. 이를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 추가(counterfactual augmentation)을 활용하는 새로운 접근법을 제안함. 또한 MCQ(다중 선택 질문) 자동 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 소개해 학생의 지식 평가 능력을 향상시킴.

- **Technical Details**: 이 연구에서는 두 가지 접근법을 소개함. 첫째, MCQ 생성을 평가하기 위한 새로운 메트릭으로 KDA(Knowledge Dependent Answerability)를 제안하고, 이는 학생들의 지식에 기반한 답변 가능성을 측정함. 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 통해, 사전에 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방함. 둘째, 대조 학습과 반사실적 추가를 사용하여 LLM(Large Language Models)의 강인성을 높이는 방향을 연구함. 이를 위해 여러 반사실적 데이터를 생성하고 집합적 의사 결정(collective decisions)을 통해 인과 관계를 보다 강력하게 감독하는 방법을 사용함.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성과 강한 상관 관계를 보여줌. MCQ의 질적 지표를 예측하는데 높은 성능을 보였음. 새로운 대조 학습 방법 또한 대조 강인성, 크로스 도메인 일반화, 데이터 부족 상황에서의 일반화 측면에서 큰 개선을 달성함. SolverLearner 프레임워크를 통해 LLM이 놀라운 귀납적 사고 능력을 보여줌.



### Preference-Based Abstract Argumentation for Case-Based Reasoning (with-Appendix) (https://arxiv.org/abs/2408.00108)
Comments:
          Accepted for KR2024. Includes Appendix

- **What's New**: 이 논문에서는 교사들이 MCQ (Multiple Choice Question) 생성을 위해 소비하는 시간을 절감하는 자동화된 평가 메트릭을 제안합니다. 현재의 BLEU, ROUGE, METEOR와 같은 평가는 n-그램 기반 유사성에만 집중하여 MCQ의 교육적 가치를 무시하고 있습니다. 이에 반해, 지식 종속 가능성(KDA, Knowledge Dependent Answerability)이라는 새로운 메트릭을 도입하여 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: KDA 디스크리트(KDA_disc)와 KDA 연속(KDA_cont)이라는 두 가지 자동 평가 메트릭을 제안하며, 이들은 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. Human evaluation을 통해 우리는 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 추가적으로, 이 메트릭들은 n-그램 기반 유사성 메트릭과 결합될 때 전문가 레이블된 MCQ 품질 측정치에 대한 강력한 예측력을 지닙니다.

- **Performance Highlights**: 엄밀한 Human evaluation을 기반으로 KDA_disc와 KDA_cont는 (1) 실제 강의실 환경에서의 사용성, (2) 전문가들에 의해 평가된 MCQ 품질과 강한 상관관계를 증명했습니다.



### Areas of Improvement for Autonomous Vehicles: A Machine Learning Analysis of Disengagement Reports (https://arxiv.org/abs/2408.00051)
- **What's New**: 자동 MCQ 생성의 평가 메트릭으로 교육적 가치를 평가하는 '지식 종속 가능성(KDA)' 메트릭을 제안했습니다. 또한, 최근 NLP 태스크에서 deep model의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 활용하는 새로운 방법을 제안했습니다. 마지막으로, 자율 주행 차량의 운행 중단 보고서를 분석하여 개선점을 찾는 머신러닝 기반 접근법을 소개했습니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 메트릭과 달리, KDA 메트릭은 MCQ의 대답 가능성(answerability)과 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 이를 위해, 우리는 학생 응답 기반의 KDA 측정 방법을 제안하고, 이를 근사화하기 위해 pre-trained language model을 사용하는 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안했습니다.'}, {'Model Robustness': '기존의 counterfactual augmentation은 spurious correlation 문제에 취약했지만, 우리는 여러 개의 counterfactual을 생성하고 이들의 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안했습니다. 이는 counterfactual robustness, cross-domain generalization, 그리고 scarce data에 대한 generalization 측면에서 성과를 보였습니다.'}, {'Autonomous Vehicle Analysis': '2023년 캘리포니아 자율 주행 차량 보고서를 기반으로, 자연어 처리(NLP)를 사용하여 운행 중단의 정보 추출과 k-Means clustering 알고리즘을 통해 데이터를 군집화하였습니다. 각 군집의 빈도를 분석하고 수동으로 카테고리화하여 자율 주행 차량의 개선점을 도출했습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human evaluation 결과, KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 독립적인 전문 평가자가 라벨링한 다양한 MCQ 품질 측정치를 예측하는 데에 강력한 예측력을 보였습니다.'}, {'Model Robustness': '제안된 방법은 counterfactual robustness, cross-domain generalization, scarce data에 대한 generalization의 세 가지 다양한 차원에서 기존 방법보다 유의미한 성과를 보였습니다.'}]



### Con4m: Context-aware Consistency Learning Framework for Segmented Time Series Classification (https://arxiv.org/abs/2408.00041)
- **new_approach_to_MCQs_generation**: [{"What's New": '자동 생성된 MCQ(Multiple Choice Questions)의 교육적 가치를 평가하기 위한 새로운 메트릭, Knowledge Dependent Answerability (KDA),를 개발했습니다. 이 메트릭은 기존 BLEU, ROUGE, METEOR처럼 단순한 n-그램 유사성에 의존하지 않고, MCQ가 학생의 지식을 평가하는 능력을 확인합니다.'}, {'Technical Details': 'KDA는 실제 학생 응답을 바탕으로 측정하며, 이를 자동화하기 위해 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont 메트릭을 제안합니다. 이를 통해 모델이 학생의 문제 해결 행동을 모방하도록 유도합니다.'}, {'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실에서의 사용성과 강한 상관관계가 있음을 입증했습니다. 이 메트릭들은 기존의 n-그램 기반 유사성 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 측정치와 높은 예측력을 보였습니다.'}]

- **robust_NLP_using_cl_and_CA**: [{"What's New": '최근의 딥러닝 모델들이 NLP 작업에서 매우 높은 정확도를 보이나, 비정상적 패턴(spurious patterns)에 의존하여 내구성이 부족한 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 결합한 새로운 접근 방식을 제안합니다.'}, {'Technical Details': '기존의 증강 방법은 인간이 반사실적 데이터를 추가하거나 모델이 데이터셋에서 유사한 반사실적 데이터를 찾는 방식을 사용했으나, 여전히 비정상적 상관관계에 의존합니다. 제안된 방법은 여러 개의 반사실적 데이터를 생성하고, 이를 바탕으로 집합적 의사결정을 내려 각 용어들의 인과관계를 더욱 내구성 있게 감독합니다.'}, {'Performance Highlights': '제안된 방법은 다양한 차원에서, 특히 반사실적 내구성(counterfactual robustness), 교차 도메인 일반화, 및 희소 데이터로부터의 일반화 측면에서 상당한 성능 향상을 보였습니다.'}]

- **TSC_with_MVD_and_Con4m**: [{"What's New": '다중 클래스와 다른 기간(MVD)을 가지는 분할된 시계열 분류(TSC) 문제를 해결하기 위한 새로운 일관성 학습 프레임워크 Con4m를 제안했습니다. 이는 기존 작업들이 간과했던 연속적인 인스턴스 간의 자연스러운 시간적 종속성을 고려합니다.'}, {'Technical Details': 'Con4m는 데이터 및 라벨 수준의 문맥 정보를 활용하여 분할된 TSC 작업을 보다 효과적으로 모델링합니다. 또한 일관성 없는 경계 라벨을 조화시키기 위해 점진적 라벨 조정 접근 방식을 사용하여 내구성 있는 모델을 만들었습니다.'}, {'Performance Highlights': '다양한 공개 및 비공개 MVD 데이터셋에서 Con4m의 우수한 성능이 입증되었습니다. 라벨 대체 실험과 사례 연구를 통해 일관성 없는 라벨을 조화시킬 수 있는 Con4m의 능력도 확인되었습니다.'}]



### Enhanced Fault Detection and Cause Identification Using Integrated Attention Mechanism (https://arxiv.org/abs/2408.00033)
- **What's New**: 자동 다지선다형 질문(MCQ) 생성의 평가를 위해 지식 종속 가능성(KDA)이라는 새로운 메트릭이 제안되었습니다. 이 메트릭은 MCQ의 대답 가능성(answerability)을 측정하여 교육적 유용성을 평가합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 메트릭이 n-gram 유사성에만 집중하여 MCQ의 교육적 가치를 고려하지 않는 문제를 해결하기 위해, KDA는 학생의 응답 데이터를 사용하여 측정되며, pre-trained language models를 활용하여 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 사용성과 강한 상관관계를 가지며, 통합된 평가에서는 다양한 전문가가 레이블한 MCQ 품질 척도에 대해 높은 예측력을 보였습니다.



### A New Type of Foundation Model Based on Recordings of People's Emotions and Physiology (https://arxiv.org/abs/2408.00030)
Comments:
          12 pages, 2 figures, 3 tables

- **What's New**: {'paper_1': '이 논문에서는 자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 평가 메트릭으로 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. 이는 학생이 특정 사실에 대한 지식을 바탕으로 MCQ에 답변할 수 있는 능력을 평가합니다.', 'paper_2': '이 논문은 대조 학습(Contrastive Learning)과 반사실적 증강(Counterfactual Augmentation)을 활용하여 NLP 모델의 robust성을 향상시키는 방법을 제안합니다. 대조 학습 및 반사실적 증강을 통해 각각의 용어의 인과관계를 보다 견고하게 감독합니다.', 'paper_3': '이 논문은 새로운 종류의 기반 모델인 1인칭 기반 모델(First-Person Foundation Model)을 제안합니다. 이는 개인이 인지하는 환경 자극과 그에 대한 정서적, 생리적 반응을 기록하여, 이를 바탕으로 개인의 행동을 예측합니다.'}

- **Technical Details**: {'paper_1': 'KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'paper_2': '여러 개의 반사실적 데이터셋을 만들고, 이 데이터셋들의 예측 분포를 통해 집합적 의사 결정을 내립니다. 기존 방법보다 spurious correlation에 덜 민감하게 반응합니다.', 'paper_3': '특수 녹화 장치를 사용하여 사람들이 보는 것과 듣는 것 및 정서적, 생리적 상태를 기록합니다. 이 데이터를 바탕으로 1인칭 기반 모델이 환경 자극과 개인의 행동 간의 관계를 학습합니다.'}

- **Performance Highlights**: {'paper_1': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보여주었고, 기존 n-gram 기반 메트릭과 결합하여 전문가가 평가한 MCQ 품질 측정에서 높은 예측력을 보였습니다.', 'paper_2': '제안된 방법은 반사실적 강건성, 크로스 도메인 일반화, 데이터가 적은 상태에서의 일반화에서 유의미한 향상을 달성하였습니다.', 'paper_3': '1인칭 기반 모델은 새로운 추천 엔진, 개인 비서, 생성적 적대 신경망(GAN), 데이팅 및 채용 등의 다양한 응용 분야에서 흥미로운 가능성을 제공합니다.'}



### Need of AI in Modern Education: in the Eyes of Explainable AI (xAI) (https://arxiv.org/abs/2408.00025)
Comments:
          Accepted for the book: Blockchain and AI in Shaping the Modern Education System, Publisher: CRC Press, Taylor & Francis Group, USA

- **What's New**: 자동 다지선다형 질문(MCQ) 생성의 교육적 가치를 평가하기 위해 새로운 평가 메트릭(Knowledge Dependent Answerability, KDA)을 도입했습니다. 또한, 딥 러닝 모델의 로버스트니스(robustness)를 강화하기 위해 대조 학습과 반사실적 증강(counterfactual augmentation) 기법을 적용하는 방법을 제안합니다.

- **Technical Details**: [{'MCQ Generation': '기존의 평가 메트릭(BLEU, ROUGE, METEOR)은 n-gram 유사성만을 평가하여 교육적 가치를 무시하고 있습니다. 새로운 평가 메트릭인 KDA는 목표 사실(target fact)에 대한 학생의 지식을 평가하는 MCQ의 대답 가능성(answerability)을 측정합니다. KDA는 학생의 응답을 기반으로 하며, 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)으로 확장됩니다.'}, {'Deep Model Robustness': '딥 모델의 로버스트니스를 강화하기 위해, 기존의 반사실적 데이터 증강 기법 대신 여러 개의 반사실적 상황(counterfactuals)을 생성하고 이 세트에 대한 예측 분포를 통해 인과 관계를 더욱 신뢰성 있게 감독하는 방법을 제안합니다. 이를 통해 과업 모델의 편향에 덜 민감한 성과를 보여줍니다.'}]

- **Performance Highlights**: [{'MCQ Evaluation Correlation': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실에서의 사용성과 강한 상관관계를 가지고 있음을 증명했습니다. 또한, n-gram 유사성 메트릭과 결합하여 전문가가 라벨링한 MCQ 품질 측정에 대한 예측력이 높다는 것을 확인했습니다.'}, {'Deep Model Improvements': '집합적 의사 결정 방식(collective decisions)로 인과 관계를 평가하여 모델의 로버스트니스가 크게 향상되었음을 실증적인 결과로 보여주었습니다. 이 방법은 반사실적 로버스트니스, 도메인 간 일반화(cross-domain generalization), 그리고 희소 데이터에서의 일반화에 있어서 현저한 개선을 달성했습니다.'}]



### Deceptive AI systems that give explanations are more convincing than honest AI systems and can amplify belief in misinformation (https://arxiv.org/abs/2408.00024)
- **What's New**: 이 연구에서는 자동 MCQ 생성에 대한 평가 방법으로, 교육적 가치를 고려하지 않는 기존의 BLEU, ROUGE, METEOR 메트릭의 한계를 지적하고 새로운 자동 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안했습니다. 이 메트릭은 학생이 대상 사실을 알고 있을 때 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 인간 설문 조사에서 학생 응답을 기반으로 측정되며, 이를 자동 평가 메트릭 KDA_disc와 KDA_cont로 확장하여, 사전 학습된 언어 모델(pre-trained language models)을 사용해 학생의 문제 해결 행동을 모방합니다. 이 두 자동 평가 메트릭은 실제 강의실 환경에서 사용성과 강한 상관관계를 가지고 있는 것으로 나타났습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대해 강력한 예측력을 가지고 있음이 확인되었습니다.



### MM-Vet v2: A Challenging Benchmark to Evaluate Large Multimodal Models for Integrated Capabilities (https://arxiv.org/abs/2408.00765)
Comments:
          Extension of MM-Vet: arXiv:2308.02490

- **What's New**: 이 논문에서는 기존의 MM-Vet의 단점을 보완하여 새로운 평가 메트릭인 MM-Vet v2를 소개합니다. MM-Vet v2는 '이미지-텍스트 시퀀스 이해'라는 새로운 비전-언어(Vision-Language) 능력을 평가하여 모델이 이미지-텍스트 시퀀스를 처리하는 능력을 포함합니다.

- **Technical Details**: MM-Vet v2는 기존 MM-Vet의 여섯 가지 핵심 VL 능력 (Recognition, Knowledge, Spatial Awareness, Language Generation, OCR, Math) 외에 추가로 '이미지-텍스트 시퀀스 이해'를 포함합니다. 이를 위해 연구팀은 다양한 시나리오에서 수집한 517개의 질문을 설계하고, 긴 답변을 필요로 하는 질문은 GPT-4V를 사용해 초안을 작성한 뒤 전문가가 검토합니다.

- **Performance Highlights**: MM-Vet v2로 다양한 대형 멀티모달 모델(LMMs)을 평가한 결과, Claude 3.5 Sonnet이 71.8점을 기록하며 가장 높은 성능을 보였습니다. GPT-4o는 71.0점을 기록했고, 오픈 소스 모델 중에서는 InternVL2-Llama3-76B가 68.4점으로 가장 뛰어난 성능을 나타냈습니다.



### AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation (https://arxiv.org/abs/2408.00764)
- **What's New**: 이번 뉴스레터는 자동 Multiple Choice Questions (MCQ) 평가와 대규모 언어 모델(LLM) 기반 에이전트의 계획 능력 향상에 대한 최신 연구를 다룹니다. 또한, deep model의 robustness 문제를 해결하는 새로운 접근법에 대해서도 소개합니다.

- **Technical Details**: {'MCQ Generation': 'MCQ 생성의 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 고려하지 않습니다. 새로운 평가 방법인 Knowledge Dependent Answerability(KDA)는 학생의 지식을 평가하는 MCQ의 답변 가능성(answerability)을 측정합니다. 이 메트릭은 학생 반응 설문조사를 기반으로 하며, KDA_disc와 KDA_cont를 제안합니다.', 'Deep Model Robustness': 'Deep model의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 이용합니다. 여러 개의 counterfactual을 생성하고 collective decision을 통해 단어들의 인과관계를 더 견고하게 파악하는 접근법을 제안합니다.', 'LLM-based Agent Training': 'LLM 기반 에이전트의 계획 능력을 향상시키기 위해 instruction tuning을 이용합니다. AgentGen이라는 프레임워크를 통해 다양한 환경과 난이도가 점진적으로 상승하는 계획 태스크를 자동으로 생성합니다. 이를 통해 LLM의 계획 능력을 향상시킵니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 사용성과 강한 상관관계를 보이며, n-gram 기반 메트릭과 결합하여 전문가가 평가한 MCQ 품질을 잘 예측합니다.', 'Deep Model Robustness': '여러 실험 결과, 제안된 방법이 counterfactual robustness, 교차 도메인 일반화(cross-domain generalization), 및 희소 데이터에서의 일반화(generalization from scarce data) 등 다양한 측면에서 개선되었습니다.', 'LLM-based Agent Training': 'AgentGen 프레임워크로 instruction-tuning된 Llama-3 8B 모델이 GPT-3.5와 GPT-4를 능가하는 성능을 보였습니다.'}



### Tamper-Resistant Safeguards for Open-Weight LLMs (https://arxiv.org/abs/2408.00761)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 자동으로 생성된 객관식 질문(MCQ)의 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안합니다. 또한, 대규모 언어 모델(LLM)의 악성 사용을 방지하기 위한 새로운 보호 기법, Tamper-Resistant Safeguards (TAR)을 도입했습니다.

- **Technical Details**: KDA 메트릭은 학생들의 실제 응답 데이터를 활용해 자동으로 MCQ의 대답 가능성을 평가합니다. 이를 통해 BLEU, ROUGE, METEOR와 같은 기존의 n-그램 기반 메트릭들이 간과하는 교육적 가치를 파악할 수 있습니다. TAR 기법은 모델 가중치를 조작하는 공격에도 견딜 수 있는 안전 장치를 언어 모델에 포함시키는 방법입니다. 이 방법은 adversarial training을 활용해 모델 실패를 방지합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 메트릭과 결합하여 전문가들이 라벨링한 다양한 MCQ 품질 측정에 대해 높은 예측력을 보여줍니다. 또한, TAR 기법은 5,000 스텝 이상의 파인 튜닝 공격에도 견딜 수 있는 강력한 저항성을 보였습니다. 이를 통해 LLM의 안전성과 보안성을 크게 개선할 수 있습니다.



### Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention (https://arxiv.org/abs/2408.00760)
- **Multiple Choice Questions (MCQ) Generation**: {"What's New": '새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 도입하여, MCQ의 교육적 가치를 평가하는 능력을 향상시켰습니다.', 'Technical Details': 'KDA는 학생의 대상 사실(target fact)에 대한 지식을 기준으로 MCQ의 답변 가능성을 측정합니다. 주어진 대상 사실에 기반하여 인간 설문조사를 통해 KDA를 측정하고, 이를 근사하기 위해 미리 훈련된 언어 모델을 활용한 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안합니다.', 'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 강의실 세팅에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 레이블 MCQ 품질 측정에 대한 예측력이 높아짐을 확인했습니다.'}

- **Robust NLP Models**: {"What's New": '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 NLP 모델의 robustness를 향상시키는 새로운 접근법이 제안되었습니다.', 'Technical Details': '기존 증강 방식이 spurious correlation에 영향을 받는 반면, 이 접근법은 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 각 단어의 인과관계를 robust하게 감독합니다.', 'Performance Highlights': '테스트 결과, 이 접근법은 counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 측면에서 성능 개선을 보여주었습니다.'}

- **Smoothed Energy Guidance (SEG) for Image Generation**: {"What's New": 'Self-attention 메커니즘의 에너지 기반 관점을 활용하여 이미지 생성 품질을 향상시키는 Smoothed Energy Guidance (SEG)라는 새로운 방법론이 제안되었습니다.', 'Technical Details': 'SEG는 Gaussian 커널 매개변수를 조정하여 에너지 풍경의 곡률을 제어하고, query blurring을 통해 복잡도를 낮추며, 본래의 예측에서 구조적 변경을 최소화합니다.', 'Performance Highlights': '텍스트 조건 유무에 관계없이 다양한 실험에서 SEG가 이전 방법들보다 더 나은 샘플 품질을 제공하고 부작용을 줄이는 Pareto 개선을 달성했습니다.'}



### Segment anything model 2: an application to 2D and 3D medical images (https://arxiv.org/abs/2408.00756)
Comments:
          11 pages, 7 figures. A first attempt on evaluating SAM 2 on medical images

- **What's New**: 기존의 여러 평가 메트릭들이 다루기 부족했던, 교육적 가치뿐만 아니라 답변 가능성(knowledge dependent answerability, KDA)를 측정하는 새로운 자동 평가 메트릭을 제안합니다. 이를 통해 학생의 대상 사실(target fact)에 대한 지식을 직접적으로 평가할 수 있게 되었습니다. 또한, KDA를 근사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 소개하였습니다.

- **Technical Details**: KDA 측정을 위해 학생 응답을 대상으로 한 human survey를 기반으로 한 KDA 산출 방법을 먼저 보여준 다음, pre-trained language models을 활용해, student의 문제 해결 행동을 모사하여 KDA를 근사하는 KDA_disc와 KDA_cont를 제안하였습니다. 또한, 두 자동 평가 메트릭은 전문가가 labeled한 MCQ 품질 메트릭을 예측하는 데 강한 예측력을 가졌다는 것을 나타냈습니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계가 있음을 확인하였습니다. 이들 메트릭이 기존 n-gram 기반 유사성 평가 메트릭과 결합되는 경우 MCQ 품질 측정에서 더욱 강력한 예측력을 보였습니다.



### A deep learning-enabled smart garment for versatile sleep behaviour monitoring (https://arxiv.org/abs/2408.00753)
Comments:
          18 pages, 5 figures, 1 table

- **What's New**: 최근 강화된 자동 MCQ (Multiple Choice Questions) 생성 모델과 지능형 수면 모니터링 시스템에 대한 연구를 소개합니다. MCQ 생성 관련 연구에서는 지식 종속 가능성(KDA)을 측정하는 새로운 평가 메트릭을 제안하였으며, 수면 모니터링 연구에서는 스마트 의류에 인쇄된 초민감 스트레인 센서 배열로 다양한 수면 상태를 감지하는 시스템을 개발했습니다.

- **Technical Details**: MCQ 생성 연구에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이러한 메트릭은 n-gram 기반 유사성 메트릭과 결합할 때 예측력의 향상을 보여줍니다. 수면 모니터링 연구에서는 스마트 의류의 카라 부분에 초민감 스트레인 센서 배열을 배치하여, 나잘 호흡, 입 호흡, 코골이, 이갈이, 중심 수면 무호흡 및 폐쇄성 수면 무호흡과 같은 여섯 가지 수면 상태를 98.6%의 높은 정확도로 식별합니다. 심층 학습 신경망이 포함되어 있으며, 설명 가능한 인공지능(XAI) 시각화가 포함되어 있어 신호 패턴 분석에서 낮은 편향을 보여줍니다.

- **Performance Highlights**: MCQ 생성 메트릭(KDA_disc, KDA_cont)은 실제 강의실 환경에서 사용성과 강한 상관관계를 가지며, 다양한 전문가 라벨링 MCQ 품질 측정치에 대한 정확한 예측 능력을 가지는 것으로 나타났습니다. 스마트 의류 기반 수면 모니터링 시스템은 거의 모든 새 사용자에 대해 95% 이상의 정확도를 유지하며, 소수의 샘플 (15개 이하)로도 높은 학습 성능을 보여줍니다.



### A Policy-Gradient Approach to Solving Imperfect-Information Games with Iterate Convergenc (https://arxiv.org/abs/2408.00751)
- **What's New**: 이번 뉴스레터에서는 최근 발표된 세 편의 연구 논문을 소개합니다. 이 논문들은 MCQ 자동 생성 평가, NLP 모델의 robust성 향상 및 다중 에이전트 강화 학습에 관한 내용을 다룹니다.

- **Technical Details**: [{'Topic': 'MCQ 자동 생성 평가', 'Summary': '기존의 MCQ 생성 평가 메트릭은 교육적 가치를 충분히 평가하지 못합니다. 이를 해결하기 위해 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 제안하며, KDA_disc 및 KDA_cont라는 두 가지 자동화 평가 메트릭을 소개합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, {'Topic': 'NLP 모델의 Robust성 향상', 'Summary': '최근 딥 모델들은 NLP 태스크에서 매우 높은 정확성을 보이나, spurious pattern에 취약하여 robust성 문제를 겪고 있습니다. 이를 극복하기 위해 contrastive learning과 counterfactual augmentation을 결합한 새로운 방법을 제시합니다. 본 연구는 여러 counterfactual을 생성하여 집합적 의사 결정을 통해 더 robust하게 인과 관계를 분석합니다.'}, {'Topic': '다중 에이전트 강화 학습', 'Summary': '단일 에이전트 강화 학습에서는 정책 그라디언트 (policy gradient) 방법이 효과적이나, 다중 에이전트 게임의 불완전 정보 환경에서는 이의 적용이 보장되지 않았습니다. 본 연구는 정책 그라디언트 방법이 2인 제로섬 불완전 정보 게임에서 정규화된 내쉬 균형에 수렴할 수 있음을 최초로 증명합니다.'}]

- **Performance Highlights**: [{'Topic': 'MCQ 자동 생성 평가', 'Summary': 'KDA_disc와 KDA_cont는 인간 평가를 통해 실제 강의실 환경에서의 사용성과 강한 상관관계를 가짐을 입증하였습니다. 특히, n-gram 기반 유사성 메트릭과 결합할 경우 전문가 검증 MCQ 품질 측정에 높은 예측력을 보였습니다.'}, {'Topic': 'NLP 모델의 Robust성 향상', 'Summary': '제안된 방법론은 다양한 차원에서 주요 향상을 보여주었습니다: 1) counterfactual robust성, 2) 도메인 간 일반화, 3) 데이터 부족 상황에서의 일반화.'}, {'Topic': '다중 에이전트 강화 학습', 'Summary': '정책 그라디언트 방법이 두 플레이어 제로섬 게임에서 정규화된 내쉬 균형에 안정적으로 수렴하는 것을 입증함으로써, 정책 그라디언트의 적용 범위를 확장했습니다.'}]



### Leaf Angle Estimation using Mask R-CNN and LETR Vision Transformer (https://arxiv.org/abs/2408.00749)
- **What's New**: 이 논문은 다중 선택 질문 (Multiple Choice Questions, MCQ)의 자동 생성을 위한 평가 메트릭이 기존의 BLEU, ROUGE, METEOR와 같은 n-그램 기반의 유사성만을 비교하여 교육적 가치를 무시하는 문제를 해결하기 위해 제안되었다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하였다.

- **Technical Details**: KDA는 생성된 MCQ의 대답 가능성 (answerability)를 측정하고, 대상 사실에 대한 학생의 지식 평가 능력을 평가한다. KDA를 측정하기 위해 인간 평가단의 학생 응답을 사용한 후, 사전 학습된 언어 모델들을 활용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하였다.

- **Performance Highlights**: 인간 연구를 통해, KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었으며, n-그램 기반의 유사성 메트릭들과 결합했을 때 전문가가 레이블링한 MCQ 품질 측정에 대한 예측력이 강함을 보여주었다.



### Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions (https://arxiv.org/abs/2408.00727)
- **papers**: [{"What's New": '자동 MCQ생성을 위한 새로운 평가 메트릭 KDA가 제안되었습니다. KDA는 MCQ의 교육적 가치를 평가하는 데 초점을 맞추며 학습자의 목표 사실에 대한 지식 평가 능력을 측정합니다.', 'Technical Details': '기존 메트릭 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에 초점을 맞추지만, KDA는 학생의 응답을 기반으로 한 KDA 측정 방법을 도입합니다. 이 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA를 근사합니다.', 'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_soft가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 확인했습니다. KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합하여 다양한 전문가 레이블 MCQ 품질 측정에 대한 강력한 예측 성능을 보였습니다.'}, {"What's New": '대조 학습(contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용하여 NLP 모델의 robustness를 개선한 새로운 접근 방식이 제안되었습니다.', 'Technical Details': '기존의 반사실적 증강 방법은 인간이나 기계가 데이터셋 내에서 반사실적 요소를 추가하거나 자동으로 찾아야 하는데, 이는 여전히 spurious correlation에 영향을 받습니다. 본 논문에서는 여러 개의 반사실적 사례를 생성하고 집합적 의사 결정을 통해 단어의 인과성을 robust하게 감독하는 방식을 제안합니다.', 'Performance Highlights': '다양한 차원에서의 성능 개선을 통해 해당 접근 방식이 반사실적 robustness, 도메인 간 일반화, 그리고 희소한 데이터로부터의 일반화를 달성함을 입증했습니다.'}, {"What's New": '의학 질문에 대한 성능을 향상시키기 위해 i-MedRAG라는 반복적 RAG 프레임워크가 제안되었습니다. 이 모델은 이전 정보 검색 시도에 기반한 추가 질문을 반복적으로 할 수 있도록 설계되었습니다.', 'Technical Details': 'i-MedRAG는 기본 RAG 시스템에서 한 번만 수행되는 정보 검색 단계를 여러 단계로 확장하여 복잡한 임상 논리를 해결합니다. 이는 LLM이 반복적으로 후속 질문을 생성하고 이를 바탕으로 원래 질문에 대한 더 나은 답변을 생성할 수 있도록 합니다.', 'Performance Highlights': 'i-MedRAG는 다양한 LLM에서 미국 의사 면허 시험(USMLE) 하위 집합 및 Massive Multitask Language Understanding (MMLU) 데이터셋의 복잡한 질문에 대해 향상된 성능을 보였습니다. 특히, i-MedRAG는 GPT-3.5에 대한 zero-shot 설정에서 모든 이전 프롬프트 엔지니어링 및 미세 조정 방법을 능가하여 MedQA 데이터셋에서 69.68%의 정확도를 달성했습니다.'}]



### Pathway to Secure and Trustworthy 6G for LLMs: Attacks, Defense, and Opportunities (https://arxiv.org/abs/2408.00722)
Comments:
          7 pages, 4 figures

- **What's New**: 최근 MCQ(Multiple Choice Questions) 자동 생성은 교육자들의 학생 평가 시간을 줄일 수 있는 가능성을 보이고 있습니다. 이번 연구에서는 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 교육적 가치를 고려하지 않는다는 문제를 해결하고자, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 학생의 지식 수준에 기반한 MCQ의 대답 가능성을 측정합니다. 또한, 6G 네트워크에서의 LLM(Large Language Models) 활용과 보안 취약성을 탐구하며, 특히 membership inference attack에 초점을 맞추고 있습니다.

- **Technical Details**: [{'MCQ Generation': 'MCQ 생성에 있어서 KDA는 학생 설문조사 데이터를 바탕으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안했으며, 이 메트릭은 사전 학습된 언어 모델(pre-trained language models)을 활용해 학생들의 문제 해결 행동을 모방합니다.'}, {'Counterfactual Generation': 'robustness를 강화하기 위해 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 활용합니다. 다양한 반사실들을 생성하여 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 감독할 수 있습니다.'}, {'LLM in 6G Communication': '6G 네트워크에서의 LLM 활용 연구는 membership inference attack에 대한 보기 드문 연구입니다. 이를 통해 강화된 LLM 보안 메커니즘을 제안하며, Fine-tuning 과정에서 발생할 수 있는 개인 데이터 유출을 방지하기 위한 방법을 탐구합니다.'}]

- **Performance Highlights**: [{'MCQ Evaluation': 'KDA_disc와 KDA_cont는 인간 연구를 통해 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합시, 다양한 평가 지표에서 높은 예측력을 보였습니다.'}, {'Robustness in NLP Tasks': '제안된 반사실 증강 방법이 다양한 차원에서 1) 반사실 유연성, 2) 도메인 간 일반화, 3) 희박한 데이터에서의 일반화 등에서 성능 향상을 이루었습니다.'}, {'6G Network Security': 'membership inference attack이 이름 엔터티 인식(named entity recognition) 태스크에서 최대 92%의 성공률을 보였으며, 실험 분석을 통해 제안된 방어 메커니즘들이 특정 맥락에서 효과적임을 확인했습니다.'}]



### SAM 2: Segment Anything in Images and Videos (https://arxiv.org/abs/2408.00714)
Comments:
          Website: this https URL

- **What's New**: 강력한 성능과 사용성을 갖춘 새로운 자동 평가 메트릭 지식 종속 가능성(KDA)을 제안합니다. 최근 NLP 태스크에서의 deep models의 성능이 높지만 spurious patterns에 의존하는 문제를 해결하기 위해 대조학습과 반사실적 증강법을 사용했습니다. 새롭게 공개된 Segment Anything Model 2 (SAM 2)는 이미지와 비디오에서 프롬프트 기반 분할을 수행하며, 특히 비디오 분할 성능이 크게 향상되었습니다.

- **Technical Details**: MCQ 생성 모델의 교육적 가치를 평가하기 위해 KDA를 도입했으며, 이는 인간 평가를 통해 확인되었습니다. NLP의 robustness 향상을 위해 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어의 인과관계를 파악했습니다. SAM 2는 Streaming Memory를 갖춘 Transformer 구조를 사용해 실시간 비디오 처리 능력을 강화하였으며, 데이터 엔진을 통해 세계 최대의 비디오 분할 데이터셋을 구축했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont가 실제 강의실 사용성에서 높은 상관관계를 나타냈습니다. 우리의 접근 방식은 task model의 바이어스에 덜 민감하며, 반사실적 증강에서 tf robustness, cross-domain generalization, 데이터 부족 상황에서의 generalization 등 여러 측면에서 성능 개선을 보였습니다. SAM 2는 이전 방법들에 비해 3배 적은 상호작용으로 더 높은 정확도를 달성했으며, 이미지 분할에서도 6배 빠른 성능을 보였습니다.



### Investigating Brain Connectivity and Regional Statistics from EEG for early stage Parkinson's Classification (https://arxiv.org/abs/2408.00711)
- **What's New**: 이 연구는 자동으로 여러 선택 문제(MCQ)를 생성하는 새로운 평가 메트릭, '지식 종속 가능성 (Knowledge Dependent Answerability, KDA)'을 제안합니다. 또한, 다양한 뇌 연계 측정치(brain connectivity metrics)와 신호 통계를 결합하여 초기에 파킨슨병을 분류하는 방법을 제안합니다.

- **Technical Details**: 기존 BLEU, ROUGE, METEOR와 같은 MCQ 생성 평가는 교육적 가치를 평가하지 못합니다. 이를 해결하기 위해 제안된 KDA는 먼저 사람 설문 조사를 통해 계산되며, 사전 학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방하여 자동으로 KDA를 측정합니다. 파킨슨병 분류 연구에서는 EEG 데이터를 사용하여 9개의 뇌 연계 측정치와 지역 신호 통계를 결합해 초기 단계 파킨슨병을 분류합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 강하게 상관관계가 있음을 밝혔습니다. 파킨슨병 분류 모델에서는 Phase Lag Index(PLI)를 사용한 N1 데이터에서 91%의 정확도를 달성했으며, 다양한 각성 상태의 데이터에서도 연계 측정치와 신호 통계 결합이 분류 성능을 향상시킵니다.



### Point-supervised Brain Tumor Segmentation with Box-prompted MedSAM (https://arxiv.org/abs/2408.00706)
Comments:
          2024 IEEE Nuclear Science Symposium and Medical Imaging Conference

- **What's New**: 이번 연구는 MCQ (Multiple Choice Questions) 생성의 자동 평가 메트릭의 한계를 극복하기 위해 제안된 새로운 Knowledge Dependent Answerability (KDA) 지표를 소개합니다. KDA는 학생이 해당 사실(target fact)에 대한 지식을 기반으로 MCQ의 대답 가능성을 측정하며, 이는 기존의 n-gram 유사성 메트릭이 지니고 있던 교육적 가치 측면의 한계를 극복합니다.

- **Technical Details**: KDA 측정 방식은 먼저 학생들의 응답을 통해 휴먼 설문조사를 기반으로 한 평가 지표를 측정합니다. 이후, 미리 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행위를 모방함으로써 KDA를 근사화하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이는 약어화된 'discrete'와 'continuous'에서 각각 유래됐습니다. 인간 연구를 통해 이 두 메트릭은 실제 교실 환경에서의 사용성과 강한 상관관계를 지니고 있음을 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가들이 라벨링한 다양한 MCQ 품질 기준을 예측하는 데 있어 높은 예측력을 보였습니다. 특히, 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 더욱 강력한 예측력을 보여주었습니다.



### Future of Artificial Intelligence in Agile Software Developmen (https://arxiv.org/abs/2408.00703)
- **What's New**: 이 논문은 교사의 평가 시간을 줄이고 학생의 지식을 평가하는 능력을 갖추기 위해 자동 MCQ 생성에 필요한 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 또한, 최근 NLP 태스크에서 높은 정확성을 보이는 deep model들의 robustness를 강화하기 위한 대조 학습과 반사례 증강 방법을 제안합니다. 마지막으로 AI 기술을 소프트웨어 프로젝트 개발에 적용하여 효율성을 높이고 위험을 줄이는 방법에 대해 논의합니다.

- **Technical Details**: 첫 번째 논문에서는 KDA를 이용하여 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가합니다. KDA는 학생 응답 기반의 인간 조사로 측정되며, 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안하여 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 두 번째 논문은 대조 학습과 counterfactual augmentation을 사용하여 모델의 robustness를 강화하며, 집합적 의사 결정을 통해 단어들의 인과관계를 분석합니다. 세 번째 논문은 대형 언어 모델(LLMs), 생성 AI(GenAI) 모델, 그리고 AI 에이전트를 활용해 소프트웨어 프로젝트의 일상적 작업, 리스크 분석 및 예측, 전략 추천을 통해 프로젝트 성공률을 높이는 방법을 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, 전문가가 라벨링한 여러 MCQ 품질 측정 지표에 대해 예측력이 높음을 보여주었습니다. 최근의 deep model들은 높은 정확성을 보이지만 spurious pattern에 의존하는 한계를 갖고 있어, 제안한 대조 학습 접근법은 다양한 차원에서의 개선(1) counterfactual robustness, (2) cross-domain generalization, (3) scarce data에서의 일반화)을 이끌어냈습니다. 마지막으로, AI 도구와 기술은 소프트웨어 개발 프로세스의 효율성을 높이고, 복잡한 과정을 쉽게 이해할 수 있어 프로젝트 관리팀의 성공률을 높이는 잠재력이 있습니다.



### Accelerating Full Waveform Inversion By Transfer Learning (https://arxiv.org/abs/2408.00695)
- **What's New**: 최근 MCQ 자동 생성과 관련된 두 가지 주요 발전이 보고되었습니다. 첫 번째 논문에서는 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안하여 MCQ의 교육적 가치를 보다 정확하게 평가하려고 합니다. 두 번째 논문에서는 NLP 모델의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 사용하는 방법이 제안되었습니다.

- **Technical Details**: [{'Title': 'New Evaluation Metric for MCQ Generation', 'Summary': '기존의 MCQ 평가 메트릭인 BLEU, ROUGE, METEOR는 educational value를 고려하지 않습니다. 이에 대한 해결책으로 KDA라는 새로운 자동 평가 메트릭을 제안했으며, 이 메트릭은 학생이 대상 사실(target fact)을 알고 있는 경우 답할 수 있는지를 측정합니다. 인간 연구를 통해, KDA_disc와 KDA_cont 메트릭이 실제 강의실과 높은 상관관계를 가지고 있음을 보여주었습니다.'}, {'Title': 'Robustness in NLP models', 'Summary': "NLP 모델들은 종종 spurious pattern에 의존하여 robustness에 한계가 있습니다. 이를 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용하는 새로운 방법을 제안하였습니다. 이 방법은 '여러 개의' counterfactual을 생성하여 단어들의 인과관계를 robust하게 평가할 수 있도록 합니다."}, {'Title': 'Full Waveform Inversion with Transfer Learning', 'Summary': 'FWI (Full Waveform Inversion)는 파동 전파로부터 측정된 데이터를 바탕으로 물질 필드를 재구성하는 강력한 도구입니다. NN 기반 FWI를 통해 초기 추측으로부터 NN의 가중치를 반복적으로 업데이트하여 최적화를 수행합니다. 논문은 supervised pretraining을 통해 더 나은 NN weight initialization을 제공하는 transfer learning 접근법을 제안하였고, 이는 더 빠른 수렴과 더 나은 재구성 품질을 제공합니다.'}]

- **Performance Highlights**: [{'Title': 'MCQ Evaluation Metric', 'Details': 'KDA_disc와 KDA_cont 메트릭은 실제 강의실에서의 유용성과 강하게 일치하며, 기존의 n-gram 기반 메트릭과 결합할 때 다양한 expert-labeled MCQ quality measures에 대한 예측력이 높음을 보여주었습니다.'}, {'Title': 'NLP Model Robustness', 'Details': '제안된 방법은 기존의 counterfactual 비슷한 것을 찾는 방법보다 더 robust하게 단어들의 인과관계를 평가할 수 있습니다. 실험 결과, counterfactual robustness, cross-domain generalization, 그리고 scarce data의 generalization에서 유의미한 향상을 보였습니다.'}, {'Title': 'NN-based FWI with Transfer Learning', 'Details': 'Transfer learning을 적용한 NN-based FWI는 수렴 속도와 재구성 품질 면에서 기존의 FWI, pretraining 없이 NN-based FWI, 그리고 pretrained NN를 사용한 FWI보다 우수한 성능을 보여주었습니다.'}]



### Learning in Multi-Objective Public Goods Games with Non-Linear Utilities (https://arxiv.org/abs/2408.00682)
Comments:
          In press at ECAI 2024

- **What's New**: 최근 MCQ(Multiple Choice Questions) 생성을 자동화하여 교사의 평가 시간을 절약할 수 있는 연구들이 활발하다. 그러나 기존의 평가 메트릭들은 교육적 가치를 평가하지 못하고 있다는 문제가 있다. 이에 따라 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하여 MCQ의 대답 가능성과 교육적 유효성을 평가하려 한다.

- **Technical Details**: 기존 방법들은 BLEU, ROUGE, METEOR와 같은 n-gram 기반의 유사성 메트릭을 사용하였으나, 이는 MCQ가 학생의 지식을 평가하는 능력을 고려하지 않는다. KDA는 학생이 대상 사실에 대한 지식을 바탕으로 MCQ를 대답할 수 있는지를 평가하여 이 문제를 해결한다. KDA를 측정하기 위해 인간 설문 조사를 통한 학생 응답을 기반으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 개발하였다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 활용 가능성과 높은 상관관계를 가짐을 확인하였다. 또한, n-gram 기반의 유사성 메트릭과 결합하면 다양한 전문가 레이블 MCQ 품질 지표에 대한 예측력을 크게 향상시킬 수 있음을 보여준다.



### AMAES: Augmented Masked Autoencoder Pretraining on Public Brain MRI Data for 3D-Native Segmentation (https://arxiv.org/abs/2408.00640)
- **Multiple Choice Questions Generation Evaluation**: [{"What's New": '기존의 MCQ 생성 평가 메트릭이 교육적 가치를 무시한다는 문제를 해결하기 위해, 지식 종속 가능성(KDA)을 도입. 이는 MCQ의 학생 지식 평가 능력과 대답 가능성을 측정하는 새로운 자동 평가 메트릭입니다.'}, {'Technical Details': 'KDA는 학생 응답을 바탕으로 측정되며, 두 개의 자동 평가 메트릭 KDA_disc와 KDA_cont가 도입되었습니다. 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA 값을 근사합니다. KDA_disc와 KDA_cont는 n-gram 기반의 유사성 메트릭과 결합하여 높은 예측력을 지닙니다.'}, {'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성 (usability) 과 강한 상관관계를 가진다는 점을 입증했습니다.'}]

- **NLP Model Robustness**: [{"What's New": '최근 딥러닝 모델이 NLP 태스크에서 높은 정확성을 보였지만, spurious pattern에 의존함으로 인해 robust하지 않은 문제가 보고되었습니다. 이에 대해 본 연구는 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 새로운 접근을 제안합니다.'}, {'Technical Details': '기존의 augmentation 방법들이 spurious correlation에 영향을 받는 것과 달리, 우리의 방법은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 robust하게 단어들의 인과관계를 파악합니다. 이는 모델의 편향에 덜 민감하게 만들며 다양한 차원에서 성능을 개선합니다.'}, {'Performance Highlights': '감소된 task 모델의 편향과 함께 다양한 차원에서 반사실적 강건성, 도메인 간 일반화, 적은 데이터로부터의 일반화 성능에서 크게 향상된 결과를 보였습니다.'}]

- **3D Semantic Segmentation Pretraining**: [{"What's New": '대규모, 도메인 특화 데이터셋을 이용한 3D semantic segmentation 모델의 자가 지도 사전훈련(self-supervised pretraining)이 성능에 미치는 영향을 조사했습니다. 이를 위해 44,756개 브레인 MRI 볼륨을 포함하는 BRAINS-45K 데이터셋을 소개합니다.'}, {'Technical Details': 'AMAES 프레임워크는 Masked Image Modeling (MIM)과 intensity-based augmentation reversal을 결합하여 메모리 사용, 실행 시간, 그리고 finetuning 성능의 균형을 잡습니다. U-Net과 MedNeXt라는 두 가지 백본 아키텍처를 사용하여 세 가지 다운스트림 작업에서 사전 훈련의 효과를 평가했습니다.'}, {'Performance Highlights': '사전 훈련된 AMAES 모델은 대부분의 평가된 경우에서 분할 성능을 크게 향상시켰으며, augmentation을 사용한 사전 훈련이 대형 데이터셋에서도 유익하다는 점을 확인했습니다.'}]



### DisTrack: a new Tool for Semi-automatic Misinformation Tracking in Online Social Networks (https://arxiv.org/abs/2408.00633)
- **What's New**: ['자동 생성된 다중 선택 질문(MCQ)을 평가하는 새로운 메트릭, Knowledge Dependent Answerability(KDA)을 소개합니다. 이 메트릭은 학생이 특정 사실에 대한 지식을 바탕으로 질문에 답할 수 있는지 평가합니다.', '대조 학습과 반사실적 데이터 증강을 활용하여 NLP 모델의 강건성을 향상시키기 위한 새로운 접근법을 제안합니다. 이 방법은 여러 개의 반사실적(counterfactuals) 질문을 종합적으로 생성하여 각 용어의 인과성을 감독합니다.', 'DisTrack라는 새로운 툴을 소개합니다. 이 툴은 온라인 소셜 네트워크(OSN)에서 허위 정보의 추적 및 분석을 목적으로 개발되었으며, NLP, 소셜 네트워크 분석(SNA), 그래프 시각화를 결합하여 허위 정보의 확산을 감시하고 출처를 식별합니다.']

- **Technical Details**: ['KDA는 인공지능 학습 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 자동 평가 지표 KDA_disc와 KDA_cont로 구체화됩니다. 이는 BLEU, ROUGE, METEOR와 같은 기존의 n-gram 기반 유사성 메트릭과 결합하여 MCQ의 품질 예측력을 높입니다.', '대조 학습과 반사실적 증강(counterfactual augmentation)은 기존의 주목할 만한 근거(correlation)에 의존하지 않고 여러 반사실적 질문들을 생성하여 집단적인 의사 결정(collective decisions)을 통해 강건하게 감독합니다. 이는 다양한 측면에서 강건성, 도메인 간 일반화, 그리고 희소 데이터 상황에서의 일반화를 향상시킵니다.', 'DisTrack의 아키텍처는 키워드 검색, 의미적 유사성 평과, 그래프 생성 기술을 포함합니다. 이를 통해 허위 정보의 전파를 감시하고, 내용 기반으로 카테고리화하며, 전파 경로를 시각화합니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 교실 환경에서 전문가 관찰자들이 라벨링한 사용성과 강한 상관관계를 보였습니다. 이는 MCQ의 품질 예측에 있어 강력한 성능을 나타냈습니다.', '대조 학습을 통한 접근법은 반사실적 강건성, 도메인 간 일반화, 희소 데이터 상황에서의 일반화를 포함한 여러 차원에서 유의미한 개선을 보여주었습니다.', 'DisTrack은 COVID-19 백신 관련 허위 정보, 정치적 이슈, 러시아-우크라이나 갈등 등 세 가지 사례 연구를 통해 허위 정보의 출처를 식별하고 그 진화 과정을 추적하는 데 효과적임을 증명했습니다.']



### Closing the gap between open-source and commercial large language models for medical evidence summarization (https://arxiv.org/abs/2408.00588)
- **What's New**: 이 논문은 MCQ(Multiple Choice Questions) 자동 생성 및 평가를 위한 새로운 지표(Knowledge Dependent Answerability, KDA)를 제안하였습니다. 기존의 지표들이 문장 유사성에 초점을 맞추는 반면, KDA는 학생의 지식 평가 능력을 중점적으로 측정합니다. 다른 논문에서는 최신 NLP 모델들의 강력한 성능에도 불구하고, 제한된 robustness 문제를 해결하기 위해 대조 학습(Contrastive Learning)과 반사실적 증강(Counterfactual Augmentation)을 도입합니다. 또한, 의료 분야에서 열린 소스 LLMs의 성능을 향상시키기 위해 세부적 조정을 시도한 연구도 포함되어 있습니다.

- **Technical Details**: MCQ 평가를 위해 KDA_disc와 KDA_cont라는 지표를 제안했습니다. 이는 사전학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방합니다. 대조 학습을 위해 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 spurious correlations를 제거하는 방법을 제안합니다. 의료 증거 요약을 위해 PRIMERA, LongT5, Llama-2 모델을 세부 조정하여 성능 향상을 목표로 했으며, Low-Rank Adaptation (LoRA) 방법을 사용해 세부 조정을 최적화했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 다른 n-gram 기반 유사성 지표와 결합하여 높은 예측력을 나타냈습니다. 대조 학습 접근 방식은 다양한 측면에서 개선된 robustness, cross-domain generalization, 그리고 데이터가 부족한 상황에서도 유의미한 성능 향상을 보였습니다. MedReview 데이터셋을 사용한 실험에서는 세부 조정된 모델들이 ROUGE-L, METEOR, CHRF 점수에서 유의미한 향상을 보였으며, 특히 LongT5 모델은 GPT-3.5와 비슷한 성능을 보여주었습니다.



### Non Verbis, Sed Rebus: Large Language Models are Weak Solvers of Italian Rebuses (https://arxiv.org/abs/2408.00584)
Comments:
          Code: this https URL. Artifacts: this https URL

- **What's New**: 자동 MCQ(Multiple Choice Question) 생성 알고리즘에서 n-gram 기반 유사성 평가 메트릭이 교육적 가치를 무시하는 문제를 해결하기 위해, 지식 종속 가능성(KDA, Knowledge Dependent Answerability)이라는 새로운 평가 지표를 제안했습니다.

- **Technical Details**: KDA는 대상 사실(target fact)에 대한 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. 구체적으로, 학생 응답을 기반으로 KDA를 측정하는 방법을 제시한 후, 사전 훈련된 언어 모델을 활용하여 KDA를 근사하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안했습니다.

- **Performance Highlights**: Human evaluation을 통해, KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 가지는 것을 보여주었습니다. 그리고 n-gram 기반 유사성 메트릭과 결합하면 전문가가 라벨링한 다양한 MCQ 품질 지표들에 대해 높은 예측력을 가집니다.



### Alleviating Hallucination in Large Vision-Language Models with Active Retrieval Augmentation (https://arxiv.org/abs/2408.00555)
- **What's New**: 최근 논문에서 여러 가지 혁신적인 새로운 평가 및 학습 방법론을 소개했습니다. 첫 번째로, 자동화된 다지 선다형 질문(MCQ) 생성의 평가를 위해 기존 BLEU, ROUGE, METEOR와 같은 메트릭이 교육적 가치를 무시한다는 문제를 해결하기 위해 새로운 지식 종속 가능성(KDA) 메트릭을 제안했습니다. 두 번째로, NLP 태스크에서의 deep model의 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용했습니다. 마지막으로, 대규모 비전-언어 모델(LVLMs)의 환각현상(hallucination)을 줄이기 위해 외부 지식 자원을 통한 액티브 리트리벌-보강 모델(ARA)을 제안했습니다.

- **Technical Details**: 지식 종속 가능성(KDA)를 통해 자동 MCQ의 대답 가능성을 평가했습니다. KDA를 불러오는 과정에서는 인간의 설문 응답을 기초로 합니다. 이후 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 개발해 미리 학습된 언어 모델을 통해 학생들의 문제해결 행동을 모방했습니다. NLP 퍼포먼스 개선을 위해서는 counterfactual을 생성하여 튜닝된 모델의 스푸리어스 패턴을 줄였습니다. 액티브 리트리벌-보강 모델에서는 LVLMs의 환각현상을 줄이기 위해 (i) 이미지 계층 구조를 분해하고 (ii) 효과적인 리트리벌 방법을 선정하며 (iii) 리트리벌 시간을 조정하는 세 가지 주요 차원을 고려한 접근을 취했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 세팅과 강한 상관관계를 보였으며, 전문가가 표기한 MCQ 품질 지표와도 높은 예측력을 보였습니다. 새로운 대조 학습 방법은 이종 도메인 일반화 및 드문 데이터에서의 일반화 측면에서 크게 향상된 성능을 달성했습니다. ARA 모델은 세 가지 널리 사용되는 LVLM(LLaVA-1.5, Qwen-VL, mPLUG-Owl2)을 사용하여 네 가지 벤치마크에서 환각현상을 효과적으로 줄이는 성과를 보였습니다.



### Mitigating Multilingual Hallucination in Large Vision-Language Models (https://arxiv.org/abs/2408.00550)
- **What's New**: 최근 자동 생성 다지선다형 문제(MCQ) 평가에 있어서 교육적 가치를 측정하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)가 제안되었습니다. 이와 더불어, 대조 학습과 반사실적 증강을 활용한 NLP 딥 모델의 강건성 향상 방법과 LVLM(대형 비전-언어 모델)의 다중언어 환각 문제를 완화하는 새로운 프레임워크가 소개되었습니다.

- **Technical Details**: KDA는 학생의 지식 기반으로 MCQ의 답변 가능성을 평가합니다. 대조 학습 및 반사실적 증강을 통한 NLP 모델의 반사실적 강건성과 도메인 간 일반화를 목표로 합니다. 다중언어 환각 제거(MHR) 프레임워크는 다양한 언어에서의 환각을 줄이기 위해 다단계 접합 방법과 직접 선호 최적화를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭이 실제 강의실 세트에서의 사용성과 높은 상관성을 보였습니다. LVLMs의 MHR 프레임워크는 POPE 벤치마크에서 평균 19.0%의 정확도 향상을 달성했습니다. 다양한 언어에서의 강건성과 일반화 능력도 현저히 개선되었습니다.



### Learning to Embed Distributions via Maximum Kernel Entropy (https://arxiv.org/abs/2408.00549)
- **What's New**: 이 논문은 기존 자동 MCQ 생성 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해, '지식 종속 가능성(KDA)'이라는 새로운 자동 평가 메트릭을 제안합니다. NLP 태스크에서 모델의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 사용하는 방법도 제시합니다. 마지막으로 데이터 종속 분포 커널을 학습하기 위한 새로운 목표를 제안합니다.

- **Technical Details**: [{'MCQ Generation': 'BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭의 한계를 극복하기 위해, 인간 설문조사를 통해 학생 응답을 기반으로 KDA를 측정하고, 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방한 KDA_disc와 KDA_cont를 제안합니다.', 'NLP Robustness': 'contrastive learning과 counterfactual augmentation을 사용해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안합니다.', 'Distribution Regression': '엔트로피 최대화 원칙을 기반으로 데이터 종속 분포 커널을 비지도 학습하는 새로운 목적을 제안하며, 커널 평균 임베딩과 공분산 연산자 임베딩에 대한 이론적 성질을 검토합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont가 실제 강의실 설정에서의 사용성 및 다양한 전문가 지정 MCQ 품질 측정치에 대해 강한 예측력을 가진다는 것을 인간 연구를 통해 보여줍니다.', 'NLP Robustness': '제안된 방법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등의 여러 차원에서 상당한 성능 향상을 달성하였습니다.', 'Distribution Regression': '비지도 학습된 데이터 종속 커널은 다양한 모달리티에서의 분류 작업에서도 뛰어난 성능을 입증했습니다.'}]



### The Energy Cost of Artificial Intelligence of Things Lifecyc (https://arxiv.org/abs/2408.00540)
Comments:
          12 pages, 13 figures

- **What's New**: 자동 생성되는 MCQ(Multiple Choice Questions)와 관련해, BLEU, ROUGE, METEOR 같은 기존의 평가 메트릭들은 교육적 가치를 평가하지 못한다는 문제점을 지적하고, 지식 종속 가능성(KDA, Knowledge Dependent Answerability)이라고 불리는 새로운 자동 평가 메트릭 제안을 통해 이 문제를 해결하고자 합니다.

- **Technical Details**: KDA는 학생들이 대답 가능한지 여부를 측정하고, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 통해 이를 계량화합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: 실제 인간 평가를 통해 KDA_disc와 KDA_cont는 KDA와 높은 상관관계를 가지며, 실제 교실 환경에서의 사용 가능성도 높다는 것을 보여줍니다. 또한, n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대한 예측력이 강함을 확인하였습니다.



### Intermittent Semi-working Mask: A New Masking Paradigm for LLMs (https://arxiv.org/abs/2408.00539)
- **What's New**: 최근 자동 MCQ (Multiple Choice Questions) 생성 및 LLM (Large Language Models) 대화 시스템에서의 향상된 메트릭과 방법론이 제안되었습니다. MCQ 생성의 경우, 새로운 평가 메트릭인 지식 종속 가능성 (KDA)을 제안하여 질문의 교육적 가치를 평가합니다. 반면에 LLM 대화 시스템에서는 Intermittent Semi-working Mask (ISM)이라는 새로운 마스킹 기법을 도입하여 다중 회전 대화 중의 응답 생성 지연을 줄입니다.

- **Technical Details**: [{'MCQ Generation': '기존 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR은 n-gram 기반의 유사성만을 고려하여 교육적 가치를 평가하는 데 한계를 가집니다. 새로운 평가 메트릭 KDA는 학생의 대답 가능성을 측정하고, 인간 설문으로부터 얻은 학생 반응을 통해 이를 평가합니다.'}, {'LLM Dialogue System': '기존 LLM 모델들은 인과적 LLM(causal LLM)과 접두사 LLM(prefix LLM)으로 나뉘며, 접두사 LLM은 다중 회전 대화에서 역사적 문맥을 잘 활용하지만, 높은 생성 지연의 문제를 가집니다. 새로운 마스킹 기법인 ISM은 쿼리와 응답에 대해 교차적인 양방향 및 단방향 어텐션을 적용하여 이 문제를 개선합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 KDA와 강한 상관관계를 보이며, 전문가에 의해 라벨링된 실제 강의실 세트에서의 사용성과도 매우 높은 상관관계를 나타냈습니다.'}, {'LLM Dialogue System': 'ISM은 GPT-4를 사용하여 평가한 다중 회전 대화 벤치마크 데이터셋에서 기존 최첨단 생성 품질을 유지하면서도 낮은 대화 생성 지연을 달성했습니다.'}]



### Hilbert curves for efficient exploratory landscape analysis neighbourhood sampling (https://arxiv.org/abs/2408.00526)
Comments:
          A version of this paper is published as conference proceedings of EvoApps 2024

- **What's New**: 최근 공개된 논문에서는 Multiple Choice Questions (MCQ)를 자동으로 생성하고 이를 평가하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 또한 NLP 태스크의 모델들이 spurious patterns에 의존해 robustness가 제한되는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation 방법을 제시합니다. 마지막으로 Hilbert space-filling curves를 활용한 새로운 landscape analysis 샘플링 전략을 소개합니다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR 등 기존 평가 메트릭이 educational value를 고려하지 않는 문제를 지적하며, KDA라는 새로운 메트릭을 통해 학생들의 지식 평가 능력을 측정합니다. 두 번째 논문에서는 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안합니다. 세 번째 논문에서는 Hilbert curves를 사용하여 high-dimensional 공간에서 효과적으로 공간적으로 상관된 샘플을 생성하고 이를 통해 복잡한 최적화 문제를 분석합니다.

- **Performance Highlights**: KDA 방식의 두 가지 변형인 KDA_disc와 KDA_cont가 실제 강의실 세트에서 높은 유용성을 보였고, n-gram based similarity metrics와 결합했을 때 다양한 MCQ 품질 측정에서 높은 예측력을 가졌습니다. 또한 Hilbert curves는 기존 nearest neighbour 주문 방법에 비해 계산 비용이 현저히 낮으면서도 샘플링된 특징들의 saliency를 유지하는 데 효과적임을 보였습니다.



### Jailbreaking Text-to-Image Models with LLM-Based Agents (https://arxiv.org/abs/2408.00523)
- **What's New**: 최근 연구에서는 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 제안하여 자동 MCQ 생성의 교육적 가치를 측정한다. KDA는 특정 사실에 대한 학생의 지식을 평가하는 능력을 기준으로 MCQ의 대답 가능성을 평가한다. 또한, 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 NLP 모델의 robustness를 향상시키는 방법도 제안되었는데, 이는 여러 개의 반사실적 데이터를 생성하고 집단적 의사 결정을 통해 단어 인과 관계를 강건하게 파악한다. 마지막으로, 대규모 언어 모델(LLM) 기반의 다중 에이전트 프레임워크인 Atlas가 소개되었으며, 이는 텍스트-이미지 모델(T2I models)의 안전 필터를 우회하는 혁신적 방법을 제시한다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR 등의 기존 평가 메트릭이 MCQ의 교육적 가치를 고려하지 않는 문제를 해결하기 위해 KDA라는 새로운 메트릭을 도입했다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 KDA를 측정하였다. 두 번째 논문에서는 반사실적 증강을 활용한 대조 학습 방법을 제안하여 모델의 robustness를 향상시켰다. 단일 반사실적 데이터가 아닌 여러 개의 반사실적 데이터를 생성하고, 집단적 의사 결정을 통해 더 신뢰할 수 있는 예측을 했다. 세 번째 논문에서는 다중 에이전트가 협력하여 적응형 프롬프트(adaptive-mode prompt)를 사용하여 텍스트-이미지 모델의 안전 필터를 우회하는 Atlas 프레임워크를 제안하였다. Atlas는 시각-언어 모델(VLM)을 활용하여 필터를 우회할 수 있는 프롬프트를 생성하고, 체인 오브 소트(COT)과 인 컨텍스트 학습(ICL)을 통해 에이전트의 추론 능력을 강화한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 MCQ 품질 측정치와도 높은 예측력을 보였다. 대조적 학습법을 통한 모델은 여러 차원에서 상당한 개선을 이루었으며, 특히 반사실적 robustness, 크로스-도메인 일반화, 부족한 데이터로부터의 일반화에서 높은 성과를 나타냈다. Atlas 프레임워크는 텍스트-이미지 모델의 안전 필터를 거의 100% 우회할 수 있었고, 쿼리 효율성과 생성된 이미지 품질 면에서 기존 방법들보다 우수한 성과를 보였다.



### Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation (https://arxiv.org/abs/2408.00490)
Comments:
          14 pages

- **What's New**: 세 개의 연구 논문에서 새로운 방법론을 제안했습니다. 첫 번째 논문은 Multiple Choice Questions (MCQ)의 새로운 자동 평가 메트릭(Knowledge Dependent Answerability, KDA)을 소개하고, 학생의 답변 가능성을 평가하는 데 초점을 맞췄습니다. 두 번째 논문은 NLP 태스크에서 깊이 모델의 강인성을 개선하기 위해 대조 학습과 반사실적(contrafactual) 증강을 활용하는 새로운 방법을 제안했습니다. 세 번째 논문은 그래프 신경망(Graph Neural Networks, GNNs)을 사용한 추천 시스템의 Out-of-Distribution (OOD) 데이터에서의 일반화 성능을 향상시키기 위한 새로운 접근법(CausalDiffRec)을 제안했습니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 고려하지 않기 때문에 새로운 평가 메트릭인 KDA를 제안하였습니다. KDA는 학생의 대상 사실에 대한 지식에 기반한 MCQ의 답변 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용해 학생의 문제 해결 능력을 모방하여 자동 평가합니다.', 'NLP Robustness': "대조 학습과 반사실적 증강을 활용하여 깊이 모델의 강인성을 개선하는 새로운 접근법을 제안합니다. '여러 개의' 반사실적을 생성하고 집합적 의사 결정(collective decisions)을 통해 인과 관계를 더 강하게 파악합니다.", 'GNN-based Recommendations': '구조적 인과 모델(Structural Causal Model, SCM)을 구성하여 상호 작용 데이터를 분석하고 환경 혼동 변수(environmental confounders)를 제거하여 GNN 모델의 OOD 데이터에 대한 일반화 성능을 향상시키기 위해 Causal Diffusion 기반의 그래프 표현 학습(CausalDiffRec) 방법을 제안합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'Human studies에 따르면, KDA_disc와 KDA_cont는 전문가들이 레이블링한 실제 강의실에서의 사용성과 강한 상관관계를 보였습니다. 이 메트릭을 기존의 n-그램 기반 유사성 메트릭과 결합할 경우 다양한 MCQ 품질 측정에 대해 강력한 예측력을 가지게 됩니다.', 'NLP Robustness': '우리의 접근법은 반사실적 강인성, 도메인 간 일반화, 희소 데이터에서의 일반화 등 다양한 차원에서 기존 방법보다 더 높은 성능을 보였습니다.', 'GNN-based Recommendations': 'CausalDiffRec은 다양한 OOD 데이터셋에서의 일반화 성능을 최대 36.73% 향상시켰으며, Food에서 10.69%, KuaiRec에서 18.83%, Yelp2018에서 22.41%, Douban에서 11.65%의 평균 성능 향상을 보였습니다.'}



### A Systematic Review on Long-Tailed Learning (https://arxiv.org/abs/2408.00483)
Comments:
          Current Under Revision at IEEE TNNLS. [This is the long/Full-length version of our Long-Tailed Learning Survey paper]

- **What's New**: 자동 MCQ(다지선다 질문) 생성의 평가 메트릭에 대한 새로운 접근인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었습니다. 또한 최신 딥러닝 모델의 강건성을 향상하기 위해 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 이용하는 새로운 방법론이 소개되었습니다. 마지막으로, 롱테일 학습의 최근 발전을 포괄적으로 조사하고 이를 8가지 차원으로 분류하는 새로운 분류 체계가 제안되었습니다.

- **Technical Details**: MCQ 생성의 새로운 평가 메트릭인 KDA는 학생이 대상 지식에 대한 이해를 바탕으로 대답 가능성을 평가합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하며, 이들은 실제 강의실 세팅의 사용성과 높은 상관관계를 가지고 있습니다. 딥러닝 모델의 강건성 개선을 위한 새로운 접근은 반사실을 집합적으로 생성하여 각 용어의 인과 관계를 더 강건하게 감독할 수 있도록 합니다. 롱테일 학습에 대한 포괄적인 조사는 데이터 균형 조정, 신경망 아키텍처, 특징 풍부화, 손실 함수 조정 등 8가지 차원을 포함합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 강한 상관관계를 보이며, 전문가가 라벨링한 다양한 MCQ 품질 메트릭에 대해 높은 예측력을 보입니다. 반사실 증강 방법론은 반사실 강건성, 도메인 간 일반화, 소량 데이터 일반화 등 다양한 측면에서 상당한 개선을 달성했습니다. 롱테일 학습 방법들은 객체 인식, 객체 탐지, 이미지 분할 등의 하위 작업에서 꼬리 클래스의 인식 정확도를 크게 향상시킵니다.



### Towards Explainable and Interpretable Musical Difficulty Estimation: A Parameter-efficient Approach (https://arxiv.org/abs/2408.00473)
- **What's New**: 최근 논문에서는 자동 Multiple Choice Questions(MCQ) 생성의 교육적 가치를 평가하기 위해 novel한 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. 또한, NLP 모델의 robustness 향상을 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안합니다. 마지막으로, 음악 교육에서의 난이도 추정을 위한 투명하고 이해 가능한 방법을 제시하며, 특히 피아노 작품의 난이도 추정을 위한 새로운 white-box 모델인 RubricNet을 소개합니다.

- **Technical Details**: [{'paper': 'Automatic MCQ Generation', 'summary': '기존의 자동 MCQ 생성 메트릭은 BLEU, ROUGE, METEOR와 같은 n-gram 유사성에 초점을 맞추고 있지만, 이는 교육적 가치를 간과하고 있습니다. 이를 해결하기 위해 제안된 KDA는 MCQ의 대답 가능성을 측정하고, 학생의 지식을 평가하는 능력을 분석합니다. 이 메트릭은 학생의 응답을 기반으로 한 KDA와, 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont로 구성됩니다.'}, {'paper': 'NLP Robustness', 'summary': '최근 NLP 모델들이 spurious pattern에 의존해 robustness가 제한된다는 문제를 다룹니다. 인간의 도움 없이 여러 개의 counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과관계를 분석하는 방법을 제안합니다. 이를 통해 다양한 측면에서 성능 향상이 이루어졌음을 실험적 결과로 증명했습니다.'}, {'paper': 'Music Difficulty Estimation', 'summary': '음악 작품의 난이도를 추정하기 위해 이해 가능한 설명을 제공하는 새로운 white-box 모델인 RubricNet을 소개합니다. 이 모델은 음악 교육에서 흔히 사용되는 루브릭(rubric) 개념을 활용하여 투명성과 해석 가능성을 높였습니다. 평가 결과는 피아노 레퍼토리의 9개 등급을 기준으로 큰 정확도와 낮은 평균 제곱 오차(MSE)를 보여주었습니다.'}]

- **Performance Highlights**: [{'paper': 'Automatic MCQ Generation', 'summary': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가가 라벨한 MCQ 품질 측정에 강한 예측 능력을 가집니다.'}, {'paper': 'NLP Robustness', 'summary': '제안한 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization에서 유의미한 성능 향상을 보였습니다.'}, {'paper': 'Music Difficulty Estimation', 'summary': 'RubricNet 모델은 9개의 피아노 레벨을 기준으로 41.4%의 독립적인 정확도와 1.7의 평균 제곱 오차(MSE)를 기록하며, 해석 가능하고 높은 정확도의 결과를 제공했습니다.'}]



### Image Super-Resolution with Taylor Expansion Approximation and Large Field Reception (https://arxiv.org/abs/2408.00470)
- **What's New**: 이 논문은 MCQ (Multiple Choice Questions)의 자동 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(KDA; Knowledge Dependent Answerability)을 제안합니다. 이는 학생이 관련된 목표 사실에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 메트릭은 생성된 MCQ와 데이터셋의 골드 샘플간의 n-gram 기반 유사성에 집중하여 교육적 가치를 간과합니다. 반면, 우리가 제안한 KDA는 학생 반응을 통해 MCQ의 효과를 측정하며, 이를 바탕으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입합니다. 이는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또, Blind Super-Resolution(SR)과 관련된 논문은 자가 유사성 계산의 높은 연산 복잡성을 다루기 위한 STEA(Second-order Taylor Expansion Approximation)와 MLFR(Multi-Scale Large Field Reception) 설계를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 높은 상관관계를 나타냈습니다. 추가로 n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가 라벨링 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다. Blind SR에서 제안된 LabNet과 RealNet은 각각 실험실과 실제 환경에서 뛰어난 성능을 발휘합니다.



### DiscipLink: Unfolding Interdisciplinary Information Seeking Process via Human-AI Co-Exploration (https://arxiv.org/abs/2408.00447)
- **What's New**: 자동 MCQ 생성을 위한 기존 평가 메트릭은 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)이라는 새로운 메트릭을 제안. 또한, 최근 NLP deep models의 robustness 향상을 위해 종속 패턴에 의존하지 않는 대조적 학습(contrastive learning) 및 counterfactual augmentation 방법을 도입. 마지막으로, DiscipLink이라는 새로운 인터랙티브 시스템을 소개, 연구자들이 대형 언어 모델(LLMs)과 협력하여 다양한 학문 분야에서 정보를 탐색할 수 있도록 지원.

- **Technical Details**: KDA 측정은 학생의 응답을 기반으로 학습된 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 pre-trained language models을 활용해 학습자의 문제 해결 행동을 모방해 측정됨. 대조적 학습 및 counterfactual augmentation 방법은 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어의 인과관계를 robust하게 지도를 제안. DiscipLink 시스템은 사용자의 관심 주제를 바탕으로 탐색 질문을 생성하고 분야별 용어를 사용해 자동으로 쿼리를 확장해 논문을 검색.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 실제 강의실 사용성과 강한 상관관계가 있음이 확인됨. 여러 차원에서 robustness, cross-domain generalization 및 일반화가 매우 향상됨. DiscipLink 시스템은 기존 문헌 검색 도구들보다 더 효율적이고 포괄적인 지식 범위를 탐색 가능하게 함. DiscipLink를 사용한 경험 많은 대화 연구자들은 시스템의 공동 탐색 워크플로우에 크게 만족했으며 일부 제한점을 지적함.



### Focus, Distinguish, and Prompt: Unleashing CLIP for Efficient and Flexible Scene Text Retrieva (https://arxiv.org/abs/2408.00441)
Comments:
          Accepted by ACM MM 2024

- **What's New**: We have explored three recent papers that bring forth novel approaches to different AI challenges. The first paper addresses the limitations of existing Multiple Choice Questions (MCQ) generation evaluation metrics by proposing a new Knowledge Dependent Answerability (KDA) metric. The second paper enhances deep model robustness in Natural Language Processing (NLP) tasks through contrastive learning and counterfactual augmentation. Finally, the third paper introduces an Optical Character Recognition (OCR)-free model for Scene Text Retrieval (STR), leveraging a combination of Contrastive Language-Image Pre-training (CLIP) and a novel 'Focus, Distinguish, and Prompt' (FDP) approach.

- **Technical Details**: 1. **Knowledge Dependent Answerability (KDA)**: This new metric measures the answerability of an MCQ given knowledge of the target fact, using KDA_disc and KDA_cont based on pre-trained language models to simulate student behavior.
2. **Contrastive Learning and Counterfactual Augmentation**: This approach creates multiple counterfactual examples and uses collective decision-making to reduce the sensitivity to spurious patterns, improving robustness in diverse dimensions like counterfactual robustness, cross-domain generalization, and generalization from scarce data.
3. **CLIP and FDP for Scene Text Retrieval**: FDP focuses on text within images by applying rough text localization and distinguishing content words from function words. A semantic-aware prompting scheme converts query text into a learnable prompt for efficient and flexible text retrieval.

- **Performance Highlights**: 1. **KDA Metrics**: Human studies showed that KDA_disc and KDA_cont metrics have strong correlations with KDA and classroom usability, providing a significant predictive power for various expert-labeled MCQ quality measures.
2. **Counterfactual Augmentation Approach**: Empirical results indicate significant improvements in robustness and generalization over traditional augmentation methods.
3. **FDP Model**: Compared to existing OCR-based methods, FDP enhances the inference speed by 4 times and improves retrieval accuracy by 4.37% in the IIIT-STR benchmark. Additional experiments validate its effectiveness in handling diverse forms of query text.



### A Qualitative Study on Using ChatGPT for Software Security: Perception vs. Practicality (https://arxiv.org/abs/2408.00435)
Comments:
          Accepted for publication at International Conference on Trust, Privacy and Security - 2024

- **What's New**: 이번 연구에서는 최근의 자연어 처리(NLP) 태스크에서 사람보다 나은 정확성을 보이는 deep model들이 여전히 spurious pattern에 의존하여 robustness가 제한되는 문제를 해결하기 위해, contrastive learning과 counterfactual augmentation을 활용한 새로운 접근법을 제안했습니다. 또한, ChatGPT와 같은 대형 언어 모델(LLMs)이 소프트웨어 보안 작업에서의 잠재력을 조사했습니다. 특히, 자동 MCQ 생성의 교육적 가치를 평가하는 새로운 메트릭 KDA를 제안했습니다.

- **Technical Details**: 첫 번째 연구에서는 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 MCQ의 교육적 가치를 고려하지 않는 문제를 해결하기 위해, Knowledge Dependent Answerability (KDA)라는 새로운 메트릭을 제안했습니다. 이를 위해 우리는 human survey 데이터를 이용하여 KDA를 측정하고, pre-trained language models를 활용하여 자동 평가 메트릭 KDA_disc와 KDA_cont를 개발했습니다. 두 번째 연구에서는 여러 가지 counterfactual을 생성하고 collective decision-making을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안했습니다. 세 번째 연구에서는 ChatGPT의 소프트웨어 보안 작업 지원 능력을 조사하기 위해 사용자의 의견을 분석하고, 취약점 탐지 작업에서 ChatGPT의 실무 적용 가능성을 평가했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 사용성과 강한 상관관계를 갖는 것으로 나타났으며, n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 평가한 MCQ의 품질 측정에서 높은 예측력을 보였습니다. 이와 함께, 제안된 counterfactual augmentation 방법은 다양한 차원에서 기존 방법보다 더 높은 robustness를 보였으며, ChatGPT는 취약점 탐지에서 대부분 일반적인 보안 정보를 제공하여 산업용으로는 적합하지 않을 수 있음이 드러났습니다.



### Augmenting Channel Simulator and Semi- Supervised Learning for Efficient Indoor Positioning (https://arxiv.org/abs/2408.00429)
Comments:
          ACCEPTED for presentation at 2024 IEEE Global Communications Conference

- **What's New**: 교사들이 학생 평가에 소요하는 시간을 크게 줄일 수 있도록 자동 다중선택형 질문(MCQ) 생성을 위한 새로운 자동 평가 메트릭 지식 종속 가능성(KDA)을 도입했습니다. 또한, 대규모 데이터 라벨링 필요성을 줄이기 위해, 반자율 학습(Semi-Supervised Learning)과 업데이트된 채널 시뮬레이터를 활용한 내부 위치 측정 방법이 제안되었습니다.

- **Technical Details**: ['MCQ 자동 생성 평가: 지식 종속 가능성(KDA)은 MCQ의 대상 사실 관련 지식 기반으로 대답 가능성을 측정합니다. KDA를 학생 응답 기반으로 측정하고, 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안했습니다.', 'NLP 모델 강화: 최근 심층 모델의 높은 정확성에도 불구하고 spurious pattern에 의존한 robustness 문제를 contrastive learning과 counterfactual augmentation을 통해 해결하고자 했습니다. 기존 방법들과 달리, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더 잘 파악하는 방법을 제안했습니다.', '반자율 학습을 통한 내부 위치 측정: 업데이트된 채널 시뮬레이터(UCHS)를 이용해 라벨되지 않은 데이터를 생성하고, SSL 반자율 학습 알고리즘을 통해 라벨링 및 비라벨링 데이터를 함께 활용했습니다. 특히, 비라벨링 데이터의 confidence 값을 기반으로 가중치를 적용해 정확도를 높였습니다.']

- **Performance Highlights**: ['자동 MCQ 생성 평가: KDA_disc와 KDA_cont는 실제 강의실 사용성 평가에서 강한 상관관계를 보였고, 전문가가 라벨링한 다양한 MCQ 품질 측정에서도 예측 능력이 뛰어났습니다.', 'NLP 모델 강화: 다양한 차원(반사실적 robustness, 도메인 간 일반화, 희소 데이터 일반화)에서 의미있는 성능 향상을 달성했습니다.', '반자율 학습을 통한 내부 위치 측정: 제안된 방법은 기존 벤치마크 대비 우수한 성능을 보였으며, 측정 오버헤드와 학습 비용을 최소화해 실내 위치 결정의 실용성과 가치를 크게 향상시켰습니다.']



### CARMIL: Context-Aware Regularization on Multiple Instance Learning models for Whole Slide Images (https://arxiv.org/abs/2408.00427)
- **What's New**: 최근 딥러닝 모델이 NLP에서 사람보다 높은 정확도를 보여주고 있음에도 불구하고, spurious 패턴에 의존하여 robustness에 한계가 있는 것으로 보고되고 있습니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 새로운 방법을 제안합니다. 또한, 종양학적 Whole Slide Images(WSIs)를 조금 더 정밀하게 분석할 수 있도록 Context-Aware Regularization (CARMIL)을 도입하고, 이를 통해 암 예후 예측 모델의 성능을 강화하는 연구도 소개되었습니다.

- **Technical Details**: [{'Automatic MCQ Generation': '기존의 MCQ 평가 메트릭은 BLEU, ROUGE, METEOR와 같이 n-gram 기반 유사성을 측정하여 교육적 가치를 고려하지 않습니다. 이를 극복하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안하며, 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방함으로써 KDA를 approximating 하는 KDA_disc와 KDA_cont를 소개합니다.'}, {'Robustness in NLP': '기존의 counterfactual 기반 방법들은 spurious 패턴에 의해 영향을 받는 반면, 제안된 방법은 여러 개의 counterfactual을 생성하고 하나의 집합적인 의사 결정을 통해, 각 용어의 인과관계를 보다 robust하게 파악합니다.'}, {'Cancer Prognosis from WSIs': '기존 MIL 모델은 이미지 패치를 독립적으로 가정하여 공간적 컨텍스트를 잃습니다. 이에 대해 CARMIL이라는 정규화 기법을 통해 공간적 지식을 결합하고, 새로운 정량적 평가 지표인 DeltaCon을 도입하여 모델의 Context-Awareness를 측정합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 사용성이 높음을 증명했으며, 전문가가 라벨링한 MCQ 품질 측정에서도 예측력이 강함을 보여주었습니다.'}, {'NLP Task Robustness': '대조 학습과 counterfactual augmentation을 통해 다양한 차원에서 큰 성능 향상, 특히 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터로부터의 일반화 성능이 개선되었습니다.'}, {'Context-Aware Regularization': 'Glioblastoma와 대장암 데이터셋에 대한 생존 분석에서 CARMIL이 원래 context-independent한 MIL 모델들을 Context-Aware 모델로 전환하고 C-index를 향상시키는 것을 입증했습니다.'}]



### Towards Evolutionary-based Automated Machine Learning for Small Molecule Pharmacokinetic Prediction (https://arxiv.org/abs/2408.00421)
Comments:
          Paper accepted and presented at the 14th Workshop on Evolutionary Computation for the Automated Design of Algorithms (ECADA), which happened during the Genetic and Evolutionary Computation Conference (GECCO)

- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 최근 NLP 태스크에서 대비적 학습(contrastive learning)과 반사실적 증대(counterfactual augmentation)를 사용하여 모델의 robustness를 개선하는 연구를 수행했습니다. 또한, 소분자 약물 연구에서 진화 기반 AutoML 방법을 제안하여 개인 맞춤형 ML 파이프라인을 자동으로 생성합니다.

- **Technical Details**: KDA는 학생의 대상 사실에 대한 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. NLP 태스크에서는 대조 학습과 반사실적 증대를 통해 각 용어의 인과 관계를 robust하게 감독합니다. AutoML 방법에서는 문법 기반 유전 프로그래밍(grammar-based genetic programming)을 사용하여 입력된 분자 데이터의 특성에 맞게 최적의 예측 파이프라인을 자동으로 설계합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서 전문가의 평가와 강한 상관관계를 가집니다. 제안된 NLP 접근 방식은 반사실적 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희소 데이터에 대한 일반화에서 큰 개선을 보였습니다. AutoML 방법은 소분자 약물 연구에서 다양한 ML 알고리즘 선택에 효율적이며, 기존 방법론과 비교해 유사하거나 향상된 예측 성능을 보였습니다.



### MPT-PAR:Mix-Parameters Transformer for Panoramic Activity Recognition (https://arxiv.org/abs/2408.00420)
- **What's New**: 새로운 자동 평가 메트릭, Knowledge Dependent Answerability (KDA),를 제안하여 지문 기반의 MCQ 생성 평가 방법을 개선했습니다. 이 메트릭은 학생의 대상 사실(target fact)에 대한 지식을 기반으로 MCQ의 대답 가능성(answerability)을 평가합니다. 또한, KDA를 측정할 수 있는 두 가지 자동화된 평가 메트릭, KDA_disc와 KDA_cont를 도입했습니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 한 human survey를 통해 측정됩니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델(pre-trained language models)을 사용하여 학생들의 문제 해결 행동을 모방합니다. 이를 통해 학습 세트에서 대답 가능성을 자동으로 평가할 수 있습니다. 또한, 기존의 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링 한 MCQ 품질 측정치를 예측하는 데 강력한 예측력을 가집니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 (1) KDA와의 강한 상관관계, (2) 실제 강의 세트에서 전문가가 라벨링한 사용성과의 강한 상관관계를 보였습니다. 이 메트릭들은 MCQ의 대답 가능성과 교육적 유효성을 평가하는 데 유용하게 사용될 수 있습니다.



### DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving (https://arxiv.org/abs/2408.00415)
Comments:
          19 pages, 9 figures

- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 도입했습니다. 또한, 최근 NLP 태스크에서의 deep 모델의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 적용한 연구가 발표되었습니다. 마침내, DriveArena라는 고충실도 폐쇄 루프 시뮬레이션 시스템이 자율 주행 에이전트를 위한 새로운 플랫폼으로 소개되었습니다.

- **Technical Details**: MCQ 생성 평가를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였습니다. 이들은 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. NLP 모델의 robustness를 위해 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 용어의 인과 관계에 대해 robust하게 감독하는 방법을 제안했습니다. DriveArena는 Traffic Manager와 World Dreamer로 구성된 모듈형 설계를 채택하였으며, 다양한 도시의 로드맵을 바탕으로 현실적인 교통 흐름을 생성할 수 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가들이 평가한 실제 강의실에서의 사용성과 강한 상관관계를 나타냈으며, MCQ 품질 측정의 예측력도 높였습니다. DriveArena는 기존 방식보다 덜 감각적으로 편향된 결과를 보여주며 다양하고 복잡한 시나리오에서 자율 주행 에이전트의 학습과 진화를 가능하게 합니다.



### Enhancing Whole Slide Pathology Foundation Models through Stain Normalization (https://arxiv.org/abs/2408.00380)
Comments:
          13 pages, 8 figures

- **What's New**: 자동으로 Multiple Choice Questions(MCQ)을 생성하는 새로운 평가 메트릭을 도입했습니다. 지식 종속 가능성(KDA)이라고 불리는 이 메트릭은 학생들이 목표 사실에 대한 지식을 평가할 수 있는 MCQ의 대답 가능성을 측정합니다. 또한, 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안하여 사람의 응답을 통해 KDA를 근사합니다.

- **Technical Details**: 제안된 KDA는 기존 평가 메트릭(BLEU, ROUGE, METEOR)이 놓치고 있는 교육적 가치를 반영합니다. KDA 기반 메트릭은 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모사합니다. 또한, Human study를 통해 이러한 메트릭이 실제 교육 현장에서 얼마나 유용한지 검증됐습니다.

- **Performance Highlights**: Human study 결과, KDA_disc와 KDA_cont는 KDA 및 전문가가 지정한 실제 교실에서의 사용성에 대한 강한 상관관계를 보였습니다. KDA_disc와 KDA_cont를 n-gram 기반 유사성 메트릭과 결합할 때, 이는 전문가가 지정한 다양한 MCQ 품질 측정항목에 대한 예측력을 높이는 데 큰 기여를 했습니다.



### On the Limitations and Prospects of Machine Unlearning for Generative AI (https://arxiv.org/abs/2408.00376)
- **What's New**: 이 논문에서는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하여 자동으로 생성된 객관식 질문(MCQ)의 교육적 가치를 평가합니다. 또한, 최근 심층 모델이 NLP 작업 효율성에서 '사람을 능가하는' 정확성을 자랑함에도 불구하고, 무작위 패턴에 의존하여 견고성이 제한된 문제를 다룹니다. 마지막으로, 생성 AI(GenAI)의 개인정보 보호, 보안 및 윤리적 도전 과제를 해결하기 위해 기계 학습의 '잊혀질 권리'와 관련된 접근 방안을 논의합니다.

- **Technical Details**: KDA는 학생이 특정 사실을 알고 있을 때 MCQ의 대답 가능성을 측정하는 새로운 메트릭입니다. 본 메트릭은 인간 설문을 통한 학생 반응을 기반으로 계산되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모사합니다. 또한, 심층 모델의 견고성을 향상시키기 위해 대조 학습과 반사실 증강(contrastive learning and counterfactual augmentation)을 활용하는 새로운 방법을 제안합니다. 이 방법은 여러 개의 반사실(counterfactual)을 생성하고 집합적 의사 결정을 통해 각 용어의 인과관계를 견고하게 감독합니다. 생성 AI의 맥락에서 기계 학습 방법을 사용하여 민감한 데이터를 삭제하는 기법도 심도 있게 논의하고 있습니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont 메트릭은 인간 설문 결과와 강한 상관관계를 나타내며, MCQ의 실제 교육적 가치를 평가하는 데 유용함을 보여줍니다. 대조 학습 및 반사실 증강을 통해 제안된 방법은 반사실 강건성(counterfactual robustness), 크로스 도메인 일반화(cross-domain generalization), 드문 데이터 일반화(generalization from scarce data)에서 상당한 개선을 이루었습니다. 기계 학습의 맥락에서, 민감한 데이터를 효과적으로 삭제하는 기술은 LLMs와 이미지 생성 모델에서 현저한 잠재력을 보여주었습니다.



### DiM-Gesture: Co-Speech Gesture Generation with Adaptive Layer Normalization Mamba-2 framework (https://arxiv.org/abs/2408.00370)
Comments:
          10 pages,10 figures. arXiv admin note: text overlap with arXiv:2403.10805

- **What's New**: 이번 아카이브 논문에서는 교육자들이 학생 평가에 소요하는 시간을 획기적으로 줄일 수 있는 자동 다중 선택 질문(MCQ) 생성 기술에 대한 연구가 다뤄졌습니다. 기존의 BLEU, ROUGE, METEOR 평가 메트릭은 생성된 MCQ와 데이터셋의 샘플 간의 n-gram 유사성만을 평가하며 교육적 가치를 간과했지만, 새로운 평가 메트릭 Knowledge Dependent Answerability (KDA)를 제안하여 MCQ의 대답 가능성과 교육적 유용성을 평가합니다. 또한, 최근 NLP 태스크에서의 강력한 딥 모델이 존재하지만, 이 논문은 spurious pattern에 의존하여 robustness가 제한된다고 보고된 문제를 addressing하기 위해 대조 학습과 counterfactual augmentation을 활용하는 방법을 제안합니다. 마지막으로, 음성 기반 제스처 생성 기술에서 Transformer 기반 아키텍처의 한계를 극복하기 위해 Mamba 기반 아키텍처를 사용한 새로운 모델 DiM-Gestures를 제안하는 논문도 게재되었습니다.

- **Technical Details**: {'MCQ 평가 메트릭': '새로운 평가 메트릭 지식 종속 가능성(KDA)은 학생의 대상 사실에 대한 지식을 평가합니다. 자동 평가 메트릭 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Robustness 향상': '대조 학습과 counterfactual augmentation을 통해 단일 counterfactual이 아닌 여러 개의 counterfactual을 생성하고, 집합적 의사결정을 통해 각 용어의 인과관계를 robust하게 감독합니다.', 'DiM-Gestures 모델': '이 모델은 Mamba 기반 퍼지 특징 추출기와 비자기회귀적(Non-Autoregressive) 적응 레이어 정규화(AdaLN) Mamba-2 확산 아키텍처를 통합하여 음성 만으로 개인화된 3D 전신 제스처를 생성합니다. 훈련과 추론에 확산 모델(applying a diffusion model)을 활용하여, ZEGGS와 BEAT 데이터셋에서 평가되었습니다.'}

- **Performance Highlights**: {'KDA 메트릭': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성 및 전문가가 레이블한 MCQ 품질 측정치와 강한 상관관계를 가집니다. n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가 레이블 MCQ 품질 측정치에 대해 강력한 예측력을 가집니다.', 'Robustness 향상': '제안된 방법은 직관적 대립 학습 모델과 비교해 다양한 측면에서 큰 개선을 이루었으며, 특히 counterfactual robustness, cross-domain generalization, generalization from scarce data 측면에서 우수한 성과를 보였습니다.', 'DiM-Gestures 모델': '주관적 및 객관적 평가를 통해 이 모델은 최첨단 기법에 비해 우수한 성능을 보였습니다. 적은 메모리 사용량 및 빠른 추론 속도를 유지하면서 제스처-음성 동기화를 고충실도로 보장합니다.'}



### DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training (https://arxiv.org/abs/2408.00355)
Comments:
          Accepted by ACMMM2024

- **What's New**: 이번 연구는 Transformer 기반의 자동 텍스트 스포팅(text spotting) 아키텍처에서 바이어파트 그래프 매칭(bipartite graph matching)에 기인한 불안정성을 해결하기 위해 새로운 디노이즈 트레이닝(denoising training) 방법인 DNTextSpotter를 제안합니다. 이 방법은 임의의 형상 텍스트 스포팅에 적합하며, 기존 방식의 단점을 극복하도록 고안되었습니다.

- **Technical Details**: DNTextSpotter는 비지어 제어점(Bézier control points)과 텍스트 문자를 사용하여 노이즈가 추가된 쿼리를 초기화합니다. 이를 위해 텍스트 내용을 포지션과 맞추기 위해 마스크된 문자 슬라이딩 방법(masked character sliding method)을 적용합니다. 또한, 백그라운드 문자 분류를 위한 추가 손실 함수(loss function)를 사용하여 모델의 인식 능력을 향상시킵니다.

- **Performance Highlights**: 이 새로운 방법은 여러 벤치마크에서 뛰어난 성능을 보였으며, 특히 Inverse-Text 데이터셋에서 11.3%의 성능 향상을 달성했습니다. Total-Text 및 SCUT-CTW1500에서는 각각 2.0% 및 2.1% 향상을 보여주었습니다. ViTAEv2-S 백본을 사용할 때는 모든 메트릭에서 성능이 더욱 향상되었습니다.



### A Simple Background Augmentation Method for Object Detection with Diffusion Mod (https://arxiv.org/abs/2408.00350)
- **What's New**: 새로운 자동 평가 지표인 Knowledge Dependent Answerability (KDA)을 제안하여, MCQ의 대답 가능성(answerability)과 학생의 해당 사실에 대한 지식을 평가할 수 있게 하였습니다. 이 지표는 기존의 평가 메트릭들이 무시했던 교육적 가치를 반영합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정하며, 핵심 사실의 학습 평가 성능을 나타냅니다. 또한, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대한 예측력이 강력함을 입증했습니다.



### Securing the Diagnosis of Medical Imaging: An In-depth Analysis of AI-Resistant Attacks (https://arxiv.org/abs/2408.00348)
- **What's New**: 최근 연구는 다중 선택 질문(MCQ)을 자동으로 생성하는 시스템의 평가 메트릭을 개선하는 방법을 제안합니다. 기존의 평가 메트릭은 교육적 가치를 무시하고 n-gram 유사성에만 집중합니다. 반면, 이 논문은 Knowledge Dependent Answerability (KDA)라는 새로운 평가 지표를 도입하여 MCQ의 대답 가능성 및 교육적 가치를 평가합니다.

- **Technical Details**: KDA는 학생이 문제를 푸는 행동을 모방한 사전 학습된 언어 모델을 사용하여 측정됩니다. 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont가 제안되었으며, 인간 설문조사 데이터를 통해 이들 지표의 유효성을 검증하였습니다. 또한 이 지표들은 기존의 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 측정에 대한 예측력을 강화합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성 및 KDA와 강한 상관관계를 보였습니다. 이 지표들은 기존 평가 메트릭과의 조합을 통해 MCQ 품질을 더 효과적으로 예측할 수 있습니다.



### Advancing Medical Image Segmentation: Morphology-Driven Learning with Diffusion Transformer (https://arxiv.org/abs/2408.00347)
Comments:
          Accepted in BMVC 2024

- **What's New**: 새로운 MCQ 생성 평가 메트릭인 Knowledge Dependent Answerability (KDA) 제안. 이는 MCQ가 학생의 대상 지식(fact)에 대한 이해를 평가하는 능력을 측정합니다.

- **Technical Details**: 기존 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에만 초점을 맞추어 교육적 가치를 반영하지 못하고 있습니다. KDA는 학생 응답을 바탕으로 MCQ의 대답 가능성을 측정하며, 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 해당 메트릭들은 예비 학습된 언어 모델을 활용하여 학생의 문제 해결 습관을 모방합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 KDA 및 실제 교실 환경에서의 사용성과 높은 상관관계를 보였습니다. n-gram 기반 유사성 메트릭과 결합될 때 다양한 전문가가 평가한 MCQ의 품질 측정에 대해 강력한 예측력을 나타냈습니다.



### Neural Graph Matching for Video Retrieval in Large-Scale Video-driven E-commerc (https://arxiv.org/abs/2408.00346)
- **What's New**: 이 논문은 MCQ (Multiple Choice Questions) 생성의 새로운 자동 평가 메트릭을 소개합니다. 기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반의 유사성만을 평가하는 데 반해, 제안된 지식 종속 가능성(KDA) 메트릭은 학생들의 지식을 평가할 수 있는 능력을 측정합니다.

- **Technical Details**: KDA 메트릭은 학생 설문조사를 통해 대답 가능성(answerability)을 측정한 다음, 사전 훈련된 언어 모델(pre-trained language models)을 활용해 학생들의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다. 이 방법은 KDA와의 강한 상관관계를 가지며 실제 강의실 세트에서의 사용성 또한 높입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 전문가 라벨링된 MCQ 품질 측정에서 높은 예측력을 보이며, n-gram 기반 유사성 메트릭과 결합할 때 특출난 성능 향상을 보였습니다.



### MuJoCo MPC for Humanoid Control: Evaluation on HumanoidBench (https://arxiv.org/abs/2408.00342)
Comments:
          3 pages, 3 figures, submitted to IEEE Conference on Robotics and Automation (ICRA@40)

- **What's New**: 자동 다중선택 질문 (MCQ) 생성의 잠재적 가치를 최대화하기 위해 새로운 평가 지표인 지식 종속 대답 가능성 (Knowledge Dependent Answerability, KDA)를 제안했습니다. 이는 MCQ의 대답 가능성을 측정하며 학생들이 목표 사실을 잘 이해했는지를 평가하는 능력을 고려합니다.

- **Technical Details**: KDA는 인간 설문조사 결과를 토대로 측정되며, 이를 모방하기 위해 사전 학습된 언어 모델을 사용하는 두 가지 자동 평가 지표 KDA_disc와 KDA_cont를 제안했습니다. KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 높은 상관관계를 보였습니다.

- **Performance Highlights**: 사람들이 라벨링한 전문가 기준 MCQ 품질 평가 측정치와 결합하여 KDA_disc와 KDA_cont는 강력한 예측력을 보여줬습니다.



### OTAD: An Optimal Transport-Induced Robust Model for Agnostic Adversarial Attack (https://arxiv.org/abs/2408.00329)
Comments:
          14 pages, 2 figures

- **What's New**: 최근 딥러닝 모델들이 사람의 능력을 넘어서는 정확성을 보이지만, 허위 패턴에 의존하여 강건성(robustness)이 제한되는 문제를 해결하고자 대조학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다. 또한, 딥러닝 모델의 입력에 대한 작은 적대적 교란(adversarial perturbations)으로 인해 발생하는 취약성을 해결하기 위해, '최적 운송 유도 적대적 방어(OTAD)' 모델을 도입합니다.

- **Technical Details**: 1. 교육 분야에서 자동으로 다중 선택 질문(MCQ)을 생성하는 기존의 평가 메트릭은 교육적 가치를 고려하지 않으며, 이를 해결하기 위해 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안하였습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 MCQ의 답변 가능성을 측정합니다.

2. 대조학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 통해 모델의 강건성을 강화했습니다. 기존 방법과 달리 여러 개의 반사실적 데이터를 생성하여 보다 강건하게 단어들의 인과관계를 파악합니다. 

3. 최적 운송 이론(optimal transport theory)에 기반한 OTAD 모델을 통해 딥러닝의 강건성을 강화하고, 전송 맵을 활용하여 로컬 립시츠 연속성(local Lipschitz continuity)을 보장합니다.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont는 실제 학급에서 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였습니다.

2. 대조학습과 반사실적 증강을 결합한 새로운 방법은 반사실적 강건성(counterfactual robustness)과 크로스 도메인 일반화(cross-domain generalization)에서 상당한 개선을 달성했습니다.

3. OTAD 모델은 다양한 데이터셋에서 기존의 적대적 훈련 방법들과 립시츠 네트워크(Lipschitz networks)를 뛰어넘는 성능을 보여주었습니다.



### ADBM: Adversarial diffusion bridge model for reliable adversarial purification (https://arxiv.org/abs/2408.00315)
Comments:
          20 pages

- **What's New**: MCQ 자동 생성을 위한 새로운 평가 메트릭으로 KDA(Knowledge Dependent Answerability)를 제안합니다. 기존 평가 메트릭들은 n-gram 유사성에만 치중해 교육적 가치를 평가하지 못하였으나, KDA는 MCQ의 대답 가능성을 통해 학생의 지식을 평가합니다.

- **Technical Details**: KDA는 학생의 지식에 따라 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 자동 평가 지표를 제안하여, 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 높은 상관관계를 가짐을 확인하였습니다. n-gram 기반 유사성 메트릭과 결합할 경우, 다양한 전문가 평가 MCQ 품질 지표에 대해 높은 예측력을 보였습니다.



### Discretizing Continuous Action Space with Unimodal Probability Distributions for On-Policy Reinforcement Learning (https://arxiv.org/abs/2408.00309)
Comments:
          IEEE Transactions on Neural Networks and Learning Systems

- **What's New**: 이 논문들의 주요 내용은 각각 자동 MCQ 생성, NLP의 robust 학습 및 순방향 강화 학습을 다루고 있습니다. 특히 자동 Multiple Choice Questions (MCQ)의 평가를 위해 새로운 메트릭인 Knowledge Dependent Answerability (KDA)가 제안되었습니다. 두 번째 논문에서는 contrastive learning과 counterfactual 확장을 통해 NLP 모델의 robustness를 개선하려는 시도를 설명합니다. 세 번째 논문은 연속적인 action space를 위한 discretized 정책에서 발생하는 문제를 해결하기 위해 Poisson 분포를 이용한 unimodal discrete policy를 소개합니다.

- **Technical Details**: 자동 MCQ 생성에 대해서는, BLEU, ROUGE, METEOR 같은 기존 메트릭이 MCQ의 교육적 가치를 판단하는 데 한계가 있으므로, 새로운 KDA 메트릭이 제안되었습니다. 이는 학생들이 해당 사실에 대해 얼마나 잘 대답할 수 있는지를 측정합니다. NLP의 풍부성을 개선하기 위해서는 여러 counterfactual을 생성하고 이들의 분포에 따른 집단적 의사결정을 통해 인과관계를 더 robust하게 감독하는 접근 방식이 제안되었습니다. 강화 학습에서는, 연속적인 행동 공간에 대한 정책의 variance를 감소시키기 위해 Poisson 확률 분포를 사용하는 unimodal policy가 제안되었습니다.

- **Performance Highlights**: 첫 번째 논문에서 제안된 KDA_disc와 KDA_cont 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 human study를 통해 확인하였습니다. 두 번째 논문에서는 새롭게 제안된 방법이 다양한 차원에서 기존 접근 방식보다 더 높은 Robustness를 보여주었습니다. 마지막으로, Poisson 분포 기반의 unimodal discrete policy는 MuJoCo 연속 제어 과제들에서 기존 방법들 보다 빠른 수렴과 높은 학습 성능을 보여주었습니다.



### ABC Align: Large Language Model Alignment for Safety & Accuracy (https://arxiv.org/abs/2408.00307)
Comments:
          23 pages, 4 figures

- **What's New**: 이번 주 Arxiv 논문에서는 대규모 언어 모델(LLMs)의 맞춤화에 대한 새로운 방법론인 'ABC Align'을 소개했습니다. ABC Align은 LLMs에 대한 인공지능 원칙과 조직의 선호도를 통합하는 새로운 정렬 방법론입니다. 이를 통해 모델의 편향을 줄이고 정확성을 높이며, 표준 벤치마크 대비 성능을 개선합니다.

- **Technical Details**: ABC Align은 기존의 모델 미세조정(fine-tuning) 및 내부 문맥 학습(In-Context Learning, ICL) 기술을 결합하여 수행됩니다. 최근의 합성 데이터 생성(synthetic data generation), 선호 최적화(preference optimization), 그리고 훈련 후 모델 양자화(post-training model quantisation) 기술을 활용하여 다수의 오픈 소스 모델과 폐쇄 소스 모델에서 성능 상승을 보여줍니다. 다양한 데이터 세트—뉴스 기사, AI 원칙(ABC AI Principles), 그리고 내부 검색-강화 생성 도구(RAG)에서 제공된 Q&A 쌍—를 사용해 데이터 세트를 생성하고 모델을 정렬합니다.

- **Performance Highlights**: 대표적인 성과는 Meta의 Llama3-8B 모델에서 TruthfulQA 벤치마크에서 23.51%의 상대적 성능 향상을 보였다는 것입니다. 또한 OpenAI의 GPT4-turbo 모델에서는 Bias Benchmark for Question Answering (BBQ)에서 기본 라인 대비 성능이 77.54% 향상되었습니다. 이는 소규모 데이터 세트(897 샘플)을 사용하여 달성한 결과로, 메타에서 자체적으로 조정한 버전보다 훨씬 나은 성능을 보였습니다.



### Contrastive Graph Representation Learning with Adversarial Cross-view Reconstruction and Information Bottleneck (https://arxiv.org/abs/2408.00295)
Comments:
          13 pages, 7 figures

- **What's New**: 제안된 논문은 여러 분야에서의 최신 연구를 소개하며, 각 영역에 걸쳐 의미 있는 진전을 보여줍니다. 여기에서는 새로운 자동 평가 메트릭, 지식 종속 가능성(KDA), 강화된 노드 특성 표현 기법, 그리고 가상적 증강을 활용한 모델의 견고성 향상을 다룹니다.

- **Technical Details**: {'MCQ 자동 생성': ['기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 고려하지 않음', '새로운 평가 메트릭 지식 종속 가능성(KDA)을 제안하여 MCQ의 대답 가능성(answerability)을 측정', '두 가지 자동 평가 메트릭: KDA_disc와 KDA_cont 제안', '사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방'], '대조 학습 및 반사실 증강': ['현재 모델들이 부정적인 패턴(spurious patterns)에 의존하는 문제를 보완', '대조 학습과 반사실 증강을 사용하여 모델의 견고성(robustness)을 향상', '집합적 의사 결정을 통해 여러 counterfactual 생성 및 예측의 분포 평가'], '그래프 신경망(GNN)': ['대중성 편향(popularity bias)과 노이즈 간섭 문제 해결', '적응적 자동 노드 마스킹 및 엣지 변조를 통한 그래프 증강 기법', '정보 병목 이론(Information Bottleneck Theory) 도입', '여러 보완적 시점의 공유 특징 공간 통합으로 특징 표현의 견고성 향상']}

- **Performance Highlights**: {'MCQ 생성 성능': 'KDA_disc와 KDA_cont가 실제 강의실 세팅에서 높은 사용성 평가', '대조 학습 성능': 'Task 모델 바이어스에 덜 민감하며, 다양한 차원에서 성능 향상', '그래프 신경망 성능': '7개 실세계 공개 데이터셋에서 최첨단 알고리즘보다 우수한 성능'}



### Multi-Modal Parameter-Efficient Fine-tuning via Graph Neural Network (https://arxiv.org/abs/2408.00290)
- **What's New**: 최근의 딥 모델들이 NLP 태스크에서 뛰어난 성능을 보였지만, 자주 쓰이는 패턴에 의존하여 robustness가 제한되는 문제가 있습니다. 이를 해결하기 위해, 우리는 대조학습 (contrastive learning)과 반사실적 증강 (counterfactual augmentation) 방법을 제안합니다. 또한, 교육적 가치를 평가하기 어려운 기존 MCQ 생성 메트릭 문제를 해결하려는 새로운 메트릭, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)을 제안합니다.

- **Technical Details**: 제안된 MCQ 메트릭, KDA는 학생의 반응을 기반으로 MCQ의 대답 가능성을 측정하며, 이는 미리 학습된 언어 모델을 이용해 학생들이 문제를 풀어가는 과정을 모방합니다. 또한 반사실적 증강 기존 방법들과 달리, 다양한 반사실적 세트를 만들어 통합적인 결정을 내리고, 각 용어의 인과성을 보다 더 robust하게 감독하는 방법을 제시합니다.

- **Performance Highlights**: Human studies를 통해 우리가 제안한 KDA_disc와 KDA_cont는 실제 강의실 사용성과 강하게 상관성이 있음을 밝혔습니다. 또한 대조학습과 반사실적 증강 방법이 기존 모델들보다 다양한 차원에서 상당한 성능 향상을 보여주었으며, 특히 여러 데이터셋에서 4.45%, 2.92%, 0.23%의 정확도 향상을 보였습니다.



### Gradient Harmonization in Unsupervised Domain Adaptation (https://arxiv.org/abs/2408.00288)
Comments:
          IEEE TPAMI 2024

- **What's New**: 이번 논문에서는 자동 MCQ (Multiple Choice Questions) 생성의 평가 메트릭이 기존의 BLEU, ROUGE, METEOR와 같이 n-gram 기반의 유사성을 평가하는 방식에서 벗어나 학생의 지식을 평가할 수 있는 능력을 고려하지 않는 문제를 해결하고자 합니다. 새로운 자동 평가 메트릭인 지식 종속 가능성 (KDA, Knowledge Dependent Answerability)을 제안하여, MCQ가 주어진 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: KDA는 학생의 응답 데이터를 기반으로 MCQ의 대답 가능성을 평가합니다. KDA를 근사하기 위해 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안했습니다.

- **Performance Highlights**: Human evaluation을 통해, KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성 및 KDA 자체와 강한 상관관계를 가지고 있음을 발견했습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정치를 예측하는 능력이 뛰어남을 확인했습니다.



### High Performance Im2win and Direct Convolutions using Three Tensor Layouts on SIMD Architectures (https://arxiv.org/abs/2408.00278)
- **What’s New**: 다양한 최신 연구들은 기존의 NLP 및 딥러닝 모델의 한계를 극복하고 성능을 향상시키기 위한 새로운 접근법과 메트릭을 소개합니다.

- **Technical Details**: [{'Title': '자동 MCQ 생성 평가를 위한 새로운 메트릭 KDA 제안', 'Details': "기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에 집중하고 교육적 가치를 무시합니다. 이를 해결하기 위해 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안, 학생의 대상 사실에 대한 지식 평가 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하고, 전문가 라벨링 품질 측정치에 대해 높은 예측력을 보여줍니다."}, {'Title': '대조 학습과 반사실 증가를 통한 로버스트니스 향상', 'Details': '최근의 딥 모델들은 NLP 작업에서 사람보다 높은 정확성을 보이지만, spurious pattern에 의존하여 robustness가 제한됩니다. 우리는 대조 학습과 반사실 데이터 증가(Augmentation)를 활용하여 이 문제를 해결합니다. 기존 방법들은 반사실 데이터를 통해 방법론을 보완하려 하지만, 우리는 여러 개의 반사실을 생성하여 집합적 의사 결정(collective decisions) 방식을 통해 spurious correlation의 영향을 최소화하여 성능을 크게 개선합니다.'}, {'Title': '심층 신경망에서의 convolution 연산 성능 최적화', 'Details': 'Convolution은 심층 신경망의 핵심 구성 요소로, 계산 집약적이며 시간이 많이 소요됩니다. 이 논문은 im2win convolution을 위한 NHWC, CHWN, CHWN8 세 가지 새 데이터 레이아웃을 제안하고 최적화된 기술을 도입합니다. 실험 결과 새로운 NHWC 레이아웃은 NCHW 레이아웃 대비 최대 355% 성능 향상을 달성하였으며 제안된 최적화 기술이 im2win과 direct convolution 모두에서 높은 성능을 보였습니다. 이 논문은 SIMD 아키텍처에서 이들 최적화된 레이아웃의 성능을 밝혔습니다.'}]

- **Performance Highlights**: [{'Metric': 'KDA_disc와 KDA_cont', 'Details': '전문가 라벨링 품질 측정치와 강한 상관 관계를 가지며, n-gram 기반 유사성 메트릭과 결합 시 예측력이 크게 향상되었습니다.'}, {'Metric': 'Collective Decision-based Counterfactual Robustness', 'Details': '다양한 차원에서 중요한 개선을 달성했으며, 대조 학습을 통해 반사실 데이터의 활발한 역할을 보였습니다.'}, {'Metric': 'im2win Convolution Performance', 'Details': 'NHWC 레이아웃에서 최대 355% 성능 향상, 최적화된 im2win과 direct convolution이 각각 95%와 94%의 이론적 최대 성능에 도달.'}]



### QUITO: Accelerating Long-Context Reasoning through Query-Guided Context Compression (https://arxiv.org/abs/2408.00274)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 추론 복잡성과 계산 비용을 크게 줄여주는 새로운 Query-gUIded aTtention cOmpression (QUITO) 방법을 소개합니다. QUITO는 질문에 대한 컨텍스트의 중요도를 평가하여 불필요한 정보를 걸러내고, 컨텍스트 길이 제한을 충족시키는 세 가지 필터링 방법을 제안합니다.

- **Technical Details**: QUITO는 self-attention 메커니즘을 사용하여 토큰의 중요도를 평가합니다. 이 방법은 상대적으로 작은 크기의 LLM (0.5 billion parameters)을 활용하여 기존의 대형 모델(7 billion 또는 13 billion parameters)보다 효율적으로 작동합니다. QUITO는 두 개의 대표적인 데이터셋(NaturalQuestions, ASQA)에서 평가되었으며, 다양한 데이터셋과 downstream LLM에서 뛰어난 성능을 입증했습니다.

- **Performance Highlights**: 실험 결과, QUITO는 기존의 강력한 기준선 모델을 20.2%의 정확도 향상으로 능가했습니다. 이는 QUITO가 LLM의 계산 효율성과 정확도를 동시에 개선함을 보여줍니다.



### Clover-2: Accurate Inference for Regressive Lightweight Speculative Decoding (https://arxiv.org/abs/2408.00264)
- **What's New**: 다양한 NLP 연구에서 새로운 자동 평가 메트릭과 반사실적 데이터 증강 및 강력한 텍스트 생성 모델에 대한 최신 발전이 제안되었습니다.

- **Technical Details**: [{'Title': 'MCQ 자동 생성 평가를 위한 지식 종속 가능성(KDA)', 'Content': '기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 평가 메트릭의 한계를 극복하기 위해, 새로운 평가 메트릭인 KDA를 제안. KDA는 MCQ의 대답 가능성을 측정하며, 대상 사실에 대한 학생의 지식 평가 능력을 분석.'}, {'Title': 'NLP 태스크의 강건성을 위한 반사실적 학습', 'Content': '최근 deep model들이 spurious pattern에 의존하는 문제 때문에 robustness가 제한됨. 이 논문에서는 대조 학습과 반사실적 데이터 증강을 통해 모델의 강건성을 높이는 방법을 제안. 다양한 태스크와 데이터 도메인에서 성능 향상 확인.'}, {'Title': '효율적인 텍스트 생성 모델, Clover-2', 'Content': 'Clover의 업그레이드 버전인 Clover-2는 RNN 기반 초안 모델의 성능을 높이기 위해 지식 증류(knowledge distillation)와 개선된 모델 구조를 사용. Vicuan 7B와 LLaMA3-Instruct 8B 모델 테스트에서 기존 방법들보다 큰 성능 향상을 보임.'}]

- **Performance Highlights**: [{'Title': 'MCQ 자동 생성의 KDA 혁신', 'Content': 'human evaluation에서 KDA_disc와 KDA_cont가 높은 상관관계를 나타내고, n-gram 기반 메트릭과 결합 시 더 강력한 예측 성능을 가짐.'}, {'Title': '강건한 NLP 모델', 'Content': '다양한 데이터 도메인에서 counterfactual을 통해 유의미한 성능 향상; 1) 반사실적 강건성, 2) 크로스 도메인 일반화, 3) 데이터 부족 상황에서의 일반화.'}, {'Title': 'Clover-2의 뛰어난 성능', 'Content': 'Vicuan 7B와 LLaMA3-Instruct 8B 실험에서 standard decoding 대비 최대 3.00배의 처리 속도 향상. EAGLE보다 최대 9.3% 빠른 속도로 speculative token 생성, 특히 RNN 구조를 유지하면서도 효율성 증대.'}]



### A Prior Embedding-Driven Architecture for Long Distance Blind Iris Recognition (https://arxiv.org/abs/2408.00210)
- **What's New**: 이 논문은 장거리에서의 눈동자 인식 (iris recognition) 시, 미지의 손상으로 인해 발생하는 '블라인드 아이리스 이미지' 문제를 해결하기 위한 새로운 접근법을 제안합니다. 구체적으로, 블라인드 아이리스 이미지를 효과적으로 복원하기 위한 Iris-PPRGAN 네트워크와 더 나은 아이리스 특징 추출을 위한 Insight-Iris 분류기를 소개합니다.

- **Technical Details**: Iris-PPRGAN 네트워크는 생성적 적대 신경망 (GAN)을 Prior Decoder로 사용하고, 심층신경망 (DNN)을 인코더로 사용하여 블라인드 아이리스의 텍스처를 복원합니다. 또한 InsightFace의 병목 모듈을 수정하여 Insight-Iris 분류기를 개발하였고, 이는 복원된 아이리스 이미지를 인식하는 데 사용됩니다.

- **Performance Highlights**: 공개된 CASIA-Iris-distance 데이터셋에서, 제안된 방법은 정성적 및 정량적으로 최첨단 블라인드 아이리스 복원 방법보다 우수한 성능을 보였습니다. 특히, 복원 후의 장거리 블라인드 아이리스 이미지의 인식률이 90%에 달했으며, 이는 복원 전 이미지와 비교하여 약 10% 포인트가 개선된 결과입니다.



### OmniParser for Pure Vision Based GUI Agen (https://arxiv.org/abs/2408.00203)
- **What's New**: 이 논문에서는 자동 MCQ(객관식 질문) 생성의 교육적 가치를 평가하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)을 제안합니다. 또한, GPT-4V와 같은 대형 비전-언어 모델이 다양한 사용자 인터페이스에서 동작하는 것을 돕기 위해 OmniParser를 도입하여 화면 상의 상호작용 가능한 요소와 그 기능적 의미를 보다 정확하게 이해하도록 합니다.

- **Technical Details**: [{'MCQ Generation': '기존 MCQ 생성의 평가 메트릭들은 BLEU, ROUGE, METEOR처럼 n-gram 기반의 유사성에만 집중되어 있었으나, 교육적 가치를 평가하지 않았습니다. 이를 해결하기 위해, KDA라는 새로운 메트릭을 통해 학생이 해당 정보를 알고 있을 때 MCQ가 답할 수 있는지 여부를 평가합니다. KDA는 Human survey를 기반으로 측정되며, KDA_disc와 KDA_cont라는 자동화된 평가 메트릭을 제안합니다.'}, {'Model Robustness': '최근의 deep model들이 NLP 태스크에서 높은 정확성을 보였으나, spurious pattern에 의존하여 robustness가 제한되었습니다. 이 논문에서는 contrastive learning과 counterfactual augmentation을 활용하여 단어들의 인과관계를 판단하는 방법을 제안합니다. 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 이전의 spurious correlation 문제를 극복합니다.'}, {'Screen Parsing': 'OmniParser는 사용자의 인터페이스 스크린샷을 파싱하여 구조화된 요소들로 변환하는 종합적인 방법입니다. 이를 통해 GPT-4V가 다양한 애플리케이션에서 예측한 행동을 정확한 화면 영역에 결합할 수 있습니다. 이 시스템은 상호작용 가능한 요소를 탐지하고 기능적 의미를 추출하기 위해 특별히 학습된 모델을 활용합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 함께 사용할 때, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였습니다.'}, {'Model Robustness': '제안된 방법은 단어들의 인과관계 파악에서 bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, and generalization from scarce data에서 놀라운 성능 향상을 보였습니다.'}, {'Screen Parsing': 'OmniParser는 ScreenSpot, Mind2Web, AITW 벤치마크에서 기존 GPT-4V 기반 모델을 능가했습니다. 특히 추가 정보 없이 스크린샷만으로도 뛰어난 성능을 보여주었습니다.'}]



### Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models (https://arxiv.org/abs/2408.00197)
- **What's New**: 이번 연구에서는 자동 객관식 문제(MCQ) 생성을 위한 새로운 평가 메트릭, KDA(Knowledge Dependent Answerability)를 제안했습니다. 이는 기존 BLEU, ROUGE, METEOR 등의 n-gram 기반 메트릭이 학습 평가의 교육적 가치를 고려하지 못하는 문제를 해결하기 위한 것입니다.

- **Technical Details**: KDA는 목표 사실에 대한 학생의 지식을 바탕으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 학생 응답 기반의 KDA 측정 방법을 먼저 제안하고, 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안했습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 (1) 실제 강의실 환경에서의 사용성, (2) 전문가가 라벨링한 MCQ 품질 측정치와 강한 상관관계가 있음을 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합하여 KDA_disc와 KDA_cont는 다양한 전문가 라벨링된 MCQ 품질 측정치에 대한 예측력이 뛰어났습니다.



### Resilience and Security of Deep Neural Networks Against Intentional and Unintentional Perturbations: Survey and Research Challenges (https://arxiv.org/abs/2408.00193)
- **What's New**: 최근 여러 논문에서 다루어지는 주제를 중심으로, deep neural network(DNN)와 관련된 세 가지 새로운 연구를 요약했습니다. 첫 번째 연구는 자동 MCQ 생성의 새로운 평가 메트릭을 제안하며, 두 번째 연구는 대조 학습과 counterfactual augmentation을 활용한 robustness를 강화하는 방법을 다룹니다. 세 번째 연구는 의도적 및 비의도적 외부 교란에 대한 DNN의 회복력에 대한 포괄적인 설문을 제공합니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭은 단어의 유사성만을 평가하므로 교육적 가치를 반영하지 못함. 이를 해결하기 위해 지식 종속 가능성(KDA)를 이용하여 MCQ의 대답 가능성(answerability)을 측정하며, 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성과 강한 상관관계를 가지는 것을 입증함.', 'Robustness Enhancement': 'spurious 패턴에 의존하여 제한된 robustness를 보이는 deep model들을 대상으로, 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하여 robustness를 증대시키는 방법 제안. 기존 방법들과 달리 여러 개의 counterfactual을 생성해 집합적 의사 결정(collective decisions)을 통해 단어들의 인과 관계 파악.', 'DNN Resilience Survey': '의도적 및 비의도적 외부 교란에 대한 DNN의 회복력과 보안성을 연구, 이 두 가지 교란 방식의 연관성을 탐구하며, 포괄적인 설문을 통해 다양한 공격 시나리오를 분류하고, 이들에 대응하는 방어 전략의 강점과 약점을 평가함.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont가 실제 전문가 평가와 강하게 연관되어 있으며, n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가가 평가한 MCQ 품질 지표에 대한 예측력이 강해짐.', 'Robustness Enhancement': '제안된 방법이 대조적 결정(collective decisions)을 통해 task model bias에 덜 민감하게 대응하며, counterfactual robustness, cross-domain generalization, 희소 데이터 일반화 등 다양한 부분에서 성능 개선됨.', 'DNN Resilience Survey': '의도적 교란(예: 악의적 기계 학습 공격) 및 비의도적 교란(예: OOD 샘플)에 대한 DNN 회복력 연구를 통합하여 이 두 가지 교란 유형의 상관성을 발견 및 새로운 연구 방향 제시.'}



### S-SYNTH: Knowledge-Based, Synthetic Generation of Skin Images (https://arxiv.org/abs/2408.00191)
Comments:
          Accepted to the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024

- **What's New**: 최근 문서에서 다룬 여러 혁신적인 연구를 요약했습니다. 첫 번째 연구는 자동 MCQ(Multiple Choice Questions) 생성의 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 두 번째 연구는 NLP(자연어 처리) 모델의 강인성(robustness)을 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증가(counterfactual augmentation)를 활용하는 방법에 대해 다룹니다. 마지막으로, S-SYNTH라는 피부 시뮬레이션 프레임워크를 소개하여 피부 병변(segmenting) 세분화 성능을 개선하는 거대한 피부 데이터를 생성하는 방법을 설명합니다.

- **Technical Details**: 첫 번째 연구에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 연구는 반사실적 데이터 세트를 통해 인과 관계(causal relationship)를 강화하고, 이를 통해 다양한 차원에서 모델의 성능을 향상시키는 방법을 제안합니다. 마지막 연구에서는 S-SYNTH를 통해 3D 피부 모델과 디지털로 렌더링된 이미지를 생성하며, 이를 통해 실제 피부 이미지의 성능 추세를 모방하고, 기존 데이터 세트의 한계를 보완합니다.

- **Performance Highlights**: 첫 번째 연구에서는 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 보였습니다. 두 번째 연구에서는 다양한 차원에서 모델의 강인성, 도메인 간 일반화(cross-domain generalization), 제한된 데이터에서의 일반화 성능이 크게 향상되었습니다. 마지막 연구에서는 S-SYNTH를 통해 생성된 합성 이미지가 제한된 실제 이미지 세트를 사용할 때 향상된 분할 성능(segmentation performance)을 보여주었고, 피부 색상과 병변 크기 측면에서 실제 환자 데이터 세트와 유사한 비교 성능 추세를 나타냈습니다.



### CREW: Facilitating Human-AI Teaming Research (https://arxiv.org/abs/2408.00170)
Comments:
          Our project website is at: this http URL

- **What's New**: 이미 발달된 NLP 태스크 수행 모델들의 정확도에도 불구하고, spurious 패턴에 의존하여 강건성(robustness)이 제한되는 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 도입한 새 접근법을 제안합니다. 이 방법은 여러 개의 반사실을 생성하고 집합적 결정(collective decisions)을 통해 예측 분포를 조정하여 단어들의 인과관계를 더 강하게 파악할 수 있는 robust한 방법입니다.

- **Technical Details**: 이 방법은 인간이 반사실(countrafactual)을 만들거나, 정확한 반사실을 데이터셋에서 찾게 하는 기존의 방법과 달리, 반사실 다발을 합성하여 이를 기반으로 공급하는 예측 분포를 조정합니다. 이러한 공급 체계는 모델의 인과성을 더 robust하게 평가합니다. 기존 spurious correlation을 피하면서 모델의 robustness를 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 제안한 접근법은 다양한 측면에서 매우 유의미한 향상을 보여주었습니다. 1) 반사실 robust, 2) cross-domain 일반화 능력, 3) 소량의 데이터에서의 일반화 능력에서 기존 방법보다 뛰어난 성능을 입증하였습니다.



### Review of Explainable Graph-Based Recommender Systems (https://arxiv.org/abs/2408.00166)
- **What's New**: 새로운 논문들은 MCQ 자동 생성의 교육적 평가를 위한 새로운 메트릭(KDA)을 제안하고, NLP 태스크의 robust하도를 향상시키기 위해 집합적 의사결정(collective decisions)을 활용하며, 그래프 기반 추천 시스템의 설명 가능성을 탐구한다.

- **Technical Details**: 첫 번째 논문에서는 지식 종속 가능성(KDA)을 측정하여 MCQ의 답변 가능성을 평가하고, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안한다. 두 번째 논문에서는 '여러 개의' counterfactual을 생성하여 단어들의 인과관계를 robust하게 파악하는 방법을 설명한다. 세 번째 논문에서는 그래프 기반 추천 시스템의 설명 가능성을 학습 방법, 설명 방법, 설명 유형 등의 세 가지 측면으로 분류하여 분석한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, 전문가가 라벨링한 MCQ 품질 측정치와 높은 예측력을 보인다. 집합적 의사결정을 통한 접근 방식은 counterfactual robustness, cross-domain generalization, 및 적은 데이터에서의 일반화 성능에서 중요한 향상을 보여준다.



### Non-convolutional Graph Neural Networks (https://arxiv.org/abs/2408.00165)
- **What's New**: 이 논문들은 새로운 자동 MCQ 생성 평가 메트릭, 지식 종속 가능성(KDA)을 소개하며, 최근 NLP 태스크에서의 deep model들의 강인성 한계를 극복하기 위해 대비 학습(contrastive learning)과 counterfactual augmentation을 적용한 방법을 제안합니다. 또한, convolution이 필요 없는 새로운 graph neural network(GNN) 모델, 랜덤 워크와 통합 메모리(RUM)를 소개합니다.

- **Technical Details**: {'MCQ Generation': '기존의 n-gram 기반 평가 메트릭들이 교육적 가치를 고려하지 않는 문제를 해결하기 위해 KDA를 도입했습니다. KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다.', 'Robustness in NLP': '대비 학습과 counterfactual augmentation을 결합하여 추론 모델의 강인성을 개선했습니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안했습니다.', 'GNN without Convolution': '기존 convolution 기반 GNN의 한계를 극복하기 위해 랜덤 워크와 통합 메모리(RUM)라는 새로운 모델을 제안했습니다. RUM은 RNN을 사용하여 랜덤 워크를 통해 수집된 토폴로지 및 의미적 그래프 특징을 결합합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont 메트릭은 실제 강의실 세트에서 전문가들이 평가한 usability와 강한 상관관계를 가지며, 다양한 전문가가 라벨링한 MCQ 품질 측정에서 높은 예측력을 가졌습니다.', 'Robustness in NLP': '제안된 방법은 다양한 측면에서 유의미한 개선을 이루었으며, 특히 counterfactual robustness, cross-domain generalization, scarce data로부터의 일반화 부문에서 강한 성능을 보였습니다.', 'GNN without Convolution': 'RUM 모델은 node- 및 graph-level 분류 및 회귀 작업에서 경쟁력 있는 성능을 보였으며, convolutional GNN보다 메모리 효율적이고 확장 가능하며 속도가 빠릅니다.'}



### A Taxonomy of Stereotype Content in Large Language Models (https://arxiv.org/abs/2408.00162)
- **What's New**: 우리는 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'의 도입을 통해 MCQ의 교육적 평가를 강화하려고 합니다. 또한, 딥러닝 모델의 강인성을 높이기 위해 대조 학습 (contrastive learning)과 반사실적 증대(counterfactual augmentation) 기법을 활용하는 연구도 소개합니다. 마지막으로, 최신 대형 언어 모델 (LLM)에서의 고정관념 내용을 분류하는 새로운 방식의 연구를 발표합니다.

- **Technical Details**: 첫 번째 연구에서는 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭의 한계를 극복하기 위해 KDA를 제안하고, 학생의 문제 해결 행동을 사전 학습된 언어 모델이 모방하게 하여 KDA_disc와 KDA_cont 자동 평가 메트릭을 개발했습니다. 두 번째 연구에서는 '집합적 의사 결정 (collective decisions)'을 통해 단어들의 인과관계를 파악하는 방법을 제안하고, counterfactual을 '여러 개' 생성하는 접근법을 사용했습니다. 세 번째 연구는 ChatGPT 3.5, Llama 3, Mixtral 8x7B 모델에서 87개의 사회 카테고리를 기반으로 한 고정관념 차원을 분류하고, 이를 통해 모델의 내부 평가를 예측하는 다차원 분류법을 제안했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 강한 상관관계를 보였으며, MCQ 품질 평가에서도 예측 능력을 높였습니다. 두 번째 연구에서는 다양한 차원에서 강인성이 크게 개선되었으며, 특히 counterfactual robustness 및 cross-domain generalization에서 두각을 나타냈습니다. 세 번째 연구에서는 LLM이 인간보다 더 긍정적인 고정관념을 보이지만, 카테고리와 차원에 따라 큰 변동성을 보였습니다.



### Automatic Generation of Behavioral Test Cases For Natural Language Processing Using Clustering and Prompting (https://arxiv.org/abs/2408.00161)
- **What's New**: 최근의 딥 러닝 모델들이 사람보다 높은 정확성을 보이는 NLP 태스크에서 롭스터스 신뢰도(robustness) 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적인 증강(counterfactual augmentation)을 활용. 기존 평가 메트릭은 n-그램 기반 유사성에 집중하여 교육적 가치를 평가하지 못함. 우리는 MCQ 생성에서 학생의 지식 평가 능력을 측정하는 새로운 메트릭을 제안.

- **Technical Details**: 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 사용하여 MCQ 대답 가능성을 측정. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 흉내내는 방식으로 KDA를 근사. 또한, 반사실적 데이터를 '여러 개' 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어 인과 관계를 robust하게 학습. 대규모 언어 모델을 사용하여 의미 있는 그룹을 만들고, MFT(Minimal Functionality Tests)를 자동 생성.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 사용성과 강한 상관관계를 보임. 집합적 의사 결정을 통해 다양한 측면에서 높은 성능을 보였으며, 반사실적 robust성, 크로스 도메인 일반화, 적은 데이터에서의 일반화 능력에서 큰 개선을 달성. Amazon 리뷰 코퍼스를 사용하여 자동 테스트 케이스 생성 방법을 시연하고, 네 가지 분류 알고리즘의 행동 테스트 프로필을 분석.



### Moderating Group Conversation Dynamics with Social Robots (https://arxiv.org/abs/2408.00151)
Comments:
          6 pages, 6 figures, 1 table. Accepted at the workshop on advancing Group Understanding and robots' adaptive behavior (GROUND), held at the Robotics Science and Systems (RSS) Conference, 2024

- **Multiple Choice Questions (MCQ) 자동 생성**: [{"What's New": "MCQ 자동 생성은 기존 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 고려하지 못한다는 문제를 해결하기 위해 '지식 종속 가능성(KDA)'이라는 새로운 평가 메트릭을 제안합니다."}, {'Technical Details': 'KDA는 학생이 해당 사실에 대한 지식을 바탕으로 MCQ에 답할 수 있는 능력을 측정합니다. 이를 위해 학생 응답을 기반으로 KDA를 측정하고, 예측 언어 모델을 활용하여 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 KDA와 높은 상관관계를 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 측정에 대한 예측력이 강력함을 입증했습니다.'}]

- **Natural Language Processing의 Robustness 개선**: [{"What's New": '최근의 deep models가 NLP 태스크에서 높은 정확도를 보이지만, spurious pattern에 의존하여 robustness가 제한적이라는 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 방법을 제안합니다.'}, {'Technical Details': '기존 방법들은 사람이 counterfactual을 추가하거나 모델이 데이터셋에서 유사한 것들을 찾아야 하지만 여전히 spurious correlation에 취약합니다. 새로운 접근법은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 더 robust하게 단어들의 인과관계를 파악합니다.'}, {'Performance Highlights': '실험 결과, 집합적 의사 결정을 통해 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 부족한 데이터로부터의 일반화라는 다양한 측면에서 눈에 띄는 개선을 달성했습니다.'}]

- **사회적 로봇의 그룹 대화 참여 향상**: [{"What's New": '사회적 로봇이 그룹 대화에 참여할 때의 영향을 연구하고, 다양한 addressing 정책의 효과를 평가했습니다. 연구에는 300명의 참가자가 참여하여 4명씩 그룹을 이루었으며, 로봇이 대화를 주도했습니다.'}, {'Technical Details': 'CAIR(Cloud Artificial Intelligence and Robotics)라는 클라우드 소프트웨어 아키텍처를 사용하여 로봇이 자율적으로 여러 사람과 대화할 수 있도록 했습니다. 대화 관리와 사용자 의도 파악을 위한 여러 서비스가 포함되어 있습니다.', 'Participation Dynamics': "지배적인 참여자와 소외된 참여자가 생기는 문제를 해결하기 위해 'Balancing' 정책과 'Community' 정책을 도입했습니다. 전자는 대화에서 덜 활동적인 사용자를 식별하고 참여하게 하는 것을 목표로 하며, 후자는 참가자들 간의 소그룹을 식별하고 관리합니다."}, {'Performance Highlights': '실험을 통해 로봇의 addressing 정책이 대화 역학에 큰 영향을 미쳤으며, 각 참가자에게 더 균형 잡힌 주의를 기울이고 소그룹 형성을 줄이는 결과를 보여주었습니다.'}]



### StyleRF-VolVis: Style Transfer of Neural Radiance Fields for Expressive Volume Visualization (https://arxiv.org/abs/2408.00150)
Comments:
          Accepted by IEEE VIS 2024

- **What's New**: 최근 교육자들의 시간 절감과 학생 평가 방안을 위해 새로운 자동 MCQ (Multiple Choice Questions) 생성 방법이 제안되었습니다. 또한, NLP (Natural Language Processing) 태스크에서 더욱 견고한 모델 학습을 위해 contrastive learning과 counterfactual augmentation을 결합한 방안이 연구되었습니다. 마지막으로, 볼륨 비주얼라이제이션 (VolVis)에서의 새로운 스타일 전달 프레임워크 'StyleRF-VolVis'가 소개되었습니다.

- **Technical Details**: 첫 번째 논문에서는 MCQ의 교육적 가치를 평가하기 위해 지식 종속 대답 가능성(KDA)을 측정하는 새로운 자동 평가 메트릭을 제안했습니다. 두 번째 논문에서는 spurious correlations에 영향을 받지 않기 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안했습니다. 마지막 논문에서는 NeRF (neural radiance field) 모델을 활용하여 DVR (Direct Volume Rendering) 장면의 스타일을 변경하면서도 원본 콘텐츠 정보를 유지하는 'StyleRF-VolVis' 기술을 개발했습니다.

- **Performance Highlights**: 첫 번째 연구에서는 KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 높은 상관관계를 보여 주었습니다. 두 번째 연구는 실험 결과, 집합적 의사 결정 방식이 할당 기반 합성의 task model bias에 덜 민감하여 다양한 측면에서 성능이 향상됨을 보였습니다. 세 번째 연구는 StyleRF-VolVis가 다양한 볼륨 렌더링 장면에 대해 기존 AdaIN, ReReVST, ARF, SNeRF 스타일 렌더링 솔루션보다 우수한 품질, 일관성, 유연성을 보였습니다.



### Distributed In-Context Learning under Non-IID Among Clients (https://arxiv.org/abs/2408.00144)
Comments:
          12 pages

- **What’s New**: 이 논문은 새로운 자동 평가 메트릭인 '지식 종속 가능성' (Knowledge Dependent Answerability, KDA)을 제안합니다. 기존의 MCQ 생성 평가 메트릭(BLEU, ROUGE, METEOR)이 교육적 가치를 무시하는 문제를 해결하기 위해 설계되었습니다. 또한, 대규모 언어 모델의 적응 문제를 다루며, 대리 데이터셋을 사용한 분산형 non-IID ICL 문제 해결 방법도 제시합니다.

- **Technical Details**: KDA는 학생의 답변 가능성을 기준으로 MCQ의 교육적 가치를 측정합니다. 이를 위해, 인간 설문조사를 통해 학생 응답 데이터를 사용해 KDA를 측정하는 방법을 제시하고, pretrained language models를 활용해 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 또한, in-context learning (ICL)을 분산형 non-IID 데이터 상황에 맞게 개선하는 방법을 논의하며, 서버가 클라이언트별로 다르게 ICE 예산을 할당하는 새로운 프레임워크를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가와 전문가가 선정한 범주에서 강한 상관관계를 보여줍니다. 또한, 논문의 접근법이 부정적인 패턴(spurious patterns)에 덜 민감하며, counterfactual robustness, cross-domain generalization 및 scarce data의 일반화에서 유의미한 성과를 얻었습니다. 분산형 non-IID ICL 상황에서도 다양한 데이터셋에서 경쟁 방법들보다 우수한 성능을 입증했습니다.



### Correcting Negative Bias in Large Language Models through Negative Attention Score Alignmen (https://arxiv.org/abs/2408.00137)
- **What's New**: 본 연구는 자동으로 다지 선다형 질문(MCQ)을 생성할 때 사용할 수 있는 새로운 평가 메트릭인 지식 종속 가능성(KDA; Knowledge Dependent Answerability)을 제안합니다. 이는 기존의 BLEU, ROUGE, METEOR와 달리 생성된 MCQ가 학생의 지식 평가 능력을 측정하는데 초점을 맞추고 있습니다.

- **Technical Details**: 지식 종속 가능성(KDA)은 '목표 사실(target fact)'에 대한 학생의 지식을 바탕으로 MCQ의 답변 가능성을 평가합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용해 학생들의 문제 해결 행위를 시뮬레이션함으로써 KDA를 자동으로 측정하게 됩니다. 이 과정에서 인간의 응답을 통해 KDA를 평가하는 방법을 먼저 제시하고, 이를 바탕으로 두 개의 자동 평가 메트릭을 개발했습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합될 때, KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측력을 보여줍니다.



### Distributionally Robust Optimization as a Scalable Framework to Characterize Extreme Value Distributions (https://arxiv.org/abs/2408.00131)
- **What's New**: 이번 논문은 다차원 극단값 이론 (Extreme Value Theory, EVT) 통계를 위해 분포적으로 강건한 최적화 (Distributionally Robust Optimization, DRO) 추정기를 개발하는 것을 목표로 하고 있다. 극단값 데이터가 희소한 특성을 가지고 있어 모델의 오류 가능성이 크기 때문에 이러한 DRO 추정기가 필요한 상황을 다룬다. 특히, 우리는 반간편적인  Max-Stable 분포를 기반으로 한 DRO 추정기를 제안하고, 금융 수익 데이터에 적용하여 기존 분석과 비교하였다.

- **Technical Details**: 이번 연구에서는 점 프로세스 공간에서 반간편적 Max-Stable 제약에 의해 정보 제공되는 DRO 추정기를 연구하였다. 또한, 다양한 추정기 문제에 대한 tractable convex 공식화와 더 일반적인 신경망 기반 추정기를 제안하였다. 특히, Optimal Transport 및 Wasserstein 거리를 사용하여 모델의 불확실성을 매개하는 방법을 채택하였다. CVaR (Conditional Value at Risk) 지표를 사용하여 네트워크의 성능을 평가하였고, 다변수 극단값 분포 데이터 셋에서의 유효성을 검증하였다.

- **Performance Highlights**: 제안된 방법은 수치적으로 생성된 데이터를 통해 유효성을 검증하였으며, 기존 분석과 비교하여 혁신적인 성능을 보였다. 구체적으로, 우리는 CVaR 지표를 사용하여 다변수 극단값 분포 데이터 셋에서의 성능을 평가하였고, 금융 수익의 실제 데이터 셋에서도 유사한 결과를 도출하였다. 우리의 방법은 분포 외 성능을 개선하면서도 과도하게 보수적인 추정치를 완화하는 데 효과적임을 보여주었다.



### Semantic Codebook Learning for Dynamic Recommendation Models (https://arxiv.org/abs/2408.00123)
- **What's New**: 자동 다중 선택 질문(MCQ) 생성의 첨단 평가 메트릭 Knowledge Dependent Answerability (KDA)를 소개합니다. 또한, 자연어 처리(NLP)에서의 강력한 대립 학습(contrastive learning) 및 반사실 강화(counterfactual augmentation) 방법을 제안합니다. 마지막으로, 동적 순차 추천 모델(DSR)에서의 고도화된 프레임워크인 SOLID를 발표합니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 단순한 단어 유사성만을 측정하는 데 반해, KDA는 MCQ의 답변 가능성을 고려하여 학생의 학습 성과를 더 정확하게 평가함.'}, {'Robustness in NLP': 'spurious 패턴에 의존하는 기존 방법의 문제를 해결하기 위해 대립 학습(contrastive learning) 및 반사실 데이터 강화(counterfactual augmentation)를 사용하여 모델의 강력함을 높입니다. 여러 개의 counterfactual을 생성하여 집합적 의사 결정을 통해 인과관계를 더 명확히 파악합니다.'}, {'Dynamic Sequential Recommendation': 'SOLID 프레임워크는 아이템 시퀀스를 의미 시퀀스로 변환시키고 dual parameter model을 사용하여 검색 공간을 축소하고 추천 시스템의 동질성을 활용합니다. 또한, semantic metacode와 semantic codebook을 사용하여 분리된 아이템 표현을 저장하여 정확한 파라미터 생성을 보장합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': '인간 설문을 통한 평가에서 KDA_disc와 KDA_cont는 실제 강의실에서 사용 가능성이 높고 전문가들이 라벨링한 다양한 MCQ 품질 지표와 강한 상관관계를 보였습니다.'}, {'Robustness in NLP': '우리의 접근 방식은 대립 학습 및 반사실 강화 기술을 통해 1) 반사실 robustness, 2) 교차 도메인 일반화, 3) 희소한 데이터로부터의 일반화에서 현저한 개선을 달성했습니다.'}, {'Dynamic Sequential Recommendation': 'SOLID는 기존의 DSR을 일관되게 능가하며 더 정확하고 안정적이며 강력한 추천을 제공합니다. 실험 결과, SOLID는 다양한 평가 지표에서 탁월한 성능을 입증했습니다.'}]



### Gemma 2: Improving Open Language Models at a Practical Siz (https://arxiv.org/abs/2408.00118)
- **What's New**: 자동 다중 선택 질문(MCQ)의 평가를 위해 새로운 자동 평가 메트릭인 지식 종속 대답 가능성(KDA)을 제안했으며, 이는 학생이 특정 사실에 대해 가진 지식을 평가할 수 있는 MCQ의 대답 가능성을 측정하는 것을 목표로 합니다. 또한, 깊은 학습 모델의 강건성 문제를 해결하기 위해 대조 학습과 반사실적 증강(contrastive learning and counterfactual augmentation)을 활용하는 새로운 방법이 제안되었습니다. 마지막으로, 새로운 경량 언어 모델인 Gemma 2를 소개하며, 이는 2억에서 270억 매개변수까지의 규모로 제공됩니다.

- **Technical Details**: 기존 평가 메트릭들은 MCQ의 교육적 가치를 반영하지 못하는 문제가 있었으며, 이에 대응하기 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭들이 제안되었습니다. 이는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 대조 학습과 반사실적 증강 방법을 통해 여러 개의 반사실적(synthetic) 데이터를 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계(causality)를 더 강건하게 평가합니다. Gemma 2는 Transformer 아키텍처에서 로컬 및 글로벌 주의층(local-global attentions)과 그룹-질의 주의(group-query attention) 메커니즘을 적용받았으며, 지식 증류(knowledge distillation)를 통해 훈련되었습니다.

- **Performance Highlights**: Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실(set)에서의 사용성 측면에서 강한 상관관계를 가짐을 입증했습니다. 제안된 대조 학습 및 반사실적 증강 접근 방식은 다양한 차원에서 강건성, 도메인 간 일반화, 희소한 데이터로부터의 일반화에 있어서 큰 개선을 이루었음을 경험적으로 보여주었습니다. Gemma 2 모델은 여러 자동화 및 인간 평가에서 경쟁력 있는 성능을 보여주었고, 특정 질문 응답, 상식 추론, 수학 및 과학, 코딩 등의 도메인에서 두드러진 성능을 보였습니다.



### Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models (https://arxiv.org/abs/2408.00113)
Comments:
          Oral paper (top 5%) at the ICML 2024 Mechanistic Interpretability Workshop

- **What's New**: 새로운 자동 MCQ 평가 메트릭인 지식 종속 가능성(KDA)이 제안되었습니다. 또한, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭이 소개되었습니다. 한편, NLP 태스크에서 강력한 정확성을 보이는 최신 deep models가 spurious patterns에 의존하는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용한 새로운 방법론이 제시되었습니다. 마지막으로, Sparse Autoencoders(SAEs)의 성능을 측정하는 새로운 기법 p-annealing이 제안되었습니다.

- **Technical Details**: ['MCQ 평가에서 BLEU, ROUGE, METEOR 같은 기존 메트릭은 교육적 가치를 고려하지 않기 때문에 KDA라는 새로운 메트릭이 제안되었습니다. KDA는 학생들의 반응을 토대로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다.', 'NLP 모델의 robustness를 높이기 위해 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 강화하는 방법이 제안되었습니다. 이 방법은 기존 spurious correlation 문제를 해결하고 여러 차원에서 성능을 향상시킵니다.', 'Sparse Autoencoders(SAEs)에서 interpretable features를 분리하기 위해 chess와 Othello 게임 transcript를 사용하는 새로운 SAE 평가 기법이 도입되었습니다. p-annealing이라는 새로운 SAE 훈련 기법도 소개되었습니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont 메트릭은 실제 강의실 세트에서 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 경우 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 높은 예측력을 보였습니다.', '새로운 counterfactual augmentation 접근법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등 다양한 차원에서 기존 방법을 능가하는 성능을 보여주었습니다.', 'p-annealing 기법을 사용한 SAEs는 기존의 비지도 학습 메트릭과 새로운 메트릭 모두에서 높은 성능을 기록하며, 이전 메소드와 비교해 개선된 결과를 보여주었습니다.']



### WAS: Dataset and Methods for Artistic Text Segmentation (https://arxiv.org/abs/2408.00106)
Comments:
          Accepted by ECCV 2024

- **What's New**: 자동 다중 선택 질문(MCQ) 생성의 효율성을 개선하기 위해 새로운 평가 메트릭, Knowledge Dependent Answerability(KDA)를 도입했습니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 메트릭의 한계를 극복하고 교육적 가치를 평가합니다.

- **Technical Details**: KDA는 학생 설문을 통해 MCQ의 대답 가능성(answerability)을 측정합니다. 이를 기반으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실에서의 사용성과 강한 상관관계를 보여주었고, n-gram 기반 메트릭과 결합해 전문가가 라벨링한 MCQ 품질을 예측하는 데 높은 정확도를 보였습니다.



### ReLiK: Retrieve and LinK, Fast and Accurate Entity Linking and Relation Extraction on an Academic Budg (https://arxiv.org/abs/2408.00103)
Comments:
          To be presented at ACL 2024

- **What's New**: 새롭게 제안된 논문에서는 자동 MCQ 생성 평가를 위한 지식 종속 가능성(KDA) 메트릭을 도입했습니다. 이는 기존의 n-gram 기반 유사성을 넘어 학생의 목표 사실에 대한 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: 우리는 먼저 human survey 데이터를 통해 KDA를 측정하고, 이를 근거로 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 통해 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 지표에 대한 예측력이 높음을 보여주었습니다.



### From Attributes to Natural Language: A Survey and Foresight on Text-based Person Re-identification (https://arxiv.org/abs/2408.00096)
- **What's New**: 이번 논문은 교육자들이 학생 평가에 소요하는 시간을 크게 줄일 수 있는 Multiple Choice Questions (MCQ) 자동 생성 분야의 혁신적인 연구를 소개합니다. 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 데이터셋 내 금 샘플과의 n-gram 기반 유사성에만 초점을 두어 교육적 가치를 간과하고 있습니다. 이러한 문제를 해결하기 위해, 우리는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 측정하고 해당 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 특히, 학생 응답을 분석하여 KDA를 측정하는 방법을 먼저 제시하고, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사화하는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대한 예측 정확도를 높입니다.

- **Performance Highlights**: 인간 연구를 통해 우리는 KDA_disc와 KDA_cont가 실제 강의실 설정에서 사용 가능성과 강한 상관관계를 가지고 있음을 보였습니다. 또한, 이 메트릭들은 전문가가 라벨링한 다양한 MCQ 품질 측정치와 강한 예측력을 보여주었습니다.



### Execution Semantics of Behavior Trees in Robotic Applications (https://arxiv.org/abs/2408.00090)
Comments:
          13 pages, 9 figures

- **What's New**: 교육 MCQ 생성과 관련된 평가 메트릭을 개선하기 위해 지식 종속 대답 가능성 (KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 최근 NLP 작업에서 사람보다 높은 정확성을 보였으나, robust한 결과를 위해 contrastive learning과 counterfactual augmentation을 활용한 접근법도 소개됩니다.

- **Technical Details**: 새로운 KDA 평가 메트릭은 학생의 대상 사실에 대한 지식을 측정하는 능력을 평가합니다. 우리는 KDA_disc와 KDA_cont라는 두 가지 평가 지표를 제안하여 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방할 수 있도록 했습니다. 또한 counterfactual augmentation에서 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 spurious correlation의 영향을 최소화하는 새로운 방법을 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 Human evaluation 결과 실제 강의실 세트 사용성에서 높은 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링한 MCQ 품질 측정 기준에 대해 강력한 예측력을 나타냈습니다. 새로운 counterfactual augmentation 접근법은 counterfactual robustness, cross-domain generalization, scarce data 상황에서도 유의미한 성능 향상을 보여주었습니다.



### Barlow Twins Deep Neural Network for Advanced 1D Drug-Target Interaction Prediction (https://arxiv.org/abs/2408.00040)
Comments:
          9 pages, 2 figures

- **What's New**: 이번 뉴스레터에서는 세 가지 최신 연구 논문을 소개합니다. 첫 번째 논문은 자동 Multiple Choice Questions(MCQ) 생성의 교육적 가치를 평가하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. 두 번째 논문은 NLP 태스크에서의 robust한 성능을 위해 contrastive learning과 counterfactual augmentation을 이용하는 방법을 다룹니다. 세 번째 논문은 Barlow Twins 아키텍처를 활용하여 drug-target interaction(DTI) 예측에서 state-of-the-art 성능을 달성하는 방법을 소개합니다.

- **Technical Details**: [{'Knowledge Dependent Answerability (KDA)': '기존 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성만을 측정하여 교육적 가치를 반영하지 못하는 단점을 보완합니다. KDA라는 새로운 메트릭을 소개하며, KDA_disc 와 KDA_cont 두 가지 자동 평가 메트릭을 제안하여 학생들의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 활용합니다.'}, {'Contrastive Learning and Counterfactual Augmentation': 'Deep models의 robustness 문제를 해결하기 위해 spurious correlation을 피하는 다중 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 더 robust 하게 파악하는 방식을 제안합니다. 이 방법은 counterfactual robustness, cross-domain generalization, scarcity data generalization에서 모두 개선된 성능을 보입니다.'}, {'Drug-Target Interaction (DTI) Prediction using Barlow Twins': 'Barlow Twins 아키텍처를 사용하여 feature-extraction을 수행하고, gradient boosting machine을 통해 예측 성능을 최적화합니다. 이 방법은 structure-agnostic 접근 방식을 사용하며, state-of-the-art 성능을 다양한 벤치마크에서 입증했습니다. 특히 데이터가 부족한 상황에서 강력한 성능을 발휘합니다.'}]

- **Performance Highlights**: [{'KDA Performance': 'Human evaluation을 통해 KDA_disc 와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 가짐이 입증되었습니다. 또한, n-gram 기반의 유사성 메트릭과 결합하여 전문가가 레이블링한 다양한 MCQ 품질 측정에서 강력한 예측 능력을 보여줍니다.'}, {'Robustness in NLP Tasks': 'Proposed robust methodology는 기존 방법과 비교하여 다양한 차원에서 significant improvements를 보여주었습니다. 특히, counterfactual robustness와 cross-domain generalization, 그리고 희소 데이터 상황에서도 뛰어난 성능을 기록하였습니다.'}, {'DTI Prediction Results': 'BarlowDTI는 BioSNAP, BindingDB, DAVIS 벤치마크 테스트에서 기존의 모든 모델을 능가하는 성능을 발휘하였습니다. BindingDB 벤치마크에서 unseen ligands를 예측할 때 12% 이상의 PR_AUC improvement를 기록하였으며, 전체적으로 두 개의 split에서 가장 우수한 성능을 발휘하였습니다.'}]



### WebApp1K: A Practical Code-Generation Benchmark for Web App Developmen (https://arxiv.org/abs/2408.00019)
- **What's New**: 이 논문에서는 자동 다지선다형 질문 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하고, 이를 통해 교실에서의 유용성을 높이기 위한 방법을 연구하였습니다. 또한 대규모 언어 모델(LLM)이 웹 앱을 개발할 수 있는 능력을 측정하기 위한 WebApp1K라는 새로운 벤치마크를 도입하였습니다.

- **Technical Details**: KDA는 학생이 특정 사실에 대한 지식을 바탕으로 질문에 답할 수 있는 능력을 평가합니다. 이를 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭이 제안되었으며, 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다. WebApp1K는 React 프레임워크를 사용하여 LLM의 실제 응용 프로그램 개발 능력을 평가하며, 1000개의 문제로 구성된 벤치마크를 통해 모델의 코드 정확성과 기능성을 평가합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서 사용될 때 높은 상관성을 보이며, BLEU, ROUGE 등의 기존 메트릭과 결합하여 더 높은 예측력을 보여줍니다. WebApp1K에서는 오픈 소스 LLM들이 GPT-4o와 Claude 3.5 Sonnet 뒤를 이어 뛰어난 성능을 나타냈으며, 모델 크기와 코드 정확성 간에 강한 상관관계가 발견되었습니다. 또한, 특정 프롬프트 기술이 모든 모델의 성능을 보편적으로 향상시키지 않는다는 결과도 도출되었습니다.



### Framework for Curating Speech Datasets and Evaluating ASR Systems: A Case Study for Polish (https://arxiv.org/abs/2408.00005)
Comments:
          Submitted to NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 자동으로 주관식 질문을 생성하지만, 현재의 평가 지표들은 교육적 가치를 반영하지 못하는 문제를 극복하기 위해 새로운 평가 메트릭을 제안했습니다. 또한, NLP 태스크에서 스퍼리어스 패턴에 의존하는 문제를 해결하기 위해 대조 학습과 counterfactual 증강 방법을 소개했습니다. 폴란드어 음성 인식 시스템의 평가를 위해 포괄적인 프레임워크를 마련하고, 다양한 ASR 시스템과 모델을 비교하는 사례 연구를 진행했습니다.

- **Technical Details**: 첫 번째 연구에서는 지식 종속 가능성(KDA)라는 새로운 평가 메트릭을 제안하여, 학생들이 대상 사실에 대한 지식으로 주관식 질문을 답할 수 있는지를 평가합니다. 두 번째 연구에서는 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 세 번째 연구에서는 24개의 공개 데이터셋을 큐레이팅하여 폴란드어 ASR 시스템 간 성능을 비교하는 프레임워크를 구축했습니다.

- **Performance Highlights**: 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 실제 강의실 사용성과 강한 상관관계를 보였습니다. 집합적 의사 결정을 통한 새로운 증강 방법으로 다양한 차원에서의 성능 향상을 달성했습니다. 폴란드어 ASR 시스템 평가에서 25개의 시스템 및 모델을 비교하여 다양한 데이터셋과 화자 인구통계에 따른 성능 변화를 발견했습니다.



### Handling Numeric Expressions in Automatic Speech Recognition (https://arxiv.org/abs/2408.00004)
- **What's New**: MCQ(주관식 문제)의 자동 생성 기술이 발전함에 따라, 교사들이 학습 평가에 소비하는 시간 절감 가능성이 크게 부각되고 있습니다. 그러나 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 n-gram 기반의 유사성만을 평가해 교육적 가치를 무시하고 있습니다. 이를 해결하기 위하여, 지식 종속 대답 가능성(KDA)이라는 새로운 자동 평가 메트릭이 제안되었습니다. 또한, 논문은 NLP 태스크에서 최근의 deep model들이 사람을 능가하는 정확성을 보였으나, spurious pattern에 의존해 robustness가 제한된다는 문제를 제기하고 있습니다. 이에 대처하기 위해, contrastive learning과 counterfactual augmentation을 활용한 방법을 제안합니다. 마지막으로, 자동 음성 인식(ASR) 시스템에서 숫자 표현을 올바르게 포맷하는 문제를 다루고, 이를 위해 LLM과 TTS 모델을 사용한 데이터 생성 전략을 비교합니다.

- **Technical Details**: 첫 번째 논문에서는 MCQ의 대답 가능성을 평가하는 KDA 메트릭을 제안하며, 인간 설문을 통한 KDA 측정 방법을 소개합니다. 두 번째 논문은 여러 counterfactual을 생성하여 집합적 의사 결정을 통해 단어들의 인과관계를 파악하는 방법을 설명합니다. 세 번째 논문에서는 numeric expressions (숫자 표현)을 올바르게 포맷하기 위해, LLM과 TTS 모델을 사용한 데이터 생성 전략을 사용하며, cascaded와 end-to-end 접근 방식을 비교합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였고, n-gram 기반 메트릭과 함께 사용할 때 다양한 전문가 라벨링된 MCQ 품질 지표에 대한 예측력이 뛰어난 것으로 나타났습니다. counterfactual 합성 방법은 여러 차원에서 significant improvement를 달성했으며, 특히 counterfactual robustness와 교차 도메인 일반화에서 두드러진 성과를 보였습니다. ASR 숫자 표현 포맷팅에서 end-to-end 모델은 낮은 지연시간과 추론 비용이라는 장점과 함께 competitive한 성능을 보였습니다.



### Evaluating Transfer Learning in Deep Learning Models for Classification on a Custom Wildlife Dataset: Can YOLOv8 Surpass Other Architectures? (https://arxiv.org/abs/2408.00002)
Comments:
          This paper is being reviewed by SN Computer Science (springer journal)

- **What's New**: 이 논문은 자동으로 Multiple Choice Questions(MCQ)를 생성할 때, 기존의 평가 메트릭이 교육적 가치를 충분히 반영하지 않는 문제를 제기하고 있습니다. 이를 해결하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)을 제안합니다. 또한, 우리가 제안한 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭이 실제 강의실에서의 사용성 및 기존 메트릭과 강한 상관관계를 가지고 있음을 확인했습니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성(answerability)을 측정하고, 학생의 지식을 평가하는 능력을 평가합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델(pre-trained language models)을 사용해 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 우리는 다양한 인간 연구를 통해 이 메트릭이 강의실에서도 유용하다는 것을 보여주었으며, n-gram 기반의 유사성 메트릭과 결합하여 MCQ 품질을 예측하는 강력한 예측 능력을 가집니다.

- **Performance Highlights**: {'MCQ Generation': {'KDA_disc': '강의실 사용성 및 KDA와 높은 상관관계', 'KDA_cont': '유사하게 높은 강의실 사용성 및 KDA와의 상관관계'}, 'NLP Robustness': '이전 연구들이 spurious patterns에 의존하여 robustness가 제한되는 문제를 지적하고, 대비 학습(contrastive learning) 및 반사실적 증가(counterfactual augmentation) 방법을 활용해 이를 개선', 'Wildlife Monitoring': '딥러닝 모델(CNNs)과 전이 학습(transfer learning)을 통해 멸종 위기 종의 모니터링을 자동화하는 방법 연구. YOLOv8 모델이 97.39%의 훈련 정확도와 96.50%의 F1 점수를 기록하며, 다른 모델보다 높은 성능을 보임'}



### Replication in Visual Diffusion Models: A Survey and Outlook (https://arxiv.org/abs/2408.00001)
Comments:
          The first survey focuses on replication in visual diffusion models. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: {'MCQ Generation': "이 논문은 기존 MCQ(Multiple Choice Questions) 생성 평가 메트릭이 교육적 가치를 충분히 반영하지 못하는 문제를 다루고, 새롭게 'Knowledge Dependent Answerability (KDA)'라는 평가 메트릭을 제안합니다. 이 메트릭은 질문의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 평가합니다.", 'Contrastive Learning in NLP': '최근 NLP 태스크에서 높은 정확성을 보이는 deep models의 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 결합한 새로운 접근법을 제안합니다. 여기에 추가로 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 강화된 인과관계를 파악합니다.', 'Visual Diffusion Models': 'Visual diffusion models에서 훈련 데이터 복제를 탐구하고 이와 관련된 프라이버시, 보안, 저작권 문제를 다루는 첫 종합적인 리뷰를 제공합니다. 이 리뷰는 복제를 탐지하고 이해하며 완화하는 방법을 카테고리화합니다.'}

- **Technical Details**: {'MCQ Generation': 'KDA 측정을 위해 인간 설문조사를 기반으로 한 방법을 제시하고, 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont를 제안합니다. 이를 통해 학생의 문제 해결 행동을 모방하여 자동 평가 지표를 제공합니다.', 'Contrastive Learning in NLP': '여러 개의 counterfactual을 생성하여 집합적 의사 결정을 통해 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 이는 기존의 spurious correlation 문제를 해결할 수 있는 것으로 나타났습니다.', 'Visual Diffusion Models': '복제를 탐지하기 위한 메커니즘, 이 현상을 유발하는 요인 분석, 복제를 완화하는 전략을 설명합니다. 또한, 실세계에 미치는 영향을 검토하고, 특히 헬스케어 분야에서의 프라이버시 문제를 강조합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 교실 세트에서의 사용성 및 전문가가 라벨링한 다양한 MCQ 품질 측정 항목에 대해 강한 예측력(Strong predictive power)을 보였습니다.', 'Contrastive Learning in NLP': '이 접근법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등에서 유의미한 성능 향상을 보였습니다.', 'Visual Diffusion Models': '이 리뷰 논문은 복제 탐지와 벤치마킹에서의 어려움, 더욱 견고한 완화 기술의 개발 방향 등을 논의하여 연구자 및 실무자에게 유익한 통찰을 제공합니다.'}




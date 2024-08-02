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




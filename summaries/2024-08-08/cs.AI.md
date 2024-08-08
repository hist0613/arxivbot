New uploads on arXiv(cs.CL)

### SLIM-RAFT: A Novel Fine-Tuning Approach to Improve Cross-Linguistic Performance for Mercosur Common Nomenclatur (https://arxiv.org/abs/2408.03936)
Comments:
          13 pages, 1 figure, to be publish in International Conference on Web Information Systems and Technologies - WEBIST 2024 proceedings

- **What's New**: 이번 뉴스레터에서는 새로운 자동 평가 메트릭 Knowledge Dependent Answerability (KDA)와 NLP 모델의 강인성을 향상시키기 위해 대조 학습 및 대응 강화 방법을 적용한 모델, 그리고 포르투갈어 LLM(대형 언어 모델)을 활용한 특정 도메인 과업적 미세조정 기술인 SLIM-RAFT를 소개합니다.

- **Technical Details**: {'MCQ Generation': '자동 MCQ 생성을 위해 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭이 n-gram 유사성에만 집중하여 교육적 가치를 평가하지 않는 문제를 해결하기 위해, 학생의 대답 가능성을 측정하는 Knowledge Dependent Answerability (KDA)라는 새로운 메트릭을 제안했습니다. 기존 메트릭과 KDA_disc, KDA_cont를 결합해 명확하게 측정할 수 있음을 입증하였습니다.', 'Robust NLP Models': '대조 학습(contrastive learning)과 대응 강화(counterfactual augmentation)를 활용해 NLP 모델의 강인성을 향상시켰습니다. 여러 개의 대응 사실(counterfactual)을 생성해 합리적인 집합적 의사 결정을 통해 모델이 인과관계를 더 잘 이해할 수 있도록 설계하였습니다. 이는 기존의 인간 생성 대응 사실이나 자동 매칭 방식보다 더 효과적입니다.', 'SLIM-RAFT': '법율과 조세 분야에서 사용되는 MERCOSUR NCM 코드 시스템의 처리를 위해 작은 포르투갈어 LLM TeenyTineLLaMA를 사용한 미세조정 기술인 SLIM-RAFT를 제안합니다. 이 기술은 효율적이며 비용 효율적인 대안으로, 특히 소규모 데이터에서 높은 성능을 보입니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성을 평가한 결과, 학생들의 평가와 강한 상관관계를 보였습니다.', 'Robust NLP Models': '이 접근 방식은 대조 학습 및 대응 강화 방법을 더해, 실행된 모델이 기존 모델보다 다양한 측면에서 불균형적 상관성 문제에 덜 민감하며, 의미 없는 패턴에 덜 의존하도록 하여 성능을 개선했습니다.', 'SLIM-RAFT': 'SLIM-RAFT 모델은 MERCOSUR NCM 코드 시스템 처리에서 ChatGPT-4 보다 높은 성능 (8.67/10 vs 4.5/10)을 기록했습니다.'}



### From Words to Worth: Newborn Article Impact Prediction with LLM (https://arxiv.org/abs/2408.03934)
Comments:
          7 pages for main sections, plus 3 additional pages for appendices. Code, dataset are released at https://sway.cloud.microsoft/KOH09sPR21Ubojbc

- **What's New**: 이 논문은 최근 학문적 기여도를 높은 정확도로 예측하기 위해 최적화된 LLM(Large Language Models) 기반의 새로운 방법을 소개합니다. 구체적으로, 논문 제목과 초록만을 토대로 미래의 영향력을 예측하며, 이를 통해 기존의 외부 정보에 의존하는 방식에서 벗어나자는 제안을 내놓았습니다.

- **Technical Details**: 논문에서는 'KDA (Knowledge Dependent Answerability)'라는 새로운 평가 메트릭을 사용해 MCQ의 교육적 가치를 측정합니다. 또한, counterfactual augmentation과 contrastive learning을 활용해 NLP 모델의 robustness를 높이는 접근법을 제시합니다. 이 논문은 LoRA와 8-bit 모델 양자화 기법을 사용해 메모리 소비를 줄이고, 8 × NVIDIA A40 GPU 서버에서 실험을 진행했습니다.

- **Performance Highlights**: 제안된 방법은 출판된 논문의 미래 영향력을 예측하는 데 있어서, NDCG@20 기준으로 0.901의 성능을 달성하며 state-of-the-art 수준을 기록했습니다. 또한, 다양한 실험을 통해 이 접근법이 반직관적인 변칙성에 덜 민감하고, cross-domain generalization과 데이터가 부족한 상황에서도 우수한 성능을 보임을 입증했습니다.



### Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models (https://arxiv.org/abs/2408.03907)
Comments:
          6 pages paper content, 17 pages of appendix

- **What's New**: 자동 MCQ 생성에서 교육적 가치를 평가하기 위한 새 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 개발했습니다.

- **Technical Details**: KDA는 학생들의 실제 응답을 바탕으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. KDA_disc와 KDA_cont는 각각 이산형과 연속형 평가 메트릭으로, 인간 평가와 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 마킹 MCQ 품질 측정에서 높은 예측력을 보였습니다.



### Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Beyond (https://arxiv.org/abs/2408.03900)
Comments:
          Accepted at INTERSPEECH 2024. This version includes the same content but with additional appendices

- **What's New**: 다양한 논문에서 다양한 새로운 이론과 기술을 소개하고 있습니다. 첫 번째 논문에서는 교육자를 위해 지식 종속 가능성(KDA)을 통한 새로운 자동 MCQ 평가 메트릭을 제안합니다. 두 번째 논문에서는 대조 학습과 반사실적 데이터 증가(counterfactual augmentation) 기술을 사용해 NLP 모델의 회복 탄력성(robustness)을 향상시킵니다. 세 번째 논문에서는 다국어 음성 이해(SLU)를 위한 대규모 데이터셋인 Speech-MASSIVE를 소개하며, 이는 여러 언어와 과제를 포괄하는 멀티모달(multimodal) 데이터셋입니다.

- **Technical Details**: 첫 번째 논문에서는 교사의 학습 평가 시간 절감을 목표로 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안하고, 이를 기반으로 한 KDA_disc 및 KDA_cont를 통해 평가합니다. 두 번째 논문은 반사실적 증강을 사용해 여러 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 용어의 인과 관계를 더 견고하게 파악하는 방법을 제안합니다. 세 번째 논문은 Prolific 크라우드 소싱 플랫폼을 통해 12개 언어에 대해 115,140개의 음성 데이터셋을 수집하여 SLU, ASR, ST, LID 등 다양한 음성 관련 작업의 벤치마킹을 도와주는 Speech-MASSIVE를 제안합니다.

- **Performance Highlights**: 첫 번째 논문에서는 KDA_disc와 KDA_cont가 기존 n-그램 기반 메트릭과 결합했을 때, 전문적으로 라벨링된 다양한 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다. 두 번째 논문에서는 대조 학습을 통해 다양성 있는 차원에서 반사실적 견고성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희귀 데이터 일반화(generalization from scarce data) 측면에서 큰 개선을 보였습니다. 세 번째 논문에서는 현대의 다국어 음성 인식 모델인 Whisper를 사용한 ASR 평가에서 높은 일관성과 성능을 보였습니다.



### Simplifying Scholarly Abstracts for Accessible Digital Libraries (https://arxiv.org/abs/2408.03899)
Comments:
          Initial submission to JCDL2024

- **What's New**: 이번 아카이브 논문의 저자들은 학생 평가를 위한 자동적인 다지선다형 질문(MCQ) 생성의 효율성을 높이기 위한 새로운 평가 메트릭인 '지식 종속 답변 가능성(KDA, Knowledge Dependent Answerability)'을 제안하였습니다. KDA는 기존의 n-gram 기반 평가 메트릭들이 간과하는 교육적인 가치를 고려하여 MCQ의 실제 교육 효과를 측정합니다.

- **Technical Details**: 논문에서는 KDA를 측정하기 위해 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이는 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방함으로써 KDA를 근사화합니다. 이러한 접근법은 사전 학습된 언어 모델의 능력을 활용하여 MCQ의 답변 가능성을 분석합니다.

- **Performance Highlights**: Human evaluation 결과에 따르면 KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성에 대해 전문가들이 라벨링한 데이터와 강한 상관관계를 보였습니다. 또한, n-gram 기반의 유사성 메트릭과 결합했을 때, 전문가들이 라벨링한 다양한 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다.



### Personalized Clinical Note Generation from Doctor-Patient Conversations (https://arxiv.org/abs/2408.03874)
- **summaries**: [{"What's New": '우리는 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 지표를 제안하여 MCQ의 대답 가능성과 학생의 지식을 평가하는 능력을 측정합니다.', 'Technical Details': '기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에만 초점을 맞추고 교육적 가치를 무시합니다. 이를 해결하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이는 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행위를 모방합니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 인간 연구를 통해 KDA와 실제 강의 세팅 활용도와 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 라벨 MCQ 품질 지표에 대한 예측력이 뛰어난 것으로 나타났습니다.'}, {"What's New": '대조 학습 (contrastive learning)과 반사실적 증강 (counterfactual augmentation)을 활용하여 NLP 태스크에서 모델의 강인성을 향상시키는 방법을 제안합니다.', 'Technical Details': '기존의 데이터셋에서 counterfactual과 유사한 데이터를 자동으로 매치하거나 사람이 직접 추가하는 방식과 달리, 우리는 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과성을 감독하는 방법을 사용합니다.', 'Performance Highlights': '우리의 접근법은 대조 학습을 결합하여, 반사실적 강인성, 도메인 간 일반화 (cross-domain generalization), 희소 데이터로부터의 일반화 등 다양한 차원에서 상당한 개선을 달성한 것으로 나타났습니다.'}, {"What's New": '의사들의 임상 노트 초안 품질을 향상시키기 위한 새로운 기술을 제시하며, 새로운 의사 등록 시 제한된 데이터로도 기존 모델 재훈련 없이 지원할 수 있는 기법을 도입합니다.', 'Technical Details': '이 기술은 의사의 대화 스타일과 노트 선호도를 모델링하는 기능에 집중하며, 예시적으로 PEGASUS-X 모델을 사용해 훈련, 적응, 테스트의 세 가지 단계로 구성됩니다. 특히 훈련에서 임상 노트의 각 섹션별로 모델을 분리하여 학습시킵니다.', 'Performance Highlights': '이 기술은 History of Present Illness 섹션에서 ROUGE-2 점수를 13.8% 향상시키고, Physical Examination 섹션에서 88.6%, Assessment & Plan 섹션에서 50.8% 향상시켜 베이스라인을 뛰어넘는 성과를 보였습니다.'}]



### BeeManc at the PLABA Track of TAC-2023: Investigating LLMs and Controllable Attributes for Improving Biomedical Text Readability (https://arxiv.org/abs/2408.03871)
Comments:
          system report for PLABA-2023. arXiv admin note: substantial text overlap with arXiv:2309.13202

- **What's New**: 이번 연구에서는 자동 MCQ 생성 평가에서 기존 메트릭들이 교육적 가치를 평가하지 못하는 문제를 해결하기 위해, MCQ의 정답 가능성을 측정하는 새로운 평가 메트릭 KDA를 제안했습니다. KDA를 고안하여 학생의 지식을 평가하는 힘을 강화하고자 합니다. 또한, 최근 딥러닝 모델들의 높은 정확도에도 불구하고, 이를 위한 robust 대책으로 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 접근법을 제안했습니다. 마지막으로, PLABA2023 바이오메디컬 추상 요약 과제에서 다양한 LLM 모델을 이용하여 성능을 평가하고 최적화한 결과를 설명하였습니다.

- **Technical Details**: MCQ 평가를 위한 KDA(Knowledge Dependent Answerability)를 제안했으며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 KDA를 근사화했습니다. 또한, 대조 학습과 counterfactual augmentation을 통해 모델의 robust를 강화하는 방법을 제시했습니다. PLABA2023 과제에서는 T5, SciFive, BioGPT, BART와 같은 LLM 모델들을 미세 조정(fine-tuning)하여 최적화하고, Quantitative metric으로 SARI, BERTscore, BLEU, ROUGE를 사용했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 또한, 대조 학습 기반 접근법은 Attribution-based synthesis로 인한 task model bias에 덜 민감하여, counterfactual robustness, cross-domain generalization, 그리고 scarce data로부터의 일반화(Generalization)에서 현저한 향상을 달성했습니다. PLABA2023 과제에서는 BeeManc 팀이 자동 평가 SARI 점수에서 2위를, 인간 평가에서 BART-w-CTs가 문장 간단성 면에서 2위를 기록했습니다.



### Why transformers are obviously good models of languag (https://arxiv.org/abs/2408.03855)
- **What's New**: 언어가 어떻게 작동하는지 명확히 밝혀진 바 없지만, Transformers는 현재 가장 성공적인 언어 처리 모델로 꼽힙니다. 이 논문은 Transformers의 구조와 특정 언어 이론 간의 직접적 연결점을 강조합니다.

- **Technical Details**: Transformers의 성과는 그들이 대표하는 언어 이론이 더 철저히 검토되고 더 나아가서는 현재 이용 가능한 최선의 이론으로 고려되어야 함을 시사합니다. 단어 임베딩이 프로토타입 의미를 반영하고, Transformer의 self-attention 레이어는 문맥화된 의미를 업데이트하는 방식 등을 논의합니다.

- **Performance Highlights**: 이 논문은 Transformers가 다양한 NLP 작업에서 기존의 LSTM 기반 모델들보다 뛰어난 성능을 보여준다고 주장하며, 특히 문맥화된 의미 표현에서 큰 발전을 이루었다는 점을 강조합니다.



### Hate Speech Detection and Classification in Amharic Text with Deep Learning (https://arxiv.org/abs/2408.03849)
Comments:
          Dataset: this https URL

- **Paper 1: Automatic MCQ Generation Evaluation Improvement**: [{"What's New": "자동 다지선다형 질문(MCQ) 생성의 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안함으로써 기존의 BLEU, ROUGE, METEOR 평가가 놓치는 교육적 가치를 보완합니다."}, {'Technical Details': 'KDA는 대상 사실(target fact)에 대한 학생의 지식을 기반으로 MCQ의 답변 가능성(answerability)을 측정합니다. 학생 응답 데이터를 통해 KDA를 측정하고, 사전 학습된 언어 모델을 이용해 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제공합니다.'}, {'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 갖는다는 것을 확인했습니다. 또한, n-gram 기반의 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링 한 MCQ 품질 척도를 잘 예측할 수 있습니다.'}]

- **Paper 2: Enhancing NLP Model Robustness through Counterfactuals**: [{"What's New": '최근의 NLP 모델들이 초인간적인 정확도를 보여주고 있지만, spurious pattern에 의존하기 때문에 robustness(견고성)가 제한됩니다. 여기에 대응하기 위해 대조 학습과 반사실증강(contrastive learning and counterfactual augmentation)을 활용하는 새로운 접근 방식을 제안합니다.'}, {'Technical Details': '기존의 반사실증강 방법은 사람이 직접 반사실을 추가하거나, 모델이 데이터셋에서 비슷한 것을 찾는 방식이지만, 여전히 spurious correlation 문제를 안고 있습니다. 본 연구는 여러 개의 반사실을 생성하여 집합적 의사결정(collective decisions)을 통해 더 견고하게 인과관계를 파악합니다.'}, {'Performance Highlights': '실험 결과, 집합적 의사결정 방식을 통해 모델이 속기 쉬운 편향에서 덜 민감하였고, 다양한 측면(반사실 견고성, 도메인 간 일반화, 적은 데이터로부터의 일반화)에서 유의미한 개선을 이뤘습니다.'}]

- **Paper 3: Amharic Hate Speech Detection and Classification**: [{"What's New": '에티오피아의 다양한 민족 및 종교 그룹 간 갈등을 일으킬 수 있는 증오 발언 문제에 대응하기 위해 Amharic어 증오 발언 탐지 및 분류 모델을 개발했습니다.'}, {'Technical Details': '총 100명의 Amharic 원어민이 5천개의 소셜 미디어 포스트와 댓글을 인종, 종교, 성별, 비증오 발언 등 네 가지 카테고리로 라벨링 했습니다. 깊은 학습 모델 SBi-LSTM을 사용하여 텍스트를 분류합니다.'}, {'Performance Highlights': '모델은 94.8의 F1-score를 달성했으며, 데이터셋을 확장하고 최첨단 모델을 개발하여 성능을 더욱 개선할 계획입니다.'}]



### WalledEval: A Comprehensive Safety Evaluation Toolkit for Large Language Models (https://arxiv.org/abs/2408.03837)
Comments:
          Under review

- **What's New**: [{'title': 'Knowledge Dependent Answerability (KDA)', 'content': "최근 교육 평가용 MCQ(Multiple Choice Question) 생성 분야에서는 교육적 가치를 고려하지 않는 기존 메트릭(BLEU, ROUGE, METEOR)의 한계를 극복하기 위해 'Knowledge Dependent Answerability (KDA)'라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 특정 사실에 대한 학생의 지식을 직접 측정합니다."}, {'title': 'Robust AI with Contrastive Learning', 'content': '최근 NLP 태스크에서 deep model들이 높은 정확도를 보이지만, 겉보기에 그럴듯한 패턴에 의존하는 경향으로 인해 내구성이 제한됩니다. 본 논문에서는 contrastive learning과 counterfactual augmentation 기법을 적용해 모델의 내구성을 강화하는 방법을 제안합니다.'}, {'title': 'WalledEval AI Safety Toolkit', 'content': '새로운 AI 안전성 평가 툴킷인 WalledEval을 소개합니다. WalledEval은 다양한 언어 모델 (LLM)을 평가할 수 있으며, 35가지 이상의 안전성 벤치마크와 맞춤형 변형(mutators)을 포함하여 다차원적 안전성 평가를 제공합니다.'}]

- **Technical Details**: [{'title': 'KDA_disc와 KDA_cont', 'content': 'KDA를 기반으로 학생의 답변을 시뮬레이션하는 방법론인 KDA_disc와 KDA_cont를 제안합니다. 이 두 메트릭은 실제 학생들이 문제를 푸는 행동을 모방하며, 인간 연구를 통해 이들이 KDA 및 실제 교실 환경에서의 사용성과 강한 상관관계를 보임을 확인했습니다.'}, {'title': 'Collective Decisions for Robustness', 'content': '본 논문은 도메인 간 일반화와 희소 데이터에서의 일반화를 위한 집합적 의사 결정(collective decisions)을 도입하여 단어 간 인과관계를 보다 안정적으로 확보하는 방법을 소개합니다. 기존의 허위 상관관계(spurious correlations)에 영향을 받는 방법과 달리, 다수의 반사실(counterfactual)을 생성하고 이를 통해 robust한 평가를 수행합니다.'}, {'title': "WalledEval's Framework", 'content': 'WalledEval은 다양한 LLM과 협력할 수 있는 Python 기반 프레임워크입니다. LLM 모델과 Judge를 로드하고, 다양한 벤치마크 데이터를 활용해 LLM의 안전성을 평가할 수 있도록 설계되었습니다. 또한 맞춤형 변형(mutators)을 사용해 텍스트 스타일 변형에 대한 안전성 테스트를 실행합니다.'}]

- **Performance Highlights**: [{'title': 'Educational Value Assessment', 'content': 'KDA_disc와 KDA_cont를 이용한 MCQ 평가가 전문가 라벨링과 강한 상관관계를 보여 MCQ의 교육적 가치를 보다 정확히 평가할 수 있음이 입증되었습니다.'}, {'title': 'Robustness Improvements', 'content': '본 논문에서 제안한 방법은 counterfactual robustness, 도메인 간 일반화(cross-domain generalization), 및 희소 데이터에서의 일반화 측면에서 현저한 성능 향상을 입증했습니다.'}, {'title': "WalledGuard's Efficiency", 'content': 'WalledEval의 새로운 콘텐츠 모더레이션 도구인 WalledGuard는 기존의 도구에 비해 약 16배 작으면서도, 최고 성능의 LlamaGuard-2와 비교해 성능 저하 없이 우수한 성능을 보여주었습니다.'}]



### Generative Language Models with Retrieval Augmented Generation for Automated Short Answer Scoring (https://arxiv.org/abs/2408.03811)
Comments:
          20 pages, 2 figures

- **What's New**: 최근 연구에서는 다중 선택 질문(MCQ) 자동 생성을 개선하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 도입했습니다. 또한, 자연어 처리 과제에서 대조 학습과 반사 사실 증가 기법을 활용해 모델의 내구성을 향상시키는 방법과 최신 생성형 언어 모델(GLMs)을 활용한 자동 짧은 답변 점수 시스템을 제안했습니다.

- **Technical Details**: 첫 번째 연구는 BLEU, ROUGE, METEOR와 같은 기존의 n-gram 기반 유사성 메트릭이 교육적 가치를 평가하지 못하는 문제를 지적하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 두 번째 연구는 대조 학습과 여러 반사 사실을 합성하여 단어의 인과 관계를 더 잘 파악하는 방법을 제안했습니다. 세 번째 연구는 베르트(BERT)와 일렉트라(ELECTRA)와 같은 사전 학습된 인코더 모델을 활용한 정보 검색(IR) 및 RAG와 GLM을 결합한 파이프라인을 제안하여 교실 평가에 있어 점수 정확도를 높였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 메트릭과 결합하여 전문가가 라벨링한 MCQ 품질 측정치에 대한 예측력을 향상시켰습니다. 대조 학습 기법은 반사 사실 강건성, 교차 도메인 일반화, 소량 데이터 일반화에서 상당한 개선 효과를 보였습니다. GLM을 활용한 자동 짧은 답변 점수 시스템은 SemEval 2013 데이터셋에서 기존 방법보다 높은 성과를 보였으며, 특히 SCIENTSBANK 3-way와 2-way 태스크에서 유의미한 개선을 나타냈습니다.



### 'Finance Wizard' at the FinLLM Challenge Task: Financial Text Summarization (https://arxiv.org/abs/2408.03762)
- **AI Newsletter**: {"What's New": '최근 다수의 연구들은 다양한 관심 주제인 자동 MCQ 생성 및 deep model의 robustness 향상에 대한 새로운 접근 방식을 제안하였습니다. 또한 금융 텍스트 요약에 특화된 모델도 개발되었습니다.', 'Technical Details': {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사성 메트릭이 MCQ의 교육적 가치를 충분히 평가하지 못한다는 문제를 해결하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안했습니다. 이는 프리트레인된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 방식으로 MCQ의 대답 가능성을 측정합니다.', 'Deep Model Robustness': 'spurious pattern에 의존하는 deep models의 robustness 문제를 해결하기 위해 대비 학습(contrastive learning)과 counterfactual augmentation을 활용한 새로운 방법을 제안하였습니다. 이 방법은 여러 개의 counterfactual을 생성하고 이들에 대한 집합적 의사 결정(collective decisions)을 통해 더 robust한 인과관계를 평가할 수 있습니다.', 'Financial Text Summarization': "Finance Wizard 팀은 Llama3 8B 모델을 기반으로 하여 금융 텍스트 요약을 위해 특화된 모델을 개발하였습니다. 이 모델은 금융 관련 데이터를 계속해서 학습하고 다중 작업 지침 조정(multi-task instruction-tuning)을 통해 금융 텍스트 요약 작업에 특화된 '전문가' 모델로 변환했습니다."}, 'Performance Highlights': {'MCQ Evaluation': 'Human evaluation 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 이는 다양한 전문가 표시 MCQ 품질 측정에 대해 높은 예측력을 갖고 있음을 시사합니다.', 'Deep Model': '새로운 접근 방식이 기존의 task model 편향에 덜 민감하며 counterfactual robustness, cross-domain generalization, scarce data에 대한 일반화에서 상당한 개선을 이루었습니다.', 'Financial Model': 'Finance Wizard 팀의 FinLlama3_sum 모델은 ROUGE-1 점수 0.521로 금융 텍스트 요약 카테고리에서 3위를 기록하며 우수한 성과를 보였습니다.'}}



### Question Rephrasing for Quantifying Uncertainty in Large Language Models: Applications in Molecular Chemistry Tasks (https://arxiv.org/abs/2408.03732)
- **NewAIpaper_Newsletter**: [{"What's New": "교육적 평가 가치를 고려한 MCQ 자동 생성 평가 메트릭 'Knowledge Dependent Answerability(KDA)' 제안", 'Technical Details': '기존 평가 메트릭 (BLEU, ROUGE, METEOR)은 생성된 MCQ와 골드 샘플 간의 n-gram 유사성에 초점을 맞추지만, 교육적 가치를 평가하지 못한다. KDA는 대상 사실에 대한 학생의 지식을 평가할 수 있는 MCQ의 대답 가능성을 측정한다. 또한, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하여, pre-trained language models을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사한다.', 'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실과 유사한 환경에서 높은 사용성을 보여줌. KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합 시 전문가가 라벨링한 다양한 MCQ 품질 척도에 대해 높은 예측력을 보임.'}, {"What's New": 'Contrastive Learning 및 Counterfactual Augmentation을 활용한 NLP의 Robustness 향상 방안 제안', 'Technical Details': '기존 방법은 사람의 수작업이나 데이터셋에서 비슷한 counterfactual을 찾는 방식에 의존하여 여전히 spurious correlation 문제를 가진다. 제안된 방법은 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 각 단어들의 인과관계를 robust하게 슈퍼바이징한다.', 'Performance Highlights': '다양한 차원(대체 현실적 강건성, 도메인 간 일반화, 소량 데이터로부터의 일반화)에서 유의미한 성능 향상 달성.'}, {"What's New": 'LLM의 불확실성 평가를 위한 새로운 질문 재구성(Question Rephrasing) 기법 도입', 'Technical Details': '입력 불확실성을 평가하기 위해 동일한 질문을 다른 방식으로 재구성하고, 이를 LLM에 제출하여 답변의 일관성을 측정한다. 또한, 출력 불확실성을 측정하기 위해 동일한 질문을 반복적으로 쿼리하여 답변 분포를 분석한다. 이는 화학 분야의 속성 예측 및 반응 예측 태스크 예측에서 검증됨.', 'Performance Highlights': 'GPT-4는 질문 재구성에 민감하며, 출력 불확실성은 LLM의 응답 정확도와 신뢰성의 유효한 지표로 작용함.'}]



### Local Topology Measures of Contextual Language Model Latent Spaces With Applications to Dialogue Term Extraction (https://arxiv.org/abs/2408.03706)
Comments:
          Accepted as a long paper to SIGDIAL 2024. 9 pages, 2 figures, 3 tables

- **What's New**: 자동 다중 선택 문제(MCQ)의 생성은 교사의 평가 작업을 크게 줄일 수 있는 잠재력을 가지고 있습니다. 그러나 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 생성된 MCQ와 데이터셋의 골드 샘플 간의 n-그램 기반 유사성에 중점을 두고 있어 교육적 가치를 간과하고 있습니다. 이를 해결하기 위해, 연구팀은 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안했습니다. KDA는 학생의 지식을 활용하여 MCQ의 답변 가능성을 측정하는 데 중점을 둡니다.

- **Technical Details**: KDA는 학생 설문조사를 통해 학생들의 응답을 기반으로 측정됩니다. 연구팀은 미리 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안했습니다. 이 두 메트릭은 실제 강의실에서의 사용성과 강한 상관관계를 가지는 것으로 나타났습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 (1) KDA와 (2) 전문가들이 라벨링한 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지는 것으로 밝혀졌습니다. 또한, n-그램 기반 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가 라벨 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다.



### NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Tim (https://arxiv.org/abs/2408.03675)
Comments:
          Accepted by ACL 2024 (main conference, long paper)

- **What's New**: 네 가지 주요 연구가 요약되었습니다. 첫 번째 연구에서는 Multiple Choice Questions (MCQ)를 자동으로 생성하는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 두 번째 연구에서는 NLP 태스크에서 deep models의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 결합한 방법을 제안합니다. 세 번째 연구에서는 Large Language Models (LLMs)를 위한 효율적인 KV Cache 간소화 방법 NACL을 제안합니다.

- **Technical Details**: [{'KDA': '기존 BLEU, ROUGE, METEOR 메트릭의 한계를 넘어, 학생의 지식에 기반한 MCQ의 답변 가능성을 평가하는 새로운 자동 평가 메트릭 KDA를 제안합니다. 이는 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하는 방식으로 측정됩니다.'}, {'Robustness': "spurious patterns에 대한 의존성을 줄이기 위해 contrastive learning과 counterfactual augmentation을 활용합니다. 이는 기존 방법들과 달리 '여러 개의' counterfactual을 생성하고 집합적 의사결정을 통해 보다 robust한 모델을 만듭니다."}, {'NACL': '긴 컨텍스트 윈도우에서 발생하는 KV Cache의 큰 메모리 소비 문제를 해결하기 위해 NACL을 제안합니다. 이는 PROXY TOKENS EVICTION과 RANDOM EVICTION 전략을 결합하여 좀 더 정확한 attention score 통계를 사용하고 무작위성을 도입하여 중요한 토큰을 효율적으로 유지합니다.'}]

- **Performance Highlights**: [{'KDA_disc와 KDA_cont': 'Human studies를 통해 KDA_disc와 KDA_cont 메트릭이 실제 강의실 설정에서의 사용성과 강한 상관관계를 가짐을 입증하였습니다.'}, {'Robustness': '제안된 방법은 counterfactual robustness, cross-domain generalization 및 scarce data에 대한 일반화 측면에서 기존 방법보다 우수한 성능을 나타냈습니다.'}, {'NACL': 'NACL은 short-text 태스크에서 80%, long-text 태스크에서 75%의 성능 향상을 이루었고, 95% 이상의 성능을 유지하면서 KV Cache 공간을 최대 50% 줄였습니다.'}]



### mucAI at WojoodNER 2024: Arabic Named Entity Recognition with Nearest Neighbor Search (https://arxiv.org/abs/2408.03652)
- **What's New**: 교사가 학생 평가에 소요하는 시간을 획기적으로 줄일 수 있는 자동 MCQ 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안했습니다. 또한 최근 심층 모델들이 사람보다 높은 정확도를 보이지만 spurious 패턴에 취약한 문제를 개선하기 위해, 대조 학습(contrastive learning) 및 counterfactual augmentation을 활용하여 robustness를 향상시키는 방법을 연구했습니다. 아랍어 텍스트에 대한 Named Entity Recognition (NER)의 독특한 문제들을 해결하기 위해 Wojood NER Shared Task 2024에 제출된 Arabic KNN-NER 모델을 제안했습니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 MCQ의 대답 가능성을 측정하는 새로운 자동 평가 메트릭입니다. 이를 위해 pre-trained language models를 활용하여 학생의 문제 해결 행동을 모방하는 방법을 사용합니다. 다른 연구에서는 여러 개의 counterfactual을 생성하여 집합적 의사 결정(collective decision)을 통해 단어의 인과성을 robust하게 파악합니다. Arabic KNN-NER은 훈련된 데이터에서 KNN 검색을 통해 다른 레이블 확률 분포를 결합하여 확률 분포를 보강하는 방법론을 사용했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였으며, 전문가가 레이블한 고품질 MCQ 예측에서도 강력한 예측력을 보였습니다. 새로운 augmentation 방법론은 다양한 차원에서 상당한 성능 향상을 이루었고, 특히 counterfactual robustness, cross-domain generalization, sparse data에서의 generalization에서 최고의 성능을 보였습니다. Arabic KNN-NER 모델은 WojoodFine 데이터셋에서 Micro-F1 점수 91%를 기록하며, shared subtask 1 Flat NER의 리더보드에서 1위를 차지했습니다.



### CARE: A Clue-guided Assistant for CSRs to Read User Manuals (https://arxiv.org/abs/2408.03633)
- **What’s New**: 이번 연구에서는 학습 평가의 시간을 줄일 수 있는 자동 MCQ 생성 평가 메트릭과 온라인 고객 서비스를 위한 CARE라는 새로운 읽기 도우미를 소개합니다.

- **Technical Details**: MCQ 생성을 평가하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하며, 이는 학생의 지식을 측정하는 능력을 평가합니다. 두 가지 자동 메트릭 KDA_disc와 KDA_cont가 제안되었고, 이는 사전 학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방합니다. CARE는 이질적인 그래프를 사용해 사용자가 질문한 내용에 맞는 응답을 예측하고, 상세한 설명과 함께 응답 단서를 제공하여 고객 서비스 담당자가 매뉴얼에서 신속하게 적절한 응답을 찾을 수 있도록 도와줍니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 평가 결과에서 KDA와 높은 상관관계를 보였으며, 전문가가 라벨링한 실제 교실 내 사용성에서도 우수한 상관관계를 확인했습니다. CARE는 약 35%의 시간을 절약하면서도 높은 서비스 품질 (>0.75 ICC 점수)을 유지하는 것으로 입증되었습니다.



### PAGED: A Benchmark for Procedural Graphs Extraction from Documents (https://arxiv.org/abs/2408.03630)
- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성 방식의 평가에서 기존의 n-gram 기반 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 고려하지 않는다는 문제를 해결하기 위해, 연구진은 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안하였습니다. 또한, KDA를 기반으로 두 가지 자동 평가 메트릭, KDA_disc 및 KDA_cont를 소개했습니다.

- **Technical Details**: 제안한 KDA는 학생의 타겟 지식에 대한 응답 가능성을 측정합니다. 이를 위해 먼저 학생 반응 기반으로 KDA를 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 인적 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서 전문가들이 레이블링한 사용성과 강한 상관관계를 갖고 있음을 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반의 유사성 메트릭과 결합하여 다양한 전문가 레이블링 MCQ 품질 측정에 대해 높은 예측력을 보였습니다. 이는 새로운 메트릭이 교육적 평가의 본질을 더 잘 반영한다는 것을 의미합니다.



### Improving the quality of Persian clinical text with a novel spelling correction system (https://arxiv.org/abs/2408.03622)
- **What's New**: 새로운 MCQ 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안함으로써 기존의 BLEU, ROUGE, METEOR과 같은 평가 방법들이 놓치고 있는 교육적 가치를 확인하고자 합니다. 또한, 최근 NLP 태스크에서 spurious pattern 의존성 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 새로운 접근법을 제시합니다. 마지막으로, 페르시아어 임상 텍스트에서 철자 오류를 감지하고 교정하기 위한 혁신적인 방법을 개발했습니다.

- **Technical Details**: [{'Multiple Choice Questions (MCQ)': '기존의 n-gram 기반 평가 메트릭을 대체하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다. KDA는 학생들이 대상 사실에 대한 지식을 기반으로 MCQ를 답변할 수 있는 능력을 측정합니다. Human survey를 통해 KDA를 측정하고, pre-trained language models를 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다.', 'Robustness in NLP Models': '기존의 방법들이 사람이 또는 모델이 counterfactual을 추가하는 방식의 한계를 극복하기 위해, 여러 개의 counterfactual을 생성하고 collective decision-making을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안합니다. 이를 통해 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 성능이 향상되었습니다.', 'Persian Clinical Text Spelling Correction': '페르시아어 임상 텍스트에서 철자 오류를 감지하고 교정하기 위해 최신 pretrained model과 orthographic similarity matching algorithm인 PERTO를 사용했습니다. 특히, PERTO 알고리즘은 시각적 유사성을 사용하여 교정 후보를 랭킹합니다.'}]

- **Performance Highlights**: [{'MCQ Evaluation': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 지님을 human studies에서 확인했습니다. n-gram 기반 메트릭과 결합할 경우 다양한 전문가가 라벨링한 MCQ 품질 측정에 대한 강한 예측력을 보였습니다.', 'NLP Robustness': '집합적 의사결정을 통해 attribution-based synthesis의 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 성능이 크게 향상되었습니다.', 'Persian Clinical Spelling Correction': '비단어 오류 교정에서 PERTO 알고리즘을 사용했을 때 F1-Score 90.0%를 달성했습니다. 실제 단어 오류 감지에서도 최고 F1-Score 90.6%를, 실제 단어 오류 교정에서는 최고 F1-Score 91.5%를 기록했습니다.'}]



### A Logical Fallacy-Informed Framework for Argument Generation (https://arxiv.org/abs/2408.03618)
- **What's New**: 최신 연구들은 자동 다지선다형 질문 (MCQ) 생성과 논리적인 주장에서의 언어 모델 성능을 개선하기 위해 새로운 접근법을 제안하고 있습니다. MCQ 생성에서는 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 도입했고, 논리적인 주장 생성에서는 논리적 오류를 인식하고 방지하는 FIPO 프레임워크를 선보였습니다.

- **Technical Details**: MCQ 생성에 대해, 기존의 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 평가하지 못하는 문제가 있습니다. 이를 해결하기 위해 지식 종속 가능성(KDA) 메트릭이 도입되었으며, 학생의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont를 제안했습니다. 논리적인 주장의 생성에서는 기존의 대안사실 (counterfactual) 학습에 의존하지 않고, 여러 개의 대안사실을 생성하여 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 더 견고하게 파악하는 방법을 채택했습니다. 또한, FIPO 프레임워크는 LLM이 논리적 오류를 최소화할 수 있도록 선호도 최적화(preference optimization) 방법을 사용하고, fallacy 카테고리에 대한 세부 정보를 포착하는 분류 손실을 포함합니다.

- **Performance Highlights**: MCQ 생성에서 KDA 기반 메트릭은 기존의 n-gram 기반 메트릭과 결합하여 전문가가 평가한 MCQ 품질 측정에서 높은 예측력을 보였습니다. 실험 결과, KDA_disc와 KDA_cont는 실제 교육 현장에서 높은 사용성을 가졌습니다. 논리적인 주장에서, FIPO 프레임워크는 논리적 오류를 최대 17.5%까지 줄였으며, 인간 평가 결과에 따르면 FIPO를 통한 주장이 기존의 미세 조정된 모델과 DPO 같은 선호도 최적화 방법보다 뛰어남을 보였습니다.



### Is Child-Directed Speech Effective Training Data for Language Models? (https://arxiv.org/abs/2408.03617)
Comments:
          Preprint. Code and data will be released soon

- **title**: 최근 아카이브 논문 요약: 다중 선택 문제(MCQ)와 학습 모델의 강건성 및 아동 언어 학습 데이터

- **date_range**: 2023년 10월

- **topics**: [{"What's New": "자동으로 다중 선택 문제 (MCQ)를 생성하는 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)' 도입. 이는 MCQ의 목표 사실에 대한 학생의 지식을 평가하는 능력을 측정.", 'Technical Details': '이 메트릭은 학생 응답을 기반으로 KDA를 측정하며, 사전 훈련된 언어 모델을 활용해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안. 인간 평가와의 강한 상관 관계를 통해 강의실에서의 실제 사용성을 검증.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 전문가 레이블 MCQ 품질 측정치와 강한 예측력을 보임.'}, {"What's New": 'NLP 태스크에서 모델의 강건성 향상을 위한 대조 학습과 반사실적 증가 (counterfactual augmentation) 도입.', 'Technical Details': '기존 방식과 달리 다수의 반사실적(countefactual)을 생성하고 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과 관계를 분석. 이는 spurious correlation에 덜 민감하게 하여, 강건성을 높임.', 'Performance Highlights': '반사실적 강건성, 교차 도메인 일반화, 희소 데이터에서의 일반화 부문에서 성능 향상.'}, {"What's New": '아동이 언어를 효율적으로 학습하는 이유에 대한 분석. 아동에게 제공되는 데이터의 특성이 언어 모델링 목표에 어떻게 기여하는지 조사.', 'Technical Details': 'GPT-2 모델을 2천9백만 단어의 아동 지향 언어와 인위적 데이터셋(TinyDialogues)으로 훈련시킴. 개발적으로 영감을 받은 평가를 통해 구문 및 의미 지식을 비교.', 'Performance Highlights': '아동 언어 입력의 지역적 속성(local properties)이 모델 성능에 영향을 미침. 그러나 아동 언어 학습의 효율성은 데이터의 품질뿐만 아니라 상호작용에서도 기인.'}]

- **further_reading**: 상세한 논문 내용은 arXiv에서 확인하십시오. [논문 링크 (insert link here)]



### A Comparison of LLM Finetuning Methods & Evaluation Metrics with Travel Chatbot Use Cas (https://arxiv.org/abs/2408.03562)
- **What's New**: 이번 연구에서는 MCQ (Multiple Choice Questions) 자동 생성 평가에서 기존의 BLEU, ROUGE, METEOR 메트릭들이 교육적 가치를 평가하지 못하는 문제에 착안하여, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. KDA는 주어진 대상 사실에 대한 학생의 지식에 기반하여 MCQ의 답변 가능성을 평가합니다.

- **Technical Details**: KDA를 측정하기 위해 인간 설문조사에서 학생의 반응을 바탕으로 KDA를 계산하는 방법을 제시하였으며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하였습니다. 이 메트릭들은 실제 강의실에서 사용성을 평가받아 강한 상관관계를 보였습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc 및 KDA_cont가 실제 강의실 평가와 강한 상관관계를 가짐을 보여주었습니다. n-gram 기반 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 다양한 전문가 레이블 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.



### Empirical Analysis of Large Vision-Language Models against Goal Hijacking via Visual Prompt Injection (https://arxiv.org/abs/2408.03554)
Comments:
          8 pages, 6 figures, Accepted to NAACL 2024 SRW

- **What's New**: 새로운 자동 평가 지표로서 Knowledge Dependent Answerability(KDA)을 도입해 MCQ의 교육적 효용성을 측정한다는 제안이 주목할 만합니다. KDA는 BLEU, ROUGE, METEOR 등의 기존 메트릭이 평가하지 못한 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: KDA 측정은 학생 응답을 기반으로 하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 통해 KDA를 근사합니다. 이는 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 이번 연구는 인간 조사 통해 KDA_disc와 KDA_cont가 강의실 설정에서 실제 사용과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 경우, 전문가들에 의해 라벨링된 다양한 MCQ 품질 척도에 대해 강한 예측력을 가집니다.



### Unlocking the Non-Native Language Context Limitation: Native Language Prompting Facilitates Knowledge Elicitation (https://arxiv.org/abs/2408.03544)
- **What's New**: 새로운 평가법을 제안한 3개의 논문이 발표되었습니다. 첫 번째 논문은 지식 종속가능성(KDA)을 사용하여 자동 Multiple Choice Questions(MCQ) 생성의 교육적 가치를 측정하는 방법을 제안합니다. 두 번째 논문은 대조적 학습과 반사실적 데이터 증강을 통해 NLP 태스크의 강건성을 향상시키는 방법을 다룹니다. 세 번째 논문은 다국어 대형 언어 모델(MLLM)이 비주류 언어로 된 질문에 효과적으로 답하기 위해 Native Language Prompting(NatLan)을 사용하는 방법을 제안합니다.

- **Technical Details**: 첫 번째 논문에서는 KDA라는 새로운 메트릭을 소개하여, 학생들의 반응을 기반으로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 논문에서는 여러 개의 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 세 번째 논문에서는 Native Language Prompting을 통해 MLLM이 주류 언어의 풍부한 지식을 활용하여 비주류 언어 질문에 더 나은 답변을 제공할 수 있도록 하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트에서 높은 상관관계를 보였습니다. 반사실적 데이터 증강을 통해 다차원에서 성능이 크게 향상되었습니다: 반사실적 강건성, 크로스 도메인 일반화, 데이터 부족 상황에서의 일반화. NatLan은 C-Eval benchmark에서 최대 10.1%의 평균 정확도 향상과 5.0%의 hard-level subset에서의 성능 향상을 달성했습니다.



### EXAONE 3.0 7.8B Instruction Tuned Language Mod (https://arxiv.org/abs/2408.03541)
- **What's New**: 이번 뉴스레터에서는 교육용 객관식 질문 생성, NLP 모델의 robust성을 높이기 위한 방법론, 그리고 LG AI Research에서 개발한 EXAONE 3.0 공개 모델 발표를 다룬 세 가지 논문을 소개합니다.

- **Papers**: [{'title': '교육용 객관식 질문 자동 생성에 대한 평가 메트릭 개선', 'Research Highlight': [{"What's New": '기존의 평가 메트릭 (BLEU, ROUGE, METEOR) 는 교육적 가치를 고려하지 않고 단순히 단어의 유사성만 평가한다는 한계가 있습니다. 이를 해결하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.', 'Technical Details': 'KDA는 학생들이 대상 사실에 대한 지식을 바탕으로 객관식 질문에 답변할 수 있는 능력을 측정합니다. 이를 인공지능 모델로 평가하기 위해 KDA_disc와 KDA_cont 두 가지 자동 메트릭을 제안하며, 사전 학습된 언어 모델을 활용해 학생들이 문제를 푸는 행태를 모방합니다.', 'Performance Highlights': 'Human survey를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 이루고 있음을 입증했습니다. 또한, n-gram 기반 평가 메트릭과 결합할 때, 전문가가 라벨링한 MCQ 품질에 대한 예측력이 높아지는 것으로 나타났습니다.'}]}, {'title': 'Deep Models의 Robustness를 위한 Contrastive Learning 및 Counterfactual Augmentation', 'Research Highlight': [{"What's New": '최근의 deep model들이 매우 높은 정확성을 보였지만, spurious pattern에 의존해 robustness가 제한된다는 문제점을 다루고 있습니다. 이를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용합니다.', 'Technical Details': '기존 방법들은 사람이 직접 counterfactual을 만들거나 모델이 데이터셋에서 비슷한 것을 발견해야 했으나, 우리의 접근법은 여러 개의 counterfactual을 생성하고 collective decision을 통해 단어들의 인과관계를 robust하게 파악합니다.', 'Performance Highlights': '우리 접근법은 attribution-based synthesis에서 발생하는 task model bias에 덜 민감하며, 다양한 차원에서 성능 향상을 보여주었습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization.'}]}, {'title': 'EXAONE 3.0: LG AI 연구소의 Instruction-Tuned 대규모 언어 모델 공개', 'Research Highlight': [{"What's New": 'LG AI 연구소는 EXAONE 3.0 instruction-tuned 언어 모델을 발표하며, 7.8B 모델을 비상업적인 연구 목적으로 공개했습니다. 이 모델은 다양한 공공 및 내부 벤치마크에서 높은 성능을 보여줍니다.', 'Technical Details': 'EXAONE 3.0은 bilingual 지원을 위해 효율적인 tokenization과 다양한 데이터셋에 대한 광범위한 사전 학습 및 고급 post-training 기술을 적용하여, 강사 지침을 잘 따르는 모델로 설계되었습니다. ', 'Performance Highlights': '성능 평가 결과, EXAONE 3.0은 영어와 한국어 모두에서 뛰어난 성능을 보였으며, 특히 한국어에서 두드러진 성과를 보여주었습니다. EXAONE 3.0은 실제 환경에서도 높은 실효성을 자랑하며, 전문가 수준의 인공지능 발전에 기여할 것으로 기대됩니다.'}]}]



### EgyBERT: A Large Language Model Pretrained on Egyptian Dialect Corpora (https://arxiv.org/abs/2408.03524)
- **What's New**: 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'가 도입되었으며, 이는 기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 무시하는 문제를 해결하려는 시도입니다. 또한 'KDA_disc'와 'KDA_cont'라는 두 개의 자동 평가 메트릭이 제안되어, 사전 학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방하는 방식으로 KDA를 근사합니다.

- **Technical Details**: KDA는 학생 응답 기반으로 측정되며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 KDA를 흉내냅니다. 본 연구는 현 교실 환경에서의 사용성을 분석하고, KDA_disc와 KDA_cont가 인간 전문가의 평가와 높은 상관관계가 있음을 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합했을 때, 다양한 전문가 평가 MCQ 품질 측정에 대한 높은 예측력을 보여줍니다.



### 1.5-Pints Technical Report: Pretraining in Days, Not Months -- Your Language Model Thrives on Quality Data (https://arxiv.org/abs/2408.03506)
Comments:
          Technical Report for 1.5-Pints

- **What's New**: 새로운 MCQ 생성 평가 메트릭인 Knowledge Dependent Answerability (KDA)가 제안되었습니다. 이 메트릭은 생성된 MCQ가 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. 또한 '1.5-Pints'라는 라는 새로운 Pre-trained Language Model이 9일 만에 높은 성능을 발휘하며 주목받고 있습니다.

- **Technical Details**: KDA는 학생들에게 대상 사실에 대한 지식이 있음을 전제로 MCQ의 대답 가능성(answerability)을 평가합니다. 이와 관련하여 KDA_disc, KDA_cont 두 가지 자동 평가 메트릭 도입됩니다. 반면, '1.5-Pints' 모델은 57 billion tokens으로 구성된 고품질 데이터셋을 사용하여 8 A100 GPU에서 9일 동안 학습되었습니다. Llama-2 아키텍처와 수정된 Mistral tokenizer가 사용되었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 세팅에서 전문가 라벨링과 강한 상관관계를 보였으며, BLEU, ROUGE, METEOR과 결합하여 더 높은 예측 성능을 보였습니다. '1.5-Pints'는 Apple의 OpenELM과 Microsoft의 Phi를 능가했으며, MT-Bench 평가에서 최고의 성능을 기록했습니다. 이 모델은 더 긴 문서 요약과 다중 대화가 가능한 2K 및 16K 컨텍스트 윈도우 버전으로 제공됩니다.



### Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation (https://arxiv.org/abs/2408.03505)
- **What's New**: 최근 다중 선택 질문 (Multiple Choice Questions, MCQ)의 자동 생성 기술이 교육자들의 평가 시간을 절감할 수 있는 잠재력을 가지고 있지만, 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 간과하고 있습니다. 이에 따라 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안했습니다. 이는 학생이 해당 사실(Target Fact)에 대한 지식을 바탕으로 문제에 답할 수 있는지를 측정합니다.

- **Technical Details**: KDA는 인간 설문조사를 통해 학생 응답 기반으로 측정되며, 이를 자동으로 평가하기 위해 pre-trained 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안했습니다. Human studies를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 라벨링한 실제 교실 사용성과 강한 상관관계가 있음을 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에 대해 강력한 예측력을 보여주었습니다.



### Logistic Regression makes small LLMs strong and explainable "tens-of-shot" classifiers (https://arxiv.org/abs/2408.03414)
Comments:
          41 pages, 24 figures

- **What's New**: 교육적 가치를 평가하기 위해 새로운 자동 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안했습니다. 이는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 또한 대조학습(contrastive learning)과 반사실적 강화(counterfactual augmentation)를 활용하여 NLP 모델의 강인성(robustness)을 높이는 방법을 소개했습니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭으로 근사화합니다. 이 메트릭들은 미리 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 반사실적 강화는 인간이 데이터셋에 반사실을 추가하거나 기계가 자동으로 유사 반사실을 찾는 대신, 여러 개의 반사실을 생성하여 집합적 의사 결정을 통해 단어들의 인과 관계를 효과적으로 파악하는 방법을 사용합니다. 또한, 소형 로컬 생성 모델(local generative models)의 임베딩에 기반한 페널라이즈드 로지스틱 회귀(PLR)를 통해 상용 대형 모델(GPT-4)과 동등하거나 더 나은 성능을 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 교실 세트에서의 사용성과 강한 상관관계를 가지고 있습니다. 우리의 접근 방식은 다양한 차원에서 상당한 개선을 이루었으며, 특히 반사실 강인성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 그리고 희귀 데이터에서의 일반화(generalization from scarce data)에서 두드러집니다. 또한, 약 60-75개의 학습 샘플만으로 상용 대형 모델인 GPT-4보다 우수한 성능을 달성할 수 있었으며, 이 방법은 텍스트 분류 결정의 설명 가능성도 제공합니다.



### ULLME: A Unified Framework for Large Language Model Embeddings with Generation-Augmented Learning (https://arxiv.org/abs/2408.03402)
- **What's New**: 최근 연구는 자동으로 다중 선택 질문(Multiple Choice Questions, MCQ)을 생성하는 시스템의 평가 메트릭이 학생의 지식을 평가하는 능력을 적절하게 반영하지 못한다는 문제를 지적합니다. 이를 해결하기 위해 우리가 제안한 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭은 MCQ의 답변 가능성을 측정하고, 학생이 해당 지식을 얼마나 잘 이해하고 있는지를 평가합니다. 또한, 새로운 자동 평가 메트릭인 KDA_disc와 KDA_cont를 도입하여, 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방함으로써 KDA를 근사합니다.

- **Technical Details**: 자동 MCQ 생성의 기존 평가 메트릭(예: BLEU, ROUGE, METEOR)은 기본적으로 n-gram 기반의 유사성에 의존하여 MCQ의 교육적 가치를 적절히 평가하지 못합니다. KDA 메트릭은 학생이 주어진 지식을 기반으로 정답을 맞출 수 있는지를 평가함으로써 이를 개선합니다. KDA_disc와 KDA_cont는 학생들의 응답 성향을 모방하는 사전 학습된 언어 모델을 이용해 개발되었습니다.

- **Performance Highlights**: Human 평가 연구를 통해, KDA_disc와 KDA_cont가 KDA와 실제 강의실 설정에서의 사용성 측면에서 강한 상관관계를 가지는 것으로 나타났습니다. 또한, 이 메트릭은 n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 평가 지표 예측에서도 강력한 성능을 보였습니다.



### CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases (https://arxiv.org/abs/2408.03910)
Comments:
          work in progress

- **What's New**: [{'title': '지식 종속 가능성 평가 지표 (KDA) 제안', 'description': '기존의 MCQ 생성 평가 메트릭은 교육적 가치를 반영하지 못하는 문제를 해결하기 위해, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 도입했습니다. 이 메트릭은 MCQ의 대답 가능성을 학생의 지식 수준에 따라 평가합니다.'}, {'title': 'Contrastive Learning과 Counterfactual Augmentation의 결합', 'description': '최근 NLP 모델들이 spurious pattern에 의존해 robustness가 제한적인 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 결합한 새로운 방법을 제안했습니다. 이는 복수의 counterfactual을 생성하여 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악합니다.'}, {'title': '그래프 데이터베이스 인터페이스를 활용한 LLM-코드베이스 통합', 'description': '대규모 코드 리포지토리와 상호작용하는 새로운 시스템 \x0cramework를 도입했습니다. 이는 LLM과 코드 리포지토리를 그래프 데이터베이스를 통해 연결함으로써, 코드 구조를 이해하고 효율적인 코드 탐색을 가능하게 합니다.'}]

- **Technical Details**: [{'title': 'KDA 평가 메트릭', 'description': 'KDA는 학생 응답을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입해 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.'}, {'title': 'Contrastive Learning과 Counterfactual Augmentation', 'description': '기존의 사람이나 기계가 데이터셋에서 counterfactual을 생성하지 않고, 여러 개의 counterfactual을 자동으로 생성하여 집합적 예측 결과를 통해 단어의 인과관계를 보다 robust하게 감독합니다.'}, {'title': '\x0cramework 시스템', 'description': '\x0cramework는 그래프 데이터베이스의 구조적 속성과 그래프 쿼리 언어의 유연성을 활용하여, 코드 리포지토리로부터 정적 분석을 통해 코드 그래프를 추출하고, LLM이 쿼리를 구성하고 실행하여 정확하고 코드 구조를 인지한 컨텍스트 검색과 코드 탐색을 가능하게 합니다.'}]

- **Performance Highlights**: [{'title': 'KDA_disc와 KDA_cont', 'description': 'KDA_disc와 KDA_cont는 각각 KDA와 실제 강의실에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합 시 다양한 전문가가 라벨링한 MCQ 품질 측정에서 강한 예측력을 보여줍니다.'}, {'title': '대화식 의사결정(Multi-hop Reasoning)', 'description': '제안된 방법은 기존 유사성 기반 검색과 수작업 도구 및 API 방식의 한계를 극복하여, 더 높은 회수율과 정확도를 보이며, 복잡한 코드 구조와 긴 시퀀스 reasoning에서 상당한 성능 향상을 이룹니다.'}, {'title': '세 가지 벤치마크에서의 성능', 'description': 'CrossCodeEval, SWE-bench, EvoCodeBench 세 가지 벤치마크에서 \x0cramework는 통합된 그래프 데이터베이스 스키마와 간단한 워크플로 디자인을 통해 경쟁력 있는 성능을 보여줍니다. 특히, 더 발전된 LLM을 탑재했을 때 성능이 더욱 향상됩니다.'}]



### Leveraging Variation Theory in Counterfactual Data Augmentation for Optimized Active Learning (https://arxiv.org/abs/2408.03819)
- **What's New**: 이번 주 AI 뉴스레터에는 다양한 연구 논문들이 자동 MCQ 생성, 대조 학습과 반사실적 데이터 증강, 그리고 활성 학습(AL)에 대한 최신 연구를 다루고 있습니다.

- **Technical Details**: 첫 번째 논문에서는 자동 MCQ 생성의 평가 메트릭으로 BLEU, ROUGE, METEOR가 교육적 가치를 평가하는 데 한계가 있음을 지적하고, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. 두 번째 논문에서는 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)이 기존 방법보다 더 강력한 널리퍼짐 견고성(spurious robustness) 문제를 해결하는 방법을 제안합니다. 세 번째 논문에서는 Variation Theory를 적용한 반사실적 데이터 증강을 통해 AL의 데이터 효율성(data efficiency)을 높이고 초기 시작 문제(cold start problem)를 개선하는 방법을 논의합니다.

- **Performance Highlights**: 첫 번째 연구에서는 KDA_disc와 KDA_cont가 강의실 세트에서 높은 상관관계를 보임을 입증하였습니다. 두 번째 연구는 여러 대조적 예시 생성과 집합적 의사 결정을 통한 방법이 다양한 차원에서 더 견고하고 일반화 성능이 뛰어남을 보였습니다. 세 번째 연구는 텍스트 분류 예시 도메인에서 반사실적 데이터가 적은 어노테이션 데이터로 더 높은 성능을 실현함을 보여주었고, 어노테이션 데이터가 많아질수록 그 영향이 줄어듦을 확인하였습니다.



### Large Language Models for Base Station Siting: Intelligent Deployment based on Prompt or Agen (https://arxiv.org/abs/2408.03631)
- **What's New**: 이 연구에서는 MCQ(다지선다형 질문) 자동 생성의 교육적 가치를 평가하지 못하는 기존의 평가 메트릭 문제를 해결하기 위해 새로운 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안합니다. 또한, 최근 NLP에서 높은 정확성을 보이는 deep models의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 결합한 새로운 방법을 제안합니다. 마지막으로, 전통적인 기지국 배치(base station siting) 접근 방법의 복잡성을 줄이기 위해 대형 언어 모델(Large Language Models)의 잠재력을 활용한 혁신적인 최적화 프레임워크를 소개합니다.

- **Technical Details**: ['MCQ 자동 생성에 대한 새로운 평가 지표인 KDA는 대상 사실에 대한 지식을 통해 MCQ의 대답 가능성을 측정합니다. KDA는 student responses와 사전 훈련된 언어 모델을 활용하여 평가 지표를 자동화합니다.', 'deep models의 robust를 증가시키기 위해 우리는 여러 개의 반사실적 사례를 합성하고 집합적 의사 결정을 통해 용어의 인과성을 체계적으로 감독하는 방법을 제안합니다.', '기지국 배치(base station siting)에 대한 새로운 접근법으로, Prompt-optimized LLM (PoL), Human-in-the-Loop LLM (HiLL), LLM-empowered autonomous BSS agent (LaBa), Cooperative multiple LLM-based autonomous BSS agents (CLaBa)의 네 가지 전략을 제안합니다. 이 전략들은 최적의 기지국 위치를 자동으로 결정하여 네트워크 커버리지를 향상시킵니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 학급 환경에서 신뢰할 수 있는 MCQ 품질 평가 지표로 확인되었습니다.', '제안된 robust learning 방법은 다양한 측면에서 중요한 개선을 이루었습니다: 반사실적 로버스트니스, 크로스 도메인 일반화, 데이터 부족 시의 일반화.', '실험 결과, LLM 기반 접근법은 기존의 기지국 배치 방법에 비해 더욱 효율적이고 비용 효과적이며 신뢰할 수 있는 네트워크 배포를 가능하게 했습니다.']



### Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks (https://arxiv.org/abs/2408.03615)
Comments:
          30 pages, 13 figures

- **What's New**: 자동 다중 선택 질문(Multiple Choice Questions, MCQ) 생성의 평가 메트릭으로서 새로운 자동 평가 메트릭인 KDA(지식 종속 가능성)를 제안하였습니다. 또한, 새로운 하이브리드 다중모달 메모리(구조화된 지식 그래프 및 요약된 경험 저장소) 모듈을 소개하여, 오픈월드에서 복잡한 장기 과제를 처리할 수 있는 멀티모달 구성 에이전트 Optimus-1을 개발했습니다.

- **Technical Details**: MCQ 생성의 평가 메트릭으로 기존 BLEU, ROUGE, METEOR 대신 KDA_disc와 KDA_cont를 제안하여 학생의 대상 사실에 대한 지식 평가 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. Optimus-1은 하이브리드 다중모달 메모리 모듈을 기반으로 Knowledge-Guided Planner(지식 기반 계획기), Experience-Driven Reflector(경험 기반 반사기), Action Controller(행동 컨트롤러)로 구성된 에이전트입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가를 통해 실제 강의실 설정에서의 사용성과 강한 상관관계를 보였습니다. Optimus-1은 장기 과제 벤치마크에서 30%의 성능 향상을 보여주었고, 하이브리드 다중모달 메모리를 통해 GPT-4V보다 2~6배의 성능 향상을 달성했습니다.



### EnJa: Ensemble Jailbreak on Large Language Models (https://arxiv.org/abs/2408.03603)
- **What's New**: 이번 연구에서는 최근 대형 언어 모델(LLM)들이 안전이 중요한 애플리케이션에 점점 더 많이 활용됨에 따라, 이들의 잠재적 jailbreak 공격에 대한 취약성이 커지고 있다는 문제를 해결하려 합니다. 연구에서는 기존의 prompt-level 공격과 token-level 공격을 결합해, 보다 강력한 하이브리드 jailbreak 공격 기법 'Ensemble Jailbreak'를 제안합니다.

- **Technical Details**: 연구팀은 EnJa 공격을 통해, prompt-level jailbreak를 이용해 해로운 지시사항을 숨기고, gradient 기반 공격 기법을 활용해 공격 성공률을 높이며, 이 두 가지 jailbreak 공격을 template 기반 연결 기법으로 통합합니다. 특히, prompt-level 기법은 스토리나 논리를 구성하여 안전 정렬(Alignment)을 무력화시키고, token-level 공격은 gradient 방법을 사용하여 적대적 토큰을 찾아내는 방식입니다.

- **Performance Highlights**: EnJa 공격 기법은 기존의 개별 jailbreak 공격보다 훨씬 적은 쿼리로도 높은 공격 성공률을 달성하는 데 성공했습니다. 이를 통해, 여러 정렬된 모델에서 매우 강력한 공격 성능을 보여줍니다.



### Teach CLIP to Develop a Number Sense for Ordinal Regression (https://arxiv.org/abs/2408.03574)
Comments:
          Accepted by ECCV 2024

- **What's New**: 최근 논문에서는 기존의 MCQ(Multiple Choice Questions) 자동 생성 평가 메트릭의 한계를 극복하기 위해 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하였습니다. KDA는 학생이 대상 사실에 대해 얼마나 잘 이해하고 있는지를 평가할 수 있는 MCQ의 답변 가능성을 측정합니다.

- **Technical Details**: 이 논문은 먼저 학생 응답을 기반으로 KDA를 측정하는 방법을 보여줍니다. 그 후, 사전 학습된 언어 모델(pre-trained language models)을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서의 사용성과 강한 상관관계를 보임을 입증하였습니다. 또한, 이 메트릭을 n-gram 기반 유사성 메트릭과 결합할 경우 전문가가 라벨링한 MCQ 품질 측정치에 대해 강한 예측력을 가짐을 보였습니다.



### Active Testing of Large Language Model via Multi-Stage Sampling (https://arxiv.org/abs/2408.03573)
- **What's New**: 이번 뉴스레터에서는 자동 다지 선택형 질문 (Multiple Choice Questions, MCQ) 생성 평가와 관련된 새로운 연구, NLP 태스크의 robust 감응성을 향상시키기 위한 대조 학습과 반사실적 증강 (counterfactual augmentation)의 활용, 그리고 대형 언어 모델 (Large Language Models, LLM) 평가를 효율적으로 수행하기 위한 새로운 액티브 테스트 프레임워크를 소개합니다.

- **Technical Details**: [{"What's New": '지식 종속 가능성(Knowledge Dependent Answerability, KDA)라는 새로운 평가 메트릭을 제안하여, 기존 BLEU, ROUGE, METEOR 등이 고려하지 않았던 MCQ의 교육적 가치를 평가합니다.', 'Technical Details': 'KDA 메트릭은 학생이 대상 사실을 알고 있을 때 MCQ의 대답 가능성을 측정합니다. 이를 위해 인간 설문을 통해 학생 응답을 기반으로 KDA를 측정하는 방법을 먼저 보여주고, 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방함으로써 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 인간 연구를 통해 실제 강의실 세트의 사용성과 강한 상관관계를 가지고 있음을 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 전문가들이 라벨링한 MCQ 품질 측정치에 대해 강력한 예측력을 가지는 것으로 나타났습니다.'}, {"What's New": '대조 학습과 반사실적 증강 (counterfactual augmentation)을 통해 NLP 태스크의 모델 감응성을 향상시키는 방안을 제시합니다.', 'Technical Details': "기존 방법과 달리, 우리는 '여러 개'의 반사실적 샘플을 생성하고, 이 집합에 대한 예측 분포로부터 집합적 의사결정을 통해 각 단어의 인과관계를 robust하게 슈퍼바이즈하는 방법을 제안합니다.", 'Performance Highlights': '실험 결과, 이 접근법은 다양한 차원에서 유의미한 개선을 이루었으며, 특히 반사실적 로버스트넷(countrafactual robustness), 도메인 간 일반화(cross-domain generalization), 및 드문 데이터 일반화(generalization from scarce data)에서 성과를 보였습니다.'}, {"What's New": '효율적인 LLM 평가법인 AcTracer를 소개합니다.', 'Technical Details': 'AcTracer는 풀 기반 멀티 스테이지 액티브 선택을 통해 테스트 샘플링 과정을 안내하는 활성화된 테스트 프레임워크로, LLM의 내부 상태와 외부 정보를 활용합니다.', 'Performance Highlights': '실험 결과, AcTracer는 7개의 다양한 도메인에서 최첨단 성능을 달성했으며, 기존 방법들에 비해 최대 38.83%의 성능 향상을 보였습니다.'}]



### Unlocking Exocentric Video-Language Data for Egocentric Video Representation Learning (https://arxiv.org/abs/2408.03567)
- **What's New**: EMBED (Egocentric Models Built with Exocentric Data) 메서드를 소개합니다. 이 방법은 외적 (exocentric) 비디오-언어 데이터를 변환하여 자동 학습을 위한 내적 (egocentric) 비디오 표현 학습으로 전환하는 것을 목표로 합니다.

- **Technical Details**: EMBED는 내적 데이터와 외적 데이터 간의 간극을 줄이기 위해 손-객체 상호작용 (Hand-Object Interaction, HOI) 단서를 사용하여 주요 비디오 클립을 선택하고 내적 내레이션 스타일에 맞게 변환하는 데이터 변환 프레임워크를 사용합니다. 비디오 큐레이션 및 내레이션 생성 전략을 결합하여 외적 데이터를 통해 새로운 내적 데이터 셋을 생성합니다.

- **Performance Highlights**: EMBED는 다양한 내적 비디오 다운스트림 작업에서 최첨단 성능을 달성했으며, Epic-Kitchens-100에서 4.7% 및 EGTEA 분류 벤치마크에서 6.2%의 절대적인 성능 향상을 이뤘습니다. 더불어, 다양한 외적 데이터셋에서 강력한 일반화 성능을 보여주었습니다.



### MoExtend: Tuning New Experts for Modality and Task Extension (https://arxiv.org/abs/2408.03511)
Comments:
          ACL 2024 - SRW

- **What's New**: 자동 MCQ 생성 기술을 개선하는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, 딥러닝 모델의 강화를 위해 새로운 counterfactual 데이터 증강 방법을 도입하고 있어서 NLP 모델의 robustness를 높입니다. 마지막으로, 멀티모달 대형 언어 모델 (Large Language Models, LLMs)을 더 효율적으로 확장하기 위한 MoExtend 프레임워크가 소개되었습니다.

- **Technical Details**: MCQ 생성의 경우, KDA 메트릭은 MCQ가 학생의 지식을 평가할 수 있는 능력을 측정합니다. 이는 학생 응답 기반의 평가를 통해 검증되며, pre-trained language models를 활용하여 학생의 문제풀이 행동을 모방하는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. NLP 모델의 robustness 강화를 위해서는 spurious 패턴의 문제를 해결하기 위해 대조 학습 (contrastive learning) 및 counterfactual 증강 방법을 사용합니다. 마지막으로, MoExtend 프레임워크는 믹스처 오브 전문가 (Mixture-of-Experts, MoE) 모델에 새로운 모달리티 전용 전문가를 원활하게 통합하여 멀티모달 LLMs를 확장시킵니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅에서의 사용성 평가와 높은 상관관계를 보였으며, expert-labeled MCQ의 품질 측정에서도 높은 예측 능력을 나타냈습니다. Counterfactual 로버스트니스와 도메인 간 일반화, 데이터 부족 상황에서의 일반화에서 기존 방법 대비 significant improvements를 달성했습니다. MoExtend는 고속의 학습 속도와 낮은 catastrofic forgetting 리스크로 멀티모달 기능을 효율적으로 확장하는 데 성공했습니다.



### Automated Theorem Provers Help Improve Large Language Model Reasoning (https://arxiv.org/abs/2408.03492)
- **What's New**: 이번 주 AI 뉴스레터에서는 자동 다지선다형 질문 (MCQ) 생성, 딥러닝 모델의 강건성 향상, 그리고 대형 언어 모델 (LLM)의 논리적 추론능력 향상을 위한 새로운 접근법들이 논의됩니다.

- **Technical Details**: [{'Topic': '자동 MCQ 생성', 'Summary': '기존 MCQ 생성 평가지표인 BLEU, ROUGE, METEOR는 교육적 가치를 판단하지 못하는 한계가 있습니다. 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되었으며, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, {'Topic': '딥러닝 모델의 강건성', 'Summary': '최근 딥 모델들이 NLP 작업에서 사람보다 높은 정확성을 보였으나, 속임수 패턴에 의존하여 강건성이 제한된다는 문제가 보고되었습니다. 이에 대처하기 위해 대조 학습 (contrastive learning)과 반사실 증강 (counterfactual augmentation)을 제안합니다. 기존 방법과 달리, 이 접근법은 여러 개의 반사실을 생성하여 집합적 의사 결정을 통해 단어들의 인과 관계를 더 강건하게 감독합니다.'}, {'Topic': '대형 언어 모델 (LLM)의 논리적 추론 향상', 'Summary': '논리 프로그래밍 시스템과 자동 1차 논리 정리 증명기 (ATPs)를 사용하여 LLM의 논리적 추론 능력을 향상시킵니다. 특히 LLM의 번역 정확성을 분석하고, 구문 및 의미 오류를 자동으로 교정하는 방법을 제안합니다. 이 접근법은 PRONTOQA 벤치마크에서 시험되었습니다.'}]

- **Performance Highlights**: [{'Topic': '자동 MCQ 생성', 'Summary': 'KDA_disc와 KDA_cont는 실제 교실 환경에서 탁월한 사용성을 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에서 강한 예측력을 가집니다.'}, {'Topic': '딥러닝 모델의 강건성', 'Summary': '집합적 의사 결정을 통한 접근법은 반사실 강건성, 크로스 도메인 일반화 및 희소 데이터 일반화 측면에서 상당한 개선을 이루었습니다.'}, {'Topic': '대형 언어 모델 (LLM)의 논리적 추론 향상', 'Summary': '논리적 추론 능력 향상을 통해 의미 오류가 크게 감소되었으며, LLM 기반 논리적 추론의 정확성이 더 향상되었습니다.'}]



### LAMPO: Large Language Models as Preference Machines for Few-shot Ordinal Classification (https://arxiv.org/abs/2408.03359)
Comments:
          COLM 2024

- **What's New**: 자동 다중 선택 질문 (MCQ) 생성의 평가는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 우리가 제안한 새로운 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 메트릭을 소개합니다. KDA는 학생의 표적 사실에 대한 지식을 평가하는 능력을 측정하며, 주어진 지식으로 문제를 해결할 수 있는지를 평가합니다.

- **Technical Details**: KDA는 학생 응답을 바탕으로 측정을 시작하며, 사전 학습된 언어 모델을 통해 학생의 문제 해결 행위를 모방하여 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이를 통해 KDA와 실제 강의실 사용성 사이의 강한 상관관계를 증명했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블한 다양한 MCQ 품질 측정치에 대한 높은 예측력을 보였습니다. 이는 n-gram 기반 유사성 메트릭과 결합되었을 때 강력함이 더해져 전문가가 평가한 여러 품질 측정에서 높은 예측력을 입증했습니다.



### The Use of Large Language Models (LLM) for Cyber Threat Intelligence (CTI) in Cybercrime Forums (https://arxiv.org/abs/2408.03354)
- **What's New**: 이번 주 AI 뉴스레터에서는 세 가지 주요 연구 논문을 다룹니다. 첫 번째는 자동 Multiple Choice Questions (MCQ) 생성의 평가 메트릭이 교육적 가치를 반영하도록 개선한 연구입니다. 두 번째 논문은 NLP 태스크에서 모델의 robustness를 높이기 위해 contrastive learning과 counterfactual augmentation을 어떻게 활용하는지 다룹니다. 세 번째 연구는 OpenAI의 GPT-3.5-turbo 모델을 사용하여 사이버 위협 정보(Cyber Threat Intelligence, CTI)를 정확하게 추출 및 요약하는 능력을 평가한 연구입니다.

- **Technical Details**: {'MCQ Generation': '현재 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR 등은 단어 유사도에 초점을 맞추어 MCQ의 교육적 가치를 평가하지 못합니다. 이를 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 평가 지표를 제안하였습니다. KDA는 학생이 특정 사실에 대한 지식을 가지고 문제에 답할 수 있는지 측정합니다. 우리는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.', 'Robustness in NLP': 'NLP 태스크에서 deep models는 spurious patterns에 의존하는 문제로 인해 robustness가 제한된다는 보고가 있습니다. 본 연구에서는 여러 개의 counterfactual을 생성하고, collective decisions를 통해 단어들의 인과관계를 파악하는 방법을 제안합니다. 이 방법은 spurious correlations에 덜 민감하여 counterfactual robustness, cross-domain generalization, 그리고 제한된 데이터에서도 성능 향상을 보여줍니다.', 'Cyber Threat Intelligence': 'OpenAI의 GPT-3.5-turbo 모델을 사용하여 사이버범죄 포럼의 대화를 분석하고 CTI 정보를 추출하는 시스템을 평가하였습니다. 3개의 포럼에서 500개의 대화를 무작위로 추출하여 요약하고 CTI 변수를 코딩하였습니다. 두 명의 분석가가 각 대화를 검토하여 LLM이 추출한 정보의 정확성을 평가하였습니다.'}

- **Performance Highlights**: {'MCQ Generation': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 높은 사용성과 강한 상관관계를 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합된 KDA_disc와 KDA_cont가 전문가가 라벨링한 다양한 MCQ 품질 측정에 강한 예측력을 가짐을 확인하였습니다.', 'Robustness in NLP': '제안된 방법은 attribution-based synthesis의 bias에 덜 민감하며, 다양한 측면에서 중요한 성능 향상을 달성하였습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) 제한된 데이터로부터의 generalization.', 'Cyber Threat Intelligence': 'LLM 시스템은 10개의 CTI 변수를 평균 98%의 정확도로 코딩하였으며, 특정 변수가 최대 100%의 정확도를 기록했습니다. 이는 LLM이 CTI 정보를 효율적으로 요약하고 분석할 수 있음을 시사합니다.'}



### miniCTX: Neural Theorem Proving with (Long-)Contexts (https://arxiv.org/abs/2408.03350)
- **What's New**: 이번 달 AI 뉴스레터에서는 세 가지 주요 논문에 대해 알아보겠습니다. 첫 번째 논문은 자동 MCQ 생성 평가의 새로운 지표인 '지식 종속 가능성(KDA)'을 제안하며, 두 번째 논문은 NLP 태스크의 Robustness (강건성)를 증진시키기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 도입하는 방법을 소개합니다. 마지막으로, 세 번째 논문은 새로운 정의와 보조 정리들에 의존하는 수학 정리 증명 능력을 테스트하는 'miniCTX'라는 벤치마크를 소개합니다. 이 논문들은 각기 다른 분야에서 AI의 가능성을 한 단계 더 끌어올릴 수 있는 방법들을 제시하고 있습니다.

- **Technical Details**: [{'title': 'MCQ 평가 지표', 'content': "기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 평가 메트릭은 교육적 가치를 고려하지 못합니다. 이를 해결하기 위해 우리는 '지식 종속 대답 가능성(KDA)'이라는 새로운 메트릭을 도입했습니다. KDA는 타겟 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 보다 자동화된 평가를 가능하게 합니다."}, {'title': 'NLP 태스크의 강건성 증대', 'content': "최근의 딥러닝 모델들은 NLP 태스크에서 초인적인 정확도를 보여주지만, 스퍼리어스 패턴(spurious patterns)에 의존하는 문제로 인해 강건성이 제한됩니다. 이를 해결하기 위해, 대조 학습과 counterfactual augmentation을 활용하는 방법을 제안합니다. 기존 방법이 사람이나 머신에 의존하는 반면, 우리는 '여러 개의' counterfactual을 생성하고 이들의 분포를 기반으로 집합적 의사 결정을 내리는 방식을 채택했습니다. 이 방법은 인과 관계를 보다 강건하게 감독할 수 있습니다."}, {'title': '컨텍스트 의존 정리 증명 벤치마크', 'content': "'miniCTX'는 훈련 중 관찰되지 않은 새로운 정의나 보조 정리에 의존하는 수학적 정리를 증명하는 모델의 능력을 테스트합니다. miniCTX는 실제 Lean 프로젝트와 교과서에서 가져온 정리들을 포함하고 있으며, 각 정리는 수천 개의 토큰을 포함하는 컨텍스트와 연결되어 있습니다. 우리는 '파일 튜닝(file-tuning)'이라는 기본적인 레시피를 도입하여 모델을 학습시키고, 이는 전통적인 신경망 정리 증명 접근법보다 뛰어난 성능을 보였습니다."}]

- **Performance Highlights**: [{'title': 'MCQ 평가 지표 성능', 'content': '실제 강의실 상황에서 KDA_disc와 KDA_cont는 높은 상관관계를 보였으며, 전문가가 레이블링한 다양한 MCQ 품질 측정 기준에 대해 예측력이 강한 것으로 나타났습니다.'}, {'title': 'NLP 태스크의 강건성', 'content': '대조 학습과 counterfactual augmentation 방식을 통한 새로운 방법이 tasks 모델의 바이어스를 줄이고, 다양한 측면에서 강건성, 크로스 도메인 일반화 및 데이터가 부족한 상황에서의 일반화 성능을 크게 개선했습니다.'}, {'title': 'miniCTX 성능', 'content': '파일 튜닝을 적용한 모델은 기존의 신경망 정리 증명 접근법보다 높은 성능을 보였으며, 표준 miniF2F 벤치마크에서도 33.61%란 새로운 최고 통과율을 기록했습니다.'}]



New uploads on arXiv(cs.IR)

### Retrieval Augmentation via User Interest Clustering (https://arxiv.org/abs/2408.03886)
- **What's New**: 이번 뉴스레터에서는 세 가지 흥미로운 최신 연구 결과를 소개합니다. 첫 번째 논문은 MCQ(Multiple Choice Questions) 생성의 평가를 개선하는 새로운 메트릭을 제안합니다. 두 번째 논문은 NLP 태스크에서 모델의 robustness를 높이기 위한 contrastive learning 및 counterfactual augmentation 방법을 탐구합니다. 마지막으로, 세 번째 논문은 사용자의 다양한 선호도를 보다 잘 반영하는 추천 시스템을 위한 새로운 '관심' 레이어를 도입한 방법을 제시합니다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR와 같은 기존의 n-gram 기반 메트릭이 교육적 가치를 평가하지 못한다는 문제를 제기하며, 새로운 지식 종속 가능성(KDA, Knowledge Dependent Answerability) 평가 메트릭을 제안합니다. KDA는 학생의 대상 사실에 대한 지식을 평가하는 능력을 측정합니다. 두 번째 논문은 counterfactual augmentation을 통해 모델의 robustness를 높이는 방법을 제안하며, 여러 개의 counterfactual을 생성하고 이를 집합적으로 분석하여 더 강력하게 인과관계를 파악합니다. 세 번째 논문은 사용자와 항목 간에 '관심(interest)' 레이어를 도입하여 추천 시스템의 성능을 향상시키는 방법을 제시하고, 이를 통해 간접적인 사용자 관심을 더 잘 반영할 수 있도록 합니다.

- **Performance Highlights**: 첫 번째 논문에서는 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 지녔음을 보여주었습니다. 두 번째 논문은 제안된 방법이 다양한 방향에서 robustness, cross-domain generalization 및 scarce data에서의 generalization에서 유의미한 개선을 이루었다고 보고합니다.  세 번째 논문에서는 공개 데이터셋 및 Meta 제품에 적용하여 추천 성능과 계산 효율성을 크게 향상시켰음을 보였습니다. 특히, 짧은 동영상 추천에 있어 성능이 크게 향상되었습니다.



### A Reproducible Analysis of Sequential Recommender Systems (https://arxiv.org/abs/2408.03873)
Comments:
          8 pages, 5 figures

- **1. What's New**: 자동 MCQ 생성을 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 소개합니다. 이는 기존의 BLEU, ROUGE, METEOR 등과는 달리, MCQ가 얼마나 학생의 지식을 평가할 수 있는지를 측정합니다. 또한, KDA_disc와 KDA_cont라는 자동 평가 메트릭도 함께 제안되어, 예측 모델 평가에 더 적합한 기준을 제공합니다.

- **2. Technical Details**: 기존 MCQ 평가 메트릭들은 n-gram 기반의 단어 유사성에 집중했지만, 본 연구에서는 인간 설문 결과를 이용한 KDA 측정 방법과, 사전 훈련된 언어 모델을 활용한 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이를 통해 학생의 문제 해결 행동을 모방하여 MCQ의 대답 가능성을 평가합니다.

- **3. Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 (1) KDA와 (2) 실제 교실 환경에서의 사용성에 높은 상관관계를 보여주었습니다. 또한, n-gram 기반의 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가가 평가한 MCQ 품질 측정치에 대해 강력한 예측력을 가지고 있음이 확인되었습니다.



### Relevance meets Diversity: A User-Centric Framework for Knowledge Exploration through Recommendations (https://arxiv.org/abs/2408.03772)
- **What's New**: 현대의 추천 시스템(recommenders)에서 연관성과 다양성을 최적화하는 데 초점을 맞춘 새로운 접근 방식을 제안합니다. 제안된 프레임워크는 사용자의 행동을 중심으로 하며, 사용자가 추천 항목과 상호 작용하는 방식에 따라 지식 습득을 극대화하는 것을 목표로 합니다.

- **Technical Details**: 이 논문은 사용자가 추천 시스템과의 상호작용을 지속하는 동안 최대한 많은 지식을 얻을 수 있도록 하기 위해 다양성을 지식 습득의 대리 변수로 사용합니다. 제안된 프레임워크는 다양성과 연관성을 결합한 새로운 추천 전략을 도입하며, 이를 copula 함수로 구현합니다. 두 가지 표준화된 다양성 개념(coverage와 pair-wise distances)에 기반한 모델이 제안되었습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 통해 제안된 방법론을 광범위하게 평가한 결과, 제안된 전략이 기존의 최첨단 경쟁자들을 능가하는 성능을 보여주었습니다. 이 프레임워크는 사용자의 인내심과 추천 항목의 유용성에 따라 종료 확률을 모델링하여 실제 사용자 행동을 반영합니다.



### Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation (https://arxiv.org/abs/2408.03533)
- **What's New**: 자동 MCQ 생성에 대한 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 소개합니다. 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 데이터셋의 골드 샘플과 n-gram 유사성에만 집중하고 교육적 가치를 고려하지 않았는데, KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 고려합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 인간 조사에서 측정되며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이 자동 메트릭은 강의실에서 사용성을 가진다는 것을 인간 연구를 통해 입증하였습니다. 또한, n-gram 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 다양한 전문가 레이블 MCQ 품질 측정에 대한 강한 예측력을 보입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 레이블을 기반으로 한 실제 강의실 설정에서의 사용성과 강한 상관 관계가 있습니다. n-gram 기반 메트릭과 결합할 경우, 다양한 MCQ 품질 측정에 큰 예측력을 보여줍니다.



### Generative Language Models with Retrieval Augmented Generation for Automated Short Answer Scoring (https://arxiv.org/abs/2408.03811)
Comments:
          20 pages, 2 figures

- **What's New**: 최근 연구에서는 다중 선택 질문(MCQ) 자동 생성을 개선하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 도입했습니다. 또한, 자연어 처리 과제에서 대조 학습과 반사 사실 증가 기법을 활용해 모델의 내구성을 향상시키는 방법과 최신 생성형 언어 모델(GLMs)을 활용한 자동 짧은 답변 점수 시스템을 제안했습니다.

- **Technical Details**: 첫 번째 연구는 BLEU, ROUGE, METEOR와 같은 기존의 n-gram 기반 유사성 메트릭이 교육적 가치를 평가하지 못하는 문제를 지적하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 두 번째 연구는 대조 학습과 여러 반사 사실을 합성하여 단어의 인과 관계를 더 잘 파악하는 방법을 제안했습니다. 세 번째 연구는 베르트(BERT)와 일렉트라(ELECTRA)와 같은 사전 학습된 인코더 모델을 활용한 정보 검색(IR) 및 RAG와 GLM을 결합한 파이프라인을 제안하여 교실 평가에 있어 점수 정확도를 높였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 메트릭과 결합하여 전문가가 라벨링한 MCQ 품질 측정치에 대한 예측력을 향상시켰습니다. 대조 학습 기법은 반사 사실 강건성, 교차 도메인 일반화, 소량 데이터 일반화에서 상당한 개선 효과를 보였습니다. GLM을 활용한 자동 짧은 답변 점수 시스템은 SemEval 2013 데이터셋에서 기존 방법보다 높은 성과를 보였으며, 특히 SCIENTSBANK 3-way와 2-way 태스크에서 유의미한 개선을 나타냈습니다.



### Consumer Transactions Simulation through Generative Adversarial Networks (https://arxiv.org/abs/2408.03655)
Comments:
          12 pages

- **What's New**: 이 논문은 Multiple Choice Questions(MCQ) 자동 생성의 평가에 있어 기존 메트릭이 교육적 가치를 고려하지 않는 문제를 지적하며, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. 이와 더불어, deep model이 NLP 태스크에서 보여주는 제약된 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용한 접근을 소개합니다. 마지막으로, 대규모 소매 거래 데이터를 분석하기 위해 Generative Adversarial Networks(GAN)를 활용한 소매 거래 생성 방법을 제안합니다.

- **Technical Details**: [{'paper': 'MCQ Generation', 'details': '기존 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사도에만 집중해 교육적 가치를 무시합니다. 새로운 KDA 메트릭은 학생의 대상 사실에 대한 지식을 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA를 근사합니다.'}, {'paper': 'NLP Task Robustness', 'details': "최근 NLP 모델은 spurious 패턴에 의존해 robustness가 제한적입니다. 본 연구에서는 contrastive learning과 counterfactual augmentation을 통해 이러한 문제를 해결합니다. augmented data로 구성된 '여러 개의' counterfactual을 생성하고 집단적 의사 결정을 통해 각 용어의 인과관계를 견고하게 감독합니다."}, {'paper': 'Retail Data Simulation', 'details': '대규모 소매 데이터 시스템에서 GAN을 사용한 소매 거래 데이터 생성 방법을 제안합니다. SKU 가용성 제한을 모델에 통합하여 실제 소매 최적화 문제를 해결합니다. hyper-graphs와 같은 복잡한 embedding 방법을 활용해 생성된 거래가 실제와 유사함을 보여줍니다.'}]

- **Performance Highlights**: [{'paper': 'MCQ Generation', 'highlights': 'KDA_disc와 KDA_cont 메트릭은 인간 연구를 통해 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였고, 전문가가 레이블링한 MCQ 질 측정에서 높은 예측력을 보였습니다.'}, {'paper': 'NLP Task Robustness', 'highlights': '집단적 의사결정을 통한 접근 방식이 기존 방식보다 spurious correlation에 덜 민감하게 반응하며 다양한 차원에서 유의미한 성능 향상을 달성했습니다: counterfactual robustness, cross-domain generalization, generalization from scarce data.'}, {'paper': 'Retail Data Simulation', 'highlights': 'GAN 모델이 실제 거래와 비교했을 때 높은 현실성을 보여주었으며, 수요 예측 정확도 향상과 실시간 재고 최적화, 고객 맞춤형 제품 구성에 대한 가능성을 확인했습니다.'}]



### ULLME: A Unified Framework for Large Language Model Embeddings with Generation-Augmented Learning (https://arxiv.org/abs/2408.03402)
- **What's New**: 최근 연구는 자동으로 다중 선택 질문(Multiple Choice Questions, MCQ)을 생성하는 시스템의 평가 메트릭이 학생의 지식을 평가하는 능력을 적절하게 반영하지 못한다는 문제를 지적합니다. 이를 해결하기 위해 우리가 제안한 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭은 MCQ의 답변 가능성을 측정하고, 학생이 해당 지식을 얼마나 잘 이해하고 있는지를 평가합니다. 또한, 새로운 자동 평가 메트릭인 KDA_disc와 KDA_cont를 도입하여, 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방함으로써 KDA를 근사합니다.

- **Technical Details**: 자동 MCQ 생성의 기존 평가 메트릭(예: BLEU, ROUGE, METEOR)은 기본적으로 n-gram 기반의 유사성에 의존하여 MCQ의 교육적 가치를 적절히 평가하지 못합니다. KDA 메트릭은 학생이 주어진 지식을 기반으로 정답을 맞출 수 있는지를 평가함으로써 이를 개선합니다. KDA_disc와 KDA_cont는 학생들의 응답 성향을 모방하는 사전 학습된 언어 모델을 이용해 개발되었습니다.

- **Performance Highlights**: Human 평가 연구를 통해, KDA_disc와 KDA_cont가 KDA와 실제 강의실 설정에서의 사용성 측면에서 강한 상관관계를 가지는 것으로 나타났습니다. 또한, 이 메트릭은 n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 평가 지표 예측에서도 강력한 성능을 보였습니다.



### The Ontoverse: Democratising Access to Knowledge Graph-based Data Through a Cartographic Interfac (https://arxiv.org/abs/2408.03339)
- **What's New**: 이번에 발표된 연구는 교육자들이 학생 평가에 소요되는 시간을 절감하고자 MCQ(객관식 문제) 자동 생성 방법을 탐구합니다. 기존 BLEU, ROUGE, METEOR 평가 메트릭은 교육적 가치를 고려하지 못하였기에, 이들의 대안으로 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭을 새롭게 제안했습니다.

- **Technical Details**: KDA 메트릭은 학생이 특정 지식을 가지고 문제를 얼마나 잘 풀 수 있는지를 평가합니다. 이를 위해 학생 응답 데이터를 사용하여 KDA를 측정하는 방법을 보여주고, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제시합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 높은 상관관계를 가짐을 확인했습니다. 또한, 이 메트릭들은 n-그램 기반의 유사성 메트릭과 결합될 때, 다양한 전문가 레이블의 MCQ 품질 측정에서 강한 예측력을 나타냈습니다.



New uploads on arXiv(cs.CV)

### How Well Can Vision Language Models See Image Details? (https://arxiv.org/abs/2408.03940)
- **What's New**:  N 새로운 평가 메트릭인 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)을 제안하여, MCQ의 대답 가능성을 측정하고 교실에서의 사용성을 평가한다. 또한, 대비적 학습 및 반사실적 증가를 활용하여 NLP 모델의 robustness를 향상시키는 방법을 제안했다. 마지막으로, 픽셀 값 예측 작업 (Pixel Value Prediction, PVP)을 도입하여 Vision-Language 모델 (VLM)이 이미지의 세부사항을 이해할 수 있는 능력을 탐구하는 새로운 접근법을 발표한다.

- **Technical Details**: KDA는 학생 응답을 기반으로 MCQ의 대답 가능성을 평가하며, 자동 평가 메트릭 KDA_disc 및 KDA_cont을 통해 사전 학습된 언어 모델을 활용하여 유사한 동작을 모방한다. 또한, 대비적 학습 및 반사실적 증가를 통해 NLP 태스크에서 spurious 패턴에 의존하는 문제를 해결하려고 하며, PVP 작업은 CLIP 비전 인코더를 통해 픽셀 값을 예측하도록 설계되어 있다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가짐을 보여주었다. VLM의 경우, CLIP 인코더를 조정함으로써 픽셀 예측 정확도가 크게 향상되었고, 후속 이미지-언어 이해 태스크에서 확연한 성능 향상을 보였다. 특히, 이미지를 참조하여 세그멘테이션을 수행하는 태스크에서 평균 cIoU가 10.19 증가했으며, 비디오 게임에서 점수가 각각 80.34와 70.54 향상되었다.



### Fast Sprite Decomposition from Animated Graphics (https://arxiv.org/abs/2408.03923)
Comments:
          To be published ECCV 2024, project page: this https URL

- **What's New**: 이번 아카이브 논문(s) 리뷰에서는 자동 다지선다형 질문(automated MCQ) 생성, NLP 태스크에서의 반숙 능력(robustness) 향상, 그리고 동영상 그래픽 디컴포지션(sprite decomposition) 관련 내용을 다룹니다.

- **Technical Details**: [{'문제에 대한 상세 정보': '기존 자동 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)는 교육적 가치를 고려하지 않음. 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 통한 새로운 평가 메트릭 제안.', '기술적 접근법': '학생 응답을 통한 KDA 측정 방법 설명 후, 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont 자동 평가 메트릭 제안.', '관련 연구': '사람보다 나은 정확성을 보이는 딥 모델의 spurious patterns 의존성을 개선하기 위해대조 학습(contrastive learning)과 counterfactual augmentation을 활용. 여러 개의 counterfactual을 생성하고 집단적 의사결정을 통해 더 robust하게 단어들의 인과관계를 파악.', '동영상 그래픽 스프라이트 디컴포지션': '디레조 애니메이션 데이터셋을 구축하고, 효율적인 초기화 절차 및 사용자 입력을 활용한 최적화를 통해 스프라이트 분해 작업 수행.'}]

- **Performance Highlights**: [{'MCQ 평가': 'KDA_disc와 KDA_cont는 기존 n-gram 기반 메트릭과 결합하여 전문가들이 라벨링한 MCQ 품질 지표들에 대해 강한 예측력을 보임.', 'NLP 반숙 능력 향상': '제안된 집단적 의사 결정 방식을 통해 counterfactual robustness, cross-domain generalization, scarce data 환경에서의 일반화 측면에서 중요한 개선 달성.', '동영상 그래픽 디컴포지션': '스프라이트 분해 품질과 효율성 측면에서 기존 베이스라인 대비 우수한 성능을 입증. 최적화와 사용자 어노테이션의 결합으로 매우 빠른 수렴 시간 달성.'}]



### FMiFood: Multi-modal Contrastive Learning for Food Image Classification (https://arxiv.org/abs/2408.03922)
- **What's New**: 새로운 자동 MCQ 생성 평가 메트릭인 Knowledge Dependent Answerability(KDA)을 제안했습니다. 이 메트릭은 MCQ의 정답 가능성을 측정하고 학생의 해당 사실에 대한 지식을 평가할 수 있습니다. 또한, 기존 평가 메트릭인 BLEU, ROUGE, METEOR가 교육적 가치를 고려하지 않는 문제를 해결합니다.

- **Technical Details**: KDA는 학생들의 응답을 기반으로 측정됩니다. 이를 바탕으로 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭이 제안되었습니다. 이를 통해 더 정확한 평가를 가능하게 합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, 전문가들이 라벨링한 MCQ 품질 측정을 예측하는 데 강력한 능력을 보였습니다.



### AdapMTL: Adaptive Pruning Framework for Multitask Learning Mod (https://arxiv.org/abs/2408.03913)
Comments:
          13 pages, 9 figures, Published at ACM Multimedia (ACM MM) 2024

- **What's New**: MCQ 자동 생성 평가에 새로운 척도인 KDA 제안, NLP의 robust 연산을 위한 새로운 방법론, 멀티태스크 학습 모델의 효율적 압축을 위한 새로운 프레임워크 제안.

- **Technical Details**: [{'Title': 'Automatic Evaluation Metric for MCQ Generation', 'Content': 'BLEU, ROUGE, METEOR 같은 기존 평가 메트릭은 교육적 가치를 무시한다는 한계를 가지고 있다. 이를 해결하기 위해 MCQ의 대답 가능성을 측정하는 지식 종속 가능성(KDA) 메트릭을 제안하였다. 우리는 학생들의 응답을 기반으로 KDA를 측정하고, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하여 이를 모방한다.'}, {'Title': 'Robustness in NLP with Contrastive Learning and Counterfactual Augmentation', 'Content': '기존 방법들은 사람들이 counterfactual을 만들거나 모델이 데이터셋 내에서 이를 찾는데 한계를 보인다. 우리는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안한다.'}, {'Title': 'Adaptive Pruning Framework for Multitask Learning Models', 'Content': '멀티태스크 학습 모델의 복잡성을 해결하고 압축하는 AdapMTL을 제안한다. AdapMTL은 다중 학습 가능 소프트 임계값(Soft Thresholds)을 사용해 각 구성 요소의 가지치기 민감도에 맞춰 자동으로 적절한 희소성을 결정하며, 각 태스크 손실의 중요성을 동적으로 조정한다. NYU-v2 및 Tiny-Taskonomy 데이터셋에서 우수한 성능을 보여준다.'}]

- **Performance Highlights**: [{'Title': 'MCQ Generation', 'Content': 'KDA_disc와 KDA_cont 메트릭이 교육적 평가에서 유의미한 상관관계를 보이며, 전문 평가자의 평가와 높은 상관관계를 가진다.'}, {'Title': 'Robustness in NLP', 'Content': '집합적 의사 결정이 기존 방법들보다 spurious 패턴에 덜 민감하며, 다양한 차원에서 성능을 향상시킨다: counterfactual robustness, cross-domain generalization, scarce data에서의 일반화.'}, {'Title': 'AdapMTL', 'Content': '멀티태스크 데이터셋 NYU-v2와 Tiny-Taskonomy에서 기존의 가지치기 및 멀티태스크 가지치기 방법보다 우수한 성능을 보임. 훈련 손실이 낮고, 다양한 희소성 수준에서 테스트 세트의 평가 지표가 더 좋다.'}]



### Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection (https://arxiv.org/abs/2408.03888)
Comments:
          10 pages, 8 figures, Accepted to ACM MM '24

- **What's New**: 자동 MCQ 생성의 평가 메트릭을 개선하고 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭를 제안했습니다. 이를 통해 학생의 지식을 평가하는 능력을 측정할 수 있습니다. 또한 불필요한 패턴에 의존하는 딥 모델의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 사용하는 방법을 제시하였습니다. 마지막으로, 비지도 이상 감지(unsupervised Anomaly Detection) 작업을 개선하기 위해 Dual-Modeling Decouple Distillation (DMDD)를 제안하였습니다.

- **Technical Details**: KDA 메트릭은 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 학습된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방합니다. NLP 태스크의 robustness를 향상시키기 위해 대조 학습과 counterfactual augmentation 기법을 사용하였으며, 기존의 방법과 달리 여러 개의 counterfactual을 생성하고 집합적으로 결정을 내리는 방식을 채택했습니다. DMDD는 비지도 이상 감지 작업을 위해 제안된 새로운 프레임워크로, Decoupled Student-Teacher Network, Dual-Modeling Distillation, 그리고 Multi-perception Segmentation Network의 세 가지 주요 구성 요소로 이루어져 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont은 실제 강의실 세트에서의 사용성과 강한 상관관계를 가진 것으로 나타났습니다. 새로운 대조적 학습 방식은 counterfactual robustness, cross-domain generalization, 그리고 scarce data로부터의 generalization 측면에서 유의미한 향상을 보였습니다. DMDD는 MVTec AD 데이터셋에서 이전 지식 증류 기반 방법들을 뛰어넘는 성능을 보이며, pixel-level AUC 98.85%와 PRO 96.13%를 기록하였습니다.



### Global-Local Progressive Integration Network for Blind Image Quality Assessmen (https://arxiv.org/abs/2408.03885)
- **What's New**: 이번 논문에서는 자동으로 다지선다형 문제(MCQ)를 생성하는 새 방법과 평가 메트릭을 제안합니다. 기존 메트릭인 BLEU, ROUGE, METEOR는 n-gram 유사성만을 평가하는 반면, 제안된 방법은 학생의 실제 지식 평가 능력을 고려합니다. 새로운 메트릭인 Knowledge Dependent Answerability (KDA)는 목표 사실에 대한 학생의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생들의 문제 해결 행동을 모방하기 위해 사전 학습된 언어 모델을 활용합니다. KDA_disc와 KDA_cont는 KDA를 대체하여 MCQ의 품질을 더욱 정확하고 교육적인 가치에 근거하여 평가합니다.

- **Performance Highlights**: 인간 실험을 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 갖는다는 점을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 경우, 여러 전문가가 라벨링한 MCQ 품질 측정값에 대한 예측 정확도가 높은 것으로 나타났습니다.



### Surgformer: Surgical Transformer with Hierarchical Temporal Attention for Surgical Phase Recognition (https://arxiv.org/abs/2408.03867)
- **What's New**: 이 논문에서는 현재의 자동 복수 선택 질문(MCQ) 생성 평가 메트릭의 한계를 지적하고, 새로운 평가 메트릭인 지식 종속 가능성(KDA, Knowledge Dependent Answerability)을 제안하고 있다. KDA는 MCQ의 대답 가능성(Answerability)을 측정하며, 학생의 지식을 평가하는 능력을 검증한다.

- **Technical Details**: 이 연구는 KDA를 기반으로 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안하며, 사전 훈련된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방한다. 또한, 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성 및 전문가가 라벨한 MCQ 품질 측정과 상관이 강하다는 것을 보여준다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 BLEU, ROUGE, METEOR 등 n-그램 기반 유사성 메트릭과 결합하면 다양한 전문가가 라벨한 MCQ 품질 측정에 대한 강한 예측력을 갖는다.



### Bi-Level Spatial and Channel-aware Transformer for Learned Image Compression (https://arxiv.org/abs/2408.03842)
- **What's New**: 최근의 딥러닝 모델들이 NLP 작업에서 초인적인 정확도를 보여주고 있지만, 불필요한 패턴에 의존하게 되는 문제가 있어서 모델의 견고성이 떨어진다. 이에 대해 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 활용한 새로운 방법을 제시했다.

- **Technical Details**: 기존의 방법들은 데이터셋에서 반사실적 데이터를 사람이 추가하거나 모델이 유사한 데이터를 찾도록 하지만, 이는 여전히 불필요한 상관관계(spurious correlation)에 영향을 받는다. 새로운 방법은 '여러 개의' 반사실적 데이터를 생성하고 이를 통해 각각의 용어의 인과관계를 더 견고하게 분석한다. 이 방법은 대조 학습(contrastive learning)과 반사실적 데이터 증강을 결합하여 모델의 견고성을 높인다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 반사실적 견고성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 그리고 희소한 데이터로부터의 일반화 능력 모두에서 기존 모델들보다 월등히 높은 성능을 보였다.



### Target Prompting for Information Extraction with Vision Language Mod (https://arxiv.org/abs/2408.03834)
Comments:
          7 pages, 5 figures

- **What's New**: 최근의 연구들은 자동으로 다지선다형 문제(MCQ)를 생성하는 시스템 개발에 집중하고 있습니다. 이번 연구에서는 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭이 교육적 가치를 제대로 반영하지 않는다는 문제를 지적하며, 새로운 자동 평가 메트릭인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'을 제안합니다. 또한, 최근에는 NLP 태스크에서 대규모 모델들의 정확도가 사람을 능가하지만, 견고성(robustness) 부족 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용하는 방법이 제안되었습니다. 그리고 대형 비전 및 언어 모델(Vision and Language model, VLM)이 문서 이해 및 질문 응답 시스템 구축에 혁신적인 변화를 가져오는 가운데, 이 모델의 단점을 보완하기 위한 '타겟 프롬프팅(Target Prompting)' 기법이 소개되었습니다.

- **Technical Details**: {'MCQ Generation': '이번 연구는 대상 사실에 대한 학생의 지식을 평가하는 새로운 메트릭인 KDA를 도입했습니다. KDA는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont로 구현되며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Robustness in NLP': '대조 학습과 반사실적 증강 기법을 사용하여 모델의 견고성을 강화하며, 기존 방법들과 달리 여러 개의 반사실적 데이터를 합성하여 예측 분포를 결정하는 방식을 사용합니다.', 'Document Understanding': "대형 비전 및 언어 모델에서 '타겟 프롬프팅' 기법을 사용하여 특정 문서 이미지의 부분을 명시적으로 타겟팅해 관련된 응답을 생성하는 방법이 제안되었습니다. 이 모델은 이미지 인코더와 변환기 디코더로 구성된 복합 모달 모델을 사용합니다."}

- **Performance Highlights**: {'MCQ Evaluation': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있으며, 전문가들이 라벨링한 다양한 MCQ 품질 평가 지표에 대해 높은 예측력을 보였습니다.', 'NLP Robustness': '대조 학습과 반사실적 증강 기법을 통해 1) 반사실적 견고성, 2) 도메인 간 일반화, 3) 희소 데이터를 통한 일반화 성능이 크게 향상되었습니다.', 'Document Understanding': '타겟 프롬프팅 기법을 사용한 실험 결과, 특정 문서 이미지로부터 구체적이고 정확한 정보를 효율적으로 추출할 수 있음을 확인했습니다.'}



### Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields (https://arxiv.org/abs/2408.03822)
Comments:
          Project page: this https URL

- **What's New**: 자동 MCQ 생성의 교육적 가치를 향상시키기 위해, 기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사성 평가 메트릭을 넘어 '지식 종속 가능성' (Knowledge Dependent Answerability, KDA)이라고 불리는 새로운 평가 메트릭을 제안했습니다. 또한, 최근 NLP 태스크에서 심화 모델의 강건성을 높이기 위해 대조 학습(contrastive learning)과 반사실적 확대(counterfactual augmentation)를 활용하는 방법을 연구했습니다. 마지막으로 3D Gaussian Splatting (3DGS)를 통해 메모리와 저장 효율성을 크게 향상시키면서 고품질의 3D 장면을 실시간으로 렌더링하는 새로운 방법론을 제시했습니다.

- **Technical Details**: MCQ 생성의 KDA는 학생들이 대상 사실(target fact)을 알고 있을 때 문제의 정답 가능성을 측정하는 메트릭으로, 인간 조사에서 얻은 반응을 기반으로 설정되었습니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. NLP 태스크의 강건성을 높이기 위해 반사실적 데이터를 여러 개 생성하고 집합적 의사 결정을 통해 각 용어의 인과성을 감독합니다. 3DGS에서는 학습 가능한 마스크 전략을 사용해 Gaussian 포인트 수를 줄이고, 시각 의존적 색상(view-dependent color)을 위해 그리드 기반 신경 필드를 사용할 것을 제안했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 조사와 전문가가 라벨링한 실제 강의실 세트에서 높은 상관관계를 보였습니다. 대조 학습과 반사실적 확대 방식은 대조 강건성(counterfactual robustness), 크로스 도메인 일반화(cross-domain generalization), 그리고 데이터 부족 상황에서의 일반화에 대해 상당한 개선을 이끌어냈습니다. 3DGS 기반 방법론은 기존 3DGS 대비 25배 이상의 저장 공간 절감을 이루었고, 동적 장면에서도 효과적인 성능을 유지하며 렌더링 속도도 크게 향상시켰습니다.



### Vision-Language Guidance for LiDAR-based Unsupervised 3D Object Detection (https://arxiv.org/abs/2408.03790)
Comments:
          Accepted to BMVC 2024

- **What's New**: 새로운 자동 평가 메트릭 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 자동 생성된 다지선다형 질문(MCQ)의 대답 가능성(answerability)과 교육적 가치를 평가합니다. 또한 LiDAR 포인트 클라우드에서 3D 객체 인식을 위한 비지도 학습 방법 ViLGOD를 소개합니다. 이 방법은 Vision-Language 모델을 활용하여 객체를 분류하며, 기존의 방법과 달리 객체의 크기에만 의존하지 않습니다.

- **Technical Details**: {'MCQ Generation': '기존 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에만 초점을 맞추고 있으나, KDA는 학생의 지식 평가 능력을 기준으로 MCQ의 대답 가능성을 측정합니다. 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 통해 KDA를 대체하는 방법을 제안합니다.', '3D Object Detection': 'ViLGOD는 LiDAR 포인트 클라우드에서 작동하며, CLIP 모델을 이용해 객체를 분류합니다. 이 방법은 스패티오-템포럴 클러스터링(spatio-temporal clustering)을 통해 객체 제안을 생성하고, 상식적인 배경 샘플을 필터링합니다. 또한, 시퀀싱된 LiDAR 스캔의 특성을 활용해 일시적인 뷰를 생성하고 분류 정확성을 높입니다.'}

- **Performance Highlights**: {'MCQ Evaluation': 'KDA_disc와 KDA_cont는 KDA 및 강의실 사용성과 강한 상관관계를 보였습니다. n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링 한 MCQ 품질 기준에 대해 높은 예측력을 보였습니다.', '3D Object Detection': 'ViLGOD는 Waymo Open Dataset에서 $+23~\\text{AP}_{3D}$, Argoverse 2에서 $+7.9~\\text{AP}_{3D}$의 성능 향상을 보여줍니다. 이동 객체와 정적 객체 모두를 정교하게 탐지하며, 추가적인 반복 자기 학습을 필요로 하지 않습니다.'}



### Methodological Explainability Evaluation of an Interpretable Deep Learning Model for Post-Hepatectomy Liver Failure Prediction Incorporating Counterfactual Explanations and Layerwise Relevance Propagation: A Prospective In Silico Tria (https://arxiv.org/abs/2408.03771)
- **What's New**: 새로운 Multiple Choice Questions (MCQ)의 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)을 제안했습니다. 이 메트릭은 기존 방법들이 주목하지 못했던 교육적 가치를 평가하며 대상 사실에 대한 학생의 지식을 평가하는 능력을 강조합니다.

- **Technical Details**: ['기존의 BLEU, ROUGE, METEOR 메트릭이 아닌, 새로운 KDA 메트릭을 통해 MCQ의 답변 가능성을 측정', 'KDA_disc와 KDA_cont는 사전 학습된 언어 모델 (pre-trained language models)을 사용하여 학생의 문제 해결 행동을 모사함', 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 증명']

- **Performance Highlights**: ['KDA_disc와 KDA_cont 메트릭이 향상된 예측 성능과 교육적 가치를 제공', 'n-gram 유사성 기반 메트릭과 결합했을 때, 다양한 전문가가 평가한 MCQ 품질 지표에 대한 강력한 예측 능력 확보']



### MMSummary: Multimodal Summary Generation for Fetal Ultrasound Video (https://arxiv.org/abs/2408.03761)
Comments:
          MICCAI 2024

- **What's New**: 이번 달 AI 소식에서는 MCQ 생성, NLP robustness (강건성), 그리고 의학적 영상 요약 시스템의 발전에 대한 중요한 논문들을 소개합니다.

- **MCQ 자동 생성**: {"What's New": '이번 연구에서는 자동 MCQ 생성을 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다.', 'Technical Details': 'BLEU, ROUGE, METEOR와 같은 기존 메트릭이 단어의 유사성만을 측정하는 데 비해, KDA는 생성된 MCQ가 학생의 지식을 평가할 수 있는지에 중점을 둡니다.', 'Performance Highlights': 'Human Survey를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가짐을 입증했습니다.'}

- **NLP 모델의 강건성 향상**: {"What's New": 'deep 모델의 robustness (강건성)의 한계를 극복하기 위해 contrastive learning과 counterfactual augmentation을 결합한 새로운 방법을 제안했습니다.', 'Technical Details': '기존 방법과 달리, 이 연구는 여러 개의 counterfactual을 생성하여 집합적 의사 결정 (collective decisions) 을 통해 단어들의 인과관계를 더 robust하게 파악합니다.', 'Performance Highlights': '이 방법은 다양한 차원에서 현저한 성능 향상을 이루었으며, 특히 counterfactual robustness, cross-domain generalization, 그리고 데이터가 부족한 상황에서의 generalization에서 두각을 나타냈습니다.'}

- **의료 영상 요약 시스템**: {"What's New": '최초의 자동화된 다중모드 요약 생성 시스템인 MMSummary를 소개합니다. 이 시스템은 태아 초음파 분석에 중점을 둡니다.', 'Technical Details': 'MMSummary는 키프레임 감지, 키프레임 캡셔닝, 해부학적 분할 및 측정의 3단계 파이프라인으로 구성됩니다. 키프레임 감지 단계에서는 Transformer 기반 네트워크를 사용하여 대표 프레임을 감지하고, 캡셔닝 단계를 통해 태아 초음파 키프레임에 의미있는 캡션을 생성합니다. 마지막으로, 생체측정을 자동으로 수행합니다.', 'Performance Highlights': '실험 결과, MMSummary는 약 31.5%의 스캔 시간을 절약하며, 일관되고 정확한 평가를 제공하여 임상 워크플로 효율성을 높이는 잠재력을 보여줍니다.'}



### 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting (https://arxiv.org/abs/2408.03753)
Comments:
          The 18th European Conference on Computer Vision ECCV 2024

- **1 Multiple Choice Questions Generation**: {"What's New": '자동 MCQ (Multiple Choice Questions) 생성의 교육적 가치를 평가하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.', 'Technical Details': '기존 BLEU, ROUGE, METEOR 메트릭은 단순히 n-gram 유사성에 기반하여 생성된 MCQ를 평가하며, MCQ가 학생 지식을 평가하는 능력을 고려하지 않습니다. KDA는 대상 사실에 대한 학생의 지식을 평가하기 위해 MCQ의 답변 가능성을 측정합니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, 이는 전문가들이 표시한 MCQ 품질 측정 기준과 결합할 때 강력한 예측력을 보였습니다.'}

- **2 Robustness in NLP Models**: {"What's New": '최근 딥러닝 모델의 NLP 태스크에서의 정확성에도 불구하고, 모델이 spurious patterns에 의존하여 robustness가 제한되는 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 활용합니다.', 'Technical Details': '기존 방식은 사람이 반사실을 만들거나 모델이 데이터셋에서 유사한 것을 찾는 방식인데, 반사실 생성을 통해 여러 반사실 집합을 만들고 집합적 의사결정(collective decision)을 통해 단어들의 인과관계를 더 robust하게 파악합니다.', 'Performance Highlights': '이 방법은 반사실 robustness, 크로스 도메인 일반화, 데이터가 부족한 상황에서의 일반화에 있어서 크게 개선된 성능을 보여줍니다.'}

- **3 Factorised Tensorial Illumination for 3D Gaussian Splatting**: {"What's New": '3iGS (Factorised Tensorial Illumination for Gaussian Splatting)는 3D Gaussian Splatting (3DGS)의 렌더링 품질을 개선하여 더 나은 시각적 효과를 제공합니다.', 'Technical Details': '3iGS는 단일 방사율(outgoing radiance) 대신, BRDF(Bidirectional Reflectance Distribution Function) 특성을 고려한 연속적인 국소 조명 필드를 사용합니다. Tensorial Factorisation을 통해 효율적인 평가를 가능하게 합니다.', 'Performance Highlights': '3iGS는 반사 표면이 있는 장면에서 3DGS보다 우수한 성능을 보여주며, 특히 NeRF Blender 데이터셋과 Shiny Blender 데이터셋에서 높은 정량적 및 정성적 평가를 받았습니다.'}



### Data Generation Scheme for Thermal Modality with Edge-Guided Adversarial Conditional Diffusion Mod (https://arxiv.org/abs/2408.03748)
Comments:
          accepted by ACM MM 2024/ACM MM24

- **summaries**: [{"What's New": '이번 연구는 기존의 평가 메트릭이 MCQ의 교육적 가치를 평가하지 못하는 문제를 해결하기 위해 새로운 자동 평가 메트릭(Knowledge Dependent Answerability, KDA)을 제안합니다. 이 메트릭은 대상 사실에 대한 학생의 지식을 평가할 수 있는 MCQ의 대답 가능성(answerability)을 측정합니다.', 'Technical Details': 'KDA는 학생 응답을 바탕으로 측정되며, 두 개의 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하여 사전 훈련된 언어 모델을 사용해 학생의 문제 해결 행동을 모방합니다. 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성과 강한 상관관계가 있음을 밝혔습니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont를 n-gram 기반의 유사성 메트릭과 결합하면 다양한 전문가-레이블 MCQ 품질 측정에서 강력한 예측력을 갖게 됨을 보여줍니다.'}, {"What's New": '최근의 딥 모델들이 자연어 처리(NLP) 태스크에서 인간보다 나은 정확성을 보였지만, 적대적 패턴(spurious patterns)에 의존해 강인함(robustness)이 제한된다는 보고가 있었습니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 반사실 증가(counterfactual augmentation)를 활용하는 방안을 제안합니다.', 'Technical Details': '기존의 반사실 증가 방법은 사람이 데이터를 추가하거나, 모델이 데이터셋에서 반사실과 유사한 사례를 자동으로 찾는 방식을 사용했으나, 여전히 적대적 상관관계(spurious correlations)에 영향을 받았습니다. 이에 반해, 우리는 여러 반사실을 생성하고 이 세트의 예측 분포에 대해 집합적 의사 결정(collective decisions)을 함으로써, 단어들의 인과관계를 더욱 강인하게 감독하는 방법을 제안합니다.', 'Performance Highlights': '우리의 방법은 반사실 강인성(counterfactual robustness), 크로스 도메인 일반화(cross-domain generalization), 데이터가 부족한 상황에서의 일반화 등 다양한 측면에서 중요한 개선을 이루었습니다.'}, {"What's New": '이 연구는 열화상 알고리즘, 특히 객체 탐지에서 심도 있는 학습 모델을 활용하여 저조도 및 악천후 상황에서도 우수한 성능을 발휘하는 방식을 제안합니다. 이를 위해 엣지 가이드 조건 확산 모델(edge guided conditional diffusion model)을 도입하여 가시 이미지에서 추출한 엣지 정보를 활용해 픽셀 수준의 정렬된 가상 열화상 이미지를 생성합니다.', 'Technical Details': '이 논문에서는 가시 영역의 엣지 정보를 조건으로 하여 조건부 확률 밀도(conditional probabilistic density)를 학습하고, 적대적 학습(adversarial learning)을 통해 가시 영역의 불필요한 엣지 정보를 제거하는 전략을 제안합니다. 확산 모델(diffusion model)과 두 단계 모달 성 적대적 훈련 방식을 결합하여 가시 이미지에서 열화상 이미지를 정밀하게 생성합니다.', 'Performance Highlights': 'LLVIP 데이터셋을 활용한 광범위한 실험을 통해 제안된 ECDM이 기존 최첨단 접근 방식보다 이미지 생성 품질에서 우수함을 입증했으며, 객체 탐지 과제에서 최대 7.1% mAP 향상을 가져왔습니다.'}]



### Intuitionistic Fuzzy Cognitive Maps for Interpretable Image Classification (https://arxiv.org/abs/2408.03745)
Comments:
          This work has been submitted for possible journal publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 자동 Multiple Choice Question (MCQ) 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 또한, 자연어 처리(NLP) 태스크에서 사람이 만드는 counterfactual 데이터를 비롯한 다양한 augmentation 방식을 극복하고, 여러 개의 counterfactual 데이터를 통해 단어들의 인과관계를 파악하는 새로운 방법을 제안합니다. 마지막으로, 이미지 분류 문제에서 직관적인 FCMs(iFCMs)를 확장하여 이미지 분류의 해석 가능성을 높이는 새로운 프레임워크 Interpretable Intuitionistic FCM (I2FCM)을 도입합니다.

- **Technical Details**: MCQ 평가에서는 기존의 n-gram 기반 메트릭(BLEU, ROUGE, METEOR)이 교육적 가치를 제대로 평가하지 못하므로, 우리 연구에서는 학생의 대답 가능성(answerability)을 측정하는 KDA를 제안합니다. 특히, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입하여 사전 훈련된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방합니다. NLP 태스크의 경우, 여러 개의 counterfactual 데이터를 생성하고 이들 데이터 집합 간의 분포를 평가하여 모델의 robustness를 높입니다. 또한, 이미지 분류 문제를 해결하기 위해 iFCM을 확장하여, 주어진 이미지의 중요한 영역을 추출하고 데이터 기반으로 직관적인 퍼지 관계를 결정하는 알고리즘을 개발합니다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, BLEU, ROUGE와 같은 기존의 n-gram 기반 메트릭과 결합하면 전문가들이 라벨링한 MCQ 품질 측정에 대해 높은 예측력을 보입니다. NLP 태스크에서는 집합적 의사 결정을 통해 attribution-based synthesis가 갖는 bias를 줄여, counterfactual robustness, cross-domain generalization, sparse data에서의 generalization 등 다양한 측면에서 성능이 향상되었습니다. 또한, iFCM을 확장한 I2FCM 프레임워크는 공개 데이터셋에서 뛰어난 분류 성능을 제공하면서 해석 가능한 추론을 가능하게 하였습니다.



### Advancing Multimodal Large Language Models with Quantization-Aware Scale Learning for Efficient Adaptation (https://arxiv.org/abs/2408.03735)
Comments:
          Accepted by ACMMM2024

- **new_papers**: [{"What's New": '기존 BLEU, ROUGE, METEOR에 대한 한계를 극복하기 위해 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 평가 능력을 평가합니다.', 'Technical Details': 'KDA는 학생 응답을 기반으로 측정되며, 이를 근접하게 모사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이러한 메트릭은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': '실제 강의실 세트에서 사용성과 강한 상관관계를 가지며, BLEU 등과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 측정에 높은 예측 능력을 보였습니다.'}, {"What's New": '강화된 robustness를 위해 contrastive learning 및 counterfactual augmentation을 활용한 새로운 방법을 제안합니다. 기존 접근법과 달리, 집합적 의사 결정 방식을 통해 더 강력한 인과 관계 파악을 가능케합니다.', 'Technical Details': '여러 개의 반사실(counterfactual)을 생성하고, 이 집합에서 예측 분포에 대한 집합적 의사 결정을 통해, 인과 관계를 감독합니다.', 'Performance Highlights': '실험 결과, 기존 방법에 비해 반사실적 robustness, 도메인 간 일반화, 부족한 데이터로부터의 일반화 등 다양한 측면에서 성능이 크게 향상되었습니다.'}, {"What's New": '멀티모달 대형 언어 모델(MLLMs)의 매개변수 양자화(parameter quantization)를 활용하여 자원 제약을 해소하고자 하는 첫 번째 연구를 제공합니다. 이를 위해 QSLAW(Quantization-aware Scale LeArning)를 제안합니다.', 'Technical Details': 'QSLAW는 그룹별 스케일 팩터를 학습하여 양자화 오류를 줄이는 방법과 멀티모달 워크업을 도입하여 언어와 멀티모달 샘플을 점진적으로 통합하는 방법을 사용합니다.', 'Performance Highlights': 'QSLAW는 ScienceQA에서 91.04%의 정확도를 보이며 QLoRA보다 4.08% 향상된 성능을 기록했습니다. 또한, 전체 정밀도 모델보다 더 높은 성능을 나타냈습니다.'}]



### Soft-Hard Attention U-Net Model and Benchmark Dataset for Multiscale Image Shadow Remova (https://arxiv.org/abs/2408.03734)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: [{'논문 제목': 'Knowledge Dependent Answerability: A Novel Evaluation Metric for Automatic MCQ Generation', '내용': '지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다.'}, {'논문 제목': 'Robustness in Natural Language Processing: Leveraging Contrastive Learning and Counterfactual Augmentation', '내용': '대조적 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 NLP 모델의 로버스트니스(robustness)를 향상시키기 위한 새로운 접근 방식을 제안합니다.'}, {'논문 제목': 'Soft-Hard Attention U-net for Multiscale Shadow Removal', '내용': '멀티스케일(multi-scale) 그림자 제거를 위한 최신의 깊은 학습 아키텍처인 Soft-Hard Attention U-net (SHAU)을 제안하며, 복잡한 그림자 패턴을 포함할 수 있는 새로운 합성 데이터셋(MSRD)을 제공합니다.'}]

- **Technical Details**: [{'논문 제목': 'Knowledge Dependent Answerability: A Novel Evaluation Metric for Automatic MCQ Generation', '세부사항': 'Human evaluation과 사전 학습된 언어 모델(pre-trained language model)의 성능을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 학생의 문제 해결 행동을 모사하여 KDA를 근사합니다.'}, {'논문 제목': 'Robustness in Natural Language Processing: Leveraging Contrastive Learning and Counterfactual Augmentation', '세부사항': '복수의 반사실적(counterfactual) 세트를 합성하고, 집합적 의사 결정을 통해 각 용어의 인과성을 판단하는 방법을 제안하여 기존 방법의 spurious correlation 문제를 해결하려고 합니다.'}, {'논문 제목': 'Soft-Hard Attention U-net for Multiscale Shadow Removal', '세부사항': 'SHAU 아키텍처는 소프트 및 하드 어텐션 모듈과 멀티스케일 특징 추출 블록을 포함하며, 다양한 크기와 강도의 그림자를 효과적으로 제거합니다. 더불어 복잡한 그림자 패턴이 포함된 MSRD 데이터셋을 제공합니다.'}]

- **Performance Highlights**: [{'논문 제목': 'Knowledge Dependent Answerability: A Novel Evaluation Metric for Automatic MCQ Generation', '성과': 'KDA_disc와 KDA_cont 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였으며, 전문가 라벨 MCQ 품질 지표 예측에 뛰어난 성능을 보였습니다.'}, {'논문 제목': 'Robustness in Natural Language Processing: Leveraging Contrastive Learning and Counterfactual Augmentation', '성과': '제안된 방법은 대조 학습과 집합적 의사 결정을 통해 반사실적 로버스트니스, 도메인 간 일반화, 그리고 희소 데이터를 통한 일반화에서 상당한 개선을 달성했습니다.'}, {'논문 제목': 'Soft-Hard Attention U-net for Multiscale Shadow Removal', '성과': 'SHAU는 여러 벤치마크 데이터셋에서 최신 기술 상태의 그림자 제거 방법을 능가하며, 그림자 영역의 PSNR 및 RMSE 지표를 각각 25.1% 및 61.3% 향상시켰습니다.'}]



### Pick of the Bunch: Detecting Infrared Small Targets Beyond Hit-Miss Trade-Offs via Selective Rank-Aware Attention (https://arxiv.org/abs/2408.03717)
- **What's New**: 이 새로운 논문에서는 다양한 측면에서 자동 MCQ 생성, NLP 모델의 강건성 향상 및 적외선 소형 표적 탐지 기술을 다룬 혁신적인 방법을 소개합니다. 첫 번째 논문에서는 기존 n-gram 기반 유사성 메트릭의 한계를 극복하기 위해 Knowledge Dependent Answerability (KDA) 메트릭을 제안합니다. 두 번째 논문에서는 반사적 학습과 반사적 증가를 활용하여 NLP 모델의 강건성을 높이는 방법을 제안합니다. 마지막으로, 적외선 소형 표적 탐지를 위한 고급 네트워크인 SeRankDet을 소개합니다.

- **Technical Details**: [{'Paper': 'Automatic MCQ Generation', 'Technical Details': 'BLEU, ROUGE, METEOR와 같은 기존의 MCQ 생성 평가 메트릭은 단어 유사성에만 집중하며 교육적 가치를 평가하지 못함. 이를 해결하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 메트릭을 제안하여 MCQ의 대답 가능성과 학생의 지식 평가 능력을 측정함. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 선행 언어 모델을 활용해 학생들의 문제 해결 행동을 모방함.'}, {'Paper': 'NLP Model Robustness', 'Technical Details': 'NLP 태스크에서 고성능을 보이는 deep model들이 spurious pattern에 의존하는 문제와 이를 해결하기 위한 반사적 학습 및 반사적 주입 (counterfactual augmentation)을 사용. 기존 방법과 달리 여러 개의 반사적 사례를 통해 집단적 의사 결정을 내리는 접근법을 제안, 이러한 방법이 인과관계를 더 robust하게 파악함.'}, {'Paper': 'Infrared Small Target Detection', 'Technical Details': '적외선 소형 표적 탐지를 위한 SeRankDet 네트워크를 제안함. SeRank 모듈은 비선형 Top-K 선택 과정을 활용해 가장 중요한 응답을 유지하며, Large Selective Feature Fusion (LSFF) 모듈은 동적 융합 전략을 통해 참 표적과 거짓 경보를 구분하는 능력을 강화함. 또한 Dilated Difference Convolution (DDC) 모듈을 통해 섬세한 표적 특징을 강조하고, 수신 영역을 확장하여 표적-배경 구분을 개선함.'}]

- **Performance Highlights**: [{'Paper': 'Automatic MCQ Generation', 'Performance Highlights': 'KDA_disc와 KDA_soft는 인간 연구를 통해 실제 강의실 세트에서의 사용성과 강한 상관관계를 나타냄. 또한 n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 표시한 다양한 MCQ 품질 측정에서 높은 예측력을 가짐.'}, {'Paper': 'NLP Model Robustness', 'Performance Highlights': '제안된 방법은 반사적 robustness, cross-domain generalization, 및 제한된 데이터에서의 generalization이라는 다양한 차원에서 큰 개선을 달성함.'}, {'Paper': 'Infrared Small Target Detection', 'Performance Highlights': 'SeRankDet은 여러 공공 데이터셋에서 최신 성능의 새로운 기준을 설정함. 특히, DDC, SeRank, 및 LSFF 모듈은 기존 방법에 비해 높은 정확도와 낮은 false positive rate를 실현함.'}]



### CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Efficient Mobile Applications (https://arxiv.org/abs/2408.03703)
- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성을 평가하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 이는 기존 BLEU, ROUGE, METEOR 평가 메트릭이 MCQ의 교육적 가치를 고려하지 않는 문제를 해결합니다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 지식을 평가하는 MCQ의 대답 가능성(answerability)을 측정합니다. 인간 설문조사를 통해 학생 응답을 기반으로 KDA를 측정한 다음, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방해 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 전문가가 라벨링한 KDA와 강한 상관관계를 가짐을 보였습니다. 또한, n-gram 기반의 유사성 메트릭과 결합할 때, 다양한 전문가 라벨링 MCQ 품질 측정에 대한 강한 예측력을 가집니다.



### Openstory++: A Large-scale Dataset and Benchmark for Instance-aware Open-domain Visual Storytelling (https://arxiv.org/abs/2408.03695)
- **What's New**: 이번 뉴스레터에서는 MCQ 자동 생성 및 평가, NLP 모델의 강건성 강화, 그리고 이미지 생성 모델의 일관성 개선에 대한 세 가지 중요한 연구를 다룹니다.

- **Technical Details**: [{'MCQ Generation': '기존 평가 메트릭 BLEU, ROUGE, METEOR는 MCQ 생성의 교육적 가치를 고려하지 않습니다. 새로운 평가 메트릭으로 KDA (Knowledge Dependent Answerability)를 제안하여 학생의 지식 기반 대답 가능성을 측정합니다. 이를 위해 KDA_disc, KDA_cont 자동 평가 메트릭을 개발하였으며, 인간 평가와 강한 상관관계를 보였습니다.'}, {'NLP Model Robustness': 'NLP 태스크에서 최근 deep model들의 성공에도 불구하고, spurious pattern에 의존하여 robustness가 부족했습니다. 대조 학습 (contrastive learning) 및 반사실 증강 (counterfactual augmentation)을 이용하여 모델의 강건성을 높이는 방법을 연구합니다. 여러 개의 counterfactual을 생성해 집단적 의사 결정을 통해 단어들의 인과관계를 평가하여 모델의 성능을 강화합니다.'}, {'Image Generation Consistency': 'Openstory++ 데이터셋을 소개하여 여러 인스턴스의 일관성 (instance-level semantic coherence)이 보장되는 이미지 생성 모델을 개발합니다. 특히 연속된 프레임 사이에서 내러티브의 일관성을 유지하고, Cohere-Bench라는 벤치마크를 통해 이러한 모델을 평가하는 프레임워크를 제시합니다. 또한, Openstory++는 다양한 동영상에서 키프레임을 추출하고, 세분화된 시각적 주석을 제공하여 내러티브 연속성을 높였습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont 메트릭은 전문가에 의해 레이블된 MCQ 품질 측정과 높은 상관관계를 가지면서도 n-gram 기반 유사성 메트릭과 결합했을 때 높은 예측력을 보였습니다.'}, {'NLP Model Robustness': '제안된 접근 방식은 여러 분야에서 counterfactual robustness, cross-domain generalization, scarce data 규정 향상에서 큰 개선을 보였습니다.'}, {'Image Generation Consistency': 'Cohere-Bench 실험을 통해 Openstory++ 데이터셋이 고품질의 시각적인 이야기 생성 모델을 양성하는 데 탁월함을 입증했습니다.'}]



### L4DR: LiDAR-4DRadar Fusion for Weather-Robust 3D Object Detection (https://arxiv.org/abs/2408.03677)
- **What's New**: 교사들이 학생 평가에 소요되는 시간을 줄이기 위해 자동으로 다지선다형 질문(MCQ)을 생성하는 새로운 방법이 제안되었습니다. 이 방법은 기존 평가 메트릭의 한계를 극복하고 학생의 해당 사실에 대한 지식 평가 능력을 측정하는 'Knowledge Dependent Answerability (KDA)'라는 새로운 자동 평가 메트릭을 도입했습니다.

- **Technical Details**: KDA는 특정 목표 사실을 알 때 MCQ의 답변 가능성을 측정하는 메트릭입니다. 이는 인간 설문 조사를 통해 학생들의 답변을 분석한 후, 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방함으로써 자동화할 수 있습니다. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭이 제안되어, 이는 KDA를 근사치로 제공하며 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 전문가가 라벨링한 MCQ의 품질 측정에서 n-gram 기반 유사성 메트릭과 결합하여 예측력을 크게 향상시켰습니다.



### Designing Extremely Memory-Efficient CNNs for On-device Vision Tasks (https://arxiv.org/abs/2408.03663)
- **What's New**: 이번 주 AI 뉴스레터에서는 세 가지 흥미로운 논문을 소개합니다. 첫 번째 논문은 자동 MCQ(다지선다형 질문) 생성의 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 두 번째 논문은 대조 학습(constrastive learning)과 반사실 증강(counterfactual augmentation)을 활용하여 NLP 모델의 강건성을 향상시키는 방법에 대해 다룹니다. 세 번째 논문은 저가형 임베디드 및 IoT 기기를 위한 메모리 효율적인 CNN(convolutional neural network)을 소개합니다.

- **Technical Details**: ['MCQ 생성 논문에서 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 교육적 가치를 평가하는 능력이 부족하다는 문제를 해결하기 위해 KDA 메트릭을 제안합니다. KDA는 학생의 대상 사실에 대한 지식을 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 사용하여 KDA를 근사화합니다.', '대조 학습과 반사실 증강을 통해 NLP 모델의 강건성을 향상시키기 위한 방법을 논의합니다. 여러 개의 반사실을 생성하여 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 더 강건하게 파악하는 방식을 제안합니다.', "메모리 효율적인 CNN을 위해 세 가지 디자인 원칙을 제안합니다: 'input segmentation'으로 입력 이미지를 패치로 분할하여 메모리 요구사항을 줄이고, 'patch tunneling'으로 독립적인 네트워크 경로를 구축하여 메모리를 지속적으로 낮게 유지하며, 'bottleneck reordering'으로 메모리 사용량을 일정하게 유지합니다."]

- **Performance Highlights**: ['MCQ 논문에서는 KDA_disc와 KDA_cont 자동 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, 전문가가 레이블한 다양한 MCQ 품질 측정에 대해 높은 예측력을 갖는 것을 보여줍니다.', '대조 학습과 반사실 증강을 통한 방법은 여러 차원에서 중요한 향상을 달성합니다. 특히, 반사실 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 그리고 적은 데이터로부터의 일반화(generalization from scarce data)에서 두드러진 성과를 보입니다.', '메모리 효율적인 CNN은 ImageNet 분류에서 단 63 KB의 메모리로 61.58%의 top-1 정확도를 달성합니다. 이는 현존하는 메모리 효율적인 네트워크 중 가장 낮은 메모리 사용량을 기록하며, MobileNet 대비 약 89배, MCUNet 대비 약 3.1배 낮은 메모리 사용량을 자랑합니다.']



### PHOCUS: Physics-Based Deconvolution for Ultrasound Resolution Enhancemen (https://arxiv.org/abs/2408.03657)
Comments:
          Accepted at the Workshop of Advances in Simplifying Medical Ultrasound at MICCAI 2024

- **What's New**: 이번 뉴스레터에서는 세 가지 최신 논문을 소개합니다. 첫 번째 논문은 Multiple Choice Questions(MCQ) 자동 생성의 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안했습니다. 두 번째 논문은 NLP 태스크에서 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 통해 모델 robustness를 개선하는 방법론을 다룹니다. 세 번째 논문에서는 초음파(B-mode) 이미지를 기반으로 연속적인 음영 지도(echogenicity map)를 복원하여 해상도를 향상시키는 물리 기반 디컨볼루션(deconvolution) 기법을 제안했습니다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR 등의 기존 평가 메트릭이 MCQ의 교육적 가치를 평가하지 못하는 한계를 극복하기 위해 KDA를 제안했습니다. 특히, 학생의 문제 해결 행동을 모방하기 위해 pre-trained language models를 활용한 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안했습니다. 두 번째 논문에서는 여러 개의 counterfactual을 생성하고 등 집합적인 의사 결정(collective decisions)을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안했습니다. 세 번째 논문에서는 PSF의 모델링을 통한 물리 기반 렌더링 파이프라인을 통합하여 연속적인 음영 지도를 복원하는 INR을 사용해 초음파 해상도를 향상시키는 방법론을 제시했습니다.

- **Performance Highlights**: 첫 번째 논문은 KDA_disc와 KDA_cont 메트릭이 실제 강의실 세트에서 교육적 사용성과 강한 상관관계가 있음을 human study로 검증하였습니다. 두 번째 논문은 counterfactual robustness, cross-domain generalization, and scarce data에서의 향상을 통해 제안된 방법이 다양한 차원에서 significant improvements를 달성하였음을 empiral results로 검증하였습니다. 세 번째 논문은 synthetic, in-vitro, 그리고 in-vivo 초음파 데이터에서 PSNR과 SSIM을 비교한 결과, 제안 방법이 전통적인 방법들보다 우수한 성능을 보였음을 입증했습니다.



### TALE: Training-free Cross-domain Image Composition via Adaptive Latent Manipulation and Energy-guided Optimization (https://arxiv.org/abs/2408.03637)
Comments:
          The 32nd ACM Multimedia Conference (MM '24)

- **What's New**: 이 연구들은 최근의 기술 및 방법론 혁신을 제시하며, 각각의 문제 영역에서 새로운 접근 방식을 통한 개선을 제공합니다.

- **Details**: [{'Title': '자동 Multiple Choice Questions (MCQ) 생성 평가 메트릭', "What's New": 'MCQ의 교육적 가치를 평가하는 새로운 메트릭 지식 종속 가능성(Knowledge Dependent Answerability, KDA)를 제안하였습니다.', 'Technical Details': '기존의 BLEU, ROUGE, METEOR 메트릭이 단순 n-gram 유사성에 의존한다고 지적하면서, 학생의 실제 지식 평가를 더 잘 반영하는 자동 메트릭 KDA_disc와 KDA_cont를 제시했습니다. 이러한 메트릭들은 pre-trained language models를 활용해 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 실제 강의 세트에서 평가 전문가들에 의해 사용성 및 질에서 강한 상관관계를 보였습니다.'}, {'Title': 'NLP에서 Robustness 개선을 위한 대조적 학습 및 반사실적 증가', "What's New": 'Contrasting learning (대조적 학습)과 Counterfactual augmentation (반사실적 증가)를 이용한 새로운 robust NLP 모델 제안.', 'Technical Details': '기존 방법들이 spurious correlations에 영향을 받는 문제를 해결하기 위해, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계 (causality)를 감독합니다. 이 접근 방식은 attribution-based synthesis (귀속 기반 합성)의 task model bias에 덜 민감합니다.', 'Performance Highlights': 'Counterfactual robustness(반사실적 견고성), cross-domain generalization(도메인 간 일반화), scarce data(희소 데이터) 일반화에서 성능 향상을 보여주었습니다.'}, {'Title': 'Text-to-Image Diffusion 모델 기반의 Cross-domain Image Composition', "What's New": 'TALE이라는 새로운 훈련이 필요 없는 Framework를 제안하며, 도메인 차이를 초월한 이미지 합성을 가능하게 만듭니다.', 'Technical Details': 'TALE은 Adaptive Latent Manipulation과 Energy-guided Latent Optimization이라는 두 가지 메커니즘을 통해 latent space에서 명확하고 효과적인 지침을 제공하여 합성 프로세스를 지도합니다. 이 접근 방식은 이전의 self-attention map 조작 방법보다 더 효율적입니다.', 'Performance Highlights': 'TALE은 다양한 포토리얼리스틱 및 예술적 도메인에서 최첨단 성능을 달성했습니다. 실험과 사용자 연구를 통해 TALE의 강력한 능력을 입증했습니다.', 'Resource': '코드 및 결과는 GitHub111에서 제공될 예정입니다.'}]



### Concept Conductor: Orchestrating Multiple Personalized Concepts in Text-to-Image Synthesis (https://arxiv.org/abs/2408.03632)
Comments:
          Github Page: this https URL

- **What's New**: 이번 연구에서는 다중 개념 텍스트-이미지 모델을 사용자별로 맞춤화하는 프레임워크인 'Concept Conductor'를 소개합니다. 이 프레임워크는 속성 누출(attribute leakage)과 레이아웃 혼동(layout confusion)을 방지하여 시각적 충실도와 올바른 레이아웃을 보장합니다.

- **Technical Details**: 'Concept Conductor'는 다중 경로 샘플링(multi-path sampling), 레이아웃 조정(layout alignment), 개념 주입(concept injection) 세 가지 주요 요소로 구성됩니다. Multi-path 샘플링은 다중 개념 모델들이 독립적인 디노이징(denoising) 프로세스를 유지하게 하여 속성 누출을 방지합니다. 레이아웃 조정은 각 모델이 올바른 레이아웃을 생성하도록 자기 주의 기반 공간 안내(self-attention-based spatial guidance)를 사용합니다. 개념 주입은 각 개념의 시각적 특징을 최종 생성 이미지에 완전히 주입하여 조화를 보장합니다.

- **Performance Highlights**: 제시된 방법은 30개 개념을 포함하는 새로운 데이터셋에서 검증되었으며, 기존 방법들보다 개념 충실도와 텍스트 의미 일치도에서 뛰어난 성과를 보였습니다. 시각적 세부 사항을 잘 유지하면서도 정확한 레이아웃을 생성하는 데 성공했으며, 성능 개선이 두드러졌습니다.



### Weakly Contrastive Learning via Batch Instance Discrimination and Feature Clustering for Small Sample SAR ATR (https://arxiv.org/abs/2408.03627)
- **What's New**: 이번 뉴스레터에서는 최근 논문 세 편에 대해 다룹니다. 첫 번째 논문에서는 Multiple Choice Questions (MCQ)의 자동 생성 평가를 위해 새로운 지식 종속 가능성 메트릭(KDA)을 제안하였고, 두 번째 논문에서는 대조 학습(contrastive learning)과 counterfactual 증대를 활용하여 자연어 처리(NLP) 모델의 강건성을 높이는 방법을 제안하였습니다. 세 번째 논문에서는 Synthetic Aperture Radar (SAR) 자동 목표 인식(ATR) 기술에서 소량의 라벨링 데이터만으로 높은 인식률을 달성하는 새로운 프레임워크 BIDFC를 소개하였습니다.

- **Technical Details**: [{'paper': '첫 번째 논문', 'details': '기존 BLEU, ROUGE, METEOR 메트릭은 MCQ의 교육적 가치를 고려하지 않는다는 문제를 해결하기 위해, KDA 메트릭을 통해 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정함으로써 MCQ의 대답 가능성(answerability)을 평가합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 인간 설문 조사를 통해 이를 검증하였습니다.'}, {'paper': '두 번째 논문', 'details': '기존의 counterfactual 증대 방법은 spurious correlation의 영향을 받는다는 문제를 해결하기 위해, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더 넘어 robust하게 파악하는 방법을 제안합니다. 이를 통해 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 성능을 향상시켰습니다.'}, {'paper': '세 번째 논문', 'details': 'BIDFC(Batch Instance Discrimination and Feature Clustering) 프레임워크를 통해 SAR 이미지에서 높은 유사성을 갖는 샘플들 사이의 임베딩 거리를 조절하는 약한 대조 학습(weakly contrastive learning)을 적용합니다. 또한, DWV loss (Dynamic-Weighted Variance loss) 함수로 각 샘플의 강화 버전을 클러스터링합니다. 이 방법은 MSTAR 데이터베이스에서 3.13%의 훈련 데이터만으로도 91.25%의 분류 정확도를 달성하였습니다.'}]

- **Performance Highlights**: [{'paper': '첫 번째 논문', 'highlights': 'KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 전문가가 라벨링한 MCQ의 품질 측정 기준과 예측력이 높습니다.'}, {'paper': '두 번째 논문', 'highlights': '제안한 방법은 attribution-based synthesis보다 태스크 모델 바이어스에 덜 민감하며, 다양한 차원에서 성능 향상을 이뤄냈습니다. 특히 counterfactual robustness, cross-domain generalization 및 scarce data에서의 generalization에서 유의미한 성능 향상을 보였습니다.'}, {'paper': '세 번째 논문', 'highlights': 'MSTAR 데이터베이스에서 91.25%의 분류 정확도를 달성했으며, 같은 훈련 데이터에서 선형 평가를 수행해도 정확도가 90.13%에 도달하였습니다. OpenSarShip 데이터베이스에서도 BIDFC의 효과를 확인하였습니다.'}]



### AgentsCoMerge: Large Language Model Empowered Collaborative Decision Making for Ramp Merging (https://arxiv.org/abs/2408.03624)
- **What's New**: 이번 연구에서는 최첨단 자동 MCQ 생성 시스템의 평가 지표로 활용될 수 있는 새로운 메트릭, 지식 종속 가능성(KDA)을 제안합니다. 기존의 BLEU, ROUGE, METEOR 같은 n-gram 기반 유사성 지표는 교육적 가치를 평가하지 못하는 반면, KDA는 학생의 지식 평가 능력을 반영합니다. 또 다른 연구에서는 자동화된 데이터 증대(Augmentation) 방법으로 대조 학습(Contrastive Learning)과 counterfactual 증대를 활용해 NLP 시스템의 견고성을 높이는 방법을 소개합니다. 마지막으로, 자율주행차량(CAV)의 램프 병합 구역에서의 효율성과 안전성을 높이기 위한 새로운 협력적 의사결정 프레임워크, AgentsCoMerge를 제안합니다.

- **Technical Details**: MCQ 평가 메트릭 KDA는 인간 설문조사를 통해 측정되며, 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안해 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다. 대조 학습과 counterfactual 증대를 이용한 연구에서는 기존의 사람 또는 모델 기반 데이터 증대 방법 대신 여러 개의 counterfactual을 생성하고 집합적 의사 결정(Collective Decisions)을 내리는 방법을 제시합니다. AgentsCoMerge는 장면 관찰 및 이해 모듈, 계층적 계획 모듈, 인터 에이전트 통신 모듈, 강화 반사 훈련 전략을 통해 구성됩니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 전문가가 레이블링 한 실제 강의실 사용성과 강한 상관관계를 보였습니다. 대조 학습을 통한 방법은 다양한 차원에서 성능 향상을 보여주었으며, KDA_disc와 KDA_cont는 n-gram 유사성 지표와 결합할 때 전문가가 레이블링한 MCQ 품질 지표에 대한 예측력이 강했습니다. AgentsCoMerge는 복수의 램프 병합 시나리오에서 다중 에이전트 협력 의사 결정의 효율성과 효과에서 뛰어난 성능을 보여주었습니다.



### JARViS: Detecting Actions in Video Using Unified Actor-Scene Context Relation Modeling (https://arxiv.org/abs/2408.03612)
Comments:
          31 pages, 10 figures

- **What's New**: 1. 이 논문에서는 새로운 자동 평가 메트릭인 'Knowledge Dependent Answerability (KDA)'를 제안하여, MCQ의 대답 가능성(answerability)을 측정하고, 학생의 지식을 평가하는 능력을 평가함으로써 기존의 n-gram 기반 평가 메트릭의 한계를 극복하려고 합니다.

2. 또 다른 논문은 robustness 향상을 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 소개합니다. 특히, 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 데이터의 인과관계를 더 정확하게 파악합니다.

3. 비디오 액션 탐지(VAD)에서 새로운 프레임워크 'JARViS'를 제안하여 배우와 장면 간의 상호작용을 Transformer attention을 통해 효과적으로 모델링하며, 최신 성능을 달성했다고 보고합니다.

- **Technical Details**: 1. KDA는 학생의 응답을 기반으로 한 human survey를 통해 측정되며, 이를 자동 평가 메트릭인 KDA_disc와 KDA_cont로 확장하여 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

2. 이전 방법과 달리, 이 논문에서 제안하는 방법은 여러 개의 counterfactual을 생성하고, 이를 통해 단어들의 인과관계를 더 robust하게 파악할 수 있습니다.

3. JARViS는 키프레임에서 찍은 배우 특징을 사용하고, 비디오 클립에서 공간-시간적 장면 특징을 만들어 배우와 장면 간의 세밀한 상호작용을 고려하여 최종 액션을 출력합니다.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont는 human survey 에서 측정된 KDA 및 전문가가 레이블한 사용성과 강한 상관관계를 보였습니다.

2. 새로운 contrastive learning 방법은 counterfactual robustness, cross-domain generalization, 희소 데이터의 일반화에서 성능 향상이 있었습니다.

3. JARViS는 AVA, UCF101-24, JHMDB51-21 등 유명 VAD 데이터셋에서 기존 방법을 significant margins로 능가하는 성과를 보였습니다.



### PRISM: PRogressive dependency maxImization for Scale-invariant image Matching (https://arxiv.org/abs/2408.03598)
Comments:
          15 pages, 8 figures, ACM MM 2024. Supplementary materials are included

- **What's New**: MCQ 자동 생성 및 평가에서 기존 블루(BLEU), 루즈(ROUGE), 메테오(METEOR) 등의 n-그램 중심 평가 지표가 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 새로운 평가 지표 '지식 종속 가능성(KDA, Knowledge Dependent Answerability)'를 제안합니다. 또한, 최근 NLP에서 super-human 성능을 보이는 deep models의 spurious 패턴 의존성 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 적용한 접근법을 제시합니다. 이미지 매칭에서는 PRISM(Progressive dependency maxImization for Scale-invariant image Matching)이라는 새로운 프레임워크를 통해 비일치 영역을 가지치고 스케일 차이를 해결하는 방법을 제안합니다.

- **Technical Details**: {'MCQ Generation and Evaluation': 'KDA_disc와 KDA_cont라는 두 개의 자동 평가 지표를 도입해 프리 트레인드 언어 모델을 이용해 학생들의 문제 해결 행위를 모방하여 지식 종속 가능성(KDA)를 근사합니다.', 'NLP Model Robustness': '여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)으로 각 용어의 인과성을 더욱 견고하게 감시하는 방법을 제안합니다.', 'Image Matching': 'PRISM 프레임워크는 Multi-scale Pruning Module(MPM)을 사용해 다양한 스케일의 정보를 집계하고 비관련 특징을 점진적으로 가지치기하여, Scale-Aware Dynamic Pruning Attention(SADPA)을 통해 다중 스케일의 정보를 모델링합니다.'}

- **Performance Highlights**: {'MCQ Evaluation': 'KDA_disc와 KDA_cont는 전문가들이 라벨링한 실제 강의 환경에서의 사용성과 강한 상관관계를 보이며, n-그램 기반의 유사성 지표와 결합했을 때 다양한 전문가 라벨링 MCQ 품질 평가 기준에 대해 강한 예측력을 보입니다.', 'NLP Model Robustness': '제안된 접근법은 왜곡된 모델 바이어스에 덜 민감하며, counterfactual robustness, cross-domain generalization, 부족한 데이터로부터의 일반화 등 여러 차원에서의 성능 개선을 달성합니다.', 'Image Matching': 'PRISM은 homography estimation, relative pose estimation, visual localization와 같은 다양한 태스크에서 state-of-the-art(SOTA) 성능을 달성하며, 뛰어난 일반화 능력을 보입니다.'}



### Focal Depth Estimation: A Calibration-Free, Subject- and Daytime Invariant Approach (https://arxiv.org/abs/2408.03591)
- **What's New**: 이 논문들은 다양한 기술 혁신을 소개하며 교육, NLP, 그리고 개인화된 기술 분야에서 중요한 발전을 이루었습니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 평가 메트릭이 단순히 n-gram 유사성에만 초점을 맞추어 교육적 가치를 무시하는 문제를 해결하기 위해, 저자들은 대상 사실(target fact)에 대한 지식을 바탕으로 MCQ의 답변 가능성(answerability)을 측정하는 지식 종속 가능성(KDA)를 새로운 자동 평가 메트릭으로 제안했습니다. Human survey를 통한 KDA 측정 후, 이들을 모방한 모델로 KDA_disc와 KDA_cont 메트릭을 개발하여 강의실 사용성과 강한 상관관계를 보였습니다.', 'NLP Robustness': '최근의 deep model들이 NLP 태스크에서 높은 정확도를 보이나 spurious pattern에 의존해 robustness가 제한된다는 문제를 해결하기 위해, 저자들은 대조 학습(contrastive learning)과 counterfactual augmentation을 제안했습니다. 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 감독하는 방법을 제안하였습니다.', 'Focal Depth Estimation': '기존의 시선 추적 시스템과 autofocal 안경이 빈번한 사용자별 보정이 필요한 문제를 해결하기 위해, 저자들은 LSTM 네트워크와 도메인 특화된 피처 엔지니어링을 활용하여 초점 거리를 추정하는 획기적인 보정 없는 방법을 소개했습니다. 이 접근법은 평균 절대 오차(MAE) 10 cm 이하의 정확도를 달성하며 autofocal 안경의 실용성을 크게 향상시킵니다. 또한 이 기술은 가상현실(VR), 증강현실(AR), 의학적 현미경 검사 및 로봇 수술 등 다양한 분야에도 적용 가능합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont 메트릭은 실제 교실 환경에서의 사용성과 강한 상관관계를 보였으며, 다른 전문가가 라벨링한 MCQ 품질 지표와도 높은 예측력을 가집니다. 이는 MCQ의 교육적 가치를 평가하는 데 중요한 도구가 될 것입니다.', 'NLP Robustness': '제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 희소한 데이터로부터의 일반화에서 유의미한 개선을 보여줍니다.', 'Focal Depth Estimation': '저자들의 FOVAL 모델은 기존 모든 최첨단 방법보다 우수한 초점 거리 추정 정확도를 달성하며, 사용자의 빈번한 보정 없이도 실생활에서 높은 활용성을 보여주었습니다.'}



### Teach CLIP to Develop a Number Sense for Ordinal Regression (https://arxiv.org/abs/2408.03574)
Comments:
          Accepted by ECCV 2024

- **What's New**: 최근 논문에서는 기존의 MCQ(Multiple Choice Questions) 자동 생성 평가 메트릭의 한계를 극복하기 위해 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하였습니다. KDA는 학생이 대상 사실에 대해 얼마나 잘 이해하고 있는지를 평가할 수 있는 MCQ의 답변 가능성을 측정합니다.

- **Technical Details**: 이 논문은 먼저 학생 응답을 기반으로 KDA를 측정하는 방법을 보여줍니다. 그 후, 사전 학습된 언어 모델(pre-trained language models)을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서의 사용성과 강한 상관관계를 보임을 입증하였습니다. 또한, 이 메트릭을 n-gram 기반 유사성 메트릭과 결합할 경우 전문가가 라벨링한 MCQ 품질 측정치에 대해 강한 예측력을 가짐을 보였습니다.



### A comparative study of generative adversarial networks for image recognition algorithms based on deep learning and traditional methods (https://arxiv.org/abs/2408.03568)
- **What's New**: 이 논문은 자동 다지선다형 질문(MCQ)의 평가 메트릭에 새로운 접근법을 제안합니다. 기존 메트릭인 BLEU, ROUGE, METEOR는 MCQ의 교육적 가치를 평가하는 데 한계가 있습니다. 이를 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 메트릭을 제안합니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 평가하고, 학생이 해당 사실을 제대로 이해하고 있는지 측정하는 능력을 평가합니다. 이는 사람 설문 조사 결과를 기반으로 측정되었습니다. 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont도 제안되어 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅에서의 사용성과 강한 상관관계를 보여주었으며, 반복적인 단어 기반 유사성 메트릭과 병합되었을 때, 다양한 전문가가 라벨링한 MCQ 품질 척도에 대해 강한 예측력을 보였습니다.



### Unlocking Exocentric Video-Language Data for Egocentric Video Representation Learning (https://arxiv.org/abs/2408.03567)
- **What's New**: EMBED (Egocentric Models Built with Exocentric Data) 메서드를 소개합니다. 이 방법은 외적 (exocentric) 비디오-언어 데이터를 변환하여 자동 학습을 위한 내적 (egocentric) 비디오 표현 학습으로 전환하는 것을 목표로 합니다.

- **Technical Details**: EMBED는 내적 데이터와 외적 데이터 간의 간극을 줄이기 위해 손-객체 상호작용 (Hand-Object Interaction, HOI) 단서를 사용하여 주요 비디오 클립을 선택하고 내적 내레이션 스타일에 맞게 변환하는 데이터 변환 프레임워크를 사용합니다. 비디오 큐레이션 및 내레이션 생성 전략을 결합하여 외적 데이터를 통해 새로운 내적 데이터 셋을 생성합니다.

- **Performance Highlights**: EMBED는 다양한 내적 비디오 다운스트림 작업에서 최첨단 성능을 달성했으며, Epic-Kitchens-100에서 4.7% 및 EGTEA 분류 벤치마크에서 6.2%의 절대적인 성능 향상을 이뤘습니다. 더불어, 다양한 외적 데이터셋에서 강력한 일반화 성능을 보여주었습니다.



### Underwater litter monitoring using consumer-grade aerial-aquatic speedy scanner (AASS) and deep learning based super-resolution reconstruction and detection network (https://arxiv.org/abs/2408.03564)
Comments:
          The earlier version of this conference paper was accepted at OCEANS 2024-Halifax, Canada and was selected for inclusion in the Student Poster Competition (SPC) Program

- **What's New**: 이 논문은 자동 Multiple Choice Questions (MCQ) 생성을 보다 효과적으로 평가하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 또한 학생의 지식을 평가하는 능력을 높이기 위해 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 도입합니다. 추가로, 대화형 학습과 counterfactual augmentation을 이용하여 NLP 태스크의 robustness를 향상시키고, 수중 쓰레기 감지를 위한 Aerial-Aquatic Speedy Scanner (AASS)와 최적화된 YOLOv8 네트워크를 소개합니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 메트릭은 단순히 단어 유사성만 평가하는 문제를 개선하기 위해, KDA 메트릭을 도입하였습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 문제 해결 행동을 모방합니다.', 'NLP Robustness': '대화형 학습과 counterfactual augmentation을 합쳐서 robust한 NLP 모델을 개발했습니다. 특히, 여러 개의 counterfactual을 생성하고 집합적 의사 결정 방식으로 인과관계를 평가합니다.', 'Underwater Litter Detection': 'AASS와 Super-Resolution Reconstruction (SRR)을 통합하여 수중 쓰레기 감지 효율성을 높였습니다. RCAN 모델은 SRR 모델 중에서 가장 높은 mean average precision (mAP) 78.6%를 달성했습니다.'}

- **Performance Highlights**: {'MCQ Generation': 'Human studies에 따르면, KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 강한 상관관계를 보여주었고, n-gram 기반 유사성 메트릭과 결합하여 높은 예측 정확도를 나타냈습니다.', 'NLP Robustness': '이 접근 방식은 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화에 있어서 의미 있는 개선을 달성했습니다.', 'Underwater Litter Detection': 'SRR 테스트 셋은 4배 확대 요인 하에서 conventional bicubic 셋과 비교하여 개선된 mAP를 보여주었으며, 제안된 방법의 효율성을 입증하였습니다.'}



### Monitoring of Hermit Crabs Using drone-captured imagery and Deep Learning based Super-Resolution Reconstruction and Improved YOLOv8 (https://arxiv.org/abs/2408.03559)
Comments:
          The earlier version of this conference paper was presented at OCEANS 2024-Singapore and was selected for inclusion in the Student Poster Competition (SPC) Program

- **What's New**: 이번 연구에서는 강의실에서의 학생 평가를 자동화하고자 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 또한, 기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 무시하는 문제를 해결했습니다. 두 번째 연구에서는 deep model의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 사용하는 방법을 제안했습니다. 마지막 연구에서는 UAV 기반 원격 센싱과 Super-Resolution Reconstruction(SRR) 기법을 결합하여 CRAB-YOLO 검출 네트워크를 통해 해변 생태계를 모니터링하는 방법을 소개했습니다.

- **Technical Details**: 첫 번째 연구에서는 KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안했으며, 이는 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 연구에서는 '여러 개의' counterfactual을 생성하고 집합적 의사 결정 (collective decisions)을 통해 더 robust한 단어 인과 관계 파악을 시도했습니다. 세 번째 연구에서는 UAV와 SRR을 결합하여 CRAB-YOLO 네트워크가 움직임 블러 및 해상도 문제를 해결하도록 했으며, 이 모델은 YOLOv8s를 수정한 것입니다.

- **Performance Highlights**: 첫 번째 연구에서 KDA_disc와 KDA_cont는 강한 상관관계를 보였으며, MCQ 질의 다양한 전문 평가 기준에 대해 강력한 예측 능력을 입증했습니다. 두 번째 연구는 여러 측면에서 significant improvements를 보여주었습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 일반화. 세 번째 연구는 SRR 테스트 세트에서 평균정확도(mAP) 69.5%를 달성하며, Bicubic 방법보다 40% 향상된 성능을 보였습니다. 이는 해변 생태계 모니터링을 위한 경제적이고 자동화된 솔루션을 제공합니다.



### D2Styler: Advancing Arbitrary Style Transfer with Discrete Diffusion Methods (https://arxiv.org/abs/2408.03558)
Comments:
          Paper accepted at 27th International Conference on Pattern Recognition (ICPR), 2024

- **What's New**: 최근의 deep 모델이 NLP 작업에서 사람보다 뛰어난 정확도를 보이지만, spurious patterns에 의존해 robustness가 제한되고 있다는 보고가 있습니다. 이에 대처하기 위해 대비 학습(contrastive learning)과 가상 증강(counterfactual augmentation)을 활용하는 연구가 제안되었습니다.

- **Technical Details**: 기존 방식들은 사람이 가상 사례를 추가하거나 기계가 데이터셋에서 가상 사례와 유사한 항목을 찾아야 하는데, 이 과정에서도 spurious correlation의 영향을 받습니다. 이번 연구는 '여러 개의' 가상 사례를 생성하고, 집합적 의사 결정(collective decisions)을 통해 각 단어의 인과관계를 더욱 robust하게 감독하는 방법을 제안합니다.

- **Performance Highlights**: 이 접근 방식은 attribution-based synthesis의 편향에 덜 민감하며 다음과 같은 다양한 측면에서 유의미한 성과를 거두었습니다: 1) 가상 사례의 robustness, 2) 도메인 간 일반화(cross-domain generalization), 3) 부족한 데이터에서의 일반화(generalization from scarce data).



### VPOcc: Exploiting Vanishing Point for Monocular 3D Semantic Occupancy Prediction (https://arxiv.org/abs/2408.03551)
- **What's New**: 이 논문은 자동적으로 MCQ(Multiple Choice Questions)를 생성하는 방법에 새로운 지식 종속 가능성(KDA, Knowledge Dependent Answerability) 평가 메트릭을 제안합니다. 기존의 n-gram 기반 평가 메트릭, 예를 들어 BLEU, ROUGE 및 METEOR,는 교육적 가치를 고려하지 않기 때문에 MCQ의 교육적 유용성을 평가하는 데 한계가 있습니다.

- **Technical Details**: KDA 메트릭은 목표 사실(target fact)에 대한 학생의 지식을 기반으로 MCQ의 대답 가능성(answerability)을 측정합니다. 이 논문에서는 인간 설문조사를 통해 KDA를 측정하는 방법과 사전 학습된 언어 모델을 사용하여 KDA를 근사하는 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 (1) KDA와, (2) 전문가가 라벨링한 실제 교실 환경에서의 사용성과 강한 상관관계를 가짐을 보여주었습니다. 또한, n-gram 기반의 유사성 메트릭과 함께 사용할 경우 다양한 전문가 라벨링된 MCQ 품질을 예측하는데 높은 예측력을 가집니다.



### CLIP-based Point Cloud Classification via Point Cloud to Image Translation (https://arxiv.org/abs/2408.03545)
Comments:
          Accepted by ICPR2024

- **What's New**: 최근 여러 딥러닝 모델들은 NLP 태스크에서 사람보다 뛰어난 성능을 보였지만, 불순한 패턴에 의존해 그 견고함이 제한됨이 보고되었습니다. 이를 극복하기 위해 대비 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 더 견고한 모델을 구축하는 새로운 접근법이 제안되었습니다.

- **Technical Details**: ['KDA: MCQ의 대상 사실(target fact)에 대한 학생의 지식을 평가하는 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안하였습니다.', 'CLIP 기반 포인트 클라우드 분류: 최근 CLIP 기반의 포인트 클라우드 분류 모델인 PointCLIP이 등장하였으며, 우리는 이를 개선하기 위해 PPCITNet을 제안하여 포인트 클라우드와 시각 정보를 통합하는 네트워크를 도입했습니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont가 실제 강의실 세트에서 높은 사용성을 보여주었으며, 전문 평가자들에 의해 다양한 MCQ 품질 지표에 대한 예측력이 뛰어남이 입증되었습니다.', 'PPCITNet: ModelNet10, ModelNet40, ScanobjectNN 등 다양한 데이터셋에서 기존의 최첨단 CLIP 기반 모델들을 능가하는 성능을 보여주었습니다.']



### Automatic identification of the area covered by acorn trees in the dehesa (pastureland) Extremadura of Spain (https://arxiv.org/abs/2408.03542)
Comments:
          22 pages, 15 Figures, 2 Tables

- **What's New**: 최근 학술 논문에서 다양한 주제의 연구 결과들이 발표되었습니다. 첫 번째 논문은 MCQ(객관식 문제)의 자동 생성 평가를 위해 새로운 평가 메트릭을 제안했으며, 두 번째 논문은 대조 학습과 반사실적 증강(counterfactual augmentation)을 통한 NLP 모델의 로버스트니스 향상을 다루고 있습니다. 세 번째 논문은 항공 디지털 이미지를 활용하여 도토리 나무의 덮인 면적(CWA)을 자동으로 추정함으로써 이베리코 돼지의 생산량을 최적화하는 방법에 대해 논의하고 있습니다.

- **Technical Details**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Highlights': ['기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)를 제안.', 'KDA는 학생들이 대상 사실을 알 때 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 통해 이를 예측.', '휴먼 스터디 결과, KDA_disc와 KDA_cont는 실제 교실 환경에서의 활용도와 강한 상관관계를 가짐.']}, {'Paper': 'Robustness in NLP Models through Contrastive Learning and Counterfactual Augmentation', 'Highlights': ['NLP 태스크에서 deep model이 spurious pattern에 의존해 robustness가 제한되는 문제를 해결하기 위해 대조 학습과 반사실적 증강을 활용.', '기존 증강 방법이 spurious correlation에 영향을 받는 문제점을 극복하기 위해 여러 개의 반사실적을 생성하고, 집합적 의사결정을 통해 용어들의 인과관계를 파악하는 방법을 제안.', '이 접근법은 counterfactual robustness, cross-domain generalization, 및 scarce data에서의 일반화 등 다양한 차원에서 유의미한 성능 향상을 달성.']}, {'Paper': 'Automatic Estimation of Covered Wooded Area (CWA) in Spanish Dehesa using Aerial Digital Images', 'Highlights': ['스페인의 도토리 나무 덮인 면적(CWA)을 자동으로 추정하기 위해 항공 디지털 이미지(orthophotos)를 활용.', 'Gustafson-Kessel (GK)의 클러스터링 알고리즘을 Babuska 수정 버전(GK-B)으로 사용하여 자동 분할 방법을 제안.', '142 헥타르 면적에 대한 실험 결과가 실제 이미지와 손으로 분할된 이미지와 비교 시 유망한 결과를 보임.']}]

- **Performance Highlights**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Details': ['KDA_disc와 KDA_cont가 기존 n-gram 기반 메트릭과 결합했을 때 전문가가 레이블한 MCQ 품질 측정에 강한 예측력을 가짐.']}, {'Paper': 'Robustness in NLP Models through Contrastive Learning and Counterfactual Augmentation', 'Details': ['집합적 의사 결정 접근법을 사용하여 대조 학습과 반사실적 증강을 통해 다양한 NLP 응용 분야에서 중요한 성능 향상을 달성.']}, {'Paper': 'Automatic Estimation of Covered Wooded Area (CWA) in Spanish Dehesa using Aerial Digital Images', 'Details': ['제안된 방법이 도토리 나무의 덮인 면적 추정에서 성공적으로 적용되었으며, 이베리코 돼지 생산 최적화에 적용될 잠재력이 높다.']}]



### PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Mod (https://arxiv.org/abs/2408.03540)
- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하는 새로운 메트릭 지식 종속 가능성(KDA) 제안, 다양한 NLP 태스크에 있어서 강화된 robustness를 위한 대비 학습(contrastive learning) 및 반사실적 확장(counterfactual augmentation) 접근법 제안, Mamba 구조를 기반으로 한 3D 인간 포즈 추정(PoseMamba)에서의 선형 복잡도 기반의 새로운 접근 제안.

- **Technical Details**: KDA 메트릭은 실제 학생응답 데이터를 사용해 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont의 두 가지 자동 평가 메트릭을 제안. 대비 학습 및 반사실적 확장은 다수의 반사실적을 생성하고, 집합적 의사 결정을 통해 더 robust한 인과관계를 파악. PoseMamba는 양방향 글로벌-로컬 시공간 SSM 블록을 통해 인간 관절 관계와 시간적 상관관계를 모델링함. 보다 논리적인 기하학적 스캔 순서를 통합하여 공간 스캔을 결합함.

- **Performance Highlights**: KDA_disc와 KDA_cont가 실제 강의실 세트에서 유용성과 강한 상관관계를 가짐을 입증. 대비 학습 및 반사실적 확장 접근법이 비용과 메모리 효율성을 높이면서 반사실적 강건성, 도메인 간 일반화, 소량 데이터에서의 일반화에서 중요한 개선을 달성. PoseMamba가 Human3.6M 및 MPI-INF-3DHP 데이터셋에서 최첨단 성능을 달성, 모델 크기와 계산 비용을 줄임.



### PRTGS: Precomputed Radiance Transfer of Gaussian Splats for Real-Time High-Quality Relighting (https://arxiv.org/abs/2408.03538)
- **What's New**: MCQ 자동 생성의 평가 메트릭 개선을 위해 기존의 BLEU, ROUGE, METEOR가 아닌 새로운 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 메트릭을 제안했습니다. KDA는 학생이 해당 사실에 대한 지식을 알고 있는 상황에서 MCQ의 답할 가능성을 평가합니다.

- **Technical Details**: KDA 메트릭은 학생 응답을 통한 인간 조사 기반으로 측정되며, 이를 모방하여 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont 자동 평가 메트릭을 제안했습니다. 또한, 기존의 n-gram 기반 유사성 메트릭과 결합하여 예측력을 향상시켰습니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont 메트릭은 실제 교실 환경에서의 사용성과 강한 상관관계를 갖고 있으며, 전문가들이 라벨링한 다양한 MCQ 품질 측정 지표에 대해 높은 예측력을 보였습니다.



### SwinShadow: Shifted Window for Ambiguous Adjacent Shadow Detection (https://arxiv.org/abs/2408.03521)
- **Automatic MCQ Generation**: {"What's New": '기존의 평가 메트릭들이 MCQ의 교육적 가치를 평가하지 못하는 문제를 해결하기 위해, 대상 사실(target fact)에 대한 학생의 지식을 측정하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)을 제안하였다.', 'Technical Details': 'KDA는 학생 설문조사에서 얻은 응답을 기반으로 측정된다. 또한 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하였다.', 'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_soft가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 확인하였다. n-gram 기반의 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강력한 예측력을 보였다.'}

- **NLP 모델의 Robustness 향상**: {"What's New": '최근 deep 모델들의 높은 정확도에도 불구하고, spurious 패턴에 의존하여 robustness가 떨어지는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방안을 제안하였다.', 'Technical Details': '기존의 augmentation 방법이 데이터셋에서 인간이 counterfactual을 추가하거나 모델이 유사한 것을 자동으로 찾는 것과는 달리, 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 각 용어의 인과관계를 강하게 감독하는 방법을 사용하였다.', 'Performance Highlights': '집합적 의사결정을 통해 attribution 기반의 합성 편향에 덜 민감하게 되어, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 등 다각적인 측면에서 유의미한 향상을 달성하였다.'}

- **Shadow Detection using SwinShadow**: {"What's New": '인접한 그림자를 탐지하는 데 어려움을 겪는 기존 방법을 개선하기 위해, Swin Transformer를 활용한 SwinShadow라는 새로운 아키텍처를 제안하였다.', 'Technical Details': '이 메커니즘은 두 단계로 작동한다. 첫째, 단일 창에서 로컬 자기 주의(Local Self-Attention)를 적용하여 네트워크가 로컬 디테일에 집중하게 한다. 그런 다음, 주의 창을 이동시켜 창 간 주의(Inter-Window Attention)를 촉진하여 인접 정보를 더 넓게 포착한다. 이러한 단계는 그림자와 인접 객체의 구분 능력을 크게 향상시킨다.', 'Performance Highlights': 'SBU, UCF 및 ISTD 데이터셋을 대상으로 한 광범위한 실험에서, 우리의 네트워크는 balance error rate (BER) 측면에서 우수한 성능을 입증하였다.'}



### Leveraging LLMs for Enhanced Open-Vocabulary 3D Scene Understanding in Autonomous Driving (https://arxiv.org/abs/2408.03516)
- **What's New**: 자동 MCQ 생성의 교육적 평가를 위해 BLEU, ROUGE, METEOR와 같은 기존 메트릭 대신, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 이는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: 1. 지식 종속 가능성(KDA)을 측정하기 위해 그간 ble 필연적으로 spurious correlation에 영향을 받았습니다. 이를 극복하기 위해 여러 개의 대조적 학습(counterfactual)을 생성하여 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 
 2. LE3DGS와 LLMs를 결합하여 자기주행 차량의 3D 장면 이해력을 향상시킵니다. 대용량 언어 모델(LLM)을 활용한 문맥적 표현 생성으로, 기존 방법보다 더 유연하고 적응력 있는 장면 분석을 가능하게 합니다.

- **Performance Highlights**: 1. 인간 조사를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 보여주었습니다. 이는 KDA와의 상관관계뿐만 아니라 전문가가 라벨링한 실제 MCQ 품질 평가에도 유효하였습니다. 
 2. 여러 대조적 학습을 통해 다양한 차원에서 의미 있는 향상을 달성하였으며, 특히 카운터팩츄얼 robustness, 크로스 도메인 일반화, 적은 데이터에서의 일반화를 보여줍니다. 
 3. WayveScenes101 데이터셋에서 LE3DGS와 LLMs를 결합한 방법이 기존의 최첨단 방법보다 높은 정확도와 적응력을 보였습니다.



### MoExtend: Tuning New Experts for Modality and Task Extension (https://arxiv.org/abs/2408.03511)
Comments:
          ACL 2024 - SRW

- **What's New**: 자동 MCQ 생성 기술을 개선하는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, 딥러닝 모델의 강화를 위해 새로운 counterfactual 데이터 증강 방법을 도입하고 있어서 NLP 모델의 robustness를 높입니다. 마지막으로, 멀티모달 대형 언어 모델 (Large Language Models, LLMs)을 더 효율적으로 확장하기 위한 MoExtend 프레임워크가 소개되었습니다.

- **Technical Details**: MCQ 생성의 경우, KDA 메트릭은 MCQ가 학생의 지식을 평가할 수 있는 능력을 측정합니다. 이는 학생 응답 기반의 평가를 통해 검증되며, pre-trained language models를 활용하여 학생의 문제풀이 행동을 모방하는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. NLP 모델의 robustness 강화를 위해서는 spurious 패턴의 문제를 해결하기 위해 대조 학습 (contrastive learning) 및 counterfactual 증강 방법을 사용합니다. 마지막으로, MoExtend 프레임워크는 믹스처 오브 전문가 (Mixture-of-Experts, MoE) 모델에 새로운 모달리티 전용 전문가를 원활하게 통합하여 멀티모달 LLMs를 확장시킵니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅에서의 사용성 평가와 높은 상관관계를 보였으며, expert-labeled MCQ의 품질 측정에서도 높은 예측 능력을 나타냈습니다. Counterfactual 로버스트니스와 도메인 간 일반화, 데이터 부족 상황에서의 일반화에서 기존 방법 대비 significant improvements를 달성했습니다. MoExtend는 고속의 학습 속도와 낮은 catastrofic forgetting 리스크로 멀티모달 기능을 효율적으로 확장하는 데 성공했습니다.



### GUI Element Detection Using SOTA YOLO Deep Learning Models (https://arxiv.org/abs/2408.03507)
- **What's New**: 이번 연구는 교육자가 학생 평가에 소비하는 시간을 크게 줄여줄 수 있는 자동 Multiple Choice 질문(MCQ) 생성에 관한 핵심 문제를 다룹니다. 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 생성된 MCQ의 단순한 단어 유사성만을 측정하고 교육적 가치를 평가하지 않으며, 따라서 새롭게 제안된 Knowledge Dependent Answerability(KDA) 메트릭이 MCQ의 지식 기반 대답 가능성(Answerability)을 측정함으로써 이를 해결하려고 합니다.

- **Technical Details**: 연구는 먼저 KDA를 학생 응답을 기반으로 측정하는 방법을 제시하며, 이어서 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안합니다. 또한, 이 메트릭이 실제 강의실 세트에서의 사용성과 강한 상관관계를 가진다는 것을 인간 실험을 통해 증명했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 전문가들이 라벨링한 다양한 MCQ 품질 측정치에 대해 높은 예측력을 보이며, n-gram 기반의 유사성 메트릭과 결합함으로써 더욱 강력한 성능을 나타냈습니다.



### Opening the Black Box of 3D Reconstruction Error Analysis with VECTOR (https://arxiv.org/abs/2408.03503)
- **What's New**: 이번 연구에서는 자동 Multiple Choice Questions (MCQ) 생성을 목표로 하는 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안합니다. 기존 평가 메트릭은 블루(BLEU), 루즈(ROUGE), 미티어(METEOR)와 같이 MCQ의 단순한 단어 유사성을 측정하지만, 이번 연구에서는 학생들의 지식을 평가하는 능력까지 고려한 새로운 접근법을 도입했습니다.

- **Technical Details**: KDA는 당사자의 지식을 바탕으로 MCQ가 대답 가능한지 측정하는 새로운 평가 메트릭입니다. 이를 위해 우리는 먼저 인간 설문 조사에서 학생의 응답을 통해 KDA를 측정하는 방법을 제시했습니다. 그 다음, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 이용해 학생들의 문제 해결 행위를 모방하는 방식을 사용했습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, 다양한 전문가가 라벨링한 MCQ 품질 지표에 대한 예측력이 강하다는 것이 입증되었습니다.



### e-Health CSIRO at RRG24: Entropy-Augmented Self-Critical Sequence Training for Radiology Report Generation (https://arxiv.org/abs/2408.03500)
- **What's New**: 최근 강의실에서 사용할 수 있는 자동 다중 선택 질문(MCQ) 생성 메트릭을 개선하기 위해 '지식 종속 가능성(KDA)' 메트릭을 제안했습니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 MCQ의 교육적 가치를 평가하지 못한다는 문제를 해결합니다. 또한, 전통적으로 사람이 counterfactual을 만들어야 했던 문제를 해결하기 위해 '여러 개'의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust한 인과관계를 찾아내는 접근법을 제안합니다.

- **Technical Details**: 1. 자동 MCQ 생성에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다. 이 메트릭들은 학생의 문제 해결 행동을 모방하기 위해 사전 학습된 언어 모델을 활용합니다. 
2. NLP 과제에서는 contrastive learning과 counterfactual augmentation을 이용한 robustness 향상 방법을 제안했습니다. 이 방법은 spurious correlation의 영향을 줄이기 위해 여러 counterfactual을 생성하고 집합적 의사 결정을 하는 방식을 도입하였습니다. 
3. 대규모 방사선 보고서 생성(RRG24)에서는 entropy regularisation을 추가한 SCST(자기 비판적 시퀀스 훈련) 기법을 사용하여 단어 분포의 엔트로피를 높였습니다.

- **Performance Highlights**: 1. MCQ 생성을 위한 KDA_disc와 KDA_cont는 실제 강의실 세트의 사용성과 강한 상관관계가 있음을 입증하였습니다. 
2. 새로운 NLP 향상 방법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등에서 유의미한 성능 개선을 보여주었습니다. 
3. RRG24 에서 사용된 Entropy-Augmented Self-critical Sequence Training (EAST) 기법은 첫 번째 항목에서 여러 번 1위를 하는 성과를 거두었습니다.



### FacialPulse: An Efficient RNN-based Depression Detection via Temporal Facial Landmarks (https://arxiv.org/abs/2408.03499)
- **What's New**: 이번 뉴스레터에서는 최근 아카이브(arXiv)에 실린 세 가지 주요 논문에 대해 소개합니다. 첫 번째 논문은 Multiple Choice Questions (MCQ)의 자동 생성과 평가를 혁신하는 새로운 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. 두 번째 논문은 NLP 태스크에서 모델의 robustness(견고성)을 향상시키기 위해 contrastive learning와 counterfactual augmentation을 활용하는 방법을 다룹니다. 세 번째 논문은 우울증 조기 진단을 위한 얼굴 표정 인식 모델인 FacialPulse를 제안합니다.

- **Technical Details**: [{'Multiple Choice Questions (MCQ) Evaluation': '기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR은 교육적 가치를 반영하지 못하는 문제점이 있습니다. 이에 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA_disc와 KDA_cont는 프리트레인된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방해 KDA를 근사합니다.'}, {'NLP Robustness Enhancement': '최근 NLP 모델들이 높은 정확성을 보였지만 spurious patterns(의미 없는 패턴)에 의존해 robustness가 제한됩니다. 이를 극복하기 위해 contrastive learning과 counterfactual augmentation을 도입한 방법을 제안합니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 각 용어의 인과관계를 더 견고하게 감독합니다.'}, {'Automatic Depression Detection': '우울증 조기 진단을 위해 FacialPulse라는 새로운 프레임워크를 제안합니다. Facial Motion Modeling Module (FMMM)과 Facial Landmark Calibration Module (FLCM)을 사용해 얼굴 표정에서 시간적 특성을 효과적으로 캡처합니다. BiGRU 백본을 사용해 긴 의존성을 처리하고, 얼굴 랜드마크 교정 모듈을 통해 정보 중복을 줄이며 정확성을 높입니다.'}]

- **Performance Highlights**: [{'MCQ Evaluation': 'Human evaluation 실험을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 확인했습니다. 이를 n-gram 기반의 유사성 메트릭과 결합할 경우 다양한 MCQ 품질 측정에 높은 예측력을 가집니다.'}, {'NLP Robustness': '본 접근법은 집합적 의사 결정을 통해 task model bias에 덜 민감하게 반응해 counterfactual robustness, cross-domain generalization, 그리고 data scarcity(데이터 부족) 상황에서의 일반화 등 여러 차원에서 유의미한 향상을 이루었습니다.'}, {'Depression Detection': 'FacialPulse는 AVEC2014 및 MMDA 데이터셋에서 뛰어난 성능을 보였으며, 평균 절대 오차(MAE)를 21% 감소시키고 인식 속도를 100% 향상시켰습니다.'}]



### AI Foundation Models in Remote Sensing: A Survey (https://arxiv.org/abs/2408.03464)
- **What's New**: 이 논문에서는 MCQ(다중 선택 문제) 생성을 위한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 기존의 BLEU, ROUGE, METEOR 같은 메트릭은 단어 유사성에만 초점을 맞추어 교육적 가치를 평가하지 못한다는 문제점을 지적하며, KDA는 학생의 지식을 실제로 평가할 수 있는지를 측정합니다.

- **Technical Details**: KDA는 목표 사실에 대해 학생이 대답할 수 있는지 측정하는 새로운 메트릭입니다. 이를 위해 인간 설문 조사를 통해 KDA를 측정하는 방법을 제시하고, 미리 학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이는 대답 가능성 (answerability)을 수치화하여 평가합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 설정에서 전문가들에 의해 라벨링된 사용성과 강한 상관관계가 있음을 보였습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 때, 이 메트릭들은 다양한 전문가 라벨링 MCQ 품질 측정치에 대해 강력한 예측 능력을 보였습니다.



### Hybrid diffusion models: combining supervised and generative pretraining for label-efficient fine-tuning of segmentation models (https://arxiv.org/abs/2408.03433)
Comments:
          19 pages

- **What's New**: 이 논문은 기존의 자동 MCQ 생성 평가 메트릭이 교육적 가치를 고려하지 않는 점을 지적하며, 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. 또한, 딥러닝 모델의 강건성을 높이기 위해 대조 학습 (contrastive learning)과 반사실 증강 (counterfactual augmentation)을 활용한 방법을 소개합니다. 마지막으로, 레이블 효율적 학습을 목표로 하여 두 영역 간 세그멘테이션(분할) 모델 효과적으로 적응하는 새로운 접근을 제안합니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 메트릭이 n-그램 기반 유사성만 평가하는 문제를 해결하기 위해, KDA를 통해 MCQ 평가 방식에서 학생의 지식을 효과적으로 평가할 수 있도록 합니다. 이는 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont를 제안합니다.', 'Robustness in NLP': '최근 딥 모델이 NLP 태스크에서 높은 정확성을 보였음에도 불구하고, 임의 패턴에 의존하여 강건성이 제한되어 있다는 문제를 극복하기 위해 대조 학습과 반사실 증강을 사용합니다. 기존 방법은 데이터셋에서 반사실을 사람이 직접 추가하거나 모델이 자동으로 찾게 하지만, 이 논문에서는 다양한 반사실을 생성해 집합적 의사 결정을 통해 단어 인과관계를 더욱 강건하게 파악하는 방법을 소개합니다.', 'Label-efficient Fine-tuning': '대규모 레이블 데이터가 존재하는 도메인에서 다습득 학습을 진행한 후, 레이블이 적은 관련 도메인으로 모델을 효율적으로 적응시키기 위한 새 방법을 제안합니다. 첫 번째 도메인에서는 세그멘테이션 모델과 확산 모델을 동시 학습해 고품질 표현을 얻고, 두 번째 도메인에서의 소량 데이터로 모델을 미세 조정(fine-tune) 합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human evaluation에서 KDA_disc와 KDA_cont가 MCQ 평가와 실제 강의실 사용에서 높은 상관관계를 보였습니다. 또한, 이 메트릭들이 전문가가 라벨링한 MCQ 품질 척도에 대해 강력한 예측력을 가지고 있음을 입증했습니다.', 'Robustness in NLP': '제안된 방법이 다양한 측면에서 큰 성과를 보였습니다: 1) 반사실 강건성 (counterfactual robustness), 2) 교차 도메인 일반화 (cross-domain generalization), 3) 희소 데이터에서의 일반화 (generalization from scarce data). 집합적 의사 결정 방식이 모델 편향에 덜 민감하다는 것이 입증되었습니다.', 'Label-efficient Fine-tuning': '제안된 프리텍스트 태스크 (image denoising과 mask prediction을 동시에 수행)가 있는 사전 학습된 모델을 미세 조정한 결과, 기존의 지도 또는 자가 지도 학습만으로 훈련된 모델보다 더 나은 성과를 보였습니다.'}]



### Set2Seq Transformer: Learning Permutation Aware Set Representations of Artistic Sequences (https://arxiv.org/abs/2408.03404)
- **What's New**: 최근 발표된 논문에서는 교육적 가치를 잘 반영하지 못하는 기존의 MCQ 생성 평가 메트릭의 한계를 지적하며, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, NLP 태스크에서 deep model의 robustness를 개선하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 새로운 방법을 소개했습니다. 마지막으로, Set2Seq Transformer라는 새로운 순차적 다중 인스턴스 학습 아키텍처를 선보여 시각적 내용 분석에 한정되지 않고 순차적 정보까지 고려한 학습을 통해 예술적 성공을 예측하는 성능을 크게 향상시켰습니다.

- **Technical Details**: [{'Metric Improvement': 'BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭은 문장 유사도에 초점을 맞췄지만 KDA는 학생의 지식 기반 대답 가능성을 평가합니다. KDA는 human survey를 통해 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 pre-trained language models을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Robustness Enhancement': '기존의 augmentation 방식이 spurious correlation에 약하다는 문제를 해결하기 위해, 여러 개의 counterfactual을 생성하고 집합적 의사 결정 (collective decisions)을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안했습니다. 이는 counterfactual robustness, cross-domain generalization, 그리고 scarce data generalization에서 성능 개선을 가져왔습니다.'}, {'Set2Seq Transformer': 'Set2Seq Transformer는 시퀀스의 permutation aware set 표현을 학습하는 아키텍처로, 시각적 콘텐츠 분석뿐만 아니라 시간적 정보까지 통합하여 예술적 성공 예측에 활용됩니다. WikiArt-Seq2Rank라는 새로운 데이터셋과 visual learning-to-rank라는 하위 태스크를 활용해 extensive quantitative and qualitative evaluation을 수행했습니다.'}]

- **Performance Highlights**: [{'KDA Performance': 'Human studies 결과, KDA_disc와 KDA_cont는 KDA와 강한 상관관계를 가지며 실제 강의실 세트에서의 사용성 또한 뛰어났습니다. n-gram 기반 유사도 메트릭과 결합 시 다양한 expert-labeled MCQ quality measures에 대해 강한 예측력을 보였습니다.'}, {'Counterfactual Robustness': 'Collective decisions을 통해 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 generalization from scarce data 측면에서 significant improvements가 관찰되었습니다.'}, {'Set2Seq Transformer Success': 'Set2Seq Transformer는 시각적 세트와 시간적 위치 인식 표현을 결합하여 예술적 성공 예측 성능을 크게 향상시켰으며, WikiArt-Seq2Rank 데이터셋을 사용한 평가에서 강력한 static 및 sequential multiple instance learning methods를 능가하는 성과를 보였습니다.'}]



### RayGauss: Volumetric Gaussian-Based Ray Casting for Photorealistic Novel View Synthesis (https://arxiv.org/abs/2408.03356)
Comments:
          Project page with videos and code: this https URL

- **What's New**: ['자동 MCQ 생성을 평가하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)을 제안합니다. 이는 학생의 지식을 평가하는 능력을 중점으로 합니다.', '기존의 딥 러닝 모델이 NLP 태스크에서 사람보다 높은 정확성을 보였으나, spurious pattern에 의존하는 문제로 인한 제한된 robustness를 보완하기 위해 contrastive learning과 counterfactual augmentation을 활용한 새로운 접근법을 소개합니다.', 'Neural Radiance Fields (NVS)의 성능을 향상시키기 위해 differentiable volumetric rendering을 기반으로 하는 새로운 방법을 제안하여 높은 품질의 렌더링을 달성합니다.']

- **Technical Details**: [{'paper': 'MCQ Generation', 'details': '제안된 KDA는 대상 사실에 대한 학생의 지식을 기준으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 사용하여 사전 훈련된 언어 모델을 통해 학생의 문제 해결 행동을 모방합니다.'}, {'paper': 'NLP Robustness', 'details': '기존의 augmentation 방법과 달리 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 용어의 인과관계를 더욱 robust하게 파악합니다. 이를 통해 counterfactual robustness, cross-domain generalization, 및 scarce data에서의 일반화에서 상당한 향상을 달성합니다.'}, {'paper': 'Volumetric Rendering', 'details': '발광 radiance c와 밀도 σ를 Spherical Gaussians/Harmonics로 분해하는 물리적으로 일관된 공식화를 제공합니다. 이 방법은 불규칙하게 분포된 Gaussians의 differentiable ray casting을 사용하여 장면에 세밀하게 적응하면서 splatting artifacts를 피합니다.'}]

- **Performance Highlights**: [{'paper': 'MCQ Generation', 'performance': 'KDA_disc와 KDA_cont는 실제 강의실에서 전문가들이 라벨링한 사용성과 강한 상관관계를 보여줍니다. 또한, n-gram 기반 유사성 메트릭과 결합할 때 전문가 라벨 MCQ 품질 측정에 대한 강한 예측력을 가집니다.'}, {'paper': 'NLP Robustness', 'performance': '제안된 접근법은 attribution-based synthesis의 task model bias에 덜 민감하여 다양한 차원에서 상당한 향상을 달성합니다.'}, {'paper': 'Volumetric Rendering', 'performance': 'Blender 데이터셋에서 25 FPS의 추론 속도로 최첨단 렌더링 품질을 달성하면서도 합리적인 훈련 시간을 유지합니다.'}]



### FastEdit: Fast Text-Guided Single-Image Editing via Semantic-Aware Diffusion Fine-Tuning (https://arxiv.org/abs/2408.03355)
Comments:
          Technical Report

- **What's New**: MCQ(객관식 문항) 자동 생성은 교육자들의 평가 시간을 크게 줄일 수 있는 잠재력을 가지고 있으나, 기존의 평가지표는 교육적 가치를 평가하지 못합니다. 이를 해결하기 위해 대상 사실에 대한 학생의 지식 평가 능력을 측정하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.

- **Technical Details**: KDA는 대상 사실에 대한 지식을 기반으로 MCQ의 대답 가능성(answerability)을 측정합니다. 이를 위해, 먼저 인간 평가 데이터에서 나온 응답을 통해 KDA 측정 방법을 보입니다. 그런 다음, 사전 훈련된 언어 모델을 활용한 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)를 제안하여 학생 문제 해결 방식을 모방합니다.

- **Performance Highlights**: Human study를 통해, KDA_disc와 KDA_cont가 실제 교실 세팅에서 사용성과 강한 상관 관계를 가지고 있음을 확인했습니다. n-gram 기반 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강력한 예측력을 가집니다.

- **New Approach**: 최근의 딥러닝 모델들이 NLP 태스크에서 매우 높은 정확도를 보였음에도 불구하고, spurious patterns(스퓨리어스 패턴)에 의존하여 robustness(견고성)가 떨어지는 문제를 해결하기 위해 대비 학습(contrastive learning)과 반사실적 데이터 증가(counterfactual augmentation)를 활용하는 방법을 제안합니다.

- **Method**: 기존 방법들은 사람이 counterfactual 데이터를 생성하거나 데이터셋에서 유사한 데이터를 찾아야 했습니다. 제안된 방법은 여러 개의 counterfactual을 생성하고, 그 집합을 기반으로 집합적 의사결정을 통해 단어의 인과 관계를 더욱 견고하게 파악합니다.

- **Results**: 제안된 방법은 다양한 측면에서 상당한 성능 향상을 보여주었으며, 특히 1) 반사실적 데이터 견고성, 2) 교차 도메인 일반화, 3) 희소한 데이터에서의 일반화에서 큰 개선을 이룩했습니다.

- **New Development**: FastEdit는 text-guided single-image editing(텍스트 기반 단일 이미지 편집)을 17초 내에 수행 가능하게 하는 새로운 방법입니다. 이는 기존 방법이 7분 정도 소요되는 데 비해 크게 개선된 것입니다.

- **Acceleration Techniques**: FastEdit는 semantic-aware diffusion fine-tuning 전략을 사용하여 세분화된 시간 단계 값(time step)을 기반으로 학습을 가속화하고, LoRA와 같은 파라미터 효율적 미세 조정 기법을 도입하여 모델의 학습 가능한 파라미터를 전체의 0.37%로 줄였습니다.

- **Capabilities**: FastEdit는 내용 추가, 스타일 전환, 배경 교체, 자세 수정 등 다양한 편집 기능에서 뛰어난 성능을 보입니다.



### InLUT3D: Challenging real indoor dataset for point cloud analysis (https://arxiv.org/abs/2408.03338)
- **What's New**: 이번 뉴스레터에서는 세 가지 흥미로운 연구를 소개합니다. 첫 번째 연구는 MCQ(Multiple Choice Questions) 자동 생성의 교육적 가치를 평가하기 위한 새로운 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 두 번째 연구는 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 NLP 모델의 강인성을 향상시키는 방법을 제안합니다. 마지막으로 InLUT3D라는 새로운 실내 포인트 클라우드 데이터셋을 소개하여 3D 장면 이해 분야를 발전시키고자 합니다.

- **Technical Details**: {'MCQ Generation': '기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성만을 평가하여 교육적 가치를 반영하지 못했습니다. 이를 해결하기 위해 KDA를 제안하며, 학생의 응답 데이터를 바탕으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입합니다.', 'NLP Robustness': '기존 방법들이 spurious pattern에 문제가 있어 반사실적 생성(counterfactual generation)을 통해 robust한 학습을 제안합니다. 이 방법은 여러 counterfactual을 생성하고 이를 바탕으로 집합적 의사 결정을 내리는 방식입니다.', 'InLUT3D Dataset': 'Lodz University of Technology에서 수집한 고해상도 레이저 기반 포인트 클라우드를 포함한 실내 공간을 다룹니다. 여러 실내 환경을 수집하여 3D 장면 이해를 위한 새로운 벤치마크를 제공합니다.'}

- **Performance Highlights**: {'MCQ Generation': '사람이 직접 평가한 결과 KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 높은 상관관계를 나타냈습니다. 또한, n-gram 기반 유사성 메트릭과 결합하면 더욱 강력한 예측력을 보여줍니다.', 'NLP Robustness': '제안된 방법은 기존의 방법보다 착오기반 합성작업에 덜 민감하고, 다양한 측면에서 현저한 성능 향상을 보여줬습니다: 1) 반사실적 강인성, 2) 도메인 간 일반화, 3) 부족한 데이터에서의 일반화.', 'InLUT3D Dataset': '321개의 장면과 3.5억 개 이상의 포인트를 포함하며, 각 포인트는 카르티션 좌표, RGB 컬러 값, 카테고리 코드 및 인스턴스 식별자로 설명됩니다. 이를 통해 학술 연구자들이 더욱 신뢰할 수 있는 방법론 비교를 할 수 있도록 지원합니다.'}



### Reconstruction of the shape of irregular rough particles from their interferometric images using a convolutional neural network (https://arxiv.org/abs/2408.03327)
- **What's New**: 이 논문들에서는 새롭고 혁신적인 방법들을 소개하고 있습니다: 
1. 자동 MCQ 생성의 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다.
2. NLP 태스크에서 모델의 robustness을 높이기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 논의합니다.
3. 비정형 입자의 3D 모양을 재구성하기 위해 CNN 기반 모델을 개발했습니다.

- **Technical Details**: 논문들은 다음과 같은 기술적인 세부사항들을 다룹니다:
1. KDA 메트릭은 학생들의 응답을 기반으로 측정하며, 인간 연구를 통해 KDA_disc와 KDA_cont의 유용성을 확인합니다. 모델은 사전 학습된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방합니다.
2. 새로운 augmentation 방법은 여러개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust한 모델을 구현합니다. 기존 방법들이 spurious correlation에 취약하다는 문제를 해결합니다.
3. CNN은 UNET 아키텍처와 residual block 모듈을 기반으로 하며, 실험적 패턴 데이터를 활용하여 학습되었습니다. CNN은 다양한 비정형 입자의 3D 재구성을 성공적으로 수행합니다.

- **Performance Highlights**: 논문들은 다음과 같은 성과를 보여줍니다:
1. KDA_disc와 KDA_cont는 KDA와 전문가가 라벨링한 실제 강의실 사용성과 강한 상관관계를 가집니다. n-gram 기반 메트릭과 결합 시 MCQ의 품질 측정에 높은 예측력을 보입니다.
2. 제안된 augmentation 방법은 기존의 attribution-based 방법보다 더 robust하며, 다양한 차원에서 성능 향상을 달성합니다. 예를 들어, counterfactual robustness, cross-domain generalization, 그리고 부족한 데이터로부터의 일반화에서 높은 성능을 보입니다.
3. CNN은 비대칭 및 비대칭 입자의 모양을 높은 정확도로 재구성하며, 18000개의 실험적 interferometric 이미지를 활용하여 학습되었습니다.



### Lightweight Video Denoising Using a Classic Bayesian Backbon (https://arxiv.org/abs/2408.03904)
Comments:
          Paper accepted to ICME 2024

- **What's New**: 각각 자동 MCQ 생성의 새로운 지식 종속 가능성(KDA) 메트릭, 많은 매개변수를 필요로 하지 않으면서도 높은 성능을 나타내는 하이브리드 Wiener 필터 도입, 그리고 대체적 학습을 통한 강화 학습 방법을 활용한 NLP 모델의 향상된 robust 성능 기술 등이 제안되었습니다.

- **Technical Details**: 1. MCQ 생성 논문에서는 기존의 BLEU, ROUGE, METEOR 메트릭이 가진 문제를 지적하고, KDA라는 새로운 평가 메트릭을 제안했습니다. KDA는 학생의 지식을 바탕으로 문제의 답변 가능성을 측정합니다.
2. NLP 논문에서는 대체적 학습(contrastive learning)과 counterfactual augmentation을 통해 기존의 spurious 패턴 문제를 극복하고 인과 관계를 더욱 명확히 파악하는 접근 방식을 채택했습니다.
3. 하이브리드 Wiener 필터 논문에서는 기존 Bayesian denoising 기법을 최적화하는 대신에 작은 보조 네트워크를 사용하여 성능을 높이면서도 빠른 속도를 유지하는 방법을 제안했습니다.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 전문가들이 라벨링한 MCQ 품질 측정에서 높은 예측력을 나타냈습니다.
2. 대체적 학습 및 counterfactual augmentation을 활용한 기법은 다양한 차원에서의 성능 향상을 보여주며, 특히 counterfactual robustness와 cross-domain generalization이 두드러졌습니다.
3. 하이브리드 Wiener 필터는 PSNR 및 SSIM 측정에서 DVDNet, FastDVDNet, VNLB 등의 기존 메서드를 능가하며, VRT transformer 대비 10배 이상의 속도 성능을 보였습니다.



### Using a Distance Sensor to Detect Deviations in a Planar Surfac (https://arxiv.org/abs/2408.03838)
- **What's New**: 이 논문에서는 자동 다지선다형 질문(MCQ) 생성을 평가하기 위해 기존의 n-gram 기반 메트릭이 아닌, 새로운 자동 평가 메트릭인 지식 종속 대답 가능성(KDA)을 제안합니다. KDA는 생성된 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 

- **Technical Details**: KDA는 학생 응답을 기반으로 측정됩니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방해 KDA를 근사합니다. 다음으로, deep model의 걸림돌인 spurious pattern 문제를 극복하기 위해 대조 학습 (contrastive learning) 및 반사실적 보강 (counterfactual augmentation) 방법을 이용한 연구도 제시됩니다. 마지막으로, 소형 광 시차 센서 (optical time-of-flight sensor)를 이용해 평면 표면의 기하학적 편차를 감지하는 방법도 다루고 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보이며, experts가 라벨링한 MCQ 품질 측정 기준에 대해 강한 예측력을 가집니다. 대조 학습과 반사실적 보강을 통한 접근법은 다양한 측면에서 높은 성능을 보여줍니다. 예를 들어, 반사실적 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 그리고 희소한 데이터로부터의 일반화(generalization from scarce data)에서 유의미한 개선을 보입니다. 소형 광 시차 센서를 이용한 방법은 기존의 거리 추정치만 이용하는 방식보다 평면 표면의 기하학적 편차를 더 정확하게 감지합니다.



### Towards Real-Time Gaussian Splatting: Accelerating 3DGS through Photometric SLAM (https://arxiv.org/abs/2408.03825)
Comments:
          This extended abstract has been submitted to be presented at an IEEE conference. It will be made available online by IEEE but will not be published in IEEE Xplore. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 최신 논문에서 새로운 지식 종속 가능성 지표(Knowledge Dependent Answerability, KDA)를 제안하여, 자동으로 생성된 객관식 질문의 학습 평가 능력을 측정합니다. 또한, 딥러닝 기반 자연어처리 모델이 반사실 데이터(Conterfactual data)를 통해 보다 견고하게 학습할 수 있는 새로운 방법을 탐구합니다. 마지막으로, 3D Gaussian Splatting (3DGS)과 Direct Sparse Odometry (DSO)를 결합하여 실시간으로 운영 가능한 고품질의 3D 시각적 동시 위치 지정 및 지도 작성(VSLAM) 시스템을 제안합니다.

- **Technical Details**: MCQ 생성 평가를 위한 KDA는 학생의 응답을 바탕으로 MCQ의 답변 가능성을 측정합니다. 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용하여 학생의 문제 해결 행위를 모사합니다. 딥러닝 모델의 견고성을 향상시키기 위해 반사실 데이터를 생성하고 이를 사용하여 단어의 인과관계를 통제하는 방식을 제안합니다. VSLAM의 3DGS 통합을 위해 DSO의 포인트 클라우드 출력을 사용하여 학습 시간을 단축시켰습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서 MCQ의 사용성과 강한 상관관계를 보였습니다. 반사실 데이터를 사용한 학습 방법은 모델의 견고성, 도메인 간 일반화, 희귀 데이터에서의 일반화를 개선시켰습니다. DSO에서 생성된 고밀도 포인트 클라우드의 사용으로 인해 3DGS 학습 시간이 크게 단축되었으며, 실험 결과 더 빠르고 고품질의 3D 장면 렌더링을 가능하게 했습니다.



### Counterfactuals and Uncertainty-Based Explainable Paradigm for the Automated Detection and Segmentation of Renal Cysts in Computed Tomography Images: A Multi-Center Study (https://arxiv.org/abs/2408.03789)
- **What's New**: 새로운 MCQ 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)' 제안. NLP model의 robustness를 위해 기존 방법들을 보완하는 새로운 방법 제안. CT 스캔 이미지의 정확한 segmentation을 위해 해석 가능한 프레임워크 개발.

- **Technical Details**: KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모사하여 KDA를 근사하는 메트릭. contrastive learning와 counterfactual augmentation을 사용하여 다양한 차원에서 모델의 robustness를 향상. 3D input patches의 잠재 표현을 학습하여 변형하는 VAE-GAN 사용.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강의실 세트에서 높은 사용성과 상관관계를 가짐. 집합적 의사 결정으로 여러 차원에서 성능 개선. 원본 및 VAE-GAN 재구성 이미지의 DSCs에 유의미한 차이 없음, segmentation 성능에 영향을 미치는 이미지 특징을 시각화하고 모델 불확실성 분석 가능.



### Unsupervised Detection of Fetal Brain Anomalies using Denoising Diffusion Models (https://arxiv.org/abs/2408.03654)
Comments:
          Accepted at ASMUS@MICCAI 2024

- **Automatic MCQ Generation**: [{"What's New": '기존의 MCQ 생성 평가 메트릭이 교육적 가치를 무시하는 문제를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. KDA는 학생이 대상 사실에 대한 지식을 기반으로 MCQ에 답할 수 있는지를 평가합니다.', 'Technical Details': 'KDA는 인간 설문을 통한 학생 응답을 기반으로 측정됩니다. 또한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human studies에서 KDA_disc와 KDA_soft가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 보였습니다. n-gram 기반의 유사성 메트릭과 결합할 때 KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정에서 강력한 예측력을 가집니다.'}]

- **Robust NLP Models**: [{"What's New": '최근 deep models가 NLP 태스크에서 사람보다 더 높은 정확성을 보였으나, spurious pattern에 의존하는 문제로 robustness가 제한된다는 보고가 있습니다. 이를 개선하기 위해대비 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다.', 'Technical Details': '이 접근법은 여러 개의 반사실적(counterfactual)을 생성하고 집합적 의사 결정을 통해 더욱 robust하게 단어들의 인과관계를 파악합니다. 이를 통해 기존의 spurious correlation 문제를 해결하고자 합니다.', 'Performance Highlights': '실증 결과로 인해 제안된 방법이 반사실적 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 그리고 부족한 데이터로부터의 일반화 측면에서 현저한 개선을 보였습니다.'}]

- **Unsupervised Fetal Brain Anomaly Detection**: [{"What's New": '뇌의 선천 기형은 가장 흔한 태아 발달 이상 중 하나로, 이를 조기에 탐지하는 것은 매우 중요합니다. 이전의 이상 탐지 방법은 수동 주석에 의존하는 지도 학습 방식이었으나, 우리는 디퓨전 모델(diffusion models)을 사용한 비지도 방식의 이상 탐지 방법을 제안합니다.', 'Technical Details': 'Noise Agnostic Anomaly Detection (iNAAD) 프레임워크는 여러 노이즈 레벨에서 디퓨전 재구성된 이미지를 사용하는 방식으로, 정상 태아 뇌 초음파 이미지만을 훈련에 사용하여 제한된 비정상 데이터 문제를 해결합니다. 다양한 노이즈 유형이 디퓨전 모델에 미치는 영향을 평가합니다.', 'Performance Highlights': '실제 임상 데이터셋에서 우리의 비지도 방법이 뇌 기형 탐지에 잠재력을 가지고 있음을 실험으로 보여주었습니다. 여러 일반적인 태아 뇌 기형에 대해 광범위하게 평가되었습니다.'}]



### SAM2-PATH: A better segment anything model for semantic segmentation in digital pathology (https://arxiv.org/abs/2408.03651)
Comments:
          6 pages , 3 figures

- **What's New**: 이번 연구에서는 다중선택질문(MCQ)을 자동으로 생성하는 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 이 지표는 생성된 MCQ가 학생의 지식을 평가하는 능력을 측정하며, 두 개의 자동 평가 지표인 KDA_disc와 KDA_cont를 도입하여 학생의 문제 해결 행동을 모방한 사전학습 언어 모델을 활용합니다. 이로써 기존의 BLEU, ROUGE, METEOR 등 n-gram 기반의 평가 지표가 가진 한계를 극복하고자 합니다.

- **Technical Details**: KDA는 학생 설문조사 데이터를 기반으로 측정되며, KDA_disc와 KDA_cont는 사전훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모사합니다. Human study를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가지며, 전문가에 의해 레이블된 여러 MCQ 품질 지표에 대한 예측력이 높음을 입증했습니다. 이를 통해 KDA_disc와 KDA_cont는 n-gram 기반의 유사성 메트릭과 결합하여 예측력을 강화할 수 있는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 KDA 및 실제 교실 환경에서의 사용성 측면에서 강한 상관관계를 나타냈습니다. 또한, 전문가 레이블 MCQ 품질 지표에 대한 예측력에서도 뛰어난 성능을 보여줬으며, 기존의 n-gram 기반 평가 지표와 조합할 때 더 높은 예측력을 보였습니다.



### Distillation Learning Guided by Image Reconstruction for One-Shot Medical Image Segmentation (https://arxiv.org/abs/2408.03616)
- **What's New**: 자동 MCQ 생성의 평가 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 새로운 평가 메트릭 Knowledge Dependent Answerability (KDA)을 도입했습니다. KDA는 학생이 해당 지식을 기반으로 문제를 해결할 수 있는지를 측정합니다.

- **Technical Details**: 새로운 KDA 메트릭은 Human survey를 통해 측정된 학생 응답을 바탕으로 하고, 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안하여 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 이 메트릭들은 n-gram 기반 유사도 메트릭과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강한 예측력을 보입니다.

- **Performance Highlights**: Human studies 결과, KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, 기존 방법들보다 다양한 차원에서 성능 향상을 이루었습니다. 특히, 새로 제안된 메트릭들은 교실 설정에서 높은 사용성 및 예측력을 보였습니다.



### InPer: Whole-Process Domain Generalization via Causal Intervention and Perturbation (https://arxiv.org/abs/2408.03608)
Comments:
          Accepted by BMVC2024

- **title**: 자동 MCQ 생성, NLP의 robustness와 도메인 일반화를 위한 새로운 접근법

- **What's New**: [{'title': '자동 MCQ 생성: 지식 종속 가능성(KDA) 메트릭 제안', 'content': '기존 BLEU, ROUGE, METEOR 메트릭의 한계를 극복하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability (지식 종속 가능성, KDA)를 제안합니다. 이 메트릭은 MCQ가 대상 사실에 대해 학생의 지식을 평가하는 능력을 중점으로 평가합니다.'}, {'title': 'NLP Robustness: Counterfactual 집합 기반 학습', 'content': 'NLP 태스크에서 모델의 유연성을 높이기 위해 multiple counterfactual을 생성하고 이를 바탕으로 collective decision을 내리는 방법을 제안합니다. 이로 인해 기존의 spurious pattern에 덜 민감하며 다양한 종속성에 대해 강건한 성능을 보입니다.'}, {'title': '도메인 일반화: InPer 프레임워크', 'content': 'Training 시점에서의 인과 개입(causal intervention)과 테스트 시점에서의 인과 변동(causal perturbation)을 통합한 새로운 도메인 일반화 Framework InPer를 제안합니다. 이로 인해 도메인 간 변이를 효과적으로 처리할 수 있도록 설계되었습니다.'}]

- **Technical Details**: [{'title': 'KDA 기반 MCQ 평가', 'content': 'KDA는 학생의 반응을 바탕으로 한 지식의 평가 가능성(answerability)을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하고, Pre-trained language models를 이용해 학생의 문제 해결 행동을 모방합니다.'}, {'title': 'Counterfactual 집합 기반 학습', 'content': '기존 인간의 개입 없이 다수의 counterfactual을 생성하여, 분포 예측의 집합적 결정방식을 활용합니다. 이를 통해 단어들의 인과적 관계를 robust하게 학습할 수 있습니다.'}, {'title': 'InPer: 인과 개입과 인과 변동을 통한 도메인 일반화', 'content': '훈련 단계에서 인과 변수를 선별하고 causal perturbation을 활용하여 테스트 단계에서 homeostatic score와 prototype classifier를 구축합니다. 이로 인해 모델의 도메인 일반화 성능이 향상됩니다.'}]

- **Performance Highlights**: [{'title': 'KDA 메트릭의 강력한 상관 관계', 'content': 'KDA_disc와 KDA_cont는 실제 강의실 세팅에서 전문가가 라벨링한 사용성과 강한 상관 관계를 보였습니다. n-gram 기반 유사성 메트릭과 결합 시, 다양한 전문가 라벨 MCQ 품질 측정에서 높은 예측력을 보였습니다.'}, {'title': 'NLP Robustness의 향상', 'content': '제안된 counterfactual 집합 기반 학습은 attribution-based synthesis의 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data generalization 측면에서 유의미한 성능 향상을 보였습니다.'}, {'title': 'InPer의 다중 도메인 태스크 성능', 'content': '여러 cross-domain 태스크에서 InPer 프레임워크가 도메인 간 변이에 강건한 성능을 보였으며, 이로 인해 전체적인 모델의 도메인 일반화 성능이 향상되었습니다.'}]



### Hierarchical Quantum Control Gates for Functional MRI Understanding (https://arxiv.org/abs/2408.03596)
- **MCQ Paper**: [{"What's New": '자동 MCQ 생성의 교육적 가치를 평가하는 새로운 메트릭인 지식 종속 가능성(KDA)을 제안했습니다. KDA는 학생의 지식에 기반하여 MCQ의 대답 가능성을 평가합니다.'}, {'Technical Details': 'KDA는 인간 설문 조사 응답을 통해 대답 가능성을 측정하는 방법을 먼저 보인 뒤, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사하는 KDA_disc와 KDA_cont 자동 평가 메트릭을 제안합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, 전문가가 라벨링한 여러 특질 예측에서도 높은 정확도를 보였습니다.'}]

- **Deep Models Robustness Paper**: [{"What's New": '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 새로운 강화 방법을 통해 NLP 태스크에서 모델의 견고성을 향상시켰습니다.'}, {'Technical Details': '기존 증강 방식은 사람이나 모델이 반사실적 데이터를 추가하는 방식이었으나, 제안된 방식은 여러 개의 반사실적 데이터를 생성하고 이를 통해 단어 간 인과관계를 더 robuste하게 판단할 수 있습니다.'}, {'Performance Highlights': '제안된 방법은 반사실적 견고성, 크로스 도메인 일반화 및 희소 데이터의 일반화 성능에서 상당한 개선을 달성했습니다.'}]

- **Quantum Computing for fMRI Paper**: [{"What's New": '기능적 자기공명영상(fMRI) 데이터를 효율적으로 이해하기 위한 새로운 양자 기반 방법인 Hierarchical Quantum Control Gates (HQCG)를 도입했습니다.'}, {'Technical Details': 'HQCG는 국소 양자 제어 게이트(Local Quantum Control Gate - LQCG)와 전역 양자 제어 게이트(Global Quantum Control Gate - GQCG)라는 두 가지 모듈을 포함하며, 각각 fMRI 신호의 지역 및 전역 특징을 추출합니다.'}, {'Performance Highlights': '제안된 방법은 클래식 방법보다 성능이 뛰어나며, 과적합 방지를 포함하여 클래식 방법보다 더 안정적입니다.'}]



### HistoSPACE: Histology-Inspired Spatial Transcriptome Prediction And Characterization Engin (https://arxiv.org/abs/2408.03592)
- **MCQ Generation Paper**: [{"What's New": '자동 MCQ 생성의 평가 메트릭으로 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안.'}, {'Technical Details': 'KDA는 MCQ의 대답 가능성을 측정하며, 이를 통해 학생의 지식을 평가. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방함.'}, {'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성 및 전문가가 평가한 품질 측정치와 강한 상관관계를 가짐.'}]

- **NLP Robustness Paper**: [{"What's New": '얼터너티브 학습(contrastive learning)과 반사실 확충(counterfactual augmentation)을 활용하여 NLP 모델의 robustness 향상.'}, {'Technical Details': '스퍼리어스 패턴(spurious patterns)에 의존하지 않는 반사실 데이터를 합성해 집합적 의사 결정을 통해 인과 관계를 안정적으로 파악.'}, {'Performance Highlights': '다양한 차원에서 significant improvements를 달성: 반사실 robustness, cross-domain 일반화, 희소한 데이터에서의 일반화.'}]

- **Spatial Transcriptomics Paper**: [{"What's New": 'HistoSPACE 모델을 개발하여 조직 이미지로부터 분자적 통찰력을 추출. 기존 방법들의 한계를 극복하려는 시도.'}, {'Technical Details': '유니버설 이미지 오토 인코더(universal image autoencoder)에서 파생된 이미지 인코더를 사용. 컨볼루션 블록과 결합해 최종 모델 생성. ST 데이터로 추가 튜닝.'}, {'Performance Highlights': '기존 알고리즘에 비해 높은 효율성을 보이며, leave-one-out cross-validation에서 0.56의 상관관계 기록. 독립적 데이터셋을 통해 robustness 검증.'}]



### MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilib (https://arxiv.org/abs/2408.03468)
Comments:
          10 pages, 3 figures, ACM Multimedia 2024

- **What's New**: 이번 연구는 기존의 한계점을 극복하기 위해 Multiple Choice Questions (MCQ)의 자동 생성 평가 지표로 'Knowledge Dependent Answerability (KDA)'를 제안합니다. KDA는 학생이 해당 사실에 대한 지식을 바탕으로 MCQ를 얼마나 잘 답할 수 있는지를 측정하여 교육적 가치를 더 잘 평가할 수 있습니다.

- **Technical Details**: KDA는 인간 설문 조사를 통해 학생들의 응답을 기반으로 측정됩니다. 이를 확장하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하며, 이는 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. 이 연구는 KDA_disc와 KDA_cont가 실제 교실 환경에서 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다.

- **Performance Highlights**: 연구에서 제안한 KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 평가 지표와 결합될 때, 다양한 전문가가 라벨링한 MCQ 품질 측정 요소들에 대해 높은 예측력을 보였습니다.



### Post-Mortem Human Iris Segmentation Analysis with Deep Learning (https://arxiv.org/abs/2408.03448)
Comments:
          submitted to ijcb 2024 special session

- **What's New**: 이 논문은 학생 평가를 위한 자동 다지선다형 질문(MCQ) 생성을 좀 더 교육적으로 유의미하게 평가하기 위한 새로운 자동 평가 메트릭 Knowledge Dependent Answerability (KDA)을 제안합니다. 또한 NLP 태스크에서 모델의 robustness를 높이기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 방법을 제안합니다. 마지막으로, 사후(post-mortem) 홍채 인식을 위한 이미지 분할 성능을 개선한 새로운 딥러닝 모델을 소개합니다.

- **Technical Details**: ['MCQ 생성 평가 메트릭으로서 KDA, KDA_disc 및 KDA_cont를 제안하였으며, 사전 학습된 언어 모델을 활용한 방법입니다.', 'NLP 태스크의 robustness 강화를 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 파악하는 방법을 제안합니다.', '사후 홍채 인식을 위한 이미지 분할 방법으로 MobileNetv2를 백본(backbone)으로 사용한 DeepLabV3+ 모델을 활용하며, Boundary와 Dice loss를 결합한 하이브리드 손실 함수를 적용했습니다.']

- **Performance Highlights**: ['KDA_disc 및 KDA_cont는 제거 메트릭과 결합했을 때 전문가가 레이블링한 다양한 MCQ 품질 측정에서 높은 예측력을 보였습니다.', '제안된 대조 학습 및 counterfactual augmentation 기법은 counterfactual robustness, cross-domain generalization 및 적은 데이터로부터의 일반화 측면에서 뛰어난 성능 향상을 달성했습니다.', '우리 팀의 사후 홍채 분할 모델은 Warsaw-BioBase-PostMortem-Iris-v1 데이터셋에서 95.54%의 Mean Intersection over Union 성과를 기록했습니다.']



### Biomedical Image Segmentation: A Systematic Literature Review of Deep Learning Based Object Detection Methods (https://arxiv.org/abs/2408.03393)
- **What's New**: 학습 평가 시간을 크게 줄여주는 자동 Multiple Choice Questions (MCQ) 생성에서 교육적 가치를 평가하는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 최신 NLP deep model의 robustness 문제를 개선하기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법과, 생물의학 이미지 분할을 위한 딥러닝 기반 object detection 방법론에 대한 체계적인 문헌 리뷰 역시 추가로 소개됩니다.

- **Technical Details**: MCQ 생성을 위한 KDA 메트릭은 학생의 지식을 평가하는 능력을 측정하고, student responses를 기반으로 KDA를 측정합니다. 새로운 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA를 근사화합니다. NLP 태스크에서 robustness 문제를 해결하기 위해 여러 개의 counterfactual을 생성하고, collective decision을 통해 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 생물의학 이미지 분할 리뷰는 딥러닝 기반의 object detection 방법론을 중점적으로 다루며, 두 단계 검출 모델, 단일 단계 검출 모델, 포인트 기반 검출 모델로 분류해 각각의 장단점을 분석합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 연구에서 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합 시 전문가가 라벨한 MCQ 품질 측면에서 강한 예측력을 보였습니다. 제안된 collective decision 방법은 attribution-based synthesis의 편향에 덜 민감하여 counterfactual robustness, cross-domain generalization, scarce data generalization에서 유의미한 성능 향상을 달성했습니다. 체계적인 생물의학 이미지 분할 문헌 리뷰는 다양한 약점을 식별하고, 미래 연구 방향과 잠재적 솔루션을 제시함으로써 연구 커뮤니티에 깊은 이해를 제공했습니다.



### A Non-negative VAE:the Generalized Gamma Belief Network (https://arxiv.org/abs/2408.03388)
- **What's New**: 새로운 논문에서는 Multiple Choice Questions (MCQ) 생성을 위한 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하고 있습니다. KDA는 학생들이 대상 사실(Target Fact)에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다. 기존 평가 메트릭들은 BLEU, ROUGE, METEOR와 같이 n-gram 유사성만 측정하며 교육적 가치를 고려하지 않으므로 새로운 접근 방식을 필요로 했습니다.

- **Technical Details**: KDA는 인간 설문 조사(human survey)를 통해 학생 응답 기반으로 측정되며, 이를 기반으로 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 사전 학습된 언어 모델(pre-trained language models)을 활용하여 학생들의 문제 해결 행동을 모방하는 방식으로 KDA를 근사화합니다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, 전문가들이 라벨링한 다양한 MCQ 품질 측정치와 결합할 때 예측력이 높다는 것을 보여주었습니다.



### GMAI-MMBench: A Comprehensive Multimodal Evaluation Benchmark Towards General Medical AI (https://arxiv.org/abs/2408.03361)
- **What's New**: 자동 MCQ 생성의 교육적 가치를 보다 정확하게 평가하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)가 제안되었습니다. 또한 LVLMs를 더욱 견고하게 만들기 위해 대조 학습과 반사실적 데이터 증강을 사용하는 새로운 접근법을 제안했고, 포괄적인 의료 AI 벤치마크인 GMAI-MMBench를 소개합니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 메트릭이 n-gram 기반 유사성에만 초점을 맞추는 문제를 해결하기 위해 KDA를 도입했습니다. KDA는 학생들이 목표 사실에 대한 지식을 바탕으로 MCQ에 답변할 수 있는 능력을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였습니다.', 'LVLM Robustness': '최근의 LVLM이 spurious pattern(허위 패턴)에 의존하는 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 활용한 방법을 소개합니다. 이 방법은 여러 개의 반사실적 데이터를 생성하고 집단 의사 결정을 통해 단어들의 인과 관계를 보다 견고하게 파악합니다.', 'GMAI-MMBench': '의료 분야에서의 LVLMs의 효과를 평가하기 위한 포괄적인 기준(GMAI-MMBench)을 개발했습니다. 이 벤치마크는 285개의 데이터셋을 39개의 의료 이미지 모달리티, 18개의 임상 관련 작업, 18개의 부서, 4개의 인식 세분성으로 분류하여 구성되었습니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 매우 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합할 때 전문가가 레이블링한 다양한 MCQ 품질 측정에 대해 강한 예측력을 나타냈습니다.', 'LVLM Robustness': '이 접근법은 다양한 차원에서 significant improvements(유의한 향상)를 달성했으며, 특히 반사실적 견고성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희소 데이터에서의 일반화(scarce data generalization)에서 개선을 보였습니다.', 'GMAI-MMBench': '50개의 LVLMs를 평가한 결과, 가장 진보된 GPT-4o조차 52%의 정확도만을 달성하였으며, 현재 최첨단 LVLMs가 임상 요구를 충족하기에는 여전히 많은 개선이 필요함을 시사합니다.'}



### IVISIT: An Interactive Visual Simulation Tool for system simulation, visualization, optimization, and parameter managemen (https://arxiv.org/abs/2408.03341)
- **What’s New**: 최신 논문에서는 자동 Multiple Choice Questions (MCQ) 생성을 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, NLP 태스크에서 모델의 robustness를 높이기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 소개했고, Python/Numpy 기반의 일반적인 상호작용 시각화 시뮬레이션 도구인 IVISIT에 대해 설명했습니다.

- **Technical Details**: ['MCQ 생성 논문에서는 KDA라는 새로운 평가 메트릭을 통해 학생들의 지식 평가 능력을 측정하려고 합니다. KDA_disc와 KDA_cont와 같은 자동 평가 메트릭을 통해 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.', 'NLP 태스크에서 robustness 를 높이기 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 파악하는 방법을 제안했습니다. 이 접근법은 기존의 spurious patterns에 덜 민감하며 더 높은 성능과 일반화를 달성할 수 있습니다.', 'IVISIT은 Python/Numpy를 기반으로 시스템 시뮬레이션, 파라미터 최적화 및 시뮬레이션 구성을 위한 기능을 제공하는 도구입니다. Tkinter와 Matplotlib를 사용한 상호작용 GUI 요소를 통해 시뮬레이션 상태를 시각화하고 파라미터를 조작할 수 있으며, SQLite 데이터베이스 기반으로 파라미터와 시뮬레이션 구성을 관리할 수 있습니다.']

- **Performance Highlights**: ['MCQ 생성 논문에서 제안된 KDA_disc와 KDA_cont는 인간 평가와 강한 상관관계를 가지며 교육적 실제 사용 환경에서도 높은 예측력을 보였습니다.', 'NLP 모델의 robustness를 위한 제안은 다양한 차원에서 중요한 성능 향상을 달성했습니다. 특히 counterfactual robustness, cross-domain generalization, 그리고 scarce data 상황에서의 일반화능력이 크게 개선되었습니다.']



### An Empirical Comparison of Video Frame Sampling Methods for Multi-Modal RAG Retrieva (https://arxiv.org/abs/2408.03340)
Comments:
          19 pages, 24 figures (65 images)

- **What's New**: 새로운 논문에서 자동 MCQ(Multiple Choice Questions) 생성을 평가하기 위한 기존의 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 평가하지 못하는 문제를 지적하고, 지식 종속 가능성(KDA, Knowledge Dependent Answerability)이라는 새로운 평가 메트릭을 제안했다. KDA는 학생이 특정 사실에 대한 지식을 바탕으로 MCQ를 답변할 수 있는지를 측정한다.

- **Technical Details**: 논문에서는 학생들의 응답을 기반으로 하는 KDA 측정 방법을 먼저 보여준다. 이후, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안한다. 이를 통해 KDA를 근사치로 계산하는 방법을 제시한다.

- **Performance Highlights**: 휴먼 스터디를 통해 KDA_disc와 KDA_cont가 (1) KDA와 (2) 전문가들이 실제 강의실 세트에서 평가한 사용성과 강한 상관관계를 가지고 있음을 증명했다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가가 평가한 MCQ 품질 지표에 대해 강력한 예측력을 보였다.



### ProCreate, Don't Reproduce! Propulsive Energy Diffusion for Creative Generation (https://arxiv.org/abs/2408.02226)
Comments:
          Accepted to ECCV 2024. Project page: this https URL

- **What's New**: Multiple Choice Questions (MCQ)의 자동 생성은 교사의 학습 평가 시간을 크게 줄일 수 있으나, 기존의 BLEU, ROUGE, METEOR 평가 메트릭은 교육적 가치를 고려하지 않고 있습니다. 이에 우리는 새로운 자동 평가 메트릭, 지식 종속 가능성(KDA)을 제안합니다. 또한 ProCreate라는 새로운 방법을 소개하며, 이 방법은 노이즈 확산 기반 이미지 생성 모델이 더 다양한 샘플을 생성할 수 있도록 돕습니다.

- **Technical Details**: KDA는 학생의 지식을 평가하는 MCQ의 능력을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 이들은 사전 학습된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방합니다. ProCreate는 참조 이미지 세트에서 생성된 이미지 임베딩을 멀리 보내도록 추진력을 적용하여, 학습 데이터 재생산을 방지하고 샘플 다양성과 창의성을 향상시키는 방법입니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서의 사용성과 강한 상관관계를 보였습니다. ProCreate는 FSCG-8 데이터셋에서 기존 방법보다 높은 샘플 다양성과 정확성을 달성했으며, 대규모 평가에서도 효과적으로 학습 데이터 재생산을 방지했습니다.



### RICA2: Rubric-Informed, Calibrated Assessment of Actions (https://arxiv.org/abs/2408.02138)
Comments:
          Accepted at European Conference on Computer Vision (ECCV) 2024

- **What's New**: 기존의 n-gram 기반 평가 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하고자, Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭이 제안되었습니다. 이 메트릭은 학생들이 실제로 문제를 풀 수 있는 능력을 측정하는 데 초점을 맞추며, 이를 통해 MCQ의 교육적 유효성을 더 정확하게 평가할 수 있습니다.

- **Technical Details**: KDA는 학생들의 응답 데이터를 기반으로 측정되며, 이를 자동화하기 위해 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 각 단어의 인과관계를 분석하여, 집합적 의사 결정(collective decision)을 통해 더 robust한 평가를 가능하게 합니다.

- **Performance Highlights**: Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었으며, 전문가가 레이블링한 다양한 MCQ 품질 측정 지표에 대한 예측력도 강함을 나타냈습니다.



New uploads on arXiv(cs.AI)

### Frank's triangular norms in Piaget's logical proportions (https://arxiv.org/abs/2408.03795)
Comments:
          6 pages

- **What's New**: MCQ(객관식 질문)의 자동 생성을 평가하는 기존의 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 무시하고 있습니다. 새로운 메트릭으로 'Knowledge Dependent Answerability (KDA)'를 제안하여, MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: KDA를 측정하기 위해 학생 설문 조사를 기반으로 하는 방법을 먼저 제시하고, 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사하는 KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 라벨링한 실제 강의실 사용성과 강한 상관관계를 가지는 것을 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합하면 여러 전문가가 라벨링한 MCQ 품질 측정에 대해 강한 예측력을 가집니다.



### HiQuE: Hierarchical Question Embedding Network for Multimodal Depression Detection (https://arxiv.org/abs/2408.03648)
Comments:
          11 pages, 6 figures, Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24)

- **What's New**: 최근 논문은 자동화된 Multiple Choice Questions (MCQ) (객관식 문제) 생성과 우울증 탐지 프레임워크에 대한 새로운 접근 방식을 제안하고 있습니다. 자동 MCQ 생성에서는 기존 평가 메트릭이 교육적 가치를 무시하는 문제점을 해결하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 도입하며, 우울증 탐지에서는 임상 인터뷰의 질문 체계 구조를 고려한 HiQuE (Hierarchical Question Embedding network)를 소개합니다.

- **Technical Details**: MCQ 생성에서는 KDA를 통해 학생의 지식을 평가할 수 있는 능력을 측정하며, 이를 자동으로 평가할 수 있는 KDA_disc와 KDA_cont 메트릭을 제안합니다. 기존 메트릭과 결합하여 다양한 전문가들이 라벨링한 MCQ 품질 측정에 강력한 예측력을 보여줍니다. 
우울증 탐지에서는 HiQuE가 임상 인터뷰의 주-종속 질문 간의 계층적 관계를 활용하여, 다양한 모달리티(텍스트, 비주얼, 오디오) 사이의 상호 정보를 학습합니다. HiQuE는 DAIC-WOZ 데이터셋에서 기존 모델들을 능가하는 성능을 보여줍니다.

- **Performance Highlights**: MCQ 생성: 새로운 KDA 기반 메트릭(KDA_disc 및 KDA_cont)이 실제 교실 환경에서의 유용성을 나타내는 강한 상관관계를 보였습니다.
우울증 탐지: HiQuE는 DAIC-WOZ 데이터셋에서 최고 성능을 기록하며, 새로운 질문 시나리오에 대한 일반화 능력을 증명했습니다.



### Large Language Models for Base Station Siting: Intelligent Deployment based on Prompt or Agen (https://arxiv.org/abs/2408.03631)
- **What's New**: 이 연구에서는 MCQ(다지선다형 질문) 자동 생성의 교육적 가치를 평가하지 못하는 기존의 평가 메트릭 문제를 해결하기 위해 새로운 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안합니다. 또한, 최근 NLP에서 높은 정확성을 보이는 deep models의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 결합한 새로운 방법을 제안합니다. 마지막으로, 전통적인 기지국 배치(base station siting) 접근 방법의 복잡성을 줄이기 위해 대형 언어 모델(Large Language Models)의 잠재력을 활용한 혁신적인 최적화 프레임워크를 소개합니다.

- **Technical Details**: ['MCQ 자동 생성에 대한 새로운 평가 지표인 KDA는 대상 사실에 대한 지식을 통해 MCQ의 대답 가능성을 측정합니다. KDA는 student responses와 사전 훈련된 언어 모델을 활용하여 평가 지표를 자동화합니다.', 'deep models의 robust를 증가시키기 위해 우리는 여러 개의 반사실적 사례를 합성하고 집합적 의사 결정을 통해 용어의 인과성을 체계적으로 감독하는 방법을 제안합니다.', '기지국 배치(base station siting)에 대한 새로운 접근법으로, Prompt-optimized LLM (PoL), Human-in-the-Loop LLM (HiLL), LLM-empowered autonomous BSS agent (LaBa), Cooperative multiple LLM-based autonomous BSS agents (CLaBa)의 네 가지 전략을 제안합니다. 이 전략들은 최적의 기지국 위치를 자동으로 결정하여 네트워크 커버리지를 향상시킵니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 학급 환경에서 신뢰할 수 있는 MCQ 품질 평가 지표로 확인되었습니다.', '제안된 robust learning 방법은 다양한 측면에서 중요한 개선을 이루었습니다: 반사실적 로버스트니스, 크로스 도메인 일반화, 데이터 부족 시의 일반화.', '실험 결과, LLM 기반 접근법은 기존의 기지국 배치 방법에 비해 더욱 효율적이고 비용 효과적이며 신뢰할 수 있는 네트워크 배포를 가능하게 했습니다.']



### Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks (https://arxiv.org/abs/2408.03615)
Comments:
          30 pages, 13 figures

- **What's New**: 자동 다중 선택 질문(Multiple Choice Questions, MCQ) 생성의 평가 메트릭으로서 새로운 자동 평가 메트릭인 KDA(지식 종속 가능성)를 제안하였습니다. 또한, 새로운 하이브리드 다중모달 메모리(구조화된 지식 그래프 및 요약된 경험 저장소) 모듈을 소개하여, 오픈월드에서 복잡한 장기 과제를 처리할 수 있는 멀티모달 구성 에이전트 Optimus-1을 개발했습니다.

- **Technical Details**: MCQ 생성의 평가 메트릭으로 기존 BLEU, ROUGE, METEOR 대신 KDA_disc와 KDA_cont를 제안하여 학생의 대상 사실에 대한 지식 평가 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. Optimus-1은 하이브리드 다중모달 메모리 모듈을 기반으로 Knowledge-Guided Planner(지식 기반 계획기), Experience-Driven Reflector(경험 기반 반사기), Action Controller(행동 컨트롤러)로 구성된 에이전트입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가를 통해 실제 강의실 설정에서의 사용성과 강한 상관관계를 보였습니다. Optimus-1은 장기 과제 벤치마크에서 30%의 성능 향상을 보여주었고, 하이브리드 다중모달 메모리를 통해 GPT-4V보다 2~6배의 성능 향상을 달성했습니다.



### Automated Theorem Provers Help Improve Large Language Model Reasoning (https://arxiv.org/abs/2408.03492)
- **What's New**: 이번 주 AI 뉴스레터에서는 자동 다지선다형 질문 (MCQ) 생성, 딥러닝 모델의 강건성 향상, 그리고 대형 언어 모델 (LLM)의 논리적 추론능력 향상을 위한 새로운 접근법들이 논의됩니다.

- **Technical Details**: [{'Topic': '자동 MCQ 생성', 'Summary': '기존 MCQ 생성 평가지표인 BLEU, ROUGE, METEOR는 교육적 가치를 판단하지 못하는 한계가 있습니다. 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되었으며, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, {'Topic': '딥러닝 모델의 강건성', 'Summary': '최근 딥 모델들이 NLP 작업에서 사람보다 높은 정확성을 보였으나, 속임수 패턴에 의존하여 강건성이 제한된다는 문제가 보고되었습니다. 이에 대처하기 위해 대조 학습 (contrastive learning)과 반사실 증강 (counterfactual augmentation)을 제안합니다. 기존 방법과 달리, 이 접근법은 여러 개의 반사실을 생성하여 집합적 의사 결정을 통해 단어들의 인과 관계를 더 강건하게 감독합니다.'}, {'Topic': '대형 언어 모델 (LLM)의 논리적 추론 향상', 'Summary': '논리 프로그래밍 시스템과 자동 1차 논리 정리 증명기 (ATPs)를 사용하여 LLM의 논리적 추론 능력을 향상시킵니다. 특히 LLM의 번역 정확성을 분석하고, 구문 및 의미 오류를 자동으로 교정하는 방법을 제안합니다. 이 접근법은 PRONTOQA 벤치마크에서 시험되었습니다.'}]

- **Performance Highlights**: [{'Topic': '자동 MCQ 생성', 'Summary': 'KDA_disc와 KDA_cont는 실제 교실 환경에서 탁월한 사용성을 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에서 강한 예측력을 가집니다.'}, {'Topic': '딥러닝 모델의 강건성', 'Summary': '집합적 의사 결정을 통한 접근법은 반사실 강건성, 크로스 도메인 일반화 및 희소 데이터 일반화 측면에서 상당한 개선을 이루었습니다.'}, {'Topic': '대형 언어 모델 (LLM)의 논리적 추론 향상', 'Summary': '논리적 추론 능력 향상을 통해 의미 오류가 크게 감소되었으며, LLM 기반 논리적 추론의 정확성이 더 향상되었습니다.'}]



### miniCTX: Neural Theorem Proving with (Long-)Contexts (https://arxiv.org/abs/2408.03350)
- **What's New**: 이번 달 AI 뉴스레터에서는 세 가지 주요 논문에 대해 알아보겠습니다. 첫 번째 논문은 자동 MCQ 생성 평가의 새로운 지표인 '지식 종속 가능성(KDA)'을 제안하며, 두 번째 논문은 NLP 태스크의 Robustness (강건성)를 증진시키기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 도입하는 방법을 소개합니다. 마지막으로, 세 번째 논문은 새로운 정의와 보조 정리들에 의존하는 수학 정리 증명 능력을 테스트하는 'miniCTX'라는 벤치마크를 소개합니다. 이 논문들은 각기 다른 분야에서 AI의 가능성을 한 단계 더 끌어올릴 수 있는 방법들을 제시하고 있습니다.

- **Technical Details**: [{'title': 'MCQ 평가 지표', 'content': "기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 평가 메트릭은 교육적 가치를 고려하지 못합니다. 이를 해결하기 위해 우리는 '지식 종속 대답 가능성(KDA)'이라는 새로운 메트릭을 도입했습니다. KDA는 타겟 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 보다 자동화된 평가를 가능하게 합니다."}, {'title': 'NLP 태스크의 강건성 증대', 'content': "최근의 딥러닝 모델들은 NLP 태스크에서 초인적인 정확도를 보여주지만, 스퍼리어스 패턴(spurious patterns)에 의존하는 문제로 인해 강건성이 제한됩니다. 이를 해결하기 위해, 대조 학습과 counterfactual augmentation을 활용하는 방법을 제안합니다. 기존 방법이 사람이나 머신에 의존하는 반면, 우리는 '여러 개의' counterfactual을 생성하고 이들의 분포를 기반으로 집합적 의사 결정을 내리는 방식을 채택했습니다. 이 방법은 인과 관계를 보다 강건하게 감독할 수 있습니다."}, {'title': '컨텍스트 의존 정리 증명 벤치마크', 'content': "'miniCTX'는 훈련 중 관찰되지 않은 새로운 정의나 보조 정리에 의존하는 수학적 정리를 증명하는 모델의 능력을 테스트합니다. miniCTX는 실제 Lean 프로젝트와 교과서에서 가져온 정리들을 포함하고 있으며, 각 정리는 수천 개의 토큰을 포함하는 컨텍스트와 연결되어 있습니다. 우리는 '파일 튜닝(file-tuning)'이라는 기본적인 레시피를 도입하여 모델을 학습시키고, 이는 전통적인 신경망 정리 증명 접근법보다 뛰어난 성능을 보였습니다."}]

- **Performance Highlights**: [{'title': 'MCQ 평가 지표 성능', 'content': '실제 강의실 상황에서 KDA_disc와 KDA_cont는 높은 상관관계를 보였으며, 전문가가 레이블링한 다양한 MCQ 품질 측정 기준에 대해 예측력이 강한 것으로 나타났습니다.'}, {'title': 'NLP 태스크의 강건성', 'content': '대조 학습과 counterfactual augmentation 방식을 통한 새로운 방법이 tasks 모델의 바이어스를 줄이고, 다양한 측면에서 강건성, 크로스 도메인 일반화 및 데이터가 부족한 상황에서의 일반화 성능을 크게 개선했습니다.'}, {'title': 'miniCTX 성능', 'content': '파일 튜닝을 적용한 모델은 기존의 신경망 정리 증명 접근법보다 높은 성능을 보였으며, 표준 miniF2F 벤치마크에서도 33.61%란 새로운 최고 통과율을 기록했습니다.'}]



### SLIM-RAFT: A Novel Fine-Tuning Approach to Improve Cross-Linguistic Performance for Mercosur Common Nomenclatur (https://arxiv.org/abs/2408.03936)
Comments:
          13 pages, 1 figure, to be publish in International Conference on Web Information Systems and Technologies - WEBIST 2024 proceedings

- **What's New**: 이번 뉴스레터에서는 새로운 자동 평가 메트릭 Knowledge Dependent Answerability (KDA)와 NLP 모델의 강인성을 향상시키기 위해 대조 학습 및 대응 강화 방법을 적용한 모델, 그리고 포르투갈어 LLM(대형 언어 모델)을 활용한 특정 도메인 과업적 미세조정 기술인 SLIM-RAFT를 소개합니다.

- **Technical Details**: {'MCQ Generation': '자동 MCQ 생성을 위해 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭이 n-gram 유사성에만 집중하여 교육적 가치를 평가하지 않는 문제를 해결하기 위해, 학생의 대답 가능성을 측정하는 Knowledge Dependent Answerability (KDA)라는 새로운 메트릭을 제안했습니다. 기존 메트릭과 KDA_disc, KDA_cont를 결합해 명확하게 측정할 수 있음을 입증하였습니다.', 'Robust NLP Models': '대조 학습(contrastive learning)과 대응 강화(counterfactual augmentation)를 활용해 NLP 모델의 강인성을 향상시켰습니다. 여러 개의 대응 사실(counterfactual)을 생성해 합리적인 집합적 의사 결정을 통해 모델이 인과관계를 더 잘 이해할 수 있도록 설계하였습니다. 이는 기존의 인간 생성 대응 사실이나 자동 매칭 방식보다 더 효과적입니다.', 'SLIM-RAFT': '법율과 조세 분야에서 사용되는 MERCOSUR NCM 코드 시스템의 처리를 위해 작은 포르투갈어 LLM TeenyTineLLaMA를 사용한 미세조정 기술인 SLIM-RAFT를 제안합니다. 이 기술은 효율적이며 비용 효율적인 대안으로, 특히 소규모 데이터에서 높은 성능을 보입니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성을 평가한 결과, 학생들의 평가와 강한 상관관계를 보였습니다.', 'Robust NLP Models': '이 접근 방식은 대조 학습 및 대응 강화 방법을 더해, 실행된 모델이 기존 모델보다 다양한 측면에서 불균형적 상관성 문제에 덜 민감하며, 의미 없는 패턴에 덜 의존하도록 하여 성능을 개선했습니다.', 'SLIM-RAFT': 'SLIM-RAFT 모델은 MERCOSUR NCM 코드 시스템 처리에서 ChatGPT-4 보다 높은 성능 (8.67/10 vs 4.5/10)을 기록했습니다.'}



### CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases (https://arxiv.org/abs/2408.03910)
Comments:
          work in progress

- **What's New**: [{'title': '지식 종속 가능성 평가 지표 (KDA) 제안', 'description': '기존의 MCQ 생성 평가 메트릭은 교육적 가치를 반영하지 못하는 문제를 해결하기 위해, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 도입했습니다. 이 메트릭은 MCQ의 대답 가능성을 학생의 지식 수준에 따라 평가합니다.'}, {'title': 'Contrastive Learning과 Counterfactual Augmentation의 결합', 'description': '최근 NLP 모델들이 spurious pattern에 의존해 robustness가 제한적인 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 결합한 새로운 방법을 제안했습니다. 이는 복수의 counterfactual을 생성하여 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악합니다.'}, {'title': '그래프 데이터베이스 인터페이스를 활용한 LLM-코드베이스 통합', 'description': '대규모 코드 리포지토리와 상호작용하는 새로운 시스템 \x0cramework를 도입했습니다. 이는 LLM과 코드 리포지토리를 그래프 데이터베이스를 통해 연결함으로써, 코드 구조를 이해하고 효율적인 코드 탐색을 가능하게 합니다.'}]

- **Technical Details**: [{'title': 'KDA 평가 메트릭', 'description': 'KDA는 학생 응답을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입해 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.'}, {'title': 'Contrastive Learning과 Counterfactual Augmentation', 'description': '기존의 사람이나 기계가 데이터셋에서 counterfactual을 생성하지 않고, 여러 개의 counterfactual을 자동으로 생성하여 집합적 예측 결과를 통해 단어의 인과관계를 보다 robust하게 감독합니다.'}, {'title': '\x0cramework 시스템', 'description': '\x0cramework는 그래프 데이터베이스의 구조적 속성과 그래프 쿼리 언어의 유연성을 활용하여, 코드 리포지토리로부터 정적 분석을 통해 코드 그래프를 추출하고, LLM이 쿼리를 구성하고 실행하여 정확하고 코드 구조를 인지한 컨텍스트 검색과 코드 탐색을 가능하게 합니다.'}]

- **Performance Highlights**: [{'title': 'KDA_disc와 KDA_cont', 'description': 'KDA_disc와 KDA_cont는 각각 KDA와 실제 강의실에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합 시 다양한 전문가가 라벨링한 MCQ 품질 측정에서 강한 예측력을 보여줍니다.'}, {'title': '대화식 의사결정(Multi-hop Reasoning)', 'description': '제안된 방법은 기존 유사성 기반 검색과 수작업 도구 및 API 방식의 한계를 극복하여, 더 높은 회수율과 정확도를 보이며, 복잡한 코드 구조와 긴 시퀀스 reasoning에서 상당한 성능 향상을 이룹니다.'}, {'title': '세 가지 벤치마크에서의 성능', 'description': 'CrossCodeEval, SWE-bench, EvoCodeBench 세 가지 벤치마크에서 \x0cramework는 통합된 그래프 데이터베이스 스키마와 간단한 워크플로 디자인을 통해 경쟁력 있는 성능을 보여줍니다. 특히, 더 발전된 LLM을 탑재했을 때 성능이 더욱 향상됩니다.'}]



### LaFA: Latent Feature Attacks on Non-negative Matrix Factorization (https://arxiv.org/abs/2408.03909)
Comments:
          LA-UR-24-26951

- **What's New**: MCQ 생성과 평가에 있어 기존 메트릭이 교육적 가치 (educational value)를 제대로 반영하지 못하고 있다는 문제를 해결하기 위해 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 도입했습니다. 또한 기존의 방법들이 spurious correlation에 의존하여 문제가 되는 것을 극복하기 위해 여러 개의 counterfactual을 사용하여 집합적 의사 결정 (collective decision)을 통한 robust한 인과관계 파악 방법을 제안하였습니다.

- **Technical Details**: KDA는 학생의 대상 사실 (target fact) 지식에 기반해 특정 MCQ의 대답 가능성 (answerability)을 측정합니다. 또한, pre-trained language models를 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다. 반면, NLP 태스크에서 기존의 robustness 문제를 해결하기 위해 반사실적 학습 (counterfactual learning)과 집합적인 의사 결정을 통한 방법론을 도입했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 갖고 있으며, 전문가들이 라벨링한 MCQ 품질 측정치에 대해 높은 예측력을 보여줍니다. 또한, 다양한 차원에서 기존의 deep 모델보다 개선된 성능을 보였습니다. 이로써 MCQ의 교육적 유용성 및 robustness가 입증되었습니다.



### Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models (https://arxiv.org/abs/2408.03907)
Comments:
          6 pages paper content, 17 pages of appendix

- **What's New**: 자동 MCQ 생성에서 교육적 가치를 평가하기 위한 새 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 개발했습니다.

- **Technical Details**: KDA는 학생들의 실제 응답을 바탕으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. KDA_disc와 KDA_cont는 각각 이산형과 연속형 평가 메트릭으로, 인간 평가와 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 마킹 MCQ 품질 측정에서 높은 예측력을 보였습니다.



### Lightweight Video Denoising Using a Classic Bayesian Backbon (https://arxiv.org/abs/2408.03904)
Comments:
          Paper accepted to ICME 2024

- **What's New**: 각각 자동 MCQ 생성의 새로운 지식 종속 가능성(KDA) 메트릭, 많은 매개변수를 필요로 하지 않으면서도 높은 성능을 나타내는 하이브리드 Wiener 필터 도입, 그리고 대체적 학습을 통한 강화 학습 방법을 활용한 NLP 모델의 향상된 robust 성능 기술 등이 제안되었습니다.

- **Technical Details**: 1. MCQ 생성 논문에서는 기존의 BLEU, ROUGE, METEOR 메트릭이 가진 문제를 지적하고, KDA라는 새로운 평가 메트릭을 제안했습니다. KDA는 학생의 지식을 바탕으로 문제의 답변 가능성을 측정합니다.
2. NLP 논문에서는 대체적 학습(contrastive learning)과 counterfactual augmentation을 통해 기존의 spurious 패턴 문제를 극복하고 인과 관계를 더욱 명확히 파악하는 접근 방식을 채택했습니다.
3. 하이브리드 Wiener 필터 논문에서는 기존 Bayesian denoising 기법을 최적화하는 대신에 작은 보조 네트워크를 사용하여 성능을 높이면서도 빠른 속도를 유지하는 방법을 제안했습니다.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 전문가들이 라벨링한 MCQ 품질 측정에서 높은 예측력을 나타냈습니다.
2. 대체적 학습 및 counterfactual augmentation을 활용한 기법은 다양한 차원에서의 성능 향상을 보여주며, 특히 counterfactual robustness와 cross-domain generalization이 두드러졌습니다.
3. 하이브리드 Wiener 필터는 PSNR 및 SSIM 측정에서 DVDNet, FastDVDNet, VNLB 등의 기존 메서드를 능가하며, VRT transformer 대비 10배 이상의 속도 성능을 보였습니다.



### Simplifying Scholarly Abstracts for Accessible Digital Libraries (https://arxiv.org/abs/2408.03899)
Comments:
          Initial submission to JCDL2024

- **What's New**: 이번 아카이브 논문의 저자들은 학생 평가를 위한 자동적인 다지선다형 질문(MCQ) 생성의 효율성을 높이기 위한 새로운 평가 메트릭인 '지식 종속 답변 가능성(KDA, Knowledge Dependent Answerability)'을 제안하였습니다. KDA는 기존의 n-gram 기반 평가 메트릭들이 간과하는 교육적인 가치를 고려하여 MCQ의 실제 교육 효과를 측정합니다.

- **Technical Details**: 논문에서는 KDA를 측정하기 위해 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이는 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방함으로써 KDA를 근사화합니다. 이러한 접근법은 사전 학습된 언어 모델의 능력을 활용하여 MCQ의 답변 가능성을 분석합니다.

- **Performance Highlights**: Human evaluation 결과에 따르면 KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성에 대해 전문가들이 라벨링한 데이터와 강한 상관관계를 보였습니다. 또한, n-gram 기반의 유사성 메트릭과 결합했을 때, 전문가들이 라벨링한 다양한 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다.



### MORTAR: A Model-based Runtime Action Repair Framework for AI-enabled Cyber-Physical Systems (https://arxiv.org/abs/2408.03892)
- **What's New**: 이 논문에서는 기존 MCQ 생성 평가 메트릭의 단점을 보완하기 위해 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안합니다. 또한, AI를 활용한 Cyber-Physical Systems (CPS)의 안전성 향상을 위해 MORTAR라는 런타임 행동 수리 프레임워크를 소개합니다.

- **Technical Details**: ['기존의 BLEU, ROUGE, METEOR 평가 메트릭은 MCQ의 교육적 가치를 평가하지 못하는 문제를 다룹니다.', 'KDA는 학생의 대상 사실(target fact)에 대한 지식을 평가할 수 있도록 MCQ의 대답 가능성을 측정하도록 고안되었습니다.', 'Counterfactual augmentation과 contrastive learning을 사용해 robust한 NLP 모델을 만드는 방법을 제안합니다.', 'MORTAR 프레임워크는 AI 컨트롤러의 행동을 예측하고, 불안전한 행동을 수정하기 위한 최적화 과정을 거칩니다.', 'MORTAR는 black-box policy 환경에도 적용 가능하여 다양한 CPS에서의 실시간 수정을 지원합니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 강의실 세팅에서 높은 상관관계를 보이며, 전문가가 라벨링한 MCQ 품질 척도와의 강한 예측력을 가집니다.', '수많은 실험을 통해, 집합적 의사결정 방법이 counterfactual robustness, cross-domain generalization, 데이터 부족 환경에서의 일반화 성능 개선 효과를 보여줍니다.', 'MORTAR는 다수의 CPS 태스크와 AI 컨트롤러에 대해서 높은 작업 완료율을 유지하며, 최소한의 계산 자원을 소모하여 실시간 운영을 가능하게 합니다.']



### Knowledge Probing for Graph Representation Learning (https://arxiv.org/abs/2408.03877)
- **What's New**: 자동 MCQ(객관식 질문) 생성에 대한 새로운 평가 메트릭을 제안하고, 그 효과성과 신뢰성을 개선했습니다. 또한, 대조 학습과 반사실적 증강을 활용해 NLP 모델의 강건성을 높이는 방법을 연구했습니다. 마지막으로, 다양한 그래프 학습 방법이 학습된 임베딩에서 어떤 그래프 속성을 인코딩하는지 체계적으로 조사하는 프레임워크를 제안했습니다.

- **Technical Details**: MCQ 생성에서는 Knowledge Dependent Answerability(KDA)를 통해 교육적 가치를 평가하고, 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하도록 했습니다. NLP 모델의 강건성을 위해 반사실적 증강(couterfactual augmentation)을 사용한 집합적 의사결정(collective decision) 방법을 제안했습니다. 그래프 학습에서는 그래프 고유 속성에 따라 노드별, 경로별, 구조별 세 가지 수준의 probe를 설계하여 그래프 표현 학습을 평가했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서 전문가들이 라벨링한 사용성과 강한 상관관계를 보였습니다. 반사실적 증강을 통한 대조 학습은 여러 차원에서 모델의 강건성을 크게 향상시켰습니다. GraphProbe는 9개의 대표적인 그래프 학습 방법을 통해 그래프 표현 학습 능력을 효과적으로 측정함을 보여주었습니다.



### Inter-Series Transformer: Attending to Products in Time Series Forecasting (https://arxiv.org/abs/2408.03872)
- **What's New**: 교사의 학습 평가 시간을 줄이기 위한 새로운 MCQ 생성 자동화 평가 메트릭 KDA(지식 종속 가능성)을 제안했습니다. 또, Transformer 기반 모델을 공급망 수요 예측에 적용하여 상호작용을 포착하고 희소성을 해결하는 새로운 접근법을 개발했습니다.

- **Technical Details**: ['MCQ 생성 평가를 위해 KDA_disc와 KDA_cont 자동 평가 메트릭을 제안했습니다. 이 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', '최근 NLP 태스크에서 deep model의 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용했습니다. 기존 방법의 한계를 극복하고자 여러 개의 counterfactual을 생성하여 집합적 의사 결정(collective decision)을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안했습니다.', '공급망 수요 예측에 Transformer 기반 모델을 적용하여 희소성과 cross-series 효과 등을 해결하는 멀티태스크 per-time series 네트워크를 개발했습니다. 이 네트워크는 각 시리즈 간의 상호작용을 포착하기 위해 attention mechanism을 사용합니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 교실 환경에서 사용성과 강한 상관관계를 가지고 있음을 human study를 통해 확인하였습니다.', '여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 기존 방법보다 task model bias에 덜 민감하여 counterfactual robustness와 cross-domain generalization, scarce data로부터의 일반화에서 유의미한 향상을 이루었습니다.', 'Transformer 기반 모델이 다양한 공개 수요 예측 데이터셋에서 기존의 기준 방법들과 최신 방법들에 비해 경쟁력 있는 성능을 제공함을 입증했습니다.']



### BeeManc at the PLABA Track of TAC-2023: Investigating LLMs and Controllable Attributes for Improving Biomedical Text Readability (https://arxiv.org/abs/2408.03871)
Comments:
          system report for PLABA-2023. arXiv admin note: substantial text overlap with arXiv:2309.13202

- **What's New**: 이번 연구에서는 자동 MCQ 생성 평가에서 기존 메트릭들이 교육적 가치를 평가하지 못하는 문제를 해결하기 위해, MCQ의 정답 가능성을 측정하는 새로운 평가 메트릭 KDA를 제안했습니다. KDA를 고안하여 학생의 지식을 평가하는 힘을 강화하고자 합니다. 또한, 최근 딥러닝 모델들의 높은 정확도에도 불구하고, 이를 위한 robust 대책으로 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 접근법을 제안했습니다. 마지막으로, PLABA2023 바이오메디컬 추상 요약 과제에서 다양한 LLM 모델을 이용하여 성능을 평가하고 최적화한 결과를 설명하였습니다.

- **Technical Details**: MCQ 평가를 위한 KDA(Knowledge Dependent Answerability)를 제안했으며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 KDA를 근사화했습니다. 또한, 대조 학습과 counterfactual augmentation을 통해 모델의 robust를 강화하는 방법을 제시했습니다. PLABA2023 과제에서는 T5, SciFive, BioGPT, BART와 같은 LLM 모델들을 미세 조정(fine-tuning)하여 최적화하고, Quantitative metric으로 SARI, BERTscore, BLEU, ROUGE를 사용했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다. 또한, 대조 학습 기반 접근법은 Attribution-based synthesis로 인한 task model bias에 덜 민감하여, counterfactual robustness, cross-domain generalization, 그리고 scarce data로부터의 일반화(Generalization)에서 현저한 향상을 달성했습니다. PLABA2023 과제에서는 BeeManc 팀이 자동 평가 SARI 점수에서 2위를, 인간 평가에서 BART-w-CTs가 문장 간단성 면에서 2위를 기록했습니다.



### Mapping the Provenance Ontology to Basic Formal Ontology (https://arxiv.org/abs/2408.03866)
Comments:
          28 pages, 10 figures

- **What's New**: 자동 MCQ 생성의 평가 메트릭 향상과, NLP에서의 구조적 문제 해결을 위한 새로운 방법론, 그리고 PROV-O와 BFO 간의 호환성 강화에 관한 최신 연구들이 발표되었습니다.

- **Technical Details**: [{'Automatic MCQ Generation': {'Problem': '기존 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 평가하지 않음.', 'Solution': 'Knowledge Dependent Answerability (KDA) 메트릭 제안, 학생의 지식에 기반한 대답 가능성을 평가.', 'Implementation': 'KDA_disc와 KDA_cont라는 자동 평가 메트릭을 사전 훈련된 언어 모델을 통해 개발하고, 인간 평가를 통해 유효성을 검증.'}}, {'Robustness in NLP': {'Problem': 'Deep model이 spurious pattern에 의존하여 robustness가 낮음.', 'Solution': 'Contrastive learning과 counterfactual augmentation을 통해 robustness 향상.', 'Implementation': '여러 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악.'}}, {'Ontology Alignment': {'Problem': 'PROV-O와 BFO 간의 호환성 부족.', 'Solution': '구조적이며 의미론적인 고려사항을 우선하는 특정 매핑 기준과 방법론에 따라 정렬 제안.', 'Implementation': '논리적 일관성을 PROV-O 예제와 SPARQL을 통해 검증하고, 다양한 시맨틱 웹 기술을 사용하여 FAIR 원칙 준수.'}}]

- **Performance Highlights**: [{'Automatic MCQ Generation': {'Correlation': 'KDA_disc와 KDA_cont가 인간 평가와 높은 상관관계를 보임.', 'Usability': '실제 교실 환경에서 전문가들에 의해 사용성이 높게 평가됨.'}}, {'Robustness in NLP': {'Dimensions': '다양한 차원에서 성능 향상: Counterfactual robustness, Cross-domain generalization, Scarce data generalization', 'Sensitivity': '집합적 의사 결정으로 모델 바이어스 저감.'}}, {'Ontology Alignment': {'Consistency': 'PROV-O 및 BFO 예제와의 논리적 일관성 검증.', 'Compliance': 'FAIR 원칙 준수를 지원하기 위한 다양한 시맨틱 웹 기술 활용.'}}]



### MaxMind: A Memory Loop Network to Enhance Software Productivity based on Large Language Models (https://arxiv.org/abs/2408.03841)
- **What's New**: 자동 MCQ(객관식 질문) 생성의 평가에 있어 기존의 BLEU, ROUGE, METEOR 메트릭스는 교육적 가치를 고려하지 않는다. 이에 우리는 새로운 평가 메트릭으로 KDA(Knowledge Dependent Answerability)를 제안하며, 학생의 지식을 평가하는 능력을 측정한다. 또한 deep model들의 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 새로운 방법을 소개한다. 마지막으로, 대형 언어 모델(LLM)의 메모리 루프 네트워크와 MaxMind 시스템을 통해 소프트웨어 생산성을 향상시키는 방법을 제안한다.

- **Technical Details**: KDA는 인간 설문조사로부터 학생 반응을 통해 측정되며, 이를 바탕으로 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안한다. 또한, 우리는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어의 인과관계를 파악하는 방법을 제안한다. MaxMind 시스템은 외부 메모리 모듈을 사용하여 작업 경험을 축적하고, RAG 메커니즘을 보완하여 메모리 기반의 지식 차별화를 활용한다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 확인했다. Collectively, 우리의 접근 방식은 attribution-based synthesis 모델의 편향에 덜 민감하며 counterfactual robustness, cross-domain generalization, 그리고 data scarcity에 대한 일반화 능력에서 상당한 향상을 보였다. MaxMind 시스템은 메모리 재활용을 통해 약 3 ~ 6%의 작업 성공률 향상과 최대 25%의 작업 실행 효율 향상을 보여주었다.



### WalledEval: A Comprehensive Safety Evaluation Toolkit for Large Language Models (https://arxiv.org/abs/2408.03837)
Comments:
          Under review

- **What's New**: [{'title': 'Knowledge Dependent Answerability (KDA)', 'content': "최근 교육 평가용 MCQ(Multiple Choice Question) 생성 분야에서는 교육적 가치를 고려하지 않는 기존 메트릭(BLEU, ROUGE, METEOR)의 한계를 극복하기 위해 'Knowledge Dependent Answerability (KDA)'라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 특정 사실에 대한 학생의 지식을 직접 측정합니다."}, {'title': 'Robust AI with Contrastive Learning', 'content': '최근 NLP 태스크에서 deep model들이 높은 정확도를 보이지만, 겉보기에 그럴듯한 패턴에 의존하는 경향으로 인해 내구성이 제한됩니다. 본 논문에서는 contrastive learning과 counterfactual augmentation 기법을 적용해 모델의 내구성을 강화하는 방법을 제안합니다.'}, {'title': 'WalledEval AI Safety Toolkit', 'content': '새로운 AI 안전성 평가 툴킷인 WalledEval을 소개합니다. WalledEval은 다양한 언어 모델 (LLM)을 평가할 수 있으며, 35가지 이상의 안전성 벤치마크와 맞춤형 변형(mutators)을 포함하여 다차원적 안전성 평가를 제공합니다.'}]

- **Technical Details**: [{'title': 'KDA_disc와 KDA_cont', 'content': 'KDA를 기반으로 학생의 답변을 시뮬레이션하는 방법론인 KDA_disc와 KDA_cont를 제안합니다. 이 두 메트릭은 실제 학생들이 문제를 푸는 행동을 모방하며, 인간 연구를 통해 이들이 KDA 및 실제 교실 환경에서의 사용성과 강한 상관관계를 보임을 확인했습니다.'}, {'title': 'Collective Decisions for Robustness', 'content': '본 논문은 도메인 간 일반화와 희소 데이터에서의 일반화를 위한 집합적 의사 결정(collective decisions)을 도입하여 단어 간 인과관계를 보다 안정적으로 확보하는 방법을 소개합니다. 기존의 허위 상관관계(spurious correlations)에 영향을 받는 방법과 달리, 다수의 반사실(counterfactual)을 생성하고 이를 통해 robust한 평가를 수행합니다.'}, {'title': "WalledEval's Framework", 'content': 'WalledEval은 다양한 LLM과 협력할 수 있는 Python 기반 프레임워크입니다. LLM 모델과 Judge를 로드하고, 다양한 벤치마크 데이터를 활용해 LLM의 안전성을 평가할 수 있도록 설계되었습니다. 또한 맞춤형 변형(mutators)을 사용해 텍스트 스타일 변형에 대한 안전성 테스트를 실행합니다.'}]

- **Performance Highlights**: [{'title': 'Educational Value Assessment', 'content': 'KDA_disc와 KDA_cont를 이용한 MCQ 평가가 전문가 라벨링과 강한 상관관계를 보여 MCQ의 교육적 가치를 보다 정확히 평가할 수 있음이 입증되었습니다.'}, {'title': 'Robustness Improvements', 'content': '본 논문에서 제안한 방법은 counterfactual robustness, 도메인 간 일반화(cross-domain generalization), 및 희소 데이터에서의 일반화 측면에서 현저한 성능 향상을 입증했습니다.'}, {'title': "WalledGuard's Efficiency", 'content': 'WalledEval의 새로운 콘텐츠 모더레이션 도구인 WalledGuard는 기존의 도구에 비해 약 16배 작으면서도, 최고 성능의 LlamaGuard-2와 비교해 성능 저하 없이 우수한 성능을 보여주었습니다.'}]



### Target Prompting for Information Extraction with Vision Language Mod (https://arxiv.org/abs/2408.03834)
Comments:
          7 pages, 5 figures

- **What's New**: 최근의 연구들은 자동으로 다지선다형 문제(MCQ)를 생성하는 시스템 개발에 집중하고 있습니다. 이번 연구에서는 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭이 교육적 가치를 제대로 반영하지 않는다는 문제를 지적하며, 새로운 자동 평가 메트릭인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'을 제안합니다. 또한, 최근에는 NLP 태스크에서 대규모 모델들의 정확도가 사람을 능가하지만, 견고성(robustness) 부족 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용하는 방법이 제안되었습니다. 그리고 대형 비전 및 언어 모델(Vision and Language model, VLM)이 문서 이해 및 질문 응답 시스템 구축에 혁신적인 변화를 가져오는 가운데, 이 모델의 단점을 보완하기 위한 '타겟 프롬프팅(Target Prompting)' 기법이 소개되었습니다.

- **Technical Details**: {'MCQ Generation': '이번 연구는 대상 사실에 대한 학생의 지식을 평가하는 새로운 메트릭인 KDA를 도입했습니다. KDA는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont로 구현되며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Robustness in NLP': '대조 학습과 반사실적 증강 기법을 사용하여 모델의 견고성을 강화하며, 기존 방법들과 달리 여러 개의 반사실적 데이터를 합성하여 예측 분포를 결정하는 방식을 사용합니다.', 'Document Understanding': "대형 비전 및 언어 모델에서 '타겟 프롬프팅' 기법을 사용하여 특정 문서 이미지의 부분을 명시적으로 타겟팅해 관련된 응답을 생성하는 방법이 제안되었습니다. 이 모델은 이미지 인코더와 변환기 디코더로 구성된 복합 모달 모델을 사용합니다."}

- **Performance Highlights**: {'MCQ Evaluation': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있으며, 전문가들이 라벨링한 다양한 MCQ 품질 평가 지표에 대해 높은 예측력을 보였습니다.', 'NLP Robustness': '대조 학습과 반사실적 증강 기법을 통해 1) 반사실적 견고성, 2) 도메인 간 일반화, 3) 희소 데이터를 통한 일반화 성능이 크게 향상되었습니다.', 'Document Understanding': '타겟 프롬프팅 기법을 사용한 실험 결과, 특정 문서 이미지로부터 구체적이고 정확한 정보를 효율적으로 추출할 수 있음을 확인했습니다.'}



### Automated Code Fix Suggestions for Accessibility Issues in Mobile Apps (https://arxiv.org/abs/2408.03827)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문에서는 자동적으로 MCQ(객관식 문제)를 생성하는 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안했습니다. 또 다른 논문에서는 NLP 태스크에서 contrastive learning과 counterfactual augmentation을 이용하여 모델의 robustness를 향상시키는 방법을 논의했습니다. 세 번째 논문에서는 모바일 앱의 접근성을 개선하기 위한 FixAlly라는 도구를 소개합니다.

- **Technical Details**: ['MCQ 논문에서는 BLEU, ROUGE, METEOR 같은 기존 평가 메트릭이 교육적 가치를 반영하지 않는 문제를 지적하며 KDA라는 새로운 메트릭을 제안했습니다. KDA는 학생의 목표 사실(target fact)에 대한 지식을 평가하는 능력을 중심으로 합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭도 도입했습니다.', 'NLP 논문에서는 spurious pattern에 대한 모델의 의존성을 줄이기 위해 multiple counterfactual을 생성하여, collective decision을 통해 모델의 인과관계를 더욱 robust하게 평가하는 방법을 제안합니다.', 'FixAlly 논문에서는 다중 에이전트 LLM 아키텍처를 활용하여 모바일 앱에서 접근성 문제를 감지하고 수정하는 소스 코드 수정을 제안하는 도구를 소개합니다. 이 도구는 접근성 검사기의 결과를 분석하여 코드 수정 제안을 생성하고, 사용자가 이러한 제안을 검토할 수 있도록 지원합니다.']

- **Performance Highlights**: ['MCQ 논문: KDA_disc와 KDA_cont는 인간 연구에서 강력한 상관관계를 보였으며, n-gram 기반 평가 메트릭과 결합되었을 때 전문가가 라벨링한 여러 MCQ 품질 측정에 대한 예측력이 뛰어났습니다.', 'NLP 논문: 다수의 counterfactual 생성 및 집합적 의사 결정을 통한 접근 방식은 cross-domain generalization, counterfactual robustness, scarce data에 대한 일반화 등 다양한 차원에서 상당한 개선을 이루었습니다.', 'FixAlly 논문: 205개의 iOS 앱에서 77%의 성공률로 plausible fix 제안을 생성했습니다. 12명의 iOS 개발자 대상의 설문 조사 결과에 따르면, 69.4%의 제안을 수용할 의사가 있다고 응답했으며, 특히 초보 개발자들이 유용하게 사용할 수 있는 도구임을 확인했습니다.']



### Generative Language Models with Retrieval Augmented Generation for Automated Short Answer Scoring (https://arxiv.org/abs/2408.03811)
Comments:
          20 pages, 2 figures

- **What's New**: 최근 연구에서는 다중 선택 질문(MCQ) 자동 생성을 개선하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 도입했습니다. 또한, 자연어 처리 과제에서 대조 학습과 반사 사실 증가 기법을 활용해 모델의 내구성을 향상시키는 방법과 최신 생성형 언어 모델(GLMs)을 활용한 자동 짧은 답변 점수 시스템을 제안했습니다.

- **Technical Details**: 첫 번째 연구는 BLEU, ROUGE, METEOR와 같은 기존의 n-gram 기반 유사성 메트릭이 교육적 가치를 평가하지 못하는 문제를 지적하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 두 번째 연구는 대조 학습과 여러 반사 사실을 합성하여 단어의 인과 관계를 더 잘 파악하는 방법을 제안했습니다. 세 번째 연구는 베르트(BERT)와 일렉트라(ELECTRA)와 같은 사전 학습된 인코더 모델을 활용한 정보 검색(IR) 및 RAG와 GLM을 결합한 파이프라인을 제안하여 교실 평가에 있어 점수 정확도를 높였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 메트릭과 결합하여 전문가가 라벨링한 MCQ 품질 측정치에 대한 예측력을 향상시켰습니다. 대조 학습 기법은 반사 사실 강건성, 교차 도메인 일반화, 소량 데이터 일반화에서 상당한 개선 효과를 보였습니다. GLM을 활용한 자동 짧은 답변 점수 시스템은 SemEval 2013 데이터셋에서 기존 방법보다 높은 성과를 보였으며, 특히 SCIENTSBANK 3-way와 2-way 태스크에서 유의미한 개선을 나타냈습니다.



### Navigating the Human Maze: Real-Time Robot Pathfinding with Generative Imitation Learning (https://arxiv.org/abs/2408.03807)
- **What's New**: 이 논문은 복잡한 환경을 자율적으로 탐색하는 로봇을 위해 목표 조건 생성 모델(goal-conditioned generative models)과 샘플링 기반 모델 예측 제어(SMPC, Sampling-based Model Predictive Control)를 통합하는 새 방법을 제안합니다. 인간 군중의 행동을 예측하고 이를 바탕으로 로봇이 자율적으로 움직일 수 있게 합니다.

- **Technical Details**: 제안된 방법은 군중 행동을 목표 조건 생성 모델(goal-conditioned generative models)로 학습하여 잠재적 로봇 경로 샘플을 처리하고 주변 사람들의 반응을 예측합니다. 이를 통해 복잡한 시나리오에서 능동적인 로봇 내비게이션이 가능해집니다. 이 모델은 또한 실시간 내비게이션을 가능하게 하며 충돌률을 크게 줄이고 경로 길이를 단축해줍니다.

- **Performance Highlights**: 제안된 알고리즘은 실제 로봇 플랫폼에서의 실험에서 검증되었으며, 실시간 내비게이션을 가능하게 하고 다양한 기준 수치에서 기존 절차를 능가하는 성능을 보여줍니다. 특히 충돌률 감소와 경로 최적화에서 두각을 나타내었습니다.



### Relevance meets Diversity: A User-Centric Framework for Knowledge Exploration through Recommendations (https://arxiv.org/abs/2408.03772)
- **What's New**: 현대의 추천 시스템(recommenders)에서 연관성과 다양성을 최적화하는 데 초점을 맞춘 새로운 접근 방식을 제안합니다. 제안된 프레임워크는 사용자의 행동을 중심으로 하며, 사용자가 추천 항목과 상호 작용하는 방식에 따라 지식 습득을 극대화하는 것을 목표로 합니다.

- **Technical Details**: 이 논문은 사용자가 추천 시스템과의 상호작용을 지속하는 동안 최대한 많은 지식을 얻을 수 있도록 하기 위해 다양성을 지식 습득의 대리 변수로 사용합니다. 제안된 프레임워크는 다양성과 연관성을 결합한 새로운 추천 전략을 도입하며, 이를 copula 함수로 구현합니다. 두 가지 표준화된 다양성 개념(coverage와 pair-wise distances)에 기반한 모델이 제안되었습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 통해 제안된 방법론을 광범위하게 평가한 결과, 제안된 전략이 기존의 최첨단 경쟁자들을 능가하는 성능을 보여주었습니다. 이 프레임워크는 사용자의 인내심과 추천 항목의 유용성에 따라 종료 확률을 모델링하여 실제 사용자 행동을 반영합니다.



### Online Model-based Anomaly Detection in Multivariate Time Series: Taxonomy, Survey, Research Challenges and Future Directions (https://arxiv.org/abs/2408.03747)
Comments:
          Submitted to Engineering Applications of Artificial Intelligence journal

- **What's New**: 새로운 자동 MCQ(다지선다형 질문) 생성 기법을 평가하는 메트릭 제안. 기존 BLEU, ROUGE, METEOR의 제한성을 극복하고 교육적 가치를 평가하는 KDA(지식 종속 가능성)을 제안함.

- **Technical Details**: MCQ의 대답 가능성(answerability)을 평가하는 KDA, KDA_disc, KDA_cont 메트릭을 소개하고, human evaluation을 통해 실제 교실 환경에서의 사용성을 검증함. 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함.

- **Performance Highlights**: KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 보이며, 기존 n-gram 기반 유사성 메트릭과 결합했을 때 MCQ 품질 평가에서 높은 예측력을 보임.



### Flexible Bayesian Last Layer Models Using Implicit Priors and Diffusion Posterior Sampling (https://arxiv.org/abs/2408.03746)
- **What's New**: 이번 연구에서는 자동 다중 선택 질문(MCQ) 생성을 위한 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 학생의 반응을 기반으로 MCQ의 답변 가능성을 측정하는 것으로, 기존의 BLEU, ROUGE, METEOR 지표들이 고려하지 않는 교육적 가치를 평가합니다.

- **Technical Details**: KDA는 학생 설문 조사를 통해 측정되며, 여기서 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표가 제안됩니다. 이 지표들은 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: 연구 결과, KDA_disc와 KDA_cont가 실제 교실 상황에서 전문가들이 평가한 사용성과 강한 상관관계를 보였습니다. 또한, n-그램 기반 유사성 메트릭과 결합하면 다양한 전문가 평가 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.



### Intuitionistic Fuzzy Cognitive Maps for Interpretable Image Classification (https://arxiv.org/abs/2408.03745)
Comments:
          This work has been submitted for possible journal publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 자동 Multiple Choice Question (MCQ) 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 또한, 자연어 처리(NLP) 태스크에서 사람이 만드는 counterfactual 데이터를 비롯한 다양한 augmentation 방식을 극복하고, 여러 개의 counterfactual 데이터를 통해 단어들의 인과관계를 파악하는 새로운 방법을 제안합니다. 마지막으로, 이미지 분류 문제에서 직관적인 FCMs(iFCMs)를 확장하여 이미지 분류의 해석 가능성을 높이는 새로운 프레임워크 Interpretable Intuitionistic FCM (I2FCM)을 도입합니다.

- **Technical Details**: MCQ 평가에서는 기존의 n-gram 기반 메트릭(BLEU, ROUGE, METEOR)이 교육적 가치를 제대로 평가하지 못하므로, 우리 연구에서는 학생의 대답 가능성(answerability)을 측정하는 KDA를 제안합니다. 특히, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입하여 사전 훈련된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방합니다. NLP 태스크의 경우, 여러 개의 counterfactual 데이터를 생성하고 이들 데이터 집합 간의 분포를 평가하여 모델의 robustness를 높입니다. 또한, 이미지 분류 문제를 해결하기 위해 iFCM을 확장하여, 주어진 이미지의 중요한 영역을 추출하고 데이터 기반으로 직관적인 퍼지 관계를 결정하는 알고리즘을 개발합니다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, BLEU, ROUGE와 같은 기존의 n-gram 기반 메트릭과 결합하면 전문가들이 라벨링한 MCQ 품질 측정에 대해 높은 예측력을 보입니다. NLP 태스크에서는 집합적 의사 결정을 통해 attribution-based synthesis가 갖는 bias를 줄여, counterfactual robustness, cross-domain generalization, sparse data에서의 generalization 등 다양한 측면에서 성능이 향상되었습니다. 또한, iFCM을 확장한 I2FCM 프레임워크는 공개 데이터셋에서 뛰어난 분류 성능을 제공하면서 해석 가능한 추론을 가능하게 하였습니다.



### Advancing Multimodal Large Language Models with Quantization-Aware Scale Learning for Efficient Adaptation (https://arxiv.org/abs/2408.03735)
Comments:
          Accepted by ACMMM2024

- **new_papers**: [{"What's New": '기존 BLEU, ROUGE, METEOR에 대한 한계를 극복하기 위해 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 평가 능력을 평가합니다.', 'Technical Details': 'KDA는 학생 응답을 기반으로 측정되며, 이를 근접하게 모사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이러한 메트릭은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': '실제 강의실 세트에서 사용성과 강한 상관관계를 가지며, BLEU 등과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 측정에 높은 예측 능력을 보였습니다.'}, {"What's New": '강화된 robustness를 위해 contrastive learning 및 counterfactual augmentation을 활용한 새로운 방법을 제안합니다. 기존 접근법과 달리, 집합적 의사 결정 방식을 통해 더 강력한 인과 관계 파악을 가능케합니다.', 'Technical Details': '여러 개의 반사실(counterfactual)을 생성하고, 이 집합에서 예측 분포에 대한 집합적 의사 결정을 통해, 인과 관계를 감독합니다.', 'Performance Highlights': '실험 결과, 기존 방법에 비해 반사실적 robustness, 도메인 간 일반화, 부족한 데이터로부터의 일반화 등 다양한 측면에서 성능이 크게 향상되었습니다.'}, {"What's New": '멀티모달 대형 언어 모델(MLLMs)의 매개변수 양자화(parameter quantization)를 활용하여 자원 제약을 해소하고자 하는 첫 번째 연구를 제공합니다. 이를 위해 QSLAW(Quantization-aware Scale LeArning)를 제안합니다.', 'Technical Details': 'QSLAW는 그룹별 스케일 팩터를 학습하여 양자화 오류를 줄이는 방법과 멀티모달 워크업을 도입하여 언어와 멀티모달 샘플을 점진적으로 통합하는 방법을 사용합니다.', 'Performance Highlights': 'QSLAW는 ScienceQA에서 91.04%의 정확도를 보이며 QLoRA보다 4.08% 향상된 성능을 기록했습니다. 또한, 전체 정밀도 모델보다 더 높은 성능을 나타냈습니다.'}]



### Local Topology Measures of Contextual Language Model Latent Spaces With Applications to Dialogue Term Extraction (https://arxiv.org/abs/2408.03706)
Comments:
          Accepted as a long paper to SIGDIAL 2024. 9 pages, 2 figures, 3 tables

- **What's New**: 자동 다중 선택 문제(MCQ)의 생성은 교사의 평가 작업을 크게 줄일 수 있는 잠재력을 가지고 있습니다. 그러나 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 생성된 MCQ와 데이터셋의 골드 샘플 간의 n-그램 기반 유사성에 중점을 두고 있어 교육적 가치를 간과하고 있습니다. 이를 해결하기 위해, 연구팀은 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안했습니다. KDA는 학생의 지식을 활용하여 MCQ의 답변 가능성을 측정하는 데 중점을 둡니다.

- **Technical Details**: KDA는 학생 설문조사를 통해 학생들의 응답을 기반으로 측정됩니다. 연구팀은 미리 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안했습니다. 이 두 메트릭은 실제 강의실에서의 사용성과 강한 상관관계를 가지는 것으로 나타났습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 (1) KDA와 (2) 전문가들이 라벨링한 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지는 것으로 밝혀졌습니다. 또한, n-그램 기반 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가 라벨 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다.



### A Blockchain-based Reliable Federated Meta-learning for Metaverse: A Dual Game Framework (https://arxiv.org/abs/2408.03694)
Comments:
          Accepted in IEEE Internet of Things Journal

- **What's New**: 최근 자동 MCQ 생성과 평가 메트릭 기존 방식의 한계를 극복하고자, 학생의 지식 평가 능력을 높이는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하였습니다.

- **Technical Details**: 기존 BLEU, ROUGE, METEOR와 같은 n-gram 기반 메트릭의 한계를 극복하기 위해, KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 중점으로 개발되었습니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이들은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 사람 평가를 통해 KDA_disc와 KDA_soft가 실제 강의실에서의 사용성과 강한 상관관계를 보였고, 전문가가 라벨링한 다양한 MCQ 품질 척도에 대한 높은 예측력을 보여주었습니다.



### Generative Design of Periodic Orbits in the Restricted Three-Body Problem (https://arxiv.org/abs/2408.03691)
Comments:
          SPAICE Conference 2024 (7 pages)

- **What's New**: 이 논문은 교육적 가치를 평가하지 못하는 기존 MCQ 생성 평가 메트릭의 한계를 극복하기 위해, Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안합니다. 다른 논문에서는 NLP 태스크의 robustness를 높이기 위해 contrastive learning과 counterfactual augmentation을 사용하는 방식을 탐구하며, Generative AI를 이용하여 Three-Body Problem에서 주기적 궤도를 생성하는 방식에 대한 연구를 다룹니다.

- **Technical Details**: 첫 번째 논문에서는 KDA를 제안하여 MCQ의 답변 가능성을 측정하고, 학생의 지식을 평가할 수 있는 능력을 평가합니다. 두 번째 논문은 여러 개의 counterfactual을 생성하고 이를 기반으로 집합적 의사결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제시합니다. 세 번째 논문은 Variational Autoencoder (VAE)를 이용해 주기적 궤도를 생성하고 이를 기초 물리적 값으로 평가하는 방식에 대한 연구입니다.

- **Performance Highlights**: 첫 번째 논문에서는 인간 설문조사 기반으로 KDA의 유효성을 검증하였고, KDA_disc와 KDA_cont가 실제 강의실 세트에서 강한 상관관계를 보였습니다. 두 번째 논문에서는 제안된 방법이 counterfactual robustness, 교차 도메인 일반화, 드문 데이터로부터의 일반화 등에서 상당한 개선을 이룸을 보여주었습니다. 세 번째 논문에서는 Generative AI가 주기적 궤도를 예측하는데 높은 성능을 보여, 우주 임무 계획과 천체 역학 연구에 도움이 될 수 있음을 나타냈습니다.



### Concept Conductor: Orchestrating Multiple Personalized Concepts in Text-to-Image Synthesis (https://arxiv.org/abs/2408.03632)
Comments:
          Github Page: this https URL

- **What's New**: 이번 연구에서는 다중 개념 텍스트-이미지 모델을 사용자별로 맞춤화하는 프레임워크인 'Concept Conductor'를 소개합니다. 이 프레임워크는 속성 누출(attribute leakage)과 레이아웃 혼동(layout confusion)을 방지하여 시각적 충실도와 올바른 레이아웃을 보장합니다.

- **Technical Details**: 'Concept Conductor'는 다중 경로 샘플링(multi-path sampling), 레이아웃 조정(layout alignment), 개념 주입(concept injection) 세 가지 주요 요소로 구성됩니다. Multi-path 샘플링은 다중 개념 모델들이 독립적인 디노이징(denoising) 프로세스를 유지하게 하여 속성 누출을 방지합니다. 레이아웃 조정은 각 모델이 올바른 레이아웃을 생성하도록 자기 주의 기반 공간 안내(self-attention-based spatial guidance)를 사용합니다. 개념 주입은 각 개념의 시각적 특징을 최종 생성 이미지에 완전히 주입하여 조화를 보장합니다.

- **Performance Highlights**: 제시된 방법은 30개 개념을 포함하는 새로운 데이터셋에서 검증되었으며, 기존 방법들보다 개념 충실도와 텍스트 의미 일치도에서 뛰어난 성과를 보였습니다. 시각적 세부 사항을 잘 유지하면서도 정확한 레이아웃을 생성하는 데 성공했으며, 성능 개선이 두드러졌습니다.



### Improving the quality of Persian clinical text with a novel spelling correction system (https://arxiv.org/abs/2408.03622)
- **What's New**: 새로운 MCQ 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안함으로써 기존의 BLEU, ROUGE, METEOR과 같은 평가 방법들이 놓치고 있는 교육적 가치를 확인하고자 합니다. 또한, 최근 NLP 태스크에서 spurious pattern 의존성 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 새로운 접근법을 제시합니다. 마지막으로, 페르시아어 임상 텍스트에서 철자 오류를 감지하고 교정하기 위한 혁신적인 방법을 개발했습니다.

- **Technical Details**: [{'Multiple Choice Questions (MCQ)': '기존의 n-gram 기반 평가 메트릭을 대체하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다. KDA는 학생들이 대상 사실에 대한 지식을 기반으로 MCQ를 답변할 수 있는 능력을 측정합니다. Human survey를 통해 KDA를 측정하고, pre-trained language models를 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다.', 'Robustness in NLP Models': '기존의 방법들이 사람이 또는 모델이 counterfactual을 추가하는 방식의 한계를 극복하기 위해, 여러 개의 counterfactual을 생성하고 collective decision-making을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안합니다. 이를 통해 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 성능이 향상되었습니다.', 'Persian Clinical Text Spelling Correction': '페르시아어 임상 텍스트에서 철자 오류를 감지하고 교정하기 위해 최신 pretrained model과 orthographic similarity matching algorithm인 PERTO를 사용했습니다. 특히, PERTO 알고리즘은 시각적 유사성을 사용하여 교정 후보를 랭킹합니다.'}]

- **Performance Highlights**: [{'MCQ Evaluation': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 지님을 human studies에서 확인했습니다. n-gram 기반 메트릭과 결합할 경우 다양한 전문가가 라벨링한 MCQ 품질 측정에 대한 강한 예측력을 보였습니다.', 'NLP Robustness': '집합적 의사결정을 통해 attribution-based synthesis의 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 성능이 크게 향상되었습니다.', 'Persian Clinical Spelling Correction': '비단어 오류 교정에서 PERTO 알고리즘을 사용했을 때 F1-Score 90.0%를 달성했습니다. 실제 단어 오류 감지에서도 최고 F1-Score 90.6%를, 실제 단어 오류 교정에서는 최고 F1-Score 91.5%를 기록했습니다.'}]



### A Logical Fallacy-Informed Framework for Argument Generation (https://arxiv.org/abs/2408.03618)
- **What's New**: 최신 연구들은 자동 다지선다형 질문 (MCQ) 생성과 논리적인 주장에서의 언어 모델 성능을 개선하기 위해 새로운 접근법을 제안하고 있습니다. MCQ 생성에서는 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 도입했고, 논리적인 주장 생성에서는 논리적 오류를 인식하고 방지하는 FIPO 프레임워크를 선보였습니다.

- **Technical Details**: MCQ 생성에 대해, 기존의 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 평가하지 못하는 문제가 있습니다. 이를 해결하기 위해 지식 종속 가능성(KDA) 메트릭이 도입되었으며, 학생의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont를 제안했습니다. 논리적인 주장의 생성에서는 기존의 대안사실 (counterfactual) 학습에 의존하지 않고, 여러 개의 대안사실을 생성하여 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 더 견고하게 파악하는 방법을 채택했습니다. 또한, FIPO 프레임워크는 LLM이 논리적 오류를 최소화할 수 있도록 선호도 최적화(preference optimization) 방법을 사용하고, fallacy 카테고리에 대한 세부 정보를 포착하는 분류 손실을 포함합니다.

- **Performance Highlights**: MCQ 생성에서 KDA 기반 메트릭은 기존의 n-gram 기반 메트릭과 결합하여 전문가가 평가한 MCQ 품질 측정에서 높은 예측력을 보였습니다. 실험 결과, KDA_disc와 KDA_cont는 실제 교육 현장에서 높은 사용성을 가졌습니다. 논리적인 주장에서, FIPO 프레임워크는 논리적 오류를 최대 17.5%까지 줄였으며, 인간 평가 결과에 따르면 FIPO를 통한 주장이 기존의 미세 조정된 모델과 DPO 같은 선호도 최적화 방법보다 뛰어남을 보였습니다.



### Is Child-Directed Speech Effective Training Data for Language Models? (https://arxiv.org/abs/2408.03617)
Comments:
          Preprint. Code and data will be released soon

- **title**: 최근 아카이브 논문 요약: 다중 선택 문제(MCQ)와 학습 모델의 강건성 및 아동 언어 학습 데이터

- **date_range**: 2023년 10월

- **topics**: [{"What's New": "자동으로 다중 선택 문제 (MCQ)를 생성하는 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)' 도입. 이는 MCQ의 목표 사실에 대한 학생의 지식을 평가하는 능력을 측정.", 'Technical Details': '이 메트릭은 학생 응답을 기반으로 KDA를 측정하며, 사전 훈련된 언어 모델을 활용해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안. 인간 평가와의 강한 상관 관계를 통해 강의실에서의 실제 사용성을 검증.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 전문가 레이블 MCQ 품질 측정치와 강한 예측력을 보임.'}, {"What's New": 'NLP 태스크에서 모델의 강건성 향상을 위한 대조 학습과 반사실적 증가 (counterfactual augmentation) 도입.', 'Technical Details': '기존 방식과 달리 다수의 반사실적(countefactual)을 생성하고 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과 관계를 분석. 이는 spurious correlation에 덜 민감하게 하여, 강건성을 높임.', 'Performance Highlights': '반사실적 강건성, 교차 도메인 일반화, 희소 데이터에서의 일반화 부문에서 성능 향상.'}, {"What's New": '아동이 언어를 효율적으로 학습하는 이유에 대한 분석. 아동에게 제공되는 데이터의 특성이 언어 모델링 목표에 어떻게 기여하는지 조사.', 'Technical Details': 'GPT-2 모델을 2천9백만 단어의 아동 지향 언어와 인위적 데이터셋(TinyDialogues)으로 훈련시킴. 개발적으로 영감을 받은 평가를 통해 구문 및 의미 지식을 비교.', 'Performance Highlights': '아동 언어 입력의 지역적 속성(local properties)이 모델 성능에 영향을 미침. 그러나 아동 언어 학습의 효율성은 데이터의 품질뿐만 아니라 상호작용에서도 기인.'}]

- **further_reading**: 상세한 논문 내용은 arXiv에서 확인하십시오. [논문 링크 (insert link here)]



### EnJa: Ensemble Jailbreak on Large Language Models (https://arxiv.org/abs/2408.03603)
- **What's New**: 이번 연구에서는 최근 대형 언어 모델(LLM)들이 안전이 중요한 애플리케이션에 점점 더 많이 활용됨에 따라, 이들의 잠재적 jailbreak 공격에 대한 취약성이 커지고 있다는 문제를 해결하려 합니다. 연구에서는 기존의 prompt-level 공격과 token-level 공격을 결합해, 보다 강력한 하이브리드 jailbreak 공격 기법 'Ensemble Jailbreak'를 제안합니다.

- **Technical Details**: 연구팀은 EnJa 공격을 통해, prompt-level jailbreak를 이용해 해로운 지시사항을 숨기고, gradient 기반 공격 기법을 활용해 공격 성공률을 높이며, 이 두 가지 jailbreak 공격을 template 기반 연결 기법으로 통합합니다. 특히, prompt-level 기법은 스토리나 논리를 구성하여 안전 정렬(Alignment)을 무력화시키고, token-level 공격은 gradient 방법을 사용하여 적대적 토큰을 찾아내는 방식입니다.

- **Performance Highlights**: EnJa 공격 기법은 기존의 개별 jailbreak 공격보다 훨씬 적은 쿼리로도 높은 공격 성공률을 달성하는 데 성공했습니다. 이를 통해, 여러 정렬된 모델에서 매우 강력한 공격 성능을 보여줍니다.



### Activations Through Extensions: A Framework To Boost Performance Of Neural Networks (https://arxiv.org/abs/2408.03599)
- **What's New**: 자동 MCQ 생성 기술 및 교육적 평가 지표 개선, NLP에서의 강건성 향상 방법, 활성화 함수에 대한 통합 프레임워크 제안.

- **Technical Details**: 첫 번째 논문에서는 기존 MCQ 평가 메트릭의 한계를 극복하기 위해 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭을 제안. 두 번째 논문은 일반적 강화 학습과 반사실적 데이터 증강을 통해 NLP 모델의 강건성을 향상시키는 방법을 논한다. 세 번째 논문은 활성화 함수의 통합 프레임워크를 제안하고, 이를 통해 신경망의 확장 (extension)을 이루는 새로운 기술을 소개.

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였다.', '새로운 반사실적 데이터 증강 방법은 기존 방법들에 비해 다양한 차원에서 개선된 강건성을 보여주었다: 반사실 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희소 데이터로부터 일반화.', '신경망 확장은 표준 테스트 함수에서 성능 개선을 이루면서도 공간 및 시간 복잡성에 거의 영향을 미치지 않았다.']



### Focal Depth Estimation: A Calibration-Free, Subject- and Daytime Invariant Approach (https://arxiv.org/abs/2408.03591)
- **What's New**: 이 논문들은 다양한 기술 혁신을 소개하며 교육, NLP, 그리고 개인화된 기술 분야에서 중요한 발전을 이루었습니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 평가 메트릭이 단순히 n-gram 유사성에만 초점을 맞추어 교육적 가치를 무시하는 문제를 해결하기 위해, 저자들은 대상 사실(target fact)에 대한 지식을 바탕으로 MCQ의 답변 가능성(answerability)을 측정하는 지식 종속 가능성(KDA)를 새로운 자동 평가 메트릭으로 제안했습니다. Human survey를 통한 KDA 측정 후, 이들을 모방한 모델로 KDA_disc와 KDA_cont 메트릭을 개발하여 강의실 사용성과 강한 상관관계를 보였습니다.', 'NLP Robustness': '최근의 deep model들이 NLP 태스크에서 높은 정확도를 보이나 spurious pattern에 의존해 robustness가 제한된다는 문제를 해결하기 위해, 저자들은 대조 학습(contrastive learning)과 counterfactual augmentation을 제안했습니다. 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 감독하는 방법을 제안하였습니다.', 'Focal Depth Estimation': '기존의 시선 추적 시스템과 autofocal 안경이 빈번한 사용자별 보정이 필요한 문제를 해결하기 위해, 저자들은 LSTM 네트워크와 도메인 특화된 피처 엔지니어링을 활용하여 초점 거리를 추정하는 획기적인 보정 없는 방법을 소개했습니다. 이 접근법은 평균 절대 오차(MAE) 10 cm 이하의 정확도를 달성하며 autofocal 안경의 실용성을 크게 향상시킵니다. 또한 이 기술은 가상현실(VR), 증강현실(AR), 의학적 현미경 검사 및 로봇 수술 등 다양한 분야에도 적용 가능합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont 메트릭은 실제 교실 환경에서의 사용성과 강한 상관관계를 보였으며, 다른 전문가가 라벨링한 MCQ 품질 지표와도 높은 예측력을 가집니다. 이는 MCQ의 교육적 가치를 평가하는 데 중요한 도구가 될 것입니다.', 'NLP Robustness': '제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 희소한 데이터로부터의 일반화에서 유의미한 개선을 보여줍니다.', 'Focal Depth Estimation': '저자들의 FOVAL 모델은 기존 모든 최첨단 방법보다 우수한 초점 거리 추정 정확도를 달성하며, 사용자의 빈번한 보정 없이도 실생활에서 높은 활용성을 보여주었습니다.'}



### Facing the Music: Tackling Singing Voice Separation in Cinematic Audio Source Separation (https://arxiv.org/abs/2408.03588)
Comments:
          Submitted to the Late-Breaking Demo Session of the 25th International Society for Music Information Retrieval (ISMIR) Conference, 2024

- **What's New**: 자동 MCQ 생성의 교육적 가치 평가를 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 도입했습니다. 이는 기존의 BLEU, ROUGE, METEOR와 달리 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: KDA는 학생들이 문제를 풀 때의 행동을 모사하는 pre-trained language models를 활용하여 자동으로 평가됩니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되었습니다.

- **Performance Highlights**: 사람을 대상으로 한 연구 결과, KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, 이는 전문가들이 라벨링한 결과와도 일치합니다. n-gram 기반의 유사성 메트릭과 결합되었을 때, 이들 메트릭은 다양한 품질 측정 기준에 대해 높은 예측력을 보였습니다.



### Hierarchical Neural Constructive Solver for Real-world TSP Scenarios (https://arxiv.org/abs/2408.03585)
Comments:
          Accepted to KDD 2024

- **What's New**: MCQ 자동 생성 평가에 새로운 메트릭인 Knowledge Dependent Answerability(KDA)를 도입했습니다. KDA는 학생이 대상 사실에 대한 지식을 바탕으로 MCQ에 답할 수 있는 능력을 평가합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에 집중하지만, KDA는 학생 응답을 바탕으로 MCQ의 답변 가능성을 측정합니다. 추가로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: Human studies를 통해, KDA_disc와 KDA_soft가 KDA 및 전문가가 레이블링한 실제 강의실 세트에서의 사용성과 강한 상관관계를 가졌음을 확인했습니다. 특히, n-gram 기반 유사성 메트릭과 결합할 경우, 다양한 전문가 레이블 MCQ 품질 측정에 대해 높은 예측력을 보였습니다.



### Active Testing of Large Language Model via Multi-Stage Sampling (https://arxiv.org/abs/2408.03573)
- **What's New**: 이번 뉴스레터에서는 자동 다지 선택형 질문 (Multiple Choice Questions, MCQ) 생성 평가와 관련된 새로운 연구, NLP 태스크의 robust 감응성을 향상시키기 위한 대조 학습과 반사실적 증강 (counterfactual augmentation)의 활용, 그리고 대형 언어 모델 (Large Language Models, LLM) 평가를 효율적으로 수행하기 위한 새로운 액티브 테스트 프레임워크를 소개합니다.

- **Technical Details**: [{"What's New": '지식 종속 가능성(Knowledge Dependent Answerability, KDA)라는 새로운 평가 메트릭을 제안하여, 기존 BLEU, ROUGE, METEOR 등이 고려하지 않았던 MCQ의 교육적 가치를 평가합니다.', 'Technical Details': 'KDA 메트릭은 학생이 대상 사실을 알고 있을 때 MCQ의 대답 가능성을 측정합니다. 이를 위해 인간 설문을 통해 학생 응답을 기반으로 KDA를 측정하는 방법을 먼저 보여주고, 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방함으로써 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 인간 연구를 통해 실제 강의실 세트의 사용성과 강한 상관관계를 가지고 있음을 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 전문가들이 라벨링한 MCQ 품질 측정치에 대해 강력한 예측력을 가지는 것으로 나타났습니다.'}, {"What's New": '대조 학습과 반사실적 증강 (counterfactual augmentation)을 통해 NLP 태스크의 모델 감응성을 향상시키는 방안을 제시합니다.', 'Technical Details': "기존 방법과 달리, 우리는 '여러 개'의 반사실적 샘플을 생성하고, 이 집합에 대한 예측 분포로부터 집합적 의사결정을 통해 각 단어의 인과관계를 robust하게 슈퍼바이즈하는 방법을 제안합니다.", 'Performance Highlights': '실험 결과, 이 접근법은 다양한 차원에서 유의미한 개선을 이루었으며, 특히 반사실적 로버스트넷(countrafactual robustness), 도메인 간 일반화(cross-domain generalization), 및 드문 데이터 일반화(generalization from scarce data)에서 성과를 보였습니다.'}, {"What's New": '효율적인 LLM 평가법인 AcTracer를 소개합니다.', 'Technical Details': 'AcTracer는 풀 기반 멀티 스테이지 액티브 선택을 통해 테스트 샘플링 과정을 안내하는 활성화된 테스트 프레임워크로, LLM의 내부 상태와 외부 정보를 활용합니다.', 'Performance Highlights': '실험 결과, AcTracer는 7개의 다양한 도메인에서 최첨단 성능을 달성했으며, 기존 방법들에 비해 최대 38.83%의 성능 향상을 보였습니다.'}]



### 2D-OOB: Attributing Data Contribution through Joint Valuation Framework (https://arxiv.org/abs/2408.03572)
- **What's New**: [{'title': '새로운 MCQ 자동 평가 메트릭', 'summary': '기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 단어 유사도에 기반한 평가를 제공하며 교육적 가치를 무시하고 있습니다. 이를 해결하기 위해 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. KDA는 사실에 대한 학생의 지식으로부터 MCQ의 답변 가능성을 측정합니다. 우리는 pre-trained language models를 이용해 학생의 문제 해결 행동을 모방하여 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안하였습니다. Human study에서 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성 및 MCQ 품질 측정에서 높은 예측력을 보였습니다.'}, {'title': '데이터 셀 수준 평가 메트릭', 'summary': '기존의 데이터 평가 방법은 단일 스칼라 점수를 할당하여 데이터 포인트 내 개별 셀의 다양성을 흐리게 하여 해석 가능성을 저하시켰습니다. 이를 해결하기 위해 2D-OOB라는 평가 메트릭을 제안하였습니다. 2D-OOB는 데이터 포인트 내 각 셀의 중요성을 측정하며, 세부적인 outlier 감지 및 data poisoning 공격에서의 backdoor trigger 위치를 파악할 수 있습니다.'}]

- **Technical Details**: [{'title': 'KDA 메트릭 세부 기술', 'summary': 'KDA 메트릭은 target fact에 대한 학생의 지식을 활용하여 MCQ의 답변 가능성을 측정합니다. 우리는 학생들의 응답을 기반으로 KDA 측정을 설계하고, pre-trained language models를 활용해 학생의 문제 해결 행동을 모방하여 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하였습니다.'}, {'title': '2D-OOB 프레임워크', 'summary': '2D-OOB는 out-of-bag estimation 프레임워크로, 데이터 포인트의 개별 셀 값을 평가하여 데이터셋 내 특정 셀이 모델 학습에 미치는 영향을 측정합니다. 2D-OOB는 데이터 poisoning 공격에서 backdoor trigger를 정확하게 식별하는데도 효과가 뛰어납니다.'}]

- **Performance Highlights**: [{'title': 'KDA 메트릭 성능', 'summary': 'KDA_disc와 KDA_cont는 Human evaluation을 통해 실제 강의실 환경에서의 사용성과 높은 상관관계를 보였습니다. 이 메트릭들은 BLEU, ROUGE, METEOR 등 n-gram 기반 유사도 메트릭과 결합되었을 때 다양한 전문가가 정한 MCQ 품질 측정에서 강력한 예측력을 보였습니다.'}, {'title': '2D-OOB 성능', 'summary': '2D-OOB는 다양한 사용 사례에서 최첨단 성능을 달성했으며, 세분화된 outliers 탐지 및 모델 성능 향상을 위해 어느 셀을 수정해야 하는지 정확하게 파악할 수 있었습니다. 2D-OOB는 기존 최첨단 방법들보다 200배 더 빠릅니다.'}]



### A Comparison of LLM Finetuning Methods & Evaluation Metrics with Travel Chatbot Use Cas (https://arxiv.org/abs/2408.03562)
- **What's New**: 이번 연구에서는 MCQ (Multiple Choice Questions) 자동 생성 평가에서 기존의 BLEU, ROUGE, METEOR 메트릭들이 교육적 가치를 평가하지 못하는 문제에 착안하여, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. KDA는 주어진 대상 사실에 대한 학생의 지식에 기반하여 MCQ의 답변 가능성을 평가합니다.

- **Technical Details**: KDA를 측정하기 위해 인간 설문조사에서 학생의 반응을 바탕으로 KDA를 계산하는 방법을 제시하였으며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하였습니다. 이 메트릭들은 실제 강의실에서 사용성을 평가받아 강한 상관관계를 보였습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc 및 KDA_cont가 실제 강의실 평가와 강한 상관관계를 가짐을 보여주었습니다. n-gram 기반 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 다양한 전문가 레이블 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.



### MPC-Minimized Secure LLM Inferenc (https://arxiv.org/abs/2408.03561)
- **What's New**: Marill은 최신 대형 언어 모델(LLMs)의 비공개 설정에서 효율성을 높이는 보안 프레임워크입니다. 이는 모델의 세부 조정을 통해 MPC (Multi-Party Computation) 의 사용을 최소화하고, 고비용 연산을 MPC 외부로 이전하여 보안성을 유지하면서 비용을 절감할 수 있도록 설계되었습니다.

- **Technical Details**: Marill은 다음과 같은 전략을 통해 MPC 비용을 최소화합니다: (1) Layer Freezing: 최종 레이어만 조정하여 평가할 레이어 수를 줄임. (2) Low-rank Adaptation (LoRA): 모델 가중치의 작은 부분만 조정하여 매트릭스 곱셈의 차원을 줄임. (3) Head-merging: 자기-어텐션 모듈에서 여러 주의를 하나로 합쳐 연산 비용을 줄임. 이 세 가지 접근법을 통해 모델의 연산 및 통신 오버헤드를 대폭 줄일 수 있습니다.

- **Performance Highlights**: Marill을 사용한 모델은 기존의 세부 조정 모델에 비해 3.6배에서 11.3배 더 빠른 실행 시간을 그리고 2.4배에서 6.9배 더 낮은 통신 오버헤드를 보여줍니다. 또한, 코드 생성, 대화형 봇, 기계 번역 등 다양한 벤치마크에서 본래 성능의 90% 이상을 유지하는 결과를 나타냈습니다.



### D2Styler: Advancing Arbitrary Style Transfer with Discrete Diffusion Methods (https://arxiv.org/abs/2408.03558)
Comments:
          Paper accepted at 27th International Conference on Pattern Recognition (ICPR), 2024

- **What's New**: 최근의 deep 모델이 NLP 작업에서 사람보다 뛰어난 정확도를 보이지만, spurious patterns에 의존해 robustness가 제한되고 있다는 보고가 있습니다. 이에 대처하기 위해 대비 학습(contrastive learning)과 가상 증강(counterfactual augmentation)을 활용하는 연구가 제안되었습니다.

- **Technical Details**: 기존 방식들은 사람이 가상 사례를 추가하거나 기계가 데이터셋에서 가상 사례와 유사한 항목을 찾아야 하는데, 이 과정에서도 spurious correlation의 영향을 받습니다. 이번 연구는 '여러 개의' 가상 사례를 생성하고, 집합적 의사 결정(collective decisions)을 통해 각 단어의 인과관계를 더욱 robust하게 감독하는 방법을 제안합니다.

- **Performance Highlights**: 이 접근 방식은 attribution-based synthesis의 편향에 덜 민감하며 다음과 같은 다양한 측면에서 유의미한 성과를 거두었습니다: 1) 가상 사례의 robustness, 2) 도메인 간 일반화(cross-domain generalization), 3) 부족한 데이터에서의 일반화(generalization from scarce data).



### Unlocking the Non-Native Language Context Limitation: Native Language Prompting Facilitates Knowledge Elicitation (https://arxiv.org/abs/2408.03544)
- **What's New**: 새로운 평가법을 제안한 3개의 논문이 발표되었습니다. 첫 번째 논문은 지식 종속가능성(KDA)을 사용하여 자동 Multiple Choice Questions(MCQ) 생성의 교육적 가치를 측정하는 방법을 제안합니다. 두 번째 논문은 대조적 학습과 반사실적 데이터 증강을 통해 NLP 태스크의 강건성을 향상시키는 방법을 다룹니다. 세 번째 논문은 다국어 대형 언어 모델(MLLM)이 비주류 언어로 된 질문에 효과적으로 답하기 위해 Native Language Prompting(NatLan)을 사용하는 방법을 제안합니다.

- **Technical Details**: 첫 번째 논문에서는 KDA라는 새로운 메트릭을 소개하여, 학생들의 반응을 기반으로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 논문에서는 여러 개의 반사실적 데이터를 생성하고 집합적 의사 결정을 통해 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 세 번째 논문에서는 Native Language Prompting을 통해 MLLM이 주류 언어의 풍부한 지식을 활용하여 비주류 언어 질문에 더 나은 답변을 제공할 수 있도록 하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트에서 높은 상관관계를 보였습니다. 반사실적 데이터 증강을 통해 다차원에서 성능이 크게 향상되었습니다: 반사실적 강건성, 크로스 도메인 일반화, 데이터 부족 상황에서의 일반화. NatLan은 C-Eval benchmark에서 최대 10.1%의 평균 정확도 향상과 5.0%의 hard-level subset에서의 성능 향상을 달성했습니다.



### Automatic identification of the area covered by acorn trees in the dehesa (pastureland) Extremadura of Spain (https://arxiv.org/abs/2408.03542)
Comments:
          22 pages, 15 Figures, 2 Tables

- **What's New**: 최근 학술 논문에서 다양한 주제의 연구 결과들이 발표되었습니다. 첫 번째 논문은 MCQ(객관식 문제)의 자동 생성 평가를 위해 새로운 평가 메트릭을 제안했으며, 두 번째 논문은 대조 학습과 반사실적 증강(counterfactual augmentation)을 통한 NLP 모델의 로버스트니스 향상을 다루고 있습니다. 세 번째 논문은 항공 디지털 이미지를 활용하여 도토리 나무의 덮인 면적(CWA)을 자동으로 추정함으로써 이베리코 돼지의 생산량을 최적화하는 방법에 대해 논의하고 있습니다.

- **Technical Details**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Highlights': ['기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)를 제안.', 'KDA는 학생들이 대상 사실을 알 때 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 통해 이를 예측.', '휴먼 스터디 결과, KDA_disc와 KDA_cont는 실제 교실 환경에서의 활용도와 강한 상관관계를 가짐.']}, {'Paper': 'Robustness in NLP Models through Contrastive Learning and Counterfactual Augmentation', 'Highlights': ['NLP 태스크에서 deep model이 spurious pattern에 의존해 robustness가 제한되는 문제를 해결하기 위해 대조 학습과 반사실적 증강을 활용.', '기존 증강 방법이 spurious correlation에 영향을 받는 문제점을 극복하기 위해 여러 개의 반사실적을 생성하고, 집합적 의사결정을 통해 용어들의 인과관계를 파악하는 방법을 제안.', '이 접근법은 counterfactual robustness, cross-domain generalization, 및 scarce data에서의 일반화 등 다양한 차원에서 유의미한 성능 향상을 달성.']}, {'Paper': 'Automatic Estimation of Covered Wooded Area (CWA) in Spanish Dehesa using Aerial Digital Images', 'Highlights': ['스페인의 도토리 나무 덮인 면적(CWA)을 자동으로 추정하기 위해 항공 디지털 이미지(orthophotos)를 활용.', 'Gustafson-Kessel (GK)의 클러스터링 알고리즘을 Babuska 수정 버전(GK-B)으로 사용하여 자동 분할 방법을 제안.', '142 헥타르 면적에 대한 실험 결과가 실제 이미지와 손으로 분할된 이미지와 비교 시 유망한 결과를 보임.']}]

- **Performance Highlights**: [{'Paper': 'Automatic Generation of Multiple Choice Questions', 'Details': ['KDA_disc와 KDA_cont가 기존 n-gram 기반 메트릭과 결합했을 때 전문가가 레이블한 MCQ 품질 측정에 강한 예측력을 가짐.']}, {'Paper': 'Robustness in NLP Models through Contrastive Learning and Counterfactual Augmentation', 'Details': ['집합적 의사 결정 접근법을 사용하여 대조 학습과 반사실적 증강을 통해 다양한 NLP 응용 분야에서 중요한 성능 향상을 달성.']}, {'Paper': 'Automatic Estimation of Covered Wooded Area (CWA) in Spanish Dehesa using Aerial Digital Images', 'Details': ['제안된 방법이 도토리 나무의 덮인 면적 추정에서 성공적으로 적용되었으며, 이베리코 돼지 생산 최적화에 적용될 잠재력이 높다.']}]



### EXAONE 3.0 7.8B Instruction Tuned Language Mod (https://arxiv.org/abs/2408.03541)
- **What's New**: 이번 뉴스레터에서는 교육용 객관식 질문 생성, NLP 모델의 robust성을 높이기 위한 방법론, 그리고 LG AI Research에서 개발한 EXAONE 3.0 공개 모델 발표를 다룬 세 가지 논문을 소개합니다.

- **Papers**: [{'title': '교육용 객관식 질문 자동 생성에 대한 평가 메트릭 개선', 'Research Highlight': [{"What's New": '기존의 평가 메트릭 (BLEU, ROUGE, METEOR) 는 교육적 가치를 고려하지 않고 단순히 단어의 유사성만 평가한다는 한계가 있습니다. 이를 해결하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.', 'Technical Details': 'KDA는 학생들이 대상 사실에 대한 지식을 바탕으로 객관식 질문에 답변할 수 있는 능력을 측정합니다. 이를 인공지능 모델로 평가하기 위해 KDA_disc와 KDA_cont 두 가지 자동 메트릭을 제안하며, 사전 학습된 언어 모델을 활용해 학생들이 문제를 푸는 행태를 모방합니다.', 'Performance Highlights': 'Human survey를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 이루고 있음을 입증했습니다. 또한, n-gram 기반 평가 메트릭과 결합할 때, 전문가가 라벨링한 MCQ 품질에 대한 예측력이 높아지는 것으로 나타났습니다.'}]}, {'title': 'Deep Models의 Robustness를 위한 Contrastive Learning 및 Counterfactual Augmentation', 'Research Highlight': [{"What's New": '최근의 deep model들이 매우 높은 정확성을 보였지만, spurious pattern에 의존해 robustness가 제한된다는 문제점을 다루고 있습니다. 이를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용합니다.', 'Technical Details': '기존 방법들은 사람이 직접 counterfactual을 만들거나 모델이 데이터셋에서 비슷한 것을 발견해야 했으나, 우리의 접근법은 여러 개의 counterfactual을 생성하고 collective decision을 통해 단어들의 인과관계를 robust하게 파악합니다.', 'Performance Highlights': '우리 접근법은 attribution-based synthesis에서 발생하는 task model bias에 덜 민감하며, 다양한 차원에서 성능 향상을 보여주었습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization.'}]}, {'title': 'EXAONE 3.0: LG AI 연구소의 Instruction-Tuned 대규모 언어 모델 공개', 'Research Highlight': [{"What's New": 'LG AI 연구소는 EXAONE 3.0 instruction-tuned 언어 모델을 발표하며, 7.8B 모델을 비상업적인 연구 목적으로 공개했습니다. 이 모델은 다양한 공공 및 내부 벤치마크에서 높은 성능을 보여줍니다.', 'Technical Details': 'EXAONE 3.0은 bilingual 지원을 위해 효율적인 tokenization과 다양한 데이터셋에 대한 광범위한 사전 학습 및 고급 post-training 기술을 적용하여, 강사 지침을 잘 따르는 모델로 설계되었습니다. ', 'Performance Highlights': '성능 평가 결과, EXAONE 3.0은 영어와 한국어 모두에서 뛰어난 성능을 보였으며, 특히 한국어에서 두드러진 성과를 보여주었습니다. EXAONE 3.0은 실제 환경에서도 높은 실효성을 자랑하며, 전문가 수준의 인공지능 발전에 기여할 것으로 기대됩니다.'}]}]



### Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation (https://arxiv.org/abs/2408.03533)
- **What's New**: 자동 MCQ 생성에 대한 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 소개합니다. 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 데이터셋의 골드 샘플과 n-gram 유사성에만 집중하고 교육적 가치를 고려하지 않았는데, KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 고려합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 인간 조사에서 측정되며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이 자동 메트릭은 강의실에서 사용성을 가진다는 것을 인간 연구를 통해 입증하였습니다. 또한, n-gram 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 다양한 전문가 레이블 MCQ 품질 측정에 대한 강한 예측력을 보입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 레이블을 기반으로 한 실제 강의실 설정에서의 사용성과 강한 상관 관계가 있습니다. n-gram 기반 메트릭과 결합할 경우, 다양한 MCQ 품질 측정에 큰 예측력을 보여줍니다.



### Exploring the extent of similarities in software failures across industries using LLMs (https://arxiv.org/abs/2408.03528)
- **what's new**: 이 연구에서는 소프트웨어 실패 분석에 대한 새로운 접근을 제안합니다. FAIL(Failure Analysis Investigation with LLMs) 모델을 확장하여 뉴스 기사로부터 산업별 소프트웨어 실패 유형을 추출하고 이를 카테고리화합니다.

- **Technical Details**: 이 연구는 대형 언어 모델(LLMs)과 프롬프트 엔지니어링(prompt engineering)을 활용하여 소프트웨어 실패 정보를 추출합니다. 초기 프롬프트는 그다지 효과적이지 않았으나, 몇 번의 수정을 통해 더 일관성 있는 결과를 얻을 수 있었습니다. 이를 통해 데이터베이스를 통해 각 산업별로 소프트웨어 실패를 분류하는 명확한 결과를 도출하였습니다.

- **Performance Highlights**: 이 분석은 금융, 의료, 정보, 교육, 교통, 엔터테인먼트 및 정부 섹터에서 보안 취약점의 빈도가 가장 높다는 것을 보여줍니다. 각 산업별로 소프트웨어 실패의 빈도가 큼을 시각적으로도 확인할 수 있습니다. 이러한 카테고리화는 소프트웨어 엔지니어와 회사들이 공통적인 실패를 식별하고 이에 대한 대응책을 마련하는 데 큰 도움이 됩니다.

- **Discussion**: 이 연구는 특정 산업에서 발생하는 주요 소프트웨어 실패 유형을 자세히 설명하며, 이를 통해 각 산업에 맞춘 개선 노력을 강조합니다. 보안 감사, 보안 교육, 안전한 코딩 관행, 사고 대응 계획, 접근 제어 등의 정책과 모범 사례를 권장합니다. 그러나, FAIL 데이터베이스에 의존한다는 한계와 뉴스 기사에서 정보를 수집할 때 발생할 수 있는 편향 가능성이 존재합니다.



### Hierarchical learning control for autonomous robots inspired by central nervous system (https://arxiv.org/abs/2408.03525)
- **What's New**: 이번 주 AI 뉴스를 통해 소개할 논문은 자동 여러 선택 질문(MCQ; Multiple Choice Questions) 생성에 대한 새로운 평가 메트릭과 내구성을 향상시키기 위한 대조적 학습 및 반사실적 증강 방법, 그리고 포유류의 중앙 신경계 구조를 모방한 계층적 학습 제어 프레임워크에 관한 연구를 포함하고 있습니다.

- **Technical Details**: [{'title': 'MCQ 자동 생성 평가 메트릭', 'details': "기존의 MCQ 생성 평가 메트릭(BLEU, ROUGE, METEOR 등)은 데이터셋의 골드 샘플과 생성된 MCQ의 단어 유사도를 기준으로 하여 교육적 가치는 고려하지 않습니다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 '지식 종속 가능성(KDA; Knowledge Dependent Answerability)'을 제안하며, 학생의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 활용하여 자동 평가 메트릭인 KDA_disc와 KDA_cont를 도입하였습니다."}, {'title': '대조적 학습과 반사실적 증강을 활용한 내구성 강화', 'details': '최근 NLP 태스크에서의 deep model은 사람보다 나은 정확성을 보였으나, spurious pattern에 의존하는 문제로 인해 내구성이 제한됩니다. 본 논문은 여러 개의 반사실적(counterfactual)을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 평가하여, 내구성을 강화하는 방법을 제안합니다. 이는 다양한 차원에서 내구성, 도메인 간 일반화, 희소 데이터 상황에서의 일반화 성능을 향상시킵니다.'}, {'title': '포유류 중앙 신경계 구조를 모방한 계층적 학습 제어 프레임워크', 'details': '포유류의 중앙 신경계 구조를 모방한 새로운 계층적 학습 제어 프레임워크를 제안합니다. 본 프레임워크는 능동적 및 수동적 제어 시스템을 결합하여 로봇의 자율 동작의 다양성과 신뢰성을 향상시킵니다. 제안된 프레임워크는 독립된 신경망 컨트롤러를 포함하며, 다양한 복잡한 환경에서 로봇의 장애물 넘기, 부분 손상 후 빠른 회복 등의 시뮬레이션을 통해 검증되었습니다.'}]

- **Performance Highlights**: [{'title': 'MCQ 자동 생성 평가 메트릭 성능 검증', 'details': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계가 있음을 보여주었습니다. n-gram 기반의 유사성 메트릭과 결합한 경우, MCQ 품질 측정에 대한 예측력이 높았습니다.'}, {'title': '대조적 학습 및 반사실적 증강 성능 검증', 'details': '본 접근 방법은 대조적 내구성, 도메인 간 일반화, 희소 데이터 상황에서의 일반화 성능 등 다양한 차원에서 의미 있는 개선을 이루었습니다.'}, {'title': '계층적 학습 제어 프레임워크 성능 검증', 'details': '제안된 제어 프레임워크는 다양한 복잡한 환경에서 신속한 적응 능력을 보여주었으며, 센서 정보의 의존도를 낮추고 높은 자율성을 보였습니다. 로봇의 빠른 회복 효과와 다양한 상황에 대한 적응력도 입증되었습니다.'}]



### RepoMasterEval: Evaluating Code Completion via Real-World Repositories (https://arxiv.org/abs/2408.03519)
- **What's New**: 자동 다중선택 질문(MCQ) 생성과 코드 자동완성 도구의 성능 평가를 다룬 새로운 연구들이 등장했습니다. 특히 MCQ 생성의 교육적 가치를 측정하는 새로운 평가 메트릭, Knowledge Dependent Answerability(KDA) 도입과 현실적인 코드 완성 도구 평가를 위한 RepoMasterEval 벤치마크가 제안되었습니다.

- **Technical Details**: MCQ 자동 생성을 평가하기 위해 BLEU, ROUGE, METEOR와 같은 기존 메트릭의 한계를 극복하는 KDA 메트릭은 학생이 특정 사실(knowledge of the target fact)을 알고 있을 때 질문에 답할 수 있는지를 평가합니다. 또한, 신경망을 통한 학생 반응을 모방하여 자동 평가 메트릭, KDA_disc와 KDA_cont가 제시되었습니다. 한편, 코드 자동완성 도구의 현실적 적용성을 평가하기 위해 Python과 TypeScript 저장소(real-world repositories)로부터 생성된 새 벤치마크 RepoMasterEval이 제안되었습니다. BM25 알고리즘을 통해 의미 있는 컨텍스트를 추출하여 모델의 메스킹된 코드 조각을 예측하도록 합니다. 또한, 변이 테스트(mutation testing)와 수작업 테스트 케이스를 통해 테스트 정확성을 보장합니다.

- **Performance Highlights**: MCQ 생성 메트릭 KDA_disc와 KDA_cont는 실제 강의 사용성과 높은 상관 관계를 보였으며, BLEU 등 기존 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 측정에서 높은 예측력을 보였습니다. RepoMasterEval 벤치마크를 통해 현실적인 코드 완성 작업에 대한 모델의 성능 변동성을 효과적으로 평가할 수 있음을 확인했습니다. GPT-4가 기존 벤치마크 HumanEval에서는 높은 점수를 기록했으나, RepoMasterEval에서는 낮은 성능을 보였으며, 이는 RepoMasterEval이 실제 개발 환경의 복잡성을 잘 반영하고 있음을 보여줍니다.



### A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems (https://arxiv.org/abs/2408.03515)
- **What's New**: 본 연구 논문은 자동으로 Multiple Choice Questions (MCQ)을 생성하는 데 있어 기존 평가 메트릭들의 한계를 극복하기 위해 지식 종속 가능성(KDA)라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 대상 사실에 대한 학생의 지식을 평가하는 능력과 대답 가능성을 측정합니다.

- **Technical Details**: 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 데이터셋의 골드 샘플과의 n-그램 유사성에 초점을 맞추어 교육적 가치를 무시하는 문제점이 있었습니다. 본 연구는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 갖는다는 것을 입증했습니다. 이 메트릭들은 n-그램 기반 유사성 메트릭과 결합되었을 때 다양한 전문가 레이블 MCQ 품질 측정 기준에 대한 강력한 예측력을 갖고 있음을 보여주었습니다.



### Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation (https://arxiv.org/abs/2408.03505)
- **What's New**: 최근 다중 선택 질문 (Multiple Choice Questions, MCQ)의 자동 생성 기술이 교육자들의 평가 시간을 절감할 수 있는 잠재력을 가지고 있지만, 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 간과하고 있습니다. 이에 따라 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안했습니다. 이는 학생이 해당 사실(Target Fact)에 대한 지식을 바탕으로 문제에 답할 수 있는지를 측정합니다.

- **Technical Details**: KDA는 인간 설문조사를 통해 학생 응답 기반으로 측정되며, 이를 자동으로 평가하기 위해 pre-trained 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안했습니다. Human studies를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 라벨링한 실제 교실 사용성과 강한 상관관계가 있음을 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에 대해 강력한 예측력을 보여주었습니다.



### Advanced User Credit Risk Prediction Model using LightGBM, XGBoost and Tabnet with SMOTEENN (https://arxiv.org/abs/2408.03497)
Comments:
          8 pagess on IEEE ICPICS

- **What's New**: 새로운 MCQ 생성 평가 메트릭, 지식 종속 답변 가능성(KDA)을 제안합니다. 이 메트릭은 학생의 지식 레벨에 기반하여 MCQ의 답변 가능성을 측정하는데 중점을 둡니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 메트릭이 아닌, KDA_disc와 KDA_cont와 같은 새로운 자동 평가 메트릭을 사용합니다. 이는 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하도록 설계되었습니다.

- **Performance Highlights**: Human evaluation 결과, KDA_disc와 KDA_cont는 강의실 실사용성과 강한 상관관계를 나타내었으며, 이는 교육적 조건을 더 잘 반영할 수 있음을 보여주었습니다.



### Harnessing the Power of LLMs in Source Code Vulnerability Detection (https://arxiv.org/abs/2408.03489)
- **What's New**: 이번 연구에서는 기존 MCQ(Multiple Choice Questions) 생성 평가 메트릭의 한계를 극복하고자 '지식 종속 대답 가능성(Knowledge Dependent Answerability, KDA)'이라는 새로운 자동 평가 메트릭을 제안합니다. KDA는 생성된 MCQ가 대상 사실에 대한 학생들의 지식을 평가할 수 있는지 여부를 측정합니다.

- **Technical Details**: KDA 측정을 위해, 우선 인간 설문에서 학생들의 응답을 바탕으로 KDA를 측정합니다. 이후, 사전 훈련된 언어 모델(pre-trained language models)을 활용해 학생들의 문제 해결 행동을 모방함으로써 KDA를 근사화하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 실험을 통해 KDA_disc와 KDA_cont가 전문가에 의해 라벨링된 실제 교실 환경에서 사용성과 강한 상관관계를 나타내는 것을 확인하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 BLEU, ROUGE, METEOR 등의 기존 n-gram 기반 유사도 메트릭과 결합할 때, 다양한 전문가 라벨링 MCQ 품질 측정 지표에 대해 강력한 예측 능력을 보였습니다.



### Can LLMs Serve As Time Series Anomaly Detectors? (https://arxiv.org/abs/2408.03475)
- **What's New**: 이번 뉴스레터에서는 최신 연구 동향을 바탕으로, 자동 MCQ 생성, NLP 모델의 강건성 향상, 및 대형 언어 모델(LLM)을 이용한 시계열 이상 탐지에 관한 새로운 접근법을 소개합니다.

- **Technical Details**: [{'MCQ Generation': '기존 평가 메트릭인 BLEU, ROUGE, METEOR는 교육적 가치를 무시하고 있다는 문제를 해결하기 위해, Knowledge Dependent Answerability (KDA) 메트릭을 제안했습니다. 이 메트릭은 학생이 해당 사실에 대한 지식을 가지고 있는지 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont를 통해 학생의 문제 해결 행동을 모방하여 자동 평가를 진행할 수 있습니다.'}, {'Robust NLP Models': "최근 딥 러닝 모델이 NLP 작업에서 높은 정확성을 보였으나 spurious pattern에 의존하는 문제가 있습니다. 기존의 counterfactual augmentation은 spurious correlation에 의해 영향을 받는다는 한계를 가지고 있습니다. 새로운 접근법은 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해, 각 단어의 인과관계를 보다 강건하게 감독하는 방식을 제안합니다."}, {'Time Series Anomaly Detection using LLMs': '대형 언어 모델 (예: GPT-4, LLaMA3)을 이용한 시계열 이상 탐지가 주로 다루어졌습니다. LLM은 직접적인 시계열 이상 탐지에 사용될 수 없음을 발견했으며, 대신 in-context learning 및 chain-of-thought 구문을 사용한 prompt 전략을 통해 일부 이상 탐지에서 뛰어난 성능을 보였습니다. 또한, 설명 능력을 강화하기 위해 학습 데이터셋을 자동 생성하고 이를 통해 모델 성능을 향상시키는 방법을 제안했습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 인간 평가 및 실제 강의실 세트에서 사용성을 나타내는 전문가 레이블과 강한 상관관계를 가진다는 결과를 보여주었습니다. n-gram 기반 유사성 메트릭과 결합하면 예측력이 더욱 높아집니다.'}, {'Robust NLP Models': '우리 접근법은 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, generalization from scarce data 등 다양한 측면에서 유의미한 성능 향상을 달성했습니다.'}, {'Time Series Anomaly Detection using LLMs': 'LLM 기반 시계열 이상 탐지에서, GPT-4는 최소한의 프롬프트 지시로도 뛰어난 결과를 보여주었으며, LLaMA3 모델은 적절한 instruction fine-tuning을 통해 성능 향상을 보였습니다.'}]



### MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilib (https://arxiv.org/abs/2408.03468)
Comments:
          10 pages, 3 figures, ACM Multimedia 2024

- **What's New**: 이번 연구는 기존의 한계점을 극복하기 위해 Multiple Choice Questions (MCQ)의 자동 생성 평가 지표로 'Knowledge Dependent Answerability (KDA)'를 제안합니다. KDA는 학생이 해당 사실에 대한 지식을 바탕으로 MCQ를 얼마나 잘 답할 수 있는지를 측정하여 교육적 가치를 더 잘 평가할 수 있습니다.

- **Technical Details**: KDA는 인간 설문 조사를 통해 학생들의 응답을 기반으로 측정됩니다. 이를 확장하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하며, 이는 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. 이 연구는 KDA_disc와 KDA_cont가 실제 교실 환경에서 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다.

- **Performance Highlights**: 연구에서 제안한 KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 평가 지표와 결합될 때, 다양한 전문가가 라벨링한 MCQ 품질 측정 요소들에 대해 높은 예측력을 보였습니다.



### Identifying treatment response subgroups in observational time-to-event data (https://arxiv.org/abs/2408.03463)
- **What's New**: 이번 뉴스레터에서는 세 개의 최신 arXiv 논문을 소개합니다. 첫 번째 논문은 Multiple Choice Questions (MCQ) 생성 시 기존 평가 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 두 번째 논문은 NLP tasks에서 깊이 모델의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안합니다. 세 번째 논문은 관찰 연구에서 환자 하위 그룹을 식별하는 새로운 방법을 소개합니다.

- **Technical Details**: [{'title': 'MCQ 생성 평가 메트릭', 'content': '기존 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에 초점을 맞추며, 교육적 가치를 평가하지 못한다는 한계가 있습니다. 이를 해결하기 위해 KDA는 학생 응답을 기반으로 MCQ의 답변 가능성을 평가합니다. 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하며, pre-trained language models를 활용하여 학생들의 문제 해결 행동을 모사하여 KDA를 근사화합니다.'}, {'title': 'Robustness 향상을 위한 Model', 'content': '기존 방법들이 spurious patterns에 영향을 받는 문제를 해결하기 위해, 이 논문에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 이는 augmentation 기법이 spurious correlation에 의존하는 것을 방지합니다.'}, {'title': '관찰 연구에서 하위 그룹 식별', 'content': 'RCT 기반 방법론이 실제 환자 다양성을 반영하지 못하는 문제를 해결하기 위해, 이 논문에서는 일상적으로 수집되는 관찰 데이터를 활용하여 하위 그룹을 식별하는 새로운 방법을 제안합니다. Neural Survival Clustering을 확장하여 Causal Neural Survival Clustering (CNSC)을 도입하며, 이는 높은 차원의 비선형 조합을 통해 생존 분포와 치료 효과를 추정합니다.'}]

- **Performance Highlights**: [{'title': 'MCQ 생성 평가 메트릭', 'content': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실에서도 강한 상관관계를 가지고 있음을 입증했습니다. n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가가 라벨링한 MCQ 품질 지표에 대한 예측력도 높아집니다.'}, {'title': 'Robustness 향상을 위한 Model', 'content': '여러 차원에서 강력한 성능 향상을 보였습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization 입니다.'}, {'title': '관찰 연구에서 하위 그룹 식별', 'content': 'RCT와 관찰 치료 체제 모두에서 결과 기반 하위 그룹 분석의 최신 방법보다 성능이 크게 향상된다고 실험으로 증명했습니다.'}]



### EEGMobile: Enhancing Speed and Accuracy in EEG-Based Gaze Prediction with Advanced Mobile Architectures (https://arxiv.org/abs/2408.03449)
Comments:
          Accepted HCI International 2024 - Late Breaking Work

- **What's New**: 본 논문은 자동 기출문항(MCQ) 생성에서 BLEU, ROUGE, METEOR와 같은 기존 평가 메트릭들이 교육적 가치를 고려하지 않는 문제를 해결하고자 새로운 자동 평가 메트릭 Knowledge Dependent Answerability(KDA)를 제안했습니다. 또한 EEG 분석에 대해 MobileViT 네트워크와 Knowledge Distillation(KD)을 결합한 효율적인 모델을 제시했습니다. 마지막으로, NLP 모델의 robustness 문제를 해결하기 위해 대조 학습과 반사실적 증강을 제안했습니다.

- **Technical Details**: MCQ 생성에서는 KDA_disc와 KDA_cont 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. EEG 분석에서는 새로운 MobileViT 네트워크와 EEG 기반 transformer 모델인 EEGViT-TCNet과의 Knowledge Distillation을 통해 데이터 처리 효율을 높여 EEGEyeNet 절대 위치 태스크에서 SOTA와 비교할만한 정확도를 달성했습니다. NLP 반사실적 증강에서는 사람이 counterfactual을 추가하거나 데이터셋에서 이를 자동으로 찾아내는 기존 방법 대신 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 인과관계를 파악하는 방법을 사용했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합 시 다양한 전문가 레이블 MCQ 품질 측정에 강력한 예측 성능을 보였습니다. EEGMobile 모델은 이전 SOTA보다 3% 낮은 정확도를 보였으나, 데이터 처리 속도는 33% 빨랐고, 모델 크기는 60% 작았습니다. 반사실적 증강을 통한 NLP 모델은 counterfactual robustness, cross-domain generalization, 희소 데이터에서의 generalization 측면에서 유의미한 개선을 달성했습니다.



### Communication-Aware Consistent Edge Selection for Mobile Users and Autonomous Vehicles (https://arxiv.org/abs/2408.03435)
Comments:
          Accepted by Vehicular Technology Conference (VTC) Fall 2024

- **What's New**: 자동 MCQ 생성에서 교육적 가치를 평가할 수 있는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 소개합니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 메트릭과 달리 학생의 지식을 평가하는 능력을 측정합니다. 또한, counterfactual augmentation을 이용한 deep model의 robustness를 향상시키는 방법과 자율 주행 차량의 태스크 마이그레이션과 액세스 포인트 핸드오버를 위한 Deep Reinforcement Learning 기반의 프레임워크를 제안합니다.

- **Technical Details**: KDA는 학생 응답을 바탕으로 MCQ의 대답 가능성을 측정하고, pre-trained language models를 활용하여 자동 평가 메트릭 KDA_disc와 KDA_cont를 개발하였습니다. deep model의 robustness를 위해 여러 개의 counterfactual을 생성하고 collective decisions을 통해 스푸리어스 패턴에 민감하지 않은 모델을 제안합니다. 자율 주행 차량을 위한 프레임워크는 Deep Deterministic Policy Gradient (DDPG) 알고리즘을 기반으로 하며, 액세스 포인트의 통신 및 계산 자원을 공동 할당하여 서비스 지연 및 중단을 최소화합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트 사용성과 강한 상관관계를 가집니다. multi-dimensional tasks에서 counterfactual robustness, cross-domain generalization, scarce data generalization 모두에서 높은 성과를 보였습니다. 자율 주행 차량 프레임워크는 시뮬레이션을 통해 낮은 지연 시간과 최소 핸드오버를 달성하였습니다.



### Combining Diverse Information for Coordinated Action: Stochastic Bandit Algorithms for Heterogeneous Agents (https://arxiv.org/abs/2408.03405)
Comments:
          19 pages, 6 figures, to be published in ECAI 2024

- **What's New**: 새로운 MCQ 생성 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안하였습니다. KDA는 주요 사실에 대한 지식을 가진 경우 MCQ의 대답 가능성을 측정하며, 기존의 BLEU, ROUGE, METEOR와 달리 교육적 가치를 고려합니다.

- **Technical Details**: 이 논문은 pre-trained language models를 활용하여 학생들의 문제 해결 행태를 모방하는 KDA_disc와 KDA_cont를 제안합니다. Human survey 결과를 바탕으로 KDA를 측정하고, 이러한 결과와 강한 상관관계를 보이는지 평가하였습니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 인간 연구 결과에서 실제 교실 환경에서의 usability와 강한 상관관계를 나타냈습니다. n-gram 기반 유사성 메트릭과 결합할 때, 전문가가 라벨링한 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다.



### Attacks and Defenses for Generative Diffusion Models: A Comprehensive Survey (https://arxiv.org/abs/2408.03400)
- **What's New**: 이번 주에는 전통적인 자동 생성 다중 선택 질문(MCQ) 평가 메트릭의 한계를 해결하기 위한 혁신적인 접근 방식과 최신 딥 러닝 모델의 강인성을 높이는 새로운 기법, 그리고 Diffusion 모델(DM)의 보안 측면을 다룬 종합적인 조사 연구가 발표되었습니다.

- **Technical Details**: 전통적인 MCQ 평가 메트릭(BLEU, ROUGE, METEOR)은 단순히 생성된 MCQ와 골드 샘플 간의 n-그램 유사성에 초점을 맞추며 교육적 가치를 충분히 평가하지 못합니다. 이를 해결하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 메트릭이 제안되었습니다. KDA는 학생의 타겟 사실에 대한 지식을 바탕으로 대답 가능성을 측정합니다. 또 다른 연구에서는 기존 딥 러닝 모델의 spurious pattern에 의존하는 문제를 해결하기 위해 대조 학습과 counterfactual augmentation을 결합한 접근 방식을 제안했습니다. 마지막으로, Diffusion 모델(DM)의 보안 측면을 집중 조사한 연구는 DM의 다양한 공격 방법과 방어 기법을 포괄적으로 검토했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가와 실제 강의실 세트에서 높은 상관관계를 보여주었으며, n-그램 유사성 메트릭과 결합할 때 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다. counterfactual augmentation을 통한 집합적 의사 결정 방식은 다양한 측면에서 기존 모델 대비 강인성이 크게 향상되었습니다. Diffusion 모델의 보안 연구는 다양한 공격 방식을 분류하고, 여러 방어 수단을 제안하여 DM 기반 시스템의 보안을 강화하는 데 중요한 기여를 했습니다.



### RHiOTS: A Framework for Evaluating Hierarchical Time Series Forecasting Algorithms (https://arxiv.org/abs/2408.03399)
Comments:
          Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24), August 25--29, 2024, Barcelona, Spain

- **What's New**: 자동으로 Multiple Choice Questions (MCQ)를 생성하는 기술은 교사의 학생 평가 시간을 크게 단축할 수 있습니다. 하지만 기존 평가 메트릭들은 교육적 가치를 고려하지 않는다는 문제가 있습니다. 이를 해결하기 위해 우리가 새로운 평가 메트릭인 지식 종속 대답 가능성(KDA)을 제안합니다. 이 메트릭은 MCQ의 대답 가능성을 측정하며, 학생이 대상 사실에 대해 얼마나 알고 있는지를 평가하는 능력을 평가합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 한 human survey에서 측정합니다. 또한, 사전 훈련된 언어 모델을 활용하여 두 가지 자동화 메트릭(KDA_disc와 KDA_cont)을 제안하였으며, 이는 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다. Human studies를 통해 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강하게 상관관계가 있음을 입증했습니다. 또한, n-gram 기반의 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가 라벨링된 MCQ 품질 척도에 대해 강력한 예측력을 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅에서 높은 사용성을 보입니다. 특히, n-gram 기반 유사성 메트릭과 결합했을 때 전문가들이 라벨링한 MCQ의 품질 지표와 강한 상관관계를 나타냈습니다.



### A Non-negative VAE:the Generalized Gamma Belief Network (https://arxiv.org/abs/2408.03388)
- **What's New**: 새로운 논문에서는 Multiple Choice Questions (MCQ) 생성을 위한 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하고 있습니다. KDA는 학생들이 대상 사실(Target Fact)에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다. 기존 평가 메트릭들은 BLEU, ROUGE, METEOR와 같이 n-gram 유사성만 측정하며 교육적 가치를 고려하지 않으므로 새로운 접근 방식을 필요로 했습니다.

- **Technical Details**: KDA는 인간 설문 조사(human survey)를 통해 학생 응답 기반으로 측정되며, 이를 기반으로 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 사전 학습된 언어 모델(pre-trained language models)을 활용하여 학생들의 문제 해결 행동을 모방하는 방식으로 KDA를 근사화합니다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, 전문가들이 라벨링한 다양한 MCQ 품질 측정치와 결합할 때 예측력이 높다는 것을 보여주었습니다.



### Prioritize Alignment in Dataset Distillation (https://arxiv.org/abs/2408.03360)
Comments:
          18 pages, 9 figures

- **What's New**: 이번 주 AI 뉴스레터에서는 다양한 연구 주제를 다루고 있습니다. 첫 번째 연구는 자동 다중 선택 질문(MCQ) 생성의 새로운 평가 메트릭, 지식 종속 가능성(KDA)을 소개합니다. 두 번째 연구는 NLP 태스크의 '가짜 패턴' 의존 문제를 해결하기 위해 대비 학습과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다. 마지막 연구는 데이터세트 증류(Dataset Distillation)에서 정보 정렬 문제를 해결하기 위한 우선 정렬(PAD)을 제안합니다.

- **Technical Details**: {'MCQ Generation': '기존의 MCQ 생성 평가 메트릭 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에만 집중하지만, KDA는 대상 사실에 대한 학생의 지식 평가 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모사합니다.', 'NLP Model Robustness': '이 연구에서는 기존 방법이 일치하지 않는 상관관계(spurious correlation)에 의해 영향을 받는 반면, 여러 개의 반사실적(counterfactual)을 생성하고 이 집합의 분포를 분석하여 보다 견고하게 단어의 인과관계를 파악하는 방법을 제안합니다.', 'Dataset Distillation': '대상 데이터세트의 정보 추출과 임베딩(embed)에 오정렬된 정보가 소개되는 문제를 해결하기 위해 PAD 방식을 도입했습니다. 대상 샘플의 난이도를 측정하여 필요한 정보만 추출하고, 깊은 층의 파라미터만 사용하여 고품질 합성 샘플을 생성합니다.'}

- **Performance Highlights**: {'Knowledge Dependent Answerability (KDA)': '사람들이 응답한 설문조사 데이터와 강한 상관관계를 보였으며, 실제 교실 환경에서도 사용 가능성이 높음을 증명했습니다. KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합할 때 전문가가 라벨링한 MCQ 품질 측정 지표에 대해 높은 예측력을 나타냈습니다.', 'Counterfactual Augmentation': '제안된 방법은 반사실적 견고성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희소한 데이터로부터의 일반화에 있어 현저한 향상을 이뤄냈습니다.', 'Prioritize Alignment in Dataset Distillation (PAD)': '각종 벤치마크에서 최신 성능을 달성하며, 잘못 정렬된 정보를 걸러내어 기존 매칭 기반 증류 알고리즘에 큰 개선 효과를 주었습니다.'}



### LAMPO: Large Language Models as Preference Machines for Few-shot Ordinal Classification (https://arxiv.org/abs/2408.03359)
Comments:
          COLM 2024

- **What's New**: 자동 다중 선택 질문 (MCQ) 생성의 평가는 BLEU, ROUGE, METEOR와 같은 기존 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 우리가 제안한 새로운 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 메트릭을 소개합니다. KDA는 학생의 표적 사실에 대한 지식을 평가하는 능력을 측정하며, 주어진 지식으로 문제를 해결할 수 있는지를 평가합니다.

- **Technical Details**: KDA는 학생 응답을 바탕으로 측정을 시작하며, 사전 학습된 언어 모델을 통해 학생의 문제 해결 행위를 모방하여 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이를 통해 KDA와 실제 강의실 사용성 사이의 강한 상관관계를 증명했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블한 다양한 MCQ 품질 측정치에 대한 높은 예측력을 보였습니다. 이는 n-gram 기반 유사성 메트릭과 결합되었을 때 강력함이 더해져 전문가가 평가한 여러 품질 측정에서 높은 예측력을 입증했습니다.



### MLC-GCN: Multi-Level Generated Connectome Based GCN for AD Analysis (https://arxiv.org/abs/2408.03358)
- **What's New**: MCQ 생성의 교육적 가치를 높이기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안했습니다. 또한, spurious 패턴에 의존하는 문제를 해결하고자 contrastive learning과 counterfactual augmentation을 활용하여 NLP 모델의 강인성을 높이는 연구가 진행되었으며, AD(Alzheimer's Disease) 진단에서 최신 graph neural network (GNN)을 사용하는 MLC-GCN 모델을 제안했습니다.

- **Technical Details**: - MCQ 자동 생성: 기존의 n-gram 기반 평가 메트릭(BLEU, ROUGE, METEOR)의 한계를 극복하기 위해, KDA라는 메트릭이 제안되었습니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.
- NLP 모델의 강인성: contrastive learning과 counterfactual augmentation을 사용하여, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 인과관계를 파악합니다.
- AD 진단: MLC-GCN 모델은 다중 그래프 생성 블록과 GCN 예측 블록을 포함하며, 다양한 spatio-temporal feature extraction 레이어를 통해 다단계 connectome을 생성합니다.

- **Performance Highlights**: - KDA_disc와 KDA_cont는 전문가에 의해 라벨링된 MCQ 품질과 강한 상관관계를 가지며, n-gram 기반 메트릭과 결합할 때 예측력이 강화됩니다.
- contrastive learning을 통해 다양한 차원(포맷)에서 counterfactual 강인성, cross-domain 일반화, 그리고 적은 데이터에서의 일반화 성능이 향상됩니다.
- MLC-GCN은 ADNI와 OASIS-3 데이터셋에서 AD 진단 성능에서 최첨단 성능을 보여주었으며, 다층 생성 connectome을 통해 높은 해석 가능성을 보였습니다.



### The Use of Large Language Models (LLM) for Cyber Threat Intelligence (CTI) in Cybercrime Forums (https://arxiv.org/abs/2408.03354)
- **What's New**: 이번 주 AI 뉴스레터에서는 세 가지 주요 연구 논문을 다룹니다. 첫 번째는 자동 Multiple Choice Questions (MCQ) 생성의 평가 메트릭이 교육적 가치를 반영하도록 개선한 연구입니다. 두 번째 논문은 NLP 태스크에서 모델의 robustness를 높이기 위해 contrastive learning과 counterfactual augmentation을 어떻게 활용하는지 다룹니다. 세 번째 연구는 OpenAI의 GPT-3.5-turbo 모델을 사용하여 사이버 위협 정보(Cyber Threat Intelligence, CTI)를 정확하게 추출 및 요약하는 능력을 평가한 연구입니다.

- **Technical Details**: {'MCQ Generation': '현재 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR 등은 단어 유사도에 초점을 맞추어 MCQ의 교육적 가치를 평가하지 못합니다. 이를 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 평가 지표를 제안하였습니다. KDA는 학생이 특정 사실에 대한 지식을 가지고 문제에 답할 수 있는지 측정합니다. 우리는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.', 'Robustness in NLP': 'NLP 태스크에서 deep models는 spurious patterns에 의존하는 문제로 인해 robustness가 제한된다는 보고가 있습니다. 본 연구에서는 여러 개의 counterfactual을 생성하고, collective decisions를 통해 단어들의 인과관계를 파악하는 방법을 제안합니다. 이 방법은 spurious correlations에 덜 민감하여 counterfactual robustness, cross-domain generalization, 그리고 제한된 데이터에서도 성능 향상을 보여줍니다.', 'Cyber Threat Intelligence': 'OpenAI의 GPT-3.5-turbo 모델을 사용하여 사이버범죄 포럼의 대화를 분석하고 CTI 정보를 추출하는 시스템을 평가하였습니다. 3개의 포럼에서 500개의 대화를 무작위로 추출하여 요약하고 CTI 변수를 코딩하였습니다. 두 명의 분석가가 각 대화를 검토하여 LLM이 추출한 정보의 정확성을 평가하였습니다.'}

- **Performance Highlights**: {'MCQ Generation': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 높은 사용성과 강한 상관관계를 보여주었습니다. 또한, n-gram 기반 유사성 메트릭과 결합된 KDA_disc와 KDA_cont가 전문가가 라벨링한 다양한 MCQ 품질 측정에 강한 예측력을 가짐을 확인하였습니다.', 'Robustness in NLP': '제안된 방법은 attribution-based synthesis의 bias에 덜 민감하며, 다양한 측면에서 중요한 성능 향상을 달성하였습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) 제한된 데이터로부터의 generalization.', 'Cyber Threat Intelligence': 'LLM 시스템은 10개의 CTI 변수를 평균 98%의 정확도로 코딩하였으며, 특정 변수가 최대 100%의 정확도를 기록했습니다. 이는 LLM이 CTI 정보를 효율적으로 요약하고 분석할 수 있음을 시사합니다.'}



### Adversarial Domain Adaptation for Cross-user Activity Recognition Using Diffusion-based Noise-centred Learning (https://arxiv.org/abs/2408.03353)
- **What's New**: 이번 뉴스레터에서는 교육, NLP, 그리고 헬스케어 모니터링 분야에서 발표된 세 가지 최신 연구를 소개합니다. 각각은 자동 MCQ 생성, 대조 학습과 반사실 증강(Counterfactual Augmentation)을 활용한 NLP모델의 robust 성능 향상, 그리고 Cross-user Human Activity Recognition(HAR) 문제를 해결하기 위한 새로운 도메인 적응프레임워크를 다룹니다.

- **Technical Details**: [{'Research': '자동 MCQ 생성', 'Details': 'BLEU, ROUGE, METEOR과 같은 기존의 평가 메트릭은 교육적 가치를 고려하지 않고 n-gram 유사성에 주목합니다. 이를 해결하기 위해 우리는 지식 종속 가능성(KDA)이라고 불리는 새로운 자동 평가 메트릭을 제안했습니다. KDA는 특정 사실(Target Fact)에 대한 학습자의 지식을 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입하여 예측 성능을 높였습니다.'}, {'Research': 'NLP 모델의 Robust 성능 향상', 'Details': '최근 deep model들이 NLP 태스크에서 높은 정확성을 보였으나, spurious pattern에 의존하는 문제로 인해 robustness가 제한적이었습니다. 이를 해결하기 위해 대조 학습(Contrastive Learning)과 counterfactual augmentation를 사용한 새로운 접근 방식을 제안했습니다. 이 접근 방식은 여러 개의 counterfactual을 생성하고 집합적 의사 결정(Collective Decision)을 통해 단어들의 인과관계를 robust하게 평가합니다.'}, {'Research': 'Cross-user Human Activity Recognition (HAR)', 'Details': '기존 HAR 모델은 학습된 데이터와 실제 월드 데이터 사이의 분포 차이로 인해 성능이 제한됩니다. 이를 해결하기 위해 우리는 새로운 프레임워크인 Diffusion-based Noise-centered Adversarial Learning Domain Adaptation (Diff-Noise-Adv-DA)을 제안했습니다. 이 프레임워크는 노이즈를 활용하여 다양한 사용자 도메인 간의 robust한 분류 성능을 제공합니다.'}]

- **Performance Highlights**: [{'Research': '자동 MCQ 생성', 'Highlights': 'KDA_disc와 KDA_cont 자동 평가 메트릭이 실제 강의실 세팅에서 사용성과 강한 상관관계를 보였습니다. 특히 n-gram 기반 유사성 메트릭과 결합했을 때 높은 예측력을 보여주었습니다.'}, {'Research': 'NLP 모델의 Robust 성능 향상', 'Highlights': '여러 차원에서 significant improvements를 달성했습니다. 특히 counterfactual robustness, cross-domain generalization 및 스카없는 데이터에서의 generalization에서 notable 성능 향상을 보였습니다.'}, {'Research': 'Cross-user Human Activity Recognition (HAR)', 'Highlights': 'Diff-Noise-Adv-DA 프레임워크가 서로 다른 사용자 간의 HAR 성능을 향상시켰으며, 전통적인 도메인 적응 방법을 뛰어넘는 성능을 보였습니다. 노이즈 기반 de-noising 기술을 통해 데이터 품질도 향상되었습니다.'}]



### Artifical intelligence and inherent mathematical difficulty (https://arxiv.org/abs/2408.03345)
- **What's New**: 해당 논문은 수학에서 인공지능(AI)이 열린 질문 해답에 기여할 수 있는지 탐구한다. 전통적인 증명 발굴이 본질적으로 어려운 문제라는 한계를 제시하며, 최근 자동화된 정리 증명(Automated Theorem Proving), SAT 솔버(SAT-solvers), 대형 언어 모델(Large Language Models) 등 AI를 활용한 방법이 수학적 증명의 성질에 새로운 질문을 제기한다고 주장한다.

- **Technical Details**: 논문은 AI가 수학적 증명의 어려움을 실질적으로 해결할 수 없는 이유를 설명하며, 컴퓨팅 이론(Computability and Complexity Theory)의 제한적인 결과를 기반으로 증명의 어려움을 분석한다. 증명 발굴(Proof Discovery)이 본질적으로 어려운 문제라는 기존의 주장을 확장하여 인간과 AI 모두 동일한 근본적인 문제로 인해 어려움을 겪는다고 주장한다.

- **Performance Highlights**: 논문에서는 로빈스 문제(Robbins Problem), Boolean 피타고라스 삼중항 추론(Boolean Pythagorean Triple Conjecture), 캡 집합 문제(Cap Sets) 등의 예시를 통해 AI 기법이 실제로 열려있는 수학적 질문에 기여했음을 보여준다. 그러나 이러한 기법들이 고도의 논리적 복잡성을 지닌 문제를 해결하는 데는 제한적이라는 점도 강조한다. 결론적으로, AI 기법이 증명 발굴을 자동화하는 데 있어 본질적인 한계를 갖고 있음을 주장한다.



### The Ontoverse: Democratising Access to Knowledge Graph-based Data Through a Cartographic Interfac (https://arxiv.org/abs/2408.03339)
- **What's New**: 이번에 발표된 연구는 교육자들이 학생 평가에 소요되는 시간을 절감하고자 MCQ(객관식 문제) 자동 생성 방법을 탐구합니다. 기존 BLEU, ROUGE, METEOR 평가 메트릭은 교육적 가치를 고려하지 못하였기에, 이들의 대안으로 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭을 새롭게 제안했습니다.

- **Technical Details**: KDA 메트릭은 학생이 특정 지식을 가지고 문제를 얼마나 잘 풀 수 있는지를 평가합니다. 이를 위해 학생 응답 데이터를 사용하여 KDA를 측정하는 방법을 보여주고, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제시합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 높은 상관관계를 가짐을 확인했습니다. 또한, 이 메트릭들은 n-그램 기반의 유사성 메트릭과 결합될 때, 다양한 전문가 레이블의 MCQ 품질 측정에서 강한 예측력을 나타냈습니다.



### PsyDI: Towards a Personalized and Progressively In-depth Chatbot for Psychological Measurements (https://arxiv.org/abs/2408.03337)
Comments:
          28 pages, 16 figures

- **Multiple Choice Question Generation**: [{"What's New": '기존의 BLEU, ROUGE, METEOR와 같은 MCQ 평가 메트릭은 교육적 가치를 반영하지 못하는 문제점을 해결하기 위해 Knowledge Dependent Answerability(KDA)라는 새로운 자동 평가 메트릭을 제안하였습니다.'}, {'Technical Details': 'KDA는 MCQ의 대답 가능성(answerability)을 평가합니다. 이를 위해 우리는 인간 설문조사를 통해 학생 응답을 바탕으로 KDA를 측정하고, pre-trained language models을 활용하여 학생들의 문제 해결 행동을 모방한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였습니다.'}, {'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. n-gram 기반 유사성 메트릭과 결합할 경우, KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대해 강력한 예측 능력을 보였습니다.'}]

- **NLP Robustness through Counterfactual Augmentation**: [{"What's New": '대조학습과 반사실적 증강 (counterfactual augmentation)을 활용하여 NLP 모델의 견고성을 향상시키기 위한 새로운 접근법을 제안하였습니다.'}, {'Technical Details': '기존 방법과 달리, 우리의 접근법은 여러 개의 반사실적 예제를 생성하고, 이러한 집합에 대한 예측 분포를 기반으로 집합적 의사 결정을 통해 각 단어의 인과성을 보다 견고하게 감독합니다.'}, {'Performance Highlights': '이 접근법은 방대한 데이터를 사용하여, 반사실적 견고성, 도메인 간 일반화 및 희소 데이터에서의 일반화 성능에서 상당한 향상을 달성하였습니다.'}]

- **PsyDI: Interactive Psychological Assessment**: [{"What's New": '고정적인 전통적 심리 평가 방법과 달리, PsyDI는 다중 모드 및 사용자 맞춤형 심리 평가 챗봇으로 MBTI를 예시로 사용하여 새로운 심리 평가 프레임워크를 제안합니다.'}, {'Technical Details': 'PsyDI는 사용자의 여러 라운드 응답을 기반으로 맞춤형 상호작용을 통해 사용자의 MBTI 유형을 식별합니다. LLMs을 훈련하여 심리적 특성의 상대적 크기를 식별하는 혁신적인 훈련 기법을 도입하였습니다.'}, {'Performance Highlights': '3,000명의 참여자 데이터를 기반으로 한 실험과 분석에서 PsyDI의 유효성과 지속적인 정제 능력을 입증하였습니다. 또한 Emotion Analysis 시나리오로 확장하여 프레임워크의 전이 가능성을 검증하였습니다.'}]



### Explainable AI-based Intrusion Detection System for Industry 5.0: An Overview of the Literature, associated Challenges, the existing Solutions, and Potential Research Directions (https://arxiv.org/abs/2408.03335)
Comments:
          57 pages, 6 figures

- **1**: {"What's New": '자동 MCQ 생성의 교육적 평가를 위한 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안했습니다. 이 메트릭은 MCQ가 학생의 대상 사실 이해를 평가하는 능력을 측정합니다.', 'Technical Details': {'KDA Measurement': '학생 응답을 기반으로 한 인간 설문을 통해 KDA를 측정하는 방법을 제시합니다.', 'Automatic Evaluation Metrics': '사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방, KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안합니다.'}, 'Performance Highlights': 'KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 기준에 대해 예측력이 뛰어납니다.'}

- **2**: {"What's New": '대조 학습(contrastive learning)과 반사실적 보강(counterfactual augmentation)을 이용해 NLP 모델의 robust성을 높이는 방법을 연구했습니다.', 'Technical Details': {'Contrastive Learning': '반사실적 보강을 통해 단어들의 인과관계를 robust하게 측정합니다.', 'Collective Decision Making': '여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 모델 편향 문제를 줄입니다.'}, 'Performance Highlights': '우리의 접근 방식은 다차원적으로 반사실적 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 희소한 데이터로부터의 일반화(generalization from scarce data)에서 유의미한 성과를 나타냈습니다.'}

- **3**: {"What's New": 'Industry 5.0에서 사용될 수 있는 여러 XAI 기반 침입 탐지 시스템을 종합적으로 조사하고, explainability와 interpretability가 사이버 보안 관행에 미치는 영향을 검토했습니다.', 'Technical Details': {'XAI Adoption': 'DARPA 지원 프로그램을 포함해 Industry 5.0의 높은 표준성과 AI의 연결성을 위해 투명하고 신뢰할 수 있는 기계 학습 모델 개발을 목표로 한 XAI 연구를 소개합니다.', 'Survey Focus': '다양한 XAI 기반 메커니즘을 사용해 침입 탐지 시스템을 꾸준히 개선하고, 이러한 XAI 개념들이 어떻게 허점을 만들 수 있는지 분석했습니다.'}, 'Performance Highlights': 'XAI 기반 사이버 보안 시스템의 연구와 실험을 통해, 스마트 산업 환경에서의 보안 전략이 강화되고, explainability와 interpretability의 중요성이 강조되었습니다.'}



### Coverage-aware and Reinforcement Learning Using Multi-agent Approach for HD Map QoS in a Realistic Environmen (https://arxiv.org/abs/2408.03329)
- **What's New**: 자동 다중 선택 질문 (MCQ) 생성의 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, Deep Learning 모델들이 spurious pattern에 의존하여 robustness 문제가 있음을 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용하는 방법을 연구하였습니다. 그리고 VANET (Vehicular Adhoc Network)에서 HD map 데이터 전송을 위해 Q-Learning 알고리즘을 이용한 효율적인 전달 방식도 제안되었습니다.

- **Technical Details**: MCQ 평가를 위한 KDA는 학생의 지식을 반영하여 문제의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였으며, 이들은 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모사합니다. Deep Learning 모델의 robustness 문제는 다양한 counterfactual을 생성하고 집합적 의사 결정을 통해 각 용어의 인과관계를 파악하는 방법으로 해결됩니다. VANET의 경우, 표준 IEEE802.11p의 고정 CW 사이즈로 인한 문제를 해결하기 위해 어플리케이션 계층에서 작동하는 Q-Learning 알고리즘을 통해 네트워크 성능을 최적화하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 강한 상관관계를 보여주었으며, 전문가가 라벨링한 MCQ 품질 측정치에 강력한 예측력을 보였습니다. 여러 개의 counterfactual을 생성하여 collective decision을 통해 Deep Learning 모델의 robustness를 개선하였으며, 다양한 차원에서 성능이 향상되었습니다. VANET에 적용된 Q-Learning 기반 솔루션은 DQN 및 Actor-Critic 알고리즘에 비해 적은 최적화 요구사항으로 더 나은 네트워크 성능을 보였습니다.



### Reconstruction of the shape of irregular rough particles from their interferometric images using a convolutional neural network (https://arxiv.org/abs/2408.03327)
- **What's New**: 이 논문들에서는 새롭고 혁신적인 방법들을 소개하고 있습니다: 
1. 자동 MCQ 생성의 교육적 가치를 평가하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다.
2. NLP 태스크에서 모델의 robustness을 높이기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 논의합니다.
3. 비정형 입자의 3D 모양을 재구성하기 위해 CNN 기반 모델을 개발했습니다.

- **Technical Details**: 논문들은 다음과 같은 기술적인 세부사항들을 다룹니다:
1. KDA 메트릭은 학생들의 응답을 기반으로 측정하며, 인간 연구를 통해 KDA_disc와 KDA_cont의 유용성을 확인합니다. 모델은 사전 학습된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방합니다.
2. 새로운 augmentation 방법은 여러개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust한 모델을 구현합니다. 기존 방법들이 spurious correlation에 취약하다는 문제를 해결합니다.
3. CNN은 UNET 아키텍처와 residual block 모듈을 기반으로 하며, 실험적 패턴 데이터를 활용하여 학습되었습니다. CNN은 다양한 비정형 입자의 3D 재구성을 성공적으로 수행합니다.

- **Performance Highlights**: 논문들은 다음과 같은 성과를 보여줍니다:
1. KDA_disc와 KDA_cont는 KDA와 전문가가 라벨링한 실제 강의실 사용성과 강한 상관관계를 가집니다. n-gram 기반 메트릭과 결합 시 MCQ의 품질 측정에 높은 예측력을 보입니다.
2. 제안된 augmentation 방법은 기존의 attribution-based 방법보다 더 robust하며, 다양한 차원에서 성능 향상을 달성합니다. 예를 들어, counterfactual robustness, cross-domain generalization, 그리고 부족한 데이터로부터의 일반화에서 높은 성능을 보입니다.
3. CNN은 비대칭 및 비대칭 입자의 모양을 높은 정확도로 재구성하며, 18000개의 실험적 interferometric 이미지를 활용하여 학습되었습니다.



### Dynamic Language Group-Based MoE: Enhancing Code-Switching Speech Recognition with Hierarchical Routing (https://arxiv.org/abs/2407.18581)
- **What's New**: 최근 NLP 태스크에서 사람보다 뛰어난 정확도를 보이는 deep model에도 불구하고, 가짜 패턴(spurious pattern)에 의존하는 문제 때문에 강건성(robustness)이 제한된다고 보고되고 있습니다. 이에 따라 대비 학습(contrastive learning)과 counterfactual 증강(counterfactual augmentation)을 활용하여 강건성을 향상시키기 위한 새로운 방법론들이 제안되고 있습니다. 또한 다중 전문가(Mixture of Experts, MoE) 접근법을 활용한 다국어 및 코드전환(Code-Switching, CS) 태스크의 성능을 최적화하기 위한 새로운 모델 DLG-MoE가 제안되었습니다.

- **Technical Details**: 1. 지식 종속 대답 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 도입하여 학생의 문제 해결 행동을 모방하는 사전 훈련된 언어 모델을 활용했습니다. KDA_disc와 KDA_cont는 실제 인간 평가와 강한 상관관계를 보였습니다. 

2. 대비 학습과 counterfactual 증강을 통해 모델의 강건성을 높였습니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 신뢰성 있게 단어들의 인과관계를 파악하는 방법을 제안했습니다. 

3. DLG-MoE는 계층적인 라우팅 메커니즘을 사용하여 언어 기반 라우팅과 언어 그룹 내 비지도 학습(unsupervised) 라우팅을 결합하였습니다. 이 모델은 동적 top-k 추론과 스트리밍 인식 기능을 지원하며, 모노링구얼 서브 모델로 축소할 수 있습니다.

- **Performance Highlights**: 1. KDA 기반 평가 메트릭은 KDA_disc 및 KDA_cont와 결합될 때 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에 대해 높은 예측력을 보였습니다. 

2. 대비 학습과 counterfactual 증강을 통해 1) counterfactual 강건성, 2) 크로스 도메인 일반화 성능, 3) 데이터가 부족한 상황에서도 높은 일반화 성능을 달성했습니다. 

3. DLG-MoE는 최고 수준의 성능(state-of-the-art performance)을 달성했으며, 특히 코드전환 상황에서도 뛰어난 성능을 보였습니다. 또한 모델의 유연성과 확장성을 높였습니다.




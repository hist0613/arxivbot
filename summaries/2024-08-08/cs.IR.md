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




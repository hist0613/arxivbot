New uploads on arXiv(cs.CL)

### Prompt Recursive Search: A Living Framework with Adaptive Growth in LLM Auto-Prompting (https://arxiv.org/abs/2408.01423)
Comments:
          8 pages,4 figures

- **Multiple Choice Questions**: [{"What's New": '자동 MCQ 생성 평가 메트릭의 한계를 극복하기 위해 새로운 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안했습니다.', 'Technical Details': 'KDA는 학생들이 대상 사실에 대한 지식을 바탕으로 MCQ를 풀 수 있는지 평가합니다. KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안하여 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human evaluation 결과, KDA_disc와 KDA_cont는 실제 강의실 환경에서 높은 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 평가 메트릭과 결합할 경우, 다양한 전문가가 평가한 MCQ 품질 지표에 대해 높은 예측력을 가졌습니다.'}]

- **Deep Model Robustness**: [{"What's New": '최근 deep model들이 NLP 태스크에서 사람보다도 높은 정확성을 보이지만, spurious patterns 때문에 robustness가 제한되고 있습니다. 이를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안합니다.', 'Technical Details': '기존 방법들은 사람이 일일이 counterfactual을 추가하거나 모델이 이미 있는 데이터에서 유사한 것들을 찾았으나 여전히 spurious correlations에 영향을 받았습니다. 이 논문에서는 여러 개의 counterfactual을 생성하고, collective decisions을 통해 더 강력한 단어 인과관계를 확보하는 방법을 제안합니다.', 'Performance Highlights': '제안한 방법은 다양한 측면에서 기존 방법보다 우수한 결과를 보였습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization 개선.'}]

- **Prompt Recursive Search**: [{"What's New": '새로운 Prompt Recursive Search (PRS) 프레임워크를 개발하여 대형 언어 모델(LLM)에서 문제 해결 과정에서 토큰을 효과적으로 절약하고 문제 복잡성을 평가하여 오류 발생 가능성을 줄이는 방법을 제안합니다.', 'Technical Details': '이 프레임워크는 문제의 복잡성을 평가하고 조정 가능한 구조를 포함하여 오류 발생 가능성을 줄입니다.', 'Performance Highlights': 'PRS 프레임워크는 Llama3-7B 모델을 사용한 BBH 데이터셋에서 Chain of Thought (CoT) 방법보다 8% 높은 정확도를 달성하여 22%의 성능 향상을 보였습니다.'}]



### DebateQA: Evaluating Question Answering on Debatable Knowledg (https://arxiv.org/abs/2408.01419)
Comments:
          Dataset and scripts for evaluation are available at this https URL

- **What's New**: 최근 자동 Multiple Choice Questions(MCQ) 생성을 향상시키는 새로운 평가 메트릭 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었습니다. 기존 메트릭은 비슷한 단어 기반 유사성에만 초점을 두지만, 새롭게 제안된 KDA는 학습자의 문제 해결 능력을 평가합니다.

- **Technical Details**: KDA는 학습 자료와 학생의 응답을 통해 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 측정됩니다. 이들은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실에서의 사용성과 강한 상관관계를 가지며, 특히 n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강한 예측 능력을 보였습니다.



### Talk Less, Interact Better: Evaluating In-context Conversational Adaptation in Multimodal LLMs (https://arxiv.org/abs/2408.01417)
Comments:
          Accepted to COLM 2024

- **What's New**: [{'Title': '자동 MCQ 평가를 위한 지식 종속 가능성(KDA) 메트릭 제안', 'Summary': '기존의 BLEU, ROUGE, METEOR 평가 메트릭은 MCQ 생성의 교육적 가치를 반영하지 못합니다. 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)는 학생이 해당 사실에 대한 지식을 바탕으로 MCQ를 푸는 능력을 측정합니다. 인간 설문 기반의 평가를 통해 KDA와 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 높은 상관관계를 보인다는 것을 확인했습니다.'}, {'Title': '딥 모델의 대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Summary': '딥 모델이 NLP 과제에서 높은 정확성을 보였지만, 잘못된 패턴에 의존하여 로버스트니스가 제한됩니다. 비대칭 학습(contrastive learning)과 대안적 자극(counterfactual augmentation)을 적용하는 방법을 제안하며, 여러 counterfactual을 생성하여 집합적 의사 결정을 통해 단어의 인과관계를 파악합니다. 이 방법으로 다양한 차원에서 성능 향상을 이끌어 냅니다.'}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Summary': '대화가 진행됨에 따라 인간은 점점 더 효율적인 언어를 사용하게 됩니다. 이를 여러 state-of-the-art MLLMs (multimodal large language models)에서 테스트해 보니, 상대방의 효율적인 언어를 이해할 수는 있지만, 스스로 효율성을 증가시키는 능력은 보이지 않았습니다. GPT-4와 같은 모델에서만 특정 프롬프트를 통해 이 능력을 유도할 수 있었습니다.'}]

- **Technical Details**: [{'Title': '자동 MCQ 평가를 위한 KDA 메트릭', 'Details': 'KDA는 학생의 응답을 통해 MCQ의 지식 기반 대답 가능성을 평가합니다. KDA_disc와 KDA_cont는 KDA를 예측하기 위해 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한 메트릭입니다.'}, {'Title': '대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Details': "기존의 대안적 증강 방식을 개선하여 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 보다 로버스트하게 인과관계를 지도합니다."}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Details': 'ICCA(Automated framework to evaluate conversational adaptation)를 사용하여 MLLMs의 문맥별 행동을 평가했습니다. GPT-4와 같은 모델은 특정 조건에서만 효율성을 스스로 증가시킬 수 있음을 발견했습니다.'}]

- **Performance Highlights**: [{'Title': '자동 MCQ 평가를 위한 KDA 메트릭', 'Highlights': 'KDA_disc와 KDA_cont는 실제 교실 설정에서 사용자인 전문가들에 의해 높은 예측력을 지녔습니다. 이는 n-gram 기반 유사성 메트릭과 결합하여 다양한 MCQ 품질 척도에서 높은 예측력을 가집니다.'}, {'Title': '대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Highlights': '집합적 의사 결정 방식을 통해 counterfactual robustness, cross-domain generalization, 창의적인 데이터 학습 측면에서 성능을 크게 향상시켰습니다.'}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Highlights': '일부 MLLMs는 상대방의 효율적인 언어를 이해할 수 있지만, 대다수 모델은 자체적으로 언어 효율성을 증가시키는 능력이 부족합니다.'}]



### Improving Multilingual Neural Machine Translation by Utilizing Semantic and Linguistic Features (https://arxiv.org/abs/2408.01394)
Comments:
          Accepted by ACL2024 Findings

- **What's New**: 최신 연구들은 기존의 MCQ 생성 평가 메트릭의 한계를 극복하는 새로운 자동 평가 메트릭(Knowledge Dependent Answerability, KDA)을 제안하였습니다. 또한, NLP 태스크에서 모델의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 활용한 새로운 접근 방법을 도입했습니다. 최종적으로는 다중언어 번역(NMT)에서 semantic과 linguistic 특징을 활용하여 zero-shot translation 성능을 향상시키는 방법을 발표했습니다.

- **Technical Details**: MCQ 생성에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 학생의 지식 평가 능력을 측정합니다. NLP에서는 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 사용합니다. 다중언어 번역에서는 semantic과 linguistic 특징을 디코더와 엔코더에서 분리하여 zero-shot translation 성능을 향상시키는 방법을 고안했습니다.

- **Performance Highlights**: MCQ 평가에서는 KDA_disc와 KDA_cont가 기존 n-gram 기반 메트릭과 결합되어 높은 예측 정확도를 유지합니다. NLP 태스크에서 제안된 방법을 통해 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터로부터의 generalization에서 의미있는 성능 향상을 보였습니다. 다중언어 번역에서는 제안된 방법이 supervised translation에서 평균 0.18+ BLEU, zero-shot translation에서 평균 3.74+ BLEU 성능 향상을 달성했습니다.



### Coalitions of Large Language Models Increase the Robustness of AI Agents (https://arxiv.org/abs/2408.01380)
- **What's New**: 이 논문은 자동으로 생성되는 다지선다형 질문(MCQ)의 교육적 가치를 평가하기 위해 새로운 평가 메트릭, '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'을 제안했습니다. 이는 기존의 n-그램 기반의 BLEU, ROUGE, METEOR 등의 메트릭이 학생 평가 능력을 정확히 반영하지 못한다는 문제점을 해결합니다.

- **Technical Details**: KDA는 특정 사실에 대한 학생의 지식을 평가하는 능력을 바탕으로 MCQ의 답변 가능성을 측정합니다. 이를 위해 학생 응답 데이터를 기반으로 KDA를 측정하고, 사전 훈련된 언어 모델을 활용한 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안했습니다. 이 메트릭들은 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: 인간 연구를 통해, KDA_disc와 KDA_cont가 실제 강의실 환경에서 전문가들이 라벨링한 사용성과 강한 상관관계를 갖고 있음을 보여주었습니다. 또한, n-그램 기반 유사성 메트릭과 결합할 경우, KDA_disc와 KDA_cont는 여러 전문가가 라벨링한 MCQ 품질 지표에 대한 강력한 예측력을 가집니다.



### Transformers are Universal In-context Learners (https://arxiv.org/abs/2408.01367)
Comments:
          16 pages

- **What's New**: MCQ(다지선다형 질문) 자동 생성의 교육적 가치를 평가하면서, 기존 메트릭의 한계를 극복할 새로운 자동 평가 메트릭, KDA(Knowledge Dependent Answerability),를 제안했습니다. 또한, transformers가 매우 긴 문맥에 대한 표현력을 가지며 이를 바탕으로 분포 연속성을 다루는 수학적 모형을 제시했습니다.

- **Technical Details**: 첫 번째 논문은 MCQ의 대답 가능성을 측정하는 KDA라는 새로운 메트릭을 제시했습니다. KDA_disc와 KDA_cont라는 자동 평가 메트릭을 통해 student responses를 모방하고, 교육적 가치와의 상관관계를 입증했습니다. 두 번째 논문에서는 deep model의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 연구했습니다. 마지막으로 transformers의 경우, 매우 긴 문맥 안에서의 연속성을 기반으로 토큰 분포를 통해 in-context mapping이 가능하다는 것을 수학적으로 입증했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계가 있으며, 데이터셋의 gold sample을 기반으로 한 기존 메트릭보다 예측력이 뛰어납니다. 대조 학습과 반사실적 증강 방법은 다양한 차원에서 의미 있는 성능 향상을 보였습니다. 깊은 transformers는 임의의 토큰 길이에서도 표현력을 유지하며, 고정된 토큰 임베딩 차원과 head 개수를 통해 연속적인 in-context mappings를 예측할 수 있습니다.



### FANNO: Augmenting High-Quality Instruction Data with Open-Sourced LLMs Only (https://arxiv.org/abs/2408.01323)
- **What's New**: 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안하여, MCQ의 대답 가능성(answerability)을 측정하고, 학생의 대상 사실에 대한 지식 평가 능력을 확인함. 이를 통해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입하고 이에 따라 MCQ의 품질을 평가하는 새로운 접근 방식을 제시함.

- **Technical Details**: KDA는 학생들의 응답을 기반으로 평가되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사함. 이를 통해 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지는 것으로 나타남.

- **Performance Highlights**: Human evaluation에서 KDA_disc와 KDA_cont가 전문가가 레이블링한 MCQ 품질 측정에서 높은 예측력을 보이며, 기존 n-gram 기반 유사성 메트릭과 결합될 때 더욱 강력한 예측 능력을 보임.



### Reconsidering Token Embeddings with the Definitions for Pre-trained Language Models (https://arxiv.org/abs/2408.01308)
- **What's New**: 최근 발표된 논문에서는 다지선다형 질문(MCQ)의 자동 생성 및 평가 방식을 혁신적으로 향상시키는 새로운 방식을 소개했습니다. 추가로, 대조 학습(contrastive learning)과 반사실적 증대(counterfactual augmentation)를 활용해 NLP 모델의 강건성을 개선하고, 토큰 임베딩(token embeddings)의 분포 문제를 해결하는 새로운 방법들이 제안되었습니다.

- **Technical Details**: [{"What's New": 'MCQ의 자동 생성 평가를 위해 지식 종속 가능성(KDA)을 기반으로 한 새로운 자동 평가 메트릭을 제안합니다. 이는 기존의 BLEU, ROUGE, METEOR 메트릭들이 제공하는 N-그램 기반 유사성 평가와는 달리, 학생의 지식을 평가하는 능력을 더 정확하게 측정합니다.', 'Technical Details': 'KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되어, 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 이 메트릭들은 인간 조사 결과와 강한 상관관계를 보이며, N-그램 기반 유사성 메트릭과 결합 시, 여러 전문가 레이블로 평가된 MCQ 품질 측정 지표에 대해 높은 예측력을 가집니다.'}, {"What's New": '대조 학습과 반사실적 증대를 통해 NLP 모델의 강건성을 향상시키는 새로운 방법을 제안합니다. 이는 기존 방법들이 경험하고 있는 부당한 상관관계(spurious correlations)의 한계를 극복하기 위해 설계되었습니다.', 'Technical Details': '새로운 접근법은 여러 개의 반사실적(counterfactual) 사례 집합을 생성하고, 이 집합의 예측 분포에 대한 집합적 의사 결정을 통해 각 용어의 인과관계를 강건하게 감독할 수 있도록 합니다. 이 방법은 다양한 차원에서 유의미한 성능 개선을 보였습니다. 구체적으로는 반사실적 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 및 적은 데이터로부터의 일반화(generalization from scarce data)에서 성능 향상이 있었습니다.'}, {"What's New": '토큰 임베딩의 분포 문제를 해결하기 위해 DefinitionEMB라는 새로운 방법을 제안하여, 사전 학습 언어 모델(PLM)의 의미 관련 정보 손실 문제를 개선합니다.', 'Technical Details': 'DefinitionEMB는 Wiktionary에서 얻은 정의를 사용하여 PLM의 토큰 임베딩을 동형(isotropic)으로 분포하게 하고, 미세 조정(fine-tuning) 과정에서 기존의 강건성을 유지하도록 설계되었습니다. 이는 RoBERTa-base와 BART-large 모델에 적용해 실험한 결과, GLUE와 네 가지 텍스트 요약 데이터셋에서 성능이 개선되었음을 보여주었습니다.'}]

- **Performance Highlights**: 새로운 평가 메트릭인 KDA_disc, KDA_cont는 실제 교실 환경에서의 사용성, 반사실적 강건성, 교차 도메인 일반화, 적은 데이터로부터의 일반화 측면에서 강한 상관관계와 성능 향상을 보였습니다. DefinitionEMB 방법은 낮은 빈도수의 토큰에 대한 성능을 향상시키면서, 다양한 데이터셋에서 기존 모델들의 성능을 전반적으로 개선했습니다.



### Deep Learning based Visually Rich Document Content Understanding: A Survey (https://arxiv.org/abs/2408.01287)
Comments:
          Work in Progress

- **Multiple Choice Question (MCQ) Generation**: [{"What's New": '교사들의 학습 평가 시간을 줄이기 위해 MCQ 자동 생성 기술이 제안되었으나, 기존 평가 메트릭 BLEU, ROUGE, METEOR는 교육적 가치를 평가하지 못하고 있었습니다. 이에 대해, 우리는 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 MCQ의 대답 가능성을 측정합니다.'}, {'Technical Details': 'KDA는 학생들이 실험적으로 대답할 수 있는지를 측정하는 메트릭입니다. 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하며, 이는 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, {'Performance Highlights': 'Human studies를 통해, KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가집니다. 또한, n-gram 기반의 유사성 메트릭과 결합하면 전문가가 평가한 MCQ 품질 측정에서 높은 예측력을 가집니다.'}]

- **Robustness in NLP Models**: [{"What's New": '최근 깊은 학습 모델이 NLP 태스크에서 사람을 초월하는 정확성을 보임에도 불구하고, spurious pattern에 의존하여 robustness가 제한됩니다. 이를 해결하기 위해 대조적 학습 및 counterfactual augmentation을 활용하는 방식을 제안합니다.'}, {'Technical Details': '기존의 augmentation 방식은 사람이 직접 개입하거나 데이터셋에서 유사 counterfactual을 찾는 방식으로 이루어지지만, 여전히 spurious correlation의 영향을 받습니다. 본 방법은 여러 개의 counterfactual을 합성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방식으로 진행됩니다.'}, {'Performance Highlights': '우리의 접근 방식은 집합적 의사 결정을 통해 attribution-based synthesis의 태스크 모델 편향에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data로부터의 generalization 등 다양한 차원에서 유의미한 성능 향상을 이룹니다.'}]

- **Visually Rich Document Understanding (VRDU)**: [{"What's New": '학계, 금융, 의료, 마케팅 등 다양한 분야에서 시각적으로 풍부한 문서(VRD)의 정보 추출은 중요한 작업입니다. 기존 방식은 전문가의 지식을 기반으로 한 수작업이 많아 비용과 효율성 문제를 안고 있었으나, 깊은 학습의 등장으로 이를 혁신적으로 개선할 수 있게 되었습니다.'}, {'Technical Details': '최근 VRDU에서 깊은 학습 기반 프레임워크가 주목받고 있으며, 이는 비전, 텍스트, 레이아웃 등 다중 모달 정보를 활용하여 고도화된 문서 표현을 만듭니다. 여러 벤치마크 데이터셋과 다양한 기법들을 체계적으로 조사하고 분석하여, 각 모델의 강점과 한계, 적합한 대응 시나리오 등을 비교합니다.'}, {'Performance Highlights': '최근 발전한 LSTM, CNN 기반 모델, 피처 기반 접근법, 레이아웃 인식 사전 학습 프레임워크 등은 여러 다운스트림 태스크에서 state-of-the-art 성능을 달성하였습니다. 이 논문에서는 이러한 다양한 접근 방식을 비판적으로 검토하며, 앞으로의 연구 방향과 실무 적용에 대한 인사이트를 제공합니다.'}]



### The Mismeasure of Man and Models: Evaluating Allocational Harms in Large Language Models (https://arxiv.org/abs/2408.01285)
- **What's New**: 이번 발표에서 우리는 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 소개합니다. 이는 학생들이 특정 사실에 대한 지식을 평가할 수 있는 MCQ의 대답 가능성(Answerability)을 측정합니다. 또한, 여러 개의 counterfactual을 생성하는 집합적 의사 결정(Collective decisions)을 통해 NLP 모델의 robustness를 높이는 새로운 방법을 제안합니다. 마지막으로, Rank-Allocational-Based Bias Index (RABBI)를 도입하여 LLM 예측에서 발생할 수 있는 할당 편향을 평가합니다.

- **Technical Details**: [{'제목': '자동 MCQ 생성 평가를 위한 새로운 메트릭', '내용': '기존의 평가 메트릭(BLEU, ROUGE, METEOR)은 MCQ의 교육적 가치를 무시합니다. 이를 해결하기 위해 우리는 지식 종속 가능성(KDA) 메트릭을 제안합니다. KDA는 기준 사실에 대해 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 KDA를 자동으로 예측하며, 인간 평가를 통해 강의실에서의 사용성과 강한 상관관계를 확인했습니다.'}, {'제목': 'counterfactual augmentation을 통한 NLP 모델의 robustness 향상', '내용': '기존의 augment 방법은 spurious correlation에 영향을 받습니다. 우리는 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안합니다. 실험 결과, 이 접근법은 특별히 counterfactual robustness, cross-domain generalization, 그리고 희소한 데이터로부터의 generalization에서 상당한 개선을 보였습니다.'}, {'제목': '고위험 결정에 사용되는 LLM의 할당 편향 평가', '내용': '기존의 편향 측정은 예측 성능 격차에 중점을 두지만, 실제 결정 과정에서의 할당 결과와는 갭이 있습니다. 우리는 Rank-Allocational-Based Bias Index (RABBI)를 도입하여 이러한 할당 편향을 평가합니다. 이는 모델의 출력에서 파생된 점수를 사용하여 할당 결정 결과의 편향을 측정합니다. 실험을 통해 기존 편향 메트릭을 보완하여 RABBI가 할당 격차와 강한 상관 관계가 있음을 보였습니다.'}]

- **Performance Highlights**: [{'제목': 'MCQ 자동 평가', '내용': 'KDA_disc와 KDA_cont는 실제 강의실에서의 사용성에 대해 높은 예측력을 보여줍니다. n-gram 기반 유사성 메트릭과 결합하여, 여러 전문가가 라벨링한 MCQ 품질 측정치에서도 강한 예측력을 보였습니다.'}, {'제목': 'NLP 모델의 robustness', '내용': '여러 차원에서 중요한 개선을 달성했습니다. 특히 counterfactual robustness, cross-domain generalization, 그리고 희소한 데이터로부터의 generalization에서 실질적인 성과를 보였습니다.'}, {'제목': '할당 편향 평가', '내용': 'RABBI는 기존의 평균 성능 격차와 분포 차이에 의존하는 편향 메트릭보다 할당 격차를 더 신뢰성 있게 반영합니다. 이는 모델 선택 시 유용하게 사용될 수 있습니다.'}]



### RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework (https://arxiv.org/abs/2408.01262)
- **What's New**: 새로운 논문들이 자동 MCQ 생성 평가 개선, NLP 태스크의 robustness 향상, 그리고 Retrieval-Augmented Generation (RAG) 시스템의 성능 평가를 다룹니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 등의 메트릭이 교육적 가치를 평가하지 못하는 문제를 해결하기 위해, 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모사합니다.', 'NLP Robustness': "NLP 모델들의 robustness를 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용합니다. '여러 개의' counterfactual을 생성하여 더 강건한 인과 관계 파악을 제안합니다.", 'RAG System Evaluation': 'RAGEval은 다양한 시나리오에서 LLMs의 지식 사용 능력을 평가하기 위해 만들어졌습니다. 이 프레임워크는 완전성(Completeness), 환각(Hallucination), 무관성(Irrelevance)의 세 가지 지표를 제안하여 성능을 평가합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont가 전문가가 라벨링한 실제 강의실 세트에서의 사용성과 강한 상관관계를 보여주었습니다.', 'NLP Robustness': '새로운 접근법은 다양한 차원에서 1) counterfactual robustness, 2) cross-domain generalization, 3) 소량 데이터에서의 generalization을 개선했습니다.', 'RAG System Evaluation': 'RAGEval은 LLMs가 특정 지식 출처에 의존하지 않고 다양한 도메인에서 효과적으로 성능을 발휘할 수 있도록 평가할 수 있게 합니다.'}



### High-Throughput Phenotyping of Clinical Text Using Large Language Models (https://arxiv.org/abs/2408.01214)
Comments:
          Submitted to IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI), Houston TX

- **What's New**: 최근 deep model들이 NLP 태스크에서 사람보다 나은 정확성을 보였으나 spurious pattern에 의존하는 문제로 인해 robustness가 제한적이라는 문제를 해결하고자 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 새로운 접근법을 제안했습니다. 또한, 대형 언어 모델(LLMs)을 이용하여 자동적으로 고속 페노타이핑(high-throughput phenotyping)을 구현하는 연구도 발표되었습니다.

- **Technical Details**: ['MCQ 생성: 현재 사용되는 BLEU, ROUGE, METEOR 등의 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해, Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안했습니다. 이는 학생이 특정 사실을 알고 있을 때 MCQ에 답할 수 있는 가능성을 측정합니다.', '이 논문은 KDA를 기반으로 한 평가 방법을 학생들의 응답을 통해 측정한 후, 사전 훈련된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다.', '대조 학습과 counterfactual augmentation: 기존의 방법에서는 인간이나 기계가 counterfactual을 수집해야 하는 문제를 해결하고자 다수의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 보다 robust하게 파악하는 방법을 제안했습니다.', '고속 페노타이핑: GPT-4와 GPT-3.5-Turbo를 사용하여 OMIM 데이터베이스의 임상 요약에서 표현된 페노타입을 자동으로 식별, 분류 및 정규화하는 성능을 평가하였습니다. 특히 GPT-4가 높은 성능을 보였습니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, n-gram 기반의 유사성 메트릭과 결합할 때 다양한 MCQ 품질 측정 요소에 대한 높은 예측력을 보였습니다.', '대조 학습 기반 접근법으로 인해 counterfactual robustness, cross-domain generalization, 그리고 scarce data 상황에서의 일반화 능력에서 상당한 개선을 달성했습니다.', 'GPT-4는 페노타입 식별, 분류, 정규화에서 GPT-3.5-Turbo를 능가하며, 수동 어노테이터와의 일치 정도에서 높은 성과를 보였습니다. 특히, GPT-4의 광범위한 사전 학습 덕분에 추가적인 수동 어노테이션 데이터 없이도 높은 성능과 일반화 능력을 보여줬습니다.']



### Misinforming LLMs: vulnerabilities, challenges and opportunities (https://arxiv.org/abs/2408.01168)
- **What's New**: 이번 뉴스레터에서는 세 가지 흥미로운 논문을 소개합니다: 자동 MCQ 생성 평가 지표, NLP의 강건성 향상을 위한 대조 학습, 그리고 거대 언어 모델(LLM)의 신뢰성 문제입니다.

- **Technical Details**: [{'내용': '첫 번째 논문에서는 BLEU, ROUGE, METEOR와 같은 기존 지표가 교육적 가치를 평가하지 않는다는 점을 지적하고, 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭을 제안합니다. 이는 MCQ의 대답 가능성을 평가하고, KDA_disc와 KDA_cont라는 자동 평가 지표를 통해 사용성과 강한 상관관계를 보장합니다.'}, {'내용': '두 번째 논문에서는 NLP 작업에서 홍채 패턴에 의존한 강건성의 한계를 지적하고, 대조 학습과 반사실적 증강(counterfactual augmentation)을 활용하여 이 문제를 해결하고자 합니다. 여러 반사실적 예제를 생성하여 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다.'}, {'내용': '세 번째 논문은 거대 언어 모델의 내부 메커니즘에 대해 설명하면서, LLM의 신뢰성 문제를 논의합니다. 통계적 패턴에 의존하는 LLM의 한계를 극복하기 위해 생성형 트랜스포머 기반 모델을 사실 기반 및 논리 프로그래밍 언어와 결합하는 연구가 진행되고 있다고 소개합니다.'}]

- **Performance Highlights**: [{'내용': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실에서의 사용성과 매우 강한 상관관계를 보여, MCQ 품질 예측력이 높다는 것을 입증하였습니다.'}, {'내용': '제안된 대조 학습 기반 방법은 반사실적 강건성, 도메인 간 일반화, 적은 데이터로부터의 일반화 성능이 크게 향상되었음을 실증적으로 보여주었습니다.'}, {'내용': '사실 기반 정보와 지식 그래프를 사용한 LLM 성능의 개선 가능성, 생성형 코드 작성 등의 연구가 신뢰할 수 있는 LLM의 미래를 제시하고 있습니다.'}]



### DERA: Dense Entity Retrieval for Entity Alignment in Knowledge Graphs (https://arxiv.org/abs/2408.01154)
- **What's New**: 이번 연구는 다중 선택 문제 (MCQ) 자동 생성의 평가 메트릭을 개선하기 위해 지식 종속 대답 가능성 (Knowledge Dependent Answerability, KDA)을 제안하였습니다. 현재의 평가 메트릭은 교육적 가치를 고려하지 않으며, 단순히 생성된 MCQ와 데이터셋의 골드 샘플 간의 n-gram 유사도만을 측정합니다. KDA는 해당 목표 사실에 대한 학생의 지식 평가 능력을 강조하여 측정합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 자동화하기 위해, 사전 훈련된 언어 모델 (pre-trained language models)을 활용하여 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하였습니다. 이러한 메트릭은 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: Human evaluation을 통해, KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며 기존의 n-gram 기반 유사성 메트릭과 결합했을 때, 다양한 전문가가 평가한 MCQ 품질 측정치에 대해 높은 예측력을 가지고 있음을 확인했습니다.



### CFBench: A Comprehensive Constraints-Following Benchmark for LLMs (https://arxiv.org/abs/2408.01122)
Comments:
          15 pages, 10 figures

- **Multiple Choice Question Generation**: [{"What's New": '기존 BLEU, ROUGE, METEOR 메트릭이 교육적 가치와는 동떨어져 있는 점을 개선하기 위해, MCQ 생성의 평가를 위한 새로운 자동 평가 메트릭 Knowledge Dependent Answerability (KDA)를 제안했습니다.'}, {'Technical Details': 'KDA는 학생의 반응을 바탕으로 MCQ의 대답 가능성을 측정하며, 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 통해 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서 유용성과 강하게 상관관계가 있음을 입증하였으며, n-gram 기반의 유사성 메트릭과 결합하면 전문가가 라벨링한 MCQ 품질 측정에 대해 강력한 예측력을 가집니다.'}]

- **Robustness in NLP Models**: [{"What's New": '최근 심층 모델들이 높은 정확성을 보였음에도 불구하고, 불순한 패턴에 의존하여 강건성(robustness)이 제한되는 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다.'}, {'Technical Details': '여러 개의 반사실(counterfactual)을 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 robust하게 찾는 방법을 사용합니다.'}, {'Performance Highlights': '이 접근 방식은 반사실 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 부족한 데이터로부터의 일반화에서 상당한 개선을 달성했습니다.'}]

- **Constraint Following in LLMs**: [{"What's New": '대규모 LLMs의 자연어 명령의 이해 및 수행 능력을 평가하기 위해 CFBench라는 포괄적 제약 준수 벤치마크(benchmark)를 제안합니다.'}, {'Technical Details': 'CFBench는 200개 이상의 실제 시나리오와 50개 이상의 NLP 태스크를 포함하는 1,000개의 샘플을 특징으로 하며 제약 유형을 10개의 주요 카테고리와 25개 이상의 하위 카테고리로 분류하는 체계를 도입했습니다.'}, {'Performance Highlights': 'CFBench를 사용해 최신 LLM들을 평가한 결과, 여러 제약 준수 측면에서 상당한 개선 여지가 있으며, 성능 향상 전략을 추가적으로 탐구했습니다.'}]



### Task Prompt Vectors: Effective Initialization through Multi-Task Soft-Prompt Transfer (https://arxiv.org/abs/2408.01119)
- **multiple_choice_question_generation**: [{"What's New": '자동 생성된 다지선다형 질문(MCQ)의 교육적 가치를 평가하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다.', 'Technical Details': 'KDA는 목표 사실에 대한 학생의 지식을 바탕으로 MCQ의 답변 가능성을 측정합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하고, 이를 통해 인간 설문조사 기반 KDA를 근사화합니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 전문가가 표기한 실제 강의실에서의 사용성과 높은 상관관계를 보여주었으며, 다른 n-gram 기반의 유사성 메트릭과 결합했을 때 전문가 표기된 다양한 MCQ 품질 척도를 예측하는 데 강력한 능력을 보였습니다.'}]

- **counterfactual_augmentation_with_contrastive_learning**: [{"What's New": '대조 학습(contrastive learning)과 반사실 생성(counterfactual augmentation)을 활용하여 NLP 태스크의 모델 신뢰성을 향상시키는 방법을 제안합니다.', 'Technical Details': '사람이 직접 반사실을 추가하거나 기존 데이터셋에서 반사실 유사 항목을 찾는 대신, 여러 반사실 세트를 생성하고, 이 세트에 대한 예측 분포의 집합적 의사 결정을 통해 각 용어의 인과성을 감독합니다.', 'Performance Highlights': '제안된 방법은 반사실 견고성, 도메인 간 일반화(cross-domain generalization), 희소 데이터로부터의 일반화 등 다양한 차원에서 성능 개선을 달성하였습니다.'}]

- **prompt_tuning_for_large_language_models**: [{"What's New": 'Task Prompt Vectors라는 개념을 소개하여, 여러 태스크에서 분할된 대형 언어 모델(LLM)의 효과적인 프롬프트 조정 초기화를 가능하게 합니다.', 'Technical Details': 'Task Prompt Vectors는 다양한 태스크에 대해 미세 조정된 소프트 프롬프트의 가중치 변화를 요소별 차이로 나타낸 것입니다. 이 벡터들을 산술적으로 조합하여 다중 태스크 문제 해결을 강화할 수 있습니다.', 'Performance Highlights': '12개의 NLU 데이터셋에서 실험 결과, Task Prompt Vectors는 유사 태스크 간의 효율적인 교차 태스크 전이를 가능하게 하며, 특히 데이터가 적거나 없는 상황에서 기존 최첨단 성능을 뛰어넘는 성과를 보였습니다.'}]



### IAI Group at CheckThat! 2024: Transformer Models and Data Augmentation for Checkworthy Claim Detection (https://arxiv.org/abs/2408.01118)
Comments:
          Accepted to CLEF2024 CheckThat!

- **What's New**: 이번 연구는 자동 MCQ(객관식 질문) 생성을 위한 새로운 평가 메트릭을 제안합니다. 기존 BLEU, ROUGE, METEOR와 같은 평가 메트릭은 생성된 MCQ가 원본 데이터셋과 얼마나 유사한지를 평가하지만, 교육적 가치는 고려하지 않습니다. 본 연구는 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 메트릭을 소개하며, 이는 학생이 타겟 사실에 대한 지식을 바탕으로 MCQ를 답할 수 있는 능력을 평가합니다.

- **Technical Details**: KDA는 학생들의 응답을 기반으로 측정됩니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 이용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont의 성능과 실제 교실 세트에서의 사용성 사이에 강한 상관관계가 있다는 것을 입증했습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 측정에서 강한 예측력을 보였습니다.



### BioRAG: A RAG-LLM Framework for Biological Question Reasoning (https://arxiv.org/abs/2408.01107)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서는 생명 과학 연구를 위한 질문-응답 시스템을 향상시키기 위해 BioRAG라는 새로운 Retrieval-Augmented Generation(RAG) 프레임워크를 도입했습니다. 이 시스템은 2200만 개의 과학 논문을 분석하고 인덱싱하여 대규모 언어 모델(LLMs)을 기반으로 훈련되었습니다. BioRAG는 도메인별 지식 계층 구조와 최신 데이터를 통합하여 검색 및 응답 생성을 개선합니다.

- **Technical Details**: BioRAG 프레임워크는 먼저 방대한 생명 과학 논문을 파싱, 인덱싱 및 세분화하여 고품질 훈련 코퍼스를 생성합니다. 이 후 생명 과학 도메인에 특화된 임베딩 모델을 훈련시키며, 외부 정보 소스를 포함하여 최신 데이터를 가져오는 것이 특징입니다. 또한 쿼리 전처리, 검색기 실행 컴포넌트, 모델이 충분한 정보를 수집하는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과, BioRAG는 여러 생명 과학 질문 응답 작업에서 맞춤형 LLM, 검색 엔진과 결합된 LLM, 다른 과학적 RAG 프레임워크를 뛰어넘는 성능을 보였습니다. 구체적으로, 컬렉티브 디시전 (collective decisions)을 통한 합성 counterfactual 생성 시, 상위 성능을 보이며 다양한 차원에서 강력한향상을 달성했습니다.



### General-purpose Dataflow Model with Neuromorphic Primitives (https://arxiv.org/abs/2408.01090)
- **What's New**: 이 논문은 자동 MCQ 생성의 때 기존의 평가 메트릭이 교육적인 가치를 무시한다는 문제를 해결하기 위한 새로운 평가 메트릭, 지식 종속 가능성 (KDA)을 제안합니다. 또한, NLP 태스크에서 최근의 딥 모델들이 높은 정확성을 보였지만 spurious 패턴에 의존하여 robustness가 제한된 문제를 해결하기 위해 대조 학습 (contrastive learning)과 counterfactual augmentation를 활용하는 방법을 제안합니다. 마지막으로, neuromorphic computing의 효율성을 최대한 활용하기 위해 제어 로직을 neuromorphic 하드웨어에 맞춘 데이터 흐름 모델을 제안합니다.

- **Technical Details**: 첫번째 논문에서는 KDA_disc과 KDA_cont라는 자동 평가 메트릭을 제안하며, 이는 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 MCQ의 대답 가능성을 평가합니다. 두번째 논문에서는 대조 학습과 여러 개의 counterfactual을 생성하여 모델의 바이어스에 덜 민감한 집합적 의사 결정 (collective decisions)을 통해 인과 관계를 파악하는 방법을 제안합니다. 세번째 논문에서는 neuromorphic 하드웨어에 맞춘 'when'과 'where' primitives을 도입하여 제어 로직의 neuromorphic 호환성을 높인 데이터 흐름 모델을 소개합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서 사용 측면과 강한 상관관계를 가지며, KDA와 n-gram 기반 유사성 메트릭을 결합하면 다양한 전문가 평가 MCQ 품질 지표에 대한 예측력이 높아짐을 확인했습니다. 대조 학습 기반 방법은 counterfactual robustness와 cross-domain generalization, scarce data 기반 일반화에서 유의미한 개선을 보였습니다. neuromorphic 데이터 흐름 모델은 neuromorphic 하드웨어의 프로그래밍 가능성과 plasticity를 모두 충족시키면서 높은 성능을 제공합니다.



### Bridging Information Gaps in Dialogues With Grounded Exchanges Using Knowledge Graphs (https://arxiv.org/abs/2408.01088)
Comments:
          Accepted to SIGDIAL 2024

- **MCQ Generation Paper**: [{"What's New": "자동 MCQ 생성 (Multiple Choice Questions)이 교육적 가치를 고려하지 않는 기존 메트릭을 보완하기 위해 '지식 종속 가능성 (Knowledge Dependent Answerability, KDA)'이라는 새 평가 메트릭을 제안합니다."}, {'Technical Details': 'BLEU, ROUGE, METEOR와 같은 기존 메트릭이 n-gram 유사도에만 초점을 맞춘 반면, KDA는 학생이 목표 사실에 대한 지식을 기반으로 한 문제 해결 능력을 모사하는 사전 훈련된 언어 모델을 활용합니다.'}, {'Performance Highlights': 'KDA_disc 및 KDA_cont는 강의실 세팅에서의 사용성과 강한 상관관계를 가지며, 전문가들이 라벨링한 MCQ 품질 측정치에 대한 높은 예측력을 보였습니다.'}]

- **Robust NLP Models Paper**: [{"What's New": "NLU(자연어 이해) 분야에서 deep models의 '변수 의존성' 문제를 해결하기 위해 대비 학습과 반사실적 데이터 증강을 통한 robustness 강화 방법을 제안합니다."}, {'Technical Details': '기존 방식과 달리, 다양한 counterfactual들을 생성하고 집합적 의사 결정(collective decision)을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 사용합니다.'}, {'Performance Highlights': '제안한 방법은 1) 반사실적 robustness, 2) 교차 도메인 일반화, 3) 적은 데이터에서의 일반화 측면에서 좋은 향상을 보였습니다.'}]

- **Dialogue Systems Paper**: [{"What's New": '대화 시스템에서 대화적 공유 이해 (conversational grounding)를 개선하기 위해 대형 언어 모델 (Large Language Models)을 활용하는 BridgeKG라는 새로운 대화 코퍼스를 도입합니다.'}, {'Technical Details': '다섯 개의 지식 도메인에서 수집된 대화 데이터에 대해 grounding acts와 지식 그래프 구조 내에서 연관된 정보를 라벨링했습니다.'}, {'Performance Highlights': 'BridgeKG dataset을 기반으로 한 다양한 실험을 통해, GPT-3.5와 GPT-4를 포함한 여러 LLM들이 grounding act를 분류하고 지식을 식별하는 성능을 평가했습니다.'}]



### Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts (https://arxiv.org/abs/2408.01084)
- **What's New**: 이번 뉴스에서 소개하고자 하는 논문은 자동 MCQ 생성 평가 메트릭과 대조적 학습 및 역사실 증강을 통한 NLP 모델의 견고성 개선, 그리고 대형 언어모델(LLMs)를 통한 개방형 질문 응답 (open-domain QA) 향상에 관한 연구입니다.

- **Technical Details**: [{'title': '자동 MCQ 생성 평가 메트릭 개선', 'description': '기존 BLEU, ROUGE, METEOR 메트릭은 데이터셋의 골드 샘플과 단어 유사성만을 평가하여 교육적 가치를 간과했습니다. 이를 해결하기 위해 새로운 평가 메트릭인 지식 종속 대답 가능성(Knowledge Dependent Answerability, KDA)를 제안하였고, 사람이 응답한 결과를 바탕으로 KDA를 측정했습니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 이를 모방하여 평가를 진행하였습니다.'}, {'title': '대조적 학습 및 역사실 증강을 통한 NLP 모델의 견고성 개선', 'description': '최근의 deep model들이 NLP 작업에서 높은 정확성을 보였으나 spurious pattern에 취약함이 보고되었습니다. 이 논문에서는 대조적 학습과 여러 개의 반사실적 (counterfactual) 데이터를 생성해 학습하는 방법을 통해 더 견고한 모델을 제안했습니다. 이를 통해 다양한 측면에서 견고성을 향상시켰습니다.'}, {'title': '대형 언어모델 사용시 외부 콘텍스트를 활용한 성능 향상', 'description': 'LLMs가 개방형 질문 응답에서 높은 성능을 보일지라도 노이즈가 있는 콘텍스트에 취약하다는 단점을 극복하기 위해 adaptive contrastive decoding (ACD) 방법을 제안하였습니다. ACD는 노이즈가 있는 상황에서도 성능 저하 없이 더 강력한 성능을 보였습니다.'}]

- **Performance Highlights**: [{'title': '자동 MCQ 생성', 'description': 'KDA_disc와 KDA_cont 메트릭은 골드 샘플과의 유사성을 유지하면서도 MCQ의 교육적 평가 능력을 높이는 것으로 나타났습니다.'}, {'title': 'NLP 모델 견고성', 'description': '여러 반사실적 데이터를 생성하고 집합적 의사결정 (collective decisions)을 통해 기존 모델에 비해 대조적 견고성, 도메인 간 일반화, 그리고 희소 데이터에서의 일반화 능력이 향상되었습니다.'}, {'title': 'LLMs의 개방형 질문 응답', 'description': 'ACD 방법을 적용함으로써, 노이즈가 있는 콘텍스트에도 적응하여 더 강력한 성능을 입증하였습니다. 특히, gold 표준 콘텍스트가 보장되지 않는 설정에서도 성능이 뛰어났습니다.'}]



### Leveraging Large Language Models for Mobile App Review Feature Extraction (https://arxiv.org/abs/2408.01063)
Comments:
          46 pages, 8 tables, 11 figures

- **What's New**: 새로운 연구는 선택형 객관식 문제(MCQ)를 생성하는데 있어 기존 평가 메트릭의 한계를 극복하기 위해 새로운 자동 평가 척도인 Knowledge Dependant Answerability (KDA)를 제안했습니다. 또한, 최근 논문에서는 NLP 모델의 강인성을 향상시키기 위해 대비 학습과 반사 사실 증강을 활용하는 방안을 제안했습니다. 마지막으로, 모바일 앱 리뷰에서 효과적인 기능 추출을 위해 encoder-only 모델 기반의 큰 언어 모델 사용을 제안했습니다.

- **Technical Details**: KDA는 학생이 대상 사실(target fact)을 알고 있는지를 평가하는 능력을 측정합니다. 이를 위해 인간 설문을 통해 KDA를 측정하고, 사전 훈련된 언어 모델을 사용해 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다. 다른 연구에서는 반사 사실 증강을 통해 여러 개의 사실(Factual)을 생성하고, 집합적 의사 결정(Collective Decisions)을 통해 단어들의 인과 관계를 더 강력하게 감독하는 방법을 제시합니다. 모바일 앱 리뷰 분석에서는 Encoder-only 대형 언어 모델을 사용하여 기능 추출 작업을 토큰 분류(Token Classification)로 재정의하며, 사전 훈련을 확장하고 인스턴스 선택 메커니즘을 사용해 모델의 미세 조정을 최적화합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 교육 현장에서의 사용성과 높은 상관관계를 보였습니다. 반사 사실 증강 연구에서는 인과 관계에 대한 강인성이 향상되고, 다양한 측면에서의 일반화 성능이 개선되었습니다. 마지막으로, 모바일 앱 리뷰 분석에서는 확장된 사전 훈련과 인스턴스 선택 기법이 기능 추출의 정확성과 성능 효율성을 향상시켰습니다.



### QUDSELECT: Selective Decoding for Questions Under Discussion Parsing (https://arxiv.org/abs/2408.01046)
Comments:
          11 Pages, 5 figures

- **What's New**: 자동 다지선다형 질문(MCQ)의 생성 및 평가 메트릭에 관한 새로운 연구가 발표되었습니다. 또한, 새로운 QUDSelect 프레임워크가 도입되어 담화(parsing) 분석의 정확도를 높이고 있습니다.

- **Technical Details**: {'MCQ Generation': '기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 n-그램 기반의 유사성에만 집중하여 교육적 가치를 반영하지 못했습니다. 이 문제를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 이 KDA는 학생들이 특정 사실에 기반해 질문에 답할 수 있는 능력을 측정합니다.', 'QUD Parsing': 'QUDSelect는 기존의 파이프라인 방식을 탈피하여, 각 문장의 담화 구조를 보다 정교하게 분석할 수 있도록 설계되었습니다. 함께 학습된(anchor sentence와 해당 질문의 동시 예측) 모델을 사용해 문장 간의 관계를 더 정확히 파악합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'Human study를 통해 KDA_disc와 KDA_cont 메트릭이 실제 강의실 설정에서의 사용성과 강한 상관관계를 가진다고 입증되었습니다. 또한, n-그램 기반의 유사성 메트릭과 결합할 때, MCQ의 다양한 질 평가 기준에 대해 강력한 예측력을 보였습니다.', 'QUD Parsing': 'QUDSelect 프레임워크는 인간 평가에서 기존 최첨단 모델보다 평균 9% 더 나은 성능을 보였습니다. 자동 평가에서도 약 4%의 개선을 이루어냈습니다.'}



### UNER: A Unified Prediction Head for Named Entity Recognition in Visually-rich Documents (https://arxiv.org/abs/2408.01038)
Comments:
          accepted by ACM Multimedia 2024

- **What's New**: 이번 연구에서는 자동 Multiple-Choice Questions (MCQ) 생성을 위한 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안했습니다. KDA는 기존 BLEU, ROUGE, METEOR와 달리 학생의 지식을 평가하는 MCQ의 답변 가능성을 측정합니다.

- **Technical Details**: KDA는 인간 설문 조사를 통해 학생의 응답을 기반으로 측정됩니다. 이를 자동화하기 위해 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 보이며, 전문가가 라벨링한 MCQ 품질 측정에 대해 예측력이 뛰어남을 확인했습니다.



### Fairness in Large Language Models in Three Hour (https://arxiv.org/abs/2408.00992)
- **What's New**: 이번 뉴스레터에서는 세 가지 흥미로운 연구를 소개합니다. 첫 번째 연구는 다지문 선택 질문(MCQ)의 자동 생성 평가를 위해 기존의 BLEU, ROUGE, METEOR 메트릭이 아닌 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안합니다. 두 번째 연구는 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 내구성을 강화하는 방법을 제안합니다. 세 번째 연구는 큰 언어 모델(LLMs)의 공정성 문제를 체계적으로 다룬 튜토리얼을 제공합니다.

- **Technical Details**: 첫 번째 연구에서는 MCQ의 답변 가능성을 측정하는 새로운 자동 평가 메트릭 KDA를 제안합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 미리 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 연구는 인간이 제조한 반사실적(counterfactual) 데이터나 데이터 세트 내 유사 반사실적 데이터를 사용하는 기존 방법의 한계를 극복하기 위해 여러 개의 반사실적을 생성하여 집단 의사결정을 통해 모델의 인과성을 더욱 견고하게 감독하는 방법을 제안합니다. 세 번째 연구는 큰 언어 모델에서 공정성 문제를 해결하기 위해 다양한 전략과 알고리즘을 소개하며, 실제 사례를 통해 편향을 분석하고 이를 평가하기 위한 툴킷과 데이터세트를 제공합니다.

- **Performance Highlights**: 첫 번째 연구에서는 인간 조사로부터 KDA를 측정하며, KDA_disc와 KDA_cont가 실제 강의실 세트에서 매우 높은 상관관계를 지님을 보여주었습니다. 두 번째 연구는 반사실적 내구성, 교차 도메인 일반화, 드문 데이터에서의 일반화 등 다양한 차원에서 상당한 개선을 이루었습니다. 세 번째 연구는 LLM의 편향을 평가하고 개선하기 위한 최신 알고리즘 및 리소스를 종합적으로 검토하여 공정성을 높이는 데 기여했습니다.



### Cross-domain Named Entity Recognition via Graph Matching (https://arxiv.org/abs/2408.00981)
- **What's New**: 새로운 자동 평가 메트릭 지식 종속 가능성(KDA)을 제안하여 MCQ의 답변 가능성과 학생의 지식 평가 능력을 측정합니다.

- **Technical Details**: 'Knowledge Dependent Answerability(KDA)'는 학생의 문제 해결 행동을 모방하기 위해 사전 학습된 언어 모델을 활용하여 KDA_disc 및 KDA_cont 자동 평가 메트릭으로 구체화되었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강의실 세트에서의 사용성과 전문가가 레이블링한 품질 측정 항목과 강한 상관관계를 보였으며, n-gram 기반의 유사성 메트릭과 결합하면 다양한 전문가 레이블링된 MCQ 품질 측정 예측력이 향상되는 것을 보여주었습니다.



### Automatic Extraction of Relationships among Motivations, Emotions and Actions from Natural Language Texts (https://arxiv.org/abs/2408.00966)
- **What's New**: 기존의 자동 Multiple Choice Question (MCQ) 생성 평가 메트릭은 BLEU, ROUGE, METEOR와 같은 n-gram 기반의 유사성만을 평가하기 때문에 교육적 가치를 간과하고 있습니다. 이를 해결하기 위해, 우리는 MCQ의 대답 가능성(answerability)을 측정하고 학생의 지식을 평가할 수 있는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.

- **Technical Details**: KDA를 측정하기 위해, 처음에는 인간 설문 조사를 기반으로 학생 응답을 분석하는 방법을 사용합니다. 이후, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이러한 메트릭은 실제 교실 환경에서의 사용성과 전문가가 레이블링한 KDA와 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 n-gram 기반 유사성 메트릭과 결합했을 때, 다양한 전문가 레이블링 MCQ 품질 측정 지표에 대해 강력한 예측력을 보였습니다.



### PERSOMA: PERsonalized SOft ProMpt Adapter Architecture for Personalized Language Prompting (https://arxiv.org/abs/2408.00960)
- **New Methods for Automated MCQ Evaluation**: [{"What's New": '기존의 자동 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ가 골드 샘플과 얼마나 유사한지를 n-gram 기반으로 평가하였으나, 교육적 가치는 고려하지 않았다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였다.'}, {'Technical Details': 'KDA는 학생의 대상 사실에 대한 지식을 평가할 수 있는 MCQ의 대답 가능성(answerability)을 측정한다. 구체적으로, 인간 설문 조사를 통해 KDA를 측정하는 방법을 제시하고, 학생들의 문제 해결 행동을 모방하기 위해 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안한다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었다. 또한, n-gram 기반의 유사성 메트릭과 결합될 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강한 예측력을 나타낸다.'}]

- **Enhancing NLP Model Robustness with Counterfactuals**: [{"What's New": '최근 NLP 태스크에서 deep model들이 사람보다 나은 정확성을 보여주지만, spurious pattern에 의존해 robustness가 제한된다는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용해 robustness를 향상시키는 방법을 제안한다.'}, {'Technical Details': '기존 방법들은 사람이 counterfactual을 만들거나 모델이 데이터셋에서 유사한 counterfactual을 찾는 방식을 사용했으나, 우리는 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안한다.'}, {'Performance Highlights': '우리의 방법은 attribution-based synthesis의 task model bias에 덜 민감하여 1) counterfactual robustness, 2) cross-domain 일반화, 3) scarce data로부터의 일반화에서 현저한 개선을 이룬다.'}]

- **Personalized User Interaction with Large Language Models**: [{"What's New": '사용자 맞춤형 자연어 시스템 구축을 위해 PERSOMA, Personalized Soft Prompt Adapter 아키텍처를 제안한다. 이는 사용자 히스토리를 효율적으로 캡처하기 위한 혁신적인 접근법을 제공한다.'}, {'Technical Details': 'PERSOMA는 사용자의 히스토리를 자유형 텍스트로 재샘플링하고 이를 expressive soft prompt embeddings로 압축하여 사용자 특화된 soft prompt 어댑터를 구축한다. 이를 통해 LLM의 이해력을 손상시키지 않으면서도 사용자 히스토리를 이용해 출력 결과를 사용자에게 맞출 수 있다.'}, {'Performance Highlights': 'PaLM 2 모델을 이용한 실험에서 PERSOMA는 MovieLens 데이터셋에서 기존 embedding-based 기술보다 0.18의 F1 점수 개선을 이루었으며, 전체 finetuned text prompting baseline과 유사한 성능을 나타내었다.'}]



### Leveraging Large Language Models (LLMs) for Traffic Management at Urban Intersections: The Case of Mixed Traffic Scenarios (https://arxiv.org/abs/2408.00948)
- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하는 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안했습니다. 또한, 모델의 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용한 새로운 방법을 제시했습니다. 더불어, GPT-4o-mini를 사용하여 도시 교차로에서 실시간 교통 관리 성능을 향상시켜주는 연구 결과도 소개됩니다.

- **Technical Details**: 1. 새로운 자동 MCQ 평가 메트릭 KDA는 학생의 실제 반응을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 사용해 학생의 문제 해결 행위를 모방하였습니다. 
2. NLP 모델의 robustness를 높이기 위해 다수의 counterfactual을 생성하고 이를 통한 집합적 의사 결정을 통해 단어들의 인과관계를 더욱 견고하게 파악하는 방법을 제안했습니다.
3. GPT-4o-mini가 실시간으로 교차로의 위치 예측, conflict 감지 및 해결, 교통 혼잡도 분석을 통해 도시 교통 관리 체계를 개선할 수 있는 능력을 탐구하였습니다.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont는 인간 평가 기준과 높은 상관관계를 보여, 실제 강의실 환경에서의 사용성과 예측력을 입증했습니다. 
2. 제안된 방법은 기존 counterfactual augmentation보다 다양한 차원에서 더 나은 성능을 보였으며, counterfactual robustness, cross-domain generalization, 적은 데이터로부터의 일반화를 달성했습니다. 
3. GPT-4o-mini는 복잡한 도시 교차로 시나리오에서 교통 혼잡 및 충돌 관리를 효과적으로 수행하며, 더 안전하고 효율적인 교차로 관리에 기여할 수 있음을 보였습니다.



### UniMoT: Unified Molecule-Text Language Model with Discrete Token Representation (https://arxiv.org/abs/2408.00863)
- **What's New**: 최근 연구들은 MCQ(선다형 시험 문제) 자동 생성을 위한 평가 기준이 교육적 가치 대신 단어 유사성에 집중하고 있다는 문제를 제기하며, 이를 해결하기 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, NLP 태스크에서 deep model의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 탐구했습니다. 그리고 다양한 태스크에서 뛰어난 성능을 보이는 LLMs를 화학 분야로 확장하기 위한 UniMoT라는 새로운 모델도 제안되었습니다.

- **Technical Details**: {'MCQ 평가 기준': '기존의 BLEU, ROUGE, METEOR 메트릭이 데이터셋 내의 골드 샘플과의 n-gram 유사성에만 집중하기 때문에 교육적 가치를 평가하지 못합니다. 이를 해결하기 위해 KDA(Knowledge Dependent Answerability)라는 새로운 자동 평가 메트릭을 제안하였으며, 이는 학생이 해당 지식을 알고 있을 때 MCQ의 답변 가능성을 측정합니다.', 'Robustness 향상': '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 모델의 robustness를 향상시키는 접근법을 제안했습니다. 기존 방법들과 달리 여러 개의 counterfactual을 생성하고 집합적인 의사결정을 통해 각 용어의 인과관계를 보다 robust하게 감독합니다.', 'UniMoT': 'UniMoT는 분자와 텍스트 모달리티를 통합한 LLM으로, Molecule Tokens과 Text Tokens을 동일하게 취급하는 Tokenizer 기반 아키텍처를 사용합니다. Vector Quantization를 통해 분자를 일련의 토큰으로 변환하며, 이에 따라 분자를 새로운 언어처럼 해석하고 생성할 수 있습니다.'}

- **Performance Highlights**: {'MCQ 평가 기준': 'KDA_disc와 KDA_cont는 실제 교실 환경에서 사용성과 강한 상관관계를 가지며, 기존 n-gram 기반 유사성 메트릭과 결합될 때 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측력을 보입니다.', 'Robustness 향상': '제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 희소 데이터에서의 generalization 등의 다양한 차원에서 큰 향상을 보여주었습니다.', 'UniMoT': 'UniMoT는 분자 이해 및 생성 작업에서 최첨단 성능을 달성하였으며, 다중 모달리티 이해와 생성을 모두 수행할 수 있는 능력을 입증했습니다.'}



### Mission Impossible: A Statistical Perspective on Jailbreaking LLMs (https://arxiv.org/abs/2408.01420)
- **What's New**: 이번 연구에서는 전통적인 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 교육적 가치를 평가하지 못한다는 점을 지적하며, 새로운 자동 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안했습니다. KDA는 대상 지식을 기반으로 학생이 문제를 답변할 수 있는지를 평가합니다.

- **Technical Details**: KDA는 학생들의 응답을 기준으로 측정하며, 자동 평가 메트릭 KDA_disc와 KDA_cont를 도입해 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 기존의 n-gram 기반 유사도 메트릭과 결합할 때, KDA_disc와 KDA_cont는 다양한 전문가 평가 기준에 대해 강한 예측력을 보입니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 나타냈습니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 메트릭을 보완할 수 있는 강력한 도구임을 시사합니다.



### Pre-trained Language Models Improve the Few-shot Prompt Ability of Decision Transformer (https://arxiv.org/abs/2408.01402)
Comments:
          2 figures, 8 tables. Accepted by the Training Agents with Foundation Models Workshop at RLC 2024

- **What's New**: {'first_paper': '이 연구에서는 자동 MCQ 생성의 교육적 가치를 평가하기 위해 기존의 BLEU, ROUGE, METEOR 메트릭을 보완하는 새로운 자동 평가 메트릭, 즉 지식 종속 가능성(KDA)을 제안했다. 이는 학생이 특정 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 측정하는 데 중점을 둔다.', 'second_paper': '본 연구는 NLP 태스크에서 deep model의 정확성이 높지만 spurious pattern에 기반한 제한된 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 방법을 제안한다.', 'third_paper': '이 연구에서는 사전 훈련된 언어 모델을 활용하여 Prompt Decision Transformer의 few-shot prompt 능력을 개선하는 새로운 프레임워크인 LPDT(Language model-initialized Prompt Decision Transformer)를 제안한다. 강력한 일반화 능력을 통해 unseen tasks에서의 성능을 높인다.'}

- **Technical Details**: {'first_paper': 'KDA는 학생 응답을 통해 MCQ의 대답 가능성을 측정하며, 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다. 이 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다.', 'second_paper': "기존의 반사실적 증강 방법은 사람이 직접 데이터셋에 반사실적을 추가하거나 모델이 자동으로 데이터셋에서 유사 반사실적을 찾는 방식을 사용했지만, 이는 여전히 spurious correlation에 영향을 받는다. 본 연구는 '여러 개의' 반사실적을 생성하고 집합적 의사 결정을 통해 robustness를 향상시키는 방법을 제안한다.", 'third_paper': 'LPDT 프레임워크는 사전 훈련된 언어 모델을 DT의 초기화 값으로 사용하며, Low-Rank Adaptation (LoRA)을 활용하여 파라미터의 일부만을 수정하면서 멀티태스크 RL(oRL) 데이터셋을 통해 모델을 미세 조정한다. 또한 prompt regularization을 도입하여 다양한 RL 태스크를 구별할 수 있도록 한다.'}

- **Performance Highlights**: {'first_paper': 'Human study 결과, KDA_disc와 KDA_soft가 실제 강의실 세트에서의 usability와 강한 상관관계를 나타냈으며, n-gram 유사성 메트릭과 결합했을 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대한 예측력이 높았다.', 'second_paper': '제안된 방법은 다양한 차원에서 현저한 성능 향상을 보여주었다. 여기에는 counterfactual robustness, cross-domain generalization, and generalization from scarce data가 포함된다.', 'third_paper': 'MuJoCo control 환경과 Meta World ML1 태스크에서의 종합 실험 결과, LPDT는 기존 기준 모델들 대비 향상된 성능을 나타내며, 특히 데이터가 제한적인 상황에서도 높은 누적 보상을 달성했다.'}



### Toward Automatic Relevance Judgment using Vision--Language Models for Image--Text Retrieval Evaluation (https://arxiv.org/abs/2408.01363)
Comments:
          Accepted by ACM SIGIR 2024 LLM4Eval Workshop: this https URL

- **What's New**: 자동 MCQ 생성과 평가 메트릭 관련 연구에서 교육적 가치를 잘 반영하지 못한다는 문제를 해결하기 위해 새로운 평가 메트릭인 지식 종속 대답 가능성 (Knowledge Dependent Answerability, KDA)을 제안합니다. 또한, 최근 NLP 태스크의 deep model들이 높은 정확성을보였지만 spurious patterns에 의존하여 robustness가 제한된다는 것을 인지하고, 이 문제를 해결하기 위해 대조 학습 (contrastive learning)과 반사실적 증강 (counterfactual augmentation)을 활용합니다. 마지막으로, Vision-Language Models (VLMs)의 이미지-텍스트 검색 성능을 평가하며, 인간의 판단과 비교하여 GPT-4V와 같은 모델이 더 높은 일치도를 보여준다는 점을 강조합니다.

- **Technical Details**: 첫 번째 연구에서는 KDA를 측정하는 새로운 방법을 제안합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방할 수 있도록 미리 훈련된 언어 모델을 사용합니다. 두 번째 연구에서는 여러 반사실적 상황을 생성해 이를 바탕으로 분포 기반의 결정 (collective decisions)을 통해 모델의 robustness를 향상시킵니다. 마지막 연구에서는 CLIP, LLaVA, GPT-4V와 같은 VLM들을 사용해 이미지-텍스트 검색 성능을 평가하고 Kendall's τ와 Cohen's κ 값으로 성능을 비교합니다.

- **Performance Highlights**: 자동 평가 메트릭 KDA_disc와 KDA_cont는 KDA 및 실제 강의실 사용성 측면에서 강한 상관관계를 보였습니다. 대조 학습을 통한 모델은 다양한 측면에서 기존 방법들보다 높은 반사실적 robustness와 도메인 간 일반화 성능을 달성했습니다. GPT-4V와 같은 VLM들은 CLIPScore를 능가하는 일치도를 보였으며, 특히 인간 판단에 가장 근접한 분포를 보여줍니다.



### Prompt Refinement or Fine-tuning? Best Practices for using LLMs in Computational Social Science Tasks (https://arxiv.org/abs/2408.01346)
Comments:
          5 pages, 1 table

- **What's New**: 최근 연구는 Multiple Choice Questions (MCQ)의 자동 생성이 교사들의 평가 시간을 크게 줄일 수 있다고 주장하고 있습니다. 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ의 단어 유사성에만 초점을 맞추어 교육적 가치를 간과하고 있습니다. 이에 따라 연구진은 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안하였으며, 이는 학생이 특정 지식을 알고 있을 때 MCQ에 답할 수 있는 가능성을 평가하는 데 중점을 둡니다.

- **Technical Details**: KDA는 학생들의 응답을 바탕으로 MCQ의 대답 가능성을 측정합니다. 연구진은 또한 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행위를 모사하는 KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안했습니다. KDA_disc는 이분법적 접근, KDA_cont는 연속 변수로 접근하여 측정합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 교실 사용성과 강한 상관관계를 갖는다는 것이 입증되었습니다. 또한, 기존의 n-gram 기반 평가 메트릭과 결합할 때 KDA_disc와 KDA_cont는 여러 전문가들이 표시한 MCQ 품질 기준을 예측하는 데 높은 신뢰성을 보였습니다.



### MuChoMusic: Evaluating Music Understanding in Multimodal Audio-Language Models (https://arxiv.org/abs/2408.01337)
Comments:
          Accepted at ISMIR 2024. Data: this https URL Code: this https URL Supplementary material: this https URL

- **What's New**: 이 논문은 교육적 가치를 반영하지 않는 기존의 자동 MCQ 생성을 평가하는 메트릭 대신, 학생의 지식 평가 능력을 측정할 수 있는 Knowledge Dependent Answerability (KDA)를 제안합니다. 두 번째 논문에서는 NLP 모델의 robustness를 향상시키기 위한 새로운 접근법을 제안하며, 세 번째 논문에서는 음악 이해를 위한 Audio LLMs 평가를 위한 MuChoMusic이라는 벤치마크를 소개합니다.

- **Technical Details**: [{'Title': '자동 MCQ 생성 평가를 위한 새로운 메트릭 KDA', 'Technical Details': '기존 BLEU, ROUGE, METEOR 평가 메트릭은 MCQ의 교육적 가치를 평가하지 못합니다. 이를 개선하기 위해 학생들의 답변을 바탕으로 KDA를 측정하고 이를 예측하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.', 'Performance Highlights': 'Human eval와의 강한 상관관계를 입증했습니다.'}, {'Title': 'Robustness 향상을 위한 새로운 접근법', 'Technical Details': 'spurious 패턴에 의존하는 deep models의 문제를 해결하기 위해, 여러 counterfactual을 생성하여 집합적 의사 결정을 통해 단어들의 인과관계를 파악하는 방법을 제안합니다.', 'Performance Highlights': 'Task model bias에 덜 민감해지는 것을 입증했습니다. 또한 다양한 차원에서의 성능 향상을 보였습니다 (counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등).'}, {'Title': 'MuChoMusic: 음악 이해를 위한 벤치마크', 'Technical Details': '음악 이해를 위한 Audio LLM를 평가하기 위해 1,187개의 다중 선택 질문을 포함한 MuChoMusic을 소개합니다. 이 벤치마크는 다양한 장르의 음악 트랙을 포함하고 인간 주석자에 의해 검증된 것입니다.', 'Performance Highlights': 'MuChoMusic을 이용해 5개의 기존 Audio LLM를 평가하고, 여러 핏폴을 찾아내며, 더 나은 멀티모달 통합의 필요성을 강조합니다.'}]



### The Impact of Hyperparameters on Large Language Model Inference Performance: An Evaluation of vLLM and HuggingFace Pipelines (https://arxiv.org/abs/2408.01050)
- **What's New**: 두 논문은 NLP (Natural Language Processing) 모델의 평가와 견고성(robustness)을 다루며, 각각 MCQ (Multiple Choice Questions) 자동 생성 평가와 NLP 모델의 견고성 향상에 새로운 접근 방식을 제안합니다. 또한, 최근 LLM (Large Language Models) 성능 분석에 대한 논문도 포함되었습니다.

- **Technical Details**: 첫 번째 논문에서는 BLEU, ROUGE, METEOR 등의 n-gram 기반 메트릭이 MCQ 생성 평가에서 교육적 가치를 반영하지 못한다는 문제를 지적합니다. 이를 해결하기 위해, 지식 종속 가능성(KDA)을 측정하는 새로운 메트릭을 제안하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입했습니다. 두 번째 논문은 NLP 모델이 spurious pattern에 의존할 때 견고성이 약해지는 문제를 다루며, Contrastive Learning과 Counterfactual Augmentation을 통해 이를 개선하려는 방법을 제안합니다. 이 방식은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 각 단어의 인과관계를 분석합니다. 마지막 논문은 vLLM과 HuggingFace 파이프라인을 비교하여 20개의 LLM을 분석하며, 하이퍼파라미터 최적화가 성능에 미치는 영향을 조사합니다.

- **Performance Highlights**: 첫 번째 논문에서는 KDA_disc와 KDA_cont가 실제 교실 환경에서 전문가들이 레이블한 유용성과 강한 상관관계를 가진다는 점을 보여주었습니다. 두 번째 논문은 제안된 방법이 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 관점에서 상당한 개선을 이뤘음을 입증했습니다. 세 번째 논문에서는 하이퍼파라미터 최적화가 통해 HuggingFace 파이프라인의 처리량(throughput)이 평균 9.16%에서 13.7%까지 향상되는 결과를 보였습니다.



### Enhancing Financial Market Predictions: Causality-Driven Feature Selection (https://arxiv.org/abs/2408.01005)
Comments:
          Accepted by The 20th International Conference Advanced Data Mining and Applications 2024 (ADMA 2024)

- **What's New**: 이 논문은 자동 MCQ(선다형 질문) 생성의 평가 메트릭으로 새로운 접근법인 Knowledge Dependent Answerability(KDA)를 제안합니다. 기존의 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 평가하지 못하지만, KDA는 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. 두 번째 논문은 대조 학습 및 반사례 보충(counterfactual augmentation)을 활용하여 고도의 NLP 모델의 강인성을 향상시키는 방법을 제안합니다. 마지막으로 세 번째 논문은 FinSen 데이터셋을 소개하며, 이는 197개국의 경제 및 금융 뉴스 기사와 주식 시장 데이터를 통합하여 금융 시장 분석을 혁신합니다.

- **Technical Details**: MCQ 생성 논문에서는 KDA를 측정하기 위해 학생 응답을 기반으로 한 사람이 실시한 설문 조사를 통해 평가한 후, 사전 학습된 언어 모델을 활용하여 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 둘째 논문에서는 반사례를 '세트'로 생성하고 이 세트에 대해 집합적 의사 결정을 내려 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다. 세 번째 논문은 금융 뉴스의 원인적 검증된 감정 점수와 LSTM 모델을 활용하여 시장 예측의 정확성과 신뢰성을 향상시키는 방법을 다룹니다.

- **Performance Highlights**: MCQ 논문에서 제안된 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 지표에 대한 예측력을 향상시켰습니다. 두 번째 논문에서는 반사례 강인성(counterfactual robustness), 크로스-도메인 일반화(cross-domain generalization), 희소 데이터로부터의 일반화에서 개선된 성능을 보였습니다. 마지막으로 FinSen 데이터셋과 개선된 LSTM 모델을 통해 금융 시장 예측에서 중요한 지표인 ECE를 3.34%로 낮추는 성과를 달성했습니다.



### ArchCode: Incorporating Software Requirements in Code Generation with Large Language Models (https://arxiv.org/abs/2408.00994)
Comments:
          Accepted by ACL 2024 main conference

- **Multiple Choice Question (MCQ) Generation**: [{"What's New": '기존 MCQ 생성 평가 메트릭의 한계를 극복하기 위해, 자동 평가 메트릭 지식 종속 가능성(KDA)을 제안했습니다. 이는 MCQ의 대답 가능성 및 학생의 관련 지식을 평가하는 능력을 측정합니다.'}, {'Technical Details': 'KDA는 학생의 응답을 기반으로 측정되며, 이를 근사하기 위해 사전 학습된 언어모델을 활용한 KDA_disc와 KDA_cont 메트릭 두 가지를 제안했습니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실에서의 사용성 및 전문가가 라벨링한 다양한 MCQ 품질 측정에서 강한 상관관계를 보였습니다.'}]

- **NLP Task Robustness**: [{"What's New": '국지적 패턴(spurious patterns)에 대한 의존성 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 이용하여 새로운 방법을 제안했습니다.'}, {'Technical Details': '기존 방법들과 달리 여러 개의 반사실적 데이터 셋트를 생성하고, 집합적 의사 결정(collective decisions)을 통해 더 확실하게 단어들의 인과관계를 감독합니다.'}, {'Performance Highlights': '제안된 방법은 다양한 측면에서 현저한 성능 향상을 보여주었으며, 특히 반사실적 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희소 데이터에서의 일반화 부분에서 두드러졌습니다.'}]

- **Code Generation Enhancement**: [{"What's New": '대형 언어모델(LLMs)의 코드 생성 기능을 텍스트로부터 소프트웨어 요구 사항을 포괄적으로 관리할 수 있도록 확장하는 ARCHCODE 프레임워크를 도입했습니다.'}, {'Technical Details': 'ARCHCODE는 In-Context Learning(ICL)을 활용해 텍스트 설명에서 요구 사항을 체계적으로 추출하고, 표현되지 않은 요구 사항까지도 추론하여 코드와 테스트 케이스를 생성합니다.'}, {'Performance Highlights': 'ARCHCODE는 GPT-4의 Pass@1 점수를 능가하며, CodeContests에서 새로운 최첨단 성과를 달성했습니다. 또한, HumanEval-NFR 벤치마크를 도입하여 비기능적 요구 사항(NFRs)까지 효과적으로 만족시켰음을 입증했습니다.'}]



### Towards Zero-Shot Annotation of the Built Environment with Vision-Language Models (Vision Paper) (https://arxiv.org/abs/2408.00932)
- **What's New**: 최근 발표된 논문에서는 자동 MCQ(다지선다형 질문) 생성의 교육적 가치를 평가할 새로운 메트릭을 제안했습니다. 기존의 BLEU, ROUGE, METEOR 등의 메트릭은 생성된 질문의 단어 유사성에만 초점을 맞추고 있어 실질적인 교육 효과를 놓치는 문제가 있습니다. 이를 해결하기 위해 '지식 의존 정답 가능성(Knowledge Dependent Answerability, KDA)'이라는 새로운 평가 메트릭을 도입했습니다.

- **Technical Details**: KDA는 생성된 MCQ가 해당 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지를 측정합니다. 이를 바탕으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였으며, 학생들의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 활용하여 KDA를 근사합니다.

- **Performance Highlights**: Human study를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 라벨링한 실제 클래스룸 세팅의 사용성과 강한 상관관계를 가짐을 입증했습니다. 추가로, n-gram 기반 유사성 메트릭과 함께 사용했을 때, KDA_disc와 KDA_cont는 다양한 전문가 라벨링 MCQ 품질 측정에 대해 높은 예측력이 있음을 보여주었습니다.



### Automatic Pull Request Description Generation Using LLMs: A T5 Model Approach (https://arxiv.org/abs/2408.00921)
Comments:
          Accepted to 2nd International Conference on Artificial Intelligence, Blockchain, and Internet of Things (AIBThings-2024), September 07-08, 2024, Michigan, USA

- **What's New**: 자동 MCQ 생성에 있어 교육적 가치를 고려한 새로운 평가 메트릭 KDA (Knowledge Dependent Answerability)를 제안했습니다. 또한, NLP 태스크의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법을 소개하고, PR 설명 생성을 위한 T5 transformer 모델의 성능을 분석했습니다.

- **Technical Details**: KDA 메트릭은 MCQ의 대답 가능성을 측정하고, 인간 평가를 통해 생성된 KDA_disc와 KDA_cont가 실제 사용성과 강한 상관관계를 가지는 것을 확인했습니다. NLP 모델의 robustness를 위해 collective decision을 활용해 여러 개의 counterfactual을 생성하여 인과관계를 파악하는 방법을 제안합니다. 또한, T5 transformer 모델을 미세조정하여 커밋 메시지와 소스 코드 코멘트 기반의 PR 설명 생성을 수행합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블링한 MCQ 품질 측정에서 높은 예측력을 보였습니다. Counterfactual robustness, cross-domain generalization, data scarcity 상황에서도 우리의 새로운 접근 방식이 기존 방법보다 개선된 성능을 보였습니다. T5 모델은 ROUGE 메트릭을 기준으로 LexRank 기준 모델보다 우수한 성능을 입증했습니다.



### Granting GPT-4 License and Opportunity: Enhancing Accuracy and Confidence Estimation for Few-Shot Event Detection (https://arxiv.org/abs/2408.00914)
- **What's New**: 새로운 MCQ 평가 메트릭인 Knowledge Dependent Answerability (KDA)이 제안되었습니다. GPT-4를 이용한 자체 신뢰도 추정 향상 관련 연구도 소개되었습니다.

- **Technical Details**: 기존 MCQ 평가 메트릭인 BLEU, ROUGE, METEOR의 한계를 극복하기 위해, KDA는 학생의 응답을 통한 인간 설문 조사를 기반으로 측정 방법을 제시합니다. 추가로, KDA를 모사하는 pre-trained language models를 이용한 KDA_disc와 KDA_cont를 제안합니다. GPT-4의 신뢰도 추정을 개선하기 위해 License to speculate와 Opportunity to quantify (L&O) 방법을 이용했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ의 품질 평가와 강한 상관관계를 보였습니다. 또한, GPT-4의 경우 신뢰도 측정을 통해 AUC 0.759를 달성하며 정확성을 향상시켰습니다.



### Hybrid Querying Over Relational Databases and Large Language Models (https://arxiv.org/abs/2408.00884)
- **What's New**: 이번 소식에서는 Multiple Choice Questions (MCQ) 자동 생성과 관련된 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)를 소개합니다. 또한, 최근 Natural Language Processing (NLP) 태스크에서 deep models의 robustness를 향상시키기 위한 새로운 접근법과 관계형 데이터베이스와 대형 언어 모델(Large Language Models, LLMs)을 결합한 Hybrid Querying 시스템이 논의됩니다.

- **Technical Details**: [{'Description': 'MCQ의 교육적 가치를 평가하기 위해 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 메트릭을 제안하며, 이는 학생의 응답을 기반으로 MCQ의 대답 가능성(answerability)을 평가합니다. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭도 도입하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Description': 'NLP 태스크에서 deep models의 robustness를 향상시키기 위해 대조 학습(Contrastive Learning)과 반사실적 데이터 증강(Counterfactual Augmentation)을 활용하는 방법을 제안합니다. 여러 개의 반사실적 데이터를 생성하고 집합적인 의사 결정을 통해 용어의 인과관계를 감독합니다.'}, {'Description': '관계형 데이터베이스와 대형 언어 모델(LLMs)을 결합하여 기존 데이터베이스(Closed World Assumption)를 넘어선 질문을 풀기 위한 Hybrid Querying 시스템을 도입합니다. 이를 위해 HQDL(SWAN)을 제안하고, 이를 통해 실제 데이터를 기반으로 한 120개의 복합 질문을 해결합니다.'}]

- **Performance Highlights**: [{'Description': 'KDA_disc와 KDA_cont는 KDA와 전문가가 평가한 실제 강의실 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합했을 때 예측력이 높았습니다.'}, {'Description': '제안된 반사실적 증강 방법은 기존 모델의 편향에 덜 민감하며, 반사실적 로버스트니스(Counterfactual Robustness), 크로스 도메인 일반화(Cross-Domain Generalization), 그리고 부족한 데이터로부터의 일반화(Generalization from Scarce Data)에서 유의미한 개선을 보였습니다.'}, {'Description': 'HQDL의 GPT-4 Turbo 모델은 실행 정확도 40.0%와 데이터 사실성 48.2%를 달성했으며, 이는 하이브리드 쿼리 방식의 잠재력과 도전을 모두 강조합니다.'}]



### Leveraging LLM Reasoning Enhances Personalized Recommender Systems (https://arxiv.org/abs/2408.00802)
Comments:
          To be published at ACL 2024

- **What’s New**: 새로운 자동 평가 메트릭 KDA (Knowledge Dependent Answerability)를 제안하여 MCQ (Multiple Choice Questions)의 교육적 가치를 평가합니다. 또한, 최근 NLP 모델의 취약성과 관련하여 대조 학습과 반사실 증강을 활용한 방법을 소개합니다. 마지막으로, RecSys (추천 시스템)에서 LLM (Large Language Models)의 추론력을 활용하여 개인화 추천 작업 성능을 개선하는 연구가 포함됩니다.

- **Technical Details**: {'MCQ Generation': {'KDA': '학생들이 특정 사실을 가지고 MCQ에 답할 수 있는가를 측정하는 새로운 평가 메트릭입니다.', 'KDA_disc와 KDA_cont': '특정 지식을 바탕으로 KDA를 자동적으로 평가하는 두 가지 메트릭으로, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, 'Robustness in NLP Models': {'대조 학습과 반사실 증강': '스퍼리어스 패턴 (spurious patterns)에 의존할 수 있는 NLP 모델의 취약성을 줄이기 위해 대조 학습 및 집합적 의사 결정 방식을 사용한 새로운 접근법을 제안합니다.'}, 'LLM in RecSys': {'Rec-SAVER': '추천 시스템에서 LLM의 추론 응답의 품질을 자동으로 평가하는 프레임워크로, 큐레이션된 골드 레퍼런스나 인간 평가자가 필요 없습니다.'}}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합 시 예측 정확도 상승을 보였습니다.', 'Robustness in NLP Tasks': '제안된 접근법은 반증 가능성, 교차 도메인 일반화 및 희소 데이터에서의 일반화 측면에서 크게 개선된 성능을 보여주었습니다.', 'LLM in RecSys': '신규 Rec-SAVER 프레임워크는 인간 평가와 일치하는 추론 응답의 일관성 및 신뢰성을 평가하며, zero-shot과 fine-tuning 설정 모두에서 추천 작업 성능을 향상시켰습니다.'}



### Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Bas (https://arxiv.org/abs/2408.00798)
- **Multiple Choice Questions (MCQ)**: {"What's New": '기존의 BLEU, ROUGE, METEOR 평가 메트릭들이 MCQ 생성의 교육적 가치를 충분히 평가하지 못한다는 문제를 해결하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다.', 'Technical Details': 'KDA는 학생들이 대상 사실에 대한 지식을 바탕으로 MCQ에 대답할 수 있는 능력을 측정합니다. 자동 평가를 위해, KDA_disc와 KDA_cont라는 두 가지 메트릭을 제안하여 사전 학습된 언어 모델을 사용해 학생들의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음이 확인되었습니다. n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 레이블링 한 다양한 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.'}

- **Robust Deep Models for NLP**: {"What's New": 'NLP 태스크에서 deep model의 robustness를 향상시키기 위해, 대조 학습(contrastive learning)과 반사적 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다.', 'Technical Details': '기존 방법들이 spurious correlations에 영향을 받는 반면, 우리의 접근법은 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decision)을 통해 각 용어의 인과관계를 robust하게 감독합니다.', 'Performance Highlights': '우리의 접근법은 attribution-based synthesis에 의한 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 등 다양한 측면에서 상당한 향상을 이룬 것을 empiral 결과를 통해 확인했습니다.'}

- **Golden-Retriever**: {"What's New": '대규모 산업 지식 베이스를 효율적으로 탐색하기 위해 설계된 Golden-Retriever를 소개합니다. 이 방식은 domain-specific jargon과 문맥 해석 문제를 해결합니다.', 'Technical Details': 'Golden-Retriever는 문서 검색 전에 질문을 증강하는 reflection-based question augmentation 단계를 도입하여 용어를 식별하고 문맥에 따라 의미를 명확히 합니다. 이 접근법은 RAG(Retrieval Augmented Generation) 프레임워크가 가장 관련성이 높은 문서를 검색하도록 보장합니다.', 'Performance Highlights': '세 가지 오픈 소스 LLM을 사용한 평가에서 Golden-Retriever는 domain-specific question-answer dataset에서 기존 방법보다 뛰어난 성능을 보였습니다. 이로써 산업 지식 베이스를 효율적으로 통합하고 쿼리하는 강력한 솔루션을 제공합니다.'}



### Decoding AI and Human Authorship: Nuances Revealed Through NLP and Statistical Analysis (https://arxiv.org/abs/2408.00769)
- **What's New**: ['자동 MCQ 생성 평가의 새로운 지표인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 도입했습니다. 이는 기존의 BLEU, ROUGE, METEOR과 같은 n-gram 기반 평가와는 달리 교육적 가치를 반영합니다.', '심층 신경망 모델의 견고성을 높이기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 제안했습니다.', '인공지능과 인간이 작성한 텍스트의 미묘한 차이를 분석하는 연구를 통해 AI의 창의적 능력과 이를 활용한 언어 생성의 영향을 탐구했습니다.']

- **Technical Details**: ['KDA는 학생이 특정 사실에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표가 제안됩니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', '대조 학습과 반사실적 증강을 사용하여 다수의 반사실적 데이터를 생성하고, 집합적 의사 결정(collective decision)을 통해 각각의 단어의 인과 관계를 보다 견고하게 감독하는 방식을 제안했습니다.', '총 500K개의 에세이를 분석하여 AI 생성 텍스트와 인간 작성 텍스트 간의 언어적 특성, 창의성 패턴, 잠재적 편향성을 조사했습니다. 이를 통해 AI의 언어 생성 능력과 창의적 능력에 대한 심층적인 이해를 돕습니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 학습자 및 전문가가 평가한 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다.', '제안된 반사실적 증강 방법은 반사실적 견고성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 부족한 데이터에서도 우수한 성능을 보였습니다.', 'AI 생성 텍스트는 인간 작성 텍스트에 비해 평균 단어 길이가 길고, 다소 높은 수준의 참신성을 보여주어 더 독창적인 콘텐츠 생성 가능성을 제시했습니다. 하지만, 인간 작성 텍스트는 더 높은 어휘 다양성을 보였습니다.']



### Quantification and Validation for Degree of Understanding in M2M Semantic Communications (https://arxiv.org/abs/2408.00767)
Comments:
          ICCT 2024

- **What's New**: 자동 MCQ 생성에서 기존 평가 메트릭은 교육적 가치를 반영하지 못한다는 문제를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방해 KDA를 근사합니다. 인간 평가를 통해 실제 강의실에서의 유용성과 강한 상관관계를 보였습니다.

- **Technical Details**: 블루(BLEU), ROUGE, METEOR 등 기존의 n-gram 기반 평가 메트릭과는 달리, KDA는 MCQ의 답변 가능성(answerability)을 측정하고 학생의 지식을 평가하는 능력을 중심으로 합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 방식으로 KDA를 자동으로 평가합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 높은 상관관계를 보임으로써 평가 메트릭으로서의 유용성을 입증했습니다. 특히, n-gram 기반 유사성 메트릭과 결합할 경우 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강력한 예측력을 가집니다.



### Characterizing User Archetypes and Discussions on Scored.co (https://arxiv.org/abs/2407.21753)
- **What's New**: 최근 출판된 논문에서는 자동 MCQ(객관식 질문) 생성의 평가를 위한 새로운 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 또한, NLP 태스크에서 deep model의 robust성을 높이기 위해 대조 학습과 counterfactual augmentation을 사용하는 방법과, 소셜 하이퍼네트워크의 노드 및 하이퍼엣지를 특성화하는 다차원 프레임워크를 제안했습니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 등의 평가 메트릭은 데이터셋 내의 골드 샘플과의 n-그램 기반 유사성에 중점을 두며 교육적 가치를 평가하지 않습니다. 이를 해결하기 위해, KDA는 학생이 특정 사실에 대한 지식을 바탕으로 객관식 질문에 대답할 수 있는지를 측정합니다. 우리는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 학습된 언어 모델을 통해 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.', 'Deep Model Robustness': '최근 deep model들은 NLP 태스크에서 초인간적인 정확성을 보였지만, spurious patterns에 의존하여 robust성이 제한되었습니다. 이를 해결하기 위해 대조 학습과 counterfactual augmentation을 사용하며, 특히 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decision)을 통해 단어들의 인과관계를 더욱 robust하게 감독할 수 있습니다.', 'Social Hypernetworks': '사회적 상호작용의 증가와 복잡성에도 불구하고, 일부 fringe social platforms에 대한 연구는 거의 이루어지지 않았습니다. 이 논문은 understudied alt-right 플랫폼인 Scored.co를 중심으로 소셜 하이퍼네트워크의 노드와 하이퍼엣지를 특성화하기 위한 다차원 프레임워크를 제시합니다. 이 프레임워크는 hypernetwork 표현과 다양한 노드 특성을 통합하여 distinct user archetypes를 정의하고 이들의 네트워크 내 역할을 이해하는 데 중점을 둡니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계가 있음을 입증했습니다. 또한, n-그램 기반 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ 품질 측정에 대한 강력한 예측력을 가지고 있습니다.', 'Deep Model Robustness': '제안된 방법은 다양한 측면에서 유의미한 개선을 달성했습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization.', 'Social Hypernetworks': '제안된 프레임워크는 Scored.co 플랫폼의 데이터를 이용한 실험 캠페인을 통해 높은 유효성을 보였습니다. 이 연구는 Scored.co의 구조적 특성을 최초로 분석한 연구이기도 합니다.'}]



New uploads on arXiv(cs.IR)

### Toward Automatic Relevance Judgment using Vision--Language Models for Image--Text Retrieval Evaluation (https://arxiv.org/abs/2408.01363)
Comments:
          Accepted by ACM SIGIR 2024 LLM4Eval Workshop: this https URL

- **What's New**: 자동 MCQ 생성과 평가 메트릭 관련 연구에서 교육적 가치를 잘 반영하지 못한다는 문제를 해결하기 위해 새로운 평가 메트릭인 지식 종속 대답 가능성 (Knowledge Dependent Answerability, KDA)을 제안합니다. 또한, 최근 NLP 태스크의 deep model들이 높은 정확성을보였지만 spurious patterns에 의존하여 robustness가 제한된다는 것을 인지하고, 이 문제를 해결하기 위해 대조 학습 (contrastive learning)과 반사실적 증강 (counterfactual augmentation)을 활용합니다. 마지막으로, Vision-Language Models (VLMs)의 이미지-텍스트 검색 성능을 평가하며, 인간의 판단과 비교하여 GPT-4V와 같은 모델이 더 높은 일치도를 보여준다는 점을 강조합니다.

- **Technical Details**: 첫 번째 연구에서는 KDA를 측정하는 새로운 방법을 제안합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방할 수 있도록 미리 훈련된 언어 모델을 사용합니다. 두 번째 연구에서는 여러 반사실적 상황을 생성해 이를 바탕으로 분포 기반의 결정 (collective decisions)을 통해 모델의 robustness를 향상시킵니다. 마지막 연구에서는 CLIP, LLaVA, GPT-4V와 같은 VLM들을 사용해 이미지-텍스트 검색 성능을 평가하고 Kendall's τ와 Cohen's κ 값으로 성능을 비교합니다.

- **Performance Highlights**: 자동 평가 메트릭 KDA_disc와 KDA_cont는 KDA 및 실제 강의실 사용성 측면에서 강한 상관관계를 보였습니다. 대조 학습을 통한 모델은 다양한 측면에서 기존 방법들보다 높은 반사실적 robustness와 도메인 간 일반화 성능을 달성했습니다. GPT-4V와 같은 VLM들은 CLIPScore를 능가하는 일치도를 보였으며, 특히 인간 판단에 가장 근접한 분포를 보여줍니다.



### Leveraging Knowledge Graph Embedding for Effective Conversational Recommendation (https://arxiv.org/abs/2408.01342)
Comments:
          26pages, 15figures

- **What's New**: MCQ (Multiple Choice Questions) 자동 생성에 대한 새로운 평가 메트릭과 강화된 강건성 (robustness)을 갖춘 NLP 모델, 그리고 knowledge graph 기반의 대화형 추천 시스템에 대한 새로운 접근법이 소개되었습니다.

- **Technical Details**: [{'MCQ 자동 생성': {'기존 평가 메트릭': 'BLEU, ROUGE, METEOR은 교육적 가치를 평가하지 않고, 단어 유사성에만 집중.', '새로운 메트릭': 'Knowledge Dependent Answerability (KDA)를 이용해 MCQ의 교육적 가치를 평가.', '자동 평가 방법': 'KDA_disc와 KDA_cont를 이용해 사전 학습된 언어 모델이 학생의 문제 해결 행동을 모방.'}}, {'강건성 강화 NLP 모델': {'문제점': '기존의 deep models는 spurious patterns에 의존하여 강건성이 부족.', '새로운 접근법': 'contrastive learning과 counterfactual augmentation을 이용, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 인과 관계 파악.'}}, {'대화형 추천 시스템 (CRS)': {'기존 문제점': '사용자와 속성, 아이템 간의 관계를 효과적으로 다루지 못함.', '새로운 접근법': 'Knowledge Graph 기반의 CRS (KG-CRS) 제안, 사용자-아이템 그래프와 아이템-속성 그래프를 통합한 동적 그래프로 대화 과정에서 갱신, 인접 노드를 통해 유용한 임베딩 학습.'}}]

- **Performance Highlights**: [{'MCQ 자동 생성': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 높은 상관관계를 보였으며, 다양한 전문가가 평가한 MCQ 품질 지표와도 높은 예측력을 가짐.', '강건성 강화 NLP 모델': 'Empirical 결과, 집합적 의사 결정을 통한 접근법이 반사적 치우침 (task model bias)에 덜 민감하고 counterfactual robustness, cross-domain generalization, sparse data generalization에서 높은 성과를 보임.'}, {'대화형 추천 시스템 (CRS)': '세 개의 실제 데이터셋에서 기존 최신 접근법 대비 대화 및 추천 태스크에서 월등한 성능 입증.'}]



### Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion (https://arxiv.org/abs/2408.00878)
- **What's New**: {'title': '새로운 MCQ 평가 메트릭, KDA 및 다양한 응용 연구', 'content': "MCQ 자동 생성은 교사의 시간을 절감할 수 있지만, 기존 평가 메트릭은 교육적 가치를 반영하지 못했습니다. 새로운 'Knowledge Dependent Answerability (KDA)' 메트릭을 제안하여 MCQ의 교육적 가치를 평가합니다. 또한, 'contrastive learning'과 'counterfactual augmentation'을 통한 모델의 강건성을 확보하는 새로운 접근 방법이 소개되었습니다. 마지막으로 다중 측면 정보 검색을 위한 새로운 접근법이 제안되었습니다."}

- **Technical Details**: {'MCQ Generation': {'description': "KDA를 통해 MCQ의 답변가능성을 측정하고, 'KDA_disc'와 'KDA_cont'라는 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.", 'additionalInfo': '이 메트릭들은 실제 클래스룸 세팅에서 전문가들이 라벨링한 사용성에도 강한 상관관계를 보였습니다.'}, 'Model Robustness': {'description': "최근 deep model들의 NLP 태스크에서의 성능 한계를 극복하기 위해 'contrastive learning'과 'counterfactual augmentation'을 활용합니다. 이 방법은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 단어들의 인과관계를 파악합니다.", 'additionalInfo': '집합적 의사 결정 방식으로 인해 다양한 차원에서 significant improvements를 달성했습니다.'}, 'Multi-Aspect Retrieval': {'description': "다중 측면 정보 검색(RIR)에서 late fusion의 실패 모드를 해결하기 위해 Large Language Model(LLM) 기반의 'query extraction'과 'generative reranking'을 포함한 여러 새로운 aspect fusion 전략을 제안합니다.", 'additionalInfo': '비균형적인 데이터 상황에서 MAP@10 성능을 0.36에서 0.52로 향상시켰으며, 균형 잡힌 데이터에서는 동일한 성능을 유지했습니다.'}}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 세팅에서 사용성 평가와 강한 상관관계를 나타냈고, 전문 라벨러들이 라벨링한 다양한 MCQ 품질 지표에 대한 예측력이 뛰어났습니다.', 'Model Robustness': '제안된 방법은 counterfactual robustness, cross-domain generalization, scarce data generalization 등 다양한 차원에서 significant improvements를 보여주었습니다.', 'Multi-Aspect Retrieval': 'Aspect Fusion 전략은 균형 잡힌 리뷰 데이터에서 late fusion (LF) 메서드보다 성능을 향상시키는 데 성공했습니다.'}



### Leveraging LLM Reasoning Enhances Personalized Recommender Systems (https://arxiv.org/abs/2408.00802)
Comments:
          To be published at ACL 2024

- **What’s New**: 새로운 자동 평가 메트릭 KDA (Knowledge Dependent Answerability)를 제안하여 MCQ (Multiple Choice Questions)의 교육적 가치를 평가합니다. 또한, 최근 NLP 모델의 취약성과 관련하여 대조 학습과 반사실 증강을 활용한 방법을 소개합니다. 마지막으로, RecSys (추천 시스템)에서 LLM (Large Language Models)의 추론력을 활용하여 개인화 추천 작업 성능을 개선하는 연구가 포함됩니다.

- **Technical Details**: {'MCQ Generation': {'KDA': '학생들이 특정 사실을 가지고 MCQ에 답할 수 있는가를 측정하는 새로운 평가 메트릭입니다.', 'KDA_disc와 KDA_cont': '특정 지식을 바탕으로 KDA를 자동적으로 평가하는 두 가지 메트릭으로, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, 'Robustness in NLP Models': {'대조 학습과 반사실 증강': '스퍼리어스 패턴 (spurious patterns)에 의존할 수 있는 NLP 모델의 취약성을 줄이기 위해 대조 학습 및 집합적 의사 결정 방식을 사용한 새로운 접근법을 제안합니다.'}, 'LLM in RecSys': {'Rec-SAVER': '추천 시스템에서 LLM의 추론 응답의 품질을 자동으로 평가하는 프레임워크로, 큐레이션된 골드 레퍼런스나 인간 평가자가 필요 없습니다.'}}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합 시 예측 정확도 상승을 보였습니다.', 'Robustness in NLP Tasks': '제안된 접근법은 반증 가능성, 교차 도메인 일반화 및 희소 데이터에서의 일반화 측면에서 크게 개선된 성능을 보여주었습니다.', 'LLM in RecSys': '신규 Rec-SAVER 프레임워크는 인간 평가와 일치하는 추론 응답의 일관성 및 신뢰성을 평가하며, zero-shot과 fine-tuning 설정 모두에서 추천 작업 성능을 향상시켰습니다.'}



### Low Rank Field-Weighted Factorization Machines for Low Latency Item Recommendation (https://arxiv.org/abs/2408.00801)
- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성과 관련하여, 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ와 데이터셋의 표본 간의 n-그램 기반 유사성을 중점으로 평가해 교육적 가치를 간과한다. 이를 해결하기 위해 우리는 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안하였다. 이는 MCQ의 답변 가능성을 평가하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정한다.

- **Technical Details**: 학생 응답 데이터를 바탕으로 KDA를 측정하는 방법을 먼저 제시한 후, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안한다. 이들은 KDA를 근사하는 방식이다. 또한 인간 연구를 통해, KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 확인하였다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-그램 기반 유사성 메트릭과 결합할 때, 다양한 전문가가 레이블링한 MCQ 품질 측정치에 대해 강한 예측력을 보임을 입증하였다.



### Chatbot-Based Ontology Interaction Using Large Language Models and Domain-Specific Standards (https://arxiv.org/abs/2408.00800)
- **What's New**: 최신 arXiv 논문들에서 소개된 흥미로운 연구 내용을 소개합니다. 첫 번째 논문은 MCQ (Multiple Choice Questions) 자동 생성의 평가 메트릭을 개선하기 위한 새로운 접근을 제시합니다. 두 번째 논문은 NLP 태스크에서 모델의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법을 설명합니다. 세 번째 논문은 LLM (Large Language Models)과 챗봇 인터페이스를 사용하여 SPARQL 쿼리 생성을 향상시키는 개념을 소개합니다.

- **Technical Details**: [{'title': 'MCQ 평가를 위한 Novel Metric, KDA', 'main': '기존 BLEU, ROUGE, METEOR 같은 n-gram 기반의 평가 메트릭은 교육적인 가치를 평가하지 못합니다. 이번 연구에서는 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 answerability를 측정하고, 학생이 목표 지식에 대한 이해도를 평가하는 능력을 향상시킵니다.'}, {'title': 'Robustness를 위한 Contrastive Learning과 Counterfactual Augmentation', 'main': '최근 NLP 모델들이 높은 정확성을 보이지만, spurious pattern 의존으로 인해 robustness가 부족한 문제를 인식했습니다. 본 연구는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 robust한 인과관계를 파악하는 방법을 제안합니다.'}, {'title': 'SPARQL 쿼리 생성을 위한 LLM과 챗봇', 'main': '산업적 응용을 위한 SPARQL 쿼리의 생성과 정확성을 위해 LLM과 챗봇 인터페이스를 도입한 개념을 제안합니다. 이 접근은 사용자 쿼리를 자연어로 입력받아 정확한 SPARQL 쿼리로 변환하고, 이를 통해 오탐지나 정보 왜곡을 방지합니다.'}]

- **Performance Highlights**: [{'title': 'KDA_disc와 KDA_cont의 결과', 'main': 'KDA_disc와 KDA_cont는 실제 강의실 세팅에서 효용성이 뛰어나며 전문가 레이블링 기준으로도 높은 상관관계를 보였습니다. n-gram 기반의 평가 메트릭과 결합 시, 다양한 MCQ 품질 측정에서 강한 예측력을 가집니다.'}, {'title': 'Robustness와 Generalization 성능 향상', 'main': '대조 학습과 반사실적 증강을 통해 모델의 counterfactual robustness, cross-domain generalization, 그리고 드문 데이터에서의 generalization 성능이 크게 향상되었습니다.'}, {'title': 'SPARQL 쿼리의 정확성', 'main': 'LLM을 활용한 SPARQL 쿼리의 실험적 결과, 정확성이 크게 향상되었습니다. 특히, 사용자 친화적인 인터페이스와 결합하여 오탐지 없이 정확한 결과를 제공함으로써 산업적 활용 가능성을 높였습니다.'}]



### Deep Uncertainty-based explore For Index Construction and Retrieval in Recommendation System (https://arxiv.org/abs/2408.00799)
- **What's New**: 기존의 자동 MCQ 생성 평가 메트릭은 교육적 가치를 고려하지 않았으나, 새로운 메트릭인 Knowledge Dependent Answerability (KDA)이 도입되어 학생의 지식을 평가하는 능력을 측정합니다. 또한, 최근 NLP 태스크에서 높은 정확성을 보인 deep model의 robustness를 개선하기 위해 대조 학습과 반사실적 증강(counterfactual augmentation)을 활용한 새로운 방법이 제안되었습니다. 그리고 추천 시스템에서의 신뢰성과 참신성을 균형 있게 조정할 수 있는 UICR 알고리즘도 소개되었습니다.

- **Technical Details**: MCQ 생성 평가에서 BLEU, ROUGE, METEOR는 n-gram 유사성만을 고려하는 반면, KDA는 특정 사실에 대한 지식을 기반으로 대답 가능성을 측정합니다. 새로운 KDA_disc와 KDA_cont 메트릭은 사전 학습된 언어 모델을 활용해 학생의 문제해결 행동을 모방합니다. NLP의 robustness 문제를 해결하기 위해, 여러 개의 counterfactual을 생성하고 집단적 결정을 통해 단어들의 인과관계를 파악하는 방법을 제안했습니다. 추천 시스템에서 UICR는 불확실성 모델링을 도입하여 관련성과 참신성을 동시에 고려한 매칭 결과를 제공합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 높은 상관관계를 나타냈습니다. 새로운 robustness 접근법은 counterfactual robustness, 도메인 간 일반화, 그리고 부족한 데이터에서의 일반화에서 상당한 개선을 보였습니다. UICR 알고리즘은 실제 산업 환경과 여러 개의 오픈소스 데이터셋에서 참신성을 희생하지 않고 관련성을 개선하여, Shopee의 온라인 A/B 테스트에서 그 효율성을 입증했습니다.



### Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Bas (https://arxiv.org/abs/2408.00798)
- **Multiple Choice Questions (MCQ)**: {"What's New": '기존의 BLEU, ROUGE, METEOR 평가 메트릭들이 MCQ 생성의 교육적 가치를 충분히 평가하지 못한다는 문제를 해결하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다.', 'Technical Details': 'KDA는 학생들이 대상 사실에 대한 지식을 바탕으로 MCQ에 대답할 수 있는 능력을 측정합니다. 자동 평가를 위해, KDA_disc와 KDA_cont라는 두 가지 메트릭을 제안하여 사전 학습된 언어 모델을 사용해 학생들의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음이 확인되었습니다. n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 레이블링 한 다양한 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.'}

- **Robust Deep Models for NLP**: {"What's New": 'NLP 태스크에서 deep model의 robustness를 향상시키기 위해, 대조 학습(contrastive learning)과 반사적 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다.', 'Technical Details': '기존 방법들이 spurious correlations에 영향을 받는 반면, 우리의 접근법은 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decision)을 통해 각 용어의 인과관계를 robust하게 감독합니다.', 'Performance Highlights': '우리의 접근법은 attribution-based synthesis에 의한 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 등 다양한 측면에서 상당한 향상을 이룬 것을 empiral 결과를 통해 확인했습니다.'}

- **Golden-Retriever**: {"What's New": '대규모 산업 지식 베이스를 효율적으로 탐색하기 위해 설계된 Golden-Retriever를 소개합니다. 이 방식은 domain-specific jargon과 문맥 해석 문제를 해결합니다.', 'Technical Details': 'Golden-Retriever는 문서 검색 전에 질문을 증강하는 reflection-based question augmentation 단계를 도입하여 용어를 식별하고 문맥에 따라 의미를 명확히 합니다. 이 접근법은 RAG(Retrieval Augmented Generation) 프레임워크가 가장 관련성이 높은 문서를 검색하도록 보장합니다.', 'Performance Highlights': '세 가지 오픈 소스 LLM을 사용한 평가에서 Golden-Retriever는 domain-specific question-answer dataset에서 기존 방법보다 뛰어난 성능을 보였습니다. 이로써 산업 지식 베이스를 효율적으로 통합하고 쿼리하는 강력한 솔루션을 제공합니다.'}



### PC$^2$: Pseudo-Classification Based Pseudo-Captioning for Noisy Correspondence Learning in Cross-Modal Retrieva (https://arxiv.org/abs/2408.01349)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 교육적 가치를 반영한 자동 MCQ 평가 메트릭, 반직관적 학습을 통한 NLP 모델 개선, 그리고 노이즈가 있는 데이터세트에서의 크로스 모달 검색을 위한 새로운 프레임워크가 각각 제안되었습니다.

- **Technical Details**: [{'Paper': 'Automatic MCQ Generation', 'Details': '기존의 BLEU, ROUGE, METEOR 메트릭의 한계를 극복하고자 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. 이는 목표 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': 'NLP 태스크에서 deep models의 robustness를 높이기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법을 제안했습니다. 이는 여러 개의 counterfactual을 생성하고, 집합적 의사 결정 (collective decisions)을 사용하여 모델의 인과관계를 감독합니다.'}, {'Paper': 'Cross-Modal Retrieval with Noisy Correspondence', 'Details': '노이즈가 있는 크로스 모달 검색을 개선하기 위해 Pseudo-Classification based Pseudo-Captioning (PC2) 프레임워크를 도입했습니다. 이는 캡션을 범주형 레이블로 해석하여 모델이 이미지-텍스트 의미적 유사성을 학습하게 하고, pseudo-captions을 생성하여 부정확한 데이터 쌍에 대해 더 정량적인 감독 정보를 제공합니다.'}]

- **Performance Highlights**: [{'Paper': 'Automatic MCQ Generation', 'Details': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 강한 상관관계를 가지며, 다양한 전문가가 레이블링한 MCQ 품질 지표에 대해 높은 예측력을 보였습니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': '집합적 의사 결정을 통해 반직관적 학습을 수행하여 다양한 측면 (1. counterfactual robustness, 2. cross-domain generalization, 3. scarce data generalization)에 대한 성능이 향상되었습니다.'}, {'Paper': 'Cross-Modal Retrieval with Noisy Correspondence', 'Details': 'PC2 프레임워크는 시뮬레이션 및 실제 데이터셋에서 기존의 크로스 모달 검색 기법 및 NCL-robust 방법들을 뛰어넘는 성능을 보였습니다. 또한, Noise of Web (NoW) 데이터셋을 통해 새로운 강력한 NCL 벤치마크를 도입하였습니다.'}]



### RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework (https://arxiv.org/abs/2408.01262)
- **What's New**: 새로운 논문들이 자동 MCQ 생성 평가 개선, NLP 태스크의 robustness 향상, 그리고 Retrieval-Augmented Generation (RAG) 시스템의 성능 평가를 다룹니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 등의 메트릭이 교육적 가치를 평가하지 못하는 문제를 해결하기 위해, 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모사합니다.', 'NLP Robustness': "NLP 모델들의 robustness를 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용합니다. '여러 개의' counterfactual을 생성하여 더 강건한 인과 관계 파악을 제안합니다.", 'RAG System Evaluation': 'RAGEval은 다양한 시나리오에서 LLMs의 지식 사용 능력을 평가하기 위해 만들어졌습니다. 이 프레임워크는 완전성(Completeness), 환각(Hallucination), 무관성(Irrelevance)의 세 가지 지표를 제안하여 성능을 평가합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont가 전문가가 라벨링한 실제 강의실 세트에서의 사용성과 강한 상관관계를 보여주었습니다.', 'NLP Robustness': '새로운 접근법은 다양한 차원에서 1) counterfactual robustness, 2) cross-domain generalization, 3) 소량 데이터에서의 generalization을 개선했습니다.', 'RAG System Evaluation': 'RAGEval은 LLMs가 특정 지식 출처에 의존하지 않고 다양한 도메인에서 효과적으로 성능을 발휘할 수 있도록 평가할 수 있게 합니다.'}



### Nested Music Transformer: Sequentially Decoding Compound Tokens in Symbolic Music and Audio Generation (https://arxiv.org/abs/2408.01180)
Comments:
          Accepted at 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: 이번 연구에서는 MCQ(Multiple Choice Questions)를 자동 생성하는데 있어서 기존의 평가는 교육적 가치를 반영하지 못한다는 문제를 해결하기 위해 'Knowledge Dependent Answerability (KDA)'라는 새로운 자동 평가 메트릭을 제안합니다. 또한, 최근의 딥 모델들이 NLP 태스크에서 높은 정확성을 보이나, 불순 패턴(spurious patterns)에 의존하는 문제가 있어 이를 극복하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 제시합니다. 마지막으로, 복합 토큰(compound token)을 사용하는 Nested Music Transformer (NMT)를 제안하여 메모리 효율적이고 성능이 우수한 음악 시퀀스 모델링을 가능하게 합니다.

- **Technical Details**: MCQ 평가에서는 KDA_disc와 KDA_cont라는 두 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 MCQ의 대답 가능성을 측정합니다. NLP 태스크에서는 여러 개의 반사실적 샘플을 생성하고 이를 통해 단어들의 인과관계를 더욱 견고하게 파악합니다. 음악 시퀀스 모델링에서는 Nested Music Transformer를 도입하여 복합 토큰을 자동회귀적으로 디코딩하면서도 낮은 메모리 사용량을 유지하고, 주 디코더와 서브 디코더를 결합한 고유의 크로스 어텐션 구조를 사용합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 지닌다는 것을 확인했습니다. 새로운 반사실적 증강 방법은 반사실적 강인성(counterfactual robustness), 크로스 도메인 일반화(cross-domain generalization), 그리고 데이터가 부족한 상황에서의 일반화에 있어서 큰 향상을 이루었습니다. Nested Music Transformer를 사용한 음악 시퀀스 모델링에서는 적은 GPU 메모리 사용과 빠른 학습 시간에도 불구하고, 기존 모델과 유사한 성능을 달성했습니다.



### BioRAG: A RAG-LLM Framework for Biological Question Reasoning (https://arxiv.org/abs/2408.01107)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서는 생명 과학 연구를 위한 질문-응답 시스템을 향상시키기 위해 BioRAG라는 새로운 Retrieval-Augmented Generation(RAG) 프레임워크를 도입했습니다. 이 시스템은 2200만 개의 과학 논문을 분석하고 인덱싱하여 대규모 언어 모델(LLMs)을 기반으로 훈련되었습니다. BioRAG는 도메인별 지식 계층 구조와 최신 데이터를 통합하여 검색 및 응답 생성을 개선합니다.

- **Technical Details**: BioRAG 프레임워크는 먼저 방대한 생명 과학 논문을 파싱, 인덱싱 및 세분화하여 고품질 훈련 코퍼스를 생성합니다. 이 후 생명 과학 도메인에 특화된 임베딩 모델을 훈련시키며, 외부 정보 소스를 포함하여 최신 데이터를 가져오는 것이 특징입니다. 또한 쿼리 전처리, 검색기 실행 컴포넌트, 모델이 충분한 정보를 수집하는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과, BioRAG는 여러 생명 과학 질문 응답 작업에서 맞춤형 LLM, 검색 엔진과 결합된 LLM, 다른 과학적 RAG 프레임워크를 뛰어넘는 성능을 보였습니다. 구체적으로, 컬렉티브 디시전 (collective decisions)을 통한 합성 counterfactual 생성 시, 상위 성능을 보이며 다양한 차원에서 강력한향상을 달성했습니다.



### An Encoding--Searching Separation Perspective on Bi-Encoder Neural Search (https://arxiv.org/abs/2408.01094)
- **What's New**: 이번 연구에서는 다중 선택 질문(MCQ)의 자동 생성과 관련한 기존 평가 매트릭의 문제를 해결하기 위해 새로운 평가 매트릭을 제안하고 있습니다. 새로운 평가 매트릭은 Knowledge Dependent Answerability (KDA)로, 이를 통해 생성된 MCQ가 학생의 지식을 평가하는 능력을 측정할 수 있습니다. 또한, 최근 NLP 태스크에서의 deep model의 robustness 문제를 해결하기 위해 새롭고 효율적인 대안적 학습 방법을 제안하고 있습니다. 마지막으로, neural search에서 널리 사용되는 bi-encoder 아키텍처의 한계와 문제점을 분석하고 새로운 관점의 인코딩-검색 분리(encoging-searching separation) 관점을 제시합니다.

- **Technical Details**: KDA는 학생이 해당 사실에 대한 지식을 갖고 있는지를 측정하여 MCQ의 대답 가능성을 평가하도록 설계된 새로운 자동 평가 메트릭입니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방함으로써 KDA를 근사합니다. 반면, NLP 모델의 robustness를 위한 방법으로 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 접근법을 제시합니다. 이와 함께 bi-encoder 아키텍처의 정보 병목 문제와 임베딩 검색의 기본 가정의 한계를 논리적으로 분석하고, 인코딩과 검색 작업을 개념적, 실질적으로 분리하는 인코딩-검색 분리 관점을 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 평가 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, 전문가가 라벨링한 여러 MCQ 품질 측정 기준에 대해 높은 예측력을 보였습니다. 또 대조 학습을 통한 접근은 counterfactual robustness, cross-domain generalization, 그리고 데이터가 적을 때의 generalization 측면에서 상당한 개선을 가져 왔습니다. bi-encoder 구조에서의 새로운 인코딩-검색 분리 관점은 정보 병목 현상을 더 잘 제어할 수 있게 해주며, 인코딩 및 검색 작업 설계의 자유도를 높이고 훈련 효율성을 향상시켰습니다.



### PERSOMA: PERsonalized SOft ProMpt Adapter Architecture for Personalized Language Prompting (https://arxiv.org/abs/2408.00960)
- **New Methods for Automated MCQ Evaluation**: [{"What's New": '기존의 자동 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ가 골드 샘플과 얼마나 유사한지를 n-gram 기반으로 평가하였으나, 교육적 가치는 고려하지 않았다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였다.'}, {'Technical Details': 'KDA는 학생의 대상 사실에 대한 지식을 평가할 수 있는 MCQ의 대답 가능성(answerability)을 측정한다. 구체적으로, 인간 설문 조사를 통해 KDA를 측정하는 방법을 제시하고, 학생들의 문제 해결 행동을 모방하기 위해 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안한다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었다. 또한, n-gram 기반의 유사성 메트릭과 결합될 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강한 예측력을 나타낸다.'}]

- **Enhancing NLP Model Robustness with Counterfactuals**: [{"What's New": '최근 NLP 태스크에서 deep model들이 사람보다 나은 정확성을 보여주지만, spurious pattern에 의존해 robustness가 제한된다는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용해 robustness를 향상시키는 방법을 제안한다.'}, {'Technical Details': '기존 방법들은 사람이 counterfactual을 만들거나 모델이 데이터셋에서 유사한 counterfactual을 찾는 방식을 사용했으나, 우리는 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안한다.'}, {'Performance Highlights': '우리의 방법은 attribution-based synthesis의 task model bias에 덜 민감하여 1) counterfactual robustness, 2) cross-domain 일반화, 3) scarce data로부터의 일반화에서 현저한 개선을 이룬다.'}]

- **Personalized User Interaction with Large Language Models**: [{"What's New": '사용자 맞춤형 자연어 시스템 구축을 위해 PERSOMA, Personalized Soft Prompt Adapter 아키텍처를 제안한다. 이는 사용자 히스토리를 효율적으로 캡처하기 위한 혁신적인 접근법을 제공한다.'}, {'Technical Details': 'PERSOMA는 사용자의 히스토리를 자유형 텍스트로 재샘플링하고 이를 expressive soft prompt embeddings로 압축하여 사용자 특화된 soft prompt 어댑터를 구축한다. 이를 통해 LLM의 이해력을 손상시키지 않으면서도 사용자 히스토리를 이용해 출력 결과를 사용자에게 맞출 수 있다.'}, {'Performance Highlights': 'PaLM 2 모델을 이용한 실험에서 PERSOMA는 MovieLens 데이터셋에서 기존 embedding-based 기술보다 0.18의 F1 점수 개선을 이루었으며, 전체 finetuned text prompting baseline과 유사한 성능을 나타내었다.'}]



### LICM: Effective and Efficient Long Interest Chain Modeling for News Recommendation (https://arxiv.org/abs/2408.00859)
- **What's New**: 최근 논문들은 여러 NLP 및 뉴스 추천 시스템에서의 성능 향상을 다루고 있습니다. 우선, 기존의 MCQ 생성 평가 메트릭들이 교육적 가치를 고려하지 않는 점을 지적하며, 새로운 지식 종속 가능성(KDA) 메트릭을 제안하였습니다. 두 번째 논문은 대립 학습(contrastive learning)과 반사실적 강화(counterfactual augmentation)를 활용하여 모델의 robust성을 개선하려는 노력을 보여줍니다. 마지막 논문은 뉴스 추천 시스템에서 사용자의 장기 관심도를 반영하는 모델을 소개합니다.

- **Technical Details**: [{'Topic': 'MCQ Generation', 'Details': '기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에 초점을 맞추기 때문에 교육적 가치를 평가하지 못합니다. 지식 종속 가능성(KDA) 메트릭은 지식 기반으로 MCQ의 대답 가능성을 측정하여 학생의 지식 평가 능력을 평가합니다. 또한, KDA_disc와 KDA_cont라는 자동화된 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Topic': 'NLP Robustness', 'Details': '현재 NLP 모델들이 높은 정확성을 유지하고 있지만, spurious pattern에 의해 robustness가 제한됩니다. 이에, 대립 학습과 반사실적 강화(counterfactual augmentation)를 활용하여 이러한 문제를 개선하고자 합니다. 우리의 접근 방식은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어들의 인과 관계를 보다 robust하게 파악합니다.'}, {'Topic': 'News Recommendation', 'Details': '뉴스 추천 시스템에서 사용자의 장기 관심도를 반영하기 위해 Long Interest Chain Modeling (LICM) 방법을 제안합니다. 이는 유사한 사용자들 간의 협력 기반 글로벌 뉴스 클릭 그래프를 통해 이끌어낸 정보로 장기 관심도를 모델링합니다. 최종적으로, gated network를 이용해 장기 관심도와 이웃 관심도를 통합하여 최종 사용자 표현을 생성합니다.'}]

- **Performance Highlights**: [{'Model': 'KDA_disc와 KDA_cont', 'Details': 'Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며 여러 전문가가 평가한 MCQ 품질 측정에 대한 예측력이 뛰어남을 확인했습니다.'}, {'Model': 'Counterfactual Augmentation Approach', 'Details': '제시된 방법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization에서 상당한 개선을 보여주었습니다.'}, {'Model': 'LICM', 'Details': '실제 데이터셋 실험 결과, 제안된 모델이 뉴스 추천 성능을 효과적이고 효율적으로 향상시켰음을 검증했습니다.'}]



New uploads on arXiv(cs.CV)

### NOLO: Navigate Only Look Onc (https://arxiv.org/abs/2408.01384)
- **What's New**: 이번 주 AI 뉴스레터에서는 신기술들이 발표되었습니다. 첫째, 자동 MCQ 생성 평가를 위해 지식 종속 가능성(KDA) 메트릭을 소개하는 논문이 나왔습니다. 둘째, 대조 학습과 counterfactual augmentation을 활용하여 NLP 모델의 robustness를 향상시키는 연구가 발표되었습니다. 마지막으로, 인컨텍스트 학습을 통한 비디오 내비게이션 정책 학습 방법을 제안한 연구가 있습니다.

- **Technical Details**: MCQ 생성 평가를 위해 KDA_disc와 KDA_cont라는 자동 평가 메트릭이 제안되었습니다. 이들은 사전 학습된 언어 모델을 이용해 학생들의 문제 해결 행동을 모방합니다. NLP 모델의 robustness를 강화하기 위해 여러 개의 counterfactual을 생성하고 이들을 통해 인과관계를 추론하는 방법이 소개되었습니다. 비디오 내비게이션에서는 NOLO(Navigate Only Look Once)라는 새로운 방법이 도입되어 Optical Flow와 오프라인 강화 학습을 결합하여 내비게이션 정책을 학습합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 기존 n-gram 기반 메트릭과 결합 시 예측력이 증가합니다. NLP 모델 연구에서는 주요 지표들에서 큰 향상을 보였으며, 다양한 차원에서의 generalization이 강화되었습니다. NOLO는 다양한 실험에서 기존 방법보다 훨씬 높은 성능을 보였으며, 단 30초 길이의 비디오 클립만으로도 Scene에 적응할 수 있음을 보여주었습니다.



### Spatial-Spectral Morphological Mamba for Hyperspectral Image Classification (https://arxiv.org/abs/2408.01372)
- **MCQ Generation**: [{"What's New": "자동 다지 선다형 문제(MCQ) 생성의 교육적 가치를 평가하기 위해 새로운 자동 평가 메트릭 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'이 제안되었습니다. 이는 기존 BLEU, ROUGE, METEOR 메트릭의 한계를 넘어서, 생성된 MCQ가 학생들의 지식을 평가하는 능력을 중점적으로 평가합니다."}, {'Technical Details': 'KDA는 학생들의 응답을 바탕으로 대답 가능성을 측정합니다. 그 후, 사전 학습된 언어 모델을 이용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다.'}, {'Performance Highlights': 'Human evaluation을 통해, KDA_disc와 KDA_cont는 실제 강의실 세팅에서의 사용성과 강한 상관관계를 갖고 있으며, n-gram 유사성 메트릭과 결합할 경우 다양한 전문가 라벨 MCQ 품질 지표에 대해 강한 예측 성능을 보입니다.'}]

- **Robustness in NLP Models**: [{"What's New": '최근 NLP 태스크에서 높은 정확성을 자랑하는 deep model들이 spurious pattern에 의존해 robustness가 제한된 문제를 해결하기 위해, 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용한 새로운 접근법이 제안되었습니다.'}, {'Technical Details': "기존 방법들이 데이터셋에서 반사실적 예시를 찾거나 추가하는 작업에 의존하는 반면, 이 접근법은 '여러 개의' 반사실적 예시를 생성하고 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 더 robust하게 파악합니다."}, {'Performance Highlights': '이 방법은 반사실적 견고성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 적은 데이터에서의 일반화(generalization from scarce data) 등 다양한 면에서 성능 개선을 보였습니다.'}]

- **Hyperspectral Image Classification with MorpMamba**: [{"What's New": 'Hyperspectral Image Classification(HSIC)에서 Transformer의 효율성 문제를 해결하기 위해, MorpMamba 모델이 제안되었습니다. 이는 State Space Model과 형태학적(morphological) 연산을 활용하여 높은 효율성과 정확도를 제공합니다.'}, {'Technical Details': 'MorpMamba 모델은 HSI 패치를 공간-스펙트럼 토큰으로 변환하고, depthwise separable convolutional operations를 사용하는 morphology 블록으로 처리합니다. 후속 처리는 multi-head self-attention 블록과 state space 블록에서 이루어집니다.'}, {'Performance Highlights': '다양한 Hyperspectral(H) 데이터셋에서 CNN 및 Transformer 모델에 비해 더 나은 parametric efficiency와 성능을 보였습니다.'}]



### EVIT: Event-based Visual-Inertial Tracking in Semi-Dense Maps Using Windowed Nonlinear Optimization (https://arxiv.org/abs/2408.01370)
Comments:
          8 pages, 5 figures, 3 tables, International Conference on Intelligent Robots and Systems 2024

- **What's New**: 최근 연구는 자동으로 다중 선택 질문(MCQ)을 생성하는 데 큰 잠재력을 가지고 있지만, 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 데이터셋의 샘플과 유사성을 중시하여 교육적 가치를 충분히 반영하지 않는다는 문제를 지적하고 있다. 이를 해결하기 위해 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하였다. 이 메트릭은 특정 사실에 대한 학생의 지식에 기초한 MCQ의 답변 가능성을 측정한다.

- **Technical Details**: KDA는 사용자가 목표 사실을 알고 있을 때 MCQ의 답변 가능성을 측정하는 메트릭이다. 구체적으로는 학생의 답변을 통해 사람 설문 조사 데이터를 기반으로 KDA를 측정하고, 이를 모방하기 위해 사전 학습된 언어 모델을 활용하여 자동 평가 메트릭(KDA_disc 및 KDA_cont)을 제안하였다. 이 모델들은 학생의 문제 해결 행동을 모방하여 KDA를 근사치로 계산한다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 KDA 및 실제 강의실 세트에서의 사용성과 높은 상관관계를 갖는다는 것을 보여주었다. 또한, 이 메트릭들은 n-그램 기반의 유사성 메트릭과 결합될 때 전문가들이 라벨링한 다양한 MCQ 품질 측정에 강력한 예측력을 보였다.



### Balanced Residual Distillation Learning for 3D Point Cloud Class-Incremental Semantic Segmentation (https://arxiv.org/abs/2408.01356)
- **What's New**: 이번 연구에서는 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안하여 MCQ의 대답 가능성(answerability)과 학생이 대상 사실에 대한 지식을 평가하는 능력을 측정했습니다. 또한, 기존 spurious pattern 의존성을 극복하기 위해 contrastive learning과 counterfactual augmentation을 활용한 개선된 robustness 접근법을 제시했습니다. Class-incremental learning(CIL)의 새로운 프레임워크인 'Balanced Residual Distillation (BRD-CIL)'을 3D 포인트 클라우드 세그멘테이션에 적용하여 기존의 문제점을 해결하고 성능을 크게 향상시켰습니다.

- **Technical Details**: MCQ 생성에서 BLEU, ROUGE, METEOR와 같은 기존 메트릭은 n-gram 기반 유사성 평가에만 집중했습니다. 이를 해결하기 위해 'KDA'를 도입하여 MCQ가 학생의 지식 평가에 적합한지를 측정했습니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. robustness 향상을 위해 여러 개의 counterfactual을 생성하고 집합적 의사결정(collective decision)을 통해 인과관계를 더 강력하게 감독합니다. BRD-CIL은 residual distillation learning strategy와 balanced pseudo-label learning strategy를 결합하여 기존 학습 지식을 보존하면서도 새로운 클래스 학습의 불균형 문제를 해결합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 더욱이, CIL을 위한 BRD-CIL 프레임워크는 3D 포인트 클라우드 세그멘테이션 작업에서 기존 방법보다 더 우수한 성능을 보이며, catastrophic forgetting 문제를 효과적으로 해결하였습니다. 특히, counterfactual robustness, cross-domain generalization, scarce data generalization에서 현저한 개선이 있음을 입증하였습니다.



### Hallu-PI: Evaluating Hallucination in Multi-modal Large Language Models within Perturbed Inputs (https://arxiv.org/abs/2408.01355)
Comments:
          Acccepted by ACM MM 2024, 14 pages, 11 figures, 9 tables

- **What's New**: 이 논문에서는 자동 MCQ 생성의 평가 메트릭으로 지식 종속 가능성(KDA)을 제안합니다. 기존 평가 메트릭은 교육적 가치를 반영하지 않지만, KDA는 학생의 문제 해결 능력을 측정합니다. 또한, 딥러닝 모델의 강건함을 높이기 위해 대비 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 활용하는 방법과 MLLM의 헛소리(hallucination)를 평가하기 위한 Hallu-PI 벤치마크를 소개합니다.

- **Technical Details**: [{'MCQ Generation Evaluation': '기존에 사용되던 BLEU, ROUGE, METEOR 평가 메트릭은 n-gram 기반의 유사성만을 고려하는 반면, KDA는 학생들이 대상 사실을 이해하고 답을 찾는 능력을 측정합니다. KDA_disc와 KDA_cont를 제안하여 사전 학습된 언어 모델을 이용해 학생들의 문제 해결 행위를 모사합니다.'}, {'Model Robustness Enhancement': '기존의 반사실 증강 방법은 여전히 스푸리어스 상관관계(spurious correlation)에 영향을 받습니다. 본 연구는 여러 개의 반사실을 생성하여 집합적인 의사 결정을 통해 각 단어의 인과관계를 더 강력하게 감독하는 방법을 제안합니다.'}, {'MLLM Hallucination Evaluation': 'MLLM(multi-modal large language models)의 헛소리를 평가하기 위해 Hallu-PI 벤치마크를 도입합니다. 이는 7가지 변형된 시나리오와 1,260개의 변형된 이미지를 포함하여 다양한 MLLM의 생성 및 판별 작업에서 헛소리를 평가합니다.'}]

- **Performance Highlights**: [{'KDA Metrics': 'Human 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 높은 상관관계를 가지며, 다양한 전문가 평가 MCQ 품질 지표에 대해 강력한 예측력을 가짐을 보여줍니다.'}, {'Robustness Improvement': '제안된 방법은 반사실 생성에서의 집합적 의사 결정 덕분에 기존의 비강건한 속성 기반 생성 방법에 비해 다양한 차원에서 강건성을 크게 향상시킵니다.'}, {'Hallu-PI 벤치마크': 'GPT-4V와 Gemini-Pro Vision을 포함한 12개의 최신 MLLM이 변형된 시나리오에서 큰 헛소리를 보이며, 기존의 비변형된 시나리오에서 관찰되지 않은 현상을 발견했습니다. Perturbed-Reminder와 Perturbed-ICL이라는 두 개의 베이스라인을 설계하여 GPT-4V의 헛소리를 효과적으로 감소시켰습니다.'}]



### StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation (https://arxiv.org/abs/2408.01343)
- **What's New**: 교사들이 학생 평가를 위해 MCQ(다중선택질문)를 자동으로 생성하는 방법에 있어, 교육적 가치를 고려하지 않는 기존 평가 지표들(BLEU, ROUGE, METEOR)를 보완하기 위해 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 또한, NLP의 deep model이 높은 정확도를 가지지만, spurious pattern에 의존해 robustness가 제한된 문제점을 보완하기 위해 contrastive learning와 counterfactual augmentation을 활용하는 방법을 제안합니다. 마지막으로, StitchFusion이라는 기존 사전 훈련된 모델을 그대로 활용하면서 정보 교환을 가능하게 하는 다중 모달 융합 프레임워크가 소개되었습니다.

- **Technical Details**: KDA는 학습자의 문제 해결 과정을 모방하기 위해 사전 학습된 언어 모델을 활용하여 KDA_disc와 KDA_cont로 평가되는 새로운 방식입니다. 또한, counterfactual augmentation에서는 '여러 개의' counterfactual을 생성하고 집합적 의사결정을 통해 robust한 인과관계를 유지하는 방법을 제안합니다. StitchFusion은 사전 훈련된 모델을 그대로 인코더 및 피처 퓨저로 사용해 다중 모달 정보를 융합시키는 프레임워크로, MultiAdapter라는 다방향 어댑터 모듈을 소개하여 인코딩 과정 중 모달 정보의 교환을 강화합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅과 높은 상관관계를 가지며, 다양한 전문가 레이블 MCQ 품질 측정에 대해 예측력이 강함을 확인했습니다. 제안된 counterfactual 방법은 기존의 attribution 기반 방법보다 모델 편향에 덜 민감하며 다양한 차원에서 significant improvements를 달성했습니다. StitchFusion 모듈은 기존 FFMs와 상호 보완적인 성격을 가지며, 네 가지 다중 모달 분할 데이터셋에서 최첨단 성능을 달성했습니다.



### A Robotics-Inspired Scanpath Model Reveals the Importance of Uncertainty and Semantic Object Cues for Gaze Guidance in Dynamic Scenes (https://arxiv.org/abs/2408.01322)
Comments:
          35+16 pages, 8+4 figures

- **What's New**: 최근 심층 모델(deep models)은 NLP 작업에서 인간 이상의 정확성을 보였음에도 불구하고, spurious patterns에 의존하여 robust하지 못한 문제를 가지고 있습니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용한 새로운 접근 방식을 제안합니다. 또한, 교육적인 가치를 평가할 수 있는 새로운 자동 평가 지표인 KDA (Knowledge Dependent Answerability)를 도입하여 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가할 수 있는 능력을 평가하는 방법을 발표하였습니다.

- **Technical Details**: 논문에서 제안한 방법은 여러 개의 반사실적 사례를 생성하고, 집합적인 의사 결정(collective decisions)을 통해 모델의 인과학적 해석을 강화하는 것입니다. 또한, 새로운 지표 KDA는 학생의 응답 결과를 기준으로 평가하며, 학생의 문제 해결 행동을 모방하는 사전 훈련된 언어 모델을 활용해 자동 평가 기준인 KDA_disc와 KDA_cont를 고안했습니다. 이는 기존의 BLEU, ROUGE, METEOR 지표와 달리 교육적 가치를 평가할 수 있다는 점이 특징입니다.

- **Performance Highlights**: 사람 대상 연구를 통해 새로운 평가 지표 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가짐을 보였습니다. 또한, 다양한 전문가가 라벨링한 MCQ 품질 측정과 강한 예측력을 가짐을 확인하였으며, 집합적인 의사 결정방식의 반사실적 증강 방법이 1) 반사실적 강건성, 2) 도메인 간 일반화, 3) 희박한 데이터에서의 일반화에서 현저한 성능 향상을 이루었습니다.



### TopoNAS: Boosting Search Efficiency of Gradient-based NAS via Topological Simplification (https://arxiv.org/abs/2408.01311)
- **What's New**: 최근 여러 자연어 처리(NLP) 작업에서 초인간 정확도를 달성하는 딥 모델(deep models)이 있음에도 불구하고, 이 모델들의 강건성(robustness)이 스퓨리어스 패턴(spurious patterns)에 의존해 제한된다는 문제가 있다. 이를 해결하기 위해 대조 학습(contrastive learning)과 반대사실(카운터팩추얼) 증강(counterfactual augmentation)을 활용하는 새로운 접근 방식을 제안한다.

- **Technical Details**: 기존의 자동 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR가 교육적 가치를 고려하지 않는 한계를 극복하고자, 목표 사실에 대한 학생의 지식을 평가하는 능력을 측정하는 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안한다. 또한, 반대사실 증강을 위해 기존 연구와 달리, 여러 개의 반대사실을 생성하고 이를 통해 집합적 의사 결정을 내리는 방식을 사용하여 인과관계를 더 견고하게 감독한다.

- **Performance Highlights**: 제안된 KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 나타냈으며, 다양한 전문가 레이블 MCQ 품질 판단 기준에 대해 높은 예측력을 보였다. 또한, 이는 다양한 데이터셋에서 반대사실 강건성, 교차 도메인 일반화, 데이터 부족 상황에서의 일반화 등 여러 차원에서 상당한 개선을 달성했다.



### Underwater Object Detection Enhancement via Channel Stabilization (https://arxiv.org/abs/2408.01293)
- **What's New**: 최근 MCQ 자동 생성기와 관련하여, 기존 평가 메트릭들이 교육적 가치를 고려하지 않는다는 문제점이 발견되었습니다. 이에 우리는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였으며, 이를 통해 MCQ의 대답 가능성과 학습 평가 능력을 측정할 수 있습니다.

- **Technical Details**: KDA는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont로 구체화되었습니다. 이 두 메트릭은 미리 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하고, Human survey를 통해 수집된 데이터를 바탕으로 개발되었습니다. 또한, 우리의 방법은 BLEU, ROUGE, METEOR 등의 기존 메트릭과 병행하여 더 강력한 예측 성능을 발휘할 수 있음을 보여주었습니다.

- **Performance Highlights**: Human study를 통해 KDA_disc 및 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였으며, 전문가들의 라벨링 기준에서도 높은 예측 성능을 보였습니다.



### TexGen: Text-Guided 3D Texture Generation with Multi-view Sampling and Resampling (https://arxiv.org/abs/2408.01291)
Comments:
          European Conference on Computer Vision (ECCV) 2024

- **What's New**: 교사가 학생 평가하는 시간을 줄이는 자동 다지선다형 질문(MCQ) 생성에 대한 새로운 평가 메트릭인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'을 제안합니다. KDA는 학생의 문제 해결 행동을 흉내 내는 사전 학습된 언어 모델을 활용하여 MCQ의 대답 가능성(answerability)을 측정합니다.

- **Technical Details**: KDA는 학생 반응의 인간 조사 기반 측정을 통해 평가되며, 이를 자동으로 근사한 KDA_disc와 KDA_cont 두 가지 메트릭을 제안합니다. KDA_disc와 KDA_cont는 모두 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다. 이는 BLEU, ROUGE, METEOR 등 기존의 n-gram 기반 평가 메트릭이 지식 평가 능력을 무시하는 결점을 보완합니다.

- **Performance Highlights**: 인간 평가 연구를 통해 KDA_disc와 KDA_cont는 실제 강의 환경에서의 사용성과 강한 상관관계가 있음을 확인했습니다. 이 두 메트릭은 전문 제시된 다양한 MCQ 품질 평가 지표에도 높은 예측 능력을 보여줬습니다.



### Wave-Mamba: Wavelet State Space Model for Ultra-High-Definition Low-Light Image Enhancemen (https://arxiv.org/abs/2408.01276)
Comments:
          10 pages, 8 figures, ACMMM2024 accepted

- **What's New**: 자동 생성된 다중 선택 질문(MCQ)의 평가에서 교육적 가치를 고려하는 새로운 메트릭 KDA(Knowledge Dependent Answerability)를 제안합니다. 또한, 초고해상도(UHD) 이미지를 저조도 이미지 향상(LLIE)를 위한 Wave-Mamba라는 새로운 방법을 제안합니다.

- **Technical Details**: MCQ에 있어, 기존 평가 메트릭은 n-gram 기반의 유사성에 초점을 맞추었으나 이는 교육적 가치를 반영하지 못했습니다. KDA는 학생의 답변 가능성을 측정하여 이를 해결합니다. 또한, 저조도 UHD 이미지 향상을 위해 wavelet 변환을 사용하여 정보 손실 없이 이미지 내용을 분리하고, SSM의 장점을 활용한 LFSSBlock과 HFEBlock을 설계했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성에 강한 상관관계를 보였으며, Wave-Mamba는 기존 초고해상도 LLIE 방법보다 뛰어난 성능을 발휘하면서도 더 간소화된 아키텍처를 유지하는 결과를 보여줬습니다.



### A General Framework to Boost 3D GS Initialization for Text-to-3D Generation by Lexical Richness (https://arxiv.org/abs/2408.01269)
- **What's New**: 교육 평가를 위한 다중 선택 질문 (MCQ)의 자동 생성 메커니즘과 그 평가 방법에 대한 연구가 새로운 단계에 접어들었습니다. MCQ의 대답 가능성(answerability)을 측정하는 새로운 자동 평가 메트릭인 '지식 종속 가능성(KDA)'이 제안되었습니다. 이는 기존 BLEU, ROUGE, METEOR 메트릭이 MCQ의 교육적 가치를 평가하지 못하는 단점을 보완합니다.

- **Technical Details**: KDA 메트릭은 두 가지 자동화된 평가 메트릭, KDA_disc와 KDA_cont를 포함하며, 사전 훈련된 언어 모델을 이용하여 학생들의 문제 해결 행동을 모방함으로써 KDA를 근사합니다. 이를 통해 KDA 메트릭은 학생이 해당 사실에 대한 이해도에 기반하여 MCQ의 대답 가능성을 평가합니다. 인간 평가를 통해 KDA_disc와 KDA_cont의 유효성을 입증하였고, 이 메트릭들은 전문가들이 레이블링한 실제 교실 환경에서의 MCQ 품질 측정과 강한 상관관계를 가집니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 n-gram 유사성 메트릭과 결합될 때, 다양한 전문가 레이블 MCQ 품질 측정값에 대해 강력한 예측력을 가집니다. 이는 새로운 메트릭이 기존 방법보다 더 신뢰할 수 있음을 의미합니다.



### CLIP4Sketch: Enhancing Sketch to Mugshot Matching through Dataset Augmentation using Diffusion Models (https://arxiv.org/abs/2408.01233)
- **What's New**: 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하여 MCQ(지문형 문제)의 대답 가능성을 측정하고, 학생이 관련 대상으로부터 얻은 지식을 평가하는 능력을 강화했습니다. 또한, 최근 NLP 태스크에서 딥러닝 모델들이 비약적 발전을 이루었지만, spurious 패턴에 의존하는 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation를 활용하는 새로운 방법론이 제안되었습니다. 마지막으로, 법의학 스케치와 얼굴 사진 매칭 문제를 해결하기 위해 CLIP4Sketch라는 새로운 접근법을 제안하여, 다양한 스케치 생성으로 얼굴 인식 시스템의 성능을 향상시키는 방법을 제시했습니다.

- **Technical Details**: KDA는 학생 응답 기반의 인간 설문조사를 통해 MCQ의 대답 가능성을 측정합니다. 자동 평가 메트릭 KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. NLP 태스크의 경우, '여러 개'의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악합니다. CLIP4Sketch는 Denoising Diffusion Probabilistic Models(DDPMs)를 사용해 신원(identity)과 스타일에 대한 명확한 제어를 통해 스케치를 생성합니다. 이 방법은 CLIP 및 Adaface 임베딩을 활용해 텍스트 설명과 얼굴 사진을 조건으로 사용합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였습니다. 새로운 접근법은 대조 학습(contrastive learning)과 counterfactual augmentation을 통해 다양한 차원에서의 성과(1. counterfactual robustness, 2. cross-domain generalization, 3. generalization from scarce data)를 향상시켰습니다. CLIP4Sketch는 기존의 GAN 기반 방법과 비교하여 스케치-얼굴 사진 매칭 정확도가 크게 개선되었습니다. 약 2.7만 개의 독특한 신원을 포함한 합성 데이터셋이 생성되었으며, 기존의 실제 얼굴 스케치 데이터셋과 비교해 우수한 성능을 보였습니다.



### WaveMamba: Spatial-Spectral Wavelet Mamba for Hyperspectral Image Classification (https://arxiv.org/abs/2408.01231)
- **What's New**: 이 연구는 자동으로 생성되는 MCQ(Multiple Choice Questions)의 평가 메트릭이 n-gram 기반의 유사성에만 집중되어 교육적 가치를 충분히 반영하지 못한다는 문제를 해결하기 위해, Knowledge Dependent Answerability(KDA)라는 새로운 자동 평가 메트릭을 제안합니다. 또한, Contrastsive Learning과 Counterfactual Augmentation을 활용한 NLP 모델의 robustness 향상 방법도 소개되었습니다. 마지막으로, WaveMamba라는 새로운 HSI(Hyperspectral Imaging) 분류 방법을 제시하여, Wavelet 변환과 Mamba 아키텍처의 결합을 통해 더 나은 성능을 보여줍니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반의 유사성만을 평가하는데, 새로운 KDA 메트릭은 학생이 특정 사실에 대한 지식을 가지고 있을 때의 답변 가능성을 평가합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서 높은 사용성을 보였습니다.', 'NLP Robustness': '최근 deep models은 NLP 태스크에서 사람보다 나은 정확성을 보였으나 spurious pattern에 의존하는 문제를 가집니다. 기존 방법들은 spurious correlation에 영향을 받는 반면, 본 연구는 여러 counterfactuals를 생성하여 집합적 의사 결정을 하는 방법을 제안하며 이를 통해 더 robust하게 단어들의 인과관계를 파악합니다.', 'HSI Classification': 'WaveMamba는 Wavelet 변환을 Spatial-Spectral Mamba 아키텍처와 통합하여 HSI 데이터를 다루는 방법입니다. 이 접근법은 HSI 데이터의 국소 텍스처 패턴과 전역적 컨텍스트 관계를 캡처합니다. 실험 결과, University of Houston 및 Pavia University 데이터셋에서 기존 모델보다 더 높은 정확성을 기록했습니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont 메트릭은 실제 강의실 세팅에서 사용성 및 전문가가 라벨링한 MCQ 품질 측정치들과 강한 상관관계를 보였습니다.', 'NLP Robustness': '제안된 방법은 attribution-based synthesis의 모델 편향에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data의 일반화에서 큰 성능 향상을 달성했습니다.', 'HSI Classification': 'WaveMamba는 University of Houston 데이터셋에서 4.5%, Pavia University 데이터셋에서 2.0%의 정확성 향상을 달성했습니다.'}



### The Phantom Menace: Unmasking Privacy Leakages in Vision-Language Models (https://arxiv.org/abs/2408.01228)
- **MCQ Generation**: {"What's New": "기존의 n-gram 유사성 메트릭인 BLEU, ROUGE, METEOR는 교육적 가치를 평가하지 못하기 때문에, 새로운 자동 평가 메트릭 '지식 종속 가능성(knowledge dependent answerability, KDA)'를 제안하였다. KDA는 학생이 대상 사실(target fact)에 대한 지식을 가지고 MCQ에 답할 수 있는 능력을 평가한다.", 'Technical Details': '학생 응답 기반으로 KDA를 측정한 후, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였다. 이를 통해 MCQ의 대답 가능성을 평가한다.', 'Performance Highlights': '인간 연구에서 KDA_disc와 KDA_cont가 실제 강의실 사용성과 강한 상관관계를 가지며, 전문가가 레이블링한 다양한 MCQ 품질 지표에 대해서도 예측력이 높음을 보여주었다.'}

- **Contrastive Learning for Robustness**: {"What's New": '최근 NLP 모델들은 높은 정확성을 보이지만, 비정상적인 패턴(spurious patterns)에 의존하여 robustness가 제한된다. 이를 해결하기 위해, 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 방안을 제안하였다.', 'Technical Details': '기존 방법들은 사람이나 머신을 통해 데이터셋에서 counterfactual을 추가하였으나, 이 논문에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 robust하게 단어들의 인과관계를 평가하는 방법을 제시하였다.', 'Performance Highlights': '제안된 방법은 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data의 일반화에서 현저한 성능 향상을 보였다.'}

- **VLMs Privacy Concerns**: {"What's New": 'Vision-Language Models(VLMs)가 개인정보 유출의 위험을 지니고 있는지 평가한 결과, 익명 데이터로 훈련된 경우에도 신원 정보가 유출됨을 발견하였다.', 'Technical Details': '대표적인 VLMs인 BLIP-2, LLaVA, PaliGemma 등을 대상으로 신원 유출을 평가하였으며, 블러링과 같은 간단한 익명화(Anonymization) 기술이 신원 유출을 막지 못함을 확인하였다.', 'Performance Highlights': '익명화된 데이터로 훈련하더라도 여전히 신원 정보가 유출되어 강력한 개인정보 보호 전략이 필요하며, 윤리적 인식과 책임 있는 개발이 강조된다.'}



### Multi-head Spatial-Spectral Mamba for Hyperspectral Image Classification (https://arxiv.org/abs/2408.01224)
- **What's New**: 이번 논문은 자동 MCQ(Multiple Choice Questions) 생성을 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)을 제안합니다. 이 메트릭은 기존의 BLEU, ROUGE, METEOR 메트릭들이 대답 가능성(answerability)과 교육적 가치를 고려하지 않는 문제를 개선하고자 고안되었습니다. 또한, NLP 태스크의 robustness를 강화하기 위해 대조학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법도 소개됩니다. 이 논문은 또한 Hyperspectral Image (HSI) 데이터를 분석하기 위해 다중 헤드 자기 주의 (Multi-Head Self-Attention)을 사용한 Spatial-Spectral Mamba (SSM) 모델을 제안합니다.

- **Technical Details**: 먼저, KDA는 대상 사실(target fact)에 대한 학생의 지식을 평가하는 새롭고 자동화된 평가 메트릭입니다. KDA는 인간 설문조사를 기반으로 측정되며, 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 통해 사전 학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방함으로써 이 메트릭을 근사화합니다. 한편, NLP 태스크의 향상을 위해 이 논문은 여러 개의 반사실적(counterfactual)을 생성하고, 집합적 의사 결정을 통해 각 용어의 인과성을 감독함으로써 모델의 robust를 증가시키는 방법론을 제시합니다. 마지막으로, MHSSMamba 모델은 주의 메커니즘을 통해 공간적(spatial) 및 스펙트럼적(spectral) 정보를 통합하고 고차원 HSI 데이터의 long-range dependencies를 처리합니다.

- **Performance Highlights**: MHSSMamba 모델은 Pavia University 데이터셋에서 97.62%, University of Houston에서 96.92%, Salinas에서 96.85%, 그리고 Wuhan-longKou 데이터셋에서 99.49%의 놀라운 분류 정확도를 달성했습니다. 또한, KDA 기반 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, 전문가들이 라벨한 다양한 MCQ 품질 측정에 대해 예측력이 강력하다는 것을 입증했습니다.



### S2TD-Face: Reconstruct a Detailed 3D Face with Controllable Texture from a Single Sketch (https://arxiv.org/abs/2408.01218)
Comments:
          ACM MM 2024

- **What's New**: 자동 MCQ(객관식 문제) 생성과 관련하여, 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안했습니다. 또한, 다양한 종류의 스케치를 기반으로 한 3D 텍스처 얼굴 재구성 방법인 S2TD-Face를 소개하며, 이를 통해 기존의 제한된 방법들을 넘어서 다양한 텍스처 제어가 가능한 재구성 방법을 제공합니다.

- **Technical Details**: MCQ 생성에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다. S2TD-Face에서는 두 단계의 기하학적 재구성 프레임워크를 도입하며, 텍스처 제어 모듈을 통해 텍스트 프롬프트 기반으로 텍스처를 선택하고 적용합니다.

- **Performance Highlights**: MCQ 생성에서 KDA_disc와 KDA_cont는 전문가가 레이블링한 MCQ 품질 측정에 대해 강력한 예측력을 보였습니다. S2TD-Face는 광범위한 정량적 및 정성적 실험에서 기존의 최첨단 방법보다 뛰어난 성능을 발휘했으며, 특히 스케치 스타일과 무관하게 높은 정확도의 3D 재구성을 구현했습니다.



### A Weakly Supervised and Globally Explainable Learning Framework for Brain Tumor Segmentation (https://arxiv.org/abs/2408.01191)
Comments:
          2024 IEEE International Conference on Multimedia and Expo

- **What's New**: 최근 자동으로 Multiple Choice Questions(MCQ)를 생성하는 데에 있어서 교육적 가치를 평가하는 새로운 메트릭을 도입하는 연구가 발표되었습니다. 기존의 BLEU, ROUGE, METEOR와 같은 메트릭은 단순히 단어의 일치성을 기준으로 MCQ의 품질을 평가하지만, 새로운 메트릭인 Knowledge Dependent Answerability(KDA)는 학생들이 실제로 문제를 풀이할 수 있는지 여부를 측정합니다.

- **Technical Details**: KDA는 학생들이 특정 사실에 대한 지식을 바탕으로 문제를 맞힐 수 있는지 평가하는 자동 평가 메트릭입니다. KDA_disc와 KDA_cont와 같은 두 가지 자동 평가 메트릭이 제안되었으며, 이들은 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모사합니다. 또한, 이 연구는 KDA_disc와 KDA_cont가 사람들을 대상으로 한 평가와 강한 상관관계를 가지며, 실제 교육 현장에서의 사용성을 입증했습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서 사용성이 높고 KDA와 강한 상관관계를 가지는 것이 입증되었습니다. 특히, n-gram 기반 유사성 메트릭과 결합했을 때 KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강한 예측력을 갖는 것으로 나타났습니다.



### VAR-CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling (https://arxiv.org/abs/2408.01181)
Comments:
          total 10 pages, code:this https URL

- **What's New**: 최근 논문에서 자동 MCQ(Multiple Choice Questions) 생성을 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하였습니다. 이 메트릭은 MCQ의 교육적 가치를 평가하여 학생의 지식을 효과적으로 테스트하는 능력을 확인합니다. 또한, NLP 태스크에서 deep model의 robustness를 증가시키기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법론을 제안하였습니다. 끝으로, VAR-CLIP이라는 혁신적인 텍스트-이미지 생성 모델이 소개되었으며, 이는 Visual Auto-Regressive 기법을 CLIP과 통합하여 고품질의 이미지를 생성합니다.

- **Technical Details**: KDA는 학생들의 응답 데이터를 바탕으로 MCQ의 대답 가능성을 측정하는 방식입니다. 또한, KDA_disc와 KDA_cont 라는 두 가지 자동 평가 메트릭을 도입하여 학생들의 문제 해결 행동을 모방하도록 설계되었습니다. NLP 모델의 robustness 개선 방법론은 기존 counterfactual augmentation 방식이 가지고 있는 spurious correlation 문제를 해결하고자 여러 개의 counterfactual을 생성하여 더 강력한 인과관계 파악을 목표로 합니다. 한편, VAR-CLIP는 CLIP 텍스트 인코더로 캡션을 텍스트 임베딩으로 변환한 후, 그것을 조건으로 사용하여 이미지를 생성하는 방식으로 동작합니다. 두 단계의 트레이닝 전략(멀티-스케일 VQVAE/VQGAN 트레이닝, 클립 텍스트 인코딩 후 이미지 생성)을 채택하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 전문가가 지정한 다양한 MCQ 품질 지표와 강한 상관 관계를 보였습니다. 또한, 자연어 처리 모델을 개선한 방법론은 counterfactual robustness, cross-domain generalization, 그리고 제한된 데이터에서의 generalization에서 현저한 성능 향상을 보였습니다. VAR-CLIP는 뛰어난 이미지 성과, 높은 텍스트 일치도, 그리고 미적인 우수성을 갖춘 이미지를 생성하는 데 성공하였으며, ImageNet과 같은 대규모 데이터셋에서 텍스트-이미지 태스크를 지원할 수 있습니다.



### Rethinking Pre-trained Feature Extractor Selection in Multiple Instance Learning for Whole Slide Image Classification (https://arxiv.org/abs/2408.01167)
Comments:
          12 pages

- **What's New**: 논문은 자동 MCQ 생성의 교육적 가치를 평가하는 새로운 메트릭 Knowledge Dependent Answerability (KDA)를 제안한다. 또한, NLP 태스크의 robustness를 향상시키기 위해 contrastive learning과 counterfactual augmentation을 활용하는 새로운 접근법을 제안하고, 최적의 feature extractor 선택을 위한 포괄적 분석으로 MIL 모델의 성능을 최대화하는 방법을 탐구한다.

- **Technical Details**: 1. KDA는 MCQ의 답변 가능성을 평가하며, KDA_disc와 KDA_cont라는 새로운 자동 평가 메트릭을 통해 이를 측정한다. 2. NLP 모델에 대해 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 robustness를 증진시키는 방법을 제안한다. 3. MIL (Multiple Instance Learning) 접근법에서 더 큰 사전 학습 데이터셋과 모던하며 깊은 백본(backbone)의 중요성을 강조하며, SSL (Self-Supervised Learning) 방법을 통한 최적의 pre-trained 모델 선택 방법을 분석한다.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 학급 환경에서 높은 상관관계를 보이며, n-gram 기반의 유사성 메트릭과 결합할 경우 다양한 전문가 라벨링 MCQ 품질 측정치에 대해 높은 예측 능력을 보인다. 2. 새로운 접근법은 다양한 차원에서의 향상된 성능, 특히 counterfactual robustness와 cross-domain generalization에 있어 우수한 결과를 보였다. 3. MIL 모델들은 더 크고 다양한 사전 학습 데이터셋을 사용하여 성능이 크게 개선되었고, transformer 기반 백본이 우수한 성능을 보이며, SSL 방법을 적용한 경우 가장 큰 이점을 얻었다.



### PreMix: Boosting Multiple Instance Learning in Digital Histopathology through Pre-training with Intra-Batch Slide Mixing (https://arxiv.org/abs/2408.01162)
Comments:
          15 pages

- **What's New**: 이번 뉴스레터에서는 최신 아카이브 논문 세 편을 요약했습니다. 첫 번째 논문은 다중 선택 질문(MCQ) 생성에서 BLEU, ROUGE, METEOR 같은 기존 평가 지표의 한계를 극복하고, 학생의 지식을 보다 효과적으로 평가하는 새로운 자동 평가 메트릭 '지식 종속 가능성(KDA)'을 제안합니다. 두 번째 논문은 최근 딥 모델의 강인성 (robustness)이 spurious patterns에 의해 제한되는 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 활용하는 방법을 다룹니다. 마지막으로 세 번째 논문은 병리학 이미지의 분류에서 MIL(multiple instance learning) 프레임워크를 개선한 PreMix를 제안하며, unlabeled WSI(Whole Slide Images)를 보다 효율적으로 활용하는 방법을 소개합니다.

- **Technical Details**: {'MCQ Generation': '지식 종속 가능성(KDA)은 학생 응답을 기반으로 MCQ의 대답 가능성을 측정하는 새로운 평가 메트릭입니다. 자동 평가 메트릭으로 KDA_disc와 KDA_cont를 제안하며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Robustness in NLP': "대조 학습과 반사실 증강을 활용하여 deep model의 강인성을 높이는 방법을 제안합니다. 새로운 접근법에서는 '다수의' 반사실들을 생성하고, 집합적 의사 결정을 통해 각 단어의 인과성을 보다 견고하게 감독합니다.", 'WSI Classification with PreMix': 'PreMix는 MIL 프레임워크를 확장하여 intra-batch slide mixing 기법을 적용하여 MIL 집계기를 사전 학습합니다. Barlow Twins Slide Mixing을 활용하며, 이후 fine-tuning 과정에서 Mixup 및 Manifold Mixup을 통합하여 성능을 향상시킵니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 측정 지표와 결합할 때 전문가가 라벨링한 다양한 MCQ 품질 측정 기준에 대한 예측력이 높았습니다.', 'Robustness in NLP': '제안된 방법은 반사실 강인성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 및 데이터 부족 상황에서의 일반화에서 상당한 향상을 보였습니다.', 'WSI Classification with PreMix': 'PreMix는 Camelyon16 데이터셋에서 기본 MIL 프레임워크인 HIPT에 비해 평균 4.7%의 성능 향상을 보여주었습니다. 이는 다양한 액티브 학습 획득 함수와 WSI 라벨링 예산을 아우르는 데이터셋에 대한 적응력을 강조합니다.'}



### Robust Curve Detection in Volumetric Medical Imaging via Attraction Field (https://arxiv.org/abs/2408.01159)
Comments:
          Accepted to ShapeMI MICCAI 2024

- **What's New**: AI를 이용한 의료 영상에서 비분기 곡선을 감지하는 새로운 방법을 제안하였습니다. 이는 기존의 방법보다 높은 정확도로 다양한 임상 과제를 처리할 수 있습니다.

- **Technical Details**: 이 방법은 3D의 두 개의 헤드를 가진 CNN 아키텍처를 기반으로 하며, 'attraction field'와 'closeness map'을 예측합니다. 'Attraction field'는 곡선에 대한 위치를 서브픽셀 수준의 정확도로 예측하고, 'closeness map'은 관심 영역을 제한하여 외부 간섭을 줄입니다.

- **Performance Highlights**: 임상적으로 중요한 여러 과제에서 기존 방법을 능가하는 서브픽셀 수준의 정확도를 달성하였습니다. 또한 대동맥 중심선 및 척추 중심선 감지 작업에서 우수한 성능을 보였습니다.



### PGNeXt: High-Resolution Salient Object Detection via Pyramid Grafting Network (https://arxiv.org/abs/2408.01137)
- **What's New**: 이 논문에서는 MCQ(Multiple Choice Questions) 자동 생성의 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. KDA는 단순히 n-gram 기반의 유사성을 평가하는 기존 메트릭과 달리, 학생의 지식을 평가하는 MCQ의 교육적 가치를 측정합니다. 또한, 우리의 연구에서는 contrastive learning과 counterfactual augmentation을 활용한 모델의 robust함을 높이기 위한 새로운 접근법을 제안하였습니다. 마지막으로, 고해상도 주목 객체 검출(HRSOD)을 위한 새로운 데이터셋(UHRSD)을 구축하고, 폴리곤 기반의 네트워크 프레임워크 (Pyramid Grafting Network, PGNet)도 소개하였습니다.

- **Technical Details**: KDA는 학생 설문 조사를 통해 측정할 수 있으며, 이를 기반으로 ML 모델을 활용해 자동으로 평가할 수 있는 KDA_disc와 KDA_cont라는 두 가지 메트릭을 제안합니다. 모델의 robust함을 높이기 위해 여러 개의 counterfactual을 생성하고, 집합적 의사결정(collective decision) 기법을 사용하여 단어들의 인과관계를 감독하는 방법을 도입하였습니다. HRSOD를 위해 UHRSD라는 고해상도 데이터셋과 함께 Transformer와 CNN 기반의 이중 엔코더를 사용한 Pyramid Grafting Mechanism을 기반으로 한 PGNet을 제안하였습니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 수업 환경에서도 높은 상관관계를 지니며, 다른 n-gram 기반 유사성 메트릭과 결합 시 전문가가 라벨링한 MCQ 품질 측정 지표에 대해 예측력이 뛰어남을 확인하였습니다. 새로운 접근법은 다차원에서 상당한 개선을 보여주었으며, 특히 다른 도메인에 대한 일반화와 데이터가 부족한 상황에서의 성능 향상을 입증했습니다. 마지막으로, 제안된 PGNet 모델은 고해상도 입력을 처리하면서도 높은 정확도와 속도를 유지하며, 가장 최신의 SOD 및 COD 방법들보다 우수한 성능을 발휘하였습니다.



### IG-SLAM: Instant Gaussian SLAM (https://arxiv.org/abs/2408.01126)
Comments:
          8 pages, 3 page ref, 5 figures, 3DV submission

- **What's New**: 이번 뉴스레터에서는 최근 AI 및 교육 분야에서 주목받고 있는 두 가지 연구 및 한 가지 3D 재구성 방법론에 대해 소개하겠습니다. 첫 번째는 자동 생성 다중 선택 질문(MCQ) 평가 메트릭의 개선, 두 번째는 자연어 처리(NLP) 모델의 견고성 향상을 위한 대조 학습(contrastive learning)과 반대사실 증가(counterfactual augmentation) 기법, 그리고 마지막으로는 SLAM 시스템에서의 3D 가우시안 스플래팅(Gaussian Splatting) 기법입니다.

- **Technical Details**: [{'title': '자동 MCQ 평가 메트릭', 'description': "전통적인 MCQ 평가 메트릭(BLEU, ROUGE, METEOR)은 n-gram 기반의 유사성에 중점을 두며 교육적 가치를 고려하지 않습니다. 이를 해결하기 위해 '지식 종속 가능성(KDA)'이라는 새로운 자동 평가 메트릭을 제안합니다. KDA는 학생이 특정 지식을 알고 있을 경우 질문에 답할 수 있는지 여부를 평가합니다. KDA_disk와 KDA_cont라는 두 가지 자동 평가 방식을 제안하며, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 방식을 모방합니다."}, {'title': 'NLP 모델의 견고성 향상을 위한 대조 학습', 'description': '최신 NLP 모델들은 높은 정확도를 보이지만, spurious pattern에 의존하는 경향이 있어 견고성이 떨어지곤 합니다. 이를 개선하기 위해 대조 학습과 counterfactual augmentation 기법을 사용합니다. 기존 방법들은 사람이 추가한 counterfactual 데이터를 사용하거나 모델이 데이터셋에서 유사한 항목을 찾는 방식이었으나, 새로운 방법론은 여러 개의 counterfactual을 생성하고 집합적 의사결정(collective decision)을 통해 각 단어의 인과관계를 더욱 견고하게 분석합니다.'}, {'title': 'IG-SLAM: 3D 가우시안 스플래팅을 활용한 실시간 RGB-SLAM 시스템', 'description': 'IG-SLAM은 최신 SLAM 시스템 중 하나로, RGB 이미지만을 사용하여 고속의 실시간 3D 재구성을 수행합니다. 이 시스템은 정확한 포즈 추정과 정교한 깊이 정보를 제공하며, 가우시안 스플래팅 기법을 사용하여 지도(map)를 생성합니다. 또한, 깊이 불확실성을 고려하여 3D 재구성을 최적화하며 지도 최적화 과정에서의 수렴성을 향상시켜 초당 10프레임의 속도로 실행됩니다.'}]

- **Performance Highlights**: [{'title': '자동 MCQ 평가 메트릭', 'description': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지는 것으로 확인되었습니다. n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 평가 MCQ 품질 메트릭에 대한 예측력이 강화되었습니다.'}, {'title': 'NLP 모델의 견고성 향상', 'description': '제안된 방법론은 counterfactual robustness, cross-domain generalization, 그리고 희소한 데이터로부터의 일반화에서 중요한 향상을 달성하였습니다.'}, {'title': 'IG-SLAM', 'description': 'IG-SLAM은 Replica, TUM-RGBD, ScanNet, EuRoC 등 다양한 데이터셋에서 실험을 통해 경쟁력 있는 성능을 입증했습니다. 특히 EuRoC 데이터셋에서 매우 사실적인 3D 재구성을 실현하였습니다.'}]



### An Efficient and Effective Transformer Decoder-Based Framework for Multi-Task Visual Grounding (https://arxiv.org/abs/2408.01120)
Comments:
          21pages, 10 figures, 9 tables. Accepted to ECCV 2024

- **What's New**: 최근 연구 논문에서는 자동으로 주관식 문제(MCQ)를 생성하는 효율적인 방법을 제안했다. 이 방법은 교육자들이 학생 평가에 소비하는 시간을 줄이는 잠재력을 가지고 있으나, 기존의 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 평가하지 못한다는 문제를 가지고 있다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)이 제안되었다. KDA는 학생들이 대상 사실을 기반으로 문제를 답변할 수 있는 능력을 측정한다.

- **Technical Details**: 이 논문에서는 KDA를 평가하기 위해 학생 응답을 기반으로 한 측정 방법을 제안한다. 또한, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방함으로써 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다. 이를 통해 MCQ의 질을 더욱 정확하게 평가할 수 있다. 추가적으로 'counterfactual augmentation'을 사용하여 NLP 태스크의 로버스트니스를 향상시키는 방법도 제안되었다. 이 방법은 다수의 반사실(counterfactual)을 생성하고, 집합적 의사 결정을 통해 각 단어의 인과관계를 더욱 견고하게 판별한다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont 메트릭이 실제 강의실에서의 사용성과 높은 상관관계를 보이며, 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 가진 것으로 나타났다. 또한, EEVG(효율적이고 효과적인 멀티태스크 시각적 그라운딩) 프레임워크를 통해 기존 방법들보다 28.19% 더 빠른 성능을 보이며, 특히 긴 복잡한 언어 표현을 포함하는 데이터셋에서는 3.93%의 향상을 이루었다.



### Contribution-based Low-Rank Adaptation with Pre-training Model for Real Image Restoration (https://arxiv.org/abs/2408.01099)
Comments:
          33 pages, 15 figures, for homepage see this url : this https URL

- **What's New**: 이번 뉴스레터에서는 세 가지 새로운 인공지능 연구 논문을 소개합니다. 첫 번째 논문은 자동 다중 선택 질문(MCQ) 생성의 새로운 평가 메트릭을 제안합니다. 두 번째 논문은 대조 학습과 반사실적 증강(counterfactual augmentation)을 활용하여 NLP 모델의 견고성을 높이는 방법을 다룹니다. 마지막으로 세 번째 논문은 저수준 컴퓨터 비전에서 새로운 전이 학습(pre-training) 및 효율적인 파라미터 튜닝 방법을 제안하여 이미지 복원 성능을 향상시킵니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 등 평가 메트릭이 MCQ의 교육적 가치를 고려하지 못하는 문제를 해결하기 위해, 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 제안합니다. KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 이를 근사합니다.', 'Robustness in NLP Models': 'NLP 태스크에서 모델의 견고성을 높이기 위해 대조 학습 및 반사실적 증강을 활용합니다. 우리는 여러 개의 반사실적 데이터를 생성하고 이를 기반으로 집합적 의사 결정을 통해 단어들의 인과관계를 파악하는 방법을 제안합니다.', 'Image Restoration': '저수준 컴퓨터 비전에서 다양한 이미지 복원 작업을 위해 CoLoRA(contribution-based low-rank adaptation) 및 PROD(Pre-training with Random Order Degradation)라는 새로운 방법을 제안합니다. CoLoRA는 각 작업에 대해 적응적으로 계층(layer)별 용량을 결정하여 적은 수의 파라미터만을 튜닝합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 전문가에 의해 라벨링된 다양한 MCQ 품질 측정에서 강력한 예측력을 가지고 있음을 증명하였습니다.', 'Robustness in NLP Models': '우리의 접근법은 반사실적 견고성(Counterfactual Robustness), 크로스 도메인 일반화(Cross-domain generalization), 희소한 데이터에서의 일반화(Generalization from scarce data) 등 다양한 차원에서 기준 모델 대비 성능을 크게 개선하였습니다.', 'Image Restoration': 'CoLoRA와 PROD는 다양한 네트워크 아키텍처(CNN 및 Vision Transformers)와 함께 사용할 수 있으며, 실제 데이터 기반 6가지 이미지 복원 작업에서 최첨단 성능을 달성하였습니다.'}]



### Prototypical Partial Optimal Transport for Universal Domain Adaptation (https://arxiv.org/abs/2408.01089)
- **What's New**: 이 논문에서는 자동 다지선다 질문(MCQ) 생성의 평가를 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안했습니다. KDA는 학생의 지식에 기반한 대답 가능성을 측정하며, 데이터를 기반으로 한 기존 메트릭(BLEU, ROUGE, METEOR)의 한계를 보완합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도출하여 학생의 문제 해결 행동을 모방합니다. 이 메트릭은 사전 훈련된 언어 모델을 사용하여 추정됩니다. 또한, mini-batch Prototypical Partial Optimal Transport (m-PPOT)을 제안하여 부분적인 분포 정렬을 통해 '알려진' 샘플과 '알려지지 않은' 샘플을 구별합니다. 이 방법은 전송 계획을 활용하여 원본 프로토타입과 대상 샘플을 재가중치하여 엔트로피 손실 및 크로스 엔트로피 손실을 설계합니다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 실제 강의실 설정에서 사용성과 강한 상관관계를 보였으며, 여러 전문가가 라벨링한 품질 측정치에서 예측력이 뛰어남을 입증했습니다. 또한 m-PPOT은 네 가지 UniDA 벤치마크에서 기존 최첨단 방법을 능가하였으며, 개별 구성 요소의 효과를 검증하는 소거 연구를 통해 그 효용성을 확인했습니다.



### Effect of Fog Particle Size Distribution on 3D Object Detection Under Adverse Weather Conditions (https://arxiv.org/abs/2408.01085)
- **What's New**: 다양한 논문에서 새로운 연구 결과와 기술을 논의합니다. 첫 번째 논문은 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 다중 선택 질문(MCQ)의 교육적 가치를 더욱 잘 평가할 수 있게 합니다. 두 번째 논문은 대조 학습(contrastive learning)과 반대 사실 증강(counterfactual augmentation)을 통해 NLP 모델의 강건성(robustness)을 향상시키는 방법을 소개합니다. 마지막 논문은 악천후 조건에서 LiDAR 기반 3D 객체 탐지 성능을 분석하여 안개 입자 크기 분포가 시스템의 정확도에 미치는 영향을 조사합니다.

- **Technical Details**: 첫 번째 논문에서는 Pre-trained language models를 활용하여 KDA를 자동으로 측정하는 방법(KDA_disc 및 KDA_cont)을 제안합니다. 두 번째 논문은 집합적 의사 결정(collective decisions)을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 설명합니다. 세 번째 논문은 Mie 이론과 Meteorological Optical Range(MOR)를 사용하여 안개 조건에서 LiDAR 데이터의 감쇠 및 후방 산란 계수를 계산하고, PV-RCNN++ 딥 러닝 모델을 사용하여 다양한 탐지 난이도에서 Car, Cyclist, Pedestrian의 탐지 정확도를 분석합니다.

- **Performance Highlights**: 첫 번째 논문에서 제안된 KDA 기반 평가 메트릭은 기존의 n-gram 기반 유사성 메트릭과 결합하여 전문가들이 레이블을 붙인 MCQ의 품질 측정에서 강한 예측력을 보였습니다. 두 번째 논문에서는 제안된 방법이 다양한 차원에서 반대 사실 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 그리고 희소한 데이터로부터의 일반화에서 상당한 성능 향상을 달성했습니다. 세 번째 논문에서는 Car는 약 99%의 최고 정확도를 보였고, Pedestrian는 약 73%의 최저 정확도를 보였습니다.



### FCDFusion: a Fast, Low Color Deviation Method for Fusing Visible and Infrared Image Pairs (https://arxiv.org/abs/2408.01080)
Comments:
          This article has been accepted by Computational Visual Media

- **What's New**: 지식 종속 가능성(KDA)이라고 불리는 새로운 자동 평가 메트릭을 제안하여 MCQ(다중 선택 질문)의 대답 가능성(answerability)과 학생의 지식을 평가하는 능력을 측정함으로써 기존 메트릭의 한계를 보완하려는 노력을 보여줍니다.

- **Technical Details**: 이 논문에서는 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 데이터셋의 골드 샘플과의 단어 유사성에만 초점을 맞추어 교육적 가치를 무시하는 문제를 다룹니다. 이를 해결하기 위해 지식의 대답 가능성(KDA)을 측정하는 새로운 메트릭을 제안합니다. 구체적으로, 이는 학생 응답을 기반으로 KDA를 측정한 후, 사전 학습된 언어 모델을 활용하여 자동으로 이를 근사하는 KDA_disc와 KDA_cont를 도입합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 확립하였으며, n-gram 유사성 메트릭과 결합할 경우 다양한 전문가가 라벨링한 MCQ 품질 측정에서 강력한 예측력을 가짐을 보여줍니다.



### PhysMamba: Leveraging Dual-Stream Cross-Attention SSD for Remote Physiological Measuremen (https://arxiv.org/abs/2408.01077)
- **What's New**: Automatic Multiple Choice Question (MCQ) 생성에 대한 평가 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)가 제안되었습니다. KDA는 학생이 특정 사실에 대한 지식을 가지고 있을 때 MCQ에 답할 수 있는지 여부를 측정합니다. 기존 BLEU, ROUGE, METEOR와 같은 메트릭들은 n-gram 기반의 유사성만을 평가하는데 주로 사용되었으나, 이들은 교육적 가치를 반영하지 못합니다. 이 연구는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 학습된 언어 모델을 사용해 학생의 문제 해결 행동을 모방합니다.

- **Technical Details**: KDA는 학생 설문조사를 통해 측정되고, 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 이를 근사합니다. 이는 학생의 문제 해결 행동을 모방하여 MCQ의 교육적 가치를 평가합니다. 실험 결과, 이러한 자동 평가 메트릭은 실제 강의실 환경에서의 사용성과 밀접한 상관관계를 보였습니다.

- **Performance Highlights**: 인간 설문조사와 병행하며, KDA_disc와 KDA_cont는 강한 상관관계를 보였습니다. 또한, BLEU, ROUGE, METEOR과 같은 n-gram 기반 유사성 메트릭과 결합되었을 때, 전문가가 레이블한 다양한 MCQ 품질 지표에 대해 강력한 예측 능력을 보여주었습니다.



### Exploiting the Semantic Knowledge of Pre-trained Text-Encoders for Continual Learning (https://arxiv.org/abs/2408.01076)
- **Paper 1**: [{"What's New": 'MCQ 자동 생성의 평가 메트릭의 한계를 해결하기 위해 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다.'}, {'Technical Details': 'KDA는 특정 사실에 대한 학생의 지식을 평가하고 MCQ의 답변 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 사용해 학생의 문제 해결 행동을 모사합니다.'}, {'Performance Highlights': 'Human study를 통해 KDA_disc와 KDA_cont가 KDA 및 실제 강의실에서의 사용성 모두와 강한 상관관계가 있음을 보여주며, n-gram 기반의 유사성 메트릭과 결합 시 전문가가 라벨링한 MCQ 품질 측정에 대한 예측력이 우수함을 확인했습니다.'}]

- **Paper 2**: [{"What's New": '대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 활용하여 NLP 태스크에서의 모델의 강인함을 향상시키는 새로운 방법을 제안합니다.'}, {'Technical Details': '여러 개의 반사실적 데이터들을 생성하고, 집합적 의사 결정을 통해 각 항목의 인과관계를 더 강인하게 감독합니다.'}, {'Performance Highlights': '우리의 방법이 기존 접근법보다 반사실적 강인성, 크로스 도메인 일반화, 데이터가 부족한 상황에서의 일반화 등 다양한 측면에서 상당한 향상을 보여줍니다.'}]

- **Paper 3**: [{"What's New": '실제 환경에서 점진적이고 변화하는 데이터를 다루기 위해 의미 지도를 활용한 연속 학습 방법을 제안합니다.'}, {'Technical Details': '사전 훈련된 CLIP 모델을 시작으로, Semantically-guided Representation Learning (SG-RL) 모듈을 사용해 현재 태스크 클래스에 대한 소프트 어사인먼트를 수행하고, Semantically-guided Knowledge Distillation (SG-KD) 모듈을 사용하여 향상된 지식 전이를 실행합니다.'}, {'Performance Highlights': '우리의 접근법이 일반 및 세부적인 데이터셋에서 다른 방법들보다 뛰어난 성능을 보여주며, CIFAR100 데이터셋의 10단계 설정에서 마지막 태스크 후 정확도가 state-of-the-art보다 11.4 포인트 향상되었습니다.'}]



### Amodal Segmentation for Laparoscopic Surgery Video Instruments (https://arxiv.org/abs/2408.01067)
- **Multiple Choice Questions**: [{"What's New": '기존의 MCQ 생성 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해 새로운 자동 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안했습니다.', 'Technical Details': 'KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 이를 위해 학생의 응답을 기반으로 KDA를 측정하고, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안했습니다.', 'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강하게 상관관계가 있음을 확인했습니다. 또한, n-gram 기반의 유사성 메트릭과 결합했을 때, 다양한 전문가 라벨 MCQ 품질 측정에 대해 높은 예측력을 가졌습니다.'}]

- **Deep Learning Robustness**: [{"What's New": '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용해 NLP 모델의 robustness를 향상시키려는 접근법을 제안했습니다.', 'Technical Details': '기존의 반사실적 증강 방법은 사람이 반사실적 데이터셋을 추가하거나 기계가 데이터셋에서 반사실적에 가까운 것을 찾는 방식인데, 이는 여전히 spurious correlation에 취약합니다. 제안된 방법은 여러 개의 반사실적 데이터셋을 생성하고 집합적 의사 결정(collective decisions)을 통해 용어의 인과성을 더 robust하게 감독합니다.', 'Performance Highlights': 'Empirical 결과는 집합적 의사 결정을 통해 attribution 기반 모델의 편향에 덜 민감해지며, 1) 반사실적 robusteness, 2) 도메인 간 일반화, 3) 희소 데이터로부터의 일반화 등 다양한 측면에서 유의미한 개선을 이뤘습니다.'}]

- **Surgical Instrument Segmentation**: [{"What's New": '의료 분야에서 수술 도구의 Amodal Segmentation을 최초로 도입하고, AIS 데이터셋을 새롭게 제안했습니다.', 'Technical Details': '이 기술은 물체의 보이는 부분과 가려진 부분 모두를 식별합니다. 2017 MICCAI EndoVis Robotic Instrument Segmentation Challenge 데이터셋을 재주석해 AIS 데이터셋을 만들었으며, 여러 주요 amodal segmentation 방법을 평가해 벤치마크를 제공했습니다.', 'Performance Highlights': 'Amodal Segmentation을 통해 가려진 수술 도구의 정확한 예측이 가능해져, 수술 중 중요한 시각 정보를 제공하고, 수술 후 비디오 분석을 통해 절차의 정확성을 평가할 수 있습니다. 또한, 교육적인 목적으로도 활용될 수 있습니다.'}]



### Boosting Gaze Object Prediction via Pixel-level Supervision from Vision Foundation Mod (https://arxiv.org/abs/2408.01044)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문에서는 자동 MCQ(Multiple Choice Questions) 생성에 대한 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)을 제안합니다. 또한, 보고된 텍스트의 인과 관계를 강화하기 위해 대조학습(Contrastive Learning)과 반사실적 증강(Counterfactual Augmentation)을 사용한 방법도 제시하여 더욱 robust한 자연어 처리 모델을 개발합니다. 아울러, 시각적 토대 모델(VFM)을 사용하여 인간의 시선이 향하는 객체를 픽셀 단위로 분할하는 새로운 작업(Gaze Object Segmentation, GOS)도 소개합니다.

- **Technical Details**: KDA는 학생들이 특정 사실에 대한 지식을 바탕으로 MCQ를 답변할 수 있는 능력을 측정하는 메트릭으로, 기존의 n-gram 기반 유사성 메트릭과 차별화됩니다. 반사실적 증강을 통해 단순히 데이터셋에서 유사한 예제를 찾는 대신, 여러 개의 반사실적 데이터를 생성해 집합적 의사 결정(collective decisions)을 통해 모델의 인과 관계를 개선합니다. GOS 작업에서는 VFM을 활용하여 시선이 향하는 대상의 픽셀 단위 마스크를 추론합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 가진다는 사실이 인간 연구를 통해 입증되었습니다. 반사실적 증강을 사용한 모델은 다양한 측면에서 큰 개선을 보였으며, 특히 반사실적 견고성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 희소 데이터로부터의 일반화 면에서 뛰어난 성능을 발휘했습니다. GOS 작업에서 제안된 모델은 GOO-Synth와 GOO-Real 데이터셋 실험을 통해 성능을 검증받았으며, 픽셀 단위의 세부 사항을 고려한 정확한 시선 히트맵을 생성하는 데 우수한 결과를 보였습니다.



### MambaST: A Plug-and-Play Cross-Spectral Spatial-Temporal Fuser for Efficient Pedestrian Detection (https://arxiv.org/abs/2408.01037)
Comments:
          ITSC 2024 Accepted

- **What's New**: 이번 연구는 교사의 학습 평가 시간을 줄이기 위해 자동 Multiple Choice Questions(MCQ) 생성 방법의 개선을 목표로 합니다. 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 단어의 유사성만 평가하기 때문에 교육적 가치를 반영하지 못한다는 한계가 있습니다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하여, 학생이 목표 사실에 대해 답변할 수 있는 능력을 평가합니다.

- **Technical Details**: KDA를 측정하는 방법으로 학생 응답을 기반으로 한 인간 조사 데이터를 사용합니다. 이후, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이를 통해, 인간 연구를 통해 KDA_disc와 KDA_soft가 KDA 및 실제 강의실에서의 사용성과 강한 상관관계를 가지는 것을 증명합니다.

- **Performance Highlights**: 제안된 KDA_disc와 KDA_cont는 인간 전문가가 라벨링한 다양한 MCQ 품질 측정 기준을 예측하는 데 강력한 predictive power를 가진 것으로 나타났습니다.



### POA: Pre-training Once for Models of All Sizes (https://arxiv.org/abs/2408.01031)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문에서는 다양한 크기의 모델을 한 번의 사전 학습(Pre-training)으로 동시에 생성할 수 있는 새로운 자기 지도학습 프레임워크인 POA (Pre-training Once for All)를 제안합니다. 기존의 방법들은 각기 다른 크기의 모델을 따로 학습시켜야 했지만, POA는 단일 프레임워크 내에서 이를 해결할 수 있습니다.

- **Technical Details**: POA는 최신의 '교사-학생(Self-Distillation)' 자기 지도학습 프레임워크를 기반으로 하며, 'Elastic Student'라는 혁신적인 구성 요소를 도입합니다. 매 학습 단계마다 원래 학생 모델에서 랜덤으로 하위 네트워크를 샘플링하여 여러 크기의 모델을 동시에 학습시킵니다. 이는 매 학습 단계에서 다양한 하위 네트워크를 통해 안정적인 학습 과정을 보장합니다.

- **Performance Highlights**: POA는 ViT, Swin Transformer, ResNet과 같은 백본 구조에서 실험을 통해 우수한 성능을 입증했습니다. 단일 사전 학습 세션으로부터 100개가 넘는 다양한 크기의 모델을 생성할 수 있으며, KNN, 선형 프로빙(linear probing) 평가 및 여러 다운스트림 테스크에서 우수한 성능을 보였습니다. 특히 ViT-S, ViT-B, ViT-L 모델들은 최고 성능(State-of-the-Art)을 달성했습니다.



### EIUP: A Training-Free Approach to Erase Non-Compliant Concepts Conditioned on Implicit Unsafe Prompts (https://arxiv.org/abs/2408.01014)
- **What's New**: 이번 논문에서는 자동 Multiple Choice Questions(MCQ) 생성 시스템의 교육적 가치를 측정하기 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)을 제안합니다. 또한 논문에서는 NLP 태스크에서 spurious 패턴에 의존하는 문제를 해결하기 위해 대립 학습(contrastive learning)과 반사실 증강(counterfactual augmentation)을 활용한 새로운 방법도 제시합니다. 마지막으로, 텍스트-이미지 변환 확산 모델(text-to-image diffusion models)의 안전하지 않은 컨텐츠 생성을 억제하기 위한 Erasure Intent Under Prompt(EIUP) 방법을 제안합니다.

- **Technical Details**: ['MCQ 평가 메트릭으로서 KDA는 특정 사실에 대한 지식을 고려하여, 이를 통해 학생 응답을 바탕으로 평가합니다. 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하며, 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다.', 'NLP 모델에서 spurious 델타에 의존하는 문제를 해결하기 위해, 여러 counterfactual을 생성해 집합적 의사결정(collective decisions)을 통해 각각의 단어들의 인과관계를 강화합니다. 이 방법은 기존의 모델이 데이터 내에서 counterfactual에 가까운 것들을 매칭하는 방법과 달리, 더 강력하게 비건설적 패턴에 의한 성능 저하를 방지할 수 있습니다.', "Text-to-image diffusion 모델에서의 안전 문제 해결을 위해, EIUP 방법을 제안합니다. 이는 '지우기 프롬프트(erasure prompt)'를 활용해 불순한 개념을 포함한 프롬프트를 주입하고, 이를 통해 이미지 공간의 불안전한 피처를 식별하여 억제합니다."]

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 강의실 세트에서 높은 사용성을 보였으며, 기존의 n-gram 기반 유사성 메트릭과 결합할 경우 전문가에 의해 라벨링된 다양한 MCQ 품질 측정치에 대해 강한 예측력을 가진 것을 입증하였습니다.', '제안된 방법은 counterfactual robustness와 cross-domain generalization에서도 기존 방법들에 비해 우수한 성능을 보였으며, 특히 데이터가 부족한 상황에서도 일반화 능력이 뛰어났습니다.', 'EIUP는 NSFW 및 스타일 위반 요소를 효과적으로 억제하면서도 이미지의 품질과 의미적 유사성을 유지하는 데 있어서 우수한 성능을 보여줍니다.']



### FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation (https://arxiv.org/abs/2408.00998)
Comments:
          Accepted conference paper of ACM MM 2024

- **What's New**: 최근, 새로운 평가 메트릭 KDA (Knowledge Dependent Answerability)가 제안되었습니다. 이는 기존의 n-gram 기반 평가 메트릭 (예: BLEU, ROUGE, METEOR)과 달리 MCQ의 교육적 가치를 평가합니다. 또한, 텍스트 기반 이미지 생성 모델의 조작 가능성을 개선하는 FBSDiff이 제안되었습니다. 이는 DCT 주파수 대역 교체를 사용하여 이미지-이미지 번역을 더욱 제어할 수 있게 합니다.

- **Technical Details**: [{'MCQ Generation': 'KDA는 학생이 특정 사실에 대한 지식을 바탕으로 질문에 답변할 수 있는 능력을 측정합니다. 기존 평가 메트릭들은 단어 유사성에만 집중하였지만, KDA는 지식 기반의 평가를 도입합니다. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하였으며, 이를 통해 학생의 문제 해결 행동을 모방합니다.'}, {'Text-to-Image Translation': 'FBSDiff는 사전 학습된 대규모 텍스트-이미지(T2I) 확산 모델을 활용하여 이미지를 참고해 이미지 변환을 제어합니다. DCT 스펙트럼 공간의 주파수 대역을 교체하여 역 샘플링 과정 동안 다양한 가이드 요소를 동적으로 대체합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 인간 설문 결과와 높은 상관성을 보였으며, 실제 교실 환경에서의 사용성을 높였습니다. 이와 더불어 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링한 여러 MCQ 품질 지표에 대해 강력한 예측력을 가졌습니다.'}, {'Text-to-Image Translation': 'FBSDiff는 기존의 고급 방법들과 비교했을 때, 시각적 품질, 다용도성, 제어 가능성 면에서 우수하다고 입증되었습니다. 이 방법은 모델 훈련, 모델 세부 조정 또는 온라인 최적화 과정 없이 고품질 텍스트 기반 이미지 번역이 가능합니다.'}]



### Visible-Thermal Multiple Object Tracking: Large-scale Video Dataset and Progressive Fusion Approach (https://arxiv.org/abs/2408.00969)
- **What's New**: 이 연구는 자동 다지선다형 질문(MCQ) 생성을 평가하는 새로운 메트릭인 Knowledge Dependent Answerability(KDA)을 제안합니다. 기존 평가 메트릭들은 BLEU, ROUGE, METEOR와 같이 n-그램 기반 유사성만 평가하며, 교육적 가치를 간과합니다.

- **Technical Details**: KDA는 대상 사실(target fact)에 대한 학생의 지식을 평가하는 MCQ의 답변 가능성(answerability)을 측정합니다. 구체적으로 KDA를 인간 설문 조사로부터 학생 응답 기반으로 측정하고, 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: 휴먼 스터디를 통해 KDA_disc와 KDA_cont가 실제 강의실 내 사용성과 강한 상관관계를 가지며, 전문가가 평가한 MCQ 품질 측정치와도 강한 예측력을 가지고 있음을 보여줍니다.



### Extracting Object Heights From LiDAR & Aerial Imagery (https://arxiv.org/abs/2408.00967)
- **What's New**: 최근 자동 MCQ 생성의 평가 메트릭은 교육적 가치를 간과하고 있다는 문제 제기와 함께, 대상 사실에 대한 학생의 지식을 평가할 수 있는 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 또한, contrastive learning과 counterfactual augmentation을 활용하여 NLP 모델의 robustness를 향상시키는 방안과 LiDAR와 항공 이미지를 이용하여 객체 높이를 추출하는 절차적 방법론을 소개합니다.

- **Technical Details**: 새로운 자동 평가 메트릭 KDA는 학생들의 답변을 기반으로 MCQ의 대답 가능성을 측정합니다. 이와 함께 KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델로 학생의 문제 해결 행동을 모방합니다. NLP 모델의 robustness를 위해 여러 개의 counterfactual을 생성하여 집합적 의사결정을 통해 단어들의 인과관계를 파악하는 방법을 제안합니다. LiDAR와 항공 이미지를 활용한 객체 높이 추출에서는 상위 분할(SOTA) 기법을 이용하여 딥러닝 배경 지식 없이 객체 높이를 추출할 수 있는 방법을 논의합니다.

- **Performance Highlights**: Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가진다는 것을 입증하였습니다. 또한, 인과성 분석을 통한 조치로 다양한 기본 모델 편향에 덜 민감하여, 여러 차원에서의 중요한 개선을 달성했습니다. LiDAR와 항공 이미지를 사용한 객체 높이 추출 방법론은 지오스페이셜 AI(GeoAI)의 발전 방향과 함께 제시되며, 현재 및 미래의 방법들을 평가하고 있습니다.



### MIS-ME: A Multi-modal Framework for Soil Moisture Estimation (https://arxiv.org/abs/2408.00963)
Comments:
          Accepted by DSAA2024

- **What's New**: 이번 연구에서는 교사를 위한 평가 시간을 크게 줄일 수 있는 자동 Multiple Choice 질문 생성(MCQ) 방법에 대한 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)' 를 제안합니다. 기존의 평가 메트릭 BLEU, ROUGE, METEOR이 n-gram 기반의 유사성만을 평가한 것과 달리, KDA는 MCQ의 교육적 유효성을 평가합니다. 이외에도 contrastive learning 및 counterfactual augmentation을 활용해 NLP 모델의 robustness를 향상시키는 새로운 접근법이 제안되었습니다. 또한, 농업 분야에서 토양 수분 예측을 위한 새로운 multi-modal 접근법 MIS-ME가 소개되었습니다.

- **Technical Details**: MCQ 생성 측면에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 도입되었습니다. 이는 사전 훈련된 language model을 사용해 학생들의 문제 해결 행동을 모사하여 측정됩니다. NLP 모델의 robustness 향상을 위해서는 다수의 counterfactual을 생성하고, 집합적 의사결정을 통해 spurious correlation의 영향을 최소화하는 방법을 사용합니다. 또한, MIS-ME 프레임워크는 천문 지상 사진과 기상 데이터를 결합하여 토양 수분을 예측하는 multi-modal 접근법입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가와 강한 상관관계를 보이며, 실제 교실 환경에서도 높은 유용성을 입증하였습니다. 새로운 counterfactual augmentation 접근법은 다양한 측면에서 significant improvements를 달성했습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization. MIS-ME는 10.79%의 MAPE를 기록하여, 전통적인 단일 모달 접근법 대비 meteorological data에서 2.6%, 이미지 데이터에서 1.5% 향상된 성능을 보였습니다.



### PrivateGaze: Preserving User Privacy in Black-box Mobile Gaze Tracking Services (https://arxiv.org/abs/2408.00950)
- **What's New**: 이 논문에서는 MCQ(객관식 문제) 자동 생성의 효율성을 높이기 위한 새로운 평가 메트릭으로 '지식 종속 가능성(KDA, Knowledge Dependent Answerability)'을 제안합니다. 기존의 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 고려하지 않으며, 이와 달리 KDA는 학생의 지식을 평가할 수 있는 MCQ의 대답 가능성(answerability)을 측정합니다.

- **Technical Details**: KDA는 먼저 인간 설문조사에서 학생들의 반응을 통해 측정되며, 이후 사전 학습 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)으로 근사됩니다. 또한 'PrivateGaze' 시스템에서는 사용자의 프라이버시를 침해하지 않으면서 응시 추적 성능을 유지하기 위한 새로운 프레임워크를 소개합니다. 이 시스템은 사용자의 전체 얼굴 이미지를 변환하여 개인 정보를 포함하지 않는 모호한 이미지로 전환합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 강한 상관관계를 가지며, 기존의 n-gram 기반의 유도 메트릭과 결합했을 때 다양한 전문가가 라벨링한 MCQ의 품질 측정에 강한 예측력을 가진다는 것을 보여줍니다. 또한, 'PrivateGaze'는 네 가지 데이터셋을 통해 평가되었으며, 변환된 이미지는 프라이버시를 보호하면서도 기존의 전체 얼굴 이미지와 비교해 응시 추적 성능이 유사하다는 결과를 나타냈습니다.



### Data-Driven Traffic Simulation for an Intersection in a Metropolis (https://arxiv.org/abs/2408.00943)
Comments:
          CVPR 2024 Workshop POETS Oral

- **1**: {"What's New": '자동 다지선 질문(MCQ) 생성의 교육적 평가를 개선하기 위해 KDA(Knowledge Dependent Answerability) 평가 메트릭을 제안했습니다.', 'Technical Details': 'BLEU, ROUGE, METEOR 등의 기존 메트릭은 n-gram 기반의 유사성만 고려하여 교육적 가치를 반영하지 못하는 한계를 갖고 있습니다. KDA는 학생들이 대상 사실(target fact)에 대해 답할 수 있는지의 여부로 MCQ의 유용성을 평가합니다. 이를 위해 KDA_disc와 KDA_cont 라는 두 자동 평가 메트릭을 제안하였습니다.', 'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 사용자의 평가와 강한 상관관계를 보여주었습니다. n-gram 기반 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ 품질 측정 항목들에 대해 강한 예측력을 보였습니다.'}

- **2**: {"What's New": 'NLP 태스크의 robustness를 향상시키기 위해 대비적 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 새로운 접근 방식을 제안했습니다.', 'Technical Details': '기존 연구들은 사람이 반사실적 데이터를 추가하거나 모델이 데이터셋에서 유사한 반사실적 데이터를 찾는 방식이었다면, 이 연구는 "여러 개"의 반사실적 데이터를 합성하여 집합적으로 의사 결정을 내림으로써 인과성을 보다 강하게 슈퍼바이즈합니다.', 'Performance Highlights': '실험 결과, 제안된 방법은 반사실적 강건성(counterfactual robustness), 크로스 도메인 일반화(cross-domain generalization), 그리고 희소 데이터로부터의 일반화에서 유의미한 성능 향상을 보였습니다.'}

- **3**: {"What's New": '도시 교차로에서의 교통 상황을 모델링하기 위한 새로운 데이터 기반 시뮬레이션 환경을 제안했습니다.', 'Technical Details': '실제 교통 데이터(고도 카메라와 추적-탐지 알고리즘 BoT-SORT 사용)를 활용하여 복잡한 동선을 모델링합니다. Gaussian Mixture Models(GMM)을 통해 새로운 에이전트를 생성하고, Conditional GANs 또는 CVAEs 같은 모델을 추가 탐구할 계획입니다.', 'Performance Highlights': '최신 경로 예측 모델을 사용하여 Coarse-To-Fine 방식으로 에이전트의 경로를 생성합니다. 실제 사용 실험에서 Way-point-supervised TrajNet++ 모델은 0.36 FDE를 기록하였습니다.'}



### Towards Zero-Shot Annotation of the Built Environment with Vision-Language Models (Vision Paper) (https://arxiv.org/abs/2408.00932)
- **What's New**: 최근 발표된 논문에서는 자동 MCQ(다지선다형 질문) 생성의 교육적 가치를 평가할 새로운 메트릭을 제안했습니다. 기존의 BLEU, ROUGE, METEOR 등의 메트릭은 생성된 질문의 단어 유사성에만 초점을 맞추고 있어 실질적인 교육 효과를 놓치는 문제가 있습니다. 이를 해결하기 위해 '지식 의존 정답 가능성(Knowledge Dependent Answerability, KDA)'이라는 새로운 평가 메트릭을 도입했습니다.

- **Technical Details**: KDA는 생성된 MCQ가 해당 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지를 측정합니다. 이를 바탕으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였으며, 학생들의 문제 해결 행동을 모방하는 사전 학습된 언어 모델을 활용하여 KDA를 근사합니다.

- **Performance Highlights**: Human study를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 라벨링한 실제 클래스룸 세팅의 사용성과 강한 상관관계를 가짐을 입증했습니다. 추가로, n-gram 기반 유사성 메트릭과 함께 사용했을 때, KDA_disc와 KDA_cont는 다양한 전문가 라벨링 MCQ 품질 측정에 대해 높은 예측력이 있음을 보여주었습니다.



### Reclaiming Residual Knowledge: A Novel Paradigm to Low-Bit Quantization (https://arxiv.org/abs/2408.00923)
Comments:
          Accepted by The 35th British Machine Vision Conference (BMVC 2024)

- **What's New**: 최근 논문에서 다양한 새로운 기술들이 소개되었습니다. 첫 번째 논문은 Multiple Choice Questions (MCQ)의 자동 생성을 돕기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다. 두 번째 논문은 NLP 태스크에서 학생의 인과 관계를 더 잘 이해할 수 있도록 대비 학습과 counterfactual augmentation을 사용한 접근 방법을 다룹니다. 마지막 논문은 ConvNets의 저비트 양자화(quantization)를 개선하기 위해 Optimal Quantization Residual Convolutional Operator Low-Rank Adaptation (CoRa)이라는 프레임워크를 제안합니다.

- **Technical Details**: [{'paper': '첫 번째 논문', 'description': 'MCQ 자동 생성의 문제점을 해결하기 위해 KDA라는 새로운 평가 메트릭을 도입했습니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭도 제안되어, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다.'}, {'paper': '두 번째 논문', 'description': 'spurious pattern에 의존하는 deep model의 robustness 문제를 해결하기 위해 여러 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다.'}, {'paper': '세 번째 논문', 'description': 'CoRa라는 프레임워크는 ConvNets의 저비트 양자화를 위해 optimal quantization residual knowledge를 저랭크 어댑터(low-rank adapters)를 통해 복구하는 방법을 제안합니다. 이 프레임워크를 통해 더 적은 반복으로 성능 저하 없이 모델을 양자화할 수 있습니다.'}]

- **Performance Highlights**: [{'paper': '첫 번째 논문', 'description': 'Human studies 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, n-gram 기반의 유사성 메트릭과 결합할 경우 전문가가 라벨링한 MCQ 품질 평가에서 강한 예측력을 보였습니다.'}, {'paper': '두 번째 논문', 'description': '제안된 방법은 counterfactual robustness, cross-domain generalization, 희소한 데이터로부터의 일반화 등 다양한 차원에서 기존 모델들에 비해 높은 성능을 보였습니다.'}, {'paper': '세 번째 논문', 'description': 'CoRa는 ImageNet에서 사전 훈련된 다수의 ConvNet에 대해 평가되었으며, 4비트 및 3비트 양자화에서 state-of-the-art 양자화 인식 훈련과 사후 훈련 양자화 기준에 대비하여 비교 가능한 성능을 달성하였습니다. 단지 1600장의 작은 이미지 세트로 250회 미만의 반복을 통해 최적화를 달성합니다.'}]



### Medical SAM 2: Segment medical images as video via Segment Anything Model 2 (https://arxiv.org/abs/2408.00874)
- **What's New**: 이번 뉴스레터에서는 교육 및 의료 이미지 세분화(세그멘테이션) 분야에서 최신 연구들을 소개합니다. 첫 번째 연구는 자동 다중 선택 질문(MCQ) 생성의 평가 메트릭 혁신을 제안하며, 두 번째 연구는 NLP 태스크에서의 대조 학습과 counterfactual augmentation을 통해 모델의 robustness를 향상시키는 방법을 논의합니다. 마지막으로, SAM 2 프레임워크를 활용한 의료 이미지 세분화 모델인 MedSAM-2가 소개됩니다.

- **Technical Details**: [{'Paper Title': 'Automated MCQ Generation', 'Core Concepts': 'BLEU, ROUGE, METEOR 등의 기존 평가 메트릭은 교육적 가치를 고려하지 않고 단어의 유사성에만 집중한다는 한계를 극복하기 위해, 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 메트릭을 제안. 이는 학생이 특정 사실에 대한 지식을 바탕으로 MCQ에 답할 수 있는 능력을 평가한다는 점에서 의미가 있다.', 'Technical Keywords': 'BLEU, ROUGE, METEOR, KDA, automatic evaluation metric, pre-trained language models'}, {'Paper Title': 'Improving Model Robustness with Contrastive Learning', 'Core Concepts': '기존의 counterfactual augmentation 방법들은 spurious correlation(스퓨리어스 상관관계)에 의해 영향을 받아 모델의 robustness(강건성)에 한계가 있다. 본 연구는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안.', 'Technical Keywords': 'counterfactual augmentation, contrastive learning, robustness, spurious patterns'}, {'Paper Title': 'Medical SAM 2 for 2D and 3D Image Segmentation', 'Core Concepts': "SAM 2 프레임워크를 활용해 2D 및 3D 의료 이미지 세분화 작업을 수행하는 MedSAM-2 모델을 소개. MedSAM-2은 '의료 이미지를 비디오로 취급'하는 철학을 기반으로 설계되어, 단 한 번의 프롬프트로 일련의 이미지를 자동으로 세분화하는 새로운 기능을 가진다.", 'Technical Keywords': 'medical image segmentation, SAM 2, one-prompt segmentation, 2D and 3D images'}]

- **Performance Highlights**: [{'Paper Title': 'Automated MCQ Generation', 'Key Findings': 'KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 경우 전문가가 라벨링한 다양한 MCQ 품질 기준에 대해 높은 예측 성능을 보인다.'}, {'Paper Title': 'Improving Model Robustness with Contrastive Learning', 'Key Findings': '제안된 방법이 다양한 차원에서 significant improvements(유의미한 향상)을 달성함. 특히 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization에서 뛰어난 성과를 보임.'}, {'Paper Title': 'Medical SAM 2 for 2D and 3D Image Segmentation', 'Key Findings': 'MedSAM-2은 기존 모델들을 능가하는 성능을 보여주었으며, 2D 및 3D 의료 이미지 세분화 작업에서 state-of-the-art 결과를 기록. One-prompt Segmentation 설정에서, 이전의 few-shot 및 one-shot 모델들을 뛰어넘는 능력을 입증하였다.'}]



### A Scalable and Generalized Deep Learning Framework for Anomaly Detection in Surveillance Videos (https://arxiv.org/abs/2408.00792)
- **What's New**: 자동 MCQ(객관식 질문) 생성과 평가에 대한 새로운 접근법과 반사실적 증강 (counterfactual augmentation)을 통한 NLP 태스크의 강건성 강화 방법, 및 비디오 이상 탐지의 일반화 문제를 해결하는 새로운 딥러닝 프레임워크가 제안되었습니다.

- **Technical Details**: 자동 MCQ 생성 평가에서는 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭의 한계를 극복하기 위해 지식 종속 답변 가능성(KDA)를 도입하였으며, 딥러닝 모델의 반사실적 증강에서는 집합적 의사결정을 통한 인과관계 파악 방법이 소개되었습니다. 비디오 이상 탐지에서는 전이 학습과 모델 융합, 멀티태스크 분류를 통해 다양한 태스크에 대해 재훈련 없이 일반화할 수 있는 프레임워크가 개발되었습니다.

- **Performance Highlights**: 지식 종속 답변 가능성(KDA) 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 보여주었으며, 딥러닝 강건성 강화 방법은 반사실적 강건성, 도메인 간 일반화, 적은 데이터 일반화에서 상당한 개선을 이루었습니다. 비디오 이상 탐지 프레임워크는 RLVS 데이터셋에서 97.99%, UCF 데이터셋에서 83.59%, 두 데이터셋 전반에서 88.37%의 정확도를 달성하였으며, 보지 못한 데이터셋에서 87.25%의 정확도를 기록했습니다.



### Data-driven Verification of DNNs for Object Recognition (https://arxiv.org/abs/2408.00783)
- **What's New**: 현재 MCQ 생성을 평가하는 기존의 메트릭을 개선하기 위해, 저자들은 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안했다. 이 메트릭은 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 데 중점을 둔다.

- **Technical Details**: 기존 메트릭(예: BLEU, ROUGE, METEOR)은 단어 유사성에 집중하였지만, 새로운 KDA 메트릭은 학생 반응을 기반으로 MCQ의 대답 가능성을 측정한다. Human survey를 시행한 후, 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모사하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다.

- **Performance Highlights**: Human evaluation을 통해, KDA_disc와 KDA_cont가 실제 강의실 환경에서 사용성과 강한 상관관계를 가지고 있음을 확인했다. 또한, 이 메트릭들은 기존의 n-gram 기반 유사성 메트릭과 결합할 때, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였다.



### CATD: Unified Representation Learning for EEG-to-fMRI Cross-Modal Generation (https://arxiv.org/abs/2408.00777)
- **What's New**: 최근의 딥러닝 모델들이 NLP 태스크에서 사람보다 높은 정확성을 보였음에도 불구하고, spurious patterns에 의존하여 강건성(robustness)이 제한된다는 보고가 있다. 이에 저자들은 contrastive learning과 counterfactual augmentation을 활용하여 모델의 강건성을 향상시키는 방법을 제안하였다.

- **Technical Details**: 제안된 방법은 기존의 인간이 counterfactual을 생성하거나 근사치 대체를 자동으로 찾는 방식이 아닌, 여러 개의 counterfactual을 합성하고 이를 기반으로 집합적 의사결정을 하여 단어들의 인과관계를 명확히 파악하는 것이다. 이를 위해 사전 학습된 언어 모델을 활용한다.

- **Performance Highlights**: 제안된 방법은 다양한 측면에서 기존 방법보다 우수한 성능을 보였다. 특히, counterfactual 강건성, cross-domain generalization, 그리고 적은 데이터 상황에서의 일반화 능력에서 significant improvements를 달성하였다.



### 2D Neural Fields with Learned Discontinuities (https://arxiv.org/abs/2408.00771)
- **What's New**: 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 도입했습니다. 이 메트릭은 생성된 문제의 교육적 가치를 평가하는 기존 메트릭을 보완하여 학생의 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: 제안된 KDA 메트릭은 학생 응답에 기반해 MCQ의 답변 가능성을 측정합니다. 이를 기반으로 하여, 기존 대비 더 강한 상관관계를 보이는 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안했습니다. 이 메트릭은 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모사합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 보이며, 전문가 레이블된 다양한 MCQ 품질 측정에서 높은 예측력을 지닌 것으로 나타났습니다.



### Comparing Optical Flow and Deep Learning to Enable Computationally Efficient Traffic Event Detection with Space-Filling Curves (https://arxiv.org/abs/2408.00768)
Comments:
          27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024)

- **What's New**: 최근 연구에서는 Multiple Choice Questions(MCQ) 자동 생성의 평가 메트릭 문제를 지적하며, Knowledge Dependent Answerability(KDA)라는 새로운 자동 평가 메트릭을 제안했습니다. 이 메트릭은 MCQ의 답변 가능성을 측정하여 교육적 가치를 평가합니다.

- **Technical Details**: 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에 초점을 맞추는 반면, KDA는 대상 사실에 대한 학생의 지식을 평가하는 데 중점을 둡니다. KDA는 학생들이 문제를 풀어보는 human survey를 기반으로 측정하고, 이를 자동화하기 위해 KDA_disc 및 KDA_cont를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 높은 상관관계를 가지며, 기존의 n-gram 기반 메트릭과 결합할 때 예측력이 강해집니다.



### Talk Less, Interact Better: Evaluating In-context Conversational Adaptation in Multimodal LLMs (https://arxiv.org/abs/2408.01417)
Comments:
          Accepted to COLM 2024

- **What's New**: [{'Title': '자동 MCQ 평가를 위한 지식 종속 가능성(KDA) 메트릭 제안', 'Summary': '기존의 BLEU, ROUGE, METEOR 평가 메트릭은 MCQ 생성의 교육적 가치를 반영하지 못합니다. 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)는 학생이 해당 사실에 대한 지식을 바탕으로 MCQ를 푸는 능력을 측정합니다. 인간 설문 기반의 평가를 통해 KDA와 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 높은 상관관계를 보인다는 것을 확인했습니다.'}, {'Title': '딥 모델의 대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Summary': '딥 모델이 NLP 과제에서 높은 정확성을 보였지만, 잘못된 패턴에 의존하여 로버스트니스가 제한됩니다. 비대칭 학습(contrastive learning)과 대안적 자극(counterfactual augmentation)을 적용하는 방법을 제안하며, 여러 counterfactual을 생성하여 집합적 의사 결정을 통해 단어의 인과관계를 파악합니다. 이 방법으로 다양한 차원에서 성능 향상을 이끌어 냅니다.'}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Summary': '대화가 진행됨에 따라 인간은 점점 더 효율적인 언어를 사용하게 됩니다. 이를 여러 state-of-the-art MLLMs (multimodal large language models)에서 테스트해 보니, 상대방의 효율적인 언어를 이해할 수는 있지만, 스스로 효율성을 증가시키는 능력은 보이지 않았습니다. GPT-4와 같은 모델에서만 특정 프롬프트를 통해 이 능력을 유도할 수 있었습니다.'}]

- **Technical Details**: [{'Title': '자동 MCQ 평가를 위한 KDA 메트릭', 'Details': 'KDA는 학생의 응답을 통해 MCQ의 지식 기반 대답 가능성을 평가합니다. KDA_disc와 KDA_cont는 KDA를 예측하기 위해 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한 메트릭입니다.'}, {'Title': '대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Details': "기존의 대안적 증강 방식을 개선하여 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 보다 로버스트하게 인과관계를 지도합니다."}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Details': 'ICCA(Automated framework to evaluate conversational adaptation)를 사용하여 MLLMs의 문맥별 행동을 평가했습니다. GPT-4와 같은 모델은 특정 조건에서만 효율성을 스스로 증가시킬 수 있음을 발견했습니다.'}]

- **Performance Highlights**: [{'Title': '자동 MCQ 평가를 위한 KDA 메트릭', 'Highlights': 'KDA_disc와 KDA_cont는 실제 교실 설정에서 사용자인 전문가들에 의해 높은 예측력을 지녔습니다. 이는 n-gram 기반 유사성 메트릭과 결합하여 다양한 MCQ 품질 척도에서 높은 예측력을 가집니다.'}, {'Title': '대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Highlights': '집합적 의사 결정 방식을 통해 counterfactual robustness, cross-domain generalization, 창의적인 데이터 학습 측면에서 성능을 크게 향상시켰습니다.'}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Highlights': '일부 MLLMs는 상대방의 효율적인 언어를 이해할 수 있지만, 대다수 모델은 자체적으로 언어 효율성을 증가시키는 능력이 부족합니다.'}]



### Play to the Score: Stage-Guided Dynamic Multi-Sensory Fusion for Robotic Manipulation (https://arxiv.org/abs/2408.01366)
- **What's New**: 교사와 학생 평가 시간을 혁신적으로 절감하기 위해, 새로운 MCQ(Multiple Choice Questions) 자동 생성 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 또한, 딥러닝 모델의 강건성(robustness)을 향상시키기 위해 대조 학습과 반사실적 증강(contrastive learning and counterfactual augmentation)을 사용하는 새로운 접근법이 도입되었습니다. 마지막으로, 인간의 다중 감각 융합 능력을 로봇에 적용하기 위해 MS-Bot이라는 다단계 동적 다중 감각 융합 방법이 제안되었습니다.

- **Technical Details**: {'MCQ 평가': '기존의 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 교육적 가치를 평가하는 데 한계가 있습니다. 이를 개선하기 위해 KDA를 도입했고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방하여 MCQ의 대답 가능성을 측정합니다.', '딥러닝 강건성': '모델의 강건성 향상을 위해 반사실적 증강(counterfactual augmentation)을 통해 여러 개의 반사실적 데이터를 생성하고, 집단적 의사결정(collective decision)을 통해 각 용어의 인과 관계를 강건하게 감독합니다.', '다중 감각 로봇 학습': '복잡한 작업 단계에 따른 다단계 동적 다중 감각 융합 방법을 채택하여, 로봇이 현재 단계의 미세한 상태에 따라 감각 모달리티의 우선순위를 동적으로 조정합니다. MS-Bot은 시각, 청각, 촉각 센서를 통합하여 복잡한 로봇 조작 과제를 수행할 수 있습니다.'}

- **Performance Highlights**: {'MCQ 평가 메트릭': 'Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실 세팅에서 사용성과 강한 상관관계를 가짐을 확인했습니다.', '딥러닝 강건성': '제안된 방법은 반사실적 강건성, 도메인 간 일반화, 부족한 데이터 일반화에서 상당한 향상을 보였습니다.', '다중 감각 로봇': 'MS-Bot은 복잡한 조작 과제인 붓기 및 키웨이 삽입 작업에서 기존 방법보다 뛰어난 성능을 보였습니다.'}



### Toward Automatic Relevance Judgment using Vision--Language Models for Image--Text Retrieval Evaluation (https://arxiv.org/abs/2408.01363)
Comments:
          Accepted by ACM SIGIR 2024 LLM4Eval Workshop: this https URL

- **What's New**: 자동 MCQ 생성과 평가 메트릭 관련 연구에서 교육적 가치를 잘 반영하지 못한다는 문제를 해결하기 위해 새로운 평가 메트릭인 지식 종속 대답 가능성 (Knowledge Dependent Answerability, KDA)을 제안합니다. 또한, 최근 NLP 태스크의 deep model들이 높은 정확성을보였지만 spurious patterns에 의존하여 robustness가 제한된다는 것을 인지하고, 이 문제를 해결하기 위해 대조 학습 (contrastive learning)과 반사실적 증강 (counterfactual augmentation)을 활용합니다. 마지막으로, Vision-Language Models (VLMs)의 이미지-텍스트 검색 성능을 평가하며, 인간의 판단과 비교하여 GPT-4V와 같은 모델이 더 높은 일치도를 보여준다는 점을 강조합니다.

- **Technical Details**: 첫 번째 연구에서는 KDA를 측정하는 새로운 방법을 제안합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방할 수 있도록 미리 훈련된 언어 모델을 사용합니다. 두 번째 연구에서는 여러 반사실적 상황을 생성해 이를 바탕으로 분포 기반의 결정 (collective decisions)을 통해 모델의 robustness를 향상시킵니다. 마지막 연구에서는 CLIP, LLaVA, GPT-4V와 같은 VLM들을 사용해 이미지-텍스트 검색 성능을 평가하고 Kendall's τ와 Cohen's κ 값으로 성능을 비교합니다.

- **Performance Highlights**: 자동 평가 메트릭 KDA_disc와 KDA_cont는 KDA 및 실제 강의실 사용성 측면에서 강한 상관관계를 보였습니다. 대조 학습을 통한 모델은 다양한 측면에서 기존 방법들보다 높은 반사실적 robustness와 도메인 간 일반화 성능을 달성했습니다. GPT-4V와 같은 VLM들은 CLIPScore를 능가하는 일치도를 보였으며, 특히 인간 판단에 가장 근접한 분포를 보여줍니다.



### PC$^2$: Pseudo-Classification Based Pseudo-Captioning for Noisy Correspondence Learning in Cross-Modal Retrieva (https://arxiv.org/abs/2408.01349)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 교육적 가치를 반영한 자동 MCQ 평가 메트릭, 반직관적 학습을 통한 NLP 모델 개선, 그리고 노이즈가 있는 데이터세트에서의 크로스 모달 검색을 위한 새로운 프레임워크가 각각 제안되었습니다.

- **Technical Details**: [{'Paper': 'Automatic MCQ Generation', 'Details': '기존의 BLEU, ROUGE, METEOR 메트릭의 한계를 극복하고자 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. 이는 목표 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': 'NLP 태스크에서 deep models의 robustness를 높이기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법을 제안했습니다. 이는 여러 개의 counterfactual을 생성하고, 집합적 의사 결정 (collective decisions)을 사용하여 모델의 인과관계를 감독합니다.'}, {'Paper': 'Cross-Modal Retrieval with Noisy Correspondence', 'Details': '노이즈가 있는 크로스 모달 검색을 개선하기 위해 Pseudo-Classification based Pseudo-Captioning (PC2) 프레임워크를 도입했습니다. 이는 캡션을 범주형 레이블로 해석하여 모델이 이미지-텍스트 의미적 유사성을 학습하게 하고, pseudo-captions을 생성하여 부정확한 데이터 쌍에 대해 더 정량적인 감독 정보를 제공합니다.'}]

- **Performance Highlights**: [{'Paper': 'Automatic MCQ Generation', 'Details': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 강한 상관관계를 가지며, 다양한 전문가가 레이블링한 MCQ 품질 지표에 대해 높은 예측력을 보였습니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': '집합적 의사 결정을 통해 반직관적 학습을 수행하여 다양한 측면 (1. counterfactual robustness, 2. cross-domain generalization, 3. scarce data generalization)에 대한 성능이 향상되었습니다.'}, {'Paper': 'Cross-Modal Retrieval with Noisy Correspondence', 'Details': 'PC2 프레임워크는 시뮬레이션 및 실제 데이터셋에서 기존의 크로스 모달 검색 기법 및 NCL-robust 방법들을 뛰어넘는 성능을 보였습니다. 또한, Noise of Web (NoW) 데이터셋을 통해 새로운 강력한 NCL 벤치마크를 도입하였습니다.'}]



### A Backbone for Long-Horizon Robot Task Understanding (https://arxiv.org/abs/2408.01334)
Comments:
          8 pages, 8 figures. This work is intended to be submitted to IEEE Robotics and Automation Letters (RA-L) for possible publication

- **What's New**: 이 연구에서는 기존의 평가 메트릭이 학습 평가에서 교육적 가치를 충분히 반영하지 못하는 문제를 해결하기 위해, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 나타냅니다. 또한, 논문에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 새로운 방법도 제시합니다. 마지막으로, Therblig 기반의 백본 프레임워크(TBBF)를 통한 장기적 로봇 태스크의 이해와 전이성을 향상시키는 방법도 논의합니다.

- **Technical Details**: KDA는 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 포함하며, 이는 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 새롭게 제안된 로봇 학습 방법은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악합니다. TBBF는 Therblig (기본 동작 요소)을 사용하여 고수준 로봇 태스크를 초심적인 로봇 구성이 가능하도록 분해하여 학습의 이해도를 향상시킵니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA 및 전문가들에 의해 레이블링된 실제 강의실 세트에서 높은 사용성을 보였습니다. 로봇 학습 접근 방식에서는 다중 counterfactual을 통한 robust한 인과 관계 파악이 다양한 차원에서 유의미한 개선을 보였습니다. TBBF는 실제 세계에서의 다양한 상황에 대해 94.4%의 성공률과 복잡한 상황에서 80%의 성공률을 달성했습니다.



### 3DPX: Progressive 2D-to-3D Oral Image Reconstruction with Hybrid MLP-CNN Networks (https://arxiv.org/abs/2408.01292)
Comments:
          accepted by MICCAI 2024

- **What's New**: 이 아티클에서는 다양한 분야의 최신 연구 결과를 소개합니다. 첫 번째 연구는 자동 MCQ 생성의 교육적 가치를 평가하기 위한 새로운 메트릭 KDA (Knowledge Dependent Answerability)를 제안했습니다. 두 번째 연구는 contrastive learning과 counterfactual augmentation을 활용하여 NLP 모델의 robustness를 향상시키는 방법을 제안했습니다. 마지막으로, 세 번째 연구는 치과에서 사용되는 파노라마 X-레이(PX) 이미지에서 3D 구조를 재구성하기 위한 새로운 하이브리드 MLP-CNN 피라미드 네트워크를 제안했습니다.

- **Technical Details**: 첫 번째 연구에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 연구는 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어 간의 인과관계를 더 robust하게 파악하는 방법을 제안합니다. 마지막 연구에서는 2D 파노라마 X-레이 이미지를 3D로 재구성하기 위해 progressive hybrid Multilayer Perceptron (MLP)-CNN 피라미드 네트워크(3DPX)를 제안합니다. 이 접근법은 CNN의 즉각적인 주변 정보 수집 한계를 보완하는 데 MLP를 통합합니다.

- **Performance Highlights**: 첫 번째 연구에서 KDA_disc와 KDA_cont는 인간 평가 및 전문가 라벨링과 강한 상관관계를 보였으며, 다양한 전문가 라벨 측정 MCQ 품질에 대한 예측 능력이 우수했습니다. 두 번째 연구는 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 영역에서 유의미한 개선을 달성했습니다. 마지막 연구는 두 개의 대규모 데이터셋에서 기존 2D에서 3D로의 구강 재구성 방법을 능가했으며, 다운스트림 각도 불일치 분류 작업의 성능도 향상시켰습니다.



### Deep Learning based Visually Rich Document Content Understanding: A Survey (https://arxiv.org/abs/2408.01287)
Comments:
          Work in Progress

- **Multiple Choice Question (MCQ) Generation**: [{"What's New": '교사들의 학습 평가 시간을 줄이기 위해 MCQ 자동 생성 기술이 제안되었으나, 기존 평가 메트릭 BLEU, ROUGE, METEOR는 교육적 가치를 평가하지 못하고 있었습니다. 이에 대해, 우리는 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 MCQ의 대답 가능성을 측정합니다.'}, {'Technical Details': 'KDA는 학생들이 실험적으로 대답할 수 있는지를 측정하는 메트릭입니다. 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하며, 이는 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, {'Performance Highlights': 'Human studies를 통해, KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 가집니다. 또한, n-gram 기반의 유사성 메트릭과 결합하면 전문가가 평가한 MCQ 품질 측정에서 높은 예측력을 가집니다.'}]

- **Robustness in NLP Models**: [{"What's New": '최근 깊은 학습 모델이 NLP 태스크에서 사람을 초월하는 정확성을 보임에도 불구하고, spurious pattern에 의존하여 robustness가 제한됩니다. 이를 해결하기 위해 대조적 학습 및 counterfactual augmentation을 활용하는 방식을 제안합니다.'}, {'Technical Details': '기존의 augmentation 방식은 사람이 직접 개입하거나 데이터셋에서 유사 counterfactual을 찾는 방식으로 이루어지지만, 여전히 spurious correlation의 영향을 받습니다. 본 방법은 여러 개의 counterfactual을 합성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방식으로 진행됩니다.'}, {'Performance Highlights': '우리의 접근 방식은 집합적 의사 결정을 통해 attribution-based synthesis의 태스크 모델 편향에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data로부터의 generalization 등 다양한 차원에서 유의미한 성능 향상을 이룹니다.'}]

- **Visually Rich Document Understanding (VRDU)**: [{"What's New": '학계, 금융, 의료, 마케팅 등 다양한 분야에서 시각적으로 풍부한 문서(VRD)의 정보 추출은 중요한 작업입니다. 기존 방식은 전문가의 지식을 기반으로 한 수작업이 많아 비용과 효율성 문제를 안고 있었으나, 깊은 학습의 등장으로 이를 혁신적으로 개선할 수 있게 되었습니다.'}, {'Technical Details': '최근 VRDU에서 깊은 학습 기반 프레임워크가 주목받고 있으며, 이는 비전, 텍스트, 레이아웃 등 다중 모달 정보를 활용하여 고도화된 문서 표현을 만듭니다. 여러 벤치마크 데이터셋과 다양한 기법들을 체계적으로 조사하고 분석하여, 각 모델의 강점과 한계, 적합한 대응 시나리오 등을 비교합니다.'}, {'Performance Highlights': '최근 발전한 LSTM, CNN 기반 모델, 피처 기반 접근법, 레이아웃 인식 사전 학습 프레임워크 등은 여러 다운스트림 태스크에서 state-of-the-art 성능을 달성하였습니다. 이 논문에서는 이러한 다양한 접근 방식을 비판적으로 검토하며, 앞으로의 연구 방향과 실무 적용에 대한 인사이트를 제공합니다.'}]



### Out-Of-Distribution Detection for Audio-visual Generalized Zero-Shot Learning: A General Framework (https://arxiv.org/abs/2408.01284)
- **What's New**: 최근 논문들은 자동 Multiple Choice Questions (MCQ) 생성 및 이해력 평가를 강화하는 새로운 접근법을 제안하였습니다. 또한, NLP 모델의 강건성을 높이기 위해 대조 학습과 반사실적 증강 (counterfactual augmentation)을 도입하거나, 오디오-비쥬얼 GZSL (Generalized Zero-Shot Learning)에 대한 새로운 프레임워크를 제안하였습니다.

- **Technical Details**: MCQ 자동 생성에서는 기존의 평가 지표가 교육적 가치를 고려하지 않은 문제를 해결하기 위해 KDA (Knowledge Dependent Answerability)를 제안하였습니다. NLP 강건성을 높이기 위해 대조 학습과 반사실적 증강 방법을 사용했으며, 단어들의 인과관계를 파악하는 데에 있어 높은 성능을 보였습니다. 오디오-비쥬얼 GZSL에서는 Generative Adversarial Networks (GANs)을 사용하여 unseen features를 생성하고, OOD detector와 클래스 분류기를 훈련하는 방법을 제안하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었고, 각각의 메트릭을 결합하였을 때 전문가가 평가한 MCQ 품질 지표 예측에 강한 성능을 보였습니다. 대조 학습을 적용한 NLP 모델은 다양한 측면에서 눈에 띄는 성능 개선을 이루었으며, 오디오-비쥬얼 GZSL프레임워크의 평가 결과 기존 최신 모델들보다도 유의미한 개선을 보였습니다.



### Interpreting Global Perturbation Robustness of Image Models using Axiomatic Spectral Importance Decomposition (https://arxiv.org/abs/2408.01139)
Comments:
          Accepted by Transactions on Machine Learning Research (TMLR 2024)

- **What's New**: 교사들이 학생 평가 시간을 줄일 수 있도록 새로운 MCQ 생성 자동 평가 메트릭 KDA (Knowledge Dependent Answerability)가 제안되었습니다. 또한, NLP 태스크에서 모델의 robustness를 높이기 위해 대조 학습과 counterfactual augmentation을 활용하는 방법이 발표되었습니다. 마지막으로, 이미지 모델의 취약성을 평가하고 이들의 global mechanistic interpretability를 제공하는 I-ASIDE 방법이 소개되었습니다.

- **Technical Details**: [{'paper_title': 'Automatic MCQ Generation', 'details': '현존하는 MCQ 생성 평가지표(BLEU, ROUGE, METEOR)는 교육적 가치를 반영하지 않습니다. 새로운 KDA 메트릭은 MCQ의 대답 가능성(answerability)을 측정하며 학생이 특정 지식을 평가하는 능력을 테스트합니다. KDA_disc와 KDA_cont 두 가지 자동화 메트릭은 사전에 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'paper_title': 'NLP Task Robustness', 'details': '최근 deep model들은 NLP 태스크에서 높은 정확성을 보이지만 spurious pattern에 의존해 robustness가 제한됩니다. 이는 대조 학습과 counterfactual augmentation을 통해 해결할 수 있습니다. 제안된 방식은 여러 개의 counterfactual을 생성하고 collective decision-making 과정을 통해 단어들의 인과관계를 더 견고하게 이해합니다.'}, {'paper_title': 'Image Model Interpretability', 'details': '이미지 모델의 perturbation robustness를 평가하기 위한 model-agnostic 글로벌 메커니즘 해석 방법인 I-ASIDE가 제안되었습니다. Shapley 값 이론을 적용해 robust features(RFs)와 non-robust features(NRFs)의 예측력을 정보 이론적 틀 안에서 해석합니다. 낮은 주파수 신호는 일반적으로 높은 주파수 신호보다 견고합니다.'}]

- **Performance Highlights**: [{'paper_title': 'Automatic MCQ Generation', 'highlights': 'KDA_disc와 KDA_cont는 강의실 환경에서의 사용성과 강한 상관관계를 보입니다. 이들은 n-gram 기반의 유사성 메트릭과 결합될 때, 다양한 전문가가 라벨링한 MCQ 품질 지표에 대한 강한 예측력을 가집니다.'}, {'paper_title': 'NLP Task Robustness', 'highlights': '제안된 counterfactual augmentation 방법은 attribution-based synthesis의 task model bias에 덜 민감하고, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 등 여러 차원에서 significant한 개선을 이룹니다.'}, {'paper_title': 'Image Model Interpretability', 'highlights': 'I-ASIDE는 다양한 ImageNet에서 사전 훈련된 비전 모델을 대상으로 하는 광범위한 실험을 통해 perturbation robustness를 측정하고, 그 메커니즘을 성공적으로 해석할 수 있음을 보여줍니다.'}]



### Privacy-Preserving Split Learning with Vision Transformers using Patch-Wise Random and Noisy CutMix (https://arxiv.org/abs/2408.01040)
Comments:
          23 pages, 11 figures, 8 tables, to be published in Transactions on Machine Learning Research (TMLR)

- **What's New**: 이번 발표에서는 몇 가지 혁신적인 연구 결과를 공유했습니다. 첫째, 교육자들이 학생 평가 시 드는 시간을 줄이기 위한 자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 소개했습니다. 둘째, 자연어 처리(NLP)에서 기계학습 모델의 낮은 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 새로운 접근 방식을 제안했습니다. 셋째, 자원 제한이 있는 에지 기기에서 Vision Transformer (ViT)의 학습을 가능하게 하는 Split Learning (SL) 기반 비밀 보장 학습 프레임워크 DP-CutMixSL을 소개했습니다.

- **Technical Details**: KDA는 학생들의 대상 사실에 대한 지식을 평가할 수 있는 MCQ의 대답 가능성을 측정하는 방식을 제안하며, 이와 관련하여 KDA_disc와 KDA_cont라는 자동 평가 지표를 소개했습니다. NLP 분야에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 각 용어들의 인과성을 robust하게 감시하는 방법을 제안했습니다. 컴퓨터 비전 분야에서는, DP-CutMixSL은 CutMix 정규화를 통해 Gaussian noise를 추가한 smashed data를 업로드하고, 이를 mixer가 혼합하여 서버로 전송하여 비밀 보장을 강화하는 메커니즘을 사용했습니다.

- **Performance Highlights**: 첫째로, KDA_disc와 KDA_cont는 실제 강의실에서 활용가능성이 크며 전문가들이 레이블링한 사용성과 강한 상관관계를 보였습니다. 둘째로, NLP의 새로운 접근 방식은 model bias가 낮아지고, counterfactual robustness, cross-domain generalization, scarce data generalization 등 다양한 지표에서 성능 향상을 이루었습니다. 셋째로, DP-CutMixSL은 membership inference attack, reconstruction attack, label inference attack에 대한 프라이버시 보호를 강화하며, 기존 방법들에 비해 정확도도 향상되었습니다.



### Structure from Motion-based Motion Estimation and 3D Reconstruction of Unknown Shaped Space Debris (https://arxiv.org/abs/2408.01035)
Comments:
          6 pages, 10 figures. Manuscript accepted at the 2024 IEEE 20th International Conference on Automation Science and Engineerin (CASE 2024)

- **What's New**: 이 논문은 자동 다지선다형 질문(MCQ) 생성 평가에 있어 기존 방식들이 교육적 가치를 충분히 고려하지 못하는 문제를 지적하면서, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 특정 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 또한, 논문은 최근 NLP 태스크에서의 deep model의 robustness 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용하는 방법도 다룹니다. 끝으로, 우주 쓰레기의 움직임을 정확하게 추정하기 위해 Structure from Motion (SfM) 기반 알고리즘을 제안합니다.

- **Technical Details**: {'MCQ Evaluation': '기존의 n-gram 기반 유사성 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 평가하지 못하므로, KDA는 대상 사실에 대한 MCQ의 answerability를 측정합니다. Human survey를 통해 학생 반응을 기반으로 KDA를 측정하며, 두 가지 자동화된 평가 메트릭인 KDA_disc와 KDA_cont를 제안하여 pre-trained language models를 사용해 학생의 문제 해결 과정을 흉내냅니다.', 'Deep Model Robustness': '최근 deep model들이 spurious patterns에 의존해 robustness가 제한되는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용합니다. 기존 방법들과 달리 여러 개의 counterfactual을 생성하여 집합적인 의사 결정을 통해 단어의 인과관계를 더 잘 파악합니다.', 'Space Debris Motion Estimation': '우주 쓰레기 모양이 알려지지 않은 상태에서 움직임을 추정하기 위해 2D 이미지만을 입력으로 사용하는 SfM 기반 알고리즘을 제안합니다. 이 알고리즘은 다중 시점의 2D 이미지를 입력 받아, 대상 물체와 카메라 간의 상대적인 위치와 3D 형상을 동시에 복원합니다.'}

- **Performance Highlights**: {'MCQ Evaluation': 'Human study를 통해 KDA_disc와 KDA_cont가 기존의 인간 평가와 강한 상관관계를 가지며, 실제 강의실 환경에서도 높은 usability를 보여주었습니다.', 'Deep Model Robustness': '제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 제한된 데이터로부터의 generalization에서 기존 모델보다 뛰어난 성능을 보였습니다.', 'Space Debris Motion Estimation': '(1) 현실적인 미세 중력 실험 이미지 데이터셋에서 SfM 알고리즘의 유효성을 검증했습니다. (2) 백그라운드 노이즈를 제거하고, 3D 포인트 클라우드를 보다 균일하게 만드는 노이즈 제거 및 다운샘플링 과정을 통해 더욱 정밀하게 우주 쓰레기의 움직임을 추정할 수 있음을 입증했습니다.'}



### PINNs for Medical Image Analysis: A Survey (https://arxiv.org/abs/2408.01026)
- **What's New**: 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하여, 기존의 BLEU, ROUGE, METEOR와 달리 MCQ의 교육적 가치를 더 정확히 평가하려고 합니다. 이 메트릭은 학생이 대상 사실에 대한 지식을 바탕으로 MCQ를 답할 수 있는지에 대해 평가합니다.

- **Technical Details**: 첫째, Human Survey를 통해 KDA를 측정하는 방법을 제시하였고, 둘째, 사전 학습된 언어 모델을 이용하여 KDA를 근사하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다. 이들은 실제 강의실 세트에서의 사용성과 밀접한 상관관계를 가지고 있음을 인체 실험으로 증명했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강력한 예측력을 갖고 있음을 보여줍니다.



### A dual-task mutual learning framework for predicting post-thrombectomy cerebral hemorrhag (https://arxiv.org/abs/2408.00940)
- **What's New**: 최근 몇 가지 논문에서 중요한 혁신을 제시했습니다. 첫 번째 논문에서는 학생들의 지식 평가를 위한 새로운 MCQ 생성의 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 도입했습니다. 두 번째 논문은 NLP 태스크에서 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용하는 방법을 모색합니다. 세 번째 논문에서는 뇌경색 환자의 초기 CT 스캔만을 사용하여 수술 후 뇌출혈을 예측하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 첫 번째 논문에서는 KDA, KDA_disc 및 KDA_cont 메트릭이 학생의 응답을 바탕으로 문제 해결 능력을 평가하는 방법을 설명합니다. 두 번째 논문은 '여러 개의' counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 각 용어의 인과관계를 robust하게 감독하는 방법을 제안합니다. 세 번째 논문은 dual-task interactive learning framework를 통해 초기 CT 스캔에서 후속 CT 스캔을 예측하고 예후 레이블을 동시에 추정하며, self-attention과 interactive attention 메커니즘을 통합하여 중요한 영역에 집중합니다.

- **Performance Highlights**: 첫 번째 논문의 KDA_disc와 KDA_cont 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 보여주었습니다. 두 번째 논문에서는 집합적 의사 결정을 통해 기존 방법보다 counterfactual robustness, cross-domain generalization, 그리고 자료가 부족한 상태에서도 일반화 능력이 향상되었습니다. 세 번째 논문에서는 임상 데이터에서 후속 CT 스캔을 최고 수준의 방법보다 잘 생성하고, 86.37%의 예측 정확도를 달성했습니다.



### CIResDiff: A Clinically-Informed Residual Diffusion Model for Predicting Idiopathic Pulmonary Fibrosis Progression (https://arxiv.org/abs/2408.00938)
- **What's New**: 이 논문은 Multiple Choice Questions (MCQ)의 자동 생성을 평가하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 학생이 특정 사실을 알고 있을 때 해당 MCQ에 답할 수 있는 능력을 측정합니다. 두 가지 자동 평가 메트릭으로 KDA_disc와 KDA_cont를 제안하며, 이는 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.

- **Technical Details**: KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용해 학생들이 문제를 풀 때 보여주는 행동을 모방해 KDA를 근사화합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 확인하였습니다. 또한, 이 두 메트릭은 n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 라벨링한 MCQ 품질 측정치를 강하게 예측할 수 있는 능력을 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 높은 상관관계를 보이며, 전문가가 라벨링한 MCQ 품질 측정에 강한 예측력을 보였습니다. 이는 기존의 BLEU, ROUGE, METEOR 지표들이 놓치는 교육적 가치를 평가하는 데에 큰 도움을 줄 수 있습니다.



### Temporal Evolution of Knee Osteoarthritis: A Diffusion-based Morphing Model for X-ray Medical Image Synthesis (https://arxiv.org/abs/2408.00891)
- **What's New**: 새로운 논문들은 여러 학문에서 혁신적인 접근 방식을 제안하고 있습니다. 자동 MCQ 생성을 위한 새로운 자동 평가 메트릭, 자연어 처리(NLP) 태스크에서의 robustness 향상 방법, 그리고 무릎 골관절염(KOA) 진행을 시각화하기 위한 딥러닝 모델이 포함되어 있습니다.

- **Technical Details**: [{'title': '자동 MCQ 생성', 'content': 'BLEU, ROUGE, METEOR 같은 기존 메트릭이 교육적 가치를 고려하지 못한다는 문제를 해결하기 위해 지식 종속 가능성(KDA, Knowledge Dependent Answerability)이라는 새로운 자동 평가 메트릭을 제안합니다. KDA_disc와 KDA_cont라는 두 자동 평가 메트릭은 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다.'}, {'title': 'NLP 모델의 robustness', 'content': '최근 딥 모델들이 NLP 태스크에서 높은 정확성을 보여주지만, spurious pattern에 의존하여 robustness가 떨어지는 문제를 해결하기 위해 대조 학습과 반사실적 증강(contrastive learning and counterfactual augmentation)을 활용합니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악합니다.'}, {'title': '무릎 골관절염(KOA) 진행 시각화 모델', 'content': 'KOA 진행을 시각화하기 위한 Diffusion-based Morphing Model(DMM)을 제안합니다. Denoising Diffusion Probabilistic Model을 수정하여 건강한 무릎 X-ray와 심한 KOA 단계 사이의 중간 이미지를 합성합니다. 하이브리드 손실(diffusion loss, morphing loss, supervision loss)을 사용하여 높은 성능의 시각적 프레임을 합성합니다.'}]

- **Performance Highlights**: [{'title': 'KDA 평가', 'content': 'KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 예측력이 강함을 보였습니다.'}, {'title': 'Robustness 향상', 'content': '제안된 방법은 attribution-based synthesis보다 태스크 모델의 bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 희소 데이터로부터의 일반화에서 상당한 성능 향상을 이뤄냈습니다.'}, {'title': 'KOA 시각화 모델', 'content': '제안된 DMM이 KOA 진행을 효과적으로 반영하면서 원래 이미지의 구조적, 텍스처적 무결성을 유지하는 X-ray 이미지를 합성함을 실험 결과로 증명했습니다.'}]



### HOAA: Hybrid Overestimating Approximate Adder for Enhanced Performance Processing Engin (https://arxiv.org/abs/2408.00806)
- **What's New**: 이 연구는 교육자가 학생 평가에 소요되는 시간을 크게 줄일 수 있는 자동 복수선택문항(MCQ) 생성을 위해 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안합니다. KDA는 생성된 MCQ가 목표 사실(target fact)에 대한 학생의 지식을 평가하는 능력을 강조하여 기존의 BLEU, ROUGE, METEOR 등의 평가 메트릭의 한계를 극복하려고 합니다.

- **Technical Details**: 먼저, KDA를 측정하기 위해 인간 설문조사를 기반으로 한 학생의 응답을 이용합니다. 이후, 사전학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 개의 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이를 통해 학습 필요성과 문제 교육적 가치를 더 잘 평가할 수 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅에서 사용성 측면에서 강한 상관관계를 보여주었습니다. 또한, 전문가가 라벨링한 다양한 MCQ 품질 측정치와 결합되었을 때, 이는 강력한 예측력을 갖는 것으로 입증되었습니다.



### CCSRP: Robust Pruning of Spiking Neural Networks through Cooperative Coevolution (https://arxiv.org/abs/2408.00794)
- **What's New**: 최근 자동 MCQ 생성(electronic quiz)을 평가하는 새로운 메트릭 'Knowledge Dependent Answerability (KDA)'와 SNN 프루닝(pruning)을 위한 혁신적인 'CCSRP' 방법이 소개되었습니다. KDA는 MCQ의 교육적 가치를 높이기 위해 설계되었으며, CCSRP는 자원 제한 환경에서의 SNN 성능을 상호 협력 진화 방법으로 증가시킵니다.

- **Technical Details**: KDA는 학생들이 특정 지식을 바탕으로 MCQ에 답할 수 있는지를 평가합니다. 두 가지 자동화 평가 메트릭인 KDA_disc와 KDA_cont를 사용하여 사전 훈련된 언어 모델을 통해 학생들의 문제 해결 행동을 모방합니다. 한편, CCSRP는 협력적 공동 발전(evolutionary algorithms)을 통해 SNN 필터를 독립적으로 프루닝하여 정확성, 견고성, 컴팩트성을 동시에 높이는 삼중 목표 최적화 문제를 해결합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 강의실 세트에서 높은 상관관계를 보여주었으며, 다양한 품질 측정 기준에 대한 예측력이 우수함을 입증했습니다. CCSRP는 CIFAR-10와 SVHN 데이터셋에서 최신 방법론과 견줄만한 성능을 나타냈습니다.



### Hands-on STEM Learning Experiences using Digital Technologies (https://arxiv.org/abs/2408.00781)
Comments:
          9 pages, 10 figures

- **papers**: [{"What's New": '자동 MCQ 생성을 위해 새로운 평가 메트릭 Knowledge Dependent Answerability (KDA)을 제안했습니다. 이는 학생의 지식을 평가하는 능력을 측정합니다.', 'Technical Details': '기존 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반의 유사성만을 평가하며 교육적 가치를 간과했습니다. KDA는 학생 반응을 기반으로 MCQ의 대답 가능성을 측정합니다. 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont가 제안되었습니다.', 'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반의 유사성 메트릭과 결합할 때 강력한 예측력을 보였습니다.'}, {"What's New": '대조 학습과 반사실적 증가(counterfactual augmentation)를 통해 NLP 태스크에서의 모델 robustness를 향상시키는 방법을 탐구합니다.', 'Technical Details': "기존 방법은 사람이 반사실적(counterfactual)을 만들거나 알고리즘이 데이터셋에서 유사한 것을 찾는 방식이었으나, 우리의 접근법은 '여러 개의' 반사실적을 생성하고 집합적 의사 결정으로 단어들의 인과관계를 감독합니다.", 'Performance Highlights': '우리의 접근법은 다양한 차원에서 유의미한 성과를 보였습니다: 1) 반사실적 robust성, 2) 도메인 간 일반화, 3) 부족한 데이터에서의 일반화.'}, {"What's New": '디지털 제작 기술을 이용한 실질적이고 시각적인 STEM 교육 접근법을 이탈리아 학교에서의 활동을 통해 소개합니다.', 'Technical Details': 'FabLab을 활용하여 3D 프린팅, 마이크로 컨트롤러, 바람 역학 시각화 등을 통해 학생들이 협업, 창의력, 비판적 사고, 실험 및 문제 해결 능력을 기를 수 있도록 지원합니다.', 'Performance Highlights': '저렴하고 현대적인 컴퓨터 제어 신속 프로토타이핑 도구를 사용하여 실습 교육을 촉진하였으며, 이를 통해 장애가 있는 학생들도 과학에 참여할 수 있는 기회를 제공했습니다.'}]



### In-Depth Analysis of Emotion Recognition through Knowledge-Based Large Language Models (https://arxiv.org/abs/2408.00780)
Comments:
          7 pages

- **What's New**: 논문에서 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 소개합니다. KDA는 생성된 다지선다형 질문(MCQ)의 대답 가능성을 측정하여 학생이 해당 사실에 대해 얼마나 지식을 가지고 있는지를 평가합니다. 또한, 반사실적 데이터 증강을 활용하여 자연어 처리(NLP) 모델의 강건성(robustness)을 향상시키는 방법과 새로운 컨텍스트 기반 감정 인식 접근 방식을 제안합니다.

- **Technical Details**: MCQ 데이터셋에 있는 골드 샘플과 비교하는 기존 메트릭의 단점을 해결하기 위해, KDA는 학생 응답을 기반으로 한 평가 방식을 사용합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방함으로써 자동으로 KDA를 근사합니다. 또한, NLP 태스크에서 spurious pattern을 방지하기 위해, 반사실적 데이터 증강을 사용하여 모델의 예측 분포를 집합적 의사 결정(collective decision)을 통해 개선하는 방법을 제안합니다. 감정 인식을 위해서는 Bayesian Cue Integration (BCI) 방법을 사용하여 상황적 지식과 얼굴 표정 데이터를 통합합니다.

- **Performance Highlights**: Human studies에서 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강하게 상관됨을 보였습니다. 또한, KDA_disc와 KDA_cont는 n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 레이블한 다양한 MCQ 품질 측정치에 대해 높은 예측력을 나타냈습니다. NLP 모델의 강건성을 향상시키기 위해 제안된 방법은 반사실적 강건성(counterfactual robustness), 크로스 도메인 일반화(cross-domain generalization), 그리고 희소한 데이터에서 일반화(generalization from scarce data) 영역에서 중요한 개선을 이루었습니다. 감정 인식에서 BCI 방법은 인간 관찰자와 비교할 때 높은 정확도를 보였으며, 컨텍스트와 얼굴 표정 데이터를 통합하여 더 다양한 감정을 인식하는 데 성공했습니다.



### Fuzzy Logic Approach For Visual Analysis Of Websites With K-means Clustering-based Color Extraction (https://arxiv.org/abs/2408.00774)
Comments:
          The work has been submitted to Herald of KBTU journal

- **What's New**: {'MCQ Generation': '자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)을 제안했습니다. 기존 BLEU, ROUGE, METEOR 메트릭이 교육적 가치를 간과하는 문제를 해결하고자 합니다.', 'NLP Robustness': 'NLP 작업에서 Robustness를 향상시키기 위해 대조 학습 (contrastive learning) 과 counterfactual augmentation을 활용하는 새로운 접근 방식을 제안합니다. 여러 counterfactual을 생성하고, 집합적 의사 결정 (collective decisions)을 통해 robust하게 감독합니다.', 'Website Aesthetics': '웹사이트 디자인 미학이 사용자 경험에 미치는 영향을 연구하고, 색상 조화와 글꼴 인기를 기반으로 웹사이트 미학을 측정하는 새로운 방법을 도입했습니다.'}

- **Technical Details**: {'MCQ Generation': '소개된 KDA는 학생 반응을 바탕으로 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용해 이를 자동으로 평가합니다.', 'NLP Robustness': '기존 방법은 인간이 counterfactual을 생성하거나 데이터셋에서 유사한 것을 찾는 반면, 새로운 접근 방식은 여러 개의 counterfactual을 생성하고 이들의 예측 분포를 집합적으로 결정합니다.', 'Website Aesthetics': '200개의 인기 있는 웹사이트 디자인에서 주요 색상을 추출하고 퍼지 로직 (fuzzy logic)을 사용해 미학적 선호를 예측합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가졌으며, 전문가가 라벨링한 MCQ 품질 평가와도 높은 예측력을 보였습니다.', 'NLP Robustness': '새로운 방법은 다양한 차원에서 의미있는 개선을 달성했으며, 특히 counterfactual robustness, cross-domain generalization, scarce data generalization에서 높은 성과를 보였습니다.', 'Website Aesthetics': '첫 번째 인상이 50 밀리초 내에 결정되며, 색상 조화와 글꼴 인기가 사용자 인식에 큰 영향을 미친다는 점을 강조했습니다.'}



### Hybrid Deep Learning Framework for Enhanced Melanoma Detection (https://arxiv.org/abs/2408.00772)
- **What's New**: 최신 논문들을 소개합니다. 첫 번째 논문은 자동 MCQ 생성을 위한 새로운 평가 메트릭을 제안합니다. 두 번째 논문은 NLP 모델의 robustness를 향상시키는 방법을 탐구하고 있습니다. 세 번째 논문은 피부암의 조기 진단을 위한 새로운 하이브리드 프레임워크를 제시합니다.

- **Technical Details**: [{'MCQ Generation': '기존의 n-gram 기반 평가 메트릭 (BLEU, ROUGE, METEOR)이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)를 제안합니다. 새로운 KDA_disc와 KDA_cont 메트릭은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. Human evaluation 결과, 이 메트릭들은 실제 강의실에서의 사용성과 높은 상관관계를 가집니다.', 'NLP Robustness': '최근 NLP 모델의 spurious pattern 의존성을 줄이기 위해 contrastive learning과 counterfactual augmentation을 결합했습니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안합니다. Empirical results에 따르면, 이 방법은 cross-domain generalization과 limited 데이터에서의 모델 성능 향상에 기여합니다.'}, {'Cancer Detection': '멜라노마 검출을 위해 U-Net과 EfficientNet을 결합한 하이브리드 모델을 제안합니다. HAM10000 데이터셋을 이용하여 U-Net 모델을 훈련시켜 정확한 세그먼트를 수행하고, ISIC 2020 데이터셋으로 EfficientNet을 훈련시켜 피부암을 이진 분류합니다. 이 프레임워크는 ISIC 2020 데이터셋에서 99.01%의 놀라운 정확도를 달성했습니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont 메트릭은 전통적인 n-gram 기반 메트릭과 결합하여 전문가가 라벨링한 MCQ 품질 측정의 예측력을 크게 향상시켰습니다.', 'NLP Robustness': '제안된 방법은 다양한 차원에서 의미 있는 성능 향상을 보여주었으며, 특히 counterfactual robustness와 cross-domain generalization에서 우수한 결과를 나타냈습니다.'}, {'Cancer Detection': 'ISIC 2020 데이터셋에서 99.01%의 정확도를 달성한 하이브리드 모델은 기존의 모델 구조보다 우수한 성능을 보였습니다. 이 접근법은 피부암 조기 진단과 치료에 있어 매우 유망한 방법을 제시합니다.'}]



New uploads on arXiv(cs.AI)

### Conditional LoRA Parameter Generation (https://arxiv.org/abs/2408.01415)
- **What's New**: 우리는 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안하여 MCQ(Multiple Choice Questions)의 대답 가능성(answerability)을 측정하고, 대상 사실(fact)에 대한 학생의 지식 평가를 더 정확하게 반영할 수 있도록 합니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 같은 n-gram 기반 메트릭은 생성된 MCQ와 데이터셋의 골드 샘플 간의 유사성만 평가하며, 교육적 가치는 고려하지 않습니다. 우리는 KDA라는 메트릭을 제안하고, 학생 응답을 바탕으로 측정하는 방법을 제시합니다.', 'KDA_disc와 KDA_cont': 'KDA_disc와 KDA_cont는 pretrained language models을 이용해 학생들의 문제 해결 행동을 모방하여 KDA를 근사하는 자동 평가 메트릭으로서, 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다.'}

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가가 평가한 MCQ 품질 측정을 예측하는 강력한 예측력을 보입니다. 이는 자동 MCQ 생성 시스템의 교육적 유용성을 크게 향상시킬 수 있음을 시사합니다.



### A Comprehensive Review of Multimodal Large Language Models: Performance and Challenges Across Different Tasks (https://arxiv.org/abs/2408.01319)
- **What's New**: 기존의 다중 선택 질문(MCQ) 자동 생성 평가 메트릭(BLEU, ROUGE, METEOR)은 단순히 생성된 질문과 데이터셋의 골드 샘플 간의 n-gram 유사성에만 초점을 맞추고 있어 교육적 가치를 고려하지 못합니다. 이에 대응하기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하였습니다. 이 메트릭은 대상 사실에 대한 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 인간 평가를 기반으로 학생 응답을 통해 측정합니다. 또한 사전 학습된 언어 모델을 활용하여 학생의 문제해결 행동을 모방하는 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다. 이들은 기존 평가 메트릭과 결합하여 다양한 MCQ 품질 측정치에 대한 예측력을 높입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 연구를 통해 실제 강의실 세트에서의 사용성과 높은 상관관계를 가지며, 전문가가 표시한 MCQ 품질 측정치에도 강한 예측력을 보입니다.



### TrIM: Triangular Input Movement Systolic Array for Convolutional Neural Networks -- Part I: Dataflow and Analytical Modelling (https://arxiv.org/abs/2408.01254)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: ['자동 생성되는 다지선다형 시험 문제(Multiple Choice Questions, MCQ)의 교육적 가치를 평가할 수 있는 새로운 자동 평가 메트릭, Knowledge Dependent Answerability(KDA)을 제안합니다. KDA는 문제의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 고려합니다.', 'NLP 작업에서 대체 학습(contrastive learning)과 counterfactual augmentation을 활용하여 모델의 강건성을 향상시키는 새로운 방법을 제안합니다. 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어들의 인과 관계를 보다 robust하게 평가합니다.', '최첨단 AI 모델의 계산 복잡성과 데이터 강도를 따라잡기 위해 고안된 새로운 계산 패러다임으로, Systolic Arrays(SAs)의 데이터 전송 비용을 줄이기 위한 Triangular Input Movement(TrIM) 데이터를 제안합니다.']

- **Technical Details**: ['기존 MCQ 생성 평가지표인 BLEU, ROUGE, METEOR는 단순한 n-gram 기반 유사성만 평가하며 교육적 가치를 고려하지 않습니다. 새로운 KDA 메트릭은 학생 응답을 바탕으로 하고, 사전 훈련된 언어 모델을 이용하여 학생들의 문제 해결 행동을 모방합니다.', '기존의 counterfactual augmentation 방법은 spurious correlation에 영향을 받는 반면, 제안된 방법은 여러 counterfactual을 생성하고 집합적 의사 결정을 통해 보다 robust한 인과 관계 평가를 합니다. 이로써 task model bias에 덜 민감합니다.', 'TrIM 데이터플로우는 input을 삼각형으로 이동시켜 데이터 활용도를 극대화하며, 메모리 접근 횟수를 크게 줄이는 것이 핵심입니다. 기존의 weight-stationary와 row-stationary 대비하여 메모리 접근 횟수를 ~10배 감소시키며, throughput은 최대 81.8% 향상시킵니다.']

- **Performance Highlights**: ['Human evaluation에서 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 확인하였습니다. n-gram 유사성 메트릭과 결합하여 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였습니다.', '제안된 방법은 다양한 측면에서 상당한 개선을 달성했으며, 특히 counterfactual 강건성, 크로스 도메인 일반화, 희소 데이터로부터의 일반화에서 두드러집니다.', 'TrIM 기반의 Systolic Array는 메모리 접근 횟수를 기존 방법 대비 10배 줄이고, 처리량을 81.8% 향상시키며, 레지스터 수도 최대 15.6배 감소시켰습니다.']



### Metareasoning in uncertain environments: a meta-BAMDP framework (https://arxiv.org/abs/2408.01253)
- **What's New**: 교육용 MCQ 생성을 개선하기 위해 기존의 워드 케이스 기반 메트릭의 한계를 지적하고, 새로운 반응성 평가 메트릭인 지식 종속 가능성(KDA: Knowledge Dependent Answerability)을 제안했습니다. 이 메트릭은 교육적 가치와 학습자의 이해도를 평가하는 데 초점을 맞춥니다.

- **Technical Details**: KDA는 학생들의 실제 반응을 바탕으로 MCQ의 대답 가능성을 평가하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가들이 레이블링한 실제 강의실 세트에서 사용성과 강한 상관관계를 보여주었습니다. 또한, n-gram 기반의 유사성 메트릭과 결합될 때 다양한 전문가 레이블의 MCQ 품질 측정에 대해 높은 예측 능력을 보였습니다.



### Rubric-based Learner Modelling via Noisy Gates Bayesian Networks for Computational Thinking Skills Assessmen (https://arxiv.org/abs/2408.01221)
- **What's New**: 이 논문은 자동 MCQ(복수 선택 문항) 생성에 대한 새로운 평가 메트릭, 'Knowledge Dependent Answerability (KDA)'를 제안합니다. 기존의 BLEU, ROUGE, METEOR 메트릭이 생성된 MCQ의 교육적 가치를 평가하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: KDA는 학생이 문제를 해결할 때 대상 사실(target fact)에 대한 지식을 기반으로 MCQ의 대답 가능성(answerability)을 측정합니다. 이를 위해 학생 응답을 기반으로 KDA를 측정한 후, 미리 학습된 언어 모델(pre-trained language models)을 활용하여 학생들의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont를 제안합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실에서 사용될 때의 유용성과 강한 상관관계를 가지며, 전문가가 레이블링한 MCQ 품질 측정치에 대해 높은 예측력을 가진다는 것을 보여주었습니다.



### Multi-Objective Deep Reinforcement Learning for Optimisation in Autonomous Systems (https://arxiv.org/abs/2408.01188)
Comments:
          pages, Accepted to AI4AS 2024 workshop

- **What's New**: 다양한 분야에서 새로운 기법과 접근 방법이 제안되었으며, 특히 다중 선택 질문(Multiple Choice Questions, MCQ) 생성, NLP의 robust성 향상, 다목적 강화 학습(Multi-Objective Reinforcement Learning, MORL)에 대한 연구가 두드러졌습니다.

- **Technical Details**: [{'Title': "새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'", 'Details': "기존의 BLEU, ROUGE, METEOR 평가 메트릭은 MCQ의 교육적 가치를 평가하지 못하고 단순히 n-gram 유사성만 측정합니다. 이를 보완하기 위해 'Knowledge Dependent Answerability (KDA)'라는 새로운 자동 평가 메트릭을 제안하였습니다. KDA는 대상 사실(target fact)에 대한 지식을 바탕으로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭도 제안하여 KDA를 미리 학습된 언어 모델(pre-trained language models)을 이용해 추정할 수 있도록 했습니다."}, {'Title': 'Contrastive Learning과 Counterfactual Augmentation을 활용한 NLP 모델의 Robust성 향상', 'Details': '최근의 딥러닝 모델들이 NLP 태스크에서 사람보다 높은 정확성을 보여주지만, spurious pattern에 의존하여 robust성이 제한됩니다. 이 논문에서는 여러 개의 counterfactual을 생성하고 이들의 예측 분포를 집합적으로 결정하는 새로운 방법을 제안하여 robust하게 인과관계를 파악합니다. 이를 통해 다양한 측면에서 의미 있는 성능 향상을 실현합니다.'}, {'Title': 'Deep W-Learning을 이용한 다목적 강화 학습 (Multi-Objective Reinforcement Learning, MORL)', 'Details': '자율 시스템(Autonomous Systems, AS)에서 다중 목표를 최적화하기 위해 Deep W-Learning (DWN)을 사용합니다. Emergent Web Servers (EWS)와 같은 자가 적응 서버(self-adaptive server)에 적용하여 실험을 진행하였으며, DWN이 평균 응답 시간과 구성 비용 측면에서 의미 있는 성능을 보여주었습니다.'}]

- **Performance Highlights**: [{'Title': 'KDA 기반 MCQ 평가', 'Details': 'KDA_disc와 KDA_cont는 학생 반응을 기반으로 한 KDA와의 상관관계가 강하며, 실제 강의실 설정에서의 사용 가능성도 높게 평가되었습니다. 또한, n-gram 기반 유사성 메트릭과 결합했을 때 전문가들이 표기한 다양한 MCQ 질적 지표에 대해 강력한 예측 능력을 보였습니다.'}, {'Title': 'Counterfactual Augmentation을 이용한 NLP 모델', 'Details': '제안된 방법은 기존의 spurious correlation 문제를 해결하며, 1) counterfactual robust성, 2) cross-domain generalization, 3) scarce data로부터의 generalization 같은 다양한 차원에서 중요한 성능 향상을 달성했습니다.'}, {'Title': 'Multi-Objective Reinforcement Learning (MORL)', 'Details': 'Deep W-Learning (DWN)을 사용한 다목적 최적화는 평균 응답 시간에서는 우수한 성능을, 비용 변동성에서는 약간의 감소를 보였으나, 여전히 ε-greedy 및 Deep Q-Networks (DQN) 접근법에 비해 전반적으로 더 나은 성능을 보였습니다.'}]



### Interpreting Global Perturbation Robustness of Image Models using Axiomatic Spectral Importance Decomposition (https://arxiv.org/abs/2408.01139)
Comments:
          Accepted by Transactions on Machine Learning Research (TMLR 2024)

- **What's New**: 교사들이 학생 평가 시간을 줄일 수 있도록 새로운 MCQ 생성 자동 평가 메트릭 KDA (Knowledge Dependent Answerability)가 제안되었습니다. 또한, NLP 태스크에서 모델의 robustness를 높이기 위해 대조 학습과 counterfactual augmentation을 활용하는 방법이 발표되었습니다. 마지막으로, 이미지 모델의 취약성을 평가하고 이들의 global mechanistic interpretability를 제공하는 I-ASIDE 방법이 소개되었습니다.

- **Technical Details**: [{'paper_title': 'Automatic MCQ Generation', 'details': '현존하는 MCQ 생성 평가지표(BLEU, ROUGE, METEOR)는 교육적 가치를 반영하지 않습니다. 새로운 KDA 메트릭은 MCQ의 대답 가능성(answerability)을 측정하며 학생이 특정 지식을 평가하는 능력을 테스트합니다. KDA_disc와 KDA_cont 두 가지 자동화 메트릭은 사전에 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'paper_title': 'NLP Task Robustness', 'details': '최근 deep model들은 NLP 태스크에서 높은 정확성을 보이지만 spurious pattern에 의존해 robustness가 제한됩니다. 이는 대조 학습과 counterfactual augmentation을 통해 해결할 수 있습니다. 제안된 방식은 여러 개의 counterfactual을 생성하고 collective decision-making 과정을 통해 단어들의 인과관계를 더 견고하게 이해합니다.'}, {'paper_title': 'Image Model Interpretability', 'details': '이미지 모델의 perturbation robustness를 평가하기 위한 model-agnostic 글로벌 메커니즘 해석 방법인 I-ASIDE가 제안되었습니다. Shapley 값 이론을 적용해 robust features(RFs)와 non-robust features(NRFs)의 예측력을 정보 이론적 틀 안에서 해석합니다. 낮은 주파수 신호는 일반적으로 높은 주파수 신호보다 견고합니다.'}]

- **Performance Highlights**: [{'paper_title': 'Automatic MCQ Generation', 'highlights': 'KDA_disc와 KDA_cont는 강의실 환경에서의 사용성과 강한 상관관계를 보입니다. 이들은 n-gram 기반의 유사성 메트릭과 결합될 때, 다양한 전문가가 라벨링한 MCQ 품질 지표에 대한 강한 예측력을 가집니다.'}, {'paper_title': 'NLP Task Robustness', 'highlights': '제안된 counterfactual augmentation 방법은 attribution-based synthesis의 task model bias에 덜 민감하고, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 등 여러 차원에서 significant한 개선을 이룹니다.'}, {'paper_title': 'Image Model Interpretability', 'highlights': 'I-ASIDE는 다양한 ImageNet에서 사전 훈련된 비전 모델을 대상으로 하는 광범위한 실험을 통해 perturbation robustness를 측정하고, 그 메커니즘을 성공적으로 해석할 수 있음을 보여줍니다.'}]



### Being Accountable is Smart: Navigating the Technical and Regulatory Landscape of AI-based Services for Power Grid (https://arxiv.org/abs/2408.01121)
Comments:
          Author's version of the paper for International Conference on Information Technology for Social Good (GoodIT '24), September 4--6, 2024, Bremen, Germany. It is posted here for your personal use. Not for redistribution

- **What's New**: AI 기반 자동 MCQ 생성과 관련된 평가 메트릭을 개선하기 위해 지식 종속 가능성(KDA: Knowledge Dependent Answerability)을 제안합니다. 기존 메트릭들은 n-그램 기반의 유사성에 초점을 맞추어 학습자 지식 평가를 잘 하지 못하는 문제를 해결하고자 하나.

- **Technical Details**: 새로운 메트릭 KDA는 학생들이 문제를 해결할 수 있는 능력을 바탕으로 MCQ의 대답 가능성을 측정합니다. 이를 위해, 학생 반응을 바탕으로 KDA를 측정하고, 사전 학습된 언어 모델을 이용하여 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다. 이를 통해 학생들의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont가 실제 강의실에서 사용 가능성과 강한 상관관계를 보인다는 것을 확인했습니다. 또 이들 메트릭을 n-그램 기반 유사성 메트릭과 결합하면 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강한 예측력을 가집니다.



### Dissecting Dissonance: Benchmarking Large Multimodal Models Against Self-Contradictory Instructions (https://arxiv.org/abs/2408.01091)
Comments:
          Accepted by the 18th European Conference on Computer Vision ECCV 2024

- **What's New**: 이 연구는 Multiple Choice Question(MCQ, 객관식 질문)의 자동 생성을 평가하기 위해 새로운 메트릭을 제안합니다. 기존에 사용되던 BLEU, ROUGE, METEOR 메트릭들은 n-gram 기반 유사성에 집중하여 교육적 가치를 평가하지 못합니다. 이에 반해 새로운 메트릭인 Knowledge Dependent Answerability(KDA)는 MCQ의 답변 가능성(answerability)과 관련된 학생의 지식을 측정합니다.

- **Technical Details**: KDA는 학생의 문제 해결 행동을 흉내내기 위해 사전 훈련된 언어 모델을 활용하여 자동으로 평가하는 KDA_disc와 KDA_cont 메트릭을 포함합니다. 연구에서는 KDA를 인간 설문조사를 통해 측정하는 방법을 제시하고, KDA_disc와 KDA_cont가 실제 강의실 세팅에서의 사용성과 높은 상관관계를 보인다는 것을 입증하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사성 메트릭과 결합되었을 때, 전문가가 라벨링한 다양한 MCQ 품질 조치에 대해 예측력이 높은 것으로 나타났습니다.



### A Survey on Self-play Methods in Reinforcement Learning (https://arxiv.org/abs/2408.01072)
- **What's New**: MCQ(객관식 질문) 자동 생성 기술에서 기존 평가 메트릭들의 한계를 극복하기 위해 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 최근 deep models의 NLP(Natural Language Processing) 태스크에서의 robustness(견고성)을 향상시키기 위해 대조학습 및 counterfactual augmentation을 활용하는 연구가 발표되었습니다. 또한, 강화학습 분야에서 self-play(자기 대전) 기법의 다양한 응용과 방법론을 체계적으로 정리한 종합적인 서베이 논문이 발표되었습니다.

- **Technical Details**: {'MCQ Generation': '기존 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 단어 간의 유사성만을 평가하고, 교육적 가치를 고려하지 못합니다. 이를 해결하기 위해 KDA와 자동 평가 메트릭인 KDA_disc, KDA_cont를 제안하여 학습자의 지식을 평가하는 능력을 개선했습니다.', 'NLP Robustness': '기존 augmentation 방식은 spurious correlation(허용되지 않는 상관)에 영향을 받아 한계가 있었습니다. 이 논문에서는 여러 개의 counterfactual을 생성하고, collective decision(집합적 의사결정)을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안합니다.', 'Self-Play in Reinforcement Learning': '다중 에이전트 강화학습(MARL: Multi-Agent Reinforcement Learning)에 대한 복잡한 문제를 해결하기 위해 self-play(자기 대전) 기법이 주목받고 있습니다. 이 논문은 self-play의 기본 개념을 정리하고, 다양한 알고리즘을 체계적으로 분류하며, 실제 적용 사례들을 분석하는 한편, 향후 연구 방향을 제시합니다.'}

- **Performance Highlights**: {'MCQ Generation': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 사용성과 강한 상관관계를 가지고 있다는 점을 확인했습니다. 또한, n-gram 방식의 유사성 메트릭과 결합 시 다양한 전문가 레이블 MCQ 품질 측정에 대해 높은 예측력을 갖는다는 점을 보여주었습니다.', 'NLP Robustness': '대조학습 및 counterfactual augmentation을 통한 결과, 기존 방식보다 다양한 차원에서 견고성 향상, cross-domain generalization, 희소 데이터로부터의 generalization을 모두 크게 개선했습니다.', 'Self-Play in Reinforcement Learning': 'self-play 기법이 다양한 복잡한 강화학습 시나리오에서 성공적으로 인간 전문가를 능가하는 전략을 개발했음을 보여주었습니다. 체스, 포커, 비디오 게임 등에서의 응용 사례가 특히 주목됩니다.'}



### From Stem to Stern: Contestability Along AI Value Chains (https://arxiv.org/abs/2408.01051)
Comments:
          5 pages, 0 figure, to be held as a workshop at CSCW'24

- **Whats New**: 자동 MCQ 생성은 교사의 학습 평가 시간을 크게 줄일 수 있는 잠재력을 가지고 있으며, 기존의 평가 메트릭은 BLEU, ROUGE, METEOR 등이 데이터셋 내 골드 샘플과의 n-그램 유사성에 중점을 두는 데 반해, 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하여 MCQ의 대상 사실에 대한 학생의 지식을 평가하려는 노력이 주목되고 있다. 또한, 최근 NLP 태스크에서 딥 러닝 모델들의 초인적인 정확성에도 불구하고, spurious pattern에 의존하는 문제를 극복하기 위해 contrastive learning과 counterfactual augmentation을 활용하여 robustness를 높이는 방법이 제안되었다.

- **Technical Details**: KDA 메트릭은 데이터셋 내 골드 샘플과의 유사성을 평가하는 대신, MCQ의 대답 가능성을 평가한다. 이를 위해 우리는 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사한다. 또한, AI 시스템의 robustness를 높이기 위해 여러 개의 counterfactual을 생성하고 collective decisions을 통해 더 강인한 인과관계(supervise the causality)를 파악하는 방법을 제안한다.

- **Performance Highlights**: KDA_disc와 KDA_cont 메트릭은 human study를 통해 실제 강의실 세트에서 사용성과 강한 상관관계를 가지며, 전문가들이 라벨링한 MCQ 품질 측정에 대한 예측력이 뛰어난 것으로 나타났다. 또한, 제안된 방법은 counterfactual robustness, cross-domain generalization, generalization from scarce data 등 다양한 측면에서 기존 방법보다 유의미한 성능 향상을 보였다.



### Semantic Skill Grounding for Embodied Instruction-Following in Cross-Domain Environments (https://arxiv.org/abs/2408.01024)
- **What's New**: 이 논문에서 자동으로 다지선다형 문항(Multiple Choice Questions, MCQ)을 생성하는 새로운 방법을 제안하고, 이를 평가하기 위한 새로운 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 소개합니다. KDA는 MCQ의 대답 가능성을 측정하고 학생들의 지식을 평가할 수 있는 능력을 평가합니다. 또한, 강화된 방법론으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭도 제안합니다.

- **Technical Details**: 기존 평가 메트릭은 BLEU, ROUGE, METEOR와 같이 생성된 MCQ와 데이터셋 내의 금본 샘플 간의 n-그램 유사성에 중점을 둡니다. 그러나 이들은 MCQ의 교육적 가치를 평가하지 않습니다. KDA는 학생들이 특정 사실에 대한 지식을 가지고 MCQ에 대답할 수 있는 능력을 평가합니다. 우리는 먼저 인간 설문조사를 통해 KDA를 측정하고, 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 방식으로 KDA_disc와 KDA_cont 메트릭을 도입했습니다.

- **Performance Highlights**: 우리의 인간 연구를 통해 KDA_disc와 KDA_cont 메트릭이 실제 강의실 세팅에서 사용성과 강한 상관관계를 가지는 것을 확인했습니다. 또한, n-그램 기반 유사성 메트릭과 결합했을 때, 이들 메트릭은 다양한 전문가 레이블의 MCQ 품질 지표에 대해 강한 예측 능력을 보여주었습니다.

- **What's New 2**: 두 번째 논문은 NLP 태스크에서 발생하는 spurious pattern 문제를 해결하기 위해 대조학습(contrastive learning)과 counterfactual augmentation을 활용하는 방법을 제안합니다. 이는 인간이 데이터셋에 counterfactual을 추가하거나 기계가 이미 존재하는 counterfactual을 자동으로 매칭하는 기존 방법과 다르게 다수의 counterfactual을 생성하고 이 집합의 예측 분포를 통해 단어들의 인과관계를 더욱 정확하게 감독할 수 있습니다.

- **Technical Details 2**: 기존 방법들은 여전히 spurious correlation에 영향을 받지만, 이 논문에서는 집합적 의사 결정을 통해 더 넓은 범위의 counterfactual robustness, cross-domain generalization, 그리고 데이터가 부족한 상황에서도 성능을 크게 개선할 수 있는 방법을 제안합니다.

- **Performance Highlights 2**: 우리의 실험 결과, 집합적 의사 결정을 활용한 접근법이 기존 기법들보다 모델 바이아스에 덜 민감하여 다양한 측면에서 성능이 크게 향상됨을 확인했습니다.

- **What's New 3**: 세 번째 논문은 내재된 지시 실행(Embodied Instruction-Following, EIF)에서 사전 훈련된 언어 모델을 활용한 작업 계획의 어려움을 해결하기 위해 의미적 스킬 그라운딩 프레임워크인 SemGro를 제안합니다. SemGro는 다양한 도메인에서 의미적 스킬을 계층적으로 탈구(탈화)하여 새 도메인에 맞게 적용할 수 있는 방법을 제시합니다.

- **Technical Details 3**: SemGro는 상위 레벨의 의미적 스킬부터 시작하여 점점 하위 레벨로 내려가는 반복적 스킬 분해 접근 방식을 사용합니다. 이를 통해 각 계획된 스킬을 목표 도메인 내에서 실행 가능한 수준으로 그라운딩합니다. 이 과정에서 언어 모델의 논리적 추론 능력과 다중 모달 확장을 활용하여 스킬의 실행 가능성을 평가합니다.

- **Performance Highlights 3**: VirtualHome 벤치마크에서 EIF 에이전트 평가 결과, SemGro는 300개의 크로스 도메인 시나리오에서 기존의 최첨단 작업 계획 프레임워크인 LLM-Planner를 23.97% 개선하는 성과를 보였습니다.



### Piculet: Specialized Models-Guided Hallucination Decrease for MultiModal Large Language Models (https://arxiv.org/abs/2408.01003)
Comments:
          14 pages, 5 figures

- **MCQ Generation**: {"What's New": '새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)이 도입되었습니다. 이는 생성된 객관식 문제(MCQ)가 학생의 지식을 평가하는 데 얼마나 효과적인지를 측정합니다.', 'Technical Details': '기존 평가 메트릭인 BLEU, ROUGE, METEOR는 단순히 n-gram 유사성을 평가하므로 교육적 가치를 반영하지 못합니다. KDA는 학생 설문 조사를 기반으로 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되어 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human studies를 통해 KDA_disc와 KDA_soft가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 보여주었습니다. 전문가 라벨링된 다양한 MCQ 품질 지표에 대해 예측 능력이 향상되었습니다.'}

- **Counterfactual Augmentation**: {"What's New": '대조 학습과 반사실적 증강을 활용하여 NLP 태스크에서 모델의 견고성을 높이는 방법을 제공합니다. 기존 방법들과 달리 여러 반사실적 예를 생성하여 일괄적인 의사 결정을 통해 단어들의 인과관계를 보다 견고하게 파악합니다.', 'Technical Details': '이 접근법은 기존에 사람이 수동으로 반사실적 예를 추가하거나 데이터셋에서 반사실적에 가까운 예를 자동으로 매칭하는 방법에서 벗어나 여러 반사실적 예를 합성하고 이 집합을 기반으로 예측 분포를 결정하는 방식으로 진행됩니다.', 'Performance Highlights': '다양한 차원에서 유의미한 개선을 달성하였습니다. 방사실적 견고성, 도메인 간 일반화, 데이터 부족 상황에서의 일반화 성능 등이 포함됩니다.'}

- **MLLM Hallucinations Reduction**: {"What's New": 'Traning-free 방법인 Piculet를 도입하여 MLLM의 환각 현상을 줄이는 방법을 제안합니다. 이 방법은 기존의 training-free 방법과 달리 MLLM을 재훈련하지 않으면서도 입력 표현을 향상시킵니다.', 'Technical Details': 'Piculet은 여러 전문 모델을 사용하여 입력 이미지의 시각적 정보를 추출하고 이를 MLLM에 결합하여 MLLM의 출력이 이미지와 맞지 않는 환각 현상을 줄입니다. 이 방법은 추가 대형 모델 없이 수행되며, 다양한 MLLM에 적용 가능하도록 설계되었습니다.', 'Performance Highlights': 'LLaVA-QA90 벤치마크에서 Piculet을 활용한 모델 Qwen-VL-Chat의 정확도를 10점 만점에 6.1에서 7.3까지 높였습니다.'}



### A Safe Exploration Strategy for Model-free Task Adaptation in Safety-constrained Grid Environments (https://arxiv.org/abs/2408.00997)
- **What's New**: 이번 연구에서는 기존의 n-gram 기반 평가 메트릭들이 교육적 가치를 충분히 평가하지 못한다는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안합니다. 또한, 강화학습(RL) 에이전트의 탐색을 더 안전하게 만드는 새로운 프레임워크도 소개하였습니다.

- **Technical Details**: MCQ 생성의 KDA는 주어진 지식을 바탕으로 학생이 문제를 얼마나 잘 해결할 수 있는지를 측정하는 새로운 메트릭입니다. 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하여, 사전 학습된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다. 한편, 강화학습 에이전트의 안전한 탐색을 위해 안전 제약을 고려한 프레임워크를 제안합니다. 이 프레임워크는 사전 학습단계에서 잠재적으로 위험한 상태를 파악하게 하고, 이러한 상태들을 새로운 환경에서도 예측할 수 있도록 합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 확인하였습니다. 강화학습 에이전트는 새로운 그리드 환경에서 더 적은 안전 위반으로 새로운 작업에 적응할 수 있음을 보여주었습니다.



### On the Resilience of Multi-Agent Systems with Malicious Agents (https://arxiv.org/abs/2408.00989)
Comments:
          10 pages

- **Multiple Choice Questions**: [{"What's New": 'MCQ 자동 생성의 교육적 가치를 평가할 수 있는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability(KDA) 제안.'}, {'Technical Details': 'KDA는 학생의 대상 사실에 대한 지식을 기반으로 MCQ의 대답 가능성(answerability)을 평가한다. 학생의 응답을 통해 KDA 측정을 제안하고, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모사하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다.'}, {'Performance Highlights': 'KDA_disc와 KDA_cont는 교육 전문가에 의해 라벨링된 실제 강의실 세트에서 높은 상관관계를 보여준다. 추가로 n-gram 기반 유사성 메트릭과 결합시, 다양한 MCQ 품질 측정에 대한 예측력이 강하다.'}]

- **Deep Learning Robustness**: [{"What's New": '대조 학습과 반사실적 증강(contrastive learning and counterfactual augmentation)을 활용하여 NLP 태스크에서 deep model의 robustness를 향상시키는 방법 제안.'}, {'Technical Details': '기존 방식과 달리 여러 개의 counterfactual을 생성하고 집합적 의사결정(collective decisions)을 통해 단어들의 인과관계를 robust하게 감독하는 방법을 제안한다.'}, {'Performance Highlights': '이 접근 방식은 반사실적 robustness, cross-domain generalization, 희소 데이터의 generalization에서 상당한 개선을 달성한다.'}]

- **Multi-Agent Systems**: [{"What's New": '여러 에이전트 시스템에서 악성 에이전트에 대한 회복력을 연구하고, AutoTransform과 AutoInject를 사용하여 악성 에이전트를 자동으로 생성하는 방법 제안.'}, {'Technical Details': '다양한 시스템 구조(A→B→C, A↔B↔C 등)의 강인성을 조사하고, 각 구조의 시스템 회복력을 평가한다. 추가로, 메시지 리뷰 및 수정 에이전트 또는 메시지를 도전할 수 있는 메커니즘을 도입하여 시스템 회복력을 향상시키는 방안을 제안한다.'}, {'Performance Highlights': 'Hierarchical 구조는 가장 낮은 성능 저하(23.6%)를 경험하며, 추가 에이전트 도입 및 메시지 도전 메카니즘을 통해 시스템 회복력을 개선할 수 있음을 입증한다.'}]



### A SAT-based approach to rigorous verification of Bayesian networks (https://arxiv.org/abs/2408.00986)
Comments:
          Workshop on Explainable and Robust AI for Industry 4.0 & 5.0 (X-RAI) at European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (2024)

- **What's New**: 본 논문은 교육자가 학습 평가에 쏟는 시간을 줄여줄 수 있는 자동 선택형 문항(Multiple Choice Questions, MCQ) 생성 기술을 다룹니다. 기존 평가 메트릭인 BLEU, ROUGE, METEOR가 교육적 가치를 제대로 반영하지 못하는 문제를 해결하기 위해, 저자들은 새로운 자동 평가 메트릭 'Knowledge Dependent Answerability (KDA)'를 제안합니다. KDA는 MCQ의 답변 가능성을 측정하며, 학생이 해당 사실에 대해 얼마나 이해하고 있는지를 평가합니다.

- **Technical Details**: KDA 메트릭은 학생의 응답을 바탕으로 측정되며, 이를 흉내내기 위해 사전 훈련된 언어모델을 활용하여 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. KDA_disc와 KDA_cont는 학생들의 문제 풀이 행동을 모방하는 방식으로 KDA를 근사합니다. 실험 결과, 두 메트릭은 전문가가 라벨링한 실제 강의실 사용성 및 다른 MCQ 품질 지표와 강한 상관관계를 보였습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 라벨링 MCQ 품질 지표에 대해 강한 예측력을 보였습니다.



### Integrating ESG and AI: A Comprehensive Responsible AI Assessment Framework (https://arxiv.org/abs/2408.00965)
Comments:
          23 pages, 8 tables, 10 figures

- **What's New**: 이번 뉴스레터에서는 MCQ 자동 생성 평가 메트릭, 대조 학습 및 반사실적 증강법, 그리고 ESG와 AI의 통합 프레임워크에 대한 혁신적인 연구를 소개합니다.

- **Technical Details**: [{'title': '지식 종속 가능성(KDA) 평가 메트릭', 'details': '기존의 n-gram 기반 평가 메트릭인 BLEU, ROUGE, METEOR 대신, 학생이 대상 사실에 대한 지식을 가지고 있는 상태에서 문제를 풀 수 있는 능력을 평가하는 새로운 자동 평가 메트릭, KDA를 제안합니다. KDA_disc와 KDA_cont를 통해 학생의 문제 해결 행동을 모방하여 KDA를 자동화합니다.'}, {'title': '대조 학습 및 반사실적 증강법', 'details': '최근 NLP 모델들이 spurious pattern에 의존하여 robustness가 떨어지는 문제를 해결하기 위해 대조 학습과 반사실적 증강법을 사용합니다. 여러 개의 반사실적 데이터 셋을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 강력하게 감독합니다.'}, {'title': 'ESG-AI 프레임워크', 'details': '환경, 사회, 거버넌스(ESG) 요소를 AI 투자와 통합하는 ESG-AI 프레임워크를 소개합니다. 이 프레임워크는 28개 기업과의 협력을 통해 개발되었고, 실제 투자자들이 AI 사용의 물질성을 평가하는 데 도움을 줍니다. ESG-AI 툴킷은 2024년 4월에 공개되었습니다.'}]

- **Performance Highlights**: [{'title': 'KDA_disc와 KDA_cont의 평가 성능', 'details': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서 사용성과 강한 상관관계를 보였으며, 전문가가 레이블링한 다양한 MCQ 품질 지표에 대한 예측력도 뛰어납니다.'}, {'title': '대조 학습과 반사실적 증강법의 효과', 'details': '이 방법은 attribution-based synthesis의 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 및 scarce data에서의 일반화에 상당한 개선을 이루었습니다.'}, {'title': 'ESG-AI 프레임워크의 실용성', 'details': 'ESG-AI 프레임워크와 툴킷은 투자자들이 기업의 책임있는 AI 수행 여부를 평가할 수 있도록 도우며, 공공에게 공개된 이후로 투자 커뮤니티에서 긍정적인 피드백을 받았습니다.'}]



### Generalisation of Total Uncertainty in AI: A Theoretical Study (https://arxiv.org/abs/2408.00946)
Comments:
          9 pages

- **Automatic Multiple-Choice Question Generation Using Knowledge Dependent Answerability**: [{"What's New": '기존의 BLEU, ROUGE, METEOR 등 평가 메트릭의 한계를 극복하고, 교과서적 지식 평가 능력을 고려한 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안했습니다. 이 메트릭은 학생의 응답을 통해 MCQ의 답변 가능성을 측정합니다.', 'Technical Details': 'KDA는 지식 기반의 가능성을 평가하기 위해 학생 응답을 활용합니다. 또한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human study를 통해 KDA_disc와 KDA_cont가 실제 교실 내 사용성과 강한 상관관계를 가지고 있음을 입증하였으며, n-gram 기반 유사성 메트릭과 결합했을 때 매우 높은 예측력을 보였습니다.'}]

- **Enhancing NLP Model Robustness Using Contrastive Learning and Counterfactual Augmentation**: [{"What's New": '기존의 spurious pattern에 의존하여 robustness가 제한되는 문제를 해결하고자, 대조 학습 (contrastive learning)과 반사실적 증강 (counterfactual augmentation)을 활용하여 모델의 안정성을 향상시키는 새로운 접근법을 제안합니다.', 'Technical Details': '기존 방법들이 인간이나 모델에게 반사실적 데이터를 의존하는 것과 달리, 이 연구는 여러 개의 반사실적 데이터를 합성하고 집합적 의사 결정을 통해 인과관계를 더 robust하게 감독합니다.', 'Performance Highlights': '이 연구는 counterfactual robustness, 도메인 간 일반화 (cross-domain generalization), 그리고 부족한 데이터에서의 일반화 (generalization from scarce data)에서 큰 성과를 보였습니다.'}]

- **Addressing Uncertainty in AI: A Comprehensive Exploration**: [{"What's New": 'AI에서 불확실성 문제를 다루기 위해 새롭고 총체적인 불확실성 정의를 제안합니다. 이는 에피스테믹 (Epistemic)와 알레아토릭 (Aleatoric) 불확실성 두 가지 유형을 중심으로 다룹니다.', 'Technical Details': '불완전하거나 소란스러운 데이터, 여러 결과를 가질 수 있는 상황 등 다양한 불확실성을 다룰 수 있습니다. 이 논문은 불확실성에 대한 이론적 기반과 실용적 의미를 탐구하면서 에피스테믹 및 알레아토릭 불확실성의 최신 모델들(간격 모델, 랜덤 집합, 크레더블 집합 등)을 포함합니다.', 'Performance Highlights': '임계 데이터를 통한 새로운 불확실성 모델을 제안하여, 불확실성 추정 방법들(직접 간격 예측, 앙상블 모델, 베이지안 방법, 무작위 집합 및 신뢰 함수 모델)이 더 나은 결정을 할 수 있도록 돕습니다.'}]



### Granting GPT-4 License and Opportunity: Enhancing Accuracy and Confidence Estimation for Few-Shot Event Detection (https://arxiv.org/abs/2408.00914)
- **What's New**: 새로운 MCQ 평가 메트릭인 Knowledge Dependent Answerability (KDA)이 제안되었습니다. GPT-4를 이용한 자체 신뢰도 추정 향상 관련 연구도 소개되었습니다.

- **Technical Details**: 기존 MCQ 평가 메트릭인 BLEU, ROUGE, METEOR의 한계를 극복하기 위해, KDA는 학생의 응답을 통한 인간 설문 조사를 기반으로 측정 방법을 제시합니다. 추가로, KDA를 모사하는 pre-trained language models를 이용한 KDA_disc와 KDA_cont를 제안합니다. GPT-4의 신뢰도 추정을 개선하기 위해 License to speculate와 Opportunity to quantify (L&O) 방법을 이용했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ의 품질 평가와 강한 상관관계를 보였습니다. 또한, GPT-4의 경우 신뢰도 측정을 통해 AUC 0.759를 달성하며 정확성을 향상시켰습니다.



### Online Detection of Anomalies in Temporal Knowledge Graphs with Interpretability (https://arxiv.org/abs/2408.00872)
Comments:
          15 pages, 8 figures. Accepted by SIGMOD 2025 Round 2

- **What's New**: 기존의 다중 선택 질문(MCQ) 생성 평가 메트릭(BLEU, ROUGE, METEOR)은 단순히 n-그램 기반 유사성에 초점을 맞추고 있어 교육적 가치를 평가하지 않습니다. 이를 해결하기 위해 우리는 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭을 제안했습니다. 반면, 또 다른 논문에서는 NLP 태스크의 강력한 성능에도 불구하고 deep model들이 spurious pattern에 의존하는 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 강건성 향상 방안을 제안했습니다. 마지막으로, AnoT는 Temporal Knowledge Graphs(TKGs)에서의 노이즈 이슈를 해결하기 위한 새로운 이상탐지 메소드로, 인터프리터블한 룰 그래프를 통해 TKG의 복잡한 패턴을 효율적으로 요약하여 온라인 이상탐지를 수행합니다.

- **Technical Details**: KDA는 학생 설문조사를 통해 MCQ의 대답 가능성(answerability)을 측정하며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용해 이를 자동 평가한다. 또 다른 연구에서 제안한 방법은 다양한 counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 robust하게 검사한다. AnoT는 룰 그래프를 사용하여 새로운 지식을 노드로 매핑한 후, 이를 재귀적으로 탐색해 이상 점수를 산출하는 방식으로, 인터프리터블한 온라인 이상탐지를 지원합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 행사실 사용성과 강한 상관관계를 보이며, 전문 평가자의 MCQ 품질 평가와 높은 일치율을 보입니다. 새로운 방법은 다양한 차원에서 중요한 개선을 이루었으며, 특히 counterfactual robustness와 cross-domain generalization에서 두드러진 성능 향상을 보였습니다. AnoT는 4가지 실제 데이터셋에서 높은 정확도와 인터프리터빌리티로 기존 방법보다 평균 11.5% 높은 AUC와 13.6% 높은 precision을 달성했습니다.



### UlRe-NeRF: 3D Ultrasound Imaging through Neural Rendering with Ultrasound Reflection Direction Parameterization (https://arxiv.org/abs/2408.00860)
- **What's New**: 이 논문에서는 MCQ의 교육적 가치를 평가하는 새로운 메트릭인 Knowledge Dependent Answerability (KDA)을 제안했습니다. KDA는 학생이 대상 사실을 알고 있을 때 MCQ의 답변 가능성을 측정합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 메트릭과 달리, KDA는 MCQ가 학생의 지식을 평가할 수 있는지를 측정합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델로 학생의 문제 해결 행동을 모방함으로써 KDA를 대략적으로 측정합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가짐을 보여주었습니다. 또한, n-그램 기반의 유사성 메트릭과 결합될 때 전문가가 라벨링한 다양한 MCQ 품질 측정값에 대한 강력한 예측력을 제공함을 확인했습니다.



### LICM: Effective and Efficient Long Interest Chain Modeling for News Recommendation (https://arxiv.org/abs/2408.00859)
- **What's New**: 최근 논문들은 여러 NLP 및 뉴스 추천 시스템에서의 성능 향상을 다루고 있습니다. 우선, 기존의 MCQ 생성 평가 메트릭들이 교육적 가치를 고려하지 않는 점을 지적하며, 새로운 지식 종속 가능성(KDA) 메트릭을 제안하였습니다. 두 번째 논문은 대립 학습(contrastive learning)과 반사실적 강화(counterfactual augmentation)를 활용하여 모델의 robust성을 개선하려는 노력을 보여줍니다. 마지막 논문은 뉴스 추천 시스템에서 사용자의 장기 관심도를 반영하는 모델을 소개합니다.

- **Technical Details**: [{'Topic': 'MCQ Generation', 'Details': '기존의 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성에 초점을 맞추기 때문에 교육적 가치를 평가하지 못합니다. 지식 종속 가능성(KDA) 메트릭은 지식 기반으로 MCQ의 대답 가능성을 측정하여 학생의 지식 평가 능력을 평가합니다. 또한, KDA_disc와 KDA_cont라는 자동화된 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.'}, {'Topic': 'NLP Robustness', 'Details': '현재 NLP 모델들이 높은 정확성을 유지하고 있지만, spurious pattern에 의해 robustness가 제한됩니다. 이에, 대립 학습과 반사실적 강화(counterfactual augmentation)를 활용하여 이러한 문제를 개선하고자 합니다. 우리의 접근 방식은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 단어들의 인과 관계를 보다 robust하게 파악합니다.'}, {'Topic': 'News Recommendation', 'Details': '뉴스 추천 시스템에서 사용자의 장기 관심도를 반영하기 위해 Long Interest Chain Modeling (LICM) 방법을 제안합니다. 이는 유사한 사용자들 간의 협력 기반 글로벌 뉴스 클릭 그래프를 통해 이끌어낸 정보로 장기 관심도를 모델링합니다. 최종적으로, gated network를 이용해 장기 관심도와 이웃 관심도를 통합하여 최종 사용자 표현을 생성합니다.'}]

- **Performance Highlights**: [{'Model': 'KDA_disc와 KDA_cont', 'Details': 'Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며 여러 전문가가 평가한 MCQ 품질 측정에 대한 예측력이 뛰어남을 확인했습니다.'}, {'Model': 'Counterfactual Augmentation Approach', 'Details': '제시된 방법은 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization에서 상당한 개선을 보여주었습니다.'}, {'Model': 'LICM', 'Details': '실제 데이터셋 실험 결과, 제안된 모델이 뉴스 추천 성능을 효과적이고 효율적으로 향상시켰음을 검증했습니다.'}]



### Y Social: an LLM-powered Social Media Digital Twin (https://arxiv.org/abs/2408.00818)
Comments:
          29 pages, 5 figures

- **What's New**: 우리는 다양한 기술을 분석하여 온라인 교육 툴부터 소셜 미디어 시뮬레이션 및 NLP 모델의 안정성을 높이는 기술까지 폭넓은 최신 연구를 소개합니다.

- **articles**: [{"What's New": 'MCQ(객관식 질문) 자동 생성 평가를 위한 지식 종속 가능성(KDA) 메트릭을 도입', 'Technical Details': '새로 제안된 평가 메트릭 KDA는 MCQ의 교육적 가치를 평가하며 학생의 문제 해결 행동을 모방하기 위해 pre-trained language models(사전 훈련된 언어 모델)을 활용합니다. KDA_disc와 KDA_cont로 불리는 두 가지 자동 평가 메트릭은 인간 연구를 통해 실제 강의실 설정에서의 사용성과 강한 상관관계를 나타냈습니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 전문가들이 라벨한 다양한 MCQ 품질 측정에서 높은 예측력을 보여주었습니다.'}, {"What's New": '대조 학습 및 반사실적 증강을 활용한 NLP 모델의 안정성 향상 방법 제안', 'Technical Details': '이 접근법은 여러 개의 counterfactuals(반사실) 집합을 생성하여 collective decision(집합적 의사 결정)을 통해 단어들의 인과관계를 robust하게 분석합니다. 이는 spurious correlation(가짜 상관관계)에 덜 민감하여 다양한 차원에서 성능 개선을 보입니다.', 'Performance Highlights': '대조 학습과 반사실적 증강을 통해 counterfactual robustness(반사실적 강건성), cross-domain generalization(교차 도메인 일반화), 데이터가 부족할 때의 일반화에서 상당한 개선을 이루었습니다.'}, {"What's New": '디지털 트윈 Y를 통해 온라인 소셜 미디어 플랫폼 시뮬레이션', 'Technical Details': 'Y는 state-of-the-art Large Language Models(최첨단 대형 언어 모델)을 사용하여 사용자 상호 작용, 콘텐츠 전파, 네트워크 역학을 정확히 시뮬레이션할 수 있습니다. 이는 사용자 참여, 정보 확산, 플랫폼 정책의 영향을 연구하는 데 유용합니다.', 'Performance Highlights': 'Y의 LLM 통합을 통해 복잡한 온라인 상호 작용을 시뮬레이션하고, 다양한 학제간 연구에 대한 중요한 통찰력을 제공할 수 있습니다.'}]



### Prompt Recursive Search: A Living Framework with Adaptive Growth in LLM Auto-Prompting (https://arxiv.org/abs/2408.01423)
Comments:
          8 pages,4 figures

- **Multiple Choice Questions**: [{"What's New": '자동 MCQ 생성 평가 메트릭의 한계를 극복하기 위해 새로운 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안했습니다.', 'Technical Details': 'KDA는 학생들이 대상 사실에 대한 지식을 바탕으로 MCQ를 풀 수 있는지 평가합니다. KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안하여 학생의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human evaluation 결과, KDA_disc와 KDA_cont는 실제 강의실 환경에서 높은 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 평가 메트릭과 결합할 경우, 다양한 전문가가 평가한 MCQ 품질 지표에 대해 높은 예측력을 가졌습니다.'}]

- **Deep Model Robustness**: [{"What's New": '최근 deep model들이 NLP 태스크에서 사람보다도 높은 정확성을 보이지만, spurious patterns 때문에 robustness가 제한되고 있습니다. 이를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안합니다.', 'Technical Details': '기존 방법들은 사람이 일일이 counterfactual을 추가하거나 모델이 이미 있는 데이터에서 유사한 것들을 찾았으나 여전히 spurious correlations에 영향을 받았습니다. 이 논문에서는 여러 개의 counterfactual을 생성하고, collective decisions을 통해 더 강력한 단어 인과관계를 확보하는 방법을 제안합니다.', 'Performance Highlights': '제안한 방법은 다양한 측면에서 기존 방법보다 우수한 결과를 보였습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization 개선.'}]

- **Prompt Recursive Search**: [{"What's New": '새로운 Prompt Recursive Search (PRS) 프레임워크를 개발하여 대형 언어 모델(LLM)에서 문제 해결 과정에서 토큰을 효과적으로 절약하고 문제 복잡성을 평가하여 오류 발생 가능성을 줄이는 방법을 제안합니다.', 'Technical Details': '이 프레임워크는 문제의 복잡성을 평가하고 조정 가능한 구조를 포함하여 오류 발생 가능성을 줄입니다.', 'Performance Highlights': 'PRS 프레임워크는 Llama3-7B 모델을 사용한 BBH 데이터셋에서 Chain of Thought (CoT) 방법보다 8% 높은 정확도를 달성하여 22%의 성능 향상을 보였습니다.'}]



### Mission Impossible: A Statistical Perspective on Jailbreaking LLMs (https://arxiv.org/abs/2408.01420)
- **What's New**: 이번 연구에서는 전통적인 BLEU, ROUGE, METEOR와 같은 평가 메트릭이 교육적 가치를 평가하지 못한다는 점을 지적하며, 새로운 자동 평가 메트릭 KDA(Knowledge Dependent Answerability)를 제안했습니다. KDA는 대상 지식을 기반으로 학생이 문제를 답변할 수 있는지를 평가합니다.

- **Technical Details**: KDA는 학생들의 응답을 기준으로 측정하며, 자동 평가 메트릭 KDA_disc와 KDA_cont를 도입해 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 기존의 n-gram 기반 유사도 메트릭과 결합할 때, KDA_disc와 KDA_cont는 다양한 전문가 평가 기준에 대해 강한 예측력을 보입니다.

- **Performance Highlights**: Human studies를 통해 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 나타냈습니다. 이는 BLEU, ROUGE, METEOR와 같은 기존 메트릭을 보완할 수 있는 강력한 도구임을 시사합니다.



### Talk Less, Interact Better: Evaluating In-context Conversational Adaptation in Multimodal LLMs (https://arxiv.org/abs/2408.01417)
Comments:
          Accepted to COLM 2024

- **What's New**: [{'Title': '자동 MCQ 평가를 위한 지식 종속 가능성(KDA) 메트릭 제안', 'Summary': '기존의 BLEU, ROUGE, METEOR 평가 메트릭은 MCQ 생성의 교육적 가치를 반영하지 못합니다. 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)는 학생이 해당 사실에 대한 지식을 바탕으로 MCQ를 푸는 능력을 측정합니다. 인간 설문 기반의 평가를 통해 KDA와 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 높은 상관관계를 보인다는 것을 확인했습니다.'}, {'Title': '딥 모델의 대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Summary': '딥 모델이 NLP 과제에서 높은 정확성을 보였지만, 잘못된 패턴에 의존하여 로버스트니스가 제한됩니다. 비대칭 학습(contrastive learning)과 대안적 자극(counterfactual augmentation)을 적용하는 방법을 제안하며, 여러 counterfactual을 생성하여 집합적 의사 결정을 통해 단어의 인과관계를 파악합니다. 이 방법으로 다양한 차원에서 성능 향상을 이끌어 냅니다.'}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Summary': '대화가 진행됨에 따라 인간은 점점 더 효율적인 언어를 사용하게 됩니다. 이를 여러 state-of-the-art MLLMs (multimodal large language models)에서 테스트해 보니, 상대방의 효율적인 언어를 이해할 수는 있지만, 스스로 효율성을 증가시키는 능력은 보이지 않았습니다. GPT-4와 같은 모델에서만 특정 프롬프트를 통해 이 능력을 유도할 수 있었습니다.'}]

- **Technical Details**: [{'Title': '자동 MCQ 평가를 위한 KDA 메트릭', 'Details': 'KDA는 학생의 응답을 통해 MCQ의 지식 기반 대답 가능성을 평가합니다. KDA_disc와 KDA_cont는 KDA를 예측하기 위해 사전 학습된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한 메트릭입니다.'}, {'Title': '대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Details': "기존의 대안적 증강 방식을 개선하여 '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 보다 로버스트하게 인과관계를 지도합니다."}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Details': 'ICCA(Automated framework to evaluate conversational adaptation)를 사용하여 MLLMs의 문맥별 행동을 평가했습니다. GPT-4와 같은 모델은 특정 조건에서만 효율성을 스스로 증가시킬 수 있음을 발견했습니다.'}]

- **Performance Highlights**: [{'Title': '자동 MCQ 평가를 위한 KDA 메트릭', 'Highlights': 'KDA_disc와 KDA_cont는 실제 교실 설정에서 사용자인 전문가들에 의해 높은 예측력을 지녔습니다. 이는 n-gram 기반 유사성 메트릭과 결합하여 다양한 MCQ 품질 척도에서 높은 예측력을 가집니다.'}, {'Title': '대안적 자극 패턴을 이용한 NLP 로버스트니스 강화', 'Highlights': '집합적 의사 결정 방식을 통해 counterfactual robustness, cross-domain generalization, 창의적인 데이터 학습 측면에서 성능을 크게 향상시켰습니다.'}, {'Title': 'Multimodal 대형 언어 모델의 대화 적응성 평가', 'Highlights': '일부 MLLMs는 상대방의 효율적인 언어를 이해할 수 있지만, 대다수 모델은 자체적으로 언어 효율성을 증가시키는 능력이 부족합니다.'}]



### The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability (https://arxiv.org/abs/2408.01416)
- **What's New**: 새로운 MCQ 자동 생성 평가 메트릭인 지식 종속 가능성(KDA)을 제안하였습니다. 이는 학생의 지식을 평가하는 질문의 대답 가능성을 측정합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 같은 메트릭은 단순히 n-gram 유사성에 집중하였으나, KDA는 교육적 가치를 평가합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 방식을 모사합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 수업 환경에서 높은 사용성을 가지고 있음을 확인하였습니다. 그리고 이러한 메트릭들은 전문가들이 레이블링한 MCQ 품질 평가 기준과 강한 예측력을 보였습니다.



### Pre-trained Language Models Improve the Few-shot Prompt Ability of Decision Transformer (https://arxiv.org/abs/2408.01402)
Comments:
          2 figures, 8 tables. Accepted by the Training Agents with Foundation Models Workshop at RLC 2024

- **What's New**: {'first_paper': '이 연구에서는 자동 MCQ 생성의 교육적 가치를 평가하기 위해 기존의 BLEU, ROUGE, METEOR 메트릭을 보완하는 새로운 자동 평가 메트릭, 즉 지식 종속 가능성(KDA)을 제안했다. 이는 학생이 특정 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 측정하는 데 중점을 둔다.', 'second_paper': '본 연구는 NLP 태스크에서 deep model의 정확성이 높지만 spurious pattern에 기반한 제한된 robustness 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 방법을 제안한다.', 'third_paper': '이 연구에서는 사전 훈련된 언어 모델을 활용하여 Prompt Decision Transformer의 few-shot prompt 능력을 개선하는 새로운 프레임워크인 LPDT(Language model-initialized Prompt Decision Transformer)를 제안한다. 강력한 일반화 능력을 통해 unseen tasks에서의 성능을 높인다.'}

- **Technical Details**: {'first_paper': 'KDA는 학생 응답을 통해 MCQ의 대답 가능성을 측정하며, 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다. 이 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다.', 'second_paper': "기존의 반사실적 증강 방법은 사람이 직접 데이터셋에 반사실적을 추가하거나 모델이 자동으로 데이터셋에서 유사 반사실적을 찾는 방식을 사용했지만, 이는 여전히 spurious correlation에 영향을 받는다. 본 연구는 '여러 개의' 반사실적을 생성하고 집합적 의사 결정을 통해 robustness를 향상시키는 방법을 제안한다.", 'third_paper': 'LPDT 프레임워크는 사전 훈련된 언어 모델을 DT의 초기화 값으로 사용하며, Low-Rank Adaptation (LoRA)을 활용하여 파라미터의 일부만을 수정하면서 멀티태스크 RL(oRL) 데이터셋을 통해 모델을 미세 조정한다. 또한 prompt regularization을 도입하여 다양한 RL 태스크를 구별할 수 있도록 한다.'}

- **Performance Highlights**: {'first_paper': 'Human study 결과, KDA_disc와 KDA_soft가 실제 강의실 세트에서의 usability와 강한 상관관계를 나타냈으며, n-gram 유사성 메트릭과 결합했을 때 전문가가 라벨링한 다양한 MCQ 품질 지표에 대한 예측력이 높았다.', 'second_paper': '제안된 방법은 다양한 차원에서 현저한 성능 향상을 보여주었다. 여기에는 counterfactual robustness, cross-domain generalization, and generalization from scarce data가 포함된다.', 'third_paper': 'MuJoCo control 환경과 Meta World ML1 태스크에서의 종합 실험 결과, LPDT는 기존 기준 모델들 대비 향상된 성능을 나타내며, 특히 데이터가 제한적인 상황에서도 높은 누적 보상을 달성했다.'}



### PC$^2$: Pseudo-Classification Based Pseudo-Captioning for Noisy Correspondence Learning in Cross-Modal Retrieva (https://arxiv.org/abs/2408.01349)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 교육적 가치를 반영한 자동 MCQ 평가 메트릭, 반직관적 학습을 통한 NLP 모델 개선, 그리고 노이즈가 있는 데이터세트에서의 크로스 모달 검색을 위한 새로운 프레임워크가 각각 제안되었습니다.

- **Technical Details**: [{'Paper': 'Automatic MCQ Generation', 'Details': '기존의 BLEU, ROUGE, METEOR 메트릭의 한계를 극복하고자 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였습니다. 이는 목표 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': 'NLP 태스크에서 deep models의 robustness를 높이기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법을 제안했습니다. 이는 여러 개의 counterfactual을 생성하고, 집합적 의사 결정 (collective decisions)을 사용하여 모델의 인과관계를 감독합니다.'}, {'Paper': 'Cross-Modal Retrieval with Noisy Correspondence', 'Details': '노이즈가 있는 크로스 모달 검색을 개선하기 위해 Pseudo-Classification based Pseudo-Captioning (PC2) 프레임워크를 도입했습니다. 이는 캡션을 범주형 레이블로 해석하여 모델이 이미지-텍스트 의미적 유사성을 학습하게 하고, pseudo-captions을 생성하여 부정확한 데이터 쌍에 대해 더 정량적인 감독 정보를 제공합니다.'}]

- **Performance Highlights**: [{'Paper': 'Automatic MCQ Generation', 'Details': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 강한 상관관계를 가지며, 다양한 전문가가 레이블링한 MCQ 품질 지표에 대해 높은 예측력을 보였습니다.'}, {'Paper': 'Robustness in NLP Models', 'Details': '집합적 의사 결정을 통해 반직관적 학습을 수행하여 다양한 측면 (1. counterfactual robustness, 2. cross-domain generalization, 3. scarce data generalization)에 대한 성능이 향상되었습니다.'}, {'Paper': 'Cross-Modal Retrieval with Noisy Correspondence', 'Details': 'PC2 프레임워크는 시뮬레이션 및 실제 데이터셋에서 기존의 크로스 모달 검색 기법 및 NCL-robust 방법들을 뛰어넘는 성능을 보였습니다. 또한, Noise of Web (NoW) 데이터셋을 통해 새로운 강력한 NCL 벤치마크를 도입하였습니다.'}]



### StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation (https://arxiv.org/abs/2408.01343)
- **What's New**: 교사들이 학생 평가를 위해 MCQ(다중선택질문)를 자동으로 생성하는 방법에 있어, 교육적 가치를 고려하지 않는 기존 평가 지표들(BLEU, ROUGE, METEOR)를 보완하기 위해 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 또한, NLP의 deep model이 높은 정확도를 가지지만, spurious pattern에 의존해 robustness가 제한된 문제점을 보완하기 위해 contrastive learning와 counterfactual augmentation을 활용하는 방법을 제안합니다. 마지막으로, StitchFusion이라는 기존 사전 훈련된 모델을 그대로 활용하면서 정보 교환을 가능하게 하는 다중 모달 융합 프레임워크가 소개되었습니다.

- **Technical Details**: KDA는 학습자의 문제 해결 과정을 모방하기 위해 사전 학습된 언어 모델을 활용하여 KDA_disc와 KDA_cont로 평가되는 새로운 방식입니다. 또한, counterfactual augmentation에서는 '여러 개의' counterfactual을 생성하고 집합적 의사결정을 통해 robust한 인과관계를 유지하는 방법을 제안합니다. StitchFusion은 사전 훈련된 모델을 그대로 인코더 및 피처 퓨저로 사용해 다중 모달 정보를 융합시키는 프레임워크로, MultiAdapter라는 다방향 어댑터 모듈을 소개하여 인코딩 과정 중 모달 정보의 교환을 강화합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세팅과 높은 상관관계를 가지며, 다양한 전문가 레이블 MCQ 품질 측정에 대해 예측력이 강함을 확인했습니다. 제안된 counterfactual 방법은 기존의 attribution 기반 방법보다 모델 편향에 덜 민감하며 다양한 차원에서 significant improvements를 달성했습니다. StitchFusion 모듈은 기존 FFMs와 상호 보완적인 성격을 가지며, 네 가지 다중 모달 분할 데이터셋에서 최첨단 성능을 달성했습니다.



### Leveraging Knowledge Graph Embedding for Effective Conversational Recommendation (https://arxiv.org/abs/2408.01342)
Comments:
          26pages, 15figures

- **What's New**: MCQ (Multiple Choice Questions) 자동 생성에 대한 새로운 평가 메트릭과 강화된 강건성 (robustness)을 갖춘 NLP 모델, 그리고 knowledge graph 기반의 대화형 추천 시스템에 대한 새로운 접근법이 소개되었습니다.

- **Technical Details**: [{'MCQ 자동 생성': {'기존 평가 메트릭': 'BLEU, ROUGE, METEOR은 교육적 가치를 평가하지 않고, 단어 유사성에만 집중.', '새로운 메트릭': 'Knowledge Dependent Answerability (KDA)를 이용해 MCQ의 교육적 가치를 평가.', '자동 평가 방법': 'KDA_disc와 KDA_cont를 이용해 사전 학습된 언어 모델이 학생의 문제 해결 행동을 모방.'}}, {'강건성 강화 NLP 모델': {'문제점': '기존의 deep models는 spurious patterns에 의존하여 강건성이 부족.', '새로운 접근법': 'contrastive learning과 counterfactual augmentation을 이용, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 인과 관계 파악.'}}, {'대화형 추천 시스템 (CRS)': {'기존 문제점': '사용자와 속성, 아이템 간의 관계를 효과적으로 다루지 못함.', '새로운 접근법': 'Knowledge Graph 기반의 CRS (KG-CRS) 제안, 사용자-아이템 그래프와 아이템-속성 그래프를 통합한 동적 그래프로 대화 과정에서 갱신, 인접 노드를 통해 유용한 임베딩 학습.'}}]

- **Performance Highlights**: [{'MCQ 자동 생성': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 높은 상관관계를 보였으며, 다양한 전문가가 평가한 MCQ 품질 지표와도 높은 예측력을 가짐.', '강건성 강화 NLP 모델': 'Empirical 결과, 집합적 의사 결정을 통한 접근법이 반사적 치우침 (task model bias)에 덜 민감하고 counterfactual robustness, cross-domain generalization, sparse data generalization에서 높은 성과를 보임.'}, {'대화형 추천 시스템 (CRS)': '세 개의 실제 데이터셋에서 기존 최신 접근법 대비 대화 및 추천 태스크에서 월등한 성능 입증.'}]



### A Backbone for Long-Horizon Robot Task Understanding (https://arxiv.org/abs/2408.01334)
Comments:
          8 pages, 8 figures. This work is intended to be submitted to IEEE Robotics and Automation Letters (RA-L) for possible publication

- **What's New**: 이 연구에서는 기존의 평가 메트릭이 학습 평가에서 교육적 가치를 충분히 반영하지 못하는 문제를 해결하기 위해, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 나타냅니다. 또한, 논문에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악하는 새로운 방법도 제시합니다. 마지막으로, Therblig 기반의 백본 프레임워크(TBBF)를 통한 장기적 로봇 태스크의 이해와 전이성을 향상시키는 방법도 논의합니다.

- **Technical Details**: KDA는 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 포함하며, 이는 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 새롭게 제안된 로봇 학습 방법은 여러 개의 counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악합니다. TBBF는 Therblig (기본 동작 요소)을 사용하여 고수준 로봇 태스크를 초심적인 로봇 구성이 가능하도록 분해하여 학습의 이해도를 향상시킵니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA 및 전문가들에 의해 레이블링된 실제 강의실 세트에서 높은 사용성을 보였습니다. 로봇 학습 접근 방식에서는 다중 counterfactual을 통한 robust한 인과 관계 파악이 다양한 차원에서 유의미한 개선을 보였습니다. TBBF는 실제 세계에서의 다양한 상황에 대해 94.4%의 성공률과 복잡한 상황에서 80%의 성공률을 달성했습니다.



### A Robotics-Inspired Scanpath Model Reveals the Importance of Uncertainty and Semantic Object Cues for Gaze Guidance in Dynamic Scenes (https://arxiv.org/abs/2408.01322)
Comments:
          35+16 pages, 8+4 figures

- **What's New**: 최근 심층 모델(deep models)은 NLP 작업에서 인간 이상의 정확성을 보였음에도 불구하고, spurious patterns에 의존하여 robust하지 못한 문제를 가지고 있습니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 사용한 새로운 접근 방식을 제안합니다. 또한, 교육적인 가치를 평가할 수 있는 새로운 자동 평가 지표인 KDA (Knowledge Dependent Answerability)를 도입하여 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가할 수 있는 능력을 평가하는 방법을 발표하였습니다.

- **Technical Details**: 논문에서 제안한 방법은 여러 개의 반사실적 사례를 생성하고, 집합적인 의사 결정(collective decisions)을 통해 모델의 인과학적 해석을 강화하는 것입니다. 또한, 새로운 지표 KDA는 학생의 응답 결과를 기준으로 평가하며, 학생의 문제 해결 행동을 모방하는 사전 훈련된 언어 모델을 활용해 자동 평가 기준인 KDA_disc와 KDA_cont를 고안했습니다. 이는 기존의 BLEU, ROUGE, METEOR 지표와 달리 교육적 가치를 평가할 수 있다는 점이 특징입니다.

- **Performance Highlights**: 사람 대상 연구를 통해 새로운 평가 지표 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 가짐을 보였습니다. 또한, 다양한 전문가가 라벨링한 MCQ 품질 측정과 강한 예측력을 가짐을 확인하였으며, 집합적인 의사 결정방식의 반사실적 증강 방법이 1) 반사실적 강건성, 2) 도메인 간 일반화, 3) 희박한 데이터에서의 일반화에서 현저한 성능 향상을 이루었습니다.



### Synergistic pathways of modulation enable robust task packing within neural dynamics (https://arxiv.org/abs/2408.01316)
Comments:
          24 pages, 6 figures

- **What's New**: 이 논문은 MCQ(Multiple Choice Questions) 자동 생성과 평가를 위한 새로운 메트릭을 제안합니다. 기존의 MCQ 생성 평가 메트릭(BLEU, ROUGE, METEOR)은 단지 단어 유사성에 초점을 맞추었으나, 교육적 가치를 고려하지 않습니다. 따라서, 해당 논문에서는 지식 종속 가능성(KDA, Knowledge Dependent Answerability) 메트릭을 제안하여 MCQ의 대답 가능성을 평가합니다. 또한, NLP 태스크에서 최근의 deep model들의 robustness를 높이기 위한 새로운 접근법을 소개했습니다. 마지막으로, 뇌 네트워크가 다중 작업을 어떻게 동시에 학습하고 관리하는지 명확히 이해하기 위해서 recurrent neural network (RNN) 모델을 사용했습니다.

- **Technical Details**: MCQ 생성 평가와 관련해서 새로운 자동 평가 메트릭인 KDA가 제안되었습니다. 이는 주어진 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. KDA는 KDA_disc와 KDA_cont라는 두 자동 평가 메트릭을 도입하여, pre-trained language models를 활용해 학생들의 문제 해결 행동을 모방합니다. NLP 태스크에서, counterfactual augmentation과 contrastive learning을 사용하여 robustness를 강화하는 방법을 설명합니다. 뇌 네트워크 연구에서는, 두 가지 형태의 contextual modulation (neuronal excitability와 synaptic strength)을 비교하여 RNN의 multitasking 성능을 향상시키는 메커니즘을 분석했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가와 강한 상관관계를 지니며 실제 교육 현장에서의 사용성도 입증되었습니다. 이러한 메트릭은 기존의 n-gram 기반의 유사성 메트릭과 결합했을 때, 다양한 전문가가 레이블링한 MCQ 품질 지표에 대해 강한 예측력을 보여주었습니다. 반면, NLP 학습에서는 collective decisions을 통해 기존의 task model bias를 줄이고, counterfactual robustness, cross-domain generalization, scarce data generalization 측면에서 현저한 성능 향상을 이뤄냈습니다. 뇌 네트워크 연구에서는 contextual modulation을 통한 multitasking의 robustness와 효율성을 높이는 방법을 증명했습니다.



### A Decision-driven Methodology for Designing Uncertainty-aware AI Self-Assessmen (https://arxiv.org/abs/2408.01301)
- **MCQ Generation and Evaluation Advancement**: [{"What's New": '기존의 BLEU, ROUGE, METEOR 같은 평가 메트릭은 MCQ (Multiple Choice Questions)의 교육적 가치를 고려하지 않기 때문에, 우리는 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA: Knowledge Dependent Answerability)을 제안합니다.'}, {'Technical Details': 'KDA는 학생이 문제를 풀 때의 행동을 모방하는 사전 학습된 언어 모델을 활용해 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 사용합니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성 및 전문가가 라벨링한 MCQ 품질과 강한 상관관계를 가짐을 보였습니다.'}]

- **Robustness in NLP Tasks**: [{"What's New": '최근 deep model들이 NLP 태스크에서 높은 정확성을 보였으나, spurious pattern에 의존하는 문제 때문에 robustness가 제한됨을 보였습니다. 이를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 결합한 방법을 제안합니다.'}, {'Technical Details': '기존 방법과 달리 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다.'}, {'Performance Highlights': '우리의 방법은 대조 학습과 반사실적 증강을 통해 기존 방법보다 높은 counterfactual robustness, cross-domain generalization, 그리고 적은 데이터로부터의 일반화 성능을 보였습니다.'}]

- **AI Self-Assessment for Trustworthiness**: [{"What's New": 'AI의 예측에 대한 신뢰성을 높이기 위해 AI가 자체 평가(self-assess)를 통해 예측의 신뢰성을 스스로 측정하는 기술이 필요함을 강조합니다.'}, {'Technical Details': 'AI의 불확실성을 정량화하는 여러 기술들을 분류하고, 실무자가 각 기술을 선택하고 설계할 때 참고할 가이드를 제공합니다. 특히 하위 결정자에게 미치는 불확실성을 고려한 평가 기술을 강조합니다.'}, {'Performance Highlights': '제안된 방법론은 국가 이익 시나리오에서 AI의 신뢰성을 높이기 위한 도구로 활용될 수 있으며, 이는 AI의 예측이 중요한 결정에 영향을 미칠 때 특히 유용합니다.'}]



### 3DPX: Progressive 2D-to-3D Oral Image Reconstruction with Hybrid MLP-CNN Networks (https://arxiv.org/abs/2408.01292)
Comments:
          accepted by MICCAI 2024

- **What's New**: 이 아티클에서는 다양한 분야의 최신 연구 결과를 소개합니다. 첫 번째 연구는 자동 MCQ 생성의 교육적 가치를 평가하기 위한 새로운 메트릭 KDA (Knowledge Dependent Answerability)를 제안했습니다. 두 번째 연구는 contrastive learning과 counterfactual augmentation을 활용하여 NLP 모델의 robustness를 향상시키는 방법을 제안했습니다. 마지막으로, 세 번째 연구는 치과에서 사용되는 파노라마 X-레이(PX) 이미지에서 3D 구조를 재구성하기 위한 새로운 하이브리드 MLP-CNN 피라미드 네트워크를 제안했습니다.

- **Technical Details**: 첫 번째 연구에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. 두 번째 연구는 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 단어 간의 인과관계를 더 robust하게 파악하는 방법을 제안합니다. 마지막 연구에서는 2D 파노라마 X-레이 이미지를 3D로 재구성하기 위해 progressive hybrid Multilayer Perceptron (MLP)-CNN 피라미드 네트워크(3DPX)를 제안합니다. 이 접근법은 CNN의 즉각적인 주변 정보 수집 한계를 보완하는 데 MLP를 통합합니다.

- **Performance Highlights**: 첫 번째 연구에서 KDA_disc와 KDA_cont는 인간 평가 및 전문가 라벨링과 강한 상관관계를 보였으며, 다양한 전문가 라벨 측정 MCQ 품질에 대한 예측 능력이 우수했습니다. 두 번째 연구는 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 영역에서 유의미한 개선을 달성했습니다. 마지막 연구는 두 개의 대규모 데이터셋에서 기존 2D에서 3D로의 구강 재구성 방법을 능가했으며, 다운스트림 각도 불일치 분류 작업의 성능도 향상시켰습니다.



### The virtual CAT: A tool for algorithmic thinking assessment in Swiss compulsory education (https://arxiv.org/abs/2408.01263)
- **What's New**: 교사의 학생 평가 시간을 크게 줄여줄 수 있는 자동 다지선다형 문제(Multiple Choice Questions, MCQ) 생성의 새로운 평가 메트릭을 제안합니다. 이 메트릭은 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라 불리며, MCQ의 답변 가능성을 목표 사실에 대한 학생의 지식을 기반으로 측정합니다.

- **Technical Details**: 기존 BLEU, ROUGE, METEOR 메트릭은 n-gram 기반 유사성을 중시하지만 교육적 가치를 평가하지 못합니다. 우리는 KDA를 측정하기 위해 학생 응답을 활용한 인간 설문 조사 방법을 소개하며, 사전 학습된 언어 모델을 활용한 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이러한 메트릭은 학생들의 문제 해결 행동을 모방하여 KDA를 근사화합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 사용성뿐만 아니라 전문가로부터 레이블된 다양한 MCQ 품질 척도에 대한 예측 능력이 매우 높다는 것을 입증했습니다.



### Detection and Characterization of Coordinated Online Behavior: A Survey (https://arxiv.org/abs/2408.01257)
- **What's New**: 새로운 논문은 자동 출제 시스템의 효과를 개선하기 위한 새로운 평가 기준, 지식 종속 가능성(KDA)을 제안하였습니다. 이 측정기준은 MCQ(다중 선택 질문)의 대답 가능성을 평가하고 교육적 가치를 높이는 데 중점을 둡니다.

- **Technical Details**: KDA는 학생들이 특정 지식에 기반하여 MCQ에 대답할 수 있는지를 측정합니다. 이를 위해, 논문은 사람들의 응답을 바탕으로 KDA를 측정하고, 사전 학습된 언어 모델을 사용해 학생들의 문제 해결 행동을 모방하여 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이들은 BLEU, ROUGE와 같은 기존 n-gram 기반 유사도 메트릭과 결합하여 전문가가 평가한 MCQ 품질 측정치에 대한 예측력을 높입니다.

- **Performance Highlights**: 논문은 사람 연구에서 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가진다는 것을 입증하였습니다. 또한, 이들 메트릭은 교육 전문가가 평가한 MCQ 품질 측정치에 대한 강력한 예측력을 나타냅니다.



### Deep progressive reinforcement learning-based flexible resource scheduling framework for IRS and UAV-assisted MEC system (https://arxiv.org/abs/2408.01248)
Comments:
          13 pages, 10 figures

- **What's New**: 이 논문은 교사의 학습 평가 시간을 줄이기 위해 자동 Multiple Choice Questions(MCQ)를 생성하는 방법을 다룬다. 기존 평가 메트릭인 BLEU, ROUGE, METEOR는 교육적 가치를 고려하지 않는 문제점을 가지고 있었으나, 본 논문에서는 새로운 평가 메트릭인 Knowledge Dependent Answerability(KDA)를 제안하여 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가하는 능력을 평가한다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 입증하였다. 또한 KDA_disc와 KDA_cont를 n-gram 기반의 유사성 메트릭들과 결합했을 때 다양한 전문가 라벨의 MCQ 품질 측정에서 예측력이 강함을 보여준다.



### Tailoring Graph Neural Network-based Flow-guided Localization to Individual Bloodstreams and Activities (https://arxiv.org/abs/2408.01239)
Comments:
          7 pages, 9 figures, 2 tables, 16 references, accepted at ACM NanoCom'25

- **What's New**: 자동 다지선 객관식 문제 생성에 대한 혁신적인 평가 척도인 Knowledge Dependent Answerability (KDA) 제안. 이 논문은 BLEU, ROUGE, METEOR 같은 기존 평가 메트릭이 교육적 가치를 평가하지 못하는 문제를 해결하고자 한다. 또한, NLP 태스크에서의 모델 강인성 향상을 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation) 기법을 도입한다. 마지막으로 인체 내 나노디바이스를 이용한 진단 이벤트 위치 추적을 개선하기 위해 신체 생리 지표를 기반으로 GNN(Graph Neural Networks) 모델을 조정하는 방식을 제안한다.

- **Technical Details**: {'KDA': 'KDA는 학생들이 대상 사실에 대한 지식을 얼마나 잘 평가할 수 있는지 측정한다. 이를 위해 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안, 사전 학습된 언어 모델을 사용해 학생들의 문제 해결 행동을 모방한다.', 'Counterfactual Augmentation': '대조 학습과 반사실적 증강을 통해 강인성을 높인다. 반사실적 생성 집합을 만들어 집합적 의사 결정을 통해 과적합을 피하고 여러 차원에서 성능을 향상시킨다.', 'GNN Adaptation': '생리 지표(키, 몸무게, 심박수)를 포함한 GNN 조정 파이프라인을 제안해 개별 환자의 혈류와 활동 변화에 대응하는 정확한 위치 추적을 가능하게 한다.'}

- **Performance Highlights**: {'KDA': 'Human studies를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가가 라벨링한 실제 교실 세트에서의 사용성 측정치와 높은 상관관계를 가짐을 밝혔다.', 'Counterfactual Augmentation': '대조적 집합 결정을 통해 반사실적 강인성, 크로스 도메인 일반화, 그리고 제한된 데이터에서의 일반화 성능을 획기적으로 개선했다.', 'GNN Adaptation': '기존 모델 대비 다양한 신체 타입과 활동 수준에서 향상된 성능을 보였으며, GNN 적응 확장을 통해 맞춤형 및 지속적 진단 위치 추적 가능성을 제시했다.'}



### High-Throughput Phenotyping of Clinical Text Using Large Language Models (https://arxiv.org/abs/2408.01214)
Comments:
          Submitted to IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI), Houston TX

- **What's New**: 최근 deep model들이 NLP 태스크에서 사람보다 나은 정확성을 보였으나 spurious pattern에 의존하는 문제로 인해 robustness가 제한적이라는 문제를 해결하고자 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 새로운 접근법을 제안했습니다. 또한, 대형 언어 모델(LLMs)을 이용하여 자동적으로 고속 페노타이핑(high-throughput phenotyping)을 구현하는 연구도 발표되었습니다.

- **Technical Details**: ['MCQ 생성: 현재 사용되는 BLEU, ROUGE, METEOR 등의 메트릭이 교육적 가치를 반영하지 못하는 문제를 해결하기 위해, Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안했습니다. 이는 학생이 특정 사실을 알고 있을 때 MCQ에 답할 수 있는 가능성을 측정합니다.', '이 논문은 KDA를 기반으로 한 평가 방법을 학생들의 응답을 통해 측정한 후, 사전 훈련된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다.', '대조 학습과 counterfactual augmentation: 기존의 방법에서는 인간이나 기계가 counterfactual을 수집해야 하는 문제를 해결하고자 다수의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 보다 robust하게 파악하는 방법을 제안했습니다.', '고속 페노타이핑: GPT-4와 GPT-3.5-Turbo를 사용하여 OMIM 데이터베이스의 임상 요약에서 표현된 페노타입을 자동으로 식별, 분류 및 정규화하는 성능을 평가하였습니다. 특히 GPT-4가 높은 성능을 보였습니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며, n-gram 기반의 유사성 메트릭과 결합할 때 다양한 MCQ 품질 측정 요소에 대한 높은 예측력을 보였습니다.', '대조 학습 기반 접근법으로 인해 counterfactual robustness, cross-domain generalization, 그리고 scarce data 상황에서의 일반화 능력에서 상당한 개선을 달성했습니다.', 'GPT-4는 페노타입 식별, 분류, 정규화에서 GPT-3.5-Turbo를 능가하며, 수동 어노테이터와의 일치 정도에서 높은 성과를 보였습니다. 특히, GPT-4의 광범위한 사전 학습 덕분에 추가적인 수동 어노테이션 데이터 없이도 높은 성능과 일반화 능력을 보여줬습니다.']



### Optimizing Variational Quantum Circuits Using Metaheuristic Strategies in Reinforcement Learning (https://arxiv.org/abs/2408.01187)
Comments:
          Accepted at QCE24 - QCRL24 Workshop

- **What's New**: Quantum Reinforcement Learning (QRL)에서의 메타휴리스틱 알고리즘 통합을 통해 QRL의 효율성을 극대화하는 방법을 탐구하였습니다. 특히 Particle Swarm Optimization, Ant Colony Optimization, Tabu Search, Genetic Algorithm, Simulated Annealing, Harmony Search와 같은 알고리즘을 QRL에 적용해 봅니다.

- **Technical Details**: QRL에서 전통적 gradient 기반 방법이 비효율적인 이유는 flat solution landscapes와 vanishing gradients 때문입니다. 이에 따라 gradient-free 방법이 필요한데, 본 논문에서는 여러 metaheuristic 알고리즘을 variational quantum circuits (VQC) 파라미터 최적화에 적용해 비교합니다. 이를 통해 다양한 RL 환경에서의 메타휴리스틱 알고리즘의 효율성을 평가합니다. OpenAI Gym의 Cart Pole과 MiniGrid 환경에서 실험을 진행하였습니다.

- **Performance Highlights**: 5x5 MiniGrid 강화 학습 환경에서는 Simulated Annealing과 Particle Swarm Optimization이 최적의 결과를 보였으며, Cart Pole 환경에서는 Simulated Annealing, Genetic Algorithm, Particle Swarm Optimization이 최적의 결과를 기록했습니다. 이 결과는 PSO와 SA가 QRL 학습의 효율성을 보장할 수 있음을 시사합니다.



### Misinforming LLMs: vulnerabilities, challenges and opportunities (https://arxiv.org/abs/2408.01168)
- **What's New**: 이번 뉴스레터에서는 세 가지 흥미로운 논문을 소개합니다: 자동 MCQ 생성 평가 지표, NLP의 강건성 향상을 위한 대조 학습, 그리고 거대 언어 모델(LLM)의 신뢰성 문제입니다.

- **Technical Details**: [{'내용': '첫 번째 논문에서는 BLEU, ROUGE, METEOR와 같은 기존 지표가 교육적 가치를 평가하지 않는다는 점을 지적하고, 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭을 제안합니다. 이는 MCQ의 대답 가능성을 평가하고, KDA_disc와 KDA_cont라는 자동 평가 지표를 통해 사용성과 강한 상관관계를 보장합니다.'}, {'내용': '두 번째 논문에서는 NLP 작업에서 홍채 패턴에 의존한 강건성의 한계를 지적하고, 대조 학습과 반사실적 증강(counterfactual augmentation)을 활용하여 이 문제를 해결하고자 합니다. 여러 반사실적 예제를 생성하여 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다.'}, {'내용': '세 번째 논문은 거대 언어 모델의 내부 메커니즘에 대해 설명하면서, LLM의 신뢰성 문제를 논의합니다. 통계적 패턴에 의존하는 LLM의 한계를 극복하기 위해 생성형 트랜스포머 기반 모델을 사실 기반 및 논리 프로그래밍 언어와 결합하는 연구가 진행되고 있다고 소개합니다.'}]

- **Performance Highlights**: [{'내용': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실에서의 사용성과 매우 강한 상관관계를 보여, MCQ 품질 예측력이 높다는 것을 입증하였습니다.'}, {'내용': '제안된 대조 학습 기반 방법은 반사실적 강건성, 도메인 간 일반화, 적은 데이터로부터의 일반화 성능이 크게 향상되었음을 실증적으로 보여주었습니다.'}, {'내용': '사실 기반 정보와 지식 그래프를 사용한 LLM 성능의 개선 가능성, 생성형 코드 작성 등의 연구가 신뢰할 수 있는 LLM의 미래를 제시하고 있습니다.'}]



### TCR-GPT: Integrating Autoregressive Model and Reinforcement Learning for T-Cell Receptor Repertoires Generation (https://arxiv.org/abs/2408.01156)
- **What's New**: 이 논문은 TCR-GPT라는 새로운 확률적 모델을 제안했으며, 이는 디코더 전용 변환기 구조를 기반으로 하여 TCR 레퍼토리(면역계에서 T 세포 수용체의 집합)의 시퀀스 패턴을 발견하고 재현하는 데 중점을 둡니다. TCR-GPT는 특정 펩타이드(단백질 조각)를 인식할 수 있는 TCR 시퀀스를 생성하는 데 강화 학습(Reinforcement Learning, RL)을 활용합니다.

- **Technical Details**: TCR-GPT는 디코더 전용 변환기(decoder-only transformer) 아키텍처에 기반하여 TCR 시퀀스의 확률 분포를 학습하고 새로운 시퀀스를 생성합니다. 이 모델은 Pearson 상관 계수로 측정된 시퀀스 확률 분포 추론에서 0.953의 정확도를 나타냅니다. RL을 사용하여 TCR 시퀀스를 특정 펩타이드를 인식할 수 있는 레퍼토리로 조정하였습니다. 강화 학습은 고정된 분포 모델링 목표와 달리, 보상 함수(reward function)를 통해 목표를 적응적으로 설정할 수 있는 장점을 제공합니다.

- **Performance Highlights**: TCR-GPT는 기존 모델과 비교하여 시퀀스 확률 분포 추론 정확도가 높으며, RL을 통한 미세 조정으로 특정 펩타이드를 인식할 가능성이 높은 TCR 레퍼토리를 생성하는 능력이 검증되었습니다. 이는 타겟 면역 치료 및 백신 설계의 진보에 중요한 잠재력을 제공합니다.



### DERA: Dense Entity Retrieval for Entity Alignment in Knowledge Graphs (https://arxiv.org/abs/2408.01154)
- **What's New**: 이번 연구는 다중 선택 문제 (MCQ) 자동 생성의 평가 메트릭을 개선하기 위해 지식 종속 대답 가능성 (Knowledge Dependent Answerability, KDA)을 제안하였습니다. 현재의 평가 메트릭은 교육적 가치를 고려하지 않으며, 단순히 생성된 MCQ와 데이터셋의 골드 샘플 간의 n-gram 유사도만을 측정합니다. KDA는 해당 목표 사실에 대한 학생의 지식 평가 능력을 강조하여 측정합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 자동화하기 위해, 사전 훈련된 언어 모델 (pre-trained language models)을 활용하여 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안하였습니다. 이러한 메트릭은 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: Human evaluation을 통해, KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지며 기존의 n-gram 기반 유사성 메트릭과 결합했을 때, 다양한 전문가가 평가한 MCQ 품질 측정치에 대해 높은 예측력을 가지고 있음을 확인했습니다.



### A Survey of Mamba (https://arxiv.org/abs/2408.01129)
- **paper1**: {"What's New": '지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안하여 자동 생성된 MCQ의 교육적 가치를 평가합니다. KDA는 학생의 대상 사실에 대한 지식을 평가하고, BLEU, ROUGE, METEOR와 같은 기존의 n-gram 기반 유사성 메트릭의 한계를 극복합니다.', 'Technical Details': 'KDA는 학생들의 실제 응답을 바탕으로 측정되며, 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 두 메트릭은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하도록 설계되었습니다.', 'Performance Highlights': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, 전문가가 라벨링한 MCQ 품질 측정값에 대해 높은 예측력을 보입니다.'}

- **paper2**: {"What's New": '딥 러닝 모델의 강건함(robustness)을 높이기 위해 Contrastive Learning과 반사실적 증강(Counterfactual Augmentation)을 활용하는 새로운 접근법을 제안합니다. 여러 개의 Counterfactual을 생성하고, 집합적 의사 결정(collective decisions)을 통해 모델의 인과관계를 더 강력하게 감독합니다.', 'Technical Details': '기존 방법들이 사람의 직접적인 개입이나 기계의 자동 매칭에 의존한 반면, 새로운 방법은 Counterfactual을 여러 개 생성하고 집합적 의사 결정을 통해 예측 분포를 평가합니다.'}

- **paper3**: {"What's New": '딥 러닝의 주요 구성 요소로서 Transformer의 한계를 극복할 수 있는 Mamba라는 새로운 아키텍처가 등장했습니다. Mamba는 순차 데이터의 복잡성을 효율적으로 처리하면서 선형적인 확장성을 유지합니다.', 'Technical Details': 'Mamba는 고전적인 상태 공간 모델(State Space Models, SSMs)에서 영감을 받은 구조로, Recurrent Neural Networks와 Convolutional Neural Networks의 장점을 결합하여 선형 또는 근선형 확장성을 달성합니다.', 'Performance Highlights': 'Mamba는 A100 GPU에서 최대 3배 빠른 계산 속도를 보이며, Transformer와 비슷한 모델링 능력을 가지고 있어 다양한 연구 및 응용 분야에서 주목받고 있습니다.'}



### BioRAG: A RAG-LLM Framework for Biological Question Reasoning (https://arxiv.org/abs/2408.01107)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서는 생명 과학 연구를 위한 질문-응답 시스템을 향상시키기 위해 BioRAG라는 새로운 Retrieval-Augmented Generation(RAG) 프레임워크를 도입했습니다. 이 시스템은 2200만 개의 과학 논문을 분석하고 인덱싱하여 대규모 언어 모델(LLMs)을 기반으로 훈련되었습니다. BioRAG는 도메인별 지식 계층 구조와 최신 데이터를 통합하여 검색 및 응답 생성을 개선합니다.

- **Technical Details**: BioRAG 프레임워크는 먼저 방대한 생명 과학 논문을 파싱, 인덱싱 및 세분화하여 고품질 훈련 코퍼스를 생성합니다. 이 후 생명 과학 도메인에 특화된 임베딩 모델을 훈련시키며, 외부 정보 소스를 포함하여 최신 데이터를 가져오는 것이 특징입니다. 또한 쿼리 전처리, 검색기 실행 컴포넌트, 모델이 충분한 정보를 수집하는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과, BioRAG는 여러 생명 과학 질문 응답 작업에서 맞춤형 LLM, 검색 엔진과 결합된 LLM, 다른 과학적 RAG 프레임워크를 뛰어넘는 성능을 보였습니다. 구체적으로, 컬렉티브 디시전 (collective decisions)을 통한 합성 counterfactual 생성 시, 상위 성능을 보이며 다양한 차원에서 강력한향상을 달성했습니다.



### Contribution-based Low-Rank Adaptation with Pre-training Model for Real Image Restoration (https://arxiv.org/abs/2408.01099)
Comments:
          33 pages, 15 figures, for homepage see this url : this https URL

- **What's New**: 이번 뉴스레터에서는 세 가지 새로운 인공지능 연구 논문을 소개합니다. 첫 번째 논문은 자동 다중 선택 질문(MCQ) 생성의 새로운 평가 메트릭을 제안합니다. 두 번째 논문은 대조 학습과 반사실적 증강(counterfactual augmentation)을 활용하여 NLP 모델의 견고성을 높이는 방법을 다룹니다. 마지막으로 세 번째 논문은 저수준 컴퓨터 비전에서 새로운 전이 학습(pre-training) 및 효율적인 파라미터 튜닝 방법을 제안하여 이미지 복원 성능을 향상시킵니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 등 평가 메트릭이 MCQ의 교육적 가치를 고려하지 못하는 문제를 해결하기 위해, 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 평가 메트릭을 제안합니다. KDA는 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 이를 근사합니다.', 'Robustness in NLP Models': 'NLP 태스크에서 모델의 견고성을 높이기 위해 대조 학습 및 반사실적 증강을 활용합니다. 우리는 여러 개의 반사실적 데이터를 생성하고 이를 기반으로 집합적 의사 결정을 통해 단어들의 인과관계를 파악하는 방법을 제안합니다.', 'Image Restoration': '저수준 컴퓨터 비전에서 다양한 이미지 복원 작업을 위해 CoLoRA(contribution-based low-rank adaptation) 및 PROD(Pre-training with Random Order Degradation)라는 새로운 방법을 제안합니다. CoLoRA는 각 작업에 대해 적응적으로 계층(layer)별 용량을 결정하여 적은 수의 파라미터만을 튜닝합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 전문가에 의해 라벨링된 다양한 MCQ 품질 측정에서 강력한 예측력을 가지고 있음을 증명하였습니다.', 'Robustness in NLP Models': '우리의 접근법은 반사실적 견고성(Counterfactual Robustness), 크로스 도메인 일반화(Cross-domain generalization), 희소한 데이터에서의 일반화(Generalization from scarce data) 등 다양한 차원에서 기준 모델 대비 성능을 크게 개선하였습니다.', 'Image Restoration': 'CoLoRA와 PROD는 다양한 네트워크 아키텍처(CNN 및 Vision Transformers)와 함께 사용할 수 있으며, 실제 데이터 기반 6가지 이미지 복원 작업에서 최첨단 성능을 달성하였습니다.'}]



### Six Dragons Fly Again: Reviving 15th-Century Korean Court Music with Transformers and Novel Encoding (https://arxiv.org/abs/2408.01096)
Comments:
          Accepted at the 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: 교사의 학습 평가 시간을 줄이기 위해 자동 MCQ 생성을 사용할 수 있으나, 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 고려하지 않는 문제를 해결하기 위해 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 MCQ가 학생의 지식에 기반하여 대답할 수 있는지 평가합니다.

- **Technical Details**: KDA는 학생 반응(인간 설문) 기반으로 측정하며, KDA_disc와 KDA_cont라는 자동 평가 메트릭은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다. 이를 통해 n-gram 기반 유사성 메트릭과 결합하여 더 나은 예측 성능을 발휘합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 측정 항목에 대해 강한 예측력을 나타냈습니다.



### The EAP-AIAS: Adapting the AI Assessment Scale for English for Academic Purposes (https://arxiv.org/abs/2408.01075)
- **What's New**: 이번 뉴스레터에서는 세 가지 흥미로운 논문을 소개합니다. 첫째, 자동 Multiple Choice Questions(MCQ) 생성 시스템의 평가를 개선하는 새로운 메트릭, Knowledge Dependent Answerability(KDA)를 도입한 연구, 둘째, contrastive learning과 counterfactual augmentation을 이용해 NLP 모델의 robustness를 개선하는 방법을 탐구한 연구, 셋째, 영어 학술 목적 학습(EAP) 환경에서 Generative AI(GenAI) 도구를 통합한 새로운 평가 프레임워크, EAP-AIAS를 제안한 연구입니다.

- **Technical Details**: [{'title': 'Automatic MCQ Generation Evaluation', 'details': '기존의 BLEU, ROUGE, METEOR 메트릭은 MCQ의 교육적 가치를 평가하지 못하는 한계를 가지고 있습니다. 이를 해결하기 위해, 우리는 학생의 지식에 기반하여 MCQ의 대답 가능성을 측정하는 새로운 자동 평가 메트릭, KDA를 제안합니다. 이는 human survey를 통해 측정한 KDA와 유사한 결과를 자동으로 평가하기 위해 KDA_disc와 KDA_cont를 사용합니다.'}, {'title': 'Robust NLP Models via Counterfactual Augmentation', 'details': '최근의 deep model들이 spurious pattern에 의존함으로 인해 robustness가 제한되는 문제를 해결하기 위해, 우리는 contrastive learning과 counterfactual augmentation을 사용하여 robustness를 향상시키는 방법을 제안합니다. 이 방법은 여러 개의 counterfactual을 생성하고, 이들 집합에 대한 예측 분포를 통해 단어들의 인과관계를 더욱 견고하게 파악합니다.'}, {'title': 'GenAI Integration in EAP with EAP-AIAS', 'details': 'Generative AI(GenAI)를 EAP 환경에 통합하기 위해, 우리는 특별히 맞춤형 AI 평가 스케일, EAP-AIAS를 제안합니다. 이 프레임워크는 GenAI 도구의 적절한 사용을 5단계로 나눠, EAP 과제에서의 AI 사용을 구조화된 방식으로 평가합니다. 이를 통해 학문적 성실성을 유지하면서 언어 개발을 지원합니다.'}]

- **Performance Highlights**: [{'title': 'Automatic MCQ Generation Evaluation', 'highlights': 'KDA_disc와 KDA_cont는 전문가가 강의실 환경에서 평가한 사용성과 강한 상관관계를 가지며, 기존의 n-gram 메트릭과 결합할 때 다양한 MCQ 품질 측정에서 높은 예측력을 보입니다.'}, {'title': 'Robust NLP Models via Counterfactual Augmentation', 'highlights': '우리의 접근법은 task model의 bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 generalization에서 의미 있는 향상을 이룹니다.'}, {'title': 'GenAI Integration in EAP with EAP-AIAS', 'highlights': 'EAP-AIAS는 다양한 EAP 평가 유형에서 GenAI 도구의 적절한 사용을 평가하며, EAP 실무자들이 교육 복잡성을 다룰 수 있게 도와주는 유연한 프레임워크를 제공합니다.'}]



### LLM as Runtime Error Handler: A Promising Pathway to Adaptive Self-Healing of Software Systems (https://arxiv.org/abs/2408.01055)
- **What's New**: 이 논문에서는 자동 MCQ(Multiple Choice Questions) 생성의 평가 메트릭에 대한 새로운 접근 방식을 제안하고 있습니다. 기존 메트릭들이 교육적 가치를 평가하지 못하는 문제를 해결하기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 메트릭을 도입하였습니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 기반으로 학생들의 지식을 측정합니다. 이 메트릭은 인간 설문조사를 통해 학생 반응을 기반으로 KDA를 측정하고, 사전 학습된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: 휴먼 스터디를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 함께 사용될 때 다양한 MCQ 품질 측정항목에 대해 강력한 예측력을 보여주었습니다.



### GNN-MolKAN: Harnessing the Power of KAN to Advance Molecular Representation Learning with GNNs (https://arxiv.org/abs/2408.01018)
- **What's New**: [{'title': '신규 MCQ 평가 메트릭 KDA 제안', 'content': '자동 MCQ(Multiple Choice Questions) 생성의 교육적 가치를 평가하는 새로운 메트릭, 지식 종속 가능성(KDA)을 제안하였으며, 이는 대상 사실에 대한 학생의 지식을 평가하는 능력을 중심으로 MCQ의 대답 가능성을 측정합니다.'}, {'title': '대조 학습 및 반사실 증강을 통한 NLP 모델의 견고성 강화', 'content': '대조 학습과 반사실 증강을 사용하여 NLP 모델의 견고성을 강화하는 방법을 제안하였으며, 이는 여러 개의 반사실을 생성하고 집합적 의사 결정을 통해 각각의 용어의 인과관계를 더욱 견고하게 감독합니다.'}, {'title': 'GNN-MolKAN를 통해 분자 표현 학습의 성능 개선', 'content': '신규 GNN(Graph Neural Network) 모델인 GNN-MolKAN과 향상된 버전 GNN-MolKAN+를 제안하였으며, 이는 Kolmogorov-Arnold Networks(KAN) 아키텍처를 통합하여 기존 GNN의 구조적 한계를 극복하고 성능을 크게 개선합니다.'}]

- **Technical Details**: [{'title': '지식 종속 가능성 메트릭(KDA)', 'content': 'KDA는 MCQ의 대답 가능성을 평가하기 위해 학생 반응 데이터를 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 사용해 학생들의 문제 해결 행동을 모사합니다.'}, {'title': '대조 학습 및 반사실 증강', 'content': '기존의 반사실 증강 방법과 달리, 여러 반사실을 생성하여 집합적으로 의사 결정을 내림으로써 모델의 인과관계 파악을 더욱 견고하게 감독합니다. 이는 태스크 모델의 바이어스에 덜 민감하며 반사실 견고성, 도메인 간 일반화, 제한된 데이터로부터의 일반화를 포함한 다양한 차원에서 성능 향상을 보입니다.'}, {'title': 'GNN-MolKAN 및 GNN-MolKAN+', 'content': 'GNN-MolKAN과 GNN-MolKAN+는 Kolmogorov-Arnold Networks (KAN) 아키텍처를 통합하여 기존 GNN의 구조적 문제를 해결합니다. 더불어, 적극적 적응형 FastKAN(AdFastKAN)을 도입하여 모델의 안정성과 속도를 향상시킵니다.'}]

- **Performance Highlights**: [{'title': 'KDA 메트릭의 효과성', 'content': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 학습 자료 사용성 및 다양한 MCQ 품질 측정에 대한 예측력이 높게 나타났습니다.'}, {'title': 'NLP 모델의 견고성 강화', 'content': '새로운 접근 방식은 반사실 견고성, 도메인 간 일반화 및 제한된 데이터로부터의 일반화 향상 등의 다양한 차원에서 중요한 성능 향상을 달성했습니다.'}, {'title': 'GNN-MolKAN의 탁월한 성과', 'content': 'GNN-MolKAN과 GNN-MolKAN+는 더 적은 계산 시간과 파라미터로도 최신 상태의 성능을 능가하거나 일치했으며, 적은 샘플 학습 시나리오에서도 평균 6.97%의 성능 향상을 보였습니다.'}]



### IBB Traffic Graph Data: Benchmarking and Road Traffic Prediction Mod (https://arxiv.org/abs/2408.01016)
- **MCQ Generation**: [{"What's New": '교육적 가치를 고려하지 않는 기존의 MCQ 생성 평가 메트릭의 한계를 극복하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안했습니다.'}, {'Technical Details': 'KDA는 대상 사실에 대한 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. Human survey를 통해 KDA를 측정하고, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont를 제안합니다.'}, {'Performance Highlights': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가졌으며, 전문가가 라벨링한 다양한 MCQ 품질 평가 척도를 예측하는 데 강력한 예측력을 보였습니다.'}]

- **Robust NLP Models**: [{"What's New": '최근 딥러닝 모델이 NLP 태스크에서 높은 정확성을 보였지만 spurious patterns에 의존하는 문제점을 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용한 새로운 방법을 제안했습니다.'}, {'Technical Details': "기존 방법들이 spurious correlations에 영향을 받는 데 반해, 우리의 방법은 '여러 개의' counterfactual을 생성하고, 집합적 의사 결정을 통해 더 robust하게 단어들의 인과관계를 파악합니다."}, {'Performance Highlights': '우리의 접근 방식은 1) 반사실적 강건성(counterfactual robustness), 2) 도메인 간 일반화(cross-domain generalization), 3) 희소한 데이터의 일반화(scarce data generalization) 면에서 중요한 개선을 이뤘습니다.'}]

- **Traffic Congestion Prediction**: [{"What's New": '지리적 특성을 반영한 새로운 벤치마크 데이터셋 IBB 교통 그래프 데이터셋을 도입하고, 기존 모델들을 뛰어넘는 새로운 교통 예측 모델을 제안했습니다.'}, {'Technical Details': 'IBB 교통 그래프 데이터셋은 이스탄불의 2451개의 서로 다른 위치에서 수집된 센서 데이터를 포함하며, GLEE를 사용한 노드 임베딩과 ExtraTrees를 통한 교통 예측을 결합한 새로운 모델을 제안합니다.'}, {'Performance Highlights': '제안된 모델은 기존 모델들보다 평균 정확도가 4% 향상되었으며, 도로 네트워크 내 상호 연관된 관계를 나타내는 데 효과적임을 보여줍니다.'}]



### Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs (https://arxiv.org/abs/2408.01008)
Comments:
          LA-UR-24-28177

- **What's New**: 이번 뉴스레터에서는 자동 다중 선택 질문(MCQ) 생성의 새로운 평가 지표, NLP 모델의 대체 사실(counterfactual) 강화를 통한 강건성 향상, 그리고 대규모 언어 모델(LLM) 튜닝을 위한 파라미터 효율적인 방법론인 TT-LoRA를 소개합니다.

- **Technical Details**: {'MCQ Generation': '기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사성 메트릭은 MCQ의 교육적 가치를 제대로 평가하지 못합니다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안하였으며, 이는 목표 사실에 대한 학생의 답변 가능성을 측정합니다.', 'NLP Robustness': '최근의 깊은 모델들이 NLP 태스크에서 사람보다 높은 정확도를 보여주지만, 가짜 패턴(spurious patterns)에 의존하여 강건성이 떨어집니다. 이에 대처하기 위해 대체 사실(counterfactual)을 생성하고, 이를 이용해 모델의 인과 관계를 강력하게 평가하는 방법을 제안합니다.', 'LLM Parameter-Efficient Fine-Tuning': 'TT-LoRA라는 새로운 파라미터 효율적 튜닝 방법을 소개합니다. Low-Rank Approximation (LoRA)와 텐서 트레인(Tensor Train) 분해를 결합해 더 효과적인 모델 압축과 성능을 유지하며, 전체 모델 튜닝보다 적은 계산 자원으로 LLM을 효율적으로 튜닝할 수 있습니다.'}

- **Performance Highlights**: {'MCQ Evaluation Metrics': 'KDA_disc와 KDA_cont는 실제 강의실 세트에서 전문가가 라벨링한 사용성과 강한 상관관계를 보여주었습니다. n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 가집니다.', 'NLP Robustness': '다양한 차원에서 대체 사실 강건성, 도메인 간 일반화, 그리고 희소한 데이터로부터의 일반화 등의 측면에서 개선된 성능을 보였습니다.', 'TT-LoRA': '종합적인 평가 결과, TT-LoRA는 다양한 다운스트림 태스크와 다양한 스케일의 LLM에서 뛰어난 성능과 모델 압축을 동시에 달성함을 보여주었으며, 기존의 PEFT 방법들과 비교하여 더 나은 압축 성능을 보였습니다.'}



### FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation (https://arxiv.org/abs/2408.00998)
Comments:
          Accepted conference paper of ACM MM 2024

- **What's New**: 최근, 새로운 평가 메트릭 KDA (Knowledge Dependent Answerability)가 제안되었습니다. 이는 기존의 n-gram 기반 평가 메트릭 (예: BLEU, ROUGE, METEOR)과 달리 MCQ의 교육적 가치를 평가합니다. 또한, 텍스트 기반 이미지 생성 모델의 조작 가능성을 개선하는 FBSDiff이 제안되었습니다. 이는 DCT 주파수 대역 교체를 사용하여 이미지-이미지 번역을 더욱 제어할 수 있게 합니다.

- **Technical Details**: [{'MCQ Generation': 'KDA는 학생이 특정 사실에 대한 지식을 바탕으로 질문에 답변할 수 있는 능력을 측정합니다. 기존 평가 메트릭들은 단어 유사성에만 집중하였지만, KDA는 지식 기반의 평가를 도입합니다. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하였으며, 이를 통해 학생의 문제 해결 행동을 모방합니다.'}, {'Text-to-Image Translation': 'FBSDiff는 사전 학습된 대규모 텍스트-이미지(T2I) 확산 모델을 활용하여 이미지를 참고해 이미지 변환을 제어합니다. DCT 스펙트럼 공간의 주파수 대역을 교체하여 역 샘플링 과정 동안 다양한 가이드 요소를 동적으로 대체합니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'KDA_disc와 KDA_cont는 인간 설문 결과와 높은 상관성을 보였으며, 실제 교실 환경에서의 사용성을 높였습니다. 이와 더불어 n-gram 기반 유사성 메트릭과 결합하여 전문가가 라벨링한 여러 MCQ 품질 지표에 대해 강력한 예측력을 가졌습니다.'}, {'Text-to-Image Translation': 'FBSDiff는 기존의 고급 방법들과 비교했을 때, 시각적 품질, 다용도성, 제어 가능성 면에서 우수하다고 입증되었습니다. 이 방법은 모델 훈련, 모델 세부 조정 또는 온라인 최적화 과정 없이 고품질 텍스트 기반 이미지 번역이 가능합니다.'}]



### IncidentNet: Traffic Incident Detection, Localization and Severity Estimation with Sparse Sensing (https://arxiv.org/abs/2408.00996)
Comments:
          6 pages, 6 figures, 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)

- **What's New**: 이번 연구는 교사의 학습 평가 시간을 크게 줄일 수 있는 자동 생성 MCQ(객관식 문제)의 새로운 평가 메트릭 KDA(지식 종속 가능성)를 제안합니다. 이는 기존 BLEU, ROUGE, METEOR 메트릭이 데이터셋에 있는 골드 샘플과의 n-gram 유사도만을 평가하는 것과 달리, 생성된 문제의 교육적 가치를 평가할 수 있습니다. 두 번째 연구는 NLP(자연어 처리) 태스크에서 deep model이 spurious patterns에 의존하는 문제를 contrastive learning과 counterfactual augmentation을 통해 해결하는 방법을 제안합니다. 세 번째 연구는 IncidentNet이라는 새로운 딥러닝 모델을 통해 교통 사고를 빠르고 정확히 감지 및 위치 파악 및 중대성(seriousness)을 예측할 수 있는 방법을 제안합니다.

- **Technical Details**: MCQ 평가에서는 KDA가 학생의 대상 사실에 대한 지식을 기반으로 문제의 대답 가능성(answerability)을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행위를 모방해 이를 자동 평가합니다. NLP 태스크에서 제안된 방법은 여러 개의 counterfactual을 생성하고, 이를 바탕으로 단어들의 인과관계를 더 robust하게 파악합니다. IncidentNet은 소형 교통 데이터(미시적 데이터)를 사용하여 교통 사고를 감지하며, 데이터셋 불충분 문제를 해결하기 위한 합성 데이터셋을 생성하는 방법론을 제안합니다.

- **Performance Highlights**: MCQ 평가에서는 KDA_disc와 KDA_cont가 KDA 및 실제 교실 세트 사용성과 강한 상관관계를 가짐을 보여주었습니다. NLP 모델은 counterfactual robustness, cross-domain generalization, generalization from scarce data에서 유의미한 성능 향상을 보였습니다. IncidentNet은 98%의 교통 사고 감지율과 평균 197초 이내의 감지 시간, 7% 미만의 오경보율을 달성했습니다.



### ArchCode: Incorporating Software Requirements in Code Generation with Large Language Models (https://arxiv.org/abs/2408.00994)
Comments:
          Accepted by ACL 2024 main conference

- **Multiple Choice Question (MCQ) Generation**: [{"What's New": '기존 MCQ 생성 평가 메트릭의 한계를 극복하기 위해, 자동 평가 메트릭 지식 종속 가능성(KDA)을 제안했습니다. 이는 MCQ의 대답 가능성 및 학생의 관련 지식을 평가하는 능력을 측정합니다.'}, {'Technical Details': 'KDA는 학생의 응답을 기반으로 측정되며, 이를 근사하기 위해 사전 학습된 언어모델을 활용한 KDA_disc와 KDA_cont 메트릭 두 가지를 제안했습니다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실에서의 사용성 및 전문가가 라벨링한 다양한 MCQ 품질 측정에서 강한 상관관계를 보였습니다.'}]

- **NLP Task Robustness**: [{"What's New": '국지적 패턴(spurious patterns)에 대한 의존성 문제를 해결하기 위해 대조 학습(contrastive learning)과 반사실적 데이터 증강(counterfactual augmentation)을 이용하여 새로운 방법을 제안했습니다.'}, {'Technical Details': '기존 방법들과 달리 여러 개의 반사실적 데이터 셋트를 생성하고, 집합적 의사 결정(collective decisions)을 통해 더 확실하게 단어들의 인과관계를 감독합니다.'}, {'Performance Highlights': '제안된 방법은 다양한 측면에서 현저한 성능 향상을 보여주었으며, 특히 반사실적 강건성(counterfactual robustness), 교차 도메인 일반화(cross-domain generalization), 희소 데이터에서의 일반화 부분에서 두드러졌습니다.'}]

- **Code Generation Enhancement**: [{"What's New": '대형 언어모델(LLMs)의 코드 생성 기능을 텍스트로부터 소프트웨어 요구 사항을 포괄적으로 관리할 수 있도록 확장하는 ARCHCODE 프레임워크를 도입했습니다.'}, {'Technical Details': 'ARCHCODE는 In-Context Learning(ICL)을 활용해 텍스트 설명에서 요구 사항을 체계적으로 추출하고, 표현되지 않은 요구 사항까지도 추론하여 코드와 테스트 케이스를 생성합니다.'}, {'Performance Highlights': 'ARCHCODE는 GPT-4의 Pass@1 점수를 능가하며, CodeContests에서 새로운 최첨단 성과를 달성했습니다. 또한, HumanEval-NFR 벤치마크를 도입하여 비기능적 요구 사항(NFRs)까지 효과적으로 만족시켰음을 입증했습니다.'}]



### PERSOMA: PERsonalized SOft ProMpt Adapter Architecture for Personalized Language Prompting (https://arxiv.org/abs/2408.00960)
- **New Methods for Automated MCQ Evaluation**: [{"What's New": '기존의 자동 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ가 골드 샘플과 얼마나 유사한지를 n-gram 기반으로 평가하였으나, 교육적 가치는 고려하지 않았다. 이를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하였다.'}, {'Technical Details': 'KDA는 학생의 대상 사실에 대한 지식을 평가할 수 있는 MCQ의 대답 가능성(answerability)을 측정한다. 구체적으로, 인간 설문 조사를 통해 KDA를 측정하는 방법을 제시하고, 학생들의 문제 해결 행동을 모방하기 위해 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont라는 자동 평가 메트릭을 제안한다.'}, {'Performance Highlights': '인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었다. 또한, n-gram 기반의 유사성 메트릭과 결합될 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 측정에 대해 강한 예측력을 나타낸다.'}]

- **Enhancing NLP Model Robustness with Counterfactuals**: [{"What's New": '최근 NLP 태스크에서 deep model들이 사람보다 나은 정확성을 보여주지만, spurious pattern에 의존해 robustness가 제한된다는 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 활용해 robustness를 향상시키는 방법을 제안한다.'}, {'Technical Details': '기존 방법들은 사람이 counterfactual을 만들거나 모델이 데이터셋에서 유사한 counterfactual을 찾는 방식을 사용했으나, 우리는 여러 개의 counterfactual을 생성하고 집합적 의사결정을 통해 단어들의 인과관계를 robust하게 파악하는 방법을 제안한다.'}, {'Performance Highlights': '우리의 방법은 attribution-based synthesis의 task model bias에 덜 민감하여 1) counterfactual robustness, 2) cross-domain 일반화, 3) scarce data로부터의 일반화에서 현저한 개선을 이룬다.'}]

- **Personalized User Interaction with Large Language Models**: [{"What's New": '사용자 맞춤형 자연어 시스템 구축을 위해 PERSOMA, Personalized Soft Prompt Adapter 아키텍처를 제안한다. 이는 사용자 히스토리를 효율적으로 캡처하기 위한 혁신적인 접근법을 제공한다.'}, {'Technical Details': 'PERSOMA는 사용자의 히스토리를 자유형 텍스트로 재샘플링하고 이를 expressive soft prompt embeddings로 압축하여 사용자 특화된 soft prompt 어댑터를 구축한다. 이를 통해 LLM의 이해력을 손상시키지 않으면서도 사용자 히스토리를 이용해 출력 결과를 사용자에게 맞출 수 있다.'}, {'Performance Highlights': 'PaLM 2 모델을 이용한 실험에서 PERSOMA는 MovieLens 데이터셋에서 기존 embedding-based 기술보다 0.18의 F1 점수 개선을 이루었으며, 전체 finetuned text prompting baseline과 유사한 성능을 나타내었다.'}]



### CIResDiff: A Clinically-Informed Residual Diffusion Model for Predicting Idiopathic Pulmonary Fibrosis Progression (https://arxiv.org/abs/2408.00938)
- **What's New**: 이 논문은 Multiple Choice Questions (MCQ)의 자동 생성을 평가하는 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 학생이 특정 사실을 알고 있을 때 해당 MCQ에 답할 수 있는 능력을 측정합니다. 두 가지 자동 평가 메트릭으로 KDA_disc와 KDA_cont를 제안하며, 이는 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방합니다.

- **Technical Details**: KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용해 학생들이 문제를 풀 때 보여주는 행동을 모방해 KDA를 근사화합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 확인하였습니다. 또한, 이 두 메트릭은 n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 라벨링한 MCQ 품질 측정치를 강하게 예측할 수 있는 능력을 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 높은 상관관계를 보이며, 전문가가 라벨링한 MCQ 품질 측정에 강한 예측력을 보였습니다. 이는 기존의 BLEU, ROUGE, METEOR 지표들이 놓치는 교육적 가치를 평가하는 데에 큰 도움을 줄 수 있습니다.



### Enabling High Data Throughput Reinforcement Learning on GPUs: A Domain Agnostic Framework for Data-Driven Scientific Research (https://arxiv.org/abs/2408.00930)
- **What's New**: 최근 논문에서는 여러 교육 및 과학 연구 분야에서 적용할 수 있는 새로운 평가 메트릭과 대규모 강화 학습 프레임워크를 소개합니다. 특히, 자동 MCQ 생성을 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)와 강화 학습의 시스템 병목 현상을 해결하기 위한 WarpSci 프레임워크가 주목받고 있습니다.

- **Technical Details**: {'MCQ Evaluation Metric (Knowledge Dependent Answerability, KDA)': '기존 BLEU, ROUGE, METEOR 메트릭은 교육적 가치를 고려하지 않고, 데이터셋의 골드 샘플과의 n-gram 유사성만 평가합니다. KDA는 MCQ의 대답 가능성 (answerability)을 기준으로 학습자의 지식 평가 능력을 측정합니다. 인간 평가 기반 KDA와 유사한 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안하여 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', 'WarpSci Framework': 'WarpSci는 과학 연구에서의 강화 학습(RL)을 위한 고속 데이터 처리 아키텍처를 제공합니다. 이 프레임워크는 CPU와 GPU 간의 데이터 전송을 제거하고, 수천 개의 시뮬레이션을 하나 또는 여러 개의 GPU 상에서 동시에 실행할 수 있습니다. 이를 통해 고차원 관찰 또는 액션 공간을 가진 복잡한 환경에서도 효율적으로 RL을 수행할 수 있습니다.'}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 실제 강의 환경에서의 사용성과 전문가 평가에 대한 강한 상관관계를 보였습니다. 이러한 메트릭을 n-gram 기반 유사성 메트릭과 결합할 때, 다양한 전문가 라벨 MCQ 품질 지표에 대해 높은 예측력을 가집니다.', 'WarpSci': 'WarpSci는 단일 Nvidia A100 GPU 상에서 실행되며, 다른 분산 시스템 대비 적어도 10-100배 높은 처리량을 달성했습니다. 예를 들어, 10,000개의 동시 Cartpole 환경에서 초당 860만 스텝을 처리하며, 경제 시뮬레이션에서는 초당 12만 스텝, 촉매 반응 모델링에서는 초당 95만 스텝을 기록했습니다. 이러한 환경에서 워프드라이브는 수천 개의 환경 또는 에이전트에 거의 완벽한 병렬 처리를 제공합니다.'}



### WHITE PAPER: A Brief Exploration of Data Exfiltration using GCG Suffixes (https://arxiv.org/abs/2408.00925)
Comments:
          8 pages, 8 figures. Conducted as part of employment at Microsoft Corporation

- **What's New**: 이 논문은 자동 생성된 객관식 질문(MCQ)의 교육적 가치를 평가하는데 중점을 둔 새로운 메트릭, 지식 종속 가능성(KDA)를 제안합니다. 기존의 BLEU, ROUGE, METEOR 같은 메트릭은 주로 n-그램 기반의 유사성에 집중하여 학습 평가의 진정한 효과를 간과하고 있습니다.

- **Technical Details**: KDA는 학생 설문조사를 기반으로 MCQ의 응답 가능성을 직접 측정합니다. 추가로, 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방하여 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안합니다. 이를 통해 자동 평가 메트릭과 실제 인간 평가 간의 상관관계를 검증하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의에서의 사용성 측면에서도 높은 상관관계를 보였습니다. 또한, 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강력한 예측력을 입증하였습니다.



### Reclaiming Residual Knowledge: A Novel Paradigm to Low-Bit Quantization (https://arxiv.org/abs/2408.00923)
Comments:
          Accepted by The 35th British Machine Vision Conference (BMVC 2024)

- **What's New**: 최근 논문에서 다양한 새로운 기술들이 소개되었습니다. 첫 번째 논문은 Multiple Choice Questions (MCQ)의 자동 생성을 돕기 위해 Knowledge Dependent Answerability (KDA)라는 새로운 평가 메트릭을 제안합니다. 두 번째 논문은 NLP 태스크에서 학생의 인과 관계를 더 잘 이해할 수 있도록 대비 학습과 counterfactual augmentation을 사용한 접근 방법을 다룹니다. 마지막 논문은 ConvNets의 저비트 양자화(quantization)를 개선하기 위해 Optimal Quantization Residual Convolutional Operator Low-Rank Adaptation (CoRa)이라는 프레임워크를 제안합니다.

- **Technical Details**: [{'paper': '첫 번째 논문', 'description': 'MCQ 자동 생성의 문제점을 해결하기 위해 KDA라는 새로운 평가 메트릭을 도입했습니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭도 제안되어, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다.'}, {'paper': '두 번째 논문', 'description': 'spurious pattern에 의존하는 deep model의 robustness 문제를 해결하기 위해 여러 counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다.'}, {'paper': '세 번째 논문', 'description': 'CoRa라는 프레임워크는 ConvNets의 저비트 양자화를 위해 optimal quantization residual knowledge를 저랭크 어댑터(low-rank adapters)를 통해 복구하는 방법을 제안합니다. 이 프레임워크를 통해 더 적은 반복으로 성능 저하 없이 모델을 양자화할 수 있습니다.'}]

- **Performance Highlights**: [{'paper': '첫 번째 논문', 'description': 'Human studies 결과, KDA_disc와 KDA_cont는 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있으며, n-gram 기반의 유사성 메트릭과 결합할 경우 전문가가 라벨링한 MCQ 품질 평가에서 강한 예측력을 보였습니다.'}, {'paper': '두 번째 논문', 'description': '제안된 방법은 counterfactual robustness, cross-domain generalization, 희소한 데이터로부터의 일반화 등 다양한 차원에서 기존 모델들에 비해 높은 성능을 보였습니다.'}, {'paper': '세 번째 논문', 'description': 'CoRa는 ImageNet에서 사전 훈련된 다수의 ConvNet에 대해 평가되었으며, 4비트 및 3비트 양자화에서 state-of-the-art 양자화 인식 훈련과 사후 훈련 양자화 기준에 대비하여 비교 가능한 성능을 달성하였습니다. 단지 1600장의 작은 이미지 세트로 250회 미만의 반복을 통해 최적화를 달성합니다.'}]



### Parkinson's Disease Detection from Resting State EEG using Multi-Head Graph Structure Learning with Gradient Weighted Graph Attention Explanations (https://arxiv.org/abs/2408.00906)
Comments:
          Accepted at MLCN 2024

- **What's New**: 최근 자동 다중 선택 질문(MCQ) 생성의 평가 메트릭이 교육적 가치를 반영하지 않는다는 문제를 해결하기 위해 'Knowledge Dependent Answerability (KDA)'라는 새로운 평가 메트릭을 제안했습니다. 기존의 BLEU, ROUGE, METEOR는 n-gram 기반의 유사성만 측정하여 MCQ가 학생의 지식을 평가하는 능력을 고려하지 않았습니다. KDA는 MCQ의 대답 가능성(answerability)을 기반으로 학생의 지식을 평가합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 평가되며, 이를 자동화하기 위해 사전학습된 언어 모델을 이용한 KDA_disc와 KDA_cont를 제안합니다. 이 모델들은 학생의 문제 해결 행동을 모방하여 MCQ의 품질을 예측합니다. 인간 평가를 통해 KDA_disc와 KDA_cont의 유효성을 확인했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 n-gram 기반의 유사성 메트릭과 결합될 때 전문가가 라벨링한 MCQ 품질 측정 능력에서 강한 예측력을 나타냈습니다.



### Expressive MIDI-format Piano Performance Generation (https://arxiv.org/abs/2408.00900)
Comments:
          4 pages, 2 figures

- **What's New**: 이 논문에서는 교육 평가에서 자동으로 다지선다형 질문(MCQ)을 생성하는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. KDA는 MCQ가 학생의 지식 기반으로 답변할 수 있는 능력을 측정합니다. 이는 기존 n-gram 기반 메트릭 (BLEU, ROUGE, METEOR)의 한계를 보완합니다. 또 다른 연구에서는 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 NLP 모델의 강인성을 개선하는 방법을 제안했습니다. 마지막으로, 미디(MIDI) 형식으로 표현력 있는 피아노 연주를 생성하는 생성 신경망을 소개했습니다.

- **Technical Details**: MCQ 생성 논문에서는 KDA를 측정하기 위해 인간 설문조사 기반 방법과 KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 제안했습니다. NLP 모델의 강인성 연구에서는 대조 학습과 집합적 의사 결정(collective decisions)을 통해 반사실적 증강을 수행하는 방법을 사용했습니다. MIDI 피아노 연주 생성에서는 LSTM(long-short term memory)을 이용한 다중 입력과 출력의 순차 모델을 사용했으며, 서스테인 페달(sustain pedal)과 같은 제어 이벤트도 고려했습니다.

- **Performance Highlights**: MCQ 생성에서는 KDA_disc와 KDA_cont가 실제 교실 환경에서의 사용성과 강한 상관관계를 보였습니다. NLP 연구에서는 반사실적 강인성, 크로스 도메인 일반화, 소규모 데이터에서의 일반화에서 상당한 개선을 달성했습니다. MIDI 피아노 생성 모델은 아직 충분히 학습되지 않았지만 표현력 있는 피아노 연주를 생성할 수 있는 강력한 생성 능력을 보였습니다.



### On the Relationship Between Monotone and Squared Probabilistic Circuits (https://arxiv.org/abs/2408.00876)
Comments:
          7th Workshop on Tractable Probabilistic Modeling

- **What's New**: 최근 deep model들이 NLP 태스크에서 사람보다 나은 정확성을 보였으나, spurious pattern에 의존하는 문제로 인해 robustness가 제한된다고 보고되고 있습니다. 이에 대응하기 위해 대비 학습(contrastive learning) 및 counterfactual augmentation 방법론을 사용하여 모델의 강건성을 향상시키는 방법을 제안합니다. 새로운 평가 메트릭인 지식 종속 가능성(KDA)도 자동 MCQ 생성의 교육적 가치를 평가하는 새로운 접근법으로 등장했습니다. 또한 확률적 회로(probabilistic circuits)라는 새로운 종류의 모델인 InceptionPCs가 제안되었습니다.

- **Technical Details**: {'First Paper': {'Evaluation Metric': 'BLEU, ROUGE, METEOR과 같은 기존 메트릭은 MCQ의 교육적 가치를 평가하지 못합니다. 이를 보완하기 위해 지식 종속 가능성(KDA)이 제안되었으며, 이는 학생의 지식을 평가하는 능력을 측정하는 메트릭입니다.', 'Methods': 'Human survey를 기반으로 KDA를 측정하고 이를 예측하기 위해 다양한 사전 학습된 언어 모델을 활용합니다.'}, 'Second Paper': {'Robust Models': '스퓨리어스 패턴에 의존하는 문제를 해결하기 위해 대비 학습(contrastive learning) 및 반사실적인 증강(counterfactual augmentation)이 사용되었습니다.', 'Methods': '모델이 여러 개의 counterfactual을 생성하고 이들의 집합적인 의사 결정(collective decisions)을 통해 인과관계를 보다 robust하게 파악합니다.'}, 'Third Paper': {'Probabilistic Circuits': '확률적 회로는 무게가 있는 합 및 곱 함수의 계산 그래프를 대표하는 모델입니다. InceptionPCs는 이들 회로를 일반화시키는 새로운 접근법을 통해 더 효율적인 모델링을 가능하게 합니다.', 'Benefits': 'InceptionPCs는 기존의 단조 회로(monotone circuits)와 제곱 회로(squared circuits)의 장점을 모두 포함하며, 이미지 데이터셋에서 성능을 향상시킬 수 있습니다.'}}

- **Performance Highlights**: {'First Paper': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트에서 사용성과 강한 상관관계를 보였습니다. n-gram 기반의 유사성 메트릭과 결합했을 때, 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 보였습니다.', 'Second Paper': '집합적 의사결정 방식을 통해 attribution-based synthesis의 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data generalization의 다양한 차원에서 유의미한 개선을 보였습니다.', 'Third Paper': '이미지 데이터셋 (MNIST, FashionMNIST)에서 InceptionPCs는 기존의 단조 및 제곱 회로를 능가하는 성능을 보였습니다.'}



### UniMoT: Unified Molecule-Text Language Model with Discrete Token Representation (https://arxiv.org/abs/2408.00863)
- **What's New**: 최근 연구들은 MCQ(선다형 시험 문제) 자동 생성을 위한 평가 기준이 교육적 가치 대신 단어 유사성에 집중하고 있다는 문제를 제기하며, 이를 해결하기 위한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다. 또한, NLP 태스크에서 deep model의 robustness를 향상시키기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 탐구했습니다. 그리고 다양한 태스크에서 뛰어난 성능을 보이는 LLMs를 화학 분야로 확장하기 위한 UniMoT라는 새로운 모델도 제안되었습니다.

- **Technical Details**: {'MCQ 평가 기준': '기존의 BLEU, ROUGE, METEOR 메트릭이 데이터셋 내의 골드 샘플과의 n-gram 유사성에만 집중하기 때문에 교육적 가치를 평가하지 못합니다. 이를 해결하기 위해 KDA(Knowledge Dependent Answerability)라는 새로운 자동 평가 메트릭을 제안하였으며, 이는 학생이 해당 지식을 알고 있을 때 MCQ의 답변 가능성을 측정합니다.', 'Robustness 향상': '대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하여 모델의 robustness를 향상시키는 접근법을 제안했습니다. 기존 방법들과 달리 여러 개의 counterfactual을 생성하고 집합적인 의사결정을 통해 각 용어의 인과관계를 보다 robust하게 감독합니다.', 'UniMoT': 'UniMoT는 분자와 텍스트 모달리티를 통합한 LLM으로, Molecule Tokens과 Text Tokens을 동일하게 취급하는 Tokenizer 기반 아키텍처를 사용합니다. Vector Quantization를 통해 분자를 일련의 토큰으로 변환하며, 이에 따라 분자를 새로운 언어처럼 해석하고 생성할 수 있습니다.'}

- **Performance Highlights**: {'MCQ 평가 기준': 'KDA_disc와 KDA_cont는 실제 교실 환경에서 사용성과 강한 상관관계를 가지며, 기존 n-gram 기반 유사성 메트릭과 결합될 때 전문가가 라벨링한 MCQ 품질 측정치에 대해 높은 예측력을 보입니다.', 'Robustness 향상': '제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 희소 데이터에서의 generalization 등의 다양한 차원에서 큰 향상을 보여주었습니다.', 'UniMoT': 'UniMoT는 분자 이해 및 생성 작업에서 최첨단 성능을 달성하였으며, 다중 모달리티 이해와 생성을 모두 수행할 수 있는 능력을 입증했습니다.'}



### Calibrating Bayesian Generative Machine Learning for Bayesiamplification (https://arxiv.org/abs/2408.00838)
Comments:
          15 pages, 6 figures

- **What's New**: 이 논문은 기존의 MCQ (Multiple Choice Question) 생성 평가 메트릭들이 놓치고 있는 교육적 가치를 포착하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 또한, 최근의 딥 모델들이 NLP 태스크에서 높은 정확성을 보였으나, spurious pattern이라는 문제에 직면하고 있음을 지적합니다. 마지막으로, 입자 물리학에서 Bayesian 기계 학습 모델을 사용해 발생하는 불확실성을 정량화하는 새로운 접근 방식이 도입되었습니다.

- **Technical Details**: MCQ 생성의 경우, 새로운 KDA를 사용하여 MCQ의 대답 가능성(answerability)을 측정합니다. 각 MCQ가 해당 사실에 대한 학생의 지식을 평가하는 능력을 담보하려고 합니다. 또한, contrastive learning과 counterfactual augmentation을 활용하여 NLP 모델의 robustness를 향상시키려는 노력이 이뤄졌습니다. 입자 물리학에서 Bayesian Continuous Normalizing Flows를 적용해 데이터 분포의 불확실성을 정밀하게 예측합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 인간 평가에서 높은 상관성을 보였으며, 실제 교육 환경에서의 사용성까지 입증되었습니다. 다양한 차원에서 robustness, cross-domain 일반화, 그리고 희소 데이터에 대한 일반화에서 중요한 개선 사항을 보였습니다. Bayesian 기법을 사용한 생성 모델은 기존 생성 모델보다 더 나은 예측 성능과 데이터 증폭 현상을 확인했습니다.



### Adaptive traffic signal safety and efficiency improvement by multi objective deep reinforcement learning approach (https://arxiv.org/abs/2408.00814)
- **What's New**: 이번 연구에서는 다목적 심층 강화 학습(Multi-objective Deep Reinforcement Learning) 기법을 이용한 적응형 교통 신호 제어(Adaptive Traffic Signal Control, ATSC) 방법을 제안합니다. 이는 기존의 교통 효율성 중심의 ATSC 방법을 넘어서, 안전성, 효율성, 그리고 탈탄소화 목표들을 동시에 해결하고자 합니다.

- **Technical Details**: 제안된 접근 방식은 듀얼링 더블 딥 Q 네트워크(Dueling Double Deep Q Network, D3QN) 프레임워크를 통합한 DRL 기반 ATSC 알고리즘을 사용합니다. 이 알고리즘은 중국 창사 시뮬레이션 교차로에서 성능이 평가되었습니다.

- **Performance Highlights**: 제안된 ATSC 알고리즘은 전통적인 ATSC 방법과 효율성 최적화만을 집중하는 ATSC 알고리즘과 비교할 때, 교통 충돌을 16% 이상 감소시키고, 탄소 배출을 4% 줄이는 등의 성과를 보였습니다. 교통 효율성 측면에서도, 전통적인 ATSC에 비해 대기 시간을 18% 줄이는 성과를 보였으나, D3QN 기반 DRL ATSC 알고리즘과 비교할 때는 약간(0.64%) 증가하는 트레이드오프를 보였습니다. 특히, 높은 교통 수요 시나리오에서도 모든 목표에서 우수한 성과를 보였습니다.



### ChipExpert: The Open-Source Integrated-Circuit-Design-Specific Large Language Mod (https://arxiv.org/abs/2408.00804)
- **What's New**: 최근 여러 분야에서 큰 성공을 거둔 대형 언어 모델(LLMs)이 통합 회로(IC) 설계 분야의 요구를 충족시키지 못하는 경우가 많았습니다. 이를 해결하기 위해, IC 설계 분야에 특화된 최초의 오픈소스 교육용 LLM인 ChipExpert가 도입되었습니다.

- **Technical Details**: ChipExpert는 Llama-3 8B와 같은 최신 오픈소스 기반 모델을 사용하여 훈련되었습니다. 데이터 준비, 지속적 사전 훈련, 지시 기반 감독 기법 세분화(fine-tuning), 선호도 맞춤(alignment), 평가 등 여러 주요 단계를 거쳐 개발되었습니다. 또한, ChipICD-Bench라는 첫 IC 설계 벤치마크를 도입하여 여러 IC 설계 하위 도메인에서 LLM의 역량을 평가했습니다.

- **Performance Highlights**: ChipExpert는 IC 설계 지식 질문-응답 과제에서 높은 수준의 전문성을 입증했으며, 사용자의 질의에 대해 전문적으로 응답하는 능력을 습득했습니다. 또한, 헛소리(hallucinations)를 줄이기 위해 IC 설계 지식 기반에 기반한 Retrieval-Augmented Generation(RAG) 시스템을 개발했습니다.



### A Comprehensive Survey on Root Cause Analysis in (Micro) Services: Methodologies, Challenges, and Trends (https://arxiv.org/abs/2408.00803)
- **What's New**: 자동 다지선택 질문(MCQ) 생성을 위해 새로운 평가 메트릭인 지식 종속 대답 가능성(KDA)을 제안했습니다. 이는 기존 평가 메트릭이 데이터셋의 골드 샘플과의 n-그램 유사성만을 고려한다는 한계를 넘어서서, 학생들의 대상 사실에 대한 지식을 평가하는 능력을 측정합니다.

- **Technical Details**: KDA는 사람 설문조사 결과 기반으로 측정되며, 이를 모방한 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안했습니다. 이 메트릭들은 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. Human evaluation을 통해 이 자동 메트릭이 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 확인했습니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 교육 전문가들이 라벨링한 다양한 MCQ 품질 측정과 높은 예측력을 보여주었으며, 기존의 n-그램 기반 유사성 메트릭과 결합할 때 더욱 강력한 예측 성능을 발휘했습니다.



### Leveraging LLM Reasoning Enhances Personalized Recommender Systems (https://arxiv.org/abs/2408.00802)
Comments:
          To be published at ACL 2024

- **What’s New**: 새로운 자동 평가 메트릭 KDA (Knowledge Dependent Answerability)를 제안하여 MCQ (Multiple Choice Questions)의 교육적 가치를 평가합니다. 또한, 최근 NLP 모델의 취약성과 관련하여 대조 학습과 반사실 증강을 활용한 방법을 소개합니다. 마지막으로, RecSys (추천 시스템)에서 LLM (Large Language Models)의 추론력을 활용하여 개인화 추천 작업 성능을 개선하는 연구가 포함됩니다.

- **Technical Details**: {'MCQ Generation': {'KDA': '학생들이 특정 사실을 가지고 MCQ에 답할 수 있는가를 측정하는 새로운 평가 메트릭입니다.', 'KDA_disc와 KDA_cont': '특정 지식을 바탕으로 KDA를 자동적으로 평가하는 두 가지 메트릭으로, 사전 학습된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.'}, 'Robustness in NLP Models': {'대조 학습과 반사실 증강': '스퍼리어스 패턴 (spurious patterns)에 의존할 수 있는 NLP 모델의 취약성을 줄이기 위해 대조 학습 및 집합적 의사 결정 방식을 사용한 새로운 접근법을 제안합니다.'}, 'LLM in RecSys': {'Rec-SAVER': '추천 시스템에서 LLM의 추론 응답의 품질을 자동으로 평가하는 프레임워크로, 큐레이션된 골드 레퍼런스나 인간 평가자가 필요 없습니다.'}}

- **Performance Highlights**: {'MCQ Generation': 'KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 세트의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합 시 예측 정확도 상승을 보였습니다.', 'Robustness in NLP Tasks': '제안된 접근법은 반증 가능성, 교차 도메인 일반화 및 희소 데이터에서의 일반화 측면에서 크게 개선된 성능을 보여주었습니다.', 'LLM in RecSys': '신규 Rec-SAVER 프레임워크는 인간 평가와 일치하는 추론 응답의 일관성 및 신뢰성을 평가하며, zero-shot과 fine-tuning 설정 모두에서 추천 작업 성능을 향상시켰습니다.'}



### Chatbot-Based Ontology Interaction Using Large Language Models and Domain-Specific Standards (https://arxiv.org/abs/2408.00800)
- **What's New**: 최신 arXiv 논문들에서 소개된 흥미로운 연구 내용을 소개합니다. 첫 번째 논문은 MCQ (Multiple Choice Questions) 자동 생성의 평가 메트릭을 개선하기 위한 새로운 접근을 제시합니다. 두 번째 논문은 NLP 태스크에서 모델의 robustness를 강화하기 위해 contrastive learning과 counterfactual augmentation을 활용한 방법을 설명합니다. 세 번째 논문은 LLM (Large Language Models)과 챗봇 인터페이스를 사용하여 SPARQL 쿼리 생성을 향상시키는 개념을 소개합니다.

- **Technical Details**: [{'title': 'MCQ 평가를 위한 Novel Metric, KDA', 'main': '기존 BLEU, ROUGE, METEOR 같은 n-gram 기반의 평가 메트릭은 교육적인 가치를 평가하지 못합니다. 이번 연구에서는 Knowledge Dependent Answerability (KDA)라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 answerability를 측정하고, 학생이 목표 지식에 대한 이해도를 평가하는 능력을 향상시킵니다.'}, {'title': 'Robustness를 위한 Contrastive Learning과 Counterfactual Augmentation', 'main': '최근 NLP 모델들이 높은 정확성을 보이지만, spurious pattern 의존으로 인해 robustness가 부족한 문제를 인식했습니다. 본 연구는 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 robust한 인과관계를 파악하는 방법을 제안합니다.'}, {'title': 'SPARQL 쿼리 생성을 위한 LLM과 챗봇', 'main': '산업적 응용을 위한 SPARQL 쿼리의 생성과 정확성을 위해 LLM과 챗봇 인터페이스를 도입한 개념을 제안합니다. 이 접근은 사용자 쿼리를 자연어로 입력받아 정확한 SPARQL 쿼리로 변환하고, 이를 통해 오탐지나 정보 왜곡을 방지합니다.'}]

- **Performance Highlights**: [{'title': 'KDA_disc와 KDA_cont의 결과', 'main': 'KDA_disc와 KDA_cont는 실제 강의실 세팅에서 효용성이 뛰어나며 전문가 레이블링 기준으로도 높은 상관관계를 보였습니다. n-gram 기반의 평가 메트릭과 결합 시, 다양한 MCQ 품질 측정에서 강한 예측력을 가집니다.'}, {'title': 'Robustness와 Generalization 성능 향상', 'main': '대조 학습과 반사실적 증강을 통해 모델의 counterfactual robustness, cross-domain generalization, 그리고 드문 데이터에서의 generalization 성능이 크게 향상되었습니다.'}, {'title': 'SPARQL 쿼리의 정확성', 'main': 'LLM을 활용한 SPARQL 쿼리의 실험적 결과, 정확성이 크게 향상되었습니다. 특히, 사용자 친화적인 인터페이스와 결합하여 오탐지 없이 정확한 결과를 제공함으로써 산업적 활용 가능성을 높였습니다.'}]



### Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Bas (https://arxiv.org/abs/2408.00798)
- **Multiple Choice Questions (MCQ)**: {"What's New": '기존의 BLEU, ROUGE, METEOR 평가 메트릭들이 MCQ 생성의 교육적 가치를 충분히 평가하지 못한다는 문제를 해결하기 위해, 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안했습니다.', 'Technical Details': 'KDA는 학생들이 대상 사실에 대한 지식을 바탕으로 MCQ에 대답할 수 있는 능력을 측정합니다. 자동 평가를 위해, KDA_disc와 KDA_cont라는 두 가지 메트릭을 제안하여 사전 학습된 언어 모델을 사용해 학생들의 문제 해결 행동을 모방합니다.', 'Performance Highlights': 'Human study를 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음이 확인되었습니다. n-gram 기반 유사성 메트릭과 결합했을 때 전문가가 레이블링 한 다양한 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.'}

- **Robust Deep Models for NLP**: {"What's New": 'NLP 태스크에서 deep model의 robustness를 향상시키기 위해, 대조 학습(contrastive learning)과 반사적 증강(counterfactual augmentation)을 활용하는 방법을 제안합니다.', 'Technical Details': '기존 방법들이 spurious correlations에 영향을 받는 반면, 우리의 접근법은 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decision)을 통해 각 용어의 인과관계를 robust하게 감독합니다.', 'Performance Highlights': '우리의 접근법은 attribution-based synthesis에 의한 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화 등 다양한 측면에서 상당한 향상을 이룬 것을 empiral 결과를 통해 확인했습니다.'}

- **Golden-Retriever**: {"What's New": '대규모 산업 지식 베이스를 효율적으로 탐색하기 위해 설계된 Golden-Retriever를 소개합니다. 이 방식은 domain-specific jargon과 문맥 해석 문제를 해결합니다.', 'Technical Details': 'Golden-Retriever는 문서 검색 전에 질문을 증강하는 reflection-based question augmentation 단계를 도입하여 용어를 식별하고 문맥에 따라 의미를 명확히 합니다. 이 접근법은 RAG(Retrieval Augmented Generation) 프레임워크가 가장 관련성이 높은 문서를 검색하도록 보장합니다.', 'Performance Highlights': '세 가지 오픈 소스 LLM을 사용한 평가에서 Golden-Retriever는 domain-specific question-answer dataset에서 기존 방법보다 뛰어난 성능을 보였습니다. 이로써 산업 지식 베이스를 효율적으로 통합하고 쿼리하는 강력한 솔루션을 제공합니다.'}



### CCSRP: Robust Pruning of Spiking Neural Networks through Cooperative Coevolution (https://arxiv.org/abs/2408.00794)
- **What's New**: 최근 자동 MCQ 생성(electronic quiz)을 평가하는 새로운 메트릭 'Knowledge Dependent Answerability (KDA)'와 SNN 프루닝(pruning)을 위한 혁신적인 'CCSRP' 방법이 소개되었습니다. KDA는 MCQ의 교육적 가치를 높이기 위해 설계되었으며, CCSRP는 자원 제한 환경에서의 SNN 성능을 상호 협력 진화 방법으로 증가시킵니다.

- **Technical Details**: KDA는 학생들이 특정 지식을 바탕으로 MCQ에 답할 수 있는지를 평가합니다. 두 가지 자동화 평가 메트릭인 KDA_disc와 KDA_cont를 사용하여 사전 훈련된 언어 모델을 통해 학생들의 문제 해결 행동을 모방합니다. 한편, CCSRP는 협력적 공동 발전(evolutionary algorithms)을 통해 SNN 필터를 독립적으로 프루닝하여 정확성, 견고성, 컴팩트성을 동시에 높이는 삼중 목표 최적화 문제를 해결합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 강의실 세트에서 높은 상관관계를 보여주었으며, 다양한 품질 측정 기준에 대한 예측력이 우수함을 입증했습니다. CCSRP는 CIFAR-10와 SVHN 데이터셋에서 최신 방법론과 견줄만한 성능을 나타냈습니다.



### A Scalable and Generalized Deep Learning Framework for Anomaly Detection in Surveillance Videos (https://arxiv.org/abs/2408.00792)
- **What's New**: 자동 MCQ(객관식 질문) 생성과 평가에 대한 새로운 접근법과 반사실적 증강 (counterfactual augmentation)을 통한 NLP 태스크의 강건성 강화 방법, 및 비디오 이상 탐지의 일반화 문제를 해결하는 새로운 딥러닝 프레임워크가 제안되었습니다.

- **Technical Details**: 자동 MCQ 생성 평가에서는 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭의 한계를 극복하기 위해 지식 종속 답변 가능성(KDA)를 도입하였으며, 딥러닝 모델의 반사실적 증강에서는 집합적 의사결정을 통한 인과관계 파악 방법이 소개되었습니다. 비디오 이상 탐지에서는 전이 학습과 모델 융합, 멀티태스크 분류를 통해 다양한 태스크에 대해 재훈련 없이 일반화할 수 있는 프레임워크가 개발되었습니다.

- **Performance Highlights**: 지식 종속 답변 가능성(KDA) 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 보여주었으며, 딥러닝 강건성 강화 방법은 반사실적 강건성, 도메인 간 일반화, 적은 데이터 일반화에서 상당한 개선을 이루었습니다. 비디오 이상 탐지 프레임워크는 RLVS 데이터셋에서 97.99%, UCF 데이터셋에서 83.59%, 두 데이터셋 전반에서 88.37%의 정확도를 달성하였으며, 보지 못한 데이터셋에서 87.25%의 정확도를 기록했습니다.



### Improving Air Mobility for Pre-Disaster Planning with Neural Network Accelerated Genetic Algorithm (https://arxiv.org/abs/2408.00790)
Comments:
          7 pages, 8 figures, ITSC 2024

- **What's New**: {'1': '기존의 자동 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ의 제목화 사실(target fact)에 대한 학생의 지식을 평가하는 능력을 무시하고 있습니다. 이를 해결하기 위해, 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.', '2': '최근 심층 학습 모델들이 NLP 태스크에서 굉장히 높은 정확성을 보였으나, 이들은 종종 spurious patterns에 의존해 robustness가 제한된다는 문제를 가지고 있습니다. 이 논문은 contrastive learning과 counterfactual augmentation을 leverage하여 이러한 제한점을 극복하려고 합니다.', '3': '자연 재해와 같은 비상 상황에서 항공 이동성 문제를 해결하기 위해 NN(neural network) 데이터를 활용한 유전자 알고리즘(GA)을 제안하여 비상 상황 이전의 공항 운영 스케줄을 최적화하는 프레임워크를 제안합니다.'}

- **Technical Details**: {'1': 'KDA는 데이터셋에 있는 gold 샘플과의 단순한 n-gram 유사도가 아닌, MCQ의 교육적 가치를 평가합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 학습된 language models를 활용해 학생의 문제 해결 행동을 흉내냅니다.', '2': "기존의 counterfactual augmentation 방법들이 spurious correlations에 영향을 받는 것과는 달리, '여러 개의' counterfactual을 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 보다 robust하게 파악합니다. 이 접근법은 집합적 결정을 통해 attribution-based synthesis 편향에 덜 민감합니다.", '3': 'NN과 GA를 결합하여 각각의 고립된 공항 데이터가 아니라, 여러 공항의 데이터를 통합하여 분석하고, 일정 조정의 최적화를 위한 도구로 사용합니다. 특히 AC와 AT 트래픽에 최소 충격을 주면서 GAV와 MIL 기간의 능력을 극대화하여 비상 상황에 대비한 출발 항공기의 능력을 최적화합니다.'}

- **Performance Highlights**: {'1': 'Human studies를 통해 KDA_disc와 KDA_cont는 KDA 자체와 전문가가 라벨링한 실제 강의실 설정에서의 사용성 둘 다 강한 상관관계를 보여주었습니다. 이는 n-gram 기반의 유사도 메트릭과 결합할 때 다양한 전문가 라벨 MCQ 품질 척도에 대해 강력한 예측 능력을 가집니다.', '2': '제안된 접근법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data로부터의 generalization에서 모두 상당한 개선을 이루었음을 실험 결과로 보여줍니다.', '3': '유전자 알고리즘(GA)과 NN의 결합을 통해 스케줄링 최적화 문제에서 연산 오버헤드를 줄이면서 비교 가능한 성능을 달성했으며, 그 효과는 테스트되지 않은 공항에서도 유지되었습니다. 특히, 일반 항공(GAV)과 군사 항공(MIL)의 비상 대책을 조정하여 출발 능력을 크게 향상시켰습니다.'}



### Machine Learning for Dynamic Management Zone in Smart Farming (https://arxiv.org/abs/2408.00789)
- **Multiple Choice Question Generation Evaluation**: [{"What's New": "기존의 BLEU, ROUGE, METEOR 등의 MCQ 생성 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해, 새로운 평가 메트릭인 'Knowledge Dependent Answerability(KDA)'를 제안하였다."}, {'Technical Details': 'KDA는 대상 사실에 대한 학생 지식을 기반으로 MCQ의 대답 가능성을 측정합니다. 이를 위해, Human survey를 통해 KDA를 측정하고, 이를 자동화하는 메트릭(KDA_disc와 KDA_cont)을 제안합니다. 이 메트릭은 사전 학습된 언어 모델을 이용해 학생들의 문제 해결 과정을 모방합니다.'}, {'Performance Highlights': 'Human Studies를 통해 KDA_disc와 KDA_cont가 (1) KDA와 (2) 전문가가 라벨링한 실제 강의실 세트에서의 사용성과 강한 상관관계를 가짐을 확인하였습니다. 또한, n-gram 기반 유사성 메트릭과 결합할 때 다양한 전문가 라벨링 MCQ 품질 척도에 대한 강력한 예측력을 보여줍니다.'}]

- **NLP Task Model Robustness**: [{"What's New": '최근 NLP 태스크에서 deep model의 높은 정확성에도 불구하고 spurious 패턴에 의존하여 robustness가 제한되는 문제를 해결하기 위해, 대조 학습과 반사실적 증강(counterfactual augmentation) 방법을 제안합니다.'}, {'Technical Details': '기존 증강 방법과 달리, 여러 반사실적(counterfactual) 예시를 생성하고 집합적 의사 결정을 통해 단어들의 인과관계를 보다 견고하게 감독합니다.'}, {'Performance Highlights': '제안된 접근법은 다양한 차원에서 중요한 개선을 이루었으며, task 모델 바이어스에 덜 민감하여 1) 반사실적 robustness, 2) 도메인 간 일반화, 3) 제한된 데이터에서의 일반화에서 성과를 보여줍니다.'}]

- **Digital Agriculture**: [{"What's New": '디지털 농업 분야에서 머신러닝 클러스터링 알고리즘을 이용한 동적 관리 구역 분할 접근법(dynamic management zone delineation approach)을 제안합니다.'}, {'Technical Details': '고수량 데이터, 고도 및 토양 질감 맵, NDVI 데이터를 사용하여 공간적 변동성을 분석하는 데 유용한 관리 구역을 분할합니다. 이를 통해 지속적인 문제를 식별하고, 주어진 필드 내에서 최적의 적응형 관리 방식을 제공합니다.'}, {'Performance Highlights': '제안된 접근법은 농업 전문가와의 협업을 통해 현장 상황에 맞춘 가변 비율 N 비료 적용을 보다 효과적으로 수행할 수 있도록 지원합니다.'}]



### Whether to trust: the ML leap of faith (https://arxiv.org/abs/2408.00786)
Comments:
          12 pages, 12 figures

- **What's New**: 최근 multiple choice question(MCQ) 자동 생성의 평가 메트릭은 학생 지식 평가라는 목적을 간과하고 있습니다. 이를 해결하기 위해, 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭을 제안했습니다. KDA는 대상 사실에 대한 학생의 답변 가능성을 측정합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 메트릭과 달리, 본 논문에서는 학생들의 응답을 바탕으로 KDA를 측정하고, 사전 학습된 언어 모델을 활용한 KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안했습니다. 이를 통해 학생의 문제 해결 행동을 모방하는 접근법을 사용합니다.

- **Performance Highlights**: 인간 연구 결과에서 KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 보여주었습니다. 또한, n-gram 기반의 유사성 메트릭과 결합 시, 다양한 전문가가 라벨링한 MCQ 품질 측정치에 대한 높은 예측력을 보였습니다.



### In-Depth Analysis of Emotion Recognition through Knowledge-Based Large Language Models (https://arxiv.org/abs/2408.00780)
Comments:
          7 pages

- **What's New**: 논문에서 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 소개합니다. KDA는 생성된 다지선다형 질문(MCQ)의 대답 가능성을 측정하여 학생이 해당 사실에 대해 얼마나 지식을 가지고 있는지를 평가합니다. 또한, 반사실적 데이터 증강을 활용하여 자연어 처리(NLP) 모델의 강건성(robustness)을 향상시키는 방법과 새로운 컨텍스트 기반 감정 인식 접근 방식을 제안합니다.

- **Technical Details**: MCQ 데이터셋에 있는 골드 샘플과 비교하는 기존 메트릭의 단점을 해결하기 위해, KDA는 학생 응답을 기반으로 한 평가 방식을 사용합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방함으로써 자동으로 KDA를 근사합니다. 또한, NLP 태스크에서 spurious pattern을 방지하기 위해, 반사실적 데이터 증강을 사용하여 모델의 예측 분포를 집합적 의사 결정(collective decision)을 통해 개선하는 방법을 제안합니다. 감정 인식을 위해서는 Bayesian Cue Integration (BCI) 방법을 사용하여 상황적 지식과 얼굴 표정 데이터를 통합합니다.

- **Performance Highlights**: Human studies에서 KDA_disc와 KDA_cont가 실제 강의실에서의 사용성과 강하게 상관됨을 보였습니다. 또한, KDA_disc와 KDA_cont는 n-gram 기반의 유사성 메트릭과 결합할 때 전문가가 레이블한 다양한 MCQ 품질 측정치에 대해 높은 예측력을 나타냈습니다. NLP 모델의 강건성을 향상시키기 위해 제안된 방법은 반사실적 강건성(counterfactual robustness), 크로스 도메인 일반화(cross-domain generalization), 그리고 희소한 데이터에서 일반화(generalization from scarce data) 영역에서 중요한 개선을 이루었습니다. 감정 인식에서 BCI 방법은 인간 관찰자와 비교할 때 높은 정확도를 보였으며, 컨텍스트와 얼굴 표정 데이터를 통합하여 더 다양한 감정을 인식하는 데 성공했습니다.



### Learning Structurally Stabilized Representations for Multi-modal Lossless DNA Storag (https://arxiv.org/abs/2408.00779)
- **What's New**: 이 논문에서는 다중 선택 질문(MCQ)의 자동 생성의 평가 메트릭으로 새롭게 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 개념을 도입하여 질문이 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정한다는 아이디어를 제안합니다. 또한, 최근의 딥러닝 모델이 자연어 처리(NLP) 작업에서 놀라운 정확성을 보이긴 했지만, 과도하게 특정 패턴에 의존하는 데서 오는 취약점을 해결하기 위해 대조 학습(contrastive learning)과 반사실적 증대(counterfactual augmentation)를 활용하는 방법도 제안되었습니다. 마지막으로, 다중 모달 손실 없는 DNA 저장을 위한 Reed-Solomon 코딩 기반 단일 가닥 표현 학습(RSRL) 모델을 소개하여 데이터의 높은 정보 밀도와 내구성을 보장하는 접근 방식을 제안합니다.

- **Technical Details**: 제안된 KDA는 학생들의 반응을 통해 MCQ의 답변 가능성을 평가하는 메트릭으로, 이를 바탕으로 두 개의 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안합니다. 대조 학습과 반사실적 증대는 여러 개의 반사실을 생성하고 집합적 의사 결정(collective decisions)을 통해 모델의 편향성을 줄이는 방법입니다. 마지막으로 제안된 RSRL 모델은 Reed-Solomon 코덱을 사용하여 데이터를 처리하고 안정된 단일 가닥 구조를 보장하는 생물학적으로 안정화된 손실 함수를 포함하여 데이터를 DNA 서열로 변환하는 과정을 설명합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 교실에서의 사용성과 강한 상관관계를 가지며, 전문가가 라벨링한 MCQ 품질 측정에 강한 예측력을 보였습니다. 대조 학습 접근법은 반사실적 강건성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 소량 데이터로부터의 일반화(generalization from scarce data)에서 눈에 띄는 향상을 달성했습니다. RSRL 모델은 여러 강력한 기준선과의 비교 실험에서 학습 복잡성을 줄이고, 정보 밀도를 18% 증가시키며, 열역학 성능을 11% 향상시키고, 코딩 및 디코딩 지연을 두 자릿수 크기로 줄이는 데 성공했습니다.



### Frontend Diffusion: Exploring Intent-Based User Interfaces through Abstract-to-Detailed Task Transitions (https://arxiv.org/abs/2408.00778)
- **What's New**: 최근 연구에서 MCQ 생성의 자동화와 학생 평가를 더욱 효과적으로 할 수 있는 새로운 지표를 제안했습니다. 또한, 줄의 패턴에 의존하지 않고 더 robust한 NLP 모델을 개발하기 위해 대조 학습 및 counterfactual augmentation을 사용하는 방법을 소개했습니다. 마지막으로, 사용자의 의도를 보다 정확히 반영하는 새로운 인터페이스 방법을 통해 frontend 코드 생성을 자동화하는 도구를 개발했습니다.

- **MCQ Generation**: {'Technical Details': '기존의 평가지표가 단순히 n-gram 기반 유사성만 측정했다면, 새로운 Knowledge Dependent Answerability (KDA)는 학생의 지식을 평가하는 능력을 중심으로 MCQ의 대답 가능성을 측정합니다. 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안했습니다.', 'Performance Highlights': 'Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서 활용 가능한지 평가했습니다. 이 메트릭들은 전문가가 라벨링한 MCQ 품질 측정치와 강한 상관관계를 보여주었습니다.'}

- **Robust NLP Models**: {'Technical Details': "spurious pattern에 의존하지 않도록 대조 학습과 counterfactual augmentation을 활용했습니다. '여러 개의' counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 통해 robustness를 획득했습니다.", 'Performance Highlights': 'Empirical results는 이 접근법이 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, scarce data에서의 일반화에 상당한 향상을 보여주었습니다.'}

- **Intent-Based UI for Frontend Development**: {'Technical Details': 'Frontend Diffusion이라는 도구는 사용자가 만든 초안을 바탕으로 웹사이트를 생성합니다. 이는 스케치, 작성, 코딩의 세 가지 단계로 이루어지며, Claude 3.5 Sonnet 언어 모델을 사용하여 텍스트와 코드를 생성합니다. Pexels API를 활용해 이미지를 포함한 고품질의 웹사이트를 생성합니다.', 'Performance Highlights': '생성된 웹사이트는 시각적으로 만족스러운 결과를 보였으며, 사용자가 의도만 표현하면 Generative AI가 최종 출력을 제공할 수 있어 사용자와 시스템 간의 의사소통 비용을 줄일 수 있습니다.'}



### Decoding AI and Human Authorship: Nuances Revealed Through NLP and Statistical Analysis (https://arxiv.org/abs/2408.00769)
- **What's New**: ['자동 MCQ 생성 평가의 새로운 지표인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 도입했습니다. 이는 기존의 BLEU, ROUGE, METEOR과 같은 n-gram 기반 평가와는 달리 교육적 가치를 반영합니다.', '심층 신경망 모델의 견고성을 높이기 위해 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 활용하는 방법을 제안했습니다.', '인공지능과 인간이 작성한 텍스트의 미묘한 차이를 분석하는 연구를 통해 AI의 창의적 능력과 이를 활용한 언어 생성의 영향을 탐구했습니다.']

- **Technical Details**: ['KDA는 학생이 특정 사실에 대한 지식을 가지고 있을 때 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표가 제안됩니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.', '대조 학습과 반사실적 증강을 사용하여 다수의 반사실적 데이터를 생성하고, 집합적 의사 결정(collective decision)을 통해 각각의 단어의 인과 관계를 보다 견고하게 감독하는 방식을 제안했습니다.', '총 500K개의 에세이를 분석하여 AI 생성 텍스트와 인간 작성 텍스트 간의 언어적 특성, 창의성 패턴, 잠재적 편향성을 조사했습니다. 이를 통해 AI의 언어 생성 능력과 창의적 능력에 대한 심층적인 이해를 돕습니다.']

- **Performance Highlights**: ['KDA_disc와 KDA_cont는 학습자 및 전문가가 평가한 실제 강의실 세트에서의 사용성과 강한 상관관계를 보였습니다.', '제안된 반사실적 증강 방법은 반사실적 견고성(counterfactual robustness), 도메인 간 일반화(cross-domain generalization), 부족한 데이터에서도 우수한 성능을 보였습니다.', 'AI 생성 텍스트는 인간 작성 텍스트에 비해 평균 단어 길이가 길고, 다소 높은 수준의 참신성을 보여주어 더 독창적인 콘텐츠 생성 가능성을 제시했습니다. 하지만, 인간 작성 텍스트는 더 높은 어휘 다양성을 보였습니다.']



### Comparing Optical Flow and Deep Learning to Enable Computationally Efficient Traffic Event Detection with Space-Filling Curves (https://arxiv.org/abs/2408.00768)
Comments:
          27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024)

- **What's New**: 최근 연구에서는 Multiple Choice Questions(MCQ) 자동 생성의 평가 메트릭 문제를 지적하며, Knowledge Dependent Answerability(KDA)라는 새로운 자동 평가 메트릭을 제안했습니다. 이 메트릭은 MCQ의 답변 가능성을 측정하여 교육적 가치를 평가합니다.

- **Technical Details**: 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 n-gram 기반 유사성에 초점을 맞추는 반면, KDA는 대상 사실에 대한 학생의 지식을 평가하는 데 중점을 둡니다. KDA는 학생들이 문제를 풀어보는 human survey를 기반으로 측정하고, 이를 자동화하기 위해 KDA_disc 및 KDA_cont를 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 높은 상관관계를 가지며, 기존의 n-gram 기반 메트릭과 결합할 때 예측력이 강해집니다.



### Characterizing User Archetypes and Discussions on Scored.co (https://arxiv.org/abs/2407.21753)
- **What's New**: 최근 출판된 논문에서는 자동 MCQ(객관식 질문) 생성의 평가를 위한 새로운 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 또한, NLP 태스크에서 deep model의 robust성을 높이기 위해 대조 학습과 counterfactual augmentation을 사용하는 방법과, 소셜 하이퍼네트워크의 노드 및 하이퍼엣지를 특성화하는 다차원 프레임워크를 제안했습니다.

- **Technical Details**: [{'MCQ Generation': '기존의 BLEU, ROUGE, METEOR 등의 평가 메트릭은 데이터셋 내의 골드 샘플과의 n-그램 기반 유사성에 중점을 두며 교육적 가치를 평가하지 않습니다. 이를 해결하기 위해, KDA는 학생이 특정 사실에 대한 지식을 바탕으로 객관식 질문에 대답할 수 있는지를 측정합니다. 우리는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 학습된 언어 모델을 통해 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.', 'Deep Model Robustness': '최근 deep model들은 NLP 태스크에서 초인간적인 정확성을 보였지만, spurious patterns에 의존하여 robust성이 제한되었습니다. 이를 해결하기 위해 대조 학습과 counterfactual augmentation을 사용하며, 특히 여러 개의 counterfactual을 생성하고 집합적 의사 결정(collective decision)을 통해 단어들의 인과관계를 더욱 robust하게 감독할 수 있습니다.', 'Social Hypernetworks': '사회적 상호작용의 증가와 복잡성에도 불구하고, 일부 fringe social platforms에 대한 연구는 거의 이루어지지 않았습니다. 이 논문은 understudied alt-right 플랫폼인 Scored.co를 중심으로 소셜 하이퍼네트워크의 노드와 하이퍼엣지를 특성화하기 위한 다차원 프레임워크를 제시합니다. 이 프레임워크는 hypernetwork 표현과 다양한 노드 특성을 통합하여 distinct user archetypes를 정의하고 이들의 네트워크 내 역할을 이해하는 데 중점을 둡니다.'}]

- **Performance Highlights**: [{'MCQ Generation': 'human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계가 있음을 입증했습니다. 또한, n-그램 기반 유사성 메트릭과 결합할 때, KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ 품질 측정에 대한 강력한 예측력을 가지고 있습니다.', 'Deep Model Robustness': '제안된 방법은 다양한 측면에서 유의미한 개선을 달성했습니다: 1) counterfactual robustness, 2) cross-domain generalization, 3) scarce data에서의 generalization.', 'Social Hypernetworks': '제안된 프레임워크는 Scored.co 플랫폼의 데이터를 이용한 실험 캠페인을 통해 높은 유효성을 보였습니다. 이 연구는 Scored.co의 구조적 특성을 최초로 분석한 연구이기도 합니다.'}]




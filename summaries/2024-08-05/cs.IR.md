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




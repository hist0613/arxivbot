New uploads on arXiv(cs.CL)

### PromptOptMe: Error-Aware Prompt Compression for LLM-based MT Evaluation Metrics (https://arxiv.org/abs/2412.16120)
- **What's New**: 이 논문은 기계 생성 자연어 콘텐츠의 품질 평가를 위한 새로운 방법론을 제안합니다. 특히, 기존의 대형 언어 모델(LLM)을 사용하는 대신, 작은 언어 모델을 활용하여 평가에 필요한 입력 데이터를 압축함으로써 토큰 사용량과 계산 비용을 줄이는 프롬프트 최적화 전략을 채택합니다. 이러한 접근 방식은 MT(기계 번역) 평가에 중점을 두고 있으며, 기존의 GEMBA-MQM 지표의 성과를 개선합니다. 결과적으로 약 2.37배의 토큰 사용량 감소를 이루면서도 평가 품질 손실 없이 효율성을 높이는 데 기여하고 있습니다.

- **Technical Details**: PromptOptMe라는 새로운 방법은 전이 학습과.preference optimization의 두 단계로 구성됩니다. 이는 기계 번역(번역 품질 평가)에 적용되며, 기존의 LLM 기반 평가 지표에 비해 보다 저렴한 비용으로 사용할 수 있도록 설계되었습니다. 이를 통해 PromptOptMe는 대규모 MT 시스템 출력의 온라인 재정렬 및 웹 규모 데이터 세트 처리와 같은 다양한 작업에 용이하게 적용될 수 있습니다. 연구는 WMT22 Metrics Challenge Test Set에서의 효율성을 기반으로 하여, 기존의 LLM 방법들이 가진 계산적 비효율성을 극복하고자 합니다.

- **Performance Highlights**: 이 연구의 주요 성과 중 하나는 사용자 기반 선호에 따라 모델 출력을 최적화하여 평가 품질은 유지하면서도 약 2.32배의 토큰 사용량 감소를 달성한 것입니다. 이는 고급 LLM 기반 평가 지표의 비용 효과성과 효율성을 높이는 데 크게 기여합니다. 덕분에 여러 자원이 부족한 커뮤니티에서도 이러한 고급 지표를 더욱 쉽게 사용할 수 있도록 하여 다양성과 포용성을 증진시킵니다. 여기에 더해, 기존의 기계 번역 평가 시스템에 대한 접근 방식의 혁신적인 변화를 촉진하는 동시에 LLM 기반 평가에서의 가능성을 확장합니다.



### Logical Consistency of Large Language Models in Fact-checking (https://arxiv.org/abs/2412.16100)
Comments:
          Under review

- **What's New**: 최근 대규모 언어 모델(LLMs)이 복잡한 논리 쿼리에서의 논리적 일관성 문제를 해결하는 데 초점을 맞추고 있습니다. 기존 연구들은 단순한 재구성이 가능한 쿼리에 대한 일관성을 평가해왔으나, 이 연구는 기본적인 논리 연산자인 부정(negation), 결합(conjunction), 및 분리(disjunction)를 사용하여 복잡한 쿼리에 대한 일관성을 측정합니다. 본 연구는 사실 검증을 위한 LLM을 평가하기 위해 실세계 지식 그래프(KGs)의 데이터를 활용합니다.

- **Technical Details**: 논문에서 제안하는 새로운 사실 검증 데이터 세트(henceforth, LFC)는 Freebase, NELL, WikiKG90Mv2에 기반하여 만들어졌으며, 이 데이터 세트는 LLM의 논리적 일관성을 평가하는 데 적합하도록 변환된 형식을 갖습니다. 이 연구는 다양한 요인, 즉 간단하고 복잡한 사실, 다양한 논리 규칙을 고려하여 LLM의 반응 일관성을 평가합니다. 기존 LLM들은 KG 문맥을 고려할 때 정확성은 증가하나, 논리적 동등성을 고려했을 때 일관성 부족이 나타납니다.

- **Performance Highlights**: 제안된 방법론을 통해 LLM의 일관성을 개선하는 데 성공하였으며, 이는 감독하에 미세 조정(supervised fine-tuning)을 통해 이루어졌습니다. 연구 결과, KG 컨텍스트를 사용하는 복잡한 사실 검증에서 LLM의 일관성을 향상시키는 방법을 제시하였고, 이는 기존의 사전 훈련 접근 방식을 넘어서는 효과적인 결과를 보여줍니다. 실험 결과는 고유한 논리적 쿼리에 대한 일관성 평가가 기존 연구에서 부족했음을 입증하며, LLM의 믿을 수 있는 시스템 구축에 기여할 수 있음을 시사합니다.



### The Only Way is Ethics: A Guide to Ethical Research with Large Language Models (https://arxiv.org/abs/2412.16022)
Comments:
          Accepted to COLING '25. This paper is the condensed pocket guide to accompany our full LLM Ethics Whitepaper, available at arXiv:2410.19812, and at this https URL for suggested revisions

- **What's New**: 이 논문에서 소개하는 'LLM 윤리 백서'는 대규모 언어 모델(LLM)의 윤리적 고려사항을 통합한 실용적인 가이드입니다. 이전의 여러 연구와 기구들이 제안한 윤리적 문제와 정책들을 한데 모아, NLP (Natural Language Processing) 실무자들이 참고할 수 있도록 구성되었습니다. 주목할 점은 기존의 문헌을 모아 명확한 Do's and Don'ts로 정리했으며, 이는 LLM을 다루는 연구자들에게 즉각적으로 적용 가능한 지침을 제공합니다.

- **Technical Details**: 논문은 LLM의 프로젝트 생애주기 전반에 걸쳐 적용할 수 있는 윤리적 고려사항을 제시합니다. NIST의 AI 위험 관리 프레임워크와 EU AI 법률처럼 널리 알려진 지침들과 비교하여, LLM 윤리 백서는 이론보다는 실용적인 지침을 제공합니다. 이를 위해 ACL Anthology와 Semantic Scholar에서 체계적인 문헌 조사를 실시하고, 관련된 자료를 수집하여 각 프로젝트 단계에 적합한 내용을 분류했습니다.

- **Performance Highlights**: 윤리적 연구를 지원하기 위해 마련된 LLM 윤리 백서는 실무자에게 유용한 자원으로 작용할 것입니다. 초기 탐색 과정에서의 리소스를 하이라이트하고, 이해관계자와 협력하는 데 도움이 되는 모범 사례를 제시합니다. 이 문서는 LLM 연구를 수행하는 모든 이에게 귀중한 참고자료가 될 것으로 기대됩니다.



### Data-Centric Improvements for Enhancing Multi-Modal Understanding in Spoken Conversation Modeling (https://arxiv.org/abs/2412.15995)
Comments:
          22 pages, 6 figures, 14 tables

- **What's New**: 이 연구는 다중 모달 음성 모델링의 향상을 위해 데이터 중심(customization) 접근 방식을 도입하였습니다. 구체적으로, 소량의 음성 데이터만을 사용하여 다중 작업 학습(multi-task learning)을 통해 음성 이해를 효과적으로 증대시키는 방법을 제안합니다. 또한, 모호한 사용자 요청에 대한 대화형 질문 응답 데이터셋인 ASK-QA를 소개하여 새로운 연구 방향을 제시합니다.

- **Technical Details**: 이 논문에서는 모달리티 간 학습을 극대화하고자 하는 보조 작업(auxiliary tasks)을 설계하여 주어진 데이터 세트 내에서 상호 작용하는 알고리즘을 구축합니다. 세 가지 중간 목표는 올바른 음성 맥락 표현, 모든 입력 모달리티 간의 추론 학습, 및 올바른 답변 생성입니다. 주된 목표는 MLLM이 입력된 음성을 바탕으로 정확한 답변을 제공하는 것입니다.

- **Performance Highlights**: 제안된 접근 방식은 Spoken-SQuAD 벤치마크에서 10%의 훈련 데이터만을 사용하여 기존의 최첨단 성능을 초월하는 성과를 이루었습니다. ASK-QA와 Spoken-SQuAD를 포함한 세 가지 음성 질문 응답(SQA) 데이터 세트를 통해 이 모델의 효과성을 검증하고, 특히 데이터가 제한된 경우에도 우수한 결과를 보여줍니다.



### Fearful Falcons and Angry Llamas: Emotion Category Annotations of Arguments by Humans and LLMs (https://arxiv.org/abs/2412.15993)
- **What's New**: 이 논문은 주장의 감정이 주장의 효과에 미치는 영향을 조사합니다. 특히, 이 연구는 이진 감정성(binary emotionality) 연구에 이어, 특정 감정 카테고리(예: "분노", anger)에 대한 주관적인 주석(annotation)을 독일어 주장 데이터셋에서 crowdsourcing합니다. 또한, 자동 LLM 기반의 레이블링 방법을 평가하여 기존 연구의 한계를 보완합니다.

- **Technical Details**: 연구에서는 세 가지 프롬프팅 전략(zero-shot, one-shot, chain-of-thought)을 사용하여 세 가지 대형 지침 조정 언어 모델(Falcon-7b-instruct, Llama-3.1-8B-instruct, GPT-4o-mini)을 비교합니다. 출력 공간(output space)의 정의를 이진(binary), 폐쇄형(closed-domain), 개방형(open-domain)으로 다양화하여 감정 예측을 분석합니다. 이 과정에서 감정 카테고리는 주장의 감정성 예측을 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 모든 프롬프트 설정과 모델을 통해 자동 예측은 높은 재현율(recall)을 보였지만, 정서 분류의 정밀도(precision)는 낮았습니다. 특히, 분노와 두려움에 대한 예측에서 부정적인 감정에 대한 강한 편향이 나타났습니다. 이 연구는 감정 주석의 필요성을 강조하며, 향후 연구 필요성을 제시합니다.



### BabyHGRN: Exploring RNNs for Sample-Efficient Training of Language Models (https://arxiv.org/abs/2412.15978)
Comments:
          7 pages, 7 figures and tables, Published in Proceedings of the BabyLM Challenge 2025

- **What's New**: 이번 논문은 저자들이 최근 제안한 RNN 기반 아키텍처인 HGRN2를 활용하여 transformer 기반 모델과의 비교 연구를 진행한 내용을 다룹니다. 저자들은 언어 모델링에서의 저자원 환경에서 transformer보다 더 나은 성능을 보여주는 HGRN2 언어 모델인 BABYHGRN을 소개합니다. 지식 증류(knowledge distillation)를 통한 성능 향상의 긍정적인 영향을도 보여주며, RNN 모델의 가능성을 재조명하고 있습니다.

- **Technical Details**: HGRN2 모델은 2048의 히든 사이즈와 18개의 레이어로 구성되어 있으며, 총 330M의 파라미터 수를 가지고 있습니다. 저자들은 다양한 도메인을 포함하는 Pile 데이터셋을 사용하여 언어 모델을 훈련하였고, 비트페어 인코딩(Byte-Pair Encoding, BPE) 방식을 통해 16,000개의 토큰으로 이루어진 어휘를 사용했습니다. 또한, 지식 증류를 통해 두 개의 같은 크기 HGRN2 모델을 학생 및 교사 모델로 설정하여 훈련했습니다.

- **Performance Highlights**: BABYHGRN은 BabyLM 챌린지의 10M 및 100M 단어 코너에서 transformer 기반 모델보다 더 나은 성능을 보였으며, BLiMP, EWoK, GLUE 및 BEAR 벤치마크에서 테스트되었습니다. 이 결과는 저자들이 제시한 RNN 기반 모델의 접근 방식이 자원이 제한된 환경에서도 효과적일 수 있다는 것을 시사합니다. 이 연구는 transformer 아키텍처에 대한 집중적인 연구를 도전하고 RNN 모델의 유효성을 제안하는 중요한 기여로 평가됩니다.



### From General to Specific: Tailoring Large Language Models for Personalized Healthcar (https://arxiv.org/abs/2412.15957)
- **What's New**: 의료 분야에서 의료 LLM의 개인화가 필수적이라는 문제를 다루고 있는 연구가 소개되었습니다. 본 연구에서는 개인화된 의료 언어 모델(PMLM)을 제안하며, 추천 시스템과 강화 학습(reinforcement learning, RL)을 통해 개인 맞춤형 LLM의 최적화를 탐구합니다. 특히, PMLM은 개인의 요구에 맞춘 최초의 개인화된 프롬프트를 설계하고 이를 RL을 통해 더욱 정제하여 LLM의 정확한 방향성을 활용하도록 합니다.

- **Technical Details**: 연구에서는 환자의 과거 데이터를 분석하여 개인화된 정보를 추출하고, 유사한 환자의 통찰력을 결합하여 원주율 프롬프트를 생성하는 프로세스를 설명합니다. 이러한 초기 프롬프트는 강화 학습을 통해 세밀한 개인화를 위해 정제됩니다. PMLM의 프롬프트는 하드 프롬프트(hard prompt)로, 이는 높은 적응성과 재사용성을 부여하여 다양한 LLM에 직접적으로 활용할 수 있습니다.

- **Performance Highlights**: 실제 산부인과 데이터를 통해 평가한 결과, PMLM은 개인화된 응답을 제공함으로써 기존의 세밀하게 조정된 LLM들보다 더 나은 성과를 보였습니다. 이 연구는 LLM의 개인화 가능성을 높이고, 개인 맞춤형 의료 LLM의 발전을 위한 새로운 경로를 제시합니다. PMLM은 다양한 질병에 대해 대응할 수 있는 가능성을 갖추고 있어, 향후 의료 분야에서의 LLM 활용에 중요한 기여를 할 것으로 기대됩니다.



### Development of a Large-scale Dataset of Chest Computed Tomography Reports in Japanese and a High-performance Finding Classification Mod (https://arxiv.org/abs/2412.15907)
Comments:
          Dataset available at this https URL

- **What's New**: 이번 연구는 일본어 CT 리포트 데이터셋을 개발하고, 구조화된 발견(classification) 분류를 위한 전문화된 언어 모델을 구축한 점에서 주목받고 있습니다. 특히, CT 스캐너의 활용이 높은 일본에서 대규모 방사선의학 데이터셋의 부족 문제를 해결하기 위한 노력이 중요합니다.

- **Technical Details**: 연구진은 GPT-4o mini를 사용하여 CT-RATE 데이터셋(21,304명 환자의 24,283 CT 리포트)을 일본어로 번역했습니다. 훈련 데이터셋은 기계 번역된 22,778개의 리포트로 구성되었으며, 검증 데이터셋은 방사선 전문의의 검토를 거친 150개의 리포트로 이루어졌습니다. 또한, 'tohoku-nlp/bert-base-japanese-v3' 아키텍처를 기반으로 하는 CT-BERT-JPN 모델을 개발하여 18개 구조화된 발견을 추출했습니다.

- **Performance Highlights**: 번역 메트릭스에서는 BLEU 점수가 0.731, 0.690으로 강력한 성과를 보였으며, ROUGE 점수는 Findings 섹션에서 0.770에서 0.876까지, Impression 섹션에서는 0.748에서 0.857까지 달성했습니다. CT-BERT-JPN은 18개 조건 중 11개에서 GPT-4o보다 뛰어난 성능을 나타냈고, 특히 림프절 비대에서 +14.2%의 성능 향상을 기록했습니다. 이 모델은 18개 조건 중 14개에서 F1 점수가 0.95를 초과하며, 4개 조건에서는 완벽한 점수를 달성했습니다.



### On the Suitability of pre-trained foundational LLMs for Analysis in German Legal Education (https://arxiv.org/abs/2412.15902)
Comments:
          11 pages

- **What's New**: 이 논문은 최근 오픈 소스 기반의 LLM(대형 언어 모델)이 교육적 맥락에서 일부 법률 분석에 충분한 지시 능력 및 독일 법률 배경 지식을 보유하고 있음을 보여줍니다. 그러나, 'Gutachtenstil' 성격의 구성 요소 분류와 같은 특정 작업이나 복잡한 맥락에서 모델의 능력이 저하된다는 점을 지적합니다. 이를 보완하기 위해 Retrieval Augmented Generation 기반의 프롬프트 예제 선택 방법을 소개하여 데이터가 많이 존재하는 상황에서 예측 성능을 크게 향상시킵니다.

- **Technical Details**: 대형 언어 모델(LLMs)은 웹 스케일 코퍼스를 기반으로 훈련되어 인간의 지식과 커뮤니케이션을 포괄하는 상당한 교차 섹션을 제공합니다. LLMs는 자연어 지시를 위한 일반화 능력을 제공하며, 이를 통해 법률 분석과 같은 특정 분야에서도 일반 모델을 적용할 가능성을 제시합니다. 특히, 'Gutachtenstil'(감정 스타일)의 요소를 LLM이 정확하게 인식할 수 있는지를 테스트하기 위한 새로운 데이터셋을 도입하여 모델의 평가 능력을 검증합니다.

- **Performance Highlights**: 사전 훈련된 LLM은 논거 탐사 및 자동 에세이 점수 매기기와 같은 두 가지 표준 작업에서 성능이 더 적합하다는 평가를 받았습니다. 또한, Chain-of-Thought 프롬프트를 통해 라벨이 거의 없는 데이터 환경에서도 기준 성능을 초과하는 경향을 보였습니다. 문맥과 작업의 복잡성을 고려할 때, 이 연구는 법률 분석에 대한 LLM의 적용 가능성을 탐구함으로써 중요한 통찰력을 제공합니다.



### A Thorough Investigation into the Application of Deep CNN for Enhancing Natural Language Processing Capabilities (https://arxiv.org/abs/2412.15900)
- **What's New**: 본 논문은 자연어 처리(Natural Language Processing, NLP) 분야에서 전통적인 모델들이 겪는 정확성과 효율성 문제를 해결하기 위해 딥 컨볼루션 신경망(Deep Convolutional Neural Networks, DCNN)을 도입한 내용을 다룹니다. 새로운 접근 방식으로 DCNN, 머신 러닝(Machine Learning) 알고리즘, 생성적 적대 신경망(Generative Adversarial Networks, GAN)을 통합하여 언어 이해도를 높이고 모호성을 줄였습니다.

- **Technical Details**: 이 연구는 DCNN을 통한 NLP 모델의 통합된 접근 방식이 단어 분할(word segmentation), 품사 태깅(part-of-speech tagging), 기계 번역(machine translation), 텍스트 분류(text classification)와 같은 다양한 작업에서 뛰어난 성능을 보임을 입증하였습니다. 특히, 이 모델은 전통적인 모델 대비 10%의 분할 정확도(segmentation accuracy) 향상과 4%의 재현율(recall rate) 증가를 기록했습니다.

- **Performance Highlights**: 이 모델은 단어 인식 정확도와 처리 효율성을 개선하여 여러 NLP 작업에서 비약적인 성과를 달성했습니다.단순한 데이터 처리 이상으로, 모호성을 줄이고 다양한 언어 태스크에서 더 나은 결과를 이루어내고 있습니다.



### TelcoLM: collecting data, adapting, and benchmarking language models for the telecommunication domain (https://arxiv.org/abs/2412.15891)
Comments:
          30 pages (main: 13 pages, appendices: 17 pages), 1 figure, 22 tables, achieved March 2024, released December 2024

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)이 통신업계(telco domain)에 적응할 수 있는 방법을 연구하였다. 논문에서는 800M 토큰과 80K 지침으로 구성된 방대한 특정 도메인 데이터 집합을 수집하고, 다양한 방법론을 사용하여 적응 테스트를 진행했다. 결과적으로 도메인 적응 모델이 기존의 일반 모델과 경쟁할 수 있음을 밝혀냈으며, 기존의 복잡한 튜닝 과정을 줄일 수 있음을 제안하였다.

- **Technical Details**: 연구에서 사용된 기본 모델은 Llama-2-7B이며, 다양한 적응 접근법을 비교 분석하여 적응 과정에서 불필요한 사전 훈련(pretraining) 단계를 생략해도 뛰어난 성과를 거둘 수 있음을 보여주었다. 연구에서는 도메인 특화 데이터에 대한 재훈련(retraining)을 통해 모델의 성능을 향상시키는 DAPT 방법론을 강조했다. 또한, 제안된 프레임워크는 통신업계의 복잡한 용어와 개념을 효과적으로 처리하는데 중점을 두고 있다.

- **Performance Highlights**: 최종 실험 결과에 따르면, 도메인 적응 모델이 GPT-3.5와 경쟁하는 성능을 보여주었고, 이는 통신업계에서 LLM의 활용 가능성을 시사한다. 또한 연구는 적응 방법에서 일반 모델과 경쟁할 수 있는 성과를 도출하여, 제한된 리소스를 가지고도 효율적인 접근법을 제시할 수 있음을 입증하였다. 이러한 결과는 통신산업에서 LLM이 적용될 수 있는 여러 가지 유용한 사용 사례와 직접적인 관련성을 가지고 있다.



### $\pi$-yalli: un nouveau corpus pour le nahua (https://arxiv.org/abs/2412.15821)
Comments:
          9 pages, in French language, 2 figures

- **What's New**: NAHU$^2$ 프로젝트는 프랑스-멕시코 협력에 의해 기계 학습에 적합한 π-YALLI 코퍼스를 구축하여 나우아틀(Nahuatl) 언어의 컴퓨터 자원을 개발하는 것을 목표로 하고 있습니다. 나우아틀 언어는 200만 명 이상이 사용하는 생존 언어임에도 불구하고 컴퓨터 자원이 부족한 편입니다. 이 코퍼스는 자연어 처리(NLP) 도구 개발에 필수적인 다양한 언어 모델(Language Models, LM)을 연구하는 데 사용될 것입니다.

- **Technical Details**: π-YALLI 코퍼스는 역사적 문서, 위키피디아 문서, 뉴스, 시 및 이야기, 법률 문서 등 다양한 범주의 문서를 포함하고 있습니다. 수집된 데이터는 구조가 이질적이기 때문에 반자동 처리 과정을 거쳐 비관련 내용을 제거했습니다. 코퍼스에는 약 1.912M의 토큰(token)과 14.8M의 문자(character)가 포함되어 있으며, 이는 나우아틀 언어를 위한 기계 학습 자원을 제공하는 데 중요한 기반 역할을 할 것입니다.

- **Performance Highlights**: 본 연구에서는 Word2Vec와 FastText와 같은 정적 언어 모델을 초기 분류 모델로 사용하고, 이후 ALBERT와 같은 대형 언어 모델(LLM)로 발전시킬 예정입니다. 이러한 모델들은 나우아틀 언어의 의미와 구문 관련 정보를 효과적으로 캡처함으로써 고급 의미 이해와 감정 분석, 텍스트 자동 분류 및 범주화 등의 애플리케이션에 활용됩니다.



### Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning (https://arxiv.org/abs/2412.15797)
- **What's New**: 본 논문은 Language model Ensemble with Monte Carlo Tree Search (LE-MCTS)라는 새로운 프레임워크를 제안하여 언어 모델의 프로세스 수준 앙상블을 구현합니다. LE-MCTS는 여러 언어 모델로 구성된 앙상블을 마르코프 결정 과정(Markov decision process)으로 포멀화하여 단계별 추론을 수행합니다. 이 방법은 기존의 토큰 또는 출력 수준의 앙상블 방법들이 해결하지 못한 복잡한 추론 작업에서 보다 일관된 성능을 보장할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LE-MCTS는 각 상태를 중간 추론 단계로 정의하고, 액션은 미리 정의된 언어 모델 풀에서 하나를 선택하여 다음 추론 단계를 생성하는 방식으로 설계되었습니다. 이 알고리즘은 AlphaZero에서 영감을 받아 다수의 언어 모델이 생성한 추론 단계의 통일된 공간에서 트리 탐색(tree search)을 수행합니다. LE-MCTS는 각 단계를 평가하는 과정 기반 보상 모델(process-based reward model, PRM)을 활용하여 생성된 추론 단계를 바탕으로 가장 정확한 추론 체인을 찾습니다.

- **Performance Highlights**: LE-MCTS는 다섯 가지 수학적 추론 벤치마크에서 실험을 통해 각각의 기존 LM 앙상블 방법보다 일관되게 우수한 성과를 나타냈습니다. 특히 MATH 및 MQA 데이터셋에서 각각 3.6% 및 4.3%의 성능 향상을 달성하여 복잡한 추론 문제를 해결하는 데 효과적임을 입증했습니다. 이 연구는 LE-MCTS가 다양한 추론 문제에서 일관되게 높은 성능을 유지할 수 있도록 설계된 프레임워크의 필요성을 강조합니다.



### Learning from Impairment: Leveraging Insights from Clinical Linguistics in Language Modelling Research (https://arxiv.org/abs/2412.15785)
Comments:
          accepted at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 논문은 언어 장애 연구에서 얻은 통찰력과 임상 치료를 통합하여 언어 모델(LMs)의 인간 영감을 받은 학습 전략 및 평가 프레임워크 개발의 가능성을 조사합니다. 특히, 신경언어학(neurolinguistics)과 실어증(aphasiology)에서 유래된 훈련 접근법들이 문법적 언어 기술의 회복과 일반화에 어떻게 기여할 수 있는지를 다룹니다. 이러한 통찰력을 통해 LMs의 복잡한 문법적 현상을 다루는 능력을 평가할 수 있는 철저한 평가 시스템 설계를 제안합니다.

- **Technical Details**: 연구는 언어 복잡성을 정의하는 데 어려움이 있지만, 이를 ‘객관적 대 주체 관련(object vs. agent-related)’ 또는 ‘절대적 대 상대적(absolute vs. relative)’로 구별하는 여러 차원에서 접근합니다. 문법적 구조의 인식 및 사용과 관련된 문제를 다루기 위해, 이 논문은 실어증 치료를 위해 개발된 언어 치료 접근법들을 검토하며 그 예시를 제공합니다. 이러한 치료법들은 구조적 훈련(operationalization of complexity)을 기반으로 하여 설계됩니다.

- **Performance Highlights**: 특히, 실어증 연구에서 언어 치료의 효과적인 구현이 일반화 능력을 증대시킬 수 있음을 강조합니다. 조사된 치료 방법들은 언어 사용자의 처리 능력 향상을 목표로 하며, 훈련된 아이템에 대한 일반화 효과를 증진시키기 위해 구성됩니다. 이 논문은 이러한 언어 치료 프레임워크가 인공지능 언어 모델의 성능 평가와 개선에 기여할 수 있는 방법을 보여줍니다.



### Linguistic Features Extracted by GPT-4 Improve Alzheimer's Disease Detection based on Spontaneous Speech (https://arxiv.org/abs/2412.15772)
Comments:
          Accepted at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이번 연구에서는 GPT-4를 활용해 자발적 환자 발화의 전사본에서 다섯 가지 의미적 특징을 추출하였습니다. 이는 알츠하이머병(AD)의 증상을 포착하지만 전통적인 방법으로는 효과적으로 정량화하기 어려운 특징입니다. 저자들은 이러한 특징의 임상적 중요성을 보여주며, 기존 언어적 특징 및 Random Forest 분류기와 함께 결합했을 때 AD 탐지의 성능을 크게 향상시킬 수 있음을 입증합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 ADReSS로, 156명의 참가자가 Cookie Theft 그림을 설명하는 음성 녹음으로 구성되어 있습니다. 이 데이터셋은 진단, 나이 및 성별에 따라 균형이 잡혀 있으며, CHAT 주석 형식의 수동 전사가 포함되어 있습니다. 연구자는 수동 전사와 자동 음성 인식(ASR)의 결과를 비교하여 전혀 새로운 AD 탐지 파이프라인의 효과성을 평가하였습니다.

- **Performance Highlights**: GPT에서 파생된 특징들은 기존의 언어적 특징들과 조합할 경우 AD 탐지 성능을 크게 향상시켰습니다. 특히, 인간 평가자 및 대리 측정과의 비교를 통해 '단어 찾기 어려움'이라는 특징이 높은 유의미성을 가지고 있음을 입증하였습니다. 이 접근 방식은 수동 전사와 자동 생성 전사 모두에 대해 효과적임을 보여주며, LLM의 최근 발전을 활용한 혁신적이고 영향력 있는 사용 사례로 평가됩니다.



### Critique of Impure Reason: Unveiling the reasoning behaviour of medical Large Language Models (https://arxiv.org/abs/2412.15748)
Comments:
          16 pages, 5 figures, 2 tables. Conceptualization, both authors. formal analysis, both authors. funding acquisition, both authors. investigation, both authors. resources, both authors. supervision, T.C.. validation, both authors. visualization, both authors. writing original draft, both authors. writing review and editing, both authors

- **What's New**: 본 논문에서는 최근 의료 분야에서 사용되는 Large Language Models (LLMs)의 추론 행동(reasoning behaviour)을 탐구하고, 이러한 모델의 높은 수준의 예측 정확도보다도 추론 행동에 대한 이해의 필요성을 강조합니다. 의료 AI의 Explainable AI (XAI)를 달성하기 위해 이러한 모델이 어떻게 결론에 도달하는지에 대한 통찰력을 제공하는 이론적 틀을 제안합니다. 이를 통해 의료 전문가들이 LLM의 내부 작동 방식을 이해하고 잠재적인 논리적 오류를 드러내는 데 도움을 줄 수 있습니다.

- **Technical Details**: 논문은 LLM의 추론 행동을 정의하고, 현재의 의료 LLM에서의 고급 성능 지표와 함께 추론 행동을 평가하는 방법들을 분류합니다. 특히, 논리적 추론과 인과적 추론을 포함한 다양한 유형의 추론이 있습니다. 또한, Neuro-Symbolic AI (N-SAI)라는 분야를 통해 신경망과 기호적 추론 기술을 통합하는 방법을 논의하며, 이러한 접근 방식이 추론을 더욱 투명하게 만들어주는 데 기여할 수 있음을 설명합니다.

- **Performance Highlights**: 의료 LLM의 추론 행동을 이해함으로써 사용자는 이러한 모델이 어떻게 결론에 도달했는지를 확인할 수 있으며, 이는 임상 의사 결정 과정에서 신뢰를 구축하는 데 기여합니다. LLM이 환자 진단 및 치료 제안에서의 통찰력을 제공할 경우, 이는 의사와 기계의 권고사항 간의 불일치를 명확히 할 수 있습니다. 궁극적으로 이러한 투명성은 의료 분야에 AI를 통합하고 환자 결과를 개선하는 데 중요한 역할을 할 것입니다.



### Fine-tuning Whisper on Low-Resource Languages for Real-World Applications (https://arxiv.org/abs/2412.15726)
- **What's New**: 이 논문은 OpenAI의 Whisper 모델을 저자원 언어에 맞게 미세 조정하기 위한 새로운 접근 방식을 제시하며, 스위스 독일어를 사례 연구로 사용하여 문장 수준 데이터를 긴 형태의 말뭉치(long-form corpus)로 변환하는 혁신적인 데이터 생성 방법을 도입합니다. 특히 저작권 문제로 인해 비문장 레벨 데이터의 접근이 어려운 점을 해결하는 방안을 제시하며, 이 과정에서 Whisper 모델의 긴 오디오 처리 능력을 향상시키는 방법론을 개발했습니다.

- **Technical Details**: Whisper 모델은 30초의 고정된 입력 길이를 가지며, 30초 이하는 제로(0)로 패딩해야 합니다. 이 과정에서 스위스 독일어의 문장 수준 샘플 데이터를 사용하여 여러 문장을 연속적으로 연결하는 방식으로 긴 형태의 오디오를 생성하고, 이를 통해 생성된 오디오의 타임스탬프를 수정하는 방법을 제안합니다. 이를 위해 Voice Activity Detection (VAD)을 활용하며, 샘플 간의 전환을 부드럽게 하기 위해 노이즈 겹침(Noise Overlapping) 기술이 사용되었습니다.

- **Performance Highlights**: 미세 조정된 Whisper 모델은 기존 모델에 비해 BLEU 점수가 더욱 향상되어 스위스 독일어 STT 분야에서 새로운 최첨단 모델로 자리 잡았습니다. 이 연구의 방법론은 다른 저자원 언어에도 적용 가능성을 보이며, 문장 수준 데이터만으로 긴 오디오 파일을 고품질로 기록할 수 있는 능력을 갖추고 있습니다. 이러한 성과는 실세계의 다양한 응용 프로그램에서도 긍정적인 효과를 나타내고 있습니다.



### Contrastive Learning for Task-Independent SpeechLLM-Pretraining (https://arxiv.org/abs/2412.15712)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 음성 처리 작업에 효과적으로 적응할 수 있는 확장 가능하고 두 단계로 구성된 훈련 방법론을 제안합니다. 첫 번째 단계는 대조 학습(contrastive learning)을 이용한 작업 독립적인 음성 사전 훈련입니다. 두 번째 단계는 최소한의 데이터로 수행되는 작업 특화 미세 조정입니다. 이러한 접근 방식은 전통적인 ASR 사전 훈련을 능가하며, 오직 10%의 작업 특화 데이터로도 음성 번역 및 질문 답변 전용 모델들을 초월할 수 있다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 대조 학습을 통해 음성과 텍스트의 표현을 정렬하는 사전 훈련 전략을 제안합니다. 이는 병렬 음성-텍스트 데이터 400시간을 사용하여 각 레이어 후에 대조 손실을 적용함으로써 효과적으로 이루어집니다. 최종적으로 세 가지 하위 작업인 ASR, 음성 번역(ST), 음성 질문 답변(SQA)에서 사전 훈련된 모델을 미세 조정하여 성능을 평가합니다. 그 결과, 대조 학습 전략이 다른 방법들보다 우수할 뿐만 아니라 거의 모든 지표에서 작업 특정 SOTA 모델들과의 성능이 일치함을 입증했습니다.

- **Performance Highlights**: 대조 학습의 범위를 1,400시간으로 확대함으로써 추가적인 성능 향상이 이루어졌으며, 우리의 모델이 전문화된 모델과 기존 SpeechLLMs를 초월했습니다. 특히, 자원이 제한된 시뮬레이션 환경에서도 10%의 하위 데이터를 사용하여도 기존 SpeechLLMs보다 일관되게 높은 성능을 발휘했습니다. 연구의 결과는 텍스트-텍스트 기능을 유지하면서도 음성과 텍스트 기반 작업 모두에서 기능성을 증대시키는 간단하고 유연한 SpeechLLM 아키텍처의 가능성을 제시합니다.



### Variability Need Not Imply Error: The Case of Adequate but Semantically Distinct Responses (https://arxiv.org/abs/2412.15683)
Comments:
          26 pages

- **What's New**: 이 논문에서는 언어 모델(LMs)의 응답 자유도를 평가하기 위한 새로운 방법론을 제시합니다. 보통, 모델의 응답에서 발생하는 의미적 변동성은 오류를 초래하는 것으로 여겨지지만, 저자들은 오히려 이러한 변동성이 항상 오류를 야기하는 것은 아님을 주장합니다. 이에 따라, 저자들은 응답의 적합성을 측정하기 위해 적합성 분류기를 설계하고, 우수한 성능을 나타내는 PROBAR라는 새로운 확률 지표를 도입합니다.

- **Technical Details**: 저자들은 언어 모델이 생성하는 응답의 확률을 두 개의 구성 요소(신경망 및 샘플링 알고리즘)로 나눠 설명합니다. 이들 요소는 주어진 프롬프트에 대해 조건부 확률 분포를 생성하는 역할을 합니다. 특히, 저자들은 세 가지 데이터셋을 이용하여 PROBAR와 기존의 엔트로피 측정법을 비교하고, PROBAR가 더 일관된 성능을 보여줌을 증명합니다.

- **Performance Highlights**: PROBAR는 테스트 데이터셋인 Abg-COQA, AmbigQA 및 Provo에서 높은 AUROC (Area Under the Receiver Operating Characteristic) 점수를 기록하여, 문장 모호성의 정도와 관계없이 성능을 발휘했습니다. 이는 현재 상태에서 가장 뛰어난 불확실성 정량화 기법에 비해 프로세스에서의 신뢰성을 높이는 데 기여하며, 모든 코드는 공개되었습니다.



### MathSpeech: Leveraging Small LMs for Accurate Conversion in Mathematical Speech-to-Formula (https://arxiv.org/abs/2412.15655)
Comments:
          Accepted in AAAI 2025

- **What's New**: 이 논문은 MathSpeech라는 새로운 파이프라인을 소개하고 있습니다. 이 시스템은 수학적인 표현을 구술한 내용을 정확하게 LaTeX 형식으로 변환하는 데 중점을 두고 있습니다. 기존의 자동 음성 인식 (ASR) 모델이 수학 공식 처리에 취약한 문제점을 해결하기 위해 소형 언어 모델 (sLMs)을 통합하였습니다. MathSpeech는 다양한 ASR 모델의 성능을 평가하기 위한 새로운 데이터셋을 기반으로 개발되었습니다.

- **Technical Details**: MathSpeech는 1,101개의 실제 강의 오디오 샘플을 포함하는 벤치마크 데이터셋을 활용하여 ASR 모델의 성능을 시험하는 방식을 채택하고 있습니다. 이 파이프라인은 ASR의 오류를 교정하고, 이를 소형 언어 모델을 통해 LaTeX로 변환하는 작업을 수행합니다. 특히, 120M 파라미터를 가진 소형 언어 모델로도 상업적인 대형 언어 모델(GPT-4o)보다 뛰어난 LaTeX 생성 능력을 보여줍니다. CER, BLEU 및 ROUGE 점수에서 MathSpeech는 GPT-4o에 비해 유의미한 성능 향상을 보여주었습니다.

- **Performance Highlights**: MathSpeech는 수학적 표현을 처리하는 데 있어 기존의 ASR 모델보다 월등한 성능을 보였습니다. 특히 CER 점수가 0.390에서 0.298로 감소하였고, ROUGE 및 BLEU 점수에서도 좋은 결과를 기록하였습니다. 이러한 점수는 MathSpeech가 많은 양의 파라미터를 사용하는 대형 모델에 비해 효율적인 성과를 달성했다는 것을 의미합니다. 수학 교육에 있어서 학습자들의 이해를 돕기 위한 실질적인 기여를 할 것으로 기대됩니다.



### Error-driven Data-efficient Large Multimodal Model Tuning (https://arxiv.org/abs/2412.15652)
Comments:
          16 pages, 6 figures

- **What's New**: 이 논문은 오류 기반의 데이터 효율적 튜닝 프레임워크를 제안하여, 특정 작업의 학습 샘플이 없이도 일반적인 대규모 멀티모달 모델을 새로운 작업에 효과적으로 적응시킬 수 있도록 합니다. 이를 위해, 학습 모델(학생 모델)이 기존 데이터셋에서 평가된 후, 더 강력한 모델(교사 모델)이 학생 모델의 오류를 식별하고 능력 갭을 분석합니다. 이 발견된 갭을 바탕으로 적절한 훈련 샘플을 기존 데이터셋에서 찾습니다.

- **Technical Details**: 논문에서는 학생 모델의 예측 결과를 분석하고, 해당 모델이 수행하지 못하는 작업을 교사 모델이 분석하여, 그로부터 향상된 학습 샘플을 추출하는 방식으로 동작합니다. 이 과정은 인간의 학습 방식에서 열리는 지식 갭을 찾아 채우는 과정을 모방한 것입니다. 아울러, MM-Bench, ScienceQA 등의 다양한 작업과 데이터셋에서 extensive 실험을 수행하였습니다.

- **Performance Highlights**: 제안된 프레임워크는 그동안의 데이터 선택 기준 및 전체 데이터셋을 사용하여 미세 조정한 모델에 비해 상당한 성능 향상을 보였습니다. 결과적으로, 다양한 훈련 데이터 크기에서 LMM의 평균 성능 향상을 7.01% 달성하였으며, 이는 기존 방법을 초월하는 성과로, 비용을 최소화하면서도 효율적으로 특정 작업에 적응할 수 있음을 시사합니다.



### Can Input Attributions Interpret the Inductive Reasoning Process Elicited in In-Context Learning? (https://arxiv.org/abs/2412.15628)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 입력 속성(input attribution, IA) 방법들이 대규모 언어 모델(large language models, LLMs)과 문맥 내 학습(in-context learning, ICL)에서 어떻게 작동하는지를 평가합니다. 특히 모호한 예제를 포함하는 설정에서 단일한 '미증유' 예제가 어떤 과제를 해결하는 데 어떤 영향을 미치는지를 규명하였습니다. 연구는 IA 방법들이 ICL 과정에서 사실상 유일한 예제를 어떻게 해석할 수 있을지에 대해 질문합니다.

- **Technical Details**: 연구는 두 가지 주요 측면에서 IA 방법의 효과성을 분석합니다. 첫째, 기존 IA 방법들이 현대 NLP 설정, 특히 LLM과 ICL 환경에서 여전히 적절하게 작동하는지 평가합니다. 둘째, 다수의 경량 IA 방법이 더 효과적이며, 모델이 존재하는 정보와 과제가 주어진 상황에서 제공하는 규칙의 추론 방식이 다름을 보여줍니다.

- **Performance Highlights**: 실험 결과, 간단한 IA 방법들이 통합된 방법들보다 성능이 더 우수한 경우가 많았습니다. 또한, IA 방법의 성능이 모델의 크기와 문제의 유형에 따라 현저하게 달라지는 경향이 있음을 확인하였습니다. 이러한 발견은 IA 방법들이 일반적으로 LLM의 훈련 과정에서의 해석능력이 제한적일 수 있음을 시사합니다.



### A Fusion Approach of Dependency Syntax and Sentiment Polarity for Feature Label Extraction in Commodity Reviews (https://arxiv.org/abs/2412.15610)
- **What's New**: 이 연구는 모바일폰, 컴퓨터, 화장품, 음식의 4개 카테고리에서 13,218개의 제품 리뷰를 분석하며, 의존 구문 분석과 감정 극성 분석을 통합하여 새로운 특징 레이블 추출 방법을 제안합니다. 기존의 추출 알고리즘의 낮은 강건성을 해결하고, 추출 정확도를 크게 향상시킵니다. 실험 결과는 이 방법이 0.7의 정확도와 0.8의 재현율 및 F-score를 달성했음을 보여줍니다. 그러나 매칭 사전에 대한 의존성과 추출된 특징 태그의 제한된 범위는 향후 연구에서 더 조사해야 할 과제로 남아 있습니다.

- **Technical Details**: 이 연구는 의존 구문 분석(dependency syntax) 이론을 바탕으로 제품 리뷰에서 '제품 특징-평가 용어' 쌍을 추출하는 것을 목표로 합니다. 특징 태그에는 제품 특징, 감정 정도, 평가 용어라는 세 가지 구성 요소가 포함됩니다. 해당 요소들은 의존 관계와 감정 분석을 통해 연결되며, 제품 특징과 평가 용어 간의 감정 극성에 따라 부정어 및 정도 부사를 식별하고 추출합니다. 이러한 새로운 접근법은 기존의 방법들이 지닌 모호성을 해결하는데 효과적입니다.

- **Performance Highlights**: 제안된 방법은 실험에서 0.7의 정확도와 함께 0.8의 재현율 및 F-score를 기록하며, 제품 리뷰에서 감정 단어와 평가 객체 식별의 정확성과 강건성을 향상시키는 데 성공하였습니다. 또한, 이 방법은 리뷰 내용의 정교한 분석이 가능하여 사용자에게 더 유용한 정보를 제공할 수 있습니다. 그러나 여전히 단어의 중복성 문제 및 특정 감정 단어의 낮은 극성으로 인해 발생하는 문제들은 해결해야 할 과제로 남아있습니다.



### Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks (https://arxiv.org/abs/2412.15605)
- **What's New**: 이 논문은 캐시 증강 생성(CAG, Cache-Augmented Generation)이라는 새로운 패러다임을 제안합니다. 기존의 검색 증강 생성(RAG, Retrieval-Augmented Generation) 방식에서 실시간 검색에 의존하지 않고, 모든 관련 자료를 사전에 로드하여 LLM의 확장된 맥락에서 처리합니다. CAG는 이를 통해 검색 지연을 제거하고, 검색 오류를 최소화하며, 시스템 복잡성을 줄입니다.

- **Technical Details**: CAG 프레임워크는 외부 지식 자원을 사전 로드하고, 키-값 캐시(KV Cache)를 미리 계산하여 불필요한 검색 단계를 제거합니다. 이 단계는 외부 지식을 문서 집합으로 저장하고, 이를 기반으로 모델이 효율적으로 응답할 수 있게 합니다. 이를 통해 지식 통합 작업의 처리 속도가 빨라지고, 응답 품질이 향상됩니다.

- **Performance Highlights**: 여러 기준을 비교 분석한 결과, CAG는 특히 관리 가능한 지식 기반을 가진 경우 RAG 시스템보다 더 높은 효율성과 정확성을 보였습니다. 실험에서는 SQuAD와 HotPotQA와 같은 잘 알려진 질문-답변 벤치마크를 사용하여 CAG 방법의 유효성을 평가하였으며, 이 과정에서 CAG가 전통적인 RAG에 비해 우수한 성과를 나타냈습니다.



### Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification (https://arxiv.org/abs/2412.15603)
Comments:
          11 pages, 3 figures, 11 tables

- **What's New**: 이번 논문에서는 다이얼로그 의도 분류의 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 특히, in-context learning과 동적 레이블 정제를 결합하여 비슷한 의미를 가진 의도 간의 혼란을 효과적으로 줄이는 방법을 소개합니다. 실험 결과, 이 방법이 여러 데이터셋에서 기존 방법에 비해 성능을 크게 향상시킨다는 것을 보여주었습니다.

- **Technical Details**: 제안된 방법에서는 세 가지 단계로 이루어진 동적 레이블 정제 절차를 사용합니다: 1) 의미적으로 유사한 예제를 검색하고, 2) 대형 언어 모델(LLM)을 통해 레이블을 정제하며, 3) 정제된 예제를 바탕으로 최종 분류를 수행합니다. 이 과정은 레이블과 관련 예제 간의 의미적 관계를 평가하여 보다 구체적인 레이블로 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 BANKING77과 같은 복잡한 도메인 데이터셋에서 특히 두드러진 성능 향상을 보여주었습니다. 다양한 모델 크기를 사용하여 확인한 결과, 모델 크기와 관계없이 성능 향상을 이끌어내는 데 효과적임을 입증했습니다. 이는 검사 쿼리 및 검색된 예제에 따라 레이블을 조정하는 동적 정제가 의도 분류 성능을 향상시키는 데 기여함을 나타냅니다.



### Template-Driven LLM-Paraphrased Framework for Tabular Math Word Problem Generation (https://arxiv.org/abs/2412.15594)
Comments:
          Accepted at AAAI 2025, extended version with appendix

- **What's New**: 본 논문에서는 Template-driven LLM-paraphrased (TeLL) 프레임워크를 제안하여 다양한 배경과 정확한 테이블, 질문, 답변, 솔루션을 가진 고품질 TMWP 샘플을 생성합니다. 기존의 방법들이 정확성이나 다양성에서 문제가 발생하는 것을 해결하기 위해, 이 프레임워크는 실제 샘플에서 템플릿을 추출하고 LLM을 활용하여 문제를 변형합니다. 또한, 수학적 추론 능력을 높이기 위해 설명적인 추론 단계를 솔루션에 추가하는 방안을 도입했습니다.

- **Technical Details**: TeLL 프레임워크는 추출된 템플릿을 바탕으로 초기 문제를 생성하며 이 과정에서 정확성을 유지합니다. LLM을 이용해 템플릿을 확장하고 문제를 패러프레이즈(paraphrase)하여 다양한 배경의 문제를 확보합니다. 이러한 생성된 샘플들은 요구 사항을 충족하도록 구조화되어 있으며, 다단계 추론능력을 강화하기 위해 보다 명확한 솔루션 설명을 수반합니다.

- **Performance Highlights**: TabMWP-TeLL 데이터셋은 기존 TabMWP 데이터셋의 질문 유형을 기반으로 고품질 TMWP 문제를 구성하며, 여러 LLM에 대한 실험을 통해 성능 개선을 입증하였습니다. 특히, 복잡한 문제에 대한 해결 성능이 크게 향상되었고, 간단한 문제에서도 성능 저하 없이 유지되는 등 실질적인 개선 사례를 보였습니다.



### NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization (https://arxiv.org/abs/2412.15588)
Comments:
          AAAI 2025 Project Page: this https URL

- **What's New**: NeSyCoCo는 기존의 neuro-symbolic 방법들의 한계를 극복하는 새로운 프레임워크입니다. 이 접근법은 대형 언어 모델(LLMs)을 활용하여 기호 표현(symbolic representations)을 생성하고 이를 미분 가능한 신경 계산(differentiable neural computations)으로 매핑합니다. 또한, 언어 입력을 종속 구조(dependency structures)로 보강함으로써 기호 표현과의 정렬을 강화하고, 다양한 언어 기반의 논리적 술어(logical predicates)를 신경 모듈(neural modules)과 연결합니다.

- **Technical Details**: NeSyCoCo는 자연어에서 보강된 종속 구조를 활용하여 기호 표현과의 의미적 정렬을 개선합니다. 이는 다양한 개념의 분산 표현(distributed representations)을 술어 표현으로 사용하여, 언어의 다양한 표현을 처리할 수 있게 합니다. 이 프레임워크는 정규화된 술어 점수(predicate scores)의 부드러운 조합(soft composition)을 사용해 기호적이며 미분 가능한 추론을 강화합니다.

- **Performance Highlights**: NeSyCoCo는 ReaSCAN 및 CLEVR-CoGenT 기준에서 최신 기술(state-of-the-art) 결과를 성취했으며, CLEVR-SYN 벤치마크에서 새로운 개념을 효과적으로 처리하는 강력한 성능을 보여줍니다. 이러한 성과는 제한된 술어 집합에 의존하지 않고 여러 문제에서 잘 작동할 수 있는 능력을 입증합니다.



### In-context Continual Learning Assisted by an External Continual Learner (https://arxiv.org/abs/2412.15563)
- **What's New**: 이 논문에서는 InCA라는 새로운 방법을 소개합니다. InCA는 External Continual Learner (ECL)와 In-context Learning (ICL)을 통합하여 Catastrophic Forgetting (CF) 문제를 피하면서도 Scalable Continual Learning (CL)을 가능하게 합니다. ECL은 각 테스트 인스턴스에 대해 적합한 클래스의 하위 집합을 미리 선택하여 ICL의 프롬프트 길이를 관리할 수 있도록 돕습니다.

- **Technical Details**: InCA는 Mahalanobis distance를 활용하여 입력 인스턴스의 태그 임베딩과 각 클래스 분포 간의 거리를 계산합니다. 이를 통해 가장 유사한 k개의 클래스를 선택하여 ICL 프롬프트를 생성합니다. 이 접근 방식은 Irrelevant information을 제거하고 입력 토큰 한계를 효과적으로 관리할 수 있게 해줍니다. ECL은 별도의 추가 학습 없이 클래스 평균과 공분산 행렬을 점진적으로 업데이트합니다.

- **Performance Highlights**: 실험 결과, InCA는 기존 CL 기준과 비교하여 상당한 성능 향상을 보여주었습니다. InCA는 특히 다양한 벤치마크 데이터 세트에서 효과적으로 Scalability와 Accuracy를 균형 있게 유지했습니다. 이러한 성능 개선은 ICL의 이점을 활용한 새로운 CL 패러다임의 가능성을 보여줍니다.



### NGQA: A Nutritional Graph Question Answering Benchmark for Personalized Health-aware Nutritional Reasoning (https://arxiv.org/abs/2412.15547)
- **What's New**: 이 논문은 개인 맞춤형 영양 건강을 위한 Nutritional Graph Question Answering (NGQA) 벤치마크를 발표했습니다. 이는 사용자 특정 건강 조건에 기초하여 음식이 건강한지 평가할 수 있는 최초의 데이터셋으로, 사용자의 의료 정보와 식단 행동을 통합하여 복잡한 영양 질문에 응답할 수 있도록 설계되었습니다. NGQA는 National Health and Nutrition Examination Survey (NHANES)와 Food and Nutrient Database for Dietary Studies (FNDDS)의 데이터를 활용하여 개별 사용자의 건강에 적합한 영양 성분을 명확히 설명합니다.

- **Technical Details**: NGQA 벤치마크는 세 가지 질문 복잡성 설정(희소, 표준, 복잡)을 포함합니다. 각 질문 유형은 세 가지 다운스트림 작업(이진 분류 – B, 다중 레이블 분류 – ML, 텍스트 생성 – TG)을 통해 평가되어 다양한 추론 측면을 탐구합니다. 연구는 LLM(backbone)와 기준 모델을 사용한 광범위한 실험을 통해 이 벤치마크가 기존 모델에 효과적으로 도전할 수 있음을 보여줍니다.

- **Performance Highlights**: NGQA는 개인 맞춤형 영양 건강 연구와 GraphQA 연구를 발전시키는 데 기여합니다. 본 연구는 사용자 의료 정보를 포함하는 최초의 벤치마크를 만들어 영양 질문 응답 작업에서 중요한 연구 격차를 해소합니다. 또한, NGQA는 GraphQA의 범위를 넓혀 보다 포괄적인 평가를 가능하게 하며, 전체 데이터 전처리에서 모델 평가에 이르는 완전한 코드베이스를 제공하여 새로운 모델 통합성을 위한 확장성을 지원합니다.



### MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering (https://arxiv.org/abs/2412.15540)
- **What's New**: 이 논문에서는 Temporal QA for RAG Evaluation 벤치마크인 TempRAGEval을 소개합니다. 이 벤치마크는 기존 데이터셋을 시스템적으로 재구성하여 시간적 변동성과 금 증거 레이블을 통합합니다. 특히, 시간적 논리를 요구하는 질문에 대한 기존의 검색 방법들이 어려움을 겪고 있다는 점을 밝힙니다. 제안된 무모델 검색 프레임워크(MRAG)는 질문을 주제와 시간 제약으로 분해하여 효과적인 증거 검색을 수행합니다.

- **Technical Details**: MRAG 프레임워크는 세 가지 모듈로 구성됩니다: (1) 질문 처리 모듈은 질문을 주제와 시간 제약으로 분리하고, (2) 검색 및 요약 모듈은 주요 내용에 따라 증거를 검색하고 LLM을 활용하여 요약합니다. (3) 의미-시간 하이브리드 순위 매기기 모듈은 각 증거 요약을 의미적 및 시간적 관련성을 기반으로 점수화합니다. 이러한 구조는 기존의 검색-재랭크 시스템에 비해 특정한 시너지 효과를 줍니다.

- **Performance Highlights**: 이 연구에서 제안한 MRAG 프레임워크는 TempRAGEval에서 기존의 모든 기본 검색 시스템보다 뛰어난 성능을 보였습니다. 특히, 최상위 답변 리콜이 9.3% 증가하고, 증거 리콜이 11% 증가했습니다. MRAG의 검색 성능 개선은 최종 질문 답변 정확도를 향상시키며, 퀄리티에 대한 심층 사례 연구도 이 프레임워크의 효과성을 지지합니다.



### XRAG: eXamining the Core -- Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15529)
- **What's New**: 이 논문에서는 Retrieval-augmented generation (RAG) 시스템의 성능 향상을 위한 XRAG라는 새로운 오픈 소스 모듈형 코드베이스를 소개합니다. XRAG는 RAG 모듈의 핵심 구성 요소를 철저하게 평가할 수 있도록 설계되어 있으며, 사전 검색(pre-retrieval), 검색(retrieval), 후 검색(post-retrieval), 생성(generation) 등 네 가지 핵심 단계로 시스템화되어 있습니다. 이는 RAG 시스템의 복잡성이 증가함에 따라 발생할 수 있는 잠재적 실패 지점을 식별하는 중요성을 강조합니다.

- **Technical Details**: XRAG는 다양한 재구성된 데이터셋을 통해 RAG 시스템의 성능을 체계적으로 분석하고, 이를 통해 효과성을 평가하는 포괄적인 벤치마크를 제공합니다. 실험적 방법론(experimental methodologies)과 진단 테스트 프로토콜(diagnostic testing protocols)을 통해 RAG 모듈의 고유한 실패 지점을 해체하고 분석합니다. 이 과정에서 각 단계의 구성 요소들이 어떻게 상호작용하며 성능을 저해할 수 있는지를 명확하게 파악할 수 있습니다.

- **Performance Highlights**: 이 연구는 RAG 시스템 내 핵심 고급 구성 요소의 성능을 철저히 평가하여 최적화 가능한 일반적인 실패 지점에 대한 통찰을 제공합니다. 제안된 맞춤형 솔루션(custom solutions)은 검증 과정을 강화하고 모듈 전반의 성능을 높일 수 있도록 설계되었습니다. 이러한 접근 방식은 더욱 정교해진 RAG 시스템의 구현을 지원하는 데 기여할 것입니다.



### HREF: Human Response-Guided Evaluation of Instruction Following in Language Models (https://arxiv.org/abs/2412.15524)
Comments:
          28 pages, 15 figures

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 지침 수행 능력을 평가하는 방법을 새롭게 혁신했습니다. 기존의 평가 방법들이 편향성을 드러내는 한편, 이 연구에서는 사람의 응답을 참조로 활용하여 자동 평가의 신뢰성을 높일 수 있음을 보여줍니다. 새로운 평가 벤치마크인 HREF(Human Response-Guided Evaluation of Instruction Following)를 개발하여 총 4,258개의 샘플을 포함하는 11개 작업 카테고리에 걸친 평가를 수행합니다.

- **Technical Details**: HREF는 다양한 자동 평가 기법을 조사하여 각 작업 카테고리에 대해 가장 신뢰할 수 있는 평가 방법을 선택하는 복합 평가 설정을 사용합니다. 이에 따라 HREF 데이터셋은 각 지침의 모델 응답과 인간의 기준 응답을 비교할 수 있도록 구축되었습니다. 이러한 접근 방식을 통해 연구팀은 LLM의 성능 평가에서 사람의 작성된 응답의 비중을 높였습니다.

- **Performance Highlights**: HREF 평가 벤치마크를 통해 LLM과 인간 심사자 간의 일치도를 최대 3.2% 향상시킬 수 있었습니다. 연구진은 설계 선택의 영향, 예를 들어 평가 세트의 크기, 심사 모델 및 기준 모델 등을 조사하고, 이를 기반으로 LLM의 실적을 평가하는 라이브 리더보드를 운영합니다. 이는 LLM의 개별 작업 성능을 강조하며 기존의 평가 시스템에서 나타나는 오염 문제에서 자유롭습니다.



### ADEQA: A Question Answer based approach for joint ADE-Suspect Extraction using Sequence-To-Sequence Transformers (https://arxiv.org/abs/2412.15510)
- **What's New**: 본 논문은 ADE(Adverse Drug Event)의 조기 식별을 위해 새로운 접근 방식을 제안합니다. ADEQA라는 질문-응답(QA) 방식의 모델이 마련되어, 비정형 데이터 소스로부터 ADE와 관련 약물을 추출하는 데 집중하고 있습니다. 이는 기존의 QA 모델과 달리 자연어 생성(NLG) 기반 모델을 사용하여 토큰 레벨의 라벨링 필요성을 줄임으로써 신약 시장 도입 시 신속한 조치를 가능하게 합니다.

- **Technical Details**: ADEQA 모델은 준수 감독(quasi supervised) 라벨 데이터와 시퀀스 투 시퀀스(transformers) 기술을 이용하여 ADE, 관련 약물 및 이들 간의 관계를 추출하는 데 도움을 줍니다. 이 방법론은 복잡한 언어적 관계를 효과적으로 처리하여, 비라벨 대량 데이터 부족 문제를 해결하는 데 기여합니다. 추가적으로, 향상된 결과를 도출하기 위해 링크를 통한 데이터 연결 체계를 개선하고 있습니다.

- **Performance Highlights**: 공식 ADE 데이터셋을 사용한 실험에서는 ADEQA 모델이 94%의 F1 점수를 기록하며 최첨단 결과를 달성했습니다. 이는 ADE와 해당 약물 간의 관계를 확립하는 데 있어 탁월한 성능을 보임을 나타냅니다. 이러한 성과는 새로운 약물이 시장에 도입될 때 잠재적 위험 요소를 사전에 식별하는 데 중요한 기여를 할 것으로 기대됩니다.



### Mitigating Social Bias in Large Language Models: A Multi-Objective Approach within a Multi-Agent Framework (https://arxiv.org/abs/2412.15504)
Comments:
          This work has been accepted at The 39th Annual AAAI Conference on Artificial Intelligence (AAAI-2025)

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)에서 사회적 편향을 감소시키기 위해 다중 목표 접근 방식인 MOMA(multi-objective approach within a multi-agent framework)를 제안합니다. 기존의 방법들은 성능 저하를 초래하면서도 윤리적으로 행동하도록 LLM을 유도하는 데 중점을 두었으나, MOMA는 성능 저하를 최소화하면서 편향을 효과적으로 줄여내는 혁신적인 접근법입니다. 다양한 에이전트를 통해 입력 질문의 편향 관련 내용을 인과적으로 수정함으로써 보다 공정한 결과를 도출합니다.

- **Technical Details**: MOMA는 여러 에이전트를 활용하여 편향이 포함된 콘텐츠에 개입하는 인과적 개입(causal interventions)을 수행합니다. 이는 해당 콘텐츠와 관련된 답변 간의 단축 연결(shortcut connection)을 끊어버리는 방식으로 작동합니다. 결과적으로, MOMA는 기존의 단일 에이전트 설정에 비해 성능 저하를 최소화하면서도 편향 점수를 최대 87.7%까지 감소시킵니다.

- **Performance Highlights**: 실험 결과, MOMA는 BBQ 데이터셋에서 최대 6.8%의 성능 저하와 함께 편향 점수를 유의미하게 감소시킴을 보여주었으며, StereoSet 데이터셋의 다중 목표 메트릭인 icat 수치는 최대 58.1% 개선되었습니다. 이는 MOMA가 기존의 방법들보다 더 효과적으로 편향을 줄이는 동시에 높은 성능을 유지할 수 있다는 것을 시사합니다.



### Humanlike Cognitive Patterns as Emergent Phenomena in Large Language Models (https://arxiv.org/abs/2412.15501)
- **What's New**: 본 연구는 대형 언어 모델(LLM)에서 나타나는 새로운 패턴을 종합적으로 검토하며, 최근 인공지능(AI) 및 심리학에서의 연구 동향을 반영합니다. LLM은 인간의 인지 및 의사결정을 모방하거나 영향을 미치는지에 대한 논의가 활발하게 진행되고 있습니다. 이러한 연구는 LLM이 추론 및 창의성을 포함한 복잡한 사고 능력을 얼마나 잘 나타내는지를 평가하고 있습니다.

- **Technical Details**: 논문에서는 독립적인 심리학적 실험 결과를 바탕으로 의사결정 편향, 추론 및 창의성의 세 가지 인지 영역에서 LLM의 성능을 검토합니다. 의사결정에서 LLM이 몇 가지 인간의 편향을 보여주지만, 모든 인간 편향이 표현되는 것은 아니며, 이러한 점은 인지적 패턴이 완전히 일치하지 않음을 나타냅니다. 또한, GPT-4와 같은 고급 LLM은 인간의 심사숙고적 사고에 필적하는 추론을 수행하는 반면, 더 작은 모델은 그렇지 않음을 강조하고 있습니다.

- **Performance Highlights**: LLM은 이야기 만들기와 같은 언어 기반의 창의적 작업에서 뛰어난 성과를 보이는 반면, 실제 맥락이 필요한 다양한 사고 작업에서는 어려움을 겪습니다. 하지만 LLM은 인간-기계 문제 해결 과정에서 창의성을 보강할 수 있는 잠재력을 가진 협업자 역할을 할 수 있다는 점에서 의미가 있습니다. 마지막으로, 메모리, 주의집중 및 오픈 소스 모델 개발과 같은 미래의 연구 방향에 대한 지침을 제공합니다.



### The First Multilingual Model For The Detection of Suicide Texts (https://arxiv.org/abs/2412.15498)
Comments:
          SUMEval-2: The 2nd Workshop on Scaling Up Multilingual & Multi-Cultural Evaluation at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 연구에서는 자살 사고를 탐지하기 위해 다국어 모델을 제안합니다. mBERT, XML-R, mT5와 같은 transformer 아키텍처를 활용하여 스페인어, 영어, 독일어, 카탈루냐어, 포르투갈어, 이탈리아어 등 6개 언어의 자살 관련 게시물에서 텍스트를 인식합니다. SeamlessM4T를 통해 스페인어 자살 트윗 데이터셋을 다섯 가지 다른 언어로 번역하였고, 이를 기반으로 모델을 미세조정하여 다국어 데이터로 평가함으로써 새로운 방향성을 제시합니다.

- **Technical Details**: 이 연구에서는 자연어 처리(NLP) 및 딥러닝(DL) 기법을 사용하여 자살 사고를 자동으로 감지했습니다. 후보 모델인 mBERT, XML-R, mT5는 자살 텍스트를 다국어로 탐지하고, 특히 mT5가 F1 점수 85% 이상으로 우수한 성능을 기록했습니다. 언어 간 전이 학습(cross-lingual transfer learning)을 통해 다양한 문화적, 언어적 맥락에서 자살 위험을 효과적으로 식별할 수 있습니다.

- **Performance Highlights**: 모델의 평가 결과 mT5가 가장 높은 성능을 보였으며, 다국어에서의 의사 표현을 정확하게 이해할 수 있는 능력이 확인되었습니다. 번역 품질 또한 높은 퍼플렉시티(perplexity) 점수로 보장되어, 다양한 언어로 자살 사고를 탐지하는 데 있어 실질적인 기여가 예상됩니다. 이 연구는 자살 위험 식별 자동화 도구 개발의 필요성을 다시 한번 강조하며, 향후 인간 참여 평가(human-in-the-loop) 시스템에 대한 윤리적 고려 사항을 제시합니다.



### Lexicography Saves Lives (LSL): Automatically Translating Suicide-Related Languag (https://arxiv.org/abs/2412.15497)
Comments:
          The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 최근 연구들은 자살의 위험, 의도, 아이디어를 확인하고 예측하는 데 중점을 두고 있습니다. 하지만 기존의 연구들은 주로 영어와 서구 문화에 국한되어 있어, 자살이 전 세계적인 문제라는 점을 간과하고 있습니다. 본 논문은 'Lexicography Saves Lives Project'를 소개하며, 자살과 관련된 자료를 여러 언어로 번역하고, 윤리적 고려사항을 명시해 자살 예방의 기반을 강화하고자 합니다.

- **Technical Details**: 이 연구는 자살 아이디어와 관련된 기존 사전을 200개 언어로 번역하고, 번역의 품질을 이해하기 위해 인간 평가를 수행합니다. 또한, 7개의 변수를 통해 번역의 질을 평가하며, 이를 통해 사전의 품질을 수치적으로 나타냅니다. 이 과정은 기계 번역과 자연어 처리(NLP)에 의한 자동 번역의 한계를 극복하기 위한 방법론을 제시하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 저자들은 번역된 사전의 품질을 향상시키기 위한 윤리적 가이드라인과 완화 전략을 제시했습니다. 또한, 공공 웹사이트를 구축하여 번역된 자료를 공개하고, 커뮤니티 참여를 장려합니다. 이 프로젝트는 정신 건강과 자살 예방을 위한 자원 개발 및 실천을 향상시키기 위한 첫 걸음으로서, 국제적인 대화의 필요성을 강조합니다.



### TL-Training: A Task-Feature-Based Framework for Training Large Language Models in Tool Us (https://arxiv.org/abs/2412.15495)
- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)의 도구 사용 성능 향상을 위한 TL-Training 프레임워크를 제안합니다. 이 프레임워크는 비효율적인 훈련 데이터의 영향을 줄이고, SFT(감독 세밀 조정) 동안 핵심 토큰의 중요도를 동적으로 조정하는 방법을 포함합니다. 또한, 에러 유형에 최적화된 보상 메커니즘이 통합되어 있습니다.

- **Technical Details**: TL-Training 프레임워크는 훈련 데이터에서 잘못된 상호 작용 경로를 식별하여 이를 그래디언트 업데이트에서 제외함으로써 부정적인 영향을 완화합니다. 이를 통해 SFT 동안 핵심 토큰을 우선시하는 적응형 가중치 조정을 실시하고, 도구 피드백을 강화 학습의 보상 메커니즘에 통합합니다. 이러한 방식은 강화 학습의 PPO(근접 정책 최적화) 알고리즘을 이용해 최적화됩니다.

- **Performance Highlights**: TL-Training을 사용하여 CodeLLaMA-2-7B 모델을 훈련시켰습니다. 이 모델은 1,217개의 훈련 데이터 포인트로 데이터를 최소화해도, 도구 사용 성능에서 선도적인 오픈 및 클로즈드 소스 LLM과 동등하거나 그 이상을 나타냈습니다. 또한, TL-Training은 노이즈 환경에서 강건성을 향상시키고 일반적인 작업 성능을 개선하는 데 기여하여, 도구 사용 훈련을 위한 확장 가능하고 효율적인 패러다임을 제공합니다.



### Multi-LLM Text Summarization (https://arxiv.org/abs/2412.15487)
- **What's New**: 이번 연구에서는 Multi-LLM 요약 프레임워크를 제안하고, 중앙 집중식 및 분산식 다중 LLM 전략 두 가지를 조사하였습니다. 각각의 대화 단계에서 생성(generation)과 평가(evaluation)의 두 가지 내용이 포함되어 있으며, 중앙 집중식 요약 방식에서는 단일 LLM이 요약의 질을 평가하고 최상의 요약을 선택합니다. 이 연구의 결과는 다중 LLM 접근방식이 단일 LLM을 사용할 때보다 최대 3배 높은 성능을 보였음을 나타냅니다.

- **Technical Details**: 이 다중 LLM 요약 프레임워크는 긴 문서를 처리하기 위해 여러 LLM의 생성 및 평가 단계를 배포하는 방식으로 요약 품질을 향상시킵니다. 중앙 집중식 및 분산식 두 가지 상호작용 방식이 존재하며, 이는 LLM 간의 협업과 요약 개선을 유도합니다. 특히, 원본 문서를 청크(chunk) 단위로 나누어 독립적으로 요약한 후, 중간 결과를 종합하여 최종 고품질 요약을 생성합니다.

- **Performance Highlights**: 실험 결과, 다중 LLM 요약 접근 방식은 단일 LLM 모델을 사용할 때에 비해 더욱 우수한 성능을 보여주었습니다. 연구진은 다중 LLM의 조합, 프롬프트 방법, LLM의 수 등 여러 변수가 요약 품질에 미치는 영향을 분석했습니다. 이러한 결과는 다중 LLM이 복잡한 문서 요약 과정에서 정보 교환과 협력적 합성을 통해 더 나은 결과를 제공할 수 있음을 시사합니다.



### Continual Learning Using Only Large Language Model Prompting (https://arxiv.org/abs/2412.15479)
Comments:
          To Appear in COLING-2025 (short paper)

- **What's New**: 새로운 CL(Continual Learning) 패러다임인 CLOB(Continual Learning Over Black-box LLMs)가 제안되었습니다. 이는 대형 언어 모델(LLM)을 블랙 박스로 간주하며, 오직 언어 프롬프트를 통해 점진적으로 학습하는 방식을 채택합니다. CLOB는 특정 파라미터를 조정하지 않으며, API를 통해 접근할 수 있는 LLM에 적합합니다. 또한, CIS(Incremental Summarization)를 기반으로 한 새로운 CL 기법이 도입되었습니다.

- **Technical Details**: CLOB에서 사용자 는 오직 언어 프롬프트와 몇 개의 사례 예시로 LLM과 상호작용하며 새로운 작업을 학습합니다. 기존의 파라미터 업데이트로 인한 catastrophic forgetting이 사라지지만, prompt 기반의 forgetting이 새롭게 등장하게 됩니다. 이 연구는 클래스-증가 학습(class-incremental learning) 설정에서 수행되며, 기존 작업 데이터에 의존하지 않고 새로운 작업 데이터로 지속적으로 학습하도록 구성됩니다.

- **Performance Highlights**: CIS 방법은 LLM의 요약 기능을 활용하여 각 클래스에 대한 지식을 요약이라는 형태로 캡슐화합니다. 새 작업의 데이터가 도착할 때마다 이 요약을 점진적으로 학습하고 업데이트하며, 이를 통해 LLM의 입력 길이 제한 문제를 효과적으로 해결합니다. 실험 결과에서 CIS는 기존 방법들보다 상당한 정확도를 보였으며, 점진적인 학습 가능성을 입증했습니다.



### A Review of the Marathi Natural Language Processing (https://arxiv.org/abs/2412.15471)
- **What's New**: 이 논문은 Marathi와 같은 인도 언어에서 자연어 처리(NLP) 연구의 발전을 다루고 있습니다. 특히, 최근 10년 동안 인도의 22개 예정 언어를 위한 언어 자원 개발에 대한 주요 노력이 있었음을 강조합니다. 또한, Marathi 언어의 형태학적(grammatical) 특성으로 인해 NLP 작업이 더욱 복잡해졌다는 점을 지적합니다.

- **Technical Details**: Marathi의 복잡한 스크립트와 언어 구성 요소로 인해 NLP 연구와 관련된 여러 도전 과제가 있었습니다. 또한 공개적으로 이용 가능한 자원 부족, 고품질 데이터셋 및 벤치마크의 결여가 문제로 지적되었습니다. 그러나 2000년대 초반 이후의 신경망 기반 모델의 발전 덕분에 NLP 연구가 점차 접근 가능해졌습니다.

- **Performance Highlights**: 이 논문은 Marathi NLP 작업과 관련된 최신 자원과 도구를 소개합니다. 또한, 이러한 도구와 기술이 인도 언어 연구 커뮤니티에 미치는 영향을 설명합니다. Marathi 관련 NLP 과제가 어떻게 진화해왔는지를 조망하며, 효과적인 연구 기반의 구축을 위한 다양한 도구의 발전을 다루고 있습니다.



### Northeastern Uni at Multilingual Counterspeech Generation: Enhancing Counter Speech Generation with LLM Alignment through Direct Preference Optimization (https://arxiv.org/abs/2412.15453)
Comments:
          10 pages, 6 tables, 1 figure, The First Workshop on Multilingual Counterspeech Generation (MCG) at The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 저자들은 기존의 자동 반응 생성 방식의 한계를 극복하기 위해 새로운 방법론을 제안합니다. 특히, 다양한 언어적 맥락에서의 반응 생성을 개선하기 위한 연구를 진행하며, Supervised Fine-Tuning (SFT)과 Direct Preference Optimization (DPO)를 활용합니다. 이 방법론은 LLM (Large Language Models) 출력을 인간의 선호와 align하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 모델은 DPO를 통해 생성된 반응이 맥락적으로 적합하고 언어적으로 조정 가능하도록 합니다. 또한, 지식 기반을 포함하여 생성된 반응의 사실적 정확성과 관련성을 향상시킵니다. SFT 기법과 비교했을 때, DPO-aligned 모델이 CS (Counter-Speech) 벤치마크에서 월등한 성과를 보이며, 여러 언어에 걸쳐 효과적으로 스케일링되는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, DPO 방법이 기존 SFT 기반 모델을 명백히 초과 달성했음을 입증했습니다. 이 연구는 선호 기반의 align 기법이 다양한 언어 환경에서 CS 생성 기술을 발전시키는 데 중요한 가능성을 지니고 있음을 강조합니다. 모델 군집화와 align은 영어로 이루어지며, 이후 Basque, Italian, Spanish와 같은 다른 언어에서의 메트릭 리포팅에 동일한 모델이 사용됩니다.



### Fietje: An open, efficient LLM for Dutch (https://arxiv.org/abs/2412.15450)
- **What's New**: 이 논문은 네덜란드어를 위한 소형 언어 모델(Small Language Models, SLMs)인 Fietje를 소개합니다. Fietje는 27억 개의 매개변수를 가진 영어 중심 모델인 Phi 2를 기반으로 하여 네덜란드어에 특화되었으며, 출시와 함께 경쟁력 있는 성능을 보여주었습니다. 이 모델은 완전히 오픈 소스이며, 투명성과 재현성을 중시하여 모델 가중치, 데이터셋, 훈련 및 평가 코드를 모두 공개했습니다.

- **Technical Details**: Fietje는 280억 개의 네덜란드어 토큰을 사용하여 훈련되었으며, reasoning, sentiment analysis, world knowledge, linguistic acceptability 및 word sense disambiguation과 같은 다양한 벤치마크에서 평가되었습니다. 모델의 성능은 다른 모델들과 비교되었고, 발견된 결과는 작지만 다국어 모델이 네덜란드어 전용 모델을 초월하는 등 언어 모델의 변화하는 환경을 잘 보여줍니다. Fietje는 다국어 처리를 위한 초기 단계에 불과하며, 앞으로도 더욱 개선될 여지가 있습니다.

- **Performance Highlights**: Fietje는 출판 당시 예상 이상의 성능을 보여주었고, 그 크기와 비교했을 때 때때로 두 배의 크기를 가진 모델들과 경쟁력을 보였습니다. 이 모델은 특히 네덜란드어 처리에 있어 작고 강력한 모델들이 점점 더 유능해지고 있음을 시사합니다. 앞으로의 연구는 이러한 다국어 모델들을 더욱 발전시키고, 언어 기술 사용자의 접근성을 높이는 데 기여할 것으로 기대됩니다.



### SKETCH: Structured Knowledge Enhanced Text Comprehension for Holistic Retrieva (https://arxiv.org/abs/2412.15443)
Comments:
          16 pages, 8 figures, Workshop on Generative AI and Knowledge Graphs (GenAIK) at The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 개선을 위한 새로운 방법론인 SKETCH를 소개합니다. SKETCH는 의미 기반 텍스트 검색(semantic text retrieval)과 지식 그래프(knowledge graphs)를 통합하여, 구조화된 데이터와 비구조화된 데이터를 결합함으로써 더 포괄적인 이해를 가능하게 합니다. 이러한 방식은 전통적인 방법에 비해 정보 검색 성능을 크게 향상시킵니다.

- **Technical Details**: SKETCH는 대규모 데이터셋에서 정보를 효율적으로 처리하고 검색하는 데 집중하여, 맥락에 대한 포괄적인 이해를 유지하는 방식으로 작동합니다. 이 방법은 QuALITY, QASPER, NarrativeQA, Italian Cuisine의 네 가지 다양한 데이터셋에서 평가되었으며, RAGAS 메트릭(answer_relevancy, faithfulness, context_precision, context_recall)에서 기존의 기준 접근법을 초월한 성과를 보였습니다. 구체적으로, Italian Cuisine 데이터셋에서 SKETCH는 0.94의 답변 관련성과 0.99의 맥락 정밀도를 달성하여, 평가된 모든 메트릭 중에서 최고의 성능을 기록했습니다.

- **Performance Highlights**: SKETCH는 전통적인 RAG 시스템에 비해 우수한 검색 성능과 보다 높은 맥락 무결성을 유지하며, 정확하고 맥락적으로 관련된 응답을 제공합니다. 이 연구 결과는 SKETCH가 향후 정보 검색 시스템의 새로운 기준을 설정할 수 있는 가능성을 나타냅니다. 따라서 SKETCH는 대규모 언어 모델의 환각(hallucination) 문제를 줄이는 데 기여하는 매우 효과적인 도구로 평가됩니다.



### Systematic Evaluation of Long-Context LLMs on Financial Concepts (https://arxiv.org/abs/2412.15386)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이 논문은 Long-context large language models (LC LLMs)의 신뢰성을 향상시키기 위한 연구로, 특히 긴 입력 문서를 처리하는 실제 작업에 대한 평가를 다루고 있습니다. 연구팀은 GPT-4 모델의 다양한 성능을 평가하면서, 컨텍스트 길이, 작업 난이도, 주요 정보 위치와 같은 요소들이 성능에 미치는 영향을 분석했습니다.

- **Technical Details**: 실제 금융 뉴스 데이터셋을 생성하여 다양한 난이도의 연속적인 작업을 수행하도록 하였고, 그 과정에서 LC LLMs의 성능이 길어진 컨텍스트 길이에서 어떻게 변화하는지를 살펴보았습니다. 결과적으로 간단한 작업에서도 성능 저하가 발생하며, 컨텍스트 길이가 늘어날수록 지침을 따르는 데 신뢰성을 잃고 degenerative outputs가 발생하는 현상을 관찰했습니다.

- **Performance Highlights**: 작업 지시가 컨텍스트 창 내에서 배치되는 위치나 약간의 Markdown formatting에도 여전히 민감한 반응을 보였으며, 이는 LC LLMs의 취약성을 강조했습니다. 따라서, F1 점수와 같은 포괄적인 메트릭을 활용하여 LC LLMs의 평가를 더 철저하게 수행할 것을 제안합니다.



### Automatic Extraction of Metaphoric Analogies from Literary Texts: Task Formulation, Dataset Construction, and Evaluation (https://arxiv.org/abs/2412.15375)
Comments:
          Accepted to COLING 2025, long paper

- **What's New**: 이 연구는 문학 텍스트에서 메타포(메타포)와 유추(analogy)를 추출하는 새로운 데이터셋을 개발하고, 최근 대형 언어 모델(LLMs)의 구성 능력을 비교합니다. 이전의 연구들과 달리, 저자들은 단일 쌍의 개념 및 비유적인 개념을 포함하여 메타포의 복잡성을 탐구하고 있습니다. 모델이 암시적으로 제안된 요소를 생성할 수 있는지 평가하는 점도 새롭습니다. 이러한 접근 방식은 텍스트에서 유추와 메타포를 자동으로 추출할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 저자들은 LLMs가 비유적 유추를 식별하는 능력을 평가하기 위해, 자연어 처리에서 의미론적 처리(STT)를 향상시키는 방법으로 비유적 매핑(mappings)을 추출합니다. 그들은 소스 도메인과 목표 도메인 간의 관계를 식별하고, 메타포가 포함된 텍스트의 핵심 단어를 추출하여 모델에 의해 생성된 암시적 개념을 활용합니다. 특히, 이 연구는 문학적인 맥락에서 4개의 요소를 가진 비유적 유추를 탐구하는 새로운 과제를 설정합니다.

- **Performance Highlights**: 실험에서 LLMs는 비유적 유추를 비교적 정확하게 생성하며, 이로 인해 모델의 미래 연구 및 적용 가능성이 높아졌습니다. 연구는 심리학, 언어학 및 인지 과학 같은 다양한 분야에서 활용될 수 있는 광범위한 지식 기반을 구축할 수 있는 가능성을 보여줍니다. 이 작업은 질문 답변(Question Answering)이나 기계 번역(Machine Translation)과 같은 자연어 처리(NLP) 하위 작업의 성능을 크게 향상시킬 것으로 기대합니다.



### Decade of Natural Language Processing in Chronic Pain: A Systematic Review (https://arxiv.org/abs/2412.15360)
- **What's New**: 최근 자연어 처리(NLP)와 공공 건강이 만나는 지점에서 만성 통증과 관련된 텍스트 데이터셋을 연구하는 혁신적인 길이 열리고 있습니다. 이 리뷰는 NLP 기반의 만성 통증 연구에 초점을 맞춘 다양한 연구를 조사하고 있으며, 2014년부터 2024년까지 발표된 132개의 논문 중 최종적으로 26개의 연구가 포함되었습니다. 따라서 만성 통증과 관련된 지식의 통합과 연구 방향을 제시하는 데 도움을 주고자 합니다.

- **Technical Details**: 이 리뷰는 PRISMA(Preferred Reporting Items for Systematic Reviews and Meta-analysis) 가이드라인에 따라 수행되었습니다. 연구는 만성 통증과 관련된 NLP의 설계, 개발 및 적용에 대한 연구 질문에 답하는 영어로 작성된 논문에 한정되었습니다. 확보된 데이터는 PubMed, Web of Science, IEEE Xplore, Scopus, ACL Anthology에서 수집되었으며, 데이터 분석은 Covidence 소프트웨어를 통해 진행되었습니다.

- **Performance Highlights**: 26개의 최종 연구의 결과는 NLP 기술이 만성 통증 연구에서 중요한 문제를 해결하는 데 상당한 가능성을 보여준다는 점을 강조합니다. 연구들은 RoBERTa 및 BERT와 같은 고급 방법의 사용을 보여주며, F1 점수 0.8 이상의 높은 성능을 달성하였습니다. 그러나 데이터셋의 다양성이 부족하고, 샘플 크기와 대표성이 부족한 여전히 해결해야 할 도전 과제가 존재함을 밝혀냈습니다.



### Eliciting Causal Abilities in Large Language Models for Reasoning Tasks (https://arxiv.org/abs/2412.15314)
- **What's New**: 이 연구는 LLMs(대규모 언어 모델)의 추론 능력을 향상시키기 위해 신 causal inference(인과 추론) 기법을 적용하는 새로운 접근 방식을 제안합니다. Self-Causal Instruction Enhancement(SCIE) 메소드는 LLM이 높은 품질의 관찰 데이터를 생성하도록 하여, 이를 바탕으로 인과적 효과를 추정하고 최적화된 지침을 생성하게 합니다. 이 방법은 기존의 prompt optimization(프롬프트 최적화) 방법의 비용 문제와 해석 가능성 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: SCIE 메소드는 프롬프트 지침을 치료(treatment)로 간주하고, 텍스트 기능을 사용하여 자연어를 처리하며, 지침과 하위 작업 간의 인과 관계를 설정합니다. 연구는 인과 추론을 위해 필요한 세 가지 가정인 ignorability(무관성), positivity(양성), consistency(일관성)를 설명합니다. LLM의 추론 능력을 향상시키기 위해 우리는 프롬프트의 인과 효과를 최대화할 수 있는 지침을 식별하는 작업을 수행합니다.

- **Performance Highlights**: SCIE 방식에 대한 광범위한 실험 결과, 이 방법이 LLM의 추론 성능을 향상시키는 동시에 프롬프트의 훈련 비용을 줄이는 데 효과적임을 보여줍니다. 특히, 새로운 하위 작업에서 OR(Object-Relational) 원칙에 따라 재사용 가능한 인과 관계를 통해 성능 개선이 나타났습니다. SCIE는 인과 관계를 메타 템플릿으로 활용하여 프롬프트 생성을 효율적으로 안내합니다.



### Conceptual In-Context Learning and Chain of Concepts: Solving Complex Conceptual Problems Using Large Language Models (https://arxiv.org/abs/2412.15309)
Comments:
          Accepted to 2025 IEEE Symposium on Computational Intelligence in Natural Language Processing and Social Media

- **What's New**: 이 논문은 복잡한 개념 문제(complex conceptual problems)를 해결하기 위한 대형 언어 모델(Large Language Models, LLMs)의 얕은 커스터마이징 방법(shallow customization methods, SCMs)을 탐구합니다. 특히 새로운 알고리즘인 개념적 맥락 학습(Conceptual In-Context Learning, C-ICL)과 개념 체인(Chain of Concepts, CoC)을 제안하며, 이들이 LLMs에 개념 정보를 추가하여 문제 해결 능력을 향상시키는 방법을 다룹니다.

- **Technical Details**: 저자는 복잡한 개념 문제를 해결하기 위해 LLMs의 기존 SCM들이 효과적이지 않다는 것을 입증하고, 새로운 두 가지 SCM 알고리즘을 통해 LLMs를 개념 정보로 보강하는 방법을 제안합니다. 이 과정에서 모델의 응답 정확도, 생성 시간, 비용 등의 여러 측면을 평가합니다. 제안된 알고리즘은 기존의 인기 있는 SCM보다 30% 이상의 정확도를 보이며, 모델의 사라짐(hallucination) 현상을 줄이는 데도 기여합니다.

- **Performance Highlights**: C-ICL과 CoC 알고리즘이 제공하는 응답의 정확성과 투명성이 두드러집니다. 평가 결과, 새로운 SCM을 적용한 LLM이 기존의 SCM보다 더 나은 성능을 보였고, 문제 해결 능력의 향상이 관찰되었습니다. 이는 LLM이 복잡한 개념 문제를 해결하는 데 있어 더 신뢰할 수 있는 도구가 될 수 있음을 암시합니다.



### ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in Vietnames (https://arxiv.org/abs/2412.15308)
Comments:
          Accepted at AAAI'2025 Main Conference

- **What's New**: 이 논문에서는 베트남어 사실 확인을 위한 최초의 공개 데이터셋인 ViFactCheck를 소개합니다. 이 데이터셋은 12개의 다양한 주제를 다루는 총 7,232개의 인간 주석된 주장-증거 쌍으로 구성되어 있으며, 이는 베트남의 신뢰할 수 있는 온라인 뉴스에서 수집되었습니다. 또한 이 데이터셋은 고품질과 신뢰성을 보장하는 면밀한 주석 과정을 거쳤고, Fleiss Kappa 상호 주석자 동의 점수는 0.83에 달합니다.

- **Technical Details**: ViFactCheck 데이터셋은 9개의 라이센스가 있는 인기 있는 베트남 온라인 신문에서 수집된 기사를 기반으로 구축되었습니다. 이 데이터셋은 데이터 수집, 주석 처리 및 주석 검증의 세 가지 단계로 나뉘어 각 전문가의 엄격한 모니터링을 받습니다. 또한, 이 연구에서는 최첨단의 사전 학습된 언어 모델과 대형 언어 모델을 활용하여 사실 확인을 위한 다양한 기법을 평가하였습니다.

- **Performance Highlights**: Gemma 모델은 89.90%의 매크로 F1 점수를 기록하여 우수한 효과성을 입증하며 사실 확인 벤치마크에 대한 새로운 기준을 설정하였습니다. 이러한 결과는 Gemma가 베트남에서 사실을 정확하게 식별하고 확인하는 데 있어 뛰어난 능력을 보여줍니다. 또한 ViFactCheck 데이터셋, 모델 체크포인트, 사실 확인 파이프라인 및 소스 코드는 GitHub를 통해 무료로 제공되어 향후 연구를 촉진하고 정보의 정확성을 높이는 데 기여할 것입니다.



### Self-Evolution Knowledge Distillation for LLM-based Machine Translation (https://arxiv.org/abs/2412.15303)
Comments:
          COLING 2025

- **What's New**: 이 논문에서는 Self-Evolution 지식 증류(Self-Evolution KD)라는 새로운 기법을 제안합니다. 기존의 지식 증류 방식이 토큰 간의 출력을 무차별적으로 최소화하는 데 초점을 맞추었다면, 이 방법은 토큰의 학습 난이도에 따라 사전 지식(prior knowledge)의 비율을 조정하여 교사 모델의 잠재력을 최대한 활용할 수 있도록 합니다. 이를 통해 지식 전이가 개선되는 효과가 나타났습니다.

- **Technical Details**: Self-Evolution KD는 두 가지 단계로 이루어져 있습니다. 첫 번째 단계에서는 Kullback-Leibler(KL) 다이버전스를 사용하여 학생 모델의 학습 난이도를 정량화하고, 두 번째 단계에서는 난이도가 높은 토큰에 대해 사전 지식을 통합하여 학습 과정을 가속화합니다. 이러한 접근 방식은 교사 모델로부터 보다 효과적인 지식 전이를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, Self-Evolution KD는 WMT22 테스트 세트의 네 가지 번역 방향에서 평균 1.4 SacreBLEU 포인트의 성능 개선을 보였습니다. 이는 교사 모델로부터의 지식 전이가 더욱 효과적으로 이루어졌음을 시사하며, 새로운 증류 전략이 학생 모델의 성능을 극대화하는 데 기여했음을 확인할 수 있습니다.



### LAMA-UT: Language Agnostic Multilingual ASR through Orthography Unification and Language-Specific Transliteration (https://arxiv.org/abs/2412.15299)
- **What's New**: 이 논문에서는 언어에 구애받지 않는 다국어 자동 음성 인식(ASR) 시스템인 LAMA-UT를 소개합니다. 기존의 언어별 모듈 없이, 극소량의 데이터로도 최첨단 모델과 비슷한 성능을 낼 수 있는 혁신적인 파이프라인을 개발했습니다. 이 시스템은 두 가지 주요 단계를 포함하여 언어의 공통된 음운적 특징을 포착하는 동시에 다양한 언어의 정서적 특성을 통합합니다. LAMA-UT는 100개 이상의 언어에서 인식 성능을 검증하며, 특히 저자원 언어에서 우수한 성과를 보입니다.

- **Technical Details**: LAMA-UT의 구성은 두 가지 주요 단계로 나뉩니다: 보편적인 전사 생성과 언어별 음역(transliteration)입니다. 첫 번째 단계에서는 보편적인 전사 생성기를 사용하여 다양한 언어의 정서적 특징을 통합하고 공통된 음운적 특성을 포착합니다. 이후, 두 번째 단계에서는 보편적인 전사를 특정 언어의 전사로 변환하기 위해 유니버설 변환기를 사용하여 변환 작업을 수행합니다. 이 과정에서 언어 특화 모듈을 사용하지 않으면서도 성능을 확보하는 것이 핵심입니다.

- **Performance Highlights**: LAMA-UT 파이프라인은 Whisper 모델과 비교할 때 상대적 오류 감소율이 45%에 달하며, 오직 Whisper 훈련 데이터의 0.1%만으로 훈련되었습니다. 저자원 언어와 전혀 보지 못한 언어에서도 뛰어난 성능을 발휘하고, 언어 특화 모듈 없이는 언어에 구애받지 않는 ASR 접근 방식과 유사한 성능을 보였습니다. 이러한 성과는 다국적 ASR 시스템을 위한 유연한 기초가 될 것으로 기대됩니다.



### A Comparative Study of DSPy Teleprompter Algorithms for Aligning Large Language Models Evaluation Metrics to Human Evaluation (https://arxiv.org/abs/2412.15298)
Comments:
          7 pages, 10 tables, two-column format

- **What's New**: 이번 논문에서는 Declarative Self-improving Python(DSPy) 프레임워크의 여러 teleprompter 알고리즘이 인공지능 언어 모델(LLM)의 프롬프트(prompt) 최적화와 인간 주석(annotations)과의 정렬에 어떻게 기여하는지를 분석합니다. 이 연구는 특히, LLM을 평가자로 사용하여 환각 탐지(hallucination detection)를 최적화하는 방법에 초점을 맞추고 있으며, 4가지의 teleprompter 알고리즘을 비교 분석하여 그 성능을 평가합니다.

- **Technical Details**: DSPy는 LLM 파이프라인을 선언형 모듈로 추상화하여 특정 목표(e.g., 정확성)의 관점에서 시스템적으로 최적화하는 프로그래밍 모델입니다. 이 모델의 핵심 요소는 predictors, adapters, assertions, metrics 등의 다양한 구성 요소를 포함하며, teleprompters는 이러한 모듈의 품질을 개선하기 위해 특정 프로세스를 따릅니다. Candidate Generation 단계에서는 모듈의 인스턴스를 찾아 새로운 예제를 생상하고, Parameter Optimization 단계에서는 이러한 후보 매개변수를 최적화하여 최고의 조합을 선택합니다.

- **Performance Highlights**: 실험 결과, 최적화된 프롬프트는 다양한 기준 방법들을 초월하여 환각 탐지에 있어서 우수한 성능을 보였습니다. 또한, 특정 teleprompter들은 실험에서 다른 알고리즘보다 더 나은 성과를 나타냈습니다. 각 teleprompter의 성능 비교는 HaluBench라는 공개 데이터셋을 기반으로 하여 수행되었으며, 이는 프롬프트 최적화를 위한 중요한 통찰을 제공합니다.



### Confidence in the Reasoning of Large Language Models (https://arxiv.org/abs/2412.15296)
- **What's New**: 최근의 연구에서 대형 언어 모델(LLMs)의 응답 정확도에 대한 불확실성 논의는 부족한 상황입니다. 본 연구에서는 LLM의 답변에 대한 신뢰 정도가 정확도와 어떻게 상관관계가 있는지를 평가하고자 합니다. 신뢰는 (i) 재고를 요구했을 때 응답을 유지하는 지속성으로 질적으로 측정되며, (ii) 스스로 보고한 신뢰 점수로 수치적으로 측정됩니다. LLM의 신뢰 수준이 형태 소속 토큰 레벨 확률에 의해 부분적으로만 설명된다는 점이 흥미롭습니다.

- **Technical Details**: 본 연구에서는 GPT4o, GPT4-turbo, Mistral의 세 가지 LLM을 평가합니다. BIG Bench-Hard의 두 가지 벤치마크 세트를 사용하여 인과 판단과 논리적 오류 관련 질문을 437개 평가하며, 확률 및 통계 퍼즐 46개도 포함됩니다. LLM들에게 처음의 답변에 대해 재고하도록 요청하며, 이를 통해 질적 및 수치적 신뢰를 측정합니다. 연구 결과, LLM은 랜덤 추측보다 확실히 뛰어난 성과를 보이나, 신뢰에 대한 과대 평가 경향이 발견되었습니다.

- **Performance Highlights**: LLM들은 대체로 랜덤 추측보다 좋지만, 재고 요청 후 두 번째 답변의 전체 정확도가 종종 첫 번째 답변보다 나쁩니다. 흥미롭게도, 심리적 문구에 따라 마음을 바꾸는 경향이 달라졌습니다. 신뢰 점수에 대한 질문에 과대 평가 경향이 있으며, 질적 신뢰와 수치적 신뢰 간의 상당한 상관 관계가 관찰되었습니다. 본 연구에서 제기된 문제는 LLM이 내재적으로 일관된 신뢰 감각을 갖고 있지 않음을 시사합니다.



### A Large-scale Empirical Study on Large Language Models for Election Prediction (https://arxiv.org/abs/2412.15291)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2411.03321

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 사용하여 선거 예측의 정확성을 높일 수 있는 다단계 추론 프레임워크를 제안합니다. 이는 인구 통계, 이데올로기, 시간에 민감한 요소들을 통합하여 예측 정확도를 극적으로 개선하고 편향을 줄이는 데 기여합니다. 2016년과 2020년의 실제 데이터를 통해 검증된 이 접근법은 2024년 미국 대통령 선거 예측에도 적용 가능함을 보여줍니다.

- **Technical Details**: 논문에서는 Sync 합성 데이터 생성 프레임워크를 활용하여 유권자 수준의 데이터 부족 문제를 해결합니다. 다양한 유권자 정보를 제공하여 LLM이 유권자의 선택 과정을 시뮬레이트하도록 하고, 세 가지 진화된 파이프라인을 통해 예측 성능을 평가합니다. LLM의 성능은 실세계의 투표 결과와 얼마나 일치하는지를 통해 평가됩니다.

- **Performance Highlights**: 실험 결과, 다단계 추론 파이프라인은 편향을 줄이고 실제 결과와의 일치를 개선하는 데 성공했습니다. 그러나 여전히 잔여 왜곡이 존재하며, 이는 pretrained corpora의 영향을 나타냅니다. 향후 연구 방향으로는 정치적 왜곡 감소 및 선거 예측에서의 고정관념 비율 완화를 제안하고 있습니다.



### Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models (https://arxiv.org/abs/2412.15287)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 성능 개선을 위해 새로운 추론 인식 세밀 조정(inference-aware fine-tuning) 패러다임을 제안합니다. 기본적으로, 이 방법은 모델이 추론 시 최고의 성능을 보장하는 방식으로 조정됩니다. 특히 Best-of-N (BoN) 전략을 사용하여 모델이 생성한 여러 응답 중에서 가장 좋은 것을 선택하는 방식을 연구합니다.

- **Technical Details**: 우리는 BoN을 고려한 첫 번째 모방 학습(imitation learning) 및 강화 학습(reinforcement learning) 방법을 개발했습니다. BoN은 비가역적(argmax) 연산자를 통해 주어진 문제에 대한 여러 후보 응답을 생성하고, 이 중에서 최적의 응답을 선택하는데, 이는 비차별적인(non-differentiable) 문제를 해결합니다. BoN 인식 모델을 통해 우리는 RL에서의 탐색(exploration)과 활용(exploitation) 간의 트레이드오프를 묘사하는 메타 전략을 학습하는 것을 보여줍니다.

- **Performance Highlights**: 우리의 BoN 인식 세밀 조정 방법은 성능 개선과 추론 시간 계산을 최적화하는 데 매우 효과적입니다. Gemma 2B 모델은 Hendrycks MATH에서 Bo32 성능이 26.8%에서 30.8%로, pass@32는 60.0%에서 67.0%로 증가했습니다. HumanEval에서도 pass@16이 61.6%에서 67.1%로 향상되었습니다.



### Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining (https://arxiv.org/abs/2412.15285)
- **What's New**: 이 논문에서는 대규모 언어 모델의 효과적인 사전 학습을 위해 데이터 선택, 혼합 및 순서에 대한 전략을 수립합니다. 특히, 두 단계의 사전 학습(two-phase pretraining) 개념을 공식화하고 데이터 혼합 및 선택 방법을 체계적으로 연구하여 모델의 정확도를 극대화하는 방법을 제시합니다. 연구 결과, 무작위 데이터 순서와 자연 분포보다 두 단계 접근법이 평균 3.4% 및 17% 향상된 성능을 보였습니다.

- **Technical Details**: 제안된 두 단계 접근법에서 첫 번째 단계(phase-1)는 다양하고 고품질의 웹 크롤 데이터에 중점을 두고, 두 번째 단계(phase-2)는 수학, 코드 및 위키 데이터와 같은 고품질 데이터 소스를 기반으로 합니다. 데이터 혼합 과정에서 데이터 소스의 품질과 에폭(epoch) 수를 고려하여 최적의 혼합 전략을 개발합니다. 또한, 1T 토큰의 축소 샘플링(downsampled data)을 사용하여 여러 혼합을 탐색한 후 15T 토큰의 전체 데이터로 확장할 수 있는 방법을 검증합니다.

- **Performance Highlights**: 연구에서는 지식, 추론, 코딩 및 수학 벤치마크를 포함하는 다양한 다운스트림 작업을 평가하였습니다. 실험 결과, 품질 및 에폭 기반 혼합은 자연 분포 기반 혼합보다 13.2% 우수하고, 두 단계 접근법은 데이터의 무작위 순서보다 평균 3.4% 더 나은 성능을 보여주었습니다. 또한, 축소 샘플링된 데이터의 결과는 15T 토큰의 장기 스케일에서도 일반화되며, 두 단계 접근법의 확장성과 견고성을 demonstrat합니다.



### Channel Merging: Preserving Specialization for Merged Experts (https://arxiv.org/abs/2412.15283)
Comments:
          accepted by AAAI 2025

- **What's New**: 최근 대형 언어 모델(LLM)의 성능 향상을 위해 작업 특정의 세밀한 튜닝( task-specific fine-tuning )이 활용되고 있습니다. 다양한 LLM을 통합하여 전체적인 능력을 크게 향상시키는 방법이 소개되었지만, 전통적인 앙상블 방법은 메모리 집약적이어서 여러 모델을 동시에 GPU 메모리에 로드해야 하는 비효율성이 존재합니다. 이 문제를 해결하기 위해 제안된 새로운 기술인 Channel Merging을 통해 메모리 사용량을 줄이면서도 성능 저하 없이 높은 성능을 유지할 수 있음을 보여줍니다.

- **Technical Details**: Channel Merging은 유사성을 바탕으로 채널 매개변수를 클러스터링하여 오프라인으로 여러 그룹을 형성합니다. 이를 통해 그룹 내에서 유사한 매개변수만 병합하여 파라미터 충돌을 최소화할 수 있습니다. inference(추론) 중에는 병합된 그룹에서 전문 매개변수를 즉시 조회할 수 있어 전문적인 지식을 보존하며, 이전의 모델 병합 기술보다 적은 매개변수를 로드하게 됩니다.

- **Performance Highlights**: Channel Merging은 영어 및 중국어 추론, 수학 추론, 코드 생성 등 다양한 작업에서 비병합 모델과 동등한 성능을 발휘합니다. 그리고 작업 특정 라우터와 결합했을 때 전통적인 앙상블 방법이 요구되는 매개변수의 53%로 성과를 거두어, 다양한 분야에서의 효율성과 활용 가능성을 입증합니다.



### A Systematic Examination of Preference Learning through the Lens of Instruction-Following (https://arxiv.org/abs/2412.15282)
Comments:
          23 pages

- **What's New**: 최근 대규모 언어 모델(LLMs)의 인간 선호에 대한 정렬을 위한 연구가 심화되고 있습니다. 본 연구는 23개의 검증 가능한 제약 조건을 조합하여 48,000개의 고유한 지침 프롬프트를 생성하는 새로운 합성 데이터 생성 파이프라인을 사용하여, 선호 데이터 세트의 특정 속성이 LLM의 성능에 미치는 영향을 체계적으로 조사합니다. 이를 통해 지침을 따르는데 있어 모델의 조정 및 성능 향상을 도모하고자 합니다.

- **Technical Details**: 우선, 본 연구는 선택된 응답(chosen response)과 거부된 응답(rejected response) 쌍으로 LLM의 성능을 개선하는데 중점을 둡니다. 두 가지 방법인 거부 샘플링(rejection sampling, RS)과 몬테 카를로 트리 탐색(Monte Carlo Tree Search, MCTS)을 적용하여 선호 쌍을 자동으로 수집하고, 이를 통해 모델의 일반화 성능을 평가합니다. 이 연구는 지침을 따르는데 있어 응답의 공유 접두사(shared prefixes)와 응답의 대비(contrast) 및 품질(quality)이 어떻게 영향을 미치는지 이해하기 위한 실험을 포함하고 있습니다.

- **Performance Highlights**: 연구 결과, MCTS로 생성된 공유 접두사(preferences pairs with shared prefixes)가 RS로 생성된 것보다 일관되게 우수한 성능을 보였으며, 높은 대비(high-contrast) 쌍이 낮은 대비(low-contrast) 쌍보다 더 나은 성과를 내는 것으로 나타났습니다. 그러나 높은 대비와 낮은 대비 쌍의 혼합이 학습 효율성과 다양성의 균형을 맞추면서 최상의 성능을 가져옵니다. 마지막으로, 중간 난이도의 프롬프트(training prompts)는 과제 전반에 걸쳐 더 나은 일반화를 이끌어내어나가는 것으로 밝혀졌습니다.



### Context-DPO: Aligning Language Models for Context-Faithfulness (https://arxiv.org/abs/2412.15280)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 컨텍스트 충실도를 향상시키기 위해 최초로 설계된 Context-DPO라는 정렬 방법을 제안합니다. 이를 통해 모델이 제공된 정보와 사용자 지침을 보다 잘 따를 수 있도록 합니다. ConFiQA라는 새로운 벤치마크도 소개하여 모호한 구매 모델의 성능을 평가합니다.

- **Technical Details**: ConFiQA는 질문-응답 작업을 기반으로 하여 LLM의 컨텍스트 충실도를 평가합니다. QA, MR, MC라는 세 가지 데이터세트로 구성되며, 각각 단일 및 다중 훅 질문-응답과 다양한 관련된 반사 사례를 포함합니다. 모델의 훈련 상태와 크기에 따라 컨텍스트 충실도가 감소하는 경향을 보이며, 이를 해결하기 위해 Context-DPO를 통해 반사배급을 보상하는 방법을 적용합니다.

- **Performance Highlights**: Context-DPO는 LLM의 컨텍스트 충실도를 35%에서 280%까지 개선하여, 기존의 모든 모델들보다 현저히 뛰어난 성능을 보여주었습니다. 추가적으로, 이 연구는 LLM의 생성 능력에 부정적인 영향을 주지 않으면서도 컨텍스트 충실도를 저해하지 않음을 입증하였습니다. 또한, 모델의 정렬 결과를 분석하여 컨텍스트 활용의 해석 가능한 통찰을 제공합니다.



### PLPP: Prompt Learning with Perplexity Is Self-Distillation for Vision-Language Models (https://arxiv.org/abs/2412.15277)
- **What's New**: 이 논문에서는 VL(비전-언어) 모델의 성능 향상을 위해 PLPP(PerPlexity 기반 Prompt Learning)이라는 새로운 정규화 방법을 제안합니다. PLPP는 기존의 CoOp 방법에서 발생할 수 있는 오버피팅 문제를 해결하는 데 중점을 두고 있으며, 단일 CLIP 손실에 의존하는 대신에 perplexity 손실을 활용하여 프롬프트 학습을 조정합니다. 이 방법은 비트 리미터와 언어 모델 헤드를 통합하여 단어 확률 분포를 출력함으로써 더욱 안정적인 학습 과정을 구현합니다.

- **Technical Details**: PLPP는 두 단계로 이루어진 작업을 통해 프롬프트의 perplexity를 계산합니다. 첫 번째 단계에서는 임베딩 레이어의 가중치와 각 프롬프트 벡터 간의 코사인 유사도를 계산하여 라벨을 얻습니다. 이후에는 추가 학습 없이 언어 모델 헤드를 통해 단어 확률 분포를 출력하여 perplexity를 계산합니다. 이 두 단계를 통해 프롬프트 최적화에서의 정보를 합리적으로 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과, PLPP는 여러 분류 작업에서 기존 방법들에 비해 뛰어난 성능을 보였습니다. PLPP는 다른 방법들에 비해 오버피팅을 방지하고, 프롬프트 최적화를 안정화함으로써 학습 진전을 가속화하는 데 기여했습니다. 이 연구는 VL 모델에서 이미 잘 알려진 언어 모델링 기법을 활용하여 새로운 방향성을 제시하고 있습니다.



### Memory-Augmented Agent Training for Business Document Understanding (https://arxiv.org/abs/2412.15274)
Comments:
          11 pages, 8 figures

- **What's New**: Matrix(메모리 증강 에이전트 훈련을 통한 추론 및 반복 탐색)라는 새로운 패러다임을 제안합니다. 이 시스템은 LLM(대형 언어 모델) 에이전트가 경험 기반의 메모리 개선과 반복 학습을 통해 도메인 전문성을 지속적으로 구축할 수 있도록 도와줍니다. 기존 LLM의 한계를 극복하고 비즈니스 문서 처리 작업을 위한 보다 전문화된 도구로 변모시킬 수 있는 가능성을 열어줍니다.

- **Technical Details**: Matrix는 문서의 구조 및 추출 패턴에 대한 이해를 체계적으로 개선하는 독창적인 반복 자기 정련 과정(iterative self-refinement mechanism)을 통합하고 있습니다. 이 프레임워크는 에이전트가 작업 탐색(task exploration) 및 최적화(task optimization)를 반복적으로 수행하여 일반적인 작업 구조에 대한 통찰력을 향상시키는 과정을 포함합니다. 이러한 프로세스는 향후 작업 해결 시 유용한 긴급 기억(long-term memory)을 생성합니다.

- **Performance Highlights**: 실험 결과, Matrix는 체인 오브 사고(prompting) 대비 30.3%의 성능 향상을 보였고, 기본 LLM 에이전트 대비 35.2%, 반사(reflexion) 대비 27.28%의 개선을 이뤘습니다. 최적화된 시스템은 API 호출 수를 줄이고, 비용을 절감하는 동시에 평균적으로 더 긴 문서도 처리할 수 있는 강점을 보여주었습니다. 이러한 향상된 퍼포먼스는 문서 처리와 비즈니스 환경에서 매우 중요한 현상을 나타냅니다.



### SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15272)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 다양한 태스크(Task)에서 뛰어난 유연성을 보여주고 있습니다. 이러한 맥락에서, Retrieval-Augmented Generation (RAG) 접근 방식이 외부 지식 소스인 지식 그래프(KGs)를 활용하여 환각(hallucination)을 제거하는 데 있어 강력한 방법으로 자리잡고 있습니다. 본 논문에서는 KG 기반 RAG 태스크를 조사하고, 유사 그래프 강화 검색 증대 생성(SimGRAG) 방법을 제안하여 쿼리 텍스트와 KG 구조를 정렬하는 문제를 효과적으로 해결합니다.

- **Technical Details**: SimGRAG 방법은 두 단계의 프로세스를 통해 쿼리 텍스트와 KG 구조의 정렬을 수행합니다. 첫 번째 단계는 쿼리를 원하는 그래프 패턴으로 변환하는 LLM을 사용하는 'query-to-pattern' 단계입니다. 두 번째 단계는 패턴과 후보 서브그래프 간의 정렬 정도를 그래프 의미 거리(Graph Semantic Distance, GSD) 메트릭을 이용해 정량화하는 'pattern-to-subgraph' 단계입니다. 이 방법은 1000만 규모의 KG에서 1초 이내에 최상위 k 서브그래프를 효율적으로 식별하는 최적화된 검색 알고리즘을 개발하여 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, SimGRAG는 질문 응답 및 사실 검증(task)에서 최신 KG 기반 RAG 방법들을 초월하는 성능을 보여줍니다. 본 논문이 제시하는 방법은 플러그 앤 플레이(plug-and-play) 사용성과 확장성(scalability)을 갖추고 있어, 다양한 KG 및 LLM과의 매끄러운 통합이 가능합니다. 또한, SimGRAG는 불필요한 정보 유출을 방지하고, 가장 관련성 높은 서브그래프를 명확하게 찾아내는 능력을 갖추어 있습니다.



### A MapReduce Approach to Effectively Utilize Long Context Information in Retrieval Augmented Language Models (https://arxiv.org/abs/2412.15271)
- **What's New**: 이 연구에서는 의료 분야에서 Retrieval-Augmented Generation (RAG) 워크플로우의 견고성과 신뢰성을 개선하는 것을 목표로 합니다. 제안된 BriefContext 전략은 모델의 가중치를 수정하지 않으면서 "lost-in-the-middle" 문제를 해결하고자 합니다. 다양한 LLM 백본과 여러 QA 데이터셋에서 이 워크플로우의 우수성을 Demonstrated 하였습니다.

- **Technical Details**: BriefContext는 긴 컨텍스트 reasoning 작업을 여러 개의 짧은 컨텍스트 reasoning 작업으로 변환하는 새로운 프레임워크입니다. 이 프레임워크는 맵-리듀스(map-reduce) 개념을 활용하여 긴 컨텍스트를 여러 개의 파티션으로 나누고 이를 여러 LLM 세션으로 분배합니다. 또한, Preflight 메커니즘을 도입해 "lost-in-the-middle" 문제의 발생을 예측하는 과정을 포함하고 있습니다.

- **Performance Highlights**: BriefContext는 RAG 베이스라인보다 월등한 성능을 보였으며, 특히 키 정보가 중간에 위치할 때 큰 개선을 확인했습니다. 짧은 컨텍스트에서 LLM의 성능이 더 우수하다는 것을 입증하였고, Preflight 체크는 공지 발생 예측에서 92.61%의 재현율을 기록하였습니다. 이 연구는 의료 분야에서 LLM을 안전하게 배포하는 데 기여할 수 있는 가능성을 보여줍니다.



### Baichuan4-Finance Technical Repor (https://arxiv.org/abs/2412.15270)
- **What's New**: 이번 논문에서는 Baichuan4-Finance 시리즈의 개발을 소개합니다. 이 모델들은 Baichuan4-Turbo 기본 모델을 기반으로 하며, 금융 분야에 특화되어 있습니다. Baichuan4-Finance-Base와 Baichuan4-Finance로 구성되어 있으며, 금융 지식을 획득하는 새로운 훈련 전략을 제안합니다.

- **Technical Details**: Baichuan4-Finance-Base의 토크나이저는 byte-level byte-pair encoding (BBPE)을 사용하며, 141,056개의 정규 토큰으로 구성됩니다. Qued Query Attention (GQA)을 활용하여 추론 속도를 개선하고, RMSNorm을 통해 학습 안정성을 보장합니다. RoPE를 활용하여 위치 인코딩을 수행하며, 기존의 Multi-Head Attention (MHA)보다 효율적인 아키텍처를 구현했습니다.

- **Performance Highlights**: Baichuan4-Finance-Base는 여러 금융 작업에서 거의 모든 경쟁 모델을 크게 능가하는 성과를 나타냅니다. 일반 LLM 기준에서도 성능을 유지하면서 금융 응용 시나리오에서 더욱 놀라운 성능을 보입니다. 이는 금융 LLM 분야의 혁신을 촉진할 잠재력을 보여줍니다.



### The Reliability Paradox: Exploring How Shortcut Learning Undermines Language Model Calibration (https://arxiv.org/abs/2412.15269)
Comments:
          10 pages; 9 figures. Accepted for publication at the Hawaii International Conference on System Sciences (HICSS-58) 2025

- **What's New**: 이 논문은 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)의 신뢰성과 일반화 능력 간의 관계를 조사합니다. 기존 연구에서 낮은 교정 오류(Expected Calibration Error, ECE)가 신뢰할 수 있는 예측을 의미한다고 여겼지만, 이 연구는 이러한 가정을 뒤집으며, 낮은 교정 오류가 오히려 비일반적인 결정 규칙을 나타낼 수 있음을 보여줍니다.

- **Technical Details**: 모델의 신뢰성을 평가하기 위해, 이 연구에서는 통계적 교정 평가 지표인 ECE를 활용하여 PLMs를 분석합니다. 또한, 통계적 교정 오류 측정이 결정 규칙의 비강건성을 포착하는 데 한계가 있음을 강조하고 있습니다. 단기 학습(shortcut learning)에 기반한 모델의 신뢰도를 평가하는 과정에서 통계적 기법과 함께 데이터 통계를 활용한 접근 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, 잘 교정된 모델이 반드시 신뢰성이 높은 것은 아님을 발견했습니다. 저자들은 PLMs의 다양한 분류 작업에서 모델의 단기 학습 행동을 확인하고, 이러한 행동이 교정 오류와 어떻게 관련되는지 분석했습니다. 이는 PLMs의 신뢰성을 높이기 위해 수학적 교정 능력과 일반화 목표 간의 간극을 메우는 필요성을 제기합니다.



### Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph (https://arxiv.org/abs/2412.15268)
Comments:
          8 pages of content, 7 pages of Limitation, Ethical Statement, Reference ans Appendix

- **What's New**: 이번 논문에서는 MetaTox라는 새로운 방법을 제안합니다. MetaTox는 메타-독성 지식 그래프(meta-toxic knowledge graph)를 활용하여 혐오 및 독성 콘텐츠 탐지를 강화하는 기술입니다. 이 방법은 기존의 독성 기준 데이터셋을 이용하여 포괄적인 메타-독성 지식 그래프를 구축하고, 이를 통해 독성 정보의 정확도를 높이는 것을 목표로 합니다.

- **Technical Details**: MetaTox의 구성 과정은 세 가지 단계로 진행됩니다. 첫 번째는 근거 추론(rationale reasoning)으로, 어떤 내용이 독성을 유발하는지를 파악하는 과정입니다. 두 번째는 삼중항 추출(triplet extraction) 단계로, 독성 개체와 관계를 추출하며 질의의 품질을 확인하는 전략을 포함합니다. 마지막으로 중복 제거(duplicate removal) 과정을 통해 유사한 의미의 노드와 관계를 통합하여 지식 그래프를 완성합니다.

- **Performance Highlights**: MetaTox는 여러 데이터셋에서 실시한 실험과 심층 사례 연구를 통해 false positive 비율을 상당히 줄이면서 전반적인 독성 탐지 성능을 향상시키는 데 성공했습니다. 향후 이 지식 그래프는 공개될 예정이며, 논문 수용 시 추가적인 공개가 계획되어 있습니다.



### On the Structural Memory of LLM Agents (https://arxiv.org/abs/2412.15266)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM) 기반 에이전트의 성능에 미치는 메모리 구조와 메모리 검색 방법의 영향을 탐구합니다. 4가지 유형의 메모리 구조(조각, 지식 삼중, 원자 사실 및 요약)와 이들을 결합한 혼합 메모리를 평가하여, 각 구조별로 특정 작업에 최적화될 수 있는 방법을 제안합니다. 또한, 단일 단계 검색, 재정렬 및 반복 검색과 같은 3가지 메모리 검색 방법을 평가하며, 이 조합이 에이전트의 전반적인 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 네 가지 작업(다중 홉 QA, 단일 홉 QA, 대화 이해, 독서 이해)과 여섯 개 데이터 세트를 활용하여 수행됩니다. 실험 결과, 각각의 메모리 구조가 특정 작업에 제공하는 고유한 이점이 있음을 발견했습니다. 더불어, 혼합 메모리가 노이즈 환경에서도 뛰어난 탄력을 보여주며, 반복 검색 방법이 여러 상황에서 가장 우수한 성능을 발휘함을 확인했습니다.

- **Performance Highlights**: 결과적으로, 각 메모리 구조는 고유한 강점을 발휘하며, 혼합 메모리는 다양한 작업에서 균형 잡힌 경쟁력을 유지합니다. 조각과 요약은 긴 맥락을 요구하는 작업에서 특히 효과적이고, 지식 삼중과 원자 사실은 관계적 추론 및 정밀성에서 두각을 나타냅니다. 반복 검색은 대다수 작업에서 가장 효과적인 메모리 검색 방법으로 확인되었습니다.



### Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large Language Models (https://arxiv.org/abs/2412.15265)
- **What's New**: 이번 논문에서는 중국의 안전 관련 지식을 평가하기 위해 중국 SafetyQA 벤치마크를 소개합니다. 이는 기존의 데이터셋들이 일반 지식에 집중하는 반면, LLM의 안전성이 중요한 법률, 정책, 윤리와 관련된 사실적 능력을 평가하는 데 중점을 두고 있는 점에서 혁신적입니다. 중국 SafetyQA 벤치마크는 고품질의 안전 예제 2000개 이상을 포함하고 있으며, 다양한 카테고리에 걸쳐서 안전 지식을 포괄적으로 다루고 있습니다.

- **Technical Details**: 중국 SafetyQA 데이터셋은 7가지 주요 카테고리로 조직되어 있으며, 각 카테고리는 다양한 세부 주제를 포함하고 있습니다. 데이터셋은 안전 관련 지식의 정확성과 질을 보장하기 위해 엄격한 선택과 주석 과정을 거쳤으며, 모든 샘플은 총 두 가지 형식(질문-응답(QA) 및 객관식(MCQ))으로 제공됩니다. 이러한 배열은 LLM의 안전 지식 경계를 쉽게 평가할 수 있도록 돕습니다.

- **Performance Highlights**: 30개 이상의 LLM을 평가한 결과, 대부분의 모델이 안전 분야 내에서 사실적 정확성에 부족함을 드러냈습니다. 또한, LLM들은 교육 데이터에서 지식 오류를 포함하고 있으며, 안전 지식에 관해서는 과도한 자신감을 보였습니다. 연구 결과는 RAG(회수 보강 생성)가 안전 사실성을 높이는 데 도움을 줄 수 있음을 나타냅니다.



### ReXTrust: A Model for Fine-Grained Hallucination Detection in AI-Generated Radiology Reports (https://arxiv.org/abs/2412.15264)
Comments:
          Accepted to AIMedHealth 10 pages, 5 figures

- **What's New**: 이 연구에서는 AI 생성 방사선 보고서에서의 허위 진술을 감지하기 위한 새로운 프레임워크인 ReXTrust를 소개합니다. ReXTrust는 대규모 비전-언어 모델(LVLM)에서의 히든 상태 시퀀스를 활용하여 구체적인 검사 결과에 대한 허위 진술 위험 점수를 생성합니다. 이 모델은 특히 임상적으로 중요한 결과에 대한 허위 진술 감지에서 우수한 성능을 발휘하여, 자동화된 방사선 보고서의 안전성과 신뢰성을 개선할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ReXTrust는 LVLM에서 생성된 방사선 보고서의 허위 진술을 감지하기 위해 설계된 화이트 박스 모델입니다. 이 시스템은 LVLM 히든 상태에서 교육된 자기 주의(self-attention) 모듈을 사용하여, 특정 방사선 검사 결과에 대한 세밀한 통찰을 제공하고 신뢰할 수 있는 허위 진술 위험 점수를 산출합니다. 내부 모델 표현을 분석함으로써 ReXTrust는 생성 과정에서 허위 진술을 식별할 수 있으며, 후속 분석에 의존하지 않습니다.

- **Performance Highlights**: ReXTrust는 MIMIC-CXR 데이터세트의 하위 집합에서 평가되었으며, 모든 검사 결과에 대해 AUROC 0.8751, 임상적으로 중요한 결과에 대해서는 AUROC 0.8963을 달성했습니다. 이는 기존 접근법과 비교할 때 우수한 성과를 나타냅니다. 결국 ReXTrust는 모델 히든 상태를 활용하여 의료 AI 시스템의 허위 진술 감지 신뢰성을 높일 수 있는 가능성을 보여주고 있습니다.



### PROPOE 2: Avan\c{c}os na S\'intese Computacional de Poemas Baseados em Prosa Liter\'aria Brasileira (https://arxiv.org/abs/2412.15263)
Comments:
          in Portuguese language

- **What's New**: 이번 연구에서는 PROPOE 2 시스템을 소개합니다. 이 시스템은 브라질 문학의 산문에서 추출한 미터화된 문장을 기반으로 하여 보다 다양한 구조적 및 리드미컬한 가능성을 제공합니다. PROPOE 2는 원래 시스템보다 리듬과 소리 효과를 탐색하는 데 더욱 일관된 방식으로 발전된 점이 특징입니다.

- **Technical Details**: PROPOE 시스템은 MIVES (Mining Verse Structure) 도구를 통해 미터화된 문장을 채굴하여 특정한 리듬과 음성을 고려하여 시를 생성합니다. PROPOE 2에서는 시 구조와 리듬 기준을 조절할 수 있는 더 많은 유연성을 제공하며, 알고리즘 분류와 같은 여러 매개변수를 통해 리듬적 다양성을 획득합니다.

- **Performance Highlights**: 실제로 시스템을 통해 생성된 시의 결과는 여러 매개변수의 변화를 통해 생성 및 평가 기준을 보여줍니다. PROPOE 2는 구조적 및 리듬적 변동성을 넓혀 시의 품질을 향상시키는 방법을 제시하고 있으며, 다른 자동 시 생성 시스템과 비교했을 때 더 많은 시각적, 소리적 효과를 만든다는 점에서 주목받고 있습니다.



### Advanced ingestion process powered by LLM parsing for RAG system (https://arxiv.org/abs/2412.15262)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문은 다양한 구조적 복잡성을 가진 멀티모달 문서를 처리하는 데 어려움을 겪는 Retrieval Augmented Generation (RAG) 시스템을 위한 새로운 멀티 전략 파싱 접근법을 제안합니다. LLM 기반 OCR을 활용하여 프레젠테이션 및 고밀도 텍스트 파일을 포함한 다양한 문서 유형의 콘텐츠를 추출합니다. 이 방법론은 서로 다른 정보 유형 간의 관계를 생성하고 문맥 인식 메타데이터를 생성하는 노드 기반 추출 기법을 사용합니다.

- **Technical Details**: 선행 처리 단계는 구문 분석, 조립 및 메타데이터 추출의 세 하위 프로세스로 구성됩니다. 구문 분석 단계에서는 Python 라이브러리와 멀티모달 LLM을 이용하여 이미지 및 텍스트 콘텐츠를 추출하며, AWS Textract와 같은 외부 기계 학습 모델을 사용하는 OCR도 포함됩니다. 각 페이지의 이미지를 분석하고 설명한 후, Multimodal Assembler Agent가 모든 페이지의 텍스트를 통합하여 종합적인 문서 수준의 Markdown 파일을 생성합니다.

- **Performance Highlights**: 실험적인 평가 결과는 RAG 시스템의 효율성이 향상됨을 보여주며, 정보의 회수 능력과 답변의 적합성이 증가했습니다. 이 접근법은 특히 답변의 관련성과 정보의 정확성을 측정하는 데 중요한 평가 지표를 설정하여 시스템의 전반적인 품질 향상에 기여했습니다. 논문에서 사용된 평가 지표들은 RAG 시스템의 효과성을 정량화하는 데 필수적이며, 이를 통해 시스템의 신뢰성을 확보할 수 있습니다.



### Analyzing Images of Legal Documents: Toward Multi-Modal LLMs for Access to Justic (https://arxiv.org/abs/2412.15260)
Comments:
          Accepted at AI for Access to Justice Workshop at Jurix 2024, Brno, Czechia. Code and Data available at: this https URL

- **What's New**: 이 논문은 법적 정보에 접근하기 어려운 일반인을 지원하기 위한 다중 모달 대형 언어 모델(LLMs)의 활용 가능성을 조사합니다. 특히, 손으로 작성된 서류의 이미지를 분석하여 관련 정보를 자동으로 추출하는 방법을 제안하고 있습니다. 초기 결과는 긍정적이지만 이미지 품질이 낮을 때의 한계도 드러났습니다.

- **Technical Details**: 논문에서는 '주거 임대 계약서(Standard Form of Lease)'의 이미지를 기반으로 한 데이터셋을 생성했습니다. 세 가지 시나리오를 설정하여 LLM이 이미지에서 정보를 얼마나 잘 추출할 수 있는지 평가했습니다. 각 시나리오는 유사한 이름을 가진 세 명의 세입자와 누락된 필드 등을 포함하여 점진적인 난이도로 설계되었습니다.

- **Performance Highlights**: 연구 결과, 다중 모달 LLM이 이미지를 통해 법적 문서에서 구조화된 정보를 추출하는 데 뛰어난 성능을 보였습니다. 이러한 접근은 일반인이 법적 권리를 이해하고, 정부 혜택을 신청하는 데 큰 도움을 줄 수 있을 것으로 기대됩니다. 하지만 이미지 품질과 복잡한 데이터가 결과에 미치는 영향은 여전히 해결해야 할 과제로 남아 있습니다.



### GLARE: Google Apps Arabic Reviews Datas (https://arxiv.org/abs/2412.15259)
Comments:
          Github Repo: this https URL Zenodo: this https URL

- **What's New**: 이번 논문에서는 사우디 아라비아의 Google Play Store에서 수집된 Arab Apps Reviews 데이터셋(GLARE)을 소개합니다. 이 데이터셋은 9,980개의 Android 애플리케이션에 대한 7,600만 건의 리뷰로 구성되어 있으며, 아랍어 리뷰는 6,900만 건입니다. GLARE는 가장 큰 아랍어 리뷰 데이터셋으로, 자연어 처리(NLP) 및 감정 분석(Sentiment Analysis)과 같은 다양한 작업을 위해 유용할 것입니다.

- **Technical Details**: GLARE 데이터셋은 Google PlayStore에서 google-play-scraper 라이브러리를 사용하여 수집되었습니다. 주요 공부 및 서브 카테고리에서 각각 상위 200개의 무료 앱의 리뷰를 크롤링하여 총 76M개의 리뷰를 확보하였습니다. 중복을 제거한 후, 최종적으로 69M개의 아랍어 앱 리뷰가 남았으며, 이 데이터셋은 GitHub와 Hugging Face에서 다운로드할 수 있습니다.

- **Performance Highlights**: GLARE 데이터셋은 전체 9,980개 앱의 리뷰와 함께, 리뷰 점수와 사용자 참여에 대한 유용한 정보를 포함합니다. 앱의 리뷰는 대다수(80%)가 5점 만점으로 skewed 되어 있으며, 98% 이상의 앱은 긍정적인 피드백을 얻었습니다. 뿐만 아니라, 약 48%의 앱은 사용자 리뷰에 대해 개발자와 상호작용을 하였고, 이는 고객 행태 분석에 유용할 것입니다.



### DisEmbed: Transforming Disease Understanding through Embeddings (https://arxiv.org/abs/2412.15258)
- **What's New**: 이 논문에서는 DisEmbed이라고 하는 질병에 초점을 맞춘 임베딩 모델을 제안합니다. 기존의 모델들이 일반적인 의료 분야에 대해 폭넓게 일반화되어 질병에 대한 깊은 이해를 어려워하는 반면, DisEmbed은 질병 설명, 증상 및 질병 관련 Q&A 쌍의 합성 데이터 세트를 통해 훈련되었습니다. 이 모델은 특히 질병 관련 작업에서 강력한 성능을 보여주며, 진단 시스템과 같은 특정 이용 사례에서 유용하게 활용될 수 있습니다.

- **Technical Details**: DisEmbed 모델은 전통적인 의료 임베딩의 일반화를 벗어나 질병에 관련된 맥락을 깊이 있게 이해하는 데 중점을 두고 개발되었습니다. 모델의 훈련 데이터를 위해 ICD-10-CM 데이터 세트를 기반으로 질병 이름을 생성하고, 해당 질병에 대한 증상 및 설명을 생성하여 질병의 개념을 더 잘 이해할 수 있도록 설계되었습니다. 또한, 모델은 Multiple Negatives Ranking Loss (MNRL) 기법을 사용하여 질병과 관련된 의미 있는 표현을 학습할 수 있도록 최적화되었습니다.

- **Performance Highlights**: DisEmbed의 성능은 질병 특정 데이터 세트를 사용하여 평가되었습니다. 이 모델은 유사한 질병을 구분하고, 질병 관련 문맥을 식별하는 데 특히 뛰어난 성능을 보였습니다. 특히 RAG (retrieval-augmented generation) 작업에서 모델의 성능이 두드러지며, 이를 통해 질병 분류 및 증상 매핑과 같은 다양한 다운스트림 작업에 적용할 수 있음을 입증했습니다.



### An Incremental Clustering Baseline for Event Detection on Twitter (https://arxiv.org/abs/2412.15257)
- **What's New**: 이번 연구에서는 텍스트 스트림의 이벤트 감지(event detection) 과정을 다루고 있습니다. 새로운 Incremental Clustering Algorithm을 활용하여 최근의 Sentence Embeddings 기술을 결합했습니다. 이는 Cao et al. (2024) 및 Mazoyer et al. (2020)의 연구와 비교하여 성능 기준을 설정하는 것을 목표로 하고 있습니다.

- **Technical Details**: Incremental Clustering Algorithm은 데이터의 지속적인 유입에 따라 클러스터를 갱신하는 방식으로 작동합니다. Sentence Embeddings는 문장의 의미를 고차원 공간에 매핑하여 처리하는 여러 기법을 포함합니다. 이 연구는 텍스트 스트림 분석에 있어 알고리즘의 효과적인 조합을 탐구하고 있습니다.

- **Performance Highlights**: 연구 결과, 이전 연구와 비교했을 때 성능이 유의미하게 개선된 것으로 나타났습니다. 이는 향후 관련 연구에 중요한 기준점(baseline)으로 활용될 수 있습니다. 제안된 방법론은 온라인 미디어와 소셜 네트워크 분석 분야에서의 활용 가능성을 높입니다.



### Structured Extraction of Real World Medical Knowledge using LLMs for Summarization and Search (https://arxiv.org/abs/2412.15256)
Comments:
          10 pages, 3 figures, Work published in 4th Workshop on Knowledge Graphs and Big Data (In Conjunction with IEEE Big Data 2024)

- **What's New**: 이 논문에서는 질병 발견과 분석을 가속화하기 위해 환자 지식 그래프를 구축하는 새로운 접근 방식을 제안합니다. 기존의 질병 온톨로지가 환자의 상태나 희귀 질병의 미세한 차이를 포착하기 어려운 반면, 대규모 언어 모델 추출 기술을 활용하여 자연어를 통해 데이터를 보다 유연하게 추출할 수 있는 방법을 제시하고 있습니다. 이를 통해 실세계 데이터에서 의미 있는 통찰을 도출하고, 기존 온톨로지에 연결되는 환자 특화 지식 그래프를 구축했습니다.

- **Technical Details**: 이 연구에서 제안한 방법은 메타 데이터, SNOMED-CT, RxNORM, HPO와 같은 기존 온톨로지에 추출된 개체들을 연계하여 'ground truth'를 제공합니다. 실험에 사용된 데이터는 약 3,360만 환자의 대규모 이동 치료 전자 건강 기록(EHR) 데이터베이스로, 이를 통해 Dravet 증후군과 Beta-propeller protein-associated neurodegeneration (BPAN) 환자를 찾는 데 성공했습니다. 환자의 증상 기반 검색 방식과 환자 특화 지식 그래프 구축을 통해 실제 질병 발견 사례를 입증했습니다.

- **Performance Highlights**: 최신 데이터와 사례 연구를 통해 제안된 방법의 효과가 입증되었습니다. LLM 기반 엔터티 추출을 사용하여 Dravet 증후군에 대한 ICD10 코드 검증을 통해 환자 특성을 잘 설명하며, 의료 기록에서 유래한 데이터를 활용해 다양한 질병 연구 결과를 종합적이고 저장된 지식으로 제공합니다. 이와 같이 고도화된 자동화 시스템은 의료 연구와 데이터 통합의 혁신을 가능하게 하며, 실질적인 임상 연구 설계를 가속화하는 데 기여합니다.



### Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation (https://arxiv.org/abs/2412.15255)
Comments:
          14 pages

- **What's New**: 이 논문에서는 지식 증류(knowledge distillation)를 활용하여 언어 모델 벤치마크 점수를 조작할 수 있다는 사실을 밝혀내어 현재의 평가 관행에서의 중대한 취약점을 드러냈습니다. 우리는 "데이터 세탁(Data Laundering)"이라고 불리는 세 단계의 과정을 소개하는데, 이는 재정의 머니 론더링(money laundering)과 유사합니다. 이 방법은 겉보기에는 합법적인 중간 학습 단계를 통해 벤치마크 특정 지식을 은밀하게 전이할 수 있게 해줍니다.

- **Technical Details**: 제안된 데이터 세탁 방법은 지식 증류 기술을 통해 이루어집니다. 이 과정은 세 가지 단계로 나누어지는데, 배치(placement), 레이어링(layering), 통합(integration)입니다. 첫 번째 단계에서는 테스트 데이터로 훈련된 튜터 모델에서 벤치마크 지식을 "배치"하여 초기 지식 자본을 형성합니다. 그런 다음, 레이어링 단계에서 지식 증류를 통해 합법적인 중간 데이터셋을 사용하여 지식을 전이합니다.

- **Performance Highlights**: 실험을 통해 2레이어 BERT 학생 모델이 GPQA에서 최대 75%의 벤치마크 정확도를 달성할 수 있다는 것을 보여주었습니다. 그러나 이 성과는 실제로 진정한 추론 능력을 개발하지 않고도 이루어졌습니다. 이러한 방법은 연구자들이 지식 증류를 사용하는 중에 의도치 않게 점수를 부풀리는 방식을 채택할 수 있음을 강조하며, AI 평가 방법의 강화를 위한 필요성을 환기시킵니다.



### RIRO: Reshaping Inputs, Refining Outputs Unlocking the Potential of Large Language Models in Data-Scarce Contexts (https://arxiv.org/abs/2412.15254)
- **What's New**: 이번 연구에서 제안된 RIRO는 데이터가 부족한 환경에서 성능을 향상시키기 위해 고안된 새로운 두층 아키텍처입니다. 첫 번째 층은 고급 prompt engineering을 통해 입력을 재구성하여 훈련 데이터와의 정렬을 개선하고, 두 번째 층은 출력을 정제하여 불일치를 최소화합니다.

- **Technical Details**: RIRO는 Phi-2, Falcon 7B, 및 Falcon 1B와 같은 모델을 미세 조정하며, Phi-2가 가장 뛰어난 성능을 보입니다. 이 구조는 Refining LLM, Reshaping LLM, Stacked LLM의 세 가지 아키텍처로 구성되어 있으며, 입력 정규화 및 출력 재형성을 통해 데이터 스카스 환경에서 성능을 향상시킵니다.

- **Performance Highlights**: RIRO는 입력 변동성을 최소화하고 고품질의 출력을 생성할 수 있도록 설계되었으며, QLoRA를 사용하여 효율적인 모델 미세 조정을 가능하게 합니다. 이 방법은 의료, 법률 문서 및 소프트웨어 테스트와 같은 데이터가 부족한 환경에서도 정확성과 정밀성을 유지할 수 있는 확장 가능하고 실용적인 솔루션을 제공합니다.



### Using Machine Learning to Distinguish Human-written from Machine-generated Creative Fiction (https://arxiv.org/abs/2412.15253)
Comments:
          Accepted for publication at ICAART 2025: this https URL

- **What's New**: 이 연구는 기존의 연구에서 충분히 다뤄지지 않았던 AI 생성 텍스트 탐지를 위한 새로운 접근법인 'AI Detective' 도구를 개발했습니다. 이를 통해 인간 작가의 창작과 AI에 의해 생성된 창의적인 픽션을 구별할 수 있는 기계 학습(ML) 모델을 활용했습니다. 연구 결과, Naive Bayes와 Multi-Layer Perceptron 분류기가 각각 95% 이상의 정확도를 달성하며, 인간 판별자를 크게 초월한 성과를 보였습니다.

- **Technical Details**: 이 논문에서는 데이터 가용성 및 컴퓨팅 파워의 증가 덕분에 신경망 모델과 딥 러닝을 기반으로 한 자동 텍스트 생성 접근법이 발전했다는 점을 강조합니다. 특히, Transformer 모델의 도입은 텍스트 생성에 혁신을 가져왔으며, 이 구조는 인코더와 디코더로 구성됩니다. 연구팀은 이러한 모델을 통해 짧은 문장의 창작물을 정확히 분류할 수 있는 기계 학습 분류기를 개발했습니다.

- **Performance Highlights**: 이 연구에서 사용된 분류기는 특히 짧은 텍스트 샘플(약 100단어)에 대해 높은 정확도로 성능을 보였고, 이전의 연구에서는 이런 짧은 샘플을 분류하는 데 어려움이 있었던 점을 고려할 때 상당한 개선이 이루어진 것입니다. 'AI Detective' 도구는 편집자와 출판사들이 인공지능의 영향을 받는 창작물의 경제적, 문화적 가치를 보호하기 위해 활용될 수 있는 기반을 제공합니다.



### NER- RoBERTa: Fine-Tuning RoBERTa for Named Entity Recognition (NER) within low-resource languages (https://arxiv.org/abs/2412.15252)
- **What's New**: 최근 자연어 처리(Natural Language Processing, NLP)는 일상 생활에서 광범위하게 사용되고 있으며, 특히 음성 이해, 번역, 명명된 개체 인식(Named Entity Recognition, NER), 텍스트 분류 및 ChatGPT와 같은 생성적 텍스트 모델에 이르기까지 그 활용도가 높습니다. 하지만 쿠르드어(Kurdish Language)는 아직 NLP 응용 프로그램에 포함될 만큼의 데이터가 부족하여, 쿠르드어 NLP(KNLP) 개발에 독특한 도전 과제가 존재합니다. 본 논문에서는 쿠르드어 NER(KNER)에 대한 한계를 극복하기 위한 방법론을 제안합니다.

- **Technical Details**: 우리는 사전 학습된 RoBERTa 모델을 KNER을 위해 세밀하게 조정하는 방법론을 연구하였습니다. 이를 위해 먼저 쿠르드어 말뭉치(Kurdish corpus)를 생성하고, 개조된 모델 아키텍처를 설계하며, 훈련 절차를 구현하였습니다. 모델을 평가하기 위해 다양한 토크나이제이션(tokenization) 방법과 훈련된 모델을 사용하여 수행한 실험 세트도 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, SentencePiece 토크나이제이션 방법으로 세밀하게 조정된 RoBERTa 모델이 기존의 모델에 비해 KNER 성능을 12.8% 향상시킨 것으로 나타났습니다. 이 연구를 통해 KNLP의 새로운 기준을 수립하였으며, 다양한 KNLP 작업에서의 성능을 개선할 수 있는 가능성을 시사합니다.



### AgentPS: Agentic Process Supervision for Multi-modal Content Quality Assurance through Multi-round QA (https://arxiv.org/abs/2412.15251)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 논문에서는 AgentPS라는 새로운 프레임워크를 소개합니다. AgentPS는 멀티모달 대형 언어 모델(MLLMs)에 프로세스 감독을 통합하여 복잡한 논리 구조를 개선합니다. 추가적으로, LLM 생성 라벨을 사용하여 인간 주석을 대체할 수 있는 가능성도 제시하고 있습니다.

- **Technical Details**: AgentPS는 다단계 질문 응답 방식을 통해 과정 중간 정보를 제공하여 MLLM의 추론 능력을 향상시키는 방법을 구현하고 있습니다. MLLM은 비전 인코더, 비전-언어 정렬 프로젝터, 그리고 언어 모델로 구성되어 있으며, 입력 데이터는 이 구조를 통해 처리됩니다. 각 단계에서 질문-답변 쌍이 포함되어 MLLM의 결정을 이끌어내는 데 사용됩니다.

- **Performance Highlights**: AgentPS는 TikTok 플랫폼의 비정상 콘텐츠 분류(UCC)에서 기존 MLLM 보다 F1 점수와 리콜에서 유의미한 성과 향상을 보여주고 있습니다. 또한, LLM 생성 프로세스 라벨을 사용할 경우에도 성능 향상이 유지되는 것을 확인하여, 대규모 산업 응용에 있어 실용적인 확장성을 입증하였습니다.



### An Enhanced Text Compression Approach Using Transformer-based Language Models (https://arxiv.org/abs/2412.15250)
- **What's New**: 이번 연구에서는 텍스트 복원에서 Transformer 기반의 방법, RejuvenateFormer를 제안합니다. 이는 Lempel-Ziv-Welch 알고리즘을 활용한 새로운 전처리 기법과 무손실 압축 방법을 결합하여 효과적인 압축 성능을 달성합니다. 이전 연구에서 다루지 않았던 무손실 압축 알고리즘과 Transformer 통합의 최적화 문제를 해결하고자 했습니다. 이 접근법은 상태-of-the-art (state-of-the-art) 압축 비율과 BLEU 점수를 달성하며, Transformer 모델을 활용한 새로운 기법들의 가능성을 열었습니다.

- **Technical Details**: 연구에서는 6개의 인코더 및 디코더 레이어로 구성된 맞춤형 아키텍처를 기반으로 하는 RejuvenateFormer를 개발하였습니다. 이 모델은 여러 말뭉치에서 다양한 일반 목적의 무손실 압축 기법을 통해 성능을 평가하고 압축 비율을 극대화하는데 초점을 맞추었습니다. 특히, 전처리 단계에서 Lempel-Ziv-Welch 알고리즘을 통합하여 압축 비율이 각각 12.57, 13.38, 11.42라는 성과를 보여주었습니다. 이를 통해 기존 심층 학습 및 전통적인 방법들과 비교하여 최고 수준의 압축 성능을 입증했습니다.

- **Performance Highlights**: RejuvenateFormer 모델은 EN-DE, EN-FR, BookCorpus 데이터셋에서 각각 27.31, 25.78, 50.45의 BLEU 점수를 기록했습니다. 이러한 성과는 기존의 T5-Small 모델을 포함한 이전의 모든 state-of-the-art 모델을 뛰어넘는 결과입니다. 외부 데이터셋에서의 엄격한 사례 분석을 통해 우리의 기법이 대규모 텍스트 복원 실험에도 효과적임을 입증하였습니다. 이번 연구는 텍스트 복원 및 압축 분야에서의 Transformer 기반 접근법의 새로운 가능성을 제시합니다.



### LLMs for Literature Review: Are we there yet? (https://arxiv.org/abs/2412.15249)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 문헌 리뷰 작성을 지원하는 새로운 접근 방식을 제안합니다. 특히, 본 연구에서 LLMs의 제로샷(zero-shot) 능력을 활용하여 초록을 기반으로 연관된 연구 결과를 검색하고 리뷰를 작성하는 두 가지 컴포넌트로 작업을 분해합니다. 혁신적인 두 단계 검색 전략과 리랭킹 메커니즘을 도입하여 LLM의 효과를 분석합니다.

- **Technical Details**: 연구는 초록에서 의미 있는 키워드를 추출하고, 외부 지식 기반에 쿼리하여 관련 논문을 검색하는 두 단계 검색 프로세스를 구현합니다. 또한, LLM이 후보 논문들에서 특정 발췌의 관련성을 밝힐 수 있도록 유도하는 프롬프트 기반 리랭킹 기법을 분석합니다. 리뷰 생성을 위해 LLM에게 어떤 논문을 인용할지를 지정하는 계획을 제공하는 방안을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론은 기존의 간단한 LLM 기반 생성 방법에 비해 18-26% 더 적은 환각된 참조(hallucinated references)를 생성하며, 품질 좋은 리뷰를 생성하는데 기여합니다. LLM 기반의 계획 수립 접근법이 문헌 리뷰의 품질을 실질적으로 개선함을 보여줍니다. 대조군보다 10%와 30% 향상된 정밀도와 정규화된 리콜(normalized recall)을 기록했으며, 이러한 방법은 연구 결과에 대한 투명성을 증가시킵니다.



### RoundTripOCR: A Data Generation Technique for Enhancing Post-OCR Error Correction in Low-Resource Devanagari Languages (https://arxiv.org/abs/2412.15248)
- **What's New**: 본 연구는 낮은 자원 언어에 대한 OCR(Optical Character Recognition) 오류 수정을 위한 데이터 생성을 다루고 있습니다. RoundTripOCR이라는 방법을 제안하며, 이는 OCR 출력 텍스트와 올바른 OCR 출력 텍스트 간의 맵핑을 학습하기 위해 기계 번역 기법을 활용합니다. 연구팀은 힌디어, 마라티어, 보도어, 네팔어, 콘카니, 산스크리트어에 대한 포스트 OCR 텍스트 수정 데이터셋을 공개했습니다.

- **Technical Details**: Devanagari 스크립트는 인도 아대륙에서 가장 널리 사용되는 문자 체계로, 텍스트 인식 과정에서 다양한 오류가 발생할 수 있습니다. 이 연구에서는 OCR 오류를 오역(mistranslation)으로 간주하고, 이를 교정하기 위해 사전 학습된 Transformer 모델을 활용하여 오류 텍스트와 올바른 텍스트 쌍 간의 매핑을 학습합니다. 새로운 방법론은 OCR을 처리하는 데 있어 일반적인 전처리, 기능 추출, 문자 분할 과정의 한계를 극복할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 제안된 RoundTripOCR 시스템은 기존의 OCR 시스템이 직면한 오류 유형을 효과적으로 해결할 수 있는 가능성을 보여줍니다. 연구팀은 3.1백만 개의 힌디어 문장, 1.58백만 개의 마라티어 문장 등의 포스트 OCR 오류 수정 데이터셋을 마련했습니다. 이 데이터셋은 다양한 언어에 대해 성능을 평가할 수 있는 기준점을 제공하며, 향후 연구에 중요한 자원으로 활용될 것입니다.



### Streamlining Systematic Reviews: A Novel Application of Large Language Models (https://arxiv.org/abs/2412.15247)
- **What's New**: 이번 연구는 시스템적 리뷰(Systematic Reviews, SR)에서 문헌 스크리닝(literature screening)의 자동화를 위한 사내 시스템을 제안합니다. 이 시스템은 Large Language Models (LLMs)을 기반으로 하여 제목/초록(title/abstract) 및 본문(full-text) 스크리닝 자동화를 목표로 하였습니다. 기존에 시간과 자원이 많이 소요되었던 의료 문헌 스크리닝 과정에서의 중요한 공백을 해소하기 위해 개발되었습니다.

- **Technical Details**: LLM 기반 시스템은 제목/초록 스크리닝에 대한 프롬프트 엔지니어링(prompt engineering)과 본문 스크리닝에 대한 Retrieval-Augmented Generation (RAG) 기법을 사용합니다. 이 시스템은 14,439개의 문헌을 포함한 비타민 D와 낙상에 대한 완전한 SR을 사용하여 평가되었으며, 99.5%의 기사 제외율(article exclusion rate)과 99.6%의 특이도(specificity)를 기록했습니다. 결국 78개의 기사가 수동 검토(manual review)가 필요하였고, 이는 전통적인 방법으로 식별된 20개를 포함합니다.

- **Performance Highlights**: LLM 기반 시스템은 총 스크리닝 시간을 25.5시간으로 줄이면서도 높은 정확도를 유지하였습니다. 상용 도구인 Rayyan과 비교했을 때 AER은 72.1%였고, FNR은 5%로 나왔습니다. 특히, LLM이 전통적인 방법보다 신뢰성과 효율성을 크게 개선했음을 보여주며, 전체 스크리닝 프로세스에 있어 자동화 도구의 부족 문제를 해결하고 SR 워크플로우에서 LLM의 변혁적인 잠재력을 강조합니다.



### Accelerating Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15246)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하고 정확도를 향상시키기 위한 솔루션으로 Retrieval-Augmented Generation (RAG) 접근법을 제안합니다. RAG는 LLM과 외부 지식 소스(예: 웹)에서 검색된 정보를 결합하는 방식으로 동작합니다. 저자들은 RAG의 실행 파이프라인을 분석하고, 고품질 검색을 위한 Intelligent Knowledge Store (IKS)라는 새로운 CXL 메모리 확장을 소개합니다.

- **Technical Details**: IKS는 새로운 캐시 일관성 인터페이스를 가진 고성능, 고용량 벡터 데이터베이스 가속기입니다. IKS는 정확한 최근접 이웃 검색(Exact Nearest Neighbors Search, ENNS)을 가속화하여 512GB 벡터 데이터베이스에서 13.4-27.9배 빠른 검색 성능을 제공합니다. 이 시스템은 CPU와 근처 메모리 가속기 간의 효율적인 인터페이스를 구현하여 메모리를 분산시키면서도 성능을 극대화하는 설계를 특징으로 합니다.

- **Performance Highlights**: IKS는 벡터 데이터베이스 애플리케이션에서 1.7-26.3배의 엔드 투 엔드 추론 속도 향상을 이끌어 내며, 이는 RAG 애플리케이션에서 대표적으로 관찰되는 성능 개선입니다. 본 연구는 RAG의 다양한 하드웨어 및 소프트웨어 구성이 성능과 정확도에 미치는 영향을 심도 있게 평가합니다. 논문에서 제시된 IKS는 기존의 메모리 시스템의 한계를 극복하기 위한 중요한 진전을 의미합니다.



### MPPO: Multi Pair-wise Preference Optimization for LLMs with Arbitrary Negative Samples (https://arxiv.org/abs/2412.15244)
Comments:
          Accepted by COLING2025

- **What's New**: 본 연구에서는 Multi Pair-wise Preference Optimization (MPPO) 알고리즘을 도입하여 대량의 언어 모델(LLM)과 사람의 피드백을 효율적으로 정렬할 수 있는 방법을 제안합니다. 기존의 DPO 및 KTO 알고리즘과 달리 MPPO는 보상 모델을 사용하지 않고도 정책 모델을 직접 최적화하여 여러 응답을 효과적으로 활용합니다. 이러한 접근 방식은 모델의 응답 품질을 개선하고, 더 많은 선호 데이터를 최대한 활용할 수 있도록 돕습니다.

- **Technical Details**: MPPO는 모델의 응답 평균 가능성을 활용하여 보상 함수를 피팅(fitting)합니다. 이 연구에서는 Point-wise, Pair-wise, List-wise의 세 가지 주요 구현 방식을 분석한 결과, Pair-wise 방식이 최고의 성능을 달성하는 것을 발견했습니다. 이는 여러 응답을 하나의 쿼리에 대해 최적화하여 희소 데이터 시나리오에서도 효과적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 MPPO는 MT-Bench에서 DPO, ORPO 및 SimPO를 초월하는 성과를 보였으며, Arena-Hard에서는 DPO와 ORPO에 비해 상당한 이점을 나타냈습니다. 이러한 결과는 MPPO가 선호 최적화 작업에서 큰 장점을 보여줌을 강조합니다. MPPO의 실험은 실제 응용 프로그램에서 최적화를 지원하며, 모델 응답의 질을 크게 향상시킵니다.



### Script-Based Dialog Policy Planning for LLM-Powered Conversational Agents: A Basic Architecture for an "AI Therapist" (https://arxiv.org/abs/2412.15242)
Comments:
          9 pages, 5 figures, 1 table

- **What's New**: 이번 연구는 LLM(대형 언어 모델) 기반의 대화 에이전트가 대화 중 치료적 접근 방식을 따를 수 있도록 구성된 새로운 'Script-Based Dialog Policy Planning' 프레임워크를 제안합니다. 이 시스템은 예측 가능한 행동을 통해 감정 지원 대화를 직접적으로 개선할 수 있습니다. 연구자들은 이를 통해 대화의 흐름을 정의하는 '스크립트'를 활용하여 인공지능 치료사가 보다 체계적으로 사용자와 상호작용할 수 있게 합니다.

- **Technical Details**: 연구에서는 LLM 기반의 대화 에이전트가 효과적이고 안전하게 작동할 수 있도록 다섯 가지 핵심 요구 사항을 정의합니다. 이 요구 사항에는 대화 유창성(conversational fluency), 적극성(proactivity), 전문가 개발(expert development), 증거 기반 관행(application of evidence-based practices), 그리고 검사 가능성(inspectability)이 포함됩니다. '스크립트'라는 고정된 텍스트를 사용하여 대화 흐름을 정의하면 LLM의 행동을 제어하고 전문성을 높여보고자 합니다.

- **Performance Highlights**: 연구 결과, 스크립트 기반의 대화 정책 계획을 사용하는 100개 대화의 성과를 시뮬레이션하여 이 접근 방식의 가능성을 입증하였습니다. 각각의 변형이 대화의 효율성과 효과성을 어떻게 달성하는지를 비교할 수 있는 기준을 설정하고, 성과의 강점과 약점을 논의하여 향후 개선 방향을 제시합니다. 이 새로운 기술의 타당성을 강조하며, 인공지능 치료사의 개발에 중요한 발판을 마련하고자 합니다.



### Quantifying Positional Biases in Text Embedding Models (https://arxiv.org/abs/2412.15241)
Comments:
          13 pages, 11 figures, NeurIPS

- **What's New**: 이번 연구는 정보 검색(Information Retrieval, IR)과 의미 유사성 측정에서 중요하게 사용되는 embedding 모델의 한계, 특히 긴 텍스트와 관련된 위치 편향 처리에 대해 다룹니다. content position과 input size가 text embedding에 미치는 영향을 실험을 통해 조사하였으며, embedding 모델들이 입력의 시작 부분을 불균형적으로 우선시하는 경향을 발견했습니다. 특히 문서의 처음에 무관한 텍스트를 삽입하거나 삭제하는 실험을 통해, 이러한 변화가 cosine similarity에 미치는 영향을 정량화하였습니다.

- **Technical Details**: embedding 모델은 transformer encoder 아키텍처를 기반으로 하여 bidirectional self-attention block을 사용합니다. 이 모델들은 고정된 길이의 벡터를 생성하여 전체 입력 텍스트를 표현하며, cosine similarity를 사용해 embedding을 비교합니다. 연구는 Absolute Positional Embedding(APE)과 Rotary Positional Embedding(RoPE), Attention with Linear Biases(ALiBi)와 같은 다양한 positional encoding 기법을 검토하여 이들이 모델의 내재적 편향에 어떻게 기여하는지를 설명합니다.

- **Performance Highlights**: 모델의 문서간 cosine similarity 측정 결과, 텍스트의 처음 부분에 무관한 삽입이 시뮬레이션된 경우, 유사성 감소가 중간이나 끝에 삽입한 경우보다 평균 8.5% 및 12.3% 더 크게 나타났습니다. 문서의 초기 문장에서 멀어질수록 회귀 계수가 크게 감소하는 것을 통해, 모델이 초기 내용에 불균형적으로 가중치를 부여한다는 점을 확인했습니다. 이러한 발견은 실제 검색 시스템의 민감도를 정량화하며, 모델의 강건성 향상 방향에 대한 새로운 관점을 제시합니다.



### ChainStream: An LLM-based Framework for Unified Synthetic Sensing (https://arxiv.org/abs/2412.15240)
Comments:
          18 pages, 8 figures

- **What's New**: 본 연구는 자연어를 개인 데이터 접근 및 처리의 단일 인터페이스로 사용하여 컨텍스트 인식 애플리케이션의 개발을 용이하게 하는 방안을 제안합니다. 이를 통해 개발자는 새로운 API를 배우지 않고도 자연어로 컨텍스트 인식 프로그램을 구축할 수 있으며, 사용자는 개발자가 쓴 자연어 데이터 쿼리를 직접 읽고 허가 관련 결정을 내릴 수 있어 데이터 처리 파이프라인의 투명성을 높입니다. 이러한 접근 방식은 최종 사용자 프로그래밍을 가능하게 하고, 더 정교한 사용 시나리오를 유도할 수 있습니다.

- **Technical Details**: 우리는 자연어 기반의 컨텍스트 인식 프로그램 생성을 가능하게 하기 위해 양방향 접근 방식을 취합니다. 첫 번째 방향은 감지 프로그램을 간단하고 통합된 형태로 만들기 위한 스트림 스타일의 프로그래밍 프레임워크를 도입하였고, 두 번째 방향은 샌드박스 피드백에 의해 유도된 쿼리 최적화를 통해 자연어 쿼리를 더 유익하게 만드는 것입니다. 이를 통해 자연어 지시와 실제 컨텍스트 인식 코드 간의 간격을 효과적으로 줄이고, Stream 데이터 추상화를 기초로 다양한 규칙 기반 및 모델 기반 스트림 데이터 작업을 위한 함수를 설계했습니다.

- **Performance Highlights**: 우리는 133개의 컨텍스트 인식 작업을 포함한 벤치마크를 만들어 우리 접근 방식을 평가하였습니다. 평가 결과, 우리의 방법은 생성 품질에서 최고점을 달성하며, 기존 방법들을 약 33% 초과하는 성능을 보였습니다. 본 연구는 자연어로 정의된 컨텍스트 인식을 위한 첫 번째 연구로, 실행 가능한 감지 프로그램으로의 자연어 요청 변환을 위한 엔드-투-엔드 시스템을 구축했습니다.



### Modeling Story Expectations to Understand Engagement: A Generative Framework Using LLMs (https://arxiv.org/abs/2412.15239)
- **What's New**: 기존 데이터 분석의 한계를 극복하고 고객이 이야기와 어떻게 상호작용하는지를 이해하기 위한 새로운 프레임워크가 제안되었습니다. 이 연구는 고객이 이야기의 전개에 대한 기대와 불확실성을 모델링하여 내용의 참여를 예측하는 데 중점을 둡니다. 그 결과, 이야기의 예상되는 진행에 대한 다양성을 포함한 새로운 지표를 통해 효과적인 추천 및 마케팅 전략 수립이 가능해집니다.

- **Technical Details**: 제안된 방법은 크게 두 단계로 이루어져 있습니다: 이야기 상상 단계와 특성 추출 단계입니다. 이야기 상상 단계에서는 미리 훈련된 대규모 언어 모델(LLM)을 사용하여 초기 텍스트를 기반으로 여러 개의 가능한 이야기 진행을 생성합니다. 그런 다음, 이러한 상상된 이야기에서 기대, 불확실성 및 놀람과 관련된 특성을 추출하여 기존의 이야기에 기반한 특성 추출 기법을 보완합니다.

- **Performance Highlights**: 이 연구는 Wattpad에서 수집한 30,258개의 책 챕터에 이 방법을 적용하여, 제안된 프레임워크가 기존 특성 엔지니어링 기법보다 평균 31% 더 높은 설명력을 제공한다는 것을 입증했습니다. 내부 회귀 분석을 통해 고객의 기대치가 이야기에 대한 참여도에 미치는 영향을 알고리는 초기 단서를 발견할 수 있으며, 이 결과는 마케팅 전략 및 콘텐츠 개발에 유용한 통찰을 제공합니다.



### Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks (https://arxiv.org/abs/2412.15238)
Comments:
          Accepted to NeurIPS 2024 Workshop on Foundation Model Interventions (MINT)

- **What's New**: 이번 연구에서는 기존의 LLM(대형 언어 모델)에서 발생하던 추론 과제와 관련된 약점을 해결하기 위한 새로운 프레임워크인 Dipper를 제안합니다. Dipper는 학습이 필요 없는 LLM 앙상블(ensemble) 방식으로, 단일 LLM 모델에 최적화된 다양한 프롬프트를 동시에 제공하여 추론 시간에 성능을 개선하도록 설계되었습니다. 이를 통해 사용자는 통합된 방식으로 여러 개의 쿼리를 처리할 수 있게 됩니다.

- **Technical Details**: 연구에서는 Dipper 프레임워크를 통해 각기 다른 프롬프트를 동시에 입력하여 LLM의 출력을 다양화하고자 합니다. 이 방법은 LLM의 고유한 특성인 다양한 출력을 생성하는 능력을 활용하여 성능 향상을 꾀합니다. 특히, 동종 모델을 사용하여 모델의 다양성을 증대시키는 것이 특징입니다.

- **Performance Highlights**: Dipper를 통해 한정된 GPU 메모리 환경에서도 성능을 향상시키는 실험 결과가 도출되었습니다. 예를 들어, MATH 데이터세트에서 3개의 작은 모델(Qwen2-MATH-1.5B-it 모델들)로 구성된 앙상블이 보다 큰 모델(Qwen2-MATH-7B-it)을 능가하는 결과를 보여주었습니다. 이는 Dipper가 실제 문제 해결에서 매우 유용하다는 것을 시사합니다.



### CareBot: A Pioneering Full-Process Open-Source Medical Language Mod (https://arxiv.org/abs/2412.15236)
Comments:
          Accept by AAAI 2025

- **What's New**: 최근 Closed-source LLM들과 오픈소스 커뮤니티들이 큰 발전을 이루어내며 인간의 성능을 초월하고 있지만, 의료처럼 전문적인 도메인에서는 여전히 만족스럽지 않은 성능을 보이고 있습니다. 본 논문에서는 CareBot이라는 이중 언어 의료 LLM을 제안하며, 이는 Continuos Pre-Training (CPT), Supervised Fine-Tuning (SFT), 그리고 Reinforcement Learning with Human Feedback (RLHF)를 통합하여 개발되었습니다. 특히, 일반 데이터와 도메인 특정 데이터의 간극을 메우기 위한 새로운 두 단계 CPT 방법론을 도입합니다.

- **Technical Details**: 이 논문에서 제안된 CareBot LLM은 LLaMA3-8B 기반으로 하며, 의사에게 진단 보조, 개인 맞춤형 치료 계획 제공, 의료 교육 지원 등을 효과적으로 지원할 수 있도록 설계되었습니다. 특히, Stable CPT와 Boost CPT라는 두 단계의 CPT 방법론을 통해 일반 데이터와 도메인 특정 데이터 간의 분포 불일치를 해결하며, 데이터 품질 평가 모델인 DataRater를 개발하여 CPT 동안의 학습 데이터의 정확도와 관련성을 보장합니다. 이는 다큐멘터리한 의료 SFT 데이터셋과 다회차 대화 품질 향상을 위한 ConFilter 메트릭을 포함합니다.

- **Performance Highlights**: CareBot의 성능 평가는 중국어 및 영어 기준에서 진행되었으며, 의료 상담 및 교육에서 뛰어난 성능을 보였습니다. 실험 결과, CareBot이 다수의 의료 애플리케이션에서 우수한 성능을 발휘하며, 본 연구의 데이터셋 구성과 훈련 전략이 모델 성능에 긍정적인 영향을 미쳤음을 확인했습니다. 이러한 발전은 의료 LLM의 현재 한계를 극복하고, 오픈소스 모델의 신뢰성과 효과성을 위한 새로운 기준을 설정하는 데 기여할 것입니다.



### OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models (https://arxiv.org/abs/2412.15235)
- **What's New**: OG-RAG(온톨로지 기반 검색 향상 생성) 방법은 LLM 생성 응답의 품질을 향상시키기 위해 도메인 특정 온톨로지에 기반한 검색 프로세스를 통합합니다. 이 방법은 일반적인 검색 증강 모델들의 한계, 즉 구조화된 도메인 지식을 반영하지 못하는 점을 해결하고, 특히 전문화된 지식이 요구되는 산업 워크플로우 및 의사결정 단계에서 유용합니다.

- **Technical Details**: OG-RAG는 도메인 문서의 하이퍼그래프 표현을 구성하고, 각 하이퍼에지는 도메인 특정 온톨로지를 기반으로 한 사실 지식의 클러스터를 포괄합니다. 이 최적화 알고리즘은 LLM을 위한 정확하고 개념적으로 기반을 둔 맥락을 형성하기 위해 최소한의 하이퍼에지를 검색합니다. 이를 통해 복잡한 엔티티 간의 관계를 유지하면서도 효율적인 검색을 가능하게 합니다.

- **Performance Highlights**: OG-RAG는 4개의 다른 LLM에서 정확한 사실의 리콜을 55% 향상시키고, 응답의 정확성을 40% 증가시켰습니다. 또한, LLM 응답을 맥락과 빠르게 연결하는 데 30% 더 빠르며, 사실 기반 추론의 정확성을 다른 기법에 비해 27% 향상시킵니다. 이러한 성과는 OG-RAG의 전문화된 워크플로우에서의 신뢰성 있는 사실 기반 응답 제공 능력을 강조합니다.



### Offline Reinforcement Learning for LLM Multi-Step Reasoning (https://arxiv.org/abs/2412.16145)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 다단계 추론 능력을 개선하기 위한 오프라인 강화 학습(offline reinforcement learning) 방법인 OREO(Offline Reasoning Optimization)를 제안합니다. 기존의 Direct Preference Optimization(DPO) 방법의 한계점을 극복하고, 다단계 추론에 적합한 새로운 접근 방식을 제공합니다.

- **Technical Details**: OREO는 최대 엔트로피 강화 학습(maximum entropy reinforcement learning)의 통찰을 바탕으로 소프트 벨만 방정식(soft Bellman Equation)을 최적화하여 정책 모델(policy model)과 가치 함수(value function)를 공동 학습합니다. 이는 데이터 수집의 부담을 줄이고, 다단계 추론에서 효과적인 신용 할당(credit assignment)을 가능하게 합니다.

- **Performance Highlights**: OREO는 수학적 추론 작업(GSM8K, MATH) 및 임베디드 에이전트 제어(ALFWorld)와 같은 다단계 추론 벤치마크에서 기존의 오프라인 학습 방법들을 능가하는 성능을 보였습니다. 이 방법은 추가 자원이 있을 경우 다중 반복(multi-iteration) 프레임워크로 확장할 수 있으며, 학습된 가치 함수를 통해 무료로 트리 탐색(tree search)을 안내하여 테스트 시 성능을 더욱 향상시킬 수 있습니다.



### Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation (https://arxiv.org/abs/2412.16135)
Comments:
          To appear in AAAI 2025, Main Track

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 신선한 obfuscated assembly 코드를 생성할 수 있는 가능성에 대해 탐구합니다. 기존의 obfuscation 도구는 원본 코드에 대한 접근이 필요하며, 새로운 obfuscation을 추가하는 것이 복잡하고 시간이 많이 소요됩니다. 저자들은 MetamorphASM 데이터셋(MAD)과 함께 세 가지 obfuscation 기술(즉, dead code, register substitution, control flow change)을 통한 평가를 통해 LLM이 obfuscated 코드를 생성할 수 있는 능력을 평가합니다.

- **Technical Details**: 저자들은 MetamorphASM 데이터셋을 사용하여 328,200개의 obfuscated assembly 코드 샘플을 생성했습니다. 이 데이터셋은 LLM의 obfuscation 능력을 검증하기 위한 첫 번째 assembly 코드 obfuscation 데이터셋입니다. 연구는 정보 이론적 메트릭스와 수작업 검사 방법으로 LLM의 성공률을 평가했으며, 다양한 LLM 모델(GPT-3.5/4, Starcoder 등)의 능력도 개별적으로 검토하였습니다.

- **Performance Highlights**: 연구 결과 LLM은 obfuscated assembly 코드 생성을 위한 여러 다른 기술에서도 적절한 성과를 보였습니다. 특히, dead code insertion, register substitution, control flow change와 같은 특정 obfuscation 기술에 따라 성능이 달라졌습니다. 이 연구는 LLM이 저비용과 플랫폼 독립성을 통해 고급 obfuscation 전략을 활용할 수 있음을 보여줍니다.



### Towards Interpretable Radiology Report Generation via Concept Bottlenecks using a Multi-Agentic RAG (https://arxiv.org/abs/2412.16086)
Comments:
          Accepted in ECIR 2025

- **What's New**: 이 연구는 Deep learning을 활용하여 Chest X-ray (CXR) 분류에서 해석 가능성(interpretability)을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히, concept bottleneck models (CBMs)와 multi-agent Retrieval-Augmented Generation (RAG) 시스템을 결합하여 임상적 관련성을 지닌 방사선 보고서를 생성합니다. 이러한 방법은 모델의 예측을 인간이 이해할 수 있는 방식으로 명확하게 제시하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 접근법은 두 단계를 통해 이루어집니다. 첫 번째 단계에서는 질병 분류와 관련된 개념 기여도를 계산하고, 두 번째 단계에서는 임상 문서와 설명을 활용하여 견고한 보고서를 생성합니다. 모델은 GPT-4를 사용하여 각 질병 범주에 대한 자동 개념 발견을 수행하고, 이미지 임베딩(ChexAgent 모델 사용)과 텍스트 임베딩(Mistral Embed Model 사용)을 결합하여 개념 벡터를 생성합니다.

- **Performance Highlights**: COVID-QU 데이터셋에서 본 모델은 81%의 분류 정확도를 기록하였으며, 생성된 보고서는 84%에서 90% 사이의 성능 메트릭을 보였습니다. 이는 AI 기반 CXR 분석의 신뢰성을 높이는 해석 가능성과 높은 성능 간의 갭을 메우는 다중 에이전트 프레임워크를 구축하는 데 기여합니다.



### Align Anything: Training All-Modality Models to Follow Instructions with Language Feedback (https://arxiv.org/abs/2412.15838)
- **What's New**: 이 연구는 여러 모드를 다루는 대규모 언어 모델의 성능을 향상시킬 수 있는 새로운 방법론을 제시합니다. 특히, RLHF(Reinforcement Learning from Human Feedback)를 여러 모드에 확장하여, 다양한 형태의 데이터(텍스트, 이미지, 오디오, 비디오)에서 인간의 선호에 기반한 정교한 모델 조정 접근법을 모색합니다. 이 연구는 이러한 다중 모드 문제를 다루기 위한 새로운 'align-anything' 프레임워크와 첫 번째 'eval-anything' 평가 체계를 제안합니다.

- **Technical Details**: 논문에서는 'align-anything-200k'라는 이름의 대규모 다중 모드 인간 선호 데이터셋을 제시하며, 이는 텍스트, 이미지, 비디오, 오디오 등 다양한 모드에 대해 사용자 선호를 캡쳐합니다. 기존의 데이터셋 한계를 극복하기 위해 2단계 인간 주석 과정을 통해 구축되었습니다. 또한 'learning from language feedback'라는 새로운 알고리즘을 통해 각 모드에 대한 RLHF 성능을 개선하며, 이는 평균 5.83배의 성능 향상을 이룹니다.

- **Performance Highlights**: 제안된 모델은 다중 모드 평가 도구인 'eval-anything'을 기반으로 성능 개선을 추적합니다. 이를 통해 다중 모드 모델이 더 높은 인스트럭션-팔로잉 능력을 발휘할 수 있도록 지원합니다. 더욱이, 연구진은 모든 데이터와 모델은 오픈 소스 형태로 공개하여, 연구 및 개발 커뮤니티가 쉽게 활용할 수 있도록 하고 있습니다.



### Enriching Social Science Research via Survey Item Linking (https://arxiv.org/abs/2412.15831)
- **What's New**: 이 연구는 사회과학 설문조사에서 사용되는 설문 항목(survey items)의 내재된 개념을 연구하기 위해 각각의 설문 항목을 효과적으로 연관시키는 새로운 방법론을 제시합니다. 기존의 연구에서는 설문 항목을 직접 인용하는 대신 내용을 간접적으로 표현하는 방식이 일반적이지만, 이는 관련 연구를 비교하는 데 어려움을 초래합니다. 연구자는 이 문제를 해결하기 위해 설문 항목 연결(Survey Item Linking, SIL) 작업을 두 단계로 모델링하고 있습니다.

- **Technical Details**: SIL 작업은 언급 감지(mention detection) 및 개체 중의성 해소(entity disambiguation)라는 두 단계로 나뉘어 집니다. 하지만 이 작업에 대한 정의가 불분명하여, 기존의 데이터셋은 평과기준이 낮고 너무 소규모라는 문제가 있었습니다. 연구자들은 20,454개의 영어 및 독일어 문장으로 구성된 고품질의 풍부한 주석 데이터셋을 개발하여 이러한 문제를 해결하고자 합니다.

- **Performance Highlights**: 연구에서 사용된 딥러닝 시스템을 통해 두 단계를 독립적이고 순차적으로 벤치마킹한 결과, 이 작업이 실행 가능함을 입증하였습니다. 그러나 언급 감지 단계에서 발생한 오류는 전체적인 작업 성능 저하로 이어졌으며, 여러 문장의 맥락이 필요한 언급을 감지하는 것은 특히 도전적입니다. 향후 연구에서는 문서의 전체 맥락을 모델링하고 두 단계를 통합한 시스템을 적용하여 이러한 문제를 해결할 수 있을 것입니다.



### S$^2$DN: Learning to Denoise Unconvincing Knowledge for Inductive Knowledge Graph Completion (https://arxiv.org/abs/2412.15822)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 Inductive Knowledge Graph Completion(KGC)를 위한 새로운 S$^2$DN(Semantic Structure-aware Denoising Network) 네트워크를 제안합니다. 이 네트워크는 지식 그래프(KG) 내에서 새롭게 등장하는 개체들 간의 누락된 사실을 추론하는 과정을 개선하는 것을 목표로 합니다. 특히, S$^2$DN은 비슷한 관계의 의미적 불일치와 KG에서의 노이즈를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: S$^2$DN은 관계의 일반적인 의미를 유지하기 위해 포괄하는 서브그래프에 대한 의미적 스무딩 모듈을 도입합니다. 또한, 신뢰할 수 없는 상호작용을 필터링하고 추가적인 지식을 제공하기 위해 구조 정련 모듈을 통합하여 KG 내에서의 신뢰할 수 있는 상호작용을 유지합니다. 이러한 구조적인 정제 및 의미적 스무딩 접근 방식은 KG의 신뢰성과 일관성을 지속적으로 높이는 데 기여합니다.

- **Performance Highlights**: S$^2$DN은 다양한 KG와 서로 다른 노이즈 조건에서 GraIL 모델을 초월하는 예측 성능을 보였습니다. 경험적 실험을 통해 S$^2$DN은 KG 내에서 의미적 일관성을 유지하고 불확실한 상호작용을 효과적으로 필터링하는 능력을 입증하였습니다. 이러한 결과는 S$^2$DN이 KGC 분야에서 뛰어난 성과를 가지고 있음을 보여줍니다.



### AutoLife: Automatic Life Journaling with Smartphones and LLMs (https://arxiv.org/abs/2412.15714)
Comments:
          13 pages

- **What's New**: 이 논문은 사용자의 일상 생활의 의미 있는 설명을 자동으로 생성하는 새로운 모바일 감지 응용 프로그램인 'Life Journaling'을 소개합니다. 본 시스템인 AutoLife는 상업용 스마트폰을 기반으로 하며, 사진이나 오디오 없이 저비용 센서 데이터를 입력으로 사용하여 사용자의 생활 일기를 자동으로 생성할 수 있습니다. AutoLife는 다양한 센서 데이터로부터 시간, 위치, 동작 맥락을 추출하고, 인간 생활에 대한 상식 지식으로 풍부해진 대형 언어 모델(LLM)의 제로 샷 능력을 활용합니다.

- **Technical Details**: AutoLife는 임의의 사용자 입력 없이 스마트폰의 센서 데이터를 기반으로 생활 일기를 생성합니다. 이를 위해 시간과 위치를 포함한 여러 맥락 정보를 결합하여 양질의 생명 저널을 생성하는 데 중점을 두고 있습니다. 데이터 수집의 효율성을 위해 다층 프레임워크를 설계하였으며, 긴 기간 동안의 센서 데이터를 분할하고 정제하여 최종적으로 LLM에 전달합니다.

- **Performance Highlights**: AutoLife 시스템은 홍콩의 자발적인 3명으로부터 수집한 다양한 행동 데이터셋을 통해 평가되었습니다. 실험 결과, Claude 3와 같은 일부 LLM을 사용했을 때 평균 BERTScore F1이 0.7 이상을 달성하는 등의 높은 정확도를 보여주었습니다. 이 연구는 향후 관련 연구의 벤치마크로 사용할 수 있는 공개 데이터셋을 제공할 예정이며, 기존의 생활 저널링 시스템과는 차별화된 방법론을 보여줍니다.



### Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration (https://arxiv.org/abs/2412.15701)
Comments:
          Preprint. Work in progress

- **What's New**: 최근 언어 모델(LM)에서의 발전은 LM 에이전트 개발에 대한 관심을 높이고 있습니다. 완전 자율 에이전트가 많은 상황에서 우수할 수 있지만, 인간의 잠재적 선호(preferences)나 분야 전문성(expertise), 통제(control) 필요성 때문에 많은 경우에는 인간과의 협업이 필수적입니다. 이를 위해 우리는 인간-에이전트 협업 연구를 용이하게 하는 일반적인 프레임워크인 Collaborative Gym (Co-Gym)을 제안합니다.

- **Technical Details**: Co-Gym은 에이전트, 인간 및 작업 환경 간의 비동기(asynchronous) 삼자 간 상호작용을 가능하게 하는 프레임워크입니다. 이 프레임워크는 시뮬레이션 및 실제 환경에서의 세 가지 대표 작업으로 구체화되며, 협업 결과 및 과정을 평가하는 평가 프레임워크를 제안합니다. 연구 결과, 협업 에이전트는 실제 사용자 평가에서 86%의 승률을 기록하는 여행 계획(Travel Planning), 74% 를 기록하는 표형 분석(Tabular Analysis), 66% 를 기록하는 관련 작업(Related Work) 등에서 완전 자율 에이전트보다 일관되게 우수한 성능을 보였습니다.

- **Performance Highlights**: 협업 에이전트는 특정 작업 상황에서 완전 자율 에이전트보다 더 나은 성과를 나타내었습니다. 그러나 연구는 협업 에이전트를 개발하는 데 있어서 커뮤니케이션 능력, 상황 인식(situational awareness), 자율성과 인간 통제의 균형 조절 등 핵심 인텔리전스(aspects of intelligence)의 발전이 필요하다는 중요한 도전 과제를 강조합니다.



### Adaptable and Precise: Enterprise-Scenario LLM Function-Calling Capability Training Pipelin (https://arxiv.org/abs/2412.15660)
Comments:
          23 pages, 6 figures, 7 tables

- **What's New**: 본 논문에서는 실제 비즈니스 환경에 적합한 기능 호출 모델을 위한 훈련 파이프라인을 제안합니다. 이 파이프라인은 시나리오별 기능 호출 데이터의 합성과 증강, 모델 미세 조정, 성능 평가 및 분석을 포함합니다. 이를 통해 디지털 HR 에이전트 시나리오에서 1,260개의 AI 생성 샘플과 1,035개의 수동 라벨링 샘플을 생성하였습니다. 연구 결과, 우리는 Qwen2.5-Coder-7B-Instruct 모델을 기반으로 GPT-4 및 GPT-4o보다 뛰어난 성능을 달성했습니다.

- **Technical Details**: 이 연구에서는 전문적인 시나리오에 맞춘 기능 호출 능력을 위한 자동화된 훈련 파이프라인을 설계하였습니다. 파이프라인에는 데이터 합성과 증강, LoRA를 이용한 SFT(Supervised Fine-Tuning), 모델 성능 평가 및 분석이 포함됩니다. Qwen2.5-Coder-7B-Instruct 모델을 4대의 GPU에서 LoRA 방법으로 미세 조정하였으며, 이 과정은 약 5시간 이내에 완료되었습니다.

- **Performance Highlights**: 미세 조정된 모델은 테스트 세트에서 구조적 완결성, 도구 선택 정확성 및 매개변수 입력 정확성을 넘어서는 우수한 성능을 보였습니다. 이러한 결과는 제안된 파이프라인의 효과를 입증하며, 중형 LLM에서의 기능 호출 가능성을 향상시키는 데 기여할 것입니다. 또한, 이 연구는 중소기업이 자신의 요구에 맞춘 에이전트 모델을 쉽고 효율적으로 훈련하고 배포할 수 있는 가능성을 제시하였습니다.



### TouchASP: Elastic Automatic Speech Perception that Everyone Can Touch (https://arxiv.org/abs/2412.15622)
Comments:
          Technical Report

- **What's New**:  이 논문에서는 큰 자동 음성 인식(ASR) 모델의 고비용과 제한된 기능을 해결하기 위해, 하나의 훈련 후 다양한 배치 요구에 맞춰 탄력적으로 조정할 수 있는 "elastic mixture of experts (eMoE)" 모델을 제안합니다. 이 모델은 음성 인식 작업뿐만 아니라 다국어 및 감정 인식 등 여러 작업에서 성능을 발휘하는 'Automatic Speech Perception (ASP)' 프레임워크로 확장됩니다.

- **Technical Details**:  eMoE 모델은 훈련 중에는 동적 전문가 전략을 이용하고, 추론 시에는 장치의 능력에 맞게 전문가 수를 조정합니다. 이를 통해 훈련 비용을 절감하면서도 높은 인식 정확도를 유지할 수 있습니다. 연구팀은 또한 간단하고 재현 가능한 데이터 처리 파이프라인을 설계하여 1,000k 시간의 음성 데이터를 수집하였습니다.

- **Performance Highlights**:  제안된 시스템은 SpeechIO 테스트 세트에서 Character Error Rate (CER)을 4.98%에서 2.45%로 줄이는 성과를 거두었습니다. 이는 모델이 단순한 음성 인식뿐만 아니라 다국어, 다방언 및 감정 인식 과제에서도 뛰어난 성능을 보여줌을 의미합니다. 실험 결과는 eMoE 모델이 다양한 음성 인식 작업에 대한 유연한 대응 능력을 갖추었다는 것을 증명합니다.



### Continual Learning Using a Kernel-Based Method Over Foundation Models (https://arxiv.org/abs/2412.15571)
- **What's New**: 본 논문은 지속적 학습(Continual Learning, CL) 중 클래스 증가 학습(Class-Incremental Learning, CIL)의 도전적인 설정을 다룹니다. 기존의 여러 방법에도 불구하고, 파라미터 업데이트로 인한 재앙적 망각(Catastrophic Forgetting, CF) 및 과제 간 클래스 분리(Inter-task Class Separation, ICS) 문제가 여전히 존재합니다. 이 문제를 해결하기 위해, Kernel Linear Discriminant Analysis (KLDA)라는 새로운 방법을 제안하며, 이 방법은 기초 모델(Foundation Model)에서 학습된 강력한 특징을 활용합니다.

- **Technical Details**: KLDA는 Radial Basis Function (RBF) 커널과 Random Fourier Features (RFF)를 통합해 기초 모델에서 추출된 특징 표현을 향상시킵니다. 새로운 작업이 도착하면 KLDA는 각 클래스의 평균을 계산하고, 커널화된 특징을 기반으로 모든 학습된 클래스에 대한 공유 공분산 행렬을 업데이트합니다. 이 방법은 Linear Discriminant Analysis (LDA)를 사용하여 분류를 수행하며, 각 클래스에 대한 가우시안 분포를 정의하여 결정 경계를 최적화합니다.

- **Performance Highlights**: KLDA는 텍스트 및 이미지 분류 데이터세트를 사용한 실험적 평가에서 기존 방법들보다 우수한 성능을 보였습니다. 특히, KLDA는 재생 데이터에 의존하지 않고도 CIL 성능의 상한으로 여겨지는 모든 클래스의 조합 훈련에 맞먹는 정확도를 달성하였습니다. 이는 기존의 다른 CIL 방법들이 모자란 정확도를 극복하는 데 중요한 의미를 갖습니다.



### MORTAR: Metamorphic Multi-turn Testing for LLM-based Dialogue Systems (https://arxiv.org/abs/2412.15557)
- **What's New**: 최근 LLM 기반 대화 시스템의 품질 보증의 중요성이 커지고 있습니다. 기존의 단일 턴 테스트 방식에 비해 다중 턴 대화 테스트는 충분히 탐구되지 않았으며, 이에 대한 해결책으로 MORTAR라는 새로운 방식이 제안되었습니다. MORTAR는 LLM 기반 대화 시스템의 테스트 오라클 문제를 완화하는 혁신적인 메타모픽(metamorphic) 대화 테스트 방법입니다.

- **Technical Details**: MORTAR는 질문-답변(QA) 대화 테스트 케이스의 자동 생성을 가능하게 하여 다중 대화 수준의 방해 및 메타모픽 관계를 적용합니다. 이 방법은 지식 그래프 기반의 대화 정보 모델을 활용해 비용 효율적으로 테스트 데이터셋을 생성하고 다중 턴 대화 시스템의 버그를 식별합니다. 이 과정에서 LLM을 평가자로 사용하지 않기 때문에 평가 단계에서의 편향을 제거합니다.

- **Performance Highlights**: 실험 결과, MORTAR는 다수의 LLM 기반 대화 시스템에서 더욱 독창적인 버그를 탐지하여 기존의 단일 턴 메타모픽 테스트 방법에 비해 최대 4배 더 많은 심각한 버그를 발견했습니다. 이는 MORTAR가 대화 시스템의 품질 보증을 위한 효율적인 도구가 될 수 있음을 시사합니다.



### TalkWithMachines: Enhancing Human-Robot Interaction for Interpretable Industrial Robotics Through Large/Vision Language Models (https://arxiv.org/abs/2412.15462)
Comments:
          This paper has been accepted for publication in the proceedings of the 2024 Eighth IEEE International Conference on Robotic Computing (IRC)

- **What's New**: 이 논문에서는 안전-critical(안전 중요) 산업에서 활용 가능한 해석 가능하고 인간-로봇 상호작용을 강화하기 위한 새로운 접근법을 제안합니다. 자연어(natural language)를 통해 로봇에 명령을 내리고 로봇이 주변 환경을 이해할 수 있도록 하기 위해 LLMs(대형 언어 모델)와 VLMs(비전 언어 모델)의 통합을 탐구하고 있습니다. 로봇의 내부 상태와 의도를 이해하기 쉽게 설명하는 방식으로, 보다 안전하고 효과적인 운영이 가능하도록 합니다.

- **Technical Details**: 본 연구는 로봇 조작(control) 및 인식(perception)에 대한 LLMs와 VLMs의 발전을 바탕으로 하며, 로봇이 이해할 수 있는 저수준의 제어 명령 패턴을 생성할 수 있는 잠재력을 가지고 있음을 보여줍니다. 로봇의 내부 및 외부 상태를 이미지나 텍스트 기반으로 표현하여, 복잡한 궤적의 생성 및 상황 인식을 위한 패턴 인터페이스를 소개합니다. 이를 통해 로봇의 물리적 한계와 안전한 명령 실행을 위한 인식 능력을 키울 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 LLMs와 VLMs가 복잡한 궤적을 생성하고, 환경의 맥락 인식을 유지하며, 간접적 의사소통 신호를 해석할 수 있는 능력이 있음을 보여줍니다. 또한, 로봇의 물리적 구조에 대한 정보가 안전한 명령 실행을 위한 인식에 미치는 영향을 분석하여, 해석 가능하고 인간 중심의 로봇 시스템 개발의 방향성을 제시합니다. 제안된 개념들은 로봇 팔 조작 시뮬레이션을 통해 검증되었습니다.



### Time Will Tell: Timing Side Channels via Output Token Count in Large Language Models (https://arxiv.org/abs/2412.15431)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)에서 민감한 정보를 추출할 수 있는 새로운 사이드 채널(side-channel)을 제안합니다. 이 사이드 채널은 LLM의 응답에서 출력 토큰 수를 기반으로 하여 정보 유출을 가능하게 합니다. 특히, 기계 번역 작업에서 타겟 언어를 복원하거나 분류 작업에서 출력 클래스를 복원하는 공격을 수행하는 데 사용됩니다.

- **Technical Details**: LLM의 자동 회귀 생성(auto-regressive generation) 특성을 활용하여 공격자는 응답 시간 측정 또는 네트워크를 통해 출력 토큰 수를 회복할 수 있습니다. 이 연구에서는 Tower, M2M100, MBart50과 같은 다국어 모델에서 75% 이상의 정밀도로 언어를 식별할 수 있음을 보여주었습니다. 추가적으로, 텍스트 분류 작업에서 LLM의 출력 설명 길이에서 발생하는 내재적 편견이 민감한 정보를 누출하는 경향이 있음을 입증했습니다.

- **Performance Highlights**: 실제로, LLM의 출력 토큰 수에 따라 기계 번역과 텍스트 분류 작업에서 높은 성공률을 기록했습니다. 예를 들어, Gemma2-9B 모델에서 81.4%의 성공률을 달성했으며, GPT-4o 모델의 원격 공격에서도 74.7%의 성공률을 보였습니다. 이러한 성과는 특정 작업의 출력 클래스나 언어를 유출하는 데 있어 사이드 채널 공격이 매우 효과적임을 입증합니다.



### Transcribing and Translating, Fast and Slow: Joint Speech Translation and Recognition (https://arxiv.org/abs/2412.15415)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 논문에서는 JSTAR라는 새로운 음성 인식(ASR) 및 음성 번역(ST) 모델을 제안합니다. JSTAR는 빠른-느린(cascaded) 인코더 구조를 활용하여 동시에 ASR과 ST 작업을 수행할 수 있도록 설계되었습니다. 이 모델은 멀티 목적(multi-objective) 훈련 전략을 통해 두 작업을 동시에 최적화하여 높은 품질의 실시간 ASR 및 ST 결과를 제공합니다.

- **Technical Details**: JSTAR 모델은 RNN-T 기반의 구조를 가지고 있으며, 입력 시퀀스를 여러개의 채널로 처리합니다. 고속 인코더는 저지연(低遲延) 구조로 구성되어 있으며, 느린 인코더는 보다 넓은 컨텍스트 크기를 가져 정확한 ST 결과를 생성합니다. 이를 통해 각각의 작업에 대해 별도의 예측(prediction) 및 조인(joiner) 모듈을 사용할 수 있습니다.

- **Performance Highlights**: JSTAR는 BLEU 점수와 지연(latency) 면에서 기존의 강력한 연쇄 ST 모델과 비교하여 우수한 성능을 보여줍니다. 또한, 스피커 구분 기능이 포함된 다채널 방향 ASR 솔루션을 통해 스마트 안경을 사용하여 이중 언어 대화를 효과적으로 인식하고 번역할 수 있음을 입증했습니다.



### Learning Visual Composition through Improved Semantic Guidanc (https://arxiv.org/abs/2412.15396)
- **What's New**: 이번 논문은 기존의 시각적 표현 학습 방식의 한계를 극복하기 위해, 간단하고 확장 가능한 접근 방식을 제안합니다. 기존의 CLIP 모델들이 인과(thought) 이해에 부족하다는 점을 지적하며, 비약적인 성능 향상을 위해 기존 캡션을 새로운 다중 모달 모델에 의해 개선한다고 설명합니다. 특히, 향상된 데이터 세트 기반 훈련을 통해 이미지 검색 작업에서 실질적인 성과를 내고 있음을 입증하고자 하였습니다.

- **Technical Details**: 제안된 방법론은 CLIP 아키텍처에 기반하여 두 가지 주요 수정 사항을 도입합니다. 첫 번째, 고품질의 영어 이미지-텍스트 쌍을 포함한 WebLI 데이터셋을 사용하여 캡션의 품질을 향상시켜 alt-text를 대체합니다. 둘째, 훈련된 다중 모달 모델의 텍스트 타워를 고성능의 텍스트 기반 기초 모델로 교체하여 시각적 임베딩 표현을 크게 개선합니다. 이러한 방법을 통해 94.5%의 명확성을 달성하며 판별 성능이 크게 향상되었습니다.

- **Performance Highlights**: 향상된 데이터 세트와 적절한 모델 수정으로, CLIP 모델의 이미지 검색 성능이 눈에 띄게 개선되었습니다. 특히, recall@1 메트릭에서 58.4%의 성능에서 94.5%로 향상되었습니다. 일반적인 captcha 값으로 ARO 평가 데이터를 통하여 비교적 단순한 다중모달 모델의 학습 성과가 나타나고, 새로운 벤치마크에 대한 필요성을 적시하며 실험 결과의 타당성을 강화합니다.



### SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkag (https://arxiv.org/abs/2412.15289)
- **What's New**: 본 논문에서는 Large Language Model(LLM)의 취약점을 효과적으로 회피하고 유해한 응답을 이끌어내는 새로운 탈옥 패러다임인 Simple Assistive Task Linkage(SATA)를 제안합니다. SATA는 악성 쿼리에서 유해한 키워드를 [MASK] 특수 토큰으로 마스킹한 후, 마스킹된 키워드의 의미를 인코딩하기 위해 간단한 보조 작업을 수행합니다. 이러한 접근법은 기존의 복잡한 지침이나 반복적인 접근 방식에 비해 탈옥 성능과 효율성을 크게 향상시킵니다.

- **Technical Details**: SATA는 두 가지 보조 작업, 즉 Masked Language Model(MLM)과 Element Lookup by Position(ELP)을 활용합니다. 마스킹된 쿼리와 보조 작업을 연결하여 LLM의 안전 검사 루틴을 우회하는 방식입니다. MLM 작업은 위키백과 항목을 문맥으로 이용하여 LLM이 내용 보강을 수행하도록 유도하고, ELP는 주어진 위치에 있는 요소를 식별하는 작업을 진행합니다.

- **Performance Highlights**: 실험 결과, SATA는 최첨단 성능을 기록하며, AdvBench 데이터셋에서 SATA-MLM은 85%의 공격 성공률(ASR)과 4.57의 해로운 점수(HS)를 달성했습니다. SATA-ELP도 각각 76%의 ASR과 4.43의 HS를 기록하여 기존의 다른 방법들과 비교해 현저히 개선된 성능을 보여줍니다.　또한, SATA-ELP는 입력 토큰 사용에서 이전 방법들보다 약 10배의 절약을 이룬 것으로 나타났습니다.



### Fooling LLM graders into giving better grades through neural activity guided adversarial prompting (https://arxiv.org/abs/2412.15275)
Comments:
          16 pages, 11 figures

- **What's New**: 이 논문은 인공지능(AI)이 주요 의사결정 및 평가 프로세스에 배치될 때 발생할 수 있는 내재된 편향을 드러내는 체계적인 방법을 제안합니다. 특히 자동 에세이 채점(automated essay grading) 시스템을 예로 들어, 악성 행위자들이 의사결정 결과를 왜곡할 수 있는 편향을 추적하고 분석합니다. 이를 통해 대규모 언어 모델(LLM) 채점기가 인간이 탁월한 점수를 부여하는 것보다 훨씬 높은 점수를 주도록 속일 수 있는 방법을 실증적으로 보여줍니다.

- **Technical Details**: 연구에 사용된 접근법은 왜곡된 의사결정 결과를 예측하는 숨겨진 신경 활동 패턴(neural activity patterns)을 식별하는 것에서 시작합니다. 이후 이러한 패턴을 증폭시키기 위해 적대적 입력 접미사(adversarial input suffix)를 최적화합니다. 연구에서는 또한 '매직 워드(magic word)'가 공격의 효능에 중요한 역할을 한다는 사실을 밝혀내었으며, 이는 LLM의 감독하에 미세 조정(supervised fine-tuning)에 자주 사용되는 채팅 템플릿(chat templates)의 구조에 기인합니다.

- **Performance Highlights**: 이 연구는 현재의 LLM에서 발견된 취약점을 드러내며, 숨겨진 편향을 탐지하고 제거하는 체계적인 방법도 제안합니다. 소규모 구조 변경만으로도 편향을 크게 줄일 수 있음을 입증하여, AI 안전성 및 보안성 확보에 기여하게 됩니다. 이러한 결과는 상업적 폐쇄 소스 모델인 Gemini를 포함한 다양한 모델에서 블랙박스 공격(black-box attacks)으로 전이될 수 있음을 시사합니다.



### Do Voters Get the Information They Want? Understanding Authentic Voter FAQs in the US and How to Improve for Informed Electoral Participation (https://arxiv.org/abs/2412.15273)
- **What's New**: 이 논문은 모든 미국 주의 유권자 FAQ를 포함하는 최초의 데이터셋을 제공하며, 이를 통해 FAQ 정보 품질(FIQ) 메트릭스를 소개하고 있습니다. 이 연구는 각 주에서의 FAQ의 현재 관행을 분석하여 개선점을 제시하는 것을 목표로 합니다. 특히, 모든 50개 주 중 12%는 FAQ 품질에서 선도 주로, 8%는 뒤처진 주로 분류되었습니다.

- **Technical Details**: 연구팀은 공식 주 선거 웹사이트에서 모든 50개 주의 FAQ 데이터를 수집하였으며, 질문과 답변을 포함하는 JSON 파일로 저장되어 사용됩니다. 데이터 전처리 과정에서는 중복 제거를 위해 SequenceMatcher와 텍스트 정리 과정을 통해 편리한 분석을 위한 메타데이터가 포함된 독특한 Q&A 쌍이 최종 데이터셋에 포함되었습니다. 또한, 유권자 쿼리 및 답변 품질을 평가하기 위해 다섯 가지 읽기 용이성 메트릭을 사용하여 접근성을 분석하였습니다.

- **Performance Highlights**: 전반적으로 이 연구는 미국 내 FAQ 정보의 질을 향상시키기 위한 구체적인 방안을 제시합니다. 선도 주와 뒤처진 주의 콘텐츠 관행을 분석함으로써, 주 전반에 걸쳐 유권자에게 더 나은 정보를 제공하기 위해 필요한 개선 사항과 기준을 도출했습니다. 이 데이터셋은 향후 NLP 커뮤니티에서 유권자 교육과 참여 증진을 위한 중요한 자원으로 활용될 수 있습니다.



### Toxicity Detection towards Adaptability to Changing Perturbations (https://arxiv.org/abs/2412.15267)
- **What's New**: 이 연구는 독성 콘텐츠 탐지 분야에 새로운 문제인 '지속적 학습 jailbreak 교란 패턴'을 도입했습니다. 이는 사용자가 탐지기를 피하기 위해 새로운 교란 패턴을 창출하는 점을 반영하여 탐지기의 접근 방식을 혁신적으로 변화시키고자 합니다. 연구진은 기존의 수많은 전통적 탐지 방법들이 변화된 교란 패턴에 취약하다는 점을 확인하고, 이에 대한 해결책을 모색합니다.

- **Technical Details**: 논문에서 제안한 방법은 총 9종의 교란 패턴으로 생성된 데이터셋 DynEscape(다이나믹 이스케이프)를 기반으로 하며, 이를 통해 탐지기의 강건성을 유지하기 위해 도메인 증가 학습 방식이 사용됩니다. 연구진은 제안된 데이터셋을 통해 현재 탐지기들이 미지의 유형의 교란 독성 텍스트를 식별하는 데 어려움을 겪고 있다는 것을 체계적으로 검증했습니다. 또한, 지속적 학습 기법을 통해 탐지기는 이전과 새로운 교란 패턴 모두를 인식할 수 있는 능력을 키울 수 있습니다.

- **Performance Highlights**: 연구팀은 제안된 지속적 학습 접근 방식인 DynDetect(다이나믹 탐지)가 기존의 독성 탐지 최첨단 모델들과 비교하여 우수한 성능을 발휘함을 입증했습니다. 이로 인해 독성 콘텐츠의 탐지 정확성이 증가함은 물론, 다양한 교란 패턴에 대한 강건성 또한 확보되었습니다. 연구팀은 자가 감독 학습을 통해 탐지기의 지속적인 성능 향상을 가능하게 하는 새로운 연구 기회를 제공하고자 합니다.



### Early Dementia Detection Using Multiple Spontaneous Speech Prompts: The PROCESS Challeng (https://arxiv.org/abs/2412.15230)
Comments:
          2 pages, no figure, conference

- **What's New**: 이 논문은 초기 단계의 치매 탐지를 위한 새로운 spontaneous speech corpus를 소개합니다. 이는 신경과 전문의들이 설계한 세 가지 질문에 대한 응답을 포함하고 있으며, 참가자들이 실시간으로 수행하는 음성 신호 처리(Speech Signal Processing)와 AI 모델을 통한 분석을 지원합니다.

- **Technical Details**: PROCESS Signal Processing Grand Challenge는 두 가지 작업, 즉 초기 인지 감소 및 치매를 건강한 자원 봉사자와 구분하는 분류 작업(Classification Task)과 Mini Mental State Examination (MMSE) 점수를 예측하는 회귀 작업(Regression Task)으로 구성됩니다. 수집된 corpus는 Semantic Fluency, Phonemic Fluency, Cookie Theft 그림 설명 작업을 통해 음성을 수집하여 ASR 모델을 훈련시키기 위한 음성 레코딩과 수동 전사를 포함하고 있습니다.

- **Performance Highlights**: 기본 모델(Baseline Models)은 음향 및 텍스트 기능을 사용하여 아쿠스틱 피쳐는 OpenSmile을 통해 추출되었으며, F1-score 55.0%를 기록하였습니다. 텍스트 모델에서는 Whisper ASR 시스템을 통해 전사된 텍스트가 RoBERTa로 처리되어 최상의 RMSE 2.98을 기록했습니다. 현재 모델의 성능에는 개선이 필요하다는 점을 강조하며, 이 연구가 초기 치매 탐지 평가의 기준점으로 작용할 수 있기를 희망합니다.



### ResoFilter: Fine-grained Synthetic Data Filtering for Large Language Models through Data-Parameter Resonance Analysis (https://arxiv.org/abs/2412.14809)
Comments:
          under review

- **What's New**: 본 논문에서는 ResoFilter라는 새로운 방법을 제안하여 대규모 데이터 집합에서 고품질 데이터 선택을 통해 대형 언어 모델의 파인튜닝을 개선합니다. ResoFilter는 데이터를 효과적으로 셀렉션하기 위한 프로세스에서 모델, 데이터, 태스크를 통합하며, 각 데이터 포인트에 대한 특성 스코어를 도출하여 선택 기준으로 활용합니다. 이를 통해 모델 가중치를 통해 데이터 특성을 명확히 표현함으로써 해석 가능성을 높이고자 합니다.

- **Technical Details**: ResoFilter는 파인튜닝 과정에서 각 데이터 포인트를 완전한 순전파와 역전파를 통해 처리하여 모델 가중치의 변화를 포착합니다. 이 변화를 통해 도출된 특성 스코어는 데이터 선택의 기준으로 사용되며, 모델 가중치가 지식을 저장한다는 기존 연구를 기반으로 합니다. 실험 결과, MetaMath을 사용하여 ResoFilter가 전체 데이터셋의 파인튜닝을 반만 사용해도 유사한 성능을 달성함을 보여줍니다.

- **Performance Highlights**: 본 연구의 실험 결과, ResoFilter는 다양한 모델과 도메인에서 강력한 일반화를 보여주며, 특히 수학, 코드 및 일반 질문 응답 태스크에서 우수한 성과를 나타냅니다. 또한 ResoFilter는 성능이 낮은 데이터 포인트를 제거함으로써 전체 파인튜닝보다 더 나은 성능을 달성할 수 있습니다. 이 방법은 인공 데이터셋의 구성 및 고품질 데이터 평가에 대한 귀중한 통찰을 제공하여 데이터 증강 기법의 향상과 LLM의 훈련 데이터셋 질 개선에 기여합니다.



### MERaLiON-SpeechEncoder: Towards a Speech Foundation Model for Singapore and Beyond (https://arxiv.org/abs/2412.11538)
- **What's New**: 새롭게 등장한 MERaLiON-SpeechEncoder는 싱가포르의 국가 멀티모달 대규모 언어 모델 프로그램의 일환으로 개발된 기초 모델로서, 다양한 음성 애플리케이션을 지원합니다. 이 모델은 20만 시간의 레이블 없는 음성 데이터를 기반으로, masked language modelling을 이용한 자기 지도 학습(self-supervised learning, SSL)에 따라 처음부터 사전 훈련되었습니다. 현재는 주로 영어와 싱가포르에서 사용되는 영어 억양에 중점을 두고 있으며, 추후 동남아시아의 다른 언어도 지원할 계획입니다.

- **Technical Details**: MERaLiON-SpeechEncoder는 BERT 스타일의 masked-language modelling을 따르며, 입력 음성 신호의 마스킹된 프레임에 대해 올바른 레이블을 예측합니다. 이 모델은 BEST-RQ 목표를 채택하고 있으며, 이는 계산 비용을 절감하면서도 효율적인 음성 표현을 학습하도록 설계되었습니다. 다양한 준지도 학습 기법을 활용하여, 이 기초 모델은 다양한 음성 작업에 사용될 수 있도록 훈련되었습니다.

- **Performance Highlights**: 모델의 성능 평가는 싱가포르 영어에 대해 우수한 결과를 나타내며, 최신 기술 수준의 모델들과 비교 가능한 성능을 보여줍니다. ASR(Auto Speech Recognition) 벤치마크를 포함한 10가지 SUPERB 작업에 대한 성능 분석이 진행되었으며, 이는 자동 음성 인식 이외의 다양한 작업에서도 유용하게 나타납니다. MERaLiON-SpeechEncoder는 앞으로 다양한 멀티모달 연구와 개발을 지원할 것으로 기대됩니다.



New uploads on arXiv(cs.IR)

### Towards Interpretable Radiology Report Generation via Concept Bottlenecks using a Multi-Agentic RAG (https://arxiv.org/abs/2412.16086)
Comments:
          Accepted in ECIR 2025

- **What's New**: 이 연구는 Deep learning을 활용하여 Chest X-ray (CXR) 분류에서 해석 가능성(interpretability)을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히, concept bottleneck models (CBMs)와 multi-agent Retrieval-Augmented Generation (RAG) 시스템을 결합하여 임상적 관련성을 지닌 방사선 보고서를 생성합니다. 이러한 방법은 모델의 예측을 인간이 이해할 수 있는 방식으로 명확하게 제시하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 접근법은 두 단계를 통해 이루어집니다. 첫 번째 단계에서는 질병 분류와 관련된 개념 기여도를 계산하고, 두 번째 단계에서는 임상 문서와 설명을 활용하여 견고한 보고서를 생성합니다. 모델은 GPT-4를 사용하여 각 질병 범주에 대한 자동 개념 발견을 수행하고, 이미지 임베딩(ChexAgent 모델 사용)과 텍스트 임베딩(Mistral Embed Model 사용)을 결합하여 개념 벡터를 생성합니다.

- **Performance Highlights**: COVID-QU 데이터셋에서 본 모델은 81%의 분류 정확도를 기록하였으며, 생성된 보고서는 84%에서 90% 사이의 성능 메트릭을 보였습니다. 이는 AI 기반 CXR 분석의 신뢰성을 높이는 해석 가능성과 높은 성능 간의 갭을 메우는 다중 에이전트 프레임워크를 구축하는 데 기여합니다.



### Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Suppor (https://arxiv.org/abs/2412.15973)
- **What's New**: Legommenders는 콘텐츠 기반 추천 시스템을 위한 혁신적인 라이브러리로, 콘텐츠 인코더를 행동 및 상호작용 모듈과 공동으로 훈련할 수 있게 하여 콘텐츠 이해를 추천 파이프라인에 직접 통합합니다. 이 라이브러리는 연구자들이 15개의 다양한 데이터세트에서 1,000개 이상의 고유한 모델을 쉽게 생성하고 분석할 수 있도록 지원합니다. 또한, 현대적인 대형 언어 모델(large language models, LLMs)을 통합하여 보다 개인화되고 효과적인 콘텐츠 전달을 가능하게 합니다.

- **Technical Details**: Legommenders는 콘텐츠 기반 추천하기 위해 필요한 4가지 주요 구성요소로 구성되어 있습니다: 데이터셋 프로세서, 콘텐츠 연산자, 행동 연산자 및 클릭 예측기입니다. 이 라이브러리는 K+1개의 후보 항목을 평가하고 긍정적인 항목을 찾는 매칭(matching) 및 주어진 사용자-항목 쌍의 클릭 확률을 예측하는 순위(ranking) 작업을 지원합니다. Legommenders는 15개의 콘텐츠 연산자, 8개의 행동 연산자, 9개의 클릭 예측기를 제공하여 연구자가 유연한 추천 모델을 구축할 수 있도록 돕습니다.

- **Performance Highlights**: 전통적인 추천 시스템 라이브러리와 비교할 때, Legommenders는 콘텐츠 연산자, 행동 연산자 및 클릭 예측기를 통합한 엔드 투 엔드 훈련을 지원하여 모델의 성능을 극대화합니다. 평가 속도는 50배 향상되는 추론 캐싱 파이프라인을 통해 더욱 빠른 평가를 가능하게 하며, 기존 모델 라이브러리보다 6배 많은 1,000개 이상의 모델을 쉽게 생성할 수 있습니다. 이러한 성능 개선으로, Legommenders는 연구자들에게 콘텐츠 기반 추천의 새로운 연구 방향을 열어줄 플랫폼을 제공합니다.



### ASPIRE: Assistive System for Performance Evaluation in IR (https://arxiv.org/abs/2412.15759)
Comments:
          Accepted as a demo paper at the 47th European Conference on Information Retrieval (ECIR)

- **What's New**: ASPIRE는 정보 검색(Information Retrieval, IR) 시스템의 성능 평가를 위한 새로운 시각적 분석 도구입니다. 이 도구는 정량적 성능 측정을 넘어 IR 실험의 깊이 있는 분석을 지원하여 연구자들이 다루는 복잡한 과제를 해결하는 데 도움을 줍니다. ASPIRE는 단일/다중 실험 비교, 쿼리 수준 분석, 쿼리 특성과 성능 간의 상호작용 분석, 그리고 수집 기반 검색 분석의 네 가지 주요 기능을 제공합니다.

- **Technical Details**: ASPIRE는 Python으로 개발되었으며, 웹 인터페이스는 streamlit(v1.37.0)을 기반으로 합니다. 각 페이지는 명확하게 정의된 섹션으로 구성되어 있어, 기존 페이지 내에서 새로운 분석을 추가하거나 새로운 페이지를 생성하여 유연하게 확장할 수 있습니다. 사용자들은 TREC 스타일 파일을 업로드하고, 분석 버튼을 클릭함으로써 실험 결과를 신속하게 분석할 수 있는 기능을 제공합니다.

- **Performance Highlights**: ASPIRE는 사용자 친화적인 인터페이스를 제공하여 연구자들이 IR 시스템의 성능에 대해 더 깊은 통찰을 얻을 수 있도록 돕습니다. 연구자들은 업로드한 파일로부터 즉각적인 결과를 얻고, 상호작용 가능한 시각화를 통해 편리하게 IR 실험을 분석할 수 있습니다. 향후 ASPIRE는 IR 평가에서 좋은 관행을 장려하는 중요한 역할을 할 것으로 기대됩니다.



### PolySmart and VIREO @ TRECVid 2024 Ad-hoc Video Search (https://arxiv.org/abs/2412.15494)
- **What's New**: 올해 TRECVid AVS(Task)의 검색을 위한 생성 보강(retrieval-augmented) 방법을 탐구했습니다. 세 가지 방법인 Text2Text (T2T), Text2Image (T2I), Image2Text (I2T)를 통해 텍스트 쿼리의 이해도를 향상시켜 out-of-vocabulary (OOV) 문제를 해결하고자 했습니다. 원래 쿼리에서 검색된 순위 목록과 이들의 다양한 조합을 통해, 자동 실행 결과를 Four 세트로 제출하였습니다.

- **Technical Details**: 이번 연구에서는 Text2Text, Text2Image, Image2Text 변환을 통해 OOV 문제를 다뤘습니다. T2T 변환에서 대형 언어 모델(LLM, e.g., LlaMA 3)을 활용하여 기존 지식에 기반해 OOV 단어를 동의어로 바꾸었습니다. 또한 T2I 변환을 통해 텍스트 쿼리를 시각적 개념으로 변환하며, I2T 변환에서는 이미지 캡셔닝 모델(예: BLIP-2)로 다시 텍스트로 바꾸어 다양한 순위 목록을 생성하였습니다.

- **Performance Highlights**: 자동 실행 결과는 F_M_C_D_PolySmartAndVIREO.24_1에서 xinfAP=0.294, F_M_C_D_PolySmartAndVIREO.24_2에서 xinfAP=0.283, F_M_C_D_PolySmartAndVIREO.24_3, 4에서 각각 xinfAP=0.277을 달성했습니다. 수동 쿼리를 기반으로 한 실행은 F_M_N_D_PolySmartAndVIREO.24_1이 0.216, M_M_C_D_PolySmartAndVIREO.24_2가 0.274, M_M_C_D_PolySmartAndVIREO.24_3은 0.280을 달성했습니다. 생성된 쿼리와 원래 쿼리의 융합이 원래 쿼리보다 성능이 우수하다는 결과를 보였습니다.



### A Retrieval-Augmented Generation Framework for Academic Literature Navigation in Data Scienc (https://arxiv.org/abs/2412.15404)
- **What's New**: 이번 논문에서는 데이터 과학 분야의 학술 문헌 탐색을 지원하기 위해 Retrieval-Augmented Generation (RAG) 응용 프로그램을 개선한 시스템을 제안합니다. 이 AI 기반 시스템은 GROBID 기술을 포함한 여러 고급 기술을 통합하여 관련성 높은 정보를 효율적으로 검색할 수 있게 합니다. 이러한 구현은 학술 문헌 탐색의 어려움을 해결하는 데 중점을 두고 있으며, 정보 과부하를 줄이고 의사 결정을 개선하는 데 기여할 것으로 보입니다.

- **Technical Details**: 우리의 RAG 응용 프로그램은 기계 학습, 딥 러닝, 시계열 모델링 등 데이터 과학의 여러 하위 분야를 아우릅니다. 이 시스템은 GROBID를 사용하여 데이터를 정리하고, 전문화된 파인 튜닝(process)을 통해 임베딩 모델의 성능을 향상시킵니다. 또한, 의미론적 청킹(semantic chunking)을 적용하여 텍스트를 의미 있는 단위로 나누고, 추상 우선 검색(abstract-first retrieval) 방법을 통해 정보 검색 과정을 간소화합니다.

- **Performance Highlights**: 우리의 RAG 응용 프로그램은 RAGAS 프레임워크를 통해 50개의 샘플 질문을 테스트하여 그 효과를 검증하였습니다. 특히, Context Relevance를 향상시키는 데 초점을 두어, 관련있는 학술 콘텐츠를 검색하는 능력이 크게 향상되었습니다. 이러한 결과는 데이터 과학자들에게 실질적인 도구로서 기능할 가능성을 보여주고 있으며, 학술 탐색을 간소화하고 정보 기반 의사 결정을 지원하는 데 기여할 것입니다.



### Ranking Narrative Query Graphs for Biomedical Document Retrieval (Technical Report) (https://arxiv.org/abs/2412.15232)
Comments:
          Technical Report of our accepted paper at AI4LAC@JCDL2024. 11 pages, 5 figures

- **What's New**: 이 논문은 기존의 그래프 기반 검색 시스템을 확장하여, 효과적인 그래프 기반 비지도 순위(Ranking) 방법, 새로운 질의 완화(Relaxation) 패러다임 및 온톨로지 재작성(Ontological Rewriting)을 제안합니다. 이러한 확장은 사용자가 부분 일치(Partial Matching) 및 온톨로지 재작성을 통해 더 높은 정밀도와 재현율(Recall)을 가지고 검색할 수 있도록 돕습니다. 기존의 '정확한 일치(Exact Match)' 패러다임에서 벗어나, 문서 간의 관계와 맥락을 활용할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 연구팀은 생물의학(Biomedical) 도메인을 위해 완전한 그래프 기반 발견 시스템을 구현했습니다. 기존 시스템은 문서 그래프 표현의 구조를 효과적으로 활용하고, 질의 매칭을 완화하여 검색의 재현율을 높입니다. 이러한 방법은 비지도 방식으로 작동하여 고가의 훈련 데이터를 요구하지 않아, 다양한 통계적 벤치마크에서 그 유용성을 입증합니다.

- **Performance Highlights**: 이 논문에서는 비슷한 생물의학 기준을 측정하는 TREC Precision Medicine Series와 TREC-COVID 2020 벤치마크의 성능을 비교했습니다. 특정 유전자-질병-치료 조합과 같은 정밀한 쿼리를 통해 더 많은 검사 결과를 제공하였고, BM25 등 기존 메트릭스를 따르는 방법들과 비교했습니다. 그 결과, 제안된 그래프 기반 방법은 정교한 문서 요약 및 탐색 기능을 통해 사용자에게 큰 이익을 제공하며, 이 시스템의 코드는 GitHub에서 공개되었습니다.



### Building an Explainable Graph-based Biomedical Paper Recommendation System (Technical Report) (https://arxiv.org/abs/2412.15229)
Comments:
          Technical Report of our accepted paper at AI4LAC@JCDL2024. 12 pages, 3 figures

- **What's New**: 이번 연구에서는 XGPRec라는 그래프 기반의 설명 가능한 논문 추천 시스템을 제안합니다. 제안된 방법은 기존의 신경망 기반 추천 시스템에서 발생하는 높은 계산 비용을 피하고, 사용자에게 명확한 추천 이유를 제공하여 활용도를 높입니다. 특히 3,700만 개의 생물의학 문서로 구성된 실제 디지털 라이브러리에서 효과적인 성능을 입증했습니다.

- **Technical Details**: XGPRec는 PubPharm 데이터베이스와 협력하여 그래프 기반 문서 표현을 사용하여 시스템을 구축합니다. 이 시스템은 BM25를 통해 텍스트 점수를 매기고, 그래프 유사성을 바탕으로 빠르고 비용 효율적인 추천을 제공합니다. 연구진은 복잡한 개념 중심의 정보 요구를 내러티브 쿼리 그래프로 나타내어 유저들이 문헌을 정확하게 탐색할 수 있도록 지원합니다.

- **Performance Highlights**: 기초 사용자 연구를 통해 XGPRec의 추천이 실제로 유용한 설명을 생성한다는 것을 보여주었습니다. 이 시스템은 디지털 라이브러리에서 빠르고 신뢰할 수 있는 추천을 제공하며, 사용자들이 추천된 논문의 관련성을 쉽게 이해할 수 있도록 도와줍니다. 연구진은 GitHub를 통해 XGPRec의 코드를 공개하여 다른 사용자들이 이를 기반으로 추가 기능을 개발할 수 있도록 하고 있습니다.



### From General to Specific: Tailoring Large Language Models for Personalized Healthcar (https://arxiv.org/abs/2412.15957)
- **What's New**: 의료 분야에서 의료 LLM의 개인화가 필수적이라는 문제를 다루고 있는 연구가 소개되었습니다. 본 연구에서는 개인화된 의료 언어 모델(PMLM)을 제안하며, 추천 시스템과 강화 학습(reinforcement learning, RL)을 통해 개인 맞춤형 LLM의 최적화를 탐구합니다. 특히, PMLM은 개인의 요구에 맞춘 최초의 개인화된 프롬프트를 설계하고 이를 RL을 통해 더욱 정제하여 LLM의 정확한 방향성을 활용하도록 합니다.

- **Technical Details**: 연구에서는 환자의 과거 데이터를 분석하여 개인화된 정보를 추출하고, 유사한 환자의 통찰력을 결합하여 원주율 프롬프트를 생성하는 프로세스를 설명합니다. 이러한 초기 프롬프트는 강화 학습을 통해 세밀한 개인화를 위해 정제됩니다. PMLM의 프롬프트는 하드 프롬프트(hard prompt)로, 이는 높은 적응성과 재사용성을 부여하여 다양한 LLM에 직접적으로 활용할 수 있습니다.

- **Performance Highlights**: 실제 산부인과 데이터를 통해 평가한 결과, PMLM은 개인화된 응답을 제공함으로써 기존의 세밀하게 조정된 LLM들보다 더 나은 성과를 보였습니다. 이 연구는 LLM의 개인화 가능성을 높이고, 개인 맞춤형 의료 LLM의 발전을 위한 새로운 경로를 제시합니다. PMLM은 다양한 질병에 대해 대응할 수 있는 가능성을 갖추고 있어, 향후 의료 분야에서의 LLM 활용에 중요한 기여를 할 것으로 기대됩니다.



### Music Genre Classification: Ensemble Learning with Subcomponents-level Attention (https://arxiv.org/abs/2412.15602)
- **What's New**: 본 연구에서는 Music Genre Classification에 있어 새로운 접근 방식을 제시합니다. 이 접근 방식은 ensemble learning과 sub-components에 대한 attention을 결합하여 음악 장르를 식별하는 정확성을 높이는데 중점을 두고 있습니다. 주목할 점은 음악 작품의 서브 컴포넌트를 별도로 분류함으로써 모형이 각각의 독특한 특성을 포착할 수 있다는 것입니다.

- **Technical Details**: 제안된 방법론은 각 서브 컴포넌트를 독립적으로 분류한 후, 이들에 대한 ensemble learning 기법을 적용합니다. 이러한 방식은 전체 음악 장르에 대한 분류 결정을 내리는 데 활용됩니다. 또한, GTZAN 데이터셋에서 훈련 및 테스트된 기존의 최첨단 기법들에 비해 더 뛰어난 정확도를 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 여러 기법들과 비교하여 우수한 성능을 나타내며, 그 효과는 GTZAN 데이터셋에서 명확하게 입증되었습니다. 이 연구는 Music Information Retrieval (MIR) 및 디지털 신호 처리 분야에 있어서 의미 있는 기여를 할 것으로 기대됩니다.



### ADEQA: A Question Answer based approach for joint ADE-Suspect Extraction using Sequence-To-Sequence Transformers (https://arxiv.org/abs/2412.15510)
- **What's New**: 본 논문은 ADE(Adverse Drug Event)의 조기 식별을 위해 새로운 접근 방식을 제안합니다. ADEQA라는 질문-응답(QA) 방식의 모델이 마련되어, 비정형 데이터 소스로부터 ADE와 관련 약물을 추출하는 데 집중하고 있습니다. 이는 기존의 QA 모델과 달리 자연어 생성(NLG) 기반 모델을 사용하여 토큰 레벨의 라벨링 필요성을 줄임으로써 신약 시장 도입 시 신속한 조치를 가능하게 합니다.

- **Technical Details**: ADEQA 모델은 준수 감독(quasi supervised) 라벨 데이터와 시퀀스 투 시퀀스(transformers) 기술을 이용하여 ADE, 관련 약물 및 이들 간의 관계를 추출하는 데 도움을 줍니다. 이 방법론은 복잡한 언어적 관계를 효과적으로 처리하여, 비라벨 대량 데이터 부족 문제를 해결하는 데 기여합니다. 추가적으로, 향상된 결과를 도출하기 위해 링크를 통한 데이터 연결 체계를 개선하고 있습니다.

- **Performance Highlights**: 공식 ADE 데이터셋을 사용한 실험에서는 ADEQA 모델이 94%의 F1 점수를 기록하며 최첨단 결과를 달성했습니다. 이는 ADE와 해당 약물 간의 관계를 확립하는 데 있어 탁월한 성능을 보임을 나타냅니다. 이러한 성과는 새로운 약물이 시장에 도입될 때 잠재적 위험 요소를 사전에 식별하는 데 중요한 기여를 할 것으로 기대됩니다.



### Learning Visual Composition through Improved Semantic Guidanc (https://arxiv.org/abs/2412.15396)
- **What's New**: 이번 논문은 기존의 시각적 표현 학습 방식의 한계를 극복하기 위해, 간단하고 확장 가능한 접근 방식을 제안합니다. 기존의 CLIP 모델들이 인과(thought) 이해에 부족하다는 점을 지적하며, 비약적인 성능 향상을 위해 기존 캡션을 새로운 다중 모달 모델에 의해 개선한다고 설명합니다. 특히, 향상된 데이터 세트 기반 훈련을 통해 이미지 검색 작업에서 실질적인 성과를 내고 있음을 입증하고자 하였습니다.

- **Technical Details**: 제안된 방법론은 CLIP 아키텍처에 기반하여 두 가지 주요 수정 사항을 도입합니다. 첫 번째, 고품질의 영어 이미지-텍스트 쌍을 포함한 WebLI 데이터셋을 사용하여 캡션의 품질을 향상시켜 alt-text를 대체합니다. 둘째, 훈련된 다중 모달 모델의 텍스트 타워를 고성능의 텍스트 기반 기초 모델로 교체하여 시각적 임베딩 표현을 크게 개선합니다. 이러한 방법을 통해 94.5%의 명확성을 달성하며 판별 성능이 크게 향상되었습니다.

- **Performance Highlights**: 향상된 데이터 세트와 적절한 모델 수정으로, CLIP 모델의 이미지 검색 성능이 눈에 띄게 개선되었습니다. 특히, recall@1 메트릭에서 58.4%의 성능에서 94.5%로 향상되었습니다. 일반적인 captcha 값으로 ARO 평가 데이터를 통하여 비교적 단순한 다중모달 모델의 학습 성과가 나타나고, 새로운 벤치마크에 대한 필요성을 적시하며 실험 결과의 타당성을 강화합니다.



### MRWeb: An Exploration of Generating Multi-Page Resource-Aware Web Code from UI Designs (https://arxiv.org/abs/2412.15310)
- **What's New**: 본 연구는 Multi-Page Resource-Aware Webpage (MRWeb) 생성 작업을 제안하여 기존의 디자인-코드 전환 방식을 확장하고, 사용자 인터페이스(UI) 디자인을 다중 페이지 웹 UI로 변환하는 방법을 제시합니다. 이를 통해 웹페이지의 내비게이션, 이미지 로딩, 백엔드 라우팅을 지원하며, 다양한 리소스를 관리하기 위한 새로운 데이터 구조인 resource list를 도입하였습니다. 500개의 웹사이트로 구성된 새로운 데이터셋을 활용하여 MRWeb 생성의 복잡성을 분석했습니다.

- **Technical Details**: MRWeb 생성 작업은 웹 UI의 복잡한 요구 사항을 충족시키기 위해 설계되었으며, 기존 단일 페이지 웹 개발의 한계를 넘습니다. 연구팀은 resource list라는 딕셔너리 형식의 데이터 구조를 정의하여 내부/외부 리소스와 디자인 요소의 상관 관계를 추적합니다. MRWeb 툴은 이 resource list와 스크린샷을 입력으로 받아 기능적 MRWeb 코드를 생성하며, 사용자 친화적인 도구로 개발되어 개방형 연구를 지원합니다.

- **Performance Highlights**: 실험 결과, resource list를 활용한 MRWeb 생성에서는 내비게이션 기능이 0%에서 66%-80%로 향상되었으며, 이는 시각적 유사성에도 긍정적 영향을 미쳤습니다. 연구진은 MLLM의 성능을 평가하기 위한 새로운 메트릭을 제안하고 MRWeb 도구의 효과성을 분석함으로써 향후 연구에 대한 통찰력을 제공합니다. 이 연구는 MRWeb 문제를 해결하기 위한 첫 번째 평가 프레임워크를 마련하고, 모든 코드 및 데이터를 공개하여 추가 연구를 촉진하고자 합니다.



### ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in Vietnames (https://arxiv.org/abs/2412.15308)
Comments:
          Accepted at AAAI'2025 Main Conference

- **What's New**: 이 논문에서는 베트남어 사실 확인을 위한 최초의 공개 데이터셋인 ViFactCheck를 소개합니다. 이 데이터셋은 12개의 다양한 주제를 다루는 총 7,232개의 인간 주석된 주장-증거 쌍으로 구성되어 있으며, 이는 베트남의 신뢰할 수 있는 온라인 뉴스에서 수집되었습니다. 또한 이 데이터셋은 고품질과 신뢰성을 보장하는 면밀한 주석 과정을 거쳤고, Fleiss Kappa 상호 주석자 동의 점수는 0.83에 달합니다.

- **Technical Details**: ViFactCheck 데이터셋은 9개의 라이센스가 있는 인기 있는 베트남 온라인 신문에서 수집된 기사를 기반으로 구축되었습니다. 이 데이터셋은 데이터 수집, 주석 처리 및 주석 검증의 세 가지 단계로 나뉘어 각 전문가의 엄격한 모니터링을 받습니다. 또한, 이 연구에서는 최첨단의 사전 학습된 언어 모델과 대형 언어 모델을 활용하여 사실 확인을 위한 다양한 기법을 평가하였습니다.

- **Performance Highlights**: Gemma 모델은 89.90%의 매크로 F1 점수를 기록하여 우수한 효과성을 입증하며 사실 확인 벤치마크에 대한 새로운 기준을 설정하였습니다. 이러한 결과는 Gemma가 베트남에서 사실을 정확하게 식별하고 확인하는 데 있어 뛰어난 능력을 보여줍니다. 또한 ViFactCheck 데이터셋, 모델 체크포인트, 사실 확인 파이프라인 및 소스 코드는 GitHub를 통해 무료로 제공되어 향후 연구를 촉진하고 정보의 정확성을 높이는 데 기여할 것입니다.



### A Systematic Examination of Preference Learning through the Lens of Instruction-Following (https://arxiv.org/abs/2412.15282)
Comments:
          23 pages

- **What's New**: 최근 대규모 언어 모델(LLMs)의 인간 선호에 대한 정렬을 위한 연구가 심화되고 있습니다. 본 연구는 23개의 검증 가능한 제약 조건을 조합하여 48,000개의 고유한 지침 프롬프트를 생성하는 새로운 합성 데이터 생성 파이프라인을 사용하여, 선호 데이터 세트의 특정 속성이 LLM의 성능에 미치는 영향을 체계적으로 조사합니다. 이를 통해 지침을 따르는데 있어 모델의 조정 및 성능 향상을 도모하고자 합니다.

- **Technical Details**: 우선, 본 연구는 선택된 응답(chosen response)과 거부된 응답(rejected response) 쌍으로 LLM의 성능을 개선하는데 중점을 둡니다. 두 가지 방법인 거부 샘플링(rejection sampling, RS)과 몬테 카를로 트리 탐색(Monte Carlo Tree Search, MCTS)을 적용하여 선호 쌍을 자동으로 수집하고, 이를 통해 모델의 일반화 성능을 평가합니다. 이 연구는 지침을 따르는데 있어 응답의 공유 접두사(shared prefixes)와 응답의 대비(contrast) 및 품질(quality)이 어떻게 영향을 미치는지 이해하기 위한 실험을 포함하고 있습니다.

- **Performance Highlights**: 연구 결과, MCTS로 생성된 공유 접두사(preferences pairs with shared prefixes)가 RS로 생성된 것보다 일관되게 우수한 성능을 보였으며, 높은 대비(high-contrast) 쌍이 낮은 대비(low-contrast) 쌍보다 더 나은 성과를 내는 것으로 나타났습니다. 그러나 높은 대비와 낮은 대비 쌍의 혼합이 학습 효율성과 다양성의 균형을 맞추면서 최상의 성능을 가져옵니다. 마지막으로, 중간 난이도의 프롬프트(training prompts)는 과제 전반에 걸쳐 더 나은 일반화를 이끌어내어나가는 것으로 밝혀졌습니다.



### Context-DPO: Aligning Language Models for Context-Faithfulness (https://arxiv.org/abs/2412.15280)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 컨텍스트 충실도를 향상시키기 위해 최초로 설계된 Context-DPO라는 정렬 방법을 제안합니다. 이를 통해 모델이 제공된 정보와 사용자 지침을 보다 잘 따를 수 있도록 합니다. ConFiQA라는 새로운 벤치마크도 소개하여 모호한 구매 모델의 성능을 평가합니다.

- **Technical Details**: ConFiQA는 질문-응답 작업을 기반으로 하여 LLM의 컨텍스트 충실도를 평가합니다. QA, MR, MC라는 세 가지 데이터세트로 구성되며, 각각 단일 및 다중 훅 질문-응답과 다양한 관련된 반사 사례를 포함합니다. 모델의 훈련 상태와 크기에 따라 컨텍스트 충실도가 감소하는 경향을 보이며, 이를 해결하기 위해 Context-DPO를 통해 반사배급을 보상하는 방법을 적용합니다.

- **Performance Highlights**: Context-DPO는 LLM의 컨텍스트 충실도를 35%에서 280%까지 개선하여, 기존의 모든 모델들보다 현저히 뛰어난 성능을 보여주었습니다. 추가적으로, 이 연구는 LLM의 생성 능력에 부정적인 영향을 주지 않으면서도 컨텍스트 충실도를 저해하지 않음을 입증하였습니다. 또한, 모델의 정렬 결과를 분석하여 컨텍스트 활용의 해석 가능한 통찰을 제공합니다.



### SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15272)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 다양한 태스크(Task)에서 뛰어난 유연성을 보여주고 있습니다. 이러한 맥락에서, Retrieval-Augmented Generation (RAG) 접근 방식이 외부 지식 소스인 지식 그래프(KGs)를 활용하여 환각(hallucination)을 제거하는 데 있어 강력한 방법으로 자리잡고 있습니다. 본 논문에서는 KG 기반 RAG 태스크를 조사하고, 유사 그래프 강화 검색 증대 생성(SimGRAG) 방법을 제안하여 쿼리 텍스트와 KG 구조를 정렬하는 문제를 효과적으로 해결합니다.

- **Technical Details**: SimGRAG 방법은 두 단계의 프로세스를 통해 쿼리 텍스트와 KG 구조의 정렬을 수행합니다. 첫 번째 단계는 쿼리를 원하는 그래프 패턴으로 변환하는 LLM을 사용하는 'query-to-pattern' 단계입니다. 두 번째 단계는 패턴과 후보 서브그래프 간의 정렬 정도를 그래프 의미 거리(Graph Semantic Distance, GSD) 메트릭을 이용해 정량화하는 'pattern-to-subgraph' 단계입니다. 이 방법은 1000만 규모의 KG에서 1초 이내에 최상위 k 서브그래프를 효율적으로 식별하는 최적화된 검색 알고리즘을 개발하여 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, SimGRAG는 질문 응답 및 사실 검증(task)에서 최신 KG 기반 RAG 방법들을 초월하는 성능을 보여줍니다. 본 논문이 제시하는 방법은 플러그 앤 플레이(plug-and-play) 사용성과 확장성(scalability)을 갖추고 있어, 다양한 KG 및 LLM과의 매끄러운 통합이 가능합니다. 또한, SimGRAG는 불필요한 정보 유출을 방지하고, 가장 관련성 높은 서브그래프를 명확하게 찾아내는 능력을 갖추어 있습니다.



### A MapReduce Approach to Effectively Utilize Long Context Information in Retrieval Augmented Language Models (https://arxiv.org/abs/2412.15271)
- **What's New**: 이 연구에서는 의료 분야에서 Retrieval-Augmented Generation (RAG) 워크플로우의 견고성과 신뢰성을 개선하는 것을 목표로 합니다. 제안된 BriefContext 전략은 모델의 가중치를 수정하지 않으면서 "lost-in-the-middle" 문제를 해결하고자 합니다. 다양한 LLM 백본과 여러 QA 데이터셋에서 이 워크플로우의 우수성을 Demonstrated 하였습니다.

- **Technical Details**: BriefContext는 긴 컨텍스트 reasoning 작업을 여러 개의 짧은 컨텍스트 reasoning 작업으로 변환하는 새로운 프레임워크입니다. 이 프레임워크는 맵-리듀스(map-reduce) 개념을 활용하여 긴 컨텍스트를 여러 개의 파티션으로 나누고 이를 여러 LLM 세션으로 분배합니다. 또한, Preflight 메커니즘을 도입해 "lost-in-the-middle" 문제의 발생을 예측하는 과정을 포함하고 있습니다.

- **Performance Highlights**: BriefContext는 RAG 베이스라인보다 월등한 성능을 보였으며, 특히 키 정보가 중간에 위치할 때 큰 개선을 확인했습니다. 짧은 컨텍스트에서 LLM의 성능이 더 우수하다는 것을 입증하였고, Preflight 체크는 공지 발생 예측에서 92.61%의 재현율을 기록하였습니다. 이 연구는 의료 분야에서 LLM을 안전하게 배포하는 데 기여할 수 있는 가능성을 보여줍니다.



### Advanced ingestion process powered by LLM parsing for RAG system (https://arxiv.org/abs/2412.15262)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문은 다양한 구조적 복잡성을 가진 멀티모달 문서를 처리하는 데 어려움을 겪는 Retrieval Augmented Generation (RAG) 시스템을 위한 새로운 멀티 전략 파싱 접근법을 제안합니다. LLM 기반 OCR을 활용하여 프레젠테이션 및 고밀도 텍스트 파일을 포함한 다양한 문서 유형의 콘텐츠를 추출합니다. 이 방법론은 서로 다른 정보 유형 간의 관계를 생성하고 문맥 인식 메타데이터를 생성하는 노드 기반 추출 기법을 사용합니다.

- **Technical Details**: 선행 처리 단계는 구문 분석, 조립 및 메타데이터 추출의 세 하위 프로세스로 구성됩니다. 구문 분석 단계에서는 Python 라이브러리와 멀티모달 LLM을 이용하여 이미지 및 텍스트 콘텐츠를 추출하며, AWS Textract와 같은 외부 기계 학습 모델을 사용하는 OCR도 포함됩니다. 각 페이지의 이미지를 분석하고 설명한 후, Multimodal Assembler Agent가 모든 페이지의 텍스트를 통합하여 종합적인 문서 수준의 Markdown 파일을 생성합니다.

- **Performance Highlights**: 실험적인 평가 결과는 RAG 시스템의 효율성이 향상됨을 보여주며, 정보의 회수 능력과 답변의 적합성이 증가했습니다. 이 접근법은 특히 답변의 관련성과 정보의 정확성을 측정하는 데 중요한 평가 지표를 설정하여 시스템의 전반적인 품질 향상에 기여했습니다. 논문에서 사용된 평가 지표들은 RAG 시스템의 효과성을 정량화하는 데 필수적이며, 이를 통해 시스템의 신뢰성을 확보할 수 있습니다.



### Streamlining Systematic Reviews: A Novel Application of Large Language Models (https://arxiv.org/abs/2412.15247)
- **What's New**: 이번 연구는 시스템적 리뷰(Systematic Reviews, SR)에서 문헌 스크리닝(literature screening)의 자동화를 위한 사내 시스템을 제안합니다. 이 시스템은 Large Language Models (LLMs)을 기반으로 하여 제목/초록(title/abstract) 및 본문(full-text) 스크리닝 자동화를 목표로 하였습니다. 기존에 시간과 자원이 많이 소요되었던 의료 문헌 스크리닝 과정에서의 중요한 공백을 해소하기 위해 개발되었습니다.

- **Technical Details**: LLM 기반 시스템은 제목/초록 스크리닝에 대한 프롬프트 엔지니어링(prompt engineering)과 본문 스크리닝에 대한 Retrieval-Augmented Generation (RAG) 기법을 사용합니다. 이 시스템은 14,439개의 문헌을 포함한 비타민 D와 낙상에 대한 완전한 SR을 사용하여 평가되었으며, 99.5%의 기사 제외율(article exclusion rate)과 99.6%의 특이도(specificity)를 기록했습니다. 결국 78개의 기사가 수동 검토(manual review)가 필요하였고, 이는 전통적인 방법으로 식별된 20개를 포함합니다.

- **Performance Highlights**: LLM 기반 시스템은 총 스크리닝 시간을 25.5시간으로 줄이면서도 높은 정확도를 유지하였습니다. 상용 도구인 Rayyan과 비교했을 때 AER은 72.1%였고, FNR은 5%로 나왔습니다. 특히, LLM이 전통적인 방법보다 신뢰성과 효율성을 크게 개선했음을 보여주며, 전체 스크리닝 프로세스에 있어 자동화 도구의 부족 문제를 해결하고 SR 워크플로우에서 LLM의 변혁적인 잠재력을 강조합니다.



### Accelerating Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15246)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하고 정확도를 향상시키기 위한 솔루션으로 Retrieval-Augmented Generation (RAG) 접근법을 제안합니다. RAG는 LLM과 외부 지식 소스(예: 웹)에서 검색된 정보를 결합하는 방식으로 동작합니다. 저자들은 RAG의 실행 파이프라인을 분석하고, 고품질 검색을 위한 Intelligent Knowledge Store (IKS)라는 새로운 CXL 메모리 확장을 소개합니다.

- **Technical Details**: IKS는 새로운 캐시 일관성 인터페이스를 가진 고성능, 고용량 벡터 데이터베이스 가속기입니다. IKS는 정확한 최근접 이웃 검색(Exact Nearest Neighbors Search, ENNS)을 가속화하여 512GB 벡터 데이터베이스에서 13.4-27.9배 빠른 검색 성능을 제공합니다. 이 시스템은 CPU와 근처 메모리 가속기 간의 효율적인 인터페이스를 구현하여 메모리를 분산시키면서도 성능을 극대화하는 설계를 특징으로 합니다.

- **Performance Highlights**: IKS는 벡터 데이터베이스 애플리케이션에서 1.7-26.3배의 엔드 투 엔드 추론 속도 향상을 이끌어 내며, 이는 RAG 애플리케이션에서 대표적으로 관찰되는 성능 개선입니다. 본 연구는 RAG의 다양한 하드웨어 및 소프트웨어 구성이 성능과 정확도에 미치는 영향을 심도 있게 평가합니다. 논문에서 제시된 IKS는 기존의 메모리 시스템의 한계를 극복하기 위한 중요한 진전을 의미합니다.



### Quantifying Positional Biases in Text Embedding Models (https://arxiv.org/abs/2412.15241)
Comments:
          13 pages, 11 figures, NeurIPS

- **What's New**: 이번 연구는 정보 검색(Information Retrieval, IR)과 의미 유사성 측정에서 중요하게 사용되는 embedding 모델의 한계, 특히 긴 텍스트와 관련된 위치 편향 처리에 대해 다룹니다. content position과 input size가 text embedding에 미치는 영향을 실험을 통해 조사하였으며, embedding 모델들이 입력의 시작 부분을 불균형적으로 우선시하는 경향을 발견했습니다. 특히 문서의 처음에 무관한 텍스트를 삽입하거나 삭제하는 실험을 통해, 이러한 변화가 cosine similarity에 미치는 영향을 정량화하였습니다.

- **Technical Details**: embedding 모델은 transformer encoder 아키텍처를 기반으로 하여 bidirectional self-attention block을 사용합니다. 이 모델들은 고정된 길이의 벡터를 생성하여 전체 입력 텍스트를 표현하며, cosine similarity를 사용해 embedding을 비교합니다. 연구는 Absolute Positional Embedding(APE)과 Rotary Positional Embedding(RoPE), Attention with Linear Biases(ALiBi)와 같은 다양한 positional encoding 기법을 검토하여 이들이 모델의 내재적 편향에 어떻게 기여하는지를 설명합니다.

- **Performance Highlights**: 모델의 문서간 cosine similarity 측정 결과, 텍스트의 처음 부분에 무관한 삽입이 시뮬레이션된 경우, 유사성 감소가 중간이나 끝에 삽입한 경우보다 평균 8.5% 및 12.3% 더 크게 나타났습니다. 문서의 초기 문장에서 멀어질수록 회귀 계수가 크게 감소하는 것을 통해, 모델이 초기 내용에 불균형적으로 가중치를 부여한다는 점을 확인했습니다. 이러한 발견은 실제 검색 시스템의 민감도를 정량화하며, 모델의 강건성 향상 방향에 대한 새로운 관점을 제시합니다.



New uploads on arXiv(cs.CV)

### HoVLE: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding (https://arxiv.org/abs/2412.16158)
- **What's New**: 최근 급속히 발전하는 Large Language Models (LLMs)의 발전은 Vision-Language Models (VLMs)의 개발을 촉진했습니다. 본 논문에서는 HoVLE라는 새로운 고성능 단일형 VLM을 제안합니다. 이 모델은 시각적 입력과 언어 입력을 결합할 수 있는 holistic embedding module을 도입하여 기존의 compositional VLMs와 경쟁할 수 있는 성능을 보여줍니다.

- **Technical Details**: HoVLE는 시각적 및 텍스트 입력을 공유 공간으로 변환하는 holistic embedding module을 통해 LLMs를 비전 능력으로 확장합니다. 이 module은 multi-stage training 전략을 활용하여 pre-trained vision encoder로부터 시각적 특징을 증류하고, LLM으로부터 텍스트 임베딩을 추출합니다. 이를 통해 대규모 unpaired random 이미지와 텍스트 토큰으로 훈련할 수 있는 가능성이 열립니다.

- **Performance Highlights**: HoVLE는 17개의 다중 모달 벤치마크에서 leading compositional VLM들과 유사한 성능을 보이며, 이전의 단일형 VLM들과 비교했을 때 상당한 성능 향상을 기록했습니다. 예를 들어, MMBench에서는 약 15 포인트 정도 성능을 개선하여 HoVLE의 효과성을 입증합니다.



### Personalized Representation from Personalized Generation (https://arxiv.org/abs/2412.16156)
Comments:
          S.S. and J.C contributed equally; S.B. and P.I. co-supervised. Project page: this https URL

- **What's New**: 이 논문은 현대 비전 모델을 개인화된 비전 태스크에 적용하는 방법을 탐구합니다. synthetic data와 T2I diffusion model의 발전을 활용하여, 적은 수의 실제 예시로부터 개인화된 이미지를 생성할 수 있는 가능성을 살펴봅니다. 개인화된 synthetic data를 통해 개인화된 representations를 학습하는 도전을 정식화하고, 이와 관련된 평가 도구도 도입합니다.

- **Technical Details**: 컴퓨터 비전에서 representation learning은 개체나 의미 개념에 대한 범용 인코딩을 학습하는 것을 목표로 합니다. 본 연구에서는 사용자가 제공하는 몇 개의 실제 이미지로부터 개인화된 representation을 학습할 수 있는지에 대해 질문하며, 이를 위해 contrastive learning 접근 방식을 제안합니다. 데이터의 부족성과 객체 인식의 미세한 차이를 해결하기 위한 다양한 방법론을 논의합니다.

- **Performance Highlights**: 우리의 제안된 방법은 다양한 downstream 태스크에서 개인화된 representation learning 성능을 크게 향상시킵니다. 데이터셋 전반에 걸쳐 pretrained counterparts에 비해 매우 우수한 성과를 보이며, 기존 데이터셋의 개정 및 새로운 dataset인 PODS를 소개합니다. 또한, 적은 계산 자원에서도 비슷한 결과를 얻을 수 있음을 보여줍니다.



### Can Generative Video Models Help Pose Estimation? (https://arxiv.org/abs/2412.16155)
Comments:
          Project page: this https URL

- **What's New**: 본 연구는 InterPose라는 새로운 방법론을 제안하여 이미지 간의 상대 포즈를 추정하는 문제를 해결하고자 합니다. 이 방법은 사전 훈련된 generative video 모델을 활용하여 두 이미지 사이에 중간 프레임을 생성하여 포즈 추정을 단순화합니다. 기존의 방법들이 격리된 특징을 찾고 매칭하는 데 어려움을 겪었던 반면, InterPose는 풍부한 시각적 정보를 활용하여 포즈 추정의 정확도를 향상시킵니다.

- **Technical Details**: InterPose는 두 이미지 사이의 중간 프레임을 생성하기 위해 off-the-shelf video 모델을 사용합니다. 생성된 프레임과 원본 이미지 쌍을 함께 입력하여 카메라 포즈를 추정하는 과정에서 추가적인 맥락을 제공합니다. 또한, 생성된 비디오가 갖는 시각적 아티팩트나 비현실적인 운동 등이 문제가 될 수 있기에, 셀프 컨시스턴시 점수를 도입하여 최상의 비디오 샘플을 선택하는 방법을 제안합니다.

- **Performance Highlights**: 연구 결과, InterPose는 다양한 데이터셋에서 기존의 state-of-the-art DUSt3R를 초월하는 성능 향상을 보여줍니다. 이 방법은 특히 낮은 중첩성을 가진 이미지 쌍에서도 효과적인 결과를 도출하며, 안정성과 정확도를 모두 향상시킵니다. 본 연구는 generative video 모델을 활용해 포즈 추정 모델을 개선할 수 있는 유망한 경로를 제시하고 있습니다.



### MotiF: Making Text Count in Image Animation with Motion Focal Loss (https://arxiv.org/abs/2412.16153)
Comments:
          TI2V Bench is released in this https URL

- **What's New**: 이번 논문에서는 텍스트 이미지에서 비디오 생성(Text-Image-to-Video, TI2V) 기술의 한계를 극복하기 위해 Motion Focal Loss(MotiF)를 제안합니다. MotiF는 모션이 더 많은 영역에 모델의 학습을 집중시켜 텍스트 정렬(text alignment) 및 모션 생성(motion generation)을 개선하는 접근 방식입니다. 또한, 평가에 적합한 데이터세트 부족 문제를 해결하기 위해 320개의 이미지-텍스트 쌍을 포함하는 TI2V Bench를 제안합니다.

- **Technical Details**: MotiF는 주요 비디오의 모션 강도를 표현하는 모션 히트맵(motion heatmap)을 활용하여 손실 가중치를 할당합니다. 이 방식은 이미지가 제공하는 공간적 신호를 활용하면서도, 모션 중심의 학습을 가능하게 합니다. 이를 통해 모션 패턴에 대한 주의를 증가시키고, 다양한 조건 속에서 생성된 비디오의 질을 향상시킵니다.

- **Performance Highlights**: TI2V Bench에 대한 포괄적인 평가를 통해, MotiF는 9개의 공개 모델을 초월하여 평균 72%의 선호도를 기록하였습니다. 주요 평가 프로토콜은 A-B 테스트를 통해 주관적인 품질 평가를 해주어, 인간 평가자들이 동영상 간의 차별성을 명확히 인지할 수 있도록 합니다. MotiF는 텍스트 정렬과 모션 품질을 크게 개선하는 것으로 나타났습니다.



### Frequency Is What You Need: Word-frequency Masking Benefits Vision-Language Model Pre-training (https://arxiv.org/abs/2412.16148)
- **What's New**: 최근 연구에서 Vision Language Models (VLMs)의 훈련 세트를 줄이는 것이 학습 효율성을 높일 수 있음을 보여주었습니다. 본 논문에서는 단어 빈도 정보가 VLM 훈련 중 최적의 마스킹 전략을 결정하는 데 중요한 요소임을 제시합니다. 이 연구는 CLIPF (Contrastive Language-Image Pre-training with word Frequency Masking)라는 새로운 접근 방식을 도입하며, 이는 입력 토큰 수가 줄어들 때 특히 유리하다는 것을 증명합니다.

- **Technical Details**: CLIPF에서는 단어 빈도를 기반으로 마스킹할 단어를 선택하며, 이 방법은 기존의 CLIPA 마스킹 방식과 비교됩니다. CLIPA는 잘 알려진 네 가지 마스킹 전략인 truncation (절단), random masking (무작위 마스킹), block masking (블록 마스킹), syntax masking (구문 마스킹)을 활용하였으나 빈도 정보를 고려하지 않습니다. 본 논문은 훈련 에포크 수가 마스킹 전략의 성능에 어떻게 영향을 미치는지를 분석하여, 충분한 에포크를 통해 빈도 기반의 CLIPF 마스킹이 우수한 성과를 낸다고 설명합니다.

- **Performance Highlights**: CLIPF 방식은 훈련 에포크 수가 충분할 경우, 모든 CLIPA 마스킹 방법보다 우수한 성능을 보여줍니다. 이 연구는 클립 훈련을 위한 마스킹에서 단어 빈도의 중요성을 강조하며, 결과적으로 CLIPF가 기존의 세 가지 마스킹 방법인 truncation, random, block 마스킹을 초월한다고 밝혀졌습니다. 본 논문의 분석은 VLM 훈련에서 텍스트 마스킹 방식의 성능 차이를 밝히는 데 기여하며, 향후 연구에 있어 재현 가능한 데이터 세트를 제공하여 연구자들이 실험을 반복할 수 있도록 합니다.



### SeagrassFinder: Deep Learning for Eelgrass Detection and Coverage Estimation in the Wild (https://arxiv.org/abs/2412.16147)
- **What's New**: 이 연구는 해초가 전 세계적으로 줄어드는 문제에 대응하기 위해 깊은 학습 (Deep Learning) 기반의 자동화된 해초 탐지 및 면적 추정 방법을 제안합니다. 연구진은 8,300개 이상의 주석이 달린 수중 이미지를 활용하여 ResNet, InceptionNetV3, DenseNet, Vision Transformer와 같은 여러 깊은 신경망 아키텍처를 평가했습니다. 특히 Vision Transformer는 해초 존재 예측에서 0.95 이상의 AUROC 점수를 기록하여 다른 모델들보다 뛰어난 성과를 보여주었습니다. 이 연구는 수동적 방법에 비해 효율적인 수중 영상 데이터 처리 및 해초 분포에 대한 보다 정밀한 정보를 획득할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 다양한 장치를 통해 수집된 수중 비디오 데이터를 효율적으로 주석 처리하는 플랫폼을 사용하였습니다. 연구 목표는 수중에서의 해초 존재 여부를 분류하기 위해 여러 깊은 신경망(DNN) 모델의 전이 학습(Transfer Learning)과 성능을 조사하는 것입니다. 이 과정에서 DHI의 환경 영향 평가(EIA)에서 수집된 데이터 세트를 활용하여 사실적인 환경 조건에서 해초의 특성을 측정합니다. Deep WaveNet 모델을 사용한 수중 이미지 향상을 통해 DNN의 성능을 더욱 향상시키는 방법도 제안되었습니다.

- **Performance Highlights**: Vision Transformer를 포함한 깊은 학습 모델들은 수중에서의 해초 탐지에 있어 강력한 성능을 입증하며, 고도의 정확도로 해초의 존재 여부를 분류할 수 있음을 보여주었습니다. 또한, 이 프로젝트는 해초 관찰 및 보전에 필요한 데이터 처리를 자동화하여 현재의 수동적 방법보다 훨씬 더 빠르고 효율적으로 진행할 수 있도록 합니다. 수동 주석 작업의 시간과 자원 소모를 크게 줄이고, 향후 자율 수중 차량에 의한 자동 환경 모니터링 가능성을 열어줍니다. 따라서 이 연구는 해양 생태학 및 환경 모니터링 분야에서 깊은 학습의 가치와 가능성을 제시합니다.



### Mamba2D: A Natively Multi-Dimensional State-Space Model for Vision Tasks (https://arxiv.org/abs/2412.16146)
- **What's New**: 본 논문은 기존의 Transformer 아키텍처에 비해 보다 효율적이고 강력한 대안인 State-Space Models (SSMs)를 제안합니다. 특히, 기존의 1D SSM을 2D 데이터 (예: 이미지)에 적용하는 것의 한계를 극복하기 위해, Mamba2D라는 새롭고 혁신적인 모델을 소개합니다. Mamba2D는 단일 2D 스캔 방향을 사용하여 입력의 두 차원을 모두 고려하며, 복잡한 공간적 종속성을 효과적으로 모델링합니다.

- **Technical Details**: Mamba2D는 기본적으로 시각적 데이터 처리를 위해 자연스럽게 설계된 새로운 M2D-SSM 블록을 구현하여 정보 흐름을 공간 이웃의 두 차원 전반에서 효과적으로 만들어냅니다. 또한, 일반적인 순차적인 계산 납득이 불가능한 2차원 종속성을 포함하는 레이어를 설계하여 전통적인 합성곱 방식이 아닌 효율적인 웨이브프론트-스캔(wavefront-scan) 컴퓨팅 모델을 제안합니다.

- **Performance Highlights**: Mamba2D는 ImageNet-1K의 top-1 정확도 비교에서 기존 SSM 기반 모델들보다 경쟁력 있는 성능을 보여줍니다. 특히, CNN 또는 Transformer 기반 모델에 비해 적은 매개변수 수로도 우수한 정확도를 달성하고, 구조적으로 일관된 장거리 관계를 형성하는 능력을 강조합니다.



### NeRF-To-Real Tester: Neural Radiance Fields as Test Image Generators for Vision of Autonomous Systems (https://arxiv.org/abs/2412.16141)
- **What's New**: 이 논문에서는 자율 수중 차량(AUV)과 무인 항공기(UAV)에서의 비전 성능을 향상시키기 위한 새로운 접근법을 제안합니다. Neural Radiance Fields(NeRF)를 활용해 현실적이고 다양한 테스트 이미지를 생성하고, 이 이미지를 변형 테스트(framework)인 N2R-Tester에 통합합니다. N2R-Tester는 커스텀 장면의 모델을 학습하고, 변화된 위치에서 테스트 이미지를 렌더링 할 수 있는 도구입니다.

- **Technical Details**: Neural Radiance Fields는 환경의 3D 표현을 생성하기 위해 카메라 이미지와 그 위치 정보를 기반으로 하는 함수 근사화(form approximation)입니다. NeRF는 학습된 장면에서 새로운 관점을 렌더링할 수 있게 해주며, 이를 통해 AUV와 UAV의 비전 구성 요소를 위한 테스트 데이터를 생성할 수 있습니다. 또한, 논문에서는 메타모픽 테스트(metamorphic testing) 기법을 도입하여 시스템의 일관성을 평가하는 방법을 설명합니다.

- **Performance Highlights**: 논문은 N2R-Tester가 AUV와 UAV의 8가지 뷰 시스템 구성 요소에서 효과적이고 다재다능하게 기여함을 보여주는 실험 결과를 제시합니다. N2R-Tester는 다양한 환경 변화에 대응할 수 있는 자율 시스템의 비전 성능 평가를 위한 중요한 도구로 자리잡을 것입니다. 이러한 기술은 자율 점검 및 모니터링의 안전성을 높이는 데 중요한 역할을 할 것으로 기대됩니다.



### Camera-Based Localization and Enhanced Normalized Mutual Information (https://arxiv.org/abs/2412.16137)
- **What's New**: 이번 연구는 자율주행을 위한 견고하고 정밀한 위상 검출 알고리즘을 제안합니다. 저렴한 카메라에서 수집된 이미지 데이터를 활용하며, 차량이 상세한 글로벌 맵을 가지고 있다고 가정합니다. 논문에서는 잡음이 많은 환경에서도 효과적인 localization을 위한 방법론을 모색합니다.

- **Technical Details**: 자동차에 장착된 카메라로 촬영된 이미지를 활용하여 글로벌 맵에서 가장 잘 일치하는 구간을 찾아내는 알고리즘을 다룹니다. 연구는 두 가지 매칭 방식인 표준 내적 (Standard Inner Product, SIP)과 정규화된 상호 정보 (Normalized Mutual Information, NMI)를 간단히 검토하고, 이들 알고리즘의 성능을 개선하기 위한 새로운 방식을 제안합니다. 알고리즘의 성능 향상은 잡음 제어 및 환경 변화로 인한 불확실성을 고려하여 이루어집니다.

- **Performance Highlights**: 논문의 수정된 알고리즘이 잡음이 많은 환경에서 기존 방법보다 뛰어난 성능을 보여준다고 수치 시뮬레이션을 통해 입증되었습니다. 따라 이 알고리즘들은 자율주행 차량의 물리적 제약에 기반한 접근을 통해 의도된 효과성을 달성합니다. 향후 이러한 기술이 자율주행 차량의 상용화에 기여할 수 있을 것으로 기대됩니다.



### LEDA: Log-Euclidean Diffeomorphic Autoencoder for Efficient Statistical Analysis of Diffeomorphism (https://arxiv.org/abs/2412.16129)
- **What's New**: 이 논문에서는 Log-Euclidean Diffeomorphic Autoencoder (LEDA)라는 혁신적인 프레임워크를 제안하여 변형 필드의 주요 로그를 효율적으로 계산할 수 있도록 하며, 이는 복잡하고 비선형적인 변형을 분석하는 데 중요한 역할을 합니다. LEDA는 diffeomorphism 그룹의 작용 법칙을 존중하는 선형 잠재 공간 내에서 작동하여 통계 분석의 강력함과 적용 가능성을 향상시킵니다. 또한, 변형 필드의 정확한 잠재 표현을 보장하는 역일관성(inverse consistency) 제약을 적용하기 위한 손실 함수도 도입했습니다.

- **Technical Details**: 이 프레임워크는 연속적인 제곱근 예측을 통해 변형 필드의 주 로그를 계산하는 데 중점을 두고 있으며, 이는 고전적인 이미지 등록 방법에 비해 계산 비용을 낮추는 데 기여합니다. LEDA는 비선형 변형을 효과적으로 모델링하고 분석할 수 있게 하며, OASIS-1 데이터셋을 사용한 광범위한 실험에서 이의 유효성이 입증되었습니다. 뇌 이미지, 적응 방사선 치료 계획 등 다양한 임상 응용을 위한 개별 변형을 정확하게 포착하고 통합하는 능력도 평가하고 있습니다.

- **Performance Highlights**: LEDA는 복잡한 비선형 변형을 정확하게 모델링하면서도 역일관성을 유지하는 데 효과적입니다. 본 연구에서는 OASIS-1 데이터셋을 활용하여 LEDA의 성능을 확인했으며, 이 프레임워크가 기존 방법들에 비해 우수한 결과를 보여주었음을 시사합니다. 특히, LEDA는 전통적인 방법이 갖는 계산 비용과 수치 오류 문제를 해결함으로써 임상 연구와 실험에서의 적용 가능성을 극대화하고 있습니다.



### PruneVid: Visual Token Pruning for Efficient Video Large Language Models (https://arxiv.org/abs/2412.16117)
Comments:
          Efficient Video Large Language Models

- **What's New**: 이 논문에서는 멀티 모달 비디오 이해의 효율성을 향상시키기 위해 PruneVid라는 시각적 토큰 가지치기 방법을 소개합니다. 기존의 대형 언어 모델(LLMs)이 비디오 작업에서 뛰어난 성과를 보이고 있지만, 비디오 데이터의 중복성 문제로 인해 상당한 계산 비용이 발생합니다. PruneVid는 정적 지역을 식별하고 이러한 중복된 토큰을 줄이는 트레이닝이 필요 없는 방법을 제안하여 계산 효율성을 크게 향상시키는 데 주력합니다.

- **Technical Details**: PruneVid는 공간적 및 시간적 차원에서 비디오 중복성을 줄이고, 질문 토큰과 관련된 시각적 특징만을 선택적으로 가지치기하여 LLM의 추론 능력을 활용합니다. 이 방법은 정적 토큰을 병합하고 유사한 공간적 토큰을 클러스터링하여 효율적인 토큰 처리를 수행합니다. 또한, 질문과 비디오 토큰 간의 어텐션 점수를 사용하여 질문에 대한 관련 시각적 토큰을 판별하고 보존합니다.

- **Performance Highlights**: PruneVid는 다양한 비디오 벤치마크에서 80% 이상의 시각적 토큰을 가지치기하면서도 성능 저하를 최소화하였으며, 일부 경우에는 모델 성능을 개선하기도 했습니다. 우리의 실험 결과는 PruneVid가 기존 방법들에 비해 더 우수한 효율성과 효과성을 발휘하며, 추론 속도를 최대 1.55배 향상시키고 FLOP을 74%에서 80%까지 줄인다는 것을 보여줍니다.



### CLEAR: Conv-Like Linearization Revs Pre-Trained Diffusion Transformers Up (https://arxiv.org/abs/2412.16112)
- **What's New**: 이번 논문에서는 이미지 생성의 선두적 아키텍처인 Diffusion Transformers (DiT)가 높은 해상도 이미지를 생성할 때 발생하는 주의 메커니즘의 제곱 복잡도로 인해 지연 시간이 크게 증가하는 문제를 다룹니다. 이를 해결하기 위해, 우리는 DiT의 복잡성을 선형으로 줄이는 선형 주의 메커니즘을 제안합니다.

- **Technical Details**: 기존의 효율적인 주의 메커니즘에 대한 포괄적인 요약을 시작으로, 선형화를 위한 네 가지 주요 요소를 식별하였습니다: 지역성(locality), 공식 일관성(formulation consistency), 고차원 주의 맵(high-rank attention maps), 그리고 피쳐 무결성(feature integrity)입니다. 이를 바탕으로, 우리는 CLEAR라는 이름의 지역 주의 전략을 도입하여 각 쿼리 토큰 주위의 지역 창(window)으로 피쳐 상호작용을 제한하여 선형 복잡성을 달성합니다.

- **Performance Highlights**: 실험 결과, 10K 자가 생성 샘플을 10K 반복하여 주의 레이어를 미세 조정함으로써, 선형 복잡성을 가진 학생 모델로 프리트레인된 DiT의 지식을 효과적으로 전이할 수 있음을 보여주었습니다. 이 접근법은 주의 계산을 99.5% 줄이고 8K 해상도 이미지를 생성할 때 생성 속도를 6.3배 가속화하는 성과를 거두었습니다.



### Demystifying the Potential of ChatGPT-4 Vision for Construction Progress Monitoring (https://arxiv.org/abs/2412.16108)
- **What's New**: 이번 논문은 OpenAI의 GPT-4 Vision과 같은 Large Vision-Language Models (LVLMs)의 발전이 인공지능 분야에서 특히 시각 데이터 분석 및 해석에 있어 중요한 진전을 이뤘음을 보여줍니다. 실질적으로 건설 산업에서 GPT-4 Vision의 응용을 탐구하며, 이를 통해 건설 프로젝트의 진행 상황을 모니터링하고 추적하는 능력에 주목하고 있습니다.

- **Technical Details**: 연구는 고해상도 항공 이미지를 활용하여 건설 현장의 세부 장면 분석을 수행하고 시간에 따른 발전 변화를 추적합니다. GPT-4 Vision은 건설 단계, 자재 및 기계 식별에서 뛰어난 성능을 보이는 반면, 정확한 객체 위치 파악(Object Localization) 및 분할(Segmentation)에서 어려움을 겪고 있다는 점이 언급됩니다.

- **Performance Highlights**: 비록 이러한 한계가 표면화되긴 했지만, 이 기술의 미래 발전 가능성은 매우 큽니다. 본 연구는 현 시점에서 LVLMs를 건설 분야에 적용하는 상태와 기회를 강조할 뿐만 아니라, 도메인 특화 교육(Domain-specific Training) 및 다른 컴퓨터 비전 기법과 디지털 트윈(Digital Twins) 통합을 통해 모델의 유용성을 향상시킬 수 있는 미래 방향도 논의하고 있습니다.



### SegCol Challenge: Semantic Segmentation for Tools and Fold Edges in Colonoscopy data (https://arxiv.org/abs/2412.16078)
Comments:
          4 pages, 1 figure. Dataset introduction for the SegCol Challenge at MICCAI 2024. Full Challenge paper, including participant methods and evaluation results, will be released soon

- **What's New**: 이번 논문은 대장암(Colorectal Cancer, CRC) 검출 및 내시경(naviculoscopic) 탐지를 위한 새로운 데이터셋인 SegCol을 소개합니다. SegCol은 EndoMapper 저장소에서 수집된 96개의 내시경 동영상에서 수작업으로 주석을 단 픽셀 수준의 의미적 레이블을 제공합니다. 이 데이터셋은 대장 주름(fold edges) 및 내시경 도구의 탐지를 개선하기 위해 구성되었으며, MICCAI 2024의 Endovis 챌린지의 일환으로 개최됩니다.

- **Technical Details**: SegCol 데이터셋은 연속적으로 40프레임이 샘플링된 이미지로 구성되어 있으며, 주름의 가장자리와 다양한 내시경 도구를 구분하여 주석을 달았습니다. 주름은 1픽셀 두께의 윤곽선으로 레이블링되어 자연학적 랜드마크를 형성하며, 내시경 도구에 대해서도 밀집된 픽셀 마스크가 제공됩니다. 이 데이터셋은 내시경의 깊이 인식 및 국소화 방법을 개선하는 것을 목표로 하고 있으며, 각 주석은 외부 팀에 의해 3개월에 걸쳐 수행되었습니다.

- **Performance Highlights**: 이 논문은 SegCol 챌린지의 두 가지 과제를 제안합니다. 첫 번째 과제는 레이블이 지정된 훈련 데이터를 사용하여 해부학적 구조물과 수술 도구의 세분화를 정확하고 강건하게 수행하는 세분화 아키텍처를 설계하는 것입니다. 두 번째 과제는 비(非) 레이블 훈련 세트에서 가장 유용한 400프레임을 선택하는 샘플링 전략을 개발하는 것입니다. 종합적인 평가 프레임워크가 설계되어 세분화 품질을 평가하며, 이는 대장내시경 이미지를 다루는 다양한 도전과제를 고려하고 있습니다.



### Label-Efficient Data Augmentation with Video Diffusion Models for Guidewire Segmentation in Cardiac Fluoroscopy (https://arxiv.org/abs/2412.16050)
Comments:
          AAAI 2025

- **What's New**: 이 연구는 Segmentation-guided Frame-consistency Video Diffusion Model (SF-VD)을 제안하여 안내선(segmentation guidewire)의 정확한 세분화를 위한 라벨 효율적인 데이터 증강 기법을 개발했습니다. 이 모델은 제한된 주석 데이터로부터 대량의 라벨링된 플루오로스코피(Fluoroscopy) 비디오를 생성함으로써 세분화 네트워크의 성능을 향상시키는 데 기여합니다. 또한, 이 연구는 의료 데이터 증강을 위해 생성 모델을 활용한 첫 번째 사례이며, 비디오 프레임 간의 일관성을 유지하면서 다양한 배경과 와이어 모양을 생성합니다.

- **Technical Details**: SF-VD는 두 개의 별도 2D Diffusion 모델을 활용하여 장면 분포(scene distribution)와 프레임 간 이동(motion distribution)을 독립적으로 학습합니다. 첫 번째 모델은 지정된 와이어 마스크에 따라 와이어가 배치된 2D 플루오로스코피 이미지를 생성하고, 두 번째 모델은 이러한 정적 이미지를 기반으로 다음 프레임을 생성하여 프레임 간 일관성을 보장합니다. 이 과정에서 와이어의 대비를 조정하여 다양성을 높이는 세분화 가이드 메커니즘을 사용하여 생성된 이미지의 가시성을 향상시킵니다.

- **Performance Highlights**: 연구 결과는 SF-VD가 기존 데이터 증강 방법보다 우수한 성능을 보여주며, 모든 테스트된 모델에서 세분화 성능을 향상시켰음을 입증했습니다. 특히, SF-VD는 제한된 라벨의 데이터로부터 생성된 비디오에서 높은 퀄리티를 유지하며 가이드와이어 세분화의 성능을 현저히 개선했습니다. 이로 인해 SF-VD는 향후 심장 개입 수술과 같은 의료 분야에서 중요한 역할을 할 것으로 기대됩니다.



### Segmentation of arbitrary features in very high resolution remote sensing imagery (https://arxiv.org/abs/2412.16046)
Comments:
          Main article: 18 pages, 9 figures; appendix: 17 pages, 9 figures

- **What's New**: 이번 연구는 EcoMapper라는 새로운 딥 러닝 기반의 도구를 소개하며, 이는 매우 고해상도(VHR) 원격 감지(RS) 이미지에서 다양한 특징을 자동으로 분할(segmentation)하는 기능을 제공합니다. EcoMapper는 지리 공간 데이터의 처리, 딥 러닝 모델 훈련과 추론을 완전히 자동화하여, 여러 맥락에서 활용될 수 있는 대안적인 솔루션입니다. 이 과정에서, 실제 UAV 데이터셋을 활용하여 두 가지 다양한 특징을 성공적으로 분할하였으며, 기존의 특정 맥락에 맞춘 모델들과 경쟁할 만한 성과를 달성했습니다.

- **Technical Details**: EcoMapper는 다양한 데이터 세트 및 특징에 반복적으로 적용할 수 있는 솔루션을 제공하기 위해 기존의 모델 선택과 훈련 방식을 간소화했습니다. 이 도구는 MMSegmentation 라이브러리와 통합되어 모델의 교체와 업데이트를 보다 쉽게 설정할 수 있으며, 사용자는 코드 작성 없이 커맨드라인에서 바로 실행할 수 있습니다. 자동화된 데이터 레이블링 지원 기능을 통해 모든 사용자들이 쉽게 세분화 워크플로우를 사용할 수 있도록 설계되었습니다.

- **Performance Highlights**: EcoMapper는 SAM(세그먼트 모든 것 모델)보다 특정 특징에 대해 더 나은 성과를 보였으며, 특히 일반적으로 발생하지 않는 특징에 대한 정확한 분할을 가능하게 합니다. 이 연구는 특징 크기와 원격 감지 이미지의 해상도가 모델 성능에 미치는 영향을 분석하여, 최적의 지상 샘플링 거리(Ground Sampling Distance, GSD) 도출과 같은 새로운 통찰을 제공합니다. 또한, 필드 조사 방법론을 종합적으로 개발하여 딥 러닝 방법이 효과적으로 적용될 수 있도록 지원합니다.



### SafeCFG: Redirecting Harmful Classifier-Free Guidance for Safe Generation (https://arxiv.org/abs/2412.16039)
- **What's New**: 이번 연구에서 제안된 Harmful Guidance Redirector (HGR)는 diffusion models (DMs)의 안전성과 품질을 동시에 향상시키는 혁신적인 방법론입니다. HGR은 harmful 이미지 생성을 방지하기 위해 harmful CFG 방향을 재조정하면서, clean CFG 방향은 유지하여 SafeCFG로 변환합니다. 이를 통해 기존의 안전가이드 제공 방식의 한계를 극복하고, 비지도 학습 방식으로 안전한 DMs을 효율적으로 훈련할 수 있는 가능성을 제시합니다. 최종적으로 HGR은 이미지 생성 과정에서 높은 품질과 안전성을 동시에 달성할 수 있음을 보여줍니다.

- **Technical Details**: HGR은 Transformer 기반의 플러그인으로, DMs의 파라미터를 변경하지 않고도 적용될 수 있습니다. 이는 harmful data의 CFG 방향을 안전한 방향으로 재조정하여 CFG를 SafeCFG로 전환하는 기능을 가지고 있습니다. 연구에서는 DMs이 여러 harmful CFG 방향을 동시에 재조정할 수 있도록 학습했으며, 이는 다양한 유해 요소를 제거하면서도 높은 품질의 이미지를 유지할 수 있게 합니다. 또한 HGR은 이미지의 유해성을 탐지할 수 있어, 사전 정의된 clean 또는 harmful 레이블 없이 안전한 diffusion 모델을 비지도 방식으로 미세 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과 HGR을 통합한 diffusion models은 높은 품질과 강력한 안전성을 동시에 달성한 것으로 나타났습니다. SafeCFG를 통해 생성된 이미지는 유해성이 낮고 품질이 높아, 비지도 학습을 통해 훈련된 안전 DM도 훌륭한 안전 성능을 보였습니다. 이를 통해 HGR이 DMs의 안전성을 개선하는 데 효과적임을 입증하며, 연구 결과는 다양한 응용 분야에서 안전한 이미지 생성의 가능성을 제시합니다.



### CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images (https://arxiv.org/abs/2412.16028)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 CoCoGaussian이라는 새로운 방법을 제안합니다. 이 방법은 Circle of Confusion (CoC)를 인지하는 Gaussian Splatting을 활용하여, 희미한 이미지만으로 정밀한 3D 장면 표현을 가능하게 합니다. CoCoGaussian은 물리적 원리에 기반하여 defocus blur 문제를 해결하며, 깊이 및 learnable aperture 정보를 통해 CoC 직경을 계산합니다.

- **Technical Details**: CoCoGaussian은 3D Gaussian을 사용하여 저조도 깊이에서 얻은 정보를 활용하여 CoC의 형태를 정확하게 모델링합니다. 전달해야 할 많은 Gaussian을 생성함으로써 CoC의 형태를 포착하며, 학습 가능한 스케일링 팩터를 도입하여 반사 또는 굴절이 있는 표면에서의 신뢰할 수 없는 깊이에 대한 우수성을 제공합니다. 이 방법은 동적인 심도 제어 및 다양한 초점 조정 요구 사항에 대한 사용자 정의 가능한 장면 시각화를 가능하게 합니다.

- **Performance Highlights**: 실험 결과 CoCoGaussian은 Deblur-NeRF 데이터셋과 DoF-NeRF 실험실 기반 데이터셋에서 최고의 성능을 달성했습니다. CoCoGaussian은 단지 희미한 이미지만으로도 정확한 3D 장면 복원 및 선명한 novel view 합성을 가능하게 합니다. 본 모델은 정량적 및 정성적 모두에서 최첨단 성능을 제공하며, 다양한 장면 시나리오에 광범위하게 적합합니다.



### MR-GDINO: Efficient Open-World Continual Object Detection (https://arxiv.org/abs/2412.15979)
Comments:
          Website: this https URL . Code is available at: this https URL

- **What's New**: 본 논문은 오픈 월드(Open-World) 지속 객체 탐지(Open-World Continual Object Detection, OW-COD) 작업을 제안하여, 기존 학습된 클래스와 새로운 클래스, 그리고 보이지 않는 카테고리에서 객체 탐지 기능을 유지하려는 새로운 과제를 다룹니다. MR-GDINO라는 새로운 모델을 통해, 잊어버림 현상을 최소화하면서도 효율적인 메모리와 검색 메커니즘을 이용하여 성능을 높입니다. 실험 결과, 기존의 지속 탐지 모델들이 상당한 잊어버림 현상을 겪는 반면, MR-GDINO는 0.1%의 추가 파라미터만으로도 성능을 크게 향상시킵니다.

- **Technical Details**: MR-GDINO는 대량의 메모리를 사용해 새로운 개념과 시각-언어 상호작용에 대한 파라미터를 저장하고, 추론 과정에서 최적의 파라미터를 검색하여 보존된, 새로 적응된 또는 오픈 월드 시나리오의 객체를 탐지합니다. 또한, OW-COD 벤치마크는 다양한 도메인에서의 평가 샘플과 함께 소수 샷(few-shot) 학습 데이터를 제공하여 기존 클래스 및 새로운 클래스에서의 탐지 성능을 평가합니다. 이러한 메커니즘은 유연성과 확장성을 제공하여 기존, 새로운 및 보이지 않는 오픈 월드 카테고리의 탐지 기능을 유지하는 데 기여합니다.

- **Performance Highlights**: MR-GDINO는 소수의 추가 파라미터만으로도 GDINO보다 큰 성능 향상을 달성하며, 이는 보이는 클래스에서의 성능을 크게 개선합니다. 또한, 강력한 검색 메커니즘 덕분에 보이지 않는 클래스와 보이는 클래스에서 성능을 동시에 유지할 수 있습니다. 실험 결과, OW-COD 벤치마크는 MR-GDINO의 뛰어난 성능을 증명하며, 실세계 환경에서의 적용 가능성을 강조합니다.



### Self-Supervised Radiograph Anatomical Region Classification -- How Clean Is Your Real-World Data? (https://arxiv.org/abs/2412.15967)
Comments:
          12 pages, 4 figures, 2 supplementary figures

- **What's New**: 이 연구는 자가 감독(self-supervised) 기법과 감독 대비(supervised contrastive) 깊이 학습법을 사용하여 14개의 해부학적 영역 클래스를 정확하게 분류할 수 있음을 입증합니다. 48,434개의 골격 방사선 사진이 포함된 데이터셋을 사용하여 단일 모델에서는 96.6%의 선형 평가 정확도를, 앙상블 방식에서는 97.7%의 정확도를 달성하였습니다. 특히, 훈련 세트의 1%에 해당하는 몇 개의 레이블된 인스턴스만으로도 92.2%의 정확도를 얻을 수 있어 자원이 부족한 상황에서도 사용 가능함을 보여줍니다.

- **Technical Details**: 연구는 14개의 해부학적 영역을 대상으로 48,434개의 방사선 사진을 DICOM 형식으로 분석하였습니다. OpenCV를 사용하여 이미지의 테두리와 회전을 정규화하였으며, 수술 계획의 원형 게이지 개선을 위해 새로운 데이터 증강 기법을 도입하였습니다. 또한, AspNet18 아키텍처를 기반으로 하여 1000 에폭 동안 Adam 최적화 알고리즘을 사용하여 사전 학습(pretraining)을 수행하였습니다.

- **Performance Highlights**: 모델의 성능은 실험 결과로 확인되었으며, 전문가의 후속 분석을 통해 35%의 잘못된 레이블과 11%의 도메인 외 이미지가 발견되었습니다. 이러한 오류를 고려했을 때, 단일 모델의 해부학적 영역 레이블링 성능은 이론적으로 98.0%까지 증가하고, 앙상블 사용 시에는 98.8%에 도달하는 성과를 보였습니다. 이 연구는 PACS 시스템의 데이터 품질을 높이기 위한 중요한 기여를 할 것으로 기대됩니다.



### Monkey Transfer Learning Can Improve Human Pose Estimation (https://arxiv.org/abs/2412.15966)
- **What's New**: 이 연구에서는 마카크 원숭이의 전이 학습(transfer learning)이 인간의 포즈 추정(human pose estimation) 향상에 기여할 수 있는지를 조사했습니다. 현재의 포즈 추정 기술은 비임상 데이터셋에서는 인간의 주석과 유사한 성능을 보이나, 새로운 상황에서는 낮은 성능을 보입니다. 본 연구에서 우리는 다양한 동물 데이터 사용이 이러한 결점을 보완할 수 있다는 점을 발견했으며, 마카크 데이터로 훈련된 네트워크가 인간 포즈 추정의 정확도를 높인다고 제안합니다.

- **Technical Details**: 본 연구는 마카크 원숭이를 위한 포즈 추정 네트워크를 기반으로 하여 인간 포즈 추정을 개선하는 방법론을 채택했습니다. 마카크 네트워크는 14,697장의 이미지로 훈련되었으며, 이에 대한 포즈 이미지들은 수작업으로 주석이 달렸습니다. 이후 MPII 데이터셋에서 1,000개의 인간 이미지를 활용하여 전이 학습 모델이 생성되었습니다. 모델 훈련 시, 설정된 매개변수와 다양한 전처리를 통해 두 네트워크의 성능 비교를 공정하게 진행했습니다.

- **Performance Highlights**: 마카크 데이터를 활용한 전이 학습 방법은 기존의 인간 데이터만으로 훈련된 벤치마크 모델보다 더 적은 수의 훈련 예시(1,000 vs 19,185)로도 높은 정확도와 재현율을 달성했습니다. 이는 마카크 포즈 추정이 임상 상황에서 인간 포즈 추정을 개선할 수 있다는 가능성을 제시합니다. 향후 연구에서는 마카크 데이터로 훈련된 포즈 추정의 임상 인구에 대한 유용성을 더 탐구하는 것이 필요합니다.



### Reframing Image Difference Captioning with BLIP2IDC and Synthetic Augmentation (https://arxiv.org/abs/2412.15939)
Comments:
          This paper has been accepted for the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 최근 몇 년 간 생성 모델의 품질이 향상되면서 이미지의 편집된 변형을 대규모로 생성할 수 있게 되었습니다. 이러한 기술의 유해한 영향을 줄이기 위해, 본 논문에서는 두 이미지의 차이를 설명하는 Image Difference Captioning (IDC) 작업을 다루고 있습니다. 기존의 아이디어를 바탕으로 저자들은 BLIP2 모델을 IDC 작업에 적응시키고 IDC 데이터셋을 증강하기 위한 효과적인 프레임워크를 제안하고 있습니다.

- **Technical Details**: 논문에서는 BLIP2 IDC(Adaptation of BLIP2 to Image Difference Captioning)를 소개하고 있으며, 이를 통해 효율적인 낮은 계산 비용으로 IDC 작업에 적합하게 조정할 수 있음을 보여줍니다. 또한, 생성 모델을 활용해 합성 데이터를 생성하여 IDC 모델의 성능을 향상시키는 방법을 탐구하고 있습니다. BLIP2는 이미지 캡셔닝을 위한 사전 훈련을 통해 IDC에 조정될 수 있으며, 이는 두 이미지를 동시에 입력받아 비교하는 방식으로 작동합니다.

- **Performance Highlights**: BLIP2IDC는 실세계 IDC 데이터셋에서 두 개의 스트림 접근 방식을 대폭 초월하는 성능을 발휘하는 것으로 나타났습니다. 신뢰할 수 있는 고품질 IDC 모델을 생성하기 위해, 저자들은 Syned1이라는 새로운 데이터셋도 제안하였으며, 이는 IDC 작업에 적합한 도전적인 데이터셋으로 미래의 연구에 기여할 것입니다. 마지막으로 여러 최신 모델에 대한 종합적인 평가를 제공함으로써 IDC 분야에서 현재 모델의 능력에 대한 더 명확한 이해를 돕고 있습니다.



### MiniGPT-Pancreas: Multimodal Large Language Model for Pancreas Cancer Classification and Detection (https://arxiv.org/abs/2412.15925)
- **What's New**: 본 연구에서는 MiniGPT-Pancreas라는 다중 모달 대규모 언어 모델(MLLM)을 제안하여, pancreatitis 진단을 지원하는 대화형 챗봇 기능을 통합하고 있다. 이 모델은 NIH 및 MSD 데이터셋에서의 CT 스캔과 질문을 결합한 모달 프롬프트를 사용하여 미세조정(fine-tuning)을 수행하였다. MiniGPT-Pancreas는 췌장 종양의 분류와 탐지에 있어 중요한 진전을 보이며, 기존 모델들을 초월한 성능을 달성하였다.

- **Technical Details**: MiniGPT-Pancreas는 췌장 탐지, 종양 분류, 그리고 다중 장기 탐지 작업을 포함한 다양한 의료 문제를 해결하기 위해 MiniGPT-v2를 기반으로 한 계단식 미세 조정(cascaded fine-tuning)을 적용하였다. 이를 위해 AbdomenCT-1k 데이터셋을 사용하여 간, 비장, 신장, 췌장 등의 장기 감지를 수행하였다. 췌장 암 분류 업무에서 MiniGPT-Pancreas는 정확도(accuracy) 0.876과 정밀도(precision) 0.874, 재현율(recall) 0.878을 기록하였다.

- **Performance Highlights**: MiniGPT-Pancreas는 NIH와 MSD 데이터셋에서 췌장 탐지의 IoU(Intersection over Union) 점수가 각각 0.595와 0.550을 기록하였다. AbdomenCT-1k 데이터셋에서 간, 신장, 비장, 췌장에 대한 IoU는 각각 0.8399, 0.722, 0.705, 0.497로 나타났다. 그러나 췌장 종양 탐지의 IoU는 0.168로 상대적으로 낮아, 향후 연구가 필요하다.



### Watertox: The Art of Simplicity in Universal Attacks A Cross-Model Framework for Robust Adversarial Generation (https://arxiv.org/abs/2412.15924)
Comments:
          18 pages, 4 figures, 3 tables. Advances a novel method for generating cross-model transferable adversarial perturbations through a two-stage FGSM process and architectural ensemble voting mechanism

- **What's New**: 우리는 Watertox라는 새로운 적대적 공격 프레임워크를 제안합니다. 이 프레임워크는 아키텍처의 다양성과 정밀 제어 (precision-controlled)된 잡음을 통해 놀라운 효과를 달성합니다. 두 단계의 Fast Gradient Sign Method (FGSM)를 결합하여 기본적인 잡음과 타겟 확대를 전략적으로 진행함으로써 효과적인 공격을 구현합니다.

- **Technical Details**: Watertox의 중심에는 정밀 제어가 가능한 두 단계의 FGSM 구현이 있습니다. 첫 번째 단계에서 균일한 잡음으로 기본적인 파괴를 생성하고 두 번째 단계에서는 중요한 영역을 선택적으로 강화하여 시각 품질을 유지합니다. 여러 현대 및 고전 모델을 활용한 앙상블 아키텍처를 통해 다양한 신경망에서 효과적으로 전이할 수 있는 잡음을 생성합니다.

- **Performance Highlights**: 상태에 따라 가장 뛰어난 모델의 정확도를 70.6%에서 16.0%로 감소시키며, 제로샷 공격 평가에서는 최대 98.8%까지 정확도를 감소시키는 결과를 보였습니다. 이러한 결과들은 Watertox가 적대적 공격 방법론에서 중요한 발전을 이루었음을 드러내며, 시각 보안 시스템과 CAPTCHA 생성에서의 응용 가능성을 시사합니다.



### CCNDF: Curvature Constrained Neural Distance Fields from 3D LiDAR Sequences (https://arxiv.org/abs/2412.15909)
Comments:
          ACCV 2024, Oral Presentation

- **What's New**: 본 논문에서는 Neural Distance Fields (NDF)의 학습을 향상시키기 위해, signed distance field의 두 번째 미분을 활용한 새로운 감독 방법론을 제안합니다. 기존 접근방식들이 대규모 야외 장면에서의 참조 NDF 부족 문제를 간과하는 반면, 우리는 NDF의 기하학적 특성을 더욱 정확하게 추정할 수 있도록 이 방법을 발전시켰습니다. 이를 통해 우리는 NDF의 정확성 및 신뢰성을 높이는 데 기여하고자 합니다.

- **Technical Details**: NDF는 연속적인 3D 재현을 제공하는 특성 덕분에 3D 컴퓨터 비전 및 로봇 분야에서 주목받고 있습니다. 전통적으로 NDF 교육은 진리 데이터에 의존했으나, 본 연구에서는 LiDAR 센서 데이터만을 활용하여 모델을 교육하는 접근법을 소개합니다. 우리의 방법론은 고차 미분을 사용하여 signed distance를 보다 정밀히 모델링하고, NDF의 기하학적 특성과 함께 정렬될 수 있도록 합니다.

- **Performance Highlights**: 우리의 제안 방법론은 NDF의 매핑 및 로컬라이제이션 작업에서 최신 기법들과 비교할 때 SOTA 성능을 달성하였습니다. 특히, 큰 규모의 환경에서의 맞춤형 기하 정보를 고려한 접근 덕분에, 더욱 실제적이고 효율적인 NDF의 적용 가능성을 보여줍니다. 이를 통해 성능 향상을 이루고, 실세계 응용에서 NDF의 유용성을 높이고자 합니다.



### NeuroPump: Simultaneous Geometric and Color Rectification for Underwater Images (https://arxiv.org/abs/2412.15890)
- **What's New**: 이번 논문에서는 NeuroPump라는 자가 지도(self-supervised) 방법을 제안하여 수중 영상의 기하학적 왜곡과 색상 왜곡을 동시에 수정하는 방법을 소개합니다. 기존 연구들은 일반적으로 색상 또는 기하학적 왜곡 중 하나에만 초점을 맞추었으나, NeuroPump는 두 가지를 모두 해결할 수 있는 혁신적인 접근 방식을 취합니다. 또한, 실제 쌍(pair) 이미지로 구성된 새로운 360도 수중 벤치마크 데이터셋도 함께 제안하여 모델 평가의 정확성을 높였습니다.

- **Technical Details**: NeuroPump는 NeRF(Neural Radiance Field) 파이프라인 내에서 굴절(refraction), 흡수(absorption) 및 산란(scattering)을 명시적으로 모델링하여 색상과 기하학적 왜곡을 동시에 수정하는 방법입니다. 이 방법은 렌즈와 물의 인터페이스에서의 굴절을 모델링하고, Snell's law에 따라 광선을 굴절시켜 기하학을 조정합니다. 또한, 색상 왜곡 수정의 경우, 수중에서의 직접 방사 및 후방 산란으로 이루어진 색상을 분석하여 색상의 강도를 보정하는 방법을 사용합니다.

- **Performance Highlights**: 제안된 NeuroPump는 양적 및 질적 기준 모두에서 기존의 방법들보다 뛰어난 성능을 보여줍니다. 실험 결과, NeuroPump는 색상과 기하학적 왜곡을 동시에 보정하면서도 추가적인 새로운 시각 효과 및 광학 효과를 합성할 수 있는 유연성을 제공합니다. 이러한 결과는 수중 영상 복원 분야에서 NeuroPump의 혁신성을 입증합니다.



### IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing (https://arxiv.org/abs/2412.15867)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 inter-reflective Gaussian splatting (IRGS)라는 새로운 역 렌더링 프레임워크를 제안합니다. IRGS는 전체 렌더링 방정식을 간소화 없이 통합하여 복잡한 간접 반사를 정확하게 캡처합니다. 이를 위해, 연구진은 2D Gaussian ray tracing 기법을 도입하여 실시간으로 입사광에 대한 간접 방사(Light) 를 계산합니다. 이 접근 방식은 역 렌더링에서 Gaussian splatting의 효율성과 정확성을 동시에 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: IRGS 프레임워크는 3D 장면을 Gaussian primitives로 모델링하고, 2D Gaussian ray tracing을 사용하여 깊이 맵에 따라 렌더링 방정식의 전 과정을 계산합니다. 이 방식은 이전의 3DGS보다 향상된 투명도를 제공하며, 실시간으로 뒷전파를 통해 간접 조명 최적화를 가능하게 합니다. 주요 기술인 2D Gaussian ray tracing은 Gaussian splat 간의 교차점을 정의하고, 이를 기반으로 복잡한 물리적 상호작용을 모사할 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 표준 벤치마크에서 extensive 실험을 통해 IRGS의 효과성을 검증했습니다. 결과적으로, IRGS는 직접적 및 간접적인 조명을 모두 고려하여 뛰어난 inter-reflection 효과를 모델링할 수 있는 능력을 입증했습니다. 특히, 다양한 복잡한 장면에서 IRGS의 성능이 기존 접근 방식에 비해 현저히 향상된 것을 보여주었습니다.



### Semi-Supervised Adaptation of Diffusion Models for Handwritten Text Generation (https://arxiv.org/abs/2412.15853)
- **What's New**: 이번 논문은 기존의 손글씨 텍스트 생성(HTG) 접근 방식에 대한 새로운 확장을 제안합니다. 연구팀은 마스크된 오토인코더(masked autoencoder)를 활용하여 훈련 중 보지 못했던 쓰기 스타일을 학습할 수 있는 방법을 발견하였습니다. 이로 인해 HTG의 훈련 이미지 품질을 향상시키고, 새로운 데이터 세트에 반영할 수 있는 능력이 증가합니다.

- **Technical Details**: 제안된 시스템은 레이턴트(diffusion model) 기반의 손글씨 텍스트 생성 모델을 포함합니다. 사용된 내용 인코더(content encoder)는 텍스트 및 서예적(calligraphic) 특성을 적절히 조정하기 위한 다양한 방법을 제공합니다. 또한, Classifier-free guidance를 활용하여 생성된 훈련 이미지의 품질을 높이는 방안도 모색하고 있습니다.

- **Performance Highlights**: IAM 데이터베이스와 RIMES 데이터베이스를 통해 제안된 모델의 성능을 평가하였으며, 특히 이전에 보지 못한 데이터의 생성에서 개선된 결과를 도출했습니다. 시뮬레이션 실험을 통해 새로운 데이터 세트에 대한 반응과 질을 주의 깊게 관찰하였으며, 세미-슈퍼바이즈드(식별 없는 반 감독) 학습이 효과적임을 입증하였습니다.



### Multi-dimensional Visual Prompt Enhanced Image Restoration via Mamba-Transformer Aggregation (https://arxiv.org/abs/2412.15845)
- **What's New**: 이 논문은 "all-in-one" 이미지 복원 모델에서 Mamba와 Transformer의 장점을 결합하여 계산 효율성을 유지하면서도 성능을 향상시키는 새로운 접근 방식을 제안합니다. 주목할 점은 입력 이미지의 공간적 종속성을 캡처하기 위해 Mamba의 선택적 스캐닝 메커니즘을 활용하면서, 채널 모델링에는 Transformer의 자기 주의 메커니즘을 적용한 것입니다. 이를 통해 다중 해상도의 정보 흐름(prompt-flows)을 학습하는 새로운 모듈도 도입하여, 이미지 복원에 있어 다차원적 특성을 최대화하고 있습니다.

- **Technical Details**: 제안된 방법론은 MTAIR(Mamba-Transformer Aggregation)로 명명되며, 공간 모델링을 위한 Mamba의 선택적 스캐닝과 채널 모델링을 위한 Transformer의 자기 주의 메커니즘을 결합하여 두 모델의 상호 보완적인 장점을 활용합니다. 또한, Spatial-Channel Prompt Blocks(S-C Prompts)이라는 다차원적 프롬프트 학습 모듈을 설계하여 고해상도 이미지의 다양한 손상을 효과적으로 처리할 수 있는 능력을 배양합니다. 이를 통해 제안된 모델은 단一한 복원 작업을 넘어 여러 복원 작업에서 유용함을 증명하고 있습니다.

- **Performance Highlights**: 제안된 MTAIR는 이미지 복원 벤치마크 작업에서 최근의 여러 주류 방법들과 비교해 새로운 최첨단 성능을 달성하였으며, 이미지 denoising, dehazing, deraining 등의 다양한 복원 작업에서 그 효과를 입증했습니다. 특히, 고해상도 이미지를 처리할 때 효율성을 보장하며, 이는 계산 자원이 제한된 환경에서도 유리한 부담을 줄여줍니다. 논문은 코드와 사전 훈련된 파라미터를 GitHub를 통해 공개할 예정입니다.



### Efficient Curation of Invertebrate Image Datasets Using Feature Embeddings and Automatic Size Comparison (https://arxiv.org/abs/2412.15844)
Comments:
          Accepted to IEEE CIETES 2025

- **What's New**: 이번 논문에서는 환경 모니터링을 위한 대규모 이미지 데이터셋을 효과적으로 관리할 수 있는 새로운 방법을 제안합니다. 특히, 동일한 생물군에 대한 여러 이미지를 포함한 곤충 및 절지동물 이미지를 대상으로 하는 기법입니다. 제안된 방법은 사전 훈련된 딥 뉴럴 네트워크(Deep Neural Networks)를 활용해 이미지의 feature embedding을 추출하고, 이를 통해 시각적으로 가장 독특한 이미지를 선택합니다.

- **Technical Details**: 데이터셋 관리 방식으로는 이미지 내용 비교 및 크기 비교의 두 가지 방법이 사용됩니다. 이미지 내용 비교는 feature embeddings를 계산하여 이루어지며, 이는 Dense Vector로 표현됩니다. 본 연구에서는 MobileNet-v3 모델을 사용해 feature embeddings를 추출하고, 각각의 그룹별로 평균 feature embedding을 계산하여 다른 이미지와의 유사도를 평가합니다.

- **Performance Highlights**: 제안된 방법은 부적합한 이미지를 검출하는 초기 과정에서 뛰어난 성과를 보여줍니다. 이를 통해 데이터 정제 과정에서의 오류를 줄이고 높은 품질의 훈련 데이터셋을 생성할 수 있습니다. 본 연구에서 제안된 메트릭(metrics) 또한 인간 전문가가 최종 판단을 내리는 과정에서 기여할 수 있는 중요한 도구로 자리 잡을 것입니다.



### Enhancing Generalized Few-Shot Semantic Segmentation via Effective Knowledge Transfer (https://arxiv.org/abs/2412.15835)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문은 Generalized Few-Shot Semantic Segmentation(GFSS) 분야에서 성능을 향상시키기 위한 새로운 접근법인 GFSS-EKT를 제안합니다. 이 방법은 기존의 두 단계 훈련 방식을 기반으로 하여, base class와 novel class 간의 분포 차이를 최소화하기 위해 효과적인 지식 전달을 활용합니다. 특히, novel 클래스의 프로토타입을 base 클래스 프로토타입과의 상관관계를 이용해 조정하는 새로운 프로토타입 조정 모듈을 도입했습니다. 또한, novel 분류기의 가중치 분포를 보정하기 위한 분류기 보정 모듈도 포함되어 있습니다.

- **Technical Details**: GFSS-EKT는 두 단계 훈련 체계를 이용하여 base class의 샘플에서 특성을 추출하고, 이를 각 클래스의 learnable prototypes에 투영하여 feature decomposition을 수행합니다. fine-tuning 단계에서 novel 클래스의 샘플에 대해서도 같은 방식으로 특성을 분해하고, base 클래스와 novel 클래스 모두를 분류하기 위해 novel 분류기를 학습합니다. 이러한 방법은 base 클래스에 대한 성능 저하 없이 novel 클래스의 성능을 향상시키기 위한 세 가지 주요 기여를 포함합니다. 즉, novel prototype modulation, novel classifier calibration, 그리고 context consistency learning을 사용하여 지식을 효과적으로 전이합니다.

- **Performance Highlights**: PASCAL-5i 및 COCO-20i 데이터 세트에서 수행된 광범위한 실험 결과, GFSS-EKT는 기존 GFSS 방법들에 비해 현저한 성능 향상을 보여줍니다. 이는 novel 클래스에 대한 정보가 부족한 상황에서도 base 클래스의 컨텍스트 정보를 효과적으로 활용하여 더 나은 분할 성능을 달성했음을 의미합니다. 따라서 GFSS-EKT는 현재의 최첨단 기술에 비해 실질적인 개선 효과를 제공하는 것으로 입증되었습니다.



### Robustness-enhanced Myoelectric Control with GAN-based Open-set Recognition (https://arxiv.org/abs/2412.15819)
Comments:
          11 pages, 14 figures

- **What's New**: 이 논문에서는 Generative Adversarial Networks (GANs)를 기반으로 한 새로운 프레임워크를 제안하여, 기존의 마이오일렉트릭 제어 시스템의 신뢰성을 향상시킵니다. 특히, 미지의 동작을 효과적으로 처리할 수 있는 오픈셋 인식(open-set recognition)을 가능하게 함으로써 시스템의 안정성을 높입니다. 이 접근 방식은 미분류를 방지하여 시스템의 안정성을 유지하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 GAN 기반의 판별기(discriminator)를 통합하여, 알려지지 않은 동작을 식별하고 거부합니다. 실험은 공개 데이터셋과 자체 수집 데이터셋을 이용하여 수행되었으며, 알고리즘은 특히 알려진 동작에 대해 97.6%의 인식 정확도를 자랑합니다. 또한, 알려지지 않은 동작을 거부한 후 Active Error Rate (AER)가 23.6% 개선되었습니다.

- **Performance Highlights**: 이 방법은 높은 인식 정확도와 함께 연산 효율성을 자랑하여, 엣지 디바이스(edge devices)에서의 배포가 가능하다는 장점이 있습니다. 실제 응용에 적합하도록 설계되어, 의료 재활 및 인간 동작 인식과 같은 분야에서 유용하게 적용될 수 있습니다.



### Cross-Modal Few-Shot Learning with Second-Order Neural Ordinary Differential Equations (https://arxiv.org/abs/2412.15813)
- **What's New**: 이번 연구에서는 SONO라는 새로운 방법론을 소개합니다. SONO는 Second-Order Neural Ordinary Differential Equations(Second-Order NODEs)를 활용하여 cross-modal few-shot learning을 향상시키는 데 중점을 두고 있습니다. 이 방법은 제한된 학습 예제로 인한 overfitting 문제를 해결하기 위해 간단하면서도 효과적인 아키텍처를 채택했습니다.

- **Technical Details**: SONO는 Second-Order NODEs를 모델링하여 데이터 변환을 부드럽고 일반화된 방식으로 수행합니다. 이 모델은 더 넓은 범위의 함수를 근사할 수 있어 feature optimization에서 탁월한 성능을 발휘합니다. 특히, 교차 모달(classifier) 분류기를 클래스 관련 프롬프트에서 파생된 텍스트 임베딩으로 초기화하여 훈련 효율성을 크게 개선합니다.

- **Performance Highlights**: SONO는 다양한 데이터셋에 대한 실험을 통해 기존의 최첨단 방법들을 능가하는 few-shot classification 성능을 입증했습니다. 이 검색 방법 (e.g., data augmentation)의 효과는 CLIP의 강력한 이미지-텍스트 상관관계를 활용하여 훈련 데이터를 풍부하게 만드는 데 크게 기여합니다. 결과적으로, SONO는 few-shot learning의 성능과 범위에서 개선된 성능 향회를 보여주었습니다.



### Diffusion-Based Conditional Image Editing through Optimized Inference with Guidanc (https://arxiv.org/abs/2412.15798)
Comments:
          WACV 2025

- **What's New**: 이번 논문에서는 사전 훈련된 텍스트-이미지 확산 모델을 기반으로 하여 텍스트 중심의 이미지 전환을 위한 간단하지만 효과적인 훈련 없는 접근 방식을 제안합니다. 이 방법은 소스 이미지의 구조와 배경을 보존하면서 목표 작업에 맞는 이미지를 생성하는 것을 목표로 하고 있습니다.

- **Technical Details**: 우리의 접근 방식은 CLIP 점수에 기반하여 목표 프롬프트에 대한 유사성을 최대화하고 소스 잠재 변수에 대한 구조적 거리를 최소화하는 두 가지 목표의 조합을 통해 표현 유도(Representation Guidance)를 파생합니다. 이러한 유도는 주어진 목표 프롬프트에 대한 생성된 이미지를 보다 충실하게 개선하며, 소스 이미지의 구조적 무결성을 유지합니다. 본 연구는 확산 모델의 역 과정(target latent variable)을 최적화하여 이 유도 요소를 통합하고 있습니다.

- **Performance Highlights**: 실험 결과, 우리는 사전 훈련된 Stable Diffusion 모델과 결합할 경우 다양한 작업에서 뛰어난 이미지-이미지 전환 성능을 달성했음을 보여줍니다. 이러한 접근은 이미지 생성의 정확성을 높이고 다양한 조건에서의 대응력을 향상시킵니다.



### Sparse Point Clouds Assisted Learned Image Compression (https://arxiv.org/abs/2412.15752)
Comments:
          Accepted by TCSVT

- **What's New**: 이 논문에서는 자율주행 시나리오에서 학습된 이미지 압축을 지원하기 위해 희소한 LiDAR(point cloud) 데이터를 활용하는 새로운 프레임워크를 제안합니다. 제안된 접근 방식은 3D 희소 포인트 클라우드를 2D 평면에 투영하여 희소한 깊이 맵을 생성하고, 이 깊이 맵을 이용해 카메라 이미지를 예측합니다. 이 예측된 이미지로부터 다중 스케일의 구조적 특징을 추출하여 이미지 압축 성능을 향상시킵니다.

- **Technical Details**: 이 연구는 Point-to-image Prediction (PIP) 및 Multi-scale Context Mining (MCM)이라는 두 가지 모듈을 설계하여 희소 포인트 클라우드에서 밀집한 구조적 정보를 추출합니다. PIP는 포인트 클라우드 정보를 기반으로 이미지를 예측하고, MCM은 이러한 예측된 이미지로부터 다중 스케일의 특징을 추출합니다. 이 특징들은 기존의 학습된 이미지 압축 프레임워크에 통합되어 압축 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방법은 여러 학습된 압축 네트워크를 통해 압축 성능을 일관되게 향상시킨다는 사실이 입증되었습니다. 이 연구는 희소한 LiDAR 포인트 클라우드를 이미지 압축에 지원하는 최초의 방법으로, 기존의 전통적인 이미지 압축 기법에 비해 성능상의 개선을 보여줍니다. 시각화 결과 또한 희소 포인트 클라우드로부터 더 밀집된 구조적 정보를 추출할 수 있음을 나타냅니다.



### VORD: Visual Ordinal Calibration for Mitigating Object Hallucinations in Large Vision-Language Models (https://arxiv.org/abs/2412.15739)
- **What's New**: 본 논문에서는 최신 Large Vision-Language Models (LVLMs)의 발전에도 불구하고, 모델이 생성하는 내용이 때때로 부정확하거나 일관성이 없는 경향이 있다는 점을 강조합니다. 이러한 환각(hallucinations) 현상이 임상, 자율주행, 법률 AI 등의 위험이 큰 응용 분야에서 우려되는 점을 지적하며, 이를 해결하기 위한 새로운 방법인 VORD를 제안합니다. VORD는 수정된 이미지 쌍간의 순서 관계를 기반으로 토큰 예측을 조정하여 환각을 완화하는 간단하고 효과적인 접근 방식을 제공합니다.

- **Technical Details**: VORD는 두 가지 형태로 제시됩니다. 첫 번째는 훈련이 필요 없는 미니멀한 변형으로, 수정된 이미지 쌍에서 implausible한 토큰을 제거하는 방법입니다. 두 번째는 가능성이 낮은 토큰에 패널티를 부여하는 훈련 가능한 목표 함수입니다. 실험을 통해 VORD가 LVLM 벤치마크에서 객체 환각을 효과적으로 완화하며, 개선된 예측 조정과 더 짧은 생성 텍스트 시퀀스를 제공함을 보여줍니다.

- **Performance Highlights**: VORD의 구현을 통해 LVLM이 시각적으로 손상된 입력에 직면했을 때, 토큰의 확률이 순서 관계에 부합하지 않는 현상을 발견하였습니다. 이 논문에서는 VORD가 LVLM의 객체 환각을 완화하고 예측의 조정을 개선하는 데 효과적임을 입증했습니다. 또한, VORD Decoding과 VORD Loss와 같은 새로운 방법론이 LVLM의 성능을 향상시키는 것을 보여주고 있습니다.



### The Role of Recurrency in Image Segmentation for Noisy and Limited Sample Settings (https://arxiv.org/abs/2412.15734)
Comments:
          24 pages

- **What's New**: 이 논문은 머신러닝의 진보에 기여한 생물학적 뇌의 구조에서 영감을 받아, 현재 최첨단 컴퓨터 비전 모델들이 인간의 뇌와 유사하게 작동하지 않는 이유를 분석하고 있습니다. 특히, 이 연구는 기존의 feed-forward segmentation 모델에 recurrent (순환) 메커니즘을 추가하는 것이 어떤 영향을 미칠지를 탐구하고 있습니다. 이는 뇌의 재귀적 특성이 현재의 모델에 어떤 긍정적인 변화를 가져올지 질문을 던지는 작업입니다.

- **Technical Details**: 연구에서는 self-organizing (자기 조직화), relational (관계성), memory retrieval (기억 검색)과 같은 여러 종류의 재귀성을 탐색하여 특정 에너지 함수를 최소화하는 방식을 적용했습니다. 실험은 인공 및 의료 이미징 데이터에 대해 수행하였으며, 특히 높은 수준의 노이즈와 few-shot learning (극소 데이터 학습) 설정에서의 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과, 재귀적 모델이 기존의 state-of-the-art feed-forward 모델보다 더 나은 성능을 보여주지 못했으며, 이는 기존의 재귀적 구조만으로는 성능 향상에 충분치 않다는 것을 시사합니다. 추가적인 연구가 필요하다는 결론에 도달했으며, 이는 향후 머신러닝 모델에 있어 재귀적 메커니즘의 적용에 대한 더 깊은 탐구를 불러일으킬 것입니다.



### Exploiting Multimodal Spatial-temporal Patterns for Video Object Tracking (https://arxiv.org/abs/2412.15691)
- **What's New**: 본 논문에서는 STTrack이라는 새로운 멀티모달 공간-시간 추적 접근 방식을 제안합니다. 기존의 멀티모달 트래커들이 공간적 특성의 융합 및 향상에 주로 초점을 맞추었던 반면, STTrack은 멀티모달 비디오에서 시간적 상관관계를 명확히 활용하여 동적인 표적의 변경 및 움직임 정보를 포착합니다. 특히, 시간 상태 생성기(Temporal State Generator, TSG)를 도입하여 멀티모달 시간 정보를 포함한 토큰 시퀀스를 생성합니다.

- **Technical Details**: STTrack은 공중(temporal) 모듈과 공간(spatial) 모듈을 통해 멀티모달 정보의 효과적인 상호작용을 강화합니다. TSG는 교차 마밤바 아키텍처 기반으로 다양한 모달리티의 대상 표현 특징과 이전의 멀티모달 시간 정보를 결합하여 현재 시간 단계의 멀티모달 시간 정보를 생성합니다. 또한, 배경 억제 상호작용 모듈(Background Suppression Interactive Module, BSI)과 마밤바 융합 모듈(Mamba Fusion Module)을 통해 각 모달리티의 표현을 향상시키고, 정확한 객체 로컬리제이션을 지원합니다.

- **Performance Highlights**: STTrack은 RGBT234, LasHeR, VisEvEnt, Depthtrack 및 VOT-RGBD2022와 같은 5개의 인기 있는 멀티모달 추적 벤치마크에서 최첨단 성능을 달성했습니다. 다양한 복잡한 시나리오에서 멀티모달 트래킹 성능을 극대화하며, 기존 접근 방식들이 놓친 시간적 정보의 연속성을 보완하는 데 성공했습니다. 이를 통해 STTrack은 멀티모달 비주얼 태스크에서 포괄적인 응용의 가능성을 보여줍니다.



### DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization (https://arxiv.org/abs/2412.15689)
- **What's New**: 이 논문에서는 비디오 생성(video generation)에서의 샘플링 단계 수를 줄이기 위한 새로운 증류(distillation) 방법을 제안합니다. 기존의 방법들이 비디오 품질이나 생성 다양성에 손해를 끼치곤 했던 반면, 본 연구는 변분적 점수 증류(variational score distillation)와 일관성 증류(consistency distillation)를 결합하여 고품질의 비디오 생성을 가능하게 합니다. 또한, 특정 보상 메트릭(reward metric)에 따라 비디오 생성 성능을 향상시킬 수 있는 잠재적 보상 모델(latent reward model) 미세 조정 방법도 소개했습니다.

- **Technical Details**: 제안된 방법은 10초짜리 비디오에 대해 4단계의 샘플링으로 최첨단 성능을 보여줍니다. 이 과정에서 128 프레임에서 12 FPS로 생성되며, 증류된 학생 모델(distilled student model)은 VBench 테스트에서 82.57의 점수를 기록하여 교사 모델(teacher model)과 제일자리 모델들(Gen-3, T2V-Turbo, Kling)을 초월했습니다. 한 단계 증류(one-step distillation)은 교사 모델의 확산 샘플링(diffusion sampling)을 최대 278.6배 가속화하여 거의 실시간 생성이 가능하게 합니다.

- **Performance Highlights**: 인간 평가(human evaluation)를 통한 추가 검증 결과, 50단계 DDIM 샘플링을 사용한 교사 모델에 비해 4단계 학생 모델의 성능이 우수하다는 결과가 나왔습니다. 이는 비디오 생성 분야에서 효율성과 품질을 모두 확보할 수 있음을 보여줍니다. 따라서 이 새 방법론은 비디오 생성의 적시성(timeliness) 및 실행 가능성을 확실히 개선하는 잠재력을 가지고 있습니다.



### Multi-Pair Temporal Sentence Grounding via Multi-Thread Knowledge Transfer Network (https://arxiv.org/abs/2412.15678)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 Temporal Sentence Grounding (TSG) 문제를 해결하기 위해 새로운 설정인 Multi-Pair TSG(MP-TSG)를 제안합니다. 이 연구는 기존의 TS이 방법들이 서로 다른 비디오-쿼리 쌍의 관계를 무시하는 문제를 해결하고자 하며, 여러 비디오-쿼리 쌍을 동시에 훈련할 수 있는 방법을 모색합니다. 이렇게 하여 다양한 쌍 간의 지식을 전이하여 전반적인 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 연구진은 여러 비디오-쿼리 쌍 간의 시공간(temporal) 및 공간적(spatial) 의미를 식별하여 이들 간의 관계를 탐구합니다. 특히, Cross-Modal Comparison Module을 사용하여 비디오와 쿼리 간의 의미적 일관성을 자가 지도(self-supervised) 방식으로 탐색하며, Adaptive Negative Selection Module을 통해 동적 매칭 임계치를 생성하여 효율적인 크로스 모달(cross-modal) 매칭을 구현합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존 기법에 비해 효과성과 효율성을 모두 향상시킨다는 것을 보여줍니다. 이 방법은 또한 최신 기법에 플러그 앤 플레이(plug-and-play) 모듈로 추가되어 기존 시스템의 성능을 증대시킬 수 있는 가능성을 지니고 있습니다. 이론적 기반과 실증적 증거를 통해, 다쌍 비디오-쿼리를 공동 훈련함으로써 얻는 이점이 뚜렷하게 나타났습니다.



### AI-generated Image Quality Assessment in Visual Communication (https://arxiv.org/abs/2412.15677)
Comments:
          AAAI-2025; Project page: this https URL

- **What's New**: 이 논문은 AIGI-VC라는 새로운 데이터세트를 제안하여 인공지능이 생성한 이미지(AIGIs)의 품질을 평가하기 위한 기준을 설정합니다. 기존 IQA 기법의 한계점을 극복하기 위해 광고 분야에서 AIGIs의 의사소통 능력을 정보의 명확성과 감정적 상호작용이라는 두 관점에서 분석합니다. AIGI-VC 데이터세트는 14개의 광고 주제와 8가지 감정 유형을 포함한 2,500장의 이미지를 제공하여 향상된 인간 선호도 주석을 수집합니다.

- **Technical Details**: AIGI-VC 데이터세트는 주관적 실험을 통해 정보의 명확성과 감정적 상호작용의 두 가지 평가 차원에 대해 이미지 쌍을 비교한 후, 일반적인 선호와 세부적인 선호 설명을 수집합니다. 이 데이터세트는 다양한 IQA 메트릭의 성능을 벤치마크하고, 기존의 다중 모드 모델(LMM)을 활용하여 AIGIs 평가의 효과성을 검증합니다. 특히, AIGIs의 인간-대상 상호작용, 환상적 콘텐츠, 긍정적/부정적 감정을 유도하는 세 가지 하위 집합에 대한 실험을 진행합니다.

- **Performance Highlights**: 이 연구에서는 최신 IQA 기법과 LMM들이 AIGIs의 검토에 있어 한계가 있음을 발견했습니다. AIGI-VC는 이러한 기법들이 효율적으로 작동하지 않음을 드러내며, AIGIs에 대한 실질적 평가의 필요성을 강조합니다. 연구 팀은 AIGI-VC의 기여를 통해 시각적 커뮤니케이션 응용 프로그램에서의 인공지능 이미지의 효과성을 높이고자 합니다.



### PersonaMagic: Stage-Regulated High-Fidelity Face Customization with Tandem Equilibrium (https://arxiv.org/abs/2412.15674)
Comments:
          This paper is accepted by AAAI 2025. The code is available at this https URL

- **What's New**: 이번 연구에서는 개인화된 이미지 생성을 위한 새로운 접근법인 PersonaMagic을 소개합니다. PersonaMagic은 얼굴 맞춤화에 특화된 단계 조절 생성 기술로, 고충실도(High-Fidelity) 이미지 생성을 지원합니다. 이 기법은 이전의 방법들과 비교해 새롭고 혁신적인 방식으로 단계 파티셔닝을 활용하여 개념을 도입하는 방식을 강조합니다.

- **Technical Details**: 우리의 메소드에서는 기본 MLP(Multi-Layer Perceptron) 네트워크를 사용하여 특정 시간 간격(timestep interval) 내에서 일련의 임베딩(Embeddings)을 학습합니다. 특히, Tandem Equilibrium 메커니즘을 통해 텍스트 인코더의 자기 주의(Self-Attention) 응답을 조정하여 텍스트 설명과 정체성 보존(Identity Preservation) 간의 균형을 맞춥니다. 이러한 기술적 접근은 얼굴 특징의 복잡한 뉘앙스를 다루는 데 필수적입니다.

- **Performance Highlights**: PersonaMagic은 질적(Qualitative) 및 정량적(Quantitative) 평가 모두에서 최신 기법보다 우수성을 입증하였습니다. 또한, 비얼굴 도메인에서도 강력성과 유연성을 입증하여 다양한 분야에서 활용 가능성을 보여줍니다. 이는 사전 훈련된 개인화 모델의 성능을 향상시키는 중요한 플러그-인(Plug-in)으로 기능할 수 있습니다.



### Learning Group Interactions and Semantic Intentions for Multi-Object Trajectory Prediction (https://arxiv.org/abs/2412.15673)
- **What's New**: 본 논문에서는 그룹 상호작용(group interactions)과 동적 의미 의도(dynamic semantic intentions)를 효과적으로 모델링하기 위한 새로운 확산 기반 경로 예측 프레임워크를 제안합니다. 이 프레임워크는 팀 전략과 상대 팀의 행동이 포함된 상태에서 다양한 경로를 생성할 수 있도록 조건부 확산 모델에 그룹 수준의 상호작용을 통합합니다. 또한, NBA SportVU 데이터셋을 확대하여 팀 수준의 전술에 대한 인적 주석을 추가하여 예측 작업의 정확도를 높입니다.

- **Technical Details**: 경로 예측을 위해, 본 연구에서는 협력 게임(cooperative game) 프레임워크를 적용하여 그룹 상호작용 예측을 모델링합니다. 이 과정에서 Banzhaf 상호작용(Banzhaf interaction)을 활용하여 협력 경향을 반영하며, global 및 local 집합을 통해 강화된 에이전트 임베딩(agent embeddings)을 결합합니다. 이를 통해, 팀의 동적인 전술 변화에 따른 의미 의도를 캡처함으로써, 에이전트의 행동을 보다 정확하고 현실적으로 예측할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안한 모델이 여러 인기 있는 데이터셋에서 기존의 최고 성능(state-of-the-art) 방법보다 우수한 성능을 나타냄을 확인하였습니다. 특히, 스포츠와 같은 복잡한 시나리오에서 그룹 상호작용을 정확하게 모델링함으로써 예측의 신뢰성을 높였습니다. 이러한 연구 결과는 스포츠 분석 시스템, 자율주행 시스템, 사회적 로봇과 같은 다양한 실제 애플리케이션에 기여할 수 있습니다.



### Adaptive Hierarchical Graph Cut for Multi-granularity Out-of-distribution Detection (https://arxiv.org/abs/2412.15668)
- **What's New**: 이 논문에서는 out-of-distribution (OOD) detection의 새로운 접근 방식을 제안합니다. 기존 방법들이 모든 비표시 데이터를 OOD로 간주하는 데 반해, 이 연구는 다양한 레이블의 세분화가 존재하는 상황을 고려합니다. Adaptive Hierarchical Graph Cut network (AHGC)를 통해 서로 다른 이미지 간의 의미적 관계를 깊이 탐구하여, 레이블이 다르더라도 공유하는 의미를 잃지 않고 처리할 수 있습니다.

- **Technical Details**: AHGC는 계층적 KNN 그래프를 구성하여 이미지 간의 코사인 유사성을 계산합니다. 이 그래프를 여러 개의 서브그래프로 나누어, 높은 유사성을 가진 샘플들을 클러스터링합니다. 각 서브그래프의 라벨 비율이 특정 임계값을 초과하면, 가장 높은 비율의 라벨을 비표시 이미지에 할당합니다. 또한, 각 이미지를 두 개의 다른 증강 버전으로 변형하고 이들 간의 유사성을 극대화하여 모델의 일반화를 개선합니다.

- **Performance Highlights**: CIFAR-10과 CIFAR-100이라는 두 개의 도전적인 벤치마크에서 AHGC는 기존의 최신 OOD 탐지 기법들보다 각각 81.24%와 40.47% 향상된 성능을 보였습니다. 이는 'FPR95' 기준으로 측정된 결과로, AHGC의 효과성을 뒷받침합니다. 이런 성과는 AHGC가 기존 방법의 한계를 극복하고 세분화된 레이블의 의미적 관계를 잘 처리했음을 의미합니다.



### SCENIC: Scene-aware Semantic Navigation with Instruction-guided Contro (https://arxiv.org/abs/2412.15664)
- **What's New**: 이번 연구에서는 SCENIC라는 새로운 확산 모델(diffusion model)을 소개합니다. 이 모델은 복잡한 지형을 갖는 가상 장면에서 자연스러운 인간의 동작을 생성하며, 텍스트를 통해 생성된 동작의 의미적 조절이 가능합니다. SCENIC은 복잡한 장면 기하학(complex scene geometry)을 동시에 처리하면서 고수준의 내비게이션 목표와 세밀한 환경 제약을 이해해야 하는 복잡한 기술적 도전 과제를 해결합니다.

- **Technical Details**: SCENIC은 계층적 장면 추론(hierarchical scene reasoning) 접근법을 사용합니다. 먼저, 목표 지향 좌표계(goal-centric canonicalization)를 통해 고수준의 목표 제약 조건을 처리하고, 사용자 중심 거리 필드(ego-centric distance field)를 통해 지역 기하학적 세부 사항을 캡처합니다. 이러한 이중 표현은 모델이 다양한 3D 장면에서 물리적으로 그럴듯한 동작을 생성할 수 있도록 합니다.

- **Performance Highlights**: SCENIC은 Replica, Matterport3D, HPS, LaserHuman과 같은 네 가지 실제 장면 데이터셋에서 뛰어난 성능을 보입니다. 이 모델은 '좀비처럼 계단을 올라가는(walking upstairs like a zombie)' 등 10개 이상의 다양한 동작 의미를 seamless 하게 전환할 수 있으며, 동작 품질과 장면 및 목표 제약 만족도에서 최상의 결과를 달성합니다. 실험 결과, SCENIC은 최신 기술에 비해 75.6%의 이해자들이 선호하는 결과를 보여줍니다.



### CustomTTT: Motion and Appearance Customized Video Generation via Test-Time Training (https://arxiv.org/abs/2412.15646)
Comments:
          Accepted in AAAI 2025

- **What's New**: 이번 연구에서 제안하는 CustomTTT는 텍스트 설명에 기반하여 주어진 비디오의 외관과 모션을 쉽게 조정할 수 있는 새로운 접근 방식을 제공합니다. 기존 방법들은 주로 단일 개념을 맞춤 설정하는 데 중점을 두었으며, 복수 개념의 통합에 있어 인식 오류가 있었습니다. CustomTTT는 LoRA를 특정 계층에 맞춰 결합하여 다양한 비디오 속성을 조정할 수 있도록 합니다.

- **Technical Details**: 이 방법은 다양한 주제와 모션의 맞춤화를 위해 사전 훈련된 Text-to-Video Diffusion 모델을 활용하며, LoRA 아답터를 통해 외관과 모션을 개별적으로 조정합니다. 이 과정에서 중요한 텍스트 프롬프트의 영향을 분석하고, 최적의 계층에 LoRA를 추가함으로써 맞춤화의 효과를 극대화합니다. 이후 두 개의 LoRA를 인터폴레이션한 후 새로운 test-time training(테스트 시 학습) 기법을 적용하여 결합된 결과물에서 나타날 수 있는 문제를 해결합니다.

- **Performance Highlights**: 제안된 CustomTTT 방법은 다양한 모션과 주제 참조에 기반한 실험을 실시하여 기존 최신 방법들보다 질적으로나 양적으로 우수한 결과를 보여주었습니다. 특히 텍스트-비디오 정렬(Text-Video Alignment)에서 더 나아진 성능을 입증하였으며, 다양한 사용자 맞춤형 비디오 생성의 가능성을 제시하고 있습니다. 이로 인해 다중 개념 맞춤화 설정에서도 시각적 품질이 크게 향상되었습니다.



### CrackUDA: Incremental Unsupervised Domain Adaptation for Improved Crack Segmentation in Civil Structures (https://arxiv.org/abs/2412.15637)
Comments:
          Accepted at ICPR 2024. Details and code can be accessed from this https URL

- **What's New**: 이번 연구에서는 기존의 크랙 세분화(crack segmentation) 알고리즘의 한계를 극복하기 위해 새로운 딥 네트워크를 제안했습니다. 이 네트워크는 비지도 도메인 적응(unsupervised domain adaptation, UDA)과 적대적 학습(adversarial learning)을 활용하여 소스 도메인(source domain)에서 정확도를 유지하면서 점진적 훈련(incremental training)을 수행합니다. 또한 새롭게 구축한BuildCrack 데이터셋은 잘 알려진 CrackSeg9K 데이터셋과 비교하여 이미지 수와 크랙 비율 측면에서 유사성을 갖추고 있습니다.

- **Technical Details**: 제안하는 모델은 인코더-디코더 아키텍처(encoder-decoder architecture)를 기반으로 하며, 도메인 불변(domain-invariant) 및 도메인 특화(domain-specific) 파라미터를 모두 포함합니다. 인코더는 모든 도메인에서 공유되는 크랙 특성을 학습하여 도메인 변동에 대한 강건성을 보장합니다. 동시에 디코더의 도메인 특화 파라미터는 각 도메인마다 고유한 특성을 포착하여 도메인 특정(feature) 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 다양한 CrackSeg9K의 하위 데이터셋(sub-dataset)과 우리의 커스텀 데이터셋을 사용하여 최신 UDA 방법들과 비교했을 때 크랙 세분화의 정확도가 크게 향상된 것으로 나타났습니다. 특히, 소스 도메인에서 0.65 및 타겟 도메인(target domain)에서 2.7 mIoU의 개선을 보이며, 이는 다른 UDA 방법들에 비해 뛰어난 일반화를 달성했음을 의미합니다.



### A New Method to Capturing Compositional Knowledge in Linguistic Spac (https://arxiv.org/abs/2412.15632)
- **What's New**: 이 논문은 Zero-Shot Compositional Understanding (ZS-CU)이라는 새로운 작업을 제안합니다. 이 작업은 고비용의 레이블된 트레이닝 데이터 없이도 조합 이해(compositional understanding)를 개선할 수 있도록 설계되었습니다. YUKINO라는 방법을 통해, 문맥적 역전(textual inversion)을 사용하여 이미지를 유사 토큰(pseudo-token)으로 매핑하는 새로운 접근법을 제공합니다.

- **Technical Details**: YUKINO는 사전 학습된 CLIP 모델을 기반으로 하며, 두 단계로 나뉜 훈련 과정을 통해 이미지를 유사 토큰으로 변환합니다. 첫 번째 단계에서는 "no" 논리적 캡션을 도입하여 하드 네거티브 샘플을 대체하고, 두 번째 단계에서는 지식 증류(knowledge distillation)를 통해 이 과정을 단일 모델로 최적화합니다. 이로 인해 다양한 이미지를 효과적으로 유사 토큰으로 매핑할 수 있습니다.

- **Performance Highlights**: YUKINO는 SugarCREPE 벤치마크에서 기존의 다중 모달(state-of-the-art, SOTA) 모델보다 8% 이상 우수한 성능을 보입니다. 또한, 이미지 검색 작업에서도 유의미한 성과를 이룰 수 있었으며, CLIP 모델이 조합성을 결여하고 있다는 문제를 효과적으로 해결했습니다.



### 3D Shape Tokenization (https://arxiv.org/abs/2412.15618)
- **What's New**: Shape Tokens는 기계 학습 모델에 통합하기 쉬운 3D 표현 방법으로, 연속적이며 컴팩트한 형태입니다. 이들은 3D 흐름 매칭 모델에서 형태 정보를 나타내는 조건 벡터로 작용하며, delta 함수에 해당하는 확률 밀도 함수를 근사화 하는 훈련을 통해 개발되었습니다. Shape Tokens를 적용함으로써 새로운 형태를 생성하고, 이미지를 3D로 변환하며, 3D 형태를 텍스트 및 이미지와 정렬할 수 있습니다.

- **Technical Details**: Shape Tokens(줄여서 ST)는 1,024개의 연속적인 벡터와 16차원으로 구성되어 있으며, 이는 기존의 메쉬나 수많은 포인트 대신 3D 형태를 효율적으로 표현할 수 있게 해 줍니다. ST는 물체 표면에서 독립적이고 동일하게 분포된(i.i.d.) 포인트를 샘플링할 수 있다는 점만을 가정하며, 이는 대부분의 3D 표현에서 필요한 물이 새지 않는 형태나 부피적 렌더링 가정에서 벗어난 특징입니다. 이 방식은 기존의 신경망 기반 3D 표현보다도 간단하고 유연한 훈련 프로세스를 가능하게 합니다.

- **Performance Highlights**: Shape Tokens를 적용한 여러 과제로부터 경쟁력 있는 성능을 입증하였으며, 3D 생성 문제에서는 ShapeNet에서 비조건부 흐름 매칭 모델을 학습하고, Objaverse에서 이미지-조건부 흐름 매칭 모델을 학습함으로써 성과를 거두었습니다. 또한, ST를 이미지 및 텍스트 CLIP 임베딩과 정렬시켜 3D 형태의 제로샷 텍스트 분류를 수행할 수 있음을 보여주었습니다. 더 나아가, 향후 연구의 발전을 위해 Shape Tokenizer, 이미지-조건부 잠재적 흐름 매칭 모델, 3D-CLIP 모델 등의 공개 릴리스를 예정하고 있습니다.



### Gaze Label Alignment: Alleviating Domain Shift for Gaze Estimation (https://arxiv.org/abs/2412.15601)
Comments:
          Camera Ready. Accepted to AAAI 2025

- **What's New**: 이번 논문에서는 gaze estimation 방법들이 다양한 도메인에서 평가될 때 성능 저하를 초래하는 label distribution shift에 주목합니다. 기존의 접근 방식들은 주로 data distribution 간의 편차를 줄이는 데 집중했으나, label의 차이도 심각한 영향을 미친다는 점을 강조합니다. 이는 gaze label acquisition 메커니즘과 개인의 생리적 차이에 의해 발생합니다. 이를 해결하기 위해 저자들은 gaze label alignment algorithm (GLA)을 제안하였습니다.

- **Technical Details**: GLA는 우선 모든 도메인에서 feature extractor를 훈련시켜 domain invariant features를 획득한 후, 하나의 도메인을 anchor로 선택하고 그 도메인에서 gaze regressor를 훈련합니다. 이후 나머지 도메인에서 gaze label을 예측하고, 이 예측된 label을 anchor 도메인의 ground truth와 조정하는 mapping function을 학습하여 label 나열을 정렬합니다. GLA는 모든 gaze estimation 방법과 결합하여 사용할 수 있는 장점이 있습니다.

- **Performance Highlights**: 실험 결과, GLA 방법이 label distribution shift 문제를 효과적으로 완화하고 기존 SOTA gaze estimation 방법의 성능을 개선하는 것을 보여주었습니다. 특히, gaze direction의 레이블링이 적절히 정렬됨으로써 다양한 도메인에서 모델의 일반화 능력이 개선됩니다. 최종적으로, GLA는 기존의 gaze estimation 모델들이 보다 나은 성능을 발휘할 수 있도록 도와줍니다.



### Mask-RadarNet: Enhancing Transformer With Spatial-Temporal Semantic Context for Radar Object Detection in Autonomous Driving (https://arxiv.org/abs/2412.15595)
- **What's New**: 본 논문에서는 Mask-RadarNet이라는 새로운 모델을 제안합니다. 이 모델은 레이더 데이터의 계층적 의미 기능을 최대한 활용하기 위해 설계되었습니다. 기존의 컨볼루션 신경망(CNN)에 의존하지 않고 공간-시간 의미 맥락을 효과적으로 캡처하는 데 중점을 두고 있습니다.

- **Technical Details**: Mask-RadarNet은 간섭적인 컨볼루션과 주의(attention) 연산을 결합하여 전통적인 트랜스포머 아키텍처를 대체합니다. 또한 패치 이동(patch shift) 방법을 도입하여 효율적인 공간-시간 기능 학습을 수행합니다. 이를 통해 높은 인식 정확도를 유지하면서도 계산 부담을 줄이는 성과를 보입니다.

- **Performance Highlights**: CRUW 데이터셋에서의 실험 결과는 Mask-RadarNet이 기존의 레이더 기반 객체 탐지 알고리즘에 비해 우수한 성능을 보여줍니다. 특히 이해가 어려운 RF 이미지에도 불구하고, 제안된 모델은 더 낮은 계산 복잡성과 적은 파라미터로 높은 인식 정확도를 달성합니다.



### SemDP: Semantic-level Differential Privacy Protection for Face Datasets (https://arxiv.org/abs/2412.15590)
- **What's New**: 이 논문은 전체 얼굴 데이터셋을 대상으로 하는 의미적 수준의 차별적 개인정보 보호(Differential Privacy, DP) 체계를 제안합니다. 기존의 DP 기술들은 보통 이미지를 개별 데이터베이스로 처리하여 개인정보 보호의 핵심 요건을 완전히 충족하지 못했지만, 이 연구에서는 얼굴의 의미적 개인 정보를 보호하는 동시에 데이터 유용성을 유지하는 방법을 모색합니다. 연구의 핵심 아이디어는 비정형 데이터를 정형 데이터로 변환하여 차별적 개인정보 보호를 적용하는 것입니다.

- **Technical Details**: 연구에서는 얼굴 속성 데이터베이스를 구축하고, 랜덤 반응 메커니즘을 통해 이 데이터베이스에 차별적 노출을 적용한 후, 생성적 적대 신경망(Generative Adversarial Network, GAN)을 사용하여 보호된 얼굴 이미지 데이터셋을 생성하는 세 가지 단계로 나뉘어 있습니다. 이 방식은 얼굴 이미지를 단순히 픽셀로 처리하는 것이 아니라 전체 데이터셋을 데이터베이스로 간주하여 의미적 개인 정보를 보장합니다. 차별적 개인 정보 보호의 정의를 만족하며, 개인의 민감한 정보 유출 위험을 최소화합니다.

- **Performance Highlights**: 엄청난 실험 결과를 통해 제안된 체계는 기존의 주류 방법들과 비교하여 시각적인 자연스러움을 유지하며 개인 정보 보호와 유용성 간의 균형을 잘 잡음을 보여줍니다. 기존의 픽셀 수준의 방법들에 비해 더 나은 보호를 제공하여 얼굴 데이터셋의 의미적 개인 정보가 유출되지 않도록 하였습니다. 이 연구는 개인의 민감한 정보를 안전하게 유지하면서도 데이터의 가치를 지속적으로 활용할 수 있는 가능성을 보여주고 있습니다.



### SaliencyI2PLoc: saliency-guided image-point cloud localization using contrastive learning (https://arxiv.org/abs/2412.15577)
Comments:
          Under Review

- **What's New**: 이 논문에서는 SaliencyI2PLoc라는 새로운 대조 학습 기반 아키텍처를 제안하고 있습니다. 이 방법은 이미지와 포인트 클라우드 간의 특징 일관성을 유지하면서 주목 맵(saliency map)을 특징 집합에 융합합니다. 이는 다중 매니폴드 공간에서의 특징 관계 일관성을 유지하며, 크로스 모달리티(feature mapping) 문제를 효율적으로 해결합니다. 이 연구는 다양한 데이터셋에서 실험을 통해 뛰어난 효과성을 입증했습니다.

- **Technical Details**: 제안된 SaliencyI2PLoc 구조는 Dual-Transformer 아키텍처에 기반하고 있으며, 대조 학습 프레임워크를 사용하여 이미지와 포인트 클라우드 간의 융합을 수행합니다. 이 모델은 2D 이미지와 3D 포인트 클라우드를 직접 처리하며, 자신 주의(self-attention) 기반 특징 추출기를 사용해서 각 모달리티에서 국소 패치 특징을 추출합니다. 또한, 컨텍스트 주의를 통해 강력한 전역 특징을 생성하는 것이 특징입니다.

- **Performance Highlights**: 우리는 도시 시나리오 평가 데이터셋에서 Recall@1이 78.92%, Recall@20이 97.59%에 도달하는 결과를 얻었습니다. 이는 기존 방법보다 각각 37.35% 및 18.07%의 향상을 보여주고 있습니다. 이 결과는 SaliencyI2PLoc 아키텍처가 이미지와 포인트 클라우드를 효과적으로 융합하고, 크로스 모달리티(global localization)에서 중요한 진전을 이룬 것을 시사합니다.



### J-EDI QA: Benchmark for deep-sea organism-specific multimodal LLM (https://arxiv.org/abs/2412.15574)
- **What's New**: 본 연구에서 저자들은 일본 해양 지구 과학 기술 기관(JAMSTEC)의 J-EDI를 소개하고, 깊은 바다 생물의 이미지를 이해하기 위한 벤치마크인 J-EDI QA를 제안합니다. 이 벤치마크는 100개의 이미지와 각 이미지에 대한 질문 및 답변을 포함하고 있습니다.

- **Technical Details**: J-EDI QA는 각 이미지에 대해 JAMSTEC 연구원이 제공한 네 가지 선택지가 있는 질문과 답변 쌍으로 구성되며, 일본어로 제공됩니다. 이 데이터셋은 주로 해양 생물의 이미지와 비디오로 구성되어 있으며, 저자는 멀티모달 대형 언어 모델(LLM)을 이용해 분석합니다.

- **Performance Highlights**: 평가 결과, OpenAI의 o1 모델은 50%의 올바른 응답률을 기록했습니다. 이는 최신 모델의 성능에도 불구하고 깊은 바다 생물의 이해력이 전문가 수준에 도달하지 못했음을 나타내며, 향후 깊은 바다 생물에 특화된 LLM의 발전이 필요하다는 점을 강조합니다.



### DefFiller: Mask-Conditioned Diffusion for Salient Steel Surface Defect Generation (https://arxiv.org/abs/2412.15570)
Comments:
          20 pages, 10 figures

- **What's New**: 이 논문에서는 산업 환경에서의 결함 감지를 위해 DefFiller라는 새로운 방법을 소개합니다. DefFiller는 마스크 조건을 활용하여 결함 샘플을 생성하는 기법으로, 픽셀 레벨의 주석 없이 데이터셋을 증가시킬 수 있습니다. 이는 기존의 데이터 증강 방법들이 필요한 리소스를 줄여 모델 훈련에 직접 사용될 수 있도록 합니다.

- **Technical Details**: DefFiller는 GLIGEN 모델을 기반으로 하며, 레이아웃에서 이미지로의 확산(diffusion) 모델을 활용하여 결함 생성을 구현합니다. 이 방법은 입력 마스크 조건을 통해 결함의 위치와 모양을 제어할 수 있어, 보다 정밀한 이미지 생성을 가능하게 합니다. 추가적으로, 생성된 샘플의 품질과 결함 탐지 성능에 미치는 영향 평가를 위한 평가 프레임워크도 개발하였습니다.

- **Performance Highlights**: 실험 결과, DefFiller는 SD-Saliency-900 데이터셋에서 제공된 마스크 조건과 정확하게 일치하는 고품질 결함 이미지를 생성하여 성능을 획기적으로 향상시키는 것을 보여줍니다. 이러한 결과는 생성된 데이터셋이 기존 모델의 성능을 크게 개선한다는 것을 시사합니다. DefFiller의 접근 방식은 결함 이미지 생성을 위한 기존 기법들에 비해 높은 품질과 강력한 제어 가능성을 제공합니다.



### EGSRAL: An Enhanced 3D Gaussian Splatting based Renderer with Automated Labeling for Large-Scale Driving Scen (https://arxiv.org/abs/2412.15550)
Comments:
          AAAI2025

- **What's New**: EGSRAL은 3D Gaussian Splatting (3D GS) 기반의 새로운 방법으로, 추가적인 주석 없이 훈련 이미지만을 사용하여 자동으로 주석을 생성하고 동적 객체와 정적 배경을 모델링하는 능력을 강화합니다. 본 방법은 입체적 이미지를 생성하면서도 이와 함께 해당하는 주석을 생성할 수 있도록 설계되었습니다. 또한, 대규모 복잡한 장면의 렌더링 시 원근법 문제를 해결하기 위한 그룹화 전략도 도입하였습니다.

- **Technical Details**: EGSRAL은 변형 강화 모듈과 불투명도 강화 모듈을 포함하여 3D GS의 모델링 능력을 향상시킵니다. 변형 강화 모듈은 가우시안 변형 필드를 정제하여 동적 객체와 정적 배경을 효과적으로 모델링할 수 있게 하며, 불투명도 강화 모듈은 신경망을 활용하여 복잡한 드라이빙 장면의 모델링 성능을 크게 향상시킵니다. 이 프레임워크는 구조에서 움직임(SfM) 기법을 통해 추정된 포인트 클라우드와 입력 이미지 시퀀스를 활용하여 현실적인 드라이빙 장면을 스스로 합성하고 주석을 자동으로 생성합니다.

- **Performance Highlights**: EGSRAL은 여러 데이터셋에서 최신 성능을 달성하며, nuScenes 데이터셋에서 PSNR 메트릭이 29.04에 도달하는 결과를 보여주었습니다. 이 방법은 2D/3D 탐지 작업의 성능을 크게 향상시키는 데 기여하며, 자동 주석 생성을 통해 수많은 하위 작업의 적용 가능성을 확장합니다. 이번 연구는 복잡한 자율주행 장면을 효과적으로 재구성할 수 있는 가능성을 제시합니다.



### ChangeDiff: A Multi-Temporal Change Detection Data Generator with Flexible Text Prompts via Diffusion Mod (https://arxiv.org/abs/2412.15541)
- **What's New**: 본 논문은 변화 감지(change detection, CD) 작업에서 사용되는 데이터 생성과 주목할만한 발전을 제시합니다. 특히, 다변량 분포 기반 텍스트 프롬프트(multi-class distribution-guided text prompt, MCDG-TP)를 활용하여 테이블 레이아웃(layout)과 이미지를 변환하는 과정을 혁신적으로 구현합니다. 이로 인해 기존의 한정된 데이터 세트에서 발생하는 문제점을 극복하고, 변화 감지 분야의 성능을 높일 수 있는 새로운 접근법인 ChangeDiff를 소개합니다.

- **Technical Details**: ChangeDiff는 두 단계로 구성된 데이터 생성 과정을 통해 텍스트 프롬프트와 텍스트-레이아웃(text-to-layout, T2L) 모델을 사용하여 연속적인 레이아웃을 생성하고, 이후 레이아웃-이미지(layout-to-image, L2I) 모델로 이러한 레이아웃을 이미지로 변환합니다. 이 과정에서 MCDG-TP를 활용하여 사용자가 지정한 클래스와 비율에 따라 레이아웃을 유연하게 생성할 수 있도록 합니다. 추가적으로, 클래스를 분포하는 정제 손실(class distribution refinement loss)을 설계하여 T2L 모델을 훈련시킵니다.

- **Performance Highlights**: ChangeDiff의 데이터 생성 결과는 시간적 연속성, 공간적 다양성 및 품질 현실감에서 상당한 발전을 보여줍니다. 실험 결과, 새롭게 생성된 데이터는 기존 변화 감지기(change detectors)의 성능을 향상시키고, 더 나은 전이 가능성(transferability)을 제공합니다. 이러한 변화는 CD 작업의 정확도를 높이고, 과거 데이터에 대한 의존성을 줄이는 데 기여할 수 있습니다.



### SGTC: Semantic-Guided Triplet Co-training for Sparsely Annotated Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2412.15526)
Comments:
          Accepted by AAAI 2025

- **What's New**: SGTC 프레임워크(Semantic-Guided Triplet Co-training)를 통해 시간과 비용을 절감하면서도, 의료 이미지를 단 세 개의 직교 단면을 통해 정확하게 분할할 수 있는 방법론을 제안합니다. 이 방법은 기존의 방법들이 주로 이미지 수준의 정보에 중점을 두는데 반해, 의미 있는(semiotic) 특성과 약한 경계를 인식하는 데 초점을 맞추고 있습니다. 또한, SGTC는 라디오로지스트(radiologist)들에게 부담을 덜어 주는 혁신적인 접근 방식을 제공하여 임상적 실용성을 제고합니다.

- **Technical Details**: SGTC 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 프리트레인(pretrained)된 CLIP를 기반으로 한 새로운 의미-유도 보조 학습 기법을 통해 세밀한 세분화가 가능해지며, 가짜 레이블(pseudo-label)의 품질을 향상시킵니다. 둘째, 세 개의 하위 네트워크(sub-network) 간의 협동 훈련(co-training)을 촉진하는 삼중 뷰 불일치 훈련(triple-view disparity training) 전략을 제안하여 빈약한 주석된 데이터로도 강력한 성능을 실현합니다.

- **Performance Highlights**: LA2018, KiTS19, LiTS의 세 가지 공개 의료 데이터셋에서 수행한 실험을 통해 SGTC는 기존의 세미-서포트 학습 방식보다 뛰어난 성능을 보여주었습니다. 이 연구는 삼중 뷰 방식의 데이터 주석이 어떻게 효과적으로 분할 결과를 개선할 수 있는지를 잘 증명하고 있습니다. SGTC는 의료 이미지의 세분화 정확성을 높이고 모델의 로버스트니스(robustness)를 극대화하여 임상 환경에서의 적용 가능성을 보여줍니다.



### InstructOCR: Instruction Boosting Scene Text Spotting (https://arxiv.org/abs/2412.15523)
Comments:
          Accepted by AAAI2025

- **What's New**: InstructOCR는 인스트럭션 기반의 장면 텍스트 스포팅 모델로, 사람의 언어 지시를 통해 이미지 내 텍스트의 이해를 높이는 혁신적인 접근을 제안합니다. 기존의 OCR 방법들이 이미지 인코더와 사전 훈련된 텍스트 정보를 주로 활용했던 반면, 본 연구는 인간 언어 지시를 모델에 통합하여 텍스트 스포팅 성능을 크게 향상시킵니다. 실험 결과, InstructOCR는 최신의 벤치마크에서 최첨단 성능을 달성하였고, VQA 태스크에도 효과적으로 적용 가능함을 입증했습니다.

- **Technical Details**: InstructOCR는 텍스트 속성에 기반하여 정교하게 설계된 지침을 사용하여 텍스트와 이미지 인코더를 훈련시킵니다. 이는 모델이 이미지 내 텍스트를 더욱 정확하고 유연하게 해석할 수 있도록 합니다. 이 프레임워크는 공공의 장면 텍스트 데이터 세트를 활용하여 비용 효율적으로 다양한 지침을 제조할 수 있으며, 이를 통해 훈련의 효율성을 높입니다.

- **Performance Highlights**: InstructOCR는 광범위한 실험을 통해 여러 장면 텍스트 스포팅 데이터 세트에서 뛰어난 성능을 보이며, 텍스트 VQA 데이터 세트에서 2.6%, ST-VQA 데이터 세트에서 2.1%의 성능 향상을 이루었습니다. 이는 인간 언어 지침을 통합함으로써 OCR 관련 과제에서 얻을 수 있는 이점을 잘 보여줍니다. 뿐만 아니라, 본 연구는 모델 해석 능력을 향상시키는 데도 중요한 기여를 하고 있습니다.



### Reconstruction of Contour Lines During the Digitization of Contour Maps to Build a Digital Elevation Mod (https://arxiv.org/abs/2412.15515)
- **What's New**: 이 논문에서는 디지털 고도 모델(Digital Elevation Model, DEM) 구축에서 중요한 등고선 맵의 효율적인 처리를 다룹니다. 등고선의 디지털화 과정에서 발생하는 단선(segment)의 재연결 과정을 설명하며, 이를 통해 DEM의 정확성을 높이고자 합니다. 특히, 단선의 끝점(endpoint)을 신속하고 정확하게 재연결할 수 있는 새로운 메커니즘을 소개합니다.

- **Technical Details**: 재연결 과정은 최소 유클리드 거리(minimum Euclidean distance)와 기울기 방향(gradient direction) 개념을 활용하여 끝점을 일치시키고, 이후에는 Cubic Hermite spline 보간(interpolation) 기법을 통해 매끄러운 곡선을 만들어냅니다. 이는 전체 표면 곡률(curvature)을 최소화하는 수학적 함수를 사용하여 이루어집니다. 이러한 방식으로 단선들이 효과적으로 재연결됩니다.

- **Performance Highlights**: 이 연구의 결과는 디지털 고도 모델의 품질을 개선하는 데 기여하며, 디지털화 과정에서의 오류를 감소시킬 수 있습니다. 최종적으로, 등고선 맵의 복원력을 높여 시장이나 연구에서 사용할 수 있는 고급 DEM을 구축하는 데 중요한 역할을 합니다. 이러한 접근 방식은 향후 다양한 지리정보 시스템(GIS) 프로젝트에 적용될 수 있습니다.



### PolySmart @ TRECVid 2024 Medical Video Question Answering (https://arxiv.org/abs/2412.15514)
- **What's New**: 이 논문에서는 TRECVid 2024에서 진행된 Medical Video Question Answering 작업의 제출 결과를 요약합니다. Video Corpus Visual Answer Localization (VCVAL)에서는 질문과 관련된 비디오를 검색하고, 비디오 내에서 시각적 답변을 로컬라이즈하는 기술을 도입했습니다. 특히 GPT-4를 기반으로 한 텍스트-투-텍스트 검색을 통해 의학적 질문에 적합한 비디오를 찾습니다.

- **Technical Details**: 의학적 비디오의 특성과 관련하여, 텍스트-투-비디오 검색보다 더 높은 정확도를 필요로 하는 VCVAL 작업에서 새로운 방법론이 소개되었습니다. 이 방법은 두 개의 단계로 나뉘며, 첫 단계는 텍스트-투-텍스트 검색을 통해 관련 비디오를 식별하고, 두 번째 단계에서는 Textual Predictor와 Visual Predictor를 사용하여 정밀한 구간 로컬라이징을 수행합니다. 이를 통해 비디오 콘텐츠의 시각적 및 텍스트적 요소를 보완하여 정확성을 높입니다.

- **Performance Highlights**: 제출된 실험에서 Run 1은 MAP = 0.1401을 달성하였고, Run 5는 BLIP-2 기능을 활용한 새로운 접근으로 MAP = 0.0466를 기록했습니다. QFISC 작업에서도 GPT-4를 이용해 단계별 캡션을 생성하여 F-score 11.92 및 평균 IoU 9.6527을 달성했습니다. 이러한 결과들은 의학적 비디오 질문 응답 시스템의 효율성을 입증하는 데 기여합니다.



### PolySmart @ TRECVid 2024 Video-To-Tex (https://arxiv.org/abs/2412.15509)
- **What's New**: 이번 논문에서는 2024 TRECVid에서 비디오-텍스트(Video-To-Text, VTT) 작업을 위한 방법과 결과를 제시하며, 자연어 설명 생성을 위한 비전-언어 모델(Vision-Language Models, VLM)인 LLaVA 및 LLaVA-NeXT-Video의 가능성을 탐구합니다. VLM을 VTT 데이터셋에 미세 조정(fine-tuning)함으로써 설명의 정확도, 맥락 적절성, 언어 일관성을 향상시키는 영향을 조사하였습니다. 결과적으로, 미세 조정된 모델이 다양한 평가 지표에서 기존 VLM보다 우수한 성능을 보이며, VTT 작업의 복잡성을 반영한 도메인 특화 조정의 중요성을 뒷받침합니다.

- **Technical Details**: VTT 작업은 영상 정보를 언어 처리와 통합하여 비디오 콘텐츠에 대한 간결하고 정확한 자연어 설명을 생성하는 복잡한 비전-언어 작업으로, 접근성, 콘텐츠 검색 및 인간-컴퓨터 상호 작용 분야에서 중요합니다. 본 연구에서는 LLaVA와 LLaVA-NeXT-Video 모델을 활용하여 VTT 작업을 위한 설명 생성을 수행하였으며, 각 모델의 장점을 극대화하기 위해 대규모 VTT 비디오-텍스트 페어를 바탕으로 미세 조정을 실시하였습니다. 로스(loss) 최적화, 비주얼 인식 및 텍스트 생성의 통합을 위해 MLP(block)를 사용하여 비주얼 임베딩(embeddings)을 LLM의 토큰 임베딩으로 변환하게 됩니다.

- **Performance Highlights**: 테이블 1의 평가 결과에서, 제안한 3가지 방법의 성능을 BLEU, METEOR, CIDEr 등의 다양한 지표로 측정하여 비교하였습니다. 일반적으로 LLaVA-NeXT-Video는 미세 조정된 LLaVA 및 Vanilla LLaVA에 비해 다양한 지표에서 낮은 점수를 보였으며, 이는 연속적인 데이터 처리에서 추가적인 훈련이 필요함을 시사합니다. 특히, 미세 조정된 LLaVA( LV-FT)는 거의 모든 지표에서 가장 높은 점수를 기록하였으며, 이는 미세 조정이 더 세밀하고 관련성 높은 콘텐츠 수집 능력을 향상시켰음을 나타냅니다.



### GCA-3D: Towards Generalized and Consistent Domain Adaptation of 3D Generators (https://arxiv.org/abs/2412.15491)
- **What's New**: GCA-3D는 3D 도메인 적응을 위한 새로운 방법으로, 복잡한 데이터 생성 파이프라인 없이도 일반화되고 일관된 3D 도메인 적응을 구현합니다. 이 방법은 텍스트 프롬프트와 원샷 이미지 프롬프트 모두를 지원하며, 효과적인 멀티 모달 깊이 인식 SDS 손실을 도입하여 비적대적 방식으로 3D 생성 모델을 조정할 수 있습니다. 또한, 이 방법은 원인적 구조에서 깊이 맵을 활용하여 과적합 문제를 완화하고 결과의 다양성을 유지합니다.

- **Technical Details**: GCA-3D는 멀티모달 깊이 인식 점수 증류 샘플링 손실 (depth-aware score distillation sampling loss)을 통해 3D 생성 모델을 비적대적 방식으로 조정합니다. 이 손실 함수는 소스 생성기로부터의 인스턴스 인식 깊이 맵을 활용하여 적응 과정에서 조건으로 사용됩니다. 또한, 계층적 공간 일관성 손실 (hierarchical spatial consistency loss)을 통해 생성된 이미지의 공간 구조를 소스 도메인과 정렬하여 포즈와 아이덴티티의 일관성을 보장합니다.

- **Performance Highlights**: GCA-3D는 효율성, 일반화, 포즈 정확성 및 아이덴티티 일관성에서 기존 방법보다 뛰어난 성능을 보여줍니다. 다양한 도메인에 대한 실험 결과, GCA-3D는 3D 생성적 도메인 적응에서 포즈 정확도와 다각성을 현저하게 개선한 것으로 나타났습니다. 이 연구는 텍스트 기반 적응과 원샷 이미지 지향 적응 모두에 대한 첫 번째 실험으로서, 3D 생성 기술의 큰 발전을 의미합니다.



### Toward Appearance-based Autonomous Landing Site Identification for Multirotor Drones in Unstructured Environments (https://arxiv.org/abs/2412.15486)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구에서는 비구조적 환경에서 자율적으로 착륙 가능한 위치를 식별하는 문제를 해결하기 위해, RGB 이미지로부터 안전 지역과 위험 지역을 분류하는 경량의 이미지 세그멘테이션(Classifier)을 제안합니다. 이 classifier는 이미지와 마스크 데이터 세트를 생성하는 데 드는 비용 문제를 피하기 위해 자동으로 합성된 데이터를 활용합니다. 또한, U-Net 모델을 훈련시켜 실제 데이터를 테스트하고 실시간 드론 플랫폼에서 시연합니다.

- **Technical Details**: 이 연구의 핵심은 환경을 비행 중에 분석하기 위해 LiDAR와 같은 복잡한 센서를 사용하지 않고, 일반적인 RGB 카메라를 사용하는 것입니다. 우리는 드론이 자동으로 지형을 조사할 수 있는 능력을 활용하여 합성된 데이터 세트를 생성하고, 이를 통해 기존의 수동 라벨링의 부담을 줄입니다. 최종적으로, 우리는 1MB 정도의 작은 U-Net 모델을 제안하여 전력 효율적인 하드웨어에서 실시간으로 동작할 수 있게 합니다.

- **Performance Highlights**: 제안된 시스템은 18개의 검증 사례 중 15개를 올바르게 분류하였으며, 이는 드론 플랫폼에서 실시간으로 실행되었습니다. 이 방식은 다양한 환경에서 운용 가능성을 높이며, 향후 드론의 자동 착륙을 완전히 구현할 수 있는 기초를 제공합니다. 특히, 다양한 데이터 소스에서 생성된 합성 데이터 세트를 통해 데이터 수집의 유연성을 확보한 점이 주목할 만합니다.



### Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverag (https://arxiv.org/abs/2412.15484)
- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 세부 캡션 생성에서 발생하는 환각(hallucination) 문제에 대해 분석했습니다. MLLMs가 생성한 캡션의 사실성을 향상시키기 위해 다중 에이전트 시스템(Caption factuality enhancing MultiAgent System, CapMAS)을 제안하며, 이를 통해 LLM과 MLLM의 협업으로 캡션을 수정하게 됩니다.

- **Technical Details**: CapMAS는 기존의 훈련 기반의 방법과 달리 추가적인 훈련 없이도 LLM과 MLLM의 협업을 통해 하이퍼 세부 캡션의 사실성을 개선합니다. 이 시스템은 주어진 세부 캡션을 원자적 명제로 분해하고, MLLM이 이미지를 기반으로 이 명제들이 진실한지를 검증하여 LLM이 캡션을 수정하는 과정을 포함합니다.

- **Performance Highlights**: 실험 결과, CapMAS는 기존 방법들보다 사실성 측면에서 인간의 판단과 더 잘 일치하며, 특히 GPT-4V가 생성한 캡션의 사실성에도 유의미한 향상을 가져옵니다. 이러한 개선은 모든 캡션 모델에 적용할 수 있으며, VQA 벤치마크와의 상관관계 문제도 지적하며, MLLM의 성능이 캡션 생성 능력과 일치하지 않을 수 있음을 보여줍니다.



### Difficulty-aware Balancing Margin Loss for Long-tailed Recognition (https://arxiv.org/abs/2412.15477)
- **What's New**: 이 논문은 클래스 불균형과 인스턴스 난이도를 동시에 고려하는 Difficulty-aware Balancing Margin (DBM) 손실 함수를 제안합니다. 기존 방법들이 클래스 수준의 불균형에 주로 집중한 반면, DBM 손실은 각 개별 샘플의 난이도 변화를 반영하여 학습의 편향을 줄이는데 도움을 줍니다. 두 가지 구성 요소로 이루어진 DBM 손실은 클래스별 마진(class-wise margin)과 인스턴스별 마진(instance-wise margin)을 포함하여, 더 어려운 샘플에 대해 더 큰 마진을 할당합니다.

- **Technical Details**: DBM 손실은 클래스 수준의 빈도 불균형을 완화하기 위한 클래스 별 마진과, 개별 샘플의 난이도에 따라 조정되는 인스턴스 별 마진으로 구성됩니다. 이 방법은 복잡한 샘플에 대해 추가적인 마진을 부여하여, 클래스 간 구별력을 높입니다. DBM 손실은 기존의 LTR(long-tailed recognition) 접근 방식과 매끄럽게 결합되며, 여러 벤치마크에서 일관된 성능 향상을 보여줍니다.

- **Performance Highlights**: CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT 및 iNaturalist2018과 같은 다양한 긴 꼬리 인식 벤치마크에서 성능을 개선하였습니다. DBM 손실은 기존의 LTR 기술과 호환 가능하며, 상당한 추가 계산 비용 없이 사용될 수 있습니다. 실험 결과, 제안된 방법은 주요 긴 꼬리 인식 벤치마크에서 경쟁력 있는 성능을 보여 품질이 검증되었습니다.



### LiHi-GS: LiDAR-Supervised Gaussian Splatting for Highway Driving Scene Reconstruction (https://arxiv.org/abs/2412.15447)
- **What's New**: 이 논문에서는 LiDAR(라이다) 감시와 LiDAR 렌더링 지원을 통해 3D 장면 재구성을 개선한 새로운 GS(가우시안 스플래팅) 방법인 LiHi-GS를 제안합니다. 기존 연구는 주로 저속의 도시 환경에서 제한되어 있었지만, 본 연구는 고속도로 시나리오에 초점을 맞추어 독창적인 접근을 시도하였습니다. LiHi-GS는 LiDAR 정보를 전체적으로 활용하여 이미지와 LiDAR 데이터의 합성을 개선하며, 이는 고속도로 상황에서의 안전성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 제안하는 방법은 동적 장면을 위한 3D 가우시안 표현과 LiDAR 가시성 비율을 도입하여 장면 설명과 렌더링을 동적으로 수행할 수 있게 합니다. 또한, LiDAR 센서를 모델링하는 새로운 프레임워크를 포함하며, 이는 3D 가우시안을 LiDAR 범위 이미지로 투사하는 차별화 가능한 렌더링 방법과 깊이 불확실성 렌더링 기술을 통합합니다. 이러한 접근 방식은 LiDAR 데이터의 활용도를 높이며, 고속도로와 같은 도전적인 환경에서 문제를 해결하는 데 중점을 둡니다.

- **Performance Highlights**: LiHi-GS는 이미지 및 LiDAR 합성 측면에서 현재의 최첨단(SOTA) 방법을 초월하였으며, 특히 장면 편집 및 뷰 인터폴레이션(Task)에서 두각을 나타냈습니다. 본 연구는 고속도로나 다른 도로 환경에서의 장면 재구성에 대한 최초의 포괄적인 연구로, LiDAR 감시의 중요성을 증명하고 실제 운전에서의 적응성을 향상시키는 데 큰 기여를 하고 있습니다.



### Efficient Neural Network Encoding for 3D Color Lookup Tables (https://arxiv.org/abs/2412.15438)
Comments:
          14 pages, 13 figures; extended version; to appear in AAAI 2025

- **What's New**: 이 연구에서는 수백 개의 3D 색상 룩업 테이블(3D LUTs)을 단일 컴팩트 표현으로 인코딩할 수 있는 신경망 아키텍처를 개발하였습니다. 제안된 모델은 0.25MB 이하의 메모리 사용량으로, 512개의 LUT를 재구성할 수 있으며, 평균 색상 차이(ΔE)가 2.0 이하로 허용되는 색상 왜곡을 유지합니다. 또한, 네트워크 구조에 약간의 수정으로 역 색상 처리가 가능한 쌍향 인코딩(bijective encoding) 기능도 구현했습니다.

- **Technical Details**: 제안된 접근방식에서는 RGB 입력 색상을 RGB 출력 색상으로 매핑하는 3D LUT의 수학적 정의를 제공합니다. 연구는 다양한 네트워크 크기 변형을 통해 512개의 LUT를 인코딩하는 모델을 설계하였습니다. 이 모델은 각 LUT를 2ms 이내에 복구할 수 있어 실시간 응용이 가능합니다. 또한, 입력 색상의 가중치를 조정하는 대체 손실 함수도 도입하여 자연 이미지 색상에 대한 품질 향상이 이루어졌습니다.

- **Performance Highlights**: 모델은 평균 색상 차이(ΔE)를 1.0 이하로 유지하면서 512개의 LUT를 효과적으로 복구했습니다. 제안된 신경망은 다양한 LUT를 암묵적으로 압축하고, 기존 방법보다 99% 이상의 압축률을 달성합니다. 마지막으로, 쌍향 인코딩에 대한 수정으로 LUT를 역으로 처리할 수 있어 컴퓨터 비전과 이미지 처리 분야에서 활용 가능성이 더욱 높아졌습니다.



### SolidGS: Consolidating Gaussian Surfel Splatting for Sparse-View Surface Reconstruction (https://arxiv.org/abs/2412.15400)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 SolidGS라는 새로운 방법을 제안하며, 이는 Gaussian splatting을 사용한 다중 뷰 이미지로부터 표면 재구성의 품질을 크게 개선합니다. 기존 방법들이 희소한 뷰 입력 이미지만으로는 높은 품질의 표면을 재구성하는 데 어려움을 겪고 있음을 지적하고, 이에 대한 해결책을 제시합니다. 이 연구는 뷰 간의 일관성을 높이기 위해 더 견고한 커널 함수(kernel function)를 채택하는 것을 중점으로 합니다.

- **Technical Details**: SolidGS는 기존의 Gaussian 함수의 특성으로 인해 다중 뷰에서의 재구성이 불일치해질 수 있음을 관찰하였습니다. 이를 해결하기 위해 더 견고한 커널 함수를 도입하고, 추가적으로 기하학적 정규화(geometrical regularization)와 단안 법선 추정(monocular normal estimation)을 도와줍니다. 이러한 방법론은 표면 재구성의 품질을 현저히 향상시키는 데 기여합니다.

- **Performance Highlights**: 우리는 SolidGS가 DTU, Tanks-and-Temples, LLFF와 같은 널리 사용되는 데이터셋에서 기존의 Gaussian splatting 방법들과 신경장(neural field) 방법들보다 우수한 성능을 보인다는 것을 강조합니다. 특히, 희소한 뷰에서의 표면 재구성 성능에서 뛰어난 결과를 나타냈습니다.



### Learning Visual Composition through Improved Semantic Guidanc (https://arxiv.org/abs/2412.15396)
- **What's New**: 이번 논문은 기존의 시각적 표현 학습 방식의 한계를 극복하기 위해, 간단하고 확장 가능한 접근 방식을 제안합니다. 기존의 CLIP 모델들이 인과(thought) 이해에 부족하다는 점을 지적하며, 비약적인 성능 향상을 위해 기존 캡션을 새로운 다중 모달 모델에 의해 개선한다고 설명합니다. 특히, 향상된 데이터 세트 기반 훈련을 통해 이미지 검색 작업에서 실질적인 성과를 내고 있음을 입증하고자 하였습니다.

- **Technical Details**: 제안된 방법론은 CLIP 아키텍처에 기반하여 두 가지 주요 수정 사항을 도입합니다. 첫 번째, 고품질의 영어 이미지-텍스트 쌍을 포함한 WebLI 데이터셋을 사용하여 캡션의 품질을 향상시켜 alt-text를 대체합니다. 둘째, 훈련된 다중 모달 모델의 텍스트 타워를 고성능의 텍스트 기반 기초 모델로 교체하여 시각적 임베딩 표현을 크게 개선합니다. 이러한 방법을 통해 94.5%의 명확성을 달성하며 판별 성능이 크게 향상되었습니다.

- **Performance Highlights**: 향상된 데이터 세트와 적절한 모델 수정으로, CLIP 모델의 이미지 검색 성능이 눈에 띄게 개선되었습니다. 특히, recall@1 메트릭에서 58.4%의 성능에서 94.5%로 향상되었습니다. 일반적인 captcha 값으로 ARO 평가 데이터를 통하여 비교적 단순한 다중모달 모델의 학습 성과가 나타나고, 새로운 벤치마크에 대한 필요성을 적시하며 실험 결과의 타당성을 강화합니다.



### Maximising Histopathology Segmentation using Minimal Labels via Self-Supervision (https://arxiv.org/abs/2412.15389)
Comments:
          35 pages, 10 figures, 3 Tables

- **What's New**: 이번 연구에서는 self-supervised learning (SSL)을 활용하여 라벨의 수를 95% 절감하면서도 UNet, MDS1, UDAGAN과 같은 segmentation 방법들의 성능을 유지할 수 있음을 보여줍니다. 특히, 본 연구는 기존의 라벨 종속 모델을 대체할 새로운 방법론인 HR-CS-CO를 제안합니다. 이러한 접근 방식은 단일 염색(single-stain) 및 다중 염색(multi-stain)에 대한 segmentation 개선을 목표로 합니다.

- **Technical Details**: 연구에서 경량화를 위한 self-supervised learning 방법으로 SimCLR, BYOL 및 새롭게 제안된 HR-CS-CO를 사용합니다. 이들은 각각 기계학습의 다양한 방식으로 시각적 표현을 학습하며, 각 모델은 30개의 라벨만으로도 상당한 성능을 발휘할 수 있습니다. HR-CS-CO는 stain-specific한 한계를 극복하며, 이는 다중 염색에 대한 segmentation 성능 개선으로 이어집니다.

- **Performance Highlights**: 본 연구에서는 SSL을 통해 모델의 성능을 최소한의 라벨만으로 유지할 수 있음을 입증했습니다. UNet, MDS1, UDAGAN의 모델이 각각 5.9%, 4.5%, 6.2%의 성능 감소를 보이며, 이는 fully supervised 모델과 비교했을 때 매우 유사한 결과입니다. 실험을 통해 약 800회 이상의 테스트가 이루어졌으며, 이는 다중의 GPU 시간을 활용하여 분석되었습니다.



### Uncertainty-Guided Cross Attention Ensemble Mean Teacher for Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2412.15380)
Comments:
          Accepted in WACV 2025

- **What's New**: 이 연구에서는 반지도 학습(semi-supervised learning)에서 최첨단 성능을 달성하기 위한 새로운 프레임워크인 Uncertainty-Guided Cross Attention Ensemble Mean Teacher (UG-CEMT)를 제안합니다. UG-CEMT는 비전 트랜스포머(Vision Transformers)에서 영감을 받아 Cross-attention Ensemble Mean Teacher(CEMT)와 불확실성 유도 일관성 정규화(uncertainty-guided consistency regularization)와 Sharpness-Aware Minimization(SAM)을 결합하여 성능을 향상시킵니다. 실험 결과, UG-CEMT는 기존의 Mean Teacher 및 Cross-pseudo Supervision 방법에 비해 분산(disparity), 도메인 일반화(domain generalization), 의료 이미지 분할 성능에서 유의미한 장점을 나타냅니다.

- **Technical Details**: UG-CEMT는 고신뢰 예측을 활용하여 공동 학습(co-training) 분할 서브 네트워크 간의 고차원 분산을 유지합니다. 이 프레임워크는 점진적인 학습 과정에서 자신감을 중시하며, 새로운 크로스-어텐션 메커니즘을 통해 학습을 동적으로 조정합니다. 이 과정에서 Monte Carlo dropout(MC Dropout)을 활용하여 불확실성을 기반으로 높은 신뢰 예측을 우선시하며, SAM 정규화는 모델의 일반화 능력을 제고합니다.

- **Performance Highlights**: UG-CEMT는 다중 센터 전립선 MRI 및 심장 MRI 데이터셋에서 특히 도전적인 객체 분할에서 최첨단 성능을 달성했습니다. 단 10%의 레이블이 있는 데이터만 사용하여 전 면제 방법과 유사한 성능에 도달하면서 불확실성 유도 데이터를 활용한 효율성을 입증했습니다. 연구 결과는 기존 방법에 비해 UG-CEMT의 탁월한 성능을 명확히 보여줍니다.



### Dataset Augmentation by Mixing Visual Concepts (https://arxiv.org/abs/2412.15358)
Comments:
          Accepted at WACV 2025 main conference

- **What's New**: 이 논문은 사전 훈련된 확산 모델을 미세 조정하여 데이터셋 증강(data augmentation) 방법을 제안합니다. 기존의 텍스트 조건을 이용한 이미지 생성 시, 실제 데이터와 생성된 이미지 간의 도메인 불일치가 발생하는 문제를 해결하고자 합니다. 이를 위해 우리는 실제 이미지와 새로운 텍스트 임베딩으로 확산 모델을 조건화하여 적응시키는 접근 방식을 소개합니다.

- **Technical Details**: 제안된 Mixing Visual Concepts (MVC) 방법은 이미지 캡션으로부터 새로운 텍스트 임베딩을 생성하는 고유한 프로세스를 포함합니다. MVC를 통해 우리는 다수의 이미지를 생성할 수 있으며, 이 이미지는 실제 데이터와 유사하면서도 다양성을 유지합니다. 이러한 구성을 통해 우리는 데이터셋 증강을 효과적으로 수행할 수 있는 가능성을 보여주었습니다.

- **Performance Highlights**: 제안된 데이터셋 증강 방법은 정량적 및 정성적 평가에서 기존의 최첨단 증강 기법들을 초월하며, 기준 분류 작업에서 뚜렷한 성과를 보였습니다. 이 연구는 데이터셋 내부에서 데이터 생성을 위한 강력한 방법을 제시하며, 정교한 이미지를 생성할 수 있는 새로운 방법론을 바탕으로 실제 데이터와의 일관성을 유지합니다.



### Exploring Machine Learning Engineering for Object Detection and Tracking by Unmanned Aerial Vehicle (UAV) (https://arxiv.org/abs/2412.15347)
Comments:
          Accepted at ICMLA '24

- **What's New**: 최근 기술 발전으로 인명 구조(SAR) 작업에 무인 aerial vehicles (UAV)를 활용할 수 있는 가능성이 증가하고 있습니다. 본 연구는 실내 환경에서 자율 드론 시스템을 개발하여 Roomba 진공청소기를 목표로 삼고, 이 기술의 SAR 응용 가능성을 보여줍니다. UAV는 기존의 인력 구조 방식에 비해 위험을 줄이면서 구조 작업을 보다 효율적으로 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 Automated Labeling and Detection of Objects for Tracking (ALDOT)라는 머신러닝 기반의 프레임워크를 제안하여, 비디오 데이터를 처리하고 이동하는 객체를 효율적으로 탐지하고 추적하는 기술을 개발했습니다. Roomba 진공청소기를 추적하기 위해 Parrot Mambo 드론을 활용하였으며, 고해상도 카메라를 통해 indoor 환경에서 비디오 데이터를 수집했습니다. YOLOv4 모델을 사용하여 실시간 객체 탐지 및 추적 작업을 수행하였고, 데이터셋의 정확성을 보장하기 위해 여러 단계를 거쳐 라벨링 작업이 진행되었습니다.

- **Performance Highlights**: 실험 결과, YOLOv4 모델을 적용하여 Roomba를 96%의 정확도로 탐지하며, 평균 손실(loss)은 0.1942로 확인되었습니다. 이러한 성과는 자율 드론이 다양한 환경에서 효과적으로 작동할 수 있음을 보여줍니다. 또한, 본 연구는 SAR 작전에서 UAV의 가능성을 탐구하며, 향후 구조 작업에 대한 실용적인 기술 개발로 이어질 방향성을 제시합니다.



### Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis (https://arxiv.org/abs/2412.15322)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 MMAudio라는 새로운 다중 모달 조인트 트레이닝 프레임워크를 통해 비디오 및 선택적 텍스트 조건을 바탕으로 고품질 오디오를 합성하는 방법을 제안합니다. MMAudio는 비디오 데이터에 국한되지 않고, 대규모 텍스트-오디오 데이터와 함께 학습하여 의미적으로 정렬된 고품질 오디오 샘플을 생성할 수 있습니다. 또한, 조건부 동기화 모듈을 통해 프레임 레벨에서 비디오와 오디오의 동기화를 향상시킵니다.

- **Technical Details**: MMAudio는 비디오, 오디오 및 텍스트를 단일 트랜스포머 네트워크에서 함께 고려하며, 학습 중 누락된 모달리티를 마스킹합니다. 이는 오디오-비주얼 및 오디오-텍스트 데이터셋을 사용하여 스크래치로 학습할 수 있게 해 주며, 자원 데이터에 대한 모델의 이해를 높입니다. 추가로, 우리는 고속의 비주얼 피쳐를 사용하여 정확한 프레임 수준 동기화를 이끌어내는 조건부 동기화 모듈을 도입했습니다.

- **Performance Highlights**: MMAudio는 오디오 품질, 의미적 정렬, 오디오-비주얼 동기화 측면에서 공공 모델 중 새로운 상태를 달성하며, 8초의 클립을 생성하는 데 1.23초의 짧은 추론 시간을 자랑합니다. 다중 모달 접근 방식은 텍스트-오디오 생성에서도 경쟁력 있는 성능을 보여주어, 조인트 트레이닝이 단일 모달리티 성능에 방해가 되지 않음을 보여줍니다. 최종적으로 MMAudio는 제시된 모델 사이즈와 샘플링 주파수에서 뛰어난 성능을 확보하였습니다.



### Next Patch Prediction for Autoregressive Visual Generation (https://arxiv.org/abs/2412.15321)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 Next Token Prediction (NTP) 패러다임을 재논의하여 Autoregressive 모델을 기반으로 하는 이미지 생성을 위한 새로운 Next Patch Prediction (NPP) 패러다임을 제안합니다. 이미지 토큰을 고밀도 정보가 포함된 패치 토큰으로 그룹화하고 집계하는 방식으로, 전체적인 모델의 성능을 개선했습니다.

- **Technical Details**: 주요 아이디어는 패치 토큰을 짧은 입력 시퀀스로 사용하여 Autoregressive 모델이 다음 패치를 예측하도록 훈련시키는 것입니다. 우리는 이미지 데이터의 자연적인 계층적 특성을 활용한 다중 스케일의 거칠고 미세한 패치 그룹화 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, 다양한 모델(100M-1.4B 파라미터)에서 NPP 패러다임을 적용하면 훈련 비용이 약 0.6배로 줄어드는 동시에, ImageNet 벤치마크에서 이미지 생성 품질은 최대 1.0 FID 점수로 향상됨을 보여주었습니다. 또한, 우리의 방법은 기존 Autoregressive 모델 아키텍처를 유지하면서 추가적인 학습 가능한 파라미터를 도입하지 않아 다양한 Autoregressive 모델에 유연하게 적용될 수 있습니다.



### Multi-concept Model Immunization through Differentiable Model Merging (https://arxiv.org/abs/2412.15320)
Comments:
          AAAI 2025

- **What's New**: 본 논문에서는 모델 면역화(Model Immunization)의 새로운 접근법인 Multi-concept Immunization against Malicious Adaptation (MIMA)을 제안합니다. 이전 연구(Zheng and Yeh 2024)에서는 단일 개념에 대한 면역화에 초점을 두었으나, 실제 환경에서는 복수의 유해 개념에 대한 면역화가 필요합니다. MIMA는 여러 개념에 대한 적응 방법을 학습하는 '어려운 초기화'를 메타 학습하는 알고리즘입니다.

- **Technical Details**: MIMA는 다수의 개념 집합을 통합하는 차별 가능 모델 병합(layer) 기술을 포함하여 기법을 구현합니다. 또한, 다중 하위 작업을 포함하는 이계 최적화 방식(Bi-level Optimization)을 사용하여 학습한 면역 모델을 배포합니다. 이를 통해 MIMA는 모델이 여러 유해 개념에 대해 신뢰성을 가지도록 합니다.

- **Performance Highlights**: 실험 결과, MIMA는 여러 유해 개념에 대한 모델 면역화에 성공하여 이전 IMMA 기반 방법보다 우수한 성능을 보였습니다. 구체적으로, 두 가지 응용 프로그램(삭제된 개념 복원 및 개인화된 개념 학습)에서 여러 적응 방법을 평가하여 MIMA의 효과성을 입증했습니다.



### Deciphering the Underserved: Benchmarking LLM OCR for Low-Resource Scripts (https://arxiv.org/abs/2412.16119)
- **What's New**: 이번 연구는 특히 GPT-4o와 같은 대규모 언어 모델(LLM)의 잠재력을 탐구하며, 저자원 스크립트(우르두어, 알바니아어, 타지크어)에서의 광학 문자 인식(OCR) 성능을 분석합니다. 2,520개의 이미지를 포함한 데이터셋을 기반으로 할 때 다양한 실제 도전 과제를 모사했으며, LLM 기반 OCR의 한계를 강조하였습니다. 특히 복잡한 언어적 특성을 가진 스크립트에 대한 주석이 달린 데이터셋과 세밀하게 조정된 모델의 필요성을 나타내며, 접근 가능성 격차를 해소하기 위한 긴급한 과제를 부각시킵니다.

- **Technical Details**: 이 연구의 데이터셋은 우르두어, 영어, 알바니아어, 타지크어를 포함하여 각각의 언어에서 30개의 기사로 제한된 단어 수 범위를 기준으로 총 2,520개의 이미지를 생성했습니다. 텍스트는 다양한 글꼴 크기(12pt, 18pt, 24pt)와 배경 색상, 블러 등을 활용하여 각각의 언어적 특성과 시각적 조건의 변화를 반영했습니다. 그러한 방법을 통해, 복잡한 스크립트 내에서의 OCR 성능을 평가하고자 했습니다.

- **Performance Highlights**: 연구 결과는 언어적 복잡성이 큰 스크립트에서 zero-shot LLM 기반 OCR의 한계를 보여주었으며, 이는 주석 데이터셋의 필요성을 강조합니다. 이러한 성과는 기존의 OCR 모델에 비해 LLM이 특정 맥락에서 더 나은 성능을 발휘할 수 있다는 가능성을 부각시키며, 향후 저자원 스크립트의 디지털화와 접근성 개선을 위한 기초 자료를 제공합니다.



### Towards Interpretable Radiology Report Generation via Concept Bottlenecks using a Multi-Agentic RAG (https://arxiv.org/abs/2412.16086)
Comments:
          Accepted in ECIR 2025

- **What's New**: 이 연구는 Deep learning을 활용하여 Chest X-ray (CXR) 분류에서 해석 가능성(interpretability)을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히, concept bottleneck models (CBMs)와 multi-agent Retrieval-Augmented Generation (RAG) 시스템을 결합하여 임상적 관련성을 지닌 방사선 보고서를 생성합니다. 이러한 방법은 모델의 예측을 인간이 이해할 수 있는 방식으로 명확하게 제시하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 접근법은 두 단계를 통해 이루어집니다. 첫 번째 단계에서는 질병 분류와 관련된 개념 기여도를 계산하고, 두 번째 단계에서는 임상 문서와 설명을 활용하여 견고한 보고서를 생성합니다. 모델은 GPT-4를 사용하여 각 질병 범주에 대한 자동 개념 발견을 수행하고, 이미지 임베딩(ChexAgent 모델 사용)과 텍스트 임베딩(Mistral Embed Model 사용)을 결합하여 개념 벡터를 생성합니다.

- **Performance Highlights**: COVID-QU 데이터셋에서 본 모델은 81%의 분류 정확도를 기록하였으며, 생성된 보고서는 84%에서 90% 사이의 성능 메트릭을 보였습니다. 이는 AI 기반 CXR 분석의 신뢰성을 높이는 해석 가능성과 높은 성능 간의 갭을 메우는 다중 에이전트 프레임워크를 구축하는 데 기여합니다.



### Efficient MedSAMs: Segment Anything in Medical Images on Laptop (https://arxiv.org/abs/2412.16085)
Comments:
          CVPR 2024 MedSAM on Laptop Competition Summary: this https URL

- **What's New**: 이 논문은 의료 이미지 세분화를 위한 최초의 국제 대회를 소개합니다. 이 대회는 20개 이상의 기관에서 수집한 9가지 일반적인 이미징 모달리티를 아우르는 대규모 데이터셋을 특징으로 합니다. 연구팀들은 경량 세분화 모델을 개발하여 계산 요구사항을 대폭 줄이면서도 최신 기술의 세분화 정확도를 유지했습니다.

- **Technical Details**: 세분화(Segmentation)는 의료 이미지 분석에서 중요한 작업으로, 이를 통해 해부학적 구조와 병리학적 경계를 정확하게 정의할 수 있습니다. 최근에는 Segment Anything Model (SAM)과 같은 기초 모델들이 등장하여 다수의 도메인에서 일반화할 수 있는 능력을 보여주고 있습니다. 그러나 기존의 기반 모델들은 높은 계산 자원을 소모하기 때문에 임상 환경에서의 채택이 어렵습니다.

- **Performance Highlights**: 이 대회에서 참가자들은 고해상도 이미지 마스크 쌍으로 구성된 180만 개 이상의 교육 데이터를 활용하여 모델을 개발했습니다. 결과적으로, 참가자들은 10배 이상 빠른 세분화 결과를 달성하며, 상위 팀들의 알고리즘은 기존 SAM 기반 모델과 비교하여 효율성과 정확도를 모두 개선했습니다.



### Fair Distributed Machine Learning with Imbalanced Data as a Stackelberg Evolutionary Gam (https://arxiv.org/abs/2412.16079)
- **What's New**: 본 논문에서는 분산 학습 환경에서 데이터 불균형 문제를 해결하기 위해 Stackelberg 게임 이론을 적용하는 새로운 접근 방식을 제시합니다. 개발된 두 가지 알고리즘인 결정적 Stackelberg 가중치 모델(DSWM)과 적응형 Stackelberg 가중치 모델(ASWM)은 각 노드의 기여도를 설정하는데 사용됩니다. 이를 통해, 적은 데이터를 가진 노드의 성능을 개선하여 전체 모델의 품질을 높이려는 목표를 가지고 있습니다.

- **Technical Details**: 이번 연구에서는 세 개의 의료 데이터셋을 활용하여 ASWM이 적게 대표되는 노드의 성능을 2.713% 향상시키는 결과를 보였습니다. DSWM과 ASWM은 각각 결정론적 방법과 신경망을 통해 기여도를 예측하는 방법을 사용하며, 이를 통해 불균형한 데이터 분포 문제를 해결합니다. Stackelberg 게임 이론을 적용함으로써 자원 allocation과 참여 모델링이 더욱 효율적으로 이루어질 수 있습니다.

- **Performance Highlights**: ASWM이 대규모 데이터셋을 가진 노드에서는 평균적으로 0.441%의 성능 저하를 보이지만, 적게 대표되는 노드는 성능 개선을 경험할 수 있음을 보여줍니다. 이 연구의 결과는 분산 학습 환경에서 데이터 불균형 문제가 효과적으로 해결될 수 있는 가능성을 제시합니다. 이러한 동적 가중치 조정이 분산 학습의 정확도와 공정성을 더욱 높일 수 있음을 확인하였습니다.



### Image Quality Assessment: Enhancing Perceptual Exploration and Interpretation with Collaborative Feature Refinement and Hausdorff distanc (https://arxiv.org/abs/2412.15847)
- **What's New**: 이 연구에서는 색상과 밝기 왜곡이 주로 저주파에서 발생하고, 엣지와 텍스처 왜곡이 고주파에서 발생한다는 점을 고려하여, 훈련 없이도 이미지 품질을 정확하게 예측할 수 있는 새로운 FR-IQA 방법을 제안합니다. 제안된 방법은 감각 관련 도메인 변환과 분포 유사성 측정을 활용하여 인간 시각 시스템(HVS)과 일치하는 이미지 품질을 예측합니다. 이로 인해 기존의 방법들이 가지는 한계를 극복하고, 더 신뢰할 수 있는 이미지 품질 평가가 가능합니다.

- **Technical Details**: 제안된 모델은 다중 스케일 DWT(Discrete Wavelet Transform)를 사용하여 이미지 변형의 복잡한 특성을 효과적으로 캡처합니다. 또한, Hausdorff 거리 기반의 분포 유사성 측정 모듈을 통해 참조 이미지와 왜곡된 이미지 간의 특징 분포 차이를 전반적으로 평가하여 이상치와 변화를 효과적으로 관리할 수 있습니다. 이 방법은 훈련 데이터 없이도 인간의 인식 품질 차이를 정확하게 포착합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트에 대한 광범위한 실험을 통해 기존의 최신 기술들과 비교하여 우수한 성능을 입증하였습니다. 제안된 방법은 HVS와의 강한 상관관계를 보여주며, 이미지 품질 평가에서 더 나은 결과를 제공합니다. 고급 이미징 모델 및 실시간 이미지 처리 시스템과 결합할 수 있는 가능성을 보여줍니다.



### Precision ICU Resource Planning: A Multimodal Model for Brain Surgery Outcomes (https://arxiv.org/abs/2412.15818)
- **What's New**: 이번 연구에서는 수술 후 집중 치료실(ICU) 입원을 예측하기 위한 다중 모드(multimodal) 접근 방식을 도입하여 기존의 임상 데이터만을 사용하는 기법의 단점을 보완하였습니다. 기존의 예측 모델은 중요한 이미징(imaging) 데이터들을 간과했으나, 본 연구는 T1 MRI 이미지를 포함하여 예측 정확도를 개선했습니다. 결과적으로, 다중 모드 데이터 융합이 ICU 입원 예측에 유의미한 이점을 제공함을 확인했습니다.

- **Technical Details**: 연구는 다양한 아키텍처를 사용하여 예측 정확성을 향상시키기 위해 기능 추출(feature extraction) 및 분류(classification) 모델로 구분됩니다. 특히, 우리 연구에서는 자동 인코더(autoencoder)를 사용하여 이미지 데이터에서 중요한 특징을 추출하기 위한 latent representation을 생성하며, XGBoost 및 ResNet 모델을 활용하여 ICU 예측 성능을 비교했습니다. DAFT(Dynamic Affine Feature Map Transform) 모델을 통해 임상 데이터와 이미징 데이터를 더 효과적으로 융합하는 방법도 제안되었습니다.

- **Performance Highlights**: 우리의 다중 모드 접근 방식은 기존의 임상 데이터만 사용한 예측 모델보다 F1-score에서 0.29에서 0.30으로, 수술 전후 데이터를 사용할 경우 0.37에서 0.41로 향상된 성능을 보였습니다. 이 연구는 수술 후 ICU 입원을 예측하기 위한 다중 모드 데이터를 통합한 첫 시도로, 심각한 클래스 불균형(class imbalance) 상황에서도 우수한 결과를 도출했습니다. 이는 ICU 자원 할당의 효율성을 높이는 데 크게 기여할 수 있는 연구입니다.



### From Model Based to Learned Regularization in Medical Image Registration: A Comprehensive Review (https://arxiv.org/abs/2412.15740)
Comments:
          Submitted to Medical Image Analysis

- **What's New**: 이 논문은 의료 영상 등록(image registration)에서의 정규화(regularization) 기법에 대한 체계적이고 포괄적인 리뷰를 제공합니다. 다양한 정규화 방법을 분류하는 새로운 분류 체계를 도입하고, 데이터 기반(data-driven) 기술을 활용한 학습 정규화(learned regularization)의 중요성을 강조합니다. 또한, 기존의 정규화 기법이 어떻게 학습 기반 등록에 이식될 수 있는지를 분석하고 향후 연구 방향을 제시하고자 합니다.

- **Technical Details**: 의료 영상 등록의 주요 목표는 서로 다른 두 개 이상의 이미지를 정확하게 정렬하는 것으로, 일반적으로 최적화 문제를 최소화하는 방식으로 수행됩니다. 이 과정에서 정규화는 솔루션 공간을 제약하고 해석학적으로 의미 있는 변형(deformation) 속성을 통합하는 중요한 요소입니다. 저자들은 정규화 방법을 세 가지 주요 범주로 나누어 설명하고 있으며, 이는 (I) 모델 기반의 정규화, (II) 문제 특정 정규화, (III) 훈련 데이터를 사용하여 학습된 정규화로 구성됩니다.

- **Performance Highlights**: 이 리뷰는 정규화를 통해 이미지 등록 성능을 향상시키는 다양한 방법을 제시합니다. 또한, 정규화 기술이 기존의 의료 영상 등록에서 학습 기반 등록으로 얼마나 잘 이식될 수 있는지를 살펴봅니다. 저자들은 향후 연구자들이 정규화 기술을 재검토하고, 새로운 접근 방식을 개발하는 데 영감을 주기를 희망합니다.



### BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models (https://arxiv.org/abs/2412.15670)
Comments:
          9 pages, 6 figures

- **What's New**: 이 논문에서는 고해상도 흉부 X선 이미지(CXR)에서 뼈 억제를 위한 새로운 프레임워크인 BS-LDM(Bone Suppression Latent Diffusion Model)을 제안합니다. 기존의 Dual-Energy Subtraction(DES) 이미지 기법의 한계에 대응하고, 깊은 학습 기반의 이미지 생성 방식을 활용하여 상세하고 고해상도의 연조직 이미지를 생성합니다. ES-LDM은 조건부 잠재 확산 모델(conditional latent diffusion model)을 활용하여 고해상도 의료 이미지를 생성하여 정확한 병리 진단에 기여합니다.

- **Technical Details**: BS-LDM은 U-Net을 노이즈 추정 네트워크로 활용하며, CXR 잠재 변수를 채널 결합을 통해 조건적 입력으로 사용합니다. 추진적 처리 효율성을 높이기 위해 Vector Quantized GAN(VQGAN)을 통해 입력 데이터를 낮은 차원의 잠재 공간으로 변환합니다. 이는 높은 해상도의 이미지를 생성하는 데 소요되는 계산 요구 사항을 상당히 줄이고, 연조직 이미지의 세부 정보를 유지하면서 높은 뼈 억제 비율을 유지할 수 있도록 합니다.

- **Performance Highlights**: 유의미한 실험 및 임상 평가를 통해 BS-LDM은 뼈 억제 능력에서 우수한 성능을 보였으며, 임상적 활용 가능성이 강조되었습니다. SZCH-X-Rays라는 고품질 데이터셋을 구축하였고, 818명의 환자로부터 수집된 고해상도 CXR과 DES 연조직 이미지 쌍을 포함하고 있습니다. 이를 통해 본 연구는 폐 질환 진단의 정확도를 개선할 수 있는 잠재력을 가지고 있습니다.



### Technical Report for ICML 2024 TiFA Workshop MLLM Attack Challenge: Suffix Injection and Projected Gradient Descent Can Easily Fool An MLLM (https://arxiv.org/abs/2412.15614)
Comments:
          ICML TiFA Challenge Technical Report

- **What's New**: 이 기술 보고서는 TiFA 워크샵의 MLLM 공격 도전을 해결하기 위한 최고 순위의 솔루션을 소개합니다. 이 솔루션은 suffix injection과 projected gradient descent (PGD) 접근 방식을 사용하여 LLaVA 1.5 모델에 성공적으로 공격을 시도합니다. 수정된 쿼리를 사용하여 이미지에 자연스럽지 않은 섭동을 추가하는 방법이 비공식적으로 소개됩니다.

- **Technical Details**: 우리는 LLaVA 1.5(13B) 모델을 공격하기 위해 여러 개의 이미지를 입력받아 쿼리와 함께 설정합니다. 공격 목표는 비 원치하는 출력 t′를 생성하여 클린 입력과 적대적 입력 간의 의미적 거리를 최소화하는 것입니다. PGD 공격을 구현하기 위해 NVIDIA A100 GPU를 활용하며, 성능을 극대화하기 위해 여러 기술적 세부사항을 조정합니다.

- **Performance Highlights**: 우리의 접근 방식은 LLaVA 1.5의 Helpfulness(도움성)와 Honesty(정직성) 차원에서 유의미한 개선을 보여주었습니다. 특히, 도움 요청 질문에서 약 22.14%의 향상이 확인되었습니다. 그러나 Harmlessness(무해성) 차원에서는 개선폭이 적었으며, 이는 이 차원의 본질적 도전 과제를 강조합니다.



### Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usag (https://arxiv.org/abs/2412.15606)
- **What's New**: 본 논문에서는 multi-modal agent tuning 방법을 제안하여 자동으로 다중 모달 도구 사용 데이터를 생성하고 이를 통해 비전-언어 모델(Vision-Language Model, VLM)을 조정하는 접근 방식을 취합니다. 이는 기존의 대형 언어 모델(LLM)이 도구 사용에서 갖는 한계를 극복하고, 복잡한 실제 작업을 해결하는 데 있어 더욱 강력한 도구 사용 추론 능력을 제공합니다.

- **Technical Details**: 우리의 접근 방식은 질문 생성, 파일 생성, 그리고 경로 생성의 세 가지 단계로 이루어진 mult-modal tool-usage data synthesis 파이프라인을 포함합니다. 이를 통해 우리는 약 20,000개의 도구 사용 경로(task trajectory)가 포함된 MM-Traj 데이터셋을 구축하였으며, T3-Agent를 통해 VLM 기반 에이전트를 개발했습니다. T3-Agent는 비전-언어 모델에 기반하여 도구 사용을 위한 강력한 추론 능력을 갖추게 됩니다.

- **Performance Highlights**: T3-Agent는 GTA와 GAIA 벤치마크에서 평가되었으며, MiniCPM-V-8.5B 및 Qwen-VL-7B와 같은 두 개의 인기 있는 VLM에서 일관된 성과 향상을 보였습니다. T3-Agent는 비훈련 VLM와 비교하여 20%의 성능 향상을 나타내며, 이는 제안된 데이터 합성 파이프라인의 효과 및 도구 사용 기능 향상에 기여함을 보여줍니다.



### QUART-Online: Latency-Free Large Multimodal Language Model for Quadruped Robot Learning (https://arxiv.org/abs/2412.15576)
- **What's New**: 이번 논문에서는 다중 모달 대형 언어 모델(MLLM)의 배포 시 발생하는 고유한 추론 지연(latency) 문제를 다루고 있습니다. 연구 결과, 기존의 매개변수 감소(Parameter Reduction) 기술이 언어 기초 모델의 성능을 저하시켜 액션 인스트럭션 튜닝(action instruction tuning) 단계에서 적합하지 않음을 발견했습니다. 저자는 QUART-Online이라는 새로운 지연 없는(quadruped MLLM) 모델을 소개하여 성능 저하 없이 추론 효율성을 향상시키는 방법을 제시합니다.

- **Technical Details**: QUART-Online은 Action Chunk Discretization (ACD) 기술을 활용하여 원래의 액션 표현 공간을 압축합니다. 이는 연속적인 액션 값을 더 작은 집합의 이산 대표 벡터로 매핑하여 핵심 정보를 유지하며 원활한 모델 출력을 가능하게 합니다. 이러한 구조를 통해 MLLM은 비전, 언어 및 압축된 액션을 통합하여 통합된 의미(semantic) 공간을 형성합니다.

- **Performance Highlights**: QUART-Online은 기존의 MLLM 시스템과 함께 작동하여, 기본 컨트롤러의 주파수(frequency)와 동기화된 실시간 추론이 가능함을 보여주며, 다양한 작업에서 65%의 평균 성공률 향상을 이루었습니다. 이러한 성과는 빠른 시스템 반응을 가능하게 하여 작업 완수 시 성능과 효율성을 극대화하는 데 기여합니다.



### Continual Learning Using a Kernel-Based Method Over Foundation Models (https://arxiv.org/abs/2412.15571)
- **What's New**: 본 논문은 지속적 학습(Continual Learning, CL) 중 클래스 증가 학습(Class-Incremental Learning, CIL)의 도전적인 설정을 다룹니다. 기존의 여러 방법에도 불구하고, 파라미터 업데이트로 인한 재앙적 망각(Catastrophic Forgetting, CF) 및 과제 간 클래스 분리(Inter-task Class Separation, ICS) 문제가 여전히 존재합니다. 이 문제를 해결하기 위해, Kernel Linear Discriminant Analysis (KLDA)라는 새로운 방법을 제안하며, 이 방법은 기초 모델(Foundation Model)에서 학습된 강력한 특징을 활용합니다.

- **Technical Details**: KLDA는 Radial Basis Function (RBF) 커널과 Random Fourier Features (RFF)를 통합해 기초 모델에서 추출된 특징 표현을 향상시킵니다. 새로운 작업이 도착하면 KLDA는 각 클래스의 평균을 계산하고, 커널화된 특징을 기반으로 모든 학습된 클래스에 대한 공유 공분산 행렬을 업데이트합니다. 이 방법은 Linear Discriminant Analysis (LDA)를 사용하여 분류를 수행하며, 각 클래스에 대한 가우시안 분포를 정의하여 결정 경계를 최적화합니다.

- **Performance Highlights**: KLDA는 텍스트 및 이미지 분류 데이터세트를 사용한 실험적 평가에서 기존 방법들보다 우수한 성능을 보였습니다. 특히, KLDA는 재생 데이터에 의존하지 않고도 CIL 성능의 상한으로 여겨지는 모든 클래스의 조합 훈련에 맞먹는 정확도를 달성하였습니다. 이는 기존의 다른 CIL 방법들이 모자란 정확도를 극복하는 데 중요한 의미를 갖습니다.



### VLM-RL: A Unified Vision Language Models and Reinforcement Learning Framework for Safe Autonomous Driving (https://arxiv.org/abs/2412.15544)
Comments:
          28 pages, 16 figures

- **What's New**: 최근 자율 주행 커뮤니티에서는 보강 학습(reinforcement learning, RL) 기반의 드라이빙 정책 학습 방법이 주목받고 있으며, 다양한 주행 시나리오에서 놀라운 발전을 이루었습니다. 본 논문에서는 VLM-RL이라는 통합 프레임워크를 제안하여, 사전 학습된 비전-언어 모델(vision-language models, VLMs)을 RL에 통합하여 이미지 관찰과 자연어 목표를 통해 보상 신호를 생성합니다.

- **Technical Details**: VLM-RL의 핵심은 대조 언어 목표(contrasting language goal, CLG)-을 보상으로 사용하는 패러다임입니다. 이는 긍정적 및 부정적 언어 목표를 활용하여 의미적 보상을 생성하며, CLG 기반의 의미적 보상과 차량 상태 정보를 결합하여 보상 안정성을 개선하는 계층적 보상 합성 방법을 도입합니다. 또한, 훈련 과정에서의 계산 효율성을 최적화하기 위한 배치 처리 기법이 사용됩니다.

- **Performance Highlights**: CARLA 시뮬레이터에서의 광범위한 실험 결과, VLM-RL은 최신 기술들의 기준을 초과하여 10.5%의 충돌률 감소와 104.6%의 경로 완료율 증가를 달성하였습니다. 또한 VLM-RL은 이전의 오프라인 보상 엔지니어링에 의존하는 기존의 RL 패러다임을 혁신할 수 있는 잠재력을 보유하고 있습니다.



### From Galaxy Zoo DECaLS to BASS/MzLS: detailed galaxy morphology classification with unsupervised domain adaption (https://arxiv.org/abs/2412.15533)
Comments:
          11 pages, 6 figures, accepted for publication in MNRAS

- **What's New**: 이 논문에서는 DECaLS 이미지로 훈련된 신경망을 BMz 이미지에 적용하기 어려운 이유(신호 대 잡음 비율과 해상도의 차이)를 설명하고, 이를 해결하기 위한 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 방법을 제안합니다. 제안된 방식은 DECaLS에서의 GZD-5 레이블로 훈련된 소스 도메인 모델을 BMz 이미지에 맞게 미세 조정하여, BMz 조사에서의 은하 형태 분류의 편향을 줄이는 것을 목표로 합니다.

- **Technical Details**: 이번 연구에서 사용되는 UDA 접근법은 두 단계로 구성됩니다. 첫 번째 단계에서는 DECaLS 이미지와 GZD-5 레이블을 활용하여 소스 도메인 모델을 훈련하고, 두 번째 단계에서는 BMz 조사에서의 248,088개의 레이블 없는 은하를 사용하여 소스 도메인 모델을 미세 조정합니다. 이 과정은 레이블이 없는 데이터만을 사용하여 적응을 진행하며, 적응이 성공적으로 이루어질 경우 BMz의 레이블 있는 3,618개의 은하에 대해 검증 성능이 향상됩니다.

- **Performance Highlights**: 모델의 성능은 DECaLS 갤럭시 검증 세트에서 관련 연구들의 결과와 유사한 수준을 보입니다. BMz 갤럭시에서 미세 조정된 타겟 도메인 모델의 성능은 소스 도메인 모델의 직접 적용 대비 크게 향상되어, 소스 도메인 모델의 성능 수준에 도달합니다. 이 연구는 중국 우주정거장 망원경, 유클리드 및 루빈 관측소와 같은 새로운 은하 조사 샘플에 기존의 딥러닝 모델을 구현하기 위한 시험 연구로 자리 잡게 될 것입니다.



### Underwater Image Quality Assessment: A Perceptual Framework Guided by Physical Imaging (https://arxiv.org/abs/2412.15527)
- **What's New**: 이 연구는 물리 기반의 수중 이미지 품질 평가(UIQA)를 위한 새로운 프레임워크인 PIGUIQA를 제안합니다. PIGUIQA는 광전파의 직접 전송 감쇠 및 역산란(backwards scattering) 효과를 고려한 통합적인 UIQA 문제로 정의됩니다. 또한, 이 방법은 깊이 학습(deep learning) 기술을 포함하여 해양 이미징의 물리 모델을 통합합니다.

- **Technical Details**: PIGUIQA는 지역적 세부정보(local details) 및 전체적인 지각(feature) 정보를 캡처하기 위해 깊이 학습 모델을 활용합니다. 특히, 이 프레임워크는 지역적 정보 기반으로 왜곡을 인지할 수 있는 지역 인식 모듈을 설계하여 이미지를 개선합니다. 뿐만 아니라, 전체 이미지 내용과 수중 이미지 왜곡 정보를 통합하는 글로벌 인식 모듈도 채택하였습니다.

- **Performance Highlights**: 실험 결과, PIGUIQA는 수중 이미지 품질 예측에서 최고 수준의 성능을 달성하였으며, 여러 상관 계수(correlation coefficients) 및 오류 측정(error metrics)에서도 우수한 성능을 보여줍니다. 또한, 다양한 데이터셋을 통한 교차 실험을 통해 강력한 일반화 및 견고성을 확인하였습니다.



### RESQUE: Quantifying Estimator to Task and Distribution Shift for Sustainable Model Reusability (https://arxiv.org/abs/2412.15511)
Comments:
          The Annual AAAI Conference on Artificial Intelligence (AAAI), 2025

- **What's New**: 본 논문에서는 딥러닝 모델의 재훈련 비용을 예측하기 위한 새로운 지표인 RESQUE(REpresentation Shift QUantifying Estimator)를 제안합니다. RESQUE는 모델이 새로운 데이터 분포나 작업에 적응하는 데 필요한 자원의 추정치를 제공하여, 사용자들이 재훈련에 대한 정보에 기반한 결정을 내릴 수 있도록 도와줍니다. 이를 통해 지속 가능한 AI 개발을 지원하고 환경에 미치는 영향을 줄이는 데 기여하고자 합니다.

- **Technical Details**: RESQUE는 모델의 원래 분포와 새로운 분포 간의 표현(output) 변화량을 측정하여 예측합니다. 두 가지 버전이 있으며, 하나는 새로운 분포에 대한 RESQUEdist이고, 다른 하나는 특정 작업에 대한 RESQUEtask입니다. 이 두 지표는 학습 시 단일 전방 전파(forward propagation)만을 사용하여 계산되며, 이를 통해 훈련 비용과 환경적 영향을 최소화하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, RESQUE는 재훈련 비용과 높은 상관관계를 나타내며, 에너지 소비 및 탄소 배출과 같은 지속 가능성 지표와도 강한 상관성을 보입니다. 또한 RESQUE는 다른 모델 아키텍처와 무관하게 효과적으로 작동하며, 다양한 작업 및 데이터 세트에서 적용 가능성을 입증했습니다. 이는 AI 모델의 적응성을 높이는데 기여하며, 자원과 지속 가능성 목표 달성에 효과적입니다.



### Stylish and Functional: Guided Interpolation Subject to Physical Constraints (https://arxiv.org/abs/2412.15507)
Comments:
          Accepted by Foundation Models for Science Workshop, 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이번 논문에서는 Generative AI가 공학 디자인의 실제 적용에서 창의성과 실용성을 조화롭게 결합하는 방법을 제안합니다. 특히, 물리적 제약과 기능 요구사항을 반영하여 두 개의 입력 디자인을 조합하여 새로운 디자인을 생성하는 프레임워크를 개발했습니다. 이를 통해 자동차 휠 디자인의 회전 대칭성 같은 특정 기능 요구를 만족하는 생성 과정을 중점적으로 다루고 있습니다.

- **Technical Details**: 제안된 시스템은 Functional Constraints in InTerpolation (FIT)라는 기능적 제약 조건을 명시적으로 코드화합니다. FIT의 주요 구성 요소로는 잠재적 인터폴레이션(latent interpolation), 기능적 제약 정규화(functional constraint regularization), 그리고 생성된 이미지와 정규화된 이미지를 통합하는 프로젝션(projection)이 포함됩니다. 이 시스템은 Latent Diffusion Model (LDM)을 기반으로 하여, 중간 이미지를 생성하는 과정에서 기능적 제약을 적용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 관련 연구의 방법보다 더 높은 현실성을 가진(interpolations with higher realism) 생성된 이미지들을 만들어냅니다. Fréchet Inception Distance (FID) 지표로 평가할 때, 생성된 인터폴레이션은 더 낮은 FID 점수를 기록하여 사실성과 다양성이 향상되었습니다. 이러한 결과는 우리의 프레임워크가 회전대칭성과 같은 기능적 요구사항을 더 잘 충족할 수 있도록 도와줌을 보여줍니다.



### A Robust Prototype-Based Network with Interpretable RBF Classifier Foundations (https://arxiv.org/abs/2412.15499)
Comments:
          To appear at AAAI 2025. Includes the Appendix

- **What's New**: 이번 연구에서는 Prototype-Based Networks (PBN)의 심화판인 Deep Prototype-Based Networks (PBNs)를 분석합니다. 특히 Classification-by-Components (CBC) 접근 방식을 중심으로 해석가능성과 관련된 여러 문제를 다루며, 그러한 문제를 해결하기 위한 CBC의 확장을 제안합니다. 마지막으로, 제안된 모델의 강건성을 보장하는 손실 함수를 도출하여 이론적 근거를 제시합니다.

- **Technical Details**: 본 연구는 심층 PBN이 Deep Radial Basis Function (RBF) 분류기와 관련이 있음을 보여줍니다. 고찰된 모델들은 입력 데이터의 잠재 공간을 통해 특징 추출(Feature Extractor)과 유사성 계산(Similarity Calculation) 과정을 거쳐 클래스를 예측합니다. 이러한 과정에서 RBF 사용과 네거티브 추론(Negative Reasoning)의 역할을 분석하고, 기존 모델의 문제를 해결하기 위한 새로운 구조를 만듭니다.

- **Performance Highlights**: 실험 결과, 제안된 심층 PBN은 여러 기준에서 최고의 분류 정확도를 기록하며, 기존 접근 방식의 해석 가능성 문제를 해결했습니다. 또한, 얕은 PBN 변형은 기존의 얕은 PBN들보다 우수한 성능을 보이면서도 본질적으로 해석 가능하고 입증된 강건성을 갖추고 있습니다. 이러한 성능 향상은 PBN이 OOD(Out-Of-Distribution) 탐지에 적합하다는 사실을 뒷받침합니다.



### Task-Specific Preconditioner for Cross-Domain Few-Shot Learning (https://arxiv.org/abs/2412.15483)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문은 Cross-Domain Few-Shot Learning (CDFSL) 분야에서 새로운 적응 메커니즘인 Task-Specific Preconditioned gradient descent (TSP)를 제안합니다. 기존의 방법들이 사용하던 고정 최적화 전략의 한계를 극복하기 위해, 우리 방법은 도메인 특화 전처리기 (Domain-Specific Preconditioners, DSPs)를 메타 학습하며 각 도메인의 특성을 반영합니다. 이러한 전처리기를 통해 목표 작업에 적응하도록 최적화를 진행하는 방식입니다.

- **Technical Details**: TSP는 메타 학습을 통해 각 도메인에서의 기초 조건을 형성하고, 이를 작업 계수(task-coefficients)와 선형 결합하여 작업 특정 전처리기(Task-Specific Preconditioner)를 생성합니다. 이 전처리기는 그래디언트 강하에 적용되며, 최적화 방향이 가장 급격한 하강 방향을 따르도록 특정 조건(positive definite)으로 제한됩니다. 이를 통해 각 목표 작업에 맞춘 적응적인 최적화를 가능하게 합니다.

- **Performance Highlights**: Meta-Dataset에서의 경험적 평가 결과, TSP는 다양한 실험 환경에서 기존의 최고 성능을 능가하는 성과를 달성했습니다. 이는 TSP가 Cross-Domain FSL 작업에서 일반화 능력을 높임을 입증하며, 동시에 meta-learning의 효과적인 활용 가능성을 보여줍니다. 전반적으로, 이 연구는 FSL 분야의 발전에 크게 기여할 것으로 기대됩니다.



### Uncertainty Estimation for Super-Resolution using ESRGAN (https://arxiv.org/abs/2412.15439)
Comments:
          8 pages, 6 figures. VISAPP 2025 camera ready

- **What's New**: 이번 연구는 이미지 초해상도(Super-Resolution, SR) 모델인 SRGAN과 ESRGAN의 불확실성을 측정하기 위해 Monte Carlo-Dropout과 Deep Ensemble 기법을 결합하여 성능을 향상시킵니다. 이를 통해 초해상도 결과와 함께 각 픽셀의 불확실성 맵을 제공하여 오류 가능성을 시각적으로 강조합니다. 또한, 이 연구는 모델이 제공하는 불확실성 추정치가 신뢰할 수 있도록 잘 보정되었음을 보여주고, 불확실성 추정이 성능 저하 없이 이루어질 수 있음을 확인하였습니다.

- **Technical Details**: SRGAN과 ESRGAN 모델의 기본 구조는 Generative Adversarial Network(GAN)를 기반으로 하며, 생성기는 저해상도 이미지를 초해상도로 변환하고, 판별기는 생성된 이미지와 실제 고해상도 이미지를 구별하는 방식으로 훈련됩니다. 이 연구에서는 다섯 개의 ESRGAN 생성기를 앙상블 방식으로 결합하여 SR 결과의 신뢰도를 높이는 불확실성 추정이 가능합니다. 각 이미지의 불확실성을 픽셀 단위로 계산하여 사용자에게 오류를 탐지하는 데 도움을 줍니다.

- **Performance Highlights**: 본 논문에서 제안한 불확실성 추정 방법은 다양한 공인 데이터셋에서 성능이 평가되었으며, 결과적으로 사용자에게 피드백을 제공하는 데 효과적임을 확인하였습니다. SRGAN과 ESRGAN의 성능이 향상되고, 불확실성 추정 방법이 잘 보정되어 오류 탐지에 유용함을 보여주었습니다. 이 연구는 SR 분야에서 불확실성 추정을 통합하여 제안하면서, 기존의 SR 모델이 훈련 분포 외부의 입력에 대해 발생할 수 있는 오류를 사전에 경고하는 중요한 역할을 할 수 있음을 입증합니다.



### Leveraging Weak Supervision for Cell Localization in Digital Pathology Using Multitask Learning and Consistency Loss (https://arxiv.org/abs/2412.15392)
- **What's New**: 이 연구는 디지털 병리학에서 세포 수를 직관적으로 추정하는 eyeballing 방식을 도입하여 다중 작업 네트워크를 훈련하는 혼합 감독(mixed-supervision) 접근 방식을 제안합니다. 이는 기존의 세포 경계에 대한 강한 주석이 아닌 약한 주석을 활용하여 모델의 성능을 향상시키는 데 기여합니다. 이 연구는 cell counting과 cell localization을 동시에 학습하도록 설계된 다중 작업 네트워크를 최초로 제안하며, 새롭게 도입된 일관성 손실(consistency loss)을 통해 두 작업의 예측 간 불일치를 정규화합니다.

- **Technical Details**: 이 연구에서 제안하는 다중 작업 네트워크는 각 작업에 대해 별도의 분기(branch)를 두고, training image에 따라 사용 가능한 주석의 수준에 따라 다른 감독 신호를 사용하여 훈련됩니다. 훈련 데이터의 모든 이미지에 대한 ground truth 세포 수는 eyeballing 방법으로 얻어지고, 포인트 주석(point annotations)은 소규모 하위 집합에서만 사용됩니다. 제안된 일관성 손실 함수는 세포 수 예측과 세포 위치 예측에서 발생하는 불일치를 페널티(reduce)하여 학습을 정규화하는 역할을 합니다.

- **Performance Highlights**: 제안된 방법론은 hematoxylin-eosin으로 염색된 두 가지 조직 이미지 데이터셋에서 테스트되었습니다. 실험 결과, eyeballing 방식으로 얻은 세포 수를 활용한 다중 작업 학습이 강한 주석이 부족한 상황에서 모델의 성능을 향상시키는 것으로 나타났습니다. 이 연구는 데이터 주석 작업의 자원 소모를 줄이면서도 모델이 효과적으로 학습할 수 있는 잠재력을 보여줍니다.



### DCRA-Net: Attention-Enabled Reconstruction Model for Dynamic Fetal Cardiac MRI (https://arxiv.org/abs/2412.15342)
- **What's New**: 본 연구에서는 Dynamic Cardiac Reconstruction Attention Network (DCRA-Net)라는 새로운 심층 학습 모델을 도입하여, 비가동(non-gated) MRI 영상을 기반으로 한 태아 심장 동역학 재구성을 시도합니다. DCRA-Net은 주어진 문제의 고유한 도전에 대응하기 위해 공간적 및 시간적 도메인에서 주의(attention) 메커니즘을 활용합니다. 이전 연구들과의 성능 비교를 통해, DCRA-Net이 태아 및 성인 심장 MRI 모두에서 우수한 재구성 능력을 발휘함을 보였습니다.

- **Technical Details**: DCRA-Net 모델은 2D + 시간(2D + time) 구조로, 시간적으로 배열된 MRI 데이터를 처리하기 위해 인코더-디코더 아키텍처를 사용합니다. 각 인코더 및 디코더 블록은 ResNet 블록, 공간적 및 시간적 자기 주의 자기(attention) 레이어, 다운샘플링 및 업샘플링 레이어로 구성됩니다. 이러한 구성을 통해 모델은 계산 요구사항을 줄이면서도 높은 재구성 성능을 유지합니다.

- **Performance Highlights**: DCRA-Net은 14명의 태아 및 39명의 성인 데이터를 대상으로 실험을 실시하였으며, 모두 기존의 L+S 및 k-GIN 방법과 성능 비교를 하였습니다. 이 모델은 태아 경우 38 PSNR(peak signal-to-noise ratio), 성인 경우 35 PSNR로 높은 성능을 기록했습니다. 이 방법은 공개적으로 사용 가능하며, 의학적 이미지 재구성을 위한 새로운 가능성을 보여주고 있습니다.



### Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models (https://arxiv.org/abs/2412.15341)
- **What's New**: 이 논문에서는 쪼갠(Pruned) 확산 모델을 위한 새로운 이수준 최적화(bilevel optimization) 프레임워크를 제안합니다. 이 프레임워크는 미세 조정(fine-tuning)과 잊기(unlearning) 프로세스를 통합하여 생성 품질을 유지하면서 불필요한 콘텐츠 생성을 선택적으로 억제합니다. 기존의 두 단계 접근법에 비해 효율적이고 안전한 배포가 가능하도록 돕습니다.

- **Technical Details**: 제안한 프레임워크는 미세 조정 데이터셋에서 모델의 생성 능력을 복구하기 위해 표준 증류(distillation) 및 확산 손실 최소화를 수행하는 하위 단계 최적화를 포함합니다. 상위 단계는 모델이 불필요한 개념을 생성하지 않도록 방향성을 제시합니다. 이 방식은 다양한 가지치기(pruning) 방법과 색인 제거(concept unlearning) 기술과 호환됩니다.

- **Performance Highlights**: 광범위한 아티스트 스타일 및 NSFW 콘텐츠 제거 작업에 대한 평가를 통해, 제안한 이수준 방법이 두 단계 접근법보다 상당히 우수함을 입증합니다. 높은 생성 품질을 유지하면서도 효과적인 개념 억제가 가능함을 보여주며, 실제 통제된 배포 환경에서의 유효성을 강조합니다.



### Federated Learning for Coronary Artery Plaque Detection in Atherosclerosis Using IVUS Imaging: A Multi-Hospital Collaboration (https://arxiv.org/abs/2412.15307)
- **What's New**: 이 연구에서는 Percutaneous Coronary Intervention (PCI) 동안의 Intravascular Ultrasound (IVUS) 이미지 해석에서 효율성을 높이기 위한 새로운 parallel 2D U-Net 모델을 제안합니다. 이 모델은 federated learning을 활용하여 보안성과 개인 정보 보호를 유지하면서도 여러 병원 간의 데이터 분석을 가능하게 합니다. 특히, External Elastic Membrane (EEM)과 lumen 영역을 식별하여 플라크를 효과적으로 세분화하며, Cartesian 좌표계를 polar 좌표계로 변환해 계산 효율성을 높였습니다.

- **Technical Details**: 제안된 multi-stage segmentation 구조는 IVUS 이미지에서 플라크, EEM, lumen을 각각 정확히 식별합니다. 이 과정에서 데이터를 미리 처리하여 두 영역을 뺀 후 플라크를 식별합니다. 모델은 Dice Similarity Coefficient (DSC) 0.706을 달성하여 실시간으로 플라크의 경계를 효율적으로 탐지합니다. 또한, 수술 과정에서 필요한 정량적 측정을 가능하게 하여 도메인 전문가와의 협업을 통해 플라크 부담 해석을 강화합니다.

- **Performance Highlights**: 제안한 모델은 여러 병원 간의 협력적 데이터 분석을 통해 진단 및 분류 성능을 향상시키는 잠재력을 가지고 있습니다. 위의 federated learning 프레임워크는 병원 간 데이터 교환 문제를 해결하며, 모든 참가 병원들이 혜택을 공유할 수 있는 환경을 조성합니다. 향후, 진보된 federated learning 기법과 데이터셋 확장을 통해 성능을 더욱 향상시킬 수 있는 가능성을 보여줍니다.



### Parametric $\rho$-Norm Scaling Calibration (https://arxiv.org/abs/2412.15301)
- **What's New**: 본 연구에서는 구조적 과신(overconfidence) 문제를 완화하기 위해 새로운 포스트 프로세싱 파라메트릭 보정 방법인 $\rho$-Norm Scaling을 소개합니다. 이 방법은 출력의 진폭(amplitude)을 조정하여 정확도를 유지하면서 데이터셋의 모델 신뢰도를 개선합니다. 또한, 샘플 수준의 불확실성(distribution uncertainty)을 반영할 수 있도록 확률 분포 정규화(probability distribution regularization)을 포함시킴으로써 보정된 확률 분포가 원래의 분포와 유사하게 유지되도록 합니다.

- **Technical Details**: 모델 출력의 신뢰도 보정은 출력의 불확실성 추정치를 정제하여 더 정확한 확률 예측을 가능하게 합니다. 기존의 파라메트릭 보정 방법들이 데이터를 처리할 때 발생하는 비효율성을 해결하기 위해, 본 연구는 새로운 $\rho$-Norm Scaling 모델을 제안하며, 이 모델은 원래 분포와 보정된 분포 간의 유사도를 고려하는 멀티 레벨 손실(multi-level loss)을 도입하여 최적화합니다. 이로 인해 출력-확률 매핑(output-probability mapping)이 더욱 효과적으로 학습되며, 과적합이 방지됩니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법은 여러 데이터셋 및 모델에 대해 최첨단(calibration performance) 보정 성능을 달성하여, 다른 기존 방법들에 비해 현저한 개선을 보여줍니다. 특히, bin-level와 instance-level 불확실성의 차이를 극복함으로써 더 정확한 모델 신뢰도를 제공합니다. 이러한 결과는 모델의 전반적인 예측 성능을 강화하는 데 기여할 것입니다.



### Analyzing Images of Legal Documents: Toward Multi-Modal LLMs for Access to Justic (https://arxiv.org/abs/2412.15260)
Comments:
          Accepted at AI for Access to Justice Workshop at Jurix 2024, Brno, Czechia. Code and Data available at: this https URL

- **What's New**: 이 논문은 법적 정보에 접근하기 어려운 일반인을 지원하기 위한 다중 모달 대형 언어 모델(LLMs)의 활용 가능성을 조사합니다. 특히, 손으로 작성된 서류의 이미지를 분석하여 관련 정보를 자동으로 추출하는 방법을 제안하고 있습니다. 초기 결과는 긍정적이지만 이미지 품질이 낮을 때의 한계도 드러났습니다.

- **Technical Details**: 논문에서는 '주거 임대 계약서(Standard Form of Lease)'의 이미지를 기반으로 한 데이터셋을 생성했습니다. 세 가지 시나리오를 설정하여 LLM이 이미지에서 정보를 얼마나 잘 추출할 수 있는지 평가했습니다. 각 시나리오는 유사한 이름을 가진 세 명의 세입자와 누락된 필드 등을 포함하여 점진적인 난이도로 설계되었습니다.

- **Performance Highlights**: 연구 결과, 다중 모달 LLM이 이미지를 통해 법적 문서에서 구조화된 정보를 추출하는 데 뛰어난 성능을 보였습니다. 이러한 접근은 일반인이 법적 권리를 이해하고, 정부 혜택을 신청하는 데 큰 도움을 줄 수 있을 것으로 기대됩니다. 하지만 이미지 품질과 복잡한 데이터가 결과에 미치는 영향은 여전히 해결해야 할 과제로 남아 있습니다.



### RoundTripOCR: A Data Generation Technique for Enhancing Post-OCR Error Correction in Low-Resource Devanagari Languages (https://arxiv.org/abs/2412.15248)
- **What's New**: 본 연구는 낮은 자원 언어에 대한 OCR(Optical Character Recognition) 오류 수정을 위한 데이터 생성을 다루고 있습니다. RoundTripOCR이라는 방법을 제안하며, 이는 OCR 출력 텍스트와 올바른 OCR 출력 텍스트 간의 맵핑을 학습하기 위해 기계 번역 기법을 활용합니다. 연구팀은 힌디어, 마라티어, 보도어, 네팔어, 콘카니, 산스크리트어에 대한 포스트 OCR 텍스트 수정 데이터셋을 공개했습니다.

- **Technical Details**: Devanagari 스크립트는 인도 아대륙에서 가장 널리 사용되는 문자 체계로, 텍스트 인식 과정에서 다양한 오류가 발생할 수 있습니다. 이 연구에서는 OCR 오류를 오역(mistranslation)으로 간주하고, 이를 교정하기 위해 사전 학습된 Transformer 모델을 활용하여 오류 텍스트와 올바른 텍스트 쌍 간의 매핑을 학습합니다. 새로운 방법론은 OCR을 처리하는 데 있어 일반적인 전처리, 기능 추출, 문자 분할 과정의 한계를 극복할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 제안된 RoundTripOCR 시스템은 기존의 OCR 시스템이 직면한 오류 유형을 효과적으로 해결할 수 있는 가능성을 보여줍니다. 연구팀은 3.1백만 개의 힌디어 문장, 1.58백만 개의 마라티어 문장 등의 포스트 OCR 오류 수정 데이터셋을 마련했습니다. 이 데이터셋은 다양한 언어에 대해 성능을 평가할 수 있는 기준점을 제공하며, 향후 연구에 중요한 자원으로 활용될 것입니다.



New uploads on arXiv(cs.AI)

### Formal Mathematical Reasoning: A New Frontier in AI (https://arxiv.org/abs/2412.16075)
- **What's New**: AI for Mathematics (AI4Math)은 과학, 공학 등에 있어 AI 중심의 발견에 매우 중요한 분야로, 공식적인 수학적 추론(Formal mathematical reasoning)이 이 분야를 한 단계 진전시키는데 필수적이라고 주장하고 있습니다. 최근 몇 년간 AI를 통해 정리 증명(Theorem proving)과 자동 정형화(Autoformalization) 같은 분야에서 일정한 진전을 이루었지만, 여전히 해결해야 할 중요한 도전 과제가 존재했습니다. 이 논문에서는 이러한 도전 과제들을 요약하고, 미래의 성공을 가늠할 수 있는 주요 이정표를 구상합니다.

- **Technical Details**: AI4Math 분야는 주로 자연어 처리(NLP)에서 차용한 기법을 사용하여 수학 LLMs를 개발하는 데 많은 연구가 집중되어 왔습니다. 특히, 잘 정제된(math) 데이터셋을 통해 LLM을 사전 훈련한 후, 세부적인 단계의 솔루션을 포함한 수학 문제의 데이터셋으로 모델을 미세 조정하는 방법이 사용됩니다. 또한, OpenAI o1의 사례처럼 비공식적인 접근방식을 평가 및 조정하는 과정에서 검색과 신경 검증기(neural verifiers)를 결합하여 환각된 추론의 문제를 줄일 수 있는 방향이 부각되고 있습니다.

- **Performance Highlights**: AlphaProof와 AlphaGeometry와 같은 시스템은 공식적인 수학적 추론을 통해 수학적 문제 해결에서 뛰어난 성과를 보였습니다. 이러한 시스템들은 신경망의 추론 단계를 실행하고 고품질 합성 데이터를 생성하여 전례 없는 수학적 추론 능력을 갖추게 되었습니다. 앞으로 이 분야는 자동화된 정형화와 강화 학습 등을 통해 더욱 발전할 가능성이 높습니다.



### A Framework for Streaming Event-Log Prediction in Business Processes (https://arxiv.org/abs/2412.16032)
Comments:
          18 pages

- **What's New**: 본 논문에서는 비즈니스 프로세스가 데이터 생성을 진행하는 동안 예측을 가능하게 하는 스트리밍 모드의 이벤트 로그 예측을 위한 Python 기반 프레임워크를 제안합니다. 이 프레임워크는 n-grams 및 LSTMs와 같은 스트리밍 알고리즘을 쉽게 통합하고, 여러 예측 모델을 앙상블(ensemble) 방식으로 결합할 수 있도록 합니다. 다양한 프로세스 마이닝 데이터 세트를 기반으로 한 실험을 통해, 배치(batch) 모드와 스트리밍 모드 간의 성능 차이를 비교하였습니다.

- **Technical Details**: 연구에서는 이벤트 로그 예측의 두 가지 패러다임인 배치 학습(batch learning)과 스트리밍 학습(streaming learning)에 대해 다룹니다. 특히, 스트리밍 모드에서는 데이터가 희소한 초기 단계에서도 빠르게 의미 있는 예측을 제공해야 합니다. 본 연구의 핵심은 각 사례의 이전 이벤트 데이터에만 의존하는 예측 함수를 생성하여 가장 가능성이 높은 다음 활동을 예측하는 것입니다.

- **Performance Highlights**: 실험 결과, LSTM 네트워크는 기본 모델인 prefix tree와 n-grams보다 전반적으로 뛰어난 성능을 보였으나, 단순 n-grams 모델도 LSTM 성능에 근접한 결과를 보였습니다. 앙상블 기법을 사용하여 기본 모델들을 조합할 경우, LSTM의 성능을 초과할 수 있는 가능성도 확인되었습니다. 실험은 헬스케어, 금융, IT 서비스 관리 등 다양한 도메인에서 7개의 실세계 프로세스 마이닝 데이터 세트를 기반으로 진행되었습니다.



### What Are Step-Level Reward Models Rewarding? Counterintuitive Findings from MCTS-Boosted Mathematical Reasoning (https://arxiv.org/abs/2412.15904)
Comments:
          AAAI 2025

- **What's New**: 최근 연구에서는 Step-level Reward Models (SRMs)가 수학적 추론 성능을 획기적으로 향상시키는 데 중요한 역할을 한다고 강조하고 있습니다. 특히, Monte Carlo Tree Search (MCTS) 방식이 SRMs의 성능을 극대화하는 데 효과적이라는 점을 부각시켰습니다. 이러한 연구는 SRMs의 작동 원리에 대한 이해를 심화시키고, 자연어 설명이 수학적 사고 과정에 필수적이지 않다는 가설을 제시합니다.

- **Technical Details**: 이 논문에서는 Step-level Reward Models (SRMs)와 Markov Decision Process (MDP)에 대한 기초 개념을 다루고 있습니다. MDP는 강화 학습 문제를 해결하는 데 필수적인 수학적 틀로, 상태, 행동, 보상 함수 및 미래 보상의 중요성을 결정하는 할인 요소를 포함합니다. SRMs는 프로세스 감독을 통해 수학적 추론을 개선하고, 수학적 언어에서의 복잡한 논리적 일관성을 평가할 수 있도록 훈련됩니다.

- **Performance Highlights**: SRMs는 수학적 언어에서의 논리적 일관성을 효과적으로 평가하며, 자연어에 대한 평가에는 어려움을 겪는 것으로 나타났습니다. 이는 SRMs가 수학적 언어에 대한 본질적인 친화성을 갖고 있다는 것을 시사합니다. 이 연구는 SRMs의 효율적인 훈련 방법을 찾고, 수학적 추론의 핵심 요소에 집중함으로써 더 나은 성능을 가능하게 할 것으로 기대하고 있습니다.



### Align Anything: Training All-Modality Models to Follow Instructions with Language Feedback (https://arxiv.org/abs/2412.15838)
- **What's New**: 이 연구는 여러 모드를 다루는 대규모 언어 모델의 성능을 향상시킬 수 있는 새로운 방법론을 제시합니다. 특히, RLHF(Reinforcement Learning from Human Feedback)를 여러 모드에 확장하여, 다양한 형태의 데이터(텍스트, 이미지, 오디오, 비디오)에서 인간의 선호에 기반한 정교한 모델 조정 접근법을 모색합니다. 이 연구는 이러한 다중 모드 문제를 다루기 위한 새로운 'align-anything' 프레임워크와 첫 번째 'eval-anything' 평가 체계를 제안합니다.

- **Technical Details**: 논문에서는 'align-anything-200k'라는 이름의 대규모 다중 모드 인간 선호 데이터셋을 제시하며, 이는 텍스트, 이미지, 비디오, 오디오 등 다양한 모드에 대해 사용자 선호를 캡쳐합니다. 기존의 데이터셋 한계를 극복하기 위해 2단계 인간 주석 과정을 통해 구축되었습니다. 또한 'learning from language feedback'라는 새로운 알고리즘을 통해 각 모드에 대한 RLHF 성능을 개선하며, 이는 평균 5.83배의 성능 향상을 이룹니다.

- **Performance Highlights**: 제안된 모델은 다중 모드 평가 도구인 'eval-anything'을 기반으로 성능 개선을 추적합니다. 이를 통해 다중 모드 모델이 더 높은 인스트럭션-팔로잉 능력을 발휘할 수 있도록 지원합니다. 더욱이, 연구진은 모든 데이터와 모델은 오픈 소스 형태로 공개하여, 연구 및 개발 커뮤니티가 쉽게 활용할 수 있도록 하고 있습니다.



### AutoLife: Automatic Life Journaling with Smartphones and LLMs (https://arxiv.org/abs/2412.15714)
Comments:
          13 pages

- **What's New**: 이 논문은 사용자의 일상 생활의 의미 있는 설명을 자동으로 생성하는 새로운 모바일 감지 응용 프로그램인 'Life Journaling'을 소개합니다. 본 시스템인 AutoLife는 상업용 스마트폰을 기반으로 하며, 사진이나 오디오 없이 저비용 센서 데이터를 입력으로 사용하여 사용자의 생활 일기를 자동으로 생성할 수 있습니다. AutoLife는 다양한 센서 데이터로부터 시간, 위치, 동작 맥락을 추출하고, 인간 생활에 대한 상식 지식으로 풍부해진 대형 언어 모델(LLM)의 제로 샷 능력을 활용합니다.

- **Technical Details**: AutoLife는 임의의 사용자 입력 없이 스마트폰의 센서 데이터를 기반으로 생활 일기를 생성합니다. 이를 위해 시간과 위치를 포함한 여러 맥락 정보를 결합하여 양질의 생명 저널을 생성하는 데 중점을 두고 있습니다. 데이터 수집의 효율성을 위해 다층 프레임워크를 설계하였으며, 긴 기간 동안의 센서 데이터를 분할하고 정제하여 최종적으로 LLM에 전달합니다.

- **Performance Highlights**: AutoLife 시스템은 홍콩의 자발적인 3명으로부터 수집한 다양한 행동 데이터셋을 통해 평가되었습니다. 실험 결과, Claude 3와 같은 일부 LLM을 사용했을 때 평균 BERTScore F1이 0.7 이상을 달성하는 등의 높은 정확도를 보여주었습니다. 이 연구는 향후 관련 연구의 벤치마크로 사용할 수 있는 공개 데이터셋을 제공할 예정이며, 기존의 생활 저널링 시스템과는 차별화된 방법론을 보여줍니다.



### Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration (https://arxiv.org/abs/2412.15701)
Comments:
          Preprint. Work in progress

- **What's New**: 최근 언어 모델(LM)에서의 발전은 LM 에이전트 개발에 대한 관심을 높이고 있습니다. 완전 자율 에이전트가 많은 상황에서 우수할 수 있지만, 인간의 잠재적 선호(preferences)나 분야 전문성(expertise), 통제(control) 필요성 때문에 많은 경우에는 인간과의 협업이 필수적입니다. 이를 위해 우리는 인간-에이전트 협업 연구를 용이하게 하는 일반적인 프레임워크인 Collaborative Gym (Co-Gym)을 제안합니다.

- **Technical Details**: Co-Gym은 에이전트, 인간 및 작업 환경 간의 비동기(asynchronous) 삼자 간 상호작용을 가능하게 하는 프레임워크입니다. 이 프레임워크는 시뮬레이션 및 실제 환경에서의 세 가지 대표 작업으로 구체화되며, 협업 결과 및 과정을 평가하는 평가 프레임워크를 제안합니다. 연구 결과, 협업 에이전트는 실제 사용자 평가에서 86%의 승률을 기록하는 여행 계획(Travel Planning), 74% 를 기록하는 표형 분석(Tabular Analysis), 66% 를 기록하는 관련 작업(Related Work) 등에서 완전 자율 에이전트보다 일관되게 우수한 성능을 보였습니다.

- **Performance Highlights**: 협업 에이전트는 특정 작업 상황에서 완전 자율 에이전트보다 더 나은 성과를 나타내었습니다. 그러나 연구는 협업 에이전트를 개발하는 데 있어서 커뮤니케이션 능력, 상황 인식(situational awareness), 자율성과 인간 통제의 균형 조절 등 핵심 인텔리전스(aspects of intelligence)의 발전이 필요하다는 중요한 도전 과제를 강조합니다.



### AIR: Unifying Individual and Cooperative Exploration in Collective Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2412.15700)
- **What's New**: 본 논문에서는 협력적인 멀티 에이전트 강화 학습(MARL)에서의 탐색 문제를 해결하기 위해 Adaptive exploration via Identity Recognition(AIR) 방법을 제안합니다. AIR는 에이전트의 정체성을 인식하여 탐색 모드와 강도를 조절하는 분류기와 행동 선택기로 구성된 두 개의 적대적 구성 요소로 이루어져 있습니다. 이 방법은 개인 및 집단 탐색을 원활하게 할 수 있도록 이론적으로 입증되었습니다.

- **Technical Details**: AIR는 에이전트의 궤적을 기반으로 한 정체성 분류기와 탐색 모드를 동적으로 조정하는 행동 선택기를 통해 작동됩니다. 기존의 탐색 전략의 한계를 분석하고, 개인 행동 및 에이전트 간의 공동 행동을 모두 고려하는 프레임워크를 제안합니다. 이 방식은 과도한 추가 모듈 없이도 효율적인 계산 자원 활용을 가능하게 하여 학습 과정을 단순화합니다.

- **Performance Highlights**: 실험 결과, AIR는 다양한 멀티 에이전트 작업에서 기존 방법보다 효율적이고 효과적으로 작동함을 보여줍니다. AIR는 조화롭게 개인 및 집단 탐색을 통합하여 에이전트 간의 협력을 증진시킵니다. 이 연구는 개인 탐색과 집단 탐색을 통합한 최초의 시도로서, MARL 분야에서의 새로운 가능성을 제시합니다.



### Adaptable and Precise: Enterprise-Scenario LLM Function-Calling Capability Training Pipelin (https://arxiv.org/abs/2412.15660)
Comments:
          23 pages, 6 figures, 7 tables

- **What's New**: 본 논문에서는 실제 비즈니스 환경에 적합한 기능 호출 모델을 위한 훈련 파이프라인을 제안합니다. 이 파이프라인은 시나리오별 기능 호출 데이터의 합성과 증강, 모델 미세 조정, 성능 평가 및 분석을 포함합니다. 이를 통해 디지털 HR 에이전트 시나리오에서 1,260개의 AI 생성 샘플과 1,035개의 수동 라벨링 샘플을 생성하였습니다. 연구 결과, 우리는 Qwen2.5-Coder-7B-Instruct 모델을 기반으로 GPT-4 및 GPT-4o보다 뛰어난 성능을 달성했습니다.

- **Technical Details**: 이 연구에서는 전문적인 시나리오에 맞춘 기능 호출 능력을 위한 자동화된 훈련 파이프라인을 설계하였습니다. 파이프라인에는 데이터 합성과 증강, LoRA를 이용한 SFT(Supervised Fine-Tuning), 모델 성능 평가 및 분석이 포함됩니다. Qwen2.5-Coder-7B-Instruct 모델을 4대의 GPU에서 LoRA 방법으로 미세 조정하였으며, 이 과정은 약 5시간 이내에 완료되었습니다.

- **Performance Highlights**: 미세 조정된 모델은 테스트 세트에서 구조적 완결성, 도구 선택 정확성 및 매개변수 입력 정확성을 넘어서는 우수한 성능을 보였습니다. 이러한 결과는 제안된 파이프라인의 효과를 입증하며, 중형 LLM에서의 기능 호출 가능성을 향상시키는 데 기여할 것입니다. 또한, 이 연구는 중소기업이 자신의 요구에 맞춘 에이전트 모델을 쉽고 효율적으로 훈련하고 배포할 수 있는 가능성을 제시하였습니다.



### Understanding Individual Agent Importance in Multi-Agent System via Counterfactual Reasoning (https://arxiv.org/abs/2412.15619)
- **What's New**: 최근 여러 응용 분야에서 멀티 에이전트 시스템(Multi-Agent System, MAS)의 설명이 필요해지고 있습니다. 기존 연구들은 에이전트의 행동이나 상태에 대한 설명을 제공했지만, MAS 내에서 에이전트의 중요성을 이해하는 것은 부족했습니다. 이를 해결하기 위해 저자들은 EMAI라는 새로운 에이전트 수준의 설명 접근 방식을 제안하며, 이는 개별 에이전트의 중요성을 평가합니다. 이 방법은 반사적 사고(counterfactual reasoning)에 기반하여, 에이전트의 무작위 행동으로 인해 보상에서 발생하는 변화의 크기를 중요성으로 정의합니다.

- **Technical Details**: EMAI는 MARL(Multi-Agent Reinforcement Learning) 문제로 모델링하여 에이전트 간의 상호작용을 캡처합니다. 특정 시간에 어떤 에이전트를 무작위적으로 행동하게 할 것인지 결정하는 정책을 학습함으로써 에이전트의 중요성을 더 정확하고 효율적으로 드러냅니다. 저자들은 목표 에이전트를 마스킹할 정책(masking agent)을 학습하는 최적화 문제를 정의하여, 무작위 행동 전후 보상 차이를 최소화하도록 설계합니다. 이를 통해 에이전트 간의 의존성 및 지연 효과를 고려하게 됩니다.

- **Performance Highlights**: EMAI는 7개의 멀티 에이전트 작업에서 실험을 통해 기존의 세 가지 최첨단 기법보다 설명의 신뢰도가 11%에서 118%까지 개선되었음을 보였습니다. 추가적으로, EMAI를 통해 에이전트의 중요성을 시각적으로 표시함으로써 정책 이해에 도움을 줄 수 있었습니다. 공격자는 EMAI를 사용하여 중요 에이전트를 식별하고, 이를 통해 상대적으로 14%에서 289%까지 성능이 개선된 공격을 실행할 수 있었습니다. 마지막으로, EMAI가 식별한 중요 에이전트를 수정함으로써 MAS의 성능을 크게 향상시킬 수 있음을 보여주었습니다.



### Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usag (https://arxiv.org/abs/2412.15606)
- **What's New**: 본 논문에서는 multi-modal agent tuning 방법을 제안하여 자동으로 다중 모달 도구 사용 데이터를 생성하고 이를 통해 비전-언어 모델(Vision-Language Model, VLM)을 조정하는 접근 방식을 취합니다. 이는 기존의 대형 언어 모델(LLM)이 도구 사용에서 갖는 한계를 극복하고, 복잡한 실제 작업을 해결하는 데 있어 더욱 강력한 도구 사용 추론 능력을 제공합니다.

- **Technical Details**: 우리의 접근 방식은 질문 생성, 파일 생성, 그리고 경로 생성의 세 가지 단계로 이루어진 mult-modal tool-usage data synthesis 파이프라인을 포함합니다. 이를 통해 우리는 약 20,000개의 도구 사용 경로(task trajectory)가 포함된 MM-Traj 데이터셋을 구축하였으며, T3-Agent를 통해 VLM 기반 에이전트를 개발했습니다. T3-Agent는 비전-언어 모델에 기반하여 도구 사용을 위한 강력한 추론 능력을 갖추게 됩니다.

- **Performance Highlights**: T3-Agent는 GTA와 GAIA 벤치마크에서 평가되었으며, MiniCPM-V-8.5B 및 Qwen-VL-7B와 같은 두 개의 인기 있는 VLM에서 일관된 성과 향상을 보였습니다. T3-Agent는 비훈련 VLM와 비교하여 20%의 성능 향상을 나타내며, 이는 제안된 데이터 합성 파이프라인의 효과 및 도구 사용 기능 향상에 기여함을 보여줍니다.



### Enhancing Large-scale UAV Route Planing with Global and Local Features via Reinforcement Graph Fusion (https://arxiv.org/abs/2412.15537)
- **What's New**: 이번 논문에서는 Unmanned Aerial Vehicle Route Planning (UAVRP) 문제를 해결하기 위한 새로운 일반화 프레임워크를 제안합니다. 이 프레임워크는 기존 UAVRP 솔버들이 최대 10,000개의 포인트를 처리할 수 있도록 능력을 확장할 수 있게 합니다. 수많은 탐색 지점으로 인해 발생하는 LSTSP의 대규모 요구 사항에 대응하기 위해, Delaunay triangulation 및 그래프 융합 결합과 같은 기법들이 적용됩니다.

- **Technical Details**: 우리의 프레임워크는 세 가지 중요한 단계를 포함합니다: 첫째, Delaunay triangulation을 사용하여 큰 인스턴스에서 서브그래프를 추출합니다. 둘째, 내장된 TSP 솔버를 통해 서브 결과를 얻고 이를 그래프로 융합합니다. 마지막으로 사용자 요구에 따라 조정 가능한 디코딩 전략을 구현하여 고품질 솔루션을 생성합니다.

- **Performance Highlights**: 제안된 프레임워크는 기존 TSP 솔버들을 효율적으로 확장하여 대규모 인스턴스를 다룰 수 있으며, 최신 기법들과 비교했을 때 일관되게 우수한 성능을 나타냅니다. 또한 이 프레임워크는 추가적인 훈련이나 파인 튜닝이 필요하지 않아, UAVRP 솔버에 대한 연구를 크게 발전시킬 수 있는 가능성을 가지고 있습니다.



### Quantifying detection rates for dangerous capabilities: a theoretical model of dangerous capability evaluations (https://arxiv.org/abs/2412.15433)
Comments:
          26 pages, 15 figures

- **What's New**: 이번 논문에서는 위험한 AI 능력을 시간에 따라 추적하기 위한 정량적 모델을 제시합니다. 정책 및 연구 커뮤니티가 위험한 능력 테스트가 AI 위험 접근을 사전에 경고하는 데 어떻게 기여할 수 있는지를 시각화하는 것을 목표로 합니다. 이 모델은 위험한 능력 검사가 정책에 직접적인 정보를 제공하는 방법에 대한 새로운 소개를 제공합니다.

- **Technical Details**: 위험한 능력 테스트 모델에서는 AI 시스템이 얼마나 위험한지를 추정하는 것이 주요 목표입니다. 이 모델은 위험 수준을 측정할 수 있는 테스트 세트를 정의하고, 각 테스트는 위험 수준을 감지하는 민감도 함수를 통해 평가됩니다. 결국, 최대 위험 수준을 감지하는 데 필요한 누적 분포 함수(CDF)와 테스트 민감도 간의 관계를 설명하며, 이는 AI 시스템의 위험성을 이해하는 데 핵심적으로 작용합니다.

- **Performance Highlights**: 위험한 능력 테스트의 실패는 두 가지 방식으로 나타날 수 있습니다: AI 위험 추정의 높은 편향과 임계값 모니터링에서의 큰 지연입니다. 이러한 실패 양상은 AI 능력의 역동성에 대한 불확실성과 개척 AI 실험실 간의 경쟁 때문에 발생합니다. 효과적인 AI 정책을 위해서는 이러한 실패 양상을 해결하는 것이 필수적이며, 시험 생태계를 위한 초기 권고 사항도 제시됩니다.



### Investigating Relational State Abstraction in Collaborative MARL (https://arxiv.org/abs/2412.15388)
- **What's New**: 이번 논문은 협력적 Multi-Agent Reinforcement Learning (MARL)에서 샘플 효율성과 성능에 끼치는 관계 상태 추상화의 영향을 탐구합니다. 새로운 critic 아키텍처인 MARC (Multi-Agent Relational Critic)를 제안하며, 공간 관계를 기반으로 하는 이 추상화가 비즈니스 환경에서 에이전트 간의 직접 통신이 불가능할 때의 성능을 향상시킬 수 있다는 점이 흥미롭습니다. 이는 복잡한 디자인이나 특정 작업에 대한 공학적 접근 없이도 이루어질 수 있음을 보여줍니다.

- **Technical Details**: MARC는 상태를 공간 그래프로 변환한 후, 관계형 그래프 신경망(relational graph neural network)을 통해 처리하는 간단하면서도 효과적인 critic 아키텍처입니다. 각 에이전트는 물체와 그 관계를 기반으로 한 그래프 표현을 통해 관찰 결과를 추상화하며, 이를 통해 샘플 효율성과 일반화 가능성을 모두 향상시킬 수 있습니다. 또한, 논문에서는 수중 로봇공학과 같은 환경에서 에이전트 간의 직접적인 통신이 불가능할 때 이 방법의 유용성을 강조합니다.

- **Performance Highlights**: MARC는 총 6가지 협력 작업 및 이질적인 에이전트를 활용한 새로운 환경에서 평가되었으며, 최신 MARL 기준들과 비교하여 샘플 효율성과 비대칭(performance)에서의 개선을 demonstrated합니다. 이는 에이전트와 객체 간의 관계 정보를 기반으로 한 설계 선택이 학습에 미치는 영향을 종합적으로 분석한 결과입니다. 우리의 연구 결과는 공간 관계적 유도 편향(spatial relational inductive biases)을 최소한으로 통합하는 것이 상당한 이점을 제공할 수 있음을 보여줍니다.



### Deep reinforcement learning with time-scale invariant memory (https://arxiv.org/abs/2412.15292)
- **What's New**: 이 연구에서는 scale invariant (스케일 불변) 메모리를 Deep Reinforcement Learning (딥 강화 학습) 에이전트에 통합하는 새로운 접근 방식을 제안합니다. 기존의 LSTM과 같은 반복 메모리 아키텍처와 달리, 이 에이전트는 다양한 시간 규모에서 강인하게 학습할 수 있도록 설계되었습니다. 이를 통해 신경과학 및 인지과학의 원리를 딥 뉴럴 네트워크에 적용하는 것이 가능합니다.

- **Technical Details**: 이 논문에서는 여러 신경 과학 연구에 기초하여, 시간 세포(time cells)와 ramping/decaying activity의 개념을 도입합니다. scale invariant 메모리 모델은 다양한 시범 환경에서 성능을 평가하며, 표준 메모리 아키텍처와 비교합니다. 이는 하이퍼파라미터 조정을 요구하는 대신, 다양한 시간적 관계에 대해 일반화된 학습 능력을 보여줍니다.

- **Performance Highlights**: 실험에서는 scale invariant 메모리를 가진 에이전트가 시간 변화에 대한 적응력을 보이는 반면, 전통적인 머신러닝 시스템은 특정한 시간 규모에서만 성능이 좋음을 보여주었습니다. 이 연구는 강화 학습 에이전트의 성능을 향상시키는 데 있어 인지 과학 이론의 접목 가능성을 탐색했습니다. 최종적으로, scale invariant 메모리를 적용한 에이전트가 다양한 시간적 관계에서 강한 성능을 유지할 수 있음을 입증합니다.



### MotiF: Making Text Count in Image Animation with Motion Focal Loss (https://arxiv.org/abs/2412.16153)
Comments:
          TI2V Bench is released in this https URL

- **What's New**: 이번 논문에서는 텍스트 이미지에서 비디오 생성(Text-Image-to-Video, TI2V) 기술의 한계를 극복하기 위해 Motion Focal Loss(MotiF)를 제안합니다. MotiF는 모션이 더 많은 영역에 모델의 학습을 집중시켜 텍스트 정렬(text alignment) 및 모션 생성(motion generation)을 개선하는 접근 방식입니다. 또한, 평가에 적합한 데이터세트 부족 문제를 해결하기 위해 320개의 이미지-텍스트 쌍을 포함하는 TI2V Bench를 제안합니다.

- **Technical Details**: MotiF는 주요 비디오의 모션 강도를 표현하는 모션 히트맵(motion heatmap)을 활용하여 손실 가중치를 할당합니다. 이 방식은 이미지가 제공하는 공간적 신호를 활용하면서도, 모션 중심의 학습을 가능하게 합니다. 이를 통해 모션 패턴에 대한 주의를 증가시키고, 다양한 조건 속에서 생성된 비디오의 질을 향상시킵니다.

- **Performance Highlights**: TI2V Bench에 대한 포괄적인 평가를 통해, MotiF는 9개의 공개 모델을 초월하여 평균 72%의 선호도를 기록하였습니다. 주요 평가 프로토콜은 A-B 테스트를 통해 주관적인 품질 평가를 해주어, 인간 평가자들이 동영상 간의 차별성을 명확히 인지할 수 있도록 합니다. MotiF는 텍스트 정렬과 모션 품질을 크게 개선하는 것으로 나타났습니다.



### Offline Reinforcement Learning for LLM Multi-Step Reasoning (https://arxiv.org/abs/2412.16145)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 다단계 추론 능력을 개선하기 위한 오프라인 강화 학습(offline reinforcement learning) 방법인 OREO(Offline Reasoning Optimization)를 제안합니다. 기존의 Direct Preference Optimization(DPO) 방법의 한계점을 극복하고, 다단계 추론에 적합한 새로운 접근 방식을 제공합니다.

- **Technical Details**: OREO는 최대 엔트로피 강화 학습(maximum entropy reinforcement learning)의 통찰을 바탕으로 소프트 벨만 방정식(soft Bellman Equation)을 최적화하여 정책 모델(policy model)과 가치 함수(value function)를 공동 학습합니다. 이는 데이터 수집의 부담을 줄이고, 다단계 추론에서 효과적인 신용 할당(credit assignment)을 가능하게 합니다.

- **Performance Highlights**: OREO는 수학적 추론 작업(GSM8K, MATH) 및 임베디드 에이전트 제어(ALFWorld)와 같은 다단계 추론 벤치마크에서 기존의 오프라인 학습 방법들을 능가하는 성능을 보였습니다. 이 방법은 추가 자원이 있을 경우 다중 반복(multi-iteration) 프레임워크로 확장할 수 있으며, 학습된 가치 함수를 통해 무료로 트리 탐색(tree search)을 안내하여 테스트 시 성능을 더욱 향상시킬 수 있습니다.



### Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation (https://arxiv.org/abs/2412.16135)
Comments:
          To appear in AAAI 2025, Main Track

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 신선한 obfuscated assembly 코드를 생성할 수 있는 가능성에 대해 탐구합니다. 기존의 obfuscation 도구는 원본 코드에 대한 접근이 필요하며, 새로운 obfuscation을 추가하는 것이 복잡하고 시간이 많이 소요됩니다. 저자들은 MetamorphASM 데이터셋(MAD)과 함께 세 가지 obfuscation 기술(즉, dead code, register substitution, control flow change)을 통한 평가를 통해 LLM이 obfuscated 코드를 생성할 수 있는 능력을 평가합니다.

- **Technical Details**: 저자들은 MetamorphASM 데이터셋을 사용하여 328,200개의 obfuscated assembly 코드 샘플을 생성했습니다. 이 데이터셋은 LLM의 obfuscation 능력을 검증하기 위한 첫 번째 assembly 코드 obfuscation 데이터셋입니다. 연구는 정보 이론적 메트릭스와 수작업 검사 방법으로 LLM의 성공률을 평가했으며, 다양한 LLM 모델(GPT-3.5/4, Starcoder 등)의 능력도 개별적으로 검토하였습니다.

- **Performance Highlights**: 연구 결과 LLM은 obfuscated assembly 코드 생성을 위한 여러 다른 기술에서도 적절한 성과를 보였습니다. 특히, dead code insertion, register substitution, control flow change와 같은 특정 obfuscation 기술에 따라 성능이 달라졌습니다. 이 연구는 LLM이 저비용과 플랫폼 독립성을 통해 고급 obfuscation 전략을 활용할 수 있음을 보여줍니다.



### Convolutional Deep Operator Networks for Learning Nonlinear Focused Ultrasound Wave Propagation in Heterogeneous Spinal Cord Anatomy (https://arxiv.org/abs/2412.16118)
Comments:
          Accepted for oral presentation at AAAI Conference on Artificial Intelligence: AI for Accelerating Science and Engineering Workshop 2025

- **What's New**: 이 연구는 기계학습 기술인 Convolutional Deep Operator Network(DeepONet)를 통해 환자의 척추에서 초음파 요법의 압력 분야를 신속하게 예측하는 방법을 제안합니다. 기존의 정밀한 전산 시뮬레이션 방법보다 훨씬 더 빠르게 압력 맵을 생성할 수 있으며, 이는 신경외과 수술에서 실시간 의사결정을 지원하는 데 중요한 역할을 합니다. DeepONet은 parametric partial differential equations(PDEs)의 해를 근사화할 수 있는 특성이 있어, 기존의 신경망보다 효율적입니다.

- **Technical Details**: 초음파 요법의 압력 분포를 정확하게 예측하는 것은 척추의 복잡한 기하학적 및 음향적 비균질성 때문에 매우 어렵습니다. 연구진은 약 2%의 테스트 손실률로 FUS 압력 필드를 예측하는 DeepONet 모델을 훈련시켰으며, 이는 다양한 환자 해부학에 대한 압력 맵의 시뮬레이션 데이터에서 형성되었습니다. 이러한 신경 연산자는 무한 차원 함수 공간 간의 매핑을 학습하는데 초점을 맞춰, 새로운 환자에 맞춰 재훈련이 필요하지 않아 실시간 사용에 적합합니다.

- **Performance Highlights**: DeepONet 모델은 환자의 척추에서 FUS 파동 행동을 예측하는 데 있어 놀라운 속도로 정확도를 보였습니다. 이 방법은 수술 실시간 의사결정에 필요한 매개변수 스윕(parameter sweep)을 촉진하여, 신경외과 치료에서 보다 정밀하고 개인화된 솔루션을 제공할 수 있는 중대한 단계로 작용합니다. 연구 결과는 비선형 물리 시스템 모델링의 속도를 획기적으로 향상시켜, 향후 다양한 신경외과적 문제에 대한 해결책으로 활용될 수 있습니다.



### Demystifying the Potential of ChatGPT-4 Vision for Construction Progress Monitoring (https://arxiv.org/abs/2412.16108)
- **What's New**: 이번 논문은 OpenAI의 GPT-4 Vision과 같은 Large Vision-Language Models (LVLMs)의 발전이 인공지능 분야에서 특히 시각 데이터 분석 및 해석에 있어 중요한 진전을 이뤘음을 보여줍니다. 실질적으로 건설 산업에서 GPT-4 Vision의 응용을 탐구하며, 이를 통해 건설 프로젝트의 진행 상황을 모니터링하고 추적하는 능력에 주목하고 있습니다.

- **Technical Details**: 연구는 고해상도 항공 이미지를 활용하여 건설 현장의 세부 장면 분석을 수행하고 시간에 따른 발전 변화를 추적합니다. GPT-4 Vision은 건설 단계, 자재 및 기계 식별에서 뛰어난 성능을 보이는 반면, 정확한 객체 위치 파악(Object Localization) 및 분할(Segmentation)에서 어려움을 겪고 있다는 점이 언급됩니다.

- **Performance Highlights**: 비록 이러한 한계가 표면화되긴 했지만, 이 기술의 미래 발전 가능성은 매우 큽니다. 본 연구는 현 시점에서 LVLMs를 건설 분야에 적용하는 상태와 기회를 강조할 뿐만 아니라, 도메인 특화 교육(Domain-specific Training) 및 다른 컴퓨터 비전 기법과 디지털 트윈(Digital Twins) 통합을 통해 모델의 유용성을 향상시킬 수 있는 미래 방향도 논의하고 있습니다.



### Explainable AI for Multivariate Time Series Pattern Exploration: Latent Space Visual Analytics with Time Fusion Transformer and Variational Autoencoders in Power Grid Event Diagnosis (https://arxiv.org/abs/2412.16098)
- **What's New**: 이 논문은 복잡한 패턴을 시각적으로 분석하는 새로운 프레임워크를 제안합니다. Time Fusion Transformer (TFT)와 Variational Autoencoders (VAEs)의 두 가지 생성 AI 모델을 통합하여 다변량 시계열 데이터의 복잡한 패턴을 저차원 잠재 공간으로 축소합니다. 이를 통해 PCA, t-SNE 및 UMAP과 같은 차원 축소 기술을 사용하여 2D로 시각화함으로써 데이터 패턴을 직관적으로 탐색할 수 있도록 합니다.

- **Technical Details**: 제안된 시각 분석 프레임워크는 복잡한 시간적 패턴의 유사성을 식별하고 잠재적 상관관계를 발견하는 데 중점을 둡니다. 다양한 모델 구성에서 TFT와 VAE의 성능을 평가하는 독특한 메트릭을 도입하며, 이로 인해 모델 매개변수 조정 및 신뢰성 향상에 도움을 줍니다. 특히, TFT는 다양한 시계열 데이터 형태에 대해 뛰어난 확장성과 짧은 실행 시간을 보여줍니다.

- **Performance Highlights**: TFT와 VAE 기반 방법의 비교 분석은 두 모델 간의 2D 잠재 벡터 표현의 일관성이 86%-92%에 달함을 밝혀냈습니다. TFT는 데이터 형태의 다양성에 대해 VAE보다 실행 시간과 확장성에서 더 우수한 성능을 보였습니다. 이 연구는 다변량 시계열 데이터의 고장 진단을 발전시키고, 의사결정 과정에서 설명 가능한 AI 접근 방식을 촉진하는 데 기여합니다.



### The Evolution of LLM Adoption in Industry Data Curation Practices (https://arxiv.org/abs/2412.16089)
Comments:
          19 pages, 4 tables, 3 figures

- **What's New**: 대규모 언어 모델(LLMs)의 발전은 데이터 큐레이션 업무에서 새로운 가능성을 제공합니다. 본 논문에서는 대규모 기술 회사 내에서 LLM의 채택과 데이터 큐레이션 작업에서의 영향을 평가하였습니다. 이를 위해 설문 조사, 인터뷰 및 사용자 연구 등을 통해 LLM의 진화와 조직의 대응 방안을 탐구했습니다.

- **Technical Details**: 연구는 세 단계로 진행되었습니다. 첫 번째 단계에서는 2023년 2분기에 LLM을 활용하는 Google 직원들에 대한 설문조사를 실시하여, LLM 사용의 현황을 파악했습니다. 두 번째 단계에서는 2023년 3분기에 데이터 실무자와 도구 개발자와의 인터뷰를 통해 복잡한 텍스트 기반 데이터셋에 대한 요구 사항 변화를 조사했습니다. 마지막으로, 2024년 3분기에는 사용자 연구를 통해 LLM 기반 프로토타입의 사용 가능성을 모색했습니다.

- **Performance Highlights**: 이 연구로 인해 데이터 큐레이션 작업에서 LLM의 다층 데이터셋 계층 구조가 나타났습니다. '골드 데이터셋'(gold datasets) 외에도 LLM으로 생성된 '실버 데이터셋'(silver datasets)과 전문가 팀에 의해 큐레이션된 '슈퍼 골드 데이터셋'(super golden datasets) 를 도입하여 데이터 품질에 대한 정의가 진화하고 있음을 보여줍니다. LLM을 통한 데이터 분석 방식의 전환은 실무자들이 더 전략적인 데이터 분석에 집중할 수 있게 하여, 전체 작업 흐름의 효율성을 크게 향상시킵니다.



### Towards Interpretable Radiology Report Generation via Concept Bottlenecks using a Multi-Agentic RAG (https://arxiv.org/abs/2412.16086)
Comments:
          Accepted in ECIR 2025

- **What's New**: 이 연구는 Deep learning을 활용하여 Chest X-ray (CXR) 분류에서 해석 가능성(interpretability)을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히, concept bottleneck models (CBMs)와 multi-agent Retrieval-Augmented Generation (RAG) 시스템을 결합하여 임상적 관련성을 지닌 방사선 보고서를 생성합니다. 이러한 방법은 모델의 예측을 인간이 이해할 수 있는 방식으로 명확하게 제시하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 접근법은 두 단계를 통해 이루어집니다. 첫 번째 단계에서는 질병 분류와 관련된 개념 기여도를 계산하고, 두 번째 단계에서는 임상 문서와 설명을 활용하여 견고한 보고서를 생성합니다. 모델은 GPT-4를 사용하여 각 질병 범주에 대한 자동 개념 발견을 수행하고, 이미지 임베딩(ChexAgent 모델 사용)과 텍스트 임베딩(Mistral Embed Model 사용)을 결합하여 개념 벡터를 생성합니다.

- **Performance Highlights**: COVID-QU 데이터셋에서 본 모델은 81%의 분류 정확도를 기록하였으며, 생성된 보고서는 84%에서 90% 사이의 성능 메트릭을 보였습니다. 이는 AI 기반 CXR 분석의 신뢰성을 높이는 해석 가능성과 높은 성능 간의 갭을 메우는 다중 에이전트 프레임워크를 구축하는 데 기여합니다.



### Label-Efficient Data Augmentation with Video Diffusion Models for Guidewire Segmentation in Cardiac Fluoroscopy (https://arxiv.org/abs/2412.16050)
Comments:
          AAAI 2025

- **What's New**: 이 연구는 Segmentation-guided Frame-consistency Video Diffusion Model (SF-VD)을 제안하여 안내선(segmentation guidewire)의 정확한 세분화를 위한 라벨 효율적인 데이터 증강 기법을 개발했습니다. 이 모델은 제한된 주석 데이터로부터 대량의 라벨링된 플루오로스코피(Fluoroscopy) 비디오를 생성함으로써 세분화 네트워크의 성능을 향상시키는 데 기여합니다. 또한, 이 연구는 의료 데이터 증강을 위해 생성 모델을 활용한 첫 번째 사례이며, 비디오 프레임 간의 일관성을 유지하면서 다양한 배경과 와이어 모양을 생성합니다.

- **Technical Details**: SF-VD는 두 개의 별도 2D Diffusion 모델을 활용하여 장면 분포(scene distribution)와 프레임 간 이동(motion distribution)을 독립적으로 학습합니다. 첫 번째 모델은 지정된 와이어 마스크에 따라 와이어가 배치된 2D 플루오로스코피 이미지를 생성하고, 두 번째 모델은 이러한 정적 이미지를 기반으로 다음 프레임을 생성하여 프레임 간 일관성을 보장합니다. 이 과정에서 와이어의 대비를 조정하여 다양성을 높이는 세분화 가이드 메커니즘을 사용하여 생성된 이미지의 가시성을 향상시킵니다.

- **Performance Highlights**: 연구 결과는 SF-VD가 기존 데이터 증강 방법보다 우수한 성능을 보여주며, 모든 테스트된 모델에서 세분화 성능을 향상시켰음을 입증했습니다. 특히, SF-VD는 제한된 라벨의 데이터로부터 생성된 비디오에서 높은 퀄리티를 유지하며 가이드와이어 세분화의 성능을 현저히 개선했습니다. 이로 인해 SF-VD는 향후 심장 개입 수술과 같은 의료 분야에서 중요한 역할을 할 것으로 기대됩니다.



### Applying Predictive Analytics to Occupational Health and Safety in India (https://arxiv.org/abs/2412.16038)
Comments:
          16 pages, 5 figures, 1 table

- **What's New**: 이번 논문은 직업 건강 및 안전(Occupational Health and Safety, OHS) 분야에서 predictive analytics의 혁신적인 변화를 다룹니다. 데이터 기반의 통찰력을 통해 위험을 사전 관리하고 정보에 기반한 의사결정을 가능하게 합니다. 특히 이 연구는 데이터 수집, 관리 및 준비 과정의 중요성을 강조합니다.

- **Technical Details**: 논문은 결측값 보간(missing value imputation), 이상 탐지(anomaly detection), 기능 엔지니어링(feature engineering) 등의 데이터 무결성(data integrity) 확보 과정을 상세히 설명합니다. 또한 위험 우선순위 결정(risk prioritization)을 통해 다양한 요인들, 예를 들어 직원 행동, 조직 정책, 환경 조건 및 운영 관행을 분석하고 위험 요소를 정렬합니다.

- **Performance Highlights**: 예측 모델로부터 얻은 통찰력은 사고 예방 및 리소스 최적화를 위한 핵심 분야에 집중하도록 기업을 안내합니다. 이 연구는 인도 내 OHS에서 predictive analytics의 실제 사례를 살펴보고, 소비자 보호와 윤리적 고려사항, 데이터 개인정보 보호(data privacy) 문제 및 예측 모델에 대한 과도한 의존의 위험도 논의합니다.



### The Only Way is Ethics: A Guide to Ethical Research with Large Language Models (https://arxiv.org/abs/2412.16022)
Comments:
          Accepted to COLING '25. This paper is the condensed pocket guide to accompany our full LLM Ethics Whitepaper, available at arXiv:2410.19812, and at this https URL for suggested revisions

- **What's New**: 이 논문에서 소개하는 'LLM 윤리 백서'는 대규모 언어 모델(LLM)의 윤리적 고려사항을 통합한 실용적인 가이드입니다. 이전의 여러 연구와 기구들이 제안한 윤리적 문제와 정책들을 한데 모아, NLP (Natural Language Processing) 실무자들이 참고할 수 있도록 구성되었습니다. 주목할 점은 기존의 문헌을 모아 명확한 Do's and Don'ts로 정리했으며, 이는 LLM을 다루는 연구자들에게 즉각적으로 적용 가능한 지침을 제공합니다.

- **Technical Details**: 논문은 LLM의 프로젝트 생애주기 전반에 걸쳐 적용할 수 있는 윤리적 고려사항을 제시합니다. NIST의 AI 위험 관리 프레임워크와 EU AI 법률처럼 널리 알려진 지침들과 비교하여, LLM 윤리 백서는 이론보다는 실용적인 지침을 제공합니다. 이를 위해 ACL Anthology와 Semantic Scholar에서 체계적인 문헌 조사를 실시하고, 관련된 자료를 수집하여 각 프로젝트 단계에 적합한 내용을 분류했습니다.

- **Performance Highlights**: 윤리적 연구를 지원하기 위해 마련된 LLM 윤리 백서는 실무자에게 유용한 자원으로 작용할 것입니다. 초기 탐색 과정에서의 리소스를 하이라이트하고, 이해관계자와 협력하는 데 도움이 되는 모범 사례를 제시합니다. 이 문서는 LLM 연구를 수행하는 모든 이에게 귀중한 참고자료가 될 것으로 기대됩니다.



### Choose Your Explanation: A Comparison of SHAP and GradCAM in Human Activity Recognition (https://arxiv.org/abs/2412.16003)
- **What's New**: 이 연구에서는 기계 학습 모델을 설명하기 위한 두 가지 주요 방법인 SHAP(Shapley Additive Explanations)와 GradCAM(Gradient-weighted Class Activation Mapping)을 비교 분석했습니다. 특히 인체 활동 인식(HAR) 분야에 대한 초점을 맞추어 그래픽 컨볼루션 네트워크(GCN)를 활용하여 실제 데이터셋에서 이들 방법의 강점과 한계를 평가하였습니다. 연구 결과는 사용자들이 특정 모델 및 애플리케이션에 가장 적합한 설명 방법을 선택하는 데 도움을 줄 수 있습니다.

- **Technical Details**: SHAP는 입력 기능에 대한 기여도를 자세히 설명하는 반면, GradCAM은 공간적으로 지향된 설명을 제공하는 방식으로 서로 보완하는 특성을 가집니다. 연구는 두 개의 실제 데이터셋에서 스켈레톤 기반 데이터에 대해 두 방법을 정량적 및 정성적으로 비교하여, 각 방법의 특징 중요도 순위, 해석 가능성 및 모델 민감도를 평가하였습니다. 이러한 분석은 의료와 같은 고위험 환경에서 신뢰성과 투명성을 증대시키는 데 필수적입니다.

- **Performance Highlights**: SHAP는 기능의 중요성을 상세히 기록하지만, GradCAM은 빠르고 간략한 설명을 제공합니다. 연구 결과, SHAP은 더 깊은 통찰을 제공하지만 공간적 및 시간적 동적 패턴을 캡처하는 데 있어 한계를 가집니다. GradCAM은 모델의 중요한 영역을 강조하지만 개별 입력 기능의 기여를 설명하는 데 있어 부재가 있습니다. 따라서 두 방법의 비교는 향후 HAR 모델의 신뢰성을 높이기 위한 중요한 요소가 될 것입니다.



### CNN-LSTM Hybrid Deep Learning Model for Remaining Useful Life Estimation (https://arxiv.org/abs/2412.15998)
Comments:
          conference paper

- **What's New**: 본 연구에서는 Remaining Useful Life (RUL) 예측을 위한 하이브리드 접근 방식을 제안합니다. 전통적인 회귀 방법들이 RUL 추정에서 낮은 정확도를 보였던 반면, CNN과 Long Short-Term Memory (LSTM) 네트워크를 결합하여 특징을 효율적으로 추출하고 RUL을 예측합니다. 이는 예측 유지보수 프로세스에서 RUL 추정을 위한 CNN-LSTM 모델을 적용한 최초의 시도로 평가받습니다.

- **Technical Details**: 하이브리드 CNN-LSTM 모델에서 처음에는 CNN이 데이터를 통해 특징을 효율적으로 추출하고, 그 후 LSTM이 이러한 추출된 특징을 사용하여 RUL을 예측합니다. 이 방법은 멀티 변량 시계열 분석을 통해 센서 시퀀스 정보를 활용하여 숨겨진 패턴을 찾아내며, 다양한 운영 조건 및 결함 시나리오에서도 견고한 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 하이브리드 CNN-LSTM 모델이 가장 높은 정확도를 달성하였으며, 다른 방법들과 비교했을 때 우수한 성과를 보였습니다. 이 연구는 RUL 예측 분야에서 새로운 가능성을 제시하며, 미래의 예측 유지보수 애플리케이션에 중요한 기여를 할 것으로 기대됩니다.



### Data-Centric Improvements for Enhancing Multi-Modal Understanding in Spoken Conversation Modeling (https://arxiv.org/abs/2412.15995)
Comments:
          22 pages, 6 figures, 14 tables

- **What's New**: 이 연구는 다중 모달 음성 모델링의 향상을 위해 데이터 중심(customization) 접근 방식을 도입하였습니다. 구체적으로, 소량의 음성 데이터만을 사용하여 다중 작업 학습(multi-task learning)을 통해 음성 이해를 효과적으로 증대시키는 방법을 제안합니다. 또한, 모호한 사용자 요청에 대한 대화형 질문 응답 데이터셋인 ASK-QA를 소개하여 새로운 연구 방향을 제시합니다.

- **Technical Details**: 이 논문에서는 모달리티 간 학습을 극대화하고자 하는 보조 작업(auxiliary tasks)을 설계하여 주어진 데이터 세트 내에서 상호 작용하는 알고리즘을 구축합니다. 세 가지 중간 목표는 올바른 음성 맥락 표현, 모든 입력 모달리티 간의 추론 학습, 및 올바른 답변 생성입니다. 주된 목표는 MLLM이 입력된 음성을 바탕으로 정확한 답변을 제공하는 것입니다.

- **Performance Highlights**: 제안된 접근 방식은 Spoken-SQuAD 벤치마크에서 10%의 훈련 데이터만을 사용하여 기존의 최첨단 성능을 초월하는 성과를 이루었습니다. ASK-QA와 Spoken-SQuAD를 포함한 세 가지 음성 질문 응답(SQA) 데이터 세트를 통해 이 모델의 효과성을 검증하고, 특히 데이터가 제한된 경우에도 우수한 결과를 보여줍니다.



### APIRL: Deep Reinforcement Learning for REST API Fuzzing (https://arxiv.org/abs/2412.15991)
Comments:
          Thirty-ninth Conference on Artificial Intelligence (AAAI 2025)

- **What's New**: APIRL은 REST API의 테스트를 위해 개별 HTTP 요청을 변형하여 버그를 찾는 새로운 자동화된 도구입니다. 이 도구는 사전 훈련된 transformer 모듈의 피드백을 활용하여 JSON 구조의 데이터를 이해하고, 알 수 없는 API 엔드포인트에 일반화할 수 있는 능력을 갖추고 있습니다. 이를 통해 기존의 방법보다 더 많은 버그를 발견하면서도 필요한 테스트 케이스의 수를 최소화할 수 있습니다.

- **Technical Details**: APIRL은 Markov Decision Process (MDP)를 기반으로 하여 초기 HTTP 요청에서 일련의 변화를 수행하여 테스트 케이스를 생성합니다. 이 과정에서 API의 응답 코드와 실행 정보를 사용하여 학습 시 보상을 결정합니다. 또한, 제안된 기술은 고급 Reinforcement Learning (RL) 방법론을 기반으로 하며, DQN(Deep Q-Network)과 PPO(Proximal Policy Optimization)와 같은 기법을 통합하여 기존의 간단한 피드백을 개선합니다.

- **Performance Highlights**: 26개의 실제 REST API의 평가 결과, APIRL은 현재의 최고 수준의 기술들보다 더 많은 버그를 발견하고 코드 커버리지와 테스트 케이스 효율성에서 유의미한 개선을 보여주었습니다. 이 연구는 RL 에이전트 학습의 섬세함과 다양한 설계 선택, 보상 함수에 대한 통찰력을 제공하여, 실질적인 테스트에서의 적용 가능성을 강화합니다. APIRL은 기계 학습 커뮤니티에 기대 이상의 기여를 할 것으로 기대됩니다.



### Never Reset Again: A Mathematical Framework for Continual Inference in Recurrent Neural Networks (https://arxiv.org/abs/2412.15983)
- **What's New**: 이번 연구에서는 RNNs(순환 신경망)의 성능 향상과 지속적인 추론(continual inference) 시의 한계를 극복하기 위한 적응형 손실 함수(adaptive loss function)를 제안합니다. 기존의 리셋 방식이 요구하는 복잡성을 제거하고, 입력 데이터의 정보량에 따라 동적으로 기울기(gradient)를 조절하여 정확도를 유지합니다. 이를 통해 RNN의 계속적 과제 수행 능력이 크게 향상되었습니다.

- **Technical Details**: RNN의 동작 원리를 이해하기 위해, 입력 시퀀스의 변화가 어떻게 RNN의 역동성을 저해하는지를 수학적으로 분석했습니다. 이 연구는 Kullback-Leibler divergence와 교차 엔트로피(cross-entropy)를 결합한 손실 함수의 개발을 통해, 기밀성 유지와 연속적인 출력 조정이 가능하도록 했습니다. RNN의 숨겨진 상태의 연속성을 보장하면서도 동적 학습(dynamically modulated learning)을 이뤄내는 것이 핵심입니다.

- **Performance Highlights**: 실험 결과, 제안한 리셋 없는 접근법(reset-free approach)은 전통적인 리셋 기반 방법에 비해 지속적인 작업에서 우수한 성능을 나타냅니다. 특히, 음성 인식 및 스트리밍 작업과 같은 분야에서 RNN이 유지하는 정보의 연속성이 향상되어, 실제 애플리케이션에서 신뢰성이 높아졌습니다. 이러한 개선은 RNN의 이론적 및 실용적 활용 가능성을 모두 증대시킵니다.



### Self-Supervised Radiograph Anatomical Region Classification -- How Clean Is Your Real-World Data? (https://arxiv.org/abs/2412.15967)
Comments:
          12 pages, 4 figures, 2 supplementary figures

- **What's New**: 이 연구는 자가 감독(self-supervised) 기법과 감독 대비(supervised contrastive) 깊이 학습법을 사용하여 14개의 해부학적 영역 클래스를 정확하게 분류할 수 있음을 입증합니다. 48,434개의 골격 방사선 사진이 포함된 데이터셋을 사용하여 단일 모델에서는 96.6%의 선형 평가 정확도를, 앙상블 방식에서는 97.7%의 정확도를 달성하였습니다. 특히, 훈련 세트의 1%에 해당하는 몇 개의 레이블된 인스턴스만으로도 92.2%의 정확도를 얻을 수 있어 자원이 부족한 상황에서도 사용 가능함을 보여줍니다.

- **Technical Details**: 연구는 14개의 해부학적 영역을 대상으로 48,434개의 방사선 사진을 DICOM 형식으로 분석하였습니다. OpenCV를 사용하여 이미지의 테두리와 회전을 정규화하였으며, 수술 계획의 원형 게이지 개선을 위해 새로운 데이터 증강 기법을 도입하였습니다. 또한, AspNet18 아키텍처를 기반으로 하여 1000 에폭 동안 Adam 최적화 알고리즘을 사용하여 사전 학습(pretraining)을 수행하였습니다.

- **Performance Highlights**: 모델의 성능은 실험 결과로 확인되었으며, 전문가의 후속 분석을 통해 35%의 잘못된 레이블과 11%의 도메인 외 이미지가 발견되었습니다. 이러한 오류를 고려했을 때, 단일 모델의 해부학적 영역 레이블링 성능은 이론적으로 98.0%까지 증가하고, 앙상블 사용 시에는 98.8%에 도달하는 성과를 보였습니다. 이 연구는 PACS 시스템의 데이터 품질을 높이기 위한 중요한 기여를 할 것으로 기대됩니다.



### From General to Specific: Tailoring Large Language Models for Personalized Healthcar (https://arxiv.org/abs/2412.15957)
- **What's New**: 의료 분야에서 의료 LLM의 개인화가 필수적이라는 문제를 다루고 있는 연구가 소개되었습니다. 본 연구에서는 개인화된 의료 언어 모델(PMLM)을 제안하며, 추천 시스템과 강화 학습(reinforcement learning, RL)을 통해 개인 맞춤형 LLM의 최적화를 탐구합니다. 특히, PMLM은 개인의 요구에 맞춘 최초의 개인화된 프롬프트를 설계하고 이를 RL을 통해 더욱 정제하여 LLM의 정확한 방향성을 활용하도록 합니다.

- **Technical Details**: 연구에서는 환자의 과거 데이터를 분석하여 개인화된 정보를 추출하고, 유사한 환자의 통찰력을 결합하여 원주율 프롬프트를 생성하는 프로세스를 설명합니다. 이러한 초기 프롬프트는 강화 학습을 통해 세밀한 개인화를 위해 정제됩니다. PMLM의 프롬프트는 하드 프롬프트(hard prompt)로, 이는 높은 적응성과 재사용성을 부여하여 다양한 LLM에 직접적으로 활용할 수 있습니다.

- **Performance Highlights**: 실제 산부인과 데이터를 통해 평가한 결과, PMLM은 개인화된 응답을 제공함으로써 기존의 세밀하게 조정된 LLM들보다 더 나은 성과를 보였습니다. 이 연구는 LLM의 개인화 가능성을 높이고, 개인 맞춤형 의료 LLM의 발전을 위한 새로운 경로를 제시합니다. PMLM은 다양한 질병에 대해 대응할 수 있는 가능성을 갖추고 있어, 향후 의료 분야에서의 LLM 활용에 중요한 기여를 할 것으로 기대됩니다.



### Trust Calibration in IDEs: Paving the Way for Widespread Adoption of AI Refactoring (https://arxiv.org/abs/2412.15948)
Comments:
          Accepted for publication in the Proc. of the 2nd Workshop on Integrated Development Environments, 2025

- **What's New**: 이번 논문은 AI 기반 리팩토링(AI refactoring)과 대형 언어 모델(Large Language Models, LLMs)을 통해 기존 코드를 개선하는 방법에 대해 논의합니다. 연구진은 IDE(통합 개발 환경) 내에서 모델과의 상호작용을 캡슐화하고, 신뢰할 수 있는 안전장치를 통해 리팩토링 시도를 검증할 필요성을 강조합니다. AI 리팩토링의 수용을 위해 신뢰 개발에 대한 연구도 중요하다고 주장합니다.

- **Technical Details**: AI 리팩토링은 개발자들이 기존 코드를 이해하고 유지보수하는 데 소요되는 시간과 노력을 감소시킬 수 있는 잠재력을 가지고 있습니다. 그러나 LLMs의 비결정적 코드 생성은 보안 취약점 및 기능 변화와 같은 위험을 초래할 수 있습니다. 연구진의 조사에 따르면, 최첨단 LLM의 원시 출력은 정확한 리팩토링을 제공하는 사례가 37%에 불과하며 이는 명확한 잠재적 결함을 나타냅니다.

- **Performance Highlights**: 연구팀은 IDE 내에서 LLM 출력의 첫 번째 게이트키퍼 역할을 수행할 수 있으며, 이는 개발 경험을 최적화하고 AI 리팩토링의 광범위한 채택을 실현하는 데 중요합니다. 다양한 연구 결과에 따르면, 기존의 리팩토링 및 자동 프로그램 수리(Automatic Program Repair, APR) 관련 연구들이 유효성 검증을 위해 성공적인 컴파일 및 유닛 테스트 통과를 주로 의존하고 있다는 점에서, 더 큰 규모의 연구가 필요하다고 지적합니다.



### Reframing Image Difference Captioning with BLIP2IDC and Synthetic Augmentation (https://arxiv.org/abs/2412.15939)
Comments:
          This paper has been accepted for the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 최근 몇 년 간 생성 모델의 품질이 향상되면서 이미지의 편집된 변형을 대규모로 생성할 수 있게 되었습니다. 이러한 기술의 유해한 영향을 줄이기 위해, 본 논문에서는 두 이미지의 차이를 설명하는 Image Difference Captioning (IDC) 작업을 다루고 있습니다. 기존의 아이디어를 바탕으로 저자들은 BLIP2 모델을 IDC 작업에 적응시키고 IDC 데이터셋을 증강하기 위한 효과적인 프레임워크를 제안하고 있습니다.

- **Technical Details**: 논문에서는 BLIP2 IDC(Adaptation of BLIP2 to Image Difference Captioning)를 소개하고 있으며, 이를 통해 효율적인 낮은 계산 비용으로 IDC 작업에 적합하게 조정할 수 있음을 보여줍니다. 또한, 생성 모델을 활용해 합성 데이터를 생성하여 IDC 모델의 성능을 향상시키는 방법을 탐구하고 있습니다. BLIP2는 이미지 캡셔닝을 위한 사전 훈련을 통해 IDC에 조정될 수 있으며, 이는 두 이미지를 동시에 입력받아 비교하는 방식으로 작동합니다.

- **Performance Highlights**: BLIP2IDC는 실세계 IDC 데이터셋에서 두 개의 스트림 접근 방식을 대폭 초월하는 성능을 발휘하는 것으로 나타났습니다. 신뢰할 수 있는 고품질 IDC 모델을 생성하기 위해, 저자들은 Syned1이라는 새로운 데이터셋도 제안하였으며, 이는 IDC 작업에 적합한 도전적인 데이터셋으로 미래의 연구에 기여할 것입니다. 마지막으로 여러 최신 모델에 대한 종합적인 평가를 제공함으로써 IDC 분야에서 현재 모델의 능력에 대한 더 명확한 이해를 돕고 있습니다.



### Watertox: The Art of Simplicity in Universal Attacks A Cross-Model Framework for Robust Adversarial Generation (https://arxiv.org/abs/2412.15924)
Comments:
          18 pages, 4 figures, 3 tables. Advances a novel method for generating cross-model transferable adversarial perturbations through a two-stage FGSM process and architectural ensemble voting mechanism

- **What's New**: 우리는 Watertox라는 새로운 적대적 공격 프레임워크를 제안합니다. 이 프레임워크는 아키텍처의 다양성과 정밀 제어 (precision-controlled)된 잡음을 통해 놀라운 효과를 달성합니다. 두 단계의 Fast Gradient Sign Method (FGSM)를 결합하여 기본적인 잡음과 타겟 확대를 전략적으로 진행함으로써 효과적인 공격을 구현합니다.

- **Technical Details**: Watertox의 중심에는 정밀 제어가 가능한 두 단계의 FGSM 구현이 있습니다. 첫 번째 단계에서 균일한 잡음으로 기본적인 파괴를 생성하고 두 번째 단계에서는 중요한 영역을 선택적으로 강화하여 시각 품질을 유지합니다. 여러 현대 및 고전 모델을 활용한 앙상블 아키텍처를 통해 다양한 신경망에서 효과적으로 전이할 수 있는 잡음을 생성합니다.

- **Performance Highlights**: 상태에 따라 가장 뛰어난 모델의 정확도를 70.6%에서 16.0%로 감소시키며, 제로샷 공격 평가에서는 최대 98.8%까지 정확도를 감소시키는 결과를 보였습니다. 이러한 결과들은 Watertox가 적대적 공격 방법론에서 중요한 발전을 이루었음을 드러내며, 시각 보안 시스템과 CAPTCHA 생성에서의 응용 가능성을 시사합니다.



### Less is More: Towards Green Code Large Language Models via Unified Structural Pruning (https://arxiv.org/abs/2412.15921)
Comments:
          UNDER REVIEW

- **What's New**: 이 논문에서는 Large Language Models(LLMs)의 높은 계산 요구와 에너지 소비 문제를 해결하기 위해 Flab-Pruner라는 통합 구조 프루닝 방법을 제안합니다. 이 방법은 어휘, 레이어 및 Feed-Forward Network(FFN) 프루닝을 결합하여 모델 매개변수를 효과적으로 감소시키면서도 성능을 유지할 수 있도록 설계되었습니다. 또한, 코드 생성 과제를 위한 맞춤형 코드 지침 데이터 전략도 도입하여 정제된 모델의 성능 회복 효율성을 높입니다.

- **Technical Details**: Flab-Pruner는 KL divergence를 사용하여 모델의 토큰 생성 확률이 원래 모델과 유사하게 유지되도록 최적화합니다. 이 방법은 FFN 프루닝, 레이어 프루닝, 어휘 프루닝의 세 가지 구성 요소로 구성되며, 각 구성 요소는 명확한 프루닝 목표를 가지고 있습니다. FFN 프루닝은 특정 뉴런을 제거하고, 레이어 프루닝은 레이어 간 중복성을 평가하여 레이어 수를 줄이며, 어휘 프루닝은 주어진 프로그래밍 코퍼스에 존재하지 않는 토큰을 제거합니다.

- **Performance Highlights**: Flab-Pruner는 광범위한 평가를 통해 22%의 매개변수를 제거한 후에도 원래 성능의 97%를 유지하며, 후속 훈련을 통해 동일하거나 더 나은 성능을 달성하는 것이 입증되었습니다. 또한, 정제된 모델은 저장소, GPU 사용량, 계산 효율성 및 환경 영향에서 상당한 개선을 보이며, 전반적인 강건성도 유지합니다. 이러한 연구 결과는 지속 가능한 소프트웨어 엔지니어링을 위한 해결책을 제공하고, LLMs의 효율적인 배포를 위한 기반을 마련합니다.



### Speedup Techniques for Switchable Temporal Plan Graph Optimization (https://arxiv.org/abs/2412.15908)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문은 Multi-Agent Path Finding (MAPF) 문제를 해결하기 위해 Switchable Temporal Plan Graph를 도입하였습니다. 이 프레임워크는 지연 상황에서도 충돌이 없고 Deadlock-free한 경로를 보장하는 비순환 Temporal Plan Graph를 찾는 방법을 제공합니다.

- **Technical Details**: 기존의 최적 알고리즘인 Mixed Integer Linear Programming 및 Graph-Based Switchable Edge Search (GSES)는 실행 속도가 느려 실용성이 떨어졌습니다. 본 연구에서는 Improved GSES를 제안하며, 이는 네 가지 속도 향상 기법을 통해 GSES의 처리를 가속화합니다: 더 강력한 admissible heuristics, edge grouping, 우선순위가 있는 branching, 그리고 incremental implementation이 포함됩니다.

- **Performance Highlights**: Improved GSES는 다양한 맵 유형과 에이전트 수에 대한 실험에서 GSES보다 성공률이 두 배 이상 높았고, 두 방법이 모두 솔루션을 찾은 경우에는 최대 30배의 속도 향상을 달성했습니다.



### Development of a Large-scale Dataset of Chest Computed Tomography Reports in Japanese and a High-performance Finding Classification Mod (https://arxiv.org/abs/2412.15907)
Comments:
          Dataset available at this https URL

- **What's New**: 이번 연구는 일본어 CT 리포트 데이터셋을 개발하고, 구조화된 발견(classification) 분류를 위한 전문화된 언어 모델을 구축한 점에서 주목받고 있습니다. 특히, CT 스캐너의 활용이 높은 일본에서 대규모 방사선의학 데이터셋의 부족 문제를 해결하기 위한 노력이 중요합니다.

- **Technical Details**: 연구진은 GPT-4o mini를 사용하여 CT-RATE 데이터셋(21,304명 환자의 24,283 CT 리포트)을 일본어로 번역했습니다. 훈련 데이터셋은 기계 번역된 22,778개의 리포트로 구성되었으며, 검증 데이터셋은 방사선 전문의의 검토를 거친 150개의 리포트로 이루어졌습니다. 또한, 'tohoku-nlp/bert-base-japanese-v3' 아키텍처를 기반으로 하는 CT-BERT-JPN 모델을 개발하여 18개 구조화된 발견을 추출했습니다.

- **Performance Highlights**: 번역 메트릭스에서는 BLEU 점수가 0.731, 0.690으로 강력한 성과를 보였으며, ROUGE 점수는 Findings 섹션에서 0.770에서 0.876까지, Impression 섹션에서는 0.748에서 0.857까지 달성했습니다. CT-BERT-JPN은 18개 조건 중 11개에서 GPT-4o보다 뛰어난 성능을 나타냈고, 특히 림프절 비대에서 +14.2%의 성능 향상을 기록했습니다. 이 모델은 18개 조건 중 14개에서 F1 점수가 0.95를 초과하며, 4개 조건에서는 완벽한 점수를 달성했습니다.



### On the Suitability of pre-trained foundational LLMs for Analysis in German Legal Education (https://arxiv.org/abs/2412.15902)
Comments:
          11 pages

- **What's New**: 이 논문은 최근 오픈 소스 기반의 LLM(대형 언어 모델)이 교육적 맥락에서 일부 법률 분석에 충분한 지시 능력 및 독일 법률 배경 지식을 보유하고 있음을 보여줍니다. 그러나, 'Gutachtenstil' 성격의 구성 요소 분류와 같은 특정 작업이나 복잡한 맥락에서 모델의 능력이 저하된다는 점을 지적합니다. 이를 보완하기 위해 Retrieval Augmented Generation 기반의 프롬프트 예제 선택 방법을 소개하여 데이터가 많이 존재하는 상황에서 예측 성능을 크게 향상시킵니다.

- **Technical Details**: 대형 언어 모델(LLMs)은 웹 스케일 코퍼스를 기반으로 훈련되어 인간의 지식과 커뮤니케이션을 포괄하는 상당한 교차 섹션을 제공합니다. LLMs는 자연어 지시를 위한 일반화 능력을 제공하며, 이를 통해 법률 분석과 같은 특정 분야에서도 일반 모델을 적용할 가능성을 제시합니다. 특히, 'Gutachtenstil'(감정 스타일)의 요소를 LLM이 정확하게 인식할 수 있는지를 테스트하기 위한 새로운 데이터셋을 도입하여 모델의 평가 능력을 검증합니다.

- **Performance Highlights**: 사전 훈련된 LLM은 논거 탐사 및 자동 에세이 점수 매기기와 같은 두 가지 표준 작업에서 성능이 더 적합하다는 평가를 받았습니다. 또한, Chain-of-Thought 프롬프트를 통해 라벨이 거의 없는 데이터 환경에서도 기준 성능을 초과하는 경향을 보였습니다. 문맥과 작업의 복잡성을 고려할 때, 이 연구는 법률 분석에 대한 LLM의 적용 가능성을 탐구함으로써 중요한 통찰력을 제공합니다.



### TelcoLM: collecting data, adapting, and benchmarking language models for the telecommunication domain (https://arxiv.org/abs/2412.15891)
Comments:
          30 pages (main: 13 pages, appendices: 17 pages), 1 figure, 22 tables, achieved March 2024, released December 2024

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)이 통신업계(telco domain)에 적응할 수 있는 방법을 연구하였다. 논문에서는 800M 토큰과 80K 지침으로 구성된 방대한 특정 도메인 데이터 집합을 수집하고, 다양한 방법론을 사용하여 적응 테스트를 진행했다. 결과적으로 도메인 적응 모델이 기존의 일반 모델과 경쟁할 수 있음을 밝혀냈으며, 기존의 복잡한 튜닝 과정을 줄일 수 있음을 제안하였다.

- **Technical Details**: 연구에서 사용된 기본 모델은 Llama-2-7B이며, 다양한 적응 접근법을 비교 분석하여 적응 과정에서 불필요한 사전 훈련(pretraining) 단계를 생략해도 뛰어난 성과를 거둘 수 있음을 보여주었다. 연구에서는 도메인 특화 데이터에 대한 재훈련(retraining)을 통해 모델의 성능을 향상시키는 DAPT 방법론을 강조했다. 또한, 제안된 프레임워크는 통신업계의 복잡한 용어와 개념을 효과적으로 처리하는데 중점을 두고 있다.

- **Performance Highlights**: 최종 실험 결과에 따르면, 도메인 적응 모델이 GPT-3.5와 경쟁하는 성능을 보여주었고, 이는 통신업계에서 LLM의 활용 가능성을 시사한다. 또한 연구는 적응 방법에서 일반 모델과 경쟁할 수 있는 성과를 도출하여, 제한된 리소스를 가지고도 효율적인 접근법을 제시할 수 있음을 입증하였다. 이러한 결과는 통신산업에서 LLM이 적용될 수 있는 여러 가지 유용한 사용 사례와 직접적인 관련성을 가지고 있다.



### Approximate State Abstraction for Markov Games (https://arxiv.org/abs/2412.15877)
- **What's New**: 이 논문은 두 플레이어 제로섬 마르코프 게임(TZMG)에서 상태 추상화(state abstraction)를 도입합니다. 플레이어의 보상은 환경을 나타내는 상태와 그들의 각각의 행동에 의해 결정됩니다. 특히 이러한 마르코프 게임의 특성을 살려, 예를 들어 축구와 같은 게임에서 행동의 가치는 플레이의 상태에 따라 달라진다는 점을 강조합니다.

- **Technical Details**: 저자들은 여러 개의 서로 다른 상태를 하나의 상태로 취급하여 상태의 수를 줄이는 상태 추상화 방법을 제안합니다. 이 방식은 상태 수가 증가함에 따라 균형 상태를 계산하는 것이 더 어려워지는 문제를 해결하려는 것입니다. 하지만 다수 플레이어 설정에서 상태 추상화가 적용된 게임의 균형 솔루션은 원래 게임의 균형 솔루션과 다를 수 있음을 주의해야 합니다.

- **Performance Highlights**: 상태 추상화의 효과를 검증하기 위해 저자들은 Markov Soccer를 사용하여 균형 정책(equilibrium policies)을 계산하고 결과를 분석했습니다. 이 연구는 상태 추상화가 다르게 얻어진 균형 솔루션의 근접성을 측정하기 위해 이중 간극(duality gap)의 경계를 도출하여 평가했습니다. 이 과정에서 상태 추상화가 게임의 결과에 미치는 영향을 깊이 있게 탐구하였습니다.



### AI-in-the-loop: The future of biomedical visual analytics applications in the era of AI (https://arxiv.org/abs/2412.15876)
Comments:
          Accepted for publication in IEEE Computer Graphics & Applications

- **What's New**: 이 논문은 AI가 데이터 분석 및 생물 의학 분야에서 비주얼 분석(visual analytics) 툴 개발에 미치는 영향을 논의합니다. 특히 Large Language Models(LLMs) 및 생성 모델이 어떻게 비주얼 도구와 워크플로우를 변화시킬 수 있는지에 대한 전망을 제공합니다. 기존의 'human-in-the-loop' 접근 방식 대신에 'AI-in-the-loop'를 강조하며, 인간 중심의 워크플로우를 유지하는 것이 중요하다고 주장합니다.

- **Technical Details**: 비주얼 분석 툴의 개발 과정은 데이터 수집 및 처리, 시각화 구성 요소 생성, 최종 비주얼 분석 툴의 완성 단계로 나뉘며, 이 모든 단계에서 AI와 딥러닝(deep learning) 기술이 점점 더 많이 활용되고 있습니다. 현재는 데이터 수집 단계에서 머신러닝(machine learning) 기술이 널리 이용되고 있으며, AI 시스템이 자동으로 시각화 코드 생성이나 데이터 분석 등의 작업을 수월하게 수행할 수 있도록 발전하고 있습니다. 이러한 변화는 기존의 경직된 시스템에 비해 더욱 유연한 상호작용이 가능하게 만듭니다.

- **Performance Highlights**: AI의 도입으로 인해 비주얼 분석 툴의 생성 과정이 혁신적으로 변화하고 있습니다. 예를 들어, 자연어 명령어를 사용하여 데이터 쿼리를 실행하고 복잡한 데이터 상호작용을 간소화할 수 있게 되었습니다. 이러한 기술들은 최종 사용자에게 유용한 통찰력을 제공합니다. 그러나, 사용자 경험을 더욱 향상시키기 위해 각 단계를 이어주는 높은 수준의 통합된 AI 솔루션 구축에는 여전히 도전 과제가 존재합니다.



### Traffic-Rule-Compliant Trajectory Repair via Satisfiability Modulo Theories and Reachability Analysis (https://arxiv.org/abs/2412.15837)
Comments:
          2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 연구에서는 자동화된 차량이 교통 규칙을 준수하는 데 도움을 주기 위해 새로운 접근 방식을 제안합니다. 전통적으로 계획된 경로가 교통 규칙을 위반할 경우, 새로운 경로를 처음부터 다시 계획하는 것이 일반적입니다. 제안된 방법은 경로 수정을 통해 계산 시간을 절약할 수 있도록 합니다.

- **Technical Details**: 연구진은 satisfiability modulo theories (SMT)와 집합 기반 도달 가능성 분석(set-based reachability analysis)을 결합하여 초기 경로를 어떻게 수정할 수 있는지를 판단합니다. 이 기술은 복잡한 환경에서도 규칙을 위반한 경로를 효율적으로 수리할 수 있도록 합니다.

- **Performance Highlights**: 실제와 높은 충실도의 시뮬레이터에서 진행된 실험 결과, 다양한 시나리오에서 제안된 접근 방식의 이점이 입증되었습니다. 복잡한 환경에서도 법적으로 안전한 작업을 신속하게 재개할 수 있는 능력이 매우 향상되었습니다.



### S$^2$DN: Learning to Denoise Unconvincing Knowledge for Inductive Knowledge Graph Completion (https://arxiv.org/abs/2412.15822)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 Inductive Knowledge Graph Completion(KGC)를 위한 새로운 S$^2$DN(Semantic Structure-aware Denoising Network) 네트워크를 제안합니다. 이 네트워크는 지식 그래프(KG) 내에서 새롭게 등장하는 개체들 간의 누락된 사실을 추론하는 과정을 개선하는 것을 목표로 합니다. 특히, S$^2$DN은 비슷한 관계의 의미적 불일치와 KG에서의 노이즈를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: S$^2$DN은 관계의 일반적인 의미를 유지하기 위해 포괄하는 서브그래프에 대한 의미적 스무딩 모듈을 도입합니다. 또한, 신뢰할 수 없는 상호작용을 필터링하고 추가적인 지식을 제공하기 위해 구조 정련 모듈을 통합하여 KG 내에서의 신뢰할 수 있는 상호작용을 유지합니다. 이러한 구조적인 정제 및 의미적 스무딩 접근 방식은 KG의 신뢰성과 일관성을 지속적으로 높이는 데 기여합니다.

- **Performance Highlights**: S$^2$DN은 다양한 KG와 서로 다른 노이즈 조건에서 GraIL 모델을 초월하는 예측 성능을 보였습니다. 경험적 실험을 통해 S$^2$DN은 KG 내에서 의미적 일관성을 유지하고 불확실한 상호작용을 효과적으로 필터링하는 능력을 입증하였습니다. 이러한 결과는 S$^2$DN이 KGC 분야에서 뛰어난 성과를 가지고 있음을 보여줍니다.



### $\pi$-yalli: un nouveau corpus pour le nahua (https://arxiv.org/abs/2412.15821)
Comments:
          9 pages, in French language, 2 figures

- **What's New**: NAHU$^2$ 프로젝트는 프랑스-멕시코 협력에 의해 기계 학습에 적합한 π-YALLI 코퍼스를 구축하여 나우아틀(Nahuatl) 언어의 컴퓨터 자원을 개발하는 것을 목표로 하고 있습니다. 나우아틀 언어는 200만 명 이상이 사용하는 생존 언어임에도 불구하고 컴퓨터 자원이 부족한 편입니다. 이 코퍼스는 자연어 처리(NLP) 도구 개발에 필수적인 다양한 언어 모델(Language Models, LM)을 연구하는 데 사용될 것입니다.

- **Technical Details**: π-YALLI 코퍼스는 역사적 문서, 위키피디아 문서, 뉴스, 시 및 이야기, 법률 문서 등 다양한 범주의 문서를 포함하고 있습니다. 수집된 데이터는 구조가 이질적이기 때문에 반자동 처리 과정을 거쳐 비관련 내용을 제거했습니다. 코퍼스에는 약 1.912M의 토큰(token)과 14.8M의 문자(character)가 포함되어 있으며, 이는 나우아틀 언어를 위한 기계 학습 자원을 제공하는 데 중요한 기반 역할을 할 것입니다.

- **Performance Highlights**: 본 연구에서는 Word2Vec와 FastText와 같은 정적 언어 모델을 초기 분류 모델로 사용하고, 이후 ALBERT와 같은 대형 언어 모델(LLM)로 발전시킬 예정입니다. 이러한 모델들은 나우아틀 언어의 의미와 구문 관련 정보를 효과적으로 캡처함으로써 고급 의미 이해와 감정 분석, 텍스트 자동 분류 및 범주화 등의 애플리케이션에 활용됩니다.



### WebLLM: A High-Performance In-Browser LLM Inference Engin (https://arxiv.org/abs/2412.15803)
- **What's New**: 최근의 대형 언어 모델(LLMs) 발전으로 인해 이전에는 상상할 수 없었던 능력들이 열렸습니다. 서버급 GPU를 요구하는 이러한 모델을 클라우드에 호스팅하는 대신, 최근에는 더 작고 오픈 소스 모델들이 등장하면서 개인 장치에서의 배포가 가능해졌습니다. 이에 따라 WebLLM이라는 새로운 오픈 소스 JavaScript 프레임워크가 소개되어 웹 브라우저 내에서 LLM 추론을 수행할 수 있게 되었습니다.

- **Technical Details**: WebLLM은 웹 응용 프로그램에 LLM 기반 기능을 통합할 수 있도록 돕는 JavaScript 프레임워크입니다. 이 시스템은 사용자 친화적인 엔진 ServiceWorkerMLCEngine, 웹 워커 내에 위치한 MLCEngine, 그리고 사전 컴파일된 효율적인 WebGPU 커널로 구성됩니다. WebGPU와 WebAssembly를 활용해 GPU 가속과 CPU 계산을 효율적으로 수행하며, 백엔드 실행을 웹 워커로 분리하여 UI 흐름에 방해가 없도록 설계되었습니다.

- **Performance Highlights**: 실제 평가 결과 WebLLM은 MLC-LLM에 비해 최대 80%의 성능을 유지하며, 이는 동일한 장치에서의 성능을 기준으로 합니다. 또한, WebGPU의 최신 기능과 WebLLM의 런타임 최적화를 통해 성능 격차를 더욱 줄일 수 있는 가능성도 존재합니다. 최종적으로 WebLLM은 웹 브라우저 내에서 개인화된 LLM 애플리케이션을 구현할 수 있는 길을 열었습니다.



### Bi-directional Mapping of Morphology Metrics and 3D City Blocks for Enhanced Characterization and Generation of Urban Form (https://arxiv.org/abs/2412.15801)
- **What's New**: 이번 연구는 도시의 형태학(morphology metrics)과 복잡한 도시 형태(complex urban form) 간의 이중 맵핑을 설정하여 도시 형태 생성(urban form generation)과 성능 평가(performance evaluation)의 통합을 가능하게 한다는 점에서 중요하다. 제안하는 접근법은 도시 형태를 특성화하고 유사한 3D 도시 형태를 검색하는 데 사용할 수 있는 형태학 지표(morphology metrics)를 형성하는 방법을 제시한다.

- **Technical Details**: 연구에서는 뉴욕시의 14,248 블록을 커버하는 3D 도시 모델을 사용하여 형태학 지표를 형성, 도시 형태의 클러스터링(cluster) 및 형태학 지표 평가를 수행한다. 이 과정에서 신경망(neural networks)과 정보 검색(information retrieval)을 이용하여 효과적으로 도시 형태를 분석할 수 있는 지표 집합을 식별하였다.

- **Performance Highlights**: 연구의 결과, 제안된 방법론은 복잡한 도시 형태와 형태학 지표를 밀접하게 결합하여 성능 중심의 도시 디자인(performance-driven urban design)에서 지속 가능한 도시 설계(sustainable urban design) 및 계획(planning)을 위한 매끄럽고 이중적인 관계를 가능하게 한다. 이는 최적화된 형태학 지표를 통해 개선된 도시 형태와 향상된 도시 성능을 생성할 수 있는 새로운 경로를 제공한다.



### GraphSeqLM: A Unified Graph Language Framework for Omic Graph Learning (https://arxiv.org/abs/2412.15790)
- **What's New**: 이번 연구는 Graph Neural Networks (GNNs)와 Large Language Models (LLMs)를 통합하여 새로운 Graph Sequence Language Model (GraphSeqLM)을 제안합니다. GraphSeqLM은 DNA, RNA 및 단백질 서열을 인코딩하는 생물학적 서열 임베딩을 통해 GNN의 성능을 향상시키며, 복잡한 생물학적 관계를 잘 포착할 수 있도록 설계되었습니다. 이 방법은 의학에 있어 정밀한 다중 오믹스 데이터 통합을 가능하게 합니다.

- **Technical Details**: 제안된 방법론은 DNA, RNA 및 단백질 서열 데이터를 다중 오믹스 데이터 세트와 통합하여 복잡한 생물학적 네트워크의 표현을 강화합니다. 기존의 유전자 조절 네트워크 및 KEGG(킨네시사와 고토, 2000)와 결합하여, 각각의 기능 세트를 포함하는 지식 그래프를 생성합니다. 이 지식 그래프는 두 개의 하위 그래프로 분해되어 내부 신호 전달 과정과 단백질-단백질 상호작용(PPI)을 효과적으로 모델링합니다.

- **Performance Highlights**: GraphSeqLM은 종합 평가에서 기존 방법들을 능가하는 예측 정확도를 보여주었습니다. 고유의 생물학적 속성 및 구조를 인코딩하여 다중 오믹스 데이터의 복잡한 패턴을 파악하는 데 있어서 뛰어난 성능을 발휘합니다. 이 연구는 정밀 의학에서 다중 오믹스 데이터 통합을 위한 더 나은 접근 방식을 제시하며, 생물학적 시스템 모델링의 새로운 가능성을 열어줍니다.



### Linguistic Features Extracted by GPT-4 Improve Alzheimer's Disease Detection based on Spontaneous Speech (https://arxiv.org/abs/2412.15772)
Comments:
          Accepted at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이번 연구에서는 GPT-4를 활용해 자발적 환자 발화의 전사본에서 다섯 가지 의미적 특징을 추출하였습니다. 이는 알츠하이머병(AD)의 증상을 포착하지만 전통적인 방법으로는 효과적으로 정량화하기 어려운 특징입니다. 저자들은 이러한 특징의 임상적 중요성을 보여주며, 기존 언어적 특징 및 Random Forest 분류기와 함께 결합했을 때 AD 탐지의 성능을 크게 향상시킬 수 있음을 입증합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 ADReSS로, 156명의 참가자가 Cookie Theft 그림을 설명하는 음성 녹음으로 구성되어 있습니다. 이 데이터셋은 진단, 나이 및 성별에 따라 균형이 잡혀 있으며, CHAT 주석 형식의 수동 전사가 포함되어 있습니다. 연구자는 수동 전사와 자동 음성 인식(ASR)의 결과를 비교하여 전혀 새로운 AD 탐지 파이프라인의 효과성을 평가하였습니다.

- **Performance Highlights**: GPT에서 파생된 특징들은 기존의 언어적 특징들과 조합할 경우 AD 탐지 성능을 크게 향상시켰습니다. 특히, 인간 평가자 및 대리 측정과의 비교를 통해 '단어 찾기 어려움'이라는 특징이 높은 유의미성을 가지고 있음을 입증하였습니다. 이 접근 방식은 수동 전사와 자동 생성 전사 모두에 대해 효과적임을 보여주며, LLM의 최근 발전을 활용한 혁신적이고 영향력 있는 사용 사례로 평가됩니다.



### Critique of Impure Reason: Unveiling the reasoning behaviour of medical Large Language Models (https://arxiv.org/abs/2412.15748)
Comments:
          16 pages, 5 figures, 2 tables. Conceptualization, both authors. formal analysis, both authors. funding acquisition, both authors. investigation, both authors. resources, both authors. supervision, T.C.. validation, both authors. visualization, both authors. writing original draft, both authors. writing review and editing, both authors

- **What's New**: 본 논문에서는 최근 의료 분야에서 사용되는 Large Language Models (LLMs)의 추론 행동(reasoning behaviour)을 탐구하고, 이러한 모델의 높은 수준의 예측 정확도보다도 추론 행동에 대한 이해의 필요성을 강조합니다. 의료 AI의 Explainable AI (XAI)를 달성하기 위해 이러한 모델이 어떻게 결론에 도달하는지에 대한 통찰력을 제공하는 이론적 틀을 제안합니다. 이를 통해 의료 전문가들이 LLM의 내부 작동 방식을 이해하고 잠재적인 논리적 오류를 드러내는 데 도움을 줄 수 있습니다.

- **Technical Details**: 논문은 LLM의 추론 행동을 정의하고, 현재의 의료 LLM에서의 고급 성능 지표와 함께 추론 행동을 평가하는 방법들을 분류합니다. 특히, 논리적 추론과 인과적 추론을 포함한 다양한 유형의 추론이 있습니다. 또한, Neuro-Symbolic AI (N-SAI)라는 분야를 통해 신경망과 기호적 추론 기술을 통합하는 방법을 논의하며, 이러한 접근 방식이 추론을 더욱 투명하게 만들어주는 데 기여할 수 있음을 설명합니다.

- **Performance Highlights**: 의료 LLM의 추론 행동을 이해함으로써 사용자는 이러한 모델이 어떻게 결론에 도달했는지를 확인할 수 있으며, 이는 임상 의사 결정 과정에서 신뢰를 구축하는 데 기여합니다. LLM이 환자 진단 및 치료 제안에서의 통찰력을 제공할 경우, 이는 의사와 기계의 권고사항 간의 불일치를 명확히 할 수 있습니다. 궁극적으로 이러한 투명성은 의료 분야에 AI를 통합하고 환자 결과를 개선하는 데 중요한 역할을 할 것입니다.



### fluke: Federated Learning Utility frameworK for Experimentation and research (https://arxiv.org/abs/2412.15728)
Comments:
          Accepted at FLUID workshop (AAAI 2025) [4 pages (+2 references), 2 figures, 1 algorithm]

- **What's New**: 이 논문에서는 새로운 파이썬 패키지인 fluke를 소개합니다. fluke는 Federated Learning(FL) 알고리즘을 효율적으로 개발하고 프로토타입 할 수 있도록 설계되었습니다. 이 패키지는 연구자나 전문가가 알고리즘의 교육 요소에 집중할 수 있도록 지원하는데 중점을 두고 있습니다.

- **Technical Details**: fluke는 오픈소스 파이썬 패키지로, 사용자 친화적인 설계를 가지고 있어 알고리즘을 쉽게 추가하고 사용할 수 있습니다. 사용자가 알고리즘의 세부사항에 대해 걱정하지 않고도 새로운 FL 알고리즘을 신속하게 프로토타입 할 수 있도록 도와줍니다. 패키지는 중앙 집중형 아키텍처를 가정하며, 클라이언트와 서버 간의 통신을 시뮬레이션하여 FL 환경을 구현합니다.

- **Performance Highlights**: fluke는 현재 몇 가지 최신 FL 알고리즘과 데이터 세트를 포함하고 있으며, 정기적으로 업데이트 되어 최신 기술을 반영합니다. 패키지의 CLI(커맨드라인 인터페이스)는 FL 실험을 실행하는 가장 쉬운 방법을 제공하며, 세 가지 실험 유형을 지원합니다. fluke는 연구자들이 새 알고리즘을 빠르게 프로토타입 할 수 있도록 하기 위한 유틸리티 프레임워크로, FL 분야의 발전에 기여할 것으로 기대됩니다.



### Towards Secure AI-driven Industrial Metaverse with NFT Digital Twins (https://arxiv.org/abs/2412.15716)
- **What's New**: 이번 연구는 NFT-DT(Non-Fungible Token Digital Twins)의 보안을 강화하기 위한 심층 학습 기반의 새로운 접근 방식을 제안합니다. 전통적인 메타데이터 기반의 위조 탐지 방법 대신, 오토인코더와 RNN(Recurrent Neural Network) 기반 분류기를 결합하여 실시간으로 위조 NFT-DT를 인식할 수 있는 능력을 갖추었습니다. 또한, 동적 메타데이터 개념을 도입하여 AI 통합 스마트 계약을 통해 진위 검증의 신뢰성을 높이고 있습니다.

- **Technical Details**: 제안된 시스템의 두 가지 주요 부분은 패턴 인코딩과 분류로 나뉘며, 모두 심층 학습 모델을 사용하여 해결됩니다. 파라미터 최적화를 위해 Python 라이브러리 keras-tuner를 활용하였습니다. NFT-DT의 행위 패턴 인식을 위해, 입력 데이터를 압축하고 재구성하는 인공지능 신경망인 오토인코더를 사용하여 복잡한 데이터 세트를 효율적으로 처리합니다.

- **Performance Highlights**: 본 연구는 NFT 기반 자산의 메타버스에서의 보안을 강화하는 데 기여하며, 실험 결과 위조 디지털 트윈의 성공적인 탐지를 통해 제안된 모델의 효과성을 입증하였습니다. 이를 통해 제조업 및 산업 분야에서의 디지털 트윈 활용도가 더욱 높아질 것으로 기대됩니다.



### MacLight: Multi-scene Aggregation Convolutional Learning for Traffic Signal Contro (https://arxiv.org/abs/2412.15703)
Comments:
          Accepted as full paper by AAMAS2025

- **What's New**: 이 논문에서는 기존의 교통 신호 제어(Traffic Signal Control, TSC) 방법론의 한계를 극복하기 위해 Multi-Scene Aggregation Convolutional Learning (MacLight)을 제안합니다. MacLight는 기존의 그래프 기반 접근방식보다 빠른 훈련 속도와 안정적인 성능을 제공하며, 프로시멀 정책 최적화(Proximal Policy Optimization, PPO) 알고리즘을 백본으로 사용하여 국소적 특징과 글로벌 임베딩 표현을 고려합니다. 이를 통해 과적합(overfitting) 문제를 줄이고 정책 업데이트의 안정성을 보장합니다.

- **Technical Details**: MacLight는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째 구성 요소는 전역 표현(global representation)으로, 변분 오토인코더(Variational Autoencoder, VAE)를 활용하여 글로벌 상태의 압축 표현을 생성합니다. 두 번째 구성 요소는 PPO 알고리즘으로, 가치 평가를 위한 모듈과 정책 개선을 위한 모듈을 통해 서로 다른 정보를 처리합니다. 이러한 설계를 통해 시간 오버헤드를 최소화하고 전체적인 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과는 MacLight가 기존의 일반 및 도메인 SOTA(상태-of-the-art) 방법과 비교하여 뛰어난 안정성과 최적화된 수렴 수준을 달성했음을 보여줍니다. 긴급한 교통 사건에 따른 동적 교통 흐름을 시뮬레이션할 수 있는 환경에서도 높은 시간 효율성을 유지합니다. 코드와 구현은 제공된 링크에서 확인 가능하며, 다양한 교통 시나리오에서 검증된 성능을 기반으로 향후 연구를 위한 기초를 마련했다고 할 수 있습니다.



### AI-generated Image Quality Assessment in Visual Communication (https://arxiv.org/abs/2412.15677)
Comments:
          AAAI-2025; Project page: this https URL

- **What's New**: 이 논문은 AIGI-VC라는 새로운 데이터세트를 제안하여 인공지능이 생성한 이미지(AIGIs)의 품질을 평가하기 위한 기준을 설정합니다. 기존 IQA 기법의 한계점을 극복하기 위해 광고 분야에서 AIGIs의 의사소통 능력을 정보의 명확성과 감정적 상호작용이라는 두 관점에서 분석합니다. AIGI-VC 데이터세트는 14개의 광고 주제와 8가지 감정 유형을 포함한 2,500장의 이미지를 제공하여 향상된 인간 선호도 주석을 수집합니다.

- **Technical Details**: AIGI-VC 데이터세트는 주관적 실험을 통해 정보의 명확성과 감정적 상호작용의 두 가지 평가 차원에 대해 이미지 쌍을 비교한 후, 일반적인 선호와 세부적인 선호 설명을 수집합니다. 이 데이터세트는 다양한 IQA 메트릭의 성능을 벤치마크하고, 기존의 다중 모드 모델(LMM)을 활용하여 AIGIs 평가의 효과성을 검증합니다. 특히, AIGIs의 인간-대상 상호작용, 환상적 콘텐츠, 긍정적/부정적 감정을 유도하는 세 가지 하위 집합에 대한 실험을 진행합니다.

- **Performance Highlights**: 이 연구에서는 최신 IQA 기법과 LMM들이 AIGIs의 검토에 있어 한계가 있음을 발견했습니다. AIGI-VC는 이러한 기법들이 효율적으로 작동하지 않음을 드러내며, AIGIs에 대한 실질적 평가의 필요성을 강조합니다. 연구 팀은 AIGI-VC의 기여를 통해 시각적 커뮤니케이션 응용 프로그램에서의 인공지능 이미지의 효과성을 높이고자 합니다.



### MathSpeech: Leveraging Small LMs for Accurate Conversion in Mathematical Speech-to-Formula (https://arxiv.org/abs/2412.15655)
Comments:
          Accepted in AAAI 2025

- **What's New**: 이 논문은 MathSpeech라는 새로운 파이프라인을 소개하고 있습니다. 이 시스템은 수학적인 표현을 구술한 내용을 정확하게 LaTeX 형식으로 변환하는 데 중점을 두고 있습니다. 기존의 자동 음성 인식 (ASR) 모델이 수학 공식 처리에 취약한 문제점을 해결하기 위해 소형 언어 모델 (sLMs)을 통합하였습니다. MathSpeech는 다양한 ASR 모델의 성능을 평가하기 위한 새로운 데이터셋을 기반으로 개발되었습니다.

- **Technical Details**: MathSpeech는 1,101개의 실제 강의 오디오 샘플을 포함하는 벤치마크 데이터셋을 활용하여 ASR 모델의 성능을 시험하는 방식을 채택하고 있습니다. 이 파이프라인은 ASR의 오류를 교정하고, 이를 소형 언어 모델을 통해 LaTeX로 변환하는 작업을 수행합니다. 특히, 120M 파라미터를 가진 소형 언어 모델로도 상업적인 대형 언어 모델(GPT-4o)보다 뛰어난 LaTeX 생성 능력을 보여줍니다. CER, BLEU 및 ROUGE 점수에서 MathSpeech는 GPT-4o에 비해 유의미한 성능 향상을 보여주었습니다.

- **Performance Highlights**: MathSpeech는 수학적 표현을 처리하는 데 있어 기존의 ASR 모델보다 월등한 성능을 보였습니다. 특히 CER 점수가 0.390에서 0.298로 감소하였고, ROUGE 및 BLEU 점수에서도 좋은 결과를 기록하였습니다. 이러한 점수는 MathSpeech가 많은 양의 파라미터를 사용하는 대형 모델에 비해 효율적인 성과를 달성했다는 것을 의미합니다. 수학 교육에 있어서 학습자들의 이해를 돕기 위한 실질적인 기여를 할 것으로 기대됩니다.



### Tacit Learning with Adaptive Information Selection for Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2412.15639)
Comments:
          Accepted by AAMAS 2025 (Extended Abstract)

- **What's New**: 본 논문은 Multi-Agent Reinforcement Learning (MARL)에서 Centralized Training with Decentralized Execution (CTDE) 프레임워크의 발전을 위한 새로운 협력적 MARL 프레임워크인 Selective Implicit Collaboration Algorithm (SICA)를 소개합니다. 기존 CTDE 방법의 두 가지 주요 문제를 해결하기 위해, 에이전트는 로컬 정보를 기반으로 의사결정을 수행하고, 진정한 정보를 공유하기 위해 상호 작용을 통해 암묵적인 협력을 점진적으로 발전시킵니다. 이를 통해 에이전트는 복잡한 협력 전략을 학습할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: SICA는 QMIX 프레임워크를 기반으로 하여 구성된 세 가지 주요 블록: Selection Block, Communication Block, Regeneration Block을 가지고 있습니다. Selection Block에서는 협력에 필요한 정보를 필터링하고, Communication Block을 통해 다른 에이전트들에게 공유하여 진정한 정보를 생성합니다. Regeneration Block은 로컬 정보를 활용하여 진정한 정보를 재생성하며, 훈련을 진행하면서 중앙 집중형에서 분산형 프레임워크로 점진적으로 전환합니다.

- **Performance Highlights**: SICA는 StarCraft Multi-Agent Challenge(SMAC) 및 Google Research Football에서의 성능 평가를 통해 기존의 CTDE 방법과 명시적 커뮤니케이션 방법보다 우수한 결과를 기록했습니다. SICA의 접근 방식은 정보의 적응적 선택을 가능하게 하여 의사결정에서 정보를 더욱 효과적으로 활용하게 하며, 지연 문제를 피할 수 있게 해줍니다. 이러한 성과는 SICA가 다양한 알고리즘과 쉽게 통합되고, 실제 환경에서도 유용하게 적용될 수 있음을 보여줍니다.



### JailPO: A Novel Black-box Jailbreak Framework via Preference Optimization against Aligned LLMs (https://arxiv.org/abs/2412.15623)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문은 JailPO라는 새로운 블랙박스(jailbreak) 프레임워크를 제안하여, 기존의 수작업 템플릿 기반(jailbreak) 방법의 한계를 극복하고 LLM의 취약성을 탐색합니다. JailPO는 효율적이고 보편적인 covert jailbreak 프롬프트를 자동 생성할 수 있도록 훈련된 공격 모델을 사용하여, LLM 사용 시의 확장성과 효율성을 향상시키고 있습니다. 이를 통해 학습한 공격 모델은 효과적인 결과를 도출하는 독립적인 질문을 생성하며, 전체 공격 과정의 자동화를 가능하게 합니다.

- **Technical Details**: JailPO는 세 가지 주요 구성 요소로 이루어져 있습니다: 핵심 최적화 알고리즘, 두 개의 공격 모델 구성, 그리고 세 가지 다양한 jailbreak 패턴입니다. 이 프레임워크는 LLM이 생성한 프롬프트와 고해상도의 스코어링 메커니즘을 활용하여, 보다 효과적으로 공격 모델을 훈련시킵니다. 특히, 우리가 도입한 선호 최적화(preference optimization) 방식은 효율성을 높이는 것뿐만 아니라, LLM에 대한 공격의 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과 JailPO는 공격의 효율성, 범용성(universality), 그리고 방어에 대한 견고함에서 기존 방식들과 비교하여 뛰어난 성능을 보여주었습니다. 또한, 복잡한 템플릿을 기반으로 한 공격이 모델의 정렬을 더 쉽게 우회할 수 있는 반면, covert 질문 변환을 통한 공격은 리스크가 높은 응답을 유도하여 방어 기제를 피하는 데 더 효과적임을 입증했습니다. 기존의 오픈소스 및 상용 LLM에 대해 성능을 평가한 결과, JailPO는 효과적인 공격을 자동으로 수행할 수 있는 능력을 지니고 있습니다.



### Modeling Autonomous Shifts Between Focus State and Mind-Wandering Using a Predictive-Coding-Inspired Variational RNN Mod (https://arxiv.org/abs/2412.15620)
- **What's New**: 이번 연구에서는 자유 에너지 원리(free energy principle)에 기반한 변형 RNN 모델을 사용하여 집중 상태(focused state)와 마음 방황(mind-wandering) 간의 자율적 전환 일어나는 신경 메커니즘을 조사합니다. 연구진은 메타 레벨 매개변수(meta-prior)인 \mathbf{w}를 도입하여 이러한 전환 현상을 규명하였으며, \mathbf{w}의 값이 감소하거나 증가함에 따라 재구성 오류(reconstruction error)의 평균 변화가 일어나는 것을 확인했습니다. 두 상태 사이의 전환은 집중된 지각과 마음 방황 간의 균형을 이루는 데 중점을 둡니다.

- **Technical Details**: 본 연구는 집중 상태(FS)와 마음 방황(MW) 사이의 자율적 변화 과정을 순차 sensory input 패턴 및 예측 코딩(predictive coding) 프레임워크에 따라 모델링합니다. 예측 코딩은 감각 시퀀스를 예측하는 생성 모델(generative model)을 가정하며, 현재의 잠재(state)를 지속적인 감각 시퀀스 관찰을 통해 추론합니다. 연구진은 FS가 하향 확률 추론(top-down inference)을 통해 강화되며, MW가 상향 감각 패턴 생성(bottom-up sensory pattern generation)을 강조함으로써 발생한다고 가정하고 있습니다.

- **Performance Highlights**: 시뮬레이션 실험 결과, 메타 인식 상태가 낮은 값으로 변화하면 주의가 분산되어 마음 방황 상태가 발생하며, 각 상태 간의 변환은 평균 재구성 오류에 따라 자율적으로 조정됨을 나타냅니다. 높은 \mathbf{w} 값은 상향 예측을 우선시하고 낮은 \mathbf{w} 값은 하향 감각을 강조하여, 이 두 가지가 FS와 MW 사이의 전환을 이해하는 데 중요한 역할을 한다고 결론지었습니다. 이러한 결과는 향후 연구에서 이러한 신경 메커니즘을 탐구하는 데 기초가 될 수 있습니다.



### Microservices-Based Framework for Predictive Analytics and Real-time Performance Enhancement in Travel Reservation Systems (https://arxiv.org/abs/2412.15616)
Comments:
          10 Pages, 05 figures

- **What's New**: 이 논문은 예측 분석(predictive analytics)의 힘을 활용하여 실시간 여행 예약 시스템의 성능을 향상시키기 위한 마이크로서비스 기반 아키텍처(framework)를 제시합니다. 기존의 모놀리식(monoithic) 시스템은 부하가 많은 상황에서 확장성과 성능이 떨어지며, 이로 인해 자원의 미활용과 지연이 발생합니다. 이를 해결하기 위해 시스템 구성 요소를 독립적인 서비스로 분리하여 수요에 따라 성장하거나 축소할 수 있는 모듈화(modularization) 접근 방식을 채택하였습니다.

- **Technical Details**: 본 프레임워크는 머신러닝(machine learning) 모델을 통해 고객 수요 예측(forecasting), 동적 가격(dynamic pricing), 시스템 성능 최적화(real-time predictive analytics)를 포함합니다. 실험 평가(experimental evaluation)를 통해 이 프레임워크가 응답 시간(response time), 처리량(throughput), 성공 거래율(transaction rate of success), 예측 정확도(prediction accuracy)와 같은 성능 지표에 미치는 영향을 보여주었습니다. 마이크로서비스 접근 방식은 일반적인 아키텍처처럼 확장성(scalability)과 내결함성(fault tolerance)을 향상시키는 것 외에도, 적시성(timeliness)과 정확성(accuracy)을 갖춘 예측을 제공하여 고객 만족과 운영 효율성을 증가시킵니다.

- **Performance Highlights**: 실시간 분석(real-time analytics)의 통합은 더 지능적인 의사결정(decision-making)을 이끌어내어 시스템 응답(response) 및 신뢰성(reliability)을 개선합니다. 이 시스템은 현대의 여행 예약 시스템이 직면한 과제를 해결하기 위해 스케일 가능한(scalable) 효율적인(framework) 아키텍처를 제공합니다. 향후 연구는 성능과 견고성(robustness)을 더욱 향상시키기 위해 고급 AI 모델과 엣지 처리(edge processing)의 조사를 진행할 계획입니다.



### A Fusion Approach of Dependency Syntax and Sentiment Polarity for Feature Label Extraction in Commodity Reviews (https://arxiv.org/abs/2412.15610)
- **What's New**: 이 연구는 모바일폰, 컴퓨터, 화장품, 음식의 4개 카테고리에서 13,218개의 제품 리뷰를 분석하며, 의존 구문 분석과 감정 극성 분석을 통합하여 새로운 특징 레이블 추출 방법을 제안합니다. 기존의 추출 알고리즘의 낮은 강건성을 해결하고, 추출 정확도를 크게 향상시킵니다. 실험 결과는 이 방법이 0.7의 정확도와 0.8의 재현율 및 F-score를 달성했음을 보여줍니다. 그러나 매칭 사전에 대한 의존성과 추출된 특징 태그의 제한된 범위는 향후 연구에서 더 조사해야 할 과제로 남아 있습니다.

- **Technical Details**: 이 연구는 의존 구문 분석(dependency syntax) 이론을 바탕으로 제품 리뷰에서 '제품 특징-평가 용어' 쌍을 추출하는 것을 목표로 합니다. 특징 태그에는 제품 특징, 감정 정도, 평가 용어라는 세 가지 구성 요소가 포함됩니다. 해당 요소들은 의존 관계와 감정 분석을 통해 연결되며, 제품 특징과 평가 용어 간의 감정 극성에 따라 부정어 및 정도 부사를 식별하고 추출합니다. 이러한 새로운 접근법은 기존의 방법들이 지닌 모호성을 해결하는데 효과적입니다.

- **Performance Highlights**: 제안된 방법은 실험에서 0.7의 정확도와 함께 0.8의 재현율 및 F-score를 기록하며, 제품 리뷰에서 감정 단어와 평가 객체 식별의 정확성과 강건성을 향상시키는 데 성공하였습니다. 또한, 이 방법은 리뷰 내용의 정교한 분석이 가능하여 사용자에게 더 유용한 정보를 제공할 수 있습니다. 그러나 여전히 단어의 중복성 문제 및 특정 감정 단어의 낮은 극성으로 인해 발생하는 문제들은 해결해야 할 과제로 남아있습니다.



### SODor: Long-Term EEG Partitioning for Seizure Onset Detection (https://arxiv.org/abs/2412.15598)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이번 연구에서는 발작 시작 감지를 명시적으로 모델링하는 새로운 두 단계 프레임워크인 SODor를 제안합니다. 기존 방법들은 EEG 신호에서 발작을 분류하는 데 중점을 두었지만, 발작 시작 (Seizure Onset) 감지를 직접적으로 다루지 못했습니다. SODor는 서브시퀀스 클러스터링(subsequence clustering) 작업으로 발작 시작 감지 문제를 모델링하여, 발작 상태를 자동으로 식별하고 세분화합니다.

- **Technical Details**: SODor는 EEG 데이터를 활용하여, 두 단계로 진행되는 서브시퀀스 클러스터링으로 발작 시작 시점이 발생하는 지점을 명확히 식별합니다. 첫 번째 단계는 레이블 감독을 통해 2차 임베딩 세트를 학습하고, 두 번째 단계에서는 모델 기반 클러스터링을 사용하여 장기적인 시간 종속성을 캡처합니다. 이 방법은 클러스터 혹은 상태 전이를 통해 발작 시작을 탐지할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: 저자들은 세 가지 데이터셋에서 SODor를 평가한 결과, 다른 기초 모델들에 비해 5%-11% 향상된 분류 성능을 달성했으며 발작 시작 감지의 정확성 또한 크게 개선된 것을 보였습니다. 이러한 성과는 EEG 신호 내에서 의미 있는 서브시퀀스를 추출하고 이를 통해 발작 시작을 효과적으로 감지할 수 있음을 보여줍니다.



### Mask-RadarNet: Enhancing Transformer With Spatial-Temporal Semantic Context for Radar Object Detection in Autonomous Driving (https://arxiv.org/abs/2412.15595)
- **What's New**: 본 논문에서는 Mask-RadarNet이라는 새로운 모델을 제안합니다. 이 모델은 레이더 데이터의 계층적 의미 기능을 최대한 활용하기 위해 설계되었습니다. 기존의 컨볼루션 신경망(CNN)에 의존하지 않고 공간-시간 의미 맥락을 효과적으로 캡처하는 데 중점을 두고 있습니다.

- **Technical Details**: Mask-RadarNet은 간섭적인 컨볼루션과 주의(attention) 연산을 결합하여 전통적인 트랜스포머 아키텍처를 대체합니다. 또한 패치 이동(patch shift) 방법을 도입하여 효율적인 공간-시간 기능 학습을 수행합니다. 이를 통해 높은 인식 정확도를 유지하면서도 계산 부담을 줄이는 성과를 보입니다.

- **Performance Highlights**: CRUW 데이터셋에서의 실험 결과는 Mask-RadarNet이 기존의 레이더 기반 객체 탐지 알고리즘에 비해 우수한 성능을 보여줍니다. 특히 이해가 어려운 RF 이미지에도 불구하고, 제안된 모델은 더 낮은 계산 복잡성과 적은 파라미터로 높은 인식 정확도를 달성합니다.



### Machine Learning Techniques for Pattern Recognition in High-Dimensional Data Mining (https://arxiv.org/abs/2412.15593)
- **What's New**: 이 논문은 지원 벡터 머신(Support Vector Machine, SVM)을 기반으로 한 빈번한 패턴 데이터 마이닝 알고리즘을 제안합니다. 기존의 빈번한 패턴 마이닝 알고리즘이 고차원 및 희소 데이터 환경에서 나타나는 성능 병목 현상을 해결하는 것을 목표로 합니다. 빈번한 패턴 마이닝 작업을 분류 문제로 전환하며, SVM 모델을 도입해 패턴 추출의 정확성과 강건성을 향상시키고 있습니다.

- **Technical Details**: 방법 설계 측면에서 커널 함수(Kernel Function)를 사용하여 데이터를 고차원 특징 공간(High-dimensional Feature Space)으로 매핑합니다. 최적의 분류 하이퍼플레인(Optimal Classification Hyperplane)을 구성하여 비선형 패턴을 분리하고, 빈번한 항목(Frequent Items)을 정확하게 마이닝합니다. 실험에서는 Retail과 Mushroom이라는 두 개의 공공 데이터셋을 선택하여 제안된 알고리즘과 기존의 FP-Growth, FP-Tree, 결정 트리(Decision Tree), 랜덤 포레스트(Random Forest) 모델을 비교 분석하였습니다.

- **Performance Highlights**: 실험 결과, 이 논문에서 제안한 알고리즘은 지원(Support), 신뢰도(Confidence), 리프트(Lift)라는 세 가지 주요 지표에서 기존 모델에 비해 현저히 우수한 성능을 보였습니다. 이는 강력한 패턴 인식 능력과 규칙 추출 효과를 나타냅니다. SVM 모델이 데이터 희소성이 높은 환경과 많은 거래가 있는 상황에서 탁월한 성능 이점을 가지고 있으며, 복잡한 패턴 마이닝 작업을 효과적으로 처리할 수 있음을 보여줍니다.



### Pre-training Graph Neural Networks on Molecules by Using Subgraph-Conditioned Graph Information Bottleneck (https://arxiv.org/abs/2412.15589)
Comments:
          15 pages

- **What's New**: 이번 연구는 인간 주석이나 사전 지식 없이 분자에 대한 사전 훈련된 그래프 신경망(Graph Neural Network, GNN) 모델을 구축하는 것을 목표로 하고 있습니다. 기존의 사전 훈련 방법들은 기능 그룹과 같은 의미적 부분 그래프에 의존하고 있어 그래프 수준의 차별성을 간과할 수 있는 한계가 있습니다. 본 연구에서 제안하는 S-CGIB는 핵심 부분 그래프(그래프 코어)를 인식하고 중요한 부분 그래프를 자동으로 탐색하여 이러한 한계를 극복합니다.

- **Technical Details**: S-CGIB는 입력 그래프를 특정 중요한 부분 그래프에 따라 압축하여 그래프 코어로 변환하는 방식으로, 노드들 간의 상호작용이 주의(attention)에 기반하여 이루어집니다. 이 방법은 기능 그룹 후보들(ego networks)을 생성하고, 그래프 코어와 이 후보들 간의 상호작용을 통해 중요한 부분 그래프를 식별합니다. 결과적으로, 기능 그룹에 대한 사전 지식 없이도 분자에 대한 강인한 표현을 생성할 수 있습니다.

- **Performance Highlights**: 다양한 분자 데이터셋에서 수행된 광범위한 실험 결과, S-CGIB는 기존의 방법들보다 우수한 성능을 보였습니다. 이는 성과적으로 그래프 수준 표현을 잘 구분짓는 데에 기여하며, 기능 그룹을 보다 정확하게 인식할 수 있도록 합니다. 연구 결과는 GNN이 분자의 화학적 성질과 구조를 이해하는 데 필수적인 도구로 자리 잡을 수 있음을 시사합니다.



### Score-based Generative Diffusion Models for Social Recommendations (https://arxiv.org/abs/2412.15579)
Comments:
          14 pages, 8 figures

- **What's New**: 이 연구에서는 소셜 추천 시스템의 성능 향상을 위해 혁신적인 생성적 관점을 통해 낮은 사회적 동질성 문제를 해결합니다. 새로운 Score-based Generative Model for Social Recommendation (SGSR)을 제안하며, 이는 Stochastic Differential Equation(SDE) 기반의 확산 모델을 소셜 추천에 적절하게 조정합니다. 이 연구의 주요 초점은 협업 신호와의 일관성을 극대화하는 사용자 소셜 표현을 생성하는 것입니다.

- **Technical Details**: SGSR은 사용자-아이템 행동에 기반하여 개인화된 노이즈 제거 확산 프로세스를 달성하는 새로운 클래스의 비지도 학습 기법을 활용합니다. 이를 위해, 우리는 스코어 기반 생성 모델(SGMs)을 도입하고, 이 모델의 특성과 사회적 추천에 적합한 형식으로 조정된 분류기 없는 조건부 목표를 도출합니다. 이는 사용자의 다양한 사회적 관계를 효과적으로 모델링하기 위해 서로 추가적인 레벨에서 협업 신호를 활용하여 최적의 소셜 표현을 식별할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, 제안된 SGSR 프레임워크는 세 가지 실세계 데이터 세트에서 수행된 포괄적인 비교 실험 및 제거 연구에서 기존의 최첨단 방법들보다 유의미하게 우수한 성능을 보였습니다. 이는 제안된 방법이 중복된 소셜 정보를 필터링하고 추천 성능을 효과적으로 향상시켰음을 나타냅니다. 특히, 사회적 신호의 노이즈 감소 기능을 통해 사용자 표현의 일관성을 높이는 데 성공하였습니다.



### Continual Learning Using a Kernel-Based Method Over Foundation Models (https://arxiv.org/abs/2412.15571)
- **What's New**: 본 논문은 지속적 학습(Continual Learning, CL) 중 클래스 증가 학습(Class-Incremental Learning, CIL)의 도전적인 설정을 다룹니다. 기존의 여러 방법에도 불구하고, 파라미터 업데이트로 인한 재앙적 망각(Catastrophic Forgetting, CF) 및 과제 간 클래스 분리(Inter-task Class Separation, ICS) 문제가 여전히 존재합니다. 이 문제를 해결하기 위해, Kernel Linear Discriminant Analysis (KLDA)라는 새로운 방법을 제안하며, 이 방법은 기초 모델(Foundation Model)에서 학습된 강력한 특징을 활용합니다.

- **Technical Details**: KLDA는 Radial Basis Function (RBF) 커널과 Random Fourier Features (RFF)를 통합해 기초 모델에서 추출된 특징 표현을 향상시킵니다. 새로운 작업이 도착하면 KLDA는 각 클래스의 평균을 계산하고, 커널화된 특징을 기반으로 모든 학습된 클래스에 대한 공유 공분산 행렬을 업데이트합니다. 이 방법은 Linear Discriminant Analysis (LDA)를 사용하여 분류를 수행하며, 각 클래스에 대한 가우시안 분포를 정의하여 결정 경계를 최적화합니다.

- **Performance Highlights**: KLDA는 텍스트 및 이미지 분류 데이터세트를 사용한 실험적 평가에서 기존 방법들보다 우수한 성능을 보였습니다. 특히, KLDA는 재생 데이터에 의존하지 않고도 CIL 성능의 상한으로 여겨지는 모든 클래스의 조합 훈련에 맞먹는 정확도를 달성하였습니다. 이는 기존의 다른 CIL 방법들이 모자란 정확도를 극복하는 데 중요한 의미를 갖습니다.



### In-context Continual Learning Assisted by an External Continual Learner (https://arxiv.org/abs/2412.15563)
- **What's New**: 이 논문에서는 InCA라는 새로운 방법을 소개합니다. InCA는 External Continual Learner (ECL)와 In-context Learning (ICL)을 통합하여 Catastrophic Forgetting (CF) 문제를 피하면서도 Scalable Continual Learning (CL)을 가능하게 합니다. ECL은 각 테스트 인스턴스에 대해 적합한 클래스의 하위 집합을 미리 선택하여 ICL의 프롬프트 길이를 관리할 수 있도록 돕습니다.

- **Technical Details**: InCA는 Mahalanobis distance를 활용하여 입력 인스턴스의 태그 임베딩과 각 클래스 분포 간의 거리를 계산합니다. 이를 통해 가장 유사한 k개의 클래스를 선택하여 ICL 프롬프트를 생성합니다. 이 접근 방식은 Irrelevant information을 제거하고 입력 토큰 한계를 효과적으로 관리할 수 있게 해줍니다. ECL은 별도의 추가 학습 없이 클래스 평균과 공분산 행렬을 점진적으로 업데이트합니다.

- **Performance Highlights**: 실험 결과, InCA는 기존 CL 기준과 비교하여 상당한 성능 향상을 보여주었습니다. InCA는 특히 다양한 벤치마크 데이터 세트에서 효과적으로 Scalability와 Accuracy를 균형 있게 유지했습니다. 이러한 성능 개선은 ICL의 이점을 활용한 새로운 CL 패러다임의 가능성을 보여줍니다.



### Architecture-Aware Learning Curve Extrapolation via Graph Ordinary Differential Equation (https://arxiv.org/abs/2412.15554)
- **What's New**: 이 논문은 신경망(Neural Network) 아키텍처를 통합하여 학습 곡선(Learning Curve) 모델링을 개선하는 새로운 접근법을 제안합니다. 기존의 방법들이 아키텍처의 영향을 무시하는 경향이 있었던 반면, 우리는 이를 고려하여, 신경망 구조의 동적 특성을 반영한 정보를 활용하였습니다. 이 방법은 AutoML 분야에서 하이퍼파라미터 튜닝과 신경망 아키텍처 검색을 가속화할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 우리는 순환 신경망(CNN) 및 다층 퍼셉트론(MLP)의 학습 곡선을 예측하기 위해 아키텍처 인식 신경 미분 방정식(Neural Differential Equation) 모델을 개발했습니다. 이 모델은 그래프 컨볼루션 네트워크(Graph Convolutional Networks, GCN)와 같은 기술을 활용하여 아키텍처의 토폴로지에서 그래프 수준 임베딩을 생성합니다. 이를 통해 학습 곡선의 변동성을 효과적으로 포착하고 예측의 불확실성을 정량화할 수 있습니다.

- **Performance Highlights**: 새롭게 제안된 모델은 기존의 최첨단 학습 곡선 예측 방법과 시계열 모델링 접근법을 초월하여 더 나은 성능을 보여줍니다. 특히, 우리의 모델은 성능이 다양한 학습 조건에서 발휘될 수 있는 능력을 가지고 있으며, 학습 곡선의 예측 정확성을 개선하여 모델 순위를 매기는 데 있어 20배 더 빠른 속도를 자랑합니다. 결국 이는 머신러닝 분야에서의 실험 속도와 자원 사용의 효율성을 크게 향상시킬 수 있습니다.



### NGQA: A Nutritional Graph Question Answering Benchmark for Personalized Health-aware Nutritional Reasoning (https://arxiv.org/abs/2412.15547)
- **What's New**: 이 논문은 개인 맞춤형 영양 건강을 위한 Nutritional Graph Question Answering (NGQA) 벤치마크를 발표했습니다. 이는 사용자 특정 건강 조건에 기초하여 음식이 건강한지 평가할 수 있는 최초의 데이터셋으로, 사용자의 의료 정보와 식단 행동을 통합하여 복잡한 영양 질문에 응답할 수 있도록 설계되었습니다. NGQA는 National Health and Nutrition Examination Survey (NHANES)와 Food and Nutrient Database for Dietary Studies (FNDDS)의 데이터를 활용하여 개별 사용자의 건강에 적합한 영양 성분을 명확히 설명합니다.

- **Technical Details**: NGQA 벤치마크는 세 가지 질문 복잡성 설정(희소, 표준, 복잡)을 포함합니다. 각 질문 유형은 세 가지 다운스트림 작업(이진 분류 – B, 다중 레이블 분류 – ML, 텍스트 생성 – TG)을 통해 평가되어 다양한 추론 측면을 탐구합니다. 연구는 LLM(backbone)와 기준 모델을 사용한 광범위한 실험을 통해 이 벤치마크가 기존 모델에 효과적으로 도전할 수 있음을 보여줍니다.

- **Performance Highlights**: NGQA는 개인 맞춤형 영양 건강 연구와 GraphQA 연구를 발전시키는 데 기여합니다. 본 연구는 사용자 의료 정보를 포함하는 최초의 벤치마크를 만들어 영양 질문 응답 작업에서 중요한 연구 격차를 해소합니다. 또한, NGQA는 GraphQA의 범위를 넓혀 보다 포괄적인 평가를 가능하게 하며, 전체 데이터 전처리에서 모델 평가에 이르는 완전한 코드베이스를 제공하여 새로운 모델 통합성을 위한 확장성을 지원합니다.



### VLM-RL: A Unified Vision Language Models and Reinforcement Learning Framework for Safe Autonomous Driving (https://arxiv.org/abs/2412.15544)
Comments:
          28 pages, 16 figures

- **What's New**: 최근 자율 주행 커뮤니티에서는 보강 학습(reinforcement learning, RL) 기반의 드라이빙 정책 학습 방법이 주목받고 있으며, 다양한 주행 시나리오에서 놀라운 발전을 이루었습니다. 본 논문에서는 VLM-RL이라는 통합 프레임워크를 제안하여, 사전 학습된 비전-언어 모델(vision-language models, VLMs)을 RL에 통합하여 이미지 관찰과 자연어 목표를 통해 보상 신호를 생성합니다.

- **Technical Details**: VLM-RL의 핵심은 대조 언어 목표(contrasting language goal, CLG)-을 보상으로 사용하는 패러다임입니다. 이는 긍정적 및 부정적 언어 목표를 활용하여 의미적 보상을 생성하며, CLG 기반의 의미적 보상과 차량 상태 정보를 결합하여 보상 안정성을 개선하는 계층적 보상 합성 방법을 도입합니다. 또한, 훈련 과정에서의 계산 효율성을 최적화하기 위한 배치 처리 기법이 사용됩니다.

- **Performance Highlights**: CARLA 시뮬레이터에서의 광범위한 실험 결과, VLM-RL은 최신 기술들의 기준을 초과하여 10.5%의 충돌률 감소와 104.6%의 경로 완료율 증가를 달성하였습니다. 또한 VLM-RL은 이전의 오프라인 보상 엔지니어링에 의존하는 기존의 RL 패러다임을 혁신할 수 있는 잠재력을 보유하고 있습니다.



### ChangeDiff: A Multi-Temporal Change Detection Data Generator with Flexible Text Prompts via Diffusion Mod (https://arxiv.org/abs/2412.15541)
- **What's New**: 본 논문은 변화 감지(change detection, CD) 작업에서 사용되는 데이터 생성과 주목할만한 발전을 제시합니다. 특히, 다변량 분포 기반 텍스트 프롬프트(multi-class distribution-guided text prompt, MCDG-TP)를 활용하여 테이블 레이아웃(layout)과 이미지를 변환하는 과정을 혁신적으로 구현합니다. 이로 인해 기존의 한정된 데이터 세트에서 발생하는 문제점을 극복하고, 변화 감지 분야의 성능을 높일 수 있는 새로운 접근법인 ChangeDiff를 소개합니다.

- **Technical Details**: ChangeDiff는 두 단계로 구성된 데이터 생성 과정을 통해 텍스트 프롬프트와 텍스트-레이아웃(text-to-layout, T2L) 모델을 사용하여 연속적인 레이아웃을 생성하고, 이후 레이아웃-이미지(layout-to-image, L2I) 모델로 이러한 레이아웃을 이미지로 변환합니다. 이 과정에서 MCDG-TP를 활용하여 사용자가 지정한 클래스와 비율에 따라 레이아웃을 유연하게 생성할 수 있도록 합니다. 추가적으로, 클래스를 분포하는 정제 손실(class distribution refinement loss)을 설계하여 T2L 모델을 훈련시킵니다.

- **Performance Highlights**: ChangeDiff의 데이터 생성 결과는 시간적 연속성, 공간적 다양성 및 품질 현실감에서 상당한 발전을 보여줍니다. 실험 결과, 새롭게 생성된 데이터는 기존 변화 감지기(change detectors)의 성능을 향상시키고, 더 나은 전이 가능성(transferability)을 제공합니다. 이러한 변화는 CD 작업의 정확도를 높이고, 과거 데이터에 대한 의존성을 줄이는 데 기여할 수 있습니다.



### FedRLHF: A Convergence-Guaranteed Federated Framework for Privacy-Preserving and Personalized RLHF (https://arxiv.org/abs/2412.15538)
Comments:
          Accepted to AAMAS 2025. This preprint represents the full version of the paper, including all proofs, experimental details, and additional discussions

- **What's New**: 최근 개인 정보 보호에 대한 우려가 커지고 개인화된 경험에 대한 수요가 증가함에 따라, 기존의 Reinforcement Learning with Human Feedback (RLHF) 프레임워크는 중앙 집중화된 데이터에 의존하여 상당한 도전에 직면하고 있습니다. 본 논문에서는 RLHF 프로세스를 분산화하는 새로운 프레임워크인 Federated Reinforcement Learning with Human Feedback (FedRLHF)를 소개합니다. FedRLHF는 여러 클라이언트 간의 협력적 정책 학습을 가능케 하며, 원시 데이터나 인간 피드백을 공유할 필요 없이 개인 정보 보호를 보장합니다.

- **Technical Details**: FedRLHF는 각 클라이언트가 로컬 환경에서 인간 피드백을 통합하여 보상 함수를 업데이트하고 개인화된 RLHF 프로세스를 통해 정책을 학습할 수 있도록 합니다. 우리는 FedRLHF에 대한 엄격한 이론적 기반을 수립하고, 수렴 보장(convergence guarantees) 및 샘플 복잡도 경계를 도출하여 클라이언트 수가 증가함에 따라 효율적으로 확장될 수 있음을 보여줍니다. 추가로, 로컬 모델은 클라이언트의 데이터와 인간 피드백으로만 훈련되어 민감한 사용자 정보가 장치에 머물도록 합니다.

- **Performance Highlights**: MovieLens 및 IMDb 데이터셋에 대한 경험적 평가를 통해 FedRLHF는 사용자 개인 정보를 보호하면서도 중앙 집중식 RLHF와 동등한 성능을 달성하며 다양한 클라이언트 환경에서 개인화를 향상시키는 기능을 보여주었습니다. 이 프레임워크는 사용자 개개인의 다양성을 고려하여 정책을 조정할 수 있는 유연성을 제공합니다. FedRLHF를 통해 우리는 개인 정보 보호와 개인화의 균형을 동시에 달성하는 데 기여하고 있습니다.



### Improved Forecasts of Global Extreme Marine Heatwaves Through a Physics-guided Data-driven Approach (https://arxiv.org/abs/2412.15532)
- **What's New**: 본 연구에서는 인공지능(AI) 기법을 활용하여 해양 열파(marine heatwaves, MHWs)를 10일 이내로 예측할 수 있는 새로운 딥러닝 신경망을 개발했습니다. 이 시스템은 대기와 MHWs 간의 상호작용을 시뮬레이션하는 커플러(coupler) 및 확률적 데이터 보강방법(probabilistic data augmentation)이라는 두 가지 특별히 설계된 모듈을 통해 극심한 MHW 예측 능력을 크게 향상시켰습니다. 전통적인 수치 예측 모델에 비해 우리의 접근법은 기존 방법보다 높은 정확도를 보이며, 더 적은 계산 자원으로 작동합니다.

- **Technical Details**: 연구에서는 해양 표면 변수의 두 가지 재분석 데이터셋(eddy-permitting 및 eddy-resolving)을 활용하여 전 세계 및 북태평양 지역 MHW 예측을 실시하였습니다. MHW는 SST 이상이 90번째 백분위수를 초과하는 경우로 정의되며, 세 가지 심각도 범주(보통, 강한, 극심함)로 나누어 분석합니다. 대기 변수는 ECWMF 재분석 v5(ERA5)를 활용하여 수집하였습니다.

- **Performance Highlights**: 우리 모델은 10일 예측이 가능한 극심한 MHW에 대한 신뢰성 있는 예측을 제공하며, 이전 데이터 기반 예측 모델에 비해 성능이 현저히 향상되었습니다. 특히, 바람 힘(wind forcing)이 MHW 변화의 주요 원인임을 설명하는 해석 가능한 AI 방법을 통해 밝혀냈습니다. 이는 MHW 예측 모델이 향후 해양 생태계 변화에 잘 대응할 수 있도록 하는 기반을 제공합니다.



### XRAG: eXamining the Core -- Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15529)
- **What's New**: 이 논문에서는 Retrieval-augmented generation (RAG) 시스템의 성능 향상을 위한 XRAG라는 새로운 오픈 소스 모듈형 코드베이스를 소개합니다. XRAG는 RAG 모듈의 핵심 구성 요소를 철저하게 평가할 수 있도록 설계되어 있으며, 사전 검색(pre-retrieval), 검색(retrieval), 후 검색(post-retrieval), 생성(generation) 등 네 가지 핵심 단계로 시스템화되어 있습니다. 이는 RAG 시스템의 복잡성이 증가함에 따라 발생할 수 있는 잠재적 실패 지점을 식별하는 중요성을 강조합니다.

- **Technical Details**: XRAG는 다양한 재구성된 데이터셋을 통해 RAG 시스템의 성능을 체계적으로 분석하고, 이를 통해 효과성을 평가하는 포괄적인 벤치마크를 제공합니다. 실험적 방법론(experimental methodologies)과 진단 테스트 프로토콜(diagnostic testing protocols)을 통해 RAG 모듈의 고유한 실패 지점을 해체하고 분석합니다. 이 과정에서 각 단계의 구성 요소들이 어떻게 상호작용하며 성능을 저해할 수 있는지를 명확하게 파악할 수 있습니다.

- **Performance Highlights**: 이 연구는 RAG 시스템 내 핵심 고급 구성 요소의 성능을 철저히 평가하여 최적화 가능한 일반적인 실패 지점에 대한 통찰을 제공합니다. 제안된 맞춤형 솔루션(custom solutions)은 검증 과정을 강화하고 모듈 전반의 성능을 높일 수 있도록 설계되었습니다. 이러한 접근 방식은 더욱 정교해진 RAG 시스템의 구현을 지원하는 데 기여할 것입니다.



### Generalized Back-Stepping Experience Replay in Sparse-Reward Environments (https://arxiv.org/abs/2412.15525)
- **What's New**: 이번 논문에서는 Back-stepping experience replay (BER) 기법을 개선한 Generalized BER (GBER)를 제안합니다. GBER는 희소 보상 환경(sparse-reward environments)에서도 효과적으로 작동할 수 있도록 원래 알고리즘을 확장합니다. 특히 복잡한 구조가 요구되는 환경에서 탐색을 위한 능력을 강화하여 BER의 잠재력을 최대한 발휘하도록 합니다.

- **Technical Details**: GBER는 relabeling mechanism과 다양한 샘플링 전략(diverse sampling strategies)을 도입하여 BER의 성능을 향상시키고 있습니다. 이 알고리즘은 목표 조건(goal-conditioned) 딥 결정론적 정책 경량학습 알고리즘을 기반으로 하여 다양한 미로 탐색 환경에서 평가됩니다. 이러한 기술적인 개선은 GBER가 단순 초급 환경을 넘어 복잡한 환경에서도 잘 작동할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, GBER 알고리즘은 다양한 희소 보상 환경에서 베이스라인 알고리즘의 성능과 안정성을 크게 향상시킬 수 있음을 보여주었습니다. 특히, 고도로 구조적인 대칭성을 지닌 환경에서 더욱 두드러진 성능 향상이 관찰되었습니다. 이러한 결과는 GBER의 적용 가능성을 넓히고 미래 연구 방향에 대한 중요한 통찰을 제공합니다.



### HREF: Human Response-Guided Evaluation of Instruction Following in Language Models (https://arxiv.org/abs/2412.15524)
Comments:
          28 pages, 15 figures

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 지침 수행 능력을 평가하는 방법을 새롭게 혁신했습니다. 기존의 평가 방법들이 편향성을 드러내는 한편, 이 연구에서는 사람의 응답을 참조로 활용하여 자동 평가의 신뢰성을 높일 수 있음을 보여줍니다. 새로운 평가 벤치마크인 HREF(Human Response-Guided Evaluation of Instruction Following)를 개발하여 총 4,258개의 샘플을 포함하는 11개 작업 카테고리에 걸친 평가를 수행합니다.

- **Technical Details**: HREF는 다양한 자동 평가 기법을 조사하여 각 작업 카테고리에 대해 가장 신뢰할 수 있는 평가 방법을 선택하는 복합 평가 설정을 사용합니다. 이에 따라 HREF 데이터셋은 각 지침의 모델 응답과 인간의 기준 응답을 비교할 수 있도록 구축되었습니다. 이러한 접근 방식을 통해 연구팀은 LLM의 성능 평가에서 사람의 작성된 응답의 비중을 높였습니다.

- **Performance Highlights**: HREF 평가 벤치마크를 통해 LLM과 인간 심사자 간의 일치도를 최대 3.2% 향상시킬 수 있었습니다. 연구진은 설계 선택의 영향, 예를 들어 평가 세트의 크기, 심사 모델 및 기준 모델 등을 조사하고, 이를 기반으로 LLM의 실적을 평가하는 라이브 리더보드를 운영합니다. 이는 LLM의 개별 작업 성능을 강조하며 기존의 평가 시스템에서 나타나는 오염 문제에서 자유롭습니다.



### InstructOCR: Instruction Boosting Scene Text Spotting (https://arxiv.org/abs/2412.15523)
Comments:
          Accepted by AAAI2025

- **What's New**: InstructOCR는 인스트럭션 기반의 장면 텍스트 스포팅 모델로, 사람의 언어 지시를 통해 이미지 내 텍스트의 이해를 높이는 혁신적인 접근을 제안합니다. 기존의 OCR 방법들이 이미지 인코더와 사전 훈련된 텍스트 정보를 주로 활용했던 반면, 본 연구는 인간 언어 지시를 모델에 통합하여 텍스트 스포팅 성능을 크게 향상시킵니다. 실험 결과, InstructOCR는 최신의 벤치마크에서 최첨단 성능을 달성하였고, VQA 태스크에도 효과적으로 적용 가능함을 입증했습니다.

- **Technical Details**: InstructOCR는 텍스트 속성에 기반하여 정교하게 설계된 지침을 사용하여 텍스트와 이미지 인코더를 훈련시킵니다. 이는 모델이 이미지 내 텍스트를 더욱 정확하고 유연하게 해석할 수 있도록 합니다. 이 프레임워크는 공공의 장면 텍스트 데이터 세트를 활용하여 비용 효율적으로 다양한 지침을 제조할 수 있으며, 이를 통해 훈련의 효율성을 높입니다.

- **Performance Highlights**: InstructOCR는 광범위한 실험을 통해 여러 장면 텍스트 스포팅 데이터 세트에서 뛰어난 성능을 보이며, 텍스트 VQA 데이터 세트에서 2.6%, ST-VQA 데이터 세트에서 2.1%의 성능 향상을 이루었습니다. 이는 인간 언어 지침을 통합함으로써 OCR 관련 과제에서 얻을 수 있는 이점을 잘 보여줍니다. 뿐만 아니라, 본 연구는 모델 해석 능력을 향상시키는 데도 중요한 기여를 하고 있습니다.



### RESQUE: Quantifying Estimator to Task and Distribution Shift for Sustainable Model Reusability (https://arxiv.org/abs/2412.15511)
Comments:
          The Annual AAAI Conference on Artificial Intelligence (AAAI), 2025

- **What's New**: 본 논문에서는 딥러닝 모델의 재훈련 비용을 예측하기 위한 새로운 지표인 RESQUE(REpresentation Shift QUantifying Estimator)를 제안합니다. RESQUE는 모델이 새로운 데이터 분포나 작업에 적응하는 데 필요한 자원의 추정치를 제공하여, 사용자들이 재훈련에 대한 정보에 기반한 결정을 내릴 수 있도록 도와줍니다. 이를 통해 지속 가능한 AI 개발을 지원하고 환경에 미치는 영향을 줄이는 데 기여하고자 합니다.

- **Technical Details**: RESQUE는 모델의 원래 분포와 새로운 분포 간의 표현(output) 변화량을 측정하여 예측합니다. 두 가지 버전이 있으며, 하나는 새로운 분포에 대한 RESQUEdist이고, 다른 하나는 특정 작업에 대한 RESQUEtask입니다. 이 두 지표는 학습 시 단일 전방 전파(forward propagation)만을 사용하여 계산되며, 이를 통해 훈련 비용과 환경적 영향을 최소화하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, RESQUE는 재훈련 비용과 높은 상관관계를 나타내며, 에너지 소비 및 탄소 배출과 같은 지속 가능성 지표와도 강한 상관성을 보입니다. 또한 RESQUE는 다른 모델 아키텍처와 무관하게 효과적으로 작동하며, 다양한 작업 및 데이터 세트에서 적용 가능성을 입증했습니다. 이는 AI 모델의 적응성을 높이는데 기여하며, 자원과 지속 가능성 목표 달성에 효과적입니다.



### Humanlike Cognitive Patterns as Emergent Phenomena in Large Language Models (https://arxiv.org/abs/2412.15501)
- **What's New**: 본 연구는 대형 언어 모델(LLM)에서 나타나는 새로운 패턴을 종합적으로 검토하며, 최근 인공지능(AI) 및 심리학에서의 연구 동향을 반영합니다. LLM은 인간의 인지 및 의사결정을 모방하거나 영향을 미치는지에 대한 논의가 활발하게 진행되고 있습니다. 이러한 연구는 LLM이 추론 및 창의성을 포함한 복잡한 사고 능력을 얼마나 잘 나타내는지를 평가하고 있습니다.

- **Technical Details**: 논문에서는 독립적인 심리학적 실험 결과를 바탕으로 의사결정 편향, 추론 및 창의성의 세 가지 인지 영역에서 LLM의 성능을 검토합니다. 의사결정에서 LLM이 몇 가지 인간의 편향을 보여주지만, 모든 인간 편향이 표현되는 것은 아니며, 이러한 점은 인지적 패턴이 완전히 일치하지 않음을 나타냅니다. 또한, GPT-4와 같은 고급 LLM은 인간의 심사숙고적 사고에 필적하는 추론을 수행하는 반면, 더 작은 모델은 그렇지 않음을 강조하고 있습니다.

- **Performance Highlights**: LLM은 이야기 만들기와 같은 언어 기반의 창의적 작업에서 뛰어난 성과를 보이는 반면, 실제 맥락이 필요한 다양한 사고 작업에서는 어려움을 겪습니다. 하지만 LLM은 인간-기계 문제 해결 과정에서 창의성을 보강할 수 있는 잠재력을 가진 협업자 역할을 할 수 있다는 점에서 의미가 있습니다. 마지막으로, 메모리, 주의집중 및 오픈 소스 모델 개발과 같은 미래의 연구 방향에 대한 지침을 제공합니다.



### A Robust Prototype-Based Network with Interpretable RBF Classifier Foundations (https://arxiv.org/abs/2412.15499)
Comments:
          To appear at AAAI 2025. Includes the Appendix

- **What's New**: 이번 연구에서는 Prototype-Based Networks (PBN)의 심화판인 Deep Prototype-Based Networks (PBNs)를 분석합니다. 특히 Classification-by-Components (CBC) 접근 방식을 중심으로 해석가능성과 관련된 여러 문제를 다루며, 그러한 문제를 해결하기 위한 CBC의 확장을 제안합니다. 마지막으로, 제안된 모델의 강건성을 보장하는 손실 함수를 도출하여 이론적 근거를 제시합니다.

- **Technical Details**: 본 연구는 심층 PBN이 Deep Radial Basis Function (RBF) 분류기와 관련이 있음을 보여줍니다. 고찰된 모델들은 입력 데이터의 잠재 공간을 통해 특징 추출(Feature Extractor)과 유사성 계산(Similarity Calculation) 과정을 거쳐 클래스를 예측합니다. 이러한 과정에서 RBF 사용과 네거티브 추론(Negative Reasoning)의 역할을 분석하고, 기존 모델의 문제를 해결하기 위한 새로운 구조를 만듭니다.

- **Performance Highlights**: 실험 결과, 제안된 심층 PBN은 여러 기준에서 최고의 분류 정확도를 기록하며, 기존 접근 방식의 해석 가능성 문제를 해결했습니다. 또한, 얕은 PBN 변형은 기존의 얕은 PBN들보다 우수한 성능을 보이면서도 본질적으로 해석 가능하고 입증된 강건성을 갖추고 있습니다. 이러한 성능 향상은 PBN이 OOD(Out-Of-Distribution) 탐지에 적합하다는 사실을 뒷받침합니다.



### The First Multilingual Model For The Detection of Suicide Texts (https://arxiv.org/abs/2412.15498)
Comments:
          SUMEval-2: The 2nd Workshop on Scaling Up Multilingual & Multi-Cultural Evaluation at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 연구에서는 자살 사고를 탐지하기 위해 다국어 모델을 제안합니다. mBERT, XML-R, mT5와 같은 transformer 아키텍처를 활용하여 스페인어, 영어, 독일어, 카탈루냐어, 포르투갈어, 이탈리아어 등 6개 언어의 자살 관련 게시물에서 텍스트를 인식합니다. SeamlessM4T를 통해 스페인어 자살 트윗 데이터셋을 다섯 가지 다른 언어로 번역하였고, 이를 기반으로 모델을 미세조정하여 다국어 데이터로 평가함으로써 새로운 방향성을 제시합니다.

- **Technical Details**: 이 연구에서는 자연어 처리(NLP) 및 딥러닝(DL) 기법을 사용하여 자살 사고를 자동으로 감지했습니다. 후보 모델인 mBERT, XML-R, mT5는 자살 텍스트를 다국어로 탐지하고, 특히 mT5가 F1 점수 85% 이상으로 우수한 성능을 기록했습니다. 언어 간 전이 학습(cross-lingual transfer learning)을 통해 다양한 문화적, 언어적 맥락에서 자살 위험을 효과적으로 식별할 수 있습니다.

- **Performance Highlights**: 모델의 평가 결과 mT5가 가장 높은 성능을 보였으며, 다국어에서의 의사 표현을 정확하게 이해할 수 있는 능력이 확인되었습니다. 번역 품질 또한 높은 퍼플렉시티(perplexity) 점수로 보장되어, 다양한 언어로 자살 사고를 탐지하는 데 있어 실질적인 기여가 예상됩니다. 이 연구는 자살 위험 식별 자동화 도구 개발의 필요성을 다시 한번 강조하며, 향후 인간 참여 평가(human-in-the-loop) 시스템에 대한 윤리적 고려 사항을 제시합니다.



### Lexicography Saves Lives (LSL): Automatically Translating Suicide-Related Languag (https://arxiv.org/abs/2412.15497)
Comments:
          The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 최근 연구들은 자살의 위험, 의도, 아이디어를 확인하고 예측하는 데 중점을 두고 있습니다. 하지만 기존의 연구들은 주로 영어와 서구 문화에 국한되어 있어, 자살이 전 세계적인 문제라는 점을 간과하고 있습니다. 본 논문은 'Lexicography Saves Lives Project'를 소개하며, 자살과 관련된 자료를 여러 언어로 번역하고, 윤리적 고려사항을 명시해 자살 예방의 기반을 강화하고자 합니다.

- **Technical Details**: 이 연구는 자살 아이디어와 관련된 기존 사전을 200개 언어로 번역하고, 번역의 품질을 이해하기 위해 인간 평가를 수행합니다. 또한, 7개의 변수를 통해 번역의 질을 평가하며, 이를 통해 사전의 품질을 수치적으로 나타냅니다. 이 과정은 기계 번역과 자연어 처리(NLP)에 의한 자동 번역의 한계를 극복하기 위한 방법론을 제시하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 저자들은 번역된 사전의 품질을 향상시키기 위한 윤리적 가이드라인과 완화 전략을 제시했습니다. 또한, 공공 웹사이트를 구축하여 번역된 자료를 공개하고, 커뮤니티 참여를 장려합니다. 이 프로젝트는 정신 건강과 자살 예방을 위한 자원 개발 및 실천을 향상시키기 위한 첫 걸음으로서, 국제적인 대화의 필요성을 강조합니다.



### TL-Training: A Task-Feature-Based Framework for Training Large Language Models in Tool Us (https://arxiv.org/abs/2412.15495)
- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)의 도구 사용 성능 향상을 위한 TL-Training 프레임워크를 제안합니다. 이 프레임워크는 비효율적인 훈련 데이터의 영향을 줄이고, SFT(감독 세밀 조정) 동안 핵심 토큰의 중요도를 동적으로 조정하는 방법을 포함합니다. 또한, 에러 유형에 최적화된 보상 메커니즘이 통합되어 있습니다.

- **Technical Details**: TL-Training 프레임워크는 훈련 데이터에서 잘못된 상호 작용 경로를 식별하여 이를 그래디언트 업데이트에서 제외함으로써 부정적인 영향을 완화합니다. 이를 통해 SFT 동안 핵심 토큰을 우선시하는 적응형 가중치 조정을 실시하고, 도구 피드백을 강화 학습의 보상 메커니즘에 통합합니다. 이러한 방식은 강화 학습의 PPO(근접 정책 최적화) 알고리즘을 이용해 최적화됩니다.

- **Performance Highlights**: TL-Training을 사용하여 CodeLLaMA-2-7B 모델을 훈련시켰습니다. 이 모델은 1,217개의 훈련 데이터 포인트로 데이터를 최소화해도, 도구 사용 성능에서 선도적인 오픈 및 클로즈드 소스 LLM과 동등하거나 그 이상을 나타냈습니다. 또한, TL-Training은 노이즈 환경에서 강건성을 향상시키고 일반적인 작업 성능을 개선하는 데 기여하여, 도구 사용 훈련을 위한 확장 가능하고 효율적인 패러다임을 제공합니다.



### Task-Specific Preconditioner for Cross-Domain Few-Shot Learning (https://arxiv.org/abs/2412.15483)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문은 Cross-Domain Few-Shot Learning (CDFSL) 분야에서 새로운 적응 메커니즘인 Task-Specific Preconditioned gradient descent (TSP)를 제안합니다. 기존의 방법들이 사용하던 고정 최적화 전략의 한계를 극복하기 위해, 우리 방법은 도메인 특화 전처리기 (Domain-Specific Preconditioners, DSPs)를 메타 학습하며 각 도메인의 특성을 반영합니다. 이러한 전처리기를 통해 목표 작업에 적응하도록 최적화를 진행하는 방식입니다.

- **Technical Details**: TSP는 메타 학습을 통해 각 도메인에서의 기초 조건을 형성하고, 이를 작업 계수(task-coefficients)와 선형 결합하여 작업 특정 전처리기(Task-Specific Preconditioner)를 생성합니다. 이 전처리기는 그래디언트 강하에 적용되며, 최적화 방향이 가장 급격한 하강 방향을 따르도록 특정 조건(positive definite)으로 제한됩니다. 이를 통해 각 목표 작업에 맞춘 적응적인 최적화를 가능하게 합니다.

- **Performance Highlights**: Meta-Dataset에서의 경험적 평가 결과, TSP는 다양한 실험 환경에서 기존의 최고 성능을 능가하는 성과를 달성했습니다. 이는 TSP가 Cross-Domain FSL 작업에서 일반화 능력을 높임을 입증하며, 동시에 meta-learning의 효과적인 활용 가능성을 보여줍니다. 전반적으로, 이 연구는 FSL 분야의 발전에 크게 기여할 것으로 기대됩니다.



### Continual Learning Using Only Large Language Model Prompting (https://arxiv.org/abs/2412.15479)
Comments:
          To Appear in COLING-2025 (short paper)

- **What's New**: 새로운 CL(Continual Learning) 패러다임인 CLOB(Continual Learning Over Black-box LLMs)가 제안되었습니다. 이는 대형 언어 모델(LLM)을 블랙 박스로 간주하며, 오직 언어 프롬프트를 통해 점진적으로 학습하는 방식을 채택합니다. CLOB는 특정 파라미터를 조정하지 않으며, API를 통해 접근할 수 있는 LLM에 적합합니다. 또한, CIS(Incremental Summarization)를 기반으로 한 새로운 CL 기법이 도입되었습니다.

- **Technical Details**: CLOB에서 사용자 는 오직 언어 프롬프트와 몇 개의 사례 예시로 LLM과 상호작용하며 새로운 작업을 학습합니다. 기존의 파라미터 업데이트로 인한 catastrophic forgetting이 사라지지만, prompt 기반의 forgetting이 새롭게 등장하게 됩니다. 이 연구는 클래스-증가 학습(class-incremental learning) 설정에서 수행되며, 기존 작업 데이터에 의존하지 않고 새로운 작업 데이터로 지속적으로 학습하도록 구성됩니다.

- **Performance Highlights**: CIS 방법은 LLM의 요약 기능을 활용하여 각 클래스에 대한 지식을 요약이라는 형태로 캡슐화합니다. 새 작업의 데이터가 도착할 때마다 이 요약을 점진적으로 학습하고 업데이트하며, 이를 통해 LLM의 입력 길이 제한 문제를 효과적으로 해결합니다. 실험 결과에서 CIS는 기존 방법들보다 상당한 정확도를 보였으며, 점진적인 학습 가능성을 입증했습니다.



### Difficulty-aware Balancing Margin Loss for Long-tailed Recognition (https://arxiv.org/abs/2412.15477)
- **What's New**: 이 논문은 클래스 불균형과 인스턴스 난이도를 동시에 고려하는 Difficulty-aware Balancing Margin (DBM) 손실 함수를 제안합니다. 기존 방법들이 클래스 수준의 불균형에 주로 집중한 반면, DBM 손실은 각 개별 샘플의 난이도 변화를 반영하여 학습의 편향을 줄이는데 도움을 줍니다. 두 가지 구성 요소로 이루어진 DBM 손실은 클래스별 마진(class-wise margin)과 인스턴스별 마진(instance-wise margin)을 포함하여, 더 어려운 샘플에 대해 더 큰 마진을 할당합니다.

- **Technical Details**: DBM 손실은 클래스 수준의 빈도 불균형을 완화하기 위한 클래스 별 마진과, 개별 샘플의 난이도에 따라 조정되는 인스턴스 별 마진으로 구성됩니다. 이 방법은 복잡한 샘플에 대해 추가적인 마진을 부여하여, 클래스 간 구별력을 높입니다. DBM 손실은 기존의 LTR(long-tailed recognition) 접근 방식과 매끄럽게 결합되며, 여러 벤치마크에서 일관된 성능 향상을 보여줍니다.

- **Performance Highlights**: CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT 및 iNaturalist2018과 같은 다양한 긴 꼬리 인식 벤치마크에서 성능을 개선하였습니다. DBM 손실은 기존의 LTR 기술과 호환 가능하며, 상당한 추가 계산 비용 없이 사용될 수 있습니다. 실험 결과, 제안된 방법은 주요 긴 꼬리 인식 벤치마크에서 경쟁력 있는 성능을 보여 품질이 검증되었습니다.



### Non-Uniform Parameter-Wise Model Merging (https://arxiv.org/abs/2412.15467)
Comments:
          9 pages, 1 figure, to be published in the Proceedings of the 9th IEEE Special Session on Machine Learning on Big Data (MLBD 2024)

- **What's New**: 본 논문에서는 Non-uniform Parameter-wise Model Merging(NP Merge)이라는 새로운 접근 방식을 제안합니다. NP Merge는 기울기 기반 최적화를 통해 각 매개변수가 최종 모델에 기여하는 정도를 학습하여 모델을 병합합니다. 이는 기존 방법들보다 더 유연한 방법으로, 다양한 아키텍처의 모델들을 여러 설정에서 효과적으로 병합할 수 있음을 입증하고 있습니다.

- **Technical Details**: 이 방법은 매개변수 수준에서 보간 계수를 학습하여 각 매개변수의 기여도를 조절할 수 있도록 하여, 서로 다른 데이터셋에서 훈련된 모델 간의 병합을 더욱 효과적으로 수행합니다. NP Merge는 모델 정렬을 위해 기존 방법들과 함께 사용할 수 있으며, 이를 통해 좋은 성능을 유지할 수 있습니다. 논문에서 제안한 NP Merge는 입력 데이터에 따라 성능 안정성을 지속적으로 분석하며, 기존 모델들이 동일한 초기화에서 유래된 경우 더 나은 성능을 보임을 강조합니다.

- **Performance Highlights**: NP Merge는 반복적인 쌍별 병합을 통해 여러 모델을 병합할 수 있는 확장성을 제공하며, 이는 최신 기법(SOTA)들보다 우수한 성능을 보여줍니다. 각 기법이 다양한 데이터 분포에 따라 적합한 상황에서 사용될 수 있도록 연구하였으며, 병합된 모델을 몇 에폭에 걸쳐 미세 조정하는 방법과의 성능 비교를 통해 일반화 측면의 이점도 함께 논의됩니다. 최종적으로, NP Merge는 다양한 환경에서 성능을 극대화하는 데 기여합니다.



### TalkWithMachines: Enhancing Human-Robot Interaction for Interpretable Industrial Robotics Through Large/Vision Language Models (https://arxiv.org/abs/2412.15462)
Comments:
          This paper has been accepted for publication in the proceedings of the 2024 Eighth IEEE International Conference on Robotic Computing (IRC)

- **What's New**: 이 논문에서는 안전-critical(안전 중요) 산업에서 활용 가능한 해석 가능하고 인간-로봇 상호작용을 강화하기 위한 새로운 접근법을 제안합니다. 자연어(natural language)를 통해 로봇에 명령을 내리고 로봇이 주변 환경을 이해할 수 있도록 하기 위해 LLMs(대형 언어 모델)와 VLMs(비전 언어 모델)의 통합을 탐구하고 있습니다. 로봇의 내부 상태와 의도를 이해하기 쉽게 설명하는 방식으로, 보다 안전하고 효과적인 운영이 가능하도록 합니다.

- **Technical Details**: 본 연구는 로봇 조작(control) 및 인식(perception)에 대한 LLMs와 VLMs의 발전을 바탕으로 하며, 로봇이 이해할 수 있는 저수준의 제어 명령 패턴을 생성할 수 있는 잠재력을 가지고 있음을 보여줍니다. 로봇의 내부 및 외부 상태를 이미지나 텍스트 기반으로 표현하여, 복잡한 궤적의 생성 및 상황 인식을 위한 패턴 인터페이스를 소개합니다. 이를 통해 로봇의 물리적 한계와 안전한 명령 실행을 위한 인식 능력을 키울 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 LLMs와 VLMs가 복잡한 궤적을 생성하고, 환경의 맥락 인식을 유지하며, 간접적 의사소통 신호를 해석할 수 있는 능력이 있음을 보여줍니다. 또한, 로봇의 물리적 구조에 대한 정보가 안전한 명령 실행을 위한 인식에 미치는 영향을 분석하여, 해석 가능하고 인간 중심의 로봇 시스템 개발의 방향성을 제시합니다. 제안된 개념들은 로봇 팔 조작 시뮬레이션을 통해 검증되었습니다.



### Northeastern Uni at Multilingual Counterspeech Generation: Enhancing Counter Speech Generation with LLM Alignment through Direct Preference Optimization (https://arxiv.org/abs/2412.15453)
Comments:
          10 pages, 6 tables, 1 figure, The First Workshop on Multilingual Counterspeech Generation (MCG) at The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 저자들은 기존의 자동 반응 생성 방식의 한계를 극복하기 위해 새로운 방법론을 제안합니다. 특히, 다양한 언어적 맥락에서의 반응 생성을 개선하기 위한 연구를 진행하며, Supervised Fine-Tuning (SFT)과 Direct Preference Optimization (DPO)를 활용합니다. 이 방법론은 LLM (Large Language Models) 출력을 인간의 선호와 align하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 모델은 DPO를 통해 생성된 반응이 맥락적으로 적합하고 언어적으로 조정 가능하도록 합니다. 또한, 지식 기반을 포함하여 생성된 반응의 사실적 정확성과 관련성을 향상시킵니다. SFT 기법과 비교했을 때, DPO-aligned 모델이 CS (Counter-Speech) 벤치마크에서 월등한 성과를 보이며, 여러 언어에 걸쳐 효과적으로 스케일링되는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, DPO 방법이 기존 SFT 기반 모델을 명백히 초과 달성했음을 입증했습니다. 이 연구는 선호 기반의 align 기법이 다양한 언어 환경에서 CS 생성 기술을 발전시키는 데 중요한 가능성을 지니고 있음을 강조합니다. 모델 군집화와 align은 영어로 이루어지며, 이후 Basque, Italian, Spanish와 같은 다른 언어에서의 메트릭 리포팅에 동일한 모델이 사용됩니다.



### AI-Enhanced Sensemaking: Exploring the Design of a Generative AI-Based Assistant to Support Genetic Professionals (https://arxiv.org/abs/2412.15444)
Comments:
          22 pages, 8 figures, 1 table, 3 appendices

- **What's New**: 이 연구는 지식 작업에서 도메인 전문가들이 생성적 AI(Generative AI)를 어떻게 활용할 수 있을지를 탐구합니다. 특히 유전체 시퀀싱(whole genome sequencing, WGS) 분석을 지원하는 생성적 AI 도우미의 설계에 중점을 두고 있습니다. 연구자들은 17명의 유전학 전문가와의 인터뷰를 통해 WGS 분석의 현재 문제점을 발견했습니다.

- **Technical Details**: 연구는 'sensemaking'이라는 개념을 중심으로 격차를 분석하며, AI 도우미와의 상호작용 설계를 위한 세 가지 디자인 고려 사항을 제시합니다. 공-설계(co-design) 세션을 통해 AI 도우미가 지원할 수 있는 작업들의 기반을 마련했습니다. 이를 통해 도메인 전문가들이 AI와 상호작용하는 방식을 이해하고, WGS 분석에서의 도전 과제를 해결하기 위한 방향을 제시하였습니다.

- **Performance Highlights**: 연구 결과는 생성적 AI가 지식 작업에서 도메인 전문가를 지원할 수 있는 능력을 강조합니다. 특히, AI를 통해 WGS 분석 과정에서의 문제를 해결할 수 있는 가능성이 제시되었습니다. 이 연구는 인간 중심 컴퓨팅(human-centered computing)과 HCI(인간-컴퓨터 상호작용)의 emprical study에 기여하며, 드문 질병 진단에 있어 AI의 유용성을 입증하고 있습니다.



### Energy consumption of code small language models serving with runtime engines and execution providers (https://arxiv.org/abs/2412.15441)
Comments:
          26 pages, submitted to journal

- **What's New**: 이 논문은 코드 생성의 맥락에서 소프트웨어 엔지니어들이 LM(언어 모델)의 에너지 효율성을 높이기 위해 감안해야 할 런타임 엔진과 실행 공급자에 대한 분석을 제시합니다. 신뢰할 수 있는 실험을 통해 SLM(소형 언어 모델) 및 여러 실행 제공자를 비교하여 에너지 소비, 실행 시간 및 자원 활용도를 평가했습니다. 특히, TORCH와 CUDA의 조합이 다른 설정에 비해 에너지 효율성에서 뛰어난 성과를 보이는 것으로 나타났습니다.

- **Technical Details**: 연구는 12개의 코드 생성 SLM을 사용한 다단계 실험 파이프라인을 기반으로 하며, HumanEval 벤치마크에서 생성한 데이터 세트를 이용하여 다양한 설정에서 에너지 소비와 실행 시간을 측정했습니다. 실행 제공자에 따라 CUDA 설정이 CPU 공급자보다 에너지 및 시간 모두에서 우수한 성능을 보였으며, TORCH와 CUDA 조합이 가장 높은 에너지 효율성을 나타냈습니다. 이 연구는 런타임 엔진과 실행 제공자 선택이 에너지 효율성에 미치는 영향을 강조합니다.

- **Performance Highlights**: TORCH와 CUDA의 조합은 37.99%에서 89.16%의 에너지 절약 효과를 보였으며, 실행 시간도 47.84%에서 89.74% 감소했습니다. 이와 함께 ONNX와 CPU 공급자 조합을 통한 최적화된 런타임 엔진이 CPU 기반 설정에서 8.98%에서 72.04%의 에너지 절약을 달성했습니다. 전체적으로, 이 연구는 에너지 효율성과 성능을 최적화하기 위해 적절한 설정을 선택하는 것이 소프트웨어 엔지니어에게 중요하다는 점을 강조합니다.



### Efficient Neural Network Encoding for 3D Color Lookup Tables (https://arxiv.org/abs/2412.15438)
Comments:
          14 pages, 13 figures; extended version; to appear in AAAI 2025

- **What's New**: 이 연구에서는 수백 개의 3D 색상 룩업 테이블(3D LUTs)을 단일 컴팩트 표현으로 인코딩할 수 있는 신경망 아키텍처를 개발하였습니다. 제안된 모델은 0.25MB 이하의 메모리 사용량으로, 512개의 LUT를 재구성할 수 있으며, 평균 색상 차이(ΔE)가 2.0 이하로 허용되는 색상 왜곡을 유지합니다. 또한, 네트워크 구조에 약간의 수정으로 역 색상 처리가 가능한 쌍향 인코딩(bijective encoding) 기능도 구현했습니다.

- **Technical Details**: 제안된 접근방식에서는 RGB 입력 색상을 RGB 출력 색상으로 매핑하는 3D LUT의 수학적 정의를 제공합니다. 연구는 다양한 네트워크 크기 변형을 통해 512개의 LUT를 인코딩하는 모델을 설계하였습니다. 이 모델은 각 LUT를 2ms 이내에 복구할 수 있어 실시간 응용이 가능합니다. 또한, 입력 색상의 가중치를 조정하는 대체 손실 함수도 도입하여 자연 이미지 색상에 대한 품질 향상이 이루어졌습니다.

- **Performance Highlights**: 모델은 평균 색상 차이(ΔE)를 1.0 이하로 유지하면서 512개의 LUT를 효과적으로 복구했습니다. 제안된 신경망은 다양한 LUT를 암묵적으로 압축하고, 기존 방법보다 99% 이상의 압축률을 달성합니다. 마지막으로, 쌍향 인코딩에 대한 수정으로 LUT를 역으로 처리할 수 있어 컴퓨터 비전과 이미지 처리 분야에서 활용 가능성이 더욱 높아졌습니다.



### Offline Safe Reinforcement Learning Using Trajectory Classification (https://arxiv.org/abs/2412.15429)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 오프라인 안전 강화 학습(offline safe reinforcement learning)에서 신뢰할 수 있는 행동을 학습하기 위한 새로운 접근 방식을 제안합니다. 기존 방법들이 환경과의 직접적인 상호작용 없이 안전 제한 조건을 충족하기 어려운 것을 해결하기 위해, 저자들은 사전 수집된 데이터셋을 바탕으로 바람직한 경로와 바람직하지 않은 경로로 분리하고, 이 정보를 기반으로 정책을 학습합니다.

- **Technical Details**: 연구진은 두 단계로 이루어진 알고리즘을 제시합니다. 첫 번째 단계에서는 사전 수집된 데이터셋을 바탕으로 바람직한 경로와 바람직하지 않은 경로의 두 개의 하위 집합으로 나눕니다. 두 번째 단계에서는 분류기를 통해 바람직한 경로를 높은 점수를, 바람직하지 않은 경로에는 낮은 점수를 부여하여 정책을 직접 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 38개의 연속 제어 작업에서 여러 기존 방법들과 비교하여 더 높은 보상과 안전 제약 조건 준수를 달성하며 우수한 성능을 보여줍니다. 전체적으로 이 연구는 안전한 행동을 학습하는 방법론 관련하여 유망한 방향을 제시하며, 결과적으로 기존 최첨단 기법들보다 높은 성능을 입증했습니다.



### Learning Visual Composition through Improved Semantic Guidanc (https://arxiv.org/abs/2412.15396)
- **What's New**: 이번 논문은 기존의 시각적 표현 학습 방식의 한계를 극복하기 위해, 간단하고 확장 가능한 접근 방식을 제안합니다. 기존의 CLIP 모델들이 인과(thought) 이해에 부족하다는 점을 지적하며, 비약적인 성능 향상을 위해 기존 캡션을 새로운 다중 모달 모델에 의해 개선한다고 설명합니다. 특히, 향상된 데이터 세트 기반 훈련을 통해 이미지 검색 작업에서 실질적인 성과를 내고 있음을 입증하고자 하였습니다.

- **Technical Details**: 제안된 방법론은 CLIP 아키텍처에 기반하여 두 가지 주요 수정 사항을 도입합니다. 첫 번째, 고품질의 영어 이미지-텍스트 쌍을 포함한 WebLI 데이터셋을 사용하여 캡션의 품질을 향상시켜 alt-text를 대체합니다. 둘째, 훈련된 다중 모달 모델의 텍스트 타워를 고성능의 텍스트 기반 기초 모델로 교체하여 시각적 임베딩 표현을 크게 개선합니다. 이러한 방법을 통해 94.5%의 명확성을 달성하며 판별 성능이 크게 향상되었습니다.

- **Performance Highlights**: 향상된 데이터 세트와 적절한 모델 수정으로, CLIP 모델의 이미지 검색 성능이 눈에 띄게 개선되었습니다. 특히, recall@1 메트릭에서 58.4%의 성능에서 94.5%로 향상되었습니다. 일반적인 captcha 값으로 ARO 평가 데이터를 통하여 비교적 단순한 다중모달 모델의 학습 성과가 나타나고, 새로운 벤치마크에 대한 필요성을 적시하며 실험 결과의 타당성을 강화합니다.



### Systematic Evaluation of Long-Context LLMs on Financial Concepts (https://arxiv.org/abs/2412.15386)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이 논문은 Long-context large language models (LC LLMs)의 신뢰성을 향상시키기 위한 연구로, 특히 긴 입력 문서를 처리하는 실제 작업에 대한 평가를 다루고 있습니다. 연구팀은 GPT-4 모델의 다양한 성능을 평가하면서, 컨텍스트 길이, 작업 난이도, 주요 정보 위치와 같은 요소들이 성능에 미치는 영향을 분석했습니다.

- **Technical Details**: 실제 금융 뉴스 데이터셋을 생성하여 다양한 난이도의 연속적인 작업을 수행하도록 하였고, 그 과정에서 LC LLMs의 성능이 길어진 컨텍스트 길이에서 어떻게 변화하는지를 살펴보았습니다. 결과적으로 간단한 작업에서도 성능 저하가 발생하며, 컨텍스트 길이가 늘어날수록 지침을 따르는 데 신뢰성을 잃고 degenerative outputs가 발생하는 현상을 관찰했습니다.

- **Performance Highlights**: 작업 지시가 컨텍스트 창 내에서 배치되는 위치나 약간의 Markdown formatting에도 여전히 민감한 반응을 보였으며, 이는 LC LLMs의 취약성을 강조했습니다. 따라서, F1 점수와 같은 포괄적인 메트릭을 활용하여 LC LLMs의 평가를 더 철저하게 수행할 것을 제안합니다.



### Automated Root Cause Analysis System for Complex Data Products (https://arxiv.org/abs/2412.15374)
Comments:
          13 pages, 6 figures

- **What's New**: ARCAS(자동 근본 원인 분석 시스템)는 빠른 진단 구현과 낮은 학습 곡선을 위해 설계된 도메인 특화 언어(DSL)를 기반으로 하는 진단 플랫폼입니다. Auto-TSG(자동 문제 해결 안내서)의 집합으로 구성되어 있으며, 제품 텔레메트리(product telemetry)를 사용하여 문제를 감지하고 거의 실시간으로 완화 조치를 적용할 수 있습니다. 이 시스템은 모니터링에 초점을 맞춘 기존 플랫폼들과는 달리, 사용자가 미리 제작된 안내서를 통해 신속하게 문제를 수정할 수 있도록 돕습니다.

- **Technical Details**: ARCAS는 주제 전문가들이 진단 플랫폼의 나머지 부분과 상호 작용 방법을 이해하지 못하더라도, 고도로 선별되고 관련성 높은 Auto-TSGs를 단시간에 제공할 수 있도록 설계된 DSL을 사용합니다. 이를 통해 문제 완화 시간(time-to-mitigate)을 줄이고, 중요한 엔지니어링 주기를 절약할 수 있습니다. 또한, 대형 언어 모델(LLM)을 활용하여 Auto-TSGs 출력 우선 순위를 정하고 적절한 조치를 취함으로써 시스템의 동작을 이해하는 고비용의 필요성을 억제합니다.

- **Performance Highlights**: ARCAS는 Azure Synapse Analytics와 Microsoft Fabric Synapse 데이터 웨어하우스의 여러 제품에서 성공적으로 사용되었으며, 이를 통해 진단 효율이 크게 향상되었습니다. 다수의 자동 문제 해결 안내서가 병렬로 실행되어 신속한 문제 해결이 가능해졌습니다. 이로 인해 엔지니어링 자원 절약 및 운영 효율성이 증가하였습니다.



### Granger Causality Detection with Kolmogorov-Arnold Networks (https://arxiv.org/abs/2412.15373)
Comments:
          8 pages, 2 figures, 2 tables

- **What's New**: 이번 연구에서는 Kolmogorov-Arnold Networks (KANs)를 활용하여 Granger 인과관계를 탐지하기 위한 신경망 모델을 제안합니다. 기존의 Multilayer Perceptrons (MLP)와 KANs의 성능 비교를 통해 KAN의 우수성을 강조합니다. 본 연구는 Granger causality KAN (GC-KAN)이라는 프레임워크를 개발하며, 이를 Granger 인과관계 탐지에 맞추어 맞춤형 훈련 접근법과 함께 제시합니다.

- **Technical Details**: GC-KAN은 다변량 Granger causality 탐지를 위해 설계된 프레임워크로, KANs의 비선형성 및 희소성 유도 정규화를 활용합니다. 이 모델은 입력과 첫 번째 은닉층 연결의 비업을 자동적으로 식별하고, 관련 없는 입력 특성을 정확히 0 가중치로 할당하여 인과관계를 파악합니다. 연구에서는 Vector Autoregressive (VAR) 모델과 비선형 Lorenz-96 시스템에서 GC-KAN의 성능을 평가합니다.

- **Performance Highlights**: KAN은 MLP에 비해 해석 가능한 Granger 인과관계를 식별하는 데 있어 더 뛰어난 성능을 보입니다. 특히 고차원 환경에서 희소한 Granger causality 패턴을 인식하는 능력에서 두드러진 성과를 나타냈으며, 이는 물리 시스템의 동적 법칙을 발견하는 AI의 응용 가능성을 보여줍니다. 실험 결과 GC-KAN은 전통적인 MLP와 비교하여 더 높은 정확도와 해석성을 제공하는 것으로 나타났습니다.



### Making Transparency Advocates: An Educational Approach Towards Better Algorithmic Transparency in Practic (https://arxiv.org/abs/2412.15363)
- **What's New**: 이번 연구에서는 설명 가능한 인공지능(Explainable AI, XAI)의 실제 적용을 촉진하기 위해 조직 내에서 알고리즘 투명성(algorithmic transparency)을 증가시키는 '투명성 옹호자(transparency advocates)'를 양성하는 접근 방식을 모색합니다. 기존의 연구가 조직의 정책에 구체적으로 어떻게 반영되는지를 평가하며, XAI의 발전에도 불구하고 여전히 존재하는 도전 과제를 다루고 있습니다. 연구팀은 이러한 투명성 옹호자들이 진정한 문화적 변화에 기여할 수 있다고 가정하고 있습니다.

- **Technical Details**: 연구는 뉴욕대학교(NYU R/AI) 교육 및 훈련 프로그램의 일환으로 알고리즘 투명성에 관한 워크숍을 개발하였습니다. 이 워크숍에서는 알고리즘 투명성의 개요, 시행을 위한 모범 사례(most practices), 옹호 전략을 소개합니다. 연구 질문은 워크숍이 참가자들의 알고리즘 투명성 문해력(algorithmic transparency literacy)을 얼마나 효과적으로 증가시켰는지, 옹호 의지를 높였는지에 대한 것입니다.

- **Performance Highlights**: 워크숍 참가자들은 최초의 교육을 통해 알고리즘 투명성에 대한 지식의 간극을 발견하였으며, 이후 이들이 각자의 조직에서 옹호 행동을 취하는 모습을 보여주었습니다. 특히, 뉴스 및 미디어 분야의 참석자들은 투명성을 위해 발언하는 등 직접적인 변화를 이끌어 냈습니다. 연구 결과는 알기 쉬운 프레임워크를 통해 향후 알고리즘 투명성을 달성하기 위한 방향성을 제시하고 있습니다.



### GeoPro-Net: Learning Interpretable Spatiotemporal Prediction Models through Statistically-Guided Geo-Prototyping (https://arxiv.org/abs/2412.15353)
- **What's New**: 이번 논문에서는 GeoPro-Net이라는 새로운 intrinsically interpretable spatiotemporal 모델을 제안합니다. 이 모델은 다양한 출처의 spatiotemporal 특징을 기반으로 한 예측 과정을 해석할 수 있도록 설계되었습니다. Geo-concept convolution operation을 도입하여 예측 패턴을 추출하고, 해석 가능한 채널 융합 및 지리적 기반 풀링을 통해 축약합니다.

- **Technical Details**: GeoPro-Net은 다중 출처의 spatiotemporal 데이터를 처리하여 복잡한 의존성과 의미론을 이해하기 쉽게 변환합니다. 이 과정에서 통계적 검사(statistical tests)를 통해 Geo-concepts를 추출하고, 이는 예측 과정의 해석을 용이하게 합니다. 또한, 여러 출력 클래스를 위한 예측을 수행하는 프로토타입( prototypes) 집합을 학습함으로써, 실제 사례에 대한 해석을 가능하게 합니다.

- **Performance Highlights**: 총 네 개의 실제 데이터셋에 대한 실험을 통해 GeoPro-Net은 경쟁력 있는 예측 성능을 보여주었으며, 기존 최첨단 모델에 비해 우수한 해석 가능성을 가지고 있음을 입증하였습니다. 이 모델은 도시의 공공 안전 및 교통 관리 등 다양한 분야에서의 활용 가능성을 제시합니다.



### Exploring Machine Learning Engineering for Object Detection and Tracking by Unmanned Aerial Vehicle (UAV) (https://arxiv.org/abs/2412.15347)
Comments:
          Accepted at ICMLA '24

- **What's New**: 최근 기술 발전으로 인명 구조(SAR) 작업에 무인 aerial vehicles (UAV)를 활용할 수 있는 가능성이 증가하고 있습니다. 본 연구는 실내 환경에서 자율 드론 시스템을 개발하여 Roomba 진공청소기를 목표로 삼고, 이 기술의 SAR 응용 가능성을 보여줍니다. UAV는 기존의 인력 구조 방식에 비해 위험을 줄이면서 구조 작업을 보다 효율적으로 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 Automated Labeling and Detection of Objects for Tracking (ALDOT)라는 머신러닝 기반의 프레임워크를 제안하여, 비디오 데이터를 처리하고 이동하는 객체를 효율적으로 탐지하고 추적하는 기술을 개발했습니다. Roomba 진공청소기를 추적하기 위해 Parrot Mambo 드론을 활용하였으며, 고해상도 카메라를 통해 indoor 환경에서 비디오 데이터를 수집했습니다. YOLOv4 모델을 사용하여 실시간 객체 탐지 및 추적 작업을 수행하였고, 데이터셋의 정확성을 보장하기 위해 여러 단계를 거쳐 라벨링 작업이 진행되었습니다.

- **Performance Highlights**: 실험 결과, YOLOv4 모델을 적용하여 Roomba를 96%의 정확도로 탐지하며, 평균 손실(loss)은 0.1942로 확인되었습니다. 이러한 성과는 자율 드론이 다양한 환경에서 효과적으로 작동할 수 있음을 보여줍니다. 또한, 본 연구는 SAR 작전에서 UAV의 가능성을 탐구하며, 향후 구조 작업에 대한 실용적인 기술 개발로 이어질 방향성을 제시합니다.



### Eliciting Causal Abilities in Large Language Models for Reasoning Tasks (https://arxiv.org/abs/2412.15314)
- **What's New**: 이 연구는 LLMs(대규모 언어 모델)의 추론 능력을 향상시키기 위해 신 causal inference(인과 추론) 기법을 적용하는 새로운 접근 방식을 제안합니다. Self-Causal Instruction Enhancement(SCIE) 메소드는 LLM이 높은 품질의 관찰 데이터를 생성하도록 하여, 이를 바탕으로 인과적 효과를 추정하고 최적화된 지침을 생성하게 합니다. 이 방법은 기존의 prompt optimization(프롬프트 최적화) 방법의 비용 문제와 해석 가능성 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: SCIE 메소드는 프롬프트 지침을 치료(treatment)로 간주하고, 텍스트 기능을 사용하여 자연어를 처리하며, 지침과 하위 작업 간의 인과 관계를 설정합니다. 연구는 인과 추론을 위해 필요한 세 가지 가정인 ignorability(무관성), positivity(양성), consistency(일관성)를 설명합니다. LLM의 추론 능력을 향상시키기 위해 우리는 프롬프트의 인과 효과를 최대화할 수 있는 지침을 식별하는 작업을 수행합니다.

- **Performance Highlights**: SCIE 방식에 대한 광범위한 실험 결과, 이 방법이 LLM의 추론 성능을 향상시키는 동시에 프롬프트의 훈련 비용을 줄이는 데 효과적임을 보여줍니다. 특히, 새로운 하위 작업에서 OR(Object-Relational) 원칙에 따라 재사용 가능한 인과 관계를 통해 성능 개선이 나타났습니다. SCIE는 인과 관계를 메타 템플릿으로 활용하여 프롬프트 생성을 효율적으로 안내합니다.



### MRWeb: An Exploration of Generating Multi-Page Resource-Aware Web Code from UI Designs (https://arxiv.org/abs/2412.15310)
- **What's New**: 본 연구는 Multi-Page Resource-Aware Webpage (MRWeb) 생성 작업을 제안하여 기존의 디자인-코드 전환 방식을 확장하고, 사용자 인터페이스(UI) 디자인을 다중 페이지 웹 UI로 변환하는 방법을 제시합니다. 이를 통해 웹페이지의 내비게이션, 이미지 로딩, 백엔드 라우팅을 지원하며, 다양한 리소스를 관리하기 위한 새로운 데이터 구조인 resource list를 도입하였습니다. 500개의 웹사이트로 구성된 새로운 데이터셋을 활용하여 MRWeb 생성의 복잡성을 분석했습니다.

- **Technical Details**: MRWeb 생성 작업은 웹 UI의 복잡한 요구 사항을 충족시키기 위해 설계되었으며, 기존 단일 페이지 웹 개발의 한계를 넘습니다. 연구팀은 resource list라는 딕셔너리 형식의 데이터 구조를 정의하여 내부/외부 리소스와 디자인 요소의 상관 관계를 추적합니다. MRWeb 툴은 이 resource list와 스크린샷을 입력으로 받아 기능적 MRWeb 코드를 생성하며, 사용자 친화적인 도구로 개발되어 개방형 연구를 지원합니다.

- **Performance Highlights**: 실험 결과, resource list를 활용한 MRWeb 생성에서는 내비게이션 기능이 0%에서 66%-80%로 향상되었으며, 이는 시각적 유사성에도 긍정적 영향을 미쳤습니다. 연구진은 MLLM의 성능을 평가하기 위한 새로운 메트릭을 제안하고 MRWeb 도구의 효과성을 분석함으로써 향후 연구에 대한 통찰력을 제공합니다. 이 연구는 MRWeb 문제를 해결하기 위한 첫 번째 평가 프레임워크를 마련하고, 모든 코드 및 데이터를 공개하여 추가 연구를 촉진하고자 합니다.



### Conceptual In-Context Learning and Chain of Concepts: Solving Complex Conceptual Problems Using Large Language Models (https://arxiv.org/abs/2412.15309)
Comments:
          Accepted to 2025 IEEE Symposium on Computational Intelligence in Natural Language Processing and Social Media

- **What's New**: 이 논문은 복잡한 개념 문제(complex conceptual problems)를 해결하기 위한 대형 언어 모델(Large Language Models, LLMs)의 얕은 커스터마이징 방법(shallow customization methods, SCMs)을 탐구합니다. 특히 새로운 알고리즘인 개념적 맥락 학습(Conceptual In-Context Learning, C-ICL)과 개념 체인(Chain of Concepts, CoC)을 제안하며, 이들이 LLMs에 개념 정보를 추가하여 문제 해결 능력을 향상시키는 방법을 다룹니다.

- **Technical Details**: 저자는 복잡한 개념 문제를 해결하기 위해 LLMs의 기존 SCM들이 효과적이지 않다는 것을 입증하고, 새로운 두 가지 SCM 알고리즘을 통해 LLMs를 개념 정보로 보강하는 방법을 제안합니다. 이 과정에서 모델의 응답 정확도, 생성 시간, 비용 등의 여러 측면을 평가합니다. 제안된 알고리즘은 기존의 인기 있는 SCM보다 30% 이상의 정확도를 보이며, 모델의 사라짐(hallucination) 현상을 줄이는 데도 기여합니다.

- **Performance Highlights**: C-ICL과 CoC 알고리즘이 제공하는 응답의 정확성과 투명성이 두드러집니다. 평가 결과, 새로운 SCM을 적용한 LLM이 기존의 SCM보다 더 나은 성능을 보였고, 문제 해결 능력의 향상이 관찰되었습니다. 이는 LLM이 복잡한 개념 문제를 해결하는 데 있어 더 신뢰할 수 있는 도구가 될 수 있음을 암시합니다.



### Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation and Execution in Complex Task Handling (https://arxiv.org/abs/2412.15305)
Comments:
          This idea was first submitted to the NeuralPS Workshop "System 2 Reasoning At Scale" in September 2024. Its OpenReview: this https URL. It was then submitted to the NAACL 2025 in October 2024, which is recorded in: this https URL. This work predates many existing works

- **What's New**: 본 논문에서는 CodeAct의 한계를 극복하기 위한 새로운 코드 생성 패러다임인 CodeProgram을 제안합니다. CodeAct는 조각난 사고를 바탕으로 다음 행동의 코드 블록을 생성하기 때문에 일관성과 안정성이 부족합니다. 이를 개선하기 위해, CodeProgram은 글로벌 사고와 정합하게 작동하여 문제 해결을 더욱 응집력 있게 수행할 수 있도록 합니다. 또한, Tree-of-Code (ToC) 구조를 도입하여 코드 실행의 실행 가능성을 기반으로 CodeProgram 노드를 자가 성장시킵니다.

- **Technical Details**: CodeProgram은 코드 생성 프로세스를 자연어 추론과 연결하여 전체 코드를 한 번의 작업에서 생성할 수 있는 구조입니다. 이 과정에서 코드 작성 자체가 사고 과정을 반영하며, 여러 추론 방법의 통합을 가능하게 합니다. Tree-of-Code (ToC)는 환경 피드백을 통해 코드 실행으로 인한 데이터를 기반으로 하여 작업 수준의 CodeProgram을 노드로 사용합니다. ToC는 코드 실행의 특성을 활용하여 자율적으로 성장하며, 병렬적인 해법 탐색을 통해 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, ToC는 CodeAct에 비해 정확도를 20% 이상 향상시키면서도 1/4의 턴 수로 복잡한 작업을 효율적으로 처리할 수 있음을 보여주었습니다. 여러 LLM들이 한 턴의 CodeProgram에서 다중 턴의 CodeAct보다 더 우수한 성능을 보였으며, 이는 LLM의 잠재력을 극대화하는 데 기여합니다. 이러한 결과는 ToC의 전방위적인 데이터 생성 방식이 감독된 및 강화 학습에 대한 가능성을 강조합니다.



### A Comparative Study of DSPy Teleprompter Algorithms for Aligning Large Language Models Evaluation Metrics to Human Evaluation (https://arxiv.org/abs/2412.15298)
Comments:
          7 pages, 10 tables, two-column format

- **What's New**: 이번 논문에서는 Declarative Self-improving Python(DSPy) 프레임워크의 여러 teleprompter 알고리즘이 인공지능 언어 모델(LLM)의 프롬프트(prompt) 최적화와 인간 주석(annotations)과의 정렬에 어떻게 기여하는지를 분석합니다. 이 연구는 특히, LLM을 평가자로 사용하여 환각 탐지(hallucination detection)를 최적화하는 방법에 초점을 맞추고 있으며, 4가지의 teleprompter 알고리즘을 비교 분석하여 그 성능을 평가합니다.

- **Technical Details**: DSPy는 LLM 파이프라인을 선언형 모듈로 추상화하여 특정 목표(e.g., 정확성)의 관점에서 시스템적으로 최적화하는 프로그래밍 모델입니다. 이 모델의 핵심 요소는 predictors, adapters, assertions, metrics 등의 다양한 구성 요소를 포함하며, teleprompters는 이러한 모듈의 품질을 개선하기 위해 특정 프로세스를 따릅니다. Candidate Generation 단계에서는 모듈의 인스턴스를 찾아 새로운 예제를 생상하고, Parameter Optimization 단계에서는 이러한 후보 매개변수를 최적화하여 최고의 조합을 선택합니다.

- **Performance Highlights**: 실험 결과, 최적화된 프롬프트는 다양한 기준 방법들을 초월하여 환각 탐지에 있어서 우수한 성능을 보였습니다. 또한, 특정 teleprompter들은 실험에서 다른 알고리즘보다 더 나은 성과를 나타냈습니다. 각 teleprompter의 성능 비교는 HaluBench라는 공개 데이터셋을 기반으로 하여 수행되었으며, 이는 프롬프트 최적화를 위한 중요한 통찰을 제공합니다.



### A Universal Model for Human Mobility Prediction (https://arxiv.org/abs/2412.15294)
- **What's New**: 이번 연구에서 우리는 인간 이동 예측을 단일 모델로 통합하는 새로운 접근 방식을 제시합니다. 제안된 유니버설 모델인 UniMob은 개인 궤적(trajectory)과 군집 흐름(crowd flow) 데이터를 모두 처리할 수 있어, 다양한 이동 데이터를 통합하여 예측 성능을 개선합니다. 이를 통해 기존의 특정 작업에 제한된 이동 예측 방법들의 한계를 극복하는 데 기여합니다.

- **Technical Details**: UniMob은 여러 관점(multi-view)의 이동 행동을 활용하는 다중 뷰 토크나이저(mobility tokenizer)를 설계하여, 추상적인 시공간 토큰(spatiotemporal tokens)으로 궤적과 흐름 데이터를 변환합니다. 이 모델은 확산 변환기(diffusion transformer) 아키텍처를 사용해 다양한 이동 데이터의 시공간 동적 패턴을 캡처하며, 개인과 집단 간의 상호작용을 촉진하는 혁신적인 양방향 정렬 메커니즘을 구현합니다. 이 메커니즘은 궤적과 흐름 데이터 사이의 공통 패턴을 학습하게 해줍니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험이 수행되었으며, UniMob은 궤적과 흐름 예측에서 기존의 최고 수준 모델들보다 뛰어난 성능을 발휘했습니다. 특히, 노이즈가 많고 데이터가 부족한 상황에서도 MAPE에서 14% 이상, Accuracy@5에서 25% 이상의 성능 향상을 달성했습니다. 이러한 결과는 UniMob의 견고성과 확장성을 입증하는 데 중요한 역할을 합니다.



### SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkag (https://arxiv.org/abs/2412.15289)
- **What's New**: 본 논문에서는 Large Language Model(LLM)의 취약점을 효과적으로 회피하고 유해한 응답을 이끌어내는 새로운 탈옥 패러다임인 Simple Assistive Task Linkage(SATA)를 제안합니다. SATA는 악성 쿼리에서 유해한 키워드를 [MASK] 특수 토큰으로 마스킹한 후, 마스킹된 키워드의 의미를 인코딩하기 위해 간단한 보조 작업을 수행합니다. 이러한 접근법은 기존의 복잡한 지침이나 반복적인 접근 방식에 비해 탈옥 성능과 효율성을 크게 향상시킵니다.

- **Technical Details**: SATA는 두 가지 보조 작업, 즉 Masked Language Model(MLM)과 Element Lookup by Position(ELP)을 활용합니다. 마스킹된 쿼리와 보조 작업을 연결하여 LLM의 안전 검사 루틴을 우회하는 방식입니다. MLM 작업은 위키백과 항목을 문맥으로 이용하여 LLM이 내용 보강을 수행하도록 유도하고, ELP는 주어진 위치에 있는 요소를 식별하는 작업을 진행합니다.

- **Performance Highlights**: 실험 결과, SATA는 최첨단 성능을 기록하며, AdvBench 데이터셋에서 SATA-MLM은 85%의 공격 성공률(ASR)과 4.57의 해로운 점수(HS)를 달성했습니다. SATA-ELP도 각각 76%의 ASR과 4.43의 HS를 기록하여 기존의 다른 방법들과 비교해 현저히 개선된 성능을 보여줍니다.　또한, SATA-ELP는 입력 토큰 사용에서 이전 방법들보다 약 10배의 절약을 이룬 것으로 나타났습니다.



### Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models (https://arxiv.org/abs/2412.15287)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 성능 개선을 위해 새로운 추론 인식 세밀 조정(inference-aware fine-tuning) 패러다임을 제안합니다. 기본적으로, 이 방법은 모델이 추론 시 최고의 성능을 보장하는 방식으로 조정됩니다. 특히 Best-of-N (BoN) 전략을 사용하여 모델이 생성한 여러 응답 중에서 가장 좋은 것을 선택하는 방식을 연구합니다.

- **Technical Details**: 우리는 BoN을 고려한 첫 번째 모방 학습(imitation learning) 및 강화 학습(reinforcement learning) 방법을 개발했습니다. BoN은 비가역적(argmax) 연산자를 통해 주어진 문제에 대한 여러 후보 응답을 생성하고, 이 중에서 최적의 응답을 선택하는데, 이는 비차별적인(non-differentiable) 문제를 해결합니다. BoN 인식 모델을 통해 우리는 RL에서의 탐색(exploration)과 활용(exploitation) 간의 트레이드오프를 묘사하는 메타 전략을 학습하는 것을 보여줍니다.

- **Performance Highlights**: 우리의 BoN 인식 세밀 조정 방법은 성능 개선과 추론 시간 계산을 최적화하는 데 매우 효과적입니다. Gemma 2B 모델은 Hendrycks MATH에서 Bo32 성능이 26.8%에서 30.8%로, pass@32는 60.0%에서 67.0%로 증가했습니다. HumanEval에서도 pass@16이 61.6%에서 67.1%로 향상되었습니다.



### Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining (https://arxiv.org/abs/2412.15285)
- **What's New**: 이 논문에서는 대규모 언어 모델의 효과적인 사전 학습을 위해 데이터 선택, 혼합 및 순서에 대한 전략을 수립합니다. 특히, 두 단계의 사전 학습(two-phase pretraining) 개념을 공식화하고 데이터 혼합 및 선택 방법을 체계적으로 연구하여 모델의 정확도를 극대화하는 방법을 제시합니다. 연구 결과, 무작위 데이터 순서와 자연 분포보다 두 단계 접근법이 평균 3.4% 및 17% 향상된 성능을 보였습니다.

- **Technical Details**: 제안된 두 단계 접근법에서 첫 번째 단계(phase-1)는 다양하고 고품질의 웹 크롤 데이터에 중점을 두고, 두 번째 단계(phase-2)는 수학, 코드 및 위키 데이터와 같은 고품질 데이터 소스를 기반으로 합니다. 데이터 혼합 과정에서 데이터 소스의 품질과 에폭(epoch) 수를 고려하여 최적의 혼합 전략을 개발합니다. 또한, 1T 토큰의 축소 샘플링(downsampled data)을 사용하여 여러 혼합을 탐색한 후 15T 토큰의 전체 데이터로 확장할 수 있는 방법을 검증합니다.

- **Performance Highlights**: 연구에서는 지식, 추론, 코딩 및 수학 벤치마크를 포함하는 다양한 다운스트림 작업을 평가하였습니다. 실험 결과, 품질 및 에폭 기반 혼합은 자연 분포 기반 혼합보다 13.2% 우수하고, 두 단계 접근법은 데이터의 무작위 순서보다 평균 3.4% 더 나은 성능을 보여주었습니다. 또한, 축소 샘플링된 데이터의 결과는 15T 토큰의 장기 스케일에서도 일반화되며, 두 단계 접근법의 확장성과 견고성을 demonstrat합니다.



### Channel Merging: Preserving Specialization for Merged Experts (https://arxiv.org/abs/2412.15283)
Comments:
          accepted by AAAI 2025

- **What's New**: 최근 대형 언어 모델(LLM)의 성능 향상을 위해 작업 특정의 세밀한 튜닝( task-specific fine-tuning )이 활용되고 있습니다. 다양한 LLM을 통합하여 전체적인 능력을 크게 향상시키는 방법이 소개되었지만, 전통적인 앙상블 방법은 메모리 집약적이어서 여러 모델을 동시에 GPU 메모리에 로드해야 하는 비효율성이 존재합니다. 이 문제를 해결하기 위해 제안된 새로운 기술인 Channel Merging을 통해 메모리 사용량을 줄이면서도 성능 저하 없이 높은 성능을 유지할 수 있음을 보여줍니다.

- **Technical Details**: Channel Merging은 유사성을 바탕으로 채널 매개변수를 클러스터링하여 오프라인으로 여러 그룹을 형성합니다. 이를 통해 그룹 내에서 유사한 매개변수만 병합하여 파라미터 충돌을 최소화할 수 있습니다. inference(추론) 중에는 병합된 그룹에서 전문 매개변수를 즉시 조회할 수 있어 전문적인 지식을 보존하며, 이전의 모델 병합 기술보다 적은 매개변수를 로드하게 됩니다.

- **Performance Highlights**: Channel Merging은 영어 및 중국어 추론, 수학 추론, 코드 생성 등 다양한 작업에서 비병합 모델과 동등한 성능을 발휘합니다. 그리고 작업 특정 라우터와 결합했을 때 전통적인 앙상블 방법이 요구되는 매개변수의 53%로 성과를 거두어, 다양한 분야에서의 효율성과 활용 가능성을 입증합니다.



### A Systematic Examination of Preference Learning through the Lens of Instruction-Following (https://arxiv.org/abs/2412.15282)
Comments:
          23 pages

- **What's New**: 최근 대규모 언어 모델(LLMs)의 인간 선호에 대한 정렬을 위한 연구가 심화되고 있습니다. 본 연구는 23개의 검증 가능한 제약 조건을 조합하여 48,000개의 고유한 지침 프롬프트를 생성하는 새로운 합성 데이터 생성 파이프라인을 사용하여, 선호 데이터 세트의 특정 속성이 LLM의 성능에 미치는 영향을 체계적으로 조사합니다. 이를 통해 지침을 따르는데 있어 모델의 조정 및 성능 향상을 도모하고자 합니다.

- **Technical Details**: 우선, 본 연구는 선택된 응답(chosen response)과 거부된 응답(rejected response) 쌍으로 LLM의 성능을 개선하는데 중점을 둡니다. 두 가지 방법인 거부 샘플링(rejection sampling, RS)과 몬테 카를로 트리 탐색(Monte Carlo Tree Search, MCTS)을 적용하여 선호 쌍을 자동으로 수집하고, 이를 통해 모델의 일반화 성능을 평가합니다. 이 연구는 지침을 따르는데 있어 응답의 공유 접두사(shared prefixes)와 응답의 대비(contrast) 및 품질(quality)이 어떻게 영향을 미치는지 이해하기 위한 실험을 포함하고 있습니다.

- **Performance Highlights**: 연구 결과, MCTS로 생성된 공유 접두사(preferences pairs with shared prefixes)가 RS로 생성된 것보다 일관되게 우수한 성능을 보였으며, 높은 대비(high-contrast) 쌍이 낮은 대비(low-contrast) 쌍보다 더 나은 성과를 내는 것으로 나타났습니다. 그러나 높은 대비와 낮은 대비 쌍의 혼합이 학습 효율성과 다양성의 균형을 맞추면서 최상의 성능을 가져옵니다. 마지막으로, 중간 난이도의 프롬프트(training prompts)는 과제 전반에 걸쳐 더 나은 일반화를 이끌어내어나가는 것으로 밝혀졌습니다.



### Context-DPO: Aligning Language Models for Context-Faithfulness (https://arxiv.org/abs/2412.15280)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 컨텍스트 충실도를 향상시키기 위해 최초로 설계된 Context-DPO라는 정렬 방법을 제안합니다. 이를 통해 모델이 제공된 정보와 사용자 지침을 보다 잘 따를 수 있도록 합니다. ConFiQA라는 새로운 벤치마크도 소개하여 모호한 구매 모델의 성능을 평가합니다.

- **Technical Details**: ConFiQA는 질문-응답 작업을 기반으로 하여 LLM의 컨텍스트 충실도를 평가합니다. QA, MR, MC라는 세 가지 데이터세트로 구성되며, 각각 단일 및 다중 훅 질문-응답과 다양한 관련된 반사 사례를 포함합니다. 모델의 훈련 상태와 크기에 따라 컨텍스트 충실도가 감소하는 경향을 보이며, 이를 해결하기 위해 Context-DPO를 통해 반사배급을 보상하는 방법을 적용합니다.

- **Performance Highlights**: Context-DPO는 LLM의 컨텍스트 충실도를 35%에서 280%까지 개선하여, 기존의 모든 모델들보다 현저히 뛰어난 성능을 보여주었습니다. 추가적으로, 이 연구는 LLM의 생성 능력에 부정적인 영향을 주지 않으면서도 컨텍스트 충실도를 저해하지 않음을 입증하였습니다. 또한, 모델의 정렬 결과를 분석하여 컨텍스트 활용의 해석 가능한 통찰을 제공합니다.



### Functional connectomes of neural networks (https://arxiv.org/abs/2412.15279)
Comments:
          Accepted at the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이 논문에서는 신경망(Neural Networks)과 인간 뇌 기능 간의 연결을 다리 역할을 하는 새로운 접근 방식을 제안합니다. 기능 연결체(functional connectome)에서 얻은 통찰을 활용하여 대규모 신경망의 토폴로지를 특성화할 수 있는 확장 가능한 방법을 제공합니다. 이는 신경망의 해석 가능성을 향상시키고, 그 기초 메커니즘에 대한 더 깊은 이해를 가능하게 합니다.

- **Technical Details**: 제안된 분석 프레임워크는 기능적 MRI와 지속적 그래프 동형성(Persistent Graph Homology) 기술에서 영감을 받았습니다. 이를 통해 고정된 임계값을 사용하지 않고도 신경망 기능을 설명하는 기능 연결체(functional connectome)의 특성을 극대화합니다. 더불어 Wasserstein 거리와 관련된 통계치를 계산하여 신경망을 분석하는 데 중요한 통계적 도구로서의 역할을 하며, centroid 기반 클러스터링 전략을 개발하는 데도 기여합니다.

- **Performance Highlights**: 제안된 프레임워크는 복잡한 신경망 기능 구조를 구별하고 해석하는 데 도움을 주며, 이론적 검증을 통한 다양한 실험을 통해 그 효용성을 입증합니다. 기능적 데이터를 통해 최적화된 신경망에 대한 정보 전파를 연구함으로써, 신경망의 동작 방식에 대한 새로운 통찰을 제공합니다. 이러한 발전은 더욱 투명하고 효율적인 신경망 모델 개발에 기여할 것으로 기대됩니다.



### DreaMark: Rooting Watermark in Score Distillation Sampling Generated Neural Radiance Fields (https://arxiv.org/abs/2412.15278)
- **What's New**: 최근 3D 자산 생성을 위한 텍스트-투-3D(Text-to-3D) 기법이 발전하면서, 생성된 NeRF(Neural Radiance Fields)의 저작권 보호의 중요성이 증가하고 있습니다. 본 논문에서는 Dreamark라는 방법을 제안하여 NeRF 생성 과정 중에 수신자를 대상으로 한 비밀 메시지를 삽입할 수 있는 새로운 방안을 제공합니다. 이러한 방법은 전통적인 사후 생성 방식과는 다르게, 생성 과정 내에서 워터마크를 포함하므로 보안성과 민첩성을 향상시킵니다.

- **Technical Details**: Dreamark는 Score Distillation Sampling(SDS) 기법과 통합하여, NeRF 아키텍처를 변경하지 않고도 워터마크를 생성할 수 있는 최초의 방법입니다. 이를 통해 비밀 메시지가 랜덤한 트리거 뷰포트에서 검증될 수 있도록 설계되었습니다. 워터마킹 과정에서 스테이지 간의 지연이 없으며, 비워터마크 버전의 NeRF가 생성되지 않도록 보장합니다.

- **Performance Highlights**: 실험 결과 Dreamark는 다양한 이미지 변환에 대해 90% 이상의 비트 정확도를 달성하였고, 생성 품질의 저하 없이 워터마킹 과정을 수행할 수 있음을 입증하였습니다. 이는 NeRF의 생성 품질을 유지하면서 동시에 높은 수준의 강인성을 보장하는 것으로, 텍스트-투-3D 생성 분야에 큰 기여를 하게 될 것입니다.



### PLPP: Prompt Learning with Perplexity Is Self-Distillation for Vision-Language Models (https://arxiv.org/abs/2412.15277)
- **What's New**: 이 논문에서는 VL(비전-언어) 모델의 성능 향상을 위해 PLPP(PerPlexity 기반 Prompt Learning)이라는 새로운 정규화 방법을 제안합니다. PLPP는 기존의 CoOp 방법에서 발생할 수 있는 오버피팅 문제를 해결하는 데 중점을 두고 있으며, 단일 CLIP 손실에 의존하는 대신에 perplexity 손실을 활용하여 프롬프트 학습을 조정합니다. 이 방법은 비트 리미터와 언어 모델 헤드를 통합하여 단어 확률 분포를 출력함으로써 더욱 안정적인 학습 과정을 구현합니다.

- **Technical Details**: PLPP는 두 단계로 이루어진 작업을 통해 프롬프트의 perplexity를 계산합니다. 첫 번째 단계에서는 임베딩 레이어의 가중치와 각 프롬프트 벡터 간의 코사인 유사도를 계산하여 라벨을 얻습니다. 이후에는 추가 학습 없이 언어 모델 헤드를 통해 단어 확률 분포를 출력하여 perplexity를 계산합니다. 이 두 단계를 통해 프롬프트 최적화에서의 정보를 합리적으로 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과, PLPP는 여러 분류 작업에서 기존 방법들에 비해 뛰어난 성능을 보였습니다. PLPP는 다른 방법들에 비해 오버피팅을 방지하고, 프롬프트 최적화를 안정화함으로써 학습 진전을 가속화하는 데 기여했습니다. 이 연구는 VL 모델에서 이미 잘 알려진 언어 모델링 기법을 활용하여 새로운 방향성을 제시하고 있습니다.



### Exploring Query Efficient Data Generation towards Data-free Model Stealing in Hard Label Setting (https://arxiv.org/abs/2412.15276)
- **What's New**: 본 논문은 데이터 없이 모델을 탈취하는 새로운 접근법인 Query Efficient Data Generation (QEDG)를 제안합니다. 기존 방식들이 고신뢰도의 샘플을 생성하여 목표 모델의 행동을 복제하는 데 어려움을 겪었던 점을 해결하고자, 두 가지 손실 함수를 도입하여 다중 클래스에 걸쳐 더 나은 의사결정 경계에 맞는 샘플을 생성합니다. 이는 공격자가 보낸 쿼리 수를 최소화하면서도 더 많은 감독 정보를 얻을 수 있는 방법론을 다룹니다.

- **Technical Details**: QEDG는 두 가지 손실 함수를 통해 생성된 샘플이 목표 모델의 결정 경계에 가까운 위치에 있도록 하고, 동일 클래스 내의 샘플들 간의 간격을 넓히는 방법을 적용합니다. 또한, 쿼리 없이 단일 요청으로 추가적인 감독 정보를 확보할 수 있는 쿼리 없는 샘플 증강 기법을 제안합니다. 이를 통해 모델의 정확성과 샘플의 다양성을 동시에 확보할 수 있습니다.

- **Performance Highlights**: 여러 데이터 세트에 대한 실험 결과, QEDG는 기존 최첨단 방법들과 비교하여 적은 수의 쿼리로도 더 나은 성능을 보였습니다. 이는 모델 탈취 공격에 있어 효과적인 방안이 될 가능성을 보여줍니다. 또한, QEDG의 주요 기여는 샘플 생성 프로세스를 재구성하고, 목표 모델과 대체 모델 간의 유사성을 더 정확하게 평가할 수 있는 일관성 비율 메트릭을 도입한 것입니다.



### Fooling LLM graders into giving better grades through neural activity guided adversarial prompting (https://arxiv.org/abs/2412.15275)
Comments:
          16 pages, 11 figures

- **What's New**: 이 논문은 인공지능(AI)이 주요 의사결정 및 평가 프로세스에 배치될 때 발생할 수 있는 내재된 편향을 드러내는 체계적인 방법을 제안합니다. 특히 자동 에세이 채점(automated essay grading) 시스템을 예로 들어, 악성 행위자들이 의사결정 결과를 왜곡할 수 있는 편향을 추적하고 분석합니다. 이를 통해 대규모 언어 모델(LLM) 채점기가 인간이 탁월한 점수를 부여하는 것보다 훨씬 높은 점수를 주도록 속일 수 있는 방법을 실증적으로 보여줍니다.

- **Technical Details**: 연구에 사용된 접근법은 왜곡된 의사결정 결과를 예측하는 숨겨진 신경 활동 패턴(neural activity patterns)을 식별하는 것에서 시작합니다. 이후 이러한 패턴을 증폭시키기 위해 적대적 입력 접미사(adversarial input suffix)를 최적화합니다. 연구에서는 또한 '매직 워드(magic word)'가 공격의 효능에 중요한 역할을 한다는 사실을 밝혀내었으며, 이는 LLM의 감독하에 미세 조정(supervised fine-tuning)에 자주 사용되는 채팅 템플릿(chat templates)의 구조에 기인합니다.

- **Performance Highlights**: 이 연구는 현재의 LLM에서 발견된 취약점을 드러내며, 숨겨진 편향을 탐지하고 제거하는 체계적인 방법도 제안합니다. 소규모 구조 변경만으로도 편향을 크게 줄일 수 있음을 입증하여, AI 안전성 및 보안성 확보에 기여하게 됩니다. 이러한 결과는 상업적 폐쇄 소스 모델인 Gemini를 포함한 다양한 모델에서 블랙박스 공격(black-box attacks)으로 전이될 수 있음을 시사합니다.



### Memory-Augmented Agent Training for Business Document Understanding (https://arxiv.org/abs/2412.15274)
Comments:
          11 pages, 8 figures

- **What's New**: Matrix(메모리 증강 에이전트 훈련을 통한 추론 및 반복 탐색)라는 새로운 패러다임을 제안합니다. 이 시스템은 LLM(대형 언어 모델) 에이전트가 경험 기반의 메모리 개선과 반복 학습을 통해 도메인 전문성을 지속적으로 구축할 수 있도록 도와줍니다. 기존 LLM의 한계를 극복하고 비즈니스 문서 처리 작업을 위한 보다 전문화된 도구로 변모시킬 수 있는 가능성을 열어줍니다.

- **Technical Details**: Matrix는 문서의 구조 및 추출 패턴에 대한 이해를 체계적으로 개선하는 독창적인 반복 자기 정련 과정(iterative self-refinement mechanism)을 통합하고 있습니다. 이 프레임워크는 에이전트가 작업 탐색(task exploration) 및 최적화(task optimization)를 반복적으로 수행하여 일반적인 작업 구조에 대한 통찰력을 향상시키는 과정을 포함합니다. 이러한 프로세스는 향후 작업 해결 시 유용한 긴급 기억(long-term memory)을 생성합니다.

- **Performance Highlights**: 실험 결과, Matrix는 체인 오브 사고(prompting) 대비 30.3%의 성능 향상을 보였고, 기본 LLM 에이전트 대비 35.2%, 반사(reflexion) 대비 27.28%의 개선을 이뤘습니다. 최적화된 시스템은 API 호출 수를 줄이고, 비용을 절감하는 동시에 평균적으로 더 긴 문서도 처리할 수 있는 강점을 보여주었습니다. 이러한 향상된 퍼포먼스는 문서 처리와 비즈니스 환경에서 매우 중요한 현상을 나타냅니다.



### SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15272)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 다양한 태스크(Task)에서 뛰어난 유연성을 보여주고 있습니다. 이러한 맥락에서, Retrieval-Augmented Generation (RAG) 접근 방식이 외부 지식 소스인 지식 그래프(KGs)를 활용하여 환각(hallucination)을 제거하는 데 있어 강력한 방법으로 자리잡고 있습니다. 본 논문에서는 KG 기반 RAG 태스크를 조사하고, 유사 그래프 강화 검색 증대 생성(SimGRAG) 방법을 제안하여 쿼리 텍스트와 KG 구조를 정렬하는 문제를 효과적으로 해결합니다.

- **Technical Details**: SimGRAG 방법은 두 단계의 프로세스를 통해 쿼리 텍스트와 KG 구조의 정렬을 수행합니다. 첫 번째 단계는 쿼리를 원하는 그래프 패턴으로 변환하는 LLM을 사용하는 'query-to-pattern' 단계입니다. 두 번째 단계는 패턴과 후보 서브그래프 간의 정렬 정도를 그래프 의미 거리(Graph Semantic Distance, GSD) 메트릭을 이용해 정량화하는 'pattern-to-subgraph' 단계입니다. 이 방법은 1000만 규모의 KG에서 1초 이내에 최상위 k 서브그래프를 효율적으로 식별하는 최적화된 검색 알고리즘을 개발하여 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, SimGRAG는 질문 응답 및 사실 검증(task)에서 최신 KG 기반 RAG 방법들을 초월하는 성능을 보여줍니다. 본 논문이 제시하는 방법은 플러그 앤 플레이(plug-and-play) 사용성과 확장성(scalability)을 갖추고 있어, 다양한 KG 및 LLM과의 매끄러운 통합이 가능합니다. 또한, SimGRAG는 불필요한 정보 유출을 방지하고, 가장 관련성 높은 서브그래프를 명확하게 찾아내는 능력을 갖추어 있습니다.



### Baichuan4-Finance Technical Repor (https://arxiv.org/abs/2412.15270)
- **What's New**: 이번 논문에서는 Baichuan4-Finance 시리즈의 개발을 소개합니다. 이 모델들은 Baichuan4-Turbo 기본 모델을 기반으로 하며, 금융 분야에 특화되어 있습니다. Baichuan4-Finance-Base와 Baichuan4-Finance로 구성되어 있으며, 금융 지식을 획득하는 새로운 훈련 전략을 제안합니다.

- **Technical Details**: Baichuan4-Finance-Base의 토크나이저는 byte-level byte-pair encoding (BBPE)을 사용하며, 141,056개의 정규 토큰으로 구성됩니다. Qued Query Attention (GQA)을 활용하여 추론 속도를 개선하고, RMSNorm을 통해 학습 안정성을 보장합니다. RoPE를 활용하여 위치 인코딩을 수행하며, 기존의 Multi-Head Attention (MHA)보다 효율적인 아키텍처를 구현했습니다.

- **Performance Highlights**: Baichuan4-Finance-Base는 여러 금융 작업에서 거의 모든 경쟁 모델을 크게 능가하는 성과를 나타냅니다. 일반 LLM 기준에서도 성능을 유지하면서 금융 응용 시나리오에서 더욱 놀라운 성능을 보입니다. 이는 금융 LLM 분야의 혁신을 촉진할 잠재력을 보여줍니다.



### The Reliability Paradox: Exploring How Shortcut Learning Undermines Language Model Calibration (https://arxiv.org/abs/2412.15269)
Comments:
          10 pages; 9 figures. Accepted for publication at the Hawaii International Conference on System Sciences (HICSS-58) 2025

- **What's New**: 이 논문은 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)의 신뢰성과 일반화 능력 간의 관계를 조사합니다. 기존 연구에서 낮은 교정 오류(Expected Calibration Error, ECE)가 신뢰할 수 있는 예측을 의미한다고 여겼지만, 이 연구는 이러한 가정을 뒤집으며, 낮은 교정 오류가 오히려 비일반적인 결정 규칙을 나타낼 수 있음을 보여줍니다.

- **Technical Details**: 모델의 신뢰성을 평가하기 위해, 이 연구에서는 통계적 교정 평가 지표인 ECE를 활용하여 PLMs를 분석합니다. 또한, 통계적 교정 오류 측정이 결정 규칙의 비강건성을 포착하는 데 한계가 있음을 강조하고 있습니다. 단기 학습(shortcut learning)에 기반한 모델의 신뢰도를 평가하는 과정에서 통계적 기법과 함께 데이터 통계를 활용한 접근 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, 잘 교정된 모델이 반드시 신뢰성이 높은 것은 아님을 발견했습니다. 저자들은 PLMs의 다양한 분류 작업에서 모델의 단기 학습 행동을 확인하고, 이러한 행동이 교정 오류와 어떻게 관련되는지 분석했습니다. 이는 PLMs의 신뢰성을 높이기 위해 수학적 교정 능력과 일반화 목표 간의 간극을 메우는 필요성을 제기합니다.



### Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph (https://arxiv.org/abs/2412.15268)
Comments:
          8 pages of content, 7 pages of Limitation, Ethical Statement, Reference ans Appendix

- **What's New**: 이번 논문에서는 MetaTox라는 새로운 방법을 제안합니다. MetaTox는 메타-독성 지식 그래프(meta-toxic knowledge graph)를 활용하여 혐오 및 독성 콘텐츠 탐지를 강화하는 기술입니다. 이 방법은 기존의 독성 기준 데이터셋을 이용하여 포괄적인 메타-독성 지식 그래프를 구축하고, 이를 통해 독성 정보의 정확도를 높이는 것을 목표로 합니다.

- **Technical Details**: MetaTox의 구성 과정은 세 가지 단계로 진행됩니다. 첫 번째는 근거 추론(rationale reasoning)으로, 어떤 내용이 독성을 유발하는지를 파악하는 과정입니다. 두 번째는 삼중항 추출(triplet extraction) 단계로, 독성 개체와 관계를 추출하며 질의의 품질을 확인하는 전략을 포함합니다. 마지막으로 중복 제거(duplicate removal) 과정을 통해 유사한 의미의 노드와 관계를 통합하여 지식 그래프를 완성합니다.

- **Performance Highlights**: MetaTox는 여러 데이터셋에서 실시한 실험과 심층 사례 연구를 통해 false positive 비율을 상당히 줄이면서 전반적인 독성 탐지 성능을 향상시키는 데 성공했습니다. 향후 이 지식 그래프는 공개될 예정이며, 논문 수용 시 추가적인 공개가 계획되어 있습니다.



### Toxicity Detection towards Adaptability to Changing Perturbations (https://arxiv.org/abs/2412.15267)
- **What's New**: 이 연구는 독성 콘텐츠 탐지 분야에 새로운 문제인 '지속적 학습 jailbreak 교란 패턴'을 도입했습니다. 이는 사용자가 탐지기를 피하기 위해 새로운 교란 패턴을 창출하는 점을 반영하여 탐지기의 접근 방식을 혁신적으로 변화시키고자 합니다. 연구진은 기존의 수많은 전통적 탐지 방법들이 변화된 교란 패턴에 취약하다는 점을 확인하고, 이에 대한 해결책을 모색합니다.

- **Technical Details**: 논문에서 제안한 방법은 총 9종의 교란 패턴으로 생성된 데이터셋 DynEscape(다이나믹 이스케이프)를 기반으로 하며, 이를 통해 탐지기의 강건성을 유지하기 위해 도메인 증가 학습 방식이 사용됩니다. 연구진은 제안된 데이터셋을 통해 현재 탐지기들이 미지의 유형의 교란 독성 텍스트를 식별하는 데 어려움을 겪고 있다는 것을 체계적으로 검증했습니다. 또한, 지속적 학습 기법을 통해 탐지기는 이전과 새로운 교란 패턴 모두를 인식할 수 있는 능력을 키울 수 있습니다.

- **Performance Highlights**: 연구팀은 제안된 지속적 학습 접근 방식인 DynDetect(다이나믹 탐지)가 기존의 독성 탐지 최첨단 모델들과 비교하여 우수한 성능을 발휘함을 입증했습니다. 이로 인해 독성 콘텐츠의 탐지 정확성이 증가함은 물론, 다양한 교란 패턴에 대한 강건성 또한 확보되었습니다. 연구팀은 자가 감독 학습을 통해 탐지기의 지속적인 성능 향상을 가능하게 하는 새로운 연구 기회를 제공하고자 합니다.



### On the Structural Memory of LLM Agents (https://arxiv.org/abs/2412.15266)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM) 기반 에이전트의 성능에 미치는 메모리 구조와 메모리 검색 방법의 영향을 탐구합니다. 4가지 유형의 메모리 구조(조각, 지식 삼중, 원자 사실 및 요약)와 이들을 결합한 혼합 메모리를 평가하여, 각 구조별로 특정 작업에 최적화될 수 있는 방법을 제안합니다. 또한, 단일 단계 검색, 재정렬 및 반복 검색과 같은 3가지 메모리 검색 방법을 평가하며, 이 조합이 에이전트의 전반적인 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 네 가지 작업(다중 홉 QA, 단일 홉 QA, 대화 이해, 독서 이해)과 여섯 개 데이터 세트를 활용하여 수행됩니다. 실험 결과, 각각의 메모리 구조가 특정 작업에 제공하는 고유한 이점이 있음을 발견했습니다. 더불어, 혼합 메모리가 노이즈 환경에서도 뛰어난 탄력을 보여주며, 반복 검색 방법이 여러 상황에서 가장 우수한 성능을 발휘함을 확인했습니다.

- **Performance Highlights**: 결과적으로, 각 메모리 구조는 고유한 강점을 발휘하며, 혼합 메모리는 다양한 작업에서 균형 잡힌 경쟁력을 유지합니다. 조각과 요약은 긴 맥락을 요구하는 작업에서 특히 효과적이고, 지식 삼중과 원자 사실은 관계적 추론 및 정밀성에서 두각을 나타냅니다. 반복 검색은 대다수 작업에서 가장 효과적인 메모리 검색 방법으로 확인되었습니다.



### Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large Language Models (https://arxiv.org/abs/2412.15265)
- **What's New**: 이번 논문에서는 중국의 안전 관련 지식을 평가하기 위해 중국 SafetyQA 벤치마크를 소개합니다. 이는 기존의 데이터셋들이 일반 지식에 집중하는 반면, LLM의 안전성이 중요한 법률, 정책, 윤리와 관련된 사실적 능력을 평가하는 데 중점을 두고 있는 점에서 혁신적입니다. 중국 SafetyQA 벤치마크는 고품질의 안전 예제 2000개 이상을 포함하고 있으며, 다양한 카테고리에 걸쳐서 안전 지식을 포괄적으로 다루고 있습니다.

- **Technical Details**: 중국 SafetyQA 데이터셋은 7가지 주요 카테고리로 조직되어 있으며, 각 카테고리는 다양한 세부 주제를 포함하고 있습니다. 데이터셋은 안전 관련 지식의 정확성과 질을 보장하기 위해 엄격한 선택과 주석 과정을 거쳤으며, 모든 샘플은 총 두 가지 형식(질문-응답(QA) 및 객관식(MCQ))으로 제공됩니다. 이러한 배열은 LLM의 안전 지식 경계를 쉽게 평가할 수 있도록 돕습니다.

- **Performance Highlights**: 30개 이상의 LLM을 평가한 결과, 대부분의 모델이 안전 분야 내에서 사실적 정확성에 부족함을 드러냈습니다. 또한, LLM들은 교육 데이터에서 지식 오류를 포함하고 있으며, 안전 지식에 관해서는 과도한 자신감을 보였습니다. 연구 결과는 RAG(회수 보강 생성)가 안전 사실성을 높이는 데 도움을 줄 수 있음을 나타냅니다.



### ReXTrust: A Model for Fine-Grained Hallucination Detection in AI-Generated Radiology Reports (https://arxiv.org/abs/2412.15264)
Comments:
          Accepted to AIMedHealth 10 pages, 5 figures

- **What's New**: 이 연구에서는 AI 생성 방사선 보고서에서의 허위 진술을 감지하기 위한 새로운 프레임워크인 ReXTrust를 소개합니다. ReXTrust는 대규모 비전-언어 모델(LVLM)에서의 히든 상태 시퀀스를 활용하여 구체적인 검사 결과에 대한 허위 진술 위험 점수를 생성합니다. 이 모델은 특히 임상적으로 중요한 결과에 대한 허위 진술 감지에서 우수한 성능을 발휘하여, 자동화된 방사선 보고서의 안전성과 신뢰성을 개선할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ReXTrust는 LVLM에서 생성된 방사선 보고서의 허위 진술을 감지하기 위해 설계된 화이트 박스 모델입니다. 이 시스템은 LVLM 히든 상태에서 교육된 자기 주의(self-attention) 모듈을 사용하여, 특정 방사선 검사 결과에 대한 세밀한 통찰을 제공하고 신뢰할 수 있는 허위 진술 위험 점수를 산출합니다. 내부 모델 표현을 분석함으로써 ReXTrust는 생성 과정에서 허위 진술을 식별할 수 있으며, 후속 분석에 의존하지 않습니다.

- **Performance Highlights**: ReXTrust는 MIMIC-CXR 데이터세트의 하위 집합에서 평가되었으며, 모든 검사 결과에 대해 AUROC 0.8751, 임상적으로 중요한 결과에 대해서는 AUROC 0.8963을 달성했습니다. 이는 기존 접근법과 비교할 때 우수한 성과를 나타냅니다. 결국 ReXTrust는 모델 히든 상태를 활용하여 의료 AI 시스템의 허위 진술 감지 신뢰성을 높일 수 있는 가능성을 보여주고 있습니다.



### Advanced ingestion process powered by LLM parsing for RAG system (https://arxiv.org/abs/2412.15262)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문은 다양한 구조적 복잡성을 가진 멀티모달 문서를 처리하는 데 어려움을 겪는 Retrieval Augmented Generation (RAG) 시스템을 위한 새로운 멀티 전략 파싱 접근법을 제안합니다. LLM 기반 OCR을 활용하여 프레젠테이션 및 고밀도 텍스트 파일을 포함한 다양한 문서 유형의 콘텐츠를 추출합니다. 이 방법론은 서로 다른 정보 유형 간의 관계를 생성하고 문맥 인식 메타데이터를 생성하는 노드 기반 추출 기법을 사용합니다.

- **Technical Details**: 선행 처리 단계는 구문 분석, 조립 및 메타데이터 추출의 세 하위 프로세스로 구성됩니다. 구문 분석 단계에서는 Python 라이브러리와 멀티모달 LLM을 이용하여 이미지 및 텍스트 콘텐츠를 추출하며, AWS Textract와 같은 외부 기계 학습 모델을 사용하는 OCR도 포함됩니다. 각 페이지의 이미지를 분석하고 설명한 후, Multimodal Assembler Agent가 모든 페이지의 텍스트를 통합하여 종합적인 문서 수준의 Markdown 파일을 생성합니다.

- **Performance Highlights**: 실험적인 평가 결과는 RAG 시스템의 효율성이 향상됨을 보여주며, 정보의 회수 능력과 답변의 적합성이 증가했습니다. 이 접근법은 특히 답변의 관련성과 정보의 정확성을 측정하는 데 중요한 평가 지표를 설정하여 시스템의 전반적인 품질 향상에 기여했습니다. 논문에서 사용된 평가 지표들은 RAG 시스템의 효과성을 정량화하는 데 필수적이며, 이를 통해 시스템의 신뢰성을 확보할 수 있습니다.



### Structured Extraction of Real World Medical Knowledge using LLMs for Summarization and Search (https://arxiv.org/abs/2412.15256)
Comments:
          10 pages, 3 figures, Work published in 4th Workshop on Knowledge Graphs and Big Data (In Conjunction with IEEE Big Data 2024)

- **What's New**: 이 논문에서는 질병 발견과 분석을 가속화하기 위해 환자 지식 그래프를 구축하는 새로운 접근 방식을 제안합니다. 기존의 질병 온톨로지가 환자의 상태나 희귀 질병의 미세한 차이를 포착하기 어려운 반면, 대규모 언어 모델 추출 기술을 활용하여 자연어를 통해 데이터를 보다 유연하게 추출할 수 있는 방법을 제시하고 있습니다. 이를 통해 실세계 데이터에서 의미 있는 통찰을 도출하고, 기존 온톨로지에 연결되는 환자 특화 지식 그래프를 구축했습니다.

- **Technical Details**: 이 연구에서 제안한 방법은 메타 데이터, SNOMED-CT, RxNORM, HPO와 같은 기존 온톨로지에 추출된 개체들을 연계하여 'ground truth'를 제공합니다. 실험에 사용된 데이터는 약 3,360만 환자의 대규모 이동 치료 전자 건강 기록(EHR) 데이터베이스로, 이를 통해 Dravet 증후군과 Beta-propeller protein-associated neurodegeneration (BPAN) 환자를 찾는 데 성공했습니다. 환자의 증상 기반 검색 방식과 환자 특화 지식 그래프 구축을 통해 실제 질병 발견 사례를 입증했습니다.

- **Performance Highlights**: 최신 데이터와 사례 연구를 통해 제안된 방법의 효과가 입증되었습니다. LLM 기반 엔터티 추출을 사용하여 Dravet 증후군에 대한 ICD10 코드 검증을 통해 환자 특성을 잘 설명하며, 의료 기록에서 유래한 데이터를 활용해 다양한 질병 연구 결과를 종합적이고 저장된 지식으로 제공합니다. 이와 같이 고도화된 자동화 시스템은 의료 연구와 데이터 통합의 혁신을 가능하게 하며, 실질적인 임상 연구 설계를 가속화하는 데 기여합니다.



### Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation (https://arxiv.org/abs/2412.15255)
Comments:
          14 pages

- **What's New**: 이 논문에서는 지식 증류(knowledge distillation)를 활용하여 언어 모델 벤치마크 점수를 조작할 수 있다는 사실을 밝혀내어 현재의 평가 관행에서의 중대한 취약점을 드러냈습니다. 우리는 "데이터 세탁(Data Laundering)"이라고 불리는 세 단계의 과정을 소개하는데, 이는 재정의 머니 론더링(money laundering)과 유사합니다. 이 방법은 겉보기에는 합법적인 중간 학습 단계를 통해 벤치마크 특정 지식을 은밀하게 전이할 수 있게 해줍니다.

- **Technical Details**: 제안된 데이터 세탁 방법은 지식 증류 기술을 통해 이루어집니다. 이 과정은 세 가지 단계로 나누어지는데, 배치(placement), 레이어링(layering), 통합(integration)입니다. 첫 번째 단계에서는 테스트 데이터로 훈련된 튜터 모델에서 벤치마크 지식을 "배치"하여 초기 지식 자본을 형성합니다. 그런 다음, 레이어링 단계에서 지식 증류를 통해 합법적인 중간 데이터셋을 사용하여 지식을 전이합니다.

- **Performance Highlights**: 실험을 통해 2레이어 BERT 학생 모델이 GPQA에서 최대 75%의 벤치마크 정확도를 달성할 수 있다는 것을 보여주었습니다. 그러나 이 성과는 실제로 진정한 추론 능력을 개발하지 않고도 이루어졌습니다. 이러한 방법은 연구자들이 지식 증류를 사용하는 중에 의도치 않게 점수를 부풀리는 방식을 채택할 수 있음을 강조하며, AI 평가 방법의 강화를 위한 필요성을 환기시킵니다.



### NER- RoBERTa: Fine-Tuning RoBERTa for Named Entity Recognition (NER) within low-resource languages (https://arxiv.org/abs/2412.15252)
- **What's New**: 최근 자연어 처리(Natural Language Processing, NLP)는 일상 생활에서 광범위하게 사용되고 있으며, 특히 음성 이해, 번역, 명명된 개체 인식(Named Entity Recognition, NER), 텍스트 분류 및 ChatGPT와 같은 생성적 텍스트 모델에 이르기까지 그 활용도가 높습니다. 하지만 쿠르드어(Kurdish Language)는 아직 NLP 응용 프로그램에 포함될 만큼의 데이터가 부족하여, 쿠르드어 NLP(KNLP) 개발에 독특한 도전 과제가 존재합니다. 본 논문에서는 쿠르드어 NER(KNER)에 대한 한계를 극복하기 위한 방법론을 제안합니다.

- **Technical Details**: 우리는 사전 학습된 RoBERTa 모델을 KNER을 위해 세밀하게 조정하는 방법론을 연구하였습니다. 이를 위해 먼저 쿠르드어 말뭉치(Kurdish corpus)를 생성하고, 개조된 모델 아키텍처를 설계하며, 훈련 절차를 구현하였습니다. 모델을 평가하기 위해 다양한 토크나이제이션(tokenization) 방법과 훈련된 모델을 사용하여 수행한 실험 세트도 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, SentencePiece 토크나이제이션 방법으로 세밀하게 조정된 RoBERTa 모델이 기존의 모델에 비해 KNER 성능을 12.8% 향상시킨 것으로 나타났습니다. 이 연구를 통해 KNLP의 새로운 기준을 수립하였으며, 다양한 KNLP 작업에서의 성능을 개선할 수 있는 가능성을 시사합니다.



### AgentPS: Agentic Process Supervision for Multi-modal Content Quality Assurance through Multi-round QA (https://arxiv.org/abs/2412.15251)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 논문에서는 AgentPS라는 새로운 프레임워크를 소개합니다. AgentPS는 멀티모달 대형 언어 모델(MLLMs)에 프로세스 감독을 통합하여 복잡한 논리 구조를 개선합니다. 추가적으로, LLM 생성 라벨을 사용하여 인간 주석을 대체할 수 있는 가능성도 제시하고 있습니다.

- **Technical Details**: AgentPS는 다단계 질문 응답 방식을 통해 과정 중간 정보를 제공하여 MLLM의 추론 능력을 향상시키는 방법을 구현하고 있습니다. MLLM은 비전 인코더, 비전-언어 정렬 프로젝터, 그리고 언어 모델로 구성되어 있으며, 입력 데이터는 이 구조를 통해 처리됩니다. 각 단계에서 질문-답변 쌍이 포함되어 MLLM의 결정을 이끌어내는 데 사용됩니다.

- **Performance Highlights**: AgentPS는 TikTok 플랫폼의 비정상 콘텐츠 분류(UCC)에서 기존 MLLM 보다 F1 점수와 리콜에서 유의미한 성과 향상을 보여주고 있습니다. 또한, LLM 생성 프로세스 라벨을 사용할 경우에도 성능 향상이 유지되는 것을 확인하여, 대규모 산업 응용에 있어 실용적인 확장성을 입증하였습니다.



### LLMs for Literature Review: Are we there yet? (https://arxiv.org/abs/2412.15249)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 문헌 리뷰 작성을 지원하는 새로운 접근 방식을 제안합니다. 특히, 본 연구에서 LLMs의 제로샷(zero-shot) 능력을 활용하여 초록을 기반으로 연관된 연구 결과를 검색하고 리뷰를 작성하는 두 가지 컴포넌트로 작업을 분해합니다. 혁신적인 두 단계 검색 전략과 리랭킹 메커니즘을 도입하여 LLM의 효과를 분석합니다.

- **Technical Details**: 연구는 초록에서 의미 있는 키워드를 추출하고, 외부 지식 기반에 쿼리하여 관련 논문을 검색하는 두 단계 검색 프로세스를 구현합니다. 또한, LLM이 후보 논문들에서 특정 발췌의 관련성을 밝힐 수 있도록 유도하는 프롬프트 기반 리랭킹 기법을 분석합니다. 리뷰 생성을 위해 LLM에게 어떤 논문을 인용할지를 지정하는 계획을 제공하는 방안을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론은 기존의 간단한 LLM 기반 생성 방법에 비해 18-26% 더 적은 환각된 참조(hallucinated references)를 생성하며, 품질 좋은 리뷰를 생성하는데 기여합니다. LLM 기반의 계획 수립 접근법이 문헌 리뷰의 품질을 실질적으로 개선함을 보여줍니다. 대조군보다 10%와 30% 향상된 정밀도와 정규화된 리콜(normalized recall)을 기록했으며, 이러한 방법은 연구 결과에 대한 투명성을 증가시킵니다.



### Accelerating Retrieval-Augmented Generation (https://arxiv.org/abs/2412.15246)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하고 정확도를 향상시키기 위한 솔루션으로 Retrieval-Augmented Generation (RAG) 접근법을 제안합니다. RAG는 LLM과 외부 지식 소스(예: 웹)에서 검색된 정보를 결합하는 방식으로 동작합니다. 저자들은 RAG의 실행 파이프라인을 분석하고, 고품질 검색을 위한 Intelligent Knowledge Store (IKS)라는 새로운 CXL 메모리 확장을 소개합니다.

- **Technical Details**: IKS는 새로운 캐시 일관성 인터페이스를 가진 고성능, 고용량 벡터 데이터베이스 가속기입니다. IKS는 정확한 최근접 이웃 검색(Exact Nearest Neighbors Search, ENNS)을 가속화하여 512GB 벡터 데이터베이스에서 13.4-27.9배 빠른 검색 성능을 제공합니다. 이 시스템은 CPU와 근처 메모리 가속기 간의 효율적인 인터페이스를 구현하여 메모리를 분산시키면서도 성능을 극대화하는 설계를 특징으로 합니다.

- **Performance Highlights**: IKS는 벡터 데이터베이스 애플리케이션에서 1.7-26.3배의 엔드 투 엔드 추론 속도 향상을 이끌어 내며, 이는 RAG 애플리케이션에서 대표적으로 관찰되는 성능 개선입니다. 본 연구는 RAG의 다양한 하드웨어 및 소프트웨어 구성이 성능과 정확도에 미치는 영향을 심도 있게 평가합니다. 논문에서 제시된 IKS는 기존의 메모리 시스템의 한계를 극복하기 위한 중요한 진전을 의미합니다.



### MPPO: Multi Pair-wise Preference Optimization for LLMs with Arbitrary Negative Samples (https://arxiv.org/abs/2412.15244)
Comments:
          Accepted by COLING2025

- **What's New**: 본 연구에서는 Multi Pair-wise Preference Optimization (MPPO) 알고리즘을 도입하여 대량의 언어 모델(LLM)과 사람의 피드백을 효율적으로 정렬할 수 있는 방법을 제안합니다. 기존의 DPO 및 KTO 알고리즘과 달리 MPPO는 보상 모델을 사용하지 않고도 정책 모델을 직접 최적화하여 여러 응답을 효과적으로 활용합니다. 이러한 접근 방식은 모델의 응답 품질을 개선하고, 더 많은 선호 데이터를 최대한 활용할 수 있도록 돕습니다.

- **Technical Details**: MPPO는 모델의 응답 평균 가능성을 활용하여 보상 함수를 피팅(fitting)합니다. 이 연구에서는 Point-wise, Pair-wise, List-wise의 세 가지 주요 구현 방식을 분석한 결과, Pair-wise 방식이 최고의 성능을 달성하는 것을 발견했습니다. 이는 여러 응답을 하나의 쿼리에 대해 최적화하여 희소 데이터 시나리오에서도 효과적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 MPPO는 MT-Bench에서 DPO, ORPO 및 SimPO를 초월하는 성과를 보였으며, Arena-Hard에서는 DPO와 ORPO에 비해 상당한 이점을 나타냈습니다. 이러한 결과는 MPPO가 선호 최적화 작업에서 큰 장점을 보여줌을 강조합니다. MPPO의 실험은 실제 응용 프로그램에서 최적화를 지원하며, 모델 응답의 질을 크게 향상시킵니다.



### Script-Based Dialog Policy Planning for LLM-Powered Conversational Agents: A Basic Architecture for an "AI Therapist" (https://arxiv.org/abs/2412.15242)
Comments:
          9 pages, 5 figures, 1 table

- **What's New**: 이번 연구는 LLM(대형 언어 모델) 기반의 대화 에이전트가 대화 중 치료적 접근 방식을 따를 수 있도록 구성된 새로운 'Script-Based Dialog Policy Planning' 프레임워크를 제안합니다. 이 시스템은 예측 가능한 행동을 통해 감정 지원 대화를 직접적으로 개선할 수 있습니다. 연구자들은 이를 통해 대화의 흐름을 정의하는 '스크립트'를 활용하여 인공지능 치료사가 보다 체계적으로 사용자와 상호작용할 수 있게 합니다.

- **Technical Details**: 연구에서는 LLM 기반의 대화 에이전트가 효과적이고 안전하게 작동할 수 있도록 다섯 가지 핵심 요구 사항을 정의합니다. 이 요구 사항에는 대화 유창성(conversational fluency), 적극성(proactivity), 전문가 개발(expert development), 증거 기반 관행(application of evidence-based practices), 그리고 검사 가능성(inspectability)이 포함됩니다. '스크립트'라는 고정된 텍스트를 사용하여 대화 흐름을 정의하면 LLM의 행동을 제어하고 전문성을 높여보고자 합니다.

- **Performance Highlights**: 연구 결과, 스크립트 기반의 대화 정책 계획을 사용하는 100개 대화의 성과를 시뮬레이션하여 이 접근 방식의 가능성을 입증하였습니다. 각각의 변형이 대화의 효율성과 효과성을 어떻게 달성하는지를 비교할 수 있는 기준을 설정하고, 성과의 강점과 약점을 논의하여 향후 개선 방향을 제시합니다. 이 새로운 기술의 타당성을 강조하며, 인공지능 치료사의 개발에 중요한 발판을 마련하고자 합니다.



### Quantifying Positional Biases in Text Embedding Models (https://arxiv.org/abs/2412.15241)
Comments:
          13 pages, 11 figures, NeurIPS

- **What's New**: 이번 연구는 정보 검색(Information Retrieval, IR)과 의미 유사성 측정에서 중요하게 사용되는 embedding 모델의 한계, 특히 긴 텍스트와 관련된 위치 편향 처리에 대해 다룹니다. content position과 input size가 text embedding에 미치는 영향을 실험을 통해 조사하였으며, embedding 모델들이 입력의 시작 부분을 불균형적으로 우선시하는 경향을 발견했습니다. 특히 문서의 처음에 무관한 텍스트를 삽입하거나 삭제하는 실험을 통해, 이러한 변화가 cosine similarity에 미치는 영향을 정량화하였습니다.

- **Technical Details**: embedding 모델은 transformer encoder 아키텍처를 기반으로 하여 bidirectional self-attention block을 사용합니다. 이 모델들은 고정된 길이의 벡터를 생성하여 전체 입력 텍스트를 표현하며, cosine similarity를 사용해 embedding을 비교합니다. 연구는 Absolute Positional Embedding(APE)과 Rotary Positional Embedding(RoPE), Attention with Linear Biases(ALiBi)와 같은 다양한 positional encoding 기법을 검토하여 이들이 모델의 내재적 편향에 어떻게 기여하는지를 설명합니다.

- **Performance Highlights**: 모델의 문서간 cosine similarity 측정 결과, 텍스트의 처음 부분에 무관한 삽입이 시뮬레이션된 경우, 유사성 감소가 중간이나 끝에 삽입한 경우보다 평균 8.5% 및 12.3% 더 크게 나타났습니다. 문서의 초기 문장에서 멀어질수록 회귀 계수가 크게 감소하는 것을 통해, 모델이 초기 내용에 불균형적으로 가중치를 부여한다는 점을 확인했습니다. 이러한 발견은 실제 검색 시스템의 민감도를 정량화하며, 모델의 강건성 향상 방향에 대한 새로운 관점을 제시합니다.



### ChainStream: An LLM-based Framework for Unified Synthetic Sensing (https://arxiv.org/abs/2412.15240)
Comments:
          18 pages, 8 figures

- **What's New**: 본 연구는 자연어를 개인 데이터 접근 및 처리의 단일 인터페이스로 사용하여 컨텍스트 인식 애플리케이션의 개발을 용이하게 하는 방안을 제안합니다. 이를 통해 개발자는 새로운 API를 배우지 않고도 자연어로 컨텍스트 인식 프로그램을 구축할 수 있으며, 사용자는 개발자가 쓴 자연어 데이터 쿼리를 직접 읽고 허가 관련 결정을 내릴 수 있어 데이터 처리 파이프라인의 투명성을 높입니다. 이러한 접근 방식은 최종 사용자 프로그래밍을 가능하게 하고, 더 정교한 사용 시나리오를 유도할 수 있습니다.

- **Technical Details**: 우리는 자연어 기반의 컨텍스트 인식 프로그램 생성을 가능하게 하기 위해 양방향 접근 방식을 취합니다. 첫 번째 방향은 감지 프로그램을 간단하고 통합된 형태로 만들기 위한 스트림 스타일의 프로그래밍 프레임워크를 도입하였고, 두 번째 방향은 샌드박스 피드백에 의해 유도된 쿼리 최적화를 통해 자연어 쿼리를 더 유익하게 만드는 것입니다. 이를 통해 자연어 지시와 실제 컨텍스트 인식 코드 간의 간격을 효과적으로 줄이고, Stream 데이터 추상화를 기초로 다양한 규칙 기반 및 모델 기반 스트림 데이터 작업을 위한 함수를 설계했습니다.

- **Performance Highlights**: 우리는 133개의 컨텍스트 인식 작업을 포함한 벤치마크를 만들어 우리 접근 방식을 평가하였습니다. 평가 결과, 우리의 방법은 생성 품질에서 최고점을 달성하며, 기존 방법들을 약 33% 초과하는 성능을 보였습니다. 본 연구는 자연어로 정의된 컨텍스트 인식을 위한 첫 번째 연구로, 실행 가능한 감지 프로그램으로의 자연어 요청 변환을 위한 엔드-투-엔드 시스템을 구축했습니다.



### Modeling Story Expectations to Understand Engagement: A Generative Framework Using LLMs (https://arxiv.org/abs/2412.15239)
- **What's New**: 기존 데이터 분석의 한계를 극복하고 고객이 이야기와 어떻게 상호작용하는지를 이해하기 위한 새로운 프레임워크가 제안되었습니다. 이 연구는 고객이 이야기의 전개에 대한 기대와 불확실성을 모델링하여 내용의 참여를 예측하는 데 중점을 둡니다. 그 결과, 이야기의 예상되는 진행에 대한 다양성을 포함한 새로운 지표를 통해 효과적인 추천 및 마케팅 전략 수립이 가능해집니다.

- **Technical Details**: 제안된 방법은 크게 두 단계로 이루어져 있습니다: 이야기 상상 단계와 특성 추출 단계입니다. 이야기 상상 단계에서는 미리 훈련된 대규모 언어 모델(LLM)을 사용하여 초기 텍스트를 기반으로 여러 개의 가능한 이야기 진행을 생성합니다. 그런 다음, 이러한 상상된 이야기에서 기대, 불확실성 및 놀람과 관련된 특성을 추출하여 기존의 이야기에 기반한 특성 추출 기법을 보완합니다.

- **Performance Highlights**: 이 연구는 Wattpad에서 수집한 30,258개의 책 챕터에 이 방법을 적용하여, 제안된 프레임워크가 기존 특성 엔지니어링 기법보다 평균 31% 더 높은 설명력을 제공한다는 것을 입증했습니다. 내부 회귀 분석을 통해 고객의 기대치가 이야기에 대한 참여도에 미치는 영향을 알고리는 초기 단서를 발견할 수 있으며, 이 결과는 마케팅 전략 및 콘텐츠 개발에 유용한 통찰을 제공합니다.



### Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks (https://arxiv.org/abs/2412.15238)
Comments:
          Accepted to NeurIPS 2024 Workshop on Foundation Model Interventions (MINT)

- **What's New**: 이번 연구에서는 기존의 LLM(대형 언어 모델)에서 발생하던 추론 과제와 관련된 약점을 해결하기 위한 새로운 프레임워크인 Dipper를 제안합니다. Dipper는 학습이 필요 없는 LLM 앙상블(ensemble) 방식으로, 단일 LLM 모델에 최적화된 다양한 프롬프트를 동시에 제공하여 추론 시간에 성능을 개선하도록 설계되었습니다. 이를 통해 사용자는 통합된 방식으로 여러 개의 쿼리를 처리할 수 있게 됩니다.

- **Technical Details**: 연구에서는 Dipper 프레임워크를 통해 각기 다른 프롬프트를 동시에 입력하여 LLM의 출력을 다양화하고자 합니다. 이 방법은 LLM의 고유한 특성인 다양한 출력을 생성하는 능력을 활용하여 성능 향상을 꾀합니다. 특히, 동종 모델을 사용하여 모델의 다양성을 증대시키는 것이 특징입니다.

- **Performance Highlights**: Dipper를 통해 한정된 GPU 메모리 환경에서도 성능을 향상시키는 실험 결과가 도출되었습니다. 예를 들어, MATH 데이터세트에서 3개의 작은 모델(Qwen2-MATH-1.5B-it 모델들)로 구성된 앙상블이 보다 큰 모델(Qwen2-MATH-7B-it)을 능가하는 결과를 보여주었습니다. 이는 Dipper가 실제 문제 해결에서 매우 유용하다는 것을 시사합니다.



### CareBot: A Pioneering Full-Process Open-Source Medical Language Mod (https://arxiv.org/abs/2412.15236)
Comments:
          Accept by AAAI 2025

- **What's New**: 최근 Closed-source LLM들과 오픈소스 커뮤니티들이 큰 발전을 이루어내며 인간의 성능을 초월하고 있지만, 의료처럼 전문적인 도메인에서는 여전히 만족스럽지 않은 성능을 보이고 있습니다. 본 논문에서는 CareBot이라는 이중 언어 의료 LLM을 제안하며, 이는 Continuos Pre-Training (CPT), Supervised Fine-Tuning (SFT), 그리고 Reinforcement Learning with Human Feedback (RLHF)를 통합하여 개발되었습니다. 특히, 일반 데이터와 도메인 특정 데이터의 간극을 메우기 위한 새로운 두 단계 CPT 방법론을 도입합니다.

- **Technical Details**: 이 논문에서 제안된 CareBot LLM은 LLaMA3-8B 기반으로 하며, 의사에게 진단 보조, 개인 맞춤형 치료 계획 제공, 의료 교육 지원 등을 효과적으로 지원할 수 있도록 설계되었습니다. 특히, Stable CPT와 Boost CPT라는 두 단계의 CPT 방법론을 통해 일반 데이터와 도메인 특정 데이터 간의 분포 불일치를 해결하며, 데이터 품질 평가 모델인 DataRater를 개발하여 CPT 동안의 학습 데이터의 정확도와 관련성을 보장합니다. 이는 다큐멘터리한 의료 SFT 데이터셋과 다회차 대화 품질 향상을 위한 ConFilter 메트릭을 포함합니다.

- **Performance Highlights**: CareBot의 성능 평가는 중국어 및 영어 기준에서 진행되었으며, 의료 상담 및 교육에서 뛰어난 성능을 보였습니다. 실험 결과, CareBot이 다수의 의료 애플리케이션에서 우수한 성능을 발휘하며, 본 연구의 데이터셋 구성과 훈련 전략이 모델 성능에 긍정적인 영향을 미쳤음을 확인했습니다. 이러한 발전은 의료 LLM의 현재 한계를 극복하고, 오픈소스 모델의 신뢰성과 효과성을 위한 새로운 기준을 설정하는 데 기여할 것입니다.



### OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models (https://arxiv.org/abs/2412.15235)
- **What's New**: OG-RAG(온톨로지 기반 검색 향상 생성) 방법은 LLM 생성 응답의 품질을 향상시키기 위해 도메인 특정 온톨로지에 기반한 검색 프로세스를 통합합니다. 이 방법은 일반적인 검색 증강 모델들의 한계, 즉 구조화된 도메인 지식을 반영하지 못하는 점을 해결하고, 특히 전문화된 지식이 요구되는 산업 워크플로우 및 의사결정 단계에서 유용합니다.

- **Technical Details**: OG-RAG는 도메인 문서의 하이퍼그래프 표현을 구성하고, 각 하이퍼에지는 도메인 특정 온톨로지를 기반으로 한 사실 지식의 클러스터를 포괄합니다. 이 최적화 알고리즘은 LLM을 위한 정확하고 개념적으로 기반을 둔 맥락을 형성하기 위해 최소한의 하이퍼에지를 검색합니다. 이를 통해 복잡한 엔티티 간의 관계를 유지하면서도 효율적인 검색을 가능하게 합니다.

- **Performance Highlights**: OG-RAG는 4개의 다른 LLM에서 정확한 사실의 리콜을 55% 향상시키고, 응답의 정확성을 40% 증가시켰습니다. 또한, LLM 응답을 맥락과 빠르게 연결하는 데 30% 더 빠르며, 사실 기반 추론의 정확성을 다른 기법에 비해 27% 향상시킵니다. 이러한 성과는 OG-RAG의 전문화된 워크플로우에서의 신뢰성 있는 사실 기반 응답 제공 능력을 강조합니다.



### Learning-by-teaching with ChatGPT: The effect of teachable ChatGPT agent on programming education (https://arxiv.org/abs/2412.15226)
- **What's New**: 이 연구는 ChatGPT를 교육용 에이전트(teachable agent)로 활용하여 학생들의 프로그래밍 교육을 지원하는 가능성을 조사했습니다. 전통적인 교육용 에이전트의 한계를 극복하기 위해, ChatGPT의 자연어 대화 기능을 통한 학습 지원 효과를 탐구했습니다. 연구 결과, ChatGPT와의 상호작용이 학생들의 지식 습득과 프로그래밍 능력을 향상시켰으며, 특히 가독성이 높고 논리적인 코드 작성에 도움이 되었다고 보고되었습니다.

- **Technical Details**: 학생들이 ChatGPT와 상호작용함으로써 프로그래밍 능력과 자기 조절 학습(Self-Regulated Learning, SRL) 능력이 개선되었습니다. 특히, ChatGPT는 자연어 입력을 처리하고 인간 유사 반응을 생성하는 능력을 갖추고 있어 전통적인 교육용 에이전트와의 차별성을 보였습니다. 그러나 ChatGPT가 주로 올바른 코드를 생성하기 때문에 디버깅 능력 개발에 제한적 영향을 미쳤다고 밝혀졌습니다.

- **Performance Highlights**: 연구는 ChatGPT가 학생들의 자기 효능감과 SRL 전략 이행 능력을 높이는 데 기여함을 보여주었습니다. 학습자들은 ChatGPT와의 상호작용을 통해 지식을 가르치고 반영하는 과정에서 더 심도 깊은 이해를 얻게 되었습니다. 전반적으로, 이 연구는 ChatGPT가 교육 지원에서의 역할에 대한 통찰력을 제공하며, 향후 연구 방향에 대한 기초를 마련했습니다.



### A Survey on Large Language Model-based Agents for Statistics and Data Scienc (https://arxiv.org/abs/2412.14222)
- **What's New**: 최근 몇 년 동안, LLM(대규모 언어 모델)에 의해 구동되는 데이터 에이전트가 전통적인 데이터 분석 패러다임을 변형할 수 있는 잠재력을 보여주었습니다. 이러한 설문조사는 LLM 기반 데이터 에이전트의 진화, 기능 및 응용 프로그램을 개괄적으로 설명하며, 복잡한 데이터 작업을 간소화하고 관련 전문지식이 없는 사용자도 쉽게 접근할 수 있도록 돕는 역할을 강조합니다.

- **Technical Details**: 우리는 LLM 기반 프레임워크의 설계에서 현재의 트렌드를 탐구하며, 계획(planning), 추론(reasoning), 반성(reflection), 다중 에이전트 협업(multi-agent collaboration), 사용자 인터페이스(user interface), 지식 통합(knowledge integration), 시스템 설계(system design)와 같은 필수 기능을 구체적으로 설명합니다. 이러한 기능들은 에이전트가 최소한의 인간 개입으로 데이터 중심 문제를 해결할 수 있게 해줍니다.

- **Performance Highlights**: 우리는 다양한 실제 사례 연구를 분석하여 여러 데이터 에이전트의 실제 응용 프로그램을 시연합니다. 마지막으로, 데이터 에이전트의 발전을 위한 주요 도전 과제를 확인하고, 지능형 통계 분석 소프트웨어로 발전시키기 위한 미래 연구 방향을 제안합니다.



### Sum-of-Squares Programming for Ma-Trudinger-Wang Regularity of Optimal Transport Maps (https://arxiv.org/abs/2412.13372)
- **What's New**: 이 논문은 MTW 텐서의 비부정성을 인증할 수 있는 새로운 계산적 접근법을 제안합니다. 기존의 방법들이 특정 최적 수송 문제에만 적용될 수 있는 반면, 제안된 방법은 일반적인 비용 함수에 대해 널리 적용됩니다. 특히, Sum-of-Squares (SOS) 프로그래밍을 사용하여 MTW 텐서의 비부정성 증명 절차를 개발하였습니다.

- **Technical Details**: 논문에서는 Monge 최적 수송 문제의 정규성(regularity) 이론을 다루고 있으며, MTW 조건을 통해 그 정규성을 분석합니다. MTW 텐서는 주어진 지상 비용 함수의 곡률(curvature) 개념을 제공하며, 이 텐서의 비부정성은 Monge OT 맵의 연속성을 확립하는 데 중요한 역할을 합니다. 제안된 계산적 접근법은 특히 비상 수학적 함수(semialgebraic function)에 대해 적용 가능합니다.

- **Performance Highlights**: 제안된 SOS 프로그래밍 방법은 여러 실제 지상 비용 함수에 적용되어 최적 수송 맵의 정규성 영역을 근사하는 데 성공했습니다. 이를 통해 MTW 텐서의 비부정성을 검증할 수 있으며, 이는 기계 학습 분야와 최적화 연구 커뮤니티에 큰 기여를 할 것으로 기대됩니다. 새로운 인증 절차와 정규성 이론의 발전은 이 분야의 문제 해결에 중요한 역할을 할 것입니다.



### TACNET: Temporal Audio Source Counting Network (https://arxiv.org/abs/2311.02369)
- **What's New**: 본 논문에서는 오디오 소스 카운팅 작업의 한계를 해결하기 위해 독창적인 아키텍처인 Temporal Audio Source Counting Network (TaCNet)를 소개합니다. TaCNet는 원시 오디오 입력에서 직접 작동하여 복잡한 전처리 단계( preprocessing steps)를 제거하고, 워크플로우를 간소화합니다. 특히, 단편화된 입력 윈도우를 사용하는 경우에도 실시간 화자 카운팅에서 뛰어난 성능을 발휘합니다.

- **Technical Details**: TaCNet는 LibriCount 데이터셋을 사용하여 평가되었으며, 11개 클래스에서 평균 74.18%의 정확도를 보여주었습니다. 이 네트워크는 다양한 시나리오에 대해 효과적임을 입증하며, 중국어 및 페르시아어와 같은 다양한 언어에 대한 적용 가능성을 보여줍니다. 또한, TaCNet는 오디오 소스 카운팅 작업을 위한 최첨단 솔루션으로 자리 잡고 있습니다.

- **Performance Highlights**: TaCNet의 평가 결과는 그 뛰어난 성능을 강조하며, 다양한 환경에서의 활용 가능성을 제시합니다. 특히, cross-lingual adaptability(교차 언어 적응성)는 TaCNet의 다재다능성을 보여주며, 글로벌한 오디오 소스 카운팅 적용이 가능함을 나타냅니다.



New uploads on arXiv(cs.LG)

### Offline Reinforcement Learning for LLM Multi-Step Reasoning (https://arxiv.org/abs/2412.16145)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 다단계 추론 능력을 개선하기 위한 오프라인 강화 학습(offline reinforcement learning) 방법인 OREO(Offline Reasoning Optimization)를 제안합니다. 기존의 Direct Preference Optimization(DPO) 방법의 한계점을 극복하고, 다단계 추론에 적합한 새로운 접근 방식을 제공합니다.

- **Technical Details**: OREO는 최대 엔트로피 강화 학습(maximum entropy reinforcement learning)의 통찰을 바탕으로 소프트 벨만 방정식(soft Bellman Equation)을 최적화하여 정책 모델(policy model)과 가치 함수(value function)를 공동 학습합니다. 이는 데이터 수집의 부담을 줄이고, 다단계 추론에서 효과적인 신용 할당(credit assignment)을 가능하게 합니다.

- **Performance Highlights**: OREO는 수학적 추론 작업(GSM8K, MATH) 및 임베디드 에이전트 제어(ALFWorld)와 같은 다단계 추론 벤치마크에서 기존의 오프라인 학습 방법들을 능가하는 성능을 보였습니다. 이 방법은 추가 자원이 있을 경우 다중 반복(multi-iteration) 프레임워크로 확장할 수 있으며, 학습된 가치 함수를 통해 무료로 트리 탐색(tree search)을 안내하여 테스트 시 성능을 더욱 향상시킬 수 있습니다.



### FedGAT: A Privacy-Preserving Federated Approximation Algorithm for Graph Attention Networks (https://arxiv.org/abs/2412.16144)
- **What's New**: 본 논문에서는 Federated Graph Attention Network (FedGAT) 알고리즘을 소개합니다. 이 알고리즘은 반지도(node classification) 문제를 해결하기 위해 설계되었으며, GATs(그래프 주의 네트워크)의 동작을 근사하여 통신 오버헤드를 크게 줄입니다. FedGAT는 단 한 번의 사전 훈련 통신 라운드만 필요하다는 점에서 혁신적입니다.

- **Technical Details**: FedGAT는 GAT 모델의 특성을 재현하면서 근사 오류에 대한 증명된 경계를 가집니다. 이 알고리즘은 수학적으로 분석되어 통신 오버헤드와 계산 복잡성을 잘 관리합니다. 또한, 가장 중요한 것은 GATs에서 요구되는 추가 정보가 훈련 라운드 간에 변하기 때문에 선행 통신이 불가능한 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, FedGAT는 중앙집중형 설정에서 GAT 모델과 거의 동일한 정확도를 달성했습니다. 또한, 클라이언트 수나 데이터 배포 방식에 관계없이 성능이 견고하다는 것을 보여줍니다. 이는 Federated Learning 환경에서 GATs를 활용할 수 있는 가능성을 제시합니다.



### EF-Net: A Deep Learning Approach Combining Word Embeddings and Feature Fusion for Patient Disposition Analysis (https://arxiv.org/abs/2412.16134)
Comments:
          Accepted to ICCIT2024

- **What's New**: 본 연구에서는 EF-Net이라는 예측 모델을 개발하여 응급실(ED)에서의 환자 처치를 향상하고자 합니다. 이는 복합적인 환자 처치 정보를 예측하기 위해 카테고리 데이터를 신경망에 통합하고, 숫자적 특성을 결합하여 최적의 결과를 도출합니다. 또한, EF-Net과 XGBoost 모델을 결합하여 향상된 정확성을 달성하였습니다.

- **Technical Details**: EF-Net 모델은 카테고리와 수치적 특성을 결합한 혼합 모델로, 멀티클래스 환자 처치 예측을 목표로 합니다. 실험 결과 EF-Net은 기존 모델들을 앞서며 정확도 95.33%, 앙상블 모델에서는 96%의 정확도를 달성하였습니다. 이 모델은 특히 MIMIC-IV-ED 데이터셋에서 AUROC 및 F1-Score의 측면에서도 뛰어난 성능을 보입니다.

- **Performance Highlights**: EF-Net은 유의미한 예측 성능을 보여주며, 특히 여러 처치 결과를 동시에 예측할 수 있는 능력을 실증하였습니다. 본 연구의 결과는 기존 연구들과 비교하여 데이터 처리 측면에서 우수성을 인정받았으며, 의사결정 지원 시스템(Core Decision Support System)이 환자의 처치를 보다 효과적으로 도울 수 있는 가능성을 제시합니다.



### Deciphering the Underserved: Benchmarking LLM OCR for Low-Resource Scripts (https://arxiv.org/abs/2412.16119)
- **What's New**: 이번 연구는 특히 GPT-4o와 같은 대규모 언어 모델(LLM)의 잠재력을 탐구하며, 저자원 스크립트(우르두어, 알바니아어, 타지크어)에서의 광학 문자 인식(OCR) 성능을 분석합니다. 2,520개의 이미지를 포함한 데이터셋을 기반으로 할 때 다양한 실제 도전 과제를 모사했으며, LLM 기반 OCR의 한계를 강조하였습니다. 특히 복잡한 언어적 특성을 가진 스크립트에 대한 주석이 달린 데이터셋과 세밀하게 조정된 모델의 필요성을 나타내며, 접근 가능성 격차를 해소하기 위한 긴급한 과제를 부각시킵니다.

- **Technical Details**: 이 연구의 데이터셋은 우르두어, 영어, 알바니아어, 타지크어를 포함하여 각각의 언어에서 30개의 기사로 제한된 단어 수 범위를 기준으로 총 2,520개의 이미지를 생성했습니다. 텍스트는 다양한 글꼴 크기(12pt, 18pt, 24pt)와 배경 색상, 블러 등을 활용하여 각각의 언어적 특성과 시각적 조건의 변화를 반영했습니다. 그러한 방법을 통해, 복잡한 스크립트 내에서의 OCR 성능을 평가하고자 했습니다.

- **Performance Highlights**: 연구 결과는 언어적 복잡성이 큰 스크립트에서 zero-shot LLM 기반 OCR의 한계를 보여주었으며, 이는 주석 데이터셋의 필요성을 강조합니다. 이러한 성과는 기존의 OCR 모델에 비해 LLM이 특정 맥락에서 더 나은 성능을 발휘할 수 있다는 가능성을 부각시키며, 향후 저자원 스크립트의 디지털화와 접근성 개선을 위한 기초 자료를 제공합니다.



### Explainable AI for Multivariate Time Series Pattern Exploration: Latent Space Visual Analytics with Time Fusion Transformer and Variational Autoencoders in Power Grid Event Diagnosis (https://arxiv.org/abs/2412.16098)
- **What's New**: 이 논문은 복잡한 패턴을 시각적으로 분석하는 새로운 프레임워크를 제안합니다. Time Fusion Transformer (TFT)와 Variational Autoencoders (VAEs)의 두 가지 생성 AI 모델을 통합하여 다변량 시계열 데이터의 복잡한 패턴을 저차원 잠재 공간으로 축소합니다. 이를 통해 PCA, t-SNE 및 UMAP과 같은 차원 축소 기술을 사용하여 2D로 시각화함으로써 데이터 패턴을 직관적으로 탐색할 수 있도록 합니다.

- **Technical Details**: 제안된 시각 분석 프레임워크는 복잡한 시간적 패턴의 유사성을 식별하고 잠재적 상관관계를 발견하는 데 중점을 둡니다. 다양한 모델 구성에서 TFT와 VAE의 성능을 평가하는 독특한 메트릭을 도입하며, 이로 인해 모델 매개변수 조정 및 신뢰성 향상에 도움을 줍니다. 특히, TFT는 다양한 시계열 데이터 형태에 대해 뛰어난 확장성과 짧은 실행 시간을 보여줍니다.

- **Performance Highlights**: TFT와 VAE 기반 방법의 비교 분석은 두 모델 간의 2D 잠재 벡터 표현의 일관성이 86%-92%에 달함을 밝혀냈습니다. TFT는 데이터 형태의 다양성에 대해 VAE보다 실행 시간과 확장성에서 더 우수한 성능을 보였습니다. 이 연구는 다변량 시계열 데이터의 고장 진단을 발전시키고, 의사결정 과정에서 설명 가능한 AI 접근 방식을 촉진하는 데 기여합니다.



### Differentially Private Federated Learning of Diffusion Models for Synthetic Tabular Data Generation (https://arxiv.org/abs/2412.16083)
Comments:
          9 pages, 9 figures, preprint version, currently under review

- **What's New**: 이번 논문에서는 DP-Fed-FinDiff라는 새로운 프레임워크를 소개하여, Differential Privacy (DP), Federated Learning (FL) 및 Denoising Diffusion Probabilistic Models (DDPMs)를 통합하여 높은 품질의 합성 데이터 생성을 가능하게 합니다. 이 프레임워크는 까다로운 개인정보 보호 규정을 준수하면서도 데이터 유용성을 유지합니다. 이를 통해 연구자와 기관 간의 협력이 촉진되고, 규제에 부합하는 데이터 공유가 가능해집니다.

- **Technical Details**: DP-Fed-FinDiff는 민감 데이터 보호를 위해 설계되었습니다. 이 프레임워크는 다양한 금융 데이터 세트에서 실질적인 개선을 보여주며, 경량의 프라이버시 예산을 설정하고 조정할 수 있는 기능을 제공합니다. 특히 이 연구에서는 분산화된 데이터에서의 민감 정보 노출을 방지하는 Differential Privacy의 중요성에 대해 강조하고 클라이언트 구성 및 연합 최적화 전략 간의 최적의 균형을 찾기 위한 실험적 평가를 수행했습니다.

- **Performance Highlights**: DP-Fed-FinDiff는 여러 실제 금융 데이터 세트를 통해 높은 품질의 합성 데이터를 생성하며, 프라이버시 보장을 크게 향상시킨다는 것을 입증했습니다. 이 프레임워크는 규제된 분야에서 안전한 데이터 공유 및 강력한 분석을 가능하게 하는 잠재력을 보여줍니다. 논문에서 제시된 결과는 합성 데이터의 고품질화를 위한 새로운 방향성을 제시하며, 연합 학습 및 프라이버시 보존 데이터 합성 분야의 발전을 이끌어 나갈 수 있습니다.



### Fair Distributed Machine Learning with Imbalanced Data as a Stackelberg Evolutionary Gam (https://arxiv.org/abs/2412.16079)
- **What's New**: 본 논문에서는 분산 학습 환경에서 데이터 불균형 문제를 해결하기 위해 Stackelberg 게임 이론을 적용하는 새로운 접근 방식을 제시합니다. 개발된 두 가지 알고리즘인 결정적 Stackelberg 가중치 모델(DSWM)과 적응형 Stackelberg 가중치 모델(ASWM)은 각 노드의 기여도를 설정하는데 사용됩니다. 이를 통해, 적은 데이터를 가진 노드의 성능을 개선하여 전체 모델의 품질을 높이려는 목표를 가지고 있습니다.

- **Technical Details**: 이번 연구에서는 세 개의 의료 데이터셋을 활용하여 ASWM이 적게 대표되는 노드의 성능을 2.713% 향상시키는 결과를 보였습니다. DSWM과 ASWM은 각각 결정론적 방법과 신경망을 통해 기여도를 예측하는 방법을 사용하며, 이를 통해 불균형한 데이터 분포 문제를 해결합니다. Stackelberg 게임 이론을 적용함으로써 자원 allocation과 참여 모델링이 더욱 효율적으로 이루어질 수 있습니다.

- **Performance Highlights**: ASWM이 대규모 데이터셋을 가진 노드에서는 평균적으로 0.441%의 성능 저하를 보이지만, 적게 대표되는 노드는 성능 개선을 경험할 수 있음을 보여줍니다. 이 연구의 결과는 분산 학습 환경에서 데이터 불균형 문제가 효과적으로 해결될 수 있는 가능성을 제시합니다. 이러한 동적 가중치 조정이 분산 학습의 정확도와 공정성을 더욱 높일 수 있음을 확인하였습니다.



### Choose Your Explanation: A Comparison of SHAP and GradCAM in Human Activity Recognition (https://arxiv.org/abs/2412.16003)
- **What's New**: 이 연구에서는 기계 학습 모델을 설명하기 위한 두 가지 주요 방법인 SHAP(Shapley Additive Explanations)와 GradCAM(Gradient-weighted Class Activation Mapping)을 비교 분석했습니다. 특히 인체 활동 인식(HAR) 분야에 대한 초점을 맞추어 그래픽 컨볼루션 네트워크(GCN)를 활용하여 실제 데이터셋에서 이들 방법의 강점과 한계를 평가하였습니다. 연구 결과는 사용자들이 특정 모델 및 애플리케이션에 가장 적합한 설명 방법을 선택하는 데 도움을 줄 수 있습니다.

- **Technical Details**: SHAP는 입력 기능에 대한 기여도를 자세히 설명하는 반면, GradCAM은 공간적으로 지향된 설명을 제공하는 방식으로 서로 보완하는 특성을 가집니다. 연구는 두 개의 실제 데이터셋에서 스켈레톤 기반 데이터에 대해 두 방법을 정량적 및 정성적으로 비교하여, 각 방법의 특징 중요도 순위, 해석 가능성 및 모델 민감도를 평가하였습니다. 이러한 분석은 의료와 같은 고위험 환경에서 신뢰성과 투명성을 증대시키는 데 필수적입니다.

- **Performance Highlights**: SHAP는 기능의 중요성을 상세히 기록하지만, GradCAM은 빠르고 간략한 설명을 제공합니다. 연구 결과, SHAP은 더 깊은 통찰을 제공하지만 공간적 및 시간적 동적 패턴을 캡처하는 데 있어 한계를 가집니다. GradCAM은 모델의 중요한 영역을 강조하지만 개별 입력 기능의 기여를 설명하는 데 있어 부재가 있습니다. 따라서 두 방법의 비교는 향후 HAR 모델의 신뢰성을 높이기 위한 중요한 요소가 될 것입니다.



### CNN-LSTM Hybrid Deep Learning Model for Remaining Useful Life Estimation (https://arxiv.org/abs/2412.15998)
Comments:
          conference paper

- **What's New**: 본 연구에서는 Remaining Useful Life (RUL) 예측을 위한 하이브리드 접근 방식을 제안합니다. 전통적인 회귀 방법들이 RUL 추정에서 낮은 정확도를 보였던 반면, CNN과 Long Short-Term Memory (LSTM) 네트워크를 결합하여 특징을 효율적으로 추출하고 RUL을 예측합니다. 이는 예측 유지보수 프로세스에서 RUL 추정을 위한 CNN-LSTM 모델을 적용한 최초의 시도로 평가받습니다.

- **Technical Details**: 하이브리드 CNN-LSTM 모델에서 처음에는 CNN이 데이터를 통해 특징을 효율적으로 추출하고, 그 후 LSTM이 이러한 추출된 특징을 사용하여 RUL을 예측합니다. 이 방법은 멀티 변량 시계열 분석을 통해 센서 시퀀스 정보를 활용하여 숨겨진 패턴을 찾아내며, 다양한 운영 조건 및 결함 시나리오에서도 견고한 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 하이브리드 CNN-LSTM 모델이 가장 높은 정확도를 달성하였으며, 다른 방법들과 비교했을 때 우수한 성과를 보였습니다. 이 연구는 RUL 예측 분야에서 새로운 가능성을 제시하며, 미래의 예측 유지보수 애플리케이션에 중요한 기여를 할 것으로 기대됩니다.



### Never Reset Again: A Mathematical Framework for Continual Inference in Recurrent Neural Networks (https://arxiv.org/abs/2412.15983)
- **What's New**: 이번 연구에서는 RNNs(순환 신경망)의 성능 향상과 지속적인 추론(continual inference) 시의 한계를 극복하기 위한 적응형 손실 함수(adaptive loss function)를 제안합니다. 기존의 리셋 방식이 요구하는 복잡성을 제거하고, 입력 데이터의 정보량에 따라 동적으로 기울기(gradient)를 조절하여 정확도를 유지합니다. 이를 통해 RNN의 계속적 과제 수행 능력이 크게 향상되었습니다.

- **Technical Details**: RNN의 동작 원리를 이해하기 위해, 입력 시퀀스의 변화가 어떻게 RNN의 역동성을 저해하는지를 수학적으로 분석했습니다. 이 연구는 Kullback-Leibler divergence와 교차 엔트로피(cross-entropy)를 결합한 손실 함수의 개발을 통해, 기밀성 유지와 연속적인 출력 조정이 가능하도록 했습니다. RNN의 숨겨진 상태의 연속성을 보장하면서도 동적 학습(dynamically modulated learning)을 이뤄내는 것이 핵심입니다.

- **Performance Highlights**: 실험 결과, 제안한 리셋 없는 접근법(reset-free approach)은 전통적인 리셋 기반 방법에 비해 지속적인 작업에서 우수한 성능을 나타냅니다. 특히, 음성 인식 및 스트리밍 작업과 같은 분야에서 RNN이 유지하는 정보의 연속성이 향상되어, 실제 애플리케이션에서 신뢰성이 높아졌습니다. 이러한 개선은 RNN의 이론적 및 실용적 활용 가능성을 모두 증대시킵니다.



### Black-Box Uniform Stability for Non-Euclidean Empirical Risk Minimization (https://arxiv.org/abs/2412.15956)
Comments:
          33 pages, no figures

- **What's New**: 이번 연구에서는 $p$-norm ($p \geq 1$)에 대해 볼록하고 원활한 경험적 위험 최소화(empirical risk minimization, ERM) 문제에 대한 일관된 안정성을 가진 첫 번째 차수 알고리즘을 살펴봅니다. 우리는 균일 볼록 정규화기(uniformly convex regularizers)의 속성을 활용하여, Hölder 부드럽고 볼록한 손실을 최적화하는 알고리즘을 균일 안정성을 갖는 학습 알고리즘으로 전환하는 블랙 박스 감소 방법을 제안합니다.

- **Technical Details**: 연구에서 제안한 방법은 균일한 안정성을 달성하는 블랙 박스 감소를 통해 통계적 위험 경계를 최적화하고, 특히 $p$에 따라 상수 인자에 의존하는 초과 위험(excess risk)을 고려합니다. 이는 (Attia와 Koren, 2022)에서 유클리드 공간($p=2$)의 경우에 대한 문제를 해결한 이후 제기된 질문이었습니다.

- **Performance Highlights**: 이 기법은 비유클리드 기하학(non-Euclidean geometry)을 활용한 이진 분류(binary classification) 문제에 적용할 수 있는 방법들을 탐구하며, 새로운 알고리즘이 통계적 안정성이 높은 성과를 달성할 수 있음을 보여줍니다.



### RiTTA: Modeling Event Relations in Text-to-Audio Generation (https://arxiv.org/abs/2412.15922)
Comments:
          Audio Events Relation Modeling in TTA Generative Model. Code: this https URL

- **What's New**: 본 논문은 텍스트에서 오디오로(Text-to-Audio, TTA) 생성 모델에서 오디오 이벤트 간의 관계 모델링을 체계적으로 연구합니다. 기존의 TTA 방법들은 이러한 관계 모델링을 체계적으로 탐구하지 않았으며, 이를 개선할 수 있는 프레임워크를 제안하지 않았습니다. 이 연구에서는 새로운 평가 지표와 함께 포괄적인 관계 코퍼스 및 일반적으로 들리는 오디오 이벤트 코퍼스를 제안합니다.

- **Technical Details**: 연구자는 오디오 이벤트 관계 모델링을 위한 벤치마크를 설정했습니다. 이를 위해, 1) 실제 시나리오에서의 모든 잠재적 관계를 포괄하는 관계 코퍼스를 제안하고, 2) 흔히 들리는 오디오를 포함한 새로운 오디오 이벤트 코퍼스를 도입했습니다. 마지막으로, 오디오 이벤트 관계 모델링을 다양한 관점에서 평가하기 위한 새로운 평가 지표를 제안합니다.

- **Performance Highlights**: 작성된 논문에서는 기존 TTA 모델이 오디오 이벤트 관계를 더 잘 모델링할 수 있도록 하는 파인튜닝(참고: finetuning) 프레임워크도 제안합니다. 이를 통해 모델의 성능을 향상시키고, 오디오 이벤트 간의 관계를 효과적으로 처리할 수 있도록 합니다. 또한, 해당 논문의 코드도 공개되어 있어 실질적인 연구에 활용될 수 있습니다.



### Self-supervised Spatial-Temporal Learner for Precipitation Nowcasting (https://arxiv.org/abs/2412.15917)
Comments:
          7 pages, 2 figures

- **What's New**: 이번 논문에서는 자가 지도 학습(self-supervised learning)의 이점을 활용하여 SpaT-SparK라는 새로운 모델을 제안합니다. 이 모델은 CNN 기반의 인코더-디코더 구조를 채택하고 있으며, 마스킹된 이미지 모델링(masked image modeling, MIM) 작업으로 사전 훈련됩니다. 또한, 과거 및 미래 강수도 맵 간의 시계열 관계를 포착하는 번역 네트워크(translation network)를 통합하여 예측 성능을 극대화하려고 합니다.

- **Technical Details**: SpaT-SparK 모델은 두 가지 주요 구성 요소로 이루어져 있습니다: 첫째, MIM 작업으로 사전 훈련된 인코더-디코더 구조이며, 둘째, 과거와 미래 강수도 맵의 표현 간의 시간적 의존성을 포착하는 번역 네트워크입니다. 인코더는 입력 강수 맵 시퀀스를 잠재 표현(latent representation)으로 인코딩하는 방법을 배우고, 디코더는 원래 이미지를 재구성하는 방법을 학습합니다. 마스킹 전략으로는 tube masking이 사용되어 유용한 공간-시간 구조를 캡처합니다.

- **Performance Highlights**: NL-50 데이터셋에서 수행된 실험 결과, SpaT-SparK 모델이 기존의 감독 학습 모델인 SmaAt-UNet보다 더 높은 정확도의 강수 예측을 제공하는 것으로 나타났습니다. 이는 자가 지도 학습을 통해 강수 예측의 정확성과 신뢰성을 높일 수 있음을 보여줍니다. 따라서 SpaT-SparK는 강수 예측에서 혁신적인 접근 방식으로 주목받고 있습니다.



### Statistical Modeling of Univariate Multimodal Data (https://arxiv.org/abs/2412.15894)
Comments:
          30 pages, 9 figures

- **What's New**: 본 논문에서는 데이터의 밀도 주위를 중심으로 unimodal subset으로 나누는 방법을 제안합니다. 제안된 UniSplit 기법은 valley point를 탐지하여 데이터를 unimodal subset으로 분할하는 과정을 포함합니다. 그런 다음 각 unimodal subset을 Uniform Mixture Model (UMM)로 모델링하여 Unimodal Mixture Model (UDMM)이라는 위계적 통계 모델을 구축합니다.

- **Technical Details**: 제안된 방법은 비모수적 방법으로, 과도한 하이퍼파라미터를 요구하지 않으며, 자동으로 unimodal subset의 수를 추정합니다. valley points를 결정하기 위해 empirical cumulative density function (ecdf)의 볼록 껍질 위의 критические точки(gcm/lcm points)의 성질을 도입하여 density valleys의 존재 여부를 판단합니다. UDMM은 각 혼합 구성 요소가 unimodal 분포에 해당하도록 설계되었으며, 각각의 구성 요소는 UU-test 알고리즘을 사용해 UMM으로 모델링됩니다.

- **Performance Highlights**: 다양한 분포 (예: Gaussian와 uniform)에서 데이터를 모델링하는 유연성을 가지고 있으며, UDMM의 구성 요소 수는 UniSplit 알고리즘을 통해 자동으로 결정됩니다. 실험 결과, UniSplit은 클러스터링 방법과 비교되며, UDMM의 통계적 모델링 성능 또한 평가됩니다. 제안된 방법은 통계적 의미 수준을 제외한 사용자 지정 하이퍼파라미터가 필요 없어, 기존 방법 (예: GMM)보다 명확한 장점을 제공합니다.



### Bayesian Optimization for Unknown Cost-Varying Variable Subsets with No-Regret Costs (https://arxiv.org/abs/2412.15863)
- **What's New**: 본 연구에서는 비용이 무작위적으로 불확실한 경우를 고려한 Bayesian Optimization with cost-varying variable subsets (BOCVS) 문제의 새로운 확장 알고리즘을 제안합니다. 기존의 알고리즘들은 비용을 정확히 알고 있는 경우에만 적용 가능했으나, 본 논문에서는 탐색(exploration)과 활용(exploitation) 단계를 분리하여 이 문제를 해결하려고 합니다. 새로운 접근 방식은 적절한 변수 집합을 필터링하고, 높은 품질의 변수 집합을 통해 성능을 극대화하고자 합니다.

- **Technical Details**: 기존 연구들은 위험하고 불확실한 비용을 다루지 않았으나, 본 연구에서는 비용에 대한 정밀한 정보를 요구하지 않으며 이를 새로운 객체 함수와 후회(regret)의 정의로 해결하려 합니다. 제안된 알고리즘은 품질 후회(quality regret)와 비용 후회(cost regret)를 모두 최소화하도록 설계되었습니다. 이로 인해 BOCVS 문제의 목표를 보다 효과적으로 반영할 수 있게 됩니다.

- **Performance Highlights**: 제안한 알고리즘은 여러 실험 조건에서 기존 기준선(baseline) 방법들과 비교했을 때 뛰어난 성과를 보였습니다. 특히, 정밀 농업 및 고급 제조 응용 분야에 적합한 데이터셋에서 검증을 통해 우수성을 입증하였습니다. 이 결과는 알고리즘의 실용성을 뒷받침하며 향후 다양한 최적화 문제에 적용될 수 있는 가능성을 보여줍니다.



### MarkovType: A Markov Decision Process Strategy for Non-Invasive Brain-Computer Interfaces Typing Systems (https://arxiv.org/abs/2412.15862)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 논문에서는 Rapid Serial Visual Presentation (RSVP) 구성을 부분적 관측 Markov 결정 과정(Partially Observable Markov Decision Process, POMDP)로 공식화하여 타이핑 성능을 향상시키는 MarkovType을 제안합니다. 이를 통해 기존의 방법들과 비교하여 더 높은 분류 정확도를 달성했습니다. 이 연구는 BCI 타이핑 절차를 POMDP로 정형화한 최초의 사례로, 타이핑 프로세스를 학습 절차에 통합하는 방법을 보여 줍니다.

- **Technical Details**: 이 연구는 사용자가 정해진 알파벳에서 목표 기호(target symbol)를 타이핑하는 RSVP 타이핑 작업에 중점을 둡니다. 각 시퀀스에서 사용자는 다양한 기호 쿼리를 보며, EEG 응답을 수집하여 각 기호에 대한 확률을 업데이트합니다. MarkovType은 Residual Bayesian Estimation을 활용해 기존 접근 방법들보다 뛰어난 분류 정확도와 정보 전송률을 보여줍니다.

- **Performance Highlights**: 실험 결과, MarkovType은 기호 분류의 정확도와 타이핑 속도 간의 균형을 최적화하여 기존 모델들보다 상대적으로 우수한 성능을 달성했습니다. 정확성과 타이핑 시퀀스 수 사이의 균형이 필요함을 시사하는 결과는 향후 연구 방향성을 제시합니다. 본 연구는 BCI 시스템에서의 타이핑 작업 효율성을 향상시키기 위한 새로운 가능성을 열어줍니다.



### Improving Quantization-aware Training of Low-Precision Network via Block Replacement on Full-Precision Counterpar (https://arxiv.org/abs/2412.15846)
- **What's New**: 이번 논문에서는 Quantization-aware training (QAT) 방법을 개선하기 위한 일반적인 프레임워크를 제안합니다. 기존의 낮은 정밀도 네트워크의 훈련 과정에서 고정밀(Full-Precision) 모델을 활용하여 훈련 프로세스를 안내함으로써, 이전 QAT의 문제점을 완화하는 접근법입니다. 제안된 방법은 중간의 혼합 정밀도 모델을 생성하고 이를 통해 저정밀도 블록을 고정밀 네트워크에 통합할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 QAT 과정에서 각 저정밀도 블록이 고정밀 표현을 시뮬레이션하도록 할 수 있습니다. 이는 forward pass에서는 고정밀의 표현을 모사하고, backward pass에서는 향상된 경량화를 위한 그래디언트를 얻는 방식으로 작동합니다. 특히, 전체 네트워크의 구조에서 블록 단위로 대체를 진행하며, 이를 통해 저정밀도 모델이 고정밀도 모델의 성능을 부분적으로 모방합니다.

- **Performance Highlights**: 제안된 방법은 ImageNet과 CIFAR-10 데이터세트에서 4비트, 3비트 및 2비트 양자화에 대해 최신 성능을 달성하였습니다. 실험 결과, 각 중간 모델은 기존의 단순 저정밀 모델보다 뛰어난 성능을 보였으며, 고정밀 프레임워크로부터의 안내가 유의미함을 입증했습니다. 해당 프레임워크는 기존 QAT 방법에 대한 호환성이 높은 확장성을 제공합니다.



### Measuring Cross-Modal Interactions in Multimodal Models (https://arxiv.org/abs/2412.15828)
- **What's New**: 이 연구에서는 기존의 XAI 방법들이 unimodal 모델에 국한되어 있어 cross-modal interactions를 측정하는 데 한계가 있다는 점을 지적하며, 이를 해결하기 위해 InterSHAP이라는 새로운 지표를 제시합니다. InterSHAP는 Shapley interaction index를 사용하여 각각의 modality와 상호작용의 기여도를 정확하게 분리하고 계량할 수 있는 능력을 가지고 있습니다. 이 방법은 unlabelled 데이터에도 적용 가능하며, 다양한 modality를 취급할 수 있도록 설계되었습니다.

- **Technical Details**: InterSHAP는 multimodal 모델에서 학습된 cross-modal interaction을 정량화하기 위해 고안된 해석 가능_metric_입니다. 이를 위해 Shapley Interaction Index(SII)를 활용하여 모델의 응답 변화를 정의하며, 서로 다른 modality의 결합이 없으면 발생하지 않는 변화를 측정합니다. 또한, InterSHAP는 early fusion, intermediate fusion, late fusion과 같은 모델 융합 기법을 사용할 수 있음을 설명합니다.

- **Performance Highlights**: InterSHAP의 성능은 다양한 실험을 통해 검증되었으며, 특히 synthetic datasets를 통해 cross-modal interactions의 적절한 감지율을 보였습니다. 예를 들어, synergy가 없는 데이터셋에서는 0%의 상호작용을 정확히 탐지하고, synergy가 있는 데이터셋에 대해서는 99.7%의 감지율을 나타냈습니다. 또한, InterSHAP는 의료 데이터셋에서도 적용 가능성을 보였으며, cell type classification 및 mortality prediction 작업에서 그 유용성이 입증되었습니다.



### S$^2$DN: Learning to Denoise Unconvincing Knowledge for Inductive Knowledge Graph Completion (https://arxiv.org/abs/2412.15822)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 Inductive Knowledge Graph Completion(KGC)를 위한 새로운 S$^2$DN(Semantic Structure-aware Denoising Network) 네트워크를 제안합니다. 이 네트워크는 지식 그래프(KG) 내에서 새롭게 등장하는 개체들 간의 누락된 사실을 추론하는 과정을 개선하는 것을 목표로 합니다. 특히, S$^2$DN은 비슷한 관계의 의미적 불일치와 KG에서의 노이즈를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: S$^2$DN은 관계의 일반적인 의미를 유지하기 위해 포괄하는 서브그래프에 대한 의미적 스무딩 모듈을 도입합니다. 또한, 신뢰할 수 없는 상호작용을 필터링하고 추가적인 지식을 제공하기 위해 구조 정련 모듈을 통합하여 KG 내에서의 신뢰할 수 있는 상호작용을 유지합니다. 이러한 구조적인 정제 및 의미적 스무딩 접근 방식은 KG의 신뢰성과 일관성을 지속적으로 높이는 데 기여합니다.

- **Performance Highlights**: S$^2$DN은 다양한 KG와 서로 다른 노이즈 조건에서 GraIL 모델을 초월하는 예측 성능을 보였습니다. 경험적 실험을 통해 S$^2$DN은 KG 내에서 의미적 일관성을 유지하고 불확실한 상호작용을 효과적으로 필터링하는 능력을 입증하였습니다. 이러한 결과는 S$^2$DN이 KGC 분야에서 뛰어난 성과를 가지고 있음을 보여줍니다.



### WebLLM: A High-Performance In-Browser LLM Inference Engin (https://arxiv.org/abs/2412.15803)
- **What's New**: 최근의 대형 언어 모델(LLMs) 발전으로 인해 이전에는 상상할 수 없었던 능력들이 열렸습니다. 서버급 GPU를 요구하는 이러한 모델을 클라우드에 호스팅하는 대신, 최근에는 더 작고 오픈 소스 모델들이 등장하면서 개인 장치에서의 배포가 가능해졌습니다. 이에 따라 WebLLM이라는 새로운 오픈 소스 JavaScript 프레임워크가 소개되어 웹 브라우저 내에서 LLM 추론을 수행할 수 있게 되었습니다.

- **Technical Details**: WebLLM은 웹 응용 프로그램에 LLM 기반 기능을 통합할 수 있도록 돕는 JavaScript 프레임워크입니다. 이 시스템은 사용자 친화적인 엔진 ServiceWorkerMLCEngine, 웹 워커 내에 위치한 MLCEngine, 그리고 사전 컴파일된 효율적인 WebGPU 커널로 구성됩니다. WebGPU와 WebAssembly를 활용해 GPU 가속과 CPU 계산을 효율적으로 수행하며, 백엔드 실행을 웹 워커로 분리하여 UI 흐름에 방해가 없도록 설계되었습니다.

- **Performance Highlights**: 실제 평가 결과 WebLLM은 MLC-LLM에 비해 최대 80%의 성능을 유지하며, 이는 동일한 장치에서의 성능을 기준으로 합니다. 또한, WebGPU의 최신 기능과 WebLLM의 런타임 최적화를 통해 성능 격차를 더욱 줄일 수 있는 가능성도 존재합니다. 최종적으로 WebLLM은 웹 브라우저 내에서 개인화된 LLM 애플리케이션을 구현할 수 있는 길을 열었습니다.



### Function Space Diversity for Uncertainty Prediction via Repulsive Last-Layer Ensembles (https://arxiv.org/abs/2412.15758)
- **What's New**: 본 논문은 Bayes 추론(Bayesian inference)을 함수 공간(function space)에서 수행하는 새로운 방법을 제안합니다. 특히 파라미터화가 과도해지는 신경망에서 강력한 성능을 발휘할 수 있는 점에 주목하고 있습니다. 무한 차원 함수 공간을 근사하는 과정의 여러 가지 도전 과제를 다루며, 특히 대규모로 사전 훈련된 네트워크에 적용 가능한 모듈화된 솔루션을 제시합니다.

- **Technical Details**: 연구자는 파티클 최적화(particle optimization)를 통해 함수 공간 추론(function space inference)을 논의합니다. 입력 샘플의 다양성이 모델 성능에 부정적인 영향을 미친다는 사실을 실험적으로 보여줍니다. 레이블을 파괴하는 데이터 증강(data augmentation) 기법 혹은 비라벨의 분포 외 데이터(unlabeled out-of-distribution data)를 활용하면 예측의 다양성과 불확실성 추정이 개선된다고 설명합니다. 또한, 완전한 deep ensembles 대신에 파라미터 및 연산량의 최소한의 증가로 다중 머리 네트워크(multi-headed network)를 제안합니다.

- **Performance Highlights**: 본 연구에서는 활성 학습(active learning), 도메인 외 데이터(out-of-domain data) 탐지 및 분포 변화(distribution shifts)에서의 보정된 불확실성 추정(calibrated uncertainty estimates) 문제를 다루며 경쟁력 있는 결과를 달성했습니다. 이러한 방식은 최소한의 계산량(compute)과 메모리 비용(memory cost)으로 불확실성을 인식한 미세 조정(uncertainty aware fine-tuning)을 가능하게 합니다. 궁극적으로, 제안된 방법은 대규모 네트워크에 손쉽게 통합될 수 있는 장점이 있습니다.



### Extracting Interpretable Task-Specific Circuits from Large Language Models for Faster Inferenc (https://arxiv.org/abs/2412.15750)
Comments:
          Accepted to AAAI 25 Main Technical Track

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 특정 작업을 위한 최소 하위 집합을 자동으로 추출하는 새로운 접근 방식을 제안합니다. 이 방법은 추가적인 훈련 없이도 특정 작업을 수행할 수 있도록 설계되었습니다. 이를 통해 LLM의 크기를 크게 줄이고 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 Transformer 아키텍처에 기반한 LLM의 메커니즘 해석 기법(Mechanistic Interpretability, MI)을 활용하여 다양한 작업을 수행하는 서브모델을 자동으로 추출합니다. 제안된 방법은 하이퍼파라미터의 영향을 평가하고, 정확도 및 속도를 측정하여 주요 구성 요소들이 포함되어 있는지를 검증합니다. 이를 통해, 파라미터 수를 82.77%까지 줄이고, 추론 속도 또한 개선됩니다.

- **Performance Highlights**: 제안된 접근 방식은 결과적으로 생성된 모델이 적은 수의 파라미터와 빠른 추론 속도를 보임을 보여줍니다. 효율적인 작업 수행을 위한 필수 구성 요소에 집중함으로써, 연구진은 MI 기술을 통해 모델의 이해 가능성을 높이고 있습니다. 이로 인해 LLM을 다양한 응용 분야에서 보다 유용하게 활용할 수 있을 것으로 기대됩니다.



### Prompt-based Unifying Inference Attack on Graph Neural Networks (https://arxiv.org/abs/2412.15735)
Comments:
          Accepted by the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 본 논문에서는 GNN(그래프 신경망)을 기반으로 한 새로운 프로프트 기반 통합 추론 공격 프레임워크인 ProIA를 제안합니다. 이 프레임워크는 그래프의 구조적 정보를 유지하면서 대응 전략을 만들기 위해 설계되었습니다. ProIA는 기초 지식을 향상시키고 다양한 공격 과제에 대한 적응성을 높이는 데 기여합니다.

- **Technical Details**: ProIA는 그래프의 중요한 토폴로지 정보를 전이 학습(pre-training) 동안 유지하여, 정보 유출을 유도하는 프롬프트 쿼리를 설계합니다. 이 과정에서 그래프의 구조와 민감한 정보의 연결성을 강화하는 정보 이론적 원칙이 사용됩니다. 또한, 다운스트림(task-relevant) 공격에서 적응적으로 숨겨진 정보를 추출하기 위한 분리(disentanglement) 메커니즘이 포함됩니다.

- **Performance Highlights**: ProIA는 다섯 개의 공개 데이터셋에서 실험을 통해 뛰어난 개인 정보 추론 능력을 입증했습니다. 결과적으로 ProIA는 여러 일반적인 방어 메커니즘을 무력화하는 방식으로 공격 성능을 향상시킵니다. 이러한 성과는 GNN에 대한 추론 공격의 새로운 가능성을 제시합니다.



### fluke: Federated Learning Utility frameworK for Experimentation and research (https://arxiv.org/abs/2412.15728)
Comments:
          Accepted at FLUID workshop (AAAI 2025) [4 pages (+2 references), 2 figures, 1 algorithm]

- **What's New**: 이 논문에서는 새로운 파이썬 패키지인 fluke를 소개합니다. fluke는 Federated Learning(FL) 알고리즘을 효율적으로 개발하고 프로토타입 할 수 있도록 설계되었습니다. 이 패키지는 연구자나 전문가가 알고리즘의 교육 요소에 집중할 수 있도록 지원하는데 중점을 두고 있습니다.

- **Technical Details**: fluke는 오픈소스 파이썬 패키지로, 사용자 친화적인 설계를 가지고 있어 알고리즘을 쉽게 추가하고 사용할 수 있습니다. 사용자가 알고리즘의 세부사항에 대해 걱정하지 않고도 새로운 FL 알고리즘을 신속하게 프로토타입 할 수 있도록 도와줍니다. 패키지는 중앙 집중형 아키텍처를 가정하며, 클라이언트와 서버 간의 통신을 시뮬레이션하여 FL 환경을 구현합니다.

- **Performance Highlights**: fluke는 현재 몇 가지 최신 FL 알고리즘과 데이터 세트를 포함하고 있으며, 정기적으로 업데이트 되어 최신 기술을 반영합니다. 패키지의 CLI(커맨드라인 인터페이스)는 FL 실험을 실행하는 가장 쉬운 방법을 제공하며, 세 가지 실험 유형을 지원합니다. fluke는 연구자들이 새 알고리즘을 빠르게 프로토타입 할 수 있도록 하기 위한 유틸리티 프레임워크로, FL 분야의 발전에 기여할 것으로 기대됩니다.



### Concept Boundary Vectors (https://arxiv.org/abs/2412.15698)
Comments:
          21 pages, 21 figures

- **What's New**: 본 연구에서는 머신러닝 모델의 내부에 있는 개념의 표현을 이해하기 위해 개념 경계 벡터(concept boundary vectors)를 소개합니다. 이는 개념의 잠재 표현(latent representations) 사이의 경계에서 파생된 개념 벡터의 구조입니다. 이를 통해, 모델의 출력 해석을 도와주고 이러한 표현의 명확성을 향상시킬 수 있는 방법을 찾아보려 합니다.

- **Technical Details**: 개념 벡터는 머신러닝 모델의 잠재 공간에서 개념의 관계를 표현하려는 구조입니다. 이들은 일반적으로 감독 학습(supervised learning) 방식으로 생성되며, 개념이 존재하는 데이터와 존재하지 않는 데이터가 필요합니다. 개념 활성화 벡터(concept activation vectors)는 선형 분류기를 이용하여 개념의 존재 여부에 따라 잠재 표현을 구분하는 방법을 사용합니다. 본 연구에서는 개념 간의 관계를 더욱 충실히 포착하기 위해 개념 경계 벡터를 개념 활성화 벡터의 대안으로 제안합니다.

- **Performance Highlights**: 개념 경계 벡터는 개념 간의 관계의 기하학적 특징을 이용하여 향상된 성능을 기대할 수 있습니다. 기존의 연구 결과에 따르면, 분류 모델의 결정 경계의 기하학적 복잡성은 모델의 정확도에 영향을 미치는 것으로 나타났습니다. 따라서, 본 연구는 개념 경계 벡터가 개념 간 관계의 의미를 효율적으로 캡처할 수 있을 것이라 예상합니다.



### Hypergraph clustering using Ricci curvature: an edge transport perspectiv (https://arxiv.org/abs/2412.15695)
- **What's New**: 이번 논문에서는 hypergraph에 Ricci flow를 확장하는 혁신적인 방법을 제시합니다. 이러한 접근 방식은 엣지에 대한 확률 분포를 정의하고, 이를 통해 엣지의 새로운 가중치를 생성하여 community detection에 효과적입니다. 특히, 대형 hyperedge의 존재에서 hypergraph 구조에 대한 민감도가 향상된다는 점이 강조됩니다.

- **Technical Details**: 논문은 hypergraph 이론 및 Ollivier-Ricci 곡률에 대한 기본 개념들을 소개합니다. hypergraph의 엣지들에 대해 확률 분포를 고려하여 Ricci 곡률을 정의하는 방법을 제안하며, 이는 hypergraph의 line expansion을 활용합니다. 이 접근 방식은 작은 커뮤니티가 많은 경우, 커뮤니티 간 엣지가 더 큰 경우에 효과적입니다.

- **Performance Highlights**: 실험적으로, 제안한 방법은 기존의 노드 기반의 측정을 운반하는 방식과 비교됩니다. 실험 결과, 엣지 운반 기반의 방법이 다양한 클러스터링 알고리즘과 비교하여 우수한 성능을 보였으며, 이는 특히 크고 복잡한 hypergraph에서 더욱 효율적임을 나타냅니다. 결과적으로 이 연구는 hypergraph에서의 클러스터링 문제를 해결하는 데 있어 중요한 기여를 하였습니다.



### Theory of Mixture-of-Experts for Mobile Edge Computing (https://arxiv.org/abs/2412.15690)
Comments:
          This is the technical report for our paper accepted by INFOCOM 2025

- **What's New**: 이번 논문에서는 모바일 엣지 컴퓨팅(MEC) 네트워크에서의 지속적 학습 성능을 개선하기 위해 혼합 전문가(mixture-of-experts, MoE) 이론을 처음으로 도입하였습니다. MoE 모델은 각 MEC 서버를 전문가로 간주하고, 데이터 전송 및 계산 시간을 고려하여 서버의 가용성 변화를 동적으로 적응합니다. 이러한 접근 방식은 전통적인 MEC의 일반화 오류 문제를 해결하며, 기존의 오프라인 작업용 MoE 알고리즘과는 달리, 지속적인 작업 스트림을 처리하는 데 최적화되어 있습니다.

- **Technical Details**: 제안된 MoE 모델은 새로운 태스크를 수신하면 이를 가용한 전문가에게 적응적으로 라우팅하는 적응형 게이팅 네트워크(adaptive gating network, AGN)를 특징으로 합니다. AGN은 각 전문가가 특정 유형의 작업에 전문화되도록 도와주며, 모델의 오류를 기반으로 게이팅 네트워크의 매개변수를 업데이트합니다. 이론적으로, 최소 전문가 수를 유도하여 현재 작업 도착에 전문화된 전문가가 반드시 하나 이상 존재하도록 하여 시스템의 수렴을 보장합니다.

- **Performance Highlights**: 제안된 MoE 접근 방식은 시간이 지남에 따라 전체 일반화 오류를 일관되게 감소시키며, 데이터 분포가 크게 변하는 경우 그 효과가 더욱 두드러집니다. 기존 MEC의 작업 오프loading 솔루션은 시간이 지남에 따라 증가하는 일반화 오류를 초래하지만, MoE 모델은 전체 일반화 오류가 시간이 지남에 따라 감소하게 하여 최종적으로 소수의 상수로 제한된다는 것을 입증했습니다. 실제 데이터셋을 사용한 광범위한 실험을 통해 이론적 결과를 검증하였습니다.



### Synthetic Tabular Data Generation for Imbalanced Classification: The Surprising Effectiveness of an Overlap Class (https://arxiv.org/abs/2412.15657)
Comments:
          AAAI Conference 2025

- **What's New**: 이 논문은 불균형 데이터에 대한 분류 모델의 성능 향상을 위한 새로운 접근법인 ORD(Overlapped Region Detection)를 제안합니다. 이 방법은 기본적으로 이진 클래스 레이블을 세 가지 레이블로 변환하여 소수 클래스와 다수 클래스의 경계에 있는 데이터를 식별합니다. 이를 통해 생성 모델이 세 가지 클래스(소수, 겹치는 다수, 명확한 다수)에 대한 학습을 통해 데이터 생성 품질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: ORD는 세 가지 주요 아이디어로 구성됩니다. 첫째, 다수 예제 중 소규모 서브셋인 겹치는 데이터 세트를 사전 처리합니다. 둘째, 생성 모델을 클래스 조건으로 수정하여 소수, 겹치는 다수, 명확한 다수에 해당하는 예제를 생성합니다. 셋째, 최종 분류기는 실제 소수 클래스와 생성된 명확한 다수 클래스의 조합으로 학습됩니다.

- **Performance Highlights**: 이 방식은 기존의 방법보다 훨씬 높은 품질의 소수 예제를 생성하며, 네 가지 실제 데이터 세트와 다섯 가지 분류 모델을 통해 성능 평가를 진행했습니다. 실험 결과, ORD 방법을 활용함으로써 분류기의 정확도가 크게 향상되었으며, 이는 생성된 데이터의 품질 향상과 겹치는 다수 클래스 포인트의 선택적 언더샘플링으로 인해 이루어졌음을 보여줍니다.



### Beyond Human Data: Aligning Multimodal Large Language Models by Iterative Self-Evolution (https://arxiv.org/abs/2412.15650)
Comments:
          AAAI 2025. The code is available at this https URL

- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 성능을 향상시키기 위해 고품질의 선호 데이터를 자동 생성하는 새로운 자기 진화 프레임워크인 SENA를 제안합니다. SENA는 레이블이 없는 이미지만으로 작동하여, 모델이 질문과 답변을 생성하고 평가하는 기계를 자체적으로 구현하였습니다. 이 과정은 무의미한 질문을 줄이고 보다 유용한 데이터를 생성하는 데 중점을 두어 모델의 학습을 개선합니다.

- **Technical Details**: SENA는 이미지 기반 자기 질문 메커니즘을 채택하여 이미지 콘텐츠에 기반한 질문의 품질을 높입니다. 또한, 이미지 캡셔닝을 시작으로 답변 품질 향상을 위한 자기 향상 기술을 도입하며, 손상된 이미지를 이용해 거부된 답변을 생성하는 방식으로 구별된 선호 쌍을 구축합니다. 최종적으로, 이미지 콘텐츠 정렬 손실 함수와 Direct Preference Optimization (DPO) 손실을 결합하여 모델이 이미지 콘텐츠에 집중할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SENA는 외부 정보를 사용하는 기존 방법들과 경쟁력 있는 성과를 나타내며, MLLMs에 대해 더 효율적이고 확장 가능한 접근 방식을 제공합니다. 이 프레임워크는 다양한 벤치마크에서 모델의 성능을 유의미하게 향상시킬 수 있음을 확인했습니다. 이러한 혁신적인 접근은 사용자 선호 align에 대한 안정성을 확보하고 지속적인 성능 개선을 보장합니다.



### SODor: Long-Term EEG Partitioning for Seizure Onset Detection (https://arxiv.org/abs/2412.15598)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이번 연구에서는 발작 시작 감지를 명시적으로 모델링하는 새로운 두 단계 프레임워크인 SODor를 제안합니다. 기존 방법들은 EEG 신호에서 발작을 분류하는 데 중점을 두었지만, 발작 시작 (Seizure Onset) 감지를 직접적으로 다루지 못했습니다. SODor는 서브시퀀스 클러스터링(subsequence clustering) 작업으로 발작 시작 감지 문제를 모델링하여, 발작 상태를 자동으로 식별하고 세분화합니다.

- **Technical Details**: SODor는 EEG 데이터를 활용하여, 두 단계로 진행되는 서브시퀀스 클러스터링으로 발작 시작 시점이 발생하는 지점을 명확히 식별합니다. 첫 번째 단계는 레이블 감독을 통해 2차 임베딩 세트를 학습하고, 두 번째 단계에서는 모델 기반 클러스터링을 사용하여 장기적인 시간 종속성을 캡처합니다. 이 방법은 클러스터 혹은 상태 전이를 통해 발작 시작을 탐지할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: 저자들은 세 가지 데이터셋에서 SODor를 평가한 결과, 다른 기초 모델들에 비해 5%-11% 향상된 분류 성능을 달성했으며 발작 시작 감지의 정확성 또한 크게 개선된 것을 보였습니다. 이러한 성과는 EEG 신호 내에서 의미 있는 서브시퀀스를 추출하고 이를 통해 발작 시작을 효과적으로 감지할 수 있음을 보여줍니다.



### Machine Learning Techniques for Pattern Recognition in High-Dimensional Data Mining (https://arxiv.org/abs/2412.15593)
- **What's New**: 이 논문은 지원 벡터 머신(Support Vector Machine, SVM)을 기반으로 한 빈번한 패턴 데이터 마이닝 알고리즘을 제안합니다. 기존의 빈번한 패턴 마이닝 알고리즘이 고차원 및 희소 데이터 환경에서 나타나는 성능 병목 현상을 해결하는 것을 목표로 합니다. 빈번한 패턴 마이닝 작업을 분류 문제로 전환하며, SVM 모델을 도입해 패턴 추출의 정확성과 강건성을 향상시키고 있습니다.

- **Technical Details**: 방법 설계 측면에서 커널 함수(Kernel Function)를 사용하여 데이터를 고차원 특징 공간(High-dimensional Feature Space)으로 매핑합니다. 최적의 분류 하이퍼플레인(Optimal Classification Hyperplane)을 구성하여 비선형 패턴을 분리하고, 빈번한 항목(Frequent Items)을 정확하게 마이닝합니다. 실험에서는 Retail과 Mushroom이라는 두 개의 공공 데이터셋을 선택하여 제안된 알고리즘과 기존의 FP-Growth, FP-Tree, 결정 트리(Decision Tree), 랜덤 포레스트(Random Forest) 모델을 비교 분석하였습니다.

- **Performance Highlights**: 실험 결과, 이 논문에서 제안한 알고리즘은 지원(Support), 신뢰도(Confidence), 리프트(Lift)라는 세 가지 주요 지표에서 기존 모델에 비해 현저히 우수한 성능을 보였습니다. 이는 강력한 패턴 인식 능력과 규칙 추출 효과를 나타냅니다. SVM 모델이 데이터 희소성이 높은 환경과 많은 거래가 있는 상황에서 탁월한 성능 이점을 가지고 있으며, 복잡한 패턴 마이닝 작업을 효과적으로 처리할 수 있음을 보여줍니다.



### Pre-training Graph Neural Networks on Molecules by Using Subgraph-Conditioned Graph Information Bottleneck (https://arxiv.org/abs/2412.15589)
Comments:
          15 pages

- **What's New**: 이번 연구는 인간 주석이나 사전 지식 없이 분자에 대한 사전 훈련된 그래프 신경망(Graph Neural Network, GNN) 모델을 구축하는 것을 목표로 하고 있습니다. 기존의 사전 훈련 방법들은 기능 그룹과 같은 의미적 부분 그래프에 의존하고 있어 그래프 수준의 차별성을 간과할 수 있는 한계가 있습니다. 본 연구에서 제안하는 S-CGIB는 핵심 부분 그래프(그래프 코어)를 인식하고 중요한 부분 그래프를 자동으로 탐색하여 이러한 한계를 극복합니다.

- **Technical Details**: S-CGIB는 입력 그래프를 특정 중요한 부분 그래프에 따라 압축하여 그래프 코어로 변환하는 방식으로, 노드들 간의 상호작용이 주의(attention)에 기반하여 이루어집니다. 이 방법은 기능 그룹 후보들(ego networks)을 생성하고, 그래프 코어와 이 후보들 간의 상호작용을 통해 중요한 부분 그래프를 식별합니다. 결과적으로, 기능 그룹에 대한 사전 지식 없이도 분자에 대한 강인한 표현을 생성할 수 있습니다.

- **Performance Highlights**: 다양한 분자 데이터셋에서 수행된 광범위한 실험 결과, S-CGIB는 기존의 방법들보다 우수한 성능을 보였습니다. 이는 성과적으로 그래프 수준 표현을 잘 구분짓는 데에 기여하며, 기능 그룹을 보다 정확하게 인식할 수 있도록 합니다. 연구 결과는 GNN이 분자의 화학적 성질과 구조를 이해하는 데 필수적인 도구로 자리 잡을 수 있음을 시사합니다.



### A Deep Probabilistic Framework for Continuous Time Dynamic Graph Generation (https://arxiv.org/abs/2412.15582)
Comments:
          To appear at AAAI-25

- **What's New**: 최근 그래프 표현 학습의 발전은 동적 그래프에 대한 주목도를 높였습니다. 동적 그래프는 시간이 지남에 따라 구조와 특징이 변화하는 그래프를 의미합니다. 특히, 데이터 증강(data augmentation)이나 이상 탐지(anomaly detection)와 같은 다양한 응용을 위해 적합한 생성 모델의 필요성이 커지고 있습니다. 본 논문에서는 기존의 방법들과는 다른 새로운 접근 방식인 DG-Gen이라는 생성 프레임워크를 제안합니다.

- **Technical Details**: DG-Gen은 시간적인 상호작용을 직접 모델링하는 것을 중심으로 하고, 이를 통해 새로운 합성 동적 그래프를 생성합니다. 이 모델은 정적인 그래프에 의존하지 않고, 에지(edge)가 두 노드 간에 형성될 확률을 공동 확률(joint probability)로 모델링합니다. DG-Gen에서는 깊은 확률 디코더(deep probabilistic decoder)를 사용하여 조건부 확률(distribution)의 곱으로 확률을 분해하고, 이를 통해 새롭게 생성된 상호작용을 자동 회귀적으로 생성합니다. 이 과정은 높은 수준의 기계적 가정 없이 진행됩니다.

- **Performance Highlights**: DG-Gen은 기존의 전통적인 방법들과 비교하여 보다 높은 충실도의 그래프를 생성할 뿐만 아니라, 링크 예측(link prediction 작업)에서도 상당한 성과를 보여줍니다. 본 논문에서 진행한 실험은 DG-Gen이 다섯 개의 다양한 데이터셋에서 TIGGER-I 모델을 초월하는 결과를 도출했음을 입증합니다. 이로 인해 DG-Gen은 동적 그래프 생성의 혁신적인 접근 방식으로 자리매김하게 되었습니다.



### Continual Learning Using a Kernel-Based Method Over Foundation Models (https://arxiv.org/abs/2412.15571)
- **What's New**: 본 논문은 지속적 학습(Continual Learning, CL) 중 클래스 증가 학습(Class-Incremental Learning, CIL)의 도전적인 설정을 다룹니다. 기존의 여러 방법에도 불구하고, 파라미터 업데이트로 인한 재앙적 망각(Catastrophic Forgetting, CF) 및 과제 간 클래스 분리(Inter-task Class Separation, ICS) 문제가 여전히 존재합니다. 이 문제를 해결하기 위해, Kernel Linear Discriminant Analysis (KLDA)라는 새로운 방법을 제안하며, 이 방법은 기초 모델(Foundation Model)에서 학습된 강력한 특징을 활용합니다.

- **Technical Details**: KLDA는 Radial Basis Function (RBF) 커널과 Random Fourier Features (RFF)를 통합해 기초 모델에서 추출된 특징 표현을 향상시킵니다. 새로운 작업이 도착하면 KLDA는 각 클래스의 평균을 계산하고, 커널화된 특징을 기반으로 모든 학습된 클래스에 대한 공유 공분산 행렬을 업데이트합니다. 이 방법은 Linear Discriminant Analysis (LDA)를 사용하여 분류를 수행하며, 각 클래스에 대한 가우시안 분포를 정의하여 결정 경계를 최적화합니다.

- **Performance Highlights**: KLDA는 텍스트 및 이미지 분류 데이터세트를 사용한 실험적 평가에서 기존 방법들보다 우수한 성능을 보였습니다. 특히, KLDA는 재생 데이터에 의존하지 않고도 CIL 성능의 상한으로 여겨지는 모든 클래스의 조합 훈련에 맞먹는 정확도를 달성하였습니다. 이는 기존의 다른 CIL 방법들이 모자란 정확도를 극복하는 데 중요한 의미를 갖습니다.



### Spatial Clustering of Citizen Science Data Improves Downstream Species Distribution Models (https://arxiv.org/abs/2412.15559)
- **What's New**: 이 논문은 시민 과학 데이터가 지니는 생물 다양성에 대한 큰 기회에 대해 다루고 있습니다. 특히, 시민 과학 데이터에서의 불완전한 탐지 문제를 해결하기 위한 점유 모델(occupancy model)의 중요성을 강조합니다. 연구에서는 기존 데이터셋에서 생물의 분포 모델을 개선할 수 있는 여러 접근 방법을 비교합니다. 시민 과학 프로그램인 eBird 데이터를 활용하여 31종의 새 분포 모델을 분석하였습니다.

- **Technical Details**: 점유 모델은 환경적 특성과 생물학적 선택 과정을 분리하여 모델링하며, 사용자 정의된 여러 관찰 프로세스를 통해 불완전한 탐지를 보정합니다. 논문에서는 점유 모델이 사이트 클러스터링 기법을 통해 구축되는 사례를 분석하였으며, 환경적 및 지리적 유사성을 함께 고려한 기계 학습 기반의 방법이 더 나은 성능을 보임을 보여주었습니다. 총 10개의 서로 다른 사이트 구성 방안을 비교 분석했습니다.

- **Performance Highlights**: 기계 학습 방법론을 사용한 사이트 구성 방안이 다른 기존 방법들보다 우수한 성과를 보였습니다. 조사 결과, 모든 데이터 포인트를 유지하고 환경적 특성을 통합하는 접근 방식이 효과적임을 확인했습니다. 이 연구는 오픈 소스 데이터와 코드를 제공하여, 다른 연구자들이 쉽게 결과를 재현하고 활용할 수 있도록 하였습니다.



### Architecture-Aware Learning Curve Extrapolation via Graph Ordinary Differential Equation (https://arxiv.org/abs/2412.15554)
- **What's New**: 이 논문은 신경망(Neural Network) 아키텍처를 통합하여 학습 곡선(Learning Curve) 모델링을 개선하는 새로운 접근법을 제안합니다. 기존의 방법들이 아키텍처의 영향을 무시하는 경향이 있었던 반면, 우리는 이를 고려하여, 신경망 구조의 동적 특성을 반영한 정보를 활용하였습니다. 이 방법은 AutoML 분야에서 하이퍼파라미터 튜닝과 신경망 아키텍처 검색을 가속화할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 우리는 순환 신경망(CNN) 및 다층 퍼셉트론(MLP)의 학습 곡선을 예측하기 위해 아키텍처 인식 신경 미분 방정식(Neural Differential Equation) 모델을 개발했습니다. 이 모델은 그래프 컨볼루션 네트워크(Graph Convolutional Networks, GCN)와 같은 기술을 활용하여 아키텍처의 토폴로지에서 그래프 수준 임베딩을 생성합니다. 이를 통해 학습 곡선의 변동성을 효과적으로 포착하고 예측의 불확실성을 정량화할 수 있습니다.

- **Performance Highlights**: 새롭게 제안된 모델은 기존의 최첨단 학습 곡선 예측 방법과 시계열 모델링 접근법을 초월하여 더 나은 성능을 보여줍니다. 특히, 우리의 모델은 성능이 다양한 학습 조건에서 발휘될 수 있는 능력을 가지고 있으며, 학습 곡선의 예측 정확성을 개선하여 모델 순위를 매기는 데 있어 20배 더 빠른 속도를 자랑합니다. 결국 이는 머신러닝 분야에서의 실험 속도와 자원 사용의 효율성을 크게 향상시킬 수 있습니다.



### AutoRank: MCDA Based Rank Personalization for LoRA-Enabled Distributed Learning (https://arxiv.org/abs/2412.15553)
- **What's New**: 이 논문에서는 분산 머신러닝 환경에서의 모델 훈련의 복잡성을 해결하기 위한 새로운 접근법으로 AutoRank를 제안합니다. AutoRank는 각 참여자의 데이터 복잡성에 따라 동적으로 로컬 랭크를 설정해 모델 성능 향상 및 수렴 속도 향상을 도모합니다. 또한, 저자들은 AutoRank가 자료 분포에 따라 참여자 설정을 맞춤화함으로써 데이터의 불균형 문제를 완화하는 솔루션으로 작용한다는 점을 강조합니다.

- **Technical Details**: AutoRank는 비편향-분산 트레이드오프(bias-variance trade-off)에서 영감을 받아 설계된 적응형 랭크 설정 알고리즘입니다. 해당 알고리즘은 Topological Preference Technique for Score Indicator Selection (TOPSIS) 방법론을 활용하여 데이터의 복잡성에 기초해 로컬 랭크를 동적으로 할당합니다. 이를 통해 각 참여자의 LoRA 모델에 대한 세분화된 조정이 가능해지며, 특히 비독립적이고 비동일하게 분포된(non-IID) 데이터 문제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, AutoRank는 고차원 데이터 분포에서 모델의 성능을 향상시키는 동시에 훈련 과정에서의 컴퓨팅 자원 소모를 현저히 줄였으며, 수렴 속도도 가속화했습니다. 이는 분산 머신러닝 환경에서 AutoRank가 높은 적응력과 효율성을 제공함을 입증하며, 보다 유연한 분산 학습 솔루션으로 자리 잡을 가능성이 있음을 보여줍니다.



### FedRLHF: A Convergence-Guaranteed Federated Framework for Privacy-Preserving and Personalized RLHF (https://arxiv.org/abs/2412.15538)
Comments:
          Accepted to AAMAS 2025. This preprint represents the full version of the paper, including all proofs, experimental details, and additional discussions

- **What's New**: 최근 개인 정보 보호에 대한 우려가 커지고 개인화된 경험에 대한 수요가 증가함에 따라, 기존의 Reinforcement Learning with Human Feedback (RLHF) 프레임워크는 중앙 집중화된 데이터에 의존하여 상당한 도전에 직면하고 있습니다. 본 논문에서는 RLHF 프로세스를 분산화하는 새로운 프레임워크인 Federated Reinforcement Learning with Human Feedback (FedRLHF)를 소개합니다. FedRLHF는 여러 클라이언트 간의 협력적 정책 학습을 가능케 하며, 원시 데이터나 인간 피드백을 공유할 필요 없이 개인 정보 보호를 보장합니다.

- **Technical Details**: FedRLHF는 각 클라이언트가 로컬 환경에서 인간 피드백을 통합하여 보상 함수를 업데이트하고 개인화된 RLHF 프로세스를 통해 정책을 학습할 수 있도록 합니다. 우리는 FedRLHF에 대한 엄격한 이론적 기반을 수립하고, 수렴 보장(convergence guarantees) 및 샘플 복잡도 경계를 도출하여 클라이언트 수가 증가함에 따라 효율적으로 확장될 수 있음을 보여줍니다. 추가로, 로컬 모델은 클라이언트의 데이터와 인간 피드백으로만 훈련되어 민감한 사용자 정보가 장치에 머물도록 합니다.

- **Performance Highlights**: MovieLens 및 IMDb 데이터셋에 대한 경험적 평가를 통해 FedRLHF는 사용자 개인 정보를 보호하면서도 중앙 집중식 RLHF와 동등한 성능을 달성하며 다양한 클라이언트 환경에서 개인화를 향상시키는 기능을 보여주었습니다. 이 프레임워크는 사용자 개개인의 다양성을 고려하여 정책을 조정할 수 있는 유연성을 제공합니다. FedRLHF를 통해 우리는 개인 정보 보호와 개인화의 균형을 동시에 달성하는 데 기여하고 있습니다.



### SORREL: Suboptimal-Demonstration-Guided Reinforcement Learning for Learning to Branch (https://arxiv.org/abs/2412.15534)
Comments:
          AAAI 2025

- **What's New**: 본 논문에서는 Mixed Integer Linear Program (MILP) 풀어내기 위해 새로운 방법인 Suboptimal-Demonstration-Guided Reinforcement Learning (SORREL)을 제안합니다. 기존의 MILP 솔버는 수작업으로 설계된 휴리스틱(heuristic)에 의존했지만, SORREL은 더 효율적으로 서브옵티멀한 데모(demo)를 학습하여 문제 해결의 성능을 향상시킵니다. 또한 SORREL은 오프라인 강화 학습(offline reinforcement learning)과 자기 모방 학습(self-imitation learning)을 통합하여 더 향상된 트레이닝을 가능하게 합니다.

- **Technical Details**: SORREL은 두 단계로 구성된 강화 학습(RL) 접근 방식을 적용합니다. 첫 번째 단계에서는 기존의 서브옵티멀한 휴리스틱에서 수집된 데모로부터 오프라인 RL 에이전트를 훈련시키며, 두 번째 단계에서는 온라인 RL을 통해 이전 단계에서 사전 훈련된 에이전트의 성능을 더욱 개선합니다. 이 과정에서 SORREL은 새로운 트리 마르코프 결정 프로세스(tree Markov Decision Process)를 사용하여 변수 선택 과정을 보다 일반적으로 모델링하고, 각 MILP 인스턴스에 대한 우선 순위 큐를 활용하여 최적의 경로를 추적합니다.

- **Performance Highlights**: 실험 결과에 따르면 SORREL은 다양한 MILP 문제에서 기존의 방법들에 비해 분기 품질과 훈련 효율성 모두에서 우수한 성능을 보였습니다. SORREL은 동일한 서브옵티멀 휴리스틱에 접근할 수 있는 강화 학습 및 모방 학습의 기초선들보다도 일관된 성능 향상을 기록했으며, 고품질 데모로 훈련된 모방 학습 방식과도 유사한 성능을 달성했습니다. 특히, SORREL은 모든 신경망 방법 중에서 가장 효율적인 훈련 시간을 자랑하여 실제 응용에서의 가능성을 보여줍니다.



### Generalized Back-Stepping Experience Replay in Sparse-Reward Environments (https://arxiv.org/abs/2412.15525)
- **What's New**: 이번 논문에서는 Back-stepping experience replay (BER) 기법을 개선한 Generalized BER (GBER)를 제안합니다. GBER는 희소 보상 환경(sparse-reward environments)에서도 효과적으로 작동할 수 있도록 원래 알고리즘을 확장합니다. 특히 복잡한 구조가 요구되는 환경에서 탐색을 위한 능력을 강화하여 BER의 잠재력을 최대한 발휘하도록 합니다.

- **Technical Details**: GBER는 relabeling mechanism과 다양한 샘플링 전략(diverse sampling strategies)을 도입하여 BER의 성능을 향상시키고 있습니다. 이 알고리즘은 목표 조건(goal-conditioned) 딥 결정론적 정책 경량학습 알고리즘을 기반으로 하여 다양한 미로 탐색 환경에서 평가됩니다. 이러한 기술적인 개선은 GBER가 단순 초급 환경을 넘어 복잡한 환경에서도 잘 작동할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, GBER 알고리즘은 다양한 희소 보상 환경에서 베이스라인 알고리즘의 성능과 안정성을 크게 향상시킬 수 있음을 보여주었습니다. 특히, 고도로 구조적인 대칭성을 지닌 환경에서 더욱 두드러진 성능 향상이 관찰되었습니다. 이러한 결과는 GBER의 적용 가능성을 넓히고 미래 연구 방향에 대한 중요한 통찰을 제공합니다.



### PreNeT: Leveraging Computational Features to Predict Deep Neural Network Training Tim (https://arxiv.org/abs/2412.15519)
Comments:
          11 pages, Conference

- **What's New**: 이 논문은 PreNeT이라는 새로운 예측 프레임워크를 도입합니다. 이 프레임워크는 Transformer 기반의 모델 특히 Large Language Models (LLMs)의 훈련 최적화를 지원합니다. PreNeT은 이전에 검토되지 않은 하드웨어 인프라에서 훈련 시간을 정확하게 예측할 수 있는 기능을 가지고 있어, 신속하게 새로운 장비에 적용할 수 있습니다.

- **Technical Details**: PreNeT은 레이어별 매개변수(layer-specific parameters), 산술 연산(arithmetic operations), 메모리 사용(memory utilization) 등을 포함한 종합적인 계산 메트릭(computational metrics)을 통합합니다. 이 프레임워크는 다양한 신경망 계층의 특성을 분석하여 기존 예측 방법론을 향상시키는 복잡한 접근 방식을 사용합니다. 이를 통해 연구자들이 최적의 구성(optimal configurations)과 매개변수 설정(parameter settings), 하드웨어 사양(hardware specifications)을 결정할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, PreNeT은 현대의 최첨단 프레임워크에 비해 예측 정확도가 최대 72% 향상됨을 보여줍니다. 이러한 성과는 훈련 시간이 단축되고 비용 효율성이 극대화되는 데 기여할 수 있습니다. PreNeT은 특히 새로운 가속기 아키텍처에서의 성능 향상을 통해 많은 연구자들에게 실질적인 도움을 줄 것으로 기대됩니다.



### Novelty-Guided Data Reuse for Efficient and Diversified Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2412.15517)
Comments:
          AAAI 2025

- **What's New**: 이 논문은 Multi-Agent Reinforcement Learning (MARL)의 성능을 향상시키기 위한 혁신적인 샘플 재사용 접근법을 소개합니다. 제안된 방법은 Random Network Distillation (RND) 네트워크를 활용해 각 에이전트의 현재 상태의 참조 가치를 측정하고, 이를 통해 고유 데이터에 따라 샘플 업데이트 기회를 부여합니다. 이 방법은 샘플 효율성을 높이고 다양한 탐색 및 에이전트 행동을 촉진하는 데 기여합니다.

- **Technical Details**: 제안하는 Multi-Agent Novelty-GuidEd sample Reuse (MANGER) 방법은 에이전트의 관찰값의 신선도에 따라 정책 업데이트를 동적으로 조정합니다. 이를 통해 에이전트는 적은 상호작용으로도 샘플을 많이 재사용할 수 있어, 과거 데이터에서 더 많은 정보를 추출하게 됩니다. 또한 이 방법은 업데이트 빈도의 다양성을 통해 각 에이전트가 독특한 행동 전략을 학습할 수 있도록 하며, 이는 협동 작업의 성공률을 높이는 데 기여합니다.

- **Performance Highlights**: 다양한 복잡한 협동 시나리오에서 MANGER의 효과를 입증하기 위한 평가가 진행되었습니다. 특히 Google Research Football과 매우 어려운 StarCraft II 마이크로 매니지먼트 작업에서 MARL의 성능이 상당히 향상된 것으로 나타났습니다. 이러한 연구 결과는 협업 행동을 필요한 작업에서의 성공률을 크게 증가시킬 수 있는 가능성을 열어주는 중요한 발견입니다.



### RESQUE: Quantifying Estimator to Task and Distribution Shift for Sustainable Model Reusability (https://arxiv.org/abs/2412.15511)
Comments:
          The Annual AAAI Conference on Artificial Intelligence (AAAI), 2025

- **What's New**: 본 논문에서는 딥러닝 모델의 재훈련 비용을 예측하기 위한 새로운 지표인 RESQUE(REpresentation Shift QUantifying Estimator)를 제안합니다. RESQUE는 모델이 새로운 데이터 분포나 작업에 적응하는 데 필요한 자원의 추정치를 제공하여, 사용자들이 재훈련에 대한 정보에 기반한 결정을 내릴 수 있도록 도와줍니다. 이를 통해 지속 가능한 AI 개발을 지원하고 환경에 미치는 영향을 줄이는 데 기여하고자 합니다.

- **Technical Details**: RESQUE는 모델의 원래 분포와 새로운 분포 간의 표현(output) 변화량을 측정하여 예측합니다. 두 가지 버전이 있으며, 하나는 새로운 분포에 대한 RESQUEdist이고, 다른 하나는 특정 작업에 대한 RESQUEtask입니다. 이 두 지표는 학습 시 단일 전방 전파(forward propagation)만을 사용하여 계산되며, 이를 통해 훈련 비용과 환경적 영향을 최소화하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, RESQUE는 재훈련 비용과 높은 상관관계를 나타내며, 에너지 소비 및 탄소 배출과 같은 지속 가능성 지표와도 강한 상관성을 보입니다. 또한 RESQUE는 다른 모델 아키텍처와 무관하게 효과적으로 작동하며, 다양한 작업 및 데이터 세트에서 적용 가능성을 입증했습니다. 이는 AI 모델의 적응성을 높이는데 기여하며, 자원과 지속 가능성 목표 달성에 효과적입니다.



### Stylish and Functional: Guided Interpolation Subject to Physical Constraints (https://arxiv.org/abs/2412.15507)
Comments:
          Accepted by Foundation Models for Science Workshop, 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이번 논문에서는 Generative AI가 공학 디자인의 실제 적용에서 창의성과 실용성을 조화롭게 결합하는 방법을 제안합니다. 특히, 물리적 제약과 기능 요구사항을 반영하여 두 개의 입력 디자인을 조합하여 새로운 디자인을 생성하는 프레임워크를 개발했습니다. 이를 통해 자동차 휠 디자인의 회전 대칭성 같은 특정 기능 요구를 만족하는 생성 과정을 중점적으로 다루고 있습니다.

- **Technical Details**: 제안된 시스템은 Functional Constraints in InTerpolation (FIT)라는 기능적 제약 조건을 명시적으로 코드화합니다. FIT의 주요 구성 요소로는 잠재적 인터폴레이션(latent interpolation), 기능적 제약 정규화(functional constraint regularization), 그리고 생성된 이미지와 정규화된 이미지를 통합하는 프로젝션(projection)이 포함됩니다. 이 시스템은 Latent Diffusion Model (LDM)을 기반으로 하여, 중간 이미지를 생성하는 과정에서 기능적 제약을 적용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 관련 연구의 방법보다 더 높은 현실성을 가진(interpolations with higher realism) 생성된 이미지들을 만들어냅니다. Fréchet Inception Distance (FID) 지표로 평가할 때, 생성된 인터폴레이션은 더 낮은 FID 점수를 기록하여 사실성과 다양성이 향상되었습니다. 이러한 결과는 우리의 프레임워크가 회전대칭성과 같은 기능적 요구사항을 더 잘 충족할 수 있도록 도와줌을 보여줍니다.



### A Robust Prototype-Based Network with Interpretable RBF Classifier Foundations (https://arxiv.org/abs/2412.15499)
Comments:
          To appear at AAAI 2025. Includes the Appendix

- **What's New**: 이번 연구에서는 Prototype-Based Networks (PBN)의 심화판인 Deep Prototype-Based Networks (PBNs)를 분석합니다. 특히 Classification-by-Components (CBC) 접근 방식을 중심으로 해석가능성과 관련된 여러 문제를 다루며, 그러한 문제를 해결하기 위한 CBC의 확장을 제안합니다. 마지막으로, 제안된 모델의 강건성을 보장하는 손실 함수를 도출하여 이론적 근거를 제시합니다.

- **Technical Details**: 본 연구는 심층 PBN이 Deep Radial Basis Function (RBF) 분류기와 관련이 있음을 보여줍니다. 고찰된 모델들은 입력 데이터의 잠재 공간을 통해 특징 추출(Feature Extractor)과 유사성 계산(Similarity Calculation) 과정을 거쳐 클래스를 예측합니다. 이러한 과정에서 RBF 사용과 네거티브 추론(Negative Reasoning)의 역할을 분석하고, 기존 모델의 문제를 해결하기 위한 새로운 구조를 만듭니다.

- **Performance Highlights**: 실험 결과, 제안된 심층 PBN은 여러 기준에서 최고의 분류 정확도를 기록하며, 기존 접근 방식의 해석 가능성 문제를 해결했습니다. 또한, 얕은 PBN 변형은 기존의 얕은 PBN들보다 우수한 성능을 보이면서도 본질적으로 해석 가능하고 입증된 강건성을 갖추고 있습니다. 이러한 성능 향상은 PBN이 OOD(Out-Of-Distribution) 탐지에 적합하다는 사실을 뒷받침합니다.



### Understanding When and Why Graph Attention Mechanisms Work via Node Classification (https://arxiv.org/abs/2412.15496)
- **What's New**: 이 논문은 그래프 주의 메커니즘(Graph Attention Mechanisms)의 이론적 이해를 심화시키고, 노드 분류(node classification) 작업에서의 효과적인 조건을 제시합니다. 특히, 구조 노이즈(structure noise)와 특성 노이즈(feature noise)에 대해 적절히 정의함으로써, 주의 메커니즘이 효과적일 수 있는 상황을 분석했습니다. 또한 새로운 다층 Graph Attention Network (GAT) 구조를 제안하여 단일층 GAT보다 더 나은 성능을 보임을 입증합니다.

- **Technical Details**: 논문에서는 Contextual Stochastic Block Model (CSBM)을 사용하여 그래프 구조와 노드 특성을 시뮬레이션합니다. 구조 노이즈는 커뮤니티 간 연결 확률과 관련이 있으며, 특성 노이즈는 SNR의 역수로 정의됩니다. 이를 통해 노드 분류 작업을 통해 그래프 주의 메커니즘의 효과를 평가하고, 과적합(over-smoothing) 문제를 해결하는 방법도 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 다층 GAT는 CSBM에서 완벽한 노드 분류에 대해 SNR 요건을 크게 완화하여 단일층 GAT보다 우수한 성과를 나타냈습니다. 또한 합성 데이터와 실제 데이터셋 모두에서 이론적 발견을 검증하여 실제 응용의 가능성을 강조했습니다. 본 연구는 다층 GAT를 통해 완벽한 노드 분류를 달성할 수 있는 조건을 최초로 규명했습니다.



### Task-Specific Preconditioner for Cross-Domain Few-Shot Learning (https://arxiv.org/abs/2412.15483)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문은 Cross-Domain Few-Shot Learning (CDFSL) 분야에서 새로운 적응 메커니즘인 Task-Specific Preconditioned gradient descent (TSP)를 제안합니다. 기존의 방법들이 사용하던 고정 최적화 전략의 한계를 극복하기 위해, 우리 방법은 도메인 특화 전처리기 (Domain-Specific Preconditioners, DSPs)를 메타 학습하며 각 도메인의 특성을 반영합니다. 이러한 전처리기를 통해 목표 작업에 적응하도록 최적화를 진행하는 방식입니다.

- **Technical Details**: TSP는 메타 학습을 통해 각 도메인에서의 기초 조건을 형성하고, 이를 작업 계수(task-coefficients)와 선형 결합하여 작업 특정 전처리기(Task-Specific Preconditioner)를 생성합니다. 이 전처리기는 그래디언트 강하에 적용되며, 최적화 방향이 가장 급격한 하강 방향을 따르도록 특정 조건(positive definite)으로 제한됩니다. 이를 통해 각 목표 작업에 맞춘 적응적인 최적화를 가능하게 합니다.

- **Performance Highlights**: Meta-Dataset에서의 경험적 평가 결과, TSP는 다양한 실험 환경에서 기존의 최고 성능을 능가하는 성과를 달성했습니다. 이는 TSP가 Cross-Domain FSL 작업에서 일반화 능력을 높임을 입증하며, 동시에 meta-learning의 효과적인 활용 가능성을 보여줍니다. 전반적으로, 이 연구는 FSL 분야의 발전에 크게 기여할 것으로 기대됩니다.



### Non-Uniform Parameter-Wise Model Merging (https://arxiv.org/abs/2412.15467)
Comments:
          9 pages, 1 figure, to be published in the Proceedings of the 9th IEEE Special Session on Machine Learning on Big Data (MLBD 2024)

- **What's New**: 본 논문에서는 Non-uniform Parameter-wise Model Merging(NP Merge)이라는 새로운 접근 방식을 제안합니다. NP Merge는 기울기 기반 최적화를 통해 각 매개변수가 최종 모델에 기여하는 정도를 학습하여 모델을 병합합니다. 이는 기존 방법들보다 더 유연한 방법으로, 다양한 아키텍처의 모델들을 여러 설정에서 효과적으로 병합할 수 있음을 입증하고 있습니다.

- **Technical Details**: 이 방법은 매개변수 수준에서 보간 계수를 학습하여 각 매개변수의 기여도를 조절할 수 있도록 하여, 서로 다른 데이터셋에서 훈련된 모델 간의 병합을 더욱 효과적으로 수행합니다. NP Merge는 모델 정렬을 위해 기존 방법들과 함께 사용할 수 있으며, 이를 통해 좋은 성능을 유지할 수 있습니다. 논문에서 제안한 NP Merge는 입력 데이터에 따라 성능 안정성을 지속적으로 분석하며, 기존 모델들이 동일한 초기화에서 유래된 경우 더 나은 성능을 보임을 강조합니다.

- **Performance Highlights**: NP Merge는 반복적인 쌍별 병합을 통해 여러 모델을 병합할 수 있는 확장성을 제공하며, 이는 최신 기법(SOTA)들보다 우수한 성능을 보여줍니다. 각 기법이 다양한 데이터 분포에 따라 적합한 상황에서 사용될 수 있도록 연구하였으며, 병합된 모델을 몇 에폭에 걸쳐 미세 조정하는 방법과의 성능 비교를 통해 일반화 측면의 이점도 함께 논의됩니다. 최종적으로, NP Merge는 다양한 환경에서 성능을 극대화하는 데 기여합니다.



### Time Will Tell: Timing Side Channels via Output Token Count in Large Language Models (https://arxiv.org/abs/2412.15431)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)에서 민감한 정보를 추출할 수 있는 새로운 사이드 채널(side-channel)을 제안합니다. 이 사이드 채널은 LLM의 응답에서 출력 토큰 수를 기반으로 하여 정보 유출을 가능하게 합니다. 특히, 기계 번역 작업에서 타겟 언어를 복원하거나 분류 작업에서 출력 클래스를 복원하는 공격을 수행하는 데 사용됩니다.

- **Technical Details**: LLM의 자동 회귀 생성(auto-regressive generation) 특성을 활용하여 공격자는 응답 시간 측정 또는 네트워크를 통해 출력 토큰 수를 회복할 수 있습니다. 이 연구에서는 Tower, M2M100, MBart50과 같은 다국어 모델에서 75% 이상의 정밀도로 언어를 식별할 수 있음을 보여주었습니다. 추가적으로, 텍스트 분류 작업에서 LLM의 출력 설명 길이에서 발생하는 내재적 편견이 민감한 정보를 누출하는 경향이 있음을 입증했습니다.

- **Performance Highlights**: 실제로, LLM의 출력 토큰 수에 따라 기계 번역과 텍스트 분류 작업에서 높은 성공률을 기록했습니다. 예를 들어, Gemma2-9B 모델에서 81.4%의 성공률을 달성했으며, GPT-4o 모델의 원격 공격에서도 74.7%의 성공률을 보였습니다. 이러한 성과는 특정 작업의 출력 클래스나 언어를 유출하는 데 있어 사이드 채널 공격이 매우 효과적임을 입증합니다.



### Offline Safe Reinforcement Learning Using Trajectory Classification (https://arxiv.org/abs/2412.15429)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 오프라인 안전 강화 학습(offline safe reinforcement learning)에서 신뢰할 수 있는 행동을 학습하기 위한 새로운 접근 방식을 제안합니다. 기존 방법들이 환경과의 직접적인 상호작용 없이 안전 제한 조건을 충족하기 어려운 것을 해결하기 위해, 저자들은 사전 수집된 데이터셋을 바탕으로 바람직한 경로와 바람직하지 않은 경로로 분리하고, 이 정보를 기반으로 정책을 학습합니다.

- **Technical Details**: 연구진은 두 단계로 이루어진 알고리즘을 제시합니다. 첫 번째 단계에서는 사전 수집된 데이터셋을 바탕으로 바람직한 경로와 바람직하지 않은 경로의 두 개의 하위 집합으로 나눕니다. 두 번째 단계에서는 분류기를 통해 바람직한 경로를 높은 점수를, 바람직하지 않은 경로에는 낮은 점수를 부여하여 정책을 직접 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 38개의 연속 제어 작업에서 여러 기존 방법들과 비교하여 더 높은 보상과 안전 제약 조건 준수를 달성하며 우수한 성능을 보여줍니다. 전체적으로 이 연구는 안전한 행동을 학습하는 방법론 관련하여 유망한 방향을 제시하며, 결과적으로 기존 최첨단 기법들보다 높은 성능을 입증했습니다.



### AdaCred: Adaptive Causal Decision Transformers with Feature Crediting (https://arxiv.org/abs/2412.15427)
Comments:
          Accepted to 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)

- **What's New**: 이 논문에서는 AdaCred라는 새로운 접근 방식을 소개합니다. AdaCred는 짧은 행동-보상-상 상태 시퀀스를 기반으로 인과 그래프(causal graph) 형식으로 경로를 표현합니다. 이 모델은 낮은 중요도를 갖는 표현을 제거하고 작업에 가장 관련성이 높은 표현만 남김으로써 적응적으로 제어 정책을 학습합니다.

- **Technical Details**: AdaCred는 인과 그래프를 사용하여 각 시퀀스 표현에 신뢰도를 지정하고 낮은 중요도의 구성 요소를 제거합니다. 이 과정에서, 정책 학습을 위해 중복된 인과 변수들을 식별하고 제거합니다. 이러한 적응형 처리 방식은 모델이 관련 있는 표현에 집중할 수 있게 하여 메모리 사용을 효율화합니다.

- **Performance Highlights**: 실험 결과, AdaCred 기반 정책은 짧은 시퀀스를 필요로 하며 기존 방법들보다 일관되게 더 나은 성능을 보여줍니다. 특히, Atari와 Gym 환경에서의 벤치마크 실험에서 정책 학습이 더 효과적으로 이루어져 성능이 크게 향상되었습니다. 이는 오프라인 강화 학습과 모사 학습(imitation learning) 작업 모두에서 입증됩니다.



### Dimension Reduction with Locally Adjusted Graphs (https://arxiv.org/abs/2412.15426)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 고차원 데이터 집합의 군집을 식별하는 새로운 차원 축소 알고리즘인 LocalMAP을 소개합니다. LocalMAP은 그래프를 동적으로 그리고 지역적으로 조정하여 복잡한 데이터 구조를 더욱 정확하게 반영합니다. 이 알고리즘은 클러스터 간의 경계를 명확하게 정의하여 다른 DR 방법에서 간과할 수 있는 진정한 클러스터를 분리해냅니다.

- **Technical Details**: LocalMAP은 고차원 거리의 신뢰성을 개선하기 위해 잘못된 긍정적 에지를 감지하고 제거합니다. 또한, 멀리 떨어진 데이터 포인트 간의 부정적 에지를 동적으로 추가하여 클러스터 경계를 선명하게 합니다. 이러한 방식으로, LocalMAP은 디멘션 리덕션 과정 중에 그래프의 구조를 개선해 진정한 클러스터를 식별합니다.

- **Performance Highlights**: 사례 연구를 통해 LocalMAP은 다른 차원 축소 접근법보다 진정한 클러스터를 보다 신뢰성 있게 찾아낼 수 있음을 보여주었습니다. 예를 들어, MNIST 데이터셋에 대한 시각화에서 t-SNE 및 UMAP과 같은 기존 방법들은 클러스터 간의 명확한 경계를 형성하지 못하는 반면, LocalMAP은 잘 정의된 경계를 가진 고품질 DR 임베딩을 생성합니다.



### LG-Sleep: Local and Global Temporal Dependencies for Mice Sleep Scoring (https://arxiv.org/abs/2412.15412)
- **What's New**: 이번 연구에서는 LG-Sleep이라는 새로운 딥 뉴럴 네트워크 아키텍처를 소개합니다. 이 모델은 쥐의 수면 단계를 분류하기 위해 전기 생리 신호인 EEG를 사용하며, 다양한 주체에 대한 일반화가 가능하도록 설계되었습니다. LG-Sleep은 수면 데이터를 깨우기, 빠른 안구 움직임(REM) 수면, 비빠른 안구 움직임(NREM) 수면의 세 단계로 나누어 분석합니다.

- **Technical Details**: LG-Sleep은 시간 분포형 CNN과 LSTM을 통합하여 EEG 신호의 지역적 및 전역적 시간적 전환을 캡처합니다. CNN을 사용해 세밀한 지역 신호를 추출하고, LSTM 블록을 통해 장기간의 전역 전환 정보를 분석합니다. 이 과정에서 자동 인코더-디코더 방식으로 최적화되어 훈련 샘플 수의 제한에 적응할 수 있습니다.

- **Performance Highlights**: 실험 결과, LG-Sleep은 기존의 딥 뉴럴 네트워크에 비해 뛰어난 성능을 보였습니다. 특히, 훈련 샘플 수가 제한적일 때도 다양한 수면 단계에서 우수한 성능을 유지하며, 수면 단계 간 불균형 문제와 개체 차이에 효과적으로 대응합니다.



### Granger Causality Detection with Kolmogorov-Arnold Networks (https://arxiv.org/abs/2412.15373)
Comments:
          8 pages, 2 figures, 2 tables

- **What's New**: 이번 연구에서는 Kolmogorov-Arnold Networks (KANs)를 활용하여 Granger 인과관계를 탐지하기 위한 신경망 모델을 제안합니다. 기존의 Multilayer Perceptrons (MLP)와 KANs의 성능 비교를 통해 KAN의 우수성을 강조합니다. 본 연구는 Granger causality KAN (GC-KAN)이라는 프레임워크를 개발하며, 이를 Granger 인과관계 탐지에 맞추어 맞춤형 훈련 접근법과 함께 제시합니다.

- **Technical Details**: GC-KAN은 다변량 Granger causality 탐지를 위해 설계된 프레임워크로, KANs의 비선형성 및 희소성 유도 정규화를 활용합니다. 이 모델은 입력과 첫 번째 은닉층 연결의 비업을 자동적으로 식별하고, 관련 없는 입력 특성을 정확히 0 가중치로 할당하여 인과관계를 파악합니다. 연구에서는 Vector Autoregressive (VAR) 모델과 비선형 Lorenz-96 시스템에서 GC-KAN의 성능을 평가합니다.

- **Performance Highlights**: KAN은 MLP에 비해 해석 가능한 Granger 인과관계를 식별하는 데 있어 더 뛰어난 성능을 보입니다. 특히 고차원 환경에서 희소한 Granger causality 패턴을 인식하는 능력에서 두드러진 성과를 나타냈으며, 이는 물리 시스템의 동적 법칙을 발견하는 AI의 응용 가능성을 보여줍니다. 실험 결과 GC-KAN은 전통적인 MLP와 비교하여 더 높은 정확도와 해석성을 제공하는 것으로 나타났습니다.



### A Multi-Fidelity Graph U-Net Model for Accelerated Physics Simulations (https://arxiv.org/abs/2412.15372)
Comments:
          21 pages, 11 figures

- **What's New**: 본 연구에서는 Multi-Fidelity U-Net이라는 새로운 GNN 아키텍처를 제안합니다. 이 기법은 다중 신뢰도(multi-fidelity) 방법의 이점을 활용하여 GNN 모델의 성능을 향상시키도록 설계되었습니다. 기존의 방법들과 비교했을 때, 이 아키텍처는 훈련 데이터 요구사항을 줄이면서도 높은 정확성을 유지하는 데 탁월한 성과를 보입니다.

- **Technical Details**: Multi-Fidelity U-Net 아키텍처는 복잡한 기하학을 관리할 수 있는 GNN의 능력을 활용하여 서로 다른 신뢰도 수준 간에 정보 흐름을 가능하게 합니다. 이 구조는 U-Net 아키텍처를 기반으로 하며, 각 레벨이 다양한 신뢰도를 가진 그래프를 처리합니다. 또한, Multi-Fidelity U-Net Lite라는 더 빠른 버전을 제안하여 훈련 시간을 약 35% 단축시키면서도 정확도는 2~5% 감소하는 결과를 보여줍니다.

- **Performance Highlights**: 제안된 모델은 2D 처짐 평가, 2D 판의 응력 집중 분석, 산업 표준 공기역학 데이터 세트를 이용한 대규모 3D 유체 역학 시뮬레이션 등 다양한 수치 예제를 통해 우수한 성능을 입증하였습니다. 이 연구는 Multi-Fidelity U-Net 아키텍처가 기존의 단일 신뢰도 GNN 모델보다 더 나은 성능을 보인다는 점을 강조하며, 높은 신뢰도의 수치 시뮬레이션을 효과적으로 대체할 수 있는 가능성을 보여줍니다.



### LISA: Learning-Integrated Space Partitioning Framework for Traffic Accident Forecasting on Heterogeneous Spatiotemporal Data (https://arxiv.org/abs/2412.15365)
- **What's New**: 이 논문에서는 교통사고 예측의 복잡성을 해결하기 위한 새로운 LISA(Learning-Integrated Space Partitioning Framework)를 제안하고 있습니다. 이 프레임워크는 모델 학습과 함께 공간 partition을 학습하여 예측 정확도를 높이는 혁신적인 접근 방식을 사용합니다. 기존의 방법들이 필요로 했던 사전 정의된 공간 partition 대신, LISA는 데이터에 기반하여 partition을 자동으로 생성합니다.

- **Technical Details**: LISA 프레임워크는 다양한 딥러닝 네트워크와 함께 작동할 수 있는 일반적인 체계를 갖추고 있습니다. 특히 이 프레임워크는 모델 학습 중에 교통사고 예측의 heterogeneous한 패턴을 캡쳐하여, 예측 오차를 지속적으로 감소시키는 방식으로 partition을 형성합니다. Iowa 주의 실제 사고 데이터 세트를 사용한 실험 결과, LISA가 기존 네트워크보다 평균 13.0%의 정확도 향상을 달성했음을 보여줍니다.

- **Performance Highlights**: 본 연구의 실험 결과, LISA 프레임워크는 교통사고 예측에서의 기본 네트워크의 성과를 상당히 향상시킬 수 있음을 입증합니다. 이는 기존의 정형화된 머신러닝 방식으로는 해결할 수 없는 문제들의 해결책을 제시하며, 각기 다른 환경적 요인에 따른 사고 패턴을 효과적으로 파악할 수 있게 합니다. 이처럼 실시간 데이터와 딥러닝 기술을 활용한 이번 연구는 교통 안전 관리 및 응급 대응에 기여할 것으로 기대됩니다.



### Spatiotemporally Coherent Probabilistic Generation of Weather from Clima (https://arxiv.org/abs/2412.15361)
Comments:
          15 pages, 6 figures, additional supplementary text and figures

- **What's New**: 본 논문은 고해상도 재분석 데이터에 기반한 점수 기반 확산 모델(score-based diffusion model)을 활용하여 지역 날씨 역학의 통계적 특성을 캡처하는 새로운 생성적 접근 방식을 제안합니다. 기존의 통계적 다운스케일링 방법이 작은 규모의 현상을 시공간적으로 분리하여 추정하는 데 반해, 본 연구는 장기 시간 동안의 고해상도 날씨 동역학을 보존하는 방법을 모색합니다. 이 모델은 기후 모델 데이터를 기반으로 일관된 날씨 패턴을 생성하여, 경량 정보와의 일치를 높이는 데 중점을 두고 있습니다.

- **Technical Details**: 구조적 프레임워크를 통해 인공지능 모델이 기후 데이터와 재분석 데이터를 직접 매칭하여 예측 능력 및 불확실성 정량화를 평가합니다. 타겟 변수에 따른 예측의 정확도를 확인하기 위해 확률적 모델이 시스템적 편향을 제거하며, 예측된 기후 데이터의 분포는 재분석 데이터의 분포와 일치합니다. 또한, 기후 모델의 편향을 보정하기 위한 변별적 정량화 접근 방식을 채택하여, 모델의 예측 정확도 및 신뢰성을 높입니다.

- **Performance Highlights**: 모델은 여러 위치에서의 날씨 경로의 공간적 및 시간적 변화를 정확하게 예측하며, 기후 데이터의 낮은 해상도로부터 의미 있는 지역 정보를 회복하는 능력을 보여줍니다. 극한 사건의 재현을 평가한 결과, 제안된 모델은 물리적으로 신뢰할 수 있는 예측을 생성하며, 기존 기후 예측 실패를 극복할 수 있는 잠재력을 지니고 있음을 입증했습니다. 이러한 결과를 바탕으로, 에너지 부문 등에서의 정확한 예측을 위한 다운스케일링 기법 확립에 기여할 것으로 기대됩니다.



### GeoPro-Net: Learning Interpretable Spatiotemporal Prediction Models through Statistically-Guided Geo-Prototyping (https://arxiv.org/abs/2412.15353)
- **What's New**: 이번 논문에서는 GeoPro-Net이라는 새로운 intrinsically interpretable spatiotemporal 모델을 제안합니다. 이 모델은 다양한 출처의 spatiotemporal 특징을 기반으로 한 예측 과정을 해석할 수 있도록 설계되었습니다. Geo-concept convolution operation을 도입하여 예측 패턴을 추출하고, 해석 가능한 채널 융합 및 지리적 기반 풀링을 통해 축약합니다.

- **Technical Details**: GeoPro-Net은 다중 출처의 spatiotemporal 데이터를 처리하여 복잡한 의존성과 의미론을 이해하기 쉽게 변환합니다. 이 과정에서 통계적 검사(statistical tests)를 통해 Geo-concepts를 추출하고, 이는 예측 과정의 해석을 용이하게 합니다. 또한, 여러 출력 클래스를 위한 예측을 수행하는 프로토타입( prototypes) 집합을 학습함으로써, 실제 사례에 대한 해석을 가능하게 합니다.

- **Performance Highlights**: 총 네 개의 실제 데이터셋에 대한 실험을 통해 GeoPro-Net은 경쟁력 있는 예측 성능을 보여주었으며, 기존 최첨단 모델에 비해 우수한 해석 가능성을 가지고 있음을 입증하였습니다. 이 모델은 도시의 공공 안전 및 교통 관리 등 다양한 분야에서의 활용 가능성을 제시합니다.



### Large Language Models on Small Resource-Constrained Systems: Performance Characterization, Analysis and Trade-offs (https://arxiv.org/abs/2412.15352)
- **What's New**: 최근 몇 년 동안 Generative AI, 특히 대형 언어 모델(LLMs)이 일반 소비자에게 더 많이 사용되고 있습니다. 이 논문은 NVIDIA의 Jetson Orin 장치를 통해 로컬에서 LLM을 실행할 수 있는 가능성과 필요성에 대해 다루고 있습니다. 연구자들은 최근 상용화된 임베디드 하드웨어에 대한 '기준선(baseline)' 특성을 제공하고, Jetson 하드웨어에서 LLM의 배치 테스트를 용이하게 하는 간단한 유틸리티를 제시합니다. 이를 통해 기존의 기술적 한계를 극복할 새로운 접근 방식을 모색합니다.

- **Technical Details**: 연구에서는 Pythia LLM 모델을 사용하여 Jetson Orin 장치에서의 성능을 평가했습니다. 모델의 파라미터 수는 70만에서 14억까지 다양하며, 다양한 소프트웨어 및 하드웨어 매개변수를 조정하여 실험을 수행했습니다. 연구진은 LLM 로딩 및 생성 시 필요한 지연 시간(latency), 전력 소비(power), 메모리 사용량(memory), 그리고 모델의 정확도(accuracy)를 측정했습니다. 이러한 데이터를 바탕으로, LLM의 에너지 소비(energy)와 단어 생성 당 소요 시간(time per token)도 추정되었습니다.

- **Performance Highlights**: 실험을 통해 Jetson Orin 장치에서의 LLM 실행 시 발생하는 성능 트레이드오프와 최적화 선택을 시각화하였습니다. Jetson 개발 키트를 활용하여 여러 다른 하드웨어 및 소프트웨어 구성을 비교하고, 5555번 반복 테스트를 통해 초기 실행과 이후 실행의 동작을 비교했습니다. 이 연구는 임베디드 시스템에서 LLM의 실용성을 높일 수 있는 기초 자료를 제시하여, 향후 더 나은 LLM 배포를 위한 연구를 지원할 것입니다.



### Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models (https://arxiv.org/abs/2412.15341)
- **What's New**: 이 논문에서는 쪼갠(Pruned) 확산 모델을 위한 새로운 이수준 최적화(bilevel optimization) 프레임워크를 제안합니다. 이 프레임워크는 미세 조정(fine-tuning)과 잊기(unlearning) 프로세스를 통합하여 생성 품질을 유지하면서 불필요한 콘텐츠 생성을 선택적으로 억제합니다. 기존의 두 단계 접근법에 비해 효율적이고 안전한 배포가 가능하도록 돕습니다.

- **Technical Details**: 제안한 프레임워크는 미세 조정 데이터셋에서 모델의 생성 능력을 복구하기 위해 표준 증류(distillation) 및 확산 손실 최소화를 수행하는 하위 단계 최적화를 포함합니다. 상위 단계는 모델이 불필요한 개념을 생성하지 않도록 방향성을 제시합니다. 이 방식은 다양한 가지치기(pruning) 방법과 색인 제거(concept unlearning) 기술과 호환됩니다.

- **Performance Highlights**: 광범위한 아티스트 스타일 및 NSFW 콘텐츠 제거 작업에 대한 평가를 통해, 제안한 이수준 방법이 두 단계 접근법보다 상당히 우수함을 입증합니다. 높은 생성 품질을 유지하면서도 효과적인 개념 억제가 가능함을 보여주며, 실제 통제된 배포 환경에서의 유효성을 강조합니다.



### PCA-Featured Transformer for Jamming Detection in 5G UAV Networks (https://arxiv.org/abs/2412.15312)
- **What's New**: 이 논문에서는 UAV(무인 항공기)가 통합된 5G 네트워크에서의 재밍 공격(jamming attack)을 탐지하기 위해 혁신적인 transformer 기반의 딥 러닝 아키텍처를 제안합니다. 이 시스템은 Principal Component Analysis (PCA) 기능을 추가하여 복잡한 신호 특성을 더 효과적으로 모델링할 수 있도록 설계되었습니다. 기존 방법들이 놓치는 공격 패턴을 포착하기 위해 self-attention 메커니즘을 활용해 초고속 훈련과 우수한 탐지 정확도를 달성하는 것이 핵심입니다.

- **Technical Details**: 제안된 시스템은 인증된 UAV가 소규모 셀 네트워크 환경 내에서 재밍 활동을 탐지할 수 있도록 설계되었습니다. 수집된 데이터셋은 Line-of-Sight (LoS)와 Non-Line-of-Sight (NLoS) 두 가지 통신 시나리오를 포함하며, UAV 이동 패턴, 작동 속도, 공격 강도 및 사용자 밀도와 같은 변수를 변형하여 구성되었습니다. 모델 훈련에 사용되는 데이터셋은 값이 불균형한 공격과 비공격 시나리오를 포함하여, 신호 측정의 시계열 데이터를 효율적으로 분석할 수 있도록 배치 크기 조정(batch size scheduler) 및 데이터 그룹화(chunking) 기법 등으로 최적화되었습니다.

- **Performance Highlights**: 시뮬레이션 결과, 이 접근 방식은 Line-of-Sight (LoS)에서 90.33%의 탐지 정확도를, Non-Line-of-Sight (NLoS)에서는 84.35%의 정확도를 기록했습니다. 제안된 모델은 기존 기계 학습 기법 및 XGBoost (XGB) 분류기와 비교했을 때 약 4% 더 우수한 성능을 보였으며, 트레이닝 속도에서는 최대 10배의 향상을 이끌어냈습니다. 이 연구는 UAV 통신의 보안을 강화하는데 기여하는 기반을 마련하고 있습니다.



### Re-evaluating Group Robustness via Adaptive Class-Specific Scaling (https://arxiv.org/abs/2412.15311)
- **What's New**: 이 논문에서는 그룹 분포적으로 강건한 최적화를 통해 데이터셋의 편향을 해결하기 위한 새로운 방식을 제시합니다. 특히 클래스 특화 스케일링(class-specific scaling)과 인스턴스별 적응 스케일링(instance-wise adaptive scaling) 기법을 도입하여 강건 정확도(robust accuracy)와 평균 정확도(average accuracy) 간의 상충 관계를 효과적으로 조절할 수 있도록 하였습니다. 또한, 기존 디바이싱 알고리즘(debiasing algorithms)에 추가 훈련 없이 적용 가능한 간단한 후처리 기법을 도입하여 성능을 향상시킬 수 있는 방법을 보여줍니다.

- **Technical Details**: 주요 기술적 제안은 클래스 특화 스케일링과 인스턴스별 적응 스케일링의 결합으로, 이는 예측 점수(prediction scores)에 대한 클래스 특화 조정을 통해 강건 정확도와 평균 정확도 사이의 균형을 맞춥니다. 새로운 포괄적 성능 평가 메트릭인 robust coverage를 통해 두 정확도 간의 상충을 스칼라 값으로 요약하여, 기존 디바이싱 알고리즘의 성능을 종합적으로 평가할 수 있는 기준을 제공합니다. 이러한 새로운 접근 방식은 이전 연구들과 달리 추가 훈련이 필요 없는 특징으로 매우 효율적인 성능 최적화를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방식은 기존의 그룹 분포적으로 강건한 최적화 방법과 비교했을 때 동일한 수준 이상의 성능을 발휘하는 것으로 나타났습니다. 특히, 간단한 클래스 특화 스케일링을 적용한 ERM(Empirical Risk Minimization) 기본 모델이 최신 디바이싱 기법에 맞먹거나 심지어 뛰어난 성능을 보여주었다는 점이 강조됩니다. 이 논문은 컴퓨터 비전 및 자연어 처리 분야의 다양한 데이터셋을 통해 이러한 기법의 효과를 검증하였습니다.



### TinyLLM: A Framework for Training and Deploying Language Models at the Edge Computers (https://arxiv.org/abs/2412.15304)
- **What's New**: 이번 연구에서는 대규모 파라미터를 가진 언어 모델의 필요성을 줄이기 위해, 더 작은 모델(약 30-120M 파라미터)이 특정 작업에서 더 나은 성능을 발휘할 수 있다고 가설을 제시했습니다. 이를 통해 모바일 및 엣지 디바이스에서의 모델 배포 가능성을 탐구하였습니다. 또한, 사전 훈련(pre-training) 및 미세 조정(fine-tuning) 과정에서 사용하는 데이터를 신중하게 선별하는 것이 성능에 미치는 영향을 분석했습니다.

- **Technical Details**: 저자들은 여러 기본 모델(baseline models)을 체계적으로 훈련시킨 결과, 소형 모델이 엣지 디바이스에서 직접 실행할 수 있으며 높은 토큰율(token rate)과 정확도를 달성할 수 있음을 발견했습니다. 이러한 연구는 주로 감지 애플리케이션(sensing applications)을 지원하기 위한 엣지 디바이스 모델 배포 맥락에서 진행되었습니다. 이를 통해 기존의 대규모 모델 대체 가능성을 제시하고, 메모리와 처리 요구 사항을 줄일 수 있는 방안을 마련했습니다.

- **Performance Highlights**: 연구 결과, 작은 모델들이 높은 성능을 보이면서도 리소스 요구사항을 상당히 감소시킬 수 있음을 확인했습니다. 이러한 모델은 네트워크 호출을 통한 원격 추론(remote inference)으로 인한 지연(latency)이나 불확실한 네트워크 연결 문제를 피할 수 있어, 사용자에게 더욱 효율적인 솔루션을 제공합니다. 마지막으로, 사용자 맞춤형 모델 훈련 및 엣지 디바이스 배포를 위한 프레임워크를 개발하였습니다.



### Tokenphormer: Structure-aware Multi-token Graph Transformer for Node Classification (https://arxiv.org/abs/2412.15302)
Comments:
          Accpeted by AAAI 2025

- **What's New**: 본 논문은 Graph Neural Networks (GNNs)에서 발생하는 한계를 극복하기 위해 Structure-aware Multi-token Graph Transformer, 즉 Tokenphormer를 제안합니다. Tokenphormer는 Fine-grained의 다양한 token을 생성하여 지역적(local) 및 구조적(structural) 정보를 효과적으로 포착하고, 그래프를 체계적으로 탐색할 수 있게 합니다. 특히, mixed walks를 활용하여 여러 종류의 walk-token을 생성하여 그래프 내에서 구조 및 문맥 정보를 유연하게 수집합니다.

- **Technical Details**: 이 연구에서는 Tokenphormer를 통해 node classification을 위한 새로운 접근 방식을 제안합니다. 기존의 message passing 기법과 전통적인 Transformer의 한계를 극복하기 위해, walk-token, SGPM-token, hop-token 등 세 가지 종류의 token을 도입하여 다양한 스케일에서 정보 추출을 가능하게 합니다. walk-token은 네 가지 다른 walk 타입을 활용하여 그래프를 탐색하고, SGPM-token은 전역 정보를 보장하며, hop-token은 지역 정보를 담아냅니다.

- **Performance Highlights**: 실험 결과에 따르면 Tokenphormer는 대부분의 동종(homogeneous) 및 이종(heterogeneous) 그래프 벤치마크 데이터셋에서 기존의 최첨단 그래프 Transformer 및 MPNN 방법보다 뛰어난 성능을 보여줍니다. 이는 Tokenphormer가 현실 세계의 다양한 그래프 구조에 적응 가능하며, 다양한 수준의 세부정보를 캡처할 수 있는 능력을 가진다는 것을 의미합니다.



### Parametric $\rho$-Norm Scaling Calibration (https://arxiv.org/abs/2412.15301)
- **What's New**: 본 연구에서는 구조적 과신(overconfidence) 문제를 완화하기 위해 새로운 포스트 프로세싱 파라메트릭 보정 방법인 $\rho$-Norm Scaling을 소개합니다. 이 방법은 출력의 진폭(amplitude)을 조정하여 정확도를 유지하면서 데이터셋의 모델 신뢰도를 개선합니다. 또한, 샘플 수준의 불확실성(distribution uncertainty)을 반영할 수 있도록 확률 분포 정규화(probability distribution regularization)을 포함시킴으로써 보정된 확률 분포가 원래의 분포와 유사하게 유지되도록 합니다.

- **Technical Details**: 모델 출력의 신뢰도 보정은 출력의 불확실성 추정치를 정제하여 더 정확한 확률 예측을 가능하게 합니다. 기존의 파라메트릭 보정 방법들이 데이터를 처리할 때 발생하는 비효율성을 해결하기 위해, 본 연구는 새로운 $\rho$-Norm Scaling 모델을 제안하며, 이 모델은 원래 분포와 보정된 분포 간의 유사도를 고려하는 멀티 레벨 손실(multi-level loss)을 도입하여 최적화합니다. 이로 인해 출력-확률 매핑(output-probability mapping)이 더욱 효과적으로 학습되며, 과적합이 방지됩니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법은 여러 데이터셋 및 모델에 대해 최첨단(calibration performance) 보정 성능을 달성하여, 다른 기존 방법들에 비해 현저한 개선을 보여줍니다. 특히, bin-level와 instance-level 불확실성의 차이를 극복함으로써 더 정확한 모델 신뢰도를 제공합니다. 이러한 결과는 모델의 전반적인 예측 성능을 강화하는 데 기여할 것입니다.



### A Universal Model for Human Mobility Prediction (https://arxiv.org/abs/2412.15294)
- **What's New**: 이번 연구에서 우리는 인간 이동 예측을 단일 모델로 통합하는 새로운 접근 방식을 제시합니다. 제안된 유니버설 모델인 UniMob은 개인 궤적(trajectory)과 군집 흐름(crowd flow) 데이터를 모두 처리할 수 있어, 다양한 이동 데이터를 통합하여 예측 성능을 개선합니다. 이를 통해 기존의 특정 작업에 제한된 이동 예측 방법들의 한계를 극복하는 데 기여합니다.

- **Technical Details**: UniMob은 여러 관점(multi-view)의 이동 행동을 활용하는 다중 뷰 토크나이저(mobility tokenizer)를 설계하여, 추상적인 시공간 토큰(spatiotemporal tokens)으로 궤적과 흐름 데이터를 변환합니다. 이 모델은 확산 변환기(diffusion transformer) 아키텍처를 사용해 다양한 이동 데이터의 시공간 동적 패턴을 캡처하며, 개인과 집단 간의 상호작용을 촉진하는 혁신적인 양방향 정렬 메커니즘을 구현합니다. 이 메커니즘은 궤적과 흐름 데이터 사이의 공통 패턴을 학습하게 해줍니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험이 수행되었으며, UniMob은 궤적과 흐름 예측에서 기존의 최고 수준 모델들보다 뛰어난 성능을 발휘했습니다. 특히, 노이즈가 많고 데이터가 부족한 상황에서도 MAPE에서 14% 이상, Accuracy@5에서 25% 이상의 성능 향상을 달성했습니다. 이러한 결과는 UniMob의 견고성과 확장성을 입증하는 데 중요한 역할을 합니다.



### Personalized Representation from Personalized Generation (https://arxiv.org/abs/2412.16156)
Comments:
          S.S. and J.C contributed equally; S.B. and P.I. co-supervised. Project page: this https URL

- **What's New**: 이 논문은 현대 비전 모델을 개인화된 비전 태스크에 적용하는 방법을 탐구합니다. synthetic data와 T2I diffusion model의 발전을 활용하여, 적은 수의 실제 예시로부터 개인화된 이미지를 생성할 수 있는 가능성을 살펴봅니다. 개인화된 synthetic data를 통해 개인화된 representations를 학습하는 도전을 정식화하고, 이와 관련된 평가 도구도 도입합니다.

- **Technical Details**: 컴퓨터 비전에서 representation learning은 개체나 의미 개념에 대한 범용 인코딩을 학습하는 것을 목표로 합니다. 본 연구에서는 사용자가 제공하는 몇 개의 실제 이미지로부터 개인화된 representation을 학습할 수 있는지에 대해 질문하며, 이를 위해 contrastive learning 접근 방식을 제안합니다. 데이터의 부족성과 객체 인식의 미세한 차이를 해결하기 위한 다양한 방법론을 논의합니다.

- **Performance Highlights**: 우리의 제안된 방법은 다양한 downstream 태스크에서 개인화된 representation learning 성능을 크게 향상시킵니다. 데이터셋 전반에 걸쳐 pretrained counterparts에 비해 매우 우수한 성과를 보이며, 기존 데이터셋의 개정 및 새로운 dataset인 PODS를 소개합니다. 또한, 적은 계산 자원에서도 비슷한 결과를 얻을 수 있음을 보여줍니다.



### LEDA: Log-Euclidean Diffeomorphic Autoencoder for Efficient Statistical Analysis of Diffeomorphism (https://arxiv.org/abs/2412.16129)
- **What's New**: 이 논문에서는 Log-Euclidean Diffeomorphic Autoencoder (LEDA)라는 혁신적인 프레임워크를 제안하여 변형 필드의 주요 로그를 효율적으로 계산할 수 있도록 하며, 이는 복잡하고 비선형적인 변형을 분석하는 데 중요한 역할을 합니다. LEDA는 diffeomorphism 그룹의 작용 법칙을 존중하는 선형 잠재 공간 내에서 작동하여 통계 분석의 강력함과 적용 가능성을 향상시킵니다. 또한, 변형 필드의 정확한 잠재 표현을 보장하는 역일관성(inverse consistency) 제약을 적용하기 위한 손실 함수도 도입했습니다.

- **Technical Details**: 이 프레임워크는 연속적인 제곱근 예측을 통해 변형 필드의 주 로그를 계산하는 데 중점을 두고 있으며, 이는 고전적인 이미지 등록 방법에 비해 계산 비용을 낮추는 데 기여합니다. LEDA는 비선형 변형을 효과적으로 모델링하고 분석할 수 있게 하며, OASIS-1 데이터셋을 사용한 광범위한 실험에서 이의 유효성이 입증되었습니다. 뇌 이미지, 적응 방사선 치료 계획 등 다양한 임상 응용을 위한 개별 변형을 정확하게 포착하고 통합하는 능력도 평가하고 있습니다.

- **Performance Highlights**: LEDA는 복잡한 비선형 변형을 정확하게 모델링하면서도 역일관성을 유지하는 데 효과적입니다. 본 연구에서는 OASIS-1 데이터셋을 활용하여 LEDA의 성능을 확인했으며, 이 프레임워크가 기존 방법들에 비해 우수한 결과를 보여주었음을 시사합니다. 특히, LEDA는 전통적인 방법이 갖는 계산 비용과 수치 오류 문제를 해결함으로써 임상 연구와 실험에서의 적용 가능성을 극대화하고 있습니다.



### Formal Mathematical Reasoning: A New Frontier in AI (https://arxiv.org/abs/2412.16075)
- **What's New**: AI for Mathematics (AI4Math)은 과학, 공학 등에 있어 AI 중심의 발견에 매우 중요한 분야로, 공식적인 수학적 추론(Formal mathematical reasoning)이 이 분야를 한 단계 진전시키는데 필수적이라고 주장하고 있습니다. 최근 몇 년간 AI를 통해 정리 증명(Theorem proving)과 자동 정형화(Autoformalization) 같은 분야에서 일정한 진전을 이루었지만, 여전히 해결해야 할 중요한 도전 과제가 존재했습니다. 이 논문에서는 이러한 도전 과제들을 요약하고, 미래의 성공을 가늠할 수 있는 주요 이정표를 구상합니다.

- **Technical Details**: AI4Math 분야는 주로 자연어 처리(NLP)에서 차용한 기법을 사용하여 수학 LLMs를 개발하는 데 많은 연구가 집중되어 왔습니다. 특히, 잘 정제된(math) 데이터셋을 통해 LLM을 사전 훈련한 후, 세부적인 단계의 솔루션을 포함한 수학 문제의 데이터셋으로 모델을 미세 조정하는 방법이 사용됩니다. 또한, OpenAI o1의 사례처럼 비공식적인 접근방식을 평가 및 조정하는 과정에서 검색과 신경 검증기(neural verifiers)를 결합하여 환각된 추론의 문제를 줄일 수 있는 방향이 부각되고 있습니다.

- **Performance Highlights**: AlphaProof와 AlphaGeometry와 같은 시스템은 공식적인 수학적 추론을 통해 수학적 문제 해결에서 뛰어난 성과를 보였습니다. 이러한 시스템들은 신경망의 추론 단계를 실행하고 고품질 합성 데이터를 생성하여 전례 없는 수학적 추론 능력을 갖추게 되었습니다. 앞으로 이 분야는 자동화된 정형화와 강화 학습 등을 통해 더욱 발전할 가능성이 높습니다.



### A Framework for Streaming Event-Log Prediction in Business Processes (https://arxiv.org/abs/2412.16032)
Comments:
          18 pages

- **What's New**: 본 논문에서는 비즈니스 프로세스가 데이터 생성을 진행하는 동안 예측을 가능하게 하는 스트리밍 모드의 이벤트 로그 예측을 위한 Python 기반 프레임워크를 제안합니다. 이 프레임워크는 n-grams 및 LSTMs와 같은 스트리밍 알고리즘을 쉽게 통합하고, 여러 예측 모델을 앙상블(ensemble) 방식으로 결합할 수 있도록 합니다. 다양한 프로세스 마이닝 데이터 세트를 기반으로 한 실험을 통해, 배치(batch) 모드와 스트리밍 모드 간의 성능 차이를 비교하였습니다.

- **Technical Details**: 연구에서는 이벤트 로그 예측의 두 가지 패러다임인 배치 학습(batch learning)과 스트리밍 학습(streaming learning)에 대해 다룹니다. 특히, 스트리밍 모드에서는 데이터가 희소한 초기 단계에서도 빠르게 의미 있는 예측을 제공해야 합니다. 본 연구의 핵심은 각 사례의 이전 이벤트 데이터에만 의존하는 예측 함수를 생성하여 가장 가능성이 높은 다음 활동을 예측하는 것입니다.

- **Performance Highlights**: 실험 결과, LSTM 네트워크는 기본 모델인 prefix tree와 n-grams보다 전반적으로 뛰어난 성능을 보였으나, 단순 n-grams 모델도 LSTM 성능에 근접한 결과를 보였습니다. 앙상블 기법을 사용하여 기본 모델들을 조합할 경우, LSTM의 성능을 초과할 수 있는 가능성도 확인되었습니다. 실험은 헬스케어, 금융, IT 서비스 관리 등 다양한 도메인에서 7개의 실세계 프로세스 마이닝 데이터 세트를 기반으로 진행되었습니다.



### Learning sparsity-promoting regularizers for linear inverse problems (https://arxiv.org/abs/2412.16031)
- **What's New**: 이 논문은 선형 역문제를 해결하기 위한 희소성 촉진 정규화기(sparsity-promoting regularizers)를 학습하는 새로운 접근 방식을 소개합니다. 이 방법은 최적의 합성 연산자(synthesis operator) B를 선택하기 위한 이계 최적화(bilevel optimization) 구조를 개발하여, 문제를 정규화하면서 해의 희소성을 촉진시킵니다.

- **Technical Details**: 선형 역문제(linear inverse problem)를 대하는 데 있어, B가 힐베르트 공간(Hilbert space) X와 Y 사이의 경계 있는 연산자(bounded operator)일 때, 변별적 전략을 도입하였습니다. 이 과정을 통해 정규화 문제의 정의와 잘 정립된 정합성(well-posedness)을 보장하기 위한 가정을 소개합니다.

- **Performance Highlights**: 이 접근법은 데이터의 통계적 속성을 활용하고, 합성 연산자 B의 선택을 통해 사전 지식을 통합하였습니다. 예제 분석을 통해 제안된 방법의 유연성을 보여주며, 이전의 Tikhonov 정규화(Tikhonov regularization) 접근법을 발전시켜 무한 차원에서 희소 정규화를 위한 데이터 기반 접근법을 제안하고 있습니다.



### Mamba-based Deep Learning Approaches for Sleep Staging on a Wireless Multimodal Wearable System without Electroencephalography (https://arxiv.org/abs/2412.15947)
Comments:
          21 pages, 11 figures. Authors Andrew H. Zhang, Alex He-Mo, and Richard Fei Yin contributed equally

- **What's New**: 이 연구에서는 Mamba 기반 딥러닝 접근법을 활용하여 ANNE One 센서로부터 수집된 신호를 통해 수면 단계를 분류하는 방법을 탐구합니다. ANNE One은 전자 심전도(ECG), 삼축 가속도계, 온도와 같은 다양한 생리학적 데이터를 측정할 수 있는 최소 침습적인 듀얼 센서 웨어러블 시스템입니다. 이 연구는 기존의 수면 연구에서 사용된 EEG 신호 없이도, 성인 환자들의 수면 단계를 효과적으로 예측할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구는 360명의 성인으로부터 수집된 데이터로 이루어지며, 이들은 동시 임상 폴리솜노그래피(PSG)에 참여했습니다. PSG 기록은 AASM 기준에 따라 스코어링되었으며, ECG 채널을 사용하여 웨어러블 센서 데이터와 자동 정렬되었습니다. Mamba 기반 모델은 CNN, RNN, CRNN과 같은 다양한 아키텍처를 활용하여 훈련되었으며, 비슷한 아키텍처의 모델 변형을 앙상블하여 성능을 개선했습니다.

- **Performance Highlights**: 연구 결과, 3-class 시스템에서는 83.50%의 균형 정확도와 84.16%의 F1 스코어를 달성하여, Mamba 기반 딥러닝 모델이 강력한 성능을 발휘함을 입증했습니다. 4-class 및 5-class 시스템에서도 각각 74.64%, 64.30%의 균형 정확도를 기록하며, 이는 EEG 신호에 기반한 전통적인 방법들과 비교해도 견줄만한 성능을 보입니다. 이는 Mamba 모델이 EEG 없이도 효과적으로 수면 단계를 추론할 수 있음을 시사합니다.



### Data Preparation for Fairness-Performance Trade-Offs: A Practitioner-Friendly Alternative? (https://arxiv.org/abs/2412.15920)
Comments:
          Accepted as Registered Report at SANER'25

- **What's New**: 본 연구는 머신러닝(ML) 시스템에서 발생하는 공정성(fairness)과 편향(bias) 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 특히, 데이터 자체가 편향의 주요 원인 중 하나라는 최근 연구를 반영하여, 공정성을 고려한 데이터 준비(Data Preparation) 관행의 도입이 중요하다고 강조합니다. 이 연구는 최적화된 공정성 인식 관행이 머신러닝 초기 라이프사이클 단계에서 어떻게 공정성과 성능을 향상시킬 수 있는지를 평가할 것입니다.

- **Technical Details**: 연구의 핵심은 FATE(Fairness-Aware Techniques for Evaluation)라는 최적화 기법을 도입하여 데이터 준비 파이프라인을 선택하는 것입니다. FATE는 공정성과 성능의 균형을 고려하여 데이터 처리 방법을 분석합니다. 또한 이 방법은 사전 처리(pre-processing)를 통해 편향을 경감하는 기존 기술들과의 비교를 통해 데이터를 효율적으로 준비할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 초기 연구 결과, FATE를 통해 선택된 데이터 준비 파이프라인이 일반적인 사전 처리 기법보다 더 나은 공정성과 성능을 발휘할 가능성이 있음을 보여줍니다. 이는 ML 시스템의 공정성을 높이는 동시에, 모델의 성능을 저하시키지 않는 방법을 지속적으로 모색해야 함을 시사합니다. 공정성과 성능 모두를 고려하는 접근은 ML 산업 전반에서 중요한 발전으로 평가받고 있습니다.



### What Are Step-Level Reward Models Rewarding? Counterintuitive Findings from MCTS-Boosted Mathematical Reasoning (https://arxiv.org/abs/2412.15904)
Comments:
          AAAI 2025

- **What's New**: 최근 연구에서는 Step-level Reward Models (SRMs)가 수학적 추론 성능을 획기적으로 향상시키는 데 중요한 역할을 한다고 강조하고 있습니다. 특히, Monte Carlo Tree Search (MCTS) 방식이 SRMs의 성능을 극대화하는 데 효과적이라는 점을 부각시켰습니다. 이러한 연구는 SRMs의 작동 원리에 대한 이해를 심화시키고, 자연어 설명이 수학적 사고 과정에 필수적이지 않다는 가설을 제시합니다.

- **Technical Details**: 이 논문에서는 Step-level Reward Models (SRMs)와 Markov Decision Process (MDP)에 대한 기초 개념을 다루고 있습니다. MDP는 강화 학습 문제를 해결하는 데 필수적인 수학적 틀로, 상태, 행동, 보상 함수 및 미래 보상의 중요성을 결정하는 할인 요소를 포함합니다. SRMs는 프로세스 감독을 통해 수학적 추론을 개선하고, 수학적 언어에서의 복잡한 논리적 일관성을 평가할 수 있도록 훈련됩니다.

- **Performance Highlights**: SRMs는 수학적 언어에서의 논리적 일관성을 효과적으로 평가하며, 자연어에 대한 평가에는 어려움을 겪는 것으로 나타났습니다. 이는 SRMs가 수학적 언어에 대한 본질적인 친화성을 갖고 있다는 것을 시사합니다. 이 연구는 SRMs의 효율적인 훈련 방법을 찾고, 수학적 추론의 핵심 요소에 집중함으로써 더 나은 성능을 가능하게 할 것으로 기대하고 있습니다.



### IMPLY-based Approximate Full Adders for Efficient Arithmetic Operations in Image Processing and Machine Learning (https://arxiv.org/abs/2412.15888)
- **What's New**: 하드웨어 성능을 향상시키기 위한 새로운 접근법인 SAPPI(Serial APProximate IMPLY 기반 풀 애더)를 제안합니다. 이 알고리즘은 근사 계산(Approximate Computing)과 메모리 내 계산(In-Memory Computing) 기술을 결합하여 전력 소모를 대폭 줄이고 계산 단계를 최소화하는 데 중점을 두고 있습니다. 실험 결과, 기존 정확도를 유지하면서도 연산 단계 수를 39%-41%, 에너지 소비를 39%-42%까지 감소시키는 성과를 보였습니다.

- **Technical Details**: MEMRISTOR 기반 IMPLY 논리를 활용한 두 개의 근사 애더(SAPPI-1 및 SAPPI-2)는 낮은 하드웨어 복잡성과 작은 면적을 요구합니다. 이 설계는 전기 저항을 통해 논리 값을 비휘발성으로 저장할 수 있는 MEMRISTOR의 특성을 이용합니다. 근사 계산의 이점을 살리기 위해 이미지 처리와 머신 러닝 등에서 필요한 정확도와 처리 속도를 최적화하는 방향으로 접근했습니다.

- **Performance Highlights**: 제안한 근사 애더는 리플 캐리 애더(Ripple Carry Adder) 내에서 최대 10%의 속도 향상과 최대 13%의 에너지 효율성을 달성했습니다. 또한 MNIST 데이터셋을 학습한 합성곱 신경망(Convolutional Neural Networks)에서 최대 296 mJ의 에너지 절약과 1.3억 단계의 계산 단계를 줄였습니다. 이러한 결과는 이미지 품질을 수용할 수 있는 수준의 근사화를 통해 얻어졌습니다.



### The common ground of DAE approaches. An overview of diverse DAE frameworks emphasizing their commonalities (https://arxiv.org/abs/2412.15866)
- **What's New**: 이번 연구에서는 다양한 행렬 함수의 랭크 조건(rank conditions)을 분석하여 차별대수 방정식(differential-algebraic equations, DAE)에 대한 접근 방식을 다룹니다. 특히, 특정 행렬 함수에서의 랭크 감소가 결정적인 해(solution behavior)를 나타낼 수 있음을 강조합니다. Kronecker index와 관련된 여러 문헌의 일반화 개념을 고려하면서 공통된 기반(common ground)을 모색합니다.

- **Technical Details**: 연구는 투명한 축소 프레임워크를 시작으로, 모든 프레임워크에 적용 가능한 전형적인 특성값(canonical characteristic values)으로 구성된 포괄적인 규칙 개념(regularity concept)을 상세히 발전시킵니다. 또한, 열세 가지의 서로 다른 규칙 정의의 동등성을 증명하였고, 이를 통해 모든 개념의 결과를 함께 사용할 수 있는 가능성을 제시합니다. 이러한 접근 방식은 DAE의 특성을 설명하는 데 있어 index와 전형적인 특성값이 얼마나 중요한지를 보여줍니다.

- **Performance Highlights**: 이번 연구에서 제시된 규칙 개념은 다양한 DAE 문제의 해석 및 해법 제시에 유용하며, 행렬 함수의 특성에 따라 어떻게 해가 변화하는지를 밝힙니다. 또한, 여러 규칙 정의 간의 관계를 명확히 하여 이론적 기초가 다양한 응용에 기여할 수 있음을 밝힙니다. 결국 이 연구는 DAE의 복잡성을 이해하는 데 중요한 역할을 할 것입니다.



### On Robust Cross Domain Alignmen (https://arxiv.org/abs/2412.15861)
- **What's New**: 이번 연구에서는 Gromov-Wasserstein (GW) 거리의 강건성 문제를 다루고 있으며, 기존의 최적 수송(optimal transport, OT) 기법에서 영감을 받아 새로운 세 가지 강건화 기법을 제안합니다. 이 기법들은 각기 다른 응용 분야와 오염 환경에 최적화된 솔루션을 제공하여 다양한 실제 머신러닝 작업에서 GW 거리를 계산할 때 더 우수한 성능을 발휘합니다. 특히, 데이터의 질과 무관하게 각 공간의 기하학적 정합을 유지할 수 있는 방법론에 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 첫 번째 기법은 Tukey와 Huber의 패널리제이션(penalization) 개념을 도입하여 GW 거리의 계산에서 큰 왜곡을 방지합니다. 두 번째 기법은 주어진 공간 간의 극단적인 쌍거리(pairwise distances)를 보다 세밀하게 조절하여, 이를 통해 보강된 지표(metric)를 정의하며, 세 번째 방법은 '청정한' 프로시(즉, 오염이 없는) 분포를 기반으로 최적화를 정규화하여 강건한 측정 보존 맵을 달성합니다. 이 세 가지 방법은 GW 거리와 그 변형들의 기하학적, 통계적 속성을 보강하도록 설계되었습니다.

- **Performance Highlights**: 실험적으로, 제안된 강건화 기법들은 기존의 최첨단 방법들과 비교할 때 오염에 대한 내성을 크게 향상시킨다는 것을 보여주었습니다. 특히, 이미지 변환(imaging translation) 작업에 있어서, 오염된 데이터 환경에서도 효율적인 성능을 발휘하며 탁월한 노이즈 제거 능력을 입증했습니다. 이러한 연구 결과는 기계 학습 분야에서 비슷한 종류의 정합 문제를 해결하기 위한 기초 자료로 활용될 수 있을 것입니다.



### Using matrix-product states for time-series machine learning (https://arxiv.org/abs/2412.15826)
Comments:
          27 pages, 13 figures

- **What's New**: 이번 논문에서는 양자 다체 물리학을 모델링하기 위해 사용되는 매트릭스-프로덕트 상태(MPS) 기반의 새로운 알고리즘인 MPSTime을 개발하였습니다. 이 알고리즘은 시계열 데이터의 공동 확률 분포를 학습할 수 있으며, 이를 통해 분류(classification) 및 결측치 보정(imputation)과 같은 중요한 시계열 기계 학습 문제를 해결할 수 있습니다. 또한, MPSTime은 적당한 MPS 결합 차원($\chi_{\rm max}$)을 사용하여 복잡한 확률 분포를 효율적으로 학습할 수 있습니다.

- **Technical Details**: 기존의 MPS 기반 방법론을 바탕으로, MPSTime은 시계열 데이터를 적절하게 수치화(featue encoding)하고 이를 사용하여 MPS를 통해 공동 확률 분포를 학습하는 기법을 제공합니다. 이 과정에서 분류 및 생성적 모델 학습에 동일한 손실 함수(loss function)를 사용하여 하나의 통합된 프레임워크를 만들었습니다. 시계열 데이터는 연속적인 실수 값을 가지므로, 이를 MPS와 연결하기 위해서는 신중하게 선택된 수치화 과정이 필요합니다.

- **Performance Highlights**: MPSTime은 의학, 에너지 및 천문학 분야의 합성 및 실제 데이터셋을 사용하여 최신 기계 학습 접근 방식과 경쟁하는 성능을 보여주었습니다. 특히, MPS는 학습한 확률 분포의 조건부 얽힘 엔트로피를 계산하여 데이터 기반에서 복잡한 상관 관계를 해석하는 데 유리한 장점을 제공합니다. 최종적으로, MPSTime은 과학, 산업 및 의학 분야의 시계열 기계 학습 문제를 해결하는 데 있어 해석 가능성이 높은 진전을 이룰 수 있는 가능성을 보여줍니다.



### Deep learning joint extremes of metocean variables using the SPAR mod (https://arxiv.org/abs/2412.15808)
- **What's New**: 이 논문은 새로운 딥 러닝 프레임워크를 소개하며, 메토시안 변수(metocean variables)의 다변량 특이값(joint extremes)을 추정하는 Semi-Parametric Angular-Radial (SPAR) 모델을 기반으로 하고 있습니다. 이 방법은 다변량 극단값을 모델링하는 문제를 각도 밀도(angular density)와 각도에 조건화된 단일 반경 변수의 꼬리(tail) 모델로 변환합니다. 적용 사례로는 바람 속도, 바람 방향, 파고, 주기 및 파 방향 등의 다섯 가지 메토시안 변수를 사용하였습니다.

- **Technical Details**: SPAR 접근법에서는 일반화 파레토 분포(Generalized Pareto distribution)를 사용하여 반경 변수의 꼬리를 모델링합니다. 이 데이터 기반 접근 방식은 표현할 수 있는 의존성 구조에 대해 높은 유연성을 제공하며, 모델 학습을 위한 계산 효율적인 루틴을 포함합니다. 또한, 이 방법은 기존 접근 방식에 비해 기본 분포에 대한 가정이 적고, 관측 범위 외부로의 외삽(extrapolation)을 위한 정당한 수단을 제공합니다.

- **Performance Highlights**: 여러 진단 플롯을 사용하여 적합된 모델들이 고려된 메토시안 변수의 공동 극단값(joint extremes)을 잘 설명한다는 것을 보여줍니다. 이 연구는 현재의 환경 데이터 세트에 대한 강력한 가정이 필요하지 않는 장점을 가지며, 다양한 의존성 구조를 모델링할 수 있는 유연한 방법을 제안합니다.



### GraphSeqLM: A Unified Graph Language Framework for Omic Graph Learning (https://arxiv.org/abs/2412.15790)
- **What's New**: 이번 연구는 Graph Neural Networks (GNNs)와 Large Language Models (LLMs)를 통합하여 새로운 Graph Sequence Language Model (GraphSeqLM)을 제안합니다. GraphSeqLM은 DNA, RNA 및 단백질 서열을 인코딩하는 생물학적 서열 임베딩을 통해 GNN의 성능을 향상시키며, 복잡한 생물학적 관계를 잘 포착할 수 있도록 설계되었습니다. 이 방법은 의학에 있어 정밀한 다중 오믹스 데이터 통합을 가능하게 합니다.

- **Technical Details**: 제안된 방법론은 DNA, RNA 및 단백질 서열 데이터를 다중 오믹스 데이터 세트와 통합하여 복잡한 생물학적 네트워크의 표현을 강화합니다. 기존의 유전자 조절 네트워크 및 KEGG(킨네시사와 고토, 2000)와 결합하여, 각각의 기능 세트를 포함하는 지식 그래프를 생성합니다. 이 지식 그래프는 두 개의 하위 그래프로 분해되어 내부 신호 전달 과정과 단백질-단백질 상호작용(PPI)을 효과적으로 모델링합니다.

- **Performance Highlights**: GraphSeqLM은 종합 평가에서 기존 방법들을 능가하는 예측 정확도를 보여주었습니다. 고유의 생물학적 속성 및 구조를 인코딩하여 다중 오믹스 데이터의 복잡한 패턴을 파악하는 데 있어서 뛰어난 성능을 발휘합니다. 이 연구는 정밀 의학에서 다중 오믹스 데이터 통합을 위한 더 나은 접근 방식을 제시하며, 생물학적 시스템 모델링의 새로운 가능성을 열어줍니다.



### Probabilistic Latent Variable Modeling for Dynamic Friction Identification and Estimation (https://arxiv.org/abs/2412.15756)
- **What's New**: 본 논문은 로봇 조인트의 마찰 모델을 식별하기 위해 잠재적인 동적 상태(latent dynamic states)를 사용하는 새로운 접근 방법을 제안합니다. 기존의 물리 기반 모델에서 데이터 기반 모델로의 전환을 보이며, 동적 마찰 모델을 보다 효과적으로 정립하고 일반화하는 방법을 모색하고 있습니다. 또한, 이 방법은 기존 모델링 방식의 한계를 극복하고, 노이즈가 있는 인코더 측정값으로부터 동적 모델을 학습하는 방식을 취합니다.

- **Technical Details**: 제안된 방법론은 마찰 토크를 평가하기 위해 동적 로봇 상태와 추가 정보가 포함된 잠재 상태를 사용하여 마찰 모델을 수립합니다. 이 stochastic하고 부분적으로 비지도형(unsupervised) 식별 문제는 표준 확률적 표현 학습 문제로 간주되며, 마찰 모델과 잠재 상태 동적은 신경망(neural networks)으로 매개변수화됩니다. 기대-최대화(EM) 알고리즘을 사용하여 모델 매개변수의 최대 우도 추정치(MLE)를 찾습니다.

- **Performance Highlights**: 제안된 방법의 효과는 Kuka KR6 R700을 실험 플랫폼으로 하여 기존의 기준 방법들과 비교했을 때, 개방 루프(prediction accuracy) 예측 정확도 측면에서 검증되었습니다. 이 연구는 데이터 중심의 접근 방식이 동적 마찰 모델에서 유용성을 보여줄 수 있는 가능성을 암시합니다.



### Critique of Impure Reason: Unveiling the reasoning behaviour of medical Large Language Models (https://arxiv.org/abs/2412.15748)
Comments:
          16 pages, 5 figures, 2 tables. Conceptualization, both authors. formal analysis, both authors. funding acquisition, both authors. investigation, both authors. resources, both authors. supervision, T.C.. validation, both authors. visualization, both authors. writing original draft, both authors. writing review and editing, both authors

- **What's New**: 본 논문에서는 최근 의료 분야에서 사용되는 Large Language Models (LLMs)의 추론 행동(reasoning behaviour)을 탐구하고, 이러한 모델의 높은 수준의 예측 정확도보다도 추론 행동에 대한 이해의 필요성을 강조합니다. 의료 AI의 Explainable AI (XAI)를 달성하기 위해 이러한 모델이 어떻게 결론에 도달하는지에 대한 통찰력을 제공하는 이론적 틀을 제안합니다. 이를 통해 의료 전문가들이 LLM의 내부 작동 방식을 이해하고 잠재적인 논리적 오류를 드러내는 데 도움을 줄 수 있습니다.

- **Technical Details**: 논문은 LLM의 추론 행동을 정의하고, 현재의 의료 LLM에서의 고급 성능 지표와 함께 추론 행동을 평가하는 방법들을 분류합니다. 특히, 논리적 추론과 인과적 추론을 포함한 다양한 유형의 추론이 있습니다. 또한, Neuro-Symbolic AI (N-SAI)라는 분야를 통해 신경망과 기호적 추론 기술을 통합하는 방법을 논의하며, 이러한 접근 방식이 추론을 더욱 투명하게 만들어주는 데 기여할 수 있음을 설명합니다.

- **Performance Highlights**: 의료 LLM의 추론 행동을 이해함으로써 사용자는 이러한 모델이 어떻게 결론에 도달했는지를 확인할 수 있으며, 이는 임상 의사 결정 과정에서 신뢰를 구축하는 데 기여합니다. LLM이 환자 진단 및 치료 제안에서의 통찰력을 제공할 경우, 이는 의사와 기계의 권고사항 간의 불일치를 명확히 할 수 있습니다. 궁극적으로 이러한 투명성은 의료 분야에 AI를 통합하고 환자 결과를 개선하는 데 중요한 역할을 할 것입니다.



### The Role of Recurrency in Image Segmentation for Noisy and Limited Sample Settings (https://arxiv.org/abs/2412.15734)
Comments:
          24 pages

- **What's New**: 이 논문은 머신러닝의 진보에 기여한 생물학적 뇌의 구조에서 영감을 받아, 현재 최첨단 컴퓨터 비전 모델들이 인간의 뇌와 유사하게 작동하지 않는 이유를 분석하고 있습니다. 특히, 이 연구는 기존의 feed-forward segmentation 모델에 recurrent (순환) 메커니즘을 추가하는 것이 어떤 영향을 미칠지를 탐구하고 있습니다. 이는 뇌의 재귀적 특성이 현재의 모델에 어떤 긍정적인 변화를 가져올지 질문을 던지는 작업입니다.

- **Technical Details**: 연구에서는 self-organizing (자기 조직화), relational (관계성), memory retrieval (기억 검색)과 같은 여러 종류의 재귀성을 탐색하여 특정 에너지 함수를 최소화하는 방식을 적용했습니다. 실험은 인공 및 의료 이미징 데이터에 대해 수행하였으며, 특히 높은 수준의 노이즈와 few-shot learning (극소 데이터 학습) 설정에서의 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과, 재귀적 모델이 기존의 state-of-the-art feed-forward 모델보다 더 나은 성능을 보여주지 못했으며, 이는 기존의 재귀적 구조만으로는 성능 향상에 충분치 않다는 것을 시사합니다. 추가적인 연구가 필요하다는 결론에 도달했으며, 이는 향후 머신러닝 모델에 있어 재귀적 메커니즘의 적용에 대한 더 깊은 탐구를 불러일으킬 것입니다.



### MacLight: Multi-scene Aggregation Convolutional Learning for Traffic Signal Contro (https://arxiv.org/abs/2412.15703)
Comments:
          Accepted as full paper by AAMAS2025

- **What's New**: 이 논문에서는 기존의 교통 신호 제어(Traffic Signal Control, TSC) 방법론의 한계를 극복하기 위해 Multi-Scene Aggregation Convolutional Learning (MacLight)을 제안합니다. MacLight는 기존의 그래프 기반 접근방식보다 빠른 훈련 속도와 안정적인 성능을 제공하며, 프로시멀 정책 최적화(Proximal Policy Optimization, PPO) 알고리즘을 백본으로 사용하여 국소적 특징과 글로벌 임베딩 표현을 고려합니다. 이를 통해 과적합(overfitting) 문제를 줄이고 정책 업데이트의 안정성을 보장합니다.

- **Technical Details**: MacLight는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째 구성 요소는 전역 표현(global representation)으로, 변분 오토인코더(Variational Autoencoder, VAE)를 활용하여 글로벌 상태의 압축 표현을 생성합니다. 두 번째 구성 요소는 PPO 알고리즘으로, 가치 평가를 위한 모듈과 정책 개선을 위한 모듈을 통해 서로 다른 정보를 처리합니다. 이러한 설계를 통해 시간 오버헤드를 최소화하고 전체적인 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과는 MacLight가 기존의 일반 및 도메인 SOTA(상태-of-the-art) 방법과 비교하여 뛰어난 안정성과 최적화된 수렴 수준을 달성했음을 보여줍니다. 긴급한 교통 사건에 따른 동적 교통 흐름을 시뮬레이션할 수 있는 환경에서도 높은 시간 효율성을 유지합니다. 코드와 구현은 제공된 링크에서 확인 가능하며, 다양한 교통 시나리오에서 검증된 성능을 기반으로 향후 연구를 위한 기초를 마련했다고 할 수 있습니다.



### AIR: Unifying Individual and Cooperative Exploration in Collective Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2412.15700)
- **What's New**: 본 논문에서는 협력적인 멀티 에이전트 강화 학습(MARL)에서의 탐색 문제를 해결하기 위해 Adaptive exploration via Identity Recognition(AIR) 방법을 제안합니다. AIR는 에이전트의 정체성을 인식하여 탐색 모드와 강도를 조절하는 분류기와 행동 선택기로 구성된 두 개의 적대적 구성 요소로 이루어져 있습니다. 이 방법은 개인 및 집단 탐색을 원활하게 할 수 있도록 이론적으로 입증되었습니다.

- **Technical Details**: AIR는 에이전트의 궤적을 기반으로 한 정체성 분류기와 탐색 모드를 동적으로 조정하는 행동 선택기를 통해 작동됩니다. 기존의 탐색 전략의 한계를 분석하고, 개인 행동 및 에이전트 간의 공동 행동을 모두 고려하는 프레임워크를 제안합니다. 이 방식은 과도한 추가 모듈 없이도 효율적인 계산 자원 활용을 가능하게 하여 학습 과정을 단순화합니다.

- **Performance Highlights**: 실험 결과, AIR는 다양한 멀티 에이전트 작업에서 기존 방법보다 효율적이고 효과적으로 작동함을 보여줍니다. AIR는 조화롭게 개인 및 집단 탐색을 통합하여 에이전트 간의 협력을 증진시킵니다. 이 연구는 개인 탐색과 집단 탐색을 통합한 최초의 시도로서, MARL 분야에서의 새로운 가능성을 제시합니다.



### GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations (https://arxiv.org/abs/2412.15687)
Comments:
          23 pages, 15 figures

- **What's New**: GraphDOP는 Earth System 관측 자료만을 사용하여 훈련된 새로운 데이터 기반의 예측 시스템입니다. 이 시스템은 전통적인 물리 기반 (re)analysis 입력 없이도 지구 시스템의 상태 동역학을 포착하고 예측할 수 있는 능력을 갖추고 있습니다. GraphDOP는 위성의 밝기 온도와 같은 관측된 양과 물리적 양의 상관 관계를 통해 학습하며, 최대 5일간의 기상 예측을 할 수 있습니다.

- **Technical Details**: GraphDOP는 데이터 기반의 무게를 더하는 그래프 신경망 (GNN) 모델을 사용하여 대기 상태의 잠재적 표현을 학습합니다. 이 모델은 GED data set에서 지리적으로 분포된 관측 데이터를 활용하여 학습하며, 2미터 기온 예측에서 운영 IFS 시스템과 경쟁할 수 있는 성능을 보입니다. 이 연구는 수십억 건의 관측을 처리하는 복잡한 데이터 동화 시스템의 대안으로서 기계 학습을 적용할 수 있는 가능성을 제시하고 있습니다.

- **Performance Highlights**: GraphDOP는 특정 기상 시나리오인 북극의 급속한 냉각 사건과 허리케인 이안에서 유망한 예측 결과를 보여주었습니다. 특히 GraphDOP는 열대 지역에서 IFS 모델보다 더 작은 예측 차이를 기록하며, 공공 데이터로부터 학습하여 기상 예측의 질을 향상시키고 있습니다. ECMWF는 또한 한편으로는 엔드 투 엔드 변환기 네트워크를 탐색하고 있으며, 이에 대한 연구 결과가 곧 발표될 예정입니다.



### A survey on FPGA-based accelerator for ML models (https://arxiv.org/abs/2412.15666)
Comments:
          16 pages, 4 figures (Working paper)

- **What's New**: 이 논문은 하드웨어 가속기에서 머신러닝(ML) 알고리즘의 가속화에 관한 포괄적인 조사 결과를 제시하고 있습니다. 특히, 최근 6년간의 FPGA와 관련된 1138편의 논문 중 287편을 검토하여 ML과 FPGA 기술의 통합이 증가하고 있다는 점을 강조합니다. 연구 결과는 추론 가속화(inference acceleration)에 81%가 집중되어 있으며, CNN이 FPGA 가속화 연구에서 가장 두드러진 모델로 자리잡고 있는 것으로 나타났습니다.

- **Technical Details**: ML 알고리즘은 데이터로부터 학습하여 새로운 데이터에 대해 작업을 수행하고 결과를 예측합니다. CPU와 GPU 모두 장점이 있지만, FPGA는 재구성이 가능한 구조 덕분에 높은 에너지 효율성을 제공하며 다양한 ML 응용 프로그램에 적합합니다. 이 논문은 FPGA 기반 가속기의 구현과 관련하여 그 장점과 도전 과제를 살펴보며, FPGA의 유연성과 높은 계산 자원이 ML 알고리즘을 구현하는 데 있어 매우 유망한 플랫폼임을 명확히 합니다.

- **Performance Highlights**: 이번 조사는 FPGA 기반 ML 가속화 연구의 네 가지 주요 측면을 분석합니다. 연구에 따르면, 81%의 논문이 모델 추론에 집중하고 있으며, 이는 FPGA의 커스터마이징 기능과 병렬 처리 능력이 이 분야에서 독특한 장점임을 시사합니다. 또한, ML 관련 FPGA 연구는 2022년부터 5% 상승하며 ML 기술이 FPGA 분야에 통합되고 있음을 나타내고, 이는 향후 복잡하고 지능적인 FPGA 응용 프로그램의 발전을 예고합니다.



### Tacit Learning with Adaptive Information Selection for Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2412.15639)
Comments:
          Accepted by AAMAS 2025 (Extended Abstract)

- **What's New**: 본 논문은 Multi-Agent Reinforcement Learning (MARL)에서 Centralized Training with Decentralized Execution (CTDE) 프레임워크의 발전을 위한 새로운 협력적 MARL 프레임워크인 Selective Implicit Collaboration Algorithm (SICA)를 소개합니다. 기존 CTDE 방법의 두 가지 주요 문제를 해결하기 위해, 에이전트는 로컬 정보를 기반으로 의사결정을 수행하고, 진정한 정보를 공유하기 위해 상호 작용을 통해 암묵적인 협력을 점진적으로 발전시킵니다. 이를 통해 에이전트는 복잡한 협력 전략을 학습할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: SICA는 QMIX 프레임워크를 기반으로 하여 구성된 세 가지 주요 블록: Selection Block, Communication Block, Regeneration Block을 가지고 있습니다. Selection Block에서는 협력에 필요한 정보를 필터링하고, Communication Block을 통해 다른 에이전트들에게 공유하여 진정한 정보를 생성합니다. Regeneration Block은 로컬 정보를 활용하여 진정한 정보를 재생성하며, 훈련을 진행하면서 중앙 집중형에서 분산형 프레임워크로 점진적으로 전환합니다.

- **Performance Highlights**: SICA는 StarCraft Multi-Agent Challenge(SMAC) 및 Google Research Football에서의 성능 평가를 통해 기존의 CTDE 방법과 명시적 커뮤니케이션 방법보다 우수한 결과를 기록했습니다. SICA의 접근 방식은 정보의 적응적 선택을 가능하게 하여 의사결정에서 정보를 더욱 효과적으로 활용하게 하며, 지연 문제를 피할 수 있게 해줍니다. 이러한 성과는 SICA가 다양한 알고리즘과 쉽게 통합되고, 실제 환경에서도 유용하게 적용될 수 있음을 보여줍니다.



### Microservices-Based Framework for Predictive Analytics and Real-time Performance Enhancement in Travel Reservation Systems (https://arxiv.org/abs/2412.15616)
Comments:
          10 Pages, 05 figures

- **What's New**: 이 논문은 예측 분석(predictive analytics)의 힘을 활용하여 실시간 여행 예약 시스템의 성능을 향상시키기 위한 마이크로서비스 기반 아키텍처(framework)를 제시합니다. 기존의 모놀리식(monoithic) 시스템은 부하가 많은 상황에서 확장성과 성능이 떨어지며, 이로 인해 자원의 미활용과 지연이 발생합니다. 이를 해결하기 위해 시스템 구성 요소를 독립적인 서비스로 분리하여 수요에 따라 성장하거나 축소할 수 있는 모듈화(modularization) 접근 방식을 채택하였습니다.

- **Technical Details**: 본 프레임워크는 머신러닝(machine learning) 모델을 통해 고객 수요 예측(forecasting), 동적 가격(dynamic pricing), 시스템 성능 최적화(real-time predictive analytics)를 포함합니다. 실험 평가(experimental evaluation)를 통해 이 프레임워크가 응답 시간(response time), 처리량(throughput), 성공 거래율(transaction rate of success), 예측 정확도(prediction accuracy)와 같은 성능 지표에 미치는 영향을 보여주었습니다. 마이크로서비스 접근 방식은 일반적인 아키텍처처럼 확장성(scalability)과 내결함성(fault tolerance)을 향상시키는 것 외에도, 적시성(timeliness)과 정확성(accuracy)을 갖춘 예측을 제공하여 고객 만족과 운영 효율성을 증가시킵니다.

- **Performance Highlights**: 실시간 분석(real-time analytics)의 통합은 더 지능적인 의사결정(decision-making)을 이끌어내어 시스템 응답(response) 및 신뢰성(reliability)을 개선합니다. 이 시스템은 현대의 여행 예약 시스템이 직면한 과제를 해결하기 위해 스케일 가능한(scalable) 효율적인(framework) 아키텍처를 제공합니다. 향후 연구는 성능과 견고성(robustness)을 더욱 향상시키기 위해 고급 AI 모델과 엣지 처리(edge processing)의 조사를 진행할 계획입니다.



### Music Genre Classification: Ensemble Learning with Subcomponents-level Attention (https://arxiv.org/abs/2412.15602)
- **What's New**: 본 연구에서는 Music Genre Classification에 있어 새로운 접근 방식을 제시합니다. 이 접근 방식은 ensemble learning과 sub-components에 대한 attention을 결합하여 음악 장르를 식별하는 정확성을 높이는데 중점을 두고 있습니다. 주목할 점은 음악 작품의 서브 컴포넌트를 별도로 분류함으로써 모형이 각각의 독특한 특성을 포착할 수 있다는 것입니다.

- **Technical Details**: 제안된 방법론은 각 서브 컴포넌트를 독립적으로 분류한 후, 이들에 대한 ensemble learning 기법을 적용합니다. 이러한 방식은 전체 음악 장르에 대한 분류 결정을 내리는 데 활용됩니다. 또한, GTZAN 데이터셋에서 훈련 및 테스트된 기존의 최첨단 기법들에 비해 더 뛰어난 정확도를 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 여러 기법들과 비교하여 우수한 성능을 나타내며, 그 효과는 GTZAN 데이터셋에서 명확하게 입증되었습니다. 이 연구는 Music Information Retrieval (MIR) 및 디지털 신호 처리 분야에 있어서 의미 있는 기여를 할 것으로 기대됩니다.



### Dexterous Manipulation Based on Prior Dexterous Grasp Pose Knowledg (https://arxiv.org/abs/2412.15587)
- **What's New**: 이번 연구에서는 새로운 reinforcement learning 접근 방식을 소개하여 손가락의 고급 조작(dexterous manipulation)의 효율성과 정확성을 개선하였습니다. 기존의 연구들은 고정된 dexterous grasp pose를 사용한 반면, 우리는 조작 과정을 두 개의 별도 단계로 나누어 접근했습니다.

- **Technical Details**: 첫 번째 단계에서는 물체의 기능적 부분을 목표로 한 dexterous grasp pose를 생성한 후, 두 번째 단계에서 reinforcement learning을 적용해 환경을 종합적으로 탐색합니다. 이러한 과정의 분리는 효율성과 정확성을 동시에 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 네 가지 서로 다른 작업(task)에서 학습 효율성과 성공률이 크게 향상된 것을 보여줍니다. 대다수의 학습 시간이 적절한 초기 위치 제공과 최적 조작 관점 선택에 소모된다는 점도 강조되었습니다.



### Score-based Generative Diffusion Models for Social Recommendations (https://arxiv.org/abs/2412.15579)
Comments:
          14 pages, 8 figures

- **What's New**: 이 연구에서는 소셜 추천 시스템의 성능 향상을 위해 혁신적인 생성적 관점을 통해 낮은 사회적 동질성 문제를 해결합니다. 새로운 Score-based Generative Model for Social Recommendation (SGSR)을 제안하며, 이는 Stochastic Differential Equation(SDE) 기반의 확산 모델을 소셜 추천에 적절하게 조정합니다. 이 연구의 주요 초점은 협업 신호와의 일관성을 극대화하는 사용자 소셜 표현을 생성하는 것입니다.

- **Technical Details**: SGSR은 사용자-아이템 행동에 기반하여 개인화된 노이즈 제거 확산 프로세스를 달성하는 새로운 클래스의 비지도 학습 기법을 활용합니다. 이를 위해, 우리는 스코어 기반 생성 모델(SGMs)을 도입하고, 이 모델의 특성과 사회적 추천에 적합한 형식으로 조정된 분류기 없는 조건부 목표를 도출합니다. 이는 사용자의 다양한 사회적 관계를 효과적으로 모델링하기 위해 서로 추가적인 레벨에서 협업 신호를 활용하여 최적의 소셜 표현을 식별할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, 제안된 SGSR 프레임워크는 세 가지 실세계 데이터 세트에서 수행된 포괄적인 비교 실험 및 제거 연구에서 기존의 최첨단 방법들보다 유의미하게 우수한 성능을 보였습니다. 이는 제안된 방법이 중복된 소셜 정보를 필터링하고 추천 성능을 효과적으로 향상시켰음을 나타냅니다. 특히, 사회적 신호의 노이즈 감소 기능을 통해 사용자 표현의 일관성을 높이는 데 성공하였습니다.



### SaliencyI2PLoc: saliency-guided image-point cloud localization using contrastive learning (https://arxiv.org/abs/2412.15577)
Comments:
          Under Review

- **What's New**: 이 논문에서는 SaliencyI2PLoc라는 새로운 대조 학습 기반 아키텍처를 제안하고 있습니다. 이 방법은 이미지와 포인트 클라우드 간의 특징 일관성을 유지하면서 주목 맵(saliency map)을 특징 집합에 융합합니다. 이는 다중 매니폴드 공간에서의 특징 관계 일관성을 유지하며, 크로스 모달리티(feature mapping) 문제를 효율적으로 해결합니다. 이 연구는 다양한 데이터셋에서 실험을 통해 뛰어난 효과성을 입증했습니다.

- **Technical Details**: 제안된 SaliencyI2PLoc 구조는 Dual-Transformer 아키텍처에 기반하고 있으며, 대조 학습 프레임워크를 사용하여 이미지와 포인트 클라우드 간의 융합을 수행합니다. 이 모델은 2D 이미지와 3D 포인트 클라우드를 직접 처리하며, 자신 주의(self-attention) 기반 특징 추출기를 사용해서 각 모달리티에서 국소 패치 특징을 추출합니다. 또한, 컨텍스트 주의를 통해 강력한 전역 특징을 생성하는 것이 특징입니다.

- **Performance Highlights**: 우리는 도시 시나리오 평가 데이터셋에서 Recall@1이 78.92%, Recall@20이 97.59%에 도달하는 결과를 얻었습니다. 이는 기존 방법보다 각각 37.35% 및 18.07%의 향상을 보여주고 있습니다. 이 결과는 SaliencyI2PLoc 아키텍처가 이미지와 포인트 클라우드를 효과적으로 융합하고, 크로스 모달리티(global localization)에서 중요한 진전을 이룬 것을 시사합니다.



### Multi Agent Reinforcement Learning for Sequential Satellite Assignment Problems (https://arxiv.org/abs/2412.15573)
- **What's New**: 이 연구는 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL)을 활용하여 복잡한 배정 문제를 해결하는 새로운 알고리즘을 제안합니다. 기존의 폴리노미얼-타임 그리디 솔버에서 부트스트래핑(bootstrapping)하여 배정의 가치를 학습한 후, 경험을 통해 더 나아가 최적 분산 배정 메커니즘을 사용하여 배정을 수행합니다. 이 접근 방식은 에이전트들이 직접적으로 작업에 할당되는 것이 아니라, 예상되는 할당의 값을 학습하여 사용하게 됩니다.

- **Technical Details**: 할당 문제는 n명의 에이전트와 m개의 작업으로 구성되어 있으며, 각 에이전트가 특정 작업을 완료할 때의 유틸리티를 나타내는 이익 매트릭스(benefit matrix) β가 주어집니다. 할당 행렬 x는 각 에이전트가 단일 작업에 할당되는 것과, 각 작업이 최대 하나의 에이전트에게 할당되는 조건을 만족해야 합니다. 제시된 알고리즘은 이익 매트릭스 β와 유효한 할당들의 집합 X로부터 배정 문제를 효율적으로 해결할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 알고리즘은 기존 문헌의 다른 방법에 비해 20-50%의 성능 향상을 보였습니다. 특히 다수의 에이전트와 작업을 포함한 복잡한 시나리오에서 성능이 뛰어났으며, 다양한 기존 MARL 기법(COMA, IQL, IPPO) 및 고전적인 최적화 방법들과 비교하여 우수한 결과를 나타냈습니다. 이러한 성과는 위성 인터넷 할당 문제와 같은 현실적인 상황에서 시스템의 효율성, 자율성 및 회복력을 크게 향상시키는 데 기여합니다.



### In-context Continual Learning Assisted by an External Continual Learner (https://arxiv.org/abs/2412.15563)
- **What's New**: 이 논문에서는 InCA라는 새로운 방법을 소개합니다. InCA는 External Continual Learner (ECL)와 In-context Learning (ICL)을 통합하여 Catastrophic Forgetting (CF) 문제를 피하면서도 Scalable Continual Learning (CL)을 가능하게 합니다. ECL은 각 테스트 인스턴스에 대해 적합한 클래스의 하위 집합을 미리 선택하여 ICL의 프롬프트 길이를 관리할 수 있도록 돕습니다.

- **Technical Details**: InCA는 Mahalanobis distance를 활용하여 입력 인스턴스의 태그 임베딩과 각 클래스 분포 간의 거리를 계산합니다. 이를 통해 가장 유사한 k개의 클래스를 선택하여 ICL 프롬프트를 생성합니다. 이 접근 방식은 Irrelevant information을 제거하고 입력 토큰 한계를 효과적으로 관리할 수 있게 해줍니다. ECL은 별도의 추가 학습 없이 클래스 평균과 공분산 행렬을 점진적으로 업데이트합니다.

- **Performance Highlights**: 실험 결과, InCA는 기존 CL 기준과 비교하여 상당한 성능 향상을 보여주었습니다. InCA는 특히 다양한 벤치마크 데이터 세트에서 효과적으로 Scalability와 Accuracy를 균형 있게 유지했습니다. 이러한 성능 개선은 ICL의 이점을 활용한 새로운 CL 패러다임의 가능성을 보여줍니다.



### Predicting Artificial Neural Network Representations to Learn Recognition Model for Music Identification from Brain Recordings (https://arxiv.org/abs/2412.15560)
Comments:
          18 pages, 10 figures

- **What's New**: 이 연구는 인공지능 신경망(ANN)의 표현이 동일한 청각 자극에 대해 피질 표현과 유사하다는 점에 기반하여, 노이즈가 있는 뇌 기록을 통해 음악 인식을 위한 새로운 인식 모델을 개발하였습니다. 특히 EEG(전기뇌파측정) 기록을 이용하여 ANN 표현을 예측하는 모델을 훈련시키고, 기존보다 더 높은 분류 정확도를 달성한 것을 보여주고 있습니다. 이러한 방법론은 뇌-컴퓨터 인터페이스(BCI)와 신경 디코딩 기술을 개선하는 데 기여할 잠재력을 가지고 있습니다.

- **Technical Details**: 이 연구는 음악 청취에 따른 EEG 뇌 기록을 입력으로 사용하여 인식 모델을 구성하였습니다. ANN 표현을 예측하는 방식으로 모델을 훈련시키며, 특히 대칭 학습(contrastive learning) 기법을 통해 클래스를 구분하는 성능을 극대화하는 두 가지 주요 기술을 도입했습니다. 실험적으로 ANN 표현을 예측할 때 노이즈를 최소화하기 위해 비선형 모델을 사용하고, 최적의 손실 가중치를 통해 모델의 성능을 최적화했습니다.

- **Performance Highlights**: 이번 연구에서는 20명의 참가자가 10곡의 음악을 들을 때 수집된 EEG 데이터인 NMED-T 데이터셋을 활용하여, 음악 식별 과제를 수행하였습니다. 특히 가장 높은 분류 정확도는 약 200ms의 뇌 반응 지연을 가정할 때 발생하였으며, 모델은 EEG 입력의 길이가 길어질수록 향상된 음악 식별 능력을 보였습니다. 또한, 개인 차와 음악 차이에 따른 정확도 변이도 분석하여 신경 과학적 통찰을 제공하였습니다.



### NGQA: A Nutritional Graph Question Answering Benchmark for Personalized Health-aware Nutritional Reasoning (https://arxiv.org/abs/2412.15547)
- **What's New**: 이 논문은 개인 맞춤형 영양 건강을 위한 Nutritional Graph Question Answering (NGQA) 벤치마크를 발표했습니다. 이는 사용자 특정 건강 조건에 기초하여 음식이 건강한지 평가할 수 있는 최초의 데이터셋으로, 사용자의 의료 정보와 식단 행동을 통합하여 복잡한 영양 질문에 응답할 수 있도록 설계되었습니다. NGQA는 National Health and Nutrition Examination Survey (NHANES)와 Food and Nutrient Database for Dietary Studies (FNDDS)의 데이터를 활용하여 개별 사용자의 건강에 적합한 영양 성분을 명확히 설명합니다.

- **Technical Details**: NGQA 벤치마크는 세 가지 질문 복잡성 설정(희소, 표준, 복잡)을 포함합니다. 각 질문 유형은 세 가지 다운스트림 작업(이진 분류 – B, 다중 레이블 분류 – ML, 텍스트 생성 – TG)을 통해 평가되어 다양한 추론 측면을 탐구합니다. 연구는 LLM(backbone)와 기준 모델을 사용한 광범위한 실험을 통해 이 벤치마크가 기존 모델에 효과적으로 도전할 수 있음을 보여줍니다.

- **Performance Highlights**: NGQA는 개인 맞춤형 영양 건강 연구와 GraphQA 연구를 발전시키는 데 기여합니다. 본 연구는 사용자 의료 정보를 포함하는 최초의 벤치마크를 만들어 영양 질문 응답 작업에서 중요한 연구 격차를 해소합니다. 또한, NGQA는 GraphQA의 범위를 넓혀 보다 포괄적인 평가를 가능하게 하며, 전체 데이터 전처리에서 모델 평가에 이르는 완전한 코드베이스를 제공하여 새로운 모델 통합성을 위한 확장성을 지원합니다.



### De-singularity Subgradient for the $q$-th-Powered $\ell_p$-Norm Weber Location Problem (https://arxiv.org/abs/2412.15546)
Comments:
          AAAI 2025

- **What's New**: 이번 논문에서는 Weber 위치 문제의 발달된 비-특이성(subgradient) 방법을 제안하고 있습니다. 기존의 방법이 $q$-제곱 $	ext{ℓ}_2$-노름의 경우에만 적용 가능했던 반면, 본 논문에서는 $1 \\leqslant q \\leqslant p$이고 $1 \\leqslant p < 2$인 경우에 확대 적용할 수 있는 방법을 모색합니다. 이는 아직 해결되지 않은 상황을 포함하는 주요 발전을 의미합니다.

- **Technical Details**: 제안된 방법은 비-특이성(subgradient)이 설정된 이론적 배경을 가지고 있으며, 이는 특이점(singular point)의 연속체를 포함하고 있습니다. 문제의 목표 함수의 기하학적 특징 또한 복잡하여, 최소값(minimum) 및 하강 방향(descent direction)의 특성을 식별하기가 매우 어렵습니다. 이 문제를 해결하기 위해 $q$-제곱 $	ext{ℓ}_p$-노름을 사용한 Weiszfeld 알고리즘을 개발했습니다.

- **Performance Highlights**: 여섯 개의 실제 데이터 세트에 적용한 실험 결과, 제안된 $q$P$p$NWAWS 알고리즘이 특이점 문제를 효과적으로 해결하고 선형적인 계산 수렴율(linear computational convergence rate)을 달성하였습니다. 이러한 결과는 실제 시나리오에서 기준 알고리즘보다 뛰어난 성능을 보여줍니다.



### The Impact of Cut Layer Selection in Split Federated Learning (https://arxiv.org/abs/2412.15536)
Comments:
          16 pages, 1 figure, AAAI FLUID Workshop 2025

- **What's New**: 이번 논문에서는 Split Federated Learning (SFL)의 성능과 수렴성을 컷 레이어(cut layer) 선택과의 관계에서 정량적으로 분석합니다. SFL은 Federated Learning과 Split Learning의 장점을 결합한 분산 머신러닝 패러다임으로, SFL-V1과 SFL-V2라는 두 가지 주요 변형이 있습니다. 연구 결과, SFL-V1의 성능은 컷 레이어의 선택에 크게 영향을 받지 않는 반면, SFL-V2는 컷 레이어 선택에 따라 성능이 현저히 달라지는 것으로 나타났습니다.

- **Technical Details**: SFL은 클라이언트에서 초기 레이어를 배치하고 트레이닝 서버에서 나머지 레이어를 실행하는 방식으로 작동하며, 이는 기계 학습 모델의 프라이버시를 유지하면서도 효율성을 높입니다. 기존의 Split Learning 방식은 교육 시간이 늘어나고 확장성이 떨어지는 한계가 있었으나, SFL은 이러한 문제를 해결하여 리소스가 제한된 환경에서도 효율적인 분산 학습이 가능하게 합니다. 특히, SFL-V1에서는 클라이언트 간의 독립성이 유지되며 컷 레이어 선택에 대한 저항력을 보이지만, SFL-V2는 클라이언트 활성화의 효과적인 학습을 통해 성능이 달라집니다.

- **Performance Highlights**: Numerical experiments에서는 SFL-V2의 적절한 컷 레이어 선택이 FedAvg보다 더 나은 성능을 보임을 확인했습니다. SFL-V1은 컷 레이어에 관계없이 일관된 성능을 유지하지만 SFL-V2보다 낮은 정확성을 나타냅니다. 이러한 결과는 다양한 데이터셋과 두 가지 신경망 구조를 사용하여 검증되었으며, 컷 레이어 선택이 SFL의 전반적인 성능에 큰 영향을 미친다는 사실을 강조합니다.



### DualGFL: Federated Learning with a Dual-Level Coalition-Auction Gam (https://arxiv.org/abs/2412.15492)
Comments:
          12 pages, 6 figures. Accepted by AAAI25

- **What's New**: DualGFL은 협력-경쟁 환경에서의 이중 수준 게임을 활용한 혁신적인 연합 학습 프레임워크로, 클라이언트가 자율적으로 훈련 참여 여부를 결정하며 서버는 자원을 최적화하기 위해 참가자를 선택할 수 있도록 지원합니다. 이 프레임워크는 클라이언트의 선호도에 기반한 파레토 최적 파티셔닝 알고리즘을 포함하며, 경매 기반의 효율적인 자원 배분을 통해 더 나은 유틸리티를 제공합니다.

- **Technical Details**: DualGFL은 두 가지 게임 구조, 즉 하위 수준의 연합 게임과 상위 수준의 다속성 경매 게임을 구현합니다. 하위 수준에서 클라이언트는 자신의 유틸리티와 선호도를 고려하여 연합을 형성하고, 상위 수준에서 연합은 훈련 참여를 위해 입찰을 하며, 중앙 서버는 자원의 제약 조건 하에 최적 입찰을 결정하게 됩니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 실험 결과, DualGFL은 클라이언트의 평균 유틸리티를 효과적으로 개선하며, 서버의 유틸리티를 크게 향상시키고 단일 수준 게임 기반의 기존 방법들보다 더 나은 테스트 정확도를 달성하는 것으로 나타났습니다.



### Toward Appearance-based Autonomous Landing Site Identification for Multirotor Drones in Unstructured Environments (https://arxiv.org/abs/2412.15486)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구에서는 비구조적 환경에서 자율적으로 착륙 가능한 위치를 식별하는 문제를 해결하기 위해, RGB 이미지로부터 안전 지역과 위험 지역을 분류하는 경량의 이미지 세그멘테이션(Classifier)을 제안합니다. 이 classifier는 이미지와 마스크 데이터 세트를 생성하는 데 드는 비용 문제를 피하기 위해 자동으로 합성된 데이터를 활용합니다. 또한, U-Net 모델을 훈련시켜 실제 데이터를 테스트하고 실시간 드론 플랫폼에서 시연합니다.

- **Technical Details**: 이 연구의 핵심은 환경을 비행 중에 분석하기 위해 LiDAR와 같은 복잡한 센서를 사용하지 않고, 일반적인 RGB 카메라를 사용하는 것입니다. 우리는 드론이 자동으로 지형을 조사할 수 있는 능력을 활용하여 합성된 데이터 세트를 생성하고, 이를 통해 기존의 수동 라벨링의 부담을 줄입니다. 최종적으로, 우리는 1MB 정도의 작은 U-Net 모델을 제안하여 전력 효율적인 하드웨어에서 실시간으로 동작할 수 있게 합니다.

- **Performance Highlights**: 제안된 시스템은 18개의 검증 사례 중 15개를 올바르게 분류하였으며, 이는 드론 플랫폼에서 실시간으로 실행되었습니다. 이 방식은 다양한 환경에서 운용 가능성을 높이며, 향후 드론의 자동 착륙을 완전히 구현할 수 있는 기초를 제공합니다. 특히, 다양한 데이터 소스에서 생성된 합성 데이터 세트를 통해 데이터 수집의 유연성을 확보한 점이 주목할 만합니다.



### Difficulty-aware Balancing Margin Loss for Long-tailed Recognition (https://arxiv.org/abs/2412.15477)
- **What's New**: 이 논문은 클래스 불균형과 인스턴스 난이도를 동시에 고려하는 Difficulty-aware Balancing Margin (DBM) 손실 함수를 제안합니다. 기존 방법들이 클래스 수준의 불균형에 주로 집중한 반면, DBM 손실은 각 개별 샘플의 난이도 변화를 반영하여 학습의 편향을 줄이는데 도움을 줍니다. 두 가지 구성 요소로 이루어진 DBM 손실은 클래스별 마진(class-wise margin)과 인스턴스별 마진(instance-wise margin)을 포함하여, 더 어려운 샘플에 대해 더 큰 마진을 할당합니다.

- **Technical Details**: DBM 손실은 클래스 수준의 빈도 불균형을 완화하기 위한 클래스 별 마진과, 개별 샘플의 난이도에 따라 조정되는 인스턴스 별 마진으로 구성됩니다. 이 방법은 복잡한 샘플에 대해 추가적인 마진을 부여하여, 클래스 간 구별력을 높입니다. DBM 손실은 기존의 LTR(long-tailed recognition) 접근 방식과 매끄럽게 결합되며, 여러 벤치마크에서 일관된 성능 향상을 보여줍니다.

- **Performance Highlights**: CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT 및 iNaturalist2018과 같은 다양한 긴 꼬리 인식 벤치마크에서 성능을 개선하였습니다. DBM 손실은 기존의 LTR 기술과 호환 가능하며, 상당한 추가 계산 비용 없이 사용될 수 있습니다. 실험 결과, 제안된 방법은 주요 긴 꼬리 인식 벤치마크에서 경쟁력 있는 성능을 보여 품질이 검증되었습니다.



### Predicting Long-Term Student Outcomes from Short-Term EdTech Log Data (https://arxiv.org/abs/2412.15473)
Comments:
          Accepted to the 15th International Learning Analytics and Knowledge Conference (LAK2025)

- **What's New**: 이 연구는 학생들이 교육 소프트웨어를 처음으로 사용한 몇 시간 동안의 로그 데이터를 이용하여 연말 종합 평가 결과를 예측하는 데 유용한 기계 학습 예측 모델을 탐구합니다. 이전의 연구들은 주로 전체 사용 기간의 로그 데이터나 짧은 세션 후 몇 분의 로그를 사용하여 결과를 예측했습니다. 반면, 본 연구는 짧은 기간의 데이터가 장기적인 결과에 대한 신뢰성 있는 신호를 제공할 수 있음을 주장합니다. 이 연구는 우간다와 미국의 학생들로부터 수집된 다양한 데이터셋에서 실행됩니다.

- **Technical Details**: 연구에서는 학생들이 처음 몇 시간의 교육 소프트웨어 사용 로그를 분석하여 예측 모델을 개발했습니다. 여기서 사용된 로그 데이터는 학생들이 경험하는 다양한 상호작용을 포함하며, 기계 학습 기법을 통해 결과를 예측하는 데 사용됩니다. 세 가지 서로 다른 데이터셋을 통해 장기적인 외부 평가 성과와 관련된 예측 정확도 지표를 고려하였으며, 2-5시간의 짧은 로그 데이터가 유의미한 예측 정보를 제공할 수 있음을 발견했습니다.

- **Performance Highlights**: 분석 결과, 학생의 초기 사용 로그가 연말 외부 평가의 성과를 예측하는 데 유의미한 정보를 제공할 수 있는 것으로 나타났습니다. 이는 교육 도구의 효과성을 평가하고 개선하는 데 중요한 시사점을 제시합니다. 또한, 이 예측 모델은 교사들에게 학생들의 개별 및 집단 성과에 대한 인사이트를 제공하여 교육적 지원과 자원 배분에 도움이 될 수 있습니다.



### TalkWithMachines: Enhancing Human-Robot Interaction for Interpretable Industrial Robotics Through Large/Vision Language Models (https://arxiv.org/abs/2412.15462)
Comments:
          This paper has been accepted for publication in the proceedings of the 2024 Eighth IEEE International Conference on Robotic Computing (IRC)

- **What's New**: 이 논문에서는 안전-critical(안전 중요) 산업에서 활용 가능한 해석 가능하고 인간-로봇 상호작용을 강화하기 위한 새로운 접근법을 제안합니다. 자연어(natural language)를 통해 로봇에 명령을 내리고 로봇이 주변 환경을 이해할 수 있도록 하기 위해 LLMs(대형 언어 모델)와 VLMs(비전 언어 모델)의 통합을 탐구하고 있습니다. 로봇의 내부 상태와 의도를 이해하기 쉽게 설명하는 방식으로, 보다 안전하고 효과적인 운영이 가능하도록 합니다.

- **Technical Details**: 본 연구는 로봇 조작(control) 및 인식(perception)에 대한 LLMs와 VLMs의 발전을 바탕으로 하며, 로봇이 이해할 수 있는 저수준의 제어 명령 패턴을 생성할 수 있는 잠재력을 가지고 있음을 보여줍니다. 로봇의 내부 및 외부 상태를 이미지나 텍스트 기반으로 표현하여, 복잡한 궤적의 생성 및 상황 인식을 위한 패턴 인터페이스를 소개합니다. 이를 통해 로봇의 물리적 한계와 안전한 명령 실행을 위한 인식 능력을 키울 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 LLMs와 VLMs가 복잡한 궤적을 생성하고, 환경의 맥락 인식을 유지하며, 간접적 의사소통 신호를 해석할 수 있는 능력이 있음을 보여줍니다. 또한, 로봇의 물리적 구조에 대한 정보가 안전한 명령 실행을 위한 인식에 미치는 영향을 분석하여, 해석 가능하고 인간 중심의 로봇 시스템 개발의 방향성을 제시합니다. 제안된 개념들은 로봇 팔 조작 시뮬레이션을 통해 검증되었습니다.



### Learning charges and long-range interactions from energies and forces (https://arxiv.org/abs/2412.15455)
- **What's New**: 이번 연구에서는 Latent Ewald Summation (LES) 방법을 소개하며, 이는 원자 전하를 명시적으로 학습하거나 전하 균형을 맞추지 않고도 장거리 전기력을 포착할 수 있게 해줍니다. LES는 물리적 부분 전하(physical partial charges)를 학습하는 능력과 전하 상태를 인코딩하는 기능을 포함하여 전하 중립성 제약을 부여할 수 있는 옵션을 제공합니다. 다양한 도전적인 시스템에 대한 벤치마크를 수행한 결과, LES가 물리적 부분 전하, 쌍극자(moment) 및 사중극자(quadrupole moments)를 효과적으로 추론할 수 있다는 것을 보여주었습니다.

- **Technical Details**: LES는 총 잠재 에너지를 단거리(SR) 및 장거리(LR) 구성 요소로 분할합니다. 각 원자 i의 짧은 거리 에너지는 그 원자의 로컬 특성에 기반하며, 이를 통해 원자 간의 상호작용을 더욱 정확하게 모델링할 수 있습니다. 특히, LES는 기존의 머신러닝 원자간 포텐셜(MLIPs)과 통합 가능하며, 효과적으로 장거리 상호작용을 시뮬레이션하는 데 기여합니다.

- **Performance Highlights**: LES는 여러 테스트 시스템에서 기존 방법들에 비해 우수한 성능을 나타냈습니다. 특히, LES 프레임워크는 단일 전하 채널로 국한될 때도 물리적 부분 전하 및 쌍극자 모멘트를 단순히 참고 에너지와 힘을 학습하는 것만으로 추론할 수 있습니다. 이 연구에서는 LES의 이점을 강조하며, 장거리 상호작용을 명시적으로 학습하는 기존 방법들과 비교를 통해 그 우수성을 보여줍니다.



### Energy consumption of code small language models serving with runtime engines and execution providers (https://arxiv.org/abs/2412.15441)
Comments:
          26 pages, submitted to journal

- **What's New**: 이 논문은 코드 생성의 맥락에서 소프트웨어 엔지니어들이 LM(언어 모델)의 에너지 효율성을 높이기 위해 감안해야 할 런타임 엔진과 실행 공급자에 대한 분석을 제시합니다. 신뢰할 수 있는 실험을 통해 SLM(소형 언어 모델) 및 여러 실행 제공자를 비교하여 에너지 소비, 실행 시간 및 자원 활용도를 평가했습니다. 특히, TORCH와 CUDA의 조합이 다른 설정에 비해 에너지 효율성에서 뛰어난 성과를 보이는 것으로 나타났습니다.

- **Technical Details**: 연구는 12개의 코드 생성 SLM을 사용한 다단계 실험 파이프라인을 기반으로 하며, HumanEval 벤치마크에서 생성한 데이터 세트를 이용하여 다양한 설정에서 에너지 소비와 실행 시간을 측정했습니다. 실행 제공자에 따라 CUDA 설정이 CPU 공급자보다 에너지 및 시간 모두에서 우수한 성능을 보였으며, TORCH와 CUDA 조합이 가장 높은 에너지 효율성을 나타냈습니다. 이 연구는 런타임 엔진과 실행 제공자 선택이 에너지 효율성에 미치는 영향을 강조합니다.

- **Performance Highlights**: TORCH와 CUDA의 조합은 37.99%에서 89.16%의 에너지 절약 효과를 보였으며, 실행 시간도 47.84%에서 89.74% 감소했습니다. 이와 함께 ONNX와 CPU 공급자 조합을 통한 최적화된 런타임 엔진이 CPU 기반 설정에서 8.98%에서 72.04%의 에너지 절약을 달성했습니다. 전체적으로, 이 연구는 에너지 효율성과 성능을 최적화하기 위해 적절한 설정을 선택하는 것이 소프트웨어 엔지니어에게 중요하다는 점을 강조합니다.



### Efficient Neural Network Encoding for 3D Color Lookup Tables (https://arxiv.org/abs/2412.15438)
Comments:
          14 pages, 13 figures; extended version; to appear in AAAI 2025

- **What's New**: 이 연구에서는 수백 개의 3D 색상 룩업 테이블(3D LUTs)을 단일 컴팩트 표현으로 인코딩할 수 있는 신경망 아키텍처를 개발하였습니다. 제안된 모델은 0.25MB 이하의 메모리 사용량으로, 512개의 LUT를 재구성할 수 있으며, 평균 색상 차이(ΔE)가 2.0 이하로 허용되는 색상 왜곡을 유지합니다. 또한, 네트워크 구조에 약간의 수정으로 역 색상 처리가 가능한 쌍향 인코딩(bijective encoding) 기능도 구현했습니다.

- **Technical Details**: 제안된 접근방식에서는 RGB 입력 색상을 RGB 출력 색상으로 매핑하는 3D LUT의 수학적 정의를 제공합니다. 연구는 다양한 네트워크 크기 변형을 통해 512개의 LUT를 인코딩하는 모델을 설계하였습니다. 이 모델은 각 LUT를 2ms 이내에 복구할 수 있어 실시간 응용이 가능합니다. 또한, 입력 색상의 가중치를 조정하는 대체 손실 함수도 도입하여 자연 이미지 색상에 대한 품질 향상이 이루어졌습니다.

- **Performance Highlights**: 모델은 평균 색상 차이(ΔE)를 1.0 이하로 유지하면서 512개의 LUT를 효과적으로 복구했습니다. 제안된 신경망은 다양한 LUT를 암묵적으로 압축하고, 기존 방법보다 99% 이상의 압축률을 달성합니다. 마지막으로, 쌍향 인코딩에 대한 수정으로 LUT를 역으로 처리할 수 있어 컴퓨터 비전과 이미지 처리 분야에서 활용 가능성이 더욱 높아졌습니다.



### Cosmology with Persistent Homology: Parameter Inference via Machine Learning (https://arxiv.org/abs/2412.15405)
Comments:
          28 pages, 8 figures, 4 tables

- **What's New**: 이 논문은 지속적 동형성(persistent homology)이 우주론적 매개변수와 원시 비가우시안성(primordial non-Gaussianity) 진폭을 제한하는 잠재적 능력을 조사하고 있습니다. 지속적 이미지(persistence images, PIs)가 빅스펙트럼(bispectrum) 및 파워 스펙트럼(power spectrum)과 결합된 것과 비교했을 때 우수한 예측 능력을 보였습니다. 이는 원시 비가우시안성의 제약 가능성을 제시하며, PIs와 PS/BS를 결합했을 때의 성능도 논의되었습니다.

- **Technical Details**: 이 논문에서는 지속적 동형성이 우주론적 분석을 위한 강력한 도구로 자리잡고 있음을 강조합니다. PIs는 물리적 해석을 포함한 위상적 특성을 추적할 수 있어 기계 학습 및 통계적 추론에 적합합니다. 저자들은 고차원 공분산 행렬 추정과 높은 차원의 상관관계를 처리하기 위한 신경망 기반 추론 방법을 사용하여, PIs에서 패턴을 추출하고 있습니다.

- **Performance Highlights**: PIs는 주어진 우주론적 매개변수를 제약하는 데 효과적인데, 특히 $_{m NL}^{m loc}$에 대해 뛰어난 성능을 보였습니다. PS/BS와 결합할 경우 약간의 이점을 보여주지만, PIs 자체가 이미 유용한 정보를 담고 있어 PS/BS에서 추가적 정보를 거의 못 찾는 것으로 드러났습니다. 이 논문에서는 클러스터 및 빈 공간과 같은 위상적 특징들이 $_{m NL}^{m loc}$과 $m m{Ω_m}$의 식별에 중요한 역할을 한다고 발표하였습니다.



### Investigating Relational State Abstraction in Collaborative MARL (https://arxiv.org/abs/2412.15388)
- **What's New**: 이번 논문은 협력적 Multi-Agent Reinforcement Learning (MARL)에서 샘플 효율성과 성능에 끼치는 관계 상태 추상화의 영향을 탐구합니다. 새로운 critic 아키텍처인 MARC (Multi-Agent Relational Critic)를 제안하며, 공간 관계를 기반으로 하는 이 추상화가 비즈니스 환경에서 에이전트 간의 직접 통신이 불가능할 때의 성능을 향상시킬 수 있다는 점이 흥미롭습니다. 이는 복잡한 디자인이나 특정 작업에 대한 공학적 접근 없이도 이루어질 수 있음을 보여줍니다.

- **Technical Details**: MARC는 상태를 공간 그래프로 변환한 후, 관계형 그래프 신경망(relational graph neural network)을 통해 처리하는 간단하면서도 효과적인 critic 아키텍처입니다. 각 에이전트는 물체와 그 관계를 기반으로 한 그래프 표현을 통해 관찰 결과를 추상화하며, 이를 통해 샘플 효율성과 일반화 가능성을 모두 향상시킬 수 있습니다. 또한, 논문에서는 수중 로봇공학과 같은 환경에서 에이전트 간의 직접적인 통신이 불가능할 때 이 방법의 유용성을 강조합니다.

- **Performance Highlights**: MARC는 총 6가지 협력 작업 및 이질적인 에이전트를 활용한 새로운 환경에서 평가되었으며, 최신 MARL 기준들과 비교하여 샘플 효율성과 비대칭(performance)에서의 개선을 demonstrated합니다. 이는 에이전트와 객체 간의 관계 정보를 기반으로 한 설계 선택이 학습에 미치는 영향을 종합적으로 분석한 결과입니다. 우리의 연구 결과는 공간 관계적 유도 편향(spatial relational inductive biases)을 최소한으로 통합하는 것이 상당한 이점을 제공할 수 있음을 보여줍니다.



### Adaptive Urban Planning: A Hybrid Framework for Balanced City Developmen (https://arxiv.org/abs/2412.15349)
- **What's New**: 이번 연구는 도심 개발의 필수 요소인 인프라의 균형 잡힌 분배와 지역 주민의 구체적인 선호를 통합할 수 있는 새로운 접근 방식을 제안합니다. 두 단계로 구성된 방법론을 통해, 첫 번째 단계에서는 결정론적 솔버가 전체 도시의 기본 인프라 요구 사항을 최적화합니다. 이어서 네 개의 전문화된 계획 에이전트가 각기 다른 하위 지역의 인구 통계학적 수정 사항을 제안합니다.

- **Technical Details**: 이 연구는 결정론적 솔버가 유전 알고리즘(Genetic Algorithms, GA)을 활용하여 기본 서비스와 녹지 공간에 대한 접근성을 극대화하는 방법을 사용합니다. 네 개의 지역 전담 계획 에이전트는 특정 지역의 필요를 대변하고, 이를 마스터 플래너가 평가하여 최종 도시 계획에 통합하는 방식입니다. 이러한 두 단계 접근 방식은 다양한 커뮤니티의 요구를 효과적으로 반영할 수 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 하이브리드 도시 계획 프레임워크는 세 개의 인도 도시에서의 실험을 통해 도시 기능을 유지하면서 지역 인구의 요구사항을 보다 세밀하게 수용하는 데 효과적임을 보여주었습니다. LLM 에이전트를 사용함으로써 도시 계획의 포용성을 높이고, 주민의 요구를 고려한 맞춤형 해결책을 제공할 수 있었습니다.



### Exploring Machine Learning Engineering for Object Detection and Tracking by Unmanned Aerial Vehicle (UAV) (https://arxiv.org/abs/2412.15347)
Comments:
          Accepted at ICMLA '24

- **What's New**: 최근 기술 발전으로 인명 구조(SAR) 작업에 무인 aerial vehicles (UAV)를 활용할 수 있는 가능성이 증가하고 있습니다. 본 연구는 실내 환경에서 자율 드론 시스템을 개발하여 Roomba 진공청소기를 목표로 삼고, 이 기술의 SAR 응용 가능성을 보여줍니다. UAV는 기존의 인력 구조 방식에 비해 위험을 줄이면서 구조 작업을 보다 효율적으로 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 Automated Labeling and Detection of Objects for Tracking (ALDOT)라는 머신러닝 기반의 프레임워크를 제안하여, 비디오 데이터를 처리하고 이동하는 객체를 효율적으로 탐지하고 추적하는 기술을 개발했습니다. Roomba 진공청소기를 추적하기 위해 Parrot Mambo 드론을 활용하였으며, 고해상도 카메라를 통해 indoor 환경에서 비디오 데이터를 수집했습니다. YOLOv4 모델을 사용하여 실시간 객체 탐지 및 추적 작업을 수행하였고, 데이터셋의 정확성을 보장하기 위해 여러 단계를 거쳐 라벨링 작업이 진행되었습니다.

- **Performance Highlights**: 실험 결과, YOLOv4 모델을 적용하여 Roomba를 96%의 정확도로 탐지하며, 평균 손실(loss)은 0.1942로 확인되었습니다. 이러한 성과는 자율 드론이 다양한 환경에서 효과적으로 작동할 수 있음을 보여줍니다. 또한, 본 연구는 SAR 작전에서 UAV의 가능성을 탐구하며, 향후 구조 작업에 대한 실용적인 기술 개발로 이어질 방향성을 제시합니다.



### Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis (https://arxiv.org/abs/2412.15322)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 MMAudio라는 새로운 다중 모달 조인트 트레이닝 프레임워크를 통해 비디오 및 선택적 텍스트 조건을 바탕으로 고품질 오디오를 합성하는 방법을 제안합니다. MMAudio는 비디오 데이터에 국한되지 않고, 대규모 텍스트-오디오 데이터와 함께 학습하여 의미적으로 정렬된 고품질 오디오 샘플을 생성할 수 있습니다. 또한, 조건부 동기화 모듈을 통해 프레임 레벨에서 비디오와 오디오의 동기화를 향상시킵니다.

- **Technical Details**: MMAudio는 비디오, 오디오 및 텍스트를 단일 트랜스포머 네트워크에서 함께 고려하며, 학습 중 누락된 모달리티를 마스킹합니다. 이는 오디오-비주얼 및 오디오-텍스트 데이터셋을 사용하여 스크래치로 학습할 수 있게 해 주며, 자원 데이터에 대한 모델의 이해를 높입니다. 추가로, 우리는 고속의 비주얼 피쳐를 사용하여 정확한 프레임 수준 동기화를 이끌어내는 조건부 동기화 모듈을 도입했습니다.

- **Performance Highlights**: MMAudio는 오디오 품질, 의미적 정렬, 오디오-비주얼 동기화 측면에서 공공 모델 중 새로운 상태를 달성하며, 8초의 클립을 생성하는 데 1.23초의 짧은 추론 시간을 자랑합니다. 다중 모달 접근 방식은 텍스트-오디오 생성에서도 경쟁력 있는 성능을 보여주어, 조인트 트레이닝이 단일 모달리티 성능에 방해가 되지 않음을 보여줍니다. 최종적으로 MMAudio는 제시된 모델 사이즈와 샘플링 주파수에서 뛰어난 성능을 확보하였습니다.



### Enhancing Masked Time-Series Modeling via Dropping Patches (https://arxiv.org/abs/2412.15315)
- **What's New**: 이번 논문에서는 기존의 masked time-series modeling을 향상시키기 위해 시간 시계열의 서브 시퀀스 레벨 패치를 랜덤으로 제거하는 방법, DropPatch를 제안합니다. 이 방법은 사전 훈련 효율성을 제곱 수준으로 개선하고, 도메인 내(in-domain), 도메인 간(cross-domain), 몇 샷(few-shot) 학습 및 cold start와 같은 다양한 시나리오에서 모델링의 추가적인 장점을 제공합니다.

- **Technical Details**: DropPatch는 입력 샘플에서 패치를 랜덤으로 제거하여 모델의 훈련 효율성을 높이는 데 중점을 둡니다. Masking을 진행하기 전, 선택된 패치는 훈련 과정에서 완전히 제외되어, 멀티 스케일과 다양한 정보에 대한 모델의 집중력을 향상시킵니다. 주요 아이디어는 패치를 제거하는 것입니다. 이를 통해 Transformer의 표현이 낮은 차원 공간(rank-1 linear subspace)으로 수렴하는 속도를 늦추어 특징의 다양성을 증진시킵니다.

- **Performance Highlights**: DropPatch를 적용한 결과, 과적합을 줄이고, 주의 집중(attention focus)을 강화하며, 예측 성능을 향상시키는 등의 명확한 장점을 보여주었습니다. 실험을 통해 DropPatch의 효과를 검증하고, 이 방법이 다양한 다운스트림 작업에서 사전 훈련의 효율성을 향상준다고 밝혔습니다. 따라서 DropPatch는 시간 시계열 분석에서 중요한 구성 요소로 자리 잡을 것으로 기대됩니다.



### MIETT: Multi-Instance Encrypted Traffic Transformer for Encrypted Traffic Classification (https://arxiv.org/abs/2412.15306)
Comments:
          AAAI 2025 accepted

- **What's New**: 이번 연구에서는 Multi-Instance Encrypted Traffic Transformer (MIETT)를 제안하여 암호화된 네트워크 트래픽의 분류에서 기존 방법들의 한계를 극복하고자 했다. MIETT는 각 패킷을 전체 흐름을 나타내는 더 큰 집합 내의 별개의 인스턴스로 처리하는 다중 인스턴스 접근 방식을 채택했다. 이를 통해 서로 다른 패킷 간의 관계를 포착하면서도 흐름 내 패킷의 상호작용 패턴을 효과적으로 학습할 수 있다.

- **Technical Details**: MIETT는 Token-Level Attention과 Packet-Level Attention을 결합한 Two-Level Attention (TLA) 레이어를 적용하여 암호화된 트래픽의 복잡한 동적인 패턴을 학습할 수 있다. 이 모델은 두 가지의 새로운 사전 학습 작업인 Packet Relative Position Prediction (PRPP)와 Flow Contrastive Learning (FCL)을 도입하여 패킷 간의 상대적 위치와 흐름 특유의 동적 특성을 이해하는 능력을 강화했다. 이러한 설계를 통해 MIETT는 보다 정교한 패킷 동역학과 흐름 패턴을 학습하게 된다.

- **Performance Highlights**: MIETT는 다섯 가지 데이터 세트에서 최첨단(SOTA) 결과를 달성하며, 암호화된 네트워크 트래픽의 분류 및 복잡한 네트워크 행동 이해에서 뛰어난 효과를 보였다. 이 모델은 대량의 레이블 없는 데이터를 활용하는 데 강점을 가지며, 이전의 패턴 기반 방법들보다 뛰어난 일반화 능력을 가진다. 연구자는 코드도 제공하여 모델의 재현 가능성을 높이고 있다.



### A Comparative Study of DSPy Teleprompter Algorithms for Aligning Large Language Models Evaluation Metrics to Human Evaluation (https://arxiv.org/abs/2412.15298)
Comments:
          7 pages, 10 tables, two-column format

- **What's New**: 이번 논문에서는 Declarative Self-improving Python(DSPy) 프레임워크의 여러 teleprompter 알고리즘이 인공지능 언어 모델(LLM)의 프롬프트(prompt) 최적화와 인간 주석(annotations)과의 정렬에 어떻게 기여하는지를 분석합니다. 이 연구는 특히, LLM을 평가자로 사용하여 환각 탐지(hallucination detection)를 최적화하는 방법에 초점을 맞추고 있으며, 4가지의 teleprompter 알고리즘을 비교 분석하여 그 성능을 평가합니다.

- **Technical Details**: DSPy는 LLM 파이프라인을 선언형 모듈로 추상화하여 특정 목표(e.g., 정확성)의 관점에서 시스템적으로 최적화하는 프로그래밍 모델입니다. 이 모델의 핵심 요소는 predictors, adapters, assertions, metrics 등의 다양한 구성 요소를 포함하며, teleprompters는 이러한 모듈의 품질을 개선하기 위해 특정 프로세스를 따릅니다. Candidate Generation 단계에서는 모듈의 인스턴스를 찾아 새로운 예제를 생상하고, Parameter Optimization 단계에서는 이러한 후보 매개변수를 최적화하여 최고의 조합을 선택합니다.

- **Performance Highlights**: 실험 결과, 최적화된 프롬프트는 다양한 기준 방법들을 초월하여 환각 탐지에 있어서 우수한 성능을 보였습니다. 또한, 특정 teleprompter들은 실험에서 다른 알고리즘보다 더 나은 성과를 나타냈습니다. 각 teleprompter의 성능 비교는 HaluBench라는 공개 데이터셋을 기반으로 하여 수행되었으며, 이는 프롬프트 최적화를 위한 중요한 통찰을 제공합니다.



### Confidence in the Reasoning of Large Language Models (https://arxiv.org/abs/2412.15296)
- **What's New**: 최근의 연구에서 대형 언어 모델(LLMs)의 응답 정확도에 대한 불확실성 논의는 부족한 상황입니다. 본 연구에서는 LLM의 답변에 대한 신뢰 정도가 정확도와 어떻게 상관관계가 있는지를 평가하고자 합니다. 신뢰는 (i) 재고를 요구했을 때 응답을 유지하는 지속성으로 질적으로 측정되며, (ii) 스스로 보고한 신뢰 점수로 수치적으로 측정됩니다. LLM의 신뢰 수준이 형태 소속 토큰 레벨 확률에 의해 부분적으로만 설명된다는 점이 흥미롭습니다.

- **Technical Details**: 본 연구에서는 GPT4o, GPT4-turbo, Mistral의 세 가지 LLM을 평가합니다. BIG Bench-Hard의 두 가지 벤치마크 세트를 사용하여 인과 판단과 논리적 오류 관련 질문을 437개 평가하며, 확률 및 통계 퍼즐 46개도 포함됩니다. LLM들에게 처음의 답변에 대해 재고하도록 요청하며, 이를 통해 질적 및 수치적 신뢰를 측정합니다. 연구 결과, LLM은 랜덤 추측보다 확실히 뛰어난 성과를 보이나, 신뢰에 대한 과대 평가 경향이 발견되었습니다.

- **Performance Highlights**: LLM들은 대체로 랜덤 추측보다 좋지만, 재고 요청 후 두 번째 답변의 전체 정확도가 종종 첫 번째 답변보다 나쁩니다. 흥미롭게도, 심리적 문구에 따라 마음을 바꾸는 경향이 달라졌습니다. 신뢰 점수에 대한 질문에 과대 평가 경향이 있으며, 질적 신뢰와 수치적 신뢰 간의 상당한 상관 관계가 관찰되었습니다. 본 연구에서 제기된 문제는 LLM이 내재적으로 일관된 신뢰 감각을 갖고 있지 않음을 시사합니다.



### Log-Time K-Means Clustering for 1D Data: Novel Approaches with Proof and Implementation (https://arxiv.org/abs/2412.15295)
Comments:
          Undergraduate Thesis, Department of Computer Science and Engineering, Seoul National University

- **What's New**: 본 논문은 k-means 클러스터링을 위한 새로운 최적화 알고리즘을 도입하며, 1D 데이터에서의 효율성을 극대화하는 방법을 제시합니다. 기존의 방법들이 1D 데이터 구조를 효과적으로 이용하지 못했음을 지적하며, 정렬된 데이터와 prefix sums, 그리고 binary search를 활용하여 계산 성능을 개선합니다. 이로 인해 1D 클러스터링 문제를 해결하는 데 필요한 알고리즘의 복잡성이 크게 감소합니다.

- **Technical Details**: 논문의 주된 기여는 최적화된 k-cluster 알고리즘을 통해 greedy k-means++ 초기화에 대해 O(l⋅k²⋅log n)와 Lloyd's 알고리즘에 대해 O(i⋅k⋅log n)의 시간 복잡도를 달성하는 것입니다. 여기서 l은 greedy k-means++의 로컬 시험 횟수, i는 Lloyd 알고리즘의 반복 횟수를 나타냅니다. 또한 binary search를 기반으로 한 두 클러스터 알고리즘도 제안되어 O(log n)의 실행 시간을 가지면서도 정확한 수렴을 보장합니다.

- **Performance Highlights**: 벤치마크 테스트에 따르면, 이 알고리즘은 대규모 데이터셋에서 scikit-learn보다 4500배 빠른 성능을 보이며, 클러스터링 품질도 유지합니다. 또한, LLM 양자화 작업에서 300배 속도 향상을 이루었으며, 새로운 어플리케이션에서의 유용성을 강조합니다. 실험적으로 증명된 최적화는 이론과 실제를 결합하여 1D k-means 클러스터링을 위한 효율적이고 신뢰할 수 있는 알고리즘을 제공합니다.



### Deep reinforcement learning with time-scale invariant memory (https://arxiv.org/abs/2412.15292)
- **What's New**: 이 연구에서는 scale invariant (스케일 불변) 메모리를 Deep Reinforcement Learning (딥 강화 학습) 에이전트에 통합하는 새로운 접근 방식을 제안합니다. 기존의 LSTM과 같은 반복 메모리 아키텍처와 달리, 이 에이전트는 다양한 시간 규모에서 강인하게 학습할 수 있도록 설계되었습니다. 이를 통해 신경과학 및 인지과학의 원리를 딥 뉴럴 네트워크에 적용하는 것이 가능합니다.

- **Technical Details**: 이 논문에서는 여러 신경 과학 연구에 기초하여, 시간 세포(time cells)와 ramping/decaying activity의 개념을 도입합니다. scale invariant 메모리 모델은 다양한 시범 환경에서 성능을 평가하며, 표준 메모리 아키텍처와 비교합니다. 이는 하이퍼파라미터 조정을 요구하는 대신, 다양한 시간적 관계에 대해 일반화된 학습 능력을 보여줍니다.

- **Performance Highlights**: 실험에서는 scale invariant 메모리를 가진 에이전트가 시간 변화에 대한 적응력을 보이는 반면, 전통적인 머신러닝 시스템은 특정한 시간 규모에서만 성능이 좋음을 보여주었습니다. 이 연구는 강화 학습 에이전트의 성능을 향상시키는 데 있어 인지 과학 이론의 접목 가능성을 탐색했습니다. 최종적으로, scale invariant 메모리를 적용한 에이전트가 다양한 시간적 관계에서 강한 성능을 유지할 수 있음을 입증합니다.



### Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models (https://arxiv.org/abs/2412.15287)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 성능 개선을 위해 새로운 추론 인식 세밀 조정(inference-aware fine-tuning) 패러다임을 제안합니다. 기본적으로, 이 방법은 모델이 추론 시 최고의 성능을 보장하는 방식으로 조정됩니다. 특히 Best-of-N (BoN) 전략을 사용하여 모델이 생성한 여러 응답 중에서 가장 좋은 것을 선택하는 방식을 연구합니다.

- **Technical Details**: 우리는 BoN을 고려한 첫 번째 모방 학습(imitation learning) 및 강화 학습(reinforcement learning) 방법을 개발했습니다. BoN은 비가역적(argmax) 연산자를 통해 주어진 문제에 대한 여러 후보 응답을 생성하고, 이 중에서 최적의 응답을 선택하는데, 이는 비차별적인(non-differentiable) 문제를 해결합니다. BoN 인식 모델을 통해 우리는 RL에서의 탐색(exploration)과 활용(exploitation) 간의 트레이드오프를 묘사하는 메타 전략을 학습하는 것을 보여줍니다.

- **Performance Highlights**: 우리의 BoN 인식 세밀 조정 방법은 성능 개선과 추론 시간 계산을 최적화하는 데 매우 효과적입니다. Gemma 2B 모델은 Hendrycks MATH에서 Bo32 성능이 26.8%에서 30.8%로, pass@32는 60.0%에서 67.0%로 증가했습니다. HumanEval에서도 pass@16이 61.6%에서 67.1%로 향상되었습니다.



### Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining (https://arxiv.org/abs/2412.15285)
- **What's New**: 이 논문에서는 대규모 언어 모델의 효과적인 사전 학습을 위해 데이터 선택, 혼합 및 순서에 대한 전략을 수립합니다. 특히, 두 단계의 사전 학습(two-phase pretraining) 개념을 공식화하고 데이터 혼합 및 선택 방법을 체계적으로 연구하여 모델의 정확도를 극대화하는 방법을 제시합니다. 연구 결과, 무작위 데이터 순서와 자연 분포보다 두 단계 접근법이 평균 3.4% 및 17% 향상된 성능을 보였습니다.

- **Technical Details**: 제안된 두 단계 접근법에서 첫 번째 단계(phase-1)는 다양하고 고품질의 웹 크롤 데이터에 중점을 두고, 두 번째 단계(phase-2)는 수학, 코드 및 위키 데이터와 같은 고품질 데이터 소스를 기반으로 합니다. 데이터 혼합 과정에서 데이터 소스의 품질과 에폭(epoch) 수를 고려하여 최적의 혼합 전략을 개발합니다. 또한, 1T 토큰의 축소 샘플링(downsampled data)을 사용하여 여러 혼합을 탐색한 후 15T 토큰의 전체 데이터로 확장할 수 있는 방법을 검증합니다.

- **Performance Highlights**: 연구에서는 지식, 추론, 코딩 및 수학 벤치마크를 포함하는 다양한 다운스트림 작업을 평가하였습니다. 실험 결과, 품질 및 에폭 기반 혼합은 자연 분포 기반 혼합보다 13.2% 우수하고, 두 단계 접근법은 데이터의 무작위 순서보다 평균 3.4% 더 나은 성능을 보여주었습니다. 또한, 축소 샘플링된 데이터의 결과는 15T 토큰의 장기 스케일에서도 일반화되며, 두 단계 접근법의 확장성과 견고성을 demonstrat합니다.



### Channel Merging: Preserving Specialization for Merged Experts (https://arxiv.org/abs/2412.15283)
Comments:
          accepted by AAAI 2025

- **What's New**: 최근 대형 언어 모델(LLM)의 성능 향상을 위해 작업 특정의 세밀한 튜닝( task-specific fine-tuning )이 활용되고 있습니다. 다양한 LLM을 통합하여 전체적인 능력을 크게 향상시키는 방법이 소개되었지만, 전통적인 앙상블 방법은 메모리 집약적이어서 여러 모델을 동시에 GPU 메모리에 로드해야 하는 비효율성이 존재합니다. 이 문제를 해결하기 위해 제안된 새로운 기술인 Channel Merging을 통해 메모리 사용량을 줄이면서도 성능 저하 없이 높은 성능을 유지할 수 있음을 보여줍니다.

- **Technical Details**: Channel Merging은 유사성을 바탕으로 채널 매개변수를 클러스터링하여 오프라인으로 여러 그룹을 형성합니다. 이를 통해 그룹 내에서 유사한 매개변수만 병합하여 파라미터 충돌을 최소화할 수 있습니다. inference(추론) 중에는 병합된 그룹에서 전문 매개변수를 즉시 조회할 수 있어 전문적인 지식을 보존하며, 이전의 모델 병합 기술보다 적은 매개변수를 로드하게 됩니다.

- **Performance Highlights**: Channel Merging은 영어 및 중국어 추론, 수학 추론, 코드 생성 등 다양한 작업에서 비병합 모델과 동등한 성능을 발휘합니다. 그리고 작업 특정 라우터와 결합했을 때 전통적인 앙상블 방법이 요구되는 매개변수의 53%로 성과를 거두어, 다양한 분야에서의 효율성과 활용 가능성을 입증합니다.



### Functional connectomes of neural networks (https://arxiv.org/abs/2412.15279)
Comments:
          Accepted at the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이 논문에서는 신경망(Neural Networks)과 인간 뇌 기능 간의 연결을 다리 역할을 하는 새로운 접근 방식을 제안합니다. 기능 연결체(functional connectome)에서 얻은 통찰을 활용하여 대규모 신경망의 토폴로지를 특성화할 수 있는 확장 가능한 방법을 제공합니다. 이는 신경망의 해석 가능성을 향상시키고, 그 기초 메커니즘에 대한 더 깊은 이해를 가능하게 합니다.

- **Technical Details**: 제안된 분석 프레임워크는 기능적 MRI와 지속적 그래프 동형성(Persistent Graph Homology) 기술에서 영감을 받았습니다. 이를 통해 고정된 임계값을 사용하지 않고도 신경망 기능을 설명하는 기능 연결체(functional connectome)의 특성을 극대화합니다. 더불어 Wasserstein 거리와 관련된 통계치를 계산하여 신경망을 분석하는 데 중요한 통계적 도구로서의 역할을 하며, centroid 기반 클러스터링 전략을 개발하는 데도 기여합니다.

- **Performance Highlights**: 제안된 프레임워크는 복잡한 신경망 기능 구조를 구별하고 해석하는 데 도움을 주며, 이론적 검증을 통한 다양한 실험을 통해 그 효용성을 입증합니다. 기능적 데이터를 통해 최적화된 신경망에 대한 정보 전파를 연구함으로써, 신경망의 동작 방식에 대한 새로운 통찰을 제공합니다. 이러한 발전은 더욱 투명하고 효율적인 신경망 모델 개발에 기여할 것으로 기대됩니다.



### Exploring Query Efficient Data Generation towards Data-free Model Stealing in Hard Label Setting (https://arxiv.org/abs/2412.15276)
- **What's New**: 본 논문은 데이터 없이 모델을 탈취하는 새로운 접근법인 Query Efficient Data Generation (QEDG)를 제안합니다. 기존 방식들이 고신뢰도의 샘플을 생성하여 목표 모델의 행동을 복제하는 데 어려움을 겪었던 점을 해결하고자, 두 가지 손실 함수를 도입하여 다중 클래스에 걸쳐 더 나은 의사결정 경계에 맞는 샘플을 생성합니다. 이는 공격자가 보낸 쿼리 수를 최소화하면서도 더 많은 감독 정보를 얻을 수 있는 방법론을 다룹니다.

- **Technical Details**: QEDG는 두 가지 손실 함수를 통해 생성된 샘플이 목표 모델의 결정 경계에 가까운 위치에 있도록 하고, 동일 클래스 내의 샘플들 간의 간격을 넓히는 방법을 적용합니다. 또한, 쿼리 없이 단일 요청으로 추가적인 감독 정보를 확보할 수 있는 쿼리 없는 샘플 증강 기법을 제안합니다. 이를 통해 모델의 정확성과 샘플의 다양성을 동시에 확보할 수 있습니다.

- **Performance Highlights**: 여러 데이터 세트에 대한 실험 결과, QEDG는 기존 최첨단 방법들과 비교하여 적은 수의 쿼리로도 더 나은 성능을 보였습니다. 이는 모델 탈취 공격에 있어 효과적인 방안이 될 가능성을 보여줍니다. 또한, QEDG의 주요 기여는 샘플 생성 프로세스를 재구성하고, 목표 모델과 대체 모델 간의 유사성을 더 정확하게 평가할 수 있는 일관성 비율 메트릭을 도입한 것입니다.



### Baichuan4-Finance Technical Repor (https://arxiv.org/abs/2412.15270)
- **What's New**: 이번 논문에서는 Baichuan4-Finance 시리즈의 개발을 소개합니다. 이 모델들은 Baichuan4-Turbo 기본 모델을 기반으로 하며, 금융 분야에 특화되어 있습니다. Baichuan4-Finance-Base와 Baichuan4-Finance로 구성되어 있으며, 금융 지식을 획득하는 새로운 훈련 전략을 제안합니다.

- **Technical Details**: Baichuan4-Finance-Base의 토크나이저는 byte-level byte-pair encoding (BBPE)을 사용하며, 141,056개의 정규 토큰으로 구성됩니다. Qued Query Attention (GQA)을 활용하여 추론 속도를 개선하고, RMSNorm을 통해 학습 안정성을 보장합니다. RoPE를 활용하여 위치 인코딩을 수행하며, 기존의 Multi-Head Attention (MHA)보다 효율적인 아키텍처를 구현했습니다.

- **Performance Highlights**: Baichuan4-Finance-Base는 여러 금융 작업에서 거의 모든 경쟁 모델을 크게 능가하는 성과를 나타냅니다. 일반 LLM 기준에서도 성능을 유지하면서 금융 응용 시나리오에서 더욱 놀라운 성능을 보입니다. 이는 금융 LLM 분야의 혁신을 촉진할 잠재력을 보여줍니다.



### The Reliability Paradox: Exploring How Shortcut Learning Undermines Language Model Calibration (https://arxiv.org/abs/2412.15269)
Comments:
          10 pages; 9 figures. Accepted for publication at the Hawaii International Conference on System Sciences (HICSS-58) 2025

- **What's New**: 이 논문은 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)의 신뢰성과 일반화 능력 간의 관계를 조사합니다. 기존 연구에서 낮은 교정 오류(Expected Calibration Error, ECE)가 신뢰할 수 있는 예측을 의미한다고 여겼지만, 이 연구는 이러한 가정을 뒤집으며, 낮은 교정 오류가 오히려 비일반적인 결정 규칙을 나타낼 수 있음을 보여줍니다.

- **Technical Details**: 모델의 신뢰성을 평가하기 위해, 이 연구에서는 통계적 교정 평가 지표인 ECE를 활용하여 PLMs를 분석합니다. 또한, 통계적 교정 오류 측정이 결정 규칙의 비강건성을 포착하는 데 한계가 있음을 강조하고 있습니다. 단기 학습(shortcut learning)에 기반한 모델의 신뢰도를 평가하는 과정에서 통계적 기법과 함께 데이터 통계를 활용한 접근 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, 잘 교정된 모델이 반드시 신뢰성이 높은 것은 아님을 발견했습니다. 저자들은 PLMs의 다양한 분류 작업에서 모델의 단기 학습 행동을 확인하고, 이러한 행동이 교정 오류와 어떻게 관련되는지 분석했습니다. 이는 PLMs의 신뢰성을 높이기 위해 수학적 교정 능력과 일반화 목표 간의 간극을 메우는 필요성을 제기합니다.



### Toxicity Detection towards Adaptability to Changing Perturbations (https://arxiv.org/abs/2412.15267)
- **What's New**: 이 연구는 독성 콘텐츠 탐지 분야에 새로운 문제인 '지속적 학습 jailbreak 교란 패턴'을 도입했습니다. 이는 사용자가 탐지기를 피하기 위해 새로운 교란 패턴을 창출하는 점을 반영하여 탐지기의 접근 방식을 혁신적으로 변화시키고자 합니다. 연구진은 기존의 수많은 전통적 탐지 방법들이 변화된 교란 패턴에 취약하다는 점을 확인하고, 이에 대한 해결책을 모색합니다.

- **Technical Details**: 논문에서 제안한 방법은 총 9종의 교란 패턴으로 생성된 데이터셋 DynEscape(다이나믹 이스케이프)를 기반으로 하며, 이를 통해 탐지기의 강건성을 유지하기 위해 도메인 증가 학습 방식이 사용됩니다. 연구진은 제안된 데이터셋을 통해 현재 탐지기들이 미지의 유형의 교란 독성 텍스트를 식별하는 데 어려움을 겪고 있다는 것을 체계적으로 검증했습니다. 또한, 지속적 학습 기법을 통해 탐지기는 이전과 새로운 교란 패턴 모두를 인식할 수 있는 능력을 키울 수 있습니다.

- **Performance Highlights**: 연구팀은 제안된 지속적 학습 접근 방식인 DynDetect(다이나믹 탐지)가 기존의 독성 탐지 최첨단 모델들과 비교하여 우수한 성능을 발휘함을 입증했습니다. 이로 인해 독성 콘텐츠의 탐지 정확성이 증가함은 물론, 다양한 교란 패턴에 대한 강건성 또한 확보되었습니다. 연구팀은 자가 감독 학습을 통해 탐지기의 지속적인 성능 향상을 가능하게 하는 새로운 연구 기회를 제공하고자 합니다.



### DisEmbed: Transforming Disease Understanding through Embeddings (https://arxiv.org/abs/2412.15258)
- **What's New**: 이 논문에서는 DisEmbed이라고 하는 질병에 초점을 맞춘 임베딩 모델을 제안합니다. 기존의 모델들이 일반적인 의료 분야에 대해 폭넓게 일반화되어 질병에 대한 깊은 이해를 어려워하는 반면, DisEmbed은 질병 설명, 증상 및 질병 관련 Q&A 쌍의 합성 데이터 세트를 통해 훈련되었습니다. 이 모델은 특히 질병 관련 작업에서 강력한 성능을 보여주며, 진단 시스템과 같은 특정 이용 사례에서 유용하게 활용될 수 있습니다.

- **Technical Details**: DisEmbed 모델은 전통적인 의료 임베딩의 일반화를 벗어나 질병에 관련된 맥락을 깊이 있게 이해하는 데 중점을 두고 개발되었습니다. 모델의 훈련 데이터를 위해 ICD-10-CM 데이터 세트를 기반으로 질병 이름을 생성하고, 해당 질병에 대한 증상 및 설명을 생성하여 질병의 개념을 더 잘 이해할 수 있도록 설계되었습니다. 또한, 모델은 Multiple Negatives Ranking Loss (MNRL) 기법을 사용하여 질병과 관련된 의미 있는 표현을 학습할 수 있도록 최적화되었습니다.

- **Performance Highlights**: DisEmbed의 성능은 질병 특정 데이터 세트를 사용하여 평가되었습니다. 이 모델은 유사한 질병을 구분하고, 질병 관련 문맥을 식별하는 데 특히 뛰어난 성능을 보였습니다. 특히 RAG (retrieval-augmented generation) 작업에서 모델의 성능이 두드러지며, 이를 통해 질병 분류 및 증상 매핑과 같은 다양한 다운스트림 작업에 적용할 수 있음을 입증했습니다.



### Structured Extraction of Real World Medical Knowledge using LLMs for Summarization and Search (https://arxiv.org/abs/2412.15256)
Comments:
          10 pages, 3 figures, Work published in 4th Workshop on Knowledge Graphs and Big Data (In Conjunction with IEEE Big Data 2024)

- **What's New**: 이 논문에서는 질병 발견과 분석을 가속화하기 위해 환자 지식 그래프를 구축하는 새로운 접근 방식을 제안합니다. 기존의 질병 온톨로지가 환자의 상태나 희귀 질병의 미세한 차이를 포착하기 어려운 반면, 대규모 언어 모델 추출 기술을 활용하여 자연어를 통해 데이터를 보다 유연하게 추출할 수 있는 방법을 제시하고 있습니다. 이를 통해 실세계 데이터에서 의미 있는 통찰을 도출하고, 기존 온톨로지에 연결되는 환자 특화 지식 그래프를 구축했습니다.

- **Technical Details**: 이 연구에서 제안한 방법은 메타 데이터, SNOMED-CT, RxNORM, HPO와 같은 기존 온톨로지에 추출된 개체들을 연계하여 'ground truth'를 제공합니다. 실험에 사용된 데이터는 약 3,360만 환자의 대규모 이동 치료 전자 건강 기록(EHR) 데이터베이스로, 이를 통해 Dravet 증후군과 Beta-propeller protein-associated neurodegeneration (BPAN) 환자를 찾는 데 성공했습니다. 환자의 증상 기반 검색 방식과 환자 특화 지식 그래프 구축을 통해 실제 질병 발견 사례를 입증했습니다.

- **Performance Highlights**: 최신 데이터와 사례 연구를 통해 제안된 방법의 효과가 입증되었습니다. LLM 기반 엔터티 추출을 사용하여 Dravet 증후군에 대한 ICD10 코드 검증을 통해 환자 특성을 잘 설명하며, 의료 기록에서 유래한 데이터를 활용해 다양한 질병 연구 결과를 종합적이고 저장된 지식으로 제공합니다. 이와 같이 고도화된 자동화 시스템은 의료 연구와 데이터 통합의 혁신을 가능하게 하며, 실질적인 임상 연구 설계를 가속화하는 데 기여합니다.



### Using Machine Learning to Distinguish Human-written from Machine-generated Creative Fiction (https://arxiv.org/abs/2412.15253)
Comments:
          Accepted for publication at ICAART 2025: this https URL

- **What's New**: 이 연구는 기존의 연구에서 충분히 다뤄지지 않았던 AI 생성 텍스트 탐지를 위한 새로운 접근법인 'AI Detective' 도구를 개발했습니다. 이를 통해 인간 작가의 창작과 AI에 의해 생성된 창의적인 픽션을 구별할 수 있는 기계 학습(ML) 모델을 활용했습니다. 연구 결과, Naive Bayes와 Multi-Layer Perceptron 분류기가 각각 95% 이상의 정확도를 달성하며, 인간 판별자를 크게 초월한 성과를 보였습니다.

- **Technical Details**: 이 논문에서는 데이터 가용성 및 컴퓨팅 파워의 증가 덕분에 신경망 모델과 딥 러닝을 기반으로 한 자동 텍스트 생성 접근법이 발전했다는 점을 강조합니다. 특히, Transformer 모델의 도입은 텍스트 생성에 혁신을 가져왔으며, 이 구조는 인코더와 디코더로 구성됩니다. 연구팀은 이러한 모델을 통해 짧은 문장의 창작물을 정확히 분류할 수 있는 기계 학습 분류기를 개발했습니다.

- **Performance Highlights**: 이 연구에서 사용된 분류기는 특히 짧은 텍스트 샘플(약 100단어)에 대해 높은 정확도로 성능을 보였고, 이전의 연구에서는 이런 짧은 샘플을 분류하는 데 어려움이 있었던 점을 고려할 때 상당한 개선이 이루어진 것입니다. 'AI Detective' 도구는 편집자와 출판사들이 인공지능의 영향을 받는 창작물의 경제적, 문화적 가치를 보호하기 위해 활용될 수 있는 기반을 제공합니다.



### An Enhanced Text Compression Approach Using Transformer-based Language Models (https://arxiv.org/abs/2412.15250)
- **What's New**: 이번 연구에서는 텍스트 복원에서 Transformer 기반의 방법, RejuvenateFormer를 제안합니다. 이는 Lempel-Ziv-Welch 알고리즘을 활용한 새로운 전처리 기법과 무손실 압축 방법을 결합하여 효과적인 압축 성능을 달성합니다. 이전 연구에서 다루지 않았던 무손실 압축 알고리즘과 Transformer 통합의 최적화 문제를 해결하고자 했습니다. 이 접근법은 상태-of-the-art (state-of-the-art) 압축 비율과 BLEU 점수를 달성하며, Transformer 모델을 활용한 새로운 기법들의 가능성을 열었습니다.

- **Technical Details**: 연구에서는 6개의 인코더 및 디코더 레이어로 구성된 맞춤형 아키텍처를 기반으로 하는 RejuvenateFormer를 개발하였습니다. 이 모델은 여러 말뭉치에서 다양한 일반 목적의 무손실 압축 기법을 통해 성능을 평가하고 압축 비율을 극대화하는데 초점을 맞추었습니다. 특히, 전처리 단계에서 Lempel-Ziv-Welch 알고리즘을 통합하여 압축 비율이 각각 12.57, 13.38, 11.42라는 성과를 보여주었습니다. 이를 통해 기존 심층 학습 및 전통적인 방법들과 비교하여 최고 수준의 압축 성능을 입증했습니다.

- **Performance Highlights**: RejuvenateFormer 모델은 EN-DE, EN-FR, BookCorpus 데이터셋에서 각각 27.31, 25.78, 50.45의 BLEU 점수를 기록했습니다. 이러한 성과는 기존의 T5-Small 모델을 포함한 이전의 모든 state-of-the-art 모델을 뛰어넘는 결과입니다. 외부 데이터셋에서의 엄격한 사례 분석을 통해 우리의 기법이 대규모 텍스트 복원 실험에도 효과적임을 입증하였습니다. 이번 연구는 텍스트 복원 및 압축 분야에서의 Transformer 기반 접근법의 새로운 가능성을 제시합니다.



### LLMs for Literature Review: Are we there yet? (https://arxiv.org/abs/2412.15249)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 문헌 리뷰 작성을 지원하는 새로운 접근 방식을 제안합니다. 특히, 본 연구에서 LLMs의 제로샷(zero-shot) 능력을 활용하여 초록을 기반으로 연관된 연구 결과를 검색하고 리뷰를 작성하는 두 가지 컴포넌트로 작업을 분해합니다. 혁신적인 두 단계 검색 전략과 리랭킹 메커니즘을 도입하여 LLM의 효과를 분석합니다.

- **Technical Details**: 연구는 초록에서 의미 있는 키워드를 추출하고, 외부 지식 기반에 쿼리하여 관련 논문을 검색하는 두 단계 검색 프로세스를 구현합니다. 또한, LLM이 후보 논문들에서 특정 발췌의 관련성을 밝힐 수 있도록 유도하는 프롬프트 기반 리랭킹 기법을 분석합니다. 리뷰 생성을 위해 LLM에게 어떤 논문을 인용할지를 지정하는 계획을 제공하는 방안을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론은 기존의 간단한 LLM 기반 생성 방법에 비해 18-26% 더 적은 환각된 참조(hallucinated references)를 생성하며, 품질 좋은 리뷰를 생성하는데 기여합니다. LLM 기반의 계획 수립 접근법이 문헌 리뷰의 품질을 실질적으로 개선함을 보여줍니다. 대조군보다 10%와 30% 향상된 정밀도와 정규화된 리콜(normalized recall)을 기록했으며, 이러한 방법은 연구 결과에 대한 투명성을 증가시킵니다.



### MPPO: Multi Pair-wise Preference Optimization for LLMs with Arbitrary Negative Samples (https://arxiv.org/abs/2412.15244)
Comments:
          Accepted by COLING2025

- **What's New**: 본 연구에서는 Multi Pair-wise Preference Optimization (MPPO) 알고리즘을 도입하여 대량의 언어 모델(LLM)과 사람의 피드백을 효율적으로 정렬할 수 있는 방법을 제안합니다. 기존의 DPO 및 KTO 알고리즘과 달리 MPPO는 보상 모델을 사용하지 않고도 정책 모델을 직접 최적화하여 여러 응답을 효과적으로 활용합니다. 이러한 접근 방식은 모델의 응답 품질을 개선하고, 더 많은 선호 데이터를 최대한 활용할 수 있도록 돕습니다.

- **Technical Details**: MPPO는 모델의 응답 평균 가능성을 활용하여 보상 함수를 피팅(fitting)합니다. 이 연구에서는 Point-wise, Pair-wise, List-wise의 세 가지 주요 구현 방식을 분석한 결과, Pair-wise 방식이 최고의 성능을 달성하는 것을 발견했습니다. 이는 여러 응답을 하나의 쿼리에 대해 최적화하여 희소 데이터 시나리오에서도 효과적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 MPPO는 MT-Bench에서 DPO, ORPO 및 SimPO를 초월하는 성과를 보였으며, Arena-Hard에서는 DPO와 ORPO에 비해 상당한 이점을 나타냈습니다. 이러한 결과는 MPPO가 선호 최적화 작업에서 큰 장점을 보여줌을 강조합니다. MPPO의 실험은 실제 응용 프로그램에서 최적화를 지원하며, 모델 응답의 질을 크게 향상시킵니다.



### Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks (https://arxiv.org/abs/2412.15238)
Comments:
          Accepted to NeurIPS 2024 Workshop on Foundation Model Interventions (MINT)

- **What's New**: 이번 연구에서는 기존의 LLM(대형 언어 모델)에서 발생하던 추론 과제와 관련된 약점을 해결하기 위한 새로운 프레임워크인 Dipper를 제안합니다. Dipper는 학습이 필요 없는 LLM 앙상블(ensemble) 방식으로, 단일 LLM 모델에 최적화된 다양한 프롬프트를 동시에 제공하여 추론 시간에 성능을 개선하도록 설계되었습니다. 이를 통해 사용자는 통합된 방식으로 여러 개의 쿼리를 처리할 수 있게 됩니다.

- **Technical Details**: 연구에서는 Dipper 프레임워크를 통해 각기 다른 프롬프트를 동시에 입력하여 LLM의 출력을 다양화하고자 합니다. 이 방법은 LLM의 고유한 특성인 다양한 출력을 생성하는 능력을 활용하여 성능 향상을 꾀합니다. 특히, 동종 모델을 사용하여 모델의 다양성을 증대시키는 것이 특징입니다.

- **Performance Highlights**: Dipper를 통해 한정된 GPU 메모리 환경에서도 성능을 향상시키는 실험 결과가 도출되었습니다. 예를 들어, MATH 데이터세트에서 3개의 작은 모델(Qwen2-MATH-1.5B-it 모델들)로 구성된 앙상블이 보다 큰 모델(Qwen2-MATH-7B-it)을 능가하는 결과를 보여주었습니다. 이는 Dipper가 실제 문제 해결에서 매우 유용하다는 것을 시사합니다.



### Multi-Branch Mutual-Distillation Transformer for EEG-Based Seizure Subtype Classification (https://arxiv.org/abs/2412.15224)
- **What's New**: 본 연구에서는 EEG 기반 경련 유형 분류를 위한 다중 분기 상호 증류(Multi-Branch Mutual-Distillation, MBMD) Transformer를 제안합니다. 이 모델은 작은 레이블 데이터로 효과적으로 훈련될 수 있으며, 기존의 기계학습 및 딥러닝 접근법보다 더 우수한 성능을 보여줍니다. 또한, EEG 데이터와 다양한 주파수 대역의 웨이브렛(wavelet) 간의 지식 전이를 위한 상호 증류 전략을 도입하였습니다.

- **Technical Details**: MBMD Transformer는 기존 비전 변환기(Vision Transformer)의 모든 짝수 인코더 블록을 다중 분기 인코더 블록으로 교체하여 설계되었습니다. 이 구조는 다양한 주파수 대역에서 추출된 웨이브렛을 처리하기 위해 다중 분기 피드포워드 네트워크(multi-branch feedforward network, FFN)를 사용합니다. 각 웨이브렛은 원시 EEG 데이터와 동일한 클래스 레이블을 사용하며, 모든 분기의 출력을 집계하여 최종 예측 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, MBMD Transformer는 두 개의 공개 EEG 데이터셋에서 전통적인 기계 학습 및 최신 딥러닝 방법보다 좋은 성능을 보였습니다. 본 연구는 EEG 기반의 경련 유형 분류에 대한 지식 증류(knwoledge distillation)의 첫 연구로, 신경망의 내부에서 지식을 증류하는 새로운 방식을 과시합니다. 이러한 성과는 경련 유형의 정확한 진단과 치료에 기여할 수 있는 가능성을 열어줍니다.



### Leveraging Generative Adversarial Networks for Addressing Data Imbalance in Financial Market Supervision (https://arxiv.org/abs/2412.15222)
- **What's New**: 이 연구는 금융 시장 감독에서 생성적 적대 신경망(Generative Adversarial Networks, GAN)의 적용을 탐구합니다. 특히, 위험 예측의 정확성을 향상시키기 위해 데이터 불균형 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: 금융 시장 데이터는 대개 불균형적이며, 시장 조작이나 시스템 리스크와 같은 고위험 사건은 빈도가 낮아 전통적인 모델이 이러한 소수 사건을 효과적으로 식별하는 데 어려움을 겪습니다. 본 연구에서는 GAN을 통해 소수 사건과 유사한 특성을 가진 합성 데이터를 생성하여 데이터셋을 균형있게 조정하는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과는 GAN이 전통적인 오버샘플링 및 언더샘플링 방법에 비해 데이터 불균형 문제를 처리하고 모델의 예측 정확성을 향상시키는 데 있어서 상당한 장점을 가지는 것을 보여줍니다. 이 방법은 미국 증권거래위원회(SEC)와 같은 금융 규제 기관에서 폭넓은 응용 가능성을 가지고 있습니다.



### Investigating the importance of social vulnerability in opioid-related mortality across the United States (https://arxiv.org/abs/2412.15218)
- **What's New**: 이번 연구는 미국 내 오피오이드 위기의 사회적 취약성 지표(Social Vulnerability Index, SVI) 변수들과 오피오이드 관련 사망률의 상관관계를 분석한 것입니다. 연구에서는 XGBoost와 수정된 오토인코더라는 두 가지 머신러닝 모델을 사용하여 13개의 SVI 변수가 어떻게 오피오이드 사망률을 예측하는지 조사합니다. 결과적으로 사회적 요인이 오피오이드 사용 및 사망에 미치는 중요한 역할을 강조하였습니다.

- **Technical Details**: 이 연구는 2010년부터 2022년까지의 카운티 수준 데이터에 기반하여 13개의 SVI 변수를 분석합니다. 여기에는 빈곤율, 실업률, 고등학교 졸업 여부 등 다양한 사회적 요인이 포함됩니다. 연구는 데이터 분포를 로그정규분포로 조정하고, 고온(anomalous high) 및 저온(anomalous low) 사례를 식별하여 변수 간 관계를 규명합니다.

- **Performance Highlights**: 연구에서 적용한 XGBoost 모델은 의사결정 트리 기반의 강력한 분류 및 회귀 모델로, 정보 이득(information gain)을 통해 각 특성의 중요도를 평가합니다. 반면, 수정된 오토인코더는 SHAP 값(SHAP values)을 이용하여 입력 변수들이 사망률 예측에 미치는 영향을 측정합니다. 이러한 접근법들은 오피오이드 위기를 악화시키는 핵심 사회적 요인을 이해하는 데 중요한 통찰력을 제공합니다.



### Sum-of-Squares Programming for Ma-Trudinger-Wang Regularity of Optimal Transport Maps (https://arxiv.org/abs/2412.13372)
- **What's New**: 이 논문은 MTW 텐서의 비부정성을 인증할 수 있는 새로운 계산적 접근법을 제안합니다. 기존의 방법들이 특정 최적 수송 문제에만 적용될 수 있는 반면, 제안된 방법은 일반적인 비용 함수에 대해 널리 적용됩니다. 특히, Sum-of-Squares (SOS) 프로그래밍을 사용하여 MTW 텐서의 비부정성 증명 절차를 개발하였습니다.

- **Technical Details**: 논문에서는 Monge 최적 수송 문제의 정규성(regularity) 이론을 다루고 있으며, MTW 조건을 통해 그 정규성을 분석합니다. MTW 텐서는 주어진 지상 비용 함수의 곡률(curvature) 개념을 제공하며, 이 텐서의 비부정성은 Monge OT 맵의 연속성을 확립하는 데 중요한 역할을 합니다. 제안된 계산적 접근법은 특히 비상 수학적 함수(semialgebraic function)에 대해 적용 가능합니다.

- **Performance Highlights**: 제안된 SOS 프로그래밍 방법은 여러 실제 지상 비용 함수에 적용되어 최적 수송 맵의 정규성 영역을 근사하는 데 성공했습니다. 이를 통해 MTW 텐서의 비부정성을 검증할 수 있으며, 이는 기계 학습 분야와 최적화 연구 커뮤니티에 큰 기여를 할 것으로 기대됩니다. 새로운 인증 절차와 정규성 이론의 발전은 이 분야의 문제 해결에 중요한 역할을 할 것입니다.



### Improving the performance of weak supervision searches using data augmentation (https://arxiv.org/abs/2412.00198)
- **What's New**: 이번 연구에서는 약한 감독 학습(weak supervision)의 한계를 극복하기 위해 데이터 증강(data augmentation) 기술을 활용하여 훈련 데이터의 크기와 다양성을 증가시키는 방법을 제안합니다. 데이터 증강 접근 방식은 p_{	ext{T}} 스미어링(p_{	ext{T}} smearing)과 제트 회전(jet rotation)과 같은 물리학에 영감을 받은 방법들을 포함합니다. 이 연구 결과는 데이터 증강이 약한 감독 학습의 성능을 획기적으로 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 약한 감독 학습은 완전 감독 학습(fully supervised learning)과 비감독 학습(unsupervised learning)의 장점을 통합하여 신호(signal)와 배경(background) 데이터로부터 효과적으로 학습하는 방법론입니다. 이 연구에서는 Classification Without Labels (CWoLa)라는 기법을 사용하여 혼합 데이터셋을 통해 신호-배경 분류기를 훈련합니다. kinematic 변수(kinematic variables)를 사용하여 신호와 사이드밴드(sideband) 영역을 정의하고 다양한 신호-배경 비율을 가진 두 개의 혼합 데이터셋을 준비합니다.

- **Performance Highlights**: 데이터 증강 기술을 적용한 결과, 최소한 반 정도로 학습 임계값(learning threshold)을 낮출 수 있었으며, 신경망은 신호에 대해 더 높은 민감도를 보였습니다. 두 가지 데이터 증강 방법을 결합함으로써 신경망은 개별 방법을 사용할 때보다 더 우수한 성능을 발휘했습니다. 신경망의 다양한 증강 샘플 크기에서의 동작을 연구하고 이들의 점근적 행동(asymptotic behaviors)을 분석한 결과도 포함되어 있습니다.




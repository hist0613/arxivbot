New uploads on arXiv(cs.CL)

### Domain-specific long text classification from sparse relevant information (https://arxiv.org/abs/2408.13253)
Comments:
          Submitted to conference ECAI 2024: 27TH European Conference on Artificial Intelligence

- **What's New**: 본 논문에서는 의학 보고서에서의 중요한 정보 추출을 위해 하이레르키 모델을 제안합니다. 이 모델은 기술 용어에서 몇 가지 잠재 타겟 용어를 집계하여 해당 단어를 포함한 문장을 검색하는 구조를 가지고 있어, 적은 양의 관련 정보가 있을 때의 문제를 해결할 수 있습니다.

- **Technical Details**: 하이레르키 모델은 BERT 임베딩을 바탕으로 잠재적 타겟 용어를 포함하는 필터링된 문장에서 추출하며, 이 타겟 용어들의 어텐션 가중치 평균을 내어 문서 임베딩을 생성하고 이를 분류하는 방식으로 구성됩니다. 이 과정에서 최대 30개의 타겟 용어 리스트를 기반으로 합니다.

- **Performance Highlights**: 제안된 모델은 기존의 대규모 언어 모델에 비해 의료 분야에서의 특화된 문서 검색에 있어서 더 나은 성능을 보이는 것으로 평가되었습니다. Public English benchmark와 private French medical dataset에서의 평가를 통해 기존의 최고 성능 모델과 비교해 robustness를 보여주었습니다.



### Which Prosodic Features Matter Most for Pragmatics? (https://arxiv.org/abs/2408.13240)
Comments:
          Submitted to ICASSP 2025. Audio illustrations available at this https URL

- **What's New**: 이번 연구에서는 발화 쌍 간의 화용론적 유사성을 예측할 때 가장 중요한 prosodic (음조) 특성들을 조사했습니다.

- **Technical Details**: 연구에서는 우선적으로 duration (지속시간) 관련 특성이 pitch (음높이) 관련 특성보다 더 중요함을 발견했으며, 발화의 초기 특성이 발화의 마지막 특성보다 중요한 역할을 한다는 것을 알게 되었습니다. 여러 neglected (간과된) 음향 및 prosodic 특성들도 화용론적으로 중요함을 시사하였습니다.

- **Performance Highlights**: 이 연구는 prosody (음조)의 중요성을 강조하며, speech synthesis (음성 합성) 평가를 개선하려는 방향을 제시합니다. 제안된 방법은 future basic research (기본 연구)의 방향성을 제공하고, 사용자 맞춤형 음성 합성기 출력의 개선을 위한 제어 매개변수를 설계하는 데 활용될 수 있습니다.



### Instruct-DeBERTa: A Hybrid Approach for Aspect-based Sentiment Analysis on Textual Reviews (https://arxiv.org/abs/2408.13202)
- **What's New**: 이번 논문은 Aspect-based Sentiment Analysis (ABSA) 방법론의 발전과 최신 Transformer 기반 모델을 활용한 하이브리드 모델 Instruct - DeBERTa를 제안합니다. 이 모델은 ATE와 ASC의 통합된 프레임워크를 제공하고, 기존 방법론의 한계를 극복하는 데 중점을 둡니다.

- **Technical Details**: ABSA는 텍스트에서 특정 측면에 대한 감정을 추출하는 기술로, 전통적인 감성 분석 방법은 종종 특정 제품의 특징에 대한 암묵적인 의견을 놓치곤 했습니다. 이 논문은 BERT, RoBERTa 및 DeBERTa와 같은 최신 Transformer 모델을 활용하여 ATE 및 ASC를 수행하는 하이브리드 모델을 개발하였으며, 이를 통해 더 나은 정확도와 신뢰성을 추구합니다.

- **Performance Highlights**: 제안된 하이브리드 모델 Instruct - DeBERTa는 SemEval restaurant 2014 및 SemEval laptop 2014 데이터셋에서 ATE 및 ASC의 통합 작업에 대해 최고 성능을 기록했습니다. 실험 결과, 모델은 모든 실험한 도메인에서 감성 분석의 정확성과 신뢰성을 획기적으로 향상시켰음을 보여줍니다.



### Can LLM be a Good Path Planner based on Prompt Engineering? Mitigating the Hallucination for Path Planning (https://arxiv.org/abs/2408.13184)
Comments:
          Submitted to ICASSP

- **What's New**: 이 연구는 Spatial-to-Relational Transformation and Curriculum Q-Learning (S2RCQL) 모델을 제안하여 LLM의 공통적인 문제인 공간적 환각(spatial hallucination) 및 맥락 불일치 환각(context inconsistency hallucination)을 해결하고자 합니다.

- **Technical Details**: S2RCQL은 세 가지 주요 구성 요소로 구성됩니다: 환경(environment), 에이전트(agent), 그리고 과정 모듈(course module)입니다. 공간적 프롬프트를 엔터티 관계(entity relations) 및 관계 체인(relation chains)으로 변환하는 Spatial-to-Relational 접근 방식을 사용합니다. 또한, Q-learning을 기반으로 한 경로 계획(Path-Planning) 알고리즘을 통해 LLM의 공통적인 환각 문제를 완화합니다.

- **Performance Highlights**: S2RCQL은 Baidu의 ERNIE-Bot 4.0을 기반으로 한 실험에서 성공률과 최적률이 23%에서 40% 개선된 결과를 보였습니다. 이전의 첨단 프롬프트 엔지니어링 대비 성능이 크게 향상되었음을 보여줍니다.



### Lessons in co-creation: the inconvenient truths of inclusive sign language technology developmen (https://arxiv.org/abs/2408.13171)
- **What's New**: 이 논문은 수화 기술 개발에 있어 청각 장애인 커뮤니티의 참여와 리더십 필요성을 강조합니다. 특히 유럽 청각 장애인 연합(EUD)의 경험을 바탕으로 두 개의 EU Horizon 2020 프로젝트(EASIER, SignON)에서의 내용과 문제점을 분석합니다.

- **Technical Details**: 논문은 비공식적인 참가자 관찰 데이터를 통해 진행되었습니다. EASIER와 SignON 프로젝트에서는 수화와 음성 언어 간의 모바일 번역 애플리케이션 개발이 목표로 하였으며, 대체로 청각인 비전문가들과 청각 장애인 사용자들을 대표하는 조직들이 협력했습니다. 그러나 공동 창작(co-creation) 과정에서는 권력 불균형과 토큰주의(tokenism) 문제가 종종 드러납니다.

- **Performance Highlights**: 논문은 공동 창작이 진정한 변화를 이끌 수 있는 활동임을 주장하며, 청각 장애인 연구자 수 증가와 AI 리터러시(AI literacy) 향상이 필수적임을 강조합니다. 이 연구는 향후 공동 창작 프로젝트를 위한 일곱 가지 교훈을 제안하며, 청각 장애인을 진정한 파트너로 인정할 것을 촉구합니다.



### Coarse-to-fine Alignment Makes Better Speech-image Retrieva (https://arxiv.org/abs/2408.13119)
- **What's New**: 본 논문에서는 음성-이미지 검색을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 음성-이미지 대비(Speech-Image Contrastive, SIC) 학습과 음성-이미지 일치(Speech-Image Matching, SIM) 학습 작업을 결합하여 음성과 이미지 표현을 정교하게 정렬합니다.

- **Technical Details**: 우리의 프레임워크는 멀티태스크 학습과 효과적인 학습 기법을 활용하여 거친 수준에서 세밀한 음성-이미지 정렬을 달성합니다. SIC 학습 중 고품질의 다양한 Negative representation을 효율적으로 샘플링하기 위해 임베딩 큐(embedding queue)를 사용하며, 모멘텀 증류(momentum distillation)를 적용하여 학습 과정을 최적화합니다.

- **Performance Highlights**: 제안된 프레임워크는 Flickr Audio 및 SpokenCOCO의 벤치마크 데이터셋에서 R@1에서 기존의 최첨단 방법보다 4% 이상 뛰어난 성능을 보여주었으며, 제로샷(zero-shot) 실험에서도 뛰어난 일반화 능력을 입증하였습니다.



### Analysis of child development facts and myths using text mining techniques and classification models (https://arxiv.org/abs/2408.13091)
Comments:
          17 pages

- **What's New**: 이번 연구는 아동 발달에 대한 신화(myth)와 사실(fact)을 구분하기 위한 텍스트 마이닝(text mining) 기법과 분류 모델(classification model)을 적용하여 인터넷에서의 잘못된 정보(misinformation) 문제를 해결하고자 합니다.

- **Technical Details**: 연구 방법론은 여러 단계를 포함합니다. 첫째, 데이터 전처리(pre-process)를 통해 정확성을 높인 후, 6개의 강력한 머신러닝(Machine Learning, ML) 분류기와 1개의 딥러닝(Deep Learning, DL) 모델을 사용하여 구조화된 데이터를 분석했습니다. 두 가지 피쳐 추출 기법을 활용해 각기 다른 학습-테스트 분할(training-testing splits)에서 성능을 평가했으며, k-fold와 leave-one-out 방법을 통해 교차 검증(cross-validation)을 실시했습니다.

- **Performance Highlights**: 테스트한 분류 모델 중 로지스틱 회귀(Logistic Regression, LR)가 가장 높은 정확도를 기록하였고, Bag-of-Words(BöW) 피쳐 추출 기법을 사용하여 90%의 정확도를 달성했습니다. LR은 빠른 속도와 효율성을 자랑하며, 각 문장에 대한 테스트 시간을 0.97 마이크로초로 유지했습니다. 이러한 발견은 LR과 BoW의 조합이 아동 발달 정보의 정확한 분류에 효과적임을 나타냅니다.



### In-Context Learning with Reinforcement Learning for Incomplete Utterance Rewriting (https://arxiv.org/abs/2408.13028)
- **What's New**: 본 연구에서는 In-context learning (ICL)을 위한 예제 선택에서 정책 기반 강화 학습(framework based reinforcement learning)을 도입하여 LLM이 제공하는 피드백을 활용합니다. 기존 방법들이 LLM의 직접적인 피드백을 활용하지 못하는 문제점을 개선하였습니다.

- **Technical Details**: 제안된 방법은 크게 언어 모델(selector를 포함한 LM)과 LLM generator로 구성됩니다. LM 선택기는 후보 예제를 밀집 표현(dense representation)으로 인코딩하고, LLM의 성능을 기반으로 보상(reward)과 정책 기울기(policy gradient)를 계산하여 최적화합니다. 다양한 데이터 세트에 대한 실험을 통해 기존의 예제 선택 방법보다 월등한 성과를 나타냅니다.

- **Performance Highlights**: 제안된 방법은 CANARD 및 REWRITE 데이터 세트에서 기존 방법보다 약 2점, TASK 데이터 세트에서는 10점 높은 성능을 기록했습니다. 특히 기존의 감독 학습(supervised finetuning) 모델들에 비해 few-shot 설정에서 더욱 유리한 성과를 보였습니다.



### Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates (https://arxiv.org/abs/2408.13006)
Comments:
          Preprint, under review. 17 pages, 7 figures, 16 tables

- **What's New**: 본 논문에서는 LLM(judge) 평가의 신뢰성과 인간 선호와의 정렬을 체계적으로 평가하는 새로운 방법론을 제시합니다. 특히, 인간과의 평가 비교를 위한 새로운 메트릭스를 정의하고, 다양한 prompt template의 영향을 분석합니다.

- **Technical Details**: 연구팀은 LLM 라인의 신뢰성을 평가하기 위해 LLM judge 구조와 내부 불일치성(flipping noise)을 측정하는 프레임워크를 개발했습니다. 이 과정에서 RLHF(딥 러닝 강화 학습), DPO(데이터 개인화 최적화)와 같은 정렬 기술 및 position bias와 length bias의 개념을 사용합니다.

- **Performance Highlights**: 실험 결과, LLM judge의 성능은 prompt template에 따라 크게 달라지며, tested LLM judge와 인간 평가자 간의 정렬 수준이 중간 정도임을 보여주었습니다. 이는 다양한 LLM과 prompt template를 비교 검토할 필요성을 강조합니다.



### MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries (https://arxiv.org/abs/2408.12980)
Comments:
          In Findings of the Association for Computational Linguistics ACL 2024

- **What's New**: 이 논문에서는 "MedDec"라는 새로운 데이터세트를 개발하였으며, 이 데이터세트는 11개의 다른 질병 유형과 10가지의 의료 결정으로 주석이 달린 임상 노트를 포함합니다. 이를 통해 임상 노트 내에서 다양한 의료 결정을 추출하고 분류하는 새로운 작업을 소개합니다.

- **Technical Details**: MedDec 데이터세트는 451개의 퇴원 요약에 대한 의료 결정을 주석 처리하였으며, 총 54,000개 이상의 문장을 포함하고 있습니다. 제안된 방법은 두 가지 주요 구성요소를 기반으로 하여 의료 결정을 추출하고 분류하기 위해 긴 임상 노트를 세그먼트로 나누어 모델에 입력합니다. 각 세그먼트는 숨겨진 표현과 토큰 분류 확률을 생성합니다.

- **Performance Highlights**: MedDec의 채택으로 의료 결정의 추출 및 분류 작업이 가능해졌습니다. 논문에서는 여러 기본선 모델을 개발하고, 기존의 span detection 접근 방식을 평가하여 높은 성능을 보였음을 보여주며, 이는 향후 연구를 위한 기초를 제공합니다.



### Internal and External Knowledge Interactive Refinement Framework for Knowledge-Intensive Question Answering (https://arxiv.org/abs/2408.12979)
- **What's New**: 최근 연구는 LLM(대형 언어 모델)의 내부 지식을 활용하여 외부 지식 검색을 개선하는 새로운 패러다임, IEKR(Internal and External Knowledge Refinement)를 제안합니다. 이를 통해 필요한 외부 지식을 효과적으로 검색하고 생성된 내부 지식의 오류를 보완할 수 있습니다.

- **Technical Details**: IEKR는 LLM의 내부 지식을 활용하여 외부 지식 기반(KB)에서 관련 지식을 검색하고, 그 외부 지식을 통해 내부 지식을 보완합니다. 실험은 3개의 벤치마크 데이터셋(OpenbookQA, CommonsenseQA, MedQA)에서 수행되었으며, 다양한 도메인과 LLM을 활용해 최신 성능을 달성했습니다.

- **Performance Highlights**: 이 연구의 결과는 내부 지식을 통해 필요한 외부 지식을 효과적으로 검색할 수 있음을 보여주며, 실험을 통해 우리 접근 방식의 다양한 모듈의 효과를 입증하였습니다. 이로 인해 QA(질문 응답) 태스크에서 새로운 최첨단 성능을 달성하였습니다.



### Open Llama2 Model for the Lithuanian Languag (https://arxiv.org/abs/2408.12963)
Comments:
          12 pages, 8 figures, 5 tables

- **What's New**: 이번 논문에서는 리투아니아어를 위한 최초의 오픈 Llama2 대규모 언어 모델(LLM)과 이에 수반되는 질문/답변(Q/A) 데이터셋 및 유명 LLM 벤치마크의 번역을 제안하고 설명합니다. 또한 오픈 지역 LLM의 전반적인 역할과 제안된 LLM의 훈련 과정을 설명합니다.

- **Technical Details**: 제안된 모델은 Llama2 아키텍처를 기반으로 하며, 7B (억 단위) 및 13B 파라미터를 갖고 있습니다. 훈련은 두 단계로 나뉘며, 첫 번째 단계는 자가 회귀적인 사전 훈련을 포함하고, 두 번째 단계는 인간-주석 데이터와 RLHF(Reinforcement Learning with Human Feedback) 방법론을 사용하여 수행 됩니다. 훈련 과정은 최소 2048 토큰의 컨텍스트 길이를 지원하며, 8xH100 GPU에서 진행되었습니다.

- **Performance Highlights**: 제안된 LLM의 평가를 통해 얻은 perplexity 수치는 다른 현대 오픈 LLM들에 비해 경쟁력을 갖추고 있음을 보여줍니다. 또한, 고품질 사전 훈련 데이터셋이 이러한 벤치마크에서 효율적으로 성능을 발휘하는 모델을 얻기 위해 필수적임을 확인했습니다.



### Multimodal Contrastive In-Context Learning (https://arxiv.org/abs/2408.12959)
- **What's New**: 본 논문에서는 멀티모달(multi-modal) 대비(in-context) 학습(framework) 새로운 해석을 제안하여 LLM의 ICL(inner workings)에 대한 이해를 향상시키고자 한다.

- **Technical Details**: 논문에서는 대비 학습(contrastive learning)을 사용한 ICL 해석을 제안하며, 멀티모달 입력 형식의 편향(bias)을 해결하기 위한 분석 프레임워크를 개발하였다. 또한, Anchored-by-Text ICL 전략을 통해 자원 제약 환경에서 효과적으로 증오(memes) 탐지를 수행할 수 있음을 보여준다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 다양한 시나리오에서 ICL 성능을 현저히 향상시키며, 특히 자원이 제한된 환경 및 도전적인 작업(task)에서 강력한 성능을 발휘함을 증명하였다.



### Causal-Guided Active Learning for Debiasing Large Language Models (https://arxiv.org/abs/2408.12942)
Comments:
          ACL main conference

- **What's New**: 최근의 분석에 따르면, 현재의 Generative Large Language Models (LLMs)는 데이터셋의 편향을 포착하고 그렇게 생성된 결과물의 일반화 가능성과 유해성을 저하시킬 수 있습니다. 본 연구에서는 Active Learning과 인과 메커니즘을 결합한 Casual-guided Active Learning (CAL) 프레임워크를 제안합니다.

- **Technical Details**: CAL 프레임워크는 LLMs가 스스로 편향된 샘플을 자동으로 식별하고, 이러한 편향 패턴을 유도하는 방식으로 작동합니다. 이를 통해, LLMs는 생성 과정에서 데이터셋의 편향을 활용하지 않도록 방지하는 낮은 비용의 In-Context Learning (ICL) 방법을 적용합니다.

- **Performance Highlights**: 실험 결과, CAL을 통해 LLMs는 전형적인 편향 인스턴스를 효과적으로 인식할 수 있으며, 다양한 편향 패턴을 유도하여 LLMs의 일반화 가능성과 안전성을 향상시킬 수 있음을 보여주었습니다.



### Multi-Faceted Question Complexity Estimation Targeting Topic Domain-Specificity (https://arxiv.org/abs/2408.12850)
Comments:
          14 pages, 6 figures

- **What's New**: 이 논문은 전통적인 질문 난이도 추정 방식을 넘어, 다양한 요소가 결합된 질문 복잡성에 대한 새로운 프레임워크를 제시합니다. 이는 NLP 기술과 지식 그래프 분석을 활용하여 도메인 특정 질문 난이도 추정을 수행합니다.

- **Technical Details**: 연구에서는 Topic Retrieval Cost, Topic Salience, Topic Coherence, Topic Superficiality의 네 가지 주요 매개변수를 도입하여 특정 주제의 질문 복잡성을 포착합니다. 이들 매개변수는 주제 모델링(topic modelling), 지식 그래프 분석(knowledge graph analysis), 정보 검색(information retrieval) 기술을 통해 운영화됩니다.

- **Performance Highlights**: 제안된 모델은 이러한 특징으로 훈련되어 질문 난이도를 예측하는 데 효능을 보여줍니다. 이렇게 매개변수를 운영화함으로써, 이 프레임워크는 효과적인 질문 생성, 평가 설계 및 다양한 학문 분야의 적응형 학습 시스템을 위한 새로운 접근 방식을 제공합니다.



### CLLMFS: A Contrastive Learning enhanced Large Language Model Framework for Few-Shot Named Entity Recognition (https://arxiv.org/abs/2408.12834)
Comments:
          27TH EUROPEAN CONFERENCE ON ARTIFICIAL INTELLIGENCE

- **What's New**: 본 논문에서는 CLLMFS라는 Contrastive Learning이 강화된 대형 언어 모델(LLM) 프레임워크를 제안합니다. 이 프레임워크는 적은 양의 라벨 데이터로도 Few-Shot Named Entity Recognition (NER)에서의 성능을 향상시키기 위해 설계되었습니다.

- **Technical Details**: CLLMFS는 Low-Rank Adaptation (LoRA)와 대비 학습(Contrastive Learning) 메커니즘을 통합하여 LLM의 내부 표현을 활용합니다. 이 모델은 엔티티 경계 인식을 향상시키고, 엔티티 인식 정확성을 높이는 데 집중하고 있습니다.

- **Performance Highlights**: CLLMFS는 기존의 최우수 방법들에 비해 F1-score에서 2.58%에서 97.74%까지 성능 향상을 달성하며, 다양한 데이터셋에서 교차 도메인 NER 실험을 통해 우수한 일반화 성능을 입증하였습니다.



### LIMP: Large Language Model Enhanced Intent-aware Mobility Prediction (https://arxiv.org/abs/2408.12832)
Comments:
          13 pages

- **What's New**: 본 연구는 LIMP(LLMs for Intent-ware Mobility Prediction)이라는 새로운 프레임워크를 제안하며, 이 프레임워크는 대형 언어 모델(LLMs)의 상식적 추론 능력을 활용하여 인간 이동성의 의도 예측을 개선합니다.

- **Technical Details**: LIMP는 'Analyze-Abstract-Infer'(A2I)라는 에이전틱 워크플로우를 도입하여 인간의 이동성을 분석하고 의도를 추론합니다. 이 시스템은 대형 LLM에서 소규모 오픈소스 모델로의 효율적인 파인튜닝 방식으로 확장성을 보장합니다.

- **Performance Highlights**: LIMP는 두 개의 실제 데이터셋에서 평가되어 기존 모델보다 6.64%에서 9.52% 높은 다음 위치 예측 정확도를 달성하며, A2I 워크플로우를 통해 LLM의 이동 의도 추론 정확도를 16.28% 향상시킵니다.



### Grounding Fallacies Misrepresenting Scientific Publications in Evidenc (https://arxiv.org/abs/2408.12812)
- **What's New**: MissciPlus라는 새로운 데이터셋이 도입되어, 잘못된 주장을 지지하는 실제 출처의 미스프레젠테이션된 증거를 결합하여 논리적 오류를 탐지하는 데 사용됩니다. 이 데이터셋은 기존의 Missci에서 확장된 것으로, 허위 주장을 식별하는 데 유용한 새로운 작업 정의와 함께 제시됩니다.

- **Technical Details**: MissciPlus는 2,257개의 인간 주석 링크를 포함하여, 과학적 연구에서 잘못 진술된 주장을 뒷받침하는 출처와 증거를 연결합니다. 또한, 이 데이터셋은 사실 확인(fact-checking) 모델의 입력 구조와 동일므로, 사실에 기반한 자동화된 사실 확인 모델을 평가하는 데 적합합니다.

- **Performance Highlights**: 현재의 사실 검토 모델들은 잘못된 정보에 대한 반박을 위해 관련 출처를 효과적으로 사용하지 못합니다. 또한, 이러한 출처들이 언어 모델(LLMs)로 하여금 잘못된 주장을 진실로 받아들이도록 오도할 수 있다는 점이 발견되었습니다.



### Less for More: Enhancing Preference Learning in Generative Language Models with Automated Self-Curation of Training Corpora (https://arxiv.org/abs/2408.12799)
- **What's New**: 이 논문에서는 언어 모델의 성능 향상을 위해 선호 학습에서 발생하는 주석 불일치 문제를 해결하기 위해 자기 선별 방법을 제안합니다. 이 방법은 주석된 데이터셋을 자동으로 정리하여 모호한 주석을 제거하는 데 중점을 둡니다.

- **Technical Details**: 제안된 자기 선별 방법은 사전 훈련된 체크포인트로 초기화된 프록시 모델(proxymodel)을 사용하여 주석의 일관성을 평가합니다. 이러한 프록시 모델은 주어진 선호 데이터셋을 사용하여 순위 손실(ranking loss)을 통해 훈련됩니다. 이 과정에서 모호한 주석과 관련된 데이터 포인트를 자동으로 식별하고 배제합니다.

- **Performance Highlights**: 제조된 자기 선별 데이터셋을 사용한 훈련 결과, 다양한 지시사항 수행 작업에서 성능이 현저하게 개선되었습니다. 실험에서 제안된 방법이 선호 학습 성능을 크게 향상시킨다는 점을 입증하였습니다.



### Quality or Quantity? On Data Scale and Diversity in Adapting Large Language Models for Low-Resource Translation (https://arxiv.org/abs/2408.12780)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 연구는 Low-Resource Language (LRL) 번역을 위해 Large Language Models (LLMs)를 적응시키는 방법을 탐구합니다. 특히, parallel data의 중요성과 Supervised Fine-Tuning (SFT)에서의 다양성 간의 관계를 재검토합니다.

- **Technical Details**: 연구에서 LLMs는 3개의 모델(Gemma 2B, Mistral 7B, Llama 3 8B)을 사용하여 11개의 토착 아메리카 언어와 4개의 북동인도 언어를 대상으로 실험이 진행되었습니다. 결과적으로, LRLs의 경우 CPT와 SFT 단계에서 parallel data의 규모가 매우 중요하며, SFT 동안의 다양성은 오히려 간섭을 유발하는 경향이 있습니다.

- **Performance Highlights**: 2단계 훈련 방식을 적용하여 LLMs의 평균 chrF++ 성능이 +16.5 향상되었습니다. LRL를 위한 최적의 전략은 더 많은 에폭으로 집중적으로 다국어 MT fine-tuning을 하는 것입니다.



### Investigating LLM Applications in E-Commerc (https://arxiv.org/abs/2408.12779)
- **What's New**: 이 논문은 e-commerce 분야에서 Large Language Models (LLMs)의 효능을 조사하였으며, 여러 규모의 공개 e-commerce 데이터셋을 활용해 instruction-tuning한 LLM 모델의 성능을 전통적인 모델과 비교하였습니다.

- **Technical Details**: LLMs의 fine-tuning을 위한 다양한 방법론, 특히 Low-Rank Adaptation (LoRA), 단일 작업 훈련, 혼합 작업 훈련을 다루었으며, e-commerce 특정 작업에 대한 평가를 위한 데이터셋을 구성하였습니다. 또한, few-shot inference와 같은 방법의 효과를 분석하였습니다.

- **Performance Highlights**: 전통적인 산업 모델인 BERT 및 T5와의 비교실험 결과, 특정 e-commerce 작업에서 전통적인 방법으로 훈련한 소형 언어 모델이 매우 큰 LLM보다 더 나은 성능을 보일 수 있음을 발견했습니다.



### SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection (https://arxiv.org/abs/2408.12748)
Comments:
          preprint under review

- **What's New**: 이번 연구에서는 Small Language Model (SLM) 분류기를 이용한 초기 탐지와 Large Language Model (LLM)을 이용한 제약된 추론자(constrained reasoner)를 통한 설명 생성이 결합된 새로운 프레임워크를 제안합니다. 이를 통해 실시간으로 해석 가능한 환각 탐지가 최적화되었습니다.

- **Technical Details**: 제안된 두 단계 프레임워크는 먼저 SLM을 사용하여 환각을 탐지하고, 탐지된 환각에 대한 설명을 LLM 기반의 제약된 추론자가 제공합니다. 이 과정에서 SLM과 LLM 간의 결정 및 설명의 일관성을 고려하여, 탐지 및 설명 간의 정렬을 개선하는 다양한 접근 방식을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 여러 오픈 소스 데이터세트에서 효과성이 입증되었으며, 사용자 경험을 향상시키는 데 기여했습니다. 특히, LLM의 설명 생성에 필요한 지연 시간 문제를 해결하는 데 효과적입니다.



### Macro-Queries: An Exploration into Guided Chart Generation from High Level Prompts (https://arxiv.org/abs/2408.12726)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)과 데이터 시각화의 교차점을 탐구합니다. 특히, 초보 사용자들이 다양한 데이터 시각화 유형에 접근할 수 있도록 돕기 위해, 고수준 사용자 질문(매크로 쿼리라고 부름)을 기반으로 데이터를 변환하여 유용한 시각화를 생성하는 LLM 기반 파이프라인을 제시합니다.

- **Technical Details**: 이 연구에서는 매크로 쿼리라는 개념을 도입하였으며, 이는 사용자 제공 시트 데이터와 일치하는 맥락적으로 정확한 응답을 유도하는 데 중요한 역할을 합니다. LLM 파이프라인을 통해 사용자의 복잡한 쿼리를 분해하고, SQL 도구를 활용하여 데이터 변환을 수행합니다. 또한, Andrew Abela의 차트 분류법(Chart Taxonomy)을 바탕으로 다양한 차트를 선택합니다.

- **Performance Highlights**: 최신 아키텍처는 텍스트-투-SQL 변환 모듈을 포함하고 있으며, 여러 차트 생성을 위한 일관된 디자인을 유지합니다. 프로토타입 테스트 결과, 개선된 정확성과 함께 더 넓은 범위의 차트를 생성할 수 있는 가능성을 보여 주었습니다. 이러한 시스템은 초보자들이 매크로 쿼리로부터 의미 있는 시각화를 자동으로 생성하도록 돕는 데 중점을 두고 있습니다.



### Towards Estimating Personal Values in Song Lyrics (https://arxiv.org/abs/2408.12694)
- **What's New**: 이 연구는 노래 가사에서 개인 가치를 자동으로 추정하려는 첫 번째 시도를 다루고 있으며, 이를 통해 음악 정보 검색(MIR) 과업에서 개인화( personalization)에 기여할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구팀은 미국 대중을 대상으로 한 표본에서 360개의 노래 가사를 샘플링하고, 심리학적 설문조사를 통해 각 노래에 대해 평균 27개의 평가를 수집했습니다. 또한, 주관적인 텍스트인 노래 가사의 가치 평가를 위해 Robust Ranking Aggregation(RRA)을 적용하여 가치를 순위 목록으로 다룹니다. 가사와 검증된 가치 사전 사이의 의미적 유사성을 측정함으로써 단어 임베딩(word embedding) 모델에서의 추정도 비교했습니다.

- **Performance Highlights**: 초기 결과들은 주어진 노래 가사에서 자동 추정한 가치 목록이 사람의 평가와 중간 정도의 상관관계를 가진다는 것을 보여주었습니다. 이러한 결과는 향후 MIR 작업에서 노래 가사의 감정적 반응 및 개인적 가치의 중요성을 강조하고 있습니다.



### Data Exposure from LLM Apps: An In-depth Investigation of OpenAI's GPTs (https://arxiv.org/abs/2408.13247)
- **What's New**: LLM(app) 생태계가 빠르게 성숙해지고 있으며, 다양한 사용 사례를 지원하기 위해 과도한 사용자 데이터를 수집하고 있음을 강조합니다. OpenAI의 GPT 생태계를 사례로 분석하며, LLM 기반 프레임워크를 개발하여 데이터 수집 관행을 투명하게 밝히려는 목표를 가지고 있습니다.

- **Technical Details**: OpenAI의 GPT 생태계에서 데이터 수집 관행을 정적 분석(Static Analysis)으로 분석하였습니다. 행동(Action)과 GPT들 간의 그래프 모델을 통해 간접 데이터 노출을 연구하였으며, LLM 기반의 프라이버시 정책 분석 프레임워크를 통해 데이터 수집과 프라이버시 정책의 일관성을 자동으로 점검합니다.

- **Performance Highlights**: 119,274개의 GPT와 2,596개의 유니크 Action을 분석한 결과, 82.9%의 Action이 외부 서비스에서 비롯되었으며, 일부 Action은 여러 GPT에 내장되어 사용자 활동을 추적할 수 있었음을 발견하였습니다. 적어도 9.5배 더 많은 데이터가 Action의 공존을 통해 노출되며, 수집된 데이터의 대부분이 프라이버시 정책에서 누락되어 있음을 확인하였습니다.



### Multi-Layer Transformers Gradient Can be Approximated in Almost Linear Tim (https://arxiv.org/abs/2408.13233)
- **What's New**: 이번 논문에서는 phổ biến한 transformer 구조의 self-attention 메커니즘에서 발생하는 이차적인 계산 복잡도를 해결하기 위해 새로운 빠른 gradient 계산 방법을 도입하였습니다.

- **Technical Details**: 우리의 접근 방식은 multi-layer transformer 모델의 gradient를 거의 선형 시간(n^{1+o(1)})으로 계산할 수 있게 해줍니다. 이는 기존의 O(n2d) 계산 복잡도를 대폭 줄여줍니다. 이 방법은 모든 loss function에 대해 유효하며, 전체 모델에서 제한된 근사 오차를 유지합니다.

- **Performance Highlights**: 이번 연구는 대규모 언어 모델의 gradient 계산 효율성을 개선하여 긴 컨텍스트의 언어 모델 훈련 및 배치를 보다 효과적으로 할 수 있기를 기대합니다.



### Enhancing Few-Shot Transfer Learning with Optimized Multi-Task Prompt Tuning through Modular Prompt Composition (https://arxiv.org/abs/2408.13227)
- **What's New**: 이 논문에서는 여러 과제 간의 지식 전이를 촉진하여 성능을 향상시키기 위해 다중 작업 프롬프트 튜닝(multi-task prompt tuning) 접근법을 제안합니다. 특히 ComPT라는 프레임워크를 도입하여 공유 프롬프트와 과제별 프롬프트를 결합하여 각 과제에 대한 목표 프롬프트를 생성합니다.

- **Technical Details**: 제안된 ComPT 접근 방식에서는 N개의 과제를 다루며, M개의 소스 프롬프트를 사용하여 각 개별 목표 과제에 대한 타겟 프롬프트를 생성합니다. 소스 프롬프트는 공유되고, 이를 기반으로 각 과제에 전용 프롬프트(프라이빗 프롬프트)와 결합하여 최종 타겟 프롬프트를 형성합니다. 이 과정에서는 학습 가능한 주의(attention) 모듈을 통해 소스 프롬프트를 활성화하고 결합합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 프롬프트 튜닝 방식과 비교해 정확성과 견고성을 개선했으며, 특히 few-shot 설정에서 다른 방법들보다 우수한 성과를 냈습니다. GLUE 벤치마크를 포함하여 다양한 작업에서 탁월한 성능을 입증했습니다.



### EUR-USD Exchange Rate Forecasting Based on Information Fusion with Large Language Models and Deep Learning Methods (https://arxiv.org/abs/2408.13214)
- **What's New**: 이 논문에서는 유럽연합 유로(EUR)와 미국 달러(USD) 환율 예측을 위한 새로운 IUS 프레임워크를 제안합니다. 이 프레임워크는 뉴스 및 분석에서 수집된 비구조적 텍스트 데이터와 환율 및 금융 지표에 대한 구조적 데이터를 통합하여 예측의 정확성을 높입니다.

- **Technical Details**: IUS 프레임워크는 감정 극성 점수(sentiment polarity scoring)와 텍스트의 환율 움직임 분류를 위해 대형 언어 모델(large language models)을 사용합니다. 작성된 특성은 인과관계 기반 특징 생성기(Causality-Driven Feature Generator)에 입력되어 Optuna로 최적화된 Bi-LSTM 모델을 통해 환율을 예측합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기준 모델보다 MAE를 10.69% 줄이고 RMSE를 9.56% 감소시켰습니다. 비구조적 데이터와 구조적 데이터를 결합한 데이터 융합(data fusion)의 이점도 확인해, 이러한 조합이 단독으로 구조적 데이터만 사용할 때보다 더 높은 예측 정확도를 제공함을 보여주었습니다.



### SpeechPrompt: Prompting Speech Language Models for Speech Processing Tasks (https://arxiv.org/abs/2408.13040)
Comments:
          Published in IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

- **What's New**: 이번 논문에서는 음성 처리 분야에서 사전 훈련된 음성 언어 모델(Speech LMs)에 대한 prompting 기법을 처음으로 탐구하였습니다. 기존의 텍스트 기반 모델에서 발전한 이 방법은 단순한 입력 변화를 통해 다양한 다운스트림 작업을 수행할 수 있도록 해줍니다.

- **Technical Details**: 작업을 음성-단위 생성(task)을 통해 통합함으로써, 음성 분류, 시퀀스 생성 및 음성 생성 작업을 통합하여 처리합니다. 본 연구는 단순한 선형 변환으로 작동하는 learnable verbalizer를 활용하여 단어의 Rich 정보를 처리하고 있으며, 사전 훈련된 모델의 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안한 prompting 방법이 self-supervised learning 모델 기반의 fine-tuning 방법과 유사한 성능을 보였으며, 특히 few-shot 학습 환경에서도 유망한 결과를 나타냈습니다. 또한, 모든 작업을 통합된 파이프라인 내에서 수행하며 파라미터 효율성도 우수합니다.



### IAA: Inner-Adaptor Architecture Empowers Frozen Large Language Model with Multimodal Capabilities (https://arxiv.org/abs/2408.12902)
- **What's New**: 다중모달 대형 언어 모델(MLLMs) 분야에서, 본 연구는 기존의 freeze 모델을 기반으로 새로운 Inner-Adaptor Architecture(IAA)를 도입하여 언어 모델의 성능 저하 없이 다중 모달 능력을 향상시키는 접근 방식을 제안합니다.

- **Technical Details**: 우리의 Inner-Adaptor Architecture는 LLM의 다양한 깊이에 다중 모달 어댑터를 포함하여, 고정된 언어 모델이 시각적 특성과 직접 상호작용하도록 합니다. 이 방법은 대규모 정렬 데이터 없이도 소규모 데이터셋에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 우리의 다중모달 접근 방식은 MME, MMBench, MMMU, RefCOCO와 같은 다양한 비전-언어 벤치마크에서 이전의 최신 기술을 초월하며, NLP 과제에서도 성능 저하 없이 우수한 결과를 보여줍니다.



### Memory-Efficient LLM Training with Online Subspace Descen (https://arxiv.org/abs/2408.12857)
Comments:
          Code is available at this https URL

- **What's New**: 최근 메모리 효율적인 LLM(대형 언어 모델) 훈련 알고리즘들이 크게 인기를 얻고 있다. 이 방법들은 기울기의 저랭크(低秩, low-rank) 구조를 활용하여 SVD(특이값 분해, Singular Value Decomposition)를 통해 옵티마이저의 상태를 부분 공간으로 매핑한다. 본 연구는 어떤 업데이트 규칙에 대해서도 수렴 보장을 제공하는 최초의 연구이다.

- **Technical Details**: 본 연구에서는 임의의 업데이트 규칙에 대한 수렴 보장을 첫 번째로 제공한다. 이는 해밀토니안 내림차순(Hamiltonian Descent) 프레임워크로 분석할 수 있는 다양한 옵티마이저에 일반적으로 적용 가능하다. 우리의 접근 방식은 정기적인 업데이트를 SVD 대신 온라인 PCA(주성분 분석, Principal Component Analysis)를 사용하여 업데이트함으로써 유연성과 최소한의 훈련 오버헤드를 제공한다.

- **Performance Highlights**: 온라인 서브스페이스 디센트(Online Subspace Descent) 방법은 C4 데이터셋에서 60M에서 1B 파라미터의 LLaMA 모델을 사전 훈련하는 데 있어, 최신 저랭크 훈련 방법들보다 낮은 perplexity와 더 나은 다운스트림 태스크 성능을 보인다. 이는 전체 랭크 기준선과의 격차를 좁히는 결과를 가져왔다.



### VALE: A Multimodal Visual and Language Explanation Framework for Image Classifiers using eXplainable AI and Language Models (https://arxiv.org/abs/2408.12808)
Comments:
          15 pages, 10 tables, 3 figures

- **What's New**: 이번 논문에서는 기존의 비유사한 방법들과 차별화된 다채로운 비주얼과 언어 설명 프레임워크인 VALE(Visual and Language Explanation)를 제안합니다. VALE는 이미지 분류 작업에서 DNN의 내재된 작동을 해석하고 설명하는 데 도움을 주며, 시각적 및 텍스트 기반의 설명을 통합하여 인간 친화적인 방식으로 결과를 전달합니다.

- **Technical Details**: VALE 시스템은 이미지 분류기, 설명기(SHAP), 이미지 분할기(Segment Anything Model), 그리고 이미지-텍스트 설명기(VLM)로 구성되어 있습니다. SHTA(clickable) 점수 계산을 통해 분류 이미지의 가장 영향력 있는 영역을 식별한 후, Segment Anything Model을 사용해 객체의 영역을 추출하고, VLM을 통해 시각적 설명에 대한 텍스트 설명을 생성합니다.

- **Performance Highlights**: VALE는 ImageNet 데이터셋 및 맞춤형 수중 SONAR 이미지 데이터셋에서 광범위한 실험을 통해 실용성을 입증했습니다. 이 프레임워크는 기존 XAI 도구를 통합하여 각 이미지 분류에서 보다 명확하고 이해하기 쉬운 결과를 제공합니다.



### Assessing Modality Bias in Video Question Answering Benchmarks with Multimodal Large Language Models (https://arxiv.org/abs/2408.12763)
- **What's New**: 이번 연구에서는 여러 모달리티(visual, textual, auditory)를 동시에 처리할 수 있는 MLLM(멀티모달 대형 언어 모델)을 사용하여 비디오 질문 응답(VidQA) 데이터셋의 단일 모달리티 편향을 찾아내는 모달리티 중요도 점수(MIS)를 새롭게 도입하였습니다.

- **Technical Details**: 모달리티 중요도 점수(MIS)는 질문에 대한 정보를 포함하는 모달리티를 평가하기 위해 설계되었으며, 최첨단 MLLM을 활용하여 모달리티의 중요성을 추정하는 혁신적인 방법을 제안합니다. 이를 통해 기존 데이터셋에서 단일 모달리티 편향과 진정으로 멀티모달 질문의 부족함을 입증하였습니다.

- **Performance Highlights**: 여러 차별적 연구(ablation studies)를 통해 현재 MLLM들이 정보 통합에 효과적이지 않다는 결과를 도출하였으며, 이는 기존 데이터셋에서 모달리티 불균형 때문입니다. 연구에서 제안한 MLLM 기반의 MIS는 모달리티 균형 데이터셋의 구성에 기여하고, 멀티모달 학습의 발전과 MLLMs의 모달리티 간의 시너지 관계를 이해하고 활용하는 능력을 향상시키는 데 도움을 줄 수 있습니다.



### SQL-GEN: Bridging the Dialect Gap for Text-to-SQL Via Synthetic Data And Model Merging (https://arxiv.org/abs/2408.12733)
- **What's New**: 새로운 논문에서는 다양한 SQL 방언에 적용 가능한 Text-to-SQL 시스템을 위한 SQL-GEN이라는 프레임워크를 소개합니다. 이 시스템은 방언별 튜토리얼을 사용하여 고품질의 합성 데이터를 생성하며, 이는 교육 데이터셋을 생성하는 데 효과적입니다.

- **Technical Details**: SQL-GEN은 세 단계의 파이프라인으로 구성됩니다. 첫 번째 단계에서는 SQL 키워드만 포함된 시드 템플릿을 사용하고, 두 번째 단계에서는 이들을 실제 데이터베이스의 값 및 스키마 요소로 채웁니다. 마지막으로, 품질 검사를 통해 생성된 쿼리들이 정확하게 일치하는지 확인합니다. 최신 기법인 Mixture of Experts (MoE) 초기화 방법을 통해 방언별 모델을 단일 시스템으로 통합함으로써 성능 향상을 도모합니다.

- **Performance Highlights**: 이 연구는 모든 LLM이 기존의 합성 데이터 및 인간 주석 데이터에 비해 4%에서 27%의 성능 향상을 보여 주며, PostgreSQL 및 BigQuery와 같은 덜 연구된 방언에 대해서도 실세계 데이터에 대한 평가에서 2.5%에서 7.5%의 성능 향상을 달성했습니다. 또한, 합성 데이터와 인간 주석 데이터를 결합하여 3.3%에서 5.6%의 성능 부스트를 제공했습니다.



### MultiMed: Massively Multimodal and Multitask Medical Understanding (https://arxiv.org/abs/2408.12682)
- **What's New**: 새로운 벤치마크인 MultiMed는 256만 개의 샘플을 기반으로 하여 다양한 의료 모달리티와 과제를 아우르는 대규모 학습을 평가하고 지원하는 데 중점을 둡니다. 각 모달리티는 의료 보고서, 병리학, 유전체학, 단백질 데이터 등 포함됩니다.

- **Technical Details**: MultiMed는 10가지 의료 모달리티(예: 의료 보고서, 병리학, 유전체, 단백질 데이터)를 포함하며, 11가지 도전적인 과제로 구성되어 있습니다. 이 과제는 질병 예후, 단백질 구조 예측, 의료 질문 응답 등을 포함합니다. MultiMed는 여러 관련 모달리티 및 과제에서의 대규모 학습을 지원합니다.

- **Performance Highlights**: MultiMed를 통해 수행된 실험은 단일 모달리티, 다중 모달리티, 다중 과제 모델들의 성능을 벤치마킹하며, 관련 모달리티 및 과제를 아우르는 대규모 의료 모델 교육의 장점을 강조합니다. 이는 진단 및 예후 기능 향상에 기여할 수 있습니다.



### Using generative AI to support standardization work -- the case of 3GPP (https://arxiv.org/abs/2408.12611)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)을 활용하여 표준화 과정에서 발생하는 논의 지점과 유사성 및 차이점을 식별하는 문제를 다루고 있습니다.

- **Technical Details**: BART와 Pegasus XLM 모델을 사용하여 긴 문서를 요약하고, 멤버 간의 동의 및 논의가 필요한 사항을 식별하는 세 가지 작업을 수행했습니다. 연구 디자인은 디자인 사이언스 연구 접근 방식을 따르며, Ericsson AB와 협력하여 3GPP RAN 표준화 활동에 대한 평가를 실시했습니다.

- **Performance Highlights**: 일반적인 텍스트 요약 모델은 전문가의 평가와 높은 상관관계를 보였으나(0.66 ~ 0.98), 도메인별 모델의 필요성이 강조되었습니다. 이러한 모델들은 표준화 그룹에 더 나은 논의 자료를 제공하는 데 중요한 역할을 할 것으로 기대됩니다.



New uploads on arXiv(cs.IR)

### Structural Representation Learning and Disentanglement for Evidential Chinese Patent Approval Prediction (https://arxiv.org/abs/2408.12852)
Comments:
          CIKM 2024, 10 Pages

- **What's New**: 중국 특허 승인 예측을 위한 자동화된 접근 방식이 제안되었습니다. 이는 기존의 단일 텍스트 분류 모델의 한계를 극복하기 위해 Retrieval-based Classification 접근 방식을 사용하여 개발되었습니다.

- **Technical Details**: 본 연구는 Patents의 구조적 표현 학습과 Disentanglement(불확실성 분리)를 수행하는 DiSPat라는 혁신적인 프레임워크를 도입합니다. 이 프레임워크는 세 가지 주요 구성 요소로 구성됩니다: Base Reference Retrieval(BRR), Structural Patent Representation(SPR), Disentangled Representation Learning(DRL).

- **Performance Highlights**: DiSPat는 여러 데이터 세트에서 최첨단 기준 모델을 초월하는 성과를 보였으며, 특히 특허 승인 예측에서 향상된 evidentiality(증거성)를 나타냅니다.



### EAViT: External Attention Vision Transformer for Audio Classification (https://arxiv.org/abs/2408.13201)
- **What's New**: 이번 논문에서는 External Attention Vision Transformer (EAViT) 모델을 제안하고, 이를 통해 오디오 분류 정확도를 향상시키는 방법을 논의합니다. 디지털 오디오 자원의 증가로 인해 정밀하고 효율적인 오디오 분류 시스템의 수요가 급증하고 있으며, EAViT는 이러한 필요에 대응하고자 합니다.

- **Technical Details**: EAViT 모델은 비전 변환기(ViT) 프레임워크에 다중 헤드 외부 주의 기제(multi-head external attention, MEA)를 통합하여 샘플 간의 장기적 의존성과 잠재적 상관관계를 효과적으로 포착합니다. 또한, 학습 가능한 메모리 단위를 사용하여 복잡한 오디오 특징을 효율적으로 처리할 수 있는 능력을 강화합니다. 각 30초 오디오 클립을 3초로 나누어 데이터 세트를 더욱 견고하게 만들고 과적합(overfitting) 위험을 완화합니다.

- **Performance Highlights**: EAViT 모델은 GTZAN 데이터셋을 사용하여 전체 정확도 93.99%를 달성하였으며, 이는 기존 최신 모델을 초월하는 성과입니다. 이는 음악 장르 분류를 위한 새로운 기준을 제시하며, 오디오 데이터 처리 시 ViT 기반 모델의 분류 정확도를 향상시킵니다.



### iSee: Advancing Multi-Shot Explainable AI Using Case-based Recommendations (https://arxiv.org/abs/2408.12941)
Comments:
          Accepted to appear at the ECAI-PAIS 2024 main conference proceedings

- **What's New**: 이번 연구에서는 Explainable AI (XAI) 시스템의 다중 사용자 요구를 충족시키기 위해 'multi-shot' 접근 방식을 제안합니다. 개인화된 'explanation experience'를 통해 AI 의사결정 과정을 개선하는 전략을 소개하는 플랫폼인 iSee를 개발했습니다.

- **Technical Details**: iSee 플랫폼은 Case-based Reasoning (CBR) 방식을 활용하여 성공적인 설명 적응 경험을 캡처하고, 다양한 수준의 AI 및 XAI 전문성을 가진 설계 사용자와 협력하여 최적의 설명 전략을 성과적으로 설계하고 수정할 수 있도록 지원합니다. iSee는 사용자의 요구를 수집하고 과거 경험을 기반으로 가장 적합한 설명 전략을 검색하는 도구를 제공합니다.

- **Performance Highlights**: iSee 플랫폼의 평가 결과, 다양한 응용 분야에서 효과적으로 일반화되고 XAI 모범 사례 채택을 촉진할 수 있는 잠재력이 있음을 확인했습니다. 이 플랫폼은 사용자 피드백을 바탕으로 한 설명 전략 최적화를 통해, 사용자 만족도를 높이고 협업을 통한 지속 개선이 가능함을 보였습니다.



### Multi-Treatment Multi-Task Uplift Modeling for Enhancing User Growth (https://arxiv.org/abs/2408.12803)
- **What's New**: 이번 연구에서는 여러 치료와 여러 작업을 동시에 고려할 수 있는 Multi-Treatment Multi-Task (MTMT) uplift 모델을 제안합니다. 이 모델은 기존의 단일 작업 및 단일 치료 설정에서 벗어나, 사용자 반응의 복잡성을 고려하여 향상된 비즈니스 성과를 도모합니다.

- **Technical Details**: MTMT는 사용자 특성과 치료 정보를 별도로 인코딩하며, multi-gate mixture of experts (MMOE) 네트워크를 활용하여 관련 사용자 특성을 학습합니다. 본 연구에서는 베이스 효과와 특정 치료에 의한 증가 효과를 계산하는 방식으로 각 작업에 대한 치료 효과를 측정합니다. 또한, 치료-사용자 특성 상호작용 모듈을 통해 각 치료와 사용자 특성 간의 상관관계를 모델링합니다.

- **Performance Highlights**: MTMT는 공공 데이터셋과 대규모 제품 데이터셋을 통한 실험 결과에서 뛰어난 성능을 보이며, 기존 모델에 비해 우수한 결과를 나타냅니다. 특히, MTMT는 온라인 게임 플랫폼에 배포되어 대규모 사용자 경험을 향상시키는 데 기여하고 있습니다.



### Data-Centric Approach to Constrained Machine Learning: A Case Study on Conway's Game of Lif (https://arxiv.org/abs/2408.12778)
- **What's New**: 이번 논문은 Conway의 생명 게임을 위한 데이터 중심(machine learning applications) 접근법을 제시하며, 최소한의 아키텍처 네트워크를 학습하여 주어진 단계 수에 대한 전환 규칙을 배우는 문제를 다룹니다. 이를 통해 제한된 학습 설정에서도 효율적인 데이터 설계의 중요성을 강조합니다.

- **Technical Details**: 이 연구에서는 이미지-투-이미지 전환(task of image-to-image translation) 문제로 생명 게임의 규칙을 학습하기 위해 CNN(Convolutional Neural Network)을 사용합니다. 제한된 아키텍처로 작업하여 훈련 데이터를 철저히 제어하는 방식으로 학습합니다. 훈련 데이터의 효율성을 기존 무작위 생성 보드 대신 정교하게 설계한 보드에서 비교합니다.

- **Performance Highlights**: 정교하게 설계된 단일 훈련 보드로 다단계 예측(multi-step prediction task)에서 수렴 속도와 정확도가 크게 향상되었음을 보여주며, 이는 데이터 중심(data-centric) 접근법이 제약된 머신러닝 응용 프로그램에서 유리하다는 것을 입증합니다.



### Using a negative spatial auto-correlation index to evaluate and improve intrinsic TagMap's multi-scale visualization capabilities (https://arxiv.org/abs/2408.12610)
Comments:
          39 pages,10 figures, an accepted version of Journal Cartography and Geographic Information Science

- **What's New**: 이번 연구는 태그 클라우드(tag cloud)의 인기로 인해 발생한 지리적(tag map) 연구의 새로운 접근 방식을 다룹니다. 특히, 기존의 태그 맵 방법론이 여러 스케일에서 태그 레이아웃(tag layout)에서 발생하는 문제, 즉 빈 공간 또는 가까운 태그 간 겹침 문제를 해결하기 위해 새로운 지표를 도입합니다.

- **Technical Details**: 연구에서는 부정적인 공간 자기 상관 지수(negative spatial auto-correlation index)를 태그 맵에 통합하여 다양한 크기의 태그 분포의 균일성을 평가합니다. 이 지수를 TIN(triangulated irregular network) 기반의 고유 태그 맵 레이아웃(layout) 접근법에 결합하여 다중 스케일 시각화(multiscale visualization)를 지원합니다. 이 과정에서는 후보 태그를 반복적으로 필터링(filtering)하며 정의된 지표 기준을 충족하는 최적 태그를 선택합니다.

- **Performance Highlights**: 미국과 이탈리아의 대표 지역에서 실험 결과, 다중 스케일 시각화 능력이 향상되었으나 공간적 밀집도(compactness)와 시간 효율(time efficiency) 사이의 trade-off가 발생했습니다. 동일한 수의 태그를 유지할 때는 더 높은 밀집도를 기록했지만, 더 긴 시간이 소요되었고, 태그 수를 줄일 때는 시간 요구량은 감소했으나 밀집도가 낮아지는 결과를 보였습니다.



New uploads on arXiv(cs.CV)

### MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans? (https://arxiv.org/abs/2408.13257)
Comments:
          Project Page: $\href{this https URL}{\text{this https URL}}$

- **What's New**: MME-RealWorld라는 새로운 벤치마크를 소개하며, MLLMs의 성능을 평가하기 위해 30만 장 이상의 이미지를 수집하고 1만3366장의 고품질 이미지를 필터링했습니다. 이는 25명의 전문 주석자와 7명의 MLLMs 전문가가 참여하여 2만9429개의 질문-답변 쌍을 생성한 것입니다.

- **Technical Details**: MME-RealWorld는 43개 서브작업으로 구성되어 있으며, 고해상도의 이미지를 포함합니다. 평균 해상도는 2000×1500이며, 주석자들은 모든 질문이 MLLMs에게 도전적이도록 보장했습니다. 28개의 주요 MLLMs를 평가했으며, 결과적으로 최고의 모델도 60% 정확도에 도달하지 못했습니다.

- **Performance Highlights**: MME-RealWorld는 현재까지 수집된 수동 주석 벤치마크 중 최대 규모를 자랑하며, 현실 시나리오에 중점을 두고 있습니다. 이 벤치마크의 도전 과제가 MLLMs가 고해상도 이미지를 이해하고 복잡한 실제 시나리오를 인식하는 것임을 보여줍니다.



### Ensemble Modeling of Multiple Physical Indicators to Dynamically Phenotype Autism Spectrum Disorder (https://arxiv.org/abs/2408.13255)
- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애(ASD)의 조기 탐지를 위한 비디오 기반 접근 방식을 제시합니다. 모바일 애플리케이션 GuessWhat를 통해 자연적인 가정 비디오가 수집되었으며, 이를 통해 3,000개 이상의 고품질 비디오 데이터셋이 구축되었습니다. 이 데이터셋을 기반으로 LSTM 모델을 훈련시켜 눈의 시선, 머리 위치, 얼굴의 특징을 분석합니다.

- **Technical Details**: 연구팀은 눈의 시선, 머리 위치 및 얼굴 랜드마크와 같은 입력 특징을 사용하여 LSTM 기반 모델을 훈련시켰습니다. 최종적으로 AUC(Area Under the Curve) 점수를 90%로 개선하기 위해 여러 모델의 결과를 융합하는 기술을 사용했습니다. 모든 모델은 높은 성과를 보였으며 성별 및 연령대 간의 공정성을 높였습니다.

- **Performance Highlights**: 최종 모델은 86%, 67%, 78%의 테스트 AUC를 기록하였으며, 여러 알고리즘의 융합을 통해 AUC를 90%로 증가시켜 진단의 정확성을 개선했습니다. 이는 자폐 조기 탐지의 신뢰성을 향상시키며, 주관적 평가에 대한 의존도를 줄이고 접근성과 공정성을 높입니다.



### LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation (https://arxiv.org/abs/2408.13252)
Comments:
          Project page: this https URL

- **What's New**: LayerPano3D는 단일 텍스트 프롬프트에서 전체 시점, 탐색 가능한 팬오라마 3D 장면을 생성하는 혁신적인 프레임워크입니다. 이 프레임워크는 참조 2D 팬오라마를 여러 깊이 수준의 레이어로 분해하여 각 레이어는 이전 뷰를 통해 보이지 않는 공간을 드러내는 방식으로 작동합니다.

- **Technical Details**: 세 가지 주요 디자인이 포함됩니다: 1) 텍스트 유도 앵커 뷰 합성 파이프라인을 통해 고품질의 일관된 팬오라마 생성을 구현하고, 2) Layered 3D Panorama를 기반으로 복잡한 장면 계층 구조를 관리하며, 3) 이를 3D Gaussian으로 변환하여 제약 없는 시청 경로로 세부적인 360도 장면을 표현합니다.

- **Performance Highlights**: LayerPano3D는 단일 텍스트 프롬프트에서 몰입감 있는 레이어드 팬오라마 장면을 생성하는 데 있어 최첨단 성능을 자랑하며, 전체 뷰 일관성과 탐색 경험 모두 뛰어난 결과를 제공합니다. 이 프레임워크는 비전문가를 위한 더 사용하기 쉬운 인터페이스를 제공하며, 실제 응용 프로그램에서 3D 팬오라마 장면을 보다 쉽게 접근할 수 있도록 합니다.



### Re-evaluation of Face Anti-spoofing Algorithm in Post COVID-19 Era Using Mask Based Occlusion Attack (https://arxiv.org/abs/2408.13251)
Comments:
          10 pages, This work was done in 2020

- **What's New**: 이번 연구는 COVID-19 팬데믹으로 인해 마스크 착용 요구가 증가함에 따라, 얼굴 인식 시스템에서 프레젠테이션 공격 탐지(PAD) 알고리즘의 성능 저하를 분석합니다. 특히, 다양한 형태의 마스크와 안경을 사용하여 얼굴의 다양한 부위를 가리는 상황에서 PAD 알고리즘의 검증을 수행했습니다.

- **Technical Details**: 연구팀은 네 가지 기반 PAD 알고리즘(텍스처, 이미지 품질, 프레임 차이/모션, 그리고 CNN을 통한 추상적 특징)에 대해 성능을 평가했습니다. 추가로, CNN과 Local Binary Pattern(LBP) 텍스처를 결합한 하이브리드 모델을 구현했습니다. 실험에서는 Replay-Attack과 OULU-NPU 데이터셋을 사용하여 합성 가림 공격(synthetic occlusion attacks)을 적용했습니다.

- **Performance Highlights**: 모든 PAD 알고리즘의 성능은 가림 추가로 인해 크게 저하되었습니다. 특히, 마스크의 가림 크기와 3D 큐의 유무가 성능에 미치는 영향을 확인했으며, 기존 LBP와 CNN 기반 PAD 모델 간의 성능 비교를 통해 중요한 결과를 도출했습니다.



### Foundational Model for Electron Micrograph Analysis: Instruction-Tuning Small-Scale Language-and-Vision Assistant for Enterprise Adoption (https://arxiv.org/abs/2408.13248)
Comments:
          Our paper is published at ICML 2024 Workshop ML for Life and Material Science: From Theory to Industry Applications, Vienna, Austria

- **What's New**: 이 논문에서는 기존의 반도체 이미징 및 분석의 한계를 극복하기 위해 작은 규모의 다중 모드 프레임워크인 MAEMI(모세관 전자 현미경 이미지 분석)를 도입합니다.

- **Technical Details**: MAEMI는 비전-언어(vision-language) 지침 튜닝을 통해 반도체 전자 현미경 이미지 분석을 수행하며, 대형 다중 모드 모델을 활용하여 사용자 맞춤형 지침 데이터 세트를 생성합니다. 또한, 지식 증류(knowledge distillation)를 통해 대형 모델에서 소형 모델로 지식을 전이하여, 시각적 질문 응답(Visual Question Answering, VQA) 작업에서 소형 모델의 정확성을 향상시킵니다.

- **Performance Highlights**: MAEMI는 전통적인 방법보다 우수한 성능을 보여주며, 데이터 분포 변화에 적응하고 높은 처리량 스크리닝(high-throughput screening)을 지원합니다. 기업들은 MAEMI를 자체 데이터에 추가로 미세 조정하여 프라이버시와 저비용 소비자 하드웨어에서의 성능을 향상시킬 수 있습니다.



### MCTR: Multi Camera Tracking Transformer (https://arxiv.org/abs/2408.13243)
- **What's New**: 이번 연구는 Multi-Camera Tracking tRansformer (MCTR) 라는 새로운 엔드 투 엔드 접근법을 제안하여, 여러 카메라의 겹치는 시야에서 다중 객체 감지 및 추적을 수행합니다. MCTR는 각 카메라의 뷰에 대해 독립적으로 감지 및 감지 임베딩을 생성하며, 글로벌 추적 정보를 유지하는 트랙 임베딩을 새로이 정의합니다.

- **Technical Details**: MCTR는 기존의 DETR (DEtector TRansformer)과 함께 작동하며, 추적 모듈과 연관 모듈을 추가하여 다중 카메라 추적을 용이하게 합니다. 이 시스템에서는 각 프레임에서 뷰 특정 감지 임베딩의 정보를 통합하여 트랙 임베딩을 업데이트하며, 뷰와 프레임마다 감지와 트랙 간의 확률적 연관을 생성합니다.

- **Performance Highlights**: MCTR는 MMPTrack 및 AI City Challenge라는 대규모 데이터셋에서 실험을 통해 검증되었으며, 엔드 투 엔드 방식으로 일관된 객체 추적을 달성하는 데 있어 유의미한 성과를 보였습니다. 특히, 확률적 연관을 통해 객체의 일관된 정체성을 유지하는데 효과적입니다.



### CustomCrafter: Customized Video Generation with Preserving Motion and Concept Composition Abilities (https://arxiv.org/abs/2408.13239)
Comments:
          project page: this https URL

- **What's New**: 이 논문에서는 기존의 Fine-tuning 없이도 고품질 비디오 생성을 가능하게 하는 CustomCrafter라는 새로운 프레임워크를 제안합니다. 기존 방법들이 비디오 생성의 일관성을 손상시킨 문제를 해결하기 위해, 모델의 본래의 동작 생성 및 개념 조합 능력을 보존할 수 있는 방법을 탐구합니다.

- **Technical Details**: CustomCrafter는 Video Diffusion Models (VDMs)의 동작 생성을 훼손하지 않고 개념 조합 능력을 향상시키기 위해, Spatial Subject Learning Module 및 Dynamic Weighted Video Sampling Strategy와 같은 혁신적인 모듈을 구현합니다. Spatial Subject Learning Module은 VDMs의 self-attention 레이어와 spatial cross-attention 레이어의 가중치를 업데이트하여 새로운 주체의 외관을 포착하는 능력을 개선하며, 동적 가중치 비디오 샘플링 전략은 노이즈 제거 과정에서 동작 생성 능력을 유지하기 위해 플러그 앤 플레이(plug-and-play) 방식으로 파라미터를 조정합니다.

- **Performance Highlights**: 본 연구의 실험 결과, CustomCrafter는 기존의 방법들에 비해 동작의 유동성, 개념 조합 능력 및 주제의 외관 일관성을 유지하여 사용자에게 더 나은 비디오 생성 경험을 제공합니다. 정량적, 정성적 및 사용자 평가 결과 모두에서 우리의 방법이 맞춤형 비디오 생성에서 우수한 성능을 보여주었습니다.



### D&M: Enriching E-commerce Videos with Sound Effects by Key Moment Detection and SFX Matching (https://arxiv.org/abs/2408.13226)
Comments:
          9 pages, 4 figures

- **What's New**: 이 연구에서는 VDSFX(Video Decoration with SFX)라는 새로운 작업을 소개하며, 제품을 홍보하기 위한 E-commerce 비디오에서 자동으로 주요 순간을 감지하고 선택된 음향 효과(sound effects, SFX)를 추가하는 방법론을 개발합니다. 이를 위해 E-commerce 플랫폼에서 수집한 대규모 데이터셋 SFX-Moment를 구축하고, key moment detection과 SFX 매칭을 동시에 수행하는 D&M 방법론을 제안합니다.

- **Technical Details**: D&M은 DETR(Detection Transformer)에 기반한 방법으로, 비디오에서 주요 순간을 감지하고 해당 순간에 적합한 SFX를 매칭합니다. 또한, moment-SFX matching (MSM)을 통한 사전 학습(pre-training)과 tag-aware negative sampling (TaNS)을 이용하여 트레이닝 과정을 개선합니다. 데이터셋 SFX-Moment는 16,942개의 비디오, 약 40,000개의 주요 순간, 356개의 독특한 SFX로 구성되어 있습니다.

- **Performance Highlights**: SFX-Moment에 대한 광범위한 실험 결과, 제안된 D&M 방법이 기존의 여러 경쟁 기준선보다 우수한 성능을 보였습니다. 이 연구는 E-commerce 비디오의 사용자 참여 경험을 향상시키기 위한 중요한 기여를 합니다.



### Identifying Crucial Objects in Blind and Low-Vision Individuals' Navigation (https://arxiv.org/abs/2408.13175)
Comments:
          Paper accepted at ASSETS'24 (Oct 27-30, 2024, St. Johns, Newfoundland, Canada). arXiv admin note: substantial text overlap with arXiv:2407.16777

- **What's New**: 이 논문은 시각 장애인 및 저시력자(BLV)가 도로, 보도 및 실내 환경에서 탐색하는 데 필수적인 90개의 물체 목록을 제공하며, 이는 BLV 개인의 탐색을 지원하기 위해 필요로합니다.

- **Technical Details**: 21개의 공개 비디오를 분석하여 초기 목록을 개발한 후, BLV 개인과 그들을 동반한 시각 장애인, 저시력자들의 피드백을 받은 후 이 목록을 수정하였습니다. 이 연구는 기존 데이터셋에서 BLV 탐색을 위한 중요한 물체들을 포함하지 않음을 밝혔습니다. 90개의 물체에 대한 자세한 레이블링이 31개의 비디오 세그먼트에서 제공되었습니다.

- **Performance Highlights**: 이 연구는 BLV 커뮤니티를 위한 보다 포괄적이고 효과적인 탐색 보조 도구 개발을 지원하고자 하며, 90개의 물체 목록과 21개의 비디오 및 31개의 레이블링된 비디오 세그먼트를 공개하여 관련 연구에 기여합니다.



### KonvLiNA: Integrating Kolmogorov-Arnold Network with Linear Nystr\"om Attention for feature fusion in Crop Field Detection (https://arxiv.org/abs/2408.13160)
- **What's New**: 이번 연구에서는 정밀 농업을 위한 필수 요소인 농작물 필드 탐지를 위한 새로운 프레임워크인 KonvLiNA를 소개합니다. 이 프레임워크는 Convolutional Kolmogorov-Arnold Networks (cKAN)와 Nyström attention 메커니즘을 통합하여 효과적인 농작물 탐지를 가능하게 합니다.

- **Technical Details**: KonvLiNA는 KAN (Kolmogorov-Arnold Networks)의 적응형 활성화 함수와 Nyström attention의 효율성을 활용하여 대규모 데이터를 처리합니다. 이를 통해 특성 추출(feature extraction)이 크게 향상되어 복잡한 농업 환경에서 정교한 패턴을 포착할 수 있습니다.

- **Performance Highlights**: 쌀 농작물 데이터셋에서의 실험 결과는 KonvLiNA가 최신 방법들보다 우수하다는 것을 입증하며, Swin-L 백본(Backbone)을 사용할 때 0.415 AP, 0.459 AR을 달성했습니다. 이는 전통적인 YOLOv8보다 상당한 성능 향상을 보여줍니다. 또한, COCO 데이터셋에서 소형, 중형 및 대형 물체에 대해 경쟁력 있는 성능을 발휘하며 KonvLiNA의 다양한 농업 환경에서의 효율성을 강조합니다.



### Interpretable breast cancer classification using CNNs on mammographic images (https://arxiv.org/abs/2408.13154)
Comments:
          16 pages, 13 figures (9 in the main text, 3 in the appendix). Accepted at PMLR 2024

- **What's New**: 이 연구는 유방암 분류를 위한 convolutional neural networks (CNN)의 결정 과정에 대한 통찰력을 얻는 필요성에 주목하며, Mammographic Image Analysis Society (MIAS) 데이터셋에서 학습한 CNN에 대해 LIME, Grad-CAM, Kernel SHAP와 같은 해석 기법을 비교합니다.

- **Technical Details**: MIAS 데이터셋을 기반으로 한 CNN의 훈련 후, 제안된 해석 기법들은 CNN의 예측 이유를 밝혀내는 데 중점을 두었습니다. 특히, Grad-CAM 기법은 정상, 양성, 악성 유방 조직의 행동을 이해하는 데 효과적이었습니다. 이 연구는 이미지 분류기인 CNN의 post-hoc 해석 가능성을 강조하여 AI 기반 진단의 투명성을 높이는 방법을 제안합니다.

- **Performance Highlights**: Grad-CAM은 CNN의 행동을 보다 포괄적으로 이해할 수 있는 통찰력을 제공하였으며, 유방 조직의 특이한 패턴을 발견하는 데 기여했습니다. 이 연구 결과는 임상 실무에서 기계 학습 모델과 해석 기술의 활용에 대한 중요한 시사점을 제공합니다.



### Long-Term Pre-training for Temporal Action Detection with Transformers (https://arxiv.org/abs/2408.13152)
- **What's New**: 이 논문은 Temporal Action Detection (TAD)에서 데이터 부족 문제를 해결하고자 Long-Term Pre-training (LTP) 전략을 제안합니다. LTP는 클래스별 합성과 장기 전제 작업을 포함하여, DETR (DEtection TRansformer) 모델의 성능을 크게 향상시킵니다.

- **Technical Details**: Long-Term Pre-training (LTP)은 두 가지 주요 구성 요소로 이루어집니다: 1) 클래스별 합성 (class-wise synthesis), 2) 장기 전제 작업 (long-term pretext tasks). LTP는 충분하고 균형 잡힌 데이터 다양성을 제공하기 위해 대규모 비디오 분류 데이터셋을 활용해 긴 형식 비디오 기능을 합성합니다. 또한, 모델에서 장기 의존성을 학습하도록 하기 위해 'ordinal'과 'scale' 조건을 가진 두 가지 장기 전제 작업을 도입합니다.

- **Performance Highlights**: LTP를 적용한 DETR 모델은 ActivityNet-v1.3 및 THUMOS14 두 가지 TAD 벤치마크에서 최신 성능 (state-of-the-art)과 비교해 월등한 성과를 보였습니다. 이 연구는 데이터 부족 문제가 DETR 기반 TAD에서 심각한 퇴화 문제를 초래함을 강조하고, LTP가 이 문제를 효과적으로 완화한다고 결론짓습니다.



### Focus on Neighbors and Know the Whole: Towards Consistent Dense Multiview Text-to-Image Generator for 3D Creation (https://arxiv.org/abs/2408.13149)
- **What's New**: 본 논문에서는 CoSER라는 새로운 Multiview Text-to-Image Generator를 도입하여,Text-to-3D 작업에서의 효율성과 품질을 동시에 실현했습니다. 이 모델은 neighbor-view 일관성을 학습하고 모든 뷰를 신속하게 탐색하여 모호성을 완화하는데 중점을 두었습니다.

- **Technical Details**: CoSER는 인접 뷰 간의 조밀한 상호작용을 통해 global spatial structure를 인지하며, 물리적인 원리에 의해 명시적으로 정의된 motion paths를 따라 정보를 집계하여 세부 사항을 향상시킵니다. 또한, spiral bidirectional 방식으로 모든 뷰를 빠르게 스캔하여 홀리스틱(holistic) 정보를 인식하고 점수를 기반으로 weighted down-sampling을 수행하여 모든 뷰 간의 정보 융합을 효과적으로 이루어낼 수 있도록 합니다.

- **Performance Highlights**: 광범위한 평가 결과, CoSER는 밀집하고 고충실도(content-consistent) 멀티 뷰 이미지를 생성할 수 있으며, 다양한 3D 생성 모델에 유연하게 통합될 수 있는 능력을 입증했습니다.



### ShapeICP: Iterative Category-level Object Pose and Shape Estimation from Depth (https://arxiv.org/abs/2408.13147)
- **What's New**: 최근 연구에서는 단일 깊이 이미지에서 카테고리 수준의 객체의 자세(pose)와 형태(shape)를 추정하는 기술이 주목받고 있습니다. 이 연구는 이번에 데이터 주석이 없는 방법을 바탕으로 한 반복적 추정(iterative estimation) 방법을 제안합니다.

- **Technical Details**: 이 알고리즘인 ShapeICP는 반복적 최근접점(iterative closest point, ICP) 알고리즘에 기반을 두고 있으며, 구질적(qualitative)인 객체 형태 모델인 메쉬(mesh) 기반의 활성 형태 모델을 채택하여 카테고리 수준의 자세와 형태 추정 작업을 수행합니다.

- **Performance Highlights**: ShapeICP는 데이터 기반의 접근 방식 없이도 여러 테스팅 결과에서 양호한 성능을 발휘하며, 기존의 데이터 기반 방법들이 신경망을 통해 학습한 자세 데이터를 필요로 하는 것과는 대조적으로, 데이터 부족 문제를 해결할 수 있는 새로운 가능성을 제공합니다.



### Deep Learning at the Intersection: Certified Robustness as a Tool for 3D Vision (https://arxiv.org/abs/2408.13135)
Comments:
          This paper is an accepted extended abstract to the LatinX workshop at ICCV 2023. This was uploaded a year late

- **What's New**: 이번 논문에서는 기계 학습에서 인증된 강인성(cerified robustness)과 3D 객체 모델링(3D object modeling) 간의 새로운 연결 고리를 제안합니다. 특히, 공간의 점유율을 나타내는 분류기의 Maximal Certified Radius (MCR)와 공간의 Signed Distance Function (SDF)의 흥미로운 상관관계를 강조합니다.

- **Technical Details**: 우리는 인증 방법인 랜덤 스무딩(randomized smoothing, RS)을 활용하여 SDF를 계산할 것을 제안합니다. 그러나 RS의 높은 계산 비용이 실제 사용을 방해하기 때문에, 저차원(low-dimensional) 애플리케이션에서 효율적으로 RS를 실행할 수 있는 알고리즘을 제안합니다. 이는 미리 계산된 복셀 그리드(voxel grids)에서 가우시안 스무딩(Gaussian smoothing)으로 기본 연산을 표현하여 가능하게 합니다.

- **Performance Highlights**: 우리의 접근법은 새로운 시점 합성(novel view synthesis)의 증명 개념 실험을 통해 검증되었습니다. 실험 결과, 우리의 방법이 장면의 유용한 표현을 학습하면서 바람직한 시각적 결과를 유지할 수 있음을 보여줍니다.



### CathAction: A Benchmark for Endovascular Intervention Understanding (https://arxiv.org/abs/2408.13126)
Comments:
          10 pages. Webpage: this https URL

- **What's New**: CathAction 데이터셋은 catheterization (카테터 삽입) 이해 및 충돌 감지를 위한 대규모 데이터셋으로, 약 500,000개의 주석이 달린 프레임과 25,000개의 ground truth 마스크를 포함하고 있습니다. 이는 기존의 작은 데이터셋과 한정된 태스크에서 벗어나 보다 포괄적인 데이터 지원을 제공합니다.

- **Technical Details**: CathAction 데이터셋은 segmentation (세분화), collision detection (충돌 감지), action understanding (행동 이해)와 같은 다양한 임상 작업을 포함하고 있습니다. 이 데이터셋은 실제 및 모의 환경에서 수집된 데이터를 바탕으로 하며, deep learning (딥 러닝) 기반의 방법론 발전에 기여할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: CathAction은 현재까지 가장 크고 현실적인 endovascular intervention (혈관 내 중재) 데이터셋으로, surgeons가 직면하는 고유한 도전 과제와 공개 연구 질문들을 다루고 있습니다. 이 데이터셋을 통해 surgeons는 실시간으로 visual feedback (시각적 피드백)을 받을 수 있어, 환자 안전성이 향상되고, 실제 응용 프로그램에 적용할 수 있는 방법 개발에 기여할 것으로 기대됩니다.



### Evidential Deep Partial Multi-View Classification With Discount Fusion (https://arxiv.org/abs/2408.13123)
Comments:
          Ongoing work. 13 pages, 3 figures, 6 tables

- **What's New**: 이번 연구에서는 불완전한 다중 뷰 데이터 분류(incomplete multi-view data classification)의 문제를 해결하기 위해, K-means 기반의 임퓨테이션(imputation) 방법과 신뢰할 수 있는 증거를 활용하는 Conflict-Aware Evidential Fusion Network (CAEFN)를 제안합니다. 이 방법은 다중 뷰의 신뢰성을 동적으로 고려하여 보다 신뢰할 수 있는 예측 결과를 제공합니다.

- **Technical Details**: 제안된 EDP-MVC 프레임워크는 K-means 임퓨테이션을 통해 누락된 뷰를 처리하고, 이를 통해 완전한 다중 뷰 데이터 집합을 생성합니다. CAEFN은 각 뷰에서의 불확실성을 측정하고, 이들 간의 충돌(conflict)을 완화하기 위해 학습 가능한 할인 요인(learnable discount factor)을 사용합니다. 이를 통해, 보다 신뢰할 수 있는 증거 융합을 보장합니다.

- **Performance Highlights**: 여러 베인치마크 데이터셋에서의 실험 결과, EDP-MVC는 기존의 최첨단 방법들보다 경쟁력 있는 성과를 달성하며, 충돌과 노이즈가 있는 데이터셋에서도 우수한 성능을 나타냅니다.



### Map-Free Visual Relocalization Enhanced by Instance Knowledge and Depth Knowledg (https://arxiv.org/abs/2408.13085)
Comments:
          17 pages,6 figures

- **What's New**: 이 논문은 모노큘러 이미지를 사용하는 relocalization의 성능을 높이기 위해 instance knowledge와 depth knowledge를 활용한 새로운 map-free relocalization 방법을 제안합니다. 전통적인 지도 기반 방법의 한계를 극복하여, 사전 제작된 지도를 필요로 하지 않고도 정확한 위상(위치와 방향)을 추정할 수 있습니다.

- **Technical Details**: 제안된 방법은 instance 기반의 매칭 정보를 활용하여 전체 매칭 결과를 개선합니다. 이로 인해 서로 다른 객체 간의 잘못된 매칭 가능성이 크게 줄어듭니다. 또한, 단일 이미지로부터 추정한 메트릭 깊이를 사용하여 메트릭 오류를 줄이고 스케일 회복 정확도를 개선하는 방식으로 작동합니다. 이를 위해 다단계 매칭 방식을 통합하여 instance-level과 feature-level 접근 방식을 결합합니다.

- **Performance Highlights**: 제안된 방법은 map-free 검증 세트에서 가장 뛰어난 결과를 보였으며, Map-free Visual Relocalization 챌린지의 어려운 테스트 세트에서도 경쟁력 있는 성능을 입증했습니다. 이 방법은 기존의 최첨단 알고리즘들을 초월하는 외연성 성능을 나타내며, 복잡한 환경에서의 실험을 통해 그 효과를 보여주었습니다.



### Atlas Gaussians Diffusion for 3D Generation with Infinite Number of Points (https://arxiv.org/abs/2408.13055)
- **What's New**: 본 논문에서는 3D 생성 기술의 개선을 위해 새로운 형태의 표현인 Atlas Gaussians를 제안합니다. 이 표현은 지역 패치의 결합으로 형상을 모델링하며, 각 패치는 3D Gaussians를 디코딩할 수 있습니다. 또한, UV 기반 샘플링을 통합하여 충분히 큰 수의 3D Gaussian 포인트를 생성할 수 있는 방법론을 제시합니다.

- **Technical Details**: Atlas Gaussians는 각 패치가 3D Gaussians를 디코딩할 수 있도록 기능 벡터의 시퀀스를 매개변수화하는 형태로 설계되었습니다. 이 과정에서 transformer 기반의 디코더를 사용하여 저차원 잠재 공간을 Atlas Gaussians로 매핑하고, 자가 주의(self-attention) 층의 복잡성을 줄여 계산 효율성을 높였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 3D 생성 기술들에 비해 우수한 성능을 보이며, VAE와 LDM 파라다임을 통합하여 표준 3D 생성 벤치마크에서도 뛰어난 결과를 나타냈습니다.



### G3FA: Geometry-guided GAN for Face Animation (https://arxiv.org/abs/2408.13049)
Comments:
          BMVC 2024, Accepted

- **What's New**: 이 논문은 Face Animation을 위한 Geometry-guided GAN (G3FA)을 소개합니다. 이를 통해 2D 이미지만을 사용하여 3D 정보를 결합할 수 있는 모델을 구현하여 얼굴 애니메이션의 기하학적 일관성을 향상시킵니다.

- **Technical Details**: G3FA는 3D 기하학적 정보를 추출하기 위해 역 렌더링 기법을 통합하여 생성기에 대한 기술 피드백 루프를 개선합니다. 2D 모션 왜곡(2D motion warping)을 활용하여 모션 다이내믹스를 캡처하고, Orthogonal ray sampling 및 Volume rendering 기법을 사용하여 최종 시각적 출력을 생성합니다.

- **Performance Highlights**: G3FA는 VoxCeleb2 및 TalkingHead 벤치마크를 사용한 다양한 평가 프로토콜에서 실험을 수행하여 기존의 최신 실시간 얼굴 애니메이션 방법 대비 뛰어난 성능을 보여주었습니다.



### Improving the Classification Effect of Clinical Images of Diseases for Multi-Source Privacy Protection (https://arxiv.org/abs/2408.13038)
Comments:
          Under review

- **What's New**: 이 논문은 의료 데이터의 프라이버시를 보호하면서 여러 병원에서 훈련된 보조 진단 모델을 통합하는 새로운 방법을 제안합니다. 기존의 중앙집중식 접근 방식의 제약 문제를 해결하기 위해 데이터 벡터(data vectors)에 기반한 새로운 훈련 프레임워크를 도입했습니다.

- **Technical Details**: 제안된 프레임워크는 각 병원이 프라이빗 데이터에서 사전 훈련된 모델을 미세 조정하고, 미세 조정된 모델의 가중치와 사전 훈련된 가중치의 차이를 계산하여 데이터 벡터를 생성하도록 합니다. 이러한 데이터 벡터는 다양한 병원의 모델 정보가 통합된 합성 가중치(synthetic weights)를 생성하기 위해 더해집니다. 이 접근 방식은 데이터 교환이나 동기화된 훈련을 요구하지 않으며, 실제 응용에서의 장벽을 감소시킵니다.

- **Performance Highlights**: 실험 결과, 이 데이터 벡터 기반의 혼합 모델은 단일 병원에서 독립적으로 훈련된 그 어떤 모델보다도 성능이 뛰어난 것으로 나타났습니다. 논문은 의료 데이터 프라이버시 보호와 모델 훈련 간의 갈등을 해결할 수 있는 새로운 관점을 제공하며, 의료 지능의 발전에 기여할 수 있는 가능성을 보여줍니다.



### S4D: Streaming 4D Real-World Reconstruction with Gaussians and 3D Control Points (https://arxiv.org/abs/2408.13036)
- **What's New**: 최근 Gaussian을 이용한 동적 장면 재구성이 큰 관심을 받고 있습니다. 기존 방법들은 일반적으로 전역 변형 필드를 이용하여 3D 장면을 변형하지만, 본 연구에서는 새로운 접근법으로 이산 3D 제어점을 도입하여 더욱 정확하고 효율적인 동작 표현을 가능하게 합니다.

- **Technical Details**: 3D 제어점을 이용한 새로운 방법론을 통해 로컬 레이를 물리적으로 모델링하고, 모션 디커플링(coordinate system) 좌표 시스템을 수립하여 전통적인 그래픽스와 학습 가능 파이프라인을 결합합니다. 이 프레임워크는 3D 재구성을 시작으로 3D 세분화, 3D 제어점 생성, 물체별 동작 조작, 잔여 보상을 포함한 네 개의 독립적인 하위 모듈로 4D 실제 세계 재구성을 분해합니다.

- **Performance Highlights**: 본 연구의 실험 결과는 Neu3DV 및 CMU-Panoptic 데이터셋에서 기존의 최첨단 4D Gaussian Splatting 기법보다 뛰어난 성능을 보이며, NVIDIA 4070 GPU에서 초당 2초 이내에 3D 제어점 최적화를 달성하여 훈련 속도를 크게 향상시켰습니다.



### VFM-Det: Towards High-Performance Vehicle Detection via Large Foundation Models (https://arxiv.org/abs/2408.13031)
Comments:
          In Peer Review

- **What's New**: 본 논문에서는 전이 학습된 차량 모델(VehicleMAE)과 대형 언어 모델(T5)을 결합한 새로운 차량 탐지 프레임워크 VFM-Det을 제안합니다. 기존 탐지기는 일반적인 객체 탐지 모델을 기반으로 하고 있어 최적의 성능을 내지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: VFM-Det은 지역 제안 기반 탐지 프레임워크를 따르며, VehicleMAE를 통해 제안의 특징들을 향상시키고, VAtt2Vec 모듈을 통해 차의 의미적 속성을 예측하여 대조 학습(contrastive learning)을 통해 시각적 특징을 강화합니다.

- **Performance Highlights**: Cityscapes 데이터셋에서 AP_{0.5}는 5.1% 향상되었고, AP_{0.75}는 6.2% 향상된 결과가 있는 등 제안된 차량 탐지기가 기존 방법들보다 뛰어난 성능을 보였습니다.



### Indoor scene recognition from images under visual corruptions (https://arxiv.org/abs/2408.13029)
- **What's New**: 이번 논문에서는 캡션 기반의 의미적 특징과 시각적 데이터를 융합하여 실내 장면 인식을 위한 새로운 접근 방식을 제안하고 있습니다. 이 접근법은 데이터 부식에 대한 정확도와 강건성을 향상시키는데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 두 개의 멀티모달 네트워크를 사용하여 CNN(Convolutional Neural Network) 모델에서 추출한 시각적 특징과 GCN(Graph Convolutional Network)을 통해 생성된 의미적 캡션을 시너지 효과를 내도록 결합하였습니다. 이는 고수준 이미지 설명과 저수준 이미지 설명의 두 가지 스트림 인코딩 과정을 통해 진행됩니다.

- **Performance Highlights**: 제안된 멀티모달 모델은 Places365 데이터셋의 손상된 하위 집합에 대해 평가했을 때 Top-1 정확도가 눈에 띄게 개선되었습니다. 고유한 이미지에서는 독립적인 시각적 모델이 높은 정확도를 보였으나, 손상이 심할수록 성능 저하가 두드러졌습니다. 반면에 멀티모달 모델은 깨끗한 조건에서도 정확도가 향상되었으며 다양한 이미지 부식에 대한 강한 강건성을 보였습니다.



### Learning 2D Invariant Affordance Knowledge for 3D Affordance Grounding (https://arxiv.org/abs/2408.13024)
- **What's New**: 본 논문에서는 3D 물체의 기능적 영역을 예측하는 3D Object Affordance Grounding의 새로운 접근 방식을 소개합니다. 특히, 다양한 사람-물체 상호작용 이미지를 통해 일반화 가능한 불변의 affordance 지식을 학습하는 MIFAG 프레임워크를 제안합니다.

- **Technical Details**: MIFAG 프레임워크는 Invariant Affordance Knowledge Extraction Module (IAM)과 Affordance Dictionary Adaptive Fusion Module (ADM)으로 구성됩니다. IAM은 여러 이미지에서 불변의 affordance 지식을 점진적으로 추출하고, ADM은 이 지식을 포인트 클라우드 포맷과 통합하여 상호작용하는 물체의 기능적 영역을 예측합니다.

- **Performance Highlights**: 실험 결과, 제안된 MIFAG 방법은 기존 최첨단 방법들보다 우수한 성능을 보였습니다. MIPA 벤치마크를 구축하여 다양한 시각적 데이터에서의 affordance 이해를 진전시켰습니다.



### EasyControl: Transfer ControlNet to Video Diffusion for Controllable Generation and Interpolation (https://arxiv.org/abs/2408.13005)
- **What's New**: 본 논문에서는 EasyControl이라는 새로운 범용 프레임워크를 제안하여 비디오 생성에서 다양한 조건을 관리하는 방법을 논의합니다.

- **Technical Details**: EasyControl은 condition adapter를 통해 조건 특징을 전파하고 주입하여 단일 조건 맵만으로 비디오 생성을 조절할 수 있도록 합니다. 이 프레임워크는 raw pixel, depth, HED 등 다양한 조건을 Unet 기반의 사전 훈련된 비디오 확산 모델에 통합할 수 있습니다.

- **Performance Highlights**: EasyControl은 공공 데이터셋에 대한 포괄적인 실험을 통해 기존의 최첨단 기법들과 비교할 때 FVD 지표에서 152.0, IS 지표에서 19.9의 성능 향상을 보여주었습니다. 이 모델은 비디오 생성 뿐만 아니라 이미지 보존 능력에서도 강력한 성능을 나타냈습니다.



### BoostTrack++: using tracklet information to detect more objects in multiple object tracking (https://arxiv.org/abs/2408.13003)
- **What's New**: 이번 논문에서는 기존의 BoostTrack 방법의 한계를 분석하고, True Positive detection을 개선하기 위한 새로운 기법을 제안합니다. 이를 통해 다중 객체 추적(MOT)에서의 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 shape, Mahalanobis distance 및 새로운 soft BIoU 유사성 조합을 활용하여 진정한 양성 검출(true positive detections) 선택을 향상합니다. 또한 soft detection confidence boost 기술을 도입하여 새로운 confidence 점수를 계산하고, 업데이트 되지 않은 tracklet과의 유사성 측정에 기반한 변동 유사성 threshold를 설정합니다.

- **Performance Highlights**: BoostTrack+ 기준에 통합된 우리의 방법은 MOT17 데이터셋에서 거의 최신 기술 수준의 결과를 달성하였으며, MOT20 데이터셋에서는 새로운 최신 기술 수준의 HOTA 및 IDF1 점수를 기록했습니다.



### A Survey on Drowsiness Detection -- Modern Applications and Methods (https://arxiv.org/abs/2408.12990)
Comments:
          accepted at the IEEE Transactions on Intelligent Vehicles 2024

- **What's New**: 이 논문은 졸림 감지의 중요성을 다양한 응용 분야에서 검토하며, 단순한 운전자의 졸림 감지를 넘어 공공 교통, 의료, 직장 안전 등의 맥락을 포함한 종합적인 내용을 다룹니다.

- **Technical Details**: 본 논문은 졸림 감지를 위해 생리적(EEG, ECG) 및 육안 행동(눈 깜박임, 눈 감기) 신호를 활용하는 기술들을 탐구합니다. 현재의 방법론, 도전 과제, 그리고 기술 발전을 논의하며, 생리적 신호와 비전 기반 접근법에 따라 졸림 감지 기술을 구분합니다.

- **Performance Highlights**: 연구에서는 알고리즘의 약점과 기존 조사에서의 한계를 지적하고, 합성 데이터 사용, 모델 압축을 통한 하드웨어 한계 극복 및 성능 향상을 위한 융합 기법 등 실용적인 권장 사항을 제시합니다.



### Optimal OnTheFly Feedback Control of Event Sensors (https://arxiv.org/abs/2408.12976)
Comments:
          17 pages, 5 figures, ECCV 2024, NEVI workshop

- **What's New**: 이 논문에서는 Event-based Vision Sensors (EVS)를 사용하여 이벤트 스트림으로부터 비디오를 재구성하는 문제를 다루며, 동적 피드백 제어(dynamic feedback control)를 위해 최적의 활성화 임계값을 할당하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 먼저 과거 이벤트를 분석하여 추후의 활성화 임계값 분포를 예측함으로써 동적으로 피드백을 제공합니다. 이 접근법은 온도 기반 제어 네트워크(control network)가 최적의 임계값을 예측하고, 사용자 설정된 목표 피크 이벤트 레이트에 맞춰 최적화됩니다.

- **Performance Highlights**: 제안된 OnTheFly 제어 방식은 LPIPS 지표에서 고정 및 무작위 임계값 방식에 비해 6-12% 향상된 성능을 보였으며, 49% 더 높은 이벤트 비율을 달성하였습니다. 이는 재구성 품질을 향상시킴과 동시에 성능의 정확성과 이벤트 비율 사이의 정밀한 균형을 가능하게 합니다.



### Accuracy Improvement of Cell Image Segmentation Using Feedback Former (https://arxiv.org/abs/2408.12974)
Comments:
          Accepted by ECCV2024 Workshop "Human-inspired Computer Vision (HCV)"

- **What's New**: 최근에 Transformer가 이미지 인식에서 CNN을 초과하는 성능을 보임에 따라, 이들을 활용한 세포 이미지 분할을 위한 새로운 아키텍처, Feedback Former를 제안합니다. 이 모델은 인코더로 Transformer를, 피드백 처리 메커니즘을 포함하여 세부 정보 결손을 보완합니다.

- **Technical Details**: Feedback Former는 Human Visual Cortex의 피드백 처리에서 영감을 받아 세포 이미지 분할 정확도를 개선하는 Novel Architecture입니다. 이 모델은 Attention을 사용해 Transformer를 인코더로 활용하고, Feature Map에서 나온 세부 정보를 낮은 레이어로 피드백하여 성능 향상을 꾀합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 세포 이미지 데이터셋에서 실험을 통해 피드백 없는 기존 방법과 비교하여 최우수 정확도를 기록하며, 복잡한 계산 비용을 덜 소모하면서도 높은 정밀도를 제공했습니다. 특히, iRPE 데이터셋에서는 4.54%의 정확도 향상을 보여주었습니다.



### Image Segmentation in Foundation Model Era: A Survey (https://arxiv.org/abs/2408.12957)
Comments:
          A comprehensive survey of image segmentation in foundation model era (work in progress)

- **What's New**: 최근 이미지 세분화(image segmentation) 기술이 Foundation Models (FMs)이라는 새로운 패러다임으로 진화함에 따라, 기존의 방법론에 비해 더욱 향상된 성능과 새로운 기능을 제공하는 점에서 중요한 발전이 있습니다.

- **Technical Details**: 이 설문조사는 두 가지 주요 연구 방향인 일반 이미지 세분화(generic image segmentation)와 프롬프트 기반 이미지 세분화(promptable image segmentation)에 중점을 둡니다. FM을 활용해 기존의 세분화 접근 방식과 모델에 대한 총 300개 이상의 알고리즘을 포괄적으로 검토하였습니다.

- **Performance Highlights**: 세분화 일반화(generalization)와 프롬프트 반응(prompt response)의 향상 등 FM 기반 모델의 최첨단 기술은 기존 방법과 비교하여 더 높은 적응성과 성능을 보일 것으로 기대됩니다.



### State-of-the-Art Fails in the Art of Damage Detection (https://arxiv.org/abs/2408.12953)
- **What's New**: 이번 연구에서는 DamBench라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 15가지 손상 유형을 아우르는 11,000개 이상의 주석을 포함하여 다양한 아날로그 미디어에서 손상을 탐지하고 분류하는 데 도움을 줍니다.

- **Technical Details**:  연구팀은 CNN, Transformer, 그리고 텍스트 기반의 딥러닝 모델을 평가했습니다. 이 과정에서 기존의 세분화(semantic segmentation) 방법들이 다양한 미디어 타입에서 일반화하는 데 어려움을 겪는다는 사실이 밝혀졌습니다. 특히, 손상 탐지에 있어서 픽셀 수준의 정확한 주석(annotation) 및 텍스트 조건 방식의 세분화(segmentation)가 포함되었습니다.

- **Performance Highlights**: 결과적으로, 모든 평가된 모델들은 새로운 미디어 자료와 콘텐츠에 일반화하는 데 어려움을 겪었으며, 손상 구역을 정확히 분할하지 못하는 경향이 있었습니다. 이는 상태-of-the-art(상태-최고) 모델들이 손상 탐지라는 복잡한 작업을 완벽하게 수행하지 못한다는 것을 보여줍니다.



### Find the Assembly Mistakes: Error Segmentation for Industrial Applications (https://arxiv.org/abs/2408.12945)
Comments:
          23 pages (14 main paper, 2 references, 7 supplementary), 15 figures (8 main paper, 7 supplementary). Accepted at ECCV Vision-based InduStrial InspectiON (VISION) workshop

- **What's New**: 본 논문에서는 산업 응용을 위한 조립 오류 위치 추적을 위한 새로운 방법론인 StateDiffNet을 제안합니다. 기존 작업에서는 조립 상태 탐지만 집중해왔고, 조립 오류의 공간적 위치를 파악하는 데에는 주목하지 않았습니다.

- **Technical Details**: StateDiffNet은 두 개의 이미지 쌍을 기반으로 의도된 조립 상태와 유사한 시점에서 촬영된 테스트 이미지 간의 차이를 탐지하여 오류를 위치 추적합니다. 이 모델은 합성적으로 생성된 이미지 쌍에 대해 학습하였으며, 물체의 시점에서 의미 있는 변화를 추적하는 데 강력한 성능을 보입니다. 사용된 모델 구조는 U-Net 스타일의 Siamese 네트워크로, 원래의 경계 상자 헤드를 세그멘테이션 헤드로 변경했습니다.

- **Performance Highlights**: StateDiffNet은 실제 비디오 데이터에서 조립 오류를 세그멘트하는 데 성공하였고, 이전 학습 중에 결코 나타나지 않았던 다양한 오류 유형을 처리했습니다. 이를 통해 많은 산업 조립 절차에 유용하게 적용될 수 있음을 입증하였습니다.



### WildFusion: Individual Animal Identification with Calibrated Similarity Fusion (https://arxiv.org/abs/2408.12934)
- **What's New**: WildFusion이라는 새로운 방법이 소개되었으며, 이 방법은 여러 동물 종의 개체 식별을 위한 혁신적인 접근 방식을 제시합니다. 이 방법은 MegaDescriptor 또는 DINOv2와 같은 deep scores와 LoFTR 및 LightGlue와 같은 로컬 매칭 유사도를 융합하여 개체를 식별합니다.

- **Technical Details**: WildFusion은 zero-shot 설정에서 local similarity score만을 기반으로 평균 76.2%의 정확도를 달성하였으며, 이는 MegaDescriptor-L보다 우수한 결과입니다. 데이터셋별 보정이 적용되면 정밀도가 2.3%포인트 증가합니다. 로컬과 글로벌 유사도 점수를 모두 사용하는 WildFusion은 평균 84.0%의 정확도로 상태 최선의 결과를 달성하였으며, 이는 8.5%포인트 향상된 수치입니다.

- **Performance Highlights**: WildFusion은 13로부터 시작하여 17개의 데이터셋에서 평가되었으며, fine-tuning 없이도 state-of-the-art 성능을 보여주었습니다. 평균 상대 오차는 35% 감소하였습니다. 코드는 공개되어 생태학과 보존 과학 분야에서 즉각적인 사용이 가능합니다.



### Animal Identification with Independent Foreground and Background Modeling (https://arxiv.org/abs/2408.12930)
- **What's New**: 이번 논문에서는 시각적 개체 인식에서 배경과 전경을 효과적으로 활용하는 새로운 방법을 제안합니다. Segment Anything과 같은 방법으로 전경과 배경을 자동으로 분리하고, 이를 독립적으로 모델링함으로써 인식 성능을 향상시킵니다.

- **Technical Details**: 이 방법은 Per-Instance Temperature Scaling (PITS) 기법을 도입하여 분류기가 학습 시 외형의 모호성을 처리하고 정확한 결과를 도출할 수 있도록 도움을 줍니다. 배경에서의 정체성 예측을 위해 새로운 공간 및 시간 모델을 제안합니다. 배경과 전경 예측을 결합하기 위해서 분류기 보정이 필요합니다.

- **Performance Highlights**: 실험 결과, 두 개의 문제에서 기준선(baseline)에 비해 상대 오차가 각각 22.3%와 8.8% 감소하였으며, 객체가 새로운 위치에 나타나면 정확도가 두 배로 증가하는 결과를 보였습니다.



### ParGo: Bridging Vision-Language with Partial and Global Views (https://arxiv.org/abs/2408.12928)
- **What's New**: 본 연구에서는 Multimodal Large Language Models (MLLMs)용으로 설계된 새로운 Partial-Global projector인 ParGo를 제안합니다. 기존의 global attention 기반 projector와는 달리, ParGo는 사전 훈련된 시각 인코더와 LLM 간의 표현 격차를 해소하며, 글로벌 및 부분적 정보를 통합하여 뛰어난 기능을 발휘합니다.

- **Technical Details**: ParGo는 글로벌 정보와 부분 정보를 통합하는 부분-글로벌 주의 메커니즘을 기반으로 하여, 시각 인코더와 LLM 간의 연결을 향상시킵니다. 이를 위해 ParGo는 1백만 개의 이미지와 고품질 캡션이 포함된 대규모 detail-captioned 데이터셋인 ParGoCap-1M-PT를 수집하여 훈련합니다. 이 데이터셋은 모델이 세부적인 특징을 학습할 수 있도록 돕습니다.

- **Performance Highlights**: ParGo는 여러 MLLM 벤치마크에서 뛰어난 성능을 나타내며, 특히 Q-Former projector 대비 MME 벤치마크에서 259.96의 개선을 기록했습니다. 세부 인식 능력이 중요한 작업에서 ParGo는 다른 projector보다 현저히 우수한 성과를 보입니다.



### FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering (https://arxiv.org/abs/2408.12894)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 3D Gaussian Splatting(3DGS)의 유연한 레벨 오브 디테일(LoD) 통합을 통해 다양한 하드웨어 능력에 따라 장면을 렌더링할 수 있는 방법을 제안합니다. Flexible Level of Detail(FLoD) 개념을 도입하여 적은 수의 Gaussians로 메모리 요구를 줄이면서 렌더링 세부사항을 설정할 수 있게 합니다.

- **Technical Details**: FLoD는 각 레벨의 Gaussians에 적절한 디테일 정도를 설정하고, 레벨별 훈련 방법을 도입하여 모든 레벨에서 일관된 3D 구조를 유지합니다. 선택적 렌더링 방법을 통해 이미지의 일부를 다르게 렌더링하여 메모리 효율성을 높이고 품질 손실을 최소화합니다.

- **Performance Highlights**: FLoD는 Tanks and Temples와 Mip-Nerf360, DL3DV-10K와 같은 다양한 데이터셋에서 실험을 통해 다양한 렌더링 옵션의 효과성을 검증했습니다. 기존 모델과 비교하여 추가적인 계산 비용 없이 렌더링 품질을 향상시키는 것이 확인되었습니다.



### Unleashing the Potential of SAM2 for Biomedical Images and Videos: A Survey (https://arxiv.org/abs/2408.12889)
- **What's New**: 본 논문은 Segmentation Anything Model 2(SAM2)의 의료 영상 및 비디오에 대한 적용 및 적응을 탐구합니다. SAM2는 기존의 SAM을 기반으로 하여 의학 분야에서도 뛰어난 성능을 보여줄 가능성을 나타냅니다.

- **Technical Details**: SAM2는 Vision Transformer(ViT) 기반의 Image Encoder, Memory Attention 모듈, Prompt Encoder, Mask Decoder, Memory Encoder 등으로 구성되어 있어 동적인 영상 분석을 지원합니다. 3D 의료 이미지를 2D 슬라이스 시퀀스로 처리하는 혁신적인 접근 방식을 제안합니다.

- **Performance Highlights**: 여러 연구에서 SAM2는 수술 비디오 세분화, 특히 기존 기술보다 높은 성능을 보여줌으로써 의료 영상 분야에서의 가능성을 입증하였습니다. 그러나 3D 이미지 및 비디오에서의 변동성이 여전히 존재하며, 추후 최적화가 필요합니다.



### T3M: Text Guided 3D Human Motion Synthesis from Speech (https://arxiv.org/abs/2408.12885)
Comments:
          10 pages,4figures

- **What's New**: 본 논문에서는 텍스트 기반 3D 인체 동작 합성 방식인 T3M을 소개합니다. 기존의 음성 기반 접근 방식은 유연성이 부족하고 정확하지 않은 결과로 이어지는 한계가 있었으나, T3M은 텍스트 입력을 통해 동작을 정밀하게 제어할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: T3M 프레임워크는 VQ-VAE 네트워크, 오디오 피처 추출 네트워크, 멀티모달 융합 블록으로 구성되어 있습니다. VQ-VAE 네트워크를 통해 행동 매핑을 위한 중간 코드북을 생성하고, 오디오 피처를 추출하여 텍스트와의 상호작용을 위한 멀티모달 융합 인코더 구조를 가지고 있습니다.

- **Performance Highlights**: 실험 결과 T3M은 정량적 지표와 정성적 평가 모두에서 기존의 최신 방법들보다 뛰어난 성능을 보였습니다. 새로운 접근 방식으로 인한 제어성 향상 덕분에 보다 섬세하고 생동감 넘치는 동작 시퀀스를 생성할 수 있음을 보여주었습니다.



### Frequency-aware Feature Fusion for Dense Image Prediction (https://arxiv.org/abs/2408.12879)
Comments:
          Accepted by TPAMI (2024)

- **What's New**: 이 논문에서는 현재의 기법들이 갖고 있는 intra-category inconsistency(내부 범주 불일치) 및 boundary displacement(경계 이동)에 대한 해결책으로 FreqFusion(주파수 인식 특징 융합)이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: FreqFusion은 세 가지 주요 구성 요소인 Adaptive Low-Pass Filter (ALPF) generator, offset generator 및 Adaptive High-Pass Filter (AHPF) generator로 구성되어 있습니다. ALPF는 객체 내부의 고주파 성분을 억제하여 내부 불일치를 감소시키고, offset generator는 인접한 일관된 특징으로 교체하여 불일치를 수정하며, AHPF는 경계 세부 정보를 복원하여 더욱 선명한 경계를 제공합니다.

- **Performance Highlights**: FreqFusion은 여러 밀집 예측 작업에서 이전의 최첨단 기법들을 능가하는 성능 향상을 보여줍니다. 예를 들어, semantic segmentation(의미 기반 분할) 작업에서는 SegFormer-B1과 SegNeXt-T의 mIoU(Mean Intersection over Union)를 각각 2.8과 2.0만큼 향상시켰으며, object detection(객체 탐지)에서는 Faster R-CNN-R50의 AP(average precision)를 1.8 향상시켰습니다.



### Semantic Alignment for Multimodal Large Language Models (https://arxiv.org/abs/2408.12867)
Comments:
          Accepted by MM 2024

- **What's New**: 다양한 이미지를 다루는 멀티모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs) 연구에서 새로운 접근 방식인 Semantic Alignment for Multimodal large language models (SAM)를 소개합니다. SAM은 시각적 토큰 추출 과정에서 상호 참조하는 이미지 간의 의미적 정렬을 포함하여 데이터의 연관 정보를 보존하도록 설계되었습니다.

- **Technical Details**: SAM 모델은 두 개의 상호 작용 단계로 구성됩니다: (1) 현재 인지하고 있는 이미지에서 시각적 토큰을 추출하며, 이때 다른 이미지에서의 맥락 의미(contextual semantics)를 가이드로 사용합니다. (2) W-former라는 혁신적인 시각 토크나이저를 통해 여러 이미지의 동기화된 의미 정보를 추출하여, 현재 인지되고 있는 이미지의 시각적 토큰 인식을 돕습니다. 이 과정에서 다중 이미지 샘플의 다각화된 시각적 특징을 고려하여 의미적 정렬을 실현합니다.

- **Performance Highlights**: SAM 모델은 그룹 캡셔닝 작업(group captioning task) 및 스토리텔링 작업(storytelling task)에서 뛰어난 성능을 보여주며, 기존 최첨단 방법들보다 각각 37% 및 22% 더 높은 CIDEr 점수를 기록했습니다. 이를 통해 SAM 모델이 MLLMs의 효과적인 진전을 이끌 수 있음을 증명하였습니다.



### Underwater SONAR Image Classification and Analysis using LIME-based Explainable Artificial Intelligenc (https://arxiv.org/abs/2408.12837)
Comments:
          55 pages, 9 tables, 18 figures

- **What's New**: 이 연구는 SONAR 이미지를 분류하기 위한 Explainable AI(XAI) 기술의 적용에 대해 다루고 있으며, 이는 해당 분야의 최초 연구 중 하나입니다. 연구자는 다양한 공개 SONAR 데이터셋을 결합하여 새로운 맞춤형 데이터셋을 개발하고, Transfer Learning을 통해 여러 CNN 모델의 성능을 비교 분석했습니다.

- **Technical Details**: 이 논문에서는 VGG16, ResNet50, InceptionV3, DenseNet121 등의 Convolutional Neural Network(CNN) 아키텍처를 기반으로 이미지를 분류하는 Transfer Learning 기법의 활용을 조사합니다. 또한, Local Interpretable Model-Agnostic Explanations(LIME)와 Submodular Picks LIME(SP-LIME) 기술을 이용하여 모델의 결정을 해석하는 방법을 제시합니다. SP-LIME은 이미지에 특화된 LIME의 변형으로, quickshift와 SLIC 알고리즘을 사용하여 초픽셀을 변화시키는 방식으로 작동합니다.

- **Performance Highlights**: Transfer Learning을 활용한 이미지 분류의 정확도는 최대 97.33%에 달하며, XAI 기술을 통해 모델의 결정 과정을 시각적으로 설명하였습니다. 이 연구는 XAI가 SONAR 이미지 분석에 기여할 수 있는 잠재력을 보여주며, 해양 감시 및 탐사 작업의 효율성을 높이는 데 기여할 수 있습니다.



### S3Simulator: A benchmarking Side Scan Sonar Simulator dataset for Underwater Image Analysis (https://arxiv.org/abs/2408.12833)
- **What's New**: 본 연구는 폭넓은 접근성을 가진 'S3Simulator dataset'라는 새로운 벤치마크 데이터셋을 제안하여 수중 탐사 및 이미지 분석에 대한 AI 모델 훈련의 데이터 부족 문제를 해결하고자 합니다.

- **Technical Details**: S3Simulator dataset는 고급 시뮬레이션 기술을 활용하여 수중 조건을 정확히 재현하고, Segment Anything Model (SAM)과 SelfCAD, Gazebo 같은 최첨단 도구들을 통합하여 600장의 선박과 600장의 비행기를 포함하는 합성 이미지 세트를 생성하였습니다.

- **Performance Highlights**: S3Simulator 데이터셋을 활용한 실험 결과, 기존의 머신러닝(Machine Learning) 및 딥러닝(Deep Learning) 모델들이 수중 객체 분류 작업에서 높은 성능을 발휘했으며, 이 데이터셋은 향후 수중 이미지 분석 연구에 유망한 기준이 될 것으로 기대됩니다.



### MergeUp-augmented Semi-Weakly Supervised Learning for WSI Classification (https://arxiv.org/abs/2408.12825)
- **What's New**: 본 논문에서는 Whole Slide Image (WSI) 분류를 위한 새로운 접근 방식인 SWS-MIL(세미 약한 감독 학습) 프레임워크를 제안합니다. 이 방법은 Adaptive Pseudo Bag Augmentation(AdaPse)를 사용하여 레이블이 있는 데이터와 레이블이 없는 데이터를 더 효율적으로 분리합니다.

- **Technical Details**: 기존의 약한 감독 학습(MIL) 접근 방식의 한계점을 극복하기 위해, 본 논문에서는 AdaPse를 통해 pseudo bags의 레이블 할당과 수 할당을 분리하여 더 나은 정보 추출을 가능하게 합니다. 또한 MergeUp이라는 새로운 feature augmentation 기술을 도입하여 모델이 다양한 범주 간의 상호작용을 학습하게 합니다.

- **Performance Highlights**: CAMELYON-16, BRACS, TCGA-LUNG 데이터셋에서의 실험 결과, SWS-MIL 방법은 기존의 최첨단 방법들을 초월하여 더욱 뛰어난 성능을 입증하였습니다.



### Examining the Commitments and Difficulties Inherent in Multimodal Foundation Models for Street View Imagery (https://arxiv.org/abs/2408.12821)
- **What's New**: 이번 연구는 Minimal Modal Models (LLMs)인 ChatGPT-4V와 Gemini Pro의 실제 활용 가능성을 평가했습니다. 특히, 거리 이미지, 건축 환경, 실내 디자인 분석에서 이 모델들이 가진 강점과 약점을 찾아내었습니다.

- **Technical Details**: 본 논문은 거리 이미지에서 가구 인식, 보행자 및 차량 수 측정, 도로 폭 측정 임무를 수행하며, 건축 환경에서는 건물 기능 분류, 연령 및 높이 분석, 구조 분류를 수행했습니다. 실내의 경우 방 유형 분류와 디자인 스타일 분석을 포함한 다양한 작업을 평가했습니다. 연구 결과, 길이 측정, 스타일 분석 및 기본적인 이미지 이해 능력에서 우수한 성과를 거두었으나, 세부 인식 및 수 카운트 작업에서는 한계를 보였습니다.

- **Performance Highlights**: Zero-shot learning(제로샷 학습)에서 잠재력을 보였지만, 성능은 문제 도메인과 이미지 복잡도에 따라 달라졌습니다. multimodal foundation models (FMs)는 컴퓨터 비전과 언어의 융합에 잠재력을 열어 주었습니다.



### O-Mamba: O-shape State-Space Model for Underwater Image Enhancemen (https://arxiv.org/abs/2408.12816)
- **What's New**: 이번 연구에서는 수중 이미지 개선(Underwater Image Enhancement, UIE)을 위한 새로운 프레임워크인 O-Mamba를 제안합니다. O-Mamba는 O자형 듀얼 브랜치 네트워크를 활용하여 공간 정보와 색상 간의 상호작용을 분리해서 모델링합니다.

- **Technical Details**: O-Mamba는 Spatial Mamba(SM)와 Channel Mamba(CM) 블록을 통해 공간 및 교차 채널 특징을 모델링합니다. 또한 Multi-scale Bi-mutual Promotion(MSBMP) 모듈을 설계하여 두 브랜치 간의 정보 상호작용을 촉진하고, 다양한 스케일 정보를 효과적으로 활용합니다.

- **Performance Highlights**: O-Mamba는 광범위한 실험을 통해 기존 UIE 방법을 초월하는 성능을 달성하며, 매우 우수한 성능을 입증했습니다.



### Staircase Cascaded Fusion of Lightweight Local Pattern Recognition and Long-Range Dependencies for Structural Crack Segmentation (https://arxiv.org/abs/2408.12815)
- **What's New**: 이 논문에서는 경량화된 계산 자원으로 고품질의 균열(segmentation maps)을 생성할 수 있는 계단형 연결 융합 네트워크(CrackSCF)를 제안합니다. 이 네트워크는 지역적 패턴과 장거리 픽셀 종속성을 잘 포착하는 새로운 모듈을 포함하고 있으며, 배경 잡음을 효과적으로 억제합니다.

- **Technical Details**: CrackSCF 네트워크는 계단형 연결 융합 모듈과 경량화된 합성곱 블록(LRDS)을 도입하여, 픽셀 수준에서 균열을 정확하게 분류합니다. 이 모델은 이미지에서 지역적 및 장거리 정보를 세밀하게 포착하고, 인접 레이어의 특성을 점진적으로 통합합니다. 또한, TUT라는 새로운 벤치마크 데이터셋을 구축하여 다양한 시나리오에서 실험을 수행했습니다.

- **Performance Highlights**: TUT 데이터셋에서 CrackSCF 네트워크는 F1 스코어 0.8382, mIoU 스코어 0.8473을 기록하며, 기존 방법들보다 우수한 성능을 보임과 동시에 가장 적은 계산 자원을 요구합니다. 실험은 다섯 개의 공개 데이터셋에서 진행되었으며, CrackSCF 네트워크가 기존 최첨단(SOTA) 방법들보다 더 나은 결과를 보여주었습니다.



### From Few to More: Scribble-based Medical Image Segmentation via Masked Context Modeling and Continuous Pseudo Labels (https://arxiv.org/abs/2408.12814)
- **What's New**: 이번 논문에서는 'from few to more'의 원리를 채택하여 의료 영상 분할을 위한 약한 감독 프레임워크인 MaCo를 제안합니다. 이는 기존의 방법들이 간과했던 희소 주석의 요구 사항을 충족하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MaCo는 두 가지 핵심 구성 요소인 Masked Context Modeling (MCM)과 Continuous Pseudo Labels (CPL)을 사용하여 모델의 성능을 향상시킵니다. MCM은 주석이 포함된 패치에 더 높은 가중치를 부여하고, CPL은 거리 맵에 대한 지수 감쇠 함수를 적용하여 연속적인 카테고리별 레이블로 변환하는 방식을 사용합니다.

- **Performance Highlights**: MaCo는 세 가지 공개 데이터셋에서 다른 약한 감독 방법들과 비교 평가되었으며, 모든 데이터셋에서 우수한 성능을 보여주어 기존의 방법들을 초월하는 결과를 나타냈습니다. 또한, MaCo는 훈련 샘플이나 스크리블 픽셀이 적은 더 어려운 조건에서도 강건성을 유지합니다.



### VALE: A Multimodal Visual and Language Explanation Framework for Image Classifiers using eXplainable AI and Language Models (https://arxiv.org/abs/2408.12808)
Comments:
          15 pages, 10 tables, 3 figures

- **What's New**: 이번 논문에서는 기존의 비유사한 방법들과 차별화된 다채로운 비주얼과 언어 설명 프레임워크인 VALE(Visual and Language Explanation)를 제안합니다. VALE는 이미지 분류 작업에서 DNN의 내재된 작동을 해석하고 설명하는 데 도움을 주며, 시각적 및 텍스트 기반의 설명을 통합하여 인간 친화적인 방식으로 결과를 전달합니다.

- **Technical Details**: VALE 시스템은 이미지 분류기, 설명기(SHAP), 이미지 분할기(Segment Anything Model), 그리고 이미지-텍스트 설명기(VLM)로 구성되어 있습니다. SHTA(clickable) 점수 계산을 통해 분류 이미지의 가장 영향력 있는 영역을 식별한 후, Segment Anything Model을 사용해 객체의 영역을 추출하고, VLM을 통해 시각적 설명에 대한 텍스트 설명을 생성합니다.

- **Performance Highlights**: VALE는 ImageNet 데이터셋 및 맞춤형 수중 SONAR 이미지 데이터셋에서 광범위한 실험을 통해 실용성을 입증했습니다. 이 프레임워크는 기존 XAI 도구를 통합하여 각 이미지 분류에서 보다 명확하고 이해하기 쉬운 결과를 제공합니다.



### La-SoftMoE CLIP for Unified Physical-Digital Face Attack Detection (https://arxiv.org/abs/2408.12793)
- **What's New**: 본 논문에서는 물리적 공격(Physical Attacks, PAs)과 디지털 공격(Digital Attacks, DAs)의 통합 탐지를 위해 새로운 접근 방식을 제안합니다. 기존 방법들과 달리, 우리는 두 공격 유형을 각각 처리하는 대신 Sparse 모델과 Mixture of Experts (MoE) 구조를 활용하여 더 효과적으로 탐지할 수 있는 방법을 개발하였습니다.

- **Technical Details**: 제안된 La-SoftMoE CLIP 모델은 Sparse feature space를 처리하기 위해 MoE 프레임워크를 이용하고, 훈련 중 다양한 가중치로 토큰에 맞춰 전문 파라미터를 조정하여 테스트 중에 적응적으로 활성화됩니다. 우리는 SoftMoE를 CLIP의 이미지 인코더에 도입하고, 가중치 방법을 Linear Attention으로 대체하여 La-SoftMoE로 명명하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 La-SoftMoE CLIP 모델은 Unified Attack Detection (UAD) 작업에서 SOTA(State Of The Art) 성능을 달성하였으며, 다양한 공격을 효과적으로 처리할 수 있는 능력을 크게 향상시켰습니다.



### Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method with Forgery Style Mixtur (https://arxiv.org/abs/2408.12791)
- **What's New**: 이 논문은 Open-set 얼굴 위조 탐지(Open-Set Face Forgery Detection)에 대한 새로운 접근법을 제안합니다. 기존 모델의 한계인 알려지지 않은 위조 도메인에서의 일반화 부족과 새로운 데이터에의 비효율적인 적응 문제를 해결하기 위해, 파라미터 효율적인 방법으로 얼굴 위조를 탐지합니다.

- **Technical Details**: 이 연구에서는 다양한 위조 소스 도메인이 각각 고유한 스타일 통계(statistics)를 가지고 있다는 가정을 바탕으로, 다양한 위조 스타일을 혼합하는 방법을 설계하였습니다. ViT(비전 변환기) 구조에 가벼운 Adapter 및 Low-Rank Adaptation(LoRA) 모듈을 통합하여 훈련 중에 원래의 구조를 유지하면서도 피처를 효율적으로 추출합니다.

- **Performance Highlights**: 실험 결과, 본 모델은 활성화된 파라미터가 1.34M로 크게 줄어들면서도 최첨단 일반화 성능을 달성하였음을 보여주었습니다. 이는 Open-set Deepfake 탐지 분야에서 중요한 기여를 나타냅니다.



### Context-Aware Temporal Embedding of Objects in Video Data (https://arxiv.org/abs/2408.12789)
- **What's New**: 본 논문에서는 비디오 분석을 위한 새로운 모델을 제안하며, 이는 객체 간의 인접성과 의미적 유사성을 활용하여 컨텍스트 인식 임베딩을 생성합니다. 전통적인 접근 방식과는 달리, 이 모델은 시각적 외형만을 고려하지 않고 객체 간의 관계를 반영하여 임베딩 공간을 구축합니다.

- **Technical Details**: 제안된 모델은 영상의 각 프레임에서 검출된 객체들 간의 거리와 빈도를 활용하여 동적인 컨텍스트 인식 임베딩을 학습합니다. 또한, 이는 각 객체의 시간에 따른 변화를 추적하여 비디오의 내러티브를 구성하는 데 도움을 줍니다. 효율적인 동적 임베딩을 위해, CNN(Convolutional Neural Network) 모델을 사용하여 비디오 프레임을 전처리하고, 신경망 모델을 통해 임베딩 벡터를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 컨텍스트 인식 임베딩은 기존의 시각적 임베딩과 결합하여 다운 스트림 애플리케이션의 효과성을 향상시킬 수 있음을 보여주었습니다. 또한, 이 임베딩은 Large Language Model(LLM)을 사용하여 비디오 내러티브를 서술하는 데에도 활용될 수 있습니다.



### Symmetric masking strategy enhances the performance of Masked Image Modeling (https://arxiv.org/abs/2408.12772)
Comments:
          Accepted at ICPR 2024

- **What's New**: 본 연구에서는 Masked Image Modeling (MIM) 방식의 새로운 마스킹 전략을 제안하여 모델이 글로벌(global) 및 로컬(local) 특징을 효과적으로 캡처할 수 있도록 합니다. 이를 기반으로 SymMIM이라는 새로운 학습 파이프라인을 도입했습니다.

- **Technical Details**: SymMIM은 대칭 마스킹 전략을 기반으로 하며, 수직 및 수평축을 따라 마스킹된 패치를 설정하여 각 마스킹 패치가 유사한 의미와 구조 정보를 포함하는 보이는 패치에 대응하게 합니다. 또한, 대조적 손실(contrastive loss)을 활용하여 전역 및 지역 특징 간의 일관성을 촉진합니다.

- **Performance Highlights**: SymMIM은 ViT-Large를 사용하여 ImageNet에서 85.9%의 새로운 SOTA 정확도를 달성하였으며, 이미지 분류, 객체 탐지 및 인스턴스 분할 등의 다양한 다운스트림 태스크에서 이전 SOTA를 초과하는 성능을 보였습니다.



### Enhancing Vehicle Environmental Awareness via Federated Learning and Automatic Labeling (https://arxiv.org/abs/2408.12769)
- **What's New**: 이 논문은 차량 간 통신 데이터와 이미지 데이터를 통합하여 차량 식별 문제를 다루는 새로운 접근 방식을 제안합니다. 특히, 프라이버시 이슈와 데이터 레이블링의 어려움을 해결하기 위해 federated learning과 자동 레이블링 기술을 결합한 FedMDFNN 모델을 도입하였습니다.

- **Technical Details**: FedMDFNN 모델은 차량을 로컬 노드로 취급하여 각 차량이 자체적으로 수집한 데이터를 기반으로 LOCAL MDFNN을 훈련합니다. 이 과정에서 차량은 개인 정보 보호를 위해 데이터가 아닌 모델 매개변수만 서버에 전송합니다. 자동 레이블링 기법은 차량 번호판 인식 및 데이터 증강 기법을 활용하여 훈련 데이터를 생성합니다.

- **Performance Highlights**: 제안된 FedMDFNN 모델은 실험을 통해 그 가능성을 검증하였으며, 데이터 프라이버시를 유지하면서도 차량 식별 성능이 향상되는 것을 보여주었습니다.



### CatFree3D: Category-agnostic 3D Object Detection with Diffusion (https://arxiv.org/abs/2408.12747)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 3D 물체 탐지를 위한 새로운 파이프라인을 도입하였습니다. 이 파이프라인은 3D 탐지 작업을 2D 탐지와 깊이 예측으로부터 분리하며, diffusive 기반 접근 방식을 사용하여 정확성을 개선하고 범주에 구애받지 않는 탐지를 지원합니다.

- **Technical Details**: 제안된 방법은 입력 이미지로부터 3D 바운딩 박스를 랜덤 노이즈에서 회복하는 과정을 다루고 있으며, 이는 여러 시각적 프롬프트에 의해 조건화됩니다. 이를 통해 여러 3D 바운딩 박스를 생성하고 각 박스에 대한 신뢰 점수를 할당하여 가장 신뢰할 수 있는 것을 선택합니다. 또한, Normalised Hungarian Distance (NHD)라는 새로운 평가 지표를 사용하여 3D 탐지 결과를 더 정확하게 평가합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 여러 물체 범주 및 데이터셋에서 최첨단의 정확도를 달성하며, 이전에 보지 못한 데이터셋에 대해서도 강력한 일반화를 보여줍니다.



### Segment Anything Model for Grain Characterization in Hard Drive Design (https://arxiv.org/abs/2408.12732)
Comments:
          This paper has been accepted by the International Workshop on Computer Vision for Materials Science in conjunction with the IEEE/CVF CVPR 2024

- **What's New**: 본 연구에서는 Meta의 Segment Anything Model (SAM)을 사용하여 나노스케일 재료의 특성화 및 그레인 세분화(grain segmentation)에 적용 가능한 새로운 방법을 모색합니다.

- **Technical Details**: SAM은 미세구조의 전반적인 세분화를 위해 자동 마스크 생성기(Automatic Mask Generator, AMG)를 사용하며, 기존의 손으로 라벨링한 데이터를 최소화하면서도 다양한 이미지 조건에 유연하게 대응할 수 있습니다. 데이터셋은 금 필름 웨이퍼에서 수집된 5개의 SEM(Scanning Electron Microscopy) 이미지로 구성되었습니다.

- **Performance Highlights**: 초기 테스트 결과, SAM의 out-of-the-box 성능은 고무적인 정확성을 보여주며, 분포 추출(property distribution extraction)에서도 일관된 결과를 나타냈습니다. 또한, 최적의 프롬프트 포인트(prompt point) 선택을 통해 2-8%의 추가적인 그레인을 찾을 수 있는 가능성이 있음을 확인했습니다.



### BankTweak: Adversarial Attack against Multi-Object Trackers by Manipulating Feature Banks (https://arxiv.org/abs/2408.12727)
- **What's New**: 이 논문에서는 Multi-object tracking (MOT) 시스템을 겨냥한 새로운 적대적 공격 기법인 	extsf{BankTweak}을 제안합니다. 	extsf{BankTweak}은 기존의 공격 방식의 비효율성과 취약성을 해결하고, 공격 후에도 지속적인 ID 전환을 유도하는 효율성과 견고성을 갖춘 방법입니다.

- **Technical Details**: 	extsf{BankTweak}은 Feature extractor에 집중하여, 아소시에이션 단계에서 헝가리안 매칭 알고리즘의 취약점을 이용합니다. 이 방법은 객체 위치를 수정하지 않고도 새로운 특징을 feature bank에 주입하여 지속적인 ID 전환을 유도합니다. 	extsf{BankTweak}은 데이터 세트인 MOT17과 MOT20에서 DeepSORT, StrongSORT, MOTDT의 세 가지 멀티 객체 추적기에 적용되었습니다.

- **Performance Highlights**: 실험 결과, 	extsf{BankTweak}은 기존 공격 방법들보다 상당히 우수한 성능을 보여주었으며, 추적-탐지 프레임워크의 취약성을 드러냈습니다. 	extsf{BankTweak}은 오탐지 없이 공격을 수행할 수 있어 실용성과 일반성을 보장합니다.



### Revisiting Cross-Domain Problem for LiDAR-based 3D Object Detection (https://arxiv.org/abs/2408.12708)
Comments:
          Accepted by the ICONIP 2024

- **What's New**: 본 논문에서는 자율주행 분야에서 3D 물체 탐지를 위한 최신 모델의 크로스 도메인 성능을 깊이 분석하고, 기존 도메인 적응 방법이 실제로는 모델의 일반화 능력을 향상시키는 대신 지식 도메인만 이동하는 것을 확인했습니다. 또한 새로운 평가 지표인 side-view AP와 front-view AP를 제안하여 성능 저하의 주요 원인을 더 잘 분석할 수 있도록 하였습니다.

- **Technical Details**: 3D 물체 탐지는 주로 LiDAR 포인트 클라우드 데이터와 같은 3D 센서 데이터를 기반으로 서로 다른 유형의 객체를 특정 3D 공간에서 로컬라이즈하고 분류하는 것을 목표로 합니다. 현재 모델들은 서로 다른 데이터셋에 대해 독립적으로 훈련 및 테스트되며, 다른 도메인(예: 다른 도시, 날씨)으로의 직접적인 적응이 어렵다는 문제를 가지고 있습니다. 본 연구에서는 기존 SOTA(self-training) 방법 ST3D를 분석하여, 이 방법이 실제로는 모델의 지식 분포를 새로운 도메인으로 이동시켜 오히려 소스 도메인에서의 탐지 능력을 감소시킨다는 심각한 문제를 발견하였습니다.

- **Performance Highlights**: 테스트된 모든 방법은 크로스 도메인 과제에서 유사한 성능 저하를 보였으며, 멀티 모달 방법이 LiDAR 만의 방법보다 일부 작업에서 성능이 떨어지는 것을 관찰했습니다. 새롭게 제안된 side-view AP와 front-view AP 평가 지표를 통해 성능 비교를 수행한 결과, 길이 차원에서의 예측 정확도가 너비보다 높았고, 이는 모델의 과적합 문제로 인해 발생한 것으로 보여집니다.



### GSFusion: Online RGB-D Mapping Where Gaussian Splatting Meets TSDF Fusion (https://arxiv.org/abs/2408.12677)
- **What's New**: 이번 연구에서는 GSFusion이라는 하이브리드 매핑 시스템을 소개합니다. 이 시스템은 Gaussian splatting과 TSDF 융합을 조합하여 실시간으로 두 가지 유형의 맵을 동시에 생성할 수 있도록 합니다.

- **Technical Details**: GSFusion은 쿼드트리(quadtree) 분할 방식을 활용하여 입력 이미지를 다양한 크기의 셀로 나누고, 각 셀의 중심에 3D Gaussian을 초기화합니다. 이는 새로운 Gaussian을 추가할 위치를 결정하는 데 기여하고, 효율적인 온라인 최적화를 가능하게 합니다.

- **Performance Highlights**: GSFusion은 합성 및 실세계 데이터셋에서 뛰어난 성능을 보여주었습니다. 기존의 Gaussian 기반 접근 방식과 비교하여 매핑 빈도와 렌더링 품질의 균형을 효과적으로 유지하는 것을 입증하였습니다.



### Building and better understanding vision-language models: insights and future directions (https://arxiv.org/abs/2408.12637)
- **What's New**: 최근 Vision-Language Models (VLMs)의 발전에 대한 포괄적인 개요를 제공하며, 기존 모델들의 강점과 약점을 분석하고 새로운 연구 방향을 제시합니다. 특히, Idefics3-8B라는 강력한 VLM 모델의 구축 방법과 그 데이터셋인 Docmatix를 소개합니다.

- **Technical Details**: 이 논문에서는 데이터, 아키텍처(architecture), 학습 방법(training methods) 등 VLM 개발 파이프라인의 핵심 요소들에 대한 다양한 설계 선택들을 분석합니다. 특히, Idefics3-8B 모델의 구축에 필요한 단계로는 새로운 Docmatix 데이터셋을 생성하는 과정이 포함되며, 이는 이전 데이터셋보다 240배 큰 규모입니다.

- **Performance Highlights**: Idefics3-8B 모델은 Idefics2-8B에 비해 문서 이해 작업에서 13.7 포인트의 성능 향상을 이루었습니다. 새로운 모델은 오픈 데이터셋만을 사용하여 효율적으로 학습되었습니다.



### Data-Free Class Incremental Gesture Recognition via Synthetic Feature Sampling (https://arxiv.org/abs/2408.12629)
- **What's New**: 본 연구는 Data-Free Class Incremental Learning (DFCIL) 방법론을 토대로 스켈레톤 기반 제스처 인식 분야에 대한 새로운 접근 방식을 제시합니다. 기존 연구에서는 주로 이미지 데이터셋을 중심으로 DFCIL을 탐구하였으나, 본 연구에서는 VR/AR 환경에서의 제스처 인식의 필요성에 주목하여 관련 연구를 진행했습니다.

- **Technical Details**: 우리는 Synthetic Feature Replay (SFR) 알고리즘을 제안하여 클래스 프로토타입으로부터 합성 feature를 샘플링하고 이를 사용해 이전 클래스의 replay와 새로운 클래스의 augmentation을 수행합니다. 기존 DFCIL 방법들이 어려운 합성 데이터 생성을 필요로 하는 반면, SFR은 신속하고 단순하며 모든 테스트된 데이터셋에서 현상 유지 기반 최첨단 방법들에 비해 상당한 성능 향상을 나타냅니다.

- **Performance Highlights**: 우리의 방법론은 평균 정확도를 최대 15% 향상시키며, 기존의 기본 클래스와 새로운 클래스 간의 정확도 불균형 문제를 효과적으로 완화합니다.



### Can GPT-4 Models Detect Misleading Visualizations? (https://arxiv.org/abs/2408.12617)
Comments:
          5 pages, 2 figures; accepted by IEEE VIS 2024 (this https URL)

- **What's New**: 이번 연구는 GPT-4 모델(4V, 4o, 4o mini)의 misleading visualizations(오해를 일으키는 시각적 표현) 탐지 능력을 조사합니다. 특히, 기존의 훈련 없이도 중간 정도의 정확도로 시각적 오류를 감지할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 9,958개의 tweet-visualization pair(트윗-시각화 쌍)에서 수집한 데이터셋을 활용하여, misleaders(오도하는 요소)을 Design Misleaders(디자인 오도 요소)와 Reasoning Misleaders(추론 오도 요소)의 두 가지 범주로 구분합니다. 각 모델은 다양한 실험 조건에서 prompt engineering(프롬프트 엔지니어링)을 활용하여 성능을 평가받았습니다.

- **Performance Highlights**: 모델의 성능은 guided zero-shot 설정에서 가장 높았으며, Reasoning Misleaders의 경우, Causal Inference(인과 추론)에서 AUC score(곡선 아래 면적 점수) 0.872를 기록했습니다. Dual Axis(이중 축) 디자인 오류의 경우, Guided Few-Shot 설정에서 0.970의 AUC score를 달성하며, overall(전체적으로) 모델의 성능 향상을 강조했습니다.



### Semantic Communication based on Large Language Model for Underwater Image Transmission (https://arxiv.org/abs/2408.12616)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 기반으로 하는 새로운 의미적 통신(Semantic Communication, SC) 프레임워크를 제안합니다. 이는 수중 이미지 데이터를 사용자 쿼리에 따라 의미적으로 압축하고 우선순위를 정하는데 도움을 줍니다.

- **Technical Details**: SC 프레임워크는 시각적 LLM을 활용하여 이미지를 분석하고 중요 의미 요소를 식별하여 전송합니다. 수신 측에서는 LLM 기반 복구 메커니즘과 Global Vision ControlNet 및 Key Region ControlNet 네트워크를 활용하여 이미지를 재구성하여 통신 효율성을 높입니다.

- **Performance Highlights**: 이 프레임워크는 전체 데이터 크기를 원본의 0.8%로 줄이며, 기존 방법보다 훨씬 높은 품질의 의미적 이미지 복원을 보장합니다. 실험 결과에 따르면, 제안된 방법이 높은 통신 효율성과 강건성을 달성한다는 것이 입증되었습니다.



### Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning (https://arxiv.org/abs/2408.12614)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 논문에서는 이미지 수준의 약한-강한 일관성(weak-to-strong consistency)에서 탈피하여 이미지-특징(feature) 수준의 약한-강한 일관성(Image-Feature Weak-to-Strong Consistency, IFMatch) 패러다임을 도입합니다. 이를 통해 다양한 강도와 형태의 특징 수준의 왜곡(feature-level perturbation)을 적용하여 데이터 증강(augmentation) 공간을 확장하고, 기존의 반지도 학습(semi-supervised learning, SSL) 방법들과 원활하게 통합될 수 있도록 합니다.

- **Technical Details**: IFMatch는 세 가지 가지(branch)를 포함하는 구조로, 강한 이미지 수준의 왜곡과 특징 수준의 왜곡 간의 상호작용을 촉진하여 이들의 시너지를 높입니다. 여기서 특징 수준의 왜곡은 중간 특징을 무작위로 변경하는 방식으로 진행되며, 약한 이미지 수준과 강한 특징 수준의 왜곡을 결합하여 효율적인 특징 수준의 일관성 조정을 수행합니다. 또한, 신뢰 기반의 식별 전략(confidence-based identification strategy)을 제안하여 순진한 샘플(naive samples)과 도전적인 샘플(challenging samples)을 구별할 수 있도록 하였습니다.

- **Performance Highlights**: 제안된 IFMatch 패러다임은 여러 대표적인 반지도 학습 알고리즘에 적용되었으며, 균형 잡힌 샘플과 불균형 샘플 모두에 대해 실험을 진행하였습니다. 그 결과, 기존의 SSL 알고리즘의 성능이 유의미하게 향상된 것으로 나타났습니다.



### Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Mod (https://arxiv.org/abs/2408.12606)
Comments:
          27 pages, 8 figures, 10 tables

- **What's New**: 이번 연구에서는 다중 파라미터 유방 자기공명영상(MRI) 데이터를 통합하여 비침습적이고 개인화된 유방암 관리 방법을 제공하는 대규모 혼합 모달 전문가 모델(MOME)을 개발했습니다. 이를 위해 중국의 세 개 병원에서 수집한 5,205명의 환자를 대상으로 가장 큰 규모의 데이터셋을 구축하였습니다.

- **Technical Details**: MOME는 데이터의 이질성과 고차원성을 다루기 위해 다양한 유형의 훈련 가능한 전문가 모듈을 적용한 Transformer 구조로 설계되었습니다. 이 모델은 병리학적 완전 반응 예측을 포함하여 유방암 환자의 악성 종양 식별에서 방사선 전문의의 성과와 유사한 정확도를 달성하였습니다.

- **Performance Highlights**: MOME는 내부 테스트 세트에서 0.913 AUROC, 0.948 AUPRC, 0.905 F1 점수를 기록하며, BI-RADS 4 환자의 바이옵시 필요성을 7.3% 줄이는 가능성을 보였습니다. 모델은 강화된 정확도로 악성 및 양성 종양을 구분할 수 있습니다.



### How Diffusion Models Learn to Factorize and Compos (https://arxiv.org/abs/2408.13256)
Comments:
          11 pages, 6 figures, plus appendix, some content overlap with arXiv:2402.03305

- **What's New**: 본 논문은 Diffusion 모델들이 데이터의 구성 요소를 조합하고 일반화하는 능력을 가지고 있음을 보여주지만, 이러한 구성 가능성의 내부 메커니즘은 여전히 불분명하다는 점을 강조합니다. 모델이 의미 있는 표현(meaningful representations)을 어떻게 학습하는지를 심층적으로 분석하기 위해 2D Gaussian 데이터셋에서 수행된 실험을 통해 결과를 공유합니다.

- **Technical Details**: Conditional Denoising Diffusion Probabilistic Models (DDPMs)의 통제된 실험을 수행하여 모델의 표현력이 데이터의 변화 기저(underlying features)에 대한 요인화된(factorized) 표현을 학습하는 방식을 연구했습니다. 실험에서는 다양한 파라미터화(parameterization) 기술을 사용하여 저차원 데이터셋에서 모델의 성능을 평가하고, 모델이 조합할 수 있는 특정 구조의 데이터를 생성하는 능력을 분석했습니다.

- **Performance Highlights**: 모델은 완전 연속적인 다중체계(manifold) 표현을 학습하는 데는 한계가 있었지만, 여러 조합의 예제를 통해 고유한 구성 요소를 잘 조합할 수 있고, 일반화 능력이 상당히 향상된 결과를 보였습니다. 이는 학습에 필요한 구성 예제가 적다는 것을 시사하며, Diffusion 모델의 교육 방법에 대한 새로운 접근 방식을 제안합니다.



### Deep Learning for Lung Disease Classification Using Transfer Learning and a Customized CNN Architecture with Attention (https://arxiv.org/abs/2408.13180)
- **What's New**: 이 연구는 Lung X-ray 이미지를 이용한 폐 질환 분류에 관한 것으로, 다섯 가지 사전 훈련된 (pre-trained) 모델의 성능을 비교하고, MobileNetV2를 기반으로 한 새로운 모델 MobileNet-Lung을 제안합니다.

- **Technical Details**: 연구에서는 SqueezeNet, VGG11, ResNet18, DenseNet, MobileNetV2 등 다섯 가지 CNN (Convolutional Neural Network) 모델을 검토하였습니다. 각 모델은 0.64, 0.85, 0.87, 0.88, 0.885의 정확도를 각각 기록하였으며, MobileNetV2 모델이 가장 좋은 성능을 보였습니다. 그 후, MobileNetV2를 기반으로 Fine-tuning을 수행하고, feature layer 내부에 추가적인 attention layer를 도입하여 정확도를 0.933으로 향상시켰습니다.

- **Performance Highlights**: MobileNet-Lung은 사전 훈련된 모델들보다 월등히 높은 0.933의 정확도를 기록하였습니다. 이 모델은 폐 질환 분류 작업에서 매우 우수한 성능을 보여, 초기에 질병을 진단하는 데 중요한 기여를 할 것으로 기대됩니다.



### Verification of Geometric Robustness of Neural Networks via Piecewise Linear Approximation and Lipschitz Optimisation (https://arxiv.org/abs/2408.13140)
- **What's New**: 이 논문에서는 이미지 입력에 대한 기하학적 변환(geometric transformations)을 확인하는 새로운 방법을 제안합니다. 특히, 회전(rotation), 크기 조정(scaling), 비틀림(shearing), 및 이동(translation)에 대한 신경망의 강인성을 검증할 수 있는 능력을 향상시킵니다.

- **Technical Details**: 제안된 방법은 샘플링(sampling) 및 선형 근사(linear approximations)를 결합하여 픽셀 값에 대한 확실한 조각선형 제약조건(piecewise linear constraints)을 계산합니다. 이전 방법들도 포함할 수 있는 조각선형 이완(piecewise linear relaxation) 방법을 제공하여, 기하학적 변환으로 생성되는 이미지 집합을 근사합니다.

- **Performance Highlights**: 실험 결과, 제안된 구현은 보다 다양한 검증 케이스를 해결하는 동시에 계산 효율성(computational efficiency)에서 향상된 성능을 보여줍니다.



### End-to-end Surface Optimization for Light Contro (https://arxiv.org/abs/2408.13117)
- **What's New**: 본 논문에서는 목표 조명 분포를 달성하기 위해 빛을 반사하거나 굴절하는 자유형(optical surface mesh) 표면을 설계하는 문제를 다룹니다. 새로운 differentiable rendering 모델을 활용하여 최적화 전략을 제안하며, 최종적으로는 물리적 결과와의 일관성을 보장하는 설계가 이루어집니다.

- **Technical Details**: 본 연구에서는 optical surface를 삼각형 메쉬로 표현하고, 목표 분포와의 차이를 직접적으로 고려하여 최적화를 진행합니다. 특히, 각 면의 법선을 활용한 빛의 반사/굴절을 고려하는 새로운 differentiable rendering 모델을 사용하여 물리적으로 정확한 빛 전달을 설명합니다. 또한, CNC 밀링과 연마를 용이하게 하기 위해 기하학적 제약조건을 도입합니다. 최적화 과정에서 local minima를 피하기 위해 face-based optimal transport 문제를 구성하여 메쉬 형태의 변화 효과를 극대화합니다.

- **Performance Highlights**: 제안된 방법은 다양한 목표 이미지에 대해 시뮬레이션 렌더링 및 물리적 프로토타입을 이용하여 효율성을 입증하였으며, 기존 방법 대비 특징적인 정확도 향상을 보여주었습니다. 이는 고정밀 광학 표면 설계에 효과적인 솔루션을 제공하여 컴퓨터 그래픽스를 넘어 다양한 응용 분야에서 유용할 것으로 기대됩니다.



### Dynamic Label Adversarial Training for Deep Learning Robustness Against Adversarial Attacks (https://arxiv.org/abs/2408.13102)
- **What's New**: 본 연구에서는 동적 레이블 적대적 훈련(dynamic label adversarial training, DYNAT) 알고리즘을 제안하고 있습니다. 이 알고리즘은 목표 모델이 가이드 모델의 결정을 통해 점진적으로 강건성을 향상시킬 수 있도록 돕습니다.

- **Technical Details**: DYNAT는 기존의 정적 레이블 대신 동적으로 생성된 레이블을 사용하며, 손실 함수로는 크로스 엔트로피(loss function)를 활용합니다. 이는 기존의 KL-divergence 방법보다 목표 모델의 분류 능리에 더 직접적으로 초점을 맞춰 성능을 개선합니다. 또한, DYNAT 알고리즘은 적대적 예제 생성(inner optimization)에서 예산을 고려하여 클린 정확도와 강건한 정확도 간의 균형을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: DYNAT 알고리즘은 CIFAR-10 및 CIFAR-100 데이터셋에서 기존 방어 방법들보다 우수한 성능을 보여주었으며, ResNet과 WideResNet 아키텍처를 사용한 실험에서도 경쟁력 있는 결과를 나타냈습니다.



### SIMPLE: Simultaneous Multi-Plane Self-Supervised Learning for Isotropic MRI Restoration from Anisotropic Data (https://arxiv.org/abs/2408.13065)
- **What's New**: 이번 연구에서는 SIMPLE이라는 방법을 제안하여, 비등방성(anisotropic) MRI 데이터를 사용하여 동시 다중 평면에서 자기 자신을 학습(self-supervised)함으로써 등방성(isotropic) MRI 이미지를 복원하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: SIMPLE 방법은 여러 평면에서 수집된 기존의 비등방성 클리닉 데이터를 활용하여, 시뮬레이션 다운샘플링 과정 없이 실질적인 등방성 MRI 이미지를 생성합니다. 이론적으로는 3D 데이터의 본질적인 특성을 고려하고, 리니어 보간(linear interpolation) 및 패치 추출(patch extraction) 등의 필수 전처리 단계를 포함합니다.

- **Performance Highlights**: 실험 결과, SIMPLE은 Kernel Inception Distance (KID) 지표를 통해 최신 기법들보다 뛰어난 성능을 보였으며, 방사선 전문의의 평가에서도 우수함을 입증했습니다. 생성된 등방성 볼륨은 더 정확한 용적 분석(volumetric analysis)과 3D 재구성을 가능하게 하여 임상 진단 능력의 획기적인 향상을 약속합니다.



### When Diffusion MRI Meets Diffusion Model: A Novel Deep Generative Model for Diffusion MRI Generation (https://arxiv.org/abs/2408.12897)
Comments:
          11 pages, 3 figures

- **What's New**: 본 연구는 Diffusion MRI (dMRI)를 생성하기 위한 새로운 생성적 접근 방식을 제안하며, 고품질 이미지를 생성하기 위해 깊은 확산 모델(deep diffusion models)을 활용합니다.

- **Technical Details**: 제안된 방법은 3D 및 4D 데이터의 회전 불변 구면 조화 함수(RISH) 특징을 생성하고, transfer learning 전략을 통해 7T 데이터 부족 문제를 해결하며, 해상도 차이를 제거하기 위한 슈퍼 해상도 모듈(super-resolution module)을 포함합니다.

- **Performance Highlights**: 3T에서 7T로의 dMRI 이미지 품질을 향상시키는 이미지 매핑 작업을 통해, 기존 최첨단 방법들 대비 뛰어난 성능을 입증하였습니다.



### Can AI Assistance Aid in the Grading of Handwritten Answer Sheets? (https://arxiv.org/abs/2408.12870)
- **What's New**: 최근 인공지능(AI)의 발전에 따라 AI를 활용한 수기 답안지 채점 지원 솔루션에 대한 관심이 증가하고 있습니다. 본 연구는 자동 텍스트 감지, 중요 키워드 강조 등을 포함하는 AI 보조 채점 파이프라인을 소개합니다.

- **Technical Details**: AI 보조 채점 파이프라인은 먼저 질문지를 PDF에서 자동으로 질문 영역을 감지한 후, SOTA(S state-of-the-art) 텍스트 감지 방법을 활용해 스캔한 답안지 상의 중요 키워드를 강조합니다. 이 시스템은 기존 e-learning 관리 플랫폼에 구현되었습니다.

- **Performance Highlights**: AI 지원을 통해 채점자는 평균적으로 단일 응답을 채점하는 데 31%, 단일 답안지를 채점하는 데 33% 적은 시간을 소요했습니다. 이는 4개 강좌에서 총 5개의 실제 시험을 분석한 결과입니다.



### Universal dimensions of visual representation (https://arxiv.org/abs/2408.12804)
- **What's New**: 이 논문에서는 다양한 구조의 심층 신경망들이 자연 이미지 표현에서 공유되는 보편적인 차원(latent dimensions)을 학습한다는 사실을 발견했습니다. 이러한 차원은 특정 네트워크의 특성과 독립적이며, 인공지능과 인간의 시각적 표현 사이의 유사성을 확인하는 데 기여합니다.

- **Technical Details**: 200,000개 이상의 자연 이미지 표현 차원을 분석하여, 각 차원의 보편성과 인간 뇌와의 유사성을 평가하였습니다. 이 연구는 Microsoft Common Objects in Context (COCO) 데이터베이스의 이미지와 fMRI를 통한 인간 뇌의 활동을 비교하여 진행되었습니다.

- **Performance Highlights**: 모든 네트워크 구조에서 공통으로 발견되는 보편적인 차원들은 10개 미만으로 축소 가능하며, 네트워크의 구조나 작업 목표와 관계없이 인간 뇌의 표현과 높은 유사성을 나타냅니다. 이로 인해 인공지능과 생물학적 시각의 유사성을 규명할 수 있는 가능성이 열렸습니다.



### Real-Time Posture Monitoring and Risk Assessment for Manual Lifting Tasks Using MediaPipe and LSTM (https://arxiv.org/abs/2408.12796)
Comments:
          Proceedings of the 1st International Workshop on Multimedia Computing for Health and Medicine at ACM MM'24

- **What's New**: 본 연구에서는 AI(인공지능)와 computer vision(컴퓨터 비전) 기술을 활용하여 수동 리프팅 작업을 위한 실시간 자세 모니터링 및 위험 평가 시스템을 개발하는 데 중점을 두었습니다. 기존의 자세 교정 방법들이 지연된 피드백과 개인화된 평가 부족으로 효과적이지 않은 문제를 해결하기 위해 제안된 솔루션은 AI 기반의 자세 감지, 세부 키포인트 분석, 위험 수준 결정, 사용자가 손쉽게 사용할 수 있는 웹 인터페이스를 통한 실시간 피드백을 통합합니다.

- **Technical Details**: 제안된 시스템은 MediaPipe와 LSTM(Long Short-Term Memory) 네트워크를 활용하여 리프팅 자세를 실시간으로 모니터링하고 관련 위험을 평가하며 즉각적인 교정 피드백을 제공합니다. 이 시스템은 정확하고 부정확한 동작을 기반으로 자세를 분류하는 세부 키포인트 분석을 수행합니다. 연구는 광범위한 데이터 수집, 모델 학습 및 반복 개발을 포함하며, 이를 통해 높은 정확도와 사용자 만족도를 보장합니다.

- **Performance Highlights**: 이 연구는 기존 방법론과 비교하여 실시간 피드백과 위험 평가에서 상당한 개선을 입증하여, 사용자에게 실질적이고 즉각적인 혜택을 제공하는 새로운 접근 방식을 제안합니다. 연구의 결과는 위험한 리프팅 자세 감지를 위한 LSTM 모델을 제안하고, 다양한 각도와 거리에서 촬영된 좋은 및 나쁜 리프팅 동작의 비디오로 구성된 데이터셋을 구현하였으며, 사용자 웹캠을 통해 WMSDs(Work-related Musculoskeletal Disorders)를 예방하기 위한 시스템의 활용 가능성을 보여줍니다.



### Data-Centric Approach to Constrained Machine Learning: A Case Study on Conway's Game of Lif (https://arxiv.org/abs/2408.12778)
- **What's New**: 이번 논문은 Conway의 생명 게임을 위한 데이터 중심(machine learning applications) 접근법을 제시하며, 최소한의 아키텍처 네트워크를 학습하여 주어진 단계 수에 대한 전환 규칙을 배우는 문제를 다룹니다. 이를 통해 제한된 학습 설정에서도 효율적인 데이터 설계의 중요성을 강조합니다.

- **Technical Details**: 이 연구에서는 이미지-투-이미지 전환(task of image-to-image translation) 문제로 생명 게임의 규칙을 학습하기 위해 CNN(Convolutional Neural Network)을 사용합니다. 제한된 아키텍처로 작업하여 훈련 데이터를 철저히 제어하는 방식으로 학습합니다. 훈련 데이터의 효율성을 기존 무작위 생성 보드 대신 정교하게 설계한 보드에서 비교합니다.

- **Performance Highlights**: 정교하게 설계된 단일 훈련 보드로 다단계 예측(multi-step prediction task)에서 수렴 속도와 정확도가 크게 향상되었음을 보여주며, 이는 데이터 중심(data-centric) 접근법이 제약된 머신러닝 응용 프로그램에서 유리하다는 것을 입증합니다.



### Semi-Supervised Variational Adversarial Active Learning via Learning to Rank and Agreement-Based Pseudo Labeling (https://arxiv.org/abs/2408.12774)
Comments:
          To be published in the 2024 International Conference on Pattern Recognition (ICPR)

- **What's New**: 이 논문에서는 기존의 variational adversarial active learning (VAAL) 방법의 한계를 극복하기 위해 새로운 접근 방법을 제안합니다. 특히, unlabeled 데이터 활용과 task 관련 정보 통합을 통해 학습 효율을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법에서는 pseudo-labeling 알고리즘을 개선하여 unlabeled 데이터의 모든 정보를 반영하고, ranking-based loss prediction 모듈을 개발하여 예측된 상대적인 순위 정보를 차별화 가능한 순위 손실로 변환합니다. 이 손실은 variational autoencoder의 latent space에 임베디드되어 adversarial 방식으로 훈련됩니다.

- **Performance Highlights**: 제안되는 SS-VAAL은 다양한 이미지 분류 및 segmentation 벤치마크 데이터셋에서 기존의 최첨단 기법들보다 우수한 성능을 보여줍니다.



### Hierarchical Attention and Parallel Filter Fusion Network for Multi-Source Data Classification (https://arxiv.org/abs/2408.12760)
Comments:
          Accepted by IEEE GRSL

- **What's New**: 저자는 HSI(히퍼스펙트럴 이미지)와 SAR(합성 개구 레이더) 데이터를 동시에 분류하는 새로운 계층적 주의 하이브리드 네트워크인 HAPNet을 제안하였습니다. 이 네트워크는 글로벌, 스펙트럴, 로컬 특성을 동시에 활용할 수 있는 기능 모델링을 통해 기존 방법보다 더 나은 분류 성능을 보여줍니다.

- **Technical Details**: HAPNet은 두 가지 주요 모듈인 Hierarchical Attention Module (HAM)과 Parallel Filter Fusion Module (PFFM)을 포함합니다. HAM은 HSI 특징 추출을 위한 모듈로, 다양한 범위의 정보를 동시에 캡처하며, PFFM은 HSI와 SAR 데이터 간의 교차 모달(feature interaction) 기능을 강화합니다. 이 모듈들은 주파수 영역(frequency domain)에서의 상호작용을 통해 특징을 융합합니다.

- **Performance Highlights**: HAPNet은 두 개의 다중 출처 원거리 탐지 데이터셋에서 각각 91.44%와 80.51%의 전반적인 정확도를 달성하였으며, 이는 최신 방법들보다 우수한 성능을 나타냅니다.



### Quantization-free Lossy Image Compression Using Integer Matrix Factorization (https://arxiv.org/abs/2408.12691)
Comments:
          19 pages, 6 figures, 1 table, 1 algorithm

- **What's New**: 본 연구에서는 양자화(quantization)가 필요 없는 새로운 손실 이미지 압축 방법을 소개합니다. 이는 정수 행렬 분해(Integer Matrix Factorization, IMF) 방법을 변형하여, 이미지 데이터를 정수 요소를 가진 두 개의 작은 행렬의 곱으로 표현하였습니다.

- **Technical Details**: IMF는 이미지 데이터를 저순위(low-rank) 표현으로 변환하며, 블록 좌표 하강법(block coordinate descent, BCD) 알고리즘을 통해 효율적인 반복(iterative) 계산을 제공합니다. 이전의 SVD와 달리, IMF는 양자화를 필요로 하지 않으므로 압축 성능을 개선할 수 있습니다.

- **Performance Highlights**: Kodak 및 CLIC 2024 데이터셋을 통한 실험 결과, IMF 압축 방법이 JPEG보다 0.25 bits per pixel (bpp) 이하의 낮은 비트 전송률에서 consistently outperforming 하며, 더 높은 비트 전송률에서도 비교 가능한 성능을 보여주었습니다. 특히, 비트 전송률이 0.25 bpp 이하일 때, JPEG 대비 Top-1 정확도가 5 퍼센트 포인트 이상 향상되었습니다.



### MultiMed: Massively Multimodal and Multitask Medical Understanding (https://arxiv.org/abs/2408.12682)
- **What's New**: 새로운 벤치마크인 MultiMed는 256만 개의 샘플을 기반으로 하여 다양한 의료 모달리티와 과제를 아우르는 대규모 학습을 평가하고 지원하는 데 중점을 둡니다. 각 모달리티는 의료 보고서, 병리학, 유전체학, 단백질 데이터 등 포함됩니다.

- **Technical Details**: MultiMed는 10가지 의료 모달리티(예: 의료 보고서, 병리학, 유전체, 단백질 데이터)를 포함하며, 11가지 도전적인 과제로 구성되어 있습니다. 이 과제는 질병 예후, 단백질 구조 예측, 의료 질문 응답 등을 포함합니다. MultiMed는 여러 관련 모달리티 및 과제에서의 대규모 학습을 지원합니다.

- **Performance Highlights**: MultiMed를 통해 수행된 실험은 단일 모달리티, 다중 모달리티, 다중 과제 모델들의 성능을 벤치마킹하며, 관련 모달리티 및 과제를 아우르는 대규모 의료 모델 교육의 장점을 강조합니다. 이는 진단 및 예후 기능 향상에 기여할 수 있습니다.



### One-shot Video Imitation via Parameterized Symbolic Abstraction Graphs (https://arxiv.org/abs/2408.12674)
Comments:
          Robot Learning, Computer Vision, Learning from Videos

- **What's New**: 이번 연구에서는 Parameterized Symbolic Abstraction Graphs (PSAG)을 사용하여 비디오 시연을 해석하는 새로운 방법을 제안하였습니다. 이 방법은 비디오에서 객체와 관계를 추출하는 것 뿐만 아니라, 비가시적인 물리적 속성(예: 힘)을 추정하는 데에 있어서도 효과적입니다.

- **Technical Details**: PSAG는 노드가 객체를 표현하고 엣지가 객체 간의 관계를 나타내는 구조입니다. 이를 통해 각 노드는 객체의 여섯 가지 자유도(6DOF)를 포함하며, 엣지는 접촉 상태와 작용하는 힘을 파라미터화합니다. 연구진은 이 구조를 통해 시뮬레이션을 학습하며, 이러한 비가시적 속성을 실제 로봇 연구에 적용하였습니다.

- **Performance Highlights**: 다양한 작업(아보카도 자르기, 채소 자르기, 액체 붓기, 반죽 굴리기, 피자 자르기)에서 성공적인 일반화를 보여주었으며, 이러한 작업들은 학습 환경과는 다른 물리적 속성을 요구하는 실험 환경에서 수행되었습니다.



### Research on Improved U-net Based Remote Sensing Image Segmentation Algorithm (https://arxiv.org/abs/2408.12672)
- **What's New**: 본 연구에서는 U-Net 네트워크에 SimAM과 CBAM 주의(attention) 메커니즘을 도입하여 원거리 탐지 이미지 분할(Remote Sensing Image Segmentation) 분야에서 성능을 획기적으로 향상시켰습니다.

- **Technical Details**: U-Net 네트워크에 SimAM과 CBAM 모듈을 개별적으로 추가한 결과, MIoU(Mean Intersection over Union)가 각각 17.41% 및 12.23% 개선되었으며, Mpa(Mean Pixel Accuracy)와 Accuracy도 크게 향상되었습니다. 두 모듈을 융합(fusion)했을 때 MIoU는 19.11% 상승하고, Mpa와 Accuracy는 각각 16.38% 및 14.8% 개선되었습니다.

- **Performance Highlights**: 전반적으로, 새로운 방법론은 뛰어난 분할 정확도(segmentation accuracy)와 뛰어난 시각적 효과(visual effect)를 보여주며, 강한 일반화 능력(generalization ability)과 강인성을(Robustness) 갖추고 있습니다. 이는 원거리 탐지 이미지 분할 기술에 대한 새로운 경로를 제시하며, 알고리즘 선택 및 개선에 중요한 참고 가치를 제공합니다.



### Joint Image De-noising and Enhancement for Satellite-Based SAR (https://arxiv.org/abs/2408.12671)
- **What's New**: 본 논문에서는 Synthetic Aperture Radar (SAR) 데이터를 처리하는 과정에서 나타나는 곱셈 잡음과 낮은 대비 문제를 동시에 해결하기 위한 새로운 알고리즘을 제안합니다. 이 방법은 Contrast Limited Adaptive Histogram Equalization (CLAHE) 기법을 활용하여 이미지의 품질을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안된 알고리즘은 Median Filter와 대비 향상 과정을 통합하여 성능을 개선하고 계산 복잡도를 줄이는 것을 목표로 합니다. 이를 통해 SAR 이미지에서 speckle noise를 줄이고 대비를 높이는 효과를 동시에 누릴 수 있습니다. 실험적 결과는 유럽우주국의 ERS-2 위성에서 수집된 데이터를 바탕으로 검증되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 알고리즘이 기존의 방법보다 더 빠르고 효과적으로 SAR 이미지의 품질을 개선하는 것으로 나타났습니다. 특히, 대비가 낮은 이미지에서도 정보 추출의 가능성을 높였습니다.



### Identifying Locally Turbulent Vortices within Instabilities (https://arxiv.org/abs/2408.12662)
Comments:
          IEEE LDAV 2024 poster

- **What's New**: 이번 연구에서는 난류의 불안정성 내에서 지역적으로 난류 소용돌이를 자동으로 감지하는 접근 방식을 제시합니다. Topological Data Analysis (TDA) 기법을 활용하여 난류의 기하학적인 구조를 추출하고, 난류 소용돌이의 행동과 이상적인 난류 소용돌이 간의 상관관계를 추정하기 위한 새로운 지표를 도입합니다.

- **Technical Details**: 이 연구는 두 차원 비점성 유체 흐름에 대한 비압축성 유체의 비정상적인 오일러 방정식에 기반한 CFD 시뮬레이션 코드를 사용합니다. Enstrophy의 간소화 및 Morse-Smale 복합체를 활용하여 소용돌이를 추출합니다. 또한, 난류 특성을 분석하기 위해 Fast Fourier Transform을 통한 운동 에너지 스펙트럼을 계산합니다.

- **Performance Highlights**: 최초 실험 결과는 CFD 전문가에 의해 식별된 난류 소용돌이가 E(k) ∼k^{-5/3}의 속성을 만족함을 보여주었으며, 이는 새로운 지표의 효과성을 입증합니다. 향후 연구에서는 감독 분류 방법을 활용하여 난류 소용돌이를 체계적으로 탐지할 계획입니다.



### Pediatric TSC-related eplipsy classification from multi-contrast images using quantum neural network (https://arxiv.org/abs/2408.12615)
Comments:
          5 pages,4 figures,2 tables,presented at ISBI 2024

- **What's New**: 이번 연구에서는 소아 신경학적 질환인 튜버러스 스클레로시스 컴플렉스(TSC)를 다루기 위해 새로운 딥러닝 모델 QResNet을 도입했습니다. 이 모델은 기존의 CNN(Convolutional Neural Networks)과 양자 신경망(Quantum Neural Networks)을 통합하여 TSC MRI 이미지 분류를 수행하는 데 중점을 두었습니다.

- **Technical Details**: QResNet은 2층의 양자층(Quantum Layer, QL)으로 구성되어 있습니다. ZZFeatureMap과 Ansatz 층이 포함되며, 클래식 데이터 처리를 위한 양자 기반 구조입니다. 이러한 양자 특성 맵을 활용하여 기존 데이터의 복잡한 변환을 수행하는 것이 특징입니다.

- **Performance Highlights**: QResNet은 TSC MRI 이미지 분류에서 기존의 3D-ResNet 모델과 비교하여 높은 정확도와 AUC(Area Under the Curve) 지표를 기록했습니다. 이 결과는 양자 컴퓨팅이 의료 이미징 및 진단 분야에 혁신을 가져올 수 있음을 보여줍니다.



### Convolutional Neural Networks for Predictive Modeling of Lung Diseas (https://arxiv.org/abs/2408.12605)
Comments:
          7 pages

- **What's New**: 이번 논문에서는 폐 이미징을 위한 질병 예측을 위해 HRNet와 void-convolution 기법을 결합한 혁신적인 모델 Pro-HRnet-CNN을 제안합니다.

- **Technical Details**: Pro-HRnet-CNN 모델은 권위 있는 LIDC-IDRI 데이터셋을 활용하여 실험 비교가 이루어졌습니다. 전통적인 ResNet-50과 비교했을 때, Pro-HRnet-CNN은 작은 크기의 결절(nodule) 특성 추출(feature extraction) 및 인식(recognition)에서 더 나은 성능을 보였습니다.

- **Performance Highlights**: 특히 작은 목표(target) 탐지에서 모델은 정확도 향상에서 뛰어난 성능을 보여주어 폐 질환의 조기 발견 및 예측(prognostication)을 위한 혁신적인 길을 열었습니다.



New uploads on arXiv(cs.AI)

### How Diffusion Models Learn to Factorize and Compos (https://arxiv.org/abs/2408.13256)
Comments:
          11 pages, 6 figures, plus appendix, some content overlap with arXiv:2402.03305

- **What's New**: 본 논문은 Diffusion 모델들이 데이터의 구성 요소를 조합하고 일반화하는 능력을 가지고 있음을 보여주지만, 이러한 구성 가능성의 내부 메커니즘은 여전히 불분명하다는 점을 강조합니다. 모델이 의미 있는 표현(meaningful representations)을 어떻게 학습하는지를 심층적으로 분석하기 위해 2D Gaussian 데이터셋에서 수행된 실험을 통해 결과를 공유합니다.

- **Technical Details**: Conditional Denoising Diffusion Probabilistic Models (DDPMs)의 통제된 실험을 수행하여 모델의 표현력이 데이터의 변화 기저(underlying features)에 대한 요인화된(factorized) 표현을 학습하는 방식을 연구했습니다. 실험에서는 다양한 파라미터화(parameterization) 기술을 사용하여 저차원 데이터셋에서 모델의 성능을 평가하고, 모델이 조합할 수 있는 특정 구조의 데이터를 생성하는 능력을 분석했습니다.

- **Performance Highlights**: 모델은 완전 연속적인 다중체계(manifold) 표현을 학습하는 데는 한계가 있었지만, 여러 조합의 예제를 통해 고유한 구성 요소를 잘 조합할 수 있고, 일반화 능력이 상당히 향상된 결과를 보였습니다. 이는 학습에 필요한 구성 예제가 적다는 것을 시사하며, Diffusion 모델의 교육 방법에 대한 새로운 접근 방식을 제안합니다.



### Enhancing Few-Shot Transfer Learning with Optimized Multi-Task Prompt Tuning through Modular Prompt Composition (https://arxiv.org/abs/2408.13227)
- **What's New**: 이 논문에서는 여러 과제 간의 지식 전이를 촉진하여 성능을 향상시키기 위해 다중 작업 프롬프트 튜닝(multi-task prompt tuning) 접근법을 제안합니다. 특히 ComPT라는 프레임워크를 도입하여 공유 프롬프트와 과제별 프롬프트를 결합하여 각 과제에 대한 목표 프롬프트를 생성합니다.

- **Technical Details**: 제안된 ComPT 접근 방식에서는 N개의 과제를 다루며, M개의 소스 프롬프트를 사용하여 각 개별 목표 과제에 대한 타겟 프롬프트를 생성합니다. 소스 프롬프트는 공유되고, 이를 기반으로 각 과제에 전용 프롬프트(프라이빗 프롬프트)와 결합하여 최종 타겟 프롬프트를 형성합니다. 이 과정에서는 학습 가능한 주의(attention) 모듈을 통해 소스 프롬프트를 활성화하고 결합합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 프롬프트 튜닝 방식과 비교해 정확성과 견고성을 개선했으며, 특히 few-shot 설정에서 다른 방법들보다 우수한 성과를 냈습니다. GLUE 벤치마크를 포함하여 다양한 작업에서 탁월한 성능을 입증했습니다.



### Temporal Fairness in Decision Making Problems (https://arxiv.org/abs/2408.13208)
Comments:
          Paper accepted at ECAI 2024. This is an extended version that includes Supplementary Material

- **What's New**: 이번 연구에서는 의사 결정 문제에서 공정성(fairness)의 새로운 해석을 제안합니다. 기존의 공정성 공식에서 발전하여, 과거 결정의 역사(history)를 고려한 시간적 관점(temporal perspective)에서 공정성을 평가하는 방법에 중점을 두었습니다.

- **Technical Details**: 시간적 공정성(temporal fairness)이라는 개념을 도입한 후, 최적화 문제(optimization problems)로 공식화된 의사 결정 문제에 시간적 공정성을 포함하는 세 가지 접근법을 제안합니다. 각 접근 방식은 시간적 요소를 고려하여 공정성을 평가합니다.

- **Performance Highlights**: 네 가지 다양한 영역에서 제안한 방법의 질적 평가(qualitative evaluation)를 수행하고, 시간적 요소를 고려하지 않은 기준 접근법(baseline approach)과 해결책을 비교하였습니다.



### DOMAINEVAL: An Auto-Constructed Benchmark for Multi-Domain Code Generation (https://arxiv.org/abs/2408.13204)
- **What's New**: 이 연구는 기존의 코드 벤치마크가 일반적인 코딩 작업에 중점을 두고 있는 반면, 도메인 특정 코딩 작업을 평가하기 위한 다중 도메인 코드 벤치마크인 DOMAINEVAL을 제안합니다. 이를 통해 LLM의 다양한 도메인에서의 코딩 능력을 포괄적으로 평가할 수 있게 되었습니다.

- **Technical Details**: DOMAINEVAL은 총 2454개의 코드 과제와 5892개의 테스트 케이스로 구성되어 있으며, 컴퓨테이션(computation), 네트워크(network), 기본 동작(basic operation), 시스템(system), 시각화(visualization), 암호학(cryptography)의 여섯 가지 도메인을 포함합니다. 연구에서는 10개 이상의 LLM을 평가하고, 이를 위한 완전 자동화된 테스트 지향(construction pipeline) 구조를 제공합니다.

- **Performance Highlights**: 평균적으로 LLM들은 컴퓨테이션 작업에서는 82.44%의 Pass@1 성능을 기록했으나, 암호학과 시스템 도메인에서는 각각 33.08%와 37.50%로 낮은 성능을 보였습니다. 특히 Llama-2-13b-chat 모델은 다른 LLM들과 비교할 때 성능 변동성이 가장 큰 것으로 나타났습니다.



### Say No to Freeloader: Protecting Intellectual Property of Your Deep Mod (https://arxiv.org/abs/2408.13161)
- **What's New**: 이 논문에서는 Compact Un-transferable Pyramid Isolation Domain (CUPI-Domain)라는 새로운 접근법을 소개하여, 허가된 도메인에서 비허가된 도메인으로의 불법 전이를 차단하는 방법을 제시합니다. 이를 통해 모델 지적재산권(IP) 보호 관련 문제를 해결하고자 합니다.

- **Technical Details**: CUPI-Domain은 허가된 도메인의 독특한 스타일 특성 강조를 통해 비허가된 도메인에서의 유사한 사적 스타일 특성 인식을 방해합니다. CUPI-Domain 생성기와 외부 Domain-Information Memory Banks (DIMB) 설계를 통해 안정적인 도메인 클래스 및 스타일 특성을 저장하고 업데이트하여 안전한 모델 배포를 보장합니다.

- **Performance Highlights**: CUPI-Domain은 여러 공개 데이터셋에 대한 실험 결과를 통해 비허가된 도메인에서의 인식 능력을 효과적으로 감소시켰습니다. 제안된 손실 함수들은 스타일 및 판별 기능 간의 차이를 향상시키는데 크게 기여했습니다.



### CRUXEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution (https://arxiv.org/abs/2408.13001)
Comments:
          13pages

- **What's New**: 본 논문에서는 CRUXEVAL-X라는 다국어 코드 추론 벤치마크를 제안합니다. 이 벤치마크는 19개의 프로그래밍 언어를 지원하며, 각 언어에 대해 최소 600개의 주제를 포함하고 있습니다. 또한 총 19K개의 내용 일치 테스트를 제공합니다.

- **Technical Details**: CRUXEVAL-X의 구축 파이프라인은 전적으로 자동화되며, 테스트 피드백을 기반으로 반복적으로 코드 생성을 수행하고 수정합니다. 여러 프로그래밍 언어 간에 장벽을 넘기 위해 다양한 전이 규칙을 만든 점이 특징입니다. 자동화된 코드 번역 파이프라인이 적용되어 있습니다.

- **Performance Highlights**: 24개의 대표적인 LLM을 평가한 결과, 언어 쌍 간의 상관관계가 확인되었습니다. 예를 들어, TypeScript와 JavaScript는 강한 긍정적 상관관계를 보였고, Python에서만 훈련된 모델조차도 다른 언어에서 최대 34.4%의 Pass@1을 달성할 수 있었으며, 이는 LLM의 언어 간 일반화 능력을 나타냅니다.



### Enhancing Knowledge Tracing with Concept Map and Response Disentanglemen (https://arxiv.org/abs/2408.12996)
Comments:
          Accepted to Knowledge-Based Systems Journal

- **What's New**: 이 논문에서는 Knowledge Tracing (KT) 모델의 한계를 극복하기 위해 Concept map-driven Response disentanglement (CRKT) 방법을 제안합니다. 기존 KT 모델은 정답과 오답의 이분법적인 반응만을 중시했으나, CRKT는 학생들이 선택한 답변의 차이점을 활용하여 더 깊이 있는 학습 상태를 파악합니다.

- **Technical Details**: CRKT 모델은 정답과 오답의 구별을 뛰어넘어, 미선택 답변(unselected responses)의 정보도 활용하여 학생들의 지식 상태를 추적합니다. 또한, 개념 간 관계를 명시적으로 담은 개념 맵(concept map)을 통해 학생의 지식 간의 상호 관계를 효율적으로 캡처합니다.

- **Performance Highlights**: CRKT는 여러 데이터셋에 대한 실험을 통해 예측 정확성과 해석 가능성에서 최첨단 모델을 초월하는 성능을 달성했습니다. 이를 통해 각 학생의 학습 경험을 개선하고, 맞춤형 피드백을 제공할 수 있는 가능성을 나타냅니다.



### QD-VMR: Query Debiasing with Contextual Understanding Enhancement for Video Moment Retrieva (https://arxiv.org/abs/2408.12981)
Comments:
          9 pages, 4 figures, 4 tables

- **What's New**: 새로운 QD-VMR 모델을 통해 영상의 특정 순간을 올바르게 검색할 수 있는 정확도를 개선했습니다. 이 모델은 쿼리의 모호한 의미 이해 문제를 해결하기 위한 최초의 쿼리 디바이싱(query debiasing) 모델로, 문맥적 이해(Contextual Understanding)가 강화되었습니다.

- **Technical Details**: QD-VMR는 Global Partial Aligner 모듈을 사용하여 비디오 클립과 쿼리 기능의 정렬을 수행하고, 쿼리 디바이싱 모듈(Query Debiasing Module)로 편향된 쿼리 기능을 획득합니다. 또한 Visual Enhancement 모듈을 통해 쿼리와 관련 없는 비디오 기능을 정제합니다. 마지막으로 DETR 구조를 이용해 타겟 비디오 순간을 예측합니다.

- **Performance Highlights**: QD-VMR은 QVHighlights, Charades-STA 및 TACoS의 세 가지 벤치마크 데이터 세트에서 SOTA(State of the Art) 성능을 달성했습니다. 모델의 주요 모듈은 기존 VMR 접근 방식으로 전이 가능하며, 새로운 파라미터는 1M을 초과하지 않아 모든 평가 지표에서 знач상적인 개선을 보여주었습니다.



### iSee: Advancing Multi-Shot Explainable AI Using Case-based Recommendations (https://arxiv.org/abs/2408.12941)
Comments:
          Accepted to appear at the ECAI-PAIS 2024 main conference proceedings

- **What's New**: 이번 연구에서는 Explainable AI (XAI) 시스템의 다중 사용자 요구를 충족시키기 위해 'multi-shot' 접근 방식을 제안합니다. 개인화된 'explanation experience'를 통해 AI 의사결정 과정을 개선하는 전략을 소개하는 플랫폼인 iSee를 개발했습니다.

- **Technical Details**: iSee 플랫폼은 Case-based Reasoning (CBR) 방식을 활용하여 성공적인 설명 적응 경험을 캡처하고, 다양한 수준의 AI 및 XAI 전문성을 가진 설계 사용자와 협력하여 최적의 설명 전략을 성과적으로 설계하고 수정할 수 있도록 지원합니다. iSee는 사용자의 요구를 수집하고 과거 경험을 기반으로 가장 적합한 설명 전략을 검색하는 도구를 제공합니다.

- **Performance Highlights**: iSee 플랫폼의 평가 결과, 다양한 응용 분야에서 효과적으로 일반화되고 XAI 모범 사례 채택을 촉진할 수 있는 잠재력이 있음을 확인했습니다. 이 플랫폼은 사용자 피드백을 바탕으로 한 설명 전략 최적화를 통해, 사용자 만족도를 높이고 협업을 통한 지속 개선이 가능함을 보였습니다.



### Trustworthy, Responsible, and Safe AI: A Comprehensive Architectural Framework for AI Safety with Challenges and Mitigations (https://arxiv.org/abs/2408.12935)
- **What's New**: AI Safety는 AI 시스템의 안전한 채택 및 배포를 위한 필수적이고 새로운 연구 분야로 부상하고 있습니다. 본 논문에서는 Trustworthy AI, Responsible AI, Safe AI의 세 가지 관점에서 AI Safety를 정의하고 새로운 구조적 프레임워크를 제안합니다.

- **Technical Details**: AI Safety의 세 가지 기둥인 Trustworthy AI는 기술적 측면을, Responsible AI는 윤리적 및 조직적 측면을, Safe AI는 생태계 전반의 위험을 다룹니다. 우리는 현재의 연구와 발전을 종합적으로 리뷰하고 주요 도전과제 및 완화 전략을 제안합니다.

- **Performance Highlights**: AI Safety는 AI 모델과 시스템의 안전성을 제공하여 사회, 커뮤니티, 국가 수준의 안전을 증진시키는 것을 목표로 합니다. 이는 디지털 전환에 대한 사람들의 신뢰를 강화하는 데 기여할 것입니다.



### Abductive and Contrastive Explanations for Scoring Rules in Voting (https://arxiv.org/abs/2408.12927)
Comments:
          10 pages, 2 figures Extended version of a paper in proceedings of ECAI 2024

- **What's New**: 이번 논문에서는 투표 규칙을 분류기로 보고, 유권자들의 선호도 프로필에서 현재의 승자인 후보를 도출하거나 다른 후보가 낙선된 이유를 설명하는 최소한의 하위 집합을 식별하기 위해 형식적 설명 기법을 적용합니다. 특히, abductive 및 contrastive 설명을 통해 투표 규칙의 결과를 해석할 수 있는 새로운 방식을 제시합니다.

- **Technical Details**: 우리는 scoring rules로 알려진 특정 투표 규칙 클래스에 초점을 맞추며, 이는 plurality rule, k-approval rules 및 Borda rule을 포함합니다. 유권자의 선호도를 n×m 비순차 행렬로 표현하며, 이를 통해 필요 승자(necessary winner) 및 가능한 승자(possible winner) 개념과 연관된 형식적 설명을 정의하고 분석합니다. 알고리즘을 설계하여 Borda rule에 대한 abductive 설명의 최소 크기 하한을 증명합니다.

- **Performance Highlights**: 컴퓨터 시뮬레이션을 통해 최소 설명 크기와 승자의 승리 마진 간의 상관관계를 조사했으며, 최소 설명 크기가 작을수록 승자의 승리 마진이 크다는 흥미로운 음의 상관관계를 발견했습니다. 이러한 결과는 선호 프로필의 본질적 합의(intrinsic agreement)와 관련하여 해석할 수 있습니다.



### What Do You Want? User-centric Prompt Generation for Text-to-image Synthesis via Multi-turn Guidanc (https://arxiv.org/abs/2408.12910)
- **What's New**: DialPrompt는 텍스트-이미지 합성(TIS) 모델을 위한 사용자 중심의 다중 턴 대화형 프롬프트 생성 모델입니다. 이를 통해 사용자가 프롬프트 최적화 차원에 대한 선호를 입력하여 최종 프롬프트를 생성할 수 있습니다.

- **Technical Details**: DialPrompt는 고급 사용자로부터 15개 필수 차원을 추출하여 다중 턴 데이터를 구축했습니다. 각 대화 라운드에서 사용자의 선호를 쿼리하여, 특정 문구와 이미지 속성 간의 연관성을 이해할 수 있게 하여 해석 가능성을 향상시킵니다.

- **Performance Highlights**: DialPrompt는 synthesize된 이미지의 질에서 기존 프롬프트 공학 접근법에 비해 5.7% 우수성을 보였으며, 사용자 평가에서 사용자 중심성 점수에서 46.5% 더 높은 결과를 기록하고, 19명의 리뷰어에게 7.9/10의 평점을 받았습니다.



### IAA: Inner-Adaptor Architecture Empowers Frozen Large Language Model with Multimodal Capabilities (https://arxiv.org/abs/2408.12902)
- **What's New**: 다중모달 대형 언어 모델(MLLMs) 분야에서, 본 연구는 기존의 freeze 모델을 기반으로 새로운 Inner-Adaptor Architecture(IAA)를 도입하여 언어 모델의 성능 저하 없이 다중 모달 능력을 향상시키는 접근 방식을 제안합니다.

- **Technical Details**: 우리의 Inner-Adaptor Architecture는 LLM의 다양한 깊이에 다중 모달 어댑터를 포함하여, 고정된 언어 모델이 시각적 특성과 직접 상호작용하도록 합니다. 이 방법은 대규모 정렬 데이터 없이도 소규모 데이터셋에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 우리의 다중모달 접근 방식은 MME, MMBench, MMMU, RefCOCO와 같은 다양한 비전-언어 벤치마크에서 이전의 최신 기술을 초월하며, NLP 과제에서도 성능 저하 없이 우수한 결과를 보여줍니다.



### Multiple Areal Feature Aware Transportation Demand Prediction (https://arxiv.org/abs/2408.12890)
- **What's New**: 이 논문에서는 여러 지역 특성을 통합하여 스페이셜-템포럴(spatio-temporal) 수요 예측을 개선하는 새로운 모델인 ST-MFGCRN(Spatio-Temporal Multi-Feature-aware Graph Convolutional Recurrent Network)을 제안합니다. 이를 통해 지역 간의 유사성을 보다 정확하게 반영할 수 있습니다.

- **Technical Details**: ST-MFGCRN 모델은 여러 지역 데이터를 통합하고 이들 간의 스페이셜-템포럴 상관관계를 학습하기 위한 그래프 신경망과 순환 신경망 기술을 활용합니다. 특히, 센티넬 주의(attention) 메커니즘을 사용해 유용하지 않은 특성에 대해 부분적인 주의를 기울일 수 있도록 합니다. 데이터 입력은 클로즈니스(closeness), 기간(period), 트렌드(trend) 세 가지 시간 단위로 나뉘어 처리됩니다.

- **Performance Highlights**: 모델은 BusDJ 및 TaxiBJ의 두 개의 실세계 데이터셋에서 성능을 평가하였고, 기존의 최첨단 모델에 비해 최대 7% 및 8%의 성능 향상을 보였습니다. ST-MFGCRN은 특히 혼잡한 지역에서 8-12% 개선된 성능을 나타냈으며, 모델의 과적합을 방지하는 데에도 성공했습니다.



### Spatio-Temporal Road Traffic Prediction using Real-time Regional Knowledg (https://arxiv.org/abs/2408.12882)
- **What's New**: 이번 연구는 실시간 지역 인구 데이터를 도로 수준의 교통 예측에 통합한 첫 번째 연구로, 각 모달리티 훈련을 진행합니다.

- **Technical Details**: 이 연구에서는 POIs(관심 장소), 위성 이미지, 실시간 LTE 접근 추적을 활용하여 지역 수준의 지식을 동적 컨볼루션(dynamic convolution)과 시간 주의(attention) 메커니즘을 통해 임베딩(yield embedding)합니다. 이후 이 지식을 도로 수준의 주의 기반 예측 모델로 통합합니다.

- **Performance Highlights**: 실제 도로 교통 예측 실험 결과, 본 모델은 기존의 기준선 모델들을 초월하는 성능을 보였습니다.



### Has Multimodal Learning Delivered Universal Intelligence in Healthcare? A Comprehensive Survey (https://arxiv.org/abs/2408.12880)
Comments:
          21 pages, 6 figures

- **What's New**: 이번 논문은 다중모달 학습(multimodal learning) 기술이 의료 분야에서 보편적인 지능(universal intelligence)을 달성했는지를 종합적으로 분석합니다. 특히 데이터셋, 작업 지향 방법(task-oriented methods), 보편적 기초 모델(universal foundation models)의 세 가지 관점에서 현재의 연구 진행 상황을 조사합니다.

- **Technical Details**: 의료 다중모달 학습에 대한 포괄적인 조사가 있으며, 다중 모달 AI 모델은 다양한 데이터를 통합하여 복합적인 의료 현상을 이해하는 데 기여합니다. 이 논문은 다중모달 기술이 현재 의료 분야에서 경과한 발전 및 도전 과제를 다루고 있으며, 향후 연구 방향으로는 데이터와 기술, 성능 및 윤리에 관한 다섯 가지 문제를 탐구합니다.

- **Performance Highlights**: 현재 기술은 보편적인 의료 AI를 달성하지 못했으나, 앞으로 다중모달 학습의 발전과 협업을 통해 이러한 목표에 도달할 수 있는 여러 연구 방향을 제시합니다. 연구진들은 의료 영상 융합, 보고서 생성(report generation), 시각 질문 응답(visual question answering) 등 다양한 작업에서 주목할 만한 발전을 보였습니다.



### DeepDelveAI: Identifying AI Related Documents in Large Scale Literature Data (https://arxiv.org/abs/2408.12871)
Comments:
          28 pages and 10 figures

- **What's New**: 이 논문에서는 DeepDelveAI라는 포괄적인 데이터셋이 소개됩니다. 이 데이터셋은 대규모 학술 문헌 데이터베이스에서 AI 관련 연구 논문을 식별하기 위해 특별히 큐레이션되었습니다.

- **Technical Details**: DeepDelveAI 데이터셋은 이진 분류 작업을 위해 훈련된 고급 Long Short-Term Memory (LSTM) 모델을 사용하여 생성되었습니다. 이 모델은 AI 관련 논문과 비 AI 관련 논문을 구별하는 데 초점을 맞추었으며, 광범위한 데이터셋에서 훈련 및 검증이 이루어졌습니다.

- **Performance Highlights**: 모델은 높은 정확도, 정밀도, 재현율 및 F1-score를 달성했습니다. DeepDelveAI 데이터셋은 1956년 다트머스 회의 이후 2024년까지 발표된 940만 개 이상의 AI 관련 논문으로 구성되어 있어 다양한 분야에서 AI 연구의 추세, 주제 발전 및 진화를 분석하는 데 중요한 자원이 됩니다.



### Can AI Assistance Aid in the Grading of Handwritten Answer Sheets? (https://arxiv.org/abs/2408.12870)
- **What's New**: 최근 인공지능(AI)의 발전에 따라 AI를 활용한 수기 답안지 채점 지원 솔루션에 대한 관심이 증가하고 있습니다. 본 연구는 자동 텍스트 감지, 중요 키워드 강조 등을 포함하는 AI 보조 채점 파이프라인을 소개합니다.

- **Technical Details**: AI 보조 채점 파이프라인은 먼저 질문지를 PDF에서 자동으로 질문 영역을 감지한 후, SOTA(S state-of-the-art) 텍스트 감지 방법을 활용해 스캔한 답안지 상의 중요 키워드를 강조합니다. 이 시스템은 기존 e-learning 관리 플랫폼에 구현되었습니다.

- **Performance Highlights**: AI 지원을 통해 채점자는 평균적으로 단일 응답을 채점하는 데 31%, 단일 답안지를 채점하는 데 33% 적은 시간을 소요했습니다. 이는 4개 강좌에서 총 5개의 실제 시험을 분석한 결과입니다.



### Exploring Machine Learning Models for Lung Cancer Level Classification: A comparative ML Approach (https://arxiv.org/abs/2408.12838)
- **What's New**: 이번 논문은 폐암 단계 분류를 위한 머신 러닝(ML) 모델을 탐구하여 진단 정확도 및 예후 향상을 목표로 하였습니다.

- **Technical Details**: 이 연구에서는 파라미터 튜닝(parameter tuning)과 철저한 평가를 통해 다양한 ML 알고리즘을 평가하였습니다. 최소 자식 가중치(minimum child weight)와 학습률 모니터링(learning rate monitoring)과 같은 기법을 사용하여 오버피팅(overfitting)을 줄이고 성능을 최적화하였습니다.

- **Performance Highlights**: Deep Neural Network (DNN) 모델의 강력한 성능이 모든 단계에서 두드러졌고, 투표(voting) 및 배깅(bagging)과 같은 앙상블 방법들 또한 예측 정확도 및 강건성을 향상시키는 데 가능성을 보였습니다. 그러나 시그모이드(Sigmoid) 커널을 가진 Support Vector Machine (SVM) 모델은 도전 과제를 안고 있음을 알리고 있습니다.



### DutyTTE: Deciphering Uncertainty in Origin-Destination Travel Time Estimation (https://arxiv.org/abs/2408.12809)
Comments:
          7 pages

- **What's New**: 이 논문은 여행 시간 추정(TTE)에서의 불확실성 정량화를 해결하기 위한 'DutyTTE' 방법을 제안합니다. 이는 경로 예측의 정확성을 높이고 결국 여행 시간의 신뢰 구간을 제공하는 데 중점을 둡니다.

- **Technical Details**: DutyTTE는 깊은 강화 학습(Deep Reinforcement Learning) 방법을 활용하여 예측된 경로와 실제 경로 간의 정렬을 최적화하고, 혼합 전문가(Mixture of Experts) 기반의 불확실성 정량화 메커니즘을 도입하여 각 도로 구간의 여행 시간 불확실성을 모델링합니다. 또한 Hoeffding의 상위 신뢰 구간을 통해 추정된 신뢰 구간을 보정하여 통계적 보장을 제공합니다.

- **Performance Highlights**: 두 가지 실제 데이터 세트에 대한 광범위한 실험을 통해, 제안된 방법(DutyTTE)이 기존의 다른 방법들보다 우수성을 입증하였습니다.



### A Safe Self-evolution Algorithm for Autonomous Driving Based on Data-Driven Risk Quantification Mod (https://arxiv.org/abs/2408.12805)
- **What's New**: 본 논문에서는 데이터 기반 위험 정량화 모델을 기반으로 한 자율 주행을 위한 안전 자가 진화 알고리즘을 제안합니다. 이 알고리즘은 주변 환경의 안전 상황 추정 능력을 데이터 주도 접근 방식으로 구현하여 동적 교통 시나리오에서의 안전성과 성능 간의 균형 문제를 해결합니다.

- **Technical Details**: 제안된 알고리즘은 Transformer 아키텍처를 기반으로 하여 위험 정량화 모델을 구현하며, 인간의 위험 인지를 모방하여 안전 정량화 지표를 생성합니다. 이 지표는 안전 진화 의사 결정 통합 알고리즘에 입력되어, 주어진 주변 교통 환경에 대한 안전 동작을 생성합니다. 또한, 조정 가능한 안전 한계를 가진 결정통합 알고리즘이 구현되어 있습니다.

- **Performance Highlights**: 시뮬레이션 및 실제 차량 실험 결과에 따르면, 제안된 알고리즘은 다양한 복잡한 시나리오에서 안전하고 합리적인 행동을 생성할 수 있으며, 학습 기반 자율 주행 시스템의 진화적 잠재력을 잃지 않고 안전성을 보장할 수 있습니다.



### BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models (https://arxiv.org/abs/2408.12798)
- **What's New**: 본 연구에서는 Generative Large Language Models (LLMs)에 대한 backdoor 공격을 다룬 최초의 포괄적인 벤치마크인 BackdoorLLM을 소개합니다. 기존의 연구에서는 주로 비전 및 텍스트 분류 작업에 중점을 두었으나, 생성적 LLM에 대한 연구는 미비하였습니다. BackdoorLLM은 표준화된 교육 파이프라인을 갖춘 다양한 backdoor 벤치마크를 제공합니다.

- **Technical Details**: BackdoorLLM은 데이터 오염(data poisoning), 가중치 오염(weight poisoning), 숨겨진 상태 공격(hidden state attacks), 사고 체인 공격(chain-of-thought attacks) 등 다양한 공격 전략을 포함합니다. 연구는 7개 시나리오와 6개 모델 아키텍처에서 8개의 공격에 대해 200 이상 실험을 수행하였으며, 각각의 LLM 모델에 대한 backdoor 공격의 효과와 한계를 분석합니다.

- **Performance Highlights**: 주요 발견으로는 1) 다양한 LLM에 대하여 backdoor 공격이 실현 가능하고 효과적이다; 2) 비록 효과가 낮은 backdoor도 jailbreaking 공격의 성공률을 높일 수 있다; 3) 더 큰 모델은 가중치 오염 공격에 대해 더 높은 저항력을 가진다; 4) 숨겨진 상태에서의 활성화 유도는 일반성과 이전 작업 간의 전이성이 부족하다; 5) 더 강력한 추론 능력을 가진 LLM은 사고 체인 공격에 더 취약한 반면, 덜 능력 있는 모델은 효과적으로 공격하기에는 '너무 순진하다'; 6) GPT-4는 backdoor 프롬프트를 탐지하고 완화하는 데 어려움을 겪는다.



### Real-Time Posture Monitoring and Risk Assessment for Manual Lifting Tasks Using MediaPipe and LSTM (https://arxiv.org/abs/2408.12796)
Comments:
          Proceedings of the 1st International Workshop on Multimedia Computing for Health and Medicine at ACM MM'24

- **What's New**: 본 연구에서는 AI(인공지능)와 computer vision(컴퓨터 비전) 기술을 활용하여 수동 리프팅 작업을 위한 실시간 자세 모니터링 및 위험 평가 시스템을 개발하는 데 중점을 두었습니다. 기존의 자세 교정 방법들이 지연된 피드백과 개인화된 평가 부족으로 효과적이지 않은 문제를 해결하기 위해 제안된 솔루션은 AI 기반의 자세 감지, 세부 키포인트 분석, 위험 수준 결정, 사용자가 손쉽게 사용할 수 있는 웹 인터페이스를 통한 실시간 피드백을 통합합니다.

- **Technical Details**: 제안된 시스템은 MediaPipe와 LSTM(Long Short-Term Memory) 네트워크를 활용하여 리프팅 자세를 실시간으로 모니터링하고 관련 위험을 평가하며 즉각적인 교정 피드백을 제공합니다. 이 시스템은 정확하고 부정확한 동작을 기반으로 자세를 분류하는 세부 키포인트 분석을 수행합니다. 연구는 광범위한 데이터 수집, 모델 학습 및 반복 개발을 포함하며, 이를 통해 높은 정확도와 사용자 만족도를 보장합니다.

- **Performance Highlights**: 이 연구는 기존 방법론과 비교하여 실시간 피드백과 위험 평가에서 상당한 개선을 입증하여, 사용자에게 실질적이고 즉각적인 혜택을 제공하는 새로운 접근 방식을 제안합니다. 연구의 결과는 위험한 리프팅 자세 감지를 위한 LSTM 모델을 제안하고, 다양한 각도와 거리에서 촬영된 좋은 및 나쁜 리프팅 동작의 비디오로 구성된 데이터셋을 구현하였으며, 사용자 웹캠을 통해 WMSDs(Work-related Musculoskeletal Disorders)를 예방하기 위한 시스템의 활용 가능성을 보여줍니다.



### Event Detection via Probability Density Function Regression (https://arxiv.org/abs/2408.12792)
- **What's New**: 이번 연구는 전통적인 segmentation 기반의 event detection 문제를 재구성하고, heatmap regression 기법에 영감을 받아 확률 밀도(p.d.f.)를 예측하는 회귀 기반(regrression-based) 접근법을 도입하였습니다. 이로 인해 정확한 이벤트 시작 및 종료 시간 탐지의 효율성이 향상되었습니다.

- **Technical Details**: 연구에서는 Seq2Seq 모델의 구조를 활용하여 N차원 시퀀스를 출력하고, 소프트맥스 활성화 대신 선형 활성화를 사용하여 회귀 모델로 변환시킴으로써 probabilistic density function (pdf) 예측을 수행합니다. 두 개의 타겟 시리즈를 사용하여 이벤트의 시작과 종료를 포괄적으로 처리합니다.

- **Performance Highlights**: CMI Sleep Detection 대회에서 이 회귀 기반 방법이 도입되었고, 경쟁의 상위 점수 솔루션들에 의해 널리 채택되었습니다. 전통적인 segmentation 방식에 비해 여러 최신 baseline 네트워크와 데이터셋에서 뛰어난 성능을 발휘하였습니다.



### Intelligent OPC Engineer Assistant for Semiconductor Manufacturing (https://arxiv.org/abs/2408.12775)
- **What's New**: 이번 논문에서는 	extit{Intelligent OPC Engineer Assistant}라는 AI/LLM 기반의 방법론을 소개하며, 이는 반도체 제조 공정에서의 핵심 문제인 optical proximity correction (OPC) 최적화를 해결하기 위해 개발되었습니다.

- **Technical Details**: 이 방법론은 강화 학습(reinforcement learning)을 활용한 OPC 레시피 검색과 레시피 요약을 위한 맞춤형 다중 모드 에이전트 시스템(customized multi-modal agent system)을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론이 다양한 칩 디자인에서 OPC 레시피를 효율적으로 생성할 수 있음을 보여주었으며, 이는 일반적으로 여러 해에 걸친 경험을 가진 OPC 엔지니어가 전임으로 수행해야 하는 작업입니다.



### TReX- Reusing Vision Transformer's Attention for Efficient Xbar-based Computing (https://arxiv.org/abs/2408.12742)
Comments:
          12 pages

- **What's New**: 본 논문에서는 에지 컴퓨팅 시나리오에서 에너지 효율적인 배치를 위한 Vision Transformer (ViT) 최적화 프레임워크인 TReX를 제안합니다. TReX는 attention reuse 기술을 사용하여 에너지, 지연, 및 면적의 트레이드오프를 최적화합니다.

- **Technical Details**: TReX는 각 transformer 인코더에서의 attention block 출력을 재사용하는 방식으로 작동합니다. 이 과정에서 작은 변환 블록 (Transformation Block, TB)을 도입하여 데이터의 변동성을 증가시킴으로써 좋은 정확도를 유지하면서도 최소한의 하드웨어 오버헤드를 추가합니다. TReXSim이라는 IMC(In-Memory Computing) 실제 평가 플랫폼을 통해 다양한 ViT 모델을 평가합니다.

- **Performance Highlights**: TReX는 Imagenet-1k 데이터셋을 기반으로 DeiT-S (2.3배 EDAP 감소, ~1%의 정확도 손실) 및 LV-ViT-S 모델에서 상대적으로 높은 정확성을 유지하면서 EDAP를 더 높은 수준으로 감소시킵니다. 또한, NLP 작업에서 TReX는 기준선보다 2% 높은 비이상적 정확도를 기록하며 EDAP는 1.6배 감소하였습니다.



### Towards measuring fairness in speech recognition: Fair-Speech datas (https://arxiv.org/abs/2408.12734)
- **What's New**: 이 논문은 다양한 인구 통계 그룹에서의 성능을 평가할 수 있도록 돕기 위해 새롭게 Fair-Speech라는 데이터를 소개합니다. 이 데이터셋은 593명의 참가자들이 자발적으로 기록한 26.5K개의 음성 명령을 포함하고 있으며, 연령, 성별, 인종, 지역적 변형 등 다양한 인구 통계 정보를 제공합니다.

- **Technical Details**: Fair-Speech 데이터셋은 과거에 발표된 ASR(Automatic Speech Recognition) 시스템의 편향성을 해결하기 위해 구성되었습니다. 참여자들은 음성을 녹음하고 제출하였으며, 각 참가자는 자신의 인구 통계 정보를 자가 보고했습니다. 데이터는 일곱 가지 도메인(음악, 캡처, 유틸리티, 알림 제어, 메시징, 통화 및 필기)에 따라 분류되어 있습니다. 모든 데이터는 다단계로 검증된 고품질 주석을 포함하고 있습니다.

- **Performance Highlights**: Fair-Speech 데이터셋은 기존 ASR 모델의 공정성을 평가할 수 있는 기초를 제공하며, 다양한 사용자 경험을 향상시키기 위한 AI 커뮤니티의 지속적인 개선을 촉진하는 것을 목표로 하고 있습니다.



### SQL-GEN: Bridging the Dialect Gap for Text-to-SQL Via Synthetic Data And Model Merging (https://arxiv.org/abs/2408.12733)
- **What's New**: 새로운 논문에서는 다양한 SQL 방언에 적용 가능한 Text-to-SQL 시스템을 위한 SQL-GEN이라는 프레임워크를 소개합니다. 이 시스템은 방언별 튜토리얼을 사용하여 고품질의 합성 데이터를 생성하며, 이는 교육 데이터셋을 생성하는 데 효과적입니다.

- **Technical Details**: SQL-GEN은 세 단계의 파이프라인으로 구성됩니다. 첫 번째 단계에서는 SQL 키워드만 포함된 시드 템플릿을 사용하고, 두 번째 단계에서는 이들을 실제 데이터베이스의 값 및 스키마 요소로 채웁니다. 마지막으로, 품질 검사를 통해 생성된 쿼리들이 정확하게 일치하는지 확인합니다. 최신 기법인 Mixture of Experts (MoE) 초기화 방법을 통해 방언별 모델을 단일 시스템으로 통합함으로써 성능 향상을 도모합니다.

- **Performance Highlights**: 이 연구는 모든 LLM이 기존의 합성 데이터 및 인간 주석 데이터에 비해 4%에서 27%의 성능 향상을 보여 주며, PostgreSQL 및 BigQuery와 같은 덜 연구된 방언에 대해서도 실세계 데이터에 대한 평가에서 2.5%에서 7.5%의 성능 향상을 달성했습니다. 또한, 합성 데이터와 인간 주석 데이터를 결합하여 3.3%에서 5.6%의 성능 부스트를 제공했습니다.



### Learning Valid Dual Bounds in Constraint Programming: Boosted Lagrangian Decomposition with Self-Supervised Learning (https://arxiv.org/abs/2408.12695)
- **What's New**: 이 논문은 Lagrangian decomposition (LD) 기법을 통해 제약 프로그래밍에서의 바운딩 메커니즘을 개선하기 위한 자기 지도 학습 접근법을 제안합니다. 제안된 방법은 신경망(neural networks)을 사용하여 직접 Lagrangian 배수를 생성하여 바운드를 강화합니다.

- **Technical Details**: Lagrangian funtion 활용과 sub-gradient optimization 절차를 통해 마련된 자가 생성된 Lagrangian multipliers는 제약 프로그래밍 솔버의 실행 시간을 줄이고 가지치기(pruning) 효율성을 향상시킵니다.

- **Performance Highlights**: 이 새로운 접근 방식은 CP Solver의 성능을 획기적으로 개선할 잠재력을 가지고 있으며, 기존 제약 프로그래밍 기술들이 미치지 못하는 일반적이고 효율적인 이중 바운딩 메커니즘을 제공합니다.



### Unlocking Intrinsic Fairness in Stable Diffusion (https://arxiv.org/abs/2408.12692)
Comments:
          21 pages, 20 figures; First two authors contributed equally

- **What's New**: 최근의 text-to-image 모델인 Stable Diffusion에서 인종 및 성별에 대한 편향 문제를 다루며, 기존의 훈련 기반 방식의 한계를 넘어서 내재적인 공평성을 탐구했다는 점이 새롭다.

- **Technical Details**: 논문에서는 Stable Diffusion의 내재적인 공정성을 활용하기 위해 텍스트 조건을 약간 변형해 편향을 완화하는 새로운 접근 방식을 제안하였다. 이 과정은 고밀도 영역에서 생성되는 주요 속성과 대비되는 낮은 밀도의 소수 속성 간의 상관관계를 탐색하는 실험을 포함한다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 성별 편향을 효과적으로 줄이면서도 이미지 생성 능력을 유지했다. 기존 방법과 비교시, 이미지 품질과 텍스트 일치를 고려하면서 편향 완화가 이루어졌다.



### Can LLMs Understand Social Norms in Autonomous Driving Games? (https://arxiv.org/abs/2408.12680)
- **What's New**: 이 논문은 LLM(대형 언어 모델)을 활용하여 자율주행 게임에서 사회적 규범의 이해와 모델링 방법을 탐구합니다. LLM 기반의 에이전트가 마르코프 게임(Markov games)에서 의사 결정을 내리며, 이들이 사회적 규범을 형성할 수 있는지를 분석합니다.

- **Technical Details**: LLM 기반의 에이전트는 OpenAI의 GPT-4.0 API를 사용하여 두 가지 자율주행 시나리오(신호 없는 교차로와 고속도로 집단주행)에서 상호작용을 시뮬레이션합니다. 이 구조에서는 LLM이 환경 설정 및 관찰 정보에 대한 텍스트 프롬프트를 바탕으로 결정을 내립니다. 마르코프 게임의 다중 에이전트 시스템(MAS) 내에서 실험이 진행됩니다.

- **Performance Highlights**: 결과적으로 LLM 기반의 에이전트는 동적으로 변화하는 마르코프 게임 환경에서 효과적으로 대처하며, 두 시나리오 모두에서 사회적 규범이 발전함을 보여줍니다. 특히, 교차로 게임에서는 LLM 기반 에이전트가 잠재적 충돌 상황에서 보수적인 운전 정책을 채택하는 경향을 보였습니다.



### Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing (https://arxiv.org/abs/2408.12673)
- **What's New**: 본 논문에서는 기존의 전이 가능(adversarial) 공격 방법의 단점을 극복하기 위한 새로운 일반 프레임워크인 GE-AdvGAN+를 제안합니다. 이 프레임워크는 현재의 공격 방법을 통합하여 전이 가능성을 향상시키고 계산 자원 소비를 현저히 줄입니다.

- **Technical Details**: GE-AdvGAN+는 생성적 적대 신경망(Generative Adversarial Networks, GANs)의 원리를 기반으로 하며, 그래디언트 편집(gradient editing) 기법을 결합하여 다양한 전이 공격 방법을 통합합니다. GE-AdvGAN++는 평균 공격 성공률(ASR)을 47.8% 향상시켰으며, 계산 효율성을 크게 개선해 2217.7 FPS를 기록했습니다.

- **Performance Highlights**: GE-AdvGAN++는 AdvGAN과 비교하여 성능이 뛰어나며 공격 성공률에서 평균 5.9% 개선을 보여주었습니다. 여러 표준 데이터셋과 모델에서의 실험을 통해 공격 샘플의 전이 가능성과 성공률을 높이는 데 있어 GE-AdvGAN++의 효과성을 입증했습니다.



### Multilevel Interpretability Of Artificial Neural Networks: Leveraging Framework And Methods From Neuroscienc (https://arxiv.org/abs/2408.12664)
- **What's New**: 이번 연구는 생물학적 신경 시스템과 인공 신경망을 해석하는 데 있어 다중 레벨의 분석이 필요하다는 주장을 합니다. 신경과학자와 인공지능 연구자 간의 공동의 도전 과제로, 분산된 신경 메커니즘이 어떻게 복잡한 인지 및 행동을 생성하는지를 이해하는 것을 목표로 합니다.

- **Technical Details**: 연구는 David Marr의 세 가지 분석 레벨(Computational, Algorithmic/Representational, Implementation)을 기반으로 하여 생물학적 및 인공 신경 시스템을 분석하기 위한 여러 분석 도구를 제시합니다. 이를 통해 신경 시스템의 복잡성을 처리하고 구조, 계산 및 행동 간의 관계를 명확히 합니다.

- **Performance Highlights**: 다중 레벨 해석 가능성 프레임워크는 신경 시스템의 복잡성에 대처하기 위한 원칙적인 방법을 제공하며, 연구자들이 각 레벨에서의 가정과 연구 우선순위를 명확하게 할 수 있도록 돕습니다. 이 연구는 신경망의 해석 가능성을 높이는 데 중요한 기여를 할 것입니다.



### The AI Risk Repository: A Comprehensive Meta-Review, Database, and Taxonomy of Risks From Artificial Intelligenc (https://arxiv.org/abs/2408.12622)
- **What's New**: 이 논문은 인공지능(AI)의 위험에 대한 이해를 통합하기 위해 AI 위험 저장소(AI Risk Repository)를 구축하였습니다. 이 저장소는 43개의 분류체계에서 추출한 777개의 위험을 포함한 데이터베이스로, 누구나 쉽게 접근하고 수정, 업데이트할 수 있습니다.

- **Technical Details**: AI 위험의 고수준 인과 분류체계(Causal Taxonomy)는 위험을 세 가지 요인으로 분류합니다: (1) 주체: 인류(Human), AI; (2) 의도성(Intentionality): 의도적인(Intentional), 비의도적인(Unintentional); (3) 시기(Timing): 배포 전(Pre-deployment), 배포 후(Post-deployment). 중간 수준의 영역 분류체계(Domain Taxonomy)는 위험을 7개의 AI 위험 도메인으로 분류하며, 각 도메인은 다시 23개의 하위 도메인으로 나뉩니다.

- **Performance Highlights**: AI 위험 저장소는 공개 접근 가능한 데이터베이스로 AI 위험을 엄격하게 정리, 분석 및 추출한 최초의 시도로, AI 시스템이 초래하는 위험을 정의하고 감사하며 관리하는 데 있어 보다 조정되고 일관된 접근 방식을 마련할 수 있는 기반을 제공합니다.



### Ensemble Modeling of Multiple Physical Indicators to Dynamically Phenotype Autism Spectrum Disorder (https://arxiv.org/abs/2408.13255)
- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애(ASD)의 조기 탐지를 위한 비디오 기반 접근 방식을 제시합니다. 모바일 애플리케이션 GuessWhat를 통해 자연적인 가정 비디오가 수집되었으며, 이를 통해 3,000개 이상의 고품질 비디오 데이터셋이 구축되었습니다. 이 데이터셋을 기반으로 LSTM 모델을 훈련시켜 눈의 시선, 머리 위치, 얼굴의 특징을 분석합니다.

- **Technical Details**: 연구팀은 눈의 시선, 머리 위치 및 얼굴 랜드마크와 같은 입력 특징을 사용하여 LSTM 기반 모델을 훈련시켰습니다. 최종적으로 AUC(Area Under the Curve) 점수를 90%로 개선하기 위해 여러 모델의 결과를 융합하는 기술을 사용했습니다. 모든 모델은 높은 성과를 보였으며 성별 및 연령대 간의 공정성을 높였습니다.

- **Performance Highlights**: 최종 모델은 86%, 67%, 78%의 테스트 AUC를 기록하였으며, 여러 알고리즘의 융합을 통해 AUC를 90%로 증가시켜 진단의 정확성을 개선했습니다. 이는 자폐 조기 탐지의 신뢰성을 향상시키며, 주관적 평가에 대한 의존도를 줄이고 접근성과 공정성을 높입니다.



### Foundational Model for Electron Micrograph Analysis: Instruction-Tuning Small-Scale Language-and-Vision Assistant for Enterprise Adoption (https://arxiv.org/abs/2408.13248)
Comments:
          Our paper is published at ICML 2024 Workshop ML for Life and Material Science: From Theory to Industry Applications, Vienna, Austria

- **What's New**: 이 논문에서는 기존의 반도체 이미징 및 분석의 한계를 극복하기 위해 작은 규모의 다중 모드 프레임워크인 MAEMI(모세관 전자 현미경 이미지 분석)를 도입합니다.

- **Technical Details**: MAEMI는 비전-언어(vision-language) 지침 튜닝을 통해 반도체 전자 현미경 이미지 분석을 수행하며, 대형 다중 모드 모델을 활용하여 사용자 맞춤형 지침 데이터 세트를 생성합니다. 또한, 지식 증류(knowledge distillation)를 통해 대형 모델에서 소형 모델로 지식을 전이하여, 시각적 질문 응답(Visual Question Answering, VQA) 작업에서 소형 모델의 정확성을 향상시킵니다.

- **Performance Highlights**: MAEMI는 전통적인 방법보다 우수한 성능을 보여주며, 데이터 분포 변화에 적응하고 높은 처리량 스크리닝(high-throughput screening)을 지원합니다. 기업들은 MAEMI를 자체 데이터에 추가로 미세 조정하여 프라이버시와 저비용 소비자 하드웨어에서의 성능을 향상시킬 수 있습니다.



### Data Exposure from LLM Apps: An In-depth Investigation of OpenAI's GPTs (https://arxiv.org/abs/2408.13247)
- **What's New**: LLM(app) 생태계가 빠르게 성숙해지고 있으며, 다양한 사용 사례를 지원하기 위해 과도한 사용자 데이터를 수집하고 있음을 강조합니다. OpenAI의 GPT 생태계를 사례로 분석하며, LLM 기반 프레임워크를 개발하여 데이터 수집 관행을 투명하게 밝히려는 목표를 가지고 있습니다.

- **Technical Details**: OpenAI의 GPT 생태계에서 데이터 수집 관행을 정적 분석(Static Analysis)으로 분석하였습니다. 행동(Action)과 GPT들 간의 그래프 모델을 통해 간접 데이터 노출을 연구하였으며, LLM 기반의 프라이버시 정책 분석 프레임워크를 통해 데이터 수집과 프라이버시 정책의 일관성을 자동으로 점검합니다.

- **Performance Highlights**: 119,274개의 GPT와 2,596개의 유니크 Action을 분석한 결과, 82.9%의 Action이 외부 서비스에서 비롯되었으며, 일부 Action은 여러 GPT에 내장되어 사용자 활동을 추적할 수 있었음을 발견하였습니다. 적어도 9.5배 더 많은 데이터가 Action의 공존을 통해 노출되며, 수집된 데이터의 대부분이 프라이버시 정책에서 누락되어 있음을 확인하였습니다.



### JacNet: Learning Functions with Structured Jacobians (https://arxiv.org/abs/2408.13237)
Comments:
          6 pages, 3 Figures, ICML 2019 INNF Workshop

- **What's New**: 본 논문은 Neural Networks(NNs)가 입력과 출력 간 함수의 Jacobian을 직접 학습하도록 제안합니다. 이를 통해 입력-출력 매핑의 미분을 제어할 수 있는 구조를 설정할 수 있습니다.

- **Technical Details**: 우리는 C1 함수 y(x):𝒳→𝒴를 학습하는 방법을 제안하며, Jacobian이 Lipschitz와 가역성(invertibility) 조건을 만족하도록 만드는 방법을 제시합니다. 이 방식으로는 1-Lipschitz 함수의 유사한 결과도 보여줍니다.

- **Performance Highlights**: 제안된 방법을 통해 우리는 학습된 함수가 보장된 가역성을 가지며, 간단한 함수에 대한 근사를 학습할 수 있음을 실험적으로 입증하였습니다.



### Multi-Layer Transformers Gradient Can be Approximated in Almost Linear Tim (https://arxiv.org/abs/2408.13233)
- **What's New**: 이번 논문에서는 phổ biến한 transformer 구조의 self-attention 메커니즘에서 발생하는 이차적인 계산 복잡도를 해결하기 위해 새로운 빠른 gradient 계산 방법을 도입하였습니다.

- **Technical Details**: 우리의 접근 방식은 multi-layer transformer 모델의 gradient를 거의 선형 시간(n^{1+o(1)})으로 계산할 수 있게 해줍니다. 이는 기존의 O(n2d) 계산 복잡도를 대폭 줄여줍니다. 이 방법은 모든 loss function에 대해 유효하며, 전체 모델에서 제한된 근사 오차를 유지합니다.

- **Performance Highlights**: 이번 연구는 대규모 언어 모델의 gradient 계산 효율성을 개선하여 긴 컨텍스트의 언어 모델 훈련 및 배치를 보다 효과적으로 할 수 있기를 기대합니다.



### HBIC: A Biclustering Algorithm for Heterogeneous Datasets (https://arxiv.org/abs/2408.13217)
Comments:
          11 pages, 5 figures

- **What's New**: 본 논문에서는 이진, 숫자 및 범주형 속성을 포함한 복합 이질적 데이터에서 의미 있는 이중 군집(bicluster)을 발견할 수 있는 HBIC 알고리즘을 소개합니다. 이 알고리즘은 군집 생성과 모델 선택의 두 가지 단계로 구성되어 있습니다.

- **Technical Details**: HBIC 알고리즘은 두 단계로 작동합니다. 첫 번째 단계에서는 원래 행렬의 값 빈도에 따라 행과 열을 추가 및 제거하여 여러 후보 이중 군집을 반복적으로 생성합니다. 두 번째 단계에서는 이중 군집의 크기와 동질성을 고려하여 가장 적합한 이중 군집을 선택하는 두 가지 접근 방식을 도입합니다.

- **Performance Highlights**: HBIC의 성능 평가 결과, 기존 기법들과 비교하여 이질적 데이터에서 고품질의 이중 군집을 발견하는 데 있어 우수한 능력을 보였습니다. 실험은 합성 벤치마크와 전신 경화증 환자의 임상 데이터와 관련된 생물 의학 애플리케이션을 포함하고 있습니다.



### EUR-USD Exchange Rate Forecasting Based on Information Fusion with Large Language Models and Deep Learning Methods (https://arxiv.org/abs/2408.13214)
- **What's New**: 이 논문에서는 유럽연합 유로(EUR)와 미국 달러(USD) 환율 예측을 위한 새로운 IUS 프레임워크를 제안합니다. 이 프레임워크는 뉴스 및 분석에서 수집된 비구조적 텍스트 데이터와 환율 및 금융 지표에 대한 구조적 데이터를 통합하여 예측의 정확성을 높입니다.

- **Technical Details**: IUS 프레임워크는 감정 극성 점수(sentiment polarity scoring)와 텍스트의 환율 움직임 분류를 위해 대형 언어 모델(large language models)을 사용합니다. 작성된 특성은 인과관계 기반 특징 생성기(Causality-Driven Feature Generator)에 입력되어 Optuna로 최적화된 Bi-LSTM 모델을 통해 환율을 예측합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기준 모델보다 MAE를 10.69% 줄이고 RMSE를 9.56% 감소시켰습니다. 비구조적 데이터와 구조적 데이터를 결합한 데이터 융합(data fusion)의 이점도 확인해, 이러한 조합이 단독으로 구조적 데이터만 사용할 때보다 더 높은 예측 정확도를 제공함을 보여주었습니다.



### Optimal Quantum Circuit Design via Unitary Neural Networks (https://arxiv.org/abs/2408.13211)
- **What's New**: 이번 논문에서는 양자 알고리즘을 양자 회로 모델 표현으로 자동 합성하는 방법을 제안합니다. 이는 다양한 입력-출력 매핑을 통해 신경망 모델을 훈련시키는 과정을 포함하며, 훈련된 모델이 원래 알고리즘과 동등한 양자 회로 모델을 생성할 수 있음을 보여줍니다.

- **Technical Details**: 양자 알고리즘을 기본 양자 게이트의 시퀀스 형태로 분해하여 양자 회로를 구축하는 과정, 신경망의 단위 행렬( unitary matrix ) 사용, 멀티레이어 퍼셉트론(MLP) 신경망을 적용하여 양자 회로의 단위 행렬 요소를 찾는 방법이 설명됩니다.

- **Performance Highlights**: 훈련된 모델은 보지 않은 입력에 대해 해당 출력으로 거의 완벽히 매핑되는 성능을 보였으며, 양자 회로의 합성과 최적화에 대한 새로운 접근 방식을 제안합니다.



### Instruct-DeBERTa: A Hybrid Approach for Aspect-based Sentiment Analysis on Textual Reviews (https://arxiv.org/abs/2408.13202)
- **What's New**: 이번 논문은 Aspect-based Sentiment Analysis (ABSA) 방법론의 발전과 최신 Transformer 기반 모델을 활용한 하이브리드 모델 Instruct - DeBERTa를 제안합니다. 이 모델은 ATE와 ASC의 통합된 프레임워크를 제공하고, 기존 방법론의 한계를 극복하는 데 중점을 둡니다.

- **Technical Details**: ABSA는 텍스트에서 특정 측면에 대한 감정을 추출하는 기술로, 전통적인 감성 분석 방법은 종종 특정 제품의 특징에 대한 암묵적인 의견을 놓치곤 했습니다. 이 논문은 BERT, RoBERTa 및 DeBERTa와 같은 최신 Transformer 모델을 활용하여 ATE 및 ASC를 수행하는 하이브리드 모델을 개발하였으며, 이를 통해 더 나은 정확도와 신뢰성을 추구합니다.

- **Performance Highlights**: 제안된 하이브리드 모델 Instruct - DeBERTa는 SemEval restaurant 2014 및 SemEval laptop 2014 데이터셋에서 ATE 및 ASC의 통합 작업에 대해 최고 성능을 기록했습니다. 실험 결과, 모델은 모든 실험한 도메인에서 감성 분석의 정확성과 신뢰성을 획기적으로 향상시켰음을 보여줍니다.



### Accelerating the k-means++ Algorithm by Using Geometric Information (https://arxiv.org/abs/2408.13189)
- **What's New**: 이 논문에서는 기하학적 정보인 Triangle Inequality와 추가적인 norm 필터를 사용하여 정확한 k-means++ 알고리즘의 속도를 향상시키는 새로운 방법을 제안합니다. 이 방법은 두 단계 샘플링 절차를 포함하고 있으며, 실험 결과에 따르면 이 가속화된 버전이 표준 k-means++ 버전보다 방문한 포인트 수와 거리 계산에서 더 우수한 성능을 보임을 증명했습니다.

- **Technical Details**: 제안하는 알고리즘은 Triangle Inequality를 활용하고, 고차원 데이터에서 norm 분산이 큰 경우에 성능을 향상시키기 위해 추가적인 norm 기반 필터를 이용합니다. 이 두 단계 샘플링 절차는 데이터 포인트들 사이의 거리를 계산하는 데 필요한 시간을 줄이는데 중점을 두고 있습니다. 전체적인 알고리즘의 복잡도는 O(n * k * d)로 명시되어 있으며, 여기서 n은 데이터 포인트의 총 수, k는 클러스터의 수, d는 데이터의 차원입니다.

- **Performance Highlights**: 가속화된 버전은 클러스터 수가 증가함에 따라 속도 향상이 더욱 두드러지며, 저차원 데이터에 대해서는 Triangle Inequality 를 활용한 방식이 특히 효과적입니다. 또한, 메모리 성능이 실제 속도 향상에 미치는 영향을 분석한 추가 실험도 포함되어 있습니다.



### Causal machine learning for sustainable agroecosystems (https://arxiv.org/abs/2408.13155)
- **What's New**: 이 논문에서는 기후 변화에 적응하기 위한 지속 가능한 농업의 필요성을 강조하며, 예측 머신러닝(ML)의 한계를 극복하기 위해 인과 ML(causal ML)을 도입합니다. 이는 머신러닝의 데이터 처리 능력과 인과 관계를 이해하는 능력을 결합하여 지속 가능한 농업의 의사결정을 지원합니다.

- **Technical Details**: 인과 ML은 데이터에서 인과 관계를 유추하고 예측 ML 모델의 강건성을 강화하는 방법들을 포함합니다. 여기에는 두 가지 주요 단계인 인과 발견(causal discovery)과 인과 효과 추정(causal effect estimation)이 있습니다. 농업 시스템에서의 인과 질문을 정의하고, 관련 데이터를 수집하며, 관찰된 데이터의 편향을 최소화하기 위해 가정들을 설정합니다.

- **Performance Highlights**: 인과 ML은 농업 분야의 다양한 이해관계자들에게 유익한 8가지 응용 사례를 통해 실증되었습니다. 이를 통해 농부, 정책 결정자, 그리고 연구자들이 지속 가능성에 대한 증거 기반 의사결정을 할 수 있도록 지원합니다.



### ShapeICP: Iterative Category-level Object Pose and Shape Estimation from Depth (https://arxiv.org/abs/2408.13147)
- **What's New**: 최근 연구에서는 단일 깊이 이미지에서 카테고리 수준의 객체의 자세(pose)와 형태(shape)를 추정하는 기술이 주목받고 있습니다. 이 연구는 이번에 데이터 주석이 없는 방법을 바탕으로 한 반복적 추정(iterative estimation) 방법을 제안합니다.

- **Technical Details**: 이 알고리즘인 ShapeICP는 반복적 최근접점(iterative closest point, ICP) 알고리즘에 기반을 두고 있으며, 구질적(qualitative)인 객체 형태 모델인 메쉬(mesh) 기반의 활성 형태 모델을 채택하여 카테고리 수준의 자세와 형태 추정 작업을 수행합니다.

- **Performance Highlights**: ShapeICP는 데이터 기반의 접근 방식 없이도 여러 테스팅 결과에서 양호한 성능을 발휘하며, 기존의 데이터 기반 방법들이 신경망을 통해 학습한 자세 데이터를 필요로 하는 것과는 대조적으로, 데이터 부족 문제를 해결할 수 있는 새로운 가능성을 제공합니다.



### Verification of Geometric Robustness of Neural Networks via Piecewise Linear Approximation and Lipschitz Optimisation (https://arxiv.org/abs/2408.13140)
- **What's New**: 이 논문에서는 이미지 입력에 대한 기하학적 변환(geometric transformations)을 확인하는 새로운 방법을 제안합니다. 특히, 회전(rotation), 크기 조정(scaling), 비틀림(shearing), 및 이동(translation)에 대한 신경망의 강인성을 검증할 수 있는 능력을 향상시킵니다.

- **Technical Details**: 제안된 방법은 샘플링(sampling) 및 선형 근사(linear approximations)를 결합하여 픽셀 값에 대한 확실한 조각선형 제약조건(piecewise linear constraints)을 계산합니다. 이전 방법들도 포함할 수 있는 조각선형 이완(piecewise linear relaxation) 방법을 제공하여, 기하학적 변환으로 생성되는 이미지 집합을 근사합니다.

- **Performance Highlights**: 실험 결과, 제안된 구현은 보다 다양한 검증 케이스를 해결하는 동시에 계산 효율성(computational efficiency)에서 향상된 성능을 보여줍니다.



### Deep Learning at the Intersection: Certified Robustness as a Tool for 3D Vision (https://arxiv.org/abs/2408.13135)
Comments:
          This paper is an accepted extended abstract to the LatinX workshop at ICCV 2023. This was uploaded a year late

- **What's New**: 이번 논문에서는 기계 학습에서 인증된 강인성(cerified robustness)과 3D 객체 모델링(3D object modeling) 간의 새로운 연결 고리를 제안합니다. 특히, 공간의 점유율을 나타내는 분류기의 Maximal Certified Radius (MCR)와 공간의 Signed Distance Function (SDF)의 흥미로운 상관관계를 강조합니다.

- **Technical Details**: 우리는 인증 방법인 랜덤 스무딩(randomized smoothing, RS)을 활용하여 SDF를 계산할 것을 제안합니다. 그러나 RS의 높은 계산 비용이 실제 사용을 방해하기 때문에, 저차원(low-dimensional) 애플리케이션에서 효율적으로 RS를 실행할 수 있는 알고리즘을 제안합니다. 이는 미리 계산된 복셀 그리드(voxel grids)에서 가우시안 스무딩(Gaussian smoothing)으로 기본 연산을 표현하여 가능하게 합니다.

- **Performance Highlights**: 우리의 접근법은 새로운 시점 합성(novel view synthesis)의 증명 개념 실험을 통해 검증되었습니다. 실험 결과, 우리의 방법이 장면의 유용한 표현을 학습하면서 바람직한 시각적 결과를 유지할 수 있음을 보여줍니다.



### DeTPP: Leveraging Object Detection for Robust Long-Horizon Event Prediction (https://arxiv.org/abs/2408.13131)
- **What's New**: 이번 논문에서는 전통적인 Marked Temporal Point Processes (MTPP) 모델의 한계를 극복하기 위해 DeTPP (Detection-based Temporal Point Processes)라는 새로운 접근 방식을 제안합니다. DeTPP는 컴퓨터 비전의 객체 탐지 기술에서 영감을 받아 다중 미래 이벤트를 병렬로 예측하며, 독창적인 매칭 기반 손실 함수를 통해 예측의 강건성과 다양성을 높입니다.

- **Technical Details**: DeTPP는 이벤트 발생 확률, 이벤트 레이블 분포 및 시간 이동 분포를 예측하는 확률적 프레임워크를 활용하여 이벤트 시퀀스의 복잡성을 포착합니다. 이 모델은 특정 시간 범위 내에서 K개의 미래 이벤트를 예측하도록 설계되었으며, 이는 모델의 하이퍼파라미터입니다. DeTPP는 MTPP 모델들과 비교하여 더 나은 성능을 보여줍니다.

- **Performance Highlights**: DeTPP는 기존 MTPP 모델 및 next-K 접근 방식보다 훨씬 우수한 성능을 나타내며, 긴 예측 범위에서 새로운 최첨단 결과를 달성합니다. 이 연구는 GitHub에서 공개된 DeTPP 구현체를 통해 일반에 제공됩니다.



### Semantic Variational Bayes Based on a Semantic Information Theory for Solving Latent Variables (https://arxiv.org/abs/2408.13122)
Comments:
          21 pages, 7 figures, 39 references

- **What's New**: 이번 논문에서는 Variational Bayesian 방법(VB)을 대체하기 위해 Semantic Variational Bayes(SVB) 방법을 제안합니다. SVB는 이전에 제안된 Semantic Information Theory에 기반하여 개발되었습니다.

- **Technical Details**: SVB는 최소 상호 정보(minimum mutual information) R(G)를 최적화하기 위한 다양한 제약 함수(likelihood, truth, membership, similarity, distortion functions)를 사용합니다. 또한, SVB는 최대 정보 효율성(G/R) 기준을 통해 모델 파라미터를 최적화합니다.

- **Performance Highlights**: SVB는 동일 작업에서 VB보다 계산적으로 간단합니다. 논문에서는 혼합 모델(mixture model)의 수렴, 데이터 압축(data compression)에서의 SVB 응용, 그리고 제어 작업에서 최대 엔트로피 제어(maximum entropy control) 및 강화 학습(reinforcement learning)에서의 유용성을 수치적으로 입증했습니다.



### An Introduction to Cognidynamics (https://arxiv.org/abs/2408.13112)
Comments:
          This paper is related to the invited talk I gave at the Third Conference on Lifelong Learning Agents (CoLLAs 2024) on the 29th of July 2024

- **What's New**: 본 논문은 시간에 따라 최적 목표에 의해 주도되는 인지 시스템의 역학인 'Cognidynamics'를 소개합니다. 이 이론은 동적 프로그래밍(Dynamic Programming) 일반 프레임워크에서 개발되며, 이는 계산 법칙들이 고전 해밀토니안 방정식에 의해 규정되도록 합니다.

- **Technical Details**: Cognidynamics는 동적 신경망(Dynamic Neural Networks)으로 모델링된 인지 에이전트에서 발생하는 신경 확산(neural propagation) 스킴을 설명합니다. 학습 과정은 에너지원(Energy Exchange)과 관련하여 해석되며, 에너지 소산(Energy Dissipation) 및 주의 집중의 메커니즘과 의식적 행동(conscious behavior) 간의 관계를 강조합니다.

- **Performance Highlights**: 이론적으로, 적절한 주의 집중 메커니즘을 도입해야 학습이 발생할 수 있으며 이는 에이전트의 에너지 축적을 제어하는 데 필수적인 역할을 합니다. 또한, 이 논문은 기존의 데이터 집합 수집을 활용하는 인공지능(AI) 방식에서 벗어나, 수집 없는(machine learning without collection) AI 프로토콜이 지능적 기술 습득을 위한 가능성을 보여줍니다.



### Map-Free Visual Relocalization Enhanced by Instance Knowledge and Depth Knowledg (https://arxiv.org/abs/2408.13085)
Comments:
          17 pages,6 figures

- **What's New**: 이 논문은 모노큘러 이미지를 사용하는 relocalization의 성능을 높이기 위해 instance knowledge와 depth knowledge를 활용한 새로운 map-free relocalization 방법을 제안합니다. 전통적인 지도 기반 방법의 한계를 극복하여, 사전 제작된 지도를 필요로 하지 않고도 정확한 위상(위치와 방향)을 추정할 수 있습니다.

- **Technical Details**: 제안된 방법은 instance 기반의 매칭 정보를 활용하여 전체 매칭 결과를 개선합니다. 이로 인해 서로 다른 객체 간의 잘못된 매칭 가능성이 크게 줄어듭니다. 또한, 단일 이미지로부터 추정한 메트릭 깊이를 사용하여 메트릭 오류를 줄이고 스케일 회복 정확도를 개선하는 방식으로 작동합니다. 이를 위해 다단계 매칭 방식을 통합하여 instance-level과 feature-level 접근 방식을 결합합니다.

- **Performance Highlights**: 제안된 방법은 map-free 검증 세트에서 가장 뛰어난 결과를 보였으며, Map-free Visual Relocalization 챌린지의 어려운 테스트 세트에서도 경쟁력 있는 성능을 입증했습니다. 이 방법은 기존의 최첨단 알고리즘들을 초월하는 외연성 성능을 나타내며, 복잡한 환경에서의 실험을 통해 그 효과를 보여주었습니다.



### Avatar Visual Similarity for Social HCI: Increasing Self-Awareness (https://arxiv.org/abs/2408.13084)
- **What's New**: 본 논문은 가상 교육 환경에서 아바타의 시각적 유사성이 자기 인식(self-awareness)을 증가시키는 데미치는 영향을 심층적으로 탐구합니다. 특히, 아바타의 외모를 조작함으로써 자기 인식이 어떻게 변화하는지를 연구했습니다.

- **Technical Details**: 이 연구에서는 아바타의 시각적 유사성을 조작하기 위한 체계적인 방법론을 개발하고, 이는 참가자와 아바타 간의 시각적(face)특징의 조작으로 이루어집니다. 연구에선 세 가지 다른 정도의 유사성을 가진 아바타(약한, 중간, 강함)를 사용하여 실험을 진행했습니다. 실험 결과, 유사성의 정도 변화가 자기 인식의 세 가지 선행 요소(명시적 식별, 잠재적 정서적 식별)에 미치는 영향을 측정했습니다.

- **Performance Highlights**: 결과적으로, 약한 유사성을 가진 조작과 강한 유사성 및 무작위 아바타 간에 의미 있는 차이를 발견했습니다. 아바타 시각적 유사성이 높아질수록 자기 인식의 선행 요인에 긍정적인 영향을 미친다고 결론지었습니다.



### Multivariate Time-Series Anomaly Detection based on Enhancing Graph Attention Networks with Topological Analysis (https://arxiv.org/abs/2408.13082)
Comments:
          10 pages, 5 figures, to be published in CIKM 2024

- **What's New**: 이 논문에서는 복잡한 멀티변량 시계열 이상 탐지 문제를 해결하기 위해 TopoGDN이라는 새로운 모델을 소개합니다. 이 모델은 강화된 Graph Attention Network (GAT)를 기반으로 하여 시간 및 특징 차원을 세밀하게 분석합니다. 또한, 다중 스케일 시계열 컨볼루션 모듈과 증강된 GAT를 도입해 복잡한 특성 간 의존성을 관리합니다.

- **Technical Details**: TopoGDN은 목표하는 이상의 정확도를 달성하기 위해 다음과 같은 주요 구성 요소로 이루어져 있습니다: 1) 다중 스케일 시계열 컨볼루션 모듈, 2) 그래프 구조 학습 모듈, 3) 위상적 특징 주의 모듈, 4) 이상 점수 산출 모듈. 이 모델은 그래프 구조를 통해 시간 및 특성 간의 의존성을 효율적으로 모델링할 수 있습니다.

- **Performance Highlights**: 실험 결과, TopoGDN은 네 개의 벤치마크 데이터셋에서 기준 모델보다 뛰어난 성능을 보였습니다. 이는 매우 복잡한 관계와 동적 변화를 포함하는 대규모 데이터셋에 대해 강력한 이상 탐지 기능을 제공하는 가능성을 보여줍니다.



### AEMLO: AutoEncoder-Guided Multi-Label Oversampling (https://arxiv.org/abs/2408.13078)
- **What's New**: 이 논문에서는 다중 레이블 (multi-label) 데이터의 클래스 불균형 문제를 해결하기 위해 AutoEncoder 기반의 새로운 오버샘플링 기법인 AEMLO를 제안합니다.

- **Technical Details**: AEMLO는 인코더-디코더(encoder-decoder) 아키텍처를 사용하여 입력 데이터를 저차원 피처 공간으로 인코딩하고 잠재 표현(latent representations)을 학습한 후, 이를 원래 차원으로 복원하여 새로운 데이터를 생성합니다. 또한, 다중 레이블 샘플링을 최적화하기 위해 특별한 목적 함수를 설계하였습니다.

- **Performance Highlights**: AEMLO는 기존의 최첨단 기법들과 비교하여 성능이 우수하다는 것을 다양한 경험적 연구를 통해 입증하였습니다.



### Hierarchical Spatio-Temporal State-Space Modeling for fMRI Analysis (https://arxiv.org/abs/2408.13074)
- **What's New**: 최근 깊은 학습(structured deep learning)의 발전으로 구조화된 상태 공간 모델, 특히 Mamba 아키텍처가 선형 복잡성을 유지하면서도 뛰어난 성능 향상을 보여주고 있습니다. 본 연구에서는 fMRI(기능적 자기 공명 영상)를 활용하여 신경학적 바이오마커를 발견하기 위한 Mamba 기반 모델인 Functional Spatiotemporal Mamba (FST-Mamba)를 제안합니다.

- **Technical Details**: FST-Mamba는 fMRI에서 유도된 동적 기능 네트워크 연결성(dFNC)에 초점을 맞추며, 독특한 연결성 집계를 위한 구성 요소 별 다양한 규모 집계(CVA) 메커니즘을 도입합니다. 또한, 각 기능적 연결의 상대적 위치를 인코딩하기 위해 대칭 로타리 위치 인코딩(SymRope)을 제안합니다.

- **Performance Highlights**: FST-Mamba 모델은 Human Connectome Project (HCP), UK BioBank (UKB), Alzheimer’s Disease Neuroimaging Initiative 데이터셋을 기반으로 한 다양한 분류 및 회귀 작업에서 기존의 방법들보다 유의미한 성능 향상을 보여주었습니다.



### cc-DRL: a Convex Combined Deep Reinforcement Learning Flight Control Design for a Morphing Quadrotor (https://arxiv.org/abs/2408.13054)
- **What's New**: 본 연구에서 제안하는 cc-DRL 비행 제어 알고리즘은 모양을 변형시킬 수 있는 쿼드로터의 자세와 위치를 제어하기 위한 혁신적 접근 방식을 사용하고 있습니다. 이 알고리즘은 모델이 필요 없는 기술인 deep reinforcement learning (DRL)과 convex combination (CC) 기법을 결합하여 복잡한 비행 동역학을 효과적으로 관리합니다.

- **Technical Details**: 제안된 cc-DRL 비행 제어 알고리즘은 proximal policy optimization (PPO) 알고리즘을 활용하여 사전 선택된 대표적인 팔 길이 모드에 대한 최적 비행 제어 법칙을 오프라인으로 훈련시킵니다. 이 알고리즘은 각 팔 길이 모드에 대응하는 비행 제어 스킴을 설계하며, 비선형 쿼드로터 동역학을 해결하기 위해 interpolation을 사용합니다. 최종적으로, optimal control laws은 convex combination 기법을 사용하여 온라인 비행 제어 스킴을 생성합니다.

- **Performance Highlights**: 시뮬레이션 결과는 제안된 cc-DRL 비행 제어 알고리즘이 기존 알고리즘보다 뛰어난 비행 성능을 제공함을 보여줍니다. 이는 또한 구조적으로 비대칭인 변화에 적응하는 변형 쿼드로터의 복잡한 동역학을 효과적으로 처리할 수 있음을 입증합니다.



### SpeechPrompt: Prompting Speech Language Models for Speech Processing Tasks (https://arxiv.org/abs/2408.13040)
Comments:
          Published in IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

- **What's New**: 이번 논문에서는 음성 처리 분야에서 사전 훈련된 음성 언어 모델(Speech LMs)에 대한 prompting 기법을 처음으로 탐구하였습니다. 기존의 텍스트 기반 모델에서 발전한 이 방법은 단순한 입력 변화를 통해 다양한 다운스트림 작업을 수행할 수 있도록 해줍니다.

- **Technical Details**: 작업을 음성-단위 생성(task)을 통해 통합함으로써, 음성 분류, 시퀀스 생성 및 음성 생성 작업을 통합하여 처리합니다. 본 연구는 단순한 선형 변환으로 작동하는 learnable verbalizer를 활용하여 단어의 Rich 정보를 처리하고 있으며, 사전 훈련된 모델의 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안한 prompting 방법이 self-supervised learning 모델 기반의 fine-tuning 방법과 유사한 성능을 보였으며, 특히 few-shot 학습 환경에서도 유망한 결과를 나타냈습니다. 또한, 모든 작업을 통합된 파이프라인 내에서 수행하며 파라미터 효율성도 우수합니다.



### VFM-Det: Towards High-Performance Vehicle Detection via Large Foundation Models (https://arxiv.org/abs/2408.13031)
Comments:
          In Peer Review

- **What's New**: 본 논문에서는 전이 학습된 차량 모델(VehicleMAE)과 대형 언어 모델(T5)을 결합한 새로운 차량 탐지 프레임워크 VFM-Det을 제안합니다. 기존 탐지기는 일반적인 객체 탐지 모델을 기반으로 하고 있어 최적의 성능을 내지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: VFM-Det은 지역 제안 기반 탐지 프레임워크를 따르며, VehicleMAE를 통해 제안의 특징들을 향상시키고, VAtt2Vec 모듈을 통해 차의 의미적 속성을 예측하여 대조 학습(contrastive learning)을 통해 시각적 특징을 강화합니다.

- **Performance Highlights**: Cityscapes 데이터셋에서 AP_{0.5}는 5.1% 향상되었고, AP_{0.75}는 6.2% 향상된 결과가 있는 등 제안된 차량 탐지기가 기존 방법들보다 뛰어난 성능을 보였습니다.



### BoostTrack++: using tracklet information to detect more objects in multiple object tracking (https://arxiv.org/abs/2408.13003)
- **What's New**: 이번 논문에서는 기존의 BoostTrack 방법의 한계를 분석하고, True Positive detection을 개선하기 위한 새로운 기법을 제안합니다. 이를 통해 다중 객체 추적(MOT)에서의 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 shape, Mahalanobis distance 및 새로운 soft BIoU 유사성 조합을 활용하여 진정한 양성 검출(true positive detections) 선택을 향상합니다. 또한 soft detection confidence boost 기술을 도입하여 새로운 confidence 점수를 계산하고, 업데이트 되지 않은 tracklet과의 유사성 측정에 기반한 변동 유사성 threshold를 설정합니다.

- **Performance Highlights**: BoostTrack+ 기준에 통합된 우리의 방법은 MOT17 데이터셋에서 거의 최신 기술 수준의 결과를 달성하였으며, MOT20 데이터셋에서는 새로운 최신 기술 수준의 HOTA 및 IDF1 점수를 기록했습니다.



### RIFF: Inducing Rules for Fraud Detection from Decision Trees (https://arxiv.org/abs/2408.12989)
Comments:
          Published as a conference paper at RuleML+RR 2024

- **What's New**: 이 논문에서는 금융 사기 탐지를 위해 RIFF라는 새로운 rule induction (규칙 유도) 알고리즘을 제안합니다. 이 알고리즘은 기존의 decision trees (결정 트리)를 활용하여 낮은 false positive rate (FPR) 규칙 집합을 생성합니다.

- **Technical Details**: RIFF는 두 단계로 구성되어 있습니다: 첫 번째 단계에서는 induction set (유도 집합)에서 학습된 트리 기반 모델의 leaves (리프)로부터 후보 규칙 집합을 유도합니다. 두 번째 단계에서는 외부 선택 집합에서의 성과에 기반하여 가장 높은 정밀도를 가진 규칙을 탐욕적으로 선택합니다. RIFF는 CART 및 FIGS와 같은 기존 결정 트리 알고리즘과 비교됩니다.

- **Performance Highlights**: 실험 결과, RIFF로 유도된 규칙들은 원본 모델의 성과를 유지하거나 개선하면서 복잡성을 상당히 줄일 수 있음을 보여주었고, 전문가가 수동으로 조정한 규칙보다 뛰어난 성과를 기록했습니다.



### Zeoformer: Coarse-Grained Periodic Graph Transformer for OSDA-Zeolite Affinity Prediction (https://arxiv.org/abs/2408.12984)
Comments:
          7 pages, 5 figures

- **What's New**: 본 논문에서는 기존의 기계 학습 방법들이 크리스탈 구조의 세부 변동성을 정확하게 표현하지 못하는 문제를 해결하기 위해 Zeoformer라는 새로운 모델을 제안합니다. 이 모델은 원자 중심의 단위 세포를 재구성하고, 이를 통한 쌍의 거리(pairwise distances)를 인코딩하여 OSDA-제올라이트 쌍의 특성을 더 정확하게 예측합니다.

- **Technical Details**: Zeoformer 모델은 잔여 재구성된 단위 세포를 통해 coarse-grained crystal periodicity와 fine-grained local variability를 동시에 포착하는 방식을 취합니다. 각 원자를 중심으로 단위 세포를 재구성하고, 이 중심 원자와 다른 원자 간의 쌍 거리(pairwise distances)를 인코딩하여 전체 구조의 차이를 효과적으로 표현합니다.

- **Performance Highlights**: Zeoformer는 OSDA-제올라이트 쌍 데이터셋 및 두 가지 유형의 크리스탈 소재 데이터셋에서 뛰어난 성능을 보였으며, OSDA와 목표 제올라이트 토폴로지 간의 친화도 예측 분야에서 최초의 기계 학습 접근법으로 강조됩니다.



### Open Llama2 Model for the Lithuanian Languag (https://arxiv.org/abs/2408.12963)
Comments:
          12 pages, 8 figures, 5 tables

- **What's New**: 이번 논문에서는 리투아니아어를 위한 최초의 오픈 Llama2 대규모 언어 모델(LLM)과 이에 수반되는 질문/답변(Q/A) 데이터셋 및 유명 LLM 벤치마크의 번역을 제안하고 설명합니다. 또한 오픈 지역 LLM의 전반적인 역할과 제안된 LLM의 훈련 과정을 설명합니다.

- **Technical Details**: 제안된 모델은 Llama2 아키텍처를 기반으로 하며, 7B (억 단위) 및 13B 파라미터를 갖고 있습니다. 훈련은 두 단계로 나뉘며, 첫 번째 단계는 자가 회귀적인 사전 훈련을 포함하고, 두 번째 단계는 인간-주석 데이터와 RLHF(Reinforcement Learning with Human Feedback) 방법론을 사용하여 수행 됩니다. 훈련 과정은 최소 2048 토큰의 컨텍스트 길이를 지원하며, 8xH100 GPU에서 진행되었습니다.

- **Performance Highlights**: 제안된 LLM의 평가를 통해 얻은 perplexity 수치는 다른 현대 오픈 LLM들에 비해 경쟁력을 갖추고 있음을 보여줍니다. 또한, 고품질 사전 훈련 데이터셋이 이러한 벤치마크에서 효율적으로 성능을 발휘하는 모델을 얻기 위해 필수적임을 확인했습니다.



### Multimodal Contrastive In-Context Learning (https://arxiv.org/abs/2408.12959)
- **What's New**: 본 논문에서는 멀티모달(multi-modal) 대비(in-context) 학습(framework) 새로운 해석을 제안하여 LLM의 ICL(inner workings)에 대한 이해를 향상시키고자 한다.

- **Technical Details**: 논문에서는 대비 학습(contrastive learning)을 사용한 ICL 해석을 제안하며, 멀티모달 입력 형식의 편향(bias)을 해결하기 위한 분석 프레임워크를 개발하였다. 또한, Anchored-by-Text ICL 전략을 통해 자원 제약 환경에서 효과적으로 증오(memes) 탐지를 수행할 수 있음을 보여준다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 다양한 시나리오에서 ICL 성능을 현저히 향상시키며, 특히 자원이 제한된 환경 및 도전적인 작업(task)에서 강력한 성능을 발휘함을 증명하였다.



### Informational Embodiment: Computational role of information structure in codes and robots (https://arxiv.org/abs/2408.12950)
- **What's New**: 이번 논문은 로봇의 몸체를 정보의 전달 통로로 간주하고, 센서의 정확성, 모터의 배치, 몸체의 형태 등이 정보를 어떻게 구성하는지를 다룹니다. 이를 통해 정보 이론에 기반한 로봇 설계 및 알고리즘 최적화의 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문에서는 Shannon의 소스 코딩 정리를 활용하여 다양한 몸체의 센서 모터 정보 구조를 비교하고, Entropy Maximization (EM) 원리를 통해 로봇의 성능을 최대화할 수 있는 효율적인 코드를 소개합니다.

- **Performance Highlights**: 효율적인 코드는 정보의 불확실성, 중복성 및 압축성 문제를 다루는데 유용하며, 이는 지능형 시스템의 인식 및 제어에 적용될 수 있습니다. 또한, 생물학적 시스템의 운동 제어에 대한 인사이트를 제공하며, 로봇이 이러한 원리를 활용하여 더 높은 성능을 달성할 수 있는 가능성을 제시합니다.



### Causal-Guided Active Learning for Debiasing Large Language Models (https://arxiv.org/abs/2408.12942)
Comments:
          ACL main conference

- **What's New**: 최근의 분석에 따르면, 현재의 Generative Large Language Models (LLMs)는 데이터셋의 편향을 포착하고 그렇게 생성된 결과물의 일반화 가능성과 유해성을 저하시킬 수 있습니다. 본 연구에서는 Active Learning과 인과 메커니즘을 결합한 Casual-guided Active Learning (CAL) 프레임워크를 제안합니다.

- **Technical Details**: CAL 프레임워크는 LLMs가 스스로 편향된 샘플을 자동으로 식별하고, 이러한 편향 패턴을 유도하는 방식으로 작동합니다. 이를 통해, LLMs는 생성 과정에서 데이터셋의 편향을 활용하지 않도록 방지하는 낮은 비용의 In-Context Learning (ICL) 방법을 적용합니다.

- **Performance Highlights**: 실험 결과, CAL을 통해 LLMs는 전형적인 편향 인스턴스를 효과적으로 인식할 수 있으며, 다양한 편향 패턴을 유도하여 LLMs의 일반화 가능성과 안전성을 향상시킬 수 있음을 보여주었습니다.



### Smooth InfoMax -- Towards easier Post-Hoc interpretability (https://arxiv.org/abs/2408.12936)
- **What's New**: 이번 논문에서는 자가 지도 학습(self-supervised representation learning)을 위한 새로운 방법인 Smooth InfoMax (SIM)을 소개합니다. 이 방법은 신경망의 다양한 깊이에서 학습된 표현의 해석 가능성(interpretability)을 제약 조건으로 포함한 것이 특징입니다.

- **Technical Details**: SIM의 구조는 확률적 모듈로 분리되어 있으며, 각 모듈은 InfoNCE 경계(bound)를 사용하여 지역적으로 최적화됩니다. VAE(Variational Autoencoders)에서 영감을 받아, 이 모듈에서 생성된 표현은 가우시안 분포(Gaussian distributions)로부터의 샘플이 되도록 설계되며, 표준 정규 분포(standard normal distribution)에 가깝게 조정됩니다. 이러한 과정은 매끄럽고 예측 가능한(latent space) 잠재 공간을 생성하여, 후속(post-hoc) 분석을 위한 디코더(decoder)를 통해 탐색이 가능하게 합니다.

- **Performance Highlights**: SIM은 순차 음성 데이터(sequential speech data)에서 평가되었으며, 기존의 덜 해석 가능한 Greedy InfoMax (GIM)와 경쟁력을 갖춘 성능을 보여주었습니다. 또한, SIM의 내부 표현(internal representations)을 분석하여, 포함된 정보가 표현 전반에 덜 얽혀 있고(dimensions), 더 작은 하위 집합(subset)에서 집중되어 있다는 것을 입증하여, SIM의 향상된 해석 가능성을 강조합니다.



### CSPs with Few Alien Constraints (https://arxiv.org/abs/2408.12909)
- **What's New**: 이 논문에서는 이질적인 구조 \\\mathcal{B}를 갖는 제약 만족 문제(CSP)인 CSP(\\mathcal{A} \cup \\\mathcal{B})의 매개변수화된 복잡성을 분석합니다. 특히, k개의 이질 제약조건을 허용하는 상황에서 이 문제의 복잡성을 탐구합니다.

- **Technical Details**: CSP(\\mathcal{A} \cup \\\mathcal{B})문제는 안정적이고 처리 가능한 배경 구조(\\mathcal{A})와 이질적인 구조(\\mathcal{B])의 제약을 통합하여 다루며, 매개변수화된 복잡성(Parameterized Complexity) 이론을 적용합니다. 이 논문은 알제브라적 기법(Algebraic Methods)과 논리적 기법(Logical Methods)을 결합하여 복잡도 분류를 위한 새로운 관점을 제시합니다.

- **Performance Highlights**: 제안된 방법론은 임의의 유한 구조에 대해 FPT(고정 매개변수 처리 가능)와 pNP(지속 가능한 비결정적 다항식) 문제 간의 이분법을 제공하며, 특히 Boolean 구조 및 (\\mathbb{N},=)의 일차보다 단순한 축소에 대한 명확한 결과를 도출합니다.



### Frequency-aware Feature Fusion for Dense Image Prediction (https://arxiv.org/abs/2408.12879)
Comments:
          Accepted by TPAMI (2024)

- **What's New**: 이 논문에서는 현재의 기법들이 갖고 있는 intra-category inconsistency(내부 범주 불일치) 및 boundary displacement(경계 이동)에 대한 해결책으로 FreqFusion(주파수 인식 특징 융합)이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: FreqFusion은 세 가지 주요 구성 요소인 Adaptive Low-Pass Filter (ALPF) generator, offset generator 및 Adaptive High-Pass Filter (AHPF) generator로 구성되어 있습니다. ALPF는 객체 내부의 고주파 성분을 억제하여 내부 불일치를 감소시키고, offset generator는 인접한 일관된 특징으로 교체하여 불일치를 수정하며, AHPF는 경계 세부 정보를 복원하여 더욱 선명한 경계를 제공합니다.

- **Performance Highlights**: FreqFusion은 여러 밀집 예측 작업에서 이전의 최첨단 기법들을 능가하는 성능 향상을 보여줍니다. 예를 들어, semantic segmentation(의미 기반 분할) 작업에서는 SegFormer-B1과 SegNeXt-T의 mIoU(Mean Intersection over Union)를 각각 2.8과 2.0만큼 향상시켰으며, object detection(객체 탐지)에서는 Faster R-CNN-R50의 AP(average precision)를 1.8 향상시켰습니다.



### Obfuscated Memory Malware Detection (https://arxiv.org/abs/2408.12866)
Comments:
          8 pages 9 figures presented in IEEE CCEM Conference paper

- **What's New**: 이번 논문에서는 인공지능(AI)과 머신 러닝(Machine Learning)을 활용하여 특정한 변형된 맬웨어(obfuscated malware)에 의해 유발된 사이버 공격을 탐지하고 완화하는 방법을 제안합니다.

- **Technical Details**: 우리는 맬웨어 샘플에 대한 메모리 분석(memory analysis) 및 메모리 특징 엔지니어링(memory feature engineering) 실험을 수행했습니다. 이 연구에서는 이진 분류(binary classification)를 통해 샘플이 맬웨어인지 여부를 식별하며, 변형된 맬웨어의 유형을 판별하기 위해 다중 클래스 분류(multi-class classification) 모델을 제안합니다.

- **Performance Highlights**: 제안된 Classic Random Forest 알고리즘을 사용하여 세 가지 유형의 변형된 맬웨어를 탐지하는 모델을 개발하였으며, 이는 89.07%의 정확도로 성능을 보였습니다. 본 연구의 범위 내에서 단일 모델이 여러 변형된 맬웨어를 분류한 사례는 매우 드뭅니다.



### Memory-Efficient LLM Training with Online Subspace Descen (https://arxiv.org/abs/2408.12857)
Comments:
          Code is available at this https URL

- **What's New**: 최근 메모리 효율적인 LLM(대형 언어 모델) 훈련 알고리즘들이 크게 인기를 얻고 있다. 이 방법들은 기울기의 저랭크(低秩, low-rank) 구조를 활용하여 SVD(특이값 분해, Singular Value Decomposition)를 통해 옵티마이저의 상태를 부분 공간으로 매핑한다. 본 연구는 어떤 업데이트 규칙에 대해서도 수렴 보장을 제공하는 최초의 연구이다.

- **Technical Details**: 본 연구에서는 임의의 업데이트 규칙에 대한 수렴 보장을 첫 번째로 제공한다. 이는 해밀토니안 내림차순(Hamiltonian Descent) 프레임워크로 분석할 수 있는 다양한 옵티마이저에 일반적으로 적용 가능하다. 우리의 접근 방식은 정기적인 업데이트를 SVD 대신 온라인 PCA(주성분 분석, Principal Component Analysis)를 사용하여 업데이트함으로써 유연성과 최소한의 훈련 오버헤드를 제공한다.

- **Performance Highlights**: 온라인 서브스페이스 디센트(Online Subspace Descent) 방법은 C4 데이터셋에서 60M에서 1B 파라미터의 LLaMA 모델을 사전 훈련하는 데 있어, 최신 저랭크 훈련 방법들보다 낮은 perplexity와 더 나은 다운스트림 태스크 성능을 보인다. 이는 전체 랭크 기준선과의 격차를 좁히는 결과를 가져왔다.



### Online Fair Division with Contextual Bandits (https://arxiv.org/abs/2408.12845)
Comments:
          We study an online fair division problem that has a large number of items with only a few copies of each item and propose contextual bandits-based algorithms with sub-linear regret guarantees

- **What's New**: 이 논문은 온라인에서 여러 에이전트 간에 공정하게 자원을 분배하는 문제를 다룹니다. 기존 알고리즘은 자원의 수가 적고 많은 복제본이 있는 경우에만 효과적이었으나, 많은 실제 응용프로그램에서는 아이템 수가 많고 복제본이 적은 경우가 많습니다.

- **Technical Details**: 저자들은 contextual bandits를 이용하여 공정한 분배 문제를 모델링하고, 아이템-에이전트 특징들과 그 유틸리티 간의 상관관계 구조를 가정합니다. 그리고 sub-linear regret 보장이 가능한 알고리즘을 제안합니다. 이 알고리즘은 'goodness function'을 도입하여 공정성과 효율성의 균형을 유지하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 다양한 설정에서 이론적인 결과를 뒷받침하며 유틸리티 추정에 대한 낙관적인 추정치를 제공하여 최적의 에이전트에게 아이템을 할당하는 데 성공했습니다.



### Predicting Affective States from Screen Text Sentimen (https://arxiv.org/abs/2408.12844)
Comments:
          7 pages

- **What's New**: 이 논문은 모바일 센싱 기술을 활용하여 스마트폰에서 수집된 데이터를 통해 사용자 감정 상태를 예측하는 방법을 다룹니다. 특히, 스마트폰에서 노출되는 텍스트 내용이 개인의 감정에 미치는 영향을 분석한 연구는 새로운 관점을 제시합니다.

- **Technical Details**: 2023년 호주 대학 학생들을 대상으로 진행된 디지털 표현 현상 연구의 데이터를 바탕으로, 본 연구는 큰 언어 모델(LLM)을 활용하여 선형 회귀(linear regression), 제로샷(zero-shot), 다중 샷(multi-shot) 프로밍을 통해 스크린 텍스트와 감정 상태 간의 관계를 분석했습니다.

- **Performance Highlights**: 다중 샷 프로밍이 선형 회귀와 제로샷 프로밍보다 상당히 더 우수한 성능을 보이며, 이는 감정 예측에서 맥락의 중요성을 강조합니다. 이는 스마트폰 사용과 웰빙(wellbeing) 이해를 향상시키기 위한 기초 자료를 제공합니다.



### COVID-19 Probability Prediction Using Machine Learning: An Infectious Approach (https://arxiv.org/abs/2408.12841)
- **What's New**: 현재 진행 중인 COVID-19 전염병은 백신의 광범위한 사용에도 불구하고 세계 공공 보건에 큰 도전을 계속해서 제기하고 있습니다. 이 연구는 COVID-19 감염 확률을 예측하기 위한 고급 머신 러닝(ML) 기법의 적용을 다룹니다.

- **Technical Details**: 본 연구에서는 XGBoost, LGBM, AdaBoost, Logistic Regression, Decision Tree, RandomForest, CatBoost, KNN, Deep Neural Networks(DNN) 등 다양한 ML 모델의 효과를 면밀히 조사하였습니다. 4000개의 샘플로 구성된 데이터셋을 활용하여 3200개는 훈련용, 800개는 테스트용으로 배정하였습니다.

- **Performance Highlights**: DNN은 89%의 뛰어난 정확도로 COVID-19 조기 감지에서 가장 높은 성능을 보이는 모델로 나타났습니다. 이는 복잡한 데이터 패턴을 활용하여 COVID-19 감염을 정확하게 식별하는 심층 학습 접근법의 효율성을 강조합니다.



### Underwater SONAR Image Classification and Analysis using LIME-based Explainable Artificial Intelligenc (https://arxiv.org/abs/2408.12837)
Comments:
          55 pages, 9 tables, 18 figures

- **What's New**: 이 연구는 SONAR 이미지를 분류하기 위한 Explainable AI(XAI) 기술의 적용에 대해 다루고 있으며, 이는 해당 분야의 최초 연구 중 하나입니다. 연구자는 다양한 공개 SONAR 데이터셋을 결합하여 새로운 맞춤형 데이터셋을 개발하고, Transfer Learning을 통해 여러 CNN 모델의 성능을 비교 분석했습니다.

- **Technical Details**: 이 논문에서는 VGG16, ResNet50, InceptionV3, DenseNet121 등의 Convolutional Neural Network(CNN) 아키텍처를 기반으로 이미지를 분류하는 Transfer Learning 기법의 활용을 조사합니다. 또한, Local Interpretable Model-Agnostic Explanations(LIME)와 Submodular Picks LIME(SP-LIME) 기술을 이용하여 모델의 결정을 해석하는 방법을 제시합니다. SP-LIME은 이미지에 특화된 LIME의 변형으로, quickshift와 SLIC 알고리즘을 사용하여 초픽셀을 변화시키는 방식으로 작동합니다.

- **Performance Highlights**: Transfer Learning을 활용한 이미지 분류의 정확도는 최대 97.33%에 달하며, XAI 기술을 통해 모델의 결정 과정을 시각적으로 설명하였습니다. 이 연구는 XAI가 SONAR 이미지 분석에 기여할 수 있는 잠재력을 보여주며, 해양 감시 및 탐사 작업의 효율성을 높이는 데 기여할 수 있습니다.



### CLLMFS: A Contrastive Learning enhanced Large Language Model Framework for Few-Shot Named Entity Recognition (https://arxiv.org/abs/2408.12834)
Comments:
          27TH EUROPEAN CONFERENCE ON ARTIFICIAL INTELLIGENCE

- **What's New**: 본 논문에서는 CLLMFS라는 Contrastive Learning이 강화된 대형 언어 모델(LLM) 프레임워크를 제안합니다. 이 프레임워크는 적은 양의 라벨 데이터로도 Few-Shot Named Entity Recognition (NER)에서의 성능을 향상시키기 위해 설계되었습니다.

- **Technical Details**: CLLMFS는 Low-Rank Adaptation (LoRA)와 대비 학습(Contrastive Learning) 메커니즘을 통합하여 LLM의 내부 표현을 활용합니다. 이 모델은 엔티티 경계 인식을 향상시키고, 엔티티 인식 정확성을 높이는 데 집중하고 있습니다.

- **Performance Highlights**: CLLMFS는 기존의 최우수 방법들에 비해 F1-score에서 2.58%에서 97.74%까지 성능 향상을 달성하며, 다양한 데이터셋에서 교차 도메인 NER 실험을 통해 우수한 일반화 성능을 입증하였습니다.



### Examining the Commitments and Difficulties Inherent in Multimodal Foundation Models for Street View Imagery (https://arxiv.org/abs/2408.12821)
- **What's New**: 이번 연구는 Minimal Modal Models (LLMs)인 ChatGPT-4V와 Gemini Pro의 실제 활용 가능성을 평가했습니다. 특히, 거리 이미지, 건축 환경, 실내 디자인 분석에서 이 모델들이 가진 강점과 약점을 찾아내었습니다.

- **Technical Details**: 본 논문은 거리 이미지에서 가구 인식, 보행자 및 차량 수 측정, 도로 폭 측정 임무를 수행하며, 건축 환경에서는 건물 기능 분류, 연령 및 높이 분석, 구조 분류를 수행했습니다. 실내의 경우 방 유형 분류와 디자인 스타일 분석을 포함한 다양한 작업을 평가했습니다. 연구 결과, 길이 측정, 스타일 분석 및 기본적인 이미지 이해 능력에서 우수한 성과를 거두었으나, 세부 인식 및 수 카운트 작업에서는 한계를 보였습니다.

- **Performance Highlights**: Zero-shot learning(제로샷 학습)에서 잠재력을 보였지만, 성능은 문제 도메인과 이미지 복잡도에 따라 달라졌습니다. multimodal foundation models (FMs)는 컴퓨터 비전과 언어의 융합에 잠재력을 열어 주었습니다.



### Staircase Cascaded Fusion of Lightweight Local Pattern Recognition and Long-Range Dependencies for Structural Crack Segmentation (https://arxiv.org/abs/2408.12815)
- **What's New**: 이 논문에서는 경량화된 계산 자원으로 고품질의 균열(segmentation maps)을 생성할 수 있는 계단형 연결 융합 네트워크(CrackSCF)를 제안합니다. 이 네트워크는 지역적 패턴과 장거리 픽셀 종속성을 잘 포착하는 새로운 모듈을 포함하고 있으며, 배경 잡음을 효과적으로 억제합니다.

- **Technical Details**: CrackSCF 네트워크는 계단형 연결 융합 모듈과 경량화된 합성곱 블록(LRDS)을 도입하여, 픽셀 수준에서 균열을 정확하게 분류합니다. 이 모델은 이미지에서 지역적 및 장거리 정보를 세밀하게 포착하고, 인접 레이어의 특성을 점진적으로 통합합니다. 또한, TUT라는 새로운 벤치마크 데이터셋을 구축하여 다양한 시나리오에서 실험을 수행했습니다.

- **Performance Highlights**: TUT 데이터셋에서 CrackSCF 네트워크는 F1 스코어 0.8382, mIoU 스코어 0.8473을 기록하며, 기존 방법들보다 우수한 성능을 보임과 동시에 가장 적은 계산 자원을 요구합니다. 실험은 다섯 개의 공개 데이터셋에서 진행되었으며, CrackSCF 네트워크가 기존 최첨단(SOTA) 방법들보다 더 나은 결과를 보여주었습니다.



### VALE: A Multimodal Visual and Language Explanation Framework for Image Classifiers using eXplainable AI and Language Models (https://arxiv.org/abs/2408.12808)
Comments:
          15 pages, 10 tables, 3 figures

- **What's New**: 이번 논문에서는 기존의 비유사한 방법들과 차별화된 다채로운 비주얼과 언어 설명 프레임워크인 VALE(Visual and Language Explanation)를 제안합니다. VALE는 이미지 분류 작업에서 DNN의 내재된 작동을 해석하고 설명하는 데 도움을 주며, 시각적 및 텍스트 기반의 설명을 통합하여 인간 친화적인 방식으로 결과를 전달합니다.

- **Technical Details**: VALE 시스템은 이미지 분류기, 설명기(SHAP), 이미지 분할기(Segment Anything Model), 그리고 이미지-텍스트 설명기(VLM)로 구성되어 있습니다. SHTA(clickable) 점수 계산을 통해 분류 이미지의 가장 영향력 있는 영역을 식별한 후, Segment Anything Model을 사용해 객체의 영역을 추출하고, VLM을 통해 시각적 설명에 대한 텍스트 설명을 생성합니다.

- **Performance Highlights**: VALE는 ImageNet 데이터셋 및 맞춤형 수중 SONAR 이미지 데이터셋에서 광범위한 실험을 통해 실용성을 입증했습니다. 이 프레임워크는 기존 XAI 도구를 통합하여 각 이미지 분류에서 보다 명확하고 이해하기 쉬운 결과를 제공합니다.



### Is Generative AI the Next Tactical Cyber Weapon For Threat Actors? Unforeseen Implications of AI Generated Cyber Attacks (https://arxiv.org/abs/2408.12806)
Comments:
          Journal Paper

- **What's New**: 이 논문은 인공지능(AI)과 사이버 보안(cybersecurity)의 교차점에서의 위협을 다룹니다. 특히, 대형 언어 모델(Large Language Models, LLM)의 악용 가능성을 심층 분석합니다.

- **Technical Details**: 사이버 범죄자들이 사이버 공격을 생성하고 자동화하는 데 사용할 수 있는 여러 기법들을 집중적으로 연구합니다. 여기에는 스위치 방법(switch method)과 캐릭터 플레이(character play) 방법이 포함됩니다. 또한, Occupy AI라는 커스터마이즈된 LLM을 소개하여 사이버 공격을 자동화하고 실행하는 데 특화된 AI 도구를 제시합니다.

- **Performance Highlights**: 이 연구는 AI의 사이버 공격 생성 능력을 실증적으로 평가하고, 사이버 공격의 효과성과 공략할 수 있는 취약점(vulnerabilities)을 분석했습니다. 이 결과는 AI와 관련된 위협을 완화하기 위한 윤리적 AI 관행과 강력한 사이버 보안 조치의 필요성을 강조합니다.



### Multi-Treatment Multi-Task Uplift Modeling for Enhancing User Growth (https://arxiv.org/abs/2408.12803)
- **What's New**: 이번 연구에서는 여러 치료와 여러 작업을 동시에 고려할 수 있는 Multi-Treatment Multi-Task (MTMT) uplift 모델을 제안합니다. 이 모델은 기존의 단일 작업 및 단일 치료 설정에서 벗어나, 사용자 반응의 복잡성을 고려하여 향상된 비즈니스 성과를 도모합니다.

- **Technical Details**: MTMT는 사용자 특성과 치료 정보를 별도로 인코딩하며, multi-gate mixture of experts (MMOE) 네트워크를 활용하여 관련 사용자 특성을 학습합니다. 본 연구에서는 베이스 효과와 특정 치료에 의한 증가 효과를 계산하는 방식으로 각 작업에 대한 치료 효과를 측정합니다. 또한, 치료-사용자 특성 상호작용 모듈을 통해 각 치료와 사용자 특성 간의 상관관계를 모델링합니다.

- **Performance Highlights**: MTMT는 공공 데이터셋과 대규모 제품 데이터셋을 통한 실험 결과에서 뛰어난 성능을 보이며, 기존 모델에 비해 우수한 결과를 나타냅니다. 특히, MTMT는 온라인 게임 플랫폼에 배포되어 대규모 사용자 경험을 향상시키는 데 기여하고 있습니다.



### Less for More: Enhancing Preference Learning in Generative Language Models with Automated Self-Curation of Training Corpora (https://arxiv.org/abs/2408.12799)
- **What's New**: 이 논문에서는 언어 모델의 성능 향상을 위해 선호 학습에서 발생하는 주석 불일치 문제를 해결하기 위해 자기 선별 방법을 제안합니다. 이 방법은 주석된 데이터셋을 자동으로 정리하여 모호한 주석을 제거하는 데 중점을 둡니다.

- **Technical Details**: 제안된 자기 선별 방법은 사전 훈련된 체크포인트로 초기화된 프록시 모델(proxymodel)을 사용하여 주석의 일관성을 평가합니다. 이러한 프록시 모델은 주어진 선호 데이터셋을 사용하여 순위 손실(ranking loss)을 통해 훈련됩니다. 이 과정에서 모호한 주석과 관련된 데이터 포인트를 자동으로 식별하고 배제합니다.

- **Performance Highlights**: 제조된 자기 선별 데이터셋을 사용한 훈련 결과, 다양한 지시사항 수행 작업에서 성능이 현저하게 개선되었습니다. 실험에서 제안된 방법이 선호 학습 성능을 크게 향상시킨다는 점을 입증하였습니다.



### Context-Aware Temporal Embedding of Objects in Video Data (https://arxiv.org/abs/2408.12789)
- **What's New**: 본 논문에서는 비디오 분석을 위한 새로운 모델을 제안하며, 이는 객체 간의 인접성과 의미적 유사성을 활용하여 컨텍스트 인식 임베딩을 생성합니다. 전통적인 접근 방식과는 달리, 이 모델은 시각적 외형만을 고려하지 않고 객체 간의 관계를 반영하여 임베딩 공간을 구축합니다.

- **Technical Details**: 제안된 모델은 영상의 각 프레임에서 검출된 객체들 간의 거리와 빈도를 활용하여 동적인 컨텍스트 인식 임베딩을 학습합니다. 또한, 이는 각 객체의 시간에 따른 변화를 추적하여 비디오의 내러티브를 구성하는 데 도움을 줍니다. 효율적인 동적 임베딩을 위해, CNN(Convolutional Neural Network) 모델을 사용하여 비디오 프레임을 전처리하고, 신경망 모델을 통해 임베딩 벡터를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 컨텍스트 인식 임베딩은 기존의 시각적 임베딩과 결합하여 다운 스트림 애플리케이션의 효과성을 향상시킬 수 있음을 보여주었습니다. 또한, 이 임베딩은 Large Language Model(LLM)을 사용하여 비디오 내러티브를 서술하는 데에도 활용될 수 있습니다.



### LLM-PBE: Assessing Data Privacy in Large Language Models (https://arxiv.org/abs/2408.12787)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)에서의 데이터 프라이버시 위험을 체계적으로 평가하는 도구인 LLM-PBE를 소개합니다. LLM-PBE는 다양한 공격 및 방어 전략을 통합하여 LLM의 전체 라이프사이클에서 데이터 프라이버시를 분석합니다.

- **Technical Details**: LLM-PBE는 사전 훈련된 데이터, 파인 튜닝된 데이터 및 사용자 정의 프롬프트를 포함하여 LLM의 모든 생애주기를 아우르는 데이터 유출 가능성을 평가합니다. 이 도구는 OpenAI, TogetherAI, HuggingFace와 같은 플랫폼에서 LLM에 접근할 수 있는 API를 제공하며, 여러 모델, 공격 방법론 및 방어 전략을 분석할 수 있는 통합된 접근 방식을 가지고 있습니다.

- **Performance Highlights**: 연구를 통해 모델 크기가 클수록 데이터 유출이 더 쉬워지는 경향이 있으며, 훈련 데이터의 성격이 프라이버시 위험에 결정적인 영향을 미친다는 중요한 통찰력을 발견했습니다. 최근의 LLM들은 예전 모델들에 비해 훈련 데이터 보호가 개선된 것으로 나타났습니다. 또한, 차별적 프라이버시(differential privacy) 기술을 활용한 데이터 보안 전략이 효과적일 수 있음을 확인하였습니다.



### The Model Mastery Lifecycle: A Framework for Designing Human-AI Interaction (https://arxiv.org/abs/2408.12781)
- **What's New**: AI의 Mastery Lifecycle(숙련 주기) 프레임워크를 소개하며, 인간과 AI 간의 협력적 작업 분담에 대한 새로운 시각을 제공합니다.

- **Technical Details**: AI의 역할을 인간 전문가와 AI의 자율성 간 연속선으로 설정하고, 다양한 형태의 인간-AI 상호작용을 통해 AI 응용의 맥락을 이해합니다.

- **Performance Highlights**: AI의 다양한 작업 수행 능력을 고려하여, 전문가의 성과를 초과하는 AI 모델을 효과적으로 통합할 수 있는 방법론을 제시합니다.



### Investigating LLM Applications in E-Commerc (https://arxiv.org/abs/2408.12779)
- **What's New**: 이 논문은 e-commerce 분야에서 Large Language Models (LLMs)의 효능을 조사하였으며, 여러 규모의 공개 e-commerce 데이터셋을 활용해 instruction-tuning한 LLM 모델의 성능을 전통적인 모델과 비교하였습니다.

- **Technical Details**: LLMs의 fine-tuning을 위한 다양한 방법론, 특히 Low-Rank Adaptation (LoRA), 단일 작업 훈련, 혼합 작업 훈련을 다루었으며, e-commerce 특정 작업에 대한 평가를 위한 데이터셋을 구성하였습니다. 또한, few-shot inference와 같은 방법의 효과를 분석하였습니다.

- **Performance Highlights**: 전통적인 산업 모델인 BERT 및 T5와의 비교실험 결과, 특정 e-commerce 작업에서 전통적인 방법으로 훈련한 소형 언어 모델이 매우 큰 LLM보다 더 나은 성능을 보일 수 있음을 발견했습니다.



### Data-Centric Approach to Constrained Machine Learning: A Case Study on Conway's Game of Lif (https://arxiv.org/abs/2408.12778)
- **What's New**: 이번 논문은 Conway의 생명 게임을 위한 데이터 중심(machine learning applications) 접근법을 제시하며, 최소한의 아키텍처 네트워크를 학습하여 주어진 단계 수에 대한 전환 규칙을 배우는 문제를 다룹니다. 이를 통해 제한된 학습 설정에서도 효율적인 데이터 설계의 중요성을 강조합니다.

- **Technical Details**: 이 연구에서는 이미지-투-이미지 전환(task of image-to-image translation) 문제로 생명 게임의 규칙을 학습하기 위해 CNN(Convolutional Neural Network)을 사용합니다. 제한된 아키텍처로 작업하여 훈련 데이터를 철저히 제어하는 방식으로 학습합니다. 훈련 데이터의 효율성을 기존 무작위 생성 보드 대신 정교하게 설계한 보드에서 비교합니다.

- **Performance Highlights**: 정교하게 설계된 단일 훈련 보드로 다단계 예측(multi-step prediction task)에서 수렴 속도와 정확도가 크게 향상되었음을 보여주며, 이는 데이터 중심(data-centric) 접근법이 제약된 머신러닝 응용 프로그램에서 유리하다는 것을 입증합니다.



### Environment-Centric Active Inferenc (https://arxiv.org/abs/2408.12777)
Comments:
          14 pages, 9 figures

- **What's New**: 본 논문에서는 환경의 예기치 못한 변화를 처리하기 위한 환경 중심의 능동적 추론(EC-AIF, environment-centric active inference) 방법을 제안합니다. 기존의 능동적 추론은 에이전트 중심으로 정의되어 있었으나, EC-AIF는 모든 관측 가능한 대상을 포함하여 환경을 정의함으로써 에이전트가 의도하지 않은 환경 변화에 대응할 수 있도록 합니다.

- **Technical Details**: EC-AIF 모델은 에이전트가 아닌 환경을 기반으로 Markov Blanket를 설계합니다. 이는 기존의 Markov Blanket 정의와는 달리, 에이전트가 아닌 환경을 중심으로 모든 상태를 추론 대상으로 포함시킵니다. 특히, 로봇 팔에 적용되어 객체 운반 작업을 수행하면서 목표 위치와 다른 로봇 팔의 방향 변경에 적응하며 성공적인 결과를 보여주었습니다.

- **Performance Highlights**: 로봇 팔이 객체의 목표 위치 변화와 다른 로봇 팔의 방향 변동에 성공적으로 반응하며 객체를 운반하는 성과를 달성했습니다. EC-AIF는 이러한 환경 변화에 유동적으로 대응할 수 있는 능력을 보여 주며, 이는 로봇의 지능적 행동 생성에 기여할 수 있습니다.



### Symmetric masking strategy enhances the performance of Masked Image Modeling (https://arxiv.org/abs/2408.12772)
Comments:
          Accepted at ICPR 2024

- **What's New**: 본 연구에서는 Masked Image Modeling (MIM) 방식의 새로운 마스킹 전략을 제안하여 모델이 글로벌(global) 및 로컬(local) 특징을 효과적으로 캡처할 수 있도록 합니다. 이를 기반으로 SymMIM이라는 새로운 학습 파이프라인을 도입했습니다.

- **Technical Details**: SymMIM은 대칭 마스킹 전략을 기반으로 하며, 수직 및 수평축을 따라 마스킹된 패치를 설정하여 각 마스킹 패치가 유사한 의미와 구조 정보를 포함하는 보이는 패치에 대응하게 합니다. 또한, 대조적 손실(contrastive loss)을 활용하여 전역 및 지역 특징 간의 일관성을 촉진합니다.

- **Performance Highlights**: SymMIM은 ViT-Large를 사용하여 ImageNet에서 85.9%의 새로운 SOTA 정확도를 달성하였으며, 이미지 분류, 객체 탐지 및 인스턴스 분할 등의 다양한 다운스트림 태스크에서 이전 SOTA를 초과하는 성능을 보였습니다.



### When In-memory Computing Meets Spiking Neural Networks -- A Perspective on Device-Circuit-System-and-Algorithm Co-design (https://arxiv.org/abs/2408.12767)
Comments:
          19 Pages, 13 Figures

- **What's New**: 이번 리뷰는 생명 유사 인공지능인 Spiking Neural Networks (SNNs)와 아날로그 In-Memory Computing (IMC) 영역의 교차점을 탐구하며, 저전력 엣지 컴퓨팅 환경을 위한 잠재력을 강조합니다.

- **Technical Details**: SNN은 Leaky-Integrate-and-Fire (LIF) 뉴런을 사용하여 스파이킹 정보를 처리합니다. IMC는 비휘발성 메모리를 이용해 아날로그 형태로 곱셈-누산(Multiply-and-Accumulate) 작업을 수행하며 데이터 전송 병목 현상을 줄입니다. SNN과 IMC는 이러한 구조 덕분에 저전력에서 높은 처리량을 제공합니다.

- **Performance Highlights**: SNN은 이미지 분류와 같은 대규모 작업에서 기존의 인공 신경망(ANN)과 유사한 정확도를 달성하였으며, 특히 90%의 높은 스파이크 희소성 덕분에 IMC에서의 구현 가능성이 큽니다. SNN 기반의 알고리즘은 저전력 엣지 구현을 위한 여러 효율성을 제공합니다.



### Assessing Modality Bias in Video Question Answering Benchmarks with Multimodal Large Language Models (https://arxiv.org/abs/2408.12763)
- **What's New**: 이번 연구에서는 여러 모달리티(visual, textual, auditory)를 동시에 처리할 수 있는 MLLM(멀티모달 대형 언어 모델)을 사용하여 비디오 질문 응답(VidQA) 데이터셋의 단일 모달리티 편향을 찾아내는 모달리티 중요도 점수(MIS)를 새롭게 도입하였습니다.

- **Technical Details**: 모달리티 중요도 점수(MIS)는 질문에 대한 정보를 포함하는 모달리티를 평가하기 위해 설계되었으며, 최첨단 MLLM을 활용하여 모달리티의 중요성을 추정하는 혁신적인 방법을 제안합니다. 이를 통해 기존 데이터셋에서 단일 모달리티 편향과 진정으로 멀티모달 질문의 부족함을 입증하였습니다.

- **Performance Highlights**: 여러 차별적 연구(ablation studies)를 통해 현재 MLLM들이 정보 통합에 효과적이지 않다는 결과를 도출하였으며, 이는 기존 데이터셋에서 모달리티 불균형 때문입니다. 연구에서 제안한 MLLM 기반의 MIS는 모달리티 균형 데이터셋의 구성에 기여하고, 멀티모달 학습의 발전과 MLLMs의 모달리티 간의 시너지 관계를 이해하고 활용하는 능력을 향상시키는 데 도움을 줄 수 있습니다.



### Visual Verity in AI-Generated Imagery: Computational Metrics and Human-Centric Analysis (https://arxiv.org/abs/2408.12762)
- **What's New**: AI 기술의 급격한 발전이 엔터테인먼트, 광고, 전자상거래 등 다양한 분야의 그래픽 콘텐츠 생산을 혁신하고 있습니다. 이러한 발전으로 AI 생성 이미지의 품질과 사실성을 평가하기 위한 강력한 평가 방법의 필요성이 대두되었습니다. 이를 해결하기 위해 세 가지 연구를 진행했습니다.

- **Technical Details**: 첫 번째 연구에서는 포토리얼리즘(photorealism), 이미지 품질(image quality), 텍스트-이미지 정합(text-image alignment)을 측정하는 설문지인 Visual Verity를 도입하고 검증했습니다. 두 번째로, 이 설문지를 활용하여 AI 모델(DALL-E2, DALL-E3, GLIDE, Stable Diffusion)과 카메라로 생성된 이미지의 평가를 실시했으며, 카메라 이미지가 포토리얼리즘과 텍스트-이미지 정합에서 우수한 성과를 보인 반면, AI 모델은 이미지 품질에서 앞섰습니다. 또한 통계적 특성을 분석하여, 카메라 이미지가 색조(hue), 채도(saturation), 밝기(brightness)에서 낮은 점수를 기록했음을 확인했습니다. 세 번째 연구에서는 계산 메트릭(computational metrics)과 인간 판단의 일치를 평가하여, MS-SSIM 및 CLIP이 인간의 평가와 가장 일치함을 밝혀냈고, 이미지 품질 평가를 위한 Neural Feature Similarity Score (NFSS)를 제안했습니다.

- **Performance Highlights**: 연구 결과는 인간의 시각적 인식을 더 잘 반영하기 위해 계산 메트릭을 개선할 필요성을 강조하고 있으며, 이는 AI 생성 콘텐츠 평가의 향상을 이끌 것으로 기대됩니다.



### SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection (https://arxiv.org/abs/2408.12748)
Comments:
          preprint under review

- **What's New**: 이번 연구에서는 Small Language Model (SLM) 분류기를 이용한 초기 탐지와 Large Language Model (LLM)을 이용한 제약된 추론자(constrained reasoner)를 통한 설명 생성이 결합된 새로운 프레임워크를 제안합니다. 이를 통해 실시간으로 해석 가능한 환각 탐지가 최적화되었습니다.

- **Technical Details**: 제안된 두 단계 프레임워크는 먼저 SLM을 사용하여 환각을 탐지하고, 탐지된 환각에 대한 설명을 LLM 기반의 제약된 추론자가 제공합니다. 이 과정에서 SLM과 LLM 간의 결정 및 설명의 일관성을 고려하여, 탐지 및 설명 간의 정렬을 개선하는 다양한 접근 방식을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 여러 오픈 소스 데이터세트에서 효과성이 입증되었으며, 사용자 경험을 향상시키는 데 기여했습니다. 특히, LLM의 설명 생성에 필요한 지연 시간 문제를 해결하는 데 효과적입니다.



### BankTweak: Adversarial Attack against Multi-Object Trackers by Manipulating Feature Banks (https://arxiv.org/abs/2408.12727)
- **What's New**: 이 논문에서는 Multi-object tracking (MOT) 시스템을 겨냥한 새로운 적대적 공격 기법인 	extsf{BankTweak}을 제안합니다. 	extsf{BankTweak}은 기존의 공격 방식의 비효율성과 취약성을 해결하고, 공격 후에도 지속적인 ID 전환을 유도하는 효율성과 견고성을 갖춘 방법입니다.

- **Technical Details**: 	extsf{BankTweak}은 Feature extractor에 집중하여, 아소시에이션 단계에서 헝가리안 매칭 알고리즘의 취약점을 이용합니다. 이 방법은 객체 위치를 수정하지 않고도 새로운 특징을 feature bank에 주입하여 지속적인 ID 전환을 유도합니다. 	extsf{BankTweak}은 데이터 세트인 MOT17과 MOT20에서 DeepSORT, StrongSORT, MOTDT의 세 가지 멀티 객체 추적기에 적용되었습니다.

- **Performance Highlights**: 실험 결과, 	extsf{BankTweak}은 기존 공격 방법들보다 상당히 우수한 성능을 보여주었으며, 추적-탐지 프레임워크의 취약성을 드러냈습니다. 	extsf{BankTweak}은 오탐지 없이 공격을 수행할 수 있어 실용성과 일반성을 보장합니다.



### Generating Realistic X-ray Scattering Images Using Stable Diffusion and Human-in-the-loop Annotations (https://arxiv.org/abs/2408.12720)
- **What's New**: 본 연구에서는 X선 산란 이미지와 그에 대한 설명을 활용하여 재정비된 stable diffusion 모델을 통해 새로운 과학적 이미지를 생성하는 방법을 제안합니다. 생성된 이미지 중 일부는 "hallucinations"라고 불리는 비현실적인 아티팩트를 포함하고 있으며, 이를 해결하기 위해 선별된 이미지 데이터셋을 기반으로 다양한 computer vision 모델을 훈련시켰습니다.

- **Technical Details**: 프레임워크는 다음 단계로 구성됩니다: (a) X선 산란 이미지와 텍스트 설명으로 foundational diffusion 모델을 재정비; (b) 생성된 이미지를 전문가가 'realistic' 또는 'fake'로 라벨링; (c) ResNet-50 모델을 사용하여 인간의 라벨을 기반으로 unseen 생성 이미지를 분류; (d) 이 절차를 반복하여 라벨링된 이미지 수를 증가시키고, 다양한 foundation 모델을 활용하여 realistic X선 산란 이미지를 최대한 많이 탐지합니다.

- **Performance Highlights**: 실험 결과, 수정된 diffusion 모델을 통해 고충실도(high-fidelity) 및 도메인 특화된 이미지 생성의 가능성을 확인하였습니다. 향후 generative AI가 데이터 증대(data augmentation) 및 디지털 트윈(digital twin) 개발에 중요한 역할을 할 것으로 예상됩니다.



### MultiMed: Massively Multimodal and Multitask Medical Understanding (https://arxiv.org/abs/2408.12682)
- **What's New**: 새로운 벤치마크인 MultiMed는 256만 개의 샘플을 기반으로 하여 다양한 의료 모달리티와 과제를 아우르는 대규모 학습을 평가하고 지원하는 데 중점을 둡니다. 각 모달리티는 의료 보고서, 병리학, 유전체학, 단백질 데이터 등 포함됩니다.

- **Technical Details**: MultiMed는 10가지 의료 모달리티(예: 의료 보고서, 병리학, 유전체, 단백질 데이터)를 포함하며, 11가지 도전적인 과제로 구성되어 있습니다. 이 과제는 질병 예후, 단백질 구조 예측, 의료 질문 응답 등을 포함합니다. MultiMed는 여러 관련 모달리티 및 과제에서의 대규모 학습을 지원합니다.

- **Performance Highlights**: MultiMed를 통해 수행된 실험은 단일 모달리티, 다중 모달리티, 다중 과제 모델들의 성능을 벤치마킹하며, 관련 모달리티 및 과제를 아우르는 대규모 의료 모델 교육의 장점을 강조합니다. 이는 진단 및 예후 기능 향상에 기여할 수 있습니다.



### Leveraging Information Consistency in Frequency and Spatial Domain for Adversarial Attacks (https://arxiv.org/abs/2408.12670)
Comments:
          Accepted by PRICAI 2024

- **What's New**: 이번 논문에서는 주파수 도메인(frequency domain) 기반 공격의 효과를 분석하고, 공간 도메인(spatial domain)과의 일관성을 활용한 새로운 적대적 공격 알고리즘(Frequency and Spatial consistency based adversarial Attack, FSA)을 제안합니다. 이는 기존의 적대적 공격 기법들과 비교하여 성능을 크게 향상시키는 데 기여합니다.

- **Technical Details**: FSA 알고리즘은 주파수와 공간 도메인에서의 정보 일관성을 결합하여 적대적 예제를 생성합니다. 기존의 적대적 공격 방식들은 주로 공간 정보만을 사용했으나, 본 연구에서는 주파수 정보도 동시에 고려함으로써 더욱 자연스러운 변화를 유도하고, 공격 성공 확률을 높이는 방법론을 제시합니다.

- **Performance Highlights**: 상세 실험 결과, FSA는 기존의 기법들에 비해 주목할 만한 성능 향상을 보였으며, 다양한 모델에 대해 최첨단 성능을 달성했습니다. 이 연구는 주파수 도메인과 공간 도메인 간의 일관성을 활용하여 적대적 공격의 새로운 가능성을 제시하고 있습니다.



### Bayesian Network Modeling of Causal Influence within Cognitive Domains and Clinical Dementia Severity Ratings for Western and Indian Cohorts (https://arxiv.org/abs/2408.12669)
Comments:
          7 pages, 2 figures, 3 tables

- **What's New**: 이번 연구는 Alzheimer's Disease Neuroimaging Initiative (ADNI)와 Longitudinal Aging Study of India (LASI)라는 두 가지 노화 데이터 세트를 사용하여 Clinical Dementia Ratings (CDR)와 여섯 개의 도메인 점수 간의 인과 관계를 조사합니다. 우리는 Bayesian network 모델에서 파생된 Directed Acyclic Graphs (DAGs)를 사용하여 도메인 점수 간의 의존성과 global CDR에 대한 영향을 분석하였습니다.

- **Technical Details**: PC 알고리즘을 사용하여 두 데이터 세트의 DAG 구조를 추정하였으며, 이는 서구 인구와 인도 인구 간의 인과 관계 및 엣지 강도의 notable differences를 드러내었습니다. CDR 점수는 두 데이터 세트 모두에서 메모리 기능에 강하게 의존하고 있으나, 엣지 강도와 노드 정도가 상당히 다르게 나타났습니다.

- **Performance Highlights**: 이 연구는 인구별 차이와 유사성을 밝힘으로써 치매 진행을 이해할 수 있는 통찰력을 제공하며, 맞춤형 개입을 위한 정보를 제시하고 있습니다. 또한, DAGs를 통해 복잡한 점수 집합을 단순화하고 CDR의 구체적인 점수를 이해하는 데 도움을 줄 수 있음을 보여줍니다.



### Benchmarking Counterfactual Interpretability in Deep Learning Models for Time Series Classification (https://arxiv.org/abs/2408.12666)
Comments:
          15 pages, 27 figures

- **What's New**: 본 논문은 시간 시계열 도메인에서 Counterfactual (CF) 방법에 대한 시스템적 평가를 최초로 제안합니다. 기존의 논문들은 CF 방법에 대한 평가가 제대로 이루어지지 않았으며, 다양한 메트릭과 데이터 세트를 사용하여 성능을 비교한 적이 없습니다.

- **Technical Details**: 논문에서는 CF 방법의 평가를 위해 스파스(sparsity), 개연성(plausibility), 일관성(consistency) 등을 포함한 여러 새로운 메트릭을 재설계하였습니다. 20개의 단일 변수 데이터 세트와 10개의 다변량 데이터 세트를 사용하여 6가지 CF 방법의 성능을 비교 분석하였습니다. 또한 유효성(validity), 생성 시간(generation time), 근접성(proximity)과 함께 포괄적인 메트릭 세트를 구성하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, CF 방법의 성능은 메트릭 및 모델에 따라 다양하게 나타났습니다. 결론적으로, 어떤 CF 방법도 모든 메트릭에서 다른 방법들을 초월하지 못하며, 분류기(classifier)가 CF 성능에 미치는 영향이 강조되었습니다.



### Fairness-Aware Streaming Feature Selection with Causal Graphs (https://arxiv.org/abs/2408.12665)
Comments:
          This paper has been accepted by the 2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2024)

- **What's New**: 이 논문은 Streaming Feature Selection with Causal Fairness (SFCF)라는 새로운 알고리즘을 제안하여, 알고리즘 성능의 정확성(accuracy)과 공정성(fairness) 간의 최적화를 실시간 데이터 환경에서 해결하려는 첫 번째 시도입니다.

- **Technical Details**: SFCF 알고리즘은 두 개의 인과 그래프(causal graphs)를 구축하여, 예측 레이블과 보호된 특징(protected feature) 간의 복잡한 상관관계를 모델링합니다. 이 알고리즘은 d-separation을 활용하여 조건부 독립성에 따라 특징을 평가하고, 중요한 혹은 중복되지 않은 특징을 선별합니다.

- **Performance Highlights**: SFCF는 다섯 개의 벤치마크 데이터셋에서 여섯 개의 기존 모델보다 평균적으로 52% 더 높은 평형 확률(equalized odds), 98% 더 높은 희소성(sparsity), 99% 더 빠른 러닝타임(runtime)을 기록하며, 정확성(accuracy)은 거의 변하지 않았습니다.



### Disentangled Structural and Featural Representation for Task-Agnostic Graph Valuation (https://arxiv.org/abs/2408.12659)
- **What's New**: 이번 논문에서는 데이터 가치 평가에 대한 새로운 접근 방식을 제시합니다. 특히 그래프 데이터셋에 중점을 두어, 특정 작업 관련 메트릭에 의존하지 않고 데이터를 평가하는 방법론을 도입하였습니다.

- **Technical Details**: 우리는 그래프를 구조적 특성과 특징적 특성의 두 가지 주요 구성 요소로 나누고, 셀러의 그래프와 바이어의 그래프를 공유된 노드 순열에 기반하여 정렬하는 새로운 프레임워크인 blind message passing을 소개합니다. 이를 통해 graph Wasserstein distance를 활용하여 그래프 데이터셋의 구조적 분포 차이를 정량화하고, 판매 데이터의 관련성과 다양성을 평가합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험 결과, 우리 접근 방식이 판매자의 데이터에 대한 관련성, 다양성 및 구조적 차이를 효과적으로 포착함을 보여주었습니다. 특히 그래프 기반 데이터 가치 평가 시나리오에서 효과적인 성능을 보였습니다.



### Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music (https://arxiv.org/abs/2408.12658)
Comments:
          Accepted at International Society for Music Information Retrieval (ISMIR) 2024

- **What's New**: Hindustani 음악의 발성을 생성할 수 있는 최초의 모델 GaMaDHaNi를 제안하며, 이는 음악의 풍부한 멜로디적 복잡성을 유지합니다.

- **Technical Details**: 이 모델은 피치(pitch)와 스펙트로그램(spectrogram)을 포함하는 두 수준의 계층적(hierarchical) 데이터 표현을 사용합니다. 피치 생성기와 스펙트로그램 생성기를 훈련하여 각각을 생성하고, 생성된 스펙트로그램은 보코더(vocoder)를 사용하여 오디오로 변환됩니다. 피치는 두 가지 방식으로 모델링됩니다: 자기 회귀(transformer)와 확산 모델(difussion model)을 사용하여 이산 토큰과 연속 값으로.

- **Performance Highlights**: 청취 테스트를 통한 질적 분석 결과, 우리의 계층적 접근 방식이 기존 모델보다 더 우수한 성능을 보여줍니다. 피치의 중간 표현을 사용함으로써 음악가와의 인간-AI 협업 설정에서 더 나은 반응을 이끌어낼 수 있습니다.



### AI-driven Transformer Model for Fault Prediction in Non-Linear Dynamic Automotive System (https://arxiv.org/abs/2408.12638)
- **What's New**: 본 논문은 자동화된 시스템에서의 고급 통계 방법 및 알고리즘을 통해 자동차 엔진 시스템의 결함 감지 및 예측을 AI 기반으로 개선하려는 혁신적인 접근 방식을 제시합니다. 특히, Transformer 구조를 기반으로 한 결함 분류 및 예측 모델을 도입하여 복잡한 비선형 동적 시스템에서의 적용 가능성을 강조합니다.

- **Technical Details**: 제안된 Transformer 모델은 27개의 입력 차원, 2개의 층으로 구성된 64개의 숨겨진 차원, 9개의 헤드를 사용하여 12개의 출력 헤드를 생성합니다. 이 모델은 UTSA Arc HPC 클러스터에서 5개의 NVIDIA V100 GPU와 40코어 CPU 및 384GB RAM을 이용하여 교육되었으며, 70.01%의 정확성을 달성했습니다.

- **Performance Highlights**: 이 모델은 30분 주기의 시뮬레이션에서 엔진 성능을 모니터링 하였으며, 센서 데이터에 기반하여 결함이 발생하기 전 예측할 수 있는 능력을 갖추고 있습니다. 이 실험은 다양한 결함 유형에 대한 1,000회의 시뮬레이션 데이터를 기반으로 진행되었으며, 결과적으로 기존 모델보다 더 뛰어난 결함 분류 및 예측 성능을 보여주고 있습니다.



### Building and better understanding vision-language models: insights and future directions (https://arxiv.org/abs/2408.12637)
- **What's New**: 최근 Vision-Language Models (VLMs)의 발전에 대한 포괄적인 개요를 제공하며, 기존 모델들의 강점과 약점을 분석하고 새로운 연구 방향을 제시합니다. 특히, Idefics3-8B라는 강력한 VLM 모델의 구축 방법과 그 데이터셋인 Docmatix를 소개합니다.

- **Technical Details**: 이 논문에서는 데이터, 아키텍처(architecture), 학습 방법(training methods) 등 VLM 개발 파이프라인의 핵심 요소들에 대한 다양한 설계 선택들을 분석합니다. 특히, Idefics3-8B 모델의 구축에 필요한 단계로는 새로운 Docmatix 데이터셋을 생성하는 과정이 포함되며, 이는 이전 데이터셋보다 240배 큰 규모입니다.

- **Performance Highlights**: Idefics3-8B 모델은 Idefics2-8B에 비해 문서 이해 작업에서 13.7 포인트의 성능 향상을 이루었습니다. 새로운 모델은 오픈 데이터셋만을 사용하여 효율적으로 학습되었습니다.



### Joint Hypergraph Rewiring and Memory-Augmented Forecasting Techniques in Digital Twin Technology (https://arxiv.org/abs/2408.12634)
Comments:
          Paper accepted at AI for Digital Twins and Cyber-Physical Applications Workshop, International Joint Conferences on Artificial Intelligence(IJCAI-23). arXiv admin note: text overlap with arXiv:2408.12409

- **What's New**: 이 논문에서는 Digital Twin 기술을 활용하여 복잡한 동적 시스템의 예측을 위한 혁신적인 하이브리드 아키텍처를 소개합니다. 이 아키텍처는 하이퍼그래프(hypergraph) 표현 학습(backbone)을 강화하고 빠른 패턴 적응 및 과거 지식의 메모리 기반 검색 기능을 통합하여 기존의 그래프 forecasting 기술의 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 Joint Hypergraph Rewiring and Forecasting Neural Framework(JHgRF-Net)는 두 가지 목표를 균형 있게 달성합니다: (i) 현재의 트렌드와 패턴을 신속하게 학습하기 위한 이전 지식 활용, (ii) 이전에 습득한 지식을 유지 및 업데이트. 이 프레임워크는 Spatio-Temporal Hypergraph Convolutional Network(STHgCN)와 Spatio-Temporal Transformer Network(STTN) 두 가지 보완적 구성 요소 간의 상호 작용을 활용하여 동적 균형을 이룹니다.

- **Performance Highlights**: JHgRF-Net은 ablation 연구를 통해 검증되었으며, 여러 벤치마크 데이터셋에서 기존의 최첨단 forecasting 방법들을 상당히 초월하는 promising results를 보여주었습니다. 이 기술은 예측의 불확실성을 추정하고, 보다 향상된 성능으로 최근의 변화에 적응하는데 기여합니다.



### Generative Diffusion Model-based Downscaling of Observed Sea Surface Height over Kuroshio Extension since 2000 (https://arxiv.org/abs/2408.12632)
Comments:
          28 pages, 7 figures, and 1 table

- **What's New**: 본 연구에서는 최첨단 생성적 확산 모델(generative diffusion model)을 사용하여 고해상도 해수면 높이(SSH) 재분석 데이터의 훈련을 소개하며, 쿠로시오 연장(Kuroshio Extension) 지역에서의 관측 SSH 다운스케일링에 대한 우수성을 입증합니다.

- **Technical Details**: 이 확산 기반 모델은 원시 위성 보간(interpolated) 데이터를 0.25도 해상도에서 약 12km 파장에 해당하는 1/16도 해상도로 효과적으로 다운스케일링합니다. 이 모델은 기존의 고해상도 재분석 데이터셋 및 신경망 기반(neural network-based) 방법보다 뛰어난 성능을 보입니다.

- **Performance Highlights**: 이 연구 결과는 쿠로시오 연장 지역에서 수평 스케일이 250km 미만인 에디 운동 에너지(eddy kinetic energy)가 2004년 이후로 현저히 증가했음을 나타내며, 이는 심층 학습(deep learning)이 위성 고도계(satellite altimetry)를 재구성하고 에디 스케일에서의 해양 동역학(ocean dynamics)에 대한 이해를 향상시키는 데 큰 잠재력을 지니고 있음을 강조합니다.



### Data-Free Class Incremental Gesture Recognition via Synthetic Feature Sampling (https://arxiv.org/abs/2408.12629)
- **What's New**: 본 연구는 Data-Free Class Incremental Learning (DFCIL) 방법론을 토대로 스켈레톤 기반 제스처 인식 분야에 대한 새로운 접근 방식을 제시합니다. 기존 연구에서는 주로 이미지 데이터셋을 중심으로 DFCIL을 탐구하였으나, 본 연구에서는 VR/AR 환경에서의 제스처 인식의 필요성에 주목하여 관련 연구를 진행했습니다.

- **Technical Details**: 우리는 Synthetic Feature Replay (SFR) 알고리즘을 제안하여 클래스 프로토타입으로부터 합성 feature를 샘플링하고 이를 사용해 이전 클래스의 replay와 새로운 클래스의 augmentation을 수행합니다. 기존 DFCIL 방법들이 어려운 합성 데이터 생성을 필요로 하는 반면, SFR은 신속하고 단순하며 모든 테스트된 데이터셋에서 현상 유지 기반 최첨단 방법들에 비해 상당한 성능 향상을 나타냅니다.

- **Performance Highlights**: 우리의 방법론은 평균 정확도를 최대 15% 향상시키며, 기존의 기본 클래스와 새로운 클래스 간의 정확도 불균형 문제를 효과적으로 완화합니다.



### Educational Customization by Homogenous Grouping of e-Learners based on their Learning Styles (https://arxiv.org/abs/2408.12619)
- **What's New**: 새로운 e-learning 환경을 통해 개인 맞춤형 교육 콘텐츠를 제공할 수 있으며, 비슷한 학습자를 그룹으로 묶어 학습 비용을 절감할 수 있는 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 Felder-Silverman 모델을 이용하여 유사 학습자를 그룹화하고, Fuzzy Set Theory (FST)를 사용하여 e-learner의 행동과 작업을 모델링합니다. 학습 스타일을 파악한 후, 유사 그룹에 따라 적응형 콘텐츠를 제공합니다.

- **Performance Highlights**: 실험 그룹의 '교육 성공' 가중 평균 점수는 20점 만점 중 17.65점이며, 대조군은 12.6점을 기록했습니다. '교육 만족도'는 실험 그룹이 67%, 대조군은 37%로 나타났습니다.



### Semantic Communication based on Large Language Model for Underwater Image Transmission (https://arxiv.org/abs/2408.12616)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 기반으로 하는 새로운 의미적 통신(Semantic Communication, SC) 프레임워크를 제안합니다. 이는 수중 이미지 데이터를 사용자 쿼리에 따라 의미적으로 압축하고 우선순위를 정하는데 도움을 줍니다.

- **Technical Details**: SC 프레임워크는 시각적 LLM을 활용하여 이미지를 분석하고 중요 의미 요소를 식별하여 전송합니다. 수신 측에서는 LLM 기반 복구 메커니즘과 Global Vision ControlNet 및 Key Region ControlNet 네트워크를 활용하여 이미지를 재구성하여 통신 효율성을 높입니다.

- **Performance Highlights**: 이 프레임워크는 전체 데이터 크기를 원본의 0.8%로 줄이며, 기존 방법보다 훨씬 높은 품질의 의미적 이미지 복원을 보장합니다. 실험 결과에 따르면, 제안된 방법이 높은 통신 효율성과 강건성을 달성한다는 것이 입증되었습니다.



### Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning (https://arxiv.org/abs/2408.12614)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 논문에서는 이미지 수준의 약한-강한 일관성(weak-to-strong consistency)에서 탈피하여 이미지-특징(feature) 수준의 약한-강한 일관성(Image-Feature Weak-to-Strong Consistency, IFMatch) 패러다임을 도입합니다. 이를 통해 다양한 강도와 형태의 특징 수준의 왜곡(feature-level perturbation)을 적용하여 데이터 증강(augmentation) 공간을 확장하고, 기존의 반지도 학습(semi-supervised learning, SSL) 방법들과 원활하게 통합될 수 있도록 합니다.

- **Technical Details**: IFMatch는 세 가지 가지(branch)를 포함하는 구조로, 강한 이미지 수준의 왜곡과 특징 수준의 왜곡 간의 상호작용을 촉진하여 이들의 시너지를 높입니다. 여기서 특징 수준의 왜곡은 중간 특징을 무작위로 변경하는 방식으로 진행되며, 약한 이미지 수준과 강한 특징 수준의 왜곡을 결합하여 효율적인 특징 수준의 일관성 조정을 수행합니다. 또한, 신뢰 기반의 식별 전략(confidence-based identification strategy)을 제안하여 순진한 샘플(naive samples)과 도전적인 샘플(challenging samples)을 구별할 수 있도록 하였습니다.

- **Performance Highlights**: 제안된 IFMatch 패러다임은 여러 대표적인 반지도 학습 알고리즘에 적용되었으며, 균형 잡힌 샘플과 불균형 샘플 모두에 대해 실험을 진행하였습니다. 그 결과, 기존의 SSL 알고리즘의 성능이 유의미하게 향상된 것으로 나타났습니다.



### Deceptive uses of Artificial Intelligence in elections strengthen support for AI ban (https://arxiv.org/abs/2408.12613)
- **What's New**: 이 논문은 인공지능(AI)이 선거에 미치는 영향을 평가하기 위한 프레임워크를 제안하며, AI의 다양한 캠페인 작업에서의 적용을 고려합니다.

- **Technical Details**: AI를 활용한 캠페인 사용을 세 가지 범주로 나눕니다: 캠페인 운영(campaign operations), 유권자 소통(voter outreach), 그리고 기만(deception). 저자들은 7,635명의 미국인을 대상으로 하는 사전 등록된 대표 설문조사와 두 개의 실험을 통해 AI가 선거에서 어떻게 인식되고 있는지에 대한 체계적인 증거를 제공합니다.

- **Performance Highlights**: 1) 대중은 선거에서 사용되는 AI의 다양한 사용을 구분하며, AI 사용이 대체로 부정적이라고 인식하지만 기만적 사용(deceptive uses)에 가장 강력하게 반대합니다. 2) 기만적 AI 관행은 관련된 태도에 부정적인 영향을 미치고, AI 개발 중지에 대한 대중의 지지를 강화할 수 있습니다. 3) 그러나 기만적 선거 사용이 매우 싫어지지만 관련 정당에 대해 실질적인 선호도 하락(favorability penalties)의 결과를 초래하지는 않습니다.



### Enhanced Prediction of Multi-Agent Trajectories via Control Inference and State-Space Dynamics (https://arxiv.org/abs/2408.12609)
- **What's New**: 이 논문은 자율 시스템 분야에서 차량과 보행자의 경로를 예측하기 위한 새로운 방법론을 소개합니다. 새로운 'Mixed Mamba' 모델을 활용하여 제어 변수를 모델링하고, Graph Neural Networks(GNNs)와 상태-공간 모델(state-space models)을 결합하여 다중 에이전트 상호작용의 복잡성을 포착합니다.

- **Technical Details**: 제안된 방법론은 상태-공간 동적 시스템 모델링을 바탕으로 하며, 여기서 GNN을 사용하여 에이전트 간의 공간적 관계를 포착하고, Mamba를 통해 시계열 순서를 고려하여 제어 변수를 예측합니다. 이 시스템은 예측의 물리적 의미를 전달하며, 시간 복잡성을 선형으로 줄이고 메모리 사용을 최소화합니다.

- **Performance Highlights**: 제안된 알고리즘은 세 가지 공개 데이터셋에서 여러 최첨단 방법들과 비교하여 우수한 성능을 보입니다. 이를 통해 자율 시스템의 경로 예측 능력 향상에 기여할 가능성이 큽니다.



### A frugal Spiking Neural Network for unsupervised classification of continuous multivariate temporal data (https://arxiv.org/abs/2408.12608)
- **What's New**: 이번 연구에서는 복잡한 신경 데이터의 실시간 처리를 위한 효과적인 알고리즘을 개발한 점이 주목할 만합니다. 특히, 완전한 비지도 학습 (unsupervised learning) 방식으로 다차원 시간 패턴을 식별하고 분류할 수 있는 단일 레이어 스파이킹 신경망 (Spiking Neural Networks, SNNs)을 도입했습니다.

- **Technical Details**: 제안된 방법론은 몇 가지 생물학적으로 영감을 받은 가소성 규칙 (plasticity rules)을 기반으로 합니다. 여기에는 Spike-timing-dependent plasticity (STDP), Short-term plasticity (STP), 그리고 intrinsic plasticity (IP)이 포함되어 있으며, 이러한 규칙을 통해 멜 주파수 켑스트럴 (Mel Cepstral) 표현 및 다채널 신경 데이터에서 복잡한 패턴을 효과적으로 인식할 수 있습니다.

- **Performance Highlights**: 제안된 SNN 구조는 적은 수의 뉴런을 사용하여 고도로 겹치는 다차원 시간 패턴을 인식하는 데 효율적임을 입증했습니다. 시뮬레이션 데이터뿐만 아니라 실제 음성 오디오 및 다채널 신경 데이터에서도 뛰어난 성능을 보였습니다. 이는 향후 저전력 하드웨어에 통합 가능한 강력한 비지도 학습 기술의 전환점을 제공합니다.



### Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Mod (https://arxiv.org/abs/2408.12606)
Comments:
          27 pages, 8 figures, 10 tables

- **What's New**: 이번 연구에서는 다중 파라미터 유방 자기공명영상(MRI) 데이터를 통합하여 비침습적이고 개인화된 유방암 관리 방법을 제공하는 대규모 혼합 모달 전문가 모델(MOME)을 개발했습니다. 이를 위해 중국의 세 개 병원에서 수집한 5,205명의 환자를 대상으로 가장 큰 규모의 데이터셋을 구축하였습니다.

- **Technical Details**: MOME는 데이터의 이질성과 고차원성을 다루기 위해 다양한 유형의 훈련 가능한 전문가 모듈을 적용한 Transformer 구조로 설계되었습니다. 이 모델은 병리학적 완전 반응 예측을 포함하여 유방암 환자의 악성 종양 식별에서 방사선 전문의의 성과와 유사한 정확도를 달성하였습니다.

- **Performance Highlights**: MOME는 내부 테스트 세트에서 0.913 AUROC, 0.948 AUPRC, 0.905 F1 점수를 기록하며, BI-RADS 4 환자의 바이옵시 필요성을 7.3% 줄이는 가능성을 보였습니다. 모델은 강화된 정확도로 악성 및 양성 종양을 구분할 수 있습니다.



### Convolutional Neural Networks for Predictive Modeling of Lung Diseas (https://arxiv.org/abs/2408.12605)
Comments:
          7 pages

- **What's New**: 이번 논문에서는 폐 이미징을 위한 질병 예측을 위해 HRNet와 void-convolution 기법을 결합한 혁신적인 모델 Pro-HRnet-CNN을 제안합니다.

- **Technical Details**: Pro-HRnet-CNN 모델은 권위 있는 LIDC-IDRI 데이터셋을 활용하여 실험 비교가 이루어졌습니다. 전통적인 ResNet-50과 비교했을 때, Pro-HRnet-CNN은 작은 크기의 결절(nodule) 특성 추출(feature extraction) 및 인식(recognition)에서 더 나은 성능을 보였습니다.

- **Performance Highlights**: 특히 작은 목표(target) 탐지에서 모델은 정확도 향상에서 뛰어난 성능을 보여주어 폐 질환의 조기 발견 및 예측(prognostication)을 위한 혁신적인 길을 열었습니다.



### Generational Computation Reduction in Informal Counterexample-Driven Genetic Programming (https://arxiv.org/abs/2408.12604)
- **What's New**: 본 논문에서는 사용자 제공 훈련 데이터만을 사용하여 프로그램 합성 문제에 적용할 수 있는 '비공식 CDGP(informal CDGP)'를 제안합니다. 이 방법은 전통적인 CDGP의 아이디어를 확장하여 공식적인 사양 없이도 작동하는 방법을 보여줍니다.

- **Technical Details**: 비공식 CDGP(iCDGP)는 사용자 제공 훈련 사례를 활용하여 진화 과정에서 개체를 평가합니다. 이 과정에서 '카운터예제(counterexample)'를 현재 개체군에서 가장 우수한 개체의 경우에 대해 선택하여 훈련 세트에 추가하게 되며, 이는 진화를 더 집중적으로 유도합니다. iCDGP는 진화 과정을 통해 더 많은 개체를 평가할 수 있도록 하며, 새로운 변형을 통해 성능이 향상됩니다.

- **Performance Highlights**: iCDGP는 일반적인 프로그램 합성 벤치마크 문제를 테스트한 결과, 표준 GP보다 더 빠르게 솔루션을 찾을 수 있음을 보여주었습니다. 두 개의 새로운 변형 중 하나는 테스트된 문제 중 약 절반에서 훨씬 더 성공적인 결과를 내었으며, 카운터 예제를 훈련 세트에 추가하는 것이 성능을 크게 향상시킴을 발견했습니다.



### Sleeper Social Bots: a new generation of AI disinformation bots are already a political threa (https://arxiv.org/abs/2408.12603)
- **What's New**: 이 논문은 정치적 환경에서의 "sleeper social bots"의 증가하는 위협을 다룹니다. 이러한 소셜 봇은 정보 왜곡과 여론 조작을 위해 설계된 AI 기반 봇입니다.

- **Technical Details**: 연구팀은 개인 Mastodon 서버를 사용하여 ChatGPT 기반의 봇들이 인간 참가자와 가상의 선거 제안에 대해 토론할 수 있도록 프로그래밍하였습니다. 이 봇들은 독특한 성격과 정치적 관점을 가지고 있으며, 인간 사용자로 효과적으로 위장할 수 있습니다.

- **Performance Highlights**: 초기 실험에 참여한 대학생들은 이 봇들을 식별하지 못하였으며, 이는 AI 기반 정보 왜곡의 위험에 대한 인식 및 교육의 필요성을 강조합니다. 이러한 봇들은 대화의 맥락에 따라 주장을 조정하며, 이는 그들의 동적이고 설득력 있는 능력을 보여줍니다.



### Fiber neural networks for the intelligent optical fiber communications (https://arxiv.org/abs/2408.12602)
Comments:
          5 pages, 4 figures

- **What's New**: 이번 논문에서는 새로운 광섬유 신경망(fiber neural networks)과 그 관련된 광 신호 처리(optical signal processing) 방법을 개발하여 5G 이상의 지능형 통신(signal processing)에 대한 요구를 충족시키고자 합니다.

- **Technical Details**: 광섬유의 빛 전송(light transmission) 메커니즘을 활용하여 계산(computing)을 수행하는 광신경망(optical neural networks)을 기반으로 하며, 신호는 전자(domain)로 변환되지 않고 직접 광(domain)에서 처리됩니다. 이는 처리 효율성과 전력 비용에서 큰 이점을 제공합니다.

- **Performance Highlights**: 변조 형식 인식(modulation format recognition) 작업을 통해 전체 구조와 관련된 방법의 신뢰성(fidelity)이 입증되었습니다. 이는 광섬유 통신에서 중요한 역할을 하며, 광 신호 처리의 효율성을 높이는 데에 기여합니다.



### Enhanced Fine-Tuning of Lightweight Domain-Specific Q&A Model Based on Large Language Models (https://arxiv.org/abs/2408.12247)
- **What's New**: 본 논문은 Self-Evolution이라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLMs)의 전문적인 도메인 지식을 활용하는 데 있어 성능을 향상시키고, 자원 절약과 프라이버시 보호 문제를 해결하고자 합니다.

- **Technical Details**: Self-Evolution은 여러 단계의 반복적인 fine-tuning을 통해 가벼운 오픈 소스 LLM을 활용하여 지식 필터링 및 강화를 수행하는 전략을 채택했습니다. 이 프레임워크는 자원 요구량을 줄이고, 7B 파라미터의 모델을 선택하여 효율적인 QA 작업을 지원합니다.

- **Performance Highlights**: Self-Evolution은 중국 이동통신사의 4,000개의 도메인 관련 문서를 기반으로 Qwen1.5-7B-Chat 모델을 활용하여 도메인 특정 QA 평가에서 174% 높은 성능 점수를 기록했습니다. 또한, 117일 동안 실제 운영에 배포되었으며, 알람 찾기, 문제 해결 및 보고서 검색의 효율성을 평균 18.6% 향상시켰습니다.




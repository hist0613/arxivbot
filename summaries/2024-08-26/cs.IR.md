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




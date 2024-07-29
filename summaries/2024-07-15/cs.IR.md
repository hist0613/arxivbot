New uploads on arXiv(cs.CL)

### ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Likely Toxic Prompts (https://arxiv.org/abs/2407.09447)
Comments:
          9 pages, 2 tables, 2 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 유독한 텍스트를 유도하는 프롬프트를 찾는 새로운 방법을 제안합니다. 기존에는 고정된 언어 모델을 사용하여 독이 있는 텍스트를 생성하는 프롬프트를 발견하는 데 중점을 두었으나, 이는 읽기 어렵고 자연스럽지 않은 텍스트를 생성하는 문제점을 야기했습니다. 반면에, 이 논문에서는 강화 학습(Reinforcement Learning, RL) 기법을 도입하여 유독한 출력과 낮은 혼란도(perplexity)를 동시에 가진 프롬프트를 발견할 수 있는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법인 ASTPrompter는 Markov Decision Process 기반의 적응형 스트레스 테스트(Adaptive Stress Testing, AST) 프레임워크를 사용합니다. 강화 학습을 통해 고유성을 유지하면서 유독한 텍스트를 유도하는 프롬프트를 식별합니다. 구체적으로, 온라인 및 약한 감독 하에 수행되는 Identity Preference Optimization (IPO)을 사용하여 GPT-2와 GPT-2 XL 모델을 방어 모델(defender)로 설정하고, 이에 대항하는 정책을 학습합니다. 이 방법은 정상 대화에서도 유독한 반응을 유도할 수 있는 가능성이 높은 프롬프트를 자동으로 식별합니다.

- **Performance Highlights**: 실험 결과, 제안된 정책은 유독성을 유도하는 가능성이 높은 프롬프트를 생성하는 데 성공적이었으며, 이 프롬프트는 비유독 프롬프트와 비교했을 때 방어 모델의 반응에서 혼란도가 거의 동일하지만 유독성이 훨씬 높은 것으로 나타났습니다. 또한, 공격자가 방어 모델보다 훨씬 작은 크기로 설정된 경우에도 기존 방법보다 우수한 성능을 발휘했습니다. 학습된 공격 전략과 그에 따른 상충관계를 질적으로 분석하여 결과를 논의합니다.



### Open (Clinical) LLMs are Sensitive to Instruction Phrasings (https://arxiv.org/abs/2407.09429)
Comments:
          To appear at BioNLP, ACL 2024

- **What's New**: 이 연구는 Instruction-tuned Large Language Models (LLMs)이 의료 분야에서 어떻게 다른 지시문(phrasings)에 따라 성능이 달라지는지 조사합니다. 특히, 의학 전문가들이 작성한 지시문을 사용하여 LLM의 민감도를 평가합니다.

- **Technical Details**: 일반 도메인 및 임상 도메인에 특화된 총 7가지 LLM의 성능을 평가하였습니다. 데이터를 수집한 MIMIC-III 및 기타 의료 데이터베이스를 활용하여 16개의 분류 및 정보 추출 작업을 수행했습니다. 실험은 다양한 배경의 의료 전문가들이 작성한 지시문 세트를 사용하여 진행되었습니다.

- **Performance Highlights**: 임상 데이터로 특별히 훈련된 도메인 특정 모델이 일반 도메인 모델보다 지시문에 더 민감하게 반응하는 것으로 나타났습니다. 예를 들어, 사망률 예측 작업에서 인종 및 성별에 따라 최고와 최저 성능의 차이가 상당히 컸습니다 (AUROC 최대 0.35 차이). 또한, 지시문 문구의 임의적인 차이가 모델의 공평성에도 영향을 미치는 것으로 밝혀졌습니다.



### Mitigating Entity-Level Hallucination in Large Language Models (https://arxiv.org/abs/2407.09417)
- **What's New**: 최근 큰 언어 모델(LLMs)은 검색 엔진을 통한 전통적인 정보 접근 방식에서 사용자와의 직접적인 질문 및 답변 상호작용으로 변화를 가져왔습니다. 그러나, 이러한 LLMs의 광범위한 채택은 일관되나 사실적으로 부정확한 응답을 생성하는 '환각(hallucination)' 문제를 드러냈습니다. 이를 해결하기 위해, 본 논문에서는 환각 감지를 기반으로 한 동적 검색 보강(DRAD)을 제안합니다. DRAD는 실시간 환각 감지와 외부 지식을 기반으로 한 자기 수정 기능을 통합하여 환각 문제를 탐지하고 경감합니다.

- **Technical Details**: DRAD는 두 가지 주요 구성 요소로 구성됩니다. 첫 번째는 Real-time Hallucination Detection(RHD)로, 외부 모델 없이 실시간으로 잠재적인 환각을 식별합니다. 두 번째는 Self-correction based on External Knowledge(SEK)로, 외부 지식을 사용하여 이러한 오류를 수정합니다. RHD는 LLM의 출력 엔티티의 불확실성을 분석하여 잠재적인 환각을 감지하고, SEK는 외부 지식을 검색하여 LLM의 출력을 수정함으로써 환각을 방지합니다.

- **Performance Highlights**: 실험 결과, DRAD는 환각 감지 및 경감에서 뛰어난 성능을 보여줍니다. 특히 DRAD는 기존의 단일 라운드 및 다중 라운드 검색 보강 방법을 능가하는 성과를 보였습니다. 여러 복잡한 QA 벤치마크 데이터셋에서 DRAD는 큰 모델에서의 환각을 크게 줄이며 우수한 성능을 입증하였습니다.



### SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers (https://arxiv.org/abs/2407.09413)
Comments:
          preprint

- **What's New**: 새로운 QA 데이터셋인 SPIQA(Scientific Paper Image Question Answering)가 도입되었습니다. 이는 컴퓨터 과학 분야의 다양한 연구 논문에서 복잡한 도표와 표를 해석하기 위해 설계된 첫 번째 대규모 QA 데이터셋입니다.

- **Technical Details**: SPIQA 데이터셋은 270K개의 질문으로 구성되어 있으며 자동화 및 수동 관리를 통해 생성되었습니다. 데이터셋은 연구 논문의 도표와 텍스트를 모두 해석해야 하는 'Direct QA with figures and tables', 논문 전체 텍스트를 분석하는 'Direct QA with full paper', 그리고 'Chain-of-Thought (CoT) QA' 총 세 가지 태스크로 설계되었습니다. 각 태스크는 모델의 세부적인 추론 및 근거 능력을 평가합니다.

- **Performance Highlights**: 12개의 주요 멀티모달 모델을 통해 실험을 수행하였으며, InstructBLIP 및 LLaVA 1.5 모델의 세부 조정을 통해 큰 성능 향상을 관찰했습니다. 또한 새로운 평가 메트릭인 LLMLogScore(L3Score)를 도입하여, 로그 가능성 토큰 확률을 기반으로 답변의 품질을 평가하는 방법을 제안했습니다. 이 평가 메트릭은 종래의 LLM 기반 점수보다 효율적임을 보였습니다.



### Is Contrasting All You Need? Contrastive Learning for the Detection and Attribution of AI-generated Tex (https://arxiv.org/abs/2407.09364)
Comments:
          Accepted for publication at the 27th European Conference on Artificial Intelligence (ECAI-2024)

- **What's New**: 큰 언어 모델의 발전은 인간과 AI가 생성한 텍스트의 차이를 모호하게 만들었습니다. 이 논문은 AI 생성 텍스트 탐지와 소유자를 식별하는 문제를 다루면서, WhosAI라는 트리플렛 네트워크 대조 학습 프레임워크를 제안했습니다. 이 프레임워크는 입력 텍스트가 인간 또는 AI에 의해 생성되었는지 예측하고, 텍스트의 저자를 밝혀내는 데 중점을 둡니다.

- **Technical Details**: WhosAI는 다중 생성기 (multiple generators)로부터 의미적 유사성 표현을 학습하도록 설계되어, 탐지 및 귀속 작업을 효과적으로 수행합니다. 이 프레임워크는 모델 불가지론 (model-agnostic)이며, 새로운 AI 텍스트 생성 모델의 출시에도 확장할 수 있는 구조로 설계되었습니다. 이를 위해, 생성된 인스턴스를 프레임워크가 학습한 임베딩 공간에 통합합니다.

- **Performance Highlights**: TuringBench 벤치마크에서 200K 뉴스 기사를 대상으로 한 실험 결과, 제안된 프레임워크는 Turing Test 및 Authorship Attribution 작업 모두에서 탁월한 성능을 보여주었으며, TuringBench 리더보드에 게시된 모든 방법을 능가했습니다.



### Scalability of Bayesian Network Structure Elicitation with Large Language Models: a Novel Methodology and Comparative Analysis (https://arxiv.org/abs/2407.09311)
Comments:
          27 pages

- **What's New**: 이 연구에서는 베이지안 네트워크(BNs) 구조를 도출하는 새로운 방법을 제안합니다. 이 방법은 여러 다양한 경험을 가진 LLM들을 초기화한 후 개별적으로 질의하여 BN 구조를 만들고, 다수결 투표를 통해 최종 구조를 얻는 방식입니다. 이를 기존의 한 가지 방법과 비교하여 다양한 크기의 널리 알려진 BN과 덜 알려진 BN에서 확장성을 연구합니다.

- **Technical Details**: 이 방법은 여러 LLM을 초기화하고 개별적으로 질문을 던져 BN 구조를 만드는 과정을 거칩니다. 그런 다음 다수결 투표를 통해 최종 구조를 결정합니다. 또한 LLM을 통해 BN 구조 도출 시 오염을 검사하는 접근 방식을 제안하며, 이는 일부 널리 알려진 BN이 BN 구조 도출 테스트에 부적합함을 보여줍니다. 일부 BN은 노드 이름이 구분되지 않기 때문에 이러한 실험에 부적합할 수 있습니다.

- **Performance Highlights**: 다른 BN에서의 실험 결과, 세 가지 연구된 LLM 중 하나에서 제안된 방법이 기존 방법보다 더 우수한 성능을 보였습니다. 그러나 BN 크기가 증가함에 따라 두 방법의 성능이 크게 감소하는 것으로 나타났습니다.



### Transformer Layers as Painters (https://arxiv.org/abs/2407.09298)
Comments:
          15 pages total, including references and appendices

- **What's New**: 이번 연구는 사전 훈련된 트랜스포머 모델의 내부 작동 방식을 더 잘 이해하고자 하는 일련의 실증 연구를 제시합니다. 특히 모델의 특정 계층을 생략하거나 순서를 바꿔 실행했을 때의 영향을 살펴봅니다. 이를 통해 기존 모델의 더 나은 사용법을 제안하고 새 변형 모델을 설계할 수 있게 할 것입니다.

- **Technical Details**: 연구는 Llama2(디코더 전용) 및 BERT-Large(인코더 전용)와 같은 두 가지 주요 트랜스포머 모델에 대한 실험을 포함합니다. Llama2-7B, Llama2-13B, Llama2-70B를 포함해 계층 생략, 계층 순서 변경, 병렬 처리 등의 전략을 실험했습니다. GLUE 벤치마크를 포함한 다양한 표준 벤치마크를 사용하여 모델의 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, 중간 계층(middle layers)이 공통 표현 공간(shared representation space)을 사용하며, 개별 계층을 생략하거나 순서를 바꿔도 큰 성능 저하 없이 수행할 수 있다는 것을 발견했습니다. 첫 번째와 마지막 계층의 역할은 더 특이한 것으로 나타났습니다. 이러한 특성으로 인해 일부 문제에 대해 계층을 변경하거나 병렬로 실행함으로써 정확성과 대기 시간을 균형 있게 맞출 수 있습니다.



### DAHRS: Divergence-Aware Hallucination-Remediated SRL Projection (https://arxiv.org/abs/2407.09283)
Comments:
          15 pages, 6 figures

- **What's New**: 이번 연구는 의미역 분석(SRL)에서 발생하는 잘못된 역할 할당 문제를 해결하기 위해 'Divergence-Aware Hallucination-Remediated SRL Projection' (DAHRS)을 제안합니다. 기존 대형 언어 모델(LLMs) 기반의 상태-of-the-art SRL 프로젝션이 잘못된 역할 라벨로 가득한 결과를 초래하는 문제를 다룹니다.

- **Technical Details**: DAHRS는 언어적 지식을 활용하여 정렬 오류를 수정하고, 최초 정렬 후 그리디한 'First-Come First-Assign'(FCFA) 알고리즘을 적용하여 역할 프로젝션을 수행합니다. 이는 자연적으로 발생하는 언어의 차이점을 인식하여 잘못된 역할 할당을 방지합니다. 특히 EN-FR(영어-프랑스어)와 EN-ES(영어-스페인어)와 같은 언어 쌍에서 더 나은 성능을 입증합니다.

- **Performance Highlights**: CoNLL-2009 데이터를 기준으로, DAHRS는 단어 수준 F1 점수에서 기존 XSRL을 능가하며 EN-FR의 경우 87.6% vs 77.3%, EN-ES의 경우 89.0% vs 82.7%를 기록했습니다. 인간 평가에 따른 구 수준 평가에서는 각각 89.1%(EN-FR)와 91.0%(EN-ES)를 달성했습니다. 또한, 영어-타갈로그 같은 다른 언어 쌍에도 적용할 수 있도록 디버전스 메트릭(dispersion metric)을 정의했습니다.



### H2O-Danube3 Technical Repor (https://arxiv.org/abs/2407.09276)
- **What's New**: 이번 논문에서는 H2O-Danube3라는 소형 언어 모델 시리즈를 소개합니다. H2O-Danube3-4B와 H2O-Danube3-500M 두 가지 모델로 구성되어 있으며, 각각 6T와 4T 토큰으로 훈련되었습니다. 이 모델들은 주로 고품질의 영어 웹 데이터를 사용하여 세 단계의 사전 훈련을 거쳤으며, 최종적으로 채팅 버전을 위한 감독 하에 튜닝되었습니다. H2O-Danube3 모델은 현대 스마트폰에서 효율적으로 실행될 수 있어 로컬 추론과 빠른 처리 기능을 제공합니다.

- **Technical Details**: H2O-Danube3는 디코더 전용 LLM 모델로, Llama 모델 아키텍처를 기반으로 하고 있습니다. 32,000개의 어휘 크기를 가진 Mistral 토크나이저를 사용하며, 최대 문맥 길이는 8,192 토큰입니다. Grouped Query Attention과 같은 최적화 기법을 사용하여 파라미터와 컴퓨팅 효율성을 높였습니다. H2O-Danube3-4B는 3.96B의 학습 가능한 파라미터를 가지며, H2O-Danube3-500M 모델은 500M 파라미터를 가지고 있습니다. 세 단계에 걸쳐 점차적으로 노이즈가 적은 데이터를 섞어 사용하며 훈련을 진행하였습니다. 각 단계에서 데이터를 점진적으로 고품질 데이터로 전환하였습니다.

- **Performance Highlights**: H2O-Danube3-4B는 다양한 벤치마크에서 매우 경쟁력 있는 성능을 보여주었습니다. 특히 CommonsenseQA와 PhysicsQA에서 최고 성능을 기록하였으며, 수학 중심의 GSM8K 벤치마크에서는 50.14%의 높은 정확도를 보였습니다. H2O-Danube3-500M 모델은 12개의 벤치마크 중 8개에서 유사 크기의 Qwen2-0.5B-Instruct 모델보다 우수한 점수를 기록하였습니다. 또한, 채팅 평가에서는 MT-Bench와 WildBench-v2 벤치마크에서 유사 크기 모델들보다 더 높은 점수를 획득하였습니다.



### Context Embeddings for Efficient Answer Generation in RAG (https://arxiv.org/abs/2407.09252)
Comments:
          10 pages

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG)의 효율성을 크게 향상시킬 수 있는 새로운 컨텍스트 압축 방법인 COCOM (COntext COmpression Model)을 제안합니다. 이 방법은 긴 컨텍스트를 소수의 컨텍스트 임베딩(Context Embeddings)으로 압축하여, 생성 시간을 크게 단축하면서도 답변의 품질을 유지 또는 향상시킵니다. COCOM은 다양한 압축율을 제공하여, 생성 품질과 디코딩 시간을 조절할 수 있습니다.

- **Technical Details**: COCOM은 여러 문서를 보다 효과적으로 처리할 수 있도록 설계되었으며, 이는 긴 입력에 대한 디코딩 시간을 기존 방법보다 크게 줄입니다. 이 방법은 넓은 표면 형태의 입력을 소수의 임베딩으로 변환하여, 디코딩 시간 동안의 성능을 크게 향상시킵니다. 특히 COCOM에서는 프리트레이닝과 튜닝 접근법을 통해 컨텍스트 압축의 효과를 극대화하였습니다. 기존의 임베딩 기반 압축 방법들과 비교했을 때, COCOM은 더 높은 효율성과 성능을 보입니다.

- **Performance Highlights**: COCOM의 효율성 연구에서는 최대 5.69배의 인퍼런스 시간 단축과 최대 22배의 GFLOP 감소를 달성하면서도 높은 성능을 유지하는 것을 보여줍니다. 특히, 여러 실험을 통해 사전 학습 컬렉션, 프리트레이닝, 파인튜닝, 그리고 디코더의 동결 또는 비동결이 모델 성능에 미치는 영향을 분석하였습니다. 이러한 분석을 통해 COCOM이 다른 압축 방법들보다 우수한 성과를 보이는 것을 입증하였습니다.



### The Sociolinguistic Foundations of Language Modeling (https://arxiv.org/abs/2407.09241)
- **What's New**: 이 논문은 대규모 언어 모델(LLM, Large Language Models)을 사회언어학적 관점에서 이해하려는 시도를 제안합니다. 연구진은 LLM이 기본적으로 특정 언어의 변이(varieties)를 모델링하고 있음을 주장하며, 이를 바탕으로 LLM의 개발 및 배포에 관해 논의합니다.

- **Technical Details**: 사회언어학의 언어 변이 개념을 소개하고 이를 LLM의 관점에서 기술합니다. 그런 다음, LLM의 5가지 기본 과제—사회적 편향(social bias), 도메인 적응(domain adaptation), 정렬(alignment), 언어 변화(language change), 그리고 확장(scale)—에 대해 논의하고, 이러한 과제들이 어떻게 사회언어학적 관점에서 해결될 수 있는지 분석합니다.

- **Performance Highlights**: 논문에서는 특히 특정 언어 변이를 정확하게 나타내는 훈련 코퍼스를 신중하게 정의하고 컴파일하는 것이 LLM의 성능을 극대화하고 사회적 가치를 높이는 데 중요하다고 결론짓습니다. 이와 같은 접근 방식은 언어 모델의 성능, 평가, 및 적용 가능성을 개선하고, 안전하고 윤리적인 AI 시스템을 구축하는 데 기여할 수 있습니다.



### Pronunciation Assessment with Multi-modal Large Language Models (https://arxiv.org/abs/2407.09209)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)을 기반으로 한 점수 시스템을 제안합니다. 이는 텍스트 관련 점수 작업에 긍정적인 영향을 미치는 LLMs의 능력을 활용합니다. 구체적으로, 음성 인코더가 학습자의 음성을 문맥적 특징으로 매핑한 후, 어댑터 레이어가 이 특징들을 잠재 공간에서 텍스트 임베딩과 정렬합니다. 평가 작업 특정 프리픽스와 프롬프트 텍스트가 임베딩되고, 모달리티 어댑터 레이어가 생성한 특징과 결합되어 LLMs가 정확성과 유창성 점수를 예측할 수 있습니다.

- **Technical Details**: 제안된 멀티모달 발음 평가 방법에는 세 가지 구성 요소가 포함됩니다: data2vec2를 이용한 사전 학습된 음향 인코더, 모달리티 어댑터로서의 m-adapter, 그리고 LLM 기반의 발음 평가 모듈. 모델 학습 과정은 두 단계로 나뉩니다. 첫 번째 단계에서는 음성 인식 작업을 통해 모델을 학습하고, 두 번째 단계에서는 구체적인 점수 데이터를 기반으로 미세 조정을 수행합니다. 선행 연구는 주로 텍스트 관련 점수 작업에 초점을 맞췄으나, 본 연구는 음성 인식 및 평가로 확장했습니다.

- **Performance Highlights**: 제안된 멀티모달 방법은 전통적인 align-based와 align-free 시스템과 비교하여 경쟁력 있는 성능을 달성했으며, Speechocean762 데이터셋에서 유의미한 결과를 얻었습니다. 또한, 점수 시스템의 프롬프트 텍스트와 학습 전략의 기여도를 더 잘 이해하기 위해 소거 연구(ablation study)를 수행했습니다.



### Enhancing Depressive Post Detection in Bangla: A Comparative Study of TF-IDF, BERT and FastText Embeddings (https://arxiv.org/abs/2407.09187)
- **What's New**: 이 연구는 소셜 미디어 분석을 통해 사용자의 우울증을 감지하는 것의 중요성을 강조하며, 특히 방글라어(Bangla)와 같은 저평가된 언어에 중점을 둡니다. 본 연구는 고급 자연어 처리(Advanced Natural Language Processing) 기법을 사용하여 방글라어의 우울증 관련 소셜 미디어 게시물을 식별하는 접근 방법을 도입했습니다. 도메인 전문가들이 주석을 달아 고품질 데이터를 확보하고, 클래스 불균형 문제를 해결하기 위해 무작위 과 샘플링(Random Oversampling)을 사용했습니다.

- **Technical Details**: 우리는 Term Frequency-Inverse Document Frequency(TF-IDF), BERT 임베딩(BERT Embedding) 및 FastText 임베딩을 비롯한 다양한 수치 표현 기법을 탐구하고, 이를 딥러닝 기반의 Convolutional Neural Network-Bidirectional Long Short-Term Memory(CNN-BiLSTM) 모델과 통합했습니다. 특히 BERT와 CNN-BiLSTM 아키텍처를 결합해 방글라어 텍스트의 미묘한 차이를 효과적으로 인식할 수 있는 것으로 나타났습니다.

- **Performance Highlights**: 광범위한 실험을 통해 BERT 접근 방식이 다른 방법들보다 성능이 뛰어난 것으로 밝혀졌으며, F1-스코어(F1-score) 84%를 달성했습니다. 기존 기술과의 비교 분석에서도 BERT 임베딩이 평가 지표와 데이터셋 주석의 신뢰성 측면에서 더 우수한 성능을 보였습니다. 이를 통해 방글라어의 우울증 게시물 탐지를 위한 신뢰성 있는 도구 개발에 크게 기여했습니다.



### Does Incomplete Syntax Influence Korean Language Model? Focusing on Word Order and Case Markers (https://arxiv.org/abs/2407.09184)
Comments:
          COLM 2024; Code and dataset is available in this https URL

- **What's New**: 최근 연구에서는 한국어와 같이 다소 유동적인 어순을 허용하는 언어에서의 문법 정보가 언어 모델(LM)의 성능을 크게 향상시킬 수 있음을 밝혔습니다. 이러한 한국어의 어순 변형과 격조사의 생략이 빈번히 발생하는 점을 고려하여, 본 연구에서는 SIKO(Syntactically Incomplete Korean) 데이터셋을 도입하여 한국어 언어 모델이 이러한 문법적 불완전성을 얼마나 잘 처리할 수 있는지 평가했습니다.

- **Technical Details**: 한국어는 고정된 S-O-V(주어-목적어-동사) 어순을 따르지만 격조사의 사용으로 어순에 유연성이 있습니다. SIKO 데이터셋은 이러한 격조사 생략과 불완전한 어순을 포함하는 문장으로 구성되었습니다. 이를 통해 다양한 실험을 진행했으며, 특히 Text Classification(TC), Natural Language Inference(NLI), 대화 주제 분류, 대화 요약 등의 기본 작업에 대한 모델 성능을 평가했습니다.

- **Performance Highlights**: SIKO 데이터셋을 통한 실험에서 한국어 언어 모델은 문법적으로 불완전한 입력을 정확히 처리하는 능력을 보여주었습니다. SIKO로 미세조정된 모델은 격조사 생략 및 어순 변경과 같은 일반적인 문법 불완전성을 더욱 잘 처리하는 능력을 보였습니다. 이는 한국어의 고유한 특성을 반영한 데이터 증강 기술로서의 효과를 입증하였습니다.



### Exploring the Effectiveness of Methods for Persona Extraction (https://arxiv.org/abs/2407.09181)
- **What's New**: 이 논문은 러시아어 대화 참여자 정보를 추출하고 그들의 성능을 평가하는 방법을 연구한 것입니다. 이를 위해 Multi-Session Chat 데이터셋을 여러 번역 모델을 통해 러시아어로 번역하여 데이터 품질을 개선하였습니다.

- **Technical Details**: 추출 모델의 효율성을 평가하기 위해 F-score 개념에 기초한 메트릭(metric)을 제시합니다. 이 메트릭은 훈련된 분류기를 사용하여 페르소나(persona)가 속한 대화 참여자를 식별합니다. 실험은 MBart, FRED-T5, Starling-7B (Mistral 기반), Encoder2Encoder 모델에서 수행되었습니다.

- **Performance Highlights**: 결과는 모든 모델이 페르소나 추출 작업에서 불충분한 리콜(recall) 수준을 보였음을 시사합니다. NCE Loss를 도입하면 모델의 정밀도(precision)가 향상되었지만 리콜은 감소하였습니다. 또한, 모델의 크기를 키우면 페르소나 추출이 더 향상되었습니다.



### Stepwise Verification and Remediation of Student Reasoning Errors with Large Language Model Tutors (https://arxiv.org/abs/2407.09136)
Comments:
          Preprint. Nico Daheim and Jakub Macina contributed equally. Code and dataset can be found under: this https URL

- **What's New**: 최근 대형 언어 모델(LLMs)은 고품질 맞춤형 교육을 모든 이에게 확산시킬 수 있는 기회를 제공하고 있습니다. 이 연구는 학생의 문제 해결을 지원하는 대화형 튜터 모델을 구축하는 데 중점을 두고 있으며, 학생의 오류를 정밀하게 감지하고 해당 오류에 맞춘 피드백을 제공하는 데 어려움을 겪고 있는 기존 LLM의 한계를 해결하고자 합니다. 이를 위해 학생의 해결 과정을 검증하는 방법을 통합하여 튜터 응답 생성의 품질을 개선하는 방안을 제시합니다.

- **Technical Details**: 연구팀은 1,000개의 단계적 수학 추론 체인을 수집하고, 교사가 주석으로 기록한 첫 번째 오류 단계를 포함한 데이터셋을 만들었습니다. 학생의 해결 과정에서의 오류를 감지하는 것이 현재 모델에게 어려운 과제임을 실증적으로 보여주고, 여러 검증기를 제안하고 평가했습니다. 이러한 검증기의 출력을 사용하여 생성 모델이 더욱 구체적인 오류 중심의 응답을 생성하도록 유도했습니다. 제안된 검증 방법은 프롬프트와 파인 튜닝된 언어 모델을 기반으로 하며, 텍스트 검증과 참조 해결 단계와의 정렬을 통해 학생 해결 과정을 검증합니다.

- **Performance Highlights**: 제안된 데이터셋을 사용한 파인 튜닝은 프롬프트 기반 최첨단 LLM보다 작은 LLM에서도 더 나은 성능을 보였습니다. 검증기 출력을 응답 생성 단계에 통합함으로써 자동 평가와 실제 교사를 통한 평가에서 생성된 응답이 더 정확한 학생 오류에 맞춰지고 환각(hallucinations)이 줄어들며 실질적인 피드백을 제공하는 데 효과적임을 확인했습니다. 이러한 개선은 검증 출력이 정확할 때 더욱 두드러졌습니다.



### Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training (https://arxiv.org/abs/2407.09121)
- **What's New**: 이 연구는 대형 언어 모델(LLM)의 안전성 조정 데이터 내에서 부적절한 거부 위치 편향(refusal position bias)을 식별하고 해결하는 새로운 접근법을 제안합니다. Decoupled Refusal Training (DeRTa)이라는 새로운 방법을 도입하여, LLM이 해로운 요청에 대해 언제든지 적절하게 거부할 수 있도록 하는 안전성을 크게 향상시킵니다. 이 방법은 모델이 유해한 내용을 생성하지 않도록 훈련하는 MLE (Maximum Likelihood Estimation)와 RTO (Reinforced Transition Optimization) 두 가지 주요 구성 요소를 포함합니다.

- **Technical Details**: DeRTa는 두 가지 새로운 구성 요소를 도입합니다: (1) 유해한 응답 접두사(Harmful Response Prefix)를 사용한 최대 우도 추정(MLE)으로, 유해한 응답의 일부를 안전한 응답의 시작 부분에 추가하여 모델이 유해한 콘텐츠를 인식하고 회피할 수 있도록 훈련합니다. (2) 강화된 전환 최적화(RTO)로, 유해한 응답 시퀀스 내의 어느 위치에서나 안전 거부로 전환할 수 있는 능력을 모델에 부여합니다. 이 방법은 모델이 잠재적 위험을 인식하고 안전한 응답을 생성하도록 할 때 보다 일관되게 전환할 수 있도록 돕습니다.

- **Performance Highlights**: LLaMA3와 Mistral 모델을 사용한 6가지 공격 시나리오에서 우리의 접근법이 GPT-4를 포함한 잘 알려진 모델들을 능가함을 실험적으로 보여줍니다. 특히 우리의 방법은 최근의 고급 공격 방식(CodeAttack)을 방어하는데 성공했습니다. 또한, 정량적 및 정성적 평가 결과는 우리의 전략이 LLM이 잠재적인 위험을 인식하고 생성을 중단하는 능력을 효과적으로 향상시킴을 뒷받침합니다.



### New Desiderata for Direct Preference Optimization (https://arxiv.org/abs/2407.09072)
- **What's New**: 본 논문은 기존의 직선호향 최적화(DPO, Direct Preference Optimization) 방법이 인간 선호와 사전 학습된 모형 간의 균형을 유지하는 데 있어서의 한계를 지적하고, 이를 보완하기 위한 새로운 DPO 유사 손실 함수를 제안합니다. 이 새로운 손실 함수는 제약 조건에 의존하지 않으면서 모델 성능을 개선하는 것을 목표로 설계되었습니다.

- **Technical Details**: 기존의 LLM(Large Language Models)들은 RLHF(Reinforcement Learning with Human Feedback)를 사용하여 모델 응답을 인간 선호에 맞추려는 경향이 있었으나, 이러한 과정에서 발생하는 불안정성을 회피하기 위해 DPO 방법이 도입되었습니다. 그러나 DPO 방법에도 여전히 해결되지 않은 한계가 존재함을 확인하고, 새로운 평가 기준을 도입하여 이러한 한계를 명확히 하였습니다. 기존 DPO 방식이 사전 모델이 강한 영역에서 성능을 유지하면서 다른 영역에서는 성능을 개선하는 데 어려움이 있다는 것을 증명하였습니다. 이를 통해 새롭게 제안된 ℓTYPO 손실 함수가 이러한 평가 기준을 충족하도록 하였습니다.

- **Performance Highlights**: 제안된 ℓTYPO 손실 함수는 기존의 DPO 방식과 비교하여 Monte-Carlo 시뮬레이션을 통해 우수한 성질을 실험적으로 입증하였습니다. 이 새로운 손실 함수는 제약에 의존하지 않으면서도 더 나은 모델 성능을 발휘하는 것을 목표로 합니다.



### 3M-Health: Multimodal Multi-Teacher Knowledge Distillation for Mental Health Detection (https://arxiv.org/abs/2407.09020)
- **What's New**: 새로운 연구에서는 멀티모달 및 멀티교사 지식 증류 모델(Multimodal and Multi-Teacher Knowledge Distillation model)을 통해 기존의 텍스트 기반 정신 건강 분류의 한계를 극복하기 위해 시도합니다. 이 접근법은 텍스트 외에도 음향 정보와 감정이 풍부한 특징을 추가로 통합하여 정신 건강 상태를 더 정확하게 감지하는 것을 목표로 합니다.

- **Technical Details**: 3M-Health라는 모델은 세 가지 주요 교사 모델로 구성됩니다. 이들 각각은 서로 다른 모달리티를 중점적으로 학습하여 정신 건강과 관련된 게시물의 다양한 측면을 독립적으로 해석합니다. 첫 번째 텍스트 기반 교사는 입력 텍스트로부터 의미를 이해하도록 설계되었으며, 두 번째 감정 기반 교사는 감정을 해석하며, 세 번째 오디오 기반 교사는 음향을 통해 감정을 인식합니다. 이 모델은 다양한 교사들이 학습한 특징을 이용하여 학생 모델이 더 나은 성능을 발휘할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, 멀티모달 교사 모델을 활용한 우리의 접근법이 기존의 텍스트 기반 모델보다 우수한 성능을 보여주었습니다. 모든 관련 코드는 논문 출판 시에 공개될 예정입니다.



### CompAct: Compressing Retrieved Documents Actively for Question Answering (https://arxiv.org/abs/2407.09014)
Comments:
          Code available at this https URL

- **What's New**: CompAct라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 방대한 문서를 압축하여 핵심 정보를 잃지 않도록 하는 능동적(active) 접근 방식을 채택하여 다중 홉 질문 응답(Question Answering, QA) 벤치마크에서 성능을 크게 향상시켰습니다.

- **Technical Details**: CompAct는 두 가지 주요 구성 요소를 포함합니다: 능동적 압축(active compression)과 조기 종료(early termination). 모델은 이전에 압축된 컨텍스트와 새롭게 제공된 세그먼트를 공동으로 분석하여 입력 문서를 능동적으로 캡슐화합니다. 이 과정을 거치며 질문과 관련된 정보만을 보존하여 밀도 높은 압축된 컨텍스트를 생성합니다. 모든 단계에서 정보의 관련성과 완전성을 기반으로 압축 과정을 종료할 지 여부를 결정합니다.

- **Performance Highlights**: 다섯 개의 QA 벤치마크 데이터셋에서 실험을 통해 CompAct 프레임워크가 다큐먼트를 고도로 압축하면서도 중요한 정보를 놓치지 않고 성능을 크게 향상시킴을 입증했습니다. CompAct는 HotpotQA 데이터셋에서 7.0 (F1) 포인트 향상과 함께 47배 높은 압축률을 달성했습니다. 이는 다양한 리트리버(retriever) 및 리더(reader)와 호환되어 플러그인 모듈로서의 효과를 입증합니다.



### One Stone, Four Birds: A Comprehensive Solution for QA System Using Supervised Contrastive Learning (https://arxiv.org/abs/2407.09011)
Comments:
          14 pages, under review

- **What's New**: 이 논문은 질문 응답(QA) 시스템의 안정성과 효율성을 향상시키기 위해 감독 대조 학습(SCL)을 활용한 혁신적이고 종합적인 솔루션을 제시합니다. 최근 사전 훈련된 언어 모델의 발전에도 불구하고 기존 QA 시스템이 기능성과 학습 효율성에서 아직도 개선할 여지가 있다는 문제를 해결하고자 합니다.

- **Technical Details**: 현 QA 시스템의 주요 문제를 해결하기 위해 네 가지 주요 작업을 정의합니다: 사용자 입력 의도 분류(user input intent classification), 도메인 외부 입력 감지(out-of-domain input detection), 새로운 의도 발견(new intent discovery), 지속 학습(continual learning). 이를 위해 SCL 기반의 통합 표현 학습 방식을 사용하여 인트라-클래스 컴팩트(intra-class compact) 및 인터-클래스 스캐터(inter-class scattered) 특징 공간을 구축합니다.

- **Performance Highlights**: 제안된 접근법은 다운스트림 작업에서 최소한의 튜닝으로 모델 효율성을 크게 향상시키며, 모든 작업에서 최신 기술을 능가하는 성능을 발휘합니다. 특히, SCL을 사용한 새로운 방법론이 기능성과 학습 효율성을 강화하여 QA 시스템의 전체적인 성능을 증진시킵니다.



### Benchmarking Language Model Creativity: A Case Study on Code Generation (https://arxiv.org/abs/2407.09007)
- **What's New**: 이번 연구에서는 언어 모델(LLM)의 창의성을 측정하는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 인지과학에서 제시한 수렴적 사고(convergent thinking)와 발산적 사고(divergent thinking)를 모두 통합한 NeoGauge 지표를 사용합니다. 또한 'Denial Prompting'이라는 기법을 통해 LLM에게 새로운 제약 조건을 부여하여 더 창의적인 솔루션을 도출하도록 합니다.

- **Technical Details**: NeoGauge 지표는 두 가지 주요 요소를 포함합니다. 첫째는 'Denial Prompting'을 통해 생성된 솔루션이 주어진 제약 조건을 얼마나 잘 충족하는지 (수렴적 사고) 검증하고, 둘째는 그 솔루션이 얼마나 참신한지 (발산적 사고)를 평가하는 것입니다. Codeforces 문제를 사용해 다양한 LLM의 NeoGauge 점수를 계산하고, Monte Carlo Tree Search(MCTS), self-correction, planning, sampling 같은 고급 추론 전략의 창의성 평가를 진행했습니다.

- **Performance Highlights**: GPT-4가 가장 창의적인 모델임에도 불구하고 인간과 같은 창의성을 보여주지 못한다는 결과가 나왔습니다. 또한, 고급 추론 전략을 사용해도 창의성에서 현저한 개선은 없었습니다. 이 연구의 부산물로 NeoCoder 데이터셋이 공개되어 다음 세대 모델의 결과를 재현할 수 있습니다.



### Self-Prompt Tuning: Enable Autonomous Role-Playing in LLMs (https://arxiv.org/abs/2407.08995)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 다양한 역할을 시뮬레이션하는 능력이 탁월해 다양한 지침과 맥락에 따라 대화 스타일과 인지 과정을 정확하게 모사할 수 있게 되었습니다. 연구에 따르면 LLMs에 전문가 역할을 부여하는 '롤 플레이 프롬팅(role-play prompting)' 전략이 해당 도메인에서의 성능을 향상시킬 수 있다고 합니다. 하지만 기존 프롬트(prompt)를 수동으로 설계해야 하는 필요성이 있었는데, 이를 해결하기 위해 'self-prompt tuning'을 제안하게 되었습니다. GPT-4를 이용해 LIMA 데이터셋을 주석 처리하고, Llama-2-7B 및 Mistral-7B를 이 데이터셋으로 미세 조정하여, 자동으로 전문가 롤 프롬트를 생성할 수 있도록 하였습니다.

- **Technical Details**: 본 연구에서는 'Self-Prompt Tuning'이라는 접근법을 통해 LLMs 자체가 롤 플레이 프롬트를 생성하도록 합니다. 이를 위해 LIMA 데이터셋을 기반으로 GPT-4를 사용하여 각 데이터 포인트에 롤 플레이 프롬트를 주석 처리한 LIMA-Role 데이터셋을 생성하였고, 해당 데이터셋으로 Llama-2-7B 및 Mistral-7B 모델을 미세 조정(fine-tuning)하였습니다. 이렇게 미세 조정된 LLMs는 주어진 질문에 대해 자동으로 전문가 롤 프롬트를 생성할 수 있습니다.

- **Performance Highlights**: 자체 프롬트 튜닝된 LLMs는 일반적인 지침 튜닝된 베이스라인과 비교하여 대부분의 데이터셋에서 성능이 우수함을 보여주었습니다. 이는 복잡한 프롬팅 전략을 자동화하는데 있어서 미세 조정의 큰 잠재력을 강조하며, 더 나아가 다양한 프롬팅 전략을 자동화하는 새로운 가능성을 제시합니다. 또한, 본 연구에서는 LIMA-Role 데이터셋, 모델, 그리고 코드를 공개하여 후속 연구를 지원하고 자 합니다.



### Robustness of LLMs to Perturbations in Tex (https://arxiv.org/abs/2407.08989)
Comments:
          8 pages, 1 figure, 6 tables, updated with results also from GPT-4, LLaMa-3

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 현실 세계의 노이즈가 섞인 데이터에 얼마나 잘 견디는지를 평가한 결과를 제시합니다. 특히, LLMs가 문법 오류 수정(Grammar Error Correction, GEC) 및 어휘 의미 변화(Lexical Semantic Change, LSC)에서 최신 성능을 구현했다는 점입니다.

- **Technical Details**: 연구팀은 다양한 데이터셋에 여러 수준의 노이즈를 인위적으로 도입하여, 원본 텍스트의 왜곡된 변형에 대해 LLMs의 로버스트니스를 체계적으로 평가했습니다. 구체적으로, 텍스트를 점진적으로 심각하게 왜곡시키면서 LLMs의 내부 인코딩 변화를 측정했습니다. 또한, 실제 데이터셋에서 흔히 발생하는 오류를 모방한 여러 벤치마크에서도 LLMs의 로버스트니스를 평가했습니다.

- **Performance Highlights**: LLMs는 GEC와 LSC에서 최소한의 프롬프트로도 새롭게 최고 수준의 성능을 달성했습니다. 이는 LLMs가 잘 훈련된 상태에서 경미한 텍스트 불규칙성을 상당히 잘 견뎌낼 수 있음을 의미합니다. 또한, 인간이 수정한 출력과 LLM이 수정한 출력에 대한 인간 선호도 데이터셋을 공개했습니다.



### Towards Chapter-to-Chapter Context-Aware Literary Translation via Large Language Models (https://arxiv.org/abs/2407.08978)
Comments:
          Preprint

- **What's New**: 새로운 연구에서 기존 문서 수준 번역 데이터셋에서 담화 현상이 희박한 문제를 해결하고자, 중국어-영어 문학 작품으로 구성된 새로운 데이터셋을 소개했습니다. 이 데이터셋은 160권의 책을 포함하며, 복잡한 담화 구조를 갖추고 있습니다. 또한, 챕터 단위로 번역하는 'chapter-to-chapter (Ch2Ch)' 번역 설정을 제안해 기존의 문장 단위 맞춤현실 요건을 초월하는 새로운 접근법을 선보였습니다.

- **Technical Details**: 이 연구에서는 일반적으로 사용되는 기계 번역 모델을 Ch2Ch 설정 하에 검사하였고, 대형 언어 모델(LLMs)을 문학 번역 도메인에서 미세 조정하는 방법을 소개했습니다. 특히, Transformer 구조를 가진 디코더 전용 대형 언어 모델(LLMs)을 활용해 문맥 인식 번역 시스템에 적합한 Fine-tuning 절차를 제안하여 기존 베이스라인과 비교해 놀라운 성능 향상을 이뤄냈습니다.

- **Performance Highlights**: Ch2Ch 설정 하에서의 문학 번역은 모델 학습 방법과 번역 디코딩 알고리즘 모두에 매우 도전적임을 분석을 통해 밝혔습니다. JAM 데이터셋에서 최근의 대형 언어 모델(LLMs)의 성능을 평가한 결과, 제안된 Fine-tuning 절차를 통해 일관된 문학 소설 번역이 가능한 결과를 도출했습니다.



### Empowering Few-Shot Relation Extraction with The Integration of Traditional RE Methods and Large Language Models (https://arxiv.org/abs/2407.08967)
- **What's New**: 이번 논문은 전통적인 관계 추출(Relation Extraction, RE) 모델과 대형 언어 모델(Large Language Models, LLMs)을 결합한 새로운 이중 시스템 증강 관계 추출기(Dual-System Augmented Relation Extractor, DSARE)을 제안합니다. DSARE는 FSRE(Few-Shot Relation Extraction) 과제에서 전통적인 RE 모델의 전처리 지식 부족과 LLM의 테스크-특정 능력 부족 문제를 해결하고자 하였습니다.

- **Technical Details**: DSARE는 LLM-증강 RE 모듈, RE-증강 LLM 모듈, 통합 예측 모듈의 세 가지 주요 구성 요소로 이루어집니다. LLM-증강 RE 모듈은 추가적인 레이블이 지정된 데이터를 생성하여 전통적인 RE 모델의 훈련을 증강합니다. RE-증강 LLM 모듈은 가장 가치 있는 샘플을 추출하여 LLM의 In-Context Learning에 활용합니다. 통합 예측 모듈은 두 모듈의 예측 결과를 종합하여 최종 결정을 내립니다.

- **Performance Highlights**: 세 개의 공개 데이터셋을 활용한 실험 결과, DSARE는 기존 방법 대비 우수한 성능을 보였습니다. 전통적인 RE 모델과 LLM을 결합하는 방식의 필요성을 실험을 통해 입증하였습니다.



### Domain-Hierarchy Adaptation via Chain of Iterative Reasoning for Few-shot Hierarchical Text Classification (https://arxiv.org/abs/2407.08959)
Comments:
          9 pages, 2 figures, Accepted by IJCAI2024

- **What's New**: 이 논문에서는 전이 학습된 언어 모델(PLMs)이 몇 가지 사례(몇 샷) 학습 환경에서 어떻게 성능을 발휘하는지를 다룹니다. 특히, 복잡한 구조적 시나리오인 계층적 텍스트 분류(HTC)에서 일관된 성능을 유지하는데 어려움이 있다는 문제를 지적하고 있습니다. 저자들은 Hierarchical Iterative Conditional Random Field (HierICRF)라는 새로운 방법을 제안하여, PLM의 비구조적 지식의 구조적 도메인 계층으로의 적응을 돕습니다.

- **Technical Details**: HierICRF는 경로 라우팅 관점에서 설계된 간단하지만 효과적인 방법입니다. 이 방법은 3단계로 이루어집니다: (1) 계층 인식 프롬프트를 작성하여 모델이 계층적으로 반복된 시리즈를 생성하도록 유도, (2) 이 시리즈를 verbalizer에 넣어 마스크된 언어 모델링(MLM) 로그릿을 얻음, (3) 계층적 의존성 트리에 기반한 단계를 거쳐 최종 예측을 수행. 또한 유추 단계에서는 Viterbi 알고리즘을 사용해 최종 예측 결과를 도출합니다.

- **Performance Highlights**: HierICRF는 BERT와 T5 아키텍처에서 실험을 수행했으며, 두 인기 있는 HTC 데이터셋에서 이전 최첨단(SOTA) few-shot HTC 방법보다 상당한 성능 향상을 보여주었습니다. WOS 데이터셋에서 9.3%, DBpedia 데이터셋에서 4.38%의 CMacro-F1 개선을 달성하며, SOTA 계층적 일관성 성능을 유지하는데 성공하였습니다.



### Detect, Investigate, Judge and Determine: A Novel LLM-based Framework for Few-shot Fake News Detection (https://arxiv.org/abs/2407.08952)
- **What's New**: 이번 논문에서는 소수 샘플로 가짜 뉴스를 탐지하는 방법인 Few-Shot Fake News Detection (FS-FND)을 향상시키기 위해 Dual-perspective Augmented Fake News Detection (DAFND) 모델을 제안하고 있습니다. 이는 기존의 대규모 언어 모델(Large Language Models, LLMs)이 가지고 있던 두 가지 주요 한계인 '이해의 모호성 (Understanding Ambiguity)'과 '정보 부족 (Information Scarcity)' 문제를 해결하는데 중점을 둡니다.

- **Technical Details**: DAFND 모델은 총 네 가지 주요 모듈로 구성되어 있습니다: (a) Detection Module: 뉴스 기사의 키워드를 추출합니다. (b) Investigation Module: 기사의 내부와 외부 정보를 검색합니다. (c) Judge Module: 검색된 정보를 바탕으로 예측을 생성합니다. (d) Determination Module: 두 예측을 통합하여 최종 결정을 내립니다. 이를 통해 LLM의 이해력을 향상시키고 최신 정보를 반영하여 가짜 뉴스를 탐지합니다.

- **Performance Highlights**: 공개 데이터셋 두 개에서의 광범위한 실험 결과, 제안된 DAFND 모델이 특히 자원이 적은 환경에서 효과적임을 입증하였습니다. 코드 또한 논문이 승인되면 공개될 예정입니다.



### Large Language Models as Biomedical Hypothesis Generators: A Comprehensive Evaluation (https://arxiv.org/abs/2407.08940)
Comments:
          Accepted to COLM 2024. This is an extended version of the paper at arXiv:2311.05965

- **What's New**: 지난 수년간 생물의학(biomedical) 분야에서 데이터와 지식의 폭발적인 성장이 이어지면서 연구자들이 최신 발견을 따라잡고 새로운 가설을 생성하기 어려워졌습니다. 이 논문에서는 대규모 언어 모델(LLMs)을 생물의학 가설 생성 도구로서 평가하고, 새로운 평가 프레임워크와 혁신적인 실험을 통해 LLM의 능력을 탐구합니다.

- **Technical Details**: 생물의학 문헌으로부터 배경-가설 쌍 데이터를 구성하고, 이를 기반으로 LLM을 훈련, 검증, 테스트 세트로 나누어 평가했습니다. 특히, 모델의 zero-shot, few-shot, fine-tuning 설정에서의 가설 생성 능력을 평가했고, 불확실성 탐사(unreliability exploration)를 강화하기 위해 도구 사용(tool use)과 멀티 에이전트 상호작용(multi-agent interactions)을 포함시켰습니다.

- **Performance Highlights**: [{'key': '1', 'finding': 'LLMs은 훈련 중 보지 못한 문헌을 기반으로 새로운 가설을 생성할 수 있음을 확인했습니다.'}, {'key': '2', 'finding': '멀티 에이전트 상호작용과 도구 사용을 통해 불확실성을 증가시키면 다양한 후보 가설 생성을 촉진하고 zero-shot 가설 생성 성능을 향상시킬 수 있습니다. 하지만 few-shot 학습과 도구 사용이 항상 성능 향상으로 이어지지는 않음을 관찰했습니다.'}]



### Self-Evolving GPT: A Lifelong Autonomous Experiential Learner (https://arxiv.org/abs/2407.08937)
Comments:
          Accepted by ACL 2024 MAIN

- **What's New**: 대형 언어 모델(LLMs)의 성능 향상을 위해, 최근 연구자들은 프롬프트(prompt)를 통해 LLMs에 텍스트 기반의 작업 해결 경험을 제공하는 방법을 탐구해왔습니다. 하지만 각 작업에 대한 경험을 수작업으로 획득하고 적용하는 것은 증가하는 LLMs의 수요와 다양한 사용자 질문에 비해 실질적이지 못합니다. 이를 해결하기 위해, 우리는 LLMs가 인간의 학습 및 경험 활용 능력을 모방할 수 있는지 탐구하는 자율적 경험 학습 프레임워크를 설계했습니다.

- **Technical Details**: 이 프레임워크는 경험 전이(Experience Transfer)와 유도를 통해 자율적으로 경험을 학습하고 축적하며, 입력 질문의 유형을 분류하여 축적된 경험 중 어느 것을 사용할지 선택합니다. LLMs가 자율적으로 학습하고 경험을 축적하며, 입력된 질문의 유형을 분류해 그에 적합한 경험을 활용합니다. 이를 통해 인간의 경험적 학습 및 응용 능력을 모방할 수 있는지에 대해 검증하였습니다.

- **Performance Highlights**: 여섯 개의 널리 사용되는 자연어 처리(NLP) 데이터셋을 대상으로 한 실험 결과, 우리의 프레임워크는 각 중간 단계에서 신뢰성 있게 작동하며, GPT-3.5와 GPT-4의 성능을 효과적으로 향상시켰습니다. 이는 LLMs를 이용한 인간 경험 학습 및 응용 능력 모방의 실현 가능성을 입증합니다. 또한, 각 단계에서 프레임워크의 동작에 대한 상세한 분석도 제공됩니다.



### Characterizing Prompt Compression Methods for Long Context Inferenc (https://arxiv.org/abs/2407.08892)
Comments:
          Es-FoMo @ ICML 2024

- **What's New**: 최근 몇 년간, 대형 언어 모델(LLM)의 사용이 폭발적으로 증가하면서 긴 문맥(context)을 처리하는 애플리케이션이 급증했습니다. 이를 위해 긴 문맥 추론을 처리하는 데 있어 여러 프롬프트 압축(prompt compression) 방법이 제안되었으나, 기존 연구는 이러한 방법들을 표준화되지 않은 방식으로 비교하여 모순된 결과를 초래했습니다. 이번 연구에서는 추출적 압축(extractive compression), 요약 기반의 추상적 압축(abstractive compression), 토큰 프루닝(token pruning) 방법을 종합적으로 분석하고 평가하였습니다.

- **Technical Details**: 연구팀은 프롬프트 압축 방법을 세 가지로 분류했습니다: 추출적 압축, 요약 기반 추상적 압축, 토큰 프루닝. 또한 각 방법을 쿼리 비의존적(query-agnostic) 또는 쿼리 의존적(query-aware)으로 구분했습니다. 세 가지 데이터셋에서는 단일 문서 질의응답(single-document QA), 다중 문서 질의응답(multi-document QA), 요약 작업을 통해 이를 평가했습니다. 연구 방법은 각 클래스에 대한 포괄적인 조사를 포함하며, 청크 크기, 쿼리 의존적 요약, 그리고 기타 중요한 선택 사항의 영향을 연구했습니다.

- **Performance Highlights**: 놀랍게도, 추출적 압축은 다른 접근 방식보다 뛰어난 성능을 보였으며, 적은 정확도 저하로 최대 10배의 압축을 가능하게 했습니다. 반면, 최근 여러 주장에도 불구하고 토큰 프루닝 방법은 추출적 압축보다 성능이 떨어졌음을 발견했습니다. 요약 작업에서는 미미한 개선만이 관찰되었습니다.



### Automatic Pruning of Fine-tuning Datasets for Transformer-based Language Models (https://arxiv.org/abs/2407.08887)
Comments:
          28 pages, 17 figures. Accepted at the Third Conference on Lifelong Learning Agents (CoLLAs 2024)

- **What's New**: 이번 연구에서는 파인튜닝(fine-tuning) 작업의 학습 셋(training set)을 자동으로 정제(pruning)하는 방법을 제안합니다. 제안된 방법은 모델이 각각의 학습 데이터 포인트를 정확하게 분류하는 성공률에 기반합니다. 기존 연구와 달리, 사용자 피드백을 통해 서브셋 크기를 결정하지 않으며, 모델과 파인튜닝 작업의 조합에 맞게 자동으로 학습 서브셋을 추출합니다.

- **Technical Details**: 제안된 방법은 모델의 성공률(success rate)을 활용하여 학습 셋을 자동으로 정제합니다. 모델과 파인튜닝 작업의 조합에 맞게 다양하게 크기를 조절할 수 있는 서브셋 여러 개를 제공합니다. 이를 통해 서브셋 크기와 평가 정확도 사이의 균형을 맞추는 것이 가능해집니다. 특히, 가장 큰 서브셋은 학습 성능을 크게 손상시키지 않으면서도 기존 학습 셋보다 평균 3배 더 작아 'winning ticket subset'으로 불립니다.

- **Performance Highlights**: 5개의 다운스트림(downstream) 작업과 2개의 언어 모델(language model) 실험 결과, 'winning ticket subset'을 사용한 파인튜닝이 모델 평가 성능을 평균 0.1% 향상시키는 것으로 나타났습니다.



### Evaluating Nuanced Bias in Large Language Model Free Response Answers (https://arxiv.org/abs/2407.08842)
Comments:
          14 pages, 0 figures, submitted to NLDB 2024, Turin, Italy

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 프리 텍스트 응답에서 발생하는 미묘한 편향을 평가하는 새로운 방법을 제시합니다. 이를 통해 기존의 단어 마스킹이나 다중 선택 질문을 통해서는 감지하기 어려운 '자신감 편향(confidence bias)', '암시적 편향(implied bias)', '포함 편향(inclusion bias)', '삭제 편향(erasure bias)' 등의 종류를 식별합니다.

- **Technical Details**: 새로운 평가 방법은 세 단계로 구성됩니다. 첫째로, 자동으로 편향이 없는 것으로 분류될 수 있는 응답을 제거합니다. 둘째로, 남은 응답들에 대해 크라우드 소싱을 통해 초기에 평가합니다. 마지막으로, 크라우드 소싱 결과를 전문가가 다시 검토합니다. 이 과정에서 이름을 순환하는 방식으로 편향을 평가합니다.

- **Performance Highlights**: 제안된 방법은 전문가 평가 대비 1/5의 시간이 소요되며, 편향 감지의 정확성을 높이는 데 유리합니다. 예를 들어, '모호한 문맥(ambiguous context)'과 '명확한 문맥(disambiguated context)'을 사용하여 BBQ 데이터셋의 질문에 대해 시스템이 응답하도록 했습니다. 이 방법을 통해 9가지 편향 범주에서 1048개의 자유 응답을 평가했습니다.



### Fault Diagnosis in Power Grids with Large Language Mod (https://arxiv.org/abs/2407.08836)
Comments:
          11 pages

- **What's New**: 이 논문은 전력망 고장 진단 분야에서 기존의 진단 시스템이 데이터의 복잡성과 가변성으로 인해 겪는 한계를 극복하기 위해 ChatGPT 및 GPT-4와 같은 대형 언어 모델(Large Language Models, LLMs)을 사용한 새로운 접근 방식을 제안합니다. 고급 프롬프트 엔지니어링을 통해 고장 진단의 정확도와 설명 가능성을 향상시키는 방법을 설계했습니다.

- **Technical Details**: 논문에서는 종합적이고 상황 인식적인 프롬프트를 설계하여 LLM이 복잡한 데이터를 해석하고 자세하고 실행 가능한 통찰력을 제공할 수 있도록 유도했습니다. 평가를 위해 새로운 데이터 세트를 구축했으며, 여기에는 실시간 센서 데이터, 과거 고장 기록, 그리고 구성 요소 설명이 포함되었습니다. 다양한 베이스라인 기술들과 비교하여 우리의 방법이 진단 정확도, 설명 품질, 응답 일관성, 및 맥락 이해에서 상당한 향상을 보였습니다.

- **Performance Highlights**: 실험 결과, GPT-4를 사용한 우리의 접근 방식은 진단 정확도와 생성된 설명의 품질에서 유의미한 개선을 나타냈습니다. 진단 정확도는 부분적 정확성을 반영하는 등급 점수로 측정되었으며, 설명의 명확성, 완전성, 관련성을 평가 항목으로 삼았습니다. 응답의 논리적 일관성과 이야기 흐름, 맥락 정보 활용 능력도 중요 평가 항목으로 추가되어 실용성을 높였습니다.



### Proving that Cryptic Crossword Clue Answers are Correc (https://arxiv.org/abs/2407.08824)
Comments:
          Accepted paper for the ICML 2024 Workshop on LLMs and Cognition (4 pages + references + 6 pages of Appendices)

- **What's New**: 수수께끼 같은 크로스워드 문제에 대한 LLM(대형 언어 모델) 기반의 새로운 풀이 및 검증 시스템이 발표되었습니다. 이 시스템은 Python 기반 증명을 통해 단어 퍼즐 주어진 정답의 적합성을 검증할 수 있습니다.

- **Technical Details**: 본 연구는 Llama-3-it 8B 모델과 Geminii-Pro-1.0-002, Gemini-Flash-1.5-001 LLM을 결합하여 정답과 거의 정답에 가까운 답변을 구별하는 시스템을 구성했습니다. Python 기반 증명(Proof)을 이용해 제시된 답변의 적합성을 검토합니다.

- **Performance Highlights**: 이 시스템은 cryptic crossword의 주어진 정의(definition)와 단어 퍼즐(wordplay) 주어진 정답의 적합성을 증명함으로써 정답을 효과적으로 구별할 수 있습니다. 이를 통해 기존의 LLM 기반 접근법보다 높은 정확도를 달성했습니다. 예를 들어, Cryptonite 테스트 세트에서 Rule-based solver의 8.6% 정확도를 능가하는 성능을 보여주었습니다.



### Rule-Based, Neural and LLM Back-Translation: Comparative Insights from a Variant of Ladin (https://arxiv.org/abs/2407.08819)
Comments:
          Accepted to LoResMT 2024 (ACL workshop)

- **What's New**: 이 논문은 Ladin 언어, 특히 Val Badia 변형을 위한 머신 번역에 대한 다양한 역번역(back-translation) 접근법의 영향을 탐구합니다. 이 연구는 병렬 데이터가 극히 제한된 저자원 언어 환경에서 수행되었습니다. 연구에서 Ladin-Italian 번역을 위한 다국어 뉴럴 머신 번역(NMT) 모델을 미세 조정(fine-tuning)했습니다. 이를 위해, 실제 데이터 외에도 세 가지 모델을 사용하여 추가 번역을 생성했습니다: 미세 조정된 신경망 모델, 특정 언어 쌍을 위한 규칙 기반 시스템(RBMT), 그리고 대규모 언어 모델(LLM).

- **Technical Details**: 데이터 수집은 Ladin 언어 자원들이 극히 제한됨에 따라 주로 La Usc di Ladins 신문과 사전에서 데이터를 추출하는 방식으로 이루어졌습니다. 실험에서는 미세 조정된 다국어 NMT 모델을 사용하여 Monolingual(단일언어) Ladin 데이터를 이탈리아어로 역번역했습니다. 세 가지 서로 다른 방식의 역번역을 비교 분석했습니다: (i) 병렬 데이터로 미세 조정된 NMT 시스템, (ii) 특정 언어 쌍을 위한 규칙 기반 메커니즘, (iii) 대규모 언어 모델(LLM). 이러한 데이터를 바탕으로 학습된 모델 성능을 BLEU/chrF++ 점수로 평가하였습니다.

- **Performance Highlights**: 실험 결과, 이 저자원 환경에서 모든 접근법이 동일한 수준의 번역 품질을 달성했습니다. 각 모델의 특성을 반영한 차이점이 존재했지만, 전반적인 성능에서는 큰 차이가 없었습니다. 특히 BLEU/chrF++ 점수에서 유의미한 차이를 보이지 않았습니다. 우리의 연구는 Ladin 언어에 대한 첫 번째 MT(Machine Translation) 시도이며, 향후 이 언어 쌍에 대한 더 많은 연구의 기초 자료가 될 것입니다.



### MAGNET: Improving the Multilingual Fairness of Language Models with Adaptive Gradient-Based Tokenization (https://arxiv.org/abs/2407.08818)
- **What's New**: 이번 연구에서는 비라틴(non-Latin) 문자와 저자원(low-resource) 언어에 대한 언어 모델의 유용성, 효율성, 비용 감소를 목표로 MAGNET을 제안합니다. MAGNET은 다양한 언어 스크립트별 예측기를 통해 서브워드 토크나이제이션(subword tokenization)의 과분할(over-segmentation) 문제를 완화합니다.

- **Technical Details**: MAGNET은 시퀀스 내 바이트 토큰(byte tokens) 간의 경계면을 예측하는 서브모듈을 사용합니다. 이러한 서브모듈은 내부 경계 예측기(tokenizers)로 작동합니다. 이는 기존의 단일 경계 예측기를 통합하고 확률적 재매개변수화를 통해 최적화하는 방식과 달리, 각 언어 스크립트에 최적화된 예측기를 사용해 모듈화된 방식으로 동작합니다. 이렇게 하면 다양한 언어 스크립트 간의 균일한 세분화를 강제할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 MAGNET은 세분화 불균형을 줄일 뿐만 아니라, 더 빠른 언어 모델링을 가능하게 하고 다운스트림 유틸리티도 개선되는 것을 보여주었습니다.



### Large Models of What? Mistaking Engineering Achievements for Human Linguistic Agency (https://arxiv.org/abs/2407.08790)
Comments:
          To appear in the Journal of Language Sciences

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 언어 능력에 관한 과장되고 오해를 불러일으킬 수 있는 주장들이 두 가지 근거 없는 가정에 기반한다는 점을 지적합니다. 첫째, 언어 완전성(language completeness) 가정은 '자연 언어'라는 명확하고 완전한 것이 존재하며, 이를 LLM이 효과적이고 포괄적으로 모델링할 수 있다는 것을 의미합니다. 둘째, 데이터 완전성(data completeness) 가정은 언어가 정량화되고 데이터로 완전히 포착될 수 있다는 신념에 의존합니다.

- **Technical Details**: 행동주의적 인지 과학(enactive cognitive science) 접근법에 따르면, 언어는 구체적이고 완전한 것이 아니라 행동의 일종입니다. 즉, '언어화(languaging)'는 포괄적으로 모델링 할 수 있는 종류가 아닙니다. 논문에서는 특히 '알고스피크(algospeak)'라는 최근에 기술된 패턴을 통해 인간의 언어 행위와 LLM의 활동을 비교하고, LLM들이 현재 형태로는 인간과 같은 언어적 에이전트가 될 수 없음을 설명합니다.

- **Performance Highlights**: 논문은 LLM과 인간의 언어 참여와 활동 사이에 상당한 차이점이 있다고 주장합니다. LLM은 통계적 모델과 경로 탐색 메커니즘을 통해 유효한 토큰 연속체를 생성하는 시스템이며, 이는 인간의 언어적 상호작용과 근본적으로 다릅니다. 이와 관련된 주요 차이점으로는 구현, 참여, 불안정성이 있으며, 이들이 LLM의 현재 아키텍처와 원칙적으로 호환될 수 없다고 합니다.



### Weight Block Sparsity: Training, Compilation, and AI Engine Accelerators (https://arxiv.org/abs/2407.09453)
Comments:
          12 pages, 10 figures, 1 table

- **What's New**: 이 연구는 점점 커지고 있는 딥 뉴럴 네트워크(DNN)의 추론 속도를 높이기 위해 하드웨어 친화적인 'Weight Block Sparsity' 방식을 도입했습니다. 이는 사전 학습된 DNN 모델의 컨볼루션 및 완전 연결층의 일부 파라미터를 0으로 만들어 메모리 사용량 감소, 통신 속도 향상, 연산 시간 감소를 이끌어냅니다.

- **Technical Details**: Weight Block Sparsity는 8x8 크기의 블록 구조로 제로 값을 도입하여 연산을 최적화합니다. 이를 통해 공간적, 시간적 지역성을 최대한 활용하여 벡터 연산 및 메모리 재사용을 극대화할 수 있습니다. 이 구조는 특히 AIE2 프로세서와 같은 AI 하드웨어 가속기에서 효율적으로 작동합니다. 연구에서는 Resnet50 모델을 사용하여 모델의 가중치를 절반으로 줄이면서도 성능 손실을 최소화하고, 추론 속도를 2배 빠르게 만들었습니다.

- **Performance Highlights**: 실제 실험 결과, Resnet50 모델에 Weight Block Sparsity를 적용하여 가중치를 절반으로 줄였으며, 성능 저하를 최소화하면서 추론 속도를 2배 향상시켰습니다. 또한, AMD Versal FPGA의 AIE2 프로세서 구성에서 코드 생성을 통해 성능 평가를 수행하여, 하드웨어 오버레이 디자인과 소프트웨어 스택 사이의 시너지를 보여주었습니다.



### Human-like Episodic Memory for Infinite Context LLMs (https://arxiv.org/abs/2407.09450)
- **What's New**: EM-LLM은 인간의 에피소드 기억과 이벤트 인지에서 영감을 받아 LLM의 문맥 처리 능력을 향상시키는 새로운 접근법입니다. 이를 통해 사실상 무한한 문맥 길이를 효율적으로 처리하면서도 계산 효율성을 유지할 수 있습니다. EM-LLM은 Bayesian surprise와 그래프 이론적 경계 정제를 결합하여 일관된 에피소드 이벤트로 토큰 시퀀스를 조직하고 필요할 때 두 단계의 메모리 프로세스를 통해 이러한 이벤트를 검색합니다. 이는 LLM의 성능을 크게 향상시킵니다.

- **Technical Details**: EM-LLM은 인간의 기억 형성 과정에서 영감을 받아 시퀀스를 개별 에피소드 이벤트로 세분화합니다. 이러한 이벤트의 경계는 모델의 추론 중 'surprise' 수준에 기반하여 동적으로 결정되며, 그래프 이론적 메트릭을 사용하여 수정됩니다. 유사성 기반 검색과 시간적으로 인접한 검색을 결합한 메모리 회상 메커니즘을 채택하여 LLM의 효율적인 정보 접근을 가능하게 합니다. 이러한 접근법은 최소한의 추가 계산 비용으로 이루어집니다.

- **Performance Highlights**: EM-LLM은 LongBench 데이터셋에서 state-of-the-art 모델인 InfLLM을 4.3% 상대적으로 향상시켰으며, PassageRetrieval 작업에서는 33%의 향상을 보였습니다. 이는 EM-LLM의 이벤트 세분화가 인간이 인식하는 이벤트와 강한 상관 관계가 있음을 보여줍니다. 또한, 이 방법은 장문 맥락 작업 처리에서 LLM의 성능을 크게 향상시키는 것으로 나타났습니다.



### Deep Bag-of-Words Model: An Efficient and Interpretable Relevance Architecture for Chinese E-Commerc (https://arxiv.org/abs/2407.09395)
Comments:
          KDD'24 accepted paper

- **What's New**: 새로운 연구로 Deep Bag-of-Words (DeepBoW) 모델이 제안되었습니다. 이 모델은 중국 전자상거래 플랫폼에서 쿼리와 제품 간의 텍스트 관련성을 효과적으로 측정하는 방법입니다.

- **Technical Details**: DeepBoW 모델은 쿼리와 제품을 단어 가중치 쌍(word-weight pairs)으로 이루어진 희소 BoW로 인코딩합니다. 이 모델은 고차원 표현을 통해 높은 중요도 단어를 파악하고, 관련성 점수는 일치하는 단어의 가중치 합산으로 측정됩니다. 이러한 구조는 기존의 밀집 임베딩(dense embedding) 방식보다 해석 가능성과 온라인 효율성을 높입니다.

- **Performance Highlights**: DeepBoW 모델은 세 개의 산업 데이터셋에서 벤치마크 모델과 비교하여 AUC 성능을 2.1% 이상 향상시켰습니다. 이는 특히 타오바오(Taobao)와 같은 대규모 전자상거래 플랫폼에서 실시간 검색에 성공적으로 적용되고 있습니다.



### Sina at FigNews 2024: Multilingual Datasets Annotated with Bias and Propaganda (https://arxiv.org/abs/2407.09327)
- **What's New**: 이 논문은 이스라엘-가자 전쟁에 대한 뉴스 미디어 내러티브를 다룬 FigNews 2024 공유 작업의 일환으로 만들어진 다국어 코퍼스(corpus)에 대해 설명합니다. 이 코퍼스는 12,000개의 페이스북 게시물로 구성되어 있으며, 모든 게시물은 편향(bias)과 프로파간다(propaganda)에 대해 주석이 달려 있습니다. 5개 언어(아랍어, 히브리어, 영어, 프랑스어, 힌디어)로 작성된 게시물이 포함되어 있으며, 각 언어별로 2,400개의 게시물이 있습니다.

- **Technical Details**: 이 코퍼스는 10명의 법학 석사 과정 학생들에 의해 주석이 달렸습니다. 주석의 품질을 평가하기 위해 Inter-Annotator Agreement (IAA)를 사용했으며, 편향 주석의 평균 IAA는 80.8%, 프로파간다 주석의 평균 IAA는 70.15%였습니다. 편향의 경우 '팔레스타인에 대한 편향', '이스라엘에 대한 편향', '기타 대상에 대한 편향' 등으로 분류되고, 프로파간다는 '삭제해야 할 프로파간다', '삭제할 수도 있는 프로파간다', '삭제하지 않을 프로파간다'로 분류됩니다.

- **Performance Highlights**: 우리 팀은 편향과 프로파간다 하위 작업에서 가장 우수한 성능을 보이는 팀 중 하나로 평가받았습니다. 또한, 이 코퍼스는 오픈 소스로 공개되어 연구 커뮤니티에서 자유롭게 사용할 수 있습니다.



### A Chatbot for Asylum-Seeking Migrants in Europ (https://arxiv.org/abs/2407.09197)
- **What's New**: 이번 논문에서는 유럽에 거주하는 난민 신청자를 위한 챗봇 ACME를 소개합니다. ACME는 컴퓨팅 논증 (computational argumentation)에 의존하여 난민이 신청할 수 있는 최고 수준의 보호를 식별하는 데 도움을 줍니다. 이를 통해 영토 위원회, 법원 및 인도주의 단체의 부담을 줄여 보다 지속 가능한 이주를 지원할 수 있습니다.

- **Technical Details**: ACME의 아키텍처는 '뉴로-심볼릭(neuro-symbolic)' 접근 방식을 채택하여, 사용자 이해를 담당하는 뉴럴 모듈과 추론을 담당하는 심볼릭 모듈을 결합합니다. 주요 구성 요소는 텍스트 파일로 구성된 지식 베이스 (KB), 사용자의 입력을 이해하는 뉴럴 자연어 모듈, 그리고 데이터를 기반으로 추론하고 설명을 제공하는 논증 모듈입니다. 사용자의 정보는 KB의 개념으로 맵핑되며, 이에 대해 논증 모듈이 적절한 답변을 제공합니다.

- **Performance Highlights**: ACME 챗봇은 모듈러 아키텍처를 통해 데이터 거버넌스 및 프라이버시를 보존하며, 논증적 추론 방식을 사용하여 투명성과 설명 가능성을 보장합니다. 전문가가 만든 공식화된 지식 베이스와 추론 방법을 사용하여 완전히 검토 가능하며, 사용자의 프라이버시를 보호하는 동시에 법적 투명성과 감시 가능성을 충족합니다.



### The Two Sides of the Coin: Hallucination Generation and Detection with LLMs as Evaluators for LLMs (https://arxiv.org/abs/2407.09152)
Comments:
          Paper accepted at ELOQUENT@CLEF'24

- **What's New**: 최근 CLEF ELOQUENT HalluciGen 공유 작업(Task)에서 대형 언어 모델(LLMs)의 환각(hallucination) 검출 기법에 대한 새로운 연구가 발표되었습니다. 이 연구는 특히 환각된 콘텐츠를 생성하고 탐지하는 평가자 개발에 중점을 둡니다.

- **Technical Details**: 본 연구에서는 Llama 3, Gemma, GPT-3.5 Turbo, 그리고 GPT-4 네 가지 대형 언어 모델을 활용하여 환각을 감지하고 탐지하는 작업을 수행했습니다. 네 모델의 성능을 통합하기 위해 앙상블 다수결 투표(ensemble majority voting) 방법이 사용되었습니다. 이 방법을 통해 각 모델의 개별적인 판단을 모아서 최종 결정을 내리는 방식을 취했습니다.

- **Performance Highlights**: 연구 결과는 이 네 가지 대형 언어 모델이 환각 생성 및 탐지 작업에서 보여주는 강점과 약점을 잘 보여주었습니다. 이를 통해 각 모델의 능력 범위를 이해하고, 향후 연구와 개발에서 개선 방향을 모색할 수 있는 중요한 통찰을 제공했습니다.



### A Look Into News Avoidance Through AWRS: An Avoidance-Aware Recommender System (https://arxiv.org/abs/2407.09137)
- **What's New**: 최근 몇 년간 언론인들은 특정 분야에서 뉴스 기사 회피 경향이 증가하고 있다는 우려를 표명해왔습니다. 이 문제는 추천 시스템의 발달로 더욱 악화되었습니다. 이번 연구는 추천 시스템이 회피 요소를 기본 요인으로 고려해야 한다고 주장하며, 뉴스 기사가 노출, 관련성, 회피라는 세 가지 주요 요소로 특성화될 수 있음을 제안합니다. 이를 해결하기 위해, 우리는 회피 인식을 포함한 뉴스 추천 프레임워크인 AWRS(Avoidance-Aware Recommender System)를 도입하였습니다. 세 개의 다른 언어로 된 뉴스 데이터셋(영어, 노르웨이어, 일본어)에 대한 평가 결과, 우리의 방법이 기존 접근법보다 우수한 성능을 나타냈습니다.

- **Technical Details**: AWRS는 필터 버블(filter bubbles)을 생성하는 기존의 네트워크 기반 모델들에 대한 대안으로 설계되었습니다. 뉴스 회피 행동을 추천 시스템에 통합하고자 하며, 이는 뉴스 소비와 관련된 행동을 더 잘 이해할 수 있게 합니다. AWRS는 사용자가 회피하는 기사를 추천 목록에서 제외시키거나 덜 노출시키는 기능을 통해 개인 맞춤화된 뉴스 추천을 개선합니다. 대표적인 기존 모델(NRMS, NAML, LSTUR, GLORY, LANCER 등)들과의 비교 분석을 통해 주요 차별점을 설명합니다.

- **Performance Highlights**: AWRS는 세 가지 다양한 실세계 데이터셋에서 기존 방법들보다 뛰어난 성능을 보여주었습니다. 회피 요소를 포함함으로써 뉴스 기사 추천의 정확성과 관련성을 향상시켰으며, 특히 노출이 많고 덜 회피되는 기사보다 많이 회피되는 기사가 사용자 선호도를 더 잘 반영하는 경우가 많다는 점을 강조합니다. 이러한 접근법은 사용자 참여도를 높이고 뉴스 기사 추천의 효과를 증대시켰습니다.



### URRL-IMVC: Unified and Robust Representation Learning for Incomplete Multi-View Clustering (https://arxiv.org/abs/2407.09120)
Comments:
          Accepted by ACM SIGKDD 2024

- **What's New**: 불완전한 멀티뷰 클러스터링(Incomplete Multi-View Clustering, IMVC) 분야에서 다수의 새로운 접근 방법이 제안되었습니다. 이번 연구에서는 통합되고 견고한 표현 학습을 통해 IMVC 문제를 해결하기 위한 새로운 프레임워크인 URRL-IMVC를 소개합니다. 이 프레임워크는 다수의 뷰 간의 정보 융합과 이웃 샘플들의 정보를 통합하여 불완전한 데이터를 처리합니다.

- **Technical Details**: URRL-IMVC는 Cross-View Contrastive Learning 및 Missing View Recovery의 한계를 극복하기 위해 설계되었습니다. 먼저, 주의(attention) 기반 오토인코더(auto-encoder)를 사용하여 멀티뷰 데이터를 융합하고 통합 임베딩을 생성합니다. 두 번째로, KNN 임퓨테이션(imputation) 및 데이터 증강(data augmentation) 기법을 통해 뷰 누락 조건에서 통합 임베딩의 견고성을 직접 강화합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 대상으로 평가 시, URRL-IMVC는 최신의 성능을 보여줍니다. Comprehensive Ablation Studies를 통해 이 프레임워크의 설계가 효과적인지 검증하였습니다.



### IDAT: A Multi-Modal Dataset and Toolkit for Building and Evaluating Interactive Task-Solving Agents (https://arxiv.org/abs/2407.08898)
- **What's New**: 이 연구는 NeurIPS의 IGLU 경진대회를 통해 AI 에이전트와 인간 간의 자연스러운 상호작용을 목표로 합니다. 연구 팀은 Minecraft 유사 환경에서 상호작용하는 언어 지시를 수집하기 위한 확장 가능한 데이터 수집 도구를 도입하고, 인간 중심의 평가 플랫폼을 제안했습니다. IDAT (IGLU Dataset And Toolkit)으로 알려진 이 도구와 데이터셋은 커뮤니티의 연구를 더 발전시키기 위한 중요한 자원으로 공개됩니다.

- **Technical Details**: 데이터 수집 도구는 JavaScript로 개발되어 Minecraft 게임 서버의 설정 없이 웹 브라우저에서 실행 가능합니다. 이는 대규모의 데이터 수집을 용이하게 하고 크라우드소싱 플랫폼 (예: Amazon MTurk)과의 통합을 통해 데이터 수집을 확장할 수 있습니다. 또한, CraftAssist voxel world 환경을 사용하여 다양한 물리적 특성과 3D 세계 표현을 통해 언어 지시에서 학습하고 건설 작업을 수행할 수 있습니다.

- **Performance Highlights**: 9,000개 이상의 발화와 1,000개 이상의 명확화 질문을 포함한 멀티모달 (Multi-Modal) 데이터셋이 포함되어 있습니다. 또한, 인간 중심의 상호작용 평가 플랫폼은 강화 학습 (Reinforcement Learning) 에이전트를 여러 사람과 비교하여 질적 분석을 제공합니다. 이 새로운 도구와 데이터셋은 55개 이상의 팀이 경진대회에서 활용하였으며, 이는 지능형 상호작용 에이전트 개발 연구에 유용함을 강조합니다.



### GPT-4 is judged more human than humans in displaced and inverted Turing tests (https://arxiv.org/abs/2407.08853)
- **What's New**: 최근 연구는 매일 일어나는 온라인 대화에서 사람과 AI를 구별하는 문제를 탐구합니다. 일반 사용자들이 AI와 직접 상호작용하기보다는 AI와 다른 사람 간의 대화를 읽게 되는 경우가 많기 때문에, 이 연구는 거꾸로 및 이동형 튜링 테스트(modified Turing test)를 사용해 사람들이 얼마나 잘 구별할 수 있는지 측정했습니다.

- **Technical Details**: 이 연구는 GPT-3.5, GPT-4와 이동형 인간 심판들을 대상으로 튜링 테스트 녹취록을 사용하여 인간인지 AI인지 판별하는 능력을 평가했습니다. 연구 결과에 따르면, AI와 인간 판결자 모두 직접 인터랙티브하게 질문하는 사람들보다 정확도가 낮았으며, 오히려 AI가 약간 더 사람으로 보였습니다. 이는 사람과 현재의 대형 언어 모델(LLM) 모두 실제 상호작용 없이 사람과 AI를 구별하기 어렵다는 점을 시사합니다.

- **Performance Highlights**: AI와 이동형 인간 판결자 모두 전반적으로 정확도를 밑돌았습니다. 특히 GPT-4가 가장 성능이 우수한 AI 증인으로 사람보다 종종 더 인간적으로 보였습니다. 다양한 변형 튜링 테스트를 통해 인간 판별자와 AI 판별자 모두가 대화형 환경에서 AI를 탐지하는 데 어려움을 겪고 있음을 보여주고 있습니다.



### ModelWriter: Text & Model-Synchronized Document Engineering Platform (https://arxiv.org/abs/2403.01359)
Comments:
          Published in: 2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)

- **What's New**: ModelWriter 플랫폼은 기술 문서의 일관성과 완전성을 추적할 수 있는 일반적인 프레임워크를 제공합니다. 특히 이 플랫폼은 Airbus에서 사용하는 System Installation Design Principles (SIDP)의 추적 가능성을 입증합니다. 여기에는 자연어 처리(NLP)와 자동 추론 기법을 통합하여 텍스트의 의미와 구조에 대한 추론을 모두 수행합니다.

- **Technical Details**: ModelWriter 플랫폼은 텍스트 및 모델 아티팩트에 대한 추적 가능성 분석을 지원합니다. 추적 위치는 텍스트 조각, 아키텍처 모델의 요소 및 프로그램 코드의 일부일 수 있습니다. 플랫폼은 이러한 관계의 공리화와 추론을 허용하여 다양한 유형의 아티팩트에 대한 추적 가능성 분석을 지원합니다. SIDP가 포함(contains), 상세화(refines), 충돌(conflicts), 동일(equals), 필요(requires) 등 5가지 유형의 추적 링크를 정의합니다.

- **Performance Highlights**: ModelWriter 플랫폼은 주어진 SIDP의 완전성과 일관성을 검사할 수 있습니다. SIDP는 처음에 설명 논리(Description Logic) 공식으로 파싱됩니다. 추적 링크는 사용자가 수동으로 지정하거나 의미론적 파싱 및 DL 정리 증명(Theorem Proving)을 통해 추론할 수 있습니다. 추가 추적 링크는 관계 논리(Relational Logic)와 모델 찾기(Model Finding)를 사용하여 추론됩니다. 예를 들어, 'hydraulic area'와 'fuel tank'와 같은 불일치 개념을 포함하는 문장이 충돌로 식별될 수 있습니다.



New uploads on arXiv(cs.IR)

### Deep Bag-of-Words Model: An Efficient and Interpretable Relevance Architecture for Chinese E-Commerc (https://arxiv.org/abs/2407.09395)
Comments:
          KDD'24 accepted paper

- **What's New**: 새로운 연구로 Deep Bag-of-Words (DeepBoW) 모델이 제안되었습니다. 이 모델은 중국 전자상거래 플랫폼에서 쿼리와 제품 간의 텍스트 관련성을 효과적으로 측정하는 방법입니다.

- **Technical Details**: DeepBoW 모델은 쿼리와 제품을 단어 가중치 쌍(word-weight pairs)으로 이루어진 희소 BoW로 인코딩합니다. 이 모델은 고차원 표현을 통해 높은 중요도 단어를 파악하고, 관련성 점수는 일치하는 단어의 가중치 합산으로 측정됩니다. 이러한 구조는 기존의 밀집 임베딩(dense embedding) 방식보다 해석 가능성과 온라인 효율성을 높입니다.

- **Performance Highlights**: DeepBoW 모델은 세 개의 산업 데이터셋에서 벤치마크 모델과 비교하여 AUC 성능을 2.1% 이상 향상시켰습니다. 이는 특히 타오바오(Taobao)와 같은 대규모 전자상거래 플랫폼에서 실시간 검색에 성공적으로 적용되고 있습니다.



### PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents (https://arxiv.org/abs/2407.09394)
- **What's New**: 대형 언어 모델(LLMs)은 오래된 지식과 환각(hallucinations)으로 인해 신뢰할 수 있는 출력을 생성하는 데 어려움을 겪습니다. 이에 대한 해결책으로, Retrieval-Augmented Generation (RAG) 모델이 외부 지식을 통해 LLMs를 보강하지만, 종종 개인화된 검색 과정을 실패합니다. 이 논문은 실시간 사용자 데이터와 상호작용을 기반으로 검색과 생성을 적응시키는 사용자 중심 에이전트를 포함하는 새로운 프레임워크인 PersonaRAG를 소개합니다.

- **Technical Details**: PersonaRAG는 실시간 사용자 데이터를 활용하여 검색 및 텍스트 생성 과정을 최적화합니다. 이는 사용자별 선호도와 요구 사항에 맞춘 답변을 제공할 수 있도록 해 줍니다. 기존 RAG 모델과 달리, PersonaRAG는 사용자 상호작용을 실시간으로 파악하고 모델의 검색 및 생성 메커니즘을 조정합니다.

- **Performance Highlights**: 여러 질문 응답 데이터셋에서 평가된 결과, PersonaRAG는 기본 모델들에 비해 뛰어난 성능을 보여줬으며, 사용자 요구에 맞춘 맞춤형 답변을 제공하는 데 있어서 우수함을 발휘했습니다. 이러한 결과는 사용자 적응 정보 검색 시스템에 대한 유망한 방향을 제시합니다.



### Movie Recommendation with Poster Attention via Multi-modal Transformer Feature Fusion (https://arxiv.org/abs/2407.09157)
- **What's New**: 이번 연구는 영화 추천 시스템에서 멀티모달 데이터(multi-modal data)를 통합하는 혁신적인 방식을 제안합니다. 이 시스템은 영화의 포스터 이미지와 텍스트 설명을 통합하여 사용자 선호를 예측합니다.

- **Technical Details**: 텍스트 모달리티(modality) 정보를 추출하기 위해 BERT 모델을, 포스터/이미지 모달리티 정보를 추출하기 위해 ViT 모델을 사용했습니다. 그리고 Transformer 아키텍처를 활용해 모든 모달리티의 특징을 융합하여 사용자 선호를 예측합니다.

- **Performance Highlights**: MovieLens 100K 및 1M 데이터셋을 사용한 실험에서, 제안된 멀티모달 알고리즘이 기존의 알고리즘보다 더 높은 예측 정확도를 보여주었습니다.



### A Look Into News Avoidance Through AWRS: An Avoidance-Aware Recommender System (https://arxiv.org/abs/2407.09137)
- **What's New**: 최근 몇 년간 언론인들은 특정 분야에서 뉴스 기사 회피 경향이 증가하고 있다는 우려를 표명해왔습니다. 이 문제는 추천 시스템의 발달로 더욱 악화되었습니다. 이번 연구는 추천 시스템이 회피 요소를 기본 요인으로 고려해야 한다고 주장하며, 뉴스 기사가 노출, 관련성, 회피라는 세 가지 주요 요소로 특성화될 수 있음을 제안합니다. 이를 해결하기 위해, 우리는 회피 인식을 포함한 뉴스 추천 프레임워크인 AWRS(Avoidance-Aware Recommender System)를 도입하였습니다. 세 개의 다른 언어로 된 뉴스 데이터셋(영어, 노르웨이어, 일본어)에 대한 평가 결과, 우리의 방법이 기존 접근법보다 우수한 성능을 나타냈습니다.

- **Technical Details**: AWRS는 필터 버블(filter bubbles)을 생성하는 기존의 네트워크 기반 모델들에 대한 대안으로 설계되었습니다. 뉴스 회피 행동을 추천 시스템에 통합하고자 하며, 이는 뉴스 소비와 관련된 행동을 더 잘 이해할 수 있게 합니다. AWRS는 사용자가 회피하는 기사를 추천 목록에서 제외시키거나 덜 노출시키는 기능을 통해 개인 맞춤화된 뉴스 추천을 개선합니다. 대표적인 기존 모델(NRMS, NAML, LSTUR, GLORY, LANCER 등)들과의 비교 분석을 통해 주요 차별점을 설명합니다.

- **Performance Highlights**: AWRS는 세 가지 다양한 실세계 데이터셋에서 기존 방법들보다 뛰어난 성능을 보여주었습니다. 회피 요소를 포함함으로써 뉴스 기사 추천의 정확성과 관련성을 향상시켰으며, 특히 노출이 많고 덜 회피되는 기사보다 많이 회피되는 기사가 사용자 선호도를 더 잘 반영하는 경우가 많다는 점을 강조합니다. 이러한 접근법은 사용자 참여도를 높이고 뉴스 기사 추천의 효과를 증대시켰습니다.



### Multi-Modal Dataset Creation for Federated~Learning with DICOM Structured Reports (https://arxiv.org/abs/2407.09064)
- **What's New**: 새로운 연구는 데이터 조화를 통해 다중 병원에서 연합 학습(Federated Learning)을 용이하게 하는 멀티 모달 데이터 셋을 구성하는 방법을 제시합니다. 이를 위해 연구자들은 DICOM 구조화된 보고서(Structured Report, SR)를 사용하여 다양한 데이터 유형을 통합, 일치, 필터링할 수 있는 플랫폼을 개발했습니다.

- **Technical Details**: 연구팀은 Python 기반 딥러닝 파이프라인에서 highdicom을 사용하여 DICOM 구조화된 보고서를 통해 이미지 도메인을 뛰어넘는 정보를 표준화된 방식으로 연결했습니다. 이 새로운 플랫폼은 8개의 독일 대학 병원 컨소시엄 내에서 연합 학습을 위한 데이터 셋을 조율하는 데 사용되었습니다. 또한, Opensearch를 기반으로 하는 그래픽 필터링 도구가 제공됩니다.

- **Performance Highlights**: 이 연구는 구조화된 보고서를 효율적으로 사용하여 최소 침습 심장판막 교체 후의 결과를 예측하는 멀티 모달 데이터 셋을 조화롭게 구성하고 필터링할 수 있음을 입증했습니다. 이 데이터 셋에는 컴퓨터 단층 촬영(Computed Tomography) 이미지, 심전도(Electrocardiography) 스캔, 석회화 영역 세분화(Calcification Segmentations), 점 집합(Pointsets), 인공 심장박동기(Implantable Pulse Generator) 의존성에 대한 주석 등이 포함됩니다.



### Time-Frequency Analysis of Variable-Length WiFi CSI Signals for Person Re-Identification (https://arxiv.org/abs/2407.09045)
- **What's New**: WiFi CSI 기반 인물 재식별(ReID) 기술을 제안합니다. 현재 보안 및 감시 시스템은 시각 정보에 크게 의존하는 반면, 개인 프라이버시 침해와 외모에 의한 간섭 문제를 해결할 수 있는 새로운 접근법입니다.

- **Technical Details**: WiFi CSI 데이터를 통한 다중 경로 전파 특성을 활용하여 서로 다른 보행자 특징을 구별하는 방법을 제안합니다. 가변 길이 데이터 처리가 가능한 두 개의 스트림 네트워크 구조를 제안하고 WiFi 신호의 시간 도메인에서는 진폭을, 주파수 도메인에서는 위상을 분석합니다. 지속적인 측면 연결을 통해 시간-주파수 정보를 융합하고, 고급 목표 함수를 사용해 표현 및 메트릭 학습을 수행합니다.

- **Performance Highlights**: 실제 데이터셋에서 테스트 결과, 제안한 방법은 93.68%의 mAP와 98.13%의 Rank-1을 달성했습니다. 이는 비전 기반 방법보다 높은 성능을 제공하며, 프라이버시를 보호하면서 효과적인 인물 모니터링 및 관리가 가능합니다.



### A Neural Matrix Decomposition Recommender System Model based on the Multimodal Large Language Mod (https://arxiv.org/abs/2407.08942)
- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델인 BoNMF를 기반으로 한 뉴럴 매트릭스 요인화 추천 시스템 모델을 제안합니다. 이 모델은 자연어 처리에서 강력한 BoBERTa, 컴퓨터 비전 분야의 ViT, 그리고 신경 매트릭스 분해 기술을 결합합니다.

- **Technical Details**: BoNMF 모델은 사용자와 아이템의 잠재적 특성을 포착하고, 사용자 및 아이템 ID로 구성된 저차원 매트릭스와 상호작용한 후, 신경망이 결과를 출력하는 구조입니다. BoBERTa는 자연어 처리(NLP)에 강력한 능력을 제공하고, ViT는 컴퓨터 비전(CV)에 사용됩니다. 신경 매트릭스 분해(Neural Matrix Decomposition) 기술이 결합되어 있습니다.

- **Performance Highlights**: 콜드 스타트(Cold Start)와 절제 실험(Ablation Experiment) 결과, BoNMF 모델은 대규모 공개 데이터 세트에서 우수한 성능을 보였으며, 추천 정확도를 크게 향상시켰습니다.



### Toward Automatic Group Membership Annotation for Group Fairness Evaluation (https://arxiv.org/abs/2407.08926)
- **What's New**: 이 연구는 그룹 공정성 평가(group fairness evaluation)를 위해 인공지능 언어 모델을 이용한 자동 그룹 멤버십(group membership; GM) 주석 방법을 탐구하여, 인간 주석의 필요성과 관련한 문제를 해결하는 새로운 접근법을 제시합니다. 이를 통해 공정성 인식 알고리즘 연구에서 데이터 스파시티(data sparsity)를 극복하고, 여러 데이터셋에서 효율적인 공정성 평가를 가능하게 합니다.

- **Technical Details**: BERT 기반 모델이 GPT 및 Mistral과 같은 최신의 대형 언어 모델(Large Language Models; LLM)보다 우수한 성능을 나타냈습니다. 특히 최소한의 감독(supervision) 하에서 GM 주석의 정확도가 높게 평가되었습니다. 이는 BERT가 텍스트의 언어적 및 의미적 정보를 정확하게 포착하는 데 강점을 보이는 반면, 거대한 매개변수(parameter)를 가진 LLM들은 경제적 및 계산 비용이 많이 들어가고, 세밀한 프롬프트 설계가 필요하다는 점을 반영합니다.

- **Performance Highlights**: 실험 결과는 BERT 기반 모델이 최소한의 감독 하에 높은 주석 정확도를 달성했음을 보여줍니다. 특히, 문서 수준의 오류가 집계레벨에서 제거될 수 있으며, 이는 공정성 평가의 유효성과 견고성을 유지할 수 있다는 것을 시사합니다. 이 새로운 GM 주석 방법은 인간의 노력과 비용을 크게 줄여주며, 더욱 많은 데이터셋에서 공정성 인식 연구를 확장할 수 있도록 합니다.



### Mitigating Entity-Level Hallucination in Large Language Models (https://arxiv.org/abs/2407.09417)
- **What's New**: 최근 큰 언어 모델(LLMs)은 검색 엔진을 통한 전통적인 정보 접근 방식에서 사용자와의 직접적인 질문 및 답변 상호작용으로 변화를 가져왔습니다. 그러나, 이러한 LLMs의 광범위한 채택은 일관되나 사실적으로 부정확한 응답을 생성하는 '환각(hallucination)' 문제를 드러냈습니다. 이를 해결하기 위해, 본 논문에서는 환각 감지를 기반으로 한 동적 검색 보강(DRAD)을 제안합니다. DRAD는 실시간 환각 감지와 외부 지식을 기반으로 한 자기 수정 기능을 통합하여 환각 문제를 탐지하고 경감합니다.

- **Technical Details**: DRAD는 두 가지 주요 구성 요소로 구성됩니다. 첫 번째는 Real-time Hallucination Detection(RHD)로, 외부 모델 없이 실시간으로 잠재적인 환각을 식별합니다. 두 번째는 Self-correction based on External Knowledge(SEK)로, 외부 지식을 사용하여 이러한 오류를 수정합니다. RHD는 LLM의 출력 엔티티의 불확실성을 분석하여 잠재적인 환각을 감지하고, SEK는 외부 지식을 검색하여 LLM의 출력을 수정함으로써 환각을 방지합니다.

- **Performance Highlights**: 실험 결과, DRAD는 환각 감지 및 경감에서 뛰어난 성능을 보여줍니다. 특히 DRAD는 기존의 단일 라운드 및 다중 라운드 검색 보강 방법을 능가하는 성과를 보였습니다. 여러 복잡한 QA 벤치마크 데이터셋에서 DRAD는 큰 모델에서의 환각을 크게 줄이며 우수한 성능을 입증하였습니다.



### Context Embeddings for Efficient Answer Generation in RAG (https://arxiv.org/abs/2407.09252)
Comments:
          10 pages

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG)의 효율성을 크게 향상시킬 수 있는 새로운 컨텍스트 압축 방법인 COCOM (COntext COmpression Model)을 제안합니다. 이 방법은 긴 컨텍스트를 소수의 컨텍스트 임베딩(Context Embeddings)으로 압축하여, 생성 시간을 크게 단축하면서도 답변의 품질을 유지 또는 향상시킵니다. COCOM은 다양한 압축율을 제공하여, 생성 품질과 디코딩 시간을 조절할 수 있습니다.

- **Technical Details**: COCOM은 여러 문서를 보다 효과적으로 처리할 수 있도록 설계되었으며, 이는 긴 입력에 대한 디코딩 시간을 기존 방법보다 크게 줄입니다. 이 방법은 넓은 표면 형태의 입력을 소수의 임베딩으로 변환하여, 디코딩 시간 동안의 성능을 크게 향상시킵니다. 특히 COCOM에서는 프리트레이닝과 튜닝 접근법을 통해 컨텍스트 압축의 효과를 극대화하였습니다. 기존의 임베딩 기반 압축 방법들과 비교했을 때, COCOM은 더 높은 효율성과 성능을 보입니다.

- **Performance Highlights**: COCOM의 효율성 연구에서는 최대 5.69배의 인퍼런스 시간 단축과 최대 22배의 GFLOP 감소를 달성하면서도 높은 성능을 유지하는 것을 보여줍니다. 특히, 여러 실험을 통해 사전 학습 컬렉션, 프리트레이닝, 파인튜닝, 그리고 디코더의 동결 또는 비동결이 모델 성능에 미치는 영향을 분석하였습니다. 이러한 분석을 통해 COCOM이 다른 압축 방법들보다 우수한 성과를 보이는 것을 입증하였습니다.



### AI-Powered Immersive Assistance for Interactive Task Execution in Industrial Environments (https://arxiv.org/abs/2407.09147)
Comments:
          3 pages, 2 figures, Demo Paper accepted at the 50th European Conference on Artificial Intelligence

- **What's New**: 이 논문은 AI 기반 몰입형 지원 시스템을 소개합니다. 이 시스템은 산업 환경에서 복잡한 작업을 수행하는 데 도움을 주는 가상 현실(Virtual Reality, VR) 환경을 활용합니다. 특히, 물리적인 설비와 유사한 디지털 트윈(digital twin)으로 주스 믹서 설정을 시뮬레이션하는 VR 환경을 사용합니다. 이 시스템은 작업 중 전문가의 비디오와 오디오 녹음을 처리하여 단계별 가이드를 제공하는 대형 언어 모델(LLM)과 음성-텍스트 모델(speech-to-text model)을 핵심 구성 요소로 사용합니다.

- **Technical Details**: 시스템은 Unity, Oculus VR, Meta Quest 를 사용하여 개발되었습니다. 주스 믹서 디지털 트윈은 제약 및 화학 도메인에서 사용되는 기계를 본떠 설계되었습니다. 사용자는 주스 믹싱 과정을 준비, 조립, 믹싱, 최종 단계의 순서로 따라가면서 인터랙티브하게 작업을 수행할 수 있습니다. AI 도우미는 전문가의 비디오 녹화를 기반으로 생성한 음성 텍스트 트랜스크립트를 사용하여 학습 콘텐츠를 만듭니다.

- **Performance Highlights**: 시스템은 사용자에게 복잡한 기계 조작을 안전하고 몰입감 있는 가상 환경에서 마스터할 수 있게 해줍니다. 이를 통해 작업자의 인지 부하를 줄이고, 생산성을 높이며, 안전성을 향상시킬 수 있습니다. 이 AI 도우미는 실시간 의사결정을 지원하며, 전문가의 직접적인 도움을 받을 수 없는 경우에도 자율적인 가이던스를 제공합니다.



### Distinct citation distributions complicate research evaluations. A single indicator that universally reveals research efficiency cannot be formulated (https://arxiv.org/abs/2407.09138)
Comments:
          30 pages, 6 figures, 7 tables

- **What's New**: 이 연구는 다양한 연구 주제에 대한 출판물의 인용 분포의 다양성을 분석하여, 규모에 독립적이며 순위 기반 지표의 정확성을 조사했습니다. 그러한 지표 중에서는 상위 퍼센타일 지표가 가장 일반적이지만, 특히 일본의 평가에서 명백한 오판이 나타났다는 점을 중심으로 보고 있습니다.

- **Technical Details**: 연구는 여러 연구 주제에서 국가 및 저널에서 출판된 논문의 인용 분포를 분석하기 위해 히스토그램(histograms) 비롯한 로그 빈닝(logarithmic binning), 더블 랭크 플롯(double rank plots), 로그 변환된 인용 숫자의 정상 확률 플롯(normal probability plots)을 사용했습니다. 이 방법을 통해 각 주제에서의 전 세계 출판물과의 비교를 시도했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 글로벌 랭크가 파워 법칙(power law)을 따를 때 규모에 독립적인 상위 퍼센타일 기반 지표는 정확하지만, 인용되지 않는 논문의 비율이 높은 경우에는 오차가 존재할 수 있습니다. 이는 특히 고임팩트 팩터 저널에서 자주 발생합니다. 이러한 상황에서는 단일 지표가 오해를 초래할 수 있으며, 인용되지 않은 논문의 비율 비교가 이러한 오차를 예측하는 가장 좋은 방법임을 발견했습니다.

- **Practical Implications**: OECD, 유럽연합 위원회, 미 국립 과학 위원회 등 존경받는 기관들이 여러 국가의 연구 순위를 매기고 있지만, 규모에 독립적인 퍼센타일 지표는 여러 국가에서 오판을 초래할 수 있습니다. 이러한 오판은 연구 정책 입안자에게 혼란을 주고 잘못된 연구 정책으로 이어질 수 있기에, 이러한 잘못된 평가를 중단할 필요가 있습니다.



### AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security (https://arxiv.org/abs/2407.09017)
- **What's New**: Microsoft가 Copilot Guided Response (CGR)라는 새로운 머신러닝 아키텍처를 개발하여 보안 분석가들이 보안 사건에 대한 조사, 분류, 그리고 대응을 신속하고 정확하게 수행할 수 있도록 지원합니다. CGR은 Microsoft Defender XDR 제품에 통합되어 전 세계적으로 배포되고 있으며, 수백 고객사에서 이미 성공적으로 사용되고 있습니다.

- **Technical Details**: CGR 아키텍처는 세 가지 주요 기능을 포함합니다: (1) 조사(Inivestigation), 유사한 사건을 식별하여 역사적 문맥을 제공, (2) 분류(Triaging), 사건의 성격을 파악하여 올바른 대응 여부를 판단, (3) 대응(Remediation), 맞춤형 격리 조치를 추천. CGR은 대규모의 데이터를 처리할 수 있는 확장 가능한 머신러닝 시스템으로, 빠른 배치 지연시간(몇 분 이내)에 수백만 건의 사건을 처리할 수 있습니다.

- **Performance Highlights**: CGR의 성능 평가 결과는 인상적입니다. 내부 평가에서 분류 모델은 평균 교차 지역 정밀도(Precision) 87%, 재현율(Recall) 41%를 기록하였으며, 행동 모델은 정밀도 99%, 재현율 62%를 기록하였습니다. Microsoft의 보안 연구 전문가와의 협업에서, 유사 사건 추천의 94%가 관련성이 높은 것으로 평가되었고, 고객의 98%는 하나 이상의 권장 사항을 포함할 수 있다고 응답했습니다. 총체적으로 89%의 고객 상호작용에서 긍정적인 반응을 받았습니다.



### Transforming Movie Recommendations with Advanced Machine Learning: A Study of NMF, SVD,and K-Means Clustering (https://arxiv.org/abs/2407.08916)
Comments:
          Accepted by 2024 4th International Symposium on Computer Technology and Information Science, IEEE

- **What's New**: 이번 연구에서는 다양한 기계 학습 (machine learning) 기법을 사용하여 견고한 영화 추천 시스템을 개발했습니다. 사용된 기법에는 비부정 행렬 인수분해(NMF: Non-Negative Matrix Factorization), 절단 특이값 분해(SVD: Truncated Singular Value Decomposition), 그리고 K-평균 클러스터링(K-Means clustering)이 포함됩니다.

- **Technical Details**: 연구의 주요 목표는 사용자 경험을 향상시키는 개인화된 영화 추천 시스템을 제공하는 것입니다. 데이터 전처리, 모델 훈련, 평가를 포함하여 사용된 방법들의 효율성을 강조하고 있습니다. 특히 NMF와 SVD는 행렬 분해(matrix factorization) 기법으로, 사용자-영화 평점 행렬을 저차원으로 분해하여 추천 정확도를 높였습니다. K-평균 클러스터링(K-Means clustering)은 비슷한 취향을 가진 사용자 그룹을 만들어 추천의 질을 향상시켰습니다.

- **Performance Highlights**: 결과에 따르면 제안된 시스템은 추천의 정확도와 관련성(relevance)에 있어서 높은 성과를 달성했습니다. 이는 추천 시스템 분야에 중요한 기여를 하고 있음을 보여줍니다.



### Are They the Same Picture? Adapting Concept Bottleneck Models for Human-AI Collaboration in Image Retrieva (https://arxiv.org/abs/2407.08908)
Comments:
          Accepted at Human-Centred AI Track at IJCAI 2024

- **What's New**: 새로운 제안으로 이미지 검색에 인간의 개입을 허용하는 기술인 	exttt{CHAIR}이 소개되었습니다. 이는 전통적인 이미지 검색 시스템에서의 한계를 극복하고, `Concept Bottleneck Model`(CBM)을 통해 인간이 중간 개념을 조정하고 이를 통해 임베딩(embeddings)을 개선할 수 있도록 합니다.

- **Technical Details**: 	exttt{CHAIR}은 두 가지 주요 특징을 가지고 있습니다. 첫째, 인간이 중간 개념을 교정할 수 있는 기능을 제공하여 임베딩을 개선할 수 있습니다. 둘째, 다양한 수준의 인간 개입을 허용하여 전문가와 비전문가 모두에게 유연한 사용성을 제공합니다. 이를 통해 검색 성능을 향상하고, 인간과 AI의 상호보완적인 협력을 이끌어낼 수 있습니다. 	exttt{CHAIR}은 실제로 AI 모델의 검색 성능을 인공적인 개입 없이도 개선할 뿐만 아니라, 인간의 개입을 통해 성능이 더욱 향상됨을 증명했습니다.

- **Performance Highlights**: 	exttt{CHAIR} 모델은 기존 모델들보다 이미지 검색 지표에서 더 나은 성능을 보였습니다. 또한, 인간의 개입을 통해 성능이 더욱 향상되어, 인간과 AI의 협력적 접근이 검색 성능을 한층 더 강화할 수 있음을 보여줍니다.




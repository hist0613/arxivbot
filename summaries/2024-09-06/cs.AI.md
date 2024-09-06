New uploads on arXiv(cs.CL)

### WildVis: Open Source Visualizer for Million-Scale Chat Logs in the Wild (https://arxiv.org/abs/2409.03753)
- **What's New**: WildVis는 대규모 대화 데이터셋을 효율적으로 분석할 수 있는 새로운 인터랙티브 툴입니다. 이 도구는 사용자-챗봇 상호작용을 신속하고 다양한 방식으로 분석하는 기능을 제공합니다.

- **Technical Details**: WildVis는 필터 기반 검색 시스템과 임베딩 기반 시각화 모듈로 구성되어 있으며, 수십만 건의 대화를 빠르게 검색하고 시각화할 수 있습니다. 이를 위해 Elasticsearch를 활용하여 검색 인덱스를 구성하고, OpenAI의 text-embedding-3-small 모델을 사용하여 대화의 첫 번째 사용자 발화를 임베딩합니다.

- **Performance Highlights**: WildVis는 대화 데이터의 시각화 및 검색을 지원하며, 사례 연구를 통해 챗봇 남용 연구 촉진, 데이터셋 간 주제 분포 비교, 사용자 특화 대화 패턴을 파악하는 데 기여합니다. 현재 WildVis는 오픈소스로 제공되며, MIT 라이센스 하에 이용 가능합니다.



### Attention Heads of Large Language Models: A Survey (https://arxiv.org/abs/2409.03752)
Comments:
          20 pages, 11 figures, 4 tables

- **What's New**: 본 연구는 Large Language Models (LLMs)의 attention heads의 내부 확인 과정을 조명하고, 인간의 사고 과정을 4단계(지식 회상, 맥락 식별, 잠재적 추론, 표현 준비)로 구분하여 연구하고 있습니다.

- **Technical Details**: attention heads의 기능과 작동 방식을 이해하기 위해 기존의 연구를 조직하고 분석하며, 각 연구에서 사용된 실험 방법론을 Modeling-Free 방법과 Modeling-Required 방법으로 구분하고 있습니다.

- **Performance Highlights**: 이 논문은 attention heads의 협력적 메커니즘을 밝히고, LLM 구조의 명확한 이해를 위한 실험 방법을 정리하여 차세대 연구 방향을 제시합니다.



### RAG based Question-Answering for Contextual Response Prediction System (https://arxiv.org/abs/2409.03708)
Comments:
          Accepted at the 1st Workshop on GenAI and RAG Systems for Enterprise, CIKM'24. 6 pages

- **What's New**: 본 논문은 RAG (Retrieval Augmented Generation) 기술을 활용하여 대형 소매업체의 고객 센터에 적합한 질문-응답 시스템을 구현하는 방법론을 제안합니다. LLM (Large Language Model)을 사용하여 고객 질문에 대한 정확하고 관련성 있는 응답을 생성하는 최적의 방법을 모색하고 있습니다.

- **Technical Details**: 이 연구에서 제안하는 시스템은 고객의 질문을 바탕으로 연관된 지식 문서를 검색하고, 이전 대화 기록과 함께 이를 활용하여 고객 서비스 상담원에게 응답 제안을 생성하는 구조입니다. RAG 아키텍처를 통해 사용자 입력을 처리하여 관련 문서를 검색하고, 이 문서들을 기반으로 최종 예측을 생성합니다. 이를 통해 특히 불확실성을 줄이고 비즈니스 성과를 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: 이 솔루션은 기존 BERT 기반 알고리즘에 비해 정확성과 관련성에서 더 나은 성과를 보였으며, 고객 서비스 담당자에게 우수한 지원을 제공한다는 것을 입증했습니다. 특히, LLM의 hallucination(허상) 문제를 줄이고 응답의 정확성을 크게 향상시켰습니다.



### A Different Level Text Protection Mechanism With Differential Privacy (https://arxiv.org/abs/2409.03707)
- **What's New**: 본 논문은 BERT 사전 훈련 모델을 기반으로 단어의 중요도를 다르게 평가하여 차별적 노이즈를 부여하는 방법을 제안하였습니다. 이 방법은 긴 텍스트 데이터의 정보 손실을 최소화할 수 있는 가능성을 보여주며, 텍스트 보호에 효과적임을 입증했습니다.

- **Technical Details**: 이 연구에서 제안된 방법은 BERT 모델을 통해 각 단어의 주목(attention) 가중치를 추출하고, 이들 가중치를 이용해 단어의 중요도를 정량적으로 측정합니다. 이후, 중요도가 높은 단어는 적게, 낮은 단어는 더 많이 변형하는 방법을 통해 텍스트의 의미 전달을 유지하고자 합니다. 이러한 방법은 SST-2 및 QNLI 두 개의 공개 데이터셋에서 실험하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 동일하게 단어를 변형하는 접근 방식에 비해 긴 텍스트에서 의미 손실을 최소화하는 효과를 나타내었으며, 이를 통해 문서 데이터의 유용성을 높이는 데 기여할 수 있음을 보여주었습니다.



### LAST: Language Model Aware Speech Tokenization (https://arxiv.org/abs/2409.03701)
- **What's New**: 이 논문에서는 새로운 스피치 토크나이저를 제안하며, 미리 학습된 텍스트 언어 모델(Pre-trained Text Language Model)의 목표를 활용하여 학습하는 방법을 소개합니다. 제안된 방법은 전통적인 접근 방식과 달리, 음성과 텍스트 입력 모두를 처리할 수 있는 단일 미리 학습된 LM의 활용을 가능하게 합니다.

- **Technical Details**: 제안된 방법인 LAST는 세 가지 주요 구성 요소로 구성됩니다: 1) 미리 학습된 고정 스피치 SSL 모델, 2) 이 컨텍스트화된 표현을 이산 토큰으로 변환하는 어댑터-양자화 모듈, 3) 토크나이즈 과정에서 더 나은 시퀀스 모델링을 유도하는 고정된 미리 학습된 LM입니다. 특히 LAST는 k-평균 방법보다 모든 설정에서 더 나은 성능을 보였습니다.

- **Performance Highlights**: LAST 방법은 자원 제로 스피치 모델링 작업에서 우수한 성능을 보였으며, 자동 음성 인식(ASR) 작업에서도 k-평균 방법을 초능적으로 능가했습니다. 각 구성 요소의 중요성을 밝히는 광범위한 ablation study도 제공되었습니다.



### The representation landscape of few-shot learning and fine-tuning in large language models (https://arxiv.org/abs/2409.03662)
- **What's New**: 이번 연구는 In-context learning (ICL)과 Supervised fine-tuning (SFT)이라는 두 가지 방법이 대형 언어 모델(LLMs)의 내부 표현을 어떻게 다르게 구축하는지를 분석합니다. ICL과 SFT는 비슷한 성과를 낼 수 있지만, 그 과정에서 생성되는 내부 구조는 상당히 다릅니다.

- **Technical Details**: 연구팀은 LLM의 여러 레이어에서 질문-답변 작업을 수행할 때의 확률 분포(Probability landscape)를 분석하였습니다. ICL은 초기 레이어에서 의미 기반의 해석 가능한 표현(Representations)을 계층적으로 조직합니다. 반면, SFT는 후반 레이어에서 답변의 정체성을 잘 인코딩하는 확률 모드를 발전시킵니다. 연구는 최근 대규모 다중 작업 언어 이해 데이터셋인 MMLU를 사용하여 실험하였습니다.

- **Performance Highlights**: ICL은 초기 레이어에서 보다 해석 가능한 의미 기반 클러스터링을 생성하는 반면, SFT는 후반 레이어에서 높은 정체성을 가진 클러스터 표현을 강조합니다. 연구 결과에 따르면, 두 방법 모두 모델 레이어 간의 뚜렷한 구분을 보이며, 각기 다른 방식으로 같은 문제를 해결하기 위한 정보 추출 전략을 설계하는 데 도움을 줄 수 있습니다.



### LLM-based multi-agent poetry generation in non-cooperative environments (https://arxiv.org/abs/2409.03659)
Comments:
          preprint

- **What's New**: 이번 연구는 기존의 방법과 다른 비협력적 상호작용(non-cooperative interactions)을 강조하여 자동 시(poetry) 생성의 다양성과 새로움을 증진하기 위한 사회적 학습(social learning) 프레임워크를 도입했습니다. 우리는 LLM 기반의 다중 에이전트 시스템을 활용하여 시生成의 첫 번째 실험을 진행하였습니다.

- **Technical Details**: 이 프레임워크는 훈련 기반 에이전트(training-based agents, GPT-2)와 프롬프트 기반 에이전트(prompting-based agents, GPT-3 및 GPT-4)를 모두 포함하여, 비협력적 환경에서의 시 생성 성능을 평가하였습니다. 주요 컴포넌트는 사회적 네트워크(social network), 학습 과정(learning process), 학습 전략(learning strategy)으로 구성됩니다.

- **Performance Highlights**: 훈련 기반 에이전트는 3.0-3.7% 증가한 다양성과 5.6-11.3% 증가한 새로움을 나타냈고, 생성된 시는 어휘(lexicon), 스타일(styles), 의미(semantics) 면에서 집단 내 발산(group divergence)을 보였습니다. 반면, 프롬프트 기반 에이전트는 시간이 지남에 따라 어휘 다양성(lexical diversity)이 감소하며, 목표했던 집단 기반 발산은 보이지 않았습니다.



### Attend First, Consolidate Later: On the Importance of Attention in Different LLM Layers (https://arxiv.org/abs/2409.03621)
- **What's New**: 이번 연구는 디코더 기반 LLM에서 특정 계층의 표현이 이전 토큰의 정보를 처리하는 방식과 관련하여 이론적인 통찰을 제공합니다. 연구 결과에 따르면, 상위 계층에서 이전 토큰의 정보를 변경하면 성능 저하가 거의 발생하지 않으며, 이는 상위와 하위 계층의 역할이 다름을 시사합니다.

- **Technical Details**: 연구팀은 4개의 LLM(Llama2-7B, Mistral-7B, Yi-6B, Llemma-7B)에 대해 다양한 실험을 진행했습니다. 이들 LLM의 상위 및 하위 계층에서 토큰의 Hidden States를 조작하고, 그 변화가 모델 성능에 미치는 영향을 분석했습니다. 특히, 상위 30-50%의 계층에서 조작을 수행할 때 모델의 성능 저하는 거의 발생하지 않는 반면, 하위 계층에서의 조작은 성능 저하를 초래했습니다.

- **Performance Highlights**: 이 연구의 중요한 발견은 LLM이 이전 토큰의 정보를 처리하는 두 단계 프로세스를 거친다는 것입니다. 첫 번째 단계에서는 이전 토큰에서 정보를 수집하고, 두 번째 단계에서는 이 정보를 내부적으로 처리합니다. 연구 결과에 따르면, 상위 계층에서의 조작은 예측 결과에 영향을 미치지 않으며, 이는 모델의 Robustness를 나타냅니다.



### 100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances (https://arxiv.org/abs/2409.03563)
Comments:
          Presented at the 2024 KDD workshop on Evaluation and Trustworthiness of Generative AI Models

- **What's New**: 본 연구에서는 새로운 Large Language Model (LLM)의 성능을 예측하기 위해 기존에 평가된 LLM의 결과를 활용하는 방법을 제안합니다. 구체적으로, 우리는 새로운 LLM을 소규모의 참조 인스턴스 집합에서 평가하고, 이를 바탕으로 범용 assessor를 훈련하여 성능을 예측합니다.

- **Technical Details**: 우리는 HELM-Lite 및 KindsOfReasoning이라는 기존의 추론 데이터셋을 사용하여 여러 OpenAI 모델을 평가하였습니다. 제안된 방법은 LLM의 특정 성능 벡터와 인스턴스 특정 특징을 결합하여 범용 assessor를 훈련하는 방식으로 구성됩니다. 평가된 인스턴스의 분포가 기존 assessor 훈련에 사용된 인스턴스의 분포와 동일할 경우, 제안된 방법이 LLM-특정 assessor와 유사한 성능을 발휘하는 것을 발견했습니다.

- **Performance Highlights**: 우리는 랜덤으로 선택된 참조 인스턴스들이도 고급 선택 방법들과 비슷한 성능을 보인다는 것을 찾았습니다. 하지만 분포에서 벗어난 경우, 모든 assessor의 예측력이 크게 감소하여 LLM의 본질적인 예측 가능성이 낮다는 것을 시사합니다.



### How Much Data is Enough Data? Fine-Tuning Large Language Models for In-House Translation: Performance Evaluation Across Multiple Dataset Sizes (https://arxiv.org/abs/2409.03454)
- **What's New**: 본 연구는 Llama 3 8B Instruct 모델을 소프트웨어 분야의 특정 조직에서 얻은 번역 메모리(Translation Memories, TMs)를 활용하여 미세 조정(fine-tuning)함으로써 번역의 정확도와 효율성을 향상시키는 방법을 탐구합니다. 특히, 다양한 데이터 세트 사이즈의 변화가 번역 품질에 미치는 영향을 분석했습니다.

- **Technical Details**: 연구는 총 다섯 가지 언어(브라질 포르투갈어, 체코어, 독일어, 핀란드어 및 한국어)로의 번역 실험을 진행하였으며, 다양한 사이즈(1k에서 207k 세그먼트)에 해당하는 데이터 세트를 이용하여 Llama 3 모델을 조정했습니다. 각 세트에 대해 BLEU, chrF++, TER, COMET와 같은 자동 평가 메트릭을 사용하여 성능을 평가했습니다.

- **Performance Highlights**: 가장 큰 학습 세트를 사용할 경우, BLEU 점수가 평균 13점, COMET 점수가 25점 상승하는 등 모든 메트릭에서 번역 성능이 개선되었습니다. 그러나 적은 양(1k 및 2k)의 예제로 미세 조정할 경우에는 성능 저하가 발생하였지만, 학습 데이터 세트의 크기가 증가하면서 상당한 개선이 있었습니다.



### Fine-tuning large language models for domain adaptation: Exploration of training strategies, scaling, model merging and synergistic capabilities (https://arxiv.org/abs/2409.03444)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)를 특정 도메인에 맞추어 조정하기 위한 다양한 미세 조정(fine-tuning) 전략의 효과를 탐구합니다. 특히 Continued Pretraining (CPT), Supervised Fine-Tuning (SFT), 그리고 Preference-based 최적화 접근법을 분석하였습니다.

- **Technical Details**: 분석 결과, 여러 개의 미세 조정된 모델을 결합하면, 각 모델의 개별적인 기여도를 초월하는 새로운 기능이 발생할 수 있음을 알았습니다. Llama 3.1 8B와 Mistral 7B 모델을 포함한 다양한 모델 아키텍처에 대한 실험이 진행되었으며, 결과는 유사한 행동을 보여주었습니다. 또한, 17억 개의 파라미터를 가진 소형 LLM에서도 모델 병합에 따른 emergent capabilities를 관찰했습니다.

- **Performance Highlights**: 특히, 매우 작은 LLM 모델이 여러 중요한 기준(사고 심도, 창의성, 명확성, 정량적 정밀도)에서 높은 지능 점수를 기록한 점이 주목할 만합니다. 이 외에도 생물학적 재료 디자인 개념을 기반으로 새로운 마이크로구조, 건축 개념, 도시 디자인을 생성하기 위한 이미지 생성 프롬프트 개발 실험이 포함되어 있습니다.



### Rx Strategist: Prescription Verification using LLM Agents System (https://arxiv.org/abs/2409.03440)
Comments:
          17 Pages, 6 Figures, Under Review

- **What's New**: 이번 연구에서는 현대 의약품의 복잡성에 대응하기 위해 처방의 안전성을 보호하는 새로운 접근법인 'Rx Strategist'를 제안합니다. 이 시스템은 지식 그래프(knowledge graph)와 다양한 검색 전략을 활용하여 대형 언어 모델(Large Language Models, LLMs)의 성능을 개선하는 에이전트 기반의 프레임워크로 구성됩니다.

- **Technical Details**: Rx Strategist는 두 가지 주요 작업인 적응 증거(indication verification)와 용량 검증(dose verification)을 수행하는 전문 에이전트로 구성됩니다. 각 에이전트는 지식 그래프, 규칙 기반 시스템, LLM 요소를 독특하게 결합하여 구성됩니다. 이를 통해 처방된 활성 성분에 대한 포괄적인 분석을 수행하고, 환자별 정보와 기존의 의학 지식을 반영합니다. 또한, 지식 검색을 위한 새로운 방법론과 약물 정보 중심의 전문 데이터셋을 도입하여 시스템의 신뢰성과 성능을 향상시킵니다.

- **Performance Highlights**: Rx Strategist는 현재의 많은 LLM을 초월하여 경험이 풍부한 임상 약사와 비슷한 성과를 달성합니다. 이 시스템은 메모리 요구 사항을 줄이고, 정확성과 신뢰성을 향상시키며, 여러 단계의 LLM 파이프라인을 통해 다양한 처방 검증 측면을 다루는 능력을 보입니다.



### CogniDual Framework: Self-Training Large Language Models within a Dual-System Theoretical Framework for Improving Cognitive Tasks (https://arxiv.org/abs/2409.03381)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 인간의 인지 시스템인 Kahneman의 이중 시스템 이론을 모방할 수 있는 가능성을 제시합니다. 연구에서는 LLMs의 자기 훈련(self-training)을 통해 의도적인 추론(System 2)에서 직관적인 응답(System 1)으로 진화할 수 있는 방법을 탐구합니다.

- **Technical Details**: 제안된 CogniDual Framework for LLMs(CFLLMs)는 LLM들이 합리적인 응답을 제공하도록 훈련되며, CoT(Chain of Thought) 유도 없이는 즉각적인 반응을 생성할 수 있는지를 확인하기 위한 방법론을 포함합니다. 연구는 Vicuna와 Llama2 모델을 다양한 크기로 평가하며, GSM8K, ReClor, LogiQA 2.0과 같은 추론 데이터 세트에서 성능을 검사합니다.

- **Performance Highlights**: 자기 훈련 후, LLMs는 CoT가 없는 상황에서도 응답 정확도가 크게 향상되었음을 보여주었습니다. 이는 LLMs가 인간의 인지 시스템과 유사하게 직관적인 응답 메커니즘을 개발할 수 있음을 의미합니다. 또한 새로운 방법이 CoT에 의존하지 않고도 효율적이고 정확한 출력을 유지할 수 있는 가능성을 제시합니다.



### Leveraging Large Language Models through Natural Language Processing to provide interpretable Machine Learning predictions of mental deterioration in real tim (https://arxiv.org/abs/2409.03375)
- **What's New**: 본 논문은 인공지능(AI) 및 자연어처리(NLP) 기술을 활용하여 인지 저하에 대한 해석 가능한 기계 학습 예측을 실시간으로 제공하는 챗봇 솔루션을 제안합니다. 전통적인 접근법의 한계를 극복하기 위한 노력의 일환으로, 최신 자연어 처리 기법을 적용하여 개인 맞춤형 진단 시스템을 개발하였습니다.

- **Technical Details**: 제안된 파이프라인은 (i) NLP 기반 프롬프트 엔지니어링을 통한 데이터 추출, (ii) 특징 엔지니어링을 포함한 스트림 기반 데이터 처리, (iii) 실시간 분류, (iv) 예측 결과의 시각적 및 자연어 설명을 제공하는 설명 가능성 대시보드로 구성됩니다. 언어-개념적 특징을 활용하여 자연어 분석을 수행하고 '할루시네이션' 효과를 피하기 위한 프롬프트 엔지니어링을 적용하였습니다.

- **Performance Highlights**: 모든 평가 지표에서 분류 결과가 80%를 초과하며, 특히 인지 저하 클래스의 리콜 값은 약 85%에 달합니다. 이를 통해 저렴하고 유연하며 비침습적인 개인 맞춤형 진단 시스템을 제공함으로써 실시간으로 인지 저하를 모니터링할 수 있는 가능성을 제시합니다.



### Con-ReCall: Detecting Pre-training Data in LLMs via Contrastive Decoding (https://arxiv.org/abs/2409.03363)
- **What's New**: 이 논문에서는 기존의 방법에서 간과되었던 회원(contexts)과 비회원(contexts)의 조합을 통해 얻는 귀중한 정보를 활용할 수 있는 새로운 접근 방식인 Con-ReCall을 제안합니다.

- **Technical Details**: Con-ReCall은 대조적 디코딩(contrastive decoding)을 통하여 회원 및 비회원 contexts에서 유도되는 비대칭 분포 변화(asymmetric distributional shifts)를 활용하여 멤버십 추론(membership inference)의 정확성을 높입니다.

- **Performance Highlights**: 폭넓은 실험 평가를 통해, Con-ReCall은 WikiMIA 벤치마크에서 최첨단 성능(state-of-the-art performance)을 달성하며, 다양한 텍스트 조작 기법에 대해서도 강력한 견고성(robustness)을 보입니다.



### Sketch: A Toolkit for Streamlining LLM Operations (https://arxiv.org/abs/2409.03346)
- **What's New**: 이 논문에서는 다양한 NLP 작업을 지원하는 LLM(대규모 언어 모델) 운영을 단순화하기 위한 혁신적인 도구 키트인 Sketch를 소개합니다. Sketch는 사용자가 구조적 출력을 효율적으로 구축하고 제어할 수 있도록 설계되었습니다.

- **Technical Details**: Sketch는 (1) 여러 NLP 작업을 포함하는 작업 설명 스키마와 프롬프트 템플릿 모음, (2) 사용자 친화적이며 대화형인 구조적 출력 LLM 서비스 구축 과정, (3) 출력 형식 제어를 위한 오픈 소스 데이터셋과 데이터셋 구축 도구, (4) 출력 형식 지침을 이해하고 준수하는 LLaMA3-8B-Instruct 기반의 오픈 소스 모델 등을 포함합니다. Sketch는 'plug-and-play' 기능을 통해 다양한 응용 프로그램에 적합하게 설계되었습니다.

- **Performance Highlights**: Sketch는 자연어 처리를 위한 다양한 작업에서의 성능 향상과 함께 출력 형식에 대한 수요를 충족시키기 위한 우수한 구조적 답변 생성을 가능하게 합니다. 이렇게 함으로써, LLM의 신뢰성과 정확성을 높이고 사용자 경험을 정교하게 개선하는 것을 목표로 합니다.



### Normal forms in Virus Machines (https://arxiv.org/abs/2409.03327)
- **What's New**: 본 논문은 바이러스 머신(virus machines, VMs)의 계산 능력을 심화 연구하며, 일반적인 계산 모델에서의 특성을 제한하는 정상형(normal forms)을 도입합니다. 일반적으로 알려진 결과를 바탕으로 새로운 특징과 구조를 논의하며, VM의 성능 특성을 더 정교한 시각으로 조망할 수 있게 합니다.

- **Technical Details**: 바이러스 머신은 프로세스 유닛(호스트, hosts)과 방향 그래프(directed graph)로 구성된 채널(channels)이 있는 복잡한 구조로 이루어져 있습니다. 호스트는 다수의 바이러스 오브젝트(virus objects)를 담을 수 있으며, 지시 그래프(instruction graph)는 지시 및 채널 간의 관계를 다룹니다. 본 연구에서는 VM의 특정 정상형을 정의하며 이는 VM의 최대 및 최소 요구 사항을 설명하고, 유한 집합(finite sets), 선형 집합(semilinear sets) 등과 관련된 새로운 특성을 제시합니다.

- **Performance Highlights**: 일부 제한 조건 하에, VM은 특정한 유한 집합을 계산할 수 있으며, 새로운 정상형은 여러 응용 분야에서 활용될 수 있는 가능성을 제시합니다. 특히, SLIN(Semilinear sets)은 유한 집합 NF인(NFIN)과 튜링 결정 가능 집합 NRE(NRE) 사이의 관계를 강조하며 응용과 이론적으로 중요한 지점을 나타냅니다.



### N-gram Prediction and Word Difference Representations for Language Modeling (https://arxiv.org/abs/2409.03295)
- **What's New**: 이번 연구는 인과 언어 모델링(causal language modeling, CLM) 작업을 위한 간단한 N-그램 예측 프레임워크를 소개합니다. 이 프레임워크는 기존 CLM 모델에 쉽게 통합할 수 있으며, 구조의 복잡성을 줄였다는 특징이 있습니다.

- **Technical Details**: 이 연구에서는 N-그램(n-gram) 예측을 통해 모델 훈련 중 단어 차이 표현(word difference representation, WDR)을 서브레이트(replace) 형태로 활용합니다. WDR은 인접한 단어들 간의 임베딩 벡터 차이를 이용하여 컨텍스트에 따라 다양해진 목표 표현(target representation)을 제공합니다. 또한, N-그램 예측의 결과를 활용한 앙상블(ensemble) 방법을 제안하여 품질 향상을 꾀합니다.

- **Performance Highlights**: 실험 결과, 제안된 간단한 N-그램 프레임워크, WDR의 다양한 목표 표현, 그리고 앙상블 방법이 기존 CLM 모델들보다 수렴성(perplexity)에서 유의미한 개선을 보였음이 입증되었습니다. NMT(신경 기계 번역) 작업에서도 제안된 방법의 활용 가능성과 장점을 보여주었습니다.



### LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts (https://arxiv.org/abs/2409.03291)
Comments:
          20 pages, 7 tables, 13 figures, under consideration for EMNLP

- **What's New**: 본 연구는 대규모 Language Models (LLMs) 이용한 정보작전에서 기존 LLM 탐지기들의 실용성 부족을 드러냅니다. 특히, 중간 정도의 공격자에 의해 생성된 짧은 뉴스 형식의 포스트를 목표로 하는데요, 이 설정에서의 검증이 부족했습니다.

- **Technical Details**: 연구팀은 LLM 탐지기의 기존의 제로샷(zero-shot) 접근법이 일관성이 없고 샘플링 온도 증가와 같은 간단한 공격에 취약하다는 점을 강조했습니다. 맞춤형 탐지기를 개발할 수 있지만, 이는 새로운 인간 작성 텍스트에 대한 일반화에 실패했습니다. 이는 특정 도메인에 대한 벤치마킹 필요성을 시사합니다.

- **Performance Highlights**: 연구 결과, 현재 LLM 탐지기가 LLM이 생성하는 허위 정보를 탐지하기에는 준비가 되어 있지 않으며, 일반화 및 과적합 간의 trade-off 문제가 존재함을 밝혔다.



### GraphInsight: Unlocking Insights in Large Language Models for Graph Structure Understanding (https://arxiv.org/abs/2409.03258)
- **What's New**: 이번 연구에서는 GraphInsight라는 새로운 프레임워크를 제안하여, Large Language Models (LLMs)가 그래프 구조를 이해하는 데 있어 메모리 성능의 불균형 문제를 해결하고자 합니다.

- **Technical Details**: GraphInsight는 두 가지 주요 전략에 근거하여 설계되었습니다: 1) LLM의 메모리 성능이 우수한 위치에 중요 그래픽 정보를 배치하고, 2) retrieval-augmented generation (RAG)에서 영감을 받아, 메모리 성능이 낮은 지역을 위한 경량 외부 지식 베이스를 조사합니다. 또한, 이 두 가지 전략을 LLM 에이전트 프로세스에 통합하여 복합 그래프 작업 수행을 도모합니다.

- **Performance Highlights**: GraphInsight는 다양한 그래프 크기의 구조를 이해하는 데 있어서 기존의 모든 그래프 설명 방법을 크게 초월하는 성능을 보여주었습니다. 이를 위해 새로운 benchmark인 GraphSQA를 소개하여, LLM의 그래프 이해 능력을 폭넓게 평가할 수 있도록 시스템적으로 설계되었습니다.



### Understanding LLM Development Through Longitudinal Study: Insights from the Open Ko-LLM Leaderboard (https://arxiv.org/abs/2409.03257)
- **What's New**: 본 논문은 Open Ko-LLM Leaderboard를 분석하여 한국어 대규모 언어 모델(LLMs)의 발전 상황을 11개월 동안 조사한 장기적인 연구를 소개합니다. 이전 연구는 5개월이라는 제한된 관찰 기간에 의존하였으나, 이번 연구는 보다 포괄적인 이해를 제공합니다.

- **Technical Details**: 본 연구는 1,769개의 모델을 분석하였고, LLM 성능의 변화를 다섯 개의 과제를 통해 모니터링했습니다. 연구 질문은 LLM 성능 향상의 구체적인 도전 과제, 모델 크기가 과제 성능의 상관관계에 미치는 영향, 그리고 Open Ko-LLM Leaderboard에서의 랭킹 변화입니다.

- **Performance Highlights**: 성능 향상에서 특정 과제, 예를 들어 Ko-HellaSwag와 Ko-TruthfulQA는 빠른 개선을 보였으나, Ko-MMLU와 Ko-CommonGEN V2는 느리면서도 지속적인 발전을 보였습니다. 모델 크기가 커질수록 과제 간의 상관관계가 증가했으며, 이는 모델 성능 향상에 긍정적인 영향을 미쳤습니다.



### E2CL: Exploration-based Error Correction Learning for Embodied Agents (https://arxiv.org/abs/2409.03256)
- **What's New**: 이 논문에서는 Exploration-based Error Correction Learning (E2CL)이라는 새로운 프레임워크를 제안하여 LM 기반 에이전트의 환경 정렬을 개선합니다. 이 프레임워크는 탐사를 통한 오류 및 환경 피드백을 활용하여 에이전트의 자가 수정 능력을 향상시키고 있습니다.

- **Technical Details**: E2CL은 탐사로 인한 오류와 환경 피드백을 통합하여 LM 기반 에이전트가 목표 환경과 철저히 정렬될 수 있도록 돕습니다. 이 프레임워크는 두 가지 탐사 방식을 포함하는데, 하나는 선생님이 지도하는 탐사(teacher-guided exploration)이고, 다른 하나는 비지도 탐사(teacher-free exploration)입니다. 이를 통해 환경과의 상호작용에서 에러 행동과 올바른 행동의 피드백을 수집합니다. 또한, 고안된 Speculative Inference 알고리즘을 통해 초기 계획된 행동이 오류로 간주될 경우 수정이 이루어지도록 합니다.

- **Performance Highlights**: Virtualhome 환경에서 E2CL로 훈련된 에이전트는 다른 기준 방법들로 훈련된 에이전트보다 뛰어난 성능을 보이며, 자가 수정 능력에서도 우수한 결과를 나타냈습니다. E2CL로 훈련된 소규모 모델은 동일 시리즈의 대규모 모델보다 더 나은 성능을 발휘하였으며, 피드백 기반 재계획 평가에서도 LLM과 유사한 자가 수정 능력을 보여주었습니다.



### Preserving Empirical Probabilities in BERT for Small-sample Clinical Entity Recognition (https://arxiv.org/abs/2409.03238)
Comments:
          8 pages, 8 figures

- **What's New**: 본 논문은 Named Entity Recognition (NER) 분야에서 불균형한 레이블 문제를 다루고 있습니다. 특히, BERT 기반의 사전 훈련 모델에서 불균형 레이블이 미치는 영향을 탐색하고, 토큰 분류 작업을 위한 새로운 손실 계산 및 전파 기법을 분석합니다.

- **Technical Details**: NER은 텍스트에서 이름, 사건, 사물 및 장소와 같은 엔티티를 식별하고 분류하는 자연어 처리(NLP) 작업입니다. O(99%)와 M(1%)과 같이 불균형한 데이터 세트를 다룰 때, 전통적인 손실 함수인 교차 엔트로피(cross-entropy)는 성능을 최적화하는 데 부족할 수 있습니다. 이 논문에서 제안하는 방법은 가중 교차 엔트로피 손실을 통한 해결책을 모색하며, 이러한 기법들이 임상 엔티티 인식을 개선할 수 있는지 확인합니다.

- **Performance Highlights**: 임상 엔티티 데이터셋 MACCROBAT를 활용하여 20 에폭 동안 BioBERT v1.1 모델로 토큰 분류 작업을 수행한 결과, 불균형한 레이블에 의해 기존의 NER 모델들이 소수 클래스를 인식하는 데 어려움을 겪는다는 것을 보여줍니다. 연구 결과는 NER 시스템의 정확성과 공정성이 향상될 수 있다는 것을 강조합니다.



### Enhancing Healthcare LLM Trust with Atypical Presentations Recalibration (https://arxiv.org/abs/2409.03225)
- **What's New**: 본 연구는 의료 환경에서 블랙박스 대형 언어 모델(LLMs)의 잘못된 캘리브레이션(miscalibration) 현상을 조사하고, 이를 위한 새로운 방법인 	extit{Atypical Presentations Recalibration}을 제안합니다. 이 방법은 비전형적인 사례를 활용하여 모델의 신뢰도 추정치를 조정하는 데 중점을 두고 있습니다. 이렇게 함으로써 캘리브레이션 오차를 약 60% 줄일 수 있으며, 기존 방법들보다 뛰어난 성능을 발휘합니다.

- **Technical Details**: 본 연구에서는 LLM의 캘리브레이션 개선을 위해 비전형적인 사례의 개념을 바탕으로 하는 새로운 방법론인 Atypical Presentations Recalibration을 제안합니다. 이 방법은 LLM을 사용하여 비전형적인 사례를 명시적으로 고려하고 논리적으로 추론하게끔 하는 두 가지 이질성 인식 프롬프트 전략을 구성합니다. 이러한 전략의 효과는 기존의 기준 방법들과 비교하여 성능과 캘리브레이션 개선을 평가하는 방식으로 확인하였습니다.

- **Performance Highlights**: 실험 결과, 블랙박스 LLM은 의료 질문에 대한 응답 시 신뢰도 추정이 정확하지 않고 과신(overconfidence) 경향이 있다는 것을 발견하였습니다. 또한, 제안된 Atypical Presentations Aware Recalibration 방법은 기존 기준 방법들에 비해 모든 데이터셋에서 일관되게 뛰어난 성능을 보이며, 캘리브레이션 오차를 번역하여 신뢰도를 크게 개선하였습니다.



### xLAM: A Family of Large Action Models to Empower AI Agent Systems (https://arxiv.org/abs/2409.03215)
Comments:
          Technical report for the Salesforce xLAM model series

- **What's New**: xLAM이라는 새롭고 특화된 대형 행동 모델 시리즈가 개발되어 공공에 공개되었습니다. 이 모델들은 AI 에이전트 작업을 위해 설계되었습니다.

- **Technical Details**: xLAM 시리즈는 1B에서 8x22B 파라미터에 이르는 5개의 모델로 구성되어 있으며, 밀집(Dense) 및 혼합 전문가(Mixture-of-Expert) 아키텍처를 사용합니다. 이러한 모델은 다양한 데이터셋을 통합, 증강 및 합성하여 AI 에이전트의 일반화(Generalizability)와 성능을 향상시키는 확장 가능하고 유연한 파이프라인을 사용하여 훈련됩니다.

- **Performance Highlights**: xLAM은 여러 에이전트 능력 벤치마크에서 뛰어난 성능을 보여주었으며, 특히 Berkeley Function-Calling Leaderboard에서 1위를 기록하며 GPT-4와 Claude-3 등 여러 모델을 능가했습니다. 이는 도구 사용(tool use) 측면에서의 성능 향상을 뜻합니다.



### An Effective Deployment of Diffusion LM for Data Augmentation in Low-Resource Sentiment Classification (https://arxiv.org/abs/2409.03203)
- **What's New**: 본 논문에서는 감정 분류(Sentiment Classification)의 데이터 증강(Data Augmentation) 문제를 해결하기 위해 Diffusion 언어 모델을 활용한 새로운 방법, DiffusionCLS를 제안합니다. 이 방법은 도메인 지식(in-domain knowledge)을 포착하고 강력한 레이블 관련 토큰을 재구성하여 새로운 샘플을 생성합니다.

- **Technical Details**: DiffusionCLS는 레이블-주도(noise-resistant) 훈련 방식을 포함하여 감정 분류 모델의 성능을 향상시키고 다양한 저자원 시나리오에서도 효과적으로 작동합니다. 주요 구성 요소로는 레이블 인지 노이즈 일정(Label-Aware Noise Schedule), 레이블 인지 프롬프트(Label-Aware Prompting), 조건부 샘플 생성(Conditional Sample Generation) 등이 있습니다.

- **Performance Highlights**: DiffusionCLS는 감정 분류 과제에서 뛰어난 성능을 보였으며, 다양한 도메인 특정 및 다국어 데이터셋을 기반으로 한 포괄적인 실험을 통해 그 우수성을 입증했습니다.



### Bypassing DARCY Defense: Indistinguishable Universal Adversarial Triggers (https://arxiv.org/abs/2409.03183)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구에서는 IndisUAT라는 새로운 Universal Adversarial Trigger (UAT) 생성 방법을 제안합니다. 이 방법은 DARCY의 탐지를 효과적으로 우회할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: IndisUAT는 텍스트 분류 모델 보호를 위해 설계된 공격 방식으로, 공격자는 반복적으로 트리거 시퀀스를 업데이트하여 DARCY의 탐지 레이어에서 트랩 도어의 서명을 활성화하지 않도록 합니다. 이렇게 생성된 적대적 예시는 DARCY로 보호받는 모델에서 최대의 손실을 가져옵니다.

- **Performance Highlights**: IndisUAT는 DARCY의 진짜 정답 비율(true positive rate)을 최소 40.8%에서 90.6%까지 감소시키며, RNN과 CNN 모델의 정확도를 각각 최소 33.3% 및 51.6%까지 하락시킵니다. 또한 BERT의 적대적 방어 모델의 정확도는 최소 34.0% 감소시키며, GPT-2 언어 모델에서는 비인종적 맥락에서도 인종차별적인 출력을 생성하게 합니다.



### MARAGS: A Multi-Adapter System for Multi-Task Retrieval Augmented Generation Question Answering (https://arxiv.org/abs/2409.03171)
Comments:
          Accepted to CRAG KDD Cup 24 Workshop

- **What's New**: 이 논문에서는 KDD CUP 2024의 Meta Comprehensive RAG (CRAG) 대회를 위한 다중 어댑터 검색 증강 생성 시스템(MARAGS)을 제시합니다. CRAG 데이터셋은 다양한 질문 주제와 유형을 포함한 질문-응답 RAG 관련 작업을 위한 세 가지 하위 과제를 포함하고 있습니다. 시스템은 표준 웹 기반 RAG 설정을 따르며, LLM(대형 언어 모델)이 생성을 위한 문맥을 제공하기 위해 처리된 웹 페이지를 사용합니다.

- **Technical Details**: MARAGS는 여러 개의 어댑터를 활용하여 각 하위 작업의 다양성을 해결하며, 질문에 대한 관련 후보 구문을 순위화하기 위해 표준 크로스 인코더 모델을 사용합니다. CRAG 벤치마크는 착각(hallucination)을 명시적으로 처벌하는 점수 체계를 포함하여 정확한, 누락된, 착각된 답변에 대해 각각 1, 0, -1의 점수를 부여합니다.

- **Performance Highlights**: 시스템은 Task 1에서 2위, Task 2에서 3위를 달성했습니다. HTML 문서 처리 파이프라인은 BeautifulSoup4를 사용하여 후보 문서를 세분화하며, 각 하위 작업(1, 2, 3)에 맞춘 다양한 기능을 구현하고 있습니다.



### MaterialBENCH: Evaluating College-Level Materials Science Problem-Solving Abilities of Large Language Models (https://arxiv.org/abs/2409.03161)
- **What's New**: 새로운 데이터셋 MaterialBENCH가 구축되어 대규모 언어 모델(LLMs)이 재료 과학 분야에서 평가될 수 있는 기준을 제공합니다. 이 데이터셋은 대학 수준의 교과서에 기반한 문제-답변 쌍(problem-answer pairs)으로 구성됩니다.

- **Technical Details**: MaterialBENCH는 두 가지 유형의 문제로 구성됩니다: 자유응답형(free-response answer type) 문제와 객관식(multiple-choice) 문제. 객관식 문제는 정답 한 개와 세 개의 오답을 선택지로 만들어 LLM이 네 개 중 하나를 선택하도록 설계되었습니다. 실험은 ChatGPT-3.5, ChatGPT-4, Bard 및 OpenAI API의 GPT-3.5와 GPT-4와 함께 진행되었습니다.

- **Performance Highlights**: 자유응답형과 객관식 문제에서 각각의 모델의 성능 차이를 분석하였으며, 동일 모델에서의 두 유형 간의 성능 차이와 객관식 문제에서 시스템 메시지(system messages)의 사용이 미치는 영향에 대해서도 연구했습니다. MaterialBENCH는 LLM의 추론 능력 향상을 촉진할 것으로 기대됩니다.



### Debate on Graph: a Flexible and Reliable Reasoning Framework for Large Language Models (https://arxiv.org/abs/2409.03155)
Comments:
          12 pages

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)과 지식 그래프를 통합하는 새로운 프레임워크인 DoG(Debating over Graphs)를 제안합니다. 이 프레임워크는 LLM의 상호작용 학습 능력을 활용하여 복잡한 질문을 단계적으로 간소화할 수 있도록 합니다.

- **Technical Details**: DoG는 서브그래프 집중 메커니즘을 사용하여 각 추론 단계 후 LLM이 답변을 시도할 수 있게 하며, 이를 통해 길고 복잡한 경로의 영향을 줄입니다. 또한 DoG는 멀티 역할 토론 팀을 활용하여 거짓 긍정 관계의 영향을 완화하고, 질문을 단순화하여 신뢰성을 높입니다.

- **Performance Highlights**: 실험 결과, DoG는 WebQuestions와 GrailQA 데이터셋에서 기존 최고 성능 방법인 ToG에 비해 각각 23.7% 및 9.1% 더 높은 정확도를 기록하며, 다섯 개의 공개 KGQA 데이터셋에서 모두 뛰어난 성능을 보였습니다.



### Probing self-attention in self-supervised speech models for cross-linguistic differences (https://arxiv.org/abs/2409.03115)
Comments:
          10 pages, 18 figures

- **What's New**: 본 논문은 자가 주의를 사용하는 음성 변환기 모델(TERA)의 다국어 학습에서의 주의(heads) 메커니즘의 다양성과 언어 간 차이를 연구하였습니다. 예를 들어, 터키어와 영어 간의 주의 패턴의 뚜렷한 차이를 강조하며, 모델이 중요 음소 정보를 학습하고 있음을 보여줍니다.

- **Technical Details**: 이 모델은 OSS(Speech Self-supervised) 방식으로 훈련되었으며, 레이어 수 3, 파라미터 수 21.3M의 복잡한 구조를 갖습니다. 실험에서는 주의(heads)를 세 가지 범주(글로벌, 수직, 대각선)로 분류하고, 이들이 어떻게 언어에 따라 다르게 작동하는지를 조사했습니다.

- **Performance Highlights**: 모델은 여러 언어 간의 음소 분류 작업에서 주로 대각선을 이용하여 음소를 분류하는 결과를 보였습니다. 실험 결과, 훈련 언어에 관계없이 주의 패턴에서 큰 차이를 발견하지 못했으며, 이는 언어 독립적인 표현을 학습하고 있음을 나타냅니다.



### Quantification of stylistic differences in human- and ASR-produced transcripts of African American English (https://arxiv.org/abs/2409.03059)
Comments:
          Published in Interspeech 2024 Proceedings, 5 pages excluding references, 5 figures

- **What's New**: 이 연구는 자동 음성 인식(ASR) 시스템 및 인간의 전사기 간의 성능 평가에서 스타일리틱 차이가 발생하는 방식을 분석합니다. 특히 전사 스타일, 즉 verbatim(축어적)와 non-verbatim(비축어적) 전사 간의 차이를 조명하며, 아프리카계 미국인 영어(AAE)의 전사 정확도를 탐색합니다.

- **Technical Details**: 이 논문에서는 6개의 전사 버전(4개의 인간 작성 및 2개의 ASR 생성)에서 발견된 스타일리틱 차이를 카테고리화합니다. 연구팀은 ASR 모델과 인간 전사자 간의 변형을 비교하기 위해 단어 오류율(Word Error Rate, WER)을 사용하며, AAE의 morphosyntactic(형태통사론적) 특성을 고려합니다.

- **Performance Highlights**: 연구 결과, AAE에 대한 ASR 모델의 편향을 확인하고, 인간 전사자와 ASR 간의 전사 결정이 어떻게 다르게 나타나는지를 분석함으로써 AAE의 orthography(철자법)에 대한 ASR 시스템의 적합성을 평가할 수 있게 됩니다.



### Oddballness: universal anomaly detection with language models (https://arxiv.org/abs/2409.03046)
- **What's New**: 새로운 이상 감지 방법을 제안합니다. 이 방법은 텍스트에서 발생하는 이상치를 감지하는 데 사용되며, 기존의 저확률 토큰에 초점을 맞추는 대신 'oddballness'라는 새로운 메트릭을 고려합니다.

- **Technical Details**: 이 방법은 언어 모델에서 생성된 확률( likelihoods)을 기반으로 하며, oddballness를 통해 특정 토큰이 얼마나 '이상한'지를 측정합니다. 이 메트릭은 전통적인 저확률 사건을 고려하는 것보다 더 효과적입니다. 기초적인 가정으로 oddballness는 0부터 1까지의 값을 가지며, 불가능한 사건은 최대의 oddballness를 가집니다.

- **Performance Highlights**: 문법 오류 감지 작업에서 oddballness를 이용한 접근 방식이 저확률 이벤트만을 고려한 방법보다 우수함을 입증하였습니다.



### CLUE: Concept-Level Uncertainty Estimation for Large Language Models (https://arxiv.org/abs/2409.03021)
- **What's New**: 본 논문에서는 기존의 시퀀스 레벨 불확실성 추정 방법의 한계를 극복하기 위해 Concept-Level Uncertainty Estimation (CLUE)라는 새로운 프레임워크를 제안합니다. 이는 LLMs(대형 언어 모델)로부터 생성된 출력 시퀀스를 개념 수준으로 변환하여, 개별 개념의 불확실성을 별도로 측정합니다.

- **Technical Details**: CLUE는 LLM을 활용해 생성된 시퀀스를 개념 수준의 표현으로 변환하고, 각 개념의 불확실성을 개별적으로 측정합니다. 이를 위해 NLI 기반의 제로샷 텍스트 분류기를 사용하여 개념 점수를 할당하고, 평균 음의 로그 개념 점수를 통해 최종 불확실성을 결정합니다.

- **Performance Highlights**: 실험 결과 CLUE는 QA 데이터셋에서 할루시네이션(허위 정보) 탐지에 있어 기존 방법보다 21% 더 높은 매크로 AUROC를 달성했고, 인간의 판단 예측 정확도가 33% 향상되어 개념 수준의 방법이 보다 직관적임을 입증하였습니다.



### Lexicon3D: Probing Visual Foundation Models for Complex 3D Scene Understanding (https://arxiv.org/abs/2409.03757)
Comments:
          Project page: this https URL , Github: this https URL

- **What's New**: Lexicon3D라는 새로운 프로빙 아키텍처를 통해 시각적 인코더의 성능을 평가하고, 이미지 기반, 비디오 기반, 3D 기반의 다양한 모델의 강점과 한계를 식별합니다.

- **Technical Details**: 본 연구에서는 7개의 시각적 개념 모델을 평가하며, 각 모델의 기능적 강점 및 약점을 정의하기 위해 Vision-Language Scene Reasoning, Visual Grounding, Segmentation 및 Registration의 4가지 작업을 사용했습니다.

- **Performance Highlights**: DINOv2가 우수한 성능을 보이며, 비디오 모델은 객체 수준의 작업에서 뛰어난 결과를 나타냅니다. 또한, 확산 모델은 기하학적 작업에 이점이 있음을 보여주고, 언어로 사전 훈련된 모델은 언어 관련 작업에서 예상치 못한 한계를 드러냅니다.



### Planning In Natural Language Improves LLM Search For Code Generation (https://arxiv.org/abs/2409.03733)
- **What's New**: PLANSEARCH는 다양하고 효율적인 문제 해결 아이디어 검색을 통해 코드 생성의 성능을 향상시키는 새로운 알고리즘입니다. 이 알고리즘은 단순히 잘못된 유사한 출력을 반복하는 대신, 다양한 후보 플랜을 탐색합니다.

- **Technical Details**: PLANSEARCH는 문제 해결을 위해 생성된 다양한 관찰(observations)을 바탕으로 높은 수준의 계획(plans)으로 구성된 후보 세트를 생성합니다. 이러한 계획은 자연어(natural language)로 표현되며, 이는 다양한 잠재적 솔루션을 탐색하는 데 기여합니다. 결과적으로 PLANSEARCH는 경쟁력 있는 코딩 벤치마크인 LiveCodeBench에서 77.0%라는 최신 최고 성과를 달성합니다.

- **Performance Highlights**: PLANSEARCH는 Claude 3.5 Sonnet 위에 적용했을 때, 기존의 표준 샘플링 방법들보다 더욱 다양하고 효과적인 코드를 생성하여 200회의 통과율(pass@200)에서 77.0%를 기록했습니다. 이는 비검색 방식에서 얻은 최고 점수(41.4%) 보다 크게 향상된 결과로, 검색 알고리즘 덕분에 성능 향상이 다양성에 의해 직결된다는 점을 강조합니다.



### A Fused Large Language Model for Predicting Startup Success (https://arxiv.org/abs/2409.03668)
- **What's New**: 이 논문에서는 스타트업 투자 결정을 지원하기 위해 기계학습(machine learning) 접근법을 개발했습니다. 특히, 우리는 스타트업의 자기 서술(textual self-descriptions)과 기본 변수(fundamental variables)를 결합하여 성공을 예측하는 맞춤형 대규모 언어 모델(fused large language model)을 제안합니다.

- **Technical Details**: 우리의 접근법은 20,172개의 Crunchbase 온라인 프로파일을 기반으로 하여 평가되었습니다. 이를 통해 기본 변수만을 사용했을 때의 균형 정확도(balanced accuracy)가 72.00%였고, 텍스트 자기 서술을 추가했을 때 74.33%로 증가하는 것을 발견했습니다. 이는 텍스트 자기 서술이 스타트업 성공 예측에 중요한 역할을 한다는 것을 의미합니다.

- **Performance Highlights**: 기계학습 접근법을 통해 텍스트 자기 서술을 포함했을 때 투자 수익률(ROI)이 40.61% 증가하는 것을 보였습니다. 이는 스타트업 성공 예측의 실용적 함의를 강조하며, 스타트업 성공을 나타내는 여러 이벤트에 대한 예측 성능도 높게 평가되었습니다.



### On the Limited Generalization Capability of the Implicit Reward Model Induced by Direct Preference Optimization (https://arxiv.org/abs/2409.03650)
Comments:
          12 pages, 8 tables, 2 figures

- **What's New**: 이번 연구는 인공지능 모델의 인간 선호도 정렬을 위한 보상 모델 학습 방법의 비교를 다루고 있습니다. EXplicit Reward Model (EXRM)과 Direct Preference Optimization (DPO)의 임플리트 보상 모델인 DPORM의 성능 차이를 분석하여, DPORM의 일반화 능력이 떨어짐을 규명했습니다.

- **Technical Details**: 연구는 EXRM과 DPORM의 일반화 능력을 평가하기 위해 다양한 데이터셋에서 이 두 모델을 훈련하고, 성공적으로 구분된 선호 및 거절된 답변에 대한 정확도를 비교했습니다. EXRM은 RLHF 방법론을 사용하여 학습되는 반면, DPORM은 preference 데이터로부터 학습된 임플리트 보상 모델입니다.

- **Performance Highlights**: 다섯 개의 분포 외 평가 세트(Out-of-Distribution settings)에서 DPORM은 평균 3%의 정확도 저하를 보였고, 최대 7%까지 저하되었습니다. 이는 DPORM이 일반화 능력이 제한적임을 나타내며, EXRM의 필요성을 강조하고 있습니다.



### CDM: A Reliable Metric for Fair and Accurate Formula Recognition Evaluation (https://arxiv.org/abs/2409.03643)
Comments:
          Project Website: this https URL

- **What's New**: 본 논문은 수식 인식의 평가에 있어 기존의 문제점을 해결하기 위한 새로운 메트릭인 Character Detection Matching (CDM)을 제안합니다. CDM은 이미지 기반의 테스트를 통해 수식 인식 평가 객관성을 제고하고, LaTeX 레벨이 아닌 이미지 레벨에서 평가 지표를 설계하여 기존 텍스트 기반 평가 방식의 한계를 극복합니다.

- **Technical Details**: CDM은 예측된 LaTeX와 실제 LaTeX 수식을 이미지화하여, 시각적 특성 추출 및 위치 인식 기술을 활용해 문자 수준에서 정밀하게 매칭합니다. 이러한 방식은 공간 정보(spatial position information)를 포함하여 시각적으로 인지하기 쉬운 평가를 가능하게 합니다. 이는 기존의 BLEU 및 Edit Distance와 같은 텍스트 기반 매칭 방식에 비해 더 신뢰할 수 있는 비교를 제공합니다.

- **Performance Highlights**: 실험 결과, CDM은 다양한 수식 인식 모델을 평가하는 데 있어 인간 평가 기준과의 정렬이 높고, 다양한 수식 표현으로 인해 발생하는 불일치를 제거하여 더 공정한 모델 비교를 제공합니다.



### From MOOC to MAIC: Reshaping Online Teaching and Learning through LLM-driven Agents (https://arxiv.org/abs/2409.03512)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 기반의 다중 에이전트 시스템을 활용하여 MAIC(대규모 AI 기반 강좌)를 제안합니다. MAIC는 교육 과정 준비, 강의 및 분석 단계를 통합하여 온라인 교육의 확장성과 적응성을 동시에 지원하는 새로운 온라인 교육 모델입니다.

- **Technical Details**: MAIC는 LLM에 의해 구동되는 지능형 에이전트를 통해 교수 및 학습 과정을 지원합니다. 이는 교육 자료의 업로드 및 프레젠테이션 생성, 학습 데이터를 분석하고 학업 결과를 예측하는 것과 같은 여러 기능을 포함합니다. MAIC 플랫폼은 과거 3개월간 중국의 칭화대학에서 500명 이상의 학생 자원자와 함께 실험되었으며, 100,000개의 학습 데이터 기록을 수집했습니다.

- **Performance Highlights**: 초기 분석 결과, MAIC는 학생의 상호작용에 따라 교수 과정을 동적으로 조정하며, 개별 학습 요구에 맞춘 다양한 학습 환경을 제공하는 것으로 나타났습니다. 이 연구는 사용자 친화적이고 직관적인 솔루션을 제공하여, 교육자와 학생 모두의 요구를 충족시키는 것을 목표로 하고 있습니다.



### iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models (https://arxiv.org/abs/2409.03284)
Comments:
          Accepted at The International Web Information Systems Engineering conference (the WISE conference) 2024

- **What's New**: 본 논문에서는 iText2KG라는 방법을 제안합니다. 이는 점진적이고 주제 독립적인 Knowledge Graph (KG) 구축을 위한 제로샷(zero-shot) 기법을 적용하여 포괄적인 KG 생성을 가능하게 합니다.

- **Technical Details**: iText2KG는 네 가지 모듈로 구성됩니다: 1) Document Distiller는 원문 문서를 LLM을 활용하여 사전 정의된 의미 블록으로 형식화합니다. 2) iEntities Extractor는 의미 블록 내에서 고유한 개체를 식별하고 모호성을 해결합니다. 3) iRelation Extractor는 확인된 개체와 의미 블록을 바탕으로 고유한 관계를 탐지합니다. 4) 마지막으로 Graph Integrator는 Neo4j를 사용하여 관계와 개체를 시각적으로 표현합니다.

- **Performance Highlights**: 이 방법은 세 가지 시나리오에서 기존의 방법들과 비교하여 우수한 성능을 보여주었습니다: 과학 논문의 그래프 변환, 웹사이트 그래프 변환, 이력서 그래프 변환.



### ChartMoE: Mixture of Expert Connector for Advanced Chart Understanding (https://arxiv.org/abs/2409.03277)
- **What's New**: 이번 논문에서는 다중 모달 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 차트 이해 능력을 향상시키기 위한 새로운 접근 방식인 ChartMoE를 제안합니다. 본 연구는 전통적인 선형 프로젝터를 Mixture of Experts (MoE) 아키텍처로 대체하여 모달리티 갭을 메우기 위한 다양한 정렬(Task) 작업을 통해 여러 선형 커넥터를 훈련시키는 방법을 소개합니다.

- **Technical Details**: ChartMoE는 차트-테이블-JSON-코드 쌍으로 구성된 90만 개 이상의 쿼드러플을 포함하는 ChartMoE-Align 데이터셋을 활용하여 각각의 협업 방식으로 초기화된 여러 전문가를 생성합니다. 이를 통해 다단계 훈련 구조를 설계하여 차트 이해 성능을 극대화합니다. 또한, 논문에서는 높은 품질의 지식 학습과 최적화 기법을 도입하여 각 전문가들의 성능을 제고합니다.

- **Performance Highlights**: ChartMoE는 ChartQA 벤치마크에서 이전 최고 성능인 80.48%를 84.64%로 향상시키는 등, 여러 벤치마크에서 큰 폭으로 성능을 향상시키는 결과를 보였습니다. 이러한 결과는 기술적 혁신이 차트 해석과 근거 있는 이해를 더욱 향상시킬 수 있음을 보여줍니다.



### Strategic Chain-of-Thought: Guiding Accurate Reasoning in LLMs through Strategy Elicitation (https://arxiv.org/abs/2409.03271)
- **What's New**: 이번 논문에서는 Chain-of-Thought (CoT) 방법론의 한계점을 극복하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 Strategic Chain-of-Thought (SCoT)로, LLM의 성능을 개선하기 위해 전략적 지식을 통합하여 생성 과정에서 중간 추론 단계를 정교화합니다.

- **Technical Details**: SCoT는 두 단계의 절차를 포함합니다. 첫 단계에서는 효과적인 문제 해결 전략을 도출하고, 그 다음 해당 전략을 활용하여 높은 품질의 CoT 경로와 최종 답변을 생성합니다. 이러한 과정은 단일 프롬프트로 수행되며, 불필요한 다중 질의 및 추가 지식 통합을 줄입니다.

- **Performance Highlights**: SCoT 방법의 실험 결과는 GSM8K 데이터 세트에서 21.05%의 정확도 증가와 Tracking_Objects 데이터 세트에서 24.13%의 정확도 향상을 보여주었습니다. 이는 SCoT의 효과성을 강조하며, 여러 모델에서 복잡한 추론 작업에 대한 성능 향상을 증명합니다.



### Continual Skill and Task Learning via Dialogu (https://arxiv.org/abs/2409.03166)
- **What's New**: 로봇이 자연어 대화를 통해 새로운 기술을 능동적으로 배울 수 있도록 하는 프레임워크를 제안합니다. 기존 연구들은 로봇의 능동적인 질문과 사용자와의 대화를 통해 새로운 기술을 배우는 방법을 탐구하지 않았습니다.

- **Technical Details**: 이 연구는 ACT-LoRA라는 새로운 비주얼-모터 제어 정책을 제안하며, 사용자와의 대화 및 언어-기술 접지 임베딩을 통해 로봇이 기술과 작업 관련 정보를 쿼리할 수 있게 합니다. 로봇은 5회의 시연만으로도 100%의 정확도로 새로운 기술을 배우며, RLBench 데이터셋에서 사전 훈련된 기술의 정확도는 74.75%입니다.

- **Performance Highlights**: 인간 참가자 8명을 대상으로 수행된 연구에서는 샌드위치 만들기 작업에서 75%의 성공률을 달성했습니다. 이는 로봇이 비전문 사용자와의 대화를 통해 새로운 기술을 배울 수 있음을 보여줍니다.



### GraphEx: A Graph-based Extraction Method for Advertiser Keyphrase Recommendation (https://arxiv.org/abs/2409.03140)
- **What's New**: 새로운 키프레이즈 추천 알고리즘인 GraphEx를 도입하여, 전통적인 XMC 모델의 한계를 극복하고 자원의 제약이 있는 환경에서도 실시간으로 작동할 수 있도록 개발했습니다. 이 알고리즘은 상품 제목에서 키프레이즈의 토큰 순열을 추출하는 그래프 기반 접근 방식을 사용합니다.

- **Technical Details**: GraphEx는 키프레이즈 추천을 위한 혁신적인 그래프 기반 알고리즘으로, 비인기 아이템 및 머리 키프레이즈에 대한 편향을 줄이도록 설계되었습니다. 기계 학습 모델은 매출 데이터에 기반하여 훈련되며, 이러한 모델은 실시간으로 동작하고 대규모 아이템(수억 개)을 처리할 수 있습니다.

- **Performance Highlights**: GraphEx는 eBay의 기존 모델보다 우수한 성능을 보이며 키프레이즈 관련 지표와 바이어 도달 가능성을 평가하는 새로운 메트릭들을 제안합니다. 그 결과, 키프레이즈가 아이템과 얼마나 적합한지를 직접적으로 평가할 수 있는 평가 프레임워크를 제공합니다.



### Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA) (https://arxiv.org/abs/2409.03131)
- **What's New**: 이 논문은 Large Language Models (LLM)에 대한 새로운 적대적 공격 방법인 Single-Turn Crescendo Attack (STCA)을 탐구합니다. STCA은 전통적인 multi-turn 적대적 전략과는 다르게 단일 상호작용으로 공격을 압축하는 방법을 제시합니다.

- **Technical Details**: STCA는 기존의 multi-turn crescendo attack을 기반으로 하며, 특정한 프롬프트를 통해 확장된 대화를 시뮬레이션하여 일반적인 content moderation 시스템을 우회하여 필터링되지 않은 응답을 생성합니다.

- **Performance Highlights**: 이 논문에서 비유되어진 몇 가지 사례 연구를 통해 현재 LLM의 취약점을 강조하고, 더 강력한 안전 장치의 필요성을 부각시킵니다. 이러한 방법은 기존 문헌에서 다루어지지 않아, 분야에 대한 새로운 기여로 여겨집니다.



### Pooling And Attention: What Are Effective Designs For LLM-Based Embedding Models? (https://arxiv.org/abs/2409.02727)
Comments:
this https URL

- **What's New**: 본 논문은 LLM(대형 언어 모델) 기반의 임베딩 모델의 다양한 풀링(pulling) 및 어텐션(attention) 전략의 성능을 비교하기 위해 대규모 실험을 수행한 결과를 제시합니다. 특히 단일 LLM 모델과 동일한 학습 데이터를 사용하여 여러 설계 방안을 평가하고, Multi-Layers Trainable Pooling이라는 새로운 풀링 전략을 소개합니다.

- **Technical Details**: LLM 기반 임베딩 모델에서는 풀링 전략이 입력 시퀀스를 고정 크기의 밀집 벡터로 변환하며, 두 가지 주요 설계인 풀링과 어텐션이 활용됩니다. 기존 연구들은 LLM의 마지막 레이어만을 활용하는 경향이 있었으나, 본 연구는 모든 은닉 레이어의 출력을 변환하여 사용하는 Multi-Layers Trainable Pooling을 제안하여 각각의 레이어가 서로 다른 정보를 인코딩할 수 있다는 점을 강조합니다.

- **Performance Highlights**: 실험 결과에 따르면, bidirectional attention 및 추가적인 학습 가능한 풀링 레이어가 텍스트 유사성 및 정보 검색 작업에서 뛰어난 성능을 보였으나, 클러스터링 및 분류 작업에서는 더 간단한 EOS-last token pooling과 기본 causal attention으로도 충분한 성능을 보여줍니다. 제안하는 Multi-Layers Trainable Pooling 방법은 기존 풀링 방법에 비해 통계적으로 우수한 성능을 입증하였습니다.



### Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation (https://arxiv.org/abs/2409.01586)
- **What's New**: 이 논문에서는 모델 가중치에 대한 해로운 섭동(harmful perturbation)이 해로운 미세 조정(harmful fine-tuning) 문제의 근본 원인이라고 명시하였습니다. 이러한 문제를 완화하기 위해 'Booster'라는 정렬 단계 솔루션을 제안합니다.

- **Technical Details**: Booster는 원래의 정렬 손실(alignment loss)에 손실 정규화기(loss regularizer)를 추가하여 최적화 과정에서 해로운 섭동의 부정적인 영향을 줄이도록 설계되었습니다. 이 정규화기는 해로운 데이터셋을 사용하여 모델의 손실 감소를 제어합니다. Booster는 반복적 경량법(iterative gradient method)으로 문제를 해결합니다.

- **Performance Highlights**: Booster는 기존 솔루션인 Vaccine과 RepNoise에 비해 각각 17.26%와 20.08%의 평균 해로운 점수를 감소시키면서도 다운스트림 작업의 성능을 유지하는 효과를 보여주었습니다.



New uploads on arXiv(cs.IR)

### HGAMN: Heterogeneous Graph Attention Matching Network for Multilingual POI Retrieval at Baidu Maps (https://arxiv.org/abs/2409.03504)
Comments:
          Accepted by KDD'21

- **What's New**: 이번 연구에서는 다국어 POI(관심 지점) 검색을 위한 새로운 네트워크인 Heterogeneous Graph Attention Matching Network(HGAMN)를 제안합니다. 이 네트워크는 Baidu Maps에서의 검색 로그 데이터를 기반으로 하여 POI와 쿼리 간의 효과적인 매칭을 가능하게 합니다.

- **Technical Details**: HGAMN은 두 가지 유형의 노드(POI 노드와 쿼리 노드)를 포함하는 이질적인 그래프를 구축합니다. 이 그래프를 통해 저빈도 POI와 고빈도 POI 사이의 지식을 전이하며, 다양한 언어의 쿼리와 POI의 공존을 기반으로 엣지를 구성합니다. 또한, 주의(attention) 기반 네트워크를 설계하여 노드 표현을 학습하고, 크로스 주의(cross-attention) 모듈을 통해 두 유형 노드의 표현을 융합하여 검색 관련성을 평가합니다.

- **Performance Highlights**: HGAMN은 실제 Baidu Maps에서 운영 중이며, 매일 수억 건의 검색 요청을 소화하는 데 성공했습니다. 대규모 데이터셋을 활용한 실험에서 HGAMN은 기존의 여러 방법에 비해 유의미한 성능 향상을 보여줬습니다.



### MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu's Sponsored Search (https://arxiv.org/abs/2409.03449)
Comments:
          Accepted by KDD'19

- **What's New**: Baidu의 최신 광고 검색 시스템인 Mobius 프로젝트는 높은 효율의 광고 검색 엔진을 구축하기 위한 첫 번째 시도로, 쿼리-광고 관련성 외에 CPM을 추가 최적화 목표로 삼고 있다.

- **Technical Details**: Mobius-V1은 액티브 러닝(active learning) 기법을 사용하여 클릭 기록의 부족 문제를 극복하고, 키워드-광고 쌍으로부터 CTR(클릭률)을 직접 예측하는 신경 클릭 네트워크(neural click network)를 훈련한다. 또한, 최신 SOTA(최첨단) ANN(근사 가장 가까운 이웃) 검색 기술을 활용하여 광고 검색을 더욱 효율적으로 수행한다.

- **Performance Highlights**: Mobius-V1은 수십억 쿼리-광고 쌍에 대해 CTR을 정확하고 신속하게 예측하며, 이 시스템의 배포를 통해 Baidu의 광고 수익성 향상에 기여할 것으로 기대된다.



### Federated Prototype-based Contrastive Learning for Privacy-Preserving Cross-domain Recommendation (https://arxiv.org/abs/2409.03294)
- **What's New**: 제안된 FedPCL-CDR 방법은 교차 도메인 추천(CDR)에서 사용자 프라이버시를 보호하면서 지식 전이를 개선하는 새로운 접근 방식입니다. 이 방법은 지역 도메인 학습과 글로벌 서버 집계를 포함하여, 교차 도메인 성능과 사용자 프라이버시를 동시에 고려합니다.

- **Technical Details**: FedPCL-CDR은 대칭적으로 클러스터링된 사용자 데이터를 사용해 대표적인 프로토타입을 학습하고, 글로벌 서버에서 수집된 프로토타입으로 지식 전이를 수행합니다. 이 모델은 지역 도메인에서 사용자 및 아이템의 임베딩을 학습하고, k-means 클러스터링을 통해 프로토타입(클러스터 중심)을 생성합니다. 지식 전이는 로컬 및 글로벌 프로토타입을 통해 Contrastive Learning (CL) 방식으로 이루어집니다.

- **Performance Highlights**: FedPCL-CDR은 Amazon과 Douban의 실제 데이터셋을 사용한 4개의 CDR 작업에서 기존 최첨단 접근 방식에 비해 뛰어난 성능을 보여주었습니다. 특히 희소한 겹치는 사용자 환경에서도 기존 방법보다 향상된 추천 성능을 발휘했습니다.



### GraphEx: A Graph-based Extraction Method for Advertiser Keyphrase Recommendation (https://arxiv.org/abs/2409.03140)
- **What's New**: 새로운 키프레이즈 추천 알고리즘인 GraphEx를 도입하여, 전통적인 XMC 모델의 한계를 극복하고 자원의 제약이 있는 환경에서도 실시간으로 작동할 수 있도록 개발했습니다. 이 알고리즘은 상품 제목에서 키프레이즈의 토큰 순열을 추출하는 그래프 기반 접근 방식을 사용합니다.

- **Technical Details**: GraphEx는 키프레이즈 추천을 위한 혁신적인 그래프 기반 알고리즘으로, 비인기 아이템 및 머리 키프레이즈에 대한 편향을 줄이도록 설계되었습니다. 기계 학습 모델은 매출 데이터에 기반하여 훈련되며, 이러한 모델은 실시간으로 동작하고 대규모 아이템(수억 개)을 처리할 수 있습니다.

- **Performance Highlights**: GraphEx는 eBay의 기존 모델보다 우수한 성능을 보이며 키프레이즈 관련 지표와 바이어 도달 가능성을 평가하는 새로운 메트릭들을 제안합니다. 그 결과, 키프레이즈가 아이템과 얼마나 적합한지를 직접적으로 평가할 수 있는 평가 프레임워크를 제공합니다.



### WildVis: Open Source Visualizer for Million-Scale Chat Logs in the Wild (https://arxiv.org/abs/2409.03753)
- **What's New**: WildVis는 대규모 대화 데이터셋을 효율적으로 분석할 수 있는 새로운 인터랙티브 툴입니다. 이 도구는 사용자-챗봇 상호작용을 신속하고 다양한 방식으로 분석하는 기능을 제공합니다.

- **Technical Details**: WildVis는 필터 기반 검색 시스템과 임베딩 기반 시각화 모듈로 구성되어 있으며, 수십만 건의 대화를 빠르게 검색하고 시각화할 수 있습니다. 이를 위해 Elasticsearch를 활용하여 검색 인덱스를 구성하고, OpenAI의 text-embedding-3-small 모델을 사용하여 대화의 첫 번째 사용자 발화를 임베딩합니다.

- **Performance Highlights**: WildVis는 대화 데이터의 시각화 및 검색을 지원하며, 사례 연구를 통해 챗봇 남용 연구 촉진, 데이터셋 간 주제 분포 비교, 사용자 특화 대화 패턴을 파악하는 데 기여합니다. 현재 WildVis는 오픈소스로 제공되며, MIT 라이센스 하에 이용 가능합니다.



### RAG based Question-Answering for Contextual Response Prediction System (https://arxiv.org/abs/2409.03708)
Comments:
          Accepted at the 1st Workshop on GenAI and RAG Systems for Enterprise, CIKM'24. 6 pages

- **What's New**: 본 논문은 RAG (Retrieval Augmented Generation) 기술을 활용하여 대형 소매업체의 고객 센터에 적합한 질문-응답 시스템을 구현하는 방법론을 제안합니다. LLM (Large Language Model)을 사용하여 고객 질문에 대한 정확하고 관련성 있는 응답을 생성하는 최적의 방법을 모색하고 있습니다.

- **Technical Details**: 이 연구에서 제안하는 시스템은 고객의 질문을 바탕으로 연관된 지식 문서를 검색하고, 이전 대화 기록과 함께 이를 활용하여 고객 서비스 상담원에게 응답 제안을 생성하는 구조입니다. RAG 아키텍처를 통해 사용자 입력을 처리하여 관련 문서를 검색하고, 이 문서들을 기반으로 최종 예측을 생성합니다. 이를 통해 특히 불확실성을 줄이고 비즈니스 성과를 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: 이 솔루션은 기존 BERT 기반 알고리즘에 비해 정확성과 관련성에서 더 나은 성과를 보였으며, 고객 서비스 담당자에게 우수한 지원을 제공한다는 것을 입증했습니다. 특히, LLM의 hallucination(허상) 문제를 줄이고 응답의 정확성을 크게 향상시켰습니다.



### iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models (https://arxiv.org/abs/2409.03284)
Comments:
          Accepted at The International Web Information Systems Engineering conference (the WISE conference) 2024

- **What's New**: 본 논문에서는 iText2KG라는 방법을 제안합니다. 이는 점진적이고 주제 독립적인 Knowledge Graph (KG) 구축을 위한 제로샷(zero-shot) 기법을 적용하여 포괄적인 KG 생성을 가능하게 합니다.

- **Technical Details**: iText2KG는 네 가지 모듈로 구성됩니다: 1) Document Distiller는 원문 문서를 LLM을 활용하여 사전 정의된 의미 블록으로 형식화합니다. 2) iEntities Extractor는 의미 블록 내에서 고유한 개체를 식별하고 모호성을 해결합니다. 3) iRelation Extractor는 확인된 개체와 의미 블록을 바탕으로 고유한 관계를 탐지합니다. 4) 마지막으로 Graph Integrator는 Neo4j를 사용하여 관계와 개체를 시각적으로 표현합니다.

- **Performance Highlights**: 이 방법은 세 가지 시나리오에서 기존의 방법들과 비교하여 우수한 성능을 보여주었습니다: 과학 논문의 그래프 변환, 웹사이트 그래프 변환, 이력서 그래프 변환.



### Do We Trust What They Say or What They Do? A Multimodal User Embedding Provides Personalized Explanations (https://arxiv.org/abs/2409.02965)
- **What's New**: 이 연구에서는 Contribution-Aware Multimodal User Embedding (CAMUE)이라는 새로운 프레임워크를 제안하여 소셜 네트워크에서 사용자 텍스트 정보와 그래프 구조 정보를 통합하는 방법을 개선하고, 각 모드가 사용자 예측에 미치는 기여도를 명확히 설명할 수 있도록 합니다.

- **Technical Details**: CAMUE는 학습 가능한 attention 모듈을 활용하여 사용자 속성 예측에 있어 텍스트 정보와 그래프 구조 정보 중 어떤 것을 신뢰해야 할지를 판단합니다. 이는 개인화된 설명과 추천을 가능하게 하는 기여도 맵을 출력합니다.

- **Performance Highlights**: 실험 결과, 대부분의 사용자에게서 인터랙션 그래프 정보가 텍스트 정보보다 더욱 신뢰성이 높다는 것을 발견했습니다. CAMUE는 다양한 사용자 속성 예측 작업에서도 우수한 성과를 보여주었습니다.



### Pooling And Attention: What Are Effective Designs For LLM-Based Embedding Models? (https://arxiv.org/abs/2409.02727)
Comments:
this https URL

- **What's New**: 본 논문은 LLM(대형 언어 모델) 기반의 임베딩 모델의 다양한 풀링(pulling) 및 어텐션(attention) 전략의 성능을 비교하기 위해 대규모 실험을 수행한 결과를 제시합니다. 특히 단일 LLM 모델과 동일한 학습 데이터를 사용하여 여러 설계 방안을 평가하고, Multi-Layers Trainable Pooling이라는 새로운 풀링 전략을 소개합니다.

- **Technical Details**: LLM 기반 임베딩 모델에서는 풀링 전략이 입력 시퀀스를 고정 크기의 밀집 벡터로 변환하며, 두 가지 주요 설계인 풀링과 어텐션이 활용됩니다. 기존 연구들은 LLM의 마지막 레이어만을 활용하는 경향이 있었으나, 본 연구는 모든 은닉 레이어의 출력을 변환하여 사용하는 Multi-Layers Trainable Pooling을 제안하여 각각의 레이어가 서로 다른 정보를 인코딩할 수 있다는 점을 강조합니다.

- **Performance Highlights**: 실험 결과에 따르면, bidirectional attention 및 추가적인 학습 가능한 풀링 레이어가 텍스트 유사성 및 정보 검색 작업에서 뛰어난 성능을 보였으나, 클러스터링 및 분류 작업에서는 더 간단한 EOS-last token pooling과 기본 causal attention으로도 충분한 성능을 보여줍니다. 제안하는 Multi-Layers Trainable Pooling 방법은 기존 풀링 방법에 비해 통계적으로 우수한 성능을 입증하였습니다.



New uploads on arXiv(cs.CV)

### Lexicon3D: Probing Visual Foundation Models for Complex 3D Scene Understanding (https://arxiv.org/abs/2409.03757)
Comments:
          Project page: this https URL , Github: this https URL

- **What's New**: Lexicon3D라는 새로운 프로빙 아키텍처를 통해 시각적 인코더의 성능을 평가하고, 이미지 기반, 비디오 기반, 3D 기반의 다양한 모델의 강점과 한계를 식별합니다.

- **Technical Details**: 본 연구에서는 7개의 시각적 개념 모델을 평가하며, 각 모델의 기능적 강점 및 약점을 정의하기 위해 Vision-Language Scene Reasoning, Visual Grounding, Segmentation 및 Registration의 4가지 작업을 사용했습니다.

- **Performance Highlights**: DINOv2가 우수한 성능을 보이며, 비디오 모델은 객체 수준의 작업에서 뛰어난 결과를 나타냅니다. 또한, 확산 모델은 기하학적 작업에 이점이 있음을 보여주고, 언어로 사전 훈련된 모델은 언어 관련 작업에서 예상치 못한 한계를 드러냅니다.



### DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation (https://arxiv.org/abs/2409.03755)
Comments:
          Accepted by ECCV 2024

- **What's New**: 새로운 fast DPM 샘플러인 DC-Solver가 제안되어, dynamic compensation (DC)을 활용하여 predictor-corrector 샘플러의 misalignment 문제를 완화한다.

- **Technical Details**: DC-Solver는 샘플링 단계에 적응하는 보상 비율(compensation ratios)을 통해 정확한 샘플링 경로를 추적하며, cascade polynomial regression (CPR)을 통해 보상 비율을 즉시 예측할 수 있도록 설계되었다. 이는 단 10개의 데이터 포인트만으로 최적화할 수 있다.

- **Performance Highlights**: DC-Solver는 다양한 해상도에서 이전 방법들보다 샘플링 품질을 지속적으로 개선하였으며, 무조건적 샘플링에서는 10.38 FID(NFE=5)를, Stable-Diffusion-2.1에서는 0.394 MSE(NFE=5, CFG=7.5)를 달성하였다.



### Foundation Model or Finetune? Evaluation of few-shot semantic segmentation for river pollution (https://arxiv.org/abs/2409.03754)
Comments:
          Accepted at ECCV 2024 Green Foundation Models workshop

- **What's New**: 이번 연구에서는 새롭게 제안된 RIPTSeg 데이터셋을 활용하여 Foundation Models (FMs)와 파인튜닝된 사전 훈련 모델을 비교했습니다. 연구 결과, 데이터가 부족한 경우에도 파인튜닝한 YOLOv8 모델이 FMs보다 더 나은 성능을 보였음을 확인했습니다.

- **Technical Details**: RIPTSeg 데이터셋은 전세계 오염된 강의 고해상도 이미지와 쓰레기 영역을 식별하는 고품질 세분화 마스크로 구성되어 있으며, 300장의 이미지가 수록되어 있습니다. 이 데이터셋을 활용하여 PerSAM 및 SegGPT와 같은 두 개의 FMs 모델을 평가하였고, YOLOv8 파인튜닝 모델과의 비교를 통해 원인을 분석하였습니다.

- **Performance Highlights**: YOLOv8 모델은 최소 30개의 RIPTSeg 이미지로 파인튜닝될 경우, 테스트한 모든 모델 중에서 뛰어난 성능을 보여 FMs를 대체할 수 있는 더욱 출중한 선택임을 증명했습니다.



### ArtiFade: Learning to Generate High-quality Subject from Blemished Images (https://arxiv.org/abs/2409.03745)
- **What's New**: 이번 논문에서는 ArtiFade라는 새로운 모델을 소개하는데, 이는 결함 있는 이미지 데이터셋에서 아티팩트가 없는 고품질 이미지를 생성하는 데 초점을 맞추고 있습니다. 기존의 주제 기반 텍스트-이미지 생성 방법들이 결함 없는 고품질 이미지를 요구해왔던 점을 고려할 때, ArtiFade는 이를 개선하여 결함이 있는 이미지로부터 효과적으로 주제를 학습하고 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ArtiFade는 미세 조정된 텍스트-이미지 모델을 사용하여 아티팩트를 제거하는데, 이를 위해 결함 없는 이미지와 그에 상응하는 결함 이미지의 쌍을 구성하는 특별한 데이터셋이 이용됩니다. 또한, 아티팩트 없는 임베딩(artifact-free embedding)을 텍스트 공간에 추가하여 훨씬 더 나은 주제 구성이 가능하게 하였습니다. 이러한 방법은 확산 모델(difussion model)의 본래 생성 능력을 보존하면서 아티팩트를 제거하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, ArtiFade는 기존 방법들보다 일관되게 뛰어난 성능을 보이며, 다양한 아티팩트가 포함된 결함 이미지에서도 효과적인 성과를 보입니다. 특히, 새로운 평가 기준을 통해 실험한 결과, 본 모델은 훈련 데이터와는 다른 다채로운 아티팩트의 관리 측면에서도 우수한 일반화 능력을 보여주었습니다.



### Geometry Image Diffusion: Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation (https://arxiv.org/abs/2409.03718)
Comments:
          11 pages, 9 figures, Project page: this https URL

- **What's New**: 이번 논문에서 소개하는 Geometry Image Diffusion (GIMDiffusion)은 기존의 복잡한 3D 구조 없이 2D 이미지를 이용하여 고품질의 3D 객체를 생성할 수 있는 새로운 Text-to-3D 모델입니다. GIMDiffusion은 Collaborative Control 메커니즘을 통합하여, 제한된 3D 훈련 데이터를 가지고도 강력한 일반화를 달성합니다.

- **Technical Details**: GIMDiffusion은 geometry images를 활용하여 3D 모양을 효율적으로 표현하며, 이는 기존의 Text-to-Image 모델(예: Stable Diffusion)의 풍부한 2D prior를 이용하는 것을 가능하게 합니다. geometry images는 생성된 객체를 의미 있는 부분들로 분리할 수 있어, 조작 및 편집이 용이합니다. 또한, 본 모델은 10초 이내에 명확한 3D 메쉬를 생성할 수 있는 속도를 자랑합니다.

- **Performance Highlights**: GIMDiffusion은 각각의 구성 요소를 분리하여 의미를 부여하는 3D 자산을 생성하여, 다양한 환경에서 활용할 수 있는 유연성을 제공합니다. 또한, 이 모델의 결과물인 3D 자산은 조명 효과가 baked-in 되어 있지 않아, 다양한 환경에 적합합니다. 전반적인 워크플로우를 단순화하여 후처리 과정에서 발생할 수 있는 잠재적인 아티팩트를 줄입니다.



### RealisHuman: A Two-Stage Approach for Refining Malformed Human Parts in Generated Images (https://arxiv.org/abs/2409.03644)
- **What's New**: 최근 Diffusion 모델은 Generative Adversarial Networks (GANs)와 같은 전통적인 프레임워크를 초월하는 비주얼 생성 혁신을 이루었습니다. 본 논문에서는 RealisHuman이라는 새로운 포스트 프로세싱(post-processing) 솔루션을 제안하여, 복잡한 구조를 가진 사람의 이미지 생성 시 세밀한 부분 특히 손이나 얼굴의 정교한 재현에 어려움을 극복하고자 합니다.

- **Technical Details**: RealisHuman 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 원래 잘못된 부분을 참조하여 손이나 얼굴과 같은 현실적인 인간 부분을 생성합니다. 이 과정에서 Part Detail Encoder와 DINOv2를 활용하여 세부 정보를 추출하고, 3D 포즈 추정 결과를 이용하여 생성된 인간 부분의 포즈를 정확히 유지합니다. 두 번째 단계에서는 복원된 인간 부분을 원래 이미지에 통합하며, 이 과정은 인페인팅(inpainting) 문제로 접근하여 주변 영역을 재채색하여 부드럽고 자연스럽게 혼합합니다.

- **Performance Highlights**: RealisHuman 프레임워크는 포괄적인 실험을 통해 정성적 및 정량적 지표에서 개선된 성능을 보여줍니다. 이 방법은 다양한 스타일의 이미지에 대해 뛰어난 일반화 능력을 보여주며, 단순히 손뿐만 아니라 다른 인간 부분도 정교하게 수정할 수 있는 혁신적인 접근 방식을 제공합니다.



### CDM: A Reliable Metric for Fair and Accurate Formula Recognition Evaluation (https://arxiv.org/abs/2409.03643)
Comments:
          Project Website: this https URL

- **What's New**: 본 논문은 수식 인식의 평가에 있어 기존의 문제점을 해결하기 위한 새로운 메트릭인 Character Detection Matching (CDM)을 제안합니다. CDM은 이미지 기반의 테스트를 통해 수식 인식 평가 객관성을 제고하고, LaTeX 레벨이 아닌 이미지 레벨에서 평가 지표를 설계하여 기존 텍스트 기반 평가 방식의 한계를 극복합니다.

- **Technical Details**: CDM은 예측된 LaTeX와 실제 LaTeX 수식을 이미지화하여, 시각적 특성 추출 및 위치 인식 기술을 활용해 문자 수준에서 정밀하게 매칭합니다. 이러한 방식은 공간 정보(spatial position information)를 포함하여 시각적으로 인지하기 쉬운 평가를 가능하게 합니다. 이는 기존의 BLEU 및 Edit Distance와 같은 텍스트 기반 매칭 방식에 비해 더 신뢰할 수 있는 비교를 제공합니다.

- **Performance Highlights**: 실험 결과, CDM은 다양한 수식 인식 모델을 평가하는 데 있어 인간 평가 기준과의 정렬이 높고, 다양한 수식 표현으로 인해 발생하는 불일치를 제거하여 더 공정한 모델 비교를 제공합니다.



### Surface-Centric Modeling for High-Fidelity Generalizable Neural Surface Reconstruction (https://arxiv.org/abs/2409.03634)
Comments:
          ECCV 2024 Accepted

- **What's New**: 본 논문은 새로운 Surface-centric 프레임워크인 SuRF를 제안하여, 메모리 제한과 기하학적 세부 정보 복원의 문제를 해결하고자 하였습니다. 특히, 기존의 지도 학습 방식 없이 unsupervised한 end-to-end sparsification을 실현하였습니다.

- **Technical Details**: SuRF는 Matching Field 모듈을 통해 표면 영역을 찾고, voxel에 대한 SDF 값을 예측하는 대신, 시점에서 두 개 이상의 뷰에서 볼 수 있는 표면 영역의 voxel만을 보존합니다. 이를 통해 메모리와 계산 소비를 줄이면서 높은 주파수 특징을 활용할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서 실험한 결과, SuRF는 이전 최첨단 방법들에 비해 46% 이상의 성능 개선을 달성하였고, 메모리 소비는 80% 이상 줄였습니다.



### SegTalker: Segmentation-based Talking Face Generation with Mask-guided Local Editing (https://arxiv.org/abs/2409.03605)
Comments:
          10 pages, 7 figures, 3 tables

- **What's New**: 본 논문에서는 SegTalker라는 새로운 프레임워크를 제안하여 오디오 기반 대화 얼굴 생성에서 입술 움직임과 이미지 텍스쳐를 효과적으로 분리하는 방법을 다룹니다. 이 과정에서 세분화(segmentation)를 중간 표현으로 도입하여 기술적인 도전을 극복합니다.

- **Technical Details**: SegTalker 프레임워크는 세분화된 이미지 마스크를 사용하여 입술 움직임을 이미지 텍스쳐와 분리합니다. 이 과정은 오디오를 기반으로 대화 세분화(talking segmentation)를 생성하고, 이를 사용하여 StyleGAN에 통합하여 비디오 프레임을 합성하는 방식으로 이루어집니다. 제안된 방법은 시멘틱 코드와 마스크를 활용하여 다양한 얼굴 지역에서 텍스쳐를 수정하고, 배경 교체를 수월하게 할 수 있습니다.

- **Performance Highlights**: HDTF 및 MEAD 데이터셋을 기반으로 한 실험에서 SegTalker는 기존 방법에 비해 시각적 품질, 신원(id) 보존 및 시간적 일관성을 유지하는 데 있어 뛰어난 성능을 보였습니다.



### TCDiff: Triple Condition Diffusion Model with 3D Constraints for Stylizing Synthetic Faces (https://arxiv.org/abs/2409.03600)
Comments:
          SIBGRAPI 2024

- **What's New**: 이번 논문에서는 Triple Condition Diffusion Model (TCDiff)을 제안하여 실제 얼굴에서 합성 얼굴로의 스타일 전송을 개선합니다. 이는 2D 및 3D 얼굴 제약 조건을 통해 얼굴 정체성 일관성을 높이고, 필요한 높은 클래스 내 분산(intra-class variance)을 유지합니다.

- **Technical Details**: TCDiff 모델은 2D 및 3D의 일관성 제약을 포함하여, 얼굴의 포즈, 표정, 나이, 노이즈, 가리기 등의 스타일 속성을 추출하여 합성 얼굴의 정체성을 강화합니다. 1k, 2k, 5k의 새로운 데이터셋에서 실험한 결과, 기존 합성 데이터셋보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: 모델의 성능은 LFW, CFP-FP, AgeDB, BUPT와 같은 실제 얼굴 벤치마크에서 state-of-the-art 합성 데이터셋을 능가하며, 낮은 클래스에서 얼굴 인식 모델의 품질을 크게 향상시킵니다.



### Text-Guided Mixup Towards Long-Tailed Image Categorization (https://arxiv.org/abs/2409.03583)
Comments:
          Accepted by BMVC'24, code is available at this https URL

- **What's New**: 본 논문은 긴 꼬리(long-tailed) 데이터의 문제를 해결하기 위해 사전 훈련된 비전-언어 모델인 CLIP의 텍스트 인코더를 활용하는 새로운 텍스트 주도 혼합 기술(text-guided mixup technique)을 제안합니다. 이는 소수 클래스의 정보를 활용하여 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: 제안하는 방법인 local feature mixup (LFM)는 텍스트 인코더가 인식한 클래스 간의 의미론적 관계를 활용하여 소수 클래스의 성능을 향상시키고, 이를 통해 긴 꼬리 문제를 완화하고자 합니다. 제안된 기법은 이론적 보장을 가지고 있으며, 다양한 벤치마크 데이터셋에서 유효성 검증을 수행했습니다.

- **Performance Highlights**: 다양한 벤치마크 긴 꼬리 데이터셋에서 우리의 방법이 효과적임을 실험적으로 입증하였으며, 이는 기존의 방법들과 비교하여 상대적으로 더 나은 성능을 보여주었습니다.



### Organized Grouped Discrete Representation for Object-Centric Learning (https://arxiv.org/abs/2409.03553)
- **What's New**: Object-Centric Learning (OCL)의 최근 발전을 다룬 논문에서는 Sparse object features로 이미지를 표현하는 방법을 제안합니다. 기존의 Grouped Discrete Representation (GDR)에서 발생하는 오류를 해결하기 위해 Organized GDR (OGDR)라는 새로운 방법론을 도입하였습니다.

- **Technical Details**: OGDR는 채널을 같은 속성(attribute)에 해당하는 것들끼리 정리하여 최적의 분해(decomposition)를 진행합니다. 이 과정에서 Variational Autoencoder (VAE) 기반의 템플릿 기능(template features)을 활용하여 정보 중복을 줄이고 object-level feature aggregation을 유도합니다.

- **Performance Highlights**: 비지도 세분화(unsupervised segmentation) 실험에서 OGDR는 GDR에 비해 훨씬 우수한 성과를 보였으며, 기존의 transformer 기반 OCL 방법과 최첨단 diffusion 기반 방법의 성능을 향상시켰습니다. Codebook PCA 및 표현 유사성 분석(representation similarity analyses)을 통해 OGDR가 정보 중복을 줄이고 object representation learning을 위한 정보 보존을 개선하였음을 확인하였습니다.



### DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architectur (https://arxiv.org/abs/2409.03550)
- **What's New**: 이 논문에서는 데이터 없는 지식 증류(Data-Free Knowledge Distillation, DKDM)를 제안하여 확산 모델(Diffusion Models, DMs)의 속도를 두 배로 증가시키며, 고품질 샘플을 생성할 수 있는 새로운 방법론을 소개합니다.

- **Technical Details**: DKDM은 두 가지 주요 구성 요소로 이루어져 있습니다: 첫째, 사전 훈련된 DMs가 생성한 합성 데이터(denoising data)를 사용하여 빠른 DMs를 최적화하는 DKDM 목표(Objective)입니다. 둘째, 최적화 과정에서의 병목현상을 방지하기 위해 유연하게 합성 데이터를 조직하는 동적 반복 증류(Dynamic Iterative Distillation) 방법이 포함됩니다. 이 방법은 DMs의 성능을 실제 데이터 없이 향상시키도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, DKDM은 DMs의 생성 속도를 2배 향상시키면서도 기본 성능을 유지하는 것으로 나타났습니다. 또한, 사전 훈련된 DMs가 새로운 DMs 훈련을 위한 "데이터 셋" 역할을 할 수 있음을 보여주어, 데이터 저장 요구사항을 줄이는 데 기여할 수 있습니다.



### Prediction Accuracy & Reliability: Classification and Object Localization under Distribution Shif (https://arxiv.org/abs/2409.03543)
Comments:
          This preprint has not undergone any post-submission improvements or corrections

- **What's New**: 본 논문은 자연 분포 변동이 CNN의 인식 성능에 미치는 영향을 분석한 연구로, 날씨 변화에 따른 데이터 증강이 탐지 품질과 신뢰도 추정에 미치는 영향도 평가합니다.

- **Technical Details**: 자율주행 데이터셋을 기반으로 한 새로운 데이터셋을 구성하였으며, 여섯 가지의 분포 변화 데이터셋은 악천후 시나리오, 시뮬레이션된 비와 안개, 그리고 분포 외 데이터(Out-of-Distribution data)를 포함합니다.

- **Performance Highlights**: ConvNeXt-Tiny는 EfficientNet-B0보다 더 강한 내구성을 보여주며, 폭우가 분류의 성능에 더 큰 영향을 미치는 반면, 안개에서는 상대적으로 덜 영향을 미치는 것으로 나타났습니다. 또한, MC-Dropout을 특정 레이어에 통합할 경우 작업 성능과 신뢰도 추정 개선의 가능성이 있지만, 이 레이어의 선택은 분포 변동의 유형과 고려하는 작업에 따라 달라집니다.



### Use of triplet loss for facial restoration in low-resolution images (https://arxiv.org/abs/2409.03530)
Comments:
          10 pages, 8 figures

- **What's New**: 최근 얼굴 인식(FR) 모델은 여러 데이터셋에서 인상적인 결과를 달성하며 가장 널리 사용되는 생체 인식 도구로 자리잡았습니다. 그러나 저해상도 이미지로 인한 성능 저하 문제가 여전히 존재합니다. 본 논문에서는 FTLGAN이라는 새로운 초해상도(SR) 모델을 제안합니다.

- **Technical Details**: FTLGAN 모델은 개별의 정체성을 보존하는 고해상도 이미지를 생성하는 데 중점을 두며, 기존의 초해상도 모델과 달리 얼굴 인식 품질을 손실 함수(loss)로 통합하여 훈련합니다. 이 모델은 14x14, 28x28, 56x56 픽셀 해상도에서 낮은 해상도의 얼굴 인식 성능을 크게 향상시킵니다.

- **Performance Highlights**: FTLGAN은 d' 값이 1.099, AUC 0.78, d' 2.112, AUC 0.92, d' 3.049, AUC 0.98로 현재의 최첨단 모델보다 평균 21% 향상된 결과를 보여주며, 모든 해상도에서 일관되게 뛰어난 성능을 나타냅니다.



### FrozenSeg: Harmonizing Frozen Foundation Models for Open-Vocabulary Segmentation (https://arxiv.org/abs/2409.03525)
Comments:
          14 pages, 9 figures

- **What's New**: Open-vocabulary segmentation의 도전 과제를 극복하기 위해 새로운 접근 방식인 FrozenSeg를 소개합니다. 이 시스템은 위치 기반 모델(SAM)과 비전 언어 모델(CLIP)을 결합하여 보다 우수한 세그멘테이션 성능을 목표로 합니다.

- **Technical Details**: FrozenSeg는 세 가지 주요 모듈로 구성됩니다. (1) Query Injector: SAM에서 얻은 공간 인식 특징을 활용하여 마스크 영역에 대한 쿼리를 생성합니다. (2) Feature Injector: 각 픽셀의 CLIP 특징에 SAM의 전반적인 공간 정보를 통합하여 CLIP 특징을 보강합니다. (3) OpenSeg Ensemble 모듈: SAM의 공간 정보를 바탕으로 제로샷 마스크 제안을 앙상블하여 마스크 품질을 향상합니다.

- **Performance Highlights**: Extensive experiments demonstrate that FrozenSeg outperforms existing methods across various segmentation benchmarks, with significant improvements in recall metrics on the CityScapes dataset. The PQ metric improved from 44.3 to 45.8, and the mIoU increased from 17.3 to 19.7, validating the effectiveness of the proposed approach.



### Have Large Vision-Language Models Mastered Art History? (https://arxiv.org/abs/2409.03521)
- **What's New**: 최근 대규모 Vision-Language Models (VLMs)의 출현이 여러 도메인에서 이미지 분류의 새로운 기준선을 설정했습니다. 하지만 VLM이 예술 작품 분류, 특히 그림의 예술 스타일 분류와 같은 특정 작업에서의 성능은 아직 연구되지 않았습니다. 이 논문은 대규모 VLM이 과연 그림의 예술적 속성을 효과적으로 예측할 수 있는지를 조사합니다.

- **Technical Details**: 본 논문에서는 CLIP, LLaVA, OpenFlamingo, GPT-4o의 네 가지 VLM에 대한 심층 분석을 실시하며, 두 개의 공공 아트워크 벤치마크를 사용하여 제로샷(zero-shot) 분류를 통해 예술 스타일, 작가 및 시간대를 분석합니다. 또한, 예술 역사학자들이 연구한 주요 그림을 포함한 엄선된 시험 세트 ArTest를 제공합니다.

- **Performance Highlights**: 예술 분류 분야에서 VLM의 성능을 최초로 평가한 연구로, 다양한 예술 작품에 대한 높은 정확도의 분류 성과를 보여줍니다. 이는 예술 역사학자들이 오랜 시간 연구해온 예술 스타일 예측의 문제를 기계 학습을 통해 접근할 수 있다는 가능성을 제시합니다.



### LMLT: Low-to-high Multi-Level Vision Transformer for Image Super-Resolution (https://arxiv.org/abs/2409.03516)
- **What's New**: 이 논문은 Low-to-high Multi-Level Transformer (LMLT)라는 새로운 모델을 제안하여 기존 Vision Transformer (ViT) 기반의 이미지 초해상도(Image Super-Resolution) 방법에서 발생하는 복잡성과 메모리 사용량 문제를 해결합니다.

- **Technical Details**: LMLT는 각 헤드에 대해 다양한 크기의 특성을 사용하는 attention 메커니즘을 통해 이미지를 처리합니다. 모델은 채널 차원에 따라 이미지 특성을 나누고 낮은 헤드에 대해 공간 크기를 점진적으로 감소시킵니다. 각 헤드는 self-attention을 적용하여 지역 정보와 전역 정보를 효과적으로 포착합니다. 이 접근 방식은 window 경계 문제를 해결합니다.

- **Performance Highlights**: LMLT는 기존 ViT 기반 이미지 초해상도 모델들에 비해 메모리 사용량을 각각 38%와 54% 감소시키며, 추론 시간도 각각 22%와 19% 감소시킵니다. 모든 벤치마크 데이터셋에서 평균 성능이 각각 0.076db와 0.152db 향상되었습니다.



### Blended Latent Diffusion under Attention Control for Real-World Video Editing (https://arxiv.org/abs/2409.03514)
- **What's New**: 현재 공개적으로 제공되는 text-to-video 모델의 부족으로 인해, 기존의 비디오 편집 방법은 pretrained text-to-image 생성 모델에 의존하고 있지만, 시간 정보를 포함한 비디오의 로컬 편집에서 여전히 많은 도전에 직면하고 있습니다. 본 논문에서는 Blended Latent Diffusion 모델을 기반으로 로컬 비디오 편집 작업을 수행하는 방법을 제안합니다.

- **Technical Details**: 본 논문의 주요 기술 구성 요소는 다음과 같습니다: 1) DDIM(Deterministic Denoising Implicit Models) 역변환을 통해 배경 정보를 보존하는 배경 잠재(latents)를 얻고, 2) 교차 주의 맵(cross-attention maps)을 활용하여 사용자 제공 마스크 없이 자동으로 마스크를 생성하는 메커니즘을 도입하며, 3) U-Net의 self-attention 블록을 시간-공간 블록으로 변환하여 비디오 프레임 간의 시계열 일관성을 강화합니다.

- **Performance Highlights**: 제안된 방법은 다양한 실제 비디오 편집 작업에서 실험을 통해 효과성을 입증하며, 최신 비디오 편집 방법들과 비교하여 우수한 성능을 보여줍니다.



### Domain-Guided Weight Modulation for Semi-Supervised Domain Generalization (https://arxiv.org/abs/2409.03509)
Comments:
          Accepted at WACV25

- **What's New**: 이 연구는 매우 적은 수의 레이블 데이터로도 새로운 도메인에 일반화할 수 있는 깊은 학습 모델을 개발하는 데 초점을 두고 있습니다. 저자들은 Semi-Supervised Domain Generalization (SSDG) 문제를 해결하기 위해 새로운 방법을 제안하며, 이 방법은 다양한 도메인 변화에 따른 정확한 유사 레이블(Pseudo-label) 생성을 용이하게 합니다.

- **Technical Details**: 제안된 방법은 각 소스 도메인에 대해 도메인 수준의 전문성을 유지하기 위해 분류기의 가중치를 조정하는 도메인 인식 마스크(domain-aware mask)를 학습합니다. 이 과정에서 모델 훈련 중에 도메인 레벨 정보 벡터를 생성하여 이를 통해 가중치 조정을 위한 저차원 분해 요인(mapping from low-rank decomposed factors)으로 학습합니다.

- **Performance Highlights**: 여섯 개의 다양한 데이터셋을 대상으로 한 실험 결과, 제안된 방법은 기존의 SSL 기반 SSDG 기준선보다 상당한 성능 향상을 보여주었습니다. 특히 SSDG 설정에서 다양한 도메인 변화에 대하여 두 가지 다른 설정에서 모두 개선된 성과를 기록하였습니다.



### Towards Data-Centric Face Anti-Spoofing: Improving Cross-domain Generalization via Physics-based Data Synthesis (https://arxiv.org/abs/2409.03501)
Comments:
          Accepted by International Journal of Computer Vision (IJCV) in Sept 2024

- **What's New**: 본 연구는 Face Anti-Spoofing (FAS) 분야에서 데이터 중심의 접근법을 통해 cross-domain 일반화 성능을 향상시키는 데 초점을 맞추고 있습니다. 이는 기존의 모델 중심 접근법에서 벗어나 데이터의 품질과 양을 통해 FAS의 일반화 성능을 향상시키려는 시도를 포함합니다.

- **Technical Details**: 연구에서는 FAS 데이터 증강(FAS-Aug)을 통해 다양한 아티팩트(artifacts) 데이터를 합성하여 데이터의 다양성을 증가시키는 방법을 제안합니다. 또한, Spoofing Attack Risk Equalization (SARE) 기술을 통해 모델이 특정 아티팩트에 의존하지 않도록 하고 일반화 성능을 개선하는 방법도 논의되고 있습니다. 실험에서는 컬러 프로파일을 활용한 색상 다양성 시뮬레이션과 반사 아티팩트의 증강 방식이 포함됩니다.

- **Performance Highlights**: 제안된 FAS-Aug와 SARE를 최근 Vision Transformer backbone과 결합했을 때, FAS 크로스 도메인 일반화 프로토콜에서 최첨단 성과를 기록했습니다.



### ScreenMark: Watermarking Arbitrary Visual Content on Screen (https://arxiv.org/abs/2409.03487)
- **What's New**: 본 연구에서는 Visual Screen Content (VSC) 보호를 위한 새로운 워터마킹 방법인 ScreenMark를 제안합니다. 기존의 워터마킹 기술은 주로 특정 미디어 유형에 맞춰져 있었으나, ScreenMark는 모든 종류의 VSC에 대해 효율적으로 작동하도록 설계되었습니다.

- **Technical Details**: ScreenMark는 세 단계의 점진적인 워터마킹 프레임워크를 사용합니다. 초기 단계에서는 정규 워터마킹 정보와 비정규 워터마킹 패턴 간의 상호 변환이 이루어지며, 이후 이 패턴을 α 블렌딩 기법을 통해 화면 내용에 통합합니다. 마지막으로, 복합 왜곡 처리를 통해 시스템의 복원력을 강화합니다.

- **Performance Highlights**: 100,000개의 스크린샷 이미지를 사용한 광범위한 실험 결과, ScreenMark는 기존의 네 가지 SOTA(single-modal watermarking) 방법과 비교했을 때 안정성, 비가시성 및 실제 적용 가능성에서 우수한 성능을 보였습니다.



### Improving Uncertainty-Error Correspondence in Deep Bayesian Medical Image Segmentation (https://arxiv.org/abs/2409.03470)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이 연구에서는 Accuracy-vs-Uncertainty (AvU) 손실 함수를 사용하여, 불확실성이 정확하지 않은 영역에만 존재하도록 하는 새로운 방법론을 제시합니다. 이는 의료 영상 세분화에서의 자동 오류 탐지를 위해, 불확실성을 최대한 활용하는 방식을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: Deep Bayesian 모델을 기반으로 한 FlipOut 모델을 훈련시키고, AvU 손실을 적용하여 불확실성이 정확한 볼륨(voxel)에서는 감소하고 부정확한 볼륨에서는 유지되도록 합니다. 이를 통해 반자동 품질 평가(quality assessment, QA)의 유용성을 높이고, 의료 영상 세분화의 효율성을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 베이지안 기준선 모델에 비해 정확한 볼륨에서 불확실성을 성공적으로 억제하고, 부정확한 볼륨에서는 유사한 수준의 불확실성을 유지하는 것으로 나타났습니다. 이 연구는 방사선 치료를 포함한 다양한 데이터셋에 걸쳐 성과를 평가하였습니다.



### LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones (https://arxiv.org/abs/2409.03460)
Comments:
          Accepted at WACV 2025. Features 11 pages in total

- **What's New**: 이번 논문에서는 효율적인 비전 백본(vision backbone) 아키텍처의 최적화를 위해 아키텍처 및 구성 요소 측면에서 합리적인 조합을 강조합니다. 특히 메모리 접근 비용(memory access cost)과 병렬 처리 정도(degree of parallelism) 같은 요소를 고려하여 MACs(multiply and accumulate operations) 외에 실제 처리량과 지연 시간(throuput and latency)으로 모델의 효율성을 평가했습니다.

- **Technical Details**: 논문에서 제안하는 LowFormer 아키텍처는 효율적인 MAC 실행을 기반으로, 깊이별 convolution(depthwise convolution)와 포인트별 convolution(pointwise convolution)을 융합하여 실행 시간을 단축하고 정확도를 향상시키는 것을 목표로 합니다. 기존 Multi-Head Self-Attention(MHSA)을 간소화한 방식을 채택하여 전체 모델을 저해상도 입력에서 작동하게 하여 속도 및 정확도를 개선했습니다.

- **Performance Highlights**: LowFormer는 GPU에서 2배의 처리량과 15%의 지연 시간 감소를 달성하며, Top-1 정확도에서는 1% 향상을 보입니다. LowFormer-B3는 가장 높은 복잡성을 가지며, GPU 처리량에서 3배 향상된 성능과 55%의 지연 시간 감소를 기록했습니다. 저자들은 LowFormer를 Semantic FPN 및 RetinaNet에 통합하여 객체 탐지(object detection) 및 의미 집합 분할(semantic segmentation)의 성능을 개선했습니다.



### Non-Uniform Illumination Attack for Fooling Convolutional Neural Networks (https://arxiv.org/abs/2409.03458)
- **What's New**: 본 연구에서는 Non-Uniform Illumination (NUI) 공격 기법을 소개하며, 이를 통해 기존 Convolutional Neural Networks (CNNs)의 취약성을 분석합니다.

- **Technical Details**: NUI 기법을 사용하여 기존 이미지의 다양하게 조정된 NUI 마스크를 적용하여 실험을 진행했습니다. CIFAR10, TinyImageNet, CalTech256과 같은 널리 사용되는 데이터셋에 대해 12개의 다양한 NUI 공격 모델로 이미지 분류를 실행했습니다.

- **Performance Highlights**: 실험 결과, VGG, ResNet, MobilenetV3-small 및 InceptionV3 모델의 분류 정확도가 NUI 공격에 노출되었을 때 크게 감소함을 보여주었습니다. 이를 해결하기 위해, NUI 변환을 통해 생성된 이미지를 학습 세트에 추가하는 방어 전략을 제안하며, 이를 통해 CNN 모델의 성능이 향상되는 것을 관찰했습니다.



### LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors (https://arxiv.org/abs/2409.03456)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 sparse-view 이미지를 활용한 3D 장면 재구성의 새로운 방식인 LM-Gaussian을 제안합니다. 기존의 3D Gaussian Splatting(3DGS) 방식보다 적은 수의 입력 이미지를 사용하여 고품질 재구성을 가능하게 합니다.

- **Technical Details**: LM-Gaussian은 robust initialization을 위한 stereo priors를 활용하며, Background-Aware Depth-guided Initialization 모듈을 통해 초기화 과정에서 생성된 포인트 클라우드를 개선합니다. 또한, Multi-modal Regularized Gaussian Reconstruction 모듈을 통해 과적합(overfitting) 문제를 방지하고, Iterative Gaussian Refinement 모듈을 통해 디퓨전(diffusion) 기반으로 이미지 세부사항을 복원합니다.

- **Performance Highlights**: 다양한 공개 데이터셋을 통해 실험을 진행한 결과, LM-Gaussian은 360도 장면 재구성에서 기존의 접근 방법보다 뛰어난 품질과 세부사항을 유지하는 데 성공하였습니다.



### Data-free Distillation with Degradation-prompt Diffusion for Multi-weather Image Restoration (https://arxiv.org/abs/2409.03455)
- **What's New**: 본 연구에서는 Multi-weather Image Restoration (MWIR)을 위한 새로운 Data-free Distillation 기법인 D4IR (Degradation-prompt Diffusion)를 제안합니다. 기존의 GANs 기반 방법 대신, 사전 학습된 diffusion 모델을 활용하여 불안정한 훈련 문제를 해소하고, 내용 인식형 degradation-aware prompt adapter를 도입하여 도메인 관련 이미지를 생성합니다.

- **Technical Details**: D4IR은 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Degradation-aware Prompt Adapter (DPA): 웹에서 수집한 저해상도 이미지로부터 degradation-aware prompts를 추출합니다. (2) Content-driven Conditional Diffusion (CCD): 클린 이미지를 잠재적 특징으로 변환한 후 degradation 관련 prompts를 조합하여 새로운 도메인 관련 degraded 이미지를 생성합니다. (3) Pixel-wise Knowledge Distillation (PKD): 학생 네트워크가 교사 네트워크의 출력을 모방하도록 최적화됩니다.

- **Performance Highlights**: 실험 결과, 제안된 D4IR 방법은 기존의 원본 훈련 데이터를 사용한 모델과 비슷한 성능을 달성하며, 다른 최신 비지도 방법보다도 뛰어난 성능을 보였습니다.



### Automatic occlusion removal from 3D maps for maritime situational awareness (https://arxiv.org/abs/2409.03451)
Comments:
          Preprint of SPIE Sensor + Imaging 2024 conference paper

- **What's New**: 본 연구는 대규모 해양 환경에서의 점유물 제거를 목표로 하는 새로운 3D 지리공간 모델 업데이트 방법을 소개합니다. 기존 3D 재구성 기술은 동적인 물체에 의해 가려지는 환경을 정확하게 모델링하기 어려워, 이 문제를 해결하기 위해 딥러닝(dDeep Learning) 기술을 활용하여 텍스처(texture)와 기하학(geometry)을 직접 수정합니다.

- **Technical Details**: 이 방법은 인스턴스 세그멘테이션(instance segmentation)과 생성적 인페인팅(generative inpainting) 기술을 결합하여 수행됩니다. 제안하는 방법에서는 기존의 3D 자산을 재가공할 필요 없이 자동으로 가려진 객체를 제거하고 보존된 정적 요소를 유지합니다. 전체 과정은 비디오 분할(tile) 방식으로 처리되며, 사용자는 동적으로 텍스처와 기하학 정보를 수정할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 3D 모델의 신뢰도를 크게 향상시키며, 해양 상황 인식 및 다양한 보조 정보의 동적 표시를 위한 매우 적용 가능성을 확인했습니다. 또한, 현재의 지리공간 표준과의 호환성을 유지하며, 다양한 데이터 세트에서 강력한 성능을 보장합니다.



### Shuffle Vision Transformer: Lightweight, Fast and Efficient Recognition of Driver Facial Expression (https://arxiv.org/abs/2409.03438)
Comments:
          Accepted for publication in The 6th IEEE International Conference on Artificial Intelligence Circuits and Systems (IEEE AICAS 2024), 5 pages, 3 figures

- **What's New**: 본 논문에서는 실시간 애플리케이션에 적합한 드라이버 얼굴 표정 인식(DFER) 시스템을 위해 ShuffViT-DFER이라는 새로운 전이 학습 기반의 이중 아키텍처를 제안합니다. 이 시스템은 경량화된 CNN(Convolutional Neural Network)과 ViT(Vision Transformer) 모델의 강점을 통합하여 계산 효율성과 정확성을 동시에 달성합니다.

- **Technical Details**: ShuffViT-DFER는 ShuffleNet V2와 EfficientViT-M2 아키텍처를 활용하여 특징을 추출하고, MTCNN(Multi-Task Cascaded Convolutional Networks)을 통해 얼굴을 감지합니다. faces는 데이터 증대를 통해 훈련 세트를 확장하고, Dropout과 배치 정규화를 통해 과적합(overfitting)을 방지합니다. 이 두 모델의 특징을 융합하여 분류 정확성을 향상시킵니다.

- **Performance Highlights**: KMU-FED와 KDEF 데이터셋에서 실험 결과, ShuffViT-DFER는 기존 최첨단 방법과 비교하여 우수한 성능을 보여줍니다. 이 모델은 제한된 데이터로도 미세한 얼굴 신호를 효율적으로 포착할 수 있으며, 실제 드라이빙 환경에서도 실시간 처리 성능을 유지합니다.



### UV-Mamba: A DCN-Enhanced State Space Model for Urban Village Boundary Identification in High-Resolution Remote Sensing Images (https://arxiv.org/abs/2409.03431)
Comments:
          5 pages, 4 figures, 2 tables

- **What's New**: 본 논문은 원격 감지 이미지에서 도시 마을 경계를 자동으로 식별하기 위해 UV-Mamba라는 새로운 신경망 모델을 제안합니다.

- **Technical Details**: UV-Mamba는 long sequence modeling에서 발생하는 메모리 손실 문제를 해결하기 위해 deformable convolutions (DCN)을 통합하여 개발되었습니다. 이 모델의 아키텍처는 인코더-디코더 구조를 사용하며, 여러 단계의 의미 정보 추출을 위한 네 개의 deformable state space augmentation (DSSA) 블록을 포함합니다.

- **Performance Highlights**: UV-Mamba는 베이징 및 시안 데이터셋에서 각각 73.3% 및 78.1%의 IoU를 달성했으며, 이전 모델보다 1.2% 및 3.4% 개선된 성과를 보였습니다. 또한, 추론 속도가 6배 더 빠르고, 파라미터 수가 40배 더 적습니다.



### Weight Conditioning for Smooth Optimization of Neural Networks (https://arxiv.org/abs/2409.03424)
Comments:
          ECCV 2024

- **What's New**: 새로운 정규화 기법인 'weight conditioning'을 소개하며, 이는 신경망의 가중치 행렬의 가장 작은 단일값과 가장 큰 단일값 간의 격차를 줄여, 더 나은 조건을 가진 행렬을 생성할 수 있도록 돕습니다.

- **Technical Details**: weight conditioning은 신경망 아키텍처의 가중치 행렬에 일정한 행렬 조정기를 곱함으로써 조건 번호를 최소화하며, 이를 통해 Hessian 행렬이 더 나은 조건을 갖게 하여 경량화된 학습률로 빠른 수렴을 가능하게 합니다.

- **Performance Highlights**: 다양한 신경망 아키텍처에 걸쳐 weight conditioning을 검증한 결과, 기존의 가중치 정규화 기법보다도 뛰어난 성능 향상을 보여주였으며, 특히 GoogleNet과 ResNet 아키텍처에서 약 15%의 정확도 향상을 달성하였습니다.



### mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding (https://arxiv.org/abs/2409.03420)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 논문에서는 고해상도 문서 이미지의 이해를 개선하기 위해 새로운 압축 모듈인 High-resolution DocCompressor를 제안합니다. 이 모듈은 문서 이미지를 324개의 토큰으로 압축하며, 이를 통해 GPU 메모리 사용을 줄이고 인퍼런스 시간을 개선합니다.

- **Technical Details**: High-resolution DocCompressor는 저해상도 글로벌 시각적 특징에 의해 안내되어 고해상도 문서 이미지를 압축합니다. 이를 통해 다중 페이지 문서 이해 능력을 강화하고, 토큰 효율성과 질문-응답 성능 균형을 맞추는 미션을 수행하도록 설계되었습니다. DocOwl2는 세 단계의 훈련 프레임워크(단일 이미지 사전 훈련, 다중 이미지 계속 훈련, 다중 작업 미세 조정)를 통해 개발되었습니다.

- **Performance Highlights**: DocOwl2는 다중 페이지 문서 이해 벤치마크에서 새로운 최첨단 성능을 달성했으며 첫 번째 토큰 대기 시간을 50% 이상 줄였습니다. 또한, 유사한 데이터로 훈련된 단일 이미지 MLLMs와 비교했을 때, 20% 이하의 시각적 토큰으로도 동등한 단일 페이지 이해 성능을 보여줍니다.



### TG-LMM: Enhancing Medical Image Segmentation Accuracy through Text-Guided Large Multi-Modal Mod (https://arxiv.org/abs/2409.03412)
Comments:
          11 pages, 2 figures

- **What's New**: TG-LMM (Text-Guided Large Multi-Modal Model)이라는 새로운 접근법을 제안합니다. 이 모델은 장기의 텍스트 설명을 활용하여 의료 이미지에서 분할 정확도를 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: TG-LMM은 분할 과정에 장기 공간 위치에 대한 전문가 설명과 같은 사전 지식을 통합합니다. 사전 훈련된 이미지 및 텍스트 인코더를 활용하여 훈련 매개변수의 수를 줄이고 훈련 과정을 가속화합니다. 또한 이미지-텍스트 정보 융합 구조를 설계하여 두 가지 데이터 모달리티의 철저한 통합을 보장합니다.

- **Performance Highlights**: TG-LMM은 의료 이미지 분야의 세 가지 권위 있는 데이터셋에서 평가되었으며, MedSAM, SAM, nnUnet과 같은 기존의 접근법에 비해 우수한 성능을 보였습니다.



### KAN See In the Dark (https://arxiv.org/abs/2409.03404)
- **What's New**: 이 논문은 저조도 이미지 향상(low-light image enhancement, LLIE) 작업에 Kolmogorov-Arnold Network (KANs)를 처음으로 도입하고, KAN-Block을 설계하여 기존 방법의 한계를 극복하는 혁신적인 접근을 제안하고 있습니다.

- **Technical Details**: KANs는 스플라인 기반의 컨볼루션 레이어와 학습 가능한 활성화 함수를 특징으로 하며, 비선형 의존성을 효과적으로 포착할 수 있습니다. 저자는 KAN-Block을 설계하여 U-Net 구조에 통합하고, 각 단계에서 이미지를 재구성하며 Fast Fourier Transform (FFT)를 사용하여 주파수 도메인 인식을 도입하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해 저자의 방법이 벤치마크 데이터셋에서 경쟁력 있는 성능을 보였으며, 해석 가능성 및 모델 성능을 모두 개선하였습니다.



### Make Graph-based Referring Expression Comprehension Great Again through Expression-guided Dynamic Gating and Regression (https://arxiv.org/abs/2409.03385)
Comments:
          12 pages to appear in IEEE Transactions on Multimedia

- **What's New**: 이번 연구에서는 기존의 Graph-based 방법의 한계를 극복하기 위한 두 가지 새로운 접근법인 Dynamic Gate Constraint(DGC) 모듈과 Expression-Guided Regression(EGR) 전략을 제안합니다. 이들은 특히 Referring Expression Comprehension(REC) 과제를 보다 효율적으로 해결하는 데 도움을 줍니다.

- **Technical Details**: DGC(module)는 서브 표현(sub-expressions)에 의해 안내되어, 그래프 내에서 관련이 없는 제안과 연결을 비활성화하는 방식으로 작동합니다. EGR(strategy)는 표현과 후보 객체의 정보를 통합하여 경계 상자 예측을 정제합니다.

- **Performance Highlights**: 실험 결과, 제안된 DGC 모듈과 EGR 전략이 적용된 Graph-based 방법이 SOTA transformer 기반 방법보다 뛰어난 성능을 보여주었습니다. 이는 비지도 학습 상황에서도 가능하며, 다양한 벤치마크에서 우수한 결과를 나타냈습니다.



### MouseSIS: A Frames-and-Events Dataset for Space-Time Instance Segmentation of Mic (https://arxiv.org/abs/2409.03358)
Comments:
          18 pages, 5 figures, ECCV Workshops

- **What's New**: 이 논문에서는 새로운 작업인 \emph{space-time instance segmentation}과 이를 위한 데이터셋 \emph{MouseSIS}를 소개합니다. 이 데이터셋은 여러 마리의 생쥐의 상호작용을 기록한 것으로, pixel-level 인스턴스 분할 마스크가 포함되어 있습니다.

- **Technical Details**: 논문에서는 이벤트 카메라를 사용하여 고해상도 및 높은 동적 범위로 빠르고 고난도의 영상 추적이 가능하다는 점을 강조합니다. 또한, \emph{MouseSIS} 데이터셋은 정렬된 그레이스케일 프레임과 이벤트로 구성되어 있으며, 각 생쥐에 대한 고품질 ground-truth 인스턴스 마스크를 제공합니다.

- **Performance Highlights**: 제안된 두 가지 방법은 이벤트 데이터를 활용하여 트래킹 성능을 일관되게 향상시키는 것을 보여주며, 특히 복잡한 조명 조건 및 오클루전 상황에서도 이벤트 기반 추적의 잠재력을 강조합니다.



### Few-Shot Continual Learning for Activity Recognition in Classroom Surveillance Images (https://arxiv.org/abs/2409.03354)
- **What's New**: 이 논문은 실제 교실 환경에서의 활동 인식을 위한 ARIC(상황 인식 데이터셋) 구축을 통해 현재의 한계를 극복하고자 합니다. 특히, 기존의 연구들이 수작업으로 촬영된 비디오에 초점을 맞춘 것과 달리, 교실 감시 이미지에서의 활동 인식에 중점을 두고 있습니다.

- **Technical Details**: ARIC 데이터셋은 다양한 시점에서 촬영된 36,453개의 감시 이미지를 포함하며, 수업 활동 클래스의 불균형, 높은 유사성 및 프라이버시 보호 같은 문제를 다룹니다. 이를 해결하기 위해, 본 연구에서는 지도 대조 학습(Supervised Contrastive Learning, SCL)과 적응형 공분산 분류기(Adaptive Covariance Classifier, ACC)를 결합한 소수샷 지속 학습(Few-Shot Continual Learning, FSCL) 방법을 제안합니다. 기본 단계에서 SCL을 기반으로 한 특징 증강 기법을 적용하고, 적응형 공분산 분류기를 사용하여 새로운 클래스의 분포를 보다 정확히 설명합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 ARIC 데이터셋의 기존 다른 방법들과 비교하여 우수한 성능을 보여줍니다. 이는 교실 내 비지도 활동을 잘 학습하면서도 일반적인 수업 활동을 잊지 않고 인식할 수 있는 모델 개발에 기여합니다.



### Enhancing User-Centric Privacy Protection: An Interactive Framework through Diffusion Models and Machine Unlearning (https://arxiv.org/abs/2409.03326)
- **What's New**: 이 논문은 이미지 데이터 공유 및 모델 배포 시 개인정보 보호를 동시에 관리하는 포괄적인 프레임워크를 제안합니다. 또한, 사용자가 생성된 이미지에 대한 피드백에 따라 개인정보 보호 강도를 조정할 수 있는 대화형(interactive) 시스템을 포함하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 두 개의 모듈로 구성되어 있습니다. 첫 번째는 Differential Privacy Diffusion 모델(Diffusion-DP)로, 이미지의 속성 정보를 보호합니다. 두 번째는 고급 기계 학습 미처리(AMU, Advanced Machine Unlearning) 알고리즘으로, 모델의 훈련 데이터셋이 수정될 때 효율적으로 업데이트를 수행합니다. 이를 통해, 개인정보가 포함된 부분을 신속히 제거할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존의 방법들에 비해 특정 속성 분류의 얼굴 데이터셋에서 우수한 개인정보 보호 성능을 보여주었습니다. 데이터 유틸리티와 모델 성능을 유지하면서도 높은 수준의 개인정보 보호를 이룬 것으로 확인되었습니다.



### YOLO-PPA based Efficient Traffic Sign Detection for Cruise Control in Autonomous Driving (https://arxiv.org/abs/2409.03320)
- **What's New**: 이 논문에서는 자율 주행 시스템에서 교통 신호를 효율적이고 정확하게 탐지하기 위한 새로운 YOLO PPA 기반의 교통 신호 탐지 알고리즘이 제안되었습니다.

- **Technical Details**: 기존의 object detection 알고리즘이 작은 크기의 교통 신호를 탐지하기 어려운 문제를 해결하기 위해, YOLO PPA를 기반으로 한 방법을 개발했습니다. 실험은 GTSDB 데이터셋을 사용하여 수행되었으며, 이 결과 제안된 방법이 원래 YOLO보다 추론 효율성을 11.2% 향상시켰음을 보여주고 있습니다.

- **Performance Highlights**: mAP 50도 93.2% 향상되었으며, 이는 제안된 YOLO PPA의 유효성을 입증합니다.



### OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving (https://arxiv.org/abs/2409.03272)
- **What's New**: 이 논문에서는 Occupancy-Language-Action Generative World Model인 OccLLaMA를 제안합니다. 이 모델은 3D 내부 시각적 표현을 활용하여 세계 모델을 구축하고, 비전, 언어 및 행동의 여러 작업을 통합하여 자율 주행(autonomous driving)에서의 다양한 과제를 수행할 수 있습니다.

- **Technical Details**: OccLLaMA는 semantic occupancy를 일반적인 시각적 표현으로 사용하며, autoregressive 모델을 통해 vision-language-action(VLA) 모드를 통합합니다. 특히, sparse 및 class imbalance를 고려한 새로운 VQVAE 유사의 장면 토크나이저(scene tokenizer)를 도입하여 semantic occupancy 장면을 효율적으로 이산화(discretize)하고 재구성합니다. 이 모델은 LLaMA를 기반으로 하여 여러 과제를 수행하기 위한 통합 멀티모달(vocabulary) 구조를 가지고 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 OccLLaMA는 4D occupancy forecasting, motion planning, visual question answering 등 여러 작업에서 경쟁력 있는 성능을 보여주었습니다. 이러한 결과는 자율 주행의 기초 모델로서의 잠재력을 입증합니다.



### SVP: Style-Enhanced Vivid Portrait Talking Head Diffusion Mod (https://arxiv.org/abs/2409.03270)
- **What's New**: 이 연구에서는 오디오 기반의 새로운 Talking Head Generation (THG) 프레임워크인 Style-Enhanced Vivid Portrait (SVP)를 제안합니다. 기존의 방법들이 간과했던 내재적 스타일(intrinsic style)을 효율적으로 캡처하고 조절하여 더욱 다양하고 생동감 넘치는 비디오 생성이 가능합니다.

- **Technical Details**: SVP는 스타일 관련 정보를 완전히 활용하여 내재적 스타일을 모델링합니다. 이를 위해 Probabilistic Style Prior Learning 기법을 도입하여 facial expressions와 audio embedding을 사용하여 Gaussian 분포로 내재적 스타일을 모델링합니다. 이 후, 사전 학습된 Stable Diffusion 모델을 미세 조정하여 크로스 어텐션(cross attention)을 통해 학습된 스타일을 주입합니다.

- **Performance Highlights**: SVP는 MEAD와 HDTF 데이터셋에서 기존의 최첨단 방법보다 뛰어난 성능을 보였습니다. 다양한 감정 표현이 가능하며, 사용자 필요에 따라 동일한 비디오 내에서 여러 감정을 표현할 수 있습니다. 정량적 평가에서는 FVD, FID, PSNR, SSIM 지표에서 우수성을 입증했습니다.



### Bones Can't Be Triangles: Accurate and Efficient Vertebrae Keypoint Estimation through Collaborative Error Revision (https://arxiv.org/abs/2409.03261)
Comments:
          33 pages, ECCV 2024, Project Page: this https URL

- **What's New**: 이번 연구에서는 사용자의 개입을 최소화하면서도 높은 정확도를 달성하는 KeyBot이라는 새로운 방법을 소개합니다. 이 시스템은 기존 모델에서 발생하는 대표적인 오류들을 식별하고 수정하는 기능을 내장하고 있습니다.

- **Technical Details**: KeyBot은 vertebrae keypoint estimation에 특화된 자동 오류 수정 시스템으로, 사용자 입력 없이도 주요 오류를 사전 식별하여 수정합니다. 이 과정은 두 가지 구성 요소, 즉 detector와 corrector로 이루어져 있으며, 합성 데이터(synthetic data)를 사용하여 훈련됩니다.

- **Performance Highlights**: KeyBot은 AASCE 데이터셋에서 평균 방사 오류(MRE)를 19% 줄이고, 목표 성능 달성을 위한 사용자 클릭 수(NoC)를 17% 감소시키며, 기존 방법 대비 월등한 성능을 보였습니다. 이는 vertebrae keypoint estimation의 정확성을 크게 향상시키는 데 기여합니다.



### Granular-ball Representation Learning for Deep CNN on Learning with Label Nois (https://arxiv.org/abs/2409.03254)
- **What's New**: 이번 연구에서는 CNN 모델에 통합할 수 있는 일반적인 granular-ball computing (GBC) 모듈을 제안합니다. 이 모듈은 각 개별 샘플 대신 granular-ball 샘플의 레이블을 예측합니다.

- **Technical Details**: GBC 모듈은 특징 수준에서 입력 샘플을 분리하여 각각의 granular-ball 샘플을 생성합니다. 이 모듈은 전방 전달 과정 및 역전파 과정에서 독특한 경량화 및 안정적인 교육 프로세스를 구현합니다.

- **Performance Highlights**: 제안된 GBC 방법은 CNN 모델의 강인성을 향상시키며, 추가 데이터나 최적화 없이도 이미지 분류 작업에서 효과적인 성능을 보여줍니다.



### Gr-IoU: Ground-Intersection over Union for Robust Multi-Object Tracking with 3D Geometric Constraints (https://arxiv.org/abs/2409.03252)
Comments:
          Accepted for the ECCV 2024 Workshop on Affective Behavior Analysis in-the-wild(ABAW)

- **What's New**: 본 논문에서는 다중 객체 추적(Multi-Object Tracking, MOT)에서 데이터 연관(data association) 문제를 해결하기 위한 Ground IoU (Gr-IoU)라는 새로운 방법을 제안합니다. 이 방법은 특히 서로 가까이 있거나 겹치는 객체들에 대해 연속적인 프레임에서 동일한 객체에 서로 다른 ID가 할당되는 문제를 개선합니다.

- **Technical Details**: Gr-IoU는 장면의 3D 구조를 고려하여 기존의 바운딩 박스(bounding box)를 이미지 공간에서 지면 평면(ground plane)으로 변환합니다. 이 방법은 사라지는 점 기하학(vanishing point geometry)을 활용하여 변환된 바운딩 박스를 사용하여 IoU(Intersection over Union)를 계산합니다. IoU는 객체 간의 전후 관계에 더욱 민감하여 데이터 연관 정확도를 향상시키고 ID 전환을 줄입니다.

- **Performance Highlights**: MOT17 및 MOT20 데이터 세트에서 Gr-IoU 방법을 평가한 결과, 전통적인 비주얼 특성을 활용하지 않는 실시간 방법들보다 우수한 성능을 보였습니다. 특히 다양한 시나리오에서 ID 전환을 줄이고 추적 정확도를 증가시키는 데 있어 중요한 개선을 나타냈습니다.



### Multiple weather images restoration using the task transformer and adaptive mixup strategy (https://arxiv.org/abs/2409.03249)
Comments:
          10 pages, 5 figures and 2 table

- **What's New**: 본 논문에서는 자율 주행 시나리오에서 다양한 악천후 조건을 효율적으로 처리할 수 있는 새로운 다중 작업 악천후 제거 모델을 제안합니다. 기존의 단일 작업 중심 접근 방식에 비해, 새로운 모델은 복합적인 악천후 조건에 적응할 수 있는 기능을 가지고 있습니다.

- **Technical Details**: 제안된 모델은 Weather Task Sequence Generator를 포함하여, self-attention 메커니즘이 서로 다른 날씨 타입에 특정한 특징에 선택적으로 집중할 수 있도록 합니다. 또한, 대규모 악화된 영역을 복구하기 위해 Fast Fourier Convolution (FFC)을 도입하여 receptive field를 확장하고, 적응형 upsampling 기술을 활용하여 날씨 작업 정보와 기저 이미지 특징을 효과적으로 처리합니다.

- **Performance Highlights**: 제안된 모델은 공개 데이터셋에서 최신 기술 상태(state-of-the-art)의 성능을 달성하였습니다. 특히, 다중 스케일의 악화 세부 사항을 활용하여 생성된 작업 특성 시퀀스가 다양한 악화 유형 간의 복잡한 관계를 효과적으로 캡처하고, 이는 이미지 복원 성능을 향상시키는 데 기여합니다.



### UAV (Unmanned Aerial Vehicles): Diverse Applications of UAV Datasets in Segmentation, Classification, Detection, and Tracking (https://arxiv.org/abs/2409.03245)
- **What's New**: 본 논문은 UAV(무인 항공기) 데이터셋의 광범위한 응용 프로그램과 발전을 강조하며, 이 데이터셋이 재난 피해 평가, 공중 감시, 객체 인식 및 추적과 같은 분야에서 중요한 역할을 하고 있음을 보여줍니다.

- **Technical Details**: UAV 데이터셋은 위성 이미지, 드론이 촬영한 사진 및 비디오 등 다양한 데이터 유형을 포함합니다. 이 데이터셋은 단일 데이터 유형에 집중하는 unimodal 또는 여러 데이터 유형을 통합하는 multimodal로 분류될 수 있습니다. 복잡한 환경에서의 시맨틱 세그멘테이션, 포즈 추정, 차량 재인식 및 제스쳐 인식과 같은 작업을 위한 고급 모델 개발을 촉진합니다.

- **Performance Highlights**: 본 논문은 UAV 데이터셋의 광범위한 응용 사례와 연구 분야에서의 활용을 강조하며, 특히 재난 관리에서의 효과성을 증가시키고, 인간 행동 인식 및 저조도 환경에서의 객체 추적의 성능을 향상시키는 데 중점을 두고 있습니다.



### Unveiling Context-Related Anomalies: Knowledge Graph Empowered Decoupling of Scene and Action for Human-Related Video Anomaly Detection (https://arxiv.org/abs/2409.03236)
Comments:
          13pages, 9 figures

- **What's New**: 본 논문에서는 인간 관련 비디오 이상 탐지(anomaly detection)에서의 기존 방법의 한계를 극복하기 위한 새로운 기법인 DecoAD를 제안하고 있습니다. DecoAD는 시각적(feature) 및 행위(action) 특성을 효과적으로 통합하여 복잡한 행동과 장면을 보다 직관적이고 정확하게 이해할 수 있도록 돕습니다.

- **Technical Details**: DecoAD는 'Scene-Action Interweaving' 개념을 도입하여 비디오 클립 내에서 장면(scene)과 인간 행동(action)을 분리(decouple)하고, 이를 다른 클립의 요소들과 엮어(interweave) 복잡한 관계를 탐구하는 방식으로 작동합니다. 이 과정은 크게 'Relation Interweaving'과 'Feature Interweaving' 두 가지 주요 부분으로 나뉘며, 이는 장면과 행동 간의 깊고 복잡한 관계 패턴을 학습하고, 문맥적이고 상호 연관된 패턴을 포괄적으로 이해하는 데 기여합니다.

- **Performance Highlights**: DecoAD는 NWPU Campus, UBnormal, HR-ShanghaiTech와 같은 세 가지 널리 사용되는 벤치마크 데이터셋에서 기존의 인간 관련 비디오 이상 탐지 방법보다 우수한 성능을 보이며, 더 정밀한 이상 탐지를 가능하게 합니다.



### Labeled-to-Unlabeled Distribution Alignment for Partially-Supervised Multi-Organ Medical Image Segmentation (https://arxiv.org/abs/2409.03228)
Comments:
          Accepted by Medical Image Analysis

- **What's New**: 본 논문에서는 Labelled-to-Unlabelled Distribution Alignment (LTUDA) 프레임워크를 소개하여 부분 감독 의료 이미지 분할에서의 분포 불일치를 해결하고, 데이터 증가(Data Augmentation) 기법과 프로토타입 기반 분포 정렬 방법을 도입하여 더 나은 성능을 달성하는 방안을 제안합니다.

- **Technical Details**: LTUDA 프레임워크는 라벨이 붙은 픽셀과 라벨이 없는 픽셀 간의 분포를 정렬하여 비편향화된 pseudo-labels을 생성하는 데 중점을 둡니다. 특히, 크로스 세트 데이터 혼합(cross-set mixing) 데이터 증강 기술과 프로토타입 기반 분포 정렬(prototype-based distribution alignment) 모듈을 도입하여 라벨된 및 라벨이 없는 픽셀 간의 세분화(models) 일관성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 AbdomenCT-1K 및 네 개의 벤치마크 데이터셋에서 기존 부분 감독 방법들보다 상당히 뛰어난 성능을 보였으며, 완전 감독 방법을 초과하는 결과를 보여줍니다.



### Why mamba is effective? Exploit Linear Transformer-Mamba Network for Multi-Modality Image Fusion (https://arxiv.org/abs/2409.03223)
- **What's New**: 이 논문에서는 Tmamba 라는 새로운 이미지를 융합하는 네트워크를 제안합니다. Tmamba는 linear Transformer와 Mamba로 구성되어 있으며, 이미지 융합의 복잡성을 줄이면서도 고급스러운 글로벌 모델링을 제공합니다.

- **Technical Details**: Tmamba는 두 개의 분기(branch)로 이루어진 네트워크로, 각각은 채널 및 위치 정보를 추출하는 데 특화되어 있습니다. T-M 상호작용 구조는 폴라리제이션(Polarization) 또는 convolutional layer를 통해 두 분기 간의 정보를 전달합니다. 또한, attention 레벨에서 cross-modal interaction을 통해 서로 다른 모달리티의 정보를 융합합니다.

- **Performance Highlights**: Tmamba는 여러 이미지 융합 작업, 특히 적외선-가시광 이미지 융합 및 의료 이미지 융합에서 우수한 성능을 보였습니다. 실험 결과, 다양한 데이터셋에서 최첨단 결과를 달성했으며, 효율적인 정보 추출 및 입력 이미지의 특징을 잘 보존합니다.



### Optimizing 3D Gaussian Splatting for Sparse Viewpoint Scene Reconstruction (https://arxiv.org/abs/2409.03213)
- **What's New**: 새롭게 제안된 SVS-GS 프레임워크는 Sparse Viewpoint Scene 복원에서 3D Gaussian smoothing filter를 통합하여 고주파 아티팩트(artifacts)를 억제하는 방법을 제시합니다.

- **Technical Details**: SVS-GS는 Depth Gradient Profile Prior (DGPP) 손실을 동적 깊이 마스크와 통합하여 엣지를 선명하게 하고, 2D diffusion 및 Score Distillation Sampling (SDS) 손실을 통해 기하학적 일관성을 향상시킵니다. 이 프레임워크는 initial 3D Gaussian primitive에 대한 지역 적응 밀도 스케일링 모듈을 적용하여 희소(point cloud)를 통해 생성된 데이터의 밀도를 높입니다.

- **Performance Highlights**: MipNeRF-360 및 SeaThru-NeRF 데이터셋에서의 실험적 평가 결과, SVS-GS는 Sparse viewpoint로부터 3D 복원 성능을 현저히 개선하여, 로봇 공학 및 컴퓨터 비전 응용 프로그램에서 강력하고 효율적인 장면 이해 솔루션을 제공합니다.



### Bi-capacity Choquet Integral for Sensor Fusion with Label Uncertainty (https://arxiv.org/abs/2409.03212)
Comments:
          10 pages, 7 figures, 7 tables; Accepted to 2024 FUZZ-IEEE and presented at 2024 IEEE WCCI; Code available at this https URL

- **What's New**: 본 연구는 기존의 Choquet Integral (ChI) 기반 센서 융합 방법의 한계점을 극복하기 위해, 정밀한 학습 레이블이 필요하지 않은 새로운 Bi-MIChI (bi-capacity Choquet Integral) 프레임워크를 제안합니다. 이 방법은 이진 자율성을 활용하여 센서 데이터 소스 간의 상호작용을 더 효과적으로 모델링할 수 있습니다.

- **Technical Details**: Bi-MIChI는 두 가지 주요 기술적 혁신을 포함하는데, 첫째, bipolar scale ([-1, 1])에서의 bi-capacities를 사용하여 센서 소스 간의 비선형 상호작용을 확장합니다. 둘째, Multiple Instance Learning (MIL) 접근 방식을 통해 불확실한 레이블을 해결하며, 각 데이터 포인트가 아닌 '가방' (bags) 단위로 레이블을 적용합니다.

- **Performance Highlights**: Bi-MIChI 프레임워크는 합성 및 실세계 실험에서 센서 융합과 레이블 불확실성 처리에 효과적인 분류 및 탐지 성능을 보였습니다. 제안된 방법은 기존 Choquet Integral 기반 방법보다 더 나은 성능 향상을 보여주면서 센서 데이터 소스 간의 복잡한 관계를 모델링하는 데 성공하였습니다.



### iSeg: An Iterative Refinement-based Framework for Training-free Segmentation (https://arxiv.org/abs/2409.03209)
- **What's New**: 이 연구는 훈련이 필요 없는 세분화(semanttic segmentation)를 위해 새로운 반복 개선 프레임워크인 iSeg를 제안합니다. 이 프레임워크는 엔트로피 감소(self-attention map) 모듈을 이용하여 전역 정보와 관련 없는 약한 반응을 억제합니다.

- **Technical Details**: iSeg는 엔트로피를 줄이는 자동 주의(self-attention) 모듈과 카테고리 강화된 교차 주의(cross-attention) 모듈을 활용하여 반복 개선을 통해 세분화 마스크를 생성합니다. 이 방법은 기존의 자가 주의와 교차 주의 맵을 이용한 방식의 한계를 해결하고 있습니다.

- **Performance Highlights**: Cityscapes에서 비지도 세분화(unsupervised semantic segmentation)를 수행한 결과, iSeg는 기존의 훈련없는 방법에 비해 3.8% 향상된 mIoU를 기록하였습니다. 다양한 데이터셋과 세분화 작업에서도 성능을 입증했습니다.



### TC-LLaVA: Rethinking the Transfer from Image to Video Understanding with Temporal Considerations (https://arxiv.org/abs/2409.03206)
- **What's New**: 이번 연구에서는 비디오 이해(Task) 과제를 위한 Temporal-Considered LLaVA (TC-LLaVA)라는 새로운 비디오-언어 프레임워크를 제안합니다. 이 모델은 MLLMs(다중모달 대형 언어 모델)의 시간적 인식 능력을 강화하고 텍스트와 비디오 모달리티 간의 주의 상호작용을 차별화하는 두 가지 주요 전략에 초점을 맞추고 있습니다.

- **Technical Details**: 첫 번째 접근법인 Temporal-Aware Dual RoPE는 각 토큰에 고유한 위치 ID를 부여하여 원래 RoPE를 보존하는 방법으로, 시각적 및 텍스트 토큰의 상대적 위치 관계를 유지하며 시간 인식 RoPE를 포함시킵니다. 두 번째 접근법은 Frame-wise Block Causal Attention Mask를 사용하는 것으로, 이는 인과 추론 메커니즘을 유지하면서 비디오 프레임 내외부에서 시각적 토큰 간의 상호작용을 확대합니다.

- **Performance Highlights**: TC-LLaVA는 여러 비디오 이해 벤치마크에서 새로운 최첨단 성능을 달성하였으며, 단순히 비디오 관련 데이터셋에 대한 supervision fine-tuning(SFT)을 통해 이루어졌습니다. 이 모델은 비디오의 동적 사건을 효과적으로 요약하고 복잡한 움직임의 변화를 정확하게 캡처하여 모델의 정확도를 높였습니다.



### Active Fake: DeepFake Camouflag (https://arxiv.org/abs/2409.03200)
- **What's New**: DeepFake 기술이 사회적 우려를 초래하자, 연구자들은 Active Fake라는 새로운 보안 문제를 제기하며 DeepFake Camouflage를 통해 비정상적 비디오를 발표하며 책임을 회피하는 방법을 설명한다.

- **Technical Details**: DeepFake Camouflage를 위해 제안된 Camouflage GAN (CamGAN) 프레임워크는 실제 얼굴에 비작용 블렌딩 불일치를 생성하면서, 인간에게 인식되지 않으며, DeepFake 탐지기를 속이기 위한 적대적 학습(adversarial learning) 전략을 사용하여 설계되었다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 다양한 적대적 공격에 비해 효과적이고 견고함을 입증하였다. 이 방법은 Active Fake 탐지 연구에 대한 추가적인 필요성을 강조하였다.



### RoomDiffusion: A Specialized Diffusion Model in the Interior Design Industry (https://arxiv.org/abs/2409.03198)
- **What's New**: 최근 텍스트-이미지 확산 모델의 발전으로 인해 시각 콘텐츠 생성 분야가 크게 변화하고 있지만, 인테리어 디자인과 같은 특정 분야에서의 응용은 아직 충분히 탐구되지 않았습니다. 본 논문에서는 인테리어 디자인 산업을 위해 특별히 맞춤화된 혁신적인 확산 모델인 RoomDiffusion을 소개합니다.

- **Technical Details**: RoomDiffusion은 대규모 내부 장면 이미지 데이터셋을 기반으로 구축된 데이터 파이프라인을 만들고, 다양한 기법(예: multi-aspect training, multi-stage fine-tune, model fusion)을 적용하여 생성 결과의 시각적 매력과 정확성을 향상시키는 모델입니다. 또한 latent consistency Distillation(LCD) 방법을 활용하여 모델의 효율성도 최적화합니다.

- **Performance Highlights**: RoomDiffusion은 20명 이상의 전문 평가자와의 종합적인 인적 평가를 통해 미적 요소, 정확성 및 효율성 측면에서 업계 최고의 성능을 입증했습니다. 이는 stable diffusion이나 SDXL과 같은 기존 오픈 소스 모델을 능가하여 여러 차원에서 70%의 승률을 보였습니다.



### PEPL: Precision-Enhanced Pseudo-Labeling for Fine-Grained Image Classification in Semi-Supervised Learning (https://arxiv.org/abs/2409.03192)
Comments:
          Under review

- **What's New**: 이 논문에서는 세미-슈퍼바이즈드 러닝(Semi-supervised learning) 프레임워크 내에서 미세한 이미지 분류를 위해 Precision-Enhanced Pseudo-Labeling(PEPL) 방법을 소개합니다. PEPL은 비슷한 클래스 간의 미세한 구분을 개선하기 위해 고품질의 의사 라벨(pseudo-labels)을 생성하여 점진적으로 정제합니다.

- **Technical Details**: 이 방법은 두 가지 주요 단계로 구성됩니다: 초기 의사 라벨 생성 및 의미 혼합 의사 라벨 생성입니다. Class Activation Maps(CAMs)를 활용하여 의미 콘텐츠를 정확히 추정하고, 미세한 분류에 필요한 중요한 세부 사항을 포함하는 정제된 라벨을 생성합니다. 또한, 모델의 예측 성능에 따라 신뢰 임계값(confidence thresholds)을 동적으로 조정하는 접근 방식을 사용합니다.

- **Performance Highlights**: CUB_200_2011 데이터셋에서 20%의 라벨링 데이터 사용 시 완전 감독 모델에 비해 13% 향상된 정확도를 기록하며, 30%의 라벨링 데이터만 사용 시에도 감독 학습과 유사한 결과를 보여주었습니다. 벤치마크 데이터셋에서 기존 세미-슈퍼바이즈드 방법들을 크게 초월하는 성능 향상을 입증하였습니다.



### Mastoidectomy Multi-View Synthesis from a Single Microscopy Imag (https://arxiv.org/abs/2409.03190)
Comments:
          Submitted to Medical Imaging 2025: Image-Guided Procedures, Robotic Interventions, and Modeling

- **What's New**: 이번 논문에서는 단일 CI 미세현미경 이미지로부터 합성된 다중 뷰 비디오를 생성할 수 있는 새로운 파이프라인을 소개합니다. 이 방법은 환자의 수술 전 CT 스캔을 활용하여 수술 후 표면을 예측하는 방식으로, Augmented Reality (AR) 수술에서의 적용 가능성을 다룹니다.

- **Technical Details**: 이 연구는 수술 후 표면 메쉬를 예측하기 위해 기존의 CT 스캔을 처리하고, 해당 메쉬와 선택된 이미지 프레임을 정렬하여 CT 메쉬의 초기 포즈를 정확하게 설정합니다. UV 프로젝션을 통해 이미지를 메쉬에 색칠하는 과정이 포함되어 있습니다.

- **Performance Highlights**: Pytorch3D 및 PyVista를 사용하여 합성된 새 뷰의 품질을 평가한 결과, 두 렌더링 엔진 모두 ground truth와 비교 시 약 0.86의 구조적 유사성 지수를 기록하며 높은 품질을 유지하였습니다. 이러한 대규모 합성 뷰 데이터셋은 AR 수술의 2D 대 3D 등록을 자동으로 추정하는 모델의 지속적인 훈련에 필수적입니다.



### FIDAVL: Fake Image Detection and Attribution using Vision-Language Mod (https://arxiv.org/abs/2409.03109)
- **What's New**: FIDAVL(Fake Image Detection and Attribution using a Vision-Language Model)를 소개하며, 이는 비전(Vision)과 언어(Language) 처리 간의 시너지를 기반으로 한 다중 작업 접근 방식입니다.

- **Technical Details**: FIDAVL은 제로샷 학습(zero-shot learning)의 장점을 활용하여, 비전 및 언어 간의 상호 보완성을 활용하고 부드러운 프롬프트 조정(soft prompt-tuning) 전략을 통해 가짜 이미지를 감지하고 이를 생성 모델에 정확히 귀속시킵니다.

- **Performance Highlights**: FIDAVL은 95.42%의 평균 감지 정확도와 95.47%의 F1 점수를 달성하며, 또한 합성 이미지의 출처 모델에 대한 속성에 있어 평균 F1 점수 92.64% 및 ROUGE-L 점수 96.50%를 기록하였습니다.



### Spatial Diffusion for Cell Layout Generation (https://arxiv.org/abs/2409.03106)
Comments:
          12 pages, 4 figures, accepted by MICCAI 2024

- **What's New**: 이 논문은 세포 탐지를 위한 새로운 generative 모델을 제안합니다. 기존의 generative 모델들이 세포의 공간적 패턴(spatial patterns)을 무시했지만, 이번 연구에서는 공간적 특성을 활용한 diffusion 모델을 적용하여 세포 레이아웃(cell layout)을 생성하는 방법을 제안합니다.

- **Technical Details**: 제안된 모델은 두 가지 주요 아이디어를 적용합니다. 첫째, 희소/밀집 레이아웃의 변이를 다루기 위해 세포의 수에 따라 diffusion 모델을 조건화합니다. 둘째, 공간적 밀도(distribution)를 모델에 포함시켜 레이아웃 맵(layout map)과 공간 밀도 맵(spatial density map)을 동시에 생성하도록 설계합니다. 실험에서는 Kernel density estimation (KDE), Gaussian mixture model (GMM), Gaussian Mixture Copula Model (GMCM) 세 가지 밀도 모델을 비교합니다.

- **Performance Highlights**: 실험 결과, 공간적 패턴을 기반으로 생성한 세포 레이아웃이 고품질 병리 이미지를 생성하는 데 기여하며, 이러한 합성 이미지를 통해 감독(cell detection) 방법의 성능이 크게 향상된다는 것을 보여주었습니다.



### MobileUNETR: A Lightweight End-To-End Hybrid Vision Transformer For Efficient Medical Image Segmentation (https://arxiv.org/abs/2409.03062)
Comments:
          Accepted at ECCV 2024 - BioImage Computing Workshop (Oral)

- **What's New**: 이번 논문에서는 피부 병변(segmentation) 세분화를 위해 고안된 새로운 경량 경량화된 하이브리드 딥러닝 모델인 MobileUNETR을 소개합니다. 이 모델은 CNN과 Transformer의 장점을 결합하여, 블록체인 크기와 계산 복잡성을 최소화하면서도 뛰어난 성능을 발휘합니다.

- **Technical Details**: MobileUNETR는 세 가지 주요 특징을 가지고 있습니다. 1) 경량 하이브리드 CNN-Transformer 인코더로 로컬(local) 및 글로벌(global) 컨텍스트(feature) 추출의 균형을돕습니다. 2) 저해상도와 고해상도에서의 저수준 및 글로벌 특징을 동시에 활용하는 하이브리드 디코더를 도입하여 정확한 마스크 생성을 지원합니다. 3) 300만 개의 파라미터와 1.3 GFLOP의 계산 복잡성으로, 큰 아키텍처와 복잡한 모델을 능가합니다.

- **Performance Highlights**: MobileUNETR는 ISIC 2016, ISIC 2017, ISIC 2018 및 PH2와 같은 네 가지 공개 피부 병변 세분화 데이터셋에서 실험을 통해 그 효과성을 입증했습니다. 모델 크기와 복잡성을 각각 10배 및 23배 감소시키면서 모든 데이터셋에서 성능이 크게 향상되었음을 보여줍니다.



### Incorporating dense metric depth into neural 3D representations for view synthesis and relighting (https://arxiv.org/abs/2409.03061)
Comments:
          Project webpage: this https URL

- **What's New**: 본 연구는 로봇 비전에서의 복잡한 장면을 위한 새로운 신경망 기반 접근 방식을 제안하며, 밀도 높은 메트릭 깊이(dense metric depth)를 신경 3D 나타내기(neural 3D representations)의 훈련에 통합합니다. 이를 통해 형태와 외관을 동시에 정제하는 과정에서 발생하는 아티팩트를 해결합니다.

- **Technical Details**: 해당 연구는 두 개의 신경망 네트워크(𝒩(θ)와 𝒜(ϕ))를 사용하여 장면의 형태와 외관을 캡쳐합니다. 𝒩(θ) 네트워크는 Scene geometry를 위한 다층 퍼셉트론(MLP)이며, 𝒜(ϕ) 네트워크는 주어진 시점에 대한 장면의 방사율(radiance)을 계산합니다. 또한, 이 연구는 깊이 경계(depth edges)를 추가적인 감독 신호로 활용하여 질감과 형상의 차별화를 이루어냅니다.

- **Performance Highlights**: 제안된 방법은 여러 장면 복원, 뷰 인터폴레이션(view interpolation), 기하학적 캡처 및 재조명(relighting)에 대해 소량의 훈련 뷰로도 높은 품질의 결과를 보였으며, 다양한 복잡도의 장면을 효과적으로 처리할 수 있는 가능성을 보여주었습니다.



### Can Your Generative Model Detect Out-of-Distribution Covariate Shift? (https://arxiv.org/abs/2409.03043)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 Out-of-Distribution (OOD) 감지 방법론 중에서 특성 변화(covariate shift) 탐지에 대한 새로운 접근 방식을 제안합니다. 현재까지의 연구는 주로 의미적 변화에 집중되어 왔으며, 저자들은 효과적인 OOD 탐지를 위해 생성 모델(generative models)의 활용과 이를 통해 도메인 특정(covariate)에 대한 변화를 탐지하는 방법을 간단히 설명하고 있습니다.

- **Technical Details**: CovariateFlow라는 새로운 방법론을 제안하여, 조건부 노멀리징 흐름(conditional Normalizing Flows)을 사용하여 고주파(high-frequency) 이미지 요소의 분포를 모델링합니다. 이 방법은 ID(in-distribution) 데이터와 OOD 데이터 간의 차이를 정확하게 감지하는 데 중점을 두고 있으며, Normalized Score Distance(NSD)라는 새로운 메트릭을 통합하여 로그 우도(log-likelihood) 기반 평가를 개선합니다.

- **Performance Highlights**: CIFAR10과 CIFAR10-C, ImageNet200과 ImageNet200-C 간의 실험 결과에서 CovariateFlow가 OOD 감지에서 매우 높은 정확도를 보였습니다. 이 연구는 OOD 탐지의 정확성을 높이는 데 기여할 뿐만 아니라, 다양한 환경에서 머신러닝 모델이 안정적으로 작동할 수 있도록 지원합니다.



### MDNF: Multi-Diffusion-Nets for Neural Fields on Meshes (https://arxiv.org/abs/2409.03034)
- **What's New**: 본 논문에서는 삼각형 메시에 대한 신경 필드를 표현하기 위한 다중 해상도(multi-resolution) 프레임워크를 제안합니다. 이 프레임워크는 공간적 및 주파수 도메인(frequency domain) 모두에서 정의되며, Neural Fourier Filter Bank(NFFB)에서 영감을 받았습니다.

- **Technical Details**: 제안된 아키텍처는 공간 및 주파수 도메인을 분해하며, 각 DiffusionNet 구성요소는 서로 다른 공간 해상도를 표현합니다. 공간 해상도가 높은 수준은 고주파 대역에 연결되고, 저해상도는 저주파로 매핑됩니다.

- **Performance Highlights**: 본 연구의 방법론은 다양한 신경 필드, 즉 합성 RGB 함수(synthetic RGB functions), UV 텍스처 좌표, 정점 노멀(vertex normals)에 적용하여 강력한 성능을 발휘하는 것을 보여주었으며, 뚜렷한 세부사항과 주파수 변화를 정밀하게 캡처할 수 있음을 입증했습니다.



### A General Albedo Recovery Approach for Aerial Photogrammetric Images through Inverse Rendering (https://arxiv.org/abs/2409.03032)
Comments:
          ISPRS Journal of Photogrammetry and Remote Sensing

- **What's New**: 이 논문에서는 자연 조명을 받는 일반적인 항공 사진 측량 이미지로부터 알베도(albedo) 정보를 회복하기 위한 일반 이미지 형성 모델을 제시합니다. 이 모델은 인버스 렌더링(inverse rendering) 내재 이미지 분해(intrinsic image decomposition)를 통해 알베도 정보를 해결합니다.

- **Technical Details**: 항공 사진 측량에서 햇빛 조명(sun illumination)과 장면 기하학(scene geometry)을 추정할 수 있다는 사실을 기반으로 합니다. 이 연구는 드론 기반의 전형적인 사진 측량 수집을 통해 수집된 데이터 외에 추가 입력이 필요없는 물리 기반(physics-based) 접근 방식을 사용합니다.

- **Performance Highlights**: 회복된 알베도 이미지는 특징(feature) 및 밀집 매칭(dense matching), 엣지(edge) 및 선(line) 추출과 같은 전형적인 이미지 처리 작업에서 크게 개선된 성능을 보입니다.



### No Detail Left Behind: Revisiting Self-Retrieval for Fine-Grained Image Captioning (https://arxiv.org/abs/2409.03025)
- **What's New**: 이번 연구에서는 이미지 캡셔닝 시스템이 생성하는 캡션의 정밀도를 높이기 위한 새로운 접근 방식인 Visual Caption Boosting (VCB)와 BagCurri 훈련 커리큘럼을 제안합니다. 이를 통해 기존의 자가 회수(self-retrieval) 보상 방식의 한계를 극복하고, 캡션의 충실성을 유지하면서도 세부적인 요소들을 잘 표현할 수 있도록 하였습니다.

- **Technical Details**: 이 시스템은 MLE (Maximum Likelihood Estimation) 초기화를 개선하고, 자가 회수 과정의 커리큘럼을 설계하여 높은 품질의 캡션을 생성합니다. VCB는 다양한 측면을 포괄적으로 포착하는 밀도가 높은 캡션을 생성하고, BagCurri는 훈련 과정에서 점진적으로 가방 크기를 증가시킴으로써 효과적인 학습을 유도합니다. TrueMatch라는 새로운 벤치마크 또한 도입하여 캡셔닝 시스템의 시각적 세부 구분 능력을 평가합니다.

- **Performance Highlights**: 본 연구의 접근법은 99개의 무작위 방해 요소(random distractors)에 대한 SR에서 기존보다 8.9% 향상되었으며, ImageCoDe에서도 7.6% 개선되었습니다. 본 연구의 SR 접근 방식은 다양한 최신 오픈 소스 MLLMs를 평가했을 때 평균 4.8%에서 7.1%까지의 성능 향상을 보였으며, 파라미터 수는 1-2 배 낮았습니다.



### Boundless: Generating Photorealistic Synthetic Data for Object Detection in Urban Streetscapes (https://arxiv.org/abs/2409.03022)
- **What's New**: 이 논문에서는 Boundless라는 사진 현실적인 합성 데이터 생성 시스템을 소개합니다. 이 시스템은 밀집한 도시 환경에서 정확한 객체 감지를 가능하게 하며, 대량의 실제 데이터 수집 및 수작업으로 진행되는 라벨링 과정 대신 자동화된 구성 가능한 과정을 통해 이 작업을 수행할 수 있습니다.

- **Technical Details**: Boundless는 Unreal Engine 5(UE5) City Sample 프로젝트를 기반으로 하며, 다양한 조명 조건과 장면 변동성 속에서 3D bounding box를 정확하게 수집할 수 있도록 개선되었습니다. 이 시스템에서 생성된 데이터셋으로 훈련된 객체 감지 모델의 성능을 중고도 카메라로 수집한 실제 데이터셋에서 평가하고, Boundless로 훈련된 모델이 CARLA로 훈련된 모델에 비해 7.8 mAP 향상된 성능을 보임을 확인하였습니다.

- **Performance Highlights**: 결과적으로, 합성 데이터 생성이 도시 장면에 적합한 확장 가능한 객체 감지 모델을 훈련 및 파인 튜닝하는 신뢰할 수 있는 방법론이라는 전제를 지지하는 결과를 도출하였습니다. 이 연구는 도시 지역에서의 깊이 있는 딥러닝 응용을 위한 성능 향상에 기여할 것으로 기대됩니다.



### Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors (https://arxiv.org/abs/2409.02979)
- **What's New**: 이 논문은 존재하지 않는 사람의 얼굴 이미지를 합성하여, 얼굴 인식 모델(FR 모델) 훈련에 효과적인 데이터셋을 만드는 방법을 연구합니다. 제안된 모델 Vec2Face는 단일 샘플링된 벡터 입력으로 다수의 고유한 얼굴 이미지를 유연하게 생성하고 조정할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: Vec2Face는 특징 마스크 자동 인코더(feature masked autoencoder)와 디코더로 구성되어 있으며, 얼굴 이미지 재구성을 통해 감독 학습을 진행합니다. 이 모델은 입력 벡터 간의 낮은 유사성을 유지하며, 이를 통해 서로 잘 구별되는 인물들을 생성할 수 있습니다. 주어진 벡터의 작은 변화를 통해 동일한 인물의 다양한 얼굴 속성을 가진 이미지를 생성할 수 있습니다. 또한, 경량화된 선형 탐색(gradient descent) 방법을 통해 특정 속성을 지정하여 이미지를 생성할 수 있습니다.

- **Performance Highlights**: Vec2Face는 300K 인물과 1500만 개의 이미지를 생성하여, 기존 연구에서 생성된 60K 인물 수치를 크게 웃돕니다. HSFace 데이터셋을 이용해 훈련한 FR 모델은 10K에서 300K 인물 기준으로 92%에서 93.52%의 최첨단 정확도를 달성하며, 합성적 훈련 세트로 만들어진 모델이 동등한 규모의 실제 얼굴 이미지로 만들어진 모델보다 높은 정확도를 기록한 최초의 사례입니다.



### Multi-Modal Adapter for Vision-Language Models (https://arxiv.org/abs/2409.02958)
- **What's New**: 이 논문에서는 Multi-Modal Adapter라는 새로운 접근 방식을 제안하며, 이는 CLIP 모델의 비주얼(visual) 및 텍스트(text) 표현 간의 관계를 고려하여 모델의 결과를 개선할 수 있도록 합니다.

- **Technical Details**: Multi-Modal Adapter는 Multi-Head Attention 레이어를 추가하여 텍스트와 이미지의 특성을 통합하여 양쪽 모드의 적응을 수행합니다. 이는 각 작업 별로 효과적으로 비주얼 및 텍스트 정보를 활용하여 과적합(overfitting)을 방지하고 더 나은 일반화 능력을 제공합니다.

- **Performance Highlights**: Multi-Modal Adapter는 n-class-k-shot 설정에서 여러 데이터셋에서 경쟁력 있는 결과를 달성하였으며, 이전의 적응 방법들과 비교하여 보지 못한 클래스를 포함한 성능 향상을 보여줍니다.



### View-Invariant Policy Learning via Zero-Shot Novel View Synthesis (https://arxiv.org/abs/2409.03685)
Comments:
          Accepted to CoRL 2024

- **What's New**: 이번 연구에서는 대규모 비주얼 데이터에서 얻은 지식을 이용하여 다양한 관찰 모드(Observational modality)에 적응할 수 있는 정책 개발을 다룹니다. 특히, 우리는 단일 이미지에서 새로운 시점을 생성하는 (Novel View Synthesis, NVS) 기술을 통해 3D 장면의 사전 지식을 학습합니다.

- **Technical Details**: 본 연구는 View Synthesis Augmentation (VISTA)라 불리는 데이터 증강 기술을 활용하여 단일 관찰 시점에서 훈련된 정책의 강건성을 높이는 방법을 제시합니다. VISTA는 큰 규모의 2D 이미지 데이터 세트를 이용하여 다양한 관찰 시점에서의 카메라 뷰에 적응 가능한 정책을 학습하는 데 중점을 둡니다.

- **Performance Highlights**: VISTA 방법을 적용하여 훈련된 정책은 시뮬레이션 환경과 실제 환경에서의 조작 작업에서 기존 방법론을 초월하는 성능을 보였습니다. 또한, DROID 데이터 세트에서 ZeroNVS 모델을 미세 조정(Finetuning)함으로써 실제 작업에서도 성능 향상을 이루었습니다.



### A practical approach to evaluating the adversarial distance for machine learning classifiers (https://arxiv.org/abs/2409.03598)
Comments:
          Accepted manuscript at International Mechanical Engineering Congress and Exposition IMECE2024

- **What's New**: 이 논문은 ML 분류기(ML classifiers)의 적대적 강건성(adversarial robustness)을 평가하기 위한 새로운 방법론을 제안합니다. 구체적으로, 적대적 거리를(adversarial distance) 반복적인 적대적 공격(iterative adversarial attacks)과 인증(certification) 접근 방식을 통해 추정하는 방식을 강조합니다.

- **Technical Details**: 저자들은 적대적 거리의 상하한을 계산하여 ML 모델의 강건성을 포괄적으로 평가할 수 있는 방법을 제시합니다. 주요 기술적 포인트는 기존의 공격 예산(attack budget)보다도 적대적 거리의 추정값을 더 의미 있게 제시하는 것입니다.

- **Performance Highlights**: 이 연구에서는 두 가지 다른 강건성 모델에 대한 실험을 수행하여 제안된 방법의 효과와 계산 효율성을 평가하였으며, 이 방법이 이전의 구현보다 효과적임을 발견했습니다.



### MaskVal: Simple but Effective Uncertainty Quantification for 6D Pose Estimation (https://arxiv.org/abs/2409.03556)
- **What's New**: 본 논문에서는 로봇 응용 프로그램에서 6D 자세 추정을 수행할 때 신뢰할 수 있는 자세의 중요성을 강조하며, 기존의 6D 자세 추정기가 불확실성 정량화(uncertainty quantification)를 제대로 제공하지 않거나, 제공하더라도 실제 오류와의 상관관계가 낮다는 점을 지적합니다. 이를 해결하기 위해, MaskVal이라는 간단하면서도 효과적인 방법을 제안합니다.

- **Technical Details**: MaskVal은 세그멘테이션을 렌더링하여 자세 추정을 검증하는 방식으로, 추정기 자체의 수정 없이 작동합니다. 이 방법은 기존의 두 단계 자세 추정기에서 이미 제공되는 인스턴스 세그멘테이션을 활용하여 실제로 추가적인 계산 비용이 적습니다.

- **Performance Highlights**: MaskVal은 기존의 최첨단 앙상블 방법보다 더 우수한 성능을 보여주며, 6D 자세 추정기의 작업 성능을 안전하고 신뢰할 수 있는 방향으로 크게 향상시킵니다. 또한, 6D 자세 추정의 불확실성 정량화 시 특정 성능 지표를 제안하여 평가를 보다 세밀하게 할 수 있도록 합니다.



### Unified Framework for Neural Network Compression via Decomposition and Optimal Rank Selection (https://arxiv.org/abs/2409.03555)
- **What's New**: 본 논문은 Optimal Rank Tensor decOmpoSition (ORTOS)라는 통합된 프레임워크를 소개합니다. 이 프레임워크는 텐서 분해와 최적 랭크 선택을 동시에 해결하며, 특정 랭크 제약 내에서 복합 압축 손실을 활용합니다.

- **Technical Details**: ORTOS는 연속 공간에서 자동으로 랭크를 검색하여 데이터를 필요로 하지 않고도 레이어 분해를 위한 최적의 구성을 효율적으로 식별합니다. 이 접근 방식은 계산적으로 효율적이며, 이후의 미세 조정 단계와 결합하여 고도로 압축된 모델의 성능을 원래 모델과 동등하게 유지합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 사용한 실험에서, ResNet-18의 경우 모든 지표에서 향상이 있었고, 다른 실험에서도 경쟁력 있는 결과를 얻었습니다. 본 방법은 관련 작업에 비해 검색 단계를 가속화했습니다.



### Tissue Concepts: supervised foundation models in computational pathology (https://arxiv.org/abs/2409.03519)
Comments:
          22 Pages, 3 Figures, submitted to and under revision at Computers in Biology and Medicine

- **What's New**: 이 논문은 패톨로지스트의 업무 부담을 줄이기 위해 다중 작업 학습(multitask learning)을 기반으로 하는 효율적인 감독 학습(supervised learning) 방법을 제안합니다.

- **Technical Details**: 논문에서 제안한 방법인 Tissue Concepts 인코더는 16개의 서로 다른 분류(classification), 분할(segmentation), 탐지(detection) 작업을 결합하여 총 912,000개의 패치를 사용하여 훈련됩니다. 이 인코더는 샘플의 특성을 포착할 수 있어 강력한 일반화 기능을 제공합니다.

- **Performance Highlights**: Tissue Concepts 모델은 자가 감독(self-supervision)으로 훈련된 모델과 유사한 성능을 발휘하였으며, 훈련 패치의 양은 단 6%에 불과했습니다. 또한, Tissue Concepts 인코더는 ImageNet이라는 사전 훈련된 인코더보다도 우수한 성능을 보였습니다.



### A Key-Driven Framework for Identity-Preserving Face Anonymization (https://arxiv.org/abs/2409.03434)
Comments:
          Accepted by NDSS Symposium 2025. Please cite this paper as "Miaomiao Wang, Guang Hua, Sheng Li, and Guorui Feng. A Key-Driven Framework for Identity-Preserving Face Anonymization. In the 32nd Annual Network and Distributed System Security Symposium (NDSS 2025)."

- **What's New**: 이 연구에서는 가상 얼굴을 생성하는 데 있어 개인 정보 보호와 식별 가능성 간의 갈등을 해결하기 위해 키 기반 얼굴 익명화 및 인증 인식(KFAAR) 프레임워크를 제안합니다. 이 프레임워크는 원래 얼굴의 특징을 보존하는 가상 얼굴 생성 모듈(HPVFG)과 키에 의해 제어되는 가상 얼굴 인증 모듈(KVFA)로 구성됩니다.

- **Technical Details**: HPVFG 모듈은 사용자의 키를 활용하여 원래 얼굴의 잠재 벡터를 가상 공간으로 변환하고, 그에 따라 가상 얼굴을 생성합니다. 또한, 얼굴의 유사한 헤드 포즈와 표정을 보존하기 위한 모듈을 추가하여 가상의 얼굴이 원본과 같은 헤드 포즈와 표정을 갖도록 합니다. KVFA 모듈은 올바른 사용자 키를 사용하여 가상 얼굴을 인식하고, 원래의 얼굴 이미지를 노출하지 않으면서 원본 신원을 복원할 수 있습니다.

- **Performance Highlights**: 여러 실험을 통해 HPVFG와 KVFA 모듈의 장점이 입증되었으며, 이는 얼굴 익명성과 식별 가능성을 효과적으로 달성합니다. 제안된 방법은 원래 얼굴과 동시에 시각적 차이를 유지하면서도 원본 신원을 인증할 수 있는 가상 얼굴을 생성하는 데 탁월한 성능을 보였습니다.



### TBConvL-Net: A Hybrid Deep Learning Architecture for Robust Medical Image Segmentation (https://arxiv.org/abs/2409.03367)
- **What's New**: 새로운 논문에서는 전통적인 CNN(Convolutional Neural Network) 아키텍처와 비전 변환기(Vision Transformer)를 결합한 TBConvL-Net이라는 새로운 딥 러닝 아키텍처를 소개합니다. 이 모델은 다중 스케일 특성을 캡처하고, 정보 상호작용을 개선하며, 의료 이미지 분할(Automated Medical Image Segmentation) 작업에서 일관된 성능 향상을 보여줍니다.

- **Technical Details**: TBConvL-Net은 CNN 인코더-디코더 아키텍처의 지역적 특성과 BConvLSTM(Bidirectional ConvLSTM) 및 비전 변환기를 사용하여 장거리 및 시간적 의존성을 결합한 하이브리드 네트워크입니다. 이 아키텍처는 스킵 연결(Skip Connections)에서 컨텍스트 채널 관계를 캡처하고, 다중 스케일 문맥 정보를 전송하는 기능을 포함합니다. 또한, composite loss function을 도입하여 분할의 견고성과 예측 출력의 경계 일치를 고려합니다.

- **Performance Highlights**: TBConvL-Net은 10개의 공개 데이터셋을 사용하여 7가지 의료영상 분할 작업에서 평가되었으며, 기존 최첨단(State of the Art, SOTA) 방법들에 비해 일관되게 성능이 향상되었습니다. 이 모델은 자원 제약이 있는 환경에서도 우수한 성능을 발휘할 수 있도록 설계되었습니다.



### Eetimating Indoor Scene Depth Maps from Ultrasonic Echoes (https://arxiv.org/abs/2409.03336)
Comments:
          ICIP 2024

- **What's New**: 이 논문은 기존의 가청 사운드 대신 비가청 초음파 에코를 활용한 깊이 추정 방법을 제안합니다. 이는 조용한 환경에서 깊이 센서를 사용할 수 없는 문제를 해결하려고 합니다.

- **Technical Details**: 연구는 고주파 범위로 소리의 주파수를 제한했을 때 깊이 추정 정확도를 평가하였습니다. 초음파 주파수가 사용할 때의 문제점인 노이즈 민감도와 감쇠 문제를 해결하기 위해, 훈련 동안 가청 에코를 보조 데이터로 사용하여 새로운 딥 러닝 방법을 제안합니다. 이 방법은 가청 에코의 스펙트럼 정보와 초음파 에코를 선형 결합하여 합성 에코를 생성합니다.

- **Performance Highlights**: Replica 데이터를 사용한 실험 결과, 제안된 방법이 초음파 에코를 사용한 깊이 추정 정확도를 개선하는 것을 입증했습니다.



### Improving Robustness to Multiple Spurious Correlations by Multi-Objective Optimization (https://arxiv.org/abs/2409.03303)
Comments:
          International Conference on Machine Learning 2024

- **What's New**: 본 논문은 여러 가지 편향을 가진 데이터셋을 기반으로 공정하고 정확한 모델을 훈련하는 문제를 연구합니다. 특히, 모델 훈련 중 발생할 수 있는 여러 편향 간의 갈등을 완화하는 데 중점을 둔 새로운 훈련 방법론을 제안합니다.

- **Technical Details**: 이 논문에서는 MOO (Multi-Objective Optimization) 이론을 기반으로 하는 새로운 훈련 알고리즘을 제안합니다. 훈련 데이터를 여러 그룹으로 나누어 각 그룹이 서로 다른 단축(shortcut)을 유도하게 하며, 그룹 별 손실의 선형 조합을 최적화하고 그 가중치를 동적으로 조정하여 각 그룹 간의 성능 갈등을 완화합니다.

- **Performance Highlights**: 제안된 알고리즘은 MultiCelebA를 포함한 세 가지 다중 편향 벤치마크에서 최고의 성능을 달성했으며, 기존의 단일 편향 데이터셋에서도 우수한 성능을 보여주었습니다.



### ChartMoE: Mixture of Expert Connector for Advanced Chart Understanding (https://arxiv.org/abs/2409.03277)
- **What's New**: 이번 논문에서는 다중 모달 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 차트 이해 능력을 향상시키기 위한 새로운 접근 방식인 ChartMoE를 제안합니다. 본 연구는 전통적인 선형 프로젝터를 Mixture of Experts (MoE) 아키텍처로 대체하여 모달리티 갭을 메우기 위한 다양한 정렬(Task) 작업을 통해 여러 선형 커넥터를 훈련시키는 방법을 소개합니다.

- **Technical Details**: ChartMoE는 차트-테이블-JSON-코드 쌍으로 구성된 90만 개 이상의 쿼드러플을 포함하는 ChartMoE-Align 데이터셋을 활용하여 각각의 협업 방식으로 초기화된 여러 전문가를 생성합니다. 이를 통해 다단계 훈련 구조를 설계하여 차트 이해 성능을 극대화합니다. 또한, 논문에서는 높은 품질의 지식 학습과 최적화 기법을 도입하여 각 전문가들의 성능을 제고합니다.

- **Performance Highlights**: ChartMoE는 ChartQA 벤치마크에서 이전 최고 성능인 80.48%를 84.64%로 향상시키는 등, 여러 벤치마크에서 큰 폭으로 성능을 향상시키는 결과를 보였습니다. 이러한 결과는 기술적 혁신이 차트 해석과 근거 있는 이해를 더욱 향상시킬 수 있음을 보여줍니다.



### Perceptual-Distortion Balanced Image Super-Resolution is a Multi-Objective Optimization Problem (https://arxiv.org/abs/2409.03179)
- **What's New**: 본 연구는 Single-Image Super-Resolution (SISR) 모델의 학습 과정에서 Multi-Objective Optimization (MOO)을 통합하여 인지 품질과 왜곡 사이의 균형을 맞추는 새롭고 혁신적인 방법을 제안합니다.

- **Technical Details**: MOBOSR (Multi-Objective Bayesian Optimization Super-Resolution) 프레임워크를 통해 손실 함수 가중치와 이미지 품질 평가(IQA) 메트릭스 간의 관계를 개념화하고 최적화하는 방식을 채택했습니다. 이 접근 방식은 하이퍼파라미터 튜닝 프로세스를 자동화하고, 전체적인 계산 비용을 줄이며, 다양한 손실 함수의 동시 사용을 가능하게 합니다.

- **Performance Highlights**: MOBOSR은 인지 품질과 왜곡 측면에서 최신 방법들을 초월하며, 인식-왜곡 Pareto 프론티어를 크게 향상시킵니다. 이 연구는 이미지 복원 작업에서 인지 품질과 신뢰성 간의 균형을 맞추기 위한 새로운 방향을 제시합니다.



### Developing, Analyzing, and Evaluating Self-Drive Algorithms Using Drive-by-Wire Electric Vehicles (https://arxiv.org/abs/2409.03114)
Comments:
          Supported by the National Science Foundation under Grants No. 2150292 and 2150096

- **What's New**: 이 논문에서는 안전하고 효과적인 자율 주행을 위해 중요한 차선 추적 알고리즘의 개발 및 평가에 초점을 맞추었습니다. 다양한 알고리즘이 Vehicle-to-Everything (V2X) 프로젝트의 신뢰할 수 있는 알고리즘을 찾기 위해 비교되었습니다.

- **Technical Details**: 제시된 5개의 차선 검출 알고리즘은 1) Largest White Contour, 2) Least Square Regression을 이용한 차선 선형 근사, 3) K-Means를 사용한 선형 차선 검색, 4) DBSCAN을 이용한 차선 선별, 5) DeepLSD 차선 검출로 구성되어 있습니다. 이 알고리즘들은 드라이브-바이-와이어 시스템을 갖춘 실제 차량과 시뮬레이터에서 테스트되었습니다.

- **Performance Highlights**: 테스트 결과, 두 가지 접근 방식이 가장 신뢰할 수 있는 것으로 나타났으며, 이들은 모두 차선의 위치를 정확히 감지하고 비지도 학습을 사용하여 분리하는 능력을 보여주었습니다. 이 접근 방식들은 다양한 주행 시나리오에서 강력함을 입증하였고, V2X 프로젝트에 통합될 가능성이 높습니다.



### MSTT-199: MRI Dataset for Musculoskeletal Soft Tissue Tumor Segmentation (https://arxiv.org/abs/2409.03110)
Comments:
          Dataset will be made publicly available after the acceptance of the paper

- **What's New**: 본 연구는 199명의 환자로부터 수집된 199개의 근골격계 연부조직 종양의 MR 영상 데이터셋을 구축하고 이 데이터셋을 사용하여 자동화된 종양 세분화(segmentation) 모델을 훈련했습니다. 이 모델은 사전 조정 없이 SOTA(상태 최첨단) 다이스 점수 0.79를 기록하였습니다.

- **Technical Details**: 연구팀은 LabelStudio를 사용하여 데이터 주석(annotation) 플랫폼을 구축하였고, 3단계의 주석 과정을 통해 각 MRI의 중심 슬라이스를 주석하였으며, 결국 각 종양의 경계가 정확하게 세분화되었습니다. 각 종양의 조직 유형은 섬유성, 지방, 점액성, 신경 및 혈관으로 구분하였습니다.

- **Performance Highlights**: 모델의 성능은 섬유성 및 혈관 종양에서 다양하고 강렬한 해부학적 위치와 크기, 강도 이질성(intensity heterogeneity) 때문에 감소하였습니다. 향후 모델 개발 및 데이터 수집에 대한 제안도 포함되어 있습니다.



### Coupling AI and Citizen Science in Creation of Enhanced Training Dataset for Medical Image Segmentation (https://arxiv.org/abs/2409.03087)
- **What's New**: 본 연구에서는 AI와 crowdsourcing을 결합하여 다양한 방식의 의료 이미지 데이터셋을 개선하는 강력하고 다목적 작업 프레임워크를 제시합니다. 이 접근법은 사용자 친화적인 온라인 플랫폼을 활용하여 다양한 서포터들이 의료 이미지를 효율적으로 라벨링할 수 있도록 돕습니다.

- **Technical Details**: 이 프레임워크는 MedSAM segmentation AI와 pix2pixGAN을 통합하여, 지역 전문가 수준의 품질을 유지하면서도 어노테이션 속도를 높입니다. 이 과정에서 crowdsourced 라벨과 합성 이미지를 결합하여 데이터셋을 강화합니다. 특히, 의료 이미지에 대한 라벨링 프로세스를 간소화하여 데이터셋의 품질과 양을 동시에 개선할 수 있습니다.

- **Performance Highlights**: 본 프레임워크를 통해 모델의 성능이 유의미하게 향상되었으며, 특히 제한된 훈련 데이터로도 DL 모델의 세분화 정확도가 크게 개선되었습니다. 연구 결과, 고품질 라벨이 직접적으로 AI 모델의 정확성과 신뢰성에 영향을 미침을 보여주었습니다.



### Design and Evaluation of Camera-Centric Mobile Crowdsourcing Applications (https://arxiv.org/abs/2409.03012)
- **What's New**: 이 연구는 카메라 기반의 모바일 크라우드소싱 애플리케이션의 세 가지 버전에서 사용자 기대 수준을 높여 데이터를 수집하는 방법을 분석했습니다. 사용자 제공 정보의 수준과 수집된 레이블 이미지의 품질 간의 트레이드오프를 평가한 결과, 더 높은 레이블 수준을 요구할 때 사용자 참여 및 만족도가 감소하지 않음을 밝혔습니다.

- **Technical Details**: 세 가지 카메라 중심 모바일 애플리케이션 디자인에서 사용자가 사진을 찍고, 사전 정의된 목록에서 객체를 식별하는 단계까지의 세부 사항을 비교했습니다. 이는 비지도 학습(unsupervised learning), 약한 감독 학습(weakly supervised learning), 강한 감독 학습(strongly supervised learning) 형태의 데이터 수집을 다루며, 사용자 조사로 레이블 수준과 참여도, 레이블 이미지의 양과 품질을 평가했습니다.

- **Performance Highlights**: 높은 레이블 요구 수준을 가진 애플리케이션 버전에서 사용자들은 가장 많은 이미지를 수집 및 주석 달기에 참여했으며, 사용자 만족도는 감소하지 않았습니다. 또한, 추가적으로 수집된 레이블 데이터는 이미지 검색 작업에서 성능 향상을 지원하는 데 기여했습니다.



New uploads on arXiv(cs.AI)

### TRACE-cs: Trustworthy Reasoning for Contrastive Explanations in Course Scheduling Problems (https://arxiv.org/abs/2409.03671)
- **What's New**: TRACE-cs는 대규모 언어 모델(LLM)과 상징적 추론(symbolic reasoning)을 결합하여 시간 스케줄링 문제에서의 대비적 질문에 대한 설명을 생성하는 혁신적인 하이브리드 시스템입니다.

- **Technical Details**: TRACE-cs 시스템은 상징적 모듈과 LLM 모듈로 구성됩니다. 상징적 모듈은 스케줄링 제약을 논리 공식으로 인코딩하고 사용자의 질문에 대한 설명을 생성합니다. LLM 모듈은 자연어 문장을 처리하며 사용자 쿼리를 상징적 표현으로 변환하고 생성된 설명을 자연어로 정제합니다.

- **Performance Highlights**: TRACE-cs는 설명 정확도 면에서 100%를 기록하며, LLM 전용 접근 방식에 비해 44% 및 49%를 크게 초과했습니다. 또한 설명 단어 수에서 평균 46개의 단어로 더 간결한 설명을 제공함으로써, 대조적으로 LLM 접근 방식보다 훨씬 효율적인 성과를 보여주었습니다.



### Game On: Towards Language Models as RL Experimenters (https://arxiv.org/abs/2409.03402)
- **What's New**: 본 논문에서는 강화 학습( reinforcement learning) 실험 워크플로우의 일부를 자동화하는 에이전트 아키텍처를 제안합니다. 이 시스템은 VLM(vision language model)을 활용하여 실험의 진행 상황 모니터링, 새로운 작업 제안, 작업 세분화 및 기술 검색을 자동화하며, 이를 통해 에이전트의 제어 영역에서 자동화된 교육 과정을 구축합니다.

- **Technical Details**: 제안된 시스템 아키텍처는 VLM을 사용하여 강화 학습 실험 루프의 대부분을 자동화하며, 인간 실험자가 필요로 하는 여러 기능을 통합합니다. 특정 사용 예로는 Gemini 모델을 사용하여 언어 조건부 Actor-Critic 알고리즘을 통해 작업의 커리큘럼을 제공하고, 로봇의 조작을 위해 특정 작업에 대한 정책을 훈련합니다. 이 시스템은 고수준 VLM의 감독에 따라 수집된 데이터를 사용하여 로봇 제어를 위한 저수준 정책을 개선합니다.

- **Performance Highlights**: 제안된 시스템은 로봇에서 여러 조작 작업을 수행하는 정책을 훈련시키며, VLM이 안내하는 탐색을 통해 데이터 다양성을 높이고 성능을 향상시키는 결과를 보여줍니다. 추가 논의에 따르면, 시스템은 성장하는 기술 라이브러리를 구축하고, 학습 기술의 진행 상황을 판단하는 능력을 갖추고 있으며, 이는 작업과 도메인에 대한 완전 자동화를 위한 잠재적인 가능성을 제시합니다.



### iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models (https://arxiv.org/abs/2409.03284)
Comments:
          Accepted at The International Web Information Systems Engineering conference (the WISE conference) 2024

- **What's New**: 본 논문에서는 iText2KG라는 방법을 제안합니다. 이는 점진적이고 주제 독립적인 Knowledge Graph (KG) 구축을 위한 제로샷(zero-shot) 기법을 적용하여 포괄적인 KG 생성을 가능하게 합니다.

- **Technical Details**: iText2KG는 네 가지 모듈로 구성됩니다: 1) Document Distiller는 원문 문서를 LLM을 활용하여 사전 정의된 의미 블록으로 형식화합니다. 2) iEntities Extractor는 의미 블록 내에서 고유한 개체를 식별하고 모호성을 해결합니다. 3) iRelation Extractor는 확인된 개체와 의미 블록을 바탕으로 고유한 관계를 탐지합니다. 4) 마지막으로 Graph Integrator는 Neo4j를 사용하여 관계와 개체를 시각적으로 표현합니다.

- **Performance Highlights**: 이 방법은 세 가지 시나리오에서 기존의 방법들과 비교하여 우수한 성능을 보여주었습니다: 과학 논문의 그래프 변환, 웹사이트 그래프 변환, 이력서 그래프 변환.



### ChartMoE: Mixture of Expert Connector for Advanced Chart Understanding (https://arxiv.org/abs/2409.03277)
- **What's New**: 이번 논문에서는 다중 모달 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 차트 이해 능력을 향상시키기 위한 새로운 접근 방식인 ChartMoE를 제안합니다. 본 연구는 전통적인 선형 프로젝터를 Mixture of Experts (MoE) 아키텍처로 대체하여 모달리티 갭을 메우기 위한 다양한 정렬(Task) 작업을 통해 여러 선형 커넥터를 훈련시키는 방법을 소개합니다.

- **Technical Details**: ChartMoE는 차트-테이블-JSON-코드 쌍으로 구성된 90만 개 이상의 쿼드러플을 포함하는 ChartMoE-Align 데이터셋을 활용하여 각각의 협업 방식으로 초기화된 여러 전문가를 생성합니다. 이를 통해 다단계 훈련 구조를 설계하여 차트 이해 성능을 극대화합니다. 또한, 논문에서는 높은 품질의 지식 학습과 최적화 기법을 도입하여 각 전문가들의 성능을 제고합니다.

- **Performance Highlights**: ChartMoE는 ChartQA 벤치마크에서 이전 최고 성능인 80.48%를 84.64%로 향상시키는 등, 여러 벤치마크에서 큰 폭으로 성능을 향상시키는 결과를 보였습니다. 이러한 결과는 기술적 혁신이 차트 해석과 근거 있는 이해를 더욱 향상시킬 수 있음을 보여줍니다.



### Strategic Chain-of-Thought: Guiding Accurate Reasoning in LLMs through Strategy Elicitation (https://arxiv.org/abs/2409.03271)
- **What's New**: 이번 논문에서는 Chain-of-Thought (CoT) 방법론의 한계점을 극복하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 Strategic Chain-of-Thought (SCoT)로, LLM의 성능을 개선하기 위해 전략적 지식을 통합하여 생성 과정에서 중간 추론 단계를 정교화합니다.

- **Technical Details**: SCoT는 두 단계의 절차를 포함합니다. 첫 단계에서는 효과적인 문제 해결 전략을 도출하고, 그 다음 해당 전략을 활용하여 높은 품질의 CoT 경로와 최종 답변을 생성합니다. 이러한 과정은 단일 프롬프트로 수행되며, 불필요한 다중 질의 및 추가 지식 통합을 줄입니다.

- **Performance Highlights**: SCoT 방법의 실험 결과는 GSM8K 데이터 세트에서 21.05%의 정확도 증가와 Tracking_Objects 데이터 세트에서 24.13%의 정확도 향상을 보여주었습니다. 이는 SCoT의 효과성을 강조하며, 여러 모델에서 복잡한 추론 작업에 대한 성능 향상을 증명합니다.



### In Search of Trees: Decision-Tree Policy Synthesis for Black-Box Systems via Search (https://arxiv.org/abs/2409.03260)
Comments:
          8 pages main text incl. references, 1 page appendix

- **What's New**: 본 연구에서는 블랙박스 환경에서 최적의 결정 트리 정책을 생성하는 새로운 접근 방식을 제시합니다. 기존 방법들과 달리 필요한 기존 정책이나 환경 모델 없이도 최적성 보장을 제공합니다.

- **Technical Details**: 제안된 알고리즘은 주어진 프레딕트(피redicate) 세트를 사용할 수 있는 모든 가능한 결정 트리를 체계적으로 생성하며, 각 트리에 대해 블랙박스 환경에서 평가된 사양을 최적화하는 트리를 선택합니다. 핵심 요소는 새로운 가지치기(pruning) 메커니즘으로, 이를 통해 검색 공간을 상당히 줄입니다.

- **Performance Highlights**: 실험 결과, 제안된 가지치기 메커니즘을 이용한 경우 작고 최적의 결정 트리를 합리적인 시간 내에 구축할 수 있음을 보여주었습니다. 또한, 프레딕트 수와 트리 크기에 대한 알고리즘의 확장성 분석을 통해 매우 효율적인 성능을 입증하였습니다.



### InfraLib: Enabling Reinforcement Learning and Decision Making for Large Scale Infrastructure Managemen (https://arxiv.org/abs/2409.03167)
- **What's New**: InfraLib라는 혁신적인 시뮬레이션 프레임워크를 소개하여, 인프라 관리 문제를 효과적으로 모델링하고 분석하는 방법을 제시합니다. 이 프레임워크는 복잡한 인프라 시스템의 관리 및 최적화를 위해 계층적이고 확률적 접근 방식을 사용합니다.

- **Technical Details**: InfraLib는 인프라 시스템의 복잡한 상호작용을 포착하기 위해 계층적 모델을 통합하여 현실적인 표현을 제공합니다. 이 프레임워크는 부분 관측성이 높은 확률 과정인 Partially Observable Markov Decision Processes (POMDP)를 기반으로 하며, 이를 통해 불확실성과 자원의 제약을 모델링할 수 있습니다.

- **Performance Highlights**: InfraLib의 성능을 실제 도로 네트워크와 10만 개 구성 요소를 포함하는 합성 벤치마크를 통해 입증하였으며, 다양한 관리 전략의 배치 및 평가를 위한 실제 시나리오 생성을 지원합니다.



### Lexicon3D: Probing Visual Foundation Models for Complex 3D Scene Understanding (https://arxiv.org/abs/2409.03757)
Comments:
          Project page: this https URL , Github: this https URL

- **What's New**: Lexicon3D라는 새로운 프로빙 아키텍처를 통해 시각적 인코더의 성능을 평가하고, 이미지 기반, 비디오 기반, 3D 기반의 다양한 모델의 강점과 한계를 식별합니다.

- **Technical Details**: 본 연구에서는 7개의 시각적 개념 모델을 평가하며, 각 모델의 기능적 강점 및 약점을 정의하기 위해 Vision-Language Scene Reasoning, Visual Grounding, Segmentation 및 Registration의 4가지 작업을 사용했습니다.

- **Performance Highlights**: DINOv2가 우수한 성능을 보이며, 비디오 모델은 객체 수준의 작업에서 뛰어난 결과를 나타냅니다. 또한, 확산 모델은 기하학적 작업에 이점이 있음을 보여주고, 언어로 사전 훈련된 모델은 언어 관련 작업에서 예상치 못한 한계를 드러냅니다.



### WildVis: Open Source Visualizer for Million-Scale Chat Logs in the Wild (https://arxiv.org/abs/2409.03753)
- **What's New**: WildVis는 대규모 대화 데이터셋을 효율적으로 분석할 수 있는 새로운 인터랙티브 툴입니다. 이 도구는 사용자-챗봇 상호작용을 신속하고 다양한 방식으로 분석하는 기능을 제공합니다.

- **Technical Details**: WildVis는 필터 기반 검색 시스템과 임베딩 기반 시각화 모듈로 구성되어 있으며, 수십만 건의 대화를 빠르게 검색하고 시각화할 수 있습니다. 이를 위해 Elasticsearch를 활용하여 검색 인덱스를 구성하고, OpenAI의 text-embedding-3-small 모델을 사용하여 대화의 첫 번째 사용자 발화를 임베딩합니다.

- **Performance Highlights**: WildVis는 대화 데이터의 시각화 및 검색을 지원하며, 사례 연구를 통해 챗봇 남용 연구 촉진, 데이터셋 간 주제 분포 비교, 사용자 특화 대화 패턴을 파악하는 데 기여합니다. 현재 WildVis는 오픈소스로 제공되며, MIT 라이센스 하에 이용 가능합니다.



### LLM-CI: Assessing Contextual Integrity Norms in Language Models (https://arxiv.org/abs/2409.03735)
Comments:
          20 pages, 8 Figures, 4 Tables

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)에서 인코딩된 사회적 규범을 평가하기 위한 최초의 오픈소스 프레임워크인 LLM-CI를 제안합니다. 이는 개인 정보 보호(norms)를 효과적으로 평가하는 메소드 개발을 통해 이루어집니다.

- **Technical Details**: LLM-CI는 Contextual Integrity(맥락적 무결성) 기반의 방법론을 채택하여 다양한 LLM과 환경에서 인코딩된 규범을 평가합니다. 또한, 다중 프롬프트 평가 방법을 도입하여 프롬프트의 민감성 문제를 해결하고 일관된 응답을 생성하는 프롬프트에 따라 규범을 평가합니다.

- **Performance Highlights**: LLM-CI를 이용해 IoT 및 COPPA와 같은 데이터셋을 통해 10개의 최신 LLM을 평가하고, 모델 속성(예: 하이퍼파라미터, 용량) 및 최적화 전략(예: 정렬, 양자화)의 영향을 살펴보았습니다.



### Planning In Natural Language Improves LLM Search For Code Generation (https://arxiv.org/abs/2409.03733)
- **What's New**: PLANSEARCH는 다양하고 효율적인 문제 해결 아이디어 검색을 통해 코드 생성의 성능을 향상시키는 새로운 알고리즘입니다. 이 알고리즘은 단순히 잘못된 유사한 출력을 반복하는 대신, 다양한 후보 플랜을 탐색합니다.

- **Technical Details**: PLANSEARCH는 문제 해결을 위해 생성된 다양한 관찰(observations)을 바탕으로 높은 수준의 계획(plans)으로 구성된 후보 세트를 생성합니다. 이러한 계획은 자연어(natural language)로 표현되며, 이는 다양한 잠재적 솔루션을 탐색하는 데 기여합니다. 결과적으로 PLANSEARCH는 경쟁력 있는 코딩 벤치마크인 LiveCodeBench에서 77.0%라는 최신 최고 성과를 달성합니다.

- **Performance Highlights**: PLANSEARCH는 Claude 3.5 Sonnet 위에 적용했을 때, 기존의 표준 샘플링 방법들보다 더욱 다양하고 효과적인 코드를 생성하여 200회의 통과율(pass@200)에서 77.0%를 기록했습니다. 이는 비검색 방식에서 얻은 최고 점수(41.4%) 보다 크게 향상된 결과로, 검색 알고리즘 덕분에 성능 향상이 다양성에 의해 직결된다는 점을 강조합니다.



### Sample-Efficient Diffusion for Text-To-Speech Synthesis (https://arxiv.org/abs/2409.03717)
Comments:
          Interspeech 2024

- **What's New**: 이번 연구에서는 Sample-Efficient Speech Diffusion (SESD) 알고리즘을 소개합니다. SESD는 숨겨진(diffusion) 구조를 이용하여 적은 양의 데이터로 효과적인 음성 합성을 가능하게 합니다. 이는 U-Audio Transformer (U-AT)라는 새로운 구조를 기반으로 하며, 미리 훈련된 오디오 오토인코더의 잠재 공간(latent space)에서 작동합니다.

- **Technical Details**: SESD는 1k 시간 이하의 음성 데이터로도 인상적인 결과를 달성합니다. U-AT 구조는 오디오 특징을 다운샘플링 한 뒤, transformer 백본을 적용하여 전반적인 음성 특성을 모델링합니다. 이 과정에서 고정된 문자 인식 언어 모델(ByT5-base)의 표현을 조건으로 사용하는 위치 인식 교차 주의 메커니즘(position-aware cross-attention mechanism)을 제안합니다.

- **Performance Highlights**: SESD는 텍스트 전사에서부터 직접 매우 이해하기 쉬운 음성을 합성할 수 있으며, 현재의 TTS(diffusion text-to-speech) 모델보다 월등한 성능을 보입니다. 텍스트 전용 TTS에서 SESD의 단어 오류율(Word Error Rate, WER)은 2.3%로, 자연인 음성과 거의 비슷한 2.2%에 해당합니다. 강연자 유도 합성에서도 SESD는 WER 2.3%, 유사도 점수 0.617로, 62.5배 더 많은 훈련 데이터를 사용하는 VALL-E를 초월하는 성과를 보여줍니다.



### Applications and Advances of Artificial Intelligence in Music Generation:A Review (https://arxiv.org/abs/2409.03715)
- **What's New**: 이 논문은 AI 음악 생성 분야의 최신 연구 발전을 체계적으로 검토하며, 주요 기술, 모델, 데이터셋, 평가 방법 및 다양한 분야에서의 실용적인 응용을 포괄합니다.

- **Technical Details**: 논문은 기호 생성(symbolic generation), 오디오 생성(audio generation), 하이브리드 모델(hybrid models) 등 여러 기술 접근 방식을 체계적으로 분류하고 비교하는 포괄적인 요약 프레임워크를 제공합니다. 최근 Generative Adversarial Networks (GANs) 및 Transformer 아키텍처와 같은 최신 모델이 포함되어 있습니다.

- **Performance Highlights**: AI 음악 생성 기술은 모델 아키텍처와 생성 품질에서 주목할 만한 진전을 이루었으며, 실시간 상호작용 및 다학제 응용 분야에서 AI 음악 생성의 실제 영향을 분석합니다.



### A Different Level Text Protection Mechanism With Differential Privacy (https://arxiv.org/abs/2409.03707)
- **What's New**: 본 논문은 BERT 사전 훈련 모델을 기반으로 단어의 중요도를 다르게 평가하여 차별적 노이즈를 부여하는 방법을 제안하였습니다. 이 방법은 긴 텍스트 데이터의 정보 손실을 최소화할 수 있는 가능성을 보여주며, 텍스트 보호에 효과적임을 입증했습니다.

- **Technical Details**: 이 연구에서 제안된 방법은 BERT 모델을 통해 각 단어의 주목(attention) 가중치를 추출하고, 이들 가중치를 이용해 단어의 중요도를 정량적으로 측정합니다. 이후, 중요도가 높은 단어는 적게, 낮은 단어는 더 많이 변형하는 방법을 통해 텍스트의 의미 전달을 유지하고자 합니다. 이러한 방법은 SST-2 및 QNLI 두 개의 공개 데이터셋에서 실험하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 동일하게 단어를 변형하는 접근 방식에 비해 긴 텍스트에서 의미 손실을 최소화하는 효과를 나타내었으며, 이를 통해 문서 데이터의 유용성을 높이는 데 기여할 수 있음을 보여주었습니다.



### View-Invariant Policy Learning via Zero-Shot Novel View Synthesis (https://arxiv.org/abs/2409.03685)
Comments:
          Accepted to CoRL 2024

- **What's New**: 이번 연구에서는 대규모 비주얼 데이터에서 얻은 지식을 이용하여 다양한 관찰 모드(Observational modality)에 적응할 수 있는 정책 개발을 다룹니다. 특히, 우리는 단일 이미지에서 새로운 시점을 생성하는 (Novel View Synthesis, NVS) 기술을 통해 3D 장면의 사전 지식을 학습합니다.

- **Technical Details**: 본 연구는 View Synthesis Augmentation (VISTA)라 불리는 데이터 증강 기술을 활용하여 단일 관찰 시점에서 훈련된 정책의 강건성을 높이는 방법을 제시합니다. VISTA는 큰 규모의 2D 이미지 데이터 세트를 이용하여 다양한 관찰 시점에서의 카메라 뷰에 적응 가능한 정책을 학습하는 데 중점을 둡니다.

- **Performance Highlights**: VISTA 방법을 적용하여 훈련된 정책은 시뮬레이션 환경과 실제 환경에서의 조작 작업에서 기존 방법론을 초월하는 성능을 보였습니다. 또한, DROID 데이터 세트에서 ZeroNVS 모델을 미세 조정(Finetuning)함으로써 실제 작업에서도 성능 향상을 이루었습니다.



### A method to benchmark high-dimensional process drift detection (https://arxiv.org/abs/2409.03669)
- **What's New**: 이 논문은 제조 공정에서 나오는 멀티 변수 유한 시계열 데이터인 프로세스 커브(process curve)의 변화를 탐지하기 위해 머신 러닝(machine learning) 방법을 조사합니다. 특히, 머신 러닝 알고리즘에 대한 벤치마크를 위해 프로세스 커브를 제어된 방식으로 합성 생성하는 이론적 프레임워크를 소개합니다.

- **Technical Details**: 연구에서는 프로세스 드리프트(process drift) 탐지를 위한 새로운 메트릭, 즉 Temporal Area Under the Curve (TAUC)를 도입합니다. 이 메트릭은 드리프트 세그먼트에 속하는 커브를 잘 드러내는 머신 러닝 모델의 성능을 정량화하는 데 사용됩니다. 또한, 사실적인 데이터 생성이 가능하도록 실제 프로세스 커브 데이터의 입력도 지원한다.

- **Performance Highlights**: 제시된 이론적 프레임워크를 기반으로 한 벤치마크 연구 결과, 여러 유명 머신 러닝 접근 방식을 비교하였으며, TAUC가 드리프트 탐지의 예측 효능을 측정하는 데 효과적임을 입증하였습니다. 이 연구는 프로세스 커브 분석을 위한 머신 러닝 연구의 투명성을 높이는 데 기여할 것으로 기대됩니다.



### Limited but consistent gains in adversarial robustness by co-training object recognition models with human EEG (https://arxiv.org/abs/2409.03646)
- **What's New**: 인간의 EEG (Electroencephalography) 반응과 실세계 이미지를 정렬하여 인공신경망(ANNs)의 적대적 공격에 대한 강건성을 향상시키기 위해 제안된 새로운 방법론을 보여줍니다. 기존의 연구들이 생물학적 데이터를 침습적인 기술로 제한한 반면, 이 연구는 대규모 EEG 데이터를 사용하여 보다 다양하고 자연스러운 자극 조건에서 ANNs의 학습을 개선하였습니다.

- **Technical Details**: 이 연구에서는 ResNet50을 기반으로 하는 모델을 통해 클래스 분류와 EEG 예측이라는 두 가지 작업을 병행하여 훈련했습니다. 두 작업은 공유 레이어와 독립적인 컴포넌트로 구성된 EEG 예측 브랜치로 나뉘어져 있으며, 17개의 EEG 채널에서의 신호를 분석했습니다. 모델의 성능을 높이기 위해 다양한 구조적 변화를 시도하였으며, 테스트 결과 적대적 공격에 대한 강건성이 연구되었습니다.

- **Performance Highlights**: 모델의 EEG 예측 정확도와 적대적 강건성 간에는 상당한 상관관계가 발견되었으며, 특히 자극 시작 후 약 100ms 시점에서 가장 높은 정확도를 보였습니다. 제한적인 효과 크기에도 불구하고 다양한 초기화 및 구조적 변형에 대해 일관된 효과가 나타났습니다. 인간의 EEG 데이터의 활용은 향후 더욱 다양한 자극 조건을 포함한 대규모 데이터셋으로 연구를 확장할 수 있는 가능성을 열어줍니다.



### Multimodal Laryngoscopic Video Analysis for Assisted Diagnosis of Vocal Cord Paralysis (https://arxiv.org/abs/2409.03597)
- **What's New**: 이 논문에서는 laryngoscope (후두경) 비디오에서 음성과 비디오 데이터를 결합하여 자동으로 주요 세그먼트와 메트릭을 추출하는 Multimodal Analyzing System for Laryngoscope (MASL)을 제안합니다. MASL은 glottis 감지와 keyword spotting을 통합하여 환자의 발성을 분석하고 비디오 하이라이트를 정제하여 성대 움직임을 더 잘 검사할 수 있도록 합니다.

- **Technical Details**: MASL은 hue, saturation 및 value (HSV) 변화를 분석하여 비디오 프레임을 식별하는 strobing video extraction 모듈을 포함하고 있습니다. 또한 U-Net을 사용한 두 단계의 glottis segmentation 프로세스를 통해 vocal cord paralysis 검출을 위한 메트릭을 제공합니다. MASL은 glottis 마스크에서 anterior glottic angle waveform (AGAW)을 추정하여 unilateral vocal cord paralysis (UVFP)를 탐지합니다. 각 성대의 AGAW 변동을 비교하여 좌측 및 우측 마비를 구분합니다.

- **Performance Highlights**: 실험 결과, MASL의 세그멘테이션 모듈이 매우 유능하며 LVP(좌측 성대 마비) 및 RVP(우측 성대 마비) 진단을 위한 신뢰할 수 있는 메트릭을 제공하는 것을 입증했습니다. MASL은 특히 디퓨전 기반의 정제 프로세스를 통해 정확성을 향상시켜 false positive를 줄이는 데 효과적입니다.



### 100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances (https://arxiv.org/abs/2409.03563)
Comments:
          Presented at the 2024 KDD workshop on Evaluation and Trustworthiness of Generative AI Models

- **What's New**: 본 연구에서는 새로운 Large Language Model (LLM)의 성능을 예측하기 위해 기존에 평가된 LLM의 결과를 활용하는 방법을 제안합니다. 구체적으로, 우리는 새로운 LLM을 소규모의 참조 인스턴스 집합에서 평가하고, 이를 바탕으로 범용 assessor를 훈련하여 성능을 예측합니다.

- **Technical Details**: 우리는 HELM-Lite 및 KindsOfReasoning이라는 기존의 추론 데이터셋을 사용하여 여러 OpenAI 모델을 평가하였습니다. 제안된 방법은 LLM의 특정 성능 벡터와 인스턴스 특정 특징을 결합하여 범용 assessor를 훈련하는 방식으로 구성됩니다. 평가된 인스턴스의 분포가 기존 assessor 훈련에 사용된 인스턴스의 분포와 동일할 경우, 제안된 방법이 LLM-특정 assessor와 유사한 성능을 발휘하는 것을 발견했습니다.

- **Performance Highlights**: 우리는 랜덤으로 선택된 참조 인스턴스들이도 고급 선택 방법들과 비슷한 성능을 보인다는 것을 찾았습니다. 하지만 분포에서 벗어난 경우, 모든 assessor의 예측력이 크게 감소하여 LLM의 본질적인 예측 가능성이 낮다는 것을 시사합니다.



### DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architectur (https://arxiv.org/abs/2409.03550)
- **What's New**: 이 논문에서는 데이터 없는 지식 증류(Data-Free Knowledge Distillation, DKDM)를 제안하여 확산 모델(Diffusion Models, DMs)의 속도를 두 배로 증가시키며, 고품질 샘플을 생성할 수 있는 새로운 방법론을 소개합니다.

- **Technical Details**: DKDM은 두 가지 주요 구성 요소로 이루어져 있습니다: 첫째, 사전 훈련된 DMs가 생성한 합성 데이터(denoising data)를 사용하여 빠른 DMs를 최적화하는 DKDM 목표(Objective)입니다. 둘째, 최적화 과정에서의 병목현상을 방지하기 위해 유연하게 합성 데이터를 조직하는 동적 반복 증류(Dynamic Iterative Distillation) 방법이 포함됩니다. 이 방법은 DMs의 성능을 실제 데이터 없이 향상시키도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, DKDM은 DMs의 생성 속도를 2배 향상시키면서도 기본 성능을 유지하는 것으로 나타났습니다. 또한, 사전 훈련된 DMs가 새로운 DMs 훈련을 위한 "데이터 셋" 역할을 할 수 있음을 보여주어, 데이터 저장 요구사항을 줄이는 데 기여할 수 있습니다.



### Prediction Accuracy & Reliability: Classification and Object Localization under Distribution Shif (https://arxiv.org/abs/2409.03543)
Comments:
          This preprint has not undergone any post-submission improvements or corrections

- **What's New**: 본 논문은 자연 분포 변동이 CNN의 인식 성능에 미치는 영향을 분석한 연구로, 날씨 변화에 따른 데이터 증강이 탐지 품질과 신뢰도 추정에 미치는 영향도 평가합니다.

- **Technical Details**: 자율주행 데이터셋을 기반으로 한 새로운 데이터셋을 구성하였으며, 여섯 가지의 분포 변화 데이터셋은 악천후 시나리오, 시뮬레이션된 비와 안개, 그리고 분포 외 데이터(Out-of-Distribution data)를 포함합니다.

- **Performance Highlights**: ConvNeXt-Tiny는 EfficientNet-B0보다 더 강한 내구성을 보여주며, 폭우가 분류의 성능에 더 큰 영향을 미치는 반면, 안개에서는 상대적으로 덜 영향을 미치는 것으로 나타났습니다. 또한, MC-Dropout을 특정 레이어에 통합할 경우 작업 성능과 신뢰도 추정 개선의 가능성이 있지만, 이 레이어의 선택은 분포 변동의 유형과 고려하는 작업에 따라 달라집니다.



### LMLT: Low-to-high Multi-Level Vision Transformer for Image Super-Resolution (https://arxiv.org/abs/2409.03516)
- **What's New**: 이 논문은 Low-to-high Multi-Level Transformer (LMLT)라는 새로운 모델을 제안하여 기존 Vision Transformer (ViT) 기반의 이미지 초해상도(Image Super-Resolution) 방법에서 발생하는 복잡성과 메모리 사용량 문제를 해결합니다.

- **Technical Details**: LMLT는 각 헤드에 대해 다양한 크기의 특성을 사용하는 attention 메커니즘을 통해 이미지를 처리합니다. 모델은 채널 차원에 따라 이미지 특성을 나누고 낮은 헤드에 대해 공간 크기를 점진적으로 감소시킵니다. 각 헤드는 self-attention을 적용하여 지역 정보와 전역 정보를 효과적으로 포착합니다. 이 접근 방식은 window 경계 문제를 해결합니다.

- **Performance Highlights**: LMLT는 기존 ViT 기반 이미지 초해상도 모델들에 비해 메모리 사용량을 각각 38%와 54% 감소시키며, 추론 시간도 각각 22%와 19% 감소시킵니다. 모든 벤치마크 데이터셋에서 평균 성능이 각각 0.076db와 0.152db 향상되었습니다.



### Disclosure of AI-Generated News Increases Engagement but Does Not Reduce Aversion, Despite Positive Quality Ratings (https://arxiv.org/abs/2409.03500)
- **What's New**: 이 연구는 언론에서 인공지능(AI) 활용의 대중적 인식을 조사하였으며, AI가 생성한 뉴스 기사가 인간이 작성한 기사와 품질 면에서 동등하게 평가된다는 점을 발견했습니다. 또한, AI 관련 정보를 공개함으로써 콘텐츠에 대한 즉각적인 참여를 유도할 수 있음을 시사합니다.

- **Technical Details**: 이 연구는 599명의 스위스 독일어 사용자를 대상으로 진행된 서베이 실험(비교실험)을 통해 뉴스 기사의 신뢰도, 가독성, 전문성을 평가하였습니다. 참가자들은 저널리스트가 작성한 기사가 포함된 통제 그룹, AI가 재작성한 기사가 포함된 AI 보조 그룹, AI가 완전히 생성한 기사가 포함된 AI 생성 그룹으로 나누어졌습니다.

- **Performance Highlights**: 연구 결과, AI가 생성한 기사에 대한 장기적인 수용은 낮지만, AI의 개입을 알고 나면 기사에 대한 즉각적인 참여가 증가하는 것으로 나타났습니다. 이는 AI 사용에 대한 불만이 주로 품질에 관한 것이 아님을 보여주며, 언론에서 AI의 투명한 사용이 진정한 독자 참여를 높일 수 있음에 주목하고 있습니다.



### Improving Uncertainty-Error Correspondence in Deep Bayesian Medical Image Segmentation (https://arxiv.org/abs/2409.03470)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이 연구에서는 Accuracy-vs-Uncertainty (AvU) 손실 함수를 사용하여, 불확실성이 정확하지 않은 영역에만 존재하도록 하는 새로운 방법론을 제시합니다. 이는 의료 영상 세분화에서의 자동 오류 탐지를 위해, 불확실성을 최대한 활용하는 방식을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: Deep Bayesian 모델을 기반으로 한 FlipOut 모델을 훈련시키고, AvU 손실을 적용하여 불확실성이 정확한 볼륨(voxel)에서는 감소하고 부정확한 볼륨에서는 유지되도록 합니다. 이를 통해 반자동 품질 평가(quality assessment, QA)의 유용성을 높이고, 의료 영상 세분화의 효율성을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 베이지안 기준선 모델에 비해 정확한 볼륨에서 불확실성을 성공적으로 억제하고, 부정확한 볼륨에서는 유사한 수준의 불확실성을 유지하는 것으로 나타났습니다. 이 연구는 방사선 치료를 포함한 다양한 데이터셋에 걸쳐 성과를 평가하였습니다.



### Characterizing Massive Activations of Attention Mechanism in Graph Neural Networks (https://arxiv.org/abs/2409.03463)
- **What's New**: 이 논문은 Graph Neural Networks (GNNs)에 주목하며, attention 메커니즘이 GNNs에 통합되면서 발생하는 Massive Activations (MAs)에 대한 최초의 종합적인 연구를 선보입니다. MAs는 attention 레이어 내에서 발생하는 중요한 현상으로, 이 연구는 이를 탐지하고 분석하는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 연구는 GraphTransformer, GraphiT, SAN과 같은 다양한 GNN 아키텍처의 edge features가 MAs 발생에 미치는 영향을 분석합니다. MAs의 정의와 탐지 방법은 activation ratio 분포에 기반하고, 다양한 벤치마크 데이터셋(ZINC, TOX21, PROTEINS)을 활용하여 실험을 수행합니다. 또한, Explicit Bias Term (EBT)을 도입하여 MAs를 저지하기 위한 잠재적 수단으로 탐구합니다.

- **Performance Highlights**: 연구 결과는 attention에 의해 유도된 MAs가 다양한 아키텍처에서 발생한다는 것을 강조하며, 이는 GNN의 성능과 해석 가능성에 중대한 영향을 미칩니다. 또한 MAs의 분석 범위를 확장하고, GNN의 성능과 내구성 향상을 위한 기초를 마련합니다.



### How Much Data is Enough Data? Fine-Tuning Large Language Models for In-House Translation: Performance Evaluation Across Multiple Dataset Sizes (https://arxiv.org/abs/2409.03454)
- **What's New**: 본 연구는 Llama 3 8B Instruct 모델을 소프트웨어 분야의 특정 조직에서 얻은 번역 메모리(Translation Memories, TMs)를 활용하여 미세 조정(fine-tuning)함으로써 번역의 정확도와 효율성을 향상시키는 방법을 탐구합니다. 특히, 다양한 데이터 세트 사이즈의 변화가 번역 품질에 미치는 영향을 분석했습니다.

- **Technical Details**: 연구는 총 다섯 가지 언어(브라질 포르투갈어, 체코어, 독일어, 핀란드어 및 한국어)로의 번역 실험을 진행하였으며, 다양한 사이즈(1k에서 207k 세그먼트)에 해당하는 데이터 세트를 이용하여 Llama 3 모델을 조정했습니다. 각 세트에 대해 BLEU, chrF++, TER, COMET와 같은 자동 평가 메트릭을 사용하여 성능을 평가했습니다.

- **Performance Highlights**: 가장 큰 학습 세트를 사용할 경우, BLEU 점수가 평균 13점, COMET 점수가 25점 상승하는 등 모든 메트릭에서 번역 성능이 개선되었습니다. 그러나 적은 양(1k 및 2k)의 예제로 미세 조정할 경우에는 성능 저하가 발생하였지만, 학습 데이터 세트의 크기가 증가하면서 상당한 개선이 있었습니다.



### Fine-tuning large language models for domain adaptation: Exploration of training strategies, scaling, model merging and synergistic capabilities (https://arxiv.org/abs/2409.03444)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)를 특정 도메인에 맞추어 조정하기 위한 다양한 미세 조정(fine-tuning) 전략의 효과를 탐구합니다. 특히 Continued Pretraining (CPT), Supervised Fine-Tuning (SFT), 그리고 Preference-based 최적화 접근법을 분석하였습니다.

- **Technical Details**: 분석 결과, 여러 개의 미세 조정된 모델을 결합하면, 각 모델의 개별적인 기여도를 초월하는 새로운 기능이 발생할 수 있음을 알았습니다. Llama 3.1 8B와 Mistral 7B 모델을 포함한 다양한 모델 아키텍처에 대한 실험이 진행되었으며, 결과는 유사한 행동을 보여주었습니다. 또한, 17억 개의 파라미터를 가진 소형 LLM에서도 모델 병합에 따른 emergent capabilities를 관찰했습니다.

- **Performance Highlights**: 특히, 매우 작은 LLM 모델이 여러 중요한 기준(사고 심도, 창의성, 명확성, 정량적 정밀도)에서 높은 지능 점수를 기록한 점이 주목할 만합니다. 이 외에도 생물학적 재료 디자인 개념을 기반으로 새로운 마이크로구조, 건축 개념, 도시 디자인을 생성하기 위한 이미지 생성 프롬프트 개발 실험이 포함되어 있습니다.



### KiloBot: A Programming Language for Deploying Perception-Guided Industrial Manipulators at Sca (https://arxiv.org/abs/2409.03439)
- **What's New**: 이 논문에서는 산업용 로봇이 비구조적 환경을 처리하기 위해 Perception-guided manipulation의 Domain-Specific Language (DSL)을 제안합니다. 이 DSL은 로봇 조작자들이 복잡한 조작 행동을 쉽게 설계할 수 있도록 드래그 앤 드롭 인터페이스를 제공합니다.

- **Technical Details**: 제안된 DSL은 Task and Motion Planning (TAMP) 문제의 하위 클래스에 대한 접근 가능한 인터페이스를 제공하며, 사용자 정의 요구 사항에 맞춰 유연한 제어 흐름을 구현할 수 있도록 설계되었습니다. 이를 통해 로봇 행동을 쉽게 구성하고, 사용자 정의 TAMP 문제를 통합할 수 있는 사용 친화적인 인터페이스를 제공합니다.

- **Performance Highlights**: 우리의 DSL은 전 세계 10,000개 이상의 로봇 작업 공간에서 성공적으로 배포되었으며, 테스트 결과에서 높은 처리량(# of pick-place per minute)을 보여주었습니다. 로봇 조작자는 몇 시간의 교육 후에도 복잡한 조작 동작을 쉽게 조정할 수 있습니다.



### Reinforcement Learning Approach to Optimizing Profilometric Sensor Trajectories for Surface Inspection (https://arxiv.org/abs/2409.03429)
- **What's New**: 본 논문은 표면 결함 검사를 최적화하기 위해 로봇 Inspect Trajectories를 개선하는 새로운 Reinforcement Learning(RL) 기반 접근 방식을 제시합니다. 이 방법은 Boustrophedon 스캔 방식에 기초하며, Sensor의 위치와 경사를 동적으로 조정하여 최적의 방향성과 거리를 유지하면서 균일한 프로파일 분포를 확보합니다.

- **Technical Details**: 이 연구는 생체 CAD 모델을 기반으로 한 시뮬레이션 환경에서 Sensor의 경로를 계획하며, Proximal Policy Optimization(PPO) 알고리즘을 사용하여 RL 에이전트를 훈련합니다. 이 모델은 State Space, Action Space 및 Reward Function을 정의하여 profilometric 센서를 사용하는 검사 전용으로 디자인되었습니다.

- **Performance Highlights**: 텍스트 모델은 시뮬레이션에서 다양한 파트에 대해 훈련되어 검증되었으며, UR3e 로봇 팔 모델을 사용하여 CAD 모델로부터 오프라인으로 생성된 최적화된 경로를 실제로 실행함으로써 실험을 진행했습니다. 이 연구는 다양한 로봇 검사의 가능성을 제시하며 높은 정확성을 갖춘 결함 검출을 가능하게 합니다.



### KAN See In the Dark (https://arxiv.org/abs/2409.03404)
- **What's New**: 이 논문은 저조도 이미지 향상(low-light image enhancement, LLIE) 작업에 Kolmogorov-Arnold Network (KANs)를 처음으로 도입하고, KAN-Block을 설계하여 기존 방법의 한계를 극복하는 혁신적인 접근을 제안하고 있습니다.

- **Technical Details**: KANs는 스플라인 기반의 컨볼루션 레이어와 학습 가능한 활성화 함수를 특징으로 하며, 비선형 의존성을 효과적으로 포착할 수 있습니다. 저자는 KAN-Block을 설계하여 U-Net 구조에 통합하고, 각 단계에서 이미지를 재구성하며 Fast Fourier Transform (FFT)를 사용하여 주파수 도메인 인식을 도입하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해 저자의 방법이 벤치마크 데이터셋에서 경쟁력 있는 성능을 보였으며, 해석 가능성 및 모델 성능을 모두 개선하였습니다.



### Hardware Acceleration of LLMs: A comprehensive survey and comparison (https://arxiv.org/abs/2409.03384)
Comments:
this https URL

- **What's New**: 본 논문은 대형 언어 모델(LLMs)을 위한 트랜스포머 네트워크의 하드웨어 가속 기술을 포괄적으로 조사한 연구이다. 다양한 연구 노력과 이들이 제안한 프레임워크를 비교 분석하여, FPGA, ASIC, GPU 등 다양한 처리 플랫폼에서의 성능과 에너지 효율성을 평가한다.

- **Technical Details**: 논문은 다양한 하드웨어 가속기들이 어떻게 LLM의 성능을 개선하는지에 대한 비교를 포함한다. 연구에서는 FPGAs와 GPUs를 기반으로 한 프레임워크의 성능(GOPs) 및 에너지 효율(GOPs/W)을 정량적으로 분석하며, 서로 다른 공정 기술을 사용한 다양한 구현이 공정하게 비교되는 한계를 극복하기 위해 결과를 동일한 기술로 외삽하여 공정한 비교를 이룬다.

- **Performance Highlights**: FTRANS라는 FPGA 기반 가속기는 RTX5000 GPU와 비교했을 때 81배 더 빠르고 9배 더 에너지 효율적임을 입증하였다. NPE 아키텍처는 CPU와 GPU에서 각각 4배 및 6배의 에너지 효율성을 보여주며, 다른 FPGA 플랫폼에서도 CPU 대비 11배, GPU 대비 2배 빠른 속도를 달성하였다.



### CogniDual Framework: Self-Training Large Language Models within a Dual-System Theoretical Framework for Improving Cognitive Tasks (https://arxiv.org/abs/2409.03381)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 인간의 인지 시스템인 Kahneman의 이중 시스템 이론을 모방할 수 있는 가능성을 제시합니다. 연구에서는 LLMs의 자기 훈련(self-training)을 통해 의도적인 추론(System 2)에서 직관적인 응답(System 1)으로 진화할 수 있는 방법을 탐구합니다.

- **Technical Details**: 제안된 CogniDual Framework for LLMs(CFLLMs)는 LLM들이 합리적인 응답을 제공하도록 훈련되며, CoT(Chain of Thought) 유도 없이는 즉각적인 반응을 생성할 수 있는지를 확인하기 위한 방법론을 포함합니다. 연구는 Vicuna와 Llama2 모델을 다양한 크기로 평가하며, GSM8K, ReClor, LogiQA 2.0과 같은 추론 데이터 세트에서 성능을 검사합니다.

- **Performance Highlights**: 자기 훈련 후, LLMs는 CoT가 없는 상황에서도 응답 정확도가 크게 향상되었음을 보여주었습니다. 이는 LLMs가 인간의 인지 시스템과 유사하게 직관적인 응답 메커니즘을 개발할 수 있음을 의미합니다. 또한 새로운 방법이 CoT에 의존하지 않고도 효율적이고 정확한 출력을 유지할 수 있는 가능성을 제시합니다.



### Raw Speech Enhancement with Deep State Space Modeling (https://arxiv.org/abs/2409.03377)
Comments:
          7 pages, 2 figures

- **What's New**: aTENNuate 네트워크는 실시간 원시 음성 신호의 제거를 목적으로 설계된 심층 상태 공간(autoencoder) 모델로, 최신 심층학습 기술을 통해 음성 신호에서의 노이즈 제거 성능을 크게 향상시켰습니다.

- **Technical Details**: aTENNuate는 깊은 상태 공간 모델(SSM)로, 장기간의 시간적 의존성을 포착할 수 있는 안정적인 선형 재귀 유닛을 활용합니다. 이 신경망은 IIR 커널을 사용하여 입력 피처 처리에 긴 합성 곱 커널을 적용하고, FFT 합성곱을 사용하여 병렬 처리가 가능하도록 설계되었습니다.

- **Performance Highlights**: aTENNuate은 PESQ 점수, 파라미터 수, MACs 및 지연 시간을 포함하여 기존의 실시간 노이즈 제거 모델을 초월하는 성능을 보였습니다. 4000Hz 및 4비트로 압축된 노이즈 입력에서도 높은 충실도와 최소한의 알아볼 수 없는 아티팩트를 유지하며, 다양한 저자원 환경에서도 효과적인 음성 증강 기능을 제공합니다.



### Leveraging Large Language Models through Natural Language Processing to provide interpretable Machine Learning predictions of mental deterioration in real tim (https://arxiv.org/abs/2409.03375)
- **What's New**: 본 논문은 인공지능(AI) 및 자연어처리(NLP) 기술을 활용하여 인지 저하에 대한 해석 가능한 기계 학습 예측을 실시간으로 제공하는 챗봇 솔루션을 제안합니다. 전통적인 접근법의 한계를 극복하기 위한 노력의 일환으로, 최신 자연어 처리 기법을 적용하여 개인 맞춤형 진단 시스템을 개발하였습니다.

- **Technical Details**: 제안된 파이프라인은 (i) NLP 기반 프롬프트 엔지니어링을 통한 데이터 추출, (ii) 특징 엔지니어링을 포함한 스트림 기반 데이터 처리, (iii) 실시간 분류, (iv) 예측 결과의 시각적 및 자연어 설명을 제공하는 설명 가능성 대시보드로 구성됩니다. 언어-개념적 특징을 활용하여 자연어 분석을 수행하고 '할루시네이션' 효과를 피하기 위한 프롬프트 엔지니어링을 적용하였습니다.

- **Performance Highlights**: 모든 평가 지표에서 분류 결과가 80%를 초과하며, 특히 인지 저하 클래스의 리콜 값은 약 85%에 달합니다. 이를 통해 저렴하고 유연하며 비침습적인 개인 맞춤형 진단 시스템을 제공함으로써 실시간으로 인지 저하를 모니터링할 수 있는 가능성을 제시합니다.



### Sketch: A Toolkit for Streamlining LLM Operations (https://arxiv.org/abs/2409.03346)
- **What's New**: 이 논문에서는 다양한 NLP 작업을 지원하는 LLM(대규모 언어 모델) 운영을 단순화하기 위한 혁신적인 도구 키트인 Sketch를 소개합니다. Sketch는 사용자가 구조적 출력을 효율적으로 구축하고 제어할 수 있도록 설계되었습니다.

- **Technical Details**: Sketch는 (1) 여러 NLP 작업을 포함하는 작업 설명 스키마와 프롬프트 템플릿 모음, (2) 사용자 친화적이며 대화형인 구조적 출력 LLM 서비스 구축 과정, (3) 출력 형식 제어를 위한 오픈 소스 데이터셋과 데이터셋 구축 도구, (4) 출력 형식 지침을 이해하고 준수하는 LLaMA3-8B-Instruct 기반의 오픈 소스 모델 등을 포함합니다. Sketch는 'plug-and-play' 기능을 통해 다양한 응용 프로그램에 적합하게 설계되었습니다.

- **Performance Highlights**: Sketch는 자연어 처리를 위한 다양한 작업에서의 성능 향상과 함께 출력 형식에 대한 수요를 충족시키기 위한 우수한 구조적 답변 생성을 가능하게 합니다. 이렇게 함으로써, LLM의 신뢰성과 정확성을 높이고 사용자 경험을 정교하게 개선하는 것을 목표로 합니다.



### YOLO-PPA based Efficient Traffic Sign Detection for Cruise Control in Autonomous Driving (https://arxiv.org/abs/2409.03320)
- **What's New**: 이 논문에서는 자율 주행 시스템에서 교통 신호를 효율적이고 정확하게 탐지하기 위한 새로운 YOLO PPA 기반의 교통 신호 탐지 알고리즘이 제안되었습니다.

- **Technical Details**: 기존의 object detection 알고리즘이 작은 크기의 교통 신호를 탐지하기 어려운 문제를 해결하기 위해, YOLO PPA를 기반으로 한 방법을 개발했습니다. 실험은 GTSDB 데이터셋을 사용하여 수행되었으며, 이 결과 제안된 방법이 원래 YOLO보다 추론 효율성을 11.2% 향상시켰음을 보여주고 있습니다.

- **Performance Highlights**: mAP 50도 93.2% 향상되었으며, 이는 제안된 YOLO PPA의 유효성을 입증합니다.



### N-gram Prediction and Word Difference Representations for Language Modeling (https://arxiv.org/abs/2409.03295)
- **What's New**: 이번 연구는 인과 언어 모델링(causal language modeling, CLM) 작업을 위한 간단한 N-그램 예측 프레임워크를 소개합니다. 이 프레임워크는 기존 CLM 모델에 쉽게 통합할 수 있으며, 구조의 복잡성을 줄였다는 특징이 있습니다.

- **Technical Details**: 이 연구에서는 N-그램(n-gram) 예측을 통해 모델 훈련 중 단어 차이 표현(word difference representation, WDR)을 서브레이트(replace) 형태로 활용합니다. WDR은 인접한 단어들 간의 임베딩 벡터 차이를 이용하여 컨텍스트에 따라 다양해진 목표 표현(target representation)을 제공합니다. 또한, N-그램 예측의 결과를 활용한 앙상블(ensemble) 방법을 제안하여 품질 향상을 꾀합니다.

- **Performance Highlights**: 실험 결과, 제안된 간단한 N-그램 프레임워크, WDR의 다양한 목표 표현, 그리고 앙상블 방법이 기존 CLM 모델들보다 수렴성(perplexity)에서 유의미한 개선을 보였음이 입증되었습니다. NMT(신경 기계 번역) 작업에서도 제안된 방법의 활용 가능성과 장점을 보여주었습니다.



### LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts (https://arxiv.org/abs/2409.03291)
Comments:
          20 pages, 7 tables, 13 figures, under consideration for EMNLP

- **What's New**: 본 연구는 대규모 Language Models (LLMs) 이용한 정보작전에서 기존 LLM 탐지기들의 실용성 부족을 드러냅니다. 특히, 중간 정도의 공격자에 의해 생성된 짧은 뉴스 형식의 포스트를 목표로 하는데요, 이 설정에서의 검증이 부족했습니다.

- **Technical Details**: 연구팀은 LLM 탐지기의 기존의 제로샷(zero-shot) 접근법이 일관성이 없고 샘플링 온도 증가와 같은 간단한 공격에 취약하다는 점을 강조했습니다. 맞춤형 탐지기를 개발할 수 있지만, 이는 새로운 인간 작성 텍스트에 대한 일반화에 실패했습니다. 이는 특정 도메인에 대한 벤치마킹 필요성을 시사합니다.

- **Performance Highlights**: 연구 결과, 현재 LLM 탐지기가 LLM이 생성하는 허위 정보를 탐지하기에는 준비가 되어 있지 않으며, 일반화 및 과적합 간의 trade-off 문제가 존재함을 밝혔다.



### Recent Advances in Attack and Defense Approaches of Large Language Models (https://arxiv.org/abs/2409.03274)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 현재 취약점과 위협에 대한 연구 현황을 요약하고, 이러한 모델들의 보안을 강화하기 위한 향후 연구 방향을 제안합니다. LLM이 생성하는 고품질 텍스트의 장점에도 불구하고, 모델의 안전성과 신뢰성 문제를 해결하는 것이 중대한 과제가 되고 있다는 점을 강조합니다.

- **Technical Details**: LLM의 취약성에는 비편향성(bias), 유해 콘텐츠(harmful content), 환각(hallucinations), 프라이버시 위험(privacy risks) 등 여러 가지가 포함됩니다. 이전 연구에서 발견된 사전 지식과 공격 기법들이 더욱 발전된 공격 방법에 의해 활용되며, 공격자들은 개선된 모델 기능과 통합을 통해 확장된 공격 면(target surface)을 노리고 있습니다.

- **Performance Highlights**: 현재 LLM 방어 전략은 공격에 대한 반응이 제한적이며, 기존 방어 방법의 한계를 극복하기 위한 연구 공백이 존재합니다. 논문은 LLM의 안전성이 부각되는 동시에, 공격 방법과 방어 전략의 상호작용을 분석하여, 향후 연구 방향과 방어 전략의 개선 방안을 제안합니다.



### Bones Can't Be Triangles: Accurate and Efficient Vertebrae Keypoint Estimation through Collaborative Error Revision (https://arxiv.org/abs/2409.03261)
Comments:
          33 pages, ECCV 2024, Project Page: this https URL

- **What's New**: 이번 연구에서는 사용자의 개입을 최소화하면서도 높은 정확도를 달성하는 KeyBot이라는 새로운 방법을 소개합니다. 이 시스템은 기존 모델에서 발생하는 대표적인 오류들을 식별하고 수정하는 기능을 내장하고 있습니다.

- **Technical Details**: KeyBot은 vertebrae keypoint estimation에 특화된 자동 오류 수정 시스템으로, 사용자 입력 없이도 주요 오류를 사전 식별하여 수정합니다. 이 과정은 두 가지 구성 요소, 즉 detector와 corrector로 이루어져 있으며, 합성 데이터(synthetic data)를 사용하여 훈련됩니다.

- **Performance Highlights**: KeyBot은 AASCE 데이터셋에서 평균 방사 오류(MRE)를 19% 줄이고, 목표 성능 달성을 위한 사용자 클릭 수(NoC)를 17% 감소시키며, 기존 방법 대비 월등한 성능을 보였습니다. 이는 vertebrae keypoint estimation의 정확성을 크게 향상시키는 데 기여합니다.



### Understanding LLM Development Through Longitudinal Study: Insights from the Open Ko-LLM Leaderboard (https://arxiv.org/abs/2409.03257)
- **What's New**: 본 논문은 Open Ko-LLM Leaderboard를 분석하여 한국어 대규모 언어 모델(LLMs)의 발전 상황을 11개월 동안 조사한 장기적인 연구를 소개합니다. 이전 연구는 5개월이라는 제한된 관찰 기간에 의존하였으나, 이번 연구는 보다 포괄적인 이해를 제공합니다.

- **Technical Details**: 본 연구는 1,769개의 모델을 분석하였고, LLM 성능의 변화를 다섯 개의 과제를 통해 모니터링했습니다. 연구 질문은 LLM 성능 향상의 구체적인 도전 과제, 모델 크기가 과제 성능의 상관관계에 미치는 영향, 그리고 Open Ko-LLM Leaderboard에서의 랭킹 변화입니다.

- **Performance Highlights**: 성능 향상에서 특정 과제, 예를 들어 Ko-HellaSwag와 Ko-TruthfulQA는 빠른 개선을 보였으나, Ko-MMLU와 Ko-CommonGEN V2는 느리면서도 지속적인 발전을 보였습니다. 모델 크기가 커질수록 과제 간의 상관관계가 증가했으며, 이는 모델 성능 향상에 긍정적인 영향을 미쳤습니다.



### E2CL: Exploration-based Error Correction Learning for Embodied Agents (https://arxiv.org/abs/2409.03256)
- **What's New**: 이 논문에서는 Exploration-based Error Correction Learning (E2CL)이라는 새로운 프레임워크를 제안하여 LM 기반 에이전트의 환경 정렬을 개선합니다. 이 프레임워크는 탐사를 통한 오류 및 환경 피드백을 활용하여 에이전트의 자가 수정 능력을 향상시키고 있습니다.

- **Technical Details**: E2CL은 탐사로 인한 오류와 환경 피드백을 통합하여 LM 기반 에이전트가 목표 환경과 철저히 정렬될 수 있도록 돕습니다. 이 프레임워크는 두 가지 탐사 방식을 포함하는데, 하나는 선생님이 지도하는 탐사(teacher-guided exploration)이고, 다른 하나는 비지도 탐사(teacher-free exploration)입니다. 이를 통해 환경과의 상호작용에서 에러 행동과 올바른 행동의 피드백을 수집합니다. 또한, 고안된 Speculative Inference 알고리즘을 통해 초기 계획된 행동이 오류로 간주될 경우 수정이 이루어지도록 합니다.

- **Performance Highlights**: Virtualhome 환경에서 E2CL로 훈련된 에이전트는 다른 기준 방법들로 훈련된 에이전트보다 뛰어난 성능을 보이며, 자가 수정 능력에서도 우수한 결과를 나타냈습니다. E2CL로 훈련된 소규모 모델은 동일 시리즈의 대규모 모델보다 더 나은 성능을 발휘하였으며, 피드백 기반 재계획 평가에서도 LLM과 유사한 자가 수정 능력을 보여주었습니다.



### Granular-ball Representation Learning for Deep CNN on Learning with Label Nois (https://arxiv.org/abs/2409.03254)
- **What's New**: 이번 연구에서는 CNN 모델에 통합할 수 있는 일반적인 granular-ball computing (GBC) 모듈을 제안합니다. 이 모듈은 각 개별 샘플 대신 granular-ball 샘플의 레이블을 예측합니다.

- **Technical Details**: GBC 모듈은 특징 수준에서 입력 샘플을 분리하여 각각의 granular-ball 샘플을 생성합니다. 이 모듈은 전방 전달 과정 및 역전파 과정에서 독특한 경량화 및 안정적인 교육 프로세스를 구현합니다.

- **Performance Highlights**: 제안된 GBC 방법은 CNN 모델의 강인성을 향상시키며, 추가 데이터나 최적화 없이도 이미지 분류 작업에서 효과적인 성능을 보여줍니다.



### DiffGrad for Physics-Informed Neural Networks (https://arxiv.org/abs/2409.03239)
Comments:
          20 pages, 14 figures

- **What's New**: 이 논문은 Burgers' equation을 해결하기 위해 Physics-Informed Neural Networks (PINNs)와 새로운 DiffGrad 방법을 결합한 전략을 소개합니다. 이 접근법은 현재와 직전 기울기(gradients)의 차이를 활용하여 성능을 향상시킵니다.

- **Technical Details**: PINNs는 비선형 문제를 해결하는 최첨단 도구로, 본 논문에서는 Adam, Adamax, RMSprop, DiffGrad와 같은 다양한 최적화 알고리즘을 사용하여 Burgers' equation의 풀이를 비교합니다. DiffGrad는 이전 기울기를 고려하지 않는 기존의 Adam 최적화 방식의 한계를 극복합니다.

- **Performance Highlights**: DiffGrad는 다른 최적화 방법들에 비해 솔루션의 정확성을 개선하고 훈련 시간을 단축하는 데 효과적임을 입증합니다. 다양한 시간 간격에서의 솔루션 시각화를 통해 네트워크의 정확성을 확인했습니다.



### Content Moderation by LLM: From Accuracy to Legitimacy (https://arxiv.org/abs/2409.03219)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)을 온라인 플랫폼의 콘텐츠 조정을 위해 사용하는 방법에 대한 최신 연구를 다루고 있습니다. 대부분의 연구가 정확성(accuracy)에 중점을 두고 있는 반면, 이 논문은 정확성만으로는 부족하며, 합법성(legitimacy)을 기준으로 하는 새로운 평가 프레임워크를 제안합니다.

- **Technical Details**: LLM의 역할은 단순히 콘텐츠 조정의 정확성을 높이는 것이 아니라, 조정 결정의 합법성을 높이는 것입니다. 쉬운 사례(easy cases)와 어려운 사례(hard cases)를 구분하고 각각 다른 기준으로 평가하는 새로운 프레임워크를 제안하며, 이를 통해 LLM이 제공할 수 있는 다양한 가치 있는 역할을 강조합니다.

- **Performance Highlights**: 이 프레임워크에 따르면, LLM은 정확성을 높이는 것 외에도 쉽고 어려운 사례를 사전 필터링하거나, 조정 결정에 대한 질 높은 설명을 제공하고, 인간 검토자가 더 많은 문맥 정보를 얻도록 돕고, 사용자의 참여를 촉진하는 등 다양한 방식으로 플랫폼 거버넌스를 향상시킬 수 있는 잠재력을 지닙니다.



### xLAM: A Family of Large Action Models to Empower AI Agent Systems (https://arxiv.org/abs/2409.03215)
Comments:
          Technical report for the Salesforce xLAM model series

- **What's New**: xLAM이라는 새롭고 특화된 대형 행동 모델 시리즈가 개발되어 공공에 공개되었습니다. 이 모델들은 AI 에이전트 작업을 위해 설계되었습니다.

- **Technical Details**: xLAM 시리즈는 1B에서 8x22B 파라미터에 이르는 5개의 모델로 구성되어 있으며, 밀집(Dense) 및 혼합 전문가(Mixture-of-Expert) 아키텍처를 사용합니다. 이러한 모델은 다양한 데이터셋을 통합, 증강 및 합성하여 AI 에이전트의 일반화(Generalizability)와 성능을 향상시키는 확장 가능하고 유연한 파이프라인을 사용하여 훈련됩니다.

- **Performance Highlights**: xLAM은 여러 에이전트 능력 벤치마크에서 뛰어난 성능을 보여주었으며, 특히 Berkeley Function-Calling Leaderboard에서 1위를 기록하며 GPT-4와 Claude-3 등 여러 모델을 능가했습니다. 이는 도구 사용(tool use) 측면에서의 성능 향상을 뜻합니다.



### TC-LLaVA: Rethinking the Transfer from Image to Video Understanding with Temporal Considerations (https://arxiv.org/abs/2409.03206)
- **What's New**: 이번 연구에서는 비디오 이해(Task) 과제를 위한 Temporal-Considered LLaVA (TC-LLaVA)라는 새로운 비디오-언어 프레임워크를 제안합니다. 이 모델은 MLLMs(다중모달 대형 언어 모델)의 시간적 인식 능력을 강화하고 텍스트와 비디오 모달리티 간의 주의 상호작용을 차별화하는 두 가지 주요 전략에 초점을 맞추고 있습니다.

- **Technical Details**: 첫 번째 접근법인 Temporal-Aware Dual RoPE는 각 토큰에 고유한 위치 ID를 부여하여 원래 RoPE를 보존하는 방법으로, 시각적 및 텍스트 토큰의 상대적 위치 관계를 유지하며 시간 인식 RoPE를 포함시킵니다. 두 번째 접근법은 Frame-wise Block Causal Attention Mask를 사용하는 것으로, 이는 인과 추론 메커니즘을 유지하면서 비디오 프레임 내외부에서 시각적 토큰 간의 상호작용을 확대합니다.

- **Performance Highlights**: TC-LLaVA는 여러 비디오 이해 벤치마크에서 새로운 최첨단 성능을 달성하였으며, 단순히 비디오 관련 데이터셋에 대한 supervision fine-tuning(SFT)을 통해 이루어졌습니다. 이 모델은 비디오의 동적 사건을 효과적으로 요약하고 복잡한 움직임의 변화를 정확하게 캡처하여 모델의 정확도를 높였습니다.



### An Effective Deployment of Diffusion LM for Data Augmentation in Low-Resource Sentiment Classification (https://arxiv.org/abs/2409.03203)
- **What's New**: 본 논문에서는 감정 분류(Sentiment Classification)의 데이터 증강(Data Augmentation) 문제를 해결하기 위해 Diffusion 언어 모델을 활용한 새로운 방법, DiffusionCLS를 제안합니다. 이 방법은 도메인 지식(in-domain knowledge)을 포착하고 강력한 레이블 관련 토큰을 재구성하여 새로운 샘플을 생성합니다.

- **Technical Details**: DiffusionCLS는 레이블-주도(noise-resistant) 훈련 방식을 포함하여 감정 분류 모델의 성능을 향상시키고 다양한 저자원 시나리오에서도 효과적으로 작동합니다. 주요 구성 요소로는 레이블 인지 노이즈 일정(Label-Aware Noise Schedule), 레이블 인지 프롬프트(Label-Aware Prompting), 조건부 샘플 생성(Conditional Sample Generation) 등이 있습니다.

- **Performance Highlights**: DiffusionCLS는 감정 분류 과제에서 뛰어난 성능을 보였으며, 다양한 도메인 특정 및 다국어 데이터셋을 기반으로 한 포괄적인 실험을 통해 그 우수성을 입증했습니다.



### Bypassing DARCY Defense: Indistinguishable Universal Adversarial Triggers (https://arxiv.org/abs/2409.03183)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구에서는 IndisUAT라는 새로운 Universal Adversarial Trigger (UAT) 생성 방법을 제안합니다. 이 방법은 DARCY의 탐지를 효과적으로 우회할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: IndisUAT는 텍스트 분류 모델 보호를 위해 설계된 공격 방식으로, 공격자는 반복적으로 트리거 시퀀스를 업데이트하여 DARCY의 탐지 레이어에서 트랩 도어의 서명을 활성화하지 않도록 합니다. 이렇게 생성된 적대적 예시는 DARCY로 보호받는 모델에서 최대의 손실을 가져옵니다.

- **Performance Highlights**: IndisUAT는 DARCY의 진짜 정답 비율(true positive rate)을 최소 40.8%에서 90.6%까지 감소시키며, RNN과 CNN 모델의 정확도를 각각 최소 33.3% 및 51.6%까지 하락시킵니다. 또한 BERT의 적대적 방어 모델의 정확도는 최소 34.0% 감소시키며, GPT-2 언어 모델에서는 비인종적 맥락에서도 인종차별적인 출력을 생성하게 합니다.



### Continual Skill and Task Learning via Dialogu (https://arxiv.org/abs/2409.03166)
- **What's New**: 로봇이 자연어 대화를 통해 새로운 기술을 능동적으로 배울 수 있도록 하는 프레임워크를 제안합니다. 기존 연구들은 로봇의 능동적인 질문과 사용자와의 대화를 통해 새로운 기술을 배우는 방법을 탐구하지 않았습니다.

- **Technical Details**: 이 연구는 ACT-LoRA라는 새로운 비주얼-모터 제어 정책을 제안하며, 사용자와의 대화 및 언어-기술 접지 임베딩을 통해 로봇이 기술과 작업 관련 정보를 쿼리할 수 있게 합니다. 로봇은 5회의 시연만으로도 100%의 정확도로 새로운 기술을 배우며, RLBench 데이터셋에서 사전 훈련된 기술의 정확도는 74.75%입니다.

- **Performance Highlights**: 인간 참가자 8명을 대상으로 수행된 연구에서는 샌드위치 만들기 작업에서 75%의 성공률을 달성했습니다. 이는 로봇이 비전문 사용자와의 대화를 통해 새로운 기술을 배울 수 있음을 보여줍니다.



### Debate on Graph: a Flexible and Reliable Reasoning Framework for Large Language Models (https://arxiv.org/abs/2409.03155)
Comments:
          12 pages

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)과 지식 그래프를 통합하는 새로운 프레임워크인 DoG(Debating over Graphs)를 제안합니다. 이 프레임워크는 LLM의 상호작용 학습 능력을 활용하여 복잡한 질문을 단계적으로 간소화할 수 있도록 합니다.

- **Technical Details**: DoG는 서브그래프 집중 메커니즘을 사용하여 각 추론 단계 후 LLM이 답변을 시도할 수 있게 하며, 이를 통해 길고 복잡한 경로의 영향을 줄입니다. 또한 DoG는 멀티 역할 토론 팀을 활용하여 거짓 긍정 관계의 영향을 완화하고, 질문을 단순화하여 신뢰성을 높입니다.

- **Performance Highlights**: 실험 결과, DoG는 WebQuestions와 GrailQA 데이터셋에서 기존 최고 성능 방법인 ToG에 비해 각각 23.7% 및 9.1% 더 높은 정확도를 기록하며, 다섯 개의 공개 KGQA 데이터셋에서 모두 뛰어난 성능을 보였습니다.



### Addressing the Gaps in Early Dementia Detection: A Path Towards Enhanced Diagnostic Models through Machine Learning (https://arxiv.org/abs/2409.03147)
- **What's New**: 최근 급속한 세계적인 노령화 추세로 인해 알츠하이머병을 포함한 치매 사례가 증가하고 있으며, 조기 및 정확한 진단 방법의 필요성이 대두되고 있습니다.

- **Technical Details**: 기존의 진단 기술인 인지 테스트(cognitive tests), 신경영상(neuroimaging), 바이오마커 분석(biomarker analysis)은 민감도(sensitivity), 접근성(accessibility), 비용(cost)에서 상당한 한계를 겪고 있습니다. 본 연구는 기계 학습(machine learning, ML)을 활용하여 인지 평가, 신경영상, 유전자 정보를 포함한 복합 멀티모달 데이터셋을 분석하고 통합하는 방법을 탐구합니다. 다양한 ML 모델(지도 학습(supervised learning), 딥러닝(deep learning), 앙상블 학습(ensemble learning), 트랜스포머 모델(transformer models))의 정확성(accuracy), 해석 가능성(interpretability), 임상 통합 가능성(potential for clinical integration)을 평가하였습니다.

- **Performance Highlights**: ML 모델은 진단 정확성을 향상시키고 조기 개입을 가능하게 하는 데 상당한 가능성을 보이지만, 일반화 가능성(generalizability), 해석 가능성, 윤리적 배포(ethical deployment)와 같은 도전 과제가 남아있는 것으로 나타났습니다. 향후 방향은 치매 탐지에서 ML 모델의 임상적 유용성을 높이는 것을 목표로 하며, interdisciplinary collaboration와 윤리적으로 건전한框架(framework)를 강조합니다.



### Backdoor defense, learnability and obfuscation (https://arxiv.org/abs/2409.03077)
Comments:
          29 pages

- **What's New**: 본 논문은 공격자와 방어자 간의 게임을 통해 백도어(Backdoor)에 대한 방어 가능성(defendability)의 공식 개념을 소개합니다. 공격자는 특정 입력에서 다르게 작동하는 백도어 트리거(trigger)를 통해 함수를 수정하고, 방어자는 이를 평가할 때 감지하는 것을 목표로 합니다.

- **Technical Details**: 이 연구에서는 무한한 계산setting에서 VC 차원(VC dimension)에 따라 defendability가 결정된다고 보여주고, 계산 제약이 있는 setting에서는 효율적인 PAC learnability가 효율적인 defendability를 의미하지만 그 반대는 성립하지 않는다고 주장합니다. 또한, 다항식 크기의 회로(class of polynomial size circuits)가 효율적으로 defendable하지 않음을 보여주고, 결정 트리(polynomial size decision trees)와 같은 예를 통해 방어가 학습보다 수월하다는 점을 강조합니다.

- **Performance Highlights**: 분포가 균일할 경우, 다항식 크기의 결정 트리에서는 단일 결정 트리를 평가하는 시간 내에 방어를 수행할 수 있음을 보여줍니다. 또, 효율적인 defendability가 효율적인 PAC learnability와 구별될 수 있을지에 대한 논의도 포함되어 있습니다.



### MobileUNETR: A Lightweight End-To-End Hybrid Vision Transformer For Efficient Medical Image Segmentation (https://arxiv.org/abs/2409.03062)
Comments:
          Accepted at ECCV 2024 - BioImage Computing Workshop (Oral)

- **What's New**: 이번 논문에서는 피부 병변(segmentation) 세분화를 위해 고안된 새로운 경량 경량화된 하이브리드 딥러닝 모델인 MobileUNETR을 소개합니다. 이 모델은 CNN과 Transformer의 장점을 결합하여, 블록체인 크기와 계산 복잡성을 최소화하면서도 뛰어난 성능을 발휘합니다.

- **Technical Details**: MobileUNETR는 세 가지 주요 특징을 가지고 있습니다. 1) 경량 하이브리드 CNN-Transformer 인코더로 로컬(local) 및 글로벌(global) 컨텍스트(feature) 추출의 균형을돕습니다. 2) 저해상도와 고해상도에서의 저수준 및 글로벌 특징을 동시에 활용하는 하이브리드 디코더를 도입하여 정확한 마스크 생성을 지원합니다. 3) 300만 개의 파라미터와 1.3 GFLOP의 계산 복잡성으로, 큰 아키텍처와 복잡한 모델을 능가합니다.

- **Performance Highlights**: MobileUNETR는 ISIC 2016, ISIC 2017, ISIC 2018 및 PH2와 같은 네 가지 공개 피부 병변 세분화 데이터셋에서 실험을 통해 그 효과성을 입증했습니다. 모델 크기와 복잡성을 각각 10배 및 23배 감소시키면서 모든 데이터셋에서 성능이 크게 향상되었음을 보여줍니다.



### Better Verified Explanations with Applications to Incorrectness and Out-of-Distribution Detection (https://arxiv.org/abs/2409.03060)
- **What's New**: VeriX+는 machine learning 모델 출력에 대한 최적화된 검증 가능 설명을 제공하는 기존의 VeriX 시스템을 기반으로 하며, 설명의 크기와 생성 시간을 크게 개선했습니다.

- **Technical Details**: 이 논문에서는 바운드 전파(bounded propagation) 방식의 민감도 기술을 도입하여 크기를 개선하고, 이진 탐색 기반의 경로 탐색(binary search-based traversal) 및 신뢰도(rank) 정렬 기법을 이용하여 생성 시간을 줄이는 방법을 제시합니다.

- **Performance Highlights**: GTSRB 데이터셋에서는 설명 크기를 38% 줄였고, MNIST에서는 생성 시간을 90% 감소시켰습니다. 설명 크기는 잘못된 예측 탐지와 분포 외(out-of-distribution) 탐지에 유용한 지표로 활용될 수 있습니다.



### Can Your Generative Model Detect Out-of-Distribution Covariate Shift? (https://arxiv.org/abs/2409.03043)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 Out-of-Distribution (OOD) 감지 방법론 중에서 특성 변화(covariate shift) 탐지에 대한 새로운 접근 방식을 제안합니다. 현재까지의 연구는 주로 의미적 변화에 집중되어 왔으며, 저자들은 효과적인 OOD 탐지를 위해 생성 모델(generative models)의 활용과 이를 통해 도메인 특정(covariate)에 대한 변화를 탐지하는 방법을 간단히 설명하고 있습니다.

- **Technical Details**: CovariateFlow라는 새로운 방법론을 제안하여, 조건부 노멀리징 흐름(conditional Normalizing Flows)을 사용하여 고주파(high-frequency) 이미지 요소의 분포를 모델링합니다. 이 방법은 ID(in-distribution) 데이터와 OOD 데이터 간의 차이를 정확하게 감지하는 데 중점을 두고 있으며, Normalized Score Distance(NSD)라는 새로운 메트릭을 통합하여 로그 우도(log-likelihood) 기반 평가를 개선합니다.

- **Performance Highlights**: CIFAR10과 CIFAR10-C, ImageNet200과 ImageNet200-C 간의 실험 결과에서 CovariateFlow가 OOD 감지에서 매우 높은 정확도를 보였습니다. 이 연구는 OOD 탐지의 정확성을 높이는 데 기여할 뿐만 아니라, 다양한 환경에서 머신러닝 모델이 안정적으로 작동할 수 있도록 지원합니다.



### Large Language Model-Based Agents for Software Engineering: A Survey (https://arxiv.org/abs/2409.02977)
- **What's New**: 이 논문은 LLM(대형 언어 모델)을 기반으로 한 에이전트의 소프트웨어 엔지니어링(SE)에서의 설계와 응용에 대한 최초의 포괄적인 조사( survey )를 제공합니다.

- **Technical Details**: LLM 기반 에이전트는 ✨planning✨(계획), ✨memory✨(기억), ✨perception✨(지각), ✨action✨(행동)이라는 네 가지 핵심 구성 요소로 이루어져 있습니다. 에이전트는 여러 에이전트 간 협력을 통해 다양한 복잡한 문제를 해결할 수 있습니다.

- **Performance Highlights**: LLM 기반 에이전트는 소프트웨어 개발 및 유지 관리의 다양한 작업에서 우수한 성능을 보여주며, 이들의 협업은 현실 세계의 복잡한 SE 문제를 해결하는 데 더 큰 가능성을 제공합니다.



### Hallucination Detection in LLMs: Fast and Memory-Efficient Finetuned Models (https://arxiv.org/abs/2409.02976)
Comments:
          5 pages, 3 figures

- **What's New**: 본 논문에서는 큰 언어 모델(LLM) 집합을 빠르고 메모리 친화적으로 훈련할 수 있는 새로운 방법을 제시합니다. 이 방법은 한 개의 GPU만으로 훈련 및 추론이 가능하여 자원이 제한된 환경에서도 유용하게 활용될 수 있습니다.

- **Technical Details**: 제안된 방법은 저차 행렬(Low-Rank Matrices)을 활용하여 미리 훈련된 모델을 미세 조정( Fine-tuning)하며, 각 집합원의 개별적인 랭크-1 빠른 가중치 행렬(Fast Weight Matrix)을 사용합니다. 예측되는 불확실성(uncertainty)은 이 집합으로부터 얻어지며, 이 데이터는 헬로시네이션(hallucination)과 정확한 예측을 구분하기 위한 이진 분류기(binary classifier)에 입력됩니다.

- **Performance Highlights**: 이 방법은 사실성 헬로시네이션(faithfulness hallucinations)을 감지하는 데 97.8%의 정확도를 달성하였고, 사실적 헬로시네이션(factual hallucinations)에 대해서도 68%의 정확도를 보였습니다. 전체적인 예측 성능을 저하시키지 않으면서 헬로시네이션 탐지 능력을 향상시키는 실용적인 접근 방식을 제시합니다.



### Managing multiple agents by automatically adjusting incentives (https://arxiv.org/abs/2409.02960)
Comments:
          7 pages

- **What's New**: 본 연구에서는 자가 이익을 추구하는 AI 에이전트들이 사회 전체에 이익이 되는 목표를 향해 협력하도록 유도하는 방법을 탐구합니다. 이를 위해 관리 에이전트를 도입해 에이전트 상호작용을 중재하고 특정 행동에 인센티브를 부여하는 방식을 제안합니다.

- **Technical Details**: 이 연구에서 우리는 강화 학습 에이전트를 마르코프 결정 프로세스(Markov Decision Process, MDP)로 모델링하고, 에이전트들이 동시에 행동하는 마르코프 게임(Markov Game)을 고려합니다. 우리는 만약 관리자가 에이전트들의 보상의 합을 극대화하려고 한다고 가정하여, 보조 상태를 제공하고 특정 행동에 인센티브를 주는 방법을 사용합니다.

- **Performance Highlights**: 제안된 프레임워크는 공급망 관리 문제에서 (1) 원시 보상을 22.2% 증가시키고, (2) 에이전트의 보상을 23.8% 증가시키며, (3) 관리자의 보상을 20.1% 증가시키는 성과를 보여주었습니다.



### Multi-Modal Adapter for Vision-Language Models (https://arxiv.org/abs/2409.02958)
- **What's New**: 이 논문에서는 Multi-Modal Adapter라는 새로운 접근 방식을 제안하며, 이는 CLIP 모델의 비주얼(visual) 및 텍스트(text) 표현 간의 관계를 고려하여 모델의 결과를 개선할 수 있도록 합니다.

- **Technical Details**: Multi-Modal Adapter는 Multi-Head Attention 레이어를 추가하여 텍스트와 이미지의 특성을 통합하여 양쪽 모드의 적응을 수행합니다. 이는 각 작업 별로 효과적으로 비주얼 및 텍스트 정보를 활용하여 과적합(overfitting)을 방지하고 더 나은 일반화 능력을 제공합니다.

- **Performance Highlights**: Multi-Modal Adapter는 n-class-k-shot 설정에서 여러 데이터셋에서 경쟁력 있는 결과를 달성하였으며, 이전의 적응 방법들과 비교하여 보지 못한 클래스를 포함한 성능 향상을 보여줍니다.



### CortexCompile: Harnessing Cortical-Inspired Architectures for Enhanced Multi-Agent NLP Code Synthesis (https://arxiv.org/abs/2409.02938)
Comments:
          17 pages, 6 figures

- **What's New**: 이 논문은 CortexCompile이라는 모듈형 코딩 생성 시스템을 소개하며, 이는 인간의 뇌의 피질 영역에서 영감을 받아 설계되었습니다. CortexCompile은 전통적인 모놀리식 모델 대비 확장성, 효율성 및 적응성을 크게 향상시킵니다.

- **Technical Details**: CortexCompile은 다양한 태스크를 수명주기에 맞춰 관리하는 Task Orchestration Agent를 통해 다수의 전문화된 에이전트로 구성됩니다. 이 시스템은 계속해서 진화하는 프로그램 작성 작업에 맞춰 동적으로 조율됩니다. 연구 결과 CortexCompile은 개발 시간, 정확성, 사용자 만족도 면에서 GPT-4o보다 뛰어난 성능을 보여주었습니다.

- **Performance Highlights**: CortexCompile은 실시간 전략 게임과 1인칭 슈팅 게임과 같은 복잡한 작업에서 높아진 성능을 보이며, 시스템이 요구하는 대규모 리소스[resources]의 활용을 감소시켜 비용 효율적인 솔루션을 제공합니다.



### The Role of Transformer Models in Advancing Blockchain Technology: A Systematic Survey (https://arxiv.org/abs/2409.02139)
- **What's New**: 이 논문은 블록체인 기술과 트랜스포머 모델 간의 상관 관계를 체계적으로 조사하고 200개 이상의 관련 논문을 리뷰하여 블록체인 애플리케이션에서 트랜스포머의 가능성과 한계를 제시합니다.

- **Technical Details**: 트랜스포머 모델은 자연어 처리(NLP) 분야에서 혁신적 발전을 이루었으며, 자가 주의(self-attention) 기법을 기반으로 복잡한 순차 데이터의 처리를 가능하게 합니다. 논문은 트랜스포머의 구조, 기능 및 블록체인 데이터 처리에 효과적인 이유를 설명합니다. 또한 무작위 언어 모델링(Masked Language Modeling) 및 다음 문장 예측(Next Sentence Prediction) 같은 사전 훈련(task)도 간단히 언급됩니다.

- **Performance Highlights**: 트랜스포머 모델은 블록체인에서의 이상 탐지(anomaly detection), 스마트 계약 보안 분석(smart contract security analysis), 암호화폐 예측 및 동향 분석(cryptocurrency prediction and trend analysis) 등 다양한 분야에서 응용 가능성을 나타내고 있습니다. 이 논문은 블록체인 기술과 머신러닝의 통합 발전을 위한 새로운 관점을 제공합니다.



### Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation (https://arxiv.org/abs/2409.01586)
- **What's New**: 이 논문에서는 모델 가중치에 대한 해로운 섭동(harmful perturbation)이 해로운 미세 조정(harmful fine-tuning) 문제의 근본 원인이라고 명시하였습니다. 이러한 문제를 완화하기 위해 'Booster'라는 정렬 단계 솔루션을 제안합니다.

- **Technical Details**: Booster는 원래의 정렬 손실(alignment loss)에 손실 정규화기(loss regularizer)를 추가하여 최적화 과정에서 해로운 섭동의 부정적인 영향을 줄이도록 설계되었습니다. 이 정규화기는 해로운 데이터셋을 사용하여 모델의 손실 감소를 제어합니다. Booster는 반복적 경량법(iterative gradient method)으로 문제를 해결합니다.

- **Performance Highlights**: Booster는 기존 솔루션인 Vaccine과 RepNoise에 비해 각각 17.26%와 20.08%의 평균 해로운 점수를 감소시키면서도 다운스트림 작업의 성능을 유지하는 효과를 보여주었습니다.



### AI-Driven Intrusion Detection Systems (IDS) on the ROAD Dataset: A Comparative Analysis for Automotive Controller Area Network (CAN) (https://arxiv.org/abs/2408.17235)
- **What's New**: 이 논문에서는 Intrusion Detection System (IDS)의 효과를 평가하기 위한 현실적이고 포괄적인 ROAD 데이터셋을 제안합니다. 이 데이터셋은 다양한 은밀한 공격을 포함하고 있으며, 기존 문헌에서 사용된 데이터셋들과의 성능 차이를 분석합니다.

- **Technical Details**: 논문은 Controller Area Network (CAN) 프로토콜의 취약성과 Intrusion Detection System (IDS) 기술을 다룹니다. 특히, ROAD 데이터셋을 통한 딥러닝 및 전통적 기계학습 모델(예: LightGBM, Random Forest 등)의 적용 사례를 제공합니다.

- **Performance Highlights**: 최신 딥러닝 모델로 Transformer-based Attention Network (TAN), Deep Convolutional Neural Network (DCNN) 및 Long Short-Term Memory (LSTM) 모델을 평가하였으며, ROAD 데이터셋에서 IDS의 성능 비교를 통해 현실적 공격 인식의 중요성을 강조합니다.




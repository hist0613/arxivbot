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




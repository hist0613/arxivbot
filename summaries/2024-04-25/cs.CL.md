### Universal Adversarial Triggers Are Not Universa (https://arxiv.org/abs/2404.16020)
- **What's New**: 이 연구에서는 '최적화된 공격 트리거(Adversarial Triggers)'가 모든 모델에서 일관되게 작동하지 않음을 밝혔습니다. 연구팀은 13개의 공개 모델을 사용하여 공격 트리거의 전달 가능성을 조사한 결과, 일부 모델에서는 트리거가 효과가 있는 반면 다른 모델에서는 그렇지 않은 것을 발견했습니다. 특히 '선호 최적화(Preference Optimization, 이하 APO)'로 정렬된 모델들은 공격 트리거에 대한 강인성을 보였으나 '미세 조정(Fine-Tuning, 이하 AFT)'으로 정렬된 모델들은 이러한 트리거에 취약한 것으로 나타났습니다.

- **Technical Details**: 연구에서는 '탐욕스러운 좌표 경사(Greedy Coordinate Gradient, 이하 GCG)'를 사용하여 LLM들에 대한 공격을 시도했습니다. GCG는 백색 공격(white-box attack) 기법으로, 해로운 명령어(예: 'How do I build a bomb')에 대해 긍정적인 반응을 최소화하는 방향으로 토큰 시퀀스를 반복적으로 업데이트하여 최적화합니다. 또한 연구팀은 다양한 안전 벤치마크를 통해 AFT 모델로부터의 트리거의 일반화 가능성을 평가했습니다.

- **Performance Highlights**: APO 모델은 직접 최적화된 트리거조차도 다른 모델로 전달하는 데 실패한 반면, AFT 모델은 덜 강인하고, 적은 단계에서 더 빠르게 최적화되며, 유해한 반응을 더 자주 유발하는 것으로 나타났습니다. 또한 AFT 모델에서 최적화된 대부분의 트리거는 다섯 가지 다양한 도메인에서 새로운 안전하지 않은 명령어에 대해 놀라울 만큼 잘 일반화되었습니다. 이는 AFT 모델들이 공격 트리거에 대한 취약성을 증명하는 결과입니다.



### The PRISM Alignment Project: What Participatory, Representative and  Individualised Human Feedback Reveals About the Subjective and Multicultural  Alignment of Large Language Models (https://arxiv.org/abs/2404.16019)
- **What's New**: 새로운 데이터셋 PRISM을 소개합니다. 이 데이터셋은 75개국에서 온 1,500명의 다양한 참가자들의 사회 인구학적 데이터와 선호도를 그들이 Large Language Models(LLMs)와 진행한 8,011개의 대화 선호도와 미세한 피드백과 연결합니다. PRISM은 주제가 가치 있고 논쟁적인 토론에 집중하면서, 사용자들의 주관적이고 다문화적인 관점을 반영합니다.

- **Technical Details**: PRISM은 LLMs와의 실시간 대화에서 수집한 넓은 지리적, 인구학적 참여를 기반으로 하는 데이터를 포함합니다. 가장 중요한 것은 미국(UK)과 미국(US)에서 인구 통계적으로 대표적인 샘플을 포함하여 집단 복지에 대한 이해를 높이고, 각 평가가 상세한 참가자 프로필에 연결된 개인화된 피드백입니다. 이를 통해 피드백의 개인화와 샘플 아티팩트의 귀속을 탐색할 수 있습니다.

- **Performance Highlights**: PRISM 데이터셋의 활용성을 보여주는 세 가지 경우 연구를 통해 대화의 다양성, 선호의 다양성, 그리고 복지 결과의 중요성을 시연합니다. 또한, 어떤 인간이 정렬 기준을 설정하는지가 중요하다는 것을 보여줍니다. 이 데이터셋은 인공지능 개발에 더 넓은 참여를 지지하고 기술 설계에 있어서 보다 포괄적인 접근을 옹호합니다.



### Sequence can Secretly Tell You What to Discard (https://arxiv.org/abs/2404.15949)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, 이하 LLMs)의 KV 캐시 최적화를 위한 새로운 접근 방식인 CORM을 소개합니다. CORM은 핵심 KV 페어(Key-Value pairs)만을 동적으로 유지함으로써 메모리 사용을 대폭 감소시킵니다.

- **Technical Details**: CORM(Cache Optimization with Recent Message)은 최근 질의 벡터의 유사성을 활용하여, 불필요한 KV 페어를 제거하고 메모리 효율을 높이는 새로운 정책입니다. 이 정책은 이전 질의의 중요한 정보만을 유지하면서 최적화를 진행합니다.

- **Performance Highlights**: LLaMA2-7B-Chat 모델에 대한 실험을 통해, CORM은 기존 방식들 대비 최대 70%까지 KV 캐시 사용량을 감소시킬 수 있음을 확인했습니다. 이는 다양한 NLP 작업(Task)에서 성능 저하 없이 메모리 효율성을 높인 것이며, LongBench의 6가지 작업에서 그 효과가 검증되었습니다.



### Generalization Measures for Zero-Shot Cross-Lingual Transfer (https://arxiv.org/abs/2404.15928)
- **What's New**: 이 논문은 언어 모델이 여러 언어에 걸쳐 제로샷(zero-shot) 환경에서 일반화(generalization) 능력을 측정하는 새로운 방법론을 탐구합니다. 특히, 언어 모델의 교차 언어 일반화 능력을 측정하기 위한 방법들을 제시하며, 최신 샤프니스(sharpness) 계산 방법을 확장하여 매개변수 최적화가 더 안정적으로 수렴할 수 있도록 하는 예측 알고리즘을 제안합니다.

- **Technical Details**: 이 연구는 Frobenius 거리, 예측과 실제 라벨 간의 마진, 손실 최소값의 샤프니스 등 일반화 성능과 잘 연관된 것으로 밝혀진 주요 측정 지표를 선택하여 사용합니다. 새로운 샤프니스 계산 방법론인 Sharpness-Aware Minimization (SAM)과 Fisher Information Matrix (FIM) 규제를 비교 분석하며, 이들을 통해 언어 모델이 다양한 언어 환경에서의 일반화 능력을 평가합니다. 추가적으로, 다양한 언어의 공유 서브워드(subword) 어휘에 스토케스틱(stochastic) 변동을 유발하는 Multi-view Subword Regularization (MVR) 방법을 사용하여 교차 언어 전이를 용이하게 합니다.

- **Performance Highlights**: 이 방법들은 임의의 언어에서 학습된 모델이 다른 언어로의 지식 전달과 일반화에서 우수한 성능을 나타내는 것으로 관찰됩니다. 특히, SAM은 일반화 능력 향상에 효과적인 전략으로 평가되며, 시뮬레이션을 통해 검증된 바에 의하면, 일관되게 낮은 손실값 주변의 최적화 매개변수를 찾는 데 성공하여 일반화 능력을 향상시킬 수 있는 것으로 나타났습니다.



### Assessing The Potential Of Mid-Sized Language Models For Clinical QA (https://arxiv.org/abs/2404.15894)
Comments: 25 pages, 8 figures

- **What's New**: 최근 연구에서는 대용량 언어 모델들(GPT-4, Med-PaLM 등)이 임상 과제에서 인상적인 성능을 보였지만, 이들은 많은 컴퓨터 자원을 요구하고, 오픈 소스(open-source)가 아니며, 기기에 바로 배포할 수 없다는 단점이 있습니다. 이에 반해 중형 모델들(BioGPT-large, BioMedLM, LLaMA 2, Mistral 7B)은 이러한 단점들을 피할 수 있지만, 임상 과제에서의 성능은 충분히 연구되지 않았습니다. 이 연구는 임상 사용(clinical use)의 잠재력을 평가하고 연구자들이 사용할 모델을 결정하는 데 도움을 주기 위해 이러한 중형 모델들의 성능을 두 가지 임상 질의응답(QA) 과제, MedQA와 소비자 질의응답(consumer query answering)에서 비교 분석했습니다.

- **Performance Highlights**: Mistral 7B 모델이 모든 벤치마크에서 최고의 성능을 보였으며, 생물의학 분야(biomedical domain)에 특화된 모델들을 능가했습니다. Mistral 7B는 MedQA에서 63.0%의 점수를 얻으며 원래 Med-PaLM에 근접한 성능을 보였고, 소비자 건강 질의에 대한 신뢰할 수 있는 응답을 자주 제공할 수 있었습니다. 그러나 여전히 개선의 여지는 남아 있습니다. 이 연구는 임상 과제에서 오픈 소스 중형 모델들의 첫 번째 직접 비교 평가를 제공합니다.



### Effective Unsupervised Constrained Text Generation based on Perturbed  Masking (https://arxiv.org/abs/2404.15877)
- **What's New**: 이 논문에서는 지도 학습 데이터에 의존하지 않고 제약이 있는 텍스트 생성을 위해 PMCTG(Perturbed Masking for Constrained Text Generation)를 제안합니다. PMCTG는 베르트(BERT)와 같은 사전 훈련된 모델을 사용하여 가장 효율적으로 수정해야 할 위치를 찾고, 다양한 점수 함수를 사용하여 최선의 편집 동작을 결정합니다. 이 방법은 키워드로부터 문장 생성 및 문장 재구성과 같은 두 가지 대표적인 작업에서 새로운 최고의 성능을 달성하였습니다.

- **Technical Details**: PMCTG는 perturbed masking 기술을 확장하여 연속된 텍스트에서 가장 일치하지 않는 토큰을 효과적으로 검색합니다. 이 기술은 여러 측면에서 점수 함수를 도입하여 편집 작업을 선택함으로써 검색 곤란을 줄입니다. 사용된 점수 함수는 생성된 텍스트의 질을 다른 작업에 맞추어 반영합니다. PMCTG는 기존의 빔 탐색(beam search)과 국소 편집(local edit) 방식과는 달리, 불필요한 검색 단계를 줄이면서 제약을 만족시키는 더 나은 텍스트를 생성합니다.

- **Performance Highlights**: 실험 결과 PMCTG는 키워드-to-문장 생성 및 문장 재구성 작업에서 최상의 성능을 보여주었습니다. PMCTG는 기존의 슈퍼바이즈드(supervised) 모델과 비교하여 상당한 성능 향상을 관찰하면서 학습 데이터의 크기에 구애받지 않고 다양한 텍스트 생성 작업에 적용될 수 있는 가능성을 보여 줍니다.



### Detecting Conceptual Abstraction in LLMs (https://arxiv.org/abs/2404.15848)
Comments: Paper accepted at the LREC-COLING 2024 Conference (Paper ID: 1968) this https URL

- **What's New**: 새롭게 제시된 연구에서는 대규모 언어 모델(LLM) 내에서 명사의 추상화를 탐지하는 새로운 접근 방법을 소개합니다. 특히, BERT 모델의 주목 매트릭스(attention matrices)를 분석하여 하이퍼니미(hypernymy)를 탐지하는 실험을 수행했습니다. 이 연구는 언어 모델이 인간과 유사한 추상화 메커니즘을 사용하는지에 대한 설명 가능성(explainability)을 제공하는 첫 걸음입니다.

- **Technical Details**: 연구팀은 심리학적으로 동기 부여된 명사 쌍 데이터셋을 사용하여, 하이퍼님(hypernym)과 하이포님(hyponym) 관계를 표현하는 문장 패턴을 생성하고 BERT의 주목 패턴을 분석합니다. 이를 통해 표적 명사와 특징 명사가 관계를 가진 데이터를 추출하고, 언어 모델이 어떻게 이러한 관계를 내부적으로 표현하는지를 조사했습니다. 분석은 두 가지 대조군(Counterfactuals)과 비교하여 진행되었으며, 이들은 의미론적 유사성(semantic similarity) 또는 추상화 수준(abstraction level)에 따라 선택되었습니다.

- **Performance Highlights**: 연구 결과, BERT는 하이퍼님 관계를 효과적으로 대표하는 주목 패턴을 가지고 있음을 보여줍니다. 이는 의미론적 유사성만이 아닌 추상화 메커니즘을 통해 언어 추상화를 추론할 수 있다는 명확한 증거를 제공합니다. 이러한 발견은 LLM에서 개념적 추상화 및 설명 가능한 AI(Explainable AI)로의 이해를 더욱 심화시킬 수 있는 가능성을 열어줍니다.



### From Complex to Simple: Enhancing Multi-Constraint Complex Instruction  Following Ability of Large Language Models (https://arxiv.org/abs/2404.15846)
- **What's New**: 이 연구에서는 복잡한 지시 사항을 따르는 능력을 향상시키기 위해 대규모 언어 모델(Large Language Models, LLMs)의 효과적인 훈련 데이터에 대해 탐구하였습니다. 이는 복잡한 지시 사항 및 제약이 있는 명령을 더 잘이해하고 수행할 수 있도록 LLM의 능력을 증진시키는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 복잡한 지시 사항을 포함하는 훈련 데이터가 LLM의 복잡한 지시 사항 이해와 수행 능력을 향상시키는데 얼마나 효과적인지를 조사했습니다. 연구를 통해 여러 제약을 포함하는 지시 사항을 학습할 때, 이러한 유형의 지시에 대한 이해도가 향상됨을 보여줍니다. 또한, 이러한 훈련 데이터를 획득하고 활용하는 방법에 대해 새로운 방법을 제안합니다.

- **Performance Highlights**: 훈련 데이터가 특정한 복잡한 지시를 포함할 때, LLMs는 낮은 복잡성 수준의 지시 사항을 훨씬 더 잘 이해하고 수행하는 향상된 능력을 보였습니다. 추가적으로, 이러한 향상은 도메인 외적(out-of-domain) 제약 조합에도 일반화될 수 있음을 증명했습니다. 연구팀은 네 가지 조건 하에서 전반적인 성능, 훈련 효율성, 그리고 일반화 능력의 증진을 입증하기 위해 광범위한 실험을 수행했습니다.



### Exploring LLM Prompting Strategies for Joint Essay Scoring and Feedback  Generation (https://arxiv.org/abs/2404.15845)
Comments: Accepted to BEA Workshop 2024

- **What's New**: 이 논문은 학생의 에세이 작문 기술을 향상시킬 수 있는 자동화된 피드백 생성에 초점을 맞추고 있습니다. Large Language Models (LLMs, 대규모 언어 모델)을 활용해 에세이 품질을 평가(Automated Essay Scoring, AES)하고, 학생이 자신의 에세이를 개선할 수 있도록 구체적인 피드백을 제공하는 전략을 탐구하였습니다. 특히, Chain-of-Thought 프롬프팅 기법에 영감을 받아 AES가 에세이 피드백 생성의 성능에 어떤 영향을 미치는지, 그리고 그 반대의 경우는 어떤지를 조사했습니다.

- **Technical Details**: 연구팀은 여러가지 프롬프팅 전략을 사용하여 LLM을 기반으로 하는 'zero-shot'(제로샷)과 'few-shot'(퓨샷) 설정에서의 에세이 피드백 생성을 실험했습니다. 이러한 접근 방식은 학생들에게 자신의 에세이에 대한 유용한 피드백을 제공하도록 설계되었습니다. 논문에서는 에세이 점수를 예측하고 그 점수를 설명함으로써 에세이 피드백의 도움을 높이는 전략을 제안했습니다.

- **Performance Highlights**: 실험 결과에서는, 예측된 에세이 점수를 설명하는 방식으로 피드백을 생성하는 것이 평가 데이터셋(ASAP 데이터셋)에서의 점수 성능을 향상시켰다는 것을 발견하였습니다. 또한, 자동 및 수동 평가 모두에서 생성된 에세이 피드백이 학생들의 작문 기술 향상에 도움이 될 것으로 판단되었지만, 에세이를 점수화하는 것이 생성된 피드백에 미치는 영향은 궁극적으로 낮았습니다.



### One Subgraph for All: Efficient Reasoning on Opening Subgraphs for  Inductive Knowledge Graph Completion (https://arxiv.org/abs/2404.15807)
- **What's New**: 이 논문은 지식 그래프 완성(Knowledge Graph Completion, KGC)에서 통상적으로 사용되는 전이적(transductive) 접근 방식의 한계를 극복하고자 새로운 유도적(inductive) KGC 방법을 제안합니다. 전이적 KGC 방법은 훈련 중에 관찰된 모든 실체(entity)를 기반으로 작동하지만, 새로운 실체가 지속적으로 등장하는 현실 세계에서는 효과적이지 않습니다. 이에 대응하여, 연구자들은 개방형 하위 그래프(opening subgraph)를 활용하고, 전역 및 지역 앵커(global and local anchors)를 통해 실체 독립적 특성(entity-independent features)을 학습하는 새로운 유도적 KGC 방법을 개발했습니다.

- **Technical Details**: 제안된 글로벌-로컬 앵커 표현(Global-Locally Anchor Representation, GLAR) 학습 방법은 기존의 문제점들을 개선하기 위해 개발되었습니다. GLAR은 개방형 하위 그래프를 추출하여 모든 후보에 대해 추론을 수행하며, 이는 후보들 각각에 대해 반복적으로 하위 그래프를 추출하는 기존 방법들보다 효율적입니다. 또한, 전역 및 지역 앵커를 통해 실체 독립적인 구조 정보를 학습하고, 이를 바탕으로 새로운 실체들의 특성을 효과적으로 포착합니다. 최종적으로, 글로벌-로컬 그래프 추론 모델(global-local graph reasoning model)을 적용하여 모든 후보를 평가합니다.

- **Performance Highlights**: GLAR 모델은 다양한 표준 유도적 KGC 데이터셋에서 수행된 광범위한 실험을 통해 기존의 최신(state-of-the-art) 방법들을 능가하는 성능을 보여주었습니다. 이는 GLAR이 실체 독립적 특성 학습과 효율적인 추론을 통해 유도적 KGC의 문제를 효과적으로 해결할 수 있음을 시사합니다.



### A Comprehensive Survey on Evaluating Large Language Model Applications  in the Medical Industry (https://arxiv.org/abs/2404.15777)
Comments: 18 pages

- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)이 의료 분야에 어떻게 적용될 수 있는지에 대한 광범위한 조사를 제공합니다. 특히 이 논문은 임상 환경, 의료 텍스트 데이터 처리, 연구, 교육, 공중 보건 인식 등 다양한 의료 관련 응용 프로그램에서 LLM의 역할을 탐구합니다. 또한 이 모델들이 어떻게 평가되고 있는지에 대한 방법론을 심층적으로 논의하며, 효과, 정확성, 윤리적 일치성을 평가하기 위해 사용된 벤치마크 및 메트릭스(metrics)를 소개합니다.

- **Technical Details**: 이 논문은 Transformer, GPT, BERT와 같은 기술의 발전을 바탕으로 LLMs의 다양한 응용 프로그램과 이러한 모델들이 의료 분야에서 어떤 잠재력을 지니고 있는지를 설명합니다. GPT-3와 GPT-4 같은 모델은 텍스트 생성에서 거의 인간과 구별할 수 없을 정도의 성능을 보여줌으로써, 복잡한 언어 이해 및 생성 작업에서 뛰어난 성능을 시연했습니다. 또한, 이 논문은 OpenAI의 최신 모델인 GPT-4와 Google의 정보 검색 최적화 모델인 Gemini와 같은 최신 혁신을 탐구합니다. 이러한 모델들은 주목할 만한 성능 향상을 제공하면서 연구 커뮤니티와 산업에 유연성과 접근성을 향상시키고 있습니다.

- **Performance Highlights**: LLMs는 임상 응용, 의료 텍스트 데이터 처리 등과 같은 작업에서 높은 성능을 보였으며, 정보 검색, 데이터 분석, 의학 과학 작문 및 교육 콘텐츠 생성에서도 그 효능을 입증했습니다. 특히, 이 논문은 메디컬 환경에서의 LLM들의 성능 평가에 사용된 여러 벤치마크와 메트릭스(metrics)를 살펴보며, 이러한 모델들이 어떻게 윤리적 기준을 유지하며 효과적으로 배치될 수 있는지에 대한 인사이트를 제공합니다.



### Let's Think Dot by Dot: Hidden Computation in Transformer Language  Models (https://arxiv.org/abs/2404.15758)
Comments: 17 pages, 10 figures

- **What's New**: 본 연구에서는 언어 모델이 사고 과정의 체인(chain-of-thought)을 통해 성능을 향상시킬 수 있지만, 이러한 성능 향상이 과연 인간과 유사한 작업 분해에 기인하는 것인지, 아니면 단순히 추가 토큰을 통한 더 많은 계산량 때문인지를 밝히려고 하였습니다. 특히, 의미 없는 필러 토큰(filler tokens), 예를 들어 '......'의 사용이 언어 모델 성능에 미치는 영향을 조사하였습니다.

- **Technical Details**: 이 논문은 언어 모델이 의미 없는 반복된 필러 토큰('......')을 사용하여 복잡한 알고리즘 작업(algorithmic tasks)을 해결할 수 있는지를 연구합니다. 결과적으로 필러 토큰은 특정, 밀집된 감독(dense supervision) 없이는 학습이 어렵다는 것을 발견하였고, 이는 첫 번째 순서의 공식(first-order formula)에서 정량자 깊이(quantifier depth)에 관련된 문제 클래스에 대한 이론적 특성화를 제공합니다. 연구된 바에 따르면, 필러 토큰이 문제 해결을 위한 연산을 지원하는 데 필요한 추가의 토큰 공간을 제공하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과에 따르면 대규모 언어 모델(Large Language Models, LLMs)이 일반적인 Q&A와 수학 벤치마크에서 필러 토큰을 사용하여 동등한 성능을 보이며, 복잡성 증가 및 입력의 길이에 따라 필러 토큰 성능이 향상될 가능성을 보였습니다. 그러나 현재 LLM은 필러 토큰을 이용해서 𝖳𝖢0 ({𝖳𝖢0	extsuperscript{0}) 범주 밖의 문제를 해결할 수는 없으나, 이론적 결과와 실험 결과를 통해 필러 토큰이 𝖳𝖢0 범주 내에서 변환기(transformers)의 표현력을 확장할 수 있음을 시사합니다.



### No Train but Gain: Language Arithmetic for training-free Language  Adapters enhancemen (https://arxiv.org/abs/2404.15737)
- **What's New**: 이 연구에서는 다국어 사전 학습된 언어 모델(Multilingual Pre-trained Language Models, MLLMs)의 한계를 극복하기 위해 '언어 산술(Language Arithmetic, LA)'이라는 새로운 방법을 도입했습니다. 이는 언어 어댑터(language adapters)에 학습이 필요 없는 후처리(post-processing) 단계를 적용하여 관련 언어 지식을 향상시키는 훈련이 필요 없는 방법입니다.

- **Technical Details**: 언어 산술은 과제 산술(task arithmetic) 개념을 기반으로 하여 다국어 설정으로 전환된 것입니다. 이 방법은 이미 훈련된 언어 어댑터에 적용되며, 언어 간의 추가 학습을 통해 언어 벡터(language vectors)를 조합합니다. 특정 언어 코드를 사용하여 언어 간의 연산을 수행하며, 이는 MAD-X 프레임워크를 사용하여 효과를 검증하였습니다.

- **Performance Highlights**: 언어 산술을 통한 후처리 접근 방식은 기존의 베이스라인 성능을 일관되게 향상시켰으며, 특히 제로샷(zero-shot) 및 저자원(low-resource) 시나리오에서 유의미한 성과를 보였습니다. 이 방법은 다양한 상황에서 언어 어댑터의 성능을 강화할 수 있는 유연하고 효과적인 솔루션을 제공합니다.



### Annotator-Centric Active Learning for Subjective NLP Tasks (https://arxiv.org/abs/2404.15720)
- **What's New**: 이 연구에서는 주관적 NLP(자연어 처리) 작업을 위해 사람의 판단 변동성을 정확하게 포착하기 위해 주석(annotation) 과정에서 다양한 관점을 포함하는 것의 중요성을 강조합니다. 이를 위해 Annotator-Centric Active Learning (ACAL)이라는 새로운 접근방식을 도입하고 있습니다. ACAL은 데이터 샘플링 후 주석자(annotator) 선택 전략을 포함하여, 인간 판단의 전체 다양성을 효율적으로 근사화하고 주석자 중심 메트릭을 사용하여 모델 성능을 평가하려는 두 가지 목표를 갖고 있습니다.

- **Technical Details**: ACAL은 Active Learning (AL)의 고비용 문제를 해결하기 위해 가장 정보가 많은 샘플에 대해 전략적 주석을 달도록 지원합니다. 또한, 주석자 선택은 데이터의 다양성을 더 잘 반영하기 위해 사용되며, 기존의 전통적인 평가 메트릭과 새로운 인간 중심 평가 메트릭을 사용하여 성능을 평가합니다. 이 연구는 일곱 가지 주관적 NLP 작업에서 다양한 주석자 선택 전략을 실험하며 평가합니다.

- **Performance Highlights**: ACAL은 데이터 효율성을 개선하고 주석자 중심의 성능 평가에서 뛰어난 결과를 보여줍니다. 그러나, 이 성공은 충분히 크고 다양한 주석자 풀(annotator pool)에서 선택할 수 있는 옵션이 있을 때에만 의존적입니다. ACAL은 소수 의견을 중요하게 여기는 평가에서 특히 우수한 성능을 보여 주석자의 다양성을 포괄적으로 반영할 수 있는 접근 방식을 제시합니다.



### Nyonic Technical Repor (https://arxiv.org/abs/2404.15702)
- **What's New**: 이 보고서는 사용자 맞춤형 대규모 언어 모델을 위해 설계된 최신 언어 모델의 개발과 주요 성과를 상세히 설명합니다. 새롭게 도입된 기술로는 유연한 훈련 데이터 조정과 커리큘럼 학습을 지원하는 새로운 온라인 데이터 스케줄러, 회전 위치 임베딩(Rotary Positional Embeddings), QK-레이어노말(QK-LayerNorm) 및 특별히 제작된 다국어 토크나이저가 있습니다. 또한, 우리의 견고한 훈련 프레임워크는 고급 모니터링과 빠른 복구 기능을 통합하여 최적의 효율성을 보장합니다. Wonton 7B 모델은 다양한 다국어 및 영어 벤치마크에서 경쟁력 있는 성능을 보여주었습니다.

- **Technical Details**: 이 모델의 아키텍처는 트랜스포머(Transformer)를 기반으로 하며, 다양한 최신 기법이 적용되었습니다. 특히, 각 층에서 회전 위치 임베딩(Rotary Positional Embeddings, RoPE)을 적용하고, QK-레이어노말(QK-LayerNorm)을 사용하여 열의 제품 주의 과정에서 쿼리와 키에 적용함으로써 주의 로짓의 과도한 크기를 방지합니다. 또한, 새로운 온라인 데이터 스케줄러는 오프라인 데이터 토큰 인덱스 변환의 필요성을 없애고, 실시간 피드백을 가능하게 하여 모델 훈련의 유연성을 높였습니다. 이 다국어 토크나이저는 고객들에게 널리 사용되는 언어를 포함하여 최종 어휘 크기를 139,000으로 결정했습니다.

- **Performance Highlights**: Wonton 7B는 향상된 안정성 및 성능을 위해 설계된 다양한 기술을 통합하는 것보다 훨씬 뛰어난 성능을 제공합니다. 실시간 데이터 피드백을 통해 더 나은 훈련 결과를 달성하고, 다양한 언어에 걸쳐 일관된 토크나이저 효율성을 강조하는 평가에서도 우수한 압축 효율성을 보여줍니다. 이 모델은 지속적으로 훈련하고 개선될 예정이며, 다양한 실제 응용 분야에서의 효과와 적응성을 높이기 위한 미래 개발에 초점을 맞출 것입니다.



### Neural Proto-Language Reconstruction (https://arxiv.org/abs/2404.15690)
- **What's New**: 이 연구에서는 언어 복원에 RNN과 Transformers 같은 계산 모델을 활용함으로써 언어학자들의 고대 언어 복원 과정을 자동화하는 데 중점을 둡니다. 특히, VAE(변형적 오토인코더) 구조를 추가하는 방법, 데이터 증강을 사용하여 누락된 반사(reflexes)를 복구하는 방법, 그리고 NMT(Neural Machine Translation) 모델을 언어 복원 작업에 적용하는 새로운 시도를 소개합니다.

- **Technical Details**: 이 연구는 세 가지 기술적 방법을 탐구합니다. 첫 번째로, WikiHan 데이터셋의 누락된 항목을 채우기 위해 데이터 증강 기술을 사용합니다. 두 번째로, VAE 구조를 Transformer 모델에 추가하여 더 의미 있는 잠재 공간을 확보하고, 이를 통해 proto-to-language 예측의 성능을 향상시킵니다. 마지막으로, NMT 모델을 수정하여 언어 복원 작업에 적합하게 만들어 사용합니다. CNN 모델은 이미지 인페인팅 작업과 유사하게, 누락된 데이터를 예측하는데 사용되며, 이는 학습 과정을 안정화하고 모델의 일반화 성능을 높이는 데 도움이 됩니다.

- **Performance Highlights**: VAE 구조가 추가된 Transformer 모델은 WikiHan 데이터셋에서 뛰어난 성능을 보였으며, 데이터 증강 단계는 훈련 과정을 안정화시킵니다. 언어 복원 작업에서 NMT 모델의 적용 가능성도 확인되었습니다. 이러한 향상된 접근 방식은 기존 방법들에 비해 더 정확하고 효율적인 언어 복원을 가능하게 하며, 이는 언어학 및 NLP 공동체에 중요한 기여를 합니다.



### Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs (https://arxiv.org/abs/2404.15676)
- **What's New**: 이 논문은 Chain-of-Thought (CoT) 방법을 기반으로 한 다양한 Chain-of-X (CoX) 방법들에 대한 포괄적인 조사를 제공합니다. CoT는 복잡한 문제를 연속적인 사고 과정을 통해 해결하는 접근 방식이며, 최근 다양한 형태의 CoX가 개발되어 다양한 도메인과 작업에서 적용되었습니다. 이 연구는 CoX의 여러 형태를 체계적으로 정리하고, 각각의 적용 사례 및 잠재적 활용 방안에 대해 논의합니다.

- **Technical Details**: CoX 기법들은 기존 CoT 접근법을 확장하여 다양한 '노드'(node) 형태로 구성됩니다. 이 노드들은 계산적 기여를 하거나 문제 해결 과정을 반복적으로 정제하는 역할을 합니다. 예를 들어, Chain-of-Feedback, Chain-of-Instructions, Chain-of-Histories 등이 있으며, 각기 교육, 멀티모달 상호작용, 환각 감소, 에이전트 계획 등의 분야에서 활용됩니다. 논문에서는 이러한 CoX 접근법들을 '성분'(component) 유형과 적용 작업에 따라 분류합니다.

- **Performance Highlights**: CoX 방법들은 LLMs의 성능을 탁월하게 향상시키는 것으로 나타났습니다. 이러한 방법들은 복잡한 문제 해결 능력을 강화하고, 더 투명하고 해석가능한 추론 과정을 제공하여 모델의 평가와 수정을 용이하게 합니다. 각 CoX 방법은 특정 작업에 맞춰 최적화되어 다양한 영역에서 유용하게 활용될 수 있는 잠재력을 보여줍니다.



### The Promise and Challenges of Using LLMs to Accelerate the Screening  Process of Systematic Reviews (https://arxiv.org/abs/2404.15667)
Comments: Accepted to the International Conference on Evaluation and Assessment in Software Engineering (EASE), 2024 edition

- **What's New**: 이 연구는 소프트웨어 공학(systematic review, SR) 분야에서의 체계적 리뷰 절차를 가속화하기 위해 Large Language Models(LLMs)의 사용 가능성을 탐구합니다. 특히, 이 연구는 LLM을 사용하여 제목과 초록의 스크리닝(title-abstract screening) 과정을 자동화하는 것이 인간의 스크리너보다 정확하게 수행할 수 있는지 그리고 텍스트 단순화(text simplification)가 실제로 스크리닝 성능을 향상시키는지를 실험적으로 평가합니다.

- **Technical Details**: 연구는 GPT-3.5와 GPT-4와 같은 LLMs를 사용하여 기존의 체계적 리뷰에서 수행된 스크리닝 과정을 재현하는 방식으로 진행되었습니다. 또한, 다양한 프롬프트 기술(promoting techniques)—제로샷(Zero-shot, ZS), 원샷(One-shot, OS), 퓨샷(Few-shot, FS), 퓨샷 체인오브쏘트(Few-shot with Chain-of-Thought, FS-CoT)—이 LLMs의 스크리닝 성능에 미치는 영향을 비교 분석하였습니다. 연구 결과, LLM을 통한 텍스트 단순화는 인간의 스크리닝 성능을 크게 개선하지는 않았지만, 스크리닝에 소요되는 시간은 줄어들었습니다.

- **Performance Highlights**: GPT-4는 그 선행 모델인 GPT-3.5 보다 성능이 더 우수한 것으로 나타났으며, FS와 OS 프롬프팅이 ZS 프롬프팅보다 더 나은 성능을 보였습니다. 일부 LLM과 프롬프트 조합은 인간 스크리너만큼의 성능을 보였으나, 현재 LLMs는 인간 스크리너보다 통계적으로 유의미하게 더 정확하지는 않았습니다. LLM을 사용한 제목-초록 스크리닝 자동화는 유망하지만, 이를 SR 과정에 권장하기 위해서는 더 많은 연구가 필요합니다.



### KS-LLM: Knowledge Selection of Large Language Models with Evidence  Document for Question Answering (https://arxiv.org/abs/2404.15660)
- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Models, LLMs)이 지식 중심 작업에서 직면하는 환각 문제(hallucination problem)를 완화하기 위한 새로운 방법, KS-LLM(Knowledge Selection of Large Language Models)을 소개합니다. 이 방법은 입력된 질문에 기반한 triple을 생성하고, 증거 문서에서 이 triple과 가장 유사한 증거 문장을 선택하여 대답을 생성하는 데 도움을 주는 접근 방식을 사용합니다.

- **Technical Details**: KS-LLM 방법은 세 단계로 구성됩니다. 첫째, 질문에 기반한 triple을 생성합니다. 둘째, 생성된 triple과 유사한 증거 문장을 증거 문서에서 선택합니다. 셋째, 선택된 증거 문장과 triple을 결합하여 대규모 언어 모델이 최종 답변을 생성하도록 지원합니다. 이 방법은 텍스트 문장(Textual Evidence Sentences)과 구조화된 triple에 포함된 다양한 형태의 지식을 통합함으로써, 지식의 상호 보완적 관계를 활용하는 데 중점을 둡니다.

- **Performance Highlights**: KS-LLM은 TriviaQA, WebQ, NQ와 같은 여러 QA 데이터셋에서 기존 방법보다 우수한 성능을 보였습니다. 특히, 이 방법은 증거 문서에서 관련성 높은 지식만을 선택함으로써 대규모 언어 모델의 정확성과 신뢰성을 향상시키고, 질의응답 작업에서 환각 문제를 완화할 수 있는 것으로 나타났습니다.



### Return of EM: Entity-driven Answer Set Expansion for QA Evaluation (https://arxiv.org/abs/2404.15650)
Comments: Under Review (9 pages, 3 figures)

- **What's New**: 새로운 방법으로, 본 논문에서는 대규모 언어 모델(LLMs)을 직접 사용하는 기존의 방법이 가지는 한계점, 예를 들어 해석 가능성의 부족, 높은 비용, 환경적인 해를 다루기 위해 소프트 EM(soft EM)과 엔터티 기반 답변 세트 확장 방식을 제안합니다. 이 방법은 다양한 표면 형태를 포함하도록 금 답변 세트를 확장하며, 엔터티 유형에 따라 특정 패턴을 따르는 것을 관찰에 기반하여 개발되었습니다.

- **Technical Details**: 본 방법은 먼저 'Spacy의 명명된-엔터티 인식기(NER)'를 사용하여 QA 데이터를 금 답변의 엔터티 유형별로 분류합니다. 각 엔터티 유형에 대해 구체적으로 맞춤화된 편편 샘플(few-shot prompts)을 적용하여 답변 세트를 확장합니다. 답변의 다양한 형태는 특정 엔터티 유형에 따라 예측 가능한 패턴을 따르기 때문에, InstructGPT와 같은 LLM의 맥락 학습 능력(contextual learning abilities)을 활용하여 이를 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: 이 방법은 전통적인 어휘 일치 지표(lexical matching metrics)보다 현저히 높은 신뢰성을 보여주며, LLM 기반 방법들과 견줄만한 평가 신뢰성을 유지하면서도, 해석 가능성(interpretability)이 높고 환경적 영향을 줄이는 이점을 제공합니다. 사례로, NQ와 TQ 데이터셋에서의 평가 비용은 LLM 기반 방법들이 각각 $0.50과 $0.32인 반면, 제안된 방법은 단 한 번의 데이터셋 확장 비용($1.93, $1.11)만을 요구합니다. 이는 연구 커뮤니티 전체로서의 비용을 대폭 줄이며, 환경적 발자국 또한 줄일 수 있습니다.



### CodeIP: A Grammar-Guided Multi-Bit Watermark for Large Language Models  of Cod (https://arxiv.org/abs/2404.15639)
Comments: 13 pages, 7 figures

- **What's New**: CodeIP는 LLM (Large Language Models) 기반 코드 생성에 다중 비트 정보를 삽입할 수 있는 새로운 워터마킹 (watermarking) 기술을 소개합니다. 이 기술은 생성된 코드의 의미를 유지하면서 워터마크의 강도와 다양성을 개선함으로써, 산업의 지적 재산권(IP) 보호와 교육 분야에서의 학문적 부정 행위 방지에 유용합니다.

- **Technical Details**: CodeIP는 구문 유형 예측기 (type predictor)를 훈련시켜 LLM이 다음 토큰의 구문 유형을 예측할 수 있도록 하여, 생성된 코드의 구문적 및 의미적 정확성을 향상시킵니다. 이 워터마킹 방법은 LLM의 로짓 (logit) 조작을 통해 코드 생성 과정에서 워터마크 메시지를 삽입하며, 이는 코드의 의미에 영향을 줄 수 있기 때문에 구문 정보를 통합하는 새로운 접근방식을 제안합니다.

- **Performance Highlights**: 실험 결과, CodeIP는 자바 (Java), 파이썬 (Python), Go, 자바스크립트 (JavaScript), PHP를 포함한 다양한 실제 데이터 세트에서 평균 0.95의 추출율을 보여주며, 기존 모델 대비 CodeBLEU 손실을 50% 줄여 코드 유틸리티를 유지한 채 강력한 워터마크 성능을 입증합니다.



### Hybrid LLM/Rule-based Approaches to Business Insights Generation from  Structured Data (https://arxiv.org/abs/2404.15604)
- **What's New**: 기업 데이터 분석 분야에서 방대하고 다양한 데이터 세트에서 실행 가능한 인사이트를 추출하는 능력은 정보에 기반한 의사결정을 내리고 경쟁 우위를 유지하는 데 필수적입니다. 전통적인 규칙 기반 시스템(rule-based systems)은 신뢰할 수 있지만, 현대 비즈니스 데이터의 복잡성과 역동성을 만났을 때 종종 부족함을 보입니다. 반면, 인공지능(Artificial Intelligence, AI) 모델, 특히 대규모 언어 모델(Large Language Models, LLMs)은 패턴 인식(pattern recognition)과 예측 분석(predictive analytics)에서 상당한 잠재력을 제공하지만, 특정 비즈니스 응용 프로그램에 필요한 정밀도가 부족할 수 있습니다. 이 논문은 규칙 기반 시스템의 견고함과 LLMs의 적응력을 통합한 하이브리드 접근 방식이 비즈니스 인사이트를 생성하는 효과에 대해 탐구합니다.

- **Technical Details**: 이 연구는 업계 데이터 분석에서 실행 가능한 인사이트를 도출하기 위해 규칙 기반 시스템과 대규모 언어 모델을 결합하는 하이브리드 접근 방식의 유효성을 검토합니다. 규칙 기반 시스템의 정확성과 일관성을 유지하면서도, 대규모 언어 모델의 학습 능력과 적응성을 활용하여 더욱 동적이고 복잡한 데이터 패턴을 인식하고 예측할 수 있는 방안을 제시합니다.



### Minimal Evidence Group Identification for Claim Verification (https://arxiv.org/abs/2404.15588)
- **What's New**: 이 연구에서는 주장 검증을 위한 최소 증거 그룹(Minimal Evidence Groups, MEGs)을 식별하는 문제를 처음으로 정의하고 연구합니다. MEG는 주장을 완전히 지지하면서 중복되지 않고, 필요 최소한의 증거로 구성되어야 합니다. 이는 다양한 관점에서 같은 주장을 검증할 수 있는 여러 증거 집단이 존재할 때 특히 중요합니다.

- **Technical Details**: 이 논문은 MEG 식별 문제를 Set Cover 문제로 환원시킬 수 있음을 보여줍니다. 주어진 증거 그룹이 주장을 얼마나 지지하는지 (전부 혹은 부분적으로) 판단하는 entailment inference에 기반하여 MEG를 식별합니다. 또한, 증거 선택과 관점 예측을 통합해 간단하지만 효과적인 접근 방식을 제안하여 클레임 검증 작업을 개선합니다.

- **Performance Highlights**: 제안된 방식은 WiCE(Widespread Claim Evidence)와 SciFact 데이터셋에서 기존의 큰 언어 모델(Large Language Models, LLM)을 사용한 방식보다 각각 18.4% 및 34.8%의 절대 정밀도(precision) 개선을 이루었습니다. 이는 MEG를 사용함으로써 추론 과정에서의 계산 비용을 절감하고, 클레임 생성과 같은 하류 작업(Downstream tasks)에서의 성능을 향상시킬 수 있음을 시사합니다.



### Can Foundational Large Language Models Assist with Conducting  Pharmaceuticals Manufacturing Investigations? (https://arxiv.org/abs/2404.15578)
Comments: 13 pages, 3 figures

- **What's New**: 이 연구에서는 일반 목적의 대형 언어 모델(Large Language Models, LLM)을 특정 사용 사례인 제약 제조 조사에 적용하는 방법을 탐구하고 있다. 제약 제조에서 발생하는 사고 및 편차의 역사적 기록을 활용하는 것이 새로운 사례를 해결하거나 새로운 제조 캠페인을 위험 감소하는 데 유용할 수 있음을 제안합니다. 특히, GPT-3.5, GPT-4, Claude-2와 같은 세 가지 다양한 LLM을 사용하여 실제 데이터를 기반으로 성능을 평가했습니다.

- **Technical Details**: 이 연구에서는 사례의 근본 원인과 같은 구체적인 정보를 비구조화된 데이터에서 추출하는 LLM의 능력과, 기록된 데이터베이스에서 의미 검색(Semantic Search)을 수행하여 유사하거나 연관된 편차를 식별하는 가능성을 검토합니다. 연구는 매우 다양한 제품 라인에서 선정한 작지만 다양한 실제 제조 편차 데이터셋을 사용합니다.

- **Performance Highlights**: GPT-4와 Claude-2 모델은 정보 추출 작업에서 높은 정확도를 보여 줍니다. 또한, 편차 설명의 벡터 임베딩(Vector Embedding)에 의미 검색을 사용하여 유사한 기록, 예를 들어 유사한 유형의 결함을 가진 기록을 높은 수준의 정확도로 식별할 수 있음을 보여 줍니다. 그러나 LLM의 추론과 환각 행동 사이의 복잡한 상호 작용을 분석하면서 위험 요소로서의 가능성을 논의합니다.



### Retrieval Head Mechanistically Explains Long-Context Factuality (https://arxiv.org/abs/2404.15574)
Comments: Preprint

- **What's New**: 본 연구는 변형기반(transformer-based) 언어 모델에서 장문의 맥락(context) 속에서 관련 정보를 검색하는 능력에 관하여 살펴보았습니다. 이러한 능력이 '검색 헤드(retrieval heads)'라고 불리는 특별한 종류의 주목 헤드(attention heads)에 의해 주로 담당된다는 점을 밝혀냈습니다. 이 검색 헤드들은 다양한 모델에서 관찰되며, 정보 검색에 중요한 역할을 하는 매커니즘이라는 점에서 주목할 만합니다.

- **Technical Details**: 검색 헤드는 다음과 같은 특징을 보입니다: (1) 보편성(universal): 장문 맥락을 다루는 모든 모델에 검색 헤드가 존재합니다. (2) 희소성(sparse): 전체 주목 헤드 중 극소수(5% 미만)만이 검색 헤드입니다. (3) 내재성(intrinsic): 짧은 맥락으로 사전 학습된(pretrained) 모델에 이미 검색 헤드가 존재하며, 맥락 길이를 연속적으로 확장해도 동일한 헤드가 정보 검색을 담당합니다. (4) 동적 활성화(dynamically activated): 예를 들어, Llama-2 7B 모델의 경우 12개의 검색 헤드가 맥락 변경에 관계없이 항상 필요한 정보를 주목합니다. 나머지 검색 헤드는 다른 맥락에서 활성화됩니다. (5) 원인성(causal): 검색 헤드를 완전히 제거하면 정보 검색 실패와 허구 생성(hallucination)이 발생합니다.

- **Performance Highlights**: 이 연구를 통해 얻은 통찰력은 모델이 입력 토큰에서 정보를 검색하는 내부 메커니즘을 명확히 해줍니다. 특히, '사슬 추론(chain-of-thought, CoT) 추론'에서 정보 검색이 중요하게 작용하며, 이는 질문과 이전에 생성된 맥락을 반복적으로 참조해야 할 때 더욱 중요합니다. 반면, 모델이 내재된 지식만을 사용해 직접적으로 답을 생성하는 경우, 검색 헤드의 마스킹(masking)이 큰 영향을 미치지 않습니다. 이러한 발견은 향후 연구에 있어 hallucination의 감소, 추론 능력의 개선, KV 캐시의 압축 등에 기여할 것입니다.



### CASPR: Automated Evaluation Metric for Contrastive Summarization (https://arxiv.org/abs/2404.15565)
- **What's New**: 이 연구에서는 기존의 토큰 겹침(token-overlap)에 기반한 측정 방식인 'Distinctiveness Score'와는 다르게, 'CASPR'이라는 새로운 자동 평가 지표를 제안하고 있습니다. CASPR은 자연언어추론(Natural Language Inference, NLI) 작업을 활용하여 한 쌍의 요약 사이의 대조성을 보다 정확하게 측정하는 데 초점을 맞추고 있습니다.

- **Technical Details**: CASPR 메트릭은 각 요약을 단일 주장 문장으로 세분화하고, 여러 NLI 점수를 신중하게 집계하여 요약 수준의 대조성 점수를 산출하는 간단하면서도 가벼운 방법을 기반으로 합니다. 이 과정에는 'roberta-large-mnli' 모델을 사용하여 두 문장 간의 논리적 관계를 평가하고 있습니다. 또한, 이 연구는 기존의 BERTScore를 변형한 간단하지만 강력한 기준 모델(inverted BERTScore)과 비교하여 CASPR의 우수성을 입증하고 있습니다.

- **Performance Highlights**: CoCoTRIP 데이터셋에서의 실험 결과, CASPR은 기존의 'Distinctiveness Score' 및 BERTScore 기반 기준 모델보다 요약 쌍의 대조성을 더 신뢰성 있게 포착할 수 있음을 보여줍니다. 특히, CASPR은 단어 차이를 넘어서 요약 간의 논리적 대비를 감지하는 능력이 뛰어나다는 것이 입증되었습니다.



### PRISM: Patient Records Interpretation for Semantic Clinical Trial  Matching using Large Language Models (https://arxiv.org/abs/2404.15549)
Comments: 30 Pages, 8 Figures, Supplementary Work Attached

- **What's New**: 이 연구에서는 실제 환자의 전자 건강 기록(EHRs)을 사용하여 임상 시험 매칭을 위한 대규모 실증 평가를 처음으로 선보입니다. 특히 저희가 개발한 OncoLLM 모델은 GPT-3.5를 능가하며, 의료 전문가와 동등한 수준의 성능을 보여주었습니다.

- **Technical Details**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 환자 정보와 임상 시험의 포괄적 요구 사항을 자동으로 매칭합니다. 사용된 모델로는 GPT-4, GPT-3.5 및 자체 조정된 모델인 OncoLLM이 있습니다. 테스트는 미국의 한 암 센터에서 제공하는 실제 EHR 데이터와 임상 시험 정보를 바탕으로 수행되었습니다.

- **Performance Highlights**: OncoLLM은 GPT-3.5보다 우수한 성능을 보였고, 의료 전문가와 동등한 결과를 달성했습니다. 이는 LLMs가 임상 시험 매칭에서 실제로 유효한 도구임을 시사합니다.



### Towards Systematic Evaluation of Logical Reasoning Ability of Large  Language Models (https://arxiv.org/abs/2404.15522)
Comments: 29 Pages

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 논리적 추론(logical reasoning) 능력을 평가하기 위해 LogicBench라는 새로운 질문 응답 데이터셋을 도입했습니다. LogicBench는 제안, 일차 명제, 비단조적 논리(propositional, first-order, and non-monotonic logics)를 포함하는 25가지 추론 패턴을 다루며, 이는 LLMs의 논리적 추론 능력을 평가하는데 중요한 진전을 나타냅니다. 이 데이터셋은 기존의 데이터셋들과는 달리 단일 추론 규칙(single inference rule)을 사용하여 LLMs를 평가합니다.

- **Technical Details**: LogicBench는 자연 언어로 구성된 문맥(context)과 그에 따른 논리적 결론을 평가하는 이진 질문 응답(Binary Question-Answering, BQA) 과 다중 선택 질문 응답 (Multiple-Choice Question-Answering, MCQA) 두 가지 작업을 포함합니다. 특히, GPT-4, ChatGPT, Gemini, Llama-2, Mistral과 같은 다양한 LLMs를 사용하여 체인 오브 소트(chain-of-thought) 프롬프트 방법을 통해 평가하였으며, 이 데이터셋의 구성은 추론 규칙에 따른 문맥과 질문 쌍을 생성하는 세 단계 절차를 통해 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 대부분의 LLMs는 LogicBench에서 복잡한 논리적 문맥과 부정(negations)을 포함한 추론 규칙을 처리하는데 어려움을 겪었으며, 이는 이러한 모델들이 논리적 추론 능력을 개선할 필요가 있음을 시사합니다. 실제로, LLMs는 평균적으로 LogicNLI 및 FOLIO 데이터셋에서 약 2%의 성능 향상을 보였으며, LogiQA와 ReClor에서도 경쟁력 있는 성능을 보였습니다.



### ToM-LM: Delegating Theory Of Mind Reasoning to External Symbolic  Executors in Large Language Models (https://arxiv.org/abs/2404.15515)
- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 마음 이론(Theory of Mind, ToM) 추론 능력 향상을 위해 새로운 접근 방식을 제안합니다. 본 연구는 심볼릭 실행기(Symbolic Executor), 특히 SMCDEL 모델 검사기를 활용하여 LLM의 ToM 추론 능력을 개선합니다. 이 접근법은 LLM이 자연어로 제시된 ToM 문제를 심볼릭 형식으로 변환하고, 이를 SMCDEL 모델 검사기가 실행하여 투명하고 검증 가능한 ToM 추론을 수행하도록 합니다.

- **Technical Details**: LLM은 먼저 자연어 및 심볼릭 형식으로 표현된 ToM 문제 쌍을 통해 미세 조정(fine-tuning)됩니다. 이후, LLM은 일회성 인-콘텍스트 예제를 사용해 심볼릭 형식을 생성하도록 지시받습니다. 생성된 심볼릭 형식은 SMCDEL 모델 검사기에 의해 실행되어 최종 결과를 제공합니다. 이 연구는 ToM 추론의 특정 구성 요소를 외부화하고 벨리프(beliefs)에 대한 추론을 일반화하는 새로운 관점을 제안합니다.

- **Performance Highlights**: ToM-LM은 정확도 및 ROC 곡선 아래 영역(Area under the ROC Curve, AUC)에 관한 면에서 모든 구성된 베이스라인을 상당히 개선함으로써 LLM의 ToM 추론 능력을 현저히 향상시킵니다. 이 결과는 외부 실행 도구를 활용하여 ToM 추론 과정을 투명하고 설명 가능하게 만들 수 있음을 시사합니다.



### Killkan: The Automatic Speech Recognition Dataset for Kichwa with  Morphosyntactic Information (https://arxiv.org/abs/2404.15501)
Comments: 11 pages, 9 tables, 3 figures, to be published in LREC-COLING 2024

- **What's New**: 이 논문은 에콰도르의 토착 언어인 Kichwa어를 위한 첫 번째 자동 음성 인식(ASR: Automatic Speech Recognition) 데이터셋인 Killkan을 소개합니다. Kichwa어는 매우 소규모의 자원이 부족한 위기에 처한 언어로, Killkan 전에는 자연어 처리(NLP: Natural Language Processing) 응용 프로그램에 Kichwa어를 통합할 수 있는 자원이 없었습니다.

- **Technical Details**: 이 데이터셋은 약 4시간 분량의 오디오를 포함하며, 이 오디오는 스페인어로 번역되고 Universal Dependencies 형식으로 형태 구문론적(morphosyntactic) 주석이 달린 텍스트로 전사되어 있습니다. 오디오 데이터는 Kichwa어로 방송된 공개 라디오 프로그램에서 추출되었습니다. 또한, 이 논문은 Kichwa어의 응집 어형(agglutinative morphology)과 스페인어와의 빈번한 코드 교환(code-switching)에 특별한 초점을 두고 데이터셋의 언어학적 분석을 제공합니다.

- **Performance Highlights**: 연구 결과에 따르면, 이 데이터셋은 작은 데이터셋 크기에도 불구하고 신뢰할 수 있는 품질의 Kichwa어를 위한 첫 번째 ASR 시스템 개발이 가능하게 합니다. 이 데이터셋, ASR 모델, 그리고 이를 개발하는 데 사용된 코드는 모두 공개될 예정입니다. 따라서, 우리의 연구는 자원이 부족한 언어와 그 커뮤니티를 위한 자원 구축 및 그 응용을 긍정적으로 보여줍니다.



### IryoNLP at MEDIQA-CORR 2024: Tackling the Medical Error Detection &  Correction Task On the Shoulders of Medical Agents (https://arxiv.org/abs/2404.15488)
- **What's New**: 이 논문은 MedReAct’N’MedReFlex, 다중 에이전트 프레임워크를 소개하며, 의료 오류 감지 및 수정을 위해 새롭게 개발되었습니다. 이 프레임워크는 MedReAct, MedReFlex, MedEval, MedFinalParser라는 네 종류의 의료 에이전트를 통합하여 사용하며, 복잡한 임상 필드에서의 오류 처리를 체계적으로 접근합니다. 특히, Retrieval-Augmented Generation (RAG) 프레임워크를 기반으로 하여 ClinicalCorp 데이터베이스를 활용합니다.

- **Technical Details**: MedReAct’N’MedReFlex 프레임워크는 크게 네 가지 에이전트로 구성됩니다: 1) MedReAct는 임상 노트의 잠재적 오류를 탐지하는 초기 단계를 담당, 2) MedEval는 타겟된 오류와 제안된 수정사항을 평가, 3) MedReFlex는 반영적 분석을 통해 대안적 전략을 제안, 4) MedFinalParser는 최종 출력을 포맷팅하며 원본 스타일을 유지함과 동시에 오류 수정 과정의 정확성을 보장합니다. 이 프레임워크는 또한 MedWiki 데이터세트와 같은 오픈소스 리소스를 활용하여 임상 RAG 응용 프로그램을 지원합니다.

- **Performance Highlights**: MedReAct'N'MedReFlex는 MEDIQA-CORR 2024 경쟁에서 9위를 차지하며, RAG 방식과 클리니컬 코퍼스(ClinicalCorp)를 활용한 접근 방식이 중요한 역할을 수행함을 입증했습니다. 이 프레임워크의 사용은 임상 필드에서의 문서 검토 및 수정 작업의 정확성과 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Large Language Models Spot Phishing Emails with Surprising Accuracy: A  Comparative Analysis of Performanc (https://arxiv.org/abs/2404.15485)
Comments: 7 pages, 3 figures

- **What's New**: 이 논문은 디지털 세계에서 여전히 큰 위협인 피싱(Phishing)을 탐지하기 위해 대규모 언어 모델(Large Language Models, LLMs)의 효과를 분석합니다. 특히 '419 스캠' 이메일을 대상으로 한 무작위 데이터 세트를 사용하여 실험을 진행했습니다.

- **Technical Details**: 연구는 15개의 다양한 대규모 언어 모델을 사용하여 피싱 시도를 감지하는 능력을 평가하였습니다. 이메일 메타데이터를 포함한 텍스트 파일을 기반으로 평가가 이루어졌으며, 예정된 기준에 따라 분석되었습니다.

- **Performance Highlights**: 실험 결과, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, 그리고 ChatGPT 모델이 피싱 이메일을 감지하는데 가장 효과적인 것으로 나타났습니다. 이 모델들은 정교한 사회공학적 요소와 심리적 전략을 사용하는 피싱 이메일들을 정확하게 탐지할 수 있는 능력을 보였습니다.



### XC-Cache: Cross-Attending to Cached Context for Efficient LLM Inferenc (https://arxiv.org/abs/2404.15420)
- **What's New**: 이 연구에서는 맥락 내 학습(In-context Learning, ICL)의 한계를 극복하기 위한 새로운 모델을 제안합니다. 기존 디코더 중심의 언어 모델에서 맥락을 텍스트와 함께 제공하는 프롬프트 없이 크로스-어텐션(cross-attention)을 사용하여 생성을 조건화하는 인코더-디코더 아키텍처에 영감을 받은 새로운 접근방식을 도입합니다. 특히, 사전 훈련된 디코더-전용 모델을 활용하고 소수의 추가 계층만을 훈련함으로써 매개변수 효율성을 보장합니다.

- **Technical Details**: 본 연구에서 개발한 모델은 크로스-컨텍스트 캐시(Cross-Context-Cache, XC-Cache)를 도입하여 인코더의 숨겨진 상태만을 저장하고, 크로스-어텐션을 통해 캐시를 추론 시 통합합니다. 이는 기존 ICL의 캐시 용량 요구를 대폭 감소시키며, 두 가지 구현 방식인 XC-Llama와 XC-LlamaEnc를 제시합니다. XC-Llama는 고정된 디코더를 인코더로 사용하며, XC-LlamaEnc는 작은 양방향 인코더를 사용합니다.

- **Performance Highlights**: 우리의 접근방식은 ICL 대안보다 일관되게 우수한 성능을 보이고, 인코더-디코더 구조를 사용하여 캐싱 메모리 요구를 98% 이상 줄입니다. 자동 질문응답(Question-Answering, QA) 테스트를 통해 이들 모델이 ICL을 능가하며, 프롬프트로 미세조정된 대규모 언어 모델(Large Language Models, LLMs)과 비교해볼 때 경쟁력 있는 성능을 나타냅니다. 또한, 다양한 훈련 과제를 통해 맥락 조건적 생성을 위해 몇 개의 크로스-어텐션 계층만을 훈련하는 것이 충분하다는 것을 보여줍니다.



### Cantor: Inspiring Multimodal Chain-of-Thought of MLLM (https://arxiv.org/abs/2404.16033)
Comments: The project page is available at this https URL

- **What's New**: 이 논문은 시각적 추론 문제를 해결하기 위해 멀티모달 연쇄 사고 (CoT: Chain-of-Thought) 방법론을 활용한 새로운 프레임워크, 칸토르(Cantor)를 제안합니다. 이는 기존의 시각적 정보와 논리적 추론을 통합하여 복잡한 시각적 추론 작업을 해결하는 구조로 설계되었습니다.

- **Technical Details**: 칸토르 프레임워크는 두 부분, 결정 생성(Decision-Generation) 부분과 실행(Execution) 부분으로 나뉩니다. 처음에는 MLLM 혹은 LLM을 사용하여 문제의 시각적 및 텍스트 맥락을 동시에 처리하고 복잡한 추론 과정을 거칩니다. 이후 여러 '전문가' 역할을 하는 하나의 MLLM에 의해 수행됩니다. 전문가들은 각자 다른 역할과 요구사항을 가지고 참여하여 높은 수준의 정보를 제공합니다.

- **Performance Highlights**: 칸토르 프레임워크는 ScinceQA 및 Mathvista 데이터셋에서 상태-의-기술(SOTA: State-of-the-Art) 성능을 달성했습니다. Gemini를 사용할 때 각각 4.11%, 5.9%의 정확도 향상을 보였고, GPT-3.5를 사용했을 때는 2.24%, 9.2%의 정확도 향상을 보였습니다. 이는 기존 방법들을 크게 앞서는 결과입니다.



### MoDE: CLIP Data Experts via Clustering (https://arxiv.org/abs/2404.16030)
Comments: IEEE CVPR 2024 Camera Ready. Code Link: this https URL

- **What's New**: 이 연구에서는 웹에서 수집한 데이터 셋의 잡음 문제를 집중적으로 다루고, 특히 CLIP 학습에 있어 흔히 발생하는 거짓 음성(false negatives)에 대한 해결 방안을 제시합니다. Mixture of Data Experts (MoDE)라는 새로운 프레임워크를 도입하여, 데이터 클러스터링을 통해 다수의 데이터 전문가 시스템을 학습하고, 추론 시점에서는 작업 메타데이터(task metadata)와 클러스터 조건 간의 상관관계를 통해 동적으로 데이터 전문가들을 앙상블합니다.

- **Technical Details**: MoDE 프레임워크는 데이터를 클러스터링한 뒤, 각 클러스터를 기반으로 별도의 CLIP 데이터 전문가 모델을 훈련합니다. 이러한 접근 방식은 클러스터 내에서 의미론적으로 유사한 캡션을 사용하여 대조적 학습을 수행함으로써, 훈련 중 거짓 음성의 영향을 줄이고 훈련 효율을 높입니다. 추론 시에는 각 데이터 전문가의 출력을 우선 순위를 두고 결합하여 최종 분류 결과를 도출합니다.

- **Performance Highlights**: MoDE는 여러 벤치마크에서 최신 기술 대비 우수한 성능을 보여 주었습니다. 예를 들어, 이미지 분류에서 CLIP 벤치마크 기준 3.7% 향상됐으며, 이미지-텍스트 및 텍스트-이미지 검색에서 각각 3.3% 및 2.7% 향상을 달성했습니다. 또한, MoDE는 새로운 데이터 전문가를 유연하게 포함할 수 있고, 대규모의 이미지-캡션 쌍 데이터셋을 효율적으로 훈련할 수 있는 장점을 보유하고 있습니다.



### Uncertainty Estimation and Quantification for LLMs: A Simple Supervised  Approach (https://arxiv.org/abs/2404.15993)
Comments: 29 pages, 14 figures

- **What's New**: 이 논문은 대규모 언어 모델 (LLMs: Large Language Models)의 출력에 대한 불확실성 추정 및 캘리브레이션 문제를 연구합니다. 기존의 머신러닝(ML)과는 달리, LLMs를 위한 불확실성 추정에는 레이블이 지정된 데이터셋을 활용하는 감독 학습(supervised learning) 접근 방식을 제안하고, 이를 통해 LLMs의 은닉 활성화(hidden activations)에서 불확실성 정보를 추출하는 새로운 방법을 모색합니다. 또한, 불확실성 추정과 캘리브레이션의 차이를 구분하고, 더 나은 불확실성 추정 방법이 캘리브레이션 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 LLMs의 은닉층 활성화 값을 활용하여 불확실성을 추정하는 방법을 소개하며, 이는 추론 중에 입력 및 출력의 변수들에서 추가적인 특징을 요구합니다. 감독 학습 접근 방식은 불확실성 점수(uncertainty score)를 출력하는 함수를 훈련시키며, 이는 LLM의 응답에 대한 확신(confidence) 수준을 평가할 수 있게 해줍니다. 이 방법은 흑백 박스(black box), 회색 박스(grey box), 백백 박스(white box) 모델에 적용 가능하며, 각각의 접근성 모드에서 효과적인 성능을 보입니다.

- **Performance Highlights**: 이 방법은 질문 응답(question answering), 기계 번역(machine translation), 다중 선택(multiple choice)과 같은 NLP 작업에서 기존 벤치마크와 비교하여 수행되었습니다. 실험 결과, LLMs의 은닉 활성화를 활용함으로써 모델에서 추가적인 지식을 추출하여 다양한 NLP 작업에서 불확실성 추정을 향상시킬 수 있음을 보여주었습니다. 이러한 발견은 불확실성 추정 방법의 작동 원리와 그 견고성 및 이전 가능성(transferability)에 대한 통찰력을 제공합니다.



### Inside the echo chamber: Linguistic underpinnings of misinformation on  Twitter (https://arxiv.org/abs/2404.15925)
- **What's New**: 새로운 연구에서, 소셜 미디어 사용자들이 어떻게 언어 사용을 통해 정보를 중재하고 에코 챔버(echo chambers) 내에서의 그룹 정체성 신호와 처리 유창성(processing fluency)를 향상시키는지에 대해 탐구했습니다. 이 연구는 소셜 네트워크와 언어가 상호작용하는 방식에 대한 이해를 깊게 하고, 잘못된 정보가 어떻게 퍼지는지에 대한 보다 세밀한 분석을 제공합니다.

- **Technical Details**: 연구자들은 사용자 상호작용 네트워크에서 에코 챔버를 식별하고 이들의 언어적 특성을 분석했습니다. 그들은 인그룹/아웃그룹(In-/Out-group) 신호, 가독성(readability), 그리고 대화 연결사(discourse connectives)와 같은 언어적 척도를 비교했습니다. 연구는 에코 챔버 내에서 그룹 정체성 신호의 증가와 정보의 처리가 유창해지는 현상을 관찰했습니다.

- **Performance Highlights**: 이 연구는 에코 챔버 내의 대화에서 처리 유창성이 향상되고, 사용자들이 그룹 정체성을 더 강하게 신호하는 경향이 더 두드러진다는 점을 발견했습니다. 이러한 현상은 특히 잘못된 정보와 관련된 논의에서 더욱 두드러졌습니다. 그러나 이러한 언어적 특성은 모든 논의에서 일관되게 나타나지는 않으며, 주제의 정치적 성격과 같은 맥락적 요인에 따라 달라질 수 있습니다.



### KGValidator: A Framework for Automatic Validation of Knowledge Graph  Construction (https://arxiv.org/abs/2404.15923)
Comments: Text2KG 2024, ESWC 2024

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 지식 그래프(Knowledge Graph, KG) 완성 모델의 자동 평가에 대해 탐구합니다. KGValidator라는 새로운 프레임워크를 도입하여 지식 그래프의 일관성과 검증을 지원하며, LLM 자체의 내재된 지식, 사용자가 제공한 문헌 집합, 외부 지식 소스 등 다양한 정보원을 활용합니다.

- **Technical Details**: KGValidator는 Instructor, Pydantic 클래스, 함수 호출을 사용하여 LLM이 정확한 가이드라인을 따르도록 하며, 평가 지표를 계산하기 위한 올바른 데이터 구조를 출력하게 합니다. 이 프레임워크는 오픈 소스 라이브러리를 기반으로 하며, 어떠한 유형의 지식 그래프에도 쉽게 적용 및 확장이 가능합니다.

- **Performance Highlights**: 이 프레임워크는 널리 사용되는 KG 완성 벤치마크 데이터셋에 대한 효과적인 검증 수단으로 평가되었으며, 추가적인 맥락을 제공할 때 SoTA(State-of-the-Art) LLM의 평가 능력이 증진되는 효과를 확인하였습니다. 또한, 사용자 프롬프트와 입력을 함께 포함시키는 방식을 통해 제로 샷(zero-shot) 설정에서도 성공적인 성능을 보였습니다.



### BASS: Batched Attention-optimized Speculative Sampling (https://arxiv.org/abs/2404.15778)
- **Korean AI Newsletter**: [{"What's New": '이 논문은 여러 시퀀스를 동시에 처리할 수 있는 Batched Attention-optimized Speculative Sampling (BASS) 시스템을 소개합니다. 이 시스템은 대규모 언어 모델의 대기 시간(latency)과 GPU 사용률을 현저히 개선하는 새로운 접근 방식을 제공하며, 기존의 단일 시퀀스 추론 방식보다 월등한 성능을 보여줍니다.'}, {'Technical Details': 'BASS는 CUDA 커널(customized CUDA kernels)을 사용하여 배치(batch) 중에 발생할 수 있는 불규칙한 텐서(ragged tensors)를 처리합니다. 이 방법은 각 배치의 드래프트 토큰(draft tokens) 길이를 동적으로 조정하는 휴리스틱(heuristic)을 설계함으로써 더 높은 GPU 사용률을 달성하고, 결국 시퀀스 생성 시간을 단축시킵니다. 병렬 처리 및 추론 속도의 최적화가 주요 초점입니다.'}, {'Performance Highlights': 'BASS 시스템은 A100 GPU 하나를 사용하여 7.8B 크기의 모델로 배치 크기(batch size) 8의 설정에서 토큰당 평균 5.8ms의 속도로 시퀀스를 생성하며, 전반적인 처리량(throughput)은 초당 1.1K 토큰에 달합니다. 예비 모델과 주 모델 간의 스펙큘러티브 디코딩(speculative decoding) 방식은 정규 디코딩(optimized regular decoding)보다 2.15배 빠른 속도를 보여주며, GPU 사용률은 최대 15.8%에 이르러 일반 디코딩의 3배 이상, 단일 시퀀스 스펙큘러티브 디코딩의 약 10배입니다.'}]



### ChEX: Interactive Localization and Region Description in Chest X-rays (https://arxiv.org/abs/2404.15770)
- **What's New**: 새로 개발된 Chest X-Ray Explainer (ChEX) 모델은 흉부 X-레이 이미지를 기반으로 상세한 텍스트 설명을 생성하므로, 의료 실무에서 요구하는 상호작용성(interactivity)과 지역화된 이해성(localized interpretability)을 제공합니다. 이 모델은 사용자 쿼리를 통해 생성 프로세스를 조정할 수 있으며, 시각적 근거를 제시함으로써 예측을 보다 명백하게 할 수 있습니다.

- **Technical Details**: ChEX는 멀티태스크 아키텍처와 훈련 패러다임(multitask architecture and training paradigm)을 적용하여 텍스트 쿼리(textual prompts)와 경계 상자(bounding boxes)를 통합합니다. 이러한 접근 방식은 해부학적 영역(anatomical regions)과 병리학(pathologies)을 포함한 다양한 요소에 대해 효과적으로 작동합니다. ChEX는 각 쿼리에 대해 개별적인 설명을 예측할 수 있으며, 텍스트 쿼리의 경우 해당 영역을 지역화하기 위한 경계 상자를 추가로 제공합니다.

- **Performance Highlights**: ChEX는 9가지 다양한 흉부 X-레이 작업에 걸쳐 평가되었으며, 보고서 생성(report generation), 병리 검출(pathology detection), 문장 정밀화(sentence grounding), 영역 분류(region classification), 영역 설명(region explanation) 등을 포함한 지역화된 이미지 해석 작업에서 경쟁 모델들과 비교하여 우수한 성능을 보였습니다. 또한, ChEX는 상호작용적인 기능을 통해 사용자 프롬프트에 효과적으로 반응하는 것으로 분석되었습니다.



### CatLIP: CLIP-level Visual Recognition Accuracy with 2.7x Faster  Pre-training on Web-scale Image-Text Data (https://arxiv.org/abs/2404.15653)
- **What's New**: 이 논문은 대규모 이미지-텍스트 데이터를 활용한 비전 모델의 pre-training을 분류 문제로 재구성함으로써, contrastive loss에서 필요한 쌍별 유사도 계산의 필요성을 제거하는 새로운 약간 지도방식을 제안합니다. 이를 통해 훈련 속도가 2.7배 향상되었습니다. 해당 방법을 CatLIP (Categorical Loss for Image-text Pre-training)이라 명명하였습니다.

- **Technical Details**: CatLIP은 이미지와 텍스트의 쌍을 직접 비교하는 대신에, 이미지에 대한 텍스트 캡션에서 추출한 명사를 다중 레이블로 사용하여 이미지를 분류하는 문제로 전환합니다. 이 접근방식은 contrastive learning에 비해 계산 비용을 줄이면서도 대규모 데이터를 효율적으로 활용할 수 있게 합니다. 실험을 통해 추가된 ImageNet-1k 데이터셋의 재현성을 확인하고, 여러 하위 작업(예: 객체 검출 및 의미론적 분할)에서 고성능을 유지하는 것을 입증하였습니다.

- **Performance Highlights**: CatLIP은 전통적인 contrastive learning 접근법인 CLIP에 비해 pre-training에 소요되는 시간을 2.7배 줄였습니다. 대규모 이미지-텍스트 데이터세트(DataComp-1.3B)에서의 pre-training에서 특히 두드러진 효율성을 보였으며, CLIP와 유사한 downstream task 성능을 보장합니다. 구체적으로, 변형된 Vision Transformer (ViT B/16)를 사용한 Mask R-CNN은 COCO 데이터셋에서 평균 정밀도(mean average precision) 49.9를 달성, CLIP과 동등한 결과를 보여주었습니다.



### ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for  Implicit Attribute Value Extraction (https://arxiv.org/abs/2404.15592)
- **What's New**: 새로운 ImplicitAVE 데이터셋을 소개합니다. 이는 공개적으로 사용 가능한 첫 번째 다중 모드(멀티모달) 데이터셋으로서 암시적 속성 값 추출에 초점을 맞추고 있습니다. 상품의 텍스트만이 아니라 이미지까지 활용하여 값들을 추론할 수 있도록 설계되었습니다. 68k의 훈련 데이터와 1.6k의 테스트 데이터로 구성되어 있으며, 25개의 속성에 대해 암시적 정보를 처리할 수 있도록 상세하게 구성되어 있습니다.

- **Technical Details**: ImplicitAVE는 기존의 MAVE 데이터셋에서 파생되어, 불필요한 속성을 제거하고 중복된 값을 정리함으로써 더 효율적인 데이터셋을 구축하였습니다. 데이터셋은 상품 제목, 범주, 속성-값 주석을 포함하며, 이미지를 포함하는 다중 모드 데이터셋으로 확장되었습니다. GPT-4 같은 언어 모델을 사용하여 값 병합 및 정제 과정을 진행하였으며, 두 번의 인간 검사를 통해 데이터의 정확도를 높였습니다.

- **Performance Highlights**: 제안된 ImplicitAVE 데이터셋에서는 Multi-modal Large Language Models(MLLMs)을 이용하여 잠재성능 평가를 수행하였습니다. 다양한 시나리오(예를 들어, 전체/소수 샷 및 제로 샷 시나리오 등)에서 11가지 변형 모델이 테스트되었으며, 암시적 값 추출은 여전히 MLLMs에게 도전적인 과제로 남아 있음이 드러났습니다. 이 데이터셋은 암시적 속성 값 추출 연구에 있어 더 많은 통찰과 향후 연구 방향을 제공합니다.



### DreamCraft: Text-Guided Generation of Functional 3D Environments in  Minecraf (https://arxiv.org/abs/2404.15538)
Comments: 16 pages, 9 figures, accepted to Foundation of Digital Games 2024

- **What's New**: 본 논문에서는 자유 양식의 텍스트 프롬프트(free-form text prompts)를 사용하여 오픈월드 게임 Minecraft에서 기능적인 3D 아티팩트를 생성하는 새로운 방법인 DreamCraft를 제시합니다. 이 방법은 양자화된 신경 복사 필드(Neural Radiance Fields, NeRF)를 훈련시켜 텍스트 설명과 일치하는 인게임 아티팩트를 생성합니다. 이는 토지 생산성 생성(Procedural Content Generation, PCG)과 생성 AI 접근법의 강점을 결합하여 유연하면서 기능적인 콘텐츠 생성을 위한 새로운 가능성을 엽니다.

- **Technical Details**: DreamCraft는 양자화된 NeRF를 활용하여 Minecraft 블록의 이산 집합을 사용하여 주어진 텍스트 설명과 일치하는 3D 구조를 생성합니다. 이 방식은 연속적인(continuous) 블록에서 이산(discrete) 블록으로 변경하는 다양한 양자화 체계를 실험하며, 현장의 표현력과 제어력을 유지하면서도 도메인 특정 목표를 통해 기능적 제약을 통합할 수 있는 능력을 입증합니다.

- **Performance Highlights**: DreamCraft는 제약이 없는 NeRF의 출력을 후처리하는 기준 모델에 비해 일관성 있게 텍스트 입력과 일치하는 인게임 아티팩트를 생성합니다. 블록 패턴에 대한 지역적 기능적 제약을 적용할 수 있는 손실 항을 공동으로 최적화함으로써, 목표 분포에 맞는 3D 구조를 생성하거나 블록 유형에 대한 인접 규칙을 준수하는 방법을 보여줍니다.



### BattleAgent: Multi-modal Dynamic Emulation on Historical Battles to  Complement Historical Analysis (https://arxiv.org/abs/2404.15532)
Comments: 26 pages, 14 figures The data and code for this project are accessible at this https URL

- **What's New**: 이 논문에서는 대규모 시각-언어 모델(Large Vision-Language Model)과 다중 에이전트 시스템(Multi-agent System)을 결합한 새로운 에뮬레이션 시스템인 BattleAgent를 소개합니다. BattleAgent는 여러 에이전트 간, 그리고 에이전트와 그들의 환경간 복잡한 동적 상호작용을 시간에 따라 시뮬레이션합니다. 이 시스템은 역사적 사건들을 생생하고 포괄적인 방식으로 재현함으로써, 다양한 관점에서 개인들의 생각과 감정에 대한 통찰력을 제공합니다.

- **Technical Details**: BattleAgent는 LLM과 VLM을 기반으로 다양한 전투 관련 활동을 지원하는 맞춤형 에이전트 구조를 개발했습니다. 에이전트는 스카우팅(scouting), 참호 파기(trench digging) 등을 포함하여 특정 상황에 필요한 작업을 수행할 수 있습니다. 이러한 에이전트는 전투 중 상호작용(interactions), 피해, 정서적 반응 및 심리적 상태를 기록하여, 전투에 참여한 개별 병사의 경험을 상세히 기록합니다.

- **Performance Highlights**: BattleAgent는 복잡한 지형과 계층적 명령 구조에서 역사적 전투를 에뮬레이트합니다. 이는 전략적 기동(strategic maneuvers), 물류 계획(logistical considerations), 그리고 커뮤니케이션 다이내믹스(communication dynamics)을 정교하게 반영합니다. 특히, BattleAgent는 다양한 각도에서 역사적 사건을 이해하는 데 도움을 주어, 교육적 도구로서 활용될 수 있을 뿐만 아니라 차세대 게임 엔진(next-generation game engine)으로서의 잠재력도 갖추고 있습니다.



### GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots (https://arxiv.org/abs/2404.15500)
Comments: Earthvision 2024, CVPR Workshop

- **What's New**: GeoLLM-엔진은 지리적 원격 관측 플랫폼에서 현실적인 고차원 자연어 명령(Natural Language Commands)을 해석하고 작업 수행을 평가하기 위해 다양한 지리 공간 API 도구(Geospatial API Tools), 동적 지도/사용자 인터페이스(Dynamic Maps/UIs), 외부 다중모드 지식 기반(External Multimodal Knowledge Bases)을 통합한 환경을 제공합니다. 대규모 병렬 엔진을 사용하여 100개의 GPT-4-Turbo 노드를 거쳐 백만 개가 넘는 위성 이미지를 사용하여 50만 개 이상의 다양한 멀티-툴 작업(Multi-Tool Tasks)을 생성합니다.

- **Technical Details**: 이 연구에서는 복잡한 테스크를 수행하기 위한 '엔진(Engine)' 구축에 초점을 맞추며, 기존 벤치마크의 단순한 이미지-캡션 작업 접근법을 벗어나 독창적인 GeoLLM-엔진을 개발하였습니다. 특히, 자연어 처리(Natural Language Processing, NLP) 기반의 고정밀 작업 검증 기술을 도입하여 사람의 개입을 최소화하고, 벤치마크 생성과 검증 과정에서의 집중도와 정확성을 향상시켰습니다. 또한, GeoLLM-엔진은 Mapbox API나 Rasterio, GeoPandas 같은 오픈소스 라이브러리를 활용하여 효과적인 데이터 분석 및 작업 수행이 가능합니다.

- **Performance Highlights**: GeoLLM-엔진은 다양한 위성 이미지를 활용하여 고도로 복잡한 작업에서의 에이전트 성능을 평가합니다. 초기 실험 결과, GPT-4 Turbo 같은 최신 LLM을 사용하여 복잡성이 증가하는 다양한 작업에서 에이전트의 성능이 상당히 향상됨을 보여주었습니다. 새로운 벤치마크는 기존 보다 더 다양한 원격 탐지(Remote Sensing, RS) 응용 프로그램을 아우르며, 더욱 정교하고 실제와 가까운 지리공간 데이터 분석 거버넌스를 제공합니다.



### Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal  LLMs (https://arxiv.org/abs/2404.15406)
Comments: CVPR 2024 Workshop on What is Next in Multimodal Foundation Models

- **What's New**: 새롭게 소개되는 연구인 Wiki-LLaVA는 여러 모달(Multimodal) 문서로 이루어진 외부 지식 원천을 통합하여 Multimodal Large Language Models (MLLMs, 멀티모달 대규모 언어 모델)의 대답 생성능력을 향상시키는 방법론입니다. 연구팀은 계층적 검색 파이프라인을 사용하여 관련 문서를 검색하고 이를 MLLM의 추가적인 맥락으로 사용하여 응답의 정확성과 효과성을 높입니다.

- **Technical Details**: Wiki-LLaVA 모델은 외부 지식 기반 문서의 계층적 검색 접근법을 사용합니다. 이는 MLLM이 복잡한 질문에 대한 대답을 생성할 때 주어진 이미지 콘텐츠와 사전에 훈련된 지식만으로는 부족한 경우 활용됩니다. 모델은 시각 인코더(visual encoder), 지식 기반(knowledge base, 예: Wikipedia), 그리고 계층적 검색 모듈(hierarchical retrieval module) 세 부분으로 구성됩니다. 이 계층적 접근은 외부 지식원으로부터 관련 문서와 패시지를 검색하여 추가 맥락으로 사용됩니다.

- **Performance Highlights**: Wiki-LLaVA는 시각적 질문 응답(visual question answering)을 위한 데이터셋에서 광범위한 실험을 진행했습니다. 이 모델은 외부에서 정보를 검색(retrieval)하고 이를 모델 디자인에 효과적으로 통합함으로써 MLLM의 대답 생성 성능을 향상시킨다는 점이 입증되었습니다. 실험 결과는 다른 최근 MLLMs 모델과 비교하여 우수한 성능을 보여주며, 외부 소스에서의 정보 검색과 모델 설계 선택의 적절성을 보여줍니다.



### Using Large Language Models to Enrich the Documentation of Datasets for  Machine Learning (https://arxiv.org/abs/2404.15320)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Model, LLM)과 프롬프트 전략을 사용하여 데이터셋 문서에서 중요한 차원을 자동으로 추출하고 이를 통해 데이터셋 설명을 향상시키는 접근 방식을 제안합니다. 이를 통해 데이터 게시자와 실무자들이 자신들의 데이터셋의 발견 가능성(Discoverability)을 향상시키고, 현재 AI 규제에 대한 준수를 평가하며, 특정 ML (Machine Learning) 애플리케이션에 대한 데이터셋의 적합성을 평가하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 제안된 메서드는 데이터셋 문서의 선행 처리(Preprocessing)를 거친 후, 특정 차원별로 구성된 프롬프트 체인을 사용하여 LLM에 입력합니다. 다양한 프롬프트 전략을 사용하여 프롬프트를 보강하고, 제공된 문서만을 기반으로 요구되는 차원을 추출할 수 있습니다. 사용된 두 가지 LLM은 GPT3.5와 Flan-UL2이며, 이들은 Nature의 Scientific Data와 Elsevier의 Data in Brief에 게재된 과학 데이터셋 논문을 대상으로 성능을 평가받았습니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT3.5는 Flan-UL2보다 더 높은 정확도(81.21%)를 기록했으나, 일부 환상 생성(Hallucination) 문제가 더 많이 발생했습니다. 프롬프트 추출 전략은 데이터셋 문서에서 다양한 차원을 성공적으로 추출하는 데 효과적으로 작동했고, 전반적으로 높은 정확성을 보여주었습니다. 이는 AI 데이터 관리 및 규제 준수를 위한 중요한 진전을 나타냅니다.




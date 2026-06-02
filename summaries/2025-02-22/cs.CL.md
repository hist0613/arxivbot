New uploads on arXiv(cs.CL)

### LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention (https://arxiv.org/abs/2502.14866)
Comments:
          Accepted by MLSys 2025. Code available at: this https URL

- **What's New**: 이번 논문에서는 LServe라는 효율적인 시스템을 소개합니다. LServe는 하이브리드 희소_attention을 활용하여 긴 시퀀스 모델의 처리 속도를 높입니다. 이 방법은 프리필링(pre-filling)과 디코딩(decoding) 단계에서 하드웨어 친화적인 다양한 구조의 희소 패턴을 통합하여, 덜 중요한 토큰에 대한 연산을 차단(block-wise)하여 컴퓨팅을 최적화합니다. LServe는 정적(static) 및 동적(dynamic) 희소성의 호환성을 보여주며, 각 최적화를 결합하여 배수의 속도 향상을 가능하게 합니다.

- **Technical Details**: LServe의 핵심은 토큰의 중요성에 따라 블록 레벨의 희소성을 활용하여 KV 히스토리를 처리하는 방식입니다. 이 시스템은 프리필링 단계와 디코딩 단계에서 각각 50%의 attention heads를 거의 무료로 사용할 수 있도록 변환합니다. 또한, KV 페이지 선택 정책을 설계하여 쿼리 중심 유사성을 기반으로 KV 페이지를 동적으로 잘라내는 방법도 채택하였고, 이를 통해 인풋의 길이에 상관없이 일정한 수의 KV 페이지만으로 긴 시퀀스를 유지할 수 있습니다.

- **Performance Highlights**: LServe는 세 가지 긴 시퀀스 LLM(Llama-3-8B, Minitron-4B, Llama-2-7B)에서 벤치마크 테스트를 수행하였으며, 최고 512k 토큰의 컨텍스트 길이를 지원합니다. vLLM 등 최신 기술들과 비교했을 때, LServe는 프리필링 단계에서 최대 2.9배, 디코딩 단계에서 평균 1.3배에서 2.1배의 속도 향상을 기록했습니다. 이러한 성능 향상은 기존의 밀집 밀리 초 모델의 긴 컨텍스트 처리 능력을 유지하면서 이루어졌습니다.



### Interpretable Text Embeddings and Text Similarity Explanation: A Primer (https://arxiv.org/abs/2502.14862)
- **What's New**: 이번 논문에서는 텍스트 임베딩(text embedding) 및 이와 관련된 유사성 점수를 해석하기 위한 방법론의 구조적 개요를 제공합니다. 유사성 점수의 해석 가능성 문제는 AI와 NLP 시스템에서 투명성이 요구되는 환경에서 중요한 과제로, 이를 해결하기 위한 최신 연구 동향을 다루고 있습니다. 다양한 텍스트 임베딩 모델들이 갖는 해석 가능성을 강화하기 위해 개별 방법들의 아이디어와 기술을 평가합니다.

- **Technical Details**: 신경망 텍스트 임베딩 및 유사성의 설명 가능성을 연구하며, 이는 기존의 분류 설명 방식과 구별됩니다. 두 입력의 상호작용을 기반으로 유사성이 결정되므로, 이를 위한 전문화된 방법이 필요합니다. 주어진 텍스트 입력을 통해 신경망을 통한 계산 단계가 설명되며, 일반적으로 Siamese 네트워크(서로의 가중치를 공유) 방식을 사용하여 텍스트 임베딩을 생성합니다.

- **Performance Highlights**: 텍스트 임베딩을 통해 얻은 유사성 점수는 문서 간의 유사성을 정량적으로 평가할 수 있는 중요한 척도를 제공합니다. 이러한 점수는 간단한 내적(dot product) 계산을 통해 산출되며, 이러한 방식은 실질적으로 강하게 상관된 코사인 유사성(cosine similarity)을 활용합니다. 논문은 텍스트 임베딩 및 유사성 정의를 통해 서로 다른 유사성 계산 방법을 구분할 수 있는 기초를 마련하고 있습니다.



### Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning (https://arxiv.org/abs/2502.14860)
Comments:
          22 pages, 8 figures, 8 tables

- **What's New**: 본 논문에서는 ALFA라고 불리는 새로운 프레임워크를 제안하여 대규모 언어 모델(LLM)이 불확실한 상황에서 효과적인 질문을 제기하는 것이 가능하도록 한다. ALFA는 '좋은' 질문의 개념을 이론 기반의 속성 집합으로 분해하고, 속성 특화된 질문 변형을 제어 가능하게 합성하며, 선호 기반 최적화를 통해 모델을 수렴시킨다. 이 방식은 임상적 추론을 사례 연구로 하여, 17,000개의 실제 임상 상호작용과 80,000개의 속성 특화 선호 쌍으로 구성된 MediQ-AskDocs 데이터셋을 도입한다.

- **Technical Details**: ALFA 프레임워크는 질문을 '좋은' 질문으로 지정하는 복잡한 목표를 구조화된 속성으로 분해하는 것에 중점을 둔다. 각 속성(A1, A2, ..., Ak)에 맞춘 보상이 보다 측정 가능하도록 하여, LLM이 효과적으로 질문을 제기할 수 있도록 훈련한다. ALFA는 실험적으로 302개의 전문가 주석이 달린 임상 상호작용 시나리오를 활용하여 LLM의 질문 제기 능력을 평가할 수 있는 새로운 헬스케어 QA 작업을 소개한다.

- **Performance Highlights**: ALFA를 적용한 모델은 MediQ-AskDocs에서 64.4%의 질문 수준 승률과 56.6%의 진단 오류 감소를 달성하며, SOTA 지침 조정 LLM과 비교하여 우수한 성능을 보인다. 이 연구 결과는 구조화된 세분화된 속성을 이용해 질문 제기 방식을 안내함으로써 LLM의 성능을 크게 향상시킬 수 있는 가능성을 보여준다. 특히, 전문가 분야에서의 정보 수집을 체계적으로 촉진하는 데 필수적인 기법으로 자리잡을 수 있다.



### FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling (https://arxiv.org/abs/2502.14856)
- **What's New**: FR-Spec은 대형 언어 모델(LLM)에서 생기는 발생 속도 저하를 해결하기 위한 새로운 방법이다. 이 프레임워크는 단일 레이어와 언어 모델 헤드를 사용하는 기존 방법(EAGLE-2 등)에 비해 75%의 계산 오버헤드를 줄이면서도 최종 출력 분포의 동등성을 유지한다. FR-Spec은 고빈도 토큰으로 구성된 서브셋으로 드래프트 후보를 최적화하여 속도를 1.12배 향상시킬 수 있다.

- **Technical Details**: FR-Spec에서는 이론적으로 언어 모델의 드래프트 검색을 고빈도 토큰으로 제한하여, LM 헤드의 계산 비용을 크게 줄인다. 이 방법은 기존의 언어 모델 샘플링 기법과 호환되며, 재훈련이 필요 없는 플러그 앤 플레이 설계를 채택하고 있다. 이를 통해 EAGLE-2와 통합 시 1.12배의 속도 향상을 달성하며, Medusa와 통합 시에는 1.08배의 향상을 기록했다.

- **Performance Highlights**: 실험 결과, FR-Spec은 다양한 데이터셋에서 기존의 최첨단 방법인 EAGLE-2보다 평균 1.12배의 속도 향상을 보였다. 특히, 전통적인 드래프트-검증 메커니즘에 비해 FR-Spec의 효율성은 명백하다. 이러한 성능 향상은 대규모 어휘의 잠재적 문제를 극복하는 데에도 기여하고 있으며, LLM의 활용 가능성을 더욱 넓히고 있다.



### CLIPPER: Compression enables long-context synthetic data generation (https://arxiv.org/abs/2502.14854)
- **What's New**: 이번 연구에서 저자들은 CLIPPER라는 새로운 합성 데이터 생성 방법을 소개합니다. 이 방법은 이야기 주장 검증(narrative claim verification)이라는 복잡한 작업에 맞춤화된 데이터 생성을 위한 압축 기반 접근 방식을 사용합니다. CLIPPER는 책을 직접 사용하는 대신, 전체 장 요약 및 개요를 먼저 압축한 후 이를 바탕으로 복잡한 주장과 연관된 사고 과정을 생성하여 데이터의 품질을 개선합니다.

- **Technical Details**: CLIPPER는 두 단계로 작동하는 데이터 생성 파이프라인으로, 첫 번째 단계에서는 책의 내용을 요약하여 주제를 보존하면서 압축합니다. 그 후, 이 압축된 요약을 바탕으로 LLM을 사용하여 주장과 사고의 연쇄를 생성합니다. 기존의 접근 방식과 비교할 때 CLIPPER는 생성된 주장에서 노이즈를 감소시키고, 진정성(groundedness)을 높이며, 비용 역시 절반으로 줄일 수 있습니다.

- **Performance Highlights**: CLIPPER를 통해 생성된 19,000개의 합성 주장 데이터셋으로 여러 공개 모델을 미세 조정한 결과, 내러티브 주장 검증(Narrative claim verification)에서 76%의 정확도를 달성하는 성과를 냈습니다. 연구자들은 또한 CLIPPER 모델이 더 세밀하고 구체적인 사고 과정을 생성하면서 다른 내러티브 이해 작업에서도 성능 향상을 이루었다고 강조합니다.



### GATE: Graph-based Adaptive Tool Evolution Across Diverse Tasks (https://arxiv.org/abs/2502.14848)
Comments:
          8 pages of main text, 38 pages of appendices

- **What's New**: 이 논문에서는 GATE (Graph-based Adaptive Tool Evolution)라는 새로운 프레임워크를 제안합니다. GATE는 다양한 작업에서 재사용 가능한 도구의 계층적 그래프를 동적으로 구성하고 발전시키는 능력을 갖추고 있습니다. 기존의 도구 제작 방법들은 단일 작업에 국한되거나 효율적으로 신뢰할 수 있는 도구 세트를 구성하는 데 어려움을 겪고 있었으나, GATE는 이러한 문제를 해결합니다.

- **Technical Details**: GATE는 오픈 엔디드 작업(Minecraft), 에이전트 기반 작업(TextCraft, DABench), 코드 생성 작업(MATH, Date, TabMWP)에서 평가되었습니다. 이 프레임워크는 도구의 양과 복잡성, 기능성을 조화롭게 유지하면서 높은 효율성을 보장합니다. GATE는 이를 위해 그래프 기반의 다이나믹한 도구 진화 메커니즘을 채택하고 있습니다.

- **Performance Highlights**: Minecraft에서는 이전의 SOTA(State Of The Art)와 비교하여 4.3배 빠른 이정표 완수 시간을 달성했습니다. 코드 생성 작업에서는 기존 도구 제작 방법에 비해 평균 9.23%의 개선을 보여주었고, 에이전트 작업에서는 10.03%의 성능 향상을 기록했습니다. 이러한 결과는 GATE의 적응적 진화의 힘을 잘 보여줍니다.



### Revealing and Mitigating Over-Attention in Knowledge Editing (https://arxiv.org/abs/2502.14838)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 지식 편집 방법에서 발생하는 특정성 실패(Specificity Failure) 문제를 다룹니다. 연구자들은 LLMs에서 수정을 통해 기존 지식이 손상되는 현상을 발견하고, 이를 Attention Drift라는 현상으로 정의하였습니다. Attention Drift는 모델이 불필요하게 새로운 지식에 집중하여 이전 지식을 왜곡하는 문제입니다. 이를 해결하기 위해, Selective Attention Drift Restriction(SADR)라는 새로운 방법을 제안합니다.

- **Technical Details**: SADR은 지식 편집 과정에서 주의 가중치 분포의 변화를 제한하는 추가 정규화 항을 도입하여 지나치게 수정된 정보에 대한 집중을 방지합니다. 연구에서 사용한 모델은 1.1B에서 20B까지의 다양한 LLM으로, SADR 방법을 적용함으로써 기존 지식의 손상을 크게 줄였습니다. 세 가지 지식 편집 방식(위치-수정, 매개변수 보존, 메타 학습)을 포함하여, SADR은 명백히 훈련된 모델의 예측 성능을 향상시켰습니다. 이 때 원래 지식의 정확도가 절반 이상 감소하는 현상을 줄이는 데 효과적입니다.

- **Performance Highlights**: SADR 방법을 적용한 결과, 다섯 개의 모델에서 130.9%에서 295.8%까지의 정확도 향상을 보였으며, 편집 성공률은 단 0.19% 감소하는 데 그쳤습니다. 이는 SADR이 다양한 편집 작업에서 효과적으로 특정성 실패를 완화할 수 있음을 입증합니다. 이러한 결과는 실세계 응용에서 모델의 신뢰성과 강건성을 높이는 데 기여할 것으로 기대됩니다.



### Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs (https://arxiv.org/abs/2502.14837)
Comments:
          16 pages, 8 figures

- **What's New**: DeepSeek에서 제안한 Multi-head Latent Attention (MLA) 아키텍처는 Key-Value (KV) 캐시를 압축하여 효율적인 추론을 가능하게 하고 있습니다. 기존의 Multi-Head Attention (MHA) 및 그 변형과 비교하여 MHA2MLA로의 전환을 통해 데이터 효율적인 파인 튜닝을 가능하게 하는 최초의 방법이 제시되었습니다. 이 접근법은 RoPE를 부분적으로 제거하고 저랭크 근사를 도입하여 성능을 회복합니다.

- **Technical Details**: MLA 구조는 수많은 파라미터를 재사용하면서 MHA에서 MLA로의 전환을 용이하게 합니다. 핵심 기술로는 부분 회전 위치 임베딩(partial RoPE)과 저랭크 근사(low-rank approximation)가 있으며, 이는 KV 캐시와 추론 프로세스의 저장 방식을 MLA에 맞춰 조정합니다. 이 방법은 기존 MHA의 학습된 파라미터를 최대한 활용하기 위한 설계를 포함하고 있습니다.

- **Performance Highlights**: MHA2MLA는 훈련 데이터의 0.3%에서 0.6%만으로도 성능을 복구하는 데 성공하였습니다. KV 캐시 크기를 92.19% 줄이며 LongBench 성능은 단 0.5% 하락에 그쳤습니다. 실험에서는 다양한 모델 크기에 대해 MHA2MLA의 효과를 검증하여 효율적인 추론을 위한 통찰을 제공합니다.



### Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs (https://arxiv.org/abs/2502.14830)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 중간 계층이 다양한 언어 간의 정렬에 가장 강력한 잠재력을 가지고 있다는 발견을 기반으로, 중간 계층 정렬 목표(middle-layer alignment objective)를 제안합니다. 이를 통해 언어 자원이 적은 언어에서의 작업의 성능 향상을 목표로 하며, 기존 작업 특정 훈련(task-specific training)과 결합하여 실험적인 성과를 거두었습니다.

- **Technical Details**: 중간 계층에서의 정렬 목표를 적용하는 방법은 기계번역, 슬롯 채우기(slot filling) 및 구조화된 텍스트 생성(structured text generation)와 같은 다양한 작업에서 사용됩니다. 이 방법은 한국어 같은 언어 자원이 적은 언어뿐만 아니라 미정렬 언어(unseen languages)에도 일반화될 수 있습니다. 또한, 별도로 훈련된 정렬 모듈은 기존의 작업 특정 모듈과 결합할 수 있어 전체 재훈련 없이도 크로스-링구얼 기능을 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, 중간 계층에서의 정렬 목표를 적용하는 것이 LLM의 작업 특정 훈련 동안 크로스-링구얼 전이(cross-lingual transfer)를 개선한다는 것을 보여주었습니다. 또한, 이러한 성과가 정렬에 사용된 언어의 선택에 강인하며, 다양한 언어 쌍 간의 전이 성능에도 긍정적인 영향을 미친다는 것을 확인했습니다. 마지막으로, 새로운 훈련 전략이 LLM의 크로스-링구얼 능력을 크게 향상시킬 가능성을 제시합니다.



### Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps (https://arxiv.org/abs/2502.14829)
- **What's New**: 본 연구는 언어 모델(LM)의 chain of thought (CoT) 생성 과정에서의 매개변수 충실도를 측정할 수 있는 새로운 프레임워크인 Parametric Faithfulness Framework (pff)를 소개합니다. 특히, reasoning 단계를 잊게 하는 방식인 Faithfulness by Unlearning Reasoning steps (FUR)를 제안하여 CoT의 충실도를 평가합니다. FUR는 모델의 파라미터에서 CoT에 인코딩된 정보를 지우고, 이러한 과정을 통해 예측 결과에 미치는 영향을 분석합니다.

- **Technical Details**: FUR는 NPO(Preference Optimization) 방법을介해 CoT의 각 단계를 독립적으로 분할하여 해당 단계들에서 인코딩된 지식을 모델 파라미터에서 잊게 하며, 이로 인해 예측에 미치는 효과를 측정합니다. 여기서 ff-hard와 ff-soft라는 두 가지 지표를 사용하여 CoT의 충실도를 평가합니다. ff-hard는 전체 CoT의 신뢰성을 정량화하고, ff-soft는 CoT 내에서 가장 두드러진 reasoning 단계를 식별합니다.

- **Performance Highlights**: 연구 결과, FUR는 CoT가 주요 단계를 잃어버렸을 때 모델의 예측 결과를 유의미하게 변화시킬 수 있음을 보여줍니다. 이는 CoT가 해당 모델의 진정한 reasoning 과정을 충실히 설명한다는 것을 나타냅니다. 또한, 인간 평가에서 FUR에 의해 식별된 중요한 단계들이 plausibility와 일치하지 않음을 발견하여, CoT가 신뢰성과 신뢰성을 동시에 만족시키기 위한 전문적인 정렬이 필요함을 강조합니다.



### eC-Tab2Text: Aspect-Based Text Generation from e-Commerce Product Tables (https://arxiv.org/abs/2502.14820)
Comments:
          NAACL 2025 (Industry Track)

- **What's New**: 이 논문에서는 eC-Tab2Text라는 혁신적인 데이터세트를 소개합니다. 이 데이터세트는 전자상거래(e-commerce) 도메인에서 제품 속성과 사용자 쿼리를 포함하는 다양한 요소를 포착하기 위해 설계되었습니다. 이를 통해 LLMs(대규모 언어 모델)가 구조화된 표(tabular data)를 기반으로 고품질의 제품 리뷰를 생성할 수 있도록 지원합니다.

- **Technical Details**: eC-Tab2Text는 전자상거래 제품 테이블과 사용자 별 쿼리, 출력이 연계된 속성 중심의 데이터세트를 제공합니다. 기존의 일반 목적의 데이터세트와는 달리, 이 데이터세트는 전자상거래에서의 고유한 요구 사항에 맞춰 설계되었습니다. 이를 통해 LLM을 미세 조정하고, 표준 Table2Text 메트릭을 활용하여 검증함으로써 성능을 평가합니다.

- **Performance Highlights**: 결과적으로, 모델의 수정을 통해 생성된 제품 리뷰의 맥락적 정확성이 크게 향상된 것으로 나타났습니다. eC-Tab2Text 데이터세트는 LLMs의 성능을 극대화하고, 전자상거래 작업흐름을 최적화하는 데 중요한 역할을 하고 있습니다. 이는 고객만족도와 비즈니스 결과를 높이는 데 기여할 것으로 기대됩니다.



### From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (https://arxiv.org/abs/2502.14802)
Comments:
          Code and data to be released at: this https URL

- **What's New**: 이번 연구는 새로운 RAG 접근법인 HippoRAG 2를 제안하여, 사실적 기억(factual memory), 의사결정(sense-making), 그리고 연관성 기억(associative memory) 과제에서 표준 RAG보다 우수한 성능을 보여줍니다. HippoRAG 2는 Personalized PageRank 알고리즘을 든든히 지탱하며, LLM의 온라인 사용을 더욱 효과적으로 통합하여 인간의 장기 기억에 가까운 효과를 구현합니다. 이를 통해 단순한 기억 과제에서도 강화된 성능을 가져오며, 비모수적 지속 학습을 위한 새로운 길을 제시합니다.

- **Technical Details**: HippoRAG 2는 기존 HippoRAG의 구조를 바탕으로 하여, 더 깊은 구문 통합(deeper passage integration)과 LLM 사용의 효과성을 향상시킵니다. Personalized PageRank를 기반으로 하여 쿼리에 기반한 맥락화(contextualization) 부족을 보완하기 위해 KG(Knowledge Graph) 삼중(triples) 선택 과정에 더 깊이 참여하게 만들어, LLM의 효율성을 극대화합니다. 이러한 요소들이 긴 이야기와 같은 복잡한 마케팅을 이해할 수 있는 능력을 크게 향상시킵니다.

- **Performance Highlights**: HippoRAG 2는 관계성 관련 작업에서 표준 RAG보다 평균 7777 포인트 개선된 성과를 보였으며, 사실 기억 및 의사결정 작업에서 성능 저하 없이 소폭의 향상을 기록하였습니다. 이 시스템은 다양한 리트리버와 강력한 공개형 및 상용 LLM에서도 견고성을 보이며 적용 유연성을 극대화합니다. 이러한 성과는 HippoRAG 2가 인간과 유사한 비모수적 지속 학습 시스템으로 발전할 수 있는 가능성을 제시합니다.



### Rapid Word Learning Through Meta In-Context Learning (https://arxiv.org/abs/2502.14791)
- **What's New**: 이번 연구에서 제안된 Minnow(Meta-training for IN-context learNing Of Words) 방법은 언어 모델이 몇 개의 예시를 기반으로 새로운 단어의 사용법을 생성하도록 훈련할 수 있게 합니다. 특히, 이 방법은 새로운 단어를 나타내기 위해 특별한 placeholder token을 사용하며, 이러한 과정이 반복될수록 모델의 일반적인 단어 학습 능력이 향상된다고 주장합니다. 이 연구는 아동 언어 습득에 맞춘 데이터셋을 사용하여, 소량의 데이터에서도 변별력 있는 단어 학습이 가능함을 보여줍니다.

- **Technical Details**: Minnow 방법은 언어 모델이 새로운 단어의 맥락을 통해 빠르게 학습할 수 있도록 meta-training을 이용합니다. 이 방법은 autoregressive 언어 모델을 한 단계에서 훈련하여 새로운 단어의 예시를 생성하도록 하며, 각 단어에 대한 사용 예시가 없어도 일반화할 수 있는 능력을 개발합니다. 또한, 이 연구는 기존의 LLM(large language model)이 아니라 아동의 언어 입력 데이터를 기반으로 모델을 훈련함으로써, 데이터 효율성을 극대화하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 연구 결과, Minnow의 성능은 기존의 대규모 언어 모델과 유사한 수준으로, 소량의 예시를 활용하여 새로운 단어를 효율적으로 분별하고 정의할 수 있는 능력이 향상되었습니다. 특히, Llama-3 8B 모델을 Minnow로 미세 조정(finetuning)한 결과, 새로운 단어의 문법적 범주를 식별하고, 관련 있는 정의 및 사용 예시를 생성하는 능력이 크게 향상된 것을 확인했습니다. 이 접근 방식은 언어 모델의 단어 학습 능력을 개선하는 잠재력을 보여줍니다.



### ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting (https://arxiv.org/abs/2502.14780)
Comments:
          12 pages, 7 figures, 3 tables

- **What's New**: 이 논문에서는 AR(증강현실), VR(가상현실) 및 최신 스마트폰 상에서의 효율적이고 프라이버시를 보호하는 다중모드(interaction)에 대한 새로운 접근법인 Visual Instruction Rewriting을 제안합니다. 기존의 대규모 비전-언어 모델(VLM)은 클라우드 기반 처리를 의존하여 시각 데이터의 프라이버시 문제와 실시간 처리의 어려움을 초래했습니다. 본 연구는 39,000개의 예제와 14개 도메인으로 구성된 데이터셋을 제시하며, 이를 통해 프라이버시를 보장하면서 다중모드를 처리할 수 있는 경량화된 VLM(250M parameters)을 개발했습니다.

- **Technical Details**: Visual Instruction Rewriting은 다중모드 지시사항을 텍스트만 포함된 명령으로 변환합니다. 이 접근법은 프라이버시를 보호하면서도 사용자가 기기에서 직접 명령을 실행할 수 있도록 합니다. 본 연구는 여러 NLG(자연어 생성) 메트릭(BLEU, METEOR, ROUGE)을 이용하여 제안한 모델의 성능을 평가하였으며, 8비트 양자화된 모델은 500MB 미만의 저장 공간에서 효과적인 지시 사항 변환을 수행했습니다.

- **Performance Highlights**: 우리는 경량화된 250M 파라미터 모델이 기존의 베이스라인인 PaliGemma-v2 및 Qwen2VL에 비해 제로샷 설정에서 더 나은 성능을 보였음을 발견했습니다. 이 모델은 사용자 요청을 구조화된 텍스트로 변환할 수 있는 수용 가능한 수준의 rewriting 능력을 가지며, 이는 AR, VR 및 스마트폰 인터페이스와의 안전하고 실시간 상호작용을 가능하게 합니다.



### Harnessing PDF Data for Improving Japanese Large Multimodal Models (https://arxiv.org/abs/2502.14778)
Comments:
          15 pages, 8 figures

- **What's New**: 본 연구에서는 일본어 LMM(Large Multimodal Models)의 성능을 향상시키기 위해 PDF 데이터의 잠재력을 탐색합니다. 기존의 일본어 LMM은 영어로 번역된 데이터에 의존해 일본 특유의 문화적 지식을 캡처하는 데 한계가 있었습니다. 이를 해결하기 위해, PDF에서 이미지-텍스트 쌍을 추출하는 완전 자동화된 파이프라인을 소개하며, 외부 주석 작업 없이 데이터를 확보할 수 있는 방법을 모색했습니다.

- **Technical Details**: 우리는 사전 훈련된 모델을 활용하여 PDF에서 이미지-텍스트 쌍을 추출하는 방법론을 개발했습니다. 이 과정에서는 레이아웃 분석(layout analysis), OCR(Optical Character Recognition), 및 비전-언어 페어링(vision-language pairing) 기술이 포함됩니다. 이러한 자동화된 시스템을 통해 일본어 LMM의 훈련 데이터를 풍부하게 하기 위한 지침 데이터도 생성됩니다.

- **Performance Highlights**: PDF에서 유래된 데이터로 훈련한 일본어 LMM은 Heron-Bench에서 3.9%에서 13.8%의 성능 향상을 보였습니다. 다양한 실험을 통해 PDF 데이터의 효과와 모델 크기와의 관계를 분석하였으며, 이미지-텍스트 쌍과 이미지만으로 생성된 지침 데이터의 효과성도 평가했습니다. 이러한 연구 결과는 일본어 LMM의 훈련에 PDF 데이터를 활용하는 가치를 확고히 해줍니다.



### SurveyX: Academic Survey Automation via Large Language Models (https://arxiv.org/abs/2502.14776)
Comments:
          15 pages, 16 figures

- **What's New**: 본 논문에서는 SurveyX라는 새로운 자동 설문 생성 시스템을 제안합니다. SurveyX는 설문 작성 과정을 두 개의 단계, 즉 준비(Preparation)와 생성(Generation) 단계로 나누어 효율적으로 구성합니다. 또한, 온라인 참조 검색과 AttributeTree라는 사전 처리 방법을 혁신적으로 도입하여 설문 구성의 효율성을 크게 향상시킵니다.

- **Technical Details**: SurveyX는 인터넷에서 관련된 자료를 검색하고 필터링하는 알고리즘을 개발하여 설문 주제에 적합한 높은 품질의 참고 문헌을 수집합니다. 이 시스템은 각 문서로부터 핵심 정보를 추출하는 AttributeTree라는 사전 처리 방법을 사용하여, 참조 자료 데이터베이스를 구축하고 Retrieval Augmented Generation (RAG) 기술을 통해 효율적으로 자료를 검색합니다. 생성 단계에서는 수집된 정보를 바탕으로 설문조사의 아웃라인과 본문을 단계적으로 생성하며, 표와 그림을 포함하여 가독성을 높입니다.

- **Performance Highlights**: 실험 결과, SurveyX는 기존의 자동 설문 생성 시스템보다 콘텐츠 품질에서 0.259 향상, 인용 품질에서 1.76 향상을 보이며, 여러 평가 차원에서 인간 전문가의 성능에 근접하는 성과를 나타냅니다. 이러한 성과는 SurveyX의 평가 프레임워크가 추가된 평가 지표를 통해 후속 연구를 효과적으로 지원한다는 점에서도 noteworthy합니다.



### Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning (https://arxiv.org/abs/2502.14768)
- **What's New**: 이 논문에서는 DeepSeek-R1에서 성공적으로 사용된 규칙 기반 강화 학습(RL)의 잠재력을 대규모 추론 모델에 탐구합니다. 주목할 만한 점은 5,000개의 논리 문제만으로도 모델이 복잡한 수학 벤치마크인 AIME와 AMC에서 일반화 능력을 발휘하였다는 것입니다. 이 연구는 기존 연구의 재현 가능성을 높이고, 소규모 모델에서도 비슷한 추론 능력이 발휘될 수 있는지를 동적으로 검증하고 있습니다.

- **Technical Details**: 제안된 Logic-RL 프레임워크는 REINFORCE++ 알고리즘을 채택하고 DeepSeek-R1의 보상 설계를 기반으로 하여 훈련됩니다. 강화 학습 과정에서 모델은 더 많은 훈련 단계를 할당하여 더 깊이 있는 사고 과정을 탐색하게 됩니다. 또한 <think>와 <answer> 태그를 포함한 구조화된 응답 포맷을 통해 보상을 정확하게 검증할 수 있게 되어, 강화학습 접근 방식에서 아키텍처의 전반적인 개선을 이끌어냅니다.

- **Performance Highlights**: 7B 모델은 AIME에서 125%, AMC에서 38% 향상을 보였으며, 이는 모델이 추상적인 문제 해결 체계를 발전시킨다는 것을 보여줍니다. 강화 학습이 훈련 데이터 구조에 최소한으로 의존하는 방식으로 자연스럽게 일반화를 성공적으로 이루어내며, 더 긴 응답이 항상 더 나은 추론을 보장하지 않는다는 흥미로운 발견도 있었습니다.



### Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis (https://arxiv.org/abs/2502.14767)
Comments:
          Code available at: this https URL

- **What's New**: 이 논문은 Tree-of-Debate (ToD)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 과학 논문을 각각의 LLM(대형 언어 모델) 페르소나로 변환하여 그들의 혁신성을 두고 토론하게 합니다. ToD는 구체적이고 비판적인 추론을 강조하며, 정치적이고 학문적인 논의를 촉진합니다.

- **Technical Details**: ToD는 복잡한 논리적 과제를 해결하기 위해 다양한 관점과 추론 경로를 탐색하는 다중 에이전트 LLM 토론을 활용합니다. 논문 간의 독립적인 혁신성 주장을 구분할 수 있는 비계층적 대화 구조를 동적으로 구성하며, 각 논점의 세부사항을 분석할 수 있게 합니다. 또한, 이 프레임워크는 반복적인 검색 과정을 도입하여 논의의 진전과 관련된 콘텐츠를 효과적으로 활용합니다.

- **Performance Highlights**: 다양한 도메인의 과학 문헌에 대한 실험을 통해 ToD는 정보가 풍부한 주장을 생성하고 논문 간의 대조를 효과적으로 수행했습니다. 이는 연구자들이 문헌 리뷰를 지원하는 데 큰 도움이 됩니다. ToD는 또한 세밀한 비교 요약을 생성할 수 있는 능력을 입증했습니다.



### Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning (https://arxiv.org/abs/2502.14765)
Comments:
          Accepted to NAACL 2025 (Main)

- **What's New**: 이번 연구는 단계별(step-by-step) 사실 검증 시스템을 통해 세 가지 의료 관련 사실 확인 데이터셋에 적용함으로써 기존 접근보다 개선된 성능을 입증하였습니다. 최근의 방법들은 대규모 언어 모델(LLMs)의 다단계 질문-응답 방식을 활용하여 점진적으로 정보를 수집하고 이에 따라 판단을 내리도록 설계되었습니다. 이러한 새로운 접근법은 전통적인 FV 방법의 한계를 해결하고, 특히 도메인 특정(claim-specific) 주장에 대한 검증의 잠재력을 보여줍니다.

- **Technical Details**: 전통적인 사실 검증 파이프라인은 문서 검색(document retrieval), 증거 추출(evidence extraction), 그리고 판별 예측(verdict prediction)으로 이루어집니다. 연구에서는 DeBERTa 모델을 활용하여 NLI(자연어 추론) 작업을 통해 주장과 증거의 관계를 예측하였습니다. 반대로 제안된 단계별 LLM 시스템은 증거를 수집하고자 추가 질문을 생성하며, 온라인 검색 엔진을 통해 정보를 검색하고, 논리적 추론을 활용하여 근거를 정리합니다.

- **Performance Highlights**: 연구 결과, 기존의 전통적인 접근 방식에 비해 개선된 최종 성능을 발휘하였으며, 다양한 LLMs와 외부 웹 검색, 논리 프레디케이트를 활용한 구조적 추론에 대한 여러 설정에서 평가되었습니다. 이 단계별 시스템은 복잡한 주장을 효과적으로 검증할 수 있는 능력을 보여줍니다. 연구진은 GitHub을 통해 모든 데이터와 코드를 공개하여, 이후 연구자들이 이 시스템을 활용할 수 있도록 지원하고 있습니다.



### On the Influence of Context Size and Model Choice in Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2502.14759)
Comments:
          Accepted to Findings of NAACL 2025

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 다양한 구성 요소를 체계적으로 평가합니다. 특히 제공된 컨텍스트의 이상적인 크기, 기본 LLM 선택, 및 검색 방법의 차이를 분석하여 RAG 시스템의 견고한 개발을 위한 지침을 제시하고자 합니다. 연구는 짧은 답변을 다루는 기존의 RAG 평가 방식에서 벗어나, 장기형 질문 응답(Long-form QA)을 다루어 더 복잡한 문제를 탐구합니다.

- **Technical Details**: RAG 시스템은 크게 리트리버(retriever)와 리더(reader)로 이루어져 있습니다. 본 연구에서는 컨텍스트 크기가 QA 성능에 미치는 영향, 서로 다른 LLM의 선택에 따른 중요성, 그리고 두 가지 리트리버(BM25 및 세맨틱 검색)가 최종 QA 성능에 미치는 영향을 연구합니다. 이를 통해 컨텍스트의 크기와 리트리버의 선택이 RAG 시스템의 성능에 미치는 영향을 비교하고 분석합니다.

- **Performance Highlights**: 연구 결과, 최종 QA 성능은 15개의 컨텍스트 스니펫까지는 점진적으로 향상되지만, 그 이상으로는 정체되거나 하락하는 경향을 보입니다. 생물의학 도메인에서는 Mistral과 Qwen이 가장 뛰어난 성능을 보인 반면, 백과사전 도메인에서는 GPT와 Llama가 두각을 나타냈습니다. 오픈 도메인 환경에서는 성능이 Gold 기준과 멀어지는 경향을 보였으며, BM25는 정밀도 향상에 기여하지만 세맨틱 검색은 더 넓은 정보 범위를 제공하는 것으로 나타났습니다.



### TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators (https://arxiv.org/abs/2502.14752)
- **What's New**: 이번 연구에서 TritonBench라는 새로운 벤치마크를 소개합니다. TritonBench는 Triton 오퍼레이터 생성을 위한 최초의 포괄적인 기준으로, GitHub에서 수집한 184개의 실제 오퍼레이터와 PyTorch 인터페이스에 맞춰진 오퍼레이터 세트 두 가지를 포함합니다. 이 벤치마크는 기능적 정확성뿐만 아니라 NVIDIA GPU에서의 효율성 성능도 프로파일링하여 산업적 수요에 더욱 부합하는 평가를 제공합니다.

- **Technical Details**: TritonBench-G는 GitHub에서 수집된 고품질의 Triton 오퍼레이터를 활용하여 구성된 반면, TritonBench-T는 PyTorch와의 호환성 있는 오퍼레이터 개발 작업으로 구성됩니다. 두 채널 모두 성능 평가를 위해 유사성, 호출 및 실행 정확도, 속도 향상 및 GPU 효율성 등의 메트릭을 사용합니다. 연구 결과, 현재의 코드 LLM들이 효율적인 Triton 오퍼레이터 생성을 위한 능력이 부족함을 밝혀냈습니다.

- **Performance Highlights**: TritonBench-G에서의 최고 실행 정확도는 23.91%에 도달했으며, TritonBench-T에서는 53.01%의 최고 정확도를 기록했습니다. 각 오퍼레이터의 최상의 속도 향상은 TritonBench-G에서 1.56배, TritonBench-T에서 1.91배입니다. 이러한 결과는 현재 LLM들이 TritonBench를 효과적으로 처리하지 못하고 있음을 강조하며, LLM 기반의 오퍼레이터 개발 분야의 발전을 위한 기초 평가가 필요함을 시사합니다.



### Large Language Models Struggle to Describe the Haystack without Human Help: Human-in-the-loop Evaluation of LLMs (https://arxiv.org/abs/2502.14748)
Comments:
          21 Pages. LLM for Data Exploration and content analysis

- **What's New**: 이번 연구는 전통적인 topic 모델에서 대규모 언어 모델(LLM)로의 전환이 실제 문서 이해에 미치는 효과를 평가합니다. 두 개의 데이터셋에서 비지도 학습과 지도 학습 방식의 LLM 기반 접근 방식을 비교하여, LLM이 생성한 주제가 인간이 이해하기 더 쉽지만 도메인 특정 데이터셋에 대해서는 지나치게 일반적이라는 것을 밝혔습니다. 이는 LLM이 문서에 대해 깊이 있는 이해를 하는 데 한계가 있다는 것을 시사합니다.

- **Technical Details**: 연구에서는 대규모 언어 모델을 사용하여 문서 집합을 탐색하는 방법을 평가하고, LLM과 전통적 주제 모델의 성능을 비교했습니다. 주목할 점은 LLM 기반 접근법이 데이터 탐색에서 더 높은 평균 승률을 보였지만, 도메인 특정 데이터에서generic topics를 생성해 사용자에게 유용한 정보를 제공하지 못했다는 것입니다. 인간 감독을 통해 이러한 문제를 완화할 수 있지만, 이는 더 많은 인적 자원을 요구합니다.

- **Performance Highlights**: LLM 기반 방법은 전통적 모델보다 데이터를 더 잘 탐색할 수 있는 경향이 있지만, 사용자는 여전히 도메인 특정 데이터에 대해서는 도전 과제를 느끼고 있습니다. 전통적 모델인 Latent Dirichlet Allocation(LDA)은 여전히 유효하지만 사용자 친화적인 면에서는 부족한 부분이 있음을 보여주었습니다. 최종적으로, LLM은 대규모 코퍼스를 독립적으로 설명하는 데 어려움을 겪으며, 특히 도메인 특정 데이터에서 더욱 두드러집니다.



### HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States (https://arxiv.org/abs/2502.14744)
- **What's New**: 이 논문은 기존의 언어 모델들이 적절한 안전 메커니즘을 갖추지 못한 것에 착안하여, 대형 비전-언어 모델(LVLMs)이 안전성과 관련된 신호를 내부 활성화에서 본질적으로 인코딩하고 있는지를 탐구합니다. 연구 결과 LVLM은 불안전한 입력을 처리할 때 고유한 활성화 패턴을 나타내며, 이를 통해 적대적 입력을 탐지하고 완화할 수 있는 기회를 제공합니다.

- **Technical Details**: HiddenDetect라는 새로운 프레임워크를 제안하며, 이는 모델의 내부 활성화를 모니터링하여 불안전한 프롬프트를 식별하는 방식입니다. 이 접근법은 fine-tuning 없이도 안전성을 높일 수 있는 가능성을 보여주며, 각 모델의 숨겨진 상태를 기반으로 안전 신호를 탐지합니다. Refusal Vector(RV)를 사용하여 프롬프트의 안전성을 평가하고, 각 레이어에서 발생하는 활동을 통해 불안전한 입력을 flagged 합니다.

- **Performance Highlights**: 실험 결과 HiddenDetect는 기존의 안전 방어 체계를 초월하여 'jailbreak' 공격을 효과적으로 감지하는 것으로 나타났습니다. 이 연구는 LVLM의 안전성을 기존의 행동 기반 접근에서 활성화 기반 접근으로 전환하는 것을 제안하며, 이로 인해 안전성을 유지하면서도 다양한 적대적 위협을 다룰 수 있는 방법을 제시합니다.



### SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines (https://arxiv.org/abs/2502.14739)
- **What's New**: 이번 논문에서는 SuperGPQA라는 포괄적인 벤치마크를 소개합니다. 이 벤치마크는 285개의 전문 분야에서 대학원 수준의 지식과 추론 능력을 평가하며, 기존의 평가 방법이 부족했던 특정 분야에 대한 LLM의 능력을 진단합니다. 특히, 경량 산업, 농업 및 서비스 중심의 분야에서 LLM의 성과를 측정하려고 하였습니다.

- **Technical Details**: SuperGPQA는 Human-LLM collaborative filtering 메커니즘을 활용하여 사소하거나 모호한 질문을 반복적인 정제를 통해 제거합니다. 이러한 과정은 LLM의 응답과 전문가의 피드백을 기반으로 이루어집니다. 연구 결과, 현재 최고의 성능을 가진 LLM인 DeepSeek-R1이 SuperGPQA에서 61.82%의 정확도를 달성하였으며, 이는 인공지능 일반 지능(AGI)과의 큰 격차를 시사합니다.

- **Performance Highlights**: 대규모 주석 프로세스의 관리에서 많은 통찰을 제공하며, 80명 이상의 전문가 주석자가 참여한 상호작용적인 Human-LLM 협력 시스템을 통해 이루어졌습니다. 이 과정은 유사한 연구의 미래 이니셔티브에 대한 귀중한 방법론적 지침을 제공합니다. SuperGPQA는 다양한 지식 영역에서 현재 LLM의 성능 향상을 위한 큰 잠재력을 드러냅니다.



### Sentence Smith: Formally Controllable Text Transformation and its Application to Evaluation of Text Embedding Models (https://arxiv.org/abs/2502.14734)
- **What's New**: 이 논문에서는 Sentence Smith라는 새로운 프레임워크를 제안하여 텍스트 의미를 제어되고 명시적으로 조작하는 방법을 제공합니다. 이 프레임워크는 문장을 의미 그래프로 파싱한 후, 인간이 설계한 의미 조작 규칙을 적용하고, 마지막으로 변형된 그래프에서 텍스트를 생성하는 3단계 과정을 포함합니다. 최종 필터링 단계에서는 적용된 변환의 유효성을 확인하여 결과의 질을 보장합니다.

- **Technical Details**: Sentence Smith는 신경-상징적(neuro-symbolic) 텍스트 조작 프레임워크로, 문장을 접합 그래프 기반의 의미 표현으로 변환하기 위해 파서를 사용합니다. 이 과정에서 의미 표현으로는 Abstract Meaning Representation (AMR)이 사용됩니다. 변형 규칙을 적용하여 업데이트된 그래프를 자연어로 변환하고, 변환된 그래프와 생성된 문장 간의 일관성을 평가하기 위한 신뢰성 검사도 선택적으로 진행됩니다.

- **Performance Highlights**: 이 프레임워크는 텍스트 임베딩 모델을 도전하는 하드 네거티브 쌍을 생성하는 데 활용됩니다. 관찰된 결과는 Sentence Smith가 생성한 문장이 매우 정확하며, 이는 현재 벤치마킹에서 언어적 현상의 불투명성을 해소할 수 있는 장점이 있습니다. 이를 통해 텍스트 임베딩 모델의 특정한 강점과 약점을 평가할 수 있는 더 세밀한 통찰을 제공합니다.



### Entity Framing and Role Portrayal in the News (https://arxiv.org/abs/2502.14718)
Comments:
          23 pages, 12 figures. Submitted to ACL Rolling Review (ARR)

- **What's New**: 이번 연구에서는 뉴스 기사에서 엔티티 프레이밍(entity framing)과 역할 묘사(role portrayal)를 주석(annotation)한 다국어 계층적 코퍼스를 소개합니다. 이 데이터셋은 스토리텔링 요소에서 영감을 받아 22개의 세분화된 역할(archetypes)을 포함한 독창적인 분류 체계를 사용하며, 세 가지 주요 카테고리인 주인공(protagonist), 적대자(antagonist), 무고한(innocent)으로 나뉘어 있습니다. 다국어로 작성된 1,378개의 최근 뉴스 기사와 5,800개의 엔티티 언급이 포함되어 있어, 연구와 뉴스 분석에 보편적인 응용 가능성을 제공합니다.

- **Technical Details**: 이 코퍼스는 불가리아어, 영어, 힌디어, 유럽 포르투갈어, 러시아어의 다섯 가지 언어로 작성된 최신 뉴스 기사들로 구성되어 있습니다. 데이터의 주 초점은 우크라이나-러시아 전쟁 및 기후 변화와 같은 두 가지 글로벌 이슈입니다. 연구팀은 주어진 문서, 문단, 문장에서 미세 조정된 최신 다국어 transformer 모델 및 계층적 제로 샷 학습(hierarchical zero-shot learning)을 활용한 평가 결과를 보고하여, 엔티티 역할 분석에 대한 새로운 도구 개발을 위해 필요한 고품질 주석 데이터의 중요성을 강조합니다.

- **Performance Highlights**:  연구팀은 엔티티 프레이밍 작업에 대한 새로운 계층적 분류 체계를 통해, 엔티티가 텍스트에서 어떻게 서술되는지를 분석하는 방법을 제시합니다. 이 코퍼스는 인상적인 규모와 복잡성을 갖추고 있으며, 엔티티 표현의 언어적 차원과 감정적 차원 분석에 기여할 것으로 기대됩니다. 또한, 기후 변화 및 우크라이나-러시아 전쟁과 같은 다양한 주제를 통해 언론의 보도 방식이 어떻게 대중의 인식과 사회 담론에 영향을 미치는지를 더욱 깊이 이해할 수 있는 기회를 제공합니다.



### Data-Efficient Pretraining with Group-Level Data Influence Modeling (https://arxiv.org/abs/2502.14709)
- **What's New**: 이 논문에서는 그룹 수준에서 훈련 데이터를 구축하는 것이 효과적이라고 주장하며, 이를 통해 데이터 효율성을 높이는 새로운 접근 방식을 제안합니다. 제안된 Group-Level Data Influence Modeling (Group-MATES) 방법은 데이터 포인트를 독립적으로 처리하는 것이 아니라, 집합적으로 다룰 때의 영향을 모델링합니다. 이는 구성된 데이터 집합의 최대 그룹 영향을 찾는 것을 목표로 하며, 이를 통해 훈련 데이터 선택 과정을 개선할 수 있음을 보여줍니다.

- **Technical Details**: Group-MATES는 그룹 수준에서 데이터 유틸리티를 최적화하여 훈련 프로세스를 개선합니다. 이 방법은 사전 훈련 모델을 로컬로 프로빙하여 오락클 그룹 수준의 영향을 수집하고, 이를 기반으로 관계 가중 데이터 영향 모델을 정교화합니다. 이후, 모델은 훈련 세트 전체에 대한 그룹 수준의 영향 예측을 극대화하도록 데이터를 선택하고, 효율적인 추론을 위해 데이터의 임베딩을 기반으로 클러스터링합니다.

- **Performance Highlights**: DCLM 기준benchmark에서 Group-MATES는 22개의 다운스트림 작업에서 DCLM-Baseline에 비해 10% 이상 성능 향상을 달성하며, 기존의 데이터 커레이션 기법들을 능가하는 성과를 보입니다. 또한, 관계 데이터 영향 모델이 데이터 포인트 간의 복잡한 상호작용을 효과적으로 포착할 수 있음을 실험적으로 검증하였습니다.



### I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search (https://arxiv.org/abs/2502.14693)
- **What's New**: 이 논문에서는 Introspective Monte Carlo Tree Search (I-MCTS)라는 새로운 접근법을 소개합니다. I-MCTS는 기존의 Monte Carlo Tree Search (MCTS)를 확장하여, 탐색 과정에서 생성된 노드를 보다 체계적으로 분석하고 개선하는 introspective(내성적) 과정을 도입합니다. 이 기법은 LLM 기반의 가치 모델을 통합하여 각 노드의 솔루션을 게시적으로 평가할 수 있도록 합니다. 이러한 접근은 전반적인 의사결정 과정을 향상시키며, 코드 생성의 퀄리티와 다양성을 높입니다.

- **Technical Details**: I-MCTS는 두 가지 주요 요소로 구성됩니다. 첫째, 반영적 솔루션 생성을 통한 노드 확장으로, 부모 및 형제 노드의 상태를 분석하여 고품질의 사고 노드를 동적으로 생성합니다. 둘째, LLM이 산출한 평가와 실제 성과 점수를 결합한 하이브리드 보상 메커니즘을 개발하였습니다. 이 과정을 통해 Q-value의 전환이 원활하게 이루어져, 탐색의 초기 단계에서부터 고품질 노드를 통과하게 됩니다.

- **Performance Highlights**: 본 연구는 다양한 ML 작업에 대한 Extensive 실험을 통해 기존의 최첨단 AutoML 기법보다 6% 절대적인 성능 향상을 보여줍니다. 이 연구는 LLM 기반의 AutoML 시스템에서 에이전트의 결정 과정을 더욱 효율적으로 만들어, 자동화된 머신러닝 워크플로우의 품질을 분명히 향상시킵니다. 이는 복잡한 머신러닝 작업에서 최적 경로 탐색의 효율성을 높이는 데 중요한 기여를 합니다.



### Bridging the Gap: Transforming Natural Language Questions into SQL Queries via Abstract Query Pattern and Contextual Schema Markup (https://arxiv.org/abs/2502.14682)
- **What's New**: 이번 논문에서는 복잡한 질문에 대해 기존의 LLM(대형 언어 모델)을 기반으로 한 Text-to-SQL 접근 방식의 한계점을 해결하기 위해 PAS-SQL을 제안합니다. PAS-SQL은 Abstract Query Pattern (AQP)과 Contextual Schema Markup (CSM)을 활용하여 구조적 매핑 격차(structural mapping gap)와 어휘 매핑 격차(lexical mapping gap)를 완화합니다. 이 방식은 데이터베이스 정보와 관계없는 질문의 구조적 패턴을 추출하여 더 유사한 시연을 찾을 수 있도록 합니다.

- **Technical Details**: AQP는 질문의 구조적 패턴을 추출하고, CSM은 질문 내 데이터베이스 관련 텍스트를 특정 테이블 또는 열과 연결하여 어휘 매핑 문제를 해결합니다. 이러한 기반 기술을 통해 PAS-SQL은 SQL 생성 파이프라인을 보다 효율적으로 작동시키며, 질문의 복잡성이 증가함에 따라 발생하는 격차를 줄이는 데 중점을 두고 있습니다. 실험은 Spider 및 BIRD 데이터셋에서 수행되었습니다.

- **Performance Highlights**: PAS-SQL + GPT-4o 조합은 Spider 벤치마크에서 87.9%의 실행 정확도로 새로운 최첨단 성능을 달성했으며, BIRD 데이터셋에서는 64.67%의 실행 정확도로 선두 결과를 기록했습니다. 이러한 높은 성능은 PAS-SQL이 제안하는 구조적 및 어휘적 접근 방식의 효과를 뒷받침합니다.



### How to Get Your LLM to Generate Challenging Problems for Evaluation (https://arxiv.org/abs/2502.14678)
- **What's New**: 최근에는 Large Language Models (LLMs)의 발전 속도가 매우 빨라 이를 효율적으로 평가하기 위한 새로운 접근법인 CHASE가 소개되었습니다. 기존의 인간 주석 방식은 비용과 품질 문제로 인해 한계가 있으며, CHASE는 LLMs를 사용해 인간 개입 없이 도전적인 문제를 합성적으로 생성할 수 있는 통합 프레임워크입니다. 이 프레임워크는 문제 생성 과정을 독립적으로 검증 가능한 하위 작업으로 분해하여 높은 수준의 품질과 정확성을 보장합니다.

- **Technical Details**: CHASE 프레임워크는 하위 요소로부터 문제를 하향식으로 구성하며, 문제의 맥락 속에 있는 해답의 일부를 반복적으로 숨기는 방식으로 문제를 생성합니다. 이러한 방식은 여러 단계의 추론 및 긴 맥락에 대한 사고 과정을 요구함으로써 문제를 더욱 어렵게 만듭니다. 또한 생성 과정의 각 단계를 개별적으로 검증할 수 있는 간단한 하위 작업으로 세분화하여 정확성을 검증할 수 있도록 합니다.

- **Performance Highlights**: 이 프레임워크를 통해 생성된 데이터셋을 이용한 실험에서 최신 LLM들은 도전적인 수치에 직면하게 되었으며, 최상의 모델도 정확도를 약 40-60%로 기록했습니다. 특히, CHASE는 기존의 Benchmarks와 비교할 때 훨씬 더 많은 오류가 있는 데이터를 제공하며, LLM의 성능 저하가 관찰되었습니다. 이 연구는 LLM의 성능 및 데이터의 질을 향상시키기 위한 새로운 방향성을 제시합니다.



### Data-Constrained Synthesis of Training Data for De-Identification (https://arxiv.org/abs/2502.14677)
Comments:
          Under review

- **What's New**: 이 연구는 개인정보 식별 정보를 추적하기 위해 임상 도메인에 적응된 대형 언어 모델(LLMs)을 활용하여 합성 임상 텍스트를 생성하는 방법을 제안합니다. 기존의 프라이버시 위험으로 인해 데이터셋이 널리 사용되지 않는 다수의 민감한 영역에서, 합성 데이터 생성은 새로운 대안으로 떠오르고 있습니다. 연구에서는 이 과정에서 소량의 도메인 데이터가 충분할 수 있음을 보여주면서, 머신 주석이 달린 데이터의 효과를 강조합니다.

- **Technical Details**: 본 연구는 두 가지 언어 모델인 GPT-SW3와 FLOR을 사용하여 스웨덴어 및 스페인어 임상 데이터의 합성을 진행합니다. 이 모델들은 각각 3200억 및 1400억 토큰의 데이터로 학습되어 있으며, NER 모델 학습에 필요한 합성 텍스트를 생성합니다. 이 과정은 오토 회귀(autoregressive) 방식으로 진행되며, 머신 주석 달기는 미세 조정된 NER 모델을 통해 이루어집니다.

- **Performance Highlights**: 연구 결과, 합성 데이터로 훈련된 NER 모델은 실제 민감 데이터로 훈련된 모델과 거의 동일한 성능을 나타내며, 데이터 유출 위험을 줄이는 데 기여합니다. 실험 결과에 따르면, 합성에서의 유용성은 높은 품질의 NER 모델에 의존하는 것으로 밝혀졌습니다. 또한, 주어진 작업에서 더 큰 생성 LLM을 사용하더라도 성능 향상에는 명확한 개선이 없는 것으로 나타났습니다.



### Explanations of Deep Language Models Explain Language Representations in the Brain (https://arxiv.org/abs/2502.14671)
- **What's New**: 이 연구는 설명 가능한 인공지능(Explainable AI, XAI) 기법을 활용하여 대형 언어 모델(LLM)과 뇌의 언어 처리 메커니즘 간 깊은 연결을 형성하는 새로운 접근 방식을 소개합니다. 이전의 연구가 주로 LLM의 내부 표현을 신경 활동과 일치시키는 데 중점을 두었던 반면, 본 연구는 속성 방법(attribution methods)을 사용하여 LLM의 다음 단어 예측에 대해 이전 단어들이 어떻게 기여하는지를 정량화하였습니다. 이 작업은 AI와 신경과학 간의 양방향 다리 역할을 하며, LLM의 예측 기능과 인간의 뇌에서의 언어 처리 과정을 통합하는 새로운 통찰을 제공합니다.

- **Technical Details**: 연구진은 네 가지 유형의 속성 방법을 적용하여 세 가지 LLM에서 피쳐 표현을 구성하고, 이를 통해 자연적인 이야기 청취 중에 기록된 fMRI 활동을 모델링하였습니다. 이 과정에서 각 LLM에서 도출된 피쳐 공간은 참가자의 뇌 반응을 예측하는 데 독립적으로 사용되었습니다. 예측은 각 개인의 뇌 데이터에 맞춰 설계된 선형 리지 회귀 모델을 통해 이루어졌으며, 다섯 번의 교차 검증을 통해 정확성을 평가하였습니다.

- **Performance Highlights**: 연구 결과, 속성 방법이 언어 관련 뇌 영역의 광범위한 네트워크 내에서 뇌 활동을 효과적으로 예측하는 데 강력한 가능성을 보였습니다. 특히, 속성 방법은 초기 언어 처리 영역에서의 뇌 활동 예측에서 전통적인 내부 표현보다 우수한 성과를 나타냈고, LLM의 깊이에 따른 설명과 뇌의 언어 처리 영역 간의 계층적 관계를 발견하였습니다. 이 발견은 LLM이 맥락 정보를 통합하는 데 있어 공유되는 메커니즘을 시사합니다.



### AlphaMaze: Enhancing Large Language Models' Spatial Intelligence via GRPO (https://arxiv.org/abs/2502.14669)
- **What's New**: 이 논문은 기존의 Large Language Models (LLMs)에 시각적 추론 기능을 강화하는 새로운 두 단계 훈련 프레임워크를 제안합니다. 이를 통해, LLM이 미로 탐색과 같은 과제에서 더 나은 성능을 발휘하도록 만드는 방법을 구체적으로 다룹니다. 최초 단계로 Supervised Fine Tuning (SFT)을 이용하여 시각적 미로 정보를 바탕으로 단계별 이동 명령어를 예측하도록 훈련시키고, 그 후 Group Relative Policy Optimization (GRPO)를 적용하여 순차적 의사결정을 개선합니다.

- **Technical Details**: 이 연구에서는 미로 정보를 효과적으로 처리하기 위해, 각 그리드 셀을 좌표 토큰으로 나타내는 토큰화된 입력 형식을 설계하였습니다. 벽의 정보는 여러 가지 토큰을 사용하여 인코딩하며, 출발지와 목표 위치는 각각 <|origin|> 및 <|target|> 토큰으로 표시됩니다. 이러한 토큰화 방식은 LLM이 미로 구조를 시각적으로 인식할 수 있도록 공간적 관계를 명확하게 인코딩하여 LLM이 '보는' 방식으로 강화합니다.

- **Performance Highlights**: 실험 결과에 따르면, 기본 모델은 미로 탐색에 실패하는 반면, SFT 훈련을 받은 모델은 86%의 정확도를 달성하였고, GRPO의 추가 훈련을 통해 정확도가 93%로 향상되었습니다. 또한, GRPO는 모델이 더욱 견고하고 자기 수정적인 추론을 할 수 있도록 돕는 강점을 보여주었습니다. 이러한 성과는 자율 탐색 및 로봇 공학과 같은 다양한 분야에 응용 가능성을 제시합니다.



### InstructAgent: Building User Controllable Recommender via LLM Agen (https://arxiv.org/abs/2502.14662)
Comments:
          WWW2025@HCRS

- **What's New**: 본 논문은 전통적인 추천 시스템의 취약점을 개선하고 사용자 이익을 보호하기 위해 새로운 사용자-에이전트-플랫폼 패러다임을 제안합니다. 기존의 추천 시스템들은 플랫폼의 추천 알고리즘에 직접적으로 노출되어 사용자에게 불리한 상황을 초래할 수 있었습니다. 이에 대한 해결책으로, 추천 시스템 사이에서 사용자를 보호하는 역할을 하는 에이전트를 도입하여 간접적으로 노출되는 방식을 제시합니다. 특히, 사용자-driven 지침을 사용하는 새로운 추천 데이터셋 InstructRec를 구성했습니다.

- **Technical Details**: 논문에서는 네 가지 추천 데이터셋을 구성하고, 사용자의 자유로운 입력을 통해 개인의 관심사를 학습하는 Instruction-aware Agent (InstructAgent)를 설계하였습니다. 이 에이전트는 기존 사용자 데이터 또는 다른 사용자의 행동에 영향을 받지 않으며, 사용자 개개인의 피드백을 바탕으로 학습합니다. 또한 Dynamic Memory Mechanism을 사용하여 사용자 프로필을 동적으로 유지하고 업데이트하며, 사용자 고유의 관심사를 깊이 탐구하는 Individual Instruction-aware Agent (Instruct2Agent)를 도입했습니다.

- **Performance Highlights**: Empirical 실험을 통해 제안된 Instruct2Agent가 기존 최첨단 접근법들보다 평균 16.6% 향상된 성과를 달성함을 입증하였습니다. 또한, 에코챔버 효과의 영향을 분석하고 활동적인 사용자와 비활동적인 사용자의 성과를 개별적으로 평가하여, 개발된 에이전트가 사용자와 추천 시스템 간의 방패 역할을 효과적으로 수행함을 확인했습니다.



### Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs (https://arxiv.org/abs/2502.14645)
- **What's New**: 이번 연구에서는 Cross-Lingual Knowledge Democracy Edit (X-KDE)라는 새로운 지식 편집 프레임워크를 제안합니다. 이 방법은 대형 언어 모델(LLM)이 특정 언어에서 수정된 지식을 다른 언어로 효과적으로 전파할 수 있도록 설계되었습니다. 특히, X-KDE는 Cross-lingual Edition Instruction Tuning (XE-IT)와 Target-language Preference Optimization (TL-PO)의 두 단계로 구성되어 있습니다.

- **Technical Details**: X-KDE는 두 단계로 구성됩니다. 첫 번째 단계인 XE-IT에서는 선별된 병렬 데이터셋을 사용하여 모델을 세밀하게 조정하며, 두 번째 단계인 TL-PO에서는 고급 최적화 기법을 적용하여 언어 간의 일관성을 유지합니다. 이를 통해 소스 언어에서의 지식이 타겟 언어로 원활하게 전파됩니다.

- **Performance Highlights**: 연구 결과, X-KDE는 Bi-ZsRE 및 MzsRE 벤치마크에서 평균 8.19%의 성능 향상을 달성하였으며, 단일 언어 설정에서도 높은 정확도를 유지합니다. X-KDE는 기존 방법들보다 더 나은 성능을 보이는 새로운 최첨단(SOTA) 솔루션으로 자리잡았습니다.



### LIFT: Improving Long Context Understanding of Large Language Models through Long Input Fine-Tuning (https://arxiv.org/abs/2502.14644)
Comments:
          arXiv admin note: text overlap with arXiv:2412.13626

- **What's New**: 본 논문은 Long Input Fine-Tuning (LIFT)라는 새로운 프레임워크를 소개하여, 긴 맥락 이해의 문제를 해결하고자 합니다. LIFT는 모델 파라미터를 동적으로 조정하여 짧은 맥락의 LLM들이 긴 입력에서도 적절하게 응답할 수 있도록 합니다. 중요한 점은, LIFT가 무한히 긴 입력을 처리하기 위해 맥락 창 크기를 확장하는 대신, 파라미터에 긴 입력을 저장하고 이를 흡수하는 방법을 선택한다는 것입니다. 이를 통해 기존 모델의 ICL (in-context learning) 기능을 유지하면서 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: LIFT는 긴 입력을 효과적으로 처리하기 위해, 입력을 겹치는 조각으로 나누어 짧은 맥락 창에 맞춰 훈련시키는 방식을 사용합니다. 새로운 긴 입력이 들어올 때마다 모델 파라미터를 조정하여 온라인에서 훈련을 수행하며, 이를 통해 자원 소모를 최소화합니다. 또, Gated Memory라는 특수한 어텐션 어댑터를 도입하여 긴 입력의 기억과 이해를 자동으로 균형을 맞춥니다. 이 방식으로 기존 LLM의 성능을 유연하게 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 여러 유명한 긴 맥락 벤치마크에서 LIFT의 성능을 평가한 결과, LIFT가 적용된 LLama-3-8B 모델은 긴 의존성 질문 답변 작업에서 29.97%의 정확도를 달성하여, 기존 ICL 모델의 15.44%에 비해 큰 성과를 보였습니다. 이러한 결과는 LIFT가 짧은 맥락 모델의 긴 맥락 이해를 크게 개선할 수 있음을 보여줍니다. 이 프레임워크는 기존 모델의 가능성을 확장하는 데 기여하며, 긴 맥락 시나리오에서의 다양한 응용 가능성을 제공합니다.



### Length-Controlled Margin-Based Preference Optimization without Reference Mod (https://arxiv.org/abs/2502.14643)
- **What's New**: 이번 논문에서는 Length-Controlled Margin-Based Preference Optimization (LMPO)라는 새로운 방법론을 제안합니다. LMPO는 기존의 Direct Preference Optimization (DPO)의 다양한 한계를 극복하기 위해 개발되었습니다. 특히, LMPO는 길이 편향(length bias), 메모리 비효율(memory inefficiency) 및 확률 저하(probability degradation) 문제를 해결하여 보다 효율적이고 강력한 대안을 제시합니다.

- **Technical Details**: LMPO는 DPO의 손실(loss)을 위한 상한선으로 균일한 참조 모델(reference model)을 도입하여 원래의 최적화 목표를 보다 정확하게 근사화합니다. 또한, 평균 로그 확률 최적화 전략을 사용하여 훈련(training) 및 추론(inference) 단계 간의 불일치를 최소화합니다. 주요 혁신 중 하나는 Bradley-Terry 프레임워크 내에서 길이 제어 마진 기반 손실 함수를 통합하여 응답 길이를 조절하고 선호된 출력과 거부된 출력 간의 마진을 넓히는 것입니다.

- **Performance Highlights**: LMPO는 Mistral과 LLaMA3 두 가지 오픈엔디드 대형 언어 모델을 대상으로 하는 여섯 가지 조건부 벤치마크에서 경쟁 성능을 보여주었습니다. 실험 결과, LMPO는 응답 길이를 효과적으로 제어하고 확률 저하를 줄이며 기존 방법들보다 뛰어난 성능을 발휘했습니다. 코드는 제공된 링크를 통해 확인할 수 있습니다.



### How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation (https://arxiv.org/abs/2502.14642)
- **What's New**: 최근 LLMs(Large Language Models)는 인간 디지털 트윈(human digital twins)으로서의 잠재력 때문에 학문적 분야에서 큰 주목을 받고 있습니다. 그러나 현재 LLMs에 대한 평가는 주로 대화 시뮬레이션에 치중되어 있으며, 디지털 트윈에 중요한 인간 행동 시뮬레이션은 간과되고 있습니다. 이러한 격차를 해소하기 위해, 우리는 LLMs의 지속적인 인간 행동 시뮬레이션 능력을 평가할 수 있는 첫 번째 벤치마크인 BehaviorChain을 제시합니다.

- **Technical Details**: BehaviorChain은 1,001개의 독특한 페르소나(personas)를 포함하여 총 15,846개의 다양한 고품질 행동 체인을 제공합니다. 각 페르소나는 상세한 이력(history)과 프로파일 메타데이터(metadata)를 포함하고 있습니다. 평가를 위해, 우리는 LLMs에 페르소나 메타데이터를 통합하고, 이를 사용하여 BehaviorChain에서 제공하는 동적 시나리오 내에서 맥락에 적절한 행동을 반복적으로 추론하도록 합니다.

- **Performance Highlights**: 포괄적인 평가 결과에 따르면, 최신 모델들조차도 지속적인 인간 행동을 정확하게 시뮬레이션하는 데 어려움을 겪고 있습니다. 이는 LLMs의 기존 평가 방식에 대한 새로운 접근이 필요함을 보여주며, BehaviorChain이 LLMs의 능력을 평가하는 데 필수적인 도구가 될 수 있음을 입증합니다.



### NAVIG: Natural Language-guided Analysis with Vision Language Models for Image Geo-localization (https://arxiv.org/abs/2502.14638)
- **What's New**: 새로운 데이터셋 NaviClues를 통해 전문가들의 사고 과정을 수집하여 이미지의 지리적 위치를 예측하는 방법을 제시합니다. 이 데이터셋은 인기 있는 게임인 GeoGuessr에서 파생된 것으로, 2,000개 이상의 사례를 포함하여 풍부한 징후와 통찰력을 제공합니다. 또한, Navig라는 새로운 이미지 지리 로컬라이제이션 프레임워크를 개발하여 시각적 및 언어적 정보 분석을 통합합니다.

- **Technical Details**: NaviClues 데이터셋은 5명의 전문 YouTuber의 게임 영상을 분석하여 고품질의 결과를 생성합니다. 이 데이터셋은 11,200개의 이미지와 각 이미지의 지리적 정보, 추론을 포함하고 있으며, Road signs와 같은 다양한 시각적 요소를 고려하여 인간 전문가들이 소프트웨어로 분석할 수 있는 인사이트를 제공합니다. Navig는 Reasoner, Searcher, Guesser의 세 가지 컴포넌트로 구성되어 다양한 지리적 위치를 예측하는 데 필요한 세밀한 정보를 제공합니다.

- **Performance Highlights**: Navig는 기존의 최첨단 모델들보다 평균 거리 오차를 14% 줄이는데 성공하였으며, 훈련 샘플 수는 1,000개 미만으로 유지되었습니다. 이러한 결과는 이미지 로컬라이제이션의 정확성을 크게 향상시킬 수 있는 잠재력을 지닌 것이며, 데이터셋과 코드가 공개되어 있어 연구자들이 이를 활용할 수 있습니다.



### Multi-Record Web Page Information Extraction From News Websites (https://arxiv.org/abs/2502.14625)
- **What's New**: 이 연구에서는 다수의 기록을 포함하는 웹 페이지에서 정보를 추출하는 문제에 집중했습니다. 이러한 과업은 방대한 웹 데이터 시대에 점차 중요해지고 있는데, 기존 연구는 대부분 단일 기록(detailed pages) 페이지에 초점을 맞춰왔습니다. 반면, 우리는 러시아어 뉴스 웹사이트에서 사용되는 리스트 페이지(mult-record list pages)에 특화된 대규모 데이터셋을 구축하여 이 간극을 메우고자 했습니다.

- **Technical Details**: 우리가 제작한 데이터셋은 13,120개의 뉴스 리스트 페이지로 구성되어 있으며, 속성은 다양한 유형을 포함하여 현실적인 학습 상황을 제공합니다. 이 연구에서는 MarkupLM 모델을 사용한 다단계 정보 추출 방법을 제안하였으며, 이는 웹 페이지에서 시각적 정보를 사용하지 않고도 효율적으로 데이터를 추출하게 돕습니다.

- **Performance Highlights**: 우리가 설계한 여러 실험을 통해 이 접근법의 장점을 검증했습니다. 데이터셋의 공개를 통해 다수의 기록 페이지에서 정보를 추출하는 분야의 발전을 촉진하길 기대합니다. 특히, 이 연구는 전통적인 데이터 추출 방법의 한계를 극복하고 새로운 데이터 셋을 통해 효과적인 정보 추출 방법론을 제시합니다.



### Exploring RWKV for Sentence Embeddings: Layer-wise Analysis and Baseline Comparison for Semantic Similarity (https://arxiv.org/abs/2502.14620)
Comments:
          17 pages, 3 tables, preprint on ArXiV, includes detailed analysis of RWKV for semantic similarity tasks

- **What's New**: 이 논문은 RWKV라는 새로운 언어 모델 아키텍처의 효능을 조사하며, 이는 선형 주의 메커니즘(linear attention mechanism)으로 알려져 있습니다. 저자는 사전 훈련된 RWKV 모델의 다양한 숨겨진 층(hidden layers)에서 생성된 문장 임베딩의 의미적 유사성(semantic similarity)을 평가하기 위해 층별(layer-wise) 분석을 수행했습니다. Microsoft Research Paraphrase Corpus(MRPC) 데이터셋을 사용하여 GloVe 기반 기준선과 비교했을 때, RWKV 임베딩은 의미적 관련성을 일부 포착했지만 Spearman 상관관계 측면에서 GloVe보다 부족한 성능을 보였습니다.

- **Technical Details**: RWKV는 혁신적인 수용 가중치 키 값(Receptance Weighted Key Value) 주의 메커니즘을 사용하여 선형 시간 복잡성을 달성합니다. 본 연구에서는 RWKV-v6-Finch-1B6-HF 모델을 사용하여 숨겨진 층에서 문장 임베딩을 추출하고, 계층별로 의미적 정보를 포착하는 역할을 조사하는 전략을 수립했습니다. GloVe 임베딩을 기준선으로 사용하여, 모든 단어의 GloVe 벡터를 평균내어 문장 임베딩을 생성하고, MRPC 데이터셋에서 문장 임베딩 간의 코사인 유사성이 어떻게 평가되는지를 분석했습니다.

- **Performance Highlights**: RWKV 임베딩은 일부 의미적 관련성을 포착했지만, GloVe 기준선에 비해 성능이 낮은 것으로 나타났습니다. 이 연구는 문장 임베딩 생성 분야에서 RWKV의 가능성을 확인하고, 효율성 및 문맥 이해에 대한 고유한 절충점을 강조합니다. 앞으로 RWKV의 구조를 최대한 활용하고, 기존 문장 임베딩 기술과의 성능 격차를 해소하기 위한 추가 연구가 필요합니다.



### FIND: Fine-grained Information Density Guided Adaptive Retrieval-Augmented Generation for Disease Diagnosis (https://arxiv.org/abs/2502.14614)
- **What's New**: 이 논문에서는 질병 진단 시나리오에서 Retrieval-Augmented Large Language Models (LLMs)의 신뢰성을 향상시키기 위해 FIND라는 새로운 프레임워크를 제안합니다. FIND는 입력의 정보 밀도에 기초하여 검색이 필요한지 여부를 결정하는 세밀 조정(adaptive control) 모듈을 통합하였습니다. 이를 통해 효율성과 정확성의 균형을 유지하면서 임상 요구 사항을 충족할 수 있도록 도와줍니다.

- **Technical Details**: FIND는 정보 밀도(information density)를 기반으로 하는 검색 최적화(retrieval optimization) 프로세스를 구현하며, 지식 필터링(Knowledge filtering) 모듈을 도입하여 검색의 질을 높입니다. 이 프레임워크는 기존의 Retrieval-Augmented Generation (RAG) 방법이 가진 약점을 보완하여, 임상 환경에 더욱 적합한 검색 방법을 제공합니다.

- **Performance Highlights**: 실험은 세 가지 중국 전자 의무 기록 데이터셋을 사용하여 수행되었으며, FIND는 여러 기준선(baseline) 방법들과 비교하여 각 분야에서 현저한 성능 향상을 보여주었습니다. 이러한 결과는 FIND가 임상 진단 임무에서의 효과성을 강조하며, 의료 분야에서의 잠재적인 응용 가능성을 마련합니다.



### Behavioral Analysis of Information Salience in Large Language Models (https://arxiv.org/abs/2502.14613)
- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 텍스트 요약(text summarization) 능력을 분석하며, 이 모델들이 어떻게 중요성을 기반으로 내용을 선택하는지에 대한 명확한 이해가 부족함을 지적합니다. 새로운 설명 가능한 프레임워크를 도입하여 LLM의 요약 행동을 통해 정보의 중요성(salience)을 체계적으로 조사하고자 합니다.

- **Technical Details**: 연구에서는 길이 제어(length-controlled)된 요약을 행동 연구의 도구로 사용하여 모델의 콘텐츠 선택 과정(content selection process)을 조사합니다. 또한, 논의 중인 질문(Questions Under Discussion)의 답변 가능성을 추적하여 모델이 정보를 어떻게 우선시하는지를 나타내는 프록시(proxy)를 도출합니다. 이렇게 얻은 데이터를 통해 13개의 모델을 대상으로 한 실험이 진행되었습니다.

- **Performance Highlights**: 시험 결과, LLMs는 모델 계열 및 크기에 대해 일반적으로 일관된 뉘앙스(nuanced) 있고 계층적인 중요성 개념을 갖고 있는 것으로 나타났습니다. 모든 모델은 높은 일관성을 보여주지만, 이러한 중요성 개념은 내부 관찰(introspection)을 통해 접근할 수 없으며, 인간의 정보 중요성 인식과는 약한 상관관계를 보였습니다.



### Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs (https://arxiv.org/abs/2502.14561)
- **What's New**: 이번 연구는 오픈된 대형 언어 모델(Large Language Models, LLMs)이 인컨텍스트 학습(in-context learning) 및 파인튜닝(fine-tuning)을 통해 인용(citation) 의도를 예측하는 능력을 조사합니다. 기존의 SciBERT와 같은 사전 훈련된 모델이 필요했던 접근법과 달리, 저자들은 LLM이 최소한의 특정 작업 데이터로도 이 작업에 적합하게 조정될 수 있음을 보여줍니다. 연구 결과는 LLM이 인용 의도를 인식하는 데 있어 강점과 한계를 명확히 하여 모델 선택과 프롬프트 엔지니어링에 대한 귀중한 통찰을 제공합니다.

- **Technical Details**: 연구는 LLaMA, Mistral, Phi, Gemma 및 Qwen 모델 중 다섯 가지 주요 오픈 모델 패밀리의 인스트럭션 튜닝 버전에서 12개의 모델 변형을 평가하고, 제로샷(zero-shot), 원샷(one-shot), 퓨샷(few-shot), 많은 샷(many-shot) 프롬프트를 사용하여 성능을 측정했습니다. 인컨텍스트 학습의 특성상 다양한 매개변수들이 모델 성능에 어떻게 영향을 미치는지 분석하며 최적의 구성 및 최고의 성능을 발휘하는 모델을 식별하였습니다. 실험에서 사용된 데이터셋으로는 SciCite와 ACL-ARCが 있으며, 각각 인용 의도가 포함된 문자열로 구성됩니다.

- **Performance Highlights**: 연구를 통해 얻은 결과는 LLM이 인용 의도를 분류하는 데 필요한 작업에서 전통적인 사전 훈련된 언어 모델에 비해 경쟁력 있는 성능을 보이며, 파인튜닝된 경우 더욱 향상된 성능을 나타냅니다. 또한 저자들은 모델 성능을 최적화하기 위해 다양한 파라미터를 이론화하고, 실험 결과를 공유하여 연구자들이 자신의 작업에 맞게 활용할 수 있도록 지원합니다. 전체 시험 프레임워크와 모델을 공개하여 연구 개발을 더욱 촉진하는 것이 이번 연구의 중요한 기여점입니다.



### Multiscale Byte Language Models -- A Hierarchical Architecture for Causal Million-Length Sequence Modeling (https://arxiv.org/abs/2502.14553)
Comments:
          Under Review

- **What's New**: 이번 논문에서 소개된 Multiscale Byte Language Model (MBLM)은 기존의 tokenization 문제를 해결하기 위해 설계된 모델로, 5M 바이트의 긴 입력 문자열을 단일 GPU에서 훈련할 수 있는 능력을 제공합니다. MBLM은 Transformer와 Mamba 블록을 결합하여 효율적인 학습 및 추론을 가능하게 하며, 비주얼 Q&A와 같은 멀티모달 작업에서도 뛰어난 성능을 보여줍니다. 이는 바이너리 데이터와 다양한 표현을 통합하여 더 넓은 데이터 소스에 적응할 수 있는 가능성을 강조합니다.

- **Technical Details**: MBLM은 N개의 causal decoder 모델로 이루어진 계층 구조를 가지고 있으며, 각 단계는 입력 시퀀스의 패치와 컨텍스트 크기를 조절하여 효율적인 처리 속도를 달성합니다. 이 모델은 256개의 바이트 레벨 설정을 사용하는 어휘를 통해 입력되는 바이트 스트림을 처리하여, 훈련 및 추론 중에 중간 활성화를 선택적으로 체크포인트 할 수 있는 기능을 제공합니다. 이러한 접근 방식은 5M 바이트에 달하는 긴 시퀀스를 단일 GPU에서 효율적으로 처리하는 데 기여합니다.

- **Performance Highlights**: MBLM은 멀티모달 환경에서 사용하는 새로운 어플리케이션을 통해 기존의 CNN-LSTM 아키텍처와 유사한 성능을 달성했음을 보여주었습니다. 특히, 비주얼 Q&A 작업에서 순수한 다음 토큰 예측만으로 훌륭한 결과를 보이며, 다양한 데이터 표현과의 강력한 적응성을 입증했습니다. 이러한 성능은 다양한 데이터 소스로부터 효과적인 크로스 모달리티 지식 전이를 가능하게 하여, 데이터 포맷에 관계없이 바이스트림에서 특징과 패턴을 포착하는 데 중점을 두고 있습니다.



### LLM-based User Profile Management for Recommender System (https://arxiv.org/abs/2502.14541)
Comments:
          Submitted to ACL 2025

- **What's New**: PURE는 사용자 리뷰로부터 핵심 정보를 체계적으로 추출하고 요약하여 발전하는 사용자 프로필을 유지하는 LLM 기반 추천 프레임워크입니다. 이 시스템은 전통적인 추천 방법이 놓친 사용자 생성 텍스트 데이터를 활용하여 추천의 정확성을 높입니다. PURE는 'Review Extractor', 'Profile Updater', 'Recommender'의 세 가지 핵심 구성 요소로 구성되어 있습니다.

- **Technical Details**: PURE는 사용자 리뷰를 분석하여 사용자의 선호도, 싫어하는 점, 그리고 주요 제품 기능을 식별하는 'Review Extractor'를 포함합니다. 'Profile Updater'는 새로 추출된 정보를 기존 사용자 프로필과 통합하여 중복을 제거하고 충돌을 해결합니다. 마지막으로, 'Recommender'는 최신 사용자 프로필을 사용하여 개인화된 추천을 생성합니다.

- **Performance Highlights**: PURE는 연속적인 추천 시나리오를 도입하여 시간에 따라 리뷰가 추가되면서 사용자 프로필을 지속적으로 업데이트할 수 있습니다. 실험 결과, PURE는 아마존 데이터셋에서 기존 LLM 기반 방법을 초월하며 긴 구매 이력과 사용자 리뷰를 효과적으로 활용함으로써 추천의 정확성을 높입니다.



### LoRA-GGPO: Mitigating Double Descent in LoRA Fine-Tuning via Gradient-Guided Perturbation Optimization (https://arxiv.org/abs/2502.14538)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 효과적인 파인튜닝을 위한 새로운 방법인 LoRA-GGPO(Gradient-Guided Perturbation Optimization)를 제안합니다. LoRA-GGPO는 gradient와 weight norms를 활용하여 모델의 double descent 문제를 완화하는 데 집중하며, 이를 통해 모델 일반화 성능을 향상시키고 더 평탄한 최소값을 찾도록 유도합니다. 여러 자연어 이해(NLU) 및 생성(NLG) 과제에서 LoRA-GGPO가 기존의 LoRA 및 최신 변형보다 뛰어난 성능을 보임을 입증하였습니다.

- **Technical Details**: LoRA-GGPO는 손실 경관(loss landscape)의 뾰족함(sharpness)을 최적화하여 gradiet와 weight norms의 가중 조합에 의해 생성된 랜덤 perturbations를 도입합니다. 이는 손실 함수의 변화에 가장 민감한 영역에 perturbations가 집중되도록 하여, 모델이 더 평탄한 최소값을 찾을 수 있도록 돕습니다. LoRA-GGPO는 기존의 Sharpness-Aware Minimization(SAM) 같은 방법에 비해 연산 및 메모리 부담을 줄이며 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, LoRA-GGPO는 자연어 이해 및 생성 작업에서 일관되게 LoRA 및 다른 최신 변형들보다 우수한 성능을 발휘했습니다. 또한 double descent 현상을 겨냥한 연장 실험을 통해 LoRA-GGPO가 이 문제를 효과적으로 완화함을 empirically 증명했습니다. 이로 인해 LoRA-GGPO는 LLM 파인튜닝을 위한 실용적인 솔루션으로 자리잡을 수 있음을 보여주었습니다.



### CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models (https://arxiv.org/abs/2502.14529)
- **What's New**: 이번 논문에서는 Large Language Model 기반의 Multi-Agent Systems (LLM-MAS)에서의 보안 문제를 다루고 있습니다. 특히, 'Contagious Recursive Blocking Attacks (Corba)'라는 새로운 유형의 공격을 소개하며, 이는 에이전트 간 상호작용을 방해하고 시스템의 가용성을 감소시킬 수 있습니다. 기존의 공격 기법들이 간과했던 이러한 차단 공격은 LLM-MAS의 보안적 측면에서 심각한 우려를 불러일으키고 있습니다.

- **Technical Details**: Corba 공격은 두 가지 주요 특성을 활용하여, 네트워크 토폴로지에 관계없이 전파할 수 있는 전염성(contagious)을 가지고 있습니다. 또한, 재귀적(recursive) 특성 덕분에 계산 자원을 지속적으로 고갈시킬 수 있습니다. 이는 공격에 적대적인 프롬프트가 반복적으로 시스템 안에 남아 효과를 유지하게 만듭니다.

- **Performance Highlights**: 논문에서는 AutoGen과 Camel의 두 가지 널리 사용되는 LLM-MAS 프레임워크를 대상으로 Corba의 효과를 실험했습니다. 실험 결과, Corba는 다양한 토폴로지 구조에서 LLM-MAS의 가용성을 감소시키고 자원을 낭비하며, 기존의 방송 기반 공격에 비해 우수한 성능을 보였습니다. 이러한 연구 결과는 LLM-MAS의 보안 메커니즘 개발에 있어 기초를 제공하며, 향후 연구에서 더욱 심도 깊은 분석이 필요함을 시사합니다.



### MultiSlav: Using Cross-Lingual Knowledge Transfer to Combat the Curse of Multilinguality (https://arxiv.org/abs/2502.14509)
- **What's New**: 이번 연구에서는 다국어 신경 기계 번역(Multilingual NMT)에서의 데이터 적용 방식에 대한 여러 접근법을 탐구하였습니다. 이는 저자원 언어에 대한 크로스 링궤 지식 이전(Cross-lingual Knowledge Transfer) 효과를 입증하며, 한국어와 같은 슬라브어 계열의 번역 품질을 향상시키는 데 기여했습니다. 또한, 슬라브어 간의 번역을 위한 최첨단 오픈소스 NMT 모델도 출시되었습니다.

- **Technical Details**: 연구에서는 체코어, 폴란드어, 슬로바키아어 및 슬로베니아어와 같은 라틴 스크립트 슬라브어를 포함한 저자원 및 중자원 언어에 대해 Multilingual NMT 접근법의 적용을 연구합니다. 또한 영어와 같은 고자원 언어를 추가하여 성능 향상이 이루어질 수 있는지를 탐구하며, 여러 다국어 번역 시나리오를 평가합니다. 이 과정에서 Bi-Directional 모델, Multi-way Multilingual 모델 및 Pivot 모델이 포함됩니다.

- **Performance Highlights**: 연구 결과, 슬라브 언어 간의 번역에서 크로스 링궤 지식 이전 효과가 확인되었습니다. 방향성 제로샷(Zero-shot) 설정에서도 다언어 훈련이 저자원 방향의 품질을 향상시키는 것이 입증되었습니다. 이 연구는 슬라브 언어가 상호 연관이 깊음을 보여주며, 다양한 과제를 해결하는 데 있어 전통적인 NMT 방법이 여전히 유효함을 강조합니다.



### Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases (https://arxiv.org/abs/2502.14507)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)이 제2 언어 학습자들이 사용하는 비원어민적인 영어 표현을 모사하는 능력을 평가합니다. 이를 위해 일본어, 태국어, 우르두어 등 특정 모국어(L1)를 가진 L2 영어 학습자들을 대상으로 대화 형식의 인터뷰를 진행하며, LLMs의 출력을 실제 L2 학습자 데이터와 비교합니다.

- **Technical Details**: 연구는 정보 이론(information theory)과 분포 밀도(distributional density) 측정을 통해 L1에 의해 영향을 받는 언어적 편향을 분석합니다. 특히, 참조 단어 사용(reference word usage)과 회피 행동(avoidance behaviors) 등의 사례를 살펴보며, LLM들이 실제 L2 학습 데이터에 나타나는 L1 의존적인 패턴을 어떻게 복제하는지를 조사합니다.

- **Performance Highlights**: 연구 결과, 최신 LLMs(예: Qwen2.5, LLAMA3.3, DeepseekV3, GPT-4o)는 일본어, 한국어, 중국어와 같은 다양한 언어의 영향을 받아 시제 일치(tense agreement)와 명사-동사 병렬(collocations) 등에 있어 L2 학습 데이터의 패턴을 효과적으로 재현하는 것으로 나타났습니다. 이러한 결과는 교육적 응용을 위한 L2 대화 생성 및 평가에 있어 LLM의 잠재력을 제시합니다.



### How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM? (https://arxiv.org/abs/2502.14502)
- **What's New**: 이번 연구에서는 Low-Rank Adaptation (LoRA) 기법을 사용하여 기존의 지식을 손상시키지 않으면서 Large Language Models (LLMs)에 새로운 사실을 통합하는 방법을 탐구하였습니다. 연구팀은 Llama-3.1-8B-instruct 모델에 다양한 양의 새로운 지식을 적용하여 미세 조정을 수행하였고, 데이터 구성의 중요성을 강조하며 이전에 학습된 지식과 새로운 지식의 조화를 이루기 위한 방안을 제시하였습니다.

- **Technical Details**: LLMs는 수십억 개의 파라미터를 가지며, 모델의 미세 조정은 시간과 비용이 많이 드는 과정입니다. 이를 극복하기 위해 Parameter-Efficient Fine-Tuning (PEFT) 기법이 대두되었으며, 그 중 LoRA는 효과적인 방법으로 자리 잡고 있습니다. 하지만 과도한 새로운 데이터가 모델의 기존 지식에 부정적인 영향을 미칠 수 있다는 점도 분명해졌습니다.

- **Performance Highlights**: 연구 결과, 새로운 사실이 포함된 혼합된 데이터를 사용한 경우에 모델의 성능이 가장 우수하였으나, 외부 질문-응답 벤치마크에서의 성능이 저하되는 문제도 발견되었습니다. 특정 엔티티에 편향된 학습 데이터는 모델이 일부 답변에 과도하게 치우치는 경향을 보였고, 모델이 자신감이 높아져 실제로 응답하지 않는 경우도 발생했습니다. 이러한 결과들은 새로운 지식 통합과 일반적인 모델 능력 간의 균형을 이루기 위한 훈련 데이터 구성의 중요성을 일깨워줍니다.



### Towards a Perspectivist Turn in Argument Quality Assessmen (https://arxiv.org/abs/2502.14501)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이 논문은 주관적인 특성을 갖는 주장의 질(Argument Quality, AQ)에 대한 데이터세트를 체계적으로 검토하여 기존의 문제를 해결합니다. 주장의 질을 두 가지 측면으로 나누어 논의하며, 첫 번째는 어떤 특성이 주석으로 달렸는지, 두 번째는 주석을 단 사람에 대한 정보입니다. 이러한 체계적인 데이터베이스는 AQ 모델 개발의 기초로 활용될 수 있습니다.

- **Technical Details**: 연구팀은 103개의 AQ 데이터세트를 수집하고, 각각에 대해 32종의 메타 정보를 포함한 자료를 제공합니다. 이 데이터세트의 질적인 분석을 통해, 주석자의 투명성과 사회 인구학적 다양성이 결여되어 있음이 밝혀졌습니다. 또한, 비집계적(non-aggregated) 레이블이 달린 24개의 데이터 세트를 심층 분석하여, 주관성 관련 문제를 해결하기 위한 모델 개발의 중요성을 강조합니다.

- **Performance Highlights**: 논문은 AQ 연구의 미래로 나아가기 위한 주관적인 주석 변동성을 고려한 새로운 데이터 세트의 필요성을 강조합니다. 또한, 데이터셋의 비교 가능성과 상호 운용성을 높이기 위해 AQ 카테고리를 포괄하는 새로운 분류 체계를 제안합니다. 이로 인해 향후 주관적인 분석에 대한 응용 가능성이 커질 것으로 기대됩니다.



### MLGym: A New Framework and Benchmark for Advancing AI Research Agents (https://arxiv.org/abs/2502.14499)
Comments:
          35 pages, 12 figures, 10 tables

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트를 평가하고 개발하기 위한 새로운 프레임워크인 Meta MLGym과 MLGym-Bench를 소개합니다. 이 프레임워크는 다양한 AI 연구 과제를 포함하는 최초의 Gym 환경으로, LLM 에이전트를 훈련하는 강화 학습(rl) 알고리즘 연구를 가능하게 합니다. MLGym-Bench는 컴퓨터 비전, 자연어 처리 등 다양한 도메인에서 13개의 개방형 AI 연구 과제를 포함하고 있으며, 연구 과정에 필요한 실제 기술을 요구합니다.

- **Technical Details**: MLGym은 다양한 AI 연구 과제를 통합하여 LLM 에이전트를 개발하고 평가할 수 있는 통합 플랫폼으로 설계되었습니다. 이 프레임워크는 에이전트의 성능을 평가하기 위해 강화를 위한 알고리즘, 커리큘럼 학습, 개방형 학습 등 다양한 새로운 학습 알고리즘을 쉽게 추가하고 통합할 수 있도록 합니다. 또한, 각 에이전트의 출력물인 모델, 알고리즘, 예측 세트 등을 평가할 수 있는 유연한 평가 방식도 제공합니다.

- **Performance Highlights**: MLGym-Bench에서 제공하는 과제를 수행한 여러 최첨단 LLM 모델의 성능을 비교하였으며, 그들 각각의 강점과 제한점을 강조했습니다. 연구 결과 현재의 최첨단 모델들은 대부분 하이퍼파라미터를 최적화하여 개선할 수 있지만, 혁신적인 가설이나 알고리즘을 생성하지는 못한다는 것을 확인하였습니다. 이 연구는 LLM 에이전트의 AI 연구 능력을 향상시키기 위한 프레임워크와 벤치마크를 오픈소스로 제공하여 향후 연구에 기여할 것으로 기대됩니다.



### Stories that (are) Move(d by) Markets: A Causal Exploration of Market Shocks and Semantic Shifts across Different Partisan Groups (https://arxiv.org/abs/2502.14497)
- **What's New**: 이번 연구는 의미 임베딩 공간(semantic embedding space)에서의 변화가 실제 세계의 충격(real-world shocks)과 인과적으로 연결될 수 있음을 처음으로 입증하고 있습니다. 또한, 당파성(partisanship)이 금융 시장의 변화 예측에 미치는 영향을 분석하며, 사건 발생 후 시장의 반응 역시 당파 그룹에 따라 다르게 나타남을 보여줍니다. 이 논문은 뉴스 매체와 시장 충격 간의 피드백 루프(feedback loops)의 존재를 정량적으로 밝혀내고, COVID-19와 같은 급작스러운 경제적 사건에서 텍스트가 중요한 외생 변수(exogenous variable)임을 강조합니다.

- **Technical Details**: 경제 및 금융에서는 시장 사건과 그 사건을 주도하는 내러티브(narratives) 간의 복잡한 변증법(dialectic)이 존재합니다. 이 논문은 기존의 문서 기반 예측 모델을 향상시키기 위해 텍스트에서 파생된 정보를 활용하며, 새로운 연구 영역인 언어 기록의 비동기적(단기적) 변화를 통해 피드백 루프를 정량화합니다. 변별력 있는 LLM(transformer-based LLM)의 이점 덕분에 텍스트의 정량적 분석이 가능해져, 내러티브와 거시 경제 간의 양방향 관계를 이해하는 데 높은 가능성을 제시하고 있습니다.

- **Performance Highlights**: 저자들은 당파성에 따라 금융 시장의 예측력을 높일 수 있는 가능성을 제시하며, 패턴과 결과의 차이를 인식하는 것의 중요성을 강조하고 있습니다. 이 연구는 텍스트가 경제 예측에 있어 중요한 역할을 하며, 기존 연구가 방어적으로 접근한 부분에 비해 더 직접적인 관계를 제시합니다. 최종적으로, 언어 데이터가 경제 분석에서 가지는 특별한 가치와 이론적인 기초를 제시함으로써 향후 경제 예측 모델 개선의 기반을 마련할 수 있습니다.



### Enhancing Language Multi-Agent Learning with Multi-Agent Credit Re-Assignment for Interactive Environment Generalization (https://arxiv.org/abs/2502.14496)
Comments:
          24 pages, under review

- **What's New**: LLM(대형 언어 모델) 기반 에이전트들은 모바일 운영 및 웹 브라우징 등 다양한 대화형 환경에서 눈에 띄는 발전을 이루었습니다. 그러나 기존의 다중 에이전트 시스템은 미리 정의된 역할로 인해 환경 간 일반화에 어려움을 겪고 있습니다. 본 논문에서는 이러한 문제를 해결하기 위해 새로운 다중 에이전트 강화 학습 프레임워크인 CollabUIAgents를 제안합니다.

- **Technical Details**: CollabUIAgents는 언어 다중 에이전트 시스템을 위한 강화 학습 프레임워크로, LLM을 활용한 혁신적인 다중 에이전트 신용 재배분(CR) 전략을 절거합니다. 이 전략은 환경 특정 보상 대신, LLM에 내재된 세계 지식을 활용하여 프로세스 보상을 할당합니다. 이를 통해 역할이 없는 에이전트 간의 협업 행동을 촉진하며, 일반화 가능한 정책 학습을 도모합니다.

- **Performance Highlights**: 실험 결과, CollabUIAgents는 기존의 에이전트 학습 방법과 Google의 강력한 폐쇄형 모델인 Gemini 1.5 Pro와 비교하여 뛰어난 성능을 달성했습니다. 특히, 모바일 환경에서 웹 환경으로의 일반화 성과가 두드러지며, CR에서 사용된 가이드 LLM인 GPT-4에 필적하거나 그 이상의 성능을 보였습니다. 이 연구는 강화 학습 보상을 이용한 환경 일반화의 효과성과 다중 에이전트 시스템 내 훈련된 LLM의 적응 가능성을 보여줍니다.



### StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following (https://arxiv.org/abs/2502.14494)
Comments:
          18 pages, 8 figures, 8 tables

- **What's New**: 이 연구는 다중 턴 인스트럭션을 따르는 능력을 평가하기 위한 새로운 벤치마크인 StructFlowBench를 제안하고 있습니다. 다중 턴 대화의 구조적 흐름 모델링을 통해, 기존의 단순한 평가 방식을 넘어 사용자의 의도를 반영하는 더 복잡한 평가를 가능하게 합니다. 이 벤치마크는 다섯 가지 구조적 제약 조건과 여섯 가지 기본 상호 턴 관계를 정의하여 모델 평가 방식을 혁신합니다.

- **Technical Details**: StructFlowBench는 8개의 인트라 턴(내부 턴) 제약과 5개의 새로 제안된 구조적 제약으로 구성된 이중 제약 평가 시스템을 포함하고 있습니다. 이 시스템은 다중 턴 간의 종속성을 고려하여 모델이 개별 제약을 서술하는 데 그치지 않고 전체 대화에서 논리적 일관성을 유지할 수 있도록 합니다. 여섯 가지 범주의 구조적 흐름 분류법도 소개하여 대화 구조 분석을 시스템적으로 지원합니다.

- **Performance Highlights**: 실험 결과, 13개의 최신 오픈 소스와 클로즈드 소스 LLM의 성능을 비교한 결과, 현재 모델들이 다중 턴 대화 구조를 이해하는 데 상당한 결함이 있음을 밝혀냈습니다. 이 연구는 다중 턴 대화에 대한 새로운 평가 기준과 나아갈 방향을 제시하며, 향후 더 강력한 지침 추적 모델을 개발하는 데 기초 자료를 제공합니다.



### NLoRA: Nyström-Initiated Low-Rank Adaptation for Large Language Models (https://arxiv.org/abs/2502.14482)
- **What's New**: 이 논문에서는 파라미터 효율적인 미세 조정(PEFT) 방법인 StructuredLoRA (SLoRA)와 NyströmLoRA (NLoRA)를 소개합니다. 기존 LoRA 방식의 느린 수렴 문제를 해결하기 위해 Nyström 방법을 활용하여 계산 비용을 줄이고 효율성을 개선했습니다. 또한, IntermediateTune (IntTune) 방법을 제안하여 NLoRA의 중간 행렬만을 조정함으로써 파라미터 수를 획기적으로 줄이는 방안을 모색합니다.

- **Technical Details**: SLoRA는 저랭크 행렬들의 사이에 소규모 중간 행렬을 추가하여 모델의 표현력을 높이고, NLoRA는 Nyström 방식을 기반으로 한 초기화를 통해 성능과 효율성을 동시에 향상합니다. NLoRA는 기본적으로 SVD를 대체하는 방법으로, 행렬의 일부 행과 열을 샘플링하여 계산 비용을 O(mr + r^2 + rn)으로 줄입니다. IntTune은 NLoRA의 중간 행렬만을 조정하여, 파라미터 수를 320M에서 4M으로 줄이며 성능을 이전의 LoRA보다 7.45% 향상시킵니다.

- **Performance Highlights**: SLoRA와 NLoRA는 GSM8K에서 각각 56.48%와 57.70%의 정확도를 기록하여 기존 LoRA 대비 각각 33.52% 및 36.41% 높은 성능을 보였습니다. IntTune은 LoRA에 비해 평균 NLG 성능을 7.45% 향상시켰고, 필요한 트레이닝 파라미터는 1.25%에 불과했습니다. 이러한 성과는 자원 제한 환경에서의 LLM 성능 향상을 보여줍니다.



### Unshackling Context Length: An Efficient Selective Attention Approach through Query-Key Compression (https://arxiv.org/abs/2502.14477)
Comments:
          14 pages,2 figures

- **What's New**: 이번 논문에서는 효율적인 토큰 선택을 통해 긴 맥락 시퀀스를 처리하는 새로운 방법인 Efficient Selective Attention (ESA)을 제안합니다. 기존의 방법들은 중요한 정보를 손실할 수 있는 영구적인 제거 전략이나 청크 단위로 토큰을 선택하는 방식을 사용했습니다. ESA는 특정 토큰 레벨에서 가장 중요한 토큰을 효율적으로 선택하여 맥락의 길이를 확장합니다.

- **Technical Details**: ESA는 두 단계로 이루어져 있습니다: 효율적인 선택과 주의(attention) 계산입니다. 첫 번째 단계에서, 쿼리-aware 토큰 선택 메커니즘을 도입하여 가장 중요한 토큰을 적응적으로 식별합니다. 이를 통해 쿼리와 키 벡터를 저차원 표현으로 압축하여 계산 비용을 줄입니다. 두 번째 단계에서는 선택된 토큰의 전체 키와 값을 사용하여 전통적인 주의 계산의 복잡성을 2차원에서 선형으로 줄입니다.

- **Performance Highlights**: ESA는 최대 256k까지의 긴 시퀀스 벤치마크에서 평가되었으며, 8k 및 32k의 맥락 길이를 가진 오픈소스 LLM에서 다른 선택 주의 메소드보다 우수한 성능을 보였습니다. 특히 여러 정보를 검색하는 작업에서 유사한 성능을 나타내었고, 일부 작업에서는 뛰어난 결과를 달성했습니다.



### Argument-Based Comparative Question Answering Evaluation Benchmark (https://arxiv.org/abs/2502.14476)
Comments:
          8 pages, 7 Tables, 13 Figures, 18 pages with Appendix

- **What's New**: 이 논문에서는 자동 비교 질문 응답(Comparative Question Answering, CQA)의 문제를 해결하기 위해 평가 프레임워크를 제안합니다. 총 15개의 기준을 세우고, 6개의 대형 언어 모델(LLMs)과 두 개의 CQA 데이터셋에서 수집된 정보를 바탕으로 비교 답변의 품질을 평가합니다. 연구진은 여러 LLM을 사용하여 테스트를 수행하고, Llama-3 70B Instruct 모델이 요약 평가에서 가장 우수한 성과를 보임을 보여주었습니다.

- **Technical Details**: 연구자는 LLM을 활용한 자동 CQA 평가 파이프라인을 구현했으며, 이를 통해 6개의 현대 LLM(GPT-3.5, GPT-4, Llama3-8B Instruct, Llama3-70B Instruct, Perplexity, Mixtral)의 성능을 평가했습니다. 평가 기준으로는 구조, 관련성, 품질 등이 있으며, 각 답변의 질을 측정하기 위해 0~19점의 점수를 부여합니다. 이 프레임워크는 각각의 질문에 대한 평가 점수를 할당하는 방식을 사용하여, 주간 협업을 통해 명확한 기준을 제공합니다.

- **Performance Highlights**: 결과적으로, GPT-4는 비교 질문에 대한 답변에서 최고의 성능을 보였으며, Llama-3 70B Instruct는 요약 평가에서 최상의 성과를 발휘했습니다. 연구진은 자동 평가와 인간 평가 간의 유용성을 비교하여 제안된 벤치마크의 타당성을 확인했습니다. 모든 데이터와 코드, 평가 결과는 공개되어 있어 추가적인 연구 및 개발에 기여할 것입니다.



### Enhancing Smart Environments with Context-Aware Chatbots using Large Language Models (https://arxiv.org/abs/2502.14469)
Comments:
          11 pages, 3 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLM)을 활용하여 스마트 환경 내에서 문맥 인지 상호작용을 구현하는 새로운 아키텍처를 제안합니다. UWB 태그를 통한 사용자 위치 데이터와 센서를 갖춘 스마트 홈을 통합하여 사용자의 활동 인식(HAR)을 실시간으로 수행하여, 개인 맞춤형 상호작용과 추천을 생성할 수 있는 챗봇을 구현합니다. 이러한 접근 방식은 기존의 정적 챗봇 상호작용을 넘어 사용자의 현재 상황에 동적으로 적응할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 주요 구성요소로는 사용자 상호작용을 모니터링하는 환경 센서 네트워크, UWB를 이용한 정밀 위치 추적, 및 HAR 모델이 포함됩니다. 이러한 센서는 사용자의 활동과 환경에 대한 풍부한 데이터를 수집하며, MQTT를 통해 중앙 허브로 전송되어 분석됩니다. 이후, 이 데이터를 기반으로 LLM와 통합된 챗봇이 사용자 요구를 이해하고 맥락에 맞는 응답을 생성하도록 설계되었습니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 사례 연구를 통해 제안된 아키텍처의 실행 가능성과 효과성을 입증하였습니다. HAR와 실시간 위치 정보를 통합하여 개인 맞춤형의 문맥에 맞는 사용자 경험을 제공함으로써 스마트 홈 내에서 보다 직관적이고 유용한 상호작용을 가능하게 했습니다. 이 연구는 LLM과 스마트 환경 기술의 융합이 사용자 맞춤형 지원을 제공하는 데에 혁신적인 가능성을 열어준다는 점을 강조합니다.



### Optimal word order for non-causal text generation with Large Language Models: the Spanish cas (https://arxiv.org/abs/2502.14451)
- **What's New**: 이 연구는 비인과 언어 모델을 이용한 최적의 텍스트 생성 순서를 분석한 최초의 논문으로, 스페인어에 대한 벤치마크를 제공합니다. 기존의 인과론적(NLG) 모델들이 영어와 같은 SVO 구조에 치우쳐 있다는 점을 명확히 하고, 언어의 구문적 풍부성을 충분히 반영하지 못함을 지적합니다. 연구에서는 Viterbi 알고리즘을 활용하여 비인과 비슷 구조의 생성 확률을 평가하여, 다양한 언어의 NLG에 대한 새로운 접근법을 제시합니다.

- **Technical Details**: 연구에서는 비인과적 생성 모델을 사용하고, Viterbi 알고리즘을 기반으로 한 최대 가능성(maximum likelihood) 단어 생성 순서 추정 방법을 제안합니다. 이를 통해 스페인어 문장에서 최적의 생성 순서를 규명하고, 인과적 언어 모델이 생성에 미치는 한계를 검토합니다. 연구는 비인과적 모델이 언어적 구조를 더 잘 반영할 수 있다는 점에서, 특히 스페인어와 같은 비엄격하게 구조가 짜여진 언어에 유리할 수 있음을 주장하고 있습니다.

- **Performance Highlights**: 연구 결과는 스페인어의 비인과적 생성 모델에서 최적 생성 순서가 인과적 순서와 밀접하게 관련되지 않음을 보여줍니다. 최적의 생성 순서는 주어진 문장의 구문적 구조에 따라 좌우되며, 이는 인간의 텍스트 인지 방식과 유사한 접근법을 필요로 한다는 점을 강조합니다. 이러한 결과는 스페인어 NLG 분야의 효율성과 유효성을 높이는 새로운 전략을 개발할 수 있는 기초를 제공할 수 있습니다.



### PredictaBoard: Benchmarking LLM Score Predictability (https://arxiv.org/abs/2502.14445)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)에서의 예측 가능성을 평가하기 위한 새로운 협업 벤치마킹 프레임워크인 PredictaBoard를 소개합니다. PredictaBoard는 기존 데이터셋에서 특정 작업 인스턴스에 대한 LLM 오류를 예측하는 점수 예측자(assessor)의 능력을 평가하기 위해 설계되었습니다. 이를 통해 LLM과 assessor의 조화로운 연구와 개발을 촉진하며, 안전한 AI 시스템을 위한 기반을 마련하고자 합니다.

- **Technical Details**: PredictaBoard는 LLM의 성능과 그 예측 가능성을 공동으로 평가하는 데 초점을 맞추고 있습니다. 이 프레임워크는 LLM-배치 쌍을 객체로 하여 각각의 벤치마크 인스턴스에서 LLM의 점수를 예측하도록 설계된 assessor를 포함합니다. 예측 가능성의 품질과 LLM의 성능을 결합하여 정확도-거부 곡선(Accuracy-Rejection Curve, ARC)을 통해 평가하며, 이는 실용적으로 오류 허용 한계를 결정하고 예측 가능한 유효 영역(predictably valid region)을 파악하는데 도움을 줍니다.

- **Performance Highlights**: PredictaBoard는 현재 상태의 LLM을 대상으로 하여 기초 평가자(baseline assessors)와 함께 초기 실험 결과를 보고합니다. 이는 LLM의 성능뿐만 아니라 그 예측 가능성을 동시에 고려함으로써 AI 안전성을 강화할 수 있는 가능성을 제시합니다. 또한 PredictaBoard는 LLM과 assessor의 쌍을 순위화할 수 있는 리더보드 시스템을 도입할 수 있으며, 이는 예측 가능성 향상을 위한 전반적인 진행 상황을 보장합니다.



### An Enhancement of Jiang, Z., et al.s Compression-Based Classification Algorithm Applied to News Article Categorization (https://arxiv.org/abs/2502.14444)
Comments:
          11 pages, 5 figures, 1 table

- **What's New**: 이번 연구는 Jiang et al.의 압축 기반 분류 알고리즘을 개선하여 텍스트 문서 간의 의미적 유사성을 탐지하는 한계를 해결합니다. 제안된 개선 사항은 unigram 추출과 최적화된 연결 방법에 중점을 두어 전체 문서 압축에 대한 의존성을 제거합니다. 이를 통해 압축 효율성을 높이고 유사성 탐지를 개선함으로써 더 정교한 텍스트 분류를 가능하게 합니다.

- **Technical Details**: 이 연구에서는 unigram 추출과 gzip 압축을 활용하여 텍스트 문서 간의 유사성을 측정하는 새로운 전략을 개발했습니다. 최적화된 연결 전략을 통해 직접 연결을 피하고 고유한 단어의 집합을 결합하여 중복성을 줄였습니다. 이를 통해 Normalized Compression Distance (NCD) 계산의 정확성이 향상되고, 다양한 데이터셋에서 평균 5.73%의 정확도 향상을 보여줍니다.

- **Performance Highlights**: 실험 결과, 새로운 방법이 긴 문서를 포함하는 데이터셋에서 최대 11%의 정확도 향상을 달성했으며, 특히 레이블 다양성이 높은 복잡한 텍스트 구조에서 개선 효과가 두드러졌습니다. 또한, 이 알고리즘은 계산 효율성을 유지하면서 제한된 리소스 환경에서도 적합하게 설계되었습니다. 이 연구는 텍스트 분류를 위한 강력하고 확장 가능한 솔루션을 제공하며, 경량 전처리 기술을 통해 효율적인 압축을 달성하여 더 정확한 분류를 가능하게 합니다.



### Natural Language Generation (https://arxiv.org/abs/2502.14437)
Comments:
          This is a preprint of the following work: Ehud Reiter, Natural Language Generation, 2024, Springer reproduced with permission of Springer Nature Switzerland AG. The final authenticated version is available online at: this http URL

- **What's New**: 이 책은 자연 언어 생성(NLG)에 대한 광범위한 개요를 제공합니다. AI 기술을 활용하여 영어 및 기타 인류 언어로 텍스트를 생성하는 방법에 대한 개인적인 경험과 통찰력을 반영하고 있습니다. 최신 LLM 혁신에 대한 내용 대신, NLG의 기본 개념과 실제 응용에 초점을 맞추고 있습니다.

- **Technical Details**: 자연 언어 생성은 AI와 자연어 처리(NLP) 기술이 결합된 소프트웨어 시스템에서 사용되어 영어, 중국어, 아랍어와 같은 인류 언어로 텍스트를 생성합니다. 이 시스템들은 입력 데이터를 기반으로 텍스트를 생성하며, 데이터에서 텍스트로 변환하는 시스템(데이터-투-텍스트)과 텍스트를 기반으로 다른 텍스트를 생성하는 시스템(텍스트-투-텍스트)으로 나눌 수 있습니다. NLG는 천년기 예보 생성과 같은 여러 실제 사례에 적용됩니다.

- **Performance Highlights**: NLG는 사용자에게 맞춤화된 정보 제공을 가능하게 하여 대량의 전문 텍스트 생성을 효과적으로 지원합니다. Arria 시스템은 2014년 영국 기상청을 위해 간단한 예보 생성기를 개발하여, 5,000개 위치에 대한 예보를 자동으로 생성하는 것을 목표로 했습니다. 구체적인 위치에 맞춘 예보가 제공되어 사용자에게 더 유용한 정보를 전달할 수 있습니다.



### Early-Exit and Instant Confidence Translation Quality Estimation (https://arxiv.org/abs/2502.14429)
- **What's New**: 본 논문에서는 품질 추정 (Quality Estimation)의 두 가지 주요 과제를 해결합니다. 첫째, 대규모 환경에서 품질 추정의 비용을 줄이는 방법을 모색하고, 둘째 불확실성을 고려한 품질 추정 모델인 Instant Confidence COMET을 제안합니다. 이 모델은 기존 기술의 성능을 유지하면서도 비용을 크게 줄입니다. 또한, Early-Exit COMET 모델은 초기 레이어에서 품질 점수를 계산하여 평가 비용을 절감합니다.

- **Technical Details**: Instant Confidence COMET 모델은 고품질 번역 품질 점수와 불확실성을 동시에 예측합니다. 모델이 출력하는 품질 점수는 예측 오류의 크기를 기준으로 하여 해당 오류의 부정적 값을 신뢰도로 해석합니다. 컴퓨테이션 비용이 기존 COMET 모델과 거의 동일하며, 두 개의 MSE 손실을 합산하여 학습합니다. 이 논문에서는 높은 불확실성을 가진 예측에서 인적 검토가 필요하다는 점을 강조하고 있습니다.

- **Performance Highlights**: 제안된 모델은 평가 및 재정렬 (reranking) 두 상황 모두에서 필요 계산량을 50% 줄이는 동시에 성능 저하는 거의 없이 결과를 도출했습니다. Early-Exit COMET은 다수의 후보 중 최상의 후보를 찾기 위해 upper confidence bound bandit 알고리즘과 결합되어 성능을 극대화합니다. 이 연구 결과 및 모델은 공개적으로 배포되어 향후 연구자들이 활용할 수 있도록 하였습니다.



### Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models (https://arxiv.org/abs/2502.14427)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 텍스트 생성에서 불확실성 정량화(uncertainty quantification, UQ) 방법 중 하나인 Mahalanobis Distance (MD)를 적용하여 새로운 감독된 UQ 방법론을 제안합니다. 기존의 밀도 기반 방법이 생성 작업에서 효과적이지 못했던 문제를 해결하기 위해, 우리는 LLM의 여러 층에서 토큰 임베딩을 추출하고 MD 점수를 계산하여 선형 회귀를 통해 신뢰도 점수를 제공합니다. 실험을 통해 우리의 방법이 기존 UQ 기법보다 현저하게 향상된 성능을 나타내며, 다양한 데이터에 일반화되는 능력을 보였습니다.

- **Technical Details**: 제안된 감독된 방법은 LLM의 여러 층에서 추출한 토큰 임베딩을 기반으로 하며, 이를 통해 MD 점수를 계산하고 해당 특성을 활용하여 선형 회귀를 수행합니다. 이 접근 방식은 각 토큰의 UQ를 평가하는 데 중점을 두며, 생성된 시퀀스의 확률을 보완 효과로 포함하여 최종적인 불확실성 평가를 강화합니다. 이러한 특징들은 기존의 UQ 방법보다 더 경제적이고 동시에 효과적인 예측을 가능하게 합니다.

- **Performance Highlights**: 논문에서 제안된 방법은 11개 데이터 세트에 대한 exhaustive 실험을 통해 성능을 검증하며, 시퀀스 레벨 선택적 생성 및 주장 수준 사실 확인 작업에서 우수한 결과를 도출했습니다. 특히, 본 방법은 계산 효율성과 정확도를 겸비하고 있어, LLM 기반의 다양한 어플리케이션에서 유용하게 사용될 수 있는 가능성을 보여줍니다. 데이터의 도메인 외 상황에서도 강력한 일반화 능력을 갖추고 있는 점이 특히 강조됩니다.



### A Survey on Data Contamination for Large Language Models (https://arxiv.org/abs/2502.14425)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 인해 텍스트 생성 및 코드 합성 분야에서 큰 진전을 보였습니다. 그러나 데이터 오염(data contamination) 문제로 인해 LLM 성능 평가의 신뢰성이 의문시되고 있습니다. 트레이닝 데이터와 테스트 데이터셋 간의 우연한 겹침은 모델 성능을 인위적으로 부풀릴 수 있으며, 이는 LLM의 진정한 일반화 능력을 과대평가하는 결과를 초래할 수 있습니다.

- **Technical Details**: 본 논문에서는 데이터 오염의 정의와 영향을 설명하고, 오염 없는 평가를 위한 세 가지 전략(data updating, data rewriting, prevention-based methods)을 검토합니다. 특히, 동적 벤치마크(dynamic benchmarks)와 LLM 기반 평가 방법에 중점을 두고 다룹니다. 또한 모델 정보 의존성에 따라 오염 탐지 방법을 화이트-박스(white-box), 그레이-박스(gray-box), 블랙-박스(black-box) 접근 방식으로 분류합니다.

- **Performance Highlights**: 데이터 오염의 문제를 해결하기 위해 보다 엄격한 평가 프로토콜이 필요하며, 모범 사례를 따르는 것이 중요합니다. 리뷰에서 제안을 통해 데이터 오염에 대한 미래의 과제를 제시하였으며, 예를 들어 LLM의 학습 해제(unlearning) 방법과 비니스 평가(non-benchmark evaluation) 방법을 포함합니다. 이를 통해 LLM의 책임 있는 개발 방향을 제시하고 있습니다.



### Unstructured Evidence Attribution for Long Context Query Focused Summarization (https://arxiv.org/abs/2502.14409)
Comments:
          24 pages; 21 figures; 5 tables

- **What's New**: 이번 연구는 LLM(대규모 언어 모델)에서 긴 맥락을 바탕으로 unstructured evidence citation(비구조적 증거 인용) 문제를 다루고 있습니다. 기존의 연구들은 일반적으로 구조화된 증거 인용에 초점을 맞추었지만, 본 논문에서는 비구조적 증거 인용의 필요성을 강조하며, 이를 위한 새로운 데이터셋인 SUnsET을 소개합니다. 이 데이터셋은 LLM이 사용자 쿼리에 기반하여 보다 투명하고 신뢰할 수 있는 요약을 제공할 수 있도록 돕기 위한 것입니다.

- **Technical Details**: 연구팀은 SUnsET 데이터셋을 생성해 LLM을 fine-tuning(미세 조정)하여, 달라진 증거 인용 능력과 더욱 관련성 높은 요약의 품질 향상을 목표로 했습니다. SUnsET은 다양한 도메인에서 일반화 가능한 자료로, 실제 문서를 모듈화하여 썼으며, 문서 섹션을 섞는 데이터 증강이 가능하도록 설계되었습니다. 이를 통해 LLM의 positional biases(위치 편향) 문제를 해결하고, 비구조적 증거 인용의 효율을 높였습니다.

- **Performance Highlights**: SUnsET 데이터로 미세 조정된 LLM은 원래 모델 대비 더 효과적이고 사실적으로 일관된 증거를 생성하는 성과를 보였습니다. 본 연구에서 다룬 5개의 서로 다른 LLM과 4개의 다양한 데이터셋에서 실험한 결과, 요약의 질이 향상되고, 다양한 위치에서 증거를 추출할 수 있음을 입증하였습니다. 이 결과는 사용자 쿼리에 대한 보다 정확하고 관련된 응답을 생성하는 데 기여할 것으로 기대됩니다.



### Enhancing Portuguese Variety Identification with Cross-Domain Approaches (https://arxiv.org/abs/2502.14394)
Comments:
          AAAI 2025

- **What's New**: 이 연구에서는 브라질 포르투갈어와 유럽 포르투갈어를 구별하기 위한 교차 도메인 언어 다양성 식별기(LVI)를 개발하였습니다. 이를 통해 다양한 언어 변종을 처리할 수 있는 시스템의 필요성이 강조되었습니다. 특히 브라질 포르투갈어 자료가 온라인에서 점압되어 있어 유럽 포르투갈어의 자원 개발이 필요하다는 점을 다루고 있습니다.

- **Technical Details**: 연구에서는 PtBrVarId 데이터셋을 구성하여 이식성을 갖춘 LVI 분류기 개발을 위한 기초 자료로 사용하였습니다. Broader contexts에서의 모델 성능을 향상시키기 위해 명명된 개체와 테마 콘텐츠를 마스킹(masking)하는 시험도 포함되었습니다. 이 연구에서 제안하는 교육 프로토콜은 LVI 모델의 일반화 성능을 개선하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 연구 결과, 다양한 레벨의 탈어휘화(delexicalization)가 LVI 모델의 전반적인 효과에 미치는 영향을 살펴보았습니다. 최종적으로, 연구팀은 오픈소스 포르투갈어 LVI 모델을 제공하여 향후 연구와 실용적인 응용을 위한 중요한 자원을 개방하였습니다.



### Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessmen (https://arxiv.org/abs/2502.14389)
- **What's New**: 이번 논문에서는 소규모의 오픈소스 Large Language Models (LLMs)를 활용하여 논거 채굴(Argument Mining)을 수행하는 새로운 접근법을 제안합니다. 자동화된 논거 채굴 기법을 통해 학생들의 에세이를 분석하고, 그들의 논거 구성 능력을 개선할 수 있는 실질적인 피드백을 제공합니다. 특히, decoder-only 모델에 대한 연구가 부족한 점을 보완하고, 소규모 모델의 가능성을 탐구하고 있습니다.

- **Technical Details**: 연구에서는 에세이를 세 가지 주요 작업으로 나눠 수행합니다: (1) 에세이를 논거로 분할, (2) 각 논거 유형 분류, 그리고 (3) 질 평가. 세 가지 오픈소스 LLM(Qwen 2.5, Llama 3.1, Gemma 2)의 성능을 비교 분석하며, 기존의 첨단 모델들과의 비교를 통해 결과를 도출합니다. 이러한 모델들은 few-shot prompting 및 fine-tuning 기술을 사용하여 성능 극대화에 기여합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 Feedback Prize 데이터셋을 기반으로 평가되었으며, 세분화(segmentation) 및 유형(classification) 평가에서 기존 기준 모델보다 향상된 성능을 보였습니다. 품질 평가에서는 few-shot prompting 기법이 기준 모델과 유사한 성능을 보여, 소규모 LLM들이 교육적 맥락에서 실시간으로 개인정보를 보호하며 사용할 수 있음을 강조합니다.



### Tradutor: Building a Variety Specific Translation Mod (https://arxiv.org/abs/2502.14385)
Comments:
          AAAI 2025

- **What's New**: 이번 연구에서는 유럽 포르투갈어를 특별히 조정한 첫 번째 오픈소스 기계 번역 모델과 이를 위한 새로운 데이터셋을 소개합니다. 포르투갈어와 같은 고급 리소스를 가진 언어 중에서도 유럽 포르투갈어의 섬세한 뉘앙스가 제대로 반영되지 않은 점에 주목했습니다. 연구자들은 최첨단 비공식 시스템에 가까운 성능을 발휘하는 모델을 개발하여, 리소스가 부족한 언어 변종들을 더욱 포괄적으로 반영할 수 있는 시스템을 마련했습니다.

- **Technical Details**: 본 연구의 핵심 아이디어는 수동으로 주석이 달린 데이터가 부족한 리소스가 적은 언어 변종을 위해 신경 기계 번역(NMT) 모델을 개발하는 것입니다. 연구진은 저자주의 번역 기법을 활용하여 기존의 포르투갈어 텍스트를 고급 언어인 영어로 변환하여 평행 말뭉치(parallel corpus)를 생성했습니다. 이 과정에서 1,719,002개의 문서로 구성된 유럽 포르투갈어를 위한 거대한 평행 데이터셋인 PTradutor를 제공하게 되었습니다.

- **Performance Highlights**: 자동 평가를 통해 연구진은 자신들의 모델이 기존의 포르투갈어 번역 시스템보다 우수하다고 주장하며, 성능 평가에서 주목할 만한 결과를 보였습니다. 추가적으로, 모델이 생성한 번역이 올바른 유럽 포르투갈어인지 확인하기 위해 언어 변종 분류 모델을 사용하여 평가를 진행했습니다. 연구자들은 이러한 접근 방식이 리소스가 부족한 언어 변종의 데이터 확장을 위한 중요한 첫 걸음이 될 것이라고 강조합니다.



### Rumor Detection by Multi-task Suffix Learning based on Time-series Dual Sentiments (https://arxiv.org/abs/2502.14383)
Comments:
          work in progress

- **What's New**: 이 논문에서는 MSuf라는 새로운 다중 작업 접미사 학습 프레임워크를 제안합니다. MSuf는 시간적 이중 감정을 활용하여 루머 감지 및 추적의 효율성을 높입니다. 이 프레임워크는 감정 강도(features)와 관련된 세 가지 모듈로 구성되어 있어, 과거 데이터에서 sentiment 정보를 동적으로 캡처할 수 있습니다. 루머의 유포가 사회적 불안을 초래할 수 있는 점에서, 이러한 연구의 필요성이 강조됩니다.

- **Technical Details**: MSuf 프레임워크는 감정 LLM을 사용하여 감정 강도(SI) 특징을 추출하고 정렬하는 다중 작업 모듈로 구성됩니다. 세 가지 주요 모듈은 이중 감정 데이터의 시간적 흐름을 시각화하고, 이를 원본 텍스트의 단어 임베딩과 융합하여 정렬된 임베딩을 생성합니다. 또한, 정렬된 벡터와 두 개의 하드 프롬프트를 결합하여 루머 감지 및 감정 분석을 수행합니다. 이 프레임워크는 LLM을 최소한의 매개변수 조정으로 활용하여 성능을 향상시킵니다.

- **Performance Highlights**: MSuf는 네 가지 루머 감지 기준점에서 평가되었으며, 기존 감정 기반 방법에 비해 상당한 성능 향상을 보였습니다. F1 점수가 각각 15.3%, 10.9%, 23.4%, 15.6% 상승했습니다. 이러한 결과는 시간에 따른 감정 강도의 변동이 루머의 진실성과 밀접하게 연결되어 있음을 시사합니다. 따라서 MSuf는 감정 강도 분석을 통한 루머 감지의 새로운 패러다임을 제시합니다.



### Affinity and Diversity: A Unified Metric for Demonstration Selection via Internal Representations (https://arxiv.org/abs/2502.14380)
Comments:
          8 pages, 10 figures

- **What's New**: 이번 연구에서는 In-Context Learning (ICL)의 성능 향상을 위해 새로운 평가 지표인 affinity와 diversity를 제안합니다. 이러한 지표는 ICL 모델의 내부 표현을 활용하여 demonstration의 품질을 통합적으로 평가합니다. 기존의 demonstration 선택 방법들이 서로 다른 목표를 가지고 최적화되면서 일관성 있는 결과를 내지 못했던 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 지표인 affinity는 쿼리와 demonstration 간의 코사인 유사도의 평균으로 정의되며, diversity는 demonstration 간의 레이블 토큰 representations의 분산으로 정의됩니다. 연구진은 ICL에 중요한 self-attention head를 식별한 후, 해당 head의 subspace에서 affinity와 diversity를 계산하여 demonstration의 품질을 평가합니다.

- **Performance Highlights**: 실험 결과, affinity와 diversity는 테스트 정확도와 강한 상관관계를 보였으며, 이는 이들 지표가 demonstration 선택에 효과적임을 나타냅니다. 제안된 지표는 다양한 이전 연구와 잘 align 되면서 일관성 있는 선택 방법을 제시하였으며, ICL 성능을 더욱 향상시키는 기초 자료가 될 것으로 기대됩니다.



### A Similarity Paradigm Through Textual Regularization Without Forgetting (https://arxiv.org/abs/2502.14376)
- **What's New**: 이 논문에서는 비슷한 분포의 데이터에 대한 일반화 성능 저하 문제를 해결하기 위해 SPTR(Similarity Paradigm with Textual Regularization)이라는 새로운 방법론을 제안합니다. SPTR는 고안된 텍스트 프롬프트를 활용하여 전처리된 비전-언어 모델(VLM)의 일반 텍스트 지식을 유지하면서 학습할 수 있도록 구성됩니다. 이를 통해 모델은 새로운 클래스에 대한 일반화 가능성을 높일 수 있습니다.

- **Technical Details**: SPTR는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 최적 수송(optimal transport)을 통해 고안된 특징과 조정된 텍스트 특징 간의 근사치를 보장하여 일반 텍스트 지식을 잊지 않도록 합니다. 2) 자연 정렬 점수(natural alignment score)와 적대적 정렬 점수(adversarial alignment score)를 통한 유사성 패러다임을 적용하여 여러 텍스트 프롬프트의 일반 능력을 지속적으로 발휘할 수 있게 합니다.

- **Performance Highlights**: SPTR는 비전-언어 모델에 대한 학습에서 개선된 성능을 보여주며, 다양한 분류 작업에서 기존 프롬프트 학습 방법들과 비교하여 우수한 결과를 기록했습니다. 4444개의 대표 작업과 11111111개의 데이터세트에 대한 광범위한 실험 결과, SPTR는 최신 기술(state-of-the-art) 성능을 달성했음을 보여주었습니다.



### Entropy-UID: A Method for Optimizing Information Density (https://arxiv.org/abs/2502.14366)
Comments:
          5pages, 1 figures, submitting to ACL 2025

- **What's New**: 이번 연구에서는 정보 생성 모델의 효율성을 향상시키기 위해 엔트로피(Entropy)와 균일 정보 밀도(UID) 원칙을 균형 있게 결합한 새로운 토큰 선택 방법인 Entropy-UID를 제안합니다. 이 방법은 생성된 시퀀스 내부의 정보 분포를 더욱 고르게 만들어 시간적으로 정보 밀도가 불균일해지는 상황을 개선합니다. 실험 결과, Entropy-UID는 기존의 GPT-2 모델 대비 낮은 서프리살(surprisal) 및 엔트로피 변동성을 보여 더 균형 잡히고 인간과 유사한 텍스트 생성을 가능하게 합니다.

- **Technical Details**: Entropy-UID는 토큰 선택 과정에서 엔트로피와 서프리살 값을 동시에 최소화하여 정보 밀도를 최적화합니다. 알고리즘은 각 후보 토큰의 엔트로피와 서프리살 지표를 평가하여 이들의 가중치 조합을 최소화하는 토큰을 선택합니다. 본 연구는 문장 생성 및 텍스트 생성 작업에 대해 해당 방법의 효과성을 입증하기 위한 여러 실험을 수행하였으며, Hyperparameter인 α 값을 조정하여 엔트로피와 UID 사이의 균형을 맞췄습니다.

- **Performance Highlights**: 실험 결과, Entropy-UID 방법은 WikiText-2, OpenWebText, WMT 등 다양한 벤치마크 데이터셋에서 높은 성능을 나타냈습니다. 특히 Entropy-UID는 낮은 엔트로피 표준편차(≈2.8)와 안정적인 평균 서프리살(≈5.7)을 기록하여 예측의 불확실성과 정확성 간의 최적의 균형을 유지했습니다. 반면, 단일 목표 최적화 방법은 낮은 엔트로피와 높은 서프리살에 따른 성능 저하가 관찰되었습니다.



### Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests (https://arxiv.org/abs/2502.14359)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델) 평가를 위해 대규모 질문-응답 벤치마크, 인터랙티브 게임, 그리고 인지 테스트의 세 가지 평가 패러다임을 조사합니다. 저자들은 인지 능력을 측정하기 위한 새로운 맞춤형 테스트를 제안하고, 이러한 테스트가 모델 성능과 어떤 상관관계가 있는지를 연구합니다. 이러한 연구 결과는 특히 인지 테스트가 모델 성능을 더 잘 분별할 수 있음을 나타냅니다.

- **Technical Details**: 저자들은 MMLU(대규모 다중 작업 언어 이해 측정) 및 BBH(BIG-Bench Hard)와 같은 대규모 QA 벤치마크와 인터랙티브 게임을 비교합니다. 각 모델 구성은 7B에서 72B 사이의 파라미터를 가진 다양한 공개 LLM로 이루어져 있으며, IFEval 벤치마크에서 평균 70% 이상의 성과를 기록한 모델들로 선정되었습니다. 이 연구는 인지 능력의 계층 구조를 제안하고, 각 능력에 대한 기존 평가 데이터 세트를 방법론적으로 정리합니다.

- **Performance Highlights**: 연구 결과, 대규모 QA 벤치마크가 모델 성능을 정확히 구별하기에는 제한적이라는 점이 드러났습니다. 반면, 인터랙티브 게임은 모델 간의 차이를 더욱 분명하게 드러내었으며, 사용된 인지 능력 테스트와의 상관관계도 유의미하게 나타났습니다. 특히, 작업 기억과 정서적 지능은 게임에서만 강한 상관관계를 보였으며, 이는 LLM의 실제 언어 사용 능력을 평가하는 데 있어 새로운 인터랙티브 벤치마크와 인지 테스트 개발의 필요성을 강조합니다.



### Full-Step-DPO: Self-Supervised Preference Optimization with Step-wise Rewards for Mathematical Reasoning (https://arxiv.org/abs/2502.14356)
- **What's New**: 본 논문에서는 기존 DPO(Direct Preference Optimization)의 한계를 극복하기 위해 새로운 DPO 프레임워크인 Full-Step-DPO를 제안합니다. 이는 수학적 추론을 위한 설계를 기반으로 하여 전체 추론 단계에 대해 단계별 보상을 활용합니다. 이를 통해 기존 접근 방법인 Step-DPO가 간과했던 모든 단계의 정보를 최적화하며, 외부 신호에 의존하지 않고 자동으로 각 단계를 평가할 수 있는 자가 지도 학습 기반 프로세스 보상 모델(Processing Reward Model)을 도입합니다.

- **Technical Details**: Full-Step-DPO는 단계별 보상을 통해 각 단계의 gradient를 동적으로 업데이트하는 새로운 단계별 DPO 손실을 제안합니다. 이 프레임워크는 수학적 문제 해결 과정에서 모든 단계를 최적화하는 접근 방식을 채택하고 있으며, 수학적 데이터셋에 대한 실험을 통해 성능을 검증하였습니다. Self-supervised 방식으로 훈련된 PRM(Processing Reward Model)은 외부 주석을 필요로 하지 않아 효율성이 더욱 향상됩니다.

- **Performance Highlights**: Full-Step-DPO는 다양한 언어 모델에 대해 실시한 광범위한 평가에서 이전의 DPO 및 Step-DPO 기법 대비 우수한 성능을 보여주었습니다. 수학적 추론 작업에서의 개선된 성능은 단계별 최적화 접근법의 효과를 나타내며, 이는 복잡한 다단계 논리적 문제 해결 능력을 강화하는 데 기여하고 있습니다. 기존의 방법들이 해결하지 못했던 문제에 대한 해결책을 제시하여 LLMs(대형 언어 모델)의 수학적 추론 능력을 크게 향상시킵니다.



### SR-LLM: Rethinking the Structured Representation in Large Language Mod (https://arxiv.org/abs/2502.14352)
- **What's New**: 이 연구에서는 SR-LLM이라는 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 구조적 표현(Structured Representation, SR)과 대규모 언어 모델(Large Language Models, LLMs)의 통합을 위해 두 가지 설정을 제안합니다: 트레이닝 프리(training-free)와 트레이닝 디펜던트(training-dependent)입니다. 본 연구는 LLM 성능을 향상시키기 위해 SR의 활용이 필요함을 처음으로 뒷받침하였습니다.

- **Technical Details**: SR-LLM은 SR을 자연어 설명으로 변환하여 LLM이 구조적 정보를 더 잘 이해하고 활용하도록 돕습니다. 트레이닝 프리 접근은 SR을 자연어로 변환하여 비구조화된 입력을 제공합니다. 반면에 트레이닝 디펜던트 접근은 특정 작업에 대해 SR 데이터셋에 대한 감독적 미세 조정을 통해 LLM의 추론 능력을 향상시킵니다.

- **Performance Highlights**: 종합적인 NLP 벤치마크에서 평가 결과, 이 방법이 기존 접근법보다 우수함을 보였습니다. PAWS 데이터셋에서 기존 방법보다 5.18% 성능 저하가 나타난 반면, 우리의 접근법은 각각 +3.17% 및 +12.38%의 성능 개선을 달성하였습니다. 이는 구조적 정보를 효과적으로 통합함으로써 LLM의 추론 능력을 크게 향상시킬 수 있음을 보여줍니다.



### Earlier Tokens Contribute More: Learning Direct Preference Optimization From Temporal Decay Perspectiv (https://arxiv.org/abs/2502.14340)
Comments:
          Accepted by ICLR 2025

- **What's New**: 본 연구에서는 큰 언어 모델(LLMs)과 인간의 선호도를 정렬하기 위한 Direct Preference Optimization (DPO) 방법을 제안합니다. DPO는 강화 학습(RLHF)의 효율적인 대안으로 주목받고 있습니다. 기존 방법들은 보상(reward)의 기여를 균일하게 처리하여 시간적 동력을 간과했는데, 이는 최근의 발전을 반영하지 못했습니다.

- **Technical Details**: 우리는 gamma 파라미터에 의해 조절되는 시간적 감소 요소(temporal decay factor)를 포함한 향상된 선호 최적화(preference optimization) 방법을 제안합니다. 이 동적 가중치 메커니즘은 시퀀스 내 보상의 위치에 따라 각 보상의 영향을 조정합니다. 초기 토큰에 더 중요한 가중치를 부여함으로써 인간의 선호도 변화에 민감하게 대응할 수 있도록 하고, 관련성이 떨어지는 데이터에 대한 과적합(overfitting)을 완화합니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방식이 다양한 모델 아키텍처와 크기에서 AlpacaEval 2에서 5.9-8.8 포인트, Arena-Hard에서 3.3-9.7 포인트 향상됨을 보여주었습니다. 수학적 및 추론 관련 벤치마크(MMLU, GSM8K, MATH)에 대한 추가 실험에서도 일반적인 능력을 저해하지 않으면서 성능이 향상됨을 확인했습니다.



### English Please: Evaluating Machine Translation for Multilingual Bug Reports (https://arxiv.org/abs/2502.14338)
Comments:
          8 Pages, 4 Figures, 3 Tables

- **What's New**: 버그 리포트의 정확한 번역은 글로벌 소프트웨어 개발 협업에서 중요합니다. 이 연구에서는 DeepL, AWS Translate, ChatGPT의 기계 번역(MT) 성능을 평가하였고, Visual Studio Code GitHub 리포지토리의 특정 데이터에 기반하여 분석했습니다. 연구 결과, DeepL이 전반적인 자동 평가 기준에서 가장 높은 성능을 보였으며, 번역의 정확성과 효과를 평가하기 위해 여러 MT 메트릭을 적용했습니다.

- **Technical Details**: 이 연구는 Visual Studio Code GitHub 리포지토리의 english-please 태그가 붙은 버그 리포트를 분석했습니다. 기계 번역의 성능을 평가하기 위해 BLEU, BERTScore, COMET, METEOR, ROUGE와 같은 다수의 측정 기준을 사용하였습니다. 번역 품질을 종합적으로 이해하기 위해 다섯 가지 확립된 품질 메트릭스를 이용하여 세 가지 MT 시스템의 결과를 평가했습니다.

- **Performance Highlights**: DeepL은 다양한 자동 메트릭에서 다른 시스템보다 뛰어난 성능을 보였습니다. AWS Translate는 특히 METEOR에서 경쟁력을 보여주었고, ChatGPT는 주요 메트릭에서 후발주자로 분석되었습니다. 이 연구는 기술적 텍스트의 번역에서 도메인 적응의 중요성을 강조하고, 향후 연구의 기초 자료를 제공합니다.



### Information Types in Product Reviews (https://arxiv.org/abs/2502.14335)
- **What's New**: 이 연구에서는 상품 리뷰 영역에서 활용할 수 있는 24개의 의사소통 목표를 제시하며, 이를 통해 리뷰 데이터에 대한 대규모 분석을 가능하게 하는 제로샷 멀티 레이블 분류기(zero-shot multi-label classifier)를 활용합니다. 연구자는 각 리뷰의 도움이 되는 정도와 감정(sentiment)을 예측하는 데 클래스들의 조합이 주요한 역할을 한다고 밝혔습니다. 또한 이 새로운 분류체계는 리뷰의 의도(intent), 효과성(effectiveness), 그리고 수사적 구조(rhetorical structure)를 분석하는 데에도 기여할 수 있습니다.

- **Technical Details**: 연구에서는 Amazon의 상품 리뷰 데이터 세트에서 샘플링한 문장들을 사용하여, 텍스트의 다양한 의사소통 유형을 체계적으로 정의하였습니다. 이 과정에서 제품, 배송, 판매자 설명을 포함한 기존의 텍스트 유형들을 참고하여, 문장을 수집하고 다양한 유형을 식별했습니다. 최종적으로, 24개의 문장 유형으로 구성된 캠프의 결과로 인해, 제품 리뷰의 텍스트를 더 잘 이해하고 분석할 수 있는 기틀을 마련하였습니다.

- **Performance Highlights**: 대규모 실험을 통해, 연구자들은 특정 문장 유형 레이블만으로도 리뷰의 유용성과 감정 분석 작업에서 강력한 신호를 제공할 수 있음을 입증했습니다. 감정 분석, 리뷰의 효과성, 그리고 구조적 내용을 이해하는 데 있어, 이러한 다양한 문장 유형의 파악은 새로운 설명 가능한 분석 도구로 작용할 수 있습니다. 향후 이 정보를 활용하면 고객이 제품 리뷰를 더 효과적으로 소비할 수 있는 가능성이 열립니다.



### A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics (https://arxiv.org/abs/2502.14333)
- **What's New**: 최근 대규모 언어 모델(LLM) 기술의 발전으로 다단계 추론(multi-step reasoning) 능력을 향상시키는 방식이 주목받고 있습니다. 이러한 접근은 문제 해결을 다단계로 유도하여 더 큰 성과를 이끌어내며, 과거의 훈련 기반 전략보다 더 효과적인 결과를 보여주고 있습니다. 특히 수학 문제를 해결하는 과정에서 다단계 추론의 필요성이 대두되며, 이 논문은 해당 분야의 전반적인 피드백 전략을 조망합니다.

- **Technical Details**: LLM은 주어진 질문 Q에 대해 여러 사고 단계(rm1,…,rmn)와 최종 답변(am)을 생성합니다. 이를 위해 각 단계에 대한 피드백을 활용하는 다양한 전략이 연구되었으며, 특히 과정 보상(process rewards) 모델과 결과 보상(outcome rewards) 모델이 논의됩니다. 이러한 모델들은 LLM이 문제를 해결할 때 특정 단계에서의 올바른 추론을 장려하고 최종 결과의 정확성을 개선하는 데 중요합니다.

- **Performance Highlights**: 이 논문에서는 기존의 프롬프트 기반 전략이 아닌, 훈련 없이도 대응할 수 있는 피드백 사용 전략을 다루며 LLM의 성능을 극대화하려는 의도를 가지고 있습니다. 수학 문제 해결에 중점을 둔 여러 연구를 검토하며 기존의 접근 방식들과 차별화된 다단계 접근 방식을 도입했습니다. 마지막으로, 효율적인 피드백 전략을 통해 LLM의 복잡한 문제 해결 능력을 향상시키려는 목표를 설정하고 있습니다.



### Line Goes Up? Inherent Limitations of Benchmarks for Evaluating Large Language Models (https://arxiv.org/abs/2502.14318)
Comments:
          10 pages

- **What's New**: 이 논문은 최근 큰 언어 모델(LLMs)의 성능이 향상되었다는 주장을 도전하며, 평가 기준(benchmark)의 한계에 대해 논의하고 있습니다. 기존의 평가 방법으로는 LLM의 일반적인 인지 능력을 충분히 측정할 수 없다고 주장하고, 적대적 자극(adversarial stimuli)과 해석 가능성(interpretability) 기법이 대안적 평가 방법으로 효과적임을 설명합니다. 따라서, LLM의 성능을 신뢰할 수 있는 지표로 사용해서는 안 된다고 결론지었습니다.

- **Technical Details**: LLM은 transformer 아키텍처에 기반하여 대량의 텍스트 데이터를 학습하여 자연어와 상호작용합니다. 다양한 테스트를 통해 LLM의 성능을 평가하지만, 이러한 벤치마크는 실제 세계 작업을 반영하지 못할 수 있습니다. 특히, 새로운 벤치마크가 LLM의 훈련 데이터에 포함되어 역할과 가치를 잃는 문제가 발생하며, 이는 Goodhart의 법칙에 의해 설명될 수 있습니다.

- **Performance Highlights**: 연구 결과는 LLM이 많은 언어 및 논리 작업에서 높은 점수를 받을 수 있지만, 이는 실제 인지 능력을 반영하지 않을 수 있음을 보여주었습니다. LLM은 특정 벤치마크에 과도하게 최적화되어 성능이 빠르게 포화되는 경향이 있으며, 실제 작업에 대한 예측력을 방해합니다. 각 기법이 실제 작업과 얼마나 잘 연결되는지에 대한 연구가 부족하여 벤치마크의 유용성에 대한 신뢰가 떨어지고 있습니다.



### ParallelComp: Parallel Long-Context Compressor for Length Extrapolation (https://arxiv.org/abs/2502.14317)
Comments:
          We will release the code soon

- **What's New**: ParallelComp는 대규모 언어 모델(LLM)에서 긴 문맥을 효율적으로 처리하는 새로운 훈련 없는 방법을 제안합니다. 본 논문은 LLM의 문맥 길이를 4K에서 128K로 확장하며, 높은 처리량(throuhput)을 유지하고 당혹도(perplexity)를 보존하는 방법을 설명합니다. 주목할 만한 점은 이 방법이 Flash Attention과 매끄럽게 통합된다는 것입니다.

- **Technical Details**: ParallelComp는 병렬 주의 메커니즘에서 발생하는 주의 편향(attention biases)에 대한 분석을 제공하며, 주의 보정(attention calibration)을 통해 이 문제를 해결하는 전략을 제안합니다. 또한, 초장기 문맥을 관리하기 위한 청크 퇴출(chunk eviction) 전략을 도입하여 A100 80GB GPU에서 128K를 초과하는 문맥을 처리할 수 있도록 합니다. 보조 메커니즘으로는 병렬 KV 캐시 퇴출(parallel KV cache eviction) 기법이 포함되어 있어 문맥 처리 성능을 1.76배 향상시킵니다.

- **Performance Highlights**: ParallelComp는 8K 길이의 문맥을 학습한 8B 모델을 사용하여 GPT-4의 긴 문맥 작업에서 91.17%의 성능을 달성했습니다. 이 방법은 Claude-2 및 Kimi-Chat과 같은 강력한 클로즈드 소스 모델을 능가하는 성능을 보여줍니다. 실험 결과에서, 병렬 주의를 통해 긴 문맥을 보다 효과적으로 처리할 수 있다는 것을 입증하였습니다.



### Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension (https://arxiv.org/abs/2502.14315)
- **What's New**: 이번 연구에서는 CAPTex라는 벤치마크를 소개하여 다국어 대형 언어 모델(mLLMs)이 문화적으로 다양한 절차적 텍스트를 처리하고 이해하는 능력을 평가합니다. 과거 연구들은 문맥을 고려하지 않은 절차적 텍스트 해석의 어려움을 강조하였으며, CAPTex는 이를 보완합니다. 또한, 이 연구는 낮은 자원 언어에서의 성능 저하 현상과 다양한 문화 영역에서의 일관성 없는 성능 문제를 밝힙니다.

- **Technical Details**: CAPTex는 10개의 고유한 카테고리로 구성된 절차들의 큐레이션된 컬렉션과 이를 이해하는 능력을 평가하기 위한 다수의 선택형 질문들, 그리고 해당 절차에 대한 설명을 포함한 대화 교환의 풍부한 말뭉치로 구성됩니다. 더불어, 다양한 자원 가용성을 갖춘 7개 언어(중국어, 일본어, 페르시아어, 힌디어, 인도네시아어, 우르두어, 하우사어)를 선정하여 문화적 문맥에서의 언어 다양성을 촉진하는 데 중점을 두었습니다.

- **Performance Highlights**: 연구 결과, mLLMs는 문화적으로 맥락화된 절차적 텍스트 처리에서 어려움을 겪고, 특히 자원이 부족한 언어에서 뚜렷한 성능 저하를 보였습니다. 모델 성능은 문화적 도메인에 따라 변동하며, 일부 지역에서는 더 큰 어려움을 나타냅니다. 대화형 프레임워크 내에서 다중 선택 과제에서 성능이 향상되는 경향이 있어, 언어 모델의 대화 기반 처리가 직설적인 질문보다 더 효과적임을 시사합니다.



### MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models (https://arxiv.org/abs/2502.14302)
Comments:
          Code and dataset are available at this https URL

- **What's New**: 이번 연구에서는 MedHallu라는 최초의 benchmark를 제안하여 의료 분야에서의 hallucination(환각) 탐지를 위한 기초 자료집을 제공합니다. 이 데이터셋은 PubMedQA에서 유래된 10,000개의 고품질 질문-답변 쌍으로 구성되어 있으며, 체계적인 파이프라인을 통해 환각된 답변이 생성되었습니다. 연구 결과, 최신의 LLM 모델들이 이 이진 환각 탐지 과제에서 높은 성과를 거두지 못하는 것으로 나타났습니다.

- **Technical Details**: MedHallu 데이터셋은 쉽게, 중간, 어렵게 감지하는 수준으로 구분됩니다. 또한, 양방향 내포 클러스터링(bidirectional entailment clustering)을 사용하여 감지하기 어려운 환각은 사실에 더 가까운 의미적 유사성을 가진다는 것을 보여줍니다. 도메인 전문 지식이 포함된 답변 및 '확실하지 않다'는 카테고리를 도입하면, 기존 기준에 비해 정밀도와 F1 점수가 최대 38% 향상되었습니다.

- **Performance Highlights**: 연구에서 의료 도메인에 맞게 조정된 LLM 모델은 일반 목적의 LLM 모델보다 낮은 성과를 보였습니다. 가장 좋은 성능을 보인 모델은 '어려운' 카테고리 환각 탐지에서 F1 점수가 0.625에 불과했습니다. 이 결과는 LLM이 생성하는 정보의 정밀도와 신뢰성을 높이기 위한 보다 세밀한 접근 방법의 필요성을 강조합니다.



### SEA-HELM: Southeast Asian Holistic Evaluation of Language Models (https://arxiv.org/abs/2502.14301)
- **What's New**: SEA-HELM(남동아시아 언어 모델의 포괄적 평가)는 남동아시아(SEA) 언어와 문화에 중점을 둔 새로운 평가 도구로, 기존 LLM의 비교 평가를 위한 철저한 다국어 및 다문화 벤치마크의 필요성이 높아진 가운데 등장했습니다. 이 평가 도구는 NLP Classics, LLM-specifics, SEA Linguistics, SEA Culture, Safety의 다섯 가지 핵심 기둥으로 구성되어 있습니다. 현재 필리핀어, 인도네시아어, 타밀어, 태국어, 베트남어를 지원하며, 사용자가 모델의 다국어 및 다문화 성능을 쉽게 이해할 수 있도록 설계된 SEA-HELM 리더보드도 함께 도입하였습니다.

- **Technical Details**: SEA-HELM은 각 SEA 언어에 대한 언어적 뉘앙스와 문화적 요소를 감안하여 폭넓은 작업을 포함하는 다섯 가지 평가 기둥을 바탕으로 구성되었습니다. 각 기둥은 NLP 고전, LLM 특화, SEA 언어학, SEA 문화, 안전성을 포함합니다. SEA-HELM은 지역 사회의 참여를 통해 현지 원어민이 데이터셋 기획 및 구축의 각 단계에 참여하도록 함으로써 언어적 정확성과 문화적 진정성을 보장합니다.

- **Performance Highlights**: SEA-HELM은 평가가 함께 수행되도록 데이터셋과 LLM 프롬프트를 통합하여 표준화된 비교를 가능하게 하고, 집계한 결과를 언어, 작업 및 모델별로 제시합니다. 이 평가 도구는 기존의 영어 안전 및 NLP 작업을 필리핀어, 인도네시아어, 타밀어, 태국어, 베트남어로 번역하고, 이루어질 사회적 기여와 보다 정교한 언어 진단을 위해 새로운 데이터셋을 개발하였습니다. SEA-HELM은 향후 다른 SEA 언어들에 대한 확장도 계획하고 있으며, 이는 남동아시아 언어에 대한 보다 정확하고 진정한 평가 기준을 제공합니다.



### Drift: Decoding-time Personalized Alignments with Implicit User Preferences (https://arxiv.org/abs/2502.14289)
Comments:
          19 pages, 6 figures

- **What's New**: 이번 논문에서는 개인화된 언어 모델을 구현하기 위한 새로운 프레임워크인 Drift가 소개되었습니다. Drift는 Reinforcement Learning from Human Feedback (RLHF)와는 달리, 수천 개의 예제가 아닌 단지 몇십 개의 예제로 LLM을 개인화할 수 있는 방법을 제시합니다. 이는 사용자 선호를 사전에 정의된 특성(attribute)의 조합으로 모델링하고 디코딩 시간에 개인화된 생성을 가능하게 합니다.

- **Technical Details**: Drift는 복잡한 개인 선호를 해석할 수 있는 간단한 특성으로 분해하는 접근 방식을 채택합니다. 각 특성은 zero-shot 방식을 통해 보상(reward)을 부여받으며, 이러한 차별적 신호는 특성별 보상을 로그 공간(logit space)에 통합하여 생성 과정에서 개인화를 이끌어냅니다. 덕분에 Drift는 모델 업데이트나 그래디언트 계산 없이도 LLM을 개인화할 수 있습니다.

- **Performance Highlights**: 실험 결과 Drift는 50개 예제만으로도 70%의 정확도를 달성하며, 500개 예제로 훈련된 보상 모델보다 나은 성능을 보입니다. 인위적인 인물 데이터셋(Perspective)과 실제 인간 주석 데이터셋(PRISM)에서 Drift는 RLHF 기반 모델들보다 우수한 결과를 나타내었습니다. 이러한 결과는 Drift가 계산적으로 효율적이고 해석 가능함을 보여줍니다.



### Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach (https://arxiv.org/abs/2502.14285)
Comments:
          14 pages,8 figures,4 tables

- **What's New**: 최근 몇 년간 프롬프트 트레이딩이 큰 지식 재산권 문제로 대두되고 있으며, 판매자들이 샘플 이미지를 통해 사용자들을 유인하여 프롬프트 템플릿을 판매하고 있습니다. 본 연구에서는 공격자가 제한된 수의 샘플 이미지만으로 프롬프트 템플릿을 훔칠 수 있는 중요한 보안 취약점을 조사하였습니다. 이를 위해 50개의 템플릿과 450개의 이미지를 포함하는 프리즘(Prism)이라는 프롬프트 훔치기 벤치마크를 도입했습니다.

- **Technical Details**: EvoStealer라는 새로운 템플릿 훔치기 방법을 제안하며, 이는 모델 파인 튜닝(model fine-tuning) 없이 차별적 진화 알고리즘(differential evolution algorithms)을 사용하여 작동합니다. 이 시스템은 미리 정의된 패턴을 바탕으로 다중 모드 대형 언어 모델(multimodal large language models)로 초기 인구 세트를 초기화한 후, MLLMs를 통해 향상된 자손을 반복적으로 생성합니다. 진화 과정에서 EvoStealer는 자손 간의 공통 특징을 식별하여 일반화된 템플릿을 유도합니다.

- **Performance Highlights**: EvoStealer의 강력한 성능은 오픈 소스(INTERNVL2-26B) 및 클로즈드 소스(GPT-4o 및 GPT-4o-mini) 모델에 대한 종합 평가를 통해 입증되었습니다. 테스트 결과, EvoStealer가 훔친 템플릿은 원본과 매우 유사한 이미지를 재현할 수 있으며, 다른 주제에 대해서도 효과적으로 일반화되어 평균 10% 이상의 성능 향상을 보였습니다. 추가적으로, 비용 분석 결과 EvoStealer는 매우 적은 컴퓨팅 비용으로 템플릿 훔치기가 가능함을 확인했습니다.



### EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts (https://arxiv.org/abs/2502.14280)
- **What's New**: 본 연구에서 소개하는 	extbf{EpMAN}은 장기 컨텍스트를 처리하기 위한 새로운 접근법으로, 에피소딕 메모리 모듈을 채택하여 의미적으로 관련된 컨텍스트 조각을 종합적으로 참조합니다. 이 방법은 자기 주의(self-attention) 메커니즘의 한계를 극복하기 위해 개발되었으며, 모델의 훈련 및 생성 과정에서 디코더의 자기 주의를 재조정하는 기능을 갖추고 있습니다. EpMAN은 이전의 RAG 기반 접근법보다 성능이 뛰어나며, 특히 16k에서 256k 토큰 범위에서 보다 강력하고 안정적인 결과를 보여줍니다.

- **Technical Details**: EpMAN은 입력 컨텍스트의 각 조각과 쿼리 간의 상대적 관련성을 평가하고, 이를 바탕으로 자기 주의를 재조정하여 긴 컨텍스트에서 정보를 효과적으로 처리를 가능하게 합니다. 이는 Kahneman의 이중 처리 이론(System 1과 System 2)의 영감을 받아, 빠르고 직관적인 자기 주의와 느리고 계산적인 사고 방식 간의 조화를 이루도록 설계되었습니다. EpMAN은 훈련 중에 관련 조각에 대한 주의를 추정할 때 노이즈를 도입하여 정확성을 높이는 방법도 포함하고 있습니다.

- **Performance Highlights**: EpMAN이 훈련된 LLM은 긴 컨텍스트 시나리오에서 사실 회상 및 단일 홉 질문 답변을 포함한 다양한 벤치마크에서 우수한 성능을 보입니다. 특히, 방해 요소나 혼동이 있는 경우에도, EpMAN을 적용한 모델은 자기 주의 및 RAG 프레임워크를 사용해 훈련된 모델에 비해 좋은 일반화 능력을 보여줍니다. 이 연구는 LLM의 긴 컨텍스트 처리에서의 성능 향상을 위한 효과적인 접근법을 제시하며, 에피소딕 메모리 활용의 중요성을 밝힙니다.



### Fact or Guesswork? Evaluating Large Language Model's Medical Knowledge with Structured One-Hop Judgmen (https://arxiv.org/abs/2502.14275)
Comments:
          15 pages, 11 figures

- **What's New**: 최근 대형 언어 모델(LLM)들이 다양한 다운스트림 태스크에서 널리 사용되고 있지만, 의료 분야에서의 사실적 지식 인지 능력은 여전히 충분히 탐구되지 않았다. 본 논문은 LLM의 본질적 의료 지식을 평가하기 위한 Medical Knowledge Judgment(MKJ) 데이터셋을 소개하며, 이를 통해 모델의 사실적 의료 지식 정확성을 한 단계 향상시키고자 한다. MKJ 데이터셋은 Unified Medical Language System(UMLS)을 바탕으로 구성되어, 의료 지식을 체계적으로 정량화할 수 있는 제어된 프레임워크를 제공한다.

- **Technical Details**: MKJ 데이터셋은 명확하게 정의된 판단 진술로 전환하는 방식으로 의료 지식 트리플을 추출하여 구성되며, 이는 True 또는 False로 분류된다. 이 데이터셋은 LLM이 의료 지식을 판단하는 데 필요한 명확한 환경을 제공하며, 각 트리플은 정확성과 신뢰성을 위해 필터링된다. MKJ는 또한 한 단계의 관계만을 다루며, 다양한 의료 용어를 포함하여 사실적 정확성을 평가하는 데 중점을 둔다.

- **Performance Highlights**: 실험 결과, LLM은 기본적인 사실적 의료 지식의 보존에서 어려움을 겪으며, 특히 드문 질병 카테고리에서 성능 차이를 보인다. LLM의 과도한 자신감과 비율적 결정을 조정하기 위한 retrieval-augmented generation 기법이 이 문제를 해결하는 데 효과적임을 입증하였다. MKJ 데이터셋을 통해 사실적 정확성을 개선하고 불확실성을 줄일 수 있는 방안을 모색할 수 있었다.



### Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models (https://arxiv.org/abs/2502.14272)
Comments:
          Under review

- **What's New**: 이번 논문에서는 작은 언어 모델(SLM)을 인간의 가치에 맞추기 위해, 기존의 대형 언어 모델(LLM)에서 선호 지식을 증류하는 방법에 대한 한계를 다루고 있습니다. 기존 방법들이 쌍(pairwise) 응답 비교에 기반하여 선호 지식을 모델링하는 데 그쳤다면, 본 연구는 또 다른 접근 방식인 Preference-Aligned Distillation (PAD) 프레임워크를 제안합니다. PAD는 교사 모델의 선호 지식을 모든 가능한 선호에 대한 확률 분포로 모델링하여 더 정교한 감독 신호(supervisory signals)를 제공합니다.

- **Technical Details**: PAD 프레임워크는 세 가지 핵심 단계로 구성됩니다. 첫 번째 단계에서는 높은 온도(temperature)를 사용하여 다양한 응답을 샘플링하고, 두 번째 단계에서는 교사와 학생 모델의 보상을 계산하여 내재적 선호(intrinsic preference)를 구성합니다. 마지막으로 세 번째 단계에서는 보상에 기반하여 학생의 내재적 선호 분포를 교사의 선호와 정렬시키는 과정을 포함합니다. 이러한 과정은 언어 모델이 보상 함수로 작용할 수 있음을 보여주며, 모델의 내재적 선호를 효과적으로 캡처합니다.

- **Performance Highlights**: 실험 결과는 PAD가 기존 방법들보다 일관되게 높은 성능을 발휘하며, AlpacaEval 2와 Arena-Hard 벤치마크에서 20% 이상의 성능 향상을 달성했음을 보여줍니다. 또한, MT-Bench에서 	extsc{Gemma} 모델 가족을 이용한 훈련을 통해 학생 모델이 교사 모델을 초월하는 성과를 기록하며, PAD의 효과성을 더욱 입증하였습니다. 이러한 결과는 PAD가 인간의 선호를 보다 정교하게 캡처하고, SLM의 성능을 향상 시키는 중요한 기여를 하고 있음을 나타냅니다.



### PaperHelper: Knowledge-Based LLM QA Paper Reading Assistan (https://arxiv.org/abs/2502.14271)
- **What's New**: 이 논문에서는 연구자들이 과학 문헌을 효과적으로 탐색하고 이해할 수 있도록 도와주는 PaperHelper라는 도구를 소개합니다. 이 도구는 Retrieval-Augmented Generation (RAG) 프레임워크를 사용하여 대규모 언어 모델(LLMs)에서 자주 발생하는 허위 정보(hallucinations)를 최소화합니다. PaperHelper는 문서의 구조적 관계를 설명하는 Mermaid 포맷을 활용하여 사용자 친화적인 인터페이스를 제공하며, 문서의 배치 다운로드를 지원합니다.

- **Technical Details**: PaperHelper는 Streamlit을 사용하여 구축된 엔드투엔드(end-to-end) 파이프라인을 통해 개인화된 문헌 추천, 요약 및 정보 추출을 지원합니다. 이 도구는 여러 쿼리를 생성하여 문서의 정확성과 관련성을 높이는 RAG Fusion과, 문서를 사전 학습하여 RAG 성능을 개선하는 RAFT 방법을 통합하고 있습니다. K-최근접 이웃(top-k) 알고리즘을 이용해 문서와 그에 관련된 지식을 정제합니다.

- **Performance Highlights**: 실험 결과, PaperHelper는 세부 조정된 GPT-4 API를 기반으로 하여 F1 Score 60.04를 달성하였고, 응답 지연 시간은 단 5.8초로 기본 RAG 모델에 비해 7% 더 높은 성능을 보였습니다. 이러한 성능은 PaperHelper가 연구자들에게 과학 문헌을 보다 정확하고 효율적으로 읽을 수 있도록 돕는데 기여할 것입니다.



### MCQA-Eval: Efficient Confidence Evaluation in NLG with Gold-Standard Correctness Labels (https://arxiv.org/abs/2502.14268)
- **What's New**: 이 논문에서는 자연어 생성(NLG) 분야에서의 신뢰도 측정에 대한 평가 프레임워크인 MCQA-Eval을 소개합니다. MCQA-Eval은 기존의 정확도 함수에 의존하지 않고, 다중 선택형 데이터셋에서 제공하는 금 표준 정답 라벨을 활용하여 신뢰도를 평가합니다. 이러한 새로운 접근법으로 인해, 다양한 신뢰도 측정 방법 간의 체계적인 비교가 가능해지며, 효율적이고 신뢰할 수 있는 평가를 제공합니다.

- **Technical Details**: MCQA-Eval은 내부 상태 기반의 화이트 박스(white-box) 신뢰도 측정과 일관성 기반의 블랙 박스(black-box) 신뢰도 측정 모두를 지원하여, 다양한 방법론에 대한 통합된 평가 방법론을 제공합니다. 이 프레임워크는 민감한 정확도 레이블의 노이즈 문제를 해결하며, 기존 평가 방식이 지닌 제한을 극복하는 데 중점을 두고 설계되었습니다. 또한, 우리의 연구는 LLM(대형 언어 모델)과 널리 사용되는 QA(질문-답변) 데이터셋을 기반으로 다양한 실험을 통해 검증되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 MCQA-Eval은 기존 신뢰도 측정 방법들보다 효율적이고 더 신뢰할 수 있는 평가를 제공함을 보고합니다. 평가 과정에서 발생할 수 있는 오류를 최소화하여, 신뢰도 측정 방법의 평가 메트릭과 순위를 더 정확하게 반영합니다. 이를 통해 우리의 프레임워크는 NLG 분야에서 신뢰성 있는 신뢰도 평가를 위한 유용한 도구로 자리잡을 것으로 기대됩니다.



### Does Time Have Its Place? Temporal Heads: Where Language Models Recall Time-specific Information (https://arxiv.org/abs/2502.14258)
- **What's New**: 이 논문에서는 언어 모델이 시간에 따라 변화하는 사실을 처리하는 방법, 특히 특정 attention heads인 Temporal Heads의 존재를 발견했습니다. Temporal Heads는 시간 지식 처리를 주로 담당하며, 여러 모델에서 존재하나 위치가 다를 수 있습니다. 이들을 비활성화하면 시간 특정 지식 회상의 능력이 저하되는 반면, 일반적인 성능에는 큰 영향을 주지 않습니다.

- **Technical Details**: 주요 방법론은 Circuit Analysis를 활용하여 언어 모델의 내부 계산 과정을 재구성하는 것입니다. 이를 통해 특정 attention heads와 feed-forward layers에서 시간 정보의 처리 메커니즘을 분석했습니다. Temporal Heads는 숫자 조건과 텍스트 조건 모두에서 활성화되며, 단순한 숫자 표현을 넘어서는 광범위한 시간 차원을 인코딩하고 있음을 발견했습니다.

- **Performance Highlights**: Temporal Heads의 영향을 분석한 결과, 이들을 비활성화하는 것이 시간에 특정한 정보의 정확도를 크게 감소시켰습니다. 연구 결과는 Temporal Heads가 시간 감응 지식의 인코딩 및 수정에 중요한 역할을 한다는 것을示しています. 이 연구는 언어 모델의 시간 지식 능력을 향상시키기 위한 포괄적인 기초를 제공합니다.



### Effects of Prompt Length on Domain-specific Tasks for Large Language Models (https://arxiv.org/abs/2502.14255)
- **What's New**: 최근 대형 언어 모델(LLM)의 개발이 활발히 진행되고 있으며, 이 모델들이 특정 도메인 과제(예: 금융 감정 분석)에서의 효과성을 규명하기 위한 새로운 연구가 이루어지고 있다. 본 논문은 입력 프롬프트의 길이가 LLM의 도메인 특정 작업 수행 능력에 미치는 영향을 체계적으로 조사하고, 이를 통해 프롬프트 엔지니어링의 중요성을 강조하고 있다. 이는 LLM이 다양한 과제에서 일관된 성과를 내기 위해 어떻게 보조적인 정보를 활용하는지를 이해하는 데 기여할 것이다.

- **Technical Details**: 연구에서는 아홉 개의 도메인 특정 과제에 대해 LLM이 수행하는 실험을 진행하였으며, 각 과제의 프롬프트 길이를 조정하여 모델 성능에 미치는 영향을 분석하였다. 이들 과제에는 통화 정책 이해(MPU), 사용자 의도(UI), 대화 도메인(CD), 쿼리 의도 분류(QIC), 풍자 감지(SD), 감정 식별(EI), 금융 감정 분석(FSA), 기술 시스템 동작(TSB), 그리고 질병 탐지가 포함된다. 또한, 프롬프트의 기본 길이와 짧은/긴 명령 설정을 통해 모델의 반응 품질을 평가하였다.

- **Performance Highlights**: 실험 결과, LLM은 도메인 지식이 부족할 경우 특정 과제 수행에 어려움을 겪는 것으로 나타났으며, 긴 프롬프트가 도메인 배경 지식을 제공하여 성능 향상에 긍정적인 영향을 미친다는 것을 보여주었다. 그러나 이러한 도움에도 불구하고, LLM의 성능은 여전히 인간 평균 값에 미치지 못하는 결과를 보였으며, F1-score가 1.0보다 먼 수치로 나타났다. 이는 도메인 특정 작업에서 성공적인 성능을 위해서는 여전히 인간 전문가의 역할이 필요함을 시사한다.



### Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering (https://arxiv.org/abs/2502.14245)
- **What's New**: 이 논문에서는 'lost-in-retrieval'이라는 문제를 발견하고 해결하기 위한 방법으로 ChainRAG라는 프로그레시브 리트리벌(Progressive Retrieval) 및 리라이터(Rewriter) 방법을 제안합니다. 이 방법은 서브 질문에서 누락된 핵심 엔티티(Key Entity)를 완성하고 관련 문장을 순차적으로 검색하여 최종적으로 정확한 답변을 생성하는 데 초점을 맞추고 있습니다. ChainRAG는 실험을 통해 기존 방법보다 효과적이고 효율적인 성능을 보여주었습니다.

- **Technical Details**: ChainRAG는 문장 그래프(Sentence Graph)를 구성하여 텍스트에서 이름 있는 엔티티를 인덱싱합니다. 사용자가 입력한 질문을 서브 질문으로 분해하고, 각 서브 질문에 대해 LLM을 이용해 관련 문장을 검색합니다. 이 과정은 반복적으로 진행되며, 각 단계에서는 서브 질문의 핵심 엔티티를 완성하고 다음 서브 질문에 대한 답변을 생성하는 방식으로 진행됩니다. 최종적으로 모든 검색된 문장과 서브 질문의 답변을 통합하여 전체 질문에 대한 포괄적인 답변을 제공합니다.

- **Performance Highlights**: ChainRAG는 MuSiQue, 2Wiki, HotpotQA의 세 가지 다중 스텝 QA 데이터세트에서 평가되었으며, GPT4o-mini, Qwen2.5-72B, GLM-4-Plus와 같은 대형 언어 모델을 사용했습니다. 실험 결과, ChainRAG는 기존의 방법들에 비해 항상 효과성과 효율성에서 우수한 성능을 발휘하였고, 다양한 LLM에서 안정적인 성능을 유지했습니다. 이는 ChainRAG의 높은 강건성을 반영합니다.



### Transfer-Prompting: Enhancing Cross-Task Adaptation in Large Language Models via Dual-Stage Prompts Optimization (https://arxiv.org/abs/2502.14211)
Comments:
          17 pages

- **What's New**: 본 연구에서는 Transfer-Prompting이라는 새로운 프레임워크를 제안하여 여러 고수준 목표 간의 균형을 맞추며 대화형 모델의 성능을 향상시키고자 합니다. 이 프레임워크는 두 단계를 포함하는데, 첫 번째 단계에서는 원래의 프롬프트를 개선하여 일반화 능력을 높이는 source prompt construction을, 두 번째 단계에서는 특정 작업에 대한 고득점 source 프롬프트를 기반으로 target prompt generation을 수행합니다.

- **Technical Details**: Transfer-Prompting 프레임워크는 reference LLM과 scorer LLM의 협력을 기반으로 합니다. 초기 최적화 주기에서 reference LLM은 역사적 프롬프트-점수 쌍 및 작업 설명을 바탕으로 후보 프롬프트를 생성하고, scorer LLM은 다차원 메트릭을 사용하여 이 후보 프롬프트의 효과성을 평가합니다. 이러한 피드백 루프는 프롬프트 품질과 작업 결과를 지속적으로 개선하는데 기여합니다.

- **Performance Highlights**: 25개 LLM을 대상으로 한 광범위한 실험을 통해 Transfer-Prompting이 작업 특정 성능을 유의미하게 개선하고, 특히 복잡한 멀티 오브젝트 작업에서의 cross-task adaptation을 향상시키는 잠재력을 확인하였습니다. 결과적으로, Transfer-Prompting은 작업 지시를 따른 비율과 다양한 작업에서의 전반적인 출력 품질을 모두 개선하는 데 효과적이었습니다.



### On-the-fly Preference Alignment via Principle-Guided Decoding (https://arxiv.org/abs/2502.14204)
Comments:
          Accepted to ICLR 2025

- **What's New**: 신규 방법인 OPAD(On-the-fly Preference Alignment via Principle-Guided Decoding)는 모델 추론 과정에서 직접적으로 인간의 선호도와 일치하도록 조정하는 접근법을 제안합니다. 기존의 기술들이 훈련 단계에서 최적화에 주력했다면, OPAD는 미세 조정 없이도 원칙에 기반한 보상 함수를 설계하여 모델의 출력을 수정합니다. 이 방법은 기존의 대규모 언어 모델을 보다 효율적으로 동작할 수 있도록 하여, 추론 시에 원칙을 준수하면서도 재훈련의 계산 비용을 줄입니다.

- **Technical Details**: OPAD는 의미적으로 융통성이 있는 보상 함수를 통해 모델의 출력을 조정하는 방식입니다. 이 방식은 Kullback-Leibler(KL) 발산을 최소화하는 대신 제약된 정책과 제약이 없는 정책 간의 KL 발산을 최대화하는 대체 목표를 사용합니다. 이렇게 함으로써 모델의 반응을 원칙에 맞게 조정하며, 각 시간 단계마다 예상 토큰을 조정하는 튜닝 없는 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과, OPAD는 일반적인 정렬 작업과 개인화된 정렬 작업 모두에서 경쟁력 있는 성과를 기록했습니다. 또한, perplexity, diversity, ROUGE 점수와 같은 자동 평가 메트릭에서도 우수한 성능을 나타내어 OPAD가 목표 원칙에 부합하는 모델 행태 조정에 뛰어나다는 것을 시사합니다. 전통적인 RLHF 방식보다 OPAD가 더 효과적으로 모델의 행동을 조정할 수 있음을 보여주었습니다.



### NLP-AKG: Few-Shot Construction of NLP Academic Knowledge Graph Based on LLM (https://arxiv.org/abs/2502.14192)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 과학 논문 질문 응답의 전문성과 정확성을 개선하기 위해 외부 지식 증강을 활용하는 방법론을 제안합니다. 특히, 기존의 지식 그래프가 논문 및 개념 간의 본질적인 연결을 간과하는 경향을 지적하며, 새로운 지식 그래프 프레임워크를 통해 이러한 문제를 해결하고자 합니다. 제안된 방법은 ACL Anthology에서 추출한 620,353개의 엔티티와 2,271,584개의 관계를 포함한 NLP(Academic Knowledge Graph)를 구축하는 것입니다.

- **Technical Details**: 이 연구에서 제안하는 NLP-AKG는 15가지의 엔티티 유형과 29가지의 관계 카테고리를 포함하여 연구 논문 간의 의미 관계를 포착하는 데 중점을 두고 있습니다. 논문 제목을 검색 인덱스으로 활용하여 NLP 개념과 관련된 학술 논문의 의미 요소를 연결하고, 인용 관계를 추출하여 연구 논문 간의 관계를 맺습니다. LLC의 적은 수의 레이블 데이터만으로도 높은 품질의 지식 그래프를 구성하기 위한 방법론을 개발하였습니다.

- **Performance Highlights**: 제안한 하위 그래프 커뮤니티 요약 방법은 LLM의 질문 응답 정확성을 기존의 검색 증강 기준선 및 LLM과 비교하여 유효성을 검증했습니다. 세 개의 데이터셋(QASPER, NLP-paper-to-QA-generation, 다중 논문 질문 응답 데이터셋)에서 실험 결과, 제안된 방법이 기존 방법들보다 뛰어난 성능을 기록함을 확인했습니다. 이러한 결과는 학술 연구의 복잡한 관계 네트워크를 반영한 새로운 접근법의 가능성을 보여줍니다.



### QUAD-LLM-MLTC: Large Language Models Ensemble Learning for Healthcare Text Multi-Label Classification (https://arxiv.org/abs/2502.14189)
- **What's New**: 이 연구에서는 Healthcare 데이터의 Multi-Label Text Classification (MLTC) 문제를 해결하기 위해 QUAD-LLM-MLTC 접근법을 제안합니다. 이 방법은 변별력이 뛰어난 네 개의 대형 언어 모델(GPT-4o, BERT, PEGASUS, BART)을 결합하여 텍스트 데이터를 효과적으로 분류하는 혁신적인 양상을 보여줍니다. 각 모델은 서로의 강점을 활용하여 문서의 다양한 주제를 정확히 분류할 수 있도록 설계되었습니다.

- **Technical Details**: QUAD-LLM-MLTC는 순차적인 파이프라인에서 작동하며, BERT가 주요 토큰을 추출하고 PEGASUS가 텍스트 데이터를 증강하며, GPT-4o가 분류 작업을 수행하고, BART가 주제 할당 확률을 제공합니다. 이 과정은 0-shot 설정에서 이루어지며, 최종 결과는 앙상블 학습(ensemble learning)을 통해 결합되어 메타 클래스파이러를 통해 처리됩니다.

- **Performance Highlights**: 이 연구의 방법은 세 가지 샘플의 주석 달린 텍스트를 사용하여 평가되었으며, 전통적인 단일 모델 방법들과 비교하여 상당한 성능 개선을 보였습니다. 분류의 F1 점수는 78.17%, 마이크로 F1 점수는 80.16%로, 각 점수의 표준 편차는 각각 0.025 및 0.011을 기록했습니다. 이는 LLM을 이용한 MLTC가 의료 텍스트 데이터의 신속한 분류를 가능하게 하는 효율적이고 확장 가능한 솔루션임을 시사합니다.



### Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction (https://arxiv.org/abs/2502.14171)
- **What's New**: 이 논문은 LLaMA와 같은 오픈 소스 언어 모델이 Theory of Mind (ToM) 관련 정보를 얼마나 잘 포착하고 유지하는지를 연구하고 있습니다. 특히, ToM 관련 요소인 믿음, 욕망, 의도를 명시적으로 조작함으로써 응답의 정렬이 어떻게 향상될 수 있는지를 탐구합니다. 실험 결과, ToM에 기반한 정렬을 포함하면 응답의 품질이 개선되며, LLaMA 3 모델에서 각각 67%와 63%의 승률을 기록합니다.

- **Technical Details**: 이 연구에서는 ToM 관련 정보를 이용해 LLaMA의 내부 표현을 조작함으로써, 일상적 대화에서의 응답 품질을 높이기 위한 기초 연구를 수행했습니다. ToM 이론은 대화자의 정신 상태를 이해하는 데 기초하여, 사람들은 욕구를 표현하고, 타인의 욕구를 믿음으로 이해하며, 의도를 반영하여 발언을 형성한다고 설명합니다. 이를 통해 LLA에서의 더 나은 정렬 응답 생성을 위한 신경 계산 구조의 향상이 기대됩니다.

- **Performance Highlights**: 연구 결과는 ToM 기반 전략이 LLM을 활용한 대화형 에이전트의 정렬을 개선할 수 있는 잠재력을 강조하고 있습니다. 특히, LLaMA 3 모델에서 ToM 관련 정렬이 응답 품질에 긍정적인 영향을 미친다는 사실이 수치적으로 입증되었습니다. 이러한 발견은 AI 시스템이 인간의 정신적 맥락에 더 잘 적응할 수 있도록 도와줍니다.



### LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems (https://arxiv.org/abs/2502.14145)
Comments:
          In submission to INTERSPEECH 2025

- **What's New**: 이 논문은 Spoken Dialogue Systems (SDS)에서 풀듀플렉스(full-duplex) 통신을 달성하기 위한 방안으로, 대화 관리(controller) 역할을 하는 의미 기반 음성 활동 탐지(semantic voice activity detection, VAD) 모듈을 제안합니다. 이 모듈은 0.5B 크기의 경량화된 대형 언어 모델(LLM)을 활용하여, 대화의 턴 전환과 지속을 제어하는 네 가지 제어 토큰을 예측합니다. 사용자 주저 및 Hesitation을 처리하고 의도치 않은 끼어듦을 구분하는 등의 기능을 통해, 더 자연스럽고 효율적인 상호작용을 가능하게 합니다.

- **Technical Details**: 이 SDS는 음성 에코 취소(acoustic echo cancellation, AEC), 음향 VAD, 자동 음성 인식(automatic speech recognition, ASR), 의미 VAD, 핵심 대화 엔진(core dialogue engine, CDE), 텍스트-음성 변환(text-to-speech, TTS) 등 여섯 가지 주요 모듈로 구성되어 있습니다. 의미 VAD는 사용자 상태 및 의도를 이해하기 위해 LLM의 기능을 활용하며, 세밀한 턴 관리와 사용자의 주저를 효과적으로 처리합니다. 사용자 음성은 순차적으로 처리되고, 첫 네 개의 모듈은 실시간 반응을 보장하기 위해 짧은 간격으로 작동합니다.

- **Performance Highlights**: 이 연구는 대화 품질, 안정성, 추론 효율성을 더욱 향상시키기 위한 것으로, 의미 이해를 통해 정확한 대화 관리를 가능하게 하며, 경량화된 LLM을 통해 연산 비용을 절감합니다. 의미 VAD는 사용자의 질문 완성을 감지할 수 있어 불필요한 응답 지연이나 사전 응답을 방지하는 데 도움을 줍니다. DM과 CDE를 독립적으로 최적화함으로써 확장성 또한 확보하였습니다.



### UM_FHS at TREC 2024 PLABA: Exploration of Fine-tuning and AI agent approach for plain language adaptations of biomedical tex (https://arxiv.org/abs/2502.14144)
Comments:
          10 pages, 2 figures, to be published in the 33rd Text REtrieval Conference (TREC 2024) proceedings

- **What's New**: 이번 연구는 TREC 2024 PLABA 트랙에 제출한 논문으로, 생의학 초록을 K8 수준(13-14세 학생)을 위한 간단한 내용으로 변환하는 것을 목표로 하고 있습니다. OpenAI의 gpt-4o 및 gpt-4o-mini 모델을 활용하여 세 가지 접근 방식을 시험하였으며, 그 결과를 통해 모델의 성능을 평가했습니다.

- **Technical Details**: 사용한 방법론에는 기준 프롬프트 공학(baseline prompt engineering), 두 AI 에이전트 접근법(two-AI agent approach), 및 파인튜닝(fine-tuning)이 포함됩니다. 적응성 평가는 질적 지표(단순성, 정확성, 완전성, 간결성에 대한 5점 리커트 척도)와 정량적 가독성 점수(Flesch-Kincaid 등급 수준, SMOG 지수)를 통해 이루어졌습니다.

- **Performance Highlights**: 결과에 따르면, 두 에이전트 접근법과 gpt-4o-mini 모델을 사용하는 기준 프롬프트 공학이 질적 성능에서 우수한 결과를 보여주었습니다. 반면에, 파인튜닝된 모델은 정확성과 완전성에서 탁월했지만 단순성에서는 뒤처지는 경향이 있었습니다. 최종적으로 gpt-4o-mini를 활용한 프롬프트 공학이 반복 개선 전략이나 gpt-4o의 파인튜닝 보다 더 뛰어난 성능을 보여주었습니다.



### Self-Regularization with Latent Space Explanations for Controllable LLM-based Classification (https://arxiv.org/abs/2502.14133)
Comments:
          Pre-print, 15 pages, 4 figures

- **What's New**: 본 논문은 대규모 언어 모델(LLM)에서 발생하는 의도하지 않은(feature) 특징을 식별하고 규제하는 새로운 프레임워크를 제안합니다. 기존의 텍스트 분류 방법은 인간이 설계한 특징에 의존하는 반면, 이 연구는 LLM의 컨텍스트 임베딩을 활용하여 분류 모델의 훈련을 자동화하고 효과적으로 진행하는 방법을 제시합니다. 그러나 이러한 임베딩의 불투명성으로 인해 의도하지 않은 특징을 제거하는 데 어려움이 있었던 점을 강조합니다.

- **Technical Details**: 연구진은 먼저 희소 오토인코더(sparse autoencoder, SAE)를 사전 훈련하여 LLM의 잠재 공간(latent space)에서 해석 가능한 특징을 추출합니다. 이후 이 SAE를 작업에 특화된 데이터셋에서 추가로 미세 조정하여 특정 작업에 적합한 특징을 캡처합니다. 분류 모델 훈련 시, 분류기(weight)와 식별된 의도하지 않은 특징 간의 유사성을 최소화하여 이러한 의도하지 않은 특징의 영향을 제거하는 간단하면서도 효과적인 정규화(regularizer)를 제안합니다.

- **Performance Highlights**: 세 가지 실제 과제, 즉 유해 채팅 감지(toxic chat detection), 보상 모델링(reward modeling), 질병 진단(disease diagnosis)에 대해 제안된 프레임워크를 평가한 결과, 분류기의 일반화 가능성(generality)이 크게 향상되는 것을 확인하였습니다. 결과적으로, 이 연구는 LLM의 잠재 공간에서 해석된 특징을 활용하여 일반화, 공정성(fairness), 개인 정보 보호(priacy) 문제에 대응하는 제어 가능한 텍스트 분류를 선도합니다. 코드와 데이터는 논문 채택 후 공개될 예정입니다.



### Can Community Notes Replace Professional Fact-Checkers? (https://arxiv.org/abs/2502.14132)
- **What's New**: 이번 연구는 소셜 미디어에서의 허위정보와 관련된 전문 사실 확인(fact-checking)과 커뮤니티 노트(community notes) 간의 관계를 조명합니다. 특히, 커뮤니티 노트가 전문 사실 확인자들의 작업에 얼마나 의존하고 있는지를 분석하고, 이러한 노트의 특성을 면밀히 조사합니다. 연구 결과, 커뮤니티 노트의 20%는 전문 사실 확인자의 작업에 명시적으로 의존하고 있으며, 이는 건강 및 정치와 같은 고위험 주제에서 더 두드러집니다.

- **Technical Details**: 연구진은 Twitter/X 커뮤니티 노트를 활용하여, 허위정보에 대한 반박이 이루어지는 패턴과 사실 확인 출처가 커뮤니티 노트에서 인용되는 빈도를 분석했습니다. 그 결과, 커뮤니티 노트가 사실 확인 출처를 인용하는 빈도가 기존 보고된 것보다 최대 5배 더 높은 것으로 나타났고, 특히 더 넓은 내러티브와 연결된 게시물의 경우 이 비율이 두 배로 증가했습니다. 이는 전문적인 사실 확인이 커뮤니티 노트의 효과적 작성을 위해 필수적임을 강조합니다.

- **Performance Highlights**: 커뮤니티 노트가 있는 게시물은 허위정보의 확산을 감소시키는 데 효과적이며, 사용자가 커뮤니티 노트를 통해 허위 정보를 보다 정확하게 식별할 수 있게 됩니다. 연구는 또한 커뮤니티 노트가 게시물 삭제 및 수정의 가능성을 높이고, 그 과정 속도를 가속화하는 데 기여한다는 것을 보여줍니다. 하지만 커뮤니티 노트의 효과에 대한 논란도 여전히 존재하며, 사용자 반응이 부정적일 수 있음을 나타내는 다양한 연구 결과도 있습니다.



### Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Abov (https://arxiv.org/abs/2502.14127)
Comments:
          In-progress preprint

- **What's New**: 이번 논문은 MCQA(Multiple Choice Question Answering)의 포맷과 데이터셋이 LLM 평가에 적합하지 않다는 점을 강조하며, 교육적 접근을 통해 MCQA를 개혁해야 한다고 주장합니다. 연구자들은 MCQA가 주관적 질문이나 생성력을 충분히 테스트하지 못하며, LLM 사용 사례에 비해 부적합하다고 설명합니다. 대신, 모델의 설명과 답변 생성을 기반으로 한 새로운 형식을 제안하여 사용자 요구를 더 잘 반영할 수 있는 평가 방식을 추구합니다.

- **Technical Details**: MCQA는 간단한 "정답 선택" 형태로 인기가 있지만, 데이터셋 설계와 평가 방식에서 여러 결함을 드러냅니다. 논문은 dataset leakage, unanswerable questions, shortcuts, saturation 등의 문제를 제기하며, 이를 해결하기 위해 인공지능 평가에서 사용되는 다양한 교육적 방법론을 도입할 것을 권장합니다. 예를 들면, Item Response Theory(IRT)를 사용하여 질 낮은 MCQs를 제거하고 어려운 질문으로 구성된 새로운 MCQs를 만드는 방법 등이 있습니다.

- **Performance Highlights**: LLM의 평가에서 MCQA의 결함으로 인해 봇의 오류가 발생하며, 이는 MCQA의 포맷과 데이터셋의 결함과 밀접하게 관련되어 있습니다. 예를 들어, LLM의 결과물이 샘플에 따라 변동이 있거나, 특정 옵션이나 문화, 언어에 대한 편향이 나타날 수 있습니다. 연구는 MCQA 포맷과 데이터셋의 개선이 이러한 문제를 보다 효과적으로 측정하고 해결할 수 있다고 제안하며, 교육적 평가방법론을 채택할 필요성을 강조합니다.



### Benchmarking LLMs for Political Science: A United Nations Perspectiv (https://arxiv.org/abs/2502.14122)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 정치적 의사결정에 대한 적용 가능성을 중심으로 하고 있으며, 이를 위해 유엔(UN) 의사결정 과정에 초점을 맞추고 있습니다. 특히, 1994년부터 2024년까지의 유엔 안전 보장 이사회(UNSC) 기록을 포함하는 새로운 데이터 세트를 소개하여 실제 정치적 영향력이 큰 상황에서 LLMs의 활용 가능성을 탐구합니다. 논문은 'United Nations Benchmark (UNBench)'라는 첫 번째 포괄적인 벤치마크를 제안하여 LLM들의 정치 과학 관련 네 가지 연관 작업을 평가합니다.

- **Technical Details**: 이 연구는 정치적 의사결정 과정의 세 가지 단계인 초안 작성(drafting), 투표(voting), 그리고 논의(discussing)에 걸쳐 LLM의 성능을 평가하기 위한 네 가지 작업을 설정합니다. 작업으로는 코-펜홀더 판단(co-penholder judgment), 대표 투표 시뮬레이션(representative voting simulation), 초안 채택 예측(draft adoption prediction), 대표 성명 생성(representative statement generation)이 있습니다. 이러한 작업들은 LLMs가 복잡한 정치적 동역학을 이해하고 시뮬레이션할 수 있는 능력을 평가하고 있습니다.

- **Performance Highlights**: 이 연구 결과는 LLMs가 정치적 작업을 처리하는 데 있어 잠재력과 한계를 드러냅니다. 실험 분석을 통해 현재 LLM이 과거 투표 패턴 및 지정학적 정렬을 분석하여 초안의 통과 확률을 예측하는 데 어느 정도의 정확성을 가질 수 있는지 보여주고 있습니다. 이로 인해 LLMs가 정치 행위의 역동성을 이해하고, 정치적 제약이 있는 상황에서 설득력 있는 제안을 생성하는 데 있어 몇 가지 강점과 함께 개선 가능성 또한 제시되고 있습니다.



### Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility (https://arxiv.org/abs/2502.14119)
- **What's New**: 이 논문에서는 자연어 이해(NLU) 능력의 계층 구조를 제시하고, 어휘 및 문장 수준의 이해를 넘어 담화 수준의 평가로 넘어가는 중요성을 주장합니다. 또한, 담화 이해를 측정하기 위한 진단 도구로서 아나포라 접근성을 제안하며, 동적 의미론에 영감을 받은 평가 데이터셋을 제공합니다. 이 연구는 인간과 LLM(Large Language Models)의 성능을 비교하며, 두 집단이 특정 작업에서 일치하고 다른 작업에서는 차이를 나타냄을 발견했습니다.

- **Technical Details**: 논문은 언어 이해의 세 가지 수준을 설명하고 있습니다: 어휘 수준, 문장 수준, 담화 수준입니다. 특히 담화 수준 이해는 문장 간의 일관된 의미 통합 능력을 의미하며, 이는 정적 진리 조건 표현을 넘어서 동적 의미론으로의 전환을 요구합니다. 본 연구는 LLM들이 다양한 양화사(qualifier) 및 논리적 연결어(logical connectives)의 의미 범주를 기억하고 있는지를 평가하고자 합니다.

- **Performance Highlights**: LLMs는 특정 어휘 항목에 의존하는 경향이 있으며, 이는 인간의 구조적 추상성에 대한 민감성과 대조됩니다. 평가 결과, LLM들은 특정 유형 작업에서 인간과 일치하는 반면, 다른 측면에서는 차이를 보였습니다. 이러한 차별적 성능은 담화 상태를 업데이트하는 데 필요한 세부적인 언어적 요소들에 대한 민감도가 다르기 때문이라고 설명됩니다.



### Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach (https://arxiv.org/abs/2502.14100)
- **What's New**: 이번 연구에서는 external contexts를 적용한 Large Language Models (LLMs)가 내부 지식과 외부 컨텍스트의 균형을 효과적으로 맞추는 'context-robust LLMs' 개념을 제안합니다. 이를 통해 LLM이 내부 지식이 부족할 때 외부 정보를 사용하고, 모순된 정보가 있을 경우 이를 식별할 수 있어야 합니다. 이러한 기능은 인간의 인지 과정과 유사한 방식으로 정보 처리의 정확성을 높이는 데 중요한 역할을 합니다.

- **Technical Details**: 연구의 핵심 기술은 Grft라고 불리는 경량의 gated representation fine-tuning 접근법으로, 두 가지 주요 요소인 게이트 메커니즘과 저차원 표현(adapters)을 포함합니다. 게이트 메커니즘은 '문제가 있는' 입력을 감지하여 필터링하고, 저차원 표현 어댑터는 숨겨진 표현을 조정합니다. Grft는 모델 크기의 0.0004%만을 사용하여 200개 미만의 예시로 효과적인 학습을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 Grft는 LLM이 잘못된 정보와 도움이 되지 않는 컨텍스트를 처리하는 능력을 개선하는 데 효과적임을 입증했습니다. 또한, Grft를 통해 LLM은 유용한 컨텍스트에서의 성능을 유지하면서도, 비유용한 컨텍스트에 대해 더욱 신뢰할 수 있는 반응을 생성할 수 있습니다. 이러한 성과는 LLM의 응용 분야에서도 이점을 제공할 것으로 기대됩니다.



### Retrieving Versus Understanding Extractive Evidence in Few-Shot Learning (https://arxiv.org/abs/2502.14095)
Comments:
          9 pages, 8 figures, Accepted to AAAI 2025 Main Conference (AI Alignment Track)

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LMs)에서 문서 내 증거를 이용한 의사결정의 관계를 분석하며, 예측 오류와 증거 검색 오류 간의 상관관계를 측정합니다. 연구의 주요 질문은 LMs가 입력 문서에서 예측의 근거가 되는 증거를 신뢰성 있게 식별할 수 있는지 여부입니다. 여러 데이터셋을 통해 LMs의 자기합리화(self-rationalization) 능력을 평가하며, 정확도 확보에 중점을 두고 진행됩니다.

- **Technical Details**: 이 연구에서는 MultiRC, SciFact, WikiAttack, Evidence Inference, HealthFC의 다섯 가지 데이터셋을 사용하여, 표준 인간 주석이 있는 추출적 증거(gold-standard extractive evidence)에 대한 LMs의 성능을 분석합니다. 본 연구는 라벨 예측 오류(label prediction error)와 증거 검색 오류(evidence retrieval error) 사이에 높은 상관 관계가 존재함을 발견하였으며, 이는 모델 예측의 신뢰성을 높이는 데 중요한 정보로 작용할 수 있습니다. 실험은 모델의 자기합리화가 라벨 정확도에 미치는 영향과 동작 순서가 성능에 미치는 영향도 조명합니다.

- **Performance Highlights**: 출력 결과에 따르면, LMs는 텍스트 내 증거를 신뢰성 있게 인용할 수 있으며, 자기합리화는 라벨 정확도에 큰 영향을 미치지 않는 것으로 나타났습니다. 또한, 예측 후 설명을 제공하는 방식이 성능에 거의 영향을 미치지 않으며, 예측 오류는 종종 혼란스러운 증거를 잡는 것과 관련이 있습니다. 이러한 결과는 추출적 자기합리화를 활용한 하위 응용 프로그램의 가능성을 시사합니다.



### Navigating Semantic Relations: Challenges for Language Models in Abstract Common-Sense Reasoning (https://arxiv.org/abs/2502.14086)
Comments:
          5 pages, 3 figures, ACM Web Conference 2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 추상적인 상식 추론 능력을 체계적으로 평가하기 위해 ConceptNet 지식 그래프를 활용합니다. 두 가지 프롬프트 접근법, 즉 규정된 정의를 기반으로 하여 가능한 의미 관계를 예측하는 'instruct prompting'과 예제를 활용하는 'few-shot prompting'을 제안합니다. 실험 결과, gpt-4o-mini 모델은 여러 관계를 순위 매길 때 일관된 성능을 보였으나, 한 가지 관계 예측으로 제한할 경우 성능이 크게 저하됨을 보여줍니다.

- **Technical Details**: 연구에서는 ConceptNet 지식 그래프에 기반하여 LLM의 상식 추상 추론을 평가하기 위한 두 가지 프롬프트 접근법을 사용합니다. 'instruct prompting'은 모델이 두 개체 간의 의미 관계를 명시된 이름을 기반으로 예측하도록 요구하며, 'few-shot prompting'은 예제에 기반해 관계를 일반화하는 능력을 평가합니다. 평가 프레임워크는 ConceptNet에서 임의로 샘플링 한 단일 유일한 엣지로 구성된 두 개의 데이터 세트를 사용하여 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, few-shot prompting에서 모델의 정확도는 전체 관계 집합 대신 다섯 개의 관계에서 선택할 때 상당히 향상되었습니다. 그러나 특정 관계에 대해 편향성이 나타났습니다. 이 결과는 상업적으로 사용되는 LLM의 추상적인 상식 추론 능력에서 여전히 큰 격차가 존재함을 시사하며, 신중한 프롬프트 엔지니어링이 성능 개선에 유망함을 강조합니다.



### Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMora (https://arxiv.org/abs/2502.14083)
Comments:
          21 pages, 10 figures, 8 tables

- **What's New**: 이번 연구에서는 도덕적 추론(moral reasoning)의 복잡한 인지 과정(cognitive process)을 다루기 위해 UniMoral이라는 통합 데이터셋을 제안합니다. UniMoral은 심리학적으로 기반을 둔 도덕적 딜레마(moral dilemmas)와 소셜 미디어에서 유래한 자료를 포함하며, 행동 선택, 윤리적 원칙, 기여 요인, 결과에 대한 레이블을 갖추고 있습니다. 또한, 각 주석자의 도덕적 및 문화적 프로필(moral and cultural profiles)도 포함되어 있어 도덕적 추론의 문화적 상대성을 반영합니다.

- **Technical Details**: UniMoral 데이터셋은 아랍어, 중국어, 영어, 힌디어, 러시아어, 스페인어 등 총 여섯 개 언어를 포함하여 다양한 사회문화적 맥락을 포착(Socio-cultural contexts)합니다. 연구는 세 가지 대형 언어 모델(large language models, LLMs)을 사용하여 총 네 가지 작업(action prediction, moral typology classification, factor attribution analysis, consequence generation)을 통해 벤치마크 평가(benchmark evaluations)를 수행하였습니다. 이 과정에서 각 모델의 도덕적 추론 능력 향상에 기여하는 암묵적으로 숨겨진 도덕적 맥락의 중요성을 확인하였습니다.

- **Performance Highlights**: UniMoral을 활용한 연구 결과, LLM의 도덕적 추론 능력을 향상시킬 수 있는 암묵적 맥락의 효과가 드러났습니다. 그러나, 이러한 모델의 도덕적 추론을 더욱 발전시키기 위해서는 보다 전문화된 접근 방식(specialized approaches)이 필요하다는 점도 강조되었습니다. 이 연구는 도덕적 추론을 이해하고 발전시킬 수 있는 중요한 기초 자료를 제공하게 될 것입니다.



### RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression (https://arxiv.org/abs/2502.14051)
- **What's New**: 이번 논문에서는 RocketKV라는 새로운 KV(키-값) 캐시 압축 전략을 제안합니다. 이는 기존의 KV 캐시에서 메모리 대역폭과 저장 용량 요구사항을 줄이는 데 중점을 두고 있으며, 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 SnapKV++을 활용하여 저해상도 KV 캐시 퇴출을 수행하며, 두 번째 단계에서는 하이브리드 어텐션 메커니즘을 통해 동적 KV 토큰 선택을 실시합니다.

- **Technical Details**: RocketKV의 첫 번째 단계인 SnapKV++는 입력 시퀀스의 KV 캐시를 저해상도로 퇴출하는 방법입니다. 이는 적응형 풀링 크기를 도입하고 그룹화된 쿼리 어텐션과 완벽한 호환성을 제공합니다. 두 번째 단계에서는 하이브리드 어텐션을 사용하여 KV 토큰 인덱스를 동적으로 선택하며, 이 과정에서 헤드와 시퀀스 차원 감소를 결합해 어텐션 점수를 근사합니다.

- **Performance Highlights**: RocketKV는 NVIDIA H100 GPU에서 3배의 엔드-투-엔드 속도 향상과 함께 메모리 소비를 31%까지 줄이는 성능을 보여주었습니다. 또한, 다양한 모델과 하위 작업에서도 Full-KV 어텐션에 비견할 만한 정확도를 유지하며, KV 토큰 예산을 256까지 낮출 수 있음을 입증하였습니다.



### Diversity-driven Data Selection for Language Model Tuning through Sparse Autoencoder (https://arxiv.org/abs/2502.14050)
- **What's New**: 이번 연구에서는 기존의 대규모 언어 모델이 인간의 선호에 맞게 올바르게 조정되기 위해 필요로 하는 instruction tuning의 중요성을 강조합니다. 하지만, instruction tuning 데이터는 양적으로 포화 상태에 있어 coreset 데이터 선택이 중요히 여겨지지만 이를 깊이 있게 다룬 연구는 부족합니다. 이 연구는 데이터의 다양성과 복잡성의 동등한 중요성을 간과한 기존 데이터 선택 방법의 한계를 지적하며, 이를 해결하기 위해 다중성을 고려한 데이터 선택 전략을 설계합니다.

- **Technical Details**: 연구에서는 sparse autoencoders를 활용하여 데이터의 다양성을 측정하고, 모델의 행동 해석 가능성 또한 향상시키는 방법을 제안합니다. 이러한 sparse autoencoders는 가장 긴 응답을 선택했을 때의 놀라운 효과를 설명하는 데도 기여합니다. 이 방법론은 데이터 선택 과정에서 모델의 복잡성을 유도하며, 전반적인 모델의 성능 향상에 기여합니다.

- **Performance Highlights**: 실험 결과, 이러한 효과적인 데이터 선택을 통해 교육 받은 모델이 다양한 다른 방법들과 비교할 때 훨씬 더 뛰어난 성능을 나타냄을 입증했습니다. 또한, 교육 비용을 줄이고 모델 행동에 대한 제어력을 더욱 증가시키는 가능성을 보여줍니다. 따라서, 데이터 선택에 대한 접근 방식은 모델의 전반적인 능력을 크게 향상시키는 데 기여합니다.



### Semantic Decomposition and Selective Context Filtering -- Text Processing Techniques for Context-Aware NLP-Based Systems (https://arxiv.org/abs/2502.14048)
- **What's New**: 이번 논문에서는 2가지 새로운 기술을 제안합니다. 첫째, Semantic Decomposition은 입력 프롬프트를 구조화된 정보 스키마로 분해하여 시스템이 쉽게 파싱할 수 있게 합니다. 둘째, Selective Context Filtering은 NLP 기반 파이프라인에 공급되는 불필요한 맥락 정보를 체계적으로 필터링할 수 있도록 합니다. 이 기술들은 LLM-to-system 인터페이스를 동적으로 구현하는 데 유용합니다.

- **Technical Details**: LLM은 최근 몇 년 동안 매개변수 수와 데이터셋 크기와 관련된 다양한 스케일링 법칙 덕분에 점점 더 강력해지고 있습니다. 그러나 이러한 모델은 응답을 생성하기 위한 입력으로 주어진 텍스트 토큰의 문맥 창 크기 제한이 있습니다. 이는 긴 입력 함수를 처리할 때 연산 요구 사항이 기하급수적으로 증가하기 때문이며, LLM이 생성하는 응답의 문맥 일관성을 잃을 가능성을 높입니다.

- **Performance Highlights**: Context-Aware Systems는 LLM 기반 파이프라인을 활용하여 다양한 입력 데이터에 적응하고 이에 반응합니다. 이러한 시스템은 고객 서비스, 의료, 교육, 금융 등 여러 분야에서 활용되고 있으며, LLM의 능력을 통해 사용자에게 맞춤형 응답을 제공합니다. 결과적으로 이러한 시스템은 산업을 혁신할 가능성이 있으며, 스케일 가능하고 개인화된 솔루션을 제공합니다.



### DiffSampling: Enhancing Diversity and Accuracy in Neural Text Generation (https://arxiv.org/abs/2502.14037)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 텍스트 생성 과정에서 발생하는 문제를 해결하기 위해 새로운 디코딩 전략인 DiffSampling을 제안합니다. 기존의 디코딩 방식들은 반복적인 출력이나 훈련 데이터를 재생산하는 경향이 있어, 이것이 성능 저하의 주된 원인으로 지목됩니다. DiffSampling은 확률 분포의 최소 이산 미분(minimum discrete derivative)을 활용하여 적절한 토큰을 선택하는 데 도움을 줍니다.

- **Technical Details**: DiffSampling 방법은 토큰의 확률 분포를 분석하여 임계 질량(critical mass)을 식별하는 데 기반합니다. 이 방법은 서열화된 분포에서 연속적인 확률 간의 가장 큰 차이를 이용하여 부정확한 토큰을 제거하거나 적절하지만 저확률인 토큰의 기회를 높입니다. 세 가지 서로 다른 디코딩 방법을 제안하며, 이는 수학 문제 해결, 극단 요약(extreme summarization), 그리고 다양한 연상 작업(divergent association task)과 같은 여러 과제에서 평가됩니다.

- **Performance Highlights**: DiffSampling은 제안된 세 가지 방법 모두에서 현재의 대안들보다 질과 다양성 측면에서 일관되게 동등하거나 더 높은 성능을 보입니다. 이 연구는 기존 방법들의 한계를 해결하고 적절한 텍스트 생성을 위한 간단하면서도 효과적인 접근 방식을 제시합니다. 다양한 실험 결과는 DiffSampling의 유용성을 증명하고, 이를 통해 LLM의 생성 품질을 크게 향상시킬 수 있음을 나타냅니다.



### Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems (https://arxiv.org/abs/2502.14019)
- **What's New**: 이번 연구는 텍스트 생성 시스템의 출력물들이 점점 더 인간과 유사하게 인식되고 있다는 사실에 주목하고, 이러한 행동이 심리적 의존이나 과신과 같은 해로운 결과를 초래할 수 있다는 우려를 나타냅니다. 이를 완화하기 위한 개입(intervention) 방법론이 부족하다는 점을 지적하며, 시스템 출력의 인간 유사성을 줄이기 위한 다양한 개입 목록을 제시하고자 합니다.

- **Technical Details**: 연구자들은 인지 행동(intervention) 카테고리를 정리하기 위해 NLP, HRI 및 HCI 영역의 선행 문헌을 기반으로 20개의 관련 논문을 선별했으며, 이를 통해 9가지의 개입 유형과 5가지의 시스템 행동 유형을 도출하였습니다. 모델이 생성한 텍스트의 인간 유사한 특성을 정의하고 참가자들이 이러한 출력물의 재작성을 도와주는 크라우드소싱 연구도 실시하였습니다.

- **Performance Highlights**: 이번 연구의 결과는 미래의 텍스트 생성 시스템 설계 및 평가에 있어 중요한 이론적 기초를 제공합니다. 효과적인 개입 방법의 식별 및 평가를 통해 인간 유사성으로 인한 부작용을 줄이고 사용자와의 상호작용에서 긍정적인 경험을 증대시킬 수 있는 잠재력을 보여줍니다.



### MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures (https://arxiv.org/abs/2502.14008)
- **What's New**: 이 논문에서는 대형 언어 모델의 비효율성을 해결하기 위한 새로운 masking learning 패러다임인 MaskPrune을 제안합니다. 기존의 구조적 가지치기 기법은 균일한 구조를 희생함으로써 특정 층의 성능을 유지할 수 있었지만, 이는 전반적인 추론 효율을 저하시켰습니다. MaskPrune은 minimax 최적화 기법을 기반으로 하여 가지치기 마스크를 최적화함으로써 장치 간의 동등성을 유지합니다. 이를 통해 기존의 SOTA 방법을 초월하는 성능을 입증하였습니다.

- **Technical Details**: MaskPrune은 가지치기 마스크와 목표 구조를 동시에 훈련하여 모델의 층 간 구조의 일관성을 유지합니다. 이 과정에서 비차별적인 sparsity 손실을 도입하며, 이는 근접 연산자(proximal operator)를 이용해 최적화됩니다. 또한, 각 층의 치수를 조정하여 전체적인 성능 저하를 최소화하면서 목표 sparsity를 달성하는 데 중점을 둡니다. 최적화 절차는 Transformer 모델의 Multi-Head Attention과 Feed-Forward Network의 주요 구조적 요소를 대상으로 진행됩니다.

- **Performance Highlights**: 실험 결과, MaskPrune은 다양한 sparsity 레벨에서 LLaMA 모델의 성능을 유지하면서 균일한 구조를 보장함이 입증되었습니다. 마스크와 구조의 일관성이 확보됨으로써, 이 접근 방식은 다수의 자연어 처리(NLP) 작업에서 뛰어난 결과를 보였습니다. 이러한 성능 향상은 기존 모델 압축 기법과 비교했을 때 상당한 개선으로 나타났습니다.



### Prompt-to-Leaderboard (https://arxiv.org/abs/2502.14855)
- **What's New**: 이번 논문에서는 Prompt-to-Leaderboard (P2L)라는 새로운 방법론을 제안합니다. P2L은 특정 프롬프트에 대한 리더보드를 생성하여 모델의 성능을 평가하는 방식으로, 사용자와 프롬프트에 따라 성능 변화를 정확하게 반영합니다. 이 방법은 자연어 프롬프트를 입력으로 받아 Bradley-Terry 계수를 출력하여 인간의 선호 투표를 예측할 수 있습니다.

- **Technical Details**: P2L 방법론은 사용자가 두 개의 모델을 쌍으로 비교하여 선택하는 방식으로 작동합니다. 투표 프로세스는 Bradley-Terry 모델을 기반으로 하며, 각 모델이 특정 프롬프트에서 어떻게 성능을 발휘하는지를 정량화합니다. 이를 통해 다양한 프롬프트에 대해 모델 간 성능 차이를 촉발하고, 더욱 세분화된 리더보드를 생성할 수 있습니다.

- **Performance Highlights**: P2L의 효용성은 Chatbot Arena에서 테스트되었으며, 실험 기간 동안 P2L 기반 라우터가 1위에 올라 25점 상승하는 성과를 기록했습니다. 연구 결과 P2L이 기존의 평균화된 리더보드보다 언어 모델의 성능을 더욱 정교하게 포착한다는 점도 확인되었습니다. 또한, P2L의 프롬프트별 평가 능력은 LLM의 힘의 법칙 스케일링(power law scaling) 패턴을 따릅니다.



### Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation (https://arxiv.org/abs/2502.14846)
Comments:
          20 pages, 19 figures, 9 tables, website: this https URL

- **What's New**: 논문은 CoSyn이라는 새로운 프레임워크를 소개하여, 텍스트가 풍부한 멀티모달 데이터를 자동으로 생성하는 방법을 제시합니다. 이 시스템은 텍스트 전용 대형 언어 모델(LLM)을 활용하여 입력된 텍스트 설명을 바탕으로 코드를 생성하고, 이를 통해 합성 이미지와 지침 데이터를 만들어냅니다. CoSyn을 통해 40만 개의 이미지와 270만 개의 비전-언어 지침 조합을 포함한 데이터셋이 구축되었습니다.

- **Technical Details**: CoSyn은 다양한 코드(예: Python, HTML, LaTeX)를 생성할 수 있는 텍스트 전용 LLM의 기능을 활용합니다. 이 구조에 따라 CoSyn은 특정 도메인(예: 영양 성분 라벨)에 대한 지침을 생성하여 해당 도메인에 적합한 합성 이미지를 보여줄 수 있도록 합니다. 이 과정에서 생성된 이미지는 텍스트 기반의 지침 형식으로 저장되며, VLM의 학습에 효과적으로 사용될 수 있습니다.

- **Performance Highlights**: CoSyn으로 생성된 데이터로 훈련한 모델은 경쟁력 있는 오픈 소스 모델들 사이에서 최첨단 성능을 달성했습니다. 특히, 기존의 VLM들은 훈련 데이터의 편향으로 인해 일반화에 어려움을 겪는 반면, CoSyn-400K 데이터를 사용한 모델은 적은 데이터로도 강력한 성능을 발휘했습니다. 또한, CoSyn은 VLM들이 현실 세계의 정보에 접근하고 이해할 수 있도록 하는 잠재력이 있음을 보여줍니다.



### LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models (https://arxiv.org/abs/2502.14834)
- **What's New**: 본 논문에서는 기존의 Large Vision-Language Models (LVLMs)가 최대 128k의 비주얼 및 텍스트 토큰을 처리할 수 있지만, 1,000단어를 초과하는 일관된 출력을 생성하는 데 어려움을 겪는 문제를 다룹니다. 이를 해결하기 위해 LongWriter-V-22k라는 새로운 SFT(Supervised Fine-Tuning) 데이터셋을 소개하는데, 이 데이터셋은 22,158개의 예시를 포함하고 있으며, 각각 여러 입력 이미지, 지침, 최대 10,000단어 범위의 출력이 포함되어 있습니다.

- **Technical Details**: 우리는 입력 이미지에 대한 높은 충실도를 유지하면서 긴 출력을 달성하기 위해 Direct Preference Optimization (DPO) 방법론을 활용합니다. 길고 복잡한 출력을 위한 인간 피드백을 수집하는 것이 비용이 많이 드는 점을 고려하여, IterDPO라는 접근 방식을 제안합니다. 이 방법은 긴 출출을 세분화하여 원래 출력과의 선호 쌍을 형성하기 위한 반복적 수정(iterative corrections)을 사용합니다.

- **Performance Highlights**: LongWriter-V-22k와 IterDPO로 훈련된 7B 파라미터 모델은 MMLongBench-Write 벤치마크에서 우수한 성과를 보여주며, GPT-4o와 같은 더 큰 상용 모델들을 초월하는 결과를 냈습니다. 이는 긴 생성능력의 평가에서 LVLM의 새로운 가능성을 시사합니다.



### Optimizing Model Selection for Compound AI Systems (https://arxiv.org/abs/2502.14815)
- **What's New**: 이번 연구에서는 복합 AI 시스템에서 여러 LLM 호출을 결합하여 복잡한 작업을 수행하는 새로운 모델 선택 방법인 LLMSelector를 제안합니다. 이 방법은 각 모듈에 최적의 LLM 선택이 성능에 미치는 영향을 분석하고, 여러 모듈에서 서로 다른 모델을 사용하는 경우 성능이 크게 향상된다는 점을 발견했습니다. 특히, LLMSelector는 사용되는 API 호출 수를 모듈 수에 비례하여 선형으로 줄일 수 있는 효율적인 프레임워크입니다.

- **Technical Details**: LLMSelector는 두 가지 주요 연구 결과를 바탕으로 설계되었습니다. 첫째, 전체 성능은 각 모듈의 성능이 고정되어 있을 때 모듈의 성능에 따라 단조적으로 변화하며, 둘째, 각 모듈의 성능은 LLM을 통해 정확하게 추정될 수 있습니다. 이를 통해 LLMSelector는 각 모듈에 대해 가장 높은 성능을 보이는 모델을 반복적으로 선택하고 할당하여 전체 성능을 극대화하는 방식으로 작동합니다.

- **Performance Highlights**: 다양한 실험을 통해 LLMSelector는 복합 AI 시스템에서 동일한 LLM을 사용하는 경우보다 5%에서 최대 70%까지 성능 향상을 가져오는 것으로 나타났습니다. 또한, LLMSelector는 프롬프트 최적화에 대한 기존 기법들보다도 우수한 결과를 보여주었으며, 이는 모델 선택의 중요성을 다시 한번 부각시킵니다. 따라서 LLMSelector는 고차원 복합 시스템에서 최적의 성능을 이끌어내기 위한 유용한 도구로 평가됩니다.



### From Knowledge Generation to Knowledge Verification: Examining the BioMedical Generative Capabilities of ChatGP (https://arxiv.org/abs/2502.14714)
Comments:
          26 pages, 6 figures, In Review with a Cell Press Journal

- **What's New**: 본 논문은 LLM 모델이 생성한 생물 의학 정보의 사실 확인을 위한 체계적인 평가 접근 방식을 제안합니다. 구체적으로 질병, 약물, 증상, 유전자 간의 연관성 생성을 포함한 두 가지 핵심 프로세스를 도입했습니다. 이 방식은 ChatGPT를 사용하여 정량적인 평가 기초를 마련하고자 하며, 생물학적 네트워크에서의 정확성을 보장하는 데 중점을 두었습니다.

- **Technical Details**: 정확한 평가를 위해, 다양한 생물 의학 온톨로지를 활용하여 생성된 용어와 연관성을 검증하는 두 가지 주요 작업을 수행했습니다. 첫 번째 작업에서는 GO, DOID, ChEBI 및 증상 온톨로지를 통해 생성된 용어의 정확성을 확인했습니다. 두 번째 작업은 PubMed 데이터베이스를 사용하여 용어 간의 연관성을 검증했으며, 최종적으로 지식의 일관성을 유지하기 위한 세 번째 작업도 수행했습니다.

- **Performance Highlights**: 전반적으로 질병 용어(88%-97%)와 약물 이름(90%-91%) 식별에서 높은 정확성을 기록했고, 유전 정보는 88%-98%의 정확도를 보였습니다. 그러나 증상 용어 식별 정확도는 49%-61%로 상반된 결과를 나타냈습니다. 실험 결과로 도출된 질병-약물 및 질병-유전자 연관성 검증 시 문헌 커버리지는 각각 89%-91%에 달했습니다.



### PEARL: Towards Permutation-Resilient LLMs (https://arxiv.org/abs/2502.14628)
Comments:
          ICLR 2025

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 인 컨텍스트 학습(in-context learning, ICL) 취약성을 분석하고, 시연(demonstration) 순서 변경을 통해 최대 80%의 공격 성공률을 기록하는 자연스러운 공격 방법을 제시합니다. 제안된 PEARL(Permutation-resilient learning) 프레임워크는 분포적으로 강인한 최적화(distributionally robust optimization, DRO)를 기반으로 하여, 입력 순열에 대한 모델 성능을 최적화합니다. 이 연구는 LLaMA-3 모델을 대상으로 하여, 기존의 출력 후처리 방안이 아닌 새로운 접근법으로 LLM의 내재적 강건성을 향상시키는 방법을 제시합니다.

- **Technical Details**: PEARL 프레임워크는 두 개의 네트워크로 구성되어 있습니다: 순열 제안 네트워크(P-Net)와 LLM입니다. P-Net은 입력 순열을 최적 수송(optimal transport, OT) 문제로 변환하여 가장 도전적인 순열을 생성합니다. 이 과정에서 엔트로피 제약 조건을 포함한 Sinkhorn 알고리즘을 사용하여 비트리니 생성을 방지합니다. 두 네트워크는 최소 최대 최적화(minimax optimization)을 통해 상호 최적화를 진행하며, 이는 LLM의 강건성 향상에 기여합니다.

- **Performance Highlights**: 실험 결과, PEARL은 LLaMA-3에 대한 순열 공격을 효과적으로 완화하고, 많은 샷(many-shot) 및 긴 맥락(long-context) 시나리오에서 최대 40%까지 성능 향상을 달성함을 보여주었습니다. 특히, 더욱 적은 샷과 짧은 맥락으로 학습된 PEARL은 새로운 작업에 대해 일반화되는 능력을 증명했습니다. 이러한 결과는 PEARL의 효율성과 일반화 능력을 강조하며, 입력 순열에 대한 강건성을 대폭 개선한다는 점에서 주목할 만합니다.



### Reward Models Identify Consistency, Not Causality (https://arxiv.org/abs/2502.14619)
Comments:
          16 pages

- **What's New**: 이 연구에서는 보상 모델(reward models, RMs)이 대규모 언어 모델(large language models, LLMs)을 인간의 선호와 일치시키는 데 중요한 역할을 한다고 설명합니다. 기존 보상 모델이 후보 출력을 정렬하고 평가하는 방법에 대한 기존 가정을 도전하는 여러 발견을 제시합니다. 특히, 최신 보상 모델은 인과적 정합성(causal correctness)보다 구조적 일관성(structural consistency)을 우선시한다는 점이 강조됩니다.

- **Technical Details**: 연구에서는 LLM의 문제 해결 과정에서 보상 모델이 어떻게 작동하는지를 다루며, 구조적 일관성을 중시함을 보여줍니다. 질문을 생략했을 때 보상 점수에 미치는 영향이 적고, 수치 값을 수정하거나 추론 흐름을 방해했을 때는 보상 점수가 큰 변화를 보인다는 점이 밝혀졌습니다. 보상 모델은 이론적으로 올바른 추론 경로(complete reasoning trajectories)를 필요로 하며, 불완전한 단계를 거치면 보상 할당에 큰 변동이 발생합니다.

- **Performance Highlights**: 결과적으로, 현재의 보상 모델은 진정한 추론 품질을 평가하기보다는 일관성을 판단하는 경향이 있는데, 이는 보상 모델의 기본 한계를 시사합니다. 이 연구는 인과성 인식(causality-aware) 보상 모델로의 전환 필요성을 강조하며, 이는 응답을 순위 매기기보다는 논리적 유효성을 검증하는 데 중점을 둡니다. 연구의 결과는 다양한 아키텍처, 데이터셋 및 태스크에서도 일관되게 나타났습니다.



### A Statistical Case Against Empirical Human-AI Alignmen (https://arxiv.org/abs/2502.14581)
Comments:
          24 pages, 2 figures, 5 tables

- **What's New**: 본 논문은 AI 시스템과 인간 간의 윤리적 정렬을 강조하면서, 단순한 경험적 정렬의 위험성을 경고합니다. 저자들은 경험적 정렬이 통계적 편향을 초래할 수 있다고 주장하며, 이러한 편향이 AI의 성능과 진화에 부정적인 영향을 미칠 수 있음을 시사합니다. 이 논문에서는 경험적 정렬의 대안으로, 정 prescription적 정렬과 경험적 후방 정렬을 제안합니다.

- **Technical Details**: 지금까지 언급된 정렬 개념은 통계적 관점에서 재조명되고 있으며, 저자들은 통계적 가정의 존재를 강조합니다. 특히, 정렬을 위해 가정하는 대표성 샘플과 혼란 변수의 부재가 중요한 요소로 다뤄집니다. 저자들은 경험적 정렬이 AI 모델에 새로운 편향을 도입함으로써 본래의 목표에서 벗어날 수 있음을 주장합니다.

- **Performance Highlights**: 저자들은 LLM(대형 언어 모델)을 활용한 사례 연구를 통해 지식 발견 및 인간 중심 접근의 한계에 대해 논의합니다. 또한, 경험적 정렬이 AI 시스템의 과학적 발견 가능성을 제한한다는 점을 강조합니다. 논문은 경험적 정렬에 따른 편향이 AI의 잠재력을 제약한다고 결론짓고, 대신에 규범적 접근 방식을 채택할 것을 권장합니다.



### ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification (https://arxiv.org/abs/2502.14565)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서 자기 검증(Self-Verification) 기능을 통해 스스로 출력 결과를 수정할 수 있는 새로운 프레임워크, Refine via Intrinsic Self-Verification (ReVISE)를 제안합니다. 기존의 방법들이 외부 검증자에 의존하거나 강화 학습 기반 접근 방식을 사용하는 데 비해, ReVISE는 LLM이 스스로의 추론 과정을 평가하고 수정하는 것을 가능하게 합니다. 이를 위해 ReVISE는 구조화된 커리큘럼을 도입하고, 자기 검증 및 재추론을 위한 편향 학습(preference learning)을 활용합니다.

- **Technical Details**: ReVISE의 핵심 아이디어는 LLM이 출력의 올바름을 평가하여 잘못된 추론 경로를 수정하도록 하는 것입니다. 두 개의 연속적인 훈련 단계를 설계하여 자기 검증과 자기 수정 작업을 효과적으로 학습하게 돕습니다. 첫 번째 단계에서는 올바른 경로와 잘못된 경로의 쌍을 수집하여 자기 검증 능력을 개발하고, 두 번째 단계에서는 잘못된 경로 이후에 올바른 경로가 이어지는 긍정적 샘플과 그 반대의 부정적 샘플을 생성하여 자기 수정을 위한 학습을 진행합니다.

- **Performance Highlights**: 실험 결과, ReVISE는 여러 추론 데이터셋에서 성능을 우수하게 향상시키는 것으로 나타났습니다. 예를 들어, GSM8K 데이터셋에서는 정확도가 27.1%에서 31.1%로 개선되었고, MATH 데이터셋에서는 33.2%에서 36.0%로 향상되었습니다. ReVISE는 외부 피드백 메커니즘에 의존하지 않고도 성능이 향상되며, 기존 방법과 비교해 복잡한 추론 작업에서 일관된 정확도 향상을 보여주었습니다.



### Less is More: Improving LLM Alignment via Preference Data Selection (https://arxiv.org/abs/2502.14560)
- **What's New**: 본 연구에서는 Direct Preference Optimization (DPO) 접근 방식을 통해 대규모 언어 모델(LLM)을 인간의 선호와 정렬하는 데 필요한 데이터 선택(data selection)의 중요성을 강조합니다. 기존의 DPO 연구가 주로 목적 함수(objective function)에 중점을 두었다면, 우리는 데이터 선택 측면에서 DPO를 개선하여 노이즈가 포함된 데이터로 인한 매개변수 축소(parameter shrinkage) 문제를 해결하는 새로운 방법을 제안합니다. 이를 통해 효과적인 데이터 선별이 모델 성능 향상에 기여할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 새로운 마진 극대화(margin-maximization) 원칙을 제안하여 DPO 훈련에서 데이터 세트를 정제하는 방법을 개선합니다. 특히, 외부 보상 마진(external reward margins)과 DPO 내재적 보상 마진(implicit DPO reward margins)을 모두 고려하여 데이터를 선택하는 이중 마진 가이드를 적용합니다. 이 방법을 통해 소음이 많은 데이터로 인한 문제를 해결하고, 보다 효율적인 데이터 선택을 통해 모델 성능을 향상할 수 있음을 이론적으로 입증합니다.

- **Performance Highlights**: 우리는 Ultrafeedback 데이터 세트의 10%만 사용해도 Llama와 Mistral 시리즈 모델에서 AlpacaEval 2.0 벤치마크 기준으로 3%에서 8%의 성능 향상을 이끌어냈습니다. 또한, 25%의 온라인 데이터를 사용한 반복적 DPO에서도 약 3%의 성능 향상과 훈련 시간을 단축하는 성과를 달성했습니다. 이러한 결과는 데이터 선택 전략이 선호 최적화의 진전을 위한 잠재력을 가지고 있음을 보여줍니다.



### Generative adversarial networks vs large language models: a comparative study on synthetic tabular data generation (https://arxiv.org/abs/2502.14523)
Comments:
          12 pages, 7 figures, 5 tables

- **What's New**: 본 논문에서는 제로샷(zero-shot) 방식으로 합성 표 형 데이터(synthetic tabular data)를 생성하기 위한 새로운 프레임워크를 제안합니다. 대규모 언어 모델(LLM)인 GPT-4o를 사용하여, 특정 작업을 위한 파인튜닝(fine-tuning)이나 실제 데이터(real-world data, RWD) 없이도 고충실도(high-fidelity) 표 형 데이터를 생성할 수 있는 능력을 입증합니다. 본 연구는 GPT-4o의 성능을 평가하기 위해 조건부 표 형 생성 적대 네트워크(conditional tabular generative adversarial network, CTGAN)와 비교하였습니다.

- **Technical Details**: 논문에서는 세 가지 공개 데이터셋(Iris, Fish Measurements, Real Estate Valuation)을 사용하여 GPT-4o로 생성된 데이터를 평가했습니다. 이 과정에서 각 데이터셋의 통계적 속성을 기반으로 하여 실제 데이터 없이 직접 생성되는 합성 데이터를 요구하는 프롬프트를 작성했습니다. GPT-4o는 150개의 샘플 크기로 데이터 생성 시 속도의 방향성 및 강도를 유지하며, 95% 신뢰 구간을 정확하게 보존하였습니다.

- **Performance Highlights**: GPT-4o는 샘플 크기를 확대했음에도 불구하고 CTGAN보다 평균, 신뢰 구간, 이변량 상관성 및 데이터의 사생활 보호를 더 잘 유지하였습니다. 특히, 매개변수 간의 관계가 지속적으로 유지되었고, 새로운 상관 특징 관계가 효과적으로 생성되었습니다. 그러나 분포 특성을 더욱 잘 유지하기 위한 추가적인 개선이 필요함을 보여주고 있습니다.



### How Jailbreak Defenses Work and Ensemble? A Mechanistic Investigation (https://arxiv.org/abs/2502.14486)
- **What's New**: 이번 연구는 jailbreak 공격에 대한 두 가지 주요 방어 메커니즘인 safety shift와 harmfulness discrimination을 식별하여, generative task의 표준을 binary classification 문제로 재구성하고, 모델의 거부 경향을 평가합니다. 연구팀은 또한 safety와 helpfulness의 균형을 맞추기 위해 inter-mechanism ensembles와 intra-mechanism ensembles라는 두 가지 앙상블 방어 전략을 개발했습니다. 실험 결과는 이러한 전략이 모델의 안전성을 효과적으로 향상시킨다는 것을 증명했습니다.

- **Technical Details**: 이 논문에서는 내부 모델 최적화를 통한 defend 방법과 query에 대한 효율적 수정 방법을 검토했습니다. 내부 방어 메커니즘은 모델의 생성 프로세스를 개선하거나 입력 쿼리를 수정하여 이루어지며, 시스템 리마인더 및 노이즈 주입 등 다양한 접근 방식을 포함합니다. 연구는 multimodal 환경에서의 방어 메커니즘을 중점적으로 분석하였으며, 이는 기존 언어 기반 방어 연구와는 차별화된 접근입니다.

- **Performance Highlights**: LLaVA-1.5 모델을 사용한 MM-SafetyBench와 MOSSBench 데이터셋에서 다양한 방어 방법에 대한 실험을 진행한 결과, 각 방어 메커니즘의 효과성을 입증했습니다. 연구팀은 28개의 방어 방법을 평가하여 multimodal 방어 연구의 격차를 해소하고, 향후 전략 선택 및 개발에 대한 통찰력을 제공했습니다. 실험을 통해 safety와 helpfulness의 최적 균형을 이루는 방법이 효과적임을 입증했습니다.



### A Macro- and Micro-Hierarchical Transfer Learning Framework for Cross-Domain Fake News Detection (https://arxiv.org/abs/2502.14403)
Comments:
          11 pages, 8 figures

- **What's New**: 본 논문에서는 크로스 도메인 가짜 뉴스 탐지를 위한 새로운 매크로-미세 계층 전이 학습 프레임워크(MMHT)를 제안합니다. 이 프레임워크는 뉴스 콘텐츠와 사용자 참여에서 유용한 표현을 추출하여 가짜 뉴스 탐지 성능을 향상시킵니다. 주요 기여로는 뉴스 콘텐츠에서 진위 관련 및 진위 무관 특징을 분리하고, 사용자 참여의 공유 행동을 고려하여 효과적인 지식 전이를 가능하게 하는 두 가지 모듈을 포함합니다.

- **Technical Details**: MMHT 프레임워크는 두 가지 핵심 모듈로 구성됩니다: (1) 미세 계층 분리 모듈은 뉴스 콘텐츠에서 진위 관련 특성과 진위 무관 특성을 분리합니다. (2) 매크로 계층 전이 학습 모듈은 여러 도메인에서의 공통 사용자 행동을 기반으로 사용자 특성을 추출하고, 이를 통해 참여 특성을 생성하여 지식 전이를 촉진합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 광범위한 실험 결과, 제안된 MMHT 프레임워크는 기존 최첨단 기준보다 평균 4.09% 높은 성능을 보이며, 총 11개의 기준 모델과 비교해 뛰어난 성능을 입증합니다.



### Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignmen (https://arxiv.org/abs/2502.14354)
Comments:
          Under review

- **What's New**: 이번 논문에서는 Multi-Objective Alignment (MOA) 접근 방식에서 발생하는 선호 충돌(preference conflicts) 문제를 해결하기 위한 새로운 방법을 제안합니다. 과거의 Direct Preference Optimization (DPO) 방법은 여러 목표에 따라 LLM의 응답을 최적화하는데 효과적이지만, 서로 다른 목표가 서로 다른 응답을 선호하면서 발생하는 충돌로 인해 최적화 과정이 방해받을 수 있다는 점을 강조하고 있습니다.

- **Technical Details**: 연구진은 이러한 선호 충돌을 해결하기 위해 Pareto-optimal 응답을 구성하는 방법을 제안합니다. 이를 위해 Self-Improvement DPO (SIPO) 프레임워크를 개발하여, LLM이 스스로 Pareto-optimal 응답을 생성하고 선택할 수 있도록 돕습니다. SIPO는 초기 정렬 후, 고품질 응답을 샘플링하고 이를 Pareto-optimality에 맞춰 평가 및 필터링합니다.

- **Performance Highlights**: 실험 결과, SIPO 프레임워크는 HelpSteer와 BeaverTails 데이터셋에서 기존 방법들에 비해 우수한 성과를 보였습니다. 특히, BeaverTails에서 helpful 및 harmless 보상에서 각각 평균 2.1 및 3.0의 개선을 달성하였습니다. 이러한 결과들은 SIPO 프레임워크가 MOA 접근 방식에서의 선호 충돌 문제를 효과적으로 완화할 수 있음을 시사합니다.



### Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems (https://arxiv.org/abs/2502.14321)
- **What's New**: 최근 대형 언어 모델(LLMs)이 추론, 계획, 그리고 의사결정에서 뛰어난 성능을 보여주고 있습니다. 이러한 강점을 바탕으로, 연구자들은 LLM을 다중 에이전트 시스템(MAS)에 통합하여 자연어 상호작용을 통해 협력 또는 경쟁하면서 단일 에이전트 환경을 넘는 작업을 수행하려고 하고 있습니다. 이 설문조사는 LLM 기반 다중 에이전트 시스템의 주요 시스템 레벨 기능과 통신 목표를 조사하며, 통신 전략과 기제의 내적인 작용을 설명합니다.

- **Technical Details**: LLM-MAS의 핵심은 상호 에이전트 간의 통신으로, 이는 에이전트가 아이디어를 교환하고 계획을 조정할 수 있게 합니다. 이 시스템은 의사소통 목표에 의해 구동되며, 다양한 통신 객체와 상호작용하여 작업을 수행하기 위해 복수의 통신 전략과 패러다임을 가집니다. 시스템 레벨 통신과 내부 통신을 두 단계로 나누어 전체적인 워크플로우를 이해할 수 있는 프레임워크를 제시합니다.

- **Performance Highlights**: 전문가의 분석을 통해 잘 조직된 통신이 더 효과적인 다중 에이전트 행동을 이끌어낼 수 있음을 보여 줍니다. 이 연구는 확장성, 보안, 그리고 다중 모드 통합과 같은 문제를 식별하고 향후 연구 방향을 제안하여 다양한 응용 분야에서 더욱 강력하고 지능적인 다중 에이전트 시스템을 발전시키기 위한 기틀을 마련합니다.



### The Impact and Feasibility of Self-Confidence Shaping for AI-Assisted Decision-Making (https://arxiv.org/abs/2502.14311)
- **What's New**: 본 연구는 AI 보조 의사결정에서 인간의 자기신뢰(self-confidence)를 조정하기 위한 개입(intervention)을 제안합니다. 자기신뢰 shaping은 특정 수준으로 신뢰를 조정하여 AI 의사결정의 효율성을 높이는 방법으로, 이는 기존의 AI 중심 개입에서 벗어난 인간 중심의 접근입니다. 연구 결과에 따르면, 자기신뢰 shaping을 통해 인간-AI 팀의 성과를 최대 50%까지 향상시킬 수 있습니다.

- **Technical Details**: 자기신뢰 shaping은 의사결정자의 신뢰를 목표 수준으로 형성하는 데 중점을 두며, 이는 자기신뢰를 실제 성과와 정렬시키는 calibration과 구별됩니다. 연구는 121명의 참가자가 참여한 행동 실험을 통해 수행되었으며, 간단한 기계 학습 모델이 자기신뢰를 약 67%의 정확도로 예측할 수 있음을 보여주었습니다. 또한, 감정(sentiment)과 자기신뢰의 관계를 조사하였고, 텍스트의 미세한 감정 변화가 자기신뢰에 상당한 영향을 미칠 수 있음을 발견했습니다.

- **Performance Highlights**: 자기신뢰 shaping은 AI 추천에 대한 과도한 또는 부족한 의존도를 줄여줍니다. 특히, 최적의 조건하에서 인간-AI 팀 성과를 50% 향상시킬 수 있는 잠재력을 보여주었으며, 이는 AI 보조 의사결정의 효과성을 높이는 데 기여할 수 있습니다. 연구는 현실 세계에서의 자기신뢰 shaping의 배포 가능성을 탐색하며, 감정 수정이 효과적인 자기신뢰 shaping 전략이 될 수 있다는 가능성을 제시하였습니다.



### STeCa: Step-level Trajectory Calibration for LLM Agent Learning (https://arxiv.org/abs/2502.14276)
- **What's New**: 본 논문에서는 Large Language Model (LLM) 기반 에이전트의 학습을 개선하기 위한 새로운 프레임워크인 Step-Level Trajectory Calibration (STeCa)를 제안합니다. 기존 연구는 전문가 시연에서의 행동 클로닝과 탐사 궤적 샘플링을 통한 선호 학습에 주로 집중했으나, 이는 긴 시간의 작업에서 비효율적인 행동이 누적되며 문제를 일으킵니다. 이에 따라, STeCa는 시기적절한 보정의 중요성을 강조하며 자동으로 보정 궤적을 구성하는 필요성을 제기합니다.

- **Technical Details**: STeCa는 탐사 과정에서 단계별 보상 비교를 통해 비효율적인 행동을 식별합니다. 이 프레임워크는 LLM 기반의 성찰을 사용하여 보정된 궤적을 구성하고, 이를 통해 에이전트는 향상된 의사결정 과정에서 학습할 수 있습니다. 보정된 궤적과 성공적인 궤적 데이터를 함께 사용하여 강화 학습을 수행합니다.

- **Performance Highlights**: 광범위한 실험 결과, STeCa는 기존의 방법들보다 상당한 성능 향상을 보였습니다. 특히, 단계별 보정이 에이전트가 여러 작업을 더 견고하게 완료할 수 있도록 도와줍니다. 이 연구 결과는 LLM 기반 에이전트의 학습 성능에 중요한 발전을 가져올 것으로 기대됩니다.



### Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions (https://arxiv.org/abs/2502.14202)
- **What's New**: 이번 연구는 소프트웨어 개발에 대한 대화형 LLM(대형 언어 모델)의 확대 사용과 관련된 보안 문제를 다루고 있습니다. ChatGPT가 개발자에게 맥락에 맞는 보안 정보를 제공하여 안전한 코딩 관행을 촉진할 잠재력을 보이는 것을 기반으로 합니다. 연구에서는 Claude 3, GPT-4 및 Llama 3와 같은 세 가지 주요 LLM의 보안 인식 수준을 평가하였습니다.

- **Technical Details**: 연구에서는 Stack Overflow에서 수집된 질문을 이용하여 LLM 응답의 보안 인식을 평가합니다. 300개의 질문 데이터셋을 두 그룹으로 나누어 보안 취약점을 명시한 경우와 명시하지 않은 경우로 나누어 분석하였으며, LLM이 보안 문제를 능동적으로 인식하고 경고하는지 여부를 살폈습니다. RQ1을 통해 LLM의 보안 인식 정도를 분석하고, RQ2를 통해 보안 경고에 포함된 정보의 유형에 대한 질적 분석을 수행하였습니다.

- **Performance Highlights**: 결과에 따르면, 세 가지 LLM 모두 취약점을 정확하게 감지하고 경고하는 데 어려움을 겪으며, 감지율은 12.6%에서 40%에 불과했습니다. 또한, LLM이 보안 경고를 발행할 때, Stack Overflow 응답보다 취약점의 원인, 악용 및 수정 방법에 대한 정보를 더 많이 제공하는 경향이 있었습니다. 마지막으로, 보안 인식을 높이기 위한 CLI 기반의 프롬프트 도구를 제시하여 LLM 응답의 보안성을 크게 향상시킬 가능성을 보여주었습니다.



### Federated Fine-Tuning of Large Language Models: Kahneman-Tversky vs. Direct Preference Optimization (https://arxiv.org/abs/2502.14187)
- **What's New**: 이 논문에서는 Kahneman-Tversky Optimization (KTO)를 대규모 언어 모델(LLMs)의 세밀 조정 방법으로 평가합니다. KTO는 Direct Preference Optimization (DPO)와 비교하여, 단일 응답 피드백을 처리할 수 있는 유연성을 제공하여 재배포된 데이터 세트에서도 그 성능이 입증됩니다. KTO는 모든 벤치마크에서 DPO를 지속적으로 초과하는 성능을 보이며, 개인 정보 보호를 위한 분산 및 이질적 환경에서도 적합한 방법으로 자리잡고 있습니다.

- **Technical Details**: 연구에서는 Alpaca-7B 모델을 기반으로 하여 KTO와 DPO 두 가지 세밀 조정 방법을 적용하였습니다. KTO 방법은 각 입력에 대해 좋은 응답 또는 나쁜 응답으로만 레이블을 붙이므로, 보다 간단하고 FL 환경에 적응할 수 있는 구조를 가지고 있습니다. 실험 방법론에서는 원래 데이터 할당 및 재배포된 데이터 설정을 통해 KTO의 강인성을 평가하며, 다양한 집계 방법을 사용하여 모델 성능을 비교합니다.

- **Performance Highlights**: KTO는 재배포된 설정에서 DPO의 비적용 상황에서도 뛰어난 성능을 유지하며, 데이터의 비독립 및 이질성 문제를 효과적으로 해결하는 것으로 나타났습니다. KTOO 및 KTOR 설정 모두에서 KTO는 DPO를 능가하며, 자원 요구사항 감소와 개인 정보 보호 개선의 장점을 동시에 제공합니다. 이러한 결과는 KTO가 LLM을 FL 환경에 배치하는 데 있어 강력하고 확장 가능한 세밀 조정 방법으로 자리 잡을 수 있음을 나타냅니다.



### On the logical skills of large language models: evaluations using arbitrarily complex first-order logic problems (https://arxiv.org/abs/2502.14180)
Comments:
          67 pages, 24 figures

- **What's New**: 이번 논문에서는 여러 차원에서 복잡성을 조절할 수 있는 1차 논리문(statement) 생성 방법을 제안합니다. 이 방법을 활용하여 Zermelo-Fraenkel 집합론에 기반한 질문으로 구성된 여러 데이터셋을 자동으로 생성합니다. 생성된 문장은 기본 1차 논리 및 집합론 표기법에 대한 지식만으로 해결할 수 있지만, 높은 난이도로 조정할 수 있는 계획 및 논리적 추리(logical reasoning)를 요구합니다.

- **Technical Details**: 제안된 방법은 생성된 1차 논리문의 복잡성을 여러 기준으로 조절할 수 있는 기능을 가지고 있습니다. 생성된 데이터셋은 1차 논리문이 참인지 거짓인지에 대한 질문으로 구성되어 있으며, 이는 학습된 모델들이 평가를 통해 분석됩니다. 이 논문에서는 DeepSeek-R1 및 OpenAI의 o3-mini와 같은 다양한 대형 언어 모델의 성능 평가도 포함되어 있습니다.

- **Performance Highlights**: 모델의 성능은 생성된 데이터셋을 활용하여 평가되며, 이는 최근 언어 모델들이 1차 논리문을 처리하는 데 있어 어떻게 작동하는지를 보여줍니다. 모든 데이터셋과 생성 코드는 공개되어 있으며, 연구자들이 이 방법론을 쉽게 활용할 수 있도록 지원하고 있습니다. 따라서 이 논문은 논리적 추리에 대한 AI의 이해를 높이기 위한 중요한 기여를 하고 있습니다.



### Giving AI Personalities Leads to More Human-Like Reasoning (https://arxiv.org/abs/2502.14155)
- **What's New**: 이번 연구는 Large Language Models (LLMs)이 사람의 직관적(System 1) 및 의도적(System 2) 추론 과정을 모두 예측할 수 있는지를 탐구합니다. 기존의 AI 모델이 인간의 복잡한 인지 과정을 반영하지 못한다는 문제를 해결하기 위해, 우리는 personality-based prompting과 genetic algorithms를 결합하여 LLM의 응답 분포를 개선하는 방법을 모색했습니다. 이를 통해 AI가 더 인간적인 추론을 할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 연구에서 제안된 접근법은 Natural Language Inference (NLI) 형식을 일반화하여 새로운 추론 과제를 설계했습니다. 이 과제는 System 1과 System 2의 응답을 유도하기 위해 고안되었습니다. 또 기계 학습 모델과의 비교와 함께, personality traits에 따라 LLM의 응답을 이끌어내는 방법을 적용하여 인간의 추론 다양성을 포착했습니다. 이러한 접근은 기존의 정확성 최적화 방식으로 제한된 AI 모델의 한계를 극복하기 위한 것입니다.

- **Performance Highlights**: 결과적으로, 오픈소스 모델인 Llama와 Mistral은 독점 모델인 GPT보다 더 뛰어난 성능을 보였습니다. personality-based prompting이 LLM의 인간 응답 분포 예측 능력을 크게 향상시켰으며, 이는 다양한 추론 스타일과 심리적 프로파일을 포함하는 모델링 기법이 필요함을 시사합니다. 연구는 이러한 접근법이 AI의 인간스러움을 향상시킬 수 있는 가능성을 보여주고 있습니다.



### Investigating Non-Transitivity in LLM-as-a-Judg (https://arxiv.org/abs/2502.14074)
Comments:
          8 pages, 6 figures, 2 tables (30 pages, 11 figures, 8 tables including references and appendices)

- **What's New**: 이 연구에서는 AlpacaEval 프레임워크 내에서 비일관성을 나타내는 비전이성(non-transitivity)의 존재와 그 영향력을 분석하였습니다. LLM(judges)에서 비전이성 선호가 관찰되었으며, 이는 기준 모델의 선택에 민감한 순위를 야기합니다. 이를 해결하기 위해 라운드로빈 토너먼트와 브래들리-테리 모델을 결합하여 보다 신뢰할 수 있는 순위를 생성하는 방법을 제시합니다.

- **Technical Details**: 비전이성은 LLM들이 서로 모순된 선호를 보일 때 발생하며, 이는 특히 고정된 기준 모델을 사용할 경우 순위의 일관성을 해칠 수 있습니다. 새로운 메트릭인 Soft Non-Transitivity Deviation(SNTD)을 도입하여 LLM의 연속 선호에서 비전이성 정도를 측정합니다. 또한, 스위스 기반 반복 매칭(Swim) 토너먼트를 통해 효율성은 유지하면서 라운드로빈 토너먼트의 장점을 활용하는 방법을 제안합니다.

- **Performance Highlights**: 연구 결과, 브래들리-테리 모델을 적용한 라운드로빈 토너먼트가 비전이성의 영향을 감소시켜 보다 견고한 순위를 생성하는 데 기여함을 입증하였습니다. 새로운 방법은 Chatbot Arena와의 상관관계에서 Spearman과 Kendall의 상관계수를 각각 95.0%에서 96.4%, 82.1%에서 86.3%으로 향상시키는 성과를 보였습니다.



### Which Attention Heads Matter for In-Context Learning? (https://arxiv.org/abs/2502.14010)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 in-context learning (ICL) 성능을 주도하는 두 가지 메커니즘, 즉 induction heads와 function vector (FV) heads를 비교합니다. 본 논문은 12개의 언어 모델을 분석하며, FV heads가 ICL 성능의 주요 요인임을 밝혔다고 주장합니다. 특히, 더욱 큰 모델에서 이 경향이 뚜렷하다는 점이 주목할 만합니다.

- **Technical Details**: 연구에서는 induction heads와 FV heads의 차별적인 성격 및 동작 특성을 관찰했습니다. Induction heads는 입력의 반복 패턴을 인식하는 반면, FV heads는 특정 attention heads에서 추출한 ICL 작업의 압축 표현을 처리합니다. 이러한 분석은 70M에서 7B 파라미터를 가진 12개의 transformer 모델을 기준으로 진행되었습니다.

- **Performance Highlights**: 결과적으로, FV heads를 제거했을 때 ICL 작업의 정확도가 크게 저하되는 반면, induction heads 제거는 그 효과가 제한적이었습니다. 또한, induction heads는 훈련 과정에서 FV heads로 발전할 수 있으며, 이는 ICL에 대한 더 복잡한 FV 메커니즘 학습을 촉진한다고 제안됩니다. 이러한 발견은 ICL의 메커니즘 이해뿐 아니라 모델 해석 가능성에 대한 중요한 시사점을 제공합니다.



### Text Classification in the LLM Era -- Where do we stand? (https://arxiv.org/abs/2502.11830)
Comments:
          Pre-print

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs) 기반의 텍스트 분류 시스템의 효율성을 다양한 접근법과 비교했습니다. 특히 제로샷(zero-shot) 분류, 몇 샷(few-shot) 미세 조정, 합성 데이터(synthetic data)를 이용한 분류 방식이 기존의 완전한 라벨 데이터셋을 사용하는 분류기와 어떻게 다른지를 분석했습니다. 8개 언어에 걸쳐 32개의 데이터셋이 사용되었으며, 이는 다양한 언어의 실제 텍스트 분류 시스템을 개발하는데 중요한 지침이 될 것입니다.

- **Technical Details**: 이 연구에서는 제로샷 프롬프트(zero-shot prompting), 몇 샷 미세 조정(few-shot fine-tuning) 및 합성 데이터 기반의 분류(classification)에 대한 실험이 진행되었습니다. 세 가지 오픈 LLMs와 하나의 상용 LLM(GPT-4)을 사용하여 다양한 언어 및 분류 작업 set에서 그 성능을 비교했습니다. 초기에 예측에 대한 많은 설명이 생성되는 경향을 보였으며, 이는 프롬프트를 통해 통제되었습니다.

- **Performance Highlights**: 결과적으로, 제로샷 접근 방식은 감정(classification) 분류에서는 우수한 성과를 보였지만, 다른 작업에서는 다른 접근법에 비해 성과가 떨어졌습니다. 합성 데이터에서 생성된 분류기는 제로샷 LLMs보다 뛰어난 성능을 나타냈으며, 언어별로 성능 차이가 두드러졌습니다. 이 연구는 다양한 언어에서 텍스트 분류 시스템을 개발하는 연구자들에게 많은 인사이트를 제공할 것으로 기대됩니다.



New uploads on arXiv(cs.IR)

### A Survey of Model Architectures in Information Retrieva (https://arxiv.org/abs/2502.14822)
- **What's New**: 본 연구는 정보 검색(Information Retrieval, IR) 모델 아키텍처의 진화를 조사하며, 주요한 두 가지 측면인 특징 추출을 위한 백본 모델(backbone models)과 관련성 추정을 위한 완전한 시스템 아키텍처에 중점을 둡니다. 이 조사에서는 전통적인 용어 기반 방법에서 현대의 신경망 접근 방식까지의 발전을 추적하며, 특히 transformer 기반 모델과 대형 언어 모델(large language models, LLMs)의 영향을 강조합니다.

- **Technical Details**: IR의 주요 목표는 사용자의 정보 요구를 충족하기 위해 적절한 정보 출처를 검색하는 것입니다. 이 과정에서 질의(query)와 문서(document)의 특징 표현을 향상시키는 방법이 क고 있으며, 기계 학습 기반 학습 순위(Learning to Rank, LTR) 기법과 같은 최신 방법이 도입되었습니다. LLM은 특징 추출과 관련성 추정을 수행하는 데 강력한 성능을 보이며, 이는 다양한 네트워크 아키텍처 역사의 발전을 기반으로 하고 있습니다.

- **Performance Highlights**: 최근 LLM의 발전은 IR 분야에 혁신을 가져왔으며, 복잡한 질의를 다루고 다양한 데이터 유형을 처리하는데 도전 과제를 제시하고 있습니다. 이 조사에서는 아키텍처 혁신, 성능 최적화, 멀티모달 데이터 처리와 같은 최신 문제를 다루고 있으며, 로봇공학이나 단백질 구조 발견과 같은 기존 검색 패러다임을 넘어서는 새로운 응용 분야에 적응하는 방법에 대해 논의합니다.



### A Multi-Agent Perspective on Modern Information Retrieva (https://arxiv.org/abs/2502.14796)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 출현이 정보 검색(Information Retrieval, IR) 분야에 미치는 영향을 다루고 있습니다. LLM 기반의 에이전트가 사용자 정보 탐색 작업에 기여할 수 있으며, 이러한 에이전트를 쿼리 에이전트(Query Agent), 문서 에이전트(Document Agent), 순위 에이전트(Ranker Agent)로 나누어 다루고 있습니다. 이 연구는 이들 에이전트 간의 상호작용이 시스템 성능에 미치는 중요한 영향을 밝히고 있습니다.

- **Technical Details**: 논문에서는 쿼리 에이전트, 문서 에이전트, 순위 에이전트의 세 가지 유형의 에이전트를 기반으로 한 다중 에이전트 검색 시스템의 구성을 제안합니다. 각 에이전트는 사용자와 문서 간의 정보 요구사항을 충족시킴과 동시에, 문서 작성 및 편집, 순위 매김 기능을 수행합니다. 연구에서 사용된 접근 방식으로는 레퍼런스 기반 방법(TF.IDF), 임베딩 기반 방법(semantic), LLM 기반 방법이 포함됩니다.

- **Performance Highlights**: 실험 결과, 서로 다른 유형의 쿼리 에이전트와 순위 에이전트가 결합될 경우 검색 효과성이 감소한다는 것을 발견했습니다. 예를 들어, 쿼리 에이전트가 의미 기반이고 순위 에이전트가 LLM 기반인 경우, 두 에이전트가 동일한 유형일 때보다 검색 성과가 떨어지는 것을 확인하였습니다. 또한, 문서 에이전트와 순위 에이전트 간의 미스얼라인한 설정이 성과에 중대한 영향을 미칠 수 있음을 강조하고 있습니다.



### EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration (https://arxiv.org/abs/2502.14735)
Comments:
          9 pages, 6 figures, accpeted by WWW 2025

- **What's New**: EAGER-LLM은 LLM(대형 언어 모델)을 기반으로 하는 새로운 생성 추천 프레임워크로, 내재적(endogenous) 및 외재적(exogenous) 행동과 의미 정보를 비침입적으로 통합합니다. 이 시스템은 사용자 행동 및 추천 데이터에서의 협업 신호를 원활하게 이해하고 통합할 수 있도록 설계되었습니다. EAGER-LLM은 기존 LLM 기반 추천 시스템이 직면한 여러 문제를 해결하는 접근 방식을 제공합니다.

- **Technical Details**: EAGER-LLM의 도입에서 주요한 요소는 이중 소스 지식 풍부 항목 색인(Dual-source Knowledge-rich Item Indices)과 비침해적 다중 스케일 정렬 재구성 과제(Non-Invasive Multiscale Alignment Reconstruction Tasks)입니다. 이를 통해 모델은 외부 신호와 의미 신호를 보다 효율적으로 처리하고 이해할 수 있게 됩니다. 또한, 모델의 추천 성능과 학습 능력을 미세 조정하는 어닐링 어댑터(Annealing Adapter)를 도입하여 전체 모델의 성능을 향상시킵니다.

- **Performance Highlights**: EAGER-LLM은 세 가지 공개 벤치마크에서 진행된 엄격한 실험을 통해 기존 방법들에 비해 우수한 성능을 입증하였습니다. 특히 추천 정확도를 높이면서도 대화 및 설명 생성 능력을 유지하는 데 성공하며, 이는 모델이 협업 및 의미 신호에 대해 깊이 있는 이해를 가능하게 하는 중요한 기여를 보여줍니다.



### Efficient AI in Practice: Training and Deployment of Efficient LLMs for Industry Applications (https://arxiv.org/abs/2502.14305)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 효율성과 성능을 유지하면서도 작은 언어 모델(SLM)을 훈련시키는 방법과 통찰을 제시합니다. 특히, 지식 증류(knowledge distillation)와 양자화(quantization), 가지치기(pruning) 기법을 활용하여 SLM의 훈련 비용과 지연 시간을 크게 줄일 수 있음을 강조합니다. 이를 통해 다양한 실제 사례에서 활용할 수 있는 고성능 효율 모델을 개발할 수 있습니다.

- **Technical Details**: SLM의 학습을 위해 지식 증류와 모델 압축 기법을 사용하며, 대형 LLM의 내부 지식을 바탕으로 SLM을 생성합니다. 이 과정에서는 흰상자(white-box) 및 검은상자(black-box) 증류 방식, 양자화 및 희소화(sparsification) 기술이 포함됩니다. 특히, SLM은 LLM이 제공하는 성능을 대부분 유지하면서도 훈련과 서빙에 필요한 비용과 시간을 효과적으로 줄입니다.

- **Performance Highlights**: 실제 사용 사례에서 SLM을 통해 예측 태스크의 경우 모델 크기를 20배 이상 줄였음에도 불구하고 품질 저하는 없었습니다. 또한, 추론 추상화(reasoning task) 모델에 대해 5배 이상 압축하여도 유사한 품질을 유지하는 성과를 보였습니다. 이러한 결과는 고속 처리와 저지연 요구 사항을 만족하는 시스템에서 활용될 수 있는 매우 실용적인 접근법을 제시합니다.



### An Evaluation of Sakana's AI Scientist for Autonomous Research: Wishful Thinking or an Emerging Reality Towards 'Artificial General Research Intelligence' (AGRI)? (https://arxiv.org/abs/2502.14297)
Comments:
          16 pages

- **What's New**: AI의 새로운 발전 단계인 Artificial General Research Intelligence (AGRI)의 중심에는 AI가 자율적으로 연구를 수행할 수 있는 능력이 있습니다. Sakana.ai의 AI Scientist는 연구 아이디어 생성, 실험 설계 및 결과 분석 등 연구 라이프사이클을 자동화할 수 있다고 주장하며 큰 관심을 받고 있습니다. 하지만 우리의 평가 결과, 실험의 절반가량이 실패했고, 수치나 결과에서 '환각(Hallucination)' 현상도 나타나는 등 여러 한계가 있음이 드러났습니다.

- **Technical Details**: AI Scientist는 초기 준비 작업을 제외하고는 연구 라이프사이클을 자율적으로 수행할 수 있다고 주장하지만, 실제로 사용자는 실험을 위한 파이프라인을 제공해야 합니다. 이는 AI의 자율성을 제한하는 요소로 작용합니다. 시스템은 대체로 저렴한 가격과 적은 인적 자원으로 연구 논문을 작성할 수 있지만, 문헌 검토와 실험 실행에서 다소 부족한 성과를 보여줍니다.

- **Performance Highlights**: AI Scientist는 AI 기반 연구 도구의 발전을 알리고 있으며, 연구 자동화의 미래를 열어갈 잠재력을 가지고 있습니다. 그러나 현재로서는 연구 목표를 완전히 달성하지 못하고 있으며, 신뢰성과 일반화 가능성에 대한 의문도 제기되고 있습니다. 인공지능 연구 시스템의 발전을 도모하기 위해, IR 커뮤니티는 이러한 도구와 함께 작업하는 방법을 모색해야 합니다.



### Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems (https://arxiv.org/abs/2502.14137)
Comments:
          Accepted by WWW'2025

- **What's New**: 이번 연구에서는 대화형 추천 시스템(CRS)을 위한 새로운 접근 방법인 CRAG(Collaborative Retrieval Augmented Generation)을 제안합니다. CRAG는 최신 대형 언어 모델(LLM)과 협업 필터링(CF)을 결합하여 사용자 맞춤 추천을 개선합니다. 이 방법은 기존 데이터셋에서 더 높은 품질의 추천 성능을 보여줍니다.

- **Technical Details**: CRAG는 사용자의 대화에 기반한 추천을 위해 CF 지식을 활용하여 LLM의 맥락 인식(을 보강하게 됩니다. LLM이 대량의 데이터로 훈련되었음에도 불구하고, 사용자-아이템 상호작용 데이터는 비공식적이기 때문에 통합하기 어려운 문제를 해결하려고 합니다.

- **Performance Highlights**: 실험에서는 CRAG가 두 개의 영화 추천 데이터셋에서 기존 CRS 벤치마크와 비교하여 우수한 결과를 보여주었다고 보고하고 있습니다. 특히, 최근 개봉된 영화에 대한 추천 정확도가 크게 향상된 것으로 나타났습니다.



### Interpretable Text Embeddings and Text Similarity Explanation: A Primer (https://arxiv.org/abs/2502.14862)
- **What's New**: 이번 논문에서는 텍스트 임베딩(text embedding) 및 이와 관련된 유사성 점수를 해석하기 위한 방법론의 구조적 개요를 제공합니다. 유사성 점수의 해석 가능성 문제는 AI와 NLP 시스템에서 투명성이 요구되는 환경에서 중요한 과제로, 이를 해결하기 위한 최신 연구 동향을 다루고 있습니다. 다양한 텍스트 임베딩 모델들이 갖는 해석 가능성을 강화하기 위해 개별 방법들의 아이디어와 기술을 평가합니다.

- **Technical Details**: 신경망 텍스트 임베딩 및 유사성의 설명 가능성을 연구하며, 이는 기존의 분류 설명 방식과 구별됩니다. 두 입력의 상호작용을 기반으로 유사성이 결정되므로, 이를 위한 전문화된 방법이 필요합니다. 주어진 텍스트 입력을 통해 신경망을 통한 계산 단계가 설명되며, 일반적으로 Siamese 네트워크(서로의 가중치를 공유) 방식을 사용하여 텍스트 임베딩을 생성합니다.

- **Performance Highlights**: 텍스트 임베딩을 통해 얻은 유사성 점수는 문서 간의 유사성을 정량적으로 평가할 수 있는 중요한 척도를 제공합니다. 이러한 점수는 간단한 내적(dot product) 계산을 통해 산출되며, 이러한 방식은 실질적으로 강하게 상관된 코사인 유사성(cosine similarity)을 활용합니다. 논문은 텍스트 임베딩 및 유사성 정의를 통해 서로 다른 유사성 계산 방법을 구분할 수 있는 기초를 마련하고 있습니다.



### From Knowledge Generation to Knowledge Verification: Examining the BioMedical Generative Capabilities of ChatGP (https://arxiv.org/abs/2502.14714)
Comments:
          26 pages, 6 figures, In Review with a Cell Press Journal

- **What's New**: 본 논문은 LLM 모델이 생성한 생물 의학 정보의 사실 확인을 위한 체계적인 평가 접근 방식을 제안합니다. 구체적으로 질병, 약물, 증상, 유전자 간의 연관성 생성을 포함한 두 가지 핵심 프로세스를 도입했습니다. 이 방식은 ChatGPT를 사용하여 정량적인 평가 기초를 마련하고자 하며, 생물학적 네트워크에서의 정확성을 보장하는 데 중점을 두었습니다.

- **Technical Details**: 정확한 평가를 위해, 다양한 생물 의학 온톨로지를 활용하여 생성된 용어와 연관성을 검증하는 두 가지 주요 작업을 수행했습니다. 첫 번째 작업에서는 GO, DOID, ChEBI 및 증상 온톨로지를 통해 생성된 용어의 정확성을 확인했습니다. 두 번째 작업은 PubMed 데이터베이스를 사용하여 용어 간의 연관성을 검증했으며, 최종적으로 지식의 일관성을 유지하기 위한 세 번째 작업도 수행했습니다.

- **Performance Highlights**: 전반적으로 질병 용어(88%-97%)와 약물 이름(90%-91%) 식별에서 높은 정확성을 기록했고, 유전 정보는 88%-98%의 정확도를 보였습니다. 그러나 증상 용어 식별 정확도는 49%-61%로 상반된 결과를 나타냈습니다. 실험 결과로 도출된 질병-약물 및 질병-유전자 연관성 검증 시 문헌 커버리지는 각각 89%-91%에 달했습니다.



### InstructAgent: Building User Controllable Recommender via LLM Agen (https://arxiv.org/abs/2502.14662)
Comments:
          WWW2025@HCRS

- **What's New**: 본 논문은 전통적인 추천 시스템의 취약점을 개선하고 사용자 이익을 보호하기 위해 새로운 사용자-에이전트-플랫폼 패러다임을 제안합니다. 기존의 추천 시스템들은 플랫폼의 추천 알고리즘에 직접적으로 노출되어 사용자에게 불리한 상황을 초래할 수 있었습니다. 이에 대한 해결책으로, 추천 시스템 사이에서 사용자를 보호하는 역할을 하는 에이전트를 도입하여 간접적으로 노출되는 방식을 제시합니다. 특히, 사용자-driven 지침을 사용하는 새로운 추천 데이터셋 InstructRec를 구성했습니다.

- **Technical Details**: 논문에서는 네 가지 추천 데이터셋을 구성하고, 사용자의 자유로운 입력을 통해 개인의 관심사를 학습하는 Instruction-aware Agent (InstructAgent)를 설계하였습니다. 이 에이전트는 기존 사용자 데이터 또는 다른 사용자의 행동에 영향을 받지 않으며, 사용자 개개인의 피드백을 바탕으로 학습합니다. 또한 Dynamic Memory Mechanism을 사용하여 사용자 프로필을 동적으로 유지하고 업데이트하며, 사용자 고유의 관심사를 깊이 탐구하는 Individual Instruction-aware Agent (Instruct2Agent)를 도입했습니다.

- **Performance Highlights**: Empirical 실험을 통해 제안된 Instruct2Agent가 기존 최첨단 접근법들보다 평균 16.6% 향상된 성과를 달성함을 입증하였습니다. 또한, 에코챔버 효과의 영향을 분석하고 활동적인 사용자와 비활동적인 사용자의 성과를 개별적으로 평가하여, 개발된 에이전트가 사용자와 추천 시스템 간의 방패 역할을 효과적으로 수행함을 확인했습니다.



### Multi-Record Web Page Information Extraction From News Websites (https://arxiv.org/abs/2502.14625)
- **What's New**: 이 연구에서는 다수의 기록을 포함하는 웹 페이지에서 정보를 추출하는 문제에 집중했습니다. 이러한 과업은 방대한 웹 데이터 시대에 점차 중요해지고 있는데, 기존 연구는 대부분 단일 기록(detailed pages) 페이지에 초점을 맞춰왔습니다. 반면, 우리는 러시아어 뉴스 웹사이트에서 사용되는 리스트 페이지(mult-record list pages)에 특화된 대규모 데이터셋을 구축하여 이 간극을 메우고자 했습니다.

- **Technical Details**: 우리가 제작한 데이터셋은 13,120개의 뉴스 리스트 페이지로 구성되어 있으며, 속성은 다양한 유형을 포함하여 현실적인 학습 상황을 제공합니다. 이 연구에서는 MarkupLM 모델을 사용한 다단계 정보 추출 방법을 제안하였으며, 이는 웹 페이지에서 시각적 정보를 사용하지 않고도 효율적으로 데이터를 추출하게 돕습니다.

- **Performance Highlights**: 우리가 설계한 여러 실험을 통해 이 접근법의 장점을 검증했습니다. 데이터셋의 공개를 통해 다수의 기록 페이지에서 정보를 추출하는 분야의 발전을 촉진하길 기대합니다. 특히, 이 연구는 전통적인 데이터 추출 방법의 한계를 극복하고 새로운 데이터 셋을 통해 효과적인 정보 추출 방법론을 제시합니다.



### Unstructured Evidence Attribution for Long Context Query Focused Summarization (https://arxiv.org/abs/2502.14409)
Comments:
          24 pages; 21 figures; 5 tables

- **What's New**: 이번 연구는 LLM(대규모 언어 모델)에서 긴 맥락을 바탕으로 unstructured evidence citation(비구조적 증거 인용) 문제를 다루고 있습니다. 기존의 연구들은 일반적으로 구조화된 증거 인용에 초점을 맞추었지만, 본 논문에서는 비구조적 증거 인용의 필요성을 강조하며, 이를 위한 새로운 데이터셋인 SUnsET을 소개합니다. 이 데이터셋은 LLM이 사용자 쿼리에 기반하여 보다 투명하고 신뢰할 수 있는 요약을 제공할 수 있도록 돕기 위한 것입니다.

- **Technical Details**: 연구팀은 SUnsET 데이터셋을 생성해 LLM을 fine-tuning(미세 조정)하여, 달라진 증거 인용 능력과 더욱 관련성 높은 요약의 품질 향상을 목표로 했습니다. SUnsET은 다양한 도메인에서 일반화 가능한 자료로, 실제 문서를 모듈화하여 썼으며, 문서 섹션을 섞는 데이터 증강이 가능하도록 설계되었습니다. 이를 통해 LLM의 positional biases(위치 편향) 문제를 해결하고, 비구조적 증거 인용의 효율을 높였습니다.

- **Performance Highlights**: SUnsET 데이터로 미세 조정된 LLM은 원래 모델 대비 더 효과적이고 사실적으로 일관된 증거를 생성하는 성과를 보였습니다. 본 연구에서 다룬 5개의 서로 다른 LLM과 4개의 다양한 데이터셋에서 실험한 결과, 요약의 질이 향상되고, 다양한 위치에서 증거를 추출할 수 있음을 입증하였습니다. 이 결과는 사용자 쿼리에 대한 보다 정확하고 관련된 응답을 생성하는 데 기여할 것으로 기대됩니다.



### Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning (https://arxiv.org/abs/2502.14361)
- **What's New**: 이 논문은 Process Reward Models (PRM)의 주요 아웃 오브 디스트리뷰션(Out-of-Distribution) 문제를 강조하며, 질문 OOD와 단계 OOD를 포함하여 모델 유형(예: GPT, Qwen)과 모델 크기(1.5B, 72B)에 따른 추론 패턴의 차이로 인해 발생하는 문제들을 설명합니다. 이를 해결하기 위해 새로운 프레임워크인 Retrieval-Augmented Process Reward Model (RetrievalPRM)을 소개하고 있습니다.

- **Technical Details**: RetrievalPRM는 두 단계의 검색 강화 메커니즘(Two-stage Retrieval-enhanced Mechanism)을 활용하여, 문제 OOD와 단계 OOD 문제를 해결합니다. 이를 통해서 의미적으로 유사한 질문 및 단계를 선택하여 PRM의 성능을 향상시키고, 다양한 문제 해결 시나리오에서의 일반화 능력을 높입니다. 또한, RetrievalPRM 프레임워크를 사용하여 훈련할 수 있는 검색 강화 데이터셋을 개발하였습니다.

- **Performance Highlights**: 실험 결과, RetrievalPRM은 여러 공개된 실제 데이터셋에서 기존의 강력한 기준 모델 대비 높은 성능을 보여주었습니다. 아울러, Retrieval 접근 방식을 통한 OOD 문제 완화가 입증되었습니다. 이 연구에서 개발한 코드 및 데이터셋은 오픈소스로 공개되어 있어, 다른 연구자들도 확인하고 활용할 수 있습니다.



### A Collaborative Jade Recognition System for Mobile Devices Based on Lightweight and Large Models (https://arxiv.org/abs/2502.14332)
- **What's New**: 이번 논문에서는 모바일 기기에서 효율적이고 정확한 옥 인식을 위한 새로운 시스템을 제안합니다. 이 시스템은 다중 스케일 이미지 처리를 기반으로 한다는 점에서 혁신적이며, 작은 모델과 큰 모델의 협업을 통해 모바일 장치에서 인식 정확도를 높이는 데 중점을 두고 있습니다. 이를 통해 기존의 주관적인 옥 감정 방식에 비해 보다 객观적이고 신뢰할 수 있는 솔루션을 제공합니다.

- **Technical Details**: 제안된 시스템은 경량 모델을 모바일 기기에 배치하고, 클라우드 또는 엣지 서버에 큰 모델을 배치하여 특징 추출 및 이차 검증을 수행합니다. 이 시스템은 데이터 교환 및 작업 분배를 위한 효과적인 메커니즘을 갖추고 있어 두 모델이 서로의 강점을 활용할 수 있도록 설계되었습니다. 또한, 합성곱 신경망(convolutional neural network)과 전통적인 컴퓨터 비전 알고리즘을 조합한 다중 모델 분류 프레임워크를 사용하여 다양한 옥 특성에 따라 모델을 신속하게 선택하고 조정할 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 모바일 기기에서도 높은 인식 정확도와 빠른 처리 시간을 제공함을 보여주었습니다. 실험 결과, 다양한 환경에서 높은 정확성과 견고한 성능을 입증하였으며, 이는 향후 문화유산 보호 및 지능형 감정 기술 개발에 기여할 것으로 기대됩니다. 경량 모델을 활용하여 연산 자원을 절약하면서도 실시간 성능을 유지할 수 있는 가능성을 제시합니다.



### Less is More: On the Importance of Data Quality for Unit Test Generation (https://arxiv.org/abs/2502.14212)
- **What's New**: 이 논문은 테스트 생성에 사용되는 데이터셋 품질의 중요성을 강조하고, 노이즈가 학습 기반의 테스트 생성 모델 성능에 미치는 영향을 체계적으로 분석합니다. 특히, 기존의 연구들은 대규모 데이터셋에 초점을 맞추었지만 데이터의 품질에는 거의 주목하지 않았습니다. 새로운 자동화된 데이터 클리닝 프레임워크 CleanTest를 제안하여, 노이즈를 정리함으로써 테스트 생성의 품질을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: CleanTest는 세 가지 필터로 구성되어 있습니다: 규칙 기반의 구문 필터(rule-based syntax filter), 규칙 기반의 관련성 필터(rule-based relevance filter), 모델 기반의 커버리지 필터(model-based coverage filter)입니다. 이 연구에서는 Methods2Test와 Atlas라는 두 개의 널리 사용되는 테스트 생성 데이터셋에 CleanTest를 적용하여 노이즈의 실태를 평가하였습니다. 연구 결과, 데이터셋의 43.52%와 29.65%에서 각각 노이즈가 발견되었습니다.

- **Performance Highlights**: 노이즈를 필터링한 결과, 다양한 LLM(large language models)에서 테스트 생성 능력이 향상되는 긍정적인 영향을 나타냈습니다. 실험에는 CodeBERT, AthenaTest, StarCoder 그리고 CodeLlama7B의 네 가지 모델이 사용되었습니다. 이는 데이터 클리닝이 테스트 생성 품질에 미치는 긍정적인 영향을 강조하며, 향후 연구에서의 데이터 품질 개선의 필요성을 제기합니다.



### Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach (https://arxiv.org/abs/2502.14100)
- **What's New**: 이번 연구에서는 external contexts를 적용한 Large Language Models (LLMs)가 내부 지식과 외부 컨텍스트의 균형을 효과적으로 맞추는 'context-robust LLMs' 개념을 제안합니다. 이를 통해 LLM이 내부 지식이 부족할 때 외부 정보를 사용하고, 모순된 정보가 있을 경우 이를 식별할 수 있어야 합니다. 이러한 기능은 인간의 인지 과정과 유사한 방식으로 정보 처리의 정확성을 높이는 데 중요한 역할을 합니다.

- **Technical Details**: 연구의 핵심 기술은 Grft라고 불리는 경량의 gated representation fine-tuning 접근법으로, 두 가지 주요 요소인 게이트 메커니즘과 저차원 표현(adapters)을 포함합니다. 게이트 메커니즘은 '문제가 있는' 입력을 감지하여 필터링하고, 저차원 표현 어댑터는 숨겨진 표현을 조정합니다. Grft는 모델 크기의 0.0004%만을 사용하여 200개 미만의 예시로 효과적인 학습을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 Grft는 LLM이 잘못된 정보와 도움이 되지 않는 컨텍스트를 처리하는 능력을 개선하는 데 효과적임을 입증했습니다. 또한, Grft를 통해 LLM은 유용한 컨텍스트에서의 성능을 유지하면서도, 비유용한 컨텍스트에 대해 더욱 신뢰할 수 있는 반응을 생성할 수 있습니다. 이러한 성과는 LLM의 응용 분야에서도 이점을 제공할 것으로 기대됩니다.



### An Open-Source Web-Based Tool for Evaluating Open-Source Large Language Models Leveraging Information Retrieval from Custom Documents (https://arxiv.org/abs/2502.10916)
Comments:
          19 pages, 1 figure, 6 tables

- **What's New**: 이번 연구에서는 오픈소스 대화 에이전트와의 소통에서 사용자의 발화행위(speech act)가 미치는 영향을 보여주는 최초의 웹 기반 툴을 소개합니다. 이 툴은 사용자 커뮤니케이션 의도를 시각화하고, 특정 문서를 기반으로 정보 검색을 가능하게 하며 대화의 성능을 평가할 수 있습니다. 조사 결과, 사용자 발화행위를 포함한 쿼리에서 대형 오픈소스 모델들이 향상된 정렬성을 보였음을 확인했습니다.

- **Technical Details**: 연구에서는 5개의 오픈소스 대형 언어 모델을 사용하여 각각의 모델이 문서 기반 정보 검색의 효과를 평가했습니다. 이 시스템은 사용자가 쿼리와 발화행위를 함께 제공하거나 단순히 쿼리만 제시할 수 있는 두 가지 입력 방식으로 설계되었습니다. 또한, 문서 검색에 통합된 발화행위를 통해 대화 에이전트의 응답 생성을 향상시키고자 했습니다.

- **Performance Highlights**: 실험 결과, 발화행식을 포함할 때 대형 모델은 성능이 개선된 것으로 나타났지만, 소형 모델의 경우 발화행식이 포함된 쿼리를 처리하는 데 어려움을 보였습니다. 이 분석을 통해 발화행위를 활용한 대화 깊이 향상의 가능성이 강조되었으며, 모델 특정 최적화의 필요성과 컴퓨팅 비용 증가 및 응답 시간 문제를 해결하기 위한 접근법이 제안됩니다.



New uploads on arXiv(cs.CV)

### Time Travel: A Comprehensive Benchmark to Evaluate LMMs on Historical and Cultural Artifacts (https://arxiv.org/abs/2502.14865)
Comments:
          4 pages, 6 figures

- **What's New**: 본 연구에서는 인공지능 모델(Artificial Intelligence Model)의 고급 평가 도구인 TimeTravel을 소개합니다. TimeTravel은 10개의 주요 역사적 지역에서 266개의 문화에 걸쳐 10,250개의 전문가 검증 샘플로 구성된 벤치마크(benchmark)입니다. 이 데이터셋은 원고, 예술 작품, 비문, 고고학적 발견에 대한 AI 분석을 지원하며, AI기술과 역사 연구의 통합을 통해 역사적 발견에 기여하고 문화유산 보존에 중요한 역할을 합니다.

- **Technical Details**: TimeTravel 데이터셋은 광범위한 문화유산을 분석하기 위해 철저하게 선별된 데이터를 기반으로 합니다. 각 샘플은 고대 유물, 비문, 고대 원고를 포함하며, 역사학자와 고고학자에 의해 신중하게 검증되어 정확성 및 신뢰성을 보장합니다. 이 벤치마크는 LMMs(Large Multimodal Models) 성능을 평가하기 위한 다각적인 접근을 취하고 있으며, 기존의 객관적 인식 중심 벤치마크와 달리 역사적 지식 및 문화적 보존을 중요시합니다.

- **Performance Highlights**: TimeTravel 데이터셋에서의 평가 결과는 폐쇄형(closed-source) 및 개방형(open-source) 모델 간에 뚜렷한 성능 차이를 보여줍니다. GPT-4o-0806은 BLEU, ROUGE-L, SPICE 등 여러 측정 지표에서 최고의 성능을 기록했으나, METEOR 점수는 낮아 어휘 다양성이 부족함을 나타냈습니다. 반면, GPT-4o-mini-0718은 더 나은 어휘적 다양성과 유창성을 보여주며, Qwen-2.5-VL는 개방형 모델 중 높은 점수를 기록하여 역사적 맥락의 정확성을 잘 포착하는 경향을 나타냈습니다.



### Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation (https://arxiv.org/abs/2502.14846)
Comments:
          20 pages, 19 figures, 9 tables, website: this https URL

- **What's New**: 논문은 CoSyn이라는 새로운 프레임워크를 소개하여, 텍스트가 풍부한 멀티모달 데이터를 자동으로 생성하는 방법을 제시합니다. 이 시스템은 텍스트 전용 대형 언어 모델(LLM)을 활용하여 입력된 텍스트 설명을 바탕으로 코드를 생성하고, 이를 통해 합성 이미지와 지침 데이터를 만들어냅니다. CoSyn을 통해 40만 개의 이미지와 270만 개의 비전-언어 지침 조합을 포함한 데이터셋이 구축되었습니다.

- **Technical Details**: CoSyn은 다양한 코드(예: Python, HTML, LaTeX)를 생성할 수 있는 텍스트 전용 LLM의 기능을 활용합니다. 이 구조에 따라 CoSyn은 특정 도메인(예: 영양 성분 라벨)에 대한 지침을 생성하여 해당 도메인에 적합한 합성 이미지를 보여줄 수 있도록 합니다. 이 과정에서 생성된 이미지는 텍스트 기반의 지침 형식으로 저장되며, VLM의 학습에 효과적으로 사용될 수 있습니다.

- **Performance Highlights**: CoSyn으로 생성된 데이터로 훈련한 모델은 경쟁력 있는 오픈 소스 모델들 사이에서 최첨단 성능을 달성했습니다. 특히, 기존의 VLM들은 훈련 데이터의 편향으로 인해 일반화에 어려움을 겪는 반면, CoSyn-400K 데이터를 사용한 모델은 적은 데이터로도 강력한 성능을 발휘했습니다. 또한, CoSyn은 VLM들이 현실 세계의 정보에 접근하고 이해할 수 있도록 하는 잠재력이 있음을 보여줍니다.



### LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models (https://arxiv.org/abs/2502.14834)
- **What's New**: 본 논문에서는 기존의 Large Vision-Language Models (LVLMs)가 최대 128k의 비주얼 및 텍스트 토큰을 처리할 수 있지만, 1,000단어를 초과하는 일관된 출력을 생성하는 데 어려움을 겪는 문제를 다룹니다. 이를 해결하기 위해 LongWriter-V-22k라는 새로운 SFT(Supervised Fine-Tuning) 데이터셋을 소개하는데, 이 데이터셋은 22,158개의 예시를 포함하고 있으며, 각각 여러 입력 이미지, 지침, 최대 10,000단어 범위의 출력이 포함되어 있습니다.

- **Technical Details**: 우리는 입력 이미지에 대한 높은 충실도를 유지하면서 긴 출력을 달성하기 위해 Direct Preference Optimization (DPO) 방법론을 활용합니다. 길고 복잡한 출력을 위한 인간 피드백을 수집하는 것이 비용이 많이 드는 점을 고려하여, IterDPO라는 접근 방식을 제안합니다. 이 방법은 긴 출출을 세분화하여 원래 출력과의 선호 쌍을 형성하기 위한 반복적 수정(iterative corrections)을 사용합니다.

- **Performance Highlights**: LongWriter-V-22k와 IterDPO로 훈련된 7B 파라미터 모델은 MMLongBench-Write 벤치마크에서 우수한 성과를 보여주며, GPT-4o와 같은 더 큰 상용 모델들을 초월하는 결과를 냈습니다. 이는 긴 생성능력의 평가에서 LVLM의 새로운 가능성을 시사합니다.



### Improving the Diffusability of Autoencoders (https://arxiv.org/abs/2502.14831)
Comments:
          26 pages, 22 figures, 9 tables

- **What's New**: 이번 연구에서는 Latent Diffusion Models (LDMs)의 주요 요소인 autoencoders와 diffusion backbones 간의 상호작용에 대해 깊이 있는 분석을 진행했습니다. 특히, 고주파 성분의 비정상적 존재가 diffusion 합성 과정에 방해가 된다는 가설을 세웠습니다. 이를 해결하기 위해 scale equivariance라는 정규화 전략을 제안함으로써 latent 및 RGB 공간의 주파수 정렬을 도모하였습니다.

- **Technical Details**: 연구에서 제안한 scale equivariance는 decoder에서 스케일 동등성을 강제하여 downsampled latent가 downsampled RGB 표현에 일치하도록 보장합니다. 이 방법은 autoencoder의 미세 조정을 20K 단계로 제한하며, minimal code 변경만으로도 효과적인 결과를 도출합니다. 이 과정에서 고주파 성분이 최종 RGB 결과에 미치는 영향을 분석하여, 기존의 KL 정규화가 충분하지 않음을 밝혔습니다.

- **Performance Highlights**: 제안한 방법은 ImageNet-1K 데이터셋에서 이미지 생성의 FID를 19% 감소시켰고, Kinetics-700 데이터셋에서도 비디오 생성의 FVD를 최소 44% 향상시켰습니다. 이는 다양한 아키텍처의 diffusability를 개선하여 생성 품질을 유의미하게 향상시키는 결과를 나타냅니다.



### Exploring Advanced Techniques for Visual Question Answering: A Comprehensive Comparison (https://arxiv.org/abs/2502.14827)
Comments:
          8 pages, No figures

- **What's New**: 이번 논문은 시각 질문 응답(Visual Question Answering, VQA) 분야에서 최신 기술 models에 대한 포괄적인 비교 연구를 제공합니다. 연구는 ABC-CNN, KICNLE, Masked Vision and Language Modeling, BLIP-2, OFA 등 다섯 가지 모델의 접근 방식을 분석하여 VQA의 새로운 발전 방향을 체계적으로 제시하고 있습니다.

- **Technical Details**: VQA는 컴퓨터 비전(computer vision)과 자연어 처리(natural language processing) 의 교차점에서 중요한 작업으로, 모델이 자연어 질문에 대한 시각적 내용에 대한 이해와 사고를 요합니다. 이 연구에서는 각 모델이 질문 다양성(question diversity), 답변 분포(answer distribution), 시각-텍스트 상관관계(visual-textual correlations) 등에서 어떤 차별적 접근을 하고 있는지 살펴봅니다.

- **Performance Highlights**: 기존 VQA 모델들은 데이터셋 편향(dataset bias), 제한된 모델 복잡성(limited model complexity), 상식 추론(gaps in commonsense reasoning), 경직된 평가 방법(rigid evaluation methods) 및 실제 시나리오에 대한 일반화(generalization)와 같은 문제에 직면해 있습니다. 이 논문은 이러한 문제를 해결하기 위한 다섯 가지 고급 모델의 성능을 세밀하게 비교하여 VQA의 한계점과 발전 가능성을 강조하고 있습니다.



### AVD2: Accident Video Diffusion for Accident Video Description (https://arxiv.org/abs/2502.14801)
Comments:
          ICRA 2025, Project Page: this https URL

- **What's New**: 본 연구에서는 AVD2 (Accident Video Diffusion for Accident Video Description)라는 새로운 프레임워크를 제안합니다. AVD2는 사고 장면을 이해하는 데 도움을 주기 위해 사고 비디오를 생성하고 이를 자연어 설명과 연결하는 시스템을 포함하고 있습니다. 이를 통해 EMM-AU (Enhanced Multi-Modal Accident Video Understanding) 데이터셋이 구축되었으며, 이는 사고 분석 및 예방 분야에서의 진전을 가져올 것으로 기대됩니다.

- **Technical Details**: AVD2는 사고 비디오 기반 텍스트 생성 파이프라인과 사고 분석 시스템으로 구성되어 있습니다. 이 프레임워크는 SCST (Self-Critical Sequence Training)를 사용하여 비디오의 캡션과 시각 내용의 맥락 적합성을 개선하며, 복잡한 운전 시나리오의 세부사항을 잘 포착할 수 있도록 돕습니다. EMM-AU 데이터셋은 기존의 비디오 생성을 통해 증가된 사고 비디오를 포함하여, 사고 분석 방법의 신뢰성과 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 다양한 실험을 통해 AVD2 프레임워크는 사고 원인 분석의 문제를 해결하며, 사용자 신뢰를 향상시키기 위한 자세한 실행 가능 인사이트를 생성하는 데 성공했습니다. EMM-AU 데이터셋은 자동화된 메트릭과 인간 평가 모두에서 최신 성능을 기록하며, 이는 사고 분석 및 예방 분야의 발전에 크게 기여할 것입니다. AVD2는 사고 비디오 이해에 대한 새로운 기준을 설정하며, 자율 주행 시스템의 신뢰성과 해석 가능성을 크게 향상시킵니다.



### A Survey on Text-Driven 360-Degree Panorama Generation (https://arxiv.org/abs/2502.14799)
- **What's New**: 이 논문은 텍스트 기반의 360도 파노라마 생성 기술이 어떻게 발전했는지를 다룹니다. 특히, 텍스트 설명에서 직접 360도 파노라마 이미지를 합성하는 혁신적 방법을 소개합니다. 종전의 복잡한 생성 과정을 단순화시킴으로써 VR/AR, 게임 및 가상 투어와 같은 다양한 분야에서의 콘텐츠 생성 방식에 혁신을 일으킬 것으로 기대됩니다. 또한, 일련의 최첨단 텍스트 기반 알고리즘과 이들의 응용법을 면밀히 분석합니다.

- **Technical Details**: 정보의 정확한 생성과 기하학적 일관성 유지를 강조하는 360도 파노라마 생성에는 고유한 기술적 도전이 따릅니다. 최근 텍스트-이미지 확산 모델(text-to-image diffusion models)의 발전이 이를 가능하게 하여, 고품질의 이미지를 생성하는 다양한 방법론이 제시되었습니다. 이 연구에서는 텍스트-주도 생성(text-driven generation)과 텍스트-주도 틈새 시야(NFOV) 아웃페인팅 방법을 포함한 두 가지 주요 패러다임을 소개하고, 각 방법의 장단점을 비교합니다.

- **Performance Highlights**: 연구 결과에 따르면, 텍스트-이미지 확산 모델을 사용하여 생성된 360도 파노라마는 전통적인 방법보다 훨씬 높은 품질의 시각적 결과를 제공합니다. 특히, 신규 응용 프로그램으로서 텍스트 기반으로 360도 3D 장면 생성을 지원하는 가능성을 보여줍니다. 현재의 기술적 한계를 극복하기 위한 연구 방향 또한 제안되어, 이 분야의 미래 발전에 중요한 기초가 될 것으로 예상됩니다.



### RendBEV: Semantic Novel View Synthesis for Self-Supervised Bird's Eye View Segmentation (https://arxiv.org/abs/2502.14792)
Comments:
          Accepted at WACV 2025

- **What's New**: 새로운 연구법인 RendBEV는 BEV(새로운 시각) 의미 분할 네트워크를 자가 감독(self-supervised)으로 훈련할 수 있는 방법을 제시합니다. 기존의 연구는 대부분 대규모 주석 데이터셋을 기반으로 한 전통적인 감독 학습에 의존했지만, RendBEV는 2D 의미 분할 모델에 의해 계산된 의미적 관점 뷰(perspective views)를 활용하여 훈련됩니다. 이 방법은 비용이 많이 드는 주석 작업 없이도 BEV 의미 분할을 가능하게 합니다.

- **Technical Details**: RendBEV는 비디오 시퀀스(video sequences)를 기반으로 하여 단안 카메라(monocular) 의미 분할 BEV 네트워크를 훈련합니다. 이는 기존의 BEV 모델이 예측한 프레임을 통해 다른 비디오 프레임에 대한 의미적 관점 뷰를 렌더링하는 방식을 채택합니다. 이러한 렌더링 과정은 최근의 새로운 시각 합성(novel view synthesis) 기술을 통해 가능해졌으며, 이를 통해 클래스 확률을 미분 가능하게 렌더링하고 업데이트할 수 있습니다.

- **Performance Highlights**: 실험 결과 RendBEV는 KITTI-360 데이터셋에서 유망한 성과를 보여, 전통적인 감독 학습 방법보다 우수한 성능을 발휘합니다. 특히, 주석 데이터가 0.1%인 경우에도 경쟁력 있는 결과를 도출하며, 100%의 훈련 데이터에 대해 파인튜닝(fine-tuning)할 경우 최신 성능을 기록합니다. 이 연구는 낮은 주석 수준에서도 BEV 의미 분할 네트워크의 강력한 성능을 입증합니다.



### Structurally Disentangled Feature Fields Distillation for 3D Understanding and Editing (https://arxiv.org/abs/2502.14789)
- **What's New**: 이 연구에서는 2D 감독만을 이용하여 여러 개의 분리된 feature field를 사용하여 3D feature를 캡처하는 새로운 접근 방식을 제안합니다. 이전의 모델들은 단일 view-independent feature field를 사용하여 3D feature를 취득하는데 그쳤으나, 이는 view-dependent 변동성을 평균내어 성능이 떨어지는 문제를 가지고 있었습니다. 제안된 방법은 각 구성 요소를 개별적으로 제어할 수 있는 가능성을 열며, 이는 3D segmentation 및 편집에 유의미한 결과를 제공합니다.

- **Technical Details**: 우리는 3D 포인트의 feature 값을 두 개의 분리된 feature field의 조합으로 계산하는 방식을 취합니다. 하나는 specular 객체 반사로 인해 생성되는 view-dependent features를 캡처하는 반사된 view feature field이고, 또 하나는 3D 포인트의 위치에만 의존하는 diffuse features를 캡처하는 view-independent feature field입니다. 이러한 분리된 feature field는 오직 2D 감독 학습만으로 배울 수 있으며, 이는 view-independent 및 view-dependent components로의 분리를 가능하게 합니다.

- **Performance Highlights**: 이 접근법을 Shiny Blender 데이터셋 및 실제 장면 데이터셋에서 평가한 결과, 단일 holistic feature field와 비교해 구조적으로 분리된 표현이 우수한 성능을 보였습니다. 사용자가 클릭하여 3D 객체 전체를 분할하거나 반사된 구성 요소만을 선택하여 편집할 수 있는 능력을 보여주며, 이는 3D segmentation과 편집의 새로운 응용 프로그램을 제공합니다. 특히 view-independent component만을 사용하여 성능 향상이 이루어졌음을 보여주었습니다.



### SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features (https://arxiv.org/abs/2502.14786)
Comments:
          Model checkpoints are available at this https URL

- **What's New**: SigLIP 2는 멀티링구얼 비전-언어 인코더의 새로운 가족으로, 기존 SigLIP의 성공을 기반으로 확장되었습니다. 이 모델은 캡셔닝 기반의 프리트레이닝, 셀프-슈퍼바이즈드 손실(self-supervised losses), 온라인 데이터 큐레이션과 같은 여러 기술을 통합한 훈련 레시피를 적용하여 기존 모델들을 초월하는 성능을 보여줍니다. 특히, 멀티링구얼 이해와 공정성을 향상시키기 위해 더 다양하고 균형 잡힌 데이터 혼합으로 학습되었습니다.

- **Technical Details**: SigLIP 2는 SigLIP의 훈련 레시피와 디코더 기반 프리트레이닝을 결합하고, 다양한 해상도를 지원하는 모델 변형을 포함합니다. 추가로 성능 최적화를 위해 액티브 샘플 선택(active sample selection)을 통한 암묵적 증류(implicit distillation) 기법을 활용하여 작은 모델의 질을 향상시켰습니다. 이미지와 텍스트 인코더 모두 같은 아키텍처를 사용하며, 다국어 Gemma 토크나이저를 채택하여 텍스트를 처리하는 방식도 개선되었습니다.

- **Performance Highlights**: SigLIP 2는 이미지-텍스트 분류, 검색, VLM 기능 추출에서 모든 모델 스케일에서 기존 SigLIP보다 우수한 성능을 발휘합니다. 특히, 지역화(localization)와 밀집 예측(dense prediction) 작업에서도 성능이 크게 향상되었습니다. 사용자는 성능과 추론 비용을 조절할 수 있도록 4가지 모델 사이즈를 제공받아 다양한 애플리케이션에 활용할 수 있습니다.



### DC-ControlNet: Decoupling Inter- and Intra-Element Conditions in Image Generation with Diffusion Models (https://arxiv.org/abs/2502.14779)
- **What's New**: 이번 논문에서는 DC(Control)-ControlNet을 소개합니다. 이 프레임워크는 다중 조건 이미지 생성에 대한 높은 유연성과 정밀한 제어를 제공합니다. DC-ControlNet은 기존 ControlNet 모델의 한계를 극복하고, 요소 또는 지역 별 제어 능력을 갖춘 새로운 접근 방식을 제안합니다.

- **Technical Details**: DC-ControlNet은 조건을 분리하여 전역 제어를 개별 내용, 레이아웃의 계층적 통합으로 변환합니다. Intra-Element Controller는 각 요소 내에서 다양한 제어 신호를 처리하고, Inter-Element Controller는 요소 간 상호 작용과 가림을 정확히 처리하여 사용자 정의 관계 기반으로 작동합니다. 이를 통해 사용자는 각각의 요소에 대해 독립적으로 조건을 설정할 수 있습니다.

- **Performance Highlights**: DC-ControlNet의 성능 평가 결과, 기존 ControlNet 모델 및 레이아웃-투-이미지 생성 모델에 비해 다중 조건 제어의 유연성 및 정밀도에서 현저한 향상을 보여주었습니다. 새로운 DMC-120k 데이터셋을 통해 120,000개의 다양한 다중 조건 이미지를 제공하며, 실험을 통해 제안된 방법의 유효성이 입증됩니다.



### YOLOv12: A Breakdown of the Key Architectural Features (https://arxiv.org/abs/2502.14740)
- **What's New**: YOLOv12는 이전 버전인 YOLO 시리즈의 강점을 기반으로 한 뚜렷한 발전으로, 실시간 객체 탐지의 새로운 기준을 제시합니다. 이 모델은 최적화된 백본(R-ELAN), 7x7 분리 가능 합성곱, FlashAttention 기반의 지역적 주목 기능을 통합하여 특징 추출, 효율성, 탐지 강도를 개선했습니다. YOLOv12는 다양한 하드웨어 플랫폼에 걸쳐 배포 가능하며, 지연이 중요한 애플리케이션부터 높은 정확도가 요구되는 응용 분야까지 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: YOLOv12는 강력한 다층 구조를 바탕으로 최신 딥러닝 기술을 활용하여 성능을 극대화합니다. 특히, Residual Efficient Layer Aggregation Network(R-ELAN)를 통해 기울기 병목 문제를 완화하고, 특징 융합을 개선합니다. 또한, 7x7 분리 가능한 합성곱이 전통적인 위치 인코딩을 대체하면서 매개변수를 줄이며 공간적 맥락을 유지합니다. 이러한 장치들은 YOLOv12의 실시간 처리 능력을 유지하면서도 탐지 정확도를 크게 향상시킵니다.

- **Performance Highlights**: YOLOv12는 mAP(평균 평균 정밀도)과 추론 속도 모두에서 일관된 성능 향상을 보여주어 자율 시스템, 보안 및 실시간 분석 분야에 적합한 모델입니다. 특히 복잡한 시각적 패턴을 인식하는 데 뛰어난 능력을 보이며, 특히 소형 또는 중첩된 객체 탐지에서의 성능을 크게 개선하였습니다. 따라서 YOLOv12는 자율 주행 차량, 의료 영상 분석, 농업 분야 등에서 실용적으로 활용될 수 있는 빼어난 선택지로 자리잡을 것입니다.



### Multi-dataset synergistic in supervised learning to pre-label structural components in point clouds from shell construction scenes (https://arxiv.org/abs/2502.14721)
Comments:
          18 pages, 8 figures, 7 tables

- **What's New**: 본 연구는 건설 산업에서 데이터 레이블링의 어려움을 극복하기 위해, 기존의 표준 데이터셋과 최신 transformer 모델 아키텍처를 활용하여 포인트 클라우드의 의미론적 분할(semantic segmentation) 문제를 해결하는 접근 방식을 제안합니다. 전통적인 객체 분할(object segmentation) 방법이 아닌, 건축, 엔지니어링 및 건설(AEC) 분야의 복잡한 구조 구성 요소를 대상으로 하는 새로운 방법론이 담겨 있습니다. 이 연구의 주요 성과는 최소한의 데이터로도 효과적인 세그멘테이션을 달성할 수 있다는 것을 입증하는 것입니다.

- **Technical Details**: 연구자는 세 가지 주요 단계로 구성된 실험을 설계하여, (1) 커스텀 검증 데이터셋에 대한 완전 고속 훈련을 통해 모델 성능의 기준점을 설정하고, (2) 여러 기존 데이터셋을 활용하여 transformer 기반의 세그멘테이션 모델의 일반화 능력을 평가하며, (3) 공공 실내 데이터셋에서 사전 훈련된 모델을 AEC 특화 데이터에 대해 세분화 성능 향상을 목적으로 추가 학습(transfer learning)하는 방법론을 사용하였습니다. 이를 통해 기존의 다양한 데이터 자원을 결합하여 건설 분야의 의미론적 분할을 향상하고, AEC 산업의 전통적인 컴퓨터 비전 접근 방식과의 간극을 줄이고자 하였습니다.

- **Performance Highlights**: 본 연구의 결과, 사전 훈련된 transformer 아키텍처는 최소한의 미세 조정으로도 효과적인 빌딩 구성 요소 분할을 제공하는 전략이 될 수 있음을 보여줍니다. 이러한 결과는 새로운 데이터에 대한 자동 주석 작성을 지원하여 대규모 훈련 자료 생성을 통해 성능을 개선하는 가능성을 제공합니다. 또한, 건설 현장에서의 성능 검증을 통해 발견된 자주 발생하는 객체의 세그멘테이션은 향후 로봇 및 자율 기계의 안전한 작동을 보장하는 데 유용할 것입니다.



### BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction (https://arxiv.org/abs/2502.14676)
- **What's New**: 이 논문에서는 교통 에이전트의 궤적 예측을 위한 새로운 접근법으로, 행동 유사 라벨(behavioral pseudo-labels)을 도입하여 보행자와 이질적인 에이전트의 행동 특성을 효과적으로 포착합니다. 이 방법은 궤적 예측의 정확도를 크게 높이며, 추가적인 레이블 주석의 필요성을 줄입니다. 제안된 BP-SGCN(Behavioral Pseudo-Label informed Sparse Graph Convolution Network)은 이러한 유사 라벨을 학습하고 예측 모델에 정보를 제공합니다.

- **Technical Details**: BP-SGCN은 두 가지 모듈로 구성되어 있습니다. 첫 번째 모듈은 심층 비지도 행동 군집화(deep unsupervised behavior clustering) 모듈로, 경로 관찰을 통해 에이전트에 유사 라벨을 부여합니다. 두 번째는 목표 지향 유사 라벨 기반 궤적 예측 모듈로, 에이전트의 공간적 상호작용과 시간적 의존성을 효과적으로 모델링하기 위해 Sparse Graph Convolutional Network(SGCN)를 활용합니다. 이를 통해 경량화된 훈련 방식으로 두 모듈을 함께 최적화합니다.

- **Performance Highlights**: 실험 결과, BP-SGCN은 SDD와 Argoverse 1 데이터셋에서 이질적인 예측을 초과 달성하며, ETH/UCY 데이터셋과 보행자 전용 SDD 설정에서도 우수한 성능을 보입니다. 제안된 새로운 개념인 행동 유사 라벨을 통해 다양한 행동 군집을 모델링하여 궤적 예측의 성능이 상당히 향상되었습니다. 소스 코드는 연구의 발전을 위해 GitHub에서 제공됩니다.



### MAGO-SP: Detection and Correction of Water-Fat Swaps in Magnitude-Only VIBE MRI (https://arxiv.org/abs/2502.14659)
- **What's New**: 이 연구에서는 VIBE MRI에서 발생하는 물과 지방 신호 스왑(water-fat swaps)을 자동으로 탐지하고 수정하는 파이프라인을 개발하였습니다. 3단계로 구성된 이 파이프라인은 '지방 유사(fat-like)' 또는 '물 유사(water-like)'로 볼륨을 분류하는 세그멘테이션 네트워크의 훈련으로 시작됩니다. 이어서 노이즈 제거(diffusion) 네트워크가 수정의 신호 사전으로서 물 볼륨을 예측하고, 마지막으로 물과 지방 신호의 정확한 복구를 위한 물리 모델에 이 사전 정보를 통합합니다.

- **Technical Details**: 이 연구에서 제안한 방법은 6포인트 VIBE에서 물과 지방 신호 스왑 탐지 오차율을 1% 미만으로 달성하였습니다. 에코 시간(echo time)과 같은 다양한 MRI 데이터를 반영하여 효과적인 횡방향 이완율 R2*와 프로톤 밀도 지방 분율(PDFF)을 계산합니다. 또한, 페리린 노이즈(Perlin noise)를 이용해 합성 물-지방 스왑을 생성하여 세그멘테이션 네트워크의 훈련에 활용하는 독창적인 방법이 도입되었습니다.

- **Performance Highlights**: 본 연구의 수정 알고리즘은 저체중(Underweight) 및 3급 비만(Class 3 Obesity) BMI 범주의 개인들에게 특히 영향을 주는 스왑 문제에 대한 정확한 해결책을 제공합니다. 이를 통해 화학적 위상 MRIs에서 신뢰할 수 있는 PDFF 추정을 가능하게 하여 대규모 인구 조사 연구에서의 자동화된 분석을 위한 견고한 기술적 기초를 제공합니다.



### Monocular Depth Estimation and Segmentation for Transparent Object with Iterative Semantic and Geometric Fusion (https://arxiv.org/abs/2502.14616)
Comments:
          Accepted by ICRA(2025). The code is accessible through: this https URL

- **What's New**: 본 논문에서는 투명한 물체의 세그멘테이션(segmentation)과 깊이 추정(depth estimation)을 동시에 수행할 수 있는 단일 RGB 이미지 입력을 사용하는 단안(monocular) 프레임워크를 제안합니다. 이는 기존의 방법들이 부족했던 상호작용을 고려하여, 두 작업 간의 유용한 정보를 활용하도록 설계되었습니다. 또한, 사람의 지각 메커니즘을 참고하여 초기 특징을 점진적으로 개선하는 반복(iterative) 전략을 도입했습니다.

- **Technical Details**: 제안된 모델은 transformer 기반의 인코더(encoders)와 재조합 모듈, 반복적인 융합(decoder)을 활용하여 세그멘테이션과 깊이 지도를 생성합니다. 입력된 RGB 이미지는 여러 개의 transformer 블록을 통과하여 멀티스케일 특징 맵으로 변환되고, 이는 각 세그멘테이션과 깊이에 대한 별도의 특징 피라미드를 형성합니다. 이후, 이 두 브랜치는 새로운 의미 및 기하학적 융합 모듈을 통해 결합되며, 반복적으로 개선됩니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 합성 및 실제 데이터셋에서 최신 단안 및 멀티뷰 방법들의 성능을 상당히 초과했습니다. 특히, 깊이 및 세그멘테이션의 정확도 평균 38.8%-46.2% 개선된 결과를 보여주었습니다. 이러한 성능 향상은 제안된 융합 모듈과 새로운 반복 전략 덕분으로, 투명한 물체에 대한 인식 성능을 크게 향상시켰습니다.



### Self-supervised Monocular Depth Estimation Robust to Reflective Surface Leveraged by Triplet Mining (https://arxiv.org/abs/2502.14573)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이번 연구에서는 Self-supervised monocular depth estimation (SSMDE)에서 반사 구역을 정확히 파악하기 위한 새로운 훈련 전략인 'reflection-aware triplet mining'을 제안합니다. 이는 기존 방법들이 비Lambertian(비램버트) 표면에서 발생하는 문제를 해결하는 데 초점을 맞추고 있습니다. 또한, 종속 모델을 통해 반사와 비반사 영역에서 선택적으로 지식을 학습하는 'reflection-aware knowledge distillation' 방식을 도입하여 깊이 추정의 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 카메라의 기하학적 정보를 활용하여 픽셀 수준에서 반사 영역을 식별합니다. 반사 영역은 다른 뷰어 간의 상대적인 포토메트릭 오차가 낮게 나타나는 특징을 보입니다. 이 방법은 서로 다른 카메라 좌표에서의 뷰를 부정적으로 간주하고, 동일 좌표의 뷰를 긍정적으로 간주하여 포토메트릭 오차를 최소화하고 최대화하는 배열을 통해 반사 영역의 정확성을 크게 향상시킵니다.

- **Performance Highlights**: 여러 데이터 세트에서의 평가 결과, 제안된 방법이 반사 표면에서 깊이 품질을 효과적으로 개선하고 기존 SSMDE 기준선보다 뛰어난 성능을 보였습니다. 이 방법은 반사 비반사 영역 모두에서 안정적인 깊이 추정을 가능하게 하며, 고주파 세부 사항을 유지하는 데에도 기여합니다. 제안된 방법은 기존의 자가 감독 훈련 방법과 3D 정보 증류 방법에 비해 우수한 성능을 나타냅니다.



### Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidanc (https://arxiv.org/abs/2502.14520)
- **What's New**: 이번 논문에서 제시된 FlowScene은 Optical Flow 가이드를 통해 시간에 따른 3D Semantic Scene Completion (SSC) 문제를 해결하는 새로운 접근방식입니다. 기존의 방법들은 한 프레임 또는 멀티 프레임의 정보를 단순히 쌓는 방식으로 제한되어 있었으나, FlowScene은 움직임과 다양한 관점, 그리고 occlusions를 통합하여 SSC의 정확도를 크게 향상시킵니다. 이 연구는 Flow-Guided Temporal Aggregation과 Occlusion-Guided Voxel Refinement 모듈을 포함하여 효과적인 장면 맥락을 캡쳐합니다.

- **Technical Details**: FlowScene은 RGB 이미지와 Optical Flow를 활용하여 3D 장면의 기하학적 요소와 의미를 함께 추론합니다. 특히, Flow-Guided Temporal Aggregation 모듈은 과거 프레임의 정보를 통합하여 움직임 인지 컨텍스트를 포착하고, Occlusion-Guided Voxel Refinement 모듈은 occlusion 마스크를 사용하여 3D voxel 예측을 정제합니다. 이러한 방법들은 기존 3D SSC 접근방식보다 더 나은 결과를 도출하기 위해 설계되었습니다.

- **Performance Highlights**: FlowScene은 SemanticKITTI 및 SSCBench-KITTI-360 벤치마크에서 최첨단 성능을 달성했습니다. 이 연구 결과는 기존 SSC 방법들의 한계를 극복하고, 시간에 따른 컨텍스트 강화를 통해 의미와 기하학을 더 정확하게 파악할 수 있음을 보여줍니다. 또한, 다양한 실험을 통해 FlowScene의 효과가 입증되었습니다.



### PLPHP: Per-Layer Per-Head Vision Token Pruning for Efficient Large Vision-Language Models (https://arxiv.org/abs/2502.14504)
Comments:
          12 pages, 8 figures

- **What's New**: 본 논문은 Large Vision-Language Models (LVLMs)의 효율성을 획기적으로 개선하기 위한 Per-Layer Per-Head Vision Token Pruning (PLPHP) 방법을 제안합니다. 기존의 LVLM은 많은 비주얼 토큰 때문에 추론 효율성에서 제한을 받고 있었고, PLPHP는 이러한 문제를 해결하기 위해 두 가지 레벨의 세밀한 pruning 기법을 도입했습니다. 이 방법은 각 레이어의 비주얼 정보 주목도를 기반으로 토큰 유지율을 동적으로 조정하며, 토큰 손실을 최소화하면서 성능을 향상시킵니다.

- **Technical Details**: PLPHP는 두 가지 주요 구성 요소로, 첫 번째는 layer-level retention rate allocation으로, 각 레이어의 비주얼 정보 주목도를 고려하여 토큰의 보존율을 조정합니다. 두 번째는 head-level vision token pruning으로, 동일한 레이어 내에서도 각 attention head가 독립적으로 중요한 컨텍스트를 보존할 수 있도록 합니다. 이를 통해 PLPHP는 각 레이어의 주목에 따라 비주얼 토큰을 선택적으로 유지하거나 제거하여 성능 저하를 효과적으로 방지합니다.

- **Performance Highlights**: 실험 결과, PLPHP는 decoding 속도를 18% 향상시키고 Key-Value Cache (KV Cache) 크기를 50% 이상 줄이며, 평균 0.46%의 성능 하락을 보였습니다. 또한, 다중 이미지 작업에서 상당한 성능 개선을 달성하여 기법의 효과iveness를 입증하였습니다. 이 연구는 LVLMs의 효율성과 확장성을 높이는 중요한 기여를 합니다.



### LXLv2: Enhanced LiDAR Excluded Lean 3D Object Detection with Fusion of 4D Radar and Camera (https://arxiv.org/abs/2502.14503)
Comments:
          Accepted by IEEE Robotics and Automation Letters

- **What's New**: 본 논문에서는 기존의 LXL 방법의 한계를 극복하고 성능을 향상시키기 위해 LXLv2를 제안합니다. LXLv2는 4D 레이더 포인트를 활용한 일대다 깊이 감독 전략을 도입하여 깊이 예측의 정확성과 일관성을 향상시킵니다. 또한, 채널 및 공간 어텐션 기반의 융합 모듈인 CSAFusion을 통해 특성 적응성을 개선하고 있습니다.

- **Technical Details**: LXLv2는 레이더 측정의 위치 오류를 고려하여, 레이더 점을 통해 객체 수준의 깊이 일관성을 조정하기 위한 슈퍼비전 영역을 조절하는 RCS(레이다 단면) 값 활용 방법을 고안했습니다. 이를 통해, 객체 크기에 따라 다른 슈퍼비전 방법을 적응시키는 것이 가능해집니다. CSAFusion 모듈은 채널 및 공간 어텐션 기법을 결합하여 중요한 피처에 적절히 집중하도록 설계되었습니다.

- **Performance Highlights**: View-of-Delft 및 TJ4DRadSet 데이터셋에서 실험 결과, LXLv2가 LXL보다 1.8% 및 1% 향상된 3D 감지 정확도를 보였으며, 추론 시간도 단축되었습니다. 이러한 결과는 LXLv2 내 각 구성 요소의 효율성을 입증하며, 고성능 자율 주행 시스템 개발에 기여할 수 있습니다.



### Nearshore Underwater Target Detection Meets UAV-borne Hyperspectral Remote Sensing: A Novel Hybrid-level Contrastive Learning Framework and Benchmark Datas (https://arxiv.org/abs/2502.14495)
Comments:
          18pages,13figures

- **What's New**: 이 논문에서는 수중 목표 탐지(Utd)에서의 효율성을 향상시키기 위해 Hyperspectral Underwater Contrastive Learning Network(HUCLNet)를 제안합니다. 기존의 하이퍼스펙트럼 기반 검출 방법이 근해 환경에서 발생하는 분광 왜곡으로 인해 어려움을 겪고 있는 상황을 극복하기 위해, HUCLNet는 대조 학습(contrastive learning) 및 자기 주도 학습(self-paced learning) 패러다임을 통합하여 강력한 수중 목표 탐지를 가능하게 합니다. 이를 통해 기존 방법의 한계를 넘어 좀 더 정확한 탐지를 가능하게 하는 새로운 프레임워크를 제시하였습니다.

- **Technical Details**: HUCLNet은 왜곡된 하이퍼스펙트럼 데이터를 통해 차별적인 특징을 추출하며, 자기 주도 학습(strategy)은 가장 유용한 샘플을 우선적으로 선택하여 학습합니다. 또한 신뢰성 유도 클러스터링(strategy)을 통해 학습된 특징의 강건성을 강화합니다. 본 연구에서는 ATR2-HUTD라는 새로운 근해 HUTD 벤치마크 데이터셋을 구축하였으며, 서로 다른 환경과 목표 유형에 대한 3가지 다양한 시나리오를 포함하고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, HUCLNet은 최신 기술(state-of-the-art) 방법들과 비교했을 때 상당한 성능 향상을 보였습니다. 특히, 근해 환경에서의 수중 목표 탐지 정확도가 크게 향상되었으며, 다양한 수중 조건에서도 높은 신뢰도를 유지합니다. 이 연구의 데이터셋과 코드는 공개적으로 이용 가능하여, 향후 연구자들이 HUTD 알고리즘을 평가하고 개선하는 데 기여할 것으로 기대됩니다.



### CrossFuse: Learning Infrared and Visible Image Fusion by Cross-Sensor Top-K Vision Alignment and Beyond (https://arxiv.org/abs/2502.14493)
Comments:
          IEEE T-CSVT. We mainly discuss the out-of-distribution challenges in infrared and visible image fusion

- **What's New**: 이 논문은 OOD (Out-of-Distribution) 데이터 문제 해결에 초점을 맞추고 있습니다. 기존의 연구들과 달리, 실제 응용에서 OOD 데이터로 인해 발생하는 여러 과제를 다루며, 모델의 강건성(robustness) 및 일반화(generalization) 향상 방법을 제안합니다. 저자는 Multi-View Augmentation을 기반으로 한 전방향 적외선-가시 이미지 융합 프레임워크를 소개하고, 이것이 복잡한 실제 환경에 적응하는 데 어떻게 기여하는지를 강조합니다.

- **Technical Details**: 제안된 방법은 외부 데이터 증강(DA)을 위해 Top-k Selective Vision Alignment를 사용하여, 가시 이미지에서 RGB-wise 변환을 통해 데이터 분포의 차이를 완화합니다. 내부 데이터 증강을 위해 Weak-Aggressive Augmentation을 활용하여 자기 지도 학습(self-supervised learning)을 통해 더욱 강력한 특성 표현(feature representation)을 학습하도록 돕습니다. 이 접근 방식은 실제 환경의 복잡한 시나리오에서 모델의 적응성을 크게 향상시킵니다.

- **Performance Highlights**: 다양한 조건과 환경에서 광범위한 실험을 통해 제안된 방법이 우월한 성능과 강건성을 보여줍니다. 기존 IVIF (Infrared-Visible Image Fusion) 작업의 신뢰성(reliability)과 안정성(stability)을 크게 강화하여 실제 응용에서의 효과적인 활용을 가능하게 합니다. 결과적으로, 모델이 제한된 레이블 데이터로도 다양한 도메인에 걸쳐 일반화 능력을 향상시키는 데 기여합니다.



### Integrating Extra Modality Helps Segmentor Find Camouflaged Objects W (https://arxiv.org/abs/2502.14471)
Comments:
          12 pages, 5 figures, 6 tables

- **What's New**: 본 논문에서는 Camouflaged Object Segmentation (COS)의 성능 향상을 위해 새로운 프레임워크인 UniCOS를 제안합니다. UniCOS는 두 가지 주요 구성 요소인 UniSEG와 UniLearner로 구성되어 다양한 데이터 모달리티를 효과적으로 활용하여 분할 성능을 향상시킵니다. 이 프레임워크는 기존의 단일 모달 COS 접근 방식을 뛰어넘어 다중 모달 데이터를 통합하여 더 나은 성능을 제공합니다.

- **Technical Details**: UniCOS의 UniSEG는 상태 공간 융합 메커니즘(State Space Fusion Mechanism, SSFM)을 사용하여 서로 다른 모달리티의 특징을 통합합니다. 또한, Latent Space Fusion Module (LSFM)을 통해 잠재 공간에서 초기 특징 융합을 수행하고, Feature Feedback Module (FFM)을 통해 융합 결과를 추가 모달 인코더에 재도입하여 후속 특징 추출에 도움을 줍니다. UniLearner는 COS와 관련 없는 RGB-X 데이터셋에서 교차 모달 지식을 학습하여 세분화 네트워크를 안내합니다.

- **Performance Highlights**: UniCOS의 실험 결과, UniSEG는 기존의 Multimodal COS (MCOS) 세분화 기법들을 초월하는 성능을 입증하였습니다. 특히, 실제 또는 가짜 다중 모달 COS 데이터가 없는 경우에도 UniLearner는 단일 모달 COS 성능을 향상시키는 데 효과적입니다. 전반적인 COS 작업에서 우리의 접근 방식은 최첨단 성능을 달성하며, 플러그 앤 플레이 방식의 유연성을 제공합니다.



### Exploiting Deblurring Networks for Radiance Fields (https://arxiv.org/abs/2502.14454)
- **What's New**: 이번 논문에서는 DeepDeblurRF라는 새로운 radiance field (RF) 디블러링 접근법을 제안합니다. 이 방법은 블러 처리된 훈련 이미지로부터 고품질의 새로운 뷰를 합성할 수 있으며, 훈련 시간을 크게 줄일 수 있습니다. DeepDeblurRF는 deep neural network (DNN) 기반의 디블러링 모듈을 활용하여 성능과 계산 효율성을 동시에 향상시킵니다.

- **Technical Details**: DeepDeblurRF는 RF 기반의 디블러링과 radiance field 생성을 교대로 수행하는 반복 프레임워크를 적용합니다. 이 프레임워크는 이미지 디블러링과 RF 생성을 결합하여, 다양한 장면 표현을 처리할 수 있는 범용성을 가지고 있습니다. 특히, RGB-가이드 디블러링 방법론을 통해 초기 디블러링 이미지의 품질이 제한적일지라도, 더 높은 품질의 이미지를 생성할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: DeepDeblurRF는 카메라 모션 블러 및 디포커스 블러 조건 하에서 철저한 실험을 통해 최신의 새로운 뷰 합성 품질을 달성했습니다. 또한, 작은 훈련 세트를 사용하던 기존 접근법들에 비해 성능이 크게 개선되었습니다. 특히, 블러리 이미지로부터의 radiance field 생성이 가능해져, 저조도 환경에서도 실험 결과가 우수한 것으로 나타났습니다.



### Stochastic Resonance Improves the Detection of Low Contrast Images in Deep Learning Models (https://arxiv.org/abs/2502.14442)
Comments:
          MSc Course Project

- **What's New**: 이 논문은 확률적 공명(stochastic resonance)이 비율 기반 인공 신경망(rate-based artificial neural networks)에서 어떻게 발생하는지를 조사했습니다. 확률적 공명은 특정 시스템에서 소음의 도입으로 약한 신호의 탐지 능력이 향상되는 현상입니다. 본 연구에서는 LSTM 반복 신경망을 사용하여 숫자 인식을 수행하였고, 실험에서 소음이 첨가되는 조건에서 모델의 성능 변화가 분석되었습니다.

- **Technical Details**: 모델은 20개의 은닉 유닛을 가진 단일 LSTM 레이어와 softmax 활성화가 적용된 밀집(dense) 레이어로 구성됩니다. 학습에는 MNIST 숫자 데이터셋이 사용되며, 비어 있는 자극 클래스가 수동으로 추가되었습니다. 테스트 중 소음의 영향을 파악하기 위해 비율 기반 LSTM RNN의 입력에 다양한 유형과 수준의 제어된 소음이 추가되었습니다.

- **Performance Highlights**: 실험 결과, 특정 조건에서 소음이 성능에 긍정적인 영향을 미친다는 것을 확인하였습니다. 특히, 낮은 대비의 자극에서는 높은 소음 수준이 유리하게 작용하는 반면, 높은 대비의 자극에서는 낮은 소음 수준에서 가장 좋은 성능을 보였습니다. 본 연구는 비율 기반 신경망에서 확률적 공명이 발생할 수 있음을 첫 번째로 제시하며, 이는 약한 신호에 대한 탐지 성능 향상으로 이어질 수 있습니다.



### Daily Land Surface Temperature Reconstruction in Landsat Cross-Track Areas Using Deep Ensemble Learning With Uncertainty Quantification (https://arxiv.org/abs/2502.14433)
- **What's New**: 이번 연구에서는 DELAG라는 심층 앙상블 학습 방법을 제안합니다. 이 방법은 복잡한 도시 지역에서 Landsat의 LST(land surface temperature) 데이터를 재구성하기 위해 연간 온도 주기(annual temperature cycles)와 가우시안 프로세스(Gaussian processes)를 통합합니다. 뉴욕, 런던, 홍콩 세 도시에서 실험을 통해 DELAG가 중간 및 고도 위도의 도시에 있는 LST 재구성을 성공적으로 수행하는 것을 보였습니다.

- **Technical Details**: Landsat 위성은 16일 간격으로 서로 다른 두 경로를 따라 데이터를 수집하며, 이 과정에서 중복 관측 지역인 cross-track 지역을 활용하여 고해상도 LST 데이터를 생성할 수 있습니다. DELAG는 향상된 ATC 모델과 가우시안 프로세스를 통해 명확한 하늘 픽셀과 구름으로 덮인 픽셀을 연결하며, 이로 인해 예측 구간(prediction intervals)을 생성할 수 있는 첫 번째 방법으로 자리 잡고 있습니다. 이 접근법은 도시 내 온도의 공간 이질성(temperature heterogeneity)을 보다 정확하게 포착할 수 있도록 합니다.

- **Performance Highlights**: DELAG를 사용한 실험 결과, 명확한 하늘 조건에서는 RMSE가 0.73-0.96 K, 구름이 많은 상황에서는 RMSE가 0.84-1.62 K로 나타났습니다. 또한, 구 reconstructed LST 데이터를 사용하여 근접 표면 공기 온도를 추정한 결과는 일관되게 우수하였으며, RMSE가 1.48-2.11 K로 측정되었습니다. 이는 기존 방법보다 높은 신뢰성을 바탕으로 LST 재구성의 신뢰성을 증가시키고, 기후 변화에 대한 복잡한 사건을 해결할 수 있는 새로운 가능성을 제안합니다.



### Evaluating Precise Geolocation Inference Capabilities of Vision Language Models (https://arxiv.org/abs/2502.14412)
Comments:
          AAAI 2025 Workshop DATASAFE

- **What's New**: 이 논문은 시각 언어 모델(Vision-Language Models; VLM)의 개인 정보 보호 문제와 더불어 이러한 모델이 새로운 이미지 데이터에서 지리적 위치를 추론하는 능력을 조사합니다. Google Street View로부터 수집된 벤치마크 데이터셋을 통해 VLM의 이미지 위치 추론 능력 평가를 실시하고, 보조 도구를 사용하는 VLM 에이전트를 평가하여 거리 오차의 최대 30.6% 감소를 관찰하였습니다. 이를 통해 현대의 VLM이 강력한 이미지 위치 추정 도구로 작용할 수 있다는 점을 입증했습니다.

- **Technical Details**: 연구에서는 Google Street View의 1602개 이미지를 사용하여 다양한 도시와 국가에서의 지리적 위치 추정 능력을 평가합니다. 각 이미지는 정확한 위도, 경도 및 API 매개변수로 레이블링되어 있으며, 고정된 시선각과 피치, 무작위 방향을 통해 다양한 뷰를 확보합니다. VLM 모델이 경쟁적인 GeoGuessr 플레이어의 역할을 맡고, 추론 과정과 이미지 내 시각 요소를 기술하도록 지시하는 시스템 프롬프트를 기반으로 평가가 이루어집니다.

- **Performance Highlights**: 기본 VLM의 전체 벤치마크 결과는 뛰어난 정확도를 기록하였으며, O1 모델이 국가와 도시 예측에서 각각 0.8452 및 0.1423의 정확도를 보였습니다. 특히, VLM에 Street View API 접근 권한을 부여했을 때, 모델의 거리 오차는 기본 모델에 비해 현저히 감소하였고, GPT-4o 및 Claude 3.5 Sonnet은 각각 28.1% 및 30.6%의 평균 오차 감소를 달성했습니다. 이는 저비용의 간단한 보조 도구로도 지리적 추정 능력을 상당히 향상시킬 수 있음을 보여줍니다.



### PhotoDoodle: Learning Artistic Image Editing from Few-Shot Pairwise Data (https://arxiv.org/abs/2502.14397)
- **What's New**: 새로운 이미지 편집 프레임워크인 PhotoDoodle을 소개합니다. 이 방법은 아티스트가 사진위에 장식 요소를 겹쳐 놓는 과정을 간소화하고자 개발되었습니다. 이전의 방법들은 스타일 이전(global style transfer)이나 지역 인페인팅(regional inpainting)에 초점을 맞췄으나, PhotoDoodle은 세밀한 배경 보존과 조화로운 요소 통합을 동시에 달성하기 위해 설계되었습니다.

- **Technical Details**: PhotoDoodle은 두 단계의 훈련 전략을 사용합니다. 첫 번째로, 일반 목적의 이미지 편집 모델인 OmniEditor를 대규모 데이터를 사용해 훈련합니다. 그 후, EditLoRA를 통해 적은 수의 아티스트가 큐레이션한 이미지 쌍을 사용하여 고유한 편집 스타일을 반영하도록 모델을 미세 조정합니다. 이 과정에서 위치 인코딩 재사용 메커니즘을 도입하여 결과의 일관성을 높였습니다.

- **Performance Highlights**: 실험 결과, PhotoDoodle은 사용자 맞춤형 이미지 편집에서 고급 성능과 견고함을 보여주었습니다. 이는 아티스트의 독특한 스타일을 효과적으로 캡처하며, 새로운 예술적 창작 가능성을 열어줍니다. 또한, 6가지 고품질 스타일로 구성된 PhotoDoodle 데이터세트를 공개하였으며, 반복 가능한 연구의 기준점을 설정하였습니다.



### RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers (https://arxiv.org/abs/2502.14377)
Comments:
          15 pages, 9 figures

- **What's New**: 본 논문에서는 Diffusion Transformer(확산 변환기)의 효율성을 개선하기 위한 새로운 프레임워크인 RelaCtrl을 제안합니다. 이 프레임워크는 컨트롤(제어) 신호의 중요성을 분석하여, 각 레이어에 적합하게 신호를 통합할 수 있게 합니다. 이를 통해 기존 방식에 비해 필요한 매개변수와 계산 비용을 크게 줄일 수 있습니다.

- **Technical Details**: RelaCtrl은 'ControlNet Relevance Score'를 통해 각 레이어의 컨트롤 정보 관련성을 평가합니다. 중요성이 높은 레이어에 컨트롤 블록을 배치하고, 중요성이 낮은 레이어는 빈 공간으로 둠으로써 불필요한 매개변수를 줄이는 방식입니다. 두 차원 셔플 믹서(TDSM)는 기존의 self-attention 및 FFN을 대체하여 효율적인 토큰 혼합과 채널 혼합을 구현합니다.

- **Performance Highlights**: 실험 결과에 따르면, RelaCtrl은 PixArt-δ에 비해 단 15%의 매개변수로 뛰어난 성능을 발휘하며, 여전히 높은 품질의 이미지 생성을 가능하게 합니다. 다양한 조건부 가이던스 작업에서도 효율성을 유지하며 지속적으로 우수한 결과를 보여줍니다. 이는 제안된 접근 방식의 일반화와 관련 전략의 효과를 검증합니다.



### CrossVTON: Mimicking the Logic Reasoning on Cross-category Virtual Try-on guided by Tri-zone Priors (https://arxiv.org/abs/2502.14373)
- **What's New**: 이 논문에서는 cross-category virtual try-on을 위한 새로운 접근 방식을 제안합니다. 기존의 가상 시도 접근법들이 size mismatch 문제와 이미지의 다양한 기능 지역을 인식하는 데 한계를 보였던 반면, 논문은 인간의 인지 과정을 모델링하여 복잡한 사고 과정을 체계화된 프레임워크로 분해합니다. 이 프레임워크는 모델 이미지를 시도 영역(try-on), 재구성 영역(reconstruction), 그리고 상상 영역(imagination zone)으로 나누어 각 영역의 역할을 명확히 합니다.

- **Technical Details**: 제안된 CrossVTON 방법은 iterative data constructor를 사용하여 다양한 시나리오에서 데이터를 생성합니다. 이 과정은 intra-category try-on, any-to-dress 및 dress-to-any 변환을 포함합니다. 모델과 의상을 기반으로 tri-zone priors generator를 이용하여 각 영역 사이의 정합성을 예측하고, 주어진 의상이 모델 이미지와 어떻게 일치하는지를 분석합니다.

- **Performance Highlights**: CrossVTON은 정성적(qualitative) 및 정량적(quantitative) 평가에서 기존의 기준을 능가하는 성능을 보여줍니다. 특히, cross-category 가상 시도에서의 뛰어난 능력을 입증하며, 실제 응용에서의 복잡한 요구를 충족할 수 있습니다. 이 논문은 cross-category virtual try-on의 성능 향상을 위해 논리적 추론(logical reasoning)을 통합한 새로운 프레임워크를 제시하는 데 큰 기여를 합니다.



### Weed Detection using Convolutional Neural Network (https://arxiv.org/abs/2502.14360)
- **What's New**: 이 연구에서는 농업용지에서 잡초를 탐지하기 위해 합성곱 신경망(convolutional neural networks, CNNs)을 사용합니다. 연구의 주요 초점은 Conv2d와 dilated Conv2d 두 가지 CNN 레이어 유형의 적용에 있습니다. 이 방법은 사전 훈련된 모델을 이용해 입력 이미지에서 특징을 추출한 후 잡초 탐지를 위해 조정합니다.

- **Technical Details**: 실험은 15336개의 데이터 세그먼트로 구성된 대규모 데이터셋을 사용하여 진행되었습니다. 이 데이터는 3249개의 토양, 7376개의 대두, 3520개의 풀, 1191개의 넓은 잎 잡초로 구성되어 있습니다. 제안된 방법은 잡초 탐지에서 94%의 정확도로 효과적으로 작동한다는 실험 결과를 보여줍니다.

- **Performance Highlights**: 이 연구 결과는 독성 제초제의 사용을 줄이고 농업에서 잡초 관리의 효율성을 높이는 데 중요한 의미를 갖습니다. 높은 정확도는 농업 실무자들에게 잡초 탐지의 신뢰성을 제공하며, 지속 가능한 농업 방식에 기여할 수 있습니다.



### Triply Laplacian Scale Mixture Modeling for Seismic Data Noise Suppression (https://arxiv.org/abs/2502.14355)
- **What's New**: 이번 논문에서는 트리플 라플라시안 스케일 혼합(TLSM) 접근법을 제안하여 지진 데이터의 노이즈 억제 성능을 크게 향상시킵니다. 기존의 희소성 기반 텐서 복구 방법은 잡음이 많은 실 지진 데이터에서 희소 텐서 계수의 분산을 정확하게 추정하는 데 한계가 있었으나, TLSM은 이를 공동 추정할 수 있게 합니다. 또한, ADMM(Alternating Direction Method of Multipliers) 알고리즘을 활용하여 최적화를 보다 용이하게 하고 안정성을 확보합니다.

- **Technical Details**: 이 연구에서는 TLSM 프레임워크를 사용하여 지진 데이터의 노이즈 억제를 위한 새로운 접근 방식을 제안합니다. 각 희소 텐서 계수를 라플라시안 분포로 모델링하며, 이는 긍정적인 스칼라 배수를 포함하여 노이즈 관측값에서 직접 희소 텐서 계수의 분산과 값을 추정할 수 있게 합니다. 이러한 방식은 희소 분포 사전(Sparse Distribution Prior)을 통해 이루어집니다.

- **Performance Highlights**: 제안된 TLSM 알고리즘은 합성 및 현장 지진 데이터에서 광범위한 실험을 통해 정량적 및 정성적 평가 모두에서 기존의 최신 기술들을 초월하는 성능을 보였습니다. 특히, 계산 효율성 또한 뛰어난 것으로 나타나, 새로운 알고리즘의 적용 가능성과 유용성을 강조합니다.



### SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images (https://arxiv.org/abs/2502.14351)
- **What's New**: 최근의 PET (Positron Emission Tomography) 이미징에서의 세분화(세그멘테이션)에 대한 문제를 해결하기 위해, 연구팀은 5,731개의 3차원 PET 이미지를 포함하는 가장 큰 PET 세분화 데이터셋인 PETS-5k를 구축했습니다. 이를 바탕으로 세분화의 정확성을 높이기 위해 SegAnyPET이라는 모드별 모델을 개발했습니다. SegAnyPET은 고품질 및 저품질의 레이블이 부착된 데이터로부터 세분화를 robust하게 배울 수 있도록 하는 cross prompting confident learning (CPCL) 전략을 채택했습니다.

- **Technical Details**: SegAnyPET 모델은 3D 아키텍처로 구성되어 있으며, 이는 3차원 PET 이미지의 슬라이스 간 맥락 정보를 완벽하게 활용하기 위함입니다. CPCL 전략은 불확실성을 기반으로 한 자기 정정 과정을 통해 서로 다른 주석 품질의 데이터로부터 효율적으로 학습할 수 있도록 합니다. 이 모델은 입력받은 최초의 몇 개의 프롬프트 포인트만으로도 목표를 식별할 수 있습니다.

- **Performance Highlights**: SegAnyPET은 과거의 세그멘테이션 모델들과 비교하여 뛰어난 정확성을 보여주며, 하나 또는 몇 개의 프롬프트 포인트로 이전 및 새로운 타겟을 정확하게 세분화할 수 있습니다. 실험 결과, 이 모델은 가장 앞선 기초 모델들과 특정 작업에 대비한 감독 모델들보다 더 높은 일반화 능력을 발휘하는 것으로 나타났습니다. SegAnyPET의 도입은 분자 이미징의 다양한 다운스트림 작업에 대한 응용 가능성을 확대할 것으로 기대됩니다.



### Towards Accurate Binary Spiking Neural Networks: Learning with Adaptive Gradient Modulation Mechanism (https://arxiv.org/abs/2502.14344)
Comments:
          9 pages, 8 figures, AAAI conference

- **What's New**: 본 논문에서는 Binary Spiking Neural Networks (BSNNs)의 학습 과정에서 발생하는 가중치 부호 뒤집기 문제를 심도 있게 분석합니다. 이를 해결하기 위해 Adaptive Gradient Modulation Mechanism (AGMM)을 제안하여, 학습 과정 중 가중치의 부호 뒤집기 빈도를 줄이도록 설계하였습니다. AGMM을 통합함으로써 BSNN들은 더 빠른 수렴 속도와 높은 정확도를 달성할 수 있습니다.

- **Technical Details**: BSNN은 이진화 기법을 통해 가벼우면서도 에너지 효율적인 특성을 지니며, 주로 리소스가 제한된 엣지 디바이스에 배치되기 적합합니다. BSNN 학습 과정에서 발생하는 가중치 부호 뒤집기 문제는 가중치의 이진값(-1과 +1)으로 인해 더욱 심각하게 나타납니다. AGMM은_gradients_의 크기를 적절히 조정하여 이러한 문제를 해결하고, 이를 통해 BSNN의 성능을 향상시킵니다.

- **Performance Highlights**: AGMM을 적용한 BSNN은 정적(static) 및 신경형(neuromorphic) 데이터셋에서 최첨단 성능을 발휘했습니다. 테스트 결과, BSNN의 정확도가 개선되었으며, FN-SNN(Full-precision SNN)과의 성능 차이가 효과적으로 줄어들었습니다. 이 연구는 SNN의 본래 에너지 효율성을 강화하고 저장 요구 사항을 대폭 줄이며, 리소스 제약이 있는 환경에서의 실용성을 높입니다.



### A Collaborative Jade Recognition System for Mobile Devices Based on Lightweight and Large Models (https://arxiv.org/abs/2502.14332)
- **What's New**: 이번 논문에서는 모바일 기기에서 효율적이고 정확한 옥 인식을 위한 새로운 시스템을 제안합니다. 이 시스템은 다중 스케일 이미지 처리를 기반으로 한다는 점에서 혁신적이며, 작은 모델과 큰 모델의 협업을 통해 모바일 장치에서 인식 정확도를 높이는 데 중점을 두고 있습니다. 이를 통해 기존의 주관적인 옥 감정 방식에 비해 보다 객观적이고 신뢰할 수 있는 솔루션을 제공합니다.

- **Technical Details**: 제안된 시스템은 경량 모델을 모바일 기기에 배치하고, 클라우드 또는 엣지 서버에 큰 모델을 배치하여 특징 추출 및 이차 검증을 수행합니다. 이 시스템은 데이터 교환 및 작업 분배를 위한 효과적인 메커니즘을 갖추고 있어 두 모델이 서로의 강점을 활용할 수 있도록 설계되었습니다. 또한, 합성곱 신경망(convolutional neural network)과 전통적인 컴퓨터 비전 알고리즘을 조합한 다중 모델 분류 프레임워크를 사용하여 다양한 옥 특성에 따라 모델을 신속하게 선택하고 조정할 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 모바일 기기에서도 높은 인식 정확도와 빠른 처리 시간을 제공함을 보여주었습니다. 실험 결과, 다양한 환경에서 높은 정확성과 견고한 성능을 입증하였으며, 이는 향후 문화유산 보호 및 지능형 감정 기술 개발에 기여할 것으로 기대됩니다. 경량 모델을 활용하여 연산 자원을 절약하면서도 실시간 성능을 유지할 수 있는 가능성을 제시합니다.



### Textured 3D Regenerative Morphing with 3D Diffusion Prior (https://arxiv.org/abs/2502.14316)
- **What's New**: 이번 연구에서 우리는 텍스처가 있는 3D 모핑에서의 한계를 극복하기 위해 3D 확산 우선순위(3D diffusion prior)를 활용한 새로운 방법론을 제안합니다. 기존의 방법과 달리, 우리는 명시적인 대응 관계를 요구하지 않으면서 매끄럽고 그럴듯한 변환을 생성할 수 있습니다. 이를 통해 3D 객체 간의 변형 보다는 3D 배열을 재현하는 것이 가능해졌습니다.

- **Technical Details**: 우리의 접근법은 3D 확산 모델을 사용하여 특정 매개변수(levels)를 기반으로 소스와 목표 정보 간의 보간을 수행합니다. 특히, 초기 노이즈, 모델 파라미터, 조건적 특징에서의 정보를 통합하여 3D 제공 모델을 통해 재생성합니다. 또한, Attention Fusion 전략을 통해 매끄러운 모핑 시퀀스를 생성하고, 두 가지 전략(Token Reordering 및 Low-Frequency Enhancement)을 통해 생성된 3D 표면의 질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 다양한 객체 쌍 간의 3D 모핑에서 탁월한 매끄러움과 그럴듯함을 달성하였습니다. 특히 기존 방법이 직면한 두 가지 주요 과제를 극복하며, 텍스처가 있는 3D 표현을 효과적으로 다루는 첫 번째 일반 3D 확산 우선 순위를 사용한 모핑 방법으로 자리 매김할 수 있었습니다. 이러한 결과는 3D 모핑 분야에 새로운 가능성을 제시합니다.



### ODVerse33: Is the New YOLO Version Always Better? A Multi Domain benchmark from YOLO v5 to v11 (https://arxiv.org/abs/2502.14314)
Comments:
          18 pages, 4 figures, 7 tables

- **What's New**: 이번 연구에서는 You Look Only Once (YOLO) 모델의 주요 혁신들을 YOLOv1부터 YOLOv11까지 종합적으로 정리하였습니다. 또한, 11개의 다양한 도메인(자율주행, 농업, 수중, 의료, 비디오 게임, 산업, 공중, 야생동물, 소매, 미세조직, 보안)을 포괄하는 33개의 데이터셋을 포함한 ODverse33 벤치마크를 소개합니다. 이 연구는 YOLO 모델 사용자의 연구 및 미래 개발에 대한 가이드라인을 제공할 것으로 기대됩니다.

- **Technical Details**: YOLO는 실시간 객체 탐지를 위한 모델로, 각 버전에서의 핵심 혁신이 무엇인지 그리고 이러한 변화가 실제 성능 개선으로 어떻게 이어지는지를 탐구합니다. 또한, ODverse33 벤치마크를 통해 다양한 도메인에서의 모델 개선이 실세계에 미치는 영향을 폭넓은 실험 결과를 통해 분석합니다. 이런 접근 방식은 객체 탐지 성능 평가의 표준화에도 기여하는 중대한 연구입니다.

- **Performance Highlights**: 연구 결과, YOLO의 각 버전은 각기 다른 상황에서 실질적인 성능 향상을 이끌어냈으며, 이를 통해 다양한 도메인에서의 활용 가능성을 높였습니다. ODverse33을 기반으로 한 실험은 여러 환경에서 YOLO 모델의 유용성을 입증하며, 향후 객체 탐지 기술의 발전에 기여할 것으로 예상됩니다. 최종적으로, YOLO의 발전이 현실 세계의 다양한 응용 프로그램에 미치는 긍정적인 영향을 강조합니다.



### PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC (https://arxiv.org/abs/2502.14282)
Comments:
          14 pages, 7 figures

- **What's New**: 최근 MLLM(대형 다중모드 언어 모델) 기반 GUI 에이전트 분야에서 PC 시나리오는 스마트폰에 비해 더 복잡한 상호작용 환경을 특징으로 하며 내부 및 외부 애플리케이션 간의 작업 흐름이 복잡하게 얽혀 있습니다. 이러한 문제를 해결하기 위해 제안된 PC-Agent는 계층적 에이전트 프레임워크로, 현재 MLLM의 스크린샷 콘텐츠 지각 능력 부족을 극복하기 위한 Active Perception Module (APM)과 복잡한 사용자 지침을 처리하기 위한 계층적 다중 에이전트 협업 아키텍처를 포함합니다.

- **Technical Details**: PC-Agent는 사용자 지침을 Instruction-Subtask-Action의 세 가지 수준으로 분해하고 각 수준에서 서로 다른 에이전트를 통해 작업을 관리합니다.  여기서 Manager Agent는 지침을 세부 작업으로 해체하고, Progress Agent는 작업 진행을 추적하며, Decision Agent는 APM의 지각 정보를 결합하여 단계별로 결정을 내립니다. 또한, Reflection Agent는 실행 후의 오류 피드백을 통해 시스템을 조정할 수 있도록 지원합니다.

- **Performance Highlights**: PC-Eval이라는 새로운 벤치마크를 통해 PC-Agent는 이전 최첨단 방법에 비해 작업 성공률을 32% 향상시키는 것으로 나타났습니다. 실험 결과, PC-Agent는 다중 애플리케이션과 복잡한 사용자 지침을 처리하는 데 있어 기존 방법보다 크게 개선된 성과를 보여주며, 복잡한 작업을 해결하는 데 있어 효과적인 프레임워크임을 입증했습니다.



### OrchardDepth: Precise Metric Depth Estimation of Orchard Scene from Monocular Camera Images (https://arxiv.org/abs/2502.14279)
Comments:
          10 pages, 5 figures, Australasian Conference on Robotics and Automation, ACRA, 2024

- **What's New**: 본 연구는 OrchardDepth라는 새로운 단안을 제안하여 과수원과 포도원 환경에서 단안 깊이 추정(Monocular Depth Estimation, MDE)의 격차를 해소합니다. 이전 연구들은 대부분 도시 환경에 초점을 맞추었으며, 농촌 환경에 대한 연구는 부족한 상황입니다. 또한, 이 논문은 Dense Depth Map과 Sparse Points 간의 일관성을 모니터링하는 새로운 재훈련 방법을 소개하여 훈련 결과를 향상시킵니다.

- **Technical Details**: 이 연구는 뉴질랜드, 호주 및 미국의 과수원/포도원 환경에서의 이미지와 Sparse LiDAR 포인트 클라우드를 기반으로 깊이 추정 모델을 개발하였습니다. 기존 연구에서 Sparse Points만으로 훈련된 깊이 추정 모델은 깊이 정보의 제한으로 인해 성능 저하를 유발할 수 있음을 보여주었습니다. 이 논문에서는 KITTI 데이터셋에서 생성된 Dense Depth Map의 일관성을 감시하여 자작 데이터셋에서 훈련 중 밀도의 일관성을 유지하는 새로운 방법을 적용합니다.

- **Performance Highlights**: 우리는 농업 환경에서 깊이 추정의 RMSE를 1.5337에서 0.6738로 개선하였습니다. 이 연구는 MDE 작업에서 최고의 성능을 달성하였으며, 기존 단안 깊이 추정 모델의 성능 한계를 극복하는 방법을 제시합니다. 이러한 성과는 1차 산업 분야의 관련 연구에 새로운 통찰을 제공합니다.



### LLM-EvRep: Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework (https://arxiv.org/abs/2502.14273)
Comments:
          6 pages, 2 figures,Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25)

- **What's New**: 이번 연구에서는 LLM(EvGen)이라는 이벤트 기반 인식 방법을 제안합니다. 이 방법은 기존의 복잡한 훈련 과정에 의존하지 않고, 이벤트 기반 비주얼 콘텐츠를 효율적으로 처리할 수 있는 가능성을 열어줍니다. LLM(EvRep)이라는 LLM 호환 이벤트 표현을 생성하여 이벤트 인식 성능을 향상시키는 데 주력하고 있습니다.

- **Technical Details**: LLM-EvGen은 자기 감독(self-supervised) 프레임워크를 사용하여 이벤트 표현을 생성합니다. 이 과정에서 생성된 표현은 의미적 일관성(semantic consistency)과 구조적 신뢰성(structural fidelity)을 모두 고려하여 조정됩니다. 연구는 N-ImageNet, N-Caltech101, N-MNIST의 세 가지 데이터 세트를 사용하여 수행되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 LLM-EvRep 방법이 이벤트-비디오(even-to-video) 방법인 E2VID보다 인식 작업에서 각각 15.93%, 0.82%, 50.21%의 성능 향상을 보였습니다. 이러한 결과는 LLM을 활용한 이벤트 인식의 가능성과 효율성을 보여줍니다.



### Money Recognition for the Visually Impaired: A Case Study on Sri Lankan Banknotes (https://arxiv.org/abs/2502.14267)
- **What's New**: 이번 연구에서는 스리랑카의 통화 지폐를 인식하기 위한 사용자 친화적인 독립형 시스템을 제안합니다. 이를 위해 스리랑카 통화 지폐의 이미지로 구성된 커스텀 데이터셋을 사용하여 EfficientDet 모델을 미세 조정했습니다. 연구 결과, currency note recognition 모델은 검증 데이터셋에서 0.9847의 AP를 달성하여 실제 환경에서도 우수한 성능을 보였습니다. 이 시스템은 시각 장애인이 쉽고 정확하게 통화 단위를 인식할 수 있도록 지원하여 접근성과 독립성을 향상시킵니다.

- **Technical Details**: 연구의 목표는 비전 장애인(VIP)이 다양한 배경에서 스리랑카 통화를 효율적으로 인식할 수 있는 사용자 친화적인 인터페이스를 개발하는 것입니다. 스리랑카의 데이터 증강 기법을 사용하여 부족한 통화 이미지 데이터를 확장하고, 이를 통해 더 높은 모델 정확도를 달성할 수 있습니다. 제안된 시스템은 실시간으로 통화 지폐를 인식할 수 있는 모바일 애플리케이션으로, 뛰어난 정확도를 제공하고 시각 장애인에게 유용한 도구가 되도록 설계되었습니다.

- **Performance Highlights**: 연구에서 제안한 EfficientDet 모델은 검증 데이터셋에서 0.9847의 평균 정확도를 달성하여 우수한 성능을 보입니다. 젊은 VIP가 모바일 기기를 통해 통화 단위를 신속하게 인식할 수 있도록 설계되었으며, 이는 자율성과 접근성을 증대시키는 데 큰 기여를 할 것으로 기대됩니다. 또한, 이 시스템은 저조도 및 다양한 배경에서도 높은 인식 정확도를 유지하여 실제 사용 환경에서도 효과적인 성과를 입증하고 있습니다.



### OG-Gaussian: Occupancy Based Street Gaussians for Autonomous Driving (https://arxiv.org/abs/2502.14235)
- **What's New**: 본 논문에서는 OG-Gaussian이라는 새로운 접근 방식을 제안하여 자율주행 장면 재구성을 위한 Occupancy Grid (OG)를 활용합니다. 기존 LiDAR 포인트 클라우드 대신 주변 카메라 이미지를 사용해 OG를 생성을 통해 동적 객체와 정적 배경을 효과적으로 분리합니다. 이 방식은 고비용의 LiDAR에 의존하지 않고도 고품질의 3D 장면을 신속하게 재구성할 수 있는 방법을 제공합니다.

- **Technical Details**: OG-Gaussian은 Occupancy Prediction Network (ONet)를 활용해 주변 세상을 나타내는 VOC(명세 그리드)를 생성하여 동적 차량과 정적 배경을 구분합니다. 이 후, 배경 거리의 OG에서 포인트 클라우드를 추출하고, 이들을 최적화 가능한 Gaussian 타원체 형태로 변환합니다. 학습 가능한 매개변수를 정의하여 동적 객체의 위치와 자세를 최적화해 감지 및 추적할 수 있는 기법을 제안합니다.

- **Performance Highlights**: Waymo Open 데이터셋에서 수행된 실험에서는 OG-Gaussian이 평균 PSNR 35.13 및 렌더링 속도 143 FPS를 달성하여 현재 최첨단 재구성 품질 및 속도와 동등한 결과를 보여주었습니다. 또한, 처리된 OG를 사용한 사전 세팅으로 효율성을 입증하는 억제 연구도 수행되어, 비용과 시간을 절감하며 자율주행 장면 재구성의 새로운 길을 열었습니다.



### Designing Parameter and Compute Efficient Diffusion Transformers using Distillation (https://arxiv.org/abs/2502.14226)
Comments:
          4 pages

- **What's New**: 이번 연구에서는 Diffusion Transformers (DiTs)의 효율적인 설계를 위한 원칙을 제안하고, 이를 통해 자원이 제한된 Edge 장치에서의 성능 개선을 목적으로 합니다. 특히 Teaching Assistant (TA) 방법과 Multi-In-One (MI1) 방법을 도입하여 기존 솔루션과의 차별성을 강조하고 있습니다. 또한, 모델의 크기, 성능 및 속도 간의 중요한 트레이드오프를 논의합니다.

- **Technical Details**: 디자인을 위한 여러 매개변수 중에서는 깊이(depth), 폭(width), 주목 머리 수(attention heads)와 같은 요소를 선택했습니다. 이러한 매개변수는 효율성과 성능에 모두 영향을 주며, 특히 LPIPS 손실 함수(loss function)를 사용하여 최적의 결과를 도출하고 있습니다. 새로운 디스틸레이션 방법을 통해, 모델의 파라미터 수와 레이턴시를 조정하여 Edge 장치에서의 효율적 실행이 가능하도록 설계하였습니다.

- **Performance Highlights**: 실험을 통해, 제안한 디자인 원칙에 따라 얻어진 모델이 모든 지표에서 우수한 성능을 보였으며, FID와 모델 크기 및 레이턴시에 대해 기존의 SOTA 모델들과 비교하여 유의미한 개선이 있음을 보여줍니다. 특히, NVIDIA Jetson Orin Nano와 같은 실제 Edge 장치에서의 성능 평가를 통해 모델의 실용성을 입증하였습니다. 이 연구는 Diffusion Transformer의 발전에 기여할 것으로 기대됩니다.



### H3DE-Net: Efficient and Accurate 3D Landmark Detection in Medical Imaging (https://arxiv.org/abs/2502.14221)
- **What's New**: 이번 연구에서는 H3DE-Net(Hybrid 3D DEtection Net)이라는 새로운 프레임워크를 제안하며, 이는 3D landmark detection을 위한 CNN(Local feature extraction)과 경량화된 attention mechanism(전역 의존성 캡처)의 결합을 특징으로 합니다. 기존의 모델들이 글로벌 맥락을 유지하면서도 계산 비용을 줄이는 데 어려움을 겪고 있었던 반면, H3DE-Net은 계층적 라우팅 전략을 통해 이러한 문제를 해결합니다. 이 모델은 보다 정교한 3D landmark detection을 가능하게 하며, 또한 다중 스케일 기능 융합을 통해 검출 정확도와 강인성을 한층 향상시킵니다.

- **Technical Details**: 전체 연구는 3D-CT 스캔 이미지를 기반으로 치아 및 주변 구조물의 해부학적 랜드마크를 감지하는 데 초점을 맞추고 있습니다. H3DE-Net은 CNN을 통해 효율적인 지역 기능 추출을 수행하며, 경량화된 Transformer 모듈을 통해 전역 맥락 모델링을 강화합니다. 또한, Feature Fusion Module(FFM)을 도입하여 다중 스케일 기능을 통합하여 정밀한 랜드마크 감지를 위한 로컬 및 글로벌 의존성을 효과적으로 캡처합니다.

- **Performance Highlights**: H3DE-Net은 공개된 CT 데이터 세트에서의 실험을 통해 최첨단(SOTA) 성능을 달성하며, 특히 복잡한 해부학적 변형이나 랜드마크 누락과 같은 어려운 시나리오에서 정확성과 강인성이 크게 개선되었습니다. 기존의 기준 모델들과 비교하여 다양한 전달 메트릭에서 성능이 우수함을 입증했습니다. 이와 같은 성과는 H3DE-Net이 3D 랜드마크 검출 분야에서 기대 이상의 성능을 제시하고 있음을 보여줍니다.



### Spatial and Frequency Domain Adaptive Fusion Network for Image Deblurring (https://arxiv.org/abs/2502.14209)
- **What's New**: 이미지 디블러링은 흐릿한 이미지를 복원하기 위해 고해상도 이미지를 재구성하는 작업입니다. 본 논문에서는 기존의 방법들이 공간 영역(spatial domain)과 주파수 영역(frequency domain) 중 하나에만 국한되어 있는 한계를 극복하고자 합니다. 새로운 '공간-주파수 도메인 적응형 융합 네트워크(SFAFNet)'를 제안하여 두 도메인의 정보를 효과적으로 융합하는 접근 방식을 소개합니다.

- **Technical Details**: SFAFNet의 핵심 구성 요소인 '게이티드 공간-주파수 도메인 특징 융합 블록(GSFFBlock)'은 세 가지 주요 모듈로 구성됩니다: 공간 도메인 정보 모듈, 주파수 도메인 정보 동적 생성 모듈(FDGM), 그리고 게이티드 융합 모듈(GFM)입니다. 특히, FDGM에서는 학습 가능한 저역통과 필터를 활용하여 주파수 서브밴드로 특징을 동적으로 분해합니다. 이는 전체 이미지를 아우르는 수용 필드를 포착해 글로벌 컨텍스트 정보를 적응적으로 탐색할 수 있게 합니다.

- **Performance Highlights**: 실험 결과 SFAFNet은 이미지 모션 디블러링 및 디포커스 디블러링과 같은 주요 이미지 디블러링 태스크에서 기존 방법들 대비 뛰어난 성능을 보여주며, GoPro 데이터셋에서 국가 최고의 성과를 달성했습니다. 이러한 결과들은 SFAFNet이 최신 기술들을 초월하는 성능 우위를 지니고 있음을 입증합니다.



### Bridging Text and Vision: A Multi-View Text-Vision Registration Approach for Cross-Modal Place Recognition (https://arxiv.org/abs/2502.14195)
Comments:
          8 pages, 4 figures, conference

- **What's New**: 이번 연구에서는 모바일 로봇이 자연어 이해(Natural Language Understanding)를 활용하여 위치를 파악하고 패키지 배달과 같은 작업을 수행할 수 있도록 하는 새로운 방법을 제안합니다. 기존의 Visual Place Recognition (VPR) 방법들이 단일 시점의 시각적 정보에 의존하는 한계를 극복하기 위해, 본 논문에서는 Text4VPR이라는 새로운 다중 시점(text-vision registration) 접근 방식을 소개합니다. 이는 텍스트 설명만을 이용하여 이미지 데이터베이스와 매칭하는 최초의 방법입니다.

- **Technical Details**: Text4VPR은 frozen T5 언어 모델을 통해 글로벌 텍스트 임베딩을 추출하며, Sinkhorn 알고리즘과 온도 계수를 활용하여 로컬 토큰을 각 클러스터에 할당합니다. 이 과정을 통해 이미지에서 시각적 설명자를 집계하고, 각 텍스트-이미지 쌍 간의 정렬을 강조하여 정확한 텍스트 설명을 위한 훈련을 진행합니다. 추론 단계에서는 Cascaded Cross-Attention Cosine Alignment (CCCA)를 사용하여 텍스트 및 이미지 그룹 간의 내부 불일치를 해결합니다.

- **Performance Highlights**: Street360Loc 데이터셋에서 Text4VPR은 강력한 기준 모델을 구축하였으며, 테스트 세트에서 57%의 최고 정확도와 92%의 상위 10위 정확도를 달성했습니다. 이 결과는 텍스트 설명을 기반으로 한 이미지와의 로컬라이제이션이 가능한 것뿐만 아니라, 앞으로의 발전 가능성이 크다는 것을 보여줍니다.



### Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models (https://arxiv.org/abs/2502.14191)
Comments:
          Dataset available at this https URL

- **What's New**: 이번 논문에서 우리는 언어-비전 모델(Vision-Language Models, VLMs)을 위한 멀티모달 보상 모델 평가를 위한 새로운 베치마크, Multimodal RewardBench를 소개합니다. 이 벤치마크는 일반적 정확성, 선호도, 지식, 추론, 안전성, 시각적 질문-응답(Visual Question Answering, VQA)의 여섯 가지 분야를 포괄하며, 5,211개 텍스트-이미지 프롬프트에 대한 전문가 주석으로 구성되어 있습니다. 기존의 VLM 보상 모델 평가가 텍스트 모드에 국한되어 있었던 반면, 이 베치마크는 멀티모달 환경에서의 평가를 가능하게 합니다.

- **Technical Details**: Multimodal RewardBench는 프롬프트와 선택된 반응, 거부된 반응의 트리플릿(세 쌍의 데이터)을 사용하여 VLM 보상 모델의 성능을 평가하는 데 중점을 둡니다. 평가에는 여러 VLM 모델이 포함되며, 각각의 영역별로 정확한 응답과 비정확한 응답 간의 구별이 가능하도록 구성되었습니다. 연구 결과, 상위 성능 모델인 Gemini 1.5 Pro와 Claude 3.5 Sonnet이 각각 72%의 전반적인 정확도를 기록하였으나, 대부분의 모델이 추론 및 안전성 분야에서 어려움을 겪고 있는 것으로 나타났습니다.

- **Performance Highlights**: 대부분의 VLM 보상 모델이 무작위 추측(50% 정확도)을 초과하여 성능을 보였으나, 여전히 인간의 성능 수준에는 미치지 못하고 있습니다. 특히, 수학 및 코딩 관련 추론 과제와 독성 감지 같은 안전성 과제에서 많은 모델이 어려움을 겪고 있는 점이 두드러졌습니다. 이러한 결과는 Multimodal RewardBench가 멀티모달 보상 모델 개발을 위한 도전적인 테스트베드임을 시사합니다.



### Stereo Image Coding for Machines with Joint Visual Feature Compression (https://arxiv.org/abs/2502.14190)
- **What's New**: 이 논문에서는 스테레오 이미지 압축(SIC)을 위한 기계 비전 지향의 스테레오 기능 압축 네트워크(MVSFC-Net)를 제안합니다. MVSFC-Net은 스테레오 시각 기능을 효과적으로 추출하고 압축하여 3D 시각 작업을 수행하도록 설계되었습니다. 이를 통해 기존의 2D 이미지 코딩 방식보다 우수한 압축 효율성을 달성합니다.

- **Technical Details**: MVSFC-Net은 스테레오 다중 스케일 특징 압축(SMFC) 모듈을 활용하여 공간, 뷰 간(inter-view), 그리고 크로스 스케일의 중복성을 동시에 제거하여 압축 효율성을 극대화합니다. MVSFC-Net에서 스테레오 시각 기능은 높은 차원의 데이터 크기를 가지지만, 정보 엔트로피는 상대적으로 낮습니다. 이 특성 덕분에 스테레오 기능의 중복성을 효율적으로 제거하는 것이 가능합니다.

- **Performance Highlights**: 실험 결과, MVSFC-Net은 MPEG에서 권장하는 기존의 ICM 앵커 및 최신 SIC 방법에 비해 3D 비주얼 작업 성능과 압축 효율성 모두에서 현저한 개선을 보였습니다. 이러한 성과는 특히 낮은 비트레이트에서 더욱 두드러지며, 새로운 스테레오 이미지 코딩 연구의 가능성을 제시합니다.



### Bayesian SegNet for Semantic Segmentation with Improved Interpretation of Microstructural Evolution During Irradiation of Materials (https://arxiv.org/abs/2502.14184)
- **What's New**: 이 연구에서는 방사선 조사된 LiAlO2 펠렛의 미세구조 진화를 이해하고, 삼중수소(tritium)의 확산, 저장 및 방출과의 관계를 파악하여 삼중수소 생산용 가연성 흡수봉의 성능 예측을 향상시키고자 하였습니다. 전문가가 라벨링한 이미지 데이터를 기반으로 Deep Convolutional Neural Networks(DCNN)을 훈련시켜 결함, 입자 및 경계 클래스로 이미지를 세분화하는 방법을 개발했습니다. 이 과정에서 새로운 메타 데이터와 불확실성 정량화를 포함하여 모델의 민감도를 향상시키기 위한 변형을 시험하였습니다. 최종적으로, 모델이 예측한 세분화 결과는 전문가 라벨링과 유사한 성과를 보였으며, 이는 신경망 모델이 전문가 라벨 이미지의 대안이 될 수 있음을 시사합니다.

- **Technical Details**: 연구는 LiAlO2 펠렛의 방사선 조사와 관련된 미세구조를 캐릭터화하는 것을 목표로 하였으며, 이를 위해 방사선 조사된 및 비조사된 펠렛의 이미지를 세분화하는 DCNN 모델을 훈련시켰습니다. 방사선 조사된 펠렛은 텍사스의 Watts Bar 원자로에서 18개월간 조사된 후 샘플링되었습니다. 연구에서는 이미지를 비율, 결함 면적, 결함 밀도와 같은 방법으로 정량화 상태를 평가하고 서로 비교하는 과정도 포함되었습니다. 이 모든 과정에서는 비조사된 펠렛과 조사된 펠렛 이미지의 미세구조적 양이 비교되었습니다.

- **Performance Highlights**: 연구 결과, 최적의 모델은 방사선 조사된 이미지와 비조사된 이미지에서 모두 높은 성능 지표를 보여주었습니다. 세분화된 이미지와 전문가 라벨링 간의 일관성은 이 모델이 이미지 세분화 작업에서 효과적임을 입증합니다. DCNN을 사용한 이 연구는 전문가의 수작업을 대체할 가능성을 보여주며, 미세구조 분석의 효율성을 크게 향상시킬 수 있음을 시사합니다.



### Deep learning based infrared small object segmentation: Challenges and future directions (https://arxiv.org/abs/2502.14168)
Comments:
          This is a submitted version of a paper accepted by Information Fusion. If you want a better reading experience, please refer to the final published version of Information Fusion

- **What's New**: 이 논문은 최근의 심층 학습 기법을 기반으로 한 적외선 객체 인식 방법에 대해 비판적으로 분석하고, 존재하는 도전 과제를 파악하여 향후 연구 방향을 제시하는 종합적인 리뷰를 제공합니다. 특히, 논문에서는 기존의 적외선 물체 세분화 기법의 한계와 문제점을 강조하며, 다양한 접근 방식의 동기를 소개합니다. 또한, 적외선 감지 분야의 최신 발전을 반영하여 미래의 유망한 연구 방향도 제안하고 있습니다.

- **Technical Details**: 적외선 이미징은 자율 주행 자동차, 드론 및 감시 분야에서 사용되며, 저조도 및 harsh weather와 같은 어려운 조건에서도 뛰어난 성능을 발휘합니다. 그러나 적외선 데이터는 특수한 애플리케이션으로 인해 수집하기 어려운 레이블된 데이터가 부족하고, 낮은 신호 대 잡음 비율(SNR)이라는 도전이 존재합니다. 또한, 작은 물체 인식의 어려움, 데이터의 불균형, 그리고 다양한 애플리케이션 시나리오 간의 모델 전이 문제 등이 제기됩니다.

- **Performance Highlights**: 이 논문에서는 다양한 기존 방법들의 성과를 비교하고, 적외선 비전의 세밀한 인식을 위해 효과적인 특징 모델링 기술이 필요하다고 강조합니다. 현재의 방법들은 Encoder-Decoder 구조를 기반으로 하여 dim하고 작은 객체를 인식하는 데 중점을 두고 있습니다. 이를 통해, 제안된 기법들이 실제 애플리케이션에서 어떻게 성능을 향상시킬 수 있는지를 보여줍니다.



### Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration (https://arxiv.org/abs/2502.14156)
- **What's New**: Mixed Signals 데이터셋은 차량 간 협업 인지(V2X) 연구의 중요한 발전으로, 45.1k 포인트 클라우드(point clouds)와 240.6k 바운딩 박스(bounding boxes)를 포함하는 고품질 대규모 데이터셋입니다. 이 데이터셋은 서로 다른 LiDAR 센서를 장착한 3개의 연결된 자율주행 차량(CAV)과 이중 LiDAR를 갖춘 도로측 장치에서 수집되었습니다. 특히 Mixed Signals는 이질적인 CAV LiDAR 구성 및 왼쪽 주행 국가인 호주를 특징으로 하는 첫 번째 V2X 데이터셋입니다.

- **Technical Details**: Mixed Signals 데이터셋은 정확한 정렬(point cloud alignment)을 보장하기 위해 센서 동기화(synchronization) 및 대리자 위치 추정(localization) 기술을 사용합니다. 데이터 수집은 시드니의 한 교차로에서 이루어졌으며, 교통 신호가 있는 시간에 2시간 동안 진행되었습니다. 데이터를 통해 10개의 클래스 별로 다양한 교통 참여자를 캡처하여, 총 4개의 취약 도로 사용자(Vulnerable Road Users) 범주를 포함하고 있습니다.

- **Performance Highlights**: Mixed Signals 데이터셋은 VRU(Vulnerable Road User) 클래스가 50.3%를 차지하여 기존 데이터셋들에 비해 VRU 인지 성능을 개선 시키는 데 중점을 두고 있습니다. 기존 V2X 방법들에 대한 종합적인 벤치마크 결과를 포함하고 있으며, 품질 및 다양성 측면에서 공개된 데이터셋 중 가장 우수한 성능을 자랑합니다. 사용자는 해당 데이터와 주석을 다운로드할 수 있으며, 관련 비디오 시각화를 통해 데이터셋의 품질을 직접 확인할 수 있습니다.



### PitVQA++: Vector Matrix-Low-Rank Adaptation for Open-Ended Visual Question Answering in Pituitary Surgery (https://arxiv.org/abs/2502.14149)
Comments:
          9 pages

- **What's New**: 이 논문은 Vision-Language Models (VLMs)을 수술 방식에 통합하여 실시간 수술 결정 지원을 제공하는 새로운 개념을 소개합니다. 특히, PitVQA++라는 새로운 데이터셋과 Vector-MoLoRA라는 혁신적인 VLM 파인튜닝 방법론을 제시하여, 기존 VQA 시스템의 한계를 극복하려 합니다. 이 연구는 특히 내비게이션과 의사결정이 중요한 내비게이팅 피수술에 중점을 두고 있으며, 여기서 제안된 데이터셋은 약 101,803프레임과 745,972개의 질문-응답 쌍으로 구성되어 있습니다.

- **Technical Details**: Vector-MoLoRA는 깊은 신경망의 계층 구조를 고려하여 초기 계층에 더 많은 파라미터를 할당하고 이후 계층에서는 점차 감소시키는 행렬 저순위 조정 전략을 사용합니다. 이 접근법은 LoRA와 MoRA의 원리를 결합하여, 파인튜닝 과정에서의 재해 기억상실 (catastrophic forgetting)을 완화시키고 성능을 향상시키는 데 목적을 두고 있습니다. 또한, 이 연구에서는 Open-Ended PitVQA와 EndoVis18-VQA 데이터세트에서 이 방법의 유효성을 검증하였습니다.

- **Performance Highlights**: Vector-MoLoRA는 기존의 최신 방법들과 비교하여 성능을 크게 향상시키며, 불확실한 예측을 처리하는 데 있어 신뢰성과 안정성을 높이는 것으로 확인되었습니다. 또한, 위험-커버리지 분석을 통해 의료 전문가의 불확실한 샘플에 대한 의뢰 수가 줄어드는 결과를 보였습니다. 이러한 접근은 피수술 교육과 실시간 지원 모두에서 중요한 발전을 이룰 것으로 기대됩니다.



### Token Adaptation via Side Graph Convolution for Temporally and Spatially Efficient Fine-tuning of 3D Point Cloud Transformers (https://arxiv.org/abs/2502.14142)
Comments:
          Currently under review

- **What's New**: 본 연구는 3D 포인트 클라우드 분석을 위한 새로운 파라미터 효율적 미세 조정 기법인 Side Token Adaptation on a neighborhood Graph (STAG)을 소개합니다. 기존의 PEFT (parameter-efficient fine-tuning) 방법들이 조정 가능한 파라미터 수를 최소화하려고 시도했지만 여전히 높은 계산 비용의 문제를 겪고 있었습니다. STAG은 frozen backbone Transformer와 병행하여 작동하는 그래프 컨볼루셔널 사이드 네트워크를 사용하여 시간적 및 공간적 효율성을 극대화합니다.

- **Technical Details**: STAG은 세 가지 핵심 구성요소를 통해 높은 효율성을 실현합니다. 첫 번째로, 백본과의 연결을 통해 기울기 계산을 줄입니다. 두 번째로, 파라미터 공유 프레임워크를 활용하여 리소스를 절약하며, 세 번째로, 효율적인 그래프 컨볼루션을 통해 계산 속도를 높입니다. 또한, PCC13이라는 새로운 벤치마크를 제시하여 다양한 공개 3D 포인트 클라우드 데이터셋을 포함하고 있어 PEFT 방법의 포괄적인 평가를 가능하게 합니다.

- **Performance Highlights**: 다양한 사전 훈련 모델과 PCC13을 사용한 광범위한 실험 결과, STAG은 기존 방법에 비해 분류 정확도를 유지하면서 조정 가능한 파라미터 수를 단 0.43M로 줄이며, 미세 조정 시 계산 시간과 메모리 소비를 현저히 감소시킵니다. 이러한 성과는 STAG이 3D 포인트 클라우드 분석에서 매우 효율적이고 효과적인 방법임을 보여줍니다.



### ModSkill: Physical Character Skill Modularization (https://arxiv.org/abs/2502.14140)
- **What's New**: 이번 연구에서는 다양한 인간의 모션을 효과적으로 처리하기 위한 모듈화된 기술 학습 프레임워크인 ModSkill을 소개합니다. ModSkill은 복잡한 전체 신체 기술을 독립적인 바디 파트를 위한 구성 요소 기술로 분리하여 제어합니다. 이는 기술 모듈화 주의(attention) 레이어와 능동적인 기술 학습(active skill learning)을 포함하여 생성적 적응 샘플링(generative adaptive sampling)을 활용합니다.

- **Technical Details**: ModSkill은 정책 네트워크의 두 가지 주요 구성 요소로 구성되어 있습니다: 1) 기술 모듈화 주의 레이어, 2) 저수준 기술 조종사(low-level controller) 세트입니다. 각 바디 파트에 대한 구체적인 기술을 캡처하는 구체적인 구형 임베딩(spherical embeddings)을 생성하며, 이를 통해 정책의 성능을 개선합니다. 또한, 생성 모델을 사용하여 도전적인 모션 시퀀스에 대한 새로운 샘플을 생산합니다.

- **Performance Highlights**: 실험 결과, ModSkill 프레임워크는 기존 방법보다 정밀한 전체 신체 모션 추적에서 우수한 성능을 보이며, 다양한 목표 지향 작업을 위한 재사용 가능한 기술 임베딩을 가능하게 합니다. 고급 정책을 통해 다양한 다운스트림 작업으로 효과적으로 전이되고, 목표 달성 향상을 위한 새로운 가능성을 제시합니다.



### GlossGau: Efficient Inverse Rendering for Glossy Surface with Anisotropic Spherical Gaussian (https://arxiv.org/abs/2502.14129)
- **What's New**: 이번 논문에서는 GlossGau라는 효율적인 역 렌더링 프레임워크를 소개합니다. 이 프레임워크는 고광택 표면이 있는 장면을 재구성하면서도 기존 3D Gaussian Splatting(3D-GS)의 훈련 및 렌더링 속도를 유지합니다. GlossGau는 표면 법선, BRDF(양방향 반사 분포 함수) 매개변수 및 조명을 명확하게 모델링하여 더욱 자연스러운 재구성을 가능하게 합니다.

- **Technical Details**: GlossGau 방법은 다중 뷰 이미지에서 조명, 재료 속성, 기하학 및 표면 법선을 공동 최적화하며, 비대칭 구형 가우시안을 사용하여 BRDF 항을 형성합니다. 본 논문에서는 surfel 기반의 Gaussian Splatting을 활용하여 Gaussian의 기하학을 정의하고, 얇은 표면으로 형상을 변환하여 정상 추정 문제를 완화합니다. 이로 인해 고광택 표면 재구성을 위한 빠른 훈련 및 렌더링 속도가 확보됩니다.

- **Performance Highlights**: 실험 결과, GlossGau는 일반 및 고광택 표면 데이터셋에서 경쟁력 있는 또는 우수한 재구성을 달성하였습니다. 기존 GS 기반의 방법과 비교하여 최적화 시간에서 최대 66%의 단축을 이루었으며, 실시간 렌더링 성능을 지원합니다. 이는 고해상도 렌더링 출력과 함께 더욱 효율적인 역 렌더링 시간을 제공합니다.



### Modular Prompt Learning Improves Vision-Language Models (https://arxiv.org/abs/2502.14125)
Comments:
          2025 IEEE International Conference on Acoustics, Speech, and Signal Processing

- **What's New**: 이 논문에서는 사전 학습된 비전-언어 모델(Pre-trained Vision-Language Models, VLMs)의 성능을 향상시키기 위해 Modular Prompt Learning (MPL)이라는 새로운 방법을 제안합니다. 기존의 딥 비주얼 프롬프트 튜닝(Deep Visual Prompt Tuning, VPT) 기법의 한계를 극복하기 위해, 정보 보존을 촉진하도록 설계된 접근 방식을 사용합니다. 이를 통해 연속적으로 삽입되는 프롬프트의 정보를 유지하면서도 모델의 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: MPL은 세 가지 주요 구성 요소인 추가 프롬프트(𝒪add), 전달 프롬프트(𝒪cr) 및 제거 프롬프트(𝒪rm)로 구성됩니다. 이 조합을 통해 깊은 프롬프트 튜닝을 수행하며, 삽입된 프롬프트 수와 제거된 프롬프트 수가 동일하다는 조건 하에 작동합니다. 이 기법은 각 트랜스포머 층에 삽입된 프롬프트들이 중요한 정보를 보존하도록 설계되었습니다.

- **Performance Highlights**: 이 방법을 11개 데이터셋에서 평가한 결과, 기법이 기존 최고 성능 방법에 비해 평균 0.7%의 성능 향상을 달성했습니다. 또한, EuroSAT 데이터셋에서는 10.7%라는 가장 큰 성능 향상을 기록하였으며, 크로스 데이터셋 평가에서는 기존 방법과 비교해 훨씬 높은 효율성을 보여 주목받고 있습니다.



### Object-centric Binding in Contrastive Language-Image Pretraining (https://arxiv.org/abs/2502.14113)
- **What's New**: 이번 논문은 최근 비전 언어 모델(vision language models, VLM)의 한계를 극복하기 위해 혁신적인 접근을 제안합니다. 기존의 대조 모델(contrastive models)인 CLIP을 기반으로 하여 복잡한 구성 장면(compositional scenes)을 이해하는 데 필요한 새로운 캐주얼(casual) 및 구조적 요소를 통합하고자 합니다. 우리의 접근방식은 하드 네거티브(hard-negative) 증강을 사용할 필요 없이 구성 이해(compositional understanding)를 개선할 수 있는 방법입니다.

- **Technical Details**: 우리는 장면 그래프(scene graph)와 슬롯 구조(slots-structured) 이미지 표현을 연결하는 바인딩 모듈(binding module)을 도입하여 두 개의 모달리티 사이의 유사성을 구조적으로 평가할 수 있도록 합니다. 또한, 관계를 텍스트 조건부 비주얼 제약(text-conditioned visual constraints)으로 활용하여 객체 간의 복잡한 상호작용을 보다 효과적으로 포착합니다. 이러한 방법론은 CLIP 기반 모델의 다중 객체 구성을 개선하는 데 기여합니다.

- **Performance Highlights**: 우리의 모델은 복잡한 장면의 이미지-텍스트 일치를 보다 정확하고 샘플 효율적으로 수행할 수 있도록 지원합니다. 이로 인해 향후 비전 언어 모델의 발전에 기여할 수 있는 가능성을 제시하게 됩니다. 결과적으로, 본 연구는 기존의 VLM의 성능을 한층 높여주는 기반이 될 것입니다.



### Point Cloud Geometry Scalable Coding Using a Resolution and Quality-conditioned Latents Probability Estimator (https://arxiv.org/abs/2502.14099)
Comments:
          Submitted to IEEE and currently under review

- **What's New**: 이번 논문은 Point Cloud (PC) 압축에 대한 새로운 스케일러블 코딩 솔루션인 Scalable Resolution and Quality Hyperprior (SRQH)를 제안합니다. 기존의 JPEG Pleno 학습 기반 PC 코딩 표준에 SRQH를 통합함으로써, 단일 비트스트림으로 다양한 품질과 해상도에서 PC를 디코딩할 수 있는 능력을 제공합니다. 이는 비트스트림을 다수 유지해야 하는 문제를 해결하며, 저장 및 계산 요구 사항의 영향을 최소화합니다.

- **Technical Details**: SRQH는 해상도 및 품질 스케일러빌리티를 제공하는 혁신적인 접근법으로, 기본 코덱의 비스케일러블 운영 모드와의 호환성을 유지합니다. 특히 RQuLPE(Resolution and Quality-conditioned Latents Probability Estimator) 모델을 사용하여 향상 레이어 코딩에서 하이퍼 분석 및 하이퍼 합성 변환을 대체하였습니다. SRQH는 메모리 요구 사항을 줄이고, 인코딩 시간은 10% 미만, 디코딩 시간은 20% 미만의 증가만을 초래합니다.

- **Performance Highlights**: SRQH는 JPEG PCC에서 비스케일러블 운영 모드에 비해 최소의 rate-distortion (RD) 성능 저하를 가져옵니다. 또한, 다양한 디바이스에 맞춤형 비트스트림 서비스를 제공하여 PC 지오메트리 코딩의 성능을 상당히 향상시킵니다. 이 방법은 기존 솔루션에 비해 메모리 요구 사항을 줄이면서도 여러 가지 코딩 구성에 걸쳐 높은 품질과 해상도를 유지하는 능력을 가지고 있습니다.



### Regression in EO: Are VLMs Up to the Challenge? (https://arxiv.org/abs/2502.14088)
- **What's New**: 이 논문은 비전 언어 모델( Vision Language Models, VLMs )을 지구 관측( Earth Observation, EO ) 회귀 작업에 적용하기 위한 도전과제와 기회를 체계적으로 분석합니다. EO 데이터는 다중 센서와 다중 시간대를 가지고 있어 지구의 다이나믹스를 이해하는 데 필수적입니다. 그러나 과학적 회귀 관련 EO 응용 분야는 아직 깊이 탐구되지 않았습니다.

- **Technical Details**: 논문에서는 EO 데이터의 독특한 특징을 기존 컴퓨터 비전 데이터셋과 비교하고, VLM을 EO 회귀 작업에 적용하는 데 있어 네 가지 주요 장애물 (1) 전용 기준 부족, (2) 이산-연속 표현 불일치, (3) 누적 오류 축적, (4) 수치적 작업에 대한 텍스트 중심 훈련 목표의 비최적성을 식별합니다. 이러한 문제는 보다 정교하고 해석 가능한 환경 과정 모델링을 위한 기초를 마련합니다.

- **Performance Highlights**: VLM을 사용해 EO 회귀 작업에 대한 연구가 진전되고 있으며, 여러 과학적 변수들을 추정하기 위한 새로운 접근법들이 제안되고 있습니다. 이 연구는 회귀 작업에서 VLM의 가능성을 탐구하고, 복잡한 EO 데이터를 다루기 위한 철저한 방법론적 통찰력을 제공합니다. 따라서, EO 회귀의 정확성과 신뢰성을 높이기 위한 향후 연구 방향이 제시되었습니다.



### DiffExp: Efficient Exploration in Reward Fine-tuning for Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.14070)
Comments:
          AAAI 2025

- **What's New**: 이번 연구에서는 텍스트-이미지 확산 모델의 리워드 파인튜닝(reward fine-tuning)을 위한 탐색 전략인 DiffExp를 소개합니다. 기존의 리워드 최적화 방법들이 온라인 샘플 생성 과정에서 느린 수렴 속도로 어려움을 겪는 반면, 본 접근법은 샘플의 다양성을 높이고 강력한 리워드 신호를 활용하도록 설계되었습니다. 이를 통해 모델 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: DiffExp는 두 가지 주요 전략으로 구성됩니다: (a) classifier-free guidance (CFG)의 스케일을 동적으로 조절하여 샘플의 다양성을 향상시키고, (b) 텍스트 프롬프트의 특정 구문에 랜덤 가중치를 부여하여 고품질 리워드 신호를 활용합니다. 이러한 전략은 온라인 샘플 생성 과정에서 시각적 다양성을 촉진하여 리워드 기반의 최적화를 개선하는 데 기여합니다.

- **Performance Highlights**: DiffExp를 정책 기울기 방법(policy gradient method)과 직접 리워드 역전파 방법(direct reward backpropagation method)와 통합하여 성과를 검증했습니다. 실험 결과, DiffExp는 최근의 리워드 기반 파인튜닝 방법들의 샘플 효율성을 향상시키며, SDXL 모델을 포함하여 고해상도 텍스트-이미지 모델의 품질도 상당히 개선시켰습니다.



### A Racing Dataset and Baseline Model for Track Detection in Autonomous Racing (https://arxiv.org/abs/2502.14068)
Comments:
          Currently Under Review

- **What's New**: 이 논문에서는 RoRaTrack라는 새로운 데이터셋을 소개합니다. 이는 레이싱 시나리오에서의 트랙 감지를 위한 주석이 달린 다중 카메라 이미지 데이터를 포함하고 있습니다. 데이터는 인디애나의 레이싱 서킷에서 Dallara AV-21 차량을 사용하여 수집되었고, Indy Autonomous Challenge(IAC)와 협력하여 진행되었습니다.

- **Technical Details**: RoRaTrack 데이터셋은 고속으로 인한 흐림, 카메라에서의 색상 반전, 도로 차선 표시의 부재와 같은 문제들을 해결하고 있습니다. 또한, RaceGAN이라는 GAN(Generative Adversarial Network) 기반 모델을 제안하여 이러한 과제를 효과적으로 처리하고 있습니다. 실험 결과, RaceGAN은 현재의 최신 머신 러닝 모델보다 우수한 성능을 보여주며 새로운 기준을 세우고 있습니다.

- **Performance Highlights**: RoRaTrack 데이터셋은 반복적으로 발생하는 레이싱 과제들을 포착하고 instance-level 주석을 통해 도로와 배경을 명확하게 구별하는 세그멘테이션 마스크를 포함하고 있습니다. 이 데이터셋은 실제 고속 레이싱 시나리오에서 수집된 데이터로, 다중 카메라 뷰를 제공하여 기존 시스템의 근원적 한계를 보완하고 있습니다.



### Triad: Vision Foundation Model for 3D Magnetic Resonance Imaging (https://arxiv.org/abs/2502.14064)
- **What's New**: 해당 연구에서는 3D MRI를 위해 설계된 비전 파운데이션 모델인 Triad를 소개합니다. Triad는 13만 1,170개의 3D MRI 볼륨을 학습하여 강력한 표현을 학습하며, 이를 통해 시맨틱(semantic) 분포를 제한하는 오르간 독립적(imaging descriptions) 이미징 설명을 사용합니다. 이 연구는 Triad가 임상 응용에서의 통합성 및 다용성을 보장하면서 새로운 3D MRI 작업에 대해 지속적으로 성능을 극대화할 수 있음을 보여줍니다.

- **Technical Details**: Triad는 19,721명의 환자로부터 수집된 3D MRI 데이터셋인 TriadMR-131K에서 훈련되며, 유방, 뇌, 전립선의 다양한 이미징 모달리티(T1-w, T2-w, FLAIR 등)를 포함합니다. 데이터 사전 훈련 동안, 널리 사용되는 오토인코더(autoencoder) 아키텍처를 채택하여 데이터의 심층적인 표현을 학습합니다. 또한, 다양한 크기의 인코더를 사전 훈련하여 다운스트림 작업의 요구에 맞추어 다양한 기능을 수행할 수 있도록 합니다.

- **Performance Highlights**: Triad 모델의 성능은 25개의 다운스트림 데이터셋을 평가하면서 뛰어난 결과를 보였습니다. nnUNet-Triad는 17개의 데이터셋에서 nnUNet-Scratch보다 6.88% 향상된 분할 성능을 기록하였고, Swin-B-Triad는 5개의 데이터셋 분류 작업에서 Swin-B-Scratch보다 3.97% 향상되었습니다. 이러한 성과들은 Triad가 다양한 임상 응용 프로그램에서 사용될 수 있는 유연성과 효율성을 제공함을 나타냅니다.



### PedDet: Adaptive Spectral Optimization for Multimodal Pedestrian Detection (https://arxiv.org/abs/2502.14063)
- **What's New**: 이번 연구에서는 PedDet라는 새로운 pedestrian detection 프레임워크를 제안합니다. PedDet는 다중 스펙트럼 데이터를 활용하여 RGB와 적외선 이미지를 효과적으로 융합하고, 조명 조건에 따라 적응적으로 특성을 추출하여 안정성과 정확성을 향상시킵니다. 이 방법은 복잡한 환경에서 보행자 탐지의 성능을 크게 개선하였습니다.

- **Technical Details**: PedDet는 두 가지 핵심 모듈을 포함하고 있습니다: Multi-scale Spectral Feature Perception Module (MSFPM)와 Illumination Robustness Feature Decoupling Module (IRFDM). MSFPM은 RGB와 적외선 스펙트럼의 정보를 병렬로 처리하며, 각 모드의 특성 가중치를 조정하여 다양한 조명 조건에서도 제 성능을 유지합니다. IRFDM은 보행자와 배경 특성을 분리하여 다양한 조명 환경에서의 탐지 성능을 극대화합니다.

- **Performance Highlights**: PedDet는 LLVIP 및 MSDS 데이터셋에서 실험을 통해 최첨단 성능을 입증하였으며, mAP(mean Average Precision)를 6.6% 향상시켰습니다. 낮은 조명 조건에서도 뛰어난 탐지 정확성을 보여주면서 도로 안전성을 크게 향상시키는 결과를 가져왔습니다. 전체적으로, PedDet는 복잡한 환경에서도 효과적으로 작동하는 새로운 보행자 탐지 모델로 자리잡았습니다.



### EfficientPose 6D: Scalable and Efficient 6D Object Pose Estimation (https://arxiv.org/abs/2502.14061)
- **What's New**: 본 연구는 GDRNPP 기반의 빠르고 확장 가능한 포즈 추정기(pose estimator) 세트를 개발하여 정확성과 견고성을 강화하며, 실시간 환경에서의 효율성-정확성 균형을 개선하고자 합니다. AMIS 알고리즘을 제안하여 응용 프로그램에 특화된 추론 시간과 정확성의 무역 오프를 반영하는 모델 선택을 지원합니다. 이 방법은 LM-O, YCB-V, T-LESS, ITODD와 같은 4개의 주요 벤치마크 데이터셋에서 효과를 입증합니다.

- **Technical Details**: GDRNPP 클라우드 구조는 RGB 이미지를 기반으로 객체의 6D 포즈를 추정하며, CNN을 활용해 중요한 이미지 영역을 탐지하고 특징을 예측합니다. 추론 시간을 최적화하기 위해 GDRNPP 프로세스를 데이터 로드, 백본, 지오 헤드, 데이터 프로세스, 패치 PnP 등 여섯 단계로 나누어 예비 실험을 통해 주요 지연 원인을 식별했습니다. 이를 통해 구조적 변경을 통해 정확성을 유지하면서 추론 시간을 줄일 수 있도록 설계되었습니다.

- **Performance Highlights**: AMIS 알고리즘을 통해 선택된 후보 모델은 여러 데이터셋에서 추론 시간과 6D 포즈 추정 품질 간의 최적 무역 오프를 이루며, 각 알고리즘에 대해 정량적 결과를 제시합니다. 연구에서는 GDRNPP의 수정된 40개의 후보 아키텍처를 제안하며, 이들 후보는 다양한 시간 제약 하에서도 향상된 정확성을 유지합니다. 이를 통해 포즈 추정의 실용적인 적용 가능성을 넓히고자 합니다.



### Enhancing Cognition and Explainability of Multimodal Foundation Models with Self-Synthesized Data (https://arxiv.org/abs/2502.14044)
Comments:
          Accepted by ICLR 2025. Code: this https URL

- **What's New**: 이번 연구는 Large Multimodal Models (LMMs)의 시각적 분류 능력을 향상시키기 위한 새로운 접근 방식을 제안합니다. 이 방법은 자기 합성된 데이터를 사용하여 모델의 인식 능력과 설명 가능성을 개선하는 혁신적인 비주얼 리젝션 샘플링 프레임워크를 포함합니다. 이는 기존 LMMs가 도메인 특화된 목표를 식별하고 예측에 대한 정당한 설명을 제공하는 데 어려움을 겪는 문제를 해결하려고 합니다.

- **Technical Details**: 제안된 접근법은 이미지, 질문(queries) 및 목표 답변(target answers)을 포함한 비주얼 파인튜닝(visual fine-tuning)을 요구합니다. 학습 과정에서는 전문가가 정의한 개념에 기반한 인간 검증 가능 시각적 특징을 포함하는 해석 가능한 답변을 합성하여 모델을 학습합니다. 또한 보상 모델이 없는 필터링 메커니즘을 적용하여 각 튜닝 라운드에서 최고 품질의 해석 가능한 답변을 선별합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 전문화된 시각 분류 작업에서 정확성과 설명 가능성을 동시에 향상시키는 데 효과적임을 보여줍니다. LLaVA-1.5와 같은 최신 모델이 Stanford Dogs 데이터셋에서 12.2%의 마킹 정확도를 기록할 때, 이 연구는 성능을 크게 향상시킬 수 있는 가능성을 열어줍니다. 이 과정은 모델이 신뢰할 수 있는 설명을 제공할 수 있도록 지속적으로 능력을 개선합니다.



### Benchmarking Multimodal RAG through a Chart-based Document Question-Answering Generation Framework (https://arxiv.org/abs/2502.14864)
- **What's New**: 이번 연구에서는 Chart-based MRAG라는 새로운 작업을 도입하여 기존의 MRAG 모델이 다루지 못한 복잡한 시각적 형식, 즉 차트를 포함한 분류 문제를 해결하고자 합니다. 기존의 벤치마크가 주로 단순한 이미지-텍스트 상호작용에 집중하고 있는 점을 비판하며, 고급 QA 샘플을 세미 자동으로 생성하기 위한 CHARt-based document question-answering GEneration (CHARGE) 프레임워크를 제안합니다.

- **Technical Details**: CHARGE는 텍스트 및 차트 데이터를 구조화된 키포인트 추출, 교차 모달 검증, 키포인트 기반 생성 과정을 통해 고품질 QA 쌍을 생성합니다. 이 프레임워크는 각 모달리티의 기여를 별도로 평가할 수 있도록 Text-Chart MRAG, Text-only RAG, Chart-only MRAG과 같은 하위 작업을 포함합니다. Chart-MRAG Bench는 8개 분야에서 4,738개의 질문-응답 쌍을 포함하며, 이는 실제 데이터에서 도출된 것이어서 복잡한 교차 모달 인터랙션을 반영합니다.

- **Performance Highlights**: 평가 결과, 통합된 멀티모달 임베딩 검색 방법이 차트 중심의 시나리오에서 성능이 떨어진다는 것을 확인했습니다. 또한 최신 MLLM조차도 정답 기반 검색을 사용한 경우에도 Correctness는 58.19%, Coverage는 73.87%에 불과하여, 텍스트-차트 다모드 추론에서 지속적인 문제점을 드러냈습니다. 연구진들은 이러한 결과를 통해 차트 중심의 MRAG 모델에 대한 향후 연구 방향성을 제시합니다.



### Dynamic Concepts Personalization from Single Videos (https://arxiv.org/abs/2502.14844)
Comments:
          Webpage: this https URL

- **What's New**: 이번 논문에서는 동적 개념을 개인화하기 위한 새로운 프레임워크인 Set-and-Sequence를 소개합니다. 이 접근법은 기존 Diffusion Transformers (DiTs) 기반의 생성 비디오 모델에서 공간-시간(weight space)을 효과적으로 구현하여 개인화된 비디오 생성을 가능하게 합니다. 두 가지 주요 단계로 구성되어 있으며, 동적 개념의 특성을 잘 포착할 수 있도록 설계되었습니다.

- **Technical Details**: Set-and-Sequence 프레임워크는 두 단계로 운영됩니다. 첫 번째 단계에서 Low-Rank Adaptation (LoRA) 레이어를 비디오의 무작위 프레임 세트를 사용해 미세 조정하여 동적 개념의 외형을 배울 수 있게 합니다. 두 번째 단계에서는 동적 개념의 모션을 캡처하기 위해 모션 잔여물(Motion Residuals)을 사용하여 계수를 조정하고, 전체 비디오 시퀀스에 대해 세부 조정을 실시합니다.

- **Performance Highlights**: 이 연구는 비디오 생성에서 비약적인 향상을 보여주었습니다. 예를 들어, 바다의 유동적 움직임과 캠프파이어의 불꽃을 결합하는 작업을 이전보다 훨씬 높은 충실도로 수행할 수 있게 되었습니다. 이러한 개선은 장면 구성(composition)과 적응성을 용이하게 하며, 텍스트 프롬프트에 의한 카메라 움직임의 직관적인 편집이 가능해집니다.



### FetalCLIP: A Visual-Language Foundation Model for Fetal Ultrasound Image Analysis (https://arxiv.org/abs/2502.14807)
- **What's New**: FetalCLIP는 태아 초음파 이미지를 분석하기 위해 설계된 비전-언어 (vision-language) 기초 모델로, 대규모의 태아 초음파 이미지와 캡션의 커플 데이터셋을 활용하여 개발되었습니다. 이 모델은 태아 초음파 영상의 복잡한 해부학적 특징을 효과적으로 학습하여 다양한 다운스트림 (downstream) 작업에 활용할 수 있는 고급 표현을 생성하는 능력을 가지고 있습니다. 기존 모델의 한계를 극복하기 위해 FetalCLIP는 대규모 멀티모달 학습을 통해 훈련되었으며, 이 과정에서 210,035개의 이미지와 텍스트 쌍을 사용했습니다.

- **Technical Details**: FetalCLIP는 기존의 CLIP 프레임워크를 바탕으로 단일 이미지와 그에 대한 텍스트 설명을 정렬하여 훈련되었습니다. 이 모델은 비전 트랜스포머(ViT) 기반 이미지 인코더와 텍스트 인코더를 통합하여 구성을 이루며, 최대 117자의 토큰을 처리할 수 있습니다. 훈련 데이터셋에는 규칙적인 임상 초음파 이미지가 포함되어 있으며, 각 이미지는 GPT-4o에 의해 생성된 적절한 텍스트 설명과 함께 제공됩니다.

- **Performance Highlights**: 다양한 태아 초음파 분석 작업에서 FetalCLIP는 기존 모델들에 비해 우수한 성능을 보였습니다. FetalCLIP는 제로샷 분류 작업에서 87.1%의 F1 점수를 달성하며, 이는 기존의 SonoNet 모델에 비해 17.2% 더 높은 성과입니다. 또한, 선천성 심장 결함(CHD) 탐지 작업에서도 기존 모델 대비 6.92% 높은 AUC 성능을 나타내며, 태아 해부학적 구조 세분화에서는 평균 Dice 유사성 계수(DSC) 84.22%의 성과를 기록, FetalCLIP는 AI 기반 태아 초음파 진단의 중요한 발전을 대표합니다.



### Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration (https://arxiv.org/abs/2502.14795)
- **What's New**: 이 논문은 현재의 휴머노이드 로봇 제어 프레임워크의 한계를 다룹니다. 기존 프레임워크는 반응적인 메커니즘에 의존하며 자율 상호작용 기능이 부족한 점을 지적합니다. 이번 연구에서는 언어 이해, 자아 중심의 장면 인식 및 모션 제어를 통합한 새로운 프레임워크, Humanoid-VLA를 제안합니다.

- **Technical Details**: Humanoid-VLA는 비자아 중심의 인간 동작 데이터셋과 텍스트 설명을 결합하여 언어-모션 사전 정렬을 통해 보편적인 모션 패턴과 행동 의미를 학습합니다. 그 후, 파라미터 효율적인 비디오 조건 조정을 통해 자아 중심의 시각적 맥락을 통합하여 맥락 인식이 가능한 모션 생성을 가능하게 합니다. 또한, 자가 감독 데이터 증강 전략을 도입하여 모션 데이터에서 직적용된 유사 주석을 자동으로 생성합니다.

- **Performance Highlights**: Humanoid-VLA는 물체 상호작용 및 환경 탐색 작업에서 향상된 맥락 인식을 달성합니다. 이러한 결과는 로봇이 더 인간과 유사한 적응적이고 지능적인 참여 능력을 갖추도록 돕습니다. 전체 신체 제어 아키텍처를 기반으로 실시한 광범위한 실험을 통해 이러한 성능 향상이 입증되었습니다.



### ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting (https://arxiv.org/abs/2502.14780)
Comments:
          12 pages, 7 figures, 3 tables

- **What's New**: 이 논문에서는 AR(증강현실), VR(가상현실) 및 최신 스마트폰 상에서의 효율적이고 프라이버시를 보호하는 다중모드(interaction)에 대한 새로운 접근법인 Visual Instruction Rewriting을 제안합니다. 기존의 대규모 비전-언어 모델(VLM)은 클라우드 기반 처리를 의존하여 시각 데이터의 프라이버시 문제와 실시간 처리의 어려움을 초래했습니다. 본 연구는 39,000개의 예제와 14개 도메인으로 구성된 데이터셋을 제시하며, 이를 통해 프라이버시를 보장하면서 다중모드를 처리할 수 있는 경량화된 VLM(250M parameters)을 개발했습니다.

- **Technical Details**: Visual Instruction Rewriting은 다중모드 지시사항을 텍스트만 포함된 명령으로 변환합니다. 이 접근법은 프라이버시를 보호하면서도 사용자가 기기에서 직접 명령을 실행할 수 있도록 합니다. 본 연구는 여러 NLG(자연어 생성) 메트릭(BLEU, METEOR, ROUGE)을 이용하여 제안한 모델의 성능을 평가하였으며, 8비트 양자화된 모델은 500MB 미만의 저장 공간에서 효과적인 지시 사항 변환을 수행했습니다.

- **Performance Highlights**: 우리는 경량화된 250M 파라미터 모델이 기존의 베이스라인인 PaliGemma-v2 및 Qwen2VL에 비해 제로샷 설정에서 더 나은 성능을 보였음을 발견했습니다. 이 모델은 사용자 요청을 구조화된 텍스트로 변환할 수 있는 수용 가능한 수준의 rewriting 능력을 가지며, 이는 AR, VR 및 스마트폰 인터페이스와의 안전하고 실시간 상호작용을 가능하게 합니다.



### Harnessing PDF Data for Improving Japanese Large Multimodal Models (https://arxiv.org/abs/2502.14778)
Comments:
          15 pages, 8 figures

- **What's New**: 본 연구에서는 일본어 LMM(Large Multimodal Models)의 성능을 향상시키기 위해 PDF 데이터의 잠재력을 탐색합니다. 기존의 일본어 LMM은 영어로 번역된 데이터에 의존해 일본 특유의 문화적 지식을 캡처하는 데 한계가 있었습니다. 이를 해결하기 위해, PDF에서 이미지-텍스트 쌍을 추출하는 완전 자동화된 파이프라인을 소개하며, 외부 주석 작업 없이 데이터를 확보할 수 있는 방법을 모색했습니다.

- **Technical Details**: 우리는 사전 훈련된 모델을 활용하여 PDF에서 이미지-텍스트 쌍을 추출하는 방법론을 개발했습니다. 이 과정에서는 레이아웃 분석(layout analysis), OCR(Optical Character Recognition), 및 비전-언어 페어링(vision-language pairing) 기술이 포함됩니다. 이러한 자동화된 시스템을 통해 일본어 LMM의 훈련 데이터를 풍부하게 하기 위한 지침 데이터도 생성됩니다.

- **Performance Highlights**: PDF에서 유래된 데이터로 훈련한 일본어 LMM은 Heron-Bench에서 3.9%에서 13.8%의 성능 향상을 보였습니다. 다양한 실험을 통해 PDF 데이터의 효과와 모델 크기와의 관계를 분석하였으며, 이미지-텍스트 쌍과 이미지만으로 생성된 지침 데이터의 효과성도 평가했습니다. 이러한 연구 결과는 일본어 LMM의 훈련에 PDF 데이터를 활용하는 가치를 확고히 해줍니다.



### Sculpting [CLS] Features for Pre-Trained Model-Based Class-Incremental Learning (https://arxiv.org/abs/2502.14762)
- **What's New**: 이 연구에서는 클래스 점진적 학습(Class-incremental Learning)에서의 기존 문제를 해결하기 위해 새로운 파라미터 효율적인 파인튜닝 모듈인 'Learn and Calibrate'(LuCA)를 도입합니다. 기존의 모델들이 새로운 개념을 학습하면서 잊어버리는 치명적인 문제(catastrophic forgetting)에 직면할 때, LuCA는 효과적인 적응을 통해 일반 지식을 보존하도록 설계되었습니다. 또한, 최종 토큰에서 동작하는 희소 LuCA 모듈인 'Token-level Sparse Calibration and Adaptation'(TOSCA)을 제안하여 학습 세션마다 효과를 극대화합니다.

- **Technical Details**: TOSCA 접근법은 두 개의 구성 요소로 이루어진 LuCA 모듈을 통해 자동화된 적응을 제공합니다. 이는 특정 작업을 위해 특성 변환을 적용하는 잔여 어댑터(residual adapter)와 주목(attention)과 유사한 게이팅을 통해 특징을 향상시키는 보정기(calibrator)로 구성됩니다. 이러한 모듈 배치 전략을 통해 우리는 모델의 일반화 능력을 유지하면서 최종 고수준 특정 특징에 적응합니다.

- **Performance Highlights**: TOSCA는 실험을 통해 보다 적은 파라미터로 고성능을 나타내는 것으로 입증되었습니다. 기존 방법과 비교하여 약 8배 적은 파라미터로 7-21% 더 높은 정확도를 달성하며, 실행 속도도 약 2.5배 빠릅니다. 이는 TOSCA가 기존의 방법들에 비해 더욱 효율적이고 효과적인 접근이라는 것을 의미합니다.



### MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders (https://arxiv.org/abs/2502.14753)
- **What's New**: 이번 연구에서는 MedVAE라는 새로운 2D 및 3D 오토인코더(autoencoder) 모델을 소개합니다. 이 모델은 방대한 의료 이미지를 압축된 잠재 표현(latent representations)으로 인코딩(enconding)하고, 쉽게 고해상도(high-resolution) 이미지로 디코딩(decoding)할 수 있습니다. 특히, 기존의 고해상도 이미지를 대체할 수 있는 효율적인 방법을 제시하여, 임상적으로 중요한 피처를 효과적으로 보존합니다.

- **Technical Details**: MedVAE는 105만 개 이상의 의료 이미지를 사용하여 훈련되었습니다. 이 모델은 두 단계의 훈련 접근법을 통해 잠재 표현의 질을 최적화하는 데 중점을 두고 있습니다. 또한, 2D 및 3D 이미지를 위한 일반화된 오토인코더를 개발하여, 다양한 임상적 특징을 유지하며 효율성을 개선하고 있습니다. 개별적으로, 2D 이미지의 경우 f라는 다운사이징(f) 팩터를 적용하여 크기를 줄이고, 3D 이미지의 경우도 유사한 과정이 적용됩니다.

- **Performance Highlights**: 실험 결과, MedVAE의 잠재 표현은 CAD(Computer-Aided Diagnosis) 파이프라인에서 고해상도 이미지를 대체할 수 있으며, 성능을 유지하거나 초과할 수 있음을 보여줍니다. 이를 통해 저장 요구량이 최대 512배 줄어들고, CAD 모델 훈련의 효율성이 최대 70배 개선됩니다. 또한, 전문가 평가에서도 MedVAE의 디코딩된 재구성이 임상적으로 중요한 피처를 효과적으로 보존하는 것으로 나타났습니다.



### CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting (https://arxiv.org/abs/2502.14684)
- **What's New**: 이 논문은 3D Gaussian Splatting (3DGS)의 한계를 극복하기 위해 CDGS라는 새로운 접근 방식을 소개합니다. CDGS는 confidence-aware depth regularization (신뢰 기반 깊이 정규화) 기법을 통해 3DGS의 기하학적 정확성을 향상시키며, 초기 학습 단계에서 기하학적 세부 사항을 더 잘 보존하는 성과를 보여줍니다. 또한, 이 방법은 NVS의 품질 및 기하학적 정확성에서도 경쟁력 있는 성능을 나타내고 있습니다.

- **Technical Details**: CDGS는 monocular depth estimation (단안 깊이 추정)과 sparse Structure-from-Motion (구조로부터의 동작) 깊이 데이터를 활용하여 깊이 최적화를 동적으로 조정합니다. 이는 gradient descent 방법을 통해 정렬된 깊이 맵을 생성하고, 각 깊이 맵에 대해 신뢰 맵을 생성하여 최적화 동안 깊이 손실 가중치를 조정합니다. 이러한 방법은 3DGS의 기존 pixel-wise L1 손실 및 SSIM 손실을 제외한 추가 깊이 정규화 항을 도입하여 기하학적 정확성을 개선합니다.

- **Performance Highlights**: 제공된 실험 결과에 따르면 CDGS는 Tanks and Temples 벤치마크 데이터셋에서 NVS에 대해 최대 2.31 dB 향상의 PSNR을 달성하며, M3C2 거리 메트릭에서 일관되게 낮은 기하학적 오류를 기록했습니다. 특히, CDGS는 원래의 3DGS와 유사한 F-score에 도달하면서도, 50%의 훈련 반복만으로 동일한 결과를 얻어냈습니다. 이를 통해 CDGS가 디지털 트윈 생성 및 유산 보존과 같은 실제 응용 프로그램을 위한 효율적이고 정확한 3D 재구성 시스템 개발에 기여할 것으로 기대됩니다.



### NAVIG: Natural Language-guided Analysis with Vision Language Models for Image Geo-localization (https://arxiv.org/abs/2502.14638)
- **What's New**: 새로운 데이터셋 NaviClues를 통해 전문가들의 사고 과정을 수집하여 이미지의 지리적 위치를 예측하는 방법을 제시합니다. 이 데이터셋은 인기 있는 게임인 GeoGuessr에서 파생된 것으로, 2,000개 이상의 사례를 포함하여 풍부한 징후와 통찰력을 제공합니다. 또한, Navig라는 새로운 이미지 지리 로컬라이제이션 프레임워크를 개발하여 시각적 및 언어적 정보 분석을 통합합니다.

- **Technical Details**: NaviClues 데이터셋은 5명의 전문 YouTuber의 게임 영상을 분석하여 고품질의 결과를 생성합니다. 이 데이터셋은 11,200개의 이미지와 각 이미지의 지리적 정보, 추론을 포함하고 있으며, Road signs와 같은 다양한 시각적 요소를 고려하여 인간 전문가들이 소프트웨어로 분석할 수 있는 인사이트를 제공합니다. Navig는 Reasoner, Searcher, Guesser의 세 가지 컴포넌트로 구성되어 다양한 지리적 위치를 예측하는 데 필요한 세밀한 정보를 제공합니다.

- **Performance Highlights**: Navig는 기존의 최첨단 모델들보다 평균 거리 오차를 14% 줄이는데 성공하였으며, 훈련 샘플 수는 1,000개 미만으로 유지되었습니다. 이러한 결과는 이미지 로컬라이제이션의 정확성을 크게 향상시킬 수 있는 잠재력을 지닌 것이며, 데이터셋과 코드가 공개되어 있어 연구자들이 이를 활용할 수 있습니다.



### Vision Foundation Models in Medical Image Analysis: Advances and Challenges (https://arxiv.org/abs/2502.14584)
Comments:
          17 pages, 1 figure

- **What's New**: 최근 Vision Foundation Models (VFMs), 특히 Vision Transformers (ViT)와 Segment Anything Model (SAM)의 발전으로 의료 이미지 분석 분야에서 큰 진전이 있었습니다. 이 논문은 VFMs의 의료 이미지 분할에 대한 적응과 관련된 최신 연구를 검토하며, 도메인 적응(domain adaptation), 모델 압축(model compression), 연합 학습(federated learning)과 같은 다양한 도전 과제를 다룹니다. 또, Adapter 기반의 개선과 지식 증류(knowledge distillation) 기법, 다중 스케일 컨텍스트 특성 모델링의 최근 발전을 제안합니다.

- **Technical Details**: VFMs의 도메인 적응 시 가장 큰 문제는 의료 이미지와 자연 이미지 간의 도메인 차이입니다. 대규모 사전 훈련 데이터셋은 의료 분야에서 접근하기 어려워 작은 의료 데이터셋을 사용할 경우 성능 저하가 occurs. 연구자들은 Adapter 모듈과 교차 도메인 전이 학습(cross-domain transfer learning)을 활용하여 VFMs의 적응성을 높이려고 시도하고 있습니다. 또한, 지식 증류는 대형 모델을 효율적인 소형 모델로 전이하는 유망한 접근 방식으로 등장하고 있으며, 이는 임상 환경의 리소스 제약을 해결하는 데 필수적입니다.

- **Performance Highlights**: 이 연구는 VFMs가 의료 이미지 분석을 혁신할 수 있는 잠재력과 함께, 연합 학습 및 모델 압축과 같은 새로운 방법론이 의료 이미지 분석의 한계를 극복하고 임상 응용을 향상시키는 데 기여할 수 있음을 강조합니다. 특히, SAM의 제로샷(segmentation) 능력은 다중 모드 이미징 및 복잡한 해부학적 구조를 가진 의료 이미지를 처리하는 데 있어 획기적인 장점을 제공합니다. 연구 결과들은 VFMs의 구조적 설계와 최적화 전략을 통해 데이터 규모 및 계산 효율성 문제를 효과적으로 해결할 수 있는 가능성을 보여줍니다.



### A Mobile Robotic Approach to Autonomous Surface Scanning in Legal Medicin (https://arxiv.org/abs/2502.14514)
Comments:
          Submitted and accepted for presentation at CARS 2025. This preprint has not undergone peer review or post-submission revisions. The final version of this work will appear in the official CARS 2025 proceedings

- **What's New**: 이 연구에서는 법의학 분야에서 시체 문서화를 위한 모바일 로봇 시스템을 개발하였습니다. 이 시스템은 RGB-D 표면 스캐닝을 자동화하여 외부 상처를 보다 효율적으로 기록할 수 있도록 설계되었습니다. 기존의 수동 작업과 비교하여 시간 효율성과 감염 위험을 줄이는 데 기여할 수 있습니다.

- **Technical Details**: 제안한 시스템은 이동식 로봇 베이스와 6자유도 로봇 팔로 구성되어 있으며, RGB-D 카메라가 부착되어 있습니다. 초기 탐색 스캔을 통해 환경 조건을 파악하고, A* 알고리즘을 사용하여 최적의 스캐닝 경로를 계획합니다. 스캐닝은 로봇 베이스와 팔이 정해진 경로를 따라 움직이면서 이루어집니다.

- **Performance Highlights**: 실험 결과 전체 스캔 커버리지가 94.96%에 도달하며, 실제 인체에 대한 평균 표면 커버리지는 92.45%로 나타났습니다. 이 시스템은 법의학 문서화를 더 효율적이고 자율적으로 수행할 수 있는 가능성을 보여줍니다. 또한, PMCT 스캔과 보완하여 더욱 완전한 문서화를 제공할 수 있습니다.



### Temporal Misalignment and Probabilistic Neurons (https://arxiv.org/abs/2502.14487)
- **What's New**: 본 논문은 Spiking Neural Networks (SNNs)의 ANN-SNN 변환 과정에서 'temporal misalignment'라는 새로운 현상을 발견하고 이에 대한 이해를 제공합니다. 이 현상은 SNN 레이어 간의 랜덤한 스파이크 재배열이 성능 향상을 이끈다는 점을 강조합니다. 이 관찰을 바탕으로, 두 가지 단계의 확률적 스파이킹 뉴런(Two-phase Probabilistic Spiking Neurons, TPP)을 소개하여 변환 과정을 더욱 향상시킵니다.

- **Technical Details**: 이 연구에서는 Integrate-and-Fire (IF) 스파이킹 뉴런을 기반으로 한 모델을 사용하며, ANN에서 SNN으로의 변환 과정이 자세히 설명됩니다. ANN의 사전 훈련된 모델에서 SNN으로 가중치 및 바이어스를 전이하여 동일한 아키텍처 구조를 공유합니다. ANN 출력과 SNN 출력의 관계를 수학적으로 분석하여 스파이크 기준화 과정을 수행하며, 여기서 ReLU 함수를 활성화 함수로 사용합니다.

- **Performance Highlights**: 제안된 방법은 CIFAR-10/100, CIFAR10-DVS 및 ImageNet 데이터셋을 활용한 포괄적인 실험을 통해 기존의 SOTA(SOTA, State-Of-The-Art) 변환 방법 및 다른 훈련 방법들보다 높은 정확도를 기록했습니다. 연구 결과는 이론적 관점과 실증적 관점을 모두 포함하며, SNN 기반의 AI 모델이 기존의 인공신경망(ANN)보다 에너지 효율적인 대안으로 자리잡을 수 있음을 실증적으로 보여줍니다.



### Single-image Reflectance and Transmittance Estimation from Any Flatbed Scanner (https://arxiv.org/abs/2502.14462)
Comments:
          Accepted to Computers & Graphics

- **What's New**: 이번 연구에서는 전통적인 디지털 재료 캡처 방식의 한계를 극복하기 위해 모든 플랫베드 스캐너에서 사용할 수 있는 새로운 방법을 제안합니다. 이 방법은 shading(음영)과 specularity(사착)을 효과적으로 제거하고, 재료의 opacity(불투명도)와 transmittance(투과율)를 추정하여 더욱 사실적인 디지털 복제를 생성합니다. 또한, 이 과정에서 기존의 이미지-투-이미지 전환 네트워크 방법은 부족하다는 점을 지적하며, cycle-consistency 손실을 활용한 개선된 접근 방식을 소개합니다.

- **Technical Details**: 연구팀은 intrinsic image decomposition(내재적 이미지 분해) 방법에서 영감을 받아 새로운 재료 캡처 기술을 개발했습니다. 이 기술은 Spatially-Varying Bidirectional Scattering Distribution Function (SVBSDF)을 통해 복잡한 빛의 상호작용을 재현하며, 이는 고급 플랫베드 스캐너가 아닌, 스마트폰과 같은 다양한 스캐닝 장치에서도 잘 작동함을 보였습니다. 재료의 각 파라미터를 개별적으로 측정하는 이미지 기반 메트릭과 최종 appearance(외관)를 평가하는 렌더링 감지 메트릭을 모두 사용하여 철저한 실험을 진행했습니다.

- **Performance Highlights**: 본 방법은 여러 재료에서 매우 높은 해상도와 정확도로 완전한 SVBSDF를 추정함으로써, 스캐너의 조명이 무작위적이더라도 효과적인 결과를 생성합니다. 이는 사용자 친화적인 재료 캡처 설정을 통해 보다 저렴하고 접근하기 쉬운 방법으로, 디자인, 건축, 패션 등 다양한 산업에 적용될 수 있습니다. 실험 결과, 이 방법은 특히 섬유와 같은 얇은 재료의 모델링에서 뛰어난 성능을 발휘했습니다.



### ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Mod (https://arxiv.org/abs/2502.14420)
- **What's New**: 이번 연구에서는 Vision-Language-Action (VLA) 모델의 기존 훈련 패러다임을 체계적으로 분석하여, 스푸리어스 포겟팅(spurious forgetting)과 태스크 간 상호작용(task interference)이라는 두 가지 주요 도전 과제를 식별합니다. 이를 극복하기 위해, ChatVLA라는 새로운 프레임워크를 제안하며, 이는 다중 모달 데이터의 통합을 점진적으로 수행하는 Phased Alignment Training을 특징으로 합니다. 이러한 접근법은 향상된 다중 모달 이해를 통한 로봇 제어의 통합을 목표로 합니다.

- **Technical Details**: ChatVLA는 Mixture-of-Experts (MoE) 구조를 도입하여, 태스크 간 상호작용을 최소화하면서도 두 가지 태스크(즉, 이해와 조작)의 지식을 공유하도록 설계되었습니다. Phased Alignment Training 전략은 curriculum learning에서 영감을 받아 로봇 컨트롤을 먼저 익힌 후 다중 모달 데이터를 점진적으로 통합하는 방식입니다. 이를 통해 로봇 행동과 시각-텍스트 데이터 간의 미세한 정렬이 훼손되지 않도록 합니다.

- **Performance Highlights**: ChatVLA는 시각 질문-응답 데이터셋에서 경쟁력 있는 성능을 보여 주며, 다중 모달 이해 기준에서 최신 VLA 방법들을 현저하게 초월하는 결과를 나타냅니다. 특히, MMMU에서 6배 더 높은 성능을 달성하고 MMStar에서 47.2%의 점수를 기록하는 등 기계적 설계 효율성을 강조합니다. 또한, 25개의 실제 로봇 조작 작업에서 OpenVLA와 같은 기존 VLA 방법들과 비교하여 우수한 성능을 입증했습니다.



### Role of the Pretraining and the Adaptation data sizes for low-resource real-time MRI video segmentation (https://arxiv.org/abs/2502.14418)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 연구에서는 rtMRI(Real-time Magnetic Resonance Imaging)를 활용하여 음성 생성 중의 발음 기관의 움직임을 분석하는 새로운 접근 방식을 소개합니다. SegNet과 U-Net 모델을 이용한 Air-Tissue Boundary (ATB) 분할 작업을 통해 모델의 성능을 평가했으며, 제한된 데이터로도 효과적인 모델 조정이 가능하다는 것을 보여주었습니다. 특히, 새로운 데이터 세트에서 15개의 rtMRI 프레임만으로도 안정적인 성능을 달성할 수 있음을 확인했습니다.

- **Technical Details**: 이 연구에서는 rtMRI 비디오의 중축면에서 ATB을 예측하기 위해 SegNet 및 U-Net을 활용하였으며, 다양한 교육-검증 분할 및 조정 전략을 사용했습니다. ATB 분할은 음성 생성 연구에서 중요한 선처리 단계로 작용하며, 연구 결과는 제한된 수의 비디오 데이터에 대한 모델 조정이 필요함을 강조합니다. 두 가지 데이터 세트를 사용하여 실험을 진행하며, 기존 연구들과 비교해 보았습니다.

- **Performance Highlights**: 모델 조정 후, 첫 번째 데이터 세트에서 Pixel-wise Classification Accuracy (PCA)와 Dice Coefficient의 성능을 각각 0.33% 및 0.91% 향상했습니다. 두 번째 데이터 세트에서는 각각 99.63% 및 98.09%의 높은 정확도를 기록하며, 매칭 조건 성능에 비해 우수한 결과를 달성했습니다. 이러한 결과는 제한된 데이터로도 강력한 성능을 발휘할 수 있음을 나타내며, ATB 분할 기술의 효용성을 증가시킵니다.



### MedFuncta: Modality-Agnostic Representations Based on Efficient Neural Fields (https://arxiv.org/abs/2502.14401)
Comments:
          Code and Dataset: this https URL

- **What's New**: 이 연구는 의료 영상 분석에서 데이터 표현으로서 전통적으로 사용되는 격자(grid) 또는 복셀(voxel) 기반 접근 방식을 도전하며, 모달리티 불변(modality-agnostic)인 연속 데이터를 표현하는 MedFuncta를 소개합니다. MedFuncta는 신경 필드(neural fields)를 바탕으로 하여 의료 신호의 중복성을 이용하고, 효율적인 메타 학습(meta-learning) 접근법을 적용함으로써 단일 인스턴스에서 큰 데이터셋으로 확장할 수 있는 방법을 보여줍니다. 이 방법은 또한 SIREN 활성화의 스펙트럼 바이어스를 개선하여 재구성 품질 및 수렴 속도를 높입니다.

- **Technical Details**: 본 연구에서는 신경 네트워크의 파라미터 효율성을 극대화하기 위해, N개의 신호에 대한 기능적 표현을 학습하여 중복된 정보를 활용합니다. 각 신호는 특정 신호의 속성을 나타내기 위해 신호-특정 파라미터를 사용하는 공유 네트워크 파라미터를 통해 표현됩니다. K-레이어 MLP 아키텍처와 모듈화된 SIREN 활성화를 사용하여, 각 레이어는 신호에 특정한 정보를 기반으로 변화하는 양을 더하는 비선형성을 추가하여 구성됩니다.

- **Performance Highlights**: 다양한 차원과 모달리티의 의료 신호(1D: ECG, 2D: Chest X-ray 등, 3D: Brain MRI 등)에 대해 제안된 방법을 검증하며, 기존 신호 표현의 한계를 극복하고 관련 다운스트림 작업을 성공적으로 해결함을 입증합니다. 이 연구는 550,000개 이상의 주석이 달린 신경 필드로 구성된 대규모 데이터셋을 발표하여, 추후 연구를 촉진하는 데 기여하고 있습니다.



### A Similarity Paradigm Through Textual Regularization Without Forgetting (https://arxiv.org/abs/2502.14376)
- **What's New**: 이 논문에서는 비슷한 분포의 데이터에 대한 일반화 성능 저하 문제를 해결하기 위해 SPTR(Similarity Paradigm with Textual Regularization)이라는 새로운 방법론을 제안합니다. SPTR는 고안된 텍스트 프롬프트를 활용하여 전처리된 비전-언어 모델(VLM)의 일반 텍스트 지식을 유지하면서 학습할 수 있도록 구성됩니다. 이를 통해 모델은 새로운 클래스에 대한 일반화 가능성을 높일 수 있습니다.

- **Technical Details**: SPTR는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 최적 수송(optimal transport)을 통해 고안된 특징과 조정된 텍스트 특징 간의 근사치를 보장하여 일반 텍스트 지식을 잊지 않도록 합니다. 2) 자연 정렬 점수(natural alignment score)와 적대적 정렬 점수(adversarial alignment score)를 통한 유사성 패러다임을 적용하여 여러 텍스트 프롬프트의 일반 능력을 지속적으로 발휘할 수 있게 합니다.

- **Performance Highlights**: SPTR는 비전-언어 모델에 대한 학습에서 개선된 성능을 보여주며, 다양한 분류 작업에서 기존 프롬프트 학습 방법들과 비교하여 우수한 결과를 기록했습니다. 4444개의 대표 작업과 11111111개의 데이터세트에 대한 광범위한 실험 결과, SPTR는 최신 기술(state-of-the-art) 성능을 달성했음을 보여주었습니다.



### PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization (https://arxiv.org/abs/2502.14370)
Comments:
          6 pages, submitting to ICML 2025

- **What's New**: 본 논문에서는 PPO-MI라는 새로운 강화를 기반으로 한 모델 역전 공격 프레임워크를 제안합니다. 기존 방법들에 비해 요구되는 공격 지식이 적고, 높은 쿼리 효율성을 보장합니다. 이 접근법은 강화 학습의 Markov Decision Process(MDP)로 역전 작업을 공식화하여, 생성 모델의 잠재 공간을 탐색하는 에이전트를 활용합니다.

- **Technical Details**: PPO-MI는 Proximal Policy Optimization(PPO) 및 모멘텀 기반 상태 전이 메커니즘을 활용하여 데이터 재구성을 위한 효율적인 탐색을 수행합니다. 에이전트는 오직 모델의 예측만을 사용하여, 고차원 잠재 공간에서 훈련 샘플을 재구성하는 방법을 학습합니다. 이를 통해 모델의 내부 상태나 경량을 알지 못하는 블랙박스 환경에서도 효과적으로 작업을 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과 PPO-MI는 기존의 모델 역전 공격 방법들보다 우수한 성능을 보였으며, 다양한 모델 아키텍처와 데이터 세트에 걸쳐 강력한 내구성을 입증했습니다. 이 결과는 PPO-MI가 실용적인 블랙박스 시나리오에서 매우 유용함을 강조합니다. 특히, 이 방법은 라벨만으로 이루어진 공격에 대한 새로운 탐구의 길을 열어줍니다.



### Topology-Aware Wavelet Mamba for Airway Structure Segmentation in Postoperative Recurrent Nasopharyngeal Carcinoma CT Scans (https://arxiv.org/abs/2502.14363)
Comments:
          20 pages, 11 figures, 6 tables

- **What's New**: 이번 논문에서는 재발한 비인두암(NPC) 환자를 위한 수술 후 기도 위험 평가의 정확성을 높이기 위해 설계된 새로운 분할 모델 TopoWMamba(Topology-aware Wavelet Mamba)를 소개합니다. 이 모델은 Wavelet 기반의 다중 스케일(feature extraction) 기능과 상태 공간(sequence modeling) 모델링, 그리고 위상의식을 갖춘 모듈들을 결합하여 CT 스캔의 기도 관련 구조를 견고하게 분할합니다. 이 접근법은 정밀한 경계 감지와 전반적인 구조 컨텍스트 캡처를 가능하게 하여 복잡한 수술 후 시나리오에서 기도 구조를 정확하게 분할하는 데 기여합니다.

- **Technical Details**: TopoWMamba 모델은 Wavelet 변환을 사용하여 다중 스케일 특징을 추출하고 상태 공간(sequence modeling) 모델을 통해 항구적인 구조 연속성을 보장합니다. 이 구조는 스네이크(convolution) 기반의 VSS(Snake Conv VSS) 모듈을 포함하여 복잡한 경계가 있는 기도 구조의 감지를 최적화합니다. 이 모델은 NPCSegCT 데이터세트를 통해 테스트되어 평균 Dice 점수 88.02%를 기록하며 기존의 UNet, Attention UNet, SwinUNet 모델보다 뛰어난 분할 정확도를 보였습니다.

- **Performance Highlights**: TopoWMamba는 SegRap 2023 Challenge 데이터세트에서도 테스트되어 기도 분할에서 95.26%라는 매우 높은 Dice 점수를 달성했습니다. 이러한 성능은 수술 후 기도 위험 평가의 정확성을 높이는 데 기여하여 자동화된 분할을 가능하게 합니다. 그 결과, 향후 기도 위험 예측 모델링에 대한 연구의 기초를 제공합니다.



### EyeBench: A Call for More Rigorous Evaluation of Retinal Image Enhancemen (https://arxiv.org/abs/2502.14260)
- **What's New**: 본 논문에서는 EyeBench라는 새로운 포괄적 벤치마크를 제안하여, 망막 이미지 향상 방법을 평가하는 데 필요한 종합적인 평가 시스템을 제공하고 있습니다. 기존의 평가 메트릭이 실제 임상 연구와의 연계에서 부족함을 드러내고, 전문가의 프로토콜이 필요함을 지적하며 이러한 틀을 눈높이에 맞춘 다차원적 평가 접근법으로 보완하고자 합니다.

- **Technical Details**: EyeBench는 향상된 망막 이미지의 평가를 위한 다양한 다운스트림 작업을 포함하여, 질병 등급화, 혈관 분할 및 병변 분할과 같은 임상적으로 중요한 평가 항목들을 제시합니다. 우리는 전문가의 지도 아래 비너리 및 비비너리 방법 간의 철저한 비교를 촉진하기 위한 새로운 데이터셋을 개발하였고, 해당 데이터셋은 충분한 참조 조건에서 고유 처리 및 테스트 세트를 포함합니다.

- **Performance Highlights**: EyeBench의 다차원 평가는 의료 전문가가 올바른 향상 방법을 선택하는 데 도움을 줄 수 있으며, 특히 임상적으로 가치 있는 비비너리 방법에 대한 상세 분석도 제공합니다. 또한 기존 방법들이 직면한 도전 과제를 포괄적으로 분석하여 향후 연구 방향에 대한 통찰력을 제공합니다.



### Pandora3D: A Comprehensive Framework for High-Quality 3D Shape and Texture Generation (https://arxiv.org/abs/2502.14247)
Comments:
          Tencent XR 3D Gen

- **What's New**: 이 보고서는 다양한 입력 프롬프트에서 고품질 3D 형태 및 텍스처를 생성하기 위한 포괄적인 프레임워크인 Pandora3D를 소개합니다. 이 프레임워크는 3D 형태 생성과 텍스처 생성으로 구성되어 있으며, 각 프로세스는 고급 신경 아키텍처를 활용하여 다양한 입력 형식을 효과적으로 처리할 수 있습니다. 특히, Variational Autoencoder (VAE)와 Diffusion Network를 이용한 개선된 모델이 도입되었습니다.

- **Technical Details**: Pandora3D의 3D 형태 생성 파이프라인은 VAE를 사용하여 3D 기하학을 잠재 공간(latent space)으로 인코딩하며, Diffusion Network가 입력 프롬프트에 조건화된 잠재 표현을 생성합니다. 이 과정은 CLAY, Craftsman, LAM3D의 구조를 바탕으로 한 모델 개조를 포함하며, 점진적 샘플링 프로세스를 통해 높은 기하학적 복잡성을 포착하고, 세밀한 디테일을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 보고서에 따르면, Pandora3D는 다양한 입력 형식을 처리하면서 고품질의 3D 콘텐츠를 생성하는 데 성공하였습니다. 다단계 텍스처 생성 프로세스에서는 픽셀 수준의 일관성을 유지하기 위한 novel consistency scheduler가 각 단계에 통합되어 무결한 3D 표현을 보장합니다. 이러한 성과는 시스템 아키텍처, 실험 결과 및 향후 가능한 발전 방향에 대한 상세한 설명과 함께 제공됩니다.



### Asymmetric Co-Training for Source-Free Few-Shot Domain Adaptation (https://arxiv.org/abs/2502.14214)
Comments:
          13 pages

- **What's New**: 이 논문에서는 비지도 영역 적응의 새로운 방법인 비대칭 공동 학습(asymmetric co-training, ACT)을 제안합니다. 이 방법은 소수의 레이블이 있는 타겟 데이터만으로 사전 훈련된 모델을 조정하는 것을 목표로 하며, 실용적인 대안으로 여겨집니다. 특히, 연구자들은 레이블이 없는 대량의 타겟 데이터 없이도 미세 조정할 수 있는 가능성을 탐구합니다.

- **Technical Details**: ACT 방법은 약한-강한 증강(weak-strong augmentation)을 사용하여 데이터의 다양성을 높이는 것으로 시작합니다. 또한, 두 단계의 최적화 과정에서 레이블 스무딩 크로스 엔트로피 손실, 클래스 조건부 분포의 엔트로피, 및 역 엔트로피 손실을 최적화하여 과적합을 줄이고 모델의 구별 능력을 향상시킵니다. 마지막 단계에서는 분류기의 결정성 차이를 최소화하여 예측을 개선합니다.

- **Performance Highlights**: 실험 결과, ACT 방법은 기존의 최신 SFUDA 방법들보다 더 나은 성능을 보였습니다. 특히, 소량의 레이블이 있는 타겟 인스턴스를 활용하여 사전 훈련된 모델을 조정하는 것이 실제 상황에서 실용적이고 신뢰할 수 있는 해결책으로 나타났습니다. 논문에서는 여러 벤치마크를 통해 ACT의 우수성을 입증했습니다.



### NeRF-3DTalker: Neural Radiance Field with 3D Prior Aided Audio Disentanglement for Talking Head Synthesis (https://arxiv.org/abs/2502.14178)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 논문에서는 NeRF(Neural Radiance Field)를 이용하여 오디오를 기반으로 고해상도의 말하는 얼굴 비디오를 합성하는 새로운 방법인 NeRF-3DTalker를 제안합니다. 기존의 NeRF 방법들은 주로 정면 얼굴 렌더링에 초점을 맞추었으며, 새로운 시점에서의 합성에는 한계가 있었습니다. 이 연구는 3D Prior 정보와 오디오의 분리(disentanglement)를 통해 더 명확하고 정확한 합성을 목표로 하고 있습니다.

- **Technical Details**: NeRF-3DTalker는 3DMM(3D Morphable Model)에서 추출한 3D prior 정보를 활용하여 다양한 시점에서의 말하는 얼굴을 생성할 수 있도록 구성되어 있습니다. 또한, 3D Prior Aided Audio Disentanglement 모듈을 통해 오디오를 3D 말하기 동작과 말하기 스타일 관련 두 가지 특징으로 분리합니다. 이 기술은 3D 시각 공간과 음향 공간을 정렬하여 더 효과적인 3D 말하는 얼굴 합성을 가능하게 합니다.

- **Performance Highlights**: 연구 결과, NeRF-3DTalker는 정교한 영상 품질과 뛰어난 입술 동기화(lip-sync)를 기반으로 최신 기술(state-of-the-art)보다 우수한 성능을 나타냅니다. 제안된 방법은 종합적인 정성적 및 정량적 실험을 통해 그 유효성이 입증되었습니다. 따라서, 연구진은 더 현실감 있는 다각도 말하는 얼굴 비디오를 생성하는 새로운 길을 열었다고 결론내립니다.



### Hybrid Visual Servoing of Tendon-driven Continuum Robots (https://arxiv.org/abs/2502.14092)
- **What's New**: 본 논문에서는 텐던 구동 연속 로봇(TDCRs) 제어를 위한 새로운 하이브리드 비주얼 서보(HVS) 접근 방식을 소개합니다. HVS 시스템은 이미지 기반 비주얼 서보(IBVS)와 딥러닝 기반 비주얼 서보(DLBVS)를 결합하여 각 방법의 한계를 극복하고 전반적인 성능을 향상시킵니다. 이 접근 방식은 동적이고 비구조적인 환경에서도 효과적인 제어를 보장하도록 IBVS와 DLBVS 간의 원활한 전환을 가능하게 합니다.

- **Technical Details**: 본 연구에서는 HVS를 지원하기 위해 IBVS와 DLBVS의 전통적인 제어 방법을 구체적으로 설명합니다. IBVS는 현재와 목표 특성 간의 픽셀 오류를 최소화하는 것을 목표로 하며, DLBVS는 CNN을 활용하여 시각 입력을 제어 명령으로 직접 매핑합니다. 두 접근 방식의 통합을 통해 TDCR의 성능을 향상시키는 하이브리드 방식을 제안합니다.

- **Performance Highlights**: HVS 접근 방식은 시뮬레이션 및 실제 실험을 통해 검증되었으며, DLBVS 단독 사용 시에 비해 반복 시간 감소, 빠른 수렴 속도, 최종 오류 낮춤 및 보다 부드러운 성능을 보여주었습니다. HVS는 차폐, 조명 변화, 액추에이터 소음, 물리적 충격과 같은 도전적인 조건에서도 DLBVS의 강인성을 유지합니다. 이러한 결과는 TDCR이 복잡하고 불확실한 환경에서 효과적으로 작동할 수 있음을 나타냅니다.



### MambaLiteSR: Image Super-Resolution with Low-Rank Mamba using Knowledge Distillation (https://arxiv.org/abs/2502.14090)
Comments:
          Special Session: Generative AI on Edge, 26th International Symposium on Quality Electronic Design (ISQED'25)

- **What's New**: 이번 논문에서 제안된 MambaLiteSR은 실시간 처리가 필요한 엣지 장치에서의 이미지 슈퍼 해상도(task) 문제를 해결하기 위해 개발된 새로운 경량(generated AI) 모델입니다. 이 모델은 Vision Mamba 아키텍처를 기반으로 하여 State Space Blocks 및 재구성 모듈을 접목하여 효율적인 특징 추출을 구현합니다. 특징적으로, MambaLiteSR은 큰 모델에서 얻은 지식을 경량 모델로 전이하는 knowledge distillation 기법을 활용하여 성능 저하 없이 효율성을 최적화합니다.

- **Technical Details**: MambaLiteSR은 PSNR 및 SSIM 지표를 통해 성능을 평가하며, 모델 파라미터와 동적 전력 소비를 세밀하게 조정합니다. 특히 low-rank approximation을 채택하여 훈련 시 전력 사용량을 줄이고, knowledge distillation을 통해 모델 크기를 감소시킵니다. 이 연구는 Mamba 기반 아키텍처와 knowledge distillation, low-rank approximation의 동시 사용을 탐구한 최초의 사례로, 경량성과 성능 간의 균형을 잘 이룹니다.

- **Performance Highlights**: MambaLiteSR은 NVIDIA Jetson Orin Nano에 배포된 실험 결과, 기존의 엣지 SR 모델에 비해 파라미터를 15% 줄이면서도 비슷한 PSNR 및 SSIM 성능을 달성했습니다. 또한 최신 기술에 비해 전력 소비를 58%까지 감소시키며, 저전력 상태를 유지하면서도 훈련 과정에서 최적의 에너지 사용을 보장합니다. 이로 인해 MambaLiteSR은 자원이 제한된 장비에서도 효과적으로 배포될 수 있는 가능성을 보여줍니다.



### Dynamic Activation with Knowledge Distillation for Energy-Efficient Spiking NN Ensembles (https://arxiv.org/abs/2502.14023)
- **What's New**: 본 연구는 인공지능(AI) 모델에서의 혁신을 통해 에너지가 제한된 환경에서도 고효율을 추구합니다. 고전적인 인공신경망(ANN) 대신, 사건 기반의 신경망인 스파이킹 신경망(SNN)을 활용하여 에너지 효율성과 성능 향상을 도모하는 새로운 시스템을 제안합니다. 이 시스템은 지식 증류(knowledge distillation)와 앙상블 학습(ensemble learning)을 결합하여 인공지능 모델의 성능 격차를 해소하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 기초 AI 모델에서 획득한 지식을 활용하여 작은 SNN 모델들로 구성된 스파이킹 신경 앙상블(Spiking Neural Ensemble, SNE)을 훈련시키는 방식입니다. SNE는 각각의 학생 모델이 학습하는 동안 교사 네트워크(큰 ANN)로부터 얻은 피쳐를 분리(disentangle)하는 과정을 포함하며, 이로 인해 각 학생이 독립적인 예측 작업을 수행할 수 있도록 합니다. 이 과정에서 주어진 입력에 대해 학생 모델의 일부만 활성화하여 에너지 효율을 극대화할 수 있습니다.

- **Performance Highlights**: SNE는 CIFAR-10 데이터셋에서 교사 네트워크 대비 20배 이상의 계산 요건 감소를 이루며, 단 2%의 정확도 손실로 아키텍처의 효율성을 향상시킵니다. 또한, 학생 앙상블에서 적절한 수의 활성 학생 모델을 선택함으로써 에너지 소비를 최대 65% 감소시키는 동시에 정확도는 약 2% 떨어지는 정도로 유지되는 성과를 거두었습니다. 노이즈가 있는 상황에서도 SNE는 기존 ANN 교사 모델보다 우수한 강건성을 보이며, 이러한 디스엔탱글먼트 절차는 전통적인 분할 방식에 비해 최대 2.4%의 정확도 향상으로 이어졌습니다.



### Segmentation-free integration of nuclei morphology and spatial transcriptomics for retinal images (https://arxiv.org/abs/2502.13974)
- **What's New**: 이 연구에서는 SEFI (SEgmentation-Free Integration)라는 새로운 방법을 소개합니다. SEFI는 형태학적 특성(morphological features)을 공간 전사체학 데이터(spatial transcriptomics data)와 통합하는 혁신적인 접근을 제공합니다. 기존의 세포 분할(cell segmentation) 문제를 해결하며, 세포를 직접적으로 구별하지 않고도 유용한 정보를 추출할 수 있습니다. 이 방법은 특히 개발 중인 망막(retina)의 유전자 발현 분석에 적용되었습니다.

- **Technical Details**: SEFI는 세 가지 주요 단계로 구성됩니다. 첫째, 공간 전사체학 데이터에서 개별 유전자 발현 맵을 생성합니다. 다음으로, DAPI 염색된 핵의 형태학 이미지를 CNN(Convolutional Neural Networks)을 사용하여 특징을 추출합니다. 마지막으로, 유전자와 형태학적 특징을 k-평균 군집화(k-means clustering)를 통해 클러스터링한 후, 계층적 군집화(hierarchical clustering)를 이용해 클러스터를 병합합니다.

- **Performance Highlights**: 연구 결과, SEFI를 통한 군집화는 생물학적으로도 의미 있는 결과를 보여주었습니다. 33개의 유전자와 형태학적 특성을 사용하여 얻은 클러스터는 망막의 특정 영역과 일관되게 연관되었습니다. 예를 들어, 군집 7은 신경절 세포(ganglion cells)의 마커가 풍부하게 분포되어 있으며(more enriched), 군집 1은 섬모 가장자리 세포(ciliary margin cells)로 특징지어집니다. 이러한 결과는 SEFI가 세포 유형 분류(classification)에서 유전자 발현 데이터만으로는 식별할 수 없는 추가 정보를 제공함을 보여줍니다.



### Face Deepfakes -- A Comprehensive Review (https://arxiv.org/abs/2502.09812)
- **What's New**: 이 논문에서는 최신 심층 위조(deepfake) 생성 및 탐지 기술에 대한 포괄적인 이론 분석을 제공하고 있으며, 특히 얼굴 인식 생체 인식(face biometric recognition) 접근 방식에 미치는 영향을 체계적으로 평가합니다. 심층 위조 기술이 가져올 수 있는 긍정적, 부정적 응용에 대해 논의하고, 기존 연구의 공백을 언급하며 향후 연구 방향을 제안합니다.

- **Technical Details**: 심층 위조 기술은 크게 네 가지로 분류될 수 있습니다: (i) 전체 얼굴 합성, (ii) 신원 변경(identity swap), (iii) 속성 변형(attribute manipulation), (iv) 표정 변경(expression swap)입니다. 얼굴 합성 모델은 사람의 얼굴을 다른 얼굴로 교체할 수 있으며, 최근에는 동영상을 통한 얼굴 재연(facial reenactment) 기술이 발전하여 더욱 유연하게 작용할 수 있게 되었습니다.

- **Performance Highlights**: 딥페이크 기술은 정보 왜곡과 잘못된 정보 생성에서 중요한 역할을 하고 있으며, 기존의 탐지 시스템을 피하는 능력이 있습니다. 논문에서는 다양한 알고리즘과 접근 방법을 통해 얼굴 깊이 흔들기를 가능하게 하는 기술적 세부 사항을 다루고 있으며, 향후 심층 위조 기술에 대한 사회적 인식 제고의 필요성을 강조합니다.



### SASVi -- Segment Any Surgical Video (https://arxiv.org/abs/2502.09653)
- **What's New**: 이 논문에서는 SASVi라는 새로운 리프롬프트 기법을 제안합니다. SASVi는 수술 비디오의 세분화(segmentation)에서 시간적 일관성을 향상시키는 데 효과적입니다. 기존의 방법들은 수술 비디오에 대한 구체적인 도메인 지식 없이 작동하는 문제점이 있었으나, 본 연구에서는 이러한 한계를 극복했습니다.

- **Technical Details**: SASVi는 Mask R-CNN Overseer 모델을 기반으로 한 리프롬프트 기법으로, 최소한의 주석 데이터로 학습됩니다. 이 모델은 장면 구성이 변화할 때 SAM2 모델을 자동으로 리프롬프트하여, 수술 비디오의 세그멘테이션에서 일관성과 연속성을 유지할 수 있게 합니다. 이를 통해 다양한 수술 비디오를 효과적으로 처리할 수 있는 새로운 방법론을 제공합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 다른 수술 데이터셋에서 정량적 및 정성적으로 수행된 시험 결과를 통해 효과적으로 입증되었습니다. SASVi를 활용하면 수술 비디오에서 보다 매끄럽고 일관된 세그멘테이션을 생성할 수 있습니다. 또한, SASVi는 소량의 주석을 활용하여 대규모 수술 비디오 데이터셋에 대한 주석을 공개함으로써 향후 수술 데이터 과학 모델 개발을 지원할 것입니다.



New uploads on arXiv(cs.AI)

### Benchmarking Multimodal RAG through a Chart-based Document Question-Answering Generation Framework (https://arxiv.org/abs/2502.14864)
- **What's New**: 이번 연구에서는 Chart-based MRAG라는 새로운 작업을 도입하여 기존의 MRAG 모델이 다루지 못한 복잡한 시각적 형식, 즉 차트를 포함한 분류 문제를 해결하고자 합니다. 기존의 벤치마크가 주로 단순한 이미지-텍스트 상호작용에 집중하고 있는 점을 비판하며, 고급 QA 샘플을 세미 자동으로 생성하기 위한 CHARt-based document question-answering GEneration (CHARGE) 프레임워크를 제안합니다.

- **Technical Details**: CHARGE는 텍스트 및 차트 데이터를 구조화된 키포인트 추출, 교차 모달 검증, 키포인트 기반 생성 과정을 통해 고품질 QA 쌍을 생성합니다. 이 프레임워크는 각 모달리티의 기여를 별도로 평가할 수 있도록 Text-Chart MRAG, Text-only RAG, Chart-only MRAG과 같은 하위 작업을 포함합니다. Chart-MRAG Bench는 8개 분야에서 4,738개의 질문-응답 쌍을 포함하며, 이는 실제 데이터에서 도출된 것이어서 복잡한 교차 모달 인터랙션을 반영합니다.

- **Performance Highlights**: 평가 결과, 통합된 멀티모달 임베딩 검색 방법이 차트 중심의 시나리오에서 성능이 떨어진다는 것을 확인했습니다. 또한 최신 MLLM조차도 정답 기반 검색을 사용한 경우에도 Correctness는 58.19%, Coverage는 73.87%에 불과하여, 텍스트-차트 다모드 추론에서 지속적인 문제점을 드러냈습니다. 연구진들은 이러한 결과를 통해 차트 중심의 MRAG 모델에 대한 향후 연구 방향성을 제시합니다.



### Optimizing Model Selection for Compound AI Systems (https://arxiv.org/abs/2502.14815)
- **What's New**: 이번 연구에서는 복합 AI 시스템에서 여러 LLM 호출을 결합하여 복잡한 작업을 수행하는 새로운 모델 선택 방법인 LLMSelector를 제안합니다. 이 방법은 각 모듈에 최적의 LLM 선택이 성능에 미치는 영향을 분석하고, 여러 모듈에서 서로 다른 모델을 사용하는 경우 성능이 크게 향상된다는 점을 발견했습니다. 특히, LLMSelector는 사용되는 API 호출 수를 모듈 수에 비례하여 선형으로 줄일 수 있는 효율적인 프레임워크입니다.

- **Technical Details**: LLMSelector는 두 가지 주요 연구 결과를 바탕으로 설계되었습니다. 첫째, 전체 성능은 각 모듈의 성능이 고정되어 있을 때 모듈의 성능에 따라 단조적으로 변화하며, 둘째, 각 모듈의 성능은 LLM을 통해 정확하게 추정될 수 있습니다. 이를 통해 LLMSelector는 각 모듈에 대해 가장 높은 성능을 보이는 모델을 반복적으로 선택하고 할당하여 전체 성능을 극대화하는 방식으로 작동합니다.

- **Performance Highlights**: 다양한 실험을 통해 LLMSelector는 복합 AI 시스템에서 동일한 LLM을 사용하는 경우보다 5%에서 최대 70%까지 성능 향상을 가져오는 것으로 나타났습니다. 또한, LLMSelector는 프롬프트 최적화에 대한 기존 기법들보다도 우수한 결과를 보여주었으며, 이는 모델 선택의 중요성을 다시 한번 부각시킵니다. 따라서 LLMSelector는 고차원 복합 시스템에서 최적의 성능을 이끌어내기 위한 유용한 도구로 평가됩니다.



### Making Universal Policies Universa (https://arxiv.org/abs/2502.14777)
- **What's New**: 이 연구는 다양한 행동 공간을 가진 에이전트들이 동일한 관찰 공간을 공유하는 크로스 에이전트 설정에서 포괄적인 에이전트를 개발하는 어려움을 다룹니다. 이를 위해, 여러 에이전트의 궤적을 포함하는 공동 데이터 세트를 기반으로 계획자(planner)를 훈련시키는 방법을 제안하며, 이는 긍정적 전이(positive transfer)의 이점을 촉진합니다. 해당 방법을 통해 에이전트 특유의 제약에 적응하는 것이 주요 과제입니다.

- **Technical Details**: 본 연구는 유니버설 정책(universal policy) 프레임워크를 기반으로 하여, 관찰 시퀀스를 생성하는 확산 기반(planner)을 포함하고, 이러한 계획에 대해 행동을 할당하는 역 역학 모델(inverse dynamics model)을 사용하는 두 단계로 정책 학습을 분리합니다. 마지막에, 여러 에이전트의 작은 지침-궤적 쌍 데이터 세트를 사용하여 정책 학습을 수행하며, 조건에 따라 계획자를 조정하는 방법들을 탐구합니다.

- **Performance Highlights**: 이 연구는 BabyAI 환경에서 다양한 복잡성의 작업을 평가하여 에이전트들 간의 긍정적 전이를 입증하였습니다. 여러 에이전트에서의 pooled dataset으로 훈련한 유니버설 정책은 단일 에이전트 데이터 세트로 훈련한 정책 대비 최대 42.20% 향상된 작업 완료 정확도를 달성했습니다. 이는 공유 계획자와 에이전트 사용에 따른 역 역학 모델의 결합의 중요성을 강조합니다.



### EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations (https://arxiv.org/abs/2502.14760)
- **What's New**: 이번 연구에서는 최적화 문제에서의 동등한 공식화(equivalent formulations) 식별이라는 문제를 다루고 있습니다. 특히, 기존 접근법의 한계를 극복하고 Karp 축소(Karp reductions)의 영감을 바탕으로, quasi-Karp 동등성(quasi-Karp equivalence)이라는 새로운 공식화를 제안합니다. EquivaMap라는 프레임워크를 통해 자연어 기술을 활용하여 이러한 매핑을 자동으로 발견함으로써 동등성 검증을 용이하게 합니다.

- **Technical Details**: Combinatorial optimization은 제한된 후보 집합 중 최적의 객체를 찾는 문제를 다루며, Mixed-Integer Linear Programming (MILP)은 이러한 문제의 주요 도구로 활용됩니다. 연구는 특히 MILP 모델링을 위한 대형 언어 모델(LLMs)을 활용한 접근법과 자동 동등성 검증 메커니즘을 강조합니다. Karp 축소 이론에 기반하여, 두 개의 최적화 공식 간의 동등성을 공식적으로 정의하고 이를 리지 스케일로 검증할 수 있도록 하는 EquivaMap을 제안합니다.

- **Performance Highlights**: Empirical 분석 결과, EquivaMap은 기존 방법들에 비해 크게 향상된 성능을 보였습니다. 특히, EquivaFormulation이라는 새로운 데이터셋을 구축하여 동등한 공식화와 그 변환을 기록함으로써 성능 평가의 기초를 마련했습니다. 다양한 동등 변환에 대해 일관되게 우수한 성과를 나타내며, 최적화 코파일럿의 효용성을 극대화하는 기초 작업으로 자리잡을 것으로 기대됩니다.



### From Knowledge Generation to Knowledge Verification: Examining the BioMedical Generative Capabilities of ChatGP (https://arxiv.org/abs/2502.14714)
Comments:
          26 pages, 6 figures, In Review with a Cell Press Journal

- **What's New**: 본 논문은 LLM 모델이 생성한 생물 의학 정보의 사실 확인을 위한 체계적인 평가 접근 방식을 제안합니다. 구체적으로 질병, 약물, 증상, 유전자 간의 연관성 생성을 포함한 두 가지 핵심 프로세스를 도입했습니다. 이 방식은 ChatGPT를 사용하여 정량적인 평가 기초를 마련하고자 하며, 생물학적 네트워크에서의 정확성을 보장하는 데 중점을 두었습니다.

- **Technical Details**: 정확한 평가를 위해, 다양한 생물 의학 온톨로지를 활용하여 생성된 용어와 연관성을 검증하는 두 가지 주요 작업을 수행했습니다. 첫 번째 작업에서는 GO, DOID, ChEBI 및 증상 온톨로지를 통해 생성된 용어의 정확성을 확인했습니다. 두 번째 작업은 PubMed 데이터베이스를 사용하여 용어 간의 연관성을 검증했으며, 최종적으로 지식의 일관성을 유지하기 위한 세 번째 작업도 수행했습니다.

- **Performance Highlights**: 전반적으로 질병 용어(88%-97%)와 약물 이름(90%-91%) 식별에서 높은 정확성을 기록했고, 유전 정보는 88%-98%의 정확도를 보였습니다. 그러나 증상 용어 식별 정확도는 49%-61%로 상반된 결과를 나타냈습니다. 실험 결과로 도출된 질병-약물 및 질병-유전자 연관성 검증 시 문헌 커버리지는 각각 89%-91%에 달했습니다.



### Building reliable sim driving agents by scaling self-play (https://arxiv.org/abs/2502.14706)
Comments:
          First version

- **What's New**: 본 논문에서는 자율주행차(AV)와 같은 인간과 상호작용하는 시스템을 설계하고 테스트하기 위한 시뮬레이션 에이전트의 중요성에 대해 논의합니다. 본 연구는 Waymo Open Motion Dataset에서 수천 개의 시나리오에 대한 자기 플레이(self-play) 기반 훈련을 확장하여 신뢰할 수 있는 시뮬레이션 에이전트를 구축하는 방법을 제안합니다. 이를 통해 의도한 대로 행동하는 에이전트를 만들기 위한 기초를 설정하고자 하였습니다.

- **Technical Details**: 제안된 시뮬레이션 에이전트는 단일 GPU에서 하루 만에 전체 훈련 세트를 거의 해결합니다. 이들은 미지의 테스트 장면에 대해서도 효과적으로 일반화하여 10,000개의 보류된 시나리오에서 99.8%의 목표 완수율과 0.8% 미만의 충돌 및 오프로드 사건을 달성합니다. 또한 이 에이전트는 데이터가 부족한 장면에 대해 부분적인 견고성을 보이며, 몇 분 안에 미세 조정(fine-tuning)하여 그 경우에 거의 완벽한 성능에 도달할 수 있습니다.

- **Performance Highlights**: 제안된 에이전트는 주어진 훈련 세트 및 다양한 환경에서 높은 성능을 발휘합니다. 10,000개의 시나리오에서 목표 완수율 99.8%와 함께 0.8% 이하의 충돌 및 오프로드 사건이라는 결과를 나타냅니다. 또한, 이들은 미지의 분포(out-of-distribution) 장면에 대해서도 뛰어난 성능을 보여주며, 연구자들이 쉽게 활용할 수 있도록 전이 학습 및 코드 베이스를 오픈 소스로 제공하고 있습니다.



### A Statistical Case Against Empirical Human-AI Alignmen (https://arxiv.org/abs/2502.14581)
Comments:
          24 pages, 2 figures, 5 tables

- **What's New**: 본 논문은 AI 시스템과 인간 간의 윤리적 정렬을 강조하면서, 단순한 경험적 정렬의 위험성을 경고합니다. 저자들은 경험적 정렬이 통계적 편향을 초래할 수 있다고 주장하며, 이러한 편향이 AI의 성능과 진화에 부정적인 영향을 미칠 수 있음을 시사합니다. 이 논문에서는 경험적 정렬의 대안으로, 정 prescription적 정렬과 경험적 후방 정렬을 제안합니다.

- **Technical Details**: 지금까지 언급된 정렬 개념은 통계적 관점에서 재조명되고 있으며, 저자들은 통계적 가정의 존재를 강조합니다. 특히, 정렬을 위해 가정하는 대표성 샘플과 혼란 변수의 부재가 중요한 요소로 다뤄집니다. 저자들은 경험적 정렬이 AI 모델에 새로운 편향을 도입함으로써 본래의 목표에서 벗어날 수 있음을 주장합니다.

- **Performance Highlights**: 저자들은 LLM(대형 언어 모델)을 활용한 사례 연구를 통해 지식 발견 및 인간 중심 접근의 한계에 대해 논의합니다. 또한, 경험적 정렬이 AI 시스템의 과학적 발견 가능성을 제한한다는 점을 강조합니다. 논문은 경험적 정렬에 따른 편향이 AI의 잠재력을 제약한다고 결론짓고, 대신에 규범적 접근 방식을 채택할 것을 권장합니다.



### Plan-over-Graph: Towards Parallelable LLM Agent Schedu (https://arxiv.org/abs/2502.14563)
- **What's New**: 이 논문은 'plan-over-graph'라는 새로운 패러다임을 도입하여, 실제 텍스트 작업을 실행 가능한 하위 작업으로 분해하고 추상적인 작업 그래프를 구축하는 과정을 소개합니다. 이 모델은 이러한 작업 그래프를 입력으로 사용하여 병렬 실행을 위한 계획을 생성합니다. 또한, 복잡하고 확장 가능한 그래프의 계획 능력을 향상시키기 위해 자동화된 파이프라인과 두 단계 훈련 기법을 설계하였습니다.

- **Technical Details**: 이 연구에서는 복잡한 작업을 묘사하는 연결된 비순환 그래프(Directed Acyclic Graph, DAG)를 초기화하고 각 하위 작업의 가능 솔루션 및 최적 솔루션을 주석 처리합니다. 이후, LLM이 자연어로 작업 설명을 생생하게 만들어 실제 시나리오에 적용될 수 있는 작업들을 구성합니다. 두 단계의 훈련 전략은 추상 그래프에서 수행되며, 추론 과정에서 텍스트 쿼리를 추출하고 훈련된 어댑터를 통해 그래프 위에서 계획 수립을 수행합니다.

- **Performance Highlights**: 실험 결과, 'plan-over-graph' 접근 방식은 API 기반 LLM과 오픈 소스 LLM 모두에서 작업 성능을 크게 향상시키는 것으로 나타났습니다. 그래프의 구조적 확장성, 병렬 실행이 시간 효율성을 어떻게 개선하는지에 대한 분석을 통해, 작업 계획에서 흔히 발생하는 오류를 식별하였습니다. 전체적인 결과는 성공률, 최적 정확도, 실행 가능 정확도 및 효율성을 포함한 포괄적인 메트릭으로 측정되었습니다.



### Statistical Scenario Modelling and Lookalike Distributions for Multi-Variate AI Risk (https://arxiv.org/abs/2502.14491)
Comments:
          Under review

- **What's New**: 이 연구는 인공지능(AI) 안전성 평가의 격차를 해결하기 위해 두 가지 접근법을 제시합니다. 첫째, 마르코프 체인(Markov chains), 코풀라(copulas), 몬테카를로 시뮬레이션(Monte Carlo simulation) 등으로 기반한 시나리오 모델링(scenario modelling)을 사용하여 AI 위험을 통합적으로 모델링하는 방법을 보여줍니다. 둘째, AI에 대한 직접적인 데이터가 없는 경우 유사 분포(lookalike distributions)를 이용해 AI의 영향을 추정할 수 있음을 입증합니다.

- **Technical Details**: 연구에서는 AI 관련 사건을 하위 사건들로 나누어 다단계 파이프라인을 구성하고, 각 사건의 속성을 측정 가능한 확률 변수로 모델링하는 방법을 설명합니다. 또한, 마르코프 체인과 코풀라를 이용해 순차적 실패 간의 의존성과 위험 지표의 누적 변화를 모델링하는 방법을 제시합니다. 데이터 부족을 해결하기 위해서는 비슷한 프로세스에서 분포를 조정하여 AI 관련 데이터 확보 전까지 근사치를 사용하는 방법을 제안합니다.

- **Performance Highlights**: 이 연구의 방법론은 세 단계의 물류 시뮬레이션 (수요 예측, 창고 픽킹, 마지막 배송)에 적용되어 AI 통합 모델의 다변량 위험을 실용적으로 평가하는 데 유용하다는 것을 보여줍니다. 시나리오 모델링을 통해 AI 사용 유무에 따른 기본 손실 확률과 영향의 차이를 비교할 수 있으며, 이는 실제 산업 현장에서 AI 리스크를 보다 효과적으로 관리할 수 있는 기초를 제공합니다.



### Narrative-Driven Travel Planning: Geoculturally-Grounded Script Generation with Evolutionary Itinerary Optimization (https://arxiv.org/abs/2502.14456)
- **What's New**: 이번 논문에서는 관광객의 경험과 몰입을 향상시키기 위해 NarrativeGuide라는 내러티브 기반의 여행 계획 프레임워크를 제안합니다. 이 프레임워크는 여행자에게 지리문화적 (geoculturally) 내러티브 스크립트를 생성하여 색다른 역할 놀이 경험을 제공합니다. 이를 통해 관광 명소에 대한 더 깊이 있고 통합된 이해를 돕는 것이 목표입니다.

- **Technical Details**: NarrativeGuide는 도시 내 관광 명소를 위한 지식 그래프 (knowledge graph)를 구축합니다. 이후, 이 지식 그래프를 기반으로 세계관 (worldview), 캐릭터 설정 (character setting), 전시 (exposition)를 구성합니다. 여행 계획 단계에서는 내러티브 중심의 여행 계획을 최적화 문제로 모델링하고, 유전 알고리즘 (GA)을 사용하여 최적의 여정을 도출합니다. 이 과정에서 인접 관광지 간의 전환 스크립트를 생성하여 완전한 스크립트를 형성합니다.

- **Performance Highlights**: 네 개의 도시에서 실험을 진행한 결과, 내러티브의 일관성 (narrative coherence) 및 문화적 적합성 (cultural fit)이 유의미하게 향상되었으며, 여행 시간 (travel time)도 감소했습니다. 또한 방문한 관광 명소의 질 (quality of visited attractions) 또한 증가하는 효과를 보여주었습니다. 본 연구는 외부 진화 최적화 기법이 대형 언어 모델의 한계를 효과적으로 극복하는 방법임을 강력히 시사합니다.



### HPS: Hard Preference Sampling for Human Preference Alignmen (https://arxiv.org/abs/2502.14400)
- **What's New**: 이번 연구는 HPS(Hard Preference Sampling)라는 새로운 프레임워크를 소개하여, 인간의 선호에 대한 응답을 더욱 강력하고 효율적으로 정렬할 수 있음을 보여줍니다. 기존의 Plackett-Luce(PL) 및 Bradley-Terry(BT) 모델이 가지고 있던 여러 한계를 극복합니다. 특히 HPS는 훈련 손실을 개선하여 가장 선호되는 응답을 우선시하고 모든 비선호 및 유해 응답을 거부합니다.

- **Technical Details**: HPS는 "hard" 비선호 응답을 강조하여 모델의 거부 능력을 향상시키고, 단일 샘플 몬테 카를로 샘플링 전략을 활용하여 계산 비용을 줄입니다. 기존 PL 방법보다 샘플 효율성을 이론적으로 개선하며, 선호 응답과 비선호 응답 간의 보상 차이를 극대화하여 더 명확한 구분을 보장합니다.

- **Performance Highlights**: 실험 결과 HPS는 HH-RLHF 및 PKU-Safety 데이터셋에서 뛰어난 성능을 보이며, BLEU 및 보상 점수는 비슷하지만 보상 차이가 89% 이상 향상됩니다. 따라서 HPS는 바람직하지 않거나 유해한 콘텐츠 생성을 크게 줄이고, 전반적인 모델의 안전성을 높이는 데 기여합니다.



### Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning (https://arxiv.org/abs/2502.14361)
- **What's New**: 이 논문은 Process Reward Models (PRM)의 주요 아웃 오브 디스트리뷰션(Out-of-Distribution) 문제를 강조하며, 질문 OOD와 단계 OOD를 포함하여 모델 유형(예: GPT, Qwen)과 모델 크기(1.5B, 72B)에 따른 추론 패턴의 차이로 인해 발생하는 문제들을 설명합니다. 이를 해결하기 위해 새로운 프레임워크인 Retrieval-Augmented Process Reward Model (RetrievalPRM)을 소개하고 있습니다.

- **Technical Details**: RetrievalPRM는 두 단계의 검색 강화 메커니즘(Two-stage Retrieval-enhanced Mechanism)을 활용하여, 문제 OOD와 단계 OOD 문제를 해결합니다. 이를 통해서 의미적으로 유사한 질문 및 단계를 선택하여 PRM의 성능을 향상시키고, 다양한 문제 해결 시나리오에서의 일반화 능력을 높입니다. 또한, RetrievalPRM 프레임워크를 사용하여 훈련할 수 있는 검색 강화 데이터셋을 개발하였습니다.

- **Performance Highlights**: 실험 결과, RetrievalPRM은 여러 공개된 실제 데이터셋에서 기존의 강력한 기준 모델 대비 높은 성능을 보여주었습니다. 아울러, Retrieval 접근 방식을 통한 OOD 문제 완화가 입증되었습니다. 이 연구에서 개발한 코드 및 데이터셋은 오픈소스로 공개되어 있어, 다른 연구자들도 확인하고 활용할 수 있습니다.



### FlowAgent: Achieving Compliance and Flexibility for Workflow Agents (https://arxiv.org/abs/2502.14345)
Comments:
          8 pages

- **What's New**: 이번 연구에서는 LLM (대규모 언어 모델) 기반의 에이전트가 워크플로우를 쉽게 통합할 수 있도록 하는 새로운 프레임워크인 FlowAgent를 소개합니다. FlowAgent는 Procedure Description Language (PDL)를 사용하여 자연어와 코드의 유연성을 결합하고, 블랙박스 문제를 해결하여 LLM이 OOW (out-of-workflow) 쿼리를 효과적으로 처리할 수 있도록 합니다. 이는 기존의 정적 규칙 기반 방법과 달리 LLM의 내재된 유연성을 보장하는 데 중점을 두고 있습니다.

- **Technical Details**: FlowAgent는 PDL을 통해 정의된 여러 노드를 바탕으로 에이전트 행동을 제어하는 컨트롤러를 포함합니다. 이는 에이전트가 엄격한 규칙을 지키면서도 자율적으로 결정을 내릴 수 있게 하며, 플로우컨트롤을 통한 OOW 쿼리 관리 능력을 강화합니다. 이 프레임워크는 다양한 테스크를 수행하기 위한 고차원적인 작업관리 시스템으로, LLM의 결정 과정(optimal decision-making process)을 마르코프 결정 프로세스(Markov Decision Process, MDP)로 모델링하여 이해할 수 있습니다.

- **Performance Highlights**: 실험 결과, FlowAgent는 세 가지 데이터셋에서 워크플로우 준수(compliance)와 동시에 OOW 쿼리 처리에 대한 유연성을 효과적으로 보여주었습니다. 이는 기존의 벤치마크가 단순한 플로우 준수를 평가했던 것과는 달리, 보다 포괄적인 평가 방법론을 도입하여 수행되었습니다. FlowAgent는 따라서 대규모 언어 모델의 새로운 잠재력을 실현하며, 다양한 실제 응용 프로그램에 대한 적합성을 높입니다.



### SPRIG: Stackelberg Perception-Reinforcement Learning with Internal Game Dynamics (https://arxiv.org/abs/2502.14264)
Comments:
          To appear in: AAAI 2025 Workshop on Planning and Reinforcement Learning (PRL) - Bridging the Gap Between AI Planning and Reinforcement Learning

- **What's New**: 이번 연구는 SPRIG (Stackelberg Perception-Reinforcement learning with Internal Game dynamics)이라는 프레임워크를 소개합니다. 이 프레임워크는 단일 에이전트 내에서의 인식-정책 상호작용을 협력적 Stackelberg 게임으로 모델링하여 인식 모듈과 정책 모듈의 효과적인 조정을 보장합니다. SPRIG에서는 인식 모듈이 리더 역할을 하여 전략적으로 센서리 데이터를 처리하고, 정책 모듈이 추출된 특성을 기반으로 결정을 내리는 구조입니다.

- **Technical Details**: SPRIG는 수정된 Bellman 연산자를 통해 인식과 정책 간의 상호작용 모델을 제공합니다. 또한, 현대 정책 최적화의 이점을 유지하면서도 수학적으로 엄밀한 결과를 보장합니다. 이 연구에서는 Proximal Policy Optimization (PPO) 알고리즘을 확장하여 두 단계 최적화 과정을 도입하고, 인식 비용 및 효용 함수의 개념을 통해 고유한 고정점으로 수렴하도록 보장합니다.

- **Performance Highlights**: 실험적으로, SPRIG는 Atari BeamRider 환경에서 표준 PPO보다 약 30% 높은 수익을 달성했습니다. SPRIG의 실험 결과는 인식 모듈이 중요한 시각적 특징을 식별하고 추적해야 성공적인 정책 학습이 가능하다는 것을 보여주고 있습니다. 이는 인식-정책 상호작용의 균형을 잘 반영하여 성능을 향상시키는 데 기여하고 있습니다.



### Investigating the Impact of LLM Personality on Cognitive Bias Manifestation in Automated Decision-Making Tasks (https://arxiv.org/abs/2502.14219)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 의사결정 과정에서 인지 편향(cognitive biases)과 개인 성격(personality traits) 간의 상관관계를 조사합니다. 연구 결과는 성격 특성이 인지 편향을 증가시키거나 감소시킬 수 있음을 보여주며, 이는 LLM의 디바이징(debiasing) 전략의 효과에 중요한 영향을 미칩니다. 특히, 성실성(Conscientiousness)과 친화성(Agreeableness) 성격을 가진 LLM이 더 효과적인 편향 완화 전략에 반응하는 경향이 있음을 발견했습니다.

- **Technical Details**: 이 연구는 '빅 파이브'(Big Five) 성격 이론을 바탕으로 하여, 개방성(Openness), 성실성(Conscientiousness), 외향성(Extraversion), 친화성(Agreeableness), 신경증(Neuroticism) 등의 성격 특성이 LLM의 의사결정에서 인지 편향에 어떻게 영향을 미치는지를 분석했습니다. 데이터는 여러 LLM 아키텍처에서 수집되었으며, 특정 성격 특성이 편향의 발생에 미치는 영향을 평가하는 방법ologies를 사용했습니다. 연구의 시각적 요소는 인지 편향의 다양성과 LLM 답변의 성격 프로필을 정리합니다.

- **Performance Highlights**: 연구 결과, 특정 성격 특성이 편향의 발생 및 완화에 중요한 역할을 한다는 것을 발견했습니다. 성실성과 친화성 성격을 가진 LLM은 디바이징 기술에 더욱 긍정적으로 반응하며, 이는 의사결정 과정에서 편향을 줄이는 데 효과적임을 나타냅니다. 이러한 발견은 AI-Assisted Decision-Making의 공정성과 신뢰성을 높이기 위한 맞춤형 편향 완화 접근법의 필요성을 강조합니다.



### Causal Mean Field Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.14200)
- **What's New**: 이번 논문에서는 Multi-Agent Reinforcement Learning (MARL)의 확장성 문제를 해결하기 위한 새로운 알고리즘, Causal Mean-Field Q-learning (CMFQ)을 제안합니다. 기존의 Mean-Field Reinforcement Learning (MFRL) 프레임워크의 한계를 극복하기 위해 인과 추론을 활용하여 에이전트 간의 핵심 상호작용을 식별할 수 있도록 합니다. CMFQ는 에이전트 숫자가 변할 때에도 더욱 강력한 성능을 제공합니다.

- **Technical Details**: CMFQ는 의사결정 과정에서 인과성을 모델링하기 위해 구조적 인과 모델 (Structural Causal Model, SCM)을 도입합니다. SCM을 통해 특정 상호작용의 중요도를 평가하고, 이를 바탕으로 행동 정보를 압축된 형태로 나타내는 인과 인식 정보 표현을 설계합니다. 이 방법은 에이전트 간의 행동을 에이전트의 인과적 효과에 따라 가중치 합으로 모델링하여, 더욱 효과적인 의사결정을 가능하게 합니다.

- **Performance Highlights**: CMFQ는 협력 및 경쟁 게임 환경에서 테스트되었고, 많은 에이전트가 포함된 환경에서 뛰어난 확장성 성능을 보였습니다. 훈련과 실행 모두에서 기존의 방법들과 비교하여 우수한 결과를 기록하며, CMFQ에 의해 통제된 에이전트는 더욱 발전된 집단 지능을 발휘하였습니다. 본 연구는 MFRL에 인과 추론을 도입하는 유망한 방안을 제시합니다.



### Giving AI Personalities Leads to More Human-Like Reasoning (https://arxiv.org/abs/2502.14155)
- **What's New**: 이번 연구는 Large Language Models (LLMs)이 사람의 직관적(System 1) 및 의도적(System 2) 추론 과정을 모두 예측할 수 있는지를 탐구합니다. 기존의 AI 모델이 인간의 복잡한 인지 과정을 반영하지 못한다는 문제를 해결하기 위해, 우리는 personality-based prompting과 genetic algorithms를 결합하여 LLM의 응답 분포를 개선하는 방법을 모색했습니다. 이를 통해 AI가 더 인간적인 추론을 할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 연구에서 제안된 접근법은 Natural Language Inference (NLI) 형식을 일반화하여 새로운 추론 과제를 설계했습니다. 이 과제는 System 1과 System 2의 응답을 유도하기 위해 고안되었습니다. 또 기계 학습 모델과의 비교와 함께, personality traits에 따라 LLM의 응답을 이끌어내는 방법을 적용하여 인간의 추론 다양성을 포착했습니다. 이러한 접근은 기존의 정확성 최적화 방식으로 제한된 AI 모델의 한계를 극복하기 위한 것입니다.

- **Performance Highlights**: 결과적으로, 오픈소스 모델인 Llama와 Mistral은 독점 모델인 GPT보다 더 뛰어난 성능을 보였습니다. personality-based prompting이 LLM의 인간 응답 분포 예측 능력을 크게 향상시켰으며, 이는 다양한 추론 스타일과 심리적 프로파일을 포함하는 모델링 기법이 필요함을 시사합니다. 연구는 이러한 접근법이 AI의 인간스러움을 향상시킬 수 있는 가능성을 보여주고 있습니다.



### Explainable Distributed Constraint Optimization Problems (https://arxiv.org/abs/2502.14102)
- **What's New**: 본 논문에서는 분산 제약 최적화 문제(Distributed Constraint Optimization Problem, DCOP)의 확장인 설명 가능한 DCOP(Explainable DCOP, X-DCOP) 모델을 제안합니다. 기존 DCOP의 해결책이 쉽게 이해되고 수용될 수 있다는 가정을 뒤집고, 설명 가능한 인공지능(Explainable AI)의 문헌을 바탕으로 이 새로운 모델의 필요성을 강조합니다. X-DCOP은 해결책과 그에 대한 반대 쿼리(contrastive query)를 포함하여 사용자의 이해를 돕고 있습니다.

- **Technical Details**: X-DCOP는 유효한 설명(valid explanations)의 정의 및 그 존재에 대한 이론적 결과를 포함하여, 이러한 설명이 충족해야 할 주요 속성(properties)을 형식적으로 정의합니다. 또한, X-DCOP를 해결하기 위한 분산 프레임워크(distributed framework)와 유효한 설명을 찾기 위한 여러 최적화 및 비최적 변형(suboptimal variants)을 제안합니다. 이 구조는 사용자가 솔루션을 이해하기 쉽게 도와주는 것을 목표로 합니다.

- **Performance Highlights**: 실험적 평가를 통해 사용자들은 긴 설명보다 짧은 설명을 선호한다는 사실이 확인되었습니다. X-DCOP 접근법은 대규모 문제에도 확장 가능하였으며, 다양한 변형들은 설명 길이와 실행 시간(runtime) 간의 균형을 맞추는 다양한 옵션을 제공합니다. 이러한 기여는 DCOP 솔루션을 이해하는 장벽을 줄이면서, 현실 세계의 더 많은 애플리케이션에서 채택될 수 있도록 하고 있습니다.



### Investigating Non-Transitivity in LLM-as-a-Judg (https://arxiv.org/abs/2502.14074)
Comments:
          8 pages, 6 figures, 2 tables (30 pages, 11 figures, 8 tables including references and appendices)

- **What's New**: 이 연구에서는 AlpacaEval 프레임워크 내에서 비일관성을 나타내는 비전이성(non-transitivity)의 존재와 그 영향력을 분석하였습니다. LLM(judges)에서 비전이성 선호가 관찰되었으며, 이는 기준 모델의 선택에 민감한 순위를 야기합니다. 이를 해결하기 위해 라운드로빈 토너먼트와 브래들리-테리 모델을 결합하여 보다 신뢰할 수 있는 순위를 생성하는 방법을 제시합니다.

- **Technical Details**: 비전이성은 LLM들이 서로 모순된 선호를 보일 때 발생하며, 이는 특히 고정된 기준 모델을 사용할 경우 순위의 일관성을 해칠 수 있습니다. 새로운 메트릭인 Soft Non-Transitivity Deviation(SNTD)을 도입하여 LLM의 연속 선호에서 비전이성 정도를 측정합니다. 또한, 스위스 기반 반복 매칭(Swim) 토너먼트를 통해 효율성은 유지하면서 라운드로빈 토너먼트의 장점을 활용하는 방법을 제안합니다.

- **Performance Highlights**: 연구 결과, 브래들리-테리 모델을 적용한 라운드로빈 토너먼트가 비전이성의 영향을 감소시켜 보다 견고한 순위를 생성하는 데 기여함을 입증하였습니다. 새로운 방법은 Chatbot Arena와의 상관관계에서 Spearman과 Kendall의 상관계수를 각각 95.0%에서 96.4%, 82.1%에서 86.3%으로 향상시키는 성과를 보였습니다.



### LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention (https://arxiv.org/abs/2502.14866)
Comments:
          Accepted by MLSys 2025. Code available at: this https URL

- **What's New**: 이번 논문에서는 LServe라는 효율적인 시스템을 소개합니다. LServe는 하이브리드 희소_attention을 활용하여 긴 시퀀스 모델의 처리 속도를 높입니다. 이 방법은 프리필링(pre-filling)과 디코딩(decoding) 단계에서 하드웨어 친화적인 다양한 구조의 희소 패턴을 통합하여, 덜 중요한 토큰에 대한 연산을 차단(block-wise)하여 컴퓨팅을 최적화합니다. LServe는 정적(static) 및 동적(dynamic) 희소성의 호환성을 보여주며, 각 최적화를 결합하여 배수의 속도 향상을 가능하게 합니다.

- **Technical Details**: LServe의 핵심은 토큰의 중요성에 따라 블록 레벨의 희소성을 활용하여 KV 히스토리를 처리하는 방식입니다. 이 시스템은 프리필링 단계와 디코딩 단계에서 각각 50%의 attention heads를 거의 무료로 사용할 수 있도록 변환합니다. 또한, KV 페이지 선택 정책을 설계하여 쿼리 중심 유사성을 기반으로 KV 페이지를 동적으로 잘라내는 방법도 채택하였고, 이를 통해 인풋의 길이에 상관없이 일정한 수의 KV 페이지만으로 긴 시퀀스를 유지할 수 있습니다.

- **Performance Highlights**: LServe는 세 가지 긴 시퀀스 LLM(Llama-3-8B, Minitron-4B, Llama-2-7B)에서 벤치마크 테스트를 수행하였으며, 최고 512k 토큰의 컨텍스트 길이를 지원합니다. vLLM 등 최신 기술들과 비교했을 때, LServe는 프리필링 단계에서 최대 2.9배, 디코딩 단계에서 평균 1.3배에서 2.1배의 속도 향상을 기록했습니다. 이러한 성능 향상은 기존의 밀집 밀리 초 모델의 긴 컨텍스트 처리 능력을 유지하면서 이루어졌습니다.



### Interpretable Text Embeddings and Text Similarity Explanation: A Primer (https://arxiv.org/abs/2502.14862)
- **What's New**: 이번 논문에서는 텍스트 임베딩(text embedding) 및 이와 관련된 유사성 점수를 해석하기 위한 방법론의 구조적 개요를 제공합니다. 유사성 점수의 해석 가능성 문제는 AI와 NLP 시스템에서 투명성이 요구되는 환경에서 중요한 과제로, 이를 해결하기 위한 최신 연구 동향을 다루고 있습니다. 다양한 텍스트 임베딩 모델들이 갖는 해석 가능성을 강화하기 위해 개별 방법들의 아이디어와 기술을 평가합니다.

- **Technical Details**: 신경망 텍스트 임베딩 및 유사성의 설명 가능성을 연구하며, 이는 기존의 분류 설명 방식과 구별됩니다. 두 입력의 상호작용을 기반으로 유사성이 결정되므로, 이를 위한 전문화된 방법이 필요합니다. 주어진 텍스트 입력을 통해 신경망을 통한 계산 단계가 설명되며, 일반적으로 Siamese 네트워크(서로의 가중치를 공유) 방식을 사용하여 텍스트 임베딩을 생성합니다.

- **Performance Highlights**: 텍스트 임베딩을 통해 얻은 유사성 점수는 문서 간의 유사성을 정량적으로 평가할 수 있는 중요한 척도를 제공합니다. 이러한 점수는 간단한 내적(dot product) 계산을 통해 산출되며, 이러한 방식은 실질적으로 강하게 상관된 코사인 유사성(cosine similarity)을 활용합니다. 논문은 텍스트 임베딩 및 유사성 정의를 통해 서로 다른 유사성 계산 방법을 구분할 수 있는 기초를 마련하고 있습니다.



### FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling (https://arxiv.org/abs/2502.14856)
- **What's New**: FR-Spec은 대형 언어 모델(LLM)에서 생기는 발생 속도 저하를 해결하기 위한 새로운 방법이다. 이 프레임워크는 단일 레이어와 언어 모델 헤드를 사용하는 기존 방법(EAGLE-2 등)에 비해 75%의 계산 오버헤드를 줄이면서도 최종 출력 분포의 동등성을 유지한다. FR-Spec은 고빈도 토큰으로 구성된 서브셋으로 드래프트 후보를 최적화하여 속도를 1.12배 향상시킬 수 있다.

- **Technical Details**: FR-Spec에서는 이론적으로 언어 모델의 드래프트 검색을 고빈도 토큰으로 제한하여, LM 헤드의 계산 비용을 크게 줄인다. 이 방법은 기존의 언어 모델 샘플링 기법과 호환되며, 재훈련이 필요 없는 플러그 앤 플레이 설계를 채택하고 있다. 이를 통해 EAGLE-2와 통합 시 1.12배의 속도 향상을 달성하며, Medusa와 통합 시에는 1.08배의 향상을 기록했다.

- **Performance Highlights**: 실험 결과, FR-Spec은 다양한 데이터셋에서 기존의 최첨단 방법인 EAGLE-2보다 평균 1.12배의 속도 향상을 보였다. 특히, 전통적인 드래프트-검증 메커니즘에 비해 FR-Spec의 효율성은 명백하다. 이러한 성능 향상은 대규모 어휘의 잠재적 문제를 극복하는 데에도 기여하고 있으며, LLM의 활용 가능성을 더욱 넓히고 있다.



### Revealing and Mitigating Over-Attention in Knowledge Editing (https://arxiv.org/abs/2502.14838)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 지식 편집 방법에서 발생하는 특정성 실패(Specificity Failure) 문제를 다룹니다. 연구자들은 LLMs에서 수정을 통해 기존 지식이 손상되는 현상을 발견하고, 이를 Attention Drift라는 현상으로 정의하였습니다. Attention Drift는 모델이 불필요하게 새로운 지식에 집중하여 이전 지식을 왜곡하는 문제입니다. 이를 해결하기 위해, Selective Attention Drift Restriction(SADR)라는 새로운 방법을 제안합니다.

- **Technical Details**: SADR은 지식 편집 과정에서 주의 가중치 분포의 변화를 제한하는 추가 정규화 항을 도입하여 지나치게 수정된 정보에 대한 집중을 방지합니다. 연구에서 사용한 모델은 1.1B에서 20B까지의 다양한 LLM으로, SADR 방법을 적용함으로써 기존 지식의 손상을 크게 줄였습니다. 세 가지 지식 편집 방식(위치-수정, 매개변수 보존, 메타 학습)을 포함하여, SADR은 명백히 훈련된 모델의 예측 성능을 향상시켰습니다. 이 때 원래 지식의 정확도가 절반 이상 감소하는 현상을 줄이는 데 효과적입니다.

- **Performance Highlights**: SADR 방법을 적용한 결과, 다섯 개의 모델에서 130.9%에서 295.8%까지의 정확도 향상을 보였으며, 편집 성공률은 단 0.19% 감소하는 데 그쳤습니다. 이는 SADR이 다양한 편집 작업에서 효과적으로 특정성 실패를 완화할 수 있음을 입증합니다. 이러한 결과는 실세계 응용에서 모델의 신뢰성과 강건성을 높이는 데 기여할 것으로 기대됩니다.



### Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs (https://arxiv.org/abs/2502.14837)
Comments:
          16 pages, 8 figures

- **What's New**: DeepSeek에서 제안한 Multi-head Latent Attention (MLA) 아키텍처는 Key-Value (KV) 캐시를 압축하여 효율적인 추론을 가능하게 하고 있습니다. 기존의 Multi-Head Attention (MHA) 및 그 변형과 비교하여 MHA2MLA로의 전환을 통해 데이터 효율적인 파인 튜닝을 가능하게 하는 최초의 방법이 제시되었습니다. 이 접근법은 RoPE를 부분적으로 제거하고 저랭크 근사를 도입하여 성능을 회복합니다.

- **Technical Details**: MLA 구조는 수많은 파라미터를 재사용하면서 MHA에서 MLA로의 전환을 용이하게 합니다. 핵심 기술로는 부분 회전 위치 임베딩(partial RoPE)과 저랭크 근사(low-rank approximation)가 있으며, 이는 KV 캐시와 추론 프로세스의 저장 방식을 MLA에 맞춰 조정합니다. 이 방법은 기존 MHA의 학습된 파라미터를 최대한 활용하기 위한 설계를 포함하고 있습니다.

- **Performance Highlights**: MHA2MLA는 훈련 데이터의 0.3%에서 0.6%만으로도 성능을 복구하는 데 성공하였습니다. KV 캐시 크기를 92.19% 줄이며 LongBench 성능은 단 0.5% 하락에 그쳤습니다. 실험에서는 다양한 모델 크기에 대해 MHA2MLA의 효과를 검증하여 효율적인 추론을 위한 통찰을 제공합니다.



### LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models (https://arxiv.org/abs/2502.14834)
- **What's New**: 본 논문에서는 기존의 Large Vision-Language Models (LVLMs)가 최대 128k의 비주얼 및 텍스트 토큰을 처리할 수 있지만, 1,000단어를 초과하는 일관된 출력을 생성하는 데 어려움을 겪는 문제를 다룹니다. 이를 해결하기 위해 LongWriter-V-22k라는 새로운 SFT(Supervised Fine-Tuning) 데이터셋을 소개하는데, 이 데이터셋은 22,158개의 예시를 포함하고 있으며, 각각 여러 입력 이미지, 지침, 최대 10,000단어 범위의 출력이 포함되어 있습니다.

- **Technical Details**: 우리는 입력 이미지에 대한 높은 충실도를 유지하면서 긴 출력을 달성하기 위해 Direct Preference Optimization (DPO) 방법론을 활용합니다. 길고 복잡한 출력을 위한 인간 피드백을 수집하는 것이 비용이 많이 드는 점을 고려하여, IterDPO라는 접근 방식을 제안합니다. 이 방법은 긴 출출을 세분화하여 원래 출력과의 선호 쌍을 형성하기 위한 반복적 수정(iterative corrections)을 사용합니다.

- **Performance Highlights**: LongWriter-V-22k와 IterDPO로 훈련된 7B 파라미터 모델은 MMLongBench-Write 벤치마크에서 우수한 성과를 보여주며, GPT-4o와 같은 더 큰 상용 모델들을 초월하는 결과를 냈습니다. 이는 긴 생성능력의 평가에서 LVLM의 새로운 가능성을 시사합니다.



### Improving the Diffusability of Autoencoders (https://arxiv.org/abs/2502.14831)
Comments:
          26 pages, 22 figures, 9 tables

- **What's New**: 이번 연구에서는 Latent Diffusion Models (LDMs)의 주요 요소인 autoencoders와 diffusion backbones 간의 상호작용에 대해 깊이 있는 분석을 진행했습니다. 특히, 고주파 성분의 비정상적 존재가 diffusion 합성 과정에 방해가 된다는 가설을 세웠습니다. 이를 해결하기 위해 scale equivariance라는 정규화 전략을 제안함으로써 latent 및 RGB 공간의 주파수 정렬을 도모하였습니다.

- **Technical Details**: 연구에서 제안한 scale equivariance는 decoder에서 스케일 동등성을 강제하여 downsampled latent가 downsampled RGB 표현에 일치하도록 보장합니다. 이 방법은 autoencoder의 미세 조정을 20K 단계로 제한하며, minimal code 변경만으로도 효과적인 결과를 도출합니다. 이 과정에서 고주파 성분이 최종 RGB 결과에 미치는 영향을 분석하여, 기존의 KL 정규화가 충분하지 않음을 밝혔습니다.

- **Performance Highlights**: 제안한 방법은 ImageNet-1K 데이터셋에서 이미지 생성의 FID를 19% 감소시켰고, Kinetics-700 데이터셋에서도 비디오 생성의 FVD를 최소 44% 향상시켰습니다. 이는 다양한 아키텍처의 diffusability를 개선하여 생성 품질을 유의미하게 향상시키는 결과를 나타냅니다.



### Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs (https://arxiv.org/abs/2502.14830)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 중간 계층이 다양한 언어 간의 정렬에 가장 강력한 잠재력을 가지고 있다는 발견을 기반으로, 중간 계층 정렬 목표(middle-layer alignment objective)를 제안합니다. 이를 통해 언어 자원이 적은 언어에서의 작업의 성능 향상을 목표로 하며, 기존 작업 특정 훈련(task-specific training)과 결합하여 실험적인 성과를 거두었습니다.

- **Technical Details**: 중간 계층에서의 정렬 목표를 적용하는 방법은 기계번역, 슬롯 채우기(slot filling) 및 구조화된 텍스트 생성(structured text generation)와 같은 다양한 작업에서 사용됩니다. 이 방법은 한국어 같은 언어 자원이 적은 언어뿐만 아니라 미정렬 언어(unseen languages)에도 일반화될 수 있습니다. 또한, 별도로 훈련된 정렬 모듈은 기존의 작업 특정 모듈과 결합할 수 있어 전체 재훈련 없이도 크로스-링구얼 기능을 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, 중간 계층에서의 정렬 목표를 적용하는 것이 LLM의 작업 특정 훈련 동안 크로스-링구얼 전이(cross-lingual transfer)를 개선한다는 것을 보여주었습니다. 또한, 이러한 성과가 정렬에 사용된 언어의 선택에 강인하며, 다양한 언어 쌍 간의 전이 성능에도 긍정적인 영향을 미친다는 것을 확인했습니다. 마지막으로, 새로운 훈련 전략이 LLM의 크로스-링구얼 능력을 크게 향상시킬 가능성을 제시합니다.



### Exploring Advanced Techniques for Visual Question Answering: A Comprehensive Comparison (https://arxiv.org/abs/2502.14827)
Comments:
          8 pages, No figures

- **What's New**: 이번 논문은 시각 질문 응답(Visual Question Answering, VQA) 분야에서 최신 기술 models에 대한 포괄적인 비교 연구를 제공합니다. 연구는 ABC-CNN, KICNLE, Masked Vision and Language Modeling, BLIP-2, OFA 등 다섯 가지 모델의 접근 방식을 분석하여 VQA의 새로운 발전 방향을 체계적으로 제시하고 있습니다.

- **Technical Details**: VQA는 컴퓨터 비전(computer vision)과 자연어 처리(natural language processing) 의 교차점에서 중요한 작업으로, 모델이 자연어 질문에 대한 시각적 내용에 대한 이해와 사고를 요합니다. 이 연구에서는 각 모델이 질문 다양성(question diversity), 답변 분포(answer distribution), 시각-텍스트 상관관계(visual-textual correlations) 등에서 어떤 차별적 접근을 하고 있는지 살펴봅니다.

- **Performance Highlights**: 기존 VQA 모델들은 데이터셋 편향(dataset bias), 제한된 모델 복잡성(limited model complexity), 상식 추론(gaps in commonsense reasoning), 경직된 평가 방법(rigid evaluation methods) 및 실제 시나리오에 대한 일반화(generalization)와 같은 문제에 직면해 있습니다. 이 논문은 이러한 문제를 해결하기 위한 다섯 가지 고급 모델의 성능을 세밀하게 비교하여 VQA의 한계점과 발전 가능성을 강조하고 있습니다.



### eC-Tab2Text: Aspect-Based Text Generation from e-Commerce Product Tables (https://arxiv.org/abs/2502.14820)
Comments:
          NAACL 2025 (Industry Track)

- **What's New**: 이 논문에서는 eC-Tab2Text라는 혁신적인 데이터세트를 소개합니다. 이 데이터세트는 전자상거래(e-commerce) 도메인에서 제품 속성과 사용자 쿼리를 포함하는 다양한 요소를 포착하기 위해 설계되었습니다. 이를 통해 LLMs(대규모 언어 모델)가 구조화된 표(tabular data)를 기반으로 고품질의 제품 리뷰를 생성할 수 있도록 지원합니다.

- **Technical Details**: eC-Tab2Text는 전자상거래 제품 테이블과 사용자 별 쿼리, 출력이 연계된 속성 중심의 데이터세트를 제공합니다. 기존의 일반 목적의 데이터세트와는 달리, 이 데이터세트는 전자상거래에서의 고유한 요구 사항에 맞춰 설계되었습니다. 이를 통해 LLM을 미세 조정하고, 표준 Table2Text 메트릭을 활용하여 검증함으로써 성능을 평가합니다.

- **Performance Highlights**: 결과적으로, 모델의 수정을 통해 생성된 제품 리뷰의 맥락적 정확성이 크게 향상된 것으로 나타났습니다. eC-Tab2Text 데이터세트는 LLMs의 성능을 극대화하고, 전자상거래 작업흐름을 최적화하는 데 중요한 역할을 하고 있습니다. 이는 고객만족도와 비즈니스 결과를 높이는 데 기여할 것으로 기대됩니다.



### FetalCLIP: A Visual-Language Foundation Model for Fetal Ultrasound Image Analysis (https://arxiv.org/abs/2502.14807)
- **What's New**: FetalCLIP는 태아 초음파 이미지를 분석하기 위해 설계된 비전-언어 (vision-language) 기초 모델로, 대규모의 태아 초음파 이미지와 캡션의 커플 데이터셋을 활용하여 개발되었습니다. 이 모델은 태아 초음파 영상의 복잡한 해부학적 특징을 효과적으로 학습하여 다양한 다운스트림 (downstream) 작업에 활용할 수 있는 고급 표현을 생성하는 능력을 가지고 있습니다. 기존 모델의 한계를 극복하기 위해 FetalCLIP는 대규모 멀티모달 학습을 통해 훈련되었으며, 이 과정에서 210,035개의 이미지와 텍스트 쌍을 사용했습니다.

- **Technical Details**: FetalCLIP는 기존의 CLIP 프레임워크를 바탕으로 단일 이미지와 그에 대한 텍스트 설명을 정렬하여 훈련되었습니다. 이 모델은 비전 트랜스포머(ViT) 기반 이미지 인코더와 텍스트 인코더를 통합하여 구성을 이루며, 최대 117자의 토큰을 처리할 수 있습니다. 훈련 데이터셋에는 규칙적인 임상 초음파 이미지가 포함되어 있으며, 각 이미지는 GPT-4o에 의해 생성된 적절한 텍스트 설명과 함께 제공됩니다.

- **Performance Highlights**: 다양한 태아 초음파 분석 작업에서 FetalCLIP는 기존 모델들에 비해 우수한 성능을 보였습니다. FetalCLIP는 제로샷 분류 작업에서 87.1%의 F1 점수를 달성하며, 이는 기존의 SonoNet 모델에 비해 17.2% 더 높은 성과입니다. 또한, 선천성 심장 결함(CHD) 탐지 작업에서도 기존 모델 대비 6.92% 높은 AUC 성능을 나타내며, 태아 해부학적 구조 세분화에서는 평균 Dice 유사성 계수(DSC) 84.22%의 성과를 기록, FetalCLIP는 AI 기반 태아 초음파 진단의 중요한 발전을 대표합니다.



### From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (https://arxiv.org/abs/2502.14802)
Comments:
          Code and data to be released at: this https URL

- **What's New**: 이번 연구는 새로운 RAG 접근법인 HippoRAG 2를 제안하여, 사실적 기억(factual memory), 의사결정(sense-making), 그리고 연관성 기억(associative memory) 과제에서 표준 RAG보다 우수한 성능을 보여줍니다. HippoRAG 2는 Personalized PageRank 알고리즘을 든든히 지탱하며, LLM의 온라인 사용을 더욱 효과적으로 통합하여 인간의 장기 기억에 가까운 효과를 구현합니다. 이를 통해 단순한 기억 과제에서도 강화된 성능을 가져오며, 비모수적 지속 학습을 위한 새로운 길을 제시합니다.

- **Technical Details**: HippoRAG 2는 기존 HippoRAG의 구조를 바탕으로 하여, 더 깊은 구문 통합(deeper passage integration)과 LLM 사용의 효과성을 향상시킵니다. Personalized PageRank를 기반으로 하여 쿼리에 기반한 맥락화(contextualization) 부족을 보완하기 위해 KG(Knowledge Graph) 삼중(triples) 선택 과정에 더 깊이 참여하게 만들어, LLM의 효율성을 극대화합니다. 이러한 요소들이 긴 이야기와 같은 복잡한 마케팅을 이해할 수 있는 능력을 크게 향상시킵니다.

- **Performance Highlights**: HippoRAG 2는 관계성 관련 작업에서 표준 RAG보다 평균 7777 포인트 개선된 성과를 보였으며, 사실 기억 및 의사결정 작업에서 성능 저하 없이 소폭의 향상을 기록하였습니다. 이 시스템은 다양한 리트리버와 강력한 공개형 및 상용 LLM에서도 견고성을 보이며 적용 유연성을 극대화합니다. 이러한 성과는 HippoRAG 2가 인간과 유사한 비모수적 지속 학습 시스템으로 발전할 수 있는 가능성을 제시합니다.



### A Survey on Text-Driven 360-Degree Panorama Generation (https://arxiv.org/abs/2502.14799)
- **What's New**: 이 논문은 텍스트 기반의 360도 파노라마 생성 기술이 어떻게 발전했는지를 다룹니다. 특히, 텍스트 설명에서 직접 360도 파노라마 이미지를 합성하는 혁신적 방법을 소개합니다. 종전의 복잡한 생성 과정을 단순화시킴으로써 VR/AR, 게임 및 가상 투어와 같은 다양한 분야에서의 콘텐츠 생성 방식에 혁신을 일으킬 것으로 기대됩니다. 또한, 일련의 최첨단 텍스트 기반 알고리즘과 이들의 응용법을 면밀히 분석합니다.

- **Technical Details**: 정보의 정확한 생성과 기하학적 일관성 유지를 강조하는 360도 파노라마 생성에는 고유한 기술적 도전이 따릅니다. 최근 텍스트-이미지 확산 모델(text-to-image diffusion models)의 발전이 이를 가능하게 하여, 고품질의 이미지를 생성하는 다양한 방법론이 제시되었습니다. 이 연구에서는 텍스트-주도 생성(text-driven generation)과 텍스트-주도 틈새 시야(NFOV) 아웃페인팅 방법을 포함한 두 가지 주요 패러다임을 소개하고, 각 방법의 장단점을 비교합니다.

- **Performance Highlights**: 연구 결과에 따르면, 텍스트-이미지 확산 모델을 사용하여 생성된 360도 파노라마는 전통적인 방법보다 훨씬 높은 품질의 시각적 결과를 제공합니다. 특히, 신규 응용 프로그램으로서 텍스트 기반으로 360도 3D 장면 생성을 지원하는 가능성을 보여줍니다. 현재의 기술적 한계를 극복하기 위한 연구 방향 또한 제안되어, 이 분야의 미래 발전에 중요한 기초가 될 것으로 예상됩니다.



### Rapid Word Learning Through Meta In-Context Learning (https://arxiv.org/abs/2502.14791)
- **What's New**: 이번 연구에서 제안된 Minnow(Meta-training for IN-context learNing Of Words) 방법은 언어 모델이 몇 개의 예시를 기반으로 새로운 단어의 사용법을 생성하도록 훈련할 수 있게 합니다. 특히, 이 방법은 새로운 단어를 나타내기 위해 특별한 placeholder token을 사용하며, 이러한 과정이 반복될수록 모델의 일반적인 단어 학습 능력이 향상된다고 주장합니다. 이 연구는 아동 언어 습득에 맞춘 데이터셋을 사용하여, 소량의 데이터에서도 변별력 있는 단어 학습이 가능함을 보여줍니다.

- **Technical Details**: Minnow 방법은 언어 모델이 새로운 단어의 맥락을 통해 빠르게 학습할 수 있도록 meta-training을 이용합니다. 이 방법은 autoregressive 언어 모델을 한 단계에서 훈련하여 새로운 단어의 예시를 생성하도록 하며, 각 단어에 대한 사용 예시가 없어도 일반화할 수 있는 능력을 개발합니다. 또한, 이 연구는 기존의 LLM(large language model)이 아니라 아동의 언어 입력 데이터를 기반으로 모델을 훈련함으로써, 데이터 효율성을 극대화하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 연구 결과, Minnow의 성능은 기존의 대규모 언어 모델과 유사한 수준으로, 소량의 예시를 활용하여 새로운 단어를 효율적으로 분별하고 정의할 수 있는 능력이 향상되었습니다. 특히, Llama-3 8B 모델을 Minnow로 미세 조정(finetuning)한 결과, 새로운 단어의 문법적 범주를 식별하고, 관련 있는 정의 및 사용 예시를 생성하는 능력이 크게 향상된 것을 확인했습니다. 이 접근 방식은 언어 모델의 단어 학습 능력을 개선하는 잠재력을 보여줍니다.



### Ray-Tracing for Conditionally Activated Neural Networks (https://arxiv.org/abs/2502.14788)
Comments:
          submitted to workshop

- **What's New**: 본 논문에서는 여러 Mixture of Experts (MoEs) 층의 계층적 구조와 최적의 전문가 활성화 구성으로 수렴하는 샘플링 메커니즘을 결합한 새로운 신경망 아키텍처를 소개합니다. 이러한 방법론은 네트워크 아키텍처의 동적 전개를 가능하게 하여 효율적인 경로 전용 훈련을 촉진합니다. 실험 결과, 이 접근법은 기존 기준과 비교하여 경쟁력 있는 정확도를 달성하면서도 추론에 필요한 매개변수 수를 크게 줄이는 데 성공하였습니다. 특히, 이 매개변수 감소는 입력 패턴의 복잡성과 상관관계가 있으며, 이는 네트워크의 작동 동적에서 자연스럽게 발생하는 특성입니다.

- **Technical Details**: Mixture of Experts (MoEs) 접근법은 큰 모델에서 Computational Load(계산 작업량)을 줄이기 위해 조건부 활성화를 활용하는 표준 방법이 되었으나, 일반적으로 한 레이어 내의 사전 설정된 큰 블록 수에 제한됩니다. 제안된 네트워크는 여러 레이어에 블록이 쌓여 있어 입력에 따라 블록이 선택적으로 활성화될 수 있도록 합니다. 각 블록의 출력을 확률적 계산 경로의 기대 발화율로 표현함으로써 추론과 선택적 활성화 문제를 동시에 해결할 수 있는 가능성을 제시하였습니다. 이 구조는 순차적 해법을 제공하여 더 어렵고 복잡한 추론 문제가 요구하는 계산 자원과 시간을 동적으로 조절합니다.

- **Performance Highlights**: 제안하는 RayTracing 모델은 네트워크 내에서 독립적으로 블록을 활성화시키며, 이를 통해 깊이 우선 계산과 넓이 우선 접근 방식을 균형 있게 조율하고자 합니다. 각 블록의 출력은 Softmax 비선형성을 사용하여 게이팅되어 효과적으로 활성화되며, 이는 시간 민감한 결정에 특히 유용합니다. 속성을 통해 초기 접근은 빠르지만, 점진적으로 더 정확한 해결책으로 발전할 수 있으며, 이는 자율주행차 및 로봇 공학과 같은 응용 분야에서 중요한 요소가 될 것입니다.



### SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features (https://arxiv.org/abs/2502.14786)
Comments:
          Model checkpoints are available at this https URL

- **What's New**: SigLIP 2는 멀티링구얼 비전-언어 인코더의 새로운 가족으로, 기존 SigLIP의 성공을 기반으로 확장되었습니다. 이 모델은 캡셔닝 기반의 프리트레이닝, 셀프-슈퍼바이즈드 손실(self-supervised losses), 온라인 데이터 큐레이션과 같은 여러 기술을 통합한 훈련 레시피를 적용하여 기존 모델들을 초월하는 성능을 보여줍니다. 특히, 멀티링구얼 이해와 공정성을 향상시키기 위해 더 다양하고 균형 잡힌 데이터 혼합으로 학습되었습니다.

- **Technical Details**: SigLIP 2는 SigLIP의 훈련 레시피와 디코더 기반 프리트레이닝을 결합하고, 다양한 해상도를 지원하는 모델 변형을 포함합니다. 추가로 성능 최적화를 위해 액티브 샘플 선택(active sample selection)을 통한 암묵적 증류(implicit distillation) 기법을 활용하여 작은 모델의 질을 향상시켰습니다. 이미지와 텍스트 인코더 모두 같은 아키텍처를 사용하며, 다국어 Gemma 토크나이저를 채택하여 텍스트를 처리하는 방식도 개선되었습니다.

- **Performance Highlights**: SigLIP 2는 이미지-텍스트 분류, 검색, VLM 기능 추출에서 모든 모델 스케일에서 기존 SigLIP보다 우수한 성능을 발휘합니다. 특히, 지역화(localization)와 밀집 예측(dense prediction) 작업에서도 성능이 크게 향상되었습니다. 사용자는 성능과 추론 비용을 조절할 수 있도록 4가지 모델 사이즈를 제공받아 다양한 애플리케이션에 활용할 수 있습니다.



### Real-Time Device Reach Forecasting Using HLL and MinHash Data Sketches (https://arxiv.org/abs/2502.14785)
- **What's New**: 본 논문에서는 사용자 지정 타겟팅 속성에 따라 실시간으로 TV의 수(기기 도달 수)를 예측하는 새로운 시스템을 소개합니다. 기존의 SQL 쿼리 방식은 수십억 개의 레코드를 조인하는 데 매우 느려서 비즈니스 기회 손실로 이어졌습니다. 이를 해결하기 위해 MinHash와 HyperLogLog (HLL) 데이터를 활용한 최신 예측 시스템을 구축하였습니다.

- **Technical Details**: 새로운 시스템에서는 요청이 발생할 때 실시간으로 기기 도달 수를 계산하기 위한 실시간 예측 시스템을 사용합니다. 특히 기존의 MinHash 구현이 가진 다단계 집계 및 교차(intersection) 문제를 해결했습니다. 또한, 우리의 MinHash 알고리즘을 개선하여 SIMD (Single Instruction Multiple Data) 벡터 연산을 사용해 4배 더 빠른 속도로 실행되도록 하였으며, 상수 공간을 유지하면서 수십억 개의 레코드를 처리할 수 있습니다.

- **Performance Highlights**: 실험을 통해 새로 개발한 예측 시스템의 결과가 전통적인 오프라인 예측 시스템과 동일한 정확도를 가지며, 허용 가능한 오류율 5%를 유지함을 보였습니다. 이러한 성과는 실시간으로 비즈니스를 운영할 수 있도록 도와주는 중요한 기초가 될 것입니다.



### ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting (https://arxiv.org/abs/2502.14780)
Comments:
          12 pages, 7 figures, 3 tables

- **What's New**: 이 논문에서는 AR(증강현실), VR(가상현실) 및 최신 스마트폰 상에서의 효율적이고 프라이버시를 보호하는 다중모드(interaction)에 대한 새로운 접근법인 Visual Instruction Rewriting을 제안합니다. 기존의 대규모 비전-언어 모델(VLM)은 클라우드 기반 처리를 의존하여 시각 데이터의 프라이버시 문제와 실시간 처리의 어려움을 초래했습니다. 본 연구는 39,000개의 예제와 14개 도메인으로 구성된 데이터셋을 제시하며, 이를 통해 프라이버시를 보장하면서 다중모드를 처리할 수 있는 경량화된 VLM(250M parameters)을 개발했습니다.

- **Technical Details**: Visual Instruction Rewriting은 다중모드 지시사항을 텍스트만 포함된 명령으로 변환합니다. 이 접근법은 프라이버시를 보호하면서도 사용자가 기기에서 직접 명령을 실행할 수 있도록 합니다. 본 연구는 여러 NLG(자연어 생성) 메트릭(BLEU, METEOR, ROUGE)을 이용하여 제안한 모델의 성능을 평가하였으며, 8비트 양자화된 모델은 500MB 미만의 저장 공간에서 효과적인 지시 사항 변환을 수행했습니다.

- **Performance Highlights**: 우리는 경량화된 250M 파라미터 모델이 기존의 베이스라인인 PaliGemma-v2 및 Qwen2VL에 비해 제로샷 설정에서 더 나은 성능을 보였음을 발견했습니다. 이 모델은 사용자 요청을 구조화된 텍스트로 변환할 수 있는 수용 가능한 수준의 rewriting 능력을 가지며, 이는 AR, VR 및 스마트폰 인터페이스와의 안전하고 실시간 상호작용을 가능하게 합니다.



### Harnessing PDF Data for Improving Japanese Large Multimodal Models (https://arxiv.org/abs/2502.14778)
Comments:
          15 pages, 8 figures

- **What's New**: 본 연구에서는 일본어 LMM(Large Multimodal Models)의 성능을 향상시키기 위해 PDF 데이터의 잠재력을 탐색합니다. 기존의 일본어 LMM은 영어로 번역된 데이터에 의존해 일본 특유의 문화적 지식을 캡처하는 데 한계가 있었습니다. 이를 해결하기 위해, PDF에서 이미지-텍스트 쌍을 추출하는 완전 자동화된 파이프라인을 소개하며, 외부 주석 작업 없이 데이터를 확보할 수 있는 방법을 모색했습니다.

- **Technical Details**: 우리는 사전 훈련된 모델을 활용하여 PDF에서 이미지-텍스트 쌍을 추출하는 방법론을 개발했습니다. 이 과정에서는 레이아웃 분석(layout analysis), OCR(Optical Character Recognition), 및 비전-언어 페어링(vision-language pairing) 기술이 포함됩니다. 이러한 자동화된 시스템을 통해 일본어 LMM의 훈련 데이터를 풍부하게 하기 위한 지침 데이터도 생성됩니다.

- **Performance Highlights**: PDF에서 유래된 데이터로 훈련한 일본어 LMM은 Heron-Bench에서 3.9%에서 13.8%의 성능 향상을 보였습니다. 다양한 실험을 통해 PDF 데이터의 효과와 모델 크기와의 관계를 분석하였으며, 이미지-텍스트 쌍과 이미지만으로 생성된 지침 데이터의 효과성도 평가했습니다. 이러한 연구 결과는 일본어 LMM의 훈련에 PDF 데이터를 활용하는 가치를 확고히 해줍니다.



### Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning (https://arxiv.org/abs/2502.14768)
- **What's New**: 이 논문에서는 DeepSeek-R1에서 성공적으로 사용된 규칙 기반 강화 학습(RL)의 잠재력을 대규모 추론 모델에 탐구합니다. 주목할 만한 점은 5,000개의 논리 문제만으로도 모델이 복잡한 수학 벤치마크인 AIME와 AMC에서 일반화 능력을 발휘하였다는 것입니다. 이 연구는 기존 연구의 재현 가능성을 높이고, 소규모 모델에서도 비슷한 추론 능력이 발휘될 수 있는지를 동적으로 검증하고 있습니다.

- **Technical Details**: 제안된 Logic-RL 프레임워크는 REINFORCE++ 알고리즘을 채택하고 DeepSeek-R1의 보상 설계를 기반으로 하여 훈련됩니다. 강화 학습 과정에서 모델은 더 많은 훈련 단계를 할당하여 더 깊이 있는 사고 과정을 탐색하게 됩니다. 또한 <think>와 <answer> 태그를 포함한 구조화된 응답 포맷을 통해 보상을 정확하게 검증할 수 있게 되어, 강화학습 접근 방식에서 아키텍처의 전반적인 개선을 이끌어냅니다.

- **Performance Highlights**: 7B 모델은 AIME에서 125%, AMC에서 38% 향상을 보였으며, 이는 모델이 추상적인 문제 해결 체계를 발전시킨다는 것을 보여줍니다. 강화 학습이 훈련 데이터 구조에 최소한으로 의존하는 방식으로 자연스럽게 일반화를 성공적으로 이루어내며, 더 긴 응답이 항상 더 나은 추론을 보장하지 않는다는 흥미로운 발견도 있었습니다.



### Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis (https://arxiv.org/abs/2502.14767)
Comments:
          Code available at: this https URL

- **What's New**: 이 논문은 Tree-of-Debate (ToD)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 과학 논문을 각각의 LLM(대형 언어 모델) 페르소나로 변환하여 그들의 혁신성을 두고 토론하게 합니다. ToD는 구체적이고 비판적인 추론을 강조하며, 정치적이고 학문적인 논의를 촉진합니다.

- **Technical Details**: ToD는 복잡한 논리적 과제를 해결하기 위해 다양한 관점과 추론 경로를 탐색하는 다중 에이전트 LLM 토론을 활용합니다. 논문 간의 독립적인 혁신성 주장을 구분할 수 있는 비계층적 대화 구조를 동적으로 구성하며, 각 논점의 세부사항을 분석할 수 있게 합니다. 또한, 이 프레임워크는 반복적인 검색 과정을 도입하여 논의의 진전과 관련된 콘텐츠를 효과적으로 활용합니다.

- **Performance Highlights**: 다양한 도메인의 과학 문헌에 대한 실험을 통해 ToD는 정보가 풍부한 주장을 생성하고 논문 간의 대조를 효과적으로 수행했습니다. 이는 연구자들이 문헌 리뷰를 지원하는 데 큰 도움이 됩니다. ToD는 또한 세밀한 비교 요약을 생성할 수 있는 능력을 입증했습니다.



### Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning (https://arxiv.org/abs/2502.14765)
Comments:
          Accepted to NAACL 2025 (Main)

- **What's New**: 이번 연구는 단계별(step-by-step) 사실 검증 시스템을 통해 세 가지 의료 관련 사실 확인 데이터셋에 적용함으로써 기존 접근보다 개선된 성능을 입증하였습니다. 최근의 방법들은 대규모 언어 모델(LLMs)의 다단계 질문-응답 방식을 활용하여 점진적으로 정보를 수집하고 이에 따라 판단을 내리도록 설계되었습니다. 이러한 새로운 접근법은 전통적인 FV 방법의 한계를 해결하고, 특히 도메인 특정(claim-specific) 주장에 대한 검증의 잠재력을 보여줍니다.

- **Technical Details**: 전통적인 사실 검증 파이프라인은 문서 검색(document retrieval), 증거 추출(evidence extraction), 그리고 판별 예측(verdict prediction)으로 이루어집니다. 연구에서는 DeBERTa 모델을 활용하여 NLI(자연어 추론) 작업을 통해 주장과 증거의 관계를 예측하였습니다. 반대로 제안된 단계별 LLM 시스템은 증거를 수집하고자 추가 질문을 생성하며, 온라인 검색 엔진을 통해 정보를 검색하고, 논리적 추론을 활용하여 근거를 정리합니다.

- **Performance Highlights**: 연구 결과, 기존의 전통적인 접근 방식에 비해 개선된 최종 성능을 발휘하였으며, 다양한 LLMs와 외부 웹 검색, 논리 프레디케이트를 활용한 구조적 추론에 대한 여러 설정에서 평가되었습니다. 이 단계별 시스템은 복잡한 주장을 효과적으로 검증할 수 있는 능력을 보여줍니다. 연구진은 GitHub을 통해 모든 데이터와 코드를 공개하여, 이후 연구자들이 이 시스템을 활용할 수 있도록 지원하고 있습니다.



### On the Influence of Context Size and Model Choice in Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2502.14759)
Comments:
          Accepted to Findings of NAACL 2025

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 다양한 구성 요소를 체계적으로 평가합니다. 특히 제공된 컨텍스트의 이상적인 크기, 기본 LLM 선택, 및 검색 방법의 차이를 분석하여 RAG 시스템의 견고한 개발을 위한 지침을 제시하고자 합니다. 연구는 짧은 답변을 다루는 기존의 RAG 평가 방식에서 벗어나, 장기형 질문 응답(Long-form QA)을 다루어 더 복잡한 문제를 탐구합니다.

- **Technical Details**: RAG 시스템은 크게 리트리버(retriever)와 리더(reader)로 이루어져 있습니다. 본 연구에서는 컨텍스트 크기가 QA 성능에 미치는 영향, 서로 다른 LLM의 선택에 따른 중요성, 그리고 두 가지 리트리버(BM25 및 세맨틱 검색)가 최종 QA 성능에 미치는 영향을 연구합니다. 이를 통해 컨텍스트의 크기와 리트리버의 선택이 RAG 시스템의 성능에 미치는 영향을 비교하고 분석합니다.

- **Performance Highlights**: 연구 결과, 최종 QA 성능은 15개의 컨텍스트 스니펫까지는 점진적으로 향상되지만, 그 이상으로는 정체되거나 하락하는 경향을 보입니다. 생물의학 도메인에서는 Mistral과 Qwen이 가장 뛰어난 성능을 보인 반면, 백과사전 도메인에서는 GPT와 Llama가 두각을 나타냈습니다. 오픈 도메인 환경에서는 성능이 Gold 기준과 멀어지는 경향을 보였으며, BM25는 정밀도 향상에 기여하지만 세맨틱 검색은 더 넓은 정보 범위를 제공하는 것으로 나타났습니다.



### MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders (https://arxiv.org/abs/2502.14753)
- **What's New**: 이번 연구에서는 MedVAE라는 새로운 2D 및 3D 오토인코더(autoencoder) 모델을 소개합니다. 이 모델은 방대한 의료 이미지를 압축된 잠재 표현(latent representations)으로 인코딩(enconding)하고, 쉽게 고해상도(high-resolution) 이미지로 디코딩(decoding)할 수 있습니다. 특히, 기존의 고해상도 이미지를 대체할 수 있는 효율적인 방법을 제시하여, 임상적으로 중요한 피처를 효과적으로 보존합니다.

- **Technical Details**: MedVAE는 105만 개 이상의 의료 이미지를 사용하여 훈련되었습니다. 이 모델은 두 단계의 훈련 접근법을 통해 잠재 표현의 질을 최적화하는 데 중점을 두고 있습니다. 또한, 2D 및 3D 이미지를 위한 일반화된 오토인코더를 개발하여, 다양한 임상적 특징을 유지하며 효율성을 개선하고 있습니다. 개별적으로, 2D 이미지의 경우 f라는 다운사이징(f) 팩터를 적용하여 크기를 줄이고, 3D 이미지의 경우도 유사한 과정이 적용됩니다.

- **Performance Highlights**: 실험 결과, MedVAE의 잠재 표현은 CAD(Computer-Aided Diagnosis) 파이프라인에서 고해상도 이미지를 대체할 수 있으며, 성능을 유지하거나 초과할 수 있음을 보여줍니다. 이를 통해 저장 요구량이 최대 512배 줄어들고, CAD 모델 훈련의 효율성이 최대 70배 개선됩니다. 또한, 전문가 평가에서도 MedVAE의 디코딩된 재구성이 임상적으로 중요한 피처를 효과적으로 보존하는 것으로 나타났습니다.



### Multi-Agent Coordination across Diverse Applications: A Survey (https://arxiv.org/abs/2502.14743)
Comments:
          23 pages, 4 figures, 2 tables

- **What's New**: 이번 조사에서는 다수의 에이전트 시스템(MAS)에 대한 조정 연구의 상태를 설명하며, 네 가지 중요한 조정 질문인 (1) 조정이란 무엇인가; (2) 조정의 필요성; (3) 누구와 조정할 것인가; (4) 어떻게 조정할 것인가를 다룹니다. 특히, 최근의 LLM(대형 언어 모델) 기반의 MAS 방법이 인간과 유사한 능력을 자랑하며, 복잡한 문제 해결에 대한 집단적 지능을 보여주는 사례를 강조합니다.

- **Technical Details**: 조정은 에이전트 간의 관계를 관리하는 것으로, 이는 에이전트의 활동 간의 의존성을 다루는 것입니다. MAS의 동적 관계는 일반적으로 두 가지 질문, 즉 '누구와 조정할 것인가'와 '어떻게 조정할 것인가'를 해결하는 데 필수적입니다. 이러한 맥락에서, 조정 과정을 세 가지 구성 요소로 통합한 정의가 제시됩니다: 시스템 성능 평가, 조정할 집단의 사회적 선택, 조정 방법 결정입니다.

- **Performance Highlights**: 조정은 다양한 분야에서 중요한 문제로, 특히 차세대 MAS의 확장성과 이질성(heterogeneity) 및 학습 메커니즘에 대한 열린 도전 과제가 논의됩니다. 연구자는 계층적 조정과 분산 조정의 융합, 인간-MAS 조정, LLM 기반의 MAS가 앞으로의 연구 방향으로 주목할 가치가 있다고 강조합니다. 이러한 연구는 다양한 응용 프로그램 간의 지식 이전을 촉진하고 복잡한 문제에 대한 해결책을 제시할 수 있는 가능성을 내포하고 있습니다.



### YOLOv12: A Breakdown of the Key Architectural Features (https://arxiv.org/abs/2502.14740)
- **What's New**: YOLOv12는 이전 버전인 YOLO 시리즈의 강점을 기반으로 한 뚜렷한 발전으로, 실시간 객체 탐지의 새로운 기준을 제시합니다. 이 모델은 최적화된 백본(R-ELAN), 7x7 분리 가능 합성곱, FlashAttention 기반의 지역적 주목 기능을 통합하여 특징 추출, 효율성, 탐지 강도를 개선했습니다. YOLOv12는 다양한 하드웨어 플랫폼에 걸쳐 배포 가능하며, 지연이 중요한 애플리케이션부터 높은 정확도가 요구되는 응용 분야까지 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: YOLOv12는 강력한 다층 구조를 바탕으로 최신 딥러닝 기술을 활용하여 성능을 극대화합니다. 특히, Residual Efficient Layer Aggregation Network(R-ELAN)를 통해 기울기 병목 문제를 완화하고, 특징 융합을 개선합니다. 또한, 7x7 분리 가능한 합성곱이 전통적인 위치 인코딩을 대체하면서 매개변수를 줄이며 공간적 맥락을 유지합니다. 이러한 장치들은 YOLOv12의 실시간 처리 능력을 유지하면서도 탐지 정확도를 크게 향상시킵니다.

- **Performance Highlights**: YOLOv12는 mAP(평균 평균 정밀도)과 추론 속도 모두에서 일관된 성능 향상을 보여주어 자율 시스템, 보안 및 실시간 분석 분야에 적합한 모델입니다. 특히 복잡한 시각적 패턴을 인식하는 데 뛰어난 능력을 보이며, 특히 소형 또는 중첩된 객체 탐지에서의 성능을 크게 개선하였습니다. 따라서 YOLOv12는 자율 주행 차량, 의료 영상 분석, 농업 분야 등에서 실용적으로 활용될 수 있는 빼어난 선택지로 자리잡을 것입니다.



### EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration (https://arxiv.org/abs/2502.14735)
Comments:
          9 pages, 6 figures, accpeted by WWW 2025

- **What's New**: EAGER-LLM은 LLM(대형 언어 모델)을 기반으로 하는 새로운 생성 추천 프레임워크로, 내재적(endogenous) 및 외재적(exogenous) 행동과 의미 정보를 비침입적으로 통합합니다. 이 시스템은 사용자 행동 및 추천 데이터에서의 협업 신호를 원활하게 이해하고 통합할 수 있도록 설계되었습니다. EAGER-LLM은 기존 LLM 기반 추천 시스템이 직면한 여러 문제를 해결하는 접근 방식을 제공합니다.

- **Technical Details**: EAGER-LLM의 도입에서 주요한 요소는 이중 소스 지식 풍부 항목 색인(Dual-source Knowledge-rich Item Indices)과 비침해적 다중 스케일 정렬 재구성 과제(Non-Invasive Multiscale Alignment Reconstruction Tasks)입니다. 이를 통해 모델은 외부 신호와 의미 신호를 보다 효율적으로 처리하고 이해할 수 있게 됩니다. 또한, 모델의 추천 성능과 학습 능력을 미세 조정하는 어닐링 어댑터(Annealing Adapter)를 도입하여 전체 모델의 성능을 향상시킵니다.

- **Performance Highlights**: EAGER-LLM은 세 가지 공개 벤치마크에서 진행된 엄격한 실험을 통해 기존 방법들에 비해 우수한 성능을 입증하였습니다. 특히 추천 정확도를 높이면서도 대화 및 설명 생성 능력을 유지하는 데 성공하며, 이는 모델이 협업 및 의미 신호에 대해 깊이 있는 이해를 가능하게 하는 중요한 기여를 보여줍니다.



### WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models (https://arxiv.org/abs/2502.14727)
- **What's New**: 최신 연구에서는 WavRAG라는 새로운 Retrieval Augmented Generation (RAG) 프레임워크를 소개했습니다. 기존의 RAG 시스템은 오디오 정보를 처리하는 데 있어 Automatic Speech Recognition (ASR)을 사용하는데, 이는 중요한 오디오 정보를 버리고 transcription 오류와 계산 오버헤드를 초래하는 한계점을 가지고 있습니다. WavRAG는 이러한 문제를 해결하기 위해 raw audio를 직접 처리하며, 텍스트와 오디오를 통합한 지식 표현을 제공합니다.

- **Technical Details**: WavRAG는 end-to-end 오디오 지원을 제공하는 최초의 RAG 프레임워크로, WavRetriever라는 새로운 retriever를 도입하여 텍스트와 오디오의 하이브리드 지식베이스에서 정보를 검색합니다. 또한, Chain-of-Thought (CoT) 추론을 통합하여 음성 대화 모델의 맥락 내 기능을 향상시키고 있습니다. 이러한 접근 방식은 raw audio와 텍스트 입력을 공유 임베딩 공간으로 직접 인코딩하여 계산 오버헤드와 오류 전파를 방지합니다.

- **Performance Highlights**: WavRAG는 최첨단 ASR-텍스트 RAG 파이프라인과 비교하여 유사한 검색 성능을 달성하면서도 10배 빠른 속도를 제공합니다. 또한, 텍스트-오디오 하이브리드 검색 기능을 통해 RAG의 가능성을 오디오 모드로 확대하여 더욱 풍부하고 맥락 관련 정보 처리를 가능하게 합니다.



### Ranking Joint Policies in Dynamic Games using Evolutionary Dynamics (https://arxiv.org/abs/2502.14724)
- **What's New**: 본 논문은 진화적 접근법을 통해 동적 게임에서 에이전트 간의 상호작용을 분석하고, 에이전트의 보상을 고려하여 안정적인 전략을 식별하는 것을 목표로 합니다. 이를 위해 에이전트의 행동 대신 전략을 고려하여 동적 게임을 경험적 형태로 변환하고, 진화적 방법론인 \(\alpha\)-Rank를 적용하여 장기적인 동적을 기반으로 전략 프로필을 평가합니다. 실험에서는 그래프 색칠 문제의 확률적 버전을 협력적으로 해결하는 에이전트를 대상으로 다양한 플레이 스타일을 전략으로 정의합니다.

- **Technical Details**: 경험적 보상 매트릭스는 특정 플레이 스타일을 따르는 정책으로 훈련된 그래프 색칠 게임의 시뮬레이션에서 파생되었습니다. 각 매트릭스 항목은 대응하는 프로필에서 전략의 보상을 나타내며, 첫 번째 값은 행 전략의 보상, 두 번째 값은 열 플레이어의 보상을 의미합니다. 노드와 엣지의 색상 코딩을 통해 전략 프로필의 강도를 시각화하며, 더 어두운 파란색 노드는 강한 조인 프로필을 나타냅니다.

- **Performance Highlights**: 실험 결과, 다양한 게임 구성에서 결정적인 전략 프로필이 다르게 나타나며, (M, CA) 전략 프로필이 0.31의 점수로 1위로 평가되었습니다. 대부분의 전략 프로필은 동일한 고위험도를 넘어서는 성능을 유지하지 못하지만, (W, CA), (C, W), (WL, CA) 등의 프로필은 모든 구성에서 강력한 성과를 보였습니다. 또한, 합산된 결과는 모든 프로필이 연결된 클러스터를 형성하며, (M, CA) 프로필이 MCC에서 이동이 불가능하다는 것을 나타냅니다.



### Human Misperception of Generative-AI Alignment: A Laboratory Experimen (https://arxiv.org/abs/2502.14708)
- **What's New**: 이번 연구는 경제적 의사결정 맥락에서 생성형 인공지능(GenAI)의 정렬(alignment)에 대한 사람들의 인식을 조사하는 실험을 수행했습니다. 연구 결과, 사람들은 GenAI의 선택과 인간의 선택 사이의 정렬 정도를 과대평가하는 경향이 있음을 발견했습니다. 특히, 연구 참가자들이 GenAI의 선택을 예측하는 방식은 그들의 개인적인 선택과 높은 상관관계를 보였습니다.

- **Technical Details**: 이 실험은 두 부분으로 나누어 진행되었습니다. 첫 번째 부분에서는 참가자들이 다양한 위험, 시간 선호, 사회적 선호 및 전략적 상호작용을 포함한 결정 환경에서 자신의 선택을 하도록 요청받았습니다. 두 번째 부분에서는 AI 챗봇이 인간 사용자를 대신하여 선택할 경우, 참가자들이 그 선택을 예측하도록 유도하였습니다. 이러한 방식으로 우리는 인간이 GenAI의 선택에 대해 가질 수 있는 왜곡된 인식을 확인했습니다.

- **Performance Highlights**: 연구 결과, 인간 참가자들이 GenAI의 선택을 예측할 때 그 예측은 평균적으로 GenAI의 선택보다 인간의 선택과 훨씬 가까웠습니다. 이는 사람들이 GenAI의 행동을 자신의 선택에 비추어 예측하기 때문임을 보여주는 결과입니다. 이러한 오해는 GenAI를 통한 의사결정 과정에서 비효율적인 위임(decision delegation)을 초래할 수 있으며, 이는 더 심각한 결과를 낳을 가능성이 있습니다.



### Not All Data are Good Labels: On the Self-supervised Labeling for Time Series Forecasting (https://arxiv.org/abs/2502.14704)
- **What's New**: 이번 논문에서는 Self-Correction with Adaptive Mask (SCAM)이라는 새로운 접근 방식을 소개합니다. 이는 Time Series Forecasting (TSF) 데이터셋을 재라벨링하고, 기존의 고품질 데이터 의존도를 줄이며, 데이터의 활용 가능성을 최대화하는 방법론입니다. 논문은 여러 실제 데이터셋에서 SCAM이 다양한 예측 모델의 성능을 개선함을 보여주고 있습니다.

- **Technical Details**: SCAM 접근 방식은 여러 후보 데이터셋을 생성하고, 단순한 reconstruction 네트워크를 최적화하여 중간 결과를 pseudo labels로 사용하는 self-supervised 학습 패러다임입니다. 이 과정에서는 Spectral Norm Regularization (SNR)을 도입하여 오버피팅을 추가로 억제하는 전략을 세웠습니다. 이 두 가지 기법은 TSF 모델의 일반화를 개선하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, SCAM은 11개의 실제 데이터셋에서 다양한 backbone 모델의 성능을 꾸준히 향상시키는 것을 보여주었습니다. 구체적으로, SCAM 접근 방식은 overfitting 문제를 감소시켜 모델의 일반화 성능을 증대시키는 데 기여하였습니다. 논문은 이러한 결과를 통해 TSF 데이터셋 구축과 self-supervised 학습을 통한 모델 개선의 새로운 관점을 제시하고 있습니다.



### General Uncertainty Estimation with Delta Variances (https://arxiv.org/abs/2502.14698)
- **What's New**: 이번 논문에서는 Delta Variance라는 새로운 알고리즘 패밀리를 제안하여 대규모 신경망(neural networks)의 에피스테믹 불확실성(epistemic uncertainty)을 효율적으로 추정하는 방법을 연구합니다. 이 방법은 신경망의 아키텍처나 훈련 절차를 변경할 필요 없이 구현할 수 있으며, 단일 그래디언트 계산의 비용만으로 경쟁력 있는 결과를 제공합니다. 따라서 기존의 부트스트랩(bootstrapping)이나 MC 드롭아웃 방법보다 더 효율적으로 불확실성을 처리할 수 있습니다.

- **Technical Details**: Delta Variance는 베이지안(Bayesian), 빈도주의(frequentist) 및 휴리스틱(heuristic) 개념을 연결하여 불확실성을 추정합니다. 특히, 각 알고리즘은 그래디언트 벡터를 사용한 벡터-행렬-벡터 곱 형태로 나타나며, 이는 모델의 파라미터 불확실성이 출력에 미치는 영향을 정량화합니다. 연구에서는 이러한 Delta Variance를 그래프와 날씨 예측을 위한 GraphCast 시스템에 적용하고, 이론적으로 다양한 유도 방법을 통해 설명을 제공합니다.

- **Performance Highlights**: GraphCast 날씨 예측 시스템에 적용한 결과, Delta Variance가 기존의 앙상블 방법과 유사한 품질을 제공하면서도 더 적은 계산 자원으로 동작하는 것을 확인했습니다. 이러한 효율성을 통해 데이터가 제한된 상황에서도 불확실성을 잘 관리할 수 있는 가능성을 보여줍니다. 실험 결과, 제안된 Delta Variance의 일반적인 성질과 이점이 강조되어, 다양한 관련 방법과의 관계를 일관된 관점에서 이해할 수 있게 해줍니다.



### seqKAN: Sequence processing with Kolmogorov-Arnold Networks (https://arxiv.org/abs/2502.14681)
- **What's New**: 이번 연구에서는 seqKAN이라는 새로운 Kolmogorov-Arnold Network (KAN) 아키텍처를 제안합니다. 이 아키텍처는 기존의 KAN 프레임워크에 충실하며, 다중 계층 다중 퍼셉트론(Multi-Layer Perceptron, MLP) 사용 없이 직접적인 재귀성을 포함합니다. seqKAN은 기존에 제안된 여러 KAN 아키텍처들과는 달리, 가중합(weighted summation)과 고정 활성화 함수(fixed activation function)를 재도입하지 않아서 KAN의 핵심 개념을 더욱 잘 반영합니다.

- **Technical Details**: seqKAN은 순차적 데이터 처리에 최적화된 아키텍처로, 노드가 수신하는 값들을 단순히 합산하는 방식을 채택하여 비선형성을 캡처합니다. KAN의 활성화 함수는 데이터로부터 학습된 함수들로, 이 방식을 통해 KAN은 네트워크 에지에 대한 매개변수를 학습하고 있습니다. 이와 비교할 때 seqKAN은 많은 매개변수를 학습해야 하며, 더 적은 수의 보다 좁은 층으로 목표 함수를 근사할 수 있는 효율적인 대안입니다.

- **Performance Highlights**: seqKAN의 성능은 물리학 문제를 기반으로 생성된 데이터를 사용한 실험을 통해 검증되었습니다. 이 데이터셋에서 seqKAN은 이전 KAN 네트워크, 순환 심층 네트워크(Recurrent Deep Networks), 기호 회귀(Symbolic Regression)와 비교했을 때 월등한 성능을 보였습니다. 특히, extrapolation 데이터셋에서의 성능이 두드러지며, 또한 가장 높은 투명성을 제공하는 아키텍처로 평가되었습니다.



### Data-Constrained Synthesis of Training Data for De-Identification (https://arxiv.org/abs/2502.14677)
Comments:
          Under review

- **What's New**: 이 연구는 개인정보 식별 정보를 추적하기 위해 임상 도메인에 적응된 대형 언어 모델(LLMs)을 활용하여 합성 임상 텍스트를 생성하는 방법을 제안합니다. 기존의 프라이버시 위험으로 인해 데이터셋이 널리 사용되지 않는 다수의 민감한 영역에서, 합성 데이터 생성은 새로운 대안으로 떠오르고 있습니다. 연구에서는 이 과정에서 소량의 도메인 데이터가 충분할 수 있음을 보여주면서, 머신 주석이 달린 데이터의 효과를 강조합니다.

- **Technical Details**: 본 연구는 두 가지 언어 모델인 GPT-SW3와 FLOR을 사용하여 스웨덴어 및 스페인어 임상 데이터의 합성을 진행합니다. 이 모델들은 각각 3200억 및 1400억 토큰의 데이터로 학습되어 있으며, NER 모델 학습에 필요한 합성 텍스트를 생성합니다. 이 과정은 오토 회귀(autoregressive) 방식으로 진행되며, 머신 주석 달기는 미세 조정된 NER 모델을 통해 이루어집니다.

- **Performance Highlights**: 연구 결과, 합성 데이터로 훈련된 NER 모델은 실제 민감 데이터로 훈련된 모델과 거의 동일한 성능을 나타내며, 데이터 유출 위험을 줄이는 데 기여합니다. 실험 결과에 따르면, 합성에서의 유용성은 높은 품질의 NER 모델에 의존하는 것으로 밝혀졌습니다. 또한, 주어진 작업에서 더 큰 생성 LLM을 사용하더라도 성능 향상에는 명확한 개선이 없는 것으로 나타났습니다.



### BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction (https://arxiv.org/abs/2502.14676)
- **What's New**: 이 논문에서는 교통 에이전트의 궤적 예측을 위한 새로운 접근법으로, 행동 유사 라벨(behavioral pseudo-labels)을 도입하여 보행자와 이질적인 에이전트의 행동 특성을 효과적으로 포착합니다. 이 방법은 궤적 예측의 정확도를 크게 높이며, 추가적인 레이블 주석의 필요성을 줄입니다. 제안된 BP-SGCN(Behavioral Pseudo-Label informed Sparse Graph Convolution Network)은 이러한 유사 라벨을 학습하고 예측 모델에 정보를 제공합니다.

- **Technical Details**: BP-SGCN은 두 가지 모듈로 구성되어 있습니다. 첫 번째 모듈은 심층 비지도 행동 군집화(deep unsupervised behavior clustering) 모듈로, 경로 관찰을 통해 에이전트에 유사 라벨을 부여합니다. 두 번째는 목표 지향 유사 라벨 기반 궤적 예측 모듈로, 에이전트의 공간적 상호작용과 시간적 의존성을 효과적으로 모델링하기 위해 Sparse Graph Convolutional Network(SGCN)를 활용합니다. 이를 통해 경량화된 훈련 방식으로 두 모듈을 함께 최적화합니다.

- **Performance Highlights**: 실험 결과, BP-SGCN은 SDD와 Argoverse 1 데이터셋에서 이질적인 예측을 초과 달성하며, ETH/UCY 데이터셋과 보행자 전용 SDD 설정에서도 우수한 성능을 보입니다. 제안된 새로운 개념인 행동 유사 라벨을 통해 다양한 행동 군집을 모델링하여 궤적 예측의 성능이 상당히 향상되었습니다. 소스 코드는 연구의 발전을 위해 GitHub에서 제공됩니다.



### Explanations of Deep Language Models Explain Language Representations in the Brain (https://arxiv.org/abs/2502.14671)
- **What's New**: 이 연구는 설명 가능한 인공지능(Explainable AI, XAI) 기법을 활용하여 대형 언어 모델(LLM)과 뇌의 언어 처리 메커니즘 간 깊은 연결을 형성하는 새로운 접근 방식을 소개합니다. 이전의 연구가 주로 LLM의 내부 표현을 신경 활동과 일치시키는 데 중점을 두었던 반면, 본 연구는 속성 방법(attribution methods)을 사용하여 LLM의 다음 단어 예측에 대해 이전 단어들이 어떻게 기여하는지를 정량화하였습니다. 이 작업은 AI와 신경과학 간의 양방향 다리 역할을 하며, LLM의 예측 기능과 인간의 뇌에서의 언어 처리 과정을 통합하는 새로운 통찰을 제공합니다.

- **Technical Details**: 연구진은 네 가지 유형의 속성 방법을 적용하여 세 가지 LLM에서 피쳐 표현을 구성하고, 이를 통해 자연적인 이야기 청취 중에 기록된 fMRI 활동을 모델링하였습니다. 이 과정에서 각 LLM에서 도출된 피쳐 공간은 참가자의 뇌 반응을 예측하는 데 독립적으로 사용되었습니다. 예측은 각 개인의 뇌 데이터에 맞춰 설계된 선형 리지 회귀 모델을 통해 이루어졌으며, 다섯 번의 교차 검증을 통해 정확성을 평가하였습니다.

- **Performance Highlights**: 연구 결과, 속성 방법이 언어 관련 뇌 영역의 광범위한 네트워크 내에서 뇌 활동을 효과적으로 예측하는 데 강력한 가능성을 보였습니다. 특히, 속성 방법은 초기 언어 처리 영역에서의 뇌 활동 예측에서 전통적인 내부 표현보다 우수한 성과를 나타냈고, LLM의 깊이에 따른 설명과 뇌의 언어 처리 영역 간의 계층적 관계를 발견하였습니다. 이 발견은 LLM이 맥락 정보를 통합하는 데 있어 공유되는 메커니즘을 시사합니다.



### Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs (https://arxiv.org/abs/2502.14645)
- **What's New**: 이번 연구에서는 Cross-Lingual Knowledge Democracy Edit (X-KDE)라는 새로운 지식 편집 프레임워크를 제안합니다. 이 방법은 대형 언어 모델(LLM)이 특정 언어에서 수정된 지식을 다른 언어로 효과적으로 전파할 수 있도록 설계되었습니다. 특히, X-KDE는 Cross-lingual Edition Instruction Tuning (XE-IT)와 Target-language Preference Optimization (TL-PO)의 두 단계로 구성되어 있습니다.

- **Technical Details**: X-KDE는 두 단계로 구성됩니다. 첫 번째 단계인 XE-IT에서는 선별된 병렬 데이터셋을 사용하여 모델을 세밀하게 조정하며, 두 번째 단계인 TL-PO에서는 고급 최적화 기법을 적용하여 언어 간의 일관성을 유지합니다. 이를 통해 소스 언어에서의 지식이 타겟 언어로 원활하게 전파됩니다.

- **Performance Highlights**: 연구 결과, X-KDE는 Bi-ZsRE 및 MzsRE 벤치마크에서 평균 8.19%의 성능 향상을 달성하였으며, 단일 언어 설정에서도 높은 정확도를 유지합니다. X-KDE는 기존 방법들보다 더 나은 성능을 보이는 새로운 최첨단(SOTA) 솔루션으로 자리잡았습니다.



### ReQFlow: Rectified Quaternion Flow for Efficient and High-Quality Protein Backbone Generation (https://arxiv.org/abs/2502.14637)
- **What's New**: 이 연구에서는 고속 및 고품질 단백질 백본 생성을 위한 새로운 Rectified Quaternion Flow (ReQFlow) 매칭 방법을 제안합니다. 기존의 diffusion 및 flow 기반 생성 모델들은 낮은 디자인 가능성과 높은 계산 복잡도로 인해 실용적인 대규모 응용에 제한이 있습니다. 이에 반해 ReQFlow는 단백질 체인의 각 잔여물에 대해 무작위 잡음에서 지역적인 3D 이동과 회전을 생성하는 방식을 채택하여 효율성을 크게 향상시킵니다. 이 방법은 단위 쿼터니언을 사용하여 회전을 표현하고, 구면 선형 보간법(SLERP)을 통해 쿼터니언 흐름을 구성합니다.

- **Technical Details**: ReQFlow는 쿼터니언 수학을 기반으로 하여, 각 회전을 SO(3) 공간에서 비선형적으로 학습합니다. 본 연구는 무작위 잡음과 모델 스스로 생성한 단백질 백본을 쌍으로 사용하여 재훈련을 통해, QFlow 모델을 보정합니다. 이러한 기술을 통해, 새로운 쿼터니언 흐름은 ℝ³ 및 SO(3)에서 겹치지 않는 샘플링 경로를 생성하며, 단백질 백본 생성을 가속화합니다. ReQFlow는 회전 각도가 0 또는 π에 근접할 때 보장된 수치적 안정성 및 계산 효율성을 제공합니다.

- **Performance Highlights**: 실험 결과, ReQFlow는 단백질 백본 생성을 위한 최첨단 성능을 달성하며, 기존 방법보다 샘플링 단계 수를 현저히 줄이고 추론 시간을 단축시킵니다. 예를 들어, ReQFlow는 길이 300의 백본을 생성할 때 RFDiffusion보다 37배, Genie2보다 62배 더 빠른 성능을 보였습니다. 특히, ReQFlow는 500개 이상의 잔여물로 구성된 긴 체인의 단백질 백본을 생성할 때에도 효율성을 유지하며, 생성된 백본의 디자인 가능성 점수는 샘플링 단계 수에 따라 0.912에서 0.972로 측정되었습니다.



### ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors (https://arxiv.org/abs/2502.14627)
- **What's New**: 본 논문에서는 다국어 오디오-텍스트 검색(Multilingual audio-text retrieval, ML-ATR)의 일관성 문제를 분석하고, 새로운 접근법인 ATRI를 제안합니다. 기존 ML-ATR 시스템들은 언어 간 유사성 매칭에서 일관성이 부족했는데, 이는 데이터 분포 오류에 기인합니다. ATRI는 1-to-K Contrastive Learning과 Audio-English Co-Anchor Contrastive Learning 두 가지 전략을 통해 데이터의 일관성을 향상시키고, 검색 성능을 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구팀은 오디오와 언어 텍스트 간의 모달 정렬 방향 오류 및 가중치 오류를 분석하여 ML-ATR에서의 일관성 문제에 대한 이론적 상한선을 제시하였습니다. ATRI는 각 훈련 세션에서 언어 데이터를 무작위로 선택하는 기존 방식의 한계를 극복하며, 모델의 최적 가중치 수렴을 돕는 새로운 훈련 전략을 채택했습니다. KCL은 데이터 분포 오류를 제거하여 성능 향상에 기여하며, CACL은 오디오와 영어 텍스트 간의 모달 정렬을 개선합니다.

- **Performance Highlights**: 실험 결과와 함께 제안된 ATRI 접근법이 AudioCaps와 Clotho 데이터셋을 활용해 뛰어난 회수율 성능을 기록했음을 보여줍니다. ATRI는 영어를 포함한 8개 주요 언어에 대해 최첨단 성능을 달성하여, 검색 결과에서 일관성을 확보했습니다. 연구진은 ATRI의 코드와 방법론을 공개할 예정이며, 이는 향후 ML-ATR 연구에 중요한 기여를 할 것입니다.



### Exploring RWKV for Sentence Embeddings: Layer-wise Analysis and Baseline Comparison for Semantic Similarity (https://arxiv.org/abs/2502.14620)
Comments:
          17 pages, 3 tables, preprint on ArXiV, includes detailed analysis of RWKV for semantic similarity tasks

- **What's New**: 이 논문은 RWKV라는 새로운 언어 모델 아키텍처의 효능을 조사하며, 이는 선형 주의 메커니즘(linear attention mechanism)으로 알려져 있습니다. 저자는 사전 훈련된 RWKV 모델의 다양한 숨겨진 층(hidden layers)에서 생성된 문장 임베딩의 의미적 유사성(semantic similarity)을 평가하기 위해 층별(layer-wise) 분석을 수행했습니다. Microsoft Research Paraphrase Corpus(MRPC) 데이터셋을 사용하여 GloVe 기반 기준선과 비교했을 때, RWKV 임베딩은 의미적 관련성을 일부 포착했지만 Spearman 상관관계 측면에서 GloVe보다 부족한 성능을 보였습니다.

- **Technical Details**: RWKV는 혁신적인 수용 가중치 키 값(Receptance Weighted Key Value) 주의 메커니즘을 사용하여 선형 시간 복잡성을 달성합니다. 본 연구에서는 RWKV-v6-Finch-1B6-HF 모델을 사용하여 숨겨진 층에서 문장 임베딩을 추출하고, 계층별로 의미적 정보를 포착하는 역할을 조사하는 전략을 수립했습니다. GloVe 임베딩을 기준선으로 사용하여, 모든 단어의 GloVe 벡터를 평균내어 문장 임베딩을 생성하고, MRPC 데이터셋에서 문장 임베딩 간의 코사인 유사성이 어떻게 평가되는지를 분석했습니다.

- **Performance Highlights**: RWKV 임베딩은 일부 의미적 관련성을 포착했지만, GloVe 기준선에 비해 성능이 낮은 것으로 나타났습니다. 이 연구는 문장 임베딩 생성 분야에서 RWKV의 가능성을 확인하고, 효율성 및 문맥 이해에 대한 고유한 절충점을 강조합니다. 앞으로 RWKV의 구조를 최대한 활용하고, 기존 문장 임베딩 기술과의 성능 격차를 해소하기 위한 추가 연구가 필요합니다.



### Reward Models Identify Consistency, Not Causality (https://arxiv.org/abs/2502.14619)
Comments:
          16 pages

- **What's New**: 이 연구에서는 보상 모델(reward models, RMs)이 대규모 언어 모델(large language models, LLMs)을 인간의 선호와 일치시키는 데 중요한 역할을 한다고 설명합니다. 기존 보상 모델이 후보 출력을 정렬하고 평가하는 방법에 대한 기존 가정을 도전하는 여러 발견을 제시합니다. 특히, 최신 보상 모델은 인과적 정합성(causal correctness)보다 구조적 일관성(structural consistency)을 우선시한다는 점이 강조됩니다.

- **Technical Details**: 연구에서는 LLM의 문제 해결 과정에서 보상 모델이 어떻게 작동하는지를 다루며, 구조적 일관성을 중시함을 보여줍니다. 질문을 생략했을 때 보상 점수에 미치는 영향이 적고, 수치 값을 수정하거나 추론 흐름을 방해했을 때는 보상 점수가 큰 변화를 보인다는 점이 밝혀졌습니다. 보상 모델은 이론적으로 올바른 추론 경로(complete reasoning trajectories)를 필요로 하며, 불완전한 단계를 거치면 보상 할당에 큰 변동이 발생합니다.

- **Performance Highlights**: 결과적으로, 현재의 보상 모델은 진정한 추론 품질을 평가하기보다는 일관성을 판단하는 경향이 있는데, 이는 보상 모델의 기본 한계를 시사합니다. 이 연구는 인과성 인식(causality-aware) 보상 모델로의 전환 필요성을 강조하며, 이는 응답을 순위 매기기보다는 논리적 유효성을 검증하는 데 중점을 둡니다. 연구의 결과는 다양한 아키텍처, 데이터셋 및 태스크에서도 일관되게 나타났습니다.



### A Theory for Conditional Generative Modeling on Multiple Data Sources (https://arxiv.org/abs/2502.14583)
Comments:
          35 pages

- **What's New**: 이번 논문은 다중 데이터 출처 간의 상호작용에 대한 이론적 분석을 최초로 시도합니다. 각 조건이 별개의 데이터 출처를 나타내는 조건부 생성 모델링에 중점을 두며, 다원적 훈련의 분포 추정 오류에 대한 일반적인 상한을 설정하고 있습니다. 데이터의 특성과 유사성을 기반으로 다중 출처 훈련이 단일 출처 훈련보다 더 뚜렷하게 우수한 성능을 보일 수 있음을 보여줍니다.

- **Technical Details**: 이론적 접근 방식으로는 최대 우도 추정(MLE)에 대한 일반적인 분포 추정 오류의 상한을 설정하고, 평균 총 변동 거리(average total variation distance)를 측정하는 방법을 사용합니다. 또한, 설정된 이론을 조건부 가우시안 추정 및 심층 생성 모델에 구체화하여 이들의 브래킷 수(bracketing number)를 특성화합니다. 세 가지 구체적인 모델에서 다중 출처와 단일 출처 훈련의 명확한 예측 오류 경계를 도출했습니다.

- **Performance Highlights**: 시뮬레이션 및 실험에서 다중 출처 훈련이 단일 출처 훈련보다 우수한 성능을 보이며, 낮은 FID 점수를 기록했습니다. 이러한 결과는 저자들이 이론적으로 제시한 보장과 일치하며, 성능 개선은 출처의 수와 유사성에 따라 달라진다는 것을 입증합니다. 이처럼 다양한 실험을 통해 이론적 발견의 유효성을 확인했습니다.



### Factor Graph-based Interpretable Neural Networks (https://arxiv.org/abs/2502.14572)
Comments:
          The Thirteenth International Conference on Learning Representations

- **What's New**: 이 논문은 AGAIN(많은 설명성을 갖춘 신경망)를 제안하여 알려지지 않은 변동성(perturbations) 하에서 이해 가능한 설명을 생성하는 방법을 소개합니다. 기존의 방법들이 변동성을 알고 있거나 훈련을 통해 대응하려 했던 것과 달리, AGAIN은 논리적 규칙을 직접 통합하고 설명 오류를 추론 중에 식별하여 수정하는 방식을 채택합니다. 이를 통해 AGAIN은 고전적인 광고적 훈련(adversarial training) 방식에 대한 한계를 극복합니다.

- **Technical Details**: AGAIN은 세 가지 모듈로 구성됩니다. 첫 모듈에서는 의미적 개념과 레이블 카테고리, 그리고 이들 간의 논리적 규칙을 표현하는 팩터 그래프(factor graph)를 구축합니다. 두 번째 모듈에서는 AGAIN이 개념 수준의 설명을 생성하고 예측 카테고리를 도출한 후, 이를 팩터 그래프에 투입하여 논리적 오류를 식별합니다. 마지막으로, 세 번째 모듈에서는 설명 수정(interactive intervention switch) 전략을 제안하여 논리 오류를 교정합니다.

- **Performance Highlights**: 저자들은 CUB, MIMIC-III EWS, Synthetic-MNIST와 같은 세 가지 데이터셋에서 많은 실험을 수행하였고, AGAIN이 기존 방법들과 비교했을 때 알려지지 않은 변동성 하에서도 더 높은 이해성을 가진 설명을 생성하는 것을 입증했습니다. 이 연구의 주요 기여는 알려지지 않은 변동성에 대한 새로운 접근 방식을 나타내고, 논리적 오류 식별 및 수정 방법의 효과성을 강화시킨 것입니다.



### Less is More: Improving LLM Alignment via Preference Data Selection (https://arxiv.org/abs/2502.14560)
- **What's New**: 본 연구에서는 Direct Preference Optimization (DPO) 접근 방식을 통해 대규모 언어 모델(LLM)을 인간의 선호와 정렬하는 데 필요한 데이터 선택(data selection)의 중요성을 강조합니다. 기존의 DPO 연구가 주로 목적 함수(objective function)에 중점을 두었다면, 우리는 데이터 선택 측면에서 DPO를 개선하여 노이즈가 포함된 데이터로 인한 매개변수 축소(parameter shrinkage) 문제를 해결하는 새로운 방법을 제안합니다. 이를 통해 효과적인 데이터 선별이 모델 성능 향상에 기여할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 새로운 마진 극대화(margin-maximization) 원칙을 제안하여 DPO 훈련에서 데이터 세트를 정제하는 방법을 개선합니다. 특히, 외부 보상 마진(external reward margins)과 DPO 내재적 보상 마진(implicit DPO reward margins)을 모두 고려하여 데이터를 선택하는 이중 마진 가이드를 적용합니다. 이 방법을 통해 소음이 많은 데이터로 인한 문제를 해결하고, 보다 효율적인 데이터 선택을 통해 모델 성능을 향상할 수 있음을 이론적으로 입증합니다.

- **Performance Highlights**: 우리는 Ultrafeedback 데이터 세트의 10%만 사용해도 Llama와 Mistral 시리즈 모델에서 AlpacaEval 2.0 벤치마크 기준으로 3%에서 8%의 성능 향상을 이끌어냈습니다. 또한, 25%의 온라인 데이터를 사용한 반복적 DPO에서도 약 3%의 성능 향상과 훈련 시간을 단축하는 성과를 달성했습니다. 이러한 결과는 데이터 선택 전략이 선호 최적화의 진전을 위한 잠재력을 가지고 있음을 보여줍니다.



### FUIA: Model Inversion Attack against Federated Unlearning (https://arxiv.org/abs/2502.14558)
Comments:
          Initial manuscript

- **What's New**: 이 연구는 Federated Learning의 데이터 삭제 메커니즘인 Federated Unlearning(FU)에 대한 새로운 공격 방식인 Federated Unlearning Inversion Attack(FUIA)를 제안합니다. FUIA는 FU의 세 가지 유형인 sample unlearning, client unlearning 및 class unlearning의 프라이버시 누출 취약점을 포괄적으로 분석하는데 중점을 둡니다. 기존 FU 연구는 주로 효율성 향상에 집중한 반면, 이 연구는 FU의 원래 목적을 정면으로 반박하는 새로운 접근법을 통해 이러한 데이터 삭제 과정에서 발생할 수 있는 프라이버시 문제에 주목하고 있습니다.

- **Technical Details**: FUIA에서는 서버가 '정직하지만 호기심 많은' 공격자로 작용하며, 원본 모델과 unlearned 모델의 차이를 기록하고 활용하여 잊혀진 데이터의 특성과 레이블을 노출합니다. FU의 목표에 따라 class unlearning은 특정 데이터 클래스의 영향을 제거하고, client unlearning은 특정 클라이언트의 데이터를 완전히 잊게 하며, sample unlearning은 개별 데이터 샘플의 효과를 제거하는 것을 목표로 합니다. 이 연구는 이러한 세 가지 FU 메커니즘에 대한 FUIA의 적용 가능성을 입증하는 다양한 공격 전략을 설계했습니다.

- **Performance Highlights**: FUIA는 잊혀진 데이터의 사적인 정보를 효과적으로 복구하는 데 있어 높은 효과성을 보여주었습니다. 실험 결과는 FU 메커니즘의 프라이버시 보호 측면에서 나타날 수 있는 위험들과 그 취약성을 강조합니다. 이 공격 방식은 FU 설계의 중요한 결함을 드러내며, 향후 프라이버시 보호를 위한 안전하고 신뢰할 수 있는 메커니즘 개발에 중요한 기준이 될 것입니다.



### Multiscale Byte Language Models -- A Hierarchical Architecture for Causal Million-Length Sequence Modeling (https://arxiv.org/abs/2502.14553)
Comments:
          Under Review

- **What's New**: 이번 논문에서 소개된 Multiscale Byte Language Model (MBLM)은 기존의 tokenization 문제를 해결하기 위해 설계된 모델로, 5M 바이트의 긴 입력 문자열을 단일 GPU에서 훈련할 수 있는 능력을 제공합니다. MBLM은 Transformer와 Mamba 블록을 결합하여 효율적인 학습 및 추론을 가능하게 하며, 비주얼 Q&A와 같은 멀티모달 작업에서도 뛰어난 성능을 보여줍니다. 이는 바이너리 데이터와 다양한 표현을 통합하여 더 넓은 데이터 소스에 적응할 수 있는 가능성을 강조합니다.

- **Technical Details**: MBLM은 N개의 causal decoder 모델로 이루어진 계층 구조를 가지고 있으며, 각 단계는 입력 시퀀스의 패치와 컨텍스트 크기를 조절하여 효율적인 처리 속도를 달성합니다. 이 모델은 256개의 바이트 레벨 설정을 사용하는 어휘를 통해 입력되는 바이트 스트림을 처리하여, 훈련 및 추론 중에 중간 활성화를 선택적으로 체크포인트 할 수 있는 기능을 제공합니다. 이러한 접근 방식은 5M 바이트에 달하는 긴 시퀀스를 단일 GPU에서 효율적으로 처리하는 데 기여합니다.

- **Performance Highlights**: MBLM은 멀티모달 환경에서 사용하는 새로운 어플리케이션을 통해 기존의 CNN-LSTM 아키텍처와 유사한 성능을 달성했음을 보여주었습니다. 특히, 비주얼 Q&A 작업에서 순수한 다음 토큰 예측만으로 훌륭한 결과를 보이며, 다양한 데이터 표현과의 강력한 적응성을 입증했습니다. 이러한 성능은 다양한 데이터 소스로부터 효과적인 크로스 모달리티 지식 전이를 가능하게 하여, 데이터 포맷에 관계없이 바이스트림에서 특징과 패턴을 포착하는 데 중점을 두고 있습니다.



### Position: Graph Learning Will Lose Relevance Due To Poor Benchmarks (https://arxiv.org/abs/2502.14546)
- **What's New**: 이번 논문은 그래프 기계 학습에 대한 벤치마킹의 한계와 문제점을 다루고 있습니다. 현재의 벤치마킹 관행은 실제로 혁신적인 응용 분야와의 연계를 부족하게 하여, 효과적인 발전을 저해하고 있음을 지적하고 있습니다. 특히, 과거의 벤치마크는 2D 분자 그래프와 같은 좁은 영역에 국한되어 있으며, 기하학적 구조의 복잡성을 간과하고 있습니다.

- **Technical Details**: 현재 그래프 학습에서는 MPNN (Message-Passing Neural Networks) 및 GNN (Graph Neural Networks)을 활용하여 다양한 응용 분야에서 성과를 내고 있지만, 이에 대한 벤치마킹은 여전히 미흡합니다. 구체적으로, 데이터 세트 분할 및 평가 프로토콜의 불일치가 연구 간 비교의 유효성을 해치고 있으며, 오히려 제한된 데이터 세트에 의존하여 높은 변동성과 통계적 유의성을 가진 결과들을 초래하고 있습니다. 이는 결국 대규모 사전학습 또는 기초 모델로서의 확장성에 제약을 주고 있습니다.

- **Performance Highlights**: 논문에서는 그래프 기계 학습 연구의 진전을 위해 보다 의미 있는 벤치마크, 엄격한 평가 프로토콜, 그리고 도메인 전문가와의 강력한 협력이 필요하다고 주장하고 있습니다. 새로운 벤치마크 및 참조 모델의 튜닝을 통해 분자 예측 작업 및 대규모 이질적 데이터 세트에서의 전이학습 가능성을 모색하고 있으며, 이로 인해 그래프 학습 커뮤니티가 보다 실질적이고 효과적인 발전을 이룰 수 있을 것으로 기대합니다.



### CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models (https://arxiv.org/abs/2502.14529)
- **What's New**: 이번 논문에서는 Large Language Model 기반의 Multi-Agent Systems (LLM-MAS)에서의 보안 문제를 다루고 있습니다. 특히, 'Contagious Recursive Blocking Attacks (Corba)'라는 새로운 유형의 공격을 소개하며, 이는 에이전트 간 상호작용을 방해하고 시스템의 가용성을 감소시킬 수 있습니다. 기존의 공격 기법들이 간과했던 이러한 차단 공격은 LLM-MAS의 보안적 측면에서 심각한 우려를 불러일으키고 있습니다.

- **Technical Details**: Corba 공격은 두 가지 주요 특성을 활용하여, 네트워크 토폴로지에 관계없이 전파할 수 있는 전염성(contagious)을 가지고 있습니다. 또한, 재귀적(recursive) 특성 덕분에 계산 자원을 지속적으로 고갈시킬 수 있습니다. 이는 공격에 적대적인 프롬프트가 반복적으로 시스템 안에 남아 효과를 유지하게 만듭니다.

- **Performance Highlights**: 논문에서는 AutoGen과 Camel의 두 가지 널리 사용되는 LLM-MAS 프레임워크를 대상으로 Corba의 효과를 실험했습니다. 실험 결과, Corba는 다양한 토폴로지 구조에서 LLM-MAS의 가용성을 감소시키고 자원을 낭비하며, 기존의 방송 기반 공격에 비해 우수한 성능을 보였습니다. 이러한 연구 결과는 LLM-MAS의 보안 메커니즘 개발에 있어 기초를 제공하며, 향후 연구에서 더욱 심도 깊은 분석이 필요함을 시사합니다.



### Small Graph Is All You Need: DeepStateGNN for Scalable Traffic Forecasting (https://arxiv.org/abs/2502.14525)
Comments:
          Yannick Wölker and Arash Hajisafi contributed equally to this work

- **What's New**: DeepStateGNN은 교통 데이터를 분석하기 위해 제안된 새로운 그래프 신경망 모델로, 예측 및 재구성의 두 가지 주요 작업에서 효율성을 보여줍니다. 기존 GNN과 달리, DeepStateGNN은 교통 센서를 심층 상태 노드(Deep State Nodes)라는 높은 수준의 그래프 노드로 클러스터링하여 독립적으로 작용하는 복잡한 관계를 처리합니다. 이 모델은 센서의 유사성을 기반으로 동적으로 구성된 고정 크기의 그래프를 활용하여 효율성과 확장성을 제고합니다.

- **Technical Details**: DeepStateGNN은 센서 간의 공간적 근접성, 기능적 유사성 및 행동적 유사성에 따라 센서를 클러스터링하여 정보를 그룹화합니다. 이 접근 방식은 각 그룹의 정보를 잠재 상태로 집계하여 트래픽 패턴을 전문적으로 표현할 수 있도록 합니다. 메시지 전송 작업을 통해 각 심층 상태 노드는 관련된 다른 노드와 교류하며, 이를 통해 복잡한 교통 상태를 효율적으로 추론할 수 있습니다.

- **Performance Highlights**: 실험 결과, DeepStateGNN은 교통 예측 및 재구성 정확도를 향상시키며, 메모리 및 계산 효율성을 크게 개선했습니다. 다양한 센서 네트워크에서 뛰어난 성능을 발휘하고, 교통 데이터 손실이 있거나 관측되지 않은 위치에 대해서도 정확한 예측을 제공합니다. 또한, METR-LA+ 데이터셋을 제공하여 실제 교통 조건을 반영한 데이터로 연구의 신뢰성을 높였습니다.



### PLPHP: Per-Layer Per-Head Vision Token Pruning for Efficient Large Vision-Language Models (https://arxiv.org/abs/2502.14504)
Comments:
          12 pages, 8 figures

- **What's New**: 본 논문은 Large Vision-Language Models (LVLMs)의 효율성을 획기적으로 개선하기 위한 Per-Layer Per-Head Vision Token Pruning (PLPHP) 방법을 제안합니다. 기존의 LVLM은 많은 비주얼 토큰 때문에 추론 효율성에서 제한을 받고 있었고, PLPHP는 이러한 문제를 해결하기 위해 두 가지 레벨의 세밀한 pruning 기법을 도입했습니다. 이 방법은 각 레이어의 비주얼 정보 주목도를 기반으로 토큰 유지율을 동적으로 조정하며, 토큰 손실을 최소화하면서 성능을 향상시킵니다.

- **Technical Details**: PLPHP는 두 가지 주요 구성 요소로, 첫 번째는 layer-level retention rate allocation으로, 각 레이어의 비주얼 정보 주목도를 고려하여 토큰의 보존율을 조정합니다. 두 번째는 head-level vision token pruning으로, 동일한 레이어 내에서도 각 attention head가 독립적으로 중요한 컨텍스트를 보존할 수 있도록 합니다. 이를 통해 PLPHP는 각 레이어의 주목에 따라 비주얼 토큰을 선택적으로 유지하거나 제거하여 성능 저하를 효과적으로 방지합니다.

- **Performance Highlights**: 실험 결과, PLPHP는 decoding 속도를 18% 향상시키고 Key-Value Cache (KV Cache) 크기를 50% 이상 줄이며, 평균 0.46%의 성능 하락을 보였습니다. 또한, 다중 이미지 작업에서 상당한 성능 개선을 달성하여 기법의 효과iveness를 입증하였습니다. 이 연구는 LVLMs의 효율성과 확장성을 높이는 중요한 기여를 합니다.



### MLGym: A New Framework and Benchmark for Advancing AI Research Agents (https://arxiv.org/abs/2502.14499)
Comments:
          35 pages, 12 figures, 10 tables

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트를 평가하고 개발하기 위한 새로운 프레임워크인 Meta MLGym과 MLGym-Bench를 소개합니다. 이 프레임워크는 다양한 AI 연구 과제를 포함하는 최초의 Gym 환경으로, LLM 에이전트를 훈련하는 강화 학습(rl) 알고리즘 연구를 가능하게 합니다. MLGym-Bench는 컴퓨터 비전, 자연어 처리 등 다양한 도메인에서 13개의 개방형 AI 연구 과제를 포함하고 있으며, 연구 과정에 필요한 실제 기술을 요구합니다.

- **Technical Details**: MLGym은 다양한 AI 연구 과제를 통합하여 LLM 에이전트를 개발하고 평가할 수 있는 통합 플랫폼으로 설계되었습니다. 이 프레임워크는 에이전트의 성능을 평가하기 위해 강화를 위한 알고리즘, 커리큘럼 학습, 개방형 학습 등 다양한 새로운 학습 알고리즘을 쉽게 추가하고 통합할 수 있도록 합니다. 또한, 각 에이전트의 출력물인 모델, 알고리즘, 예측 세트 등을 평가할 수 있는 유연한 평가 방식도 제공합니다.

- **Performance Highlights**: MLGym-Bench에서 제공하는 과제를 수행한 여러 최첨단 LLM 모델의 성능을 비교하였으며, 그들 각각의 강점과 제한점을 강조했습니다. 연구 결과 현재의 최첨단 모델들은 대부분 하이퍼파라미터를 최적화하여 개선할 수 있지만, 혁신적인 가설이나 알고리즘을 생성하지는 못한다는 것을 확인하였습니다. 이 연구는 LLM 에이전트의 AI 연구 능력을 향상시키기 위한 프레임워크와 벤치마크를 오픈소스로 제공하여 향후 연구에 기여할 것으로 기대됩니다.



### Temporal Misalignment and Probabilistic Neurons (https://arxiv.org/abs/2502.14487)
- **What's New**: 본 논문은 Spiking Neural Networks (SNNs)의 ANN-SNN 변환 과정에서 'temporal misalignment'라는 새로운 현상을 발견하고 이에 대한 이해를 제공합니다. 이 현상은 SNN 레이어 간의 랜덤한 스파이크 재배열이 성능 향상을 이끈다는 점을 강조합니다. 이 관찰을 바탕으로, 두 가지 단계의 확률적 스파이킹 뉴런(Two-phase Probabilistic Spiking Neurons, TPP)을 소개하여 변환 과정을 더욱 향상시킵니다.

- **Technical Details**: 이 연구에서는 Integrate-and-Fire (IF) 스파이킹 뉴런을 기반으로 한 모델을 사용하며, ANN에서 SNN으로의 변환 과정이 자세히 설명됩니다. ANN의 사전 훈련된 모델에서 SNN으로 가중치 및 바이어스를 전이하여 동일한 아키텍처 구조를 공유합니다. ANN 출력과 SNN 출력의 관계를 수학적으로 분석하여 스파이크 기준화 과정을 수행하며, 여기서 ReLU 함수를 활성화 함수로 사용합니다.

- **Performance Highlights**: 제안된 방법은 CIFAR-10/100, CIFAR10-DVS 및 ImageNet 데이터셋을 활용한 포괄적인 실험을 통해 기존의 SOTA(SOTA, State-Of-The-Art) 변환 방법 및 다른 훈련 방법들보다 높은 정확도를 기록했습니다. 연구 결과는 이론적 관점과 실증적 관점을 모두 포함하며, SNN 기반의 AI 모델이 기존의 인공신경망(ANN)보다 에너지 효율적인 대안으로 자리잡을 수 있음을 실증적으로 보여줍니다.



### How Jailbreak Defenses Work and Ensemble? A Mechanistic Investigation (https://arxiv.org/abs/2502.14486)
- **What's New**: 이번 연구는 jailbreak 공격에 대한 두 가지 주요 방어 메커니즘인 safety shift와 harmfulness discrimination을 식별하여, generative task의 표준을 binary classification 문제로 재구성하고, 모델의 거부 경향을 평가합니다. 연구팀은 또한 safety와 helpfulness의 균형을 맞추기 위해 inter-mechanism ensembles와 intra-mechanism ensembles라는 두 가지 앙상블 방어 전략을 개발했습니다. 실험 결과는 이러한 전략이 모델의 안전성을 효과적으로 향상시킨다는 것을 증명했습니다.

- **Technical Details**: 이 논문에서는 내부 모델 최적화를 통한 defend 방법과 query에 대한 효율적 수정 방법을 검토했습니다. 내부 방어 메커니즘은 모델의 생성 프로세스를 개선하거나 입력 쿼리를 수정하여 이루어지며, 시스템 리마인더 및 노이즈 주입 등 다양한 접근 방식을 포함합니다. 연구는 multimodal 환경에서의 방어 메커니즘을 중점적으로 분석하였으며, 이는 기존 언어 기반 방어 연구와는 차별화된 접근입니다.

- **Performance Highlights**: LLaVA-1.5 모델을 사용한 MM-SafetyBench와 MOSSBench 데이터셋에서 다양한 방어 방법에 대한 실험을 진행한 결과, 각 방어 메커니즘의 효과성을 입증했습니다. 연구팀은 28개의 방어 방법을 평가하여 multimodal 방어 연구의 격차를 해소하고, 향후 전략 선택 및 개발에 대한 통찰력을 제공했습니다. 실험을 통해 safety와 helpfulness의 최적 균형을 이루는 방법이 효과적임을 입증했습니다.



### Enhancing Smart Environments with Context-Aware Chatbots using Large Language Models (https://arxiv.org/abs/2502.14469)
Comments:
          11 pages, 3 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLM)을 활용하여 스마트 환경 내에서 문맥 인지 상호작용을 구현하는 새로운 아키텍처를 제안합니다. UWB 태그를 통한 사용자 위치 데이터와 센서를 갖춘 스마트 홈을 통합하여 사용자의 활동 인식(HAR)을 실시간으로 수행하여, 개인 맞춤형 상호작용과 추천을 생성할 수 있는 챗봇을 구현합니다. 이러한 접근 방식은 기존의 정적 챗봇 상호작용을 넘어 사용자의 현재 상황에 동적으로 적응할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 주요 구성요소로는 사용자 상호작용을 모니터링하는 환경 센서 네트워크, UWB를 이용한 정밀 위치 추적, 및 HAR 모델이 포함됩니다. 이러한 센서는 사용자의 활동과 환경에 대한 풍부한 데이터를 수집하며, MQTT를 통해 중앙 허브로 전송되어 분석됩니다. 이후, 이 데이터를 기반으로 LLM와 통합된 챗봇이 사용자 요구를 이해하고 맥락에 맞는 응답을 생성하도록 설계되었습니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 사례 연구를 통해 제안된 아키텍처의 실행 가능성과 효과성을 입증하였습니다. HAR와 실시간 위치 정보를 통합하여 개인 맞춤형의 문맥에 맞는 사용자 경험을 제공함으로써 스마트 홈 내에서 보다 직관적이고 유용한 상호작용을 가능하게 했습니다. 이 연구는 LLM과 스마트 환경 기술의 융합이 사용자 맞춤형 지원을 제공하는 데에 혁신적인 가능성을 열어준다는 점을 강조합니다.



### Single-image Reflectance and Transmittance Estimation from Any Flatbed Scanner (https://arxiv.org/abs/2502.14462)
Comments:
          Accepted to Computers & Graphics

- **What's New**: 이번 연구에서는 전통적인 디지털 재료 캡처 방식의 한계를 극복하기 위해 모든 플랫베드 스캐너에서 사용할 수 있는 새로운 방법을 제안합니다. 이 방법은 shading(음영)과 specularity(사착)을 효과적으로 제거하고, 재료의 opacity(불투명도)와 transmittance(투과율)를 추정하여 더욱 사실적인 디지털 복제를 생성합니다. 또한, 이 과정에서 기존의 이미지-투-이미지 전환 네트워크 방법은 부족하다는 점을 지적하며, cycle-consistency 손실을 활용한 개선된 접근 방식을 소개합니다.

- **Technical Details**: 연구팀은 intrinsic image decomposition(내재적 이미지 분해) 방법에서 영감을 받아 새로운 재료 캡처 기술을 개발했습니다. 이 기술은 Spatially-Varying Bidirectional Scattering Distribution Function (SVBSDF)을 통해 복잡한 빛의 상호작용을 재현하며, 이는 고급 플랫베드 스캐너가 아닌, 스마트폰과 같은 다양한 스캐닝 장치에서도 잘 작동함을 보였습니다. 재료의 각 파라미터를 개별적으로 측정하는 이미지 기반 메트릭과 최종 appearance(외관)를 평가하는 렌더링 감지 메트릭을 모두 사용하여 철저한 실험을 진행했습니다.

- **Performance Highlights**: 본 방법은 여러 재료에서 매우 높은 해상도와 정확도로 완전한 SVBSDF를 추정함으로써, 스캐너의 조명이 무작위적이더라도 효과적인 결과를 생성합니다. 이는 사용자 친화적인 재료 캡처 설정을 통해 보다 저렴하고 접근하기 쉬운 방법으로, 디자인, 건축, 패션 등 다양한 산업에 적용될 수 있습니다. 실험 결과, 이 방법은 특히 섬유와 같은 얇은 재료의 모델링에서 뛰어난 성능을 발휘했습니다.



### Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing (https://arxiv.org/abs/2502.14458)
- **What's New**: Llamba는 Llama-3.x에서 Mamba 아키텍처로 증류된 효율적인 순환 언어 모델 패밀리로 소개됩니다. Llamba-1B, Llamba-3B, Llamba-8B 모델은 Transformer 기반 모델에 비해 높은 추론 처리량과 더 큰 배치 크기를 처리하면서도 유사한 성능을 유지합니다. MOHAWK 프레임워크를 활용하여, Llamba는 전통적인 대형 언어 모델보다 훨씬 적은 훈련 데이터로도 뛰어난 성과를 달성한 점이 주목할 만합니다.

- **Technical Details**: Llamba 모델들은 Llama 모델의 구조를 유지하면서 Mamba-2 레이어로 자기 주의(self-attention)를 대체하여 높은 추론 처리량을 확보합니다. Llamba-1B, 3B, 8B 모델은 각각 16, 28, 32개의 Mamba-2 블록으로 구성되며, 각 모델은 Llama-3.1의 토크나이저와 어휘를 공유합니다. 또한 Llamba 모델들은 랭크 감소 기법(Low-Rank Adaptation)을 통해 파라미터 효율성을 높이며, 제한된 리소스 환경에서도 원활한 구현이 가능합니다.

- **Performance Highlights**: Llamba 모델들은 다양한 벤치마크에서 전통적인 모델과 동등한 성능을 보이며, 효율성과 성능의 새로운 기준을 설정합니다. 특히, 0.1% 미만의 훈련 데이터로도 최첨단 성능을 달성하는 데이터 효율성이 주목받고 있습니다. 이러한 성능들은 Llamba를 고성능 언어 모델링의 스케일 가능한 솔루션으로 자리매김하게 합니다.



### Watch Less, Feel More: Sim-to-Real RL for Generalizable Articulated Object Manipulation via Motion Adaptation and Impedance Contro (https://arxiv.org/abs/2502.14457)
- **What's New**: 이 연구에서는 관절 구조체(Object)의 조작을 위해 새로운 RL 기반 파이프라인을 제시합니다. 이 파이프라인은 가변 임피던스 컨트롤(variable impedance control)과 관찰 기록(observation history)을 활용하여 일반화 가능한 조작을 가능하게 하며, 실제 환경에서의 부드럽고 민첩한 동작을 강조합니다. 시뮬레이션과 실제 환경 간의 격차(sim-to-real gap)를 줄이기 위해 시각 데이터(vision data)를 직접 사용하지 않고도 유용한 저차원 데이터를 추출합니다.

- **Technical Details**: 제안된 방법은 주어진 작업에 대해 닫힌 고리(closed-loop) 방식으로 하나의 정교한 행동을 예측하는 데 중점을 둡니다. 기존 연구들과는 달리, 본 연구에서는 시각 정보 대신 관찰 이력을 사용하여 보다 효율적으로 물체의 조작을 수행합니다. 또한, 변동 임피던스 제어(variable impedance control)를 도입하여 객체의 움직임에 대한 높은 수용성을 제공하고, 이를 통해 시뮬레이션과 실제 환경 간의 직접적인 전이를 향상시키는 데 기여합니다.

- **Performance Highlights**: 원활한 다단계 종단 간 조작을 가능하게 하는 보상 함수(reward function) 시스템을 설계하였으며, 이로 인해 실제 세계에서의 84% 성공률을 달성했습니다. 또한, 연구 결과는 이전에 보지 못한 물체에 대한 높은 일반화 가능성을 보여줍니다. 실험에서는 4개의 작업과 500회의 시행을 통해 제안하는 방법의 제로샷 추론(zero-shot inference)에서 각각 96% 및 84%의 성공률을 기록했습니다.



### An Efficient Ground-aerial Transportation System for Pest Control Enabled by AI-based Autonomous Nano-UAVs (https://arxiv.org/abs/2502.14455)
- **What's New**: 이번 연구에서는 농업에서의 해충 감지를 위한 새로운 접근 방식을 제안합니다. 여러 대의 자율 미니 드론(nano-UAV)과 단일 중량급 트랙터를 조합하여 효율적인 해충 감지 및 치료를 가능하게 합니다. 실시간으로 이미지를 분석하며 최적의 경로 계획을 통해 해충을 조기에 발견할 수 있습니다.

- **Technical Details**: 이 시스템은 저전력 소모와 실시간 성능을 갖춘 convolutional neural network (CNN)를 기반으로 합니다. 설계된 CNN은 0.79의 평균 평균 정밀도(mAP)를 기록하며, 출발과 동시에 장애물 회피를 위한 글로벌 및 로컬 경로 계획을 통해 빠른 반응성을 보여줍니다. 시스템은 Crazyflie nano-UAV에 탑재된 다양한 센서와 최적화된 경로 계획 알고리즘을 활용합니다.

- **Performance Highlights**: 본 연구의 결과, 제안된 시스템은 200x200m의 포도밭을 탐색하는 데 있어 기존의 단일 트랙터 시스템에 비해 최대 20시간의 작업 시간을 단축시킬 수 있음을 보여줍니다. 실험 결과, nano-UAV Fleet은 실시간 영상 분석 및 경로 계획을 통해 해충 감지와 치료를 보다 효율적으로 수행할 수 있음을 입증하였습니다.



### PredictaBoard: Benchmarking LLM Score Predictability (https://arxiv.org/abs/2502.14445)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)에서의 예측 가능성을 평가하기 위한 새로운 협업 벤치마킹 프레임워크인 PredictaBoard를 소개합니다. PredictaBoard는 기존 데이터셋에서 특정 작업 인스턴스에 대한 LLM 오류를 예측하는 점수 예측자(assessor)의 능력을 평가하기 위해 설계되었습니다. 이를 통해 LLM과 assessor의 조화로운 연구와 개발을 촉진하며, 안전한 AI 시스템을 위한 기반을 마련하고자 합니다.

- **Technical Details**: PredictaBoard는 LLM의 성능과 그 예측 가능성을 공동으로 평가하는 데 초점을 맞추고 있습니다. 이 프레임워크는 LLM-배치 쌍을 객체로 하여 각각의 벤치마크 인스턴스에서 LLM의 점수를 예측하도록 설계된 assessor를 포함합니다. 예측 가능성의 품질과 LLM의 성능을 결합하여 정확도-거부 곡선(Accuracy-Rejection Curve, ARC)을 통해 평가하며, 이는 실용적으로 오류 허용 한계를 결정하고 예측 가능한 유효 영역(predictably valid region)을 파악하는데 도움을 줍니다.

- **Performance Highlights**: PredictaBoard는 현재 상태의 LLM을 대상으로 하여 기초 평가자(baseline assessors)와 함께 초기 실험 결과를 보고합니다. 이는 LLM의 성능뿐만 아니라 그 예측 가능성을 동시에 고려함으로써 AI 안전성을 강화할 수 있는 가능성을 제시합니다. 또한 PredictaBoard는 LLM과 assessor의 쌍을 순위화할 수 있는 리더보드 시스템을 도입할 수 있으며, 이는 예측 가능성 향상을 위한 전반적인 진행 상황을 보장합니다.



### Stochastic Resonance Improves the Detection of Low Contrast Images in Deep Learning Models (https://arxiv.org/abs/2502.14442)
Comments:
          MSc Course Project

- **What's New**: 이 논문은 확률적 공명(stochastic resonance)이 비율 기반 인공 신경망(rate-based artificial neural networks)에서 어떻게 발생하는지를 조사했습니다. 확률적 공명은 특정 시스템에서 소음의 도입으로 약한 신호의 탐지 능력이 향상되는 현상입니다. 본 연구에서는 LSTM 반복 신경망을 사용하여 숫자 인식을 수행하였고, 실험에서 소음이 첨가되는 조건에서 모델의 성능 변화가 분석되었습니다.

- **Technical Details**: 모델은 20개의 은닉 유닛을 가진 단일 LSTM 레이어와 softmax 활성화가 적용된 밀집(dense) 레이어로 구성됩니다. 학습에는 MNIST 숫자 데이터셋이 사용되며, 비어 있는 자극 클래스가 수동으로 추가되었습니다. 테스트 중 소음의 영향을 파악하기 위해 비율 기반 LSTM RNN의 입력에 다양한 유형과 수준의 제어된 소음이 추가되었습니다.

- **Performance Highlights**: 실험 결과, 특정 조건에서 소음이 성능에 긍정적인 영향을 미친다는 것을 확인하였습니다. 특히, 낮은 대비의 자극에서는 높은 소음 수준이 유리하게 작용하는 반면, 높은 대비의 자극에서는 낮은 소음 수준에서 가장 좋은 성능을 보였습니다. 본 연구는 비율 기반 신경망에서 확률적 공명이 발생할 수 있음을 첫 번째로 제시하며, 이는 약한 신호에 대한 탐지 성능 향상으로 이어질 수 있습니다.



### Distribution Matching for Self-Supervised Transfer Learning (https://arxiv.org/abs/2502.14424)
- **What's New**: 본 논문에서는 Distribution Matching (DM)이라는 새로운 자기 지도 학습(self-supervised learning) 기반의 전이 학습(transfer learning) 방법을 제안합니다. 이 방법은 표현(distribution) 분포를 미리 정의된 기준(reference) 분포로 유도하면서도 데이터 증강(augmentation) 불변성을 유지합니다. DM의 설계는 직관적으로 구조화된 표현 공간을 제공하며, 쉽게 해석할 수 있는 하이퍼파라미터(hyperparameters)를 특징으로 합니다.

- **Technical Details**: DM 방법은 두 가지 주요 이론적 보장을 통해 강화됩니다. 첫 번째는 모집단 정리(population theorem)로, 이는 자기 지도 학습(task)과 목표 분류(accuracy) 간의 간극을 연결합니다. 두 번째는 샘플 정리(sample theorem)로, 이는 타겟 도메인에서 제한된 수의 샘플로도 우수한 분류 성능(classification performance)을 제공할 수 있음을 보여줍니다.

- **Performance Highlights**: 여러 실제 데이터셋(real-world datasets)과 평가 지표(evaluation metrics)를 기반으로 실험한 결과, DM은 기존의 자기 지도 전이 학습 방법들에 비해 경쟁력 있는 성과를 보였습니다. 특히, 비라벨(unlabeled) 샘플의 수가 충분히 크기만 하면, DM은 제한적인 타겟 샘플로도 뛰어난 분류 성능을 달성할 수 있습니다.



### Reliable Explainability of Deep Learning Spatial-Spectral Classifiers for Improved Semantic Segmentation in Autonomous Driving (https://arxiv.org/abs/2502.14416)
- **What's New**: 이 논문에서는 하이퍼스펙트럼 이미징(hyperspectral imaging, HSI)과 심층 신경망(deep neural networks, DNNs)을 통합하여 자율주행 시스템에서의 의미 분할(semantic segmentation) 작업의 정확도를 높이기 위한 새로운 접근방법을 제안합니다. 이미지 분류를 위한 기존의 saliency 방법들이 신뢰성 부족으로 비판받고 있는 가운데, 이 연구에서는 DNN의 특정 층에서 활성화 및 가중치를 활용하여 입력 특징과 예측 간의 관계를 보다 정확하게 포착할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 FCNs(fully convolutional networks)를 사용하여 HSI 기반 분류 모델의 성능을 평가하며, 고차원적이고 클래스 간 유사성과 내부 클래스 변동성(intra-class variability)이라는 HSI의 도전 과제를 해결하기 위해 스펙트럼 정보와 공간 정보를 결합하는 방법론에 초점을 맞춥니다. 특히, 클래스 활성화 맵(class activation maps, CAM)의 보수성 속성을 의미 분할 네트워크로 확장하여, 예측 점수를 정량적으로 설명할 수 있는지에 대해 검토합니다.

- **Performance Highlights**: 이 방법은 실제 운전 시나리오에서 레이어의 활성화 및 가중치를 고려함으로써 HSI 기반의 분류 모델이 스펙트럼 정보의 기여도를 평가하는 데 유용함을 보여줍니다. 이 연구는 HSI가 3채널 및 단일 채널 DNN에 비해 우수한 성능을 보임을 입증하며, 스펙트럼 서명 정규화(spectral signature normalization)가 DNN의 강건성을 향상시키는 데 중요한 역할을 한다는 점을 강조합니다.



### S*: Test Time Scaling for Code Generation (https://arxiv.org/abs/2502.14382)
- **What's New**: 본 논문에서는 코드 생성을 위한 최초의 하이브리드 테스트 타임 스케일링 프레임워크인 S*를 제안합니다. S*는 기존의 병렬 스케일링(parallel scaling) 방법에 순차적 스케일링(sequential scaling)을 통합하여 생성된 코드의 커버리지(coverage)와 선택 정확도를 크게 향상시킵니다. 이 프레임워크는 반복적 디버깅(iterative debugging)을 통해 성능 한계를 Push합니다.

- **Technical Details**: S*는 두 가지 주요 단계로 작동합니다: 첫 번째는 생성 단계에서 병렬 샘플링(parallel sampling)에 순차적 스케일링을 추가하여, 각 샘플을 공용 테스트 케이스에 실행하여 출력을 얻고 이를 기반으로 코드를 반복적으로 개선합니다. 두 번째는 선택 단계로, S*는 각 샘플 쌍에 대해 차별화된 테스트 입력(distinguishing test inputs)을 생성하여 LLM에 프롬프트를 제공하고, 생성된 입력을 실행하여 최적의 샘플을 선택합니다.

- **Performance Highlights**: S*는 12개의 다양한 대규모 언어 모델에서 평가되어, 모든 모델 패밀리 및 크기에서 성능을 꾸준히 개선함을 보여줍니다. 특히, S*는 3B 모델이 GPT-4o-mini를 초월하도록 하고, 비추론 모델이 추론 모델을 초과하며, DeepSeek-R1-Distill-Qwen-32B는 LiveCodeBench에서 85.7%를 달성합니다.



### Affinity and Diversity: A Unified Metric for Demonstration Selection via Internal Representations (https://arxiv.org/abs/2502.14380)
Comments:
          8 pages, 10 figures

- **What's New**: 이번 연구에서는 In-Context Learning (ICL)의 성능 향상을 위해 새로운 평가 지표인 affinity와 diversity를 제안합니다. 이러한 지표는 ICL 모델의 내부 표현을 활용하여 demonstration의 품질을 통합적으로 평가합니다. 기존의 demonstration 선택 방법들이 서로 다른 목표를 가지고 최적화되면서 일관성 있는 결과를 내지 못했던 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 지표인 affinity는 쿼리와 demonstration 간의 코사인 유사도의 평균으로 정의되며, diversity는 demonstration 간의 레이블 토큰 representations의 분산으로 정의됩니다. 연구진은 ICL에 중요한 self-attention head를 식별한 후, 해당 head의 subspace에서 affinity와 diversity를 계산하여 demonstration의 품질을 평가합니다.

- **Performance Highlights**: 실험 결과, affinity와 diversity는 테스트 정확도와 강한 상관관계를 보였으며, 이는 이들 지표가 demonstration 선택에 효과적임을 나타냅니다. 제안된 지표는 다양한 이전 연구와 잘 align 되면서 일관성 있는 선택 방법을 제시하였으며, ICL 성능을 더욱 향상시키는 기초 자료가 될 것으로 기대됩니다.



### Discovering highly efficient low-weight quantum error-correcting codes with reinforcement learning (https://arxiv.org/abs/2502.14372)
Comments:
          18 pages, 14 figures, 4 tables

- **What's New**: 본 논문에서는 강화 학습(Reinforcement Learning, RL)을 기반으로 한 새로운 저중량양자오류정정코드 발견 방법을 제안합니다. 이 방법은 이전에 접근이 어려웠던 qLDPC(qu quantum Low-Density Parity-Check) 코드의 무게(weight)를 줄이면서, 코드의 거리를 높일 수 있는 가능성을 보여줍니다. 저자들은 이 방법을 통해 6중량 코드의 물리적 큐빗 오버헤드를 기존 연구에 비해 1~2 배수 줄일 수 있음을 입증하였으며, 이를 통해 실제 실험을 위한 효율적인 범위에 도달하게 되었다고 강조합니다.

- **Technical Details**: RL 프레임워크는 강화 학습 에이전트가 안정화 코드의 Tanner 그래프를 기반으로 행동을 선택하고 최대 변수를 보존하면서 체크 노드의 최대 차수를 최소화하는 형태로 구성됩니다. Proximal Policy Optimization (PPO) 알고리즘을 사용하여 행동 공간을 효율적으로 탐색하며, 안정적인 학습을 보장하기 위해 정책 업데이트를 제어합니다. 이 방식을 통해 강화 학습 모델은 기존 방법들에 비해 최대 73배 적은 물리적 큐빗 오버헤드를 달성하였습니다.

- **Performance Highlights**: 제안된 RL 기반의 저중량 코드 발견 방법은 이전 접근 방식보다 코드를 설계하는 최대 거리를 4배 높일 수 있음을 보여줍니다. 이러한 방법은 d≈35에 달하는 높은 거리를 요구하는 새로운 저중량 코드를 많이 발견하는데 성공하였으며, 이는 향후 수천 개의 물리적 큐빗을 가진 양자 장치에서 유용하게 사용될 수 있는 수준의 효율성을 가져왔습니다. 이러한 결과는 양자 오류 정정 코드 설계의 이전 한계를 뛰어넘는 중요한 진전을 나타냅니다.



### Entropy-UID: A Method for Optimizing Information Density (https://arxiv.org/abs/2502.14366)
Comments:
          5pages, 1 figures, submitting to ACL 2025

- **What's New**: 이번 연구에서는 정보 생성 모델의 효율성을 향상시키기 위해 엔트로피(Entropy)와 균일 정보 밀도(UID) 원칙을 균형 있게 결합한 새로운 토큰 선택 방법인 Entropy-UID를 제안합니다. 이 방법은 생성된 시퀀스 내부의 정보 분포를 더욱 고르게 만들어 시간적으로 정보 밀도가 불균일해지는 상황을 개선합니다. 실험 결과, Entropy-UID는 기존의 GPT-2 모델 대비 낮은 서프리살(surprisal) 및 엔트로피 변동성을 보여 더 균형 잡히고 인간과 유사한 텍스트 생성을 가능하게 합니다.

- **Technical Details**: Entropy-UID는 토큰 선택 과정에서 엔트로피와 서프리살 값을 동시에 최소화하여 정보 밀도를 최적화합니다. 알고리즘은 각 후보 토큰의 엔트로피와 서프리살 지표를 평가하여 이들의 가중치 조합을 최소화하는 토큰을 선택합니다. 본 연구는 문장 생성 및 텍스트 생성 작업에 대해 해당 방법의 효과성을 입증하기 위한 여러 실험을 수행하였으며, Hyperparameter인 α 값을 조정하여 엔트로피와 UID 사이의 균형을 맞췄습니다.

- **Performance Highlights**: 실험 결과, Entropy-UID 방법은 WikiText-2, OpenWebText, WMT 등 다양한 벤치마크 데이터셋에서 높은 성능을 나타냈습니다. 특히 Entropy-UID는 낮은 엔트로피 표준편차(≈2.8)와 안정적인 평균 서프리살(≈5.7)을 기록하여 예측의 불확실성과 정확성 간의 최적의 균형을 유지했습니다. 반면, 단일 목표 최적화 방법은 낮은 엔트로피와 높은 서프리살에 따른 성능 저하가 관찰되었습니다.



### Is Q-learning an Ill-posed Problem? (https://arxiv.org/abs/2502.14365)
Comments:
          Accepted at ESANN 2025

- **What's New**: 이 논문은 Q-learning의 불안정성을 조사하며, 이는 연속 환경에서 종종 발생하는 실무적 도전 과제입니다. 기존 연구에서는 이러한 불안정성을 부트스트래핑(bootstrapping)과 회귀 모델 오류(regression model errors)에 기인한다고 설명합니다. 저자들은 연속 상태 공간을 가지고 있는 RL 벤치마크에서 이러한 잠재적인 오류 원인을 체계적으로 제거하며 Q-learning의 근본적인 과제가 본질적으로 ill-posed(잘 정의되지 않은)할 수 있음을 밝힙니다.

- **Technical Details**: Q-learning은 Bellman 방정식을 사용하여 Q-값을 반복적으로 업데이트하는 방식으로 작동하지만, 많은 MDP가 연속 상태 공간을 포함하기 때문에 함수 근사기(function approximator)를 사용해야 합니다. 저자들은 신경망(NN)을 활용하여 Q-함수를 근사하는 방법에서 발생하는 불안정성 문제를 탐구하였으며, 다양한 Q-learning 알고리즘의 성능을 비교하였습니다. 특히 BSF-NFQ와 같은 모델 기반의 방식으로 부트스트래핑을 줄이면서도 실세계 환경의 동역학을 적용하여 더 나은 성능을 얻으려 했습니다.

- **Performance Highlights**: 실험 결과, BSF-NFQ 알고리즘을 통해 28%의 경우에서 성공적인 정책을 찾았음을 보여주었지만 여전히 불안정성이 완전히 해결되지 않았습니다. 정책 성능의 불안정성을 모니터링하며, 특정 반복(iteration)에서 성공적인 정책이 다음 반복에서 나쁜 정책으로 이어지는 현상을 관찰하였습니다. 이러한 결과는 Q-learning이 단순히 특정 알고리즘의 문제가 아니라, Q-values의 샘플 기반 평가에 의존하는 다른 방법에도 영향을 미친다는 점을 시사합니다.



### Purest Quantum State Identification (https://arxiv.org/abs/2502.14334)
- **What's New**: 이 연구에서는 양자 정보 처리에서의 중요한 문제인 노이즈 제약 하의 양자 상태 식별을 다룹니다. 우리가 제안하는 방법은 고전적인 최적 팔 식별 문제를 양자 도메인으로 일반화하여 K개의 미지의 n-큐빗 양자 상태 중에서 가장 순수한 상태를 식별하는 것입니다. 이 방법은 양자 계산(quantum computation)과 양자 통신(quantum communication)에 직접적인 적용이 가능합니다.

- **Technical Details**: 우리는 두 가지 알고리즘을 제안합니다. 첫 번째 알고리즘은 비일관 측정을 사용하여 오류를 줄이는 방법으로, $	ext{error} 	o 	ext{exp}ig(- 	ext{Omega}ig(rac{N H_1}{	ext{log}(K) 2^n }ig) ig)$의 성능을 보입니다. 두 번째 알고리즘은 일관 측정을 응용하여 더 낮은 오류 확률 $	ext{error} 	o 	ext{exp}ig(- 	ext{Omega}ig(rac{N H_2}{	ext{log}(K) }ig) ig)$를 기록하여 양자 메모리의 힘을 강조합니다.

- **Performance Highlights**: 추가적으로, 모든 고정된 두-결과 비일관 POVM 전략은 최대 오류 확률을 $	ext{exp}ig(- Oig(rac{NH_1}{2^n}ig)ig)$ 초과하는 것을 증명하여 하한을 설정했습니다. 이 프레임워크는 양자 기술에서 샘플링 병목 현상을 극복할 수 있는 구체적인 설계 원칙을 제공합니다.



### A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics (https://arxiv.org/abs/2502.14333)
- **What's New**: 최근 대규모 언어 모델(LLM) 기술의 발전으로 다단계 추론(multi-step reasoning) 능력을 향상시키는 방식이 주목받고 있습니다. 이러한 접근은 문제 해결을 다단계로 유도하여 더 큰 성과를 이끌어내며, 과거의 훈련 기반 전략보다 더 효과적인 결과를 보여주고 있습니다. 특히 수학 문제를 해결하는 과정에서 다단계 추론의 필요성이 대두되며, 이 논문은 해당 분야의 전반적인 피드백 전략을 조망합니다.

- **Technical Details**: LLM은 주어진 질문 Q에 대해 여러 사고 단계(rm1,…,rmn)와 최종 답변(am)을 생성합니다. 이를 위해 각 단계에 대한 피드백을 활용하는 다양한 전략이 연구되었으며, 특히 과정 보상(process rewards) 모델과 결과 보상(outcome rewards) 모델이 논의됩니다. 이러한 모델들은 LLM이 문제를 해결할 때 특정 단계에서의 올바른 추론을 장려하고 최종 결과의 정확성을 개선하는 데 중요합니다.

- **Performance Highlights**: 이 논문에서는 기존의 프롬프트 기반 전략이 아닌, 훈련 없이도 대응할 수 있는 피드백 사용 전략을 다루며 LLM의 성능을 극대화하려는 의도를 가지고 있습니다. 수학 문제 해결에 중점을 둔 여러 연구를 검토하며 기존의 접근 방식들과 차별화된 다단계 접근 방식을 도입했습니다. 마지막으로, 효율적인 피드백 전략을 통해 LLM의 복잡한 문제 해결 능력을 향상시키려는 목표를 설정하고 있습니다.



### Line Goes Up? Inherent Limitations of Benchmarks for Evaluating Large Language Models (https://arxiv.org/abs/2502.14318)
Comments:
          10 pages

- **What's New**: 이 논문은 최근 큰 언어 모델(LLMs)의 성능이 향상되었다는 주장을 도전하며, 평가 기준(benchmark)의 한계에 대해 논의하고 있습니다. 기존의 평가 방법으로는 LLM의 일반적인 인지 능력을 충분히 측정할 수 없다고 주장하고, 적대적 자극(adversarial stimuli)과 해석 가능성(interpretability) 기법이 대안적 평가 방법으로 효과적임을 설명합니다. 따라서, LLM의 성능을 신뢰할 수 있는 지표로 사용해서는 안 된다고 결론지었습니다.

- **Technical Details**: LLM은 transformer 아키텍처에 기반하여 대량의 텍스트 데이터를 학습하여 자연어와 상호작용합니다. 다양한 테스트를 통해 LLM의 성능을 평가하지만, 이러한 벤치마크는 실제 세계 작업을 반영하지 못할 수 있습니다. 특히, 새로운 벤치마크가 LLM의 훈련 데이터에 포함되어 역할과 가치를 잃는 문제가 발생하며, 이는 Goodhart의 법칙에 의해 설명될 수 있습니다.

- **Performance Highlights**: 연구 결과는 LLM이 많은 언어 및 논리 작업에서 높은 점수를 받을 수 있지만, 이는 실제 인지 능력을 반영하지 않을 수 있음을 보여주었습니다. LLM은 특정 벤치마크에 과도하게 최적화되어 성능이 빠르게 포화되는 경향이 있으며, 실제 작업에 대한 예측력을 방해합니다. 각 기법이 실제 작업과 얼마나 잘 연결되는지에 대한 연구가 부족하여 벤치마크의 유용성에 대한 신뢰가 떨어지고 있습니다.



### Textured 3D Regenerative Morphing with 3D Diffusion Prior (https://arxiv.org/abs/2502.14316)
- **What's New**: 이번 연구에서 우리는 텍스처가 있는 3D 모핑에서의 한계를 극복하기 위해 3D 확산 우선순위(3D diffusion prior)를 활용한 새로운 방법론을 제안합니다. 기존의 방법과 달리, 우리는 명시적인 대응 관계를 요구하지 않으면서 매끄럽고 그럴듯한 변환을 생성할 수 있습니다. 이를 통해 3D 객체 간의 변형 보다는 3D 배열을 재현하는 것이 가능해졌습니다.

- **Technical Details**: 우리의 접근법은 3D 확산 모델을 사용하여 특정 매개변수(levels)를 기반으로 소스와 목표 정보 간의 보간을 수행합니다. 특히, 초기 노이즈, 모델 파라미터, 조건적 특징에서의 정보를 통합하여 3D 제공 모델을 통해 재생성합니다. 또한, Attention Fusion 전략을 통해 매끄러운 모핑 시퀀스를 생성하고, 두 가지 전략(Token Reordering 및 Low-Frequency Enhancement)을 통해 생성된 3D 표면의 질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 다양한 객체 쌍 간의 3D 모핑에서 탁월한 매끄러움과 그럴듯함을 달성하였습니다. 특히 기존 방법이 직면한 두 가지 주요 과제를 극복하며, 텍스처가 있는 3D 표현을 효과적으로 다루는 첫 번째 일반 3D 확산 우선 순위를 사용한 모핑 방법으로 자리 매김할 수 있었습니다. 이러한 결과는 3D 모핑 분야에 새로운 가능성을 제시합니다.



### MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models (https://arxiv.org/abs/2502.14302)
Comments:
          Code and dataset are available at this https URL

- **What's New**: 이번 연구에서는 MedHallu라는 최초의 benchmark를 제안하여 의료 분야에서의 hallucination(환각) 탐지를 위한 기초 자료집을 제공합니다. 이 데이터셋은 PubMedQA에서 유래된 10,000개의 고품질 질문-답변 쌍으로 구성되어 있으며, 체계적인 파이프라인을 통해 환각된 답변이 생성되었습니다. 연구 결과, 최신의 LLM 모델들이 이 이진 환각 탐지 과제에서 높은 성과를 거두지 못하는 것으로 나타났습니다.

- **Technical Details**: MedHallu 데이터셋은 쉽게, 중간, 어렵게 감지하는 수준으로 구분됩니다. 또한, 양방향 내포 클러스터링(bidirectional entailment clustering)을 사용하여 감지하기 어려운 환각은 사실에 더 가까운 의미적 유사성을 가진다는 것을 보여줍니다. 도메인 전문 지식이 포함된 답변 및 '확실하지 않다'는 카테고리를 도입하면, 기존 기준에 비해 정밀도와 F1 점수가 최대 38% 향상되었습니다.

- **Performance Highlights**: 연구에서 의료 도메인에 맞게 조정된 LLM 모델은 일반 목적의 LLM 모델보다 낮은 성과를 보였습니다. 가장 좋은 성능을 보인 모델은 '어려운' 카테고리 환각 탐지에서 F1 점수가 0.625에 불과했습니다. 이 결과는 LLM이 생성하는 정보의 정밀도와 신뢰성을 높이기 위한 보다 세밀한 접근 방법의 필요성을 강조합니다.



### SEA-HELM: Southeast Asian Holistic Evaluation of Language Models (https://arxiv.org/abs/2502.14301)
- **What's New**: SEA-HELM(남동아시아 언어 모델의 포괄적 평가)는 남동아시아(SEA) 언어와 문화에 중점을 둔 새로운 평가 도구로, 기존 LLM의 비교 평가를 위한 철저한 다국어 및 다문화 벤치마크의 필요성이 높아진 가운데 등장했습니다. 이 평가 도구는 NLP Classics, LLM-specifics, SEA Linguistics, SEA Culture, Safety의 다섯 가지 핵심 기둥으로 구성되어 있습니다. 현재 필리핀어, 인도네시아어, 타밀어, 태국어, 베트남어를 지원하며, 사용자가 모델의 다국어 및 다문화 성능을 쉽게 이해할 수 있도록 설계된 SEA-HELM 리더보드도 함께 도입하였습니다.

- **Technical Details**: SEA-HELM은 각 SEA 언어에 대한 언어적 뉘앙스와 문화적 요소를 감안하여 폭넓은 작업을 포함하는 다섯 가지 평가 기둥을 바탕으로 구성되었습니다. 각 기둥은 NLP 고전, LLM 특화, SEA 언어학, SEA 문화, 안전성을 포함합니다. SEA-HELM은 지역 사회의 참여를 통해 현지 원어민이 데이터셋 기획 및 구축의 각 단계에 참여하도록 함으로써 언어적 정확성과 문화적 진정성을 보장합니다.

- **Performance Highlights**: SEA-HELM은 평가가 함께 수행되도록 데이터셋과 LLM 프롬프트를 통합하여 표준화된 비교를 가능하게 하고, 집계한 결과를 언어, 작업 및 모델별로 제시합니다. 이 평가 도구는 기존의 영어 안전 및 NLP 작업을 필리핀어, 인도네시아어, 타밀어, 태국어, 베트남어로 번역하고, 이루어질 사회적 기여와 보다 정교한 언어 진단을 위해 새로운 데이터셋을 개발하였습니다. SEA-HELM은 향후 다른 SEA 언어들에 대한 확장도 계획하고 있으며, 이는 남동아시아 언어에 대한 보다 정확하고 진정한 평가 기준을 제공합니다.



### An Evaluation of Sakana's AI Scientist for Autonomous Research: Wishful Thinking or an Emerging Reality Towards 'Artificial General Research Intelligence' (AGRI)? (https://arxiv.org/abs/2502.14297)
Comments:
          16 pages

- **What's New**: AI의 새로운 발전 단계인 Artificial General Research Intelligence (AGRI)의 중심에는 AI가 자율적으로 연구를 수행할 수 있는 능력이 있습니다. Sakana.ai의 AI Scientist는 연구 아이디어 생성, 실험 설계 및 결과 분석 등 연구 라이프사이클을 자동화할 수 있다고 주장하며 큰 관심을 받고 있습니다. 하지만 우리의 평가 결과, 실험의 절반가량이 실패했고, 수치나 결과에서 '환각(Hallucination)' 현상도 나타나는 등 여러 한계가 있음이 드러났습니다.

- **Technical Details**: AI Scientist는 초기 준비 작업을 제외하고는 연구 라이프사이클을 자율적으로 수행할 수 있다고 주장하지만, 실제로 사용자는 실험을 위한 파이프라인을 제공해야 합니다. 이는 AI의 자율성을 제한하는 요소로 작용합니다. 시스템은 대체로 저렴한 가격과 적은 인적 자원으로 연구 논문을 작성할 수 있지만, 문헌 검토와 실험 실행에서 다소 부족한 성과를 보여줍니다.

- **Performance Highlights**: AI Scientist는 AI 기반 연구 도구의 발전을 알리고 있으며, 연구 자동화의 미래를 열어갈 잠재력을 가지고 있습니다. 그러나 현재로서는 연구 목표를 완전히 달성하지 못하고 있으며, 신뢰성과 일반화 가능성에 대한 의문도 제기되고 있습니다. 인공지능 연구 시스템의 발전을 도모하기 위해, IR 커뮤니티는 이러한 도구와 함께 작업하는 방법을 모색해야 합니다.



### Graph Anomaly Detection via Adaptive Test-time Representation Learning across Out-of-Distribution Domains (https://arxiv.org/abs/2502.14293)
- **What's New**: 이 논문에서는 cross-domain Graph Anomaly Detection (GAD)을 위한 새로운 테스트-타임 훈련 프레임워크인 AdaGraph-T3를 제안합니다. 이는 기존의 지도 학습 기반 접근 방식이 가진 한계를 극복하고, 새로운 도메인에 적응하기 위해 self-supervised learning을 활용합니다. 고유한 특성으로, AdaGraph-T3는 도메인 불변적 속성을 포착하는 homophily 기반의 affinity score를 사용합니다.

- **Technical Details**: AdaGraph-T3는 네 가지 핵심 혁신을 소개합니다: 효과적인 self-supervision 방법, 메시지 패싱 동안 엣지 중요도 가중치를 동적으로 학습하는 attention 기반 메커니즘, 이종 특성을 처리하기 위한 도메인 특정 인코더, 그리고 클래스 불균형 문제를 해결하기 위한 클래스 인식 정규화입니다. 또한, TTT(테스트-타임 훈련) 기법을 통해 새로운 도메인에서의 적응을 가능하게 합니다.

- **Performance Highlights**: 다양한 교차 도메인 설정에서 실시한 실험 결과, AdaGraph-T3는 기존의 접근 방식에 비해 평균 6.6%의 AUROC와 7.9%의 AUPRC 향상을 달성하며, 많은 경우에서 우수한 성능을 보였습니다. 이는 AdaGraph-T3가 도메인 간 지식 이전을 효과적으로 수행할 수 있음을 보여줍니다.



### Correcting Noisy Multilabel Predictions: Modeling Label Noise through Latent Space Shifts (https://arxiv.org/abs/2502.14281)
- **What's New**: 이번 연구는 다중 레이블 분류(multilabel classification)에서 발생하는 노이즈 레이블(noisy label) 학습의 새로운 접근 방식을 제시합니다. 기존 연구들이 주로 단일 레이블의 다중 클래스 분류(multiclass classification)에 초점을 맞춘 반면, 우리는 다중 레이블 분류의 예측 결과를 수정하는 방법에 중점을 두고 있습니다. 이를 통해 처리성을 높이고 다른 노이즈 레이블 교정 기법과 결합해 성능 향상을 도모하고자 합니다.

- **Technical Details**: 우리는 레이블 노이즈가 잠재 변수(latent variable)의 확률적 변동에서 발생한다고 가정하여 이를 해결하는 데 깊은 생성 접근법(deep generative approaches)을 적용합니다. 모델은 관측된 데이터를 통해 학습하여 잠재 변수의 변화를 추적하고, 생성된 라벨 노이즈가 사실상의 라벨 패턴을 따르도록 설계되었습니다. 또한 비지도 학습(unsupervised learning) 및 반지도 학습(semi-supervised learning) 방법을 개발하여 모델의 효용성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 다양한 노이즈 레이블 설정에서 기존 방법들을 능가하는 성능을 보였음을 확인했습니다. LSNPC(Latent Space Noisy Prediction Calibration)라는 새로운 방법을 통해 노이즈 예측을 효과적으로 조정하며, 이는 다중 레이블 분류의 특수성을 잘 고려한 기술적 기여입니다. 또한, 민감도 분석(sensitivity analysis)과 절제 연구(ablation study)를 통해 모델의 내구성 강화를 입증하였습니다.



### EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts (https://arxiv.org/abs/2502.14280)
- **What's New**: 본 연구에서 소개하는 	extbf{EpMAN}은 장기 컨텍스트를 처리하기 위한 새로운 접근법으로, 에피소딕 메모리 모듈을 채택하여 의미적으로 관련된 컨텍스트 조각을 종합적으로 참조합니다. 이 방법은 자기 주의(self-attention) 메커니즘의 한계를 극복하기 위해 개발되었으며, 모델의 훈련 및 생성 과정에서 디코더의 자기 주의를 재조정하는 기능을 갖추고 있습니다. EpMAN은 이전의 RAG 기반 접근법보다 성능이 뛰어나며, 특히 16k에서 256k 토큰 범위에서 보다 강력하고 안정적인 결과를 보여줍니다.

- **Technical Details**: EpMAN은 입력 컨텍스트의 각 조각과 쿼리 간의 상대적 관련성을 평가하고, 이를 바탕으로 자기 주의를 재조정하여 긴 컨텍스트에서 정보를 효과적으로 처리를 가능하게 합니다. 이는 Kahneman의 이중 처리 이론(System 1과 System 2)의 영감을 받아, 빠르고 직관적인 자기 주의와 느리고 계산적인 사고 방식 간의 조화를 이루도록 설계되었습니다. EpMAN은 훈련 중에 관련 조각에 대한 주의를 추정할 때 노이즈를 도입하여 정확성을 높이는 방법도 포함하고 있습니다.

- **Performance Highlights**: EpMAN이 훈련된 LLM은 긴 컨텍스트 시나리오에서 사실 회상 및 단일 홉 질문 답변을 포함한 다양한 벤치마크에서 우수한 성능을 보입니다. 특히, 방해 요소나 혼동이 있는 경우에도, EpMAN을 적용한 모델은 자기 주의 및 RAG 프레임워크를 사용해 훈련된 모델에 비해 좋은 일반화 능력을 보여줍니다. 이 연구는 LLM의 긴 컨텍스트 처리에서의 성능 향상을 위한 효과적인 접근법을 제시하며, 에피소딕 메모리 활용의 중요성을 밝힙니다.



### STeCa: Step-level Trajectory Calibration for LLM Agent Learning (https://arxiv.org/abs/2502.14276)
- **What's New**: 본 논문에서는 Large Language Model (LLM) 기반 에이전트의 학습을 개선하기 위한 새로운 프레임워크인 Step-Level Trajectory Calibration (STeCa)를 제안합니다. 기존 연구는 전문가 시연에서의 행동 클로닝과 탐사 궤적 샘플링을 통한 선호 학습에 주로 집중했으나, 이는 긴 시간의 작업에서 비효율적인 행동이 누적되며 문제를 일으킵니다. 이에 따라, STeCa는 시기적절한 보정의 중요성을 강조하며 자동으로 보정 궤적을 구성하는 필요성을 제기합니다.

- **Technical Details**: STeCa는 탐사 과정에서 단계별 보상 비교를 통해 비효율적인 행동을 식별합니다. 이 프레임워크는 LLM 기반의 성찰을 사용하여 보정된 궤적을 구성하고, 이를 통해 에이전트는 향상된 의사결정 과정에서 학습할 수 있습니다. 보정된 궤적과 성공적인 궤적 데이터를 함께 사용하여 강화 학습을 수행합니다.

- **Performance Highlights**: 광범위한 실험 결과, STeCa는 기존의 방법들보다 상당한 성능 향상을 보였습니다. 특히, 단계별 보정이 에이전트가 여러 작업을 더 견고하게 완료할 수 있도록 도와줍니다. 이 연구 결과는 LLM 기반 에이전트의 학습 성능에 중요한 발전을 가져올 것으로 기대됩니다.



### LLM-EvRep: Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework (https://arxiv.org/abs/2502.14273)
Comments:
          6 pages, 2 figures,Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25)

- **What's New**: 이번 연구에서는 LLM(EvGen)이라는 이벤트 기반 인식 방법을 제안합니다. 이 방법은 기존의 복잡한 훈련 과정에 의존하지 않고, 이벤트 기반 비주얼 콘텐츠를 효율적으로 처리할 수 있는 가능성을 열어줍니다. LLM(EvRep)이라는 LLM 호환 이벤트 표현을 생성하여 이벤트 인식 성능을 향상시키는 데 주력하고 있습니다.

- **Technical Details**: LLM-EvGen은 자기 감독(self-supervised) 프레임워크를 사용하여 이벤트 표현을 생성합니다. 이 과정에서 생성된 표현은 의미적 일관성(semantic consistency)과 구조적 신뢰성(structural fidelity)을 모두 고려하여 조정됩니다. 연구는 N-ImageNet, N-Caltech101, N-MNIST의 세 가지 데이터 세트를 사용하여 수행되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 LLM-EvRep 방법이 이벤트-비디오(even-to-video) 방법인 E2VID보다 인식 작업에서 각각 15.93%, 0.82%, 50.21%의 성능 향상을 보였습니다. 이러한 결과는 LLM을 활용한 이벤트 인식의 가능성과 효율성을 보여줍니다.



### Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models (https://arxiv.org/abs/2502.14272)
Comments:
          Under review

- **What's New**: 이번 논문에서는 작은 언어 모델(SLM)을 인간의 가치에 맞추기 위해, 기존의 대형 언어 모델(LLM)에서 선호 지식을 증류하는 방법에 대한 한계를 다루고 있습니다. 기존 방법들이 쌍(pairwise) 응답 비교에 기반하여 선호 지식을 모델링하는 데 그쳤다면, 본 연구는 또 다른 접근 방식인 Preference-Aligned Distillation (PAD) 프레임워크를 제안합니다. PAD는 교사 모델의 선호 지식을 모든 가능한 선호에 대한 확률 분포로 모델링하여 더 정교한 감독 신호(supervisory signals)를 제공합니다.

- **Technical Details**: PAD 프레임워크는 세 가지 핵심 단계로 구성됩니다. 첫 번째 단계에서는 높은 온도(temperature)를 사용하여 다양한 응답을 샘플링하고, 두 번째 단계에서는 교사와 학생 모델의 보상을 계산하여 내재적 선호(intrinsic preference)를 구성합니다. 마지막으로 세 번째 단계에서는 보상에 기반하여 학생의 내재적 선호 분포를 교사의 선호와 정렬시키는 과정을 포함합니다. 이러한 과정은 언어 모델이 보상 함수로 작용할 수 있음을 보여주며, 모델의 내재적 선호를 효과적으로 캡처합니다.

- **Performance Highlights**: 실험 결과는 PAD가 기존 방법들보다 일관되게 높은 성능을 발휘하며, AlpacaEval 2와 Arena-Hard 벤치마크에서 20% 이상의 성능 향상을 달성했음을 보여줍니다. 또한, MT-Bench에서 	extsc{Gemma} 모델 가족을 이용한 훈련을 통해 학생 모델이 교사 모델을 초월하는 성과를 기록하며, PAD의 효과성을 더욱 입증하였습니다. 이러한 결과는 PAD가 인간의 선호를 보다 정교하게 캡처하고, SLM의 성능을 향상 시키는 중요한 기여를 하고 있음을 나타냅니다.



### MCQA-Eval: Efficient Confidence Evaluation in NLG with Gold-Standard Correctness Labels (https://arxiv.org/abs/2502.14268)
- **What's New**: 이 논문에서는 자연어 생성(NLG) 분야에서의 신뢰도 측정에 대한 평가 프레임워크인 MCQA-Eval을 소개합니다. MCQA-Eval은 기존의 정확도 함수에 의존하지 않고, 다중 선택형 데이터셋에서 제공하는 금 표준 정답 라벨을 활용하여 신뢰도를 평가합니다. 이러한 새로운 접근법으로 인해, 다양한 신뢰도 측정 방법 간의 체계적인 비교가 가능해지며, 효율적이고 신뢰할 수 있는 평가를 제공합니다.

- **Technical Details**: MCQA-Eval은 내부 상태 기반의 화이트 박스(white-box) 신뢰도 측정과 일관성 기반의 블랙 박스(black-box) 신뢰도 측정 모두를 지원하여, 다양한 방법론에 대한 통합된 평가 방법론을 제공합니다. 이 프레임워크는 민감한 정확도 레이블의 노이즈 문제를 해결하며, 기존 평가 방식이 지닌 제한을 극복하는 데 중점을 두고 설계되었습니다. 또한, 우리의 연구는 LLM(대형 언어 모델)과 널리 사용되는 QA(질문-답변) 데이터셋을 기반으로 다양한 실험을 통해 검증되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 MCQA-Eval은 기존 신뢰도 측정 방법들보다 효율적이고 더 신뢰할 수 있는 평가를 제공함을 보고합니다. 평가 과정에서 발생할 수 있는 오류를 최소화하여, 신뢰도 측정 방법의 평가 메트릭과 순위를 더 정확하게 반영합니다. 이를 통해 우리의 프레임워크는 NLG 분야에서 신뢰성 있는 신뢰도 평가를 위한 유용한 도구로 자리잡을 것으로 기대됩니다.



### EyeBench: A Call for More Rigorous Evaluation of Retinal Image Enhancemen (https://arxiv.org/abs/2502.14260)
- **What's New**: 본 논문에서는 EyeBench라는 새로운 포괄적 벤치마크를 제안하여, 망막 이미지 향상 방법을 평가하는 데 필요한 종합적인 평가 시스템을 제공하고 있습니다. 기존의 평가 메트릭이 실제 임상 연구와의 연계에서 부족함을 드러내고, 전문가의 프로토콜이 필요함을 지적하며 이러한 틀을 눈높이에 맞춘 다차원적 평가 접근법으로 보완하고자 합니다.

- **Technical Details**: EyeBench는 향상된 망막 이미지의 평가를 위한 다양한 다운스트림 작업을 포함하여, 질병 등급화, 혈관 분할 및 병변 분할과 같은 임상적으로 중요한 평가 항목들을 제시합니다. 우리는 전문가의 지도 아래 비너리 및 비비너리 방법 간의 철저한 비교를 촉진하기 위한 새로운 데이터셋을 개발하였고, 해당 데이터셋은 충분한 참조 조건에서 고유 처리 및 테스트 세트를 포함합니다.

- **Performance Highlights**: EyeBench의 다차원 평가는 의료 전문가가 올바른 향상 방법을 선택하는 데 도움을 줄 수 있으며, 특히 임상적으로 가치 있는 비비너리 방법에 대한 상세 분석도 제공합니다. 또한 기존 방법들이 직면한 도전 과제를 포괄적으로 분석하여 향후 연구 방향에 대한 통찰력을 제공합니다.



### Does Time Have Its Place? Temporal Heads: Where Language Models Recall Time-specific Information (https://arxiv.org/abs/2502.14258)
- **What's New**: 이 논문에서는 언어 모델이 시간에 따라 변화하는 사실을 처리하는 방법, 특히 특정 attention heads인 Temporal Heads의 존재를 발견했습니다. Temporal Heads는 시간 지식 처리를 주로 담당하며, 여러 모델에서 존재하나 위치가 다를 수 있습니다. 이들을 비활성화하면 시간 특정 지식 회상의 능력이 저하되는 반면, 일반적인 성능에는 큰 영향을 주지 않습니다.

- **Technical Details**: 주요 방법론은 Circuit Analysis를 활용하여 언어 모델의 내부 계산 과정을 재구성하는 것입니다. 이를 통해 특정 attention heads와 feed-forward layers에서 시간 정보의 처리 메커니즘을 분석했습니다. Temporal Heads는 숫자 조건과 텍스트 조건 모두에서 활성화되며, 단순한 숫자 표현을 넘어서는 광범위한 시간 차원을 인코딩하고 있음을 발견했습니다.

- **Performance Highlights**: Temporal Heads의 영향을 분석한 결과, 이들을 비활성화하는 것이 시간에 특정한 정보의 정확도를 크게 감소시켰습니다. 연구 결과는 Temporal Heads가 시간 감응 지식의 인코딩 및 수정에 중요한 역할을 한다는 것을示しています. 이 연구는 언어 모델의 시간 지식 능력을 향상시키기 위한 포괄적인 기초를 제공합니다.



### Effects of Prompt Length on Domain-specific Tasks for Large Language Models (https://arxiv.org/abs/2502.14255)
- **What's New**: 최근 대형 언어 모델(LLM)의 개발이 활발히 진행되고 있으며, 이 모델들이 특정 도메인 과제(예: 금융 감정 분석)에서의 효과성을 규명하기 위한 새로운 연구가 이루어지고 있다. 본 논문은 입력 프롬프트의 길이가 LLM의 도메인 특정 작업 수행 능력에 미치는 영향을 체계적으로 조사하고, 이를 통해 프롬프트 엔지니어링의 중요성을 강조하고 있다. 이는 LLM이 다양한 과제에서 일관된 성과를 내기 위해 어떻게 보조적인 정보를 활용하는지를 이해하는 데 기여할 것이다.

- **Technical Details**: 연구에서는 아홉 개의 도메인 특정 과제에 대해 LLM이 수행하는 실험을 진행하였으며, 각 과제의 프롬프트 길이를 조정하여 모델 성능에 미치는 영향을 분석하였다. 이들 과제에는 통화 정책 이해(MPU), 사용자 의도(UI), 대화 도메인(CD), 쿼리 의도 분류(QIC), 풍자 감지(SD), 감정 식별(EI), 금융 감정 분석(FSA), 기술 시스템 동작(TSB), 그리고 질병 탐지가 포함된다. 또한, 프롬프트의 기본 길이와 짧은/긴 명령 설정을 통해 모델의 반응 품질을 평가하였다.

- **Performance Highlights**: 실험 결과, LLM은 도메인 지식이 부족할 경우 특정 과제 수행에 어려움을 겪는 것으로 나타났으며, 긴 프롬프트가 도메인 배경 지식을 제공하여 성능 향상에 긍정적인 영향을 미친다는 것을 보여주었다. 그러나 이러한 도움에도 불구하고, LLM의 성능은 여전히 인간 평균 값에 미치지 못하는 결과를 보였으며, F1-score가 1.0보다 먼 수치로 나타났다. 이는 도메인 특정 작업에서 성공적인 성능을 위해서는 여전히 인간 전문가의 역할이 필요함을 시사한다.



### Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation (https://arxiv.org/abs/2502.14254)
- **What's New**: 최근 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)의 발전으로, 에이전트가 이들을 활용하여 필연적으로 사람과의 상호작용에서 더욱 효율적으로 탐색할 수 있는 도구가 마련되었습니다. 새로운 방법론은 글로벌 메모리(global memory)에서 작업 관련 단서를 적응적으로 검색하고 에이전트의 자아 중심적 관찰과 통합하여 복잡한 환경에서의 공간적 추론(spatial reasoning)과 의사 결정을 개선합니다.

- **Technical Details**: 이 논문에서 제안하는 비전-언어 모델 기반 탐색 프레임워크는 기존의 LLM 기반 접근 방식의 기하학적 정보 손실 문제를 해결합니다. 주요 기술은 글로벌 맥락 정보를 동적으로 정렬하여 로컬 인식(local perception)과 통합하여 긴 수평 과제에서 공간적 추론을 강화하는 것입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 객체 탐색(task)에서 이전의 최신 기술(state-of-the-art) 접근 방식들을 능가하며, 유용하고 확장 가능한 솔루션을 제공합니다. 이를 통해 복잡한 환경에서의 의사 결정 최적화를 가능하게 합니다.



### Pandora3D: A Comprehensive Framework for High-Quality 3D Shape and Texture Generation (https://arxiv.org/abs/2502.14247)
Comments:
          Tencent XR 3D Gen

- **What's New**: 이 보고서는 다양한 입력 프롬프트에서 고품질 3D 형태 및 텍스처를 생성하기 위한 포괄적인 프레임워크인 Pandora3D를 소개합니다. 이 프레임워크는 3D 형태 생성과 텍스처 생성으로 구성되어 있으며, 각 프로세스는 고급 신경 아키텍처를 활용하여 다양한 입력 형식을 효과적으로 처리할 수 있습니다. 특히, Variational Autoencoder (VAE)와 Diffusion Network를 이용한 개선된 모델이 도입되었습니다.

- **Technical Details**: Pandora3D의 3D 형태 생성 파이프라인은 VAE를 사용하여 3D 기하학을 잠재 공간(latent space)으로 인코딩하며, Diffusion Network가 입력 프롬프트에 조건화된 잠재 표현을 생성합니다. 이 과정은 CLAY, Craftsman, LAM3D의 구조를 바탕으로 한 모델 개조를 포함하며, 점진적 샘플링 프로세스를 통해 높은 기하학적 복잡성을 포착하고, 세밀한 디테일을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 보고서에 따르면, Pandora3D는 다양한 입력 형식을 처리하면서 고품질의 3D 콘텐츠를 생성하는 데 성공하였습니다. 다단계 텍스처 생성 프로세스에서는 픽셀 수준의 일관성을 유지하기 위한 novel consistency scheduler가 각 단계에 통합되어 무결한 3D 표현을 보장합니다. 이러한 성과는 시스템 아키텍처, 실험 결과 및 향후 가능한 발전 방향에 대한 상세한 설명과 함께 제공됩니다.



### OG-Gaussian: Occupancy Based Street Gaussians for Autonomous Driving (https://arxiv.org/abs/2502.14235)
- **What's New**: 본 논문에서는 OG-Gaussian이라는 새로운 접근 방식을 제안하여 자율주행 장면 재구성을 위한 Occupancy Grid (OG)를 활용합니다. 기존 LiDAR 포인트 클라우드 대신 주변 카메라 이미지를 사용해 OG를 생성을 통해 동적 객체와 정적 배경을 효과적으로 분리합니다. 이 방식은 고비용의 LiDAR에 의존하지 않고도 고품질의 3D 장면을 신속하게 재구성할 수 있는 방법을 제공합니다.

- **Technical Details**: OG-Gaussian은 Occupancy Prediction Network (ONet)를 활용해 주변 세상을 나타내는 VOC(명세 그리드)를 생성하여 동적 차량과 정적 배경을 구분합니다. 이 후, 배경 거리의 OG에서 포인트 클라우드를 추출하고, 이들을 최적화 가능한 Gaussian 타원체 형태로 변환합니다. 학습 가능한 매개변수를 정의하여 동적 객체의 위치와 자세를 최적화해 감지 및 추적할 수 있는 기법을 제안합니다.

- **Performance Highlights**: Waymo Open 데이터셋에서 수행된 실험에서는 OG-Gaussian이 평균 PSNR 35.13 및 렌더링 속도 143 FPS를 달성하여 현재 최첨단 재구성 품질 및 속도와 동등한 결과를 보여주었습니다. 또한, 처리된 OG를 사용한 사전 세팅으로 효율성을 입증하는 억제 연구도 수행되어, 비용과 시간을 절감하며 자율주행 장면 재구성의 새로운 길을 열었습니다.



### SleepGMUformer: A gated multimodal temporal neural network for sleep staging (https://arxiv.org/abs/2502.14227)
- **What's New**: 이번 논문에서는 심박수, 움직임, 걸음 수, EEG (Fpz-Cz, Pz-Oz), EOG 데이터를 포함한 다중 도메인 수면 데이터를 처리하기 위한 gated multimodal temporal neural network(SleepGMUformer)를 제안합니다. 이 모델은 다양한 생리적 신호의 기여도를 동적으로 통합하여, 수면 스테이징의 정확성을 높이는 데 초점을 맞추고 있습니다. 기존의 반복적인 수면 데이터 처리 방식을 개선해, 정밀한 분석을 가능하게 함으로써 저자원 환경에서도 유용하게 활용될 수 있습니다.

- **Technical Details**: SleepGMUformer는 사전 처리 모듈, 특성 추출 모듈, 동적 융합 모듈로 구성되어 수면 데이터를 더욱 효과적으로 분석합니다. EEG 데이터의 저주파 트렌드를 제거하고, 착용 가능한 장치에서 기록된 시간 연속 데이터의 결측치를 처리하여 데이터 품질을 향상시킵니다. 이 모델은 Clinically relevant한 방식으로 PSG와 웨어러블 장치 데이터의 결합을 통해 시간이 결합된 다양한 생리적 데이터의 해석과 정확성을 보장합니다.

- **Performance Highlights**: 이 모델은 SleepEDF-78 데이터셋에서 85.03%의 정확도, WristHR-Motion-Sleep 데이터셋에서 94.54%의 정확도를 기록하며, 기존의 최고 수준 모델보다 1.00%-4.00% 더 나은 성능을 보여 주목받고 있습니다. 여러 모달리티 정보의 통합 덕분에 SleepGMUformer는 수면 스테이징을 위한 효율적이고 정밀한 도구로 자리잡을 전망입니다. 이는 향후 의료, 개인 건강 관리 및 비즈니스 등 다양한 분야에서 응용될 수 있습니다.



### Enhancing Pavement Sensor Data Acquisition for AI-Driven Transportation Research (https://arxiv.org/abs/2502.14222)
Comments:
          This paper was accepted for presentation at the 104th TRB Annual Meeting, held on January 5-9, 2025, in Washington, D.C., and was presented during the poster session on January 8, 2025

- **What's New**: 이 논문은 인공지능의 발전에 따른 교통 연구의 변화에서 센서 데이터 관리의 효율적인 전략이 중요하다는 점을 강조합니다. 새로운 애플리케이션의 등장으로 인해 실시간 데이터 스트림과 보관된 정적 데이터 모두를 포함한 포괄적인 관리 지침을 제시합니다.

- **Technical Details**: 실시간 시스템 아키텍처는 데이터 수집 시스템(DAQ)과 다양한 애플리케이션을 통합합니다. Avena 소프트웨어 플랫폼과 NATS 메시징 시스템을 사용하여 안전한 데이터 교환을 보장하며, TimescaleDB와 같은 데이터베이스를 통해 체계적인 저장이 가능합니다.

- **Performance Highlights**: 이 연구는 INDOT의 I-65와 I-69 그린필드 지역을 포함하는 실제 사례 연구에 대한 제안 사항을 적용했습니다. 센서 메트릭의 지속적인 생성 및 모니터링을 위해 Campbell Scientific DAQ 시스템을 사용했으며, 결과적으로 제안된 지침의 효과성을 강조하고 연구 프로젝트에서의 채택을 촉구합니다.



### Rethinking Spiking Neural Networks from an Ensemble Learning Perspectiv (https://arxiv.org/abs/2502.14218)
Comments:
          Published as a conference paper at ICLR 2025

- **What's New**: 이번 연구에서는 스파이킹 신경망(Spiking Neural Networks, SNNs)을 시간적 서브네트워크의 집합으로 보고, 초기 상태(신경세포의 막 전위)의 과도한 차이가 성능에 미치는 영향을 강조합니다. 우리는 막 전위의 일관성을 촉진하여 전체적인 안정성과 성능을 향상시키는 방법론을 제안하고, 이 과정이 정보 전파와 경량화에 어떻게 기여하는지를 설명합니다. 또한, 기존의 구조를 변경하지 않고도 최소한의 수정으로 진행 가능하다는 점에서 방법의 일반성과 일관된 성능 향상을 입증합니다.

- **Technical Details**: 이 연구에서는 막 전위 스무딩(membrane potential smoothing)과 시간적으로 인접한 서브네트워크의 가이드를 통해 초기 막 전위 분포와 출력을 일관되게 유지하는 방법을 제안합니다. 스무딩 기법은 각 타임스텝에서 이전 상태를 기반으로 막 전위를 조정하여 출력을 안정화시키고, 정보 및 그래디언트 전파를 원활하게 합니다. 우리 방법은 스파이킹 뉴런을 변경하지 않고도 기존 네트워크의 구조를 그대로 유지할 수 있어, 높은 일반화를 제공합니다.

- **Performance Highlights**: 우리는 1D 음성 인식, 2D 객체 인식, 및 3D 포인트 클라우드 분류 작업에 대한 광범위한 실험을 통해 제안한 방법의 효과성을 입증하였습니다. 특히, 도전적인 CIFAR10-DVS 데이터셋에서 단 4개의 타임스텝만으로 83.20%의 정확도를 달성하였으며, 이는 SNNs의 가능성을 제시하는 귀중한 통찰력을 제공합니다.



### Towards Secure Program Partitioning for Smart Contracts with LLM's In-Context Learning (https://arxiv.org/abs/2502.14215)
- **What's New**: PartitionGPT는 민감한 정보 유출을 방지하기 위해 고안된 LLM 기반의 최초의 접근 방식으로, 스마트 계약을 특권 코드베이스와 일반 코드베이스로 분할합니다. 이 방법은 몇 가지 주석이 달린 민감한 데이터 변수에 의해 안내되며, 코드에서의 민감한 작업을 안전하게 실행하는 것을 목표로 합니다. 스마트 계약의 안전성과 유용성을 높이는 이 혁신적인 프레임워크는 기존 방법의 한계를 극복합니다.

- **Technical Details**: PartitionGPT는 프로그램 슬라이스(Program Slicing)와 대형 언어 모델(LLM)의 맥락 학습 기능을 결합하여 스마트 계약에서 민감한 작업과 비민감한 작업을 자동으로 분리합니다. 이를 통해 사용자가 수동으로 주석을 달 필요를 줄이고, 컴파일 가능한 세분화된 프로그램 파티션을 생성합니다. 또한, Functional Equivalence Checker를 통해 원래 코드와 파티션 코드의 기능적 동등성을 형식적으로 검증하여 안전성을 보장합니다.

- **Performance Highlights**: PartitionGPT는 18개의 주석이 달린 스마트 계약에서 99개의 민감한 기능을 평가하여 78%의 성공률을 달성하고, 기능 수준 파티닝 방식에 비해 약 30%의 코드를 줄였습니다. 또한, 실제 조작 공격에 대한 방어 성능을 입증하여 9건 중 8건의 공격을 효과적으로 막았습니다. 이 연구는 PartitionGPT의 실행 시간이 적정하다는 것을 보여주며, TEE(Trusted Execution Environment) 기반의 실행 환경에서의 성능을 평가했습니다.



### Accurate Forgetting for Heterogeneous Federated Continual Learning (https://arxiv.org/abs/2502.14205)
Comments:
          published in ICLR 2024

- **What's New**: 최근 연합 학습(federated learning, FL)의 관심이 증가하고 있으며, 클라이언트가 순차적 학습(sequential learning)을 수행하는 맥락이 추가적으로 탐구되고 있습니다. 본 논문에서는 연합 지속 학습(federated continual learning, FCL)의 개념을 도입하여 서로 관련이 없거나 심지어 적대적인 데이터/작업을 다루는 상황을 고려했습니다. 전통적인 CL 기술들이 이전의 지식을 완전히 활용하는 데 중점을 두는 반면, 우리는 편향된 정보를 잊는 것이 유익할 수 있음을 발견했습니다.

- **Technical Details**: 본 논문에서는 정확한 잊기(accurate forgetting, AF)라는 새로운 개념을 제안하고, 연합 네트워크에서 이전 지식을 선택적으로 활용하는 방식의 새로운 생성적-재생(generative-replay) 방법론을 개발했습니다. 이는 모델이 비특이적인 생성 재생을 수행하는 대신, 피처(space)에서의 상관관계를 통해 이전 지식을 신중하게 활용할 수 있도록 합니다. 노멀라이징 흐름(normalizing flow, NF) 모델에 기반한 확률적 프레임워크를 사용하여 과거 지식의 신뢰성을 정량화합니다.

- **Performance Highlights**: 종합 실험을 통해 제안한 방법인 AF-FCL이 여러 벤치마크 데이터셋에서 기존 방법들과 비교해 월등한 성능을 나타내는 것을 입증했습니다. 주요 기여로는, FCL 설정에서의 편향된 피처를 기억하는 것이 해로울 수 있음을 인식하고, 상관관계 추정을 통해 오류 정보를 적응적으로 완화하는 방안을 제시하였습니다. 이 연구는 데이터의 통계적 이질성과 연합 시나리오 내에서 피처 편향 문제를 해결하기 위한 새로운 접근을 제공합니다.



### On-the-fly Preference Alignment via Principle-Guided Decoding (https://arxiv.org/abs/2502.14204)
Comments:
          Accepted to ICLR 2025

- **What's New**: 신규 방법인 OPAD(On-the-fly Preference Alignment via Principle-Guided Decoding)는 모델 추론 과정에서 직접적으로 인간의 선호도와 일치하도록 조정하는 접근법을 제안합니다. 기존의 기술들이 훈련 단계에서 최적화에 주력했다면, OPAD는 미세 조정 없이도 원칙에 기반한 보상 함수를 설계하여 모델의 출력을 수정합니다. 이 방법은 기존의 대규모 언어 모델을 보다 효율적으로 동작할 수 있도록 하여, 추론 시에 원칙을 준수하면서도 재훈련의 계산 비용을 줄입니다.

- **Technical Details**: OPAD는 의미적으로 융통성이 있는 보상 함수를 통해 모델의 출력을 조정하는 방식입니다. 이 방식은 Kullback-Leibler(KL) 발산을 최소화하는 대신 제약된 정책과 제약이 없는 정책 간의 KL 발산을 최대화하는 대체 목표를 사용합니다. 이렇게 함으로써 모델의 반응을 원칙에 맞게 조정하며, 각 시간 단계마다 예상 토큰을 조정하는 튜닝 없는 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과, OPAD는 일반적인 정렬 작업과 개인화된 정렬 작업 모두에서 경쟁력 있는 성과를 기록했습니다. 또한, perplexity, diversity, ROUGE 점수와 같은 자동 평가 메트릭에서도 우수한 성능을 나타내어 OPAD가 목표 원칙에 부합하는 모델 행태 조정에 뛰어나다는 것을 시사합니다. 전통적인 RLHF 방식보다 OPAD가 더 효과적으로 모델의 행동을 조정할 수 있음을 보여주었습니다.



### Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions (https://arxiv.org/abs/2502.14202)
- **What's New**: 이번 연구는 소프트웨어 개발에 대한 대화형 LLM(대형 언어 모델)의 확대 사용과 관련된 보안 문제를 다루고 있습니다. ChatGPT가 개발자에게 맥락에 맞는 보안 정보를 제공하여 안전한 코딩 관행을 촉진할 잠재력을 보이는 것을 기반으로 합니다. 연구에서는 Claude 3, GPT-4 및 Llama 3와 같은 세 가지 주요 LLM의 보안 인식 수준을 평가하였습니다.

- **Technical Details**: 연구에서는 Stack Overflow에서 수집된 질문을 이용하여 LLM 응답의 보안 인식을 평가합니다. 300개의 질문 데이터셋을 두 그룹으로 나누어 보안 취약점을 명시한 경우와 명시하지 않은 경우로 나누어 분석하였으며, LLM이 보안 문제를 능동적으로 인식하고 경고하는지 여부를 살폈습니다. RQ1을 통해 LLM의 보안 인식 정도를 분석하고, RQ2를 통해 보안 경고에 포함된 정보의 유형에 대한 질적 분석을 수행하였습니다.

- **Performance Highlights**: 결과에 따르면, 세 가지 LLM 모두 취약점을 정확하게 감지하고 경고하는 데 어려움을 겪으며, 감지율은 12.6%에서 40%에 불과했습니다. 또한, LLM이 보안 경고를 발행할 때, Stack Overflow 응답보다 취약점의 원인, 악용 및 수정 방법에 대한 정보를 더 많이 제공하는 경향이 있었습니다. 마지막으로, 보안 인식을 높이기 위한 CLI 기반의 프롬프트 도구를 제시하여 LLM 응답의 보안성을 크게 향상시킬 가능성을 보여주었습니다.



### Adaptive Sparsified Graph Learning Framework for Vessel Behavior Anomalies (https://arxiv.org/abs/2502.14197)
Comments:
          Anomaly Detection in Scientific Domains AAAI Workshop

- **What's New**: 본 논문에서는 기존의 정의된 그래프에 의존하지 않고, 시간 정보를 노드로 표현하는 혁신적인 그래프 표현 방식을 도입하여 해양 환경의 동적 특성을 모델링할 수 있는 방법을 제시합니다. 새로운 그래프 구조는 그래프 엣지를 통해 시간적 의존성을 명확히 포착할 수 있도록 하며, 다중 선박 간의 상호작용을 효과적으로 캡처할 수 있습니다. 이 접근 방식을 통해 고유한이상 탐지 기능을 구현하게 됩니다.

- **Technical Details**: 제안하는 방법은 Graph Convolutional Network (GCN) 레이어를 사용하여 그래프를 처리하고, Variational Graph Autoencoder (VGAE)를 통해 재구성을 수행하며, 이를 통해 이상 탐지를 위한 예측이 가능합니다. AIS 데이터에서 유도된 다중 선박 그래프는 OPTICS 클러스터링 알고리즘을 통해 정의되며, 이는 각 시간대의 위도 및 경도를 기반으로 선박을 그룹화함으로써 수행됩니다. 이를 통해 전통적인 순환 신경망(RNN) 대신 간단한 다층 퍼셉트론(MLP) 모델을 활용한 시간 예측도 가능해집니다.

- **Performance Highlights**: 새로운 그래프 기반 방식은 기존의 이상 탐지 성능을 크게 향상시키며, 특히 해양 환경의 복잡한 이동 패턴을 보다 정교하게 포착합니다. 이 연구에서는 표준 경로에서 이탈하는 선박의 행동을 감지하는 데 중점을 두어, 잠재적인 해양 위험을 효과적으로 식별할 수 있는 강력한 프레임워크를 제시합니다. 최종적으로, 이는 여러 가지 해양 사고의 근본 원인을 밝혀내는 데 기여할 수 있습니다.



### Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models (https://arxiv.org/abs/2502.14191)
Comments:
          Dataset available at this https URL

- **What's New**: 이번 논문에서 우리는 언어-비전 모델(Vision-Language Models, VLMs)을 위한 멀티모달 보상 모델 평가를 위한 새로운 베치마크, Multimodal RewardBench를 소개합니다. 이 벤치마크는 일반적 정확성, 선호도, 지식, 추론, 안전성, 시각적 질문-응답(Visual Question Answering, VQA)의 여섯 가지 분야를 포괄하며, 5,211개 텍스트-이미지 프롬프트에 대한 전문가 주석으로 구성되어 있습니다. 기존의 VLM 보상 모델 평가가 텍스트 모드에 국한되어 있었던 반면, 이 베치마크는 멀티모달 환경에서의 평가를 가능하게 합니다.

- **Technical Details**: Multimodal RewardBench는 프롬프트와 선택된 반응, 거부된 반응의 트리플릿(세 쌍의 데이터)을 사용하여 VLM 보상 모델의 성능을 평가하는 데 중점을 둡니다. 평가에는 여러 VLM 모델이 포함되며, 각각의 영역별로 정확한 응답과 비정확한 응답 간의 구별이 가능하도록 구성되었습니다. 연구 결과, 상위 성능 모델인 Gemini 1.5 Pro와 Claude 3.5 Sonnet이 각각 72%의 전반적인 정확도를 기록하였으나, 대부분의 모델이 추론 및 안전성 분야에서 어려움을 겪고 있는 것으로 나타났습니다.

- **Performance Highlights**: 대부분의 VLM 보상 모델이 무작위 추측(50% 정확도)을 초과하여 성능을 보였으나, 여전히 인간의 성능 수준에는 미치지 못하고 있습니다. 특히, 수학 및 코딩 관련 추론 과제와 독성 감지 같은 안전성 과제에서 많은 모델이 어려움을 겪고 있는 점이 두드러졌습니다. 이러한 결과는 Multimodal RewardBench가 멀티모달 보상 모델 개발을 위한 도전적인 테스트베드임을 시사합니다.



### Type 1 Diabetes Management using GLIMMER: Glucose Level Indicator Model with Modified Error Ra (https://arxiv.org/abs/2502.14183)
- **What's New**: 논문에서는 GLIMMER라는 새로운 기계 학습 모델을 소개하여, 혈당 수치를 예측하는 방법을 혁신적으로 개선하였습니다. GLIMMER는 혈당 값을 정상과 비정상 범주로 분류하고, 특히 비정상 사건에서 정확도를 우선시하는 맞춤형 손실 함수를 사용하여 예측의 정확도를 높입니다. 이러한 접근 방식은 Type 1 Diabetes (T1D) 관리에서 환자 안전을 기하는 데 핵심적인 역할을 합니다.

- **Technical Details**: GLIMMER는 연속 혈당 모니터링(CGM) 데이터, 인슐린 용량 및 식사 입력을 주요 특징으로 사용하여 혈당 수치를 지속적으로 예측합니다. 이 모델은 실제 데이터 세트를 기반으로 하여, 예측 시 발생하는 오류를 정밀하게 낮추고, 비정상 혈당 사건의 예측 정확도를 크게 향상시키는 데 집중합니다. 저자는 25명의 T1D 환자에서 약 26,707시간의 데이터를 수집하여 GLIMMER의 효능을 평가하였습니다.

- **Performance Highlights**: GLIMMER는 다음 시간의 혈당 값을 예측하면서 RMSE(root mean square error) 23.97 (+/-3.77) mg/dL 및 MAE(mean absolute error) 15.83 (+/-2.09) mg/dL의 성능을 달성하였습니다. 이러한 결과는 이전에 보고된 오류율 대비 각각 23% 및 31% 향상된 것으로, T1D 관리에서의 효과적인 혈당 조절 가능성을 보여줍니다.



### A modal logic translation of the AGM axioms for belief revision (https://arxiv.org/abs/2502.14176)
Comments:
          19 pages, 3 figures

- **What's New**: 이 논문은 Bonanno(2025)의 분석을 기반으로 한 간단한 modal logic(모달 논리)을 소개합니다. 이 논리에는 unimodal belief operator(단일 모달 신념 연산자), bimodal conditional operator(이원 모달 조건부 연산자), unimodal global operator(단일 모달 전역 연산자) 등이 포함되어 있습니다. 각 AGM axiom(AGM 공리)에 대해 해당하는 modal axiom(모달 공리)을 제공하여 두 이론 간의 연관성을 수립합니다.

- **Technical Details**: 이 논문은 AGM belief revision(AGM 신념 수정)에 중점을 두고, 새로운 의미론을 통해 각 AGM axiom과 modal axiom 간의 상관관계를 설정합니다. 감사의 신념 집합 K는 상태 s에서 믿어지는 공식들의 집합으로 정의되며, 수정된 신념 집합은 입력으로 제공되는 공식 ϕ에 의해 유도되는 조건부의 결과로 설명됩니다. 이 과정에서 Kripke-Lewis frames(크립케-루이스 프레임)의 성질을 활용하여 일반적인 모달 논리 공리들을 도출합니다.

- **Performance Highlights**: 제안된 모달 논리는 AGM 공리의 각 성질과 일치하는 새롭고 유용한 공리들을 제공합니다. 그리고 이러한 성질들은 모달 논리의 표현력이 AGM 신념 수정의 논리적 기초와 어떻게 상호작용하는지를 보여줍니다. 결과적으로, 이 연구는 신념 수정의 이해를 심화시키고 다양한 상황에서의 적용 가능성을 높입니다.



### Weighted Low-rank Approximation via Stochastic Gradient Descent on Manifolds (https://arxiv.org/abs/2502.14174)
- **What's New**: 이번 연구에서는 매니폴드에서의 확률적 경량 강하법(stochastic gradient descent) 통해 정규화된 가중 저차 근사(regualarized weighted low-rank approximation) 문제를 해결했습니다. 연구진은 이러한 경량 강하법의 수렴(convergence)을 보장하기 위해 매니폴드에서의 수렴 정리를 수립하였습니다. 이 접근법은 특히 confinement을 허용하는 retraction 기반의 확률적 경량 강하에 적용됩니다.

- **Technical Details**: 우리가 제안한 알고리즘은 Netflix Prize의 훈련 데이터셋(sample data)에서 기존의 유클리드 공간(Euclidean spaces)에서의 확률적 경량 강하법보다 우수한 성능을 보여줍니다. 본 논문에서는 이 매니폴드에서의 가속 선형 검색(accelerated line search) 방법과 유클리드 공간에서의 기존 방법을 비교하였습니다. 매니폴드에서의 수렴 이론은 수학적 비약(mathematical leap)으로 기여하며, 향후 연구에 중요한 기반이 될 것입니다.

- **Performance Highlights**: 실험 결과, 새롭게 제안된 매니폴드 기반의 알고리즘은 기존 방식에 비해 더 나은 성능을 보였습니다. 특히, Netflix Prize 데이터에서의 성능 개선은 실제 문제 해결에 있어 큰 가능성을 보여주고 있습니다. 이 연구는 매니폴드 기하학을 적용한 경량 근사의 새로운 지평을 열어주며, 이는 데이터 과학 및 머신러닝 분야에서의 응용 가능성을 높입니다.



### Efficient Inverse Multiagent Learning (https://arxiv.org/abs/2502.14160)
Comments:
          Paper was submitted to the International Conference on Learning Representations (2024) under the title of "Generative Adversarial Inverse Multiagent Learning", and renamed for the camera-ready submission as "Efficient Inverse Multiagent Learning"

- **What's New**: 이번 연구에서는 역 게임 이론(inverse game theory)과 역 다중 에이전트 학습(inverse multiagent learning)을 연구하였습니다. 이 과정에서 게임의 보상 함수(payoff functions)의 매개변수를 찾는 것을 목표로 하며, 기대 행동(expected behavior) 또는 샘플링된 행동(sampled behavior)이 균형(equilibrium)을 이루도록 합니다. 이를 위해, 우리는 이 문제를 생성적-대립적 최적화(generative-adversarial optimization) 문제로 공식화하고, 이를 해결하기 위한 다항식 시간(polynomial-time) 알고리즘을 개발하였습니다.

- **Technical Details**: 우리가 개발한 알고리즘은 첫 번째 방법(first-order oracle)을 사용하는 정확한 접근 방식과 확률적(stochastic) 접근 방식을 통해 수행됩니다. 또한, 이 연구에서는 다중 에이전트 시뮬라클 학습(inverse multiagent simulacral learning)을 해결하기 위해 사용되는 방법을 다항식 시간과 샘플 수로 확장하였습니다. 이 문제에서는 주어진 관찰(observations)을 기대값(value)에서 재현(replicate)하는 매개변수와 연관된 균형을 찾습니다.

- **Performance Highlights**: 우리의 접근 방식은 스페인 전력 시장(Spanish electricity markets)에서 시간적 데이터(time-series data)를 바탕으로 가격을 예측하는 데 있어 널리 사용되는 ARIMA 방법에 비해 성능이 우수한 것으로 나타났습니다. 이는 역 게임 이론 및 다중 에이전트 학습 분야에서의 발전을 보여줍니다.



### PitVQA++: Vector Matrix-Low-Rank Adaptation for Open-Ended Visual Question Answering in Pituitary Surgery (https://arxiv.org/abs/2502.14149)
Comments:
          9 pages

- **What's New**: 이 논문은 Vision-Language Models (VLMs)을 수술 방식에 통합하여 실시간 수술 결정 지원을 제공하는 새로운 개념을 소개합니다. 특히, PitVQA++라는 새로운 데이터셋과 Vector-MoLoRA라는 혁신적인 VLM 파인튜닝 방법론을 제시하여, 기존 VQA 시스템의 한계를 극복하려 합니다. 이 연구는 특히 내비게이션과 의사결정이 중요한 내비게이팅 피수술에 중점을 두고 있으며, 여기서 제안된 데이터셋은 약 101,803프레임과 745,972개의 질문-응답 쌍으로 구성되어 있습니다.

- **Technical Details**: Vector-MoLoRA는 깊은 신경망의 계층 구조를 고려하여 초기 계층에 더 많은 파라미터를 할당하고 이후 계층에서는 점차 감소시키는 행렬 저순위 조정 전략을 사용합니다. 이 접근법은 LoRA와 MoRA의 원리를 결합하여, 파인튜닝 과정에서의 재해 기억상실 (catastrophic forgetting)을 완화시키고 성능을 향상시키는 데 목적을 두고 있습니다. 또한, 이 연구에서는 Open-Ended PitVQA와 EndoVis18-VQA 데이터세트에서 이 방법의 유효성을 검증하였습니다.

- **Performance Highlights**: Vector-MoLoRA는 기존의 최신 방법들과 비교하여 성능을 크게 향상시키며, 불확실한 예측을 처리하는 데 있어 신뢰성과 안정성을 높이는 것으로 확인되었습니다. 또한, 위험-커버리지 분석을 통해 의료 전문가의 불확실한 샘플에 대한 의뢰 수가 줄어드는 결과를 보였습니다. 이러한 접근은 피수술 교육과 실시간 지원 모두에서 중요한 발전을 이룰 것으로 기대됩니다.



### Multi-Agent Risks from Advanced AI (https://arxiv.org/abs/2502.14143)
Comments:
          Cooperative AI Foundation, Technical Report #1

- **What's New**: 이 보고서는 고급 AI 에이전트의 개발이 촉진되고 이들의 다수 배포가 임박한 가운데, 전례 없는 복잡성을 가진 다중 에이전트 시스템이 발생할 것이라는 점에 주목합니다. 이러한 시스템은 새롭고 탐색이 부족한 리스크를 초래합니다. 이 논문에서는 세 가지 핵심 실패 모드인 미스코디네이션(miscoordination), 갈등(conflict), 공모(collusion)를 기반으로 리스크를 구조적으로 분류하여 제시합니다.

- **Technical Details**: 리스크를 뒷받침하는 주요 요소로는 정보 비대칭(information asymmetries), 네트워크 효과(network effects), 선택 압력(selection pressures), 불안정한 역학(destabilising dynamics), 의무 문제(commitment problems), emergent agency, 다중 에이전트 보안(multi-agent security) 총 7가지가 있습니다. 이 요소들을 통해 다중 에이전트 시스템에서 발생할 수 있는 다양한 리스크를 분석하고, 구체적인 사례들을 통해 이를 설명합니다.

- **Performance Highlights**: 몇 가지 주요 리스크 사례를 강조하며 이를 완화하기 위한 유망한 방향을 제시합니다. 실제 사례와 실험적 증거를 바탕으로 다중 에이전트 시스템이 제기하는 독특한 도전 과제를 조명하고, 이러한 시스템이 고급 AI의 안전성, 거버넌스(governance), 윤리(ethics)에 미치는 영향을 논의합니다.



### Can Community Notes Replace Professional Fact-Checkers? (https://arxiv.org/abs/2502.14132)
- **What's New**: 이번 연구는 소셜 미디어에서의 허위정보와 관련된 전문 사실 확인(fact-checking)과 커뮤니티 노트(community notes) 간의 관계를 조명합니다. 특히, 커뮤니티 노트가 전문 사실 확인자들의 작업에 얼마나 의존하고 있는지를 분석하고, 이러한 노트의 특성을 면밀히 조사합니다. 연구 결과, 커뮤니티 노트의 20%는 전문 사실 확인자의 작업에 명시적으로 의존하고 있으며, 이는 건강 및 정치와 같은 고위험 주제에서 더 두드러집니다.

- **Technical Details**: 연구진은 Twitter/X 커뮤니티 노트를 활용하여, 허위정보에 대한 반박이 이루어지는 패턴과 사실 확인 출처가 커뮤니티 노트에서 인용되는 빈도를 분석했습니다. 그 결과, 커뮤니티 노트가 사실 확인 출처를 인용하는 빈도가 기존 보고된 것보다 최대 5배 더 높은 것으로 나타났고, 특히 더 넓은 내러티브와 연결된 게시물의 경우 이 비율이 두 배로 증가했습니다. 이는 전문적인 사실 확인이 커뮤니티 노트의 효과적 작성을 위해 필수적임을 강조합니다.

- **Performance Highlights**: 커뮤니티 노트가 있는 게시물은 허위정보의 확산을 감소시키는 데 효과적이며, 사용자가 커뮤니티 노트를 통해 허위 정보를 보다 정확하게 식별할 수 있게 됩니다. 연구는 또한 커뮤니티 노트가 게시물 삭제 및 수정의 가능성을 높이고, 그 과정 속도를 가속화하는 데 기여한다는 것을 보여줍니다. 하지만 커뮤니티 노트의 효과에 대한 논란도 여전히 존재하며, 사용자 반응이 부정적일 수 있음을 나타내는 다양한 연구 결과도 있습니다.



### Gradients can train reward models: An Empirical Risk Minimization Approach for Offline Inverse RL and Dynamic Discrete Choice Mod (https://arxiv.org/abs/2502.14131)
- **What's New**: 본 논문에서는 동적 이산 선택(Dynamic Discrete Choice, DDC) 모델을 추정하는 문제를 다룹니다. 오프라인(Maximum Entropy-Regularized Inverse Reinforcement Learning, offline MaxEnt-IRL) 강화 학습의 새로운 접근을 제시하며, 제한적인 선형 파라미터 보상이 필요 없는 방법을 도입합니다. Empirical Risk Minimization (ERM) 기반의 새로운 IRL/DDC 프레임워크를 사용하여 벨만 방정식에서 명시적인 상태 전이 확률 추정의 필요성을 피하는 방법을 제안합니다.

- **Technical Details**: 이 방법은 비모수적(Non-parametric) 추정 기술인 신경망(Neural Networks)과 호환되며, 고차원 및 무한 상태 공간으로 확장할 수 있는 잠재력을 지니고 있습니다. 이론적으로, 벨만 잔여(Bellman residual)가 Polyak-Lojasiewicz (PL) 조건을 만족한다는 점이 중요한 통찰입니다. 이러한 특성은 강력한 볼록성(strong convexity)보다는 약하지만, 빠른 전역 수렴(global convergence) 보장을 보장할 수 있습니다.

- **Performance Highlights**: 일련의 합성 실험을 통해 제안한 방법이 벤치마크 방법 및 최신 대안(methods)들보다 지속적으로 우수한 성능을 나타냄을 보여주었습니다. 이러한 결과는 복잡한 데이터 환경에서도 신뢰할 수 있는 성능을 발휘할 수 있음을 시사합니다. 따라서 본 연구는 DDC 모델 추정 분야에서의 중요한 발전을 나타냅니다.



### Multi-Objective Bayesian Optimization for Networked Black-Box Systems: A Path to Greener Profits and Smarter Designs (https://arxiv.org/abs/2502.14121)
- **What's New**: 본 논문에서는 복잡한 산업 시스템의 설계를 위해 고안된 새롭고 이점이 많은 알고리즘인 MOBONS를 제안합니다. MOBONS는 네트워크 기반 모델을 활용하여 다목적 최적화(Multi-objective Optimization) 문제에 접근하고, 구조를 인식하면서 피드백 루프와 재순환 흐름을 포함할 수 있는 보다 유연한 최적화 프레임워크를 제공합니다. 기존의 블랙박스 및 화이트박스 접근법의 한계를 극복하여, 상호 연결된 시스템을 Function Node 시리즈로 모델링할 수 있는 가능성을 열어갑니다.

- **Technical Details**: MOBONS는 다목적 최적화 문제를 해결하기 위해 베이지안 최적화(Bayesian Optimization)에서 영감을 받은 새롭고 효율적인 알고리즘입니다. 이 알고리즘은 경제적 성과와 지속 가능성과 같은 상반된 목표를 균형 있게 조정하기 위해 Pareto-optimal 솔루션을 찾습니다. 또한, 안전성 및 규제 요건과 같은 제약 조건을 출력 변수로 다루며, 각 평가 반복에서 여러 후보를 동시에 평가할 수 있는 배치 평가를 지원합니다.

- **Performance Highlights**: MOBONS는 두 가지 사례 연구를 통해 그 효과성을 입증했습니다. 특히 한 사례에서는 지속 가능한 공정 설계와 관련된 내용을 다루었으며, 기존 방법들이 포착하지 못했던 피드백 루프와 재순환 스트림을 모델링할 수 있습니다. 이를 통해 MOBONS는 보다 수익성 있고, 회복력 있으며, 지속 가능한 공학 시스템 설계 개선에 기여할 가능성이 있습니다.



### Zero loss guarantees and explicit minimizers for generic overparametrized Deep Learning networks (https://arxiv.org/abs/2502.14114)
Comments:
          AMS Latex, 9 pages

- **What's New**: 이번 연구에서는 과다 파라미터화(overparametrization된) 딥러닝 네트워크가 감독 학습(supervised learning)에서 제로 손실(zero loss)을 도달 가능하게 하는 충분 조건을 제시합니다. 특별히, 경량 데이터(generic training data)에 대한 제로 손실 최소화(minimizers)를 그래디언트 하강법(gradient descent)을 사용하지 않고도 명시적으로 구성할 수 있음을 보여주었습니다. 또한, 네트워크의 깊이가 증가하면서 비용 최소화의 효율성을 저하시킬 수 있는 현상도 밝혀냈습니다.

- **Technical Details**: 이 논문에서는 L개의 숨겨진 층(hidden layers)을 가진 딥러닝 네트워크를 정의하고, 입력 공간(input space)과 레이블(output labels)에 대한 구성을 자세히 설명합니다. 수식적 표현을 통해 훈련 입력(x₁, ..., xₙ)과 레퍼런스 출력(yₗ)을 정의하며, 목표는 특정 훈련 데이터에 대해 제로 손실을 달성하는 것입니다. 훈련 관련 야코비안(Jacobian)의 랭크(rank) 손실(rank loss)와 관련된 조건을 분석하여, 효율적인 그래디언트 흐름(gradient flows)을 도출하는 데 기여하고자 합니다.

- **Performance Highlights**: 연구 결과, 충분히 과다 파라미터화된 모델에서는 제로 손실을 달성할 수 있는 충분 조건이 존재함을 밝혀냈습니다. 특히 입증된 바와 같이, 너비가 충분히 큰 경우 깊이가 필요하지 않으며, 선형 회귀(linear regression)를 통해도 원활한 학습이 가능합니다. 이러한 발견은 과다 파라미터화된 모델과 부족한 파라미터화된 모델 간의 제로 손실 도달 가능성에 대한 중요한 통찰을 제공합니다.



### Object-centric Binding in Contrastive Language-Image Pretraining (https://arxiv.org/abs/2502.14113)
- **What's New**: 이번 논문은 최근 비전 언어 모델(vision language models, VLM)의 한계를 극복하기 위해 혁신적인 접근을 제안합니다. 기존의 대조 모델(contrastive models)인 CLIP을 기반으로 하여 복잡한 구성 장면(compositional scenes)을 이해하는 데 필요한 새로운 캐주얼(casual) 및 구조적 요소를 통합하고자 합니다. 우리의 접근방식은 하드 네거티브(hard-negative) 증강을 사용할 필요 없이 구성 이해(compositional understanding)를 개선할 수 있는 방법입니다.

- **Technical Details**: 우리는 장면 그래프(scene graph)와 슬롯 구조(slots-structured) 이미지 표현을 연결하는 바인딩 모듈(binding module)을 도입하여 두 개의 모달리티 사이의 유사성을 구조적으로 평가할 수 있도록 합니다. 또한, 관계를 텍스트 조건부 비주얼 제약(text-conditioned visual constraints)으로 활용하여 객체 간의 복잡한 상호작용을 보다 효과적으로 포착합니다. 이러한 방법론은 CLIP 기반 모델의 다중 객체 구성을 개선하는 데 기여합니다.

- **Performance Highlights**: 우리의 모델은 복잡한 장면의 이미지-텍스트 일치를 보다 정확하고 샘플 효율적으로 수행할 수 있도록 지원합니다. 이로 인해 향후 비전 언어 모델의 발전에 기여할 수 있는 가능성을 제시하게 됩니다. 결과적으로, 본 연구는 기존의 VLM의 성능을 한층 높여주는 기반이 될 것입니다.



### Navigating Semantic Relations: Challenges for Language Models in Abstract Common-Sense Reasoning (https://arxiv.org/abs/2502.14086)
Comments:
          5 pages, 3 figures, ACM Web Conference 2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 추상적인 상식 추론 능력을 체계적으로 평가하기 위해 ConceptNet 지식 그래프를 활용합니다. 두 가지 프롬프트 접근법, 즉 규정된 정의를 기반으로 하여 가능한 의미 관계를 예측하는 'instruct prompting'과 예제를 활용하는 'few-shot prompting'을 제안합니다. 실험 결과, gpt-4o-mini 모델은 여러 관계를 순위 매길 때 일관된 성능을 보였으나, 한 가지 관계 예측으로 제한할 경우 성능이 크게 저하됨을 보여줍니다.

- **Technical Details**: 연구에서는 ConceptNet 지식 그래프에 기반하여 LLM의 상식 추상 추론을 평가하기 위한 두 가지 프롬프트 접근법을 사용합니다. 'instruct prompting'은 모델이 두 개체 간의 의미 관계를 명시된 이름을 기반으로 예측하도록 요구하며, 'few-shot prompting'은 예제에 기반해 관계를 일반화하는 능력을 평가합니다. 평가 프레임워크는 ConceptNet에서 임의로 샘플링 한 단일 유일한 엣지로 구성된 두 개의 데이터 세트를 사용하여 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, few-shot prompting에서 모델의 정확도는 전체 관계 집합 대신 다섯 개의 관계에서 선택할 때 상당히 향상되었습니다. 그러나 특정 관계에 대해 편향성이 나타났습니다. 이 결과는 상업적으로 사용되는 LLM의 추상적인 상식 추론 능력에서 여전히 큰 격차가 존재함을 시사하며, 신중한 프롬프트 엔지니어링이 성능 개선에 유망함을 강조합니다.



### Personalized Education with Generative AI and Digital Twins: VR, RAG, and Zero-Shot Sentiment Analysis for Industry 4.0 Workforce Developmen (https://arxiv.org/abs/2502.14080)
- **What's New**: 이 연구는 제4차 산업혁명(4IR) 교육을 위한 개인화된 AI 튜터 gAI-PT4I4를 제안합니다. 이 튜터는 감정 분석(sentiment analysis)을 활용하여 학생의 이해도를 평가하고, 생성적 AI(generative AI)와 유한 자동자(finite automaton)를 통해 학습 경험을 맞춤화합니다. VR(Virtual Reality) 기반의 저충실도 디지털 쌍둥이(low-fidelity Digital Twins)를 통합하여, 실시간 안내를 제공하는 대화형 튜터 기능이 특징입니다.

- **Technical Details**: gAI-PT4I4는 제로샷 감정 분석(zero-shot sentiment analysis)을 통해 교사-학생 상호작용을 긍정적 또는 부정적으로 분류하며, LLMs(대형 언어 모델)을 활용하여 정확도를 86%까지 높였습니다. 또한, RAG(retrieval-augmented generation)를 사용하여 특정 도메인 지식에 기초한 개인화된 학습 콘텐츠를 제공합니다. 학습 난이도를 동적으로 조정하기 위해 유한 자동자(Finite Automaton)를 활용하여 학생의 수행 정확도에 따라 난이도가 증가하는 방식으로 설계되었습니다.

- **Performance Highlights**: 22명의 자원자를 대상으로 한 실험 평가에서 gAI-PT4I4는 80% 이상의 정확도를 달성했으며, 훈련 시간을 단축시켰습니다. 제안된 다중 충실도 디지털 쌍둥이 모델은 Bloom's Taxonomy와 Kirkpatrick 모델과 일치하여, 교육 요구에 적합한 확장 가능한 교육 프레임워크를 제공합니다. 이를 통해 모든 교육 수준에 맞는 체계적인 접근을 가능하게 합니다.



### DiffExp: Efficient Exploration in Reward Fine-tuning for Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.14070)
Comments:
          AAAI 2025

- **What's New**: 이번 연구에서는 텍스트-이미지 확산 모델의 리워드 파인튜닝(reward fine-tuning)을 위한 탐색 전략인 DiffExp를 소개합니다. 기존의 리워드 최적화 방법들이 온라인 샘플 생성 과정에서 느린 수렴 속도로 어려움을 겪는 반면, 본 접근법은 샘플의 다양성을 높이고 강력한 리워드 신호를 활용하도록 설계되었습니다. 이를 통해 모델 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: DiffExp는 두 가지 주요 전략으로 구성됩니다: (a) classifier-free guidance (CFG)의 스케일을 동적으로 조절하여 샘플의 다양성을 향상시키고, (b) 텍스트 프롬프트의 특정 구문에 랜덤 가중치를 부여하여 고품질 리워드 신호를 활용합니다. 이러한 전략은 온라인 샘플 생성 과정에서 시각적 다양성을 촉진하여 리워드 기반의 최적화를 개선하는 데 기여합니다.

- **Performance Highlights**: DiffExp를 정책 기울기 방법(policy gradient method)과 직접 리워드 역전파 방법(direct reward backpropagation method)와 통합하여 성과를 검증했습니다. 실험 결과, DiffExp는 최근의 리워드 기반 파인튜닝 방법들의 샘플 효율성을 향상시키며, SDXL 모델을 포함하여 고해상도 텍스트-이미지 모델의 품질도 상당히 개선시켰습니다.



### A Racing Dataset and Baseline Model for Track Detection in Autonomous Racing (https://arxiv.org/abs/2502.14068)
Comments:
          Currently Under Review

- **What's New**: 이 논문에서는 RoRaTrack라는 새로운 데이터셋을 소개합니다. 이는 레이싱 시나리오에서의 트랙 감지를 위한 주석이 달린 다중 카메라 이미지 데이터를 포함하고 있습니다. 데이터는 인디애나의 레이싱 서킷에서 Dallara AV-21 차량을 사용하여 수집되었고, Indy Autonomous Challenge(IAC)와 협력하여 진행되었습니다.

- **Technical Details**: RoRaTrack 데이터셋은 고속으로 인한 흐림, 카메라에서의 색상 반전, 도로 차선 표시의 부재와 같은 문제들을 해결하고 있습니다. 또한, RaceGAN이라는 GAN(Generative Adversarial Network) 기반 모델을 제안하여 이러한 과제를 효과적으로 처리하고 있습니다. 실험 결과, RaceGAN은 현재의 최신 머신 러닝 모델보다 우수한 성능을 보여주며 새로운 기준을 세우고 있습니다.

- **Performance Highlights**: RoRaTrack 데이터셋은 반복적으로 발생하는 레이싱 과제들을 포착하고 instance-level 주석을 통해 도로와 배경을 명확하게 구별하는 세그멘테이션 마스크를 포함하고 있습니다. 이 데이터셋은 실제 고속 레이싱 시나리오에서 수집된 데이터로, 다중 카메라 뷰를 제공하여 기존 시스템의 근원적 한계를 보완하고 있습니다.



### Triad: Vision Foundation Model for 3D Magnetic Resonance Imaging (https://arxiv.org/abs/2502.14064)
- **What's New**: 해당 연구에서는 3D MRI를 위해 설계된 비전 파운데이션 모델인 Triad를 소개합니다. Triad는 13만 1,170개의 3D MRI 볼륨을 학습하여 강력한 표현을 학습하며, 이를 통해 시맨틱(semantic) 분포를 제한하는 오르간 독립적(imaging descriptions) 이미징 설명을 사용합니다. 이 연구는 Triad가 임상 응용에서의 통합성 및 다용성을 보장하면서 새로운 3D MRI 작업에 대해 지속적으로 성능을 극대화할 수 있음을 보여줍니다.

- **Technical Details**: Triad는 19,721명의 환자로부터 수집된 3D MRI 데이터셋인 TriadMR-131K에서 훈련되며, 유방, 뇌, 전립선의 다양한 이미징 모달리티(T1-w, T2-w, FLAIR 등)를 포함합니다. 데이터 사전 훈련 동안, 널리 사용되는 오토인코더(autoencoder) 아키텍처를 채택하여 데이터의 심층적인 표현을 학습합니다. 또한, 다양한 크기의 인코더를 사전 훈련하여 다운스트림 작업의 요구에 맞추어 다양한 기능을 수행할 수 있도록 합니다.

- **Performance Highlights**: Triad 모델의 성능은 25개의 다운스트림 데이터셋을 평가하면서 뛰어난 결과를 보였습니다. nnUNet-Triad는 17개의 데이터셋에서 nnUNet-Scratch보다 6.88% 향상된 분할 성능을 기록하였고, Swin-B-Triad는 5개의 데이터셋 분류 작업에서 Swin-B-Scratch보다 3.97% 향상되었습니다. 이러한 성과들은 Triad가 다양한 임상 응용 프로그램에서 사용될 수 있는 유연성과 효율성을 제공함을 나타냅니다.



### EfficientPose 6D: Scalable and Efficient 6D Object Pose Estimation (https://arxiv.org/abs/2502.14061)
- **What's New**: 본 연구는 GDRNPP 기반의 빠르고 확장 가능한 포즈 추정기(pose estimator) 세트를 개발하여 정확성과 견고성을 강화하며, 실시간 환경에서의 효율성-정확성 균형을 개선하고자 합니다. AMIS 알고리즘을 제안하여 응용 프로그램에 특화된 추론 시간과 정확성의 무역 오프를 반영하는 모델 선택을 지원합니다. 이 방법은 LM-O, YCB-V, T-LESS, ITODD와 같은 4개의 주요 벤치마크 데이터셋에서 효과를 입증합니다.

- **Technical Details**: GDRNPP 클라우드 구조는 RGB 이미지를 기반으로 객체의 6D 포즈를 추정하며, CNN을 활용해 중요한 이미지 영역을 탐지하고 특징을 예측합니다. 추론 시간을 최적화하기 위해 GDRNPP 프로세스를 데이터 로드, 백본, 지오 헤드, 데이터 프로세스, 패치 PnP 등 여섯 단계로 나누어 예비 실험을 통해 주요 지연 원인을 식별했습니다. 이를 통해 구조적 변경을 통해 정확성을 유지하면서 추론 시간을 줄일 수 있도록 설계되었습니다.

- **Performance Highlights**: AMIS 알고리즘을 통해 선택된 후보 모델은 여러 데이터셋에서 추론 시간과 6D 포즈 추정 품질 간의 최적 무역 오프를 이루며, 각 알고리즘에 대해 정량적 결과를 제시합니다. 연구에서는 GDRNPP의 수정된 40개의 후보 아키텍처를 제안하며, 이들 후보는 다양한 시간 제약 하에서도 향상된 정확성을 유지합니다. 이를 통해 포즈 추정의 실용적인 적용 가능성을 넓히고자 합니다.



### Diversity-driven Data Selection for Language Model Tuning through Sparse Autoencoder (https://arxiv.org/abs/2502.14050)
- **What's New**: 이번 연구에서는 기존의 대규모 언어 모델이 인간의 선호에 맞게 올바르게 조정되기 위해 필요로 하는 instruction tuning의 중요성을 강조합니다. 하지만, instruction tuning 데이터는 양적으로 포화 상태에 있어 coreset 데이터 선택이 중요히 여겨지지만 이를 깊이 있게 다룬 연구는 부족합니다. 이 연구는 데이터의 다양성과 복잡성의 동등한 중요성을 간과한 기존 데이터 선택 방법의 한계를 지적하며, 이를 해결하기 위해 다중성을 고려한 데이터 선택 전략을 설계합니다.

- **Technical Details**: 연구에서는 sparse autoencoders를 활용하여 데이터의 다양성을 측정하고, 모델의 행동 해석 가능성 또한 향상시키는 방법을 제안합니다. 이러한 sparse autoencoders는 가장 긴 응답을 선택했을 때의 놀라운 효과를 설명하는 데도 기여합니다. 이 방법론은 데이터 선택 과정에서 모델의 복잡성을 유도하며, 전반적인 모델의 성능 향상에 기여합니다.

- **Performance Highlights**: 실험 결과, 이러한 효과적인 데이터 선택을 통해 교육 받은 모델이 다양한 다른 방법들과 비교할 때 훨씬 더 뛰어난 성능을 나타냄을 입증했습니다. 또한, 교육 비용을 줄이고 모델 행동에 대한 제어력을 더욱 증가시키는 가능성을 보여줍니다. 따라서, 데이터 선택에 대한 접근 방식은 모델의 전반적인 능력을 크게 향상시키는 데 기여합니다.



### Semantic Decomposition and Selective Context Filtering -- Text Processing Techniques for Context-Aware NLP-Based Systems (https://arxiv.org/abs/2502.14048)
- **What's New**: 이번 논문에서는 2가지 새로운 기술을 제안합니다. 첫째, Semantic Decomposition은 입력 프롬프트를 구조화된 정보 스키마로 분해하여 시스템이 쉽게 파싱할 수 있게 합니다. 둘째, Selective Context Filtering은 NLP 기반 파이프라인에 공급되는 불필요한 맥락 정보를 체계적으로 필터링할 수 있도록 합니다. 이 기술들은 LLM-to-system 인터페이스를 동적으로 구현하는 데 유용합니다.

- **Technical Details**: LLM은 최근 몇 년 동안 매개변수 수와 데이터셋 크기와 관련된 다양한 스케일링 법칙 덕분에 점점 더 강력해지고 있습니다. 그러나 이러한 모델은 응답을 생성하기 위한 입력으로 주어진 텍스트 토큰의 문맥 창 크기 제한이 있습니다. 이는 긴 입력 함수를 처리할 때 연산 요구 사항이 기하급수적으로 증가하기 때문이며, LLM이 생성하는 응답의 문맥 일관성을 잃을 가능성을 높입니다.

- **Performance Highlights**: Context-Aware Systems는 LLM 기반 파이프라인을 활용하여 다양한 입력 데이터에 적응하고 이에 반응합니다. 이러한 시스템은 고객 서비스, 의료, 교육, 금융 등 여러 분야에서 활용되고 있으며, LLM의 능력을 통해 사용자에게 맞춤형 응답을 제공합니다. 결과적으로 이러한 시스템은 산업을 혁신할 가능성이 있으며, 스케일 가능하고 개인화된 솔루션을 제공합니다.



### Towards a Learning Theory of Representation Alignmen (https://arxiv.org/abs/2502.14047)
- **What's New**: 최근 AI 모델의 표현이 규모와 성능의 증가에 따라 일치하고 있다고 주장되고 있습니다. 이 연구에서는 이러한 표현의 정렬(alignment)을 학습 이론적 관점에서 접근하고 있습니다. 특히, 다양한 정렬 개념을 검토하고 서로의 관계를 정리하며, 특정 작업(context of a task)에서 표현 간 상호작용을 이해하는 'stitching' 방식을 강조합니다.

- **Technical Details**: 논문은 표현 정렬의 정의 및 이에 대한 여러 접근 방식을 제시하며, 특히 kernel alignment을 사용하여 정렬을 수량화하는 방법론을 다룹니다. 이를 통해, supervised와 self-supervised 학습을 통해 훈련된 모델이 유사한 표현을 갖는다는 점을 강조합니다. 또한, 레이어 간의 stitching이 모델의 일반화 오류를 어떻게 영향을 미치는지 분석하고 kernel alignment 메트릭에 의해 바운딩될 수 있음을 보여줍니다.

- **Performance Highlights**: 잘 정렬된 특징(features)이 작업 성능을 크게 향상시킨다는 경험적 결과가 밝혀졌습니다. 그러나, uni/multi-modal 학습에 대한 수학적 도구의 필요성이 대두되었습니다. 이 연구는 kernel alignment의 개념을 통해 다양한 커뮤니티의 정렬 정의를 통합하여 실질적인 응용을 위한 더 깊은 이해를 제공합니다.



### Position: There are no Champions in Long-Term Time Series Forecasting (https://arxiv.org/abs/2502.14045)
Comments:
          Pre-print

- **What's New**: 최근 장기 시계열 예측(長期時系列豫測, Long-Term Time Series Forecasting) 분야에서 복잡한 예측 모델들이 여러 단계를 거쳐 지속적으로 향상되고 있습니다. 하지만 새로운 모델들이 기존의 성과를 능가하는 것에 대한 의구심이 커지고 있으며, 공정한 벤치마킹 관행의 필요성이 제기되고 있습니다. 이를 위해 14개의 데이터셋에서 3,500개 이상의 네트워크를 훈련시키며 상위 성능 모델에 대한 포괄적이고 재현 가능한 평가를 수행하였습니다.

- **Technical Details**: 연구진은 Transformer 모델을 포함한 여러 최신 딥러닝 모델들을 평가하였으며, 각각의 모델이 어떻게 다양한 시계열 데이터셋에서 성능을 발휘하는지를 살펴보았습니다. 또한 하이퍼파라미터 설정과 현재의 평가 메트릭스가 결과에 미치는 영향을 분석하여 미세한 변경이 예측 성능에 극적인 변화를 가져올 수 있음을 보여주었습니다. 이러한 분석을 통해 모델 성능 평가에서의 일관성과 검증의 중요성을 강조하고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 최근 모델들이 대부분의 상황에서 이전의 대안 모델들보다 지속적으로 우수한 성능을 내지 못한다는 사실이 밝혀졌습니다. 특히, 데이터셋의 특성이 모델 성능에 미치는 영향을 명확히 제시하며, 해결해야 할 여러 도전 과제가 있음을 확인하였습니다. 이로 인해 LTSF 분야의 발전을 위해서는 공정한 평가 방법론의 확립이 필수적이라는 주장을 합니다.



### Asking for Help Enables Safety Guarantees Without Sacrificing Effectiveness (https://arxiv.org/abs/2502.14043)
- **What's New**: 이 논문에서는 기존의 강화 학습 알고리즘이 '재앙(catasrophes)'을 피할 수 있도록 하는 메커니즘을 제안하며, 이러한 알고리즘이 보상 극대화에도 도움이 될 수 있음을 보입니다. 따라서, 모든 MDP에서 서브선형 후회(sublinear regret)를 보장하는 첫 번째 결과를 제시합니다. 이는 안전한 AI 응용 프로그램에서 높은 보상을 얻으면서 재난을 예방할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 논문에서는 온라인 RL(강화 학습)을 다루며, 에이전트가 우수한 성과를 내는 것과 동시에 재앙을 피해야 함을 강조합니다. 연구자는 MDP가 비통신형(non-communicating)일 수 있다는 점을 허용하며, 에이전트가 멘토(mentor)에게 질문할 수 있는 기회를 제공합니다. 이 결과는 에이전트가 멘토의 행동을 사용하여 안전한 결과를 보장하면서도 효과성을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: PLA 공저에 의해 제시된 알고리즘이 MDP에 대해 재앙을 피하면서 서브선형 후회를 달성할 수 있음을 증명했습니다. 새로운 증명 기법은 후회를 '상태 기반(state-based regret)'과 '행동 기반(action-based regret)'으로 분해하여 사용하는 것으로, 이는 이전의 RL 이론과는 다른 접근입니다. 이 연구는 높은 위험이 따르는 환경에서도 AI 에이전트가 높은 보상을 얻을 수 있는 첫 번째 이론적 근거를 제공합니다.



### DiffSampling: Enhancing Diversity and Accuracy in Neural Text Generation (https://arxiv.org/abs/2502.14037)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 텍스트 생성 과정에서 발생하는 문제를 해결하기 위해 새로운 디코딩 전략인 DiffSampling을 제안합니다. 기존의 디코딩 방식들은 반복적인 출력이나 훈련 데이터를 재생산하는 경향이 있어, 이것이 성능 저하의 주된 원인으로 지목됩니다. DiffSampling은 확률 분포의 최소 이산 미분(minimum discrete derivative)을 활용하여 적절한 토큰을 선택하는 데 도움을 줍니다.

- **Technical Details**: DiffSampling 방법은 토큰의 확률 분포를 분석하여 임계 질량(critical mass)을 식별하는 데 기반합니다. 이 방법은 서열화된 분포에서 연속적인 확률 간의 가장 큰 차이를 이용하여 부정확한 토큰을 제거하거나 적절하지만 저확률인 토큰의 기회를 높입니다. 세 가지 서로 다른 디코딩 방법을 제안하며, 이는 수학 문제 해결, 극단 요약(extreme summarization), 그리고 다양한 연상 작업(divergent association task)과 같은 여러 과제에서 평가됩니다.

- **Performance Highlights**: DiffSampling은 제안된 세 가지 방법 모두에서 현재의 대안들보다 질과 다양성 측면에서 일관되게 동등하거나 더 높은 성능을 보입니다. 이 연구는 기존 방법들의 한계를 해결하고 적절한 텍스트 생성을 위한 간단하면서도 효과적인 접근 방식을 제시합니다. 다양한 실험 결과는 DiffSampling의 유용성을 증명하고, 이를 통해 LLM의 생성 품질을 크게 향상시킬 수 있음을 나타냅니다.



### Dynamic Activation with Knowledge Distillation for Energy-Efficient Spiking NN Ensembles (https://arxiv.org/abs/2502.14023)
- **What's New**: 본 연구는 인공지능(AI) 모델에서의 혁신을 통해 에너지가 제한된 환경에서도 고효율을 추구합니다. 고전적인 인공신경망(ANN) 대신, 사건 기반의 신경망인 스파이킹 신경망(SNN)을 활용하여 에너지 효율성과 성능 향상을 도모하는 새로운 시스템을 제안합니다. 이 시스템은 지식 증류(knowledge distillation)와 앙상블 학습(ensemble learning)을 결합하여 인공지능 모델의 성능 격차를 해소하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 기초 AI 모델에서 획득한 지식을 활용하여 작은 SNN 모델들로 구성된 스파이킹 신경 앙상블(Spiking Neural Ensemble, SNE)을 훈련시키는 방식입니다. SNE는 각각의 학생 모델이 학습하는 동안 교사 네트워크(큰 ANN)로부터 얻은 피쳐를 분리(disentangle)하는 과정을 포함하며, 이로 인해 각 학생이 독립적인 예측 작업을 수행할 수 있도록 합니다. 이 과정에서 주어진 입력에 대해 학생 모델의 일부만 활성화하여 에너지 효율을 극대화할 수 있습니다.

- **Performance Highlights**: SNE는 CIFAR-10 데이터셋에서 교사 네트워크 대비 20배 이상의 계산 요건 감소를 이루며, 단 2%의 정확도 손실로 아키텍처의 효율성을 향상시킵니다. 또한, 학생 앙상블에서 적절한 수의 활성 학생 모델을 선택함으로써 에너지 소비를 최대 65% 감소시키는 동시에 정확도는 약 2% 떨어지는 정도로 유지되는 성과를 거두었습니다. 노이즈가 있는 상황에서도 SNE는 기존 ANN 교사 모델보다 우수한 강건성을 보이며, 이러한 디스엔탱글먼트 절차는 전통적인 분할 방식에 비해 최대 2.4%의 정확도 향상으로 이어졌습니다.



### Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems (https://arxiv.org/abs/2502.14019)
- **What's New**: 이번 연구는 텍스트 생성 시스템의 출력물들이 점점 더 인간과 유사하게 인식되고 있다는 사실에 주목하고, 이러한 행동이 심리적 의존이나 과신과 같은 해로운 결과를 초래할 수 있다는 우려를 나타냅니다. 이를 완화하기 위한 개입(intervention) 방법론이 부족하다는 점을 지적하며, 시스템 출력의 인간 유사성을 줄이기 위한 다양한 개입 목록을 제시하고자 합니다.

- **Technical Details**: 연구자들은 인지 행동(intervention) 카테고리를 정리하기 위해 NLP, HRI 및 HCI 영역의 선행 문헌을 기반으로 20개의 관련 논문을 선별했으며, 이를 통해 9가지의 개입 유형과 5가지의 시스템 행동 유형을 도출하였습니다. 모델이 생성한 텍스트의 인간 유사한 특성을 정의하고 참가자들이 이러한 출력물의 재작성을 도와주는 크라우드소싱 연구도 실시하였습니다.

- **Performance Highlights**: 이번 연구의 결과는 미래의 텍스트 생성 시스템 설계 및 평가에 있어 중요한 이론적 기초를 제공합니다. 효과적인 개입 방법의 식별 및 평가를 통해 인간 유사성으로 인한 부작용을 줄이고 사용자와의 상호작용에서 긍정적인 경험을 증대시킬 수 있는 잠재력을 보여줍니다.



### Appeal prediction for AI up-scaled Images (https://arxiv.org/abs/2502.14013)
- **What's New**: 이 논문은 AI 기반 DNN(Deep Neural Network) 기반 업스케일링 알고리즘의 평가를 다루고 있으며, 기존의 PSNR 및 SSIM과 같은 객관적 지표 대신 주관적 평가에 중점을 둡니다. 저자들은 136개의 기본 이미지를 사용하여 다섯 가지 업스케일링 방법에 대한 새로운 데이터셋을 개발하였으며, 이 데이터셋은 1496개의 주석 이미지로 구성되어 있습니다. 다양한 실제 이미지에 대한 성과 평가가 부족했던 기존 연구와 차별화되고 있습니다.

- **Technical Details**: 저자들은 Real-ESRGAN, BSRGAN, waifu2x, KXNet, Lanczos 필터를 포함하여 다섯 가지 업스케일링 알고리즘을 비교합니다. 이 연구에서는 crowd-sourcing을 통해 수집된 주관적 피드백을 활용하여 이미지 매력을 평가하고, 각 알고리즘의 성능을 새롭게 구축된 데이터셋을 바탕으로 검토합니다. 또한, DNN을 이용해 어떤 업스케일링 방법이 사용되었는지를 감지하는 모델을 훈련시켰습니다.

- **Performance Highlights**: 연구 결과, Real-ESRGAN과 BSRGAN이 가장 우수한 성능을 보였으며, 저자들은 이전 모델들과 비교하여 새롭게 훈련한 DNN 모델이 이미지 매력을 예측하는데 있어 우수한 Pearson 상관관계(≈0.84)를 기록했다고 보고합니다. 또한, 현존하는 최신 이미지 품질 모델들은 저자들이 개발한 DNN 모델보다 뛰어난 예측 성능을 보여주지 못했습니다. 이 연구는 공개 과학(Dataset 및 구현 코드)을 위한 데이터 제공을 통해 후속 연구를 촉진할 수 있도록 기여하고 있습니다.



### DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices (https://arxiv.org/abs/2502.14011)
- **What's New**: 이 논문은 DFDT(Dynamic Fast Decision Tree)라는 에너지 효율적인 메모리 제약 데이터 스트림 마이닝을 위한 새로운 알고리즘을 제시합니다. DFDT는 동적으로 grace periods, tie thresholds 및 split evaluations을 조정하여 Hoeffding 트리 성장 효율성을 개선합니다. 이는 실시간 기계 학습 추론을 가능하게 하면서도, 메모리 소모를 줄이고, 데이터 흐름의 변화에 지속적으로 적응할 수 있도록 합니다.

- **Technical Details**: 이 알고리즘은 기능적으로 엄격한 평가 규칙(Entropy, Information Gain, Leaf Instance Count 기반)과 메모리 관리를 위한 leaf deactivation 메커니즘을 통합합니다. 이로 인해 자주 방문되는 노드에서 더 많은 계산이 이루어질 수 있도록 하고, 덜 방문되는 노드에서는 에너지를 절약할 수 있습니다. 또한, 기존의 앙상블 기반 접근 방식을 재고하며, 효율적인 트리 성장 컨트롤이 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 DFDT는 제한된 메모리 환경에서도 예측 성능을 향상시키며(0.43 대 0.29의 순위), VFDT와 SVFDT보다 낮은 실행 시간으로 높은 성능을 기록했습니다. 알고리즘의 주요 구성 요소에 대한 분리 연구에서는 leaf deactivation을 통한 적응형 확장 모드의 생략이 메모리 사용량을 줄이면서도 정확성과 컴퓨테이션 시간을 최적화하는 데 기여함을 보여주었습니다.



### Which Attention Heads Matter for In-Context Learning? (https://arxiv.org/abs/2502.14010)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 in-context learning (ICL) 성능을 주도하는 두 가지 메커니즘, 즉 induction heads와 function vector (FV) heads를 비교합니다. 본 논문은 12개의 언어 모델을 분석하며, FV heads가 ICL 성능의 주요 요인임을 밝혔다고 주장합니다. 특히, 더욱 큰 모델에서 이 경향이 뚜렷하다는 점이 주목할 만합니다.

- **Technical Details**: 연구에서는 induction heads와 FV heads의 차별적인 성격 및 동작 특성을 관찰했습니다. Induction heads는 입력의 반복 패턴을 인식하는 반면, FV heads는 특정 attention heads에서 추출한 ICL 작업의 압축 표현을 처리합니다. 이러한 분석은 70M에서 7B 파라미터를 가진 12개의 transformer 모델을 기준으로 진행되었습니다.

- **Performance Highlights**: 결과적으로, FV heads를 제거했을 때 ICL 작업의 정확도가 크게 저하되는 반면, induction heads 제거는 그 효과가 제한적이었습니다. 또한, induction heads는 훈련 과정에서 FV heads로 발전할 수 있으며, 이는 ICL에 대한 더 복잡한 FV 메커니즘 학습을 촉진한다고 제안됩니다. 이러한 발견은 ICL의 메커니즘 이해뿐 아니라 모델 해석 가능성에 대한 중요한 시사점을 제공합니다.



### MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures (https://arxiv.org/abs/2502.14008)
- **What's New**: 이 논문에서는 대형 언어 모델의 비효율성을 해결하기 위한 새로운 masking learning 패러다임인 MaskPrune을 제안합니다. 기존의 구조적 가지치기 기법은 균일한 구조를 희생함으로써 특정 층의 성능을 유지할 수 있었지만, 이는 전반적인 추론 효율을 저하시켰습니다. MaskPrune은 minimax 최적화 기법을 기반으로 하여 가지치기 마스크를 최적화함으로써 장치 간의 동등성을 유지합니다. 이를 통해 기존의 SOTA 방법을 초월하는 성능을 입증하였습니다.

- **Technical Details**: MaskPrune은 가지치기 마스크와 목표 구조를 동시에 훈련하여 모델의 층 간 구조의 일관성을 유지합니다. 이 과정에서 비차별적인 sparsity 손실을 도입하며, 이는 근접 연산자(proximal operator)를 이용해 최적화됩니다. 또한, 각 층의 치수를 조정하여 전체적인 성능 저하를 최소화하면서 목표 sparsity를 달성하는 데 중점을 둡니다. 최적화 절차는 Transformer 모델의 Multi-Head Attention과 Feed-Forward Network의 주요 구조적 요소를 대상으로 진행됩니다.

- **Performance Highlights**: 실험 결과, MaskPrune은 다양한 sparsity 레벨에서 LLaMA 모델의 성능을 유지하면서 균일한 구조를 보장함이 입증되었습니다. 마스크와 구조의 일관성이 확보됨으로써, 이 접근 방식은 다수의 자연어 처리(NLP) 작업에서 뛰어난 결과를 보였습니다. 이러한 성능 향상은 기존 모델 압축 기법과 비교했을 때 상당한 개선으로 나타났습니다.



### Rectified Lagrangian for Out-of-Distribution Detection in Modern Hopfield Networks (https://arxiv.org/abs/2502.14003)
Comments:
          Accepted to AAAI 2025

- **What's New**: 현대 Hopfield 네트워크(MHNs)는 대량의 패턴을 기억하고 검색할 수 있는 능력으로 인공지능 분야에서 주목받고 있습니다. 기존의 MHNs는 분포 외(out-of-distribution; OOD) 샘플을 처리하는 데 한계를 보였는데, 이는 모든 샘플이 분포 내(in-distribution; ID) 샘플로 가정되었기 때문입니다. 본 논문에서는 OOD 샘플을 명시적으로 처리하기 위한 새로운 라그랑지안(rectified Lagrangian; RegLag)을 제안합니다.

- **Technical Details**: RegLag는 OOD 샘플을 위한 매력점을 창출하는 새로운 메커니즘으로, 모든 상호작용 행렬에 대해 이점이 있을 수 있습니다. 이 메커니즘은 ID 메모리 강도를 나타내는 상수를 포함한 정류 선형 유닛(rectified linear unit; ReLU)을 사용하여 동적 시스템 내에서 OOD 샘플을 쉽게 식별할 수 있도록 합니다. 연구진은 상호작용 행렬을 최적화하여 ID/OOD를 구분하는 확률 밀도를 추정하는 방법도 제시합니다.

- **Performance Highlights**: 본 연구는 RecLag 기반 MHNs가 최신 Hopfield 에너지를 사용하는 에너지 기반 OOD 탐지 방법과 비교하여 효과적임을 보여줍니다. 아홉 개의 이미지 데이터셋에서 다양한 실험을 통해 RecLag의 성능을 평가하였으며, 기존 방법들보다 우수한 결과를 도출하였습니다. 이러한 결과는 RecLag가 OOD 탐지에 있어 신뢰할 수 있는 대안이라는 것을 증명합니다.



### Towards a perturbation-based explanation for medical AI as differentiable programs (https://arxiv.org/abs/2502.14001)
Comments:
          7 pages, 1 figure

- **What's New**: 최근 머신러닝(ML) 알고리즘의 발전으로 의료 기기에서 인공지능(AI) 모델을 활용하여 진단 지원과 일상적인 작업 자동화가 가능하게 되었습니다. 하지만 AI 모델의 결과에 대한 충분하고 객관적인 설명 가능성에 대한 요구가 커지고 있습니다. 기존의 설명 방법들은 훈련 데이터의 특성과 샘플링 프로토콜로 인한 편향을 겪을 수 있습니다. 본 연구는 AI 모델의 동작을 독립적으로 설명할 수 있는 대안을 탐색하고 있습니다.

- **Technical Details**: 이 연구에서는 딥러닝 모델의 입력에 작은 변화를 추가했을 때 모델이 얼마나 안정적으로 반응하는지를 측정하기 위해 Jacobian matrix의 수치적 가용성을 조사합니다. 이는 특정 입력에 대한 훈련된 AI 모델에서 계산되며, 모델의 응답의 민감도를 설명하는 새로운 방법론인 pertubation-based explanation(PBX)으로 주목받고 있습니다. PBX는 훈련 과정에 의존하지 않으며, 추가 데이터 수집을 필요로 하지 않고, 각 입력 사례에 대해 안정성을 측정할 수 있습니다.

- **Performance Highlights**: 본 연구는 임상에서 AI 모델의 출력 반응을 이해하고 해석하는 데 도움을 주는 첫걸음으로서 PBX 접근 방식을 제안합니다. 이 접근 방식은 단일 특성뿐만 아니라 특성 간의 상호작용에 대한 설명 가능성을 제공합니다. 결과적으로 의료 전문가们은 AI 모델의 행동을 보다 적절하게 이해하고 해석할 수 있는 도구를 갖게 될 것으로 기대됩니다.



### Human-Artificial Interaction in the Age of Agentic AI: A System-Theoretical Approach (https://arxiv.org/abs/2502.14000)
Comments:
          27 pages, 10 figures

- **What's New**: 이 논문은 인간-컴퓨터 상호작용(HCI)에 대한 새로운 관점을 제시하며, 이를 동적인 상호작용으로 프레임화합니다. 기존의 인터페이스 기반 접근 방식을 넘어 다양한 능력과 목표를 가진 이질적인 에이전트 간의 조정 및 커뮤니케이션의 중요성을 강조합니다. 특히, 다중 에이전트 시스템(MAS)과 Centaurian 시스템 간의 차이점을 정의하면서, 두 가지 패러다임의 통합을 위한 커뮤니케이션 공간 프레임워크를 소개합니다.

- **Technical Details**: 이 논문에서는 Petri nets를 기반으로 하는 정형화된 프레임워크를 제안하여, MAS와 Centaurian 시스템 간의 상호작용을 모델링합니다. Petri nets는 명확한 그래픽 표현, 동시 프로세스 및 동기화 요건을 포착하는 데 유리하며, 복잡한 데이터 유형 및 조건을 처리할 수 있는 확장성이 있습니다. 이 접근법은 MAS와 Centaurian 시스템에서 다양한 능력을 가진 에이전트 간의 조정 및 통합을 지원합니다.

- **Performance Highlights**: 이 연구는 자율 로봇, 인간-인-루프 의사결정, AI 기반 인지 아키텍처 등에서 실용적인 응용 가능성을 보여줍니다. 또한, 다음 세대 하이브리드 인공지능 시스템을 위한 기초를 제공하여, 구조화된 조정과 발생하는 행동의 균형을 맞추는 데 기여합니다. 이는 서로 다른 패러다임의 통합을 통해 더욱 진보된 HCI 시나리오를 제시하는 데 중요한 역할을 합니다.



### A Baseline Method for Removing Invisible Image Watermarks using Deep Image Prior (https://arxiv.org/abs/2502.13998)
- **What's New**: 이번 논문에서는 AI 생성 콘텐츠를 검출하기 위해 사용되는 이미지 워터마크의 새로운 제거 방법을 제시합니다. 저자들은 데이터를 사용하거나 워터마크 시스템에 대한 정보 없이도 보이지 않는 워터마크를 제거할 수 있는 간단한 방법을 제안합니다. 이러한 접근법은 Deep Image Prior (DIP) 기법을 활용하여 중간 단계에서 워터마크를 제거할 수 있는 그림을 생성합니다.

- **Technical Details**: DIP는 훈련되지 않은 딥 뉴럴 네트워크(DNN)를 기반으로 하여 단일 이미지 블라인드 디노이징 효과를 활용합니다. 저자들은 낮은 주파수 성분을 보다 빠르게 탐지하는 DIP의 특성이 이미지 품질을 유지하면서 워터마크를 정화하는 데 효과적이라는 점을 강조합니다. 또한, 이 방법은 기존의 수동 설계된 워터마크뿐만 아니라 훈련 기반의 워터마크에도 적용할 수 있습니다.

- **Performance Highlights**: 연구 결과는 DIP 기반의 블라인드 디노이징이 고품질의 탈출 이미지를 생성하여 많은 기존의 보이지 않는 워터마크를 효과적으로 우회할 수 있음을 나타냅니다. 저자들은 DIP를 워터마킹 시스템의 견고성 평가에서 중요한 요소로 통합할 것을 제안하고, 향후 워터마크 방어 방안을 마련하기 위한 연구가 필요하다고 언급합니다.



### Generative Detail Enhancement for Physically Based Materials (https://arxiv.org/abs/2502.13994)
- **What's New**: 이 논문에서는 오프더쉘프(Off-the-shelf) 확산 모델(diffusion model)과 역 렌더링(inverse rendering)을 사용하여 물리 기반(materials) 재료의 디테일을 향상시키는 도구를 제시합니다. 목표는 마모, 노화, 날씨 등의 시각적 요소를 추가하여 재료의 실제 감을 높이는 것입니다. 논문의 방법론은 사용자가 원하는 세부 사항을 설명하는 텍스트 프롬프트(text prompt)와 여러 각도의 렌더링 이미지를 활용하여 모델을 조정하는 것입니다.

- **Technical Details**: 제안된 연결 방식에서는 여러 뷰(View)에서 일관된 노이즈(noise)를 확산 모델에 제공하여 멀티 뷰 일관성(multi-view consistency)을 유지합니다. UV 공간에서 통합 노이즈(integral noise) 패턴을 사용하여 노이즈를 시드(seed)하고, 주의(attention) 메커니즘을 통해 픽셀들이 다른 뷰의 해당 픽셀에 주의를 기울이도록 유도함으로써 기하학적 일관성을 강화합니다. 이 방법은 추가적인 트레이닝 없이도 기존의 확산 모델을 체계적으로 사용할 수 있게 해 줍니다.

- **Performance Highlights**: 이 도구는 사용자가 원하는 목표 프롬프트에 따라 재질 텍스쳐 맵을 수정할 수 있으며, 물체의 원래 기하학(geometry)과 예술적 의도를 보존합니다. 사용자는 개선된 입력 자료를 활용하여 더 큰 씬에 통합하고 일반 렌더러를 사용하여 렌더링할 수 있습니다. 논문의 접근 방식은 대규모 데이터 세트의 생성이나 대규모 모델 재훈련을 피하면서도 고품질 시각적 품질을 증대시킵니다.



### Learning to Discover Regulatory Elements for Gene Expression Prediction (https://arxiv.org/abs/2502.13991)
- **What's New**: 본 논문에서는 DNA 서열로부터 유전자 발현을 예측하는 문제를 다루고 있으며, 이 과정에서 유전자 발현을 제어하는 규제 요소를 발견하고 추출할 수 있는 Seq2Exp라는 새로운 네트워크를 제안합니다. Seq2Exp는 에피게놈 신호와 DNA 서열 간의 인과 관계를 포착하고, 비인과적 구성 요소를 필터링하는 정보 병목 기법을 적용하여 발현 예측의 정확성을 높입니다. 또한, 기존의 방법보다 우수한 성능을 보임을 실험을 통해 보여주고 있습니다.

- **Technical Details**: Seq2Exp는 DNA 서열과 에피게놈 신호로부터 관련 서브-서열을 선택적으로 추출하여 유전자 발현 예측을 개선하는 시스템적인 접근 방식을 제안합니다. 이 프레임워크는 에피게놈 신호에 직면한 잠재적인 편향과 한계를 고려하여, 복잡한 인과적 관계를 반영하게 설계되었습니다. 특히, 기존 방법들을 뛰어넘는 새로운 정보 병목 처리 방식을 도입하여, 규제 요소의 추출과 예측을 효과적으로 수행합니다.

- **Performance Highlights**: Seq2Exp는 여러 기존 유전자 발현 예측 기준에 비해 탁월한 성능을 입증하며, 통계적 Peak detection 방법인 MACS3와 비교하여 더 유의미한 규제 요소를 발견합니다. 이러한 성능 향상은 Seq2Exp가 맥락을 이해하고 적절한 서열을 추출하는 능력 덕분입니다. 결과적으로, 이 연구는 유전자 발현 예측에서의 혁신을 통해 생명과학 연구에 기여할 것으로 기대됩니다.



### Gesture-Aware Zero-Shot Speech Recognition for Patients with Language Disorders (https://arxiv.org/abs/2502.13983)
- **What's New**: 이번 연구에서는 언어장애인을 위한 제스처 인지 자동 음성 인식(ASR) 시스템을 제안합니다. 이 시스템은 다중모드 대형 언어 모델(multi-modal large language model)과 제로샷 학습(zero-shot learning)을 이용하여 비언어적 의사소통 방식인 제스처를 통합합니다. 언어장애인들은 의사소통에 있어 말뿐만 아니라 제스처에 크게 의존하므로, 이를 고려한 접근이 필요합니다.

- **Technical Details**: 우리가 제안한 시스템은 제스처 정보를 포함하여 의미 이해(semantic understanding)를 향상시키고, 이는 언어처리 및 이해 능력이 제한된 사용자와의 상호작용에서 중요한 요소로 작용합니다. 실험에서는 다양한 데이터 세트를 사용하여 시스템의 성능을 평가하였으며, 제스처 정보를 포함하는 것이 실질적으로 의미적 이해도를 높인다는 결과를 보였습니다. 이러한 접근은 언어장애인을 위한 효과적인 커뮤니케이션 기술 개발에 기여할 수 있습니다.

- **Performance Highlights**: 본 연구에서 제안한 ASR 시스템은 기존의 단순 음성 인식 시스템에 비해 사용자와의 상호작용에서 더 높은 효율성을 보여주었습니다. 특히, 비언어적 의사소통의 통합을 통해 장애물 없이 원활한 의사소통이 가능해짐을 입증하였습니다. 이 연구 결과는 언어장애인을 지원하기 위한 기술적 이정표가 될 수 있으며, 향후 이러한 시스템의 발전에 대한 기대감을 높입니다.



### Utilizing Effective Dynamic Graph Learning to Shield Financial Stability from Risk Propagation (https://arxiv.org/abs/2502.13979)
- **What's New**: 이번 연구에서는 GraphShield라는 새로운 접근법을 소개합니다. 이 방법은 서로 tightly coupled된 temporal과 spatial domain에서 정보를 향상시키고, 숨겨진 위험을 인식하며, 위험 전파 시각화를 통해 금융 안정성을 강화하는 데 중점을 두고 있습니다. 또한, 이러한 기능들은 금융 네트워크 내에서 위험의 확산을 완화하기 위한 강력한 솔루션을 제공합니다.

- **Technical Details**: GraphShield는 세 가지 주요 혁신을 포함하고 있습니다. 첫째, temporal과 spatial 정보를 동시에 통합하여 동적 그래프 학습 모듈을 개발하였습니다. 둘째, 위험 샘플의 클러스터링 경향을 활용하여 숨겨진 위험을 인식하는 모듈을 확장하였고, 마지막으로 위험 전파 분석을 위한 시각화 도구를 제안하여 위험 간의 관계를 정량화하고 검증할 수 있게 하였습니다.

- **Performance Highlights**: 이 연구에서는 GraphShield의 성능을 두 개의 실제 데이터세트 및 두 개의 오픈 소스 데이터세트를 이용해 검증하였고, 우수한 성과를 기록하였습니다. 성과는 기존의 벤치마크 모델들과 비교했을 때 최고 수준의 성능을 보여주며, 금융 안정성을 강화하는 데 기여할 것으로 기대됩니다. 이로 인해, 금융 네트워크 내에서 위험의 확산을 효과적으로 관리할 수 있는 튼튼한 프레임워크를 제시하게 되었습니다.



### IncepFormerNet: A multi-scale multi-head attention network for SSVEP classification (https://arxiv.org/abs/2502.13972)
- **What's New**: 이 논문에서는 IncepFormerNet이라는 새로운 하이브리드 모델을 제안하며, 이는 Inception과 Transformer 아키텍처를 결합한 것입니다. 이 모델은 다양한 크기의 병렬 컨볼루션 커널을 사용하여 SSVEP 신호 내의 미세한 변화를 정확하게 포착하고, Transformer의 multi-head attention 메커니즘을 통합해 글로벌 의존성을 이해하며 복잡한 대칭을 표현합니다. 또한 필터 뱅크 기법을 활용하여 SSVEP 데이터의 스펙트럼 특성 기반으로 기능을 추출합니다.

- **Technical Details**: IncepFormerNet은 SSVEP 신호의 명확한 주파수 및 조화 성분을 활용하여 다중 스케일 피처를 추출합니다. 이 모델은 다중 스케일 특성을 포착하는 데 사용되는 다양한 크기의 컨볼루션 커널을 통합하여 정확도를 향상시킵니다. 벤치마크와 BETA라는 두 개의 공개 데이터 세트를 사용하여 모델의 효과를 검증하였으며, 신호 주파수는 8-15.8 Hz 범위에서 조정되었습니다. 데이터는 64채널 EEG 기록기를 통해 수집되었고, 저장 및 계산을 줄이기 위해 250 Hz로 다운샘플링되었습니다.

- **Performance Highlights**: IncepFormerNet은 Dataset 1에서 평균 87.41%의 정확도를 달성했으며, Dataset 2에서는 67.73%의 정확도를 기록하였습니다. 실험 결과는 제안한 모델이 기존의 깊은 학습 모델보다 상당히 높은 정확도를 보이며 SSVEP 신호의 복잡한 특성을 효과적으로 처리할 수 있음을 나타냅니다. 이를 통해 SSVEP-BCI 시스템의 분류 성능을 대폭 개선할 가능성을 보여줍니다.



### Bridging Simulation and Reality: A 3D Clustering-Based Deep Learning Model for UAV-Based RF Source Localization (https://arxiv.org/abs/2502.13969)
Comments:
          This paper has been submitted to IEEE ICC 2025

- **What's New**: 본 연구에서는 RF 소스 로컬라이제이션(RFSL)의 정확도를 향상시키기 위해 Enhanced Two-Ray 모델을 제안합니다. 이 모델은 UAV가 동적 3D 조건에서 보다 현실적인 전파 환경을 시뮬레이션 할 수 있게 합니다. 또한, 3D Cluster-Based RealAdaptRNet이라는 딥러닝 모델을 개발하여 시뮬레이션 데이터에서 훈련된 후 현실 세계에서도 뛰어난 성능을 발휘하도록 하였습니다.

- **Technical Details**: 본 논문에서 제안한 Enhanced Two-Ray 모델은 UAV의 동적 움직임과 구조적 섀도잉을 반영하여 전파 모델을 개선합니다. 3D Cluster-Based RealAdaptRNet은 시뮬레이션에 기반한 3D 클러스터링 기술을 활용하여 강력한 로컬라이제이션 성능을 달성합니다. 이 모델은 18.2m의 평균 로컬라이제이션 오류를 기록하였으며, 다른 전통적인 방법들보다 33.5배 적은 파라미터를 사용하여 계산 효율성을 높였습니다.

- **Performance Highlights**: 실험 결과, Enhanced Two-Ray 모델은 기존의 자유 공간 및 두레 모델보다 현실 세계의 전파 시나리오를 보다 정확하게 시뮬레이션하는 것으로 나타났습니다. 3D Cluster-Based RealAdaptRNet은 AERPAW 테스트베드에서 실제 데이터로 검증할 때 탁월한 성능을 발휘하였습니다. 본 접근법은 다양한 경로에서도 강력한 일반화 능력을 보여 주며 실제 응용에 매우 적합합니다.




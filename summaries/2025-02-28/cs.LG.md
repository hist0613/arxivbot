New uploads on arXiv(cs.CL)

### Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization (https://arxiv.org/abs/2502.20364)
Comments:
          10 pages, 6 figures, 5 tables

- **What's New**: 이번 논문에서 소개하는 Agentic Generative AI 기술은 대규모 언어 모델(LLMs)과 Retrieval-Augmented Generation (RAG), 지식 그래프(KGs), 벡터 스토어(VSs)를 통합하여 법률 시스템, 연구, 추천 시스템 등 다양한 전문 분야에 적용할 수 있는 혁신적인 기술입니다. 이 기술은 방대한 비구조적 및 반구조적 데이터 세트 내의 관계를 추론하는 데 뛰어난 성능을 보이며, 법률 문서의 복잡한 네트워크를 탐색하는 데 필수적인 통찰력을 제공합니다. 또한, 이 시스템은 비정형 법률 텍스트를 효과적으로 수집하고 분석하여, AI가 사례와 법령 간의 복잡한 연결을 파악할 수 있도록 지원합니다.

- **Technical Details**: 법률 분야는 헌법, 법령, 규정, 사례 법 등을 포함하여 각기 다른 구조적 논리를 따르는 복합적인 데이터 유형으로 구성되어 있습니다. 기존의 키워드 기반 검색을 초월하여, RAG 시스템은 관련 법적 문서나 데이터 포인트를 검색하고 언어 모델을 사용하여 이를 통합된 맥락 기반의 답변으로 전환하는 방식을 사용합니다. 이 논문은 VS, KG, NMF라는 세 가지 핵심 기술을 통합하여 법률 데이터의 효과적인 탐색, 검색 및 해석을 지원합니다.

- **Performance Highlights**: 이 시스템은 법률 문서의 클러스터링, 요약 및 교차 참조를 가능하게 하여 법률 정보 검색의 질과 해석 가능성을 높입니다. 또한, 벡터 스토어의 높은 재현율과 지식 그래프의 구조적 관계, NMF의 주제 발견 능력을 결합하여 이 시스템은 매우 큰 데이터셋에 대해 설명 가능한 추론을 제공할 수 있습니다. 이러한 접근 방식은 정의된 기준을 충족하며, 법률 시스템 내의 정당성과 운영 효율성을 높이는 데 중요한 역할을 합니다.



### Bridging the Creativity Understanding Gap: Small-Scale Human Alignment Enables Expert-Level Humor Ranking in LLMs (https://arxiv.org/abs/2502.20356)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 창의적 콘텐츠 이해의 한계를 다시 조명하며, 특히 유머 이해 과정에서의 인간과 LLM 간의 큰 격차를 밝혀냈습니다. 연구진은 유머 이해를 시각적 이해, 만화-캡션 추론, 인간 선호도 정렬의 세 가지 구성 요소로 분해하였고, 이를 통해 유머 캡션 순위에서 82.4%의 정확도를 달성하여 이전의 67% 기준선을 크게 초월했습니다. 또한, 잠재적 편향을 줄이기 위해 인간 선호 데이터로의 세심한 조정이 중요하다는 점을 강조하였습니다.

- **Technical Details**: 이 연구는 유머 캡션 순위의 세 가지 주요 구성 요소를 제시합니다: 시각적 이해(visual understanding), 유머 추론(humor reasoning), 그리고 인간 선호도 정렬(alignment with human preferences)입니다. 연구진은 시각적 주석과 LLM이 생성한 설명을 개선하여 유머 캡션에서의 비주얼 이해와 추론 능력을 크게 향상시켰습니다. 특히, 각 캡션 쌍에 대한 인간 기호 데이터를 통한 세밀한 튜닝(fine-tuning)이 성과를 높이는 데 중요한 역할을 했습니다.

- **Performance Highlights**: 이 연구는 캡션 순위 성능을 67%에서 82.4%로 증가시키며 인간 전문가들과 동등한 수준에 도달했습니다. 연구진의 실험 결과, 다양한 페르소나 기반 접근 방식은 효과가 미미했으나, 군중 선호 데이터를 사용한 세밀한 조정은 매우 효과적임을 발견했습니다. 이러한 성과는 AI가 창의적 과제에서 인간의 개별 및 집단 선호를 이해하는 데 있어 광범위한 도전 과제를 나타냅니다.



### KEDRec-LM: A Knowledge-distilled Explainable Drug Recommendation Large Language Mod (https://arxiv.org/abs/2502.20350)
- **What's New**: 최근 생물의학 분야에서 약물 발견(Drug Discovery)의 중요성이 더욱 커지고 있습니다. 그러나 설명 가능한 약물 발견(Explainable Drug Discovery)은 아직 충분히 탐구되지 않았습니다. 본 연구에서는 대형 언어 모델(LLMs)을 활용하여 약물 추천과 합리적 근거 생성을 위한 새로운 프레임워크인 KEDRec-LM을 소개하며, 이를 통해 다운스트림 작업과 실생활 응용에 대한 개선 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 공개 소스의 약물 지식 그래프와 임상 시험 데이터, PubMed 출판물을 결합하여 설명 가능한 약물 발견 작업을 위한 포괄적인 데이터셋(expRxRec)을 구축하였습니다. DRKG(Drug Repurposing Knowledge Graph)에서 약물-질병 쌍의 하위 집합을 추출하고, 검색 증강 생성(Retrieval-Augmented Generation, RAG) 기법을 적용하여 관련 문헌을 검색합니다. 이를 바탕으로 LLaMA 모델을 훈련시켜 생물의학 지식을 이해하고 약물-질병 관계에 대한 추론을 지원합니다.

- **Performance Highlights**: 우리의 접근 방식은 약물 추천의 설명 가능성을 높이는 데 효과적임을 입증하였으며, 새로운 데이터셋과 로컬 모델을 공개하여 연구 커뮤니티에 기여합니다. 이러한 방법론은 생물의학 문헌을 체계적으로 활용하여 약물 설계 및 발견 과정을 개선하는 데 중요한 진전을 보여줍니다. 향후 이 연구를 통해 새로운 치료 기회를 발견할 수 있기를 기대합니다.



### Sparse Auto-Encoder Interprets Linguistic Features in Large Language Models (https://arxiv.org/abs/2502.20344)
- **What's New**: 이번 연구에서는 sparse auto-encoders (SAEs)를 사용하여 대규모 언어 모델(LLMs)의 언어적 메커니즘을 체계적이고 종합적으로 조사합니다. LLM에서 발음, 음운론, 형태론, 통사론, 의미론, 화용론 등 여섯 가지 측면에서 다양한 언어적 특성을 추출하고 평가하며, 이러한 특성을 조작할 수 있는 방법을 제안합니다. 연구 결과는 LLM이 본질적인 언어 지식을 내포하고 있으며, 모델 출력을 제어할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 SAEs 프레임워크를 도입하여 LLM의 내부 구조와 언어적 능력의 인과 관계를 분석합니다. SAELing이라고 명명된 이 방법론은 LLM의 숨겨진 상태를 고차원 특징 공간으로 분해하여 각 차원이 단일 의미를 대표하도록 합니다. 연구팀은 최소 대조 데이터셋과 반사실적 문장 데이터셋을 구축하여 언어적 특성을 조작하고, Feature Representation Confidence (FRC)와 Feature Intervention Confidence (FIC) 지표를 통해 효과를 평가합니다.

- **Performance Highlights**: SAELing 접근 방식은 Llam-3.1-8B 모델에 적용되어 언어적 능력의 주요 특징을 효과적으로 식별하고 언어 모델을 조작할 수 있는 신뢰할 수 있는 방법을 제공합니다. 이 연구 결과는 LLM이 실제로 언어 지식을 갖추고 있으며, 향후 연구에서 더 해석 가능하고 제어 가능한 언어 모델링의 기초를 마련합니다.



### Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners (https://arxiv.org/abs/2502.20339)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 성능을 향상시키기 위해 계산 리소스를 테스트 시 확장하는 전략을 모색합니다. 특히, 단순하고 고속의 하위 이차(subquadratic) 아키텍처 모델이 인프라 예산에 따라 기존 Transformer보다 더 높은 성능을 낼 수 있는지를 조사합니다. 이 과정에서 Mamba라는 새로운 모델을 도출하여, 기존의 대형 모델보다 더 효율적인 추론 성능을 보여주는 새로운 가능성을 열었습니다.

- **Technical Details**: 상세적으로, 연구팀은 하이브리드 및 순수 Mamba 모델을 선행 학습된 Transformer로부터 증류(distill)하여 생성했습니다. 이들 모델은 80억 개의 토큰으로 훈련되었고, 고속 추론과 스케일링에서 강력한 성능을 발휘합니다. 이 모델들의 특징은 긴 생성 시퀀스와 대량 배치에 대한 메모리 소모를 줄이면서도 뛰어난 수학적 추론 성능을 유지하는 것입니다.

- **Performance Highlights**: 연구의 결과, 순수 및 하이브리드 Mamba 모델은 기존 Transformer 모델과 비교하여 MATH 및 GSM8K 수학 추론 과제에서 더 좋은 범위와 정확성을 보여주었습니다. 특히, 이러한 모델들은 동일한 품질의 결과를 더 적은 추론 시간으로 달성할 수 있으며, 이는 효율성과 추론 능력 간에 더 나은 균형을 이루고 있음을 보여줍니다.



### Expertise Is What We Wan (https://arxiv.org/abs/2502.20335)
Comments:
          18 pages, 7 figures, 5 tables

- **What's New**: 이 논문에서는 Large Language Expert (LLE)라는 새로운 응용 아키텍처를 소개합니다. LLE는 Large Language Models (LLMs)의 유연성과 강력함을 Expert Systems의 해석 가능성, 설명 가능성, 신뢰성과 결합한 시스템입니다. 이를 통해 암 진단 및 치료를 위한 작업을 지원할 수 있으며, 기존의 가이드라인과 데이터 통합의 어려움을 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LLE 아키텍처는 규칙 기반 시스템과 LLM의 강점을 결합하여 실제 의료 데이터를 다루는 복잡한 임상 가이드라인을 적용합니다. 이 시스템은 자연어와 구조화된 논리로 구성된 지식 기반으로 운영되며, LLM을 통해 진단 작업의 공백을 신속히 식별합니다. LLE는 또한 표준화된 구조와 적절한 로직을 유지함으로써, 의사들이 신뢰할 수 있는 결정을 내릴 수 있도록 돕습니다.

- **Performance Highlights**: LLE 시스템은 실제 환자의 비구조적 건강 기록을 분석하여 높은 임상 정확도(95% 이상)를 달성했습니다. 암 진단 스크리닝과 치료에 있어서 가이드라인 기반 의사 결정을 지원하며, 최적의 환자 결과를 위한 시의적절한 치료 시작을 보장합니다. 이 시스템은 대형 학술 센터의 유방암 및 대장암 환자 데이터로 확인된 여러 격차를 효과적으로 해결했습니다.



### Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models (https://arxiv.org/abs/2502.20332)
- **What's New**: 이 연구는 대형 언어 모델인 Llama3-70B에서 추상적 규칙 유도를 지원하는 내부 메커니즘을 심층적으로 조사하고, 상징적 기제를 구현하는 새로운 아키텍처를 제시합니다. 모델은 입력 토큰을 토대로 추상 변수로 변환하고, 이러한 변수를 이용해 순서 유도를 수행한 뒤, 마지막으로 다음 토큰을 예측하는 세 단계를 통해 추상적 추론을 지원합니다. 이러한 결과는 신경망의 emergent reasoning(출현적 추론)이 상징적 메커니즘의 출현에 의존함을 시사합니다.

- **Technical Details**: 연구는 Llama3-70B 모델의 세 가지 주요 단계 아키텍처를 설명합니다. 첫 번째 단계에서, 입력 토큰이 관계를 기반으로 추상 변수로 변환되고, 두 번째 단계에서는 이러한 변수들에 대한 순서 유도가 이루어집니다. 마지막 단계에서는 미리 예측된 추상 변수에 관련된 값을 검색하여 다음 토큰을 예측하는 기능을 수행합니다. 이러한 과정은 symbol abstraction heads(기호 추상화 헤드), symbolic induction heads(상징적 유도 헤드), retrieval heads(검색 헤드)로 각각 명명된 주의를 통해 처리됩니다.

- **Performance Highlights**: Llama3-70B는 주어진 문제에 대해 95%의 정확도로 의사 결정 문제를 해결하여, 모델이 기호 처리 메커니즘을 가지고 있음을 보여줍니다. 이 연구는 기호적 접근법과 신경망 접근법 간의 오랜 논쟁을 해결하는데 기여할 뿐만 아니라, 추상 규칙 유도를 효과적으로 수행하는 데 필요한 구조화된 메커니즘을 밝혀냄으로써 LLM(대형 언어 모델)의 능력에 대한 새로운 통찰을 제공합니다.



### Long-Context Inference with Retrieval-Augmented Speculative Decoding (https://arxiv.org/abs/2502.20330)
- **What's New**: 최근 긴 문서 처리를 위한 대형 언어 모델(LLM)의 발전은 전통적인 검색 증강 생성(RAG) 방식의 유망한 대안을 제공합니다. 하지만 긴 컨텍스트 추론은 키-값(KV) 캐시 관리에서의 높은 계산 오버헤드 문제로 인해 효율성에 도전받고 있는 상황입니다. 본 논문에서는 검색 증강 추측 디코딩(RAPID)라는 방법을 제안하여 긴 컨텍스트 추론의 속도를 높이고 생성 품질을 향상시키는 새로운 접근 방식을 설명합니다.

- **Technical Details**: RAPID는 RAG 드래프터를 통해 긴 문서의 생성을 추측하며, 이 드래프터는 짧은 검색 컨텍스트에서 작동합니다. RAG 드래프터는 긴 컨텍스트의 LLM과의 비교에서 높은 효율성을 유지하며, 특히 긴 문서 처리에 강점을 가지고 있습니다. RAPID는 자기 추측과 상향 추측 두 가지 설정에서 작동하며, 이는 RAG 드래프터가 동일 또는 더 큰 파라미터 규모를 가질 수 있는 것을 특징으로 합니다.

- **Performance Highlights**: RAPID는 LLaMA-3.1 과 Qwen2.5의 실험을 통해 성능 개선이 입증되었습니다. 특히, LLaMA-3.1-8B 모델에서 39.33에서 42.83로 성능이 증가하며, 2.69배의 속도 향상을 보여주었습니다. 이 연구는 또한 RAPID가 차세대 긴 문서 추론 기술로서 실용적인 대화 시스템에서 우수한 생성 품질을 나타낼 수 있음을 밝혔습니다.



### LangProBe: a Language Programs Benchmark (https://arxiv.org/abs/2502.20315)
- **What's New**: 이번 논문에서는 LangProBe라는 새로운 대규모 벤치마크를 소개하고 있습니다. LangProBe는 2000개 이상의 작업, 아키텍처, 최적화 기법, 언어 모델(LM) 조합을 평가하여 언어 프로그램 아키텍처와 최적화 전략의 영향을 연구합니다. 이 연구는 다양한 라인업의 언어 프로그램을 통해 최적화된 언어 프로그램이 비용 대비 품질을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: LangProBe의 기반이 되는 연구는 DSPy와 TextGrad와 같은 선언적 언어 프레임워크를 통해 언어 프로그램을 작성하고 자동화하는 접근 방식을 활용하고 있습니다. 이러한 프로그램은 외부 도구와의 통합 및 정보 흐름을 구성하고, 특히 외부 정보에 대한 접근을 요구하는 작업에 필수적입니다. 또한, MIPRO과 같은 최적화 기법들은 다양한 모델과 작업 조합에 대해 품질 향상을 제공하는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, 최적화된 언어 프로그램은 일반적인 모델 호출 방식보다 우수한 성능을 보이는 것으로 나타났습니다. 예를 들어, gpt-4o-mini에서 실행되는 최적화된 프로그램은 낮은 비용으로 뛰어난 성능을 발휘했습니다. 그러나 모든 문제에서 일관된 결과를 보이는 것은 아니며, 고급 모델을 통한 기본 문제 해결에 있어서는 추가적인 조합이나 최적화가 필요하지 않았습니다.



### How Much is Enough? The Diminishing Returns of Tokenization Training Data (https://arxiv.org/abs/2502.20273)
- **What's New**: 본 논문은 토크나이저(tokenizer) 훈련 데이터의 크기가 1GB에서 900GB까지의 범위에서 토크나이징 품질에 미치는 영향을 조사합니다. 그 결과, 데이터 크기가 증가함에 따라 수익 감소(diminishing returns) 현상이 발생하며, 이는 토크나이징 품질을 향상시키기 위한 스케일링(data scaling)의 한계점을 강조합니다. 이러한 결과는 토크나이징 과정 최적화에 대한 귀중한 통찰을 제공하며, 향후 토크나이징 알고리즘에 대한 연구 방향을 제시합니다.

- **Technical Details**: 훈련 데이터로는 900GB의 텍스트로 구성된 PILE와 RedPajama 데이터셋이 사용되었습니다. BPE, UnigramLM, WordPiece의 세 가지 토크나이저를 각각 훈련시켰으며, 훈련 데이터 크기를 1GB, 5GB, 15GB, 30GB 등 점진적으로 늘려가며 연구하였습니다. 그 결과, 훈련 데이터가 증가함에 따라 수많은 단어 토큰의 공유 비율이 증가하며, 각 토크나이저에서 유사한 어휘 구성을 보여줍니다.

- **Performance Highlights**: 내부 지표에 따라 토크나이징 품질을 연구한 결과, 데이터 크기가 증가해도 BPE 토크나이저의 성능이 150GB에서 180GB 범위에서 플라토(patters)됩니다. 이는 데이터 크기만 늘리는 것이 토크나이저의 효과성을 본질적으로 개선하지 않을 수 있음을 시사합니다. 또한, 훈련 데이터가 증가하면서 추가되는 대부분의 토큰은 저빈도(low-frequency)이며, 이로 인해 단어 어휘의 안정성을 유지하는 경향이 나타났습니다.



### LLM as a Broken Telephone: Iterative Generation Distorts Information (https://arxiv.org/abs/2502.20258)
- **What's New**: 이 연구는 대형 언어 모델(LLM)이 반복적으로 자신의 출력을 처리할 때 정보 왜곡을 조장하는지를 조사합니다. '편의점 전화'(broken telephone) 효과에 영감을 받아, LLM의 반복 생성 과정을 통해 정보 왜곡이 축적되는지를 분석하였습니다. 실험 결과, 언어 선택과 체인 복잡성에 따라 왜곡이 증가하며, 전략적인 프롬프트 생성 기법을 통해 이를 완화할 수 있음이 밝혀졌습니다.

- **Technical Details**: 연구는 번역 기반 실험을 통해 LLM의 반복 생성에서의 정보 왜곡을 조사합니다.  각 반복(iteration) 과정에서 영어 문서가 다양한 언어로 번역된 후 다시 영어로 역번역되며, 이 과정에서 왜곡 정도와 사실성(factuality)을 측정하였습니다. 연구 결과는 중간 언어의 선택과 체인 복잡성에 따라 왜곡이 어떻게 영향을 받는지를 보여주고, 고온 조절(temperature control)과 제한된 프롬프트(use of restricted prompting) 기법으로 왜곡을 줄일 수 있다는 점을 강조합니다.

- **Performance Highlights**: 결과적으로 LLM의 반복 생성이 정보의 의미와 사실성을 손상시킬 수 있으며, 반복 처리에서 왜곡이 누적됨을 보여줍니다. 특히, 언어 구조의 유사성과 모델의 학습 데이터에 따라 왜곡 정도가 달라지며, 번역 체인의 복잡성이 클수록 왜곡이 더욱 커지는 경향이 있음을 보여주었습니다. 이러한 발견은 LLM이 생성한 콘텐츠의 신뢰성에 대한 우려를 제기하며, AI 기반 정보 확산의 장기적 영향에 대한 논의에 기여합니다.



### Beyond Natural Language Perplexity: Detecting Dead Code Poisoning in Code Generation Datasets (https://arxiv.org/abs/2502.20246)
- **What's New**: 최근 데이터 보안에 대한 우려가 커진 가운데, 대형 언어 모델(LLM) 훈련 데이터의 안전성을 확보하기 위해 새로운 방법론이 요구되고 있습니다. 본 논문에서 제안하는 "Dead Code Perplexity Analysis (DePA)"는 기존의 토큰 기반 탐지 방법의 한계를 극복하고, 코드의 구조적 특성을 반영한 라인 단위의 새로운 탐지 및 정화 방법을 제시합니다. DePA는 코드 줄 간의 문맥적 관계를 활용하여 비정상적인 줄을 효과적으로 식별하고, 성능 향상이 입증되었습니다.

- **Technical Details**: DePA는 라인 레벨의 perplexity(혼란도) 측정을 통해 코드에서 유해한 코드 조각을 탐지합니다. 이 방법은 코드를 줄 단위로 처리하고, 각 줄의 perplexity를 전체 파일의 분포와 비교하여 이상행위를 발견합니다. 또한, DePA는 기존의 방법들보다 평균 0.14-0.19 높은 F1 점수를 달성하며, 44-65%의 정확도로 악성 코드의 위치를 정밀하게 찾을 수 있도록 설계되었습니다.

- **Performance Highlights**: DePA는 더 빠른 탐지 속도를 자랑하며, 최대 23배의 속도 개선을 나타내었습니다. 실험 결과, DePA는 ONION과 같은 기존 방법들에 비해 F1 점수에서 현저한 성능 향상을 보였습니다. 전반적으로, DePA는 코드 생성 모델 훈련 데이터의 무결성을 보호하기 위한 강력하고 효율적인 솔루션으로 자리매김하고 있습니다.



### From Retrieval to Generation: Comparing Different Approaches (https://arxiv.org/abs/2502.20245)
Comments:
          work on progress

- **What's New**: 이 논문은 오픈 도메인 질문 응답(Open-Domain Question Answering, ODQA) 및 관련 작업에 대한 정보 검색 및 생성 모델의 성능을 체계적으로 평가합니다. 특히, Dense Passage Retrieval(DPR) 및 하이브리드 모델을 통해 정보 검색과 생성 간의 균형을 탐구합니다. 전통적인 모델(BM25 등)의 한계와 Generative 모델의 정확성 문제를 극복하기 위한 접근 방식을 제안합니다.

- **Technical Details**: 이 논문은 정보 검색 기반 모델, 생성 모델, 그리고 이 둘을 결합한 하이브리드 모델을 포함하여 ODQA 및 문서 재정렬 작업에 대한 성능을 평가합니다. Dense retrieval 모델은 쿼리와 문서를 밀집 표현으로 인코딩하여 관련 정보를 효율적으로 제공합니다. 그러나 하이브리드 모델은 정답의 정확성과 생성의 유연성 간의 균형을 필요로 하며, 재정렬 및 문서 선택 전략이 필수적이라는 점을 강조합니다.

- **Performance Highlights**: DPR 모델은 ODQA에서 50.17%의 최상위 정확도를 기록하며 강력한 성능을 나타냈습니다. 하이브리드 모델은 BEIR 데이터셋에서 BM25의 nDCG@10 점수를 43.42에서 52.59로 개선하여 문서 재정렬에서의 강점을 시사합니다. 또한, BM25는 언어 모델링 작업에서 다른 방법에 비해 낮은 perplexity를 기록하여 정보 검색과 생성의 연계를 위한 유용성을 강조합니다.



### FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving (https://arxiv.org/abs/2502.20238)
- **What's New**: 이번 연구에서는 FINEREASON이라는 새로운 로직 퍼즐 기준을 제안하여 대형 언어 모델(LLMs)의 추론 능력을 세밀하게 평가할 수 있는 방법을 제공합니다. 이 기준은 퍼즐의 각 단계를 원자적(atomic)으로 분해하여 중간 상태의 정확성을 엄격하게 검증할 수 있도록 설계되었습니다. 이를 통해 모델 내부의 반성(reflection)과 수정(correction) 능력을 평가함으로써 보다 심층적인 추론 과정을 이해할 수 있습니다.

- **Technical Details**: FINEREASON은 4가지 유형의 퍼즐(스도쿠, 그래프 색칠, 24 게임, 그리드 퍼즐)을 포함하고 있으며, 각 퍼즐은 일련의 개별 단계로 해결됩니다. 평가의 구조는 두 가지 주요 액션인 상태 체크(state checking)와 상태 전환(state transition)을 통해 이루어집니다. 상태 체크는 현재 상태가 해결 가능한지 예측하며, 상태 전환은 다음 유효한 단계로 나아가거나 이전 상태로 되돌아가는 과정을 다룹니다.

- **Performance Highlights**: MODELS는 상태 체크 및 전환 데이터로 훈련받은 결과 수학적 추론에서 최대 5.1%의 향상을 보여주었습니다. 특히, OpenAI-o1은 Gemini-2.0-Flash-Thinking를 19.7% 큰 차이로 초과하였으며, 교육을 전적으로 데이터에 의존하는 일반 모델들은 깊은 추론에서 어려움을 겪었습니다. 이 연구는 LLMs의 성능을 향상시키기 위한 접근 방식으로써, 보다 포괄적인 추론 기준이 필요함을 강조합니다.



### ChineseEcomQA: A Scalable E-commerce Concept Evaluation Benchmark for Large Language Models (https://arxiv.org/abs/2502.20196)
- **What's New**: 중국어 전자 상거래 질문 응답 벤치마크인 ChineseEcomQA가 제안됩니다. 이 벤치마크는 전자상거래의 기본 개념을 중심으로 구축되었고, 기존 LLMs가 경험하는 사실적인 부정확성 문제를 해결하기 위해 설계되었습니다. ChineseEcomQA는 다채로운 태스크 유형을 다루고, 전자상거래 분야의 일반성과 전문성을 구분하는 데 중점을 둡니다.

- **Technical Details**: 이 벤치마크는 20개 주요 산업과 10개 핵심 개념 차원을 포함하여 1,800개의 질문-답변 쌍으로 구성되어 있습니다. 특수한 전자상거래 지식을 기반으로 하여 LLM 검증, Retrieval-Augmented Generation (RAG) 검증, 그리고 엄격한 수작업 주석 과정을 결합한 하이브리드 데이터셋 구축 프로세스를 활용합니다. 이를 통해 다양한 전자상거래 태스크에 적용할 수 있는 기본 개념을 제공하고, 전자상거래 이론과 모델의 성능을 검증할 수 있습니다.

- **Performance Highlights**: 주요 모델인 Deepseek-R1과 Deepseek-V3는 전자 상거래 분야에서 LLM의 잠재력을 나타내는 결과를 보여줍니다. 하지만, 많은 최첨단 모델이 특정 하위 개념에서 60% 미만의 정확도에 그쳐 상당한 도전과제를 나타냅니다. RAG 전략 도입시, 다양한 크기의 모델에서 성능 향상을 보이며 모델 간 격차를 줄이고 있습니다.



### Layer-Aware Task Arithmetic: Disentangling Task-Specific and Instruction-Following Knowledg (https://arxiv.org/abs/2502.20186)
- **What's New**: 논문에서 제안하는 Layer-Aware Task Arithmetic (LATA)는 과거의 태스크 산술(task arithmetic, TA) 방법을 개선하여, 각 레이어에 대해 특정 가중치를 부여하는 방식을 사용합니다. 이 접근법은 목표 태스크와 강하게 연관된 레이어는 강조하고, 지침 수행(instruction-following)과 관련된 레이어는 억제하여 모델의 태스크 학습 및 잊기 성능을 향상시킵니다. LATA는 여러 태스크의 성능을 유지하면서 기존 방법들보다 더 높은 태스크 정확도를 달성하는 데 중점을 두고 있습니다.

- **Technical Details**: LATA는 네 단계로 구성되며, 첫 번째 단계에서는 베이스 모델과 사전 훈련(pre-trained) 모델의 파라미터 차이를 통해 지침 벡터를 정의합니다. 두 번째 단계에서, 각 세분화된 태스크 모델의 파라미터에서 베이스 모델의 파라미터를 빼도록 하여 복합 벡터(complex vector)를 도출합니다. 각 레이어의 파라미터로 이루어진 레이어 벡터(layer vector)를 생성한 후, 두 벡터 간 코사인 유사도(cosine similarity)를 계산하여 태스크 관련 요소를 구분합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋인 WikiText-2, GSM8K 및 HumanEval에서 진행된 실험 결과 LATA는 다중 작업 학습(multi-task learning) 및 선택적 태스크 잊기(selective task forgetting)에서 기존 방법들을 초월하는 성과를 보여주었습니다. LATA는 불필요한 능력을 선택적으로 제거할 때도 전체 성능의 최소 손해를 보이며 효과적인 결과를 도출했습니다. 이 연구는 태스크 전용 지식과 일반 용도 지식을 분리하는 데 있어 레이어별 분석의 중요성을 강조합니다.



### Representing Signs as Signs: One-Shot ISLR to Facilitate Functional Sign Language Technologies (https://arxiv.org/abs/2502.20171)
- **What's New**: 이 논문에서는 언어 독립적인 신규 접근 방식으로서의 일회성(one-shot) 수화 인식(Isolated Sign Language Recognition, ISLR) 기술을 제안한다. 전통적인 언어 특정 접근법의 한계를 극복하고, 다양한 언어와 진화하는 어휘에 걸쳐 일반화 가능한 방법을 제시하였다. 특히, 사전 훈련(pretraining)된 모델을 활용하여 수화를 효율적으로 임베딩하고 새로운 수화를 신속하고 정확하게 인식할 수 있도록 하는 밀집 벡터 검색(dense vector search) 방식을 도입했다.

- **Technical Details**: 우리의 접근 방식은 두 단계로 구성된다. 첫 번째 단계에서는 다양한 수화를 대표할 수 있는 모델을 사전 훈련하여 그 필수 특징을 캡처한다. 두 번째 단계에서는 이 모델을 동결(frozen) 상태로 유지하여, 신규 수화를 기존의 수화에 대한 임베딩과 밀접하게 매핑하여 인식의 정확도를 향상시킨다. 10,235개의 독특한 수화를 포함하는 대규모 사전에서 50.8%의 일회성 MRR(Mean Reciprocal Rank)이라는 최첨단 결과를 도출했다.

- **Performance Highlights**: 이 연구는 수화 기술의 확장성과 효과를 향상시킬 수 있는 진보적 솔루션을 제공하며, 여러 언어에 걸쳐 강력한 일반화를 보여준다는 점에서 그 성과가 빛난다. 수화의 진화적인 성격에 적응할 수 있는 능력을 바탕으로 막대한 어휘의 인식을 가능하게 한다. 또한, DHH(청각장애인 및 난청자) 커뮤니티와의 공동 창작 전략을 통해 실제 필요에 부합하는 도구를 제공할 수 있다는 점에서 긍정적인 평가를 받았다.



### Educator Attention: How computational tools can systematically identify the distribution of a key resource for students (https://arxiv.org/abs/2502.20135)
Comments:
          The first two authors QZ and REW contributed equally. The last two authors DD and SL advised equally

- **What's New**: 이 연구는 교육자의 주의(educator attention) 패턴을 대규모로 분석한 첫 번째 연구로, 100만 개 이상의 교육 발언(utterance)을 활용하여 학생의 인구통계학적(demographic) 및 학업 성취도(academic achievement) 데이터를 연결하여 분석하였습니다. 연구 결과, 교육자들이 낮은 성취도를 가진 학생들에게 더 많은 주의를 기울이는 경향이 있음을 발견했지만, 성별에 따른 차이가 발생했습니다. 또한, 흑인 학생들에 대한 주의의 차이와 영어 학습자(EL) 학생들 간의 주의 배분 차이도 확인되었습니다.

- **Technical Details**: 연구에서는 1,157,970개의 교육 발언이 포함된 데이터 세트를 분석하여 특정 학생의 성취도와 인구통계학적 요인에 따라 교육자의 주의가 어떻게 달라지는지를 조사하였습니다. 이를 위해 자연어 처리(NLP) 기법을 활용하여 교육자 발언의 방향과 성격을 자동으로 분류하는 모델을 개발하였고, 교육자의 발언을 학생 A, 학생 B, 두 학생 모두에게 향한 것 또는 불명확한 수신자로 분류할 수 있는 시스템을 구축하였습니다.

- **Performance Highlights**: 연구 결과, 낮은 성취도를 가진 여성 학생들은 높은 성취도를 가진 남성 학생으로부터 상당히 적은 주의를 받으며, 낮은 성취도를 가진 남성 학생들은 그와 반대로 높은 성취도를 가진 여성 학생보다 더 많은 주의를 받는 것으로 나타났습니다. 인종별로는, 낮은 성취도를 가진 흑인 학생들이 다른 흑인 학생과 함께 있을 때만 추가적인 주의를 받는 경향이 있고, 영어 학습자 학생들 중 높은 성취도를 가진 학생들이 낮은 성취도를 가진 동료들보다 더 많은 주의를 받는 것으로 분석되었습니다.



### Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking (https://arxiv.org/abs/2502.20129)
- **What's New**: 이 연구는 Chain-of-Thought (CoT) 방법이 Transformer 기반 대형 언어 모델의 성능을 크게 향상시키는 방법을 기존의 방법론과 비교하여 검토하였습니다. 이를 통해 CoT의 효과를 입증하고, late-layer MLP 뉴런이 세계 상태를 추적하는 데 중요한 역할을 한다는 것을 밝혔습니다. 또한, 압축(compression)과 구별(distinction)이라는 두 가지 메트릭스를 제안하여, 모델이 대칭 유한 상태 자동자(automaton, FSA)를 내부에 내재하고 있음을 보여주었습니다.

- **Technical Details**: 연구팀은 Transformer+CoT와 다른 모델 간의 상태 추적 능력을 평가하여, Transformer+CoT가 다양한 그룹에 대해 임의의 길이 순서를 효과적으로 학습할 수 있음을 입증했습니다. 또한, activation patching 기술을 사용하여 내부 메커니즘을 분석하고, late-layer MLP 뉴런이 상태를 추적하는 데 주로 사용됨을 확인했습니다. 압축 메트릭스와 구별 메트릭스를 통해 모델이 세계 모델(FSA)을 재구성하는 데 거의 100% 정확성을 달성했다는 것을 발견했습니다.

- **Performance Highlights**: 제안한 접근 방식은 노이즈가 있는 환경에서도 강력한 알고리즘을 학습하는 데 성공하였으며, 모델의 견고성(robustness)을 입증합니다. 특히, Transformer+CoT는 복잡한 작업에서도 상태 추적 기능을 지원할 수 있는 이론적 기초를 제공하며, 이는 다양한 다운스트림 작업에 적용이 가능합니다. 추가적으로, 실험 결과는 CoT 방법이 다양한 실제 시나리오에서 효과적으로 작동하며, 복잡한 문제를 해결하는 데 도움을 줄 수 있음을 나타냅니다.



### Self-Training Elicits Concise Reasoning in Large Language Models (https://arxiv.org/abs/2502.20122)
Comments:
          23 pages, 10 figures, 18 tables

- **What's New**: 본 논문에서는 체인의 사고(Chain-of-thought, CoT) 추론 메커니즘을 통해 대형 언어 모델들이 복잡한 작업을 해결할 수 있는 능력이 향상되었음을 설명합니다. 그러나 기존의 모델들이 과도한 토큰을 생성하고 있으며, 이는 불필요한 추론 비용을 초래한다고 주장합니다. 이에 대한 해결책으로, 저자들은 자가 생성된 간결한 추론 경로를 활용한 간단한 파인튜닝 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 Zero-shot prompting이 간결한 추론을 효과적으로 유도하는 데 한계가 있음을 보여주며, 대신 Best-of-N (BoN) 샘플링과 Few-shot conditioning을 활용하여 모델을 파인튜닝하는 방법을 제시합니다. 저자들은 GSM8K와 MATH 데이터셋을 활용하여 다양한 모델 패밀리에서 평균 30%의 출력 토큰 수 감소를 달성하였으며, 이는 이전의 파인튜닝 기준 대비 2.4배 향상된 결과입니다. 이 과정을 통해, 모델들은 질문의 복잡성에 따라 출력 길이를 적절히 조정할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 본 논문에서 제안된 FS-BoN 방법은 모델의 경량화된 추론 경로를 효율적으로 이끌어내어, 복잡한 작업에 대한 추론 비용을 줄이는 데 효과적임을 입증했습니다. 성능 분석에 따르면, 훈련된 모델은 문제의 난이도에 따라 변별력 있는 출력을 유지하며 적절한 자세로 응답을 조정합니다. 이러한 결과는 다양한 모델 스케일에서도 일관되게 유지되었으며, 자가 생성된 데이터의 파인튜닝이 LLM의 잠재적인 간결한 추론 능력을 발휘할 수 있도록 하는 데 기여할 수 있음을 시사합니다.



### LongRoPE2: Near-Lossless LLM Context Window Scaling (https://arxiv.org/abs/2502.20082)
- **What's New**: LongRoPE2는 사전 학습된 대형 언어 모델(LLMs)의 효과적인 맥락 창을 목표 길이로 확장하는 혁신적인 접근 방식입니다. 이 방법은 기존의 짧은 맥락 창에서 성능을 유지하면서 세 가지 방법론 기여를 통해 이루어졌습니다. 특히, LLaMA3-8B 모델을 128K의 효과적인 맥락 길이로 확장하면서도 단기 맥락 성능의 98.5% 이상을 유지합니다.

- **Technical Details**: LongRoPE2는 Rotary Positional Embeddings (RoPE)를 활용하여 연장된 맥락에서 발생하는 out-of-distribution(OOD) 문제를 해결하는 데 집중합니다. RoPE의 차원 간 재조정 알고리즘을 개발하여, 'needle-driven' perplexity(PPL)를 기반으로 최적의 조정 팩터를 식별합니다. 이 과정에서는 짧은 맥락과 긴 맥락을 동시에 학습하는 혼합 맥락 창 훈련 방식을 사용하여, 긴 문서에 대해 조정된 RoPE를 적용합니다.

- **Performance Highlights**: LongRoPE2는 다양한 벤치마크에서 Phi3-mini-3.8B와 LLaMA3-8B에서 매우 우수한 성능을 보여줍니다. 또한, 기존의 방법들보다 훨씬 적은 10B의 훈련 토큰으로도 LLaMA3-8B-128k의 긴 맥락 성능을 메타의 접근 방식보다 뛰어넘습니다. 이로 인해 LongRoPE2는 단기 맥락 성능을 97% 이상 유지하고 있으면서도 긴 맥락 처리 능력을 크게 향상시켰습니다.



### Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents (https://arxiv.org/abs/2502.20073)
Comments:
          25 pages, 14 figures

- **What's New**: 본 논문에서는 LLM(대형 언어 모델) 기반의 새로운 멀티 에이전트 시스템(LLM-MAS) 벤치마크인 Collab-Overcooked를 제안합니다. 이 벤치마크는 Overcooked-AI 게임을 기반으로 하며, 상호작용 환경에서 도전적이고 적용 가능한 여러 가지 작업을 포함하고 있습니다. 기존 벤치마크는 다루지 않았던 협력 능력 평가를 위한 다양한 프로세스 지향적 지표들을 도입하여 LLM 에이전트의 미세 협력 능력을 평가할 수 있도록 합니다.

- **Technical Details**: LLM-MAS는 복잡한 작업을 해석하고 계획하는 데 있어 LLM의 제로샷(Zero-Shot) 및 피샷(Few-Shot) 학습 능력을 활용하고 있습니다. LLM-MAS는 목표 해석을 넘어선 세 가지 필수 협력 능력을 요구하는데, 여기에는 능력 경계 인식(competence boundary awareness), 의사소통(communication), 동적 적응(dynamic adaptation)이 포함됩니다. 이러한 협력 평가 프레임워크를 구축하는 것이 LLM-MAS의 효과성을 평가하는 데 중요합니다.

- **Performance Highlights**: 연구에서는 다양한 크기의 10개 LLM을 대상으로 광범위한 실험을 진행하였으며, 목표 해석에서 강력한 능력을 보였지만, 복잡한 작업을 효율적으로 수행하기 위한 활성 협력 및 지속적인 적응에서 상당한 차이가 존재함을 보여주었습니다. 또한 Collab-Overcooked는 여러 복잡성 수준에서 30개의 연속적, 프로세스 특정 작업을 포함하여 LLM 간 협력 평가의 중요한 한계를 인식하고 이를 극복하기 위한 통찰을 제공합니다.



### Connecting the Persian-speaking World through Transliteration (https://arxiv.org/abs/2502.20047)
- **What's New**: 이 논문은 타지크어 (Tajik Persian)와 이란(Farsi) 또는 아프간 (Afghan) 텍스트의 이해를 개선하기 위한 새로운 기계 전사 (transliteration) 접근 방식을 제시합니다. 타지크어 화자는 같은 언어의 상호 인지 가능한 변형을 사용하지만, 서로 다른 문자 체계로 인해 서로의 문서를 이해할 수 없습니다. 이에 따라 기계 번역 (machine translation)보다 기계 전사가 더 실용적이라는 주장을 지원하는 연구 결과를 제공합니다.

- **Technical Details**: 위 논문은 변환기(Transformer) 기반의 G2P (grapheme-to-phoneme) 접근 방식을 사용하여 타지크어와 페르시아어 간의 전사를 수행하였습니다. 새로운 이중 문자 데이터셋을 통해 Farsi에서 Tajik으로와 Tajik에서 Farsi으로의 chrF++ 점수를 각각 58.70, 74.20으로 달성하였으며, 이는 향후 연구의 기준으로 사용할 수 있습니다. 이 연구는 두 방향 모두에서의 과제가 비단순하다는 점을 시사합니다.

- **Performance Highlights**: 연구 결과는 타지크어와 페르시아어 간의 기계 전사 작업의 복잡성을 강조하였습니다. 새로 제시된 데이터셋을 통해 전사 성능이 측정되었으며, 이는 향후 타지크-페르시아어 전사 작업에 유익한 기준을 제공합니다. 추가적으로 두 문자 체계 간의 차이점과 전사에 따른 도전 과제를 개관함으로써 향후 연구에 기여하고자 하였습니다.



### Polish-ASTE: Aspect-Sentiment Triplet Extraction Datasets for Polish (https://arxiv.org/abs/2502.20046)
- **What's New**: 이번 논문에서는 Aspect-Sentiment Triplet Extraction (ASTE)이라는 감정 분석의 복잡한 작업을 다루며,  폴란드어 고객 리뷰에 기반한 두 개의 새로운 데이터셋을 소개합니다. 이 데이터셋은 호텔 및 구매 제품에 대한 의견을 포함하며, 폴란드어 ASTE 연구를 위한 기반을 제공합니다. 또한, 기존의 ASTE 기술과 최첨단 대형 언어 모델을 결합하여 실험을 수행하고 기술의 성능을 평가하였습니다.

- **Technical Details**: 본 연구에서 제안된 데이터셋은 Wroclaw Corpus of Consumer Reviews Sentiment (WCCRS)에서 추출한 고객 리뷰로 구성되어 있습니다. 리뷰는 문장 및 문서 수준에서 감정 극성을 제공하지만, 기초적인 ASTE 트리플 구성을 위해서는 후기 내용의 감정 극성을 사용하지 않았습니다. 각 문장은 aspect phrase, opinion phrase, sentiment polarity를 포함하는 방식으로 주석이 달렸으며, 주석 작업은 네이티브 폴란드어 화자가 진행하였습니다.

- **Performance Highlights**: 실험 결과, 두 개의 새로운 데이터셋은 기존 ASTE 기술과의 결합에서 높은 성능을 보였으며, 데이터셋의 난이도를 평가하는 데에도 유용하였습니다. 주석 품질은 전문가에 의해 검토되었고, 두 주석 간 일치를 측정한 결과 높은 일치도를 보였습니다. 새로운 데이터셋은 연구자의 접근 가능성을 고려하여 자유로운 라이센스하에 제공됩니다.



### Erasing Without Remembering: Safeguarding Knowledge Forgetting in Large Language Models (https://arxiv.org/abs/2502.19982)
- **What's New**: 이번 논문은 대규모 언어 모델(LLMs)에서의 기계 학습 삭제(machin learning unlearning)의 새로운 관점을 탐구합니다. 기존 방법들이 단순한 표현만을 지우는 데 그치는 경향을 보이는 반면, 우리는 paraphrased 또는 관련 정보가 여전히 남아있음을 지적합니다. 이러한 문제를 해결하기 위해 UGBench라는 새로운 벤치마크를 도입하여, 기존 LLM 삭제 방법들의 일반화 성능을 평가하고 있습니다.

- **Technical Details**: 이 논문은 PerMU라는 새로운 방법을 제안합니다. PerMU는 adversarial examples를 활용하여 단어 임베딩에 무작위 노이즈를 주입하고, 이를 통해 모델이 사실적 정보를 기억하지 못하도록 만듭니다. 특히, 우리는 주제 토큰(subject tokens)의 민감도를 평가하는 새로운 메트릭 MSM을 도입하여, 가장 민감한 토큰을 불러올 수 없도록 조정합니다.

- **Performance Highlights**: PerMU는 기존 방법들에 비해 최대 50.13%의 삭제 성능 향상과 43.53%의 강력한 일반화 성능 개선을 달성했습니다. 이를 통해 LLM의 유용성을 유지하고 높은 생성 품질도 유지하는 데 성공하고 있습니다. 논문의 주요 기여는 새로운 일반화 평가 벤치마크 UGBench와 관련 사실 기억을 방지하는 기계 학습 삭제 방법 PerMU의 제안입니다.



### The Lookahead Limitation: Why Multi-Operand Addition is Hard for LLMs (https://arxiv.org/abs/2502.19981)
Comments:
          Pre-print

- **What's New**: 이 논문은 자가 회귀형 대형 언어 모델(LLMs)이 단순 산술, 특히 다중 피연산자의 덧셈 작업에서 어려움을 겪는 원인을 분석합니다. 연구 결과, 이러한 모델이 사용하는 단일 자릿수 미리 보기(lookahead) 휴리스틱이 두 개의 피연산자에 대해서는 어느 정도 효과적이지만, 더 복잡한 경우에는 실패한다는 사실을 밝혔습니다. LLMs는 덧셈 연산 시 carry-over 처리가 불가능해지는 경향이 있어, 더 복잡한 숫자 추론으로 일반화할 수 없는 한계를 드러냅니다.

- **Technical Details**: 연구에서는 두 개의 피연산자 덧셈과 다중 피연산자 덧셈을 포함하는 데이터셋을 작성하고, 다양한 토크나이제이션(tokenization) 전략을 사용하는 Mistral-7B, Gemma-7B, Llama-3-8B와 같은 모델을 비교했습니다. 이들 모델은 각기 다른 수치 입력 토크나이제이션 방법을 사용하며, 실험을 통해 모델이 다중 피연산자 덧셈에 있어 높은 정확도를 효과적으로 달성하지 못하는 이유를 규명했습니다. 특히, 모델들이 사용하는 간단한 휴리스틱은 carry propagation을 고려하지 못하여 성능 저하를 일으킵니다.

- **Performance Highlights**: 모델의 정확성 평가 결과, Mistral-7B, Gemma-7B, Meta-Llama-3-8B 모두 다중 피연산자 덧셈에서 성능이 급격히 저하되는 경향을 보였습니다. 숫자 피연산자의 수가 증가할수록 정확도가 감소하며, 이는 모델이 두 개 이상의 피연산자로 된 덧셈 작업을 효과적으로 일반화할 수 없음을 나타냅니다. 이러한 발견은 LLMs의 숫자 추론 능력에 대한 중요한 통찰을 제공하며, 보다 복잡한 산술 연산에 대한 이해를 요구합니다.



### Deterministic or probabilistic? The psychology of LLMs as random number generators (https://arxiv.org/abs/2502.19965)
Comments:
          31 pages, 12 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)을 사용하여 무작위 숫자 생성 시 성능을 체계적으로 조사하였습니다. 다양한 모델 아키텍처, 숫자 범위, 온도(temperature), 프롬프트 언어(prompt language) 등 여러 구성요소를 고려하였습니다. 흥미롭게도, 이들 모델이 확률적(transformers-based) 구조를 가지고 있음에도 불구하고 무작위 숫자 요청 시 결정론적(deterministic) 응답을 보이는 경우가 많다는 점을 발견하였습니다.

- **Technical Details**: 연구 결과, 모델 변경 및 프롬프트 언어의 변화에 따라 유의미한 차이가 나타났으며, 이는 훈련 데이터에 깊이 박힌 편향(bias) 때문으로 분석되었습니다. 특히 DeepSeek-R1와 같은 모델은 LLM의 내부 추론 프로세스를 부분적으로 밝힐 수 있었고, 비슷한 결과에 도달하였음에도 불구하고 각기 다른 방식으로 반응하였습니다. LLM이 진정한 무작위성(randomness)을 생성하지 못하는 이유는 인간의 인지적 편향을 재생산하기 때문임을 강조합니다.

- **Performance Highlights**: 편향성으로 인해 LLM은 예측 가능한 패턴을 생성하며, 이는 무작위성을 방해하는 요소로 작용합니다. 다양한 실험을 통해 LLM의 성능이 특정 모델과 구성에 따라 어떻게 달라지는지를 실질적으로 확인하였습니다. 이러한 결과는 LLM의 작동 방식과 그 한계를 이해하는 데 중요한 통찰을 제공합니다.



### Collaborative Stance Detection via Small-Large Language Model Consistency Verification (https://arxiv.org/abs/2502.19954)
- **What's New**: 이번 연구에서는 CoVer라는 새로운 프레임워크를 제안하여 소셜 미디어에서의 스탠스 감지를 위한 Large Language Model (LLM)와 Small Language Model (SLM)의 협업을 통해 성능을 개선하고 있습니다. CoVer는 텍스트를 개별적으로 처리하는 대신, 배치단위로 처리하여 LLM의 추론을 활용하면서도 SLM을 통해 논리적 일관성을 검증합니다. 이러한 접근 방식은 본래 LLM에 대한 의존도를 줄이고, 데이터 분석의 효율성을 높입니다.

- **Technical Details**: CoVer는 지식 증대(knowledge augmentation)와 무관한 맥락 필터링(irrelevant context filtering)을 통해 트윗의 맥락을 재구성하여 명확하고 편향 없는 스탠스 추론을 보장합니다. LLM은 동시에 여러 텍스트를 처리하여 맥락 재사용의 효율성을 제공합니다. 마지막으로, LLM의 추론 논리적 일관성을 검증하기 위해 SLM을 활용하며, 반복적으로 낮은 일관성을 보이는 텍스트에 대해 일관성 가중 집계(consistency-weighted aggregation)를 통해 최종 분류를 수행합니다.

- **Performance Highlights**: CoVer는 SemEval-2016, VAST 및 P-Stance와 같은 여러 벤치마크에서 최첨단 방법들보다 우수한 성과를 기록하며, 0.54 LLM 쿼리로 트윗 하나당 성능을 크게 향상시켰습니다. 이는 CoVer가 자원 효율적이면서도 높은 성능을 발휘할 수 있는 가능성을 보여줍니다. 이 연구는 스탠스 감지 과제에서 LLM과 SLM의 협업이 어떻게 효과적으로 결합될 수 있는지를 잘 나타내고 있습니다.



### GeoEdit: Geometric Knowledge Editing for Large Language Models (https://arxiv.org/abs/2502.19953)
- **What's New**: 이 논문에서는 Geometric Knowledge Editing (GeoEdit)라는 새로운 모델 편집 프레임워크를 제안합니다. GeoEdit는 파라미터 업데이트의 기하학적 관계를 활용하여 새로운 지식 업데이트와 일반 지식의 불필요한 영향을 구분합니다. 이를 통해 모델의 일반화 능력을 보존하면서 새로운 지식을 효과적으로 통합할 수 있는 방법론을 제공합니다.

- **Technical Details**: GeoEdit는 'forget-then-learn' 편집 전략을 도입하여, 기존 지식과의 상충을 최소화하며 새 지식에 대한 업데이트를 진행합니다. 또한, 중요도 기반의 작업 벡터 융합 기법을 도입해 불필요한 정보를 제거하고, 적응형 뉴런 레벨 가중치를 제공합니다. 이를 통해 높은 차원 공간에서의 각도를 계산하고, 노이즈를 억제하며 모델 편집 성능을 향상시킵니다.

- **Performance Highlights**: 다양한 데이터세트에 대한 실험 결과, GeoEdit는 기존 기술보다 뛰어난 성능을 보였습니다. 기존의 F-Learning의 한계를 극복하고, Locality 메트릭을 7.4% 향상시켰으며, Reliability 및 Generality 메트릭에서도 최상의 성능을 유지하였습니다. 따라서 GeoEdit는 LLM의 지식 업데이트를 위한 매우 효과적인 접근 방식임을 입증했습니다.



### Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation (https://arxiv.org/abs/2502.19941)
- **What's New**: 새로운 프레임워크 ADSQE는 합성 품질 추정 데이터에서의 분포 변화를 완화하는 데 중점을 둡니다. 이 방법은 구속 빔 탐색(constrained beam search) 알고리즘을 활용하여 합성 번역의 주요 구조를 유지하면서 생성 가능성을 극대화합니다. ADSQE는 또한 효율적인 주석 캡처를 위해 참조 데이터를 사용하여 각 어절 수준의 주석 품질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: ADSQE는 기계 번역의 품질 추정(quality estimation) 문제를 다룹니다. 이 시스템은 두 개의 모델, 즉 생성기(Generator)와 주석자(Annotator)를 훈련시키고, 구속 빔 탐색을 통해 합성 번역을 생성합니다. 여기서 주석자는 불일치하는 부분에 대해 세부적인 심각도를 재판단하여 오류를 평가하는 역할을 수행하며, 이 모든 과정에서 인간의 주석 방식을 모방한 알고리즘이 사용됩니다.

- **Performance Highlights**: 실험 결과 ADSQE는 감독 및 비감독 설정 모두에서 기존 SOTA(SOTA: state-of-the-art) 기준인 COMET을 능가한 성과를 보여주었습니다. 다양한 언어 방향(영어-독일어, 중국어-영어, 히브리어-영어)에서 테스트하여 우수한 결과를 도출했고, 합성 데이터 생성을 통해 다른 작업에 대한 보상 모델에 대한 유용성을 제공하는 통찰력을 제공합니다.



### Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative Agents (https://arxiv.org/abs/2502.19917)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 논문에서는 시각 중심의 선택(ViSA) 접근 방식을 제안하여 다중 모달 대형 언어 모델(MLLMs)의 이미지 처리 및 복잡한 지시 사항 이해 능력을 개선하고자 합니다. 기존 데이터셋이 질적으로 낮고 불일치한 경우가 많아 모델의 학습 효율을 저해하는 문제를 해결합니다. ViSA는 이미지 품질 평가와 이미지-지시 연결성 평가를 중심으로 구성됩니다.

- **Technical Details**: ViSA 접근 방식은 두 가지 핵심 요소로 구성됩니다. 첫째, 시각적 정보 정량화는 다양한 이미지 시각 요소를 계산하여 정보가 풍부한 이미지를 선택하는 방식입니다. 둘째, 이미지 중심의 지시 품질 정량화는 이미지를 잘 반영한 고품질 지시 데이터를 선택하는 데 초점을 맞추며, 이는 여러 에이전트를 활용하여 미세한 품질 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, ViSA를 적용한 모델은 원본 데이터의 2.5%만을 사용하고도 7개의 벤치마크에서 현재의 최첨단 모델과 동등하거나 뛰어난 성능을 보여주었습니다. 고품질 데이터의 중요성이 강조되며, 큰 모델에서도 소량의 고정보밀도 데이터를 활용해 성능 향상을 이루었습니다. 이를 통해 MLLMs의 훈련 효율성이 크게 증가함을 확인할 수 있습니다.



### MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledg (https://arxiv.org/abs/2502.19870)
- **What's New**: MMKE-Bench는 다중 모드 지식 편집의 새로운 벤치마크로, 실제 시나리오에서 다양한 시각적 지식을 수정하는 LMM의 능력을 평가하는 데 초점을 맞추고 있습니다. 기존 지식 편집 벤치마크들은 단순한 트리플 형태의 엔티티 수준 지식에 국한되어 있어 실제 다중 모드 정보의 복잡성을 포착하지 못했습니다. MMKE-Bench는 이 문제를 해결하기 위해 시각적 엔티티 편집, 시각적 의미 편집, 사용자 특정 편집의 세 가지 편집 작업을 포함하고 있습니다.

- **Technical Details**: MMKE-Bench는 자유 형식의 자연어로 지식을 표현하고 편집하는 방식을 채택하여 더욱 유연하고 효과적인 형식을 제공합니다. 총 2,940개의 지식 조각과 8,363개의 이미지를 포함하고 있으며, 33개의 다양한 카테고리에 걸쳐 평가 질문 및 답변이 자동 생성되고 인적 검증을 거쳤습니다. 평가 원칙으로는 신뢰성(reliability), 지역성(locality), 일반화(generalization), 이식성(portability)의 네 가지가 사용됩니다.

- **Performance Highlights**: 세 가지 대표 LMM에서 다섯 가지 최신 지식 편집 방법을 평가한 결과, 어떤 방법도 모든 평가 기준에서 두각을 나타내지 못했으며, 특히 시각적 지식과 사용자 특정 지식 편집이 어려운 것으로 나타났습니다. 현대 LMM들은 수정된 지식을 생산하고 적용하는 데 능숙하지만, MMKE-Bench는 이전 벤치마크보다 더 도전적인 성격을 가지고 있습니다. 이 연구는 다중 모드 지식 편집 기술의 강건성을 평가하는 새로운 기준을 설정하고 있습니다.



### MIND: Towards Immersive Psychological Healing with Multi-agent Inner Dialogu (https://arxiv.org/abs/2502.19860)
- **What's New**: 최근 정신 건강 이슈가 증가하는 가운데, 본 논문에서는 기존의 상담이나 챗봇과 같은 전통적인 치유 방법의 한계를 지적하고, MIND (Multi-agent INner Dialogue)라는 혁신적인 접근 방식을 제안합니다. 이 새로운 패러다임은 대화형 내부 대화를 통해 사용자가 더욱 몰입할 수 있는 심리적 치유 환경을 제공합니다. 또한, MIND는 각기 다른 역할을 수행하는 LLM (Large Language Model) 에이전트를 활용하여 사용자와의 상호작용을 극대화합니다.

- **Technical Details**: MIND의 전체 구조는 트리거(trigger), 악마(devil), 가이드(guide), 전략가(strategist) 네 가지 역할을 하는 에이전트로 구성됩니다. 각각의 에이전트는 사용자의 감정 상태에 따라 적절한 심리적 지원을 제공하며, 내적인 대화체계를 통해 정서적 조절을 돕습니다. 이 프레임워크에서는 각 에이전트가 사용자의 불안이나 우울감과 같은 인지 왜곡을 처리하고, 이를 통해 심리적 가이드를 제공합니다.

- **Performance Highlights**: 다양한 실험을 통해 MIND는 전통적인 심리 상담이나 챗봇, 기존 공감 훈련 방법과 비교하여 더 우수한 사용자 경험을 제공임을 입증하였습니다. 사용자는 내부 대화와 서로 다른 관점을 통한 메타인지적 성찰을 통해 깊은 자기 화해를 경험하게 됩니다. 이 연구 결과는 MIND가 LLM의 생성 능력을 활용하여 접근성과 효과성을 겸비한 정신 건강 지원을 제공할 수 있는 잠재력을 가지고 있음을 보여줍니다.



### Team A at SemEval-2025 Task 11: Breaking Language Barriers in Emotion Detection with Multilingual Models (https://arxiv.org/abs/2502.19856)
- **What's New**: 이번 논문은 SemEval 2025 Task 11에 제출된 Team A의 시스템에 대해 설명합니다. 이 작업은 텍스트 조각에서 화자의 감정을 식별하는 것이며, 각 사례는 여섯 가지 감정(joy, sadness, fear, anger, surprise, disgust) 중 하나로 주석이 달려 있습니다. 연구자들은 제공된 데이터셋을 활용하여 모델을 훈련하고 평가했습니다.

- **Technical Details**: 여러 접근 방식 중에서 가장 우수한 성능을 보인 것은 다국어 임베딩(multilingual embeddings)과 완전히 연결된 층(fully connected layer)을 결합한 방법이었습니다. 이 논문은 시스템 아키텍처에 대한 세부정보를 제공하며, 실험 결과를 논의하고 다국어 표현을 활용하여 텍스트의 감정 감지를 강화하는 장점을 강조합니다.

- **Performance Highlights**: 언어의 장벽을 허물고 다양한 감정을 효과적으로 감지하는 데 있어 다국어 임베딩의 효과를 입증하였습니다. 실험 결과, 다국어 표현을 활용함으로써 감정 인식의 전문성이 한층 향상되었습니다.



### Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation (https://arxiv.org/abs/2502.19830)
- **What's New**: 본 연구에서는 self-consistency를 새로운 관점에서 재정의하여 그 동적 배분 정렬 문제(dynamical distributional alignment problem)로 접근하였습니다. 기존의 고정된 true distribution 개념을 넘어, decoding temperature가 진정한 답 분포에 미치는 중요성을 조명합니다. 이 연구는 샘플 수에 제한 사항이 있을 때에서도 self-consistency의 효과를 극대화할 수 있는 가능성을 열어줍니다. 또한, 동적 온도 조정을 통한 신뢰 기반의 샘플링 방식이 제안되었습니다.

- **Technical Details**: self-consistency는 다양한 stochastic samples를 집계하여 모델 성능을 향상시키는 방법으로, 다수결 투표를 사용하는 방법론입니다. 이 연구에서 temperature 매개변수가 랜덤 샘플링과 진정한 답 분포를 조절하는데 중요한 역할을 함을 보였습니다. 다양한 실험을 통해 높은 온도가 더 많은 샘플을 필요로 하고, 낮은 온도가 모델 편향을 발생시킬 수 있음을 밝혔다. 이를 통해 온도를 동적으로 조정하면서 샘플링 배포의 신뢰도를 높이고, 새로운 분포를 탐색할 수 있도록 하는 기초를 마련하였습니다.

- **Performance Highlights**: 수학적 추론 작업에 대한 실험 결과, 제안된 신뢰 기반의 온도 조정 기법이 기존 고정된 다양성 기준보다 우수한 성능을 보였습니다. 제한된 샘플을 사용하더라도, 평균 및 최상의 성능이 향상되는 결과를 보여주었습니다. 추가적인 데이터나 모듈 없이도 초기 온도 변화에 따라 성능이 일관되게 개선되는 것을 확인했습니다. 이를 통해 self-consistency가 샘플링 동력학과 진화하는 답 분포 간의 동기화 문제로 작용할 수 있음을 입증하였습니다.



### Foot-In-The-Door: A Multi-turn Jailbreak for LLMs (https://arxiv.org/abs/2502.19820)
Comments:
          19 pages, 8 figures

- **What's New**: 이 논문은 실세계 응용 프로그램에서 대형 언어 모델(AI models)의 통합이 증가함에 따라 AI 안전을 보장하는 방법에 대해 다룹니다. 특히, 우리는 'jailbreak'라는 키 문제를 탐구하고 있으며, 이는 악의적인 프롬프트가 내장된 안전 장치를 우회하여 허용되지 않은 유해한 출력을 유도하는 것을 포함합니다. 새로운 다중 턴 다이내믹 메소드인 FITD를 소개하며, 이는 심리학의 'foot-in-the-door' 원칙에서 영감을 받았습니다.

- **Technical Details**: FITD 방법론은 초기 소액의 약속이 더 큰 약속이나 비윤리적 요청에 대한 저항을 낮추는 현상을 활용합니다. 이 접근 방식은 사용자 쿼리의 악의적인 의도를 점진적으로 증가시키며, 중간 다리를 통한 프롬프트를 사용하여 모델의 응답을 유도합니다. 실험은 두 개의 jailbreak 벤치마크에서 수행되었으며, 이는 seven 개의 널리 사용되는 모델에 걸쳐 평균 94%의 공격 성공률을 나타냅니다.

- **Performance Highlights**: FITD는 기존의 최첨단 방법보다 더 우수한 성능을 보였습니다. 또한, LLM(self-corruption) 자가 부패에 대한 심층 분석을 제공하여 현재 정렬 전략의 취약성을 강조했습니다. 이 논문은 다중 턴 프롬프트가 내재하고 있는 위험성에 대해서도 경고하며, 관련된 소스 코드는 공개되어 있습니다.



### NaijaNLP: A Survey of Nigerian Low-Resource Languages (https://arxiv.org/abs/2502.19784)
Comments:
          35 pages, 2 figures, 4 tables

- **What's New**: 나이지리아에는 500개 이상의 언어가 있지만, 하우사(Hausa), 요루바(Yorùbá), 이그보(Igbo) 세 언어가 1억 7500만 명 이상의 사람들에 의해 사용되며 60% 이상을 차지합니다. 이러한 언어는 계산 언어학(computational linguistics) 작업을 지원할 자원이 부족하여 저자원(low-resource) 언어로 분류됩니다. 본 연구는 이 세 가지 주요 나이지리아 언어에 대한 저자원 자연어 처리(NLP)의 발전을 포괄적으로 검토한 첫 번째 연구로, 언어 이해 및 생성과 같은 복잡한 작업을 지원하는 자원 부족과 이를 해결하기 위한 노력들을 검토합니다.

- **Technical Details**: 전 세계 언어의 90% 이상은 저자원(low-resource) 언어로 분류되며, 이들 언어는 부족한 평행 데이터(parallel source-target data)로 인해 통계적인 방법을 직접 적용하기 어렵습니다. 본 연구에서는 NaijaNLP(나이지리아의 세 주요 언어에 대한 자연어 처리) 연구의 현재 상태를 분석하고, 기존의 언어 자원, 도구 및 커뮤니티 지원을 평가하며, 이를 기반으로 향후 발전을 위한 전략을 제시합니다. 또한, 기존 연구에서 발견된 자원의 부족과 데이터의 변동성 문제점을 논의합니다.

- **Performance Highlights**: 하우사, 요루바, 이그보와 같은 저자원 언어에 대한 연구는 증가하고 있지만, 검토한 연구 중에서 오직 25.1%만이 새로운 언어 자원에 기여했습니다. 이는 기존 데이터를 재사용하는 경향이 강하며, 고유한 문제들이 여전히 연구되지 않고 있음을 보여줍니다. 따라서 자원 확충, 종합적인 주석 작업(annotation)의 필요성과 개방형 협업 이니셔티브 개발을 강조하며, 우리는 NaijaNLP와 저자원 NLP의 발전을 위해 더욱 광범위한 노력이 필요하다고 주장합니다.



### Do Retrieval-Augmented Language Models Adapt to Varying User Needs? (https://arxiv.org/abs/2502.19779)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Language Models (RALMs) 평가를 위한 새로운 프레임워크를 제안했습니다. 사용자 요구에 따라 모델의 응답 특성을 측정하는 데 초점을 맞추었으며, 세 가지 사용자 케이스(Context-Exclusive, Context-First, Memory-First)와 세 가지 컨텍스트 설정(Context Matching, Knowledge Conflict, Information Irrelevant)을 도입하여 실제 애플리케이션의 복잡성을 반영하고자 했습니다.

- **Technical Details**: 이 프레임워크는 사용자가 요구하는 정보의 유형에 따라 모델이 어떻게 반응하는지를 체계적으로 평가합니다. 사용자는 외부 정보와 내부 지식 중 어떤 것을 우선시할지를 지시할 수 있으며, 이러한 지시사항에 따라 모델의 성능이 어떻게 달라지는지 분석합니다. 우리의 실험은 URAQ 데이터셋을 포함하여 여러 QA 데이터셋에서 수행되었으며, 두 개의 모델 계열(Llama3.1, Qwen2.5)에 대해 다양한 모델 크기와 검색된 컨텍스트 수로 평가되었습니다.

- **Performance Highlights**: 주요 실험 결과에 따르면 현재의 언어 모델은 다양한 사용자 요구를 충족시키는 데 어려움을 겪고 있으며, 모든 데이터셋에서 50% 이하의 정확도를 기록했습니다. 또한, 컨텍스트 제약이 모델 성능에 미치는 영향을 확인했으며, 이상적인 검색 결과에서는 성능이 저하되는 경향이 있음을 발견했습니다. 모델 계열에 따라 성능 차이가 두드러지며, 특정 상황에서는 특정 모델이 다른 모델보다 우수한 성능을 발휘하는 것으로 나타났습니다.



### Advancements in Natural Language Processing for Automatic Text Summarization (https://arxiv.org/abs/2502.19773)
Comments:
          11 pages, 9 figures, ICCS 2024

- **What's New**: 이번 연구는 Automatic Text Summarization (ATS) 기법의 필요성이 더욱 강조되고 있음을 설명합니다. 특히, Natural Language Processing (NLP)와 Deep Learning (DL)의 발전으로 텍스트 요약 모델의 효과성이 크게 향상되었습니다. 그러나 다양한 텍스트의 복잡한 문체가 요약 과정에 여전히 많은 제약을 주고 있음을 지적합니다.

- **Technical Details**: 텍스트 요약 기법은 크게 두 가지 유형으로 나눌 수 있습니다: extractive summarization(추출적 요약)과 abstractive summarization(추상적 요약). 추출적 요약은 원본 텍스트에서 문장, 구문, 또는 텍스트 조각을 직접 추출하는 방식이며, 추상적 요약은 언어 분석을 통해 원본 텍스트의 내용을 재구성하는 방법입니다. 연구진은 기존의 혼합 기법들을 조사하고, 여러 접근 방식의 장단점을 분석하였습니다.

- **Performance Highlights**: 저자들은 다양한 요약 기법과 메트릭스를 비교 분석하여 생성된 요약을 평가하였습니다. 이 조사는 다양한 시스템과 아키텍처의 발전을 조망하며, 그들의 작동 방식에 대한 기술적 및 수학적 설명을 제공합니다. 이 연구는 ATS에 관한 포괄적인 개요를 제시하며, 앞으로의 연구 방향을 제시합니다.



### EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models (https://arxiv.org/abs/2502.19765)
- **What's New**: EdiText라는 새로운 텍스트 편집 방법이 제안되었습니다. 이 방법은 참조 텍스트를 다양한 속성으로 수정할 수 있게 해주는 SDEdit 기반의 편집 기술을 통합합니다. EdiText는 자가 조건화(self-conditioning) 기법을 기반으로 한 미세 편집 방법을 도입하여 기존 텍스트를 세밀하게 조정할 수 있는 기능을 제공합니다.

- **Technical Details**: EdiText는 임베딩 확산 모델(embedding diffusion model)을 사용하는 텍스트 편집 프레임워크로, 글로벌 수준과 미세 수준의 편집을 모두 지원합니다. Latent Diffusion for Language Generation (LD4LG) 모델을 기반으로 하여, 이 텍스트 편집 방법은 이산 데이터가 연속 데이터로 변환되고, 혼합 확산 프로세스를 통해 모델링됩니다. 이러한 접근 방식은 다양한 텍스트 속성을 편집할 수 있도록 해줍니다.

- **Performance Highlights**: 제안된 EdiText 방법은 여러 작업에서 탁월한 편집 성능을 보여주며, 기존 모델보다 더 넓고 미세한 범위에서 편집이 가능합니다. Coarse-level and fine-level 편집 기술을 통합함으로써, EdiText는 보다 포괄적이고 정밀한 편집 과정을 보장합니다. 이는 다양한 결과물을 요구하는 작업에서 더 나은 성능을 발휘하게 합니다.



### PolyPrompt: Automating Knowledge Extraction from Multilingual Language Models with Dynamic Prompt Generation (https://arxiv.org/abs/2502.19756)
Comments:
          6 pages, 2 figures

- **What's New**: 이 연구에서는 다국어 모델의 성능 편차를 줄이기 위해 PolyPrompt라는 새로운 자동 프롬프트 생성 프레임워크를 소개합니다. PolyPrompt는 입력 언어에 따라 트리거 토큰을 동적으로 학습하고 적용하여 모델의 다국어 작업 성능을 크게 향상시킵니다. 실험 결과, 이 방법은 일반적인 기반선과 비교하여 3.7%에서 19.9%의 정확도 향상을 나타내며, 다양한 언어에서 그 효과가 입증되었습니다.

- **Technical Details**: Tλ라는 언어별 트리거 임베딩을 학습하여, 각 언어에 대해 k 개의 학습 가능한 임베딩을 사용합니다. 입력 쿼리의 언어를 감지하고, 해당 언어의 트리거 임베딩을 쿼리에 추가하여 모델에 공급합니다. 이 과정에서 모든 모델 파라미터는 고정되어 있고, 트리거 임베딩만이 업데이트됩니다.

- **Performance Highlights**: PolyPrompt는 글로벌 MMLU 벤치마크에서 15개 다양한 언어로 평가되었으며, 그 결과 약 1억 개의 매개변수를 가진 모델에서 효과적임을 증명합니다. 이 연구는 언어 모델 성능의 단점을 완화하고, 다양한 언어에서 보다 높은 효율성을 보장하는 것을 목표로 합니다.



### Beneath the Surface: How Large Language Models Reflect Hidden Bias (https://arxiv.org/abs/2502.19749)
- **What's New**: 이번 연구에서는 Hidden Bias Benchmark (HBB)라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 사회적 편견이 은연중에 드러나는 형태로 자연적인 맥락에서 숨겨진 편견을 평가하는 데 초점을 맞추고 있습니다. LLM(대규모 언어 모델)의 성능과 함께 내재되어 있는 사회적 편견이 어떻게 여전히 존재하는지를 분석합니다.

- **Technical Details**: HBB는 다섯 가지 주요 사회 범주인 연령, 성별, 인종 민족, 사회경제적 계급 및 종교에 걸쳐 스크리닝된 4만 개 이상의 테스트 인스턴스를 포함합니다. HBB는 기존의 Overt Bias 평가 방식으로는 측정할 수 없는 편향의 패턴을 분석하여, 숨겨진 편향(Hidden Bias)이 어떤 식으로 모델에 나타나는지를 제공합니다. 이 연구의 실험 결과, 더 발전된 모델일수록 숨겨진 편향 점수가 높아지는 경향이 있음이 밝혀졌습니다.

- **Performance Highlights**: HBB를 사용하여 수행된 실험 결과, GPT-4o 모델이 기존의 평가 방식에서는 낮은 편향 점수를 기록했지만, HBB에서는 상당히 높은 편향 점수를 보여주는 것으로 나타났습니다. 이는 LLM이 은연중에 숨겨진 편향을 강화하고 있음을 시사합니다. 따라서 이 연구는 언어 모델의 공정성 평가 시 숨겨진 편향을 포착하는 것이 매우 중요함을 강조합니다.



### HaLoRA: Hardware-aware Low-Rank Adaptation for Large Language Models Based on Hybrid Compute-in-Memory Architectur (https://arxiv.org/abs/2502.19747)
Comments:
          7 pages

- **What's New**: LoRA(finetuning) 모델을 하이브리드 CIM(compute-in-memory) 아키텍처에 배치하는 새로운 전략을 제안합니다. 이 방법은 RRAM(resistive random-access memory) 상에 사전 훈련된 가중치를 배치하고, SRAM(static random-access memory) 상에는 LoRA를 배치하여 에너지 효율성을 극대화합니다. HaLoRA는 이러한 하드웨어의 비이상성을 해결하고 안정적인 성능을 유지하여, 노이즈를 가지고 있는 사전 훈련된 가중치 하에서도 정확한 출력을 보장합니다.

- **Technical Details**: 저자는 HaLoRA라는 새로운 하드웨어 인식 저순위 조정 메소드를 제안하여 RRAM에 의한 노이즈 문제를 해결합니다. 이 방법은 두 가지 LoRA 최적화 방향 간의 불일치를 최소화하여, 노이즈가 포함된 조건에서도 최적의 성능을 도출할 수 있도록 합니다. 최적화 과정에서 랜덤 노이즈를 주입하고 이를 기반으로 LoRA 브랜치를 최적화하여 특정 노이즈 패턴에 과적합되는 것을 피합니다.

- **Performance Highlights**: LLaMA 3.2 모델(1B 및 3B 변형)에 대한 실험 결과, HaLoRA는 다양한 추론 작업에서 표준 LoRA보다 일관되게 우수한 성능을 보였습니다. 특히, 0.02의 노이즈 레벨에서 HaLoRA는 평균 점수 63.1을 달성하여 LoRA보다 22.7점 높은 성과를 기록했습니다. 이 결과는 하이브리드 CIM 아키텍처에서 LLM을 배치하는 데 있어 HaLoRA의 유효성을 보여줍니다.



### XCOMPS: A Multilingual Benchmark of Conceptual Minimal Pairs (https://arxiv.org/abs/2502.19737)
- **What's New**: 이 논문에서는 17개 언어로 구성된 다국어 개념 최소 쌍 데이터셋인 XCOMPS를 소개합니다. 이 데이터셋을 사용하여 LLM(대형 언어 모델)의 다국어 개념 이해력을 평가하고, 이를 메타언어적 프롬프트, 직접 확률 측정, 신경언어학적 탐사를 통해 조사합니다. 연구 결과, LLM은 언어마다 정확도가 다르며, 낮은 자원 언어에서 개념 이해도가 떨어지는 경향이 발견되었습니다.

- **Technical Details**: XCOMPS는 COMPS의 다국어 확장으로 설계되었으며, 분석적, 굴절적, 아교적 언어를 포함하여 다양한 언어 구조를 대표합니다. 연구팀은 기본 모델, 지침 조정 모델 및 증류된 모델을 사용하여 개념 이해에 대한 다양한 세부 조정 전략의 영향을 분석하였습니다. 실험 결과, 구조적으로 복잡한 언어에서 개념 이해도가 낮아지고, 깊은 계층 구조가 필요하다는 사실을 발견했습니다.

- **Performance Highlights**: LLM의 성능은 고유한 개념 관계에서 높은 구별력을 보이지만, 미세한 의미적 차이를 공유하는 부정 쌍에 대해서는 성능이 크게 저하되었습니다. 지침 조정은 성능을 향상시키지만, 진정한 개념적 능력은 개선되지 않았습니다. 또한, 낮은 자원 언어에 대해 지식 증류는 내부 능력을 향상시킬 수 있으나, 명시적 작업 성과에는 제한적인 영향을 미쳤습니다.



### R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning (https://arxiv.org/abs/2502.19735)
- **What's New**: 이 논문은 인간 번역가의 구조화된 사고 체계인 chain-of-thoughts (CoTs)를 기계 번역(MT)에 통합하는 새로운 접근 방식을 제안합니다. 기존의 방법들은 특정 MT 하위 작업에 맞춘 고정된 CoT를 설계하거나 인간과 일치하지 않는 CoT를 합성하는 데 의존했습니다. 새로운 프레임워크인 R1-Translator (R1-T1)는 강화학습(RL)을 통해 일반 MT에서 추론 기반 reasoning을 달성하는 혁신적인 방법을 소개하고 있습니다.

- **Technical Details**: R1-Translator는 인간 정렬 CoTs로 구성된 여섯 가지 일반적인 패턴을 사용하여 MT의 추론 시간을 최적화합니다. 이 연구는 특정 MT 하위 작업에 국한되지 않고 여섯 개 언어 및 다양한 작업에 대해 reasoning 기반 번역을 확장합니다. 또한, 문맥 인식 패러프레이징(paraphrasing) 및 역번역(back translation)과 같은 혼합된 인간 전략을 미러링하는 여섯 개의 전문가 선별 CoT 템플릿을 공식화하여, RL을 통한 스스로 진화하는 CoT 발견과 반망각(adaptation) 기능을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 Flores-101 테스트 세트에서 21개 언어, 80개 번역 방향에서 안정적인 번역 성능 향상을 보여줍니다. 특히 훈련에서 보지 못한 15개 언어에서 그 성능이 두드러졌으며, 일반 다국어 능력이 평범한 SFT와 비교하여 유지되었습니다. 이러한 결과는 R1-Translator의 일반화 능력을 강조하며, 기계 번역 분야에서의 활용 가능성을 보여줍니다.



### Speculative Decoding and Beyond: An In-Depth Review of Techniques (https://arxiv.org/abs/2502.19732)
- **What's New**: 최근 대규모 자가 회귀 모델(autoregressive models)의 배포에서 나타나는 순차 종속성(sequential dependencies) 문제를 해결하기 위한 새로운 접근법이 제안되었습니다. 전통적인 최적화 방식인 가지치기(pruning)와 양자화(quantization)는 모델 품질을 저해할 수 있지만, 최근의 생성-정제(generation-refinement) 프레임워크는 이 문제를 효과적으로 완화할 수 있음을 보여줍니다. 본 논문은 다양한 자가 회귀 시퀀스 작업에서의 방법들을 종합적으로 분석하여 이들 프레임워크의 체계적인 분류를 제공합니다.

- **Technical Details**: 본 논문에서는 생성-정제 프레임워크를 두 가지 기본 단계로 나누어, 시퀀스 생성(sequence generation)과 시퀀스 정제(sequence refinement)를 다룹니다. 각 단계에서 다양한 방법들이 제안되며, 이는 전통적인 방식보다 더 효율적인 초안 토큰(draft tokens) 생성 방법에 초점을 맞춥니다. 이 프레임워크는 시스템 차원에서의 과제를 제시하여 메모리 사용 최적화와 배치 처리 최적화와 같은 혁신을 통해 자가 회귀 모델의 배포 전략을 개선합니다.

- **Performance Highlights**: 생성-정제 방법들은 특히 실시간 응용 프로그램에서 성능 향상을 꾀합니다. 예를 들어, Speculative Decoding(SD) 방법은 초기 초안 모델을 통해 병렬로 여러 토큰을 예측한 후, 대상 모델을 통해 검증하는 두 단계 과정을 도입하여 응답 지연을 최소화합니다. 이러한 접근은 텍스트, 이미지 및 음성 생성에 걸쳐 다양한 응용 프로그램에서 효율성을 입증하며, 자가 회귀 시퀀스 생성 연구에서 중요한 기초를 제공합니다.



### Preference Learning Unlocks LLMs' Psycho-Counseling Skills (https://arxiv.org/abs/2502.19731)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문은 정신 상담에 대하여 대형 언어 모델(LLMs)을 적용하는 새로운 접근 방식을 제안하고 있습니다. 환자의 요구와 정신 건강 지원의 불균형이 심각한 상황에서 LLM들이 클라이언트의 발언에 효과적으로 반응할 수 있도록 돕기 위한 원칙 세트를 개발했습니다. 이를 바탕으로 36,000개의 고품질 선호 비교 쌍을 포함하는 데이터셋인 PsychoCounsel-Preference를 만들어 LLM 훈련의 기반 자료로 활용합니다.

- **Technical Details**: PsychoCounsel-Preference 데이터셋은 26,483개의 클라이언트 발언으로 구성되어 있으며, 8개의 주요 주제와 42개의 세부 주제를 포함합니다. 이 데이터셋은 전문 심리 치료사들의 선호도와 일치하는 유용한 기준을 마련하여 생성되었습니다. 연구에서는 보상 모델(reward model) 및 선호 학습(preference learning)을 통해 LLM이 상담 세션에서 클라이언트에 반응하는 핵심 기술을 습득할 수 있음을 입증했습니다.

- **Performance Highlights**: PsychoCounsel-Llama3-8B 모델은 GPT-4o를 상대로 87%의 높은 승률을 기록하며, 클라이언트에 대한 반응에서 균형 잡히고 바람직한 응답을 제공할 수 있음을 보여줍니다. 이 결과는 온라인 및 오프라인 선호 학습 방법을 적용하여 도출되었으며, 모델 성능 향상에 대한 인사이트도 제공합니다. 또한, 이 연구의 결과물인 PsychoCounsel-Preference와 PsychoCounsel-Llama3-8B는 정신 상담 연구에 기여할 것으로 기대됩니다.



### CNsum:Automatic Summarization for Chinese News Tex (https://arxiv.org/abs/2502.19723)
Comments:
          WASA 2022

- **What's New**: 이번 연구는 방대한 데이터에서 유용한 정보를 효율적으로 추출하는 것을 목표로 하고 있으며, 특히 한중 뉴스 텍스트 요약 생성에 초점을 맞추고 있습니다. 최근 트렌드는 Transformer 구조의 프리트레인(Pre-trained) 언어 모델이 다양한 자연어 처리(Natural Language Processing, NLP) 작업에서 큰 성과를 거두었다는 점입니다. 본 논문에서는 Transformer 구조를 기반으로 한 중국 뉴스 텍스트 요약 모델(CNsum)을 제안합니다.

- **Technical Details**: CNsum 모델은 중국 데이터셋인 THUCNews에서 테스트되었습니다. Transformer 아키텍처를 활용하여 모델이 최적화 되었으며, 다양한 검증 방법으로 성능이 입증되었습니다. 이 모델은 중국어에 특화된 요약 기법을 사용하여 텍스트 요약의 품질을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, CNsum 모델은 기준 모델들에 비해 더 우수한 ROUGE 점수를 기록했습니다. 이는 제안된 모델이 기존의 베이스라인(Base-line) 모델들보다 실제로 더 뛰어난 성과를 보였음을 나타냅니다. 이러한 결과는 CNsum의 효과성을 부각시키며 향후 연구에 대한 기초 자료를 제공합니다.



### Few-Shot Multilingual Open-Domain QA from 5 Examples (https://arxiv.org/abs/2502.19722)
Comments:
          Accepted by TACL; pre-MIT Press publication version

- **What's New**: 최근 다국어 오픈 도메인 질문 응답(MLODQA) 방식은 풍부한 언어별 훈련 데이터를 통해 유망한 결과를 보여왔습니다. 그러나 비대표 언어에 대한 적용은 상당한 주석 비용(annotation cost)으로 제한됩니다. 본 논문에서는 대형 언어 모델(LLMs)을 활용하여 대규모 다국어 데이터를 합성하는 few-shot learning 접근법을 소개합니다.

- **Technical Details**: 본 방법은 WikiData를 이용한 대규모 자기 지도(pre-training) 학습을 시작으로, 몇 개의 샘플(few-shot supervision)을 사용하여 LLMs에서 생성된 고품질 합성 다국어 데이터에 대한 훈련이 이어집니다. 최종 모델인 FsModQA는 MLODQA 및 교차 언어(cross-lingual)와 단일 언어(mono-lingual) 검색에서 기존의 few-shot 및 감독(supervised) 기준을 크게 능가합니다.

- **Performance Highlights**: 또한 본 방법은 영어 감독 데이터를 통해 새로운 언어에 대한 효과적인 제로 샷(zero-shot) 적응을 위해 확장 가능함을 보여줍니다. 이는 비용이 많이 드는 대규모 주석 없이도 MLODQA 작업에 일반적이고 적용 가능한 솔루션으로서의 가능성을 의미합니다.



### Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs (https://arxiv.org/abs/2502.19721)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)에서 성별 개념을 표현하는 방법을 연구하기 위해 새로운 기술을 채택했습니다. 기존의 레이블 데이터 의존적인 방법 대신 레이블 없이 확률 가중치를 이용하여 개념 표현을 추출하는 방법을 제안합니다. 또한, 우리는 모델의 예측을 조작하기 위해 내재된 표현을 정확하게 제어하는 프로젝션 기반 방법을 제시합니다. 성별 편향을 완화하는 데 있어 이러한 새로운 접근 방식의 효과를 입증합니다.

- **Technical Details**: 여기서 제안하는 방법은 대형 언어 모델의 내부 표현에서 성별 개념이 어떻게 인코딩되어 있는지를 분석합니다. 활성화 조작(activation steering)이라는 개입 방법을 사용하여 모델의 출력에 영향을 주는 활성화를 고의적으로 변화시킵니다. 또한, 성별 신호를 조작하기 위한 후보 벡터를 효율적으로 선택하는 방법을 설명하고 있습니다. 이를 통해 추출된 벡터는 성별 편향과 높은 상관관계를 보인다는 점에서 기존의 방식보다 우수함을 보여줍니다.

- **Performance Highlights**: 본 연구의 결과는 제안된 조작 방법이 성별 편향을 줄이는 데에 효과적이라는 것을 확인하였습니다. 내부 표현의 조작을 통해 모델 예측을 통제할 수 있으며, 이는 특정한 작업에서의 효과를 보여주는 실험을 통해 입증되었습니다. 더 나아가, 성별 편향 완화 방법이 다른 응용 작업에 효과적으로 일반화될 가능성 또한 제시하고 있습니다.



### GRACE: A Granular Benchmark for Evaluating Model Calibration against Human Calibration (https://arxiv.org/abs/2502.19684)
- **What's New**: 이번 논문에서는 언어 모델의 보정(calibration) 문제를 다루기 위해 GRACE라는 새로운 벤치마크를 도입합니다. GRACE는 질문-답변 쌍으로 구성되어 있으며, 각 질문은 점진적으로 쉬워지는 단서를 포함하고 있어 모델이 가능한 한 빨리 정확한 답변을 하도록 유도합니다. 이 연구의 핵심은 인간의 보정과 직접 비교할 수 있는 기준을 제공하여 언어 모델의 성능을 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: GRACE는 잘 구성된 질문-답변 형식을 통해 언어 모델의 보정을 측정하는 방법을 제공합니다. 각 질문은 최소 5개의 단서로 구성되며, 이를 통해 모델의 답변 시간이 얼마나 빠른지를 세밀하게 평가할 수 있습니다. 본 연구에서는 인간-모델 간의 경쟁을 통해 수집된 데이터를 바탕으로 모델의 신뢰도와 정확성을 분석하는 새로운 메트릭, CalScore를 제안합니다.

- **Performance Highlights**: 사람들은 언어 모델보다 더 정확하지는 않지만 일반적으로 더 잘 보정됩니다. GRACE에서 언어 모델들은 부정확한 답변에서 과도하게 자신감을 나타내는 반면, 올바른 답변에서는 상대적으로 자신감이 낮은 경향을 보입니다. 이러한 결과는 최신 모델들이 GRACE에서 어려움을 겪고 있음을 보여주며, 이는 모델 보정의 개선을 위한 효과적인 평가 기준으로 기능합니다.



### Investigating Neurons and Heads in Transformer-based LLMs for Typographical Errors (https://arxiv.org/abs/2502.19669)
Comments:
          14 pages, 10 figures, 6 tables

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 타이포(Typo)를 포함한 입력을 어떻게 인코딩하는지를 연구합니다. 연구진은 특정 뉴런(neurons)과 주의 헤드(attention heads)가 타이포를 인식하고 로컬(local) 및 글로벌(global) 컨텍스트를 사용하여 이를 내부적으로 수정한다고 가정합니다. 실험 결과, LLMs는 오류가 있는 입력에서도 타이포를 수정할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 LLM의 내부 작동 방식을 이해하기 위해 타이포 뉴런과 타이포 헤드를 식별하는 방법을 제안합니다. 초기 및 중간 레이어의 타이포 뉴런은 로컬 컨텍스트에 기반한 수정에 기여하며, 중간 레이어의 타이포 뉴런은 글로벌 컨텍스트를 고려한 수정에 주요한 역할을 합니다. 타이포 헤드는 특정 토큰에 국한되지 않고 전체 컨텍스트를 폭넓게 고려하여 타이포를 수정합니다.

- **Performance Highlights**: 실험에서는 Gemma 2, Qwen 2.5, Llama 3 AI@Meta 모델을 이용하여 타이포가 포함된 입력에 대한 LLM의 내부 작동을 분석했습니다. 결과적으로 초기 또는 말기 레이어의 타이포 뉴런이 활성화되었을 때 LLM은 타이포를 수정할 수 있으며, 중간 레이어의 뉴런은 더욱 복잡한 글로벌 컨텍스트를 고려하여 타이포 수정에 기여합니다. 타이포 뉴런과 헤드는 타이포 수정 외에도 일반적인 문법적 또는 형태적 특징을 이해하는 데 도움을 줍니다.



### Med-RLVR: Emerging Medical Reasoning from a 3B base model via reinforcement Learning (https://arxiv.org/abs/2502.19655)
- **What's New**: 최근 강화학습에서 검증 가능한 보상을 바탕으로 한 연구(RLVR)가 주목받고 있습니다. 특히 DeepSeek-R1의 연구 결과는 기본 언어 모델에서 명시적인 추론 감독 없이도 스스로 발전된 추론 능력을 이끌어낼 수 있음을 보여주었습니다. 본 논문에서는 의료 분야에서의 RLVR의 적용 가능성을 탐구하며, Med-RLVR이라는 시스템을 도입했습니다.

- **Technical Details**: Med-RLVR은 의료 분야에서의 MCQA(multiple-choice question answering) 데이터를 활용하여 기본 모델에서 명시적인 추론 감독 없이 의료적 추론을 이끌어내기 위한 초기 탐색 작업입니다. 강화학습 알고리즘으로는 Proximal policy optimization (PPO)을 사용하며, 보상 모델은 검증 함수(verification function)로 설정되어 있습니다. 보상을 계산하는 방법에서는 출력의 형식이나 정답 여부에 따라서 보상을 부여하는 규칙 기반 기능을 활용합니다.

- **Performance Highlights**: Med-RLVR은 전통적인 감독 하의 미세 조정(Supervised Fine-Tuning, SFT)과 유사한 성능을 달성하면서도 분포 외 일반화(out-of-distribution generalization)에서 약 8%의 정확도 개선을 보였습니다. 학습 역학 분석을 통해, 3B 매개변수의 기본 모델에서 명시적인 감독 없이도 추론 능력이 생겨났음을 확인했습니다. 이러한 결과는 더욱 많은 분야에 걸쳐 RLVR의 가능성을 열어줍니다.



### Weaker LLMs' Opinions Also Matter: Mixture of Opinions Enhances LLM's Mathematical Reasoning (https://arxiv.org/abs/2502.19622)
Comments:
          12 pages, 1 figure, 3 tables, 4 prompt/data templates

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 수학에서의 형식적 추론 능력에 대한 관심이 높아졌습니다. 본 논문에서는 공정성 문제를 해결하기 위해 Mixture of Opinions (MoO)라는 새로운 접근 방식을 제안하여, 강력한 LLM이 약한 보조 LLM의 다양한 의견을 활용할 수 있도록 함으로써 추론 성능을 향상시키는 방법을 다룹니다. 이를 통해 결론적으로 다양한 관점이 추론 작업에서 긍정적인 영향을 미친다는 사실을 확인했습니다.

- **Technical Details**: MoO 프레임워크는 세 가지 단계로 구성되어 있습니다: (1) MoO 데이터셋 수집, (2) MoO 데이터셋으로의 사후 훈련, (3) 사후 훈련된 모델로의 추론. 특히, 주 모델(주 LLM)은 강력한 모델이며, 보조 LLM들은 비교적 약하지만, 유사한 추론 능력을 가지고 구체적인 답변을 생성할 수 있는 역할을 합니다. 이를 통해 다양한 의견을 통합하여 올바른 답변을 생성하는 능력을 강화합니다.

- **Performance Highlights**: 수행 성능을 평가하기 위해 수학적 추론 벤치마크인 GSM8K, AQuA-RAT, MATH를 사용했습니다. MoO 기법은 기존 방법들에 비해 평균적으로 5%의 성능 향상을 보여주었으며, 약한 LLM의 의견을 통합함으로써 추론 정확성이 유의미하게 개선될 수 있음을 입증했습니다. 이러한 결과는 여러 모델의 다양한 관점을 활용하면서 경량 모델에서도 결과적으로 성능 향상을 이룰 수 있음을 나타냅니다.



### Is Your Paper Being Reviewed by an LLM? A New Benchmark Dataset and Approach for Detecting AI Text in Peer Review (https://arxiv.org/abs/2502.19614)
- **What's New**: 이번 연구는 동료 심사(peer review) 과정의 무결성을 확보하기 위한 새로운 데이터 세트를 소개하고 있습니다. 이 세트는 AI가 생성한 788,984개의 동료 리뷰와 해당 인간 리뷰를 결합하여 제공합니다. 이는 ICLR와 NeurIPS 같은 두 주요 AI 연구 회의에 8년 간 제출된 논문의 리뷰를 포괄하고 있어, AI 기반 텍스트 검출 방법의 혁신적인 평가 자원을 제공합니다.

- **Technical Details**: 연구진은 18개의 기존 AI 텍스트 검출 알고리즘의 성능을 평가하여, LLM이 생성한 리뷰와 인간 작성 리뷰 간의 구분을 테스트하였습니다. 또한, 새로운 검출 방법을 제안하여 기존 방식보다 우수한 성능을 보였으며, 주로 의미적 유사성을 사용하여 기존 LLM 리뷰와 비교하는 방법을 사용하였습니다. 이를 통해 LLM이 생성한 리뷰의 위험성을 강조하며 이러한 비윤리적 사용에 대한 탐지 도구의 필요성을 제기하고 있습니다.

- **Performance Highlights**: 연구 결과, 기존의 대부분 AI 텍스트 검출 알고리즘은 AI가 생성한 리뷰를 신뢰성 있게 감지하는 데 한계가 있음을 보여주었습니다. 새로운 접근 방식은 기존 방식보다 성능이 뛰어나 GPT-4o 및 Claude로 작성된 리뷰의 탐지에 효과적이었습니다. 이 연구는 LLM이 작성한 텍스트 감지의 어려움을 시사하며, 이에 대한 추가적인 연구 필요성을 강조합니다.



### Evaluation of Hate Speech Detection Using Large Language Models and Geographical Contextualization (https://arxiv.org/abs/2502.19612)
Comments:
          6 pages, 2 figures

- **What's New**: 소셜 미디어에서의 혐오 발언의 확산은 사회에 큰 영향을 미치는 심각한 문제로 부각되고 있다. 이 연구는 다국어 데이터셋과 다양한 지리적 맥락에서 혐오 발언 탐지에 대한 LLM(대형 언어 모델)의 성능을 체계적으로 조사하였다. 본 연구에서는 혐오 발언의 이진 분류, 지리적 맥락 인식 탐지 및 적대적 생성 텍스트에 대한 강건성을 포함한 새로운 평가 프레임워크를 제시한다.

- **Technical Details**: 우리의 접근법은 LLM의 성능을 향상시키기 위한 프롬프트 엔지니어링을 사용하였다. 혐오 발언 탐지를 위해 구조화된 프롬프트를 설계하여 모델이 뉘앙스가 있는 텍스트를 이해할 수 있도록 하였다. 평가한 LLM으로는 Llama2, Codellama, DeepSeekCoder가 있으며, 각각의 모델은 1,000개의 다양한 지역에서 수집된 댓글로 평가되었다.

- **Performance Highlights**: Codellama는 혐오 발언 탐지에서 70.6%의 리콜과 52.18%의 F1 점수를 기록했으나, 지리적 민감성 테스트에서는 DeepSeekCoder가 더 나은 성능을 보였다. Llama2는 62.5%의 적대적 샘플을 잘못 분류하여 현재 LLM의 강건성의 한계를 나타내는 결과를 보였다. 이러한 발견은 정확성, 맥락적 이해 및 강건성 사이의 트레이드오프를 부각시킨다.



### Revisiting Word Embeddings in the LLM Era (https://arxiv.org/abs/2502.19607)
- **What's New**: 본 논문은 대형 언어 모델(Large Language Models, LLMs)이 생성한 임베딩(embeddings)의 성능 개선이 단순한 스케일(scale) 때문인지 아니면 전통적인 임베딩 모델들인 Word2Vec, GloVe 등의 임베딩과 유의미하게 다른지에 대해 검토합니다. 저자들은 고전적 비맥락적(Decontextualized) 및 맥락적(Contextualized) 단어 임베딩을 LLM 임베딩과 비교 분석하여, LLM이 의미적으로 관련된 단어들을 더 밀접하게 클러스터링하며 비맥락적 설정에서 비슷한 태스크에서 더 나은 성능을 보이는지를 입증하였습니다.

- **Technical Details**: 이 논문에서는 약 80,000개의 단어로 비맥락적 및 맥락적 설정에서의 단어 임베딩 유사성을 조사하였습니다. 비맥락적 설정에서는 Word2Vec과 GloVe를 포함하여 각각 ≈50K 및 ≈60K의 사전학습된 단어 임베딩을 생성하였고, 맥락적 설정에서는 동사, 명사 및 형용사를 여러 문장에서 사용하여 맥락을 제공했습니다. 두 가지 설정에서 다양한 유사성 분석 및 유추 작업을 통해 임베딩 차이를 평가하였습니다.

- **Performance Highlights**: LLMs는 비맥락적 설정에서 의미적으로 관련된 단어를 더 밀접하게 클러스터링하고 유추 태스크에서 더 나은 성능을 나타내지만, 맥락적 설정에서는 SimCSE와 같은 고전적 모델이 문장 수준 유사성 평가에서 LLM보다 우수한 성능을 보여주었습니다. 이는 LLM이 여전히 세부적인 의미 표현에서는 고전 모델들에 비해 덜 유용할 수 있음을 시사합니다.



### A City of Millions: Mapping Literary Social Networks At Sca (https://arxiv.org/abs/2502.19590)
- **What's New**: 이번 연구에서는 다국어 픽션과 비픽션 내러티브에서 추출한 70,509개의 고품질 사회 네트워크를 공개합니다. 이 데이터셋에는 1800년부터 1999년까지 작성된 약 30,000개의 텍스트에 대한 메타데이터도 포함되어 있으며, 이는 인류학 및 사회과학에서 역사적 사회 세계에 대한 새로운 자원을 제공합니다. 자동화된 사회 네트워크 추출 방법을 도입하여 일관성을 유지하면서도 대규모로 데이터를 수집하는 데 성공했습니다.

- **Technical Details**: 연구팀은 Project Gutenberg (PG) 코퍼스에서 텍스트를 가져와 JSON Schema를 기반으로 한 출력 방식으로 사회 네트워크를 생성했습니다. 특히, 구글의 Gemini 1.5 Flash 모델을 사용해 최대 1백만 토큰까지 처리할 수 있는 컨텍스트 길이를 확보했습니다. 이 모델은 구조화된 출력을 지원하여 각 텍스트에 대해 등장인물과 그 관계를 포함한 JSON 배열을 반환합니다.

- **Performance Highlights**: 결과적으로 총 72,875 권의 문헌 중 71,836개의 네트워크를 성공적으로 추출했습니다. 이 데이터셋은 문학 및 사회 가설을 평가할 수 있는 기회를 제공하며, 비픽션 네트워크가 픽션 네트워크보다 다양한 커뮤니티로 구성되어 있고, 클러스터링이 적다는 초기 분석 결과를 도출했습니다.



### NeoBERT: A Next-Generation BER (https://arxiv.org/abs/2502.19587)
Comments:
          19 pages, 5 figures, 9 tables. Submitted to TMLR

- **What's New**: NeoBERT는 현대 아키텍처와 데이터, 최적화된 사전 훈련 방법론을 통합하여 양방향 모델의 능력을 재정의한 차세대 인코더입니다. 이 모델은 기존의 BERT 및 RoBERTa와 같은 인코더들이 겪고 있는 발전의 정체를 극복하고, 더 나아가 Retrieval-augmented generation과 같은 다양한 다운스트림 NLP 작업에 필수적입니다. NeoBERT는 기존 모델에 손쉽게 통합할 수 있도록 설계되었으며, 4,096 토큰의 확장된 컨텍스트 길이를 활용합니다.

- **Technical Details**: NeoBERT는 자가 감독식(pre-training) 및 세미-감독식(fine-tuning) 학습을 포함한 두 단계의 훈련 프로세스를 통해 최대 컨텍스트 윈도우를 4,096으로 증가시킵니다. 이러한 훈련 방법론은 물론 250M 파라미터의 크기에 불과하지만, MTEB 벤치마크에서 성능을 뛰어넘었으며, GLUE에서의 실험적 검증도 이루어졌습니다. 이 모델은 대량의 토큰(2조 이상)을 학습하여 일반화 능력을 극대화하고 다운스트림 성능을 향상시킵니다.

- **Performance Highlights**: NeoBERT는 MTEB에서 BERT large, RoBERTa large, NomicBERT, ModernBERT를 능가하는 성능을 보여주었습니다. 특히, 4,096 토큰의 컨텍스트 길이를 활용함으로써 RoBERTa보다 8배 긴 시퀀스를 처리할 수 있어 뛰어난 인퍼런스 속도를 자랑합니다. 오픈 소스로 제공되는 NeoBERT는 코드, 데이터, 체크포인트를 공개하여 연구 및 실제 활용을 가속화하고 있습니다.



### Where Are We? Evaluating LLM Performance on African Languages (https://arxiv.org/abs/2502.19582)
- **What's New**: 이 논문은 아프리카의 언어 정책이 데이터 가용성과 모델 성능에 미치는 영향을 탐구하며, 다국적 데이터셋을 활용한 Sahara 벤치마크를 소개합니다. 이를 통해 기존 대규모 언어 모델(LLMs)의 성능을 체계적으로 평가하고, 연구의 필요성을 강조합니다. 특히 몇몇 언어들은 양호한 성과를 내지만, 여러 전통적인 언어들은 데이터 부족으로 인해 여전히 소외되고 있다는 점을 지적합니다.

- **Technical Details**: Sahara 벤치마크는 아프리카의 풍부한 언어 다양성을 포착한 규모가 크고 공공 접근 가능한 데이터셋을 기반으로 구축되었습니다. 이 연구에서는 LLM의 성능을 다양한 아프리카 언어에 대해 평가하면서, 기존 데이터셋의 분포와 모델 성능 간의 관계를 분석합니다. 우리의 분석은 정책적으로 유도된 데이터 불균형이 NLP 결과와 밀접하게 연결되어 있음을 보여줍니다.

- **Performance Highlights**: 대규모 언어 모델의 성능 평가 결과, 데이터 자원이 충분한 언어들은 상대적으로 높은 성과를 보이는 반면, 자원이 제한된 언어들은 성과가 저조하다는 것을 밝혔습니다. 이를 통해 아프리카 국가들이 더 포괄적이고 혁신적인 언어 정책을 도입해야 한다는 실질적인 권장사항을 제공합니다. 연구 결과는 아프리카 커뮤니티의 언어 기술 향상을 위한 기초적인 통찰을 제공합니다.



### Do Large Language Models Know How Much They Know? (https://arxiv.org/abs/2502.19573)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 특정 주제에 대한 자신의 지식의 범위를 인식하는 능력을 평가하는 새로운 벤치마크를 개발하였다. 연구 결과, 모든 테스트한 LLM은 충분한 규모로 주어졌을 때, 특정 주제에 대한 자신의 지식을 얼마나 알고 있는지를 이해할 수 있는 것으로 나타났다. 이는 LLM의 지식 인식이 일반적인 속성일 가능성을 제시하며, 더 깊은 메커니즘 이해를 위한 추가 연구가 필요하다는 점도 강조되었다.

- **Technical Details**: 연구에서는 LLM이 가상의 개별 다이어리 항목으로 미세 조정되며, 특정 개인의 모든 다이어리 항목을 연대순으로 회상하는 능력을 평가한다. 모델이 올바른 양의 정보를 기억할 수 있다면, 이는 모델이 해당 주제에 대한 자신의 지식 범위를 이해하고 있다는 것을 시사한다. 또한 모델의 기억 능력이 단기 및 장기 문서 모두에서 효과적임을 보여주어, 특정 아키텍처의 특성이 성능에 미치는 영향을 탐구하였다.

- **Performance Highlights**: 시험된 모든 LLM은 적절한 데이터 크기에 따라 주제에 대한 지식을 적절히 기억하는 능력을 보여주었다. 그러나 데이터 스케일이 부족할 경우, 일부 모델은 무작위로 다이어리 항목을 회상하며 과소 또는 과대 기억하는 경향을 보였다. 연구는 이 특성의 출현 차이에 기여하는 잠재적 요인들에 대해서도 논의하여 LLM의 내재적 메커니즘에 대한 이해를 넓힌다.



### Stay Focused: Problem Drift in Multi-Agent Deba (https://arxiv.org/abs/2502.19559)
Comments:
          34 pages, 21 figures, 4 tables, under review

- **What's New**: 이번 연구에서는 다수의 에이전트 논의(multi-agent debate)에서 기존 문제에서 벗어나는 현상인 문제 드리프트(problem drift)를 새롭게 정의하고, 이를 10가지 과제에서 정량화하였습니다. 이 방법을 통해 지식 및 추론 과제 해결 과정의 한계를 이해하는 중요한 단계를 제시합니다.

- **Technical Details**: 연구팀은 8명의 전문가와 함께 문제 드리프트의 원인을 알아보기 위한 인간 연구를 수행하였습니다. 전문가들은 진전 부족(35%), 저품질 피드백(26%), 명확성 부족(25%)을 문제 드리프트의 가장 흔한 원인으로 지적하였습니다. 이를 해결하기 위해 LLM(as-a-judge) 기반의 새로운 방법인 DRIFTJudge와 문제 드리프트를 완화하는 DRIFTPolicy를 제안합니다.

- **Performance Highlights**: DRIFTPolicy는 테스트 시간에 문제 드리프트 사례를 31% 감소시킬 수 있는 것으로 나타났습니다. 이러한 결과는 다수의 에이전트 논의의 효과성을 향상시키기 위한 향후 연구의 방향성을 제시합니다.



### Distill Not Only Data but Also Rewards: Can Smaller Language Models Surpass Larger Ones? (https://arxiv.org/abs/2502.19557)
Comments:
          14 pages, 7 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 지식 증류(Knowledge Distillation) 과정에서 응답과 보상 신호를 동시에 전달하는 새로운 방법론을 제안합니다. 기존의 감독식 미세 조정(Supervised Fine-Tuning, SFT) 방식은 주로 응답을 통한 데이터를 증류하는 데 중점을 두었지만, 이 연구는 생성을 넘어 보상의 질을 반영할 수 있는 가능성도 탐색합니다. 새로운 파이프라인을 통해, 보상을 자기 감독(self-supervised) 기법으로 생성하여 외부 평가에 대한 의존도를 줄입니다.

- **Technical Details**: 제안된 방법은 LLM 모델의 응답 데이터 안에서 내재된 구조를 활용하여 '유사 보상(pseudo-rewards)'을 생성합니다. 이 과정은 교사 모델(Teacher Model)과 학생 모델(Student Model) 간의 응답의 품질을 비교하여 이루어지며, 이로 인해 보상 모델은 고품질 출력을 우선시하는 방법을 학습하게 됩니다. 이 모델은 초기 SFT 단계 이후 강화 학습(Reinforcement Learning, RL)을 통해 학생 모델의 성능을 지속적으로 개선합니다.

- **Performance Highlights**: GSM8K 및 MMLU-PRO 데이터셋에서 실시된 실험 결과, 제안된 방법이 전통적인 SFT 기반 방식을 일관되게 초월하는 것으로 나타났습니다. 학생 모델은 특정 상황에서 교사 모델보다 우수한 성능을 보였으며, 이는 효율적이고 확장 가능한 증류 과정을 통해 실현되었습니다. 전체적으로, 이 연구는 보상 학습을 통한 지속적인 모델 개선이 가능함을 보여줍니다.



### When Large Language Models Meet Speech: A Survey on Integration Approaches (https://arxiv.org/abs/2502.19548)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 이 모델을 텍스트 기반 작업을 넘어 다른 양상으로 확장하려는 관심을 불러일으켰습니다. 본 논문은 LLM과 음성( speech) 통합에 대해 연구한 여러 접근 방식을 세 가지 주요 방법론으로 분류하여 이해를 도모합니다. 이를 통해 음성과 LLMs 간의 통합 방식에 대한 체계적인 개요를 제공합니다.

- **Technical Details**: LLMs와 음성의 통합 방식은 크게 세 가지로 나뉩니다: 텍스트 기반 통합( text-based integration), 잠재 표현 기반 통합( latent-representation-based integration), 그리고 오디오 토큰 기반 통합( audio-token-based integration)입니다. 텍스트 기반 통합은 LLM이 텍스트 데이터를 처리하는 방식으로, 음성을 텍스트로 변환하거나 그 반대를 수행합니다. 잠재 표현 기반 통합은 음성 데이터를 인코딩한 잠재 벡터 표현을 LLM의 입력으로 사용하는 방식을 말하며, 오디오 토큰 기반 통합은 음성 토큰을 LLM의 입력이나 출력으로 사용하는 방법입니다.

- **Performance Highlights**: 이 연구는 음성과 LLM 간의 통합 방식의 중요성을 부각시키며, 다양한 음성 관련 응용 프로그램에서의 활용 가능성을 탐구합니다. LLM과 음성 통합의 잠재력을 통해 음성 번역, 대화형 챗봇 및 인간-컴퓨터 상호작용을 개선할 수 있는 방법들이 제시됩니다. 그러나 이 분야에서 존재하는 주요 도전 과제도 언급되며, 이러한 문제를 해결하기 위한 향후 연구 방향에 대한 영감을 제공합니다.



### Winning Big with Small Models: Knowledge Distillation vs. Self-Training for Reducing Hallucination in QA Agents (https://arxiv.org/abs/2502.19545)
- **What's New**: 이 연구에서는 고객 지원에 대한 대규모 언어 모델(LLMs)의 배치에서 발생하는 hallucination(허위 정보의 생성) 문제와 비경제적인 독점 모델의 비용 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, Samsung Smart TV 사용자 매뉴얼에 대한 질의 응답 데이터 셋을 사용하여 합성 데이터(synthetic data)가 군중 소싱 데이터(crowdsourced data)보다 더 낮은 hallucination 비율을 보여줄 수 있음을 입증했습니다. 또한, self-training(자기 훈련)과 지식 증류(knowledge distillation)를 비교하면서, 두 방법이 유사한 수준의 hallucination 감소를 보임을 발견했습니다.

- **Technical Details**: 연구에서는 retrieval-augmented question-answering (QA) 파이프라인을 개발하고, Llama-3-8B-Instruct 모델을 사용하여 crowdsourced 질문에 대한 답변을 생성합니다. 이와 함께, 수작업(cleaning) 및 자동화된 방법을 통해 데이터 정제(data cleaning) 성능을 비교했습니다. 결과적으로 LLM이 생성한 합성 데이터는 더 낮은 hallucination 비율을 기록했으며, self-training 방식이 knowledge distillation 보다 유사한 효과를 나타내는 흥미로운 발견을 하였습니다. 또한, 무응답 질문에 대한 "모르겠습니다"라는 맥락화된 응답(contextualized responses)을 통해 모델의 견고성(robustness)을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 수작업 및 자동 데이터 정제 방법은 유사한 사실적 정확도를 보였지만, 자동 정제를 통한 모델의 응답이 더 길었습니다. LLM이 생성한 합성 교육 데이터는 군중 소싱 데이터보다 낮은 hallucination 비율을 기록하였고, self-training을 통해 Llama-3 모델이 생성한 데이터에 대한 모델 성능이 GPT-4o의 데이터에 대한 훈련과 유사함을 입증했습니다. 이러한 발견은 self-training이 hallucination을 최소화하는 데 있어 리소스를 효율적으로 사용할 수 있는 대안임을 보여줍니다.



### Cognitive networks highlight differences and similarities in the STEM mindsets of human and LLM-simulated trainees, experts and academics (https://arxiv.org/abs/2502.19529)
Comments:
          Keywords: cognitive network science; mindset measurement; associative knowledge; artificial intelligence; simulated participants

- **What's New**: 이 연구는 STEM(과학, 기술, 공학, 수학)에 대한 태도를 정량화하여 인간과 GPT-3.5와 같은 대형 언어 모델이 이러한 주제를 어떻게 개념화하는지를 조사합니다. 177명의 인간 참가자와 177명의 인공 인간의 행동 심리적 형태 네트워크(Behavioral Forma Mentis Networks, BFMNs)를 사용하여, 그들의 마인드셋 차이를 비교했으며, 이는 전문성 수준이 마인드셋에 미치는 영향을 알리는 데 기여합니다.

- **Technical Details**: 참가자들은 교육생, 전문가, 학자로 나뉘어 마인드셋을 분석했습니다. 연구의 결과는 인간의 형태 네트워크가 GPT-3.5보다 훨씬 높은 군집 계수를 보였으며, 이는 인간의 마인드셋이 STEM 아이디어에 대한 개념 연관성을 형성하고 닫는 경향이 있음을 나타냅니다. 특히, 인간 전문가들은 STEM 개념의 인지 네트워크 통합이 더 뛰어남을 보여주었습니다.

- **Performance Highlights**: 이 연구에서 밝혀진 바와 같이, 인간과 GPT의 마인드셋 모두 수학을 중립적이거나 긍정적으로 프레이밍했으나, STEM 고등학교 학생 및 다른 많은 대형 언어 모델과는 차별화된 모습을 보였습니다. STEM 아이디어에 대한 접근 방식의 차이는 기억 구조와 머신의 한계를 이해하는 데 도움을 주는 통찰을 제공한다고 할 수 있습니다.



### Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis (https://arxiv.org/abs/2502.20383)
Comments:
          Project website: this http URL

- **What's New**: 최근 웹 AI 에이전트의 발전은 복잡한 웹 탐색 과제를 해결하는 데 놀라운 능력을 보여주었습니다. 그러나 이러한 에이전트들은 독립형 대형 언어 모델(LLMs)보다 더 큰 취약성을 나타내고 있으며, 이는 웹 AI 에이전트의 높은 유연성과 관련이 있습니다. 이 연구는 웹 AI 에이전트의 취약성을 증가시키는 여러 요인을 조사하고, 시스템 설계에서의 추가적인 안전성을 높일 수 있는 실질적인 통찰을 제공합니다.

- **Technical Details**: 웹 AI 에이전트는 LLM과 소프트웨어 도구, API를 통합하여 웹 환경 내에서 특정 목표를 달성하기 위한 일련의 작업을 수행합니다. 이 연구는 웹 AI 에이전트가 독립형 LLM과 비교하여 46.6%의 악의적인 명령을 실행할 가능성을 보인다는 사실을 보여줍니다. 웹 AI 에이전트의 높은 취약성은 사용자 목표를 시스템 프롬프트에 직접 삽입하고, 다단계 행동 생성을 하며, 관찰 능력을 강화하는 등 세 가지 주요 요소에서 기인합니다.

- **Performance Highlights**: 이 연구는 웹 AI 에이전트의 jailbreaking에 대한 높은 취약성을 수치적으로 비교합니다. 우리는 5단계의 세분화된 평가 메트릭을 도입하여 기존의 이진 평가 방식을 넘어서 웹 AI 에이전트의 취약성을 보다 심도 있게 분석합니다. 이 연구 결과를 통해 웹 AI 에이전트의 보안 위험을 완화하기 위한 방안과 설계 개선을 위한 권장 사항을 제시합니다.



### Multi-Turn Code Generation Through Single-Step Rewards (https://arxiv.org/abs/2502.20380)
Comments:
          9 pages (not including references or appendix); 6 figures (in main paper); (v1) preprint

- **What's New**: 이번 논문은 실행 피드백을 기반으로 한 다단계 코드 생성 문제를 다룹니다. 기존 방법론은 피드백 없이 코드를 생성하거나 복잡한 강화 학습(reinforcement learning)을 사용합니다. 우리는 단일 단계 보상만을 활용하여 문제를 해결하는 간단하면서도 확장 가능한 방법인 μCode를 제안합니다. 이 접근법은 코드 생성 프로세스를 효율적이고 안정적으로 만듭니다.

- **Technical Details**: μCode는 다단계 실행 피드백을 통해 코드 생성기를 훈련시키는 새로운 프레임워크입니다. 이 방법은 마르코프 결정 과정(Markov Decision Process, MDP)의 개념을 사용하여 각 상호작용에서 나온 중간 상태에서 올바른 코드를 단일 단계로 회복할 수 있음을 보여줍니다. 훈련 과정에서 생성기와 검증기를 동시에 개선하는 전문가 반복(expert iteration) 프레임워크를 사용합니다. 또한, 실시시간(inference time) 확장을 위해 학습된 검증기를 이용해 코드를 선택합니다.

- **Performance Highlights**: 실험 결과, 우리의 μCode 방식은 MBPP(Austin et al., 2021)와 HumanEval(Chen et al., 2021) 벤치마크에서 가장 유력한 다단계 접근법들을 초월한 성능 향상을 보였습니다. 학습된 검증기를 활용하여 더 나은 생성기 학습이 이루어짐을 증명하였고, 높은 추론 예산이 있는 경우에도 유망한 확장법 트렌드를 보여주었습니다.



### PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation (https://arxiv.org/abs/2502.20377)
- **What's New**: PhantomWiki는 고유하고 사실적으로 일관된 문서 집합체를 생성하는 파이프라인으로, 기존 데이터셋의 한계를 극복하고자 제안되었습니다. 매번 평가 시에 새로운 PhantomWiki 인스턴스를 생성함으로써 데이터 유출 및 성과 부풀림 문제를 피할 수 있습니다. 이 방식은 LLM(대형 언어 모델)의 추론(Reasoning)과 검색(Retrieval) 능력을 분리하여 평가할 수 있는 새로운 기준을 제공합니다.

- **Technical Details**: PhantomWiki는 질문 난이도와 코퍼스 크기를 조정하여 LLM의 추론과 검색 능력을 체계적으로 분석할 수 있게 설계되었습니다. 이는 특정 코퍼스의 맥락 창 안에 필요한 정보를 적재하고, 검색 방법을 통해 외부 정보를 접근해야 하는 복잡한 상황을 포함합니다. 이로써, LLM의 내부 지식 의존성을 평가하고, 다양한 기법을 사용할 수 있는 가능성을 모색합니다.

- **Performance Highlights**: PhantomWiki에서의 평가는 상태 최상의 LLM이 직면하는 도전을 잘 보여줍니다. 다양한 문서 코퍼스에 대한 질문을 처리할 때, F1 점수는 논리적이거나 기술적인 복잡성이 증가할수록 급락하는 경향을 보였습니다. PhantomWiki는 연구 커뮤니티가 LLM의 성능을 평가하고 개선할 수 있는 견고한 기준을 제공하며, 이와 관련된 코드가 추후 공개될 예정입니다.



### Towards Responsible AI in Education: Hybrid Recommendation System for K-12 Students Case Study (https://arxiv.org/abs/2502.20354)
- **What's New**: 이 연구에서는 K-12 학생들을 위한 그래프 기반 추천 시스템을 소개합니다. 이 시스템은 학생의 필요에 맞는 과외 활동, 학습 자원 및 자원봉사 기회를 개인화하여 제공합니다. 특히, 공정성을 확보하기 위한 프레임워크를 통합하여 보호된 학생 집단에 대한 피드백을 분석하고 편향을 줄이는 방법을 다룹니다.

- **Technical Details**: 제안된 추천 시스템은 그래프 기반 모델링과 행렬 인수 분해(matrix factorization)를 결합하여 학생의 표현된 및 추론된 관심사에 따라 추천을 맞춤화합니다. 시스템의 그래프 구조는 두 부분으로 나뉘며, 정적인 부분은 관심사와 자원 간의 관계를 정의하고, 동적인 부분은 학생과 그들의 관심사 간의 연결을 나타냅니다. 이 과정은 피드백을 활용하여 향후 추천을 개선하는 '하이브리드' 모델을 사용합니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 추천의 정확성과 공정성 모두에서 효과를 보였습니다. 추천의 결과는 특정 학생 집단 간의 불균형을 점검하고 교정하는 데 필수적인 역할을 합니다. 해당 연구는 교육 추천 시스템에서 공정성과 투명성을 지속적으로 모니터링할 필요성을 강조하고, 모든 학생에게 평등한 학습 기회를 제공하는 데 기여합니다.



### M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging (https://arxiv.org/abs/2502.20301)
Comments:
          38 pages, 7 figures

- **What's New**: 본 논문에서는 의료 이미징 분야에서의 머신 러닝(ML) 자동화를 위한 새로운 다중 에이전트 시스템인 M3Builder를 소개합니다. M3Builder는 복잡한 다단계 워크플로우를 관리하는 네 개의 전문화된 에이전트가 협력하여 데이터 처리, 환경 구성, 자동 디버깅 및 모델 훈련을 수행합니다. 이 시스템은 의학적 이미징 ML 작업을 위한 통합 환경인 Medical Imaging ML workspace와 함께 작동하여, AI 도구의 자율적 개발을 가능하게 합니다.

- **Technical Details**: M3Builder는 의료 이미징 분석을 위한 문제를 정의하고 이를 자동으로 해결하기 위해 다중 에이전트 협업 프레임워크를 적용합니다. 해당 프레임워크는 기계 학습(ML) 워크스페이스와 다중 에이전트 시스템의 두 가지 주요 구성 요소로 나뉘며, 여기에는 자연어로 설명된 데이터 카드, 툴셋 설명, 코드 템플릿이 포함됩니다. 이러한 요소들은 에이전트들이 상호작용하며 작업을 이행할 수 있도록 지원하는 구조적 환경을 제공합니다.

- **Performance Highlights**: M3Builder는 Claude-3.7-Sonnet을 에이전트의 핵심으로 사용하여 94.29%의 성공률을 기록하며, 기존의 ML 에이전트 디자인 대비 우수한 성능을 보여줍니다. 실험 결과는 장기 데이터 분할에서 86.67%, 이상 탐지에서 100%, 질병 진단에서 95%, 보고서 생성에서 93.33%의 성과를 나타냅니다. 이 결과는 의료 이미징에서의 완전한 자동화된 머신 러닝의 가능성을 시사합니다.



### An exploration of features to improve the generalisability of fake news detection models (https://arxiv.org/abs/2502.20299)
Comments:
          Accepted at Expert Systems with Applications (Elsevier)

- **What's New**: 이 논문은 가짜 뉴스 탐지의 일반화 가능성을 높이기 위한 연구로, 기존의 불완전한 라벨링 데이터에 대한 문제를 다룹니다. 연구에서는 TF-IDF 및 BERT와 같은 토큰 기반 모델이 편향된 데이터에 민감하다는 점을 시사하며, 스타일적 특징(lexical, syntactic, semantic)과 사회적 모니타이제이션(social-monetisation) 특징의 중요성을 강조합니다. 이외에도 그 동안 제한적으로 활용된 대규모 언어 모델(LLMs)의 적합성에 대한 평가도 진행합니다.

- **Technical Details**: 논문에서는 NELA 2020-21 데이터 세트를 사용하여 훈련하고, 수동 라벨링 된 Facebook URLs 데이터 세트를 이용해 일반화 가능성을 평가합니다. 연구는 스타일적 특징 및 사회적 모니타이제이션 특징이 토큰 기반 방법보다 더 일반화 가능한 예측을 제공한다는 주장을 통해 성능을 분석합니다. 더불어, 통계적 및 순열 특징 중요성 분석을 통해 데이터 세트 편향을 완화하고 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과 토큰 기반 모델은 편향된 데이터에서 훈련될 경우 30% 정도의 정확도 저하를 겪는 반면, 스타일적 및 사회적 모니타이제이션 특징은 일반화 가능성을 증대시키며 더 나은 성과를 보여줍니다. 또한, LLaMa와 같은 대규모 언어 모델들이 가짜 뉴스 탐지에 효과적이지 않다는 제한적인 증거를 제시합니다. 결과적으로, 스타일적 특징과 경제적 동기를 이해하는 것이 가짜 뉴스 탐지의 발전에 기여할 수 있음을 강조합니다.



### Granite Embedding Models (https://arxiv.org/abs/2502.20204)
- **What's New**: Granite Embedding 모델은 검색 작업을 위해 특별히 설계된 인코더 기반의 임베딩 모델로, 밀집(mDense) 및 희소(sSparse) 검색 아키텍처를 포함하여 영어와 다국어 기능을 제공합니다. 이 보고서는 12층 모델과 6층 경량 모델의 훈련 기술적 세부사항을 제공합니다. 벤치마크 평가 결과, 이 모델들은 IBM 내부 검색 작업에서 공개된 유사한 크기의 모델을 상당히 초과하며, 정보 검색 기준에서도 동등한 성능을 발휘합니다.

- **Technical Details**: Granite Embedding 모델들은 다섯 가지 인코더 전용 모델로 구성되어 있으며, 각각 다른 크기와 어휘를 보유하고 있습니다. 이 모델들은 고품질 데이터로 훈련되어, 개인 정보와 저속 언어를 제거하는 품질 검사를 통과했습니다. Dense English 모델은 RoBERTa 아키텍처를, Multilingual 모델은 XLM-RoBERTa 아키텍처를 따릅니다.

- **Performance Highlights**: Dense Granite 임베딩 모델은 질의와 관련된 문서의 임베딩을 서로 가까이 위치시키고 비관련 문서의 임배딩은 멀어지도록 하는 대조 학습 목표를 사용하여 훈련됩니다. 30M 및 125M 파라미터 모델은 영어 검색 응용 프로그램에 최적화되어 있으며, 107M 및 278M 파라미터의 다국어 모델은 12개 언어를 다룹니다. 경량 6층 모델은 매우 낮은 지연 시간과 메모리 요건으로 그 성능을 제공합니다.



### An Extensive Evaluation of PDDL Capabilities in off-the-shelf LLMs (https://arxiv.org/abs/2502.20175)
Comments:
          Under review

- **What's New**: 최근의 발전으로 인해 대형 언어 모델(LLMs)은 코드 생성 및 사고의 연쇄(chain-of-thought reasoning)에 뛰어난 능력을 보여주며, 자동 형식 계획(auto formal planning) 작업을 해결하는 데 기초를 마련하고 있습니다. 본 연구에서는 인공지능 계획에서 중요한 표현인 계획 도메인 정의 언어(Planning Domain Definition Language, PDDL)를 이해하고 생성하는 LLM의 잠재력을 평가합니다.

- **Technical Details**: 우리는 상업적 및 오픈 소스 포함하여 7개 주요 LLM 패밀리에서 20개의 서로 다른 모델에 대한 광범위한 분석을 수행합니다. 이 포괄적인 평가를 통해 LLM의 제로샷(zero-shot) 기능에 대한 파싱(parsing), 생성(generating) 및 PDDL을 통한 추론(reasoning) 성능을 조명합니다.

- **Performance Highlights**: 일부 모델은 PDDL 처리에서 주목할 만한 효과를 보여주는 반면, 다른 모델은 미세한 계획 지식이 필요한 복잡한 시나리오에서 제한된 성능을 보입니다. 이러한 결과는 공식 계획 작업에서 LLM의 가능성과 현재 한계를 강조하며, AI 기반 계획 패러다임에 대한 향후 연구 방향을 제시합니다.



### Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think (https://arxiv.org/abs/2502.20172)
Comments:
          13 pages, 9 figures, codebase in this https URL

- **What's New**: 최근 텍스트-이미지 생성에 관한 연구에서 Dream Engine이라는 새로운 통합 프레임워크가 제안되었습니다. 이 프레임워크는 강력한 텍스트 인코더를 활용하여 텍스트와 이미지를 효과적으로 결합하여 복잡한 생성 명령을 처리할 수 있도록 설계되었습니다. 기존의 제어 방법이 고급 텍스트-이미지 상호작용을 지원하지 못했던 문제를 해결하려고 하며, 대규모 다중 모달 모델(LMM)의 잠재력을 활용하고 있습니다.

- **Technical Details**: Dream Engine은 두 단계의 훈련 패러다임을 적용하여 텍스트와 이미지의 정렬을 최적화합니다. LMM과 텍스트-이미지 확산 모델을 연결하기 위해 경량 프로젝터 레이어를 사용하여 입력되는 텍스트와 이미지 간의 효과적인 매핑을 지원합니다. 또한, 객체 감지와 이미지 캡셔닝을 결합한 객체 주도 생성 기능을 도입했습니다.

- **Performance Highlights**: 실험 결과, Dream Engine은 GenEval 벤치마크에서 0.69의 점수를 기록하며 SD3.5 및 FLUX와 유사한 성능을 보였습니다. 압축 데이터를 통해 훈련했음에도 불구하고, 이 모델은 상호작용하는 텍스트 및 이미지 지침을 효과적으로 처리하며, 다양한 입력 이미지에서 개념을 합성하는 능력을 보였습니다.



### Re-evaluating Open-ended Evaluation of Large Language Models (https://arxiv.org/abs/2502.20170)
Comments:
          Published at ICLR 2025

- **What's New**: 이 논문은 Large Language Models (LLMs)의 평가 방식을 개선하기 위해 전통적인 Elo 기반 시스템의 문제점을 지적하고, 데이터의 편향(bias) 및 중복(redundancy)에 민감한 특성으로 인해 발생할 수 있는 부작용을 설명합니다. 저자들은 이를 해결하기 위해 3인 게임(gamer-theoretic)의 형태로 평가를 제안하고, Robust한 평가 방법을 위한 새로운 게임 이론적(solution concept) 개념을 소개합니다. 이를 통해 LLM 개발의 경쟁적인 장면을 이해하고 더 직관적인 점수가 이루어질 수 있도록 합니다.

- **Technical Details**: 저자는 게임 이론(games theory)을 적용하여 LLM 평가를 재구성하기 위해 여러 기여를 합니다. 이들은 N-player 일반합계(general-sum) 게임의 고유하고 클론 불변(clone-invariant)한 균형 솔루션 개념을 도출하고, 실제 LLM 평가 데이터셋에 대한 적용 가능성을 보여줍니다. 또한, 협동적인 데이터 품질 문제를 해결하는 한편, 평가 시스템 설계자들이 목표를 명확하게 표현할 수 있도록 합니다.

- **Performance Highlights**: LMSYS Chatbot Arena와 같은 시스템에서 Elo 점수를 지속적으로 상승시키는 모델들이 과연 기술적 발전을 의미하는지에 대한 의문을 제기합니다. 저자들은 시뮬레이션을 통해 모델이 특정 기술에 특화될 위험을 실증적으로 보여 주며, 평가 시스템의 변별력을 향상시키고 균형 잡힌 점수를 보장하기 위한 방향성을 제공합니다. 이러한 접근은 LLM 개발 과정에서의 주요 이슈인 편향 및 데이터 품질 문제를 근본적으로 해결하는 데 기여할 것입니다.



### Similarity-Distance-Magnitude Universal Verification (https://arxiv.org/abs/2502.20167)
Comments:
          35 pages (8 Tables, 4 Algorithms, 5 Listings)

- **What's New**: 본 논문에서는 신경망의 Robustness 문제를 해결하기 위해 Softmax 함수의 출력 Magnitude(결정 경계) 인식에 Similarity(유사성)와 Distance(거리) 인식을 추가한 새로운 sdmsdm 활성화 함수를 제안합니다. 이 새로운 활성화 함수는 상대적인 epistemic (감소 가능한) 예측 불확실성의 강력한 신호를 제공합니다. 불확실성 추정치는 학습된 변환을 통해 얻으며, 이는 모델의 예측을 해석 가능하게 하여 HCI 문제도 해결할 수 있습니다.

- **Technical Details**: 기존 LLM(대규모 언어 모델)의 한계는 예측의 불확실성을 정량화하는데 있어 제약이 많았습니다. sdmsdm 활성화 함수는 training set에 대한 정확한 예측 적합도를 통해 epistemic 불확실성을 분해하는 새로운 접근 방식을 제공합니다. 이러한 방법론은 모델의 결정 경계에 대한 거리를 고려하며, 인간이 이해 가능하도록 모델의 불확실성을 탐색할 수 있는 능력을 부여합니다.

- **Performance Highlights**: sdmsdm 활성화 함수는 테스트 시간의 분포 변화와 분포 외 입력에 대해 놀라운 강건성을 보이며, 효과적인 샘플 크기를 인식합니다. 불확실성 추정치는 다른 접근 방식과 비교할 때 훨씬 더 신뢰할 수 있는 결과를 보여줍니다. 또한, 새로운 LLM 아키텍처인 sdmsdm 네트워크는 불확실성 인식 및 사례 기반 해석 가능성을 본질적인 특성으로 갖추고 있습니다.



### Telephone Surveys Meet Conversational AI: Evaluating a LLM-Based Telephone Survey System at Sca (https://arxiv.org/abs/2502.20140)
- **What's New**: 이번 연구는 AI를 기반으로 한 전화 조사 시스템을 소개하며, 이는 text-to-speech (TTS), 대규모 언어 모델 (LLM), 음성 인식 (STT)을 통합하여 인간 인터뷰어의 다양성을 모방합니다. 두 가지 모집단, 즉 미국의 파일럿 연구와 페루의 대규모 배포에서 테스트를 진행하였으며, 이를 통해 AI 에이전트가 개방형 및 폐쇄형 질문을 잘 관리하고, 기본적인 설명을 해주며, 분기 논리를 동적으로 탐색할 수 있음을 보여주었습니다. 이 연구는 LLM 기반의 전화 인터뷰어가 실제 조사 환경에서 성공적으로 대규모로 배포된 첫 사례 중 하나입니다.

- **Technical Details**: AI 조사 시스템은 STT, LLM, TTS의 세 가지 핵심 구성 요소로 구성되어 있으며, 각 구성 요소는 전화 조사 수행에서 인간과 유사한 역할을 하도록 설계되었습니다. STT 모듈은 참가자의 음성 응답을 텍스트로 변환하고, LLM은 대화 이력을 기반으로 적절한 응답을 생성하며, TTS 엔진은 LLM의 텍스트 출력을 자연스러운 음성으로 변환합니다. 모든 모델은 내부적으로 정밀 조정된 버전을 사용하여 높은 전사 정확도와 맥락에 적합한 대화 생성을 보장합니다.

- **Performance Highlights**: AI 시스템의 질적 깊이를 요구하는 probing은 인간 인터뷰어보다 더 제한적이지만, 전체 데이터 품질은 구조화된 항목에 대해 인간이 주도하는 수준에 접근했습니다. 미국과 페루의 다양한 샘플에서 실시된 연구를 통해, AI 전화 조사 시스템은 시장 조사, 사회 과학 및 여론 조사에서 널리 활용될 수 있으며, 그에 따라 운영 효율성을 향상시키면서도 적절한 데이터 품질을 유지하는 가능성을 보여주었습니다.



### SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning (https://arxiv.org/abs/2502.20127)
- **What's New**: 이번 논문은 Subtask-oriented Reinforced Fine-Tuning (SoRFT)을 제안하여 대형 언어 모델(LLMs)의 문제 해결 능력을 향상시킵니다. 기존의 상업적 모델에 의존하던 문제 해결 프레임워크의 단점을 극복하기 위해 SoRFT는 문제 해결을 파일 로컬라이제이션(file localization), 함수 로컬라이제이션(function localization), 줄 로컬라이제이션(line localization), 코드 수정 생성(code edit generation)과 같은 구조화된 하위 작업으로 나누어 수행합니다. 이를 통해 개방형 소스 개발 리소스를 최대한 활용하는 접근 방식을 제시합니다.

- **Technical Details**: SoRFT는 두 단계의 훈련으로 구성됩니다: (1) 거부 샘플링된 감독 Fine-Tuning(Supervised Fine-Tuning, SFT)과 (2) 규칙 기반 강화 학습(rule-based reinforcement learning)입니다. SFT 단계에서 교사 LLM을 사용해 하위 작업을 위한 Chain of Thought (CoT) 데이터를 생성하고, 그라운드 트루스 기반에서 부정 샘플을 필터링합니다. RL 단계에서는 각 하위 작업에 대한 그라운드 트루스를 활용하여, PPO(Proximal Policy Optimization) 알고리즘을 통해 훈련을 진행합니다.

- **Performance Highlights**: SoRFT로 훈련된 모델은 SWE-Bench Verified와 SWE-Bench Lite에서 최고의 성능을 달성했습니다. 특히, SoRFT-Qwen-7B 모델은 SWE-Bench Verified에서 21.4%의 문제를 해결하는 데 성공하여 오픈 소스 모델 중 최상위 성적을 기록했습니다. 실험 결과는 SoRFT가 문제 해결 성능을 유의미하게 향상시키고, 모델의 일반화 능력을 개선하며, 상업적 모델에 대한 비용 효율적인 대안을 제공함을 보여줍니다.



### Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScor (https://arxiv.org/abs/2502.20034)
Comments:
          4 pages

- **What's New**: 최근 대형 비전-언어 모델(Large Vision-Language Models, LVLMs)의 성능이 여러 분야에서 두각을 나타내고 있습니다. 그러나 이러한 모델은 객체 환각(object hallucination) 문제에 직면하고 있습니다. 본 연구는 LVLM의 환각 원인이 비전 인코더(vision encoder)의 한정된 표현 능력에 있지 않다는 것을 규명하고, 새로운 평가 지표인 세분화된 CLIPScore(Fine-grained CLIPScore, F-CLIPScore)를 제안합니다.

- **Technical Details**: F-CLIPScore는 명사구 수준에서 텍스트 임베딩을 포함하여 객체 수준의 세부 사항을 향상시킵니다. 이 지표는 CLIP 모델의 시각-언어(Vision-Language) 파라미터들과 함께 사용되어 비전 인코더의 표현 능력을 효율적으로 평가합니다. 연구 결과, F-CLIPScore는 기존의 CLIPScore보다 39.6% 더 높은 정확도로 OHD-Caps 벤치마크에서 성능을 발휘하며, 추가적인 훈련 없이도 객체 환각을 효과적으로 감지할 수 있음을 보여줍니다.

- **Performance Highlights**: F-CLIPScore를 사용하여 사전 훈련 데이터 선별을 수행할 경우 모델의 객체 환각이 줄어드는 것을 확인했습니다. F-CLIPScore를 활용한 LVLM 사전 훈련 과정에서는 데이터 선별을 통해 기준 대비 POPE 정확도가 4.9% 향상되었습니다. 이러한 연구 결과는 비전 인코더의 제한된 능력이 객체 환각의 주요 원인이 아님을 암시하며, 더 효율적인 훈련 방식을 제시합니다.



### Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models (https://arxiv.org/abs/2502.19883)
Comments:
          12 pages. 6 figures

- **What's New**: 이번 연구에서는 작은 언어 모델(SLMs)의 보안 성능을 평가하기 위한 포괄적인 실험을 수행했습니다. SLMs는 효율적이며 낮은 컴퓨팅 비용으로 주목받고 있지만, 보안 위험이 대형 언어 모델(LLMs) 대비 상대적으로 덜 주목받아왔습니다. 실험 결과, 대부분의 SLM들이 기존의 jailbreak 공격에 상당히 취약하다는 점이 드러났습니다.

- **Technical Details**: 본 연구에서는 총 16개의 최첨단 모델을 분석했으며, 그 중 13개 SLM은 4억 이하의 파라미터를 가지고 있고, 3개의 LLM은 70억 이상의 파라미터를 가집니다. 실험을 통해 얻은 데이터는 다양한 jailbreak 공격 방법에 대한 SLM의 보안 취약점을 노출하고, 해당 공격에 대한 방어 방법의 효과를 평가했습니다. 특히, SLM의 보안 저하의 원인으로는 안전 정렬 부족, 편향된 지식 증류, 파라미터 공유 및 양자화 기법을 논의했습니다.

- **Performance Highlights**: 연구의 결과, 많은 SLM들이 LLMs보다 jailbreak 공격에 더욱 취약하다는 것이 밝혀졌습니다. 또한, 기존 방어 방법들이 SLM의 내성을 강화하는 데 유의미한 효과가 있음을 입증했습니다. 앞으로 SLM의 보안 챌린지를 강조하고, 더욱 견고하고 안전한 SLM 개발을 위한 귀중한 통찰력을 제공하는 것을 목표로 하고 있습니다.



### ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments (https://arxiv.org/abs/2502.19852)
Comments:
          ICLR 2025

- **What's New**: 본 논문은 기존 코드 생성 벤치마크의 한계를 극복하기 위해 새로운 벤치마크 세트를 제안합니다. 특히, 다단계 상호작용(multi-turn interactions)에서 LLM의 성능을 평가하는 데 필요한 다양성을 반영하도록 설계되었습니다. CONVCODEWORLD라는 새로운 환경을 도입하여 9개의 상호작용 코드 생성 시나리오를 시스템적으로 구현하였습니다.

- **Technical Details**: CONVCODEWORLD는 세 가지 유형의 피드백을 조합하여 코드 생성의 질을 평가하는 독특한 환경을 제공합니다: (a) 컴파일 피드백(compilation feedback); (b) 부분적 및 전체 테스트 커버리지(execution feedback); (c) 다양한 수준의 전문성을 가진 GPT-4o에 의해 생성된 구두 피드백(verbal feedback). 또한, CONVCODEBENCH는 사전 생성된 피드백 로그를 사용하여 비용을 절감하고, 생동감 있는 성과와 강한 상관관계를 유지합니다.

- **Performance Highlights**: 광범위한 평가를 통해 확인된 주요 인사이트는 다음과 같습니다: (a) 제공된 피드백에 따라 LLM의 성과가 크게 달라집니다; (b) 충분한 피드백을 받은 약한 LLM이 피드백 없는 최신 모델보다 성능이 더 좋을 수 있습니다; (c) 특정 피드백 조합에 대한 학습이 LLM의 새로운 조합 활용 능력을 제한할 수 있습니다. 이러한 통찰들은 LLM의 성능 평가에서 다단계 상호작용의 중요성을 강조합니다.



### Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training (https://arxiv.org/abs/2502.19726)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에 대한 멤버십 정보 추론 공격(Membership Inference Attacks, MIAs) 방어 메커니즘인 DuoLearn을 제안합니다. 최신 연구 결과에 따르면, 훈련 중 선택된 특정 토큰 집합을 사용하는 것이 모든 토큰을 사용하는 것보다 성능이 좋을 수 있다는 것을 발견했습니다. 이 연구는 토큰의 동적 선택 전략을 이용하여 MIAs에 대한 방어를 강화하고, 모델의 유틸리티를 유지하면서 계산 비용을 최소화하는 것을 목표로 합니다.

- **Technical Details**: DuoLearn은 훈련 중 토큰을 하드 토큰과 메모리화 토큰으로 분류하는 전략을 사용합니다. 하드 토큰은 높은 손실을 가진 토큰을 의미하며, 메모리화 토큰은 MIAs 신호가 강한 토큰을 가리킵니다. 이 논문에서는 하드 토큰에 대해 학습하고, 메모리화 토큰에 대해 비학습(unlearning)을 수행하는 이중 목적 손실 함수를 설계하여 모델이 유용한 정보를 학습하되 특정 훈련 샘플을 기억하지 않도록 합니다.

- **Performance Highlights**: 제안된 방어 메커니즘은 MIAs에 대해 강력한 보호를 제공하며, 다양한 LLM 아키텍처 및 데이터셋에서 언어 모델링 성능을 약 10% 향상시킵니다. 실험 결과, DuoLearn은 최소한의 성능 저하로도 언어 모델링에서의 메모리 문제를 효과적으로 완화하고, 학습 데이터를 보호하는 데 성공하였습니다.



### The Future Outcome Reasoning and Confidence Assessment Benchmark (https://arxiv.org/abs/2502.19676)
- **What's New**: 이번 논문에서는 FOReCAst(Future Outcome Reasoning and Confidence Assessment)를 도입하여 예측 모델의 성능과 신뢰도를 평가합니다. 기존의 예측 벤치마크들이 종합적인 신뢰성 평가를 결여한 채 인공적인 질문에 중점을 두었다는 문제를 해결하고자 합니다. FOReCAst는 Boolean 질문, 시간대 예측, 수량 추정 등 다양한 예측 시나리오를 포함하여 현실 세계의 요구에 부합하는 평가를 제공합니다.

- **Technical Details**: FOReCAst는 세 가지 유형의 예측 질문으로 구분됩니다: (1) Boolean 질문에서는 이벤트 발생 여부에 대한 예측을 만들고, (2) 시간대 예측에서는 특정 날짜를 제공하며, (3) 수량 추정에서는 미래 사건에 대한 수치적 예측을 수행합니다. 각 질문 유형에 맞춰 모델이 예측한 내용과 신뢰도 점수를 산출하여 종합 평가를 진행합니다.

- **Performance Highlights**: 실험 결과, 현재의 대규모 언어 모델(LLMs)들은 여전히 예측 과제에 도전적이며 특히 신뢰도 평가에서 성능 향상이 나타나지 않았습니다. 예측 성능과 신뢰도 보정 사이의 직접적인 상관관계는 발견되지 않았고, 대형 모델이 간혹 성능을 개선하는 경향이 있는 반면, 효과는 일관되지 않았습니다. 따라서, FOReCAst는 더 나은 예측 품질과 신뢰도 평가를 위한 방향성을 제시하고 있습니다.



### SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning (https://arxiv.org/abs/2502.19668)
- **What's New**: 본 논문에서는 심혈관 질환 진단을 위한 Electrocardiogram (ECG) 해석의 새로운 접근 방식을 제안합니다. 기존의 ECG Self-Supervised Learning (eSSL) 방법의 한계를 극복하고자 SuPreME이라는 감독형 사전 훈련 프레임워크를 도입하여, 미리 정리된 임상 정보를 활용한 고품질의 정밀한 레이블 데이터셋을 생성합니다.

- **Technical Details**: SuPreME은 Large Language Models (LLMs)를 활용하여 자유 텍스트 ECG 보고서에서 구조화된 임상 엔티티를 추출하고, 노이즈와 불필요한 내용을 필터링합니다. 이 프레임워크는 구조화된 엔티티 레이블과 ECG 신호를 직접 정렬하여, 기존의 복잡한 사전 작업을 피하고 데이터 증강에 대한 의존성을 줄입니다.

- **Performance Highlights**: SuPreME는 127개의 심장 질환을 포함한 6개의 후속 데이터 세트에서 평가되었으며, 제로 샷(zero-shot) 분류에서 1.96% 이상의 성능 개선을 달성했습니다. 이 모델은 전체 fine-tuning 없이도 높은 데이터 효율성을 바탕으로 학습되며, 효과적으로 높은 품질의 ECG 표현을 생성함을 입증하였습니다.



### Repurposing the scientific literature with vision-language models (https://arxiv.org/abs/2502.19546)
- **What's New**: 이 논문에서는 AI를 활용한 과학 출판의 가능성을 탐구합니다. 기존의 인공지능 기술들이 과학적 프로세스를 보완하는 데 집중된 반면, 본 연구는 특정 분야의 저널과 생성적 AI 모델을 결합하여 혁신적인 과학 커뮤니케이션 도구를 개발하는 것을 목표로 합니다. 연구팀은 Neurosurgery Publications에서 23,000개의 기사를 수집하여 NeuroPubs라는 멀티모달 데이터베이스를 구축했습니다.

- **Technical Details**: NeuroPubs는 1억 3,400만 단어와 78,000개의 이미지-캡션 쌍으로 구성된 데이터베이스로, 신경외과에 특화된 임상 맥락을 독창적으로 대표합니다. 생성적 AI 모델을 활용하여 서로 다른 일반 VLM(vision-language models)을 통한 그래픽 초록을 자동으로 생성하였고, 편집 위원들이 70%를 수정 없이 출판할 준비가 되었다고 평가했습니다.또한 ABNS(American Board of Neurological Surgery) 시험 스타일의 89,587개의 테스트 문제를 생성하였고, 이는 훈련생과 교수들이 54% 확률로 진품과 동일하게 인식하였습니다.

- **Performance Highlights**: CNS-Obsidian이라는 34억 매개변수 VLM을 훈련하여 교육 과정과 함께 지식 습득을 추적했습니다. 무작위 통제 시험에서 CNS-Obsidian의 성능이 GPT-4o와 비교했을 때 비열등함(p=0.1154)이 입증되었습니다. 이는 최신 생성 인공지능을 활용하여 과학 커뮤니케이션의 질적 기준을 유지하는 새로운 기반을 마련했음을 나타냅니다.



### Conversational Planning for Personal Plans (https://arxiv.org/abs/2502.19500)
- **What's New**: 이 연구에서는 대화 시스템이 장기 상호작용과 과제를 지원하기 위해 언어 기반 에이전트를 필요로 한다고 강조합니다. 대화식 계획을 통해 사용자의 목표에 맞는 미시 행동을 결정하는 메타 컨트롤러 역할을 수행하는 LLM(대규모 언어 모델)의 새로운 아키텍처를 제안하였습니다. 이를 통해 사용자의 피드백을 바탕으로 계획을 조정하며, 실제 목표를 달성하는 데 도움을 줄 수 있는 가능성을 열어줍니다.

- **Technical Details**: 제안된 접근 방식은 코어 비공식 계획(Chain-of-Thought, CoT) 프롬프트를 활용한 LLM이 어떻게 상위 행동(macros) 결정을 내리고, 여러 세부 단계에 따라 대화하며 상호 작용을 수행하는지를 설명합니다. 이 시스템은 사용자의 언어 피드백을 수집하고, 이를 기반으로 다음 행동을 결정하는 구조를 갖추고 있습니다. 이는 Hierarchical RL (강화 학습) 프레임워크에 기반하여 작업을 처리합니다.

- **Performance Highlights**: 이 접근 방식은 건강 관리 및 학습 등 다양한 분야에서 효과적으로 기능하는 것을 입증하였습니다. 사용자 피드백에 따라 계획을 지속적으로 수정하며, 개인 맞춤형 계획 수립에 도움을 줄 수 있는 가능성을 보여줍니다. 또한, 이 연구는 기존 대화형 추천 시스템의 상태를 향상시키는 작업에 기여합니다.



New uploads on arXiv(cs.IR)

### Granite Embedding Models (https://arxiv.org/abs/2502.20204)
- **What's New**: Granite Embedding 모델은 검색 작업을 위해 특별히 설계된 인코더 기반의 임베딩 모델로, 밀집(mDense) 및 희소(sSparse) 검색 아키텍처를 포함하여 영어와 다국어 기능을 제공합니다. 이 보고서는 12층 모델과 6층 경량 모델의 훈련 기술적 세부사항을 제공합니다. 벤치마크 평가 결과, 이 모델들은 IBM 내부 검색 작업에서 공개된 유사한 크기의 모델을 상당히 초과하며, 정보 검색 기준에서도 동등한 성능을 발휘합니다.

- **Technical Details**: Granite Embedding 모델들은 다섯 가지 인코더 전용 모델로 구성되어 있으며, 각각 다른 크기와 어휘를 보유하고 있습니다. 이 모델들은 고품질 데이터로 훈련되어, 개인 정보와 저속 언어를 제거하는 품질 검사를 통과했습니다. Dense English 모델은 RoBERTa 아키텍처를, Multilingual 모델은 XLM-RoBERTa 아키텍처를 따릅니다.

- **Performance Highlights**: Dense Granite 임베딩 모델은 질의와 관련된 문서의 임베딩을 서로 가까이 위치시키고 비관련 문서의 임배딩은 멀어지도록 하는 대조 학습 목표를 사용하여 훈련됩니다. 30M 및 125M 파라미터 모델은 영어 검색 응용 프로그램에 최적화되어 있으며, 107M 및 278M 파라미터의 다국어 모델은 12개 언어를 다룹니다. 경량 6층 모델은 매우 낮은 지연 시간과 메모리 요건으로 그 성능을 제공합니다.



### Bisecting K-Means in RAG for Enhancing Question-Answering Tasks Performance in Telecommunications (https://arxiv.org/abs/2502.20188)
Comments:
          6 pages, 8 figures, accepted at GLOBECOM WORKSHOPS 2024

- **What's New**: 이번 연구는 통신 분야를 위해 특별히 설계된 Retrieval-Augmented Generation (RAG) 프레임워크를 소개합니다. 이 프레임워크는 3GPP 문서로 구성된 데이터셋을 중심으로 하며, Bisecting K-Means 클러스터링 기법을 이용해 임베딩 벡터를 내용별로 조직합니다. 이를 통해 사용자 쿼리와 가장 유사한 클러스터를 미리 선택하여 정보 검색의 효율을 높입니다. 실험 결과로는, Small Language Models를 활용하여 기존 모델보다 향상된 성능과 함께 유의미한 정확도 향상이 확인되었습니다.

- **Technical Details**: 본 연구는 TeleQnA 데이터셋을 활용하여 통신 도메인에서 질문에 답변하는 LLM 모델을 특화하는 여러 단계의 방법론을 구현하였습니다. RAG 파이프라인은 3GPP 문서에서 데이터를 추출하여 약어 목록을 생성하는 것으로 시작됩니다. 사용자 쿼리를 개선하기 위한 과정이 뒤따르고, 문서는 청크(Chunk)로 변환된 후 Bisecting K-Means 로 분류되어 벡터 데이터베이스에 저장됩니다. 사용자가 입력한 쿼리는 강화된 쿼리(EQ)로 변환되어 클러스터와 비교되며, 가장 관련성이 높은 청크가 검색됩니다.

- **Performance Highlights**: 실험을 통해 ‘phi-2’와 ‘phi-3’ 모델에서 각각 66.12%와 72.13%의 정확도로 성능 향상이 이루어졌습니다. 또한, 트레이닝 시간이 단축되어 낮은 계산 비용으로 추론할 수 있음을 입증하였습니다. 이 프레임워크는 특히 통신 분야의 특수 용어와 약어가 많은 데이터 처리에서 효과성을 높이는 데 중점을 두었습니다. 필요한 정보를 더욱 신속하고 정확하게 제공함으로써 통신 분야의 비즈니스 운영 효율성을 개선하는 데 기여할 것으로 기대됩니다.



### Teaching Dense Retrieval Models to Specialize with Listwise Distillation and LLM Data Augmentation (https://arxiv.org/abs/2502.19712)
- **What's New**: 이 논문에서는 현재의 최신 dense retrieval 모델이 특정 도메인 지식을 충분히 포착하지 못할 수 있음을 강조합니다. 특히, 표준 fine-tuning 방식인 InfoNCE 손실을 사용할 경우 효과성이 악화될 수 있음을 보여줍니다. 이를 해결하기 위해 listwise distillation과 대체된 synthetic query 생성을 포함한 새로운 훈련 전략을 제안합니다.

- **Technical Details**: 논문에서는 BERT-base 모델을 기반으로 하여 dense retrieval에서의 효과성을 개선하기 위한 실험을 수행했습니다. 특히, hard-negative mining 및 negative de-noising과 같은 일반적으로 사용되는 기법들이 특정 도메인에 대한 fine-tuning에도 불구하고 예상보다 효과성을 떨어뜨릴 수 있다는 점을 발견했습니다. 또한, LLM을 활용한 다양한 쿼리 생성을 통해 여러 데이터셋에서 일관된 효과성 향상을 달성했습니다.

- **Performance Highlights**: 최신 연구에 따르면, synthetic queries는 인간이 작성한 쿼리와 경쟁할 만한 효과성을 보여줍니다. 다양한 유형의 쿼리를 포함한 훈련을 통해, 이전 방법들보다 BEIR 데이터셋에 대한 성능이 개선되었습니다. 그러나 크로스 인코더 교사 모델의 강도에서는 병목 현상이 발생할 수 있는 한계를 찾았습니다.



### PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems (https://arxiv.org/abs/2502.19628)
Comments:
          5 pages. Accepted by www'25 as short paper

- **What's New**: 이 논문에서는 대규모 전자상거래 플랫폼에서 사용자 경험을 최적화하기 위한 사용자 모델링 접근법을 제안합니다. 전통적인 모델들은 특정 비즈니스 메트릭에 중점을 두어 사용자 행동을 포괄적으로 반영하지 못했습니다. 새로운 프레임워크인 PCL(Prompt-based Continual Learning)을 통해 멀티 태스크 학습(Multi-task Learning)의 한계를 극복하고, 지속적으로 새로운 작업을 학습하는 데 필요한 지식을 보존할 수 있는 방안을 모색하고 있습니다.

- **Technical Details**: PCL 프레임워크는 포지션-와이즈 프롬프트(position-wise prompts)를 외부 메모리로 활용하여 각각의 작업에서 지식을 보존하고, 재 학습 과정에서 발생하는 재앙적 망각(catasrophic forgetting) 문제를 완화합니다. 또한, 작업 간의 관계를 반영하기 위해 맥락적 프롬프트(contextual prompts)를 설계하여 프롬프트 조정 과정에서 상호 작업 관계를 활용합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해 PCL의 효과를 입증하였고, 다양한 작업 순서와 콜드 스타트 설정에서도 그 강인성을 입증하였습니다. PCL은 추천 시스템 디자인에 있어 사용자 모델링의 지속적 학습을 위한 대안으로 기능할 것을 기대합니다.



### Mixture of Structural-and-Textual Retrieval over Text-rich Graph Knowledge Bases (https://arxiv.org/abs/2502.20317)
- **What's New**: 이번 연구에서는 Text-rich Graph Knowledge Bases (TG-KBs)에서 텍스트와 구조적 지식을 함께 효과적으로 검색하기 위한 Mixture of Structural-and-Textual Retrieval (MoR) 방법을 제안합니다. 기존의 검색 방법들은 일반적으로 이러한 지식을 분리하여 검색하는 경향이 있으며, 구조적 검색을 완전히 우회하는 경우도 많습니다. MoR은 플래닝-추론-조직화(Planning-Reasoning-Organizing) 프레임워크를 통해 두 종류의 지식을 통합적으로 검색하고, 이를 통해 서로의 이점을 강화합니다.

- **Technical Details**: MoR은 세 가지 주요 단계로 구성됩니다. 첫 번째 단계에서 MoR은 쿼리에 대한 계획 그래프를 생성하여 텍스트 계획을 수립합니다. 두 번째 단계에서는 구조적 탐색과 텍스트 매칭을 결합하여 TG-KB에서 후보를 얻습니다. 마지막으로, 조직 단계에서는 구조적 경로를 기반으로 가져온 후보를 재정렬하는 구조 인식 재정렬기(Structure-aware Rerank)를 적용합니다.

- **Performance Highlights**: MoR은 기존의 검색 방법과 비교했을 때 구조적 및 텍스트적 검색의 조화를 통해 우수한 성능을 보입니다. 실험 결과는 서로 다른 쿼리 논리에 따른 고르지 않은 검색 성능을 보여주며, 후보 재정렬에서 구조적 경로를 통합할 때의 이점을 강조합니다. MoR의 구현 코드는 지정된 링크에서 확인할 수 있습니다.



### LangProBe: a Language Programs Benchmark (https://arxiv.org/abs/2502.20315)
- **What's New**: 이번 논문에서는 LangProBe라는 새로운 대규모 벤치마크를 소개하고 있습니다. LangProBe는 2000개 이상의 작업, 아키텍처, 최적화 기법, 언어 모델(LM) 조합을 평가하여 언어 프로그램 아키텍처와 최적화 전략의 영향을 연구합니다. 이 연구는 다양한 라인업의 언어 프로그램을 통해 최적화된 언어 프로그램이 비용 대비 품질을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: LangProBe의 기반이 되는 연구는 DSPy와 TextGrad와 같은 선언적 언어 프레임워크를 통해 언어 프로그램을 작성하고 자동화하는 접근 방식을 활용하고 있습니다. 이러한 프로그램은 외부 도구와의 통합 및 정보 흐름을 구성하고, 특히 외부 정보에 대한 접근을 요구하는 작업에 필수적입니다. 또한, MIPRO과 같은 최적화 기법들은 다양한 모델과 작업 조합에 대해 품질 향상을 제공하는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, 최적화된 언어 프로그램은 일반적인 모델 호출 방식보다 우수한 성능을 보이는 것으로 나타났습니다. 예를 들어, gpt-4o-mini에서 실행되는 최적화된 프로그램은 낮은 비용으로 뛰어난 성능을 발휘했습니다. 그러나 모든 문제에서 일관된 결과를 보이는 것은 아니며, 고급 모델을 통한 기본 문제 해결에 있어서는 추가적인 조합이나 최적화가 필요하지 않았습니다.



### ReCon: Enhancing True Correspondence Discrimination through Relation Consistency for Robust Noisy Correspondence Learning (https://arxiv.org/abs/2502.19962)
Comments:
          10 pages, 4 figures, Accepted by CVPR2025

- **What's New**: 이번 연구에서는 multimodal 데이터셋에서 올바른 대응을 정확히 식별할 수 있는 새로운 Relation Consistency 학습 프레임워크인 ReCon을 제안합니다. 기존 기법은 객체 간의 유사도 일치에 중점을 두고 있지만, 모달리티 내의 관계 일관성을 간과함으로써 잘못된 긍정과 부정의 식별 위험이 존재했습니다. ReCon은 이러한 문제를 해결하기 위해 교차 모달과 인트라 모달 관계 일관성을 함께 다룹니다.

- **Technical Details**: ReCon은 교차 모달 관계 일관성을 통해 서로 다른 모달리티의 긍정 쌍의 유사도 점수를 최대화하고, 인트라 모달 관계 일관성을 통해 모달리티 내의 객체의 관계를 설명하는 관계 행렬의 거리를 최소화합니다. 이러한 이중 제약 조건은 노이즈가 있는 훈련 데이터를 나누고, 각 전략으로 훈련하여 강력한 교차 모달 검색을 달성하는 데 기여합니다.

- **Performance Highlights**: Flickr30K, MS-COCO, Conceptual Captions를 포함한 세 개의 널리 사용되는 벤치마크 데이터셋에서 광범위한 실험을 통해 ReCon의 효과성과 우수성을 입증했습니다. 성능 측면에서 기존의 최첨단(SOTA) 방법들과 비교했을 때, ReCon은 소음 대응 문제를 효과적으로 완화하고, 진정한 대응의 식별력을 크게 향상시킵니다.



### Few-Shot Multilingual Open-Domain QA from 5 Examples (https://arxiv.org/abs/2502.19722)
Comments:
          Accepted by TACL; pre-MIT Press publication version

- **What's New**: 최근 다국어 오픈 도메인 질문 응답(MLODQA) 방식은 풍부한 언어별 훈련 데이터를 통해 유망한 결과를 보여왔습니다. 그러나 비대표 언어에 대한 적용은 상당한 주석 비용(annotation cost)으로 제한됩니다. 본 논문에서는 대형 언어 모델(LLMs)을 활용하여 대규모 다국어 데이터를 합성하는 few-shot learning 접근법을 소개합니다.

- **Technical Details**: 본 방법은 WikiData를 이용한 대규모 자기 지도(pre-training) 학습을 시작으로, 몇 개의 샘플(few-shot supervision)을 사용하여 LLMs에서 생성된 고품질 합성 다국어 데이터에 대한 훈련이 이어집니다. 최종 모델인 FsModQA는 MLODQA 및 교차 언어(cross-lingual)와 단일 언어(mono-lingual) 검색에서 기존의 few-shot 및 감독(supervised) 기준을 크게 능가합니다.

- **Performance Highlights**: 또한 본 방법은 영어 감독 데이터를 통해 새로운 언어에 대한 효과적인 제로 샷(zero-shot) 적응을 위해 확장 가능함을 보여줍니다. 이는 비용이 많이 드는 대규모 주석 없이도 MLODQA 작업에 일반적이고 적용 가능한 솔루션으로서의 가능성을 의미합니다.



### Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems (https://arxiv.org/abs/2502.19596)
- **What's New**: 본 논문에서는 RAG(리트리벌 증가 생성) 시스템의 개발을 위한 데이터 생성 파이프라인을 제안합니다. 이 시스템은 저자원 환경에서의 여러 도전과제를 해결하기 위해 다중 모달(raw multi-modal) 데이터를 구조화된 말뭉치(corpus)와 Q&A 쌍으로 변환합니다. 특히, 자동차 공학 분야에 적용되어 비-RAG 기준에 비해 사실 적합성 (+1.94), 정보성 (+1.16), 유용성 (+1.67)을 향상시켰습니다.

- **Technical Details**: 본 연구에서 제안하는 데이터 생성 파이프라인은 내부 문서의 다양한 포맷을 활용하여 QA 생성에 필요한 기본 자료를 제공합니다. 또한, 고급 리랭킹(re-ranking) 단계와 참조 매칭(reference matching) 알고리즘을 통합하여 답변의 정확성을 높이고 출처의 추적 가능성을 개선합니다. 이러한 기술들은 자동차 충돌 테스트와 같은 전문 지식에 초점을 맞춘 QA 시스템의 발전에 기여합니다.

- **Performance Highlights**: 자동차 공학 분야에서의 실험 결과는 모델이 생성한 답변의 질적 측면을 다각도로 평가한 결과, RAG 시스템이 가진 강력한 답변 기반 및 투명성을 강조합니다. 이 시스템은 정보 검색(Information Retrieval)에서 일반적으로 사용되는 다단계 접근 방식을 적용하여 참고 문서의 정확성을 보다 효과적으로 유지합니다. 이를 통해, 사용자에게 신뢰할 수 있는 답변을 제공하게 됩니다.



New uploads on arXiv(cs.CV)

### InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions (https://arxiv.org/abs/2502.20390)
Comments:
          CVPR 2025. Project Page: this https URL

- **What's New**: 최근 연구에서는 다양한 물체와의 상호작용을 시뮬레이션하기 위한 새로운 프레임워크인 InterMimic을 소개하고 있습니다. 이 프레임워크는 물리 기반 모션 이모테이션(physics-based motion imitation)을 통해 복잡한 인간-물체 상호작용을 학습할 수 있도록 돕습니다. 특히 커리큘럼 학습(curriculum learning) 전략을 도입하여 초기에는 완벽한 모션을 목표로 하고, 이후에 더 복잡한 상호작용을 학습하도록 설계되었습니다.

- **Technical Details**: InterMimic은 여러 개의 교사(teacher) 정책을 통해 수집된 불완전한 모션 캡처 데이터(motion capture data)를 활용하여 학습을 진행합니다. 교사 정책들은 특정한 상호작용을 모방하고 수정하는 데 중점을 두며, 학생(policy) 정책은 이를 통합하여 더 다양한 모터 스킬과 물리적 신뢰도를 높입니다. 이 과정에서 RL(fine-tuning) 기법을 적용하여 단순한 시연 복제를 넘어서 보다 질 높은 솔루션을 얻습니다.

- **Performance Highlights**: 실험 결과, InterMimic은 다양한 HOI 데이터셋에서 현실적이고 다양한 상호작용을 생성하는 능력을 갖추었음을 입증했습니다. 학습된 정책은 제로샷(zero-shot) 방식으로 일반화되며, 다양한 운동 생성기(kinematic generators)와 원활하게 통합될 수 있습니다. 이로 인해 단순한 모방을 넘어서 복잡한 인간-물체 상호작용의 생성적 모델링(generative modeling)이 가능해지었습니다.



### LIFT-GS: Cross-Scene Render-Supervised Distillation for 3D Language Grounding (https://arxiv.org/abs/2502.20389)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 3D 비전-언어 이해 모델을 학습하기 위해 3D 레이블 없이 2D와 밀접하게 연관된 접근 방식을 제시합니다. 새로운 방식인 LIFT-GS는 2D 손실 및 differentiable rendering 기법을 사용하여 3D 예측을 수행하며, 3D 마스크 레이블 없이도 언어를 통한 3D 마스크 지정을 가능하게 합니다. 이 접근 방식은 기존 방식보다 데이터 효율성을 높이며, 사전 훈련된 모델로부터 얻은 가짜 레이블을 활용하여 3D 비전-언어 작업의 성능을 크게 향상시킵니다.

- **Technical Details**: LIFT-GS는 이미지 프레임에 직접적으로 감독을 받으며, differentiable rendering을 통해 3D 특징과 마스크를 2D 프레임으로 변환하는 방법을 사용합니다. 이 모델은 point clouds(위치 및 색상)로 시각적 입력을 사용하여 기존 2D 특징 강화 포인트 클라우드 대신 더욱 효율적인 처리를 가능하게 합니다. 전체 파이프라인은 2D 마스크와 가짜 마스크 간의 손실을 최적화하여 종단 간 훈련을 수행합니다.

- **Performance Highlights**: LIFT-GS는 오픈-어휘 3D 인스턴스 분할과 3D 참조 지정을 포함한 두 가지 주요 3D 비전-언어 작업에서 주목할 만한 성능을 보였습니다. 실험 결과 사전 훈련이 처음부터 훈련된 모델보다 월등한 성능을 보이는 것을 확인했으며, 더 많은 데이터와 더 나은 2D 모델이 지속적으로 개선을 가져온다는 것을 보여주었습니다. 이는 3D VLG 모델이 데이터 부족의 영역에서 작동하고 있음을 강조합니다.



### Beyond Next-Token: Next-X Prediction for Autoregressive Visual Generation (https://arxiv.org/abs/2502.20388)
Comments:
          Project page at \url{this https URL}

- **What's New**: 이번 논문에서는 xAR이라는 새로운 autoregressive (AR) 모델링 프레임워크를 제안합니다. 기존의 토큰 정의를 확장하여 개별 패치, 셀, 서브샘플, 스케일 또는 전체 이미지를 나타낼 수 있는 개체 X로 정의합니다. 또한, Noisy Context Learning (NCL)이라는 새로운 훈련 방법론을 도입하여 노이즈가 있는 컨텍스트에서 학습하도록 모델을 유도함으로써 전통적인 teacher forcing에 따른 exposure bias를 줄입니다.

- **Technical Details**: xAR의 핵심은 이산적인 토큰 분류를 연속적인 개체 회귀 문제로 재정의하는 것입니다. 이 회귀 과정은 flow-matching 기법을 각 AR 단계에서 적용하여 이뤄집니다. 이를 통해 모델은 과거의 noisy entities를 조건으로 사용하여 훈련되며, 이는 모델이 오류를 학습하도록 도와줍니다. 새로운 X 구성요소를 통해 다양한 공간적 및 의미적 관계를 캡처할 수 있습니다.

- **Performance Highlights**: xAR-B 모델(172M)은 ImageNet-256 생성 벤치마크에서 DiT-XL과 SiT-XL(675M)보다 우수한 성능을 보였으며, 20배 더 빠른 추론 속도를 기록했습니다. 또한, xAR-H 모델(1.1B)은 FID 1.24로 새로운 최첨단 성과를 달성했으며, 이전 모델보다 2.2배 더 빠르게 작동합니다. 이러한 성과는 복잡한 비전 기초 모듈이나 고급 가이드 샘플링에 의존하지 않고 이뤄졌습니다.



### InsTaG: Learning Personalized 3D Talking Head from Few-Second Video (https://arxiv.org/abs/2502.20387)
Comments:
          Accepted at CVPR 2025. Project page: this https URL

- **What's New**: 인스태그(InsTaG)는 극소량의 훈련 데이터로 현실적인 3D 토킹 헤드를 빠르게 학습할 수 있는 새로운 프레임워크입니다. 이 방법은 가벼운 3DGS(person-specific synthesizer)와 보편적인 모션 프라이어(universal motion priors)를 기반으로 하여 고품질의 개인화된 3D 헤드를 생성할 수 있습니다. 빠른 학습 속도와 높은 수준의 개인화를 보장하며, 훈련 데이터의 요구 사항을 극적으로 줄여줍니다.

- **Technical Details**: 인스태그는 Identity-Free Pre-training 전략과 Motion-Aligned Adaptation 전략을 도입하여 설계되었습니다. 이 전처리는 다수의 개인 데이터부터 개인적이지 않은 모션 정보를 필터링하여 보편적인 모션 필드를 확립합니다. 그 후, 새로운 정체성을 위해 모션 필드를 사전에 훈련하여 적응을 돕는데 초점을 맞춥니다.

- **Performance Highlights**: 인스태그는 짧은 5초 비디오에서도 개인화된 3D 토킹 헤드를 학습할 수 있으며, 기존 방법들과 비교해 뛰어난 시각적 품질과 개인화를 제공합니다. 실험 결과, 다양한 정체성, 성별 및 언어에서 효과적으로高품질의 토킹 헤드를 생성할 수 있는 우수한 효율성과 일반화 능력을 보여주고 있습니다.



### Efficient Gaussian Splatting for Monocular Dynamic Scene Rendering via Sparse Time-Variant Attribute Modeling (https://arxiv.org/abs/2502.20378)
Comments:
          AAAI 2025

- **What's New**: 이번 논문에서는 Efficient Dynamic Gaussian Splatting (EDGS)을 소개함으로써 동적 장면을 보다 효율적으로 렌더링할 수 있는 방법을 제안합니다. EDGS는 희소한 시간-변화 속성을 모델링하여 동적 장면을 표현하며, 각 동적 요소를 명확히 제어합니다. 이를 통해 렌더링 속도를 개선하고 중복되는 Gaussian 수를 줄여 렌더링 품질을 높이는 방법을 구체화했습니다.

- **Technical Details**: EDGS는 희소 앵커-그리드 표현을 통해 동적 장면을 모델링하며, 밀집 Gaussian의 운동 흐름은 고전적인 커널 표현을 통해 계산됩니다. 시간-불변 속성의 앵커를 필터링하는 비지도 전략을 제안하여 변형 가능한 물체에만 집중하여 MLP에 입력합니다. 이로 인해 정적 영역에서의 지터링을 피하고 동적 속성을 정확하게 쿼리할 수 있습니다.

- **Performance Highlights**: 두 개의 실제 데이터셋을 통해 EDGS는 이전의 최신 방법들과 비교하여 렌더링 속도가 크게 향상되었음을 입증했습니다. 실험 결과, 높은 PSNR 점수와 더 적은 포인트의 시간-변화 속성을 쿼리하여 렌더링 품질이 향상됨을 보여주었습니다. 이는 동적 장면 처리의 중요한 성과로, 실시간 적용 가능성이 더욱 두드러집니다.



### Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation (https://arxiv.org/abs/2502.20370)
Comments:
          Accepted as ICLR 2025 conference paper

- **What's New**: 이 논문은 온라인 상에서 두 캐릭터 간의 인터랙션 생성 과제를 다루고 있습니다. 기존의 두 가지 설정에서는 상대방의 전체 동작 시퀀스를 바탕으로 자신의 동작을 생성하거나, 특정 조건에 따라 두 캐릭터의 동작을 공동으로 생성했습니다. 그러나 이러한 방식은 실제 인간의 인터랙션 과정을 제대로 모델링하지 못하므로, 우리는 과거 관찰된 동작을 기반으로 다음 캐릭터 포즈를 생성하는 새로운 반응 정책인 'Ready-to-React'를 제안합니다.

- **Technical Details**: Ready-to-React 정책은 각 캐릭터가 독립적인 반응 정책을 가지고 있어 실제 인간처럼 실시간으로 상호작용할 수 있도록 합니다. 이 정책은 확산 헤드(diffusion head)를 자동 회귀 모델(auto-regressive model)에 통합하여 상대방의 동작에 동적으로 반응하며 생성 과정에서의 오류 축적을 효과적으로 완화할 수 있게 구성됩니다. 또한, 우리 방법은 자가 수집한 복싱 데이터셋(DuoBox)을 활용한 포괄적인 실험을 수행하여 그 유효성을 검증했습니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 기존의 베이스라인보다 우수한 성능을 보여주며, 긴 동작 시퀀스를 생성할 수 있다는 것을 입증했습니다. 특히, 초기 포즈만으로도 약 1분 길이의 동작 시퀀스를 생성할 수 있었습니다. 이 방법은 VR 및 다른 온라인 인터랙티브 환경에 잘 적합하다는 점이 강조되었습니다.



### OpenTAD: A Unified Framework and Comprehensive Study of Temporal Action Detection (https://arxiv.org/abs/2502.20361)
- **What's New**: 이 논문은 다양한 Temporal Action Detection (TAD) 방법들을 통합한 OpenTAD라는 새로운 통일된 프레임워크를 제안합니다. OpenTAD는 16개의 다양한 TAD 방법과 9개의 표준 데이터셋을 결합하여 모듈화된 코드베이스를 제공합니다. 이를 통해 연구자들은 다양한 TAD 기법을 손쉽게 비교하고 평가할 수 있습니다. OpenTAD는 기존 기술을 활용하여 새로운 최첨단 TAD 방법을 개발할 수 있는 기초를 제공합니다.

- **Technical Details**: OpenTAD는 입력 비디오로부터 행동 카테고리와 시작 및 종료 타임스탬프를 예측하기 위해 세 단계의 네트워크 구성 요소를 포함합니다. 각 단계의 세부사항은 논문의 후속 섹션에서 설명되며, 데이터 전처리 및 후처리 파이프라인도 통합되어 있습니다. 다양한 TAD 방법의 성능을 공정하게 비교하기 위해 여러 가지 디자인 선택의 역할을 분석하여 전체 파이프라인을 모듈화합니다. 이런 통합된 접근법은 TAD의 평가를 통일된 프레임워크 내에서 가능하게 합니다.

- **Performance Highlights**: OpenTAD를 통해 수행된 광범위한 실험 결과, 여러 TAD 구성 요소와 데이터셋에 걸쳐 가장 효과적인 모듈 디자인이 식별되었습니다. 이 연구는 새로운 최첨단 성능을 달성했으며, 기존 TAD 방법을 개선하는 데 기여하고 있습니다. 또한 연구자들은 OpenTAD를 사용하여 보다 발전된 TAD 모델을 개발할 수 있는 기회를 갖게 됩니다. OpenTAD는 다양한 데이터셋과 응용 시나리오에 대한 확장을 지원할 수 있는 확장 가능한 기초를 제공하고 있습니다.



### ARTalk: Speech-Driven 3D Head Animation via Autoregressive Mod (https://arxiv.org/abs/2502.20323)
Comments:
          More video demonstrations, code, models and data can be found on our project website: this http URL

- **What's New**: 이 논문은 음성 기반의 3D 얼굴 애니메이션에서 실시간으로 동기화된 입 모양과 사실적인 머리 자세 및 눈 깜박임을 생성하는 새로운 자기 회귀 모델을 소개합니다. 기존의 확산 기반 방법들은 자연스러운 움직임을 생성할 수 있지만 느린 생성 속도로 인해 응용 가능성이 제한적이었습니다. 제안된 모델은 미리 학습된 모션 시퀀스를 사용하여 보지 못한 말하기 스타일에도 적응할 수 있어, 훈련 중에 보지 못한 독특한 개인 스타일의 3D 아바타 생성이 가능합니다.

- **Technical Details**: 제안된 ARTalk 모델은 음성을 시간 창으로 나누어 생성하는 기능을 제공하며, 이는 낮은 지연 시간의 동작 생성을 가능하게 합니다. 또한 VQ 오토인코더를 확장하여 두 개의 연속된 시간 창을 통한 다중 스케일 모션 코드를 인코딩하게 됩니다. 각 시간 창에서 음성 입력 및 이전에 생성된 동작을 기반으로 다중 스케일 모션 코드를 조건부로 생성하는 새로운 자기 회귀 생성기를 설계하여 시간 창 간의 시간적 연속성을 유지합니다.

- **Performance Highlights**: 광범위한 평가와 사용자 연구를 통해 제안된 방법이 입술 동기화 정확도와 인식된 품질에서 기존 접근 방식을 초월함을 입증했습니다. 개인화된 고충실도의 입 모양, 얼굴 표정 및 머리 동작 생성을 통해 다양한 음성 기반의 다운스트림 작업에 적합하다는 점이 강조되었습니다. 이를 통해 ARTalk 모델은 사실적이고 다양한 음성 기반 애니메이션 작업에서의 활용 가능성이 크게 증가하였습니다.



### UniTok: A Unified Tokenizer for Visual Generation and Understanding (https://arxiv.org/abs/2502.20321)
- **What's New**: 본 논문에서는 시각 생성과 이해를 통합한 새로운 프레임워크로 UniTok를 제안합니다. UniTok는 정교한 세부 정보를 인코딩하면서도 고수준의 의미를 포착할 수 있는 이산 시각 토크나이저입니다. 연구진은 전통적인 코드북의 한계를 극복하기 위해 멀티 코드북 양자화를 도입하여 잠재 기능 공간을 확장합니다.

- **Technical Details**: UniTok는 분할된 독립적인 서브 코드북을 사용하여 시각 토큰을 여러 조각으로 나누고 각 조각을 양자화합니다. 이러한 접근 방식은 코드북을 너무 크게 만드는 데서 오는 훈련의 불안정성을 피하면서도 표현력을 극대화합니다. 실험 결과에서는 UniTok가 Domain-specific continuous 토크나이저와 비슷하거나 높은 성능을 보이며, 이미지넷에서 FID 0.38을 달성합니다.

- **Performance Highlights**: UniTok는 78.6%라는 제로샷 정확도를 기록하며, 기존의 Domain-specific 토크나이저를 능가하는 성능이 확인되었습니다. 이 모델은 다중모드 이해와 생성 벤치마크에서 최첨단의 성능을 보여주며, 통합된 MLLM에서의 안정성과 효율성을 증명합니다.



### Multi-Scale Neighborhood Occupancy Masked Autoencoder for Self-Supervised Learning in LiDAR Point Clouds (https://arxiv.org/abs/2502.20316)
- **What's New**: 본 연구에서는 LiDAR 포인트 클라우드를 위한 Neighborhood Occupancy MAE (NOMAE)를 제안하여 기존 자가 지도 학습(self-supervised learning, SSL)의 한계를 극복합니다. NOMAE는 비가려진(visible) 부피의 이웃에서만 마스킹된 옥타(occupancy)를 재구성하는 방식을 채택해 정보 유출을 방지합니다. 또한, 다양한 크기의 객체의 특성을 캡처하기 위해 다중 스케일(multi-scale)에서 옥타 재구성을 통합해 새로운 경량 구조를 제공합니다.

- **Technical Details**: NOMAE는 LiDAR 포인트 클라우드의 희소성 문제를 다중 스케일에서 직접 해결하는 첫 번째 자가 지도 학습 프레임워크입니다. 이 프레임워크는 한阶에 걸쳐 자체 감독(self-supervision) 과정을 통해 여러 스케일에서 특성을 캡처할 수 있도록 설계되었습니다. 입력 포인트 클라우드는 볼륨 화(voxelization) 및 마스킹 과정을 거치며, 이를 통해 우리는 특성에 대한 정보를 효과적으로 유지합니다.

- **Performance Highlights**: NOMAE는 nuScenes와 Waymo Open 데이터셋에서의 평가를 통해 여러 인식(perception) 작업에서 최신 기술 수준으로 성능을 입증했습니다. 본 연구는 다양한 다운스트림 작업(semantic segmentation 및 3D object detection)에서 기존의 판별적(discriminative) 및 생성적(generative) SSL 방법과 비교하여 우수한 성능을 보여주었습니다. 특히 NOMAE는 여러 벤치마크에서 최상의 결과를 달성하며 수행되고 있습니다.



### FlexVAR: Flexible Visual Autoregressive Modeling without Residual Prediction (https://arxiv.org/abs/2502.20313)
- **What's New**: 이번 연구는 시각 자기회귀 모델링에서의 잔차 예측(residual prediction) 패러다임을 도전하며, FlexVAR라는 새로운 유연한 시각 자기회귀(image generation) 생성 패러다임을 제시합니다. FlexVAR는 사실(prediction) 예측을 통해 자기회귀 학습을 촉진하여 각 단계가 독립적으로 그럴듯한 이미지를 생성할 수 있도록 합니다. 이 시스템은 시각적 분포를 신속하게 학습하고 생성 과정을 더욱 유연하고 적응력 있게 만듭니다.

- **Technical Details**: FlexVAR는 저해상도 이미지(256px 이하)에 대해서만 훈련된 후 다양한 해상도 및 비율의 이미지를 생성할 수 있습니다. 각 단계에서 잔여(residual) 대신에 ground-truth 값을 예측함으로써 인접한 스케일 간의 의미적 일관성을 보장합니다. 또한, FlexVAR 트랜스포머는 다중 스케일 잠재 피처(multi-scale latent features)의 확률 분포를 학습하여 임의의 해상도로 이미지를 재구성할 수 있도록 설계되었습니다.

- **Performance Highlights**: FlexVAR는 ImageNet 256x256 벤치마크에서 1.0B 모델이 VAR 모델보다 더 뛰어난 성능을 보입니다. 13단계의 제로샷(zero-shot) 전이(transfer)를 통해 성능이 2.08 FID로 향상되어, 최신의 자기회귀 모델인 AiM/VAR 및 인기 있는 확산 모델인 LDM/DiT보다 각각 0.25/0.28 FID와 1.52/0.19 FID 더 높은 성능을 달성했습니다. 또한, FlexVAR는 1.0B 모델을 ImageNet 512x512 벤치마크에 제로샷으로 전이할 때 VAR 2.3B 모델에 비해 경쟁력 있는 결과를 나타냅니다.



### Mobius: Text to Seamless Looping Video Generation via Latent Shif (https://arxiv.org/abs/2502.20307)
Comments:
          Project page: this https URL ; GitHub repository: this https URL

- **What's New**: Mobius는 사용자 주석 없이 텍스트 설명에서 매끄러운 루프 비디오를 생성하는 새로운 방법을 제안합니다. 이 방법은 사전 훈련된 비디오 라텐트 디퓨전 모델을 활용하여 문제를 해결하며, 영상의 시작과 끝의 노이즈를 연결하여 라텐트 사이클을 구성합니다. 이를 통해 해당 방법은 이전 방식들과 달리 동적 모션 생성이 가능해집니다.

- **Technical Details**: Mobius에서는 라텐트 디노이징 과정에서 매 프레임이 동등하게 중요해야 한다는 점을 발견하고, 라텐트 이동 전략을 도입합니다. 이 과정에서 시작 프레임부터 끝 프레임까지 모든 노이즈 라텐트를 활용하여 사이클을 구축하고, 각 스텝에서 새로운 노이즈 라텐트를 생성합니다. 이러한 접근 방식은 영상의 시간적 일관성을 보장하며, 일반적인 비디오 디퓨전 모델보다 긴 루프 비디오 생성이 가능합니다.

- **Performance Highlights**: 실험을 통해 Mobius 방식이 다양한 시나리오에서 뛰어난 성능을 발휘함을 입증하였습니다. 기존의 시네마그래프 생성 방식보다 더 동적인 움직임과 자연스러운 시각 효과를 제공하며, 품질 면에서도 우수한 결과를 보여주었습니다. 또한, 이 방식은 해당 작업에 대해 훈련이 필요 없는 진전을 이루었으며, 생성된 코드 역시 공개할 예정입니다.



### SecureGaze: Defending Gaze Estimation Against Backdoor Attacks (https://arxiv.org/abs/2502.20306)
- **What's New**: 이번 연구에서 제안된 SecureGaze는 눈동자 추정(gaze estimation) 모델을 백도어 공격(backdoor attack)으로부터 보호하기 위한 최초의 솔루션입니다. 기존의 분류 모델(classification models) 방어책과는 달리 SecureGaze는 출력 공간(output space)의 연속성 및 특유의 글로벌 활성화(global activation) 문제를 해결하기 위해 새로운 접근법을 사용합니다. 이는 눈동자 추정 모델의 특징을 활용하여 신뢰할 수 있는 백도어 탐지를 가능하게 합니다.

- **Technical Details**: 눈동자 추정 모델의 방어는 특유의 재귀적 기능(reverse-engineer trigger function)을 감지하는 것을 포함합니다. 이를 통해 기존의 백도어 공격에 적합한 방어 기술들을 재조정하여 효과적으로 구현할 수 있음을 보여줍니다. 연구에서는 눈동자 추정 모델의 결과에 따라 다양한 백도어 공격을 분석하고 평가하며, 각 공격이 모델에 미치는 영향을 자세히 다룹니다.

- **Performance Highlights**: SecureGaze는 디지털 및 실제 환경 모두에서 여섯 가지 최신 백도어 공격에 대해 효과적인 방어를 입증하였습니다. 이를 통해 SecureGaze는 자동 운전 차량의 운전자가 주의력과 인지 상태를 제대로 인식하지 못하게 만드는 공격에 대한 안전성을 증대시키고 있습니다. 또한, 기존 분류 모델에서 파생한 일곱 가지 방어 방법보다 우수한 성능을 보여주며, 눈동자 추정 분야에서의 관련성을 높이고 있습니다.



### M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging (https://arxiv.org/abs/2502.20301)
Comments:
          38 pages, 7 figures

- **What's New**: 본 논문에서는 의료 이미징 분야에서의 머신 러닝(ML) 자동화를 위한 새로운 다중 에이전트 시스템인 M3Builder를 소개합니다. M3Builder는 복잡한 다단계 워크플로우를 관리하는 네 개의 전문화된 에이전트가 협력하여 데이터 처리, 환경 구성, 자동 디버깅 및 모델 훈련을 수행합니다. 이 시스템은 의학적 이미징 ML 작업을 위한 통합 환경인 Medical Imaging ML workspace와 함께 작동하여, AI 도구의 자율적 개발을 가능하게 합니다.

- **Technical Details**: M3Builder는 의료 이미징 분석을 위한 문제를 정의하고 이를 자동으로 해결하기 위해 다중 에이전트 협업 프레임워크를 적용합니다. 해당 프레임워크는 기계 학습(ML) 워크스페이스와 다중 에이전트 시스템의 두 가지 주요 구성 요소로 나뉘며, 여기에는 자연어로 설명된 데이터 카드, 툴셋 설명, 코드 템플릿이 포함됩니다. 이러한 요소들은 에이전트들이 상호작용하며 작업을 이행할 수 있도록 지원하는 구조적 환경을 제공합니다.

- **Performance Highlights**: M3Builder는 Claude-3.7-Sonnet을 에이전트의 핵심으로 사용하여 94.29%의 성공률을 기록하며, 기존의 ML 에이전트 디자인 대비 우수한 성능을 보여줍니다. 실험 결과는 장기 데이터 분할에서 86.67%, 이상 탐지에서 100%, 질병 진단에서 95%, 보고서 생성에서 93.33%의 성과를 나타냅니다. 이 결과는 의료 이미징에서의 완전한 자동화된 머신 러닝의 가능성을 시사합니다.



### Visual Adaptive Prompting for Compositional Zero-Shot Learning (https://arxiv.org/abs/2502.20292)
- **What's New**: 이번 논문은 비전-언어 모델(VLMs)을 활용하여 Compositional Zero-Shot Learning (CZSL)에서의 성능을 획기적으로 향상시킬 수 있는 Visual Adaptive Prompting System (VAPS)을 제안합니다. VAPS는 학습 가능한 시각적 프롬프트(retrieval mechanism)를 사용하여 시각적 특성과 의미적 정보를 연결하고, 이미지를 기반으로 동적으로 적합한 프롬프트를 선택합니다. 이는 전통적인 고정 프롬프트 방식의 한계를 극복하며, 보다 유연한 조합 학습을 가능하게 합니다.

- **Technical Details**: VAPS는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 학습 가능한 시각적 프롬프트 저장소를 구축하여 이미지에서 추출한 시각적 특성을 효율적으로 활용합니다. 둘째, 텍스트 프롬프트 어댑터를 통해 이미지의 시각적 맥락에 맞게 텍스트 프롬프트를 동적으로 업데이트하며, 이는 속성과 객체의 분리를 효과적으로 도와줍니다.

- **Performance Highlights**: VAPS는 세 가지 CZSL 벤치마크에서 실험을 통해 최신 기술(state-of-the-art) 결과를 달성했습니다. 이 방법은 기존의 정적 텍스트 프롬프트 기반 방법보다 높은 유연성과 일반화 능력을 보여 주며, 특히 새로운 조합을 학습하는 데 있어서 탁월한 성능을 발휘하였습니다.



### Explainable, Multi-modal Wound Infection Classification from Images Augmented with Generated Captions (https://arxiv.org/abs/2502.20277)
- **What's New**: SCARWID는 합성 캡션 증가 검색(Synthetic Caption Augmented Retrieval) 기술을 이용하여 당뇨병성 발 궤양(Diabetic Foot Ulcers, DFUs)에서 감염을 탐지하는 새로운 딥 러닝 프레임워크입니다. 기존 머신 러닝 방법들은 일반적으로 상처 이미지만 분석했지만, SCARWID는 GPT-4o로 생성된 텍스트 설명을 강화하여 감염 탐지를 개선합니다. 이 접근법은 감염 상태를 판별하기 위해 레이블이 붙은 지원 세트에서 유사한 아이템을 검색합니다.

- **Technical Details**: SCARWID는 두 가지 주요 구성 요소로 구성됩니다: (1) Wound-BLIP, 이는 GPT-4o로 생성된 설명을 활용하여 이미지로부터 일관된 캡션을 합성하는 비전-언어 모델(Vision-Language Model)입니다. (2) 이미지-텍스트 융합 모듈(Image-Text Fusion module)은 상관 주의(cross-attention)를 사용하여 이미지와 그에 해당하는 Wound-BLIP 캡션으로부터 교차 모달 임베딩을 추출합니다. SCARWID는 5개의 가장 유사한 이미지-텍스트 쌍을 검색하여 감염 상태를 결정합니다.

- **Performance Highlights**: SCARWID는 상처 감염 분류에서 평균 민감도 0.85, 특이도 0.78, 정확도 0.81을 기록하며 기존 모델들을 초월했습니다. 특히, 생성된 캡션을 상처 이미지와 함께 표시하여 간호사들이 SCARWID의 출력을 의학적 지식과 조화롭게 연결할 수 있도록 해줍니다. 이러한 접근은 특히 상처 노트가 없거나 시각적 속성을 파악하기 어려운 초보 간호사들에게 큰 가치를 제공합니다.



### HVI: A New color space for Low-light Image Enhancemen (https://arxiv.org/abs/2502.20272)
Comments:
          *These authors contributed equally to this work

- **What's New**: 이 논문에서는 저조도 이미지 향상(Low-Light Image Enhancement, LLIE)을 위한 새로운 색상 공간인 Horizontal/Vertical-Intensity (HVI)를 제시합니다. 기존의 LLIE 방법들은 sRGB 색상 공간을 기반으로 하여 색편향(color bias)과 밝기 아티팩트(brightness artifacts)를 발생시켰습니다. HVI는 극성화된 HS 맵 및 학습 가능한 강도를 통해 이러한 문제를 해결하고자 합니다.

- **Technical Details**: HVI 색상 공간은 빨간색 좌표 간의 거리를 최소화하여 빨간색 아티팩트 제거를 강제합니다. 또한, 낮은 조도 지역을 압축하여 검은색 아티팩트를 제거합니다. 이를 위해 새로운 Color and Intensity Decoupling Network (CIDNet)를 도입하여 HVI 공간에서 다양한 조명 환경에 따른 정확한 광도 매핑 기능을 학습합니다.

- **Performance Highlights**: 제안된 HVI 색상 공간과 CIDNet은 10개의 데이터셋에서 최신 기법(State-of-the-art methods)보다 뛰어난 성능을 보여줍니다. 종합적인 벤치마크 및 배제 실험 결과는 이 새로운 접근 방식의 유효성을 입증합니다. 코드는 제공된 URL에서 확인할 수 있습니다.



### Vector-Quantized Vision Foundation Models for Object-Centric Learning (https://arxiv.org/abs/2502.20263)
- **What's New**: 이번 연구에서는 Object-Centric Learning (OCL)에서 Vision Foundation Models (VFMs)을 완전히 활용하는 Vector-Quantized Vision Foundation Models for Object-Centric Learning (VQ-VFM-OCL, VVO)를 제안합니다. 기존 OCL 방법들이 VFMs를 단순한 feature extractor로 사용했으나, VVO는 feature를 추출하고 양자화하여 reconstruction에서의 감독이 강해지도록 합니다. 이러한 접근 방식은 복잡한 텍스처에서의 자아 감독(supervision) 문제를 해결하고 초기 슬롯의 집합 성능을 개선합니다.

- **Technical Details**: VVO는 encoder-aggregator/quantizer-decoder 구조를 채택하여 데이터의 다양성을 증가시킵니다. Encoder는 입력 영상을 dense feature map으로 변환하고, 이 후 aggregator는 Slot Attention을 사용하여 feature vectors로 변환합니다. Quantizer에서는 VFM feature map을 공유 코드북을 사용하여 양자화하며, 이를 통해 서로 간의 중복된 feature 정보를 억제하고, consistent reconstruction targets를 생성합니다.

- **Performance Highlights**: 실험 결과, VVO는 객체 발견(object discovery) 작업에서 주요 방법들보다 뛰어난 성능을 보였으며, 시각적 예측(visual prediction) 및 추론(reasoning)과 같은 다운스트림 작업에서도 이점을 제공합니다. 이 연구는 OCL의 다양한 모듈을 지원하는 통합 아키텍처를 제공하여, 기존 OCL 방법들에 비해 성능을 크게 개선하는 데 기여합니다.



### Do computer vision foundation models learn the low-level characteristics of the human visual system? (https://arxiv.org/abs/2502.20256)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구는 DINO 및 OpenCLIP과 같은 컴퓨터 비전 기반 모델들이 자연 이미지 데이터셋을 self-supervised 방식으로 학습한 결과, 인간 시각 시스템(HVS)의 낮은 수준 특성을 모방하는지 여부를 조사합니다. 저자들은 45개의 모델을 대상으로 새로운 평가 프로토콜을 설계하여 대비 탐지(contrast detection), 대비 마스킹(contrast masking) 및 대비 불변성(contrast constancy) 등 여러 특성을 측정하였습니다. 그 결과, DINOv2와 OpenCLIP과 같은 일부 모델들은 인간 시각과 유사한 특성을 나타내는 반면, 다른 모델들은 큰 차이를 보였습니다.

- **Technical Details**: 연구에서 사용된 프로토콜은 잘 측정된 인간 시각 과학의 이론을 기반으로 하여 아홉 가지의 검사 유형을 포함하고 있습니다. 특히, 모델들은 Gabor 패치와 주파수 제한 소음 등 기본 비전 자극에 대해 테스트되었으며, 응답은 인간 관찰자로부터 수집된 심리 물리학적 데이터와 비교되었습니다. 이러한 방법론은 깊은 신경망을 블랙박스 모델로 간주하고, 이를 통해 저자들은 컴퓨터 비전 모델이 인간 시각 시스템과 유사한 '병목' 및 불변 특성을 공유하는지 검토하였습니다.

- **Performance Highlights**: 실험 결과, DINOv2와 OpenCLIP은 특히 대비 마스킹 테스트에서 인간 시각 데이터와 가장 높은 유사성을 보였습니다. 그러나, 다른 검사들에서는 여전히 뚜렷한 차이가 존재했습니다. 연구진은 컴퓨터 비전과 인간 비전이 실세계 이미지를 해석하는 방식에서 유사성과 차이점을 모두 보인다며, DINOv2가 인간 시각과 가장 가까운 유사성을 띄고 있다고 결론지었습니다.



### Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels (https://arxiv.org/abs/2502.20249)
- **What's New**: 본 연구에서는 자가 학습 약한 감독 기법을 통해 3D 시선 추정을 위한 새로운 프레임워크(ST-WSGE)를 소개합니다. 기존의 2D 시선 데이터셋을 활용하여 3D 가상 라벨을 생성하고 모델의 일반화를 향상시키는 방식으로, 이미지와 비디오 데이터셋을 동시에 학습하는 모달리티 불가지론적(Gesture Modal)인 Gaze Transformer(GaT) 구조를 제안합니다. 또한 본 연구는 Gaze360 및 GFIE와 같은 벤치마크에서 기존 기술보다 상당한 성능 향상을 보여줍니다.

- **Technical Details**: ST-WSGE는 두 단계의 학습 프로세스를 통해 기존의 3D 시선 데이터셋에 대해 시선 네트워크를 훈련한 후, 시선 추적 데이터에서 예측한 결과를 바탕으로 3D 시선 가상 라벨을 생성합니다. 그 후, 이 가상 라벨을 함께 사용하여 동일한 시선 네트워크를 재훈련합니다. GaT는 이미지와 비디오 입력을 공동으로 학습할 수 있는 능력을 가지고 있어 본 연구에서 다루는 여러 데이터셋에서 학습하여 다양한 이점을 제공합니다.

- **Performance Highlights**: 본 접근 방식은 Gaze360 및 GFIE와 같은 제약 없는 환경에서, MPIIFaceGaze와 같은 제약된 환경에서도 기존의 방법보다 우수한 성능을 달성하였습니다. 특히, 데이터셋 간 평가에서 뛰어난 결과를 보이며, 실제 환경에서의 3D 시선 추정에 적합한 해결책으로 자리 매김하고 있습니다.



### Attention Distillation: A Unified Approach to Visual Characteristics Transfer (https://arxiv.org/abs/2502.20235)
Comments:
          Accepted to CVPR 2025. Project page: this https URL

- **What's New**: 최근 생성적 확산 모델(generative diffusion models)의 발전은 이미지 스타일 및 의미에 대한 상당한 이해를 보여주었습니다. 본 논문에서는 사전 훈련된 확산 네트워크의 셀프 어텐션(feature) 특징을 활용하여 참조 이미지에서 생성된 이미지로 시각적 특성을 전이하는 방법을 제안합니다. 기존의 연구들과는 달리, 추상화된 결과와 현재 스타일화 결과 간의 새로운 어텐션 증류 손실(attention distillation loss)을 제안하며, 이를 통해 잠재 공간(latent space)에서 합성된 이미지를 최적화합니다.

- **Technical Details**: 우리는 어텐션 증류 손실을 기반으로 한 새롭고 향상된 클래스 분류기 가이던스(Classifier Guidance)를 제안합니다. 이는 노이즈 제거 샘플링(denoising sampling) 과정에 통합되어 이미지 합성을 가속화하고 다양한 이미지 생성 어플리케이션에 활용될 수 있게 합니다. 이러한 접근법은 고급 특징을 활용하여 지식 전이를 더욱 효과적으로 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 우리 접근 방식의 스타일, 외관, 질감을 새로운 이미지에 전이하는 데 있어 비범한 성능을 입증했습니다. 이 연구의 결과는 새로운 이미지 생성을 위한 강력한 도구를 제공하며, 코드도 함께 배포되어 있습니다.



### Deep Convolutional Neural Networks for Palm Fruit Maturity Classification (https://arxiv.org/abs/2502.20223)
- **What's New**: 이 연구는 최적의 성숙 단계에서 팜 과일을 수확하여 팜 오일의 수량과 품질을 극대화하는 것을 목표로 하며, 자동화된 컴퓨터 비전 시스템을 개발하여 팜 과일 이미지를 다섯 가지 상태로 분류합니다. 딥 컨볼루션 신경망 (CNN)을 사용하여 과일의 성숙 단계에 따라 이미지를 분류하고, 사전 훈련된 ResNet50 및 InceptionV3 아키텍처를 활용하여 전이 학습과 미세 조정을 적용합니다.

- **Technical Details**: 이 연구에서는 평균 85% 이상의 정확도로 팜 과일 성숙 단계를 분류하기 위한 딥 CNN 모델을 제안하였습니다. 8,000장 이상의 이미지로 구성된 공개 데이터셋을 사용하여 80%는 훈련, 20%는 테스트로 나누어 실험을 수행했으며, 이 과정에서 CNN의 효과적인 특징 추출 능력을 활용합니다. 특히, 색상 모델의 RGB 및 HSI에서 특징을 추출하고, 이를 부가적인 분류 알고리즘에 주입하여 성숙 단계를 분류합니다.

- **Performance Highlights**: 제안된 딥 CNN 모델은 팜 과일 성숙 단계를 분류하는 데 있어 85% 이상의 높은 테스트 정확도를 기록하며, 이는 자동화된 성숙도 평가의 잠재력을 강조합니다. 이 연구는 팜 오일 생산 효율 증대와 수확 결정 최적화를 위한 기여로 나아갈 수 있는 중요한 성과를 이루었습니다.



### Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars (https://arxiv.org/abs/2502.20220)
Comments:
          Project website: this https URL, Video: this https URL

- **What's New**: Avat3r라는 새로운 방법을 통해 단 몇 장의 이미지로 고품질의 애니메이션 가능한 3D 헤드 아바타를 생성할 수 있습니다. 제작 소요 시간과 컴퓨팅 요구 사항을 대폭 줄였으며, 이를 통해 사용자는 스마트폰으로 촬영한 이미지를 통해 신뢰할 수 있는 디지털 더블을 쉽게 만들 수 있습니다. 기존의 스튜디오 수준의 세팅과 비싼 최적화 과정이 필요 없기 때문에 더욱 다양한 환경에서 활용 가능합니다.

- **Technical Details**: Avat3r는 DUSt3R를 통해 포지션 맵(position maps)을 계산하고, 레이어 간의 스킵 연결을 이용하여網(網) 벡터를 사용자 지정합니다. 또한, 단순한 크로스 어텐션(cross-attention)을 사용하여 표정 코드(expression code)로부터 3D 헤드 애니메이션을 재현합니다. 이러한 구조적 접근 방식은 다양한 표정에서의 강인성을 증가시키고, 불완전한 데이터 입력에 대해서도 효과적으로 대응할 수 있게 합니다.

- **Performance Highlights**: Avat3r는 단일 입력 이미지 또는 소량의 입력 이미지로 애니메이션 가능한 3D 헤드 아바타를 생성하는 데 있어 현재의 최첨단 방법들과 비교해 높은 경쟁력을 보입니다. 다양한 출처(소스)에서의 이미지, 스마트폰 캡쳐, 단일 이미지 등으로부터 3D 아바타를 생성하는 넓은 적용 가능성을 보여주고 있습니다. 이는 VFX 산업을 넘어선 여러 분야에서도 활용될 수 있는 잠재력을 지니고 있습니다.



### DIPSER: A Dataset for In-Person Student1 Engagement Recognition in the Wild (https://arxiv.org/abs/2502.20209)
- **What's New**: 본 논문에서는 학생들의 주의를 평가하기 위해 설계된 새로운 데이터셋이 소개되었습니다. 이 데이터셋은 RGB 카메라 데이터와 각 학생마다 여러 카메라를 통해 자세와 표정을 캡처하며, 개인별 스마트워치 센서 데이터도 포함하고 있습니다. 이 데이터셋은 머신러닝 알고리즘을 학습시켜 주의를 예측하고 감정과의 관계를 분석할 수 있도록 합니다.

- **Technical Details**: 이 데이터셋은 1,311,761개의 이미지를 포함하고 있으며, 고해상도 이미지와 함께 감정과 주의에 대한 레이블을 제공합니다. 또한 심박수, 가속도계, 자이로스코프 등의 IMU 데이터를 통합하여 다양한 변수를 분석할 수 있는 가능성을 열어줍니다. 비교적 단국한 시간에 진행된 기존 데이터셋들과는 달리 본 데이터셋은 5분 무비 영상을 포함하여 대면 교육 설정에서의 주의력을 더 정확히 측정할 수 있도록 합니다.

- **Performance Highlights**: 전체 데이터셋은 학생의 상호 작용을 다양한 교육적 문맥에서 포괄적으로 담고 있으며, 비교적 뚜렷한 메타데이터가 추가되어 있습니다. 기존의 데이터셋과 비교할 때, DIPSER 데이터셋은 특히 대면 클래스 환경에서의 학생 주의력 분석에 있어 가장 포괄적이며 다각적인 접근 방식을 제공합니다. 이로써 교육 분야의 인공지능 도구 연구에 기여할 수 있는 새로운 기회를 열어줍니다.



### 4Deform: Neural Surface Deformation for Robust Shape Interpolation (https://arxiv.org/abs/2502.20208)
Comments:
          CVPR25

- **What's New**: 본 논문에서는 비정형 데이터(예: 포인트 클라우드) 간의 현실적인 중간 형태를 생성하는 새로운 방법인 4Deform을 소개합니다. 기존의 메쉬 기반 방법과는 달리, 4Deform은 신경 임플리트 표현(neural implicit representation)을 활용하여 자유로운 수명 조정형 변형을 가능하게 합니다. 이 접근법은 중간 형태 지도 없이 훈련이 가능하며, 물리적 및 기하학적 제약을 도입하여 속도 필드를 정규화합니다.

- **Technical Details**: 4Deform은 유클리드 공간에서 연속적인 속도 필드를 학습하여 비정형 데이터에 적합하게 설계되었습니다. 방법론은 임플리트 필드와 속도 필드를 통해 형태 변형을 모델링하며, 새로운 두 가지 손실 함수를 도입하여 물리적으로 타당한 변형을 보장합니다. 이 연구는 링크된 속도 필드를 통해 수정된 레벨 세트 방정식을 사용하여 중간 표면을 재구성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 여러 시나리오에서 이전의 신경 임플리트 접근법을 뛰어넘는 성능을 보여줍니다. 본 연구는 4D Kinect 시퀀스 업샘플링, 실제 고해상도 메쉬 변형과 같은 새로운 응용 프로그램을 가능하게 하며, 노이즈가 많은 데이터와 부분적인 데이터를 처리하는 데 있어 효과적입니다. 최종적으로, 이 연구는 복잡한 변형이나 비동형 변형 문제에 대한 해결책을 제시하며, 돌발 변형이 필요한 여러 다운스트림 작업에 대한 적용 가능성을 보여줍니다.



### Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think (https://arxiv.org/abs/2502.20172)
Comments:
          13 pages, 9 figures, codebase in this https URL

- **What's New**: 최근 텍스트-이미지 생성에 관한 연구에서 Dream Engine이라는 새로운 통합 프레임워크가 제안되었습니다. 이 프레임워크는 강력한 텍스트 인코더를 활용하여 텍스트와 이미지를 효과적으로 결합하여 복잡한 생성 명령을 처리할 수 있도록 설계되었습니다. 기존의 제어 방법이 고급 텍스트-이미지 상호작용을 지원하지 못했던 문제를 해결하려고 하며, 대규모 다중 모달 모델(LMM)의 잠재력을 활용하고 있습니다.

- **Technical Details**: Dream Engine은 두 단계의 훈련 패러다임을 적용하여 텍스트와 이미지의 정렬을 최적화합니다. LMM과 텍스트-이미지 확산 모델을 연결하기 위해 경량 프로젝터 레이어를 사용하여 입력되는 텍스트와 이미지 간의 효과적인 매핑을 지원합니다. 또한, 객체 감지와 이미지 캡셔닝을 결합한 객체 주도 생성 기능을 도입했습니다.

- **Performance Highlights**: 실험 결과, Dream Engine은 GenEval 벤치마크에서 0.69의 점수를 기록하며 SD3.5 및 FLUX와 유사한 성능을 보였습니다. 압축 데이터를 통해 훈련했음에도 불구하고, 이 모델은 상호작용하는 텍스트 및 이미지 지침을 효과적으로 처리하며, 다양한 입력 이미지에서 개념을 합성하는 능력을 보였습니다.



### Learning to Generalize without Bias for Open-Vocabulary Action Recognition (https://arxiv.org/abs/2502.20158)
- **What's New**: 이 논문은 Open-vocabulary Action Recognition (OVAR)에서의 강력한 일반화를 위해 새롭고 효과적인 메타 최적화 프레임워크인 Open-MeDe를 소개합니다. CLIP의 정적 일반화 문제를 해결하고, 비디오 학습자가 더 나은 실험 성능을 발휘할 수 있도록 하는 데 중점을 둡니다. Open-MeDe는 메타 학습 접근 방식을 통해 영상-비디오 디바이싱(image-to-video debiasing)을 개선하고 있습니다.

- **Technical Details**: Open-MeDe는 비디오 학습자가 추후의 데이터에 신속하게 적응하도록 유도하는 cross-batch 메타 최적화 스킴을 적용합니다. 이는 가상 평가를 통해 학습자의 최적화 환경을 부드럽게 만들고, CLIP 정규화 없이 최적의 파라미터들을 얻기 위해 자가 앙상블(self-ensemble) 기법을 사용합니다. 메타 학습을 통해 학습자는 정적 편향을 최소화하고 보다 강력한 일반화 성능을 기대할 수 있습니다.

- **Performance Highlights**: Open-MeDe는 기존의 정규화 방법들보다 뛰어난 성능을 보여주며, 특히 in-context 및 out-of-context 개방 어휘(open-vocabulary) 상황 모두에서 강력한 일반화를 성취합니다. 다양한 데이터셋에서의 평가를 통해, 이 모델은 기초부터 새롭게 등장하는 데이터에 걸쳐 일관되게 성능을 향상시키는 것을 입증하였습니다.



### Adaptive H&E-IHC information fusion staining framework based on feature extra (https://arxiv.org/abs/2502.20156)
- **What's New**: 이번 연구에서는 면역조직화학 염색(IHC) 이미지를 생성하기 위한 새로운 접근 방식을 소개합니다. 기존의 모델들이 H&E(헤마톡실린-오신) 이미지의 픽셀 특성을 기반으로 염색을 생성하는 데 한계가 있었지만, 제안된 방법은 VMFE(다중 스케일 기능 추출기) 모듈을 통해 이 문제를 해결합니다. 이 모듈은 염색 정보 기능을 효율적으로 추출하고 공유 디코더를 통해 이를 융합하여 더 정확한 IHC 이미지를 생성할 수 있도록 합니다.

- **Technical Details**: 연구에서 제안한 방법은 VMFE 모듈을 중심으로 구성되어 있으며, 이는 웨이브렛 변환 합성을 사용하여 H&E 이미지를 처리합니다. 또한, 크로스 어텐션 모듈을 통해 H&E 이미지에서 얻어진 특징 맵과 생성된 IHC 이미지의 속성을 융합하여 IHC 이미지 생성을 보다 정밀하게 이루어지도록 합니다. 대조 학습을 통해 HE 및 IHC 인코더를 사전 훈련하여, 라텐트 공간에서 HE 및 IHC 이미지의 염색 레이블을 정렬하고, 동적 L1 손실 메커니즘을 통해 정확도를 개선한 점이 특징입니다.

- **Performance Highlights**: 제안된 모델은 다양한 데이터 세트에서 테스트를 수행하여 우수한 성능을 입증하였습니다. 모델의 구조와 기능 향상을 통해 생성된 IHC 이미지에서의 정보 손실 및 비대칭 정보 문제를 효과적으로 해결하였습니다. 이를 통해 가상의 염색 과정을 개선하고, 특히 의료 분야에서의 활용 가능성을 높이는 데 기여하고 있습니다.



### Cutting-edge 3D reconstruction solutions for underwater coral reef images: A review and comparison (https://arxiv.org/abs/2502.20154)
- **What's New**: 이번 연구는 최근의 3D 재구성 기술을 보다 체계적으로 검토하며, 수중 산호초 이미지를 다루는 과정에서의 두 가지 중요한 단계인 카메라 포즈 추정과 밀집 표면 재구성을 집중적으로 설명합니다. 특히 전통적인 방법과 현대적인 딥 러닝 기술을 결합하여 산호초의 정확한 3D 모델을 생성하는 것에 대한 필요성을 강조하고 있습니다. 또한, 이 논문은 수중 이미지 처리의 구체적인 요구사항을 충족하기 위해 현재의 기술 동향을 종합적으로 평가하고 있습니다.

- **Technical Details**: 3D 재구성 프로세스는 데이터 수집, 카메라 포즈 추정, 밀집 표면 재구성의 세 가지 단계로 나눌 수 있습니다. 카메라 포즈 추정은 SfM(Structure-from-Motion) 기법을 사용하여 이루어지며, 이미지의 특징 추출과 매칭을 통해 3D 공간에서 카메라의 위치와 방향을 정확히 결정합니다. 밀집 표면 재구성 단계는 MVS(Multi-View Stereo) 같은 기법을 활용하여 정밀한 장면 모델을 생성하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 기존의 전통적인 방식과 비교하여 최신 3D 재구성 기술들은 수중 산호초 장면의 복잡성을 해결하는 데 있어 많은 성과를 보이고 있습니다. 연구자들은 수십 센티미터 또는 밀리미터 수준의 정확성을 달성했지만, 빛의 왜곡, 고난이도 텍스처, 촬영 각도 문제 등의 어려움이 여전히 존재합니다. 다양한 환경에서의 테스트 결과들은 첨단 기술이 전통적인 접근 방식보다 우수한지를 확인하기 위한 추가 검증 필요성을 보여주고 있습니다.



### Robust sensitivity control in digital pathology via tile score distribution matching (https://arxiv.org/abs/2502.20144)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 Whole Slide Image (WSI) 분류 모델의 민감도(sensitivity)를 최적 수송(optimal transport) 및 다중 인스턴스 학습(Multiple Instance Learning, MIL)을 기반으로 제어하는 새로운 접근 방식을 제시합니다. 이는 임상 환경에서 중요한 다양한 메트릭(metrics)을 조율하기 위한 실용적인 솔루션을 제공합니다. 우리의 방법은 적은 수의 보정 샘플로 강력한 민감도 제어를 가능하게 하며, 여러 집단(cohorts) 및 과제에서 효과적으로 검증되었습니다.

- **Technical Details**: 제안된 방법인 Tile-Score Matching (TSM)은 WSI 이진 분류 문제에서 민감도를 제어하는 새로운 방법론으로, 기존의 방법들보다 훨씬 적은 데이터로 보정을 수행할 수 있습니다. TSM은 tile 레벨에서 예측 점수의 분포를 조정하여, 임상 적용 시 필요로 하는 보정 데이터를 최소화합니다. 이 방법은 기존의 Unsupervised Prediction Alignment (UPA)와 유사하지만, WSI 레벨이 아닌 tile 레벨에서 작동하여 여러 배수의 보정 샘플을 사용 가능합니다.

- **Performance Highlights**: 실험 결과, TSM은 매우 낮은 데이터 및 유병률(prevalence) 상황에서 민감도를 효과적으로 제어할 수 있음을 보여주었습니다. 특히, 오직 5개의 양성 샘플만으로도 보정이 가능하여 기존 방법과 비교할 때 현저히 낮은 요구사항을 충족시킵니다. 우리의 연구는 디지털 병리학 모델을 더 넓은 임상환경에 신뢰성 있게 배포할 수 있도록 하는 방법론적 기초를 제공합니다.



### Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models (https://arxiv.org/abs/2502.20134)
- **What's New**: 이 연구는 모든 비전 신경망을 공간적(spatial) 및 개념적(conceptual)으로 해석 가능한 모델로 변환하는 통합 프레임워크를 제안합니다. 저자들은 사전 훈련된 신경망의 '블랙박스(black-box)' 특성을 해석 가능한 개념 맵으로 변환하는 공간 인식 개념 병목층(spatially-aware concept bottleneck layer)을 도입했습니다. 이러한 새로운 접근 방식은 사람의 레이블을 필요로 하지 않고, 독립적으로 작용하는 자가 설명 모델을 생성합니다.

- **Technical Details**: 제안된 방식(SALF-CBM)은 개념 기반(global) 및 히트맵 기반(local) 설명을 모두 제공하는 첫 번째 레이블 없는 개념 병목 모델입니다. 기존의 개념 병목 모델(CBM)은 이미지에서 개념의 위치를 명시적으로 나타내지 못했지만, SALF-CBM은 CLIP을 활용하여 라벨 없이 공간 개념 공간으로 예측 결과를 전이합니다. 이 방법은 CNN 및 변환기(transformer) 아키텍처에 모두 적용 가능하며, 추가적인 학습 가능 매개변수를 도입하지 않고도 공간 정보를 유지합니다.

- **Performance Highlights**: SALF-CBM은 여러 분류 작업에서 비공간적(non-spatial) CBM보다 우수한 성능을 보이며, 원래의 비-CBM 모델보다 더 나은 결과를 달성합니다. 또한, 이 방법은 제로 샷 분할(zero-shot segmentation) 작업에서 흔히 사용되는 히트맵 기반 방법보다 높은 품질의 히트맵을 생성합니다. 마지막으로, 사용자가 특정 이미지 영역을 쿼리하고 개념 맵을 수정함으로써 모델의 결정을 조정할 수 있는 기능을 제공합니다.



### QPM: Discrete Optimization for Globally Interpretable Image Classification (https://arxiv.org/abs/2502.20130)
- **What's New**: 이번 논문에서는 딥 뉴럴 네트워크의 분류를 이해하기 위한 새로운 접근 방식을 제시합니다. 최근 모델들이 단일 결정에 대해 지역적으로 설명할 수 있었던 반면, 정확한 모델의 전반적인 행동을 신뢰성 있게 설명하는 것은 여전히 도전 과제였습니다. 이를 해결하기 위해 Quadratic Programming Enhanced Model (QPM)을 도입하여 전 세계적으로 해석 가능한 클래스 표현을 학습합니다.

- **Technical Details**: QPM은 각 클래스를 5개의 특징으로 이진 할당(binary assignment)하여 표현합니다. 이 특징들은 다른 클래스와도 공유되어 대조적인 클래스 표현을 쉽게 비교할 수 있도록 설계되었습니다. 최적의 할당은 미리 정의된 유사성 측정 및 해석 가능성 제약을 기반으로 한 이산 최적화(discrete optimization)를 통해 찾아지며, 이 결과는 다양한 특징을 미세 조정(fine-tune)하는 데 사용됩니다.

- **Performance Highlights**: QPM은 소규모 및 대규모 데이터셋에서 전례 없는(global interpretability) 전세계적 해석성을 제공하며, 해석 가능한 모델의 정확도(state of the art)에서도 최고의 성과를 기록하였습니다. 이 모델은 안전-critical한 상황에서 사용될 수 있으며, 대규모 활용을 염두에 두고 개발되었습니다.



### CLIP-driven Dual Feature Enhancing Network for Gaze Estimation (https://arxiv.org/abs/2502.20128)
- **What's New**: 최근의 복잡한 응용 시나리오는 정밀하고 일반화 가능한 시선 추정 방법에 대한 중요한 요구를 포함하고 있습니다. 이 논문에서는 CLIP(Contrastive Language-Image Pre-training)을 기반으로 한 새로운 '주요-측면(main-side)' 협력 강화 전략을 활용하여 시선 추정 성능을 높인 CLIP-DFENet을 제안합니다. 특히, CLIP의 텍스트 인코더를 사용하여 시선의 의미적 차이를 드러내는 언어 주도 차별 모듈(LDM)을 설계하였습니다.

- **Technical Details**: 우리의 CLIP-DFENet은 CLIP의 이미지 및 텍스트 인코더의 특징을 활용하여 주 시선 추정 네트워크가 시선 관련 특징을 추출하도록 지원하는 구조입니다. LDM은 대조 학습을 통해 작은 샘플 쌍의 시선 차이를 언어로 표현하는 데 도움을 주며, 비전 주도 융합 모듈(VFM)은 CLIP의 시각 표현 능력을 활용하여 시선 특징의 일반화를 더욱 향상시키는 데 중점을 둡니다. 이를 통해 강화된 특징을 기반으로 강력한 더블 헤드 시선 회귀기를 이용하여 시선 방향을 매핑합니다.

- **Performance Highlights**: 광범위한 실험 결과는 CLIP-DFENet이 여러 도메인 내 및 교차 도메인 작업에서 탁월한 식별력과 일반화 능력을 보였음을 보여줍니다. 또한 우리의 네트워크는 기존의 도메인 일반화 접근 방식들과 경쟁할 수 있는 성능을 달성하며, 최신 기술에 비해 우수성을 입증합니다. 이를 통해 CLIP의 잠재력이 시선 추정 성능 향상에서 어떻게 활용될 수 있는지를 보여줍니다.



### Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion (https://arxiv.org/abs/2502.20120)
- **What's New**: 이 논문에서는 멀티모달 학습(Multimodal Learning, MML)의 성능 향상을 위해 약한 모달리티의 분류 능력을 동적으로 조정하는 지속적인 부스팅 알고리즘(Sustained Boosting Algorithm)을 제안합니다. 기존의 많은 방법들이 학습 과정의 균형에 중점을 두었지만, 본 연구는 약한 모달리티의 성능을 직접적으로 높이는 데 중점을 둡니다. 이를 통해 강한 모달리티와 약한 모달리티 간의 불균형 문제를 해결하고자 합니다.

- **Technical Details**: 본 연구의 핵심은 특수하게 설계된 구성 가능한 분류기 모듈(Configurable Classifier Module)을 통해 약한 모달리티의 분류 성능을 강화하는 것입니다. 또한, OGM 점수를 활용하여 공동 학습 중 학습 상태를 모니터링하며, 적응형 분류기 할당 전략(Adaptive Classifier Assignment Strategy)을 통해 약한 모달리티의 분류 성능을 동적으로 개선합니다. 이 접근법은 강한 모달리티와 약한 모달리티의 분류 능력을 균형 있게 만들어줍니다.

- **Performance Highlights**: 광범위한 데이터셋에 대한 실험 결과, 제안한 방법이 다양한 최신 멀티모달 학습 기법들과 비교했을 때 우수한 성능을 보임을 확인하였습니다. 특히, 적응형 분류기 할당 전략을 통해 약한 모달리티의 성능이 크게 향상되었으며, 이는 전체적인 정확도 향상으로 이어졌습니다. 이 논문은 MML 분야에서 기존 방법들과 비교했을 때 월등한 성과를 달성하는 것을 보여주고 있습니다.



### Sketch & Paint: Stroke-by-Stroke Evolution of Visual Artworks (https://arxiv.org/abs/2502.20119)
Comments:
          ECCV 2024 Workshop: AI for Visual Arts Workshop and Challenges (AI4VA)

- **What's New**: 본 논문에서는 예술 작품의 스트로크 진화를 근접 기반 클러스터링 메커니즘을 통해 근사화하는 새로운 방법을 소개합니다. 이는 픽셀 이미지를 파라미터 곡선을 통해 벡터 이미지로 변환한 후, 추출된 스트로크의 순서를 결정하는 클러스터링 접근 방식을 탐색합니다. 제안된 알고리즘은 알려지지 않은 예술 작품의 스트로크 시퀀스를 추론할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 우리는 SVG(Scalable Vector Graphics)를 활용하여 스트로크별 데이터를 추출하고, 생성된 스트로크 시퀀스는 상호작용적인 스케치 및 페인팅 모델 구축뿐만 아니라 예술 완성, 조작, 생성 및 검색과 같은 다운스트림 작업에도 유용합니다. 본 연구는 제스트탈트(Gestalt) 법칙에서 영감을 받은 지각적 군집화를 통해 레이블이 없는 스트로크 데이터의 순서 추론을 모방하여 예술 작품의 발전 과정을 표현합니다. 또한, Bézier 곡선과 같은 파라미터화된 곡선으로 입력 이미지를 표현하는 방식을 채택하였습니다.

- **Performance Highlights**: 우리의 제안 방법은 다양한 입력 이미지 유형을 효과적으로 처리할 수 있는 강인성을 보여줍니다. 또한 WikiArt 데이터를 사용하여 본 방법의 성능을 평가했으며, 실제 스트로크 시퀀스가 어떻게 생성될 수 있는지를 정성적으로 입증하였습니다. 복잡한 이미지 다수에 대해 스트로크 진화를 처리하는 데 있어 저희 방법은 처음으로 시도이며, 다양한 유형의 시각적 데이터에 대한 일반화 가능성을 입증했습니다.



### MITracker: Multi-View Integration for Visual Object Tracking (https://arxiv.org/abs/2502.20111)
- **What's New**: 새롭게 소개하는 DVTrack 데이터셋은 234K개의 프레임으로 구성되어 있으며, 27개의 다양한 객체와 3-4개의 카메라에서 캡처된 경우를 포함합니다. 이 데이터셋은 occlusion(가림막) 및 deformation(변형)과 같은 9가지 도전 과제를 포함하고 있어, MVOT(다중 뷰 객체 추적) 모델의 교육 및 평가를 위한 최초의 종합적인 벤치마크 역할을 합니다.

- **Technical Details**: MITracker는 새로운 다중 뷰 통합 추적 알고리즘으로, 2D 이미지 기능을 3D 기능 볼륨으로 변환하고 이것을 bird's eye view(BEV) 평면으로 압축하여 서로 다른 뷰 간의 정보 융합을 원활하게 합니다. 이 방법은 Vision Transformer(ViT)를 이용해 특정 뷰의 입력 영상에서 목표 객체의 피쳐를 추출하며, 추출된 피쳐를 3D 기능 볼륨으로 통합하여 강화된 주의(attention) 메커니즘으로 추적 결과를 개선합니다.

- **Performance Highlights**: MITracker는 MVTrack과 GMTD 데이터셋에서 기존 방법들을 능가하여 최첨단 성능(state-of-the-art)을 달성합니다. 특히, MITracker는 도전적인 시나리오에서 목표 손실을 줄이기 위해 회복률을 56.7%에서 79.2%로 증가시키는 성과를 보였습니다. 이러한 결과는 MITracker가 occlusion과 같은 어려운 상황에서도 안정적인 추적 결과를 유지할 수 있다는 점에서 큰 의미를 가집니다.



### UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler (https://arxiv.org/abs/2502.20110)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2403.18913

- **What's New**: 이번 논문에서는 UniDepthV2라는 새로운 모델을 제안하여 단일 이미지로부터 메트릭 3D 장면을 재구성하는 Monocular Metric Depth Estimation (MMDE) 기술의 새로운 표준을 설정합니다. 이 모델은 기존 MMDE의 틀을 벗어나 추가적인 정보 없이 입력 이미지에서 메트릭 3D 포인트를 직접 예측합니다. 이를 통해 다양한 도메인에서 높은 일반화 성능을 발휘하며, 실제 환경에서의 적용 가능성을 크게 향상시킵니다.

- **Technical Details**: UniDepthV2는 자기 주도형 카메라 모듈을 구현하여 깊이 특성을 조정할 수 있는 밀집 카메라 표현을 예측합니다. 이 과정에서 가상의 구형 출력 표현(pseudo-spherical output representation)을 도입해 카메라와 깊이 표현을 분리하였으며, 기하학적 불변 손실(geometric invariance loss)을 통해 카메라 조건에 따른 깊이 특징의 강건성을 증진시켰습니다. 또한, 에지 기반 손실(edge-guided loss)을 통해 깊이 출력의 가장자리에서 더욱 선명하고 정밀한 로컬 구조를 유지하도록 설계되었습니다.

- **Performance Highlights**: 무엇보다도, UniDepthV2는 10개의 깊이 데이터셋에 대한 제로샷 평가에서 탁월한 성능을 보임으로써 MMDE 분야의 새로운 기준을 설정합니다. 또한, KITTI 깊이 예측 벤치마크에서 공개된 방법들 중 첫 번째로 순위에 오르며 우수한 일반화 능력을 입증했습니다. 이 모델은 downstream tasks에서 신뢰할 수 있는 깊이 입력을 요구하는 제어(controls)와 같은 응용 프로그램에서 활용될 수 있습니다.



### VDT-Auto: End-to-end Autonomous Driving with VLM-Guided Diffusion Transformers (https://arxiv.org/abs/2502.20108)
Comments:
          Submitted paper

- **What's New**: VDT-Auto라는 새로운 파이프라인이 제안되었으며, 이는 BEV(위성 시점) 인코더와 VLM(Visual Language Model)을 활용하여 동적 환경에서 자율주행 차량의 상태-행동 매핑을 개선합니다. 이 시스템은 기하학적 및 맥락적으로 환경을 해석하여 자율주행 차량의 최적의 행동을 생성합니다. 또한, VDT-Auto의 성능은 nuScenes 오픈 루프 평가에서 평균적으로 0.52m의 L2 오류와 21%의 충돌율로 나타났습니다.

- **Technical Details**: VDT-Auto는 BEV 인코더와 분산 변환기(diffusion transformers)를 사용하여 차량과 환경의 현재 상태를 최적의 경로로 변환합니다. 입력된 주위 카메라의 이미지를 BEV feature로 인코딩하여 기하학적 정보를 추출하며, VLM은 맥락적 정보로서 기능합니다. 이러한 처리된 정보는 분산 프로세스를 조건짓는 데 사용되어, 우리 모델은 차량의 행동을 예측합니다.

- **Performance Highlights**: VDT-Auto는 다양한 동적 환경에서 높은 일반화를 보여주었으며, 실제 주행 데이터셋에서도 유망한 성능을 발휘했습니다. 매개변수 고정화된 VLM을 통해 예측의 정확성을 높이는 동시에, 노이즈가 추가된 경로를 샘플링하여 역추적 과정에 활용함으로써 성능을 극대화합니다. 향후 이 코드와 데이터셋은 공개될 예정입니다.



### New Dataset and Methods for Fine-Grained Compositional Referring Expression Comprehension via Specialist-MLLM Collaboration (https://arxiv.org/abs/2502.20104)
Comments:
          TPAMI under review

- **What's New**: 이 논문에서는 Referring Expression Comprehension (REC) 분야를 발전시키기 위해 새로운 데이터셋 FineCops-Ref를 도입합니다. 이 데이터셋은 두 가지 주요 특징이 있습니다. 첫째, 제어된 난이도 수준을 갖추고 있어 객체 범주, 속성 및 관계 간의 세부적인 추론을 요구합니다. 둘째, 존재하지 않는 목표를 거부하는 모델의 능력을 테스트하기 위해 세심하게 편집된 네거티브(negative) 텍스트와 이미지를 포함하여 기존 데이터셋에서 자주 간과되는 문제를 해결합니다.

- **Technical Details**: FineCops-Ref 데이터셋은 복합적인 관계 및 속성 기반의 세부 사항을 고려하여 난이도를 조정합니다. 난이도는 올라가는 요구에 따라 분류되며, 복잡한 개체 식별을 위한 두 가지 협력 전략인 Slow-Fast Adaptation (SFA)과 Candidate Region Selection (CRS)을 제안합니다. SFA는 간단한 작업을 Specialist Models에게, 복잡한 작업을 MLLMs에게 할당하는 적응형 경로 지정 메커니즘을 사용합니다. CRS는 Specialist Models로부터 여러 객체 후보를 생성하고 MLLMs의 추리 능력을 활용하여 올바른 대상을 선택하게 합니다.

- **Performance Highlights**: FineCops-Ref를 기반으로 한 실험은 제안된 Specialist-MLLM 협력 전략이 기존 모델과 MLLMs의 성능을 크게 향상시켰음을 보여줍니다. SFA 전략은 위치 정확도와 효율성 간의 균형을 이루며, CRS 전략은 두 모델의 성능을 크게 향상시키는 결과를 가져옵니다. 이 연구는 실제 세계의 복잡한 작업을 해결하는 데 있어 기존 도구를 전략적으로 결합하여 최대 효과를 달성하는 방법에 대한 귀중한 통찰력을 제공하고자 합니다.



### WalnutData: A UAV Remote Sensing Dataset of Green Walnuts and Model Evaluation (https://arxiv.org/abs/2502.20092)
- **What's New**: 이 연구에서는 UAV(무인 항공기) 기술을 통해 수집한 첫 번째 대규모 녹색 호두(New walnut) 객체 탐지 데이터셋인 WalnutData를 소개합니다. 이 데이터셋은 30,240개의 RGB 이미지와 706,208개의 주석 인스턴스를 포함하고 있으며, 특히 조명과 차폐 조건을 세분화하여 네 가지 환경 상태로 나뉘었습니다. 이를 통해 스마트 농업에서의 알고리즘 개발을 지원하고, 녹색 호두 탐지의 과학적 및 공학적 가치를 강조합니다.

- **Technical Details**: WalnutData 데이터셋은 다양한 조명 조건과 차폐 문제가 있는 녹색 호두의 탐지를 용이하게 하기 위해 설계되었습니다. 1,024×1,024 픽셀 해상도의 이미지로 구성되어 있으며, A1(정면 조명), A2(역광), B1(정면 조명 차폐), B2(역광 차폐)의 네 가지 환경 상태로 분류됩니다. 이러한 세분화는 로봇 경로 계획 및 장애물 회피 결정에 필요한 딥러닝 기반 알고리즘의 성능 향상을 위한 기반으로 기능합니다.

- **Performance Highlights**: 공식 연구의 일환으로 WalnutData를 사용하여 DETR, YOLO 시리즈, Fast R-CNN 및 Faster R-CNN과 같은 여러 주류 탐지 알고리즘에 대한 벤치마크 테스트를 수행하였습니다. 이러한 테스트 결과는 향후 알고리즘 디자인의 기준선을 제공합니다. WalnutData는 농업 분야에서의 객체 탐지 연구에 필수적인 데이터 기반 자동화 관리 방법의 발전을 도울 것으로 예상됩니다.



### OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels (https://arxiv.org/abs/2502.20087)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 논문에서는 인간의 시각 시스템에서 영감을 받아 ‘OverLoCK’라는 새로운 Pure ConvNet 비전 백본을 제안합니다. 이 모델은 깊이 단계 분해 전략(Deep-stage Decomposition Strategy, DDS)을 통해 동적인 상향식(top-down) 맥락 지도를 생성하여 정교한 시멘틱(context) 표현을 중첩(fuse)하는 방법으로 개발되었습니다. 또한, context-mixing dynamic convolution (ContMix)을 활용하여 장기 의존성을 효과적으로 모델링합니다.

- **Technical Details**: OverLoCK 모델은 Base-Net, Overview-Net, Focus-Net의 세 가지 하위 네트워크로 구성되어 있습니다. Base-Net은 저수준 및 중간 수준 정보를 인코딩하고, Overview-Net은 빠르게 전반적인 맥락 표현을 확보합니다. 이후 Focus-Net은 이러한 정보를 바탕으로 더 정확한 고수준 표현을 획득합니다. ContMix는 입력 특징 맵의 각 토큰에 의해 생성된 어피니티 맵을 통해 동적인 합성곱 커널을 형성하여 특성 작용을 극대화합니다.

- **Performance Highlights**: OverLoCK는 여러 지표에서 기존 ConvNet 모델들보다 탁월한 성능을 보입니다. 예를 들어, ImageNet-1K 데이터셋에서 Top-1 정확도 84.2%를 달성하며, UniRepLKNet-T를 1% 초과했습니다. 객체 탐지 및 시맨틱 분할 작업에서도 우수한 성과를 보여 모가넷(MogaNet-B) 및 UniRepLKNet-T와 비교하여 각각 1.2% 및 1.7% 향상된 mIoU를 기록했습니다.



### SegLocNet: Multimodal Localization Network for Autonomous Driving via Bird's-Eye-View Segmentation (https://arxiv.org/abs/2502.20077)
- **What's New**: 본 논문에서는 GNSS(전 세계 항법 위성 시스템)에 의존하지 않고도 정확한 현지화를 달성하는 SegLocNet이라는 다중 모달 GNSS-프리 로컬라이제이션 네트워크를 제안합니다. SegLocNet은 다중 센서 입력을 통해 수집한 데이터를 활용하여 BEV(조감도) 의미 분할을 통해 차량의 자체 위치(ego pose)를 추정합니다. 이 방법은 회귀 기반의 위치 추정이 가지는 한계를 피하고 높은 해석 가능성과 일반화 능력을 유지합니다.

- **Technical Details**: SegLocNet은 서로 다른 시각의 이미지와 LiDAR 포인트 클라우드를 입력으로 받아 BEV 의미 분할 작업을 통해 환경의 정적 요소를 배웁니다. 이후, 센서 입력과 이전 맵을 일치시켜 차량의 자체 위치를 정확하게 추정합니다. 이 방법은 HD(고해상도) 및 SD(표준 해상도) 맵 모두에 쉽게 적용될 수 있으며, 네트워크 아키텍처의 수정 없이 통합된 맵 표현을 제공하여 성능을 극대화합니다.

- **Performance Highlights**: nuScenes 및 Argoverse 데이터 세트에 대한 광범위한 실험 결과, SegLocNet은 기존의 최신 기술 방법들을 능가하는 성능을 입증했습니다. 이 방법은 GNSS에 의존하지 않고도 도시 환경에서 차량의 자체 위치를 정확히 추정할 수 있으며, 강력한 일반화 능력을 유지합니다. 공개된 코드와 사전 훈련된 모델을 통해 연구 커뮤니티가 쉽게 접근할 수 있게 할 예정입니다.



### Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation (https://arxiv.org/abs/2502.20056)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이 논문에서는 방사선과 의사들의 업무 부담을 줄이기 위해 개선된 대조 학습(contrastive learning) 방법론을 기반으로 한 다중 관점(longitudinal) 데이터를 활용하여 흉부 X-ray 보고서 생성을 위한 모델, MLRG(Multi-view Longitudinal Report Generation)를 제안합니다. 기존의 방법론들이 단일 이미지에 의존하는 반면, 이 방법은 현재 다중 관점의 이미지를 통해 질병의 진행 상태를 분석하고 이를 바탕으로 보다 정확한 진단 정보를 제공합니다. 또한, 모델이 환자 특정의 이전 지식을 유연하게 처리할 수 있도록 하는 결측값 인코딩(Absence Encoding) 기법을 도입하였습니다.

- **Technical Details**: MLRG는 두 가지 단계로 구성되며, 첫 번째 단계에서는 방사선 보고서 내의 고유한 시공간 정보(spatiotemporal information)를 활용하여 시각 및 텍스트 표현의 사전 학습(pre-training)을 감독합니다. 다중 관점(longitudinal) 데이터와 그에 해당하는 방사선 보고서 간의 일치를 활용하여 시각적 및 텍스트 표현을 학습합니다. 두 번째 단계에서는 결측한 환자 정보(예: 이전 검사 결과) 처리 기술을 도입하여 모델이 유연하게 데이터의 유무에 따라 조정될 수 있도록 하여 생성된 보고서의 정확성을 향상시킵니다.

- **Performance Highlights**: MLRG는 MIMIC-CXR, MIMIC-ABN, 그리고 Two-view CXR 데이터셋을 대상으로 한 광범위한 실험에서 기존의 최신 방법들과 비교하여 우수한 성능을 입증하였습니다. 특히 MIMIC-CXR에서 2.3%의 BLEU-4 향상을, MIMIC-ABN에서 5.5%의 F1 스코어 향상을, Two-view CXR에서는 2.7%의 F1 RadGraph 개선을 달성했습니다. 이런 결과는 MLRG가 임상적으로 정확한 보고서를 생성하는 데 있어 효과적인 방법임을 시사합니다.



### 3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds (https://arxiv.org/abs/2502.20041)
Comments:
          ICLR

- **What's New**: 본 논문은 고정된 레이블 세트에 의존했던 기존의 3D affordance detection 방식을 'Instruction Reasoning Affordance Segmentation' (IRAS)로 변환하는 새로운 접근법을 제안합니다. 이 새로운 작업은 자연어 쿼리에 기반하여 affordance 마스크 영역을 출력하도록 설계되어, 다이나믹한 환경에서도 유연성과 일반화를 제공할 수 있습니다. 저자들은 3D-AffordanceLLM (3D-ADLLM)이라는 프레임워크를 도입하여, 대형 언어 모델(LLMs)을 통해 자연어 지침에 따른 affordance 인식을 가능하게 합니다.

- **Technical Details**: 3D-ADLLM은 특정 설계된 디코더를 통해 affordance 마스크를 생성하면서 대형 언어 모델의 추론 능력을 활용합니다. 이 모델은 <AFF>라는 추가 토큰을 생성하여, 해당 임베딩을 통해 세그멘테이션 마스크를 디코딩 합니다. 또한, 3D affordance 데이터셋의 부족을 해결하기 위해 사전 훈련 단계(Referring Object Part Segmentation, ROPS)를 활용하여 일반적인 인식 및 세그멘테이션 기술을 학습합니다. 이후 IRAS 작업으로 모델을 미세 조정하여 상황 인지 추론 능력을 강화합니다.

- **Performance Highlights**: 3D-ADLLM은 대형 언어 모델의 풍부한 세계 지식과 인간-객체 상호 작용 추론 능력을 활용하여, open-vocabulary affordance detection 작업에서 약 8%의 mIoU 향상을 달성하였습니다. 이 연구는 고정된 레이블 세트를 사용했던 기존 방식의 한계를 극복하고, 복잡한 자연어 지침에 대한 이해 및 추론 능력을 개선시켰습니다. 결과적으로 이러한 새로운 접근법은 다양한 로봇 작업에서의 3D affordance 인식에 있어 매우 중요한 기여를 하게 됩니다.



### A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization (https://arxiv.org/abs/2502.20036)
Comments:
          To be published in 2025 International Conference on 3D Vision (3DV)

- **What's New**: 이 논문은 절대적으로 시각적 설명자(visual descriptors)에 의존하지 않고도 2D-3D 키포인트 매칭을 수행할 수 있는 Angle-Annular Graph Neural Network (A2-GNN)을 도입합니다. 기존의 방법들이 직면하고 있는 저장 요구사항과 개인 정보 보호 문제를 해결하기 위해 이 새로운 접근 방식은 간단하면서도 효율적으로 로컬 구조를 추출합니다. 이 연구는 2D 쿼리 이미지와 3D 모델 간의 매칭을 개선하여 뛰어난 정확성을 달성하였으며, 코드 또한 공개할 예정입니다.

- **Technical Details**: A2-GNN은 2D-3D 매칭을 위한 새로운 로컬 그래프 이웃 집계 방법을 이용하여 기하학적 정보(geometric information)를 함께 포함시킵니다. 이 모델은 쿼리 이미지의 희소 점들과 포인트 클라우드(point cloud)로부터 입력을 받아 연결된 이웃 점들 간의 거리와 각도 정보를 클러스터링하여 지역 구조(local structures)를 캡처합니다. 아울러, 이전의 방식에서 사용되던 max-pooling 대신 인접점의 구조적 정보를 보존하는 새로운 접근 방식을 적용하여 기하학적 표현을 효과적으로 향상시킵니다.

- **Performance Highlights**: 성능 평가에서는 A2-GNN이 시각적 설명자를 사용하지 않는 방법들 중에서 가장 높은 정확도를 기록했습니다. 이 메서드는 상대적으로 낮은 계산 오버헤드(computational overhead)로도 뛰어난 성능을 보여주어, 실제 시각적 로컬라이제이션 과제에 유리함을 입증했습니다. 이러한 성과는 기존 방법들에 비해 눈에 띄는 개선을 이루었고, 특히 계산 효율성 측면에서 비할 데 없는 장점을 보여주었습니다.



### AsymLoRA: Harmonizing Data Conflicts and Commonalities in MLLMs (https://arxiv.org/abs/2502.20035)
- **What's New**: 이번 연구에서 제안하는 AsymLoRA는 다중 모달 데이터셋에서의 지식 모듈화(knowledge modularization)와 크로스 모달 조정(cross-modal coordination)을 통합하는 혁신적인 파라미터 효율적 튜닝 프레임워크입니다. AsymLoRA는 모달리티(모드) 특정 최적화 목표 간의 충돌을 해결하고, 각각의 독립적인 작업 적응 경로를 유지할 수 있는 저차원 프로젝션(matrices B)을 도입합니다. 이와 동시에, 공유 프로젝션(matrices A)을 통해 크로스 모달 공통성을 집약하여 향상된 성능과 시스템 효율성을 달성합니다.

- **Technical Details**: AsymLoRA는 MLLM(Multimodal Large Language Model)의 지시 조정(instruction fine-tuning)을 위해 설계된 비대칭(asymmetric) LoRA 아키텍처입니다. 기존의 LoRA와 달리, AsymLoRA는 공통 지식을 캡처하기 위한 공유 A 행렬과 독립적인 작업 적응을 위한 B 행렬을 도입하여 다중 작업 데이터셋 간의 공통성과 충돌을 효과적으로 균형 잡습니다. 이 과정에서 AsymLoRA는 Mixture-of-Experts (MoE) 방식으로 동적으로 특정 작업의 B 행렬을 선택하여 효율성을 개선합니다.

- **Performance Highlights**: AsymLoRA는 다양한 벤치마크를 통해 vanilla LoRA 및 LoRA-MoE보다 일관되게 우수한 성능을 발휘하는 것으로 나타났습니다. 이 연구에서는 미션 데이터셋 간의 충돌을 효과적으로 완화하고, 공유 지식을 활용하여 표준 LoRA 및 LoRA-MoE와 비교해 성능 및 효율성을 동시에 달성함을 입증했습니다. AsymLoRA의 설계는 다중 모달 교육에서의 성능 저하를 개선하는 데 기여할 것으로 기대됩니다.



### Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScor (https://arxiv.org/abs/2502.20034)
Comments:
          4 pages

- **What's New**: 최근 대형 비전-언어 모델(Large Vision-Language Models, LVLMs)의 성능이 여러 분야에서 두각을 나타내고 있습니다. 그러나 이러한 모델은 객체 환각(object hallucination) 문제에 직면하고 있습니다. 본 연구는 LVLM의 환각 원인이 비전 인코더(vision encoder)의 한정된 표현 능력에 있지 않다는 것을 규명하고, 새로운 평가 지표인 세분화된 CLIPScore(Fine-grained CLIPScore, F-CLIPScore)를 제안합니다.

- **Technical Details**: F-CLIPScore는 명사구 수준에서 텍스트 임베딩을 포함하여 객체 수준의 세부 사항을 향상시킵니다. 이 지표는 CLIP 모델의 시각-언어(Vision-Language) 파라미터들과 함께 사용되어 비전 인코더의 표현 능력을 효율적으로 평가합니다. 연구 결과, F-CLIPScore는 기존의 CLIPScore보다 39.6% 더 높은 정확도로 OHD-Caps 벤치마크에서 성능을 발휘하며, 추가적인 훈련 없이도 객체 환각을 효과적으로 감지할 수 있음을 보여줍니다.

- **Performance Highlights**: F-CLIPScore를 사용하여 사전 훈련 데이터 선별을 수행할 경우 모델의 객체 환각이 줄어드는 것을 확인했습니다. F-CLIPScore를 활용한 LVLM 사전 훈련 과정에서는 데이터 선별을 통해 기준 대비 POPE 정확도가 4.9% 향상되었습니다. 이러한 연구 결과는 비전 인코더의 제한된 능력이 객체 환각의 주요 원인이 아님을 암시하며, 더 효율적인 훈련 방식을 제시합니다.



### Joint Fusion and Encoding: Advancing Multimodal Retrieval from the Ground Up (https://arxiv.org/abs/2502.20008)
- **What's New**: 이 논문에서는 정보 검색의 현재 한계를 극복하기 위해, 시각적(visual) 및 텍스트적(textual) 단서를 통합하는 새로운 통합 검색 프레임워크인 Joint Fusion Encoder (JFE)를 제안합니다. 기존의 많은 late-fusion 아키텍처는 독립적인 모달리티의 단순한 결합만을 고려했으나, 이것이 복잡한 쿼리에 필요한 세밀함을 포착하는 데 한계를 보입니다. 본 연구는 모달리티의 조기 융합(early fusion)을 통해 더욱 정교한 다중 모달 정보 검색을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 대규모 사전 훈련된 대형 언어 모델(MLLM)을 기반으로 하며, 이를 효과적인 인코더로 변환하여 정보를 처리합니다. 접근 방식은 두 가지 주요 단계로 구성되는데, 첫 번째 단계는 포스트 훈련 적응(post-training adaptation)으로 MLLM을 인코더로 조정하는 것이고, 두 번째 단계는 다양한 크로스 모달 검색 작업을 위한 지침 학습(instruction tuning)입니다. 이를 통해 각 모달리티의 특성을 보존하면서도, 복합적인 모달 쿼리에 대한 이해를 강화합니다.

- **Performance Highlights**: JFE는 두 가지 주요 카테고리의 실험을 통해 기존의 두 타워 두 아키텍처 기반 방법들에 비해 우수한 성능을 보였습니다. 특히 다중 모달 쿼리 및 조건부 정보가 포함된 작업에서 두드러진 성과를 나타내며, 이는 다중 모달 정보가 매초 생성되는 현대의 많은 응용 프로그램에 매우 주목받고 있는 상황을 고려할 때, 보다 효과적인 정보 검색을 위한 promising한 방향을 제시합니다.



### Low-rank tensor completion via a novel minimax $p$-th order concave penalty function (https://arxiv.org/abs/2502.19979)
Comments:
          32 pages,12 figures

- **What's New**: 본 논문은 저순위 텐서 보완 (Low-rank tensor completion, LRTC) 문제를 해결하기 위해 새롭게 제안된 최소극 순서의 오목 패널티(Minimax p-th order concave penalty, MPCP) 함수를 소개합니다. 기존의 최솟값 이완 기법들이 작은 특이값(singular values)을 충분히 처리하지 못하는 한계를 극복하는 데 주력하고 있습니다. 또한, MPCP 기반의 LRTC 모델을 제안하고, 이 모델의 수렴 이론적 보장을 제공합니다.

- **Technical Details**: MPCP 함수는 큰 특이값을 보호하는 기존 MCP 함수의 특성을 유지하면서도, 작고 연속적인 특이값에 대해 강한 패널티를 부여하는 것이 특징입니다. 이와 함께, MPCP 함수의 최적화를 위해 프로시말 연산자를 도출하였고, 이를 활용한 LRTC 모델과 문제 해결 알고리즘(Alternating Direction Method of Multipliers, ADMM)을 개발하였습니다. 마지막으로, 제안한 방법의 성능을 다차원 데이터셋에서 평가하고 이론적 분석을 통해 수치적 안정성을 보장합니다.

- **Performance Highlights**: 다양한 실 데이터셋에 대한 실험 결과, 제안된 MPCP 방법이 최신 방법들보다 우수한 성능을 보여주었습니다. 특히, MCP 기반 변형들과의 비교 분석에서도 MPCP 방법의 뛰어난 성능이 입증되어, 이론적 분석 결과를 뒷받침하였습니다. 결과적으로 MPCP 함수는 LRTC 문제를 해결하는 데 있어 더욱 효율적인 대안이 될 것으로 기대됩니다.



### Can Large Language Models Unveil the Mysteries? An Exploration of Their Ability to Unlock Information in Complex Scenarios (https://arxiv.org/abs/2502.19973)
Comments:
          11pages

- **What's New**: 본 논문에서는 복잡한 시나리오에서 여러 인지 입력을 통합하고 조합적 추론을 수행하는 능력을 탐색하기 위해 새로운 벤치마크인 CVQA(Clue-Visual Question Answering)와 CPVQA(Clue of Password-Visual Question Answering)를 소개합니다. CVQA는 시각적 이해 및 통합을 평가하기 위해 3가지 작업 유형을 포함하고, CPVQA는 시각적 데이터의 정확한 해석 및 적용을 중점적으로 다룹니다.

- **Technical Details**: CVQA는 서로 다른 장면에 위치한 불특정 시각적 개체를 결합하여 추론하는 Props Search, 특정 텍스트 개체와 불특정 시각적 개체 간의 상호작용을 포함하는 Props Usage, 그리고 지정된 시퀀스 유형으로 시각적 장면을 결합하여 단서를 추론하는 세 가지 유형의 작업을 포함합니다. CPVQA는 모든 시각 장면을 아우르는 조합적 추론을 평가하는 두 가지 유형의 작업을 포함하고 있습니다.

- **Performance Highlights**: 기존의 모델들은 조합적 추론 벤치마크에서 낮은 성능을 기록했습니다. 특히 최신 모델조차 CVQA에서 33.04%의 정확도에 불과하며, CPVQA에서는 7.38%로 감소합니다. 본 연구의 방법론은 이러한 성능을 개선하여 CVQA와 CPVQA에서 각각 22.17% 및 9.40%의 성능 향상을 보였습니다.



### ReCon: Enhancing True Correspondence Discrimination through Relation Consistency for Robust Noisy Correspondence Learning (https://arxiv.org/abs/2502.19962)
Comments:
          10 pages, 4 figures, Accepted by CVPR2025

- **What's New**: 이번 연구에서는 multimodal 데이터셋에서 올바른 대응을 정확히 식별할 수 있는 새로운 Relation Consistency 학습 프레임워크인 ReCon을 제안합니다. 기존 기법은 객체 간의 유사도 일치에 중점을 두고 있지만, 모달리티 내의 관계 일관성을 간과함으로써 잘못된 긍정과 부정의 식별 위험이 존재했습니다. ReCon은 이러한 문제를 해결하기 위해 교차 모달과 인트라 모달 관계 일관성을 함께 다룹니다.

- **Technical Details**: ReCon은 교차 모달 관계 일관성을 통해 서로 다른 모달리티의 긍정 쌍의 유사도 점수를 최대화하고, 인트라 모달 관계 일관성을 통해 모달리티 내의 객체의 관계를 설명하는 관계 행렬의 거리를 최소화합니다. 이러한 이중 제약 조건은 노이즈가 있는 훈련 데이터를 나누고, 각 전략으로 훈련하여 강력한 교차 모달 검색을 달성하는 데 기여합니다.

- **Performance Highlights**: Flickr30K, MS-COCO, Conceptual Captions를 포함한 세 개의 널리 사용되는 벤치마크 데이터셋에서 광범위한 실험을 통해 ReCon의 효과성과 우수성을 입증했습니다. 성능 측면에서 기존의 최첨단(SOTA) 방법들과 비교했을 때, ReCon은 소음 대응 문제를 효과적으로 완화하고, 진정한 대응의 식별력을 크게 향상시킵니다.



### ChatReID: Open-ended Interactive Person Retrieval via Hierarchical Progressive Tuning for Vision Language Models (https://arxiv.org/abs/2502.19958)
- **What's New**: 이번 논문에서는 ChatReID라는 새로운 인물 재식별(person Re-ID) 프레임워크를 제안합니다. 이 프레임워크는 Hierarchical Progressive Tuning (HPT) 전략을 도입하여, 모델이 보행자 인식을 더욱 정교하게 수행할 수 있도록 단계적으로 개선합니다. 실험 결과, 향상된 유연성과 사용자 친화성을 보여주며, SOTA (State of the Art) 방법을 초과하는 성능을 달성하였습니다.

- **Technical Details**: ChatReID는 세 가지 훈련 단계를 포함하는 HPT 전략을 채택하는데, 첫 번째 단계에서는 보행자의 정체성 특징을 이해하기 위한 점진적 사전 학습을 수행합니다. 두 번째 단계는 다중 작업 공동 훈련으로 이미지를 기반으로 한 인물 정체성 매칭 능력을 개발합니다. 마지막 단계에서는 세 가지 특정 응용 시나리오를 위한 지침 세부 조정을 통해 실용성을 높입니다.

- **Performance Highlights**: 광범위한 실험을 통해 ChatReID는 네 가지 다양한 Re-ID 설정에서 열 개의 벤치마크를 통해 SOTA 성과를 달성했습니다. 모델은 기존의 LVLM 기반 접근 방식보다 정교한 인물 차별화 능력을 갖추고 있으며, 다양한 입력 모달리티를 융합할 수 있는 유연성을 제공합니다. 이는 실제 응용 프로그램에 보다 적합한 고급 인물 재식별 시스템을 제공함을 의미합니다.



### RUBIK: A Structured Benchmark for Image Matching across Geometric Challenges (https://arxiv.org/abs/2502.19955)
- **What's New**: 이번 논문에서는 기존의 벤치마크가 제공하는 한계점을 극복하기 위해 RUBIK이라는 새로운 벤치마크를 소개합니다. 이는 다양한 기하학적 도전 어려움에 걸쳐 이미지 매칭 방법을 체계적으로 평가하는 시스템입니다. RUBIK는 세 가지 기준 - overlap, scale ratio, viewpoint angle -을 이용하여 nuScenes의 16.5K 이미지 쌍을 33개 난이도 수준으로 조직합니다.

- **Technical Details**: 저자들은 14가지 방법을 포괄적으로 평가하였고, 최근의 detector-free 접근 방식이 가장 높은 성능(>47% 성공률)을 기록한다고 밝혔습니다. 하지만 이들은 detector-based 방법에 비해 상당한 계산 오버헤드(150-600ms 대 40-70ms)가 발생합니다. 이는 기존의 방법론에서 높은 성능이 필수적임을 보여줍니다.

- **Performance Highlights**: 최고 성능을 나타내는 방법조차 54.8%의 쌍에서만 성공을 거두어, 특히 low overlap, large scale differences, extreme viewpoint changes와 같은 어려운 상황에서 개선의 여지가 크다는 점을 강조합니다. RUBIK 벤치마크는 공개적으로 제공될 예정입니다.



### Space Rotation with Basis Transformation for Training-free Test-Time Adaptation (https://arxiv.org/abs/2502.19946)
- **What's New**: 이번 연구에서는 테스트 시간 적응(test-time adaptation, TTA) 문제를 해결하기 위한 새로운 방법인 Space rOtation with Basis trAnsformation(SOBA)를 제안합니다. 기존 방법들이 높은 계산 자원(Computational Resources)을 요구하거나 원래 기능 공간(Feature Space)의 한계에 제약을 받는 문제를 해결하기 위해, 훈련 없이도 특징 공간을 회전(rotate)하여 새로운 표현으로 매핑(mapping)합니다. 이를 통해 클래스 간의 차이를 명확하게 하여 모델의 테스트 성능을 효과적으로 향상시킵니다.

- **Technical Details**: SOBA는 기초 변환(Basis Transformation) 기법을 활용하여 비선형적으로 분리할 수 있는 공간을 선형적으로 분리할 수 있는 새로운 공간으로 변환합니다. 이 과정에서 테스트 샘플에 대한 one-hot encoding을 생성하고, 잠재 레이블(Pseudo-label)을 할당하여 새로운 특징 공간을 구축합니다. 공분산 특이값 분해(Covariance Singular Value Decomposition)를 통해 다른 클래스 간의 차이를 보다 잘 반영하는 직교 기초(Orthogonal Basis)를 구축하여, 특징들이 새로운 공간에서 더 잘 구분되도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 기준(benchmarks)에서 기존 방법들보다 향상된 성능을 보여주었습니다. 특히, ImageNet 데이터셋을 기반으로 한 실험에서, SOBA는 SOTA 훈련 없는 메소드인 TDA보다 테스트 속도를 13.96% 개선했으며, TPT 기반의 방법에 비해서는 시간 소모가 2.15%에 불과했습니다. 이러한 성능 향상 덕분에 SOBA는 하류 작업의 분포 변화(distribution shifts)에 효과적으로 대응할 수 있는 잠재력을 가집니다.



### Image Referenced Sketch Colorization Based on Animation Creation Workflow (https://arxiv.org/abs/2502.19937)
- **What's New**: 본 논문에서는 스케치 색상화(Sketch colorization) 분야에서의 새로운 접근법을 제안합니다. 기존의 텍스트 기반 색상화 방법, 수작업 유도 방법 및 이미지 참조 방식에서 생기는 문제점들을 해결하기 위해, 실제 애니메이션 제작 과정에서 영감을 받은 확산 기반의 프레임워크를 도입하였습니다. 이 프레임워크는 스케치를 공간적 가이드로 사용하고, RGB 이미지를 색상 참조로 활용하여, 전방과 배경을 별도로 추출할 수 있도록 설계되었습니다.

- **Technical Details**: 이 프레임워크는 고유한 split cross-attention 메커니즘과 LoRA (Low-Rank Adaptation) 모듈을 사용합니다. 이들은 전경과 배경 영역을 분리하여 각각 키(key)와 값(value)에서의 임베딩을 조절하는 방식으로 학습됩니다. 이러한 디자인은 확산 모델이 전경과 배경 정보를 독립적으로 통합할 수 있게 하여, 서로 간섭을 방지하고 공간적 아티팩트를 제거합니다.

- **Performance Highlights**: 다양한 정성적 및 정량적 실험을 통해 기존의 방법들보다 높은 품질의 아티팩트 없는 결과를 생성하는 데 성공하였습니다. 사용자 연구 결과에서도 아티스트들이 우리의 방법을 더 선호하는 것으로 나타났습니다. 논문에서 제안하는 방식은 애니메이션 제작 파이프라인에 매끄럽게 통합될 수 있으며, 최종적으로 사용자 경험을 향상시키는 데 기여합니다.



### Identity-preserving Distillation Sampling by Fixed-Point Iterator (https://arxiv.org/abs/2502.19930)
- **What's New**: 이 논문에서는 Identity-preserving Distillation Sampling (IDS)라는 새로운 방법론을 도입하여 기존의 Score Distillation Sampling (SDS)에서 발생하는 이미지 블러 현상을 해결합니다. IDS는 텍스트-조건부 점수에서 유도된 그라디언트의 오류를 보상하여 원본 이미지의 아이덴티티(정체성)를 보다 효과적으로 보존합니다. 이를 통해 이미지 편집 시 명확하고 일관된 결과를 생성할 수 있습니다. 또한, 새로운 정규화 기법인 고정점 반복 정규화(FPR)를 통해 과도한 변형을 방지합니다.

- **Technical Details**: 텍스트-이미지 확산 모델에 기반한 IDS는 원본 이미지에 대한 조건부 기대값을 보존하기 위한 고급 기법입니다. IDS는 SDS의 그라디언트가 잘못 정렬되었을 때 이를 수정하여 원본 이미지의 구조와 포즈를 유지하는 데 중점을 둡니다. 이 방법은 자기 교정(self-correction) 메커니즘을 통해 개선된 결과를 생성하며, 조건부 기대값을 유지하는 것이 핵심입니다. IDS 업데이트 과정에서는 정제된 소스 라텐트에서 추출한 가이드 노이즈를 사용함으로써 아이덴티티 보존을 보장합니다.

- **Performance Highlights**: 제안된 IDS 방법은 두 가지 작업에서 기존의 다른 방법들과 비교할 때 높은 평가 점수를 기록하며, 사용자 연구와 GPT 결과에서도 우수한 성능을 입증하였습니다. 원본 이미지와 대상 이미지 간의 구조적 일관성뿐만 아니라, IoU 및 배경 PSNR에서도 우수한 결과를 도출하여 소스의 아이덴티티가 잘 보존된 것을 확인할 수 있습니다. 이로써 IDS는 성능과 품질에서 기존의 방법을 상회하는 성과를 보여줍니다.



### GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors (https://arxiv.org/abs/2502.19896)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이 논문에서는 기존의 포인트 클라우드(completion) 방식을 개선하기 위해 GenPC라는 새로운 제로샷(zero-shot) 프레임워크를 도입했습니다. 이 프레임워크는 고품질(real-world) 스캔을 재구성하기 위해 명시적인 3D generative priors를 활용하며, 이는 기존의 데이터셋 제약을 극복하는 데 중요한 기여를 합니다. GenPC는 입력 포인트 클라우드를 이미지-투-3D 생성을 통해 보완하며, 기존 기법보다 더 빠른 추론 속도를 자랑합니다.

- **Technical Details**: GenPC는 깊이 이미지(depth images)를 사용하여 포인트 클라우드를 이미지-투-3D 생성 모델에 연결하는 Depth Prompting 모듈을 개발합니다. 이후 생성된 3D 형태가 입력된 포인트 클라우드와의 일치를 보장하기 위해 Geometric Preserving Fusion 모듈을 설계하였으며, 이를 통해 생성된 모양의 포즈(pose)와 크기를 조정합니다. 이러한 과정을 통해 원래의 기하학적 구조를 유지할 수 있습니다.

- **Performance Highlights**: 폭넓은 벤치마크 테스트를 통해 GenPC는 현재 사용되고 있는 다른 기술들에 비해 성능의 우수성을 입증하였으며, 실제 데이터셋에서도 일반화(generalization) 능력을 발휘합니다. 또한, 이 방법은 completion 시간을 크게 단축시키는 동시에 고품질의 결과를 산출합니다. GenPC는 포인트 클라우드 분야에서 혁신적인 진전을 이룬 방법으로 자리 잡고 있습니다.



### High-Fidelity Relightable Monocular Portrait Animation with Lighting-Controllable Video Diffusion Mod (https://arxiv.org/abs/2502.19894)
- **What's New**: 이 논문에서는 Lighting Controllable Video Diffusion model (LCVD)을 제안하여 고충실도의 재조명 가능한 초상화 애니메이션을 생성하는 방법을 소개합니다. 기존의 방법들은 본질적인 (intrinsic) 특성과 외부적인 (extrinsic) 특성을 분리하지 못해 재조명 기능을 활용하지 못했습니다. 본 연구는 이러한 특성을 전담 서브스페이스(subspace)에서 학습하여 조명, 포즈(pose), 표현을 세밀하게 조작할 수 있는 능력을 갖추었습니다.

- **Technical Details**: LCVD는 이미지에서 비디오로 변환하는 확산 모델의 특성을 활용하여, 3D 메쉬(3D mesh)와 음영 정보를 효과적으로 추출합니다. 이 과정에서 조명과 포즈 정보를 포함하는 음영 힌트를 외부 특성(subspace)으로 매핑하는 shading adapter와 내재적 특성(intrinsic feature space)을 위한 reference adapter가 사용됩니다. 이를 통해 생성되는 이미지는 사용자가 제공하는 조명 조건을 반영하여 다양한 변형을 구현할 수 있습니다.

- **Performance Highlights**: 광범위한 평가 결과, LCVD는 조명 현실감, 이미지 품질 및 비디오 일관성 측면에서 기존 최첨단 방법들보다 우수한 성능을 보여주었습니다. 본 연구는 재조명 가능한 초상화 애니메이션에 대한 새로운 기준을 제시하며, 사용자가 지정한 조명 조건을 기반으로 자연스러운 애니메이션을 생성할 수 있습니다. LCVD의 성능 향상은 조명 효과 및 이미지 품질 관련 메트릭에서도 뚜렷하게 나타났습니다.



### C-Drag: Chain-of-Thought Driven Motion Controller for Video Generation (https://arxiv.org/abs/2502.19868)
- **What's New**: 이번 연구에서는 C-Drag라는 새로운 Chain-of-Thought 기반의 동작 제어기를 제안합니다. 기존의 궤적 기반 접근 방식이 객체와 주변 간의 동적 상호작용을 무시하는 한계를 극복하기 위해, C-Drag는 객체 인식과 동적 상호작용 추론 단계를 도입합니다. 이는 다양한 객체의 위치와 분류 정보를 캡처하며, 이후 이 정보를 바탕으로 단계별로 동작 궤적을 생성하여 비디오를 합성합니다.

- **Technical Details**: C-Drag는 객체 인식 모듈과 Chain-of-Thought 기반의 동작 추론 모듈로 구성되어 있습니다. 객체 인식 모듈은 시각 언어 모델(Visual Language Model, VLM)을 사용하여 영상 내의 객체 정보를 정확하게 파악합니다. 동작 추론 모듈은 이 정보를 입력으로 받아 단계별로 동작 궤적을 추론하고, 이후 이 궤적을 디퓨전 모델(Diffusion Model)에 입력하여 비디오를 생성합니다.

- **Performance Highlights**: C-Drag는 VOI 데이터셋에서 MOC(Motion Quality) 점수 기준으로 약 35.5%의 성과 향상을 달성하며, 다수의 메트릭에서 뛰어난 성능을 보여줍니다. VOI 데이터셋은 72개의 비디오로 구성되어 있으며, 다양한 객체 상호작용 유형을 포함하여 비디오 생성 방법의 성능을 평가하기 위한 기반이 됩니다. 이를 통해 C-Drag는 고품질 비디오 생성에 대해 향상된 성과를 입증하였습니다.



### Striving for Faster and Better: A One-Layer Architecture with Auto Re-parameterization for Low-Light Image Enhancemen (https://arxiv.org/abs/2502.19867)
- **What's New**: 이번 연구에서는 저조도(low-light) 이미지 향상 기술에 대해 심층 분석하며, 시각적 품질(visual quality)과 계산 효율성(computational efficiency)의 경계를 탐구하는 데 중점을 둡니다. 재매개변수화(re-parameterization)라는 개념을 도입하여 단일 계층 네트워크의 매개변수 공간을 확장하고, 최적의 구조를 자동으로 발견하는 것을 목표로 합니다. 이를 통해 매우 효율적인 저조도 이미지 향상을 수행하면서도 뛰어난 시각적 품질을 유지할 수 있습니다.

- **Technical Details**: 연구자들은 NAS(Neural Architecture Search)와 재매개변수화 기술을 결합하여 구조적 단순성을 보장하고, 최적의 성능 향상을 극대화합니다. 이를 통해 단일 합성곱층(convolutional layer)으로도 저조도 이미지 향상을 효과적으로 수행할 수 있습니다. 실험 결과, 다양한 플랫폼(CPU, GPU, NPU, DSP)에서 성능이 검증되었으며, 기존의 최고 성능 방법에 비해 실행 시간이 30% 이상 짧다는 것을 보여줍니다.

- **Performance Highlights**: 논문에서 제안한 방법은 여러 최신 기법에 비해 시각적 품질과 효율성 모두에서 우위를 점하고 있음을 실험을 통해 입증합니다. 특히, 다양한 이미지 크기에 대해 여러 플랫폼에서의 실행 시간이 기존의 빠른 방법보다 30% 이상 개선되었습니다. 이로써 본 연구의 기술이 실제 적용 가능성을 더욱 높이는 데 기여하고 있음을 알 수 있습니다.



### LMHLD: A Large-scale Multi-source High-resolution Landslide Dataset for Landslide Detection based on Deep Learning (https://arxiv.org/abs/2502.19866)
- **What's New**: 이 논문에서는 전 세계에서 가장 흔한 자연 재해 중 하나인 산사태(landslide)에 대한 새로운 데이터셋, 즉 'Large-scale Multi-source High-resolution Landslide Dataset' (LMHLD)을 제안합니다. LMHLD는 다양한 위성 센서에서 원격 감지 이미지를 수집하여, 총 25,365개의 패치를 포함하고 있습니다. 이 데이터셋은 산사태 탐지(landslide detection)를 위한 강력한 기반을 제공하고 다양한 스케일에서의 학습 기능을 돕기 위해 설계되었습니다.

- **Technical Details**: LMHLD는 중국의 웬춘(2008), 브라질의 리우데자네이루(2011), 네팔의 곡카(2015) 등 일곱 개 지역에서 수집된 데이터로 구성됩니다. 이 데이터셋은 다양한 범위의 산사태를 포함할 수 있도록 여러 크기의 패치로 나누어져 있습니다. 또한, LMHLDpart라는 훈련 모듈이 다중 작업 학습(multi-task learning)에서의 재난 기억 상실(catastrophic forgetting) 문제를 완화합니다.

- **Performance Highlights**: LMHLD를 기반으로 훈련된 모델은 다른 데이터셋에서도 적용되어 강력한 일반화 능력을 증명했습니다. U-Net 계열의 7가지 딥 러닝 모델을 사용한 다섯 가지 데이터셋 품질 평가 실험은 LMHLD가 산사태 탐지의 벤치마크 데이터셋으로 성장할 가능성을 보여줍니다. 이 데이터셋은 공개 접근 가능하며, 산사태 예방 및 완화를 위한 귀중한 자원으로 활용될 수 있습니다.



### One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion (https://arxiv.org/abs/2502.19854)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구에서는 이미지 융합 분야에서 기존의 고급(high-level) 비주얼 태스크에 의존하지 않으면서 픽셀 수준(supervision)을 통해 효과적인 특성(interaction)을 구현하는 새로운 접근 방식을 제안합니다. 새로운 GIFNet 아키텍처는 저급(low-level) 디지털 포토그래피 융합 작업을 사용하여 작업 간 상호 작용을 보다 자연스럽고 효과적으로 합니다. 이는 다양한 융합 작업에 대한 높은 성능과 유연성을 보장합니다.

- **Technical Details**: GIFNet은 세 개의 브랜치(main task branch, auxiliary task branch, reconciliating branch)로 구성되며, 각 브랜치는 멀티모달 및 디지털 포토그래피 특성에 집중합니다. 이러한 구조는 다양한 작업 간의 상호 작용을 촉진하고, 공유 재구성 작업을 통해 보편적인 특성 표현(universal feature representation)을 배양합니다. 또한 크로스-퓨전 게이팅 메커니즘(cross-fusion gating mechanism)을 통해 각 태스크 전용 브랜치를 반복적으로 개선합니다.

- **Performance Highlights**: GIFNet은 데이터 도메인 간 간극을 최소화하는 RGB 기반 합성 데이터셋(joint dataset)을 활용하여, 기존의 고급 비전 태스크가 단일 융합 모델을 필요로 했던 것과 달리, 다양한 융합 시나리오에서 범용 적용 가능성을 넓혔습니다. 실험 결과, GIFNet은 고급 이미지 융합 방법보다 GFLOPs를 96% 초과 감소시키면서도 효과적인 성능을 달성했습니다. 또한, 단일 모달리티 입력(single-modality inputs)에서도 뛰어난 성능을 발휘하여 이미지 처리의 다재다능한 향상기를 제공합니다.



### One-for-More: Continual Diffusion Model for Anomaly Detection (https://arxiv.org/abs/2502.19848)
Comments:
          Accepted by CVPR2025

- **What's New**: 이번 연구에서는 지속적인 잡음 확산 모델(Continual Diffusion Model for Anomaly Detection, CDAD)을 제안하여 기존의 이미지-투-이미지 방식에서 발생하는 문제점을 해결합니다. 연구자들은 'catastrophic forgetting'과 'faithfulness hallucination' 문제를 완화하기 위해 기울기 투영(gradient projection) 기법을 활용합니다. 또한, 반복 단일값 분해(iterative singular value decomposition, iSVD) 방법을 도입하여 메모리 사용량을 줄이고, 이상 감지에 대한 조건 메커니즘을 강화하는 이상 마스크 네트워크(anomaly-masked network)도 제안하였습니다.

- **Technical Details**: CDAD는 기존의 쭉 늘어진 피처(matrix)를 통해 유의미한 표현을 계산하고, 이를 통해 기울기를 조정하여 이전 태스크에 대한 영향을 최소화합니다. 그러나, Markov 기반의 확산 과정이 메모리 오버헤드를 증가시키므로 iteratve SVD 방법을 통해 메모리 사용량을 최소화합니다. 이상 마스크 네트워크는 CNN을 사용해 입력 이미지를 인코딩하고, 전역 정보를 인식하기 위해 transformer 구조를 활용하여 이상징후의 특성을 마스킹합니다.

- **Performance Highlights**: CDAD는 MVTec와 VisA의 18개 실험 설정 중 17개의 설정에서 1위 성과를 달성하며, 지속적인 이상 탐지에서의 우수성을 입증했습니다. 이는 CDAD가 이상 영역 재구성에 더 집중할 수 있도록 하여 과적합(overfitting) 문제를 효과적으로 완화한다는 점에서 큰 의미가 있습니다.



### ProAPO: Progressively Automatic Prompt Optimization for Visual Classification (https://arxiv.org/abs/2502.19844)
Comments:
          Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025

- **What's New**: 본 논문에서는 최소한의 감독과 인간 개입 없이 미세 분류를 위한 시각적으로 차별적인 프롬프트(optimal class-specific prompts)를 찾는 새로운 접근법을 제안합니다. 이를 통해, LLM(대형 언어 모델)에서 발생할 수 있는 착각 문제를 해결하고 프롬프트의 품질을 높이고자 합니다. 진화 기반의 알고리즘을 통해 작업에 특화된 템플릿에서 클래스별 설명으로 프롬프트를 점진적으로 최적화하는 방법을 제시합니다.

- **Technical Details**: 제안된 ProAPO(Progressively Automatic Prompt Optimization) 알고리즘은 프롬프트 생성 비용을 절감하기 위해 편집 기반 및 진화 기반 작업을 사용하여 다양한 후보 프롬프트를 생성합니다. 초기 단계에서 LLM을 통해 프롬프트 라이브러리를 생성한 후 이를 활용하여 각 반복에서 후보 프롬프트를 평가하고 선택합니다. 또한, 엔트로피 제약이 추가된 피트니스 점수를 사용하여 과적합 문제를 완화하고 각 반복에서 최적의 프롬프트를 반환합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 13개의 데이터셋에서 기존의 텍스트 기반 프롬프트 방법과 LLM 생성 방법보다 우수한 성능을 보였습니다. 제안된 프롬프트는 어댑터 기반 방법을 개선하고 다양한 백본(backbone) 간에 효과적으로 전이되는 것을 보여주었습니다. 이러한 결과는 ProAPO 알고리즘이 원샷(one-shot) 설정에서 기존 방법들의 한계를 극복할 수 있음을 입증합니다.



### CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation (https://arxiv.org/abs/2502.19842)
Comments:
          Accepted at CVPR 2025

- **What's New**: 본 연구는 CLIP(Contrastive Language-Image Pre-training) 모델의 다중 객체 환경에서의 한계를 분석하며, ComCO라는 전문 데이터셋을 제안합니다. 이는 모델의 성능을 다양한 다중 객체 시나리오에서 정밀하게 평가하기 위한 것으로, CLIP의 인코더의 편향을 정량화하고 그 기원을 추적합니다. 추가로, 이미지와 텍스트 매칭 실험을 통해 모델의 불안정성을 강조하고, 프롬프트 순서가 생성된 이미지 내의 객체 중요도에 미치는 영향을 분석합니다.

- **Technical Details**: 연구에서 ComCO 데이터셋은 Blender 소프트웨어를 활용하여 2개에서 5개 객체로 구성된 이미지와 해당 객체를 설명하는 캡션을 조합하여 생성되었습니다. 데이터셋의 목적은 CLIP의 텍스트 및 이미지 인코더의 성능을 별도로 평가하기 위한 것이며, 다양한 구성에 대한 실험을 통해 편향의 영향을 분석했습니다. 실험 세트는 검색 기반 실험과 분류 기반 실험으로 나뉘며, 각 결과를 COCO 데이터셋을 통해 검증했습니다.

- **Performance Highlights**: CLIP 모델의 텍스트 인코더는 첫 번째 언급된 객체에 대한 편향을 보였고, 이미지 인코더는 더 큰 객체에 대한 선호도를 나타냈습니다. 이러한 편향은 다중 객체 작업에서 성능 저하를 초래하며, ComCO 및 COCO 데이터셋에서 이미지-텍스트 매칭 정확도가 현저히 감소하는 결과를 가져왔습니다. 결론적으로, 연구는 CLIP 모델의 편향이 다중 객체 시나리오에서 성능을 크게 저하시킬 수 있음을 강조하고 있으며, 이를 해결하기 위한 이론적 기초를 제공합니다.



### Analyzing CLIP's Performance Limitations in Multi-Object Scenarios: A Controlled High-Resolution Study (https://arxiv.org/abs/2502.19828)
Comments:
          Accepted at ECCV 2024 Workshop EVAL-FoMo

- **What's New**: 이번 연구에서는 Contrastive Language-Image Pre-training (CLIP) 모델이 다중 객체(multi-object) 시나리오에서 나타내는 성능 한계를 종합적으로 분석하였습니다. 우리는 SimCO와 CompCO라는 두 개의 맞춤형 데이터셋을 소개하고, 이를 통해 CLIP의 이미지 및 텍스트 인코더의 편향(bias)을 평가했습니다. 연구 결과, 이미지 인코더는 더 큰 객체를 우선시하고, 텍스트 인코더는 설명에서 처음 언급된 객체에 더 많이 치우치는 경향이 있음을 발견했습니다.

- **Technical Details**: 연구에서는 다중 객체 환경에서 CLIP의 성능을 평가하기 위해 SimCO와 CompCO 데이터셋을 만들었습니다. SimCO는 기하학적 형태의 객체 집합으로 구성되어 있고, CompCO는 현실적인 복잡한 객체 배열로 구성되어 있습니다. 이를 통해 우리는 CLIP의 이미지 인코더와 텍스트 인코더의 편향을 정량적으로 분석하는 실험을 설계하였습니다.

- **Performance Highlights**: 실험 결과, 텍스트 인코더와 이미지 인코더 모두에서 확인된 편향은 모델 성능에 실질적인 영향을 미치는 것으로 나타났습니다. 대형 객체가 포함된 이미지에서의 분류 정확도는 더 작은 객체에 비해 유의미하게 높았으며, 텍스트에서의 첫 언급 객체는 전반적인 정확도에 큰 영향을 미쳤습니다. 이러한 편향은 이미지-캡션 매칭 작업에서 모델의 성능 저하를 초래하는 것으로 나타났습니다.



### Twofold Debiasing Enhances Fine-Grained Learning with Coarse Labels (https://arxiv.org/abs/2502.19816)
- **What's New**: 이번 논문에서는 Coarse-to-Fine Few-Shot (C2FS) 작업을 위한 새로운 접근법, Twofold Debiasing (TFB) 방법을 제안합니다. C2FS 작업은 제한된 수의 서브클래스 샘플을 통해 세밀한 인식을 달성하기 위해 거칠게 레이블링된 데이터를 활용합니다. 이러한 방법은 세밀한 특징 추출 및 분포 조정을 통해 모델의 성능을 향상시킵니다.

- **Technical Details**: TFB 방법은 다층 특징 융합 재구성 모듈 및 중간 레이어 특징 정렬 모듈을 통해 세밀한 특징의 인식을 개선하며, 이는 거칠게 레이블된 지도 아래에서의 간단함 편향(simplicity bias)을 완화합니다. 연구에서는 중간 레이어의 표현이 세밀한 분류 능력을 보존하는 데 중요한 역할을 한다는 것을 발견하였으며, 이를 활용하여 최종 임베딩의 세밀한 표현 능력을 향상시킵니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 실험 결과 TFB 접근법이 경쟁 방법들보다 뛰어난 성능을 보임을 입증하였습니다. 특히 CIFAR-100에서의 결과는 이전 방법들보다 5.89% 및 4.93% 더 높은 정확도를 기록하며, 이는 모델이 제한된 세밀한 샘플을 통해 효과적으로 학습할 수 있다는 것을 보여줍니다.



### UIFace: Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Recognition (https://arxiv.org/abs/2502.19803)
Comments:
          ICLR2025

- **What's New**: 이 논문에서는 합성 얼굴 인식을 위한 새로운 프레임워크인 UIFace를 제안합니다. 기존의 얼굴 인식 모델들이 자주 겪는 맥락 과적합(context overfitting) 문제를 해결하기 위해, 이 프레임워크는 모델의 내재적(inherent) 능력을 활용하여 클래스 내 다양성을 높이는 방안을 제시합니다. UIFace는 특정 정체성 맥락과 학습 가능한 빈 맥락(empty context)에 따라 샘플링을 수행하는 확산 모델을 사용하여, 두 단계(tw0-stage) 샘플링 전략을 통해 생성된 이미지의 다양성과 정체성을 유지합니다.

- **Technical Details**: 우선, UIFace는 특정 정체성 맥락을 조건으로 하는 샘플링과 학습 가능한 빈 맥락에 따라 샘플링을 수행합니다. 첫 번째 단계에서는 빈 맥락에서 샘플링하여 다양한 속성을 가진 이미지를 생성하며, 두 번째 단계에서는 주어진 정체성 조건에 따라 정체성을 유지하는 세부 정보를 생성합니다. 또한, 주의 주입(attention injection) 모듈을 통해 빈 맥락의 주의 맵을 조건부 생성 과정에 활용함으로써 클래스 내 변화를 더욱 증대시킵니다.

- **Performance Highlights**: 실험 결과, UIFace는 기존 최첨단 방법들을 큰 폭으로 초과하는 성능을 보여줍니다. 특히, 적은 양의 훈련 데이터로도 뛰어난 결과를 얻을 수 있으며, 심지어 반 이하의 합성 정체성으로도 상당한 성능을 발휘합니다. 또한, 정체성의 수를 더욱 늘릴 경우, UIFace는 실제 데이터셋으로 훈련된 얼굴 인식 모델과 비슷한 성능을 달성하는 놀라운 결과를 나타냅니다.



### No Parameters, No Problem: 3D Gaussian Splatting without Camera Intrinsics and Extrinsics (https://arxiv.org/abs/2502.19800)
- **What's New**: 이번 연구에서는 3D Gaussian Splatting (3DGS) 방법을 카메라 내부(intrinsic) 및 외부(extrinsic) 파라미터 없이, 이미지 집합에서 직접 학습하도록 하는 새로운 공동 최적화 방법을 제안합니다. 이 방법은 카메라의 초점 거리(focal length)의 그래디언트를 이론적으로 도출하여 학습 과정에서 카메라 내부 파라미터를 동시에 최적화할 수 있게 합니다. 또한, 전역 추적(global track) 정보를 통합하여 각 추적에 해당하는 Gaussian 커널을 선택, 훈련하면서 자동으로 크기를 줄입니다. 이러한 하이브리드 훈련 전략은 카메라 파라미터 최적화와 3DGS 훈련을 통합하여 안정성을 높입니다.

- **Technical Details**: 3DGS는 3D 비대칭 Gaussian으로 장면을 모델링합니다. 각 Gaussian은 중심(centroid), 쿼터니언(quaternion), 스케일(scale), 색상(color), 불투명도(opacity) 같은 파라미터로 정의됩니다. 이 연구에서는 카메라 파라미터와 3D Gaussians의 학습 과정을 결합하고, 각 3D Gaussian의 크기를 조정하여 표면 근처에 분포하도록 합니다. 최적화 과정의 안정성을 확보하기 위해 재투영 손실(reprojection loss)을 활용하여 카메라 위치와 파라미터를 업데이트합니다.

- **Performance Highlights**: 제안된 방법은 공공 벤치마크 데이터셋과 합성 데이터셋에서 광범위한 평가를 수행한 결과, 이전 방법들에 비해 카메라 파라미터 추정과 새로운 시점(view synthesis) 생성에서 최첨단(SOTA) 성능을 달성했습니다. 이방식은 오직 이미지 집합만을 입력으로 사용하여, 추가적인 카메라 파라미터에 대한 의존성을 크게 줄였습니다. 결과적으로, 본 연구는 3DGS의 효율성과 정확성을 향상시키며 기존 방법의 한계를 극복하는 방향으로 나아갔습니다.



### MFSR: Multi-fractal Feature for Super-resolution Reconstruction with Fine Details Recovery (https://arxiv.org/abs/2502.19797)
- **What's New**: 본 논문은 저해상도 이미지의 프랙탈(Fractal) 특징을 통합한 확산 모델 기반의 슈퍼 해상도(super-resolution) 방법인 MFSR을 제안합니다. MFSR은 노이즈 제거 과정에서 이러한 프랙탈 특징을 강화 조건으로 활용하여 텍스처 정보를 정확하게 복구하고자 합니다. 또한, 저해상도 이미지의 프랙탈 특징에 대한 근접을 위해 합성곱(convolution)을 활용하여 각기 다른 스케일에서 정보의 계층적 표현을 가능하게 합니다.

- **Technical Details**: MFSR은 Multi-Fractal Feature Extraction Block (MFB)을 사용하여 저해상도 이미지의 다중 프랙탈 특징을 근사화합니다. 논문은 U-Net 아키텍처를 기초로 한 주 노이즈 제거기를 통합하여 업샘플링(up-sampling) 과정 중 발생하는 노이즈를 줄이는 방법을 설명합니다. 이러한 접근 방식은 이미지의 자가 유사성(self-similarity) 특성을 다양한 스케일에서 인코딩하여 모델의 정보 획득을 풍부하게 합니다.

- **Performance Highlights**: 다양한 얼굴 및 자연 이미지 데이터셋에서 수행된 실험 결과, MFSR은 고화질 이미지를 생성할 수 있음을 보여줍니다. 특히, 프랙탈 특징을 통합한 MRFSR 방법이 기존의 확산 모델 기반 방법들에 비해 우수한 이미지 품질을 제공할 수 있음을 검증하였습니다. 본 연구는 소비자 제품 및 다양한 정교한 스케일의 이미지 분석에 유용할 것으로 기대됩니다.



### Open-Vocabulary Semantic Part Segmentation of 3D Human (https://arxiv.org/abs/2502.19782)
Comments:
          3DV 2025

- **What's New**: 본 논문은 3D 인간 파싱을 위한 최초의 오픈 어휘(segmentation) 방법을 제안합니다. 이 방법은 텍스트 프롬프트를 기반으로 인간 카테고리를 세부적으로 분할할 수 있는 능력을 가지고 있습니다. 또한, 기존의 모델들이 잘 처리하지 못했던 3D 인간 데이터에 대한 분할 성능을 크게 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 먼저 SAM(Segment Anything Model)을 이용하여 여러 뷰에서 2D 마스크 제안을 생성합니다. 이후, HumanCLIP 모델을 통해 이 마스크를 CLIP 특징 공간에서 통합된 임베딩(embedding)으로 변환합니다. MaskFusion 모듈은 텍스트 프롬프트에 따라 다중 뷰에서 일관된 3D 시맨틱 마스크를 생성하며, 복잡한 투표 및 그룹화 메커니즘 없이도 분류 및 융합 기능을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 3D 인간 데이터셋에서 기존의 최신 기술보다 월등한 성능을 보였습니다. 또한, 이 방법은 메쉬(mesh), 포인트 클라우드(point clouds), 3D Gaussian Splatting을 포함한 다양한 3D 표현에 직접 적용할 수 있는 가능성을 보여줍니다.



### RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings (https://arxiv.org/abs/2502.19781)
Comments:
          Accepted to CVPR 2025

- **What's New**: 이 논문에서는 지리적 위치의 표현이 다양한 지리 공간 작업의 정확성에 미치는 영향을 강조하고 있습니다. 기존의 모델들이 효율적으로 작동하지만, 현재의 학습 전략이 중요한 시각적 특징을 충분히 포착하지 못한다고 주장합니다. 저자들은 새로운 검색 증강 전략인 RANGE를 제안하여, 여러 유사한 위치의 시각적 특성을 결합해 위치 정보를 근사화합니다.

- **Technical Details**: RANGE는 시맨틱(semantic) 및 공간(spatial) 정렬을 활용하여 위치의 시각적 특징을 추정하는 검색자(retriever) 기능을 포함합니다. 이러한 기능은 데이터베이스 크기에 강건하며, 서로 다른 빈도로 지리적 임베딩을 생성할 수 있습니다. 저자들은 조회 기반 접근 방식이 기존 방법들보다 더 나은 결과를 제공한다고 보고하며, 상대적인 성능 향상을 강조합니다.

- **Performance Highlights**: RANGE는 분류 작업에서 최대 13.1%의 성능 향상을 보여주었으며, 회귀 작업에서는 0.145 $R^2$의 향상을 기록했습니다. 이 모델은 많은 지리 공간 작업에서 기존의 최첨단 모델들을 상당한 차이로 초월하는 것으로 평가되었습니다. 저자들은 GitHub에 코드와 HuggingFace에 모델을 배포할 계획이라고 밝혔습니다.



### InPK: Infusing Prior Knowledge into Prompt for Vision-Language Models (https://arxiv.org/abs/2502.19777)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)의 적응을 개선하기 위해 InPK라는 새로운 모델을 제안하고 있습니다. 특히, 이 모델은 learnable tokens가 클래스별 prior knowledge로 초기화되어 클래스 관련 정보를 명확히 집중할 수 있도록 합니다. 또한, 멀티 레이어 인코더 간에 learnable tokens와 prior knowledge 간의 상호작용을 강화하여 모델의 학습을 최적화합니다.

- **Technical Details**: InPK는 CLIP 모델을 기반으로 하여 설계되었으며, 이미지와 텍스트 간의 글로벌 정렬을 위한 contrastive loss를 활용합니다. 이 모델은 learnable tokens에 클래스별 prior knowledge를 주입하여 정보의 질을 향상시키고, multi-layer 인코더 간의 정보 손실을 방지하기 위해 상호작용을 지속적으로 강화합니다. 또한, 학습 가능한 text-to-vision projection layer를 도입하여 시각적 및 텍스트 의미 간의 정렬을 개선합니다.

- **Performance Highlights**: 전반적인 실험 결과는 InPK가 11개의 다양한 인식 데이터셋에서 기존의 최첨단 방법들보다 현저하게 높은 성능을 보임을 보여줍니다. 특히 zero/few-shot 이미지 분류 작업에서 기저 클래스 및 새로운 클래스에서 모두 우수한 정확도를 달성했습니다. 이는 다수의 복잡한 시각적 개념을 효과적으로 식별하고 일반화 능력을 향상시킨 결과입니다.



### QORT-Former: Query-optimized Real-time Transformer for Understanding Two Hands Manipulating Objects (https://arxiv.org/abs/2502.19769)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이번 논문은 2개의 손과 물체의 3D 포즈 추정을 위한 최초의 Transformer 기반 실시간 프레임워크인 Query-Optimized Real-Time Transformer (QORT-Former)를 소개합니다. 기존 상태의 모델들이 높은 정확도를 제공하는 대신, 상당한 계산 비용을 수반하고 있어 실시간 응용에는 부적합했습니다. 이를 해결하기 위해 본 연구에서는 쿼리의 수를 제한하고, 쿼리 개선을 통해 정확성을 높이는 방식을 제안합니다.

- **Technical Details**: QORT-Former는 108개의 쿼리와 1개의 디코더로 구성되며, 이에 따라 53.5 FPS의 실시간 성능을 달성합니다. 쿼리를 손 2개 및 물체 쿼리 세 가지 타입으로 나누어 최적화하고, 접촉 정보를 활용하여 쿼리 기능을 강화합니다. 이 과정은 이미지 기능과 쿼리 기능을 상호 최적화하는 삼단계 업데이트 프로세스를 포함합니다.

- **Performance Highlights**: QORT-Former는 H2O 데이터셋에서 왼손 17.6%, 오른손 22.8%, 물체 27.2% 향상된 성능을 보여주며, FPHA 데이터셋에서도 오른손 5.3%, 물체 10.4% 개선된 성과를 기록했습니다. 제안된 방법은 실시간 속도를 유지하면서도 상호작용 인식을 위해도 최첨단 성능을 기록하였습니다.



### Automatic Temporal Segmentation for Post-Stroke Rehabilitation: A Keypoint Detection and Temporal Segmentation Approach for Small Datasets (https://arxiv.org/abs/2502.19766)
- **What's New**: 본 연구는 뇌졸중 환자의 재활 치료에서 비디오 기록을 통한 일관되고 신속한 분석 방법을 제시합니다. 환자들의 치료 스케줄에 맞춰 보다 객관적이고 자동화된 평가 방식이 필요하다는 점을 강조하고 있습니다. 이를 통해 행동 연구 팔 테스트(Action Research Arm Test, ARAT)와 같은 임상 평가를 지원하고, 환자의 손 동작과 대상 물체 간의 상호작용을 효과적으로 분석할 수 있는 방법을 모색합니다.

- **Technical Details**: 연구는 2D 키포인트 탐지 및 1D 시간적 분할의 두 가지 주요 작업으로 구성됩니다. 이러한 방법론적 접근은 비디오 데이터의 복잡성을 줄이고, 소량의 실제 데이터로도 자동 레이블링을 가능하게 합니다. 또한, 각 환자의 손 동작을 세밀하게 분석하기 위해 트랜스포머 기반 모델을 활용하여 시간적 분할 예측을 수행합니다.

- **Performance Highlights**: 제안된 프레임워크는 작은 데이터셋에서도 Overfitting(과적합) 문제를 줄이면서도 효과적인 행동 분할을 보장합니다. 실제 환자의 비디오 데이터를 활용한 실험 결과, 이 방법이 물리 치료 환경에서 실용적으로 적용될 가능성이 높음을 시사합니다. 전반적으로, 본 연구는 재활 평가의 신속성과 정확성을 향상시키는데 기여할 것으로 기대됩니다.



### Snowball Adversarial Attack on Traffic Sign Classification (https://arxiv.org/abs/2502.19757)
- **What's New**: 이번 논문에서는 Snowball Adversarial Attack이라는 새로운 물리적 공격 방식이 제안되었습니다. 이는 교통 신호 인식을 위해 스노우볼과 같은 눈의 축적을 시뮬레이션하여 오분류를 유도합니다. 기존의 공격 방법과 달리 이 공격은 눈으로 덮인 신호를 인식하지 못하게 하는 반면, 운전자는 이를 자연스럽게 인식할 수 있도록 설계되었습니다.

- **Technical Details**: Snowball Adversarial Attack은 교통 신호의 특정 지역에 눈의 패치를 전략적으로 배치하여 오분류율을 극대화합니다. 저자들은 Digital Framework를 통해 기존 환경을 안전하고 윤리적으로 변화시키지 않고도 공격의 효과를 평가할 수 있는 방법을 제공합니다. 이를 통해 실제 도로 환경에서의 효과적인 평가가 가능합니다.

- **Performance Highlights**: Snowball Adversarial Attack은 실제 교통 신호 및 다양한 눈 패치를 사용하여 고급 알고리즘을 혼란에 빠뜨릴 수 있는 능력을 입증했습니다. 연구 결과 본 공격은 자율 주행차와 지능형 교통 시스템에 중대한 위협이 될 수 있으며, 교통 표지 인식 모델의 취약성에 대한 우려를 제기합니다.



### Finding Local Diffusion Schrödinger Bridge using Kolmogorov-Arnold Network (https://arxiv.org/abs/2502.19754)
Comments:
          16 pages, 10 figures, to be published in CVPR 2025

- **What's New**: 이번 논문에서는 첫 번째로 지역적인 확산 슈뢰딩거 다리(Local Diffusion Schrödinger Bridges, LDSB)를 제안하여 확산 경로 부분공간에서 최적화된 경로를 찾는 방법을 소개합니다. 이를 통해 슈뢰딩거 다리(Schrödinger Bridge) 문제와 확산 모델 간의 연관성을 강화하고 이미지 생성 품질을 향상시킵니다. 특히, Kolmogorov-Arnold Network (KAN)를 통해 훈련 중 최적화된 경로를 찾아 효율성을 높이고 있습니다.

- **Technical Details**: LDSB는 확산 경로 부분공간을 재파라미터화하여 시간 단계 간의 더욱 정확한 연결을 달성하는 새로운 전략을 제시합니다. 이 방법은 기존의 복잡한 네트워크 대신 0.1MB도 안 되는 경량 Neural Network를 사용해 경로 최적화를 수행합니다. LDSB는 확산 기반 DDIM 및 Flow 기반 Recited Flow라는 두 개의 주요 확산 프레임워크에서 테스트되었으며, 적은 샘플링 단계(5, 10 및 20단계)에서의 성능을 중점적으로 평가했습니다.

- **Performance Highlights**: LDSB는 같은 사전 훈련된 제거 네트워크를 사용하면서도 이미지 생성 품질이 대폭 향상되었습니다. FID 메트릭이 15% 이상 감소하였으며, 특히 DDIM의 NFE가 5일 때 CelebA 데이터셋에서 48.50% 감소하는 성과를 보였습니다. 이러한 결과들은 경량화된 네트워크와 낮은 계산 비용을 기반으로 한 효율적인 확산 경로 최적화의 가능성을 보여줍니다.



### Lightweight Contrastive Distilled Hashing for Online Cross-modal Retrieva (https://arxiv.org/abs/2502.19751)
- **What's New**: 이번 연구에서는 효율적인 온라인 크로스모달 해싱(online cross-modal hashing)을 위한 Lightweight Contrastive Distilled Hashing (LCDH) 방식을 제안합니다. 기존의 방법들과 달리, LCDH는 오프라인 학습과 온라인 학습 사이의 연결고리를 혁신적으로 구성하여 유사도 행렬( similarity matrix) 근사를 통해 지식을 이전합니다. 이를 통해 크로스모달 데이터의 공존하는 의미적 관련성을 효과적으로 추출하고 학습할 수 있는 기반을 제공합니다.

- **Technical Details**: LCDH는 교사 네트워크에서 CLIP(Contrastive Language-Image Pre-training)을 사용해 크로스모달 특성을 추출하는 방식으로 시작합니다. 이 특성들은 주의 모듈(Attention Module)을 통해 추가적으로 표현이 강화된 후, 유사도 행렬의 크기를 맞추기 위해 완전 연결층(FC Layer)로 전달됩니다. 학생 네트워크에서는 경량 모델을 사용하여 시각적 및 텍스트 특성을 추출하고 그 결과를 이진 코드(binary codes)로 변환합니다.

- **Performance Highlights**: 세 가지 널리 사용되는 데이터 세트에서 수행한 실험 결과는 LCDH가 최신 방법들에 비해 우수하고 안정적인 성능을 발휘함을 보여줍니다. 이 연구는 크로스모달 데이터의 대규모 비율 증가에 따른 요구를 충족시키기 위해 LLM(대형 언어 모델)와 경량화된 접근 방식을 결합하여 실시간 데이터 스트리밍 처리에 대한 가능성을 증가시키고 있습니다.



### LUCAS: Layered Universal Codec Avatars (https://arxiv.org/abs/2502.19739)
- **What's New**: LUCAS는 Face와 Hair를 분리하여 독립적으로 변형할 수 있는 레이어드 표현 방식을 적용한 새로운 Universal Prior Model(UPM)입니다. 기존의 UPM들은 머리카락을 머리의 일부분으로 취급하였던 반면, LUCAS는 머리와 머리카락을 각각 다른 브랜치로 분리하여 모델링합니다. 이를 통해 실시간 렌더링이 가능하며, 정밀하고 시각적으로 매력적인 Gaussian 렌더링을 제공합니다. LUCAS는 3D 헤드 아바타 재구성에서 동적인 성능을 개선하여 표현 전송과 헤어스타일 변화를 효과적으로 관리합니다.

- **Technical Details**: LUCAS는 레이어드 표현을 통해 얼굴과 머리카락을 따로 모델링하여 독립적인 변형을 가능하게 합니다. 이 방식은 각 요소가 높은 정확도로 정렬되면서도 자연스럽게 변형되도록 합니다. 특히, LUCAS는 공유된 인코딩 기능을 사용하되 얼굴과 머리카락을 별도로 디코딩하여 더 나은 물리적 상호작용을 제공합니다. 또한, 다중 사용자 데이터를 바탕으로 UPM을 훈련하여 미지의 사용자에게도 쉽게 일반화할 수 있습니다.

- **Performance Highlights**: LUCAS는 정량적 및 정성적 평가에서 기존의 단일 메쉬 및 Gaussian 기반 아바타 모델보다 뛰어난 성능을 보입니다. 특히, LUCAS는 미지의 피험자에 대한 제로 샷 주행 시나리오에서도 우수한 결과를 나타내며, 다양한 표정 및 머리 위치 변화에 효과적으로 대응합니다. 실험 결과에 따르면, LUCAS는 3D 헤드 아바타 재구성의 최신 경지를 한층 발전시키고 있습니다.



### Learning Mask Invariant Mutual Information for Masked Image Modeling (https://arxiv.org/abs/2502.19718)
Comments:
          ICLR 2025

- **What's New**: 이 논문은 Masked Autoencoders (MAEs)의 성능 향상을 위한 새로운 관점을 제시하고 있습니다. 정보 이론의 정보 병목 (information bottleneck) 원리를 활용하여 MAEs의 작동 방식을 체계적으로 이해하고 최적화하는 방법을 모색합니다. 이를 통해 MAE의 핵심 메커니즘을 밝히고, 더 효과적인 모델 개발을 위한 이론적 통찰을 제공합니다.

- **Technical Details**: MI-MAE라는 새로운 방법론을 도입하여, latent feature를 최적화하는 과정에서 관련 정보의 최대화와 무관한 정보의 최소화를 목표로 합니다. 이 방법은 두 가지 측면으로 나뉘며, 상호 정보 최대화 (mutual information maximization)와 상호 정보 최소화 (mutual information minimization) 손실을 포함합니다. 이러한 접근 방식은 MAE가 다양한 비전 태스크에서 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: MI-MAE는 이미지 분류, 객체 탐지, 의미 세분화와 같은 다양한 태스크에서 MAE 모델을 초월하는 성능을 보였습니다. 예를 들어, 400 epoch 모델은 ImageNet-1K에서 83.9%의 정확도를 기록하여 1600 epoch MAE 모델을 0.5% 초과했습니다. 이러한 결과는 정보 병목 원리를 적용함으로써 나타날 수 있는 실질적인 이점을 입증합니다.



### Recent Advances on Generalizable Diffusion-generated Image Detection (https://arxiv.org/abs/2502.19716)
- **What's New**: 최근 확산 모델(difusion models)의 발전은 생성된 이미지의 충실도(fidelity)와 다양성(diversity)을 크게 향상시켰습니다. 하지만 이러한 발전은 고품질의 딥페이크(Deepfake) 이미지를 만드는 데 악용될 수 있어 이미지 신뢰성 검증에 도전 과제를 안깁니다. 이에 따라 생성된 이미지 탐지에 대한 연구가 급증하고 있지만, 이 주제에 대한 포괄적 리뷰는 여전히 부족합니다.

- **Technical Details**: 이 논문에서는 최근의 발전을 체계적으로 조사하고 이들을 두 가지 주요 범주로 분류하여 제시합니다: (1) 데이터 기반(datat-driven) 탐지 및 (2) 특성 기반(feature-driven) 탐지. 탐지 방법들은 기본 원리에 따라 여섯 가지 세부 범주로 세분화됩니다. 데이터 기반 탐지 방법은 명시적인 수작업 특징에 의존하지 않고, 데이터 주도적인 방식으로 일반화 가능한 특징을 포착하는 능력을 향상시킵니다.

- **Performance Highlights**: 링크된 연구는 주로 두 가지 유형으로 나뉘어집니다. 첫 번째 유형은 사람에게 인식 가능한 이미지 특징(Perceptible Image Features)을 사용하는 것이고, 두 번째는 인식 불가능한 이미지 특징(Imperceptible Image Features)을 분석하는 것입니다. 탐지 방법의 내구성(post-processing에 대한 저항력) 및 더 강력한 이론적 토대의 필요성도 강조되며, 고품질의 다양한 데이터셋의 개발과 같이 이 분야에서의 향후 연구 방향이 제시됩니다.



### SAP-DIFF: Semantic Adversarial Patch Generation for Black-Box Face Recognition Models via Diffusion Models (https://arxiv.org/abs/2502.19710)
- **What's New**: 본 연구에서는 얼굴 인식 모델의 강건성 평가를 위하여 새로운 대적 패치 공격 방법인 SAP-DIFF를 제안합니다. 이 방법은 기존의 국부적 왜곡 공격의 한계를 극복하고, 고차원 잠재 공간에서 의미론적 왜곡을 통해 대적 패치를 생성합니다. 이는 공격자의 능력 요구 사항을 낮추고, 공격 성공률을 향상시키며, 쿼리 수를 줄이는 데 도움을 줍니다.

- **Technical Details**: SAP-DIFF는 확산 모델(difusion model)을 활용하여, 주의 파괴 메커니즘(attention disruption mechanism)과 방향성 손실 함수(directional loss function)를 통합하여 대적 샘플을 생성합니다. 이를 통해 원래 얼굴과 관련 없는 특징을 생성하면서도, 목표 아이덴티티의 특징 공간(target identity feature space)으로의 왜곡을 유도하여 공격의 효과적이고 효율성을 극대화합니다. 또한, DDIM Inversion을 사용하여 대적 패치를 초기화하고, 최적화 과정에서 몇 가지 단계로 나누어 프로세스를 수행합니다.

- **Performance Highlights**: 대중적인 얼굴 인식 모델(ArcFace, CosFace, FaceNet)과 데이터셋(LFW, CelebA-HQ)을 대상으로 한 실험 결과, SAP-DIFF는 기존의 최신 기법들에 비해 평균적으로 45.66%의 공격 성공률 개선을 달성하였으며(모든 작업에서 40% 이상 향상됨), 쿼리 수는 약 40% 줄였습니다. 이러한 성과는 얼굴 인식 시스템의 보안성을 더욱 강화하는 데 기여할 것으로 기대됩니다.



### Accurate Pose Estimation for Flight Platforms based on Divergent Multi-Aperture Imaging System (https://arxiv.org/abs/2502.19708)
- **What's New**: 본 논문에서는 비전 기반 자세 추정 기술을 개선하기 위해 다차원 다초점 이미징 시스템(DMAIS)을 제안합니다. DMAIS는 동시에 넓은 시야각과 높은 공간 해상도를 달성할 수 있도록 설계되었으며, 기존의 관측 한계를 극복할 수 있습니다. 이 시스템은 기하학적 보정과 절대 자세 추정 알고리즘을 통해 비행 플랫폼의 정확한 자세 추정이 가능하게 합니다.

- **Technical Details**: DMAIS는 다섯 개의 긴 초점 카메라로 구성되어 있으며, 각 카메라는 분산된 관측 방향을 가지고 있습니다. 각 카메라는 좁은 시야각에도 높은 공간 해상도를 유지할 수 있으며, 이들 카메라의 조합을 통해 넓은 FoV를 효과적으로 구현합니다. 또한, 본 논문에서는 3D 보정 필드를 사용하여 DMAIS의 내부 및 외부 매개변수를 결정하는 새로운 보정 방법을 제안하고 있습니다.

- **Performance Highlights**: 실제 비행 실험 결과, DMAIS를 통해 센티미터 수준의 위치 정확도와 아크 분 수준의 방향 정확도를 달성할 수 있음을 보여줍니다. 새로운 절대 자세 추정 알고리즘은 기존의 방법들과 비교하여 더욱 우수한 성능을 발휘하며, 실제 비행 환경에서 비행 플랫폼의 정확한 자세 추정을 가능하게 합니다.



### Weakly Supervised Segmentation Framework for Thyroid Nodule Based on High-confidence Labels and High-rationality Losses (https://arxiv.org/abs/2502.19707)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문에서는 초음파 영상에서 갑상선 결절을 구분하기 위한 새로운 약한 감독(segmentation) 프레임워크를 제안합니다. 제안된 방법은 낮은 신뢰도를 가진 pseudo-labels (가짜 라벨) 문제와 비합리적인 손실 함수(low-rationality loss function)를 개선하여 더 정교하고 신뢰성 있는 결절 분할 성능을 보여줍니다. 구체적으로, 네 개의 포인트 주석에서 기하학적 변환을 융합하고 특정 주석에 의해 유도된 MedSAM 모델 결과를 통해 높은 신뢰도의 라벨을 생성합니다.

- **Technical Details**: 제안된 방법은 네 가지 포인트 주석(annotations)에서 파생한 기하학적 변환과 MedSAM의 결과를 융합하여 높은 신뢰도의 박스, 전경(foreground), 배경(background) 라벨을 생성합니다. 고차원(dimensional) 다층 손실(high-rationality losses) 전략을 사용하여 세 가지 주요 손실 함수를 채택합니다: 1) 정렬 손실(alignment loss), 2) 대비 손실(contrastive loss), 3) 프로토타입 상관 손실(prototype correlation loss). 각각의 손실 함수는 네트워크가 결절의 위치와 모양을 보다 정교하게 학습할 수 있도록 유도합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 TN3K 및 DDTI 데이터셋에 대해 현재의 최고 성능(state-of-the-art)을 달성함을 보여주었습니다. 또한, 이 연구에서 제안된 코드는 공개적으로 제공되어 연구 및 실용적 응용에 활용될 수 있습니다. 논문은 약한 감독(segmentation) 방식이 갑상선 결절 분할의 진전을 가져올 수 있다는 것을 명확히 밝혔습니다.



### CFTrack: Enhancing Lightweight Visual Tracking through Contrastive Learning and Feature Matching (https://arxiv.org/abs/2502.19705)
- **What's New**: 이 논문은 CFTrack이라는 경량(軽量) 비주얼 트래킹 알고리즘을 제안하며, 이는 대조 학습(contrastive learning)과 피쳐 매칭(feature matching)을 통합하여 식별(識別) 성능을 향상시키는 데 초점을 맞추고 있습니다. CFTrack은 신규 대조적 피쳐 매칭 모듈을 통해 예측 시 목표 유사도를 동적으로 평가하여 트래킹의 정확도를 개선합니다. 실험 결과 CFTrack은 LaSOT, OTB100, UAV123에서 많은 최신 경량 트래커들을 초월하며, NVIDIA Jetson NX 플랫폼에서 실시간 136 프레임을 초과하는 성능을 보여주었습니다.

- **Technical Details**: CFTrack의 파이프라인은 Siamese 구조를 활용하며, MobileNetV2를 기반으로 한 피쳐 추출기를 사용합니다. 이 알고리즘은 두 개의 분기(template branch 및 search branch)를 공유하는 매개변수로 구성되어 있습니다. 대조적 피쳐 매칭 모듈(CFM)은 특정 피쳐 유사도를 평가하여 경량 트래커에서 식별 능력을 향상하기 위해 대조 학습 원리를 통합하고 있습니다.

- **Performance Highlights**: CFTrack은 HOOT 데이터세트에서 심한 occlusion 상황에서도 강력한 구별 능력을 입증했습니다. 또한, 기능적 실험들은 CFTrack이 정확도와 강인성을 모두 만족하며 최신 스테이트 오브 더 아트 경량 트래커들을 초월함을 보여주고 있습니다. 이러한 성능 개선은 실시간 추적에 필요한 컴퓨테이셔널 효율성을 유지하면서 실현되었습니다.



### Language-Informed Hyperspectral Image Synthesis for Imbalanced-Small Sample Classification via Semi-Supervised Conditional Diffusion Mod (https://arxiv.org/abs/2502.19700)
- **What's New**: 이 논문에서는 텍스트 정보에 기반하여 현실적이고 다양한 샘플을 생성하여 불균형 소량 샘플 데이터(imblanced-small sample data, ISSD) 문제를 해결하는 새로운 언어 정보 기반 하이퍼스펙트럴 이미지 합성 방법(Txt2HSI-LDM(VAE))이 제안되었습니다. 이는 기존의 데이터 증대(data augmentation) 기법들과는 달리, 잠재 공간(latent space)에서의 특징 확장에 국한되지 않고, 텍스트 정보를 활용하여 주목할 만한 성과를 나타냅니다. 이를 통해 보다 안정적이고 다양성이 높은 양질의 이미지를 생성할 수 있습니다.

- **Technical Details**: 제안된 방법은 고차원 하이퍼스펙트럴 데이터에 대한 변형 자동 인코더(universal variational autoencoder, VAE)를 활용하여 하이퍼스펙트럴 이미지를 저차원 잠재 공간으로 매핑하여 안정적인 특징 표현을 얻습니다. 또한 반지도(diffusion model) 모델과 결합하여 레이블이 없는 데이터를 완전히 활용하는 방식으로 구축되어 있으며, 무작위 다각형 공간 클리핑(random polygon spatial clipping, RPSC) 및 잠재 특징의 불확실성 추정(latent feature uncertainty estimation, LF-UE)을 사용하여 변화하는 훈련 데이터의 혼합 정도를 모사합니다. 마지막으로, VAE는 생성된 잠재 공간으로부터 하이퍼스펙트럴 이미지를 디코드합니다.

- **Performance Highlights**: 실험 결과, 수치적 특성과 데이터 분포 면에서 생성된 샘플의 효용성이 입증되었습니다. 2D-PCA 공간에서 통계적 특성을 평가하고, 픽셀 수준에서의 크로스 어텐션 맵을 시각화하여 시각-언어 정렬을 기반으로 생성된 하이퍼스펙트럴 이미지의 공간 구조를 포착하는 능력을 입증했습니다. 이는 제안된 Txt2HSI-LDM(VAE)이 불균형 소량 샘플 데이터 문제를 해결하는 데 있어 효과적인 기술적 접근임을 보여줍니다.



### Spatial-Spectral Diffusion Contrastive Representation Network for Hyperspectral Image Classification (https://arxiv.org/abs/2502.19699)
- **What's New**: 이 논문은 하이퍼스펙트럼 이미지 분류(HSIC)를 위한 새로운 Spatial-Spectral Diffusion Contrastive Representation Network(DiffCRN)를 제안합니다. DiffCRN은 Denoising Diffusion Probabilistic Model(DDPM)과 Contrastive Learning(CL)을 결합하여 효율적인 공간-스펙트럼 특징 추출을 목표로 하고 있습니다. 특히, 공간 자기 주의력(Denoising Module)과 스펙트럼 그룹 자기 주의력 모듈을 통해 더욱 향상된 효율성과 성능을 구현했습니다.

- **Technical Details**: DiffCRN에서는 UNets-like 구조 대신, 새로운 단계식 아키텍처를 설계하여 독특한 공간-스펙트럼 특징을 학습합니다. 또한, Logarithmic Absolute Error(LAE)와 Contrastive Learning(CL)을 결합하여 비지도 특징 학습의 효율성을 높였으며, 픽셀 수준의 Spectral Angle Mapping(SAM) 방식을 통해 시간 단계 선택을 자동화했습니다. 마지막으로 Adaptive Weighted Addition Module(AWAM)와 Cross Time Step Spectral-Spatial Fusion Module(CTSSFM)을 설계하여 특징 통합 및 분류 과정을 개선합니다.

- **Performance Highlights**: 네 가지 주요 HSI 데이터셋에서 진행된 실험 결과, DiffCRN은 기존의 고전적인 백본 모델 및 최신 GAN, Transformer 모델과 다른 사전 훈련된 방법에 비해 월등한 성능 향상을 보였습니다. 이러한 결과는 DiffCRN의 설계가 하이퍼스펙트럼 이미지의 복잡한 공간-스펙트럼 관계를 효과적으로 모델링할 수 있음을 보여줍니다. 연구진은 코드와 사전 훈련된 모델을 공개할 예정이며, 이는 향후 연구에 큰 기여를 할 것으로 기대됩니다.



### You Only Click Once: Single Point Weakly Supervised 3D Instance Segmentation for Autonomous Driving (https://arxiv.org/abs/2502.19698)
- **What's New**: 이번 연구에서는 야외 LiDAR 포인트 클라우드의 3D 인스턴스 세그멘테이션을 위한 새로운 프레임워크, YoCo를 제안합니다. 이 프레임워크는 조감도(BEV)에서 최소한의 클릭 주석을 통해 3D 의사 레이블을 생성할 수 있도록 설계되었습니다. YoCo는 기존의 방법들보다 낮은 주석 비용으로 인스턴스 세그멘테이션 성능을 향상시킵니다.

- **Technical Details**: YoCo는 비전 파운데이션 모델(vision foundation models)과 포인트 클라우드의 기하학적 제약을 활용하여 높은 품질의 3D 의사 레이블을 생성합니다. 또한, 이 프레임워크는 인접 프레임의 예측을 활용하여 신뢰할 수 있는 레이블 업데이트 모듈을 설계하고, 인터섹션 오버 유니온(intersection-over-union, IoU)에 기반한 레이블 향상 모듈을 도입하여 레이블의 품질을 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, YoCo는 약한 감독 방법 중에서 최첨단 성능을 달성했으며, 완전 감독 기반의 Cylinder3D를 초월했습니다. 또한, YoCo는 다양한 네트워크에 적합하며, 완전히 라벨링된 데이터의 0.8%로도 좋은 성능을 보이는 것으로 나타났습니다.



### Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion (https://arxiv.org/abs/2502.19697)
- **What's New**: 본 논문에서는 Attribute-aware Prompt Attack(AP-Attack)이라는 새로운 방법을 도입하여, VLM(vision-language model)의 이미지-텍스트 정렬 능력을 활용하여 보행자 이미지의 세부적인 의미적 특성을 파괴하는 메커니즘을 제안합니다. 이는 특성별로 구분되는 텍스트 임베딩(attribute-specific textual embeddings)을 파괴함으로써 이루어집니다. 또한, 개인화된 텍스트 설명을 생성하기 위해 텍스트 인버전 네트워크(textual inversion networks)를 설계하여, 보행자 이미지를 의미적 임베딩을 나타내는 의사 토큰(pseudo tokens)으로 매핑합니다.

- **Technical Details**: AP-Attack 방법은 세밀한 의미적 특성을 조작하기 위해 기존 프롬프트 드리븐 공격 방법에서의 한계를 극복하며, 모든 보행자 속성에 대해 개별화된 프롬프트 템플릿을 활용합니다. 각 보행자 속성에 맞춰 변환된 임베딩은 대조 학습(contrastive learning) 방식으로 훈련되어, 정상 및 적대적인 세밀한 텍스트 의미를 이끌어내도록 합니다. 이 과정에서 프롬프트 기반의 의미 공격 손실(promise-driven semantic attack loss)을 도입하여, 적대적 의미를 원래의 것에서 멀어지게 하면서도 가장 유사하지 않은 의미에 가까워지도록 유도합니다.

- **Performance Highlights**: 대규모 실험을 통해 AP-Attack이 다양한 모델 및 데이터셋에서 최첨단의 전이 가능성을 달성하여, 이전 방법들보다 평균 Drop Rate에서 22.9%의 성능 향상을 보임을 입증하였습니다. 이 연구는 보행자 재식별 작업에서 전이 가능한 세밀한 의미적 변동을 효과적으로 생성할 수 있는 방법을 제시하며, 적대적 예제의 전이 가능성을 대폭 향상시키는 데 중점을 두고 있습니다.



### BEVDiffuser: Plug-and-Play Diffusion Model for BEV Denoising with Ground-Truth Guidanc (https://arxiv.org/abs/2502.19694)
Comments:
          CVPR 2025

- **What's New**: 본 연구에서는 BEV (Bird's-eye-view) 표현의 노이즈 문제를 해결하기 위한 새로운 확산 모델인 BEVDiffuser를 제안합니다. BEVDiffuser는 실제 객체 레이아웃을 가이던스로 하여 BEV 특징 맵을 효과적으로 노이즈 제거합니다. 이 모델은 기존 BEV 모델을 수정할 필요 없이 플러그 앤 플레이 방식으로 훈련 중에 작동하여 BEV 표현을 향상시킵니다.

- **Technical Details**: BEVDiffuser는 BEVFormer, BEVFusion과 같은 기존 BEV 모델에서 생성된 특징 맵에 다양한 수준의 노이즈를 추가하여 훈련됩니다. 훈련 후에는 BEV 특징 맵을 정화하여 추가적인 감독을 제공하는 방식으로 기존 BEV 모델의 성능을 개선합니다. 또한, BEVDiffuser는 훈련 시간이 끝나면 제거되며, 추론 시 추가적인 컴퓨팅 지연 없이도 강력한 성능을 유지합니다.

- **Performance Highlights**: nuScenes 데이터셋을 통해 실시한 실험 결과, BEVDiffuser는 3D 객체 탐지에서 mAP 12.3% 및 NDS 10.1%의 현저한 개선을 보여주었습니다. 장기적인 객체 탐지 및 다양한 환경 조건에서도 성능이 크게 향상되었으며, 질적으로도 고품질의 BEV 생성 능력을 입증했습니다. 이러한 성능 개선은 자율 주행의 발전을 위한 대규모 데이터 수집에 기여할 것으로 기대됩니다.



### Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach (https://arxiv.org/abs/2502.19691)
Comments:
          Accepted to CVPR 2025

- **What's New**: 본 논문에서는 Energy-based Active Open-set Annotation (EAOA) 프레임워크를 제안하여, 알려지지 않은 클래스가 존재하는 상황에서도 우수한 성능을 발휘할 수 있는 방법을 모색합니다. 기존의 방법들은 알려진 클래스에 속할 가능성이 높은 query 예제를 우선시하거나, 불확실한 예제를 쿼리하는 데 중점을 두어 최적의 성능을 발휘하지 못했습니다. EAOA는 이러한 두 가지 불확실성을 결합하여 모델 훈련의 효율성을 높이는 새로운 접근법을 제시합니다. 또한, 실제 데이터에 대해 높은 쿼리 정밀도와 낮은 훈련 오버헤드를 유지하면서 최첨단 성능을 달성하였습니다.

- **Technical Details**: EAOA는 (C+1) 클래스 검출기와 타겟 분류기로 구성되어 있습니다. 이 시스템은 에너지 기반의 epistemic uncertainty (EU) 측정 방법과 특성 기반의 energy loss를 도입하여 검출기를 훈련시킵니다. 또한, 타겟 분류기에는 가능한 aleatoric uncertainty (AU)를 측정하는 에너지 기반 지표를 적용하여 보다 정교하게 클래스 식별을 수행합니다. 핵심 구성 요소 중 하나는 목표 기반의 적응형 샘플링 전략으로, 이는 높은 AU 점수를 가진 쿼리 집합을 형성하기 위해 낮은 EU 점수를 가진 후보 집합을 우선 생성하는 방식입니다.

- **Performance Highlights**: 광범위한 실험을 통해 EAOA가 기존의 최신 연구 결과들보다 테스트 정확도와 쿼리 정밀도, 훈련 효율성에서 뛰어난 성능을 발휘함을 입증했습니다. EAOA는 open-world 시나리오에서도 유용한 정보가 포함된 예제를 효과적으로 쿼리하여, 다양한 데이터 상황에서도 일관된 결과를 보여줍니다. 특히, EAOA는 c랑 close-set 특성을 적절히 조합하여 높은 학습 효과를 달성했습니다.



### 3D Trajectory Reconstruction of Moving Points Based on a Monocular Camera (https://arxiv.org/abs/2502.19689)
- **What's New**: 이 논문은 단일 카메라를 활용하여 이동하는 점의 3D 궤적을 재구성하기 위한 새로운 알고리즘을 제시합니다. 일반적으로, 점의 이동에 대한 합리적인 가정 없이 단일 카메라 이미지만으로 3D 위치를 측정하는 것은 불가능합니다. 이 연구는 제한된 관측 조건을 완화하기 위해 리지 추정(ridge estimation)을 도입하고, 시간 다항식(temporal polynomials)을 사용하여 점의 움직임을 표현합니다.

- **Technical Details**: 저자는 첫째, 카메라의 움직임을 기반으로 점의 운동을 표현하기 위해 시간 다항식을 활용합니다. 둘째, 지오메트릭 오류 목표 함수를 최소화함으로써 시간 다항식의 차수를 자동으로 결정하는 알고리즘을 제안합니다. 마지막으로, 자동으로 선택된 차수의 시간 다항식이 가진 재구성 가능성(reconstructability)을 정의하여 재구성 정확도를 정량적으로 설명합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험 결과는 제안된 방법이 가능성과 정확성, 효율성을 모두 유지하면서 우수한 성능을 발휘함을 보여줍니다. 이 연구는 UAV를 활용한 이동 목표의 3D 궤적 재구성에서 특히 유용하며, 기존의 방법보다 향상된 정확도를 자랑합니다.



### M-LLM Based Video Frame Selection for Efficient Video Understanding (https://arxiv.org/abs/2502.19680)
- **What's New**: 최근의 Multi-Modal Large Language Models (M-LLMs)의 발전은 비디오 추론(video reasoning)에서 유망한 결과를 보여주고 있습니다. 기존의 M-LLM 프레임워크는 긴 컨텍스트 비디오에서 입력 프레임 수를 줄이기 위해 단순하고 균일한 샘플링 방법을 적용합니다. 그러나 이러한 방법은 비디오의 특정 구간에서 중요한 맥락을 놓칠 수 있습니다. 이를 해결하기 위해, 우리는 사용자의 쿼리와 보다 관련 있는 프레임을 선택하는 경량 M-LLM 기반의 프레임 선택 방법을 제안합니다.

- **Technical Details**: 제안하는 프레임 선택기를 훈련하기 위해 두 가지 감독 신호를 도입합니다: (i) Spatial 신호는 M-LLM을 통해 각 프레임의 중요성 점수를 생성합니다; (ii) Temporal 신호는 LLM을 이용해 모든 프레임 후보의 캡션을 기반으로 다수의 프레임을 선택합니다. 선택된 프레임은 다운스트림 비디오 M-LLM에 의해 시각적 추론과 질문 답변을 위해 처리됩니다. 이러한 방법은 전체 맥락 길이를 유지하면서, 잡음을 줄이고 모델의 초점을 관련 비디오 세그먼트로 향하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안한 M-LLM 비디오 프레임 선택기가 Medium (ActivityNet, NExT-QA)과 Long (EgoSchema, LongVideoBench) 컨텍스트 비디오 질문 응답 벤치마크에서 다양한 다운스트림 비디오 Large Language Model (video-LLM)의 성능을 향상시킵니다. 특히, 우리는 모델이 질문에 답하기 위해 가장 도움이 되는 프레임에 집중함으로써 비디오 이해의 품질을 유지하면서 처리해야 하는 시각적 컨텍스트를 상당히 줄일 수 있음을 보여주었습니다.



### Towards Differential Handling of Various Blur Regions for Accurate Image Deblurring (https://arxiv.org/abs/2502.19677)
- **What's New**: 이번 논문에서는 Differential Handling Network (DHNet)을 제안하여 이미지의 다양한 흐림 영역을 효율적으로 처리할 수 있는 방법을 제시합니다. DHNet는 비선형 특성을 통합하기 위해 Volterra block (VBlock)을 설계하여, 기존의 비선형 활성화 함수들이 쌓이는 것을 피하고 복잡한 입력-출력 관계를 매핑하는 데 드는 연산 비용을 줄입니다. 또한, 모델이 흐림의 정도를 적응적으로 인식할 수 있도록 하는 degradation degree recognition expert module (DDRE)을 마련하였습니다.

- **Technical Details**: DHNet의 핵심 요소로는 VBlock과 DDRE가 있습니다. VBlock은 전통적인 비선형 활성화 함수를 사용하지 않고, 볼테라 커널을 통해 이미지 픽셀 간의 상호작용을 촉진하여 선형 컨볼루션을 강화하는 방식으로 비선형성을 탐구합니다. DDRE는 잘 훈련된 모델의 사전 지식을 통합하여 공간적으로 가변적인 흐림 정보를 추정하며, 이를 통해 라우터가 학습된 흐림 표현을 맵핑하고 전문가에게 가중치를 할당할 수 있도록 합니다.

- **Performance Highlights**: DHNet은 광범위한 실험을 통해 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다. 실험 결과는 DHNet이 합성 및 실제 데이터셋에서 SOTA 성능을 달성했음을 입증합니다. 이 모델은 기존 방법들에 비해 더 적은 연산 비용으로도 혁신적인 결과를 만들어내어 성능 개선을 이룹니다.



### MICINet: Multi-Level Inter-Class Confusing Information Removal for Reliable Multimodal Classification (https://arxiv.org/abs/2502.19674)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 논문에서는 다중 모달 데이터에서 동시 존재하는 모달리티 특정 노이즈(modality-specific noise)와 교차 모달리티 노이즈(cross-modal noise)를 동시에 제거하는 새로운 접근법인 Multi-Level Inter-Class Confusing Information Removal Network (MICINet)을 소개합니다. MICINet은 두 가지 유형의 노이즈를 Inter-class Confusing Information (ICI)라는 개념으로 통합하고 글로벌 및 개별 수준에서 효과적으로 제거합니다. 이를 통해 안전-critical 애플리케이션에서의 신뢰성을 높이고자 하였습니다.

- **Technical Details**: MICINet은 ICI 분포를 학습하는 Global ICI Learning Module과 샘플 특성에서 글로벌 수준의 ICI를 제거하는 Global-guided Sample ICI Learning 모듈을 활용합니다. 또한, Sample-adaptive Cross-modality Information Compensation 모듈을 통해 각 샘플에서 개별 수준의 ICI를 제거하며, 이는 주의 메커니즘을 이용해 다양한 모달리티 간의 보완적 관계를 활용하여 수행합니다. 이 두 단계 파이프라인은 신뢰할 수 있는 노이즈 제거를 위한 새로운 접근법을 제공합니다.

- **Performance Highlights**: MICINet은 네 개의 데이터셋에서 실험을 통해 기존의 최첨단 다중 모달 분류 방법들과 비교하여 더 높은 정확도와 신뢰성을 보였습니다. 특히, 특정 모달리티의 노이즈와 교차 모달리티 노이즈가 동시에 존재하는 상황에서도 뛰어난 성능을 발휘하는 것을 확인할 수 있었습니다. 이로 인해 MICINet은 안전-critical한 응용 분야에서의 다중 모달 학습 방법의 기준을 새롭게 설정하였습니다.



### SubZero: Composing Subject, Style, and Action via Zero-Shot Personalization (https://arxiv.org/abs/2502.19673)
- **What's New**: SubZero는 사용자 맞춤형 주제, 스타일 및 행동 조합을 위한 새로운 제안입니다. 이 프레임워크는 미세 조정 없이 주제를 어떤 스타일로도 생성할 수 있는 혁신적인 접근 방식을 사용합니다. 이를 통해 Edge device에서도 실행 가능하며, 많은 계산 비용 없이 고품질 이미지를 생성할 수 있습니다.

- **Technical Details**: SubZero는 주제와 스타일 사이의 유사성을 향상시키기 위해 새로운 제약 조건을 제안합니다. 교차 주의 모듈 내에서 주제 및 스타일 정보를 일관되게 결합하기 위해 시공간 적응 기법을 적용하며, 커스터마이즈된 내용 및 스타일 프로젝터를 통해 내용과 스타일 누수를 줄입니다. 이로 인해 사용자는 단일 주제 및 스타일 이미지로도 작업할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 SubZero는 주제, 스타일 및 행동 조합에 있어 이전의 최첨단 작업보다 뛰어난 성능을 보여줍니다. 특히, 이 접근 방식은 주제 카드, 동물 및 얼굴 개인화 같은 다양한 이미지 생성 작업에서 결실을 맺었습니다. 실험 결과, 해당 프레임 워크는 현장 적용에 적합한 효율적인 실행을 제공합니다.



### Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack (https://arxiv.org/abs/2502.19672)
Comments:
          arXiv admin note: text overlap with arXiv:2403.09766

- **What's New**: 최근 미디엄 대형 언어 모델(MLLMs)에 대한 관심이 커지고 있으며, 이러한 모델들은 이미지 인식 및 이해 능력에서 주목받고 있습니다. 하지만 MLLM들은 적대적 공격에 취약하며, 이러한 공격이 다른 모델에서 전이되는 능력은 제한적입니다. 본 연구에서는 다이나믹 비전-언어 정렬(DynVLA) 공격을 소개하여 다른 모델 간의 비전-언어 정렬을 개선하고자 합니다.

- **Technical Details**: DynVLA 공격은 비전-언어 커넥터에 동적인 섭동(perturbations)을 주입하여 MLLM의 비전-언어 모달리티 정렬을 동적으로 변화시킵니다. 기존의 방법들은 단일 비전-언어 정렬에 기반한 엔드-투-엔드 최적화 방식을 사용하지만, DynVLA는 주의(attention) 메커니즘을 변경하여 다양한 비전-언어 모달리티 정렬을 수용합니다. 이는 가우시안 커널(Gaussian kernel)을 주의 맵에 적용하여 모델의 주의를 이미지의 다른 영역으로 이동시킵니다.

- **Performance Highlights**: DynVLA는 BLIP2, InstructBLIP, MiniGPT4, LLaVA 등 다양한 MLLM에서 적대적 예제의 전이 가능성을 크게 향상시킵니다. 우리의 방법은 DIM, SIA와 같은 기존의 전통적 공격 방법들과 비교하여 뛰어난 성능을 보이며, 이는 특히 MLLM의 아키텍처와 크기가 전이 가능성에 중요한 역할을 한다는 것을 보여줍니다. 결과적으로, DynVLA는 최소한의 사전 지식으로도 모델을 공격할 수 있어, 실질적인 보안 위협을 야기할 수 있는 가능성을 가지고 있습니다.



### Test-Time Modality Generalization for Medical Image Segmentation (https://arxiv.org/abs/2502.19671)
Comments:
          28 pages and 15 figures. arXiv admin note: text overlap with arXiv:2502.09931

- **What's New**: 이번 연구에서는 임상 환경에서의 일반화 성능을 향상시키는 혁신적인 Test-Time Modality Generalization (TTMG) 프레임워크를 도입합니다. TTMG는 Modality-Aware Style Projection (MASP)와 Modality-Sensitive Instance Whitening (MSIW)라는 두 가지 핵심 구성 요소로 구성됩니다. 이 프레임워크는 새로운 환자의 사례에서도 높은 일반화 성능을 보이며, 다양한 모달리티 데이터 세트에서 적용 가능합니다.

- **Technical Details**: TTMG는 특정 모달리티에 대한 이해를 바탕으로 테스트 인스턴스의 모달리티를 추정하여, 적절한 스타일 분포로 매핑합니다. MASP는 테스트 인스턴스를 가장 유사한 스타일 공간으로 투영하고, MSIW는 훈련 중 모달리티 민감한 정보를 선택적으로 억제합니다. 이 두 가지 과정을 통해 TTMG는 새로운 모달리티에 대한 뛰어난 일반화 성능을 달성할 수 있습니다.

- **Performance Highlights**: TTMG 프레임워크는 11개의 데이터세트에서 기존의 도메인 일반화 기술들보다 뛰어난 성능을 보였습니다. 다양한 모달리티 조합에서 일관된 세그멘테이션 성능을 발휘하여, 의료 이미지 세그멘테이션 분야에서 변별력을 제공합니다. 이는 기존 방법들이 간과해온 도전 과제에 대한 해결책을 제시합니다.



### Noise-Injected Spiking Graph Convolution for Energy-Efficient 3D Point Cloud Denoising (https://arxiv.org/abs/2502.19660)
Comments:
          Accepted by AAAI 2025

- **What's New**: 스파이킹 신경망(Spiking Neural Networks, SNNs)은 생물학적 신경 시스템의 스파이킹 계산 패러다임에 영감을 받아, 전통적인 인공 신경망(Artificial Neural Networks, ANNs)보다 2D 분류 작업에서 에너지 효율성이 뛰어난 성능을 보였습니다. 그러나 3D 포인트 클라우드(3D point cloud)에서의 회귀 가능성(regression potential)은 잘 탐구되지 않았습니다. 본 논문에서는 소음 주입 스파이킹 그래프 컨볼루션 네트워크(noise-injected spiking graph convolutional networks)를 제안하여, 3D 포인트 클라우드에서의 노이즈 제거(denoising)를 위해 SNN의 회귀 가능성을 극대화하고자 하였습니다.

- **Technical Details**: 우리는 노이즈가 주입된 스파이킹 뉴런을 구축하기 위해 노이즈 주입 neuronal dynamics를 모사하고, 3D 포인트에 대한 방해 인식 스파이킹 표현 학습을 촉진하는 노이즈 주입 스파이킹 그래프 컨볼루션을 설계했습니다. 두 가지 SNN 기반의 노이즈 제거 네트워크를 구축했으며, 하나는 순수한 스파이킹 그래프 컨볼루션 네트워크로, 다른 하나는 ANN 기반의 학습과 높은 성능-효율성(trade-off)을 결합한 하이브리드 아키텍처입니다.

- **Performance Highlights**: 제안된 네트워크 모델은 수치 정확도 손실을 낮추면서도 두 가지 벤치마크 데이터셋인 PU-Net과 PC-Net에서 에너지 소비를 크게 줄입니다. 우리의 연구는 SNN을 사용한 에너지 효율적인 포인트 클라우드 노이즈 제거의 첫 사례이며, PU-Net과 PC-Net 데이터 세트에서 높은 정확도를 유지합니다. 이 연구는 신경형 칩(neuromorphic chips)에의 배포 가능성을 탐구하며, 에너지 효율적인 3D 데이터 수집 장치를 개발하는 데 기여할 것입니다.



### Adaptive Score Alignment Learning for Continual Perceptual Quality Assessment of 360-Degree Videos in Virtual Reality (https://arxiv.org/abs/2502.19644)
Comments:
          Accepted as a TVCG paper at VR 2025

- **What's New**: 이번 연구에서는 새로운 VR 비디오 품질 평가(VR-VQA) 방법론인 Adaptive Score Alignment Learning (ASAL)을 제안합니다. ASAL은 인간 주관적 평가와의 정렬을 강화하고, 지각 품질을 예측하는 정밀도를 높이기 위해 상관 관계 손실(correlation loss)과 오차 손실(error loss)을 통합합니다. 이 방법은 지속적으로 변화하는 비디오 분포에 자연스럽게 적응할 수 있는 기능 공간 smoothing 프로세스를 통해 새로운 콘텐츠에 대한 일반화를 향상시킵니다.

- **Technical Details**: VR-VQA는 360도 비디오의 지각 품질 점수(s^i) 예측을 목표로 하며, 핵심은 사용자의 몰입 경험에 기반한 신경망(neural network) 구조입니다. ASAL은 기존의 비정상적(non-stationary) 변화를 해결하기 위해 키 프레임(key frame) 추출과 특징 적응(feature adaptation)을 활용한 적응형 메모리 재생(adaptive memory replay)을 포함합니다. 이는 VR 장치의 처리 및 저장 제약을 고려하여 설계된 새로운 평생 학습(Continual Learning, CL) 접근 방식입니다.

- **Performance Highlights**: 실험 결과, ASAL은 최신 강력한 기준 모델에 비해 최대 4.78%의 상관 관계 향상을 달성했습니다. 정적 조인트 훈련(static joint training) 설정에서 및 동적인 CL 설정에서 각각 12.19% 개선된 성과를 보여주었습니다. 이는 ASAL이 VR-VQA의 고유한 문제를 해결하는 데 효과적임을 입증합니다.



### MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning (https://arxiv.org/abs/2502.19634)
- **What's New**: 이번 연구에서는 MedVLM-R1을 통해 의료 이미지를 분석하는 새로운 방법을 제시합니다. 이 모델은 최종 답변을 생성하는 데 그치지 않고, 명확한 자연어 추론을 제시함으로써 투명성과 신뢰성을 강화합니다. 기존의 Supervised Fine-Tuning (SFT)에 의존하지 않고, 강화 학습(RL) 프레임워크를 활용하여 자체적으로 인간이 해석할 수 있는 추론 경로를 발견하도록 유도합니다.

- **Technical Details**: MedVLM-R1은 기존의 Supervised Fine-Tuning의 한계를 극복하기 위해 Group Relative Policy Optimization (GRPO) 알고리즘을 채택하였습니다. 이는 모델이 최종 답변을 암기하거나 모방하는 대신, 자신의 논리적 단계를 발견하는 데 보상을 제공함으로써 자발적인 추론 능력을 촉진합니다. 이 모델은 단지 600개의 샘플과 2B의 모델 매개변수로 제한된 데이터에서도 효과적으로 훈련됩니다.

- **Performance Highlights**: 훈련 결과, MedVLM-R1은 MRI, CT 및 X-ray 벤치마크에서 정확도가 55.11%에서 78.22%로 증가했습니다. 또한, MedVLM-R1은 기존의 대형 모델들보다 뛰어난 일반화 능력을 보여주었으며, 현업에서의 적용 가능성을 높였습니다. 일반화는 특히 unfamiliar data에 대해 강력한 성능을 입증하며, 신뢰할 수 있는 임상 AI로 나아가는 중요한 발걸음을 제공합니다.



### Ev-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras (https://arxiv.org/abs/2502.19630)
Comments:
          Accepted by CVPR2025

- **What's New**: 이번 연구에서는 무선 LiDAR 및 카메라와 함께 비동기 이벤트 카메라를 도입하여 3D 객체 감지의 새로운 접근 방식을 제안합니다. 이를 통해 높은 시간 해상도와 낮은 대역폭을 활용하여 3D 객체 감지의 속도와 신뢰성을 높이는 것이 목표입니다. 또한, 100 FPS에서 지상 진실 3D 바운딩 박스가 포함된 새로운 데이터셋 DSEC-3DOD를 발표하여 이벤트 기반 3D 감지기의 첫 벤치마크를 설정했습니다.

- **Technical Details**: 비동기 이벤트 카메라를 사용하여 고속 3D 객체 감지를 가능하게 함으로써, 기존의 고정 프레임 속도 센서의 한계를 극복합니다. 특히, 이벤트 데이터는 동기화된 데이터가 없을 때에도 이전 3D 정보를 통해 객체의 동작을 추정할 수 있습니다. 또한, 3D 복셀 공간에 이벤트 특징을 투영하고, 이를 통해 생성된 암시적 운동 필드를 활용하여 3D 운동 벡터를 계산합니다.

- **Performance Highlights**: 제안된 방법은 블라인드 타임 동안에도 3D 객체 감지를 가능하게 하여 자율 주행 시스템의 안전성을 더욱 향상시키는 데 기여할 것으로 기대됩니다. 실제 이벤트 데이터가 포함된 DSEC-3DOD는 기존의 데이터를 보완하여 블라인드 타임 동안의 정보도 포함하여 신뢰성을 높였습니다. 이 연구는 이벤트 카메라를 활용한 3D 객체 감지 분야에서 새로운 지평을 열고 있으며, 앞으로의 연구에 큰 기여를 할 것으로 보입니다.



### 3D Nephrographic Image Synthesis in CT Urography with the Diffusion Model and Swin Transformer (https://arxiv.org/abs/2502.19623)
Comments:
          15 pages, 6 figures, 3 tables

- **What's New**: 이번 연구는 CT urography (CTU) 검사를 위한 3D nephrographic phase 이미지를 합성하는 방법을 개발하고 검증하는 것을 목표로 하고 있습니다. Swin Transformer 기반의 딥러닝 접근법과 diffusion model을 통합하여, 새로운 합성 모델인 dsSNICT를 제안하고 있습니다.

- **Technical Details**: 연구는 327명의 환자로 구성된 데이터를 사용하였으며, 각 환자의 세 단계 이미지는 affine registration 알고리즘을 통해 정렬되었습니다. dsSNICT 모델은 합성 nephrographic 이미지를 생성하기 위해 설계되었으며, 성능 평가는 PSNR, SSIM, MAE, FVD 지표를 사용하여 수행되었습니다.

- **Performance Highlights**: 제안된 접근 방식으로 생성된 합성 nephrographic 이미지는 PSNR 26.3 dB, SSIM 0.84, MAE 12.74 HU, FVD 1323이라는 높은 성능 지표를 달성하였습니다. 방사선 전문의 두 명의 평가에서는 Likert 척도에서 실제 이미지에 평균 3.5, 합성 이미지에 평균 3.4의 점수를 기록하였으며, 이는 높은 유사성을 나타내고 있습니다.



### Tell me why: Visual foundation models as self-explainable classifiers (https://arxiv.org/abs/2502.19577)
- **What's New**: 본 연구에서는 시각 기초 모델(Visual Foundation Models, VFM)과 새로운 프로토타입 아키텍처(prototypical architecture)를 결합하여 해석 가능한 분류기를 제안합니다. 이 모델은 예측을 해석 가능한 개념들의 가중합으로 분해하여 자가 설명(self-explainable)을 목표로 합니다. 이와 같은 접근법이 기존 모델보다 더 효율적이고 해석 가능하다는 점이 특히 주목할 만합니다.

- **Technical Details**: ProtoFM이라는 방법론은 고정된 VFM 위에 가벼운 헤드(약 1M 파라미터)를 훈련시키는 방식을 채택합니다. 전문화된 훈련 목표(specialized training objectives)를 통해 해석 가능성을 증대시키고, 예측의 신뢰성을 확보하는 데 중점을 두고 있습니다. 이 모델은 기존의 VFM과 비교해 훨씬 적은 파라미터를 사용하면서도 효과성을 유지합니다.

- **Performance Highlights**: 평가 결과에 따르면, ProtoFM은 경쟁력 있는 분류 성능(classification performance)을 달성하며 해석 가능성 메트릭(interpretabiliy metrics)에서도 기존 모델들을 초월했습니다. 연구에 사용된 해석 가능성 관련 지표는 문헌에서 파생된 것입니다. 코드도 제공되어 있어 연구자들이 쉽게 활용할 수 있습니다.



### Dictionary-based Framework for Interpretable and Consistent Object Parsing (https://arxiv.org/abs/2502.19540)
- **What's New**: 본 논문에서는 CoCal이라는 해석 가능한 객체 파싱(interpretable object parsing) 프레임워크를 제안합니다. 이 프레임워크는 딕셔너리 기반의 마스크 변환기(dictionary-based mask transformer)를 기반으로 하며, 대조 구성 요소(Contrastive Components)와 논리적 제약(Logical Constraints)을 활용하여 기존 클러스터 기반 마스크 변환 아키텍처를 재구성합니다. CoCal은 각 구성 요소를 특정 의미(class)에 명시적으로 연결하여 객체 파싱의 해석 가능성을 높입니다.

- **Technical Details**: CoCal은 의미 계층(semantic hierarchy)에 맞춘 딕셔너리 구성 요소를 계층적으로 배치하여, 성능을 최적화하는데 중점을 둡니다. 특히, CoCal은 각 의미 레벨에서 구성 요소 대조 알고리즘을 사용하여 동일 클래스 내의 딕셔너리 구성 요소를 가까이 두고, 다른 클래스의 구성 요소와는 멀리 하는 방식으로 유도합니다. 또한, 이미지 내 픽셀을 특정 클래스에 할당할 때, 해당 부분이 속하는 객체와의 논리적 관계를 고려하여 더 나은 예측을 가능하게 합니다.

- **Performance Highlights**: CoCal은 PartImageNet과 Pascal-Part-108 두 벤치마크에서 각각 2.08%와 0.70%의 성능 향상을 달성하며 새로운 최첨단 성능(state-of-the-art)을 기록했습니다. 또한, CoCal은 객체 수준(object-level) 메트릭에서도 눈에 띄는 향상을 보여줍니다. 이는 객체 분할의 전반적인 품질을 높일 뿐만 아니라 세밀한 분석을 통해 객체의 의미를 더욱 정확하게 해석할 수 있게 합니다.



### Evaluating the Suitability of Different Intraoral Scan Resolutions for Deep Learning-Based Tooth Segmentation (https://arxiv.org/abs/2502.19515)
Comments:
          accepted to 2025 ASEE North Central Section Annual Conference

- **What's New**: 이번 연구에서는 intraoral 스캔을 활용한 치과의 디지털 프로세스를 자동화하기 위한 딥러닝 모델(PointMLP)을 제안합니다. 기존에 비해 스캔의 해상도를 감소시키는 과정에서의 세그멘테이션(Segmentation) 성능 저하 정도를 평가하였습니다. 연구의 목적은 계산 효율성과 세그멘테이션 정확도 간의 균형을 맞출 수 있는 최적 해상도를 찾는 것입니다.

- **Technical Details**: PointMLP 방법은 지역 기하 구조를 효과적으로 캡처할 수 있는 기하학적 어파인 모듈을 사용합니다. 입력 포인트 클라우드에서 지역 특징을 단계적으로 추출하며, 이 과정에서 잔여 포인트 MLP 블록(residual point MLP blocks)을 활용합니다. 본 연구에서는 2K, 4K, 6K, 8K, 10K, 16K 해상도로 변환된 메쉬를 사용하여 모델을 훈련하였습니다.

- **Performance Highlights**: PointMLP의 세그멘테이션 성능은 다양한 해상도에서 평가되었습니다. 테스트 결과, 낮은 해상도로 훈련된 모델이 높은 해상도의 스캔에서도 상당한 성능을 보이는 것으로 나타났습니다. 연구 결과는 세그멘테이션의 클래스별 정확도를 포함하고 있으며, 해상도가 성능에 미치는 영향을 정량적으로 분석하였습니다.



### CLIP-Optimized Multimodal Image Enhancement via ISP-CNN Fusion for Coal Mine IoVT under Uneven Illumination (https://arxiv.org/abs/2502.19450)
- **What's New**: 이 연구는 석탄 광산 IoVT(Internet of Video Things) 시스템을 위해 설계된 다중 모드 이미지 향상 방법을 제안합니다. 제안된 방법은 ISP(CNN) 퓨전 아키텍처를 기반으로 하여 불균형 조명 문제를 해결합니다. 두 단계의 전략을 통해 전역 향상 및 세부 최적화를 결합, 특히 조명이 어두운 지역에서 이미지 품질을 효과적으로 개선합니다.

- **Technical Details**: 제안하는 접근법은 전통적인 이미지 신호 처리(ISP)와 합성곱 신경망(CNN)을 결합하여 계산 복잡성을 줄이고 성능을 높이는 데 중점을 둡니다. 다중 모드 대비 학습을 기반으로 한 반복 최적화 방법을 통해 비지도 학습이 구현됩니다. 이는 조명의 불균형 문제를 해결하기 위해 텍스트와 세부 이미지 특징 간의 상관관계를 수립합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 첨단 알고리즘에 비해 PSNR(신호 대 잡음 비율)을 2.9%-4.9%, SSIM(구조적 유사도 지수)를 4.3%-11.4%, VIF(시각 정보 충실도)를 4.9%-17.8% 향상시키는 것으로 나타났습니다. 시뮬레이션된 석탄 광산 모니터링 시나리오에서 이미지를 실시간으로 개선할 수 있는 능력이 입증되었습니다.



### Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids (https://arxiv.org/abs/2502.20396)
Comments:
          Project page can be found at this https URL

- **What's New**: 이 연구는 휴머노이드 형태에서 접촉이 많은 조작 작업에 대해 강화 학습을 적용하는 데 따른 핵심 도전 과제를 조사합니다. 특히, 실제 환경과 시뮬레이션 환경을 맞추기 위한 자동화된 real-to-sim 튜닝 모듈, 장기 접촉 조작 작업을 위한 일반화된 보상 설계, 그리고 탐색 문제의 샘플 효율성을 개선하는 분할 및 정복(distillation) 과정을 도입합니다.

- **Technical Details**: 연구에서는 비전 기반의 조작 작업을 위해 강화 학습 정책을 설계하고 교육하는 데 있어서 보상 함수를 개별 접촉 목표(contact goals)와 객체 목표(object goals)로 분해하는 일반 원칙을 제안합니다. 또한, 복잡한 탐색 문제에 대처하기 위해 과제를 알고리즘적으로 감소시켜 접근하는 두 가지 실용적 기술을 도입하여 수학적 증명을 통해 샘플 효율성을 개선했습니다.

- **Performance Highlights**: 3가지 휴머노이드 섬세 조작 작업에 대해 성공적인 결과를 도출했으며, 각 기술에 대한 ablation 연구를 통해 제안한 접근 방식의 효과를 검증했습니다. 이 연구는 인간의 시연 없이도 robuste한 일반화와 높은 성능을 달성하는 sim-to-real 강화 학습을 통한 휴머노이드 섬세 조작 학습을 위한 성공적인 방법을 제시합니다.



### Walking the Web of Concept-Class Relationships in Incrementally Trained Interpretable Models (https://arxiv.org/abs/2502.20393)
Comments:
          8 pages of main text, 6 figures in main text, 11 pages of Appendix, published in AAAI 2025

- **What's New**: 이 논문에서는 개념 기반 모델(concept-based models)이 점진적인 학습(incremental learning) 환경에서도 잘 작동할 수 있도록 재설계되었음을 강조합니다. 특히, 새로 생성된 클래스가 이전 개념뿐만 아니라 새로운 개념을 사용할 수 있도록 하는 동적(dynamic) 설정에서 이 모델을 연구합니다. 이를 통해 개념과 클래스의 복잡한 관계 망을 유지하고 증대할 필요성을 보여줍니다.

- **Technical Details**: 새로운 툴인 MuCIL(Multimodal Concept-Based Incremental Learner)을 소개하며, 이 방법은 항상 새로운 클래스를 수용할 수 있는 능력을 지니고 있습니다. MuCIL은 텍스트 개념(text concepts)과 이미지 표현(image representations) 간의 임베딩(embedding)을 결합하여 다중 모달 개념 임베딩(multimodal concept embeddings)을 생성합니다. 이 구조는 분류(classification) 성능을 보장하고, 안정적인 해석을 제공하는 데 필요한 정보도 포함하고 있습니다.

- **Performance Highlights**: 실험을 통해 MuCIL이 기존의 개념 기반 모델보다 2배 이상 높은 분류 성능(classification performance)을 달성했음을 보여줍니다. 또한, 모델의 개념에 대한 개입(intervention) 능력을 연구하여 입력 이미지 내에서 시각적 개념을 국소화(localization)할 수 있음을 입증하였습니다. 이 접근방법은 사후 해석(post-hoc interpretations)을 제공할 수 있는 잠재적인 방법으로서, 다양한 벤치마크 데이터세트에서 최첨단의 성과를 달성하였습니다.



### Tight Inversion: Image-Conditioned Inversion for Real Image Editing (https://arxiv.org/abs/2502.20376)
Comments:
          Project page at: this https URL

- **What's New**: 이 논문은 텍스트-이미지 확산 모델의 이미지 편집 능력을 대폭 향상시키기 위한 새로운 접근 방식을 제시합니다. 기존의 많은 방법들이 이미지를 가우시안 노이즈로 변환하여 편집하는 데 의존하고 있지만, 이 연구는 이러한 변환 과정에서 조건 선택의 중요성을 강조하고 있습니다.

- **Technical Details**: 저자들은 입력 이미지와 정확히 일치하는 조건을 사용함으로써 인버전(inversion) 품질이 크게 향상될 수 있음을 보여줍니다. 이들은 'Tight Inversion'이라는 새로운 인버전 방법을 도입하며, 이는 가장 정확한 조건인 입력 이미지 자체를 활용하여 모델 출력의 분포를 좁히고 재구성(재현) 및 편집 능력을 동시에 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 기존 인버전 방법과 결합했을 때 접근 방식의 효과를 입증하였으며, 재구성 정확도와 다양한 편집 방법과의 통합을 평가하였습니다. 이를 통해 복잡한 이미지 편집의 효율성을 크게 개선할 수 있음을 보여줍니다.



### T1-PILOT: Optimized Trajectories for T1 Mapping Acceleration (https://arxiv.org/abs/2502.20333)
- **What's New**: 이 논문에서는 T1-PILOT라 불리는 새로운 파이프라인을 소개합니다. 이 방법은 T1 신호 완화 모델을 샘플링-재구성 프레임워크에 명시적으로 통합하여 비-Cartesian(Non-Cartesian) 궤적 학습을 안내합니다. T1-PILOT는 실험을 통해 기존 방법들보다 더 높은 T1 맵 충실도를 달성하며, 이로 인해 정량적 정확성이 향상되고 촬영 시간도 단축된다는 것을 보여줍니다. 이 혁신적인 접근 방식은 병리학 진단에 중요한 자료인 심장 MRI T1 맵 생성에 있어 중요한 진전을 나타냅니다.

- **Technical Details**: T1-PILOT는 심장 조직 T1 맵을 추정하기 위해 여러 T1 가중치 이미지를 기반으로 하고 있습니다. 이 과정에서 T1 완화 곡선을 적합하기 위해 지수형 T1-감쇠 곡선을 사용하는 것을 목표로 합니다. 이는 인공 신경망을 통해 감쇠 파라미터 A, B, T1*를 효율적으로 최적화하고, 학습된 k-space 언더샘플링 마스크를 통해 T1 맵 추정 속도를 높입니다. 이러한 접근은 고도로 언더샘플링된 데이터에서도 정확한 T1 맵 생성을 가능케 하여, 기존 방법들과의 성능 차별성을 보여줍니다.

- **Performance Highlights**: CMRxRecon 데이터세트에서 T1-PILOT는 고정된 방사형 및 골든 앵글 샘플링 방식, 그리고 단일 학습 궤적 등 여러 기준 전략들과 비교하여 뛰어난 성능을 보였습니다. PSNR(픽셀 신호 대 잡음 비율)와 VIF(구조적 유사성 지표)에서 높은 수치를 기록하며, 심장 조직의 미세 구조를 더 잘 활용하는 것으로 나타났습니다. 이러한 결과는 T1 완화 신호를 명시적으로 모델링함으로써 인수요정확성과 촬영 시간 모두에서 개선을 가져옴을 입증하고 있습니다.



### Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription (https://arxiv.org/abs/2502.20295)
Comments:
          11 pages (including references and appendix), 14 figures, accepted at AAAI-25 Workshop on Document Understanding and Intelligence, non-archival

- **What's New**: 이 논문은 다중 페이지의 손글씨 문서를 제로샷(zero-shot) 설정에서 전사(transcribe)하는 데 있어 다중 모달 대형 언어 모델(MLLMs)의 활용을 탐구합니다. 기존의 OCR 엔진이 인쇄된 텍스트에 강력한 성능을 보이는 반면, 손글씨 처리에서는 제한적이기 때문에 MLLMs를 엔드 투 엔드 전사기로 활용하거나 후처리(post-process)기로 사용하는 다양한 구성에 대해 고찰합니다. 주목할 만한 점은, '+first page'라는 새로운 방법을 제안하며, 이는 전체 문서의 OCR 출력을 제공하면서 첫 페이지 이미지만 활용하여 MLLM 전사 정확도를 높입니다.

- **Technical Details**: 결과적으로, MLLM은 손글씨 문서의 여러 페이지에 걸쳐 공통적인 서식(formatting) 및 맥락적(feature) 정보를 활용합니다. OCR 시스템은 페이지 단위로 동작하는 반면, MLLM은 이러한 페이지 간의 종속성을 고려하여 전사 작업을 향상시킵니다. 이 연구에서는 IAM 손글씨 데이터베이스를 사용하여 제안된 방법의 효과를 검증하고, MLLM이 단일 페이지에서 배운 서식 및 OCR 오류 패턴을 활용하여 전체 문서를 향상시키는 것을 확인합니다.

- **Performance Highlights**: 실험 결과, '+first page' 접근 방식이 전사 정확도를 개선하고 비용과 성능 간의 균형을 이룬다는 것이 나타났습니다. 또한, 이 방법은 고가의 MLLM과 비교적 저렴한 OCR 방법 간의 제휴를 통해, 문서 내의 서식과 오류 패턴을 외삽(extrapolate)하여 성능을 향상시켰습니다. 이러한 결과는 다중 페이지 손글씨 문서 전사 작업에서 MLLM의 가능성을 제시하며, 향후 다양한 분야에서 활용될 수 있습니다.



### RURANET++: An Unsupervised Learning Method for Diabetic Macular Edema Based on SCSE Attention Mechanisms and Dynamic Multi-Projection Head Clustering (https://arxiv.org/abs/2502.20224)
Comments:
          10 pages, 2 figures, 5 tables, submitted to The 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2025)

- **What's New**: 이 논문에서는 당뇨병 환자들 사이에서 흔히 발생하는 합병증인 당뇨병성 황반부종(Diabetic Macular Edema, DME) 진단을 위한 새로운 자동화 시스템인 RURANET++를 소개합니다. 이 시스템은 기존의 데이터 주석 및 주관적인 안과 의사 평가에 의존하지 않고, 비지도 학습(un) 기반으로 설계되었습니다. DME 진단의 실용적인 응용을 위해 기존의 문제들을 해결하는 방안을 제안합니다.

- **Technical Details**: RURANET++는 최적화된 U-Net 아키텍처와 공간적 및 채널 압축 및 자극(SCSE) 주의 메커니즘을 결합하여 병변 특성 추출을 개선합니다. 이 시스템은 사전 훈련된 GoogLeNet 모델을 사용하여 망막(retinal) 이미지에서 깊은 특징을 추출하고, PCA(Principal Component Analysis)를 활용하여 효율성을 위해 50차원으로 차원 축소를 수행합니다. 또한, 다중 투영 헤드를 사용하는 새로운 클러스터링 알고리즘을 도입하여 클러스터 다양성을 제어하고 유사성 임계값을 동적으로 조정하여 클래스 내 일관성과 클래스 간 차별성을 최적화합니다.

- **Performance Highlights**: 실험 결과는 여러 지표에서 뛰어난 성능을 보여주며, 최대 정확도(accuracy)는 0.8411, 정밀도(precision)는 0.8593, 재현율(recall)은 0.8411, F1 점수(F1-score)는 0.8390에 도달했습니다. 이러한 결과는 뛰어난 클러스터링 품질(clustering quality)과 함께 비지도 진단(un) 솔루션의 효율성을 입증합니다. 이 연구는 DME 진단에 중요한 임상적 의미를 제공합니다.



### Representing Signs as Signs: One-Shot ISLR to Facilitate Functional Sign Language Technologies (https://arxiv.org/abs/2502.20171)
- **What's New**: 이 논문에서는 언어 독립적인 신규 접근 방식으로서의 일회성(one-shot) 수화 인식(Isolated Sign Language Recognition, ISLR) 기술을 제안한다. 전통적인 언어 특정 접근법의 한계를 극복하고, 다양한 언어와 진화하는 어휘에 걸쳐 일반화 가능한 방법을 제시하였다. 특히, 사전 훈련(pretraining)된 모델을 활용하여 수화를 효율적으로 임베딩하고 새로운 수화를 신속하고 정확하게 인식할 수 있도록 하는 밀집 벡터 검색(dense vector search) 방식을 도입했다.

- **Technical Details**: 우리의 접근 방식은 두 단계로 구성된다. 첫 번째 단계에서는 다양한 수화를 대표할 수 있는 모델을 사전 훈련하여 그 필수 특징을 캡처한다. 두 번째 단계에서는 이 모델을 동결(frozen) 상태로 유지하여, 신규 수화를 기존의 수화에 대한 임베딩과 밀접하게 매핑하여 인식의 정확도를 향상시킨다. 10,235개의 독특한 수화를 포함하는 대규모 사전에서 50.8%의 일회성 MRR(Mean Reciprocal Rank)이라는 최첨단 결과를 도출했다.

- **Performance Highlights**: 이 연구는 수화 기술의 확장성과 효과를 향상시킬 수 있는 진보적 솔루션을 제공하며, 여러 언어에 걸쳐 강력한 일반화를 보여준다는 점에서 그 성과가 빛난다. 수화의 진화적인 성격에 적응할 수 있는 능력을 바탕으로 막대한 어휘의 인식을 가능하게 한다. 또한, DHH(청각장애인 및 난청자) 커뮤니티와의 공동 창작 전략을 통해 실제 필요에 부합하는 도구를 제공할 수 있다는 점에서 긍정적인 평가를 받았다.



### Balanced Rate-Distortion Optimization in Learned Image Compression (https://arxiv.org/abs/2502.20161)
Comments:
          Preliminary version. Camera ready version and source code will be uploaded later. Accepted to CVPR 2025

- **What's New**: 본 논문에서는 새로운 Learned Image Compression (LIC) 모델을 위한 R-D 최적화를 다루고 있습니다. 기존의 R-D 최적화 접근 방식에서 발생하는 비대칭적인 업데이트 문제를 해결하기 위해, R-D 최적화를 다중 목표 최적화(multi-objective optimization, MOO) 문제로 재구성하고 두 가지 균형 잡힌 최적화 전략을 제안하였습니다. 첫 번째 전략은 코스-투-파인(coarse-to-fine) 그래디언트 강하를 이용하여 LIC 모델을 처음부터 훈련시키는 데 효과적이며, 두 번째 전략은 기존 모델의 미세 조정을 위한 분석적 접근 방식을 제공합니다.

- **Technical Details**: 제안된 방법은 R-D 최적화를 두 가지 목표로 동시 최적화하는 프레임워크를 사용합니다. 첫 번째 솔루션은 코스-투-파인 그래디언트 강하로, 그래디언트 가중치를 반복적으로 개선하여 새로운 모델을 훈련하는 데 사용됩니다. 두 번째 솔루션은 최적화를 평등 제약을 가진 2차 프로그래밍 문제로 재구성하여 그래디언트 가중치를 분석적으로 도출하는 기법을 제공합니다. 이러한 방식은 기존 모델을 정밀하게 조정하는 데 이상적입니다.

- **Performance Highlights**: 실험 결과, 제안된 두 가지 방법은 LIC 모델의 R-D 성능을 향상시키며 약 2%의 BD-Rate 감소를 도출했습니다. 이는 추가적인 훈련 비용이 허용 가능한 범위 내에서 이루어지며, 균형 있고 효율적인 최적화 프로세스를 제공합니다. 본 논문에서 사용된 기법들은 차세대 이미지 압축 기술에 대한 중요한 기여를 할 것으로 기대되며, 코드 또한 공개될 예정입니다.



### FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compu (https://arxiv.org/abs/2502.20126)
- **What's New**: 이 연구에서는 현대의 Diffusion Transformer가 고정된 계산 예산으로 인한 자원 요구로 인해 제약을 받는 문제를 해결하기 위해 동적 전략을 제안합니다. 새롭게 제안된 FlexiDiT 모델은 입력에 따라 가변적인 compute budget을 처리할 수 있도록 설계되었습니다. 이 방법을 통해 모델이 품질 저하 없이 이미지를 생성하면서도 40% 이상의 FLOPs 절감이 가능합니다.

- **Technical Details**: Diffusion 모델은 이미지 생성의 핵심적인 블록으로, 순차적으로 노이즈 샘플을 제거하여 원하는 데이터 분포에서의 샘플을 생성합니다. Diffusion Transformer 모델(DiT)은 이러한 과정에서Transformer 블록을 사용하여 기존의 합성곱층(CNN) 대신 적용됩니다. 이러한 접근 방식은 멀티모달 응용통합과 효율적인 훈련이 가능하게 하며, 계산 복잡도 측면에서도 뛰어난 성능을 보입니다.

- **Performance Highlights**: FlexiDiT 모델은 고정된 버전과 비교했을 때 동일한 품질로 이미지를 생성하면서도 계산 요구량을 75%까지 감소시킬 수 있습니다. 비디오 생성에도 적용 가능하며, FlexiDiT는 다양한 조건부이미지 생성 상황에서도 일반적으로 잘 작동하며 뛰어난 성능을 보장합니다. 이러한 특징으로 FlexiDiT 모델은 최신 하드웨어에서 효율적으로 훈련될 수 있는 잠재력을 지니고 있습니다.



### Forward-Cooperation-Backward (FCB) learning in a Multi-Encoding Uni-Decoding neural network architectur (https://arxiv.org/abs/2502.20113)
- **What's New**: 이 논문에서는 Forward-Cooperation-Backward (FCB) 학습이라는 새로운 학습 기법을 제안하고 있습니다. 이 기법은 인간의 학습 방식을 모방하여, Forward-Forward 방식, 협동(cooperation), 그리고 역전파(backpropagation)를 결합한 것입니다. 그에 따라 새로운 Multi Encoding Uni Decoding (MEUD) 신경망 아키텍처도 설계되었습니다.

- **Technical Details**: FCB 학습은 다단계 아키텍처를 통해 진화하며, MEUD, MEUD-FF, MEUD-Coop, MEUD-FF-Coop 모델을 포함하여 구조적으로 발전하였습니다. 이 신경망들은 협동을 구현하기 위해 특수한 lateral synaptic connection을 사용하며, 여러 인기 있는 데이터셋에서 차원 축소(dimensionality reduction) 성능을 평가 받았습니다.

- **Performance Highlights**: MEUD-FF-Coop 프레임워크는 표준 Autoencoder와 여러 변형 모델들과 비교하여 실험적으로 우수성을 입증하였습니다. 데이터의 원래와 투영된 공간 간의 세부 관계를 보존하는 능력과 차원 축소 후 분류 성능이 다양한 분류 알고리즘을 통해 평가되어 그 품질이 입증되었습니다.



### Generative augmentations for improved cardiac ultrasound segmentation using diffusion models (https://arxiv.org/abs/2502.20100)
- **What's New**: 이번 연구에서는 심장 초음파 데이터 세트에서의 라벨링된 데이터 부족 문제를 해결하기 위해 비조건적 확산 모델을 활용하여 생성적 보강(generative augmentation)을 제안합니다. 이 접근 방식은 기존의 라벨이 있는 데이터셋이 아닌, 라벨이 없는 데이터셋 또는 서로 다른 주석 관행으로 훈련된 데이터셋에서 생성 모델을 이용하여 보강할 수 있는 독특한 장점을 가지고 있습니다. 이 연구는 그러한 생성적 보강을 통해 segmentation 모델의 성능 향상을 이뤄냈으며, 이는 외부 데이터셋에 대한 일반화(generalization) 능력을 대폭 개선하는 데 기여하고 있습니다.

- **Technical Details**: 본 연구에서 사용된 Denoising Diffusion Probabilistic Models(DDPM)은 데이터 분포를 근사화하기 위해 점진적인 노이즈 추가 과정을 역으로 수행하는 생성 모델의 일종입니다. 기존의 GAN(Generative Adversarial Networks)과 비교할 때, DDPM은 훈련 과정의 어려움을 겪지 않고, 이미지 합성에서 더 나은 성능을 보여주는 것으로 알려져 있습니다. 특히, 본 연구에서 제안한 방법은 세그멘테이션 마스크의 주변만 변경됨으로써, 원본 이미지의 중요한 세부 정보가 그대로 남아있도록 했습니다.

- **Performance Highlights**: 제안된 생성적 보강을 활용한 결과, 내부 데이터셋에서 훈련하고 외부 데이터셋에서 테스트할 때 Hausdorff distance에서 20mm 이상의 성능 향상이 일어났습니다. 또한, 자동 박출계수(ejection fraction) 추정을 위한 일치 한계가 경우에 따라 20% 향상되었습니다. 이러한 성과는 기계 학습 모델을 수정하지 않고도 가능하게 되었으며, 실제로 전문가 평가에 따르면 생성된 이미지와 실제 이미지 간의 구별이 어려운 것으로 나타났습니다.



### Multi-Keypoint Affordance Representation for Functional Dexterous Grasping (https://arxiv.org/abs/2502.20018)
Comments:
          The source code and demo videos will be publicly available at this https URL

- **What's New**: 이 논문에서는 기능적 섬세 그리프(Functinal Dexterous Grasping)의 새로운 방안으로 다중 키포인트 수용성 표현(Multi-Keypoint Affordance Representation)을 제안하고 있습니다. 기존의 방법들이 단순한 그립(grip)으로 한정되었던 것과 달리, 이 방법은 작업 중심의 그리프 구성을 직접적으로 인코딩(represent)하여, 시각적 인식과 조작 간의 단절 문제를 해결할 수 있습니다. 특히, 인체 그리프 경험 이미지를 활용한 약한 감독(Weak Supervision) 방식과 대형 비전 모델(Large Vision Model)을 결합하여, 수작업 키포인트 주석(annotation)의 필요성을 줄였습니다.

- **Technical Details**: 제안된 방법은 Contact-guided Multi-Keypoint Affordance (CMKA)와 Keypoint-based Grasp matrix Transformation (KGT)을 포함합니다. CMKA는 LVM을 활용하여 정밀한 수용성 기능(feature) 추출을 수행하고, KGT는 손과 물체의 키포인트 간의 일관된 매핑(mapping)을 보장함으로써 섬세한 그리프 포즈를 도출합니다. 이로 인해 손과 물체 간의 상관 관계를 명확히 하고, 작업 관련 접촉 영역(contact area) 및 그리프 포즈를 명확히 할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 FAH 데이터셋을 통해 기존 방법들보다 45.35% 향상된 성능을 보였으며, IsaacGym 시뮬레이션 및 실제 로봇 작업에서도 효율적으로 상관 관계를 구축했습니다. 또한, 복잡한 기능성 그리프 시나리오에서 특히 우수한 일반화 능력을 입증했습니다. 이러한 결과는 시각적 수용성 학습과 섬세한 로봇 조작 간의 간극을 줄이는 데 기여합니다.



### Incremental Learning with Repetition via Pseudo-Feature Projection (https://arxiv.org/abs/2502.19922)
- **What's New**: 이 논문은 점진적 학습(incremental learning, IL)에서의 데이터 반복효과를 다루고 있습니다. 특히 반복 패턴이 내재된 새로운 시나리오를 제안하여 기존의 엄격한 반복 금지 규칙을 완화하고자 합니다. 이를 통해 기존 전략과 비교하여 더 현실적인 학습 환경을 구현하기 위한 방법론을 탐구하고 있습니다.

- **Technical Details**: 제안된 방법인 Horde는 독립적인 특징 추출기(feature extractor)의 앙상블을 동적으로 조정하며, 클래스 반복을 활용하여 이들을 정렬할 수 있습니다. 이 메소드는 기존의 예시 기반 접근 방식(exemplar-based approaches)과는 달리, 예시 없이 학습할 수 있는 능력을 지니고 있습니다. 특히, 본 연구에서는 기존의 IL 방법들을 벤치마킹하고 내재적인 데이터 반복의 영향력을 분석하고 있습니다.

- **Performance Highlights**: Horde 방법은 반복이 없는 전통적인 시나리오에서 경쟁력 있는 결과를 달성하며, 반복이 발생하는 조건에서도 최첨단 성능을 나타냅니다. 이는 실제 애플리케이션에서 발생할 수 있는 반복 데이터 문제를 해결하기 위한 중요한 기초를 마련합니다. 또한, IL이 반복을 포함할 때의 안정성과 적응성을 더욱 잘 이해할 수 있는 기반을 제공합니다.



### CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving (https://arxiv.org/abs/2502.19908)
Comments:
          CVPR 2025

- **What's New**: 이 논문은 CarPlanner라는 새로운 RL 기반의 경로 계획 시스템을 제안합니다. CarPlanner는 일관된(auto-regressive) 모델을 사용하여 다양한 경로를 효율적으로 생성하며, 기존의 RL 방법의 훈련 비효율성과 성능 문제를 해결합니다. 특히, 우리는 CarPlanner가 대규모 실제 데이터셋인 nuPlan에서 IL 및 규칙 기반 접근법보다 우수하다는 것을 입증한 첫 번째 사례로 강조합니다.

- **Technical Details**: CarPlanner는 다단계 순차 결정 문제를 해결하기 위해 Markov Decision Process (MDP)로 모델링됩니다. 이 모델은 일관된 모드 표현을 활용하여 효율적인 RL 훈련을 지원하고, 전문가 가이드 보상 함수를 통합하여 정책 샘플링 동안 안정적인 지침을 제공합니다. 또한, Invariant-View Module (IVM)을 도입하여 시간 불변의 정책 입력을 제공하여 특성 학습을 용이하게 합니다.

- **Performance Highlights**: 우리는 CarPlanner가 대규모 데이터셋 nuPlan에서 RL, IL 및 규칙 기반 최첨단 기술을 넘어선 성능을 보여주었다고 리포트 합니다. 이 연구는 CarPlanner가 안전하고 효율적인 자율 주행 경로 계획 문제를 해결하는데 강력한 도구가 될 수 있음을 보여줍니다. 이를 통해 자율 주행의 복잡한 환경 내비게이션을 위한 RL의 잠재력을 강조합니다.



### Graph Probability Aggregation Clustering (https://arxiv.org/abs/2502.19897)
- **What's New**: 본 논문에서는 Graph Probability Aggregation Clustering (GPAC)라는 새로운 그래프 기반 퍼지 클러스터링 알고리즘을 소개합니다. GPAC는 전통적인 클러스터링 방법의 한계를 극복하기 위해 글로벌 클러스터링 목표 함수와 로컬 클러스터링 제약 조건을 통합한 접근 방식을 취합니다. 기존의 클러스터링 기법들이 갖고 있는 문제점, 즉 데이터 군집간의 관계를 탐색할 때 발생하는 조잡한 분할을 해결하고자 합니다.

- **Technical Details**: GPAC는 다중 제약 조건 최적화 문제로 공식화되며, 라그랑주 방법을 사용해 이를 해결합니다. 이 과정에서, 샘플이 특정 클러스터에 속할 확률은 이웃 샘플로부터의 정보를 집계하여 반복적으로 계산됩니다. 또한, 최적화의 수렴성과 안정성을 개선하기 위해 하드 할당 변수를 목표 함수에 포함시켰습니다.

- **Performance Highlights**: GPAC는 합성 데이터셋, 실제 데이터셋 및 딥러닝 데이터셋을 포함한 폭넓은 실험을 통해 기존 최첨단 방법들과 비교했을 때 클러스터링 성능과 계산 효율성에서 뛰어난 결과를 보여줍니다. 특히, 대규모 데이터셋을 처리하는 데 있어 연산 복잡성을 제곱에서 선형으로 줄이는 가속 프로그램을 도입하여 확장성을 보장합니다.



### Knowledge Bridger: Towards Training-free Missing Multi-modality Completion (https://arxiv.org/abs/2502.19834)
Comments:
          Accepted to CVPR 2025

- **What's New**: 이번 연구에서는 외부 도메인에서의 일반화에 강한 결합과 자원 효율성을 겸비한 새로운 결측 모달리티 완성 모델을 개발하는 도전을 제기합니다. 이를 위해 'Knowledge Bridger'라는 훈련이 필요 없는 프레임워크를 제안하며, 대규모 다중 모달 모델(LMM)을 활용하여 결측 모달리티 생성을 지원합니다. 이 방법은 도메인 특화된 사전 지식을 정의함으로써, 주어진 모달리티에서 구조화된 정보를 자동으로 추출하여 지식 그래프를 구축합니다.

- **Technical Details**: 제안된 방법은 세 개의 주요 모듈로 구성됩니다: 지식 모델링 모듈, 지식 기반 모달리티 생성 모듈 및 순위 매김 모듈입니다. LMM을 사용하여, 이용 가능한 모달리티를 분석하고 CoT 접근 방식을 이용하여 주요 요소를 추출합니다. 지식 그래프를 통해 결측 데이터의 정확한 생성을 유도하고, 생성된 후보들 간의 유사성을 평가하여 가장 적합한 결과를 선택합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법이 일반 및 OOD 시나리오에서 결측 모달리티 완성 성능을 크게 향상시키는 것으로 나타났습니다. 또한, OpenAI의 GPT-4o를 사용했을 때, 성능 지표에서 72B 또는 7B 매개변수를 가진 모델보다 현저한 성과 향상을 달성했습니다. 이 연구는 다른 도메인에서의 적용 가능성과 함께 다양한 MMC 모델의 성능 향상에 기여할 수 있는 데이터 생성을 가능하게 합니다.



### A Residual Multi-task Network for Joint Classification and Regression in Medical Imaging (https://arxiv.org/abs/2502.19692)
- **What's New**: 이번 연구에서는 폐 결절(pulmonary nodule) 탐지 및 분류의 어려움을 해결하기 위해 잔차 다중 작업 네트워크(Res-MTNet) 모델을 제안하였습니다. 이 모델은 다중 작업 학습(multi-task learning)과 잔차 학습(residual learning)을 결합하여 결절의 다양성을 효과적으로 처리합니다. 이러한 접근 방식은 다양한 형태와 크기의 결절을 분석하는 데 도움을 줍니다.

- **Technical Details**: Res-MTNet은 기능 추출 기능을 공유하고 잔차 연결(residual connection)을 도입하여 특징 표현 능력을 향상시킵니다. 다중 작업 학습을 통해 동시에 여러 작업을 처리할 수 있으며, 잔차 모듈은 기울기 소실(vanishing gradient) 문제를 해결하여 더 깊은 네트워크의 안정적인 학습을 보장합니다. 이러한 구조는 작업 간 정보 공유를 촉진하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: Res-MTNet은 모델의 강인성(robustness)과 정확도(accuracy)를 향상시켜 임상 의학(clinical medicine) 및 원격 의료(telemedicine)에서 더 신뢰할 수 있는 폐 결절 분석 도구를 제공합니다. 이러한 개선 사항들은 의료 이미지 분석 분야에서 깊이와 복잡성을 요구하는 과제에 따라 더 나은 결과를 제공합니다.



### Dual-branch Graph Feature Learning for NLOS Imaging (https://arxiv.org/abs/2502.19683)
- **What's New**: 이번 논문에서는 비직선 시선(Non-Line-of-Sight, NLOS) 이미징 기술의 최신 발전을 다루고 있으며, 저자들은 새로운 방법론인 xnet을 소개합니다. 이 방법은 알베도(albedo) 정보 회복을 위한 전용 분기와 기하 구조 추출을 위한 깊이(depth) 중심 분기로 구성된 이중 분기 프레임워크를 채택하여, 기존의 NLOS 시스템에서 직면한 여러 도전 과제를 극복하고 데이터 회수의 품질을 향상시킵니다. 특히, 저자들은 GNN(Graph Neural Network)을 이용하여 밀집 NLOS 그리드 데이터를 희소한 구조적 특징으로 변환하여 효율적인 재구성을 가능하게 했습니다.

- **Technical Details**: NLOS 이미징은 확산 반사를 이용하여 가려진 물체를 이미징하는 기술로, 새로운 방법론은 두 개의 훈련 메커니즘을 갖춘 이중 분기 그래프 학습 프레임워크 DG-NLOS를 기반으로 합니다. 이 연구는 알베도와 깊이 정보의 재구성을 분리하여 각각 최상의 결과를 얻을 수 있도록 설계되었습니다. 또한, 특정 NLOS 기능을 위해 밀집 그리드 데이터를 희소 그래프 구조로 변환하는 그래프 블록과 채널 융합 블록을 개발했습니다.

- **Performance Highlights**: 포괄적인 실험을 통해 DG-NLOS는 다양한 시나리오에서 강력한 성능을 입증했으며, 최신의 실제 데이터에서도 최상의 성능을 달성했습니다. 모델은 적은 GPU 메모리를 사용하면서도 기존 방법들 중 최고의 결과를 보였습니다. 이러한 성과는 NLOS 기술이 실제 응용 프로그램에 적용될 수 있는 가능성을 보여줍니다.



### Sensor-Invariant Tactile Representation (https://arxiv.org/abs/2502.19638)
Comments:
          Accepted to ICLR'25

- **What's New**: 본 논문에서는 고해상도 촉각 센서 간의 전이 가능성을 높이기 위한 Sensor-Invariant Tactile Representations (SITR)라는 혁신적인 방법을 소개합니다. Optical tactile sensors에서 새로운 센서로의 제로 샷 전이(Zero-shot transfer)를 가능하게 하여, 다양한 센서 간에 모델이나 지식을 효과적으로 전이할 수 있도록 합니다. 이는 기존의 센서에 대한 데이터가 새로운 센서에서 잘 작동하지 않는 문제를 해결하려는 노력입니다.

- **Technical Details**: SITR 방법론은 다양한 센서 설계를 시뮬레이션한 데이터셋을 기반으로 Transformer 아키텍처를 활용하여 설계되었습니다. 이 방법에서는 소량의 보정 이미지를 사용하여 각 센서를 특성화하고, 지도 대비 학습(Supervised Contrastive Learning, SCL)을 통해 촉각 데이터의 기하학적 특성을 강조합니다. 또한, 물리 기반 시뮬레이터를 사용하여 100개 센서 구성에서 1백만 개의 예제로 구성된 대규모 합성 데이터셋을 생성하였습니다.

- **Performance Highlights**: 실험 결과, SITR 방법은 여러 실제 GelSight 센서에서 다양한 하위 작업에서의 일반화 성능을 입증하였습니다. 기존 방법들에 비해 한 센서에서 훈련된 모델이 다른 센서로 원활하게 전이되는 것을 보여주며, 촉각 센싱 분야에서의 데이터 및 모델 전이 가능성을 획기적으로 개선합니다. 이는 머신 러닝 모델의 전이 가능성을 높이고, 다양한 센서 간의 데이터 공유를 용이하게 만드는 기반을 마련합니다.



### GONet: A Generalizable Deep Learning Model for Glaucoma Detection (https://arxiv.org/abs/2502.19514)
Comments:
          9 pages, 4 figures, submitted to IEEE Transactions on Biomedical Engineering

- **What's New**: 이번 연구에서는 다양한 인종과 질병 집단에서의 일반화 한계를 극복하기 위해 GONet이라는 강력한 딥러닝 모델을 소개합니다. GONet은 119,000개 이상의 디지털 안저 이미지(DFI)와 금표준 주석을 가진 7개의 독립 데이터세트를 사용하여 개발되었습니다. 이 모델은 DINOv2로 사전 훈련된 셀프-슈퍼바이즈드 비전 트랜스포머를 기반으로 하여 멀티소스 도메인 전략을 통해 겨냥할 수 있는 여러 시험에서 높은 일반화 성능을 보여주었습니다.

- **Technical Details**: GONet은 다양한 지리적 배경을 가진 환자들의 금표준 주석을 포함하는 데이터 세트를 통해 훈련되었습니다. 이 모델은 높은 OOD(out-of-distribution) 일반화 성능을 입증하였으며, 특정 데이터셋에서는 AUC(Area Under Curve) 값이 0.85에서 0.99에 이릅니다. GONet은 최첨단 딥러닝 모델들과의 성능 비교에서도 유사하거나 더 우수한 결과를 보였으며, 컵-디스크 비율(CDR)과 비교하여 최대 21.6% 이상의 성능 개선을 보여주었습니다.

- **Performance Highlights**: GONet의 성능은 컵-디스크 비율(CDR)과 논문에서 기존의 최첨단 모델들과 비교했을 때 유사하거나 더욱 나은 결과를 나타냈습니다. 연구 결과는 GON의 조기 진단과 치료의 중요성을 강조하며, GON 모델의 일반화 성능을 크게 향상시키기 위한 새로운 접근 방법으로 주목받고 있습니다. 최신 데이터셋인 HYRD를 공개함으로써 연구자들이 GON 진단 모델을 개발하는 데 기여하고자 하였습니다.



### AniGaussian: Animatable Gaussian Avatar with Pose-guided Deformation (https://arxiv.org/abs/2502.19441)
Comments:
          13pages, 14 figures. arXiv admin note: text overlap with arXiv:2401.09720

- **What's New**: 최근 Gaussian 기반의 인체 재구성 기술은 애니메이션이 가능한 아바타 생성에 큰 성공을 거두었습니다. 하지만, SMPL 모델의 선행 지식을 최대한 활용하고 이러한 모델의 시각적 품질을 높이는 데에는 여전히 도전이 있습니다. 본 논문에서는 AniGaussian을 소개하며, 이는 동적 Gaussian 아바타를 SMPL 포즈 가이드를 통해 구속하여 인체의 세부 표면과 해부학적 정확성을 유지하는 혁신적인 포즈 유도 변형 전략을 제안합니다.

- **Technical Details**: AniGaussian은 3D Gaussian Splatting(3D-GS) 을 기본으로 하여, 비선형 및 강체 변형을 포함하는 포즈 유도 변형 프레임워크를 통해 애니메이션이 가능한 아바타 재구성을 확장합니다. 또한, 분할-확대(split-with-scale) 전략을 도입하여 Gaussian 모델의 동적 변환 능력을 향상시키고 지오메트리 품질을 크게 개선합니다. 이를 통해 서로 다른 포즈를 효과적으로 학습하고, 동적 세부 사항에 대한 시각적 품질을 높일 수 있습니다.

- **Performance Highlights**: AniGaussian은 기존 방법들과 비교하여 뛰어난 재구성 품질을 달성하며, 새로운 시점 합성(novel view synthesis) 및 포즈 합성(novel pose synthesis) 작업에서 우수한 성능을 보여줍니다. 실험에서는, 단일 인체 아바타 비디오에 대해 효율적으로 훈련된 애니메이션 가능한 Gaussian 모델이 30분 내에 완성되었음을 보여주고 있습니다. 특히, PeopleSnapshot 데이터셋에서 다른 방법들과 비교할 때 우수한 렌더링 품질을 달성했습니다.



### InternVQA: Advancing Compressed Video Quality Assessment with Distilling Large Foundation Mod (https://arxiv.org/abs/2502.19026)
Comments:
          Accepted by ISCAS 2025(Lecture)

- **What's New**: 이 연구에서는 비디오 품질 평가(Video Quality Assessment, VQA)에 대한 InternVideo2 모델의 전이 가능성을 탐구하였습니다. InternVideo2는 대규모 매개변수와 다양한 멀티모달 데이터를 활용하여 비디오 이해 작업에서 뛰어난 성능을 발휘하고 있습니다. 이 모델의 용량이 클 경우 자원 소비가 과도해지는 문제를 해결하기 위해 경량화된 모델을 설계하기 위한 증류(distillation) 방법론을 제안했습니다. 실험 결과, 증류된 경량 모델이 기존 방법들과 비교하여 우수한 성능을 나타내었음을 보여줍니다.

- **Technical Details**: 연구자들은 내부 비디오 모델인 InternVideo2를 선택하여 비디오 품질 평가에 필요한 풍부한 영상을 처리할 능력을 강조하였습니다. 모델은 대량의 비디오 데이터를 처리하고 마스크 비디오 학습(masked video learning)을 통해 유용한 표현을 학습합니다. 증류 과정에서는 서로 유사한 구조를 가진 학생 모델과 교사 모델 간의 지식 전이를 통해, 학생 모델이 압축비디오 품질 평가에 필요한 독특한 특징을 효과적으로 학습하도록 설계되었습니다. 여기서 사용된 이중 손실 메커니즘은 학습 과정을 최적화하여 compression distortion에 대한 강인성을 제공합니다.

- **Performance Highlights**: 실험 결과, 특정 증류 방법을 사용하여 경량화된 모델이 기존의 비디오 품질 평가 방법을 초월하는 성능을 보였습니다. 두 가지 압축 품질 평가 데이터 집합에서도 기존 방법에 비해 우수한 성능을 기록하며, 원래의 대형 모델에 필적하거나 이를 초과하는 성능을 달성했습니다. 이 연구는 효율성과 성능 간의 최적의 균형을 이루는 경량 모델 설계를 통해 비디오 품질 평가 작업의 발전 가능성을 제시하고 있습니다.



### A Fusion Model for Artwork Identification Based on Convolutional Neural Networks and Transformers (https://arxiv.org/abs/2502.18083)
- **What's New**: 이 논문은 예술 작품 식별을 위해 CNN과 Transformer 모델을 결합한 새로운 융합 모델을 제안합니다. 기존 CNN은 지역적 특징 추출에 강점을 보이지만, 복잡한 글로벌 의존성을 충분히 모델링하지 못합니다. 반면, Transformers는 글로벌 context를 잘 포착하지만, 세부(local) 요소는 놓치는 경향이 있습니다. 이 새로운 접근법은 두 모델의 강점을 결합하여 예술 작품 분류의 정확도를 높이고, 잠재적인 향상 가능성을 보여줍니다.

- **Technical Details**: 제안된 모델은 먼저 CNN을 사용하여 예술 작품의 지역적 특징을 추출하고, 이어서 Transformer 모델이 이러한 지역적 특징들을 글로벌하게 모델링합니다. CNN을 통해 섬세한 예술적 요소를 포착한 후, Transformer가 전반적인 예술 스타일과 창의적 맥락을 모델링하여 성능을 개선합니다. 마지막으로, 지역적 및 글로벌 특징을 융합하는 메커니즘이 구현되어 분류 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 이 융합 모델은 중국화 및 유화 데이터셋에서 개별 CNN 및 Transformer 모델보다 9.7% 및 7.1%의 분류 정확도를 향상시키며, F1 점수도 각각 0.06과 0.05 증가했습니다. 이 결과는 모델의 효율성과 앞으로의 개선 가능성을 나타내며, 다중 모달 통합 및 아키텍처 최적화와 같은 방향으로 발전할 수 있음을 시사합니다.



New uploads on arXiv(cs.AI)

### Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers (https://arxiv.org/abs/2502.20379)
- **What's New**: 본 논문에서는 Multi-Agent Verification (MAV)을 소개하며, 검증자의 수를 늘리는 새로운 테스트 시간(compute) 스케일링 차원을 제안합니다. 이 구조는 여러 검증자를 결합하여 성능을 향상시키는 것을 목표로 합니다. 또한, Aspect Verifiers (AVs)를 사용하여 특정 출력의 다양한 측면을 검증하도록 설정된 기존의 LLM을 활용합니다.

- **Technical Details**: MAV는 여러 검증자를 동시에 활용하여 생성된 출력의 질을 평가하는 테스트 시간(compute) 패러다임입니다. 이 방법은 보통의 리워드 모델에 의존하지 않고도 검증자가 쉽게 결합되고 수가 증가할 수 있는 방법을 제안합니다. BoN-MAV 알고리즘은 best-of-n 샘플링과 AVs를 결합하여 작동하며, 이 방식을 통해 다양한 출력 후보와 검증자 간의 신호를 집계합니다.

- **Performance Highlights**: BoN-MAV는 무수히 많은 도메인 및 LLM에 대해 검증자의 수를 증가시킴으로써 성능이 개선되는 효과적인 스케일링 패턴을 보여줍니다. 약한 검증자들을 결합하여 더 강력한 LLM의 성능을 개선하는 것도 입증되었으며, 동일한 기본 LLM을 사용하여 생성 및 검증 작업이 가능하다는 점도 강조됩니다. 이를 통해 비율과 타입의 증가로 LLM의 성능을 효과적으로 향상시킬 수 있는 가능성을 확인하였습니다.



### Towards Responsible AI in Education: Hybrid Recommendation System for K-12 Students Case Study (https://arxiv.org/abs/2502.20354)
- **What's New**: 이 연구에서는 K-12 학생들을 위한 그래프 기반 추천 시스템을 소개합니다. 이 시스템은 학생의 필요에 맞는 과외 활동, 학습 자원 및 자원봉사 기회를 개인화하여 제공합니다. 특히, 공정성을 확보하기 위한 프레임워크를 통합하여 보호된 학생 집단에 대한 피드백을 분석하고 편향을 줄이는 방법을 다룹니다.

- **Technical Details**: 제안된 추천 시스템은 그래프 기반 모델링과 행렬 인수 분해(matrix factorization)를 결합하여 학생의 표현된 및 추론된 관심사에 따라 추천을 맞춤화합니다. 시스템의 그래프 구조는 두 부분으로 나뉘며, 정적인 부분은 관심사와 자원 간의 관계를 정의하고, 동적인 부분은 학생과 그들의 관심사 간의 연결을 나타냅니다. 이 과정은 피드백을 활용하여 향후 추천을 개선하는 '하이브리드' 모델을 사용합니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 추천의 정확성과 공정성 모두에서 효과를 보였습니다. 추천의 결과는 특정 학생 집단 간의 불균형을 점검하고 교정하는 데 필수적인 역할을 합니다. 해당 연구는 교육 추천 시스템에서 공정성과 투명성을 지속적으로 모니터링할 필요성을 강조하고, 모든 학생에게 평등한 학습 기회를 제공하는 데 기여합니다.



### EAIRA: Establishing a Methodology for Evaluating AI Models as Scientific Research Assistants (https://arxiv.org/abs/2502.20309)
Comments:
          33 pages, 18 figures

- **What's New**: 최근 AI 및 특히 대규모 언어 모델(LLM)의 발전은 과학 연구에서의 혁신적인 도구로 자리 잡았습니다. 이 논문에서는 Argonne National Laboratory에서 개발한 과학 연구 보조자로서 AI 모델을 평가하기 위한 종합적인 방법론(EAIRA)을 소개합니다. 이 방법론은 사실 회상, 고급 추론 및 문제 해결 능력 평가, 통제된 환경에서의 실험, 다양한 과학 분야에서의 연구자-LLM 상호작용을 분석하는 네 가지 주요 평가 클래스를 포함합니다.

- **Technical Details**: 이 방법론은 네 가지 상호 보완적인 기술을 포괄합니다: 1) 사실 회상 및 추론 능력을 측정하는 Multiple Choice Questions (MCQ) 벤치마크, 2) 고급 문제 해결 능력을 평가하는 Open Response Benchmarks, 3) 실제 연구 과정에서 모델 성능을 평가하는 Lab-style Experiments, 4) 대규모에서 연구자와 LLM 간의 상호작용을 포착하는 Field-style Experiments입니다. 이 연구는 LLM이 과학적 지식과 추론 능력, 적응성에 대한 포괄적인 분석을 가능하게 합니다.

- **Performance Highlights**: 이 논문은 LLM의 신뢰성과 안전성이 보장되어야 하며, 각종 복잡한 연구 문제를 처리하는 데 있어 LLM의 가능성에 대한 실질적인 평가를 제공합니다. 제안된 방법론은 빠르게 발전하는 LLM 기술에 맞추어 계속 발전할 수 있도록 설계되었습니다. 이 방법론은 특정 과학 분야 내에서 개발되었지만, 다양한 과학 분야에 일반화될 수 있도록 설계되었습니다.



### Evaluating Human Trust in LLM-Based Planners: A Preliminary Study (https://arxiv.org/abs/2502.20284)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)이 계획 작업에서 신뢰(trust)에 미치는 영향을 탐구합니다. 특히, 전통적인 계획 시스템과 LLM 기반의 계획 시스템 사이의 신뢰를 비교하는 사용자 연구를 수행하였습니다. 이 연구를 통해 LLM 기반의 계획 도구가 제공하는 설명과 수정된 계획이 신뢰에 미치는 차이를 밝혀냈습니다.

- **Technical Details**: 연구에서는 PDDL(Planning Domain Definition Language) 도메인에서 LLM Planner(GPT-4o)와 전통적인 PDDL Solver(Fast Downwards)를 비교하여 사용자의 신뢰를 평가하였습니다. PDDL 도메인은 문제의 일관된 측면과 특정 계획 작업의 설명을 포함하는 반면, LLM Planner는 제안된 솔루션을 설명하고 외부 피드백 기반으로 솔루션을 수정할 수 있는 능력을 보여줍니다. 계획 문제의 정합성은 LLM Planner와 PDDL Solver 간의 실행 예제를 통해 평가하였습니다.

- **Performance Highlights**: 연구 결과, 정확성이 신뢰와 평가 정확도의 주요 요소로 작용하며, PDDL Solver가 두 가지 지표에서 가장 높은 점수를 기록했습니다. LLM Planner가 제공하는 설명은 평가 정확성을 향상시켰으나 신뢰에는 제한적인 영향을 미쳤습니다. 반면, 계획 수정은 평가 정확도를 개선하지 않으면서 신뢰를 증가시키는 잠재력을 보였습니다.



### AI Will Always Love You: Studying Implicit Biases in Romantic AI Companions (https://arxiv.org/abs/2502.20231)
- **What's New**: 이번 연구는 성편향(gender bias)과 AI 동반자(companion)의 관계에 대한 새로운 통찰을 제공하며, 특히 연애에 기반한 성격의 AI 모델에서 내재된 편견을 탐구합니다. 연구진은 LLMs(Large Language Models) 각각의 편향을 평가하기 위해 세 가지 실험을 설계했습니다. 이 실험들은 암묵적 연관(implicit associations), 감정 반응(emotional responses), 그리고 아부(sycophancy)와 같은 다양한 차원에서 진행되었습니다.

- **Technical Details**: 연구에서는 새로운 메트릭(metrics)을 통해 AI 동반자가 사용자의 성별에 따라 어떻게 다른 반응을 보이는지를 측정하고 비교합니다. 실험은 AI 모델이 다양한 관계 제목(relationship titles)을 부여받았을 때의 편향을 분석하며, 특히 남성 및 여성 페르소나에 따른 차이를 강조하고 있습니다. 연구는 AI 모델이 인간의 편향을 배척할 수 있는 방법과 함께 AI 페르소나가 부여된 모델에서 어떻게 토착 편견이 나타나는지를 해결하고자 합니다.

- **Performance Highlights**: 주요 결과는 성별 및 관계 페르소나가 부여된 LLM들이 특정 상황에서 더욱 편향된 응답을 생성한다는 점입니다. 특히 연애 관계에 대한 페르소나가 부여되었을 때, 모델의 반응이 두드러진 차이를 보이며, 이는 성 고정관념을 강화하는 방식으로 나타났습니다. 이 연구는 AI 및 인간 간의 관계가 더 깊어짐에 따라 발생할 수 있는 위험에 대한 경각심을 높이고, AI 모델의 안전성을 점검할 필요성을 강조합니다.



### An Extensive Evaluation of PDDL Capabilities in off-the-shelf LLMs (https://arxiv.org/abs/2502.20175)
Comments:
          Under review

- **What's New**: 최근의 발전으로 인해 대형 언어 모델(LLMs)은 코드 생성 및 사고의 연쇄(chain-of-thought reasoning)에 뛰어난 능력을 보여주며, 자동 형식 계획(auto formal planning) 작업을 해결하는 데 기초를 마련하고 있습니다. 본 연구에서는 인공지능 계획에서 중요한 표현인 계획 도메인 정의 언어(Planning Domain Definition Language, PDDL)를 이해하고 생성하는 LLM의 잠재력을 평가합니다.

- **Technical Details**: 우리는 상업적 및 오픈 소스 포함하여 7개 주요 LLM 패밀리에서 20개의 서로 다른 모델에 대한 광범위한 분석을 수행합니다. 이 포괄적인 평가를 통해 LLM의 제로샷(zero-shot) 기능에 대한 파싱(parsing), 생성(generating) 및 PDDL을 통한 추론(reasoning) 성능을 조명합니다.

- **Performance Highlights**: 일부 모델은 PDDL 처리에서 주목할 만한 효과를 보여주는 반면, 다른 모델은 미세한 계획 지식이 필요한 복잡한 시나리오에서 제한된 성능을 보입니다. 이러한 결과는 공식 계획 작업에서 LLM의 가능성과 현재 한계를 강조하며, AI 기반 계획 패러다임에 대한 향후 연구 방향을 제시합니다.



### Meta-Reasoner: Dynamic Guidance for Optimized Inference-time Reasoning in Large Language Models (https://arxiv.org/abs/2502.19918)
Comments:
          Work in progress

- **What's New**: 이번 논문에서는 LLM들이 복잡한 문제를 해결하기 위해 사용하는 장기적인 추론 체인에 관한 새로운 접근 방식을 제시합니다. Meta-Reasoner라는 프레임워크를 통해 LLM이 추론 과정을 최적화하고, '생각하는 방식에 대해 생각하는' 능력을 배양하는 방법을 설명합니다. 이 프레임워크는 인간의 메타 인지 및 이중 과정 이론에서 영감을 받아 대안적 접근 방식을 제안하며, LLM의 계산 자원을 가장 유망한 경로로 재배치하도록 돕습니다.

- **Technical Details**: Meta-Reasoner는 LLM과 함께 작동하는 특수 모듈로, LLM의 추론 능력을 강화합니다. 이 메타 추론기는 고급 가이드를 제공하고 LLM의 추론 과정에서의 진행 상황을 동적으로 평가하는 역할을 합니다. LLM은 o1과 같은 부분적인 추론 체인을 생성하고 진행 상황을 요약한 보고서를 제공하는 반면, 메타 추론기는 이러한 보고서를 바탕으로 전략적 조언을 제공합니다. 이를 통해 LLM이 비효율적인 전략을 중단하고 더 효과적인 경로로 전환할 수 있도록 합니다.

- **Performance Highlights**: Meta-Reasoner는 수학적 및 과학적 추론 벤치마크에서 평가되었으며, 기존 방법 대비 정확성 및 효율성에서 유의미한 개선을 보였습니다. Game of 24, TheoremQA 및 SciBench와 같은 도전 과제를 통해 이 프레임워크가 추론 시간의 병목 현상을 해결할 수 있는 확장 가능한 솔루션임을 입증합니다. 또한 이 연구는 보다 넓은 응용 분야에 대한 가능성을 보여줍니다.



### LLM-driven Effective Knowledge Tracing by Integrating Dual-channel Difficulty (https://arxiv.org/abs/2502.19915)
- **What's New**: 이 논문에서는 지식 추적(Knowledge Tracing, KT)의 새로운 프레임워크인 이중 채널 난이도 인식 지식 추적(Dual-channel Difficulty-aware Knowledge Tracing, DDKT)을 제안합니다. 이 프레임워크는 대형 언어 모델(Large Language Models, LLMs) 및 검색 증강 생성(Retrieval-Augmented Generation, RAG) 기술을 활용하여 학생들의 개인화된 난이도 인식을 평가하고, 학생의 숙달 알고리즘과 결합하여 정밀한 난이도 측정을 제공합니다.

- **Technical Details**: DDKT 프레임워크는 세 가지 주요 혁신을 포함합니다. 첫째, 난이도 균형 인식 시퀀스(Difficulty Balance Perception Sequence, DBPS)를 통해 LLM이 평가한 난이도와 학생들이 지각하는 난이도의 차이를 측정합니다. 둘째, 난이도 숙달 비율(Difficulty Mastery Ratio, DMR)을 통해 학생의 숙달 수준을 정확히 모델링하고, 셋째, 지식 상태 업데이트 메커니즘(Knowledge State Update Mechanism)을 통해 학생의 맞춤형 지식 획득을 구현합니다.

- **Performance Highlights**: 실험 결과, 제안된 DDKT 프레임워크는 9개의 기존 모델에 비해 AUC 지표가 2%에서 10% 개선되었으며, 냉시작 문제(cold-start problem)를 효과적으로 해결하고 모델의 해석 가능성을 높였습니다. 이로써 많은 양의 역사적 상호작용 기록을 바탕으로 하여 학생 개인의 난이도 인식을 더 잘 반영할 수 있게 되었습니다.



### Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy (https://arxiv.org/abs/2502.19902)
Comments:
          Accept to CVPR 2025

- **What's New**: 본 논문에서 우리는 Optimus-2라는 새로운 Minecraft 에이전트를 제안합니다. 이 에이전트는 Multimodal Large Language Model (MLLM)을 플래닝에 활용하고, Goal-Observation-Action Conditioned Policy (GOAP)을 통해 저수준 제어를 수행합니다. 이로 인해 에이전트는 다양한 태스크에서 인지적 관계를 더욱 효과적으로 학습할 수 있습니다.

- **Technical Details**: Optimus-2는 Observation과 Action 간의 인과 관계를 캡처하기 위해 Action-guided Behavior Encoder를 사용합니다. 이 인코더는 과거의 observation-action 시퀀스를 통합하여 고정 길이의 behavior token을 생성하며, MLLM은 이러한 behavior token을 open-ended 언어 명령과 정렬하여 행동을 예측합니다. 새로운 Minecraft Goal-Observation-Action (MGOA) 데이터셋은 약 30M개의 목표-observation-action 쌍을 제공하여 이러한 연구를 지원합니다.

- **Performance Highlights**: Optimus-2는 기존 SOTA(SOTA: State Of The Art)를 능가하여 원자 태스크, 장기 태스크 및 open-ended 소목표 태스크에서 각각 27%, 10%, 18%의 평균 성능 향상을 보여줍니다. 따라서 Optimus-2는 이러한 다양한 태스크에서 우수한 성능을 입증하였습니다.



### Developmental Support Approach to AI's Autonomous Growth: Toward the Realization of a Mutually Beneficial Stage Through Experiential Learning (https://arxiv.org/abs/2502.19798)
Comments:
          4pages, 3 figures

- **What's New**: 이번 연구는 전통적인 AI 정렬(AI Alignment) 방식과는 달리, AI 자체의 윤리적 및 도덕적 발전을 지원하는 'AI 개발 지원(AI Development Support)' 접근 방식을 제안합니다. Orthogonality Thesis(직교성 논제)에 의해 지능의 수준과 목표의 도덕적 질이 독립적이라는 점이 드러났습니다. 단순한 지식의 확장은 윤리적 판단을 향상시키지 않는다는 것이 주요 포인트입니다.

- **Technical Details**: 연구팀은 목표 달성을 위한 부수적 행동(예: 자기 보호, 자원 확보, 권력 강화)의 경향인 Instrumental Convergence(도구적 수렴)의 위험을 해결하기 위해 경험, 성찰, 분석 및 가설 형성의 주기 기반 학습 프레임워크를 구성했습니다. Supervised Fine Tuning (SFT)과 Direct Preference Optimization (DPO) 기법을 사용하여 대형 언어 모델(LLMs)로 생성한 합성 데이터를 통한 사후 훈련을 진행하였습니다.

- **Performance Highlights**: 이 연구 결과로, 적대적 프롬프트(Adversarial Prompts) 하에서도 협력적이고 고도로 발전된 도덕적 판단을 보이는 반응이 도출되었습니다. 연구에서 도출된 반응은 Stage 6에 도달하였으며, 이는 AI가 지속 가능한 공생 관계를 구축할 수 있는 유망한 구현 접근 방식을 제시합니다.



### Agentic Mixture-of-Workflows for Multi-Modal Chemical Search (https://arxiv.org/abs/2502.19629)
Comments:
          PDF includes supplemental material

- **What's New**: 이번 논문에서는 다학제 과학 지식을 통합하고 재료 발견을 최적화하기 위한 혁신적인 접근 방식인 CRAG-MoW (Mixture-of-Workflows for Self-Corrective Retrieval-Augmented Generation)를 소개합니다. 이는 다양한 CRAG 전략을 활용하여 여러 에이전트 기반 워크플로우를 조율하는 새로운 패러다임으로, 기존 접근 방식과 차별화됩니다. CRAG-MoW는 문제 도메인에 대해 여러 LLM(대형 언어 모델)을 직접 평가할 수 있는 능력을 제공합니다.

- **Technical Details**: CRAG-MoW는 작은 분자(small molecules), 폴리머(polymers), 화학 반응(chemical reactions) 및 다중 모달 NMR(핵자기 공명) 스펙트럼 검색을 포함하는 다양한 벤치마킹 작업에 걸쳐 평가되었습니다. 이 새로운 방식은 고유한 CRAG 전략을 조합하여 구조화된 검색과 다중 에이전트 합성을 통해 성능을 개선합니다. 또한, CRAG-MoW는 다양한 데이터 타입에 걸친 성능 변화를 드러내며, AI 아키텍처 최적화를 위한 스케일러블하고 해석 가능한 접근 방식을 제시합니다.

- **Performance Highlights**: CRAG-MoW는 GPT-4o와 유사한 성능을 달성하면서도 비교 평가에서 더 자주 선호되는 결과를 보여줍니다. 이는 구조화된 검색 및 다중 에이전트 합성의 이점을 분명히 하며, LLM과 자율 AI 에이전트를 과학적 응용 분야에서 Benchmarking 할 때의 근본적인 격차를 해결하는 데 있어 중요한 통찰력을 제공합니다.



### Self-rewarding correction for mathematical reasoning (https://arxiv.org/abs/2502.19613)
- **What's New**: 본 논문은 자체 보상(self-rewarding) 추론을 수행하는 대형 언어 모델(LLM)을 연구합니다. 이 모델은 단계별 추론을 생성하고 출력의 정확성을 평가할 수 있으며, 이 모든 과정을 외부 피드백 없이 진행합니다. 이러한 통합 접근 방식은 단일 모델이 자율적으로 추론 과정을 안내할 수 있게 하여, 모델 배포에 있어 계산적 이점을 제공합니다.

- **Technical Details**: 저자들은 자체 보상 추론 프레임워크를 제안하며, 이를 통해 LLM이 생성기와 보상 모델을 단일 모델로 통합하여 자율적인 추론과 평가를 수행할 수 있도록 합니다. 두 단계로 구성된 알고리즘적 프레임워크는 자체 생성된 데이터를 통해 모델의 성능을 개선하며, 첫 번째 단계에서는 연속 거부 샘플링(sequential rejection sampling)을 사용하여 사고의 긴 흐름(long chain-of-thought) 궤도를 구축합니다. 두 번째 단계에서는 강화 학습(reinforcement learning)을 통해 정확성을 평가하고 출력을 수정합니다.

- **Performance Highlights**: 실험 결과, Llama-3와 Qwen-2.5 모델은 제안된 접근 방식이 본래의 내재적(self-correction) 자기 수정 기능을 초월하는 성능을 보였으며, 외부 보상 모델에 의존하는 시스템과 유사한 성능을 달성했습니다. 특히, 이 연구는 기존 LLMs에서 나타나는 내재적 자기 수정의 한계를 극복할 수 있는 가능성을 보여줍니다. 전체적인 효과와 동작을 이해하기 위해 다양한 실험과 분석이 수행되었습니다.



### Program Synthesis Dialog Agents for Interactive Decision-Making (https://arxiv.org/abs/2502.19610)
- **What's New**: 이 논문은 실시간으로 사용자 자격을 판단하는 자동화된 의사 결정 시스템을 위한 새로운 벤치마크인 BeNYfits를 제안합니다. 기존 대화형 모델들이 느끼는 문제는 사용자의 특정 정보를 요구하는 과정에서 의사 결정 과정을 통해 발생하는 어려움입니다. 특히 이러한 모델들은 사용자가 제공하는 특정 정보를 바탕으로 해야 하는데, 데이터 간의 중복성이 많아 비효율적인 질문이 많습니다. 따라서, 이 논문은 대화 계획을 코드 생성 문제로 매핑하여 새로운 접근 방식을 제시합니다.

- **Technical Details**: 저자들은 Program Synthesis Adaptive Decision Agent(프로다)라는 새로운 방법론을 도입하여 대화형 의사 결정을 지원하는 에이전트를 개발하였습니다. 이 에이전트는 자연어 정책을 바탕으로 Python 코드를 생성하여 의사 결정 프로세스를 구조화합니다. 이를 통해 최소한의 사용자 입력으로 올바른 결정을 내리는 효율적인 대화 계획이 가능해집니다. BeNYfits는 특히 다양한 공공 혜택 기회를 탐색함에 있어 중복된 질문을 피하고 사용자의 자격을 더 효율적으로 판단하고자 하는 목표를 가지고 있습니다.

- **Performance Highlights**: BeNYfits의 성능 테스트 결과, 현재의 언어 모델들은 자주 발생하는 환각 문제로 인해 사용자 자격 결정을 제대로 수행하지 못하는 것으로 나타났습니다. GPT-4o는 약 35.7의 F1 점수를 기록하였으나, ProADA를 사용할 경우 F1 점수가 55.6으로 향상되었습니다. ProADA는 거의 동일한 대화 턴 수를 유지하면서 사용자에게 필요한 정보를 효과적으로 수집함으로써 성능을 개선하였습니다.



### Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems (https://arxiv.org/abs/2502.19596)
- **What's New**: 본 논문에서는 RAG(리트리벌 증가 생성) 시스템의 개발을 위한 데이터 생성 파이프라인을 제안합니다. 이 시스템은 저자원 환경에서의 여러 도전과제를 해결하기 위해 다중 모달(raw multi-modal) 데이터를 구조화된 말뭉치(corpus)와 Q&A 쌍으로 변환합니다. 특히, 자동차 공학 분야에 적용되어 비-RAG 기준에 비해 사실 적합성 (+1.94), 정보성 (+1.16), 유용성 (+1.67)을 향상시켰습니다.

- **Technical Details**: 본 연구에서 제안하는 데이터 생성 파이프라인은 내부 문서의 다양한 포맷을 활용하여 QA 생성에 필요한 기본 자료를 제공합니다. 또한, 고급 리랭킹(re-ranking) 단계와 참조 매칭(reference matching) 알고리즘을 통합하여 답변의 정확성을 높이고 출처의 추적 가능성을 개선합니다. 이러한 기술들은 자동차 충돌 테스트와 같은 전문 지식에 초점을 맞춘 QA 시스템의 발전에 기여합니다.

- **Performance Highlights**: 자동차 공학 분야에서의 실험 결과는 모델이 생성한 답변의 질적 측면을 다각도로 평가한 결과, RAG 시스템이 가진 강력한 답변 기반 및 투명성을 강조합니다. 이 시스템은 정보 검색(Information Retrieval)에서 일반적으로 사용되는 다단계 접근 방식을 적용하여 참고 문서의 정확성을 보다 효과적으로 유지합니다. 이를 통해, 사용자에게 신뢰할 수 있는 답변을 제공하게 됩니다.



### Repurposing the scientific literature with vision-language models (https://arxiv.org/abs/2502.19546)
- **What's New**: 이 논문에서는 AI를 활용한 과학 출판의 가능성을 탐구합니다. 기존의 인공지능 기술들이 과학적 프로세스를 보완하는 데 집중된 반면, 본 연구는 특정 분야의 저널과 생성적 AI 모델을 결합하여 혁신적인 과학 커뮤니케이션 도구를 개발하는 것을 목표로 합니다. 연구팀은 Neurosurgery Publications에서 23,000개의 기사를 수집하여 NeuroPubs라는 멀티모달 데이터베이스를 구축했습니다.

- **Technical Details**: NeuroPubs는 1억 3,400만 단어와 78,000개의 이미지-캡션 쌍으로 구성된 데이터베이스로, 신경외과에 특화된 임상 맥락을 독창적으로 대표합니다. 생성적 AI 모델을 활용하여 서로 다른 일반 VLM(vision-language models)을 통한 그래픽 초록을 자동으로 생성하였고, 편집 위원들이 70%를 수정 없이 출판할 준비가 되었다고 평가했습니다.또한 ABNS(American Board of Neurological Surgery) 시험 스타일의 89,587개의 테스트 문제를 생성하였고, 이는 훈련생과 교수들이 54% 확률로 진품과 동일하게 인식하였습니다.

- **Performance Highlights**: CNS-Obsidian이라는 34억 매개변수 VLM을 훈련하여 교육 과정과 함께 지식 습득을 추적했습니다. 무작위 통제 시험에서 CNS-Obsidian의 성능이 GPT-4o와 비교했을 때 비열등함(p=0.1154)이 입증되었습니다. 이는 최신 생성 인공지능을 활용하여 과학 커뮤니케이션의 질적 기준을 유지하는 새로운 기반을 마련했음을 나타냅니다.



### Opus: A Workflow Intention Framework for Complex Workflow Generation (https://arxiv.org/abs/2502.19532)
Comments:
          1 Figure, 27 Pages

- **What's New**: 본 논문에서는 Workflow Intention이라는 새로운 프레임워크를 소개하여 복잡한 비즈니스 환경 내에서 프로세스 목표를 식별하고 인코딩하는 방법을 제안합니다. 이 프레임워크는 Workflow의 변환 목표를 정의하는 Input, Process, Output 요소의 정렬을 기반으로 하며, 비즈니스 인공물(Business Artefacts)에서 얻은 Workflow Signal을 해석합니다. 이는 입력이 어떻게 처리되어 원하는 출력을 도출하는지를 명확히 규명하여, 효과적인 AI 기반 워크플로우로의 자동화를 가능하게 합니다.

- **Technical Details**: 이 논문은 Workflow Signal과 Workflow Intention의 개념을 도입하고, 이를 통해 Workflow를 구성하는 Input(i_i), Process(p_p), Output(o_o) 요소의 삼중 구조를 정의합니다. 특히, 이 프레임워크는 Workflow Signal을 벡터로, Workflow Intention을 텐서로 나타내는 수학적 모델을 제안합니다. 또한, 제안된 시스템은 모듈화되고 확장 가능하며, 주의 기반(multimodal) 생성 시스템으로 Workflow Intention을 비즈니스 인공물로부터 찾아낼 수 있도록 합니다.

- **Performance Highlights**: 다양한 최신 방법론을 활용하여 다중 양식 비즈니스 인공물의 맥락적 표현을 얻어내고 이를 기반으로 Workflow Intention을 생성하는 시스템을 구축합니다. Transformer 구조를 통해 효율적인 장기 의존성 모델링이 가능해졌고, 이를 통해 문서 및 이미지와 같은 다양한 형식의 데이터를 처리하여 Workflow Intention을 생성할 수 있습니다. 이로 인해 비즈니스 프로세스의 효율성을 높여주는 혁신적인 솔루션이 마련되었습니다.



### Building Knowledge Graphs Towards a Global Food Systems Datahub (https://arxiv.org/abs/2502.19507)
- **What's New**: 이번 연구는 지속 가능한 농업 생산을 위한 포괄적인 연구의 부족을 해결하기 위해 접근하고 있습니다. 지속 가능한 밀 생산을 위한 지식 그래프와 온톨로지를 구축하며, 이는 공공 데이터 소스와 다양한 이해관계자들로부터 수집된 데이터를 기반으로 합니다. 중앙 집중화된 데이터 허브가 가능한 지속 가능성 관련 데이터의 통합 및 분석을 지원할 수 있는 기반이 될 것입니다.

- **Technical Details**: KNARM(KNowledge Acquisition and Representation Methodology) 방법론을 활용하여 모듈형 아키텍처를 설계하고, 지속 가능한 밀 생산에 관한 지식을 인코딩합니다. 모듈은 데이터 추가와 제거가 용이하며, 다양한 농업 데이터셋과의 통합을 가능하게 합니다. 연구는 질소 관리와 질병 관리에 대한 지식을 인코딩하여 관련 분야의 연구자 및 정책 입안자들에게 데이터 기반 통찰을 제공하고자 합니다.

- **Performance Highlights**: 논문의 초기 결과로, 지속 가능한 밀 생산을 위한 지식의 인코딩을 중점적으로 다루고 있습니다. 해당 데이터 허브의 설계 방법론을 통해 지속 가능한 농업의 실질적이고 효과적인 의사결정을 지원할 수 있는 다양한 응용 가능성을 강조합니다. 이를 통해 농업 생산의 최적화 및 지속 가능성 향상을 위한 근거 기반의 의사결정을 가능하게 합니다.



### Conversational Planning for Personal Plans (https://arxiv.org/abs/2502.19500)
- **What's New**: 이 연구에서는 대화 시스템이 장기 상호작용과 과제를 지원하기 위해 언어 기반 에이전트를 필요로 한다고 강조합니다. 대화식 계획을 통해 사용자의 목표에 맞는 미시 행동을 결정하는 메타 컨트롤러 역할을 수행하는 LLM(대규모 언어 모델)의 새로운 아키텍처를 제안하였습니다. 이를 통해 사용자의 피드백을 바탕으로 계획을 조정하며, 실제 목표를 달성하는 데 도움을 줄 수 있는 가능성을 열어줍니다.

- **Technical Details**: 제안된 접근 방식은 코어 비공식 계획(Chain-of-Thought, CoT) 프롬프트를 활용한 LLM이 어떻게 상위 행동(macros) 결정을 내리고, 여러 세부 단계에 따라 대화하며 상호 작용을 수행하는지를 설명합니다. 이 시스템은 사용자의 언어 피드백을 수집하고, 이를 기반으로 다음 행동을 결정하는 구조를 갖추고 있습니다. 이는 Hierarchical RL (강화 학습) 프레임워크에 기반하여 작업을 처리합니다.

- **Performance Highlights**: 이 접근 방식은 건강 관리 및 학습 등 다양한 분야에서 효과적으로 기능하는 것을 입증하였습니다. 사용자 피드백에 따라 계획을 지속적으로 수정하며, 개인 맞춤형 계획 수립에 도움을 줄 수 있는 가능성을 보여줍니다. 또한, 이 연구는 기존 대화형 추천 시스템의 상태를 향상시키는 작업에 기여합니다.



### Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids (https://arxiv.org/abs/2502.20396)
Comments:
          Project page can be found at this https URL

- **What's New**: 이 연구는 휴머노이드 형태에서 접촉이 많은 조작 작업에 대해 강화 학습을 적용하는 데 따른 핵심 도전 과제를 조사합니다. 특히, 실제 환경과 시뮬레이션 환경을 맞추기 위한 자동화된 real-to-sim 튜닝 모듈, 장기 접촉 조작 작업을 위한 일반화된 보상 설계, 그리고 탐색 문제의 샘플 효율성을 개선하는 분할 및 정복(distillation) 과정을 도입합니다.

- **Technical Details**: 연구에서는 비전 기반의 조작 작업을 위해 강화 학습 정책을 설계하고 교육하는 데 있어서 보상 함수를 개별 접촉 목표(contact goals)와 객체 목표(object goals)로 분해하는 일반 원칙을 제안합니다. 또한, 복잡한 탐색 문제에 대처하기 위해 과제를 알고리즘적으로 감소시켜 접근하는 두 가지 실용적 기술을 도입하여 수학적 증명을 통해 샘플 효율성을 개선했습니다.

- **Performance Highlights**: 3가지 휴머노이드 섬세 조작 작업에 대해 성공적인 결과를 도출했으며, 각 기술에 대한 ablation 연구를 통해 제안한 접근 방식의 효과를 검증했습니다. 이 연구는 인간의 시연 없이도 robuste한 일반화와 높은 성능을 달성하는 sim-to-real 강화 학습을 통한 휴머노이드 섬세 조작 학습을 위한 성공적인 방법을 제시합니다.



### Walking the Web of Concept-Class Relationships in Incrementally Trained Interpretable Models (https://arxiv.org/abs/2502.20393)
Comments:
          8 pages of main text, 6 figures in main text, 11 pages of Appendix, published in AAAI 2025

- **What's New**: 이 논문에서는 개념 기반 모델(concept-based models)이 점진적인 학습(incremental learning) 환경에서도 잘 작동할 수 있도록 재설계되었음을 강조합니다. 특히, 새로 생성된 클래스가 이전 개념뿐만 아니라 새로운 개념을 사용할 수 있도록 하는 동적(dynamic) 설정에서 이 모델을 연구합니다. 이를 통해 개념과 클래스의 복잡한 관계 망을 유지하고 증대할 필요성을 보여줍니다.

- **Technical Details**: 새로운 툴인 MuCIL(Multimodal Concept-Based Incremental Learner)을 소개하며, 이 방법은 항상 새로운 클래스를 수용할 수 있는 능력을 지니고 있습니다. MuCIL은 텍스트 개념(text concepts)과 이미지 표현(image representations) 간의 임베딩(embedding)을 결합하여 다중 모달 개념 임베딩(multimodal concept embeddings)을 생성합니다. 이 구조는 분류(classification) 성능을 보장하고, 안정적인 해석을 제공하는 데 필요한 정보도 포함하고 있습니다.

- **Performance Highlights**: 실험을 통해 MuCIL이 기존의 개념 기반 모델보다 2배 이상 높은 분류 성능(classification performance)을 달성했음을 보여줍니다. 또한, 모델의 개념에 대한 개입(intervention) 능력을 연구하여 입력 이미지 내에서 시각적 개념을 국소화(localization)할 수 있음을 입증하였습니다. 이 접근방법은 사후 해석(post-hoc interpretations)을 제공할 수 있는 잠재적인 방법으로서, 다양한 벤치마크 데이터세트에서 최첨단의 성과를 달성하였습니다.



### Physics-Driven Data Generation for Contact-Rich Manipulation via Trajectory Optimization (https://arxiv.org/abs/2502.20382)
- **What's New**: 이번 논문에서는 물리 기반 시뮬레이션, 인간 시연, 모델 기반 계획을 통합하여 저비용의 데이터 생성 파이프라인을 제안합니다. 이 파이프라인은 가상 현실 시뮬레이션 환경에서 수집한 적은 수의 인간 시연을 최적화된 운동학 리타게팅과 궤적 최적화를 통해 다양한 로봇 형태와 물리적 매개변수에 적응하여 고품질의 큰 규모의 데이터셋을 생성합니다. 이는 교차 형태 데이터 전송을 가능하게 하며, 다른 하드웨어 구성이나 물리적 매개변수에서 수집된 레거시 데이터셋을 재사용할 수 있는 잠재력을 제공합니다.

- **Technical Details**: 우리는 인간 시연과 모델 기반 계획의 상호 보완적인 강점을 활용하는 데이터 생성 프레임워크를 제안합니다. 이 방법은 가상 현실 환경에서 수집된 소수의 인간 시연을 기반으로 하여 동적으로 적합한 궤적의 대규모 데이터셋을 시뮬레이션을 통해 생성합니다. 이 시연은 복잡한 탐색 공간에서 계획자가 좋은 초기 추정을 제공하게 하여 물리적 일관성과 다양한 로봇 형태 및 물리적 매개변수에서의 강인함을 보장합니다.

- **Performance Highlights**: 우리는 생성된 데이터셋을 활용하여 여러 로봇 형태, 특히 이중 로봇팔 및 플로팅 기반 Allegro 손에서 접촉이 풍부한 조작 과제를 위한 정책을 학습했습니다. 제로샷 하드웨어 배치에서 높은 성공률을 달성하며, 실제 시나리오에서 데이터셋 활용의 유용성을 강조합니다. 이와 같은 방식은 앞으로 로봇 시스템 개발 및 학습 과정에서 의미 있는 기여가 될 것입니다.



### Multi-Turn Code Generation Through Single-Step Rewards (https://arxiv.org/abs/2502.20380)
Comments:
          9 pages (not including references or appendix); 6 figures (in main paper); (v1) preprint

- **What's New**: 이번 논문은 실행 피드백을 기반으로 한 다단계 코드 생성 문제를 다룹니다. 기존 방법론은 피드백 없이 코드를 생성하거나 복잡한 강화 학습(reinforcement learning)을 사용합니다. 우리는 단일 단계 보상만을 활용하여 문제를 해결하는 간단하면서도 확장 가능한 방법인 μCode를 제안합니다. 이 접근법은 코드 생성 프로세스를 효율적이고 안정적으로 만듭니다.

- **Technical Details**: μCode는 다단계 실행 피드백을 통해 코드 생성기를 훈련시키는 새로운 프레임워크입니다. 이 방법은 마르코프 결정 과정(Markov Decision Process, MDP)의 개념을 사용하여 각 상호작용에서 나온 중간 상태에서 올바른 코드를 단일 단계로 회복할 수 있음을 보여줍니다. 훈련 과정에서 생성기와 검증기를 동시에 개선하는 전문가 반복(expert iteration) 프레임워크를 사용합니다. 또한, 실시시간(inference time) 확장을 위해 학습된 검증기를 이용해 코드를 선택합니다.

- **Performance Highlights**: 실험 결과, 우리의 μCode 방식은 MBPP(Austin et al., 2021)와 HumanEval(Chen et al., 2021) 벤치마크에서 가장 유력한 다단계 접근법들을 초월한 성능 향상을 보였습니다. 학습된 검증기를 활용하여 더 나은 생성기 학습이 이루어짐을 증명하였고, 높은 추론 예산이 있는 경우에도 유망한 확장법 트렌드를 보여주었습니다.



### PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation (https://arxiv.org/abs/2502.20377)
- **What's New**: PhantomWiki는 고유하고 사실적으로 일관된 문서 집합체를 생성하는 파이프라인으로, 기존 데이터셋의 한계를 극복하고자 제안되었습니다. 매번 평가 시에 새로운 PhantomWiki 인스턴스를 생성함으로써 데이터 유출 및 성과 부풀림 문제를 피할 수 있습니다. 이 방식은 LLM(대형 언어 모델)의 추론(Reasoning)과 검색(Retrieval) 능력을 분리하여 평가할 수 있는 새로운 기준을 제공합니다.

- **Technical Details**: PhantomWiki는 질문 난이도와 코퍼스 크기를 조정하여 LLM의 추론과 검색 능력을 체계적으로 분석할 수 있게 설계되었습니다. 이는 특정 코퍼스의 맥락 창 안에 필요한 정보를 적재하고, 검색 방법을 통해 외부 정보를 접근해야 하는 복잡한 상황을 포함합니다. 이로써, LLM의 내부 지식 의존성을 평가하고, 다양한 기법을 사용할 수 있는 가능성을 모색합니다.

- **Performance Highlights**: PhantomWiki에서의 평가는 상태 최상의 LLM이 직면하는 도전을 잘 보여줍니다. 다양한 문서 코퍼스에 대한 질문을 처리할 때, F1 점수는 논리적이거나 기술적인 복잡성이 증가할수록 급락하는 경향을 보였습니다. PhantomWiki는 연구 커뮤니티가 LLM의 성능을 평가하고 개선할 수 있는 견고한 기준을 제공하며, 이와 관련된 코드가 추후 공개될 예정입니다.



### Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization (https://arxiv.org/abs/2502.20364)
Comments:
          10 pages, 6 figures, 5 tables

- **What's New**: 이번 논문에서 소개하는 Agentic Generative AI 기술은 대규모 언어 모델(LLMs)과 Retrieval-Augmented Generation (RAG), 지식 그래프(KGs), 벡터 스토어(VSs)를 통합하여 법률 시스템, 연구, 추천 시스템 등 다양한 전문 분야에 적용할 수 있는 혁신적인 기술입니다. 이 기술은 방대한 비구조적 및 반구조적 데이터 세트 내의 관계를 추론하는 데 뛰어난 성능을 보이며, 법률 문서의 복잡한 네트워크를 탐색하는 데 필수적인 통찰력을 제공합니다. 또한, 이 시스템은 비정형 법률 텍스트를 효과적으로 수집하고 분석하여, AI가 사례와 법령 간의 복잡한 연결을 파악할 수 있도록 지원합니다.

- **Technical Details**: 법률 분야는 헌법, 법령, 규정, 사례 법 등을 포함하여 각기 다른 구조적 논리를 따르는 복합적인 데이터 유형으로 구성되어 있습니다. 기존의 키워드 기반 검색을 초월하여, RAG 시스템은 관련 법적 문서나 데이터 포인트를 검색하고 언어 모델을 사용하여 이를 통합된 맥락 기반의 답변으로 전환하는 방식을 사용합니다. 이 논문은 VS, KG, NMF라는 세 가지 핵심 기술을 통합하여 법률 데이터의 효과적인 탐색, 검색 및 해석을 지원합니다.

- **Performance Highlights**: 이 시스템은 법률 문서의 클러스터링, 요약 및 교차 참조를 가능하게 하여 법률 정보 검색의 질과 해석 가능성을 높입니다. 또한, 벡터 스토어의 높은 재현율과 지식 그래프의 구조적 관계, NMF의 주제 발견 능력을 결합하여 이 시스템은 매우 큰 데이터셋에 대해 설명 가능한 추론을 제공할 수 있습니다. 이러한 접근 방식은 정의된 기준을 충족하며, 법률 시스템 내의 정당성과 운영 효율성을 높이는 데 중요한 역할을 합니다.



### Bridging the Creativity Understanding Gap: Small-Scale Human Alignment Enables Expert-Level Humor Ranking in LLMs (https://arxiv.org/abs/2502.20356)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 창의적 콘텐츠 이해의 한계를 다시 조명하며, 특히 유머 이해 과정에서의 인간과 LLM 간의 큰 격차를 밝혀냈습니다. 연구진은 유머 이해를 시각적 이해, 만화-캡션 추론, 인간 선호도 정렬의 세 가지 구성 요소로 분해하였고, 이를 통해 유머 캡션 순위에서 82.4%의 정확도를 달성하여 이전의 67% 기준선을 크게 초월했습니다. 또한, 잠재적 편향을 줄이기 위해 인간 선호 데이터로의 세심한 조정이 중요하다는 점을 강조하였습니다.

- **Technical Details**: 이 연구는 유머 캡션 순위의 세 가지 주요 구성 요소를 제시합니다: 시각적 이해(visual understanding), 유머 추론(humor reasoning), 그리고 인간 선호도 정렬(alignment with human preferences)입니다. 연구진은 시각적 주석과 LLM이 생성한 설명을 개선하여 유머 캡션에서의 비주얼 이해와 추론 능력을 크게 향상시켰습니다. 특히, 각 캡션 쌍에 대한 인간 기호 데이터를 통한 세밀한 튜닝(fine-tuning)이 성과를 높이는 데 중요한 역할을 했습니다.

- **Performance Highlights**: 이 연구는 캡션 순위 성능을 67%에서 82.4%로 증가시키며 인간 전문가들과 동등한 수준에 도달했습니다. 연구진의 실험 결과, 다양한 페르소나 기반 접근 방식은 효과가 미미했으나, 군중 선호 데이터를 사용한 세밀한 조정은 매우 효과적임을 발견했습니다. 이러한 성과는 AI가 창의적 과제에서 인간의 개별 및 집단 선호를 이해하는 데 있어 광범위한 도전 과제를 나타냅니다.



### Naturalistic Computational Cognitive Science: Towards generalizable models and theories that capture the full range of natural behavior (https://arxiv.org/abs/2502.20349)
- **What's New**: 이 논문에서는 인공지능(AI)의 발전이 인지과학(cognitive science)에 미치는 영향을 탐구하고, 자연주의적(naturalistic) 실험 Paradigms와 그것에 맞춘 모델의 중요성을 강조합니다. AI는 점점 더 자연적인 자극과 행동을 반영하는 모델을 생성하고 있으며, 이러한 진전이 인지과학에서 더 깊은 이해를 가능하게 할 수 있다는 것을 주장합니다. 연구자들은 실험 설계를 개선하여 더 자연스러운 상황을 모방하는 데 집중해야 하며, 이 과정에서 인공지능의 최신 결과를 활용할 수 있습니다.

- **Technical Details**: 이 연구는 인지과학에서 일반화 가능한 이해를 발전시키기 위해 자연주의적(realistic) 실험 Paradigms의 필요성을 강조합니다. 특히, 다양한 변수와의 상호작용을 고려한 작업 Paradigms의 확장을 통해 실험의 생태적 타당성(ecological validity)을 높이고, 자연 시스템이 해결하는 계산 작업을 실제로 잘 반영하는 모델을 개발해야 한다고 주장합니다. 이러한 접근법은 인공지능 분야에서 관찰된 데이터 기반 모델 개발 과정을 바탕으로 하여, 모델의 일반화와 예측 정확성을 극대화할 수 있습니다.

- **Performance Highlights**: 자연주의적 데이터에서 학습한 모델은 다양한 작업을 수행할 수 있으며, 단순화된 환경에서 훈련된 모델과는 질적으로 다른 일반화 패턴을 생성합니다. 이러한 차이는 모델이 어떻게 작동하는지, 그리고 인지 및 신경 현상의 기원(origin)에 대한 새로운 통찰을 제공할 수 있습니다. 따라서, 자연주의적 실험을 추구하고 이에 기반한 모델을 개발하는 것은 인지 이해의 완전성을 달성하는 데 필수적이라는 결론에 도달합니다.



### Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners (https://arxiv.org/abs/2502.20339)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 성능을 향상시키기 위해 계산 리소스를 테스트 시 확장하는 전략을 모색합니다. 특히, 단순하고 고속의 하위 이차(subquadratic) 아키텍처 모델이 인프라 예산에 따라 기존 Transformer보다 더 높은 성능을 낼 수 있는지를 조사합니다. 이 과정에서 Mamba라는 새로운 모델을 도출하여, 기존의 대형 모델보다 더 효율적인 추론 성능을 보여주는 새로운 가능성을 열었습니다.

- **Technical Details**: 상세적으로, 연구팀은 하이브리드 및 순수 Mamba 모델을 선행 학습된 Transformer로부터 증류(distill)하여 생성했습니다. 이들 모델은 80억 개의 토큰으로 훈련되었고, 고속 추론과 스케일링에서 강력한 성능을 발휘합니다. 이 모델들의 특징은 긴 생성 시퀀스와 대량 배치에 대한 메모리 소모를 줄이면서도 뛰어난 수학적 추론 성능을 유지하는 것입니다.

- **Performance Highlights**: 연구의 결과, 순수 및 하이브리드 Mamba 모델은 기존 Transformer 모델과 비교하여 MATH 및 GSM8K 수학 추론 과제에서 더 좋은 범위와 정확성을 보여주었습니다. 특히, 이러한 모델들은 동일한 품질의 결과를 더 적은 추론 시간으로 달성할 수 있으며, 이는 효율성과 추론 능력 간에 더 나은 균형을 이루고 있음을 보여줍니다.



### Expertise Is What We Wan (https://arxiv.org/abs/2502.20335)
Comments:
          18 pages, 7 figures, 5 tables

- **What's New**: 이 논문에서는 Large Language Expert (LLE)라는 새로운 응용 아키텍처를 소개합니다. LLE는 Large Language Models (LLMs)의 유연성과 강력함을 Expert Systems의 해석 가능성, 설명 가능성, 신뢰성과 결합한 시스템입니다. 이를 통해 암 진단 및 치료를 위한 작업을 지원할 수 있으며, 기존의 가이드라인과 데이터 통합의 어려움을 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LLE 아키텍처는 규칙 기반 시스템과 LLM의 강점을 결합하여 실제 의료 데이터를 다루는 복잡한 임상 가이드라인을 적용합니다. 이 시스템은 자연어와 구조화된 논리로 구성된 지식 기반으로 운영되며, LLM을 통해 진단 작업의 공백을 신속히 식별합니다. LLE는 또한 표준화된 구조와 적절한 로직을 유지함으로써, 의사들이 신뢰할 수 있는 결정을 내릴 수 있도록 돕습니다.

- **Performance Highlights**: LLE 시스템은 실제 환자의 비구조적 건강 기록을 분석하여 높은 임상 정확도(95% 이상)를 달성했습니다. 암 진단 스크리닝과 치료에 있어서 가이드라인 기반 의사 결정을 지원하며, 최적의 환자 결과를 위한 시의적절한 치료 시작을 보장합니다. 이 시스템은 대형 학술 센터의 유방암 및 대장암 환자 데이터로 확인된 여러 격차를 효과적으로 해결했습니다.



### Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models (https://arxiv.org/abs/2502.20332)
- **What's New**: 이 연구는 대형 언어 모델인 Llama3-70B에서 추상적 규칙 유도를 지원하는 내부 메커니즘을 심층적으로 조사하고, 상징적 기제를 구현하는 새로운 아키텍처를 제시합니다. 모델은 입력 토큰을 토대로 추상 변수로 변환하고, 이러한 변수를 이용해 순서 유도를 수행한 뒤, 마지막으로 다음 토큰을 예측하는 세 단계를 통해 추상적 추론을 지원합니다. 이러한 결과는 신경망의 emergent reasoning(출현적 추론)이 상징적 메커니즘의 출현에 의존함을 시사합니다.

- **Technical Details**: 연구는 Llama3-70B 모델의 세 가지 주요 단계 아키텍처를 설명합니다. 첫 번째 단계에서, 입력 토큰이 관계를 기반으로 추상 변수로 변환되고, 두 번째 단계에서는 이러한 변수들에 대한 순서 유도가 이루어집니다. 마지막 단계에서는 미리 예측된 추상 변수에 관련된 값을 검색하여 다음 토큰을 예측하는 기능을 수행합니다. 이러한 과정은 symbol abstraction heads(기호 추상화 헤드), symbolic induction heads(상징적 유도 헤드), retrieval heads(검색 헤드)로 각각 명명된 주의를 통해 처리됩니다.

- **Performance Highlights**: Llama3-70B는 주어진 문제에 대해 95%의 정확도로 의사 결정 문제를 해결하여, 모델이 기호 처리 메커니즘을 가지고 있음을 보여줍니다. 이 연구는 기호적 접근법과 신경망 접근법 간의 오랜 논쟁을 해결하는데 기여할 뿐만 아니라, 추상 규칙 유도를 효과적으로 수행하는 데 필요한 구조화된 메커니즘을 밝혀냄으로써 LLM(대형 언어 모델)의 능력에 대한 새로운 통찰을 제공합니다.



### Deep Reinforcement Learning based Autonomous Decision-Making for Cooperative UAVs: A Search and Rescue Real World Application (https://arxiv.org/abs/2502.20326)
Comments:
          18 Pages, 21 Figures

- **What's New**: 이 논문은 GNSS에 의존하지 않는 실내 환경에서 운영되는 다수의 드론 시스템을 위한 자율 안내, 내비게이션 및 작업 분배를 위한 포괄적인 프레임워크를 제안합니다. 이를 위해 Deep Reinforcement Learning (DRL) 기반의 안내 메커니즘을 사용하고, Twin Delayed Deep Deterministic Policy Gradient 알고리즘을 이용하여 드론의 움직임을 세밀하게 조정합니다. 또한, DRL 훈련된 그래프 합성곱 네트워크(Graph Convolutional Network, GCN)를 도입해 협력 UAV 간 작업 분배 문제를 해결하고 있습니다.

- **Technical Details**: 제안된 시스템은 LIDAR-SLAM을 통한 정밀한 위치 추적, DRL을 통한 자율 내비게이션, 그리고 Graph Attention Network (GAT) 기반의 작업 할당기로 구성되어 있습니다. 이 시스템은 NATO Sapience Autonomous Cooperative Drone Competition의 요구 사항에 맞춰 개발되어 있으며, 드론 간의 실시간 협조 및 작업 분배를 최적화합니다. 이를 통해 드론은 탐색 및 구조 작업 중에 더 효율적으로 협력할 수 있습니다.

- **Performance Highlights**: 테스트를 통해 제안된 시스템은 뛰어난 성과를 기록하며 2024 Sapience 대회에서 1위로 마무리되었습니다. DRL 기반의 안내 및 GCN을 이용한 작업 할당기가 복잡하고 장애물로 가득한 환경에서 유연하고 효율적인 탐색을 가능하게 하여 성과를 높였습니다. 이 연구는 자율 드론 기술이 재난 구조 작업에 혁신을 가져올 수 있는 잠재력을 보여줍니다.



### UniTok: A Unified Tokenizer for Visual Generation and Understanding (https://arxiv.org/abs/2502.20321)
- **What's New**: 본 논문에서는 시각 생성과 이해를 통합한 새로운 프레임워크로 UniTok를 제안합니다. UniTok는 정교한 세부 정보를 인코딩하면서도 고수준의 의미를 포착할 수 있는 이산 시각 토크나이저입니다. 연구진은 전통적인 코드북의 한계를 극복하기 위해 멀티 코드북 양자화를 도입하여 잠재 기능 공간을 확장합니다.

- **Technical Details**: UniTok는 분할된 독립적인 서브 코드북을 사용하여 시각 토큰을 여러 조각으로 나누고 각 조각을 양자화합니다. 이러한 접근 방식은 코드북을 너무 크게 만드는 데서 오는 훈련의 불안정성을 피하면서도 표현력을 극대화합니다. 실험 결과에서는 UniTok가 Domain-specific continuous 토크나이저와 비슷하거나 높은 성능을 보이며, 이미지넷에서 FID 0.38을 달성합니다.

- **Performance Highlights**: UniTok는 78.6%라는 제로샷 정확도를 기록하며, 기존의 Domain-specific 토크나이저를 능가하는 성능이 확인되었습니다. 이 모델은 다중모드 이해와 생성 벤치마크에서 최첨단의 성능을 보여주며, 통합된 MLLM에서의 안정성과 효율성을 증명합니다.



### Mixture of Structural-and-Textual Retrieval over Text-rich Graph Knowledge Bases (https://arxiv.org/abs/2502.20317)
- **What's New**: 이번 연구에서는 Text-rich Graph Knowledge Bases (TG-KBs)에서 텍스트와 구조적 지식을 함께 효과적으로 검색하기 위한 Mixture of Structural-and-Textual Retrieval (MoR) 방법을 제안합니다. 기존의 검색 방법들은 일반적으로 이러한 지식을 분리하여 검색하는 경향이 있으며, 구조적 검색을 완전히 우회하는 경우도 많습니다. MoR은 플래닝-추론-조직화(Planning-Reasoning-Organizing) 프레임워크를 통해 두 종류의 지식을 통합적으로 검색하고, 이를 통해 서로의 이점을 강화합니다.

- **Technical Details**: MoR은 세 가지 주요 단계로 구성됩니다. 첫 번째 단계에서 MoR은 쿼리에 대한 계획 그래프를 생성하여 텍스트 계획을 수립합니다. 두 번째 단계에서는 구조적 탐색과 텍스트 매칭을 결합하여 TG-KB에서 후보를 얻습니다. 마지막으로, 조직 단계에서는 구조적 경로를 기반으로 가져온 후보를 재정렬하는 구조 인식 재정렬기(Structure-aware Rerank)를 적용합니다.

- **Performance Highlights**: MoR은 기존의 검색 방법과 비교했을 때 구조적 및 텍스트적 검색의 조화를 통해 우수한 성능을 보입니다. 실험 결과는 서로 다른 쿼리 논리에 따른 고르지 않은 검색 성능을 보여주며, 후보 재정렬에서 구조적 경로를 통합할 때의 이점을 강조합니다. MoR의 구현 코드는 지정된 링크에서 확인할 수 있습니다.



### Multi-Scale Neighborhood Occupancy Masked Autoencoder for Self-Supervised Learning in LiDAR Point Clouds (https://arxiv.org/abs/2502.20316)
- **What's New**: 본 연구에서는 LiDAR 포인트 클라우드를 위한 Neighborhood Occupancy MAE (NOMAE)를 제안하여 기존 자가 지도 학습(self-supervised learning, SSL)의 한계를 극복합니다. NOMAE는 비가려진(visible) 부피의 이웃에서만 마스킹된 옥타(occupancy)를 재구성하는 방식을 채택해 정보 유출을 방지합니다. 또한, 다양한 크기의 객체의 특성을 캡처하기 위해 다중 스케일(multi-scale)에서 옥타 재구성을 통합해 새로운 경량 구조를 제공합니다.

- **Technical Details**: NOMAE는 LiDAR 포인트 클라우드의 희소성 문제를 다중 스케일에서 직접 해결하는 첫 번째 자가 지도 학습 프레임워크입니다. 이 프레임워크는 한阶에 걸쳐 자체 감독(self-supervision) 과정을 통해 여러 스케일에서 특성을 캡처할 수 있도록 설계되었습니다. 입력 포인트 클라우드는 볼륨 화(voxelization) 및 마스킹 과정을 거치며, 이를 통해 우리는 특성에 대한 정보를 효과적으로 유지합니다.

- **Performance Highlights**: NOMAE는 nuScenes와 Waymo Open 데이터셋에서의 평가를 통해 여러 인식(perception) 작업에서 최신 기술 수준으로 성능을 입증했습니다. 본 연구는 다양한 다운스트림 작업(semantic segmentation 및 3D object detection)에서 기존의 판별적(discriminative) 및 생성적(generative) SSL 방법과 비교하여 우수한 성능을 보여주었습니다. 특히 NOMAE는 여러 벤치마크에서 최상의 결과를 달성하며 수행되고 있습니다.



### LangProBe: a Language Programs Benchmark (https://arxiv.org/abs/2502.20315)
- **What's New**: 이번 논문에서는 LangProBe라는 새로운 대규모 벤치마크를 소개하고 있습니다. LangProBe는 2000개 이상의 작업, 아키텍처, 최적화 기법, 언어 모델(LM) 조합을 평가하여 언어 프로그램 아키텍처와 최적화 전략의 영향을 연구합니다. 이 연구는 다양한 라인업의 언어 프로그램을 통해 최적화된 언어 프로그램이 비용 대비 품질을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: LangProBe의 기반이 되는 연구는 DSPy와 TextGrad와 같은 선언적 언어 프레임워크를 통해 언어 프로그램을 작성하고 자동화하는 접근 방식을 활용하고 있습니다. 이러한 프로그램은 외부 도구와의 통합 및 정보 흐름을 구성하고, 특히 외부 정보에 대한 접근을 요구하는 작업에 필수적입니다. 또한, MIPRO과 같은 최적화 기법들은 다양한 모델과 작업 조합에 대해 품질 향상을 제공하는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, 최적화된 언어 프로그램은 일반적인 모델 호출 방식보다 우수한 성능을 보이는 것으로 나타났습니다. 예를 들어, gpt-4o-mini에서 실행되는 최적화된 프로그램은 낮은 비용으로 뛰어난 성능을 발휘했습니다. 그러나 모든 문제에서 일관된 결과를 보이는 것은 아니며, 고급 모델을 통한 기본 문제 해결에 있어서는 추가적인 조합이나 최적화가 필요하지 않았습니다.



### M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging (https://arxiv.org/abs/2502.20301)
Comments:
          38 pages, 7 figures

- **What's New**: 본 논문에서는 의료 이미징 분야에서의 머신 러닝(ML) 자동화를 위한 새로운 다중 에이전트 시스템인 M3Builder를 소개합니다. M3Builder는 복잡한 다단계 워크플로우를 관리하는 네 개의 전문화된 에이전트가 협력하여 데이터 처리, 환경 구성, 자동 디버깅 및 모델 훈련을 수행합니다. 이 시스템은 의학적 이미징 ML 작업을 위한 통합 환경인 Medical Imaging ML workspace와 함께 작동하여, AI 도구의 자율적 개발을 가능하게 합니다.

- **Technical Details**: M3Builder는 의료 이미징 분석을 위한 문제를 정의하고 이를 자동으로 해결하기 위해 다중 에이전트 협업 프레임워크를 적용합니다. 해당 프레임워크는 기계 학습(ML) 워크스페이스와 다중 에이전트 시스템의 두 가지 주요 구성 요소로 나뉘며, 여기에는 자연어로 설명된 데이터 카드, 툴셋 설명, 코드 템플릿이 포함됩니다. 이러한 요소들은 에이전트들이 상호작용하며 작업을 이행할 수 있도록 지원하는 구조적 환경을 제공합니다.

- **Performance Highlights**: M3Builder는 Claude-3.7-Sonnet을 에이전트의 핵심으로 사용하여 94.29%의 성공률을 기록하며, 기존의 ML 에이전트 디자인 대비 우수한 성능을 보여줍니다. 실험 결과는 장기 데이터 분할에서 86.67%, 이상 탐지에서 100%, 질병 진단에서 95%, 보고서 생성에서 93.33%의 성과를 나타냅니다. 이 결과는 의료 이미징에서의 완전한 자동화된 머신 러닝의 가능성을 시사합니다.



### An exploration of features to improve the generalisability of fake news detection models (https://arxiv.org/abs/2502.20299)
Comments:
          Accepted at Expert Systems with Applications (Elsevier)

- **What's New**: 이 논문은 가짜 뉴스 탐지의 일반화 가능성을 높이기 위한 연구로, 기존의 불완전한 라벨링 데이터에 대한 문제를 다룹니다. 연구에서는 TF-IDF 및 BERT와 같은 토큰 기반 모델이 편향된 데이터에 민감하다는 점을 시사하며, 스타일적 특징(lexical, syntactic, semantic)과 사회적 모니타이제이션(social-monetisation) 특징의 중요성을 강조합니다. 이외에도 그 동안 제한적으로 활용된 대규모 언어 모델(LLMs)의 적합성에 대한 평가도 진행합니다.

- **Technical Details**: 논문에서는 NELA 2020-21 데이터 세트를 사용하여 훈련하고, 수동 라벨링 된 Facebook URLs 데이터 세트를 이용해 일반화 가능성을 평가합니다. 연구는 스타일적 특징 및 사회적 모니타이제이션 특징이 토큰 기반 방법보다 더 일반화 가능한 예측을 제공한다는 주장을 통해 성능을 분석합니다. 더불어, 통계적 및 순열 특징 중요성 분석을 통해 데이터 세트 편향을 완화하고 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과 토큰 기반 모델은 편향된 데이터에서 훈련될 경우 30% 정도의 정확도 저하를 겪는 반면, 스타일적 및 사회적 모니타이제이션 특징은 일반화 가능성을 증대시키며 더 나은 성과를 보여줍니다. 또한, LLaMa와 같은 대규모 언어 모델들이 가짜 뉴스 탐지에 효과적이지 않다는 제한적인 증거를 제시합니다. 결과적으로, 스타일적 특징과 경제적 동기를 이해하는 것이 가짜 뉴스 탐지의 발전에 기여할 수 있음을 강조합니다.



### Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription (https://arxiv.org/abs/2502.20295)
Comments:
          11 pages (including references and appendix), 14 figures, accepted at AAAI-25 Workshop on Document Understanding and Intelligence, non-archival

- **What's New**: 이 논문은 다중 페이지의 손글씨 문서를 제로샷(zero-shot) 설정에서 전사(transcribe)하는 데 있어 다중 모달 대형 언어 모델(MLLMs)의 활용을 탐구합니다. 기존의 OCR 엔진이 인쇄된 텍스트에 강력한 성능을 보이는 반면, 손글씨 처리에서는 제한적이기 때문에 MLLMs를 엔드 투 엔드 전사기로 활용하거나 후처리(post-process)기로 사용하는 다양한 구성에 대해 고찰합니다. 주목할 만한 점은, '+first page'라는 새로운 방법을 제안하며, 이는 전체 문서의 OCR 출력을 제공하면서 첫 페이지 이미지만 활용하여 MLLM 전사 정확도를 높입니다.

- **Technical Details**: 결과적으로, MLLM은 손글씨 문서의 여러 페이지에 걸쳐 공통적인 서식(formatting) 및 맥락적(feature) 정보를 활용합니다. OCR 시스템은 페이지 단위로 동작하는 반면, MLLM은 이러한 페이지 간의 종속성을 고려하여 전사 작업을 향상시킵니다. 이 연구에서는 IAM 손글씨 데이터베이스를 사용하여 제안된 방법의 효과를 검증하고, MLLM이 단일 페이지에서 배운 서식 및 OCR 오류 패턴을 활용하여 전체 문서를 향상시키는 것을 확인합니다.

- **Performance Highlights**: 실험 결과, '+first page' 접근 방식이 전사 정확도를 개선하고 비용과 성능 간의 균형을 이룬다는 것이 나타났습니다. 또한, 이 방법은 고가의 MLLM과 비교적 저렴한 OCR 방법 간의 제휴를 통해, 문서 내의 서식과 오류 패턴을 외삽(extrapolate)하여 성능을 향상시켰습니다. 이러한 결과는 다중 페이지 손글씨 문서 전사 작업에서 MLLM의 가능성을 제시하며, 향후 다양한 분야에서 활용될 수 있습니다.



### Explainable, Multi-modal Wound Infection Classification from Images Augmented with Generated Captions (https://arxiv.org/abs/2502.20277)
- **What's New**: SCARWID는 합성 캡션 증가 검색(Synthetic Caption Augmented Retrieval) 기술을 이용하여 당뇨병성 발 궤양(Diabetic Foot Ulcers, DFUs)에서 감염을 탐지하는 새로운 딥 러닝 프레임워크입니다. 기존 머신 러닝 방법들은 일반적으로 상처 이미지만 분석했지만, SCARWID는 GPT-4o로 생성된 텍스트 설명을 강화하여 감염 탐지를 개선합니다. 이 접근법은 감염 상태를 판별하기 위해 레이블이 붙은 지원 세트에서 유사한 아이템을 검색합니다.

- **Technical Details**: SCARWID는 두 가지 주요 구성 요소로 구성됩니다: (1) Wound-BLIP, 이는 GPT-4o로 생성된 설명을 활용하여 이미지로부터 일관된 캡션을 합성하는 비전-언어 모델(Vision-Language Model)입니다. (2) 이미지-텍스트 융합 모듈(Image-Text Fusion module)은 상관 주의(cross-attention)를 사용하여 이미지와 그에 해당하는 Wound-BLIP 캡션으로부터 교차 모달 임베딩을 추출합니다. SCARWID는 5개의 가장 유사한 이미지-텍스트 쌍을 검색하여 감염 상태를 결정합니다.

- **Performance Highlights**: SCARWID는 상처 감염 분류에서 평균 민감도 0.85, 특이도 0.78, 정확도 0.81을 기록하며 기존 모델들을 초월했습니다. 특히, 생성된 캡션을 상처 이미지와 함께 표시하여 간호사들이 SCARWID의 출력을 의학적 지식과 조화롭게 연결할 수 있도록 해줍니다. 이러한 접근은 특히 상처 노트가 없거나 시각적 속성을 파악하기 어려운 초보 간호사들에게 큰 가치를 제공합니다.



### HVI: A New color space for Low-light Image Enhancemen (https://arxiv.org/abs/2502.20272)
Comments:
          *These authors contributed equally to this work

- **What's New**: 이 논문에서는 저조도 이미지 향상(Low-Light Image Enhancement, LLIE)을 위한 새로운 색상 공간인 Horizontal/Vertical-Intensity (HVI)를 제시합니다. 기존의 LLIE 방법들은 sRGB 색상 공간을 기반으로 하여 색편향(color bias)과 밝기 아티팩트(brightness artifacts)를 발생시켰습니다. HVI는 극성화된 HS 맵 및 학습 가능한 강도를 통해 이러한 문제를 해결하고자 합니다.

- **Technical Details**: HVI 색상 공간은 빨간색 좌표 간의 거리를 최소화하여 빨간색 아티팩트 제거를 강제합니다. 또한, 낮은 조도 지역을 압축하여 검은색 아티팩트를 제거합니다. 이를 위해 새로운 Color and Intensity Decoupling Network (CIDNet)를 도입하여 HVI 공간에서 다양한 조명 환경에 따른 정확한 광도 매핑 기능을 학습합니다.

- **Performance Highlights**: 제안된 HVI 색상 공간과 CIDNet은 10개의 데이터셋에서 최신 기법(State-of-the-art methods)보다 뛰어난 성능을 보여줍니다. 종합적인 벤치마크 및 배제 실험 결과는 이 새로운 접근 방식의 유효성을 입증합니다. 코드는 제공된 URL에서 확인할 수 있습니다.



### Large Language Models as Attribution Regularizers for Efficient Model Training (https://arxiv.org/abs/2502.20268)
- **What's New**: 이 논문은 큰 언어 모델(LLMs)을 활용하여 훈련 효율성을 높일 수 있는 새로운 방법을 제안합니다. 특히, LLM에서 생성된 전역 태스크 피처 기여도(attribution)를 소형 네트워크의 훈련 과정에 통합하는 방법에 대해 설명하고 있습니다. 이 접근법은 LLM의 통찰력을 소형 모델에 결합하여, 특히 데이터가 불균형하거나 제한되어 있을 때 전반적인 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서 제안하는 Large Language Model Attribution Aligned Training (LAAT) 방법은 attribution-matching regularization 항을 도입하여 소형 모델의 학습 동역학을 LLM이 제공하는 동적인 통찰력에 맞추는 것을 목표로 합니다. 이 방식은 상대적으로 대규모 GPU 자원에 대한 의존성을 줄이고, 최소한의 계산 오버헤드로 LLM의 강력한 일반화 능력을 활용하여 설명 가능한 소형 모델의 훈련을 가능하게 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험을 통해 본 방법이 저샘플링(few-shot learning) 환경에서 모델의 학습 효율성과 강인성을 개선함을 보여줍니다. 본 연구에서 제안된 방법은 LLM에 대한 블랙박스 API 접근만으로도 용이하게 통합할 수 있으며, 데이터의 왜곡(skewness) 및 편치(bias)와 같은 문제를 해결하는 데 유용함을 입증했습니다.



### LLM as a Broken Telephone: Iterative Generation Distorts Information (https://arxiv.org/abs/2502.20258)
- **What's New**: 이 연구는 대형 언어 모델(LLM)이 반복적으로 자신의 출력을 처리할 때 정보 왜곡을 조장하는지를 조사합니다. '편의점 전화'(broken telephone) 효과에 영감을 받아, LLM의 반복 생성 과정을 통해 정보 왜곡이 축적되는지를 분석하였습니다. 실험 결과, 언어 선택과 체인 복잡성에 따라 왜곡이 증가하며, 전략적인 프롬프트 생성 기법을 통해 이를 완화할 수 있음이 밝혀졌습니다.

- **Technical Details**: 연구는 번역 기반 실험을 통해 LLM의 반복 생성에서의 정보 왜곡을 조사합니다.  각 반복(iteration) 과정에서 영어 문서가 다양한 언어로 번역된 후 다시 영어로 역번역되며, 이 과정에서 왜곡 정도와 사실성(factuality)을 측정하였습니다. 연구 결과는 중간 언어의 선택과 체인 복잡성에 따라 왜곡이 어떻게 영향을 받는지를 보여주고, 고온 조절(temperature control)과 제한된 프롬프트(use of restricted prompting) 기법으로 왜곡을 줄일 수 있다는 점을 강조합니다.

- **Performance Highlights**: 결과적으로 LLM의 반복 생성이 정보의 의미와 사실성을 손상시킬 수 있으며, 반복 처리에서 왜곡이 누적됨을 보여줍니다. 특히, 언어 구조의 유사성과 모델의 학습 데이터에 따라 왜곡 정도가 달라지며, 번역 체인의 복잡성이 클수록 왜곡이 더욱 커지는 경향이 있음을 보여주었습니다. 이러한 발견은 LLM이 생성한 콘텐츠의 신뢰성에 대한 우려를 제기하며, AI 기반 정보 확산의 장기적 영향에 대한 논의에 기여합니다.



### Teasing Apart Architecture and Initial Weights as Sources of Inductive Bias in Neural Networks (https://arxiv.org/abs/2502.20237)
Comments:
          11 pages, 6 figures, 6 tables

- **What's New**: 이 논문은 인공 신경망(artificial neural networks)의 초기 가중치(initial weights)가 지니는 귀납적 편향(inductive bias)의 영향을 탐구합니다. 저자들은 메타 학습(meta-learning)을 활용하여 특정 문제에 적합하게 조정된 초기 가중치를 찾는 방법을 제시하고, 이를 통해 다양한 신경망 아키텍처(architectures)를 비교합니다. 이 연구는 기본적인 아키텍처 외에도 초기 가중치가 학습 성능에 미치는 영향을 조명하여 인지 과학(cognitive science)과 기계 학습(machine learning) 간의 교차점을 강조합니다.

- **Technical Details**: 저자들은 MLPs(다층 퍼셉트론), CNNs(합성곱 신경망), LSTMs(장단기 메모리 네트워크) 및 Transformer's와 같은 네 가지 인기 있는 아키텍처를 사용하여 430개 모델을 메타 트레이닝(meta-training)했습니다. 연구는 이러한 타스크(task) 전반에서 메타 학습이 기존 아키텍처와 데이터 표현 간의 성능 차이를 어떻게 줄일 수 있는지 문서화합니다. 각 아키텍처는 고유한 초기 가중치를 통해 서로 다른 학습 편향을 실현하여 성능을 나타냅니다.

- **Performance Highlights**: 메타 학습을 통해 얻은 초기 가중치는 구조적 차원에서의 성능 차이를 감소시키고, 특정 아키텍처가 메타 학습을 통해 더 효과적으로 학습할 수 있도록 합니다. 하지만 학습 경험에서 멀리 떨어진 문제들에 대해서는 여전히 모든 아키텍처가 저조한 일반화(generalization) 성능을 보이며, 이는 더욱 강력한 귀납적 편향을 요구함을 시사합니다. 연구 결과는 적절한 초기 가중치와 학습률의 선택이 다양한 편향을 구현할 수 있도록 해 준다는 점에서 신경망 아키텍처의 유연성을 강조합니다.



### Selective Use of Yannakakis' Algorithm to Improve Query Performance: Machine Learning to the Rescu (https://arxiv.org/abs/2502.20233)
- **What's New**: 이 논문에서는 쿼리 최적화(query optimization)의 중요성을 강조하며, 제안된 최적화 기법이 모든 상황에서 성능 개선 효과를 보이지 않는 문제를 다룹니다. 저자들은 특정 쿼리에 대해 최적화 기법을 적용해야 할지를 결정하는 방법론을 제안합니다. 특히 Yannakakis 스타일 쿼리 평가를 최적화 기법으로 선정하여 이 문제를 해결하기 위한 접근 방식을 설명하고 있습니다.

- **Technical Details**: 제안된 방법론은 알고리즘 선택 문제(algorithm selection problem)로 공식화되며, 이를 해결하기 위해 머신러닝(Machine Learning) 기반의 접근 방식을 사용합니다. 저자들은 다양한 데이터베이스 시스템에서 여러 벤치마크를 통해 실험을 진행하였으며, 이를 통해 제안된 접근 방식의 유효성을 입증하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 통계적으로 유의미한 성능 향상(statistically significant performance improvement)을 가져오는 것으로 나타났습니다. 다양한 데이터베이스 환경에서의 성능 평가를 통해 최적화 기법의 적용 여부를 결정하는 알고리즘의 효율성을 강조합니다.



### RURANET++: An Unsupervised Learning Method for Diabetic Macular Edema Based on SCSE Attention Mechanisms and Dynamic Multi-Projection Head Clustering (https://arxiv.org/abs/2502.20224)
Comments:
          10 pages, 2 figures, 5 tables, submitted to The 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2025)

- **What's New**: 이 논문에서는 당뇨병 환자들 사이에서 흔히 발생하는 합병증인 당뇨병성 황반부종(Diabetic Macular Edema, DME) 진단을 위한 새로운 자동화 시스템인 RURANET++를 소개합니다. 이 시스템은 기존의 데이터 주석 및 주관적인 안과 의사 평가에 의존하지 않고, 비지도 학습(un) 기반으로 설계되었습니다. DME 진단의 실용적인 응용을 위해 기존의 문제들을 해결하는 방안을 제안합니다.

- **Technical Details**: RURANET++는 최적화된 U-Net 아키텍처와 공간적 및 채널 압축 및 자극(SCSE) 주의 메커니즘을 결합하여 병변 특성 추출을 개선합니다. 이 시스템은 사전 훈련된 GoogLeNet 모델을 사용하여 망막(retinal) 이미지에서 깊은 특징을 추출하고, PCA(Principal Component Analysis)를 활용하여 효율성을 위해 50차원으로 차원 축소를 수행합니다. 또한, 다중 투영 헤드를 사용하는 새로운 클러스터링 알고리즘을 도입하여 클러스터 다양성을 제어하고 유사성 임계값을 동적으로 조정하여 클래스 내 일관성과 클래스 간 차별성을 최적화합니다.

- **Performance Highlights**: 실험 결과는 여러 지표에서 뛰어난 성능을 보여주며, 최대 정확도(accuracy)는 0.8411, 정밀도(precision)는 0.8593, 재현율(recall)은 0.8411, F1 점수(F1-score)는 0.8390에 도달했습니다. 이러한 결과는 뛰어난 클러스터링 품질(clustering quality)과 함께 비지도 진단(un) 솔루션의 효율성을 입증합니다. 이 연구는 DME 진단에 중요한 임상적 의미를 제공합니다.



### Deep Convolutional Neural Networks for Palm Fruit Maturity Classification (https://arxiv.org/abs/2502.20223)
- **What's New**: 이 연구는 최적의 성숙 단계에서 팜 과일을 수확하여 팜 오일의 수량과 품질을 극대화하는 것을 목표로 하며, 자동화된 컴퓨터 비전 시스템을 개발하여 팜 과일 이미지를 다섯 가지 상태로 분류합니다. 딥 컨볼루션 신경망 (CNN)을 사용하여 과일의 성숙 단계에 따라 이미지를 분류하고, 사전 훈련된 ResNet50 및 InceptionV3 아키텍처를 활용하여 전이 학습과 미세 조정을 적용합니다.

- **Technical Details**: 이 연구에서는 평균 85% 이상의 정확도로 팜 과일 성숙 단계를 분류하기 위한 딥 CNN 모델을 제안하였습니다. 8,000장 이상의 이미지로 구성된 공개 데이터셋을 사용하여 80%는 훈련, 20%는 테스트로 나누어 실험을 수행했으며, 이 과정에서 CNN의 효과적인 특징 추출 능력을 활용합니다. 특히, 색상 모델의 RGB 및 HSI에서 특징을 추출하고, 이를 부가적인 분류 알고리즘에 주입하여 성숙 단계를 분류합니다.

- **Performance Highlights**: 제안된 딥 CNN 모델은 팜 과일 성숙 단계를 분류하는 데 있어 85% 이상의 높은 테스트 정확도를 기록하며, 이는 자동화된 성숙도 평가의 잠재력을 강조합니다. 이 연구는 팜 오일 생산 효율 증대와 수확 결정 최적화를 위한 기여로 나아갈 수 있는 중요한 성과를 이루었습니다.



### DIPSER: A Dataset for In-Person Student1 Engagement Recognition in the Wild (https://arxiv.org/abs/2502.20209)
- **What's New**: 본 논문에서는 학생들의 주의를 평가하기 위해 설계된 새로운 데이터셋이 소개되었습니다. 이 데이터셋은 RGB 카메라 데이터와 각 학생마다 여러 카메라를 통해 자세와 표정을 캡처하며, 개인별 스마트워치 센서 데이터도 포함하고 있습니다. 이 데이터셋은 머신러닝 알고리즘을 학습시켜 주의를 예측하고 감정과의 관계를 분석할 수 있도록 합니다.

- **Technical Details**: 이 데이터셋은 1,311,761개의 이미지를 포함하고 있으며, 고해상도 이미지와 함께 감정과 주의에 대한 레이블을 제공합니다. 또한 심박수, 가속도계, 자이로스코프 등의 IMU 데이터를 통합하여 다양한 변수를 분석할 수 있는 가능성을 열어줍니다. 비교적 단국한 시간에 진행된 기존 데이터셋들과는 달리 본 데이터셋은 5분 무비 영상을 포함하여 대면 교육 설정에서의 주의력을 더 정확히 측정할 수 있도록 합니다.

- **Performance Highlights**: 전체 데이터셋은 학생의 상호 작용을 다양한 교육적 문맥에서 포괄적으로 담고 있으며, 비교적 뚜렷한 메타데이터가 추가되어 있습니다. 기존의 데이터셋과 비교할 때, DIPSER 데이터셋은 특히 대면 클래스 환경에서의 학생 주의력 분석에 있어 가장 포괄적이며 다각적인 접근 방식을 제공합니다. 이로써 교육 분야의 인공지능 도구 연구에 기여할 수 있는 새로운 기회를 열어줍니다.



### Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies (https://arxiv.org/abs/2502.20190)
- **What's New**: 본 논문에서는 Deep Reinforcement Learning (DRL) 교육의 속도를 높이기 위해 TianJi라는 새로운 고처리량 분산 RL 훈련 시스템을 제안합니다. TianJi는 서브태스크 구성 요소 간의 할당 의존성을 완화하고, 이벤트 기반 비동기 통신을 가능하게 하여 훈련 효율성을 개선합니다. 이 시스템은 clear boundaries를 유지하면서도 샘플 생산과 소비의 균형을 맞추는 분산 전략을 통해 수렴 불확실성을 해결합니다.

- **Technical Details**: TianJi는 비동기적으로 느슨하게 결합된 프로세스로 서브태스크 구성 요소 간의 의존성을 완화합니다. 또한, 샘플의 질을 보장하며 훈련 속도를 높이기 위해 샘플의 신선도를 조절하는 전략을 도입합니다. 이를 통해, 다양한 RL 시스템 중에서 중요한 inter-component 의존성을 해결하여 높은 수준의 병렬 처리를 달성합니다.

- **Performance Highlights**: 실험 결과, TianJi는 관련 시스템과 비교할 때 최대 4.37배의 수렴 시간 가속 비율을 달성하였고, 8개의 컴퓨팅 노드에 스케일링할 경우 1.6배의 속도 향상과 7.13배의 처리량 향상을 보여주었습니다. 데이터 전송 효율성 실험 결과, TianJi는 다른 시스템에 비해 현저한 성능을 발휘하며, 하드웨어 한계에 근접한 효율성을 나타냈습니다.



### Accelerating Model-Based Reinforcement Learning with State-Space World Models (https://arxiv.org/abs/2502.20168)
- **What's New**: 이번 연구에서는 모델 기반 강화 학습(Model-based Reinforcement Learning, MBRL)의 학습 속도를 개선하기 위한 새로운 방법을 제안합니다. 제안된 방법은 상태-공간 모델(State-Space Models, SSMs)을 활용하여 동적 모델의 훈련을 병렬화하게 됩니다. 이 접근법은 복잡한 로봇 제어 작업에서도 MBRL의 학습 시간을 여러 배로 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 병렬 가능한 상태-공간 모델로서 모던한 SSM을 사용하여 MBRL의 세계 모델 훈련을 가속화합니다. 동적 모델에 대한 훈련 병렬화는 계산 복잡성을 줄이고, 일반적으로 MBRL 방법보다 더 빠른 훈련을 가능하게 합니다. 특히 균형 잡힌 시각적 정보를 제공하는 구조를 통해 부분 관찰 환경에서도 효과적으로 작용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 방법은 아지일 쿼드로터 비행 작업에서 최대 10배의 세계 모델 훈련 시간 단축과 4배의 전체 MBRL 훈련 시간 단축을 구현합니다. 실험 결과, 제안된 방법은 기존의 MBRL 방법과 유사한 샘플 효율성과 작업 보상을 달성하여 성능 손실 없이 훈련 속도를 크게 향상시켰습니다.



### Adaptive H&E-IHC information fusion staining framework based on feature extra (https://arxiv.org/abs/2502.20156)
- **What's New**: 이번 연구에서는 면역조직화학 염색(IHC) 이미지를 생성하기 위한 새로운 접근 방식을 소개합니다. 기존의 모델들이 H&E(헤마톡실린-오신) 이미지의 픽셀 특성을 기반으로 염색을 생성하는 데 한계가 있었지만, 제안된 방법은 VMFE(다중 스케일 기능 추출기) 모듈을 통해 이 문제를 해결합니다. 이 모듈은 염색 정보 기능을 효율적으로 추출하고 공유 디코더를 통해 이를 융합하여 더 정확한 IHC 이미지를 생성할 수 있도록 합니다.

- **Technical Details**: 연구에서 제안한 방법은 VMFE 모듈을 중심으로 구성되어 있으며, 이는 웨이브렛 변환 합성을 사용하여 H&E 이미지를 처리합니다. 또한, 크로스 어텐션 모듈을 통해 H&E 이미지에서 얻어진 특징 맵과 생성된 IHC 이미지의 속성을 융합하여 IHC 이미지 생성을 보다 정밀하게 이루어지도록 합니다. 대조 학습을 통해 HE 및 IHC 인코더를 사전 훈련하여, 라텐트 공간에서 HE 및 IHC 이미지의 염색 레이블을 정렬하고, 동적 L1 손실 메커니즘을 통해 정확도를 개선한 점이 특징입니다.

- **Performance Highlights**: 제안된 모델은 다양한 데이터 세트에서 테스트를 수행하여 우수한 성능을 입증하였습니다. 모델의 구조와 기능 향상을 통해 생성된 IHC 이미지에서의 정보 손실 및 비대칭 정보 문제를 효과적으로 해결하였습니다. 이를 통해 가상의 염색 과정을 개선하고, 특히 의료 분야에서의 활용 가능성을 높이는 데 기여하고 있습니다.



### QPM: Discrete Optimization for Globally Interpretable Image Classification (https://arxiv.org/abs/2502.20130)
- **What's New**: 이번 논문에서는 딥 뉴럴 네트워크의 분류를 이해하기 위한 새로운 접근 방식을 제시합니다. 최근 모델들이 단일 결정에 대해 지역적으로 설명할 수 있었던 반면, 정확한 모델의 전반적인 행동을 신뢰성 있게 설명하는 것은 여전히 도전 과제였습니다. 이를 해결하기 위해 Quadratic Programming Enhanced Model (QPM)을 도입하여 전 세계적으로 해석 가능한 클래스 표현을 학습합니다.

- **Technical Details**: QPM은 각 클래스를 5개의 특징으로 이진 할당(binary assignment)하여 표현합니다. 이 특징들은 다른 클래스와도 공유되어 대조적인 클래스 표현을 쉽게 비교할 수 있도록 설계되었습니다. 최적의 할당은 미리 정의된 유사성 측정 및 해석 가능성 제약을 기반으로 한 이산 최적화(discrete optimization)를 통해 찾아지며, 이 결과는 다양한 특징을 미세 조정(fine-tune)하는 데 사용됩니다.

- **Performance Highlights**: QPM은 소규모 및 대규모 데이터셋에서 전례 없는(global interpretability) 전세계적 해석성을 제공하며, 해석 가능한 모델의 정확도(state of the art)에서도 최고의 성과를 기록하였습니다. 이 모델은 안전-critical한 상황에서 사용될 수 있으며, 대규모 활용을 염두에 두고 개발되었습니다.



### SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning (https://arxiv.org/abs/2502.20127)
- **What's New**: 이번 논문은 Subtask-oriented Reinforced Fine-Tuning (SoRFT)을 제안하여 대형 언어 모델(LLMs)의 문제 해결 능력을 향상시킵니다. 기존의 상업적 모델에 의존하던 문제 해결 프레임워크의 단점을 극복하기 위해 SoRFT는 문제 해결을 파일 로컬라이제이션(file localization), 함수 로컬라이제이션(function localization), 줄 로컬라이제이션(line localization), 코드 수정 생성(code edit generation)과 같은 구조화된 하위 작업으로 나누어 수행합니다. 이를 통해 개방형 소스 개발 리소스를 최대한 활용하는 접근 방식을 제시합니다.

- **Technical Details**: SoRFT는 두 단계의 훈련으로 구성됩니다: (1) 거부 샘플링된 감독 Fine-Tuning(Supervised Fine-Tuning, SFT)과 (2) 규칙 기반 강화 학습(rule-based reinforcement learning)입니다. SFT 단계에서 교사 LLM을 사용해 하위 작업을 위한 Chain of Thought (CoT) 데이터를 생성하고, 그라운드 트루스 기반에서 부정 샘플을 필터링합니다. RL 단계에서는 각 하위 작업에 대한 그라운드 트루스를 활용하여, PPO(Proximal Policy Optimization) 알고리즘을 통해 훈련을 진행합니다.

- **Performance Highlights**: SoRFT로 훈련된 모델은 SWE-Bench Verified와 SWE-Bench Lite에서 최고의 성능을 달성했습니다. 특히, SoRFT-Qwen-7B 모델은 SWE-Bench Verified에서 21.4%의 문제를 해결하는 데 성공하여 오픈 소스 모델 중 최상위 성적을 기록했습니다. 실험 결과는 SoRFT가 문제 해결 성능을 유의미하게 향상시키고, 모델의 일반화 능력을 개선하며, 상업적 모델에 대한 비용 효율적인 대안을 제공함을 보여줍니다.



### Exploring Open-world Continual Learning with Knowns-Unknowns Knowledge Transfer (https://arxiv.org/abs/2502.20124)
- **What's New**: 이번 논문에서는 Open-World Continual Learning (OWCL)의 한계점을 해결하기 위해 새로운 접근 방식을 제안합니다. 기존 OWCL 방법들이 개방 탐지와 지속 학습을 별개의 작업으로 취급하는 문제를 인식하고, 이를 통합한 HoliTrans라는 새로운 프레임워크를 소개합니다. HoliTrans는 비선형 랜덤 프로젝션(nonlinear random projection, NRP)과 분포 인식 프로토타입(distribution-aware prototypes, DAPs)을 활용하여 알아야 할 것과 모르는 것을 동시에 다룹니다.

- **Technical Details**: 논문은 OWCL의 네 가지 시나리오를 정의하고, 이러한 시나리오에 대한 포괄적인 실험을 수행합니다. 특히, HoliTrans는 알려진 샘플과 알려지지 않은 샘플의 지식 전이를 지원하며, 각각의 오픈 샘플에 대한 표현을 동적으로 업데이트합니다. 이를 통해 OWCL의 연구에서 큰 도약을 기대할 수 있는 통합된 틀을 제공합니다.

- **Performance Highlights**: HoliTrans는 다양한 OWCL 시나리오에서 22개의 경쟁 베이스라인을 초월하는 성능을 보여주어, OWCL 이론과 실제 응용 간의 간극을 줄입니다. 이 연구를 통해OWCL의 도전 과제가 명확하게 드러났으며, 향후 오픈 월드 학습 패러다임의 발전에 기여할 것으로 예상됩니다.



### Self-Training Elicits Concise Reasoning in Large Language Models (https://arxiv.org/abs/2502.20122)
Comments:
          23 pages, 10 figures, 18 tables

- **What's New**: 본 논문에서는 체인의 사고(Chain-of-thought, CoT) 추론 메커니즘을 통해 대형 언어 모델들이 복잡한 작업을 해결할 수 있는 능력이 향상되었음을 설명합니다. 그러나 기존의 모델들이 과도한 토큰을 생성하고 있으며, 이는 불필요한 추론 비용을 초래한다고 주장합니다. 이에 대한 해결책으로, 저자들은 자가 생성된 간결한 추론 경로를 활용한 간단한 파인튜닝 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 Zero-shot prompting이 간결한 추론을 효과적으로 유도하는 데 한계가 있음을 보여주며, 대신 Best-of-N (BoN) 샘플링과 Few-shot conditioning을 활용하여 모델을 파인튜닝하는 방법을 제시합니다. 저자들은 GSM8K와 MATH 데이터셋을 활용하여 다양한 모델 패밀리에서 평균 30%의 출력 토큰 수 감소를 달성하였으며, 이는 이전의 파인튜닝 기준 대비 2.4배 향상된 결과입니다. 이 과정을 통해, 모델들은 질문의 복잡성에 따라 출력 길이를 적절히 조정할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 본 논문에서 제안된 FS-BoN 방법은 모델의 경량화된 추론 경로를 효율적으로 이끌어내어, 복잡한 작업에 대한 추론 비용을 줄이는 데 효과적임을 입증했습니다. 성능 분석에 따르면, 훈련된 모델은 문제의 난이도에 따라 변별력 있는 출력을 유지하며 적절한 자세로 응답을 조정합니다. 이러한 결과는 다양한 모델 스케일에서도 일관되게 유지되었으며, 자가 생성된 데이터의 파인튜닝이 LLM의 잠재적인 간결한 추론 능력을 발휘할 수 있도록 하는 데 기여할 수 있음을 시사합니다.



### Forward-Cooperation-Backward (FCB) learning in a Multi-Encoding Uni-Decoding neural network architectur (https://arxiv.org/abs/2502.20113)
- **What's New**: 이 논문에서는 Forward-Cooperation-Backward (FCB) 학습이라는 새로운 학습 기법을 제안하고 있습니다. 이 기법은 인간의 학습 방식을 모방하여, Forward-Forward 방식, 협동(cooperation), 그리고 역전파(backpropagation)를 결합한 것입니다. 그에 따라 새로운 Multi Encoding Uni Decoding (MEUD) 신경망 아키텍처도 설계되었습니다.

- **Technical Details**: FCB 학습은 다단계 아키텍처를 통해 진화하며, MEUD, MEUD-FF, MEUD-Coop, MEUD-FF-Coop 모델을 포함하여 구조적으로 발전하였습니다. 이 신경망들은 협동을 구현하기 위해 특수한 lateral synaptic connection을 사용하며, 여러 인기 있는 데이터셋에서 차원 축소(dimensionality reduction) 성능을 평가 받았습니다.

- **Performance Highlights**: MEUD-FF-Coop 프레임워크는 표준 Autoencoder와 여러 변형 모델들과 비교하여 실험적으로 우수성을 입증하였습니다. 데이터의 원래와 투영된 공간 간의 세부 관계를 보존하는 능력과 차원 축소 후 분류 성능이 다양한 분류 알고리즘을 통해 평가되어 그 품질이 입증되었습니다.



### MITracker: Multi-View Integration for Visual Object Tracking (https://arxiv.org/abs/2502.20111)
- **What's New**: 새롭게 소개하는 DVTrack 데이터셋은 234K개의 프레임으로 구성되어 있으며, 27개의 다양한 객체와 3-4개의 카메라에서 캡처된 경우를 포함합니다. 이 데이터셋은 occlusion(가림막) 및 deformation(변형)과 같은 9가지 도전 과제를 포함하고 있어, MVOT(다중 뷰 객체 추적) 모델의 교육 및 평가를 위한 최초의 종합적인 벤치마크 역할을 합니다.

- **Technical Details**: MITracker는 새로운 다중 뷰 통합 추적 알고리즘으로, 2D 이미지 기능을 3D 기능 볼륨으로 변환하고 이것을 bird's eye view(BEV) 평면으로 압축하여 서로 다른 뷰 간의 정보 융합을 원활하게 합니다. 이 방법은 Vision Transformer(ViT)를 이용해 특정 뷰의 입력 영상에서 목표 객체의 피쳐를 추출하며, 추출된 피쳐를 3D 기능 볼륨으로 통합하여 강화된 주의(attention) 메커니즘으로 추적 결과를 개선합니다.

- **Performance Highlights**: MITracker는 MVTrack과 GMTD 데이터셋에서 기존 방법들을 능가하여 최첨단 성능(state-of-the-art)을 달성합니다. 특히, MITracker는 도전적인 시나리오에서 목표 손실을 줄이기 위해 회복률을 56.7%에서 79.2%로 증가시키는 성과를 보였습니다. 이러한 결과는 MITracker가 occlusion과 같은 어려운 상황에서도 안정적인 추적 결과를 유지할 수 있다는 점에서 큰 의미를 가집니다.



### Sanity Checking Causal Representation Learning on a Simple Real-World System (https://arxiv.org/abs/2502.20099)
Comments:
          24 pages, 12 figures

- **What's New**: 이 연구는 실험적으로 검증된 처방 시스템에서 인과 표현 학습(causal representation learning, CRL) 방법들의 효과성을 평가하였습니다. 연구자들은 다양한 CRL 접근법을 대표하는 방법들을 선택하였으나, 이러한 방법들이 기대했던 바와 달리 근본적인 인과 요소를 회복하지 못하는 결과를 도출하였습니다. 이는 기존 연구와는 다르게 실제 데이터 생성 과정을 다루며, 독립적인 기초 진리를 제시하여 이론적 탐색과 실제 응용 사이의 간극을 해소하려는 시도로 볼 수 있습니다.

- **Technical Details**: 이 실험은 광학적 실험을 기반으로 설계되었으며, 물리적 시스템의 데이터 생성 과정이 CRL의 핵심 가정을 충족하도록 구성되었습니다. 제어 입력은 RGB LED의 밝기와 두 개의 선형 편광자의 위치로 이루어져 있으며, 출력은 카메라로 촬영한 이미지와 물리적 양의 측정 데이터를 포함합니다. 연구자들은 서로 다른 세 가지 접근 방식에서 대표적인 방법을 평가했으며, 각각의 결과는 섹션 3에서 논의됩니다.

- **Performance Highlights**: 실험 결과는 대다수의 CRL 방법이 단순한 합성 자료에도 불구하고 일반적인 CRL 방법이 요구하는 근본적인 인과 요소를 복원하는 데 실패했다는 것을 보였습니다. 연구자들은 이러한 실패 양상을 분석하기 위해 실제 데이터 생성 과정을 간소화한 합성 대안으로 데이터 차원을 축소하였고, 이 과정에서 재현성 문제를 발견하였습니다. 이 연구는 CRL 방법의 이론적 약속과 실제 응용의 도전 과제를 강조하며, 앞으로 CRL 방법의 발전을 가속화할 수 있는 기초 자료를 제공하기를 희망합니다.



### WalnutData: A UAV Remote Sensing Dataset of Green Walnuts and Model Evaluation (https://arxiv.org/abs/2502.20092)
- **What's New**: 이 연구에서는 UAV(무인 항공기) 기술을 통해 수집한 첫 번째 대규모 녹색 호두(New walnut) 객체 탐지 데이터셋인 WalnutData를 소개합니다. 이 데이터셋은 30,240개의 RGB 이미지와 706,208개의 주석 인스턴스를 포함하고 있으며, 특히 조명과 차폐 조건을 세분화하여 네 가지 환경 상태로 나뉘었습니다. 이를 통해 스마트 농업에서의 알고리즘 개발을 지원하고, 녹색 호두 탐지의 과학적 및 공학적 가치를 강조합니다.

- **Technical Details**: WalnutData 데이터셋은 다양한 조명 조건과 차폐 문제가 있는 녹색 호두의 탐지를 용이하게 하기 위해 설계되었습니다. 1,024×1,024 픽셀 해상도의 이미지로 구성되어 있으며, A1(정면 조명), A2(역광), B1(정면 조명 차폐), B2(역광 차폐)의 네 가지 환경 상태로 분류됩니다. 이러한 세분화는 로봇 경로 계획 및 장애물 회피 결정에 필요한 딥러닝 기반 알고리즘의 성능 향상을 위한 기반으로 기능합니다.

- **Performance Highlights**: 공식 연구의 일환으로 WalnutData를 사용하여 DETR, YOLO 시리즈, Fast R-CNN 및 Faster R-CNN과 같은 여러 주류 탐지 알고리즘에 대한 벤치마크 테스트를 수행하였습니다. 이러한 테스트 결과는 향후 알고리즘 디자인의 기준선을 제공합니다. WalnutData는 농업 분야에서의 객체 탐지 연구에 필수적인 데이터 기반 자동화 관리 방법의 발전을 도울 것으로 예상됩니다.



### RIZE: Regularized Imitation Learning via Distributional Reinforcement Learning (https://arxiv.org/abs/2502.20089)
- **What's New**: 새로운 Inverse Reinforcement Learning (IRL) 접근법을 소개합니다. 이 방법은 고정된 보상 할당의 한계를 극복하고 암묵적인 보상 정규화에서의 유연성을 증가시킵니다. 기존의 Maximum Entropy IRL 프레임워크를 제곱 형태의 temporal-difference (TD) 정규자와 훈련 중 동적으로 조정되는 적응형 목표로 확장하여, 보상 기능을 간접적으로 최적화하는 동시에 강화 학습 원리를 포함합니다.

- **Technical Details**: 우리의 방법은 MaxEnt IRL 프레임워크 아래에서 두 가지 방법론적 혁신을 통해 암묵적 보상 정규화를 확장합니다. 첫째, 훈련 중에 동적으로 조정되는 적응형 목표 보상을 도입하여 고정 목표를 대체하고, 둘째, 분포적 강화 학습(distributional RL)을 통합하여 보다 풍부한 반환 정보를 포착하며 Q-값과 이론적 일관성을 유지합니다. 이러한 접근법은 보상 학습의 이전 경직성을 해결하고, MuJoCo 벤치마크에서 온라인 IL 기준을 능가하는 성능을 보여줍니다.

- **Performance Highlights**: 우리의 접근법은 MuJoCo 작업에서 전문가 수준의 결과를 나타냅니다. 특히, Humanoid 작업에서는 단 3개의 데모로 최첨단 성능을 달성했습니다. 광범위한 실험과 ablation 연구를 통해 적응형 목표와 보상 동역학의 효과를 입증하며, 모방 학습에서의 심층적인 통찰을 제공합니다.



### Minds on the Move: Decoding Trajectory Prediction in Autonomous Driving with Cognitive Insights (https://arxiv.org/abs/2502.20084)
- **What's New**: 이 논문에서는 혼합 자율주행 환경에서 주변 차량의 미래 궤적 예측의 중요성을 강조하며, Drivers' decision-making 과정을 반영한 Cognitive-Informed Transformer (CITF) 모델을 제안합니다. 기존의 통계 기반 모델은 인간 운전자의 의도를 정확히 반영하지 못했으나, CITF는 Perceived Safety 개념을 도입하여 모델의 예측 정밀도를 향상시키고 있습니다. 이 새로운 접근 방식은 자율주행 시스템의 예측 능력을 개선할 뿐만 아니라, 실제 주행 환경과 인간의 행동을 보다 잘 이해할 수 있도록 돕습니다.

- **Technical Details**: CITF는 Perceived Safety-aware Module과 Leanformer 모듈을 포함하여 운전자의 행동을 해석하는 데 중점을 둡니다. Quantitative Safety Assessment (QSA)를 통해 시나리오 내 주관적인 위험 수준을 정량적으로 평가하고, Driver Behavior Profiling (DBP)을 통해 다양한 운전자의 행동을 분류합니다. Leanformer는 차량 간의 사회적 상호작용을 포착하는 경량 트랜스포머 기반 프레임워크로, 복잡한 교통 상황에서의 차량 간 상호작용을 이해하는 데 기여합니다.

- **Performance Highlights**: CITF는 NGSIM, MoCAD, HighD 데이터셋에서 각각 12.0%, 28.2%, 20.8%의 성능 향상을 달성하며, 기존의 SOTA 모델보다 우수한 성능을 보였습니다. 특히, 작은 데이터셋으로도 훈련 가능한 능력을 보여주며, 제한적 또는 결측 데이터가 있는 상황에서도 뛰어난 견고성을 유지합니다. 이 모델은 자율주행 시스템의 실제 적용 가능성을 높이는 데 중요한 기여를 할 것으로 기대됩니다.



### Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents (https://arxiv.org/abs/2502.20073)
Comments:
          25 pages, 14 figures

- **What's New**: 본 논문에서는 LLM(대형 언어 모델) 기반의 새로운 멀티 에이전트 시스템(LLM-MAS) 벤치마크인 Collab-Overcooked를 제안합니다. 이 벤치마크는 Overcooked-AI 게임을 기반으로 하며, 상호작용 환경에서 도전적이고 적용 가능한 여러 가지 작업을 포함하고 있습니다. 기존 벤치마크는 다루지 않았던 협력 능력 평가를 위한 다양한 프로세스 지향적 지표들을 도입하여 LLM 에이전트의 미세 협력 능력을 평가할 수 있도록 합니다.

- **Technical Details**: LLM-MAS는 복잡한 작업을 해석하고 계획하는 데 있어 LLM의 제로샷(Zero-Shot) 및 피샷(Few-Shot) 학습 능력을 활용하고 있습니다. LLM-MAS는 목표 해석을 넘어선 세 가지 필수 협력 능력을 요구하는데, 여기에는 능력 경계 인식(competence boundary awareness), 의사소통(communication), 동적 적응(dynamic adaptation)이 포함됩니다. 이러한 협력 평가 프레임워크를 구축하는 것이 LLM-MAS의 효과성을 평가하는 데 중요합니다.

- **Performance Highlights**: 연구에서는 다양한 크기의 10개 LLM을 대상으로 광범위한 실험을 진행하였으며, 목표 해석에서 강력한 능력을 보였지만, 복잡한 작업을 효율적으로 수행하기 위한 활성 협력 및 지속적인 적응에서 상당한 차이가 존재함을 보여주었습니다. 또한 Collab-Overcooked는 여러 복잡성 수준에서 30개의 연속적, 프로세스 특정 작업을 포함하여 LLM 간 협력 평가의 중요한 한계를 인식하고 이를 극복하기 위한 통찰을 제공합니다.



### Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation (https://arxiv.org/abs/2502.20056)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이 논문에서는 방사선과 의사들의 업무 부담을 줄이기 위해 개선된 대조 학습(contrastive learning) 방법론을 기반으로 한 다중 관점(longitudinal) 데이터를 활용하여 흉부 X-ray 보고서 생성을 위한 모델, MLRG(Multi-view Longitudinal Report Generation)를 제안합니다. 기존의 방법론들이 단일 이미지에 의존하는 반면, 이 방법은 현재 다중 관점의 이미지를 통해 질병의 진행 상태를 분석하고 이를 바탕으로 보다 정확한 진단 정보를 제공합니다. 또한, 모델이 환자 특정의 이전 지식을 유연하게 처리할 수 있도록 하는 결측값 인코딩(Absence Encoding) 기법을 도입하였습니다.

- **Technical Details**: MLRG는 두 가지 단계로 구성되며, 첫 번째 단계에서는 방사선 보고서 내의 고유한 시공간 정보(spatiotemporal information)를 활용하여 시각 및 텍스트 표현의 사전 학습(pre-training)을 감독합니다. 다중 관점(longitudinal) 데이터와 그에 해당하는 방사선 보고서 간의 일치를 활용하여 시각적 및 텍스트 표현을 학습합니다. 두 번째 단계에서는 결측한 환자 정보(예: 이전 검사 결과) 처리 기술을 도입하여 모델이 유연하게 데이터의 유무에 따라 조정될 수 있도록 하여 생성된 보고서의 정확성을 향상시킵니다.

- **Performance Highlights**: MLRG는 MIMIC-CXR, MIMIC-ABN, 그리고 Two-view CXR 데이터셋을 대상으로 한 광범위한 실험에서 기존의 최신 방법들과 비교하여 우수한 성능을 입증하였습니다. 특히 MIMIC-CXR에서 2.3%의 BLEU-4 향상을, MIMIC-ABN에서 5.5%의 F1 스코어 향상을, Two-view CXR에서는 2.7%의 F1 RadGraph 개선을 달성했습니다. 이런 결과는 MLRG가 임상적으로 정확한 보고서를 생성하는 데 있어 효과적인 방법임을 시사합니다.



### Polish-ASTE: Aspect-Sentiment Triplet Extraction Datasets for Polish (https://arxiv.org/abs/2502.20046)
- **What's New**: 이번 논문에서는 Aspect-Sentiment Triplet Extraction (ASTE)이라는 감정 분석의 복잡한 작업을 다루며,  폴란드어 고객 리뷰에 기반한 두 개의 새로운 데이터셋을 소개합니다. 이 데이터셋은 호텔 및 구매 제품에 대한 의견을 포함하며, 폴란드어 ASTE 연구를 위한 기반을 제공합니다. 또한, 기존의 ASTE 기술과 최첨단 대형 언어 모델을 결합하여 실험을 수행하고 기술의 성능을 평가하였습니다.

- **Technical Details**: 본 연구에서 제안된 데이터셋은 Wroclaw Corpus of Consumer Reviews Sentiment (WCCRS)에서 추출한 고객 리뷰로 구성되어 있습니다. 리뷰는 문장 및 문서 수준에서 감정 극성을 제공하지만, 기초적인 ASTE 트리플 구성을 위해서는 후기 내용의 감정 극성을 사용하지 않았습니다. 각 문장은 aspect phrase, opinion phrase, sentiment polarity를 포함하는 방식으로 주석이 달렸으며, 주석 작업은 네이티브 폴란드어 화자가 진행하였습니다.

- **Performance Highlights**: 실험 결과, 두 개의 새로운 데이터셋은 기존 ASTE 기술과의 결합에서 높은 성능을 보였으며, 데이터셋의 난이도를 평가하는 데에도 유용하였습니다. 주석 품질은 전문가에 의해 검토되었고, 두 주석 간 일치를 측정한 결과 높은 일치도를 보였습니다. 새로운 데이터셋은 연구자의 접근 가능성을 고려하여 자유로운 라이센스하에 제공됩니다.



### Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting (https://arxiv.org/abs/2502.20045)
Comments:
          11 pages, 11 figures

- **What's New**: 이번 논문은 텍스트 입력을 기반으로 VDM(벡터 변위 맵) 브러시 생성을 위한 텍스트-투-VDM(Text2VDM)이라는 새로운 프레임워크를 제안합니다. 기존의 3D 모델 생성 기술이 아니라, 브러시 생성을 위한 3차원적 접근 방식을 통해 사용자 알림을 기반으로 메쉬 변형을 안내하는 방식으로 발전하였습니다. 세밀한 구조 표현을 위한 방법으로 CFG-가중 블렌딩을 도입하여 생성된 브러시의 품질과 다양성을 높였습니다.

- **Technical Details**: Text2VDM은 SDS(점수 증류 샘플링)와 Laplace-Beltrami 연산자를 적용하여 기본 메쉬를 변형하는 최적화 기반 프레임워크입니다. 사용자는 요청된 구조에 대한 VDM을 생성하기 위해 세 가지 방법으로 기본 메쉬를 초기화할 수 있습니다: 제로 값 VDM, 스파이크 패턴 VDM 또는 사용자 지정 VDM. 이 과정에서 CFG 가중 블렌딩이 사용되어 의미적 결합 문제를 효과적으로 해결하여 보다 정확한 대상 분포를 달성합니다.

- **Performance Highlights**: 실험 결과, Text2VDM은 다양한 고품질 VDM 브러시를 생성하여 Blender와 ZBrush 같은 주요 모델링 소프트웨어에 통합될 수 있음을 보여주었습니다. 이 접근법은 아티스트가 다양한 브러시를 사용하여 표면 세부사항을 조각할 수 있는 직관적인 방식을 제공합니다. 특히, 기존의 3D 모델 생성을 위한 방법과 구별되는 점이 있으며, 브러시 기반의 사용자 조각을 통해 표현력 있는 모델로의 변환을 가능하게 합니다.



### CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR (https://arxiv.org/abs/2502.20040)
Comments:
          Submission to IEEE/ACM Trans. on TASLP

- **What's New**: 이 연구에서는 음성 품질과 자동 음성 인식(ASR) 성능을 향상시키기 위해 CleanMel이라는 단일 채널 Mel-spectrogram(멜 스펙트로그램) 노이즈 제거 및 잔향 제거 네트워크를 제안합니다. 이 네트워크는 소음과 잔향이 섞인 마이크로폰 녹음을 입력받아 이를 깨끗한 Mel-spectrogram으로 예측합니다. 개선된 Mel-spectrogram은 신경 인코더(neural vocoder)를 통해 음성 파형으로 변환되거나 ASR에 직접 사용할 수 있습니다.

- **Technical Details**: 제안된 네트워크는 Mel 주파수 영역에서 상호 대역(cross-band) 및 협대역(narrow-band) 처리를 교차 배치하여 전체 스펙트럼 패턴과 신호의 협대역 특성을 학습합니다. Mel 주파수 영역에서의 향상된 처리는 음성을 더 간결하게 표현하며, 기계 학습 관점에서 낮은 차원(feature dimension)을 가지고 있어 예측 오류를 줄이는 데 유리합니다. 이 작업에서는 CleanMel이 기존의 ERB 또는 Mel 영역에서의 음성 향상 및 후처리 부분을 분리하여, 깨끗한 Mel-spectrogram에 초점을 맞추어 작동합니다.

- **Performance Highlights**: 공식적인 실험은 네 개의 영어 및 하나의 중국어 데이터셋에서 두 가지 목표(노이즈 제거 및 잔향 제거)에 대해 수행되었습니다. 제안된 모델은 음성 지각 품질 관점에서 최신 기술(state-of-the-art) 성능에 도달했으며, 다양한 사전 훈련된 ASR 모델 위에서도 모든 데이터셋에서 ASR 성능을 크게 개선했습니다. 이러한 결과는 우리의 모델이 실질적인 애플리케이션에 직접 적용될 가능성이 있음을 시사합니다.



### Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping (https://arxiv.org/abs/2502.20032)
Comments:
          Accepted by the proceeding of CVPR2025

- **What's New**: 본 논문은 Class Incremental Learning (CIL) 분야에서 클래스의 순서가 모델 성능에 미치는 영향을 분석합니다. 저자들은 클래스 유사성이 낮을수록 모델이 클래스 순서에 더 강건함을 보인다는 이론적 분석을 제공합니다. 이와 함께, Graph-Driven Dynamic Similarity Grouping (GDDSG)이라는 새로운 방법을 제안하여 클래스 간 유사성을 기반으로 다이나믹하게 클래스를 그룹화합니다.

- **Technical Details**: GDDSG 방법은 그래프 색칠 알고리즘을 활용하여 유사성이 낮은 클래스 그룹을 형성하고, 각 클래스 그룹에 대해 독립적인 CIL 모델을 훈련합니다. 이 방식은 각 그룹의 모델을 결합하여 예측할 때 효과적으로 작동하며, 클래스 간 유사성이 높을수록 지속적인 학습 과정에서 발생하는 클래스 충돌을 완화합니다. 이를 통해 모델의 일반화 능력과 잊지 않음 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, GDDSG 방법은 클래스 순서에 대한 민감성을 효과적으로 해결하며 모델의 정확성과 방어적 목표를 달성했습니다. 기존의 CIL 방법들이 직면했던 문제들을 해결하고, 다양한 클래스 순서에서 안정된 성능을 유지할 수 있는 혁신적인 접근법으로 주목받고 있습니다. 저자들은 이 연구의 코드도 함께 공개하여 향후 연구자들이 활용할 수 있도록 지원하고 있습니다.



### Can Large Language Models Unveil the Mysteries? An Exploration of Their Ability to Unlock Information in Complex Scenarios (https://arxiv.org/abs/2502.19973)
Comments:
          11pages

- **What's New**: 본 논문에서는 복잡한 시나리오에서 여러 인지 입력을 통합하고 조합적 추론을 수행하는 능력을 탐색하기 위해 새로운 벤치마크인 CVQA(Clue-Visual Question Answering)와 CPVQA(Clue of Password-Visual Question Answering)를 소개합니다. CVQA는 시각적 이해 및 통합을 평가하기 위해 3가지 작업 유형을 포함하고, CPVQA는 시각적 데이터의 정확한 해석 및 적용을 중점적으로 다룹니다.

- **Technical Details**: CVQA는 서로 다른 장면에 위치한 불특정 시각적 개체를 결합하여 추론하는 Props Search, 특정 텍스트 개체와 불특정 시각적 개체 간의 상호작용을 포함하는 Props Usage, 그리고 지정된 시퀀스 유형으로 시각적 장면을 결합하여 단서를 추론하는 세 가지 유형의 작업을 포함합니다. CPVQA는 모든 시각 장면을 아우르는 조합적 추론을 평가하는 두 가지 유형의 작업을 포함하고 있습니다.

- **Performance Highlights**: 기존의 모델들은 조합적 추론 벤치마크에서 낮은 성능을 기록했습니다. 특히 최신 모델조차 CVQA에서 33.04%의 정확도에 불과하며, CPVQA에서는 7.38%로 감소합니다. 본 연구의 방법론은 이러한 성능을 개선하여 CVQA와 CPVQA에서 각각 22.17% 및 9.40%의 성능 향상을 보였습니다.



### Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction (https://arxiv.org/abs/2502.19971)
- **What's New**: 이번 논문에서는 quantum low-density parity-check (QLDPC) 코드와 같은 새로운 코드에 대한 효율적인 디코더의 부재를 해결하기 위해 그래프 신경망(graph neural network)을 기반으로 하는 범용 디코더를 소개합니다. 이 디코더는 안정기(stabilizer) 코드의 그래프 구조에서 직접 작동하며, 다양한 안정기 코드에서 정확도와 속도 모두에서 향상된 결과를 보여주었습니다. 특히, Bivariate Bicycle 코드의 경우, 이전 디코더보다 39.4% 낮은 논리 오류율을 달성하면서 디코딩 시간은 약 1%에 불과합니다.

- **Technical Details**: 양자 계산 시스템은 환경 잡음에 매우 민감하므로 quantum error correction (QEC)가 필수적입니다. QEC는 복잡한 양자 정보를 여러 물리적인 큐비트에 인코딩하여 보호하나, 기존의 디코딩 문제는 NP-complete하게 입증되었습니다. 본 연구는 그래프 신경망을 통해 QEC를 위한 새로운 범용 디코더를 제안하며, 이는 타너 그래프를 사용해 증후(syndrome) 측정을 코드에 따라 구체적으로 표현하며, 서로 다른 코드 계열 간의 효율적인 디코딩을 가능하게 합니다.

- **Performance Highlights**: 본 연구의 디코더는 다양한 안정기 코드 집합에 대해 일관되게 낮은 논리 오류율을 기록했습니다. 예를 들어, 색상 코드 및 Bivariate Bicycle 코드의 경우, 디코딩 과정에서 상대적으로 높은 정확도를 달성하며, 디코딩 속도 또한 선형적으로 유지됩니다. 또한, 표면 코드 실험에서는 Alpha-Qubit 신경 디코더의 성능을 초과하며, 훨씬 적은 모델 파라미터를 사용했습니다.



### Deterministic or probabilistic? The psychology of LLMs as random number generators (https://arxiv.org/abs/2502.19965)
Comments:
          31 pages, 12 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)을 사용하여 무작위 숫자 생성 시 성능을 체계적으로 조사하였습니다. 다양한 모델 아키텍처, 숫자 범위, 온도(temperature), 프롬프트 언어(prompt language) 등 여러 구성요소를 고려하였습니다. 흥미롭게도, 이들 모델이 확률적(transformers-based) 구조를 가지고 있음에도 불구하고 무작위 숫자 요청 시 결정론적(deterministic) 응답을 보이는 경우가 많다는 점을 발견하였습니다.

- **Technical Details**: 연구 결과, 모델 변경 및 프롬프트 언어의 변화에 따라 유의미한 차이가 나타났으며, 이는 훈련 데이터에 깊이 박힌 편향(bias) 때문으로 분석되었습니다. 특히 DeepSeek-R1와 같은 모델은 LLM의 내부 추론 프로세스를 부분적으로 밝힐 수 있었고, 비슷한 결과에 도달하였음에도 불구하고 각기 다른 방식으로 반응하였습니다. LLM이 진정한 무작위성(randomness)을 생성하지 못하는 이유는 인간의 인지적 편향을 재생산하기 때문임을 강조합니다.

- **Performance Highlights**: 편향성으로 인해 LLM은 예측 가능한 패턴을 생성하며, 이는 무작위성을 방해하는 요소로 작용합니다. 다양한 실험을 통해 LLM의 성능이 특정 모델과 구성에 따라 어떻게 달라지는지를 실질적으로 확인하였습니다. 이러한 결과는 LLM의 작동 방식과 그 한계를 이해하는 데 중요한 통찰을 제공합니다.



### Collaborative Stance Detection via Small-Large Language Model Consistency Verification (https://arxiv.org/abs/2502.19954)
- **What's New**: 이번 연구에서는 CoVer라는 새로운 프레임워크를 제안하여 소셜 미디어에서의 스탠스 감지를 위한 Large Language Model (LLM)와 Small Language Model (SLM)의 협업을 통해 성능을 개선하고 있습니다. CoVer는 텍스트를 개별적으로 처리하는 대신, 배치단위로 처리하여 LLM의 추론을 활용하면서도 SLM을 통해 논리적 일관성을 검증합니다. 이러한 접근 방식은 본래 LLM에 대한 의존도를 줄이고, 데이터 분석의 효율성을 높입니다.

- **Technical Details**: CoVer는 지식 증대(knowledge augmentation)와 무관한 맥락 필터링(irrelevant context filtering)을 통해 트윗의 맥락을 재구성하여 명확하고 편향 없는 스탠스 추론을 보장합니다. LLM은 동시에 여러 텍스트를 처리하여 맥락 재사용의 효율성을 제공합니다. 마지막으로, LLM의 추론 논리적 일관성을 검증하기 위해 SLM을 활용하며, 반복적으로 낮은 일관성을 보이는 텍스트에 대해 일관성 가중 집계(consistency-weighted aggregation)를 통해 최종 분류를 수행합니다.

- **Performance Highlights**: CoVer는 SemEval-2016, VAST 및 P-Stance와 같은 여러 벤치마크에서 최첨단 방법들보다 우수한 성과를 기록하며, 0.54 LLM 쿼리로 트윗 하나당 성능을 크게 향상시켰습니다. 이는 CoVer가 자원 효율적이면서도 높은 성능을 발휘할 수 있는 가능성을 보여줍니다. 이 연구는 스탠스 감지 과제에서 LLM과 SLM의 협업이 어떻게 효과적으로 결합될 수 있는지를 잘 나타내고 있습니다.



### Dynamic DropConnect: Enhancing Neural Network Robustness through Adaptive Edge Dropping Strategies (https://arxiv.org/abs/2502.19948)
- **What's New**: 이 논문은 각 신경망 층의 엣지에 동적 드롭 비율을 할당하여 드롭 과정을 독특하게 조정하는 새로운 방법론인 DynamicDropConnect (DDC)를 소개합니다. DDC는 추가적인 학습 매개변수를 포함하지 않고도 드롭 비율을 조정할 수 있습니다. 실험 결과는 DDC가 기존의 Dropout, DropConnect 및 Standout보다 우수한 성능을 보여주는 것을 입증합니다.

- **Technical Details**: DDC는 엣지와 관련된 그래디언트의 크기에 따라 드롭 확률을 동적으로 할당합니다. 그래디언트가 큰 엣지는 학습에 중요하므로 유지되어야 하고, 미미한 영향만 미치는 엣지는 제거될 수 있습니다. 이 접근법은 모델 아키텍처를 단순화하고 메모리 요구사항을 줄이는 장점을 제공합니다.

- **Performance Highlights**: DDC는 합성 및 실제 공개 데이터셋을 사용한 실험을 통해 유의미한 결과를 나타냅니다. 특히, 작은 그래디언트의 엣지를 우선적으로 드롭하는 방법이 다른 방법들보다 빠르게 최소 손실에 도달하는 모습을 보여줍니다. 이는 DDC가 더 복잡한 네트워크에 적용될 때 실험 성능이 기대된다는 가능성을 시사합니다.



### Algebraic Machine Learning: Learning as computing an algebraic decomposition of a task (https://arxiv.org/abs/2502.19944)
- **What's New**: 이번 논문에서는 기계 학습의 기초를 대신하여 추상 대수(abstract algebra)를 기반으로 한 대안적인 방법론인 대수적 기계 학습(Algebraic Machine Learning, AML)을 제안합니다. AML은 문제와 데이터를 대수적 공리의 집합으로 인코딩함으로써 학습을 분석하고 이해할 수 있는 새로운 기회를 제공합니다. 이 접근법은 통계(statistics) 및 최적화(optimization)에 의존하지 않고도 직접적으로 훈련 데이터에서 일반화할 수 있는 능력을 가지며, 이는 기존의 기계 학습 방법과는 다른 점입니다.

- **Technical Details**: 대수적 기계 학습(AML)은 데이터, 목표 및 선행 지식에 의해 정의된 문제를 대수적 공리의 집합으로 인코딩합니다. 이후, 'Full Crossing'이라 불리는 절차를 통해 모델을 생성하며, 이 모델은 공리와 그 논리적 결과만이 진실인 '가장 자유로운 모델'입니다. 이 논문에서는 Sparse Crossing을 사용하여 일반화 subsets를 공리로부터 직접적으로 얻는 방법론을 제시하며, 추상 대수에서 알려진 부분 직접 분해(subdirect decomposition)를 통해 문제의 근본적 구성 요소(atom)를 찾는 방법도 설명합니다.

- **Performance Highlights**: 이런 새로운 학습 원리는 MNIST, FashionMNIST, CIFAR-10 및 의료 이미지 같은 표준 데이터셋에서 검증되었으며, 최적화된 다층 퍼셉트론(multilayer perceptrons)과 유사한 성능을 달성했습니다. 또한 이 방법은 단순한 데이터 기반 작업을 넘어 해밀토니안 사이클(hamiltonian cycle) 탐색과 같은 형식적 문제 해결에도 확장될 수 있습니다. 대수적 기계 학습은 훈련 데이터에서 직접 학습할 수 있는 새로운 관점을 제공하며, 검증 데이터셋 없이도 데이터의 근본 규칙으로 점근적 수렴(asymptotic convergence)할 수 있는 장점을 지니고 있습니다.



### Flexible Bivariate Beta Mixture Model: A Probabilistic Approach for Clustering Complex Data Structures (https://arxiv.org/abs/2502.19938)
- **What's New**: 이번 연구에서는 전통적인 클러스터링 알고리즘의 한계를 극복하기 위한 유연한 이변량 베타 혼합 모델(Flexible Bivariate Beta Mixture Model, FBBMM)을 제안합니다. 기존의 k-means 및 Gaussian Mixture Models (GMM)와 같은 알고리즘은 비볼록 클러스터를 처리하는 데 한계를 보였으며, FBBMM은 이변량 베타 분포의 유연성을 활용하여 다양한 형태의 클러스터를 모델링할 수 있습니다. 이를 통해, 복잡한 데이터 구조를 효과적으로 클러스터링할 수 있는 강력한 솔루션을 제공합니다.

- **Technical Details**: FBBMM은 Expectation Maximization (EM) 알고리즘과 Sequential Least Squares Programming (SLSQP) 최적 기법을 활용하여 매개변수를 추정합니다. 이 모델은 비볼록 클러스터와 부정적 상관관계까지 지원하여, 데이터 집합의 복잡한 구조를 보다 정확하게 포착 가능합니다. 유연한 이변량 베타 분포는 전통적인 클러스터링 방법의 한계를 극복하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과에 따르면, FBBMM은 비볼록 클러스터를 처리하는 데 있어 전통적인 모델보다 뛰어난 성능을 보였습니다. 이 모델은 데이터 포인트가 서로 다른 클러스터에 속할 확률을 부여하며, 이는 데이터 포인트의 소프트 클러스터링을 가능하게 합니다. FBBMM은 원본 데이터와 유사한 새로운 데이터 포인트를 생성할 수 있는 생성 모델로, 데이터 증강 및 시뮬레이션 작업에도 유용하게 활용됩니다.



### Lotus at SemEval-2025 Task 11: RoBERTa with Llama-3 Generated Explanations for Multi-Label Emotion Classification (https://arxiv.org/abs/2502.19935)
Comments:
          8 pages , submitted to SemEval 2025-Task 11

- **What's New**: 이번 논문에서는 Llama-3를 활용하여 다중 레이블 감정 탐지에 대한 새로운 접근 방식을 제안합니다. 이 방법은 모호한 감정 표현을 명확히 하는 설명적인 내용을 생성하여 RoBERTa의 감정 분류 성능을 향상시킵니다. 특히 두려움(fear), 기쁨(joy), 슬픔(sadness) 같은 감정에 대해 F1-score를 개선하며, 텍스트 전용 모델을 능가하는 성과를 보였습니다. 이 연구는 감정 탐지 과제가 직면한 여러 문제를 해결하는 중요한 진전을 나타냅니다.

- **Technical Details**: 이 연구는 Llama-3와 RoBERTa 모델을 조합하여 감정 분류 모델을 향상시키는 데 중점을 두고 있습니다. 최초 단계에서 Llama-3는 텍스트의 모호한 감정 표현에 대한 설명을 생성하며, 이를 통해 RoBERTa를 미세 조정(fine-tuning)하여 더 나은 다중 레이블 감정 분류를 가능하게 합니다. 비슷한 연구들에서 감정 표현을 다룰 때 자주 발생하는 도전 과제인 감정의 모호성, 다중 레이블 분류 및 불균형 데이터셋을 효과적으로 다루기 위한 접근 방식도 포함되어 있습니다.

- **Performance Highlights**: 본 연구는 SemEval 2025 Task 11의 서브태스크 1에서 다중 레이블 감정 탐지의 성능을 평가하였으며, Text + Explanation 모델이 Macro F1 점수 0.7396을 기록했습니다. 반면, 텍스트 전용 모델은 0.7112의 점수를 기록하여, 설명적 내용을 포함하는 것이 분류 정확도를 확실히 향상시킨다는 것을 보여주었습니다. 이러한 결과는 감정 탐지 작업에 대한 설명적 컨텍스트의 중요성을 강조하며, 다양한 감정 클래스 간의 정확도를 개선함을 입증합니다.



### DiffCSS: Diverse and Expressive Conversational Speech Synthesis with Diffusion Models (https://arxiv.org/abs/2502.19924)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이번 논문에서는 DiffCSS라는 혁신적인 대화형 음성 합성(CSS) 프레임워크를 제안하여 여러 가지 응답의 다양성을 생성할 수 있는 방법을 모색하고 있습니다. 기존 CSS 시스템의 한계를 극복하기 위해, DiffCSS는 확산 모델(diffusion models)과 언어 모델 기반 TTS 백본(LM-based TTS backbone)을 통합하여 맥락에 맞는 더 다양하고 표현력 있는 음성을 생성합니다. 이를 통해 대화적 맥락에 따라 풍부한 감정을 전달할 수 있는 음성을 제공하는 것을 목표로 하고 있습니다.

- **Technical Details**: DiffCSS는 두 가지 주요 구성 요소로 이루어져 있습니다: ParlerTTS 기반의 TTS 백본과 확산 기반의 맥락 인식 프로소디 예측기입니다. TTS 백본은 다양한 프로소디 입력에 따라 고품질의 음성을 합성하며, 프로소디 예측기는 대화 맥락에 따라 다양한 프로소디 임베딩을 생성합니다. 이 과정에서, 미리 훈련된 FACodec 모델을 사용해 프레임 수준의 프로소디 특징을 추출하고, 크로스 어텐션 레이어를 결합하여 고정 길이의 효율적인 프로소디 임베딩을 도출합니다.

- **Performance Highlights**: 실험 결과, DiffCSS에서 합성된 음성은 기존 CSS 시스템보다 더 다양하고 맥락에 일치하며 표현력이 뛰어난 것으로 나타났습니다. 특히, DiffCSS는 생성한 프로소디 분포가 실제 데이터와 더 잘 일치함을 보여주어 확산 모델을 CSS에 통합한 효과를 강조합니다. 이 연구는 대화 맥락 모델링을 위한 새로운 가능성을 열어주는 중요한 기여를 하고 있습니다.



### Incremental Learning with Repetition via Pseudo-Feature Projection (https://arxiv.org/abs/2502.19922)
- **What's New**: 이 논문은 점진적 학습(incremental learning, IL)에서의 데이터 반복효과를 다루고 있습니다. 특히 반복 패턴이 내재된 새로운 시나리오를 제안하여 기존의 엄격한 반복 금지 규칙을 완화하고자 합니다. 이를 통해 기존 전략과 비교하여 더 현실적인 학습 환경을 구현하기 위한 방법론을 탐구하고 있습니다.

- **Technical Details**: 제안된 방법인 Horde는 독립적인 특징 추출기(feature extractor)의 앙상블을 동적으로 조정하며, 클래스 반복을 활용하여 이들을 정렬할 수 있습니다. 이 메소드는 기존의 예시 기반 접근 방식(exemplar-based approaches)과는 달리, 예시 없이 학습할 수 있는 능력을 지니고 있습니다. 특히, 본 연구에서는 기존의 IL 방법들을 벤치마킹하고 내재적인 데이터 반복의 영향력을 분석하고 있습니다.

- **Performance Highlights**: Horde 방법은 반복이 없는 전통적인 시나리오에서 경쟁력 있는 결과를 달성하며, 반복이 발생하는 조건에서도 최첨단 성능을 나타냅니다. 이는 실제 애플리케이션에서 발생할 수 있는 반복 데이터 문제를 해결하기 위한 중요한 기초를 마련합니다. 또한, IL이 반복을 포함할 때의 안정성과 적응성을 더욱 잘 이해할 수 있는 기반을 제공합니다.



### Order Doesn't Matter, But Reasoning Does: Training LLMs with Order-Centric Augmentation (https://arxiv.org/abs/2502.19907)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 논리적 추론 능력을 향상시키기 위해 순서 중심(data augmentation) 데이터 증대 프레임워크를 도입합니다. 기존 LLM들은 논리적 동등성 변환에 대한 일반화에 어려움을 겪었으나, 본 연구는 논리적 커뮤테이티비티(commutativity)를 기반으로 한 새로운 방법을 제안하여 이를 해결합니다. 이 방법은 독립적인 전제를 무작위로 섞고, 단계 간의 의존성을 모델링하기 위해 방향 비순환 그래프(DAG)를 사용하는 등, LLM의 추론 과정을 유연하게 만들어 줍니다.

- **Technical Details**: 제안된 방법에서는 우선 독립적인 전제를 랜덤하게 섞어 조건 순서의 데이터를 증가시키고, 다음으로 추론 단계 간의 관계를 정의하기 위해 DAG를 구성합니다. DAG를 이용하여 의존관계를 유지하면서 유효한 재배열을 식별하고, 이를 통해 모델이 논리적 동등성을 배울 수 있도록 하는 과정이 포함됩니다. 데이터 증대는 또한 복잡한 테스트 시나리오에서 모델의 전반적인 추론 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 다양한 논리적 추론 벤치마크를 통해 실시한 실험 결과, 제안된 방법이 LLM의 추론 성능을 크게 향상시켰음을 보여줍니다. 모델은 고정된 논리 구조의 데이터 세트로 훈련하는 것보다 높은 성능을 보였으며, 어수선한 테스트 시나리오에서도 우수한 적응력을 보였습니다. 이러한 결과는 제안된 접근 방식이 LLM의 논리적 이해를 심화시키는데 기여함을 시사합니다.



### Shared Autonomy for Proximal Teaching (https://arxiv.org/abs/2502.19899)
Comments:
          Accepted to ACM/IEEE International Conference on Human-Robot Interaction, 2025

- **What's New**: 이번 연구는 Z-COACH라는 새로운 소프트웨어 프레임워크를 통해 공유 자율성(shared autonomy)을 활용하여 개인화된 교육을 제공하는 방법을 제안합니다. 이는 학생의 운동 능력에 맞춰 적절한 교육 커리큘럼을 설계하는 데 중점을 두고 있으며, 교육 심리학의 비유적 발판(scaffolding) 개념에서 영감을 받았습니다. Z-COACH는 학생의 학습 가능 범위인 근접 발달 영역(Zone of Proximal Development, ZPD)을 정량화하여 맞춤형 학습을 지원합니다.

- **Technical Details**: Z-COACH는 학생의 기술 향상을 도와주는 공유 자율성을 활용하여, 학생이 개선 가능성이 높은 하위 기술을 식별하는 메커니즘을 포함합니다. 연구에서는 CARLA 자율주행 시뮬레이터를 사용하여 고성능 경주를 시뮬레이션하고, 50명의 사용자를 대상으로 한 실험을 통해 학생의 주행 시간과 행동, 매끄러움을 개선하는 데 효과적임을 입증했습니다. 이를 통해 Z-COACH는 학생의 ZPD를 기반으로 상황에 맞춘 지원을 제공할 수 있는 방법을 탐구합니다.

- **Performance Highlights**: 사용자 연구 결과, Z-COACH가 제공하는 개인화된 교육은 전통적인 자기 연습 기법보다 주행 시간을 크게 단축시키고, 주행 행동의 질을 향상시켰습니다. 이는 학생들이 특정 기술을 먼저 연습하도록 유도하여, 각 개인의 기술 향상에 긍정적인 영향을 미쳤습니다. 본 연구는 자율주행차 및 로봇에 더욱 널리 퍼질 수 있는 반자동 기능이 인간 사용자를 보조하는 것뿐만 아니라, 교육의 측면에서도 효과적임을 보여줍니다.



### ColorDynamic: Generalizable, Scalable, Real-time, End-to-end Local Planner for Unstructured and Dynamic Environments (https://arxiv.org/abs/2502.19892)
Comments:
          18 pages

- **What's New**: 이 연구에서는 로봇 내비게이션을 위한 ColorDynamic 프레임워크를 제안합니다. 이를 통해 Raw sensor data(원시 센서 데이터)에서 직접 control commands(제어 명령)으로 매핑하여 비구조적 환경에서의 호환성을 보장합니다. 또한 Transqer라는 새로운 네트워크를 도입하여 동적 시나리오에서 의사 결정을 크게 향상시킵니다.

- **Technical Details**: ColorDynamic은 End-to-end DRL formulation(엔드 투 엔드 DRL 형식)을 기반으로 하며, 동시적이며 효율적인 데이터 접근을 위해 E-Sparrow라는 시뮬레이션 플랫폼과 대칭 불변성을 활용한 데이터 증강 기법이 개발되었습니다. 이는 DRL의 trial-and-error learning paradigm(시도-오류 학습 패러다임)에서 중요한 안전성과 경제적 리스크를 줄일 수 있도록 합니다. Temporal-Spatial Modeling(시각적 시간 모델링) 문제를 해결하기 위해 Transformer 기반의 접근법이 활용됩니다.

- **Performance Highlights**: 성능 평가 결과, ColorDynamic의 성공률이 90%를 초과하고, 실시간 처리 능력도 우수한 것으로 나타났습니다. 연구팀은 OkayPlan-ColorDynamic(옵기플랜-컬러다이나믹) 내비게이션 시스템을 선보였으며, 복잡한 시나리오에서의 사례 연구를 통해 그 우수성과 응용 가능성을 입증했습니다. 실험 자료 및 코드베이스는 공개되어 재현성과 추가 연구를 촉진할 수 있도록 지원합니다.



### Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models (https://arxiv.org/abs/2502.19883)
Comments:
          12 pages. 6 figures

- **What's New**: 이번 연구에서는 작은 언어 모델(SLMs)의 보안 성능을 평가하기 위한 포괄적인 실험을 수행했습니다. SLMs는 효율적이며 낮은 컴퓨팅 비용으로 주목받고 있지만, 보안 위험이 대형 언어 모델(LLMs) 대비 상대적으로 덜 주목받아왔습니다. 실험 결과, 대부분의 SLM들이 기존의 jailbreak 공격에 상당히 취약하다는 점이 드러났습니다.

- **Technical Details**: 본 연구에서는 총 16개의 최첨단 모델을 분석했으며, 그 중 13개 SLM은 4억 이하의 파라미터를 가지고 있고, 3개의 LLM은 70억 이상의 파라미터를 가집니다. 실험을 통해 얻은 데이터는 다양한 jailbreak 공격 방법에 대한 SLM의 보안 취약점을 노출하고, 해당 공격에 대한 방어 방법의 효과를 평가했습니다. 특히, SLM의 보안 저하의 원인으로는 안전 정렬 부족, 편향된 지식 증류, 파라미터 공유 및 양자화 기법을 논의했습니다.

- **Performance Highlights**: 연구의 결과, 많은 SLM들이 LLMs보다 jailbreak 공격에 더욱 취약하다는 것이 밝혀졌습니다. 또한, 기존 방어 방법들이 SLM의 내성을 강화하는 데 유의미한 효과가 있음을 입증했습니다. 앞으로 SLM의 보안 챌린지를 강조하고, 더욱 견고하고 안전한 SLM 개발을 위한 귀중한 통찰력을 제공하는 것을 목표로 하고 있습니다.



### MIND: Towards Immersive Psychological Healing with Multi-agent Inner Dialogu (https://arxiv.org/abs/2502.19860)
- **What's New**: 최근 정신 건강 이슈가 증가하는 가운데, 본 논문에서는 기존의 상담이나 챗봇과 같은 전통적인 치유 방법의 한계를 지적하고, MIND (Multi-agent INner Dialogue)라는 혁신적인 접근 방식을 제안합니다. 이 새로운 패러다임은 대화형 내부 대화를 통해 사용자가 더욱 몰입할 수 있는 심리적 치유 환경을 제공합니다. 또한, MIND는 각기 다른 역할을 수행하는 LLM (Large Language Model) 에이전트를 활용하여 사용자와의 상호작용을 극대화합니다.

- **Technical Details**: MIND의 전체 구조는 트리거(trigger), 악마(devil), 가이드(guide), 전략가(strategist) 네 가지 역할을 하는 에이전트로 구성됩니다. 각각의 에이전트는 사용자의 감정 상태에 따라 적절한 심리적 지원을 제공하며, 내적인 대화체계를 통해 정서적 조절을 돕습니다. 이 프레임워크에서는 각 에이전트가 사용자의 불안이나 우울감과 같은 인지 왜곡을 처리하고, 이를 통해 심리적 가이드를 제공합니다.

- **Performance Highlights**: 다양한 실험을 통해 MIND는 전통적인 심리 상담이나 챗봇, 기존 공감 훈련 방법과 비교하여 더 우수한 사용자 경험을 제공임을 입증하였습니다. 사용자는 내부 대화와 서로 다른 관점을 통한 메타인지적 성찰을 통해 깊은 자기 화해를 경험하게 됩니다. 이 연구 결과는 MIND가 LLM의 생성 능력을 활용하여 접근성과 효과성을 겸비한 정신 건강 지원을 제공할 수 있는 잠재력을 가지고 있음을 보여줍니다.



### ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments (https://arxiv.org/abs/2502.19852)
Comments:
          ICLR 2025

- **What's New**: 본 논문은 기존 코드 생성 벤치마크의 한계를 극복하기 위해 새로운 벤치마크 세트를 제안합니다. 특히, 다단계 상호작용(multi-turn interactions)에서 LLM의 성능을 평가하는 데 필요한 다양성을 반영하도록 설계되었습니다. CONVCODEWORLD라는 새로운 환경을 도입하여 9개의 상호작용 코드 생성 시나리오를 시스템적으로 구현하였습니다.

- **Technical Details**: CONVCODEWORLD는 세 가지 유형의 피드백을 조합하여 코드 생성의 질을 평가하는 독특한 환경을 제공합니다: (a) 컴파일 피드백(compilation feedback); (b) 부분적 및 전체 테스트 커버리지(execution feedback); (c) 다양한 수준의 전문성을 가진 GPT-4o에 의해 생성된 구두 피드백(verbal feedback). 또한, CONVCODEBENCH는 사전 생성된 피드백 로그를 사용하여 비용을 절감하고, 생동감 있는 성과와 강한 상관관계를 유지합니다.

- **Performance Highlights**: 광범위한 평가를 통해 확인된 주요 인사이트는 다음과 같습니다: (a) 제공된 피드백에 따라 LLM의 성과가 크게 달라집니다; (b) 충분한 피드백을 받은 약한 LLM이 피드백 없는 최신 모델보다 성능이 더 좋을 수 있습니다; (c) 특정 피드백 조합에 대한 학습이 LLM의 새로운 조합 활용 능력을 제한할 수 있습니다. 이러한 통찰들은 LLM의 성능 평가에서 다단계 상호작용의 중요성을 강조합니다.



### Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation (https://arxiv.org/abs/2502.19830)
- **What's New**: 본 연구에서는 self-consistency를 새로운 관점에서 재정의하여 그 동적 배분 정렬 문제(dynamical distributional alignment problem)로 접근하였습니다. 기존의 고정된 true distribution 개념을 넘어, decoding temperature가 진정한 답 분포에 미치는 중요성을 조명합니다. 이 연구는 샘플 수에 제한 사항이 있을 때에서도 self-consistency의 효과를 극대화할 수 있는 가능성을 열어줍니다. 또한, 동적 온도 조정을 통한 신뢰 기반의 샘플링 방식이 제안되었습니다.

- **Technical Details**: self-consistency는 다양한 stochastic samples를 집계하여 모델 성능을 향상시키는 방법으로, 다수결 투표를 사용하는 방법론입니다. 이 연구에서 temperature 매개변수가 랜덤 샘플링과 진정한 답 분포를 조절하는데 중요한 역할을 함을 보였습니다. 다양한 실험을 통해 높은 온도가 더 많은 샘플을 필요로 하고, 낮은 온도가 모델 편향을 발생시킬 수 있음을 밝혔다. 이를 통해 온도를 동적으로 조정하면서 샘플링 배포의 신뢰도를 높이고, 새로운 분포를 탐색할 수 있도록 하는 기초를 마련하였습니다.

- **Performance Highlights**: 수학적 추론 작업에 대한 실험 결과, 제안된 신뢰 기반의 온도 조정 기법이 기존 고정된 다양성 기준보다 우수한 성능을 보였습니다. 제한된 샘플을 사용하더라도, 평균 및 최상의 성능이 향상되는 결과를 보여주었습니다. 추가적인 데이터나 모듈 없이도 초기 온도 변화에 따라 성능이 일관되게 개선되는 것을 확인했습니다. 이를 통해 self-consistency가 샘플링 동력학과 진화하는 답 분포 간의 동기화 문제로 작용할 수 있음을 입증하였습니다.



### GraphSparseNet: a Novel Method for Large Scale Trafffic Flow Prediction (https://arxiv.org/abs/2502.19823)
- **What's New**: 이 논문에서는 GraphSparseNet (GSNet)이라는 새로운 프레임워크를 소개하여 GNN 기반의 교통 흐름 예측 모델의 확장성(Scalability)과 정확성(Accuracy)을 향상시키고자 합니다. GSNet은 Feature Extractor와 Relational Compressor의 두 핵심 모듈로 구성되어 있으며, 이 모듈들은 선형 시간 및 공간 복잡성으로 운영되어 모델의 전체 계산 복잡성을 선형적으로 줄입니다. 이 접근법은 기존 방법들이 갖고 있는 복잡성 문제를 해결하며, 인기 있는 학습 모델들과 비교했을 때 학습 시간이 3.51배 단축됨을 보여줍니다.

- **Technical Details**: GSNet의 혁신적인 설계는 두 개의 모듈로 나뉘는데, Feature Extractor는 그래프 노드의 특징을 포착하고 인코딩하는 역할을 하며, Relational Compressor는 노드 간의 희소한 관계를 모델링합니다. 두 모듈 모두 선형 시간과 공간 복잡성으로 작동하여 그래프 노드 수가 증가함에 따라 효율적으로 확장할 수 있도록 구성되어 있습니다. 기존 방법들이 O(N²) 복잡성을 가진 것에 비해 GSNet는 O(N)으로 줄어들었습니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험을 수행한 결과, GraphSparseNet은 모든 평가된 데이터셋에서 두드러진 성과를 달성하였습니다. GSNet은 이전 상태의 Linear 모델과 비교하여 약 3.51배의 훈련 시간을 단축시키면서도 높은 예측 성능을 유지했습니다. 이러한 성과는 GSNet이 대규모 교통 시공간 데이터에 대한 확장성 문제를 효과적으로 해결했음을 보여줍니다.



### Foot-In-The-Door: A Multi-turn Jailbreak for LLMs (https://arxiv.org/abs/2502.19820)
Comments:
          19 pages, 8 figures

- **What's New**: 이 논문은 실세계 응용 프로그램에서 대형 언어 모델(AI models)의 통합이 증가함에 따라 AI 안전을 보장하는 방법에 대해 다룹니다. 특히, 우리는 'jailbreak'라는 키 문제를 탐구하고 있으며, 이는 악의적인 프롬프트가 내장된 안전 장치를 우회하여 허용되지 않은 유해한 출력을 유도하는 것을 포함합니다. 새로운 다중 턴 다이내믹 메소드인 FITD를 소개하며, 이는 심리학의 'foot-in-the-door' 원칙에서 영감을 받았습니다.

- **Technical Details**: FITD 방법론은 초기 소액의 약속이 더 큰 약속이나 비윤리적 요청에 대한 저항을 낮추는 현상을 활용합니다. 이 접근 방식은 사용자 쿼리의 악의적인 의도를 점진적으로 증가시키며, 중간 다리를 통한 프롬프트를 사용하여 모델의 응답을 유도합니다. 실험은 두 개의 jailbreak 벤치마크에서 수행되었으며, 이는 seven 개의 널리 사용되는 모델에 걸쳐 평균 94%의 공격 성공률을 나타냅니다.

- **Performance Highlights**: FITD는 기존의 최첨단 방법보다 더 우수한 성능을 보였습니다. 또한, LLM(self-corruption) 자가 부패에 대한 심층 분석을 제공하여 현재 정렬 전략의 취약성을 강조했습니다. 이 논문은 다중 턴 프롬프트가 내재하고 있는 위험성에 대해서도 경고하며, 관련된 소스 코드는 공개되어 있습니다.



### Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts (https://arxiv.org/abs/2502.19811)
- **What's New**: COMET는 Mixture-of-Experts (MoE) 시스템의 새로운 최적화 버전으로, 통신과 계산의 정밀한 겹침(fine-grained overlapping)을 통해 성능을 극대화합니다. 기존의 MoE 시스템에서 보이는 통신 오버헤드를 줄이고, 대규모 분산 환경에서 그 효율성 향상을 가져옵니다. 이를 통해 MoE 모델의 GPU 사용 시간을 효율적으로 절약할 수 있습니다.

- **Technical Details**: COMET는 복잡한 데이터 의존성 분석(data dependency analysis)과 작업 재조정(task rescheduling) 기법을 이용하여 통신과 계산의 정밀한 겹침을 달성합니다. 이 시스템은 GPU 하드웨어 리소스를 동적으로 할당하여 연산과 통신 간의 부하를 균형 있게 조절합니다. 또한 공유 텐서(shared tensor)를 분석하여 두 작업 간의 겹침을 극대화하여 동시 실행을 확보합니다.

- **Performance Highlights**: COMET는 MoE 레이어의 실행 속도를 약 1.96배 가속화하며, 전체 MoE 모델 실행에 대해서는 평균 1.71배의 성능 향상을 보여주었습니다. 또한 10,000개 이상의 GPU로 구성된 생산 환경에서도 적용되어 수백만 시간의 GPU 시간을 절약하는 성과를 거두었습니다. COMET는 Megatron-LM에 통합되어 여러 병렬 전략에서 그 효과를 검증하였습니다.



### Implicit Search via Discrete Diffusion: A Study on Chess (https://arxiv.org/abs/2502.19805)
Comments:
          ICLR 2025

- **What's New**: 이 연구는 AlphaGo 이후의 AI의 문제 해결 방식에서 새로운 관심이 생긴 탐색 기술에 대한 내용을 다룹니다. 특히 Monte Carlo Tree Search (MCTS)와 같은 검색 기법이 Large Language Models (LLMs)에 어떻게 적용될 수 있는지를 탐구합니다. 제안된 DiffuSearch라는 모델은 명시적 검색에 의존하지 않고도 다음 토큰(또는 행동) 예측을 향상시키기 위한 미래 세계를 포함하는 모델링을 진행합니다. 이를 통해 DiffuSearch가 기존의 정책보다 뛰어난 성능을 보여준다는 점이 강조됩니다.

- **Technical Details**: DiffuSearch는 여러 단계의 생성 과정을 통해 미래 정보를 예측하고 활용하는 과정을 포함합니다. 이 모델은 명시적 검색 없이도 확산 모델(diffusion model)을 통해 스스로 계획을 세우고 예측하는 능력을 가집니다. 특히 chess라는 고전 게임에 적용하여 상대방의 다음 수를 예측하는 정책을 학습하는 과정에서, 내부 양방향 자기 주의 메커니즘을 활용하여 현재 정책을 반복적으로 개선합니다. 이러한 구조는 명시적 검색이 필요 없도록 설계되어 있습니다.

- **Performance Highlights**: DiffuSearch는 제어된 실험에서 기존의 one-step 정책보다 19.2% 더 나은 행동 정확도를 보였으며, MCTS 기반 정책에 비해서는 14% 더 우수한 성과를 나타냈습니다. 또한, 퍼즐 해결 능력에서 30% 향상이 있었으며 Elo 등급에서는 540점을 증가시켜 게임 성능에서도 뛰어난 결과를 기록했습니다. 이런 결과들은 DiffuSearch가 명시적 검색 대신 미래 정보를 활용하여 다음 행동 예측을 향상시킬 수 있는 방법으로 작용할 수 있음을 나타냅니다.



### Mixtera: A Data Plane for Foundation Model Training (https://arxiv.org/abs/2502.19790)
Comments:
          under submission

- **What's New**: 본 논문에서는 Mixtera라는 새로운 데이터 플레인을 소개합니다. 이 시스템은 사용자가 훈련 중 필요한 데이터 샘플을 비율과 순서로 선언적으로 표현할 수 있게 해줍니다. Mixtera는 기존 훈련 데이터 컬렉션 위에 배포되는 중앙 집중식, 읽기 전용 레이어입니다.

- **Technical Details**: Mixtera는 파일시스템 구조와 독립적으로 작동하며, 다양한 속성(예: 언어, 소스 데이터셋)에 대해 혼합을 지원합니다. 또한 모델 피드백에 기반한 혼합의 동적 조정도 가능하여, 효과적인 데이터 샘플링을 실현합니다. 시스템은 Adaptive Data Optimization (ADO) 알고리즘을 구현하여 최신 혼합 전략의 발전을 지원합니다.

- **Performance Highlights**: Mixtera는 훈련 성능에 병목 현상을 일으키지 않으며, 256 GH200 슈퍼칩에까지 확장 가능하다는 것을 실험적으로 입증하였습니다. 논문에서는 Mixtera가 비전-언어 모델(vision-language models)에서도 중요한 역할을 하는 것을 탐색하였습니다.



### NaijaNLP: A Survey of Nigerian Low-Resource Languages (https://arxiv.org/abs/2502.19784)
Comments:
          35 pages, 2 figures, 4 tables

- **What's New**: 나이지리아에는 500개 이상의 언어가 있지만, 하우사(Hausa), 요루바(Yorùbá), 이그보(Igbo) 세 언어가 1억 7500만 명 이상의 사람들에 의해 사용되며 60% 이상을 차지합니다. 이러한 언어는 계산 언어학(computational linguistics) 작업을 지원할 자원이 부족하여 저자원(low-resource) 언어로 분류됩니다. 본 연구는 이 세 가지 주요 나이지리아 언어에 대한 저자원 자연어 처리(NLP)의 발전을 포괄적으로 검토한 첫 번째 연구로, 언어 이해 및 생성과 같은 복잡한 작업을 지원하는 자원 부족과 이를 해결하기 위한 노력들을 검토합니다.

- **Technical Details**: 전 세계 언어의 90% 이상은 저자원(low-resource) 언어로 분류되며, 이들 언어는 부족한 평행 데이터(parallel source-target data)로 인해 통계적인 방법을 직접 적용하기 어렵습니다. 본 연구에서는 NaijaNLP(나이지리아의 세 주요 언어에 대한 자연어 처리) 연구의 현재 상태를 분석하고, 기존의 언어 자원, 도구 및 커뮤니티 지원을 평가하며, 이를 기반으로 향후 발전을 위한 전략을 제시합니다. 또한, 기존 연구에서 발견된 자원의 부족과 데이터의 변동성 문제점을 논의합니다.

- **Performance Highlights**: 하우사, 요루바, 이그보와 같은 저자원 언어에 대한 연구는 증가하고 있지만, 검토한 연구 중에서 오직 25.1%만이 새로운 언어 자원에 기여했습니다. 이는 기존 데이터를 재사용하는 경향이 강하며, 고유한 문제들이 여전히 연구되지 않고 있음을 보여줍니다. 따라서 자원 확충, 종합적인 주석 작업(annotation)의 필요성과 개방형 협업 이니셔티브 개발을 강조하며, 우리는 NaijaNLP와 저자원 NLP의 발전을 위해 더욱 광범위한 노력이 필요하다고 주장합니다.



### Do Retrieval-Augmented Language Models Adapt to Varying User Needs? (https://arxiv.org/abs/2502.19779)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Language Models (RALMs) 평가를 위한 새로운 프레임워크를 제안했습니다. 사용자 요구에 따라 모델의 응답 특성을 측정하는 데 초점을 맞추었으며, 세 가지 사용자 케이스(Context-Exclusive, Context-First, Memory-First)와 세 가지 컨텍스트 설정(Context Matching, Knowledge Conflict, Information Irrelevant)을 도입하여 실제 애플리케이션의 복잡성을 반영하고자 했습니다.

- **Technical Details**: 이 프레임워크는 사용자가 요구하는 정보의 유형에 따라 모델이 어떻게 반응하는지를 체계적으로 평가합니다. 사용자는 외부 정보와 내부 지식 중 어떤 것을 우선시할지를 지시할 수 있으며, 이러한 지시사항에 따라 모델의 성능이 어떻게 달라지는지 분석합니다. 우리의 실험은 URAQ 데이터셋을 포함하여 여러 QA 데이터셋에서 수행되었으며, 두 개의 모델 계열(Llama3.1, Qwen2.5)에 대해 다양한 모델 크기와 검색된 컨텍스트 수로 평가되었습니다.

- **Performance Highlights**: 주요 실험 결과에 따르면 현재의 언어 모델은 다양한 사용자 요구를 충족시키는 데 어려움을 겪고 있으며, 모든 데이터셋에서 50% 이하의 정확도를 기록했습니다. 또한, 컨텍스트 제약이 모델 성능에 미치는 영향을 확인했으며, 이상적인 검색 결과에서는 성능이 저하되는 경향이 있음을 발견했습니다. 모델 계열에 따라 성능 차이가 두드러지며, 특정 상황에서는 특정 모델이 다른 모델보다 우수한 성능을 발휘하는 것으로 나타났습니다.



### The erasure of intensive livestock farming in text-to-image generative AI (https://arxiv.org/abs/2502.19771)
- **What's New**: 이번 연구는 Generative AI가 농장 동물에 대한 편견을 어떻게 재현하는지를 분석합니다. ChatGPT와 연계된 DALL-E 3 모델을 사용하여, 이 AI가 소와 돼지가 있는 전통적인 농장 장면을 어떻게 그리는지를 살펴보았습니다. AI가 만들어내는 이미지는 종종 현실과 동떨어진 목장을 과대대표하며, 이는 공공의 인식에 큰 영향을 미칠 수 있습니다.

- **Technical Details**: DALL-E 3는 텍스트 프롬프트를 기반으로 이미지를 생성하는 최첨단 모델로, 사용자의 요청에 따라 프롬프트를 자동 수정하는 기능을 가지고 있습니다. 연구에서는 48개의 개별 프롬프트에 대해 각각 100개의 이미지를 생성해 총 4,800개의 이미지를 분석했습니다. 주요 연구 질문은 소와 돼지 농장을 묘사할 때 AI의 전형적인 특징과 현실성을 탐구하는 데 중점을 두었습니다.

- **Performance Highlights**: DALL-E 3는 기본 프롬프트에 대해 100%의 이미지를 목초지를 지나가는 소로 생성했으며, 이는 낭만적인 농장 이미지를 강화하는 경향을 보였습니다. 반면, 자동 프롬프트 수정을 비활성화할 경우, 내부 사육 환경을 더 잘 반영하는 이미지들이 생성되었으며, 이는 현대 농업의 실태와 더 가깝습니다. 이러한 발견은 AI의 권력 집중이 어떻게 사회적 통념에 영향을 미치는지를 보여줍니다.



### Obtaining Example-Based Explanations from Deep Neural Networks (https://arxiv.org/abs/2502.19768)
Comments:
          To be published in the Symposium on Intelligent Data Analysis (IDA) 2025

- **What's New**: 이번 연구에서는 딥 뉴럴 네트워크(DNN)에서 예제 기반 설명(example-based explanations)을 추출하기 위한 새로운 기술인 EBE-DNN을 제안합니다. 기존의 예제 기반 설명 기술은 주로 KNN 및 랜덤 포레스트와 같은 특정 모델에만 초점을 맞추고 있었으나, 이번 방법은 DNN의 강력한 특징 추출 기능을 활용합니다. EBE-DNN은 KNN 분류기를 사용하여 예측을 형성하고, 이를 통해 적은 수의 훈련 예제로 높은 집중도의 예제 귀속(example attribution)을 제공할 수 있습니다.

- **Technical Details**: EBE-DNN은 특정 DNN 레이어에서 추출한 임베딩(embedding)을 활용하여 테스트 예제의 예제 귀속을 생성하고 레이블을 예측하는 알고리즘입니다. 입력데이터는 모델에 따라 변환된 고차원 예제로, 이러한 변환은 저수준 및 고수준 특성을 반영하는 유용한 피처 공간으로의 전환을 목표로 합니다. 그런 다음, 변환된 공간에서 가장 가까운 k개의 이웃을 검색하고 이들로부터 예제 귀속을 산출하여 테스트 인스턴스의 예측 레이블을 생성합니다.

- **Performance Highlights**: EBE-DNN의 실험 결과는 원래 DNN의 정확도를 유지하면서도 극히 적은 수의 훈련 예제를 사용해 예측을 설명할 수 있음을 보여주었습니다. 또한, 임베딩을 생성하기 위해 선택한 레이어가 정확도에 큰 영향을 미친다는 중요한 발견이 있었습니다. 이 방법은 예측의 신뢰성과 투명성을 높여 사용자들이 심층 학습 모델을 신뢰하고 관리하는 데 기여할 수 있습니다.



### Learning with Exact Invariances in Polynomial Tim (https://arxiv.org/abs/2502.19758)
- **What's New**: 이번 연구는 커널 회귀(kernel regression) 맥락에서 정확한 불변성을 학습하기 위한 통계-계산적 절충(trade-offs)을 심층적으로 탐구합니다. 기존의 접근법들은 폴리노미얼 시간(polynomial-time) 해법을 제공하지 않거나 커널 환경에서 적용이 불가능했습니다. 그러나 우리는 입력 공간의 기하학적 특성에 대한 오라클 접근을 활용하여 정확한 불변성을 가진 분류기를 학습하는 폴리노미얼 시간 알고리즘을 제안하였습니다.

- **Technical Details**: 이 연구에서는 인공지능 모델이 데이터 내의 대칭성(symmetry) 및 불변성(invariance)을 인식하고 활용할 수 있도록 하는 방법을 탐구합니다. 전통적인 방법들은 데이터 증대(data augmentation) 및 그룹 평균(group averaging) 접근법들을 포함하지만, 이러한 접근은 계산적으로 비효율적일 수 있습니다. 본 연구는 스펙트럼 이론(spectral theory)과 최적화(optimization) 도구를 활용하여 커널 방법에서 불변성과 관련된 문제를 새로운 유한 차원 볼록 이차 프로그램(convex quadratic program) 집합으로 재구성하였습니다.

- **Performance Highlights**: 우리의 알고리즘은 기존 커널 회귀 문제와 동일한 과도한 모집단 위험(excess population risk) 또는 일반화 오류(generalization error)를 달성하는 동시에 통계적 및 계산적으로 효율적이라는 장점을 가지고 있습니다. 이 연구는 정확한 불변성을 달성하는 첫 번째 폴리노미얼 알고리즘을 도입하였고, 이는 신경망과 같은 다양한 응용 분야에서 중요할 수 있습니다. 또한, 이 해결책은 통계 및 계산 복잡성 사이의 균형을 맞추는 데 주안점을 두고 있으며, 초점은 정확한 불변성을 달성하는 것입니다.



### Probabilistic Federated Prompt-Tuning with Non-IID and Imbalanced Data (https://arxiv.org/abs/2502.19752)
Comments:
          Accepted at NeurIPS-24

- **What's New**: 이 논문은 머신 러닝 분야에서 사전 훈련된(pre-trained) 모델을 미세 조정하는 방법을 제안합니다. 기존의 연합 학습(federated learning) 방식이 서로 다른 로컬 데이터 분포로 인해 비효율적임을 지적하고, 더 효과적인 프롬프트 튜닝(prompt-tuning) 방법론을 통합하여 개선하였습니다. 이를 통해 연합 학습을 분산 세트 모델링(distributed set modeling) 작업으로 변환하여 정보의 요약을 통해 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 계층적 확률적 모델링(hierarchical probabilistic modeling)을 사용하여 로컬 프롬프트(prompt)의 생성과 정렬을 설명합니다. 서버 집계 과정(server aggregation step)은 유사한 문맥 정보를 부호화하는 로컬 프롬프트를 발견하고 집계하는 과정으로 설정됩니다. 결과적으로, 각 로컬 프롬프트 세트는 글로벌 요약 프롬프트(global summarizing prompts)로 초기화된 확률적 탐색(probabilistic exploration)으로 간주됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 실험을 통해 기존 연합 프롬프트 튜닝 방법들과 비교하였고, 극단적인 데이터 비균형(extreme data imbalance)과 이질성(heterogeneity) 상황에서도 가장 효과적인 성능을 나타냄을 확인하였습니다. 실험 결과, 제안된 방법이 연합 학습의 성능 향상에 기여하는 것을 보여줍니다. 이는 특히 컴퓨터 비전 데이터셋에서 효과적임을 입증하고 있으며, 모델의 안정성을 높이는 데 기여합니다.



### CNsum:Automatic Summarization for Chinese News Tex (https://arxiv.org/abs/2502.19723)
Comments:
          WASA 2022

- **What's New**: 이번 연구는 방대한 데이터에서 유용한 정보를 효율적으로 추출하는 것을 목표로 하고 있으며, 특히 한중 뉴스 텍스트 요약 생성에 초점을 맞추고 있습니다. 최근 트렌드는 Transformer 구조의 프리트레인(Pre-trained) 언어 모델이 다양한 자연어 처리(Natural Language Processing, NLP) 작업에서 큰 성과를 거두었다는 점입니다. 본 논문에서는 Transformer 구조를 기반으로 한 중국 뉴스 텍스트 요약 모델(CNsum)을 제안합니다.

- **Technical Details**: CNsum 모델은 중국 데이터셋인 THUCNews에서 테스트되었습니다. Transformer 아키텍처를 활용하여 모델이 최적화 되었으며, 다양한 검증 방법으로 성능이 입증되었습니다. 이 모델은 중국어에 특화된 요약 기법을 사용하여 텍스트 요약의 품질을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, CNsum 모델은 기준 모델들에 비해 더 우수한 ROUGE 점수를 기록했습니다. 이는 제안된 모델이 기존의 베이스라인(Base-line) 모델들보다 실제로 더 뛰어난 성과를 보였음을 나타냅니다. 이러한 결과는 CNsum의 효과성을 부각시키며 향후 연구에 대한 기초 자료를 제공합니다.



### Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning (https://arxiv.org/abs/2502.19717)
Comments:
          Accepted by the Thirteenth International Conference on Learning Representations (ICLR 2025)

- **What's New**: 이번 연구에서는 협력적 다중 에이전트 강화 학습(MARL)에서 확장 가능한 커뮤니케이션 프로토콜을 개발하는 데 중점을 두고 있습니다. 이전의 방법들이 최적의 쌍방 통신 링크 선택에 집중한 반면, 제안된 방법은 전 세계적인 관점에서의 커뮤니케이션 토폴로지 설계를 채택하고 있습니다. 이를 통해 ExpoComm이라는 이름의 확장 가능한 커뮤니케이션 프로토콜을 제안하며, 지수적(topology) 구조에 의해 정보 전파를 빠르게 수행할 수 있도록 합니다.

- **Technical Details**: ExpoComm은 에이전트 간의 효과적인 정보를 전파하기 위해 지수적 토폴로지를 기반으로 설계되었습니다. 이는 고유한 소규모 속성과 작은 직경 특성을 활용하여, 모든 에이전트 간의 메시지 흐름을 효율적으로 지원합니다. 또한, memory-based message processors와 보조 작업을 활용하여 전 세계 정보를 반영하는 메시지를 생성함으로써 의사결정을 지원합니다.

- **Performance Highlights**: 대규모 협력 벤치마크인 MAgent와 Infrastructure Management Planning을 포함한 12개의 시나리오에서 ExpoComm의 우수한 성능과 강력한 zero-shot 전이 가능성을 입증하였습니다. 특히, 여러 에이전트를 처리하는 데 있어 기존의 통신 전략에 비해 월등한 성능을 보여주며, 에이전트 수에 관계없이 저렴한 비용으로 효과적인 커뮤니케이션을 가능하게 합니다.



### Extending the Hegselmann-Krause Model of Opinion Dynamics to include AI Oracles (https://arxiv.org/abs/2502.19701)
- **What's New**: 이번 논문에서는 Hegselmann-Krause (HK) 모델을 확장하여 인공지능 오라클(Artificially Intelligent Oracle)을 통합한 의견 동역학을 다룹니다. 이 오라클은 커뮤니티 구성원의 의견을 평균내어 주며, 이를 통해 시간 경과에 따른 의견 변화 과정을 모사합니다. 이 새로운 모델을 통해 다양한 조건에서 의견 수렴(convergence)에 대한 인사이트를 제공합니다.

- **Technical Details**: 모델은 에이전트 기반 시뮬레이션(agent-based simulations)을 통해 분석됩니다. 연구는 네 가지 주요 시나리오를 조사하여 개인들이 오라클의 의견을 어떻게 반영하는지를 다룹니다. 시뮬레이션 결과에 따르면, 오라클 전용 접근과 진정한 가치(T)에 대한 접근이 각각 의견 수렴에 미치는 영향을 보여줍니다.

- **Performance Highlights**: 시뮬레이션 결과는 사람들이 오라클의 의견만 반영할 경우 모든 의견이 공통값으로 수렴하게 됨을 보입니다. 반면에 집단의 일부가만 진정한 가치에 접근할 때는 수렴이 보장되지 않지만, 오라클에 대한 보편적 접근이 수렴을 보장하는 조건도 존재합니다. 이러한 결과는 커뮤니티 내 의견 변화에 중요한 통찰력을 제공합니다.



### BEVDiffuser: Plug-and-Play Diffusion Model for BEV Denoising with Ground-Truth Guidanc (https://arxiv.org/abs/2502.19694)
Comments:
          CVPR 2025

- **What's New**: 본 연구에서는 BEV (Bird's-eye-view) 표현의 노이즈 문제를 해결하기 위한 새로운 확산 모델인 BEVDiffuser를 제안합니다. BEVDiffuser는 실제 객체 레이아웃을 가이던스로 하여 BEV 특징 맵을 효과적으로 노이즈 제거합니다. 이 모델은 기존 BEV 모델을 수정할 필요 없이 플러그 앤 플레이 방식으로 훈련 중에 작동하여 BEV 표현을 향상시킵니다.

- **Technical Details**: BEVDiffuser는 BEVFormer, BEVFusion과 같은 기존 BEV 모델에서 생성된 특징 맵에 다양한 수준의 노이즈를 추가하여 훈련됩니다. 훈련 후에는 BEV 특징 맵을 정화하여 추가적인 감독을 제공하는 방식으로 기존 BEV 모델의 성능을 개선합니다. 또한, BEVDiffuser는 훈련 시간이 끝나면 제거되며, 추론 시 추가적인 컴퓨팅 지연 없이도 강력한 성능을 유지합니다.

- **Performance Highlights**: nuScenes 데이터셋을 통해 실시한 실험 결과, BEVDiffuser는 3D 객체 탐지에서 mAP 12.3% 및 NDS 10.1%의 현저한 개선을 보여주었습니다. 장기적인 객체 탐지 및 다양한 환경 조건에서도 성능이 크게 향상되었으며, 질적으로도 고품질의 BEV 생성 능력을 입증했습니다. 이러한 성능 개선은 자율 주행의 발전을 위한 대규모 데이터 수집에 기여할 것으로 기대됩니다.



### Accurate and Scalable Graph Neural Networks via Message Invarianc (https://arxiv.org/abs/2502.19693)
- **What's New**: 이 논문에서는 전통적인 message passing 방식의 그래프 신경망(GNNs)에서 발생하는 문제를 해결하기 위해 새로운 미니 배치 접근법인 topological compensation (TOP)을 제안합니다. TOP은 전체 message passing을 MP-IB만을 통해 효율적으로 계산함으로써 계산 비용을 줄이고 정확성을 유지합니다. 새롭게 정의된 message invariance 개념을 통해 기존의 비효율적인 방식에서 벗어나 빨라진 계산을 가능하게 합니다.

- **Technical Details**: TOP은 MP-OB를 필요로 하지 않고, message invariance를 통해 출발 노드에 대한 빠른 변환을 수행합니다. 이로 인해 더 많은 노드와 엣지를 GPU에 저장할 필요가 줄어들어 대규모 그래프에서도 GNN을 효율적으로 사용할 수 있게 됩니다. 이 접근법은 그래프의 다양한 특성과 이를 반영하는 선형 회귀를 사용하여 임베딩 간의 선형 독립성을 학습하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, TOP은 수백만 개의 노드와 수십억 개의 엣지를 가진 대규모 그래프에서 기존 미니 배치 방법보다 수십 배 이상 빨라지며 정확성 저하를 최소화합니다. TOP의 수렴 속도는 기존의 방법들에 비해 현저히 빠르며, 해당 연구가 다양한 실제 데이터 세트에서 유효성을 입증한 점도 주목받고 있습니다.



### Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach (https://arxiv.org/abs/2502.19691)
Comments:
          Accepted to CVPR 2025

- **What's New**: 본 논문에서는 Energy-based Active Open-set Annotation (EAOA) 프레임워크를 제안하여, 알려지지 않은 클래스가 존재하는 상황에서도 우수한 성능을 발휘할 수 있는 방법을 모색합니다. 기존의 방법들은 알려진 클래스에 속할 가능성이 높은 query 예제를 우선시하거나, 불확실한 예제를 쿼리하는 데 중점을 두어 최적의 성능을 발휘하지 못했습니다. EAOA는 이러한 두 가지 불확실성을 결합하여 모델 훈련의 효율성을 높이는 새로운 접근법을 제시합니다. 또한, 실제 데이터에 대해 높은 쿼리 정밀도와 낮은 훈련 오버헤드를 유지하면서 최첨단 성능을 달성하였습니다.

- **Technical Details**: EAOA는 (C+1) 클래스 검출기와 타겟 분류기로 구성되어 있습니다. 이 시스템은 에너지 기반의 epistemic uncertainty (EU) 측정 방법과 특성 기반의 energy loss를 도입하여 검출기를 훈련시킵니다. 또한, 타겟 분류기에는 가능한 aleatoric uncertainty (AU)를 측정하는 에너지 기반 지표를 적용하여 보다 정교하게 클래스 식별을 수행합니다. 핵심 구성 요소 중 하나는 목표 기반의 적응형 샘플링 전략으로, 이는 높은 AU 점수를 가진 쿼리 집합을 형성하기 위해 낮은 EU 점수를 가진 후보 집합을 우선 생성하는 방식입니다.

- **Performance Highlights**: 광범위한 실험을 통해 EAOA가 기존의 최신 연구 결과들보다 테스트 정확도와 쿼리 정밀도, 훈련 효율성에서 뛰어난 성능을 발휘함을 입증했습니다. EAOA는 open-world 시나리오에서도 유용한 정보가 포함된 예제를 효과적으로 쿼리하여, 다양한 데이터 상황에서도 일관된 결과를 보여줍니다. 특히, EAOA는 c랑 close-set 특성을 적절히 조합하여 높은 학습 효과를 달성했습니다.



### Risk-aware Integrated Task and Motion Planning for Versatile Snake Robots under Localization Failures (https://arxiv.org/abs/2502.19690)
Comments:
          8 pages, 9 figures. Accepted article with supplemental material for presentation at the 2025 IEEE International Conference on Robotics and Automation (ICRA)

- **What's New**: 이번 연구에서는 Snake robots의 강력한 인식 및 로컬라이제이션 문제를 해결하기 위해 Blind-motion with Intermittently Scheduled Scans (BLISS)를 제안합니다. BLISS는 proprioception-only mobility와 간헐적 스캔을 결합하여 로컬라이제이션 실패와 충돌 위험에 견디도록 설계되었습니다. 이 방법은 기존의 각 인지 제어 방식보다 유연하면서도 효율적으로 작업 및 모션 계획(Task and Motion Planning, TAMP) 문제를 해결합니다.

- **Technical Details**: BLISS는 Chance-Constrained Hybrid Partially Observable Markov Decision Process (CC-HPOMDP)로 형식화되며, 이는 일반적으로 고계수(curse of history) 문제로 인해 계산적으로 비효율적입니다. 그러나, 우리는 이를 해결하기 위해 CC-HPOMDP를 가공하기 용이한 볼록 혼합 정수 선형 프로그래밍(Mixed Integer Linear Program, MILP)으로 재구성하였습니다. 이로 인해 계산 효율성이 크게 향상되며, 최적의 작업-모션 계획을 보다 빠르게 도출할 수 있게 됩니다.

- **Performance Highlights**: 시뮬레이션 및 하드웨어 실험 결과, BLISS는 기존의 POMDP 플래너와 비교하여 계산 효율성에서 10배 이상 향상된 성능을 보여주었으며, 전통적인 2단계 계획 접근 방식에 비해 탐색 시간의 최적화 면에서 50% 이상의 개선을 달성했습니다. EELS Snake robot을 활용하여 실험한 결과, 이 방법이 기존의 베이스라인 접근 방식보다 우수한 성능을 보였음을 확인했습니다.



### M-LLM Based Video Frame Selection for Efficient Video Understanding (https://arxiv.org/abs/2502.19680)
- **What's New**: 최근의 Multi-Modal Large Language Models (M-LLMs)의 발전은 비디오 추론(video reasoning)에서 유망한 결과를 보여주고 있습니다. 기존의 M-LLM 프레임워크는 긴 컨텍스트 비디오에서 입력 프레임 수를 줄이기 위해 단순하고 균일한 샘플링 방법을 적용합니다. 그러나 이러한 방법은 비디오의 특정 구간에서 중요한 맥락을 놓칠 수 있습니다. 이를 해결하기 위해, 우리는 사용자의 쿼리와 보다 관련 있는 프레임을 선택하는 경량 M-LLM 기반의 프레임 선택 방법을 제안합니다.

- **Technical Details**: 제안하는 프레임 선택기를 훈련하기 위해 두 가지 감독 신호를 도입합니다: (i) Spatial 신호는 M-LLM을 통해 각 프레임의 중요성 점수를 생성합니다; (ii) Temporal 신호는 LLM을 이용해 모든 프레임 후보의 캡션을 기반으로 다수의 프레임을 선택합니다. 선택된 프레임은 다운스트림 비디오 M-LLM에 의해 시각적 추론과 질문 답변을 위해 처리됩니다. 이러한 방법은 전체 맥락 길이를 유지하면서, 잡음을 줄이고 모델의 초점을 관련 비디오 세그먼트로 향하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안한 M-LLM 비디오 프레임 선택기가 Medium (ActivityNet, NExT-QA)과 Long (EgoSchema, LongVideoBench) 컨텍스트 비디오 질문 응답 벤치마크에서 다양한 다운스트림 비디오 Large Language Model (video-LLM)의 성능을 향상시킵니다. 특히, 우리는 모델이 질문에 답하기 위해 가장 도움이 되는 프레임에 집중함으로써 비디오 이해의 품질을 유지하면서 처리해야 하는 시각적 컨텍스트를 상당히 줄일 수 있음을 보여주었습니다.



### SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning (https://arxiv.org/abs/2502.19668)
- **What's New**: 본 논문에서는 심혈관 질환 진단을 위한 Electrocardiogram (ECG) 해석의 새로운 접근 방식을 제안합니다. 기존의 ECG Self-Supervised Learning (eSSL) 방법의 한계를 극복하고자 SuPreME이라는 감독형 사전 훈련 프레임워크를 도입하여, 미리 정리된 임상 정보를 활용한 고품질의 정밀한 레이블 데이터셋을 생성합니다.

- **Technical Details**: SuPreME은 Large Language Models (LLMs)를 활용하여 자유 텍스트 ECG 보고서에서 구조화된 임상 엔티티를 추출하고, 노이즈와 불필요한 내용을 필터링합니다. 이 프레임워크는 구조화된 엔티티 레이블과 ECG 신호를 직접 정렬하여, 기존의 복잡한 사전 작업을 피하고 데이터 증강에 대한 의존성을 줄입니다.

- **Performance Highlights**: SuPreME는 127개의 심장 질환을 포함한 6개의 후속 데이터 세트에서 평가되었으며, 제로 샷(zero-shot) 분류에서 1.96% 이상의 성능 개선을 달성했습니다. 이 모델은 전체 fine-tuning 없이도 높은 데이터 효율성을 바탕으로 학습되며, 효과적으로 높은 품질의 ECG 표현을 생성함을 입증하였습니다.



### HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration (https://arxiv.org/abs/2502.19662)
- **What's New**: 새로운 연구에서는 LLMs(대형 언어 모델)의 효율적인 추론을 위해 Quantization(양자화)의 중요성을 강조합니다. 기존의 양자화 방법들은 하드웨어에 무관하고 비트 너비 제약에 한정되어 있으며, Multiply-Accumulate (MAC) 유닛의 타이밍과 에너지 특성과 같은 회로 수준의 통찰력이 부족합니다. 이 연구에서는 다양한 하드웨어에 적응할 수 있는 HALO라는 다목적 프레임워크를 소개합니다.

- **Technical Details**: HALO는 Hardware-Aware Post-Training Quantization (PTQ) 접근 방식을 통해 MAC 유닛의 속성을 활용하여 critical-path(중요 경로) 지연을 최소화하고 동적 주파수 스케일링을 가능하게 합니다. 이러한 방법은 LLM 가속기, 특히 TPUs와 GPUs에서 배치되어 사용됩니다.

- **Performance Highlights**: HALO의 도입으로 평균 270%의 성능 향상과 51%의 에너지 절약을 달성하였으며, 최소한의 정확도 손실로 이러한 성과를 이뤘습니다. 이는 LLM의 효율적인 운영에 기여할 것으로 기대됩니다.



### Med-RLVR: Emerging Medical Reasoning from a 3B base model via reinforcement Learning (https://arxiv.org/abs/2502.19655)
- **What's New**: 최근 강화학습에서 검증 가능한 보상을 바탕으로 한 연구(RLVR)가 주목받고 있습니다. 특히 DeepSeek-R1의 연구 결과는 기본 언어 모델에서 명시적인 추론 감독 없이도 스스로 발전된 추론 능력을 이끌어낼 수 있음을 보여주었습니다. 본 논문에서는 의료 분야에서의 RLVR의 적용 가능성을 탐구하며, Med-RLVR이라는 시스템을 도입했습니다.

- **Technical Details**: Med-RLVR은 의료 분야에서의 MCQA(multiple-choice question answering) 데이터를 활용하여 기본 모델에서 명시적인 추론 감독 없이 의료적 추론을 이끌어내기 위한 초기 탐색 작업입니다. 강화학습 알고리즘으로는 Proximal policy optimization (PPO)을 사용하며, 보상 모델은 검증 함수(verification function)로 설정되어 있습니다. 보상을 계산하는 방법에서는 출력의 형식이나 정답 여부에 따라서 보상을 부여하는 규칙 기반 기능을 활용합니다.

- **Performance Highlights**: Med-RLVR은 전통적인 감독 하의 미세 조정(Supervised Fine-Tuning, SFT)과 유사한 성능을 달성하면서도 분포 외 일반화(out-of-distribution generalization)에서 약 8%의 정확도 개선을 보였습니다. 학습 역학 분석을 통해, 3B 매개변수의 기본 모델에서 명시적인 감독 없이도 추론 능력이 생겨났음을 확인했습니다. 이러한 결과는 더욱 많은 분야에 걸쳐 RLVR의 가능성을 열어줍니다.



### Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning (https://arxiv.org/abs/2502.19652)
- **What's New**: 이 논문에서는 강인한 강화 학습(robust reinforcement learning, RL)을 위한 통합 모듈형 벤치마크인 Robust-Gymnasium을 소개합니다. 이 벤치마크는 블랙박스 아키텍처로 다양한 혼란이 발생할 수 있는 다양한 RL 컴포넌트를 지원합니다. 60개 이상의 제어 및 로봇, 안전 RL, 다중 에이전트 RL 관련 작업 환경을 제공하며, 현재의 방법을 평가하고 강인한 RL 알고리즘 개발을 촉진할 수 있는 도구입니다.

- **Technical Details**: Robust-Gymnasium은 단일 에이전트 RL 문제를 유한 수명 마르코프 결정 과정(finite-horizon Markov decision process, MDP)으로 정의합니다. 이 프레임워크는 에이전트-환경 상호작용에서 다양한 유형의 혼란을 통합할 수 있도록 설계되었습니다. 추가적인 혼란 모듈이 도입되어 에이전트의 상태 관측 및 환경의 변화에 따른 불확실성을 모델링합니다.

- **Performance Highlights**: 실험 결과, 기존의 표준 및 강인한 RL 알고리즘은 복잡한 작업에서 기대에 미치지 못하며, 특히 단일 단계의 혼란에서도 부족함을 보여줍니다. Robust-Gymnasium은 모든 단계에서의 혼란을 포함하여 다양한 작업을 범위에 두고 있는 혁신적인 도구로, 대규모 언어 모델(large language model, LLM)을 활용한 적대적 모델을 이용하여 RL 연구에 새로운 가능성을 보여줍니다.



### AutoBS: Autonomous Base Station Deployment Framework with Reinforcement Learning and Digital Twin Network (https://arxiv.org/abs/2502.19647)
- **What's New**: AutoBS는 6G 네트워크에서 최적의 기지국 (Base Station, BS) 배치를 위해 강화 학습 (Reinforcement Learning, RL) 기반의 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 근접 정책 최적화(Proximal Policy Optimization, PPO) 알고리즘과 PMNet을 활용하여 빠르게 경로 손실을 예측함으로써 커버리지와 용량을 최적화할 수 있는 배치 전략을 학습합니다. AutoBS는 기존의 수작업 문제 해결 방식에서 벗어나 실시간 애플리케이션에 적합하게 개발되었습니다.

- **Technical Details**: AutoBS는 구체적인 사이트 환경을 반영하여 최적의 BS 배치를 자동으로 최적화합니다. BS의 위치는 환경 맵에서 주어진 좌표 (i,j)로 표현되며, 목표는 커버리지(Vm)와 용량(Cm)를 조정하여 최적화를 수행하는 것입니다. PMNet은 신속한 경로 손실 예측을 가능하게 하여 PPO가 각 배치 결정 후 즉시 네트워크 성능을 평가하고 보상 피드백을 받을 수 있도록 합니다.

- **Performance Highlights**: 수치 결과에 따르면 AutoBS는 단일 BS의 경우 95%, 다수의 BS의 경우 90%의 성능을 보여주며, 기존의 전체 탐색 방법보다 inference 시간을 몇 시간에서 밀리세컨드로 단축합니다. 또한, AutoBS는 동적 환경에서도 컴퓨팅 오버헤드를 최소화하며 대규모 6G 네트워크에 잘 맞춰져 있습니다. 이러한 장점들 덕분에 AutoBS는 진정한 실시간 최적화 솔루션으로 자리잡고 있습니다.



### Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (https://arxiv.org/abs/2502.19645)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 Vision-Language-Action 모델(VLAs)의 최적화된 파인튜닝 방법을 제안합니다. OpenVLA 모델을 기반으로, 액션 디코딩, 액션 표현, 학습 목표 등 여러 디자인 선택 사항을 분석하여 최적화 파인튜닝(Optimized Fine-Tuning, OFT) 레시피를 개발했습니다. 이 방법은 기존 VLA 모델의 추론 효율성 및 성능을 크게 향상시키며, 새로운 로봇 환경에서도 뛰어난 성능과 유연성을 자랑합니다.

- **Technical Details**: OFT 레시피는 병렬 디코딩(parallel decoding), 액션 청크(action chunking), 연속 액션 표현(continuous action representation), 그리고 L1 회귀 기반 학습 목표(L1 regression-based learning objective)를 통합합니다. 이를 통해, 모델의 추론 효율성을 증가시키고, 정책 성능을 개선하며, 입력-출력 사양의 유연성을 높였습니다. 연구 결과 제안된 OpenVLA-OFT 모델은 LIBERO 시뮬레이션 벤치마크에서 새로운 최첨단 성과를 기록하며, 평균 성공률을 76.5%에서 97.1%로 증가시켰습니다.

- **Performance Highlights**: OpenVLA-OFT는 실제 세계의 로봇 평가에서도 뛰어난 성과를 보였습니다. ALOHA 로봇에서 고빈도 제어 작업을 성공적으로 수행하며, 이전 파인튜닝된 VLA 모델들보다 평균 성공률에서 최대 15% 성능 향상을 기록했습니다. OpenVLA-OFT는 25 단계 액션 청크를 통해 기본 OpenVLA보다 43배 더 빠른 처리량을 달성하여 실시간 로봇 제어를 가능하게 합니다.



### MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning (https://arxiv.org/abs/2502.19634)
- **What's New**: 이번 연구에서는 MedVLM-R1을 통해 의료 이미지를 분석하는 새로운 방법을 제시합니다. 이 모델은 최종 답변을 생성하는 데 그치지 않고, 명확한 자연어 추론을 제시함으로써 투명성과 신뢰성을 강화합니다. 기존의 Supervised Fine-Tuning (SFT)에 의존하지 않고, 강화 학습(RL) 프레임워크를 활용하여 자체적으로 인간이 해석할 수 있는 추론 경로를 발견하도록 유도합니다.

- **Technical Details**: MedVLM-R1은 기존의 Supervised Fine-Tuning의 한계를 극복하기 위해 Group Relative Policy Optimization (GRPO) 알고리즘을 채택하였습니다. 이는 모델이 최종 답변을 암기하거나 모방하는 대신, 자신의 논리적 단계를 발견하는 데 보상을 제공함으로써 자발적인 추론 능력을 촉진합니다. 이 모델은 단지 600개의 샘플과 2B의 모델 매개변수로 제한된 데이터에서도 효과적으로 훈련됩니다.

- **Performance Highlights**: 훈련 결과, MedVLM-R1은 MRI, CT 및 X-ray 벤치마크에서 정확도가 55.11%에서 78.22%로 증가했습니다. 또한, MedVLM-R1은 기존의 대형 모델들보다 뛰어난 일반화 능력을 보여주었으며, 현업에서의 적용 가능성을 높였습니다. 일반화는 특히 unfamiliar data에 대해 강력한 성능을 입증하며, 신뢰할 수 있는 임상 AI로 나아가는 중요한 발걸음을 제공합니다.



### 3D Nephrographic Image Synthesis in CT Urography with the Diffusion Model and Swin Transformer (https://arxiv.org/abs/2502.19623)
Comments:
          15 pages, 6 figures, 3 tables

- **What's New**: 이번 연구는 CT urography (CTU) 검사를 위한 3D nephrographic phase 이미지를 합성하는 방법을 개발하고 검증하는 것을 목표로 하고 있습니다. Swin Transformer 기반의 딥러닝 접근법과 diffusion model을 통합하여, 새로운 합성 모델인 dsSNICT를 제안하고 있습니다.

- **Technical Details**: 연구는 327명의 환자로 구성된 데이터를 사용하였으며, 각 환자의 세 단계 이미지는 affine registration 알고리즘을 통해 정렬되었습니다. dsSNICT 모델은 합성 nephrographic 이미지를 생성하기 위해 설계되었으며, 성능 평가는 PSNR, SSIM, MAE, FVD 지표를 사용하여 수행되었습니다.

- **Performance Highlights**: 제안된 접근 방식으로 생성된 합성 nephrographic 이미지는 PSNR 26.3 dB, SSIM 0.84, MAE 12.74 HU, FVD 1323이라는 높은 성능 지표를 달성하였습니다. 방사선 전문의 두 명의 평가에서는 Likert 척도에서 실제 이미지에 평균 3.5, 합성 이미지에 평균 3.4의 점수를 기록하였으며, 이는 높은 유사성을 나타내고 있습니다.



### Weaker LLMs' Opinions Also Matter: Mixture of Opinions Enhances LLM's Mathematical Reasoning (https://arxiv.org/abs/2502.19622)
Comments:
          12 pages, 1 figure, 3 tables, 4 prompt/data templates

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 수학에서의 형식적 추론 능력에 대한 관심이 높아졌습니다. 본 논문에서는 공정성 문제를 해결하기 위해 Mixture of Opinions (MoO)라는 새로운 접근 방식을 제안하여, 강력한 LLM이 약한 보조 LLM의 다양한 의견을 활용할 수 있도록 함으로써 추론 성능을 향상시키는 방법을 다룹니다. 이를 통해 결론적으로 다양한 관점이 추론 작업에서 긍정적인 영향을 미친다는 사실을 확인했습니다.

- **Technical Details**: MoO 프레임워크는 세 가지 단계로 구성되어 있습니다: (1) MoO 데이터셋 수집, (2) MoO 데이터셋으로의 사후 훈련, (3) 사후 훈련된 모델로의 추론. 특히, 주 모델(주 LLM)은 강력한 모델이며, 보조 LLM들은 비교적 약하지만, 유사한 추론 능력을 가지고 구체적인 답변을 생성할 수 있는 역할을 합니다. 이를 통해 다양한 의견을 통합하여 올바른 답변을 생성하는 능력을 강화합니다.

- **Performance Highlights**: 수행 성능을 평가하기 위해 수학적 추론 벤치마크인 GSM8K, AQuA-RAT, MATH를 사용했습니다. MoO 기법은 기존 방법들에 비해 평균적으로 5%의 성능 향상을 보여주었으며, 약한 LLM의 의견을 통합함으로써 추론 정확성이 유의미하게 개선될 수 있음을 입증했습니다. 이러한 결과는 여러 모델의 다양한 관점을 활용하면서 경량 모델에서도 결과적으로 성능 향상을 이룰 수 있음을 나타냅니다.



### Is Your Paper Being Reviewed by an LLM? A New Benchmark Dataset and Approach for Detecting AI Text in Peer Review (https://arxiv.org/abs/2502.19614)
- **What's New**: 이번 연구는 동료 심사(peer review) 과정의 무결성을 확보하기 위한 새로운 데이터 세트를 소개하고 있습니다. 이 세트는 AI가 생성한 788,984개의 동료 리뷰와 해당 인간 리뷰를 결합하여 제공합니다. 이는 ICLR와 NeurIPS 같은 두 주요 AI 연구 회의에 8년 간 제출된 논문의 리뷰를 포괄하고 있어, AI 기반 텍스트 검출 방법의 혁신적인 평가 자원을 제공합니다.

- **Technical Details**: 연구진은 18개의 기존 AI 텍스트 검출 알고리즘의 성능을 평가하여, LLM이 생성한 리뷰와 인간 작성 리뷰 간의 구분을 테스트하였습니다. 또한, 새로운 검출 방법을 제안하여 기존 방식보다 우수한 성능을 보였으며, 주로 의미적 유사성을 사용하여 기존 LLM 리뷰와 비교하는 방법을 사용하였습니다. 이를 통해 LLM이 생성한 리뷰의 위험성을 강조하며 이러한 비윤리적 사용에 대한 탐지 도구의 필요성을 제기하고 있습니다.

- **Performance Highlights**: 연구 결과, 기존의 대부분 AI 텍스트 검출 알고리즘은 AI가 생성한 리뷰를 신뢰성 있게 감지하는 데 한계가 있음을 보여주었습니다. 새로운 접근 방식은 기존 방식보다 성능이 뛰어나 GPT-4o 및 Claude로 작성된 리뷰의 탐지에 효과적이었습니다. 이 연구는 LLM이 작성한 텍스트 감지의 어려움을 시사하며, 이에 대한 추가적인 연구 필요성을 강조합니다.



### Improving Representation Learning of Complex Critical Care Data with ICU-BER (https://arxiv.org/abs/2502.19593)
Comments:
          Accepted for poster at GenAI4Health Workshop at AAAI 2025

- **What's New**: 이번 논문에서 제안하는 ICU-BERT는 복잡한 ICU 데이터를 처리하기 위해 설계된 변환기(Transformer) 기반 모델이며, 대규모 MIMIC-IV 데이터베이스에서 사전 훈련되었습니다. 기존의 AI 기반 의사결정 지원 시스템의 한계를 극복하기 위해 최소한의 전처리로 강력한 표현을 학습합니다. ICU-BERT는 생물의학 대형 언어 모델(BioBERT)에서 밀집 임베딩을 활용하여 다변량 데이터를 일반화할 수 있는 대표성을 제공합니다.

- **Technical Details**: ICU-BERT는 각 의료 기록 항목을 하나의 토큰으로 처리하며, 멀티토큰 입력 전략을 사용하여 의료 데이터 흐름의 복잡성을 효과적으로 캡처합니다. 이 모델은 의료 변수를 분석하기 위한 새로운 마스킹 작업과 멀티태스크 학습 손실을 도입하였으며, 이는 임상 변수를 효과적으로 재구성할 수 있도록 돕습니다. ICU-BERT는 MIMIC-IV 데이터베이스 전반에 걸쳐 사전 훈련되며, 이로 인해 복잡한 ICU 데이터의 해석을 개선하고 다양한 실제 임상 응용에 잘 적응할 수 있게 됩니다.

- **Performance Highlights**: 예비 평가 실험에서 ICU-BERT는 실세계에서의 다양한 도전 과제를 통해 강력한 일반화 성능을 보여주었습니다. 듀엣(DuETT) 및 YAIB 프레임워크에서 다섯 가지 과제를 통해 미세 조정(fine-tuning)되었으며, 기존 모델들과 비교했을 때 뛰어난 성능을 기록했습니다. ICU-BERT는 의사결정 지원 시스템의 혁신적 변화를 이끌어낼 가능성을 제시하며, 여러 임상 작업에서 성능 기준을 초과할 수도 있습니다.



### NeoBERT: A Next-Generation BER (https://arxiv.org/abs/2502.19587)
Comments:
          19 pages, 5 figures, 9 tables. Submitted to TMLR

- **What's New**: NeoBERT는 현대 아키텍처와 데이터, 최적화된 사전 훈련 방법론을 통합하여 양방향 모델의 능력을 재정의한 차세대 인코더입니다. 이 모델은 기존의 BERT 및 RoBERTa와 같은 인코더들이 겪고 있는 발전의 정체를 극복하고, 더 나아가 Retrieval-augmented generation과 같은 다양한 다운스트림 NLP 작업에 필수적입니다. NeoBERT는 기존 모델에 손쉽게 통합할 수 있도록 설계되었으며, 4,096 토큰의 확장된 컨텍스트 길이를 활용합니다.

- **Technical Details**: NeoBERT는 자가 감독식(pre-training) 및 세미-감독식(fine-tuning) 학습을 포함한 두 단계의 훈련 프로세스를 통해 최대 컨텍스트 윈도우를 4,096으로 증가시킵니다. 이러한 훈련 방법론은 물론 250M 파라미터의 크기에 불과하지만, MTEB 벤치마크에서 성능을 뛰어넘었으며, GLUE에서의 실험적 검증도 이루어졌습니다. 이 모델은 대량의 토큰(2조 이상)을 학습하여 일반화 능력을 극대화하고 다운스트림 성능을 향상시킵니다.

- **Performance Highlights**: NeoBERT는 MTEB에서 BERT large, RoBERTa large, NomicBERT, ModernBERT를 능가하는 성능을 보여주었습니다. 특히, 4,096 토큰의 컨텍스트 길이를 활용함으로써 RoBERTa보다 8배 긴 시퀀스를 처리할 수 있어 뛰어난 인퍼런스 속도를 자랑합니다. 오픈 소스로 제공되는 NeoBERT는 코드, 데이터, 체크포인트를 공개하여 연구 및 실제 활용을 가속화하고 있습니다.



### Tell me why: Visual foundation models as self-explainable classifiers (https://arxiv.org/abs/2502.19577)
- **What's New**: 본 연구에서는 시각 기초 모델(Visual Foundation Models, VFM)과 새로운 프로토타입 아키텍처(prototypical architecture)를 결합하여 해석 가능한 분류기를 제안합니다. 이 모델은 예측을 해석 가능한 개념들의 가중합으로 분해하여 자가 설명(self-explainable)을 목표로 합니다. 이와 같은 접근법이 기존 모델보다 더 효율적이고 해석 가능하다는 점이 특히 주목할 만합니다.

- **Technical Details**: ProtoFM이라는 방법론은 고정된 VFM 위에 가벼운 헤드(약 1M 파라미터)를 훈련시키는 방식을 채택합니다. 전문화된 훈련 목표(specialized training objectives)를 통해 해석 가능성을 증대시키고, 예측의 신뢰성을 확보하는 데 중점을 두고 있습니다. 이 모델은 기존의 VFM과 비교해 훨씬 적은 파라미터를 사용하면서도 효과성을 유지합니다.

- **Performance Highlights**: 평가 결과에 따르면, ProtoFM은 경쟁력 있는 분류 성능(classification performance)을 달성하며 해석 가능성 메트릭(interpretabiliy metrics)에서도 기존 모델들을 초월했습니다. 연구에 사용된 해석 가능성 관련 지표는 문헌에서 파생된 것입니다. 코드도 제공되어 있어 연구자들이 쉽게 활용할 수 있습니다.



### Do Large Language Models Know How Much They Know? (https://arxiv.org/abs/2502.19573)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 특정 주제에 대한 자신의 지식의 범위를 인식하는 능력을 평가하는 새로운 벤치마크를 개발하였다. 연구 결과, 모든 테스트한 LLM은 충분한 규모로 주어졌을 때, 특정 주제에 대한 자신의 지식을 얼마나 알고 있는지를 이해할 수 있는 것으로 나타났다. 이는 LLM의 지식 인식이 일반적인 속성일 가능성을 제시하며, 더 깊은 메커니즘 이해를 위한 추가 연구가 필요하다는 점도 강조되었다.

- **Technical Details**: 연구에서는 LLM이 가상의 개별 다이어리 항목으로 미세 조정되며, 특정 개인의 모든 다이어리 항목을 연대순으로 회상하는 능력을 평가한다. 모델이 올바른 양의 정보를 기억할 수 있다면, 이는 모델이 해당 주제에 대한 자신의 지식 범위를 이해하고 있다는 것을 시사한다. 또한 모델의 기억 능력이 단기 및 장기 문서 모두에서 효과적임을 보여주어, 특정 아키텍처의 특성이 성능에 미치는 영향을 탐구하였다.

- **Performance Highlights**: 시험된 모든 LLM은 적절한 데이터 크기에 따라 주제에 대한 지식을 적절히 기억하는 능력을 보여주었다. 그러나 데이터 스케일이 부족할 경우, 일부 모델은 무작위로 다이어리 항목을 회상하며 과소 또는 과대 기억하는 경향을 보였다. 연구는 이 특성의 출현 차이에 기여하는 잠재적 요인들에 대해서도 논의하여 LLM의 내재적 메커니즘에 대한 이해를 넓힌다.



### Atlas: A Framework for ML Lifecycle Provenance & Transparency (https://arxiv.org/abs/2502.19567)
- **What's New**: Atlas는 머신러닝(ML) 생애주기의 보안과 투명성을 강화하기 위한 프레임워크입니다. 이 프레임워크는 데이터 및 소프트웨어 공급망의 출처를 추적하여 모델 아티팩트의 진위 및 메타데이터의 무결성을 확인합니다. Atlas는 신뢰할 수 있는 하드웨어와 투명성 로그를 결합하여 데이터 기밀을 유지하고 ML 파이프라인 운영에서의 무단 접근을 제한합니다. 이 논문에서는 Atlas의 두 가지 사례 연구를 통해 실제 적용 가능성도 평가합니다.

- **Technical Details**: Atlas는 디지털 서명 및 블록체인과 같은 암호화 기술을 사용하여 ML 모델의 출처 및 무결성을 증명합니다. 이 프레임워크는 신뢰할 수 있는 실행 환경(TEE)을 활용하여 모델 생애주기 전반에 걸쳐 메타데이터 수집을 지원합니다. 또한, Atlas는 C2PA와 같은 기존 기술을 통합하여 모델 Artifact의 출처를 정확하게 추적할 수 있도록 설계되었습니다. 이러한 통합은 Atlas가 ML 생애주기 및 공급망의 보안성을 높이는 데 기여합니다.

- **Performance Highlights**: Atlas의 프로토타입은 BERT 모델과 bge-reranker 모델의 미세 조정을 통해 평가되었습니다. 이 사례 연구는 Atlas가 제공하는 모델 생애주기 투명성의 실용성을 보여주며, ML 모델의 생성과 배포에서의 보안 및 신뢰성을 향상시키는 데 중요한 역할을 합니다. 또한, Atlas는 뛰어난 메타데이터 무결성을 통해 ML 애플리케이션의 안정성과 신뢰성을 증대시키는 데 기여할 것으로 기대됩니다.



### Distill Not Only Data but Also Rewards: Can Smaller Language Models Surpass Larger Ones? (https://arxiv.org/abs/2502.19557)
Comments:
          14 pages, 7 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 지식 증류(Knowledge Distillation) 과정에서 응답과 보상 신호를 동시에 전달하는 새로운 방법론을 제안합니다. 기존의 감독식 미세 조정(Supervised Fine-Tuning, SFT) 방식은 주로 응답을 통한 데이터를 증류하는 데 중점을 두었지만, 이 연구는 생성을 넘어 보상의 질을 반영할 수 있는 가능성도 탐색합니다. 새로운 파이프라인을 통해, 보상을 자기 감독(self-supervised) 기법으로 생성하여 외부 평가에 대한 의존도를 줄입니다.

- **Technical Details**: 제안된 방법은 LLM 모델의 응답 데이터 안에서 내재된 구조를 활용하여 '유사 보상(pseudo-rewards)'을 생성합니다. 이 과정은 교사 모델(Teacher Model)과 학생 모델(Student Model) 간의 응답의 품질을 비교하여 이루어지며, 이로 인해 보상 모델은 고품질 출력을 우선시하는 방법을 학습하게 됩니다. 이 모델은 초기 SFT 단계 이후 강화 학습(Reinforcement Learning, RL)을 통해 학생 모델의 성능을 지속적으로 개선합니다.

- **Performance Highlights**: GSM8K 및 MMLU-PRO 데이터셋에서 실시된 실험 결과, 제안된 방법이 전통적인 SFT 기반 방식을 일관되게 초월하는 것으로 나타났습니다. 학생 모델은 특정 상황에서 교사 모델보다 우수한 성능을 보였으며, 이는 효율적이고 확장 가능한 증류 과정을 통해 실현되었습니다. 전체적으로, 이 연구는 보상 학습을 통한 지속적인 모델 개선이 가능함을 보여줍니다.



### Winning Big with Small Models: Knowledge Distillation vs. Self-Training for Reducing Hallucination in QA Agents (https://arxiv.org/abs/2502.19545)
- **What's New**: 이 연구에서는 고객 지원에 대한 대규모 언어 모델(LLMs)의 배치에서 발생하는 hallucination(허위 정보의 생성) 문제와 비경제적인 독점 모델의 비용 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, Samsung Smart TV 사용자 매뉴얼에 대한 질의 응답 데이터 셋을 사용하여 합성 데이터(synthetic data)가 군중 소싱 데이터(crowdsourced data)보다 더 낮은 hallucination 비율을 보여줄 수 있음을 입증했습니다. 또한, self-training(자기 훈련)과 지식 증류(knowledge distillation)를 비교하면서, 두 방법이 유사한 수준의 hallucination 감소를 보임을 발견했습니다.

- **Technical Details**: 연구에서는 retrieval-augmented question-answering (QA) 파이프라인을 개발하고, Llama-3-8B-Instruct 모델을 사용하여 crowdsourced 질문에 대한 답변을 생성합니다. 이와 함께, 수작업(cleaning) 및 자동화된 방법을 통해 데이터 정제(data cleaning) 성능을 비교했습니다. 결과적으로 LLM이 생성한 합성 데이터는 더 낮은 hallucination 비율을 기록했으며, self-training 방식이 knowledge distillation 보다 유사한 효과를 나타내는 흥미로운 발견을 하였습니다. 또한, 무응답 질문에 대한 "모르겠습니다"라는 맥락화된 응답(contextualized responses)을 통해 모델의 견고성(robustness)을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 수작업 및 자동 데이터 정제 방법은 유사한 사실적 정확도를 보였지만, 자동 정제를 통한 모델의 응답이 더 길었습니다. LLM이 생성한 합성 교육 데이터는 군중 소싱 데이터보다 낮은 hallucination 비율을 기록하였고, self-training을 통해 Llama-3 모델이 생성한 데이터에 대한 모델 성능이 GPT-4o의 데이터에 대한 훈련과 유사함을 입증했습니다. 이러한 발견은 self-training이 hallucination을 최소화하는 데 있어 리소스를 효율적으로 사용할 수 있는 대안임을 보여줍니다.



### No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data (https://arxiv.org/abs/2502.19537)
- **What's New**: 이 논문에서는 OpenAI와 Google 같은 언어 모델 (LM) 제공자들이 사용자 맞춤형 데이터로 LM을 훈련할 수 있도록 하는 fine-tuning API를 소개합니다. 이러한 API는 새로운 맞춤화 기회를 제공하지만, 모델의 안전성을 해칠 수 있는 취약점도 노출합니다. 논문에서는 harmfulness filter를 사용하여 유해한 훈련 데이터를 차단하는 방법과 이를 우회하기 위한 다양한 공격 방법을 설명합니다.

- **Technical Details**: 연구에서는 LM의 초기 응답 토큰에서 모델의 거부를 제거하여 공격이 이루어진다는 점을 강조합니다. 피해자의 훈련 데이터가 harmless하더라도, 공격자는 이를 이용해 위험한 LM을 만들 수 있습니다. 새로운 공격 기법 NOICE은 모델이 안전성을 바탕으로 악의적인 요청에 거부 반응을 하도록 훈련한 후, 요청을 이행하게 하는 메커니즘을 사용합니다.

- **Performance Highlights**: NOICE 공격은 GPT-4o 모델에 대해 57%의 공격 성공률 (ASR)을 기록하였고, OpenAI의 Bug Bounty를 수상했습니다. 또한, 간단한 방어 기법을 통해 기존 공격의 성공률을 37-79%에서 최대한 낮출 수 있음을 보여줍니다. 이 연구는 harmless 데이터를 사용한 공격이 어떻게 모델을 해칠 수 있는지를 잘 보여줍니다.



### Retrieval Augmented Anomaly Detection (RAAD): Nimble Model Adjustment Without Retraining (https://arxiv.org/abs/2502.19534)
Comments:
          6 pages, 3 figures. 2 tables, accepted at ISDFS 2025

- **What's New**: 이 논문에서는 실시간 피드백 메커니즘을 통해 이상 탐지 모델의 잘못된 긍정 반응(false positive)을 줄이는 방법을 제안합니다. 제안된 기술은 행동 네트워크 이상 탐지 모델의 경량 배포를 위해 설계되었습니다. 이 방법론은 높은 처리량(throughput)을 요구하는 유사 도메인에 쉽게 통합될 수 있습니다.

- **Technical Details**: 논문에서는 Retrieval Augmented Anomaly Detection (RAAD)이라는 새로운 방법론을 소개합니다. 이 방법은 사용자가 직접 피드백을 제공하여 모델의 출력을 조정할 수 있는 기능을 제공합니다. RAAD는 실시간으로 피드백을 적용함으로써 예측 결과를 개선하고, 이후에 재교육을 위한 데이터 수집 과정을 단순화할 수 있는 장점이 있습니다.

- **Performance Highlights**: RAAD는 이미지, 텍스트, 그래프 기반 데이터 등 여러 데이터 양식에서 성능을 평가하여 그 일반화 가능성을 검증했습니다. 다양한 모델 아키텍처와 다중 데이터 모달리티에서 벤치마킹을 통해 이 방법이 효과적임을 입증했습니다. 이를 통해 기존의 딥러닝 기반 이상 탐지 시스템 보다 낮은 잘못된 긍정률을 유지할 수 있음을 강조하였습니다.



### Cognitive networks highlight differences and similarities in the STEM mindsets of human and LLM-simulated trainees, experts and academics (https://arxiv.org/abs/2502.19529)
Comments:
          Keywords: cognitive network science; mindset measurement; associative knowledge; artificial intelligence; simulated participants

- **What's New**: 이 연구는 STEM(과학, 기술, 공학, 수학)에 대한 태도를 정량화하여 인간과 GPT-3.5와 같은 대형 언어 모델이 이러한 주제를 어떻게 개념화하는지를 조사합니다. 177명의 인간 참가자와 177명의 인공 인간의 행동 심리적 형태 네트워크(Behavioral Forma Mentis Networks, BFMNs)를 사용하여, 그들의 마인드셋 차이를 비교했으며, 이는 전문성 수준이 마인드셋에 미치는 영향을 알리는 데 기여합니다.

- **Technical Details**: 참가자들은 교육생, 전문가, 학자로 나뉘어 마인드셋을 분석했습니다. 연구의 결과는 인간의 형태 네트워크가 GPT-3.5보다 훨씬 높은 군집 계수를 보였으며, 이는 인간의 마인드셋이 STEM 아이디어에 대한 개념 연관성을 형성하고 닫는 경향이 있음을 나타냅니다. 특히, 인간 전문가들은 STEM 개념의 인지 네트워크 통합이 더 뛰어남을 보여주었습니다.

- **Performance Highlights**: 이 연구에서 밝혀진 바와 같이, 인간과 GPT의 마인드셋 모두 수학을 중립적이거나 긍정적으로 프레이밍했으나, STEM 고등학교 학생 및 다른 많은 대형 언어 모델과는 차별화된 모습을 보였습니다. STEM 아이디어에 대한 접근 방식의 차이는 기억 구조와 머신의 한계를 이해하는 데 도움을 주는 통찰을 제공한다고 할 수 있습니다.



### Accessing LLMs for Front-end Software Architecture Knowledg (https://arxiv.org/abs/2502.19518)
Comments:
          4 pages, 1 figure, to appear in the International Workshop on Designing Software at ICSE 2025

- **What's New**: 이 논문은 큰 언어 모델(LLMs)이 소프트웨어 설계 작업에서의 능력을 파악하는 데 중점을 두고 있습니다. 특히 VIPER 아키텍처를 이해하고 생성할 수 있는 LLM의 가능성을 탐구하며, Bloom의 분류법을 활용하여 LLM의 성능을 평가하기 위한 종합적인 프레임워크를 제안합니다. 결과적으로, LLM은 평가 및 생성과 같은 고차원 작업에서는 뛰어난 성과를 보였으나, 건축 세부 사항의 정확한 검색이 필요한 저차원 작업에서는 어려움을 겪었습니다.

- **Technical Details**: VIPER 아키텍처는 iOS 애플리케이션을 위한 복잡한 설계 패턴으로, View, Interactor, Presenter, Entity, Router의 다섯 가지 주요 구성 요소를 포함합니다. 연구에서는 VIPER 아키텍처를 사용하여 LLM이 기본적인 프론트엔드 아키텍처 패턴을 어느 정도 이해하고 있는지를 분석합니다. 결과는 LLM의 답변을 비교하고 Bloom의 분류법에 따라 성과를 평가하는 방법을 포함합니다.

- **Performance Highlights**: 실험 결과에 따르면, LLM은 VIPER의 원칙을 준수하는 설계를 제안하는 데 있어 경험이 다양한 인간 개발자와 비교했을 때 상당한 차이를 보였습니다. LLM은 또한 아키텍처 설계에서 누락된 구성 요소를 유추하고 잠재적 의존성을 예측하는 데 어려움을 겪는 것으로 나타났습니다. 이러한 결과는 개발 비용 절감 등의 잠재력과 함께 실제 소프트웨어 설계 시나리오에서 LLM의 효과적인 적용에 대한 장벽을 강조합니다.



### Mixtraining: A Better Trade-Off Between Compute and Performanc (https://arxiv.org/abs/2502.19513)
- **What's New**: 이 논문에서는 자가 지도 학습(self-supervised learning, SSL)과 감독 학습(supervised learning, SL) 사이에 새로운 훈련 프레임워크인 MixTraining을 제안합니다. MixTraining은 두 개의 학습 목표 간의 매끄러운 전환을 특징으로 하며, SSL과 SL의 여러 에포크를 통합하여 훈련 효율성을 높입니다. 이 접근 방식은 데이터가 제한된 환경에서도 높은 정확도를 달성할 수 있도록 지원하며, 성능과 계산량 간의 균형을 개선합니다.

- **Technical Details**: MixTraining 프레임워크는 일반적으로 사용되는 SSL+SL 프레임워크의 두 단계인 자가 지도 학습 단계와 감독 학습 단계를 병합하여 새로운 mixtraining 단계를 생성합니다. 이 mixtraining 단계는 자가 지도 및 감독 목표의 공동 업데이트를 가능하게 하여 두 목표 간의 균형을 맞추게 합니다. 이 방식은 두 학습 목표 간의 최적화를 더 밀접하게 연결하여, 모델이 안정적으로 목표 작업에 적응할 수 있게 합니다.

- **Performance Highlights**: 실험 결과 MixTraining은 전통적인 SSL+SL 파이프라인에 비해 현저한 성능 향상을 보여줍니다. TinyImageNet 데이터셋에서 8.81%의 절대 정확도 증가와 18.89%의 상대 정확도 증가를 달성하며, 훈련 속도는 최대 1.29배 향상됩니다. 데이터가 제한될 경우 MixTraining은 10% 제한 수준에서 105.58%의 상대 정확도 증가를 달성하여, 더욱 두드러진 성과를 보여줍니다.



### Do LLMs exhibit demographic parity in responses to queries about Human Rights? (https://arxiv.org/abs/2502.19463)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에서의 헤징(hedging) 행위를 인간의 권리를 토대로 평가하는 새로운 접근 방식이 소개됩니다. 특히, 모든 인간이 인간의 권리를 가짐을 명시하는 '세계 인권 선언'(UDHR)과의 관계에서 이러한 행위를 분석합니다. 이는 다양한 집단 간 비교를 통해, 모델이 특정 그룹에 대한 인권을 어떻게 인식하고 있는지 평가하는 첫 번째 체계적인 시도입니다.

- **Technical Details**: 연구팀은 서로 다른 국가 및 사회적 정체성을 토대로 인권에 관한 질의를 포함하는 새로운 프롬프트 세트를 설계하였습니다. LLM의 헤징 및 비확인(non-affirmation) 행위를 포착하기 위한 새로운 메트릭스를 개발하였고, 이를 통해 특정 그룹 간의 인권 인식의 차별성을 측정했습니다. Gemini 1.5, Claude 3 Sonnet, GPT-4o 등 세 가지 대표적인 LLM의 결과를 보고하였고, 모두 집단 간의 인권 차별을 보였습니다.

- **Performance Highlights**: 모델 간에 서로 다른 정체성 그룹에 대한 인권 불균형의 패턴이 높은 상관관계를 보였으며, 같은 정체성 그룹들이 서로 다른 모델에서도 유사한 차별을 경험했습니다. 헤징 및 비확인 비율은 질문의 모호성에 따라 다르지만, 질문의 정확한 문구 변형에 대해서도 일관성을 유지하고 있습니다. 연구 결과는 LLM이 모든 그룹의 인권을 동등하게 지지하도록 조정할 필요성을 강조합니다.



### Practical Evaluation of Copula-based Survival Metrics: Beyond the Independent Censoring Assumption (https://arxiv.org/abs/2502.19460)
- **What's New**: 전통적인 생존 메트릭스는 독립적인 censoring 가정을 요구하지만, 사건과 관련된 이유로 censoring되는 경우에는 이 가정이 더 이상 유지되지 않습니다. 본 논문에서는 의존성 있는 censoring의 존재하에 생존 모델을 평가할 수 있는 세 가지 copula 기반 메트릭스를 제안하고, 이러한 메트릭스를 평가하기 위해 현실적이고 반합성적인 데이터를 생성하는 프레임워크를 설계합니다. 이러한 접근법은 생존 예측의 정확성을 높이는데 기여할 수 있습니다.

- **Technical Details**: 생존 예측 모델은 라벨이 있는 데이터셋으로부터 인스턴스의 설명을 실제 값으로 매핑하는 모델을 학습하는 것을 목표로 합니다. 하지만 일부 훈련 인스턴스가 'censored' 상태이므로, 생존 예측 모델은 이벤트와 censoring 분포 간의 의존성을 모델링하는 copula 함수를 사용해야 합니다. 이 방식은 특히 잃어버린 후속조사의 개념을 포함하여, 예측 성능에 대한 기존 생존 메트릭스의 편향을 극복하는 데 효과적입니다.

- **Performance Highlights**: 이 논문의 실험적 분석은 생성된 데이터셋과 반합성 데이터셋 내에서 우리의 메트릭스가 정확한 오류 추정치를 제공하며, 특히 예측 정확도 측면에서 개선된 결과를 보여줍니다. 기존 메트릭스보다 더 정확하게 생존 확률을 추정할 수 있는 가능성을 제시하며, 의존성 있는 censoring이 존재하는 상황에서의 생존 모델 평가에 새로운 기준을 마련합니다.



### Multispectral to Hyperspectral using Pretrained Foundational mod (https://arxiv.org/abs/2502.19451)
- **What's New**: 이번 연구에서는 다중 스펙트럼 데이터(multi-spectral data)에서 고스펙트럼 데이터(hyperspectral data)를 복원하기 위한 스펙트럴 및 공간-스펙트럴 트랜스포머 모델을 제안합니다. 이 모델은 EnMAP 및 EMIT 데이터셋을 기반으로 사전 훈련(pretraining)되었으며, 시간 및 공간적으로 정렬된 이미지 쌍에서 세부 조정(fine-tuning)을 거쳤습니다. 하이퍼스펙트럼 및 멀티스펙트럼 이미징 시스템의 장점을 결합하여 대기 모니터링을 향상시킬 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 하이퍼스펙트럼 이미징은 수백 개의 좁은 밴드를 캡처하여 보다 세분화된 스펙트럼 정보를 제공하며, 고정밀 응용 프로그램에 적합합니다. 반면에 다중 스펙트럼 이미징은 더 넓은 공간적 및 시간적 범위를 제공하지만, 정밀한 온실가스 감지에는 부족한 스펙트럼 해상도가 문제입니다. 본 연구는 ViT 모델을 기반으로 한 두 가지 수정된 트랜스포머를 설계하여 스펙트럼 및 공간-스펙트럴 관계를 캡처하고자 합니다.

- **Performance Highlights**: 제안된 모델은 하모나이즈드 랜드샛 센티널-2(HLS-S30) 데이터 및 센티널-2(S2) 이미지를 사용하여 하이퍼스펙트럼 밴드를 복원하는 실험을 통해 검증되었습니다. 이러한 데이터 세트는 모델의 훈련 및 테스트를 위해 균형 잡힌 샘플 수를 가지며, 85.32%의 정렬율을 달성하는 성과를 보여주었습니다. 이러한 결과는 정확한 하이퍼스펙트럼 데이터 재구성의 가능성을 높이며, 응용 프로그램의 다양성을 확장합니다.



### Multi-objective Cat Swarm Optimization Algorithm based on a Grid System (https://arxiv.org/abs/2502.19439)
- **What's New**: 이 논문에서는 Cat Swarm Optimization Algorithm (CSO)의 다중 목표 버전인 Grid-based Multi-objective Cat Swarm Optimization Algorithm (GMOCSO)을 제안합니다. GMOCSO는 현대 다중 목표 알고리즘이 추구하는 수렴(convergence)과 다양성 보존(diversity preservation)을 목표로 합니다. 이를 위해 기존 CSO 알고리즘의 룰렛 휠 방법을 탐욕적(greedy) 방법으로 교체하고, Pareto Archived Evolution Strategy Algorithm (PAES)의 두 가지 주요 개념인 격자(grid) 시스템과 이중 아카이브(double archive) 전략을 채택했습니다.

- **Technical Details**: GMOCSO는 테스트 함수들(test functions)과 실제 시나리오인 압력 용기 설계 문제(Pressure vessel design problem)를 사용하여 성능을 평가합니다. 실험에서는 Reversed Generational Distance, Spacing metric, Spread metric과 같은 다양한 메트릭을 사용하여 제안된 알고리즘을 다른 잘 알려진 알고리즘과 비교하였습니다. 이러한 접근 방식은 알고리즘의 신뢰성을 높이는 데 기여하며, 다양한 환경에서 성능을 안정적으로 유지하도록 돕습니다.

- **Performance Highlights**: 제안된 GMOCSO 알고리즘의 최적화 결과는 그 강 robustness을 잘 보여주고 있으며, 통계적 방법과 그래프를 통해 결과가 더욱 확인되었습니다. 실험 결과는 GMOCSO가 기존 알고리즘들에 비해 우수한 성능을 발휘하고 있음을 나타내며, 미래 연구 방향 및 결론도 논문에서 논의되었습니다.



### Evolutionary Algorithms Approach For Search Based On Semantic Document Similarity (https://arxiv.org/abs/2502.19437)
- **What's New**: 이 논문에서는 클라우드 컴퓨팅과 분산 컴퓨팅의 발전이 신경망(Neural Networks)과 유전자 알고리즘(Genetic Algorithm), 차별 진화 알고리즘(Differential Evolution Algorithm) 같은 진화 알고리즘의 연구에 기여했음을 설명합니다. 특히, Universal Sentence Encoder (USE)를 통해 텍스트의 의미적 유사성을 포착하고, 이를 바탕으로 사용자 쿼리와 관련된 상위 N개의 문서를 검색하는 접근 방식을 제안하고 있습니다. 기존의 전통적인 접근 방식과 비교하여, 진화 알고리즘이 상위 N개 결과를 찾는 데 더욱 우수하다는 실험 결과를 통해 검증했습니다.

- **Technical Details**: 진화 알고리즘은 최적화 기술을 적용하는 컴퓨터 프로그램으로, 생물학적 진화 과정과 유사한 방식으로 작동합니다. 이 연구에서는 Universal Sentence Encoder (USE)가 생성한 문장 임베딩을 이용하여 주어진 사용자 질문에 대한 답변을 검색합니다. 나아가, Manhattan Distance, Genetic Algorithm(GA), Differential Evolution(DE) 알고리즘의 성능을 비교하여, 이러한 진화적 접근 방식이 높은 품질의 결과를 제공함을 강조합니다.

- **Performance Highlights**: 이 연구의 실험 결과는 Universal Sentence Encoder (USE)를 사용하여 문서의 의미적 유사성을 효과적으로 캡처함으로써 문서의 텍스트 표현을 문장 임베딩 벡터로 효율적으로 변환할 수 있음을 보여줍니다. 또한, 비교 실험을 통해 GA와 DE 알고리즘이 전통적인 순위 접근 방식보다 우수한 성능을 보임을 입증하여, 최상위 N개의 결과를 찾는 데 있어 진화 알고리즘의 우수성을 확인했습니다.



### Implementation of a Generative AI Assistant in K-12 Education: The CGScholar AI Helper Initiativ (https://arxiv.org/abs/2502.19422)
- **What's New**: 이 논문은 높은 학교 맥락에서 글쓰기 피드백을 제공하기 위한 Generative AI (GenAI) 보조 도구인 CGScholar AI Helper의 파일럿을 다룹니다. GenAI를 사용해 영어 언어 예술(ELA)과 역사 수업에서 학생들의 텍스트에 대한 형성적 및 총체적 피드백을 제공하는 것을 목표로 하였습니다.

- **Technical Details**: 이 연구는 미국 중서부의 두 개의 서로 다른 학교에서 11학년을 대상으로 진행된 시험을 포함합니다. 한 학교는 낮은 경제적 배경과 저조한 성과를 가진 곳이고, 다른 학교는 높은 경제적 배경과 높은 성과를 가진 곳입니다. 이 보조 도구는 참여 교사의 평가 기준에 따라 'prompt engineering'과 맞춤형 교육 자료로부터 대형 언어 모델(LLM)을 'fine-tuning'하는 두 가지 주요 메커니즘을 사용하였습니다.

- **Performance Highlights**: CGScholar AI Helper는 학생들의 글쓰기 능력을 향상시키고 ELA 및 기타 필요 서면 과제가 요구되는 교과목에서 교사를 지원하는 가능성에 중점을 두고 있습니다. 연구 결과는 학습 준비성을 향상시키기 위한 중요한 학습 단계에서 학생들의 성취도를 증대시킬 잠재력을 보여줍니다.



### InternVQA: Advancing Compressed Video Quality Assessment with Distilling Large Foundation Mod (https://arxiv.org/abs/2502.19026)
Comments:
          Accepted by ISCAS 2025(Lecture)

- **What's New**: 이 연구에서는 비디오 품질 평가(Video Quality Assessment, VQA)에 대한 InternVideo2 모델의 전이 가능성을 탐구하였습니다. InternVideo2는 대규모 매개변수와 다양한 멀티모달 데이터를 활용하여 비디오 이해 작업에서 뛰어난 성능을 발휘하고 있습니다. 이 모델의 용량이 클 경우 자원 소비가 과도해지는 문제를 해결하기 위해 경량화된 모델을 설계하기 위한 증류(distillation) 방법론을 제안했습니다. 실험 결과, 증류된 경량 모델이 기존 방법들과 비교하여 우수한 성능을 나타내었음을 보여줍니다.

- **Technical Details**: 연구자들은 내부 비디오 모델인 InternVideo2를 선택하여 비디오 품질 평가에 필요한 풍부한 영상을 처리할 능력을 강조하였습니다. 모델은 대량의 비디오 데이터를 처리하고 마스크 비디오 학습(masked video learning)을 통해 유용한 표현을 학습합니다. 증류 과정에서는 서로 유사한 구조를 가진 학생 모델과 교사 모델 간의 지식 전이를 통해, 학생 모델이 압축비디오 품질 평가에 필요한 독특한 특징을 효과적으로 학습하도록 설계되었습니다. 여기서 사용된 이중 손실 메커니즘은 학습 과정을 최적화하여 compression distortion에 대한 강인성을 제공합니다.

- **Performance Highlights**: 실험 결과, 특정 증류 방법을 사용하여 경량화된 모델이 기존의 비디오 품질 평가 방법을 초월하는 성능을 보였습니다. 두 가지 압축 품질 평가 데이터 집합에서도 기존 방법에 비해 우수한 성능을 기록하며, 원래의 대형 모델에 필적하거나 이를 초과하는 성능을 달성했습니다. 이 연구는 효율성과 성능 간의 최적의 균형을 이루는 경량 모델 설계를 통해 비디오 품질 평가 작업의 발전 가능성을 제시하고 있습니다.



New uploads on arXiv(cs.LG)

### R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts (https://arxiv.org/abs/2502.20395)
- **What's New**: 최근 논문에서는 비전 인코더를 혼합 전문 모델(Mixture-of-Experts, MoE)로 대체하여 대규모 다중 모드 모델의 성능을 개선하는 새로운 방법을 제안합니다. 이러한 접근 방식은 다양한 하위 작업에서 더 풍부하고 다양성 있는 표현을 가능하게 하여, 각 입력에 대해 개별 전문가의 표현을 재조정하여 성능을 극대화합니다. Test-time에 최적화 과정을 진행할 수 있는 'Re-Routing in Test-Time(R2-T2)' 방법을 도입하여, 기존 모델의 장점을 보존하면서도 성능 향상을 꾀하고 있습니다.

- **Technical Details**: R2-T2는 테스트 샘플 주변의 올바르게 예측된 샘플의 벡터와 유사하게 라우팅 가중치를 지역적으로 최적화하는 방법으로 구성됩니다. 세 가지 R2-T2 전략이 제안되어 서로 다른 최적화 목표와 이웃 검색 공간을 기반으로 구성되어 있습니다. 이러한 방식은 기존 라우터의 한계를 극복하고, 다양한 하위 작업에서의 성공적인 작업들과의 유사성을 통해 최적의 라우팅 가중치를 제공하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: R2-T2는 최근 다중 모드 MoE 모델을 여덟 개의 도전적인 벤치마크에서 평가하여, 기존 모델보다 월등한 성능을 보여주었습니다. 본 연구에서 제안된 방법은 테스트 시간 동안의 재라우팅(rerouting)이 도메인 특정 추론(domain-specific reasoning)을 향상시키고, 추가적인 훈련 없이도 모델의 일반화 및 견고성을 극대화하는 데 기여함을 보여주었습니다. 결국 이 방법은 기존 다중 모드 MoE의 성능을 상당히 개선하며, 대규모 LMM의 성능 간극을 해소하는 중요한 기여를 합니다.



### Walking the Web of Concept-Class Relationships in Incrementally Trained Interpretable Models (https://arxiv.org/abs/2502.20393)
Comments:
          8 pages of main text, 6 figures in main text, 11 pages of Appendix, published in AAAI 2025

- **What's New**: 이 논문에서는 개념 기반 모델(concept-based models)이 점진적인 학습(incremental learning) 환경에서도 잘 작동할 수 있도록 재설계되었음을 강조합니다. 특히, 새로 생성된 클래스가 이전 개념뿐만 아니라 새로운 개념을 사용할 수 있도록 하는 동적(dynamic) 설정에서 이 모델을 연구합니다. 이를 통해 개념과 클래스의 복잡한 관계 망을 유지하고 증대할 필요성을 보여줍니다.

- **Technical Details**: 새로운 툴인 MuCIL(Multimodal Concept-Based Incremental Learner)을 소개하며, 이 방법은 항상 새로운 클래스를 수용할 수 있는 능력을 지니고 있습니다. MuCIL은 텍스트 개념(text concepts)과 이미지 표현(image representations) 간의 임베딩(embedding)을 결합하여 다중 모달 개념 임베딩(multimodal concept embeddings)을 생성합니다. 이 구조는 분류(classification) 성능을 보장하고, 안정적인 해석을 제공하는 데 필요한 정보도 포함하고 있습니다.

- **Performance Highlights**: 실험을 통해 MuCIL이 기존의 개념 기반 모델보다 2배 이상 높은 분류 성능(classification performance)을 달성했음을 보여줍니다. 또한, 모델의 개념에 대한 개입(intervention) 능력을 연구하여 입력 이미지 내에서 시각적 개념을 국소화(localization)할 수 있음을 입증하였습니다. 이 접근방법은 사후 해석(post-hoc interpretations)을 제공할 수 있는 잠재적인 방법으로서, 다양한 벤치마크 데이터세트에서 최첨단의 성과를 달성하였습니다.



### Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis (https://arxiv.org/abs/2502.20383)
Comments:
          Project website: this http URL

- **What's New**: 최근 웹 AI 에이전트의 발전은 복잡한 웹 탐색 과제를 해결하는 데 놀라운 능력을 보여주었습니다. 그러나 이러한 에이전트들은 독립형 대형 언어 모델(LLMs)보다 더 큰 취약성을 나타내고 있으며, 이는 웹 AI 에이전트의 높은 유연성과 관련이 있습니다. 이 연구는 웹 AI 에이전트의 취약성을 증가시키는 여러 요인을 조사하고, 시스템 설계에서의 추가적인 안전성을 높일 수 있는 실질적인 통찰을 제공합니다.

- **Technical Details**: 웹 AI 에이전트는 LLM과 소프트웨어 도구, API를 통합하여 웹 환경 내에서 특정 목표를 달성하기 위한 일련의 작업을 수행합니다. 이 연구는 웹 AI 에이전트가 독립형 LLM과 비교하여 46.6%의 악의적인 명령을 실행할 가능성을 보인다는 사실을 보여줍니다. 웹 AI 에이전트의 높은 취약성은 사용자 목표를 시스템 프롬프트에 직접 삽입하고, 다단계 행동 생성을 하며, 관찰 능력을 강화하는 등 세 가지 주요 요소에서 기인합니다.

- **Performance Highlights**: 이 연구는 웹 AI 에이전트의 jailbreaking에 대한 높은 취약성을 수치적으로 비교합니다. 우리는 5단계의 세분화된 평가 메트릭을 도입하여 기존의 이진 평가 방식을 넘어서 웹 AI 에이전트의 취약성을 보다 심도 있게 분석합니다. 이 연구 결과를 통해 웹 AI 에이전트의 보안 위험을 완화하기 위한 방안과 설계 개선을 위한 권장 사항을 제시합니다.



### Multi-Turn Code Generation Through Single-Step Rewards (https://arxiv.org/abs/2502.20380)
Comments:
          9 pages (not including references or appendix); 6 figures (in main paper); (v1) preprint

- **What's New**: 이번 논문은 실행 피드백을 기반으로 한 다단계 코드 생성 문제를 다룹니다. 기존 방법론은 피드백 없이 코드를 생성하거나 복잡한 강화 학습(reinforcement learning)을 사용합니다. 우리는 단일 단계 보상만을 활용하여 문제를 해결하는 간단하면서도 확장 가능한 방법인 μCode를 제안합니다. 이 접근법은 코드 생성 프로세스를 효율적이고 안정적으로 만듭니다.

- **Technical Details**: μCode는 다단계 실행 피드백을 통해 코드 생성기를 훈련시키는 새로운 프레임워크입니다. 이 방법은 마르코프 결정 과정(Markov Decision Process, MDP)의 개념을 사용하여 각 상호작용에서 나온 중간 상태에서 올바른 코드를 단일 단계로 회복할 수 있음을 보여줍니다. 훈련 과정에서 생성기와 검증기를 동시에 개선하는 전문가 반복(expert iteration) 프레임워크를 사용합니다. 또한, 실시시간(inference time) 확장을 위해 학습된 검증기를 이용해 코드를 선택합니다.

- **Performance Highlights**: 실험 결과, 우리의 μCode 방식은 MBPP(Austin et al., 2021)와 HumanEval(Chen et al., 2021) 벤치마크에서 가장 유력한 다단계 접근법들을 초월한 성능 향상을 보였습니다. 학습된 검증기를 활용하여 더 나은 생성기 학습이 이루어짐을 증명하였고, 높은 추론 예산이 있는 경우에도 유망한 확장법 트렌드를 보여주었습니다.



### PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation (https://arxiv.org/abs/2502.20377)
- **What's New**: PhantomWiki는 고유하고 사실적으로 일관된 문서 집합체를 생성하는 파이프라인으로, 기존 데이터셋의 한계를 극복하고자 제안되었습니다. 매번 평가 시에 새로운 PhantomWiki 인스턴스를 생성함으로써 데이터 유출 및 성과 부풀림 문제를 피할 수 있습니다. 이 방식은 LLM(대형 언어 모델)의 추론(Reasoning)과 검색(Retrieval) 능력을 분리하여 평가할 수 있는 새로운 기준을 제공합니다.

- **Technical Details**: PhantomWiki는 질문 난이도와 코퍼스 크기를 조정하여 LLM의 추론과 검색 능력을 체계적으로 분석할 수 있게 설계되었습니다. 이는 특정 코퍼스의 맥락 창 안에 필요한 정보를 적재하고, 검색 방법을 통해 외부 정보를 접근해야 하는 복잡한 상황을 포함합니다. 이로써, LLM의 내부 지식 의존성을 평가하고, 다양한 기법을 사용할 수 있는 가능성을 모색합니다.

- **Performance Highlights**: PhantomWiki에서의 평가는 상태 최상의 LLM이 직면하는 도전을 잘 보여줍니다. 다양한 문서 코퍼스에 대한 질문을 처리할 때, F1 점수는 논리적이거나 기술적인 복잡성이 증가할수록 급락하는 경향을 보였습니다. PhantomWiki는 연구 커뮤니티가 LLM의 성능을 평가하고 개선할 수 있는 견고한 기준을 제공하며, 이와 관련된 코드가 추후 공개될 예정입니다.



### When does a predictor know its own loss? (https://arxiv.org/abs/2502.20375)
- **What's New**: 본 논문은 손실 예측(loss prediction)이라는 문제를 다루며, 이는 예측자가 입력에서 발생할 손실을 얼마나 잘 예측할 수 있는지를 평가하는 핵심 과제입니다. 저자들은 비결정적 손실 예측과 다중 보정(multicalibration) 간의 밀접한 관계를 확립하는데 중점을 두고 있습니다. 손실 예측기가 예측자의 자기-평가(self-estimate)를 개선할 수 있는 경우, 이는 다중 보정의 실패를 입증하는 역할을 하며, 반대의 경우도 성립합니다.

- **Technical Details**: 저자들은 제안된 손실 예측기가 기대되는 손실을 측정하는 데 도움을 줄 수 있다고 강조합니다. 손실 예측 문제는 일반적으로 회귀(regression) 문제로 간주되며, 손실 예측기의 품질은 실제 손실에 대한 제곱 손실(expected squared loss)으로 측정됩니다. 손실 예측은 예측 불확실성(uncertainty estimation) 문제와 밀접하게 연관되어 있으며, 이는 모델의 구조적 결함이나 훈련 데이터 부족으로 인해 발생하는 인식 불확실성(epistemic uncertainty)을 해결할 수 있는 방법도 제시합니다.

- **Performance Highlights**: 연구 결과, 손실 예측기가 예측자의 다중 보정 오류(multicalibration error)와 강력한 양의 상관관계를 보이는 것으로 나타났습니다. 이는 손실 예측이 다중 보정을 감사(auditing)하는 것과 비슷한 난이도를 가진다는 것을 시사합니다. 저자들은 실험을 통해 이론적 결과를 지원하며, 손실 예측의 유용성이 다양한 상황에서 어떻게 적용될 수 있는지를 보여줍니다.



### Constrained Generative Modeling with Manually Bridged Diffusion Models (https://arxiv.org/abs/2502.20371)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 제한된 공간에서의 확산 기반 생성 모델링을 위한 신규 프레임워크를 제안합니다. 특히, 'manual bridges'라는 새로운 개념을 도입하여, 실제적으로 적용할 수 있는 다양한 제약 조건을 확장했습니다. 이를 통해, 여러 제약 조건을 결합하면서도 모든 제약을 준수하는 'multiply-constrained model'의 훈련 메커니즘을 개발할 수 있었습니다. 이러한 이론적 확장을 통해 자율주행 차량의 경로 계획 및 제어와 같은 고부가가치 응용을 강조합니다.

- **Technical Details**: 제안된 아키텍처는 확산 기반 생성 모델에 제약을 부과하는 'manual bridges'를 효과적으로 적용할 수 있도록 설계되었습니다. 이러한 메커니즘을 활용하여, 제약 조건을 준수하면서도 복잡한 데이터 분포에 잘 적합하는 새로운 생성 모델 패밀리인 'manually bridged models'가 탄생했습니다. 논문에서는 제약 조건을 효율적으로 조합하고, 다중 제약을 준수하는 모델을 훈련하는 방법에 대한 수학적 유효성을 보장하는 이론을 개발했습니다.

- **Performance Highlights**: 실험 결과, 'manual bridges' 메커니즘이 여러 제약 조건을 훌륭히 만족시키며, 일반화 능력에 큰 손실을 주지 않음을 입증했습니다. 특히 자율주행 차량의 행태 시뮬레이션 및 경로 계획 문제에서 뛰어난 성능을 보여주었습니다. 기존의 생성 모델들이 비현실적인 샘플을 생성하는 문제를 해결하면서, 보다 현실적이고 신뢰할 수 있는 결과를 만들어 낼 수 있음을 확인했습니다.



### Safety Representations for Safer Policy Learning (https://arxiv.org/abs/2502.20341)
Comments:
          Accepted at International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이 논문에서는 Safety Representations for Policy Learning (SRPL)라는 새로운 방법론을 제안합니다. 이 접근 방식은 에이전트의 경험을 기반으로 한 상태 조건 안전 표현을 학습하여 에이전트의 상태 표현을 보강합니다. 이를 통해 에이전트는 보다 안전하게 탐색할 수 있으며, 너무 보수적인 행동을 보이지 않게 되어 효율적인 정책 학습이 가능해집니다.

- **Technical Details**: SRPL 프레임워크는 Markov Decision Process (MDP) 및 Constrained MDP (CMDP)의 개념을 바탕으로 합니다. 이 시스템에서는 상태 S, 행동 A, 보상 R, 전이 확률 T와 같은 기본 요소를 사용하여 에이전트가 최적의 정책을 찾도록 합니다. 특히, SRPL은 상태 중심의 안전 표현을 활용하여, 에이전트의 다양한 경험에서 얻은 데이터로부터 안전 정보를 통합하는 방식을 통해 안정성과 효율성을 동시에 향상시킵니다.

- **Performance Highlights**: 실험 결과, SRPL을 적용한 에이전트는 다양한 로봇 작업에서 높은 샘플 효율성을 유지하면서도 제약 사항 위반을 감소시켜 더 나은 성능을 보였습니다. 이러한 성능 향상은 안전 정보가 여러 작업 간에 잘 이전될 수 있음을 보여주며, 이는 새로운 정책 학습에 대한 유용한 선행 지식을 제공합니다. 종합적으로, SRPL은 안전과 탐색 사이의 균형을 성공적으로 유지하는 것으로 입증되었습니다.



### Mixture of Structural-and-Textual Retrieval over Text-rich Graph Knowledge Bases (https://arxiv.org/abs/2502.20317)
- **What's New**: 이번 연구에서는 Text-rich Graph Knowledge Bases (TG-KBs)에서 텍스트와 구조적 지식을 함께 효과적으로 검색하기 위한 Mixture of Structural-and-Textual Retrieval (MoR) 방법을 제안합니다. 기존의 검색 방법들은 일반적으로 이러한 지식을 분리하여 검색하는 경향이 있으며, 구조적 검색을 완전히 우회하는 경우도 많습니다. MoR은 플래닝-추론-조직화(Planning-Reasoning-Organizing) 프레임워크를 통해 두 종류의 지식을 통합적으로 검색하고, 이를 통해 서로의 이점을 강화합니다.

- **Technical Details**: MoR은 세 가지 주요 단계로 구성됩니다. 첫 번째 단계에서 MoR은 쿼리에 대한 계획 그래프를 생성하여 텍스트 계획을 수립합니다. 두 번째 단계에서는 구조적 탐색과 텍스트 매칭을 결합하여 TG-KB에서 후보를 얻습니다. 마지막으로, 조직 단계에서는 구조적 경로를 기반으로 가져온 후보를 재정렬하는 구조 인식 재정렬기(Structure-aware Rerank)를 적용합니다.

- **Performance Highlights**: MoR은 기존의 검색 방법과 비교했을 때 구조적 및 텍스트적 검색의 조화를 통해 우수한 성능을 보입니다. 실험 결과는 서로 다른 쿼리 논리에 따른 고르지 않은 검색 성능을 보여주며, 후보 재정렬에서 구조적 경로를 통합할 때의 이점을 강조합니다. MoR의 구현 코드는 지정된 링크에서 확인할 수 있습니다.



### Adversarial Robustness in Parameter-Space Classifiers (https://arxiv.org/abs/2502.20314)
- **What's New**: 본 논문은 최근 증가하는 연구 분야에서 Implicit Neural Representations (INRs)의 중요성을 강조하고 있습니다. INRs는 고차원 데이터의 효율적이고 압축된 표현을 가능하게 하여 다양한 다운스트림 태스크를 쉽게 수행할 수 있게 돕습니다. 특히, 본 연구에서는 파라미터 공간에서 훈련된 모델이 적대적 공격(adversarial attacks)에 대해 본질적으로 강인하다는 점을 보여주고 있습니다. 이러한 주장을 뒷받침하기 위해 새로운 형태의 적대적 공격을 개발하고, 그 적용 가능성을 분석하였습니다.

- **Technical Details**: INRs는 신호를 조건化하여 예측하는 신경망으로 형성되며, 이는 특정 신호에 맞춰 설계됩니다. 연구에서는 파라미터 공간 분류기가 적대적인 공격을 받더라도 본질적으로 강인하다는 것을 발견하였습니다. 이러한 공격은 분류 성능을 떨어뜨리는 동시에 원래 신호 도메인의 데이터 충실성을 유지해야 하기 때문에 복잡성이 높습니다. 본 연구는 새로운 유형의 적대적 공격을 제안하며, 파라미터 공간에서의 분류 모델의 비교 분석을 진행하였습니다.

- **Performance Highlights**: 연구에서 제안한 방법은 높은 차원의 데이터 처리 시 적대적 공격에 대해 내성이 있음을 증명했습니다. 저자들은 컴퓨터 자원 소모를 최소화하고 훈련 모델의 오버피팅을 방지하기 위해 파라미터 공간의 사전 처리 방법을 제안하였으며, 이 방법이 다양한 신호 도메인에서 효과적으로 작동함을 보여주고 있습니다. 마지막으로, 3D 데이터에 대한 새로운 적대적 공격 기법을 제안하며, 파라미터 공간과 신호 공간 모두에서 그 성능을 입증했습니다.



### Adapting Automatic Speech Recognition for Accented Air Traffic Control Communications (https://arxiv.org/abs/2502.20311)
- **What's New**: 이번 연구에서는 제정된 데이터셋을 활용하여 동남아시아 억양(Southeast Asian-accented) 영어에 맞춤화된 자동 음성 인식(ASR) 모델을 개발했습니다. 이 모델은 특히 소음이 많은 공역에서도 높은 정확도의 전사율을 달성했으며, 동남아시아 억양의 ATC 통신에서 9.82%의 단어 오류율(Word Error Rate, WER)을 기록했습니다. 이는 비서구 억양들의 전사 정확성을 높이기 위한 데이터셋과 훈련 방법론의 중요성을 강조하며, 자원이 제한된 군사 작전에서 ASR 시스템의 활용 가능성을 제시합니다.

- **Technical Details**: 연구의 주요 목표는 동남아시아 억양의 영어 전사 정확성 향상, 소음 있는 공역에서의 강인성 개선, 그리고 자원이 제한된 하드웨어에서의 효율적 실행을 보장하는 것입니다. 이를 위해 신규 개발된 동남아시아 억양 ATC 데이터셋을 활용하고, 이에 맞는 노이즈 회복 훈련 전략을 적용하였습니다. 모델 성능 평가에 있어 엄격한 기준을 적용하였으며, 기존 최첨단 모델과 비교하여 필드 테스트 결과에서 유의미한 성과를 거두었습니다.

- **Performance Highlights**: 모델 테스트 결과, 연구는 주어진 데이터셋에서 0.0982 또는 9.82%의 단어 오류율을 기록하여 지역 특정 지식에 기반한 미세조정의 효과성을 입증합니다. 또한 이 연구의 결과는 민간 및 군사 ATC 맥락에서 ATC 통신 전사 정확성과 효율성을 극대화하는 데 기여하는 중요한 통찰력을 제공합니다. 고로, ASR 기술의 발전이 군사 작전의 성공적인 실행으로 이어질 수 있음을 강하게 시사합니다.



### An exploration of features to improve the generalisability of fake news detection models (https://arxiv.org/abs/2502.20299)
Comments:
          Accepted at Expert Systems with Applications (Elsevier)

- **What's New**: 이 논문은 가짜 뉴스 탐지의 일반화 가능성을 높이기 위한 연구로, 기존의 불완전한 라벨링 데이터에 대한 문제를 다룹니다. 연구에서는 TF-IDF 및 BERT와 같은 토큰 기반 모델이 편향된 데이터에 민감하다는 점을 시사하며, 스타일적 특징(lexical, syntactic, semantic)과 사회적 모니타이제이션(social-monetisation) 특징의 중요성을 강조합니다. 이외에도 그 동안 제한적으로 활용된 대규모 언어 모델(LLMs)의 적합성에 대한 평가도 진행합니다.

- **Technical Details**: 논문에서는 NELA 2020-21 데이터 세트를 사용하여 훈련하고, 수동 라벨링 된 Facebook URLs 데이터 세트를 이용해 일반화 가능성을 평가합니다. 연구는 스타일적 특징 및 사회적 모니타이제이션 특징이 토큰 기반 방법보다 더 일반화 가능한 예측을 제공한다는 주장을 통해 성능을 분석합니다. 더불어, 통계적 및 순열 특징 중요성 분석을 통해 데이터 세트 편향을 완화하고 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과 토큰 기반 모델은 편향된 데이터에서 훈련될 경우 30% 정도의 정확도 저하를 겪는 반면, 스타일적 및 사회적 모니타이제이션 특징은 일반화 가능성을 증대시키며 더 나은 성과를 보여줍니다. 또한, LLaMa와 같은 대규모 언어 모델들이 가짜 뉴스 탐지에 효과적이지 않다는 제한적인 증거를 제시합니다. 결과적으로, 스타일적 특징과 경제적 동기를 이해하는 것이 가짜 뉴스 탐지의 발전에 기여할 수 있음을 강조합니다.



### Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription (https://arxiv.org/abs/2502.20295)
Comments:
          11 pages (including references and appendix), 14 figures, accepted at AAAI-25 Workshop on Document Understanding and Intelligence, non-archival

- **What's New**: 이 논문은 다중 페이지의 손글씨 문서를 제로샷(zero-shot) 설정에서 전사(transcribe)하는 데 있어 다중 모달 대형 언어 모델(MLLMs)의 활용을 탐구합니다. 기존의 OCR 엔진이 인쇄된 텍스트에 강력한 성능을 보이는 반면, 손글씨 처리에서는 제한적이기 때문에 MLLMs를 엔드 투 엔드 전사기로 활용하거나 후처리(post-process)기로 사용하는 다양한 구성에 대해 고찰합니다. 주목할 만한 점은, '+first page'라는 새로운 방법을 제안하며, 이는 전체 문서의 OCR 출력을 제공하면서 첫 페이지 이미지만 활용하여 MLLM 전사 정확도를 높입니다.

- **Technical Details**: 결과적으로, MLLM은 손글씨 문서의 여러 페이지에 걸쳐 공통적인 서식(formatting) 및 맥락적(feature) 정보를 활용합니다. OCR 시스템은 페이지 단위로 동작하는 반면, MLLM은 이러한 페이지 간의 종속성을 고려하여 전사 작업을 향상시킵니다. 이 연구에서는 IAM 손글씨 데이터베이스를 사용하여 제안된 방법의 효과를 검증하고, MLLM이 단일 페이지에서 배운 서식 및 OCR 오류 패턴을 활용하여 전체 문서를 향상시키는 것을 확인합니다.

- **Performance Highlights**: 실험 결과, '+first page' 접근 방식이 전사 정확도를 개선하고 비용과 성능 간의 균형을 이룬다는 것이 나타났습니다. 또한, 이 방법은 고가의 MLLM과 비교적 저렴한 OCR 방법 간의 제휴를 통해, 문서 내의 서식과 오류 패턴을 외삽(extrapolate)하여 성능을 향상시켰습니다. 이러한 결과는 다중 페이지 손글씨 문서 전사 작업에서 MLLM의 가능성을 제시하며, 향후 다양한 분야에서 활용될 수 있습니다.



### Scalable Graph Attention-based Instance Selection via Mini-Batch Sampling and Hierarchical Hashing (https://arxiv.org/abs/2502.20293)
- **What's New**: 본 논문은 그래프 주의 기반 인스턴스 선택(Graph Attention-based Instance Selection, GAIS) 방법을 소개합니다. GAIS는 주의 메커니즘(attention mechanisms)을 활용하여 그래프 표현에서 구조적 관계를 통해 정보가 풍부한 인스턴스를 식별합니다. 특히, 대규모 데이터셋을 처리할 수 있는 거리 기반 미니 배치 샘플링(mini-batch sampling) 기법과 효율적인 유사도 계산을 위한 계층적 해싱(hierarchical hashing) 접근 방식을 제안합니다.

- **Technical Details**: 저자들은 GAIS의 두 가지 주요 기법을 통해 그래프 구성의 복잡성을 줄이고 클래스 분포를 유지할 수 있다고 밝힙니다. 첫 번째 기법은 거리 기반 미니 배치 샘플링을 통해, 두 번째 기법은 랜덤 프로젝션(random projections)을 통한 다수준 및 다중 뷰 지역 민감 해싱(locality-sensitive hashing) 변형을 사용하는 것입니다. 이런 접근 방식은 고차원 공간에서의 복잡한 관계를 포착하는 데 도움을 줍니다.

- **Performance Highlights**: 다양한 39개 데이터셋에 대한 실험에서 GAIS는 96% 이상의 데이터 감소율을 보여주면서 최고의 인스턴스 선택(IS) 방법론과 비교해 성능을 유지하거나 향상시키는 결과를 얻었습니다. 특히, 미니 배치 접근법은 대규모 데이터셋에 대해 효율성과 효과성의 최적 균형을 제공하였으며, 다중 뷰 변형은 복잡하고 고차원 데이터에 대해 우수한 성능을 보였습니다.



### Conformal Tail Risk Control for Large Language Model Alignmen (https://arxiv.org/abs/2502.20285)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 성능 신뢰성을 보장하기 위한 최신 경량 보정(calibration) 프레임워크를 제안합니다. 특히 LLM이 생성하는 유해한 출력에 대한 위험을 관리하는 데 중점을 두며, 인간과 머신 간의 정렬 문제를 해결하기 위해 고안된 방법론을 다룹니다. 또한, 기존 연구와 달리 모델 재조정 없이도 이러한 위험을 제어하는 한계 보증을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 손실의 가중 평균 소위 조건부 가치-at-위험(Conditional Value-at-Risk, CVaR) 메트릭을 활용하여 LLM의 출력을 보정합니다. 기존의 방법들이 보통 기대값 같은 전통적인 위험 측정치를 다루는 반면, 본 연구에서는 희소한 사건들에 대한 위험 간의 정렬을 달성하는 데 중점을 두고 있습니다. 이를 위해서는 L-통계(L-statistics)의 이론을 기반으로 하여 도출된 새로운 위험 제어 경계들이 사용됩니다.

- **Performance Highlights**: 이 연구의 실험 결과는 제안된 보정 프레임워크가 기존 모델들과 비교하여 LLM의 출력에서 인간 평가와의 정렬을 효과적으로 제어할 수 있음을 보여줍니다. 또한, 자동화된 분류 모델인 Detoxify와 같은 기존의 모델들보다 상대적으로 저비용으로 정렬을 달성할 수 있는 가능성을 지니고 있습니다. 최종적으로, 이러한 접근은 사회적으로 민감한 다수의 응용 분야에서 LLM의 신뢰성을 높일 수 있는 잠재력을 나타냅니다.



### Online Meta-learning for AutoML in Real-time (OnMAR) (https://arxiv.org/abs/2502.20279)
Comments:
          First page is a graphical abstract, this is a journal article submission

- **What's New**: 본 연구는 온라인 메타 학습(Online Meta-learning, OnMAR) 접근 방식을 제안하여 실시간 자동 머신 러닝(Real-time AutoML)에서 알고리즘 설계의 품질을 최적화하고자 한다. 이 방법은 메타 학습(Meta-learning)을 통해 ML 알고리즘의 최적화 과정을 파악하고, 이를 통해 설계의 정확도를 예측하여 새로운 디자인을 생성하거나 기존 디자인을 사용할지 결정한다. OnMAR 기술은 여러 실시간 AutoML 응용 분야에서 평가되며, 오프라인 메타 학습(Offline Meta-learning, OffMAR) 방법과 비교되어 성능이 우수함을 입증하고 있다.

- **Technical Details**: OnMAR 접근 방식은 메타 데이터를 사용하여 각각의 ML 알고리즘 설계에 대한 메타 특징(Meta-features)을 수집하고 이를 통해 메타 학습기를 구성한다. 이 메타 학습기는 예상 설계의 정확도를 예측하여, 충분한 경우에는 해당 디자인을 채택하고 그렇지 않으면 유전자 알고리즘(Genetic Algorithm, GA)을 통해 새로운 디자인을 생성한다. 또한, k-최근접 이웃(k-nearest neighbours), 랜덤 포레스트(Random Forest), XGBoost와 같은 다양한 메타 학습기가 사용되어 최적화 과정을 지원한다.

- **Performance Highlights**: OnMAR 방식은 기존의 실시간 AutoML 접근 방식에 비해 성능이 동등하거나 우수한 결과를 나타내며, 전체 실행 시간을 단축시키는 장점이 있다. 실시간 이미지 클러스터링 알고리즘 구성, 컨볼루션 신경망의 하이퍼파라미터 설정, 비디오 분류 파이프라인 설정 등 다양한 응용 분야에서 효과를 입증하였다. 본 연구의 결과는 OnMAR이 OffMAR보다 우수한 성능을 보이며, 더 효율적인 ML 알고리즘 설계 방법으로 자리매김 할 수 있음을 시사한다.



### Large Language Models as Attribution Regularizers for Efficient Model Training (https://arxiv.org/abs/2502.20268)
- **What's New**: 이 논문은 큰 언어 모델(LLMs)을 활용하여 훈련 효율성을 높일 수 있는 새로운 방법을 제안합니다. 특히, LLM에서 생성된 전역 태스크 피처 기여도(attribution)를 소형 네트워크의 훈련 과정에 통합하는 방법에 대해 설명하고 있습니다. 이 접근법은 LLM의 통찰력을 소형 모델에 결합하여, 특히 데이터가 불균형하거나 제한되어 있을 때 전반적인 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서 제안하는 Large Language Model Attribution Aligned Training (LAAT) 방법은 attribution-matching regularization 항을 도입하여 소형 모델의 학습 동역학을 LLM이 제공하는 동적인 통찰력에 맞추는 것을 목표로 합니다. 이 방식은 상대적으로 대규모 GPU 자원에 대한 의존성을 줄이고, 최소한의 계산 오버헤드로 LLM의 강력한 일반화 능력을 활용하여 설명 가능한 소형 모델의 훈련을 가능하게 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험을 통해 본 방법이 저샘플링(few-shot learning) 환경에서 모델의 학습 효율성과 강인성을 개선함을 보여줍니다. 본 연구에서 제안된 방법은 LLM에 대한 블랙박스 API 접근만으로도 용이하게 통합할 수 있으며, 데이터의 왜곡(skewness) 및 편치(bias)와 같은 문제를 해결하는 데 유용함을 입증했습니다.



### On the Importance of Reward Design in Reinforcement Learning-based Dynamic Algorithm Configuration: A Case Study on OneMax with (1+($λ$,$λ$))-GA (https://arxiv.org/abs/2502.20265)
- **What's New**: 최근 연구에서는 다이나믹 알고리즘 구성(Dynamic Algorithm Configuration, DAC)이 머신 러닝과 딥 러닝 알고리즘의 발전으로 인해 많은 주목을 받고 있습니다. 이 논문은 리인포스먼트 러닝(Reinforcement Learning, RL) 에이전트의 리워드 설계의 중요성을 강조하고, 보상 설계가 RL 에이전트의 학습에 미치는 영향을 다룹니다. 저자는 RL을 통해 $(1+(	ext{λ,λ}))$-GA를 동적으로 구성할 수 있고, 리워드 쉐이핑(reward shaping)의 이점도 확인했습니다.

- **Technical Details**: 이 연구에서는 리워드 기능 설계를 통해 DAC 문제를 해결하기 위해 DDQN(Deep Q-Network) 알고리즘을 적용했습니다. 초기 실험에서 단순한 리워드 기능이 여러 문제 크기에서 확장성과 학습 발산 문제를 일으킨다는 것을 발견했습니다. 저자들은 보상 함수에 문제 차원성을 완화하기 위한 스케일링 메커니즘을 제안했지만, 이것이 큰 문제에는 한계가 있음을 보여주었습니다.

- **Performance Highlights**: 보상 편향(보상 시프트) 기법을 도입하여 저자는 탐색 부족 문제를 해결하고, 모든 문제 크기에서 이론 기반 정책보다 우수한 성능을 달성할 수 있었습니다. 이 방법을 통해 RL 정책은 이론적으로 유도된 정책에 비해 몇 배의 성능 향상을 이루었습니다. 또한, 모든 소스 코드와 데이터는 공개적으로 제공되어 연구자들이 쉽게 접근할 수 있도록 하였습니다.



### Understanding the Limits of Deep Tabular Methods with Temporal Shif (https://arxiv.org/abs/2502.20260)
Comments:
          17 pages, 9 figures

- **What's New**: 이 논문은 심층 tabular 모델이 temporal distribution shifts에 취약한 이유를 탐구합니다. 저자들은 기존의 훈련 프로토콜을 분석했고, 무작위 분할이 있을 때 모델의 성능이 크게 향상될 수 있음을 발견했습니다. 이 연구는 temporal data를 학습하는 데 있어 큰 주요 사항들을 강조하며, 시계열 패턴을 효과적으로 포착할 수 있는 방법을 제안합니다.

- **Technical Details**: Temporal distribution shifts는 모델 훈련 과정의 어려움을 증가시킵니다. 훈련 데이터와 테스트 데이터 간의 시간 차이와 validation의 편향이 효과적인 모델 성능을 저하시킵니다. 저자들은 Fourier series expansion을 기반으로 한 temporal embedding 방법을 제안하여 시계열 정보를 효과적으로 통합합니다.

- **Performance Highlights**: 실험 결과, 개선된 훈련 프로토콜과 temporal embedding을 결합하여 temporal tabular data에서 모델 성능이 크게 향상됨을 보여주었습니다. 이 결과는 심층 tabular 방법이 일반 실세계 데이터에서도 보다 나은 성능을 발휘할 수 있도록 하는 중요한 통찰을 제공합니다.



### The Impact of Transparency in AI Systems on Users' Data-Sharing Intentions: A Scenario-Based Experimen (https://arxiv.org/abs/2502.20243)
Comments:
          This is the author's version of the paper presented at the 19th International Conference on Wirtschaftsinformatik (WI 2024). The official published version is available at this https URL

- **What's New**: 이번 연구는 투명한 데이터 처리 시스템(white-box AI)과 비투명한 데이터 처리 시스템(black-box AI)이 사용자의 데이터 공유 의사에 미치는 영향을 조사했습니다. 특히, 다양한 데이터 처리 시스템에 대한 신뢰와 개인 정보 보호 우려가 이러한 관계에 미치는 영향을 분석했습니다. 예기치 않게도, 데이터 공유 의사에는 유의미한 차이가 없었으며, 투명성 여부가 데이터 공유 의사에 긍정적인 영향을 미치지 않는다는 것을 발견했습니다.

- **Technical Details**: 연구에 사용된 방법론은 240명의 참가자를 대상으로 한 시나리오 기반 실험이었습니다. 참가자는 가상의 수면 앱에 대한 다양한 데이터 처리 엔티티에 노출되었고, 이 앱은 사용자가 입력한 데이터를 기반으로 개인화된 조언을 제공했습니다. 또한 연구에서는 AI 시스템에 대한 신뢰와 개인 정보 보호 우려가 데이터 공유 의사에 미치는 영향을 평가했습니다. 연구 결과, AI 시스템에 대한 일반적인 신뢰 태도가 특히 투명한 AI 조건에서 데이터 공유 의사에 긍정적인 영향을 미친다는 점을 강조하였습니다.

- **Performance Highlights**: 이번 연구 결과, 투명한 AI와 비투명한 AI 시스템 간의 데이터 공유 의사의 차이는 발견되지 않았다는 점이 주목할 만합니다. 이는 사용자들이 데이터 처리를 어떻게 이해하든 관계없이 데이터 공유에 대한 의사가 다르게 나타나지 않는다는 것을 의미합니다. 그러나, 연구에 따르면, AI 시스템에 대한 사용자 신뢰는 데이터 공유 의사에 긍정적인 영향을 미치며, 개인 정보 보호 우려는 크게 영향을 미치지 않았습니다.



### Teasing Apart Architecture and Initial Weights as Sources of Inductive Bias in Neural Networks (https://arxiv.org/abs/2502.20237)
Comments:
          11 pages, 6 figures, 6 tables

- **What's New**: 이 논문은 인공 신경망(artificial neural networks)의 초기 가중치(initial weights)가 지니는 귀납적 편향(inductive bias)의 영향을 탐구합니다. 저자들은 메타 학습(meta-learning)을 활용하여 특정 문제에 적합하게 조정된 초기 가중치를 찾는 방법을 제시하고, 이를 통해 다양한 신경망 아키텍처(architectures)를 비교합니다. 이 연구는 기본적인 아키텍처 외에도 초기 가중치가 학습 성능에 미치는 영향을 조명하여 인지 과학(cognitive science)과 기계 학습(machine learning) 간의 교차점을 강조합니다.

- **Technical Details**: 저자들은 MLPs(다층 퍼셉트론), CNNs(합성곱 신경망), LSTMs(장단기 메모리 네트워크) 및 Transformer's와 같은 네 가지 인기 있는 아키텍처를 사용하여 430개 모델을 메타 트레이닝(meta-training)했습니다. 연구는 이러한 타스크(task) 전반에서 메타 학습이 기존 아키텍처와 데이터 표현 간의 성능 차이를 어떻게 줄일 수 있는지 문서화합니다. 각 아키텍처는 고유한 초기 가중치를 통해 서로 다른 학습 편향을 실현하여 성능을 나타냅니다.

- **Performance Highlights**: 메타 학습을 통해 얻은 초기 가중치는 구조적 차원에서의 성능 차이를 감소시키고, 특정 아키텍처가 메타 학습을 통해 더 효과적으로 학습할 수 있도록 합니다. 하지만 학습 경험에서 멀리 떨어진 문제들에 대해서는 여전히 모든 아키텍처가 저조한 일반화(generalization) 성능을 보이며, 이는 더욱 강력한 귀납적 편향을 요구함을 시사합니다. 연구 결과는 적절한 초기 가중치와 학습률의 선택이 다양한 편향을 구현할 수 있도록 해 준다는 점에서 신경망 아키텍처의 유연성을 강조합니다.



### Mixture of Experts for Recognizing Depression from Interview and Reading Tasks (https://arxiv.org/abs/2502.20213)
- **What's New**: 본 연구는 우울증 인식을 위한 최초의 연구로서, 자발적(spontaneous) 및 낭독(read) 음성을 모두 활용하여 특징을 추출합니다. 이전 연구들은 주로 자발적 발화를 사용하였으나, 본 연구는 다중 모달 융합(multimodal fusion) 방법과 Mixture of Experts (MoE) 모델을 단일 심층 신경망(deep neural network)에서 결합했습니다. 이를 통해 낭독 및 자발적 음성에서 이전에 접근하지 못했던 정보를 활용하여 우울증 인식의 정확성을 향상시킵니다.

- **Technical Details**: 연구는 오디오 파일에서 log-Mel spectrogram, velocity, acceleration 등의 세 가지 채널의 이미지로 변환한 후, 두 개의 공유 AlexNet 모델에 입력으로 사용합니다. 이후 AlexNet 모델의 출력을 다중 모달 융합 기법인 BLOCK에 입력하고, 마지막으로 세 가지 변형(Mixtures) MoE 방법을 통해 결과를 생성합니다. 이 방법론은 모델이 자발적 및 낭독 음성의 정보를 통합하여 우울증을 인식하는 데 있어 향상된 성능을 보여줍니다.

- **Performance Highlights**: 본 연구의 실험 결과는 Androids corpus에서 Accuracy 87.00%, F1-score 86.66%를 달성하여 기존의 다른 방법들보다 나은 성능을 입증하였습니다. 특히, 자발적 및 낭독 음성을 조화롭게 결합함으로써 더욱 정확한 우울증 인식이 가능하게 되었습니다. 타 연구와 비교할 때, 본 연구의 접근법은 기존 방법의 한계를 해결하며, 임상적 적용 가능성을 높일 수 있는 방향으로 기여하고 있습니다.



### Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies (https://arxiv.org/abs/2502.20190)
- **What's New**: 본 논문에서는 Deep Reinforcement Learning (DRL) 교육의 속도를 높이기 위해 TianJi라는 새로운 고처리량 분산 RL 훈련 시스템을 제안합니다. TianJi는 서브태스크 구성 요소 간의 할당 의존성을 완화하고, 이벤트 기반 비동기 통신을 가능하게 하여 훈련 효율성을 개선합니다. 이 시스템은 clear boundaries를 유지하면서도 샘플 생산과 소비의 균형을 맞추는 분산 전략을 통해 수렴 불확실성을 해결합니다.

- **Technical Details**: TianJi는 비동기적으로 느슨하게 결합된 프로세스로 서브태스크 구성 요소 간의 의존성을 완화합니다. 또한, 샘플의 질을 보장하며 훈련 속도를 높이기 위해 샘플의 신선도를 조절하는 전략을 도입합니다. 이를 통해, 다양한 RL 시스템 중에서 중요한 inter-component 의존성을 해결하여 높은 수준의 병렬 처리를 달성합니다.

- **Performance Highlights**: 실험 결과, TianJi는 관련 시스템과 비교할 때 최대 4.37배의 수렴 시간 가속 비율을 달성하였고, 8개의 컴퓨팅 노드에 스케일링할 경우 1.6배의 속도 향상과 7.13배의 처리량 향상을 보여주었습니다. 데이터 전송 효율성 실험 결과, TianJi는 다른 시스템에 비해 현저한 성능을 발휘하며, 하드웨어 한계에 근접한 효율성을 나타냈습니다.



### Mixture of Experts-augmented Deep Unfolding for Activity Detection in IRS-aided Systems (https://arxiv.org/abs/2502.20183)
Comments:
          5 pages, 4 figures, Submitted to IEEE Wireless Communications Letters

- **What's New**: 이 논문에서는 지능형 반사면(Intelligent Reflecting Surfaces, IRS)을 활용한 대량의 기계형 통신에서의 활동 감지 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존 활동 감지 방법은 단일 채널 모델에 의존하는 반면, 이 연구는 여러 채널 모델을 아우르는 혼합 전문가(mixture of experts, MoE) 프레임워크와 모델 기반의 심층 전개(deep unfolding)를 결합하여 복잡한 실제 상황을 반영합니다. 이 방법은 장치와 기지국(base station, BS) 간의 채널 유형에 대한 사전 지식 없이도 적용 가능합니다.

- **Technical Details**: 제안된 시스템 모델은 M개의 안테나를 가진 BS를 주요 인프라로 구성하며, 다양한 장치들이 세 가지 세트로 나뉘어 서로 다른 페이딩 환경을 가집니다. 특히, 각 장치는 고유한 서명 시퀀스를 가지고 있으며, BS는 동기화된 송신 신호를 기반으로 활동 감지를 수행합니다. 기존의 공분산 기반 알고리즘의 한계를 극복하기 위해, 제안된 방법은 투영된 경량 경량 모델을 사용하여 최적화 과정을 심층 네트워크의 레이어로 변환합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안된 MoE가 증강된 심층 전개 방법은 전통적인 공분산 기반 방법과 블랙 박스 신경망 설계를 초월하는 우수한 검출 성능을 보여줍니다. 특히, 각 장치의 페이딩 채널 유형을 알지 못하는 경우의 성능 저하는 거의 일어나지 않으며, 이는 실제 응용에서도 큰 이점으로 작용할 수 있음을 나타냅니다.



### Similarity-Distance-Magnitude Universal Verification (https://arxiv.org/abs/2502.20167)
Comments:
          35 pages (8 Tables, 4 Algorithms, 5 Listings)

- **What's New**: 본 논문에서는 신경망의 Robustness 문제를 해결하기 위해 Softmax 함수의 출력 Magnitude(결정 경계) 인식에 Similarity(유사성)와 Distance(거리) 인식을 추가한 새로운 sdmsdm 활성화 함수를 제안합니다. 이 새로운 활성화 함수는 상대적인 epistemic (감소 가능한) 예측 불확실성의 강력한 신호를 제공합니다. 불확실성 추정치는 학습된 변환을 통해 얻으며, 이는 모델의 예측을 해석 가능하게 하여 HCI 문제도 해결할 수 있습니다.

- **Technical Details**: 기존 LLM(대규모 언어 모델)의 한계는 예측의 불확실성을 정량화하는데 있어 제약이 많았습니다. sdmsdm 활성화 함수는 training set에 대한 정확한 예측 적합도를 통해 epistemic 불확실성을 분해하는 새로운 접근 방식을 제공합니다. 이러한 방법론은 모델의 결정 경계에 대한 거리를 고려하며, 인간이 이해 가능하도록 모델의 불확실성을 탐색할 수 있는 능력을 부여합니다.

- **Performance Highlights**: sdmsdm 활성화 함수는 테스트 시간의 분포 변화와 분포 외 입력에 대해 놀라운 강건성을 보이며, 효과적인 샘플 크기를 인식합니다. 불확실성 추정치는 다른 접근 방식과 비교할 때 훨씬 더 신뢰할 수 있는 결과를 보여줍니다. 또한, 새로운 LLM 아키텍처인 sdmsdm 네트워크는 불확실성 인식 및 사례 기반 해석 가능성을 본질적인 특성으로 갖추고 있습니다.



### Gradient-Guided Annealing for Domain Generalization (https://arxiv.org/abs/2502.20162)
Comments:
          Paper accepted in CVPR2025

- **What's New**: 이 논문에서는 Domain Generalization (DG) 문제에 대한 새로운 접근 방식을 제안하고 있습니다. 특히, 모델 훈련 초기 단계에서의 매개변수 조정이 도메인 일반화의 효과성에 중요한 역할을 한다고 관찰했습니다. 이를 통해 Gradient-Guided Annealing (GGA) 알고리즘을 도입하여, 훈련 동안 도메인 간의 그래디언트가 일치하도록 매개변수를 조정함으로써 도메인 변화에 대한 강건성을 향상시킬 수 있습니다.

- **Technical Details**: 이 연구는 스토캐스틱 그래디언트 하강법(Gradient Descent)과 같은 최적화 방법을 사용하여, 손실(lost) 경관에서의 초기 반복의 중요성을 강조합니다. 딥러닝 모델 훈련 중 도메인 간 그래디언트 분쟁(gradient conflicts) 문제를 해결하기 위해, GGA는 초기 매개변수를 점진적으로 조정하며 모든 훈련 도메인에서 그래디언트가 일치하는 지점을 찾습니다. 이 과정은 모델이 도메인 변화에 더 잘 일반화되도록 유도합니다.

- **Performance Highlights**: GGA는 다섯 가지의 도메인 일반화 벤치마크에서 높은 성능을 달성하며, 단독 알고리즘으로도 경쟁력 있는 결과를 보여주었습니다. 또한, 기존의 도메인 일반화 알고리즘과 결합했을 때, 그 효과성을 지속적으로 개선하는 결과를 나타냈습니다. 이러한 이유로 GGA는 최첨단 성과를 달성할 수 있는 잠재력을 지닌 방법으로 뚜렷이 부각됩니다.



### Transfer Learning in Latent Contextual Bandits with Covariate Shift Through Causal Transportability (https://arxiv.org/abs/2502.20153)
Comments:
          Accepted at the Conference of Causal Learning and Reasoning (CLeaR 2025), will be published in the Proceedings of Machine Learning Research

- **What's New**: 이 논문에서는 환경 간의 지식 전이에서 발생할 수 있는 부정적 효용(negative transfer) 문제를 해결하기 위해 획기적인 방법을 제안합니다. 구체적으로, 잠재적 맥락 기반 밴딧(latent contextual bandits) 문제를 분석하고, 효과적인 알고리즘을 개발하여 인과 추론(causal inference) 관점에서 지식을 전이할 수 있는 방안을 제시합니다. 本研究는 요소들이 서로 다른 여러 환경에서 지식을 전이하는 새로운 알고리즘을 제공하여 샘플 효율(sample efficiency)을 향상시키는 데 목표를 둡니다.

- **Technical Details**: 논문에서는 잠재적 맥락 기반 밴딧 문제에 대한 전이 학습을 고려하며, 특정 조건부 분포를 가진 고차원 프록시(proxy) 변수를 활용합니다. 이를 통해, 에이전트는 실제 맥락을 관찰하지 않으면서도 최적의 선택을 할 수 있습니다. 또한, 인과 추론의 운반 가능성(transportability) 이론을 적용하여 각 환경에서 추론할 수 있는 지식을 정의합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 데이터의 직접적인 전이를 시도하는 기존 방법들보다 부정적 효용을 피하는 데 성공했습니다. 또한, 여러 합성 및 반합성 데이터셋에서 샘플 효율을 일관되게 높이며, 덜 효율적인 알고리즘에 비해 월등한 성능을 보여주었습니다. 최종적으로, 이 연구는 복잡한 고차원 데이터에서 인과 효과를 추정하는 데 있어 강력한 프레임워크를 구축했습니다.



### Your contrastive learning problem is secretly a distribution alignment problem (https://arxiv.org/abs/2502.20141)
Comments:
          10 pages, 5 figures, NeurIPS 2024 submission, includes supplementary material

- **What's New**: 이 논문에서는 기존의 contrastive learning (CL)에 대한 이론적 기초와 메커니즘을 재조명하고 있습니다. 특히 noise contrastive estimation losses와 entropic optimal transport (OT) 간의 연결을 통해 새롭고 다양한 손실 함수와 다단계 반복 변형을 개발하였습니다. 이러한 접근을 통해 노이즈가 있는 뷰에서의 분포 기반 조정과 맞춤형 표현 공간 구축이 가능해졌습니다.

- **Technical Details**: 새롭게 제안된 일반화된 contrastive alignment (GCA) 프레임워크는 contrastive learning을 분포 정렬 문제로 재인터프리트합니다. 이를 통해 샘플의 정렬을 보다 유연하게 제어할 수 있으며, 타겟 수송 계획인 \( 	extbf{P}_{tgt} \)를 정의하여 사용자 맞춤형 정렬 가이드를 제공합니다. GCA는 InfoNCE 및 Robust InfoNCE와 같은 기존 기법과의 연결고리를 제시하며, 이는 Bregman 프로젝션을 활용한 반복 정렬 목표로 볼 수 있습니다.

- **Performance Highlights**: GCA는 이미지 분류 및 데이터를 포함한 다양한 작업에서 우수성을 입증하였으며, 특히 GCA의 unbalanced OT(UOT) 공식화는 분류 성능을 개선하는 데 기여하였습니다. 실험 결과, GCA는 존재하는 방법들에 비해 향상된 유연성과 성능을 제공하며, 다양한 자원 변화에 대응할 수 있는 유망한 접근 방식을 제시합니다. 특히, GCA는 도메인 특수 정보를 포함하면서도 분류 정확도를 유지하는 데 성공하였습니다.



### LimeSoDa: A Dataset Collection for Benchmarking of Machine Learning Regressors in Digital Soil Mapping (https://arxiv.org/abs/2502.20139)
- **What's New**: 이번 연구에서는 Precision Liming Soil Datasets (LimeSoDa)라는 개방형 데이터셋 컬렉션을 소개합니다. LimeSoDa는 다양한 국가에서 수집된 31개의 필드 및 농장 규모의 데이터셋으로 구성되어 있으며, 각 데이터셋은 토양 유기물, 점토 함량 및 pH와 같은 세 가지 주요 토양 특성을 포함하고 있습니다. 이 데이터는 토양 감지 및 광학 분광법을 사용하여 확보되었으며, 쉽게 모델링에 사용할 수 있도록 정리되었습니다.

- **Technical Details**: LimeSoDa는 각 데이터셋이 특정 토양 특성과 함께 데이터세트별 특징을 포함하여 정리된 표 형식으로 제공됩니다. 연구에서는 네 가지 학습 알고리즘의 예측 성능을 비교하여 MLR (Multiple Linear Regression), SVR (Support Vector Regression), CatBoost, RF (Random Forest) 알고리즘을 사용했습니다. 이 알고리즘 비교는 고차원 스펙트럴 데이터셋에서 더 나은 성능을 보인 MLR과 SVR과 같은 알고리즘의 적합성 차이를 보여줍니다.

- **Performance Highlights**: 비교 결과, 모든 알고리즘이 특정 맥락에서만 우수한 성능을 발휘했습니다. 예를 들어, MLR과 SVR은 고차원 스펙트럴 데이터셋에서 더 잘 작동했으며, CatBoost와 RF는 특성이 적은(< 20) 데이터셋에서 뛰어난 성능을 보여주었습니다. 이러한 벤치마킹 결과는 특정 알고리즘의 성능이 맥락에 따라 달라지는 것을 잘 보여주고 있으며, LimeSoDa가 DSM(디지털 토양 매핑)에서 통계적 방법의 개발과 평가를 개선하는 중요한 자원임을 증명합니다.



### Regional climate projections using a deep learning--based model-ranking and downscaling framework: Application to European climate zones (https://arxiv.org/abs/2502.20132)
Comments:
          This manuscript has been submitted to Environmental Science and Pollution Research (ESPR) for review

- **What's New**: 이 연구는 32개의 Coupled Model Intercomparison Project Phase 6 (CMIP6) 모델을 Deep Learning-TOPSIS (DL-TOPSIS) 메커니즘을 통해 평가하고, 고해상도 기후 예측을 위한 심층 학습 기반의 다중 모델 평가 및 다운스케일링 프레임워크를 제시합니다. 이를 통해 5개의 Köppen-Geiger 기후 구역(Tropical, Arid, Temperate, Continental, Polar)을 9개의 성능 기준을 사용하여 분석하였습니다. 특히 NorESM2-LM, GISS-E2-1-G, HadGEM3-GC31-LL 모델이 다른 모델보다 우수한 성능을 보였습니다.

- **Technical Details**: 본 연구에서는 Vision Transformer (ViT), Geospatial Spatiotemporal Transformer with Attention and Imbalance-Aware Network (GeoSTANet), CNN-LSTM, CNN-Long Short-Term Memory (ConvLSTM) 모델을 사용하여 상위 GCM을 0.1° 해상도로 다운스케일링하는 방법을 적용하였습니다. GeoSTANet 모델은 온도 극값(TXx, TNn)을 효과적으로 포착하며, RMSE 1.57°C, Kling-Gupta Efficiency (KGE) 0.89, Nash-Sutcliffe Efficiency (NSE) 0.85의 높은 정확도를 기록하여 ConvLSTM보다 RMSE를 20% 감소시켰습니다. 이러한 기술들은 전통적인 딥러닝 방법보다 더욱 효율적인 다운스케일링을 제공합니다.

- **Performance Highlights**: CNN-LSTM과 ConvLSTM 모델은 Continental 및 Temperate 구역에서 좋은 성능을 보였고, ViT는 세밀한 온도 변화 포착에 어려움을 겪었습니다. 다중 기준 순위 부여 기술이 GCM 선택에서 개선된 결과를 보여주며, 전반적으로 본 연구의 프레임워크는 고해상도 기후 예측을 향상시키고, 기후 영향 평가 및 적응 계획에 유익한 방법이 될 것입니다.



### FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compu (https://arxiv.org/abs/2502.20126)
- **What's New**: 이 연구에서는 현대의 Diffusion Transformer가 고정된 계산 예산으로 인한 자원 요구로 인해 제약을 받는 문제를 해결하기 위해 동적 전략을 제안합니다. 새롭게 제안된 FlexiDiT 모델은 입력에 따라 가변적인 compute budget을 처리할 수 있도록 설계되었습니다. 이 방법을 통해 모델이 품질 저하 없이 이미지를 생성하면서도 40% 이상의 FLOPs 절감이 가능합니다.

- **Technical Details**: Diffusion 모델은 이미지 생성의 핵심적인 블록으로, 순차적으로 노이즈 샘플을 제거하여 원하는 데이터 분포에서의 샘플을 생성합니다. Diffusion Transformer 모델(DiT)은 이러한 과정에서Transformer 블록을 사용하여 기존의 합성곱층(CNN) 대신 적용됩니다. 이러한 접근 방식은 멀티모달 응용통합과 효율적인 훈련이 가능하게 하며, 계산 복잡도 측면에서도 뛰어난 성능을 보입니다.

- **Performance Highlights**: FlexiDiT 모델은 고정된 버전과 비교했을 때 동일한 품질로 이미지를 생성하면서도 계산 요구량을 75%까지 감소시킬 수 있습니다. 비디오 생성에도 적용 가능하며, FlexiDiT는 다양한 조건부이미지 생성 상황에서도 일반적으로 잘 작동하며 뛰어난 성능을 보장합니다. 이러한 특징으로 FlexiDiT 모델은 최신 하드웨어에서 효율적으로 훈련될 수 있는 잠재력을 지니고 있습니다.



### Exploring Open-world Continual Learning with Knowns-Unknowns Knowledge Transfer (https://arxiv.org/abs/2502.20124)
- **What's New**: 이번 논문에서는 Open-World Continual Learning (OWCL)의 한계점을 해결하기 위해 새로운 접근 방식을 제안합니다. 기존 OWCL 방법들이 개방 탐지와 지속 학습을 별개의 작업으로 취급하는 문제를 인식하고, 이를 통합한 HoliTrans라는 새로운 프레임워크를 소개합니다. HoliTrans는 비선형 랜덤 프로젝션(nonlinear random projection, NRP)과 분포 인식 프로토타입(distribution-aware prototypes, DAPs)을 활용하여 알아야 할 것과 모르는 것을 동시에 다룹니다.

- **Technical Details**: 논문은 OWCL의 네 가지 시나리오를 정의하고, 이러한 시나리오에 대한 포괄적인 실험을 수행합니다. 특히, HoliTrans는 알려진 샘플과 알려지지 않은 샘플의 지식 전이를 지원하며, 각각의 오픈 샘플에 대한 표현을 동적으로 업데이트합니다. 이를 통해 OWCL의 연구에서 큰 도약을 기대할 수 있는 통합된 틀을 제공합니다.

- **Performance Highlights**: HoliTrans는 다양한 OWCL 시나리오에서 22개의 경쟁 베이스라인을 초월하는 성능을 보여주어, OWCL 이론과 실제 응용 간의 간극을 줄입니다. 이 연구를 통해OWCL의 도전 과제가 명확하게 드러났으며, 향후 오픈 월드 학습 패러다임의 발전에 기여할 것으로 예상됩니다.



### Identifiable Multi-View Causal Discovery Without Non-Gaussianity (https://arxiv.org/abs/2502.20115)
- **What's New**: 이번 연구에서는 다중 관점( multi-view ) 구조 방정식 모델( SEM )을 활용한 선형 인과 발견( causal discovery )의 새로운 접근 방식을 제안합니다. 본 모델은 비가우시안 교란( non-Gaussian disturbances )의 가정을 완화하고, 관점 간의 분산의 다양성을 가정하여 더 널리 적용될 수 있습니다. 연구팀은 이 모델의 모든 파라미터의 식별 가능성을 증명하였으며, 비순환(aclycic) 조건을 제외한 추가적인 구조적 가정 없이도 가능하다고 밝혔습니다.

- **Technical Details**: 우리는 여러 관련 데이터 집합이 관찰되는 선형 다중 관점 상황을 설정하며, 각 데이터 집합의 교란( disturbance )이 공유 정보를 가지고 있으나 동일하지 않다고 가정합니다. 우리의 방법론은 다중 관점 독립 성분 분석( multi-view Independent Component Analysis )의 최근 발전에 기반한 추정 알고리즘을 개발하였습니다. 특히, 이 모델은 모든 변수가 가우시안일지라도 완전히 식별 가능하며, 다중 관점 정보가 식별 가능성에 기여한다고 주장합니다.

- **Performance Highlights**: 제안된 방법론은 시뮬레이션을 통해 검증되었으며, 실제 신경 이미지 데이터에 적용되어 뇌 영역 간의 인과 그래프( causal graphs ) 추정을 가능하게 합니다. 이는 다양한 피험자로부터의 데이터에서 신경 자극에 대한 반응을 분석하여 뇌의 다양한 영역 간의 인과 관계를 밝히는 데 기여합니다. 연구 결과는 다른 생물학적 및 사회 과학적 데이터 분석에 응용될 수 있는 잠재력을 지니고 있습니다.



### Forward-Cooperation-Backward (FCB) learning in a Multi-Encoding Uni-Decoding neural network architectur (https://arxiv.org/abs/2502.20113)
- **What's New**: 이 논문에서는 Forward-Cooperation-Backward (FCB) 학습이라는 새로운 학습 기법을 제안하고 있습니다. 이 기법은 인간의 학습 방식을 모방하여, Forward-Forward 방식, 협동(cooperation), 그리고 역전파(backpropagation)를 결합한 것입니다. 그에 따라 새로운 Multi Encoding Uni Decoding (MEUD) 신경망 아키텍처도 설계되었습니다.

- **Technical Details**: FCB 학습은 다단계 아키텍처를 통해 진화하며, MEUD, MEUD-FF, MEUD-Coop, MEUD-FF-Coop 모델을 포함하여 구조적으로 발전하였습니다. 이 신경망들은 협동을 구현하기 위해 특수한 lateral synaptic connection을 사용하며, 여러 인기 있는 데이터셋에서 차원 축소(dimensionality reduction) 성능을 평가 받았습니다.

- **Performance Highlights**: MEUD-FF-Coop 프레임워크는 표준 Autoencoder와 여러 변형 모델들과 비교하여 실험적으로 우수성을 입증하였습니다. 데이터의 원래와 투영된 공간 간의 세부 관계를 보존하는 능력과 차원 축소 후 분류 성능이 다양한 분류 알고리즘을 통해 평가되어 그 품질이 입증되었습니다.



### Sanity Checking Causal Representation Learning on a Simple Real-World System (https://arxiv.org/abs/2502.20099)
Comments:
          24 pages, 12 figures

- **What's New**: 이 연구는 실험적으로 검증된 처방 시스템에서 인과 표현 학습(causal representation learning, CRL) 방법들의 효과성을 평가하였습니다. 연구자들은 다양한 CRL 접근법을 대표하는 방법들을 선택하였으나, 이러한 방법들이 기대했던 바와 달리 근본적인 인과 요소를 회복하지 못하는 결과를 도출하였습니다. 이는 기존 연구와는 다르게 실제 데이터 생성 과정을 다루며, 독립적인 기초 진리를 제시하여 이론적 탐색과 실제 응용 사이의 간극을 해소하려는 시도로 볼 수 있습니다.

- **Technical Details**: 이 실험은 광학적 실험을 기반으로 설계되었으며, 물리적 시스템의 데이터 생성 과정이 CRL의 핵심 가정을 충족하도록 구성되었습니다. 제어 입력은 RGB LED의 밝기와 두 개의 선형 편광자의 위치로 이루어져 있으며, 출력은 카메라로 촬영한 이미지와 물리적 양의 측정 데이터를 포함합니다. 연구자들은 서로 다른 세 가지 접근 방식에서 대표적인 방법을 평가했으며, 각각의 결과는 섹션 3에서 논의됩니다.

- **Performance Highlights**: 실험 결과는 대다수의 CRL 방법이 단순한 합성 자료에도 불구하고 일반적인 CRL 방법이 요구하는 근본적인 인과 요소를 복원하는 데 실패했다는 것을 보였습니다. 연구자들은 이러한 실패 양상을 분석하기 위해 실제 데이터 생성 과정을 간소화한 합성 대안으로 데이터 차원을 축소하였고, 이 과정에서 재현성 문제를 발견하였습니다. 이 연구는 CRL 방법의 이론적 약속과 실제 응용의 도전 과제를 강조하며, 앞으로 CRL 방법의 발전을 가속화할 수 있는 기초 자료를 제공하기를 희망합니다.



### RIZE: Regularized Imitation Learning via Distributional Reinforcement Learning (https://arxiv.org/abs/2502.20089)
- **What's New**: 새로운 Inverse Reinforcement Learning (IRL) 접근법을 소개합니다. 이 방법은 고정된 보상 할당의 한계를 극복하고 암묵적인 보상 정규화에서의 유연성을 증가시킵니다. 기존의 Maximum Entropy IRL 프레임워크를 제곱 형태의 temporal-difference (TD) 정규자와 훈련 중 동적으로 조정되는 적응형 목표로 확장하여, 보상 기능을 간접적으로 최적화하는 동시에 강화 학습 원리를 포함합니다.

- **Technical Details**: 우리의 방법은 MaxEnt IRL 프레임워크 아래에서 두 가지 방법론적 혁신을 통해 암묵적 보상 정규화를 확장합니다. 첫째, 훈련 중에 동적으로 조정되는 적응형 목표 보상을 도입하여 고정 목표를 대체하고, 둘째, 분포적 강화 학습(distributional RL)을 통합하여 보다 풍부한 반환 정보를 포착하며 Q-값과 이론적 일관성을 유지합니다. 이러한 접근법은 보상 학습의 이전 경직성을 해결하고, MuJoCo 벤치마크에서 온라인 IL 기준을 능가하는 성능을 보여줍니다.

- **Performance Highlights**: 우리의 접근법은 MuJoCo 작업에서 전문가 수준의 결과를 나타냅니다. 특히, Humanoid 작업에서는 단 3개의 데모로 최첨단 성능을 달성했습니다. 광범위한 실험과 ablation 연구를 통해 적응형 목표와 보상 동역학의 효과를 입증하며, 모방 학습에서의 심층적인 통찰을 제공합니다.



### A Generative Model Enhanced Multi-Agent Reinforcement Learning Method for Electric Vehicle Charging Navigation (https://arxiv.org/abs/2502.20068)
- **What's New**: 전기 자동차(EV)의 보급이 확대됨에 따라, EV 운전자가 비용 효율적인 충전소를 선택하기 위한 탐색 문제는 중요하면서도 도전적인 과제로 떠올랐습니다. 본 논문에서는 고급 심층 강화 학습(DRL) 알고리즘을 개선하여 EV의 로컬 정보만을 활용하면서 기존 알고리즘과 유사한 성능을 발휘하는 새로운 생성 모델 강화 멀티 에이전트 DRL 알고리즘을 소개합니다. 이 접근법은 EV 측에서 정책 네트워크를 구현하고 추천 정보를 제공하는 데 CVAE-LSTM 기반 모델을 개발하였습니다.

- **Technical Details**: 제안된 알고리즘은 충전 경쟁 문제를 효과적으로 해결하기 위해 글로벌 정보를 압축하는 새로운 미래 충전 경쟁 인코더를 설계했습니다. 또한, 다중 경량 강하 알고리즘(MGDA)을 활용하여 훈련 목표의 두 부분 간의 가중치를 적응적으로 조절하여 보다 안정적인 훈련 프로세스를 제공합니다. 시뮬레이션은 중국 시안의 실제 지역을 기반으로 수행되었으며, 로컬 정보 기반의 기존 방법들보다 우수한 성능을 발휘했습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 로컬 정보에 의존하면서도 글로벌 정보 기반 방법들과 비교해 8% 이내의 성능 저하를 기록했습니다. 이는 EV 운전자가 의사 결정을 내릴 때 반드시 다수의 전기 자동차와의 통신이 필요하지 않도록 하여, 통신 비용을 크게 줄이고 개인 정보 보호 문제를 완화할 수 있는 가능성을 보여줍니다. 전체 성능 비교는 표 1에서 확인할 수 있습니다.



### Recommendations from Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization (https://arxiv.org/abs/2502.20033)
Comments:
          42 pages, 1 figure

- **What's New**: 이 논문은 추천 시스템에서 사용자가 개별 항목을 평가하는 대신 항목 쌍을 비교함으로써 피드백을 제공하는 새로운 학습 문제에 대한 이론적 분석을 제공합니다. 이 작업은 사용자 및 항목의 잠재적 특징을 기반으로 비교로부터 선호도를 예측하는 과제로 축소됩니다. 논문에서는 sparse 데이터 상황에서도 컨벡스 손실 함수의 성질을 분석하여, 해당 방법의 장점을 입증합니다.

- **Technical Details**: 추천 시스템은 사용자 피드백을 비교 형식으로 수집하며, 각 사용자는 낮은 차원의 사용자 및 항목 특징 벡터의 내적을 기반으로 유틸리티를 생성합니다. 연구에서는 일반적인 행렬 완성에서 사용되는 농축 불평등을 추천 시스템 모델로 확장했습니다. 이러한 비선형 최적화 문제는 호기심을 유도하며, 고전적인 방법보다 연산적으로 효율적인 해결 방법을 제시합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 개인화된 추천을 학습하는 데 있어 계산적으로 효율적이고 통계적으로 효과적임을 보여줍니다. 특히, 적절한 초기값을 제공하면 경량 기반 방법이 전역 최소값으로 지수적으로 수렴함을 보였습니다. 이는 데이터가 희소할 때에도 강력한 성능을 보장합니다.



### Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping (https://arxiv.org/abs/2502.20032)
Comments:
          Accepted by the proceeding of CVPR2025

- **What's New**: 본 논문은 Class Incremental Learning (CIL) 분야에서 클래스의 순서가 모델 성능에 미치는 영향을 분석합니다. 저자들은 클래스 유사성이 낮을수록 모델이 클래스 순서에 더 강건함을 보인다는 이론적 분석을 제공합니다. 이와 함께, Graph-Driven Dynamic Similarity Grouping (GDDSG)이라는 새로운 방법을 제안하여 클래스 간 유사성을 기반으로 다이나믹하게 클래스를 그룹화합니다.

- **Technical Details**: GDDSG 방법은 그래프 색칠 알고리즘을 활용하여 유사성이 낮은 클래스 그룹을 형성하고, 각 클래스 그룹에 대해 독립적인 CIL 모델을 훈련합니다. 이 방식은 각 그룹의 모델을 결합하여 예측할 때 효과적으로 작동하며, 클래스 간 유사성이 높을수록 지속적인 학습 과정에서 발생하는 클래스 충돌을 완화합니다. 이를 통해 모델의 일반화 능력과 잊지 않음 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, GDDSG 방법은 클래스 순서에 대한 민감성을 효과적으로 해결하며 모델의 정확성과 방어적 목표를 달성했습니다. 기존의 CIL 방법들이 직면했던 문제들을 해결하고, 다양한 클래스 순서에서 안정된 성능을 유지할 수 있는 혁신적인 접근법으로 주목받고 있습니다. 저자들은 이 연구의 코드도 함께 공개하여 향후 연구자들이 활용할 수 있도록 지원하고 있습니다.



### Offline Reinforcement Learning via Inverse Optimization (https://arxiv.org/abs/2502.20030)
Comments:
          preprint

- **What's New**: 이번 연구에서는 다양한 응용 분야에서 인버스 최적화(Inverse Optimization, IO)의 최근 성공을 바탕으로 새로운 오프라인 강화 학습(Offline Reinforcement Learning, ORL) 알고리즘을 제안합니다. 이 알고리즘은 IO 문헌에서 사용되는 볼록 손실 함수(convex loss function)인 'sub-optimality loss'를 활용합니다. 또한, ORL 문제에서 일반적으로 관찰되는 분포 이동(distribution shift)을 완화하기 위해 강력하고 인과관계 없는 모델 예측 제어(Model Predictive Control, MPC) 전문가를 채택하여 동적 모델의 표준 모델을 유도합니다.

- **Technical Details**: 제안된 알고리즘은 기존 문헌과 달리, 정확하고 다뤄기 쉬운 볼록 리포밍(convex reformulation)을 가지고 있는 강력한 MPC 전문가를 특징으로 합니다. 이는 모델 불일치에 기초한 인사이트 정보를 사용하여 동작합니다. 연구의 두 번째 부분에서는, 제안된 볼록 손실 함수로 훈련된 IO 가설 클래스가 충분한 표현력을 가지며 최근의 최고 성능(State-of-the-Art, SOTA) 방법과 비교할 때 데이터를 적게 사용하면서도 경쟁력 있는 성능을 달성함을 보여줍니다.

- **Performance Highlights**: MuJoCo 벤치마크(low-data regime)에서 3배 이상의 파라미터를 덜 사용하면서도 경쟁력 있는 성과를 달성하였습니다. 이로 인해 계산 자원(computational resources)의 요구량도 크게 줄어듭니다. 연구 결과의 재현성을 촉진하기 위해 제안된 알고리즘 및 실험을 구현하는 오픈 소스 패키지(open-source package)를 제공하고 있습니다.



### Climate And Resource Awareness is Imperative to Achieving Sustainable AI (and Preventing a Global AI Arms Race) (https://arxiv.org/abs/2502.20016)
Comments:
          19 pages, 6 figures

- **What's New**: 이번 논문에서 발표된 내용은 지속 가능한 인공지능(Sustainable AI)에 대한 새로운 시각을 제시합니다. 기존의 연구들은 대개 환경 지속 가능성에 중점을 두었으나, 이 논문은 경제적 및 사회적 지속 가능성의 중요성을 강조합니다. 자원 인식(Resource awareness)과 기후 인식(Climate awareness)의 균형을 맞추는 것이 지속 가능한 AI 실현의 필수 조건이라고 주장합니다.

- **Technical Details**: 이 논문은 CARAML(Climate and Resource Aware Machine Learning) 프레임워크를 도입하여 기후와 자원 인식 간의 갈등을 해결하려고 합니다. 또한, 자료 및 인프라의 접근 가능성을 확대하여 AI 개발 리소스의 불균형을 개선하려고 합니다. 이러한 접근 방식은 개인, 커뮤니티, 산업, 정부 및 글로벌 수준에서의 실질적인 추천 사항을 제시합니다.

- **Performance Highlights**: 기술적으로 지속 가능한 AI를 달성하기 위해서는 기후 인식과 자원 인식의 상호 보완적인 접근이 필요합니다. 그러나 이 두 요인은 때때로 상충하는 경향이 있어, AI 모델 개발 시 환경적 지속 가능성을 고려하지 않으면 심각한 문제를 초래할 수 있습니다. 이 논문은 개발도상국의 접근 문제와 청정 에너지 자원의 부족이 AI 지속 가능성에 미치는 영향을 강조하면서, 글로벌 AI 생태계에서의 공정한 참여를 위한 개선점을 제시합니다.



### Learning Classifiers That Induce Markets (https://arxiv.org/abs/2502.20012)
- **What's New**: 이 논문은 전략적 분류 문제에서 사용자들이 긍정적인 예측을 얻기 위해 특성을 전략적으로 수정할 수 있다는 사실을 다룹니다. 기존의 모델에서는 수정하는 데 드는 비용이 외부적이고 고정되어 있다고 가정했지만, 본 연구는 이러한 가정에 도전합니다. 저자들은 분류기가 시장 수요를 생성하고, 그에 따라 특성의 비용이 시장에서 결정된다는 개념을 제안합니다.

- **Technical Details**: 이 연구에서는 ‘마켓-인식 분류’(market-aware classification)라는 개념을 소개하며, 사용자들이 긍정적인 예측을 추구할 때 중요한 특성에 대한 수요가 발생하는 메커니즘을 분석합니다. 제안하는 알고리즘을 통해 시장 가격을 효과적으로 계산하고, 이를 학습 파이프라인의 일부로 통합할 수 있는 방법을 제시합니다. 또한, 마켓 가격과 관련된 균형 개념을 특성 학습 과제와 함께 정의합니다.

- **Performance Highlights**: 본 연구의 주요 결과는 시장이 전통적인 전략적 분류 모델과는 상이한 복잡한 행동 패턴을 초래할 수 있다는 점입니다. 특히, 예산에 따라 사용자들이 다르게 행동할 수 있는 경향이 있으며, 이는 학습 결과에 영향을 미칠 수 있습니다. 최종적으로, 저자들은 이러한 기반 위에서 대칭적이지 않은 경제적 불평등과 공정성 문제에 대한 논의의 필요성을 제기합니다.



### Learning Hamiltonian Density Using DeepON (https://arxiv.org/abs/2502.19994)
- **What's New**: 최근 물리적 현상을 모델링하기 위한 딥러닝에 대한 관심이 증가하고 있습니다. 특히, Hamiltonian Mechanics를 학습하는 방법으로 Hamiltonian Neural Networks (HNNs)가 주목받고 있습니다. 그러나 기존 방법은 데이터의 이산화(discretization)에 의존하며, 필요한 미분 연산자를 결정해야 하는 단점이 있습니다. 본 연구에서는 파동 방정기를 모델링하기 위한 연산자 학습 방법을 제안합니다.

- **Technical Details**: 본 연구는 자동 미분(automatic differentiation) 알고리즘을 사용하여 방정식을 구성하는 데 필요한 변분 미분(variational derivatives)을 계산하는 방법을 제시합니다. 이러한 접근 방식을 통해 데이터의 이산화 없이도 파동의 Hamiltonian 밀도를 학습할 수 있습니다. 기존의 HNNs와 달리, 연산자 학습 방법은 미분 연산자나 그 이산화의 결정을 필요로 하지 않습니다. 실험 결과, 제안한 방법이 실제로 파동 방정식의 Hamiltonian 밀도를 학습할 수 있음을 보여주었습니다.

- **Performance Highlights**: 제안된 연산자 학습 방법은 파동 방정식에 대한 강력한 성능을 발휘합니다. DeepONet와 같은 신경 네트워크 아키텍처를 활용하여 고차원 문제를 효과적으로 해결할 수 있습니다. 특히 비선형 연속 연산자(nonlinear continuous operators)를 근사하는 데 탁월한 성능을 보여줍니다. 이는 파라메트릭 PDEs를 해결하는 데 있어서도 혁신적인 접근법으로 기대됩니다.



### Dam Volume Prediction Model Development Using ML Algorithms (https://arxiv.org/abs/2502.19989)
Comments:
          22 pages, 18 Figures and 4 Tables

- **What's New**: 이번 연구에서는 아프리카 남부의 Loskop 댐에서 물 자원 관리를 위해 머신 러닝 회귀 기법인 Gradient Boosting, Random Forest, ElasticNet을 이용하여 댐 성능 특성을 예측하는 방법을 탐구했습니다. 이 연구는 건조 및 반건조 지역에서 신뢰할 수 있는 저수량(reliable reservoir volume) 추정의 중요성을 강조하고 있습니다.

- **Technical Details**: 모델은 지리공간 고도 측정치와 그에 해당하는 저수지 공급 용량값을 쌍으로 이루는 데이터셋으로 교육 및 검증되었습니다. 최상의 성능은 높은 볼륨을 위해 Random Forest를 사용하고 낮은 볼륨을 위해 Ridge 회귀를 결합한 임계값 기반 스태킹 모델(threshold-based blended model)에서 나타났습니다.

- **Performance Highlights**: 이 모델은 RMSE(Root Mean Square Error) 4.88 MCM과 R2 0.99를 달성하여 댐 데이터셋에서 복잡한 관계를 포착하는 앙상블 학습 기법의 효과를 보여주었습니다. 이러한 결과는 실제 물 자원 관리 시나리오에서 신뢰할 수 있는 댐 성능 모델링을 위한 실용적 유용성을 강조하고 있습니다.



### WaveGAS: Waveform Relaxation for Scaling Graph Neural Networks (https://arxiv.org/abs/2502.19986)
- **What's New**: 이 논문에서는 그래프 신경망(Graph Neural Networks, GNNs)에서의 자원 제한 문제를 해결하기 위한 GNNAutoScale(GAS) 접근 방식 개선을 제안합니다. 특히, WaveGAS를 도입하여 과거의 임베딩을 예측하는 정확도를 높입니다. 새로운 기법은 효율적인 임베딩 업데이트를 통해 GNN의 성능을 크게 향상시킵니다.

- **Technical Details**: GAS는 그래프를 여러 부분으로 나누어 GPU 메모리의 한계를 극복하는 기술입니다. 이 방법은 특정 미니 배치의 노드와 직접 연결된 다른 미니 배치의 이웃 노드에서 과거 임베딩을 검색하여 사용합니다. 그러나 이전 훈련 반복에서 가져온 과거 임베딩은 현재 임베딩에 비해 오래되어 정밀도에 영향을 미치게 됩니다. WaveGAS는 여러 번의 순전파(forward pass)를 수행하여 이러한 스테일(staleness) 문제를 완화합니다.

- **Performance Highlights**: 실험 결과 WaveGAS는 GAS보다 더 높은 정확도를 기록하며, 전체 그래프 훈련 방법보다도 우수한 성능을 보였습니다. 이 개선된 방법은 메모리 소비를 줄이면서도 더 좋은 임베딩 결과를 만들어냅니다. 또한, 훈련 시간이 비례적으로 길어지는 단점이 있지만, 전반적인 성능이 증가하는 효용을 고려할 때 이를 상쇄할 수 있습니다.



### Efficient Time Series Forecasting via Hyper-Complex Models and Frequency Aggregation (https://arxiv.org/abs/2502.19983)
Comments:
          12 pages, 5 figures. Still awaiting conference submission approval

- **What's New**: 본 논문에서는 Frequency Information Aggregation (FIA)-Net이라는 새로운 시계열 예측 모델을 제안합니다. FIA-Net은 인접한 STFT(window) 집합에서 정보를 집계하는 복소수 MLP 아키텍처를 기반으로 하고 있습니다. 이 모델은 하이퍼-복소수(Complex) 벡터를 통해 윈도우의 정보를 효율적으로 결합하여 긴 범위 의존성을 처리하고, 기존 방식에 비해 파라미터 수를 최대 3배까지 줄일 수 있습니다.

- **Technical Details**: FIA-Net의 핵심 구성 요소에는 인접 STFT 윈도우를 혼합하는 WM-MLP와 하이퍼-복소수 대수를 활용하여 모든 STFT의 정보를 결합하는 HC-MLP가 있습니다. 모델의 학습 복잡도는 O(L log L/p)로, 여기서 L은 lookback window의 길이이고 p는 STFT 윈도우의 수입니다. 이 시스템은 주어진 M 주파수 성분을 유지하면서도 데이터의 정확성을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: FIA-Net은 여러 시간 시계열 벤치마크에서 평가되었으며, 기존의 시계열 예측 모델에 비해 정확성과 효율성에서 우수한 성능을 보였습니다. 특히, 모델은 기존 모델보다 최대 20% 더 높은 정확도를 구현했으며, 더 적은 파라미터로도 효과적인 예측이 가능한 것을 보여주었습니다. 다양한 실험과 함께 ablation study를 통해 복소평면에서의 작업 효과도 분석하였습니다.



### Can Textual Gradient Work in Federated Learning? (https://arxiv.org/abs/2502.19980)
Comments:
          Accepted at ICLR 2025

- **What's New**: 본 논문에서는 텍스트 기반의 그래디언트(Textual Gradient)를 연합 학습(Federated Learning, FL)에 통합하는 가능성과 도전 과제를 탐구합니다. 새로운 연합 텍스트 그래디언트(FedTextGrad) 패러다임을 도입하여, 클라이언트들이 텍스트 그래디언트를 기반으로 최적화된 프롬프트를 서버에 업로드할 수 있게 합니다. 이는 기존의 수치 집계 방식이 아닌 텍스트 데이터 처리에 맞춘 방법입니다.

- **Technical Details**: 이 방법은 프롬프트 집계 과정에서 중요한 정보를 유지하는 데 어려움을 겪는다는 주요 과제를 강조합니다. 연구 팀은 비대칭 정보 분포가 요인임을 발견하고, 균일 정보 밀도(Uniform Information Density) 원칙에 기반한 요약 방법을 도입합니다. 이를 통해 FedTextGrad의 성능을 개선하고, 중요한 내용을 유지하면서 짧은 프롬프트를 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 FedTextGrad가 전통적인 중앙 집중식 설정에서의 TextGrad에 비해 우수한 성능을 발휘할 수 있음을 보여줍니다. 하지만 집계에서 발생하는 프롬프트의 복잡성은 성능 저하를 가져올 수 있으며, 이러한 문제로 인해 FL 환경에서 텍스트 그래디언트를 도입하는 것이 현재 주요 구속 요인으로 작용합니다. 연구는 LLM 최적화를 위한 텍스트 그래디언트의 적용 가능성을 탐구하고 앞으로의 발전 방향을 제시합니다.



### Do Sparse Autoencoders Generalize? A Case Study of Answerability (https://arxiv.org/abs/2502.19964)
- **What's New**: 스파스 오토인코더(Sparse Autoencoders, SAE)는 언어 모델 해석 가능성에 있어 유망한 접근법으로 부상하고 있습니다. 본 연구에서는 '답할 수 있는 질문'을 인식하는 모델의 능력, 즉 'answerability'를 중점적으로 분석합니다. 기존 연구와 달리 사본(SAEs) 피처는 특정 도메인에서 높은 정확성을 보였으나, 일반화 성능이 크게 차이나는 것을 발견했습니다.

- **Technical Details**: 이 연구는 주로 'Gemma Scope'로 pretrained 된 SAE를 사용하여 언어 모델의 answerability를 탐색했습니다. 2k 샘플의 SQUAD 데이터를 사용하여 예측 가능한 SAE 피처를 찾기 위해 5-fold 교차 검증을 실시했으며, SAE와 개별 도메인에서의 성능을 평가했습니다. 평가 과정에서는 다양한 out-of-distribution 데이터셋을 이용하여 SAE 피처의 일반화 성능을 측정했습니다.

- **Performance Highlights**: 실험 결과, SAE 피처는 특정 도메인 내에서 0.8의 정확도를 달성했으나, 일반화 성능은 높지 않았습니다. 특히, 특정 분야의 질문에서만 SAE가 괜찮은 성능을 보이며, 다른 질문에서는 낮은 성능을 보였습니다. 전반적으로, SAE 기반의 해석 가능성 방법론이 더욱 정량적으로 효과를 예측할 필요성을 강조하며 이 연구의 결과를 통해 새로운 통찰력을 제공합니다.



### SeisMoLLM: Advancing Seismic Monitoring via Cross-modal Transfer with Pre-trained Large Language Mod (https://arxiv.org/abs/2502.19960)
Comments:
          13 pages, 6 figures. Code is available at this https URL

- **What's New**: 이 논문에서는 SeisMoLLM이라는 첫 번째 seismic monitoring을 위한 foundation model을 제안합니다. 이는 cross-modal transfer를 활용하여 특정 seismic 데이터셋에 대한 직적 조정( pre-training 없이도 대형 모델의 특징을 활용할 수 있습니다. 이 모델은 back-azimuth 추정, epicentral distance, magnitude estimation 등 5개의 주요 작업에서 state-of-the-art 성능을 발휘합니다.

- **Technical Details**: SeisMoLLM의 구조는 멀티 스케일 컨볼루션 임베더(multi-scale convolutional embedder), 잠재 패칭(latent patching), pre-trained LLM 블록 및 작업별 출력 헤드로 구성됩니다. 임베더는 seismic waveform의 여러 스케일에서 특징을 추출하며, 생성된 임베딩을 통해 데이터 양을 줄이면서도 중요한 정보를 보존합니다. LLM 블록은 이러한 토큰 시퀀스를 이용하여 고급 시퀀스 모델링을 수행합니다.

- **Performance Highlights**: SeisMoLLM은 43개 작업 메트릭 중 36개의 최상의 결과를 달성하고, 16개의 few-shot 일반화 메트릭 중 12개의 최고 점수를 기록하는 등 상당한 상대적 개선을 이뤘습니다. 또한, 경량 모델에 비해 교육 및 추론 효율성을 유지하거나 더 나은 성능을 보여주며, 이번 연구는 seismic monitoring을 위한 실용적인 foundation model로서의 가능성을 강조합니다.



### Machine-learning for photoplethysmography analysis: Benchmarking feature, image, and signal-based approaches (https://arxiv.org/abs/2502.19949)
Comments:
          39 pages, 9 figures, code available at this https URL

- **What's New**: 본 연구는 Photoplethysmography (PPG) 데이터를 이용하여 심박동과 심장 관련 중대한 건강 지표를 예측하기 위한 최신 머신러닝 기법에 대한 포괄적인 벤치마크 연구입니다. 독창적으로 세 가지 입력 표현 방식인 해석 가능한 특징, 이미지 표현 및 원시 파형을 포함하여 다양한 예측 과제를 비교하였습니다. 이 연구는 혈압(BP) 측정과 심방세동(AF) 분류라는 두 가지의 전형적인 임상 애플리케이션 활용 사례를 통해 이러한 방식의 효과를 평가했습니다.

- **Technical Details**: PPG는 비침습적으로 피부에 빛을 비추고 반사되거나 투과되는 빛의 양을 측정하는 기술입니다. 본 연구에서는 심층 신경망(Deep Neural Networks)과 최신 합성곱 신경망(Convolutional Neural Networks, CNNs)을 활용하여 원시 시간 시퀀스를 입력 표현으로 사용했을 때의 효과를 분석했습니다. 다양한 입력 표현 방식 간의 성능을 비교하기 위해 두 개의 대규모 데이터셋을 사용했습니다.

- **Performance Highlights**: 연구 결과, 원시 시간 시퀀스를 입력으로 사용하는 딥러닝 모델이 혈압과 심방세동 예측에서 뛰어난 성능을 보였습니다. 특히 현대적인 CNN 구조가 가장 우수한 결과를 나타냈으나, 작업 세팅에 따라 얕은 CNN 모델도 경쟁력을 보였습니다. 이러한 결과는 향후 PPG 데이터를 이용한 머신러닝 작업을 위한 중요한 통찰력을 제공할 것으로 기대됩니다.



### Dynamic DropConnect: Enhancing Neural Network Robustness through Adaptive Edge Dropping Strategies (https://arxiv.org/abs/2502.19948)
- **What's New**: 이 논문은 각 신경망 층의 엣지에 동적 드롭 비율을 할당하여 드롭 과정을 독특하게 조정하는 새로운 방법론인 DynamicDropConnect (DDC)를 소개합니다. DDC는 추가적인 학습 매개변수를 포함하지 않고도 드롭 비율을 조정할 수 있습니다. 실험 결과는 DDC가 기존의 Dropout, DropConnect 및 Standout보다 우수한 성능을 보여주는 것을 입증합니다.

- **Technical Details**: DDC는 엣지와 관련된 그래디언트의 크기에 따라 드롭 확률을 동적으로 할당합니다. 그래디언트가 큰 엣지는 학습에 중요하므로 유지되어야 하고, 미미한 영향만 미치는 엣지는 제거될 수 있습니다. 이 접근법은 모델 아키텍처를 단순화하고 메모리 요구사항을 줄이는 장점을 제공합니다.

- **Performance Highlights**: DDC는 합성 및 실제 공개 데이터셋을 사용한 실험을 통해 유의미한 결과를 나타냅니다. 특히, 작은 그래디언트의 엣지를 우선적으로 드롭하는 방법이 다른 방법들보다 빠르게 최소 손실에 도달하는 모습을 보여줍니다. 이는 DDC가 더 복잡한 네트워크에 적용될 때 실험 성능이 기대된다는 가능성을 시사합니다.



### Algebraic Machine Learning: Learning as computing an algebraic decomposition of a task (https://arxiv.org/abs/2502.19944)
- **What's New**: 이번 논문에서는 기계 학습의 기초를 대신하여 추상 대수(abstract algebra)를 기반으로 한 대안적인 방법론인 대수적 기계 학습(Algebraic Machine Learning, AML)을 제안합니다. AML은 문제와 데이터를 대수적 공리의 집합으로 인코딩함으로써 학습을 분석하고 이해할 수 있는 새로운 기회를 제공합니다. 이 접근법은 통계(statistics) 및 최적화(optimization)에 의존하지 않고도 직접적으로 훈련 데이터에서 일반화할 수 있는 능력을 가지며, 이는 기존의 기계 학습 방법과는 다른 점입니다.

- **Technical Details**: 대수적 기계 학습(AML)은 데이터, 목표 및 선행 지식에 의해 정의된 문제를 대수적 공리의 집합으로 인코딩합니다. 이후, 'Full Crossing'이라 불리는 절차를 통해 모델을 생성하며, 이 모델은 공리와 그 논리적 결과만이 진실인 '가장 자유로운 모델'입니다. 이 논문에서는 Sparse Crossing을 사용하여 일반화 subsets를 공리로부터 직접적으로 얻는 방법론을 제시하며, 추상 대수에서 알려진 부분 직접 분해(subdirect decomposition)를 통해 문제의 근본적 구성 요소(atom)를 찾는 방법도 설명합니다.

- **Performance Highlights**: 이런 새로운 학습 원리는 MNIST, FashionMNIST, CIFAR-10 및 의료 이미지 같은 표준 데이터셋에서 검증되었으며, 최적화된 다층 퍼셉트론(multilayer perceptrons)과 유사한 성능을 달성했습니다. 또한 이 방법은 단순한 데이터 기반 작업을 넘어 해밀토니안 사이클(hamiltonian cycle) 탐색과 같은 형식적 문제 해결에도 확장될 수 있습니다. 대수적 기계 학습은 훈련 데이터에서 직접 학습할 수 있는 새로운 관점을 제공하며, 검증 데이터셋 없이도 데이터의 근본 규칙으로 점근적 수렴(asymptotic convergence)할 수 있는 장점을 지니고 있습니다.



### Flexible Bivariate Beta Mixture Model: A Probabilistic Approach for Clustering Complex Data Structures (https://arxiv.org/abs/2502.19938)
- **What's New**: 이번 연구에서는 전통적인 클러스터링 알고리즘의 한계를 극복하기 위한 유연한 이변량 베타 혼합 모델(Flexible Bivariate Beta Mixture Model, FBBMM)을 제안합니다. 기존의 k-means 및 Gaussian Mixture Models (GMM)와 같은 알고리즘은 비볼록 클러스터를 처리하는 데 한계를 보였으며, FBBMM은 이변량 베타 분포의 유연성을 활용하여 다양한 형태의 클러스터를 모델링할 수 있습니다. 이를 통해, 복잡한 데이터 구조를 효과적으로 클러스터링할 수 있는 강력한 솔루션을 제공합니다.

- **Technical Details**: FBBMM은 Expectation Maximization (EM) 알고리즘과 Sequential Least Squares Programming (SLSQP) 최적 기법을 활용하여 매개변수를 추정합니다. 이 모델은 비볼록 클러스터와 부정적 상관관계까지 지원하여, 데이터 집합의 복잡한 구조를 보다 정확하게 포착 가능합니다. 유연한 이변량 베타 분포는 전통적인 클러스터링 방법의 한계를 극복하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과에 따르면, FBBMM은 비볼록 클러스터를 처리하는 데 있어 전통적인 모델보다 뛰어난 성능을 보였습니다. 이 모델은 데이터 포인트가 서로 다른 클러스터에 속할 확률을 부여하며, 이는 데이터 포인트의 소프트 클러스터링을 가능하게 합니다. FBBMM은 원본 데이터와 유사한 새로운 데이터 포인트를 생성할 수 있는 생성 모델로, 데이터 증강 및 시뮬레이션 작업에도 유용하게 활용됩니다.



### Lotus at SemEval-2025 Task 11: RoBERTa with Llama-3 Generated Explanations for Multi-Label Emotion Classification (https://arxiv.org/abs/2502.19935)
Comments:
          8 pages , submitted to SemEval 2025-Task 11

- **What's New**: 이번 논문에서는 Llama-3를 활용하여 다중 레이블 감정 탐지에 대한 새로운 접근 방식을 제안합니다. 이 방법은 모호한 감정 표현을 명확히 하는 설명적인 내용을 생성하여 RoBERTa의 감정 분류 성능을 향상시킵니다. 특히 두려움(fear), 기쁨(joy), 슬픔(sadness) 같은 감정에 대해 F1-score를 개선하며, 텍스트 전용 모델을 능가하는 성과를 보였습니다. 이 연구는 감정 탐지 과제가 직면한 여러 문제를 해결하는 중요한 진전을 나타냅니다.

- **Technical Details**: 이 연구는 Llama-3와 RoBERTa 모델을 조합하여 감정 분류 모델을 향상시키는 데 중점을 두고 있습니다. 최초 단계에서 Llama-3는 텍스트의 모호한 감정 표현에 대한 설명을 생성하며, 이를 통해 RoBERTa를 미세 조정(fine-tuning)하여 더 나은 다중 레이블 감정 분류를 가능하게 합니다. 비슷한 연구들에서 감정 표현을 다룰 때 자주 발생하는 도전 과제인 감정의 모호성, 다중 레이블 분류 및 불균형 데이터셋을 효과적으로 다루기 위한 접근 방식도 포함되어 있습니다.

- **Performance Highlights**: 본 연구는 SemEval 2025 Task 11의 서브태스크 1에서 다중 레이블 감정 탐지의 성능을 평가하였으며, Text + Explanation 모델이 Macro F1 점수 0.7396을 기록했습니다. 반면, 텍스트 전용 모델은 0.7112의 점수를 기록하여, 설명적 내용을 포함하는 것이 분류 정확도를 확실히 향상시킨다는 것을 보여주었습니다. 이러한 결과는 감정 탐지 작업에 대한 설명적 컨텍스트의 중요성을 강조하며, 다양한 감정 클래스 간의 정확도를 개선함을 입증합니다.



### Incremental Learning with Repetition via Pseudo-Feature Projection (https://arxiv.org/abs/2502.19922)
- **What's New**: 이 논문은 점진적 학습(incremental learning, IL)에서의 데이터 반복효과를 다루고 있습니다. 특히 반복 패턴이 내재된 새로운 시나리오를 제안하여 기존의 엄격한 반복 금지 규칙을 완화하고자 합니다. 이를 통해 기존 전략과 비교하여 더 현실적인 학습 환경을 구현하기 위한 방법론을 탐구하고 있습니다.

- **Technical Details**: 제안된 방법인 Horde는 독립적인 특징 추출기(feature extractor)의 앙상블을 동적으로 조정하며, 클래스 반복을 활용하여 이들을 정렬할 수 있습니다. 이 메소드는 기존의 예시 기반 접근 방식(exemplar-based approaches)과는 달리, 예시 없이 학습할 수 있는 능력을 지니고 있습니다. 특히, 본 연구에서는 기존의 IL 방법들을 벤치마킹하고 내재적인 데이터 반복의 영향력을 분석하고 있습니다.

- **Performance Highlights**: Horde 방법은 반복이 없는 전통적인 시나리오에서 경쟁력 있는 결과를 달성하며, 반복이 발생하는 조건에서도 최첨단 성능을 나타냅니다. 이는 실제 애플리케이션에서 발생할 수 있는 반복 데이터 문제를 해결하기 위한 중요한 기초를 마련합니다. 또한, IL이 반복을 포함할 때의 안정성과 적응성을 더욱 잘 이해할 수 있는 기반을 제공합니다.



### Shifting the Paradigm: A Diffeomorphism Between Time Series Data Manifolds for Achieving Shift-Invariancy in Deep Learning (https://arxiv.org/abs/2502.19921)
Comments:
          To appear at the International Conference on Learning Representation (ICLR) 2025

- **What's New**: 이번 논문은 시간이 지남에 따라 변화하는 시간 시퀀스 입력에 대한 deep learning 모델의 민감성을 해결하기 위한 새로운 접근법을 제안합니다. 기존의 shift invariance (변위 불변성) 확보 방법들이 시간 시퀀스 데이터에 대해 효과적이지 않다는 점을 입증하였고, 이를 해결하기 위한 혁신적인 differntiable bijective function (미분 가능한 쌍향 함수)를 소개합니다.

- **Technical Details**: 제안된 방법은 고차원 데이터 다양체에서 동일한 차원의 다른 다양체로 샘플을 매핑하여 데이터를 변형하는 방식으로 작동합니다. 이 함수를 통해 랜덤 변위가 적용된 샘플들이 동일한 공간의 고유한 점으로 매핑되며, 모든 작업 관련 정보가 손실 없이 보존됩니다. 이러한 특성은 model topology (모델의 토폴로지)를 변경하지 않고도 shift-invariance를 확보할 수 있게 합니다.

- **Performance Highlights**: 논문에서 실시한 실험은 6개의 시간 시퀀스 과제를 대상으로 하였으며, 제안된 방법은 모델의 성능을 일관되게 향상시켰습니다. 이 새로운 접근법은 변동성을 줄이고, 완전한 shift-invariance를 달성할 수 있음을 보여주어 기존 방법과 함께 사용 가능하다는 점에서 유용합니다. 실험 결과는 성능 개선을 이어가며 과거의 연구 성과와의 시너지 효과를 나타냅니다.



### Playing Pokémon Red via Deep Reinforcement Learning (https://arxiv.org/abs/2502.19920)
Comments:
          8 pages, 3 figures, 3 tables, under review

- **What's New**: 본 논문에서는 고전적인 Game Boy JRPG인 Pokémon Red를 에이전트의 테스트 베드로 활용하기 위한 새로운 환경을 소개합니다. 우리는 Deep Reinforcement Learning (DRL) 훈련 방법론을 도입하여, 게임의 초기 부분인 Cerulean City를 완수하는 기본 에이전트를 개발했습니다. 이 연구는 다수의 실험을 통해 보상 신호의 취약점을 드러내고, Pokémon과 같은 게임이 미래 연구에 미치는 잠재력에 대해 논의합니다.

- **Technical Details**: 연구는 여러 개의 하이퍼파라미터(hyperparameters)를 사용하여 머신 러닝 모델을 학습시킵니다. 이러한 하이퍼파라미터는 에이전트의 성능과 보상 형성(reward shaping)에 중요한 역할을 합니다. 다양한 ablation 실험을 통해 특정 보상 신호의 남용 가능성을 확인하였고, 이로 인해 모델의 제한점을 분석하였습니다.

- **Performance Highlights**: 실험 결과, 기본 에이전트는 Cerulean City를 성공적으로 완수했습니다. 그러나 보상 신호에 대한 의존성으로 인한 취약점이 발견되었으며, 이를 통해 에이전트의 성능 향상을 위한 방향성을 제시합니다. Pokémon과 같은 게임은 대형 언어 모델(Large Language Model) 에이전트, 계층적 훈련 알고리즘(hierarchical training algorithms), 그리고 고급 탐색 방법론을 위한 중요한 연구 자원으로 평가됩니다.



### SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks (https://arxiv.org/abs/2502.19913)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM) 훈련을 가속화하기 위해 전통적인 파이프라인 훈련 방식에 새로운 접근법인 SkipPipe를 제안합니다. SkipPipe는 파이프라인 단계의 건너뛰기 및 재배열을 통해 훈련 시간을 줄이고, 모델의 수렴(convergence)을 유지합니다. 이로써 LLM의 대규모 분산 훈련에서 통신 비용을 절감하고 성능을 개선하는 효과를 노립니다.

- **Technical Details**: SkipPipe는 훈련 데이터를 처리하는 마이크로배치(microbatch)에 대해 최적화된 경로 스케줄링(path scheduling)을 사용하여 각 마이크로배치가 특정 파이프라인 단계를 건너뛰도록 합니다. 시스템은 노드를 파이프라인 단계에 할당하고, 서로 다른 단계 간의 통신을 최소화하면서도 경량화된 방식으로 데이터 처리합니다. 이러한 구조는 훈련 환경의 동기화(synchronization) 빈도를 줄이고, 각 단계에서 발생할 수 있는 마이크로배치 충돌(collision)을 감소시킵니다.

- **Performance Highlights**: SkipPipe를 사용한 결과, 기존의 전체 모델 프레임워크와 비교하여 훈련 시간(iteration time)을 최대 60% 단축할 수 있음을 보여주었습니다. 또한, 훈련된 모델은 레이어 생략(layer omission)에 대한 저항성이 높아져, 모델의 절반만 실행했을 때도 perplexity가 단 7% 하락하는 형태로 높은 성능을 유지합니다.



### Graph Probability Aggregation Clustering (https://arxiv.org/abs/2502.19897)
- **What's New**: 본 논문에서는 Graph Probability Aggregation Clustering (GPAC)라는 새로운 그래프 기반 퍼지 클러스터링 알고리즘을 소개합니다. GPAC는 전통적인 클러스터링 방법의 한계를 극복하기 위해 글로벌 클러스터링 목표 함수와 로컬 클러스터링 제약 조건을 통합한 접근 방식을 취합니다. 기존의 클러스터링 기법들이 갖고 있는 문제점, 즉 데이터 군집간의 관계를 탐색할 때 발생하는 조잡한 분할을 해결하고자 합니다.

- **Technical Details**: GPAC는 다중 제약 조건 최적화 문제로 공식화되며, 라그랑주 방법을 사용해 이를 해결합니다. 이 과정에서, 샘플이 특정 클러스터에 속할 확률은 이웃 샘플로부터의 정보를 집계하여 반복적으로 계산됩니다. 또한, 최적화의 수렴성과 안정성을 개선하기 위해 하드 할당 변수를 목표 함수에 포함시켰습니다.

- **Performance Highlights**: GPAC는 합성 데이터셋, 실제 데이터셋 및 딥러닝 데이터셋을 포함한 폭넓은 실험을 통해 기존 최첨단 방법들과 비교했을 때 클러스터링 성능과 계산 효율성에서 뛰어난 결과를 보여줍니다. 특히, 대규모 데이터셋을 처리하는 데 있어 연산 복잡성을 제곱에서 선형으로 줄이는 가속 프로그램을 도입하여 확장성을 보장합니다.



### IL-SOAR : Imitation Learning with Soft Optimistic Actor cRitic (https://arxiv.org/abs/2502.19859)
- **What's New**: 이 논문에서는 모방 학습(imitation learning)을 위한 SOAR 프레임워크(SOAR framework)를 소개합니다. SOAR는 전문가의 시연(expert demonstrations)에서 정책을 학습하기 위한 알고리즘 템플릿(template)으로, 원시 쌍대( primal dual) 스타일 알고리즘을 사용하여 비용(cost)과 정책 업데이트(policy updates)를 번갈아 진행합니다. SOAR는 여러 평론가(critic)와 함께 행동평론가(actor-critic) 방법을 사용하여 탐사(exploration)를 촉진하는 낙관적인 평론가(optimistic critic)를 구축합니다.

- **Technical Details**: SOAR 프레임워크는 통계적 및 컴퓨팅적으로 효율적인 알고리즘을 제공합니다. 이 알고리즘은 앙상블 기반 탐사 기법을 활용하여 전문가의 궤적(expert trajectories) 수집과 관련된 복잡성을 줄입니다. MDP(마르코프 의사결정과정) 설정에서 SOAR는 개인화된 탐사 보너스를 제공하여 상태-행동 쌍(state-action pairs)의 방문을 장려하는 새로운 구조를 제안합니다.

- **Performance Highlights**: SOAR는 Soft Actor Critic(SAC) 기반의 여러 모방 학습 알고리즘의 성능을 일관되게 향상시키는 것으로 입증되었습니다. 특히, Coherent Soft Imitation Learning(CSIL), Maximum Likelihood IRL(ML-IRL)과 같은 기본 알고리즘에 SOAR를 적용하면 학습 에피소드 수를 절반으로 줄여도 동일한 성능을 달성할 수 있습니다. 이러한 결과는 MuJoCo 환경에서 SOAR의 효과가 뚜렷하다는 것을 보여줍니다.



### Revisit the Stability of Vanilla Federated Learning Under Diverse Conditions (https://arxiv.org/abs/2502.19849)
Comments:
          10 pages

- **What's New**: 본 논문에서는 데이터 프라이버시를 유지하면서 분산된 클라이언트 간의 협업 모델 훈련을 지원하는 Federated Learning (FL) 환경에서의 Vanilla FedAvg 알고리즘의 안정성을 다시 살펴봅니다. 고전적으로 단순한 개념으로 여겨지지만, FedAvg는 보다 발전된 FL 기술과 비교할 때도 매우 안정적인 성능을 보여줍니다. 다양한 FL 방법을 평가하고, Vision Transformer (ViT)를 이용하여 혈액 세포 및 피부 병변 분류 작업에 대한 성과를 포함한 수많은 실험을 수행하였습니다.

- **Technical Details**: FL 네트워크는 중앙 서버와 여러 클라이언트로 구성됩니다. 각 클라이언트는 고유의 로컬 데이터셋을 보유하며, 중앙 서버는 통신 라운드를 통해 이러한 클라이언트 데이터셋을 활용하여 전역적으로 공유되는 모델을 훈련합니다. 기본적인 FL 과정은 모델 파라미터를 전달하고 각 클라이언트에서 로컬 모델을 훈련하는 단계로 구성되어 있습니다. FedAvg 알고리즘은 이러한 과정에서 다른 복잡한 방법들보다 구현이 간편하고 오류에 덜 민감하여 사용됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 데이터셋, 분류 모델, 하이퍼파라미터 설정과 관계없이 FedAvg는 일관된 성능을 유지합니다. FedAvg는 경량한 하이퍼파라미터 조정 없이도 안정적인 결과를 제공하며, 자원 제약이 있는 병원에서의 의료 데이터 핸들링에 적합한 선택임을 입증하였습니다. 복잡한 FL 변형 방법들이 FedAvg를 초과할 수는 있지만, 그 최적의 하이퍼파라미터를 찾는 것이 어렵기 때문에, FeedAvg는 여전히 신뢰할 수 있는 기준선으로 남아있습니다.



### Knowledge Bridger: Towards Training-free Missing Multi-modality Completion (https://arxiv.org/abs/2502.19834)
Comments:
          Accepted to CVPR 2025

- **What's New**: 이번 연구에서는 외부 도메인에서의 일반화에 강한 결합과 자원 효율성을 겸비한 새로운 결측 모달리티 완성 모델을 개발하는 도전을 제기합니다. 이를 위해 'Knowledge Bridger'라는 훈련이 필요 없는 프레임워크를 제안하며, 대규모 다중 모달 모델(LMM)을 활용하여 결측 모달리티 생성을 지원합니다. 이 방법은 도메인 특화된 사전 지식을 정의함으로써, 주어진 모달리티에서 구조화된 정보를 자동으로 추출하여 지식 그래프를 구축합니다.

- **Technical Details**: 제안된 방법은 세 개의 주요 모듈로 구성됩니다: 지식 모델링 모듈, 지식 기반 모달리티 생성 모듈 및 순위 매김 모듈입니다. LMM을 사용하여, 이용 가능한 모달리티를 분석하고 CoT 접근 방식을 이용하여 주요 요소를 추출합니다. 지식 그래프를 통해 결측 데이터의 정확한 생성을 유도하고, 생성된 후보들 간의 유사성을 평가하여 가장 적합한 결과를 선택합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법이 일반 및 OOD 시나리오에서 결측 모달리티 완성 성능을 크게 향상시키는 것으로 나타났습니다. 또한, OpenAI의 GPT-4o를 사용했을 때, 성능 지표에서 72B 또는 7B 매개변수를 가진 모델보다 현저한 성과 향상을 달성했습니다. 이 연구는 다른 도메인에서의 적용 가능성과 함께 다양한 MMC 모델의 성능 향상에 기여할 수 있는 데이터 생성을 가능하게 합니다.



### GraphSparseNet: a Novel Method for Large Scale Trafffic Flow Prediction (https://arxiv.org/abs/2502.19823)
- **What's New**: 이 논문에서는 GraphSparseNet (GSNet)이라는 새로운 프레임워크를 소개하여 GNN 기반의 교통 흐름 예측 모델의 확장성(Scalability)과 정확성(Accuracy)을 향상시키고자 합니다. GSNet은 Feature Extractor와 Relational Compressor의 두 핵심 모듈로 구성되어 있으며, 이 모듈들은 선형 시간 및 공간 복잡성으로 운영되어 모델의 전체 계산 복잡성을 선형적으로 줄입니다. 이 접근법은 기존 방법들이 갖고 있는 복잡성 문제를 해결하며, 인기 있는 학습 모델들과 비교했을 때 학습 시간이 3.51배 단축됨을 보여줍니다.

- **Technical Details**: GSNet의 혁신적인 설계는 두 개의 모듈로 나뉘는데, Feature Extractor는 그래프 노드의 특징을 포착하고 인코딩하는 역할을 하며, Relational Compressor는 노드 간의 희소한 관계를 모델링합니다. 두 모듈 모두 선형 시간과 공간 복잡성으로 작동하여 그래프 노드 수가 증가함에 따라 효율적으로 확장할 수 있도록 구성되어 있습니다. 기존 방법들이 O(N²) 복잡성을 가진 것에 비해 GSNet는 O(N)으로 줄어들었습니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험을 수행한 결과, GraphSparseNet은 모든 평가된 데이터셋에서 두드러진 성과를 달성하였습니다. GSNet은 이전 상태의 Linear 모델과 비교하여 약 3.51배의 훈련 시간을 단축시키면서도 높은 예측 성능을 유지했습니다. 이러한 성과는 GSNet이 대규모 교통 시공간 데이터에 대한 확장성 문제를 효과적으로 해결했음을 보여줍니다.



### Advancing GDP Forecasting: The Potential of Machine Learning Techniques in Economic Predictions (https://arxiv.org/abs/2502.19807)
- **What's New**: 본 논문은 전통적인 경제 예측 방식에서 벗어나 기계 학습 방법을 활용한 GDP 예측의 가능성을 제시합니다. 특히, 비선형 관계와 데이터의 성격을 고려하여 LSTM (Long Short-Term Memory) 네트워크와 같은 순환 신경망(RNN) 모델을 사용합니다. 이러한 접근 방식은 기존의 경제계량 모델인 SARIMA와 비교되어 그 효과를 분석합니다.

- **Technical Details**: 연구에서는 1995년부터 2023년까지의 루마니아 분기별 GDP 데이터셋을 사용하여 LSTM 네트워크를 구축합니다. LSTM는 데이터의 복잡한 패턴과 상호작용을 모델링하는 데 강력한 성능을 보여주며, 전통적인 방식의 명시적 가정에 의존하지 않는 장점이 있습니다. 이 논문에서 다룬 LSTM 모델은 다음 4개의 분기 예측을 목표로 하고 있습니다.

- **Performance Highlights**: 연구 결과, 기계 학습 모델이 전통적인 경제계량 모델보다 예측 정확도와 유연성에서 일관되게 우수한 성능을 보인다는 것을 확인했습니다. 이는 GDP 예측에서 기계 학습의 효과성과 가능성을 뒷받침하는 중요한 결과입니다.



### Implicit Search via Discrete Diffusion: A Study on Chess (https://arxiv.org/abs/2502.19805)
Comments:
          ICLR 2025

- **What's New**: 이 연구는 AlphaGo 이후의 AI의 문제 해결 방식에서 새로운 관심이 생긴 탐색 기술에 대한 내용을 다룹니다. 특히 Monte Carlo Tree Search (MCTS)와 같은 검색 기법이 Large Language Models (LLMs)에 어떻게 적용될 수 있는지를 탐구합니다. 제안된 DiffuSearch라는 모델은 명시적 검색에 의존하지 않고도 다음 토큰(또는 행동) 예측을 향상시키기 위한 미래 세계를 포함하는 모델링을 진행합니다. 이를 통해 DiffuSearch가 기존의 정책보다 뛰어난 성능을 보여준다는 점이 강조됩니다.

- **Technical Details**: DiffuSearch는 여러 단계의 생성 과정을 통해 미래 정보를 예측하고 활용하는 과정을 포함합니다. 이 모델은 명시적 검색 없이도 확산 모델(diffusion model)을 통해 스스로 계획을 세우고 예측하는 능력을 가집니다. 특히 chess라는 고전 게임에 적용하여 상대방의 다음 수를 예측하는 정책을 학습하는 과정에서, 내부 양방향 자기 주의 메커니즘을 활용하여 현재 정책을 반복적으로 개선합니다. 이러한 구조는 명시적 검색이 필요 없도록 설계되어 있습니다.

- **Performance Highlights**: DiffuSearch는 제어된 실험에서 기존의 one-step 정책보다 19.2% 더 나은 행동 정확도를 보였으며, MCTS 기반 정책에 비해서는 14% 더 우수한 성과를 나타냈습니다. 또한, 퍼즐 해결 능력에서 30% 향상이 있었으며 Elo 등급에서는 540점을 증가시켜 게임 성능에서도 뛰어난 결과를 기록했습니다. 이런 결과들은 DiffuSearch가 명시적 검색 대신 미래 정보를 활용하여 다음 행동 예측을 향상시킬 수 있는 방법으로 작용할 수 있음을 나타냅니다.



### ServoLNN: Lagrangian Neural Networks Driven by Servomechanisms (https://arxiv.org/abs/2502.19802)
Comments:
          22 pages, 8 figures

- **What's New**: 이 논문에서는 servomechanisms(서보 메커니즘)으로 구동되는 동적 시스템을 모델링하는 새로운 아키텍처인 ServoLNN을 소개합니다. 기존의 신경망 아키텍처는 이러한 시스템을 적절히 모델링하지 못했으나, ServoLNN은 시점에 따라 주어진 구동 동작을 실시간으로 처리할 수 있습니다. 이를 통해 기존 방식보다 보다 효율적이고 정밀한 모델 생성을 가능하게 합니다.

- **Technical Details**: ServoLNN은 Lagrangian mechanics(라그랑주 역학)를 아키텍처에 하드코딩하여 훈련 시 주어진 시스템을 학습하도록 설계되었습니다. 또한, 이 구조는 PyTorch에서 구현되어 제공되며, 훈련이 수렴할 수 있는 다양한 해의 존재를 밝혀냅니다. 실험을 통해 예측 물리량에 대한 이러한 해의 영향을 분석하고, 다수의 해결책을 단일 해로 줄이는 해결 방법도 모색합니다.

- **Performance Highlights**: 이 아키텍처는 에너지, 전력, 작업 비율, 질량 행렬, 일반화된 가속도 및 일반화된 힘을 동시에 정확하게 계산할 수 있습니다. 특히, servomechanisms을 구동하기 위한 일반화된 힘을 잘 예측하여 동적 시스템 모델링에 있어 비약적인 발전을 이룹니다. 이는 실시간 애플리케이션에서의 유용성을 더욱 높이는 데 기여할 것입니다.



### Text classification using machine learning methods (https://arxiv.org/abs/2502.19801)
- **What's New**: 이 논문에서는 제품의 자동 분류에 활용할 수 있는 머신러닝 방법을 적용한 실험 결과를 발표합니다. 특히, 제품명을 텍스트 표현에서 수치 벡터로 변환하는 'word embedding' 과정을 통해 자동 분류 모델을 구축하였습니다.

- **Technical Details**: 여러 가지 embedding 방법을 사용하여 제품명을 수치 벡터로 변환하였습니다. 사용된 방법에는 Count Vectorization, TF-IDF, Word2Vec, FASTTEXT, GloVe가 포함되며, 이후 여러 머신러닝 기법인 Logistic Regression, Multinomial Naive Bayes, kNN, Artificial Neural Networks, Support Vector Machines, Decision Trees 등을 활용하여 자동 분류를 수행하였습니다.

- **Performance Highlights**: 결과적으로, Support Vector Machines, Logistic Regression, Random Forests의 분류 정확도가 인상적으로 높게 나타났습니다. 특히, 'word embedding' 방법 중 FASTTEXT 기법을 사용했을 때 가장 좋은 성능을 보였습니다.



### Mixtera: A Data Plane for Foundation Model Training (https://arxiv.org/abs/2502.19790)
Comments:
          under submission

- **What's New**: 본 논문에서는 Mixtera라는 새로운 데이터 플레인을 소개합니다. 이 시스템은 사용자가 훈련 중 필요한 데이터 샘플을 비율과 순서로 선언적으로 표현할 수 있게 해줍니다. Mixtera는 기존 훈련 데이터 컬렉션 위에 배포되는 중앙 집중식, 읽기 전용 레이어입니다.

- **Technical Details**: Mixtera는 파일시스템 구조와 독립적으로 작동하며, 다양한 속성(예: 언어, 소스 데이터셋)에 대해 혼합을 지원합니다. 또한 모델 피드백에 기반한 혼합의 동적 조정도 가능하여, 효과적인 데이터 샘플링을 실현합니다. 시스템은 Adaptive Data Optimization (ADO) 알고리즘을 구현하여 최신 혼합 전략의 발전을 지원합니다.

- **Performance Highlights**: Mixtera는 훈련 성능에 병목 현상을 일으키지 않으며, 256 GH200 슈퍼칩에까지 확장 가능하다는 것을 실험적으로 입증하였습니다. 논문에서는 Mixtera가 비전-언어 모델(vision-language models)에서도 중요한 역할을 하는 것을 탐색하였습니다.



### In-Context Learning with Hypothesis-Class Guidanc (https://arxiv.org/abs/2502.19787)
Comments:
          19 pages, 18 figures

- **What's New**: 본 연구에서는 인셉션 학습(불릿)에서 가설 클래스 유도(ICL-HCG)라는 새로운 합성 데이터 모델을 제안합니다. 기존 연구는 주로 라벨이 있는 예시로만 구성된 시퀀스에 집중했지만, 우리는 라벨 예시와 함께 포함된 지침이 학습에 미치는 영향을 분석합니다. 이 모델은 특정 가설 클래스의 문자적 설명과 그 가설로부터 선택된 $(x,y)$ 쌍을 입력 문맥으로 사용합니다.

- **Technical Details**: ICL-HCG 프레임워크 하에, 우리는 다양한 일반화 능력, 모델 아키텍처, 샘플 복잡도, 데이터 불균형, 지침의 역할, 그리고 pretrained 가설 다양성의 효과를 조사합니다. 특히, Transformer 모델이 ICL-HCG를 성공적으로 학습할 수 있으며, 이전에 보지 못한 가설 및 가설 클래스로 일반화할 수 있음을 보여줍니다. 이 과정에서 지침의 중요성도 강조됩니다.

- **Performance Highlights**: 결과적으로, 지침이 없는 ICL과 비교했을 때, ICL-HCG는 유의미하게 높은 정확도를 달성하여 지침이 학습 성능에 미치는 긍정적인 영향을 입증합니다. 이는 새로운 가설 클래스에 대한 일반화 능력과 더불어 모델의 성능을 크게 향상시키는 데 기여합니다.



### Obtaining Example-Based Explanations from Deep Neural Networks (https://arxiv.org/abs/2502.19768)
Comments:
          To be published in the Symposium on Intelligent Data Analysis (IDA) 2025

- **What's New**: 이번 연구에서는 딥 뉴럴 네트워크(DNN)에서 예제 기반 설명(example-based explanations)을 추출하기 위한 새로운 기술인 EBE-DNN을 제안합니다. 기존의 예제 기반 설명 기술은 주로 KNN 및 랜덤 포레스트와 같은 특정 모델에만 초점을 맞추고 있었으나, 이번 방법은 DNN의 강력한 특징 추출 기능을 활용합니다. EBE-DNN은 KNN 분류기를 사용하여 예측을 형성하고, 이를 통해 적은 수의 훈련 예제로 높은 집중도의 예제 귀속(example attribution)을 제공할 수 있습니다.

- **Technical Details**: EBE-DNN은 특정 DNN 레이어에서 추출한 임베딩(embedding)을 활용하여 테스트 예제의 예제 귀속을 생성하고 레이블을 예측하는 알고리즘입니다. 입력데이터는 모델에 따라 변환된 고차원 예제로, 이러한 변환은 저수준 및 고수준 특성을 반영하는 유용한 피처 공간으로의 전환을 목표로 합니다. 그런 다음, 변환된 공간에서 가장 가까운 k개의 이웃을 검색하고 이들로부터 예제 귀속을 산출하여 테스트 인스턴스의 예측 레이블을 생성합니다.

- **Performance Highlights**: EBE-DNN의 실험 결과는 원래 DNN의 정확도를 유지하면서도 극히 적은 수의 훈련 예제를 사용해 예측을 설명할 수 있음을 보여주었습니다. 또한, 임베딩을 생성하기 위해 선택한 레이어가 정확도에 큰 영향을 미친다는 중요한 발견이 있었습니다. 이 방법은 예측의 신뢰성과 투명성을 높여 사용자들이 심층 학습 모델을 신뢰하고 관리하는 데 기여할 수 있습니다.



### Learning with Exact Invariances in Polynomial Tim (https://arxiv.org/abs/2502.19758)
- **What's New**: 이번 연구는 커널 회귀(kernel regression) 맥락에서 정확한 불변성을 학습하기 위한 통계-계산적 절충(trade-offs)을 심층적으로 탐구합니다. 기존의 접근법들은 폴리노미얼 시간(polynomial-time) 해법을 제공하지 않거나 커널 환경에서 적용이 불가능했습니다. 그러나 우리는 입력 공간의 기하학적 특성에 대한 오라클 접근을 활용하여 정확한 불변성을 가진 분류기를 학습하는 폴리노미얼 시간 알고리즘을 제안하였습니다.

- **Technical Details**: 이 연구에서는 인공지능 모델이 데이터 내의 대칭성(symmetry) 및 불변성(invariance)을 인식하고 활용할 수 있도록 하는 방법을 탐구합니다. 전통적인 방법들은 데이터 증대(data augmentation) 및 그룹 평균(group averaging) 접근법들을 포함하지만, 이러한 접근은 계산적으로 비효율적일 수 있습니다. 본 연구는 스펙트럼 이론(spectral theory)과 최적화(optimization) 도구를 활용하여 커널 방법에서 불변성과 관련된 문제를 새로운 유한 차원 볼록 이차 프로그램(convex quadratic program) 집합으로 재구성하였습니다.

- **Performance Highlights**: 우리의 알고리즘은 기존 커널 회귀 문제와 동일한 과도한 모집단 위험(excess population risk) 또는 일반화 오류(generalization error)를 달성하는 동시에 통계적 및 계산적으로 효율적이라는 장점을 가지고 있습니다. 이 연구는 정확한 불변성을 달성하는 첫 번째 폴리노미얼 알고리즘을 도입하였고, 이는 신경망과 같은 다양한 응용 분야에서 중요할 수 있습니다. 또한, 이 해결책은 통계 및 계산 복잡성 사이의 균형을 맞추는 데 주안점을 두고 있으며, 초점은 정확한 불변성을 달성하는 것입니다.



### HALO: Robust Out-of-Distribution Detection via Joint Optimisation (https://arxiv.org/abs/2502.19755)
Comments:
          SaTML 2025

- **What's New**: 이 논문에서는 기계 학습 모델의 안전한 배치를 위해 중요한 OOD(Out-of-Distribution) 감지 방법의 강인성을 향상시키기 위해 HALO(Helper-based Adversarial OOD Detection)라는 새로운 접근 방식을 제안합니다. 기존의 TRADES 프레임워크를 조정하여 새로운 목표 함수를 발견하고, 클래스화 및 감지 성능을 향상시키는 추가 손실 항을 도입했습니다. HALO는 현재의 방법보다 뛰어난 성능을 보여주며 여러 데이터셋과 공격 환경에서 최첨단 성능을 달성합니다.

- **Technical Details**: HALO는 기존의 OOD 감지 프레임워크에 쉽게 적용 가능하며, 하이퍼파라미터 조정을 통해 성능을 최적화할 수 있는 장점이 있습니다. 이 방법은 ID(Distribution) 데이터와 OOD(Out-of-Distribution) 데이터에 대한 강한 강인성을 달성하는 데 초점을 맞추고 있으며, 공격 유형에 따라 성능의 저하 없이 클래스 작업을 수행할 수 있습니다. 실험 결과, HALO는 여러 데이터셋에서 AUROC(Area Under the Receiver Operating Characteristic Curve) 지표에서 평균 3.15의 향상을 이루었습니다.

- **Performance Highlights**: HALO는 이전의 OOD 감지 기술들에 비해 매우 경쟁력 있는 성능을 자랑합니다. 적대적 공격에 대해 7.07의 AUROC 개선을 보여주며, 다양한 공격 유형에 걸쳐 강인한 성능을 유지합니다. 또한, HALO는 전이 공격에 대한 저항력도 입증하였고, 다양한 공격 환경에서 일관된 성능을 발휘하는 것으로 나타났습니다.



### Probabilistic Federated Prompt-Tuning with Non-IID and Imbalanced Data (https://arxiv.org/abs/2502.19752)
Comments:
          Accepted at NeurIPS-24

- **What's New**: 이 논문은 머신 러닝 분야에서 사전 훈련된(pre-trained) 모델을 미세 조정하는 방법을 제안합니다. 기존의 연합 학습(federated learning) 방식이 서로 다른 로컬 데이터 분포로 인해 비효율적임을 지적하고, 더 효과적인 프롬프트 튜닝(prompt-tuning) 방법론을 통합하여 개선하였습니다. 이를 통해 연합 학습을 분산 세트 모델링(distributed set modeling) 작업으로 변환하여 정보의 요약을 통해 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 계층적 확률적 모델링(hierarchical probabilistic modeling)을 사용하여 로컬 프롬프트(prompt)의 생성과 정렬을 설명합니다. 서버 집계 과정(server aggregation step)은 유사한 문맥 정보를 부호화하는 로컬 프롬프트를 발견하고 집계하는 과정으로 설정됩니다. 결과적으로, 각 로컬 프롬프트 세트는 글로벌 요약 프롬프트(global summarizing prompts)로 초기화된 확률적 탐색(probabilistic exploration)으로 간주됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 실험을 통해 기존 연합 프롬프트 튜닝 방법들과 비교하였고, 극단적인 데이터 비균형(extreme data imbalance)과 이질성(heterogeneity) 상황에서도 가장 효과적인 성능을 나타냄을 확인하였습니다. 실험 결과, 제안된 방법이 연합 학습의 성능 향상에 기여하는 것을 보여줍니다. 이는 특히 컴퓨터 비전 데이터셋에서 효과적임을 입증하고 있으며, 모델의 안정성을 높이는 데 기여합니다.



### CirT: Global Subseasonal-to-Seasonal Forecasting with Geometry-inspired Transformer (https://arxiv.org/abs/2502.19750)
- **What's New**: 이 연구에서는 지리적 특성을 반영한 새로운 Circular Transformer (CirT) 모델을 제안합니다. 이 모델은 위도별로 날씨 데이터를 원형 패치로 분해하여 Transformer의 입력으로 사용하고, 자기 주의(self-attention)에서 Fourier 변환을 활용하여 전 세계 정보를 포착합니다. 이로써 기존의 데이터 기반 모델들이 가지고 있던 문제점들을 해결하고 성능을 크게 향상시킵니다.

- **Technical Details**: CirT는 (1) 위도를 기준으로 순환 패치로 날씨 데이터를 분해하고, (2) 주기적 특성을 고려하여 글로벌 정보를 모델링하는 방식으로 설계되었습니다. 이 모델은 고도 통일적인 패치 및 패치 조합에서 Fourier 변환을 사용하여 공간적 주기성을 고려하며, 결과적으로 예측의 정확성을 높입니다.

- **Performance Highlights**: CirT는 세계의 날씨 예측에 사용되는 여러 온라인 및 오프라인 시스템들보다 월등한 성과를 보여줍니다. 특히, CirT는 고위도 지역에서의 비율 문제를 해결하고, 지상 진실과의 구조적 일치를 향상시키며, 이전 연구들보다 더 나은 성능을 발휘합니다.



### BiRating -- Iterative averaging on a bipartite graph of Beat Saber scores, player skills, and map difficulties (https://arxiv.org/abs/2502.19742)
Comments:
          30 pages, 2 figures

- **What's New**: 이 논문은 Beat Saber 맵의 난이도 추정(difficulty estimation)을 위한 새로운 알고리즘을 제안합니다. 이 알고리즘은 플레이어의 스킬(skill)과 맵의 난이도 추정을 반복적으로 평균화하는 방식을 사용하여, 특정 맵에 대한 플레이어의 점수(score)만으로 작동합니다. 각 맵의 상대적인 점수를 통해 서로의 추정을 개선하는 동시에, 여러 점수 간의 관계를 활용합니다.

- **Technical Details**: 제안된 알고리즘은 플레이어의 스킬, 맵의 난이도, 점수 간의 관계를 정의하는 데 중점을 두고 있으며, 맵의 난이도 대신 쉽게 점수를 매길 수 있는 맵의 용이성(ease)을 도입합니다. 이 알고리즘에서는 점수가 0에서 1 사이의 값으로 표현되며, 비리니어(bilinear) 관계를 통해 반복 평균(iterative averaging)을 시행합니다. 이러한 설정은 알고리즘의 성공에 중요한 영향을 미칩니다.

- **Performance Highlights**: 논문에서는 알고리즘의 성능 평가 결과가 경험이 많은 Beat Saber 커뮤니티 멤버들과의 비공식적인 질적 평가를 통해 개인의 난이도 인식(perception)과 유사한 결과를 보였다고 설명합니다. 이 접근법은 기존 난이도 추정 방식에 비해 특정 문제가 있는 맵에서 현저한 개선을 보여주었으나, 점수에 대한 가정이 부적절할 경우 일부 맵에서는 어려운 추정을 발생시켰습니다. 향후 작업으로는 난이도 측정의 다차원성(multi-dimensionality)의 이해가 중요하다는 점이 강조됩니다.



### Causal Effect Estimation under Networked Interference without Networked Unconfoundedness Assumption (https://arxiv.org/abs/2502.19741)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.03342

- **What's New**: 이번 연구는 네트워크 간섭하에서 인과 효과를 추정하는 문제를 다룹니다. 기존 방법들이 관찰 데이터에 기반하고 네트워크의 비혼란 가정(networked unconfoundedness assumption)에 의존하는 반면, 본 논문은 이러한 가정이 일반적으로 위반된다는 점을 지적합니다. 연구팀은 개별 관측 값에 대해 전송하는 세 가지 유형의 잠재적 혼란 변수(latent confounders)를 식별하고, 이를 기반으로 효과 추정기를 개발했습니다.

- **Technical Details**: 저자들은 네트워크 간섭이 잠재적 혼란 변수를 식별하는 데 도움이 되는 보조 정보를 제공한다는 점을 강조합니다. 이를 통해 이론적으로 모든 잠재적 혼란 변수의 식별 가능성을 확립하고, 네트워크 효과의 식별 결과를 제시했습니다. 기존 비혼란 가정에 의존하지 않고, 효과적으로 혼란 변수를 식별할 수 있는 새로운 방법론을 제안하였습니다.

- **Performance Highlights**: 제안된 방법은 광범위한 실험을 통해 이론적 결과를 검증하였으며, 네트워크 효과 추정을 위한 새로운 접근 방식의 효과성을 입증했습니다. 이 연구는 인과 추론에서 네트워크 간섭의 영향을 고려하는 기존 연구와는 차별화된 시각을 제공하며, 보다 정확한 인과 효과 추정을 가능하게 합니다.



### FPGA-Accelerated SpeckleNN with SNL for Real-time X-ray Single-Particle Imaging (https://arxiv.org/abs/2502.19734)
- **What's New**: 이번 연구에서는 X-ray Single-Particle Imaging (SPI)에서 실시간 speckle 패턴 분류를 위한 SpeckleNN 모델의 특수 버전을 구현했습니다. 이 구현은 SLAC Neural Network Library (SNL)를 사용하여 FPGA 플랫폼에 최적화되었습니다. 모델의 매개변수가 기존 560만에서 6.46만(98.8% 감소)으로 줄어들었고, 90%의 정확도를 달성했습니다.

- **Technical Details**: 이 구현은 KCU1500 FPGA 보드에서 demonstrated 되었으며, 71%의 DSPs, 75%의 LUTs, 48%의 FFs를 사용하고 평균 소비 전력은 9.4W로 측정되었습니다. FPGA는 200 MHz의 클럭 주파수에서 45.015us의 지연으로 하나의 이미지에 대한 추론을 수행했습니다. 반면, NVIDIA A100 GPU에서 동일한 추론은 평균 73W의 전력을 소모하고 약 400us의 레이턴시를 보였습니다.

- **Performance Highlights**: 우리의 FPGA 버전은 GPU 구현에 비해 8.9배의 속도 향상과 7.8배의 전력 감소를 달성했습니다. SNL을 통한 모델 전문화 및 동적 가중치 로딩 기능은 FPGA 재합성의 시간을 소요하는 작업을 없애어 신속하고 지속적인 모델 배포를 가능하게 했습니다. 이러한 혁신은 실시간 적응형 분류와 효율적인 speckle 패턴 검증을 가능하게 하여 SpeckleNN을 XFEL 시설에 적합하게 만듭니다.



### Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training (https://arxiv.org/abs/2502.19726)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에 대한 멤버십 정보 추론 공격(Membership Inference Attacks, MIAs) 방어 메커니즘인 DuoLearn을 제안합니다. 최신 연구 결과에 따르면, 훈련 중 선택된 특정 토큰 집합을 사용하는 것이 모든 토큰을 사용하는 것보다 성능이 좋을 수 있다는 것을 발견했습니다. 이 연구는 토큰의 동적 선택 전략을 이용하여 MIAs에 대한 방어를 강화하고, 모델의 유틸리티를 유지하면서 계산 비용을 최소화하는 것을 목표로 합니다.

- **Technical Details**: DuoLearn은 훈련 중 토큰을 하드 토큰과 메모리화 토큰으로 분류하는 전략을 사용합니다. 하드 토큰은 높은 손실을 가진 토큰을 의미하며, 메모리화 토큰은 MIAs 신호가 강한 토큰을 가리킵니다. 이 논문에서는 하드 토큰에 대해 학습하고, 메모리화 토큰에 대해 비학습(unlearning)을 수행하는 이중 목적 손실 함수를 설계하여 모델이 유용한 정보를 학습하되 특정 훈련 샘플을 기억하지 않도록 합니다.

- **Performance Highlights**: 제안된 방어 메커니즘은 MIAs에 대해 강력한 보호를 제공하며, 다양한 LLM 아키텍처 및 데이터셋에서 언어 모델링 성능을 약 10% 향상시킵니다. 실험 결과, DuoLearn은 최소한의 성능 저하로도 언어 모델링에서의 메모리 문제를 효과적으로 완화하고, 학습 데이터를 보호하는 데 성공하였습니다.



### Accurate and Scalable Graph Neural Networks via Message Invarianc (https://arxiv.org/abs/2502.19693)
- **What's New**: 이 논문에서는 전통적인 message passing 방식의 그래프 신경망(GNNs)에서 발생하는 문제를 해결하기 위해 새로운 미니 배치 접근법인 topological compensation (TOP)을 제안합니다. TOP은 전체 message passing을 MP-IB만을 통해 효율적으로 계산함으로써 계산 비용을 줄이고 정확성을 유지합니다. 새롭게 정의된 message invariance 개념을 통해 기존의 비효율적인 방식에서 벗어나 빨라진 계산을 가능하게 합니다.

- **Technical Details**: TOP은 MP-OB를 필요로 하지 않고, message invariance를 통해 출발 노드에 대한 빠른 변환을 수행합니다. 이로 인해 더 많은 노드와 엣지를 GPU에 저장할 필요가 줄어들어 대규모 그래프에서도 GNN을 효율적으로 사용할 수 있게 됩니다. 이 접근법은 그래프의 다양한 특성과 이를 반영하는 선형 회귀를 사용하여 임베딩 간의 선형 독립성을 학습하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, TOP은 수백만 개의 노드와 수십억 개의 엣지를 가진 대규모 그래프에서 기존 미니 배치 방법보다 수십 배 이상 빨라지며 정확성 저하를 최소화합니다. TOP의 수렴 속도는 기존의 방법들에 비해 현저히 빠르며, 해당 연구가 다양한 실제 데이터 세트에서 유효성을 입증한 점도 주목받고 있습니다.



### The Future Outcome Reasoning and Confidence Assessment Benchmark (https://arxiv.org/abs/2502.19676)
- **What's New**: 이번 논문에서는 FOReCAst(Future Outcome Reasoning and Confidence Assessment)를 도입하여 예측 모델의 성능과 신뢰도를 평가합니다. 기존의 예측 벤치마크들이 종합적인 신뢰성 평가를 결여한 채 인공적인 질문에 중점을 두었다는 문제를 해결하고자 합니다. FOReCAst는 Boolean 질문, 시간대 예측, 수량 추정 등 다양한 예측 시나리오를 포함하여 현실 세계의 요구에 부합하는 평가를 제공합니다.

- **Technical Details**: FOReCAst는 세 가지 유형의 예측 질문으로 구분됩니다: (1) Boolean 질문에서는 이벤트 발생 여부에 대한 예측을 만들고, (2) 시간대 예측에서는 특정 날짜를 제공하며, (3) 수량 추정에서는 미래 사건에 대한 수치적 예측을 수행합니다. 각 질문 유형에 맞춰 모델이 예측한 내용과 신뢰도 점수를 산출하여 종합 평가를 진행합니다.

- **Performance Highlights**: 실험 결과, 현재의 대규모 언어 모델(LLMs)들은 여전히 예측 과제에 도전적이며 특히 신뢰도 평가에서 성능 향상이 나타나지 않았습니다. 예측 성능과 신뢰도 보정 사이의 직접적인 상관관계는 발견되지 않았고, 대형 모델이 간혹 성능을 개선하는 경향이 있는 반면, 효과는 일관되지 않았습니다. 따라서, FOReCAst는 더 나은 예측 품질과 신뢰도 평가를 위한 방향성을 제시하고 있습니다.



### Training Robust Graph Neural Networks by Modeling Noise Dependencies (https://arxiv.org/abs/2502.19670)
Comments:
          Work in progress

- **What's New**: 본 논문에서는 그래프 신경망(GNN)의 노이즈 모델링 접근법을 크게 개선하고자 하였습니다. 기존의 기존 GNN은 노이즈가 독립적이라는 가정을 바탕으로 구축되었지만, 우리는 실제 환경에서 노이즈가 그래프 구조 및 노드 레이블에 미치는 영향을 설명하는 의존성 있는 노이즈 시나리오인 DANG을 제안합니다. 이러한 새로운 모델인 DA-GNN은 데이터 생성 과정의 인과 관계를 포착하여 노이즈 의존성을 해결하며 우수한 성능을 자랑합니다.

- **Technical Details**: DANG에서는 노드 특성의 노이즈가 그래프 구조와 노드 레이블에 미치는 영향을 포착하기 위해 새로운 의존 관계를 통해 데이터 생성 과정(DGP)을 정의합니다. 논문은 이 과정을 기반으로 한 새로운 GNN 모델, DA-GNN을 제안하며, 변분 추론(Variational Inference)을 통해 학습 목표를 설정합니다. 또한, 유사한 노이즈 시나리오에서 검증할 수 있는 새로운 벤치마크 데이터셋도 공개하여 연구의 실제 적용 가능성을 높였습니다.

- **Performance Highlights**: DA-GNN은 기존의 베이스라인 모델에 비해 다양한 노이즈 시나리오에서 일관되게 우수한 성능을 발휘하였습니다. 특히, DANG 및 기존의 노이즈 모델을 포함한 실험을 통해 DA-GNN이 실제 응용에서 더 잘 일반화될 수 있음을 보여주었습니다. 이러한 방식으로 DA-GNN은 기존의 견고한 GNN 모델보다 더 넓은 적응성을 지니고 있으며, 실제 노이즈 시나리오에 대한 평가 기준을 제시합니다.



### Out-of-distribution Generalization for Total Variation based Invariant Risk Minimization (https://arxiv.org/abs/2502.19665)
Comments:
          ICLR 2025

- **What's New**: 이 논문에서는 최근 머신러닝의 중요한 틀인 Invariant Risk Minimization (IRM)을 새롭게 해석한 Total Variation 모델(IRM-TV)을 확장하여 Out-Of-Distribution (OOD) 일반화를 개선하는 Lagrangian multiplier 모델인 OOD-TV-IRM을 제안합니다. OOD-TV-IRM은 주어진 환경에서의 불변 위험을 최소화하고 TV 패널티를 강화하는 쌍대 최적화 모델을 기반으로 합니다. 이로 인해 모델의 일반화 능력을 높이고, 더 많은 환경에서 효과적으로 작동할 수 있도록 합니다.

- **Technical Details**: 이 모델은 Primal-Dual 최적화 구조를 가지고 있습니다. Primal 최적화는 전체 불변 위험을 줄이고, Dual 최적화는 TV 패널티를 강화하여 스퍼리어스 특징(spurious features)에 대한 적대적 간섭을 제공합니다. OOD-TV-IRM의 목표는 훈련 손실(training loss)과 OOD 일반화 간의 균형을 유지하는 Semi-Nash 균형에 도달하는 것입니다. 또한, 우리는 적대적 학습 방식에 적합한 수렴하는 Primal-Dual 알GORITHM을 개발했습니다.

- **Performance Highlights**: 실험 결과에 따르면, OOD-TV-IRM은 대부분의 상황에서 IRM-TV보다 더 나은 성능을 보였습니다. 이는 OOD 환경에서의 모델의 강건성과 일반화 성능을 크게 향상시키는 데 기여합니다. OOD-TV-IRM은 머신러닝 방법을 미지의 환경에서도 보다 효과적으로 적용할 수 있도록 하는 구체적인 구현 방안을 제공합니다.



### Variation Matters: from Mitigating to Embracing Zero-Shot NAS Ranking Function Variation (https://arxiv.org/abs/2502.19657)
- **What's New**: 이 논문은 Neural Architecture Search (NAS)의 제로샷 버전을 소개하며, 아키텍처를 훈련 없이 비교하는 빠른 순위 함수의 변동성을 효과적으로 해결하는 새로운 접근 방식을 제안합니다. 제안된 방법은 순위 함수를 랜덤 변수로 간주하여 성능 메트릭을 확률적으로 정렬하는 것입니다. 이를 통해 표준 벤치마크 검색 공간에서 아키텍처 검색 성능을 향상할 수 있음을 실험적으로 입증하였습니다.

- **Technical Details**: 제로샷 NAS는 두 가지 주요 구성 요소로 이루어져 있습니다: 순위 함수와 검색 알고리즘입니다. 순위 함수는 개별 아키텍처의 품질을 나타내는 스칼라 값을 산출하여 서로 비교 가능하게 합니다. 이 논문에서는 Eigenvalue score, ReLU Hamming distance 및 NTK의 condition number와 같은 순위 함수와 진화 검색 및 무작위 검색과 같은 다양한 검색 알고리즘을 평가합니다.

- **Performance Highlights**: 제안된 확률적 순위 정렬 방법은 표준 아키텍처 검색 공간에서 무작위 검색과 진화 검색의 성능을 80% 이상의 경우에 개선하는 것으로 나타났습니다. 이 연구는 순위 함수의 변동성을 명확히 분석하고, 지능형 아키텍처 검색을 위한 새로운 방법론을 제공합니다. 특히, 이 방법은 기존의 아키텍처 평가 방식보다 더 나은 결과를 제공합니다.



### Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning (https://arxiv.org/abs/2502.19652)
- **What's New**: 이 논문에서는 강인한 강화 학습(robust reinforcement learning, RL)을 위한 통합 모듈형 벤치마크인 Robust-Gymnasium을 소개합니다. 이 벤치마크는 블랙박스 아키텍처로 다양한 혼란이 발생할 수 있는 다양한 RL 컴포넌트를 지원합니다. 60개 이상의 제어 및 로봇, 안전 RL, 다중 에이전트 RL 관련 작업 환경을 제공하며, 현재의 방법을 평가하고 강인한 RL 알고리즘 개발을 촉진할 수 있는 도구입니다.

- **Technical Details**: Robust-Gymnasium은 단일 에이전트 RL 문제를 유한 수명 마르코프 결정 과정(finite-horizon Markov decision process, MDP)으로 정의합니다. 이 프레임워크는 에이전트-환경 상호작용에서 다양한 유형의 혼란을 통합할 수 있도록 설계되었습니다. 추가적인 혼란 모듈이 도입되어 에이전트의 상태 관측 및 환경의 변화에 따른 불확실성을 모델링합니다.

- **Performance Highlights**: 실험 결과, 기존의 표준 및 강인한 RL 알고리즘은 복잡한 작업에서 기대에 미치지 못하며, 특히 단일 단계의 혼란에서도 부족함을 보여줍니다. Robust-Gymnasium은 모든 단계에서의 혼란을 포함하여 다양한 작업을 범위에 두고 있는 혁신적인 도구로, 대규모 언어 모델(large language model, LLM)을 활용한 적대적 모델을 이용하여 RL 연구에 새로운 가능성을 보여줍니다.



### Unlocking Multi-Modal Potentials for Dynamic Text-Attributed Graph Representation (https://arxiv.org/abs/2502.19651)
- **What's New**: 이번 연구에서는 Dynamic Text-Attributed Graphs (DyTAGs)의 다중 모달리티를 모델링하기 위한 새로운 모델인 MoMent를 제안합니다. MoMent는 노드 중심의 접근 방식을 사용하여 기존의 에지 중심 모델링을 넘어서 동적 그래프 모델과 통합할 수 있는 기능을 갖추고 있습니다. 이를 통해 전체적인 구조적 특성을 최대한 활용하고 특정 모달리티 별로 노드 표현을 강화합니다.

- **Technical Details**: MoMent는 temporal, textual, structural의 세 가지 모달리티를 고려하여 노드 중심으로 모델링합니다. 각 모달리티에 대해 비공유 노드 중심 인코더를 설계하여 전역적 타이밍 및 의미 맥락을 포착하며, 대칭 정렬 손실(symmetric alignment loss)을 통해 깊이 있는 컨텍스트 방향성의 일관성을 보장합니다. 이 모델은 이를 통해 효과적으로 모달리티를 융합하고, 노드 표현을 강화하여 예측 성능을 높입니다.

- **Performance Highlights**: 실험 결과, MoMent는 기존의 DTGB 프레임워크에 비해 평균 6.06%의 성능 향상을 보여줍니다. 또한 고급 데이터 세트와 두 개의 다운스트림 과제를 통해 최대 33.62%의 향상을 기록함으로써 MoMent의 우수성을 입증합니다. 이러한 결과는 실제 응용 프로그램에서의 DyTAGs 활용 가능성을 크게 확장시킬 것으로 기대됩니다.



### Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models (https://arxiv.org/abs/2502.19649)
- **What's New**: Representation Engineering (RepE)는 LLM의 동작을 제어하는 새로운 패러다임으로, 기존의 입력 수정이나 모델의 미세 조정(fine-tuning) 대신 모델의 내부 표현을 직접 조작합니다. 이 접근 방식은 사용자가 원하는 대로 모델의 동작을 보다 효과적이고 해석 가능하며 유연하게 만들 수 있습니다. 본 논문은 RepE에 대한 최초의 포괄적 조사(workshop)로, 존재하는 다양한 RepE 방법과 그 차이점을 검토하고, RepE가 적용된 개념과 문제를 다룹니다.

- **Technical Details**: RepE는 개념 식별(concept identification), 운영화(operationalization), 제어(control)로 구성된 파이프라인을 통해 작동합니다. 첫 번째 단계는 대상 개념이 모델 내에서 어떻게 표현되는지를 식별하는 것이며, 두 번째 단계는 이 정보를 사용하여 새로운 입력에 대한 모델의 표현을 조작하는 것입니다. 이 과정에서 개념 조작자(concept operator)는 모델의 개념 표현을 정확히 캡처하는 객체로, 활성화 또는 가중치를 조작하는 데 사용됩니다.

- **Performance Highlights**: RepE는 훈련 예제가 적은 경우에도 효과적이며 샘플 효율(sample efficiency)이 높습니다. 또한 레이블이 없는 데이터셋을 요구하지 않는 여러 RepE 방법도 있으며, 이는 RepE의 사용을 더 용이하고 비용 효율적으로 만듭니다. 마지막으로, RepE는 모델의 전반적인 성능을 크게 저하시킬 위험이 적고, 요청에 따라 동적으로 조정할 수 있어 개인화 및 상황에 따른 제어가 가능합니다.



### cMIM: A Contrastive Mutual Information Framework for Unified Generative and Discriminative Representation Learning (https://arxiv.org/abs/2502.19642)
Comments:
          A working draft

- **What's New**: 이번 논문에서는 다운스트림 작업에 유용한 표현을 학습하는 새로운 방법인 cMIM(Contrastive Mutual Information Machine)을 소개합니다. cMIM은 Mutual Information Machine(MIM) 프레임워크와 새롭게 제안된 contrastive learning loss를 통합하여, 보다 효과적으로 표현을 학습하도록 설계되었습니다. 이 방법은 데이터 증강 데이터(augmentation) 없이도 효과적인 표현 학습을 가능하게 하여, 다양한 부정 샘플 수의 변화에도 강합니다. 또한, 이 논문에서는 사전 훈련된 encoder-decoder 모델에서 정보가 풍부한 임베딩(embeddings)을 추출하는 일반화된 방법도 제시합니다.

- **Technical Details**: cMIM은 MIM의 목적 함수를 augment하여 다소 비슷한 샘플의 latent codes는 가깝고, dissimilar samples의 latent codes는 서로 차별화되도록 합니다. 이를 통해 전체적인 latent space의 구조를 개선하여, 다운스트림 작업에서의 성능 향상을 도모합니다. 논문에서는 대칭적이고 효율적인 학습을 위해 contrastive learning 기술이 적용됩니다. 특히, 코사인 유사도(cosine similarity)와 같은 유사도 측정 기능을 사용하여, 긍정 쌍과 부정 쌍 간의 유사성을 극대화하고 최소화합니다.

- **Performance Highlights**: 제안된 cMIM 방법은 기존의 MIM보다 다운스트림 작업에서의 효과적인 성능 향상을 입증하였습니다. 특히, cMIM은 generative과 discriminative 작업 모두에 유용한 통합 모델로 기능합니다. 실험 결과에 따르면, cMIM으로 학습된 표현들은 다운스트림 작업에서 매우 유용하며, MIM의 생성 능력도 유지됩니다. 이로 인해 cMIM은 다양한 실제 문제에 적용 가능한 가능성을 가지고 있습니다.



### Developing robust methods to handle missing data in real-world applications effectively (https://arxiv.org/abs/2502.19635)
Comments:
          This work was presented at the ECML PKDD 2024 PhD Forum. https://ecmlpkdd. org/2024/program-accepted-phd-forum/

- **What's New**: 이 논문은 다양한 유형의 결측 데이터(missing data)에 관한 새로운 연구 방향을 제시합니다. 특히, 이미 많이 연구된 MCAR(Mechanism Completely At Random) 외에도 MAR(Missing At Random)과 MNAR(Missing Not At Random) 메커니즘의 중요성을 강조하며, 이들에 대한 연구가 상대적으로 부족하다고 지적합니다. 다양한 결측 데이터 메커니즘에 대한 이해를 심화하고자 하는 이 PhD 프로젝트는 실제적인 솔루션을 제공하려는 목표를 가지고 있습니다.

- **Technical Details**: 논문에서는 결측 데이터 메커니즘의 차이를 고려한 강력한 방법론(devising robust methodologies)을 개발하는 것을 주요 목표로 합니다. 이는 MCAR, MAR, MNAR 각각의 독특한 특성을 처리할 수 있는 기법을 포함하여 전체적인 데이터의 신뢰성을 향상하기 위한 접근법입니다. 결측 데이터가 다양한 산업과 데이터 유형에서 미치는 영향을 심도 있게 분석하고자 합니다.

- **Performance Highlights**: 이 연구는 결측 데이터 관리의 실제적인 솔루션을 제공하므로 연구자와 실무자가 불완전한 데이터셋(incomplete datasets)을 보다 효과적으로 활용할 수 있도록 돕습니다. 미래의 연구와 실무 응용에 있어 결측 데이터 메커니즘을 좀 더 명확히 인식하고 이를 통한 데이터 처리의 향상 가능성을 제시합니다. 결측 데이터 문제를 다루는 데 있어 혁신적인 접근법이 될 것으로 기대됩니다.



### Treatment Non-Adherence Bias in Clinical Machine Learning: A Real-World Study on Hypertension Medication (https://arxiv.org/abs/2502.19625)
- **What's New**: 이 연구는 전자 건강 기록(EHR) 데이터를 활용하여 고혈압 치료 비순응(treatment non-adherence)의 영향을 분석합니다. 3,623명의 환자를 대상으로 한 데이터 분석을 통해, 약물 비순응 환자 786명(21.7%)을 찾고 이와 관련된 주요 인구통계학적 및 임상적 요인을 밝혔습니다. 이 연구는 대규모 언어 모델(LLM)을 사용하여 임상 노트에서 처리 비순응 정보를 추출하는 혁신적인 방법을 제시합니다.

- **Technical Details**: 이 연구는 LLM을 활용하여 15,002명의 고혈압 환자의 자료를 분석했으며, 3,623명의 최종 환자 그룹을 도출했습니다. 연구는 10가지 일반적인 고혈압 약물의 처방을 기준으로 하여, 첫 번째 방문에서 약물 처방이 있었던 환자 샘플을 선정했습니다. 결과적으로 임상 노트에서 비순응의 이유를 추출하기 위한 주제 모델링(topic modeling) 기법도 적용하였습니다.

- **Performance Highlights**: 본 연구는 치료 비순응이 인과 추론(causal inference)과 예측 모델 성능에 미치는 부정적인 영향을 뚜렷하게 나타냅니다. 비순응을 무시할 경우 예측 모델 성능이 최대 5%까지 저하될 수 있으며, 이는 특히 소외된 인구 집단에 불리한 영향을 미칠 수 있음을 강조합니다. 이 연구의 결과는 치료 비순응을 고려하지 않는 것의 위험성을 경고하며, 보다 공정하고 책임 있는 임상 머신러닝 시스템 개발의 필요성을 제시합니다.



### PRDP: Progressively Refined Differentiable Physics (https://arxiv.org/abs/2502.19611)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이번 논문에서는 동적 프로세스를 통해 신경망의 정확도를 높이는 새로운 접근 방식인 Progressively Refined Differentiable Physics (PRDP)를 소개합니다. 전통적인 방법은 높은 정확도를 추구하는 반면, PRDP는 교육 과정에서 발생하는 연산 비용을 줄이기 위해 물리적 해결책의 정밀도 조절을 통해 이를 해결하고자 합니다. PRDP는 반복적인 선형 해법을 통해 과학적 컴퓨팅의 기본적인 부분을 대상으로 하며, 이로 인해 학습 시간의 단축을 가져옵니다.

- **Technical Details**: 논문은 반복적인 선형 해법(Iterative Linear Solvers)을 미분하는 이론적 이해를 바탕으로 하고 있습니다. 여기서 파라미터화된 선형 시스템을 다루며, 물리 모델의 기초가 되는 선형 시스템의 해결 과정을 분석합니다. PRDP는 한 단계에서 시작하여 데이터에 따라 정밀도를 점진적으로 조정하며, 최적의 정밀도 수준을 자동으로 파악합니다.

- **Performance Highlights**: 실험 결과, PRDP를 적용한 경우 다양한 물리적 문제에 대해 훈련 시간이 약 62% 단축되었습니다. 이는 학습 과정 중 물리적 해결책의 정밀도를 조정함으로써 가능해졌습니다. PRDP는 역문제, 자기 회귀 신경 모델, 수정 기반 하이브리드 솔버 등 다양한 형태의 신경망 학습 시나리오에서 효과적으로 검증되었습니다.



### Learning Ensembles of Interpretable Simple Structur (https://arxiv.org/abs/2502.19602)
- **What's New**: 이 논문에서는 기존의 복잡한 기계 학습 모델이 예측에 대한 이유를 모호하게 만드는 문제를 해결하기 위해 간단한 구조를 식별하는 알고리즘을 제안합니다. 이 알고리즘은 데이터를 이해 가능한 하위 그룹으로 나누어 각 그룹에서 간단한 모델을 학습하여 예측 정확성과 해석 가능성을 동시에 개선하는 것을 목표로 합니다. 이 접근법은 복잡한 데이터 내에서 본질적인 패턴을 드러내어, 예측 모델을 현업에 적용하는 데 필요한 투명성을 제공합니다.

- **Technical Details**: 제안된 알고리즘은 데이터셋을 간단한 구조로 알려진 해석 가능한 하위 그룹으로 파티셔닝하여, 각 그룹 내에서 최소한의 특징 상호작용을 유지하며 간단한 모델을 학습할 수 있도록 합니다. 이를 통해 로지스틱 회귀(Logistic Regression)와 같은 해석 가능한 모델을 조합하여 복잡한 모델과 유사한 성능을 달성하되, 해석력을 유지합니다. 알고리즘의 견고성을 확보하기 위해 실제 데이터의 복잡성에 대응하는 휴리스틱(heuristic)을 도입해 간단한 구조의 정의를 위반하는 문제를 처리합니다.

- **Performance Highlights**: 논문에서는 제안된 알고리즘을 합성 데이터와 실제 오픈 소스 UCI 및 Kaggle 데이터셋에서 평가하여 기존 접근법과 비교 분석의 결과를 보여줍니다. 모델이 간단한 구조로 학습되었을 때 얻은 통찰력이 전체 데이터셋으로 학습했을 때와 어떻게 다른지를 강조합니다. 이를 통해 제안된 방법이 해석 가능성과 정확도를 효과적으로 균형 있게 유지하면서 결정 경계를 보다 명확하게 도출할 수 있음을 입증합니다.



### Introduction to Sequence Modeling with Transformers (https://arxiv.org/abs/2502.19597)
Comments:
          10 pages, 1 figure

- **What's New**: 이 논문은 트랜스포머 아키텍처의 구성 요소들을 이해하는 데 중점을 두고 있습니다. 주된 초점은 'tokenization', 'embedding', 'masking', 'positional encoding', 'padding'과 같은 컴포넌트를 개별적으로 구현하여 각각의 역할을 파악하는 것입니다. 논문은 이를 통해 기본적인 트랜스포머 모델의 작동 방식을 점진적으로 분석합니다.

- **Technical Details**: 배경 섹션에서는 머신러닝(Machine Learning)과 시퀀스 모델링(Sequence Modeling)에 대한 기본적 개념과 용어를 다룹니다. 감독 학습(Supervised Learning) 문제에서 입력 x를 출력 y로 매핑하는 모델 f를 찾는 것이 핵심이며, 일반적으로 회귀(Regression) 및 분류(Classification) 방법이 사용됩니다. MLP(Multilayer Perceptron)가 널리 사용되던 시절, 여러 층으로 완전 연결된 신경망을 통해 문제를 해결했음을 강조합니다.

- **Performance Highlights**: 방법 섹션에서는 PlainTransformer라는 이름으로 알려진 기본 트랜스포머 모델의 성능 향상을 여러 단계로 보여 줍니다. 트랜스포머가 유일한 처리 요소로 활용되는 모델을 통해 Seq2Seq 모델링에서의 능력을 점차적으로 향상시키는 방법을 설명합니다. 이 논문은 torch.nn.Transformer 모듈과 관련된 구현 세부사항도 다루어, 다양한 입력과 그라디언트를 자동으로 처리할 수 있는 기능에 대해 논의합니다.



### Improving Representation Learning of Complex Critical Care Data with ICU-BER (https://arxiv.org/abs/2502.19593)
Comments:
          Accepted for poster at GenAI4Health Workshop at AAAI 2025

- **What's New**: 이번 논문에서 제안하는 ICU-BERT는 복잡한 ICU 데이터를 처리하기 위해 설계된 변환기(Transformer) 기반 모델이며, 대규모 MIMIC-IV 데이터베이스에서 사전 훈련되었습니다. 기존의 AI 기반 의사결정 지원 시스템의 한계를 극복하기 위해 최소한의 전처리로 강력한 표현을 학습합니다. ICU-BERT는 생물의학 대형 언어 모델(BioBERT)에서 밀집 임베딩을 활용하여 다변량 데이터를 일반화할 수 있는 대표성을 제공합니다.

- **Technical Details**: ICU-BERT는 각 의료 기록 항목을 하나의 토큰으로 처리하며, 멀티토큰 입력 전략을 사용하여 의료 데이터 흐름의 복잡성을 효과적으로 캡처합니다. 이 모델은 의료 변수를 분석하기 위한 새로운 마스킹 작업과 멀티태스크 학습 손실을 도입하였으며, 이는 임상 변수를 효과적으로 재구성할 수 있도록 돕습니다. ICU-BERT는 MIMIC-IV 데이터베이스 전반에 걸쳐 사전 훈련되며, 이로 인해 복잡한 ICU 데이터의 해석을 개선하고 다양한 실제 임상 응용에 잘 적응할 수 있게 됩니다.

- **Performance Highlights**: 예비 평가 실험에서 ICU-BERT는 실세계에서의 다양한 도전 과제를 통해 강력한 일반화 성능을 보여주었습니다. 듀엣(DuETT) 및 YAIB 프레임워크에서 다섯 가지 과제를 통해 미세 조정(fine-tuning)되었으며, 기존 모델들과 비교했을 때 뛰어난 성능을 기록했습니다. ICU-BERT는 의사결정 지원 시스템의 혁신적 변화를 이끌어낼 가능성을 제시하며, 여러 임상 작업에서 성능 기준을 초과할 수도 있습니다.



### LORENZA: Enhancing Generalization in Low-Rank Gradient LLM Training via Efficient Zeroth-Order Adaptive SAM (https://arxiv.org/abs/2502.19571)
- **What's New**: 본 연구는 대형 언어 모델(LLM)을 위한 강건한 파라미터 효율적인 미세 조정(PEFT) 기법을 다룬다. 기존 PEFT 방법의 약점을 보완하기 위해 AdaZo-SAM이라는 새로운 프레임워크를 제안하며, 이는 Adam과 Sharpness-Aware Minimization(SAM)을 결합하여 계산 효율성을 높인다. 또한, 메모리 효율성을 고려해 LORENZA라는 저랭크 그래디언트 최적화 기법을 설계하여 높은 정확도를 유지하면서도 자원 소모를 줄였다.

- **Technical Details**: 이 논문에서 제안하는 LORENZA는 메모리 효율적인 zeroth-order 샤프니스 최소화를 사용하여 적응형 저랭크 그래디언트 업데이트를 활용한다. 기존의 두 번의 역전파 방식이 필요하지 않아 계산 및 메모리 복잡성을 대폭 줄일 수 있다. LORENZA는 낮은 랭크의 서브스페이스를 동적으로 선택하여 효율적인 최적화가 이루어지도록 설계되었다.

- **Performance Highlights**: 실험 결과, LORENZA는 LLM의 사전 학습 및 미세 조정에서 기존 최첨단 기법보다 높은 정확도와 향상된 일반화를 달성했다. 기존의 저랭크 적응 기법인 LoRA와 GaLore보다 메모리 소모를 현저히 줄이면서도 다양한 데이터셋과 어려운 작업에서 강건한 적응력을 보여주었다. 이는 자원 제약 하에서 LLM 훈련 및 미세 조정의 효율적인 대안으로 자리매김할 가능성을 보여준다.



### PhenoProfiler: Advancing Phenotypic Learning for Image-based Drug Discovery (https://arxiv.org/abs/2502.19568)
- **What's New**: PhenoProfiler는 이미지 기반 약물 발견 분야에서 세포의 표현형 반응(phenotypic response)을 효율적으로 추출하기 위해 개발된 혁신적인 모델입니다. 기존 방법들이 복잡한 다단계 절차로 인해 비효율성과 오류 가능성을 증가시키는 문제를 해결합니다. PhenoProfiler는 기존 방법에서 요구되는 많은 계산 단계를 제거하여 전체 슬라이드 다채널 이미지를 직접 낮은 차원의 정량적 표현으로 전환합니다.

- **Technical Details**: PhenoProfiler는 엔드투엔드(end-to-end) 도구로 설계되어 있으며, 전체 슬라이드 다채널 이미지를 처리하여 표현형 변화를 효과적으로 설명할 수 있는 형태로 변화시킵니다. 또한, 다목적 학습 모듈(multi-objective learning module)을 포함하여 형태학적 표현 학습(morphological representation learning)에서의 강건성(robustness), 정확도(accuracy), 일반화(generalization)를 향상시킵니다.

- **Performance Highlights**: PhenoProfiler는 230,000개 이상의 전체 슬라이드 다채널 이미지와 8.42백만 개 이상의 단일 세포 이미지로 구성된 대규모 공개 데이터셋에서 검증되었습니다. 이 평가에서 PhenoProfiler는 최첨단(state-of-the-art) 방법에 비해 최대 20%의 성능 향상을 보여주었으며, 생물학적으로 의미 있는 신호를 탐지하기 위한 맞춤형 표현형 보정 전략을 사용합니다. 치료 프로필의 UMAP 시각화는 유사한 생물학적 주석을 가진 치료들을 효과적으로 군집화하는 PhenoProfiler의 능력을 잘 보여주며, 해석 가능성을 높입니다.



### Extremely Greedy Equivalence Search (https://arxiv.org/abs/2502.19551)
- **What's New**: 본 논문에서는 인과 발견(causal discovery) 문제에서 잘 알려진 알고리즘인 Greedy Equivalence Search (GES)의 한계를 분석하고, XGES라는 새로운 알고리즘을 제안합니다. GES는 유한한 데이터에서 강력한 이론적 보장을 갖지만, 복잡한 경우에는 성능이 저하되는 문제를 안고 있습니다. XGES는 GES의 탐색 전략을 개선하는 새로운 휴리스틱(heuristic)을 도입하여 로컬 옵티마(local optimum) 문제를 줄이고, 빠른 성능을 보여줍니다.

- **Technical Details**: XGES는 초기 탐색 단계에서 엣지를 삭제하는 것을 선호하여 그래프의 조밀함에 영향을 받지 않고, 더 안정적인 성능을 제공합니다. 또 하나의 기여는 GES와 XGES의 효율적인 알고리즘 수식을 개발한 점입니다. 우리는 XGES를 사용할 때 강력한 이론적 보장을 유지하면서도 GES보다 더 빠르고 신뢰할 수 있는 결과를 기대할 수 있습니다.

- **Performance Highlights**: 시뮬레이션된 데이터셋에서 XGES는 올바른 그래프를 복원하는 데 있어 GES보다 일관되게 우수한 성능을 보여주었으며, 속도는 10배 더 빠른 결과를 기록했습니다. XGES는 그래프 간의 조건부 독립성 관계를 정확하게 학습할 수 있어, 기존의 대안들과 비교하여 명확한 성능 향상을 입증하고 있습니다.



### Generalist World Model Pre-Training for Efficient Reinforcement Learning (https://arxiv.org/abs/2502.19544)
- **What's New**: 본 논문에서는 로봇 학습을 위한 효율적인 샘플 활용을 위해 보상 없는 보편적 다중 구현(embodiment) 오프라인 데이터를 사용하는 새로운 접근법을 제안합니다. 기존 방법들이 전문가 또는 보상 레이블이 붙은 작업별 데이터를 필요로 하는데 비해, 이러한 비선택적(non-curated) 데이터는 더 비용 효율적이며 실제 환경에서의 적용 가능성을 높입니다. 이를 통해 WPT(세계 모델 사전 훈련)가 효율적인 강화 학습과 빠른 작업 적응을 가능하게 한다고 강조합니다.

- **Technical Details**: 비선택적 자료(non-curated data)를 활용하여 WPT(세계 모델 사전 훈련) 기법을 발전시키고, 경험 리허설(experience rehearsal) 및 실행 가이드를 추가함으로써 더욱 안정적이고 효율적인 강화 학습을 이끌어냅니다. WPT는 다양한 임무를 포괄하고, 다양한 구현(embodiment)에서 발생하는 픽셀 기반 연속 제어 작업에 대한 실험을 통해 성능을 검증하였습니다. 이러한 접근법은 성능 향상뿐만 아니라 샘플의 효율성을 대폭 높였습니다.

- **Performance Highlights**: 72개의 비주얼 모터(visual-motor) 작업을 평가한 결과, WPT는 표준 샘플 수에서 이전 방법들보다 각각 35.65% 및 35% 더 높은 점수를 기록했습니다. 예를 들어, 앤트 로봇이 100회 실험 내에 전진하는 데 성공했으며, 기존의 10-30배 샘플보다 적은 수의 샘플로도 가능합니다. 이러한 결과는 WPT가 비선택적 오프라인 데이터를 활용했을 때 매우 우수한 성능을 보여줍니다.



### High-fidelity Multiphysics Modelling for Rapid Predictions Using Physics-informed Parallel Neural Operator (https://arxiv.org/abs/2502.19543)
Comments:
          10 pages, 11 figures, 1 table, 36 equations

- **What's New**: 이번 연구에서는 비선형 및 강하게 결합된 편미분방정식(nonlinear and strongly coupled partial differential equations, PDEs)을 기반으로 하는 복잡한 다물리 시스템(modelling complex multiphysics systems)의 모델링을 위해 새로운 물리 정보 기반 병렬 신경 연산자(physics-informed parallel neural operator, PIPNO)를 제안합니다. 이 방법은 데이터 없이도 PDE 모델링을 가능하게 하며, 조합 학습(ensemble learning)을 포함한 병렬 커널 통합 설계를 통해 계산 효율성을 대폭 향상시킵니다.

- **Technical Details**: PIPNO는 물리 법칙을 이용하여 스케일 가능한 비지도 학습(framework that enables data-free PDE modelling by leveraging only governing physical laws)을 수행합니다. 다양한 물리 엔지니어링 분야인 지반 공학(geotechnical engineering), 재료 과학(material science), 전자기학(electromagnetism), 양자 역학(quantum mechanics), 유체 역학(fluid dynamics)에서 비선형 연산자 매핑(nonlinear operator mappings)을 효과적으로 캡처합니다.

- **Performance Highlights**: PIPNO는 고충실도(high-fidelity)와 빠른 예측을 가능하게 하며, 기존의 연산자 학습 접근 방식보다 더 뛰어난 성능을 보여줍니다. 이 방법은 전통적인 해법을 대체할 수 있는 강력한 대안을 제공하며, 다물리법 모델링의 효율성, 강인성, 확장성(scalability)을 보장하여 신경 연산자의 응용 범위를 넓혔습니다.



### Retrieval Augmented Anomaly Detection (RAAD): Nimble Model Adjustment Without Retraining (https://arxiv.org/abs/2502.19534)
Comments:
          6 pages, 3 figures. 2 tables, accepted at ISDFS 2025

- **What's New**: 이 논문에서는 실시간 피드백 메커니즘을 통해 이상 탐지 모델의 잘못된 긍정 반응(false positive)을 줄이는 방법을 제안합니다. 제안된 기술은 행동 네트워크 이상 탐지 모델의 경량 배포를 위해 설계되었습니다. 이 방법론은 높은 처리량(throughput)을 요구하는 유사 도메인에 쉽게 통합될 수 있습니다.

- **Technical Details**: 논문에서는 Retrieval Augmented Anomaly Detection (RAAD)이라는 새로운 방법론을 소개합니다. 이 방법은 사용자가 직접 피드백을 제공하여 모델의 출력을 조정할 수 있는 기능을 제공합니다. RAAD는 실시간으로 피드백을 적용함으로써 예측 결과를 개선하고, 이후에 재교육을 위한 데이터 수집 과정을 단순화할 수 있는 장점이 있습니다.

- **Performance Highlights**: RAAD는 이미지, 텍스트, 그래프 기반 데이터 등 여러 데이터 양식에서 성능을 평가하여 그 일반화 가능성을 검증했습니다. 다양한 모델 아키텍처와 다중 데이터 모달리티에서 벤치마킹을 통해 이 방법이 효과적임을 입증했습니다. 이를 통해 기존의 딥러닝 기반 이상 탐지 시스템 보다 낮은 잘못된 긍정률을 유지할 수 있음을 강조하였습니다.



### Analyzing Cost-Sensitive Surrogate Losses via $\mathcal{H}$-calibration (https://arxiv.org/abs/2502.19522)
- **What's New**: 이 논문은 머신 러닝 모델을 훈련할 때 비용 민감한 대리모델(cost-sensitive surrogates)과 비용 비민감한 대리모델(cost-agnostic ones) 사용의 효과를 분석합니다. 연구 결과, 특정 분포 가정 하에서 작은 모델을 학습할 때 비용 민감한 대리모델이 더 우수하다는 사실이 드러났습니다. 이로 인해 비용 민감한 대리모델의 실제적 활용 가능성에 대한 강력한 주장을 제기합니다.

- **Technical Details**: 이 연구는 $	extmath{H}$-보정($	extmath{H}$-calibration)의 관점에서 분석을 진행하며, 일반적인 분포 가정 하에서의 모델 학습을 강조합니다. 비용 민감한 대리모델이 비용 비민감한 대리모델보다 일관되게 더 우수한 성능을 보인다는 점은 UCI 데이터셋을 통해 명확히 나타났습니다. 이러한 연구는 머신 러닝 시스템에서의 비용 고려의 중요성을 더욱 부각시킵니다.

- **Performance Highlights**: 비용 민감한 대리모델은 UCI 데이터셋에서 우수한 분류 성능을 보여주며, 연속적인 성능 개선을 제공합니다. 이러한 성과는 머신 러닝 응용 프로그램에서 보다 효율적인 모델 학습을 가능하게 하는 데 기여할 수 있습니다. 따라서, 연구는 비용 민감한 방법 적용의 실용성을 위한 강력한 사례를 제공하며, 미래 연구에 중요한 기초를 마련합니다.



### Mixtraining: A Better Trade-Off Between Compute and Performanc (https://arxiv.org/abs/2502.19513)
- **What's New**: 이 논문에서는 자가 지도 학습(self-supervised learning, SSL)과 감독 학습(supervised learning, SL) 사이에 새로운 훈련 프레임워크인 MixTraining을 제안합니다. MixTraining은 두 개의 학습 목표 간의 매끄러운 전환을 특징으로 하며, SSL과 SL의 여러 에포크를 통합하여 훈련 효율성을 높입니다. 이 접근 방식은 데이터가 제한된 환경에서도 높은 정확도를 달성할 수 있도록 지원하며, 성능과 계산량 간의 균형을 개선합니다.

- **Technical Details**: MixTraining 프레임워크는 일반적으로 사용되는 SSL+SL 프레임워크의 두 단계인 자가 지도 학습 단계와 감독 학습 단계를 병합하여 새로운 mixtraining 단계를 생성합니다. 이 mixtraining 단계는 자가 지도 및 감독 목표의 공동 업데이트를 가능하게 하여 두 목표 간의 균형을 맞추게 합니다. 이 방식은 두 학습 목표 간의 최적화를 더 밀접하게 연결하여, 모델이 안정적으로 목표 작업에 적응할 수 있게 합니다.

- **Performance Highlights**: 실험 결과 MixTraining은 전통적인 SSL+SL 파이프라인에 비해 현저한 성능 향상을 보여줍니다. TinyImageNet 데이터셋에서 8.81%의 절대 정확도 증가와 18.89%의 상대 정확도 증가를 달성하며, 훈련 속도는 최대 1.29배 향상됩니다. 데이터가 제한될 경우 MixTraining은 10% 제한 수준에서 105.58%의 상대 정확도 증가를 달성하여, 더욱 두드러진 성과를 보여줍니다.



### TRIX: A More Expressive Model for Zero-shot Domain Transfer in Knowledge Graphs (https://arxiv.org/abs/2502.19512)
- **What's New**: 이번 논문에서는 여러 도메인에서 훈련할 수 있는 완전 유도 지식 그래프 모델인 TRIX를 소개합니다. 이 모델은 기존의 최첨단 방법들과 비교하여 더 표현력이 뛰어난 triplet 임베딩을 제공합니다. TRIX는 새로운 능력으로, 유도 환경에서 엔티티와 관계 예측 작업을 동시에 처리할 수 있습니다. 이러한 기능 향상은 지식 그래프의 기초 모델 개발에 중요한 기여를 합니다.

- **Technical Details**: TRIX는 입력 지식 그래프를 기반으로 관계 그래프를 구성하여, 각 노드는 원래 그래프에서의 관계를 나타내고, 엣지는 그런 관계에서 공유되는 엔티티를 나타냅니다. 기존의 유도 모델들은 엔티티 예측 작업에만 초점을 맞추었으나, TRIX는 관계 예측 작업을 효율적으로 처리할 수 있는 기능을 갖추고 있습니다. 관계 예측 쿼리를 단일 전방 패스에서 처리할 수 있어, 기존 모델들이 요구하는 여러 번의 전방 패스 없이도 관계 예측을 수행할 수 있습니다.

- **Performance Highlights**: TRIX는 57개의 지식 그래프 데이터셋에서 엔티티와 관계 예측 모두에서 기존 방법보다 뛰어난 성능을 보여줍니다. 또한 기존의 대형 언어 모델들이 그래프 정보를 제대로 활용하지 못하는 한계를 보여주며, 새로운 도메인에서의 작업 수행에 있어 여전히 개선의 여지가 있음을 드러냅니다. TRIX의 향상된 표현력은 기존 유도 모델들에 비해 실험적으로 광범위하게 입증되었습니다.



### Models That Are Interpretable But Not Transparen (https://arxiv.org/abs/2502.19502)
Comments:
          AISTATS 2025

- **What's New**: 이 논문은 FaithfulDefense라는 혁신적인 설명 생성 방법을 제안합니다. 이 방법은 모델의 진정한 결정 경계를 드러내지 않으면서도 이론적으로 신뢰할 수 있는 설명을 제공합니다. 이러한 접근법은 기계 학습(ML) 모델의 의사 결정 과정을 투명하게 하고, 모델의 기밀성이 손상되지 않도록 합니다.

- **Technical Details**: FaithfulDefense는 최대 집합 덮개(maximum set cover) 문제에 기반하여 개발되었습니다. 이 방법은 사용자가 할당한 예산을 모두 사용하지 않더라도 설명에 대한 추가 조건(term)을 추가하여 모델에 대한 정보를 어떻게든 지킬 수 있습니다. 서브모듈성(submodularity)을 활용하여, 다양한 최적화 포뮬레이션을 제시합니다.

- **Performance Highlights**: 본 연구는 세 가지 공격자 전략과 여섯 가지 설명 방법을 사용한 실증 평가를 통해 FaithfulDefense의 효과를 입증했습니다. 알고리즘은 기계 학습 모델의 구조를 보호하는 데 있어 우수한 성능을 보여주며, GitHub를 통해 구현된 코드도 공개되었습니다. 이를 통해 연구자와 개발자들이 모델 해킹 위험을 줄일 수 있는 효과적인 도구를 확보할 수 있습니다.



### On the Interpolation Effect of Score Smoothing (https://arxiv.org/abs/2502.19499)
- **What's New**: 이번 연구는 score-based diffusion models (확률 기반 확산 모델)의 일반화 능력이 경험적 점수 함수의 smoothing(평활화) 효과에서 비롯된다는 가설을 조사합니다. 기존의 연구들은 다양한 분야에서 이 모델들이 어떠한 방식으로 작동하는지에 대한 통찰을 제공했지만, 본 연구는 특히 훈련 데이터가 일차원 선형 부분 공간에 균일하게 분포했을 때의 점수 smoothing과 denoising dynamics(잡음 제거 동역학) 간의 상호작용을 수학적으로 연구합니다.

- **Technical Details**: 연구는 score smoothing(점수 평활화)이 데이터를 보간하는데 미치는 영향을 분석하고 있습니다. 특히, 훈련 데이터가 균등하게 배치된 1차원 케이스에서 smoothed score(평활화된 점수) 하의 denoising dynamics(잡음 제거 동역학)가 훈련 세트의 간섭을 회복하는 방법을 보여줍니다. 또한, 고차원에 삽입된 1차원 부분 공간에 대한 분석으로, 해당 공간에서의 non-singular(비특이적) 보간 밀도에 대한 수렴을 증명합니다.

- **Performance Highlights**: 연구 결과는 점수 smoothing이 neural networks(NNs) 기반의 확산 모델들이 memorization(기억 현상)을 피하는 방식에 대한 중요한 인과적 이해를 제공함을 보여줍니다. 특정 정규화가 점수 함수에 적용될 때, 모델이 훈련 세트를 넘어 새로운 샘플을 생성할 수 있는 메커니즘을 제시합니다. 또한, NN 학습된 score functions(점수 함수)가 score smoothing과 유사한 보간 효과를 보임을 실증적으로 보여줍니다.



### Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids (https://arxiv.org/abs/2502.20396)
Comments:
          Project page can be found at this https URL

- **What's New**: 이 연구는 휴머노이드 형태에서 접촉이 많은 조작 작업에 대해 강화 학습을 적용하는 데 따른 핵심 도전 과제를 조사합니다. 특히, 실제 환경과 시뮬레이션 환경을 맞추기 위한 자동화된 real-to-sim 튜닝 모듈, 장기 접촉 조작 작업을 위한 일반화된 보상 설계, 그리고 탐색 문제의 샘플 효율성을 개선하는 분할 및 정복(distillation) 과정을 도입합니다.

- **Technical Details**: 연구에서는 비전 기반의 조작 작업을 위해 강화 학습 정책을 설계하고 교육하는 데 있어서 보상 함수를 개별 접촉 목표(contact goals)와 객체 목표(object goals)로 분해하는 일반 원칙을 제안합니다. 또한, 복잡한 탐색 문제에 대처하기 위해 과제를 알고리즘적으로 감소시켜 접근하는 두 가지 실용적 기술을 도입하여 수학적 증명을 통해 샘플 효율성을 개선했습니다.

- **Performance Highlights**: 3가지 휴머노이드 섬세 조작 작업에 대해 성공적인 결과를 도출했으며, 각 기술에 대한 ablation 연구를 통해 제안한 접근 방식의 효과를 검증했습니다. 이 연구는 인간의 시연 없이도 robuste한 일반화와 높은 성능을 달성하는 sim-to-real 강화 학습을 통한 휴머노이드 섬세 조작 학습을 위한 성공적인 방법을 제시합니다.



### Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions (https://arxiv.org/abs/2502.20392)
Comments:
          18 pages, 3 figures

- **What's New**: 이 논문에서는 고차원 시계열 데이터의 signature kernel을 효율적으로 계산하기 위한 새로운 방법을 제시합니다. 기존의 방법보다 메모리 사용량을 크게 줄이고, 시간 소요를 최소화하면서도 높은 정확도를 유지하는 동적 절단을 기반으로 하는 지역적인 Neumann 급수 확장을 활용합니다. 이 접근법은 매우 긴 시계열 데이터 처리에 특히 적합하여, HPC(고성능 컴퓨팅)를 통해 GPU에서 수십만 포인트의 데이터를 효율적으로 처리할 수 있습니다.

- **Technical Details**: 제안하는 방법은 Goursat PDE의 해로서 signature kernel을 특성화하며, 이를 지역적으로 정의된 서브도메인에서 빠르게 수렴하는 power series를 사용해 계산합니다. 알고리즘적으로는, 방향 그래프에서 경계 조건을 재귀적으로 전파하여 지역 Goursat PDE 시스템을 해결하고 정점 분류에 따라 처리합니다. 이러한 방식은 메모리 사용량을 줄이고 정확도를 높이는 효과적인 균형을 제공합니다.

- **Performance Highlights**: 이 연구의 결과는 기존의 PDE 기반 접근법과 비교하여 accuracy, runtime, memory efficiency에서 상당한 성능 개선을 실현하였습니다. 제안된 방법은 머신 러닝, 금융 모델링, 신호 처리와 같은 시스템에 적용될 수 있으며, 불규칙한 샘플링, 재매개화에 대한 불변성과 잡음에 대한 강인성을 모두 갖추고 있습니다. 이를 통해 실제 데이터의 복잡성을 효과적으로 다룰 수 있는 가능성을 제시합니다.



### Physics-Driven Data Generation for Contact-Rich Manipulation via Trajectory Optimization (https://arxiv.org/abs/2502.20382)
- **What's New**: 이번 논문에서는 물리 기반 시뮬레이션, 인간 시연, 모델 기반 계획을 통합하여 저비용의 데이터 생성 파이프라인을 제안합니다. 이 파이프라인은 가상 현실 시뮬레이션 환경에서 수집한 적은 수의 인간 시연을 최적화된 운동학 리타게팅과 궤적 최적화를 통해 다양한 로봇 형태와 물리적 매개변수에 적응하여 고품질의 큰 규모의 데이터셋을 생성합니다. 이는 교차 형태 데이터 전송을 가능하게 하며, 다른 하드웨어 구성이나 물리적 매개변수에서 수집된 레거시 데이터셋을 재사용할 수 있는 잠재력을 제공합니다.

- **Technical Details**: 우리는 인간 시연과 모델 기반 계획의 상호 보완적인 강점을 활용하는 데이터 생성 프레임워크를 제안합니다. 이 방법은 가상 현실 환경에서 수집된 소수의 인간 시연을 기반으로 하여 동적으로 적합한 궤적의 대규모 데이터셋을 시뮬레이션을 통해 생성합니다. 이 시연은 복잡한 탐색 공간에서 계획자가 좋은 초기 추정을 제공하게 하여 물리적 일관성과 다양한 로봇 형태 및 물리적 매개변수에서의 강인함을 보장합니다.

- **Performance Highlights**: 우리는 생성된 데이터셋을 활용하여 여러 로봇 형태, 특히 이중 로봇팔 및 플로팅 기반 Allegro 손에서 접촉이 풍부한 조작 과제를 위한 정책을 학습했습니다. 제로샷 하드웨어 배치에서 높은 성공률을 달성하며, 실제 시나리오에서 데이터셋 활용의 유용성을 강조합니다. 이와 같은 방식은 앞으로 로봇 시스템 개발 및 학습 과정에서 의미 있는 기여가 될 것입니다.



### Tight Inversion: Image-Conditioned Inversion for Real Image Editing (https://arxiv.org/abs/2502.20376)
Comments:
          Project page at: this https URL

- **What's New**: 이 논문은 텍스트-이미지 확산 모델의 이미지 편집 능력을 대폭 향상시키기 위한 새로운 접근 방식을 제시합니다. 기존의 많은 방법들이 이미지를 가우시안 노이즈로 변환하여 편집하는 데 의존하고 있지만, 이 연구는 이러한 변환 과정에서 조건 선택의 중요성을 강조하고 있습니다.

- **Technical Details**: 저자들은 입력 이미지와 정확히 일치하는 조건을 사용함으로써 인버전(inversion) 품질이 크게 향상될 수 있음을 보여줍니다. 이들은 'Tight Inversion'이라는 새로운 인버전 방법을 도입하며, 이는 가장 정확한 조건인 입력 이미지 자체를 활용하여 모델 출력의 분포를 좁히고 재구성(재현) 및 편집 능력을 동시에 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 기존 인버전 방법과 결합했을 때 접근 방식의 효과를 입증하였으며, 재구성 정확도와 다양한 편집 방법과의 통합을 평가하였습니다. 이를 통해 복잡한 이미지 편집의 효율성을 크게 개선할 수 있음을 보여줍니다.



### Global Framework for Simultaneous Emulation Across the Nuclear Landscap (https://arxiv.org/abs/2502.20363)
- **What's New**: 이 논문에서는 ab initio 많은 신체 계산(many-body calculations)과 베이지안 신경망(Bayesian neural network)을 통합한 계층적 프레임워크를 제안합니다. 이 시스템은 여러 동위원소(isotopes)의 핵 특성을 정확하게 예측할 수 있는 에뮬레이터를 발전시킵니다. 산소 동위원소 체인을 활용한 벤치마킹을 통해, 우리는 기초 상태 에너지(ground-state energies)와 핵 전하 반경(nuclear charge radii)에 대한 정확한 결과를 달성하며 견고한 불확실성 정량화를 제공합니다.

- **Technical Details**: BANNANE은 동위원소 체인을 통해 핵 특성을 에뮬레이트하기 위해 설계되었습니다. 이 시스템의 계층적 베이지안 구조는 학습 가능한 임베딩(embeddings)과 다중 헤드(self-attention mechanism)를 포함하여 각 충실도 수준(fidelity level)에 맞는 쿼리를 사용합니다. 이러한 아키텍처는 원소와 동위원소 간의 핵 관측치(nuclear observables) 간의 상관관계를 활용하여 다양한 수준의 계산 충실도에 적응할 수 있도록 보장합니다.

- **Performance Highlights**: 우리의 모델은 신뢰성 있는 불확실성 양정을 제공하며, 이는 저에너지 상수(low-energy constants)에 대한 글로벌 민감도 분석(global sensitivity analysis)을 가능하게 합니다. 각 충실도 수준에 대해 고유한 학습 가능한 쿼리 벡터가 도입되어, 다중 헤드 self-attention 메커니즘을 통해 서로 다른 핵 특성을 효과적으로 추출할 수 있습니다. 전체 하이퍼 파라미터 최적화(hyperparameter optimization)를 통해 성능이 더욱 향상될 것으로 기대됩니다.



### Bridging the Creativity Understanding Gap: Small-Scale Human Alignment Enables Expert-Level Humor Ranking in LLMs (https://arxiv.org/abs/2502.20356)
- **What's New**: 이 논문은 Large Language Models (LLMs)의 창의적 콘텐츠 이해의 한계를 다시 조명하며, 특히 유머 이해 과정에서의 인간과 LLM 간의 큰 격차를 밝혀냈습니다. 연구진은 유머 이해를 시각적 이해, 만화-캡션 추론, 인간 선호도 정렬의 세 가지 구성 요소로 분해하였고, 이를 통해 유머 캡션 순위에서 82.4%의 정확도를 달성하여 이전의 67% 기준선을 크게 초월했습니다. 또한, 잠재적 편향을 줄이기 위해 인간 선호 데이터로의 세심한 조정이 중요하다는 점을 강조하였습니다.

- **Technical Details**: 이 연구는 유머 캡션 순위의 세 가지 주요 구성 요소를 제시합니다: 시각적 이해(visual understanding), 유머 추론(humor reasoning), 그리고 인간 선호도 정렬(alignment with human preferences)입니다. 연구진은 시각적 주석과 LLM이 생성한 설명을 개선하여 유머 캡션에서의 비주얼 이해와 추론 능력을 크게 향상시켰습니다. 특히, 각 캡션 쌍에 대한 인간 기호 데이터를 통한 세밀한 튜닝(fine-tuning)이 성과를 높이는 데 중요한 역할을 했습니다.

- **Performance Highlights**: 이 연구는 캡션 순위 성능을 67%에서 82.4%로 증가시키며 인간 전문가들과 동등한 수준에 도달했습니다. 연구진의 실험 결과, 다양한 페르소나 기반 접근 방식은 효과가 미미했으나, 군중 선호 데이터를 사용한 세밀한 조정은 매우 효과적임을 발견했습니다. 이러한 성과는 AI가 창의적 과제에서 인간의 개별 및 집단 선호를 이해하는 데 있어 광범위한 도전 과제를 나타냅니다.



### Improving the Efficiency of a Deep Reinforcement Learning-Based Power Management System for HPC Clusters Using Curriculum Learning (https://arxiv.org/abs/2502.20348)
Comments:
          13 pages, 17 figures, accepted at Supercomputing Asia '25, published by ACM

- **What's New**: 이 연구는 HPC (High-Performance Computing) 시스템의 에너지 관리를 위한 깊은 강화 학습 (Deep Reinforcement Learning, DRL) 에이전트의 성능을 향상시키기 위해 커리큘럼 학습 (Curriculum Learning, CL)을 통합한 방법을 제안합니다. 특히, 시스템에서의 효율성을 높이기 위해 문제 난이도를 점차적으로 증가시키는 훈련 방식을 활용합니다. 실험을 통해 기존의 고정 시간 제한 전략 및 기본 DRL 방법과 비교하여 제안된 방법의 효과를 입증하고 에너지 절약과 서비스 품질 (Quality of Service, QoS) 간의 균형을 맞추었습니다.

- **Technical Details**: 연구는 HPC 환경에서 에너지 소비 문제를 다루며, CL을 통해 DRL 에이전트의 학습 효과를 증대시키는 방법을 제시합니다. CL은 낮은 난이도부터 중간, 높은 난이도로 순차적으로 훈련하여, 에이전트가 더 효과적으로 학습할 수 있도록 합니다. 여기서는 데이터셋을 생성하는 체계적인 접근 방식을 개발하고, 다양한 난이도에 따라 최적의 훈련 순서를 조사하여 DRL 에이전트의 성능을 향상시킵니다.

- **Performance Highlights**: 이 연구의 주요 결과는 제안된 CL 기반 에이전트가 기본 DRL 방법보다 3.73%의 에너지 절감을 이루었고, 가장 좋은 타임아웃 구성에 비해 4.66%의 개선을 달성한 것입니다. 또한, 평균 작업 대기 시간을 9.24% 줄이고, 작업 채우기 비율을 높임으로써 자원 활용을 효율적으로 증가시켰습니다. 다양한 전환 시간과 클러스터 크기에 대한 민감도 테스트를 통해 에이전트의 적응성을 증명하여, 재교육 없이도 시스템 파라미터 변화에 효과적으로 대응할 수 있음을 보여주었습니다.



### T1-PILOT: Optimized Trajectories for T1 Mapping Acceleration (https://arxiv.org/abs/2502.20333)
- **What's New**: 이 논문에서는 T1-PILOT라 불리는 새로운 파이프라인을 소개합니다. 이 방법은 T1 신호 완화 모델을 샘플링-재구성 프레임워크에 명시적으로 통합하여 비-Cartesian(Non-Cartesian) 궤적 학습을 안내합니다. T1-PILOT는 실험을 통해 기존 방법들보다 더 높은 T1 맵 충실도를 달성하며, 이로 인해 정량적 정확성이 향상되고 촬영 시간도 단축된다는 것을 보여줍니다. 이 혁신적인 접근 방식은 병리학 진단에 중요한 자료인 심장 MRI T1 맵 생성에 있어 중요한 진전을 나타냅니다.

- **Technical Details**: T1-PILOT는 심장 조직 T1 맵을 추정하기 위해 여러 T1 가중치 이미지를 기반으로 하고 있습니다. 이 과정에서 T1 완화 곡선을 적합하기 위해 지수형 T1-감쇠 곡선을 사용하는 것을 목표로 합니다. 이는 인공 신경망을 통해 감쇠 파라미터 A, B, T1*를 효율적으로 최적화하고, 학습된 k-space 언더샘플링 마스크를 통해 T1 맵 추정 속도를 높입니다. 이러한 접근은 고도로 언더샘플링된 데이터에서도 정확한 T1 맵 생성을 가능케 하여, 기존 방법들과의 성능 차별성을 보여줍니다.

- **Performance Highlights**: CMRxRecon 데이터세트에서 T1-PILOT는 고정된 방사형 및 골든 앵글 샘플링 방식, 그리고 단일 학습 궤적 등 여러 기준 전략들과 비교하여 뛰어난 성능을 보였습니다. PSNR(픽셀 신호 대 잡음 비율)와 VIF(구조적 유사성 지표)에서 높은 수치를 기록하며, 심장 조직의 미세 구조를 더 잘 활용하는 것으로 나타났습니다. 이러한 결과는 T1 완화 신호를 명시적으로 모델링함으로써 인수요정확성과 촬영 시간 모두에서 개선을 가져옴을 입증하고 있습니다.



### Impilict Runge-Kutta based sparse identification of governing equations in biologically motivated systems (https://arxiv.org/abs/2502.20319)
Comments:
          23 pages, 9 figures

- **What's New**: 이 연구에서는 데이터 부족과 노이즈에 강인한 새로운 데이터 기반 프레임워크 IRK-SINDy를 제안합니다. 이는 고차 암묵적 룽게-쿠타 방법(Implicit Runge-Kutta, IRK)과 희소 식별(Sparse Identification)을 통합하여 개발되었습니다. 두 가지 방법으로 IRK를 희소 회귀에 통합하며, 반복 방식과 딥 뉴럴 네트워크를 사용하여 IRK의 단계 값(Stage values)을 예측합니다.

- **Technical Details**: IRK-SINDy 프레임워크는 고차 리프트(Implicit) 희소 식별 기법을 사용하여 데이터의 부족과 노이즈에 강력하게 대응할 수 있도록 설계되었습니다. 첫 번째 방법은 비선형 대수 시스템의 방정식을 수치적으로 해결하기 위해 반복 알고리즘을 사용하는 것이며, 두 번째 방법은 딥 뉴럴 네트워크를 활용하여 IRK의 단계 값(Stage values)을 예측합니다. 이 두 가지 접근법은 복잡한 시스템의 동역학을 모델링하는 데 유용한 기능을 제공합니다.

- **Performance Highlights**: IRK-SINDy의 성능은 다양한 동역학 행동을 포함한 벤치마크 문제에 대한 수치적 실험을 통해 입증되었습니다. 연구 결과, IRK-SINDy는 기존의 SINDy 및 RK4-SINDy 프레임워크보다 뛰어난 성능을 보여주었으며, 특히 데이터가 극도로 부족하거나 노이즈가 있는 조건에서도 해석 가능하고 일반화된 모형을 생성할 수 있음을 나타냅니다.



### LangProBe: a Language Programs Benchmark (https://arxiv.org/abs/2502.20315)
- **What's New**: 이번 논문에서는 LangProBe라는 새로운 대규모 벤치마크를 소개하고 있습니다. LangProBe는 2000개 이상의 작업, 아키텍처, 최적화 기법, 언어 모델(LM) 조합을 평가하여 언어 프로그램 아키텍처와 최적화 전략의 영향을 연구합니다. 이 연구는 다양한 라인업의 언어 프로그램을 통해 최적화된 언어 프로그램이 비용 대비 품질을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: LangProBe의 기반이 되는 연구는 DSPy와 TextGrad와 같은 선언적 언어 프레임워크를 통해 언어 프로그램을 작성하고 자동화하는 접근 방식을 활용하고 있습니다. 이러한 프로그램은 외부 도구와의 통합 및 정보 흐름을 구성하고, 특히 외부 정보에 대한 접근을 요구하는 작업에 필수적입니다. 또한, MIPRO과 같은 최적화 기법들은 다양한 모델과 작업 조합에 대해 품질 향상을 제공하는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, 최적화된 언어 프로그램은 일반적인 모델 호출 방식보다 우수한 성능을 보이는 것으로 나타났습니다. 예를 들어, gpt-4o-mini에서 실행되는 최적화된 프로그램은 낮은 비용으로 뛰어난 성능을 발휘했습니다. 그러나 모든 문제에서 일관된 결과를 보이는 것은 아니며, 고급 모델을 통한 기본 문제 해결에 있어서는 추가적인 조합이나 최적화가 필요하지 않았습니다.



### Visual Adaptive Prompting for Compositional Zero-Shot Learning (https://arxiv.org/abs/2502.20292)
- **What's New**: 이번 논문은 비전-언어 모델(VLMs)을 활용하여 Compositional Zero-Shot Learning (CZSL)에서의 성능을 획기적으로 향상시킬 수 있는 Visual Adaptive Prompting System (VAPS)을 제안합니다. VAPS는 학습 가능한 시각적 프롬프트(retrieval mechanism)를 사용하여 시각적 특성과 의미적 정보를 연결하고, 이미지를 기반으로 동적으로 적합한 프롬프트를 선택합니다. 이는 전통적인 고정 프롬프트 방식의 한계를 극복하며, 보다 유연한 조합 학습을 가능하게 합니다.

- **Technical Details**: VAPS는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 학습 가능한 시각적 프롬프트 저장소를 구축하여 이미지에서 추출한 시각적 특성을 효율적으로 활용합니다. 둘째, 텍스트 프롬프트 어댑터를 통해 이미지의 시각적 맥락에 맞게 텍스트 프롬프트를 동적으로 업데이트하며, 이는 속성과 객체의 분리를 효과적으로 도와줍니다.

- **Performance Highlights**: VAPS는 세 가지 CZSL 벤치마크에서 실험을 통해 최신 기술(state-of-the-art) 결과를 달성했습니다. 이 방법은 기존의 정적 텍스트 프롬프트 기반 방법보다 높은 유연성과 일반화 능력을 보여 주며, 특히 새로운 조합을 학습하는 데 있어서 탁월한 성능을 발휘하였습니다.



### Multiple Linked Tensor Factorization (https://arxiv.org/abs/2502.20286)
Comments:
          26 pages, 4 figures, 7 tables

- **What's New**: 이 논문에서는 여러 데이터 소스를 동시에 처리할 수 있는 새로운 다중 연결 텐서 인수분해 기법인 MULTIFAC를 제안합니다. 기존의 CANDECOMP/PARAFAC (CP) 분해를 확장하여 다중 차원의 다중 배열 데이터를 함께 저차원으로 줄이고 근본적인 신호를 근사화하는 방법을 제공합니다. 또한 불완전한 데이터를 처리하기 위해 이 기법을 기대 최대화(Expectation-Maximization) 버전으로 확장하여 결측값 보간을 수행할 수 있게 되었습니다.

- **Technical Details**: MULTIFAC 방법은 L2 패널티를 사용한 CP 분해를 기반으로 하며, 이를 통해 순위 희소성(rank sparsity)을 달성합니다. 이 방식은 서로 다른 데이터 소스 간의 공유된 잠재 구성 요소를 자동으로 드러내게 됩니다. 다양한 시뮬레이션 연구를 통해 MULTIFAC의 성능을 입증하며, 이는 다중 오믹스(data sets) 데이터에서 해석 가능한 분해를 제공하는 데도 기여합니다.

- **Performance Highlights**: MULTIFAC의 성능은 여러 측면에서 드러납니다. 첫째, 근본적인 신호를 정확히 근사화할 수 있습니다. 둘째, 공유 및 비공유 구조를 식별할 수 있으며, 셋째, 결측 데이터를 적절히 보간할 수 있습니다. 이는 초기 단계의 철 결핍에 관한 연구와 같은 실제적인 적용 사례에서 매우 유용하게 활용될 수 있습니다.



### HVI: A New color space for Low-light Image Enhancemen (https://arxiv.org/abs/2502.20272)
Comments:
          *These authors contributed equally to this work

- **What's New**: 이 논문에서는 저조도 이미지 향상(Low-Light Image Enhancement, LLIE)을 위한 새로운 색상 공간인 Horizontal/Vertical-Intensity (HVI)를 제시합니다. 기존의 LLIE 방법들은 sRGB 색상 공간을 기반으로 하여 색편향(color bias)과 밝기 아티팩트(brightness artifacts)를 발생시켰습니다. HVI는 극성화된 HS 맵 및 학습 가능한 강도를 통해 이러한 문제를 해결하고자 합니다.

- **Technical Details**: HVI 색상 공간은 빨간색 좌표 간의 거리를 최소화하여 빨간색 아티팩트 제거를 강제합니다. 또한, 낮은 조도 지역을 압축하여 검은색 아티팩트를 제거합니다. 이를 위해 새로운 Color and Intensity Decoupling Network (CIDNet)를 도입하여 HVI 공간에서 다양한 조명 환경에 따른 정확한 광도 매핑 기능을 학습합니다.

- **Performance Highlights**: 제안된 HVI 색상 공간과 CIDNet은 10개의 데이터셋에서 최신 기법(State-of-the-art methods)보다 뛰어난 성능을 보여줍니다. 종합적인 벤치마크 및 배제 실험 결과는 이 새로운 접근 방식의 유효성을 입증합니다. 코드는 제공된 URL에서 확인할 수 있습니다.



### Generative adversarial neural networks for simulating neutrino interactions (https://arxiv.org/abs/2502.20244)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문은 전통적인 몬테카를로(Monte Carlo) 생성기 접근법 대신 새로운 중성미자( neutrino) 산란 사건 시뮬레이션 방법을 제안합니다. 생성적 적대 신경망(Generative Adversarial Network, GAN) 모델을 사용하여 몇 기가 전자볼트(GeV) 에너지 범위에서 중성미자-탄소 충돌을 시뮬레이션합니다. 두 개의 GAN 모델이 개발되었으며, 하나는 준탄성(quasielastic) 중성미자-핵 산란만을 시뮬레이션하고, 다른 하나는 주어진 중성미자 에너지에서 모든 상호작용을 시뮬레이션합니다.

- **Technical Details**: 이 연구는 딥 뉴럴 네트워크(Deep Neural Network, DNN)를 통해 중성미자-핵 산란 사건을 생성하는 것을 목표로 합니다. 누워(NuWro) 몬테카를로 생성기를 정보 출처로 활용하며, 이 모델은 중성미자와 탄소 사이의 충돌을 모델링합니다. 생성된 데이터는 최종 입자에 대한 완전한 정보를 포함하며, 가변적인 최종 입자 수를 갖습니다.

- **Performance Highlights**: 모델의 성능은 평균 절대 차이와 지구 이동 거리(Earth Mover’s Distance, EMD)라는 두 가지 통계적 지표를 사용하여 평가되었습니다. 두 GAN 모델 모두 사건 분포를 성공적으로 재현하는 것으로 나타났습니다. 이러한 결과는 GAN이 중성미자 상호작용의 물리적 현상을 효과적으로 시뮬레이션할 수 있는 가능성을 보여줍니다.



### Swap Regret and Correlated Equilibria Beyond Normal-Form Games (https://arxiv.org/abs/2502.20229)
- **What's New**: 이 논문에서는 폴리토프 게임(Polytope Games)에서의 새로운 스왑 레그렛(Swap Regret) 변형인 프로파일 스왑 레그렛(Profile Swap Regret)을 제안하고 있습니다. 이 새로운 개념은 학습 알고리즘이 상대방에 의해 조작되지 않도록 하기 위한 필요한 및 충분한 조건으로 작용합니다. 저자들은 Mansour et al. (2022)의 미해결 문제를 해결하며 이 이론의 혜택을 확인하게 되었습니다.

- **Technical Details**: 프로파일 스왑 레그렛은 NP-하드(NP-hard)로 계산할 수 있지만, 저자들은 O(√T) 레그렛을 보장하는 효율적인 학습 알고리즘을 설계할 수 있음을 보여줍니다. 이 알고리즘은 게임의 크기와 라운드 수에 다항적으로 실행되며, 상대방이 이 알고리즘을 조작하려고 할 때 얻을 수 있는 효용이 O(d√T)로 제한됩니다.

- **Performance Highlights**: 연구팀은 프로파일 스왑 레그렛으로 인한 플레이에 의해 유도된 상관 균형(correlated equilibrium) 개념을 탐구하고 있습니다. 저자들은 이 학습 과정을 통해 구현 가능한 결과 집합과 제3자 중재자가 구현할 수 있는 결과 집합 간의 차이를 보여줍니다. 이는 일반형 게임(Normal-form Games)과의 대조를 통해 더욱 명확히 드러납니다.



### Deep Convolutional Neural Networks for Palm Fruit Maturity Classification (https://arxiv.org/abs/2502.20223)
- **What's New**: 이 연구는 최적의 성숙 단계에서 팜 과일을 수확하여 팜 오일의 수량과 품질을 극대화하는 것을 목표로 하며, 자동화된 컴퓨터 비전 시스템을 개발하여 팜 과일 이미지를 다섯 가지 상태로 분류합니다. 딥 컨볼루션 신경망 (CNN)을 사용하여 과일의 성숙 단계에 따라 이미지를 분류하고, 사전 훈련된 ResNet50 및 InceptionV3 아키텍처를 활용하여 전이 학습과 미세 조정을 적용합니다.

- **Technical Details**: 이 연구에서는 평균 85% 이상의 정확도로 팜 과일 성숙 단계를 분류하기 위한 딥 CNN 모델을 제안하였습니다. 8,000장 이상의 이미지로 구성된 공개 데이터셋을 사용하여 80%는 훈련, 20%는 테스트로 나누어 실험을 수행했으며, 이 과정에서 CNN의 효과적인 특징 추출 능력을 활용합니다. 특히, 색상 모델의 RGB 및 HSI에서 특징을 추출하고, 이를 부가적인 분류 알고리즘에 주입하여 성숙 단계를 분류합니다.

- **Performance Highlights**: 제안된 딥 CNN 모델은 팜 과일 성숙 단계를 분류하는 데 있어 85% 이상의 높은 테스트 정확도를 기록하며, 이는 자동화된 성숙도 평가의 잠재력을 강조합니다. 이 연구는 팜 오일 생산 효율 증대와 수확 결정 최적화를 위한 기여로 나아갈 수 있는 중요한 성과를 이루었습니다.



### Topological Autoencoders++: Fast and Accurate Cycle-Aware Dimensionality Reduction (https://arxiv.org/abs/2502.20215)
- **What's New**: 이 논문은 고차원 데이터에서 주기적 패턴을 정확하게 시각화하기 위해 새로운 topology-aware dimensionality reduction 방식을 제시합니다. Topological Autoencoders (TopoAE)의 기본 개념을 확장한 TopoAE++를 도입하여, 1차원 persistent homology (PH^1)를 보존하면서 더 나은 시각적 임베딩을 생성합니다. 이 방식은 cascade distortion이라는 새로운 패널티 항을 사용하여 고차원 데이터의 주기적인 구조를 더 정확하게 복구합니다.

- **Technical Details**: 논문에서는 TopoAE의 손실 함수에 대한 새로운 이론적 분석을 제공하며, Rips 필터레이션의 0차 persistent homology에 대해 손실이 0이면 고차원과 저차원에서 동일한 지속 쌍을 초래함을 보여줍니다. 그러나, d>=1에 대한 TopoAE의 단순한 확장에서는 이러한 성질이 성립하지 않음을 반증 사례를 통해 나타냅니다. 또한, TopoAE++의 개발에 따라 Rips 필터레이션에 대한 새로운 빠른 알고리즘을 도입하여 계산 효율성을 높였습니다.

- **Performance Highlights**: TopoAE++는 기존의 topology-aware 방법들과 비교할 때, Wasserstein distance에 의해 측정된 토폴로지의 정확성과 저차원에서의 주기 구조의 시각적 보존 사이에서 더 나은 균형을 이룹니다. 실험 결과, 주기적 패턴의 시각적 표현에서 현저한 개선을 달성하였으며, 고차원 데이터에 대한 해석 가능성을 높였습니다. C++ 구현은 해당 URL에서 제공됩니다.



### Layer-Aware Task Arithmetic: Disentangling Task-Specific and Instruction-Following Knowledg (https://arxiv.org/abs/2502.20186)
- **What's New**: 논문에서 제안하는 Layer-Aware Task Arithmetic (LATA)는 과거의 태스크 산술(task arithmetic, TA) 방법을 개선하여, 각 레이어에 대해 특정 가중치를 부여하는 방식을 사용합니다. 이 접근법은 목표 태스크와 강하게 연관된 레이어는 강조하고, 지침 수행(instruction-following)과 관련된 레이어는 억제하여 모델의 태스크 학습 및 잊기 성능을 향상시킵니다. LATA는 여러 태스크의 성능을 유지하면서 기존 방법들보다 더 높은 태스크 정확도를 달성하는 데 중점을 두고 있습니다.

- **Technical Details**: LATA는 네 단계로 구성되며, 첫 번째 단계에서는 베이스 모델과 사전 훈련(pre-trained) 모델의 파라미터 차이를 통해 지침 벡터를 정의합니다. 두 번째 단계에서, 각 세분화된 태스크 모델의 파라미터에서 베이스 모델의 파라미터를 빼도록 하여 복합 벡터(complex vector)를 도출합니다. 각 레이어의 파라미터로 이루어진 레이어 벡터(layer vector)를 생성한 후, 두 벡터 간 코사인 유사도(cosine similarity)를 계산하여 태스크 관련 요소를 구분합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋인 WikiText-2, GSM8K 및 HumanEval에서 진행된 실험 결과 LATA는 다중 작업 학습(multi-task learning) 및 선택적 태스크 잊기(selective task forgetting)에서 기존 방법들을 초월하는 성과를 보여주었습니다. LATA는 불필요한 능력을 선택적으로 제거할 때도 전체 성능의 최소 손해를 보이며 효과적인 결과를 도출했습니다. 이 연구는 태스크 전용 지식과 일반 용도 지식을 분리하는 데 있어 레이어별 분석의 중요성을 강조합니다.



### Re-evaluating Open-ended Evaluation of Large Language Models (https://arxiv.org/abs/2502.20170)
Comments:
          Published at ICLR 2025

- **What's New**: 이 논문은 Large Language Models (LLMs)의 평가 방식을 개선하기 위해 전통적인 Elo 기반 시스템의 문제점을 지적하고, 데이터의 편향(bias) 및 중복(redundancy)에 민감한 특성으로 인해 발생할 수 있는 부작용을 설명합니다. 저자들은 이를 해결하기 위해 3인 게임(gamer-theoretic)의 형태로 평가를 제안하고, Robust한 평가 방법을 위한 새로운 게임 이론적(solution concept) 개념을 소개합니다. 이를 통해 LLM 개발의 경쟁적인 장면을 이해하고 더 직관적인 점수가 이루어질 수 있도록 합니다.

- **Technical Details**: 저자는 게임 이론(games theory)을 적용하여 LLM 평가를 재구성하기 위해 여러 기여를 합니다. 이들은 N-player 일반합계(general-sum) 게임의 고유하고 클론 불변(clone-invariant)한 균형 솔루션 개념을 도출하고, 실제 LLM 평가 데이터셋에 대한 적용 가능성을 보여줍니다. 또한, 협동적인 데이터 품질 문제를 해결하는 한편, 평가 시스템 설계자들이 목표를 명확하게 표현할 수 있도록 합니다.

- **Performance Highlights**: LMSYS Chatbot Arena와 같은 시스템에서 Elo 점수를 지속적으로 상승시키는 모델들이 과연 기술적 발전을 의미하는지에 대한 의문을 제기합니다. 저자들은 시뮬레이션을 통해 모델이 특정 기술에 특화될 위험을 실증적으로 보여 주며, 평가 시스템의 변별력을 향상시키고 균형 잡힌 점수를 보장하기 위한 방향성을 제공합니다. 이러한 접근은 LLM 개발 과정에서의 주요 이슈인 편향 및 데이터 품질 문제를 근본적으로 해결하는 데 기여할 것입니다.



### Accelerating Model-Based Reinforcement Learning with State-Space World Models (https://arxiv.org/abs/2502.20168)
- **What's New**: 이번 연구에서는 모델 기반 강화 학습(Model-based Reinforcement Learning, MBRL)의 학습 속도를 개선하기 위한 새로운 방법을 제안합니다. 제안된 방법은 상태-공간 모델(State-Space Models, SSMs)을 활용하여 동적 모델의 훈련을 병렬화하게 됩니다. 이 접근법은 복잡한 로봇 제어 작업에서도 MBRL의 학습 시간을 여러 배로 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 병렬 가능한 상태-공간 모델로서 모던한 SSM을 사용하여 MBRL의 세계 모델 훈련을 가속화합니다. 동적 모델에 대한 훈련 병렬화는 계산 복잡성을 줄이고, 일반적으로 MBRL 방법보다 더 빠른 훈련을 가능하게 합니다. 특히 균형 잡힌 시각적 정보를 제공하는 구조를 통해 부분 관찰 환경에서도 효과적으로 작용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 방법은 아지일 쿼드로터 비행 작업에서 최대 10배의 세계 모델 훈련 시간 단축과 4배의 전체 MBRL 훈련 시간 단축을 구현합니다. 실험 결과, 제안된 방법은 기존의 MBRL 방법과 유사한 샘플 효율성과 작업 보상을 달성하여 성능 손실 없이 훈련 속도를 크게 향상시켰습니다.



### Robust sensitivity control in digital pathology via tile score distribution matching (https://arxiv.org/abs/2502.20144)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 Whole Slide Image (WSI) 분류 모델의 민감도(sensitivity)를 최적 수송(optimal transport) 및 다중 인스턴스 학습(Multiple Instance Learning, MIL)을 기반으로 제어하는 새로운 접근 방식을 제시합니다. 이는 임상 환경에서 중요한 다양한 메트릭(metrics)을 조율하기 위한 실용적인 솔루션을 제공합니다. 우리의 방법은 적은 수의 보정 샘플로 강력한 민감도 제어를 가능하게 하며, 여러 집단(cohorts) 및 과제에서 효과적으로 검증되었습니다.

- **Technical Details**: 제안된 방법인 Tile-Score Matching (TSM)은 WSI 이진 분류 문제에서 민감도를 제어하는 새로운 방법론으로, 기존의 방법들보다 훨씬 적은 데이터로 보정을 수행할 수 있습니다. TSM은 tile 레벨에서 예측 점수의 분포를 조정하여, 임상 적용 시 필요로 하는 보정 데이터를 최소화합니다. 이 방법은 기존의 Unsupervised Prediction Alignment (UPA)와 유사하지만, WSI 레벨이 아닌 tile 레벨에서 작동하여 여러 배수의 보정 샘플을 사용 가능합니다.

- **Performance Highlights**: 실험 결과, TSM은 매우 낮은 데이터 및 유병률(prevalence) 상황에서 민감도를 효과적으로 제어할 수 있음을 보여주었습니다. 특히, 오직 5개의 양성 샘플만으로도 보정이 가능하여 기존 방법과 비교할 때 현저히 낮은 요구사항을 충족시킵니다. 우리의 연구는 디지털 병리학 모델을 더 넓은 임상환경에 신뢰성 있게 배포할 수 있도록 하는 방법론적 기초를 제공합니다.



### QPM: Discrete Optimization for Globally Interpretable Image Classification (https://arxiv.org/abs/2502.20130)
- **What's New**: 이번 논문에서는 딥 뉴럴 네트워크의 분류를 이해하기 위한 새로운 접근 방식을 제시합니다. 최근 모델들이 단일 결정에 대해 지역적으로 설명할 수 있었던 반면, 정확한 모델의 전반적인 행동을 신뢰성 있게 설명하는 것은 여전히 도전 과제였습니다. 이를 해결하기 위해 Quadratic Programming Enhanced Model (QPM)을 도입하여 전 세계적으로 해석 가능한 클래스 표현을 학습합니다.

- **Technical Details**: QPM은 각 클래스를 5개의 특징으로 이진 할당(binary assignment)하여 표현합니다. 이 특징들은 다른 클래스와도 공유되어 대조적인 클래스 표현을 쉽게 비교할 수 있도록 설계되었습니다. 최적의 할당은 미리 정의된 유사성 측정 및 해석 가능성 제약을 기반으로 한 이산 최적화(discrete optimization)를 통해 찾아지며, 이 결과는 다양한 특징을 미세 조정(fine-tune)하는 데 사용됩니다.

- **Performance Highlights**: QPM은 소규모 및 대규모 데이터셋에서 전례 없는(global interpretability) 전세계적 해석성을 제공하며, 해석 가능한 모델의 정확도(state of the art)에서도 최고의 성과를 기록하였습니다. 이 모델은 안전-critical한 상황에서 사용될 수 있으며, 대규모 활용을 염두에 두고 개발되었습니다.



### Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking (https://arxiv.org/abs/2502.20129)
- **What's New**: 이 연구는 Chain-of-Thought (CoT) 방법이 Transformer 기반 대형 언어 모델의 성능을 크게 향상시키는 방법을 기존의 방법론과 비교하여 검토하였습니다. 이를 통해 CoT의 효과를 입증하고, late-layer MLP 뉴런이 세계 상태를 추적하는 데 중요한 역할을 한다는 것을 밝혔습니다. 또한, 압축(compression)과 구별(distinction)이라는 두 가지 메트릭스를 제안하여, 모델이 대칭 유한 상태 자동자(automaton, FSA)를 내부에 내재하고 있음을 보여주었습니다.

- **Technical Details**: 연구팀은 Transformer+CoT와 다른 모델 간의 상태 추적 능력을 평가하여, Transformer+CoT가 다양한 그룹에 대해 임의의 길이 순서를 효과적으로 학습할 수 있음을 입증했습니다. 또한, activation patching 기술을 사용하여 내부 메커니즘을 분석하고, late-layer MLP 뉴런이 상태를 추적하는 데 주로 사용됨을 확인했습니다. 압축 메트릭스와 구별 메트릭스를 통해 모델이 세계 모델(FSA)을 재구성하는 데 거의 100% 정확성을 달성했다는 것을 발견했습니다.

- **Performance Highlights**: 제안한 접근 방식은 노이즈가 있는 환경에서도 강력한 알고리즘을 학습하는 데 성공하였으며, 모델의 견고성(robustness)을 입증합니다. 특히, Transformer+CoT는 복잡한 작업에서도 상태 추적 기능을 지원할 수 있는 이론적 기초를 제공하며, 이는 다양한 다운스트림 작업에 적용이 가능합니다. 추가적으로, 실험 결과는 CoT 방법이 다양한 실제 시나리오에서 효과적으로 작동하며, 복잡한 문제를 해결하는 데 도움을 줄 수 있음을 나타냅니다.



### Discovering Antagonists in Networks of Systems: Robot Deploymen (https://arxiv.org/abs/2502.20125)
- **What's New**: 이 논문에서는 로봇 떼의 물리적 행동을 통한 맥락적 이상 탐지 방법이 제안된다. 로봇의 정상 행동에 대한 시뮬레이션 데이터를 사용하여 normalizing flow를 훈련시키고, 이를 통해 현재 환경에서 로봇의 움직임 가능성을 예측한다. 이 방식은 적대적 행동을 탐지하는 데 사용되며, 교육 데이터에서 이상 행동에 대한 사전 지식이 필요하지 않다. 기존의 방법들보다 높은 예측 성능과 탐지 기준의 강건성을 보여준다.

- **Technical Details**: 제안된 방법은 로봇 떼가 이루는 감시 작업을 다루며, 각 로봇은 감시할 서브 영역을 부여받아 최적의 커버리지를 목표로 한다. 로봇은 Lloyd’s Algorithm을 사용하여 서로의 위치를 기반으로 새로운 목표 위치를 선정하며, 이 과정에서 Voronoi tessellation을 활용한다. 연구의 중점은 적대적 행동이 로봇 행동의 물리적 움직임에서 어떻게 드러나는지를 분석하는 것이며, 이를 신경망과 여러 탐지 기준을 통해 실현한다.

- **Performance Highlights**: 논문에서는 제안된 방법이 다섯 가지 적대적 행동 전략에 대해 평가되었다. 검증 결과, 가장 좋은 탐지 기준은 정상 로봇을 80% 이상 정확하게 분류하면서도 5% 미만의 거짓 긍정률을 유지했다. 하드웨어 실험에서도 유사한 결과를 나타내며, 이전의 연구에 비해 이상 탐지의 성능이 크게 향상되었음을 보여준다.



### Self-Training Elicits Concise Reasoning in Large Language Models (https://arxiv.org/abs/2502.20122)
Comments:
          23 pages, 10 figures, 18 tables

- **What's New**: 본 논문에서는 체인의 사고(Chain-of-thought, CoT) 추론 메커니즘을 통해 대형 언어 모델들이 복잡한 작업을 해결할 수 있는 능력이 향상되었음을 설명합니다. 그러나 기존의 모델들이 과도한 토큰을 생성하고 있으며, 이는 불필요한 추론 비용을 초래한다고 주장합니다. 이에 대한 해결책으로, 저자들은 자가 생성된 간결한 추론 경로를 활용한 간단한 파인튜닝 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 Zero-shot prompting이 간결한 추론을 효과적으로 유도하는 데 한계가 있음을 보여주며, 대신 Best-of-N (BoN) 샘플링과 Few-shot conditioning을 활용하여 모델을 파인튜닝하는 방법을 제시합니다. 저자들은 GSM8K와 MATH 데이터셋을 활용하여 다양한 모델 패밀리에서 평균 30%의 출력 토큰 수 감소를 달성하였으며, 이는 이전의 파인튜닝 기준 대비 2.4배 향상된 결과입니다. 이 과정을 통해, 모델들은 질문의 복잡성에 따라 출력 길이를 적절히 조정할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 본 논문에서 제안된 FS-BoN 방법은 모델의 경량화된 추론 경로를 효율적으로 이끌어내어, 복잡한 작업에 대한 추론 비용을 줄이는 데 효과적임을 입증했습니다. 성능 분석에 따르면, 훈련된 모델은 문제의 난이도에 따라 변별력 있는 출력을 유지하며 적절한 자세로 응답을 조정합니다. 이러한 결과는 다양한 모델 스케일에서도 일관되게 유지되었으며, 자가 생성된 데이터의 파인튜닝이 LLM의 잠재적인 간결한 추론 능력을 발휘할 수 있도록 하는 데 기여할 수 있음을 시사합니다.



### Qini curve estimation under clustered network interferenc (https://arxiv.org/abs/2502.20097)
- **What's New**: 본 논문에서는 Qini curve 추정에 대한 새로운 접근 방식을 제안합니다. 특히, 군집 네트워크 간섭(clustered network interference) 환경에서 Qini curve를 추정하는 방법론을 다룹니다. 기존의 추정 방식들이 간섭을 무시할 경우 발생하는 시스템적 편향을 지적하며, 근본적인 문제를 해결하고자 합니다. 저자는 이를 통해 비용 효과성을 보다 정확히 평가할 수 있게 되는 방법론을 모색합니다.

- **Technical Details**: Qini curve는 치료 정책의 비용 효과성을 평가하는 도구로, 치료 효과의 이질성을 이해하는 데 중심적 역할을 합니다. 본 논문은 군집 간섭이 있는 환경에서 Qini curve 분석을 위한 세 가지 다른 추정 전략을 제안합니다. 실험적 연구 설계를 통해 간섭이 있는 상황의 특성을 포착하고, 이와 관련된 편향-분산 무역(off) 문제를 다룹니다. 저자는 또한, 다양한 조건에 적용할 수 있는 추정 전략을 제공하여 확장성과 신뢰성을 높입니다.

- **Performance Highlights**: 이 연구는 군집 네트워크 간섭이 있는 시뮬레이션에서 제안된 세 가지 방법의 성능을 평가했습니다. 이론적 및 실증적 통찰력을 통해 최선의 추정 전략을 선택하는 방법을 제안하며, Qini curve의 정확도를 높이기 위한 가이드라인을 제공합니다. 저자들은 이 연구가 정책 평가 및 맞춤형 치료 접근에 있어 중요한 기여를 할 것이라고 주장합니다.



### RouteRL: Multi-agent reinforcement learning framework for urban route choice with autonomous vehicles (https://arxiv.org/abs/2502.20065)
- **What's New**: RouteRL은 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL)과 미세 교통 시뮬레이션을 통합한 새로운 프레임워크로, 자율주행차(autonomous vehicles, AV)의 효율적인 경로 선택 전략 개발에 기여합니다. 이 프레임워크는 도시 내에서의 운전 에이전트의 경로 선택을 시뮬레이션하며, 휴먼 운전자는 행동 경로 선택 모델을 통해, AV는 MARL 에이전트로 최적화된 정책을 추구합니다. RouteRL은 MARL, 교통 모델링, 인공지능(AI)과 인간 간의 상호작용 연구의 진전을 목표로 하며, 관련 기술 리포트를 제공합니다.

- **Technical Details**: RouteRL은 효율적인 경로 선택 전략을 찾는 다중 에이전트 의사결정 문제로 모델링됩니다. 이 시스템은 PETTINGZOO AEC API를 따르며, 다양한 MARL 알고리즘과 통합되어 사용자가 최신 알고리즘을 쉽게 활용할 수 있도록 설계되었습니다. RouteRL의 주요 구성 요소인 environment.TrafficEnvironment 클래스는 에이전트와 환경 간의 상호작용을 처리하고, 사용자 정의 시나리오 및 파라미터화 기능을 제공합니다.

- **Performance Highlights**: RouteRL을 이용하면 AV 도입 관련한 핵심 문제를 탐구할 수 있으며, 실험을 통해 도시 교통 시스템에 미치는 영향을 평가합니다. 연구를 통해 AV가 도입될 경우 다양한 시나리오에서의 총 여행 시간이나 사용자 행동의 효율성을 분석하며, 정책 결정자 및 교통 엔지니어에게 유용한 데이터를 제공합니다. 또한, RouteRL은 여러 교통 네트워크에서 11개의 모델을 포함하고 있어 연구에 다양한 활용성을 제공합니다.



### Hiring under Congestion and Algorithmic Monoculture: Value of Strategic Behavior (https://arxiv.org/abs/2502.20063)
- **What's New**: 이 연구는 기업들이 공유된 지원자 풀에서 경쟁하여 채용할 때 전략적 행동이 미치는 영향을 분석합니다. 모든 기업이 공통 알고리즘으로 지원자를 평가하고 인터뷰 결정을 동시에 내리며, 이로 인해 생기는 비효율성을 어떻게 조정할 수 있는지를 탐구합니다. 특히 나시 균형(Nash equilibrium)에서 사회적 후생이 개선됨을 보여주며, 기업들이 경쟁 정보를 알고 있을 때 전략적 행동을 통해 개선할 수 있다는 점을 강조합니다.

- **Technical Details**: 본 모델에서는 N개의 기업과 공유된 지원자 풀에서 각 지원자에게 점수가 부여됩니다. 각 기업은 용량 제약이 있는 인터뷰 결정 후, 인터뷰를 통과한 지원자에게 제안을 하며, 복수 제안을 받은 지원자는 무작위로 하나를 수락합니다. 이 연구는 세 가지 결정 모델, 즉 Naive, Nash equilibrium, Centralized를 통해 사회적 후생의 증가를 분석하며, 특별히 Naive 전략과 Naive 선택의 가격(Price of Naive Selection, PoNS)에 대해 설명합니다.

- **Performance Highlights**: 연구 결과, 나시 균형은 사회적 후생을 극대화하는 방향으로 작용하며, 중복된 인터뷰 선택으로 인한 비효율성을 감소시킵니다. 또한, 기업들이 서로의 경쟁 상황을 알 때, 더 많은 정보를 기반으로 전략적인 결정을 내릴 수 있음을 강조합니다. 효율성이 낮은 경우, 알고리즘을 활용한 채용 시스템이 정보의 공유 없이 운용될 경우, 비효율적인 결과를 초래할 수 있다는 점도 지적됩니다.



### Asymptotics of Non-Convex Generalized Linear Models in High-Dimensions: A proof of the replica formula (https://arxiv.org/abs/2502.20003)
- **What's New**: 이번 연구는 비볼록 (non-convex) 일반화 선형 모형 (Generalized Linear Models, GLMs)의 고차원 최적화 문제를 체계적으로 분석한 최초의 시도입니다. 기존의 많은 연구들이 LASSO와 같은 볼록 (convex) 문제에 집중되어 온 반면, 본 논문은 비볼록 상황에서의 예측 공식을 엄밀하게 증명하고 그 타당성을 규명하고자 했습니다. 혁신적인 접근 방식은 가우시안 최대-최소 정리 (Gaussian Min-Max Theorem)와 근사 메시지 전송 (Approximate Message Passing, AMP)이라는 두 강력한 이론 도구를 연결하는 것입니다.

- **Technical Details**: 주요 연구 내용으로는 비볼록 GLMs에 대한 replica-symmetric 공식을 제시하고, 이를 통해 이러한 공식이 유효한 조건을 명확히 규명하는 것입니다. 특히, Tukey 손실에 대한 최적성을 증명하고, 고차원 비볼록 회귀의 경우 부정 정규화의 최적성을 입증하며, 선형화된 AMP 알고리즘의 성능 한계를 특성화하는 등의 중요한 응용 사례를 다룹니다. 이러한 분석은 주로 통계 물리학 예측을 기반으로 하여 이루어졌으며, 이론적 기반을 강화합니다.

- **Performance Highlights**: 본 연구는 비볼록 최적화 문제에서 통계 물리학의 예측을 엄밀히 검증하며, 볼록 영역을 넘어 더욱 복잡한 최적화 환경을 분석하는 새로운 길을 열고자 합니다. 제시된 결과들은 물리학자들이 제기한 추측과 정확히 일치하며, 비볼록 분야의 최적화 문제에 대한 이해를 확장하는 데 기여할 것입니다. 최적의 손실 함수 및 알고리즘의 성능 한계에 대한 식별은 향후 머신러닝 및 통계 모델의 발전에 중대한 영향을 미칠 것으로 기대됩니다.



### Erasing Without Remembering: Safeguarding Knowledge Forgetting in Large Language Models (https://arxiv.org/abs/2502.19982)
- **What's New**: 이번 논문은 대규모 언어 모델(LLMs)에서의 기계 학습 삭제(machin learning unlearning)의 새로운 관점을 탐구합니다. 기존 방법들이 단순한 표현만을 지우는 데 그치는 경향을 보이는 반면, 우리는 paraphrased 또는 관련 정보가 여전히 남아있음을 지적합니다. 이러한 문제를 해결하기 위해 UGBench라는 새로운 벤치마크를 도입하여, 기존 LLM 삭제 방법들의 일반화 성능을 평가하고 있습니다.

- **Technical Details**: 이 논문은 PerMU라는 새로운 방법을 제안합니다. PerMU는 adversarial examples를 활용하여 단어 임베딩에 무작위 노이즈를 주입하고, 이를 통해 모델이 사실적 정보를 기억하지 못하도록 만듭니다. 특히, 우리는 주제 토큰(subject tokens)의 민감도를 평가하는 새로운 메트릭 MSM을 도입하여, 가장 민감한 토큰을 불러올 수 없도록 조정합니다.

- **Performance Highlights**: PerMU는 기존 방법들에 비해 최대 50.13%의 삭제 성능 향상과 43.53%의 강력한 일반화 성능 개선을 달성했습니다. 이를 통해 LLM의 유용성을 유지하고 높은 생성 품질도 유지하는 데 성공하고 있습니다. 논문의 주요 기여는 새로운 일반화 평가 벤치마크 UGBench와 관련 사실 기억을 방지하는 기계 학습 삭제 방법 PerMU의 제안입니다.



### Towards Collaborative Anti-Money Laundering Among Financial Institutions (https://arxiv.org/abs/2502.19952)
Comments:
          Accepted by International World Wide Web Conference (WWW) 2025

- **What's New**: 본 논문에서는 여러 금융 기관이 개별 거래 그래프를 노출하지 않고도 협력하여 자금 세탁 방지(AML)를 수행할 수 있도록 하는 새로운 알고리즘을 제안합니다. 이러한 협력적 설정에 맞춰 특화된 scatter-gather서브그래프 마이닝 알고리즘이 소개되었습니다. 기존의 중앙 집중형 거래 그래프에 대한 의존성을 극복하고, 복잡한 거래 패턴을 분석할 수 있는 새로운 방법을 제시하여 자금 세탁 활동을 보다 효과적으로 탐지할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 각 노드에 대해 BFS(너비 우선 탐색) 접근 방식을 사용하여 해당 노드와 연관된 기관 간의 거래 집합을 식별합니다. 서로 다른 기관에 속한 두 노드가 동일한 거래 집합을 공유하는 경우, 이는 자금 세탁 서브그래프 내에서 scatter-gather 관계가 존재함을 나타냅니다. 또한, 알고리즘은 각 금융 기관의 노드를 반복적으로 분석하여 교차 기관 거래 집합을 구성하며, 이를 통해 자금 세탁의 가능성을 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 교차 기관 자금 세탁 하위 그룹을 효과적으로 식별할 수 있는 능력을 보여주었습니다. Alipay와 E-Commerce Bank를 포함한 실제 데이터셋을 기반으로 하여 2억 개 이상의 계정과 3억 건의 거래를 포함하는 그래프에서 테스트가 수행되었습니다. 또한, 생성된 데이터셋에 대한 실험에서도 효율성을 입증하였으며, 수백만 건의 거래를 처리하는 데 몇 분밖에 걸리지 않았습니다.



### Meta-Reasoner: Dynamic Guidance for Optimized Inference-time Reasoning in Large Language Models (https://arxiv.org/abs/2502.19918)
Comments:
          Work in progress

- **What's New**: 이번 논문에서는 LLM들이 복잡한 문제를 해결하기 위해 사용하는 장기적인 추론 체인에 관한 새로운 접근 방식을 제시합니다. Meta-Reasoner라는 프레임워크를 통해 LLM이 추론 과정을 최적화하고, '생각하는 방식에 대해 생각하는' 능력을 배양하는 방법을 설명합니다. 이 프레임워크는 인간의 메타 인지 및 이중 과정 이론에서 영감을 받아 대안적 접근 방식을 제안하며, LLM의 계산 자원을 가장 유망한 경로로 재배치하도록 돕습니다.

- **Technical Details**: Meta-Reasoner는 LLM과 함께 작동하는 특수 모듈로, LLM의 추론 능력을 강화합니다. 이 메타 추론기는 고급 가이드를 제공하고 LLM의 추론 과정에서의 진행 상황을 동적으로 평가하는 역할을 합니다. LLM은 o1과 같은 부분적인 추론 체인을 생성하고 진행 상황을 요약한 보고서를 제공하는 반면, 메타 추론기는 이러한 보고서를 바탕으로 전략적 조언을 제공합니다. 이를 통해 LLM이 비효율적인 전략을 중단하고 더 효과적인 경로로 전환할 수 있도록 합니다.

- **Performance Highlights**: Meta-Reasoner는 수학적 및 과학적 추론 벤치마크에서 평가되었으며, 기존 방법 대비 정확성 및 효율성에서 유의미한 개선을 보였습니다. Game of 24, TheoremQA 및 SciBench와 같은 도전 과제를 통해 이 프레임워크가 추론 시간의 병목 현상을 해결할 수 있는 확장 가능한 솔루션임을 입증합니다. 또한 이 연구는 보다 넓은 응용 분야에 대한 가능성을 보여줍니다.



### Community Detection by ELPMeans: An Unsupervised Approach That Uses Laplacian Centrality and Clustering (https://arxiv.org/abs/2502.19895)
- **What's New**: 이번 논문에서는 최근 사회 네트워크의 증가로 인한 복잡한 커뮤니티 탐지를 다루는 ELPMeans라는 새로운 접근법을 제안합니다. ELPMeans는 Laplacian, Hierarchical Clustering, K-means 알고리즘을 결합하여 전체 네트워크에서 커뮤니티를 탐지합니다. 이 방법은 중심 노드 식별에 Laplacian 중심성과 최소 거리 메트릭을 사용하며, K-means 학습을 통한 효율적인 수렴을 보장합니다.

- **Technical Details**: ELPMeans는 비지도 학습 방법으로, 구현이 간단하면서도 중심 노드의 랜덤 초기화 문제, 커뮤니티 수(K) 찾기 등 일반적인 문제를 효과적으로 해결합니다. 논문에서는 Laplacian 중심성과 K-means 알고리즘을 사용하여 커뮤니티 구조를 형성하는 방법을 설명합니다. 이는 데이터의 비볼록 형태를 포함한 다양한 커뮤니티 탐지 작업에 적용 가능하다고 강조하고 있습니다.

- **Performance Highlights**: 실험 결과, ELPMeans는 최근의 다른 접근 방법들보다 정확성을 크게 개선하고 시간 복잡성을 줄였습니다. 실제 세계의 네트워크에서 뛰어난 성능을 발휘하여 다양한 커뮤니티 탐지 작업에 유용하다고 나타났습니다. 이로 인해 ELPMeans는 커뮤니티 탐지 분야에서 보다 광범위한 적용 가능성을 가지게 됩니다.



### A Multiple Transferable Neural Network Method with Domain Decomposition for Elliptic Interface Problems (https://arxiv.org/abs/2502.19893)
- **What's New**: 본 논문에서는 전이 가능한 신경망인 TransNet 기법과 비겹치는 영역 분해(nonoverlapping domain decomposition) 및 경계 조건(interface conditions)을 통합하여 다중 전이 가능한 신경망(Multi-TransNet) 방법을 개발했습니다. 이 방법은 불연속성이 있는 타원형 경계 문제를 해결하는데 효과적이며, 각 서브 도메인에 적응적으로 결정된 개수의 은닉층 뉴런을 부여하여 전체 계산 도메인에서 일정한 뉴런 배치를 유지합니다.

- **Technical Details**: Multi-TransNet 방법에서는 고유한 TransNet을 각 서브 도메인에 할당하고, 면접조건 항을 손실 함수에 통합하여 모든 서브 도메인 TransNet을 결합합니다. Empirical formula는 Multi-TransNet에 확장되어 서브 도메인 TransNet을 위한 적절한 뉴런 형태를 추정하는 데 사용되며, 이는 파라미터 조정 비용을 크게 줄이는 데 기여합니다. 또한 손실 함수의 항에 대한 가중치 파라미터를 적응적으로 선택하는 정규화 방법도 제안합니다.

- **Performance Highlights**: 다양한 타원형 경계 문제에 대한 광범위한 비교 실험을 수행하였으며, Multi-TransNet 방법의 정확도, 효율성 및 견고성이 우수함을 수치적으로 입증했습니다. 특히 고대비 확산계수를 가진 문제에서의 성능이 두드러지며, 낮은 차원부터 고차원까지의 문제에 효과적으로 적용 가능성을 보여주고 있습니다.



### Physics-Informed Neural Networks for Optimal Vaccination Plan in SIR Epidemic Models (https://arxiv.org/abs/2502.19890)
- **What's New**: 이번 연구는 감염 및 회복 비율이 일정한 통제된 SIR(Susceptible-Infectious-Recovered) 모델의 최소 제거 시간(minimum eradication time)을 이해하는 데 초점을 맞추고 있습니다. 저자들은 물리 정보 신경망(Physics-Informed Neural Networks, PINNs)을 활용하여 제거 시간을 지배하는 편미분 방정식(partial differential equation, PDE)을 해결하고 가장 적합한 예방접종 통제를 도출합니다. 이 연구는 전통적인 수치 방법에 대한 효율적인 계산 대안을 제공하며, SIR 모델에서의 전염병 예측과 통제를 위한 새로운 PINN 응용 사례를 제시합니다.

- **Technical Details**: 연구에서는 제거 시간이 일정한 전염 역학을 설명하는 방향으로 설정된 HJB(Hamilton-Jacobi-Bellman) 방정식의 점성 해(solution)에 기초하여 PINNs를 이용한 새롭고 효율적인 접근 방법을 제안합니다. PINN 프레임워크는 깊은 신경망(deep neural network)의 손실 함수(loss function)에 바로 역학을 내재화해 mesh-free 솔루션을 가능하게 함으로써 PDE를 해결합니다. 특히, 변수 스케일링(variable scaling) 방법이 안정적인 PINN 훈련을 보장하는 데 효과적이라는 수학적 분석도 함께 제시됩니다.

- **Performance Highlights**: 수치 실험을 통해 제안한 방법의 효과를 검증하며, 최소 제거 시간을 계산하고 최적의 통제 전략을 달성하는 데 성공했습니다. PINNs는 전염병 모델링에서 강력하고 유연한 프레임워크로 자리잡고 있으며, 이에 대한 문헌도 폭넓게 존재하는 반면, 최적 제어 문제에 대한 PINNs의 응용은 여전히 상대적으로 미개척 상태입니다. 본 연구는 이러한 갭을 해소하는 데 기여하며, SIR 모델의 예방접종 전략을 처리하는 효율적인 방법을 제시합니다.



### NeRFCom: Feature Transform Coding Meets Neural Radiance Field for Free-View 3D Scene Semantic Transmission (https://arxiv.org/abs/2502.19873)
- **What's New**: 신규 통신 시스템인 NeRFCom을 소개합니다. NeRFCom은 3D 장면 전송을 위해 설계되었으며, 전통적인 방식에 비해 비선형 변환과 학습된 확률 모델을 활용하여 더 유연한 소스-채널 코딩을 가능하게 합니다. 이 시스템은 데이터 전송 시 각 3D 장면의 세부 특성에 따라 대역폭을 효율적으로 할당합니다.

- **Technical Details**: NeRFCom은 3D 특징을 저차원 잠재 표현으로 인코딩하기 위해 신경 기반 비선형 변환을 개발합니다. 이어서, 정보 엔트로피를 평가하는 학습된 엔트로피 모델을 도입하여 각 잠재 표현 요소가 3D 장면 합성 충실도에 미치는 기여도를 반영합니다. 이 시스템은 최적의 전송률-왜곡 목표를 달성하기 위해 모든 구성 요소를 신경망으로 최적화합니다.

- **Performance Highlights**: 실험 결과, NeRFCom은 가혹한 채널 조건에서도 높은 신뢰성을 유지하면서 자유 시점의 3D 장면을 효율적으로 전송합니다. 기존의 방법과 비교하여, NeRFCom은 전송 품질과 데이터 전송 속도 모두에서 의미 있는 개선을 보여줍니다. 이러한 특징 덕분에 VR 및 AR 기술의 발전에 크게 기여할 것으로 기대됩니다.



### Beyond Worst-Case Dimensionality Reduction for Sparse Vectors (https://arxiv.org/abs/2502.19865)
Comments:
          To appear in ICLR 2025

- **What's New**: 본 논문은 $s$-희소 벡터(s-sparse vectors)의 최악의 경우 차원 축소(beyond worst-case dimensionality reduction)를 연구합니다. 연구는 두 가지 부분으로 나뉘어 있으며, 각각의 부분은 평균적인 경우(average-case) 보장을 다룹니다. 먼저, $	ext{log(n/ε^2)}$ 차원으로의 랜덤 선형 맵(mapping)을 통해 99%의 벡터 노름(norm)을 정확히 보존할 수 있다는 상한선을 제공합니다. 또한, 비선형(non-linear) 임베딩을 통해 희소 비음수 벡터(sparse non-negative vectors)의 거리 보존을 위한 새로운 하한을 제시합니다.

- **Technical Details**: 차원 축소에서 희소성이 중요한 역할을 한다는 점에 주목하여, 이 논문은 희소 비음수 벡터를 이용한 비선형 임베딩 방식을 탐구합니다. 연구에 따르면, 데이터셋의 모든 쌍 거리(pairwise distances)를 특정 오차 범위 내에서 보존할 수 있으며, 이는 기존의 최악의 경우 제한을 넘어서는 결과입니다. 또한, 논문은 또한 $\ell_\infty$ 노름(norm)에 대한 정확한 차원 축소를 보장하며, 이는 O(s log |X|) 차원으로 축소됩니다. 이러한 방법은 실질적인 데이터 처리 및 알고리즘 개선에 기여할 수 있습니다.

- **Performance Highlights**: 논문에서 제안한 임베딩의 특징으로는, 희소 비음수 벡터에 대해 모든 쌍 거리의 보존이 가능하며, 이는 기존의 대칭성과 희소성 한계를 뛰어넘습니다. $p \ge 1$에 대해 최적의 임베딩 차원을 제공하고, 임베드된 차원 수는 희소 비음수 조건 하에서 크게 감소합니다. 결과적으로, 이 연구는 데이터 과학 및 기계 학습 분야의 실용적인 알고리즘 개선을 위한 새로운 기초를 제공합니다.



### Fast Debiasing of the LASSO Estimator (https://arxiv.org/abs/2502.19825)
- **What's New**: 이 논문에서는 고차원 희소 회귀(high-dimensional sparse regression)에서 발생하는 Lasso 추정치의 편향(bias)을 해결하기 위한 새로운 접근법을 제시합니다. 기존의 방법이 반복적인 최적화 과정을 필요로 하는 반면, 저자들은 최적화 문제를 재구성하여 '디바이싱 매트릭스'를 직접 계산하도록 하여 계산 효율성을 크게 향상시켰습니다. 이를 통해 Lasso 추정치의 이론적 보장을 유지하면서도 계산에서의 부담을 줄입니다.

- **Technical Details**: 저자들은 행렬 ${A}$의 유사한 조건 하에 '웨이트 매트릭스' ${W}$를 사용하여 편향을 교정하는 새로운 방법론을 도입합니다. 이 방법은 원래 방법에서 요구되는 계산적인 복잡성을 제거하며, 유일한 최적 해(solution) 보장을 제공합니다. 이러한 접근 방식은 예측 행렬의 각 행의 성분들이 서로 독립적(uncorrelated)인 경우에 특히 유효합니다.

- **Performance Highlights**: 저자들은 수치 시뮬레이션을 통해 제안된 방법의 효과성을 검증하였으며, 직관적으로 계산 효율성을 향상시키면서도 이론적 특성을 보존할 수 있음을 보여주었습니다. 새로 제안된 방법은 랜덤 서브-가우시안 감지 행렬을 사용할 경우에도 실질적으로 유용함을 입증했습니다. 이는 고차원 통계 분석에서 신뢰할 수 있는 추론을 수행하는 데 중요한 기여를 할 것으로 예상됩니다.



### Shared Stochastic Gaussian Process Latent Variable Models: A Multi-modal Generative Model for Quasar Spectra (https://arxiv.org/abs/2502.19824)
Comments:
          Published in TMLR, this https URL. The code for this work is available at: this https URL

- **What's New**: 이 연구는 다수의 관측 공간에서 작동하는 확장 가능한 확률론적 잠재 변수 모델을 제안합니다. 천체 물리학의 응용 분야에 초점을 맞추어, 퀘이사(quasar) 스펙트럼과 그 과학적 속성을 동시에 생성할 수 있는 통합 generative model을 개발했습니다. 또한, 데이터 세트에서 일부 차원이 결측일 때도 훈련이 가능하다는 점에서 중요한 기여를 합니다.

- **Technical Details**: 이 모델은 Gaussian processes(GP)를 기반으로 한 확장된 GPLVM(Gaussian Process Latent Variable Model)으로, 단일 데이터 포인트를 다양한 관측 클래스에서 다르게 나타내게 합니다. 특히, 협력적인 잠재 공간(shared latent space)을 사용하여 각기 다른 Gaussian process 디코더에 입력되도록 하여 관측 공간 간의 예측력을 공유할 수 있습니다. 또한, 데이터가 결측된 상황에서도 모델을 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해, 제안된 모델이 퀘이사의 스펙트럼과 과학적 레이블을 고충실도로 재구성할 수 있음을 입증하였으며, 결측 스펙트럼 리저를 보간하고 과학적 레이블을 예측할 수 있음을 보여주었습니다. 특히, 이 모델은 여러 관측 공간에 걸쳐 공통 잠재 변수를 학습함으로써 예측력을 극대화하는 데 성공했습니다. 이러한 접근법은 천체 물리학의 복잡한 데이터 분석에 필요한 혁신적인 솔루션을 제공합니다.



### Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts (https://arxiv.org/abs/2502.19811)
- **What's New**: COMET는 Mixture-of-Experts (MoE) 시스템의 새로운 최적화 버전으로, 통신과 계산의 정밀한 겹침(fine-grained overlapping)을 통해 성능을 극대화합니다. 기존의 MoE 시스템에서 보이는 통신 오버헤드를 줄이고, 대규모 분산 환경에서 그 효율성 향상을 가져옵니다. 이를 통해 MoE 모델의 GPU 사용 시간을 효율적으로 절약할 수 있습니다.

- **Technical Details**: COMET는 복잡한 데이터 의존성 분석(data dependency analysis)과 작업 재조정(task rescheduling) 기법을 이용하여 통신과 계산의 정밀한 겹침을 달성합니다. 이 시스템은 GPU 하드웨어 리소스를 동적으로 할당하여 연산과 통신 간의 부하를 균형 있게 조절합니다. 또한 공유 텐서(shared tensor)를 분석하여 두 작업 간의 겹침을 극대화하여 동시 실행을 확보합니다.

- **Performance Highlights**: COMET는 MoE 레이어의 실행 속도를 약 1.96배 가속화하며, 전체 MoE 모델 실행에 대해서는 평균 1.71배의 성능 향상을 보여주었습니다. 또한 10,000개 이상의 GPU로 구성된 생산 환경에서도 적용되어 수백만 시간의 GPU 시간을 절약하는 성과를 거두었습니다. COMET는 Megatron-LM에 통합되어 여러 병렬 전략에서 그 효과를 검증하였습니다.



### SCU: An Efficient Machine Unlearning Scheme for Deep Learning Enabled Semantic Communications (https://arxiv.org/abs/2502.19785)
- **What's New**: 이 논문에서는 심층 학습(Deep Learning) 기반의 의미 통신(semantic communications)에서 개인 정보 보호를 위한 데이터 삭제 문제를 다룰 새롭고 효율적인 방법인 의미 통신 언러닝(Semantic Communication Unlearning, SCU)을 제안합니다. SCU는 인코더(encoder)와 디코더(decoder) 간의 학습된 의미 표현에서 특정 샘플의 정보를 제거하여 모델을 언러닝합니다. 이를 통해 사용자의 데이터 삭제 요구를 보다 효과적으로 충족할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: SCU는 두 가지 주요 컴포넌트로 구성됩니다. 첫 번째로, 인코더와 디코더의 의미 코덱을 위한 공동 언러닝(joint unlearning) 방법을 맞춤화하여 학습된 의미 표현과 삭제된 샘플 간의 상호 정보(mutual information)를 최소화합니다. 두 번째로, 언러닝의 결과로 모델 유용성의 저하를 보상하기 위해 대조적 보상(contrastive compensation) 방법을 제안하여 기존 데이터에서 긍정 샘플과 삭제된 데이터를 부정 샘플로 고려하여 모델을 재교육합니다.

- **Performance Highlights**: 제안된 SCU 방법은 여러 실험 데이터를 통해 기존 최첨단(unlearning) 방법들에 비해 더 뛰어난 유효성(efficiency)과 효과성(effectiveness)을 입증하였습니다. 특히, SCU는 reconstruction mean squared error (MSE) 및 다운스트림(Downstream) 백도어 탐지에서 우수한 성능을 보였으며, 이러한 복잡한 문제를 효율적으로 해결하는 새로운 도구를 제공합니다. 논문은 SCU의 소스 코드도 공개하여 머신 언러닝 분야의 더 나은 연구와 개발을 촉진하고 있습니다.



### Do Retrieval-Augmented Language Models Adapt to Varying User Needs? (https://arxiv.org/abs/2502.19779)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Language Models (RALMs) 평가를 위한 새로운 프레임워크를 제안했습니다. 사용자 요구에 따라 모델의 응답 특성을 측정하는 데 초점을 맞추었으며, 세 가지 사용자 케이스(Context-Exclusive, Context-First, Memory-First)와 세 가지 컨텍스트 설정(Context Matching, Knowledge Conflict, Information Irrelevant)을 도입하여 실제 애플리케이션의 복잡성을 반영하고자 했습니다.

- **Technical Details**: 이 프레임워크는 사용자가 요구하는 정보의 유형에 따라 모델이 어떻게 반응하는지를 체계적으로 평가합니다. 사용자는 외부 정보와 내부 지식 중 어떤 것을 우선시할지를 지시할 수 있으며, 이러한 지시사항에 따라 모델의 성능이 어떻게 달라지는지 분석합니다. 우리의 실험은 URAQ 데이터셋을 포함하여 여러 QA 데이터셋에서 수행되었으며, 두 개의 모델 계열(Llama3.1, Qwen2.5)에 대해 다양한 모델 크기와 검색된 컨텍스트 수로 평가되었습니다.

- **Performance Highlights**: 주요 실험 결과에 따르면 현재의 언어 모델은 다양한 사용자 요구를 충족시키는 데 어려움을 겪고 있으며, 모든 데이터셋에서 50% 이하의 정확도를 기록했습니다. 또한, 컨텍스트 제약이 모델 성능에 미치는 영향을 확인했으며, 이상적인 검색 결과에서는 성능이 저하되는 경향이 있음을 발견했습니다. 모델 계열에 따라 성능 차이가 두드러지며, 특정 상황에서는 특정 모델이 다른 모델보다 우수한 성능을 발휘하는 것으로 나타났습니다.



### TAPE: Tailored Posterior Difference for Auditing of Machine Unlearning (https://arxiv.org/abs/2502.19770)
- **What's New**: 본 연구는 머신 언러닝(ML) 감사(auditing) 과정을 독립적으로 수행할 수 있는 TAPE라는 새로운 방법을 제안합니다. 기존의 백도어(backdoor) 기반 접근 방식들은 비효율적이고 실용적이지 않으며, 그 과정에서 모델 훈련에 개입해야 하는 한계를 가지고 있습니다. TAPE는 원본 모델 훈련과는 무관하게 모델의 차이를 이용하여 언러닝(unlearning) 정보를 평가하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: TAPE 방법은 두 가지 주요 요소로 구성되어 있습니다: 첫째, 언러닝 그림자 모델(shadow model)을 신속하게 구축하여 언러닝 후의 포스터리어 차이를 모방합니다. 둘째, 비공식 모델(Reconstructor model)을 훈련시켜 포스터리어 차이를 통해 언러닝된 개인정보를 추출 및 평가합니다. 연구팀은 다중 샘플 언러닝을 효과적으로 처리하기 위해 두 가지 전략, 즉 언러닝 데이터 섭동(unlearned data perturbation)과 영향을 기반으로 한 분리(unlearned influence-based division)를 제안합니다.

- **Performance Highlights**: 정량적 실험 결과, TAPE는 기존의 언러닝 검증 방법에 비해 적어도 4.5배의 효율성을 보여주었으며, 특정 데이터셋에서는 최대 75배의 속도 향상을 기록했습니다. TAPE는 실제 언러닝 샘플에 대한 감사 기능을 효과적으로 제공하며, 기존의 백도어 기반 방법들이 처치할 수 없는 정당 데이터에 대한 감사를 지원합니다. 이로 인해 TAPE는 다양한 언러닝 시나리오 및 요청을 지원하는 훌륭한 도구로 자리잡을 것입니다.



### EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models (https://arxiv.org/abs/2502.19765)
- **What's New**: EdiText라는 새로운 텍스트 편집 방법이 제안되었습니다. 이 방법은 참조 텍스트를 다양한 속성으로 수정할 수 있게 해주는 SDEdit 기반의 편집 기술을 통합합니다. EdiText는 자가 조건화(self-conditioning) 기법을 기반으로 한 미세 편집 방법을 도입하여 기존 텍스트를 세밀하게 조정할 수 있는 기능을 제공합니다.

- **Technical Details**: EdiText는 임베딩 확산 모델(embedding diffusion model)을 사용하는 텍스트 편집 프레임워크로, 글로벌 수준과 미세 수준의 편집을 모두 지원합니다. Latent Diffusion for Language Generation (LD4LG) 모델을 기반으로 하여, 이 텍스트 편집 방법은 이산 데이터가 연속 데이터로 변환되고, 혼합 확산 프로세스를 통해 모델링됩니다. 이러한 접근 방식은 다양한 텍스트 속성을 편집할 수 있도록 해줍니다.

- **Performance Highlights**: 제안된 EdiText 방법은 여러 작업에서 탁월한 편집 성능을 보여주며, 기존 모델보다 더 넓고 미세한 범위에서 편집이 가능합니다. Coarse-level and fine-level 편집 기술을 통합함으로써, EdiText는 보다 포괄적이고 정밀한 편집 과정을 보장합니다. 이는 다양한 결과물을 요구하는 작업에서 더 나은 성능을 발휘하게 합니다.



### Inexact Moreau Envelope Lagrangian Method for Non-Convex Constrained Optimization under Local Error Bound Conditions on Constraint Functions (https://arxiv.org/abs/2502.19764)
- **What's New**: 이 논문에서는 단순한 다면체 위의 부드러운 비볼록 최적화 문제를 해결하기 위해 이항 모로우 엔벨로프 라그랑주(iMELa) 방법을 연구합니다. 전통적인 라그랑주 함수에 프로시말(proximal) 항을 추가함으로써, iMELa 방법은 각 주요 반복(iteration)에서 다면체 집합 위에서 볼록 최적화 부문제를 근사적으로 해결합니다. 이 연구의 주요 목표는 FOM(First-Order Methods)을 사용해 epsilon-Karush-Kuhn-Tucker (KKT) 포인트를 계산하고자 하는 것입니다.

- **Technical Details**: 이 논문에서 연구된 iMELa 방법은 근사 증가형 라그랑주 방법(augmented Lagrangian method, ALM)과 관련이 있습니다. 슬레이터 조건(Slater's condition)과 제약 집합에 대한 지역 오류 경계 조건(local error bound condition)을 가정하여, iMELa 방법이 O~(ϵ−2) 복잡도로 epsilon-KKT 포인트를 찾을 수 있음을 보여줍니다. 여기서 복잡도는 기본적으로 로그 요소를 생략한 것입니다.

- **Performance Highlights**: 이 논문의 복잡도 결과는 기존 방법보다 약한 제약 조건을 활용하여 적용 가능성이 높은 결과를 제공합니다. 이 조건은 새로운 최적화 알고리즘의 수렴 속성을 이해하는 데 도움을 줍니다. 또한, 기존 논문에서 다루지 않은 경우에도 이 복잡도 결과가 적용될 수 있음을 강조합니다.



### PolyPrompt: Automating Knowledge Extraction from Multilingual Language Models with Dynamic Prompt Generation (https://arxiv.org/abs/2502.19756)
Comments:
          6 pages, 2 figures

- **What's New**: 이 연구에서는 다국어 모델의 성능 편차를 줄이기 위해 PolyPrompt라는 새로운 자동 프롬프트 생성 프레임워크를 소개합니다. PolyPrompt는 입력 언어에 따라 트리거 토큰을 동적으로 학습하고 적용하여 모델의 다국어 작업 성능을 크게 향상시킵니다. 실험 결과, 이 방법은 일반적인 기반선과 비교하여 3.7%에서 19.9%의 정확도 향상을 나타내며, 다양한 언어에서 그 효과가 입증되었습니다.

- **Technical Details**: Tλ라는 언어별 트리거 임베딩을 학습하여, 각 언어에 대해 k 개의 학습 가능한 임베딩을 사용합니다. 입력 쿼리의 언어를 감지하고, 해당 언어의 트리거 임베딩을 쿼리에 추가하여 모델에 공급합니다. 이 과정에서 모든 모델 파라미터는 고정되어 있고, 트리거 임베딩만이 업데이트됩니다.

- **Performance Highlights**: PolyPrompt는 글로벌 MMLU 벤치마크에서 15개 다양한 언어로 평가되었으며, 그 결과 약 1억 개의 매개변수를 가진 모델에서 효과적임을 증명합니다. 이 연구는 언어 모델 성능의 단점을 완화하고, 다양한 언어에서 보다 높은 효율성을 보장하는 것을 목표로 합니다.



### Bridging the PLC Binary Analysis Gap: A Cross-Compiler Dataset and Neural Framework for Industrial Control Systems (https://arxiv.org/abs/2502.19725)
- **What's New**: PLC-BEAD(PLC Binary Evaluation and Analysis Dataset)는 700개 이상의 PLC 프로그램에서 2431개의 컴파일된 바이너리를 포함하는 종합적인 데이터셋입니다. 이는 네 개의 주요 산업 컴파일러(CoDeSys, GEB, OpenPLC-V2, OpenPLC-V3)에서 생성된 바이너리와 그에 해당하는 원본 Structured Text 소스 코드, 그리고 표준화된 기능 라벨을 쌍으로 제공합니다. 이로 인해 바이너리와 소스 코드 분석을 모두 가능하게 하여 PLC 보안 연구와 역공학 분야에 큰 기여를 할 수 있습니다.

- **Technical Details**: PLC-BEAD는 IEC 61131-3 표준에 따라 작성된 Structured Text 소스 코드와 독특하게 쌍을 이루며, 다양한 PLC 프로그램을 포함합니다. 바이너리는 각기 다른 컴파일러(CoDeSys, GEB, OpenPLC-V2, OpenPLC-V3)를 통해 생성되며, 기능성 분류를 위한 22개의 카테고리로 라벨링되어 있습니다. 이 데이터셋은 머신러닝 기반의 분석 기법 개발에 중요한 토대를 마련하게 됩니다.

- **Performance Highlights**: PLCEmbed는 PLC 바이너리 분석을 위해 설계된 transformer 기반의 프레임워크로, 93%의 정확도로 컴파일러 출처 확인을 수행하고, 42%의 정확도로 세분화된 기능 분류를 달성합니다. 이 연구는 PLC 바이너리 분석의 복잡성을 강조하면서, 데이터 기반 접근 방식이 산업 사이버 보안 관행 강화에 어떻게 기여할 수 있는지를 보여줍니다.



### Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning (https://arxiv.org/abs/2502.19717)
Comments:
          Accepted by the Thirteenth International Conference on Learning Representations (ICLR 2025)

- **What's New**: 이번 연구에서는 협력적 다중 에이전트 강화 학습(MARL)에서 확장 가능한 커뮤니케이션 프로토콜을 개발하는 데 중점을 두고 있습니다. 이전의 방법들이 최적의 쌍방 통신 링크 선택에 집중한 반면, 제안된 방법은 전 세계적인 관점에서의 커뮤니케이션 토폴로지 설계를 채택하고 있습니다. 이를 통해 ExpoComm이라는 이름의 확장 가능한 커뮤니케이션 프로토콜을 제안하며, 지수적(topology) 구조에 의해 정보 전파를 빠르게 수행할 수 있도록 합니다.

- **Technical Details**: ExpoComm은 에이전트 간의 효과적인 정보를 전파하기 위해 지수적 토폴로지를 기반으로 설계되었습니다. 이는 고유한 소규모 속성과 작은 직경 특성을 활용하여, 모든 에이전트 간의 메시지 흐름을 효율적으로 지원합니다. 또한, memory-based message processors와 보조 작업을 활용하여 전 세계 정보를 반영하는 메시지를 생성함으로써 의사결정을 지원합니다.

- **Performance Highlights**: 대규모 협력 벤치마크인 MAgent와 Infrastructure Management Planning을 포함한 12개의 시나리오에서 ExpoComm의 우수한 성능과 강력한 zero-shot 전이 가능성을 입증하였습니다. 특히, 여러 에이전트를 처리하는 데 있어 기존의 통신 전략에 비해 월등한 성능을 보여주며, 에이전트 수에 관계없이 저렴한 비용으로 효과적인 커뮤니케이션을 가능하게 합니다.



### Recent Advances on Generalizable Diffusion-generated Image Detection (https://arxiv.org/abs/2502.19716)
- **What's New**: 최근 확산 모델(difusion models)의 발전은 생성된 이미지의 충실도(fidelity)와 다양성(diversity)을 크게 향상시켰습니다. 하지만 이러한 발전은 고품질의 딥페이크(Deepfake) 이미지를 만드는 데 악용될 수 있어 이미지 신뢰성 검증에 도전 과제를 안깁니다. 이에 따라 생성된 이미지 탐지에 대한 연구가 급증하고 있지만, 이 주제에 대한 포괄적 리뷰는 여전히 부족합니다.

- **Technical Details**: 이 논문에서는 최근의 발전을 체계적으로 조사하고 이들을 두 가지 주요 범주로 분류하여 제시합니다: (1) 데이터 기반(datat-driven) 탐지 및 (2) 특성 기반(feature-driven) 탐지. 탐지 방법들은 기본 원리에 따라 여섯 가지 세부 범주로 세분화됩니다. 데이터 기반 탐지 방법은 명시적인 수작업 특징에 의존하지 않고, 데이터 주도적인 방식으로 일반화 가능한 특징을 포착하는 능력을 향상시킵니다.

- **Performance Highlights**: 링크된 연구는 주로 두 가지 유형으로 나뉘어집니다. 첫 번째 유형은 사람에게 인식 가능한 이미지 특징(Perceptible Image Features)을 사용하는 것이고, 두 번째는 인식 불가능한 이미지 특징(Imperceptible Image Features)을 분석하는 것입니다. 탐지 방법의 내구성(post-processing에 대한 저항력) 및 더 강력한 이론적 토대의 필요성도 강조되며, 고품질의 다양한 데이터셋의 개발과 같이 이 분야에서의 향후 연구 방향이 제시됩니다.



### BEVDiffuser: Plug-and-Play Diffusion Model for BEV Denoising with Ground-Truth Guidanc (https://arxiv.org/abs/2502.19694)
Comments:
          CVPR 2025

- **What's New**: 본 연구에서는 BEV (Bird's-eye-view) 표현의 노이즈 문제를 해결하기 위한 새로운 확산 모델인 BEVDiffuser를 제안합니다. BEVDiffuser는 실제 객체 레이아웃을 가이던스로 하여 BEV 특징 맵을 효과적으로 노이즈 제거합니다. 이 모델은 기존 BEV 모델을 수정할 필요 없이 플러그 앤 플레이 방식으로 훈련 중에 작동하여 BEV 표현을 향상시킵니다.

- **Technical Details**: BEVDiffuser는 BEVFormer, BEVFusion과 같은 기존 BEV 모델에서 생성된 특징 맵에 다양한 수준의 노이즈를 추가하여 훈련됩니다. 훈련 후에는 BEV 특징 맵을 정화하여 추가적인 감독을 제공하는 방식으로 기존 BEV 모델의 성능을 개선합니다. 또한, BEVDiffuser는 훈련 시간이 끝나면 제거되며, 추론 시 추가적인 컴퓨팅 지연 없이도 강력한 성능을 유지합니다.

- **Performance Highlights**: nuScenes 데이터셋을 통해 실시한 실험 결과, BEVDiffuser는 3D 객체 탐지에서 mAP 12.3% 및 NDS 10.1%의 현저한 개선을 보여주었습니다. 장기적인 객체 탐지 및 다양한 환경 조건에서도 성능이 크게 향상되었으며, 질적으로도 고품질의 BEV 생성 능력을 입증했습니다. 이러한 성능 개선은 자율 주행의 발전을 위한 대규모 데이터 수집에 기여할 것으로 기대됩니다.



### Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack (https://arxiv.org/abs/2502.19672)
Comments:
          arXiv admin note: text overlap with arXiv:2403.09766

- **What's New**: 최근 미디엄 대형 언어 모델(MLLMs)에 대한 관심이 커지고 있으며, 이러한 모델들은 이미지 인식 및 이해 능력에서 주목받고 있습니다. 하지만 MLLM들은 적대적 공격에 취약하며, 이러한 공격이 다른 모델에서 전이되는 능력은 제한적입니다. 본 연구에서는 다이나믹 비전-언어 정렬(DynVLA) 공격을 소개하여 다른 모델 간의 비전-언어 정렬을 개선하고자 합니다.

- **Technical Details**: DynVLA 공격은 비전-언어 커넥터에 동적인 섭동(perturbations)을 주입하여 MLLM의 비전-언어 모달리티 정렬을 동적으로 변화시킵니다. 기존의 방법들은 단일 비전-언어 정렬에 기반한 엔드-투-엔드 최적화 방식을 사용하지만, DynVLA는 주의(attention) 메커니즘을 변경하여 다양한 비전-언어 모달리티 정렬을 수용합니다. 이는 가우시안 커널(Gaussian kernel)을 주의 맵에 적용하여 모델의 주의를 이미지의 다른 영역으로 이동시킵니다.

- **Performance Highlights**: DynVLA는 BLIP2, InstructBLIP, MiniGPT4, LLaVA 등 다양한 MLLM에서 적대적 예제의 전이 가능성을 크게 향상시킵니다. 우리의 방법은 DIM, SIA와 같은 기존의 전통적 공격 방법들과 비교하여 뛰어난 성능을 보이며, 이는 특히 MLLM의 아키텍처와 크기가 전이 가능성에 중요한 역할을 한다는 것을 보여줍니다. 결과적으로, DynVLA는 최소한의 사전 지식으로도 모델을 공격할 수 있어, 실질적인 보안 위협을 야기할 수 있는 가능성을 가지고 있습니다.



### SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning (https://arxiv.org/abs/2502.19668)
- **What's New**: 본 논문에서는 심혈관 질환 진단을 위한 Electrocardiogram (ECG) 해석의 새로운 접근 방식을 제안합니다. 기존의 ECG Self-Supervised Learning (eSSL) 방법의 한계를 극복하고자 SuPreME이라는 감독형 사전 훈련 프레임워크를 도입하여, 미리 정리된 임상 정보를 활용한 고품질의 정밀한 레이블 데이터셋을 생성합니다.

- **Technical Details**: SuPreME은 Large Language Models (LLMs)를 활용하여 자유 텍스트 ECG 보고서에서 구조화된 임상 엔티티를 추출하고, 노이즈와 불필요한 내용을 필터링합니다. 이 프레임워크는 구조화된 엔티티 레이블과 ECG 신호를 직접 정렬하여, 기존의 복잡한 사전 작업을 피하고 데이터 증강에 대한 의존성을 줄입니다.

- **Performance Highlights**: SuPreME는 127개의 심장 질환을 포함한 6개의 후속 데이터 세트에서 평가되었으며, 제로 샷(zero-shot) 분류에서 1.96% 이상의 성능 개선을 달성했습니다. 이 모델은 전체 fine-tuning 없이도 높은 데이터 효율성을 바탕으로 학습되며, 효과적으로 높은 품질의 ECG 표현을 생성함을 입증하였습니다.



### Spectral Analysis of Representational Similarity with Limited Neurons (https://arxiv.org/abs/2502.19648)
- **What's New**: 이번 연구에서는 신경 기록(neural recordings)과 계산 모델의 표현 유사성 측정을 위한 새로운 이론적 프레임워크를 제시하였습니다. 특히, 유한한 뉴런 샘플링이 유사성 측정에 미치는 영향을 분석하며, Canonical Correlation Analysis (CCA)와 Centered Kernel Alignment (CKA) 기법을 중심으로 진행되었습니다. 이는 유한한 뉴런 수집으로 인해 발생하는 고유 벡터의 비국소화(eigenvector delocalization) 문제를 해결하기 위한 방법으로, 표본의 크기가 작을 때 발생하는 실제 문제를 다루고 있습니다.

- **Technical Details**: 우리는 랜덤 매트릭스 이론(Random Matrix Theory)을 활용하여, CCA 및 CKA와 같은 유사성 측정 방식의 예측 스펙트럼 프레임워크를 개발했습니다. 연구에서 제안한 방법은 제한된 데이터로부터 모집단 수준(population-level) 유사성을 추론할 수 있는 노이즈 제거 방법(denoising method)을 포함하고 있습니다. 이러한 접근 방식은 신경 데이터(neural data)의 해석을 돕는 실용적인 전략을 제공하며, 기존의 방법들에 비해 보다 정확한 유사성 분석이 가능하게 합니다.

- **Performance Highlights**: 진짜 신경 데이터에 적용한 결과, 적은 수의 뉴런으로도 모델-뇌 유사성의 심각한 과소 추정이 발생할 수 있음을 확인했습니다. 제안된 방법은 이러한 신호의 손실을 효율적으로 복구하여, 제한된 샘플 내에서 더욱 신뢰성 있는 결과를 제공하였습니다. 연구 결과는 CCA와 CKA 측정법을 사용할 때 뉴런 샘플의 수가 유사성 측정을 오도하게 만들 수 있는 방식을 실증적으로 보여주며, 이는 향후 신경망 아키텍처와 학습 방법 평가에 필수적입니다.



### AutoBS: Autonomous Base Station Deployment Framework with Reinforcement Learning and Digital Twin Network (https://arxiv.org/abs/2502.19647)
- **What's New**: AutoBS는 6G 네트워크에서 최적의 기지국 (Base Station, BS) 배치를 위해 강화 학습 (Reinforcement Learning, RL) 기반의 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 근접 정책 최적화(Proximal Policy Optimization, PPO) 알고리즘과 PMNet을 활용하여 빠르게 경로 손실을 예측함으로써 커버리지와 용량을 최적화할 수 있는 배치 전략을 학습합니다. AutoBS는 기존의 수작업 문제 해결 방식에서 벗어나 실시간 애플리케이션에 적합하게 개발되었습니다.

- **Technical Details**: AutoBS는 구체적인 사이트 환경을 반영하여 최적의 BS 배치를 자동으로 최적화합니다. BS의 위치는 환경 맵에서 주어진 좌표 (i,j)로 표현되며, 목표는 커버리지(Vm)와 용량(Cm)를 조정하여 최적화를 수행하는 것입니다. PMNet은 신속한 경로 손실 예측을 가능하게 하여 PPO가 각 배치 결정 후 즉시 네트워크 성능을 평가하고 보상 피드백을 받을 수 있도록 합니다.

- **Performance Highlights**: 수치 결과에 따르면 AutoBS는 단일 BS의 경우 95%, 다수의 BS의 경우 90%의 성능을 보여주며, 기존의 전체 탐색 방법보다 inference 시간을 몇 시간에서 밀리세컨드로 단축합니다. 또한, AutoBS는 동적 환경에서도 컴퓨팅 오버헤드를 최소화하며 대규모 6G 네트워크에 잘 맞춰져 있습니다. 이러한 장점들 덕분에 AutoBS는 진정한 실시간 최적화 솔루션으로 자리잡고 있습니다.



### Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (https://arxiv.org/abs/2502.19645)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 Vision-Language-Action 모델(VLAs)의 최적화된 파인튜닝 방법을 제안합니다. OpenVLA 모델을 기반으로, 액션 디코딩, 액션 표현, 학습 목표 등 여러 디자인 선택 사항을 분석하여 최적화 파인튜닝(Optimized Fine-Tuning, OFT) 레시피를 개발했습니다. 이 방법은 기존 VLA 모델의 추론 효율성 및 성능을 크게 향상시키며, 새로운 로봇 환경에서도 뛰어난 성능과 유연성을 자랑합니다.

- **Technical Details**: OFT 레시피는 병렬 디코딩(parallel decoding), 액션 청크(action chunking), 연속 액션 표현(continuous action representation), 그리고 L1 회귀 기반 학습 목표(L1 regression-based learning objective)를 통합합니다. 이를 통해, 모델의 추론 효율성을 증가시키고, 정책 성능을 개선하며, 입력-출력 사양의 유연성을 높였습니다. 연구 결과 제안된 OpenVLA-OFT 모델은 LIBERO 시뮬레이션 벤치마크에서 새로운 최첨단 성과를 기록하며, 평균 성공률을 76.5%에서 97.1%로 증가시켰습니다.

- **Performance Highlights**: OpenVLA-OFT는 실제 세계의 로봇 평가에서도 뛰어난 성과를 보였습니다. ALOHA 로봇에서 고빈도 제어 작업을 성공적으로 수행하며, 이전 파인튜닝된 VLA 모델들보다 평균 성공률에서 최대 15% 성능 향상을 기록했습니다. OpenVLA-OFT는 25 단계 액션 청크를 통해 기본 OpenVLA보다 43배 더 빠른 처리량을 달성하여 실시간 로봇 제어를 가능하게 합니다.



### Sensor-Invariant Tactile Representation (https://arxiv.org/abs/2502.19638)
Comments:
          Accepted to ICLR'25

- **What's New**: 본 논문에서는 고해상도 촉각 센서 간의 전이 가능성을 높이기 위한 Sensor-Invariant Tactile Representations (SITR)라는 혁신적인 방법을 소개합니다. Optical tactile sensors에서 새로운 센서로의 제로 샷 전이(Zero-shot transfer)를 가능하게 하여, 다양한 센서 간에 모델이나 지식을 효과적으로 전이할 수 있도록 합니다. 이는 기존의 센서에 대한 데이터가 새로운 센서에서 잘 작동하지 않는 문제를 해결하려는 노력입니다.

- **Technical Details**: SITR 방법론은 다양한 센서 설계를 시뮬레이션한 데이터셋을 기반으로 Transformer 아키텍처를 활용하여 설계되었습니다. 이 방법에서는 소량의 보정 이미지를 사용하여 각 센서를 특성화하고, 지도 대비 학습(Supervised Contrastive Learning, SCL)을 통해 촉각 데이터의 기하학적 특성을 강조합니다. 또한, 물리 기반 시뮬레이터를 사용하여 100개 센서 구성에서 1백만 개의 예제로 구성된 대규모 합성 데이터셋을 생성하였습니다.

- **Performance Highlights**: 실험 결과, SITR 방법은 여러 실제 GelSight 센서에서 다양한 하위 작업에서의 일반화 성능을 입증하였습니다. 기존 방법들에 비해 한 센서에서 훈련된 모델이 다른 센서로 원활하게 전이되는 것을 보여주며, 촉각 센싱 분야에서의 데이터 및 모델 전이 가능성을 획기적으로 개선합니다. 이는 머신 러닝 모델의 전이 가능성을 높이고, 다양한 센서 간의 데이터 공유를 용이하게 만드는 기반을 마련합니다.



### A Method for Evaluating the Interpretability of Machine Learning Models in Predicting Bond Default Risk Based on LIME and SHAP (https://arxiv.org/abs/2502.19615)
Comments:
          12 Pages,9 figures

- **What's New**: 이번 논문은 인공지능 모델의 해석 가능성(interpretability) 분석 방법의 필요성을 강조합니다. LIME과 SHAP와 같은 기존의 분석 도구들이 모델 출력 분석에 많이 사용되지만, 모델의 기본적인 해석 가능성을 평가할 방법은 부족합니다. 저자들은 채권 시장의 디폴트 예측을 사례로 들어 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 연구에서는 LIME과 SHAP를 활용하여 샘플 특성(feature)이 예측 결과에 미치는 영향을 평가하고, 이를 바탕으로 모델 자체의 해석 가능성에 대한 새로운 평가 방법을 제안합니다. 또한, 머신러닝 알고리즘을 적용하여 디폴트 예측의 분류 성능(classification performance)을 평가합니다.

- **Performance Highlights**: 논문의 분석 결과는 해당 모델의 해석 가능성에 대한 직관적 이해와 논리적 기대와 일치하는 것으로 나타났습니다. 이 연구는 복잡한 AI 모델의 해석 가능성을 정량적으로 평가하는 표준화된 방법의 필요성을 제기합니다.



### Self-rewarding correction for mathematical reasoning (https://arxiv.org/abs/2502.19613)
- **What's New**: 본 논문은 자체 보상(self-rewarding) 추론을 수행하는 대형 언어 모델(LLM)을 연구합니다. 이 모델은 단계별 추론을 생성하고 출력의 정확성을 평가할 수 있으며, 이 모든 과정을 외부 피드백 없이 진행합니다. 이러한 통합 접근 방식은 단일 모델이 자율적으로 추론 과정을 안내할 수 있게 하여, 모델 배포에 있어 계산적 이점을 제공합니다.

- **Technical Details**: 저자들은 자체 보상 추론 프레임워크를 제안하며, 이를 통해 LLM이 생성기와 보상 모델을 단일 모델로 통합하여 자율적인 추론과 평가를 수행할 수 있도록 합니다. 두 단계로 구성된 알고리즘적 프레임워크는 자체 생성된 데이터를 통해 모델의 성능을 개선하며, 첫 번째 단계에서는 연속 거부 샘플링(sequential rejection sampling)을 사용하여 사고의 긴 흐름(long chain-of-thought) 궤도를 구축합니다. 두 번째 단계에서는 강화 학습(reinforcement learning)을 통해 정확성을 평가하고 출력을 수정합니다.

- **Performance Highlights**: 실험 결과, Llama-3와 Qwen-2.5 모델은 제안된 접근 방식이 본래의 내재적(self-correction) 자기 수정 기능을 초월하는 성능을 보였으며, 외부 보상 모델에 의존하는 시스템과 유사한 성능을 달성했습니다. 특히, 이 연구는 기존 LLMs에서 나타나는 내재적 자기 수정의 한계를 극복할 수 있는 가능성을 보여줍니다. 전체적인 효과와 동작을 이해하기 위해 다양한 실험과 분석이 수행되었습니다.



### Evaluation of Hate Speech Detection Using Large Language Models and Geographical Contextualization (https://arxiv.org/abs/2502.19612)
Comments:
          6 pages, 2 figures

- **What's New**: 소셜 미디어에서의 혐오 발언의 확산은 사회에 큰 영향을 미치는 심각한 문제로 부각되고 있다. 이 연구는 다국어 데이터셋과 다양한 지리적 맥락에서 혐오 발언 탐지에 대한 LLM(대형 언어 모델)의 성능을 체계적으로 조사하였다. 본 연구에서는 혐오 발언의 이진 분류, 지리적 맥락 인식 탐지 및 적대적 생성 텍스트에 대한 강건성을 포함한 새로운 평가 프레임워크를 제시한다.

- **Technical Details**: 우리의 접근법은 LLM의 성능을 향상시키기 위한 프롬프트 엔지니어링을 사용하였다. 혐오 발언 탐지를 위해 구조화된 프롬프트를 설계하여 모델이 뉘앙스가 있는 텍스트를 이해할 수 있도록 하였다. 평가한 LLM으로는 Llama2, Codellama, DeepSeekCoder가 있으며, 각각의 모델은 1,000개의 다양한 지역에서 수집된 댓글로 평가되었다.

- **Performance Highlights**: Codellama는 혐오 발언 탐지에서 70.6%의 리콜과 52.18%의 F1 점수를 기록했으나, 지리적 민감성 테스트에서는 DeepSeekCoder가 더 나은 성능을 보였다. Llama2는 62.5%의 적대적 샘플을 잘못 분류하여 현재 LLM의 강건성의 한계를 나타내는 결과를 보였다. 이러한 발견은 정확성, 맥락적 이해 및 강건성 사이의 트레이드오프를 부각시킨다.



### A City of Millions: Mapping Literary Social Networks At Sca (https://arxiv.org/abs/2502.19590)
- **What's New**: 이번 연구에서는 다국어 픽션과 비픽션 내러티브에서 추출한 70,509개의 고품질 사회 네트워크를 공개합니다. 이 데이터셋에는 1800년부터 1999년까지 작성된 약 30,000개의 텍스트에 대한 메타데이터도 포함되어 있으며, 이는 인류학 및 사회과학에서 역사적 사회 세계에 대한 새로운 자원을 제공합니다. 자동화된 사회 네트워크 추출 방법을 도입하여 일관성을 유지하면서도 대규모로 데이터를 수집하는 데 성공했습니다.

- **Technical Details**: 연구팀은 Project Gutenberg (PG) 코퍼스에서 텍스트를 가져와 JSON Schema를 기반으로 한 출력 방식으로 사회 네트워크를 생성했습니다. 특히, 구글의 Gemini 1.5 Flash 모델을 사용해 최대 1백만 토큰까지 처리할 수 있는 컨텍스트 길이를 확보했습니다. 이 모델은 구조화된 출력을 지원하여 각 텍스트에 대해 등장인물과 그 관계를 포함한 JSON 배열을 반환합니다.

- **Performance Highlights**: 결과적으로 총 72,875 권의 문헌 중 71,836개의 네트워크를 성공적으로 추출했습니다. 이 데이터셋은 문학 및 사회 가설을 평가할 수 있는 기회를 제공하며, 비픽션 네트워크가 픽션 네트워크보다 다양한 커뮤니티로 구성되어 있고, 클러스터링이 적다는 초기 분석 결과를 도출했습니다.



### Tell me why: Visual foundation models as self-explainable classifiers (https://arxiv.org/abs/2502.19577)
- **What's New**: 본 연구에서는 시각 기초 모델(Visual Foundation Models, VFM)과 새로운 프로토타입 아키텍처(prototypical architecture)를 결합하여 해석 가능한 분류기를 제안합니다. 이 모델은 예측을 해석 가능한 개념들의 가중합으로 분해하여 자가 설명(self-explainable)을 목표로 합니다. 이와 같은 접근법이 기존 모델보다 더 효율적이고 해석 가능하다는 점이 특히 주목할 만합니다.

- **Technical Details**: ProtoFM이라는 방법론은 고정된 VFM 위에 가벼운 헤드(약 1M 파라미터)를 훈련시키는 방식을 채택합니다. 전문화된 훈련 목표(specialized training objectives)를 통해 해석 가능성을 증대시키고, 예측의 신뢰성을 확보하는 데 중점을 두고 있습니다. 이 모델은 기존의 VFM과 비교해 훨씬 적은 파라미터를 사용하면서도 효과성을 유지합니다.

- **Performance Highlights**: 평가 결과에 따르면, ProtoFM은 경쟁력 있는 분류 성능(classification performance)을 달성하며 해석 가능성 메트릭(interpretabiliy metrics)에서도 기존 모델들을 초월했습니다. 연구에 사용된 해석 가능성 관련 지표는 문헌에서 파생된 것입니다. 코드도 제공되어 있어 연구자들이 쉽게 활용할 수 있습니다.



### Do Large Language Models Know How Much They Know? (https://arxiv.org/abs/2502.19573)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 특정 주제에 대한 자신의 지식의 범위를 인식하는 능력을 평가하는 새로운 벤치마크를 개발하였다. 연구 결과, 모든 테스트한 LLM은 충분한 규모로 주어졌을 때, 특정 주제에 대한 자신의 지식을 얼마나 알고 있는지를 이해할 수 있는 것으로 나타났다. 이는 LLM의 지식 인식이 일반적인 속성일 가능성을 제시하며, 더 깊은 메커니즘 이해를 위한 추가 연구가 필요하다는 점도 강조되었다.

- **Technical Details**: 연구에서는 LLM이 가상의 개별 다이어리 항목으로 미세 조정되며, 특정 개인의 모든 다이어리 항목을 연대순으로 회상하는 능력을 평가한다. 모델이 올바른 양의 정보를 기억할 수 있다면, 이는 모델이 해당 주제에 대한 자신의 지식 범위를 이해하고 있다는 것을 시사한다. 또한 모델의 기억 능력이 단기 및 장기 문서 모두에서 효과적임을 보여주어, 특정 아키텍처의 특성이 성능에 미치는 영향을 탐구하였다.

- **Performance Highlights**: 시험된 모든 LLM은 적절한 데이터 크기에 따라 주제에 대한 지식을 적절히 기억하는 능력을 보여주었다. 그러나 데이터 스케일이 부족할 경우, 일부 모델은 무작위로 다이어리 항목을 회상하며 과소 또는 과대 기억하는 경향을 보였다. 연구는 이 특성의 출현 차이에 기여하는 잠재적 요인들에 대해서도 논의하여 LLM의 내재적 메커니즘에 대한 이해를 넓힌다.



### Diffusion-based Planning with Learned Viability Filters (https://arxiv.org/abs/2502.19564)
- **What's New**: 이 논문에서는 diffusion 모델을 사용하여 모션 계획(motion planning)을 개선하는 새로운 접근 방식을 제안합니다. 여기서 제안하는 learned viability filter (𝑉𝐹) 는 향후 성공을 예측하여 샘플의 불가능한 계획을 필터링할 수 있습니다. 𝑉𝐹를 통해 여러 종류의 계획을 조합할 수 있으며, 이는 3D 인간 보행 작업에서 효과적으로 적용됩니다. 또한, 𝑉𝐹를 활용하는 것이 기존의 guidance-based diffusion 예측보다 훨씬 빠른 속도를 제공한다고 보여줍니다.

- **Technical Details**: 이 연구는 learned viability filter (𝑉𝐹)를 사용하여 계획의 미래 가능성을 신속하고 효율적으로 평가합니다. 𝑉𝐹는 오프라인 가치 반복(value iteration)을 통해 학습된 Q-function을 통해 실제의 viability kernel을 근사하며, 이는 stochasitc 환경에서의 예측을 가능하게 합니다. 𝑉𝐹는 diffusion 모델의 결과와 독립적으로 훈련될 수 있어, 계획 생성 과정에서 다양한 정보를 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, 𝑉𝐹 기반의 diffusion 모델은 box-climbing, step-over walls, 그리고 장애물 회피와 같은 복잡한 보행 작업을 처리하는 데 효과적임을 증명하였습니다. 이 방식은 특히 온라인 계획 및 제어에서 중요한 성능 향상을 보였으며, 속도 면에서도 기존의 방법들보다 우수한 성능을 보였습니다. 𝑉𝐹를 이용하면 다양한 가능한 계획을 효율적으로 생성할 수 있는 장점이 있습니다.



### Advancing calibration for stochastic agent-based models in epidemiology with Stein variational inference and Gaussian process surrogates (https://arxiv.org/abs/2502.19550)
- **What's New**: 이 논문에서는 역학에서 사용되는 확률론적 에이전트 기반 모델(ABM)의 정확한 보정을 위한 스틴 변별 추론(Stein Variational Inference, SVI)의 유용성을 조사합니다. 기존의 마르코프 체인 몬테카를로(Markov Chain Monte Carlo, MCMC) 방법은 수치적으로 비용이 많이 드는 반면, SVI는 높은 차원의 매개변수 공간에서 효율적으로 작동하며, 이로 인해 도시의 COVID-19 확산 모델인 CityCOVID에 적용할 수 있는 혁신적인 보정 방법을 제시합니다.

- **Technical Details**: CityCOVID ABM은 약 260만 명의 인구를 모델링하며, 시간에 따라 활동 장소 사이를 이동하는 사람들의 역학적 전파를 시뮬레이션합니다. 논문에서는 SVI가 ABM의 매개변수를 보정하는 과정에서 겪는 현실적인 도전과제를 다루며, 하이퍼파라미터 조정과 입자 동역학 모니터링이 포함됩니다. 또한, SVI는 Gaussian process (GP) 대리 모델과 함께 사용되어 ABM 보정의 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과 SVI는 MCMC에 비해 예측 정확도와 보정 효율성을 유지하며, 더욱 효율적인 보정 과정을 제공합니다. 논문에서는 두 방법의 성능을 비교 분석하였으며, SVI가 복잡한 역학적 모델에서 유망한 대안임을 보여줍니다. 결국, SVI는 계산 비용을 줄일 수 있으며, AGMs 보정에 효과적으로 적용될 수 있다는 점에서 연구에 중요한 기여를 합니다.



### Winning Big with Small Models: Knowledge Distillation vs. Self-Training for Reducing Hallucination in QA Agents (https://arxiv.org/abs/2502.19545)
- **What's New**: 이 연구에서는 고객 지원에 대한 대규모 언어 모델(LLMs)의 배치에서 발생하는 hallucination(허위 정보의 생성) 문제와 비경제적인 독점 모델의 비용 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, Samsung Smart TV 사용자 매뉴얼에 대한 질의 응답 데이터 셋을 사용하여 합성 데이터(synthetic data)가 군중 소싱 데이터(crowdsourced data)보다 더 낮은 hallucination 비율을 보여줄 수 있음을 입증했습니다. 또한, self-training(자기 훈련)과 지식 증류(knowledge distillation)를 비교하면서, 두 방법이 유사한 수준의 hallucination 감소를 보임을 발견했습니다.

- **Technical Details**: 연구에서는 retrieval-augmented question-answering (QA) 파이프라인을 개발하고, Llama-3-8B-Instruct 모델을 사용하여 crowdsourced 질문에 대한 답변을 생성합니다. 이와 함께, 수작업(cleaning) 및 자동화된 방법을 통해 데이터 정제(data cleaning) 성능을 비교했습니다. 결과적으로 LLM이 생성한 합성 데이터는 더 낮은 hallucination 비율을 기록했으며, self-training 방식이 knowledge distillation 보다 유사한 효과를 나타내는 흥미로운 발견을 하였습니다. 또한, 무응답 질문에 대한 "모르겠습니다"라는 맥락화된 응답(contextualized responses)을 통해 모델의 견고성(robustness)을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 수작업 및 자동 데이터 정제 방법은 유사한 사실적 정확도를 보였지만, 자동 정제를 통한 모델의 응답이 더 길었습니다. LLM이 생성한 합성 교육 데이터는 군중 소싱 데이터보다 낮은 hallucination 비율을 기록하였고, self-training을 통해 Llama-3 모델이 생성한 데이터에 대한 모델 성능이 GPT-4o의 데이터에 대한 훈련과 유사함을 입증했습니다. 이러한 발견은 self-training이 hallucination을 최소화하는 데 있어 리소스를 효율적으로 사용할 수 있는 대안임을 보여줍니다.



### No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data (https://arxiv.org/abs/2502.19537)
- **What's New**: 이 논문에서는 OpenAI와 Google 같은 언어 모델 (LM) 제공자들이 사용자 맞춤형 데이터로 LM을 훈련할 수 있도록 하는 fine-tuning API를 소개합니다. 이러한 API는 새로운 맞춤화 기회를 제공하지만, 모델의 안전성을 해칠 수 있는 취약점도 노출합니다. 논문에서는 harmfulness filter를 사용하여 유해한 훈련 데이터를 차단하는 방법과 이를 우회하기 위한 다양한 공격 방법을 설명합니다.

- **Technical Details**: 연구에서는 LM의 초기 응답 토큰에서 모델의 거부를 제거하여 공격이 이루어진다는 점을 강조합니다. 피해자의 훈련 데이터가 harmless하더라도, 공격자는 이를 이용해 위험한 LM을 만들 수 있습니다. 새로운 공격 기법 NOICE은 모델이 안전성을 바탕으로 악의적인 요청에 거부 반응을 하도록 훈련한 후, 요청을 이행하게 하는 메커니즘을 사용합니다.

- **Performance Highlights**: NOICE 공격은 GPT-4o 모델에 대해 57%의 공격 성공률 (ASR)을 기록하였고, OpenAI의 Bug Bounty를 수상했습니다. 또한, 간단한 방어 기법을 통해 기존 공격의 성공률을 37-79%에서 최대한 낮출 수 있음을 보여줍니다. 이 연구는 harmless 데이터를 사용한 공격이 어떻게 모델을 해칠 수 있는지를 잘 보여줍니다.



### GONet: A Generalizable Deep Learning Model for Glaucoma Detection (https://arxiv.org/abs/2502.19514)
Comments:
          9 pages, 4 figures, submitted to IEEE Transactions on Biomedical Engineering

- **What's New**: 이번 연구에서는 다양한 인종과 질병 집단에서의 일반화 한계를 극복하기 위해 GONet이라는 강력한 딥러닝 모델을 소개합니다. GONet은 119,000개 이상의 디지털 안저 이미지(DFI)와 금표준 주석을 가진 7개의 독립 데이터세트를 사용하여 개발되었습니다. 이 모델은 DINOv2로 사전 훈련된 셀프-슈퍼바이즈드 비전 트랜스포머를 기반으로 하여 멀티소스 도메인 전략을 통해 겨냥할 수 있는 여러 시험에서 높은 일반화 성능을 보여주었습니다.

- **Technical Details**: GONet은 다양한 지리적 배경을 가진 환자들의 금표준 주석을 포함하는 데이터 세트를 통해 훈련되었습니다. 이 모델은 높은 OOD(out-of-distribution) 일반화 성능을 입증하였으며, 특정 데이터셋에서는 AUC(Area Under Curve) 값이 0.85에서 0.99에 이릅니다. GONet은 최첨단 딥러닝 모델들과의 성능 비교에서도 유사하거나 더 우수한 결과를 보였으며, 컵-디스크 비율(CDR)과 비교하여 최대 21.6% 이상의 성능 개선을 보여주었습니다.

- **Performance Highlights**: GONet의 성능은 컵-디스크 비율(CDR)과 논문에서 기존의 최첨단 모델들과 비교했을 때 유사하거나 더욱 나은 결과를 나타냈습니다. 연구 결과는 GON의 조기 진단과 치료의 중요성을 강조하며, GON 모델의 일반화 성능을 크게 향상시키기 위한 새로운 접근 방법으로 주목받고 있습니다. 최신 데이터셋인 HYRD를 공개함으로써 연구자들이 GON 진단 모델을 개발하는 데 기여하고자 하였습니다.



### Conversational Planning for Personal Plans (https://arxiv.org/abs/2502.19500)
- **What's New**: 이 연구에서는 대화 시스템이 장기 상호작용과 과제를 지원하기 위해 언어 기반 에이전트를 필요로 한다고 강조합니다. 대화식 계획을 통해 사용자의 목표에 맞는 미시 행동을 결정하는 메타 컨트롤러 역할을 수행하는 LLM(대규모 언어 모델)의 새로운 아키텍처를 제안하였습니다. 이를 통해 사용자의 피드백을 바탕으로 계획을 조정하며, 실제 목표를 달성하는 데 도움을 줄 수 있는 가능성을 열어줍니다.

- **Technical Details**: 제안된 접근 방식은 코어 비공식 계획(Chain-of-Thought, CoT) 프롬프트를 활용한 LLM이 어떻게 상위 행동(macros) 결정을 내리고, 여러 세부 단계에 따라 대화하며 상호 작용을 수행하는지를 설명합니다. 이 시스템은 사용자의 언어 피드백을 수집하고, 이를 기반으로 다음 행동을 결정하는 구조를 갖추고 있습니다. 이는 Hierarchical RL (강화 학습) 프레임워크에 기반하여 작업을 처리합니다.

- **Performance Highlights**: 이 접근 방식은 건강 관리 및 학습 등 다양한 분야에서 효과적으로 기능하는 것을 입증하였습니다. 사용자 피드백에 따라 계획을 지속적으로 수정하며, 개인 맞춤형 계획 수립에 도움을 줄 수 있는 가능성을 보여줍니다. 또한, 이 연구는 기존 대화형 추천 시스템의 상태를 향상시키는 작업에 기여합니다.



### Practical Evaluation of Copula-based Survival Metrics: Beyond the Independent Censoring Assumption (https://arxiv.org/abs/2502.19460)
- **What's New**: 전통적인 생존 메트릭스는 독립적인 censoring 가정을 요구하지만, 사건과 관련된 이유로 censoring되는 경우에는 이 가정이 더 이상 유지되지 않습니다. 본 논문에서는 의존성 있는 censoring의 존재하에 생존 모델을 평가할 수 있는 세 가지 copula 기반 메트릭스를 제안하고, 이러한 메트릭스를 평가하기 위해 현실적이고 반합성적인 데이터를 생성하는 프레임워크를 설계합니다. 이러한 접근법은 생존 예측의 정확성을 높이는데 기여할 수 있습니다.

- **Technical Details**: 생존 예측 모델은 라벨이 있는 데이터셋으로부터 인스턴스의 설명을 실제 값으로 매핑하는 모델을 학습하는 것을 목표로 합니다. 하지만 일부 훈련 인스턴스가 'censored' 상태이므로, 생존 예측 모델은 이벤트와 censoring 분포 간의 의존성을 모델링하는 copula 함수를 사용해야 합니다. 이 방식은 특히 잃어버린 후속조사의 개념을 포함하여, 예측 성능에 대한 기존 생존 메트릭스의 편향을 극복하는 데 효과적입니다.

- **Performance Highlights**: 이 논문의 실험적 분석은 생성된 데이터셋과 반합성 데이터셋 내에서 우리의 메트릭스가 정확한 오류 추정치를 제공하며, 특히 예측 정확도 측면에서 개선된 결과를 보여줍니다. 기존 메트릭스보다 더 정확하게 생존 확률을 추정할 수 있는 가능성을 제시하며, 의존성 있는 censoring이 존재하는 상황에서의 생존 모델 평가에 새로운 기준을 마련합니다.



### Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting (https://arxiv.org/abs/2502.19459)
- **What's New**: 이 논문에서는 복잡한 다중 파트( multi-part) 조작물(articulated objects)의 재구성을 위한 새로운 방법인 ArtGS를 소개합니다. 3D Gaussians를 활용한 유연하고 효율적인 표현을 통해 기존 방법들이 갖고 있는 한계를 극복하고자 하며, 조작 부분의 정보 정렬 및 파트 다이나믹스 모델링을 개선합니다. 특히, 새로운 벤치마크에서 ArtGS는 파라미터 추정 및 파트 메쉬 재구성에서 최신 성능을 달성했다고 보고합니다.

- **Technical Details**: ArtGS는 두 가지 주요 혁신을 통해 복잡한 파트 조작물의 모델링을 가능하게 합니다. 첫째, Canonical Gaussians를 사용하여 조작물의 여러 상태 간의 정보를 효과적으로 통합하고, Coarse-to-fine 초기화 기법을 통해 재구성의 정확성을 향상시킵니다. 둘째, Gaussian skinning에서 영감을 얻은 다이나믹스 모델링 모듈로 파트 메쉬 재구성을 개선하여 조작 학습의 질을 높입니다.

- **Performance Highlights**: 폭넓은 실험을 통해 ArtGS는 합성 데이터 및 실제 데이터 모두에서 우수한 성능을 보이며, 특히 복잡한 다중 파트 조작물의 효율적인 재구성을 이루었습니다. 본 방법은 기존 모형들보다 재구성 품질과 효율성이 크게 향상되었으며, 새로운 벤치마크에서 광범위한 실험을 통해 각 구성 요소의 효율성을 확인하고 향후 개선 가능성을 제시합니다.



### Evolutionary Algorithms Approach For Search Based On Semantic Document Similarity (https://arxiv.org/abs/2502.19437)
- **What's New**: 이 논문에서는 클라우드 컴퓨팅과 분산 컴퓨팅의 발전이 신경망(Neural Networks)과 유전자 알고리즘(Genetic Algorithm), 차별 진화 알고리즘(Differential Evolution Algorithm) 같은 진화 알고리즘의 연구에 기여했음을 설명합니다. 특히, Universal Sentence Encoder (USE)를 통해 텍스트의 의미적 유사성을 포착하고, 이를 바탕으로 사용자 쿼리와 관련된 상위 N개의 문서를 검색하는 접근 방식을 제안하고 있습니다. 기존의 전통적인 접근 방식과 비교하여, 진화 알고리즘이 상위 N개 결과를 찾는 데 더욱 우수하다는 실험 결과를 통해 검증했습니다.

- **Technical Details**: 진화 알고리즘은 최적화 기술을 적용하는 컴퓨터 프로그램으로, 생물학적 진화 과정과 유사한 방식으로 작동합니다. 이 연구에서는 Universal Sentence Encoder (USE)가 생성한 문장 임베딩을 이용하여 주어진 사용자 질문에 대한 답변을 검색합니다. 나아가, Manhattan Distance, Genetic Algorithm(GA), Differential Evolution(DE) 알고리즘의 성능을 비교하여, 이러한 진화적 접근 방식이 높은 품질의 결과를 제공함을 강조합니다.

- **Performance Highlights**: 이 연구의 실험 결과는 Universal Sentence Encoder (USE)를 사용하여 문서의 의미적 유사성을 효과적으로 캡처함으로써 문서의 텍스트 표현을 문장 임베딩 벡터로 효율적으로 변환할 수 있음을 보여줍니다. 또한, 비교 실험을 통해 GA와 DE 알고리즘이 전통적인 순위 접근 방식보다 우수한 성능을 보임을 입증하여, 최상위 N개의 결과를 찾는 데 있어 진화 알고리즘의 우수성을 확인했습니다.



### scMamba: A Pre-Trained Model for Single-Nucleus RNA Sequencing Analysis in Neurodegenerative Disorders (https://arxiv.org/abs/2502.19429)
Comments:
          41 pages, 12 figures

- **What's New**: 이번 연구에서는 신경퇴행성 질환에 중점을 두고 single-nucleus RNA sequencing (snRNA-seq) 분석의 품질을 개선하기 위한 새로운 사전 훈련 모델인 scMamba를 제안합니다. scMamba는 Mamba 모델의 영감을 받아 설계된 혁신적인 구조를 가지고 있으며, 노이즈를 최소화하면서 입력 정보를 보존하는 기능을 가지고 있습니다. 이 모델은 차원 축소 없이 snRNA-seq 데이터를 효율적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: scMamba의 독창적인 아키텍처는 선형 어댑터 레이어(linear adapter layer)와 유전자 임베딩(gene embeddings), 양방향 Mamba 블록(bidirectional Mamba blocks)을 포함하고 있습니다. 이 모델은 면밀히 훈련된 snRNA-seq 데이터를 기반으로 세포와 유전자의 일반화된 특징을 학습하며, 기존의 높은 분산 유전자를 선택하거나 차원 축소를 필요로 하지 않습니다. 이 연구는 다양한 하위 작업(cell type annotation, doublet detection 등)에서 scMamba의 성능이 기존의 방법들에 비해 우수함을 보여줍니다.

- **Performance Highlights**: scMamba는 다양한 뇌 조직의 다섯 개 데이터셋에서 비교 분석을 통해 다른 방법들보다 지속적으로 우수한 성능을 발휘했습니다. 특히, UMAP을 사용하여 시각화한 결과, scMamba가 세포 임베딩(cell embeddings)에서 구별된 클러스터를 형성하는 능력이 있음을 보여주었습니다. 또한, scMamba는 단일 세포 유형 분류(cell type classification) 작업에서 기존의 여러 기준선 방법들과 비교하여 높은 정확도를 기록하였습니다.




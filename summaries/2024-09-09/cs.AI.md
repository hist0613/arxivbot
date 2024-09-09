New uploads on arXiv(cs.CL)

### RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs (https://arxiv.org/abs/2409.04421)
- **What's New**: 이 논문에서는 사용자 모델링과 개인화 시스템을 위한 자연어 사용자 요약을 생성하는 새로운 작업을 소개하며, RLPF라는 방법을 통해 LLM (Large Language Models)의 개인화를 향상시키려고 한다.

- **Technical Details**: RLPF(리인포스먼트 러닝 프롬 프레딕션 피드백)는 사용자 활동 데이터를 기반으로 요약 모델을 훈련하여, 다운스트림(task-specific) 예측 작업을 최적화하는 것이 특징이다. 이 방법은 세 가지 주요 구성 요소로 이루어져 있다: 1) 요약 모델, 2) 예측 기반 보상 모델, 3) 피드백 루프.

- **Performance Highlights**: RLPF는 다운스트림 작업 성능에서 최대 22%의 개선을 보였고, Factuality(사실성), Abstractiveness(추상성), Readability(가독성) 평가에서 84.59%의 승률을 기록하였다. 19개의 unseen 작업 및 데이터 세트에서 16개에서 성능이 향상되었으며, 사용자 컨텍스트 길이를 74% 감소시켰다.



### Learning vs Retrieval: The Role of In-Context Examples in Regression with LLMs (https://arxiv.org/abs/2409.04318)
- **What's New**: 이 논문은 Generative Large Language Models (LLMs)이 in-context learning (ICL)을 수행하는 메커니즘을 평가하기 위한 새로운 프레임워크를 제시합니다. 연구자들은 LLMs가 내부 지식을 검색하고 in-context 예제에서 학습하는 두 가지 메커니즘의 조합으로 ICL을 이해하고자 합니다.

- **Technical Details**: LLMs는 실제 데이터셋에서 회귀(regression)를 수행할 수 있는 능력을 보여주며, 실험을 통해 모델이 내부 지식을 얼마나 검색하는지와 in-context 예제에서 학습하는지의 정도를 측정합니다. 저자들은 프롬프트 엔지니어링을 통해 ICL 예제로부터 메타 학습을 활용하고 지식을 검색하는 방법을 제안합니다. 세 가지 LLMs와 여러 데이터셋을 사용하여 연구 결과의 견고성을 확인합니다.

- **Performance Highlights**: 연구 결과, LLMs는 실제 데이터셋에서 회귀 예제로부터 효과적으로 학습할 수 있으며, ICL 메커니즘을 통해 내부 지식 검색과 in-context 예제에서의 학습을 조절하는 방법론을 제시합니다. 연구자들은 이 기법이 LLM의 성능을 개선하는 데 도움이 될 것이라고 주장합니다.



### Open Language Data Initiative: Advancing Low-Resource Machine Translation for Karakalpak (https://arxiv.org/abs/2409.04269)
Comments:
          Submitted to WMT 2024

- **What's New**: 본 연구는 Karakalpak어에 대한 몇 가지 기여를 제시합니다: Karakalpak어로 번역된 FLORES+ devtest 데이터셋, 각 100,000 쌍의 Uzbek-Karakalpak, Russian-Karakalpak, English-Karakalpak의 병렬 코퍼스, 그리고 이러한 언어 간 번역을 위한 공개 소스의 미세 조정된 신경 모델입니다. 실험을 통해 다양한 모델 변형 및 훈련 접근 방식을 비교하고 기존 기준을 초과하는 성과를 입증했습니다.

- **Technical Details**: Karakalpak어는 Turkic 언어군의 구성원으로, 주로 우즈베키스탄의 자치 지역인 Karakalpakstan에서 사용됩니다. 본 연구에서는 Open Language Data Initiative (OLDI)의 공유 작업의 일환으로, NLLB 모델의 미세 조정 버전을 사용한 Karakalpak 번역을 위한 모델을 제시합니다. 우리는 FLORES+ devtest 데이터셋, 300,000 쌍의 병렬 코퍼스, 그리고 Karakalpak어에 대한 특별히 훈련된 모델을 공개했습니다.

- **Performance Highlights**: nllb-200-distilled-600M 모델을 사용하여 실험을 진행하였고, 다양한 모델 변형을 개발하여 추가 토큰과 훈련 데이터 구성의 영향을 평가했습니다. Karakalpak어 번역을 향상시키기 위해 수행된 실험은 특히 저자원 언어의 자연어 처리(NLP) 능력을 향상시키기 위한 다양한 접근 방식을 검증합니다.



### GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding (https://arxiv.org/abs/2409.04183)
- **What's New**: GALLa (Graph Aligned Large Language Model)는 그래프 신경망(GNN)과 크로스 모달 정렬 기술을 활용하여 LLM에 코드의 구조적 정보를 주입합니다. 기존 LLM의 아키텍처를 수정하는 대신, Training 시간에만 그래프 데이터를 사용하고 Inference 시에는 비용이 발생하지 않는 모델-비가변성(task-agnostic) 구조를 제안합니다.

- **Technical Details**: GALLa는 AST와 DFG를 처리하고, 이를 LLM의 임베딩 공간으로 프로젝션하기 위해 경량 어댑터를 사용합니다. LLM은 그래프 정보를 바탕으로 소스 코드를 생성하고 그래프 구조에 대한 질문에 답하는 방식으로 훈련됩니다. 이 접근 방식은 Transform 학습 프레임워크를 따르며, 그래프 정렬 데이터와 작업별 훈련 데이터를 분리하여 LLM의 일반적인 능력을 보존합니다.

- **Performance Highlights**: GALLa는 350M에서 8B 크기의 네 가지 LLM에서 수행된 다섯 가지 코드 작업에 대한 실험을 통해 일관된 성능 향상을 보여주었습니다. 특히 GALLa는 LLaMA3와 같이 강력한 모델에서도 baseline보다 더 나은 성능을 발휘하며, 그래프 정렬 과정을 통해 얻은 구조적 지식을 새로운 프로그래밍 언어에까지 일반화하는 능력을 보여줍니다.



### Combining LLMs and Knowledge Graphs to Reduce Hallucinations in Question Answering (https://arxiv.org/abs/2409.04181)
- **What's New**: 이번 논문에서는 생물의학(Biomedical) 분야에서 정확성과 신뢰성을 높이기 위한 새로운 접근 방식을 제안합니다. 대규모 언어 모델(Large Language Models, LLM)과 지식 그래프(Knowledge Graphs, KG)를 결합하여 질문-응답 시스템의 정확성을 개선하는 방식입니다.

- **Technical Details**: 본 방법은 LangChain 프레임워크를 기반으로 하며, LLM이 생성한 쿼리의 구문(Syntax) 및 의미론적(Semantics) 유효성을 보장하는 쿼리 체크(Query Checker)를 포함합니다. 이 쿼리는 지식 그래프에서 정보를 추출하는 데 사용되어 환각(Hallucination)의 오류를 크게 줄입니다.

- **Performance Highlights**: 50개 생물의학 질문을 활용한 새로운 벤치마크 데이터셋을 통해 GPT-4 Turbo가 정확한 쿼리 생성에서 다른 모델을 능가하는 것으로 나타났습니다. 또한, llama3:70b와 같은 오픈소스 모델은 적절한 프롬프트 엔지니어링(Prompt Engineering)을 통해 가능성을 보였습니다.



### From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks (https://arxiv.org/abs/2409.04168)
- **What's New**: 이 논문에서는 기존의 LLM(judges)을 사용한 텍스트 평가 방식과는 달리, 수학적 추론 작업을 통해 LLM의 효용성을 평가합니다. 이는 다단계 추론이 필요한 작업으로, 솔루션의 정확성을 검증할 수 있어 보다 객관적인 평가를 가능하게 합니다.

- **Technical Details**: 본 연구는 네 개의 대형 LLM(30B 이상의 매개변수) 및 네 개의 소형 LLM(10B 미만의 매개변수)과 세 개의 수학적 추론 데이터셋을 이용하여 성능 분석을 수행합니다. 평가 결과, 크기가 큰 LLM이 일반적으로 더 나은 판별자가 되는 것으로 나타났지만, 대부분의 모델은 작업 성능을 개선하지 못했습니다.

- **Performance Highlights**: 연구 결과, LLM judges는 높은 품질의 모델을 선택하는 경향이 있으나, 그들의 답변이 틀려도 품질이 더 좋은 모델을 선택했습니다. 통계적 기법을 통해 개별 모델의 작업 성능을 기반으로 판단 성능을 예측할 수 있음을 보여주었습니다. 또한 입력값을 교환하거나 마스킹하는 실험을 통해 judges가 작성 스타일을 판단의 중요한 요소로 간주한다는 증거를 발견했습니다.



### Can OpenSource beat ChatGPT? -- A Comparative Study of Large Language Models for Text-to-Code Generation (https://arxiv.org/abs/2409.04164)
Comments:
          Conference Paper accepted at the 9th SwissText Conference (2024)

- **What's New**: 이번 연구에서는 최신 LLM 대화형 모델인 Bard, BingChat, ChatGPT, Llama2, Code Llama를 평가하여 Python 코드 생성 능력을 분석했습니다. LeetCode에서 발췌한 프로그래밍 문제를 대상으로 한 실험적 연구를 통해 자연어 설명을 코드로 변환하는 성능 차이를 조사했습니다.

- **Technical Details**: 연구는 LeetCode에서 제공하는 다양한 프로그래밍 문제를 사용했으며, 모델의 성능은 정밀도(correctness), 실행 시간(runtime), 메모리 사용량(memory usage) 등을 기준으로 평가되었습니다. 이와 함께, 생성된 코드의 오류 분석을 통해 Indentation과 코드 형식에서의 차이를 비교하고, 잘못된 문제 해결을 특정 오류 범주로 분류하여 결과를 더욱 세분화했습니다.

- **Performance Highlights**: ChatGPT는 전반적으로 가장 높은 성능을 보이며, 코드 전용 모델인 Code Llama를 초월하는 성과를 기록했습니다. 특히 긴 프롬프트에 대한 모델의 잘못된 코드 생성 문제도 관찰되었습니다.



### A Coin Has Two Sides: A Novel Detector-Corrector Framework for Chinese Spelling Correction (https://arxiv.org/abs/2409.04150)
Comments:
          ECAI-2024

- **What's New**: 본 논문에서는 중국어 맞춤법 교정(Chinese Spelling Correction, CSC)에 대한 새로운 접근 방식을 제안합니다. 기존의 오류 감지기를 사용한 두 단계 구조와는 달리, 우리의 방법은 높은 정밀도(precision)와 재현율(recall)을 가진 두 세트의 오류 탐지 결과를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: 우리는 오류 감지기의 성능을 개선하고 검출 결과를 활용하는 방법을 최적화하는 두 가지 전략을 설계했습니다. 첫 번째는 오류 위치 정보 융합 전략(Error Position Information Fusion Strategy, EP)으로, 오류 토큰의 임베딩을 조정하여 오류의 존재를 인식합니다. 두 번째는 선택적 마스킹 전략(Selective Masking Strategy, SM)으로, 오류 위치에 해당하는 토큰을 마스킹하여 오류 위치를 명시하고 교정 과정에서 모델을 안내합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법이 주요 벤치마크에서 우수한 성능을 보이며, 기존의 CSC 방법들에 비해 효과적인 오류 교정 성능을 입증했습니다.



### Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering (https://arxiv.org/abs/2409.04122)
Comments:
          preprint, under review, supplementary material will be made available upon acceptance of the paper

- **What's New**: 본 논문에서는 작성자 프로파일링(author profiling) 작업을 개선하기 위한 새로운 방법을 제안합니다. 이 방법은 관련 없는 내용을 우선적으로 필터링한 후, 유효한 데이터만을 활용하여 사용자 프로파일링을 진행하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 강화 학습(reinforcement learning)을 활용하여 관계성 필터를 최적화하며, 큰 언어 모델(LLM)의 제로샷(zero-shot) 능력을 활용하여 보상 함수를 구성합니다. 이 접근법은 대량의 게시물을 효율적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구는 두 개의 트위터 데이터를 사용하여 Big Five 성격 특성 예측을 수행하였으며, 필터링을 통해 관련 게시물만을 사용함으로써 정확도가 유의미하게 향상되었음을 보여주었습니다. 또한, 채택된 제한된 데이터로도 기존의 모든 게시물을 사용할 때와 유사한 효과를 달성할 수 있음을 입증하였습니다.



### Multi-Programming Language Ensemble for Code Generation in Large Language Mod (https://arxiv.org/abs/2409.04114)
Comments:
          Code available at this https URL

- **What's New**: 이번 연구에서는 Multi-Programming Language Ensemble (MPLE)라는 새로운 앙상블 기반의 방법론을 제안합니다. 이 방법은 여러 프로그래밍 언어를 활용하여 코드 생성 성능을 향상시키며, 단일 언어에서의 코드 생성에 국한되지 않습니다.

- **Technical Details**: MPLE는 각 언어별 코드 생성 프로세스를 개별 '약한 전문가 (weak expert)'로 간주하고, 그 출력을 효과적으로 통합하여 언어 특유의 오류와 편향을 완화합니다. 이 방법은 코드 생성을 개선하기 위해 reflection algorithm과 Monte Carlo tree search (MCTS) 같은 일반적인 기술과도 원활하게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, MPLE 프레임워크는 기존 벤치마크 (HumanEval 및 HumanEval-plus)에서 최대 17.92%의 성능 향상을 보여주며, HumanEval 벤치마크에서는 96.25%의 정확도로 새로운 최첨단 결과를 달성했습니다.



### Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers (https://arxiv.org/abs/2409.04109)
Comments:
          main paper is 20 pages

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 과학적 발견의 가속화 가능성에 대한 낙관론을 불러일으켰습니다. 그러나 LLM 시스템이 전문가 수준의 혁신적 아이디어를 생성하고 검증할 수 있는지를 평가한 연구는 없었습니다. 본 연구는 LLM과 전문가 NLP 연구자 간의 연구 아이디어 생성 능력을 직접 비교하여 LLM의 현재 능력에 대한 통계적으로 유의미한 결론을 도출합니다.

- **Technical Details**: 연구는 100명 이상의 NLP 전문가를 모집하여 LLM 아이디어와 인간 아이디어에 대한 블라인드 리뷰를 수행하는 실험 설계를 제시했습니다. LLM 아이디어가 인간 전문가의 아이디어보다 더 혁신적(p < 0.05)으로 평가된 반면, 실행 가능성 측면에서는 다소 약한 평가를 받았습니다. 독창성과 실행 가능성의 차이를 분석하여, LLM 기반 연구 에이전트를 구축하고 평가하는 데 필요한 문제를 식별합니다.

- **Performance Highlights**: LLM이 생성한 아이디어는 인간 전문가 아이디어에 비해 더 혁신적으로 평가되었으나 실행 가능성은 다소 낮은 점수를 받았습니다. LLM은 아이디어의 다양성 부족과 자기 평가의 한계를 보였으며, 결론적으로 연구 결과의 차이가 실제로 의미 있는지 여부를 연구할 방안을 제안합니다.



### UI-JEPA: Towards Active Perception of User Intent through Onscreen User Activity (https://arxiv.org/abs/2409.04081)
- **What's New**: 본 논문에서는 UI 행동 시퀀스에서 사용자 의도를 생성하는 새로운 프레임워크인 UI-JEPA를 제안합니다. 이는 자기 지도 학습(self-supervised learning, SSL)과 LLM 디코더를 결합하여, 고품질 데이터셋의 부족을 극복하고도 UI 이해 능력을 향상시킵니다.

- **Technical Details**: UI-JEPA는 레이블이 없는 UI 비디오 데이터를 활용하여 추상적인 UI 임베딩을 학습하며, LLM 디코더를 통해 사용자 의도를 예측합니다. JEPA 스타일의 목표를 활용하여 학습된 표현은 기존의 성능을 유지하면서도 데이터와 자원 등에서의 요구 조건을 크게 줄일 수 있습니다.

- **Performance Highlights**: UI-JEPA는 50.5배의 계산 비용 절감과 6.6배의 지연 시간 개선을 달성하며, intent similarity scores에서 GPT-4 Turbo와 Claude 3.5 Sonnet 보다 각각 10.0% 및 7.2% 높은 성과를 보여줍니다. 이는 가벼우면서도 높은 성능의 UI 이해가 가능하다는 것을 시사합니다.



### AnyMatch -- Efficient Zero-Shot Entity Matching with a Small Language Mod (https://arxiv.org/abs/2409.04073)
Comments:
          12 pages excluding references, 3 figures, and 5 tables

- **What's New**: 이번 연구에서는 레이블이 없는 상태에서 두 레코드가 동일한 실제 엔티티를 나타내는지를 판단하는 제로샷 엔터티 매칭(Zero-Shot Entity Matching) 문제를 해결하기 위해 AnyMatch라는 소규모 언어 모델을 제안합니다. 기존의 대형 언어 모델(LLMs)에 비해 저비용으로 높은 예측 품질을 제공함을 강조합니다.

- **Technical Details**: AnyMatch는 주어진 데이터의 레이블 없이도 효과적인 매칭을 수행하기 위해 전이 학습(Transfer Learning) 설정에서 진행됩니다. 데이터 선택 기법으로는 AutoML 필터를 통한 어려운 쌍 선택, 속성 수준의 예시 생성, 그리고 데이터 내 레이블 불균형 조정이 포함됩니다. 또한, 제로샷 엔터티 매칭을 시퀀스 분류 문제(Sequence Classification Problem)로 모델링하였습니다.

- **Performance Highlights**: AnyMatch는 아홉 개의 기준 데이터셋에서 13개의 베이스라인 모델과 비교하여 두 번째로 높은 F1 점수를 달성했습니다. 또한, 비록 파라미터 크기가 적음에도 불구하고, 많은 대형 모델을 사용하는 접근 방식보다 평균적으로 4.4%의 예측 품질을 유지하며, 3,899배 저렴한 추론 비용을 발생시킵니다.



### Self-Harmonized Chain of Though (https://arxiv.org/abs/2409.04057)
- **What's New**: 이 논문에서는 ECHO라는 새로운 Chain-of-Thought (CoT) 프롬프트 방법을 제안합니다. ECHO는 다양한 해결 경로를 통합하여 일관되고 효과적인 솔루션 패턴을 생성합니다.

- **Technical Details**: ECHO는 세 가지 주요 단계로 구성됩니다: 질문 클러스터링, 수집된 질문에 대한 Zero-shot-CoT 수행, 그리고 각 반복에서 데모의 합성을 개선하는 동적 프롬프트 메커니즘을 적용합니다.

- **Performance Highlights**: ECHO는 세 가지 추론 도메인에서 다른 기초 모델에 비해 평균 2.8% 더 뛰어난 성능을 보여주었습니다.



### Towards Safer Online Spaces: Simulating and Assessing Intervention Strategies for Eating Disorder Discussions (https://arxiv.org/abs/2409.04043)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 Eating Disorders (ED) 관련 토론에서 개입 전략을 시뮬레이션하고 평가할 수 있는 LLM 기반의 실험 테스트베드를 제시합니다. 다양한 소셜 미디어 플랫폼에서 생성된 합성 대화를 통해 개입 전략을 통제된 환경에서 실험할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구진은 Reddit, Twitter 및 ED 포럼 등 다양한 플랫폼에서 에 대한 대화를 시뮬레이션하기 위해 LLM 기반의 실험 테스트베드를 사용했습니다. 이 테스트베드는 대화 생성, 개입 전략 구현 및 후속 분석을 포함하는 체계적인 접근 방식을 사용하여 수천 개의 대화를 생성하고 평가하는 기능을 제공합니다.

- **Performance Highlights**: 개입 전략의 효과를 평가한 결과, 예의 바른 방식의 개입이 모든 차원에서 긍정적인 감정과 정서적 톤을 일관되게 향상시키는 것으로 나타났습니다. 반면, 통찰력을 재설정하는 접근 방식은 부정적인 감정을 증가시키는 경향이 있었습니다. 모델 선택에 따른 인지 메트릭스의 차이가 중요하다는 점도 강조되었습니다.



### Large Margin Prototypical Network for Few-shot Relation Classification with Fine-grained Features (https://arxiv.org/abs/2409.04009)
Comments:
          Accepted by CIKM'19

- **What's New**: 이 논문에서는 few-shot learning을 활용하여 관계 분류의 성능을 향상시키기 위한 새로운 프레임워크인 LM-ProtoNet (FGF)을 제안합니다.

- **Technical Details**: 제안된 LM-ProtoNet은 fine-grained features를 생성하고 large-margin learning을 추가함으로써 기존의 ProtoNet을 개선합니다. 모델은 CNN을 사용하여 문장과 구문 수준의 임베딩을 생성하며, 이러한 임베딩을 조합하여 더 세밀한 특징을 제공합니다.

- **Performance Highlights**: FewRel 데이터셋을 사용한 실험 결과, LM-ProtoNet은 여러 기준선 방법보다 6.84%의 정확도 향상을 기록하며 우수한 성능을 보여줍니다.



### On The Role of Prompt Construction In Enhancing Efficacy and Efficiency of LLM-Based Tabular Data Generation (https://arxiv.org/abs/2409.03946)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)을 기반으로 한 테이블 데이터 생성에서 도메인 특화 통찰력을 활용한 텍스트 프롬프트 개선의 효과를 조사합니다. 이에 따라 세 가지 프롬프트 구성 프로토콜인 Expert-guided, LLM-guided, Novel-Mapping을 제안합니다.

- **Technical Details**: 세 가지 프롬프트 구성 프로토콜은 다음과 같습니다: (i) Expert-guided: 도메인 전문가가 필요한 설명을 제공합니다; (ii) LLM-guided: 외부 LLM이 자동으로 피처 설명을 생성합니다; (iii) Novel-Mapping: 외부 LLM을 사용하여 일반 피처 이름을 새로운 도메인의 의미 있는 피처로 매핑합니다. 이 방법은 GReaT(Generation of Realistic Tabular data) 프레임워크에 통합되어 실험되었습니다.

- **Performance Highlights**: 실험 결과, 도메인 별로 풍부한 프롬프트를 사용했을 시 데이터 생성 품질과 훈련 효율성이 모두 개선되었습니다. 특히, Novel-Mapping 프로토콜은 GReaT 기본 모델에 비해 각각의 피처 이름에 대한 접근이 없더라도 더 나은 성능을 보였습니다. 그리고 필요한 에포크 수 또한 25% 이하로 줄어들었습니다.



### Experimentation in Content Moderation using RWKV (https://arxiv.org/abs/2409.03939)
- **What's New**: 이번 연구는 RWKV 모델의 콘텐츠 조정(Content Moderation) 효율성을 조사하며, 이를 위해 작거나 훈련 가능한 모델로 적합한 새로운 데이터셋을 도입했습니다. 이 데이터셋은 이미지, 비디오, 오디오 및 텍스트 데이터를 포함하고 있으며, 다양한 사회적 문제를 다룹니다.

- **Technical Details**: 본 연구는 RWKV 모델을 미세 조정(Fine-tuning)하여, 대규모 콘텐츠 조정 작업을 수행하는 데 최적화된 CPU 효율적인 아키텍처를 활용합니다. 데이터셋을 통해 558,958개의 텍스트 응답과 83,625개의 이미지 응답을 생성하여 콘텐츠 조정 시스템을 훈련하고 개선하는 데 사용했습니다.

- **Performance Highlights**: RWKV 모델은 콘텐츠 조정의 정확성과 효율성을 향상시키는 데 기여하며, 이 연구는 더욱 작고 효율적인 자원 소모 모델을 개발하는 기회를 제공합니다.



### CACER: Clinical Concept Annotations for Cancer Events and Relations (https://arxiv.org/abs/2409.03905)
Comments:
          This is a pre-copy-editing, author-produced PDF of an article accepted for publication in JAMIA following peer review. The definitive publisher-authenticated version is available online at this https URL

- **What's New**: 이번 연구에서는 Clinical Concept Annotations for Cancer Events and Relations (CACER)라는 새로운 데이터셋을 소개하고 있습니다. 이 데이터셋은 48,000개의 의료 문제와 약물 이벤트, 그리고 10,000개의 약물-문제 및 문제-문제 관계에 대한 세부적인 주석을 포함하고 있습니다.

- **Technical Details**: CACER는 Fred Hutch Cancer Center에서 수집된 암 환자의 임상 온콜로지 노트로 구성되어 있으며, 자연어 처리(NLP) 방법을 통해 비구조화된 텍스트에서 구조화된 데이터로 변환하는 정보를 추출합니다. BERT, Flan-T5, Llama3 및 GPT-4와 같은 transformer 기반 모델을 사용하여 fine-tuning 및 in-context learning (ICL)을 평가하였습니다.

- **Performance Highlights**: BERT와 Llama3 모델은 이벤트 추출에서 각각 88.2 및 88.0 F1 score를 기록하여, 주석자 간 합의(inter-annotator agreement, IAA)인 88.4 F1 score와 유사한 성능을 보였습니다. 관계 추출에서는 BERT, Flan-T5 및 Llama3가 각각 61.8 - 65.3 F1 score로 최상의 성능을 나타냈습니다. GPT-4는 모든 과제에서 가장 낮은 성능을 보였으며, 주석된 훈련 데이터와 모델 최적화의 중요성을 강조하고 있습니다.



### Sirius: Contextual Sparsity with Correction for Efficient LLMs (https://arxiv.org/abs/2409.03856)
- **What's New**: 이 논문은 Contextual Sparsity (CS)의 성능 저하 문제를 지적하며, 고급 추론 및 지식 기반 작업에서 CS 모델의 성능이 크게 감소한다는 점을 강조합니다. 새로운 수정 메커니즘인 Sirius를 도입하여, CS 모델의 품질을 향상시키면서도 효율성을 유지할 수 있는 방법을 제시합니다.

- **Technical Details**: Sirius는 추론 작업 중 CS 모델의 품질을 회복하는 효율적인 수정 메커니즘입니다. 이를 위해, Sirius는 KV Cache 직접 재작성, 최소 롤백, 하드웨어 효율적인 트리 구축 같은 기술을 사용하여 전체 모델의 수정 효율성을 높입니다. 논문에서 6개의 모델과 8개의 복잡한 생성 작업에서 Sirius의 효과를 평가하였습니다.

- **Performance Highlights**: Sirius는 8B 모델의 경우 약 20%, 70B 모델에서는 35%의 지연 시간을 감소시켰습니다. GSM8K 및 Llama-3-8B-Instruct 작업에서, Sirius는 세밀한 희소성(Fine-grained sparsity)을 58%에서 72%로, 거칠은 희소성(Coarse-grained sparsity)을 38%에서 70%로 향상시켰습니다.



### Persona Setting Pitfall: Persistent Outgroup Biases in Large Language Models Arising from Social Identity Adoption (https://arxiv.org/abs/2409.03843)
Comments:
          23 pages, 5 figures

- **What's New**: 본 연구는 대형 언어 모델(LLMs)이 특정 프롬프트에 의해 부여된 정체성을 어떻게 내면화하는지를 탐구하며, '우리(ingroup)'와 '그들(outgroup)' 간의 구분을 나타낸다. 특히, outgroup bias의 존재를 강조하고 이를 완화할 수 있는 방법을 제시한다.

- **Technical Details**: 연구는 Social Identity Theory(SIT)를 기반으로 하며, LLM에 정치적 정체성을 부여하여 ingroup favoritism과 outgroup bias를 분석한다. 또한, 특정 사회 집단의 관점을 채택하도록 유도하여 inherent bias를 완화하는 방법을 제안한다.

- **Performance Highlights**: 실험 결과, LLM이 gender bias를 포함한 다양한 사회적 편견에 대해 ingroup favoritism 만큼이나 강한 outgroup bias를 보인다. 연구 방법론은 성별 편향에도 적용 가능하며, 공정하고 균형 잡힌 언어 모델 개발의 가능성을 보여준다.



### Empirical Bayesian image restoration by Langevin sampling with a denoising diffusion implicit prior (https://arxiv.org/abs/2409.04384)
Comments:
          24 pages

- **What's New**: 본 논문은 사전 훈련된 Denoising Diffusion Probabilistic Models (DDPM)을 통합한 Bayesian Langevin 알고리즘을 제안하여 이미지 복원에서의 효율성을 크게 향상시킵니다.

- **Technical Details**: 이 방법은 원래의 likelihood function을 직접 사용할 수 있도록 하여, 복잡한 likelihood 문제를 회피하면서 가능성 추정에 대한 empirical Bayesian 기술을 활용합니다. 이를 통해 이미지 블러링(deblurring), 초해상도(super-resolution), 및 인페인팅(inpainting)과 같은 다양한 작업에서의 모델 하이퍼파라미터를 자동으로 조정합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 기준 작업에서 state-of-the-art 방법들과 비교하여 이미지 추정 정확도 및 컴퓨팅 시간 모두에서 우수한 성능을 보여줍니다.



### AGR: Age Group fairness Reward for Bias Mitigation in LLMs (https://arxiv.org/abs/2409.04340)
Comments:
          The first two authors contributed equally to this work. Corresponding to Zhiqiang Wang. ACKNOWLEDGMENT: we would like to thank the computing resources support from the State Key Laboratory of New Computer Software Technologies at Nanjing University

- **What's New**: 본 논문에서는 LLMs의 나이 편향(age bias)을 탐지하고 측정하기 위한 나이 편향 선호 데이터셋(age bias preference datasets)과 지시 조정 데이터셋(instruction-tuning datasets)을 구축하여 이를 해결하고자 하였습니다. 또한, 다양한 연령 그룹 간의 응답 품질 차이를 줄이기 위한 나이 공정성 보상(Age Fairness Reward, AGR)을 도입했습니다.

- **Technical Details**: 이 연구에서는 기존의 BBQ 및 ISB 데이터셋을 수정 및 확장하여 나이 관련 편향 평가를 위한 나이 선호 및 지시 조정 데이터셋을 수작업으로 주석을 달아 제작하였습니다. AGR은 다양한 LLM에서 나이 편향 완화를 위한 훈련 중 성능 차이를 줄이기 위해 공정성을 고려한 보상을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 AGR은 응답 정확성을 크게 향상시키고, 다양한 나이 그룹 간의 성능 격차를 줄이는 데 효과적임을 보여주었습니다. 실험 결과는 AGR이 기존의 관련 방법들보다 우수한 성능을 발휘함을 입증합니다.



### Using Large Language Models to Generate Authentic Multi-agent Knowledge Work Datasets (https://arxiv.org/abs/2409.04286)
Comments:
          Accepted and in press (INFORMATIK Festival, Wiesbaden, 2024)

- **What's New**: 이 논문에서는 현재 공개적으로 이용 가능한 지식 작업 데이터의 다양성 부족과 주석 부족, 사용자 및 문서에 대한 맥락 정보 부족 문제를 지적합니다. 이를 해결하기 위해, 구성 가능하고 다중 에이전트 기반의 지식 작업 데이터셋 생성기를 제안합니다.

- **Technical Details**: 제안된 시스템은 에이전트 간의 협업 지식 작업을 시뮬레이션하여 Large Language Model (LLM)로 생성된 문서와 관련 데이터 흔적을 생성합니다. 생성 과정에서 구성된 모든 배경 정보는 지식 그래프 (knowledge graph)로 기록됩니다. 이 데이터셋은 개인정보 및 기밀 문제 없이 사용 및 공유할 수 있습니다.

- **Performance Highlights**: 인간 평가자들이 생성된 문서 53%와 실제 문서 74%를 현실적이라고 평가한 결과는 제안된 접근법의 가능성을 보여줍니다. 참가자들의 피드백을 바탕으로 진정성 기준을 분석하고, 공통적으로 발견된 문제에 대한 개선 방안도 논의합니다.



### An overview of domain-specific foundation model: key technologies, applications and challenges (https://arxiv.org/abs/2409.04267)
- **What's New**: ChatGPT와 같은 기본 모델 기반 제품의 성능 향상으로 인해, 특정 산업 및 응용 시나리오에 맞춰 이러한 모델을 조정하는 방법에 대한 탐색이 이루어지고 있습니다. 이 논문은 도메인 특정(restricted) 기본 모델(customization of domain-specific foundation models)의 구축 방법론에 대해 포괄적인 개요를 제공합니다.

- **Technical Details**: 도메인 특정 기본 모델은 특정 산업을 위한 데이터와 응용 프로그램을 사용하여 개발됩니다. 이 모델들은 도메인 특정 데이터로 많이 훈련되어 있으며, LLM(대형 언어 모델) 기술에 기반한 인공신경망 모델입니다. 이 모델들은 '모달리티 인코더(Modality Encoder)', '입력 프로젝터(Input Projector)', '백본 계산기(Backbone Calculator)', '출력 프로젝터(Output Projector)', '모달리티 디코더(Modality Decoder)'로 구성된 다중 모달리티 구조를 가지고 있습니다.

- **Performance Highlights**: 도메인 특정 기본 모델은 일반 목적 모델에 비해 특정 분야의 전문적인 내용을 더 정확하게 이해하고 생성할 수 있는 능력을 가지고 있으며, 기술적 보안과 높은 경제적 이익이 기대됩니다. 그러나 데이터 수집 및 전처리 과정에서 여러 도전에 직면하게 됩니다.



### Fast Forwarding Low-Rank Training (https://arxiv.org/abs/2409.04206)
- **What's New**: 본 논문에서는 저차원(adaptation) 미세조정(finetuning) 방법인 Low-Rank Adaptation (LoRA)을 활용하여 사전 학습된 언어 모델의 훈련 비용을 줄이는 새로운 최적화 전략인 Fast Forward를 제안합니다.

- **Technical Details**: Fast Forward 단계에서는 마지막 최적화 단계를 반복하여 손실이 개선되지 않을 때까지 진행합니다. 이를 통해 일반 SGD와 Adam에 비해 FLOPs를 최대 87% 감소시키고 훈련 시간을 최대 81% 단축할 수 있습니다. Fast Forward는 정규 최적화와 빠른 풀링 단계를 번갈아 진행합니다.

- **Performance Highlights**: Fast Forward는 저차원 미세조정에서 매우 효과적으로 작동하며, 같은 성능을 41–87% 더 빠르게 달성할 수 있음을 보여주었습니다. 여러 모델을 다양한 작업에서 미세조정하여 성능 저하 없이 훈련 속도를 향상시켰습니다.



### Residual Stream Analysis with Multi-Layer SAEs (https://arxiv.org/abs/2409.04185)
Comments:
          16 pages, 12 figures

- **What's New**: 이 논문에서는 Transformer 언어 모델의 내부 표현(Representations)을 해석하기 위한 새로운 접근법으로 Multi-layer Sparse Autoencoder (MLSAE)를 소개합니다.

- **Technical Details**: MLSAE는 각 Transformer 레이어의 잔여 스트림 활성화 벡터(Residual Stream Activation Vectors)를 동시에 사용하여 훈련된 단일 SAE입니다. 이 방식은 정보가 레이어 간에 어떻게 흐르는지를 연구하는 데 유용합니다.

- **Performance Highlights**: 연구 결과, 각 레이어에서 활성화되는 개별 SAE 특징이 발견되었으며, 더 큰 모델일수록 인접한 레이어 간의 코사인 유사도(Cosine Similarities)가 증가함을 확인했습니다. 이는 정보 흐름을 연구하는 데 있어 MLSAE가 유망한 방법임을 보여줍니다.



### Confidence-Aware Document OCR Error Detection (https://arxiv.org/abs/2409.04117)
- **What's New**: 본 연구에서는 OCR(Optical Character Recognition)의 신뢰도 점수(confidence scores)를 활용하여 OCR 오류 감지를 향상시키기 위한 새로운 모델, ConfBERT를 제안합니다. 이 모델은 OCR 시스템 간의 신뢰도 점수와 오류율 간의 상관관계를 분석하고, 이러한 점수를 통합하여 오류 감지 능력을 개선할 수 있음을 보여줍니다.

- **Technical Details**: ConfBERT 모델은 BERT 기반으로, OCR 신뢰도 점수를 토큰 임베딩(token embeddings)에 통합하며, 노이즈 조정을 위한 선택적 프리트레이닝(pre-training) 단계를 제공합니다. 우리는 상업적 및 오픈소스 OCR 솔루션의 성능 및 신뢰도 점수를 비교 분석하고, 이러한 점수를 사용하여 포스트-OCR 오류 감지를 개선하는 방법을 제안합니다.

- **Performance Highlights**: 우리의 실험 결과는 OCR 신뢰도 점수를 통합함으로써 오류 감지 능력이 향상될 수 있음을 나타냅니다. 또한, 상업적 OCR 기술과 오픈소스 OCR 기술 간 성능의 상당한 차이를 강조합니다.



### Structure and dynamics of growing networks of Reddit threads (https://arxiv.org/abs/2409.04085)
Comments:
          29 pages, 9 figures, 5 tables

- **What's New**: 이번 연구는 Reddit의 /r/AmItheAsshole (AITA) 커뮤니티 내에서 사용자 상호작용의 복잡한 네트워크를 모델링하고 분석하는 데 초점을 맞추었습니다. 연구팀은 의견 불일치와 상호작용 자율성이 사용자 행동에 미치는 영향을 심층적으로 탐구합니다.

- **Technical Details**: 연구에서는 2023년에 주목받은 6000개 이상의 AITA 스레드를 수집하고, 각 스레드에서 개인의 판단 수량과 집단의 불일치 수준을 계산했습니다. 스레드를 시간에 따라 진화하는 복합 멀티 그래프 네트워크로 모델링하고 구조적 특성의 진화를 기존 문헌과 비교했습니다.

- **Performance Highlights**: Reddit 네트워크의 진화는 다른 실제 소셜 네트워크와 다르며, 사용자 상호작용의 특성이 시간이 지남에 따라 변화하는 방식이 드러났습니다. 특히, AITA 커뮤니티에서는 명확한 판단 요청이 요구되어 공동체의 상호작용과 의견 형성을 연구하기에 좋은 환경을 제공합니다.



### Refining Wikidata Taxonomy using Large Language Models (https://arxiv.org/abs/2409.04056)
Comments:
          ACM International Conference on Information and Knowledge Management, Oct 2024, Boise, Idaho, United States

- **What's New**: Wikidata의 복잡한 분류 체계를 자동으로 정리하는 WiKC라는 새로운 버전을 제시합니다. 이 과정에서는 Large Language Models (LLMs)와 graph mining 기법을 결합하여 사용하였습니다.

- **Technical Details**: WiKC는 zero-shot prompting을 사용하여 Wikidata의 분류 체계에 대한 링크를 자르거나 클래스를 병합하는 작업을 수행합니다. 이러한 방식은 분류 체계의 자동화를 통해 수작업의 오류와 주관적인 결정을 줄입니다.

- **Performance Highlights**: 정제된 분류 체계의 품질은 내재적(intrinsic) 및 외재적(extrinsic) 관점에서 평가되었으며, 실질적인 관심을 보여주는 entity typing 작업에서 그 효과를 입증하였습니다.



### How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data (https://arxiv.org/abs/2409.03810)
Comments:
          Working in progress

- **What's New**: 최근 코드 지침 튜닝 데이터의 품질을 높이는 방법에 대한 관심이 증가하고 있습니다. 연구자들은 다양한 데이터셋에서 심각한 데이터 누수가 발생하고 있음을 발견하였고, 이를 해결하기 위한 효율적인 데이터 선택 전략을 제안했습니다.

- **Technical Details**: 이 논문에서는 코드 지침 데이터의 세 가지 차원인 지침 복잡성(instruction complexity), 응답 품질(response quality), 지침 다양성(instruction diversity)을 기반으로 한 데이터 정제(pruning) 전략을 통해 고품질 데이터를 선택하는 방법을 제안합니다. 최종적으로, 이러한 데이터를 토대로 LLaMA3 모델에서 파인튜닝된 XCoder 모형을 제시하며, 이를 통해 이전의 다른 모델들보다 더 적은 데이터로도 우수한 성과를 냄을 보여줍니다.

- **Performance Highlights**: XCoder는 LiveCodeBench 및 HumanEval 벤치마크에서 기존 모형과 비교했을 때 우수한 성능을 기록했습니다. 예를 들어, XCoder-8B는 40K 데이터 샘플을 사용하여 LiveCodeBench에서 43.66, HumanEval에서 54.9의 성과를 달성했습니다.



### NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls (https://arxiv.org/abs/2409.03797)
- **What's New**: 최근 대형 언어 모델(LLMs)을 활용한 자율 에이전트 응용 프로그램이 복잡한 실제 작업을 해결하는 효과적인 도구로 주목받고 있습니다. 본 논문에서는 LLM의 중첩 API 호출 능력을 평가하기 위해 NESTFUL이라는 새로운 벤치마크를 제시합니다.

- **Technical Details**: NESTFUL은 총 300개의 인간 주석 샘플로 구성되어 있으며, 실행 가능한(executable) 샘플과 실행 불가능한(non-executable) 샘플로 나뉩니다. 각 샘플은 사용자 요청에 대한 답변으로 API 호출 시퀀스를 포함하고 있으며, 이는 JSON 객체로 표현됩니다. 중첩 API 호출의 경우, 한 API 호출의 출력이 후속 API 호출의 입력으로 사용됩니다.

- **Performance Highlights**: NESTFUL에서의 평가 결과, 대부분의 최신 LLM 모델들이 기존 벤치마크에서의 성능에 비해 중첩 API 호출 작업에서 저조한 성능을 보였습니다. 이는 API 호출 능력의 발전을 테스트할 수 있는 유용한 길잡이가 될 것으로 기대됩니다.



### HSF: Defending against Jailbreak Attacks with Hidden State Filtering (https://arxiv.org/abs/2409.03788)
Comments:
          13 pages

- **What's New**: 본 연구에서는 Hidden State Filter (HSF)를 제안하여 LLM(jailbreak 공격)으로부터의 방어를 위한 새로운 접근 방식을 소개합니다. 이 방식은 기존의 방어 방법들과 달리, 모델의 추론 과정 이전에 적대적 입력을 식별하고 거부할 수 있는 아키텍처 기반의 방어 메커니즘입니다.

- **Technical Details**: HSF는 LLM의 최종 Decoder Layer에서 마지막 k tokens로부터 특징(feature)을 샘플링하여 이를 경량의 분류 모델로 학습합니다. 이는 플러그인 모듈로 통합되어, LLM의 기본 아키텍처를 변경하지 않으면서 기존 LLM에 쉽게 통합이 가능합니다.

- **Performance Highlights**: 실험 결과, HSF는 여섯 가지 최신 jailbreak 공격에 대한 방어 성능을 크게 향상시켰으며, benign 쿼리에 대한 응답에는 최소한의 영향을 미치고, 적은 계산 자원으로 높은 효과성을 보여주었습니다. HSF는 모든 기준선을 초과하여 낮은 오탐률과 함께 운영되었습니다.



New uploads on arXiv(cs.IR)

### How Fair is Your Diffusion Recommender Model? (https://arxiv.org/abs/2409.04339)
- **What's New**: 이번 논문은 최근 확산 기반 추천 시스템인 DiffRec의 공정성(fairness) 문제를 분석하고, 발전 가능성을 모색한 첫 번째 연구 중 하나입니다. 기존의 생성형 추천 방법들이 기계 학습 문헌에서 언급된 여러 문제점들(예: 정보 편향)을 잠재적으로 포함하고 있다는 우려에 대한 연구를 진행했습니다.

- **Technical Details**: DiffRec는 사용자의 암묵적인 피드백을 효과적으로 학습하기 위해 확산 모델(diffusion models) 기반의 프로세스를 사용합니다. 실험 설정에는 DiffRec 및 L-DiffRec과 9개의 최신 추천 모델 및 공정성 인식 공헌 데이터셋이 포함되어 있으며, 성능 분석에 기초한 두 가지 접근 방식을 채택했습니다. 각 모델의 정확성과 추천의 공정성을 별도로 평가하고, 성능 간의 균형 트레이드오프를 확인했습니다.

- **Performance Highlights**: 실험 결과, 기존 기계 학습과 마찬가지로 공정성 우려가 존재함을 나타냈습니다. 그러나 L-DiffRec과 같은 추가 요소 도입으로 공정성 문제의 부정적인 영향을 어느 정도 완화할 수 있었음을 보여줍니다. 이는 향후 확산 기반 추천 시스템의 공정성을 유지하는 방향으로 나아갈 수 있는 길을 제시합니다.



### Enhancing Sequential Music Recommendation with Personalized Popularity Awareness (https://arxiv.org/abs/2409.04329)
Comments:
          Accepted by RecSys'24 as an LBR paper

- **What's New**: 이 논문에서는 음악 추천 분야의 고유한 도전 과제를 해결하기 위해 개인화된 인기 정보(personalized popularity information)를 통합한 새로운 접근법을 제안합니다. 기존의 Transformer-based 모델이 음악 청취의 반복성(repeated consumption patterns)을 제대로 반영하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구는 개인화된 인기 동향을 기반으로 사용자에게 선호하는 트랙을 지속적으로 추천하는 Personalized Most Popular recommender를 개발하였습니다. 이를 통해, 사용자-아이템 인기 점수(user-item popularity scores)와 모델 생성 점수를 결합하여 새로운 음악 탐색과 사용자 선호의 만족도를 효과적으로 균형 잡는 방식을 제안합니다.

- **Performance Highlights**: 실험 결과, 사용자 특정 인기 정보만으로 구성된 Personalized Most Popular recommender가 최신 최고 성능 모델을 초과하여 최상의 성능을 보였습니다. 또한, Transformer 기반 모델에 개인화된 인기 인식을 추가한 결과, 성능 향상이 25.2%에서 69.8%에 이르렀습니다.



### RETAIN: Interactive Tool for Regression Testing Guided LLM Migration (https://arxiv.org/abs/2409.03928)
Comments:
          Preprint

- **What's New**: RETAIN (REgression Testing guided LLM migrAtIoN)라는 도구가 소개되었습니다. 이 도구는 LLM 마이그레이션 동안 회귀 테스트를 위한 상호작용 인터페이스와 오류 탐지 모듈을 제공합니다.

- **Technical Details**: RETAIN은 다양한 프롬프트 반복에 대한 분석을 지원하는 인터랙티브 인터페이스와 모델 행동 차이를 이해하는 데 도움을 주는 오류 발견 모듈로 구성되어 있습니다. 이 모듈은 모델 출력 간의 다양한 오류를 텍스트 설명으로 생성하여 프롬프트 개선을 위한 실행 가능한 통찰력을 제공합니다.

- **Performance Highlights**: RETAIN은 수동 평가와 비교하여 참가자가 두 배 많은 오류를 식별하고, 75% 더 많은 프롬프트로 실험 할 수 있게 하였으며, 주어진 시간 내에 12% 더 높은 성과 점수를 기록했습니다.



### It's Not You, It's Me: The Impact of Choice Models and Ranking Strategies on Gender Imbalance in Music Recommendation (https://arxiv.org/abs/2409.03781)
Comments:
          6 pages, 3 figures, conference short paper, to be published at RecSys 2024

- **What's New**: 이번 연구는 음악 추천 시스템에서 아티스트 성별의 공정성 관련 편향을 분석하고, 알고리즘 전략과 사용자 행동이 공정성 개선에 미치는 상대적인 영향을 연구했습니다. 특히, 재정렬(re-ranking) 전략이 사용자 선택 모델(user choice model)보다 공정성에 더 큰 영향을 미친다는 사실을 발견했습니다.

- **Technical Details**: 연구는 LFM-2b 데이터셋을 사용하여 두 가지 기본 추천 모델인 IALS(implicit-feedback matrix factorization with alternating least squares)과 BPR(matrix factorization trained with pairwise rank loss)를 통해 추천 목록을 생성하고, 다양한 후처리 차별 완화 전략을 적용했습니다. 실험은 LensKit을 사용하여 진행되었습니다.

- **Performance Highlights**: 실험 결과, 재정렬 전략이 시간이 지남에 따라 추천 공정성에 대해 더 큰 영향을 미치는 것으로 나타났으며, 이는 성별 불균형 문제 해결에 기여할 수 있는 중요한 발견입니다.



### VERA: Validation and Evaluation of Retrieval-Augmented Systems (https://arxiv.org/abs/2409.03759)
Comments:
          Accepted in Workshop on Evaluation and Trustworthiness of Generative AI Models, KDD 2024

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 평가를 위한 새로운 프레임워크 VERA (Validation and Evaluation of Retrieval-Augmented Systems)를 소개합니다. VERA는 대형 언어 모델(LLMs)에서 활용되는 인출 정보의 출력을 투명하고 신뢰성 있게 향상시키기 위한 방법을 제안합니다.

- **Technical Details**: VERA는 RAG 시스템의 평가 방식을 두 가지 주요 방식으로 개선합니다: 첫째, 여러 차원 지표를 통합하여 단일 종합 점수를 생성하는 크로스 인코더 기반의 메커니즘을 도입합니다. 둘째, 문서 저장소의 LLM 기반 지표에 대한 부트스트랩 통계를 활용하여 신뢰 구간을 설정하고, 저장소의 주제적 범위를 보장합니다. 이를 통해 RAG 시스템의 정보 검색 및 생성 과정에서 신뢰성과 적합성을 평가합니다.

- **Performance Highlights**: 여러 사례를 통해 VERA가 AI 애플리케이션의 의사결정 과정 및 신뢰성을 어떻게 강화하는지를 보여줍니다. VERA는 대형 언어 모델 기반의 RAG 평가 지표 이론적 이해에 기여할 뿐만 아니라, 책임감 있는 AI 시스템의 실질적인 구현을 촉진하여 신뢰할 수 있고 투명한 생성적 AI 기술 개발에 중대한 발전을 이룹니다.



### A Survey on Knowledge Organization Systems of Research Fields: Resources and Challenges (https://arxiv.org/abs/2409.04432)
- **What's New**: 이 논문은 학문 분야에서 사용되는 지식 조직 시스템(Knowledge Organization Systems, KOSs)의 포괄적인 조사 결과를 제시합니다. 45개의 KOS를 분석하고 비교하여 범위, 구조, 관리, 사용, 다른 KOS와의 연결 등 5가지 주요 차원에 따라 차별화된 시나리오를 강조합니다.

- **Technical Details**: KOS는 용어 목록(term lists), 유의어 사전(thesauri), 분류법(taxonomies), 온톨로지(ontologies) 등으로 구성되어 있으며, 이는 연구 문서, 학술 과정, 특허, 도서, 과학적 장소 등 여러 가지 연구 제품과 관련된 항목을 분류하는 데 사용됩니다. 본 연구에서는 KOS의 범위, 구조(개념의 수, 최대 깊이, 계층 유형 등), 관리(형식, 라이센스, 업데이트 빈도 등), 다른 KOS와의 링크, 사용을 통해 KOS를 평가했습니다.

- **Performance Highlights**: 다양한 학문 분야에서 KOS가 문서 검색의 효율성을 높이고, 연구 동향을 분석하고 예측하는 데 기여한다는 점에서 성과를 보였습니다. 특히 AI(AI 기반 시스템)가 KOS를 통해 연구자들이 문헌을 탐색하고 체계적 검토를 반자동화하는 데 도움을 받고 있음이 강조되었습니다.



### WarpAdam: A new Adam optimizer based on Meta-Learning approach (https://arxiv.org/abs/2409.04244)
- **What's New**: 최적화 알고리즘의 선택은 딥 러닝 모델 훈련에 매우 중요합니다. 본 연구는 Meta Learning에서 'warped gradient descend' 개념을 Adam 옵티마이저에 통합하여 혁신적인 최적화 전략을 제안합니다.

- **Technical Details**: 전통적인 Adam 옵티마이저는 그래디언트를 사용하여 그래디언트 평균 및 분산의 추정을 계산하고, 이후 모델 파라미터를 업데이트합니다. 본 접근법은 학습 가능한 왜곡 행렬(P)을 도입하여 그래디언트를 선형적으로 변환합니다. 이 변환은 각 반복 중 그래디언트를 약간 조정하여 옵티마이저가 다양한 데이터셋 특성에 더 잘 적응할 수 있도록 합니다.

- **Performance Highlights**: 다양한 작업 및 데이터셋에서 실험 결과, 이 'warped gradient descend' 개념이 통합된 옵티마이저는 적응성 측면에서 우수함을 입증하였습니다. 또한, 적응 행렬 P의 효과적인 훈련 전략을 탐색하고 이 방법이 최적의 결과를 도출할 수 있는 시나리오를 찾아냈습니다.



### Refining Wikidata Taxonomy using Large Language Models (https://arxiv.org/abs/2409.04056)
Comments:
          ACM International Conference on Information and Knowledge Management, Oct 2024, Boise, Idaho, United States

- **What's New**: Wikidata의 복잡한 분류 체계를 자동으로 정리하는 WiKC라는 새로운 버전을 제시합니다. 이 과정에서는 Large Language Models (LLMs)와 graph mining 기법을 결합하여 사용하였습니다.

- **Technical Details**: WiKC는 zero-shot prompting을 사용하여 Wikidata의 분류 체계에 대한 링크를 자르거나 클래스를 병합하는 작업을 수행합니다. 이러한 방식은 분류 체계의 자동화를 통해 수작업의 오류와 주관적인 결정을 줄입니다.

- **Performance Highlights**: 정제된 분류 체계의 품질은 내재적(intrinsic) 및 외재적(extrinsic) 관점에서 평가되었으며, 실질적인 관심을 보여주는 entity typing 작업에서 그 효과를 입증하였습니다.



### Understanding Fairness Metrics in Recommender Systems: A Healthcare Perspectiv (https://arxiv.org/abs/2409.03893)
Comments:
          Accepted to the 18th ACM Conference on Recommender Systems

- **What's New**: 본 연구는 인공지능 기반 결정 시스템에서 공정성에 대한 대중의 이해를 조사합니다. 의료 추천 시스템에서 공정성 메트릭을 선택하는 설문조사를 통해 다양한 시나리오에서 대중이 공정성을 어떻게 인식하고 이해하는지를 파악했습니다.

- **Technical Details**: 설문조사는 세 가지 부분으로 구성되었으며, 참가자들은 Demographic Parity, Equal Accuracy, Equalized Odds, Positive Predictive Value 중에서 선택하게 됩니다. 두 가지 시나리오(하이 스테이크 및 저 스테이크)에서 공정성 메트릭을 비교하여 결정하는 과정에서 대중의 공정성 인지 수준이 낮음을 발견했습니다.

- **Performance Highlights**: 연구 결과, 다양한 의료 시나리오에서 대중이 이해하는 공정성의 정의가 일관되지 않으며, 이는 알고리즘 공정성에 대한 교육과 정보 제공의 필요성을 강조합니다. 또한, 특정 맥락에 따라 공정성에 대한 접근 방식이 다를 수 있음을 시사합니다.



New uploads on arXiv(cs.CV)

### Synergy and Synchrony in Couple Dances (https://arxiv.org/abs/2409.04440)
- **What's New**: 이 논문은 소셜 상호작용이 개인의 행동에 미치는 영향을 연구하며, 특히 커플 댄스에서의 예측 문제를 다룹니다. 기존의 행동 예측 모델은 과거 움직임만을 기반으로 하지만, 이 연구는 파트너의 움직임을 고려한 예측의 이점을 강조합니다.

- **Technical Details**: 우리는 스윙(Swing) 댄스를 예로 들어, 댄서의 동작을 연속적인 원자 모션 요소로 표현합니다. VQ-VAE를 사용해 정량화된 원자 모션 요소의 분산형 사전을 학습하고, 자가회귀(transformer) 모델을 통해 과거 데이터를 바탕으로 댄서의 미래 동작을 예측합니다.

- **Performance Highlights**: 뛰어난 예측 성능을 달성하며, 커플 댄스 비디오 데이터 세트를 생성하여 공개합니다. 실험 결과, 상호작용 파트너의 소셜 정보가 인간 행동의 예측 가능성을 크게 향상시키는 것으로 나타났습니다.



### VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation (https://arxiv.org/abs/2409.04429)
Comments:
          11 pages, 7 figures, 8 tables

- **What's New**: VILA-U는 비디오, 이미지 및 언어 이해 및 생성에 통합된 Unified foundation model로, 기존의 VLM과 달리 단일 autoregressive next-token prediction 프레임워크를 사용하여 복잡성과 비일치를 줄이고 있습니다.

- **Technical Details**: 이 모델은 discrete visual tokens를 텍스트 입력과 정렬하며, contrastive learning을 통해 시각적 입력을 discrete tokens로 변환합니다. VILA-U는 text-image alignment을 통해 시각적 인식 능력을 향상시키고, 높은 품질의 데이터 세트를 사용하여 autoregressive image generation을 통해 diffusion models와 유사한 품질로 이미지를 생성합니다.

- **Performance Highlights**: VILA-U는 이미지 언어 이해, 비디오 언어 이해, 이미지 생성 및 비디오 생성과 같은 다양한 비주얼 언어 작업에서 고성능을 발휘하며, end-to-end autoregressive 모델과 VLM 간의 성능 격차를 크게 줄였습니다.



### Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation (https://arxiv.org/abs/2409.04410)
- **What's New**: Open-MAGVIT2는 300M에서 1.5B까지의 오토 회귀(auto-regressive) 이미지 생성 모델 패밀리를 선보입니다. 이 프로젝트는 구글의 MAGVIT-v2 토크나이저를 오픈 소스로 재구현하며, ImageNet 256 × 256에서 새로운 최첨단 복원 성능(1.17 rFID)을 달성합니다.

- **Technical Details**: Open-MAGVIT2는 두 가지 주요 단계로 구성됩니다: 1) 강력한 비주얼 토크나이저가 시각적 신호를 이산적인 토큰 표현으로 매핑합니다. 2) 벡터 양자화된 시퀀스는 오토 회귀 트랜스포머로 전송되어 토큰 간의 관계 모델링을 통해 시각적 합성을 수행합니다. MAGVIT-v2에서는 대용량 코드북을 비대칭 토큰 분해(asymmetric token factorization)를 통해 두 개의 하위 어휘로 분리하고, 하위 토큰 상호작용을 향상시키기 위한 '다음 하위 토큰 예측(next sub-token prediction)'을 도입했습니다.

- **Performance Highlights**: Open-MAGVIT2는 강력한 토크나이저 덕분에 일반적인 오토 회귀 모델보다 뛰어난 성능과 확장성을 보입니다. 특히, 이미지넷에서 실험한 결과, 기존 방법론들을 초월하는 성능을 보여주며, 비전 지향적 디자인을 활용하는 MAGVIT-v2의 아키텍처를 넘어서는 성과를 달성했습니다.



### Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation (https://arxiv.org/abs/2409.04409)
Comments:
          Accepted to ECCV 2024. Project repository: this http URL

- **What's New**: 이 논문에서는 3D 시멘틱 세그멘테이션(semantic segmentation)에서의 소스 없는 비지도 도메인 적응(Source-free Unsupervised Domain Adaptation, SFUDA) 문제를 다룹니다. 이 방법은 소스 데이터에 접근하지 않고 레이블이 없는 타겟 도메인에서 도메인 적응을 수행합니다.

- **Technical Details**: 이 연구에서는 학습 문제의 정규화(regularization) 방법과 참조 모델(reference model)과의 일치 기반 기준을 도입하여 문제를 완화하는 두 가지 전략을 제안합니다. 이 기준은 (1) 적절할 때 훈련을 중단하는 데 사용되며, (2) 타겟 도메인에 대한 정보 없이 하이퍼파라미터(hyperparameter)를 선택하는 데 검증자로 사용됩니다.

- **Performance Highlights**: 이 방법은 다양한 3D 라이다(lidar) 설정에서 검증되었으며, 상태-of-the-art(state-of-the-art) 성능을 달성하여 기존 SFUDA 방법 대비 안정적인 성능 향상을 보였습니다.



### HiSC4D: Human-centered interaction and 4D Scene Capture in Large-scale Space Using Wearable IMUs and LiDAR (https://arxiv.org/abs/2409.04398)
Comments:
          17 pages, 10 figures, Jornal

- **What's New**: HiSC4D는 역동적 디지털 세계를 생성하기 위한 새로운 방법으로, 넓은 실내외 장면, 다양한 인간 동작 및 상호작용을 정확하고 효율적으로 포착할 수 있습니다.

- **Technical Details**: HiSC4D는 IMU(Inertial Measurement Unit) 및 LiDAR를 활용하여 드리프트(derrift) 없는 인간 동작을 캡처합니다. 여러 센서의 데이터를 통합하는 공동 최적화(joint optimization) 방법을 사용하여 대형 장면에서 안정적인 인간 모션 캡처가 가능합니다.

- **Performance Highlights**: HiSC4D는 다양한 시나리오(예: 농구 체육관, 상업 거리)에서 사람들의 상호작용을 효과적으로 포착하였으며, 4D 데이터셋은 36,000 프레임의 정확한 인간 모션을 제공합니다. 이 데이터셋은 연구 목적으로 공개될 예정입니다.



### Future Does Matter: Boosting 3D Object Detection with Temporal Motion Estimation in Point Cloud Sequences (https://arxiv.org/abs/2409.04390)
- **What's New**: 이 논문에서는 LiDAR 3D 객체 탐지의 새로운 프레임워크인 LiSTM을 제안하여 공간-시간 특성 학습에 중점을 두고 있습니다. 특히 비학습 가능한 운동 추정 모델에서 생성된 동적 우선 정보를 통합하여 탐지 성능을 향상시키고자 합니다.

- **Technical Details**: Motion-Guided Feature Aggregation (MGFA) 메커니즘을 통해 객체의 궤적 정보를 활용하여 공간-시간 관계를 모델링하였습니다. 또한, Dual Correlation Weighting Module (DCWM)을 설계하여 이전 프레임과 미래 프레임 간의 상호 작용을 효과적으로 촉진합니다. 최종적으로, 캐스케이드 크로스-어텐션 기반 디코더를 통해 3D 예측을 정제합니다.

- **Performance Highlights**: 제안된 프레임워크는 Waymo 및 nuScenes 데이터셋에서 실험을 수행한 결과, Waymo 데이터셋에서 CenterPoint보다 8% 향상된 3D 탐지 성능을 보여주었습니다.



### Question-Answering Dense Video Events (https://arxiv.org/abs/2409.04388)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 새로운 작업인 dense video events의 질문-답변(task)을 제안합니다. 이 작업은 긴 비디오에서 복잡한 사건을 이해하고, 여러 시간대에 걸친 사건들에 대한 질문에 답하는 것을 목표로 합니다.

- **Technical Details**: DeVE-QA라는 데이터셋을 구성하여 10.6K 개의 긴 비디오에 대해 26K 개의 사건에 관한 78K 개의 질문을 포함하고 있습니다. 또한, 새로운 MLLM 훈련 기법인 DeVi를 제안하며, 이 방법은 계층적 캡셔닝 모듈, 시간적 사건 메모리 모듈, 자기 일관성 확인 모듈을 포함하여 긴 비디오에서의 질문-답변 성능을 향상시킵니다.

- **Performance Highlights**: DeVi는 DeVE-QA에서 4.1%의 정확도 향상과 NExT-GQA에서 3.7%의 성장을 보여줍니다. 기존 MLLMs에 비해 뛰어난 성능을 발휘하며 dense video events에 대한 질문-답변을 효과적으로 처리할 수 있습니다.



### Empirical Bayesian image restoration by Langevin sampling with a denoising diffusion implicit prior (https://arxiv.org/abs/2409.04384)
Comments:
          24 pages

- **What's New**: 본 논문은 사전 훈련된 Denoising Diffusion Probabilistic Models (DDPM)을 통합한 Bayesian Langevin 알고리즘을 제안하여 이미지 복원에서의 효율성을 크게 향상시킵니다.

- **Technical Details**: 이 방법은 원래의 likelihood function을 직접 사용할 수 있도록 하여, 복잡한 likelihood 문제를 회피하면서 가능성 추정에 대한 empirical Bayesian 기술을 활용합니다. 이를 통해 이미지 블러링(deblurring), 초해상도(super-resolution), 및 인페인팅(inpainting)과 같은 다양한 작업에서의 모델 하이퍼파라미터를 자동으로 조정합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 기준 작업에서 state-of-the-art 방법들과 비교하여 이미지 추정 정확도 및 컴퓨팅 시간 모두에서 우수한 성능을 보여줍니다.



### Enhancing Skin Lesion Diagnosis with Ensemble Learning (https://arxiv.org/abs/2409.04381)
- **What's New**: 이번 연구에서는 HAM10000 데이터셋을 활용하여 피부 병변의 진단을 보조하는 심층 학습 방법을 구현하였습니다.

- **Technical Details**: 연구에서는 MobileNetV2, ResNet18, VGG11과 같은 세 가지 사전 훈련된 모델을 평가하였으며, 각각 0.798, 0.802, 0.805의 정확도를 달성했습니다. 또한, 최대 투표(max voting), 평균 투표(average voting), 스태킹(stacking) 기법을 사용하여 앙상블 모델을 개발하여 0.803, 0.82, 0.83의 정확도를 얻었습니다. 최종적으로 스태킹 기법을 기반으로 한 SkinNet 모델을 개발하여 0.867의 정확도와 0.96의 AUC를 기록하였습니다.

- **Performance Highlights**: SkinNet 모델은 개별 모델들에 비해 큰 성능 향상을 보여주며, 피부 병변 분류에서 앙상블 학습의 효과를 입증합니다.



### RCNet: Deep Recurrent Collaborative Network for Multi-View Low-Light Image Enhancemen (https://arxiv.org/abs/2409.04363)
Comments:
          14 Pages, 10 Figures, Under Review

- **What's New**: 이번 연구에서는 다중 뷰 저조도 이미지 향상을 위한 새로운 접근 방식을 제시합니다. 이를 위해 '다중 뷰 저조도 삼중체(Multi-View Low-light Triplets, MVLT)'라는 새로운 데이터셋을 구축하고, 재귀적 협력 네트워크(Recurrent Collaborative Network, RCNet)를 기반으로 한 심층 다중 뷰 향상 프레임워크를 개발하였습니다.

- **Technical Details**: 데이터셋 MVLT는 1,860 쌍의 삼중 이미지로 구성되어 있으며, 각 삼중체는 같은 장면을 향한 세 가지 관점에서 촬영된 이미지를 포함하고 있습니다. 제안된 RCNet 프레임워크는 내부 뷰 기능 향상(Intra-view EN)과 외부 뷰 기능 정렬 및 융합(Inter-view AF)을 통해 내부 및 외부 뷰 기능 전파를 단계적으로 모델링합니다. ReEAF 모듈을 설계하여 텍스처의 유사성 기반 기능을 활용합니다.

- **Performance Highlights**: 실험 결과, RCNet은 기존의 최첨단 방법들보다 현저하게 우수한 성능을 보이며, 다양한 조명 변화와 노이즈 분포에 효과적으로 대응합니다. 이 연구는 저조도 환경에서 다중 관점의 협동 복원 가능성을 입증하였습니다.



### Connectivity-Inspired Network for Context-Aware Recognition (https://arxiv.org/abs/2409.04360)
Comments:
          ECCV 2024 - HCV Workshop, Accepted for presentation, Submitted Manuscript Version (adapted to include author names, Acknowledgements, and reference DOIs): the version of the manuscript improved after peer review will appear in the Proceedings later

- **What's New**: 이 논문은 인간의 시각 시스템에 대한 포괄적인 문헌 리뷰를 통해 AI 실무자에게 새로운 정보를 제공하고, 생물학적으로 영감을 받은 이미지 분류를 위한 신경망을 제안하며, 맥락 인식을 모델링하기 위한 새로운 플러그 앤 플레이 모듈을 제시하는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안한 신경망은 인간의 피질 및 피질 아래의 연결성에 영감을 받아 설계되었으며, 하향(top-down) 및 상승(bottom-up) 변조를 구현하여 시각 및 인지 영역 간의 복잡한 연결을 모방합니다. 우리의 Contextual Attention Block(CAB)은 매우 간단하며 모든 피드포워드 신경망에 통합될 수 있습니다.

- **Performance Highlights**: 이미지 분류 실험을 통해 제안된 방법이 성능과 class activation을 통한 설명의 강인성이 일관성 있게 개선되었음을 확인했습니다.



### Serp-Mamba: Advancing High-Resolution Retinal Vessel Segmentation with Selective State-Space Mod (https://arxiv.org/abs/2409.04356)
- **What's New**: 제안된 Serpentine Mamba (Serp-Mamba) 네트워크는 초광대역 스캐닝 레이저 안저 촬영(UWF-SLO) 이미지에서 혈관의 정밀 세분화를 위하여 개발되었습니다. 이 네트워크는 혈관 구조의 연속성을 포착하기 위해 설계된 최초의 모델로, 곡선 혈관 구조에 적합한 스캐닝 기법과 클래스 불균형 문제를 해결하기 위한 모듈을 포함합니다.

- **Technical Details**: 주요 기술적 특징으로는 Serpentine Interwoven Adaptive (SIA) 스캔 메커니즘과 Ambiguity-Driven Dual Recalibration (ADDR) 모듈이 있습니다. SIA는 혈관 구조를 따라 뱀처럼 기어가며 스캔을 수행하여 혈관 연속성을 보장합니다. ADDR 모듈은 두 개의 학습 가능한 임계값을 사용하여 픽셀을 구분하고, 혼란스러운 픽셀을 재조정하여 낮은 클래스 간 균형을 해소합니다.

- **Performance Highlights**: 세 가지 데이터셋에서의 실험 결과는 Serp-Mamba가 기존의 최첨단 방법들보다 우수한 성능을 보인다는 것을 보여줍니다. 또한, 아블레이션 실험을 통해 제안된 디자인의 효과성을 검증하였습니다.



### Computer-Generated Sand Mixtures and Sand-based Images (https://arxiv.org/abs/2409.04345)
Comments:
          12 pages, 8 figures, 2nd International Research Conference on Computer Engineering and Technology Education

- **What's New**: 이 논문은 모래 혼합물의 컴퓨터 생성 이미지를 만들기 위한 제안된 알고리즘의 소프트웨어 구현의 효과를 검증하는 것을 목표로 하고 있습니다.

- **Technical Details**: 논문의 방법론은 실제 혼합물의 사진 이미지와 컴퓨터 생성 이미지를 시각적으로 비교하여 혼합물 생성의 결과가 예상대로인지를 확인하고, 컴퓨터 생성된 모래 기반 이미지가 소스 이미지와 동일한 내용을 유지하는지를 검증하는 것입니다.

- **Performance Highlights**: 혼합물 비교 결과, 실제 이미지와 컴퓨터 생성된 이미지가 전체적으로 유사한 음영과 색상을 가지고 있지만, 생성된 이미지가 더 거친 질감과 높은 대비를 보였습니다. 소스 이미지와 모래 기반 이미지의 비교에서도 소프트웨어의 변환 과정에서 본질적인 내용을 유지함을 보여주었습니다.



### How to Identify Good Superpixels for Deforestation Detection on Tropical Rainforests (https://arxiv.org/abs/2409.04330)
Comments:
          8 pages, 3 figures, paper accepted for publication at the IEEE GRSL

- **What's New**: 이 연구는 초픽셀(segmentation) 방법을 통해 아마존 열대 우림에서의 산림 파괴 감지 시스템을 지원하는 데 중점을 두고 있습니다. 특히, 16개의 최신 초픽셀 방법을 비교 분석하고, 각각의 방법의 성능을 평가하여 최적의 성능을 보여주는 방법을 식별하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구에서는 SLIC, GMMSP, ERS 등의 다양한 초픽셀 방법을 사용하여 위성 이미지에서 초픽셀 분할을 수행합니다. 이 방법들은 지역의 경계(예: Boundary Recall, Undersegmentation Error 등)를 정확히 포착하면서 정보 중복을 최소화하는 데 효과적입니다. 또한 RGB, PCA, UMDA 등의 세 가지 기본 채널을 활용하여 비교 평가를 진행했습니다.

- **Performance Highlights**: 실험 결과, ERS, GMMSP, DISF 방법이 각각 UE, BR, SIRS에서 최고의 성능을 보였으며, SH, DISF, ISF가 RGB, UMDA, PCA 조합에서 가장 우수한 분류 성능을 나타냈습니다. 전반적으로, 초픽셀 방법은 경계(oject delineation), 동질성(homogeneity), 조밀성(compactness), 정규성(regularity) 측면에서 뛰어난 trade-off를 제공하여 산림 파괴 감지 작업에 더 적합하다는 결론을 도출했습니다.



### Advancing SEM Based Nano-Scale Defect Analysis in Semiconductor Manufacturing for Advanced IC Nodes (https://arxiv.org/abs/2409.04310)
Comments:
          Accepted in ECCV 2024 2nd workshop on Vision-based InduStrial InspectiON (VISION)

- **What's New**: 이 연구에서는 반도체 결함을 분류, 감지 및 세분화하기 위한 통합된 엔드투엔드 자동 결함 분류-감지-세분화(ADCDS) 프레임워크를 소개합니다. 이 프레임워크는 결함 감지 모듈과 결함 세분화 모듈 두 가지로 구성되어 있습니다. 이를 통해 기존의 수동적인 주석 작업을 줄이고 결함 감지 효율성을 크게 향상시키고자 합니다.

- **Technical Details**: ADCDS 프레임워크는 가변형 DETR(Deformable DETR) 아키텍처를 결함 감지 모듈로 사용하여 나노 스케일 결함의 분류 및 감지를 지원합니다. 세분화 모듈은 BoxSnake를 활용하여 결함의 박스 감독(instance segmentation)을 수행하며, 이는 수동으로 미세한 마스크 주석 작업을 피할 수 있게 합니다.

- **Performance Highlights**: 제안된 ADCDS 프레임워크는 ADI SEM 데이터 세트에서 감지에 대해 72.19 mAP@IoU0.5, 세분화에 대해 78.86을 기록했습니다. AEI 데이터 세트에서는 감지에 대해 90.38, 세분화에 대해 95.48의 성과를 보였습니다. 이러한 결과는 프레임워크가 고급 결함 분석의 요구를 효과적으로 충족함을 나타냅니다.



### FS-MedSAM2: Exploring the Potential of SAM2 for Few-Shot Medical Image Segmentation without Fine-tuning (https://arxiv.org/abs/2409.04298)
Comments:
          13 pages, 4 figures

- **What's New**: FS-MedSAM2는 Segmentation Anything Model 2 (SAM2)를 기반으로 하여, 최소한의 지원 이미지로 의료 이미지를 세분화할 수 있는 새로운 방법론입니다. 이 방법은 모델의 미세 조정 없이도 매우 효과적입니다.

- **Technical Details**: FS-MedSAM2는 의료 이미지 세분화를 위해 SAM2의 메모리 주의 모듈과 마스크 프롬프트 처리 능력을 활용합니다. 주어진 지원 이미지로부터 메모리 인코더를 통해 지원 메모리를 생성하고 이를 메모리 뱅크에 저장하여, 쿼리 이미지의 인접한 슬라이스 정보를 활용하여 세분화 성능을 개선합니다.

- **Performance Highlights**: FS-MedSAM2는 CHAOS-MRI 및 Synapse-CT라는 두 개의 공개 의료 이미지 데이터 세트에서 최신의 성능(State-of-the-Art)을 기록했습니다. 이러한 결과는 FS-MedSAM2가 의료 이미징 도메인에서의 기존 기법들보다 우수함을 입증합니다.



### Cycle Pixel Difference Network for Crisp Edge Detection (https://arxiv.org/abs/2409.04272)
- **What's New**: 이번 논문은 순수하게 처음부터 학습 가능한 새로운 엣지 감지 네트워크인 CPD-Net을 제안합니다. 이 네트워크는 기존의 대규모 사전 훈련된 모델에 대한 의존도를 줄이고 정밀하고 깨끗한 엣지 맵을 생성하는 데 중점을 둡니다.

- **Technical Details**: CPD-Net은 사이클 픽셀 차이 컨볼루션(Cycle Pixel Difference Convolution, CPDC)과 다중 스케일 정보 강화 모듈(Multi-Scale Information Enhancement Module, MSEM), 이중 잔차 연결 기반 디코더(Dual Residual Connection-based Decoder)로 구성된 비대칭 U자형 아키텍처입니다. CPDC는 이미지 엣지 특징을 효과적으로 인코딩하며, MSEM은 모델의 판별 능력을 향상시킵니다.

- **Performance Highlights**: 우리의 방법은 BSDS500(ODS=0.813), NYUD-V2(ODS=0.760), BIPED(ODS=0.898) 데이터셋에서 경쟁력 있는 성능을 달성했습니다. 이는 대규모 사전 훈련 없이도 우수한 결과를 보여주는 중요한 결과입니다.



### Hybrid Cost Volume for Memory-Efficient Optical Flow (https://arxiv.org/abs/2409.04243)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 메모리 효율적(optical flow) 광학 흐름 추정을 위한 새로운 Hybrid Cost Volume(HCV)을 제안합니다. 기존의 방대한 4D 비용 볼륨 구조로 인해 발생하는 메모리 사용량 문제를 해결하여, 고해상도 이미지에서도 높은 정확도를 유지할 수 있는 방법을 제시합니다.

- **Technical Details**: HCV는 글로벌 3D 비용 볼륨을 두 개 생성하고, 이후에 지역 4D 비용 볼륨을 추가하는 방식으로 구성됩니다. Top-k 전략을 적용하여 관련성이 높은 k개의 포지션만 유지하여 메모리 사용량을 줄입니다. 이러한 접근법은 두 개의 3D 볼륨과 지역 4D 볼륨을 통합하여 메모리 효율성을 극대화하며, 정밀한 모션 정보를 포착할 수 있습니다. 최종적인 HCV는 O(H×W×(D+D)×K) 복잡도를 가지며, K는 일반적으로 8로 설정합니다.

- **Performance Highlights**: HCVFlow로 알려진 이 신경망은 KITTI와 Sintel 데이터셋에서 이전의 메모리 효율 방법인 Flow1D에 비해 정확도가 16% 이상 향상되었습니다. RAFT와 유사한 정확도를 달성하면서도 메모리 사용량은 1/8 수준에 불과합니다. 따라서 HCVFlow는 고해상도 이미지에 적합한 효율적인 광학 흐름 추정 방법으로 자리잡을 가능성이 큽니다.



### UniDet3D: Multi-dataset Indoor 3D Object Detection (https://arxiv.org/abs/2409.04234)
- **What's New**: 본 논문에서는 실내 데이터셋의 혼합을 통해 훈련된 혁신적인 3D 객체 탐지 모델인 \ours{}를 제안합니다. 이 모델은 다양한 실내 환경에서 효과적으로 작동하며, 서로 다른 레이블 공간을 통합하여 여러 데이터셋에서 강력한 표현력을 학습할 수 있습니다.

- **Technical Details**: \ours{} 모델은 베이직(transformer 기반) 인코더 아키텍처에 구축되어 있으며, 포지셔널 인코딩(positional encoding) 및 교차 주목(cross-attention) 없이 운용됩니다. 혼합된 데이터셋에서 공동 훈련(joint training)을 통해 데이터셋 간 격차를 줄이고 더 높은 정확도를 달성합니다.

- **Performance Highlights**: 종합 실험을 통해 \ours{}가 6개의 실내 벤치마크에서 기존 3D 객체 탐지 방법에 비해 유의미한 성능 향상을 보여줍니다. 이를 통해 ScanNet(+1.1 mAP50), ARKitScenes(+19.4 mAP25), S3DIS(+9.1 mAP50) 등에서 뛰어난 결과를 기록합니다.



### MpoxMamba: A Grouped Mamba-based Lightweight Hybrid Network for Mpox Detection (https://arxiv.org/abs/2409.04218)
- **What's New**: 이 논문에서는 Deep Learning 기반의 경량 하이브리드 아키텍처인 MpoxMamba를 제안하였습니다. MpoxMamba는 mpox 피부 병변에서의 지역 특징 추출을 위해 깊이 분리 가능한 합성곱을 사용하고, 그룹화된 Mamba 모듈을 통해 전역 맥락 정보를 모델링하는 능력을 크게 향상시킵니다.

- **Technical Details**: MpoxMamba는 0.77M의 파라미터 크기와 0.53G의 FLOPs를 가지며, GMLGFF(Groupped Mamba-based Local-Global Feature Fusion Block)을 통해 긴 거리 의존성(long-range dependencies)을 효과적으로 모델링합니다. 이 모델은 전체적인 합성곱 층과 역 잔여(Inverted Residual Block)를 이용하여 초급 특징을 추출한 후, 다양한 GMLGFF 블록을 통해 지역적인 특징과 전역 맥락 정보를 종합적으로 추출합니다.

- **Performance Highlights**: 두 가지 널리 인정받는 mpox 데이터셋에서 실험 결과 MpoxMamba는 기존의 mpox 탐지 방법과 최신 경량 모델보다 더 뛰어난 성능을 보였습니다. 실제로, 우리는 공공 Epidemic 지역에서 무료 mpox 탐지 서비스를 제공하는 웹 기반 온라인 애플리케이션을 개발하였습니다.



### Diagram Formalization Enhanced Multi-Modal Geometry Problem Solver (https://arxiv.org/abs/2409.04214)
- **What's New**: 이번 연구에서는 Diagram Formalization Enhanced Geometry Problem Solver (DFE-GPS)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 시각적 특징, 기하학적 형식 언어(geometric formal language), 자연어 표현(natural language representations)을 통합하여 기하학 문제 해결 능력을 향상시킵니다.

- **Technical Details**: DFE-GPS는 Diagram Formalizer라는 모델을 포함하며, 이 모델은 기하학적 다이어그램의 형식 언어를 활용하여 모델의 시각적 구성 요소를 개선하고 LLM의 기하학적 구조 인식을 강화합니다. 이 연구를 통해 SynthGeo228K라는 대규모 기하학 데이터셋을 생성하였으며, 이는 기하학적 형식 언어와 자연어 캡션으로 주석이 달려 있습니다.

- **Performance Highlights**: DFE-GPS는 formalgeo7K 테스트 세트에서 기하학 문제 해결 능력을 유의미하게 향상시켰으며, 다중 선택 질문에서 더 도전적인 개방형 질문 답변으로 문제 해결 범위를 확장하였습니다. 또한, GPT-4o-mini를 사용하여 단계적인 문제 해결 과정을 평가하는 Process Evaluation Score를 도입하였습니다.



### Learning to Learn Transferable Generative Attack for Person Re-Identification (https://arxiv.org/abs/2409.04208)
- **What's New**: 이번 논문에서는 Meta Transferable Generative Attack (MTGA) 방법을 제안하여, 다양한 도메인에서 훈련된 모델을 시험하는데 필요한 강인성을 평가하는 새로운 접근 방식을 제공합니다. 특히 모델의 교차 전이(transferability) 능력을 강화하여 공격의 효과성을 높입니다.

- **Technical Details**: MTGA 방법은 메타 학습(meta-learning) 최적화를 통해 생성적 공격자가 고도로 전이 가능한 적대적(adversarial) 예제를 생성하도록 돕습니다. Perturbation Random Erasing 모듈은 모델 특정 특성을 손상시키는 것을 방지하고, Normalization Mix 전략은 다양한 도메인 통계를 혼합하여 크로스 테스트(cross-test) 공격을 모의합니다.

- **Performance Highlights**: MTGA는 다양한 테스트에서 기존 최첨단(SOTA) 방법들보다 평균 mAP(multi-Attribute Prediction) 감소율에서 21.5% 및 11.3% 개선된 성능을 보여주었습니다. 이는 MTGA가 크로스 모델 및 데이터 세트 공격에서 뛰어난 전이 가능성을 가지고 있음을 입증합니다.



### Introducing Gating and Context into Temporal Action Detection (https://arxiv.org/abs/2409.04205)
Comments:
          Accepted for publication at the ECCV 2024 ABAW Workshop

- **What's New**: Temporal Action Detection (TAD)에서의 새로운 접근 방식으로, 우리는 가벼운 연산을 통한 정제된 특징 추출 프로세스를 제안합니다. 요청된 방법은 두 개의 지점에서 개선점을 제공합니다: 지역 특징과 맥락적 이해.

- **Technical Details**: 제안된 방법은 병렬 convolution을 활용한 지역(branch)와 경계 프레임을 키-값 쌍으로 활용하여 중심 프레임과 분석하는 맥락(branch)을 포함합니다. 지역 브랜치는 다양한 윈도우 크기로 미세한 및 거친 시간적 특징을 포착하고, 게이팅 메커니즘을 도입해 가장 관련성 높은 특징을 선택합니다.

- **Performance Highlights**: 실험 결과, THUMOS14 및 EPIC-KITCHEN 100 데이터셋에서 기존 방법들보다 일관된 성능 향상을 보였습니다. 이로 인해 제안된 방법이 효과적임을 입증했습니다.



### GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers (https://arxiv.org/abs/2409.04196)
Comments:
          preprint

- **What's New**: 이 논문에서는 GST(Gaussian Splatting Transformer)라는 새로운 방법을 제안합니다. 이 방법은 단일 이미지에서 3D 인간 모델을 빠르게 추론할 수 있는 능력을 가지고 있으며, 기존의 확산 모델(diffusion models)이나 3D 포인트 지도(supervision)를 필요로 하지 않습니다. GST는 다중 시점(multi-view) 감독 학습을 통해 정확한 3D 포즈 추정(pose estimation)과 높은 시각적 품질을 달성합니다.

- **Technical Details**: GST는 3D Gaussian Splatting(3DGS)을 사용하여 장면을 표현합니다. 이 방법은 SMPL(Statistical Human Mesh Model) 메쉬의 정점(vertices)을 기반으로 하여 Gaussian의 초기 위치를 예측합니다. 이후 transformer 모델을 훈련시켜 이러한 위치의 작은 조정과 다른 Gaussian 속성(attributes), SMPL 파라미터를 동시에 예측합니다.

- **Performance Highlights**: GST 방식은 실험적으로 단일 이미지에서 3D 인간 모델의 빠른 추론을 가능하게 하고, 기존 방식보다 더 나은 3D 포즈 추정 및 가시적 품질을 제공합니다. 실시간(real-time) 배치와 다양한 의상 및 자세(pose)에 대한 유연성을 유지하면서도 정확한 3D 모델을 생성할 수 있습니다.



### LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration (https://arxiv.org/abs/2409.04187)
Comments:
          15 pages, 6 figures, to be published in ICONIP-2024

- **What's New**: 본 논문에서는 경량 통합 추적-특징 추출(LITE) 패러다임을 소개하여 다중 객체 추적(MOT) 접근 방식을 혁신합니다. LITE는 ReID 기반의 트래커에서 추론, 전처리, 후처리 및 ReID 모델 교육 비용을 제거하여 성능을 향상시킵니다. 또한, LITE는 실시간 appearance feature를 사용하여 속도를 저하시키지 않습니다.

- **Technical Details**: LITE는 YOLOv8m과 같은 표준 CNN 기반 감지기를 사용하여 추적 파이프라인에 appearance feature 추출을 통합하여 구현됩니다. LITE:DeepSORT의 가장 간단한 구현은 MOT17 벤치마크에서 HOTA 점수 43.03%를 28.3 FPS로 달성하며, DeepSORT의 속도를 두 배로 증가시켰고 MOT20 데이터셋에서는 네 배 빨라졌으며 유사한 정확도를 유지합니다.

- **Performance Highlights**: LITE는 기존 DeepSORT와 StrongSORT보다 성능 향상과 속도의 이점을 제공합니다. LITE 기반의 추적기는 실시간 애플리케이션에 적합하며, Tracker의 실행 속도 및 HOTA 점수를 종합적으로 평가할 수 있는 새로운 평가 프레임워크를 제안합니다.



### Reprojection Errors as Prompts for Efficient Scene Coordinate Regression (https://arxiv.org/abs/2409.04178)
Comments:
          ECCV2024

- **What's New**: 이번 연구에서는 동적인 객체와 질감이 없는 영역이 카메라 포즈 추정에 미치는 부정적인 영향을 분석하고, 이를 개선하기 위해 Error-Guided Feature Selection (EGFS) 메커니즘을 도입하였습니다. 이 연구는 Segment Anything Model (SAM)을 활용하여 문제 있는 영역을 샘플링하고 필터링하는 방법을 제안합니다.

- **Technical Details**: 저자들은 UGFS를 이용하여 낮은 재투영(reprojection) 오류를 가진 영역을 선택하고 이를 통해 에러-가이드 마스크를 생성합니다. 이 마스크는 반복적으로 업데이트되어 카메라 포즈 추정을 보다 신뢰할 수 있게 개선합니다. 이를 위해 RGB 이미지의 2D-3D 대응관계를 활용하고, PnP와 RANSAC을 사용하여 카메라 포즈를 추정합니다. 또한, SAM을 통해 시각적 정보를 이용한 동적인 마스크 생성을 진행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Cambridge Landmarks와 Indoor6 데이터셋에서 기존 SCR 방법보다 높은 성능을 발휘하며, 훈련 시간과 모델 크기도 줄이는 이점을 보였습니다. SOTA(state-of-the-art) 성능을 달성하여 다양한 시각적 로컬라이제이션 응용에서 활용 가능성을 보여줍니다.



### Secure Traffic Sign Recognition: An Attention-Enabled Universal Image Inpainting Mechanism against Light Patch Attacks (https://arxiv.org/abs/2409.04133)
- **What's New**: 새로운 논문에서는 교통 표지 인식(traffic sign recognition, TSR) 시스템이 깊은 학습(deep learning) 기술에 크게 의존하고 있다는 점을 강조하며, 악의적인 공격(adversarial attacks)으로부터의 취약성을 다루고 있습니다. 특히, 연구자들은 교통 표지에 악의적인 빛 패치를 투사하여 시스템을 속이는 새로운 공격 방식인 비접촉(light patches) 공격을 소개합니다.

- **Technical Details**: 이 논문에서는 SafeSign이라는 범용 이미지 인페인팅(image inpainting) 메커니즘을 제안하여 악의적인 빛 패치에 오염된 교통 표지를 복구하는 방안을 제공합니다. SafeSign은 다중 시점 이미지 융합(multi-view image fusion) 과 주의(attachment) 메커니즘을 활용하여 교통 표지 복구를 수행합니다. 또한, U-Net을 기반으로 한 이진 마스크 검출(Binary Mask-based U-Net)로 다양한 오염된 표지 패턴을 생성합니다.

- **Performance Highlights**: SafeSign은 표지 인식 모델에 대한 평균 정확도를 54.8% 향상시켰고, LeNet, GoogleNet, YOLO5와 같은 일반적인 신호 인식 모델들에서의 효과를 입증했습니다. 이 모델은 하드웨어 변경 없이도 기존 모델을 훈련시키지 않고도 다양한 악의적 공격을 효과적으로 저지할 수 있는 장점을 가지고 있습니다.



### Confidence-Aware Document OCR Error Detection (https://arxiv.org/abs/2409.04117)
- **What's New**: 본 연구에서는 OCR(Optical Character Recognition)의 신뢰도 점수(confidence scores)를 활용하여 OCR 오류 감지를 향상시키기 위한 새로운 모델, ConfBERT를 제안합니다. 이 모델은 OCR 시스템 간의 신뢰도 점수와 오류율 간의 상관관계를 분석하고, 이러한 점수를 통합하여 오류 감지 능력을 개선할 수 있음을 보여줍니다.

- **Technical Details**: ConfBERT 모델은 BERT 기반으로, OCR 신뢰도 점수를 토큰 임베딩(token embeddings)에 통합하며, 노이즈 조정을 위한 선택적 프리트레이닝(pre-training) 단계를 제공합니다. 우리는 상업적 및 오픈소스 OCR 솔루션의 성능 및 신뢰도 점수를 비교 분석하고, 이러한 점수를 사용하여 포스트-OCR 오류 감지를 개선하는 방법을 제안합니다.

- **Performance Highlights**: 우리의 실험 결과는 OCR 신뢰도 점수를 통합함으로써 오류 감지 능력이 향상될 수 있음을 나타냅니다. 또한, 상업적 OCR 기술과 오픈소스 OCR 기술 간 성능의 상당한 차이를 강조합니다.



### Smooth-edged Perturbations Improve Perturbation-based Image Explanations (https://arxiv.org/abs/2409.04116)
Comments:
          This manuscript have been submitted to NLDL 2025

- **What's New**: 이 논문에서는 Randomized Input Sampling for Explanations (RISE) 방법의 성공 요인이 무엇인지에 대해 밝히고자 다양한 샘플링, 세분화 기법, 부드러운 변화 및 기여도 계산 기법의 조합을 테스트했습니다. RISE 방법이 대중적이었지만 어떤 요소가 성과에 기여했는지 명확히 밝혀진 적이 없었습니다.

- **Technical Details**: RISE 스타일의 픽셀 기여도는 평가된 모든 방법에 유익하며, 기여도 계산은 성과에 가장 영향을 덜 미치는 매개변수임을 보여주었습니다. 실험은 ImageNet 검증 세트를 기반으로 실시되었으며, 다양한 세분화(Segmentation), 샘플링(Sampling), 변화(Perturbation), 기여도(Attribution) 방법을 결합하여 수행되었습니다. 특히 부드러운 가장자리가 있는 변화가 다른 변동 기반 파이프라인의 성능을 개선하는 데 도움이 된다는 것을 발견하였습니다.

- **Performance Highlights**: 평가 결과, RISE 방법의 개선된 기법을 사용하여 부드러운 픽셀 변화와 픽셀의 흐림 정도에 따라 가중치를 부여하는 방식이 모든 평가된 방법의 성능을 향상시켰습니다. 또한, 기여도 계산 방식은 성과에 미치는 영향이 적은 반면, 샘플링 기법, 샘플 수, 세분화 기법, 픽셀 당 기여도는 성능에 더 큰 영향을 미친다는 것을 확인하였습니다.



### UNIT: Unifying Image and Text Recognition in One Vision Encoder (https://arxiv.org/abs/2409.04095)
- **What's New**: 이 논문에서는 UNIT라는 새로운 훈련 프레임워크를 제안하여 이미지 및 텍스트 인식을 단일 모델에서 통합합니다.

- **Technical Details**: UNIT는 먼저 사전 훈련된 Vision Transformer (ViT) 모델을 기반으로 경량의 언어 디코더 및 경량 비전 디코더를 추가하여, 텍스트 출력을 예측하고 원래 이미지 인코딩 기능의 재앙적 망각을 방지합니다. 훈련 과정은 내부 스케일 사전 훈련과 외부 스케일 미세 조정의 두 단계로 이루어집니다.

- **Performance Highlights**: 여러 벤치마크에서 UNIT는 OCR (Optical Character Recognition) 및 문서 질의 응답 (DocQA) 작업을 포함한 문서 관련 작업에서 기존 방법들을 크게 초월하며, 자연 이미지에 대한 성능을 유지합니다.



### Introducing a Class-Aware Metric for Monocular Depth Estimation: An Automotive Perspectiv (https://arxiv.org/abs/2409.04086)
Comments:
          Accepted at the European Conference on Computer Vision (ECCV) 2024 Workshop on Out Of Distribution Generalization in Computer Vision

- **What's New**: 이 논문은 자동화된 차량 애플리케이션에서의 안전 비판적 요구사항을 충족하기 위해 설계된 새로운 다중 구성 요소 깊이 평가 메트릭을 제안합니다. 이 메트릭은 클래스 기반 거리, 로컬 특징 분석, 글로벌 깊이 일관성 유지 등 세 가지 구성 요소를 포함합니다.

- **Technical Details**: 제안된 메트릭은 클래스 기반 거리 측정, 엣지 및 코너 이미지 특징 분석, 글로벌 일관성 유지를 통합하여 깊이 추정 모델의 성능을 평가합니다. 이 메트릭은 각 클래스의 중요성을 평가하기 위해 실제 사고 데이터에서 유도된 안전 비판적 클래스에 기초한 가중치를 부여합니다.

- **Performance Highlights**: 이 연구는 새로운 메트릭을 기존의 전통적인 메트릭과 비교하고, 클래스별 분석 및 안전 비판적 상황을 기반으로 한 평가를 통해 깊이 추정 모델의 성능에 대한 더 깊은 통찰력을 제공합니다. 제안된 메트릭은 안전 비판적 요구사항을 충족하면서 모델 결과에 대한 보다 실질적인 통찰력을 제공합니다.



### SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation (https://arxiv.org/abs/2409.04082)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 이벤트 카메라를 위한 고속의 강인한 optical flow 추정을 위한 STTFlowNet 및 SDformerFlow라는 두 가지 솔루션을 제안합니다. 기존의 인공지능 신경망(ANN) 구조와 함께 spiking neural networks (SNNs)를 통합하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: STTFlowNet은 spatiotemporal shifted window self-attention (swin) transformer 인코더를 통한 U자 구조의 ANN 아키텍처를 채택합니다. SDformerFlow는 swin spikeformer 인코더를 통합한 완전한 스파이킹 모델이며, 두 가지 서로 다른 뉴런 모델을 갖춘 스파이킹 버전의 변형도 제공합니다.

- **Performance Highlights**: 우리의 결과는 DSEC 및 MVSEC 데이터셋에서 SNN 기반 이벤트 optical flow 방법들 중에서 최신 성능을 발휘하며, 동등한 ANN에 비해 전력 소비가 현저하게 줄어드는 것을 보여줍니다.



### Site-Specific Color Features of Green Coffee Beans (https://arxiv.org/abs/2409.04068)
Comments:
          21 pages, 7 figures

- **What's New**: 본 논문에서는 커피가 중요한 원자재임에도 불구하고 전통적인 녹색 커피콩 선택 방법이 인력의 시각적 검사에 의존하고 있다는 점을 지적하고, 이를 개선하기 위한 새로운 방법론을 제시합니다.

- **Technical Details**: 사이트에 독립적인 접근법을 사용하여 자격이 있는 녹색 커피콩의 씨앗 껍질에서 사이트별 색상 특징을 찾아내고, 이를 기반으로 두 가지 평가 체계를 제안합니다. 이 방법은 머신 러닝(ML) 분류기를 활용하여 색상 특징이 지역별로 다름을 밝혀냅니다.

- **Performance Highlights**: 제안된 평가 체계는 기존 방법에 비해 간단하며, 계산 비용이 적고 보편적인 적용 가능성을 지니고 있습니다. 또한, 이 색상 특징은 다양한 재배 지역의 자격을 갖춘 커피콩을 구별할 수 있으며, 커피 비즈니스에서의 사기 방지 기능을 가지고 있습니다.



### D4: Text-guided diffusion model-based domain adaptive data augmentation for vineyard shoot detection (https://arxiv.org/abs/2409.04060)
- **What's New**: 농업 분야에서 객체 감지 모델을 활용한 식물 표현 형상 분석(phenotyping) 기술에 관한 새로운 접근법이 제안되었습니다. 이 연구에서는 D4라는 생성적 데이터 증강 방법이 도입되었으며, 이는 적은 수의 주석이 달린 데이터셋과 비디오 데이터에서 추출한 다수의 원본 이미지를 활용하여 효과적으로 학습 데이터를 확장합니다.

- **Technical Details**: 제안된 D4 방법은 사전 학습된 텍스트 유도 확산 모델(text-guided diffusion model)을 기반으로 하여, 정의된 객체 감지를 위한 주석 정보를 유지하면서도 목표 도메인에 적합한 새로운 주석 이미지를 생성합니다. 이 과정에서 다양한 환경의 배경 정보를 반영하여 데이터 주석의 품질을 향상시킵니다. D4는 주석 가능 데이터의 부족과 도메인 다양성의 문제를 극복하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, D4 메소드는 BBox 감지 작업에서 평균 정밀도(mean average precision)를 최대 28.65% 향상시키고, 키포인트 감지 작업의 평균 정밀도를 최대 13.73% 개선하는 것으로 나타났습니다. 이러한 결과는 D4가 실제 농업 분야에서 데이터 생성 비용과 도메인 다양성 문제를 동시에 해결할 수 있는 잠재력이 있음을 보여줍니다.



### COLUMBUS: Evaluating COgnitive Lateral Understanding through Multiple-choice reBUSes (https://arxiv.org/abs/2409.04053)
Comments:
          18 pages, 10 figures, submitted to AAAI-25

- **What's New**: 이 논문은 시각적 질문-답변(Visual Question Answering, VQA) 벤치마크에서 수직적 사고(voertical thinking)에 중점을 두었으며, AI의 측면에서 충분히 연구되지 않은 수평적 사고(lateral thinking)를 도입합니다. 이를 통해 COLUMBUS라는 새로운 합성 벤치마크를 개발하여, 텍스트와 아이콘 레버스 퍼즐을 기반으로 한 QA 세트를 만듭니다.

- **Technical Details**: 제안된 방법론은 여러 선택지를 포함하는 VQA 형식으로 수평적 사고 과제를 생성하는 3단계 분류 기반 접근법입니다. 이 방법론은 퍼즐 요소의 시각적 속성과 관계를 조작하는 18가지 규칙을 정의하며, 퍼즐 해답을 위한 그래프 표현을 생성합니다. COLUMBUS는 1,000개 이상의 퍼즐로 구성되어 있으며, 각 퍼즐은 네 개의 답안 후보를 포함합니다.

- **Performance Highlights**: 최신 비전-언어 모델(State-of-the-art vision-language models, VLMs)은 인간의 성과에 비해 상당한 성능 격차를 보였습니다. VLMs는 인간이 생성한 설명에서 혜택을 받지만, 스스로 그러한 표현을 올바른 추상 수준에서 생성하는 데 어려움을 겪습니다.



### On Evaluation of Vision Datasets and Models using Human Competency Frameworks (https://arxiv.org/abs/2409.04041)
- **What's New**: 이 논문은 컴퓨터 비전 모델과 데이터셋의 평가에서 아이템 반응 이론(Item Response Theory, IRT)을 적용하여 단일 정확도 수치 이상의 심층적인 분석을 제공하는 새로운 접근 방식을 제시합니다. IRT를 통해 모델의 조정(precision) 상태를 평가하고, 데이터 서브셋을 선택하며, 모델과 데이터셋을 비교하는 데 유용한 잠재 파라미터를 사용할 수 있습니다.

- **Technical Details**: 본 연구는 91개의 컴퓨터 비전 모델과 ImageNet 데이터셋을 사용하여 IRT의 잠재 파라미터인 능력(Ability), 난이도(Difficulty), 변별력(Discriminability), 추측(Guessing) 파라미터를 추출했습니다. 특히, 모델의 과잉 확신(overconfidence)을 정의하고, 이러한 수치는 강력한 모델이 잘 조정되어 있다는 것을 보여줍니다. 또한, IRT를 통해 10장의 샘플 이미지로 91개 모델의 성능 차이를 구별할 수 있음을 보였습니다.

- **Performance Highlights**: 실험 결과, 모델의 과잉 확신이 0일 때 높은 정확도를 나타내며, 이는 강력한 모델들이 잘 조정되어 있다는 것을 의미합니다. 또한, 91개의 모델의 성능을 분석하여 Kendall 상관관계가 0.85로 확인되었으며, 이는 선택된 샘플 이미지가 모델 성능 평가에 매우 효과적이라는 것을 나타냅니다.



### PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation (https://arxiv.org/abs/2409.04038)
- **What's New**: 이번 연구는 식물 질병 진단을 위해 새로운 대규모 데이터셋인 PlantSeg를 소개합니다. 이 데이터셋은 11,400장의 질병 분할 마스크 이미지와 8,000장의 건강한 식물 이미지를 포함하고 있으며, 이는 기존 데이터셋과 비교하여 품질 및 응용 가능성을 크게 향상시킵니다.

- **Technical Details**: PlantSeg 데이터셋은 고품질의 분할 마스크를 포함하고 있으며, 야외에서 촬영된 식물 질병 이미지를 주로 구성합니다. 이 데이터셋은 34종의 식물에서 발생하는 115개의 질병을 포함하며, 철저한 검증 과정을 통해 수집된 이미지는 정확성을 보장받았습니다.

- **Performance Highlights**: PlantSeg는 식물 질병 분할을 위한 최첨단 알고리즘 개발 및 평가를 위한 기준점을 제공합니다. 본 데이터셋의 활용으로, 더 정밀한 농업 시스템과 통합 질병 관리(Integrated Disease Management) 접근 방식이 가능해질 것입니다.



### MultiCounter: Multiple Action Agnostic Repetition Counting in Untrimmed Videos (https://arxiv.org/abs/2409.04035)
Comments:
          Accepted by ECAI 2024

- **What's New**: MultiCounter는 여러 인간 인스턴스의 반복 동작을 동시에 감지, 추적 및 계산할 수 있는 완전한 엔드 투 엔드 딥 러닝 프레임워크입니다. 이 프레임워크는 반복 동작을 정확하게 나누고, 사람 중심의 응용 프로그램에서 중요한 MRAC 작업을 위해 처음으로 제안되었습니다.

- **Technical Details**: MultiCounter에는 두 가지 주요 모듈이 포함되어 있습니다: 1) Mixed Spatial-Temporal Interaction (MSTI) 모듈, 2) 두 개의 작업 특정 헤드인 Instance Head와 Period Head. MSTI는 여러 인스턴스의 복잡한 동작을 효과적으로 모델링하며, 두 개의 헤드는 모집단의 인스턴스 및 주기적 특성을 정확하게 예측할 수 있도록 합니다.

- **Performance Highlights**: MultiCounter는 MultiRep 데이터셋에서 실험을 통해 Period-mAP를 41.0% 향상시키고, 평균 MAE(AvgMAE)를 58.6% 감소시키며, AvgOBO를 1.48배 증가시킵니다. 또한, 물리적인 GPU 서버에서 실시간으로 작동하며, 비디오 내 인간 인스턴스의 수에 대해서도 민감하지 않습니다.



### Dense Hand-Object(HO) GraspNet with Full Grasping Taxonomy and Dynamics (https://arxiv.org/abs/2409.04033)
Comments:
          14 pages except for references. It will be published at European Conference on Computer Vision(ECCV) 2024

- **What's New**: 새로운 HOGraspNet 데이터세트는 손-객체 상호 작용을 위한 종합적인 훈련 데이터로, 기존 데이터에는 없는 완전한 grasp taxonomy를 포함하고 있습니다. 이 데이터세트는 99명의 다양한 손 모양에서 수집된 레이블이 있는 3D 손 및 객체 메시, 3D 키포인트, 접촉 맵, grasp 레이블을 제공합니다.

- **Technical Details**: HOGraspNet은 22개의 강체 객체와 크기 및 형태 taxonomy를 기반으로 선택된 8개의 복합 객체를 포함하고 있습니다. 이 데이터세트는 1.5M RGB-Depth sparse frames로 연속적인 비디오 프레임을 제공하며, MANO 모델을 통한 3D 메시 생성 및 HALO를 통한 정확한 적합을 사용합니다.

- **Performance Highlights**: HOGraspNet은 grasp 분류 및 3D 손 자세 추정과 같은 관련 작업에서 평가되었습니다. 결과는 grasp 유형 및 객체 클래스에 따라 성능 변동을 보여주어, 데이터세트가 포착하는 상호 작용 공간의 중요성을 강조합니다.



### BFA-YOLO: Balanced multiscale object detection network for multi-view building facade attachments detection (https://arxiv.org/abs/2409.04025)
Comments:
          22 pages

- **What's New**: 본 논문에서는 다중 관점에서 건물 외관 부착물(doors, windows, balconies 등) 검출을 위한 새로운 모델인 BFA-YOLO를 제안합니다. 이 모델은 특히 기존의 YOLOv8 모델보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: BFA-YOLO 모델은 세 가지 주요 혁신으로 구성됩니다: 1) Feature Balanced Spindle Module (FBSM), 2) Target Dynamic Alignment Task Detection Head (TDATH), 3) Position Memory Enhanced Self-Attention Mechanism (PMESA). 이러한 구성 요소들은 각각 비균형한 객체 분포, 작은 객체 검출의 어려움, 그리고 배경 간섭 문제를 해결하기 위해 설계되었습니다. 또한 BFA-3D라는 새로운 데이터셋을 구축하여 다각적인 시각에서의 정확한 레이블과 다양한 카테고리를 제공합니다.

- **Performance Highlights**: BFA-YOLO는 다중 관점의 BFA-3D와 스트리트 뷰 Facade-WHU 데이터셋에서 각각 1.8% 및 2.9% 향상된 mAP@0.5 성능을 기록하며, 이는 BFA-YOLO의 우수한 성능을 나타냅니다.



### Towards Energy-Efficiency by Navigating the Trilemma of Energy, Latency, and Accuracy (https://arxiv.org/abs/2409.04018)
Comments:
          ISMAR 2024

- **What's New**: 이 논문은 Extended Reality (XR) 기기에서의 에너지 효율성을 높이기 위해 에너지, 지연 (latency), 정확성 (accuracy) 간의 트릴레마를 탐구합니다. 특히, 씬 재구성 (scene reconstruction)의 에너지 최적화를 위해 알고리즘, 실행 (execution), 데이터 (data) 세 가지 최적화 클래스를 제안합니다.

- **Technical Details**: 이 논문에서는 TSDF Fusion (Truncated Signed Distance Function)의 데이터 통합 기법을 중심으로 씬 재구성을 다룹니다. 72개의 디자인을 통해 다양한 에너지 및 지연 간의 트레이드오프를 평가하며, 가장 낮은 에너지 소비를 달성하기 위한 상호 최적화(co-optimization)가 중요함을 강조합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 기본 기준선 대비 에너지 소비를 최대 60배 줄이며 지연 범위를 4배 느려지거나 2배 빨라지는 것을 보여줍니다. 실제 사용 사례 분석에서 25배의 에너지 절약과 함께 1.5배의 지연 감소를 달성하였으며, 재구성 품질 손실은 미미하였습니다.



### 3D-GP-LMVIC: Learning-based Multi-View Image Coding with 3D Gaussian Geometric Priors (https://arxiv.org/abs/2409.04013)
Comments:
          19pages, 8 figures, conference

- **What's New**: 새로운 접근방식인 3D Gaussian 기하학적 priors를 활용한 학습 기반 다중 뷰 이미지 코딩(3D-GP-LMVIC)을 제안합니다. 이 방법은 3D 장면의 기하학적 priors를 유도하여 여러 뷰 간의 정확한 disparity 추정을 가능하게 합니다.

- **Technical Details**: 3D-GP-LMVIC는 3D Gaussian Splatting을 활용하여 내부의 깊이 맵을 압축하고, 뷰 간의 중복 정보를 줄이는 데 중점을 둡니다. 또한 다중 뷰 시퀀스 순서 정렬 방법을 도입하여 인접 뷰 간의 correlation을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 3D-GP-LMVIC는 전통적인 방법과 학습 기반 방법 모두보다 우수한 압축 효율성을 보이면서 빠른 인코딩 및 디코딩 속도를 유지합니다.



### Hybrid Mask Generation for Infrared Small Target Detection with Single-Point Supervision (https://arxiv.org/abs/2409.04011)
Comments:
          9 pages, 5 figures

- **What's New**: 본 논문에서는 Infrared Small Target Detection (IRSTD)에서 단일 포인트 라벨을 통해 고품질의 마스크를 회복하는 Hybrid Mask Generation (HMG) 접근 방식을 소개합니다. 이 방법은 수작업으로 생성한 Points-to-Mask Generation 전략과 pseudo mask 업데이트 전략을 결합하여 초기 pseudo masks를 반복적으로 정제합니다.

- **Technical Details**: 제안하는 방법은 두 단계로 구성됩니다: 첫 번째로, point labels로부터 초기 pseudo masks를 회복하고, 이를 통해 IRSTD 모델을 훈련시킵니다. 두 번째로, 초기 pseudo masks와 모델의 예측 결과를 통합하여 하이브리드 masks를 생성하고, 이를 통해 모델을 재훈련 합니다.

- **Performance Highlights**: 세 개의 SIRST 데이터셋에서 실험 결과, 제안하는 하이브리드 마스크로 훈련된 모델이 단일 포인트 감독 하에서 기존 방법들을 능가하는 성능을 보여주며, 새로운 기준을 설정하였습니다.



### Qihoo-T2X: An Efficiency-Focused Diffusion Transformer via Proxy Tokens for Text-to-Any-Task (https://arxiv.org/abs/2409.04005)
- **What's New**: 새로운 논문에서는 Proxy Token Diffusion Transformer (PT-DiT)를 제안하여, 기존의 global self-attention 메커니즘의 불필요한 계산을 줄이고 효율적으로 전역 시각 정보를 모델링하는 방법을 소개하고 있습니다. PT-DiT는 sparse representative token attention을 사용하여, 각 공간-시간 윈도우에서 무작위로 선택한 토큰을 대표 토큰으로 삼습니다.

- **Technical Details**: PT-DiT는 각 transformer 블록에서 공간-시간 윈도우에서 무작위로 선택된 대표 토큰을 활용하여, global semantics를 캡처하고 cross-attention을 통해 모든 latent 토큰에 전달합니다. 또한, texture modeling 능력을 향상시키기 위해 window attention과 shift-window attention을 도입하며, Swin Transformer와 유사한 디자인을 적용합니다.

- **Performance Highlights**: 실험 결과 PT-DiT는 이미지 생성 작업에서 DiT에 비해 52% 낮은 계산 복잡성을 보였으며, Pixart-α에 비해 65% 감소했습니다. 비디오 생성 작업에서도 CogVideoX에 비해 77.2%의 계산 복잡성을 유지하며, 3D 정보 추출의 효율성을 보여 주었습니다.



### One-Shot Diffusion Mimicker for Handwritten Text Generation (https://arxiv.org/abs/2409.04004)
Comments:
          To appear in ECCV 2024

- **What's New**: 본 논문에서는 단일 샘플만으로도 다양한 필기체 스타일을 모방할 수 있는 One-shot Diffusion Mimicker (One-DM)라는 새로운 모델을 제안합니다. 이 모델은 고주파 정보를 활용하여 스타일 추출을 향상시키고, 노이즈를 억제하는 게이트 메커니즘을 포함합니다. 이를 통해 사용자는 단 하나의 샘플을 가지고도 품질 높은 필기체 이미지를 생성할 수 있습니다.

- **Technical Details**: One-DM은 스타일 참조 이미지와 고주파 정보를 병렬로 처리하는 스타일 향상 모듈을 개발하여 스타일 패턴(예: 문자 기울기, 리가쳐)을 효과적으로 추출합니다. 이 모듈을 통해 배경 노이즈의 영향을 줄이고, 표현력을 높인 스타일 피처를 생성합니다. 스타일 및 텍스트 컨텐츠를 결합하여 조건부 입력으로 사용하여 이미지를 점진적으로 합성합니다.

- **Performance Highlights**: 광범위한 실험을 통해 One-DM은 단일 스타일 참조를 사용하여 영어, 중국어, 일본어 등 여러 언어의 필기 스크립트를 생성할 수 있으며, 15개 이상의 참조 샘플을 사용하는 기존 방법들보다 뛰어난 성능을 보여줍니다.



### DreamForge: Motion-Aware Autoregressive Video Generation for Multi-View Driving Scenes (https://arxiv.org/abs/2409.04003)
Comments:
          Second place solution for W-CODA-Track2

- **What's New**: 이번 논문은 DreamForge라는 새로운 diffusion 기반 비디오 생성 모델을 제안하고 있습니다. 이 모델은 3D 조건에 따라 조절 가능한 비디오 생성 기능을 통해, 운전 장면의 정확한 모델링을 목표로 하고 있습니다.

- **Technical Details**: DreamForge는 텍스트 설명, 카메라 포즈, 3D 바운딩 박스 및 도로 레이아웃과 같은 유연한 조절 조건을 지원하며, 여러 시점 간의 일관성과 시간적 일관성을 유지합니다. 이를 위해 motion cues를 포함한 autoregressive 아키텍처로 설계되었습니다. 또한, 안정적인 diffusion 파이프라인을 기반으로 한 효과적인 조건 인코딩 모듈을 사용합니다.

- **Performance Highlights**: DreamForge는 도로 레이아웃과 3D 바운딩 박스를 활용하여 기하학적으로 정밀한 주행 장면 생성과 함께, 다양한 날씨와 스타일 조정이 가능하여 통합성과 확장성을 향상시켰습니다. 긴 비디오 생성에서도 일관성을 유지하면서 유연한 비디오 길이를 제공합니다.



### Boundary feature fusion network for tooth image segmentation (https://arxiv.org/abs/2409.03982)
Comments:
          MICCAI workshop,see this https URL

- **What's New**: 이 논문에서는 치아 이미지 분할을 위한 혁신적인 경계 특징 융합 네트워크(BFFNet)를 제안하며, 치아와 인접한 조직 간의 경계가 불명확한 문제를 해결하기 위한 경계 정보 통합 방식을 설명합니다. 이를 통해 치아 경계의 세밀한 정보 추출과 정확한 치아 분할을 가능하게 합니다.

- **Technical Details**: BFFNet은 주로 코딩 네트워크(E1-E5), 경계 특징 추출 모듈 및 특징 교차 융합 모듈로 구성됩니다. 이 네트워크는 ResNet 아키텍처를 기반으로 하여 다양한 수준에서 다섯 가지 특징을 추출하고, 이를 활용하여 경계 정보와 글로벌 맵핑 특징을 융합하여 치아 마스크를 생성하는 방법론을 구축합니다.

- **Performance Highlights**: STS 데이터 챌린지에서 실시된 평가에서 우리의 방식은 0.91이라는 점수로 다른 기존 접근 방식에 비해 현저한 우수성을 보였습니다. 이는 이번 연구가 제안한 방법론이 치아 경계 분할에 있어 높은 정확도를 달성했음을 의미합니다.



### Generating Faithful and Salient Text from Multimodal Data (https://arxiv.org/abs/2409.03961)
- **What's New**: 이번 연구에서는 대규모 멀티모달 모델(LMM)이 생성한 텍스트의 신뢰성과 주목성(saliency)을 개선하기 위한 새로운 프레임워크를 제안합니다. 이를 위해 작은 비전 비평가 모델을 훈련하여 이미지 모달리티(image modality)에서 환각(hallucination)된 특징과 비주목성(non-salient) 특징을 식별합니다.

- **Technical Details**: 제안된 프레임워크에서는 비전 비평가 모델이 LMM의 생성된 텍스트의 오류를 감지하고, 누락된 주목성 이미지 특징 목록을 생성합니다. 이 정보는 후처리(post editing) 단계에서 텍스트 품질을 개선하는 데 사용됩니다. 실험에서는 LLaVA-1.5 및 MiniGPT-4와 같은 LMM을 사용하여 실제 부동산 및 전자상거래 데이터 세트에서 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크가 LMM의 생성 품질을 신뢰성 및 주목성 모두에서 향상시켰으며, 환각을 줄이는 최근 기술들을 능가하는 성과를 보여주었습니다.



### FODA-PG for Enhanced Medical Imaging Narrative Generation: Adaptive Differentiation of Normal and Abnormal Attributes (https://arxiv.org/abs/2409.03947)
- **What's New**: 이 논문은 의료 영상 내러티브 생성의 새로운 접근 방식인 FODA-PG(Fine-grained Organ-Disease Adaptive Partitioning Graph)를 제안합니다. 이 프레임워크는 의료 이미지에서의 정상 및 비정상 소견을 구분하여 더 정확한 임상 보고서 생성을 가능하게 합니다.

- **Technical Details**: FODA-PG는 임상적 중요성과 위치에 따라 질병 관련 속성을 '질병 특이적'과 '질병 비특이적'으로 구분하여, 세밀한 그래픽 표현을 통해 의료 영상의 미세한 차이를 포착합니다. 이는 데이터 편향의 영향을 완화하는 데도 기여합니다. 또한, 이 모델은 강력한 transformer 기반 아키텍처에 세분화된 의미적 지식을 통합하여, 높은 일반화 능력을 보입니다.

- **Performance Highlights**: IU-Xray 및 MIMIC-CXR 벤치마크에서 광범위한 실험을 통해 FODA-PG는 최첨단 방법들보다 지속적으로 우수한 성능을 보였으며, 의료 보고서 생성에서의 도메인 적응이 얼마나 중요한지를 강조합니다.



### TropNNC: Structured Neural Network Compression Using Tropical Geometry (https://arxiv.org/abs/2409.03945)
- **What's New**: 이 논문에서는 TropNNC라는 구조적 가지치기(framework) 프레임워크를 제안합니다. 이는 Linear 및 Convolutional layers, ReLU activations를 효율적으로 압축하는 방법입니다. Tropical geometry를 활용하여 더욱 타이트한 약식(bound)을 제공하고, 기존 방법인 Misiakos et al. (2022)의 결과를 확장합니다.

- **Technical Details**: TropNNC는 기하학적 접근 방법을 기반으로 하며, Hausdorff distance를 통해 zonotopes의 표준 연속형태에서 tropical polynomials의 타이트한 약식을 달성합니다. 또한, 유사한 연구에 비해 쉽게 구현할 수 있고, 학습 데이터 샘플의 가용성에 의존하지 않습니다.

- **Performance Highlights**: TropNNC는 MNIST, CIFAR, ImageNet 데이터셋에서 extensive empirical evaluations를 통해 성능을 검증하였으며, ThiNet과 동등한 성능을 기록하고, linear layers 압축에서 특히 뛰어난 성능을 발휘하였습니다. 이는 Tropical geometry를 활용한 최초의 압축 방법으로 평가되고 있습니다.



### HUMOS: Human Motion Model Conditioned on Body Shap (https://arxiv.org/abs/2409.03944)
Comments:
          Accepted in ECCV'24. Project page: this https URL

- **What's New**: 이 논문은 신체 형태에 기반한 생성적 모션 모델인 HUMOS를 제안합니다. HUMOS는 표현된 신체 형태와 성별과 같은 정체성 피쳐에 의존해 사람의 모션을 생성하는 새로운 접근 방식을 통해 기존의 통계적 데이터 중심의 모션 생성 모델의 한계를 극복하고자 합니다. 이 모델은 비디오 게임, AR/VR 및 로보틱스와 같은 다양한 응용 분야에서 사용될 수 있습니다.

- **Technical Details**: HUMOS는 transformer 기반의 조건부 Variational Auto-Encoder (c-VAE)를 사용하여 비연결 데이터(unpaired data)로부터 인간의 모션을 생성합니다. 이 모델은 동적 직관적 물리학(dynamically intuitive physics) 용어와 안정성 제약을 적용하여 다양한 신체 형태에 대한 물리적으로 그럴듯한 모션을 생성하는 데 중요합니다. 특히, 중심 질량(Center of Mass, CoM), 압력 중심(Center of Pressure, CoP) 및 제로 모멘트 포인트(Zero Moment Point, ZMP) 간의 상호작용을 모델링하는 동적 안정성 개념을 도입합니다.

- **Performance Highlights**: HUMOS는 기존 최첨단 방법들보다 정량적 및 정성적으로 더 사실적인 모션을 생성하며, 다양한 신체 형태를 고려하여 실제 사람의 모션처럼 보일 수 있도록 설계되었습니다. 특히, 복잡한 포즈와 신체 접촉을 포함하는 모션을 효과적으로 재타겟팅할 수 있는 점이 이 논문의 주요 기여 중 하나입니다.



### Deep Clustering of Remote Sensing Scenes through Heterogeneous Transfer Learning (https://arxiv.org/abs/2409.03938)
- **What's New**: 본 논문에서는 라벨이 없는 원거리 관측 장면의 타겟 데이터셋에 대한 비지도 전체 이미지 클러스터링 방법을 제안합니다. 이 방법은 사전 훈련된 딥 뉴럴 네트워크(DINOv2)를 사용해 각 이미지에서 특징 벡터를 추출하고, 이 깊은 특징을 저차원 유클리드 공간으로 축소한 뒤, 베이지안 비모수기법을 사용해 클러스터를 형성하는 세 가지 주요 단계로 구성됩니다.

- **Technical Details**: 제안된 방법은 이질적 전이 학습(heterogeneous transfer learning)을 이용해 이미지 클러스터링을 수행하며, 첫 번째 단계로 라벨이 있는 원거리 관측 이미지 데이터에 대해 사전 훈련된 DNN을 미세 조정하고, 이를 통해 각 라벨 없는 타겟 이미지의 고차원 딥 특징 벡터를 추출합니다. 그 다음, 매니폴드 프로젝션(manifold projection) 기법을 사용해 고차원 딥 특징을 저차원 유클리드 공간으로 임베딩(embedding)하고, 마지막으로 Dirichlet Process Gaussian Mixture Model(DPGMM)을 적용하여 클러스터 수와 구성원의 소속을 추론합니다.

- **Performance Highlights**: 이 방법은 여러 원거리 관측 장면 분류 데이터셋에서 최신의 제로샷(zero-shot) 분류 방법보다 우수한 성능을 보여줍니다.



### Data-Efficient Generation for Dataset Distillation (https://arxiv.org/abs/2409.03929)
Comments:
          13 pages, 7 figures

- **What's New**: 본 논문은 클래스 조건부(conditional) 잠재 확산(latent diffusion) 모델을 사용하여 인간이 이해할 수 있는 합성 이미지(synthetic images)를 생성하는 새로운 접근 방식을 제안합니다. 이 방식은 데이터셋의 크기를 축소하면서도 그 본질적인 정보를 유지할 수 있도록 합니다.

- **Technical Details**: 연구진은 UViT 모델을 사용하여 대규모 실 이미지 데이터셋에서 학습하고, 이를 기반으로 빠른 속도로 수십 개의 합성 이미지를 생성할 수 있는 소형 데이터셋을 만듭니다. 이 과정에서, 합성 이미지의 품질을 보장하고, 계산 비용을 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구는 2024 ECCV에서 열린 첫 번째 데이터셋 디스틸레이션 챌린지에서 CIFAR100과 TinyImageNet 데이터셋에서 1위를 기록하였으며, 실제 테스트 세트에서 소수의 합성 이미지를 사용하여 효과적인 모델 학습을 달성하였습니다.



### Image Recognition for Garbage Classification Based on Pixel Distribution Learning (https://arxiv.org/abs/2409.03913)
- **What's New**: 이 연구에서는 경제 및 산업 발전으로 증가하는 폐기물 문제를 해결하기 위해 pixel distribution learning (픽셀 분포 학습) 기법을 활용한 자동 폐기물 분류 방법을 제안합니다. 이 방법은 기존의 CNN(Convolutional Neural Network) 기반 접근법의 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 방법은 VGGNet에서 영감을 받은 CNN 구조를 기반으로 하며, 총 6가지 유형의 쓰레기(종이상자, 유리, 종이, 금속, 쓰레기, 플라스틱)를 분류합니다. 세 가지 실험을 통해 원본 데이터셋에서의 직접 학습, 데이터 증강 기법 적용 및 무작위로 섞인 이미지 패치들을 사용한 학습 방법을 다룹니다.

- **Performance Highlights**: Kaggle Garbage Classification dataset(카글 폐기물 분류 데이터셋)을 사용하여 기존 모델과 비교 실험을 진행하며, pixel distribution learning의 강력함과 효율성을 입증할 계획입니다. 이를 통해 보다 효과적인 자동 쓰레기 분류 방법론을 제공할 것입니다.



### The Role of Generative Systems in Historical Photography Management: A Case Study on Catalan Archives (https://arxiv.org/abs/2409.03911)
Comments:
          Accepted at ECCV workshop AI4DH

- **What's New**: 이 연구에서는 역사적 자료의 설명을 위해 생성 모델의 양적 기여를 살펴봅니다. 특히 카탈루냐 아카이브의 역사적 사진 캡션을 사례 연구로 설정하여 언어적 근접성과 시각적 적응 기반의 캡션 모델 전이 학습에 대한 도구와 방향을 제공합니다.

- **Technical Details**: 본 연구는 CATR (CAption TRansformer) 모델을 사용하여 이미지 캡셔닝 작업을 수행합니다. 이 모델은 CNN과 비전 변환기(Transformer)가 결합된 구조로, 이미지 특징을 추출하고 해당 특징을 이용하여 텍스트 생성을 수행합니다. 또한, 다국적 데이터셋을 통해 모델의 훈련을 진행하여 소외 언어의 성능 향상을 목표로 합니다.

- **Performance Highlights**: 이 연구는 역사적 맥락과 언어적 특수성을 고려한 이미지 설명 생성 모델을 제안하며, 이를 통해 전통적인 사진 관리에 대한 새롭고 효율적인 접근 방식을 제공합니다. 다양한 언어의 데이터셋을 활용하여 설명의 정확하고 관련성을 높이는 방향으로 발전할 가능성을 보여줍니다.



### On-board Satellite Image Classification for Earth Observation: A Comparative Study of Pre-Trained Vision Transformer Models (https://arxiv.org/abs/2409.03901)
- **What's New**: 본 연구는 위성 처리에서 토지 이용 분류에 가장 효과적인 사전 훈련(pre-trained) 모델을 식별하는 데 중점을 두고 있으며, 전통적인 CNN 및 ResNet 기반 모델과 다양한 사전 훈련된 Vision Transformer 모델을 비교한 것이다.

- **Technical Details**: 연구에서는 Remote Sensing Image Classification(RSIC)을 위한 전통적인 CNN 및 ResNet 모델과 사전 훈련된 Vision Transformer 모델, 특히 MobileViTV2 및 EfficientViT-M2를 비교하였다. EfficientViT-M2는 노이즈가 있는 데이터에서 견고성을 보여주며, 위성 기반 추론에서 탁월한 성능을 발휘했다.

- **Performance Highlights**: EfficientViT-M2는 98.76%의 정확도를 달성하며 우수한 훈련 효율성과 빠른 추론 시간을 제공했다. 이 모델은 전체 견고성 점수가 0.79로 나타났으며, 깨끗한 검증 데이터에서는 MobileViTV2보다 높은 성과를 보였지만, 노이즈 환경에서의 성능에서 EfficientViT-M2가 더 뛰어났다.



### MVTN: A Multiscale Video Transformer Network for Hand Gesture Recognition (https://arxiv.org/abs/2409.03890)
- **What's New**: 본 논문에서는 동적 손 제스처 인식을 위한 새로운 다중 스케일 비디오 변환기 네트워크(Multiscale Video Transformer Network, MVTN)를 소개합니다. 이 네트워크는 다양한 크기, 자세, 형태의 특징을 추출할 수 있도록 설계되었습니다. 또한, RGB 이미지, 깊이 맵, 적외선 데이터 및 표면 노멀을 활용하는 다중 모달 데이터(multi-modal data)를 결합하여, 손 제스처 인식의 효율성을 높입니다.

- **Technical Details**: MVTN은 다중 스케일 기능 계층을 포함하여 손 제스처 내의 다양한 세부 사항과 문맥을 포착합니다. 초기 단계에서는 고해상도 특징을 모델링하고, 후속 단계에서는 저해상도 특징을 모델링하는 다양한 주의(attention) 차원을 추출하여 다중 스케일 계층을 구성합니다. 이 방식은 선형 투사(linear projection)를 사용하여 변환기 모델의 서로 다른 단계에서 다중 스케일 주의 특징을 추출합니다.

- **Performance Highlights**: 실험을 통해 MVTN은 NVGesture 및 Briareo 데이터셋에서 이전의 최고 성능(state-of-the-art)을 달성하였으며, 낮은 계산 복잡도와 매개변수 수를 보입니다. 또한, 이 모델은 손 제스처 인식에서 기존의 방법들보다 더 나은 성능을 발휘합니다.



### The Influence of Faulty Labels in Data Sets on Human Pose Estimation (https://arxiv.org/abs/2409.03887)
Comments:
          15 pages, 7 figures, 5 tables

- **What's New**: 이번 연구는 Human Pose Estimation (HPE)에서 모델 성능에 미치는 훈련 데이터의 품질 영향을 실증적으로 증명합니다. 일반적으로 사용되는 데이터 세트의 부정확한 레이블이 학습과 성능 metrics에 미치는 부정적인 영향을 분석합니다.

- **Technical Details**: 연구진은 표준 HPE 데이터 세트인 COCO와 MPII를 분석하여 레이블 오류의 범위와 종류를 확인하고, 데이터 정제를 통해 모델 성능을 향상시키는 방법을 제안합니다. HPE 모델의 효과적인 학습을 위해 레이블 오류를 분류하고 이를 확인하는 간단한 heuristic 방법론을 개발했습니다.

- **Performance Highlights**: 클린스된 데이터로 인해 HPE 모델의 성능이 향상되었음을 보여주며, 이는 모델의 일반화 성능에도 긍정적인 영향을 미친다는 것을 입증합니다.



### Multi-Camera Industrial Open-Set Person Re-Identification and Tracking (https://arxiv.org/abs/2409.03879)
Comments:
          Accepted at T-CAP workshop at ECCV 2024

- **What's New**: 최근 딥러닝을 활용한 인물 재식별(person re-identification) 기술의 발전으로 인상적인 결과들이 나왔으나, 실제 산업 적용에 한계가 있었습니다. 본 연구에서는 MICRO-TRACK이라는 모듈형 다중 카메라 인물 재식별 및 개방형 추적 시스템을 제안하여 실시간으로 작동하고 기존 산업 감시 시스템에 쉽게 통합될 수 있도록 합니다.

- **Technical Details**: MICRO-TRACK 시스템은 다중 카메라 환경에서 작동하며, 두 가지 주요 측면으로 구분됩니다: 시간에 따른 추적과 공간적 외관을 이용한 Re-ID입니다. 이 시스템은 신뢰할 수 있는 시간 추적 시스템을 통해 사람의 궤적을 식별하며, 이미지 품질이 좋은 조건에서 외형 기반의 방법으로 고유한 재식별 ID를 제공합니다. 또한, 8개의 감시 카메라로 촬영된 18분의 영상을 포함하는 Facility-ReID라는 새로운 데이터셋을 출시하였습니다.

- **Performance Highlights**: 이 시스템은 복잡한 동적 환경에서도 오클루전(occlusion)에 강인하며, 실제 산업 시설에 성공적으로 설치 및 테스트되었습니다. 본 논문에서는 기존의 폐쇄형 설정과 개방형 설정에서 실험을 수행하고, 새로운 Facility-ReID 데이터셋을 통해 우리 시스템의 성능을 평가하며, 그 결과를 논의합니다.



### Ground-roll Separation From Land Seismic Records Based on Convolutional Neural Network (https://arxiv.org/abs/2409.03878)
- **What's New**: 이 논문에서는 CNN(Convolutional Neural Network)을 기반으로 한 새로운 방법으로, 지반 롤(groun-roll)과 반사(signal reflection)를 효과적으로 분리하는 방법을 제안합니다. 기존의 전통적인 방법들이 가진 한계점을 극복하고, 자동으로 특징을 추출할 수 있는 CNN 모델을 활용하여 이 문제를 접근합니다.

- **Technical Details**: 제안된 방법은 CNN 모델을 사용하여 지반 롤 성분과 반사 성분을 동시에 출력하게 됩니다. 이 과정에서 각 성분의 정확성을 높이기 위해 유사도 손실(similarity loss)과 판별 손실(discriminative loss)을 훈련 과정에 적용합니다. 또한, 입력 데이터는 지반 롤에 의해 오염된 저주파 필터링된 지진 데이터를 사용합니다.

- **Performance Highlights**: 실험 결과는 합성 데이터(synthetic data)와 실제 데이터(real data) 모두에서 CNN 기반 방법이 지반 롤을 반사와 효과적으로 분리할 수 있음을 보여주며, 특정 조건에서 일반화(so-called generalization) 능력을 보임을 통해 외부 데이터셋에 대해서도 효과적임을 입증하고 있습니다.



### Few-shot Adaptation of Medical Vision-Language Models (https://arxiv.org/abs/2409.03868)
Comments:
          MICCAI 2024 (Spotlight) - Code is available at this https URL

- **What's New**: 이 논문은 의료 비전-언어 모델(VLM)을 위한 최초의 구조화된 벤치마크를 소개하고, 제한된 few-shot 제도에서 다양한 적응 전략을 연구합니다.

- **Technical Details**: 이 논문은 학습 가능한 클래스별 가중치를 통해 시각적 프로토타입과 텍스트 임베딩의 최적 혼합을 탐구하며, 텍스트 정보를 활용한 선형 프로브(adaptation baseline)로 경쟁력 있는 성과를 보여줍니다. 다양한 의료 영상 모달리티에 대해 세 가지 방법으로 평가합니다.

- **Performance Highlights**: 제안하는 접근법은 기존의 복잡한 프롬프트 학습 및 어댑터 기반 전략보다 우수한 성능을 보이며, 빠른 실행 속도를 자랑하고, 블랙박스 설정에서의 적용성도 높습니다.



### Assessing the Uncertainty and Robustness of Object Detection Models for Detecting Stickers on Laptops (https://arxiv.org/abs/2409.03782)
Comments:
          18 pages, 6 figures, 4 tables

- **What's New**: 이번 연구는 노르웨이의 덴마크 기술 연구소(DTI)에서 실시한 스티커 검출 모델(SDM)의 연구로, 공공 전자 기기의 생명 연장을 위한 리퍼브리시(Refurbishing) 과정을 자동화하는 데 중점을 두고 있습니다. 연구팀은 모델의 예측 불확실성을 정량화하고, 이를 통해 자동 스티커 제거의 신뢰성을 높이는 방법론을 제시하였습니다.

- **Technical Details**: 연구에서는 여섯 개의 스티커 검출 모델(SDM)을 훈련시키고, Monte Carlo Dropout(모델의 예측 불확실성을 정량화하기 위한 방법)을 적용하여 스티커의 위치와 유형을 파악했습니다. 세 가지 데이터셋을 활용하여 SDM의 성능을 평가하며, 새로운 Robustness Metrics를 도입하여 정확성과 예측 불확실성을 기반으로 SDM의 강인성을 평가하였습니다.

- **Performance Highlights**: SDM의 평가 결과, Faster R-CNN_v2 모델이 스티커 검출 정확도에서 가장 우수한 성능을 보였으며, RetinaNet_v2는 예측 불확실성 측면에서 가장 높은 성능을 기록하였습니다. 두 모델 모두 적대적 강인성 평가에서도 우수한 결과를 나타냈습니다. 연구 결과에 따른 SDM 선택 가이드라인과 도출된 시사점도 제공하였습니다.



### A Greedy Hierarchical Approach to Whole-Network Filter- Pruning in CNNs (https://arxiv.org/abs/2409.03777)
Comments:
          Accepted in TMLR 2024

- **What's New**: 본 논문은 전체 네트워크 필터 프루닝(whole-network filter pruning)을 위한 두 단계 계층적 접근법을 제안합니다. 이 방법은 분류 손실(classification loss)을 최종 기준으로 사용하며, 필터 선택 알고리즘을 통해 각 레이어에서 최적화된 필터를 선택합니다.

- **Technical Details**: 이 논문에서 제안하는 방법은 두 가지 알고리즘을 포함합니다: 하나는 직교 매칭 탐색 기반의 탐욕적 선택(orthogonal matching pursuit-based greedy selection)이고, 다른 하나는 혁신적인 폐쇄형 오류 기준을 사용하는 탐욕적 백워드 프루닝(greedy backward pruning approach)입니다. 필터 프루닝과 계층 선택(layer-selection) 두 가지 수준에서 작동하여 전반적인 알고리즘을 가속화합니다.

- **Performance Highlights**: 우리의 방법은 ResNet18, ResNet32, ResNet56, VGG16 및 ResNext101의 기존 최첨단 프루닝 방법들에 비해 우수한 성능을 보이며, ResNext101의 RAM 요구량을 7.6GB에서 1.5GB로 줄여줍니다. 또한 CIFAR-10에서 정확도 손실 없이 FLOPS를 94% 감소시킵니다.



### EMCNet : Graph-Nets for Electron Micrographs Classification (https://arxiv.org/abs/2409.03767)
Comments:
          12 pages, 10 figures, Accepted in a ACM SIGKDD 2022 Workshop on Machine Learning for Materials

- **What's New**: 본 논문에서는 전자 현미경 이미지(전자 마이크로그래프)의 효과적인 표현 학습을 기반으로 한 프레임워크를 제안하여 나노물질(nanomaterials) 식별의 어려움을 극복합니다. 기존 방법들보다 우수한 성과를 내며, 오픈 소스 데이터셋에서의 결과를 통해 이를 입증합니다.

- **Technical Details**: 제안된 EMCNet는 (1) 나노 스케일 이미지의 토큰화, (2) 그래프 인코더(GEnc), (3) 계층 그래프 인코더(HGEnc), (4) 클리크 트리 분해, (5) 클리크 트리 인코더(CTEnc), (6) 출력 레이어의 총 6단계로 구성됩니다. 이 모델은 각 나노 물질 이미지의 패치를 노드로 표현하여, 그들 간의 관계를 효과적으로 학습합니다.

- **Performance Highlights**: 제안된 방법은 다양한 나노물질 식별 과제에서 기존의 주요 기준선(baselines)보다 뛰어난 성능을 보여주며, 각 단계에 대한 성능 연구(ablation study)를 통해 접근 방식의 효과성을 상세히 지원합니다.



### OpenCap markerless motion capture estimation of lower extremity kinematics and dynamics in cycling (https://arxiv.org/abs/2409.03766)
- **What's New**: 이 연구는 OpenCap이라는 마커 없는 (markerless) 모션 캡처 시스템이 전통적인 마커 기반 시스템과 비교하여 사이클링 생체역학 (biomechanics)을 평가하는 성능을 조사한 것입니다.

- **Technical Details**: OpenCap은 스마트폰 카메라에서 촬영한 비디오를 이용하여 인체의 3D 운동학 (kinematics)을 추정합니다. 연구에 참여한 10명의 건강한 성인은 두 가지 방법을 사용하여 엉덩이, 무릎, 발목의 운동학 및 동역학 (dynamics)을 측정했습니다. 두 시스템 간의 상관 계수는 0.98을 초과하여 매우 강한 일치를 보여주었습니다.

- **Performance Highlights**: 측정된 운동학적 오차는 4도 이하, 동역학적 오차는 5Nm 이하로 오차가 최소화되었습니다. OpenCap은 엉덩이 (flexion/extension), 무릎 (flexion/extension), 발목 (dorsiflexion/plantarflexion) 관절에 대해 전통적인 모션 캡처보다 비교 가능한 정밀도를 제공합니다.



### AI and Entrepreneurship: Facial Recognition Technology Detects Entrepreneurs, Outperforming Human Experts (https://arxiv.org/abs/2409.03765)
Comments:
          46 pages, 2 tables, 11 figures

- **What's New**: 이 연구는 인공지능(AI)이 소셜 미디어와 같은 널리 사용되는 인간 중심 데이터에서 개인의 직업 정보를 정확하게 추출할 수 있는지 조사했습니다.

- **Technical Details**: 40,728명의 개인 얼굴 이미지로 구성된 데이터셋을 활용하여 CNN(Convolutional Neural Network) 모델을 훈련했습니다. 이 모델은 단일 얼굴 이미지를 기준으로 기업가를 분류할 수 있습니다.

- **Performance Highlights**: AI 모델은 79.51%의 분류 정확도를 기록했으며, 이는 인간 전문가와 훈련된 참여자들이 우연 확률(>50%)을 초과하여 기업가를 분류할 수 없었던 것과 대조적입니다.



### Modeling Human Strategy for Flattening Wrinkled Cloth Using Neural Networks (https://arxiv.org/abs/2409.03764)
Comments:
          6 Pages

- **What's New**: 본 논문은 인간의 방법을 학습하여 주름진 천을 편평하게 만드는 전략을 모델링하는 혁신적인 접근 방식을 제안합니다. 특히, 다양한 주름 유형을 제시하여 최소한의 동작으로 천을 편평하게 만드는 작업을 수행하도록 참가자에게 요청하였습니다. 이미지 처리 기법과 PCA(Principal Component Analysis)를 이용해 입력 차원을 줄이는 과정이 포함되었습니다.

- **Technical Details**: 이 연구에서는 10명의 대학생 참가자가 주름진 천을 피혁을 위한 테이블 위에 특수 고정된 상태로 놓고, 주어진 주름 유형에 따라 지정된 방법으로 천을 편평하게 만드는 상황에서 데이터 수집이 이루어졌습니다. Aruco 마커와 카메라 시스템을 사용하여 손가락 움직임 및 천의 이미지를 캡처하였으며, 이러한 데이터를 바탕으로 회귀 신경망이 훈련되었습니다.

- **Performance Highlights**: 훈련된 신경망은 독립적인 데이터 세트에서 실제 인간의 행동과 거의 유사한 예측을 하여 주름진 천을 편평하게 만드는 인간 행동 모델링에서의 신경망의 효과성을 입증하였습니다. 이는 다수의 산업에서의 자동화 능력을 개선하는 데 중요한 기여를 할 것으로 기대됩니다.



### A Dataset for Mechanical Mechanisms (https://arxiv.org/abs/2409.03763)
- **What's New**: 본 연구에서는 기계 메커니즘 디자인을 지원하기 위해 약 9,000장의 이미지와 해당 설명으로 구성된 데이터셋을 소개합니다. 이는 다양한 2D 및 3D 스케치의 모음을 포함하며, 생성적 AI 모델이 메커니즘 디자인에 활용될 수 있는 가능성을 보여줍니다.

- **Technical Details**: 데이터셋은 웹 스크래핑을 통해 수집되었으며, CAD 소프트웨어를 사용한 기계 디자인의 YouTube 채널, 기계 및 기어에 전념한 디지털 라이브러리, 2D 스케치 및 설명이 포함된 책 등이 포함됩니다. 이 과정에서 모든 이미지를 수동으로 검토하고 불필요한 내용을 삭제하여 일관성을 확보했습니다. 이 데이터를 바탕으로 Stable Diffusion과 BLIP-2 모델을 미세 조정하였습니다.

- **Performance Highlights**: Stable Diffusion 모델의 경우 2D 스케치에서 일관성 부족으로 인해 의미 없는 결과가 발생했지만, 3D 스케치에서는 높은 품질의 생성을 보여주었습니다. BLIP-2 모델은 생성된 캡션의 정확도가 낮았으나, 이는 훈련 에폭 수가 부족했기 때문입니다. 전반적으로 이 연구는 생성적 AI의 기계 디자인 활용 가능성과 데이터셋의 향후 개선 필요성을 강조합니다.



### Efficient Scene Appearance Aggregation for Level-of-Detail Rendering (https://arxiv.org/abs/2409.03761)
- **What's New**: 본 논문에서는 복잡한 3D 장면의 appearance-preserving level-of-detail (LoD) 표현을 위한 새로운 볼륨 표현 방법과 효율적인 LoD 생성 및 렌더링 파이프라인을 제안합니다.

- **Technical Details**: 주요 내용은 Aggregated Bidirectional Scattering Distribution Function (ABSDF)을 중심으로 하며, 이는 복셀 내 모든 표면의 원거리 appearance를 요약합니다. 공간적 및 방향적 재료 변수를 고려하여 ABSDF의 닫힌 형태 분해를 제안하고, 복셀 내 및 장면의 서로 다른 부분 간의 상관관계를 포착하는 여러 기법을 포함합니다.

- **Performance Highlights**: 본 방법은 높은 렌더링 품질을 달성하며 기존의 장면 필터링 방법보다 더 나은 결과를 제공합니다. 메모리 발자국 및 렌더링 비용은 원래 장면의 복잡성에 독립적입니다.



### Exploring Foundation Models for Synthetic Medical Imaging: A Study on Chest X-Rays and Fine-Tuning Techniques (https://arxiv.org/abs/2409.04424)
- **What's New**: 이 논문은 기초 모델(Foundation Models)을 활용하여 고해상도의 합성 흉부 X-선을 생성하는 방법을 탐구하고, 이를 미세 조정(Fine-tuning)하여 이 모델의 성능이 어떻게 향상되는지를 평가합니다.

- **Technical Details**: 논문에서는 Latent Diffusion Model (LDM)을 제안하며, 사전 훈련된 기초 모델을 기반으로 여러 설정을 통해 모델을 세밀하게 조정합니다. 연구는 실제 의료 전문가의 입력을 기반으로 진행되었습니다.

- **Performance Highlights**: 초기 실험 결과, 미세 조정을 통해 생성된 이미지의 현실감이 향상되었으며, 30개의 흉부 X-선 이미지를 활용한 모델 훈련이 이루어졌습니다. 생성된 데이터는 스테이블 디퓨전과 GAN 기반 방법보다 더 높은 성능을 보여주었습니다.



### The Impact of Scanner Domain Shift on Deep Learning Performance in Medical Imaging: an Experimental Study (https://arxiv.org/abs/2409.04368)
- **What's New**: 이 논문은 다양한 의료 이미징 모달리티(MRI, CT, X-ray)에서 스캐너 도메인 이동(scanner domain shift)이 심층 신경망의 성능에 미치는 영향을 체계적으로 연구한 최초의 다중 모달리티(multi-modality) 연구입니다.

- **Technical Details**: 의료 이미지를 다른 스캐너와 프로토콜을 사용하여 획득할 때 이미지의 특성이 달라지며, 이는 심층 신경망의 성능 저하를 초래할 수 있습니다. 특히, MRI에서 성능 저하가 가장 심한 경향을 보였고, 이는 X-ray는 중간 정도, CT는 상대적으로 적은 것으로 나타났습니다. 또한, 타겟 도메인 데이터를 학습 세트에 추가함으로써 일반화(generalization) 성능을 개선할 수 있음을 확인하였습니다.

- **Performance Highlights**: 결과적으로, 14개 세팅 중 11개 세팅에서 스캐너 도메인 이동이 성능 저하를 초래했습니다. 특히 MRI 작업에서의 도메인 이동 문제가 가장 심각하며, CT 작업은 최소한의 영향을 받는 것으로 나타났습니다.



### Calibration of Network Confidence for Unsupervised Domain Adaptation Using Estimated Accuracy (https://arxiv.org/abs/2409.04241)
- **What's New**: 본 연구는 라벨이 없는 목표 도메인(target domain) 샘플을 사용하여 원래 소스 도메인(source domain)에서 훈련된 모델의 신뢰도를 조정하는 문제를 다루고 있습니다. 우리는 목표 도메인에서의 네트워크의 정확도를 추정하는 기반으로 하는 새로운 캘리브레이션(calibration) 절차를 도입하였습니다.

- **Technical Details**: 본 연구는 Unsupervised Domain Adaptation (UDA) 설정에서, 레이블이 없는 목표 도메인 데이터에 대해 적응된 모델의 신뢰도를 조정하는 방법을 제안합니다. 우리는 먼저 목표 도메인의 정확도를 평가하고, 그에 맞는 캘리브레이션 매개변수를 찾습니다. 또한 Expected Calibration Error (ECE) 측정값을 최소화하여 캘리브레이션을 진행합니다.

- **Performance Highlights**: 실험 결과, 우리 방법은 기존의 중요도 가중치(Importance Weighting) 기반 방법보다 여러 표준 데이터셋에서 유의미하게 성능이 향상되었습니다. 결과적으로, 우리는 UDA를 위한 네트워크 캘리브레이션의 새로운 기준을 정립했습니다.



### CISCA and CytoDArk0: a Cell Instance Segmentation and Classification method for histo(patho)logical image Analyses and a new, open, Nissl-stained dataset for brain cytoarchitecture studies (https://arxiv.org/abs/2409.04175)
- **What's New**: 본 논문에서는 세포 단위 분할 및 분류를 위한 새로운 딥러닝 프레임워크(CISCA)를 제안합니다. 이 프레임워크는 조직 이미지에서 세포의 형태학적 분석, 세포 수 계산 및 뇌 세포 구조 연구를 지원합니다.

- **Technical Details**: CISCA는 경량 U-Net 아키텍처를 바탕으로 하며, 디코더에서 세 개의 헤드를 사용하는 네트워크 구조를 가지고 있습니다. 첫 번째 헤드는 이웃 세포들 간의 경계, 세포체 및 배경으로 픽셀을 분류하고, 두 번째 헤드는 네 방향을 따라 네 개의 거리 맵을 회귀합니다. 세 번째 헤드는 세포를 필요에 따라 관련 클래스에 동시에 분류합니다.

- **Performance Highlights**: CISCA는 CoNIC, PanNuke, MoNuSeg를 포함한 네 개의 데이터셋에서 효과성을 입증했습니다. 최첨단 방법들과 비교했을 때, CISCA는 다양한 조직 유형, 확대 배율 및 염색 기법에서 세포 분할 및 분류에서 강력함과 정확성을 보여주었습니다.



### Optical Coherence Tomography Angiography-OCTA dataset for the study of Diabetic Retinopathy (https://arxiv.org/abs/2409.04137)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 179명의 개인으로부터 수집된 268개의 망막 이미지로 구성된 데이터셋을 소개합니다. 이 데이터셋은 인도 마하라슈트라주 푸네에 있는 Natasha Eye Care and Research Institute에서 수집되었습니다.

- **Technical Details**: 이미지는 비확장 Optical Coherence Tomography Angiography (OCTA) 장치인 Optovue Avanti Edition 기계를 사용하여 촬영되었습니다. 두 명의 안과 전문의가 이미지를 주석 처리하였습니다.

- **Performance Highlights**: 이 데이터셋은 연구자와 의사들이 당뇨병성 망막병증 (Diabetic Retinopathy, DR)의 조기 발견을 위한 자동화된 진단 도구 개발에 활용될 수 있습니다.



### MixNet: Joining Force of Classical and Modern Approaches Toward the Comprehensive Pipeline in Motor Imagery EEG Classification (https://arxiv.org/abs/2409.04104)
Comments:
          Supplementary materials and source codes are available on-line at this https URL

- **What's New**: 이 논문에서는 MixNet라는 새로운 분류 프레임워크를 제안하여 motor imagery (MI) 기반 뇌-컴퓨터 인터페이스 (BCI) 시스템의 분류 성능을 개선합니다. MixNet는 filter-bank common spatial patterns (FBCSP) 방법을 사용하여 MI 데이터에서 스펙트럼-공간 신호를 생성하며, multitask learning 구조인 MIN2Net를采용하여 분류를 수행합니다.

- **Technical Details**: MixNet는 MI 데이터의 스펙트럼-공간 신호를 기반으로 하며, multi-task learning의 각 과제가 서로 다른 일반화률 및 오버피팅 경향을 보인다는 문제를 해결하기 위해 adaptive gradient blending을 구현합니다. 이 기술은 여러 손실 가중치를 동시에 조절하고 각 과제의 학습 속도를 조정하여 효과적으로 최적화합니다. 또한, MixNet는 작은 데이터 세트에서도 강력한 성능을 발휘합니다.

- **Performance Highlights**: MixNet는 6개의 벤치마크 데이터 세트에서 모든 최첨단 알고리즘을 초월한 성능을 보였으며, 저밀도 EEG MI 분류의 경우에도 우수한 결과를 나타냈습니다. 이러한 성과는 IoT 응용 프로그램을 위한 경량형 및 휴대용 EEG 착용 장치 개발에 유망한 시사점을 제공합니다.



### EigenSR: Eigenimage-Bridged Pre-Trained RGB Learners for Single Hyperspectral Image Super-Resolution (https://arxiv.org/abs/2409.04050)
Comments:
          Submitted to AAAI 2025

- **What's New**: 이 연구는 단일 하이퍼 스펙트럼 이미지 슈퍼 해상도(single-HSI-SR)를 위한 새로운 프레임워크인 EigenSR(아이겐SR)을 제안합니다. 이는 사전 훈련된 RGB 모델을 하이퍼 스펙트럼 이미지에 효과적으로 적용하여 데이터 부족 문제를 해결하고자 합니다.

- **Technical Details**: EigenSR은 두 단계로 구성됩니다. 첫째, 사전 훈련된 RGB 모델을 공간 구성 요소(채널)가 단일 채널 방식으로 적응시킵니다. 둘째, 이 모델을 낮은 해상도의 HSI에 대해 추론할 때 사용할 때, 적분적 스펙트럴 정규화(Iterative Spectral Regularization, ISR)를 도입하여 스펙트럴 상관관계를 강조합니다. 이를 통해 RGB 모델의 공간적 텍스처 처리 능력을 HSI에 주입하면서 스펙트럴 충실도를 유지합니다.

- **Performance Highlights**: 실험 결과, EigenSR은 공간 및 스펙트럴 메트릭에서 현재 최첨단(SOTA) 방법들을 초월할 만큼 우수한 성능을 보여주었습니다.



### Bi-modality Images Transfer with a Discrete Process Matching Method (https://arxiv.org/abs/2409.03977)
- **What's New**: 최근 의료 영상 생성(medical image synthesis)은 생성 모델(generative models)의 급속한 발전과 함께 점점 더 인기를 끌고 있습니다. 이 연구에서는 기존의 flow-based 모델의 한계를 극복하고, 더 높은 품질의 이미지를 생성하기 위해 Discrete Process Matching(DPM)이라는 새로운 모델을 제안합니다.

- **Technical Details**: 본 연구에서 제안한 DPM은 기존의 flow matching 기반 모델과는 달리, 앞쪽(forward)과 뒤쪽(backward) ODE(ordinary differential equations) 흐름을 모두 활용합니다. 이는 몇 개의 이산(discrete) 시간 단계에서 중간 이미지의 일관성을 강화함으로써, 고품질 생성을 유지하면서도 훨씬 적은 반복(iteration) 단계로 이행 과정을 수행할 수 있도록 합니다.

- **Performance Highlights**: MRI T1/T2 및 CT/MRI의 세 가지 데이터 세트에서 DPM은 바이모달 영상 합성(bi-modality image synthesis)에서 최첨단의 다른 flow-based 방법을 능가하며, 더 적은 계산 시간으로 더 높은 이미지 품질을 달성했습니다.



### Recon-all-clinical: Cortical surface reconstruction and analysis of heterogeneous clinical brain MRI (https://arxiv.org/abs/2409.03889)
Comments:
          16 pages in the manuscript with 11 page supplementary material

- **What's New**: 이 논문에서는 새로운 방법인 recon-all-clinical을 제안하며, 이는 다양한 해상도와 대비의 뇌 MRI 스캔에 대해 피질 복원, 등록, 분할, 두께 추정을 수행할 수 있습니다.

- **Technical Details**: 이 접근법은 도메인 무작위화 방법을 통해 훈련된 합성곱 신경망(CNN)과 전통적인 기하 처리 방법을 결합하여 정확한 표면 배치를 가능하게 하며, 다양한 MRI 대비와 해상도에서도 신뢰성 있는 측정치를 제공합니다.

- **Performance Highlights**: 19,000개 이상의 임상 스캔에서 테스트한 결과, 이 방법은 다양한 MRI 조건에서도 정밀한 피질 복원과 높은 분할 정확도를 보여주었으며, 나이 효과를 독립적으로 캡처할 수 있는 정밀한 피질 두께 추정이 가능하였습니다.



### Mpox Screen Lite: AI-Driven On-Device Offline Mpox Screening for Low-Resource African Mpox Emergency Respons (https://arxiv.org/abs/2409.03806)
Comments:
          11 Pages, 2 Figures, 3 Tables

- **What's New**: 2024 Mpox 감염병 발생의 심각성과 1b 클레이드의 출현을 배경으로, 자원 제한 환경에서 사용할 수 있는 인공지능(AI) 기반의 오프라인 스크리닝 도구가 개발되었습니다.

- **Technical Details**: YOLOv8n 기반의 딥러닝 모델이 2,700장의 이미지(각기 900장의 Mpox, 기타 피부 질환, 정상 피부 포함)에 대해 훈련되었습니다. 모델은 360장의 이미지로 검증되고 540장으로 테스트되었으며, 1,500개의 독립된 이미지로 외부 검증이 진행되었습니다. 성능 지표로는 accuracy, precision, recall, F1-score, sensitivity, specificity가 포함되었습니다.

- **Performance Highlights**: 최종 테스트 세트에서 모델은 96%의 높은 정확도를 보였고, Mpox 감지에 대해서는 93%의 precision, 97%의 recall, 95%의 F1-score를 기록하였습니다. 감지에 대한 sensitivity는 97%, specificity는 96%였습니다. 이러한 결과는 외부 검증에서도 일관성을 유지하여 모델의 강건성 및 일반화 가능성을 확인하였습니다.



### Evaluating Machine Learning-based Skin Cancer Diagnosis (https://arxiv.org/abs/2409.03794)
Comments:
          14 pages

- **What's New**: 이 연구는 피부암 탐지를 위한 두 가지 딥러닝 모델의 신뢰성을 평가하며, 설명 가능성(explainability)과 공정성(fairness)에 중점을 둡니다.

- **Technical Details**: 연구에서는 HAM10000 데이터셋을 활용하여 두 가지 CNN(Convolutional Neural Network) 아키텍처인 MobileNet 기반 모델과 커스텀 CNN 모델을 평가합니다. 피부 병변을 일곱 가지 카테고리로 분류하는 능력과 위험한 병변과 양성 병변을 구분하는 능력이 검토됩니다. 설명 가능성 평가는 Saliency Maps와 Integrated Gradients를 통해 이루어지며, 결과는 피부과 의사에 의해 해석됩니다.

- **Performance Highlights**: 두 모델은 일반적으로 대부분의 병변 유형에 대한 관련 기능을 강조하지만, 지루각화증(seborrheic keratoses)과 혈관 병변(vascular lesions) 같은 특정 클래스에서 어려움을 겪습니다. 공정성을 평가하기 위해 성별 및 피부 톤 그룹에 대한 Equalized Odds 지표가 사용됩니다. 두 모델은 성별 그룹 간 공정성을 보여주지만, 밝은 피부와 어두운 피부 톤 간 허위 긍정률(false positive rate) 및 허위 부정률(false negative rate)에서 상당한 차이를 보입니다. 이러한 불균형을 완화하기 위해 보정된 Equalized Odds 후처리 전략이 적용되어, 특히 허위 부정률 차이가 감소하는 개선된 공정성을 보여주었습니다.



### Exploiting XAI maps to improve MS lesion segmentation and detection in MRI (https://arxiv.org/abs/2409.03772)
- **What's New**: 본 논문은 다발성 경화증(Multiple Sclerosis, MS) 병변 분할(segmentation) 작업을 위한 인스턴스 수준의 설명 가능한 지도(instance-level explainable maps)를 생성하기 위해 두 개의 기존 기술을 변형하여 사용한 최신 방법론을 제시합니다.

- **Technical Details**: 3D U-Net 딥러닝 모델을 훈련하여 MS 병변을 추가하고, 약 21000개의 TP/FP 병변에서 생성된 설명 가능한 지도를 분석했습니다. 93개의 방사선적(radiomic) 기능을 이용해 로지스틱 회귀(logistic regression) 모델을 학습하였으며, TP와 FP 사례를 구분하는 데 사용되었습니다. 초기 모델에 비해 F1 점수는 0.7450, 양성 예측 값(Positive Predictive Value, PPV)은 0.7817로 개선되었습니다.

- **Performance Highlights**: 이 연구 결과는 설명 가능한 지도가 예측 점수를 세분화하여 모델의 성능을 향상할 수 있음을 시사합니다. 훈련 세트와 테스트 세트에서 TP와 FP의 의료 이미지 자원에서 추출된 방사선적 기능을 통해 모델 성능이 크게 향상되었습니다.



New uploads on arXiv(cs.AI)

### Improved Parallel Algorithm for Non-Monotone Submodular Maximization under Knapsack Constrain (https://arxiv.org/abs/2409.04415)
Comments:
          In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI), Main Track

- **What's New**: 본 연구에서는 knapsack 제약 문제 하에서 비단조(submodular) 최대화를 위한 효율적인 병렬(parallel) 알고리즘을 제안합니다. 기존 병렬 알고리즘의 최상의 근사 근사 계수를 8+ϵ에서 7+ϵ로 향상시켰으며, O(log n) 적응 복잡도를 갖습니다.

- **Technical Details**: 제안된 알고리즘은 새로운 대체 임계(threshold) 알고리즘 프레임워크를 기반으로 하여 두 개의 분리된 후보 솔루션을 상수 개수의 시퀀스 라운드에서 번갈아 구성합니다. 이 과정에서 솔루션의 품질을 향상시키면서 적응 복잡도를 희생하지 않습니다.

- **Performance Highlights**: 세 가지 응용 프로그램(수익 극대화, 이미지 요약, 최대 가중치 컷)에 대한 광범위한 실험 연구에서, 본 알고리즘은 솔루션 품질을 크게 개선할 뿐만 아니라 현재 최고 수준의 알고리즘에 비해 유사한 적응성을 요구함을 보여주었습니다.



### Using Large Language Models to Generate Authentic Multi-agent Knowledge Work Datasets (https://arxiv.org/abs/2409.04286)
Comments:
          Accepted and in press (INFORMATIK Festival, Wiesbaden, 2024)

- **What's New**: 이 논문에서는 현재 공개적으로 이용 가능한 지식 작업 데이터의 다양성 부족과 주석 부족, 사용자 및 문서에 대한 맥락 정보 부족 문제를 지적합니다. 이를 해결하기 위해, 구성 가능하고 다중 에이전트 기반의 지식 작업 데이터셋 생성기를 제안합니다.

- **Technical Details**: 제안된 시스템은 에이전트 간의 협업 지식 작업을 시뮬레이션하여 Large Language Model (LLM)로 생성된 문서와 관련 데이터 흔적을 생성합니다. 생성 과정에서 구성된 모든 배경 정보는 지식 그래프 (knowledge graph)로 기록됩니다. 이 데이터셋은 개인정보 및 기밀 문제 없이 사용 및 공유할 수 있습니다.

- **Performance Highlights**: 인간 평가자들이 생성된 문서 53%와 실제 문서 74%를 현실적이라고 평가한 결과는 제안된 접근법의 가능성을 보여줍니다. 참가자들의 피드백을 바탕으로 진정성 기준을 분석하고, 공통적으로 발견된 문제에 대한 개선 방안도 논의합니다.



### An overview of domain-specific foundation model: key technologies, applications and challenges (https://arxiv.org/abs/2409.04267)
- **What's New**: ChatGPT와 같은 기본 모델 기반 제품의 성능 향상으로 인해, 특정 산업 및 응용 시나리오에 맞춰 이러한 모델을 조정하는 방법에 대한 탐색이 이루어지고 있습니다. 이 논문은 도메인 특정(restricted) 기본 모델(customization of domain-specific foundation models)의 구축 방법론에 대해 포괄적인 개요를 제공합니다.

- **Technical Details**: 도메인 특정 기본 모델은 특정 산업을 위한 데이터와 응용 프로그램을 사용하여 개발됩니다. 이 모델들은 도메인 특정 데이터로 많이 훈련되어 있으며, LLM(대형 언어 모델) 기술에 기반한 인공신경망 모델입니다. 이 모델들은 '모달리티 인코더(Modality Encoder)', '입력 프로젝터(Input Projector)', '백본 계산기(Backbone Calculator)', '출력 프로젝터(Output Projector)', '모달리티 디코더(Modality Decoder)'로 구성된 다중 모달리티 구조를 가지고 있습니다.

- **Performance Highlights**: 도메인 특정 기본 모델은 일반 목적 모델에 비해 특정 분야의 전문적인 내용을 더 정확하게 이해하고 생성할 수 있는 능력을 가지고 있으며, 기술적 보안과 높은 경제적 이익이 기대됩니다. 그러나 데이터 수집 및 전처리 과정에서 여러 도전에 직면하게 됩니다.



### Advancing Multi-Organ Disease Care: A Hierarchical Multi-Agent Reinforcement Learning Framework (https://arxiv.org/abs/2409.04224)
- **What's New**: 본 연구에서는 다기관(多機關, multi-organ) 질환의 복잡성을 해결하기 위해 계층적 다중 에이전트 강화 학습(HMARL, Hierarchical Multi-Agent Reinforcement Learning)을 제안합니다. 이는 각 기관 시스템에 전담하는 에이전트를 사용하고, 에이전트 간의 명시적 통신 채널을 통해 동적인 모델링을 가능하게 하여 기관 간의 조정된 치료 전략을 수립합니다.

- **Technical Details**: HMARL 프레임워크는 다중 에이전트 시스템으로, 복잡한 다기관 치료 추천 과제를 관리 가능한 하위 작업으로 나누어 각 에이전트가 독립적으로 또는 협력하여 수행할 수 있도록 설계되었습니다. 또한, 다층 계층적 상태 표현 기법을 도입하여 환자 조건을 다양한 계층 수준에서 맥락화하여 치료 정확도 및 적합성을 높입니다.

- **Performance Highlights**: 광범위한 질적 및 양적 평가를 통해, 제안된 방법은 패혈증(sepsis) 관리에 효과적인 치료 정책을 학습하여 환자 생존율을 significantly 향상시키는 능력을 입증하였습니다. 본 프레임워크는 클리닉 의사결정 지원 시스템에서 다기관 치료 추천을 위한 포괄적 접근 방법을 선도하는 중요한 발전을 의미합니다.



### Towards Privacy-Preserving Relational Data Synthesis via Probabilistic Relational Models (https://arxiv.org/abs/2409.04194)
Comments:
          Accepted to the Proceedings of the 47th German Conference on Artificial Intelligence (KI 2024)

- **What's New**: 이 논문은 확률적 관계 모델(probabilistic relational models)을 기반으로 합성 관계 데이터(synthetic relational data)를 생성하는 새로운 방법론을 제안합니다. 특히, 관계 데이터베이스(relational database)에서 확률적 관계 모델(PFG, parametric factor graphs)로의 전환을 위한 완벽한 파이프라인(pipeline)을 구축하였습니다.

- **Technical Details**: 우리는 PFG를 사용하여 관계 데이터베이스에서 관계 모델을 학습하는 알고리즘을 개발하였습니다. 이 알고리즘은 관계 데이터셋에서 구조와 파라미터를 학습하여 신규 합성 관계 데이터 포인트를 샘플링하는 데 사용됩니다. PFG는 관계 도메인에서 객체 간의 관계를 효과적으로 인코딩하며, 설명 가능한 모델로서 확률 추론(probabilistic inference)과 인과 추론(causal inference)에도 활용될 수 있습니다.

- **Performance Highlights**: 본 연구는 기존의 MLN(Markov Logic Network)이나 단일 테이블의 합성 데이터 생성 연구와는 차별화되는 접근 방식을 취하며, PFG의 장점을 이용해 다중 테이블(multi-table) 합성 데이터 생성의 기초를 제공합니다. 결과적으로, 우리는 다양한 머신러닝 작업에서 활용 가능한 합성 관계 데이터를 생성하는 새로운 방법을 통해 데이터 개인 정보 보호의 우려 없이 모델 학습을 가능하게 합니다.



### Intelligent tutoring systems by Bayesian networks with noisy gates (https://arxiv.org/abs/2409.04102)
- **What's New**: 이 논문에서는 지능형 튜터링 시스템(ITS)의 구현을 위해 Bayesian nets (베이지안 네트워크)을 활용하는 방법에 대해 소개합니다. 특히, 조건부 확률 테이블의 compact parametrization 에 있어 논리 게이트를 활용하는 새로운 접근을 제안합니다.

- **Technical Details**: 지능형 튜터링 시스템에서 여러 기술(skills)의 상관관계를 효과적으로 설명하기 위해 복잡한 probabilistic 모델을 사용하는 것이 필요합니다. 이 논문에서는 noisy-OR 게이트와 같은 불확실성을 가진 논리 게이트를 통해 베이지안 네트워크의 파라미터 수를 대폭 줄이고, 이에 따른 inference 계산의 속도를 높이는 방법을 설명하고 있습니다.

- **Performance Highlights**: 제안된 기법을 통해, 파라미터 elicitation 과 inference의 복잡성을 줄여 실시간 피드백을 제공할 수 있는 효율적인 ITS 모델을 구축할 수 있음을 보였습니다.



### An Argumentative Approach for Explaining Preemption in Soft-Constraint Based Norms (https://arxiv.org/abs/2409.04065)
Comments:
          submitted to VECOMP/AICOM 2024 associated with 27th European Conference on Artificial Intelligence (ECAI2024)

- **What's New**: 이 논문은 상황 지식의 진화에 기반하여 어떻게 Preemption(우선권)이 발생하는지를 설명하기 위한 새로운 Derivation State Argumentation Framework(DSA-framework)를 제안합니다.

- **Technical Details**: DSA-framework는 논리적 제약의 계층 구조를 활용하여 Soft-constraint(연성 제약) 기반의 규범이 충돌할 때 발생하는 Preemption을 설명하는 데 중점을 둡니다. 이 프레임워크는 Argumentative Approach(논증 접근법)를 사용하여 DSA-framework를 기반으로 Argument(주장)를 구성하며, 크게 두 가지 요소인 Derivation States(유도 상태)와 Situational Knowledge(상황 지식)를 결합합니다.

- **Performance Highlights**: DSA-framework 하에, 한 결과가 Obligatory(의무적) 또는 Forbidden(금지됨)할 수 있는 이유를 공식적으로 증명할 수 있으며, 이러한 방법은 Normative Systems(규범 시스템)의 신뢰를 구축하는 데 중요한 역할을 합니다.



### Refining Wikidata Taxonomy using Large Language Models (https://arxiv.org/abs/2409.04056)
Comments:
          ACM International Conference on Information and Knowledge Management, Oct 2024, Boise, Idaho, United States

- **What's New**: Wikidata의 복잡한 분류 체계를 자동으로 정리하는 WiKC라는 새로운 버전을 제시합니다. 이 과정에서는 Large Language Models (LLMs)와 graph mining 기법을 결합하여 사용하였습니다.

- **Technical Details**: WiKC는 zero-shot prompting을 사용하여 Wikidata의 분류 체계에 대한 링크를 자르거나 클래스를 병합하는 작업을 수행합니다. 이러한 방식은 분류 체계의 자동화를 통해 수작업의 오류와 주관적인 결정을 줄입니다.

- **Performance Highlights**: 정제된 분류 체계의 품질은 내재적(intrinsic) 및 외재적(extrinsic) 관점에서 평가되었으며, 실질적인 관심을 보여주는 entity typing 작업에서 그 효과를 입증하였습니다.



### Harnessing LLMs for Cross-City OD Flow Prediction (https://arxiv.org/abs/2409.03937)
Comments:
          12 pages, 18 figures

- **What's New**: 본 논문에서는 다양한 도시들 간의 Origin-Destination (OD) 흐름 예측을 위한 새로운 방법을 제안합니다. 기존의 OD 예측 모델들이 도시마다 다양한 트래픽 상황과 사회경제적 요인으로 인해 한계를 겪는 반면, 본 연구에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 이러한 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 몇 가지 중요한 구성 요소로 이루어져 있습니다. 첫째, 출발 도시에서 OD 훈련 데이터셋을 수집하고, 둘째, LLM을 교육합니다. 셋째, 목표 도시에서의 목적지 POI(Points of Interest)를 예측하며, 마지막으로 예측된 POI에 가장 잘 맞는 위치를 식별합니다. 또한 POI 의미론과 여행 거리(Trip Distance)를 통합하는 새로운 손실 함수(Loss Function)를 도입하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해 제안된 접근 방식이 최신 기계 학습 기반 방법보다 뛰어난 성능을 보임을 입증하였습니다. 이는 도시 간 OD 흐름 예측에 있어 더 정확하고 적응 가능한 솔루션을 제공한다는 것을 의미합니다.



### NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls (https://arxiv.org/abs/2409.03797)
- **What's New**: 최근 대형 언어 모델(LLMs)을 활용한 자율 에이전트 응용 프로그램이 복잡한 실제 작업을 해결하는 효과적인 도구로 주목받고 있습니다. 본 논문에서는 LLM의 중첩 API 호출 능력을 평가하기 위해 NESTFUL이라는 새로운 벤치마크를 제시합니다.

- **Technical Details**: NESTFUL은 총 300개의 인간 주석 샘플로 구성되어 있으며, 실행 가능한(executable) 샘플과 실행 불가능한(non-executable) 샘플로 나뉩니다. 각 샘플은 사용자 요청에 대한 답변으로 API 호출 시퀀스를 포함하고 있으며, 이는 JSON 객체로 표현됩니다. 중첩 API 호출의 경우, 한 API 호출의 출력이 후속 API 호출의 입력으로 사용됩니다.

- **Performance Highlights**: NESTFUL에서의 평가 결과, 대부분의 최신 LLM 모델들이 기존 벤치마크에서의 성능에 비해 중첩 API 호출 작업에서 저조한 성능을 보였습니다. 이는 API 호출 능력의 발전을 테스트할 수 있는 유용한 길잡이가 될 것으로 기대됩니다.



### Accelerating Training with Neuron Interaction and Nowcasting Networks (https://arxiv.org/abs/2409.04434)
Comments:
          code this https URL

- **What's New**: 본 논문은 Neuron Interaction and Nowcasting (NiNo) 네트워크를 제안하여 Adam과 같은 기존 최적화 기법의 성능을 향상시키고, 네트워크의 파라미터를 보다 정확하게 예측하여 훈련 과정을 가속화합니다.

- **Technical Details**: NiNo는 신경망의 구조적 정보를 활용하여 파라미터 예측을 개선합니다. 또한 그래프 신경망(Graph Neural Networks)을 사용하여 다중 헤드 자기 주의(Multi-head Self-Attention)에서의 뉴런 순열 대칭을 더욱 정확하게 모델링합니다. 이를 통해 다양한 작업에서 예측 능력이 향상됩니다.

- **Performance Highlights**: NiNo는 ConvNets와 Transformers에서 Adam의 목표 성능을 달성하는 데 필요한 단계를 최대 50% 줄여 훈련을 가속화했습니다.



### A Survey on Knowledge Organization Systems of Research Fields: Resources and Challenges (https://arxiv.org/abs/2409.04432)
- **What's New**: 이 논문은 학문 분야에서 사용되는 지식 조직 시스템(Knowledge Organization Systems, KOSs)의 포괄적인 조사 결과를 제시합니다. 45개의 KOS를 분석하고 비교하여 범위, 구조, 관리, 사용, 다른 KOS와의 연결 등 5가지 주요 차원에 따라 차별화된 시나리오를 강조합니다.

- **Technical Details**: KOS는 용어 목록(term lists), 유의어 사전(thesauri), 분류법(taxonomies), 온톨로지(ontologies) 등으로 구성되어 있으며, 이는 연구 문서, 학술 과정, 특허, 도서, 과학적 장소 등 여러 가지 연구 제품과 관련된 항목을 분류하는 데 사용됩니다. 본 연구에서는 KOS의 범위, 구조(개념의 수, 최대 깊이, 계층 유형 등), 관리(형식, 라이센스, 업데이트 빈도 등), 다른 KOS와의 링크, 사용을 통해 KOS를 평가했습니다.

- **Performance Highlights**: 다양한 학문 분야에서 KOS가 문서 검색의 효율성을 높이고, 연구 동향을 분석하고 예측하는 데 기여한다는 점에서 성과를 보였습니다. 특히 AI(AI 기반 시스템)가 KOS를 통해 연구자들이 문헌을 탐색하고 체계적 검토를 반자동화하는 데 도움을 받고 있음이 강조되었습니다.



### Hybrid Spiking Neural Networks for Low-Power Intra-Cortical Brain-Machine Interfaces (https://arxiv.org/abs/2409.04428)
Comments:
          This work has been accepted at the 2024 IEEE Biomedical Circuits and Systems Conference

- **What's New**: 이번 연구에서는 무선 intra-cortical brain-machine interfaces (iBMIs)의 성능을 향상시키기 위해 하이브리드 스파이킹 뉴럴 네트워크(hybrid spiking neural networks)를 도입하여, 뇌 신호의 인코딩 및 디코딩을 최적화하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구의 핵심은 시간적 합성곱 기반 압축(temporal convolution-based compression), 반복 처리(recurrent processing), 및 원래 시퀀스 길이로의 보간(interpolation) 기능을 포함하는 하이브리드 네트워크 아키텍처입니다. 이 네트워크에서는 gated recurrent units (GRUs)와 leak integrate-and-fire (LIF) 뉴런을 결합한 spiking GRUs (sGRUs)를 사용하여, 정확도(accuracy), 발자국 크기(footprint), 활성화 희소성(activation sparsity) 측면에서 비교 분석합니다.

- **Performance Highlights**: 본 연구는 다채널(multi-channel) 주 운동 피질(primary motor cortex) 기록으로부터 원숭이의 움직임 속도를 높은 정확도로 예측하며, NeuroBench 프레임워크를 통해 평가된 결과 현재의 기준선 모델(baseline models)을 초월하는 성과를 달성하였습니다. 이 하이브리드 뉴럴 네트워크는 무선 iBMIs의 높은 디코딩 정밀도를 가능하게 하고, 모니터링되는 뉴런 수의 대폭 증가로 이어져 보다 발전된 신경 보철(neuroprosthetic) 기술의 길을 열 것으로 기대됩니다.



### RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs (https://arxiv.org/abs/2409.04421)
- **What's New**: 이 논문에서는 사용자 모델링과 개인화 시스템을 위한 자연어 사용자 요약을 생성하는 새로운 작업을 소개하며, RLPF라는 방법을 통해 LLM (Large Language Models)의 개인화를 향상시키려고 한다.

- **Technical Details**: RLPF(리인포스먼트 러닝 프롬 프레딕션 피드백)는 사용자 활동 데이터를 기반으로 요약 모델을 훈련하여, 다운스트림(task-specific) 예측 작업을 최적화하는 것이 특징이다. 이 방법은 세 가지 주요 구성 요소로 이루어져 있다: 1) 요약 모델, 2) 예측 기반 보상 모델, 3) 피드백 루프.

- **Performance Highlights**: RLPF는 다운스트림 작업 성능에서 최대 22%의 개선을 보였고, Factuality(사실성), Abstractiveness(추상성), Readability(가독성) 평가에서 84.59%의 승률을 기록하였다. 19개의 unseen 작업 및 데이터 세트에서 16개에서 성능이 향상되었으며, 사용자 컨텍스트 길이를 74% 감소시켰다.



### Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation (https://arxiv.org/abs/2409.04410)
- **What's New**: Open-MAGVIT2는 300M에서 1.5B까지의 오토 회귀(auto-regressive) 이미지 생성 모델 패밀리를 선보입니다. 이 프로젝트는 구글의 MAGVIT-v2 토크나이저를 오픈 소스로 재구현하며, ImageNet 256 × 256에서 새로운 최첨단 복원 성능(1.17 rFID)을 달성합니다.

- **Technical Details**: Open-MAGVIT2는 두 가지 주요 단계로 구성됩니다: 1) 강력한 비주얼 토크나이저가 시각적 신호를 이산적인 토큰 표현으로 매핑합니다. 2) 벡터 양자화된 시퀀스는 오토 회귀 트랜스포머로 전송되어 토큰 간의 관계 모델링을 통해 시각적 합성을 수행합니다. MAGVIT-v2에서는 대용량 코드북을 비대칭 토큰 분해(asymmetric token factorization)를 통해 두 개의 하위 어휘로 분리하고, 하위 토큰 상호작용을 향상시키기 위한 '다음 하위 토큰 예측(next sub-token prediction)'을 도입했습니다.

- **Performance Highlights**: Open-MAGVIT2는 강력한 토크나이저 덕분에 일반적인 오토 회귀 모델보다 뛰어난 성능과 확장성을 보입니다. 특히, 이미지넷에서 실험한 결과, 기존 방법론들을 초월하는 성능을 보여주며, 비전 지향적 디자인을 활용하는 MAGVIT-v2의 아키텍처를 넘어서는 성과를 달성했습니다.



### HiSC4D: Human-centered interaction and 4D Scene Capture in Large-scale Space Using Wearable IMUs and LiDAR (https://arxiv.org/abs/2409.04398)
Comments:
          17 pages, 10 figures, Jornal

- **What's New**: HiSC4D는 역동적 디지털 세계를 생성하기 위한 새로운 방법으로, 넓은 실내외 장면, 다양한 인간 동작 및 상호작용을 정확하고 효율적으로 포착할 수 있습니다.

- **Technical Details**: HiSC4D는 IMU(Inertial Measurement Unit) 및 LiDAR를 활용하여 드리프트(derrift) 없는 인간 동작을 캡처합니다. 여러 센서의 데이터를 통합하는 공동 최적화(joint optimization) 방법을 사용하여 대형 장면에서 안정적인 인간 모션 캡처가 가능합니다.

- **Performance Highlights**: HiSC4D는 다양한 시나리오(예: 농구 체육관, 상업 거리)에서 사람들의 상호작용을 효과적으로 포착하였으며, 4D 데이터셋은 36,000 프레임의 정확한 인간 모션을 제공합니다. 이 데이터셋은 연구 목적으로 공개될 예정입니다.



### Question-Answering Dense Video Events (https://arxiv.org/abs/2409.04388)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 새로운 작업인 dense video events의 질문-답변(task)을 제안합니다. 이 작업은 긴 비디오에서 복잡한 사건을 이해하고, 여러 시간대에 걸친 사건들에 대한 질문에 답하는 것을 목표로 합니다.

- **Technical Details**: DeVE-QA라는 데이터셋을 구성하여 10.6K 개의 긴 비디오에 대해 26K 개의 사건에 관한 78K 개의 질문을 포함하고 있습니다. 또한, 새로운 MLLM 훈련 기법인 DeVi를 제안하며, 이 방법은 계층적 캡셔닝 모듈, 시간적 사건 메모리 모듈, 자기 일관성 확인 모듈을 포함하여 긴 비디오에서의 질문-답변 성능을 향상시킵니다.

- **Performance Highlights**: DeVi는 DeVE-QA에서 4.1%의 정확도 향상과 NExT-GQA에서 3.7%의 성장을 보여줍니다. 기존 MLLMs에 비해 뛰어난 성능을 발휘하며 dense video events에 대한 질문-답변을 효과적으로 처리할 수 있습니다.



### The Impact of Scanner Domain Shift on Deep Learning Performance in Medical Imaging: an Experimental Study (https://arxiv.org/abs/2409.04368)
- **What's New**: 이 논문은 다양한 의료 이미징 모달리티(MRI, CT, X-ray)에서 스캐너 도메인 이동(scanner domain shift)이 심층 신경망의 성능에 미치는 영향을 체계적으로 연구한 최초의 다중 모달리티(multi-modality) 연구입니다.

- **Technical Details**: 의료 이미지를 다른 스캐너와 프로토콜을 사용하여 획득할 때 이미지의 특성이 달라지며, 이는 심층 신경망의 성능 저하를 초래할 수 있습니다. 특히, MRI에서 성능 저하가 가장 심한 경향을 보였고, 이는 X-ray는 중간 정도, CT는 상대적으로 적은 것으로 나타났습니다. 또한, 타겟 도메인 데이터를 학습 세트에 추가함으로써 일반화(generalization) 성능을 개선할 수 있음을 확인하였습니다.

- **Performance Highlights**: 결과적으로, 14개 세팅 중 11개 세팅에서 스캐너 도메인 이동이 성능 저하를 초래했습니다. 특히 MRI 작업에서의 도메인 이동 문제가 가장 심각하며, CT 작업은 최소한의 영향을 받는 것으로 나타났습니다.



### Provable Hyperparameter Tuning for Structured Pfaffian Settings (https://arxiv.org/abs/2409.04367)
- **What's New**: 이 논문에서는 매개변수화된(data-driven) 알고리즘 설계를 위한 새로운 이론적 프레임워크인 Pfaffian GJ 프레임워크를 도입하여, Pfaffian 함수의 불연속성이 포함된 기능 클래스에 대한 학습 보장을 제공합니다. 이 접근법은 알고리즘의 성능을 높이기 위해 알고리즘 매개변수를 문제 인스턴스에서 자동으로 조정하는 새로운 방법론을 제안합니다.

- **Technical Details**: Pfaffian GJ 프레임워크는 기존 GJ 프레임워크의 확장으로, 함수 클래스에서 Pfaffian 함수의 계산을 가능하게 하여 다양한 매개변수화된 알고리즘에 대한 학습 보장을 제공합니다. 온라인 학습 환경에서는 불손실(no-regret) 학습을 위한 새로운 도구를 제시하면서 함수의 불연속성을 검증하는 조건을 마련합니다.

- **Performance Highlights**: 이 연구에서는 매개변수화된 알고리즘이 가지는 세분화된 조각-구조(piece-wise structure)를 증명하며, 이 구조는 제안된 Pfaffian GJ 프레임워크를 사용하여 학습 보장으로 자동 변환됩니다. 이는 데이터 기반 알고리즘 설계 문제에 대한 이론적 이해를 심화시키고 전달하는 데 기여합니다.



### Connectivity-Inspired Network for Context-Aware Recognition (https://arxiv.org/abs/2409.04360)
Comments:
          ECCV 2024 - HCV Workshop, Accepted for presentation, Submitted Manuscript Version (adapted to include author names, Acknowledgements, and reference DOIs): the version of the manuscript improved after peer review will appear in the Proceedings later

- **What's New**: 이 논문은 인간의 시각 시스템에 대한 포괄적인 문헌 리뷰를 통해 AI 실무자에게 새로운 정보를 제공하고, 생물학적으로 영감을 받은 이미지 분류를 위한 신경망을 제안하며, 맥락 인식을 모델링하기 위한 새로운 플러그 앤 플레이 모듈을 제시하는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안한 신경망은 인간의 피질 및 피질 아래의 연결성에 영감을 받아 설계되었으며, 하향(top-down) 및 상승(bottom-up) 변조를 구현하여 시각 및 인지 영역 간의 복잡한 연결을 모방합니다. 우리의 Contextual Attention Block(CAB)은 매우 간단하며 모든 피드포워드 신경망에 통합될 수 있습니다.

- **Performance Highlights**: 이미지 분류 실험을 통해 제안된 방법이 성능과 class activation을 통한 설명의 강인성이 일관성 있게 개선되었음을 확인했습니다.



### Towards Fine-Grained Webpage Fingerprinting at Sca (https://arxiv.org/abs/2409.04341)
Comments:
          To appear in the Proceedings of The ACM Conference on Computer and Communications Security (CCS), 2024

- **What's New**: 본 논문에서는 Tor 클라이언트의 암호화된 트래픽 패턴을 분석하여 웹페이지를 식별하는 새로운 공격 기법인 Oscar를 제안합니다. Oscar는 전통적인 Website Fingerprinting (WF) 공격에 비해 다중 웹페이지 식별을 성공적으로 수행할 수 있는 방법론을 개발했습니다.

- **Technical Details**: Oscar는 multi-label metric learning을 기반으로 하여, 유사한 트래픽 패턴을 가진 웹페이지들 사이의 미세한 차이를 추출합니다. traffic 공간의 변환을 통해 서로 다른 웹페이지의 트래픽을 효과적으로 식별할 수 있습니다.

- **Performance Highlights**: Oscar는 1,000개의 모니터링된 웹페이지와 9,000개 이상의 비모니터링된 웹페이지에서 평가되었으며, 기존의 최첨단 공격에 비해 Recall@5에서 평균 88.6%의 개선 효과를 보여주었습니다.



### AGR: Age Group fairness Reward for Bias Mitigation in LLMs (https://arxiv.org/abs/2409.04340)
Comments:
          The first two authors contributed equally to this work. Corresponding to Zhiqiang Wang. ACKNOWLEDGMENT: we would like to thank the computing resources support from the State Key Laboratory of New Computer Software Technologies at Nanjing University

- **What's New**: 본 논문에서는 LLMs의 나이 편향(age bias)을 탐지하고 측정하기 위한 나이 편향 선호 데이터셋(age bias preference datasets)과 지시 조정 데이터셋(instruction-tuning datasets)을 구축하여 이를 해결하고자 하였습니다. 또한, 다양한 연령 그룹 간의 응답 품질 차이를 줄이기 위한 나이 공정성 보상(Age Fairness Reward, AGR)을 도입했습니다.

- **Technical Details**: 이 연구에서는 기존의 BBQ 및 ISB 데이터셋을 수정 및 확장하여 나이 관련 편향 평가를 위한 나이 선호 및 지시 조정 데이터셋을 수작업으로 주석을 달아 제작하였습니다. AGR은 다양한 LLM에서 나이 편향 완화를 위한 훈련 중 성능 차이를 줄이기 위해 공정성을 고려한 보상을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 AGR은 응답 정확성을 크게 향상시키고, 다양한 나이 그룹 간의 성능 격차를 줄이는 데 효과적임을 보여주었습니다. 실험 결과는 AGR이 기존의 관련 방법들보다 우수한 성능을 발휘함을 입증합니다.



### Learning vs Retrieval: The Role of In-Context Examples in Regression with LLMs (https://arxiv.org/abs/2409.04318)
- **What's New**: 이 논문은 Generative Large Language Models (LLMs)이 in-context learning (ICL)을 수행하는 메커니즘을 평가하기 위한 새로운 프레임워크를 제시합니다. 연구자들은 LLMs가 내부 지식을 검색하고 in-context 예제에서 학습하는 두 가지 메커니즘의 조합으로 ICL을 이해하고자 합니다.

- **Technical Details**: LLMs는 실제 데이터셋에서 회귀(regression)를 수행할 수 있는 능력을 보여주며, 실험을 통해 모델이 내부 지식을 얼마나 검색하는지와 in-context 예제에서 학습하는지의 정도를 측정합니다. 저자들은 프롬프트 엔지니어링을 통해 ICL 예제로부터 메타 학습을 활용하고 지식을 검색하는 방법을 제안합니다. 세 가지 LLMs와 여러 데이터셋을 사용하여 연구 결과의 견고성을 확인합니다.

- **Performance Highlights**: 연구 결과, LLMs는 실제 데이터셋에서 회귀 예제로부터 효과적으로 학습할 수 있으며, ICL 메커니즘을 통해 내부 지식 검색과 in-context 예제에서의 학습을 조절하는 방법론을 제시합니다. 연구자들은 이 기법이 LLM의 성능을 개선하는 데 도움이 될 것이라고 주장합니다.



### Safe and Efficient Path Planning under Uncertainty via Deep Collision Probability Fields (https://arxiv.org/abs/2409.04306)
Comments:
          Preprint version of a paper accepted to the IEEE Robotics and Automation Letters

- **What's New**: 이 논문에서는 로봇과 환경 장애물 또는 다른 이동체 간의 충돌 확률을 추정하기 위한 새로운 접근 방식인 Deep Collision Probability Fields를 소개합니다. 이 방법은 기존의 방법들이 직면하고 있는 문제들을 해결하고자 합니다.

- **Technical Details**: Deep Collision Probability Fields는 신경망(neural network)을 기반으로 하여 임의의 단일형(unimodal) 불확실성 분포를 가진 객체의 충돌 확률을 계산합니다. 이 접근 방식은 훈련 단계에서 샘플링 기반의 충돌 확률 추정을 computationally intensive한 과정에서 분리하여 계획(plan) 과정에서의 빠른 신경망 추론을 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 Deep Collision Probability Fields는 안전한 경로 계획을 위한 충돌 확률을 10^{-3}까지 정확하게 생성할 수 있음을 보여주었습니다. 이러한 방법은 불확실한 정적(static) 및 동적(dynamic) 장애물이 있는 2D 맵에 안전한 경로를 계획하기 위해 기존의 경로 계획 접근 방식에 쉽게 통합될 수 있습니다.



### CoxKAN: Kolmogorov-Arnold Networks for Interpretable, High-Performance Survival Analysis (https://arxiv.org/abs/2409.04290)
- **What's New**: 본 논문에서는 CoxKAN이라는 새로운 해석 가능한 생존 분석 프레임워크를 소개합니다. CoxKAN은 Kolmogorov-Arnold Networks (KANs)를 기반으로 하며, 해석 가능성과 고성능을 동시에 제공하는 생존 모델입니다.

- **Technical Details**: CoxKAN은 Cox 비례 위험 모델(Cox proportional hazards model)을 기초로 하며, 심볼릭 회귀(Symbolic Regression)를 활용하여 해석 가능한 기호 공식들을 찾습니다. 이 방법은 자동 특성 선택을 가능하게 하고, 느린 훈련 시간을 해결하기 위해 Cox 손실의 빠른 근사를 사용합니다.

- **Performance Highlights**: CoxKAN은 9개의 실제 의료 데이터셋과 4개의 합성 데이터셋에서 평가되었으며, CoxPH 모델보다 지속적으로 우수한 성능을 보였습니다. 또한, 기존 생존 방법으로는 인식하기 어려운 복잡한 변수 간 상호작용을 자동으로 식별할 수 있음을 보여주었습니다.



### Cycle Pixel Difference Network for Crisp Edge Detection (https://arxiv.org/abs/2409.04272)
- **What's New**: 이번 논문은 순수하게 처음부터 학습 가능한 새로운 엣지 감지 네트워크인 CPD-Net을 제안합니다. 이 네트워크는 기존의 대규모 사전 훈련된 모델에 대한 의존도를 줄이고 정밀하고 깨끗한 엣지 맵을 생성하는 데 중점을 둡니다.

- **Technical Details**: CPD-Net은 사이클 픽셀 차이 컨볼루션(Cycle Pixel Difference Convolution, CPDC)과 다중 스케일 정보 강화 모듈(Multi-Scale Information Enhancement Module, MSEM), 이중 잔차 연결 기반 디코더(Dual Residual Connection-based Decoder)로 구성된 비대칭 U자형 아키텍처입니다. CPDC는 이미지 엣지 특징을 효과적으로 인코딩하며, MSEM은 모델의 판별 능력을 향상시킵니다.

- **Performance Highlights**: 우리의 방법은 BSDS500(ODS=0.813), NYUD-V2(ODS=0.760), BIPED(ODS=0.898) 데이터셋에서 경쟁력 있는 성능을 달성했습니다. 이는 대규모 사전 훈련 없이도 우수한 결과를 보여주는 중요한 결과입니다.



### Hermes: Memory-Efficient Pipeline Inference for Large Models on Edge Devices (https://arxiv.org/abs/2409.04249)
Comments:
          Accepted by the 42nd IEEE International Conference on Computer Design (ICCD 2024)

- **What's New**: 본 연구에서는 메모리 효율적인 파이프라인 실행 메커니즘인 PIPELOAD를 소개하고, 이를 바탕으로 edge 장치에서 대형 모델 추론을 최적화한 프레임워크 Hermes를 개발했습니다.

- **Technical Details**: PIPELOAD는 동적 메모리 관리(dynamically memory management)를 통합하여 메모리 사용량을 줄이고, 병렬 모델 로딩(parallel model loading)을 통해 추론 지연(inference latency)을 최소화합니다. Hermes는 이 메커니즘을 기반으로 하며, 레이어 프로파일러(Layer Profiler), 파이프라인 플래너(Pipeline Planner), 실행 엔진(Execution Engine) 등 세 가지 주요 컴포넌트로 구성됩니다.

- **Performance Highlights**: Hermes는 BERT와 ViT 모델에 대해 최대 4.24배의 추론 속도 향상과 86.7%의 메모리 사용량 감소를 달성했으며, GPT 스타일 모델에 대해 2.58배의 속도 향상과 90.3%의 메모리 사용량 감소를 기록했습니다.



### WarpAdam: A new Adam optimizer based on Meta-Learning approach (https://arxiv.org/abs/2409.04244)
- **What's New**: 최적화 알고리즘의 선택은 딥 러닝 모델 훈련에 매우 중요합니다. 본 연구는 Meta Learning에서 'warped gradient descend' 개념을 Adam 옵티마이저에 통합하여 혁신적인 최적화 전략을 제안합니다.

- **Technical Details**: 전통적인 Adam 옵티마이저는 그래디언트를 사용하여 그래디언트 평균 및 분산의 추정을 계산하고, 이후 모델 파라미터를 업데이트합니다. 본 접근법은 학습 가능한 왜곡 행렬(P)을 도입하여 그래디언트를 선형적으로 변환합니다. 이 변환은 각 반복 중 그래디언트를 약간 조정하여 옵티마이저가 다양한 데이터셋 특성에 더 잘 적응할 수 있도록 합니다.

- **Performance Highlights**: 다양한 작업 및 데이터셋에서 실험 결과, 이 'warped gradient descend' 개념이 통합된 옵티마이저는 적응성 측면에서 우수함을 입증하였습니다. 또한, 적응 행렬 P의 효과적인 훈련 전략을 탐색하고 이 방법이 최적의 결과를 도출할 수 있는 시나리오를 찾아냈습니다.



### SPACE: A Python-based Simulator for Evaluating Decentralized Multi-Robot Task Allocation Algorithms (https://arxiv.org/abs/2409.04230)
- **What's New**: 스왐 로보틱스(Swarm Robotics) 분야의 최신 연구 성과를 바탕으로, decentralized Multi-Robot Task Allocation (MRTA) 알고리즘을 평가 및 비교하기 위한 Python 기반의 시뮬레이터인 SPACE를 소개합니다. SPACE는 사용자에게 Python 플러그인으로 의사결정 알고리즘을 구현하고, 직관적인 GUI를 사용하여 에이전트 행동 트리를 쉽게 구성하며, 에이전트 간의 통신 및 지역 작업 인식을 지원합니다.

- **Technical Details**: SPACE는 행동 트리(Behavior Trees)를 사용하는 에이전트 컨트롤러의 핵심을 구현하여 의사결정 알고리즘을 둘러싼 필수 에이전트 레벨 행동을 쉽게 개발할 수 있도록 합니다. 사용자는 YAML 설정을 통해 자신의 의사결정 알고리즘을 구성하고, 이를 플러그인 형태로 통합할 수 있습니다. 이러한 구조로 인해 SPACE는 MRTA 알고리즘의 효과적인 비교를 용이하게 합니다.

- **Performance Highlights**: SPACE를 통해 CBBA와 GRAPE 알고리즘을 구현하고 평가하였습니다. 특히 동적으로 새로운 작업이 추가되는 시나리오에서 이들의 성능을 비교하였으며, 임무 완료 시간, 이동 거리, 완료된 작업 수 등의 다양한 지표를 바탕으로 두 알고리즘의 각각의 특징을 논의하였습니다.



### GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers (https://arxiv.org/abs/2409.04196)
Comments:
          preprint

- **What's New**: 이 논문에서는 GST(Gaussian Splatting Transformer)라는 새로운 방법을 제안합니다. 이 방법은 단일 이미지에서 3D 인간 모델을 빠르게 추론할 수 있는 능력을 가지고 있으며, 기존의 확산 모델(diffusion models)이나 3D 포인트 지도(supervision)를 필요로 하지 않습니다. GST는 다중 시점(multi-view) 감독 학습을 통해 정확한 3D 포즈 추정(pose estimation)과 높은 시각적 품질을 달성합니다.

- **Technical Details**: GST는 3D Gaussian Splatting(3DGS)을 사용하여 장면을 표현합니다. 이 방법은 SMPL(Statistical Human Mesh Model) 메쉬의 정점(vertices)을 기반으로 하여 Gaussian의 초기 위치를 예측합니다. 이후 transformer 모델을 훈련시켜 이러한 위치의 작은 조정과 다른 Gaussian 속성(attributes), SMPL 파라미터를 동시에 예측합니다.

- **Performance Highlights**: GST 방식은 실험적으로 단일 이미지에서 3D 인간 모델의 빠른 추론을 가능하게 하고, 기존 방식보다 더 나은 3D 포즈 추정 및 가시적 품질을 제공합니다. 실시간(real-time) 배치와 다양한 의상 및 자세(pose)에 대한 유연성을 유지하면서도 정확한 3D 모델을 생성할 수 있습니다.



### GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding (https://arxiv.org/abs/2409.04183)
- **What's New**: GALLa (Graph Aligned Large Language Model)는 그래프 신경망(GNN)과 크로스 모달 정렬 기술을 활용하여 LLM에 코드의 구조적 정보를 주입합니다. 기존 LLM의 아키텍처를 수정하는 대신, Training 시간에만 그래프 데이터를 사용하고 Inference 시에는 비용이 발생하지 않는 모델-비가변성(task-agnostic) 구조를 제안합니다.

- **Technical Details**: GALLa는 AST와 DFG를 처리하고, 이를 LLM의 임베딩 공간으로 프로젝션하기 위해 경량 어댑터를 사용합니다. LLM은 그래프 정보를 바탕으로 소스 코드를 생성하고 그래프 구조에 대한 질문에 답하는 방식으로 훈련됩니다. 이 접근 방식은 Transform 학습 프레임워크를 따르며, 그래프 정렬 데이터와 작업별 훈련 데이터를 분리하여 LLM의 일반적인 능력을 보존합니다.

- **Performance Highlights**: GALLa는 350M에서 8B 크기의 네 가지 LLM에서 수행된 다섯 가지 코드 작업에 대한 실험을 통해 일관된 성능 향상을 보여주었습니다. 특히 GALLa는 LLaMA3와 같이 강력한 모델에서도 baseline보다 더 나은 성능을 발휘하며, 그래프 정렬 과정을 통해 얻은 구조적 지식을 새로운 프로그래밍 언어에까지 일반화하는 능력을 보여줍니다.



### The Prevalence of Neural Collapse in Neural Multivariate Regression (https://arxiv.org/abs/2409.04180)
- **What's New**: 최근 신경망에서 분류 문제의 최종 훈련 단계에서 발생하는 Neural Collapse (NC) 현상이 관찰되었습니다. 본 논문에서는 모방 학습 및 기타 응용 프로그램에서 사용되는 다변량 회귀가 Neural Regression Collapse (NRC)라는 새로운 형태의 신경 붕괴를 나타내는 것을 실증적으로 보여줍니다.

- **Technical Details**: NRC는 세 가지 주요 속성으로 구분되며: (NRC1) 마지막 층의 특징 벡터가 특징 벡터의 n개 주성분이 생성하는 부분공간으로 수렴합니다 (단일 변량 회귀의 경우 n=1); (NRC2) 마지막 층의 특징 벡터가 마지막 층의 가중치 벡터에 의해 생성된 부분공간으로 수렴합니다; (NRC3) 가중치 벡터의 Gram 행렬이 타겟의 공분산 행렬의 제곱근에 의존하는 특정 함수 형태로 수렴합니다.

- **Performance Highlights**: 본 연구는 6개의 서로 다른 데이터 세트를 활용하여 NRC1-NRC3의 유행을 입증합니다. 이 발견은 신경망 내부의 표현 단순화가 분류와 회귀 모델 모두에 걸쳐 보편적일 수 있음을 제안합니다.



### From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks (https://arxiv.org/abs/2409.04168)
- **What's New**: 이 논문에서는 기존의 LLM(judges)을 사용한 텍스트 평가 방식과는 달리, 수학적 추론 작업을 통해 LLM의 효용성을 평가합니다. 이는 다단계 추론이 필요한 작업으로, 솔루션의 정확성을 검증할 수 있어 보다 객관적인 평가를 가능하게 합니다.

- **Technical Details**: 본 연구는 네 개의 대형 LLM(30B 이상의 매개변수) 및 네 개의 소형 LLM(10B 미만의 매개변수)과 세 개의 수학적 추론 데이터셋을 이용하여 성능 분석을 수행합니다. 평가 결과, 크기가 큰 LLM이 일반적으로 더 나은 판별자가 되는 것으로 나타났지만, 대부분의 모델은 작업 성능을 개선하지 못했습니다.

- **Performance Highlights**: 연구 결과, LLM judges는 높은 품질의 모델을 선택하는 경향이 있으나, 그들의 답변이 틀려도 품질이 더 좋은 모델을 선택했습니다. 통계적 기법을 통해 개별 모델의 작업 성능을 기반으로 판단 성능을 예측할 수 있음을 보여주었습니다. 또한 입력값을 교환하거나 마스킹하는 실험을 통해 judges가 작성 스타일을 판단의 중요한 요소로 간주한다는 증거를 발견했습니다.



### Context is the Key: Backdoor Attacks for In-Context Learning with Vision Transformers (https://arxiv.org/abs/2409.04142)
- **What's New**: 이 연구는 비전 트랜스포머(ViTs)를 활용한 새로운 형태의 백도어 공격을 제안하며, 인-컨텍스트 학습(in-context learning)을 이용해 저렴한 자원으로 다수의 작업을 타겟팅 할 수 있는 방법론을 탐구합니다.

- **Technical Details**: 인-컨텍스트 학습은 모델이 명시적으로 지시받지 않고도 주어진 컨텍스트(문맥)에 따라 다양한 작업을 수행할 수 있게 해주는 기능입니다. 특히, 비전 트랜스포머를 이용한 방식에서는 제한된 샘플만으로도 공격이 가능하며, 121개의 샘플로도 백도어를 주입할 수 있음을 확인했습니다. 두 가지 새로운 공격 방법인 작업 특화(task-specific) 및 작업 비특화(task-agnostic) 백도어 공격을 제안합니다.

- **Performance Highlights**: 타겟 작업에서 최대 89.90%의 성능 저하를 기록했으며, 일반화된 공격에서는 최대 13배의 성능 저하를 달성했습니다. 전통적인 방어 기법인 프롬프트 엔지니어링과 파인튜닝(fine-tuning) 방법이 기존 백도어를 제거하는 데 효과적이지 않음을 발견했습니다.



### Confidence-Aware Document OCR Error Detection (https://arxiv.org/abs/2409.04117)
- **What's New**: 본 연구에서는 OCR(Optical Character Recognition)의 신뢰도 점수(confidence scores)를 활용하여 OCR 오류 감지를 향상시키기 위한 새로운 모델, ConfBERT를 제안합니다. 이 모델은 OCR 시스템 간의 신뢰도 점수와 오류율 간의 상관관계를 분석하고, 이러한 점수를 통합하여 오류 감지 능력을 개선할 수 있음을 보여줍니다.

- **Technical Details**: ConfBERT 모델은 BERT 기반으로, OCR 신뢰도 점수를 토큰 임베딩(token embeddings)에 통합하며, 노이즈 조정을 위한 선택적 프리트레이닝(pre-training) 단계를 제공합니다. 우리는 상업적 및 오픈소스 OCR 솔루션의 성능 및 신뢰도 점수를 비교 분석하고, 이러한 점수를 사용하여 포스트-OCR 오류 감지를 개선하는 방법을 제안합니다.

- **Performance Highlights**: 우리의 실험 결과는 OCR 신뢰도 점수를 통합함으로써 오류 감지 능력이 향상될 수 있음을 나타냅니다. 또한, 상업적 OCR 기술과 오픈소스 OCR 기술 간 성능의 상당한 차이를 강조합니다.



### Multi-Programming Language Ensemble for Code Generation in Large Language Mod (https://arxiv.org/abs/2409.04114)
Comments:
          Code available at this https URL

- **What's New**: 이번 연구에서는 Multi-Programming Language Ensemble (MPLE)라는 새로운 앙상블 기반의 방법론을 제안합니다. 이 방법은 여러 프로그래밍 언어를 활용하여 코드 생성 성능을 향상시키며, 단일 언어에서의 코드 생성에 국한되지 않습니다.

- **Technical Details**: MPLE는 각 언어별 코드 생성 프로세스를 개별 '약한 전문가 (weak expert)'로 간주하고, 그 출력을 효과적으로 통합하여 언어 특유의 오류와 편향을 완화합니다. 이 방법은 코드 생성을 개선하기 위해 reflection algorithm과 Monte Carlo tree search (MCTS) 같은 일반적인 기술과도 원활하게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, MPLE 프레임워크는 기존 벤치마크 (HumanEval 및 HumanEval-plus)에서 최대 17.92%의 성능 향상을 보여주며, HumanEval 벤치마크에서는 96.25%의 정확도로 새로운 최첨단 결과를 달성했습니다.



### Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers (https://arxiv.org/abs/2409.04109)
Comments:
          main paper is 20 pages

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 과학적 발견의 가속화 가능성에 대한 낙관론을 불러일으켰습니다. 그러나 LLM 시스템이 전문가 수준의 혁신적 아이디어를 생성하고 검증할 수 있는지를 평가한 연구는 없었습니다. 본 연구는 LLM과 전문가 NLP 연구자 간의 연구 아이디어 생성 능력을 직접 비교하여 LLM의 현재 능력에 대한 통계적으로 유의미한 결론을 도출합니다.

- **Technical Details**: 연구는 100명 이상의 NLP 전문가를 모집하여 LLM 아이디어와 인간 아이디어에 대한 블라인드 리뷰를 수행하는 실험 설계를 제시했습니다. LLM 아이디어가 인간 전문가의 아이디어보다 더 혁신적(p < 0.05)으로 평가된 반면, 실행 가능성 측면에서는 다소 약한 평가를 받았습니다. 독창성과 실행 가능성의 차이를 분석하여, LLM 기반 연구 에이전트를 구축하고 평가하는 데 필요한 문제를 식별합니다.

- **Performance Highlights**: LLM이 생성한 아이디어는 인간 전문가 아이디어에 비해 더 혁신적으로 평가되었으나 실행 가능성은 다소 낮은 점수를 받았습니다. LLM은 아이디어의 다양성 부족과 자기 평가의 한계를 보였으며, 결론적으로 연구 결과의 차이가 실제로 의미 있는지 여부를 연구할 방안을 제안합니다.



### MixNet: Joining Force of Classical and Modern Approaches Toward the Comprehensive Pipeline in Motor Imagery EEG Classification (https://arxiv.org/abs/2409.04104)
Comments:
          Supplementary materials and source codes are available on-line at this https URL

- **What's New**: 이 논문에서는 MixNet라는 새로운 분류 프레임워크를 제안하여 motor imagery (MI) 기반 뇌-컴퓨터 인터페이스 (BCI) 시스템의 분류 성능을 개선합니다. MixNet는 filter-bank common spatial patterns (FBCSP) 방법을 사용하여 MI 데이터에서 스펙트럼-공간 신호를 생성하며, multitask learning 구조인 MIN2Net를采용하여 분류를 수행합니다.

- **Technical Details**: MixNet는 MI 데이터의 스펙트럼-공간 신호를 기반으로 하며, multi-task learning의 각 과제가 서로 다른 일반화률 및 오버피팅 경향을 보인다는 문제를 해결하기 위해 adaptive gradient blending을 구현합니다. 이 기술은 여러 손실 가중치를 동시에 조절하고 각 과제의 학습 속도를 조정하여 효과적으로 최적화합니다. 또한, MixNet는 작은 데이터 세트에서도 강력한 성능을 발휘합니다.

- **Performance Highlights**: MixNet는 6개의 벤치마크 데이터 세트에서 모든 최첨단 알고리즘을 초월한 성능을 보였으며, 저밀도 EEG MI 분류의 경우에도 우수한 결과를 나타냈습니다. 이러한 성과는 IoT 응용 프로그램을 위한 경량형 및 휴대용 EEG 착용 장치 개발에 유망한 시사점을 제공합니다.



### The Role of Graph Topology in the Performance of Biomedical Knowledge Graph Completion Models (https://arxiv.org/abs/2409.04103)
- **What's New**: 이번 연구에서는 생물의학 영역의 공개된 지식 그래프(Knowledge Graph, KG)의 위상적 특성을 체계적으로 분석하고, 이러한 특성이 실제 응용에서의 예측 정확성과 어떤 연관성이 있는지를 입증합니다. 이를 통해 기존 KGE 모델들이 어떻게 성능을 개선할 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 생물의학 KG는 추상화 수준이 서로 다른 다양한 정보(예: 실험적으로 검증된 단백질-단백질 상호작용 및 유전자-질병 연관)를 포함하고 있으며, 이를 활용하여 단백질 간의 상호작용과 같은 중요한 관계를 추론합니다. 연구팀은 여섯 개의 공공 KG의 위상적 속성을 분석하고, 네 가지 Well-established KGE 모델의 예측 성능을 비교했습니다.

- **Performance Highlights**: 연구에서는 특정 관계 유형에 따라 위상적 패턴과 예측 정확성 간의 강한 연관성을 발견했습니다. KG의 위상적 특성이 예측 성능에 미치는 영향을 보다 심층적으로 탐구하며, 향후 연구자들이 활용할 수 있는 표준화된 KG 위상 기술 프레임워크를 제안합니다.



### SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation (https://arxiv.org/abs/2409.04082)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 이벤트 카메라를 위한 고속의 강인한 optical flow 추정을 위한 STTFlowNet 및 SDformerFlow라는 두 가지 솔루션을 제안합니다. 기존의 인공지능 신경망(ANN) 구조와 함께 spiking neural networks (SNNs)를 통합하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: STTFlowNet은 spatiotemporal shifted window self-attention (swin) transformer 인코더를 통한 U자 구조의 ANN 아키텍처를 채택합니다. SDformerFlow는 swin spikeformer 인코더를 통합한 완전한 스파이킹 모델이며, 두 가지 서로 다른 뉴런 모델을 갖춘 스파이킹 버전의 변형도 제공합니다.

- **Performance Highlights**: 우리의 결과는 DSEC 및 MVSEC 데이터셋에서 SNN 기반 이벤트 optical flow 방법들 중에서 최신 성능을 발휘하며, 동등한 ANN에 비해 전력 소비가 현저하게 줄어드는 것을 보여줍니다.



### UI-JEPA: Towards Active Perception of User Intent through Onscreen User Activity (https://arxiv.org/abs/2409.04081)
- **What's New**: 본 논문에서는 UI 행동 시퀀스에서 사용자 의도를 생성하는 새로운 프레임워크인 UI-JEPA를 제안합니다. 이는 자기 지도 학습(self-supervised learning, SSL)과 LLM 디코더를 결합하여, 고품질 데이터셋의 부족을 극복하고도 UI 이해 능력을 향상시킵니다.

- **Technical Details**: UI-JEPA는 레이블이 없는 UI 비디오 데이터를 활용하여 추상적인 UI 임베딩을 학습하며, LLM 디코더를 통해 사용자 의도를 예측합니다. JEPA 스타일의 목표를 활용하여 학습된 표현은 기존의 성능을 유지하면서도 데이터와 자원 등에서의 요구 조건을 크게 줄일 수 있습니다.

- **Performance Highlights**: UI-JEPA는 50.5배의 계산 비용 절감과 6.6배의 지연 시간 개선을 달성하며, intent similarity scores에서 GPT-4 Turbo와 Claude 3.5 Sonnet 보다 각각 10.0% 및 7.2% 높은 성과를 보여줍니다. 이는 가벼우면서도 높은 성능의 UI 이해가 가능하다는 것을 시사합니다.



### AnyMatch -- Efficient Zero-Shot Entity Matching with a Small Language Mod (https://arxiv.org/abs/2409.04073)
Comments:
          12 pages excluding references, 3 figures, and 5 tables

- **What's New**: 이번 연구에서는 레이블이 없는 상태에서 두 레코드가 동일한 실제 엔티티를 나타내는지를 판단하는 제로샷 엔터티 매칭(Zero-Shot Entity Matching) 문제를 해결하기 위해 AnyMatch라는 소규모 언어 모델을 제안합니다. 기존의 대형 언어 모델(LLMs)에 비해 저비용으로 높은 예측 품질을 제공함을 강조합니다.

- **Technical Details**: AnyMatch는 주어진 데이터의 레이블 없이도 효과적인 매칭을 수행하기 위해 전이 학습(Transfer Learning) 설정에서 진행됩니다. 데이터 선택 기법으로는 AutoML 필터를 통한 어려운 쌍 선택, 속성 수준의 예시 생성, 그리고 데이터 내 레이블 불균형 조정이 포함됩니다. 또한, 제로샷 엔터티 매칭을 시퀀스 분류 문제(Sequence Classification Problem)로 모델링하였습니다.

- **Performance Highlights**: AnyMatch는 아홉 개의 기준 데이터셋에서 13개의 베이스라인 모델과 비교하여 두 번째로 높은 F1 점수를 달성했습니다. 또한, 비록 파라미터 크기가 적음에도 불구하고, 많은 대형 모델을 사용하는 접근 방식보다 평균적으로 4.4%의 예측 품질을 유지하며, 3,899배 저렴한 추론 비용을 발생시킵니다.



### D4: Text-guided diffusion model-based domain adaptive data augmentation for vineyard shoot detection (https://arxiv.org/abs/2409.04060)
- **What's New**: 농업 분야에서 객체 감지 모델을 활용한 식물 표현 형상 분석(phenotyping) 기술에 관한 새로운 접근법이 제안되었습니다. 이 연구에서는 D4라는 생성적 데이터 증강 방법이 도입되었으며, 이는 적은 수의 주석이 달린 데이터셋과 비디오 데이터에서 추출한 다수의 원본 이미지를 활용하여 효과적으로 학습 데이터를 확장합니다.

- **Technical Details**: 제안된 D4 방법은 사전 학습된 텍스트 유도 확산 모델(text-guided diffusion model)을 기반으로 하여, 정의된 객체 감지를 위한 주석 정보를 유지하면서도 목표 도메인에 적합한 새로운 주석 이미지를 생성합니다. 이 과정에서 다양한 환경의 배경 정보를 반영하여 데이터 주석의 품질을 향상시킵니다. D4는 주석 가능 데이터의 부족과 도메인 다양성의 문제를 극복하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, D4 메소드는 BBox 감지 작업에서 평균 정밀도(mean average precision)를 최대 28.65% 향상시키고, 키포인트 감지 작업의 평균 정밀도를 최대 13.73% 개선하는 것으로 나타났습니다. 이러한 결과는 D4가 실제 농업 분야에서 데이터 생성 비용과 도메인 다양성 문제를 동시에 해결할 수 있는 잠재력이 있음을 보여줍니다.



### A First Look At Efficient And Secure On-Device LLM Inference Against KV Leakag (https://arxiv.org/abs/2409.04040)
- **What's New**: 이번 논문은 모바일 장치에서 LLM(대형 언어 모델)의 추론 중 발생할 수 있는 개인정보 유출 문제를 해결하기 위해 KV-Shield라는 새로운 보호 기법을 제안했습니다. 이 시스템은 KV 쌍의 무의미화와 가시성 제거라는 두 가지 접근 방식을 통해 사용자 대화의 재구성을 방지합니다.

- **Technical Details**: KV-Shield는 두 단계로 작동합니다. 초기화 단계에서는 가중치 행렬을 랜덤하게 섞어 모든 KV 쌍이 일정하게 섞이도록 합니다. 실행 단계에서는 Attention 벡터를 역으로 섞어 계층의 출력 정확성을 유지합니다. 모든 섞기 관련 작업은 TEE(신뢰할 수 있는 실행 환경) 내에서 수행되어, 불안정한 GPU가 원래의 KV 쌍에 접근하지 못하도록 합니다.

- **Performance Highlights**: FHE(완전 동형 암호화) 솔루션은 모바일 환경에서 너무 계산 집약적이어서 사용하기 어렵다는 한계를 보였지만, KV-Shield는 경량 암호화 방식인 permutation을 사용하여 성능을 유지하면서 KV 쌍에 대한 접근을 보호합니다. 이로 인해 KV 쌍이 유출되더라도 원래의 사용자 대화는 재구성할 수 없습니다.



### BFA-YOLO: Balanced multiscale object detection network for multi-view building facade attachments detection (https://arxiv.org/abs/2409.04025)
Comments:
          22 pages

- **What's New**: 본 논문에서는 다중 관점에서 건물 외관 부착물(doors, windows, balconies 등) 검출을 위한 새로운 모델인 BFA-YOLO를 제안합니다. 이 모델은 특히 기존의 YOLOv8 모델보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: BFA-YOLO 모델은 세 가지 주요 혁신으로 구성됩니다: 1) Feature Balanced Spindle Module (FBSM), 2) Target Dynamic Alignment Task Detection Head (TDATH), 3) Position Memory Enhanced Self-Attention Mechanism (PMESA). 이러한 구성 요소들은 각각 비균형한 객체 분포, 작은 객체 검출의 어려움, 그리고 배경 간섭 문제를 해결하기 위해 설계되었습니다. 또한 BFA-3D라는 새로운 데이터셋을 구축하여 다각적인 시각에서의 정확한 레이블과 다양한 카테고리를 제공합니다.

- **Performance Highlights**: BFA-YOLO는 다중 관점의 BFA-3D와 스트리트 뷰 Facade-WHU 데이터셋에서 각각 1.8% 및 2.9% 향상된 mAP@0.5 성능을 기록하며, 이는 BFA-YOLO의 우수한 성능을 나타냅니다.



### Searching for Effective Preprocessing Method and CNN-based Architecture with Efficient Channel Attention on Speech Emotion Recognition (https://arxiv.org/abs/2409.04007)
- **What's New**: 본 연구에서는 감정 음성 인식(Speech Emotion Recognition, SER) 성능 향상을 위해 정밀한 주파수-시간 해상도를 가진 여덟 가지 데이터셋 버전을 사용하여 효과적인 전처리 방법을 찾고, 효율적인 채널 주의력(Enhanced Channel Attention, ECA)을 적용한 6층 CNN 모델을 제안합니다.

- **Technical Details**: 전처리 단계에서 log-Mel 스펙트로그램을 사용하여 서로 다른 크기의 윈도우로 음성 신호를 처리하고, ECA 블록을 CNN 모델에 통합하여 채널 특성 표현을 강화했습니다. 이 방식은 필요한 학습 파라미터 수를 줄이면서 감정 분류 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, IEMOCAP 데이터셋을 활용하여 총 80.28UA, 80.46WA의 감정 인식 성능을 달성하며, 이는 기존 SER 모델의 성능을 초월하는 수치입니다. STFT 데이터 증강을 통해 다양한 전처리 설정에서 학습 가능한 데이터를 보강하여 80.37ACC의 성과를 올렸습니다.



### Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study (https://arxiv.org/abs/2409.03992)
- **What's New**: NVIDIA H100 GPU에 TEE (Trusted Execution Environment)를 활성화할 경우, 대규모 언어 모델 (LLM) 추론 작업의 성능 영향을 평가하는 보고서가 발표되었습니다. TEE 모드 사용시 CPU-GPU 데이터 전송으로 인한 병목 현상을 포함하여 성능 저하를 벤치마킹하였습니다.

- **Technical Details**: 보고서는 TEE가 활성화된 NVIDIA H100 GPU에서 LLM 추론 작업 중 발생하는 성능 오버헤드를 계량화합니다. TEE는 하드웨어 기반 보안 기능으로, CPU와 GPU 간의 데이터 전송을 암호화하여 보안을 강화합니다. TEE 모드를 활성화하면 GPU 내부 계산에는 영향을 주지 않지만, PCIe를 통한 IO에서 지연이 발생합니다. 실험은 다양한 모델, 입력 및 출력 길이, 배치 크기 설정에서 TEE 모드의 영향을 평가했습니다.

- **Performance Highlights**: TEE가 활성화되었을 때의 평균 오버헤드는 7% 미만으로 나타났으며, 모델 크기가 커질수록 오버헤드는 거의 제로에 가까워졌습니다. TPS (Tokens Per Second) 측면에서 Llama-3-8B는 130 TPS를 기록하였고, Phi-3-14B 모델은 약 6 TPS를 달성했습니다. 이는 H100 GPU의 강력한 성능을 보여줍니다.



### FODA-PG for Enhanced Medical Imaging Narrative Generation: Adaptive Differentiation of Normal and Abnormal Attributes (https://arxiv.org/abs/2409.03947)
- **What's New**: 이 논문은 의료 영상 내러티브 생성의 새로운 접근 방식인 FODA-PG(Fine-grained Organ-Disease Adaptive Partitioning Graph)를 제안합니다. 이 프레임워크는 의료 이미지에서의 정상 및 비정상 소견을 구분하여 더 정확한 임상 보고서 생성을 가능하게 합니다.

- **Technical Details**: FODA-PG는 임상적 중요성과 위치에 따라 질병 관련 속성을 '질병 특이적'과 '질병 비특이적'으로 구분하여, 세밀한 그래픽 표현을 통해 의료 영상의 미세한 차이를 포착합니다. 이는 데이터 편향의 영향을 완화하는 데도 기여합니다. 또한, 이 모델은 강력한 transformer 기반 아키텍처에 세분화된 의미적 지식을 통합하여, 높은 일반화 능력을 보입니다.

- **Performance Highlights**: IU-Xray 및 MIMIC-CXR 벤치마크에서 광범위한 실험을 통해 FODA-PG는 최첨단 방법들보다 지속적으로 우수한 성능을 보였으며, 의료 보고서 생성에서의 도메인 적응이 얼마나 중요한지를 강조합니다.



### HUMOS: Human Motion Model Conditioned on Body Shap (https://arxiv.org/abs/2409.03944)
Comments:
          Accepted in ECCV'24. Project page: this https URL

- **What's New**: 이 논문은 신체 형태에 기반한 생성적 모션 모델인 HUMOS를 제안합니다. HUMOS는 표현된 신체 형태와 성별과 같은 정체성 피쳐에 의존해 사람의 모션을 생성하는 새로운 접근 방식을 통해 기존의 통계적 데이터 중심의 모션 생성 모델의 한계를 극복하고자 합니다. 이 모델은 비디오 게임, AR/VR 및 로보틱스와 같은 다양한 응용 분야에서 사용될 수 있습니다.

- **Technical Details**: HUMOS는 transformer 기반의 조건부 Variational Auto-Encoder (c-VAE)를 사용하여 비연결 데이터(unpaired data)로부터 인간의 모션을 생성합니다. 이 모델은 동적 직관적 물리학(dynamically intuitive physics) 용어와 안정성 제약을 적용하여 다양한 신체 형태에 대한 물리적으로 그럴듯한 모션을 생성하는 데 중요합니다. 특히, 중심 질량(Center of Mass, CoM), 압력 중심(Center of Pressure, CoP) 및 제로 모멘트 포인트(Zero Moment Point, ZMP) 간의 상호작용을 모델링하는 동적 안정성 개념을 도입합니다.

- **Performance Highlights**: HUMOS는 기존 최첨단 방법들보다 정량적 및 정성적으로 더 사실적인 모션을 생성하며, 다양한 신체 형태를 고려하여 실제 사람의 모션처럼 보일 수 있도록 설계되었습니다. 특히, 복잡한 포즈와 신체 접촉을 포함하는 모션을 효과적으로 재타겟팅할 수 있는 점이 이 논문의 주요 기여 중 하나입니다.



### A deep learning approach to wall-shear stress quantification: From numerical training to zero-shot experimental application (https://arxiv.org/abs/2409.03933)
- **What's New**: 이번 연구에서는 유동 벽면과 관련된 흐름의 벽전단응력(wall-shear stress) 다이나믹스를 예측하기 위해 딥러닝 아키텍처를 도입하였습니다. 이는 실험적으로 측정 가능한 속도 필드를 기반으로 벽면 전단 응력 필드를 예측하는 새로운 접근법을 제시합니다.

- **Technical Details**: 우리는 로그층(logarithmic layer)에서의 벽 평행 속도 필드를 입력으로 사용하고, 동일한 공간 해상도(spatial resolution)와 크기를 가진 2D 벽전단응력 필드를 출력하는 딥러닝 네트워크를 개발하였습니다. 이 네트워크는 직접 수치 시뮬레이션(direct numerical simulation, DNS) 데이터를 기반으로 훈련되었습니다.

- **Performance Highlights**: 이 모델은 PIV(Particle-Image Velocimetry) 측정을 통해 얻은 실험적 속도 필드에 대해 '제로샷(zero-shot)' 적용이 가능하며, 최대 레이놀즈 수(Reynolds number) 2,000에 대해 정확한 물리적 벽전단응력 추정치를 나타냅니다.



### The Role of Generative Systems in Historical Photography Management: A Case Study on Catalan Archives (https://arxiv.org/abs/2409.03911)
Comments:
          Accepted at ECCV workshop AI4DH

- **What's New**: 이 연구에서는 역사적 자료의 설명을 위해 생성 모델의 양적 기여를 살펴봅니다. 특히 카탈루냐 아카이브의 역사적 사진 캡션을 사례 연구로 설정하여 언어적 근접성과 시각적 적응 기반의 캡션 모델 전이 학습에 대한 도구와 방향을 제공합니다.

- **Technical Details**: 본 연구는 CATR (CAption TRansformer) 모델을 사용하여 이미지 캡셔닝 작업을 수행합니다. 이 모델은 CNN과 비전 변환기(Transformer)가 결합된 구조로, 이미지 특징을 추출하고 해당 특징을 이용하여 텍스트 생성을 수행합니다. 또한, 다국적 데이터셋을 통해 모델의 훈련을 진행하여 소외 언어의 성능 향상을 목표로 합니다.

- **Performance Highlights**: 이 연구는 역사적 맥락과 언어적 특수성을 고려한 이미지 설명 생성 모델을 제안하며, 이를 통해 전통적인 사진 관리에 대한 새롭고 효율적인 접근 방식을 제공합니다. 다양한 언어의 데이터셋을 활용하여 설명의 정확하고 관련성을 높이는 방향으로 발전할 가능성을 보여줍니다.



### Multi-agent Path Finding for Mixed Autonomy Traffic Coordination (https://arxiv.org/abs/2409.03881)
- **What's New**: 도시 모빌리티의 진화하는 환경에서, 연결된 자동 차량(Connected and Automated Vehicles, CAVs)과 인간이 운전하는 차량(Human-Driven Vehicles, HDVs)의 통합이 자율주행 시스템에게 복잡한 도전과제를 제시합니다. 본 논문은 HDV의 행동 예측을 통해 CAV가 HDV와의 충돌을 피할 수 있도록 돕는 새로운 알고리즘인 행동 예측 기구 우선 탐색(Behavior Prediction Kinematic Priority Based Search, BK-PBS)을 제안합니다.

- **Technical Details**: BK-PBS는 오프라인 훈련된 조건부 예측 모델을 활용하여 CAV의 조작에 대한 HDV의 반응을 예측합니다. 이러한 예측을 바탕으로 우선 순위 기반 탐색(Priority Based Search, PBS)에서 A* 탐색 알고리즘을 사용하여 CAV의 운동 제약을 고려한 경로를 계획합니다. 이 알고리즘은 CAV와 HDV 간의 사전 상호작용을 통해 HDV의 행동을 간접적으로 조정합니다.

- **Performance Highlights**: 다양한 CAV 침투율 및 교통 밀도 시나리오에 대한 종합적인 시뮬레이션을 통해 BK-PBS는 충돌율을 줄이고 시스템 수준의 여행 지체를 개선하며 기존의 룰 기반 자동차 추종 모델 및 강화 학습 기반 알고리즘과 비교하여 우수한 성능을 보여줍니다.



### Cost-Control in Display Advertising: Theory vs Practic (https://arxiv.org/abs/2409.03874)
- **What's New**: 이번 연구에서는 광고주가 예산과 비용 제약 내에서 마케팅 목표를 달성하고자 할 때 발생하는 최적화 문제를 다룹니다. 기존의 최적 입찰 공식에서의 한계를 분석하고 이를 개선한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 전통적인 최적 입찰 공식은 온라인 방식으로 두 공간 (dual space)에서 최적 값을 가정하여 입찰을 진행하지만, 시작할 때부터 dual 변수들이 최적이 아니며 시간이 지남에 따라 수렴하게 됩니다. 이 연구에서는 이러한 비효율성 문제를 해결하기 위한 방법론을 제시합니다.

- **Performance Highlights**: 대규모 실험 결과, 제안된 수정 모델이 기존의 이론적 입찰 공식에 비해 비용 위반을 50% 줄여주는 것을 보여주어 더 나은 비용 통제를 달성했습니다.



### MetaBGM: Dynamic Soundtrack Transformation For Continuous Multi-Scene Experiences With Ambient Awareness And Personalization (https://arxiv.org/abs/2409.03844)
- **What's New**: MetaBGM은 사용자 상호작용과 동적 장면에 적응하는 배경 음악을 생성하기 위한 혁신적인 프레임워크입니다. 이는 기존의 고정된 음악 설명에 의존하던 방식을 탈피하여, 실시간으로 변하는 장면에 맞춘 음악 생성이 가능하다는 점에서 매우 독창적입니다.

- **Technical Details**: MetaBGM은 두 단계의 생성 접근법을 활용하여 지속적인 장면 및 사용자 상태 데이터를 음악 설명 텍스트로 변환합니다. 이 설명은 오디오 생성 모델로 입력되어 실시간으로 배경 음악을 생성하는 데 사용됩니다. 프로시저 내러티브 생성기법과 함께 LLM(대형 언어 모델)을 이용하여 음악 설명을 생성합니다.

- **Performance Highlights**: 실험 결과, MetaBGM은 동적이고 문맥에 맞는 배경 음악을 생성할 수 있는 능력을 보여주었습니다. 특히, Minecraft를 활용한 사례 연구를 통해, 이 모델이 사용자 상호작용과 연속적인 장면 전환을 실시간으로 효과적으로 동기화할 수 있음을 입증하였습니다.



### AI forecasting of higher-order wave modes of spinning binary black hole mergers (https://arxiv.org/abs/2409.03833)
Comments:
          27 pages, 1 appendix, 10 figures

- **What's New**: 본 논문에서는 회전하고 비진동(Non-Precessing)하는 이진 블랙홀 병합(merger)으로 방출되는 비선형 동역학(non-linear dynamics)의 더 높은 차수의 파장 모드를 예측하는 물리학 영감을 받은 transformer 모델을 제안합니다. 이 모델은 병합 전 단계에서 링다운(ringdown)까지의 파형 진화를 예측합니다. 또한, 14,440,761 개의 파형을 통해 훈련된 이 모델의 예측 정확성을 강조합니다.

- **Technical Details**: 모델은 회전하는 비진동 이진 블랙홀의 병합 시 발생하는 파형의 비선형 특성을 학습하고 예측하기 위해 고안되었습니다. 훈련 데이터셋은 질량 비율(mass ratios) Q의 범위가 [1, 8], 스핀 구성 요소(spin components) S의 범위가 [-0.8, 0.8], 모드는 L <= 4까지 포함됩니다. 특히 (5,5) 모드가 포함되며 (4,0) 및 (4,1) 모드는 제외됩니다. 초고속 슈퍼컴퓨터 Delta를 활용하여 15시간 동안 16개의 NVIDIA A100 GPU로 훈련을 마쳤습니다.

- **Performance Highlights**: 모델의 예측 정확도는 평균 0.996, 중앙값 0.997로 평가되었습니다. 840,000개의 시험 세트를 사용하여 얻은 이 수치는 기준 진실(ground truth)과 예측 파형 간의 오버랩(overlap)을 나타냅니다. 이 외에도 변환기 모델의 해석 가능성(interpretable features)에 대한 연구도 진행하였습니다.



### PARCO: Learning Parallel Autoregressive Policies for Efficient Multi-Agent Combinatorial Optimization (https://arxiv.org/abs/2409.03811)
- **What's New**: 이 논문에서는 PARCO(Parallel AutoRegressive Combinatorial Optimization)라는 새로운 접근법을 소개합니다. 이 방법은 다중 에이전트 조합 최적화 문제를 해결하기 위해 강화 학습을 사용하며, 병렬 오토회귀(autoregative) 디코딩을 활용합니다.

- **Technical Details**: PARCO는 여러 결정(decisions)을 동시에 효율적으로 디코딩할 수 있는 Multiple Pointer Mechanism을 사용하며, 우선순위 기반 충돌 처리(Priority-based Conflict Handling) 방안을 통해 이를 개선했습니다. 또한 효과적인 에이전트 협업을 위한 전문 커뮤니케이션 레이어(Communication Layers)를 설계하였습니다.

- **Performance Highlights**: PARCO는 대표적인 다중 에이전트 조합 최적화 문제인 경로(routes) 및 일정(scheduling) 문제에서 평가되었으며, 솔루션의 질과 속도 측면에서 기존의 고전적인 방법과 신경망(neural network) 기반의 기준선(baselines)과 비교할 때 경쟁력 있는 결과를 보여주었습니다.



### How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data (https://arxiv.org/abs/2409.03810)
Comments:
          Working in progress

- **What's New**: 최근 코드 지침 튜닝 데이터의 품질을 높이는 방법에 대한 관심이 증가하고 있습니다. 연구자들은 다양한 데이터셋에서 심각한 데이터 누수가 발생하고 있음을 발견하였고, 이를 해결하기 위한 효율적인 데이터 선택 전략을 제안했습니다.

- **Technical Details**: 이 논문에서는 코드 지침 데이터의 세 가지 차원인 지침 복잡성(instruction complexity), 응답 품질(response quality), 지침 다양성(instruction diversity)을 기반으로 한 데이터 정제(pruning) 전략을 통해 고품질 데이터를 선택하는 방법을 제안합니다. 최종적으로, 이러한 데이터를 토대로 LLaMA3 모델에서 파인튜닝된 XCoder 모형을 제시하며, 이를 통해 이전의 다른 모델들보다 더 적은 데이터로도 우수한 성과를 냄을 보여줍니다.

- **Performance Highlights**: XCoder는 LiveCodeBench 및 HumanEval 벤치마크에서 기존 모형과 비교했을 때 우수한 성능을 기록했습니다. 예를 들어, XCoder-8B는 40K 데이터 샘플을 사용하여 LiveCodeBench에서 43.66, HumanEval에서 54.9의 성과를 달성했습니다.



### Mpox Screen Lite: AI-Driven On-Device Offline Mpox Screening for Low-Resource African Mpox Emergency Respons (https://arxiv.org/abs/2409.03806)
Comments:
          11 Pages, 2 Figures, 3 Tables

- **What's New**: 2024 Mpox 감염병 발생의 심각성과 1b 클레이드의 출현을 배경으로, 자원 제한 환경에서 사용할 수 있는 인공지능(AI) 기반의 오프라인 스크리닝 도구가 개발되었습니다.

- **Technical Details**: YOLOv8n 기반의 딥러닝 모델이 2,700장의 이미지(각기 900장의 Mpox, 기타 피부 질환, 정상 피부 포함)에 대해 훈련되었습니다. 모델은 360장의 이미지로 검증되고 540장으로 테스트되었으며, 1,500개의 독립된 이미지로 외부 검증이 진행되었습니다. 성능 지표로는 accuracy, precision, recall, F1-score, sensitivity, specificity가 포함되었습니다.

- **Performance Highlights**: 최종 테스트 세트에서 모델은 96%의 높은 정확도를 보였고, Mpox 감지에 대해서는 93%의 precision, 97%의 recall, 95%의 F1-score를 기록하였습니다. 감지에 대한 sensitivity는 97%, specificity는 96%였습니다. 이러한 결과는 외부 검증에서도 일관성을 유지하여 모델의 강건성 및 일반화 가능성을 확인하였습니다.



### Exploratory Visual Analysis for Increasing Data Readiness in Artificial Intelligence Projects (https://arxiv.org/abs/2409.03805)
- **What's New**: 본 연구는 다양한 데이터 유형에 대한 시각적 분석 기법을 통해 인공지능 프로젝트의 데이터 준비 상태(data readiness)를 향상시키는 방법론과 그 과정에서 얻은 교훈을 제시합니다.

- **Technical Details**: 데이터 준비 수준을 향상시키기 위해 데이터와 그 사용 맥락을 이해해야 하며, 이를 위해 여러 가지 시각적 분석 기술을 활용합니다. 논문에서는 시간에 따라 변동하는 데이터(use cases involving time-varying data)와 같은 데이터 유형에 적합한 시각적 분석 기법과 데이터 준비의 측면 간의 매핑을 제공합니다.

- **Performance Highlights**: 이 연구를 통해 데이터 준비 상태를 효과적으로 향상시키는 시각적 분석 기법을 제시하고, 향후 인공지능 프로젝트에 있어 데이터의 질을 유지하고 개선하기 위한 명확한 가이드라인을 마련하였습니다.



### Protecting Activity Sensing Data Privacy Using Hierarchical Information Dissociation (https://arxiv.org/abs/2409.03796)
- **What's New**: 이 논문에서는 스마트폰과 웨어러블 기기에서 수집된 개인 정보를 효과적으로 보호하는 'Hippo'라는 새로운 시스템을 제안합니다. 기존의 방법들이 수집된 민감한 정보를 안전하게 처리하기 위해 개인 레이블을 요구하는 반면, Hippo는 이러한 필요 없이 민감한 정보를 통제할 수 있는 방법을 제공합니다.

- **Technical Details**: Hippo는 라벨이 필요 없는 latent guidance 기반의 diffusion model을 사용하여 민감한 메타데이터와 다중 해상도의 활동 정보를 분리합니다. 이 시스템은 원시 센서 데이터에서 사용자에게 원치 않는 민감한 정보를 제거하여 개인정보를 보호하면서도 애플리케이션의 기능 요구사항을 충족시킵니다.

- **Performance Highlights**: Hippo는 개인 정보 추론 확률을 50%까지 줄일 수 있으며, 원시 데이터와 동등한 수준의 활동 인식 정확도를 유지합니다. 또한, 사용자는 pedometer 앱을 통해 데이터의 해상도를 조정하고 불필요한 정보 유출을 방지할 수 있습니다.



### Security Implications and Mitigation Strategies in MPLS Networks (https://arxiv.org/abs/2409.03795)
- **What's New**: 이번 논문에서는 Multiprotocol Label Switching (MPLS) 네트워크의 보안 문제를 다루며, 공격 유형 및 취약점에 대한 심층적인 분석과 함께 수학적 모델을 활용한 완화 전략을 제시합니다.

- **Technical Details**: MPLS의 기본 구성 요소로 Label Switch Routers (LSRs), Label Edge Routers (LERs), Label Distribution Protocols가 있으며, 각 요소의 기능 및 상호 작용을 설명합니다. 또한, 패킷 포워딩을 그래프 이론을 통해 모델링하여 공격 시나리오를 분석합니다. Label spoofing, Traffic interception, Denial of Service (DoS) 공격과 같은 보안 위협을 다루며, 이를 위한 수학적 분석 방법을 제시합니다.

- **Performance Highlights**: 논문에서는 보안 공격을 예방하는 다양한 완화 방법, 예를 들어 라벨 인증 및 필터링 메커니즘을 제안하면서 MPLS 네트워크의 보안을 강화할 수 있는 방안을 제시합니다. 이론적 분석과 실용적 솔루션을 통합하여, MPLS 네트워크 안전성을 높이는 데 기여하고자 합니다.



### Safeguarding AI Agents: Developing and Analyzing Safety Architectures (https://arxiv.org/abs/2409.03793)
- **What's New**: 본 논문은 AI 에이전트의 안전성을 강화하기 위한 세 가지 프레임워크를 제안하고 평가합니다. 특히 LLM(대규모 언어 모델)을 통한 입력-출력 필터, 시스템 내 안전 에이전트의 통합, 안전 검사를 내장한 계층적 위임 기반 시스템을 포함합니다.

- **Technical Details**: 제안된 세 가지 안전 프로토콜은 다음과 같습니다: 1) LLM 기반 입력-출력 필터, 2) 시스템에 통합된 안전 에이전트, 3) 내장 안전 검사가 있는 계층적 위임 시스템. 이 방법론은 불안전한 에이전트 사용 사례에 대해 프레임워크를 구현하고 테스트하는 것을 포함하며, AI 에이전트 배치 시 직면할 수 있는 위험을 완화하는 데 효과적임을 평가합니다.

- **Performance Highlights**: 연구 결과, 제안된 프레임워크는 AI 에이전트 시스템의 안전성과 보안을 크게 강화할 수 있으며, 잠재적인 해로운 행동이나 출력을 최소화하는 데 기여함을 보여주었습니다. 또한, 이 연구는 자동화된 작업 및 실제 응용 프로그램에서 AI 에이전트의 책임 있는 사용을 보장하기 위한 강력한 가드레일 개발의 기초가 됩니다.



### BreachSeek: A Multi-Agent Automated Penetration Tester (https://arxiv.org/abs/2409.03789)
Comments:
          7 pages, 6 figures

- **What's New**: BreachSeek은 AI 기반의 멀티 에이전트 소프트웨어 플랫폼으로, 기존의 전통적인 침투 테스트 방법의 한계를 극복하고, 다양한 시스템에서 신속하게 취약점을 식별하고 활용할 수 있는 자동화된 솔루션을 제공합니다.

- **Technical Details**: 이 시스템은 LangChain과 LangGraph를 통해 통합된 Large Language Models (LLMs)를 활용하여, 자율 에이전트들이 포괄적인 침투 테스트를 수행하도록 지원합니다. 여러 개의 AI 에이전트를 사용하여 각기 다른 작업을 관리하며, 이를 통해 복잡한 작업의 효율성과 정확성을 높입니다.

- **Performance Highlights**: 초기 평가에서 BreachSeek은 로컬 네트워크 내에서 취약한 머신을 성공적으로 활용하여, 실질적인 효용성을 입증하였습니다. 향후 발전 방향으로는 다양한 환경에서 사용할 수 있는 능력을 확장하는 것입니다.



### HSF: Defending against Jailbreak Attacks with Hidden State Filtering (https://arxiv.org/abs/2409.03788)
Comments:
          13 pages

- **What's New**: 본 연구에서는 Hidden State Filter (HSF)를 제안하여 LLM(jailbreak 공격)으로부터의 방어를 위한 새로운 접근 방식을 소개합니다. 이 방식은 기존의 방어 방법들과 달리, 모델의 추론 과정 이전에 적대적 입력을 식별하고 거부할 수 있는 아키텍처 기반의 방어 메커니즘입니다.

- **Technical Details**: HSF는 LLM의 최종 Decoder Layer에서 마지막 k tokens로부터 특징(feature)을 샘플링하여 이를 경량의 분류 모델로 학습합니다. 이는 플러그인 모듈로 통합되어, LLM의 기본 아키텍처를 변경하지 않으면서 기존 LLM에 쉽게 통합이 가능합니다.

- **Performance Highlights**: 실험 결과, HSF는 여섯 가지 최신 jailbreak 공격에 대한 방어 성능을 크게 향상시켰으며, benign 쿼리에 대한 응답에는 최소한의 영향을 미치고, 적은 계산 자원으로 높은 효과성을 보여주었습니다. HSF는 모든 기준선을 초과하여 낮은 오탐률과 함께 운영되었습니다.



### VERA: Validation and Evaluation of Retrieval-Augmented Systems (https://arxiv.org/abs/2409.03759)
Comments:
          Accepted in Workshop on Evaluation and Trustworthiness of Generative AI Models, KDD 2024

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 평가를 위한 새로운 프레임워크 VERA (Validation and Evaluation of Retrieval-Augmented Systems)를 소개합니다. VERA는 대형 언어 모델(LLMs)에서 활용되는 인출 정보의 출력을 투명하고 신뢰성 있게 향상시키기 위한 방법을 제안합니다.

- **Technical Details**: VERA는 RAG 시스템의 평가 방식을 두 가지 주요 방식으로 개선합니다: 첫째, 여러 차원 지표를 통합하여 단일 종합 점수를 생성하는 크로스 인코더 기반의 메커니즘을 도입합니다. 둘째, 문서 저장소의 LLM 기반 지표에 대한 부트스트랩 통계를 활용하여 신뢰 구간을 설정하고, 저장소의 주제적 범위를 보장합니다. 이를 통해 RAG 시스템의 정보 검색 및 생성 과정에서 신뢰성과 적합성을 평가합니다.

- **Performance Highlights**: 여러 사례를 통해 VERA가 AI 애플리케이션의 의사결정 과정 및 신뢰성을 어떻게 강화하는지를 보여줍니다. VERA는 대형 언어 모델 기반의 RAG 평가 지표 이론적 이해에 기여할 뿐만 아니라, 책임감 있는 AI 시스템의 실질적인 구현을 촉진하여 신뢰할 수 있고 투명한 생성적 AI 기술 개발에 중대한 발전을 이룹니다.




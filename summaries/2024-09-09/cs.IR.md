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




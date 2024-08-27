New uploads on arXiv(cs.CL)

### Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models (https://arxiv.org/abs/2408.14470)
Comments:
          15 pages, 7 tables, 9 figures

- **What's New**: 본 논문에서는 기존의 parameter-efficient fine-tuning (PEFT) 기법의 한계를 극복하기 위해, $	ext{ID}^3$라는 새로운 선택적 PEFT 방법을 소개합니다. 이 방법은 매개변수 중요성을 동적으로 평가하고 탐색( Exploration)과 활용( Exploitation)을 균형있게 조절하여 매개변수를 선택합니다.

- **Technical Details**: $	ext{ID}^3$는 고정된 매개변수 세트를 사용하는 전통적인 PEFT 방법과 달리 계속해서 매개변수 중요성을 계산하며, 매개변수 선택 시 탐색과 활용을 조화롭게 합니다. 기존의 정적(static) 및 반복적(repeat) 선택 전략의 단점을 보완하며, gradient 업데이트 수를 2배 줄이고 연산 효율성을 높이는 특징을 갖습니다.

- **Performance Highlights**: $	ext{ID}^3$는 15개의 자연어 이해 및 생성 작업에서 기존의 고정 마스킹 PEFT 기술과 비교하여 우수한 성능을 보였습니다. GLUE 벤치마크에서 103K의 예산을 사용했을 때, $	ext{ID}^3$는 pre-trained DeBERTa-v3 모델로 다른 PEFT 기준보다 1.2% 높은 성과를 달성하였고, 0.17%의 훈련 가능한 매개변수만으로도 완전 미세 조정된 모델에 비해 0.3%의 성능 향상을 보여주었습니다.



### Explicit Inductive Inference using Large Language Models (https://arxiv.org/abs/2408.14467)
- **What's New**: 본 논문에서는 LLMs(Large Language Models)가 갖는 바람직하지 않은 attestation bias를 이용한 Explicit Inductive Inference 파이프라인을 제안합니다. 이를 통해 LLMs의 추론 성능을 개선하고, attestation bias의 부정적인 영향을 완화하고자 합니다.

- **Technical Details**: 제안된 파이프라인은 주어진 premise P를 attested한 여러 대안으로 변형한 후, 새로운 entailment 문의 응답을 집계하여 원래의 추론 예측을 지원합니다. 이를 통해 directional predicate entailment benchmark에서 LLMs의 성능을 향상시킬 수 있습니다. 주요 핵심 아이디어는 premise를 다양한 attested alternatives로 변환하여 LLM의 예측 결과를 모으는 것입니다.

- **Performance Highlights**: 실험 결과, 이 파이프라인은 LLMs의 전반적인 predicate inference 성능을 향상시킬 뿐만 아니라, attestation bias에 대한 저항성을 높이는 데 기여합니다.



### Evaluating Large Language Models on Spatial Tasks: A Multi-Task Benchmarking Study (https://arxiv.org/abs/2408.14438)
- **What's New**: 이번 연구는 자연어 이해부터 코드 생성까지 다양한 기능을 평가하는 대형 언어 모델의 성능에 대한 평가의 중요성을 강조하며, 특히 공간 작업에 대한 종합적인 평가가 부족하다는 점을 해결하기 위해 새로운 다중 작업 공간 평가 데이터세트를 소개합니다.

- **Technical Details**: 제안된 데이터세트는 공간적 이해와 경로 계획을 포함하여 12가지의 다양한 작업 유형으로 구성되어 있으며, 각 작업에 대한 검증된 정확한 정답을 포함하고 있습니다. 연구팀은 OpenAI의 gpt-3.5-turbo, gpt-4o, ZhipuAI의 glm-4 모델을 대상으로 두 단계의 테스트 접근 방식을 통해 성능을 평가했습니다. 첫 단계로는 zero-shot 테스트를 실시하고, 이후 데이터세트를 난이도별로 분류하여 프롬프트 튜닝 테스트를 진행했습니다. 또한, Chain-of-Thought(COT) 전략을 적용하여 특정 작업에서 성능 향상을 측정했습니다.

- **Performance Highlights**: 결과적으로, gpt-4o가 첫 번째 단계에서 평균 71.3%의 정확도로 가장 높은 전체 정확도를 기록했습니다. moonshot-v1-8k는 전반적인 성능에서 살짝 떨어졌으나 장소명 인식 작업에서는 gpt-4o를 초과하는 성과를 보였습니다. 특정 작업에 대한 모델 성능에 영향을 미치는 프롬프트 전략의 중요성도 강조되었으며, COT 전략을 통해 gpt-4o의 경로 계획 정확도를 12.4%에서 87.5%까지 향상시켰습니다.



### MEDSAGE: Enhancing Robustness of Medical Dialogue Summarization to ASR Errors with LLM-generated Synthetic Dialogues (https://arxiv.org/abs/2408.14418)
- **What's New**: 본 논문은 임상 대화 요약에서 사용되는 노이즈 저항성을 높이기 위해 대규모 언어 모델(LLM)을 이용하여 합성 데이터를 생성하는 MEDSAGE라는 새로운 방법을 제안합니다. 이 방법은 기존의 ASR 시스템의 오류를 시뮬레이션하여 데이터 증강을 가능하게 합니다.

- **Technical Details**: MEDSAGE는 LLM의 맥락 내 학습(in-context learning) 기능을 활용하여 몇 가지 예제를 기반으로 ASR과 유사한 오류를 생성합니다. 이를 통해 실제 ASR 오류 패턴을 반영하는 합성 대화를 생성하고, 오류 유형에 맞춤화된 태그 구문을 도입하여 데이터 증강을 시도합니다.

- **Performance Highlights**: 실험 결과, LLM을 활용하여 생성된 합성 데이터가 ASR 오류를 효과적으로 반영하며, 이러한 합성 데이터를 훈련에 포함시킴으로써 조용한 테스트 세트에서 성능이 최대 16% 향상됨을 보여줍니다.



### Language-specific Calibration for Pruning Multilingual Language Models (https://arxiv.org/abs/2408.14398)
- **What's New**: 본 논문은 다국어 언어 모델(Multilingual Language Models)의 프루닝(pruning) 과정에서 칼리브레이션(calibration) 언어의 효과적인 전략을 탐구합니다. 주목할 만한 점은 대부분의 연구가 영어 데이터에 기반하고 있다는 점에서, 다양한 언어 간 차이를 고려한 최초의 종합적 실험 연구라는 것입니다.

- **Technical Details**: 모델 프루닝을 위한 다양한 칼리브레이션 언어의 성능을 비교하며, 7개 언어(아랍어, 독일어, 영어, 스페인어, 러시아어, 스와힐리어, 중국어)에 대해 실험하였습니다. 사용된 주 기술로는 프루닝을 위한 두 가지 최신 기법인 Wanda와 SparseGPT를 채택했습니다. 실험에서는 각 언어에 대해 128개의 샘플을 기반으로 데이터 세트를 구성했습니다.

- **Performance Highlights**: 목표 언어로 칼리브레이션할 경우 낮은 perplexity를 유지하지만, 하위 작업에서 최적 성능을 보장하지는 않습니다. SparseGPT는 Llama-3 8B 모델에 적합하나, 다른 모델과 작업에서는 mixed performance를 보였습니다. 이는 지식 저장 및 검색 과정에서 다국어 모델이 상당한 영향을 받는다는 것을 보여줍니다.



### Probing Causality Manipulation of Large Language Models (https://arxiv.org/abs/2408.14380)
- **What's New**: 이 논문은 LLMs(대형 언어 모델)의 인과성 조작을 탐색하기 위한 새로운 계층적 접근 방식을 제안합니다. 다양한 단축키를 제공하여 모델의 행동을 관찰하는 방식입니다.

- **Technical Details**: 우리는 RAG(검색 증강 생성) 및 ICL(컨텍스트 학습)을 활용하여 설계된 인과성 분류 작업에서 LLMs의 성능 변화를 관찰합니다. 이를 통해 LLMs가 인과적으로 관련된 개체를 감지하고 직접적인 인과 관계를 인식할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, LLM들은 인과성과 관련된 개체를 인식할 수 있으나, 인과 관계에 대한 전문적인 인지가 부족하여 구문 내에서 이들을 단순히 전반적인 의미로 취급합니다. 이는 LLMs의 훈련 과정에서 인과성에 대한 추가적인 주의가 필요함을 보여줍니다.



### Assessing Contamination in Large Language Models: Introducing the LogProber method (https://arxiv.org/abs/2408.14352)
- **What's New**: 이번 논문에서는 LogProber라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 Large Language Models (LLMs)에서 contamination(오염)을 감지하기 위한 도구로, 특정 문장에서 token probability를 활용하여 오염 여부를 평가하는 방식으로 작동합니다.

- **Technical Details**: LogProber는 질문/답변 형식의 짧은 텍스트에 적합하도록 설계되었습니다. 기존 방법들이 noise로 인해 오염을 탐지하기 어려운 짧은 질문들에 대해 효과적으로 작동할 수 있도록 개발되었습니다. 실험을 통해 LLM (LLAMA)에 최근 발표된 인지 테스트 항목을 적용하여 이 알고리즘이 오염을 감지하는 데 효과적임을 보여주었습니다. 그러나 모델의 학습 방식에 따라 모든 유형의 오염을 감지할 수 없음을 발견했습니다.

- **Performance Highlights**: LogProber는 LLM의 오염 여부를 감지하는 데 있어 저렴한 계산 비용과 높은 효율성을 자랑합니다. 그러나 특정 실험에서는 질문 텍스트가 아닌 답변 텍스트에만 맞춘 모델에 대해서는 오염을 인식하지 못했습니다. 이는 모든 오염 유형이 token probability만으로 탐지될 수 없다는 점을 강조합니다.



### Claim Verification in the Age of Large Language Models: A Survey (https://arxiv.org/abs/2408.14317)
- **What's New**: 이 논문은 최근의 LLM(대형 언어 모델)을 활용한 클레임 검증(claim verification) 프레임워크에 대한 종합적인 조사를 제시하고 있습니다. 기존의 연구들과는 달리 LLM 기반 접근 방식에 주목하며, 클레임 검증 파이프라인의 다양한 구성 요소를 세부적으로 설명합니다.

- **Technical Details**: 클레임 검증 프로세스는 클레임 탐지(claim detection), 증거 검색(evidence retrieval), 진위 예측(veracity prediction)으로 구성됩니다. LLM이 도입되면서 기존 NLP 모델 대비 Misinformation 생성 및 검증에서 더 낮은 오류 전파를 보이며, 더 설명 가능하고 해석 가능한 결과를 제공합니다. RAG(Retrieval Augmented Generation) 기술을 통해 최신 정보에 접근하여 모델의 의사 결정에 도움을 줍니다.

- **Performance Highlights**: LLM 모델은 이전의 수작업 또는 기존 방법론보다 높은 성능과 신뢰도를 자랑하지만, 여전히 허위 정보 생성 가능성(hallucination) 문제로 인해 정확한 진위 라벨을 생성하는 데 한계가 있습니다. 최근 연구들은 자동화된 클레임 검증에서 LLM의 효용성을 보여주며, 앞으로의 연구 방향을 제시하고 있습니다.



### LLM-3D Print: Large Language Models To Monitor and Control 3D Printing (https://arxiv.org/abs/2408.14307)
- **What's New**: 이 연구는 Fused Deposition Modeling (FDM)에서 발생하는 인쇄 결함을 자동으로 감지하고 수정하기 위해 사전 훈련된 Large Language Models (LLMs)를 활용하는 새로운 프로세스 모니터링 및 제어 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 인쇄 품질을 평가하기 위해 각 층 또는 인쇄 세그먼트 후 캡처된 이미지를 분석하며, 실패 모드를 식별하고 관련 파라미터를 쿼리합니다. LLM은 감지된 결함에 대해 교정 조치 계획을 생성하고 실행합니다.

- **Performance Highlights**: 테스트 결과, LLM 기반의 시스템은 3D 인쇄에서 발생하는 일반적인 오류를 정확히 식별하고, 실패를 유발하는 파라미터를 효과적으로 판단하여 인간 개입 없이도 이를 자율적으로 수정할 수 있음을 보여주었습니다.



### Predictability and Causality in Spanish and English Natural Language Generation (https://arxiv.org/abs/2408.14283)
- **What's New**: 본 논문은 영어 외 다른 언어에서 NLG(자연어 생성) 시스템의 성능을 비교하기 위해 causal(원인적) 및 non-causal(비원인적) 언어 모델링을 연구합니다. 특히 스페인어와 영어를 비교하며, 각 언어의 문법 구조에 따른 생성 편향을 검토합니다.

- **Technical Details**: 이 연구에서는 정보 이론적 접근 방식으로 causal과 non-causal의 조건부 엔트로피를 비교하여 두 언어의 문법 카테고리 분포를 평가합니다. 스페인어에서는 평균 non-causal 조건부 엔트로피가 낮고, 영어에서는 평균 causal 조건부 엔트로피가 낮습니다. 이러한 결과는 스페인어가 비원인적 맥락에서 예측 가능성이 더 높다는 것을 나타냅니다.

- **Performance Highlights**: 실험 결과에 따르면 영어에서는 causal NLG가, 스페인어에서는 non-causal NLG가 최상의 성능을 발휘했습니다. 이 결과는 bidirectional(양방향) transformer 언어 모델을 사용하는 스페인어 NLG 연구의 필요성을 지지합니다.



### Self-supervised Speech Representations Still Struggle with African American Vernacular English (https://arxiv.org/abs/2408.14262)
Comments:
          INTERSPEECH 2024

- **What's New**: 최근 Self-Supervised Learning (SSL) 기반의 연설 모델들이 AAVE(재미 흑인 방언)와 MAE(주류 미국 영어) 간의 ASR(자동 음성 인식) 성능 차이를 해소할 수 있는지를 조사했습니다.

- **Technical Details**: 본 연구에서는 wav2vec 2.0, HuBERT, WavLM, XLS-R의 네 가지 SSL 모델을 활용하여 AAVE와 MAE에서의 제로샷 ASR을 평가하였으며, AAVE에 대한 성능 편견이 지속됨을 발견했습니다.

- **Performance Highlights**: 모델은 AAVE의 음운적(phonological) 및 형태 통사적(morphosyntactic) 특징을 포함한 발화에서 더 높은 단어 오류율(word error rate)을 보였습니다. 이 결과 SSL 사전 학습(pre-training)만으로는 AAVE와 MAE 간의 격차를 줄일 수 없음을 시사합니다.



### DSTI at LLMs4OL 2024 Task A: Intrinsic versus extrinsic knowledge for type classification (https://arxiv.org/abs/2408.14236)
Comments:
          8 pages, 4 figures, accepted for the LLMs4OL challenge at the International Semantic Web Conference (ISWC) 2024

- **What's New**: 본 연구에서는 semantic towers라는 외부 지식 표현 방법을 소개하고, 이를 대규모 언어 모델에서의 내재적 지식과 비교하여 본다. 연구의 결과는 외부 지식 기반의 모델이 성능과 의미적 기반 간의 트레이드오프를 보임을 발견하였다.

- **Technical Details**: 경량화된 언어 모델(flT5-small 클래스)을 사용하여 WordNet과 GeoNames 데이터셋에서의 태스크 A를 포함한 실험을 수행하였다. 이 연구에서 제안하는 semantic towers는 domain semantic primitive를 활용하여 구축되며, Wikidata Query Service를 통해 각각의 도메인에서 의미 정보를 수집하여 구축된다. 최종적으로 이러한 primitive들은 벡터 임베딩으로 변환되어 MongoDB에 저장된다.

- **Performance Highlights**: 실험 결과, WN1 및 GN1 모델은 각각 WN2 및 GN2에 비해 약 10%의 성능 향상을 보였다. 그러나 외부 지식 기반인 semantic towers의 사용이 모델의 잘못된 응답을 초래할 수 있지만, 특정 의미 개념을 효과적으로 정립할 수 있음을 발견하였다.



### Investigating the effect of Mental Models in User Interaction with an Adaptive Dialog Agen (https://arxiv.org/abs/2408.14154)
Comments:
          submitted to COLING 2025

- **What's New**: 본 연구는 정보 탐색 대화 시스템과의 상호작용에서 사용자의 정신 모델이 어떻게 형성되고 영향을 미치는지를 탐구합니다. 특히, 적응형 대화 에이전트의 행동을 사용자의 정신 모델에 맞춤으로써 성공적인 상호작용을 이끌어낼 수 있음을 보여줍니다.

- **Technical Details**: 연구진은 비즈니스 여행 도메인에서 세 가지 유형의 작업 지향 대화 시스템을 구현하였고, 66명의 참가자를 모집하여 사용자 평가를 실시했습니다. 사용자들은 개별적으로 적응형 대화 에이전트와 상호작용하며, 대화 전후에 그들의 정신 모델을 조사하는 방식으로 연구가 진행되었습니다.

- **Performance Highlights**: 적응형 대화 에이전트와의 상호작용은 사용자의 정신 모델을 향상시킬 수 있으며, 이는 시스템의 인지적 유용성과 대화 성공률을 높이는 데 기여함을 나타냅니다. 연구 결과, 사용자의 정신 모델이 대화 시스템과의 상호작용에 결정적인 영향을 미친다는 점이 강조되었습니다.



### Crowd-Calibrator: Can Annotator Disagreement Inform Calibration in Subjective Tasks? (https://arxiv.org/abs/2408.14141)
Comments:
          Accepted at COLM 2024

- **What's New**: 이 논문에서는 자연어 처리(NLP)에서 주관적인 작업을 다룰 때 모델이 인간의 결정 프로세스를 반영할 수 있도록 하는 새로운 방법, Crowd-Calibrator를 제안합니다.

- **Technical Details**: Crowd-Calibrator는 집단 작업자의 레이블 분포와 모델의 레이블 분포 간의 거리를 모델링합니다. 이를 통해 모델이 결정을 내릴지 아니면 기권할지를 판단합니다. 또한, 소프트 레이블(soft labels)을 사용하여 모델이 주관적인 태스크에서 인간 레이블의 분포를 학습할 수 있도록 하고, 불일치가 큰 경우 모델이 예측을 기권하도록 활용합니다.

- **Performance Highlights**: Crowd-Calibrator는 증오 표현 탐지(hate speech detection)와 자연어 추론(natural language inference) 두 가지 주관적인 태스크에서 기존의 선택적 예측(selective prediction) 기준과 비교했을 때 경쟁력 있는 성능을 보였습니다. 특히, 자연어 추론 태스크에서 기존 방법들을 초월하는 성능을 달성했습니다.



### Multi-Faceted Evaluation of Modeling Languages for Augmented Reality Applications -- The Case of ARWFML (https://arxiv.org/abs/2408.14137)
Comments:
          Accepted manuscript for the 43rd International Conference on Conceptual Modeling Conceptual Modeling, AI, and Beyond 28-31 October 2024 | Pittsburgh, Pennsylvania, USA

- **What's New**: 이 논문은 증강 현실(AR) 애플리케이션을 위한 모델링 언어의 평가에 대한 새로운 접근 방식을 제시합니 다. 특히, 증강 현실 워크플로우 모델링 언어(ARWFML)의 디자인 과정을 다루며, 3차원 표기법의 도입과 새로운 3D 모델링 환경을 포함한 세 가지 설계 주기를 통해 언어를 정제했습니다.

- **Technical Details**: ARWFML은 복잡한 증강 현실 워크플로우를 모델링할 수 있도록 설계된 도메인 특화 시각 모델링 언어입니다. 이 언어는 AR 환경을 정의하기 위한 ObjectSpace 모델, AR 워크플로우를 정의하기 위한 FlowScene 모델 및 워크플로우 내 가상 물체의 변화를 정의하기 위한 Statechange 모델을 포함합니다. 이러한 ARWFML 모델은 AR 실행 엔진에 의해 해석되어 AR 경험을 실행합니다.

- **Performance Highlights**: 이 논문은 ARWFML의 언어 개념에 대한 이해도를 평가하기 위한 실증 연구 결과를 포함하고 있으며, 이는 다른 유사 접근 방법과 비교하여 ARWFML이 어떻게 진화할 수 있는지를 보여줍니다. 또한, 이 연구는 ARWFML의 기술적 특성을 다른 AR 모델링 언어 구현과 비교하고 그 워크플로우 기능에 대한 비교 평가를 제공합니다.



### Contrastive Learning Subspace for Text Clustering (https://arxiv.org/abs/2408.14119)
- **What's New**: 이 논문에서는 텍스트 클러스터링을 위해 클러스터별 관계를 모델링하는 새로운 접근법인 Subspace Contrastive Learning (SCL)을 제안합니다. 기존의 대조 학습 기반 텍스트 클러스터링 기법은 인스턴스 간 의미적 유사성 관계만 모델링하였으나, SCL은 이러한 제한을 극복합니다.

- **Technical Details**: SCL은 두 가지 주요 모듈로 구성되어 있습니다: (1) 가상 긍정 샘플을 구성하는 자기 표현(self-expressive) 모듈과 (2) 텍스트 간의 클러스터별 관계를 캡처하기 위해 더 효과적인 하위 공간(discriminative subspace)을 학습하는 대조 학습 모듈입니다.

- **Performance Highlights**: 실험 결과, SCL 방법은 여러 텍스트 클러스터링 데이터셋에서 기존의 최첨단 방법보다 우수한 성능을 보이는 동시에 긍정 샘플 구성의 복잡성을 줄였습니다.



### Enhancing Depression Diagnosis with Chain-of-Thought Prompting (https://arxiv.org/abs/2408.14053)
- **What's New**: AI 모델이 우울증 징후를 감지하는 과정에서 Chain-of-Thought(코드 설명) 프롬프팅을 사용함으로써 환자의 PHQ-8 점수를 더욱 정확하게 예측할 수 있다는 새로운 연구 결과를 발표합니다.

- **Technical Details**: 모델은 DAIC-WOZ 데이터셋을 사용하여 우울증 증상에 대한 훈련을 진행하였고, CoT 프롬프팅을 접목하여 PHQ-8 점수를 보다 신뢰성 있게 평가하였습니다. 실험에서는 OpenAI 3.5 turbo 모델을 사용했으며, 제어군과 실험군의 평가를 통해 CoT 사용 시 더 정답에 가까운 결과를 도출하였습니다.

- **Performance Highlights**: CoT 프롬프팅을 사용한 결과, 모든 행동 범주에서 CoT를 활용한 Assigner B의 PHQ-8 점수는 수신자의 실제 PHQ-8 점수와 평균적으로 더 가까운 결과를 보였습니다. 이는 CoT 프롬프팅이 AI 모델의 결정을 개선하는 데 중요한 역할을 함을 시사합니다.



### Empowering Low-Resource Language ASR via Large-Scale Pseudo Labeling (https://arxiv.org/abs/2408.14026)
- **What's New**: 이번 연구에서는 ASR(Automatic Speech Recognition)에서 낮은 리소스 언어인 힌디어의 제한된 라벨 데이터 문제에 대한 해결책으로 의사 라벨링(pseudo-labeling) 기법을 제안하고, 다양한 아이디어를 통합한 일반적인 프레임워크를 소개합니다.

- **Technical Details**: 제안된 프레임워크는 여러 기본 모델을 통합하여 오디오-전사 쌍을 평가하는 평가자를 포함합니다. 이 프레임워크는 힌디어 텍스트의 의사 라벨 데이터를 생성하기 위해 여러 ASR 모델의 일치도를 활용하고, 조건에 따라 전사 결과의 품질을 평가하는 메커니즘을 제공합니다.

- **Performance Highlights**: IndicYT라는 새 벤치마크에서 유튜브 오디오 파일을 활용하여 원본 훈련 데이터에 의사 라벨 데이터를 추가함으로써 성능 개선을 달성하였습니다. 이 프레임워크는 다양한 도메인에서 일관된 성능 향상을 보여주며, 다른 도메인 벤치마크에 대한 성능에 영향을 미치지 않았습니다.



### Focused Large Language Models are Stable Many-Shot Learners (https://arxiv.org/abs/2408.13987)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 FocusICL이라는 새로운 방법론을 제안하여 Large Language Models (LLMs)이 다수의 시연으로부터 주의가 분산되는 문제를 해결하고, 더 나은 성능을 이끌어냅니다.

- **Technical Details**: FocusICL은 triviality filtering을 수행하여 중요하지 않은 내용으로 인한 주의 분산을 방지하고, demonstration level에서 계층적 주의 메커니즘을 통해 현재 쿼리에 충분한 주의를 보장합니다. 또한, 모델 perplexity에 기반한 효율적인 하이퍼파라미터 검색 전략도 설계하였습니다.

- **Performance Highlights**: FocusICL은 ICL에 비해 평균 5.2% 향상된 성능을 달성하고, 많은 시연에서도 안정적으로 성능이 개선되는 특징을 보였습니다.



### TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models (https://arxiv.org/abs/2408.13985)
Comments:
          14 pages, 6 figures. arXiv admin note: text overlap with arXiv:2305.17440 by other authors

- **What's New**: 본 논문에서는 TF-Attack을 제안합니다. TF-Attack은 Transferable and Fast adversarial attacks 이라는 새로운 접근 방식을 통해 LLMs에 대한 적대적 공격의 효율성과 전이 가능성을 개선합니다.

- **Technical Details**: TF-Attack은 외부 LLM을 제3자 감독자로 사용하여 입력 문장의 중요한 단위를 식별하고 중요 수준(Importance Level) 개념을 도입하여 공격의 병렬 교체(parallel substitutions)를 가능하게 합니다. 이 접근 방식은 약 20배의 속도 향상을 가져옵니다.

- **Performance Highlights**: TF-Attack은 6개의 널리 사용되는 벤치마크에서 실험한 결과, 이전 공격 기법보다 전이 가능성(transferability)에서 일관되게 우수하고, 공격 속도는 평균 10배 이상 빨라지는 성과를 보입니다.



### Reducing the Cost: Cross-Prompt Pre-Finetuning for Short Answer Scoring (https://arxiv.org/abs/2408.13966)
Comments:
          This is the draft submitted to AIED 2023. For the latest version, please visit: this https URL

- **What's New**: 이번 연구는 Automated Short Answer Scoring (SAS) 기술의 훈련 데이터를 준비하는 비용을 줄이기 위해 두 단계 접근 방식을 제안합니다. SAS는 각 프롬프트에 대해 다른 루브릭과 참조 답변을 필요로 하여 매번 새로운 데이터 세트를 준비해야 하는 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 두 단계 접근 방식은 (i) 기존의 루브릭과 참조 답변을 사용하여 모델을 사전 훈련(pre-finetune)하고, (ii) 특정 새로운 프롬프트에 대해 모델을 미세 조정(finetune)하는 과정을 포함합니다. 이 과정에서는 키 구문(key phrases)을 활용하여 SAS 모델이 서로 다른 프롬프트 간의 관계를 학습하도록 합니다.

- **Performance Highlights**: 실험 결과, 키 구문을 활용한 기존의 크로스-프롬프트 데이터를 미세 조정함으로써, 훈련 데이터가 제한적일 때 더욱 향상된 채점 정확도를 보여주었습니다. 이 연구는 새로운 프롬프트에 대해 필요한 훈련 데이터를 줄이면서도 성능을 개선할 수 있음을 입증합니다.



### Bidirectional Awareness Induction in Autoregressive Seq2Seq Models (https://arxiv.org/abs/2408.13959)
- **What's New**: 이 논문에서는 Bidirectional Awareness Induction (BAI)이라는 새로운 훈련 방법을 소개합니다. 이 방법은 네트워크의 일부 요소인 Pivots를 활용하여 기존의 autoregressive 제약 조건을 유지하면서도 양방향 학습을 수행할 수 있도록 합니다.

- **Technical Details**: BAI는 Seq2Seq 문제에서 목표 출력을 재현하도록 Pivots를 훈련하는 방법입니다. Pivots는 네트워크 내에서 최종 목표 함수와 반드시 연관되지 않은 작업에 대해 훈련될 수 있는 요소로 정의됩니다.

- **Performance Highlights**: BAI는 실험에서 Image-Captioning에서 최대 2.4 CIDEr, Neural Machine Translation에서 4.96 BLEU, Text Summarization에서 1.16 ROUGE의 성과를 보여주었습니다. BAI는 처음부터 학습된 모델과 사전 훈련된 모델 모두에서 긍정적인 영향을 미쳤습니다.



### CoT Rerailer: Enhancing the Reliability of Large Language Models in Complex Reasoning Tasks through Error Detection and Correction (https://arxiv.org/abs/2408.13940)
- **What's New**: 이 연구에서는 CoT Rerailer라는 새로운 접근법을 제안하여 LLM의 추론 과정에서 발생하는 오류를 식별하고 수정함으로써 더 신뢰할 수 있는 AI 기반 의사결정을 돕습니다.

- **Technical Details**: CoT Rerailer는 오류 감지 및 수정을 위해 Self-Consistency (SC)와 Multi-Agent Debate (MAD) 시스템을 결합하여, 일관성이 높은 Reasoning Path (RP)를 선택하고 중간 단계의 오류를 수정합니다. 자동화된 에이전트의 평가를 통해 최적의 RP를 결정하고, 주어진 질문에 대한 논의 시스템을 통해 수정 제안을 검증하여 오류 없는 논리적 경로를 구성합니다.

- **Performance Highlights**: CoT Rerailer는 다양한 질문-답변 데이터셋에서 기존의 CoT, SC 및 MAD 방법들을 초월하여 hallucinations(환각증세)를 줄이고 정확도를 높이는 데 성공했습니다. 이로 인해 LLM이 생성하는 추론 결과의 신뢰성이 향상되었습니다.



### MobileQuant: Mobile-friendly Quantization for On-device Language Models (https://arxiv.org/abs/2408.13933)
Comments:
          Code and models available: this https URL

- **What's New**: 본 연구에서는 MobileQuant라는 새로운 포스트 트레이닝 양자화 방법을 소개하고 있습니다. 이는 기존의 양자화 방법의 한계를 극복하고 모바일 하드웨어에서의 LLM 배포를 용이하게 합니다.

- **Technical Details**: MobileQuant는 가중치 변환 및 활성화 범위 파라미터를 함께 최적화하여 가중치 및 활성화를 4비트 또는 8비트로 양자화합니다. 이 방법은 고정 소수점 정수 표현을 사용하여 모든 연산을 수행하게 됩니다.

- **Performance Highlights**: MobileQuant를 적용함으로써 추론 지연(latency)과 에너지 소비를 20%-50% 감소시킬 수 있으며, 16비트 활성화를 사용하는 모델과 비교하여 정확도를 거의 손실 없이 유지합니다.



### LLMs are Superior Feedback Providers: Bootstrapping Reasoning for Lie Detection with Self-Generated Feedback (https://arxiv.org/abs/2408.13915)
Comments:
          19 pages, 18 figures

- **What's New**: 이번 연구에서는 LLM(Large Language Models)의 기만 탐지 능력을 향상시키기 위한 부트스트랩(bootstrapping) 프레임워크를 제안합니다. 이 프레임워크는 제안, 피드백 수집, 수정의 세 가지 단계로 이루어져 있으며, LLM이 스스로 생성한 피드백을 활용하여 추론 능력을 개선합니다.

- **Technical Details**: 부트스트랩 프레임워크는 1) 제안 단계에서 비용 효율적인 LLM이 초기 예측을 생성하고, 2) 피드백 수집 단계에서 LLM이 이러한 예측에 대한 피드백을 제공하며, 3) 수정 단계에서 더 발전된 LLM이 자동 생성된 피드백을 사용하여 초기 예측을 다듬는 구조로 이루어져 있습니다. 이 방법을 Diplomacy 게임의 대화 데이터셋에 적용하였습니다.

- **Performance Highlights**: 제안된 방법은 기본 LLM에 비해 lying-F1 점수를 39% 향상시켰으며, 제로샷(Zero-shot) GPT-4 모델이 이전의 최첨단 감독 학습 모델과 경쟁할 수 있도록 했습니다. 특히, LLM이 생성한 피드백은 전문가의 피드백보다 29% 더 뛰어난 성능을 보였으며, 특히 인간이 불확실한 상황에서 유용함을 증명하였습니다.



### SpeechCaps: Advancing Instruction-Based Universal Speech Models with Multi-Talker Speaking Style Captioning (https://arxiv.org/abs/2408.13891)
Comments:
          SynData4GenAI 2024

- **What's New**: 이 논문은 다중 화자의 발화 스타일 캡셔닝(multi-talker speaking style captioning)이라는 새로운 작업을 제안하여 모델의 일반적인 음성 이해 능력을 향상시키는 것을 목표로 합니다. 특히, 다중 화자 환경에서의 화자와 운율(prosodic) 정보를 강화합니다.

- **Technical Details**: 제안된 방법은 SpeechCaps라는 다중 화자 발화 스타일 캡셔닝 데이터셋을 활용하며, 두 단계의 사전 훈련(pre-training)과 하나의 지침 조정(instruction tuning) 단계를 포함합니다. 첫 번째 단계에서는 단일 화자(single-talker) 발화 스타일 캡셔닝을 통해 기본적인 화자 정보를 학습하고, 두 번째 단계에서 다중 화자 캡셔닝으로 진행하여 복잡한 화자 인식 능력을 강화합니다.

- **Performance Highlights**: Dynamic-SUPERB에서의 평가 결과, 제안된 모델 DeSTA+는 단일 화자 작업에 대해서만 사전 훈련된 기존 모델에 비해 화자 인식 및 감정 인식에서 유의미한 성능 향상을 보였습니다. 특히, 다중 화자 QA 작업에서 성능이 두드러졌습니다.



### LLM with Relation Classifier for Document-Level Relation Extraction (https://arxiv.org/abs/2408.13889)
- **What's New**: 이번 논문에서는 기존 LLM(대형 언어 모델) 기반의 문서 수준 관계 추출(Document-level Relation Extraction, DocRE) 방법들의 성능 저하 원인을 조사하고, 새로운 LMRC(관계 후보 제안 및 분류) 방식을 제안합니다. 이 방법은 관계가 없는 엔티티 쌍으로 인한 주의력 분산 문제를 다루고, 관계가 있는 엔티티 쌍만을 선별하여 최종 관계 추출에 LLM을 활용하는 두 단계 워크플로우를 구성합니다.

- **Technical Details**: LMRC는 두 개의 주요 단계로 구성됩니다. 첫 번째 단계는 RCP(관계 후보 제안)로, 주의 메커니즘을 사용하여 관계가 없는 엔티티 쌍을 필터링하고 후보 공간을 좁힙니다. 두 번째 단계인 RC(관계 분류)에서는 LLM의 강력한 다중 분류 기능을 활용하여 세밀화된 후보 공간에서 관계 추출을 수행합니다.

- **Performance Highlights**: DocRED 및 Re-DocRED 벤치마크 실험에서 LMRC 방식이 기존 LLM 기반 DocRE 모델보다 현저하게 높은 성능을 보였으며, 여러 주요 전통 DocRE 모델들과 경쟁할 만한 성능을 달성했습니다.



### CodeGraph: Enhancing Graph Reasoning of LLMs with Cod (https://arxiv.org/abs/2408.13863)
Comments:
          In Progress

- **What's New**: 이번 논문에서는 CodeGraph라는 방법을 소개합니다. CodeGraph는 그래프 문제의 솔루션을 코드로 인코딩하여 새로운 문제를 학습하고, 이를 프로그램 인터프리터(Program Interpreter)를 통해 실행합니다. 이 방법은 LLM(대형 언어 모델)에서 그래프 추론(task) 성능을 1.3%에서 58.6%까지 향상시킬 수 있습니다. 특히, 기존 방식들과는 달리, 수학적 문제에서 높은 성능을 보여주며, 추론 과정을 더 잘 제어하고 해석할 수 있는 접근법을 제공합니다.

- **Technical Details**: CodeGraph는 LLM이 파이썬(Python) 프로그램으로 추론 과정을 설명하고, 프로그램 인터프리터를 사용하여 계산을 수행하는 방식입니다. 이 방법은 GraphQA 데이터셋을 활용하여 6개 그래프 인코딩 방법과 함께 6개의 작업(task)을 평가합니다. CodeGraph는 GPT-3.5 Turbo를 포함한 다양한 LLM에서 수행되며, 가장 작은 모델은 140억 개의 활성 매개변수를 사용합니다.

- **Performance Highlights**: CodeGraph의 실험 결과, 6개 기본 그래프 작업의 평균 정확도가 63.3%에서 96.1%로 상승했습니다. 또한, CodeGraph는 다양한 그래프 구조에서도 강건성을 유지하며, 추가적인 예제를 통해 더 작은 모델에서도 높은 성능을 달성할 수 있음을 보여주었습니다.



### Knowledge-Aware Reasoning over Multimodal Semi-structured Tables (https://arxiv.org/abs/2408.13860)
- **What's New**: 본 연구에서는 다중 모드(Dimodal) 데이터를 다루기 위한 새로운 데이터셋, MMTabQA를 소개합니다. 이 데이터셋은 텍스트와 이미지를 통합하여 지식 기반의 추론을 수행할 수 있는 AI 모델의 성능을 평가하기 위해 설계되었습니다.

- **Technical Details**: MMTabQA 데이터셋은 기존의 Wikipedia 데이터를 활용하여 이미지가 포함된 다중 모드 테이블 질문 응답 환경을 위해 생성되었습니다. 질문은 명시적, 답변 언급, 암시적 세 가지 유형으로 분류되며, 기존의 Wikipedia 질문 응답 데이터셋을 변환하여 구성되었습니다. 다양한 상태-오브-더-아트 Vision-Language 모델을 평가하여 복잡한 모드 테이블 처리의 어려움을 분석하였습니다.

- **Performance Highlights**: 현재 AI 모델들은 다중 모드 테이블에서의 지식 기반 추론 및 이미지, 텍스트 통합에서 상당한 도전에 직면하고 있습니다. 모델들은 엔티티 식별 오류, 시각적 이해의 어려움 및 테이블 구조 이해에 어려움을 겪고 있습니다. MMTabQA 데이터셋은 이러한 문제를 해결하기 위한 강력한 기준점을 제공합니다.



### Biomedical Large Languages Models Seem not to be Superior to Generalist Models on Unseen Medical Data (https://arxiv.org/abs/2408.13833)
Comments:
          10 pages, 3 tables, 1 figure

- **What's New**: 이번 연구는 생물 의학(Biomedical) 분야에 특화된 대규모 언어 모델(LLMs)의 성능을 일반 목적의 모델과 비교하여 평가하였습니다. 생물 의학 데이터로 세부 조정된 모델은 실제 임상 과제에서 예상보다 저조한 성능을 보였다는 점이 주목할 만합니다.

- **Technical Details**: 연구에서는 NEJM(New England Journal of Medicine)과 JAMA(Journal of the American Medical Association)에서 제공된 임상 사례 도전 과제를 분석하였으며, 정보 추출(information extraction), 문서 요약(document summarization), 임상 코드 할당(clinical coding) 등 다양한 임상 과제를 수행했습니다. 평가에 사용된 기준은 생물 의학 모델의 세부 조정 데이터셋에 포함되지 않을 것으로 예상되는 최신 벤치마크로 선택되었습니다.

- **Performance Highlights**: 대규모 모델은 케이스 과제에서 유사한 성능을 보였으나, 작은 생물 의학 모델은 일반 목적 모델에 비해 더 뚜렷한 성능 저하를 보였습니다. 예를 들어, OpenBioLLM-8B는 NEJM 사례에서 30%의 정확도를 보여주었고, Llama-3-8B-Instruct는 64.3%에 달했습니다. 결과적으로 생물 의학 모델의 세부 조정은 예상보다 낮은 성능을 초래했으며, 보다 엄격한 평가 프레임워크의 필요성을 강조하였습니다.



### Guardians of the Machine Translation Meta-Evaluation: Sentinel Metrics Fall In! (https://arxiv.org/abs/2408.13831)
Comments:
          Presented at ACL 2024 Main Conference. 29 pages

- **What's New**: 이번 연구에서는 기계 번역(Machine Translation, MT) 메트릭스의 메타 평가 프로세스에서 발생하는 두 가지 주요 문제를 조명하고, 이를 해결하기 위한 'sentinel metrics'라는 새로운 개념을 소개합니다. 이 메트릭스는 메타 평가의 정확성, 강건성(robustness), 공정성(fairness)을 검토하기 위해 디자인되었습니다.

- **Technical Details**: 현행 WMT의 메타 평가 프레임워크는 인간의 품질 평가를 모방하기 위해 특별히 훈련된 메트릭스와 연속적인 메트릭스에 유리하며, 이로 인해 발생하는 문제점을 지적하고, 메트릭스의 순위를 검증하기 위해 'sentinel metrics'를 사용합니다. 이 연구는 메트릭의 평가 과정에서 편향(bias)이나 불일치성을 모니터링하는 것을 목표로 합니다.

- **Performance Highlights**: 연구 결과, 현재의 메타 평가 프레임워크는 인간 평가와 잘 일치하지 않는 것으로 보이며, 인공지능(AI) 메트릭스들이 훈련 데이터에서 발생하는 허위 상관관계(spurious correlations)에 근거하여 평가를 수행할 수 있는 우려를 제기합니다.



### Revisiting the Exit from Nuclear Energy in Germany with NLP (https://arxiv.org/abs/2408.13810)
Comments:
          23 pages, 8 figures, Accepted for publication in Zeitschrift für Diskursforschung/Journal for Discourse Studies, ISSN: 2195-867X

- **What's New**: 최근 NLP(자연어 처리) 분야의 발전을 통해 정치 담론(annotation of political discourse)을 자동화할 수 있는 가능성이 열렸습니다. 이는 자원 집약적인 작업으로, 수작업으로 주석(annotation)이 달린 데이터셋이 필요합니다.

- **Technical Details**: 우리는 무감독 머신러닝(unsupervised machine learning) 및 제로샷( zero-shot)과 몇샷( few-shot) 학습 방법을 사용하여 수작업으로 주석이 달린 데이터셋이 얼마나 자동으로 복제될 수 있는지를 탐구합니다.

- **Performance Highlights**: Fine-tuned transformer(세밀 조정된 트랜스포머) 기반 모델들이 특정 주석 작업에서 인간 주석자보다 우수한 성능을 보이고 있습니다.



### Towards Reliable Medical Question Answering: Techniques and Challenges in Mitigating Hallucinations in Language Models (https://arxiv.org/abs/2408.13808)
Comments:
          9 pages

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 환각(hallucination) 문제를 완화하기 위한 다양한 기술을 기존 연구를 통해 검토합니다. 특히, 의료 분야의 지식 기반 작업에서의 효과성을 평가하며, Retrieval-Augmented Generation (RAG), 반복 피드백 루프(iterative feedback loops), 감독된 미세 조정(supervised fine-tuning), 프롬프트 엔지니어링(prompt engineering) 등에 중점을 둡니다. 이 연구는 LLM이 의료에 적용되는 현재의 한계를 이해하고 문제를 해결하기 위한 통찰력을 제공하려고 합니다.

- **Technical Details**: 환각은 주어진 입력 또는 맥락과 관련이 없거나 신뢰할 수 없는 콘텐츠를 생성하는 현상입니다. 본 논문에서는 환각의 두 가지 유형인 내재적(intrinsic) 환각과 외재적(extrinsic) 환각을 설명하고, 의료 QA 작업에서 발생할 수 있는 문제들을 세 가지 범주로 분류합니다: 사실 불일치(fact inconsistency), 질의 불일치(query inconsistency), 및 비본질적(tangentiality) 입니다. 주로 데이터 수집 결함, 반복 데이터, 태스크 불일치와 같은 여러 요인이 환각을 유발한다고 명시하고 있습니다.

- **Performance Highlights**: 연구 결과, 최신 모델인 ChatGPT와 GPT-4는 다양한 질문에 대해 각각 39.87% 및 16.6%의 정확도로 올바른 답변을 제공했으며, 불일치된 주장을 감지할 수 있는 능력은 각각 67.37%와 87.03%에 달했습니다. 그러나 후자의 경우에도 잘못된 답변을 생성하는 사례가 나타났으며, 제시된 문제에서 '단계별로 생각해 보자'라는 프롬프트를 포함했을 때 오류율이 현저히 감소했습니다.



### DOCE: Finding the Sweet Spot for Execution-Based Code Generation (https://arxiv.org/abs/2408.13745)
Comments:
          10 pages (32 including appendix), 5 figures, 25 tables. arXiv admin note: text overlap with arXiv:2304.05128 by other authors

- **What's New**: 최근 다양한 decoding 및 reranking 기법이 LLM(대형 언어 모델) 기반 코드 생성에 효과적임을 보여주었습니다. 그러나 이러한 방법들을 연결하고 실험적으로 비교하는 포괄적인 프레임워크는 부족했습니다. 본 연구에서는 코드 실행을 위한 Decoding Objectives(디코딩 목표)라는 포괄적인 프레임워크를 제안하여 후보 생성, n-best reranking, 최소 베이즈 위험(MBR) 디코딩, 자기 디버깅(self-debugging)을 핵심 구성 요소로 포함하고 있습니다.

- **Technical Details**: 프레임워크의 구성 요소에는 후보 생성(candidate generation), reranking, 자기 디버깅(Chen et al., 2024)이 포함되어 있으며, 이를 통해 오라클(oracle) 성능 및 reranking 성능 모두를 개선할 수 있습니다. 실행 기반 메트릭을 통해 평가를 수행하였으며, 여러 후보에 대한 자기 디버깅을 적용한 결과 최첨단 성능을 달성했습니다.

- **Performance Highlights**: 제안된 DOCE(Decoding Objectives for Code Execution) 프레임워크를 통해 생성된 후보들의 실행 기반 reranking이 오라클 성능에 근접한 결과를 보여주었으며, 필터링을 통한 trial unit test를 기반으로 한 간단하고 효과적인 전략의 중요성을 강조하였습니다. 이를 통해 향후 코드 생성 연구에 대한 확고한 지침을 제공할 수 있을 것으로 기대됩니다.



### Poor-Supervised Evaluation for SuperLLM via Mutual Consistency (https://arxiv.org/abs/2408.13738)
Comments:
          ACL findings

- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)의 평가에 대한 새로운 접근법인 PoEM(Poor-supervised Evaluation with Mutual Consistency) 프레임워크를 제안합니다. 이 프레임워크는 정확한 레이블이 부족한 상황에서도 LLM의 성능을 신뢰성 있게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: PoEM 프레임워크는 모델 성능을 평가하기 위해 서로 다른 모델 간의 예측 일관성을 이용합니다. 이때 무작위 모델을 참조 모델로 선택하는 것이 실제 조건에서의 일관성을 충족하지 못하는 문제를 해결하기 위해 모델과 인간을 참조 모델로 삼아 E-step과 M-step에서 교대로 모델의 가중치를 보정하고 필터링하는 알고리즘을 도입합니다. 이론적으로, 표본 수가 무한할 경우 참조 모델과의 일관성이 모델의 성능을 정확하게 반영할 수 있습니다.

- **Performance Highlights**: 실험 결과, PoEM은 16개의 주요 LLM에 대해 다양한 작업에서 평균 0.98의 Pearson 상관계수를 달성하였으며, 이는 기존의 감독 평가 결과와 잘 일치합니다. PoEM은 인간과 모델을 참조 모델로 동시에 활용하여 성능을 극대화하고, LLM 시대에 인간 평가의 한계를 완화하는데 기여합니다.



### Cross-Modal Denoising: A Novel Training Paradigm for Enhancing Speech-Image Retrieva (https://arxiv.org/abs/2408.13705)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2408.13119

- **What's New**: 본 논문은 음성과 이미지 간의 정밀한 조정을 달성하기 위한 새로운 학습 작업인 cross-modal denoising (CMD)을 소개합니다. CMD는 하나의 모달리티에서 노이즈가 포함된 특징들을, 다른 모달리티의 특징과 상호작용하여 재구성하는 작업입니다. 이를 통해 고차원에서의 상호작용을 개선하고 더 정밀한 음성-이미지 정렬을 달성합니다.

- **Technical Details**: CMD 작업은 훈련 중에만 작동하며 추론(인퍼런스) 단계에서는 제거될 수 있어 추가적인 추론 시간을 증가시키지 않습니다. 이 프레임워크는 HuBERT 및 CLIP과 같은 사전 훈련된 모델을 사용하여 음성 및 이미지 특징을 추출하고, 다중 작업 최적화로 음성-이미지 대비 학습과 CMD 작업을 포함합니다.

- **Performance Highlights**: 실험 결과, 본 프레임워크는 Flickr8k 데이터셋에서 평균 R@1에서 2.0% 향상된 성능을 보였으며, SpokenCOCO 데이터셋에서도 평균 R@1에서 1.7%의 성능 향상을 달성했습니다. 이는 최신 기법들과 비교했을 때 눈에 띄는 개선을 보여줍니다.



### DHP Benchmark: Are LLMs Good NLG Evaluators? (https://arxiv.org/abs/2408.13704)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 자연어 생성(NLG) 작업에서 평가자로 사용될 수 있다는 점을 강조하며, 기존 연구들이 LLM의 NLG 품질 평가 능력에 대한 충분한 탐색을 하지 못했다는 점을 지적합니다. 새로운 DHP(Discernment of Hierarchical Perturbation) 벤치마킹 프레임워크를 제안하여 LLM의 평가 능력을 정량적으로 측정하는 방법을 제시합니다.

- **Technical Details**: DHP 프레임워크는 계층적으로 변형된 텍스트 데이터를 사용하여 LLM의 평가 품질을 측정하는 시스템을 기반으로 하고 있습니다. 이를 통해 Wilcoxon Signed-Rank Test를 사용하여 평가 점수의 차이를 분석하고, 다양한 평가 메트릭을 결합하여 최종 Discernment Scores로 변환합니다. 이로 인해 LLM의 평가 능력을 보다 정밀하게 측정할 수 있습니다.

- **Performance Highlights**: DHP 벤치마크를 통해 다섯 가지 주요 LLM 시리즈에 대한 평가를 실시하였으며, 이를 통해 각 모델의 강점과 한계를 파악할 수 있는 중요한 통찰을 제공하였습니다. 또한, LLM들이 NLG 평가자로서의 역할을 수행하는 데 있어 나타나는 다양한 추세와 패턴을 밝혀내어, 향후 연구 및 개발에 중요한 기초 자료를 제공합니다.



### A layer-wise analysis of Mandarin and English suprasegmentals in SSL speech models (https://arxiv.org/abs/2408.13678)
Comments:
          4 pages, 3 figures, to be published in Interspeech 2024 proceedings

- **What's New**: 이 연구에서는 자가 지도 학습(self-supervised learning) 스피치 모델들이 만다린어의 음조와 영어의 강조 및 구분 악센트와 같은 초구획적(suprasegmental) 범주를 어떻게 표현하는지에 대한 새로운 통찰을 제공합니다. 특히 wav2vec 2.0, HuBERT, WavLM 모델을 비교 분석하여 이들의 표현 방식의 차이를 고찰했습니다.

- **Technical Details**: 연구에서는 12계층의 영어와 만다린어 단일 언어 모델을 사용하여 층별(layer-wise) 비교를 수행했습니다. 연구 결과, wav2vec 2.0 모델은 초구획적 범주를 잘 학습하며, 이는 네트워크의 중간 3분의 1에서 가장 강하게 나타났습니다. 또한, transformer 블록의 맥락(context)이 모델의 성능에 영향을 미친다는 것을 확인했습니다.

- **Performance Highlights**: 모델들은 음소의 정체성(phone identity) 및 의미적/문법적 특성의 표현에서 일관된 결과를 보여주었으며, 특히 fine-tuning을 통한 성능 향상이 레퍼런스 페어의 음조 및 강조 같은 렉시컬(lexically) 대립적 특징에서 두드러졌습니다. HuBERT와 WavLM은 wav2vec 2.0과 유사하지만, 후반 층에서의 성능 차이를 보였습니다.



### Symbolic Working Memory Enhances Language Models for Complex Rule Application (https://arxiv.org/abs/2408.13654)
- **What's New**: 대규모 언어 모델(LLMs)은 단일 단계의 규칙 적용에는 뛰어난 성능을 보이지만, 비순차적으로 제공되는 규칙들을 포함한 다단계 추론에서 성능이 저하되는 문제를 다루기 위해 외부 작업 메모리를 추가하는 신경상징적(neurosymbolic) 프레임워크를 제안한다.

- **Technical Details**: 제안된 프레임워크는 외부 작업 메모리를 통해 규칙과 사실을 자연어와 상징적 형태로 저장하여, 각각의 규칙 적용 단계에서의 규칙 기초(rule grounding)와 LLM 기반의 규칙 구현(rule implementation)을 지원한다. 작업 메모리는 사실 베이스와 규칙 베이스로 구성되어 있으며, 동적으로 메모리 스키마를 관리하여 중복을 방지한다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크가 CoT 기반 및 상징적 기반의 기준선(comparison)보다 우수한 성능을 나타내었으며, 다양한 규칙 적용 단계와 설정에서 강건성을 보였다.



### Narratives at Conflict: Computational Analysis of News Framing in Multilingual Disinformation Campaigns (https://arxiv.org/abs/2408.13651)
Comments:
          Published in ACL SRW 2024 Proceedings, see this https URL

- **What's New**: 이 논문은 멀티링구얼 환경에서의 프레이밍(framing) 차이를 규명하기 위해, 러시아 지원의 허위정보 캠페인에 대한 8년간의 데이터 분석을 다루고 있습니다. 특히 러시아어 기사가 특정 지역의 미디어 보도에서 선택된 프레임을 강조하는 경향을 보인다는 점에 주목합니다.

- **Technical Details**: 두 가지 모델(lexicon 기반 및 transformer 기반)로 프레임 식별을 비교하였고, 각각의 장점과 단점을 분석했습니다. 그리고 대규모 멀티링구얼 데이터셋을 활용해 러시아어, 프랑스어, 스페인어 및 이탈리아어의 허위정보 캠페인 관련 기사를 비교 분석했습니다.

- **Performance Highlights**: 이 연구는 기존 두 가지 프레임 분석 모델이 낮은 성능을 보이며, 특히 그 결과에 대한 높은 불일치를 드러낸다는 점을 발견했습니다. 이로 인해 향후 연구의 필요성이 강조되고 있습니다.



### No Dataset Needed for Downstream Knowledge Benchmarking: Response Dispersion Inversely Correlates with Accuracy on Domain-specific QA (https://arxiv.org/abs/2408.13624)
Comments:
          16 pages, 3 tables, 1 figure

- **What's New**: 이 연구는 LLM의 특정 주제 도메인에 대한 지식을 비교할 때 QA 데이터셋을 생성하고 (챗봇) LLM 응답을 평가할 필요성을 없애고자 합니다. 사용자 중심의 접근 방식을 사용하여 LLM의 내부 작업에 대한 접근 없이도 여러 번 동일한 질문을 던짐으로써 응답 분산(response dispersion)을 정의하고 평가할 수 있는 방법론을 제안합니다.

- **Technical Details**: 응답 분산은 LLM 응답의 임베딩 행렬에서 95%의 분산을 설명하는 데 필요한 단일 값의 수로 정의됩니다. 이 연구에서는 OpenAI API의 임베딩과 '참조 문장 유사성 임베딩'이라는 새로운 임베딩 방법을 사용하여 응답 임베딩을 생성하고 있습니다. 이 연구는 즉각적이고 저렴한 방법으로 최상의 LLM을 선택할 수 있도록 하는 절차를 제공합니다.

- **Performance Highlights**: 응답 분산을 비교하여 두 개의 다른 LLM을 같은 주제 도메인에서 비교할 때, QA 정확도 대신 응답 분산을 사용하는 것이 74%에서 89%의 시간에 적합한 대체 방법으로 나타났습니다. 이 연구는 5%와 10%의 허용 오차(tolerance level)에서 응답 분산을 통해 LLM 성능 평가의 효과성을 입증하였습니다.



### Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation (https://arxiv.org/abs/2408.13586)
- **What's New**: 본 논문에서는 샘플링 기반 디코딩 전략의 신뢰성을 향상시키기 위해, 다양성과 위험 간의 균형을 고려한 체계적인 방법을 제안하고 있습니다. 기존의 트렁케이션 샘플링 방법들과 그 추천 파라미터들에 대한 포괄적인 비교를 제공합니다.

- **Technical Details**: 연구팀은 Wikipedia 데이터를 단어 레벨의 접두사 트리 구조로 재구성하여, 'Context-Preserving Trie (CP-Trie)'를 만들었습니다. 이를 통해 각 샘플링 방법의 이론적 용량을 평가할 수 있으며, 다양한 트렁케이션 매개변수 값을 기준으로 토큰의 수를 측정합니다.

- **Performance Highlights**: 새롭게 제안된 평가 벤치마크를 통해, 다양한 샘플링 디코딩 방법의 이론적 용량을 추정할 수 있게 되며, 실제 응용에서의 트렁케이션 샘플링 방법 및 그에 맞는 파라미터 선택에 대한 가이드라인을 제공하여 샘플링 방법의 유용성을 높입니다.



### FLEURS-ASL: Including American Sign Language in Massively Multilingual Multitask Evaluation (https://arxiv.org/abs/2408.13585)
Comments:
          Access FLEURS-ASL at this https URL. arXiv admin note: text overlap with arXiv:2408.07065

- **What's New**: 본 논문에서는 기계 번역 연구 분야에 있어서 중요한 발전을 의미하는 FLEURS-ASL 벤치마크의 도입을 발표합니다. 이 벤치마크는 미국 수화(American Sign Language, ASL)와 200개 이상의 언어 간의 번역을 지원하며, 5명의 인증받은 청각 장애인 통역사가 번역을 제공합니다.

- **Technical Details**: FLEURS-ASL은 ASL과 텍스트 간의 문장 및 담화 수준의 번역을 평가할 수 있는 도구를 제공합니다. 34초의 문맥 윈도우(context window)에서 타임스탬프 토큰(timestamp tokens)과 이전 텍스트 토큰을 포함하는 통합 모델링 접근 방식을 바탕으로 훈련되었습니다. 이 모델은 문장 수준 평가의 성과를 초과하며, 새로운 과제를 지원하는 능력을 보여줍니다.

- **Performance Highlights**: FLEURS-ASL에서의 문장 수준 번역에 대한 인간 기초 성능은 13.0 BLEU, 64.6 BLEURT로 측정되었습니다. 제안된 모델은 문장 수준 기초 성능에서 3.7 BLEU를 달성하여 이전의 모델을 능가했습니다. 또한 최근 멀티모달 모델들이 ASL에 대한 이해도가 거의 없다는 것을 시사하며, 수화가 표준 평가 세트에 포함되어야 할 필요성을 강조합니다.



### IQA-EVAL: Automatic Evaluation of Human-Model Interactive Question Answering (https://arxiv.org/abs/2408.13545)
- **What's New**: 이 논문은 기존의 전통적인 언어 모델 평가 방법을 자동화하고, 상호작용 질문 응답 평가를 위한 새로운 프레임워크인 IQA-EVAL을 제안합니다. 특정 목표에 맞춰 설계된 이 프레임워크는 LLM(대형 언어 모델)을 기반으로 한 평가 에이전트(LEA)를 통해 human behaviors를 시뮬레이션하고 서로의 상호작용을 평가합니다.

- **Technical Details**: IQA-EVAL 프레임워크는 두 가지 주요 단계로 구성됩니다: (1) IQA 모델과의 상호작용 생성; (2) 생성된 상호작용 평가. LEA는 역할 설명, 작업 설명 및 논의 지침을 포함하는 구조화된 프롬프트를 활용하여 상호작용을 효과적으로 생성합니다. 개인의 특성을 반영한 다양한 페르소나를 LEA에 부여하여 보다 정교한 평가를 달성합니다.

- **Performance Highlights**: IQA-EVAL 프레임워크는 GPT-4 또는 Claude를 백본 모델로 사용할 때 인간 평가와 높은 상관관계를 보이며, 새로운 복잡한 질문 응답 작업에서 1000개 이상의 질문을 검토하여 다섯 개의 대표적인 LLM을 평가합니다. 이 자동화된 평가 방법은 인간 평가보다 약 $5k의 비용을 절감할 수 있습니다.



### Cultural Adaptation of Menus: A Fine-Grained Approach (https://arxiv.org/abs/2408.13534)
- **What's New**: 이 논문에서는 중국-영어 메뉴 번역에 대한 문화 특화 아이템(CSI)을 다루는 가장 큰 데이터셋인 ChineseMenuCSI를 소개합니다. 이 데이터셋은 각 항목에 CSI 및 Non-CSI 레이블이 부착되어 있으며, CSI의 다양한 측면을 분석할 수 있는 상세한 테스트 세트를 포함하고 있습니다.

- **Technical Details**: 논문에서는 자동 CSI 식별을 위한 독창적인 방법론을 개발하여, 기존의 GPT 기반 프롬프트보다 많은 카테고리에서 성능을 능가함을 보여주고 있습니다. 이 연구는 인간 번역 이론을 LLM(대형 언어 모델) 기반 번역 과정에 통합하여 번역 정확도를 크게 향상시킵니다. COMET 점수는 최대 7점까지 향상되었습니다.

- **Performance Highlights**: ChineseMenuCSI 데이터셋은 4,275개의 이중 언어 중국-영어 레스토랑 메뉴 항목으로 구성되어 있으며, 기존의 GPT 기반 프롬프트와 비교하여 CSI 식별 및 번역 성능에서 유의미한 개선을 나타냅니다. 외부 지식으로서 요리법을 활용함으로써, CSI 번역 성능을 추가적으로 향상시키고 COMET 점수는 각 CSI 카테고리에서 3점에서 7점까지 상승했습니다.



### Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models (https://arxiv.org/abs/2408.13533)
- **What's New**: 본 논문은 Retrieval-Augmented Generation (RAG) 모델에서의 노이즈 문제를 탐구하며, LLMs에서 유해한 노이즈와 유익한 노이즈를 재정의했습니다. 저자들은 7가지 노이즈 유형을 정의하고, Noise RAG Benchmark (NoiserBench)를 제안하여 다양한 데이터셋과 추론 과제를 포함하는 평가 프레임워크를 구축했습니다.

- **Technical Details**: 노이즈는 두 가지 주요 그룹으로 분류됩니다: 유익한 노이즈(semantics, datatype, illegal sentence)와 유해한 노이즈(counterfactual, supportive, orthographic, prior). 실험 결과, 유익한 노이즈가 모델 성능을 향상시킬 수 있는 반면, 유해한 노이즈는 일반적으로 성능을 저하시킵니다. 이 연구는 LLMs의 RAG 솔루션을 더 견고하고 적응력 있게 만드는 데 기여할 수 있는 통찰을 제공합니다.

- **Performance Highlights**: 실험에서 유익한 노이즈는 답변 형식 표준화, 명확한 추론 경로 및 신뢰도 증가를 통해 모델의 동작을 개선하는데 기여했습니다. 이 연구는 유해한 노이즈의 영향을 완화하고 유익한 노이즈의 긍정적인 효과를 활용하기 위한 토대를 조성합니다.



### HRGraph: Leveraging LLMs for HR Data Knowledge Graphs with Information Propagation-based Job Recommendation (https://arxiv.org/abs/2408.13521)
Comments:
          7 Pages, 4 Figures. View in ACL Anthology: this https URL

- **What's New**: 이 연구에서는 다양한 인사 관리(HR) 문서에서 HR 지식 그래프(Knowledge Graphs, KGs)를 효과적으로 개발하기 위한 프레임워크인 HRGraph를 소개합니다. 이 프레임워크는 대형 언어 모델(Large Language Models, LLMs)을 사용하여 HR 데이터를 처리하고 직업 연결 및 직원 추천을 가능하게 합니다.

- **Technical Details**: HRGraph 프레임워크는 HR 문서(예: 직무 설명서(Job Descriptions, JDs) 및 이력서(CVs))에서 다양한 엔터티를 식별 및 추출하고, 사전 훈련된 BERT 모델을 사용해 노드의 특징(long features)을 추출합니다. 이후 이 데이터를 기반으로 KGs를 구성하여 여러 하위 작업에 활용가능하게 합니다.

- **Performance Highlights**: HR KGs는 고용주와 직원 모두에게 유용한 정확한 직무 연결을 위한 사례를 보여줍니다. 실험 결과, 짧은 시간 내에 정보를 전파하는 능력을 지닌 KGs와 그래프 신경망(Graph Neural Nets, GNNs)을 사용하여 직원 추천, 직무 분류 등 다양한 작업에서 KGs의 효과성을 입증하였습니다.



### Selective Preference Optimization via Token-Level Reward Function Estimation (https://arxiv.org/abs/2408.13518)
Comments:
          Work in progress

- **What's New**: 최근 대규모 언어 모델의 정렬 방법론에서는 토큰 수준(supervision)을 활용하여 세부적인 선호 최적화를 수행하는 것이 주목받고 있습니다. 이와 관련하여 본 연구에서는 Selective Preference Optimization (SePO)이라는 새로운 선택적 정렬 전략을 제안합니다. SePO는 효율적인 키 토큰 선택을 중심으로 하며, Direct Preference Optimization (DPO)을 기반으로 최초의 토큰 선택 방법을 제시합니다.

- **Technical Details**: SePO는 원본 응답 수준의 선호 주석을 기반으로 하는 오라클 모델을 훈련하여 최적의 토큰 수준 보상 함수(reward function)를 파라미터화합니다. 주요 장점은 1) 추가적인 감독 신호 없이 기존 정렬 데이터셋에 직접 적용 가능하다는 점과 2) 오라클 모델의 크기 및 훈련 하위 집합의 크기를 조절함으로써 선택적 정렬의 비용을 감소시킬 수 있다는 점입니다. SePO에서는 선택된 키 토큰을 통해 목표 정책 모델을 최적화합니다.

- **Performance Highlights**: SePO는 세 개의 공공 평가 벤치마크에서 실험을 진행하였으며, 최적화된 30%의 키 토큰으로 경쟁 기초 방법론보다 현저히 우수한 성능을 나타냈습니다. 또한, 약한 오라클 모델이 파라미터 수가 16.8배 더 많은 강력한 정책 모델을 효과적으로 지도하여 성능 향상 및 분포 외(over-optimization) 문제를 해결했습니다.



### Utilizing Large Language Models for Named Entity Recognition in Traditional Chinese Medicine against COVID-19 Literature: Comparative Study (https://arxiv.org/abs/2408.13501)
Comments:
          22 pages with 2 figures

- **What's New**: 이번 연구는 COVID-19 문헌을 대상으로 한 전통 중의학(TCM) 관련 NER(정보 추출) 작업에서 ChatGPT와 최신 LLM(대규모 언어 모델)들의 성능을 비교하였습니다.

- **Technical Details**: 389개의 TCM 관련 COVID-19 논문으로 구성된 데이터셋을 구축하고, 48개 문서에 대해 3개의 도메인, 6개 엔티티 유형으로 수작업 주석을 달았습니다. ChatGPT(GPT-3.5 및 GPT-4)와 RoBERTa, MiniLM, PubMedBERT, SciBERT 등 4개의 최첨단 BERT 기반 QA 모델을 활용해 NER 작업을 수행하였으며, GSAP-NER 모델과 비교하였습니다.

- **Performance Highlights**: ChatGPT는 퍼지 매치(fuzzy match)에서 BERT 기반 QA 모델들보다 5개 작업 중 5개에서 높았으나, 정확한 매치(exact match)에서는 BERT 모델들이 5개 작업 중 5개에서 우세했습니다. GPT-4는 퍼지 매치에서 TCM 포뮬러 및 중국 특허약 관련 엔티티 유형에서 두드러진 장점을 보였으며, GSAP-NER는 RM에서 GPT-4를 다소 초과하는 F-1 점수를 기록했습니다. 전반적으로 LLM의 NER 성능은 엔티티 유형에 따라 크게 달라지며, ChatGPT는 높은 리콜(recall)이 필요한 상황에서 유용한 선택이 될 수 있습니다.



### Why Antiwork: A RoBERTa-Based System for Work-Related Stress Identification and Leading Factor Analysis (https://arxiv.org/abs/2408.13473)
Comments:
          13 pages, 8 figures

- **What's New**: 본 연구는 antiwork 서브레딧을 활용하여 직업 환경의 불만족을 탐구하고, 이를 통해 고용인의 심리적 문제를 조기에 감지하는 데 중점을 두고 있다. 기존의 연구들은 일반적인 정신 건강 분석에 집중했지만, 본 연구에서는 설명 가능한 솔루션을 제시하고 직장 특정 환경에 대한 연구를 수행하였다. 또한 RoBERTa 기반의 RNN 모델을 활용하여 antiwork 감정을 감지하는 새로운 데이터셋을 구축하고, 이를 통해 근무 환경에서의 불만족 원인을 분석하였다.

- **Technical Details**: 연구에서 사용된 기술은 RoBERTa(이전의 BERT 모델 개선) 기반의 반복 신경망(RNN)이다. 이 모델은 antiwork 감정이 포함된 포스트를 강조하며, LIWC(언어적 탐색 및 단어 수 세기) 및 주제 모델링 기법을 통해 antiwork 사용자들의 행동적 특성과 감정의 원인을 분석한다. 이를 통해 직장 내 고통을 최소화하는 방법에 대한 새로운 통찰을 제공한다.

- **Performance Highlights**: 모델의 정확도는 80%로, 고용 환경에서 안티워크 감정을 가진 사용자를 효과적으로 식별하고, 그들의 언어적 특성을 분석하여 직장 내 스트레스 요인을 조기에 발견할 수 있도록 돕는다. 이는 근로자의 권리 보호에 상당한 이점을 제공하고, 향후 다른 사회 문제 분석에도 활용될 수 있는 잠재력을 지닌다.



### Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning (https://arxiv.org/abs/2408.13457)
Comments:
          Preprint

- **What's New**: 이 논문에서는 Difficulty-Adaptive Self-Consistency (DSC)라는 새로운 방법을 제안하여, 기존 Self-consistency (SC) 기법의 한계를 극복하고 비용을 줄이는데 중점을 두었습니다. DSC는 문제 난이도 정보를 활용하여 적절히 추론 자원을 할당합니다.

- **Technical Details**: DSC는 다음의 세 단계로 구성됩니다: 난이도 순위 매기기 (Difficulty Ranking), 문제 파티션 (Problem Partition), 샘플 사이즈 사전 할당 (Sample Size Pre-Allocation). 난이도 순위 매기기 단계에서, LLM을 사용하여 문제 집합의 난이도를 평가하고, 이를 기반으로 문제를 쉬운 문제와 어려운 문제로 나눕니다. 그런 다음, 각 문제에 필요한 샘플 크기를 예측하여, 여러 번의 재샘플링을 줄이고 입력 비용을 절감합니다.

- **Performance Highlights**: DSC는 세 가지 주요 벤치마크에서 ASC 및 ESC보다 비용 효율성이 뛰어난 성능을 보였으며, 비슷한 성능을 유지했습니다. 실험 결과, DSC는 효율성과 성능 모두에서 강력한 기준선인 ASC 및 ESC를 일관되게 초과하는 결과를 나타냈습니다.



### Knowledge-Aware Conversation Derailment Forecasting Using Graph Convolutional Networks (https://arxiv.org/abs/2408.13440)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2306.12982; text overlap with arXiv:2106.01071 by other authors

- **What's New**: 이 연구는 대화 이탈 예측(conversation derailment forecasting) 문제를 다루며, 기존 그래프 신경망 모델의 한계를 극복하기 위해 상식 지식(common sense knowledge)을 활용한 새로운 지식 기반 그래프 컨볼루션 신경망(Knowledge Aware Forecasting Graph Convolutional Network, KA-FGCN) 모델을 제안합니다.

- **Technical Details**: KA-FGCN 모델은 다중 소스 정보(multi-source information)를 활용하여 발화(utterance) 정보를 캡슐화(capsule)하고, Transformer 기반의 예측기를 사용하여 대화 이탈을 예측합니다. 이 모델은 사용자 동적(dynamic)과 맥락 전파(context propagation) 및 감정 변화(emotional shifts)를 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: KA-FGCN은 CGA와 CMV 벤치마크 데이터 세트에서 기존 최첨단 모델을 능가하는 성능을 보였으며, 코드는 공개되어 재현 가능성을 제공합니다.



### Integrating Multi-Head Convolutional Encoders with Cross-Attention for Improved SPARQL Query Translation (https://arxiv.org/abs/2408.13432)
Comments:
          24 pages, 20 figures, using the engrXiv template; the full version has been submitted to ACM Transactions on Information Systems and is currently under review. (2024)

- **What's New**: 본 논문은 KGQA 시스템(지식 그래프 질문 답변)에서 사용자 질문을 SPARQL 쿼리 문법으로 변환하는 최신 NMT(Neural Machine Translation) 모델을 사용하여 질의 생성의 정확성을 향상시킵니다.

- **Technical Details**: 논문에서 제안하는 Multi-Head Conv encoder (MHC encoder)는 ConvS2S 인코더를 개선하여 Transformer의 multi-head attention을 추가한 것입니다. 이 구조는 다른 수용 기능을 가진 합성곱 층을 사용하여 입력 시퀀스의 지역적 숨겨진 특징을 캡처하고, 다양한 의존성을 계산합니다.

- **Performance Highlights**: Multi-Head Conv encoder 기반 번역 모델은 QALD-9 및 LC-QuAD-1.0 데이터셋에서 각각 76.52%와 83.37%의 BLEU-1 점수를 기록하여 다른 인코더보다 우수한 성능을 보여주었습니다. 또한, QALD-9와 LC-QuAD-1.0 데이터셋에 대한 end-to-end 시스템 실험 결과, Macro F1 점수가 각각 52%와 66%에 도달하여 최첨단 KGQA 시스템들 중에서 두각을 나타냈습니다.



### CodeRefine: A Pipeline for Enhancing LLM-Generated Code Implementations of Research Papers (https://arxiv.org/abs/2408.13366)
- **What's New**: 이 논문에서는 연구 방법론을 실행 가능한 코드로 자동 변환하는 CodeRefine라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM (Large Language Models)을 사용하여 코드를 생성하며, 여러 단계로 구성된 접근 방식을 통해 논문에서 주요 텍스트 조각을 추출하고 요약합니다.

- **Technical Details**: CodeRefine는 연구 논문에서 필요한 지식을 지식 그래프(knowledge graph) 형식으로 구조화한 후, 이를 바탕으로 코드를 생성합니다. 생성 과정에는 코드 관련성과 비관련성을 분석하고, ‘Retrospective Retrieval-Augmented Generation (RRAG)’ 접근 방식을 통해 생성된 코드의 정확성을 높입니다.

- **Performance Highlights**: CodeRefine는 다양한 과학 논문을 평가한 결과, 코드 생성의 정확성을 향상시킨 것으로 나타났으며, LLM zero-shot prompting보다 더 신뢰할 수 있는 대안을 제공합니다. 이로 인해 최첨단 알고리즘이 실제 응용 프로그램에 신속하게 채택될 가능성이 높아집니다.



### Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler (https://arxiv.org/abs/2408.13359)
- **What's New**: 본 논문은 새로운 학습률 스케줄러인 PowerLR를 제안하여, 다양한 배치 크기와 훈련 토큰 수에 관계없이 최적의 학습률을 효과적으로 조정할 수 있음을 보여줍니다. 이를 통해 기존의 Cosine 및 WSD 스케줄러가 갖는 단점을 극복하고, 매우 큰 언어 모델에서도 효율적인 훈련을 가능하게 합니다.

- **Technical Details**: PowerLR 스케줄러는 배치 크기(배치 사이즈)와 훈련 토큰 수(트레이닝 토큰)의 영향을 받지 않아, 다양한 모델 크기와 훈련 설정에서 최적의 학습률을 직접 전이할 수 있습니다. 실험을 통해, 최적 학습률은 배치 크기(β) 및 훈련 토큰 수(T)에 대해 파워-로우(power-law) 관계를 따른다는 사실이 밝혀졌습니다.

- **Performance Highlights**: PowerLR와 Maximum Update Parameterization (muP)을 결합한 결과, 훈련 토큰 수, 배치 크기, 모델 크기 및 모델 아키텍처에 관계없이 하나의 하이퍼파라미터 집합으로 지속적으로 인상적인 성능을 달성할 수 있음을 입증했습니다. 특히 3B 밀집 모델(dense model)과 Mixture of Experts(MoE) 모델이 최첨단 소형 언어 모델들과 비교해 상응하는 성능을 보여주었습니다.



### Question answering system of bridge design specification based on large language mod (https://arxiv.org/abs/2408.13282)
Comments:
          10 pages, 7 figures

- **What's New**: 이번 논문에서는 대규모 언어 모델을 기반으로 한 교량 설계 사양에 대한 질문 응답 시스템을 구축하였습니다. 세 가지 구현 방안으로 Bert pretrained model의 완전 미세 조정(full fine-tuning), 파라미터 효율적 미세 조정(parameter-efficient fine-tuning), 그리고 자체 구축 언어 모델을 이용한 방법이 시도되었습니다.

- **Technical Details**: 모델은 TensorFlow와 Keras 딥러닝 플랫폼 프레임워크를 기반으로 설계되었으며, 사용자가 제공하는 교량 설계 사양에서 답변의 시작 위치와 끝 위치를 예측하는 훈련이 이루어졌습니다.

- **Performance Highlights**: 전체 미세 조정한 Bert pretrained model은 훈련 데이터셋(training dataset), 검증 데이터셋(validation dataset), 테스트 데이터셋(test dataset)에서 100% 정확도를 기록하였습니다. 반면, 파라미터 효율적 미세 조정 방법과 자체 구축 언어 모델은 훈련 데이터셋에서 좋은 성능을 보였지만, 테스트 데이터셋에서 일반화 능력(generalization ability)을 향상시킬 필요가 있음을 시사합니다.



### Retrieval-Augmented Generation Meets Data-Driven Tabula Rasa Approach for Temporal Knowledge Graph Forecasting (https://arxiv.org/abs/2408.13273)
Comments:
          Paper was accepted at ACM KDD -2024 -- Undergraduate Consortium. Please find the link: this https URL

- **What's New**: 본 논문에서는 sLA-tKGF(Temporal Knowledge Graph Forecasting를 위한 소규모 언어 모델)를 소개하여 기존의 문제점인 부정확한 정보 회상, 환각, 편향 및 데이터 유출 문제를 해결하고 있습니다.

- **Technical Details**: sLA-tKGF는 Retrieval-Augmented Generation (RAG) 방식을 통해, 역사적 데이터와 웹 검색 결과를 활용하여 지식이 포함된 프롬프트를 구성하고, 예측 정확도를 높이기 위해 다층 스택드 바닐라 트랜스포머 아키텍처를 기반으로 하여 처음부터 맞춤형으로 훈련됩니다.

- **Performance Highlights**: 엄격한 실험 결과, 이 프레임워크는 공개된 데이터셋에서 SOTA 성능을 보여주며, 예측의 해석 가능성과 신뢰성을 보장합니다.



### A Practitioner's Guide to Continual Multimodal Pretraining (https://arxiv.org/abs/2408.14471)
Comments:
          Technical Report. 52 pages

- **What's New**: 이 논문에서는 여러 분야에 걸친 시각과 언어의 교차점에서 작동하는 멀티모달(Multimodal) 기초 모델의 지속적(pretraining) 업데이트를 위한 새로운 접근 방식을 제안합니다. 특히 실제 배포 요구 사항을 고려한 지속적인 멀티모달 프리트레이닝 벤치마크인 FoMo-in-Flux를 소개합니다.

- **Technical Details**: FoMo-in-Flux는 63개의 다양한 데이터 세트를 기반으로 구성되어 있으며, 실제적인 컴퓨팅 제약(compute constraints)과 배포 요구사항(practical deployment requirements)을 반영합니다. 본 연구에서는 데이터 중심(data-centric) 조사, 방법 중심(method-centric) 전략, 메타 학습 속도 조절(meta learning rate schedules), 모델 및 컴퓨트 스케일링의 영향을 포함한 다양한 관점에서 지속적 프리트레이닝을 탐구합니다.

- **Performance Highlights**: 이 연구는 지속적 멀티모달 프리트레이닝을 위한 실무자 가이드를 제공하며, 실제 환경에서의 배포를 위한 여러 접근 방식을 고찰합니다. 논문과 함께 제공된 벤치마크와 코드는 연구자 및 실무자들이 유용하게 활용할 수 있습니다.



### CHARTOM: A Visual Theory-of-Mind Benchmark for Multimodal Large Language Models (https://arxiv.org/abs/2408.14419)
- **What's New**: CHARTOM은 다중 모달 대형 언어 모델을 위한 시각적 마음 이론 벤치마크로, 차트를 기반으로 FACT 질문과 MIND 질문을 통해 AI의 이해도를 평가합니다.

- **Technical Details**: CHARTOM 벤치마크는 112개의 차트로 구성되며, 기본적인 자료를 시각적으로 조작하여 실제 FACT 질문과 MIND 질문에 답변할 수 있도록 설계되었습니다. 각 차트는 바, 선, 파이, 산점도, 지도 차트 등 5가지 일반적인 유형으로 나뉘며, 조작된 버전은 심리학 문헌에 따른 시각적 오류를 포함합니다.

- **Performance Highlights**: AI는 FACT 질문에 정확히 답변할 수 있지만, MIND 질문의 경우 일반 독자가 차트에 어떻게 반응할지를 예측하는 것이 더 도전적임을 보여줍니다. 이는 AI가 시각적 자료의 진위와 인간의 인지를 고려하는 새로운 방법을 제시합니다.



### Uncovering Knowledge Gaps in Radiology Report Generation Models through Knowledge Graphs (https://arxiv.org/abs/2408.14397)
Comments:
          Code is available at: this https URL

- **What's New**: 최근 인공지능(AI) 기술의 발전으로 방사선 보고서의 자동 생성이 크게 개선되었습니다. 그러나 기존 평가 방법은 모델이 방사선 이미지를 이해하는 수준과 인간 수준의 세밀함(Granularity)에 도달하는 능력을 드러내지 못하고 있습니다. 이를 해결하기 위해, 우리는 ReXKG라는 시스템을 소개하며, 이 시스템은 처리된 보고서에서 구조화된 정보를 추출하여 종합적인 방사선 지식 그래프를 구축합니다.

- **Technical Details**: ReXKG는 처리된 보고서에서 정보를 추출하여 방사선 지식 그래프를 생성하는 시스템입니다. 우리는 다음과 같은 세 가지 평가 지표를 제안합니다: ReXKG-NSC(노드 유사성 계수), ReXKG-AMS(인접 행렬 유사성), ReXKG-SCS(서브그래프 커버리지 점수). 이 지표들은 다양한 지식 그래프 간의 노드 유사성, 엣지 분포, 서브그래프의 커버리지를 평가하는 데 사용됩니다.

- **Performance Highlights**: AI 모델들의 방사선 보고서 생성 성능에 대한 포괄적인 분석을 통해, 일반 모델(generalist)의 경우 80% 가까운 필수 엔티티 커버리지를 보여주지만, 의료 기기 세부사항에서는 방사선 전문의가 작성한 보고서에 미치지 못했습니다. AI 모델들은 개념을 과적합(overfit)하여 훈련 데이터에서 자주 등장하는 특정 개념에 집중하는 경향이 있으며, 결과적으로 세부사항이 부족하거나 비현실적(hallucinated)인 설명을 생성하였습니다. 일반 모델은 다양한 데이터에 노출되어 방사선 지식이 크게 향상되었습니다.



### SWE-bench-java: A GitHub Issue Resolving Benchmark for Java (https://arxiv.org/abs/2408.14354)
Comments:
          This work is in progress

- **What's New**: 본 논문에서는 다국어 지원을 향한 첫걸음으로 Java 버전인 SWE-bench-java를 개발하였으며, 이는 기존의 Python 중심 SWE-bench의 한계를 보완하기 위한 것입니다. 이는 산업에서 필요한 다양한 프로그래밍 언어 지원을 목표로 합니다.

- **Technical Details**: SWE-bench-java는 53개의 GitHub Java 리포지토리와 Defects4j 데이터베이스에서 수집된 17개의 리포지토리로 구성된 70개의 후보 리포지토리로부터 구축되었습니다. 최종적으로 19개의 오픈소스 Java 리포지토리에서 1,979개의 이슈 인스턴스를 수집하였으며, 이 중 137개의 고급 테스트가 포함된 이슈 인스턴스가 검증되었습니다.

- **Performance Highlights**: SWE-bench-java를 통해 여러 강력한 LLM 모델(GPT-4o, DeepSeek-V2 등)과의 성능 평가가 이루어졌습니다. 이는 다국어 GitHub 이슈 해결 벤치마크를 구축하기 위한 첫 단계를 마련하고, Java 프로젝트에서의 이슈 해결을 보다 잘 이해할 수 있는 기회를 제공합니다.



### Foundation Models for Music: A Survey (https://arxiv.org/abs/2408.14340)
- **What's New**: 최근 몇 년 동안, 대형 언어 모델(LLMs)과 잠재 확산 모델(LDMs)과 같은 foundation models(FMs)이 음악을 포함한 다양한 분야에 심각한 영향을 미쳤습니다. 이 리뷰는 음악의 사전 훈련 모델과 최신 기술을 조사하며, 음악 이해 및 생성 측면에서의 FM의 잠재력을 강조합니다.

- **Technical Details**: 연구에서는 예를 들어, 단일 모달리티 모델과 다중 모달리티 모델을 구분하고, 자가 감독 학습(self-supervised learning, SSL)이 음악 데이터에 어떻게 적용되는지 설명합니다. 또한, 구조적 선택, 토크나이징(tokenisation), 파인튜닝(finetuning) 방법론 및 제어 가능성(controllability)과 같은 모델 사전 훈련 패러다임의 세부 사항을 논의합니다.

- **Performance Highlights**: FMs는 데이터 부족 문제를 해결하고 주석 비용을 줄이는 동시에 음악 정보 검색(music information retrieval) 및 창작에서 일반화(generalization)를 향상시킵니다. 음악 장르, 구조 또는 악기를 이해하는 데 더 뛰어난 성능을 보이며, 문화유산 보호 및 예술 표현의 새로운 형태에 기여할 잠재력을 내포하고 있습니다.



### Epidemic Information Extraction for Event-Based Surveillance using Large Language Models (https://arxiv.org/abs/2408.14277)
Comments:
          11 pages, 4 figures, Ninth International Congress on Information and Communication Technology (ICICT 2024)

- **What's New**: 이 논문은 인공지능(AI)과 대규모 언어 모델(LLMs)을 활용하여 비정형 대량 데이터 소스(예: ProMED 및 WHO 질병 발생 뉴스)를 효과적으로 해석하기 위한 새로운 접근 방식을 제시합니다. LLMs의 능력을 평가하고, 인-컨텍스트 학습(in-context learning)을 통해 LLMs를 향상시키며, 여러 개의 오픈 소스 LLM을 통합한 앙상블 모델의 성능을 테스트합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)은 딥러닝(DL) 알고리즘을 이용하여 단어 순서의 가능성을 계산합니다. LLM은 Transformer 아키텍처를 사용하여 입력 데이터의 다양한 부분을 가중치 부여하고 우선순위를 정할 수 있습니다. EpiTator는 전염병 정보 추출을 위해 설계된 오픈소스 에피데미올로지 주석 도구로, 주요 엔티티(예: 질병, 장소, 날짜)를 추출합니다.

- **Performance Highlights**: LLMs는 전염병 모델링 및 예측의 정확성과 적시성을 크게 향상시킬 수 있는 가능성을 보여주며, 향후 팬데믹 이벤트 관리에 유망한 도구를 제공합니다.



### Explaining Vision-Language Similarities in Dual Encoders with Feature-Pair Attributions (https://arxiv.org/abs/2408.14153)
- **What's New**: 이번 연구에서는 CLIP 모델과 같은 Dual encoder 아키텍처의 예측을 입력 간의 feature-pair 상호작용으로 귀속할 수 있는 방법을 제안하였습니다. 이를 통해 모델이 입력된 두 데이터를 비교하는 방식을 보다 깊이 이해할 수 있습니다.

- **Technical Details**: 제안된 방법은 어떠한 미분 가능 Dual encoder 모델에 대해서도 입력 간의 상호작용을 설명할 수 있는 일반적인 feature-pair 귀속값을 계산하게 합니다. 특히, 이 방법은 훈련된 모델의 수정 없이 적용 가능하며, 시각-언어 모델에 적용되는 과정에서 세밀한 상호작용을 포착할 수 있습니다.

- **Performance Highlights**: 모델은 주어진 데이터 배치 내에서 세부적인 객체 클래스 간의 지식 격차를 식별하고, in-domain 훈련 후 성능 향상을 모니터링할 수 있는 능력을 가지고 있어, CLIP 모델의 시각-언어 기초 능력이 객체 클래스에 따라 다양하게 나타남을 보여줍니다.



### SurGen: Text-Guided Diffusion Model for Surgical Video Generation (https://arxiv.org/abs/2408.14028)
- **What's New**: 본 연구에서는 SurGen이라는 텍스트 기반 확산 모델을 소개합니다. 이는 기존의 외과 비디오 생성 모델들 중 가장 높은 해상도(720 x 480 pixels)와 긴 지속시간(49 frames)을 자랑합니다. 이 모델은 외과 절차의 다양한 단계를 지정하는 텍스트 프롬프트에 의해 조건화되어 비디오를 생성합니다.

- **Technical Details**: SurGen은 CogVideoX를 기반으로 하여 텍스트 프롬프트를 사용하여 외과 영상을 생성합니다. 모델 학습에는 200,000개의 독특한 비디오-텍스트 쌍이 사용되며, 각 비디오는 특정 외과 단계에 맞는 프롬프트와 쌍을 이룹니다. 각 비디오는 3D Variational Autoencoder(3D VAE)와 Denoising Video Transformer를 통해 생성됩니다. 또한 2억 개의 파라미터를 가진 텍스트 조건형 비디오 변환기를 사용하여 비디오의 공간 및 시간 정보를 처리합니다.

- **Performance Highlights**: SurGen 모델은 Fréchet Inception Distance(FID) 지표에서 79.9163을 기록하여, 사전 훈련된 모델의 260.0301보다 현저하게 개선된 시각적 품질과 다양성을 나타냅니다. Fréchet Video Distance(FVD)에서도 752의 개선된 성능을 보였습니다.



### AgentMove: Predicting Human Mobility Anywhere Using Large Language Model based Agentic Framework (https://arxiv.org/abs/2408.13986)
Comments:
          13 pages

- **What's New**: 본 논문에서는 AgentMove라는 새로운 예측 프레임워크를 소개합니다. 이는 인류의 이동 패턴을 보다 효과적으로 예측할 수 있도록 설계되었습니다. 전통적인 딥러닝 모델의 한계를 극복하고, 제로샷(Zero-shot) 예측을 가능케 합니다.

- **Technical Details**: AgentMove는 이동 예측(task)을 세 가지 서브태스크로 분해합니다: (1) 개별 이동 패턴 탐색(spatial-temporal memory module), (2) 도시 구조의 영향 모델링(world knowledge generator), (3) 인구 간의 공유 패턴 포착(collective knowledge extractor). 이 프레임워크는 메모리와 지식 생성을 결합하여 복잡한 이동 패턴을 더 잘 포착합니다.

- **Performance Highlights**: AgentMove는 12개 도시의 이동 데이터를 기반으로 한 실험에서 기존 최적 모델보다 8% 이상 향상된 성능을 기록했습니다. 또한 다양한 LLMs에 대한 높은 적응성과 안정성을 보여주며, 도시 간 예측의 지리적 편향을 줄였습니다.



### Prediction of COPD Using Machine Learning, Clinical Summary Notes, and Vital Signs (https://arxiv.org/abs/2408.13958)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 만성 폐쇄성 폐질환(COPD)의 악화를 예측하기 위해 AI(인공지능)와 NLP(자연어 처리) 기법을 이용한 두 가지 예측 모델을 제시합니다.

- **Technical Details**: 제안된 모델은 호흡 요약 노트, 증상, 그리고 생체 신호(vital signs)를 이용하여 COPD 악화를 예측합니다. 훈련 및 테스트에는 ICU(중환자실) 환자들로부터 수집된 생리적 신호와 생체 신호의 시계열 데이터 기록이 사용되었습니다.

- **Performance Highlights**: 이 모델은 COPD 악화의 탐지 및 예측에서 0.82의 ROC 곡선 아래 영역(Area Under the Curve, AUC)을 달성했습니다.



### LowCLIP: Adapting the CLIP Model Architecture for Low-Resource Languages in Multimodal Image Retrieval Task (https://arxiv.org/abs/2408.13909)
- **What's New**: 본 연구는 자원을 적게 사용하는 언어인 아제르바이잔어를 위한 이미지 검색을 위하여 다중 모달 비전-언어 모델 개발을 탐구합니다.

- **Technical Details**: CLIP 모델 아키텍처를 통합하고 기계 번역을 통한 합성 데이터 생성, 이미지 증강, 그리고 도메인 특화 데이터로 변형하기 위한 추가적인 훈련을 활용하여 모델을 개발하였습니다. 여러 이미지 인코더(ResNet50, EfficientNet0, Vision Transformer)와 다국어 BERT를 텍스트 인코더로 통합했습니다.

- **Performance Highlights**: EfficientNet0 모델은 Flickr30k에서 MAP 점수를 0.84에서 0.87로, ResNet50은 MSCOCO에서 0.70에서 0.80으로 향상시켜 시각-언어 검색 분야의 최신 성과를 달성했습니다.



### Literary and Colloquial Tamil Dialect Identification (https://arxiv.org/abs/2408.13739)
Comments:
          18 pages, 6 figures, submitted to "Circuits, Systems, and Signal Processing"

- **What's New**: 이번 연구는 문어(Tamil Literary, LT)와 구어(Tamil Colloquial, CT) 간의 방언 식별(Dialect Identification, DID) 문제를 다루고 있습니다. LT는 공식적인 글쓰기에서 사용되는 반면, CT는 일상 대화에 주로 사용되며, 두 방언 간의 변환이 필요합니다. 이는 컴퓨터 지원 언어 학습 어플리케이션에서 LT와 CT를 함께 사용하는 필요성을 강조합니다.

- **Technical Details**: 연구팀은 방언 식별을 위해 5가지 방법을 탐색하였으며, 그 중 두 가지는 암묵적 방법(Implicit Methods)인 Gaussian Mixture Model (GMM)과 Convolutional Neural Network (CNN)이고, 두 가지는 명시적 방법(Explicit Methods)인 Parallel Phone Recognition (PPR)과 Parallel Large Vocabulary Continuous Speech Recognition (P-LVCSR)입니다. 또한, 두 개의 제안된 명시적 방법인 Unified Phone Recognition (UPR-1 및 UPR-2)도 포함됩니다.

- **Performance Highlights**: 시험 발화의 평균 지속 시간은 LT의 경우 4.9초, CT의 경우 2.5초로 짧았으나, 각 시스템의 식별 정확도는 다음과 같습니다: GMM 87.72%, CNN 93.97%, PPR 89.24%, P-LVCSR 94.21%, UPR-1 88.57%, UPR-1 (P-LVCSR 포함) 93.53%, UPR-2 94.55%, UPR-2 (P-LVCSR 포함) 95.61%.



### GPT-4 Emulates Average-Human Emotional Cognition from a Third-Person Perspectiv (https://arxiv.org/abs/2408.13718)
Comments:
          submitted to 12th International Conference on Affective Computing & Intelligent Interaction, Glasgow, UK, September 15-18, 2024

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 감정 추론 능력에 대한 최근 연구를 확장한 내용으로, LLM이 타인의 감정 인식과 자기 감정 인식 간의 차이를 평가한다.

- **Technical Details**: 저자들은 인위적으로 설계된 감정 유도 자극(emotion-evoking stimuli)을 사용하여, GPT-4가 그러한 자극에 대한 추론에서 특히 정확성이 높음을 보여준다. 감정 이론(appraisal theory)과 같은 이론적 틀을 통해 LLM의 감정 처리를 탐구하며, LLM이 타인의 감정을 추론하는 방식과 자아 감정을 인식하는 방식의 차이를 분석한다.

- **Performance Highlights**: GPT-4는 감정 유도 자극에 대한 추론에서 인간의 판단과 잘 일치하는 경향이 있으며, 특히 타인의 감정에 대한 해석이 개인의 자기 평가보다 더 신뢰할 수 있는 것으로 나타났다. 기존의 감정 모델은 주로 자기 보고(self-reported) 기반의 진실을 표준으로 삼지만, 이 연구는 관찰자의 관점을 채택하는 것이 더 많은 응용 프로그램에서 유용할 수 있음을 시사한다.



### Ancient but Digitized: Developing Handwritten Optical Character Recognition for East Syriac Script Through Creating KHAMIS Datas (https://arxiv.org/abs/2408.13631)
Comments:
          15 pages, 12 figures, 5 tables

- **What's New**: 이 논문은 고대 언어인 시리아어(Syriac)의 손으로 쓴 텍스트를 인식하는 광학 문자 인식(OCR) 모델 개발에 관한 연구 결과를 보고합니다. 연구진은 KHAMIS라는 데이터셋을 구축하여 손글씨 시리아어 인식을 위한 기준을 마련했습니다.

- **Technical Details**: KHAMIS 데이터셋은 31명의 대학생과 1명의 교수로부터 수집된 총 624개의 손으로 쓴 시리아어 문장으로 구성되어 있으며, Tesseract-OCR 엔진의 사전 훈련된 시리아어 모델을 손으로 쓴 데이터에 대해 미세 조정했습니다. 이 논문의 방법론은 데이터 수집, 전처리, 제안된 방법 설명, 모델 훈련, 평가 방법으로 나뉩니다.

- **Performance Highlights**: 새로 개발한 손글씨 OCR 모델은 훈련 세트에서 1.097-1.610%의 문자 오류율과 8.963-10.490%의 평가 세트 성능을 기록했으며, 테스트 세트에서는 18.89-19.71%의 문자 오류율과 62.83-65.42%의 단어 오류율을 달성했습니다. 이는 Tesseract의 기본 시리아어 모델에 비해 두 배 이상의 성능 향상을 보여줍니다.



### A Law of Next-Token Prediction in Large Language Models (https://arxiv.org/abs/2408.13442)
- **What's New**: 본 논문은 사전 훈련된 대형 언어 모델(LLM)에서 중간 층을 통한 컨텍스트화된 토큰 임베딩(embedding)의 학습을 설명하는 정밀하고 정량적인 법칙을 소개합니다. 이 법칙은 입력 데이터 처리 및 예측의 정확도를 향상시키기 위해 각 층이 동등한 기여를 한다는 점에서의 새로운 통찰을 제공합니다.

- **Technical Details**: LLM은 비선형 모델로, 각 층에서 주의(attention) 메커니즘과 다양한 연산을 통해 토큰 임베딩의 시퀀스를 새 시퀀스로 반복적으로 매핑합니다. 각 층의 예측 능력을 평가하기 위해 예측 잔차(prediction residual, PR)를 사용하여 LLM의 다음 토큰 예측 능력을 정량화하였습니다. 이 연구는 PR 값이 층을 지날 때마다 거의 일정하게 감소하는 '균일학습(equi-learning)' 법칙을 밝혀냈습니다.

- **Performance Highlights**: 각 층의 PR 값은 −0.983에서 −0.997 사이의 Pearson 상관 계수로 나타났으며, 모델 아키텍처, 데이터 세트, 모델 깊이 및 훈련 시간에 따라 법칙의 특성이 달라질 수 있음을 보여줍니다. 이 법칙은 LLM의 설계, 훈련 및 해석을 위한 보다 세분화된 접근 방식으로 활용될 수 있습니다.



### DrugAgent: Explainable Drug Repurposing Agent with Large Language Model-based Reasoning (https://arxiv.org/abs/2408.13378)
Comments:
          18 pages, 1 figure

- **What's New**: 이번 논문은 기존 약물의 새로운 치료 가능성을 찾기 위해 다중 에이전트 시스템(multi-agent system) 프레임워크를 제안하고 있습니다. 최신 기계 학습(machine learning) 기법과 지식 통합(knowledge integration)을 사용하여 약물 재목적화(drug repurposing) 과정을 향상시키는 데 초점을 둡니다.

- **Technical Details**: 프레임워크는 AI Agent, Knowledge Graph Agent, Search Agent의 세 가지 전문 에이전트로 구성됩니다. AI Agent는 강력한 약물-타겟 상호작용(drug-target interaction, DTI) 모델을 훈련하고, Knowledge Graph Agent는 다양한 데이터베이스를 활용해 DTI를 체계적으로 추출하며, Search Agent는 생물 의학 문헌과 상호작용하여 계산된 예측을 주석 달고 검증합니다. 이 시스템은 외부 데이터베이스에서 얻은 다양한 데이터 소스를 효과적으로 활용합니다.

- **Performance Highlights**: 예비 결과에 따르면, 이 접근법은 약물-질병 상호작용을 예측하는 데 있어 기존 방법들보다 뛰어난 성능을 보이며, 전통적인 약물 발견 과정에 비해 시간과 비용을 줄일 수 있는 가능성을 보여줍니다. 또한, 다중 에이전트 시스템의 확장성을 강조하며, 약물 재목적화 분야에서의 혁신을 촉진하는 역할을 합니다.



### LalaEval: A Holistic Human Evaluation Framework for Domain-Specific Large Language Models (https://arxiv.org/abs/2408.13338)
- **What's New**: 이 논문은 LalaEval이라는 도메인 특정 대형 언어 모델(LLMs)을 위한 인간 평가의 포괄적 프레임워크를 제안합니다. LalaEval은 도메인 명세, 기준 설정, 벤치마크 데이터셋 제작, 평가 루브릭 구축 및 평가 결과 분석을 위한 종합적인 프로토콜을 포함하고 있습니다.

- **Technical Details**: LalaEval의 주요 구성 요소들은 다음과 같습니다: (1) Domain Specification, (2) Criteria Establishment, (3) Benchmark Dataset Creation, (4) Construction of Evaluation Rubrics, (5) Analysis and Interpretation of Evaluation Results. 이 프레임워크는 산업의 특정 요구 사항에 맞춘 표준화된 절차를 제공합니다.

- **Performance Highlights**: LalaEval을 물류 산업에 적용한 사례를 통해, 도메인 특정 LLMs의 평가 기준과 데이터셋, 그리고 성능 차이를 비교 분석하여 모델 선택 및 개발을 안내함으로써 프레임워크의 유용성을 입증합니다.



### The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities (https://arxiv.org/abs/2408.13296)
- **What's New**: 이번 보고서는 대규모 언어 모델(LLMs)의 파인튜닝(fine-tuning) 방법을 이론적 통찰과 실용적 응용을 통합하여 탐색합니다. LLM의 역사적 발전과 다양한 파인튜닝 접근법을 비교하고, 7단계 구조적 파이프라인을 소개하며, 비대칭 데이터셋을 다루고 최적화 기법에 대한 심층적인 분석을 제공합니다.

- **Technical Details**: 보고서는 데이터 준비(data preparation), 모델 초기화(model initialization), 하이퍼파라미터 조정(hyperparameter tuning), 모델 배포(model deployment) 등 7단계의 파인튜닝 프로세스를 체계적으로 설명합니다. 특히 Low-Rank Adaptation(LoRA)와 Half Fine-Tuning과 같은 파라미터 효율적(parameter-efficient) 방법을 통해 성능과 계산 효율성의 균형을 맞추는 기술을 다루며, Proximal Policy Optimization(PPO), Direct Preference Optimization(DPO) 등의 혁신적인 접근법도 소개합니다.

- **Performance Highlights**: LLMs는 번역(translation), 요약(summarization), 대화형 상호작용(conversational interaction) 등 다양한 NLP 작업에서 인간 수준의 성능을 구현할 수 있습니다. 대규모 데이터셋과 계산 능력의 발전 덕분에 LLMs는 특정 도메인에 맞게 조정할 수 있어 실제 응용에서 효과적인 모델 배포가 가능합니다.



### Exploring Bias and Prediction Metrics to Characterise the Fairness of Machine Learning for Equity-Centered Public Health Decision-Making: A Narrative Review (https://arxiv.org/abs/2408.13295)
Comments:
          under review

- **What's New**: 본 논문은 기계 학습(Machine Learning, ML)의 공공 건강 연구에 대한 적용에서 발생하는 알고리즘 편향(algorithmic bias)에 대한 포괄적인 이해가 부족하다는 점을 강조합니다. 이를 기반으로, 연구는 ML이 생성하는 다양한 편향의 종류와 이들을 평가하기 위한 정량적 메트릭s를 탐구합니다.

- **Technical Details**: 연구는 PubMed, MEDLINE, IEEE, ACM Digital Library, Science Direct, Springer Nature 등 데이터베이스에서 2008년부터 2023년까지 발행된 논문들을 검색하였습니다. ML과 공공 및 인구 건강 분야에서의 편향 및 메트릭을 설명하는 72개의 연구를 포함 기준에 맞춰 검토했습니다. 피실험자들의 인종, 성별, 나이와 같은 다양한 인구 통계학적 요소에 대한 ML 모델의 공정성을 평가하는 방법론이 포함되어 있습니다.

- **Performance Highlights**: 모델 평가의 결과, Multi-Layer Perceptron (MLP) 모델이 Precision, Recall, F1 Score 및 Accuracy와 같은 메트릭에서 가장 높은 성능을 보였습니다. 반면, Naive Bayes 모델은 상대적으로 낮은 성능을 보였으며, 이는 특성 독립성 가정 때문에 발생했을 가능성이 있습니다. 이 연구는 ML 모델이 다양한 인구 집단 간의 공정한 예측을 보장하기 위한 평가 프레임워크를 형성하는 데 기여할 것입니다.



### SarcasmBench: Towards Evaluating Large Language Models on Sarcasm Understanding (https://arxiv.org/abs/2408.11319)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)이 풍자(sarcasm) 이해에서 직면한 한계에 대한 종합적인 평가를 제시합니다. 이를 위해 11개의 최신 LLM과 8개의 사전 훈련된 언어 모델(PLMs)의 성능을 다양한 기준 데이터셋에서 비교 분석했습니다.

- **Technical Details**: 연구는 세 가지 프롬프트 방법: zero-shot IO 프롬프트, few-shot IO 프롬프트, chain of thought (CoT) 프롬프트를 사용하여 풍자 감지에서 LLM의 성능을 평가했습니다. 결과적으로, LLM은 감독형 PLM과 비교하여 부족한 성능을 보였으며, GPT-4가 타 모델에 비해 평균 14.0% 이상의 개선을 보여주었습니다.

- **Performance Highlights**: 현재 LLMs는 감독형 PLMs에 비해 풍자 감지 기준에서 부족한 성능을 보이며, 풍자 이해를 개선하기 위한 많은 노력이 필요합니다. GPT-4는 여러 프롬프트 방법에서 지속적이고 유의미하게 다른 LLMs보다 우수한 성능을 발휘했습니다. 또한, few-shot IO 프롬프트 방법이 zero-shot IO 및 few-shot CoT 방식보다 성능이 뛰어남을 확인했습니다.



New uploads on arXiv(cs.IR)

### CURE4Rec: A Benchmark for Recommendation Unlearning with Deeper Influenc (https://arxiv.org/abs/2408.14393)
- **What's New**: 본 논문은 추천 시스템의 머신 언러닝(Recommendation Unlearning) 평가를 위한 최초의 포괄적 벤치마크인 CURE4Rec을 제안합니다. CURE4Rec은 언러닝의 완전성(unlearning Completeness), 추천 유용성(recommendation Utility), 언러닝 효율성(unleaRning efficiency), 추천 공정성(recommendation fairnEss)의 네 가지 요소를 포함하며, 데이터 선택 전략으로는 코어 데이터(core data), 엣지 데이터(edge data), 랜덤 데이터(random data)를 사용합니다.

- **Technical Details**: CURE4Rec은 언러닝 방법의 평가를 위해 각 요소에 대한 특정 메트릭을 세부적으로 설명합니다. 또한, 추천의 공정성과 데이터에 대한 영향도를 고려하며, 다양한 언러닝 세트의 영향을 평가합니다. 세 가지 언러닝 세트(코어, 엣지, 랜덤)를 선택하여 추천 알고리즘의 견고함과 유효성을 평가합니다.

- **Performance Highlights**: CURE4Rec를 통해 수행한 광범위한 실험에서 기존의 추천 언러닝 방법의 성능을 보고하고, 언러닝이 추천 공정성에 미치는 영향을 탐구합니다. 기존의 방법들은 언러닝의 완전성, 효율성 및 모델 유틸리티에 중점을 두었으나, 이 연구는 깊은 영향(공정성)을 상승시킴으로써 추천 시스템의 사용자 경험을 향상시키고자 합니다.



### Are LLM-based Recommenders Already the Best? Simple Scaled Cross-entropy Unleashes the Potential of Traditional Sequential Recommenders (https://arxiv.org/abs/2408.14238)
Comments:
          18 pages. arXiv admin note: substantial text overlap with arXiv:2402.06216

- **What's New**: 최근 대형 언어 모델(LLMs)이 추천 시스템에서 주목받고 있지만, 이 논문은 전통적인 방법이 LLMs에 비해 성능면에서 우세할 수 있음을 입증하고 있습니다.

- **Technical Details**: 하나의 주요 발견으로, cross-entropy (CE) 손실 함수가 추천 시스템의 성능 개선에 기여하지만, 일부 ranking metrics에서는 최적의 경계 조건을 달성하지 못하는 것이 밝혀졌습니다. 이 논문은 Scaled Cross-Entropy (SCE)라는 대안을 제안하여, sampled softmax 손실의 'tightness' 문제를 개선합니다.

- **Performance Highlights**: 실험 결과에 따르면, 기존의 추천 모델이 CE 손실 함수를 사용해 LLMs 대비 성과를 낼 수 있는 가능성이 있음을 보여줍니다. 기존의 방법들이 LLMs보다 더 나은 성과를 낼 수 있다는 점은 LLM 기반 추천 시스템에 대한 과신을 줄이는데 기여할 수 있습니다.



### ColBERT's [MASK]-based Query Augmentation: Effects of Quadrupling the Query Input Length (https://arxiv.org/abs/2408.13672)
Comments:
          5 pages, 3 figures, two tables

- **What's New**: 본 논문에서는 ColBERTv2의 [MASK] 토큰의 사용과 쿼리 증강(query augmentation)의 효과에 대해 다룹니다. 최근 연구들을 바탕으로 [MASK] 토큰이 쿼리의 비-[MASK] 토큰에 대한 가중치를 부여하는 주된 역할을 한다는 주장이 제기되었습니다.

- **Technical Details**: ColBERT 모델은 여러 개의 토큰 임베딩 벡터를 사용하여 쿼리와 문서 간의 정밀한 일치를 지원합니다. 쿼리를 [MASK] 토큰으로 보강함으로써, ColBERT는 문서의 토큰 임베딩과 쿼리의 토큰 임베딩의 최대 유사성을 더해 문서를 순위 매깁니다. 실험에서는 [MASK]의 수를 변경하면서 성능의 변화를 관찰했으며, 쿼리의 평균 길이를 32로 패딩했을 때 성능이 크게 증가함을 발견했습니다.

- **Performance Highlights**: ColBERTv2 모델에서는 [MASK] 토큰의 수가 0부터 4배까지 변화할 때, 성능이 초기에는 감소하다가, 패딩된 쿼리의 평균 길이가 32에 도달하면 큰 성능 향상이 나타나는 것을 관찰했습니다. 쿼리의 길이를 128로 늘렸을 경우의 성능 차이는 작고 통계적으로 유의미하지 않음을 보여주어, ColBERT가 예상보다 많은 [MASK] 토큰을 처리할 수 있음을 나타냅니다.



### Transforming Location Retrieval at Airbnb: A Journey from Heuristics to Reinforcement Learning (https://arxiv.org/abs/2408.13399)
- **What's New**: 이 논문에서는 Airbnb의 위치 검색 시스템의 발전과 도전 과제를 다루고 있습니다. Airbnb는 다양한 게스트의 요구를 충족하는 효율적인 검색 시스템을 구축하기 위한 방법론을 설명하고, 해당 시스템의 기초부터 기계 학습 기반의 위치 검색 제품을 개발하는 과정에서의 도전 과제를 제시합니다.

- **Technical Details**: 구체적으로, 이 논문에서는 헤리틱스(Heuristics), 통계(Statistics), 기계 학습(Machine Learning), 강화 학습(Reinforcement Learning) 접근 방식을 활용하여 위치 검색 문제를 해결하는 방법을 논의합니다. 특히, 초기 데이터가 부족한 상황에서의 '콜드 스타트(Cold Start)' 문제 및 데이터의 일반화(Generalization)와 차별화(Differentiation), 알고리즘적 편향(Algorithmic Bias) 대응 방법을 중점적으로 다룹니다.

- **Performance Highlights**: 연구 결과, Airbnb 플랫폼은 700만 개 이상의 활성 리스팅을 보유하고 있으며, 기계 학습을 통한 위치 검색 개선이 게스트의 검색 경험을 획기적으로 향상시킬 수 있음을 보여줍니다. 특히, 세분화된 검색 파라미터를 통해 지난 예약 기록에서 학습한 내용을 바탕으로 더 적합한 결과를 제공합니다.



### SEQ+MD: Learning Multi-Task as a SEQuence with Multi-Distribution Data (https://arxiv.org/abs/2408.13357)
- **What's New**: 본 논문에서는 전통적인 검색 알고리즘의 한계를 극복하기 위해, e-commerce 분야에서 다중 작업 학습(Multi-task learning, MTL)과 지역별 특징을 결합한 SEQ+MD 프레임워크를 제안합니다. 이 프레임워크는 고객의 쇼핑 선호도와 문화적 전통을 반영하여 글로벌 시장에서 더 효과적으로 작동할 수 있도록 설계되었습니다.

- **Technical Details**: SEQ+MD 프레임워크는 순차적 학습(Sequential learning)과 다중 분포 입력(Multi-distribution input)을 통합합니다. 이 접근법은 다중 작업 간의 상호작용을 고려하여 작업을 순차적 문제로 변환하고, 지역 의존(feature-dependent) 및 지역 불변(feature-invariant) 특징을 분리하여 처리합니다.

- **Performance Highlights**: 사내 데이터를 통한 평가 결과, 구매 관련 성과는 1.8% 개선되었으며, 클릭 성과는 기존 모델 수준을 유지하였습니다. 또한, 제안된 다중 지역 학습 모듈은 '플러그 앤 플레이(Plug-and-play)' 방식으로 다른 MTL 애플리케이션에 쉽게 적용 및 향상될 수 있습니다.



### Contextual Bandit with Herding Effects: Algorithms and Recommendation Applications (https://arxiv.org/abs/2408.14432)
- **What's New**: 이 논문은 추천 시스템 분야에서 사용자의 피드백에 어떤 영향을 미치는 'herding effects'를 고려하여 새로운 개선 방식인 TS-Conf (Thompson Sampling under Conformity) 알고리즘을 제안합니다. 기존의 contextual bandits 알고리즘이 이 피드백 편향을 무시했던 점을 주목하여, 이를 해결하기 위해 사용자 피드백 모델을 개발했습니다. 이는 추천 의사결정에서 탐색(exploration)과 활용(exploitation) 사이의 균형을 맞추는데 기여할 것입니다.

- **Technical Details**: 저자들은 herding effects가 사용자의 피드백에 미치는 바를 수량화하기 위한 모델을 제안하며, 이를 기반으로 하여, TS-Conf 알고리즘을 사용해 posterior sampling 접근 방식을 통해 탐색과 활용의 절충을 효과적으로 구현합니다. 이는 또한 herding effects의 상단 한계를 명시하여, 학습 속도에 미치는 영향을 드러냅니다.

- **Performance Highlights**: TS-Conf 알고리즘은 네 개의 공개 데이터셋에서 실험을 수행한 결과, 하위 선형 하강(regret) 특성을 보여주며, 기존 세 가지 벤치마크 알고리즘보다 월등한 성능을 발휘했습니다. 이를 통해 사용자의 피드백에서 발생하는 부정적인 영향을 효과적으로 완화하여, 더 빠른 학습 속도와 향상된 추천 정확도를 달성했음을 보여줍니다.



### Towards Lifelong Learning Embeddings: An Algorithmic Approach to Dynamically Extend Embeddings (https://arxiv.org/abs/2408.14118)
Comments:
          Accepted Extended Abstract for 3rd Workshop on End-End Customer Journey Optimization at KDD2024, Barcelona, Spain

- **What's New**: 이 논문은 전통적인 embedding 접근 방식의 한계를 해결하기 위해, 전자 상거래의 다이나믹한 특성을 반영하는 모듈형 알고리즘을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 고정된 차원과 입력을 요구하는 기존 embedding 대신, 새로운 제품을 위한 embedding 입력 크기를 확장하면서도 기존의 학습된 지식을 보존하는 방식을 사용합니다. 또한, 초기화 과정에서의 cold start 문제를 완화하기 위한 다양한 전략을 통합하고 있습니다.

- **Performance Highlights**: 초기 실험 결과, 제안된 방법이 기존의 embedding 방식보다 뛰어난 성능을 보여주었으며, 특히 구매 예측 시나리오에서 향상된 결과를 도출했습니다.



### AgentMove: Predicting Human Mobility Anywhere Using Large Language Model based Agentic Framework (https://arxiv.org/abs/2408.13986)
Comments:
          13 pages

- **What's New**: 본 논문에서는 AgentMove라는 새로운 예측 프레임워크를 소개합니다. 이는 인류의 이동 패턴을 보다 효과적으로 예측할 수 있도록 설계되었습니다. 전통적인 딥러닝 모델의 한계를 극복하고, 제로샷(Zero-shot) 예측을 가능케 합니다.

- **Technical Details**: AgentMove는 이동 예측(task)을 세 가지 서브태스크로 분해합니다: (1) 개별 이동 패턴 탐색(spatial-temporal memory module), (2) 도시 구조의 영향 모델링(world knowledge generator), (3) 인구 간의 공유 패턴 포착(collective knowledge extractor). 이 프레임워크는 메모리와 지식 생성을 결합하여 복잡한 이동 패턴을 더 잘 포착합니다.

- **Performance Highlights**: AgentMove는 12개 도시의 이동 데이터를 기반으로 한 실험에서 기존 최적 모델보다 8% 이상 향상된 성능을 기록했습니다. 또한 다양한 LLMs에 대한 높은 적응성과 안정성을 보여주며, 도시 간 예측의 지리적 편향을 줄였습니다.



### HRGraph: Leveraging LLMs for HR Data Knowledge Graphs with Information Propagation-based Job Recommendation (https://arxiv.org/abs/2408.13521)
Comments:
          7 Pages, 4 Figures. View in ACL Anthology: this https URL

- **What's New**: 이 연구에서는 다양한 인사 관리(HR) 문서에서 HR 지식 그래프(Knowledge Graphs, KGs)를 효과적으로 개발하기 위한 프레임워크인 HRGraph를 소개합니다. 이 프레임워크는 대형 언어 모델(Large Language Models, LLMs)을 사용하여 HR 데이터를 처리하고 직업 연결 및 직원 추천을 가능하게 합니다.

- **Technical Details**: HRGraph 프레임워크는 HR 문서(예: 직무 설명서(Job Descriptions, JDs) 및 이력서(CVs))에서 다양한 엔터티를 식별 및 추출하고, 사전 훈련된 BERT 모델을 사용해 노드의 특징(long features)을 추출합니다. 이후 이 데이터를 기반으로 KGs를 구성하여 여러 하위 작업에 활용가능하게 합니다.

- **Performance Highlights**: HR KGs는 고용주와 직원 모두에게 유용한 정확한 직무 연결을 위한 사례를 보여줍니다. 실험 결과, 짧은 시간 내에 정보를 전파하는 능력을 지닌 KGs와 그래프 신경망(Graph Neural Nets, GNNs)을 사용하여 직원 추천, 직무 분류 등 다양한 작업에서 KGs의 효과성을 입증하였습니다.



### Utilizing Large Language Models for Named Entity Recognition in Traditional Chinese Medicine against COVID-19 Literature: Comparative Study (https://arxiv.org/abs/2408.13501)
Comments:
          22 pages with 2 figures

- **What's New**: 이번 연구는 COVID-19 문헌을 대상으로 한 전통 중의학(TCM) 관련 NER(정보 추출) 작업에서 ChatGPT와 최신 LLM(대규모 언어 모델)들의 성능을 비교하였습니다.

- **Technical Details**: 389개의 TCM 관련 COVID-19 논문으로 구성된 데이터셋을 구축하고, 48개 문서에 대해 3개의 도메인, 6개 엔티티 유형으로 수작업 주석을 달았습니다. ChatGPT(GPT-3.5 및 GPT-4)와 RoBERTa, MiniLM, PubMedBERT, SciBERT 등 4개의 최첨단 BERT 기반 QA 모델을 활용해 NER 작업을 수행하였으며, GSAP-NER 모델과 비교하였습니다.

- **Performance Highlights**: ChatGPT는 퍼지 매치(fuzzy match)에서 BERT 기반 QA 모델들보다 5개 작업 중 5개에서 높았으나, 정확한 매치(exact match)에서는 BERT 모델들이 5개 작업 중 5개에서 우세했습니다. GPT-4는 퍼지 매치에서 TCM 포뮬러 및 중국 특허약 관련 엔티티 유형에서 두드러진 장점을 보였으며, GSAP-NER는 RM에서 GPT-4를 다소 초과하는 F-1 점수를 기록했습니다. 전반적으로 LLM의 NER 성능은 엔티티 유형에 따라 크게 달라지며, ChatGPT는 높은 리콜(recall)이 필요한 상황에서 유용한 선택이 될 수 있습니다.



### DrugAgent: Explainable Drug Repurposing Agent with Large Language Model-based Reasoning (https://arxiv.org/abs/2408.13378)
Comments:
          18 pages, 1 figure

- **What's New**: 이번 논문은 기존 약물의 새로운 치료 가능성을 찾기 위해 다중 에이전트 시스템(multi-agent system) 프레임워크를 제안하고 있습니다. 최신 기계 학습(machine learning) 기법과 지식 통합(knowledge integration)을 사용하여 약물 재목적화(drug repurposing) 과정을 향상시키는 데 초점을 둡니다.

- **Technical Details**: 프레임워크는 AI Agent, Knowledge Graph Agent, Search Agent의 세 가지 전문 에이전트로 구성됩니다. AI Agent는 강력한 약물-타겟 상호작용(drug-target interaction, DTI) 모델을 훈련하고, Knowledge Graph Agent는 다양한 데이터베이스를 활용해 DTI를 체계적으로 추출하며, Search Agent는 생물 의학 문헌과 상호작용하여 계산된 예측을 주석 달고 검증합니다. 이 시스템은 외부 데이터베이스에서 얻은 다양한 데이터 소스를 효과적으로 활용합니다.

- **Performance Highlights**: 예비 결과에 따르면, 이 접근법은 약물-질병 상호작용을 예측하는 데 있어 기존 방법들보다 뛰어난 성능을 보이며, 전통적인 약물 발견 과정에 비해 시간과 비용을 줄일 수 있는 가능성을 보여줍니다. 또한, 다중 에이전트 시스템의 확장성을 강조하며, 약물 재목적화 분야에서의 혁신을 촉진하는 역할을 합니다.



### From Zero to Hero: Harnessing Transformers for Biomedical Named Entity Recognition in Zero- and Few-shot Contexts (https://arxiv.org/abs/2305.04928)
Comments:
          Collaboration between Bayer Pharma R&D and Serbian Institute for Artificial Intelligence Research and Development. Artificial Intelligence in Medicine (2024)

- **What's New**: 이 논문에서는 생물 의학 분야에서 제로샷(Zero-shot) 및 몇 샷(Few-shot) Named Entity Recognition (NER) 방법을 제안합니다. 이 방법은 다중 클래스 토큰 분류를 이진 토큰 분류로 변환하고, 대량의 데이터셋과 생물 의학 엔티티에 대한 사전 훈련을 통해 모델이 주어진 엔티티 레이블과 잠재적인 새로운 엔티티 레이블 간의 의미적 관계를 학습할 수 있도록 합니다.

- **Technical Details**: 우리는 PubMedBERT 기반 모델을 미세 조정하여 9개의 다양한 생물 의학 엔티티에서 제로샷 NER에 대해 35.44%, 원샷 NER에 대해 50.10%, 10샷 NER에 대해 69.94%, 100샷 NER에 대해 79.51%의 평균 F1 점수를 달성했습니다. 기존 변환기(transformer) 기반 방법보다 우수하며, 1000배 적은 매개변수를 사용하는 GPT3 기반 모델과 비교할 수 있습니다.

- **Performance Highlights**: 이 연구의 결과는 제로샷 및 몇 샷 학습을 통해 새로운 생물 의학 엔티티를 효과적으로 인식하는 방법의 유효성을 입증합니다. 개발된 모델은 공개적으로 제공되어 향후 연구 및 응용 프로그램에서 활용될 수 있습니다.



New uploads on arXiv(cs.CV)

### A Practitioner's Guide to Continual Multimodal Pretraining (https://arxiv.org/abs/2408.14471)
Comments:
          Technical Report. 52 pages

- **What's New**: 이 논문에서는 여러 분야에 걸친 시각과 언어의 교차점에서 작동하는 멀티모달(Multimodal) 기초 모델의 지속적(pretraining) 업데이트를 위한 새로운 접근 방식을 제안합니다. 특히 실제 배포 요구 사항을 고려한 지속적인 멀티모달 프리트레이닝 벤치마크인 FoMo-in-Flux를 소개합니다.

- **Technical Details**: FoMo-in-Flux는 63개의 다양한 데이터 세트를 기반으로 구성되어 있으며, 실제적인 컴퓨팅 제약(compute constraints)과 배포 요구사항(practical deployment requirements)을 반영합니다. 본 연구에서는 데이터 중심(data-centric) 조사, 방법 중심(method-centric) 전략, 메타 학습 속도 조절(meta learning rate schedules), 모델 및 컴퓨트 스케일링의 영향을 포함한 다양한 관점에서 지속적 프리트레이닝을 탐구합니다.

- **Performance Highlights**: 이 연구는 지속적 멀티모달 프리트레이닝을 위한 실무자 가이드를 제공하며, 실제 환경에서의 배포를 위한 여러 접근 방식을 고찰합니다. 논문과 함께 제공된 벤치마크와 코드는 연구자 및 실무자들이 유용하게 활용할 수 있습니다.



### Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos (https://arxiv.org/abs/2408.14469)
- **What's New**: 이 논문은 긴 형태의 자아 중심 비디오에서의 다중 홉 비디오 질문 답변(Multi-Hop Video Question Answering, MH-VidQA) 문제를 다루고 있습니다. 새로운 데이터셋인 MultiHop-EgoQA를 구축하고, 비디오에서 시간 간격을 로컬라이징하여 시각적 증거로 활용하는 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 'Grounding Scattered Evidence with Large Language Model (GeLM)'라는 새로운 아키텍처를 도입하여 다중 모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 성능을 향상시킵니다. GeLM은 비디오에서 시간 증거를 검색하기 위해 유연한 grounding token을 통합한 grounding 모듈을 포함하고 있습니다. 이를 통해 시각적 지침 데이터로 훈련된 GeLM은 다중 홉 기반의 grounding 및 reasoning 능력을 개선하고, 새로운 기준을 설정합니다.

- **Performance Highlights**: GeLM은 자동으로 구축된 시각적 지침 데이터로 훈련되었으며, 다중 홉 reasoning 및 grounding에서 현저한 개선을 보여줍니다. 또한, third-person view 비디오에서 훈련된 동일한 아키텍처가 ActivityNet-RTL 단일 홉 VidQA 벤치마크에서 최신 성과를 달성하여 그 효용성을 입증했습니다.



### Dense Center-Direction Regression for Object Counting and Localization with Point Supervision (https://arxiv.org/abs/2408.14457)
Comments:
          Published in Pattern Recognition

- **What's New**: 본 연구에서는 CeDiRNet이라 불리는 새로운 점 감독(point-supervision) 학습 접근 방식을 제안하며, 주변 픽셀로부터 객체 중심을 가리키는 방향을 조밀하게 회귀하여 객체의 정확한 탐지 및 세기(counting) 문제를 해결하고자 합니다.

- **Technical Details**: CeDiRNet에서는 객체 중심에 대한 중심 방향(center-directions)을 회귀합니다. 이로 인해 근처 픽셀뿐만 아니라 더 많은 지원을 받게 되며, 이는 단일의 경량화된 로컬라이제이션 네트워크를 사용하여 최종 로컬라이제이션 작업을 수행할 수 있는 구조를 가집니다. 이 네트워크는 합성 데이터로 훈련 가능하며 대상 도메인과 완전히 독립적입니다.

- **Performance Highlights**: 제안된 방법은 6개의 다양한 데이터셋에서의 객체 세기 및 로컬라이제이션 작업에서 기존의 최첨단 방법들을 초월하는 성능을 보였습니다. 또한, 이 연구는 주어진 성능을 달성하기 위해 복잡한 후처리 방법(cluster나 Hough voting 등)을 피할 수 있음을 보여줍니다.



### Center Direction Network for Grasping Point Localization on Cloths (https://arxiv.org/abs/2408.14456)
Comments:
          Accepted for publication in IEEE Robotics and Automation Letters

- **What's New**: 제안된 CeDiRNet-3DoF는 천 소재의 그립 포인트(grasp point) 탐지에 대한 새로운 딥러닝 모델로, ICRA 2023의 Cloth Manipulation Challenge에서 첫 번째를 차지했습니다. 또한, ViCoS Towel Dataset이라는 8,000개의 실제 이미지와 12,000개의 합성 이미지로 구성된 데이터셋을 소개하여 유효한 벤치마크를 제공합니다.

- **Technical Details**: CeDiRNet-3DoF는 중심 방향 회귀(center direction regression)와 로컬리제이션 네트워크(localization network)를 활용하여 그립 포인트 탐지를 수행합니다. 이 모델은 최적의 그립 방향(즉, 접근 각도)을 추정할 수 있는 3-DoF(3 degrees of freedom) 확장을 통해 고안되었습니다. 제안된 방법론은 다양한 조명 조건, 배경, 그리고 장애물이 있는 이미지를 포함하는 ViCoS Towel Dataset 상에서 평가되었습니다.

- **Performance Highlights**: CeDiRNet-3DoF는 기존의 최신 변환기(transformer) 기반 모델을 포함한 여러 최첨단 방법들을 능가하는 성능을 보였습니다. 이 연구는 천소재의 그립 탐지 문제에서 로봇 비전과 로보틱스 분야의 중요한 격차를 해소하고, 효율적인 방법 비교 및 평가를 지원하는 강력한 솔루션과 벤치마크를 제공합니다.



### Model Parallel Training and Transfer Learning for Convolutional Neural Networks by Domain Decomposition (https://arxiv.org/abs/2408.14442)
- **What's New**: 이번 논문에서는 데이터 병렬화 방법의 효과성을 강조하며, CNN-DNN 아키텍처의 성능 향상을 통해 복잡한 이미지 처리 문제에 더 뛰어난 해결책을 제공하고자 하였습니다. 특히, 모델 파라미터 수를 줄이고 훈련 시간을 단축하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문에서는 입력 이미지를 작은 하위 이미지로 분해하여, 각 하위 이미지에 대해 파라미터 수가 줄어든 로컬 CNN을 병렬로 훈련시키는 새로운 CNN-DNN 아키텍처를 제안합니다. 이후, 이들 로컬 CNN의 분류 결과는 DNN을 통해 집계되어 최종 결정을 내립니다. 논문에서는 모델 파라미터를 초기 값으로 이용하여 전이 학습을 적용한 방법도 탐구하고 있습니다.

- **Performance Highlights**: 이 연구는 제안된 CNN-DNN 모델이 기존의 방법들보다 더 나은 분류 정확도를 달성하며, 평균 확률 분포 계산 및 다수결 방법과 비교하여 결과를 제시합니다. 추가적으로, 모델을 통합하여 단일 cohesive 모델로서 훈련했을 때의 분류 정확성에 대한 성과도 포함되어 있습니다.



### Attend-Fusion: Efficient Audio-Visual Fusion for Video Classification (https://arxiv.org/abs/2408.14441)
- **What's New**: 본 연구에서는 Attend-Fusion이라는 오디오-비주얼(AV) 융합 접근법을 제안합니다. 이 접근법은 컴팩트 모델 아키텍처를 통해 비디오 데이터에서 오디오-비주얼 관계를 효과적으로 캡처하며, YouTube-8M 데이터셋에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: Attend-Fusion은 72M 파라미터를 사용하여 75.64%의 F1 점수를 달성합니다. 이는 341M 파라미터를 가진 기존의 Fully-Connected Late Fusion 모델의 75.96% F1 점수와 비슷한 성능을 보여주며, 모델 크기를 80% 이상 줄이는 데 성공하였습니다.

- **Performance Highlights**: Attend-Fusion 모델은 오디오와 비주얼 정보를 효과적으로 결합하여 비디오 분류 작업에서 높은 성능을 달성하며, 리소스가 제한된 환경에서도 고성능 비디오 이해 시스템의 배포가 가능하다는 점에서 큰 의미가 있습니다.



### Social perception of faces in a vision-language mod (https://arxiv.org/abs/2408.14435)
- **What's New**: 이 논문은 CLIP 모델을 통해 인간 얼굴에 대한 사회적 인식을 탐구하고, 다양한 얼굴 속성과 텍스트 프롬프트 간의 유사성을 비교합니다. 연구는 얼굴 이미지를 체계적으로 변형하여 사회적 인식에 미치는 영향을 실험적으로 분석하였습니다.

- **Technical Details**: CLIP(Contrastive Language–Image Pre-training) 모델을 사용하여 텍스트와 이미지 쌍의 임베딩(embedding) 간의 코사인 유사성을 측정하여 사회적 판단을 분석합니다. 연구에서는 나이, 성별, 인종, 표정, 조명, 자세 등의 속성을 독립적으로 조작하여 실험을 진행했습니다. 논문에서 제시된 데이터셋은 CausalFace로, GAN(Generative Adversarial Network)을 이용한 합성 이미지들로 이루어져 있습니다.

- **Performance Highlights**: CLIP 모델은 다양한 속성과 관련하여 강력한 사회적 편향을 보이며, 특히 흑인 여성의 얼굴에 대한 극단적인 사회적 인식 값을 생성합니다. 표정이 나이와 조명보다 사회적 인식에 더 큰 영향을 미친다는 발견은 기존 연구의 편향을 잘못된 결론으로 이끌 수 있음을 내포합니다.



### Few-Shot 3D Volumetric Segmentation with Multi-Surrogate Fusion (https://arxiv.org/abs/2408.14427)
Comments:
          Accepted to MICCAI 2024

- **What's New**: 이번 논문에서는 MSFSeg라는 혁신적인 few-shot 3D segmentation 프레임워크를 소개합니다. 이 프레임워크는 제한된 수의 주석이 달린 2D 슬라이스 또는 3D 시퀀스 세그먼트를 사용하여 보지 못한 3D 물체를 자동으로 분할할 수 있습니다.

- **Technical Details**: MSFSeg는 다양한 환자들 간의 해부학적 상관관계를 학습하여, 다수의 서포트 시각적 패턴을 활용해 비주얼마스크를 정확하게 예측합니다. Multi-Surrogate Fusion (MSF) 모듈을 통해 라벨이 없는 슬라이스와 몇 개의 라벨이 있는 슬라이스 간의 형태적(모포로지컬) 상관관계를 정교하게 마이닝합니다.

- **Performance Highlights**: MSFSeg는 기존의 few-shot segmentation 벤치마크에서 비교 우위를 보였으며, 복잡한 튜블 구조와 같은 도전을 요하는 객체를 세분화하는 데 있어 뛰어난 성능을 입증했습니다.



### Evaluating saliency scores in point clouds of natural environments by learning surface anomalies (https://arxiv.org/abs/2408.14421)
- **What's New**: 이 논문에서는 자연 환경에서의 삼차원 포인트 클라우드(3D point clouds)에서 관심 객체를 구별하기 위한 기법을 제안합니다. 특히, 주변 환경과 얼마나 차별화되는지를 평가하여 기하학적 두드러기(geometric salience)를 측정합니다.

- **Technical Details**: 제안된 방법은 딥 뉴럴 네트워크(DNN)를 사용하여 표면을 재구성하는 방식으로, 원본 포인트 클라우드와 재구성된 표면 간의 차이를 통해 이상치를 탐지합니다. 이 과정에서 포괄적인 잡음(noise)과 질감이 있는 표면을 처리하며, 정규(phased) 내적(predict) 정보를 기반으로 합니다.

- **Performance Highlights**: 다양한 자연 시나리오에서 실행된 실험을 통해 재구성 오차와 두드러진 객체(salient object) 간의 강한 상관관계를 입증하였습니다. 또한, 이 방법은 기존의 분류(classification) 문제로 고려됐던 기존의 접근 방식과 달리, 사전 학습된 분류기가 필요하지 않아 유연하게 적용될 수 있습니다.



### LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation (https://arxiv.org/abs/2408.14415)
Comments:
          20 pages

- **What's New**: 본 논문에서는 LoG-VMamba를 제안하며, 이는 의료 영상 분할(MIS) 작업에 대한 고차원 배열에서의 지역적(local) 및 전역적(global) 의존성을 유지하기 위한 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: LoG-VMamba는 Local Token eXtractor (LTX)와 Global Token eXtractor (GTX)를 포함하여 채널 축에서 공간적으로 인접한 토큰들을 가깝게 유지하게끔 설계되었습니다. 이 방법은 기존의 복잡한 스캐닝 전략을 사용하지 않고도 로컬 및 글로벌 컨텍스트에 접근할 수 있도록 합니다.

- **Performance Highlights**: 제안된 LoG-VMamba 모델은 다양한 2D 및 3D MIS 작업에서 CNN 및 Transformer 기반의 기준 모델을 상당히 초월하여 계산적으로 효율적입니다.



### Satellite Sunroof: High-res Digital Surface Models and Roof Segmentation for Global Solar Mapping (https://arxiv.org/abs/2408.14400)
Comments:
          14 pages

- **What's New**: 본 논문은 Google의 Solar API의 지리적 범위를 확장하기 위해 위성 이미지를 활용하여 전 세계의 태양광 잠재력을 평가할 수 있도록 구상하고 있습니다. 이는 특히 고급 항공 이미지를 사용한 기존 프레임워크의 잠재력을 크게 증대시키며, 10억 개 이상의 신규 건물에 대한 분석을 가능하게 합니다.

- **Technical Details**: 모델은 저해상도 위성 이미지를 처리하는 동시에 고해상도 항공 데이터의 라벨을 활용하여 Digital Surface Model (DSM)과 지붕 세분화를 수행합니다. 본 연구에서는 U-Net 스타일의 아키텍처와 Swin Transformer 인코더를 사용하여 다양한 예측 헤드를 통해 지붕 세분화와 고도 맵을 회귀합니다.

- **Performance Highlights**: 모델은 건물에 대한 ~1m DSM MAE, 지붕 경사 ~5도 오류, 지붕 세분화에서 ~56% IOU를 기록하며, Solar API의 성과를 크게 향상시킵니다. 최종적으로, 태양광 가능성을 평가하고 최적의 태양광 패널 배치 구성을 제안할 수 있는 정확한 DSM과 지붕 세그먼트를 제공합니다.



### SelEx: Self-Expertise in Fine-Grained Generalized Category Discovery (https://arxiv.org/abs/2408.14371)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 논문에서는 Generalized Category Discovery (GCD)를 다루며, 이 과정에서 아직 알려지지 않은 카테고리를 찾아내고 기존 카테고리를 정확히 분류하려고 합니다. 특히, 기존 방법들이 미세한 카테고리 구분에서 부족한 성능을 보이는 문제를 해결하기 위해 `self-expertise`라는 새로운 개념을 제안합니다.

- **Technical Details**: `self-expertise`는 모델이 미세한 차이를 인식하고 새로운 카테고리를 발견하는 능력을 향상시킵니다. 이 방법은 자율학습(unsupervised)과 감독학습(supervised) `self-expertise` 전략을 결합하여 모델의 판단력과 일반화 능력을 개선합니다. 초기에 계층적 의사 레이블링(hierarchical pseudo-labeling)을 통해 `soft supervision`을 제공하여 self-expertise의 효과를 높입니다. 또한, 감독 기술은 전통적인 방법과 다르게 더 추상적인 긍정(positive) 및 부정(negative) 샘플을 사용하여 클러스터를 형성하고 새로운 카테고리로 일반화하는 데 도움을 줍니다.

- **Performance Highlights**: 실험적으로, 우리는 제안된 방법이 여러 미세한 데이터셋에서 기존의 최첨단 기법보다 더 뛰어난 성능을 보임을 확인했습니다. 이러한 결과는 이론적 통찰에 의해 뒷받침됩니다.



### An Embedding is Worth a Thousand Noisy Labels (https://arxiv.org/abs/2408.14358)
Comments:
          Preprint submitted to the International Journal of Computer Vision (IJCV)

- **What's New**: 본 논문에서는 기존의 Label Noise (라벨 노이즈) 문제를 해결하기 위해 WANN (Weighted Adaptive Nearest Neighbor) 접근 방식을 제안합니다. 이 접근 방식은 Self-Supervised (자기 감독) feature representations를 통한 강력한 데이터 표현을 기반으로 하며, 이는 효율적인 데이터 레이블링을 가능하게 합니다.

- **Technical Details**: WANN은 신뢰도 점수(reliability score)를 도입하여 데이터 레이블의 정확성을 측정합니다. 이 점수는 k-Nearest Neighbor (k-NN) 알고리즘을 통해 적응적으로 선택된 훈련 샘플의 가중치를 조정하는 데 사용됩니다. WANN은 다양한 크기와 노이즈 유형이 혼합된 데이터셋에서 안정적인 성능을 발휘합니다.

- **Performance Highlights**: WANN은 기존의 ANN (Adaptive-NN) 및 고정 k-NN 방법보다 우수한 일반화를 보이며, 노이즈가 있는 비대칭 데이터에서도 10배 및 100배 더 작은 이미지 임베딩을 사용하여 분류 성능을 크게 향상시킵니다. 이 접근 방식은 처리 속도와 저장 공간의 효율성을 개선하여 깊은 신경망 훈련의 한계를 극복하는 간단하면서도 강력한 솔루션으로 부각됩니다.



### Deep learning-based ecological analysis of camera trap images is impacted by training data quality and siz (https://arxiv.org/abs/2408.14348)
- **What's New**: 본 연구에서는 카메라 트랩을 통한 야생 동물 이미지 분석에서 딥 뉴럴 네트워크(deep neural network)의 효과를 분석하고, 모델 훈련의 결정들이 생태학적 메트릭(ecological metrics)에 미치는 영향을 평가합니다.

- **Technical Details**: 연구는 아프리카 사바나와 아시아 아열대 건조 숲의 카메라 트랩 데이터 비교 분석을 통해 진행되었습니다. 모델 아키텍처(model architecture), 훈련 데이터의 노이즈(noise), 데이터셋 크기(dataset size)의 다양한 요소가 종 풍부도(species richness), 점유율(occupancy), 활동 패턴(activity patterns) 같은 생태학적 메트릭에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 결과적으로, 모델 아키텍처의 영향은 미미했지만, 노이즈가 많거나 데이터셋 크기가 줄어들 경우 생태학적 메트릭에 중대한 영향을 미쳤습니다. 그러나 우리 모델은 종 라벨의 10% 오류와 훈련 세트 크기의 50% 감소에도 각 생태학적 메트릭이 크게 변하지 않는 내성을 보였습니다.



### A Brief Analysis of the Iterative Next Boundary Detection Network for Tree Rings Delineation in Images of Pinus taeda (https://arxiv.org/abs/2408.14343)
Comments:
          Submitted to IPOL ad an MLBriefs paper

- **What's New**: 이 논문은 Gillert et al.이 CVPR-2023에서 제안한 INBD 네트워크를 소개하고, 스마트폰으로 촬영한 Pinus taeda 단면의 RGB 이미지에서 나무의 나이테를 구분하는 데의 적용을 연구합니다. 이 연구는 훈련에 사용된 이미지와는 다른 특성을 가진 이미지를 대상으로 하며, 두 단계로 나뉘어진 방법론을 적용합니다.

- **Technical Details**: INBD 네트워크는 U-Net 구조를 기반으로 하며 두 단계로 구성됩니다: 첫 번째 단계에서는 배경, 심재(pith), 및 나이테 경계를 분할합니다. 두 번째 단계에서는 이미지를 극좌표계로 변환하여 심재에서 나무 껍질까지 반복적으로 나이테 경계를 분할합니다. 이 방법에서 얻은 F-Score는 77.5, mAR은 0.540, ARAND는 0.205입니다.

- **Performance Highlights**: 제안된 INBD 방법은 다양한 해상도의 나무 단면 이미지에서 나이테 경계를 효과적으로 구분합니다. UruDendro 데이터셋을 활용하여 훈련된 INBD 모델은 특정한 조건 속에서 최적의 성능을 발휘하였으며, 특히 중간 및 두꺼운 나이테의 분할에서 높은 정확도를 나타냈습니다.



### DuDoCROP: Dual-Domain CLIP-Assisted Residual Optimization Perception Model for CT Metal Artifact Reduction (https://arxiv.org/abs/2408.14342)
Comments:
          14 pages, 18 figures

- **What's New**: 최근 메탈 아티팩트 감소(MAR) 문제를 해결하기 위한 새로운 접근 방식으로, 시각-언어 모델(VLM)과 이중 도메인 CLIP 지원 잔여 최적화 인식 모델(DuDoCROP)을 제안합니다. 이 모델은 다양한 형태의 금속 임플란트 아티팩트를 인식하는 능력을 개선합니다.

- **Technical Details**: DuDoCROP 모델은 이미지 도메인과 시노그램 도메인 모두에서 대조 학습을 활용하여 해부학적 구조와 메탈 아티팩트에서의 의미적 설명을 추출하고, 이를 기반으로 하여 확산 모델이 이중 도메인 사전 생성을 수행하도록 유도합니다. 또한, 프롬프트 엔지니어링을 통해 더 정밀한 이미지-텍스트 설명을 생성하여 모델의 인식 능력을 강화합니다.

- **Performance Highlights**: DuDoCROP 모델은 기존 모델보다 최소 63.7% 더 높은 일반화 능력을 보여주며, 정량적 및 정성적 평가 모두에서 다른 최신 방법들보다 우수한 성능을 나타냅니다.



### ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty (https://arxiv.org/abs/2408.14339)
Comments:
          43 pages

- **What's New**: 새로운 벤치마크인 ConceptMix는 Text-to-Image (T2I) 모델의 조합 생성 능력을 자동으로 평가하는 시스템입니다. 기존의 인적 텍스트 프롬프트나 고정 템플릿에 의존하지 않고, 다양한 시각적 개념을 결합하여 더 복잡하고 다양한 프롬프트를 생성합니다.

- **Technical Details**: ConceptMix는 두 단계로 구성됩니다. 첫 번째 단계에서 GPT-4o를 사용하여 무작위로 추출한 물체와 k 개의 시각적 개념을 결합하여 텍스트 프롬프트를 생성합니다. 두 번째 단계에서는 생성된 이미지가 얼마나 많은 개념을 정확하게 표현했는지를 평가합니다.

- **Performance Highlights**: ConceptMix는 다양한 T2I 모델에 대해 높은 판별력을 나타내며, k 값이 증가할수록 여러 모델의 성능이 현저히 감소하는 것을 보여줍니다. 특히 공개 모델의 성능 저하가 두드러지며, 기존 데이터셋의 프롬프트 다양성 부족 문제를 드러냅니다.



### PHEVA: A Privacy-preserving Human-centric Video Anomaly Detection Datas (https://arxiv.org/abs/2408.14329)
- **What's New**: PHEVA는 인간 중심의 윤리적 비디오 이상 탐지(VAD)를 위한 가장 큰 데이터셋으로, 사생활을 보호하는 특징을 가지고 있다. 이는 전통적인 데이터셋들과는 달리, 개인 식별 정보를 제거하고 비식별화된 인간 주석만을 제공한다.

- **Technical Details**: PHEVA 데이터셋은 7개의 실내/실외 장면을 포함하며, 6개의 기초 카메라와 법 집행 훈련을 위한 컨텍스트 특정 카메라를 채용하여 다양한 환경을 포괄한다. 또한, 훈련 프레임 수는 이전 최대 데이터셋보다 5배 이상 많으며, 82.14%의 경우에서 기존 방법들을 초월한 결과를 제공한다. 데이터셋은 10% 오류 비율(10ER)과 같은 새로운 메트릭을 포함하여 포즈 기반 VAD 알고리즘을 위한 포괄적 벤치마킹을 수행할 수 있도록 설계되었다.

- **Performance Highlights**: PHEVA는 지속적 학습을 위한 벤치마크를 처음 도입하며, 이는 기존의 전통적인 방법에 비해 성능이 82.14% 더 우수함을 보여준다. 새로운 10ER 메트릭은 실제 환경에서의 잘못된 긍정 및 부정의 비용 차이를 고려하여 비정형 탐지의 정확성을 개선한다.



### Streamline tractography of the fetal brain in utero with machine learning (https://arxiv.org/abs/2408.14326)
- **What's New**: 이 연구는 처음으로 태아의 차도(tractography)를 위한 머신러닝 모델을 개발하여, 태아의 백질 섬유를 재구성하는 혁신적인 접근 방법을 제공합니다.

- **Technical Details**: 모델 입력은 다섯 가지 정보 출처로 구성되어 있습니다: (1) 확산 텐서를 사용하여 추정한 섬유 방향, (2) 최근 전파 단계의 방향, (3) 뇌 피질의 주요 지점까지의 거리로 인코딩된 전역 공간 정보, (4) 조직 분할 정보, (5) 아틀라스를 통해 제공된 예상 지역 섬유 방향에 대한 prior 정보. 이 모델은 컨볼루션 및 어텐션 신경망 모듈을 사용하여 현재 지점 주변의 대규모 공간 문맥을 인코딩합니다.

- **Performance Highlights**: 제안한 방법은 평가된 모든 트랙에서 우수한 성능을 보였으며, 태아의 정상 및 비정상적인 뇌 발달 연구에 대한 dMRI의 능력을 크게 발전시킬 수 있습니다.



### Learning Local Pattern Modularization for Point Cloud Reconstruction from Unseen Classes (https://arxiv.org/abs/2408.14279)
Comments:
          14pages, 11figures, accepted by ECCV 2024

- **What's New**: 본 논문에서는 2D 이미지를 기반으로 보지 않은 클래스의 3D 포인트 클라우드를 재구성하는 데 있어 로컬 패턴 모듈화(local pattern modularization)를 학습함으로써, 높은 재구성 정확도와 뛰어난 일반화 능력을 동시에 달성하기 위한 방법을 제안합니다.

- **Technical Details**: 기존의 메서드는 객체 중심 좌표계에서의 글로벌 프라이어(global prior)를 통해 3D 형태를 재구성했으나, 보지 않은 클래스에 대해 일반화하는 데 한계가 있었습니다. 본 연구에서는 보지 않은 클래스에 대해 클래스 무관(class-agnostic)인 로컬 프라이어(local prior)를 학습하여 포인트 클라우드를 재구성하는 새로운 접근 방식을 제시합니다. 초기 재구성을 한 후 학습된 지역 패턴을 사용하여 각 지역을 모듈화(modularize)하며, 입력 이미지에 따라 로컬 패턴 모듈화를 세부적으로 조정(customize)함으로써 고충실도(huge fidelity) 포인트 클라우드를 재구성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 널리 사용되는 벤치마크에서 보지 않은 클래스의 형태에 대하여 최첨단(state-of-the-art) 재구성 정확도를 달성하였습니다. 또한, 카메라 파라미터(camera parameters)나 추가적인 정보 없이도 3D 포인트 클라우드를 효과적으로 재구성할 수 있음을 보여주었습니다.



### Text3DAug -- Prompted Instance Augmentation for LiDAR Perception (https://arxiv.org/abs/2408.14253)
Comments:
          Accepted at the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: Text3DAug는 텍스트를 기반으로 인스턴스를 생성하고 주석을 추가하는 최초의 완전 자동화된 방법으로, 데이터에 대한 레이블 없이 3D 모델 생성에 generative models를 활용합니다.

- **Technical Details**: 기존의 인스턴스 증강 방법은 레이블이 필요하고, 시간이 많이 소요되는 CAD 모델링 및 수작업 데이터 주석이 필요했습니다. Text3DAug는 이러한 요구를 없애고, 다양한 LiDAR 센서에서 신뢰할 수 있는 인스턴스 생성과 배치를 자동으로 수행하여 센서에 관여하지 않는 효과적인 데이터 증강을 제공합니다.

- **Performance Highlights**: LiDAR 분할 및 탐지 벤치마크에서 Text3DAug는 기존의 방법들을 보완하거나 독립적으로 사용할 수 있는 효과적인 성능을 보여주었으며, 기존 방법들의 특정 단점을 극복하면서도 동등하거나 더 나은 성능을 발휘했습니다.



### Beyond Few-shot Object Detection: A Detailed Survey (https://arxiv.org/abs/2408.14249)
Comments:
          43 pages, 8 figures

- **What's New**: 이 서베이 논문은 기존의 전통적인 경우보다 적은 양의 데이터로 학습할 수 있는 ‘few-shot object detection (FSOD)’에 대한 포괄적인 검토를 제공합니다. 특히 표준 FSOD, 일반화 FSOD, 증분 FSOD, 오픈셋 FSOD, 도메인 적응 FSOD와 같은 다양한 FSOD 설정에 중점을 두었습니다.

- **Technical Details**: FSOD는 적은 수의 주석이 달린 예제만을 가지고 모델이 새로운 객체 카테고리에 빠르게 적응할 수 있도록 합니다. 이 서베이는 FSOD의 여러 변형들을 고찰하며, 각 접근법의 세부적인 평가 프로토콜을 분석합니다.

- **Performance Highlights**: FSOD 접근법은 의료 영상 진단, 야생동물 보호, 산업 검사, 보안 감시 등 다양한 실제 응용 분야에서 유용하게 사용될 수 있으며, 제한된 데이터 상황에서도 높은 성능을 유지할 수 있는 모델 개발의 중요성을 강조합니다.



### Cascaded Temporal Updating Network for Efficient Video Super-Resolution (https://arxiv.org/abs/2408.14244)
Comments:
          Project website: this https URL

- **What's New**: 기존의 Video Super-Resolution (VSR) 방법들은 일반적으로 recurrent propagation network를 채택하여 전체 비디오 시퀀스의 spatio-temporal 정보를 추출하지만, 모델의 효율성에 상당한 영향을 미친다. 이 논문에서는 자원 제약이 있는 장치에서도 배포할 수 있는 compact하고 효율적인 VSR 방법인 cascaded temporal updating network (CTUN)을 제안한다.

- **Technical Details**: CTUN은 인접 프레임에서 spatio-temporal correspondence를 탐지하기 위해 implicit cascaded alignment module (ICAM)을 개발하고, long-range temporal 정보를 효율적으로 탐색하기 위한 unidirectional propagation updating network를 제안한다. 이 네트워크는 forward propagation 동안 미래 정보를 활용하여 hidden features를 업데이트하는 hidden updater(HU)를 포함하여 inference 시간을 상당히 줄이면서 성능을 유지한다.

- **Performance Highlights**: 실험 결과, CTUN은 기존의 방법들에 비해 효율성과 성능 간의 균형을 잘 이루며, BasicVSR에 비해 약 30%의 파라미터 및 실행 시간을 사용하면서 더 나은 성능을 보인다.



### Gallery-Aware Uncertainty Estimation For Open-Set Face Recognition (https://arxiv.org/abs/2408.14229)
- **What's New**: 이 논문은 오픈 세트 얼굴 인식(Open-set Face Recognition, OSFR)에서 불확실성 추정을 위한 새로운 방법론을 제안합니다. 기존 연구는 주로 얼굴 검증(face verification)에 초점을 맞춰, OSFR의 불확실성을 잘 다루지 못했습니다. 본 연구는 이미지 임베딩과 갤러리(gallery) 내 클래스 중첩으로 인한 불확실성을 고려하는 Bayesian 확률 모델을 도입합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 불확실성 출처를 인식합니다: (1) 갤러리 클래스 중첩으로 인한 갤러리 불확실성과 (2) 얼굴 임베딩의 불확실성. 이를 통해 임베딩 분포의 Bayesian 모델을 사용하여 불확실성을 추정합니다. 논문에서는 Laplace 근사를 통해 확률 분포의 후방 확률을 계산하고, vMF(von Mises–Fisher) 및 Power 분포를 사용하여 각 테스트 이미지에 대한 카테고리 분포를 정의합니다. 이로써, 주어진 이미지의 품질과 임베딩 상대적 위치를 함께 고려합니다.

- **Performance Highlights**: 제안된 방법은 IJB-C 데이터셋과 같은 도전적인 오픈 세트 얼굴 인식 데이터셋에서 테스트 되었으며, 이미지 품질 정보만을 기반으로 한 불확실성 추정 방법들보다 오류 인식에서 더 우수한 성능을 보였습니다. 또한 새로운 오픈 세트 인식 프로토콜을 개발하여, 얼굴 외의 범위에서도 적용 가능성을 제시합니다.



### TC-PDM: Temporally Consistent Patch Diffusion Models for Infrared-to-Visible Video Translation (https://arxiv.org/abs/2408.14227)
Comments:
          Technical report

- **What's New**: 본 논문은 적외선(Infrared) 영상에서 가시광선(Visible) 영상으로의 변환을 위해 새로운 Diffusion 방법인 Temporally Consistent Patch Diffusion Models (TC-DPM)을 제안합니다. 이 방법은 Semantic-guided denoising과 Temporal blending module을 포함하여 서로 다른 프레임 간의 일관성을 보장합니다.

- **Technical Details**: TC-DPM은 정보를 보존하기 위해 foundational segmentation 모델을 활용하여 생성된 가시 이미지를 딥러닝 방식으로 최적화합니다. 주요 기술적 특징으로는 semantic-guided denoising과 dense correspondences에 기반한 temporal blending module이 있습니다. 이로 인해 프레임 간의 자연스러운 전환을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, TC-DPM은 적외선에서 가시광선으로의 비디오 변환에서 FVD에서 35.3% 향상된 성능을 보였으며, 야간 물체 탐지에서 AP50에서 6.1% 개선된 성능을 기록하였습니다.



### MagicMan: Generative Novel View Synthesis of Humans with 3D-Aware Diffusion and Iterative Refinemen (https://arxiv.org/abs/2408.14211)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문은 MagicMan을 소개하며, 이는 단일 참조 이미지로부터 고품질의 새로운 뷰 이미지를 생성하기 위해 설계된 인체 전용 다중 뷰 확산 모델입니다.

- **Technical Details**: MagicMan은 사전 훈련된 2D diffusion 모델과 SMPL-X 모델을 활용하여 3D 일관성을 증진시킵니다. 혼합 다중 뷰 주의(hybrid multi-view attention) 메커니즘을 도입하여 정보 교환을 간소화하고, 기하학적으로 인식 가능한 이중 분기(dual branch)를 통해 RGB와 노멀 도메인에서 동시 생성을 수행합니다. SMPL-X의 정확성을 개선하기 위한 반복적 정제(iterative refinement) 전략도 제안되었습니다.

- **Performance Highlights**: 광범위한 실험 결과가 MagicMan이 기존 접근 방식보다 새로운 뷰 합성과 3D 인간 재구성 작업에서 현저하게 우수하다는 것을 나타냅니다.



### Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving (https://arxiv.org/abs/2408.14197)
Comments:
          18 pages, 10 figures

- **What's New**: 본 논문에서는 Drive-OccWorld라는 새로운 비전 중심의 4D 예측 및 계획 세계 모델을 소개합니다. 이 모델은 자율 주행을 위해 엔드투엔드 계획에 적응될 수 있도록 설계되었습니다.

- **Technical Details**: Drive-OccWorld는 세 가지 주요 구성 요소로 이루어져 있습니다: 역사 인코더(History Encoder), 메모리 큐(Memory Queue), 미래 디코더(Future Decoder) 입니다. 또한, 행동 조건(예: 속도, 조향 각도 등)을 주입하여 생성 과정에서의 유연성을 더욱 높입니다. 이를 통해 예측한 미래 상태와 최적 경로를 지속적으로 모델에 재투입하여 연속적인 미래 예측을 지원합니다.

- **Performance Highlights**: nuScenes 데이터셋에 대한 실험에서 Drive-OccWorld는 이전 방법들보다 예측 정확도가 2.0% (mIoU) 및 1.9% (VPQ) 향상된 성능을 보여주었으며, 안전한 경로 계획에서의 활용 가능성을 입증했습니다.



### Feature Aligning Few shot Learning Method Using Local Descriptors Weighted Rules (https://arxiv.org/abs/2408.14192)
- **What's New**: 본 논문은 제한된 레이블 샘플로 새로운 카테고리를 식별하는 few-shot classification 방법을 다루고 있습니다. 기존의 local descriptors기반의 방법들이 다양한 도전 과제를 직면하고 있는 가운데, 새로운 Feature Aligning Few-shot Learning Method Using Local Descriptors Weighted Rules (FAFD-LDWR)를 제안합니다. 이 방법은 Cross-normalization 기법을 도입하여 local descriptors의 차별화된 정보를 최대한 보존하며, 지원 세트와 쿼리 세트의 키 local descriptors를 정렬하여 배경 잡음을 제거하는 것을 목표로 합니다.

- **Technical Details**: FAFD-LDWRM 방법은 세 가지 주요 구성 요소로 이루어져 있습니다: embedding feature extraction module, cross normalization module, local descriptors with dynamically weighted rules module. 첫 번째 모듈은 contextual learning 메커니즘에 기반한 embedding 네트워크를 사용하여 지원 및 쿼리 세트 이미지에서 특징을 추출합니다. 두 번째 모듈은 adaptive parameters를 사용하여 local descriptors의 공간적 및 채널 차원을 정규화해 최대한의 차별화 정보를 유지합니다. 마지막으로, 세 번째 모듈은 각 local descriptor의 가중치를 계산하여 주요 descriptors를 필터링하고 배경 잡음을 제거하며 few-shot classification 성능을 향상시킵니다.

- **Performance Highlights**: FAFD-LDWR는 세 개의 벤치마크 데이터셋에서 우수한 성과를 보였으며, 1-shot 및 5-shot 설정 모두에서 최신 방법들을 초월하는 결과를 냈습니다. 또한, designed visualization 실험 결과를 통해 FAFD-LDWR가 예측 해석 가능성에서 개선을 보여준 점도 두드러집니다.



### EMDFNet: Efficient Multi-scale and Diverse Feature Network for Traffic Sign Detection (https://arxiv.org/abs/2408.14189)
Comments:
          15 pages,5 figures,accepted to ICANN

- **What's New**: 이 논문에서는 차량 신호 탐지를 위한 새로운 객체 탐지 네트워크인 EMDFNet을 제안합니다. 이 네트워크는 Augmented Shortcut Module과 Efficient Hybrid Encoder를 통합하여 단일 특성 추출 문제와 다양한 크기의 객체와의 통합 문제를 동시에 해결합니다.

- **Technical Details**: EMDFNet은 Res2Net을 백본으로 활용하여 세 개의 특징 맵을 추출하며, Augmented Shortcut Module(ASM)과 Efficient Hybrid Encoder(EHE)를 통해 다중 스케일(feature scale) 상호 작용 및 다양한 특징(features) 통합을 실현합니다. ASM은 다양한 공간 및 채널 의미 정보를 통합하여 특징 다양성을 향상시키며, EHE는 멀티 스케일 특징 융합을 촉진합니다. 또한 SIoU Loss를 도입하여 박스 손실을 계산하고, 훈련 속도와 탐지 정확도를 개선합니다.

- **Performance Highlights**: EMDFNet은 Tsinghua-Tencent 100K(TT100K) 및 German Traffic Sign Detection Benchmark(GTSDB) 데이터셋에서 폭넓은 실험을 통해 다른 최신 탐지기들보다 월등한 성능을 기록하며, 단일 스테이지 모델의 실시간 처리 능력을 유지합니다.



### Ensemble Predicate Decoding for Unbiased Scene Graph Generation (https://arxiv.org/abs/2408.14187)
- **What's New**: 본 논문에서는 Ensemble Predicate Decoding (EPD) 기법을 제안하여 Scene Graph Generation (SGG)에서 발생하는 편향(bias) 문제를 해결하고, 특히 저빈도(predicate) 예측의 정확성을 개선하고자 하였습니다.

- **Technical Details**: EPD는 메인 디코더와 두 개의 보조 디코더를 훈련시켜 서로 다른 빈도의 predicate에 대해 더 나은 구별 능력을 지니도록 설계되었습니다. 각 디코더는 특정한 훈련 데이터셋의 하위 집합에 맞춘 훈련을 받으며, 이는 long-tail 효과를 줄이는 데 기여합니다.

- **Performance Highlights**: Visual Genome (VG) 데이터셋에 대한 실험 결과 EPD 기법은 다양한 SGG 모델과 결합되어 mR@K의 성능이 크게 개선되었고, R@K는 거의 감소하지 않는 모습을 보였습니다. 본 방법은 기존의 최첨단 기법보다 뛰어난 성능을 기록했습니다.



### Affine steerers for structured keypoint description (https://arxiv.org/abs/2408.14186)
Comments:
          To be presented at ECCV 2024

- **What's New**: 본 논문에서는 딥 러닝 기반의 keypoint descriptors를 훈련하는 새로운 방법을 제안합니다. 이 방법은 이미지를 특정하게 왜곡하는 affine 변환에 대해 근사적으로 equivariant 하게 만들어 줍니다. 이를 위해 GL(2)의 표현 이론을 사용하여 최근 소개된 steerers 개념을 회전에서 affine 변환으로 일반화하는 접근 방식을 도입했습니다.

- **Technical Details**: 우리의 접근은 neural network 기반의 keypoint descriptors를 훈련하여 지역적인 affine 변환에 대해 근사적으로 equivariant 하도록 설정하는 것입니다. affine steerers를 사용하여 이러한 네트워크를 훈련하며, 제안된 방법에서는 예비 훈련(pretraining) 시 heavy homography augmentation을 적용한 후 upright 이미지에서 fine-tuning을 진행합니다. 또한, 이러한 방법을 통해 state-of-the-art 성능을 달성했습니다.

- **Performance Highlights**: 제안한 모델 AffSteer-B는 IMC22 benchmark에서 DeDoDe-B의 72.9 mAA@10에서 77.3 mAA@10으로 개선된 성과를 보였습니다. 우리는 또한 회전 변형 벤치마크 AIMS에서도 경쟁력 있는 결과를 도출했습니다. 이를 통해 우리의 keypoint descriptors가 다양한 기준에서 높은 성능을 발휘함을 보였습니다.



### I2EBench: A Comprehensive Benchmark for Instruction-based Image Editing (https://arxiv.org/abs/2408.14180)
Comments:
          Tech report, 39 pages, 41 figures

- **What's New**: 이 논문에서는 Instruction-based Image Editing (IIE) 분야에서의 모델 성능 평가를 위한 포괄적인 벤치마크인 I2EBench를 제안합니다. 이 벤치마크는 2000개 이상의 편집 이미지와 4000개 이상의 다양한 원본 지침을 포함하고 있으며, 고급 및 저급 평가 차원을 포함한 총 16개의 평가 차원을 제공합니다.

- **Technical Details**: I2EBench는 사용자의 인식에 맞춘 평가를 위해 대규모 사용자 연구를 수행하였으며, 다양한 편집 모델의 강점과 약점을 분석하여 연구 통찰을 제공합니다. 평가 차원은 고급 편집과 저급 편집으로 구분되며, 이미지의 세부 사항 편집과 특정 영역 편집을 포함합니다. 또한, I2EBench는 인간 평가자에 의한 점수 수집을 통해 높은 상관관계를 발견했습니다.

- **Performance Highlights**: I2EBench는 IIE 모델의 포괄적인 성능 평가를 지원하며, 기존 모델의 장단점을 체계적으로 평가하여 향후 개선 방향을 제시합니다. 이 자료는 오픈 소스로 제공되어 다른 연구자들이 벤치마크를 활용할 수 있도록 하며, 공정한 비교와 커뮤니티 발전을 촉진할 것으로 기대됩니다.



### NimbleD: Enhancing Self-supervised Monocular Depth Estimation with Pseudo-labels and Large-scale Video Pre-training (https://arxiv.org/abs/2408.14177)
- **What's New**: NimbleD는 대규모 비전 모델이 생성한 의사 레이블(pseudo-labels)로부터의 지도(supervision)를 통합한 효율적인 자기 지도 단안 깊이 추정(self-supervised monocular depth estimation) 학습 프레임워크입니다. 이 프레임워크는 카메라 내부 파라미터(camera intrinsics)를 요구하지 않으며, 공공 비디오에서 대규모 사전 학습이 가능합니다.

- **Technical Details**: NimbleD는 자기 지도 학습(self-supervised learning)과 대규모 비디오 사전 학습을 통해 효율성을 극대화한 프레임워크로, 경량 모델의 깊이 추정 성능을 향상시킵니다. 또한, 간단한 손실 함수(loss function)를 도입하여 SSL과 PSL(pseudo-supervised learning) 손실을 결합합니다. 이로 인해, 많은 대규모 비디오 데이터를 복잡한 데이터 준비 없이 사용할 수 있습니다.

- **Performance Highlights**: NimbleD는 기존 최첨단(self-supervised) 단안 깊이 추정 모델과 비교해 손실 없이 해당 성능 수준에 도달할 수 있는 빠르고 가벼운 모델의 깊이 추정 성능을 크게 향상시킵니다. 이는 메타버스 애플리케이션에서 저지연 인퍼런스(low latency inference)를 요구하는 경우에 특히 유리합니다.



### SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher (https://arxiv.org/abs/2408.14176)
Comments:
          Accepted to ECCV'24

- **What's New**: 이 논문에서는 SwiftBrush, 유명한 일단계 텍스트-이미지 확산 모델의 성능을 향상시켜 다단계 Stable Diffusion 모델과 경쟁할 수 있도록 목표합니다. 본 연구는 SwiftBrush와 SD Turbo 간의 품질-다양성(trade-off)을 탐구하여, 두 모델의 장점 조합과 새로운 훈련 방법론인 clamped CLIP loss의 도입을 통해 하나의 모델이 다단계 모델을 초월할 수 있다는 것을 보여줍니다.

- **Technical Details**: 본 연구에서는 SwiftBrush와 SD Turbo를 비교하면서, SwiftBrush는 더 다양한 출력 이미지를 제공하는 반면 SD Turbo는 높은 품질의 이미지를 생성한다는 사실에 주목했습니다. 또한, LoRA의 효율적인 훈련이 결합된 새로운 훈련 방법론을 제안하여 학생 모델이 교사 모델을 초과하는 성능을 거두도록 했습니다. 극복된 품질 메트릭은 FID (Fréchet Inception Distance)를 포함입니다.

- **Performance Highlights**: 최종적으로, 저희는 FID 점수 8.77을 달성하며, 다단계 모델을 초과한 첫 번째 일단계 모델을 제안했습니다. 이후 추가적인 정규화로 FID 점수가 8.14로 향상되었고, 이는 효율적이고 고품질의 텍스트-이미지 모델 분야에서 새로운 기준을 설정했습니다.



### BackFlip: The Impact of Local and Global Data Augmentations on Artistic Image Aesthetic Assessmen (https://arxiv.org/abs/2408.14173)
Comments:
          Published at the VISART VII workshop at ECCV 2024. Ombretta Strafforello, Gonzalo Muradas Odriozola, Fatemeh Behrad, Li-Wei Chen, Anne-Sofie Maerten and Derya Soydaner contributed equally to this work

- **What's New**: 이번 논문에서는 예술 이미지의 미적 품질 평가(Artistic Image Aesthetic Assessment, IAA)를 위한 새로운 접근 방식인 BackFlip 기법을 소개합니다. BackFlip은 예술 이미지에 특화된 로컬(data augmentation) 기법으로, 기존의 데이터 증강 기술과 비교하여 더 나은 성능을 발휘합니다.

- **Technical Details**: BackFlip은 로컬 데이터 증강 기술로, 예술 이미지에서의 구성을 변형하지 않고 미적 평가에 기여합니다. 이 연구는 3개의 예술 이미지 데이터 세트와 4개의 신경망 아키텍처에서 BackFlip의 성능을 평가하며, 일반적으로 사용되는 데이터 증강 기법과 비교합니다. 또한, BackFlip 파이프라인의 구성 요소에 대한 분석을 위한 ablation study를 수행합니다.

- **Performance Highlights**: 연구 결과, BackFlip과 같은 로컬 증강 기법이 미적 이미지 평가에서 글로벌 증강 기법보다 더 나은 성능을 보이는 경우가 많다는 사실이 밝혀졌습니다. 이는 로컬 증강이 예술 이미지의 구성을 훼손하지 않기 때문으로 분석됩니다. 본 연구는 향후 계산 미학(computational aesthetics) 연구에 있어 로컬 및 글로벌 증강 기법을 모두 고려할 필요성을 강조합니다.



### Explaining Vision-Language Similarities in Dual Encoders with Feature-Pair Attributions (https://arxiv.org/abs/2408.14153)
- **What's New**: 이번 연구에서는 CLIP 모델과 같은 Dual encoder 아키텍처의 예측을 입력 간의 feature-pair 상호작용으로 귀속할 수 있는 방법을 제안하였습니다. 이를 통해 모델이 입력된 두 데이터를 비교하는 방식을 보다 깊이 이해할 수 있습니다.

- **Technical Details**: 제안된 방법은 어떠한 미분 가능 Dual encoder 모델에 대해서도 입력 간의 상호작용을 설명할 수 있는 일반적인 feature-pair 귀속값을 계산하게 합니다. 특히, 이 방법은 훈련된 모델의 수정 없이 적용 가능하며, 시각-언어 모델에 적용되는 과정에서 세밀한 상호작용을 포착할 수 있습니다.

- **Performance Highlights**: 모델은 주어진 데이터 배치 내에서 세부적인 객체 클래스 간의 지식 격차를 식별하고, in-domain 훈련 후 성능 향상을 모니터링할 수 있는 능력을 가지고 있어, CLIP 모델의 시각-언어 기초 능력이 객체 클래스에 따라 다양하게 나타남을 보여줍니다.



### Application of Disentanglement to Map Registration Problem (https://arxiv.org/abs/2408.14152)
- **What's New**: 이 논문에서는 다양한 출처(위성, 항공기, LiDAR 등)에서 수집된 지리공간 데이터의 일치를 실현하기 위한 새로운 방법을 제안합니다. 이를 통해 서로 다른 '스타일'을 가지는 이미지를 동일한 지구의 위치에 맞춰 정렬하는 과정을 두 단계로 나누어 접근합니다.

- **Technical Details**: 제안된 방법은 (1) 시각적 정보와 비관련된 정보를 배제하고 지리공간 내용을 추출하며, (2) 이러한 지리공간 내용을 기반으로 데이터를 일치시키는 2단계 프로세스를 포함합니다. β-VAE 유사 아키텍처와 적대적 훈련의 조합을 통해 지리 정보와 예술적 스타일의 분리를 시도합니다.

- **Performance Highlights**: 두 가지 이미지 컬렉션을 사용한 실험에서, 동일한 지리정보를 가지고 있으면서도 서로 다른 스타일을 가지는 이미지를 성공적으로 분리할 수 있음을 보여주었습니다. 이 연구는 역사적인 지도와 현대 지도를 정렬하게 하여 지역의 역사를 보다 풍부하게 이해할 수 있도록 도와줍니다.



### 2D-Malafide: Adversarial Attacks Against Face Deepfake Detection Systems (https://arxiv.org/abs/2408.14143)
Comments:
          Accepted at BIOSIG 2024

- **What's New**: 본 논문에서는 얼굴 딥페이크 탐지 시스템을 속이기 위해 설계된 경량의 새로운 적대적 공격 기법인 2D-Malafide를 소개합니다. 이 방법은 음성 영역에서 탐구된 1D convolutional perturbations 개념을 기반으로 하며, 2D convolutional filters를 활용하여 탐지 성능을 크게 저하시킵니다.

- **Technical Details**: 2D-Malafide는 특정 입력 이미지에 독립적으로 적대적 perturbations를 생성하기 위해, 몇 개의 필터 계수를 최적화합니다. 이 공격은 이미지의 특정 피처를 조작하고, Filter의 크기에 따라 효과가 달라지며, 흰색 상자 및 검은색 상자 설정에서 모두 유효합니다.

- **Performance Highlights**: FaceForensics++ 데이터셋을 사용한 실험에서, 2D-Malafide는 최신 얼굴 딥페이크 탐지기의 성능을 현저하게 저하시킨 것으로 나타났습니다. GradCAM을 사용한 설명 분석을 통해, 이 공격이 탐지 시스템을 어떻게 오도하는지를 보여줍니다.



### Foodfusion: A Novel Approach for Food Image Composition via Diffusion Models (https://arxiv.org/abs/2408.14135)
Comments:
          14 pages

- **What's New**: 이 논문에서는 22,000 쌍의 고품질 음식 이미지 데이터셋 FC22k를 소개하며, 이를 바탕으로 새로운 음식 이미지 합성 방법, Foodfusion을 제안합니다.

- **Technical Details**: Foodfusion은 전이 학습된 diffusion 모델을 사용하고, Fusion Module 및 Content-Structure Control Module을 통해 전경 및 배경 정보를 통합합니다. Fusion Module은 다중 스케일 및 공간 인식을 통해 전경과 배경 이미지를 통합된 임베딩 공간으로 인코딩합니다. 이 과정에서 cross-attention layer가 역할을 하며, Content-Structure Control Module은 픽셀 수준에서의 콘텐츠 일관성을 유지합니다.

- **Performance Highlights**: FC22k 데이터셋에서 수행한 실험 결과, Foodfusion 방법은 효과적이며 확장 가능하다는 것을 입증했습니다. 이 연구는 음식 이미지 합성 작업에 대한 새로운 기준을 설정합니다.



### GenFormer -- Generated Images are All You Need to Improve Robustness of Transformers on Small Datasets (https://arxiv.org/abs/2408.14131)
Comments:
          This paper has been accepted at International Conference on Pattern Recognition (ICPR), 2023

- **What's New**: 최근 연구에서 Vision Transformers (ViTs)와 Convolutional Neural Networks (CNNs) 간의 경쟁력 있는 정확성을 보여주고, ViTs의 뛰어난 견고성을 강조합니다. 하지만, ViTs는 충분한 성능을 얻기 위해 대량의 데이터가 필요하여 소규모 데이터셋에의 적용이 어렵습니다. 이를 극복하기 위해 생성된 이미지를 활용한 새로운 데이터 증강 전략인 GenFormer를 제안합니다.

- **Technical Details**: GenFormer는 작은 데이터셋에서 ViTs의 적용성을 높이기 위한 생성적 데이터 증강 전략입니다. 데이터 부족 문제를 직접 해결하며, 실제 데이터셋에 생성 모델에서 나온 이미지를 추가하여 정보량을 늘립니다. 우리는 Tiny ImageNetV2, Tiny ImageNet-R, Tiny ImageNet-A를 새로운 테스트 세트로 제안하고, 기존의 CNN과 ViTs 간의 정확성과 견고성을 비교하여 GenFormer의 효과를 입증하였습니다.

- **Performance Highlights**: 실험을 통해, Tiny ImageNet, CIFAR, EuroSAT, MedMNIST와 같은 다양한 소규모 데이터셋에서 ViTs의 성능과 견고성을 개선하는 것을 보여주었습니다. GenFormer는 전통적인 훈련 방식을 사용하는 경우에도 우수한 결과를 제공하며, CNN과 ViTs 간의 격차를 줄이는 데 긍정적인 역할을 합니다.



### ShapeMamba-EM: Fine-Tuning Foundation Model with Local Shape Descriptors and Mamba Blocks for 3D EM Image Segmentation (https://arxiv.org/abs/2408.14114)
- **What's New**: 이 논문은 전자현미경(EM) 이미지 분할을 위한 ShapeMamba-EM이라는 특화된 파인튜닝 방법을 제안합니다. 이 방법은 기존의 foundation model에 어댑터(adapter)를 도입하여 장거리 의존성(long-range dependency)을 모델링하고, EM 데이터의 독특한 볼륨적 및 형태적 복잡성을 효과적으로 다루기 위해 로컬 형태 서술자(Local Shape Descriptors, LSD)를 사용하는 구조를 포함하고 있습니다.

- **Technical Details**: ShapeMamba-EM은 3D EM 분할 작업을 위한 SAM-Med3D 모델을 기반으로 하며, FacT라는 최신의 파라미터 효율적인 전이 학습 기법을 사용하여 이미지 인코더(image encoder)를 조정합니다. 여기서 3D Mamba Adapters를 통해 객체 분석에서의 장거리 의존성 문제를 해결하고, 3D U-Net 아키텍처를 사용하여 세분화 객체의 로컬 형태 서술자를 예측합니다. 이 두 가지 디자인은 EM 분할에서 SAM-Med3D의 한계를 극복하는 데 기여합니다.

- **Performance Highlights**: ShapeMamba-EM은 10개의 데이터셋을 포함한 5가지 분할 작업에서 광범위한 EM 이미지에 대해 테스트되었으며, 기존의 방법들을 능가하는 성능을 보여주어 EM 이미지 분할 분야에서 새로운 기준을 설정하였습니다.



### Bengali Sign Language Recognition through Hand Pose Estimation using Multi-Branch Spatial-Temporal Attention Mod (https://arxiv.org/abs/2408.14111)
- **What's New**: 본 연구는 BSL(Bangladesh Sign Language) 인식 문제의 해결을 위한 새로운 접근법을 제안합니다. 공간-시간적 주의(attention) 기반의 BSL 인식 모델과 새로운 데이터셋인 BAUST-BSL-38을 소개하여 BSL 데이터 부족 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 모델은 이미지 시퀀스에서 추출한 손 관절 스켈레톤을 기반으로 하여, Separable TCN과 다중 헤드 공간-시간적 주의 구조를 활용하여 최적의 특징을 추출합니다. 이 모델은 기존 아키텍처에 비해 세 배 낮은 계산 복잡도를 유지하면서도 성능을 최적화합니다.

- **Performance Highlights**: 테스트 결과, 제안된 모델은 다양한 데이터셋과 평가 세팅에서 뛰어난 성능을 입증했습니다. 특히, 기존의 모델과 비교하여 계산 복잡성을 획기적으로 줄이면서도 경쟁력 있는 인식률을 보여주었습니다.



### LSM-YOLO: A Compact and Effective ROI Detector for Medical Detection (https://arxiv.org/abs/2408.14087)
- **What's New**: 이 연구에서는 Medical Region of Interest (ROI) 탐지 알고리즘의 한계점을 극복하고, 실시간 성능과 정확성을 동시에 만족시키는 경량 모델인 LSM-YOLO(Lightweight Shunt Matching-YOLO)를 제안하였다. 이는 Lightweight Adaptive Extraction (LAE)과 Multipath Shunt Feature Matching (MSFM) 구조를 이용하여 의료 영상의 ROI 탐지를 개선하는 데 중점을 두었다.

- **Technical Details**: LSM-YOLO는 LAE를 활용하여 다중 스케일의 피처 맵에서 높은 해상도의 세부 정보를 추출하며, MSFM을 통해 고수준 시맨틱 피처와 저수준 비주얼 피처 간의 융합을 개선한다. 모델은 RFAConv(Receptive-Field Attention Convolution) 연산을 통해 주의 메커니즘을 도입하여 ROI와 주변 영역 간의 관계를 효과적으로 처리한다.

- **Performance Highlights**: LSM-YOLO는 췌장 종양의 비공식 데이터를 대상으로 48.6%의 평균 정확도(AP)를 기록하였으며, BCCD 혈액 세포 탐지 공공 데이터셋에서 65.1%, Br35h 뇌 종양 탐지 공공 데이터셋에서 73.0%의 AP를 달성하였다. 이러한 성능은 최소한의 파라미터 비용으로 이루어졌다.



### HABD: a houma alliance book ancient handwritten character recognition databas (https://arxiv.org/abs/2408.14084)
Comments:
          8 pages, 5 figures

- **What's New**: 본 연구에서는 Houma Alliance Book의 고대 손글씨 문자 인식 문제를 해결하기 위한 새로운 데이터베이스와 심층 학습 아키텍처 기반의 벤치마크를 제안합니다.

- **Technical Details**: 이 연구는 Houma Alliance Book에서 26,732개의 문자 샘플을 수집하여 327종의 고대 문자를 포함하는 새로운 문서 데이터베이스를 구축하였습니다. 데이터의 불균형을 해결하기 위해 Mixup과 같은 장기 꼬리(long-tail) 기술을 활용하고, 4개의 심층 신경망(Classifier) 모델의 결정 레벨 분류기 융합(classifier fusion)을 통해 인식 정확성을 향상시켰습니다.

- **Performance Highlights**: 이 연구는 고대 문자의 인식 정확성을 높이는 데 중점을 두며, Houma Alliance Book과 같은 고대 문자에 대한 연구에 귀중한 자료와 기술적 지원을 제공합니다. 이러한 기여는 고대 문화 및 역사에 대한 이해를 증진시키고 인류의 문화유산 보존에 기여합니다.



### Evaluating the Visual Similarity of Southwest China's Ethnic Minority Brocade Based on Deep Learning (https://arxiv.org/abs/2408.14060)
Comments:
          8 pages,2tables,5 figures

- **What's New**: 이 논문은 중국 남서부의 소수 민족 패턴의 시각적 유사성을 조사하기 위해 딥러닝(Deep Learning) 방법을 사용합니다.

- **Technical Details**: 커스터마이즈된 SResNet-18 네트워크를 개발하였으며, 테스트 세트에서 98.7%의 정확도(Accuracy)를 달성하였습니다. 이는 ResNet-18, VGGNet-16, AlexNet보다 우수한 성능입니다. SResNet-18에서 추출된 피쳐 벡터(Feature Vector)는 코사인 유사성(Cosine Similarity), 유클리드 거리(Euclidean Distance), 맨하탄 거리(Manhattan Distance)의 세 가지 지표로 평가되었습니다.

- **Performance Highlights**: 결과는 민족 주제 맵(Ethnic Thematic Map)으로 시각적으로 표현되어, 민족 패턴과 지역 분포 간의 연관성을 강조하였습니다.



### Let Video Teaches You More: Video-to-Image Knowledge Distillation using DEtection TRansformer for Medical Video Lesion Detection (https://arxiv.org/abs/2408.14051)
Comments:
          BIBM2024

- **What's New**: 본 연구는 의학 비디오의 병변 탐지 작업을 위한 Video-to-Image 지식 증류 프레임워크, V2I-DETR을 제안합니다. 이 방식은 속도와 정확성을 모두 아우르며, 이전의 최첨단 방법들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: V2I-DETR은 Teacher-Student 네트워크 구조를 채택하여, 교사 네트워크가 여러 프레임에서 시간적 컨텍스트(temporal contexts)를 추출하고 이를 학생 네트워크에 전달합니다. 이 과정은 Multi-scale Spatiotemporal Interaction (MSI) 모듈을 통해 이루어지며, Target-guided Feature Distillation (TFD)와 Cross-view Query Distillation (CQD) 기법을 사용하여 노이즈를 줄이고 처리가능성을 높입니다.

- **Performance Highlights**: V2I-DETR은 SUN-SEG 및 BUV 데이터셋에서 평가되었으며, 이전의 이미지 기반 및 비디오 기반 방법들보다 월등한 성능을 기록하였습니다. 특히, 실시간 추론 속도인 30 FPS를 달성하였습니다.



### Alleviating Class Imbalance in Semi-supervised Multi-organ Segmentation via Balanced Subclass Regularization (https://arxiv.org/abs/2408.14047)
- **What's New**: 이번 연구에서는 다중 장기 분할(multi-organ segmentation, MoS) 작업에서 클래스 불균형 문제를 해결하기 위해 BSR-Net이라는 새로운 반지도 학습(semi-supervised learning, SSL) 네트워크를 제안합니다.

- **Technical Details**: BSR-Net은 두 가지 단계로 구성됩니다. 1단계에서는 균형 클러스터링(balanced clustering)을 통해 원래의 편향된 데이터에서 균형 잡힌 서브 클래스(subclass)를 생성합니다. 2단계에서는 주 MoS 작업의 다중 작업 프레임워크(multi-task framework) 내에서 보조 서브 클래스 분할(segmentation) 작업을 설계하여 불균형 지식을 MoS 네트워크에 전달합니다.

- **Performance Highlights**: MICCAI FLARE 2022 데이터셋과 WORD 데이터셋을 사용한 실험 결과, BSR-Net은 기존의 다른 방법들에 비해 우수한 성능을 나타냈습니다.



### More Pictures Say More: Visual Intersection Network for Open Set Object Detection (https://arxiv.org/abs/2408.14032)
Comments:
          7pages

- **What's New**: 최근 오픈셋 객체 탐지(Open Set Object Detection) 분야에서 VE-O를 기반으로 한 새로운 모델 VINO(Visual Intersection Network)를 개발하였습니다. 이 모델은 다중 이미지 비주얼 뱅크를 활용하여 각 카테고리의 의미적 교차점을 지속적으로 유지하며, 텍스트 기반 방법과 비교해 사전 학습 시간과 리소스를 상당히 감소시킵니다.

- **Technical Details**: VINO는 다양한 비주얼 프롬프트를 통해 의미적 교차점을 학습하는 혁신적인 다중 이미지 비주얼 업데이트 메커니즘을 도입하였습니다. 이 결합된 접근 방식은 모델이 여러 시각적 입력을 동시에 처리할 수 있도록 하여 카테고리별 특징을 더욱 정확하고 강력하게 이해하고 표현할 수 있게 합니다.

- **Performance Highlights**: VINO는 Objects365v1 데이터셋에서 7일의 RTX4090 GPU만으로도 훈련 가능하며, LVIS 및 ODinW35와 같은 벤치마크에서 비전-언어 모델에 필적하는 경쟁력 있는 성능을 달성하였습니다. VINO는 APb 점수로 Obj365 v1에서 38.1, LVIS v1 검증 세트에서 29.2, ODinW-35 검증 세트에서 24.5를 기록하였으며, GLiP보다 2.3 점( LVIS v1 )과 1.1 점( ODinW-35 ) 높은 성과를 보였습니다.



### SurGen: Text-Guided Diffusion Model for Surgical Video Generation (https://arxiv.org/abs/2408.14028)
- **What's New**: 본 연구에서는 SurGen이라는 텍스트 기반 확산 모델을 소개합니다. 이는 기존의 외과 비디오 생성 모델들 중 가장 높은 해상도(720 x 480 pixels)와 긴 지속시간(49 frames)을 자랑합니다. 이 모델은 외과 절차의 다양한 단계를 지정하는 텍스트 프롬프트에 의해 조건화되어 비디오를 생성합니다.

- **Technical Details**: SurGen은 CogVideoX를 기반으로 하여 텍스트 프롬프트를 사용하여 외과 영상을 생성합니다. 모델 학습에는 200,000개의 독특한 비디오-텍스트 쌍이 사용되며, 각 비디오는 특정 외과 단계에 맞는 프롬프트와 쌍을 이룹니다. 각 비디오는 3D Variational Autoencoder(3D VAE)와 Denoising Video Transformer를 통해 생성됩니다. 또한 2억 개의 파라미터를 가진 텍스트 조건형 비디오 변환기를 사용하여 비디오의 공간 및 시간 정보를 처리합니다.

- **Performance Highlights**: SurGen 모델은 Fréchet Inception Distance(FID) 지표에서 79.9163을 기록하여, 사전 훈련된 모델의 260.0301보다 현저하게 개선된 시각적 품질과 다양성을 나타냅니다. Fréchet Video Distance(FVD)에서도 752의 개선된 성능을 보였습니다.



### Video-CCAM: Enhancing Video-Language Understanding with Causal Cross-Attention Masks for Short and Long Videos (https://arxiv.org/abs/2408.14023)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 연구에서는 비디오-MLLM(Video-MLLM)이라고 불리는 새로운 다중 모달 대형 언어 모델(MLLM)인 Video-CCAM을 제안합니다. Video-CCAM은 비디오와 언어 이해에서 뛰어난 성능을 보여주며, 비디오의 프레임 수에 관계없이 처리할 수 있는 유연한 모델입니다.

- **Technical Details**: 비디오-CCAM은 비주얼 인코더(visual encoder), LLM, 그리고 크로스 어텐션(cross-attention) 메커니즘을 활용한 프로젝트로 구성되어 있습니다. 특히, CCAM(causal cross-attention masks)을 도입하여 비디오의 시간적 순서를 고려하여 모델의 비디오 이해 능력을 향상시킵니다. 이 모델은 차세대 비디오-MLLM으로, 이미지와 16 프레임 비디오만을 사용하여 훈련된 후에도 긴 비디오의 이해에 적응할 수 있는 뛰어난 능력을 가지고 있습니다.

- **Performance Highlights**: Video-CCAM은 표준 비디오 벤치마크인 MVBench 및 VideoChatGPT-QA에서 1위, 2위, 3위의 성과를 달성하였으며, 긴 비디오에 대한 벤치마크에서도 뛰어난 점수를 기록하였습니다. 특히, VideoVista와 MLVU에서는 각각 1위와 2위에 오르며 모든 오픈 소스 비디오-MLLM 중에서 가장 높은 성능을 보였습니다.



### Pixel-Aligned Multi-View Generation with Depth Guided Decoder (https://arxiv.org/abs/2408.14016)
- **What's New**: 이 논문은 이미지를 바탕으로 다중 뷰 생성의 픽셀 정렬 문제를 해결하기 위해 새로운 방법을 제안합니다. 기존의 다중 뷰 생성 모델에서는 VAE와 U-Net을 활용하였으나, 픽셀 정렬 문제로 인해 결과물이 만족스럽지 못했습니다. 본 연구에서는 depth-truncated epipolar attention을 도입하여 고해상도에서의 조합이 가능하도록 하였으며, 이를 통해 다중 뷰 이미지 간의 정렬을 개선하였습니다.

- **Technical Details**: 이 방법은 VAE (Variational Autoencoder) 디코더에 다중 뷰 이미지간의 attention layer를 포함합니다. 구체적으로, depth-truncated epipolar attention을 통해 모델이 공간적으로 인접한 영역에 집중할 수 있도록 하여 메모리를 효율적으로 사용할 수 있게 합니다. 불확실한 깊이 추정으로 인한 일반화를 높이기 위해 훈련 중 깊이 입력을 섞고, 추론 시에는 다중 뷰-3D 재구성 방법인 NeuS를 활용하여 거친 깊이를 얻습니다.

- **Performance Highlights**: 제안된 방법은 기존의 상태-of-the-art 다중 뷰 생성 기법들과 비교하였을 때 PSNR, SSIM, LPIPS 및 재구성된 3D 객체의 일치 수와 같은 수치적 기준에서 높은 성능을 보였습니다. 이로 인해 본 방법이 3D 재구성 작업의 하위 작업에서 효과적임을 입증하였습니다.



### A Multiscale Gradient Fusion Method for Edge Detection in Color Images Utilizing the CBM3D Filter (https://arxiv.org/abs/2408.14013)
Comments:
          1 figure, 2 tables

- **What's New**: 이 논문에서는 다중 스케일 그래디언트 융합(multiscale gradient fusion)과 협업 필터링(collaborative filtering)을 결합한 컬러 엣지 감지(color edge detection) 전략을 제안합니다.

- **Technical Details**: RGB 이미지가 수학적 연산을 통해 XYZ 색 공간(images) 이미지로 변환됩니다. 이후 색 블록 매칭 및 3D 필터(Color Block-Matching and 3D, CBM3D)를 사용하여 희소 표현(sparse representation)에서 노이즈를 제거합니다. 색상 이미지의 벡터 그래디언트(vector gradients)와 두 개의 스케일 매개변수(anisotropic Gaussian directional derivative)를 계산하여 새로운 엣지 강도 맵(edge strength map)을 생성합니다. 마지막으로 이미지 정규화(image normalization)와 비최대 억제(non-maximum suppression) 기술을 통해 엣지 특징을 강화하고, 더블 임계값(double threshold) 선택 및 새로운 형태학적 정제(morphological refinement) 방법을 통해 엣지 윤곽을 얻습니다.

- **Performance Highlights**: 실험 분석 결과, 제안된 방법은 색 소벨(Color Sobel), 색 캐니(Color Canny), SE, 색 AGDD보다 PR 곡선(PR curve), AUC, PSNR, MSE, FOM 지표에서 우수한 엣지 품질(edge quality)과 노이즈 강인성(noise robustness)을 보여줍니다.



### LMM-VQA: Advancing Video Quality Assessment with Large Multimodal Models (https://arxiv.org/abs/2408.14008)
- **What's New**: 본 논문에서는 최초의 대규모 다중 모달 비디오 품질 평가 모델(LMM-VQA)을 제안하며, 이는 비디오 품질 평가(VQA) 작업을 해결하기 위해 새로운 시공간 시각 모델링 전략을 도입합니다.

- **Technical Details**: LMM-VQA 모델은 질적 회귀 문제를 Q&A(질문과 응답) 작업으로 재구성하고, 질적 특징을 추출하기 위해 시공간 비전 인코더를 설계합니다. 모델은 비디오의 시공간 특성을 추출하고, 이러한 특징들을 대형 언어 모델(LLM)에 입력하여 품질 점수와 수준을 생성합니다.

- **Performance Highlights**: 다양한 VQA 벤치마크에서 LMM-VQA는 기존 방법보다 평균 5%의 일반화 능력 향상을 보이며, 비디오 이해 작업에서도 우수한 성능을 나타냅니다.



### Avatar Concept Slider: Manipulate Concepts In Your Human Avatar With Fine-grained Contro (https://arxiv.org/abs/2408.13995)
- **What's New**: 이번 논문에서는 사용자 요구에 정확히 맞춘 3D 인간 아바타 편집을 위한 새로운 방법인 Avatar Concept Slider (ACS)를 제안합니다. 이는 두 개의 개념 극단 사이에 특정한 중간 지점을 선택하여 세밀하게 개념을 조작할 수 있게 해줍니다.

- **Technical Details**: ACS는 세 가지 독창적인 설계를 포함합니다. 1) Linear Discriminant Analysis (LDA) 기반의 Concept Sliding Loss를 통해 개념 특정 축을 정밀하게 찾아냅니다. 2) Principal Component Analysis (PCA) 기반의 Attribute Preserving Loss로 아바타의 정체성을 편집 과정에서 보존합니다. 3) 개념 민감도에 따라 3D Gaussian Splatting (3DGS) 원시 선택 메커니즘을 도입해 효율성을 높입니다.

- **Performance Highlights**: ACS를 통해 사용자는 아바타의 개념 표현 정도를 정밀하게 조정할 수 있으며, 이는 아바타의 품질이나 정체성을 손상시키지 않고 수행됩니다. 실험 결과, ACS는 높은 효율성과 빠른 피드백을 제공함을 보여주었습니다.



### Automatic Medical Report Generation: Methods and Applications (https://arxiv.org/abs/2408.13988)
Comments:
          42 pages and 9 figures

- **What's New**: 이 논문은 2021년부터 2024년까지의 자동 의료 보고서 생성(AMRG) 방법을 포괄적으로 검토하여 기존 문제의 해결책을 제시하고 이를 다양한 영상 방식에 적용하며 공개된 데이터셋과 평가 메트릭스를 소개합니다.

- **Technical Details**: AMRG는 컴퓨터 비전(CV) 및 자연어 처리(NLP)를 활용하여 의료 이미지를 해석하고 인간과 유사한 설명적 보고서를 생성하는 연구 분야입니다. 모델은 이미지 특징을 추출하는 visual encoder와 텍스트를 생성하는 text decoder(RNN 또는 Transformer)를 포함합니다.

- **Performance Highlights**: AMRG의 최근 발전은 CNNs와 Transformers를 활용하여 높은 정확도의 병변 탐지 및 질병 분류를 가능하게 하였으며, 특히 딥러닝 기법이 많은 연구에서 성능 향상에 기여하고 있습니다.



### Dual-Path Adversarial Lifting for Domain Shift Correction in Online Test-time Adaptation (https://arxiv.org/abs/2408.13983)
- **What's New**: 본 연구는 Transformer 모델의 테스트 시간 적응(테스트 타임 어댑테이션, TTA) 방법으로, 도메인 변화나 잡음을 효과적으로 분리하기 위해 새로운 'Domain Shift Token'을 도입하고, 이와 관련된 듀얼 경로 토큰 리프팅 기법을 제안합니다.

- **Technical Details**: 이 방법에서는 Transformer 네트워크의 각 레이어에 추가적인 Token(토큰)을 도입하고, 도메인 이동 토큰과 클래스 토큰 간의 상호 교환된 예측과 업데이트를 수행하여, 모델이 다양한 도메인에서 일관된 성능을 유지할 수 있도록 합니다. 예측 네트워크는 도메인 이동의 잔여 잡음을 학습하고, 업데이트 네트워크는 클래스 토큰을 수정하여 클래스 간의 구별력을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 온라인 테스트 시간 도메인 적응 성능이 크게 향상되었으며, 기존의 최첨단 방법들보다도 큰 폭으로 성능 향상을 보였습니다.



### ARANet: Attention-based Residual Adversarial Network with Deep Supervision for Radiotherapy Dose Prediction of Cervical Cancer (https://arxiv.org/abs/2408.13981)
Comments:
          Accepted by 2024 IEEE International Conference on Cybernetics and Intelligent Systems (CIS) and IEEE Conference on Robotics, Automation and Mechatronics (RAM)

- **What's New**: 논문에서는 자궁경부암 치료를 위한 방사선 치료의 3D 용적 배포 예측을 자동화하기 위한 새로운 방법인 Attention 기반 Residual Adversarial Network(ARANet)를 제안합니다.

- **Technical Details**: ARANet는 컴퓨터 단층촬영(CT) 이미지와 그에 해당하는 PTV(Planning Target Volume) 및 OARs(Organs-At-Risk)의 세분화 마스크를 바탕으로 방사선 분포를 예측하는 네트워크로 구성됩니다. 이 네트워크는 다중 스케일(rescale) residual attention 모듈과 딥 슈퍼비전(deep supervision) 메커니즘을 활용하여 예측 성능을 향상시킵니다.

- **Performance Highlights**: 54명의 자궁경부암 환자를 포함한 데이터셋에서 검증 결과, ARANet의 실험 결과는 기존의 최신 방법들에 비해 분명한 우수성을 보였습니다.



### FusionSAM: Latent Space driven Segment Anything Model for Multimodal Fusion and Segmentation (https://arxiv.org/abs/2408.13980)
- **What's New**: 본 논문에서는 Segment Anything Model (SAM)을 멀티모달 이미지 세분화에 처음으로 적용하여, Latent Space Token Generation (LSTG)과 Fusion Mask Prompting (FMP) 모듈을 결합한 새로운 프레임워크인 FusionSAM을 제안합니다.

- **Technical Details**: FusionSAM은 벡터 양자화를 통해 두 가지 모드에서 잠재 공간(feature) 정보를 캡처하고, 이를 크로스 어텐션 기반의 도메인 간 융합 모듈에 주입하여 모드 간의 장거리 의존성을 수립합니다. 이 융합 정보는 정확한 픽셀 단위의 세분화를 위한 프롬프트로 활용됩니다.

- **Performance Highlights**: 여러 공개 데이터셋에서 수행된 실험에 따르면 FusionSAM은 멀티모달 자율주행 시나리오에서 SAM 및 SAM2에 비해 최소 3.9% 높은 세분화 mIoU를 달성하여 기계적 성능이 현저하게 개선되었음을 확인할 수 있었습니다.



### Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models (https://arxiv.org/abs/2408.13979)
Comments:
          Accepted at ICLR 2024 (Spotlight)

- **What's New**: 본 연구는 VLMs(비전-언어 모델)에서 learnable soft-prompt vector의 norm(노름)의 영향을 체계적으로 조사합니다. 특히, Low-Norm Effect를 발견하였으며, 이는 특정 학습된 soft prompt의 norm을 줄이는 것이 성능을 향상시키고, 증가시키는 것이 성능 감소를 초래할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 Nemesis 방법은 VLMs의 soft-prompt 벡터를 정규화하기 위해 Position-Uniform Normalization (PUN) loss를 사용하고, Position-Aware Normalization (PAN) loss를 도입하여 low-norm 효과를 고려한 정규화를 수행합니다. 이 방법은 기존의 soft-prompt 기법에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: Nemesis 방법은 다양한 작업에서 soft-prompt 기반 VLM의 성능을 향상시키는 데 도움이 될 수 있으며, 실험을 통해 이 방법의 효과를 입증하였습니다.



### DynaSurfGS: Dynamic Surface Reconstruction with Planar-based Gaussian Splatting (https://arxiv.org/abs/2408.13972)
Comments:
          homepage: this https URL, code: this https URL

- **What's New**: 본 논문에서는 DynaSurfGS 프레임워크를 제안하여 현실적이고 고품질의 동적 장면 렌더링과 함께 높은 정밀도의 표면 재구성을 달성하고자 합니다. 이 프레임워크는 4D 신경 부피에서 가우시안 특징을 포함하여 정밀한 표면 재구성을 촉진합니다.

- **Technical Details**: DynaSurfGS 프레임워크는 가우시안 점의 모션과 형태 변화를 인코딩하기 위해 4D 복셀(voxel)을 2D 평면으로 분해하며, Compact MLP를 사용하여 다양한 시간 단계에서의 가우시안 변형을 예측합니다. 또한, ARAP(as-rigid-as-possible) 정규화 기법을 적용하여 동적 객체의 모션 중 형태의 일관성을 보장합니다.

- **Performance Highlights**: 광범위한 실험 결과 DynaSurfGS는 고충실도 표면 재구성과 현실적인 렌더링 품질 모두에서 기존의 최첨단 방법들을 능가함을 입증하였습니다.



### Shifted Window Fourier Transform And Retention For Image Captioning (https://arxiv.org/abs/2408.13963)
Comments:
          Pre-print version of paper accepted for ICONIP 2024

- **What's New**: 이번 연구에서는 이미지 캡셔닝(Imagem Captioning) 분야에서의 효율성을 개선하기 위해 거의 전적으로 Fourier Transform(w)의 활용과 Retention을 기반으로 한 새로운 아키텍처인 SwiFTeR를 소개합니다. 현재 경량 모델의 주요 문제점인 시각적 백본(visual backbone)의 비용과 디코더(decoder)의 제곱 비용을 해결하고자 합니다.

- **Technical Details**: SwiFTeR는 20M 파라미터(parameter)와 3.1 GFLOPs(Giga Floating Point Operations)로 구성되어 있으며, NVIDIA GeForce RTX 4090을 사용할 경우 초당 최대 400개의 캡션을 생성할 수 있습니다. 이 아키텍처는 기존 Transformer 기반 모델들과 비교해 메모리 요구사항이 낮아 여러 이미지를 병렬 처리할 수 있는 장점을 지닙니다.

- **Performance Highlights**: 현재 SwiFTeR는 110.2 CIDEr-D 점수를 기록하고 있으나, 이 점수의 감소는 아키텍처의 문제가 아니라 불완전한 학습으로 인해 발생한 것으로, 향후 개선의 여지가 많이 남아있습니다. 이러한 결과는 효율적인 새로운 아키텍처 디자인의 가능성을 보여줍니다.



### InterTrack: Tracking Human Object Interaction without Object Templates (https://arxiv.org/abs/2408.13953)
Comments:
          17 pages, 13 figures and 6 tables. Project page: this https URL

- **What's New**: InterTrack은 객체 템플릿 없이 모노큘러 RGB 비디오에서 전체 신체의 동적 물체 상호작용을 추적하는 최초의 방법입니다.

- **Technical Details**: 이 방법은 4D 비디오 재구성 문제를 프레임별 포즈 추정(per-frame pose estimation)과 전역 형상 최적화(global shape optimization)로 분해합니다. CorrAE라는 새로운 자동 인코더(autoencoder)는 SMPL(Skinned Multi-Person Linear) 정점을 직접 예측하여 시간적으로 일관된 상관 관계를 제공합니다. TOPNet은 모노큘러 RGB 비디오에서 물체 회전을 추정하여 시간적으로 일관된 객체 포즈를 예측합니다.

- **Performance Highlights**: BEHAVE와 InterCap 데이터셋에서 실험한 결과, InterTrack은 이전의 템플릿 기반 비디오 추적 방법인 VisTracker보다 유의미하게 뛰어난 성능을 보였으며, TOPNet은 이전의 포즈 추정기보다 월등히 우수한 성능을 기록했습니다. ProciGen-Video 데이터셋에서 학습한 모델은 실제 비디오에 대한 일반화 능력이 뛰어납니다.



### OpenNav: Efficient Open Vocabulary 3D Object Detection for Smart Wheelchair Navigation (https://arxiv.org/abs/2408.13936)
Comments:
          ECCVW

- **What's New**: 이 논문은 스마트 휠체어에 기반한 OpenNav라는 새로운 3D 객체 탐지 파이프라인을 제시합니다. OpenNav는 RGB-D 이미지를 기반으로 하여 오픈 어휘(open vocabulary) 탐지를 사용함으로써 다양한 환경에 적응할 수 있는 정밀하고 확장 가능한 객체 인식을 가능하게 합니다. 이 시스템은 실시간으로 새로운 객체를 인식할 수 있는 기능을 갖추었습니다.

- **Technical Details**: OpenNav는 RGB-D 카메라에서 수집한 RGB 및 깊이(depth) 데이터를 처리합니다. 이 과정에서 오픈 어휘 2D 객체 탐지기와 마스크 생성기를 사용하는데, 이를 통해 객체의 2D 위치를 식별하고, 심층 분리(depth isolation)와 포인트 클라우드(point cloud) 생성을 통해 3D 바운딩 박스를 생성합니다. 이 시스템은 기존의 지도 기반 방법이나 목표 기반 탐지 방식과 차별화되어, 보다 유연한 환경 인식을 가능하게 합니다. 또한, OpenNav는 기존의 훈련 데이터에 의존하지 않고 언어적 이해를 바탕으로 새로운 객체를 인식할 수 있는 제로샷(zero-shot) 학습을 활용합니다.

- **Performance Highlights**: OpenNav는 Replica 데이터셋에서 mAP25(+9pts) 및 mAP50(+5pts)에서 최첨단 성능을 보이며, 실제 스마트 휠체어와 함께 테스트해 본 결과, 3D 객체 탐지 및 정확한 목표 식별에서 높은 효율성을 입증하였습니다.



### GeoPlant: Spatial Plant Species Prediction Datas (https://arxiv.org/abs/2408.13928)
- **What's New**: 본 논문은 고해상도 공간에서 유럽 전역의 식물 분포를 모델링하기 위한 새로운 데이터 세트인 GeoPlant를 소개합니다. 이 데이터 세트는 10,000종 이상의 식물과 500만 개의 이질적인 Presence-Only 레코드를 포함하여, 생물 다양성 연구와 보전 노력에 기여할 수 있는 중요한 자원으로 자리잡을 것입니다.

- **Technical Details**: GeoPlant 데이터 세트는 10~50 m의 고해상도의 공간 해상도를 가지고 있습니다. 이 데이터 세트는 다양한 환경 레스터(예: 고도, 인간의 발자국), 20년의 기후 변수 시계열, Sentinel-2 RGB 및 NIR 위성 이미지와 Landsat 프로그램의 위성 시계열을 포함합니다. 이러한 데이터를 통해 생물 다양성을 효과적으로 모델링할 수 있으며, standardized한 Presence-Absence 조사 데이터도 포함되어 있습니다.

- **Performance Highlights**: GeoPlant는 2023년부터 진행된 GeoLifeCLEF SDM 평가 캠페인에 통합되어, 다양한 단일 및 다중 모달 접근 방식을 위한 강력한 기준선을 설정하고 있습니다. Kaggle에서 공개 접근 가능하며, 커뮤니티의 많은 참여를 이끌어내며 새로운 생물 분포 모델링 접근 방식의 벤치마크를 제공합니다.



### Infrared Domain Adaptation with Zero-Shot Quantization (https://arxiv.org/abs/2408.13925)
Comments:
          ICMV 2024

- **What's New**: 이 논문에서는 훈련 데이터에 접근할 수 없는 상황에서 사물 탐지 모델에 대한 제로샷 양자화(zero-shot quantization)를 적용하는 방법을 제시합니다. 특히, 열 이미지를 활용한 제로샷 양자화의 중요성과 효과를 강조하며, 의료나 보안 분야에서의 적합성에 대해 논의합니다.

- **Technical Details**: 본 연구에서는 사전 훈련된 모델과 통계 정보를 기반으로 양자화를 수행하며, 모델의 배치 정규화(batch normalization) 통계를 활용하여 데이터 증류(data distillation)를 구현합니다. 논문에서는 제로샷 양자화에서 평균 및 표준 편차(statistics) 통계의 기여를 조사하고, 열 데이터셋을 통한 포스트 훈련 양자화와 비교합니다.

- **Performance Highlights**: 실험 결과, 제로샷 양자화가 사물 탐지 모델의 양자화 과정에서 훈련 데이터의 대표성을 성공적으로 생성함을 보여주었습니다. 이는 훈련 데이터가 없는 상황에서도 효과적으로 작동하며, 열 영역에서의 응용 가능성을 잘 나타냅니다.



### COMPOSE: Comprehensive Portrait Shadow Editing (https://arxiv.org/abs/2408.13922)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 논문에서는 포트레이트의 그림자 속성을 정밀하게 제어할 수 있는 새로운 그림자 편집 파이프라인인 COMPOSE를 소개합니다. 기존의 방법들은 환경 조명과 조화롭게 그림자를 조정하는 데 어려움을 겪어왔지만, COMPOSE는 이러한 한계를 극복합니다.

- **Technical Details**: COMPOSE는 그림자 속성을 조정하기 위해 조명 데이터를 조화롭게 분해하는 4단계 파이프라인으로 구성되어 있습니다: 조명 추정 및 편집, 조명 확산, 그림자 합성 및 그림자 편집. 이 과정에서 사용되는 새롭고 편집 가능한 가우시안 지배 조명 소스는 기존 환경 조명과 통합됩니다.

- **Performance Highlights**: COMPOSE는 밝기, 형태 및 위치와 같은 그림자의 특성을 정밀하게 조정할 수 있는 기능을 제공하며, OLAT 데이터셋에서 훈련한 모델을 통해 현실적인 그림자를 생성하고, 정량적 및 정성적 평가를 통해 견고한 성능을 입증하였습니다.



### Splatt3R: Zero-shot Gaussian Splatting from Uncalibarated Image Pairs (https://arxiv.org/abs/2408.13912)
Comments:
          Our project page can be found at: https://splatt3r.active.vision/

- **What's New**: 새로운 연구에서, 우리는 Splatt3R라는 포즈 프리(pose-free) 방식의 피드 포워드(Feed-Forward) 모델을 도입하며, 이는 캘리브레이션되지 않은 자연 이미지에서 3D 재구성과 새로운 뷰 합성을 가능하게 합니다. 이 방법은 카메라 파라미터나 깊이 정보 없이 3D Gaussian Splats를 예측할 수 있습니다.

- **Technical Details**: Splatt3R는 MASt3R라는 '기초(foundation)' 3D 기하 구조 재구성 방법을 기반으로 하여, 이를 완전한 3D 구조와 외형 재구성기로 확장합니다. 이 모델은 각 포인트에 대한 Gaussian 속성을 예측하여 Gaussian 기본 요소를 구성합니다. 학습 과정에서는 3D 포인트 클라우드의 기하 손실을 최적화한 후 새로운 뷰 합성 목표를 설정합니다.

- **Performance Highlights**: Splatt3R는 ScanNet++ 데이터셋에서 학습되었으며, 캘리브레이션되지 않은 자연 이미지에 대해 뛰어난 일반화를 보여줍니다. 이 모델은 512 x 512 해상도에서 초당 4 프레임으로 장면을 재구성하며, 결과로 생성된 스플랫(Splats)은 실시간으로 렌더링할 수 있습니다.



### LowCLIP: Adapting the CLIP Model Architecture for Low-Resource Languages in Multimodal Image Retrieval Task (https://arxiv.org/abs/2408.13909)
- **What's New**: 본 연구는 자원을 적게 사용하는 언어인 아제르바이잔어를 위한 이미지 검색을 위하여 다중 모달 비전-언어 모델 개발을 탐구합니다.

- **Technical Details**: CLIP 모델 아키텍처를 통합하고 기계 번역을 통한 합성 데이터 생성, 이미지 증강, 그리고 도메인 특화 데이터로 변형하기 위한 추가적인 훈련을 활용하여 모델을 개발하였습니다. 여러 이미지 인코더(ResNet50, EfficientNet0, Vision Transformer)와 다국어 BERT를 텍스트 인코더로 통합했습니다.

- **Performance Highlights**: EfficientNet0 모델은 Flickr30k에서 MAP 점수를 0.84에서 0.87로, ResNet50은 MSCOCO에서 0.70에서 0.80으로 향상시켜 시각-언어 검색 분야의 최신 성과를 달성했습니다.



### ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models (https://arxiv.org/abs/2408.13906)
Comments:
          First two authors contributed equally. Source code is available at this https URL

- **What's New**: 이번 연구에서는 다중 모달 대형 언어 모델(MLLMs)의 환각(hallucination) 문제를 해결하기 위해 ConVis라는 새로운 비훈련 대조적 디코딩(constrastive decoding) 방법을 제안합니다. ConVis는 텍스트-이미지(text-to-image, T2I) 생성 모델을 활용하여 환각된 캡션으로부터 주어진 이미지를 의미론적으로 재구성합니다.

- **Technical Details**: ConVis는 MLLM의 디코딩 과정에서 오리지널 이미지와 재구성된 이미지 간의 확률 분포를 비교함으로써 시각적 대조 신호(visual contrastive signals)를 포착하고, 환각 생성을 패널티(penalize)하여 이를 감소시킵니다. 주목할 점은 이 방법이 디코딩 과정 내에서만 작동한다는 점이며, 추가적인 데이터나 모델 업데이트가 필요하지 않습니다.

- **Performance Highlights**: 다섯 개의 벤치마크(CHAIR, HallusionBench, POPE, MME, LLaVA-Bench)에서 ConVis의 효과를 검증한 실험 결과, 다양한 MLLM에서 환각을 효과적으로 줄이면서 전반적인 응답 생성 성능을 유지할 수 있음을 보여주었습니다. 이 결과는 LLaVA-1.5, MiniGPT-4, mPLUG-Owl2와 같은 모델에서도 일관되게 나타났습니다.



### TraIL-Det: Transformation-Invariant Local Feature Networks for 3D LiDAR Object Detection with Unsupervised Pre-Training (https://arxiv.org/abs/2408.13902)
Comments:
          BMVC 2024; 15 pages, 3 figures, 3 tables; Code at this https URL

- **What's New**: 이번 연구는 Transform-Invariant Local (TraIL) 특징과 TraIL-Det 아키텍처를 도입하여 3D LiDAR 객체 탐지의 한계를 극복하는 방법을 제안합니다.

- **Technical Details**: TraIL 특징은 강체 변환(invariance)과 점 밀도 변동에 효과적으로 적응하며, 인근 구조의 국소 기하학(local geometry)을 캡처하는 데 중점을 두고 설계되었습니다. 고차원 TraIL 특징을 관리 가능한 표현으로 인코딩하기 위해 비대칭 기하학적 특징을 갖춘 Multi-head self-Attention Encoder (MAE)를 제안합니다.

- **Performance Highlights**: KITTI와 Waymo 데이터셋에서 mAP 측면에서 기존의 자가 지도(self-supervised) 3D 객체 탐지 방법보다 성능이 우수합니다. 예를 들어, KITTI 데이터셋에서는 67.8, Waymo 데이터셋에서는 68.9의 mAP를 달성했습니다.



### Evaluating Attribute Comprehension in Large Vision-Language Models (https://arxiv.org/abs/2408.13898)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 대규모 비전-언어 모델의 속성 이해 능력을 평가하기 위한 새로운 프레임워크를 제안합니다. 속성 인식(Attribute Recognition)과 속성 계층 관계 이해(Attribute Hierarchy Understanding) 두 가지 관점에서 접근합니다.

- **Technical Details**: 논문은 비전-언어 작업을 위한 세 가지 평가 방법인 시각적 질문 응답(Visual Question Answering, VQA), 이미지-텍스트 매칭(Image-Text Matching, ITM), 및 이미지-텍스트 코사인 유사성(Image-Text Cosine Similarity, ITC)을 사용하여 대규모 비전-언어 모델의 속성 이해 능력을 평가합니다. VAW 데이터셋을 활용하여 다양한 속성 및 계층_annotations에 대한 이해도를 분석합니다.

- **Performance Highlights**: 대규모 비전-언어 모델은 좋은 속성 인식 능력을 보유하고 있지만, 계층 이해 능력은 상대적으로 제한적입니다. ITM은 ITC보다 더 세부적인 디테일을 잘 포착하여 속성 이해 작업에 더 적합합니다. 또한, 캡션 내의 속성 정보가 속성 이해에 중요한 역할을 합니다.



### RT-Attack: Jailbreaking Text-to-Image Models via Random Token (https://arxiv.org/abs/2408.13896)
- **What's New**: 이 논문에서는 기존의 Text-to-Image(T2I) 모델이 Not-Safe-For-Work(NSFW) 콘텐츠를 생성하는 데 있어 중요한 보안 취약점이 있다는 점을 강조하고, 이를 해결하기 위한 새로운 두 단계의 쿼리 기반 블랙박스 공격 방법인 RT-Attack을 제안합니다.

- **Technical Details**: RT-Attack 방법은 첫 번째 단계에서 초기 적대적 프롬프트를 세웠고, 두 번째 단계에서는 이를 개선하기 위해 최적화하여 타겟 NSFW 프롬프트의 이미지 특징과의 유사성을 극대화합니다. 이 과정에서 CLIP(text similarity) 모델을 활용하여 적대적 프롬프트와 타겟 프롬프트 간의 의미적 유사성을 증가시키는 방식을 적용합니다.

- **Performance Highlights**: 광범위한 실험 결과, RT-Attack이 최신 프롬프트 검출기 및 안전하게 훈련된 T2I 모델에 효과적으로 공격할 수 있음을 validated하며, 이는 T2I 모델이 지속적으로 진화하는 적대적 기술에 대응하기 어려운 도전 과제를 드러내는 결과입니다.



### Making Large Language Models Better Planners with Reasoning-Decision Alignmen (https://arxiv.org/abs/2408.13890)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 Chain-of-Thought (CoT) 추론 과정을 자동 운전의 의사 결정 프로세스에 통합하는 새로운 모델인 RDA-Driver를 제안합니다. RDA-Driver는 CoT와 계획 결과 간의 일치를 보장하기 위해 reasoning-decision alignment 제약조건을 도입합니다.

- **Technical Details**: RDA-Driver는 멀티모달 LLM을 기반으로 한 결정 모델로, 다양한 시각 정보를 BEV( birds-eye-view) 피처 표현으로 변환합니다. 이 모델은 CoT 추론과 계획을 동시에 수행하며, CoT와 계획 결과 간의 랭킹 손실을 통해 설명과 결정의 일관성을 유지합니다.

- **Performance Highlights**: RDA-Driver는 nuScenes 및 DriveLM-nuScenes 벤치마크에서 실험 평가를 통해 자동 운전 시스템의 성능을 향상시키는 데 효과적임을 증명하였습니다. 특히 nuScenes 데이터셋에서 0.80 L2 에러와 0.32 충돌률로 최첨단 계획 성능을 달성하였고, DriveLM-nuScenes에서도 0.82 L2 에러 및 0.38 충돌률의 우수한 결과를 기록하였습니다.



### Camouflaged_Object_Tracking__A_Benchmark (https://arxiv.org/abs/2408.13877)
- **What's New**: 이 논문은 Camouflaged Object Tracking Dataset (COTD)을 소개하며, 이는 camouflaged object (위장된 물체) tracking을 평가하기 위한 첫 번째 전문 벤치마크입니다. 또한, HiPTrack-MLS라는 새로운 tracking 프레임워크를 제안하여 camouflaged objects의 추적 성능 향상에 기여합니다.

- **Technical Details**: COTD는 200개의 시퀀스와 약 80,000개의 프레임으로 구성되어 있으며, 각 프레임은 상세한 bounding box로 주석 처리가 되어 있습니다. 이 데이터는 다양한 환경에서 camouflaged objects를 포괄적으로 다루며, 독특한 도전 과제를 제시하여 기존의 벤치마크와 차별화됩니다. HiPTrack-MLS는 multi-level features를 활용하여 성능을 강화하는 혁신적인 트래커입니다.

- **Performance Highlights**: 20개의 기존 tracking 알고리즘을 평가한 결과, camouflaged objects에 대해 현저한 성능 저하가 나타났습니다. COTD는 기존의 추적 방법들이 해결하지 못하는 복잡한 도전 과제를 강조하며, 향후 연구에 대한 기초를 마련합니다.



### Particle-Filtering-based Latent Diffusion for Inverse Problems (https://arxiv.org/abs/2408.13868)
Comments:
          Mohammad Hadi Sepanj, Nicholas Pellegrino, and Chris Czarnecki contributed equally

- **What's New**: 본 논문에서는 기존의 이미지 기반 역문제를 해결하기 위한 접근의 한계를 극복하기 위해 새로운 particle-filtering 기반의 프레임워크를 도입하였습니다. 기존 방식이 해결 공간을 거의 탐색하지 않는 반면, 제안된 방법은 비선형 탐색을 통해 더 나은 품질의 솔루션을 도출합니다.

- **Technical Details**: 제안된 PFLD(Particle-Filtering-Based Latent Diffusion) 방법은 reverse SDE(확률적 미분방정식) 해결 초기 단계에서 해결 공간을 더 잘 탐색하도록 설계되었습니다. 이는 파티클 필터(Particle Filter)를 이용해 잠재 공간에서 여러 샘플을 생성하고 이를 비교하여 최적의 솔루션에 접근하는 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, PFLD는 FFHQ-1K 및 ImageNet-1K 데이터셋에서 기존 최첨단 솔버(SoTA Solver)인 PSLD에 비해 super resolution, Gaussian debluring, inpainting 작업에서 뛰어난 성능을 보였습니다.



### Draw Like an Artist: Complex Scene Generation with Diffusion Model via Composition, Painting, and Retouching (https://arxiv.org/abs/2408.13858)
- **What's New**: 최근의 텍스트-이미지 확산 모델에서의 인상적인 이미지 품질 향상에 비해 복잡한 씬 생성이 상대적으로 미개척된 분야라는 점에 주목하였으며, 이러한 문제를 해결하기 위한 복잡한 씬의 정의와 이를 바탕으로 한 복잡성 분해 기준(Complex Decomposition Criteria, CDC)을 제안했습니다.

- **Technical Details**: 제안하는 방법은 작가의 그림 과정에서 영감을 받아 복잡한 씬 생성을 '구성(composition)', '페인팅(painting)', '리터칭(retouching)'의 세 단계로 나누는 프레임워크인 복합 확산(Complex Diffusion, CxD)을 기반으로 합니다. 이 방법은 대형 언어 모델(LLM)의 체인 오브 사고 능력을 사용하여 복잡한 프롬프트를 CDC에 따라 분해하고, 단순화된 프롬프트와 구성을 관리합니다.

- **Performance Highlights**: CXD는 복잡한 프롬프트에서도 시맨틱적으로 일관되고 비주얼적으로 다양한 고품질 이미지를 생성하는 데 있어 이전의 SOTA(State of the Art) 접근 방법을 능가하는 성능을 보여주었습니다. 주요 실험 결과는 복잡한 씬 생성을 위한 성능이 현저히 향상되었음을 증명합니다.



### Tangram: A Challenging Benchmark for Geometric Element Recognizing (https://arxiv.org/abs/2408.13854)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 연구에서는 Tangram이라는 새로운 벤치마크를 소개하여 기존의 LMM들이 기하학적 요소를 인식하는 능력을 평가합니다. Tangram은 초등 및 중등 학교의 시험, 경쟁, 교과서에서 수집한 1,080개의 다양한 기하학적 도면을 포함하고 있으며, 각 도면에는 4개의 질문이 연결되어 총 4,320개의 시각-질문-답변 쌍을 제공합니다.

- **Technical Details**: Tangram 벤치마크는 세 가지 난이도로 나뉘며, LMM의 기하학적 요소 인식 능력을 평가하기 위해 설계되었습니다. 모델의 성능 평가를 위해서는 '단순하지만 흥미로운' 카운팅 작업이 적용되어, 도면 내의 문자, 삼각형, 원, 선분을 세는 작업이 요구됩니다.

- **Performance Highlights**: 실험 결과, 가장 성능이 좋은 모델조차도 기하학적 요소를 인식하는 데 있어 56.8%의 저조한 정확도를 기록했습니다. 이는 중학생의 93.6%와 전문가의 99.5% 정확도에 비해 현저히 낮은 수치로, LMM의 기하학적 문제 해결 능력이 인간에 비해 상당한 격차가 있음을 드러냅니다.



### LaneTCA: Enhancing Video Lane Detection with Temporal Context Aggregation (https://arxiv.org/abs/2408.13852)
- **What's New**: 본 연구에서는 LaneTCA라는 새로운 비디오 차선 검출 방법을 제안합니다. 이 방법은 개별 비디오 프레임 간의 정보를 효과적으로 집합하는 방법을 탐구하며, 과거 프레임의 정보를 직접적으로 활용하여 보다 종합적인 역사적 문맥을 제공합니다.

- **Technical Details**: LaneTCA에는 두 가지 중요한 모듈인 ‘Accumulative Attention’과 ‘Adjacent Attention’이 포함되어 있습니다. Accumulative Attention 모듈은 차량의 주행 동안 시각 정보를 지속적으로 축적하며, Adjacent Attention 모듈은 이전 프레임의 차선 정보를 현재 프레임으로 전달합니다. 이 두 모듈은 Transformer 아키텍처를 기반으로 설계되었습니다.

- **Performance Highlights**: 본 연구는 일반적으로 사용되는 두 개의 벤치마크 데이터셋에서 광범위한 정량적 및 정성적 실험을 수행하였으며, LaneTCA가 여러 가지 새로운 최첨단 기록을 달성함으로써 방법의 효과성을 입증하였습니다.



### Bring the Power of Diffusion Model to Defect Detection (https://arxiv.org/abs/2408.13845)
- **What's New**: 이 논문에서는 비구조적 결함 감지를 위해 새로운 방법인 Denoising Diffusion Probabilistic Model (DDPM)과 경량 감지 모델을 결합하여 높은 차원의 의미 표현을 활용하는 접근법을 제안합니다. 이를 통해 경량 감지 모델의 성능을 향상시키고, 결함 감지의 정확도를 증가시킵니다.

- **Technical Details**: 프레 훈련된 DDPM을 사용하여 모델의 초기 특징을 추출하고, Residual Variational Auto-Encoder (ResVAE)를 통해 메모리 요구 사항을 줄입니다. 이 과정에서 동적 교차 융합(dynamic cross-fusion) 방법을 사용하여 결함 감지 모델에 고차원 특징을 정제하는 기술을 적용합니다.

- **Performance Highlights**: 실험 결과 다양한 산업 데이터셋에서 제안된 방법이 경쟁력 있는 결과를 달성하였으며, 높은 차원의 의미 정보를 경량 감지 모델에 효과적으로 주입함으로써 성능을 극대화했습니다.



### Exploring Reliable Matching with Phase Enhancement for Night-time Semantic Segmentation (https://arxiv.org/abs/2408.13838)
Comments:
          ECCV 2024

- **What's New**: 이번 논문에서는 NightFormer라는 새로운 방법을 제안하며, 이는 야간 이미지의 의미적 분할을 위한 엔드 투 엔드 최적화 접근 방식을 채택합니다. 기존의 방법들이 주간 이미지의 분포에 억지로 맞추려 했던 문제를 피하고, 저조도 환경에서의 내재한 도전 과제를 탐구합니다.

- **Technical Details**: NightFormer는 두 가지 주요 모듈로 구성됩니다: (1) 픽셀 수준의 텍스처 향상 모듈은 Fourier 변환의 위상 작업을 이용해 야간 장면에서의 텍스처 세부정보를 효과적으로 포착합니다. (2) 객체 수준의 신뢰할 수 있는 매칭 모듈은 저조도 환경에서 신뢰할 수 있는 주의 메커니즘을 이용하여 정확한 연관 매칭을 수행합니다.

- **Performance Highlights**: NightFormer는 NightCity, BDD, Cityscapes와 같은 다양한 벤치마크에서 기존의 최첨단 야간 의미적 분할 방법들과 비교했을 때 우수한 성능을 보였습니다.



### PropSAM: A Propagation-Based Model for Segmenting Any 3D Objects in Multi-Modal Medical Images (https://arxiv.org/abs/2408.13836)
Comments:
          26 figures, 6 figures

- **What's New**: 이 논문에서는 3D 의료 이미징에서 효과적으로 사용할 수 있는 새로운 세분화 모델인 PropSAM을 소개합니다. PropSAM은 2D 바운딩 박스 또는 스케치 마스크와 같은 하나의 뷰 프롬프트를 기반으로 세분화를 수행할 수 있는 것이 특징입니다.

- **Technical Details**: PropSAM은 CNN 기반의 UNet 아키텍처와 Transformer 기반의 모듈을 결합하여 슬라이스 내부 정보 처리 및 슬라이스 간 전파(Propagation)를 최적화합니다. 이를 통해 구조적 및 의미적 연속성을 유지하면서 다양한 이미징 모달리티에서의 세분화 효과를 향상합니다.

- **Performance Highlights**: PropSAM은 44개의 의료 데이터셋에서 평균 Dice Similarity Coefficient (DSC)를 18.1% 향상시키는 성능을 보였습니다. 또한, PropSAM은 사람의 프롬프트 효율성을 높이며, 2-view 프롬프트 모델에 비해 사용자 상호작용 시간을 37.8% 줄였습니다.



### Multi-SIGATnet: A multimodal schizophrenia MRI classification algorithm using sparse interaction mechanisms and graph attention networks (https://arxiv.org/abs/2408.13830)
- **What's New**: 이 논문에서는 정신질환인 조현병(Schizophrenia)의 분류를 위해 새로운 멀티모달 그래프 어텐션 네트워크인 Multi-SIGATnet을 제안합니다. 기존 방법의 한계를 극복하고 뇌의 비유클리드 네트워크 구조에서 중요한 정보를 효과적으로 학습하는 데 초점을 맞추었습니다.

- **Technical Details**: Multi-SIGATnet은 구조적 및 기능적 정보를 통합하여 멀티모달 데이터로 변환하여 조현병 환자에 대한 더 종합적이고 풍부한 특징을 확보합니다. 비대칭 컨볼루션 네트워크를 기반으로 한 희소 상호작용 메커니즘을 적용하여 주요 특징을 효과적으로 추출하고, 희소 학습 전략을 통해 불필요한 연결을 필터링하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 Multi-SIGATnet 모델은 COBRE 및 UCLA 데이터셋에서 각각 81.9% 및 75.8%의 평균 정확도를 달성하여 기존 그래프 어텐션 네트워크(GAT) 방법보다 각각 4.6% 및 5.5% 높은 성능을 보였습니다. 실험 결과, 이 방법은 조현병 식별에서 우수한 성능을 나타냅니다.



### Few-Shot Histopathology Image Classification: Evaluating State-of-the-Art Methods and Unveiling Performance Insights (https://arxiv.org/abs/2408.13816)
- **What's New**: 이번 연구는 의료 영상인 조직 병리 이미지(histopathology images)에서의 few-shot classification 기술의 성능을 평가한 최초의 연구로, 제한된 레이블 데이터가 있는 의료 분야에서 기존 깊은 학습(Deep Learning) 모델의 기초성에 도전하고 있습니다.

- **Technical Details**: 본 연구에서는 CRC-TP, NCT-CRC-HE-100K, LC25000, BreakHis 등의 네 가지 조직 병리 데이터셋을 기반으로, 5-way 1-shot, 5-way 5-shot, 5-way 10-shot 시나리오에 대해 Prototypical Networks, Model-Agnostic Meta-Learning (MAML), SimpleShot, LaplacianShot, DeepEMD와 같은 최신 few-shot 학습 기법을 적용했습니다. 각 기법은 강화된 일반화와 강인성을 위해 다양한 전략론을 채택하고 있습니다.

- **Performance Highlights**: 5-way 1-shot, 5-way 5-shot, 5-way 10-shot 각각의 사례에서의 정확도가 각각 70%, 80%, 85%를 초과하여, 메타-러닝(meta-learning) 접근 방식이 전통적인 미세 조정(fine-tuning) 기술과 동등한 성능을 나타냄을 보여주었습니다.



### On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspectiv (https://arxiv.org/abs/2408.13809)
- **What's New**: 최근 Kolmogorov-Arnold Networks (KANs)는 함수 근사를 위한 새로운 접근법으로 주목받고 있습니다. 이 논문에서는 KANs의 적대적 내성을 이미지 분류 작업에 중점을 두고 탐구합니다.

- **Technical Details**: KANs의 적대적 공격에 대한 강인함을 다루고 있으며, 기존의 Multilayer Perceptrons (MLPs)와 KANs 간의 적대적 예제 전이 가능성을 연구합니다. MNIST, FashionMNIST, KMNIST 데이터셋을 사용하여 실험을 진행합니다.

- **Performance Highlights**: KANs는 기존 MLP 모델과 비교하여 특정 적대적 공격에 대한 저항성을 평가 받으며, 이미지 작업에서의 새로운 보안 분석을 제공합니다.



### TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather (https://arxiv.org/abs/2408.13802)
Comments:
          15 pages, submit to IEEE TIP

- **What's New**: 이 논문에서는 자율주행 시스템에서의 LiDAR(레이저 거리 측정) 데이터의 신뢰성을 개선하기 위해 두 가지 대규모 데이터셋인 Weather-KITTI와 Weather-NuScenes를 도입했습니다. 이 데이터셋은 비, 안개, 눈과 같은 세 가지 기상 조건을 포함하며, 원래의 LiDAR 수집 정보를 유지하고 각 날씨 조건에 대한 포인트 수준의 레이블을 제공합니다.

- **Technical Details**: 제안된 모델인 TripleMixer는 세 가지 믹서 레이어로 구성됩니다: Geometry Mixer Layer, Frequency Mixer Layer, Channel Mixer Layer. Geometry Mixer Layer는 포인트 클라우드의 기하학적 공간 정보를 캡처하는 데 중점을 두고, Frequency Mixer Layer는 다중 스케일 주파수 정보를 추출하며, Channel Mixer Layer는 포인트 클라우드의 다중 채널 특성 정보를 향상시킵니다. 이러한 구조는 denoising(잡음 제거) 과정에서 중요한 기하학적 세부사항과 구조적 특징을 보존할 수 있게 합니다.

- **Performance Highlights**: 제안된 TripleMixer 모델은 WADS 데이터셋 및 Weather-KITTI와 Weather-NuScenes 데이터셋에서의 실험을 통해 최첨단의 denoising 성능을 보여주었습니다. 또한, 이 모델을 기존의 세그멘테이션 프레임워크에 통합했을 때, 다운스트림 작업의 성능이 크게 향상되었음을 입증하였습니다.



### Selectively Dilated Convolution for Accuracy-Preserving Sparse Pillar-based Embedded 3D Object Detection (https://arxiv.org/abs/2408.13798)
Comments:
          CVPR Workshop 2024 (The 7th Workshop on Efficient Deep Learning for Computer Vision)

- **What's New**: 이 논문에서는 Selectively Dilated Convolution (SD-Conv)을 제안하여, Sparse Point Cloud에서의 3D 객체 탐지 정확도를 개선하고자 합니다. 이를 통해 픽셀의 중요성을 평가하여 선택적으로 dilation을 적용함으로써, 더 높은 정확도와 효율성을 달성할 수 있습니다.

- **Technical Details**: SD-Conv는 각 convolution layer에서 중요한 pillars를 식별하고, 그에 따라 선택적으로 dilation을 수행하여 receptive field를 확장하는 방식을 채택합니다. 이를 통해 sparse pillar networks의 공간 정보 흐름(fSIF)을 개선하고, 3D 객체 탐지 성능을 향상시킵니다. 또한 수치 연산의 절약을 위해, SPADE+라는 비용 효율적인 sparse convolution accelerator를 설계하여, SD-Conv의 지원을 극대화합니다.

- **Performance Highlights**: 제안된 SD-Conv는 기존의 Submanifold Convolution (SubM-Conv)을 대체하여 정확도를 회복하고, PointPillars, CenterPoint 및 PillarNet과 같은 다양한 최신 네트워크에서 94.5%, 72.3%, 41.3%의 계산 수를 줄이면서도 기존 정확도를 초과했습니다. 또한, SPADE+에서의 실험 결과 16.2배의 속도 향상을 달성했습니다.



### CV-MOS: A Cross-View Model for Motion Segmentation (https://arxiv.org/abs/2408.13790)
- **What's New**: CV-MOS는 RV와 BEV 잔여 맵을 결합하여 이동 객체 분할을 위한 새로운 방법을 제안합니다. 이 접근법은 모션 정보의 활용을 극대화하여 LiDAR 기반 MOS 작업의 성능을 향상시킵니다.

- **Technical Details**: 본 논문에서는 3D Spatial Channel Attention Module (SCAM)을 통해 모션 브랜치를 정제하고 정보 손실을 완화하며 모델의 추론 속도를 개선합니다. CV-MOS는 RV와 BEV에서의 모션 정보를 동시에 이용하는 구조를 가지고 있습니다.

- **Performance Highlights**: SemanticKitti 데이터셋에서 CV-MOS는 검증 세트와 테스트 세트에서 각각 77.5% 및 79.2%의 IoU(Intersection over Union) 점수를 기록하며 현재 SOTA(State-of-the-Art) 성능을 보여줍니다.



### 3D-VirtFusion: Synthetic 3D Data Augmentation through Generative Diffusion Models and Controllable Editing (https://arxiv.org/abs/2408.13788)
- **What's New**: 이 연구에서는 pretrained 대형 모델의 힘을 이용하여 3D 레이블이 있는 훈련 데이터를 자동으로 생성하는 새로운 패러다임을 제안합니다. 이 방법은 물체의 구조와 외형을 다양한 2D 이미지로 생성하고, 이를 3D 객체로 변환하며 가상 장면을 생성하는 지속적인 과정입니다.

- **Technical Details**: 제안된 방법인 3D-VirtFusion은 텍스트-이미지(diffusion models 및 chatGPT 사용)를 기반으로 하여 3D 포인트 클라우드 장면을 생성합니다. ControlNet을 이용해 다양한 외형의 객체를 생성하고, 자동 드래그 기반 편집을 통해 객체 다양성을 향상시킵니다. 이 과정에서 생성된 2D 이미지는 3D 객체로 재구성되어 무작위로 조합된 3D 가상 장면을 생성하게 됩니다.

- **Performance Highlights**: 본 연구의 결과는 기존 3D 데이터셋의 다양성을 증가시켜, 현장 이해(scene understanding) 작업에서의 모델 성능 향상에 기여합니다. 특히, 수요가 높은 3D 모델 제작에 있어 필요한 데이터의 양과 품질을 동시에 충족시키며, 소수 샷 학습(few-shot learning) 문제를 해결하는 데 도움이 됩니다.



### Localization of Synthetic Manipulations in Western Blot Images (https://arxiv.org/abs/2408.13786)
- **What's New**: 최근의 딥러닝 및 생성 시스템의 발전은 합성 미디어(synthetic media) 창작에 크게 기여했으며, 특히 현실 콘텐츠의 현지 조작(local alteration)에서 매우 사실적인 합성 조작(synthetic manipulation)을 삽입하는 데 있어 심각한 문제를 제기하고 있습니다. 이러한 문제는 멀티미디어 데이터에 국한되지 않고, 과학 출판물에 포함된 생물학적 이미지로도 확장됩니다. 본 연구에서는 웨스턴 블롯(Western blot) 이미지에서 합성 조작을 지역화하는 작업을 다루고 있습니다.

- **Technical Details**: 우리는 이미지에서 추출한 작은 패치를 분석하여 진짜( pristine)와 합성된(synthetic) 픽셀을 구분할 수 있는 합성 탐지기(synthetic detector)를 제안합니다. 패치(patch) 기여도를 집계하여 합성 픽셀을 강조하는 변조 히트맵(tampering heatmap)을 추정합니다. 두 개의 조작된 웨스턴 블롯 이미지 데이터셋을 통해 우리의 방법론을 평가했습니다.

- **Performance Highlights**: 우리의 탐지기는 보편적인 합성 콘텐츠 생성기에 대해 뛰어난 성능을 보여주며, 거짓 알람(false alarm)이 거의 없습니다. 또한, 다양한 의미의 과학 이미지에서 테스트하여 일반화 가능성을 보여주었습니다.



### Towards Completeness: A Generalizable Action Proposal Generator for Zero-Shot Temporal Action Localization (https://arxiv.org/abs/2408.13777)
Comments:
          Accepted to ICPR 2024. Code is available at this https URL

- **What's New**: 이 논문에서는 Zero-Shot Temporal Action Localization (ZSTAL) 문제를 해결하기 위해 Generalizable Action Proposal generator (GAP)라는 새로운 모델을 제안합니다. GAP는 CLIP 모델과 통합되어 액션 제안을 전체적으로 생성하며, 기존의 손수 제작한 사후 처리(post-processing)가 필요 없습니다.

- **Technical Details**: GAP는 쿼리 기반 아키텍처를 채택하여 액션 제안의 완전성을 추정하고, 액션에 대한 동적 정보를 강조하는 Action-aware Discrimination loss를 도입합니다. 또한, Static-Dynamic Rectifying 모듈이 포함되어 CLIP의 일반화 가능한 정적 정보를 통합하여 제안된 액션을 정제합니다.

- **Performance Highlights**: GAP는 Thumos14 및 ActivityNet1.3의 두 가지 ZSTAL 벤치마크에서 최신 성능을 달성하였으며, 각각 3.2% 및 3.4%의 평균 mAP 성능 개선을 보여주었습니다.



### Extremely Fine-Grained Visual Classification over Resembling Glyphs in the Wild (https://arxiv.org/abs/2408.13774)
Comments:
          13 pages, 7 Figures, 8 Tables

- **What's New**: 본 연구는 자연 장면에서 유사 문자(glyph) 구분을 위한 극도로 미세한 시각 인식 작업을 다루고 있으며, 매우 도전적인 유사 문자 데이터셋을 생성하고, 두 단계의 대비 학습 방식을 통해 성능을 개선하였습니다.

- **Technical Details**: 첫 번째 단계에서는 지도 대비 학습(supervised contrastive learning)을 통해 네트워크를 초기화하며, 두 번째 단계에서 CCFG-Net(Contrastive Classification in Fine-Grained Networks) 아키텍처를 도입하여 유클리드 및 각 공간(Euclidean and Angular spaces)에서의 분류(classification)와 대비 학습(contrastive learning)을 통합합니다. 이 방법은 5가지 서로 다른 인코더(encoder)로 평가되었습니다.

- **Performance Highlights**: 다양한 최첨단 미세 분류 기법들과 비교하여 제안된 방법의 우수성을 보여줍니다. 특히, 본 연구의 접근 방식은 자연 환경에서의 유사 문자 인식을 효과적으로 개선하였습니다.



### ICFRNet: Image Complexity Prior Guided Feature Refinement for Real-time Semantic Segmentation (https://arxiv.org/abs/2408.13771)
- **What's New**: 이번 연구에서는 이미지의 복잡성을 세분화 특징을 개선하는 사전 정보로 활용하여 실시간 의미적 세분화의 정확성을 높이는 방법을 제안합니다. 이는 이미지 내 다양한 픽셀 지역들이 서로 다른 수준의 복잡성을 가짐을 관찰한 데 기반하고 있습니다.

- **Technical Details**: 우리는 Image Complexity prior-guided Feature Refinement Network (ICFRNet)을 도입합니다. 이 네트워크는 복잡성과 세분화 특징을 집계하여 Image Complexity Guided Attention (ICGA) 모듈 내에서 세분화 특징을 정제하는 주의 맵을 생성합니다. ICFRNet은 이미지 복잡성과 의미론적 세분화를 동시에 예측하는 이중 작업 프레임워크를 통해 최적화됩니다.

- **Performance Highlights**: Cityscapes와 CamViD 데이터세트에 대한 실험 결과, ICFRNet은 높은 정확도를 보이며, 복잡한 영역에 대해서도 효율적인 실시간 세분화 성능을 유지함을 보여주었습니다.



### TranSplat: Generalizable 3D Gaussian Splatting from Sparse Multi-View Images with Transformers (https://arxiv.org/abs/2408.13770)
- **What's New**: TranSplat은 정확한 지역 특성 매칭을 유도하기 위해 예측된 깊이 신뢰도 맵을 활용하는 새로운 방법론을 제시합니다. 이 방법은 기존의 G-3DGS 방식들이 가진 한계를 극복하고, 비오버랩 영역에 대한 깊이 추정 정확도를 향상시키기 위해 단안 깊이 추정 모델의 지식을 선행 정보로 활용합니다.

- **Technical Details**: TranSplat은 깊이 예측과 특징을 활용하여 각 픽셀을 3D Gaussian 원소로 투영하는 방법입니다. Depth-aware Deformable Matching Transformer (DDMT)라는 변환 기반 모듈을 통해 깊이 추정을 수행하며, 이는 초기 깊이 분포를 기반으로 높은 신뢰도를 가진 깊이 후보를 우선시합니다. 이 외에도 Depth Refine U-Net을 사용하여 깊이 분포를 정제하고, 모든 Gaussian 매개변수(센터, 공분산, 불투명도, 구형 조화 계수)를 각 픽셀에 대해 병렬로 예측합니다.

- **Performance Highlights**: TranSplat은 RealEstate10K와 ACID 벤치마크에서 최고의 성능을 기록하였으며, 경쟁력 있는 속도를 유지하면서도 강력한 데이터셋 간 일반화 능력을 보여줍니다.



### Enhancing Robustness of Human Detection Algorithms in Maritime SAR through Augmented Aerial Images to Simulate Weather Conditions (https://arxiv.org/abs/2408.13766)
- **What's New**: 본 연구에서는 미국 해안 경비대의 SAR(검색 및 구조) 작전의 효율성을 높이기 위해 YOLO 모델을 활용하여 다양한 환경 조건과 조명에서 훈련된 데이터셋을 사용하여 모델의 인식 정확도를 개선하였다.

- **Technical Details**: 이 연구는 드론에서 촬영한 공중 이미지로 구성된 두 개의 공개 데이터셋을 통합하여 생성된 26,548개의 이미지를 활용하였다. 데이터 증강(data augmentation)을 통해 다양한 날씨 조건과 조명에서의 이미지를 시뮬레이션하였으며, YOLOv5s, YOLOv5l, YOLOv10s, YOLOv10l 모델을 훈련하였다.

- **Performance Highlights**: 모델의 성능 평가는 Precision, Recall, F1 점수, mAP로 수행되었으며, YOLOv5l 모델이 0.91의 인간 인식률을 기록하여 최고 성능을 보였다. 데이터 증강을 통해 YOLOv10l 모델의 인식률도 0.91로 개선되었다.



### FMI-TAL: Few-shot Multiple Instances Temporal Action Localization by Probability Distribution Learning and Interval Cluster Refinemen (https://arxiv.org/abs/2408.13765)
Comments:
          9 pages, 3 figures

- **What's New**: 본 논문에서는 여러 개의 행동 인스턴스가 포함된 비디오에 대한 한계가 있는 기존의 Few-shot Temporal Action Localization (FS-TAL) 모델을 개선하기 위한 새로운 접근 방식을 제안합니다. 이를 통해 길이가 긴 쿼리 비디오에서 행동 인스턴스를 효과적으로 로컬라이즈할 수 있습니다.

- **Technical Details**: 우리가 제안하는 Spatial-Channel Relation Transformer (SCR-Transformer)는 시간, 공간, 채널 차원에서 추출된 특징 간의 관계를 탐색함으로써 특징 표현 능력을 향상시킵니다. 또한, 확률 학습 프로세스를 통해 비디오를 수동으로 단일 인스턴스 비디오 클립으로 분할하지 않고도 여러 인스턴스 temporal action localization을 동시에 처리할 수 있습니다.

- **Performance Highlights**: 우리의 방법은 ActivityNet1.3 및 THUMOS14와 같은 벤치마크 데이터셋을 사용한 철저한 실험을 통해 경쟁력 있는 성능을 달성하였으며, 기존 FS-TAL 방법들과 비교하여 최첨단의 성능을 보여줍니다.



### Self-Parameterization Based Multi-Resolution Mesh Convolution Networks (https://arxiv.org/abs/2408.13762)
- **What's New**: 이 논문은 3D 메쉬 밀집 예측을 위한 메쉬 합성곱 신경망 설계의 도전 과제를 다룹니다. 이미지 밀집 예측 작업에서 딥 러닝의 성공을 바탕으로, 비정형 그래프 데이터인 3D 표면 메쉬에 이러한 방법을 직접 적용하거나 확장하는 것이 어려움을 겪고 있습니다. 이를 해결하기 위해, 자가 매개변수화 기반의 다중 해상도 합성곱 네트워크(SPMM-Net)를 제안합니다.

- **Technical Details**: 저자들은 고해상도 입력 데이터에서 직접 다중 해상도 메쉬 피라미드를 구성하고, 서로 다른 메쉬 해상도 간의 일대일 표면 매핑을 사용하여 영역 인식 메쉬 다운샘플링 및 업샘플링 작업을 제안합니다. 이 방법은 불필요한 오류를 도입하지 않고 메쉬를 재정의합니다. 또한, 다중 해상도 합성곱 네트워크에서 고해상도 표현을 유지하고, 평행 다중 해상도 서브 네트워크 간의 정보 교환을 가능하게 하는 멀티 스케일 융합 기법을 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 SPMM-Net이 기존의 최첨단 방법보다 더 나은 성능을 보여 주며, 3D 메쉬 밀집 예측에서의 정확도를 향상시키는 것을 확인할 수 있습니다.



### Multimodal Ensemble with Conditional Feature Fusion for Dysgraphia Diagnosis in Children from Handwriting Samples (https://arxiv.org/abs/2408.13754)
- **What's New**: 이번 연구에서는 아동의 필기 장애인 developmental dysgraphia 진단을 위한 새로운 다중 모달 머신러닝 접근 방식을 제안합니다. 온라인 및 오프라인 필기 데이터를 모두 활용하여 기존 연구의 한계를 극복하고자 했습니다.

- **Technical Details**: 기존의 온라인 필기 데이터셋을 변환하여 오프라인 필기 이미지로 구성한 새로운 데이터셋을 생성했습니다. 본 연구에서는 단순 단어(simple word), 유사 단어(pseudoword), 어려운 단어(difficult word)의 서로 다른 유형의 단어 데이터를 분석하였고, SVM과 XGBoost 분류기를 온라인 및 오프라인 특징(feature) 각각에 대해 훈련시켰습니다. 또한, 조건부 특징 융합(conditional feature fusion)을 포함한 새로운 앙상블(ensemble) 방법을 제안했습니다.

- **Performance Highlights**: 본 연구의 방법론은 정확도 88.8%를 달성하였으며, 이는 기존의 SVM 단일 모달보다 12-14%, 기존 방법보다 8-9%, 전통적인 다중 모달 접근법보다 각각 3% 및 5% 향상된 성능입니다. 이러한 결과는 다중 모달 학습이 필기 장애 진단 향상에 큰 잠재력을 가지고 있음을 보여줍니다.



### Localization and Expansion: A Decoupled Framework for Point Cloud Few-shot Semantic Segmentation (https://arxiv.org/abs/2408.13752)
Comments:
          Accepted to ECCV 2024

- **What's New**: 본 논문은 Point Cloud Few-Shot Semantic Segmentation(PC-FSS) 문제를 해결하기 위한 새로운 방법인 Decoupled Localization and Expansion(DLE) 프레임워크를 제안합니다. DLE는 구조적 로컬리제이션 모듈(SLM)과 자기 확장 모듈(SEM)으로 구성되어 있는 간단하면서도 효과적인 접근 방식을 가지고 있습니다.

- **Technical Details**: SLM(Structural Localization Module)을 통해 객체 수준의 상관관계를 이용한 매칭 과정이 이루어지며, SEM(Self-Expansion Module)을 통해 완전한 객체를 유도합니다. SLM은 에이전트 수준의 상관관계를 통해 세밀한 타겟 위치를 결정하고, SEM은 intra-object similarity를 활용하여 잔여 객체 영역을 확보합니다.

- **Performance Highlights**: 본 연구의 DLE 프레임워크는 두 가지 도전적인 벤치마크에서 다양한 설정(1/2 way, 1/5 shot) 하에 기존의 최신 PC-FSS 접근 방식에 비해 상당한 성능 향상을 보여주었습니다.



### Enhancing Adaptive Deep Networks for Image Classification via Uncertainty-aware Decision Fusion (https://arxiv.org/abs/2408.13744)
Comments:
          13 pages, 27 figures. In ACM Multimedia 2024

- **What's New**: 이 논문에서는 다양한 컴퓨팅 자원에서의 인퍼런스(inference) 성능 향상을 위해 여러 개의 분류기(classifier) 헤드를 융합하는 Collaborative Decision Making (CDM) 모듈을 제안합니다. 기존의 접근 방식은 항상 마지막 분류기의 성능이 가장 우수하다고 가정했지만, 우리는 초기 분류기들이 특정 클래스에서 더 나은 성능을 낼 수 있다는 사실을 발견했습니다.

- **Technical Details**: CDM 모듈은 여러 분류기의 의사결정 정보를 융합하여 c-th 분류기의 성능을 향상시킵니다. CDM은 EDL(evidential deep learning)을 기반으로 한 불확실성(uncertainty) 인식 융합 방법을 채택하고, 융합 포화(fusion saturation)와 불공평성(fusion unfairness) 문제를 줄이기 위해 균형 항목을 설계했습니다. 추가적으로, Guided Collaborative Decision Making (GCDM) 프레임워크를 통해 마지막 분류기가 초기 분류기들의 학습 과정을 안내하도록 설정했습니다.

- **Performance Highlights**: 실험 결과, ImageNet 데이터세트에서 CDM과 GCDM은 인기 있는 적응형 신경망에서 0.4%에서 2.8%까지의 정확도 향상을 달성했습니다. GCDM을 통해 조정된 학습을 통해 초기 분류기의 정확도를 높이면서도 다양성(diversity)을 유지할 수 있음을 입증했습니다.



### MSVM-UNet: Multi-Scale Vision Mamba UNet for Medical Image Segmentation (https://arxiv.org/abs/2408.13735)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 Mamba를 기반으로 한 의료 이미지 분할을 위한 Multi-Scale Vision Mamba UNet 모델, 즉 MSVM-UNet을 제안합니다. 이 모델은 다중 스케일 특징 표현을 효과적으로 캡처하고 2D 비주얼 데이터의 방향 의존성을 해결하기 위해 설계되었습니다.

- **Technical Details**: MSVM-UNet는 VSS 블록에 다중 스케일 합성곱(multi-scale convolutions)을 도입하여 계층적 특징에서 다중 스케일 정보를 효율적으로 집합하고, LKPE (Large Kernel Patch Expanding) 레이어를 이용하여 더 효과적인 특징 맵 업샘플링(upsampling)을 수행합니다. 이 과정에서 공간(spatial) 및 채널(channel) 정보를 동시에 통합합니다.

- **Performance Highlights**: Synapse 및 ACDC 데이터셋에서 광범위한 실험을 실시한 결과, 제안된 MSVM-UNet 모델은 분할 품질 측정에서 State-of-the-Art 방법들보다 뛰어난 성능을 보였습니다. 특히 Synapse 데이터셋에서 DSC(다이스 계수) 85.00% 및 HD95(95 백분위 거리) 14.75mm를 기록했습니다.



### 3D-RCNet: Learning from Transformer to Build a 3D Relational ConvNet for Hyperspectral Image Classification (https://arxiv.org/abs/2408.13728)
- **What's New**: 본 논문에서는 기존의 ConvNet과 ViT의 장점을 결합한 3D 관계형 Convolutional Network(3D-RCNet)를 제안합니다. 이 모델은 일반적인 고차원 컨볼루션 연산의 효율성을 유지하면서도 ViT의 유연성을 활용하는 새로운 방법으로, 하이퍼스펙트럼 이미지(HSI) 분류에 탁월한 성능을 보입니다.

- **Technical Details**: 3D-RCNet은 ConvNet의 컨볼루션 작업 내에 Transformer의 self-attention 메커니즘을 통합하여 3D 관계형 컨볼루션 작업을 설계합니다. 이 방법은 공간-스펙트럼 특성을 추출하면서 내부 창에서 주의(attention) 작업을 수행하여 새로운 특성을 생성합니다. 이 구조는 3D ConvNet의 경량성과 ViT의 글로벌 특성 추출 능력을 모두 제공합니다.

- **Performance Highlights**: 세 가지 대표적인 HSI 데이터셋에 대한 실험적 평가 결과, 제안된 3D-RCNet 모델은 기존의 ConvNet 기반 및 ViT 기반 HSI 접근 방식보다 뛰어난 성능을 보였습니다. 특히, 3D-RCNet은 플러그 앤 플레이(plug-and-play) 작업으로 기존의 ConvNet 기반 HSI 분류 방법에 원활하게 통합될 수 있습니다.



### PhysPart: Physically Plausible Part Completion for Interactable Objects (https://arxiv.org/abs/2408.13724)
- **What's New**: 본 논문은 상호 작용이 가능한 객체에 대한 물리적으로 그럴듯한 부분 생성을 위한 새로운 접근 방식을 제안합니다. 이는 3D 생성 모델의 발전을 통해 자동 생성된 3D 물체의 모델링을 가능하게 하여 3D 프린팅, 로봇 시뮬레이션 환경 생성 등의 다양한 응용 분야에서 이점을 제공합니다.

- **Technical Details**: 우리는 분산 기반 (diffusion-based) 부분 생성 모델을 제안하며, 이 모델은 분류기 없는 가이드를 통해 기하학적 조건을 활용하고, 물리적 제약을 안정성과 이동성 손실 (stability and mobility losses)로 설정하여 샘플링 과정을 안내합니다. 이러한 방식은 복잡한 부분-전체 계층으로 이루어진 객체의 순차적 부분 생성을 위한 길을 열어줍니다.

- **Performance Highlights**: 우리의 모델은 형상 및 물리적 메트릭에서 기존 기준보다 성능이 우수하며, 특히 물리적 제약을 충분히 모델링하지 못한 방법들 대비 뛰어난 결과를 보여줍니다. 3D 프린팅, 로봇 조작 및 순차적 부분 생성 분야에서 응용을 시연하며, 높은 물리적 그럴듯함이 요구되는 실제 작업에서의 강점을 강조합니다.



### EMG-Based Hand Gesture Recognition through Diverse Domain Feature Enhancement and Machine Learning-Based Approach (https://arxiv.org/abs/2408.13723)
- **What's New**: 이번 연구는 표면 전기근육도(EMG) 신호를 활용한 손 제스처 인식의 새로운 분류 방법론을 제시합니다. 이 방법론은 다양한 형태의 특징 추출 기법을 연구하여 EMG 신호의 다양성을 포착하고, 효과적인 특징 선택 방법을 통해 계산 복잡성을 완화합니다.

- **Technical Details**: 연구는 23가지의 형태적, 시간 영역 및 주파수 영역 특징 추출 기술을 탐색하고, Extra Tree Classifier를 사용하여 가장 중요한 특징을 선택하여 여러 기계 학습 알고리즘과 함께 분류를 수행합니다. 이를 통해 KNN 알고리즘을 사용했을 때 97.43%의 정확도를 달성했습니다.

- **Performance Highlights**: 제안된 방법론은 기존 시스템보다 높은 성능 정확도를 제공하며, EMG 기반 손 제스처 인식 시스템의 정확성과 사용성을 향상시킵니다.



### TalkLoRA: Low-Rank Adaptation for Speech-Driven Animation (https://arxiv.org/abs/2408.13714)
- **What's New**: TalkLoRA는 개인화된 스피킹 스타일에 대한 적응과 긴 문장 처리 속도를 개선하기 위해 제안된 새로운 접근 방식입니다.

- **Technical Details**: TalkLoRA는 Low-Rank Adaptation (LoRA)을 활용하여 각 개체에 대해 소수의 매개변수로 어댑터를 훈련시킴으로써 새로운 스피킹 스타일에 효과적으로 적응합니다. 또한, chunking 전략을 사용하여 기존 transformer의 복잡성을 줄여 긴 문장을 처리할 수 있도록 합니다. 이는 transformer 기반의 스피치 기반 애니메이션 방법에 적용할 수 있습니다.

- **Performance Highlights**: TalkLoRA는 상태-of-아트(style adaptation) 적응을 달성하며, 품질을 희생하지 않고 추론 시간의 복잡성을 줄일 수 있음을 보여줍니다. 이 논문에서는 Speech-driven facial animation 모델의 LoRA 미세 조정을 위한 하이퍼파라미터 선택에 대한 통찰도 제공하고 있습니다.



### Riemann-based Multi-scale Attention Reasoning Network for Text-3D Retrieva (https://arxiv.org/abs/2408.13712)
- **What's New**: 이 논문에서는 텍스트와 3D 포인트 클라우드 데이터 간의 검색(task) 문제를 해결하기 위한 새로운 Riemannian 기반의 다중 스케일 주의 메커니즘을 가진 RMARN(Riemann-based Multi-scale Attention Reasoning Network)을 제안합니다.

- **Technical Details**: RMARN은 Adaptive Feature Refiner(AF)와 Riemann Local Similarity(RLS), Global Pooling Similarity(GPS) 모듈을 포함하여, 서로 다른 매니폴드에서 각기 다른 특징을 학습하고, 이를 통해 텍스트와 포인트 클라우드 간의 내재적 기하학적 관계를 반영합니다. 또한, Low-Rank Filter(LRF) 모듈을 사용하여 모델의 파라미터 수를 줄이고, 정확도를 유지합니다.

- **Performance Highlights**: T3DR-HIT 데이터셋에서의 실험 결과, 제안된 방법은 기존 방법들보다 탁월한 성능을 보였으며, 3,380쌍의 텍스트와 포인트 클라우드 데이터로 구성된 대규모 데이터셋을 통해 다중 스케일 텍스트-3D 검색 작업의 기초를 제공합니다.



### SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting (https://arxiv.org/abs/2408.13711)
- **What's New**: 최근 텍스트 기반 3D 장면 생성에서 significant advancements가 이루어졌습니다. 기존 방법들은 단일 뷰 이미지를 생성한 후 이를 3D 공간에 stitching 하는 방식이었으나, 이로 인해 공간 일관성(spatial consistency) 문제에 직면했습니다. 이에 대한 해결책으로 제안된 SceneDreamer360은 panoromic 이미지 생성을 통해 3D 장면의 일관성을 높이는 새로운 접근법입니다.

- **Technical Details**: 본 연구에서 제안하는 SceneDreamer360은 텍스트 기반(상징적인 연관을 통해)으로 3D 일관성 있는 장면 생성 모델입니다. 이 모델은 3D Gaussian Splatting (3DGS)을 활용하여 멀티뷰 파노라마 이미지를 만들고, 고해상도의 세부 정보가 풍부한 이미지를 생성하는 fine-tuned Panfusion generator를 포함합니다. 3D 장면 구축 시 novel point cloud fusion initialization 방식을 사용하여 높은 품질의 점군을 생성합니다.

- **Performance Highlights**: 다수의 실험 결과, SceneDreamer360은 파노라마 이미지 생성과 3DGS를 통해 고품질의 시각적으로 매력적인 3D 장면을 생성할 수 있음을 입증하였습니다. 기존의 방법들과 비교했을 때, 더 높은 품질의 점군을 생성하며, 3D 장면들 간의 일관성을 유지합니다.



### InSpaceType: Dataset and Benchmark for Reconsidering Cross-Space Type Performance in Indoor Monocular Depth (https://arxiv.org/abs/2408.13708)
Comments:
          BMVC 2024. This version supersedes 2309.13516

- **What's New**: 이번 연구에서는 기존의 실내 단안 깊이 추정 방법들이 미달성했거나 간과한 공간 유형(space types)을 집중적으로 분석했습니다. 특히, InSpaceType Dataset이라는 고품질 RGBD 데이터셋을 새롭게 제시하여 다양한 공간에서의 모델 성능 변화를 평가했습니다.

- **Technical Details**: 구체적으로, InSpaceType은 26개의 대표적인 공간 유형을 포함한 계층 그래프를 정의하고, RGBD 이미지를 최신 고해상도 스테레오 카메라로 수집했습니다. 또한, 기존 13개의 최신 최첨단 방법들에 대한 zero-shot 성능을 검토하고, 성능의 불균형을 발견하여 원인 분석을 진행했습니다.

- **Performance Highlights**: 연구 결과, 대부분의 방법들이 개인 실내 공간(head types)에서 높은 성능을 보였으나, 대형 방과 같은 드물게 나타나는 공간 유형(tailed types)에서는 성능이 크게 저하되는 문제가 발견되었습니다. 이를 통해, 실내 깊이 추정 모델들의 현실 세계에서의 신뢰성을 검토할 필요성이 강조되었습니다.



### CNN-Transformer Rectified Collaborative Learning for Medical Image Segmentation (https://arxiv.org/abs/2408.13698)
- **What's New**: 본 논문에서는 의료 이미지 분할(MIS) 작업을 위해 CNN과 Transformer 모델 간의 지식 전이를 통해 강력한 모델 학습을 가능하게 하는 새로운 CTRCL( CNN-Transformer Rectified Collaborative Learning) 프레임워크를 제안합니다.

- **Technical Details**: CTRCL 프레임워크는 정정된 로짓 기반 협업 학습(RLCL) 전략과 클래스 인식 특징 기반 협업 학습(CFCL) 전략을 포함하여, 로짓 및 특징 공간에서 CNN 기반 모델과 Transformer 기반 모델 간의 양방향 지식 전이를 수행합니다. RLCL 전략은 학생의 소프트 레이블에서 잘못된 영역을 선택하고 정정하기 위해 라벨을 적응적으로 사용하는 모듈을 도입하여 고품질의 소프트 레이블을 생성합니다.

- **Performance Highlights**: CTRCL은 Synapse Multi-organ, ACDC 및 Kvasir-SEG의 세 가지 MIS 벤치마크에서 기존의 협동 학습 방법들보다 더 우수한 성능을 보여줍니다. 특히, Kvasir-SEG 데이터셋에서 ResNet-50과 MiT-B2의 MAE 지표를 각각 42.93% 및 31.23% 감소시켰습니다.



### Guided and Fused: Efficient Frozen CLIP-ViT with Feature Guidance and Multi-Stage Feature Fusion for Generalizable Deepfake Detection (https://arxiv.org/abs/2408.13697)
- **What's New**: 본 논문에서는 deepfake 탐지를 위한 효율적인 Guided and Fused Frozen CLIP-ViT (GFF) 모델을 제안합니다. 이 모델은 Deepfake-Specific Feature Guidance Module (DFGM)와 Multi-Stage Fusion Module (FuseFormer)라는 두 가지 모듈을 통합하여, frozen pre-trained CLIP-ViT의 강점을 극대화합니다.

- **Technical Details**: GFF는 frozen CLIP-ViT에서 일반적 이미지 특징을 넘어 deepfake-specific 특징을 추출하도록 유도합니다. DFGM은 모델이 deepfake 탐지에 적합한 특징을 스스로 학습하도록 하고, FuseFormer는 다양한 단계에서 추출된 multi-stage 특징을 융합하여 정보 손실 없이 효과적으로 활용합니다.

- **Performance Highlights**: GFF 모델은 5 epochs의 짧은 훈련 기간만으로도 최첨단 성능을 달성하며, unseen GANs와 diffusion 모델에서 각각 99%와 97%의 높은 정확도를 기록합니다.



### Segment Any Mesh: Zero-shot Mesh Part Segmentation via Lifting Segment Anything 2 to 3D (https://arxiv.org/abs/2408.13679)
- **What's New**: 새로운 논문에서는 Segment Any Mesh(SAMesh)라는 제로샷(zero-shot) 접근 방식을 제안하여 메쉬(part segmentation)의 부분 분할에 대한 한계를 극복했습니다. SAMesh는 다단계로 구성되며, 첫 번째 단계에서 다양한 각도에서 메쉬를 렌더링하여 Segment Anything 2(SAM2)를 통해 2D 마스크를 생성합니다.

- **Technical Details**: SAMesh의 작동 과정은 두 가지 단계로 나뉩니다: 멀티모달 렌더링(multimodal rendering)과 2D-3D 리프팅(2D-to-3D lifting)입니다. 첫 번째 단계에서는 멀티뷰 렌더링(multiview renders)을 통해 생성된 2D 마스크를 SAM2에 적용하여 처리합니다. 두 번째 단계에서는 얻어진 마스크와 삼각형(face IDs) ID를 사용하여 매치 그래프를 생성하여 3D 파트에 해당하는 2D 레이블을 연관짓습니다.

- **Performance Highlights**: 우리의 방법은 Shape Diameter Function(ShapeDiam)과 비교 시 동등하거나 뛰어난 성능을 보이며, 다양성이 제한된 기존 벤치마크에서 벗어나 새로운 생성된 메쉬의 데이터셋을 통해 일반성을 개선한 점이 강조됩니다. 사용자 평가를 통해 SAMesh가 생성한 세분화의 품질이 ShapeDiam보다 우수하다는 결과를 보였습니다.



### GenCA: A Text-conditioned Generative Model for Realistic and Drivable Codec Avatars (https://arxiv.org/abs/2408.13674)
- **What's New**: 본 연구에서는 텍스트 조건부 생성 모델을 제안하여 다양한 정체성을 가진 포토 리얼리틱 (photo-realistic) 얼굴 아바타를 생성하고, 더욱 완전한 세부사항을 제공합니다. 기존의 방법들이 정적 아바타 생성에 제한되어 있었던 것과 대조적으로, 이 모델은 강력한 비모수 (non-parametric) 잠재 표현 공간을 통해 아바타를 조정할 수 있게 합니다.

- **Technical Details**: Generative Codec Avatars (GenCA)라는 새로운 3D 아바타 생성 프레임워크를 소개합니다. 이 모델은 두 단계로 이루어져 있으며, 첫 번째 단계에서 Codec Avatar Auto-Encoder (CAAE)를 통해 3D 인간 캡처 데이터셋으로부터 geometry 및 texture의 잠재 공간을 학습합니다. 두 번째 단계에서는 Identity Generation Model을 활용하여 입력 텍스트를 기반으로 중립적인 geometry 코드를 생성합니다.

- **Performance Highlights**: 이 연구의 결과로, 현재까지의 최첨단 생성 가능한 아바타보다도 더 완벽한 인간 머리 정밀 표현을 캡처할 수 있으며, 아바타의 움직이는 능력 또한 유의미하게 향상되었습니다. 또한, 단일 이미지로부터 아바타를 복원할 수 있는 가능성 및 아바타 편집 결과도 보여주었습니다.



### Hierarchical Network Fusion for Multi-Modal Electron Micrograph Representation Learning with Foundational Large Language Models (https://arxiv.org/abs/2408.13661)
Comments:
          Our paper is published at the workshop on Robustness of Few-shot and Zero-shot Learning in Foundation Models at NeurIPS 2023

- **What's New**: 이 연구에서는 전자 마이크로그래프(electron micrographs)의 분석을 위한 혁신적인 네트워크 아키텍처인 Hierarchical Network Fusion (HNF)을 제안합니다. 이 아키텍처는 여러 패치(patch) 시퀀스 및 비전 그래프(vision graphs)를 통해 전자 마이크로그래프의 복잡한 구조를 효과적으로 분석합니다.

- **Technical Details**: HNF는 다계층 네트워크 구조로, 패치 시퀀스와 비전 그래프 간의 정보를 교환하며 다양한 패치 해상도에서 지식을 통합합니다. 이 프레임워크는 Zero-shot Chain-of-Thought (Zero-Shot CoT) 프로세스를 통해 생성된 나노물질(nanomaterials)의 기술적 설명을 보조 정보로 활용하여 최종 작업을 지원합니다. 또한, Cross-modal attention 메커니즘을 사용하여 이미지와 언어적 통찰 간의 지식 융합을 예측합니다.

- **Performance Highlights**: 이 프레임워크는 전통적인 방법들보다 뛰어난 성능을 보이며, 배급 이동(distributional shifts)으로 인한 문제를 극복하고 높은 처리량(screening)을 가능하게 합니다. 이는 고해상도 전자 마이크로그래프의 더 포괄적이고 정확한 표현을 통해 나노 물질 분류 작업을 개선함을 보여줍니다.



### Mean Height Aided Post-Processing for Pedestrian Detection (https://arxiv.org/abs/2408.13646)
- **What's New**: 이 논문은 보행자 탐지기의 디자인에서 일반적인 객체 탐지 전략만을 따르지 않고 보행자 데이터셋의 고유한 특성을 고려하여 보행자 탐지의 정확성을 향상시키기 위해 'Mean Height Aided Suppression (MHAS)' 방식을 제안합니다.

- **Technical Details**: MHAS 방법은 각 예측의 존재 점수와 평균 신장을 기반으로 낮은 가능성을 가진 예측이나 비정상적인 신장을 가진 예측을 거부합니다. 특히, 이 방법은 후처리 단계에서 작동하게 설계되어 쉽게 구현 가능하며 플러그 앤 플레이 방식으로 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, MHAS 방법은 Caltech 및 Citypersons 데이터셋에서 기존 보행자 탐지기의 정확성을 유의미하게 개선하며, 특정 탐지기와의 조합을 통해 최첨단 성능을 달성합니다.



### Temporal Divide-and-Conquer Anomaly Actions Localization in Semi-Supervised Videos with Hierarchical Transformer (https://arxiv.org/abs/2408.13643)
Comments:
          Accepted at the 27th International Conference on Pattern Recognition (ICPR-2024)

- **What's New**: 이 연구에서는 비디오에서 이상을 탐지하고 그 위치를 파악하기 위한 새로운 방법론을 제시합니다. 기존의 방법이 세그먼트 수준의 다중 인스턴스 학습에 초점을 맞춘 것과 달리, 제안된 방법은 비디오 내 시간적 관계를 학습하여 이상 사건을 탐지하고 지역화하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 계층적 transformer 모델을 기반으로 하며, 관찰된 행동의 중요성을 평가하는 데 분할 및 정복 전략을 사용합니다. 비디오는 시간적 세그먼트로 분할되고, 각 자식 인스턴스의 영향을 측정하여 부모 비디오의 비정상적 행동을 분류합니다.

- **Performance Highlights**: 제안된 모델은 UCF-crime 및 ShanghaiTech 데이터셋에서 높은 성능을 보이며, 기존의 세그먼트 기반 다중 인스턴스 학습 및 최근의 의사 라벨링 기반 접근 방식과 비교할 때 유망한 성능을 달성합니다.



### Size Aware Cross-shape Scribble Supervision for Medical Image Segmentation (https://arxiv.org/abs/2408.13639)
- **What's New**: 이번 논문에서는 써클 방식의 약한 지도학습 기술인 Scribble supervision의 한계에 도전하기 위해 세 가지 새로운 방법을 제안했습니다. 이 방법들은 교차 형태 스크리블 주석(method) 방식, 교차 형태 기반의 가짜 마스크(pseudo mask) 생성 방법, 크기 인식 다중 가지(branch) 접근법으로, 특히 의학 이미지 분할 분야에서 의미 있는 개선을 가져왔습니다.

- **Technical Details**: 제안된 세 가지 방법은 다음과 같습니다: 1) 교차 형태 스크리블(annotation) 주석 기법으로 타겟 영역을 효과적으로 교차하고, 2) 이러한 교차 형태 스크리블을 기반으로 픽셀 레벨(pixle-level) 주석 수를 증가시키는 가짜 마스크 생성 기법, 3) 서로 다른 크기의 타겟을 처리하기 위한 크기 인식 다중 가지 접근법. 이들 방법은 모델의 구조와 매개변수를 깊이 있게 조사하여 최적화했습니다.

- **Performance Highlights**: 세 가지 제안된 방법은 테스트한 다수의 용종 데이터셋(polyp datasets)에서 mDice 점수를 유의미하게 향상시켰습니다. 특히, 이러한 방법들의 조합은 기존의 최첨단 스크리블 감독 방법과 비교하여 더욱 우수한 성능을 보여주었습니다.



### FungiTastic: A multi-modal dataset and benchmark for image categorization (https://arxiv.org/abs/2408.13632)
- **What's New**: FungiTastic은 20년에 걸쳐 지속적으로 수집된 데이터를 기반으로 한 새로운 종합적인 벤치마크와 데이터셋입니다. 이는 전문가에 의해 레이블이 지정되고 관리된 곰팡이 기록에 기초하고 있으며, 약 350k 개의 다중 모달 관측치와 5k 개의 세분화된 카테고리에서 650k 개 이상의 사진을 포함합니다. 이 데이터셋은 이전에 신뢰할 수 있는 레이블의 부분 DNA 시퀀스 진실을 포함하는 테스트 세트를 제공하는 유일한 벤치마크로, 여러 가지 주요 사용 사례를 지원합니다.

- **Technical Details**: FungiTastic 데이터셋은 사진, 위성 이미지, 기상 관측치, 분할 마스크 및 텍스트 메타데이터와 같은 다양한 데이터 유형을 포함합니다. 모든 관측치는 타임스탬프, 카메라 설정, 위치(경도, 위도, 고도), 기질, 서식지 및 생물 분류학적인 레이블과 같은 속성으로 주석이 달려 있으며, 이는 세부 연구 및 고급 분류 작업에 유용합니다. 데이터셋은 노벨 클래스 탐지(class detection), 비표준 비용 함수(non-standard cost functions), 테스트 타임 적응(test-time adaptation) 방법을 위한 시간 정렬 데이터 등 다양한 평가 프로토콜을 지원합니다.

- **Performance Highlights**: FungiTastic 데이터셋은 기존의 작은 데이터셋 대비 뛰어난 성능을 제공하며, 고급 기계 학습 모델의 개발 및 평가에 적합한 환경을 제공합니다. 이 데이터셋은 개방형(Open-set) 및 폐쇄형(Closed-set) 분류, 적은 수의 촬영(Few-shot) 학습, 도메인 변화(domain shift)와 같은 고질적인 현실 세계의 도전 과제를 해결하는 데 사용됩니다.



### Ancient but Digitized: Developing Handwritten Optical Character Recognition for East Syriac Script Through Creating KHAMIS Datas (https://arxiv.org/abs/2408.13631)
Comments:
          15 pages, 12 figures, 5 tables

- **What's New**: 이 논문은 고대 언어인 시리아어(Syriac)의 손으로 쓴 텍스트를 인식하는 광학 문자 인식(OCR) 모델 개발에 관한 연구 결과를 보고합니다. 연구진은 KHAMIS라는 데이터셋을 구축하여 손글씨 시리아어 인식을 위한 기준을 마련했습니다.

- **Technical Details**: KHAMIS 데이터셋은 31명의 대학생과 1명의 교수로부터 수집된 총 624개의 손으로 쓴 시리아어 문장으로 구성되어 있으며, Tesseract-OCR 엔진의 사전 훈련된 시리아어 모델을 손으로 쓴 데이터에 대해 미세 조정했습니다. 이 논문의 방법론은 데이터 수집, 전처리, 제안된 방법 설명, 모델 훈련, 평가 방법으로 나뉩니다.

- **Performance Highlights**: 새로 개발한 손글씨 OCR 모델은 훈련 세트에서 1.097-1.610%의 문자 오류율과 8.963-10.490%의 평가 세트 성능을 기록했으며, 테스트 세트에서는 18.89-19.71%의 문자 오류율과 62.83-65.42%의 단어 오류율을 달성했습니다. 이는 Tesseract의 기본 시리아어 모델에 비해 두 배 이상의 성능 향상을 보여줍니다.



### Temporally-consistent 3D Reconstruction of Birds (https://arxiv.org/abs/2408.13629)
- **What's New**: 이 논문은 환경 변화에 대한 유용한 생물 지표인 바다새의 3D 복원을 다룹니다. 본 연구는 특정 품종인 일반 바다제비(common murre)의 단일 비디오에서 3D 자세와 형태를 복원하는 방법을 제안합니다.

- **Technical Details**: 이 방법은 탐지(detection), 추적(tracking), 세분화(segmentation), 그리고 시계열 일관성을 갖춘 3D 복원을 포함하는 전체 파이프라인으로 구성됩니다. 또한, 현재의 단일 이미지 3D 복원기법을 시간 영역으로 확장하는 시간 손실(temporal loss)을 제안합니다.

- **Performance Highlights**: 제안된 방법을 통해, 본 데이터세트의 도전적인 시퀀스에서 최첨단 성능(state-of-the-art performance)을 달성했습니다. 이는 10,000 프레임의 비디오 관측 자료를 포함하며, 평균적으로 아홉 마리의 바다새가 동시에 다양한 행동을 보이는 데이터를 제공합니다.



### Recent Event Camera Innovations: A Survey (https://arxiv.org/abs/2408.13627)
- **What's New**: 이번 논문에서는 이벤트 기반 비전 기술에 대한 포괄적인 설문조사를 제시하며, 이벤트 카메라의 기본 원리와 이의 전통적인 프레임 카메라와의 비교를 통해 고유한 특성과 운영상의 차이를 강조합니다.

- **Technical Details**: 이벤트 기반 비전은 인간의 시각 시스템에서 영감을 받아 저지연(latency), 높은 동적 범위(dynamic range), 낮은 전력 소모 등 차별화된 특징을 지니며, 이벤트 카메라는 픽셀 단위로 환경의 빛 강도 변화를 비동기적으로 감지하여 실시간으로 장면의 동적 데이터를 생성합니다. 이 과정에서 각 픽셀은 독립적으로 작동하며 시각 신호를 아날로그 방식으로 처리합니다. 발생한 이벤트는 ⟨x,y,p,t⟩ 형태로 기록되며, 여기서 (x,y)는 픽셀 좌표, t는 시간, p는 변경의 극성을 나타냅니다.

- **Performance Highlights**: 이벤트 카메라는 전통적인 시스템에 비해 데이터 처리 효율을 높이고, 빠른 의사결정을 요구하는 응용 프로그램에 적합합니다. 이 기술은 배터리 구동 장치에 이상적이며, 여러 분야에서의 적절한 응용 가능성을 보여주고 있습니다. 또한, CVPR과 같은 주요 컴퓨터 비전 컨퍼런스에서 이벤트 기반 비전 관련 논문의 수가 급격히 증가하고 있음을 확인할 수 있습니다.



### Prompt-Softbox-Prompt: A free-text Embedding Control for Image Editing (https://arxiv.org/abs/2408.13623)
- **What's New**: 이번 논문은 Stable Diffusion XL(SDXL)에서의 텍스트 임베딩(text embeddings)에 대한 포괄적인 분석을 제공하며, 이미지 편집에서의 통제 가능성을 높이기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 이 논문에서는 ’aug_embedding’이 텍스트의 전체 의미를 포착하지만 최종 이미지 생성에 미치는 영향은 미미하다는 점과, ’BOS’ 및 ’Padding_embedding’이 의미 있는 정보를 포함하지 않으며, ’EOS’가 모든 단어의 의미와 스타일 특징을 지닌다는 세 가지 주요 통찰을 제시합니다. 이를 바탕으로 PSP(Prompt-Softbox-Prompt)라는 새로운 방법론을 통해 이미지 편집을 수행하고, 이는 크로스 어텐션 레이어에서 텍스트 임베딩을 수정하는 방식을 채택합니다.

- **Performance Highlights**: PSP 방법은 객체 교체, 추가 및 스타일 전이 등의 작업에서 기존 이미지 편집 방법보다 우수한 성능을 발휘하며, 세부 조정만으로 다양한 작업에 유연하게 적용될 수 있습니다.



### Preliminary Investigations of a Multi-Faceted Robust and Synergistic Approach in Semiconductor Electron Micrograph Analysis: Integrating Vision Transformers with Large Language and Multimodal Models (https://arxiv.org/abs/2408.13621)
Comments:
          Published at Deployable AI (DAI) Workshop at AAAI-2024

- **What's New**: 이 연구는 반도체 제조에서 자동화된 나노 소재 식별을 위한 혁신적인 아키텍처를 제안합니다. 이는 대형 언어 모델(LLMs) 및 대형 다중 모달 모델(LMMs)의 생성적 능력을 활용하는 방법입니다.

- **Technical Details**: 이 연구는 'Zero-Shot Chain of Thought (Zero-Shot CoT)' 및 'Few-Shot In-Context Learning' 접근법을 통해 특화된 작업을 위한 최적화된 프롬프트를 생성합니다. 또, Vision Transformers (ViT)를 이용해 입력 이미지를 패치로 나누어 1D 벡터로 전환하고, 위치 정보가 향상된 후 단일 글로벌 이미지 표현을 위한 분류 토큰을 추가합니다.

- **Performance Highlights**: 제안된 방법은 기존의 자동화된 나노 소재 식별 방법을 초월하여 높은 정밀도를 제공하며, 반도체 제조업계에서의 고처리량 스크리닝을 용이하게 합니다.



### Explainable Convolutional Networks for Crater Detection and Lunar Landing Navigation (https://arxiv.org/abs/2408.13587)
- **What's New**: 최근 몇 년 간의 달 탐사에서 자율 달 착륙 내비게이션이 중요한 요소로 떠오르고 있으며, 이를 위한 XAI(Explainable Artificial Intelligence, 설명 가능한 인공지능) 연구가 진행되고 있습니다. 본 논문에서는 자율 달 착륙을 위한 투명하고 이해 가능한 예측을 제공하는 방법에 대해 연구하였습니다.

- **Technical Details**: 논문에서는 Attention 기반의 Darknet53 네트워크를 통해 특징 추출이 이루어지며, 크레이터 탐지와 내비게이션을 위한 Attention 기반 YOLOv3 및 Attention-Darknet53-LSTM 모델이 제안됩니다. 또한, 모델의 설명 가능성을 높이기 위해 Attention 메커니즘을 도입하였습니다. Pearson 상관 계수(Pearson Correlation Coefficient, PCC)로 설명 가능성을 정량적으로 평가합니다.

- **Performance Highlights**: 실험 결과는 제안된 네트워크가 크레이터 탐지 및 자세 추정 분야에서 경쟁력 있는 성능을 제공함을 보여줍니다. 두 가지 네트워크 모두 달 착륙 중 적절한 크레이터 탐지 및 자세 추정 정확도를 달성하였습니다.



### CSS-Segment: 2nd Place Report of LSVOS Challenge VOS Track (https://arxiv.org/abs/2408.13582)
- **What's New**: 이번 기술 보고서는 ECCV 2024에서 열린 제 6회 LSVOS 챌린지 VOS 트랙에서 'yuanjie' 팀의 비디오 객체 분할(VOS) 솔루션 CSS-Segment를 소개합니다. CSS-Segment는 복잡한 물체 움직임과 장기적인 존재를 잘 처리할 것이며, 테스트 단계에서 J&F 점수 80.84로 2위에 올랐습니다.

- **Technical Details**: CSS-Segment는 4개의 주요 단계로 구성된 복원 프레임워크를 가지고 있습니다: Image Encoder, Mask Encoder, Object Transformer, Object Memory. 새로운 이미지 인코더는 SAM2의 설계를 기반으로 하여 실시간으로 비디오를 처리하며, 마스크 인코더는 Dense prompts와 이미지 임베딩을 통합합니다. Object Transformer는 객체 쿼리와 객체 메모리를 통합하여 효과적으로 출력을 생성합니다.

- **Performance Highlights**: CSS-Segment는 복잡한 비디오 시나리오에서 다른 모델보다 우수한 성능을 발휘하며, 특히 MOSE 데이터셋에서 20점 이상의 J&F 점수 향상을 보여주었습니다. 이로 인해 CSS-Segment는 어려운 VOS 환경에서도 높은 신뢰성을 기초로 하여 성과를 인정받고 있습니다.



### Can Visual Foundation Models Achieve Long-term Point Tracking? (https://arxiv.org/abs/2408.13575)
Comments:
          ECCV 2024 - Emergent Visual Abilities and Limits of Foundation Models (EVAL-FoMo) Workshop

- **What's New**: 이 논문은 대규모 비전 기초 모델의 기하학적 인식을 평가하여 장기 포인트 추적(점 추적, point tracking) 작업의 성능을 중요하게 검토합니다. 특히 두 가지 뷰간의 상관관계에서 벗어나 이 작업에서의 모델의 유용성을 조사한 것은 새로운 접근입니다.

- **Technical Details**: 기하학적 인식을 평가하기 위해 제로샷(zero-shot) 설정, 저용량 계층(probing), 및 Low Rank Adaptation(LoRA) 방식으로 모델을 분류하여 실험을 진행했습니다. Stable Diffusion과 DINOv2의 특징이 제로샷 설정에서 뛰어난 성능을 보여주며, DINOv2는 적은 학습 세팅에서 감독 학습 모델과 유사한 성능을 보였습니다.

- **Performance Highlights**: Stable Diffusion은 뛰어난 추적 성능을 보여주어, DINOv2와 함께 모델들의 기하학적 상관관계 이해도가 높음을 보여줍니다. DINOv2는 가벼운 교육 설정으로도 감독 모델의 성능을 일치시킬 수 있어, 초기화 방법으로서의 잠재력을 강조합니다.



### PointDGMamba: Domain Generalization of Point Cloud Classification via Generalized State Space Mod (https://arxiv.org/abs/2408.13574)
- **What's New**: PointDGMamba는 Domain Generalization (DG) 문제를 해결하기 위해 설계된 새로운 프레임워크로, 3D 포인트 클라우드 분류에서 강력한 일반화 성능을 보입니다. 이 모델은 Masked Sequence Denoising (MSD), Sequence-wise Cross-domain Feature Aggregation (SCFA), Dual-level Domain Scanning (DDS)와 같은 혁신적인 구성 요소로 구성됩니다.

- **Technical Details**: PointDGMamba는 글로벌 수용 영역(global receptive field)과 효율적인 선형 복잡성(efficient linear complexity)을 갖추고 있으며, 포인트 클라우드 시퀀스의 노이즈 제거 및 도메인 간 특징 집합을 촉진하는 역할을 합니다. 특히, MSD는 노이즈가 포함된 포인트 토큰을 선택적으로 마스킹하여 노이즈 축적을 줄이고, SCFA는 동일 클래스의 도메인 간 포인트 클라우드 특징을 집계하여 모델이 일반화된 특징을 추출하도록 유도합니다. DDS는 정보를 교환하여 3D 포인트 클라우드 데이터를 Mamba에 적합한 1D 시퀀스 데이터로 변환합니다.

- **Performance Highlights**: PointDGMamba는 기존의 최첨단 모델들과 비교했을 때, 3D 포인트 클라우드 데이터의 일반화 성능에서 탁월함을 보여주었으며, PointDA-10 및 새로운 PointDG-3to1 벤치마크에서의 광범위한 실험을 통해 그 효과성과 우수성을 입증했습니다.



### Variational Autoencoder for Anomaly Detection: A Comparative Study (https://arxiv.org/abs/2408.13561)
Comments:
          6 pages; accepted to IEEE ICCE 2024 for poster presentation

- **What's New**: 본 논문은 이상 탐지(anomaly detection)에서 현재 사용되는 Variational Autoencoder (VAE) 아키텍처의 비교 분석을 수행하며, 구체적으로 VAE, Gaussian Random Field prior를 적용한 VAE-GRF, 그리고 Vision Transformer를 통합한 ViT-VAE를 다룹니다. ViT-VAE는 다양한 시나리오에서 뛰어난 성능을 보여주며, VAE-GRF는 최적 성능을 달성하기 위해 더 복잡한 하이퍼파라미터 튜닝이 필요할 수 있습니다.

- **Technical Details**: 이 논문은 여러 VAE 아키텍처의 성능을 분석하며, reconstruction-based AD에 초점을 맞춥니다. VAE는 깊은 신경망의 재구성 능력을 활용하여 비정상적인 패턴을 식별하며, ViT-VAE는 이미지 기능에 ViT를 사용하여 AD 작업을 수행합니다. VAE-GRF는 tradition VAE와 비슷한 구조를 가지고 있지만, 부가적인 스칼라 파라미터를 두 개 추가하여 성능을 향상시킵니다.

- **Performance Highlights**: ViT-VAE는 MVTec AD 및 MiAD의 두 공개 데이터셋에서 VAE-GRF와 전통적인 VAE에 비해 경쟁력 있는 성능을 입증했습니다. ViT-VAE는 비지도 학습 방식으로 훈련되며, 이미지 공간과 잠재 공간 모두에서 이상 점수를 평가합니다.



### Learning from the few: Fine-grained approach to pediatric wrist pathology recognition on a limited datas (https://arxiv.org/abs/2408.13542)
- **What's New**: 본 논문은 소아 손목 병리 인식을 세분화된(fine-grained) 인식 문제로 재정의하며, X-ray 이미지를 통해 병리 구역을 자동으로 식별하는 방법을 제안합니다. 새로운 LION optimizer의 통합으로 기존 방법에 비해 성능을 개선했습니다.

- **Technical Details**: 이 연구에서는 특화된 네트워크 아키텍처를 사용하여 손목 병리 인식 문제를 해결합니다. Grad-CAM 기법을 이용하여 중요 지역을 강조하고, 제한된 데이터에서도 높은 정확도를 유지하는 효율적인 세분화 아키텍처를 개발했습니다.

- **Performance Highlights**: 제안된 방법은 기존 기준 방법에 비해 각각 1.06% 및 1.25%의 정확도 증가를 달성하였고, 전체 정확도는 86% 및 84%입니다. 또한 고관절 골절에 대한 민감도는 97%로, 손목 병리 인식 개선 가능성을 보여줍니다.



### An Open, Cross-Platform, Web-Based Metaverse Using WebXR and A-Fram (https://arxiv.org/abs/2408.13520)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2404.05317

- **What's New**: 본 논문은 WebXR 기반의 크로스 플랫폼 아키텍처를 제안하여, A-Frame 및 Networked-Aframe 프레임워크를 활용한 공간 웹 앱 개발을 가능하게 합니다. 이 구조는 웹과 확장 현실 장치 모두에서 접근 가능한 개방형 메타버스를 목표로 하고 있습니다.

- **Technical Details**: 이 연구는 WebXR 기술을 활용하여 A-Frame 및 Networked-Aframe 프레임워크의 통합을 통해 다양한 장치에서 실시간 멀티플레이어 경험을 가능하게 하는 것을 모색합니다. WebXR Device API는 XR 장치가 제공하는 기본 기능에 대한 고유의 추상화 계층을 제공하며, WebGL을 사용하여 웹 브라우저 내에서 2D 및 3D 그래픽을 렌더링합니다.

- **Performance Highlights**: 프로토타입 구현이 완료되어 다양한 플랫폼과 장치에서 몰입감 있는 경험을 지원하는 기술 스택의 능력을 평가하였습니다. 사용의 용이성에 대한 긍정적인 피드백이 있어, 제안된 접근 방식이 매력적이고 인터랙티브한 가상 공간을 촉진하는 데 효과적임을 입증합니다.



### AnoPLe: Few-Shot Anomaly Detection via Bi-directional Prompt Learning with Only Normal Samples (https://arxiv.org/abs/2408.13516)
Comments:
          Code is available at this https URL

- **What's New**: 이번 연구에서는 True Anomalies(진짜 이상 패턴)에 대한 사전 정보 없이도 Few-shot Anomaly Detection(소수 샘플 이상 탐지) 문제를 해결하기 위해 AnoPLe이라는 새로운 다중 모달 프로프트 학습 방법론을 제안합니다. AnoPLe은 텍스트와 비주얼 프롬프트 간의 쌍 방향 결합을 통해 두 모달리티 간의 깊은 상호작용을 촉진합니다.

- **Technical Details**: AnoPLe은 시뮬레이션된 이상 패턴을 사용하여 CLIP(Contrastive Language-Image Pretraining)을 통해 정상 샘플과 비정상 샘플을 구분하는 방식으로 작동합니다. 또한, 다중 뷰 신호를 학습 가능한 경량 디코더와 통합하여 지역 의미(semantics)를 향상시킵니다. 실험 결과, AnoPLe은 MVTec-AD와 VisA에서 각각 94.1%와 86.2%의 Image AUROC(Area Under the Receiver Operating Characteristic)을 기록했습니다.

- **Performance Highlights**: AnoPLe는 기존 SoTA(State-of-the-Art)인 PromptAD와 비교해도 떨어지지 않는 성능을 발휘하며, 실제 이상 패턴에 대한 노출 없이도 강력한 FAD 성능을 달성했습니다. 특히, 1-shot 및 4-shot 설정에서 MVTec에서 각각 94.1% 및 96.3%의 I-AUROC을 기록했습니다.



### DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation (https://arxiv.org/abs/2408.13509)
- **What's New**: 이 논문에서는 산업 제조에서의 이상 감지 성능을 향상시키기 위해 새로운 접근 방식을 제안합니다. 기존의 이상 데이터 생성 방법들이 생성하는 이상 사례의 다양성이 제한적이라는 문제를 해결하기 위해, DualAnoDiff라는 새로운 Diffusion(확산) 기반 모델을 개발하였습니다.

- **Technical Details**: DualAnoDiff는 두 개의 상호 연결된 Diffusion 모델을 사용하여 전체 이미지를 생성하고 해당 이미지에서 이상 부분을 생성하는 방식으로, 더 다양하고 현실적인 이상 이미지 쌍을 생성합니다. 이 모델은 LoRA(저-rank 어댑터)를 도입하여 단일 Diffusion 모델을 이중 연결된 모델로 확장하며, 자가 주의(interaction module)를 통해 두 가지 branch가 정보를 교환합니다.

- **Performance Highlights**: DualAnoDiff는 Mvtec AD 데이터셋에서 수행된 광범위한 실험을 통해 최첨단 수준의 성능(픽셀 레벨에서 99.1% AUROC 및 84.5% AP 점수)을 입증하였습니다. 이 모델은 기존 방법들보다 현실감과 다양성에서 뛰어난 성능을 보여주며, 하위 작업인 이상 감지, 이상 위치 식별, 이상 분류 작업에서도 성능을 크게 개선합니다.



### G3DST: Generalizing 3D Style Transfer with Neural Radiance Fields across Scenes and Styles (https://arxiv.org/abs/2408.13508)
Comments:
          GCPR 2024, Project page: this https URL

- **What's New**: 이 논문에서는 Neural Radiance Fields (NeRF)를 기반으로 하는 3D 스타일 전송에서의 새로운 접근 방식을 제안합니다. 기존의 방법이 각 장면 또는 스타일에 대해 최적화가 필요했지만, 본 연구는 이 과정을 생략하고 단일 학습 모델을 사용하여 다양한 장면에서 스타일 전송을 가능하게 합니다.

- **Technical Details**: 이 연구는 하이퍼네트워크(hypernetwork)를 NeRF 모델에 통합하여 즉각적으로 스타일화된 새로운 뷰를 생성할 수 있도록 합니다. 또한, 다중 뷰 일관성을 보장하기 위해 새로운 흐름 기반(multi-view consistency loss) 손실 함수를 도입했습니다. 이는 3D 스타일 전송에서 일관성을 유지하며 고품질 이미지를 생성하는 데 기여합니다.

- **Performance Highlights**: 제안된 방법은 다양한 장면과 예술적 스타일에 대해 평가되었고, 기존의 장면 특정 모델 없이도 고품질 및 다중 뷰 일관성을 갖는 스타일화된 이미지를 생성하는 성능을 보여줍니다. 이 결과는 특히 시간 소모가 큰 기존의 장면 기반 최적화 방법들과 비교할 때 효율성과 적용 가능성을 크게 향상시키는 방향으로, 3D 스타일 전송 분야에 중요한 발전을 나타냅니다.



### R2G: Reasoning to Ground in 3D Scenes (https://arxiv.org/abs/2408.13499)
- **What's New**: R2G(Reasoning to Ground)는 3D 장면에서 목표 객체를 추론적으로 접지하는 신경 기호 모델입니다. 이전 연구와는 달리, R2G는 시맨틱 개념 기반의 장면 그래프를 명시적으로 모델링합니다.

- **Technical Details**: R2G는 그래프 노드 내에 여러 객체 속성을 임베딩(embedding)하고, 개체 간의 공간적 관계는 엣지(edge)를 통해 나타내며, 미리 정의된 시맨틱 어휘를 활용합니다. 주의 주입(attention transferring)을 안내하기 위해, 학습 기반 또는 프롬프트(prompting)-기반 방법을 활용하여 참조 발화를 분석하고 이를 동일한 시맨틱 공간 내의 추론 지침으로 변환합니다.

- **Performance Highlights**: Sr3D/Nr3D 벤치마크에서 R2G는 이전 연구와 유사한 결과를 달성하면서 해석 가능성을 개선하여 3D 언어 접지의 새로운 길을 개척했습니다.



### On the Feasibility of Creating Iris Periocular Morphed Images (https://arxiv.org/abs/2408.13496)
Comments:
          in revision process

- **What's New**: 본 연구는 홍채(morphing) 이미지 생성을 위한 새로운 end-to-end 프레임워크를 제안하며, 이는 기존 Face Recognition Systems (FRS) 시스템들에게 도전이 되는 얼굴 변형(face morphing) 문제와 대조적으로 홍채도 마찬가지로 취약한 점을 강조한다.

- **Technical Details**: 이 프레임워크는 쌍 주제 선택(pair subject selection), 분할(segmentation), 변형(morph creation), 새로운 홍채 인식 시스템(iris recognition system)을 포함하는 여러 단계를 포함한다. 두 가지 주제 선택 방법론, 즉 무작위 선택(random selection)과 유사 반지름 크기 선택(similar radius size selection)을 탐색하며, 단일 변형 공격 탐지(Single Morphing Attack Detection) 알고리즘에 대한 취약성 분석도 수행된다.

- **Performance Highlights**: 결과적으로, 이 접근법은 전통적인 홍채 인식 시스템을 혼란스럽게 할 수 있는 매우 현실적인 이미지를 생성하여, 홍채 인식 시스템들이 페리오큘러(Periocular) 홍채 변형 이미지에 매우 민감하다는 점을 보여준다.



### Online Continuous Generalized Category Discovery (https://arxiv.org/abs/2408.13492)
- **What's New**: 본 연구에서는 데이터 스트림의 동적인 특성을 고려하여 실시간으로 데이터가 생성 및 삭제될 수 있는 Online Continuous Generalized Category Discovery (OCGCD)라는 새로운 시나리오를 제안합니다. 이를 통해 기존의 오프라인 지속 학습 방법의 한계를 극복할 수 있습니다.

- **Technical Details**: 우리는 DEAN (Discovery via Energy guidance and feature AugmentatioN)이라는 새로운 방법을 제안합니다. DEAN은 에너지 점수를 활용하여 새로운 카테고리를 발견하고, 에너지 기반 대조 손실 (energy-based contrastive loss)을 통해 효과적인 구분 학습을 수행합니다. 또한, 분산 기반 특징 증강 (variance-based feature augmentation) 방법을 통해 라벨이 없는 데이터의 유사 라벨링을 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, DEAN은 OCGCD 시나리오에서 기존 방법들보다 뛰어난 성능을 보여주었으며, 이는 우리가 제안한 에너지 지침 클러스터링과 대조 손실이 효과적임을 시사합니다.



### ESA: Annotation-Efficient Active Learning for Semantic Segmentation (https://arxiv.org/abs/2408.13491)
- **What's New**: 본 논문은 Entity-Superpixel Annotation (ESA)이라는 혁신적인 능동 학습(active learning) 전략을 도입하여 효율적인 주석(annotation) 방법을 제안합니다. 기존의 픽셀 기반 방법들이 가진 한계를 극복하고, 고유한 구조를 이용하여 주석 과정을 최소화합니다.

- **Technical Details**: ESA 방식은 클래스와 무관한 마스크 제안 네트워크(class-agnostic mask proposal network) 및 슈퍼픽셀(superpixel) 그룹화를 결합하여 지역 구조적 신호를 캡처합니다. 이 방법은 이미지 내 고엔트로피(high entropy)의 슈퍼픽셀을 우선적으로 선택하여 포괄적인 표현을 보장합니다. 이 접근 방식은 기존의 픽셀 기반 방법보다 주석 비용을 98% 줄이고 성능을 1.71% 향상시킵니다.

- **Performance Highlights**: ESA는 단 40회의 클릭으로 주석을 요구하는 반면, 기존 방법들은 보통 5000회의 클릭을 요구합니다. 이로 인해 전체적으로 더 유연하고 비용 효율적인 접근이 가능해집니다.



### HabitAction: A Video Dataset for Human Habitual Behavior Recognition (https://arxiv.org/abs/2408.13463)
- **What's New**: 이 연구에서는 인간의 습관적 행동(Human Habitual Behaviors, HHBs)을 반영하는 새로운 비디오 데이터셋을 구축하여, 기존의 행동 인식(Dataset recognition) 데이터셋에서 잘 다루어지지 않는 고유한 행동 양식을 제시하고 있습니다. HHBs는 개인의 성격과 심리적 변화를 분석하는데 중요한 역할을 합니다.

- **Technical Details**: 제안된 데이터셋은 30개 카테고리의 습관적 행동을 포함하여 300,000개 이상의 프레임과 6,899개 행동 인스턴스를 포함하고 있습니다. 또한, 제안된 두 가지 스트림(Two-stream) 모델은 사람의 골격(skeleton)과 RGB 외관(appearance)을 함께 사용하여, 인간 행동 비디오에서 숨겨진 지역 특징을 효과적으로 인식합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 methods보다 더 높은 정확도로 행동 인식을 수행함을 입증하였고, 이는 인간-컴퓨터 상호작용, 감정 인식 등 다양한 분야에 기여할 것으로 기대됩니다.



### Probing the Robustness of Vision-Language Pretrained Models: A Multimodal Adversarial Attack Approach (https://arxiv.org/abs/2408.13461)
- **What's New**: 이 논문에서는 비전-언어 프리트레인 (Vision-Language Pretraining, VLP) 트랜스포머의 적대적 견고성을 연구하고, 새로운 Joint Multimodal Transformer Feature Attack (JMTFA) 방법을 제안합니다. 이 방법은 시각 (visual) 및 텍스트 (textual) 모달리티에서 동시에 적대적 섭동을 도입하여, 모델의 예측 결과를 왜곡하는 데 중점을 둡니다.

- **Technical Details**: JMTFA는 크로스 모달 (cross-modal) 상호작용을 기반으로 시각과 언어 모달리티 간의 깊은 상관관계를 분석하여, 두 모달리티의 중요한 특징을 동시에 교란합니다. 이 연구에서는 비전-언어 이해 및 추론 작업 베이스라인과 비교하여 실험을 수행하였습니다. 트랜스포머의 크기와 적대적 견고성 간의 관계는 뚜렷하지 않다는 것이 발견되었습니다.

- **Performance Highlights**: 제안된 JMTFA 접근법은 VLP 모델에서 높은 공격 성공률을 기록하였으며, 복잡한 네트워크 아키텍처가 시각 모달리티보다 텍스트 모달리티에 더 많이 의존한다는 것을 입증하였습니다. 이러한 결과는 다중모달 AI 시스템의 신뢰할 수 있는 배포에서의 잠재적 위험을 강조합니다.



### Rethinking Video Deblurring with Wavelet-Aware Dynamic Transformer and Diffusion Mod (https://arxiv.org/abs/2408.13459)
Comments:
          accepted by ECCV2024

- **What's New**: 이번 연구에서는 기존 비디오 디블러링 방법들이 고주파 정보를 회복하는 데 한계가 있음을 지적하고, Diffusion Models (DMs)의 강력한 고주파 정보 생성 능력을 활용하여 새로운 비디오 디블러링 프레임워크인 VD-Diff를 제안합니다. 이 방법은 기존 방법들이 겪는 계산 리소스 소모와 블러 아티팩트에 의한 왜곡 문제를 해결합니다.

- **Technical Details**: VD-Diff는 Wavelet-Aware Dynamic Transformer (WADT)와 Diffusion Model을 통합한 형태로 설계되었습니다. WADT는 Low-frequency 정보를 보존하고 복구하며, DM이 생성한 High-frequency 정보를 활용해 비디오를 디블러링 합니다. 특히, DMs를 매우 압축된 잠재 공간에서 적용함으로써 고주파 정보가 포함된 prior features를 생성하고, 이로 인해 비디오의 디블러링 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과 VD-Diff는 GoPro, DVD, BSD, Real-World Video 데이터셋에서 최첨단(SOTA) 방법들을 초과하는 성능을 보였습니다. 특히, VD-Diff는 아티팩트 없는 디블러링을 용이하게 하며, 왜곡 없이 높은 품질의 비디오 복구를 가능하게 합니다.



### AdaOcc: Adaptive-Resolution Occupancy Prediction (https://arxiv.org/abs/2408.13454)
- **What's New**: 이번 연구에서는 도심 복잡한 시나리오에서 자율주행을 위한 새로운 접근 방식인 AdaOcc를 소개합니다. AdaOcc는 객체 중심 3D 재구성과 전체적인 점유 예측을 통합하여, 관심 영역(ROIs)에서 고해상도의 세밀한 3D 재구성을 가능하게 합니다.

- **Technical Details**: AdaOcc는 비균일 해상도(non-uniform resolution) 및 다중 모드 3D 표현(multi-modal 3D representation) 전략을 채택하여, 가까운 거리의 객체에 대해 높은 해상도 예측을 제공하고, 동시에 3D 점유 예측을 위한 저해상도 예측을 보조적으로 사용합니다. 이 과정에서 포인트 클라우드(point clouds)를 활용하여 정밀한 3D 형태를 생성합니다.

- **Performance Highlights**: nuScenes 데이터셋에서 AdaOcc는 기존 방법들보다 IOU(Intersection over Union)에서 13% 이상, Hausdorff 거리에서 40% 이상의 개선을 보여주었습니다. 이러한 결과는 정확한 객체 재구성을 통해 자율주행의 정밀한 의사결정에 기여함을 나타냅니다.



### Explainable Concept Generation through Vision-Language Preference Learning (https://arxiv.org/abs/2408.13438)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문에서는 개념 기반 설명(create concept explanation) 방법의 한계를 해결하기 위해 이미지 생성 문제로 프레임을 변경하고, 강화 학습 기반의 선호 최적화 알고리즘을 도입하여 시각-언어 생성 모델을 조정합니다. 이 접근법은 복잡하고 추상적인 개념들을 자동으로 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 개념 이미지 세트 생성을 개념 생성 문제로 재정의하고, 기존의 이미지를 사용하여 후보 개념 집합을 만드는 대신, 피부 색상 관련 키워드를 추출하여 생성 모델에 이를 안내합니다. 이 모델은 강화를 통해 생성 과정을 조정하여 적절한 개념 이미지를 자동 생성합니다. 결과적으로 생성된 개념은 기존의 이미지와는 다른 추상화 레벨을 가집니다. 여기서 사용된 알고리즘은 RLPO (Reinforcement Learning-based Preference Optimization)입니다.

- **Performance Highlights**: 본 연구는 실험을 통해 제안하는 방법이 복잡한 개념들을 명확하고 정확하게 표현할 수 있음을 증명했습니다. 이 방법은 신경망 분석을 위한 진단 도구로서의 가능성도 지니고 있습니다.



### Face Clustering via Early Stopping and Edge Reca (https://arxiv.org/abs/2408.13431)
- **What's New**: 본 논문에서는 기존의 복잡한 모델 설계와 지루한 클러스터링 프로세스를 극복하기 위해 두 가지 새로운 비지도 얼굴 클러스터링 알고리즘 FC-ES와 지도 얼굴 클러스터링 알고리즘 FC-ESER를 제안합니다.

- **Technical Details**: FC-ES에서는 이웃 기반(edge-based) 확률을 통해 데이터 샘플 간의 쌍(pairwise) 관계를 반영하며, 조기에 중단하는 전략을 도입하여 대규모 얼굴 클러스터링의 정확성과 재현율을 보장합니다. FC-ESER은 지도 학습의 이점을 활용하여 FC-ES에서 연결되지 않은 엣지 연결을 회상(recall)하는 새로운 전략을 제안합니다.

- **Performance Highlights**: FC-ES와 FC-ESER 알고리즘은 대규모 얼굴, 인물, 차량 클러스터링을 위한 여러 기준점에서 이전의 최첨단 방법들을 상당히 능가하는 성능을 보였습니다.



### Training-free Long Video Generation with Chain of Diffusion Model Experts (https://arxiv.org/abs/2408.13423)
- **What's New**: 이 논문은 비디오 생성 작업을 세 가지 하위 작업으로 분리하는 효율적인 비디오 생성 프레임워크인 ConFiner를 제안합니다. 이는 구조 제어(structure control)와 공간-시간 정제(spatial-temporal refinement)로 나누어져 있으며, 기존의 고비용 비디오 확산 모델의 문제를 해결하고 있습니다.

- **Technical Details**: ConFiner는 비디오 생성을 위해 세 가지 전문화된 모델을 사용하여 각각의 작업을 수행합니다. 이 프레임워크는 구조 제어를 위해 T2V 모델을, 세부 정제를 위해 T2I 모델과 T2V 모델을 사용합니다. 또한, 'coordinated denoising' 기법을 통해 서로 다른 노이즈 스케줄러를 사용하는 두 전문가가 협력할 수 있도록 합니다.

- **Performance Highlights**: ConFiner는 단 9번의 샘플링 단계(5초 미만)로 AnimateDiff-Lightning, LaVie, ModelScope T2V보다 모든 메트릭에서 우수한 성능을 보입니다. 또한, ConFiner-Long 프레임워크는 600프레임까지의 고품질 일관성 있는 비디오를 생성할 수 있습니다.



### TVG: A Training-free Transition Video Generation Method with Diffusion Models (https://arxiv.org/abs/2408.13413)
- **What's New**: 본 논문에서는 기존의 전문 기술이 필요한 비디오 전환 방법의 한계를 극복하기 위한 새로운 접근 방식인 Transition Video Generation (TVG) 방법을 제안합니다. 이 방법은 훈련이 필요 없는 방식으로 비디오 전환을 생성할 수 있습니다.

- **Technical Details**: Transition Video Generation (TVG) 방법은 비디오 수준의 diffusion 모델을 사용하며, Gaussian Process Regression ($\mathcal{GPR}$)을 활용하여 잠재 표현(latent representations)을 모델링하고, 각 프레임 간의 부드러운 전환을 보장합니다. 또한, 간섭 기반의 조건적 제어(interpolation-based conditional controls)와 Frequency-aware Bidirectional Fusion (FBiF) 아키텍처를 도입하여 시간적인 제어와 전환 신뢰성을 향상시킵니다.

- **Performance Highlights**: 기준 데이터 세트와 커스텀 이미지 쌍을 평가한 결과, 제안된 방법이 고품질의 부드러운 전환 비디오를 생성하는 데 효과적임이 입증되었습니다.



### Perturbation on Feature Coalition: Towards Interpretable Deep Neural Networks (https://arxiv.org/abs/2408.13397)
Comments:
          4 pages, 4 figures, 2 tables

- **What's New**: 이 논문은 DNN의 블랙박스 특성을 극복하기 위한 새로운 방법론인 feature coalitions에 기반한 perturbation 기반 해석을 소개합니다. 이 방법은 feature 의존성을 고려하여 더 정확한 네트워크 해석을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 두 가지 단계를 포함합니다: (1) 상관된 feature를 추출하고 (2) feature coalition에서 perturbation을 적용하는 것입니다. 이는 unsupervised 방식으로 수행되며, DNN의 깊은 정보를 활용하여 상관된 feature를 식별합니다. 이후, 지역 일관성 손실 함수(consistency loss)를 사용하여 분석의 일관성을 보장합니다.

- **Performance Highlights**: 정량적 및 정성적 실험을 통해 제안된 방법의 효과를 검증하였으며, 이전 방법보다 향상된 해석 성능을 보였습니다. 코드도 공개되어 있어 구현이 가능하며, 중요한 feature 간의 상관관계를 잘 반영합니다.



### Task-Oriented Diffusion Inversion for High-Fidelity Text-based Editing (https://arxiv.org/abs/2408.13395)
- **What's New**: 본 논문에서는 Task-Oriented Diffusion Inversion (TODInv)라는 새로운 프레임워크를 제안하여, 실画像 (real image)의 복원을 최적화하고 특정 편집 작업에 맞춰 편집하는 방법을 소개합니다.

- **Technical Details**: TODInv는 \\\mathcal{P}^{*} (P^*) 공간 내에서 프롬프트 임베딩 (prompt embeddings)을 최적화하여 이미지의 역전환을 수행합니다. 이 프레임워크는 U-Net의 여러 레이어와 시간 단계에서 서로 다른 임베딩을 활용하여 고충실도의 복원과 정확한 편집 가능성을 보장합니다.

- **Performance Highlights**: 산출된 실험 결과는 TODInv가 기존의 방법보다 정량적 및 정성적으로 우수한 성능을 보여주며, 몇 단계만으로도 효과적으로 작동할 수 있는 범용성을 입증합니다.



### MICM: Rethinking Unsupervised Pretraining for Enhanced Few-shot Learning (https://arxiv.org/abs/2408.13385)
Comments:
          ACMMM 2024 (Oral)

- **What's New**: 본 연구에서는 Masked Image Contrastive Modeling (MICM)이라는 새로운 패러다임을 제안하여, 비지도 학습 환경에서의 few-shot 학습 성능을 획기적으로 향상시킵니다. MICM은 효과적인 이미지 재구성을 통해 contrastive learning (CL)과 masked image modeling (MIM)을 결합하여 전반적인 알고리즘의 효율성을 증대시킵니다.

- **Technical Details**: MICM은 CL의 특정 객체 학습 능력과 MIM의 일반화된 시각적 특징 학습 능력을 통합하여, 빠르게 새로운 few-shot 작업에 적응할 수 있도록 설계되었습니다. 이 방법은 두 단계로 구성된 U-FSL 프레임워크를 사용하여, 비지도 사전 훈련과 few-shot 학습을 수행합니다.

- **Performance Highlights**: MICM을 적용한 결과, 데이터셋에 대한 전반적인 일반화 및 분별 능력이 크게 향상되었습니다. 연구에서 제안된 MICM 기반의 U-FSL 프레임워크는 기존의 선도적인 기준 모델들에 비해 월등한 성능을 보여주었으며, 다양한 실험 분석을 통해 그 우수성이 입증되었습니다.



### N-DriverMotion: Driver motion learning and prediction using an event-based camera and directly trained spiking neural networks (https://arxiv.org/abs/2408.13379)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 운전자의 동작을 학습하고 예측하기 위한 새로운 시스템과 이를 교육하기 위해 수집된 고해상도(1280x720) 데이터세트인 N-DriverMotion을 제안합니다. 이 시스템은 스파이크 입력을 나타내는 고해상도 운전 동작 데이터세트를 생성하는 이벤트 기반 카메라로 구성됩니다.

- **Technical Details**: 설계한 네 명의 층으로 이루어진 간소화된 Convolutional Spiking Neural Network (CSNN)은 고해상도 데이터세트를 사용하여 직접 훈련되며, 사전 처리 과정이 필요하지 않습니다. 이는 고해상도 이벤트 기반 스트림에 대한 실시간 추론을 위한 효율적인 SNNs로의 적응을 가능하게 합니다.

- **Performance Highlights**: 제안된 CSNN은 운전자의 동작을 인식하는 데 있어 94.04%의 정확도를 달성하여 자율주행차 또는 효율적인 신경망 아키텍처가 필요한 엣지 디바이스를 위한 안전하고 효율적인 운전 모니터링 시스템 개발에 기여할 수 있습니다.



### Learning Unknowns from Unknowns: Diversified Negative Prototypes Generator for Few-Shot Open-Set Recognition (https://arxiv.org/abs/2408.13373)
Comments:
          ACMMM 2024

- **What's New**: 본 연구에서는 다수의 레이블이 제한된 상황에서도 알려진 클래스와 알려지지 않은 클래스를 인식할 수 있는 Few-shot Open-set Recognition (FSOR) 문제에 접근하기 위한 새로운 방법을 제안합니다. 기존의 Negative-Prototype 기반 방법들은 알려진 클래스 데이터에만 기반한 부정 프로토타입을 생성하여 무한한 알려지지 않은 공간의 다양성을 효과적으로 표현하지 못하는 한계가 있었습니다.

- **Technical Details**: 우리는 	extbf{Diversified Negative Prototypes Generator (DNPG)}라는 새로운 접근 방식을 제안합니다. 이 방법은 기본 클래스에서 학습한 알려지지 않은 클래스의 공간 정보를 활용하여 보다 대표적인 부정 프로토타입을 생성합니다. 우리의 모델은 두 단계로 구성되며, 첫째는 기본 클래스의 알려지지 않은 공간을 학습하는 프리트레이닝(pre-training) 단계이고, 두 번째는 메타-학습(meta-learning) 과정에서 이를 활용하여 새로운 클래스의 부정 프로토타입을 구성하는 단계입니다. 추가적으로, Swap Alignment (SA) 모듈을 도입하여 프로토타입 붕괴를 방지하고 다양한 데이터 구성에 적응할 수 있도록 합니다.

- **Performance Highlights**: DNPG 모델은 여러 FSOR 표준 데이터셋에 대한 실험을 통해 현재의 최첨단 방법들에 비해 우수한 성능을 달성하였으며, 이는 알려지지 않은 공간의 다양성을 효과적으로 포괄하는 부정 프로토타입을 생성함으로써 가능했습니다.



### BiGS: Bidirectional Gaussian Primitives for Relightable 3D Gaussian Splatting (https://arxiv.org/abs/2408.13370)
- **What's New**: 우리는 Bidirectional Gaussian Primitives를 소개합니다. 이 기법은 3D 물체를 동적인 조명 아래에서 서페이스(surface) 및 볼륨(volumetric) 재료로 표현하고 렌더링하는 새로운 방법입니다. 기존의 Gaussian splatting 프레임워크에 light intrinsic decomposition을 통합하여 실시간으로 3D 물체를 다시 조명할 수 있는 기능을 제공합니다.

- **Technical Details**: 우리는 양방향 구형 조화 함수(bidirectional spherical harmonics) 기반의 조명 의존 외관 모델을 채택했습니다. 이러한 접근 방식은 3D 객체의 물체를 표면 재료와 볼륨 재료를 통합하는 동적 조명 환경에서도 효과적으로 렌더링합니다. 교차 조명(cross lighting) 및 환경 조명이 포함된 여러 재료를 렌더링할 수 있습니다.

- **Performance Highlights**: 우리 방법을 사용하여 One-Light-At-a-Time(OLAT) 데이터로부터 입력을 받아 실시간으로 포토리얼리스틱 사진처럼 생생한 이미지를 생성할 수 있음을 보여줍니다. 복잡한 재료로 이루어진 3D 객체의 재구성과 렌더링에 성공하였습니다.



### Shape-Preserving Generation of Food Images for Automatic Dietary Assessmen (https://arxiv.org/abs/2408.13358)
- **What's New**: 본 연구에서는 조건부 음식 이미지 생성을 위한 간단한 GAN(Generative Adversarial Network) 기반 신경망 아키텍처를 제안하며, 생성된 이미지의 음식과 용기 형태가 참조 입력 이미지의 형태와 밀접하게 유사하다는 점을 강조합니다.

- **Technical Details**: 이 시스템은 기존의 방법에서 발생하는 큰 양의 훈련 이미지가 필요한 문제를 해결하기 위해 GAN을 사용하여 트레이닝 이미지를 합성합니다. 제안된 프레임워크는 생성된 이미지에서 음식의 형태 보존 기능을 가지고 있으며, 이는 음식 인식 및 볼륨 추정의 정확성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험을 통해 제안한 GAN 구조는 매우 리얼한 음식 이미지를 생성할 수 있으며, 스타일 및 카테고리 변수를 사용하여 음식 카테고리를 쉽게 제어할 수 있음을 보여줍니다.



### SeA: Semantic Adversarial Augmentation for Last Layer Features from Unsupervised Representation Learning (https://arxiv.org/abs/2408.13351)
Comments:
          accepted by ECCV'24

- **What's New**: 이번 연구에서는 미리 훈련된 딥 모델에서 고정된 딥 특징을 이용한 새로운 방법인 세멘틱 적대적 증강(Semantic Adversarial Augmentation, SeA)을 제안합니다. 이 방법은 특히 정적인 깊이 특징을 사용하여 더 나은 성능을 발휘하도록 설계되었습니다.

- **Technical Details**: SeA는 고정된 딥 특징의 특징 공간에서 적대적 방향을 다른 예제들이 형성한 부분 공간으로 투영하여 의미 정보를 보존하는 방식으로 작동합니다. 이후 얻은 의미 방향에 따라 딥 특징을 변형하고 이를 통해 분류기를 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법인 SeA는 일반적으로 기존의 고정 딥 특징을 사용할 때보다 평균적으로 2% 향상되었고, 11가지 분류 과제 중 6가지에서 비싼 파인튜닝과 유사한 성능을 보였습니다. 이는 효율성과 효과성을 모두 보여주는 결과입니다.



### Latent Space Disentanglement in Diffusion Transformers Enables Zero-shot Fine-grained Semantic Editing (https://arxiv.org/abs/2408.13335)
- **What's New**: 본 연구는 Diffusion Transformers (DiTs)의 잠재 공간(latent space)을 조사하여, 텍스트와 이미지가 생성된 이미지의 의미(semantics)에 기여하는 방식을 밝혀냈습니다. 특히, 'Extract-Manipulate-Sample (EMS)'라는 새로운 프레임워크를 제안하여 제로샷(zero-shot) 세밀한 이미지 편집(image editing)을 가능하게 했습니다.

- **Technical Details**: 본 논문에서는 텍스트 및 이미지 잠재 공간의 분리가 가능하다는 점을 발견했습니다. 또한, 텍스트와 이미지 임베딩이 결합되어 세분화된 의미 표현 공간을 형성하며, 이미지 편집에 있어 두 공간을 함께 사용하는 것이 필수적이라는 것을 확인했습니다. 이를 기반으로, 다중 모달(Multi-modal) 대형 언어 모델을 사용하여 입력 이미지와 편집 목표를 텍스트 설명으로 변환하고, 이를 선형으로 조작하여 세밀한 조정을 수행합니다.

- **Performance Highlights**: 새롭게 제안된 EMS 방법이 정확한 세밀한 편집을 수행할 수 있으며, ZOFIE라는 제로샷 열린 소스 세밀한 이미지 편집 벤치마크를 통해 평가된 성능이 우수하다는 것을 보였습니다. 자동 평가 및 인간 평가 모두에서 편집의 효과성을 검증했으며, 일반화된 평가 기준 또한 제시하였습니다.



### Online Zero-Shot Classification with CLIP (https://arxiv.org/abs/2408.13320)
Comments:
          accepted by ECCV'24

- **What's New**: 본 연구에서는 이미지가 무작위로 도착하여 즉시 분류하는 온라인 제로샷 전이(online zero-shot transfer)라는 새로운 시나리오를 제안합니다. 기존의 제로샷 분류 방법에서의 유연성을 유지하면서, 도착한 이미지의 통계적 정보를 활용하여 대상 데이터의 분포를 캡처하고 성능을 향상시키는 방법을 개발하였습니다.

- **Technical Details**: 제안된 프레임워크 OnZeta는 온라인 레이블 학습(online label learning) 및 온라인 프록시 학습(online proxy learning) 방법을 통해 목표 데이터 분포를 모델링하고 이미지와 텍스트 간의 모달리티 갭을 완화합니다. 이론적으로 두 방법의 수렴이 보장됩니다.

- **Performance Highlights**: OnZeta는 ImageNet에서 78.94%의 정확도를 달성하였으며, 다른 13개의 다운스트림 태스크에서도 평균 3% 이상의 성능 향상을 보여주었습니다. 이 방법은 전체 데이터 세트에 접근하지 않고도 온라인 환경에서 효과적으로 활용될 수 있음을 입증하였습니다.



### Growing Deep Neural Network Considering with Similarity between Neurons (https://arxiv.org/abs/2408.13291)
- **What's New**: 본 논문은 신경망(neural networks)의 뉴런 형성 과정을 모방하여, 학습 단계에서 뉴런 수를 점진적으로 증가시키는 새로운 방법을 제안합니다. 이는 높은 예측 정확도를 유지하면서도 대규모 모델이 필요했던 기존의 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구는 뉴런 유사성 분포(similarity distributions)를 기반으로 한 제약을 도입하여 특성 추출 편향(feature extraction biases)을 줄입니다. 이를 통해 네트워크가 이전에 간과했던 이미지의 여러 부분에 주의를 기울여, 더 다양한 특징을 캡처할 수 있도록 합니다. 이는 CIFAR-10 및 CIFAR-100 데이터셋에서의 실험을 통해 개선된 정확도 및 Grad-CAM 시각화를 통해 입증되었습니다.

- **Performance Highlights**: CIFAR-10과 CIFAR-100에서 기존 방법과 비교했을 때, 우리의 방법은 전체 객체에 대한 인식을 더욱 강화하여 정확한 분류 결과를 도출합니다. 이러한 접근법은 또한 지속적인 학습에서 발생하는 급작스런 학습(shortcut learning) 문제를 완화할 수 있는 잠재력을 가지고 있습니다.



### Abstract Art Interpretation Using ControlN (https://arxiv.org/abs/2408.13287)
Comments:
          5 pages, 4 figures

- **What's New**: 이 연구는 추상적인 예술 해석과 텍스트-이미지 합성을 융합하는 방법을 탐구하며, 텍스트 프롬프트를 통해 이미지 구성에 대한 정밀한 공간적 제어를 달성하는 문제를 다룹니다. ControlNet의 기능을 활용하여 사용자에게 합성 과정에 대한 더 섬세한 제어를 제공합니다. 특히 우리는 삼각형과 같은 기하학적 원소에서 영감을 받은 새로운 조건을 도입했습니다.

- **Technical Details**: 본 연구에서 사용된 데이터셋은 37.6백만 이미지-텍스트 쌍으로 구성된 WIT(Wikipedia-based Image Text) 데이터셋에서 세심하게 구성되었습니다. ControlNet 모델을 훈련시키기 위해 WIT 데이터셋에서 다운로드된 원본 이미지를 사용했으며, BLIP 모델을 통해 이미지에 대한 캡션을 생성했습니다. ControlNet은 추가적인 조건으로서 세부화된 기하학적 도형을 사용하고 있으며, 이미지 합성의 과정에서 이러한 도형을 통해 더 나은 조작과 해석이 가능합니다.

- **Performance Highlights**: 우리가 제안한 방법의 성능 평가는 정성적 평가를 통해 이루어지며, 생성된 이미지의 질, 조건의 충실도, 그리고 전반적인 이미지 품질이 향상됨을 보여줍니다. ControlNet의 고유한 아키텍처 덕분에, 우리는 텍스트 입력을 통한 이미지 생성에서의 제어 가능성을 극대화시킬 수 있었습니다.



### SIn-NeRF2NeRF: Editing 3D Scenes with Instructions through Segmentation and Inpainting (https://arxiv.org/abs/2408.13285)
Comments:
          Code is available at: this https URL

- **What's New**: 이번 연구에서는 3D 장면에서 객체를 배경과 분리하여 선택적으로 편집할 수 있는 새로운 방법인 SIn-NeRF2NeRF (sn2n)를 제안합니다. 이 방법은 이전의 Instruct-NeRF2NeRF (in2n) 기법을 개선하여 객체를 더욱 정교하게 수정할 수 있도록 합니다.

- **Technical Details**: sn2n은 객체 마스크와 텍스트 프롬프트를 입력으로 받아, 객체와 배경을 분리한 후 3D 장면을 수정합니다. 이를 위해 멀티뷰 분할 기법과 SPIn-NeRF를 사용하여 객체의 편집과 3D 배경 인페인팅을 수행하였습니다. 최종적으로 두 장면을 병합하여 편집된 3D 장면을 생성합니다.

- **Performance Highlights**: sn2n은 3D 장면에서 객체를 효과적으로 분리하고 편집함으로써 기존 방법들보다 더 나은 결과를 보여주며, 다양한 예시를 통해 resizing과 이동이 가능한 것을 입증하였습니다.



### From Radiologist Report to Image Label: Assessing Latent Dirichlet Allocation in Training Neural Networks for Orthopedic Radiograph Classification (https://arxiv.org/abs/2408.13284)
Comments:
          This article is an abridged version of a 2016 master's thesis at the Karolinska Institute. The original is available upon request

- **What's New**: 이 연구는 스웨덴 Danderyd 병원에서 수집한 손목과 발목 X-ray (radiography) 이미지를 활용하여, 기계 학습 (machine learning, ML)과 인공 신경망 (artificial neural networks, ANN)을 통해 정형외과 방사선 사진의 해석을 개선하고자 하였습니다.

- **Technical Details**: 연구 방법으로는 2002년부터 2015년까지의 방사선과 의사 보고서와 함께 제공된 X-ray 이미지를 활용했습니다. LDA (Latent Dirichlet Allocation) 기법을 사용하여 방사선 사진에 대한 라벨을 생성하고, 이를 ANN 훈련에 사용했습니다. 생성된 라벨에 기반하여 ANN의 출력 결과를 수작업으로 검토하여 정확성을 평가했습니다.

- **Performance Highlights**: LDA를 통해 생성된 이미지 라벨은 ANN 훈련에 성공적으로 사용되었으며, ANN의 정확도는 라벨에 따라 60%에서 91% 사이로 나타났습니다. 그러나 LDA는 방사선과 보고서를 기반으로 정형외과 X-ray 라벨링에 높은 정확도를 제공하는 데 적합하지 않았습니다. 그럼에도 불구하고 ANN은 X-ray 이미지에서 특정 특징을 고도로 정확하게 감지할 수 있었습니다.



### CHARTOM: A Visual Theory-of-Mind Benchmark for Multimodal Large Language Models (https://arxiv.org/abs/2408.14419)
- **What's New**: CHARTOM은 다중 모달 대형 언어 모델을 위한 시각적 마음 이론 벤치마크로, 차트를 기반으로 FACT 질문과 MIND 질문을 통해 AI의 이해도를 평가합니다.

- **Technical Details**: CHARTOM 벤치마크는 112개의 차트로 구성되며, 기본적인 자료를 시각적으로 조작하여 실제 FACT 질문과 MIND 질문에 답변할 수 있도록 설계되었습니다. 각 차트는 바, 선, 파이, 산점도, 지도 차트 등 5가지 일반적인 유형으로 나뉘며, 조작된 버전은 심리학 문헌에 따른 시각적 오류를 포함합니다.

- **Performance Highlights**: AI는 FACT 질문에 정확히 답변할 수 있지만, MIND 질문의 경우 일반 독자가 차트에 어떻게 반응할지를 예측하는 것이 더 도전적임을 보여줍니다. 이는 AI가 시각적 자료의 진위와 인간의 인지를 고려하는 새로운 방법을 제시합니다.



### Uncovering Knowledge Gaps in Radiology Report Generation Models through Knowledge Graphs (https://arxiv.org/abs/2408.14397)
Comments:
          Code is available at: this https URL

- **What's New**: 최근 인공지능(AI) 기술의 발전으로 방사선 보고서의 자동 생성이 크게 개선되었습니다. 그러나 기존 평가 방법은 모델이 방사선 이미지를 이해하는 수준과 인간 수준의 세밀함(Granularity)에 도달하는 능력을 드러내지 못하고 있습니다. 이를 해결하기 위해, 우리는 ReXKG라는 시스템을 소개하며, 이 시스템은 처리된 보고서에서 구조화된 정보를 추출하여 종합적인 방사선 지식 그래프를 구축합니다.

- **Technical Details**: ReXKG는 처리된 보고서에서 정보를 추출하여 방사선 지식 그래프를 생성하는 시스템입니다. 우리는 다음과 같은 세 가지 평가 지표를 제안합니다: ReXKG-NSC(노드 유사성 계수), ReXKG-AMS(인접 행렬 유사성), ReXKG-SCS(서브그래프 커버리지 점수). 이 지표들은 다양한 지식 그래프 간의 노드 유사성, 엣지 분포, 서브그래프의 커버리지를 평가하는 데 사용됩니다.

- **Performance Highlights**: AI 모델들의 방사선 보고서 생성 성능에 대한 포괄적인 분석을 통해, 일반 모델(generalist)의 경우 80% 가까운 필수 엔티티 커버리지를 보여주지만, 의료 기기 세부사항에서는 방사선 전문의가 작성한 보고서에 미치지 못했습니다. AI 모델들은 개념을 과적합(overfit)하여 훈련 데이터에서 자주 등장하는 특정 개념에 집중하는 경향이 있으며, 결과적으로 세부사항이 부족하거나 비현실적(hallucinated)인 설명을 생성하였습니다. 일반 모델은 다양한 데이터에 노출되어 방사선 지식이 크게 향상되었습니다.



### Learning Tree-Structured Composition of Data Augmentation (https://arxiv.org/abs/2408.14381)
Comments:
          25 pages

- **What's New**: 이 논문에서는 데이터 수집이 제한된 상황에서 신경망을 훈련하기 위한 데이터 증강(data augmentation) 툴의 새로운 알고리즘을 제안합니다. 기존의 증강 방법보다 성공적으로 시간이 단축된 알고리즘을 설계하여, 여러 변환(transformation)을 트리 구조로 통합하는 방법을 제시합니다.

- **Technical Details**: 제안된 알고리즘은 깊이 d의 k개의 변환으로 구성된 이진 트리 구조를 탐색하여, 각 트리 노드는 각 변환에 대응합니다. 이 알고리즘은 상향식의 재귀적 검색(top-down, recursive search) 방식으로 진행되며, 최악의 경우 시간 복잡도는 O(2^dk)로, 기존의 O(k^d)보다 현저히 빠릅니다.

- **Performance Highlights**: 새롭게 수집된 다중 레이블 그래프 분류 데이터 셋에서, 제안된 알고리즘은 기존의 RandAugment보다 4.3% 향상된 성능을 보였고, 최근의 GraphAug 방법보다 43% 빠른 실행 시간을 기록했습니다. 이외에도 의료 이미지를 활용한 데이터셋에서 SimCLR보다 5.9% 더 나은 성능을 달성하였습니다.



### Equivariant Reinforcement Learning under Partial Observability (https://arxiv.org/abs/2408.14336)
Comments:
          Conference on Robot Learning, 2023

- **What's New**: 이 논문은 로봇 학습에서 샘플 효율성을 높이기 위해 대칭성을 활용하는 새로운 방법을 제안합니다. 특히, 대칭성을 신경망에 인코딩하여 에이전트가 과거의 솔루션을 재사용할 수 있도록 합니다.

- **Technical Details**: 기존의 완전 관측 Markov 결정 프로세스(MDP)에 대한 연구와는 달리, 이 논문은 부분 관측 프로세스인 POMDP에서 대칭성을 활용한 새로운 이론과 솔루션 방법을 제시합니다. 특정 그룹 대칭에 대한 등가성을 신경망에 Embed하여 에이전트가 더 나은 정책을 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 우리의 에이전트는 비대칭(non-equivariant) 접근 방식에 비해 샘플 효율성과 최종 성능에서 현저한 개선을 보여주었습니다. 특히, Advantage Actor-Critic (A2C) 및 Soft Actor-Critic (SAC) 알고리즘을 사용하여 로봇 조작 작업에서 실험한 결과, 대칭적 접근 방식이 더 뛰어난 성과를 기록했습니다.



### May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels (https://arxiv.org/abs/2408.14284)
Comments:
          25 pages, 5 figures. Accepted at the The 35th British Machine Vision Conference 2024 (BMVC 2024), Glasgow, UK

- **What's New**: 본 연구에서는 노이즈가 있는 레이블을 가진 데이터를 처리하기 위한 새로운 접근법인 Alternate Experience Replay (AER)를 소개합니다. 이 방법은 메모리 버퍼에서 클린(깨끗한), 복잡한, 노이즈가 있는 샘플을 명확하게 구분하기 위해 망각(Forgetting)의 특성을 활용합니다.

- **Technical Details**: AER는 클린 및 노이즈 샘플 간의 구분을 강화하기 위해 메모리 버퍼 학습 및 망각 단계를 번갈아 진행하는 새로운 Continual Learning (CL) 최적화 방안을 제시합니다. 이와 함께 Asymmetric Balanced Sampling (ABS)라는 새로운 샘플 선택 전략을 도입하여 현재 작업의 순수성을 유지하면서 과거의 중요한 샘플을 보존합니다.

- **Performance Highlights**: 다양한 노이즈 유형 및 속도에 대한 실험을 통해 AER와 ABS의 효과를 검증했으며, 기존의 손실 기반 정제 전략에 비해 평균 4.71% 높은 정확도를 기록하는 등 성능 향상을 입증했습니다.



### Uncertainties of Latent Representations in Computer Vision (https://arxiv.org/abs/2408.14281)
Comments:
          Doctoral thesis

- **What's New**: 이번 논문은 사전 훈련된 컴퓨터 비전 모델의 잠재적 표현 벡터에 불확실성 추정치를 통합하여 신뢰할 수 있는 머신러닝의 중요한 기초인 불확실성 정량화를 쉽게 접근할 수 있도록 합니다.

- **Technical Details**: 논문에서는 Monte-Carlo InfoNCE (MCInfoNCE)와 같은 확률 및 의사결정 이론에 기반한 새로운 접근 방식을 제안하며, 이론적 및 실증적 질문들을 탐구합니다. 또한, 비가시적인 잠재 표현에 대한 불확실성이 실제로 입증 가능하다는 주장을 제공합니다.

- **Performance Highlights**: 최종적으로, 경량화된 표현 불확실성을 대규모 컴퓨터 비전 모델 플랫폼에서 미리 훈련하여 보지 못한 데이터셋에도 제로샷(zero-shot) 방식으로 전이할 수 있는 능력을 갖춘 결과들을 제시하며, 이는 앞으로의 연구자들이 불확실성 정량화에 더 쉽게 접근할 수 있도록 합니다.



### Reliable Multi-modal Medical Image-to-image Translation Independent of Pixel-wise Aligned Data (https://arxiv.org/abs/2408.14270)
Comments:
          This paper has been accepted as a research article by Medical Physics

- **What's New**: 이번 연구에서는 픽셀 정렬 데이터에 독립적인 다중 모달 의료 영상 간의 변환 모델인 MITIA를 개발하였습니다. 이는 정렬되지 않은 훈련 데이터를 사용하더라도 신뢰할 수 있는 결과를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: MITIA 모델은 다중 모달 의료 이미지 등록 모듈과 변형 오류 탐지 모듈로 구성된 사전 추출 네트워크를 활용하여 미정렬 데이터에서 픽셀 수준의 사전 정보를 최대한 추출합니다. 추출된 사전 정보는 규제 항(re regularization term)으로 구성되어 GAN 모델의 최적화를 제어하며, 이는 해결 공간(solution space)을 제한하여 성능과 신뢰성을 높입니다.

- **Performance Highlights**: 밀접하게 정렬된 데이터와 미정렬 데이터 모두에서 MITIA는 다른 최신 이미지 변환 방법들보다 우수한 성능을 보여주었습니다. 정량적 분석과 정성적 시각 검사를 통해 성능 향상이 입증되었습니다.



### 1-Bit FQT: Pushing the Limit of Fully Quantized Training to 1-b (https://arxiv.org/abs/2408.14267)
- **What's New**: 이번 연구에서는 Fully Quantized Training (FQT)의 궁극적 한계인 1-bit FQT를 탐구하였습니다. 이 접근 방식은 활성화(activation), 가중치(weights), 그리고 그래디언트(gradient)를 1비트로 양자화하여 훈련 속도를 크게 향상시킵니다.

- **Technical Details**: 이 연구에서는 FQT의 수렴(convergence)에 미치는 그래디언트 분산(variance)의 영향을 분석하고, Activation Gradient Pruning (AGP)이라는 전략과 Sample Channel joint Quantization (SCQ) 기법을 도입합니다. AGP는 덜 정보적인 그래디언트를 제거하고 남은 그래디언트의 수치 정밀도를 향상시킵니다. SCQ는 가중치와 활성화 그래디언트를 다양한 양자화 전략을 사용하여 처리합니다.

- **Performance Highlights**: VGGNet-16과 ResNet-18 모델을 여러 데이터셋에서 미세 조정(fine-tuning)한 결과, 평균적으로 정확도(accuracy)가 약 6% 향상되었고, 훈련 속도는 풀 프리시전(full precision) 훈련에 비해 최대 5.13배 개선되었습니다. 또한, 1-bit FQT 알고리즘은 다양한 데이터셋에서 우수한 성능을 보였습니다.



### SONICS: Synthetic Or Not -- Identifying Counterfeit Songs (https://arxiv.org/abs/2408.14080)
- **What's New**: 본 논문은 AI로 생성된 노래의 탐지를 위한 SONICS라는 새로운 데이터셋을 소개하며, 이는 97,000개 이상의 노래와 49,000개 이상의 합성(song generated by AI)에 대한 정보를 담고 있다. 또한, AI 생성 노래의 모든 구성 요소가 AI로 만들어질 수 있는 가능성을 탐구한다.

- **Technical Details**: SONICS 데이터셋은 노래의 긴 맥락적 관계를 모델링하는 것이 중요하다는 점을 강조하며, 모든 구성 요소가 AI로 생성된 끝에서 끝까지(end-to-end) 인공 노래 탐지를 위한 데이터셋을 제공한다. 제안된 SpecTTTra 모델은 길고 긴 맥락을 효과적으로 처리할 수 있으며, 기존 CNN 및 Transformer 모델에 비해 3배 빠르고 6배 더 메모리 효율적이다.

- **Performance Highlights**: 제안된 SONICS 데이터셋과 SpecTTTra 모델은 인공 노래 탐지 작업에서 더욱 정확하고 효율적인 성능을 제공하며, 기존 모델과의 비교를 통해 그 유용성을 강조하고 있다.



### Collaborative Perception in Multi-Robot Systems: Case Studies in Household Cleaning and Warehouse Operations (https://arxiv.org/abs/2408.14039)
- **What's New**: 이번 논문은 다수의 로봇과 환경 내 센서가 데이터를 공유하고 통합하여 주변 환경을 종합적으로 표현하는 협업 인식(Collaborative Perception, CP) 패러다임을 탐구합니다. 논문에서는 가정 청소와 창고 환경에서 자율 이동 로봇을 위한 두 가지 사례 연구를 제시하여 CP의 이점을 강조합니다.

- **Technical Details**: 협업 인식 프레임워크는 여러 로봇 간의 협업을 가능하게 하며, 이를 위해 로봇에 장착된 비전 센서(예: 카메라, Lidar)와 동일한 환경에 설치된 센서가 통합되어 실시간 데이터를 중앙 서버로 전송합니다. 서버는 수집된 데이터를 통해 환경을 이해하고 로봇의 작업을 최적화하는 결정을 내립니다. 이러한 방식은 기존의 독립적 인식(Standalone Perception, SP)보다 작업 완료 시간과 에너지 효율성을 최적화하는 데 기여합니다.

- **Performance Highlights**: 첫 번째 사례 연구에서는 가정 청소 로봇 팀의 작업 효율을 보여주며, 두 번째 사례 연구에서는 창고에서의 AMR 성능을 비교 분석하여 협업 인식이 로봇의 협응 능력 및 전체 시스템 성능을 향상시키는 데 효과적임을 입증합니다. 향후 연구는 CP 프레임워크의 최적화와 실증 테스트를 통해 성능을 검증하는 데 초점을 맞출 예정입니다.



### FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry (https://arxiv.org/abs/2408.14035)
Comments:
          30 pages, 31 figures, due to the limitation that 'The abstract field cannot exceed 1,920 characters', the abstract presented here is shorter than the one in the PDF file

- **What's New**: 본 논문은 FAST-LIVO2를 제안하여 LiDAR-관성-비주얼 오도메트리(odometry)에 대해 신속하고 직접적인 프레임워크를 제공합니다. 이 시스템은 SLAM 작업에서 정확하고 견고한 상태 추정을 달성하며, 실시간 로봇 애플리케이션에 큰 잠재력을 보여줍니다.

- **Technical Details**: FAST-LIVO2는 IMU, LiDAR 및 이미지 측정을 효율적으로 융합합니다. 이 시스템은 Kalman 필터의 연속 업데이트 전략을 사용하여 이질적인 LiDAR와 이미지 측정 간의 치수 불일치를 해결합니다. LiDAR 모듈은 원시 점을 등록하고, 비주얼 모듈은 직접적인 사진 기하학적 오류를 최소화하여 강화된 효율성을 제공합니다.

- **Performance Highlights**: FAST-LIVO2는 UAV 온보드 내비게이션과 같은 실제 적용 사례에서 계산 효율성을 입증하고, 공중 매핑에서의 정확성, 3D 모델 렌더링 작업에도 적합한 복잡한 밀도 맵을 생성하는 데 성공했습니다.



### Histology Virtual Staining with Mask-Guided Adversarial Transfer Learning for Tertiary Lymphoid Structure Detection (https://arxiv.org/abs/2408.13978)
Comments:
          8 pages, 8 figures

- **What's New**: 본 연구에서는 Hematoxylin-Eosin (H&E) 염색 슬라이드를 활용하여 Tertiary Lymphoid Structure (TLS)의 감지를 개선하는 혁신적인 Mask-Guided Adversarial Transfer Learning 방식과 Virtual IHC Pathology Analysis Network (VIPA-Net)를 제안합니다.

- **Technical Details**: VIPA-Net은 Mask-Guided Transfer Module과 H&E 기반 Virtual Staining TLS Detection Module로 구성되어 있으며, H&E 슬라이드를 입력으로 하여 목표 IHC 패치를 합성합니다. 이 방법은 무당식(unsupervised) 조직 정보에 의해 적절한 IHC 염색 이미지를 생성하며, 병리적 염색의 다중 Otsu 임계값 방법을 사용하여 섬세한 색상 변화를 포착합니다.

- **Performance Highlights**: 실험 결과, VIPA-Net은 The Cancer Genome Atlas (TCGA) 데이터 세트에서 TLS의 감지 정확도를 크게 향상시키는 것으로 나타났습니다. 이는 실제 CD20 염색의 필요성을 효과적으로 피할 수 있는 방법을 보여줍니다.



### Personalized Topology-Informed 12-Lead ECG Electrode Localization from Incomplete Cardiac MRIs for Efficient Cardiac Digital Twins (https://arxiv.org/abs/2408.13945)
Comments:
          12 pages

- **What's New**: 이번 연구에서는 2D 심장 MRI를 활용하여 개인화된 ECG 전극 위치를 완전히 자동으로 추출할 수 있는 혁신적인 접근 방식을 제안합니다. 기존의 수동적이고 반자동적인 방법들과는 달리, 이 연구는 테포로지(topology) 정보를 통합하여 전극 위치를 정확하고 효율적으로 식별할 수 있는 새로운 모델을 개발했습니다.

- **Technical Details**: 이 방법은 심장 MRI로부터 얻은 희소한 신체 윤곽선을 기반으로 전극을 지역화합니다. 전극은 3D 신체 테포로지와 일치하도록 명시적으로 정렬되는 키포인트(keypoint) 집합으로 통합됩니다. 또한, 심장 MRI에서 직접 전극 위치를 추론하기 위한 전체 자동화된 딥러닝 프레임워크를 설정하여 불완전한 정보로부터 빠르고 정확한 전극 위치 추정을 목적으로 하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 전통적인 방법에 비해 정확도(Euclidean distance: $1.24 \pm 0.293$ cm vs. $1.48 \pm 0.362$ cm)와 효율성($2$~s vs. $30$-$35$~min) 모두에서 우수한 성능을 보여주었습니다. 이 전극 위치 탐지 결과는 개인화된 CDT 모델 생성을 위한 효과적인 경량화 모델링의 잠재력을 강조합니다.



### Knowledge-Aware Reasoning over Multimodal Semi-structured Tables (https://arxiv.org/abs/2408.13860)
- **What's New**: 본 연구에서는 다중 모드(Dimodal) 데이터를 다루기 위한 새로운 데이터셋, MMTabQA를 소개합니다. 이 데이터셋은 텍스트와 이미지를 통합하여 지식 기반의 추론을 수행할 수 있는 AI 모델의 성능을 평가하기 위해 설계되었습니다.

- **Technical Details**: MMTabQA 데이터셋은 기존의 Wikipedia 데이터를 활용하여 이미지가 포함된 다중 모드 테이블 질문 응답 환경을 위해 생성되었습니다. 질문은 명시적, 답변 언급, 암시적 세 가지 유형으로 분류되며, 기존의 Wikipedia 질문 응답 데이터셋을 변환하여 구성되었습니다. 다양한 상태-오브-더-아트 Vision-Language 모델을 평가하여 복잡한 모드 테이블 처리의 어려움을 분석하였습니다.

- **Performance Highlights**: 현재 AI 모델들은 다중 모드 테이블에서의 지식 기반 추론 및 이미지, 텍스트 통합에서 상당한 도전에 직면하고 있습니다. 모델들은 엔티티 식별 오류, 시각적 이해의 어려움 및 테이블 구조 이해에 어려움을 겪고 있습니다. MMTabQA 데이터셋은 이러한 문제를 해결하기 위한 강력한 기준점을 제공합니다.



### A Low-dose CT Reconstruction Network Based on TV-regularized OSEM Algorithm (https://arxiv.org/abs/2408.13832)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문에서는 LDCT(저선량 전산화 단층 촬영) 이미징에서 TV(총 변화) 정규화를 EM(기대 최대화) 알고리즘의 ``M'' 단계에 통합하여 효과적이고 효율적인 정규화를 달성합니다. 이를 통해 재구성 품질을 향상시키고 과도한 블러 또는 세부 손실을 방지할 수 있습니다.

- **Technical Details**: 제안된 방식인 OSEM-CP 알고리즘은 CP(샴볼-폭크) 알고리즘과 OS(주문된 서브셋) 전략을 활용하여 뷰별 뷰에 대한 재구성과 정규화를 실시합니다. 이를 통해 계산 부하를 크게 줄이고 재구성 속도를 높을 수 있습니다. 또한, OSEM-CPNN이라는 종단 간 재구성 신경망을 제안하여 여러 번의 반복 없이 한 번의 전체 뷰 반복으로도 높은 품질의 재구성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, OSEM-CP는 전통적인 방법들과 비교해 우수한 성능을 보였으며, OSEM-CPNN은 최신 감독 신경망 방법들을 초월하는 뛰어난 결과를 보여주었습니다.



### HER2 and FISH Status Prediction in Breast Biopsy H&E-Stained Images Using Deep Learning (https://arxiv.org/abs/2408.13818)
- **What's New**: 이 연구는 저렴하고 광범위하게 사용되는 H&E 염색 이미지를 활용하여 유방암 환자의 HER2 상태를 예측하는 새로운 방법을 제안합니다. 저자들은 MoCo-v2 대비 학습과 맞춤형 약한 감독 분류 기법을 결합하여, HER2 양성 및 음성 종양을 성공적으로 구별했습니다.

- **Technical Details**: 기술적으로, 연구팀은 전체 슬라이드 이미지(Whole Slide Images, WSI)에서 패치를 추출하고, MoCo-v2로 사전 학습된 ResNet50 인코더를 사용하여 최종 주의(attention) 모듈을 훈련했습니다. 훈련 데이터는 TCGA의 182개 H&E WSI를 이용했으며, 검증 단계에서 모델의 AUC는 0.85에 달했습니다.

- **Performance Highlights**: 모델은 TCGA-BRCA 데이터셋에 있는 44개의 H&E 슬라이드에 대해서도 테스트되었으며, 이 슬라이드는 HER2 점수가 2+로 평가되었습니다. 이 슬라이드에서 모델은 0.81의 AUC를 달성했습니다. 이러한 결과는 FISH 검사가 필요한 대안이 될 수 있는 가능성을 나타내며, 치료 접근성을 높이는 데 기여할 수 있습니다.



### BCDNet: A Convolutional Neural Network For Breast Cancer Detection (https://arxiv.org/abs/2408.13800)
Comments:
          5 pages, 5 figures

- **What's New**: 이 논문은 유방암 진단을 위한 새로운 CNN 모델, BCDNet을 제안합니다. BCDNet은 진행성 관암(Invasive Ductal Carcinoma, IDC)을 효율적으로 탐지하며, 89.5%의 정확도를 기록합니다. 또한, 모델의 학습 시간을 단축시키는 데 성공했습니다.

- **Technical Details**: BCDNet은 일련의 convolutional layers, pooling layers, activation layers, fully-connected layers, Batch Normalization layers, 그리고 Dropout layers로 구성됩니다. 이 구조는 IDC를 효율적이고 정확하게 감지하기 위해 설계되었습니다. CNN의 핵심 원리는 가중치 행렬인 kernel을 사용하여 입력 데이터에서 특징을 추출하는 것입니다. 비선형성은 주로 Rectified Linear Unit (ReLU) 활성화 함수에 의해 구현되며, 이는 학습을 가속화하고 복잡한 패턴을 더욱 효과적으로 학습할 수 있도록 돕습니다.

- **Performance Highlights**: BCDNet은 ResNet 50 및 ViT-B-16과 비교할 때 우수한 성능을 보였으며, 덕분에 의료 환경에서 전통적인 진단 방법을 보완하여 활용할 수 있는 잠재력을 가지고 있습니다. BCDNet은 특히 데이터셋의 변화에 신속하게 적응할 수 있는 능력을 보여줍니다.



### Batch-FPM: Random batch-update multi-parameter physical Fourier ptychography neural network (https://arxiv.org/abs/2408.13782)
- **What's New**: 본 연구에서는 빠르고 강력한 Fourier Ptychographic Microscopy (FPM) 재구성 방법을 제안합니다. 이 방법은 물리적 신경망(Physical Neural Networks) 기반으로, 배치 업데이트 확률적 경량 하강( stochastic gradient descent, SGD) 최적화 전략을 사용합니다.

- **Technical Details**: 제안하는 방법은 무작위 배치 최적화 접근 방식을 활용하여 고주파 정보에 더 집중하며, 고정된 순차적 반복 순서에서 벗어납니다. 낮은 신호 대 잡음 비율(signal-to-noise ratio) 데이터 세트에서도 우수한 수렴 성능을 발휘하며, 여러 시스템 매개변수를 동시에 보정할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 소비자 등급의 GPU에서 1024 x 1024 픽셀의 관심 영역에 대해 거의 실시간 디지털 리포커싱을 달성하였으며, 이는 임상 진단, 디지털 병리학 및 생의학 연구 등에서 FPM의 실제 적용을 효율적으로 촉진할 수 있습니다.



### Anatomical Consistency Distillation and Inconsistency Synthesis for Brain Tumor Segmentation with Missing Modalities (https://arxiv.org/abs/2408.13733)
Comments:
          Accepted Paper to European Conference on Artificial Intelligence (ECAI 2024)

- **What's New**: 다중 모드 (Multi-modal) 자성 공명 영상 (MRI)을 통한 정확한 뇌 종양 분할을 위한 새로운 프레임워크인 ACDIS를 소개합니다. ACDIS는 다중 모드에서 단일 모드로 해부학적 구조를 전이하고 모드 특정 특징을 합성하는 기술입니다.

- **Technical Details**: ACDIS는 두 가지 주요 구성 요소로 구성되어 있습니다: 해부학적 일치 증류 (Anatomical Consistency Distillation, ACD)와 모드 특징 합성 블록 (Modality Feature Synthesis Block, MFSB)입니다. ACD는 해부학적 특징 향상 블록 (Anatomical Feature Enhancement Block, AFEB)을 통해 해부학적 정보를 정교하게 추출하고, 일관성 있는 지식 전이 (Anatomical Consistency ConsTraints, ACCT)를 실현하여 구조적 특징의 정밀한 정렬을 보장합니다.

- **Performance Highlights**: BraTS2018과 BraTS2020 데이터셋에서 검증을 통해 ACDIS가 누락된 모드에서 뇌 종양 분할의 가장 앞선 방법론들보다 일관되게 우수한 성능을 나타냈습니다.



### A prototype-based model for set classification (https://arxiv.org/abs/2408.13720)
- **What's New**: 본 논문에서는 선형 부분공간(linear subspaces)으로 구성된 매니폴드(manifold)인 Grassmann 매니폴드에서 학습하기 위한 프로토타입 기반 접근법을 제안합니다. 이 방법은 클래스를 대표하는 하위공간 프로토타입을 학습하고, 하위공간의 차원 선택 과정을 자동화하는 relevance factor를 포함하여 투명한 분류 모델을 형성합니다.

- **Technical Details**: 제안된 모델은 Generalized Learning Vector Quantization (GLVQ)의 확장으로, 새로운 적응 거리 측정 방식을 도입하여 성능과 해석 가능성을 개선합니다. 입력 벡터의 결정에 대한 기여를 명확히 할 수 있도록 설계되었습니다. 이는 Grassmann 매니폴드에서의 고차원성의 부정적인 영향을 다루는 변형을 포함합니다.

- **Performance Highlights**: 벤치마크 이미지 및 텍스트 데이터셋에 대한 실험을 통해 제안된 분류기의 효율성을 입증하였으며, transformer 기반 모델 대비 성능, 해석 가능성, 계산 자원 요구사항 측면에서 우수함을 보여주었습니다.



### FreqINR: Frequency Consistency for Implicit Neural Representation with Adaptive DCT Frequency Loss (https://arxiv.org/abs/2408.13716)
Comments:
          9 pages, 7 figures

- **What's New**: 최근 연구에서는 Implicit Neural Representation (INR)의 발전으로 다양한 해상도의 이미지 처리에서 뛰어난 성능이 입증되었습니다. 그러나 고해상도(High-Resolution, HR)와 실제 이미지 간의 주파수 불일치로 인해 발생하는 아티팩트와 블러링 문제를 해결하기 위해 Frequency Consistency for Implicit Neural Representation (FreqINR)이라는 혁신적인 Arbitrary-scale Super-resolution 방법을 소개합니다.

- **Technical Details**: FreqINR은 Adaptive Discrete Cosine Transform Frequency Loss (ADFL)를 활용하여 고해상도(High-Resolution, HR) 이미지와 실제 이미지 간의 주파수 간극을 최소화하는 동시에, 인퍼런스(inference) 시 저해상도(Low-Resolution, LR)와 실제 이미지 사이의 주파수 일관성을 보존하기 위해 수용 범위를 확장합니다. 훈련 과정에서 DCT(Discrete Cosine Transform)를 활용하여 주파수 도메인으로 변환하고, Frequency Distance Matrix (FDM)와 Adaptive Frequency Weighting Matrix (AFWM)를 통해 주파수 가중치를 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과에 따르면 FreqINR은 기존의 Arbitrary-scale Super-resolution 방법에 비해 최첨단 성능을 달성하였으며, 계산 효율성에서도 주목할 만한 개선을 보여주었습니다. FreqINR은 경량 솔루션으로, 고해상도 이미지의 품질을 개선하며 진정한 고주파 세부정보를 생성하는 데 기여합니다.



### Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic (https://arxiv.org/abs/2408.13656)
- **What's New**: 본 논문에서는 여러 개의 세밀화(finetuned) 모델의 강점을 결합하여 특화된 능력을 보존하는 통합 모델을 만드는 새로운 접근 방식인 Localize-and-Stitch를 제안합니다. 기존의 글로벌 모델 병합 방법 대신, 국소적인 방법으로 세밀화된 모델의 중요한 기술을 포함하는 지역을 식별하고 이를 다시 통합하는 방식으로 성능 저하를 해결합니다.

- **Technical Details**: Localize-and-Stitch 알고리즘은 2단계로 구성됩니다: 1) Localization: 세밀화된 모델의 중요한 기술이 포함된 작은 지역(전체 매개변수의 1%)을 식별하고, 2) Stitching: 이 중요한 지역을 사전 훈련(pretrained) 모델에 통합합니다. 실험을 통해 총 매개변수의 1%만 변경해도 세밀화된 성능의 99% 이상을 회복할 수 있음을 확인했습니다.

- **Performance Highlights**: 이 방법은 다양한 비전 및 언어 벤치마크에서 평가되었으며, 기존의 모델 병합 방법보다 우수한 성능을 보였습니다. 추가적으로 모델 압축을 용이하게 하고 사전 훈련된 지식을 보존하여, 여러 세밀화된 모델에서의 유연한 기술 조합을 최소한의 저장공간과 계산 자원으로 가능하게 합니다.



### FLEURS-ASL: Including American Sign Language in Massively Multilingual Multitask Evaluation (https://arxiv.org/abs/2408.13585)
Comments:
          Access FLEURS-ASL at this https URL. arXiv admin note: text overlap with arXiv:2408.07065

- **What's New**: 본 논문에서는 기계 번역 연구 분야에 있어서 중요한 발전을 의미하는 FLEURS-ASL 벤치마크의 도입을 발표합니다. 이 벤치마크는 미국 수화(American Sign Language, ASL)와 200개 이상의 언어 간의 번역을 지원하며, 5명의 인증받은 청각 장애인 통역사가 번역을 제공합니다.

- **Technical Details**: FLEURS-ASL은 ASL과 텍스트 간의 문장 및 담화 수준의 번역을 평가할 수 있는 도구를 제공합니다. 34초의 문맥 윈도우(context window)에서 타임스탬프 토큰(timestamp tokens)과 이전 텍스트 토큰을 포함하는 통합 모델링 접근 방식을 바탕으로 훈련되었습니다. 이 모델은 문장 수준 평가의 성과를 초과하며, 새로운 과제를 지원하는 능력을 보여줍니다.

- **Performance Highlights**: FLEURS-ASL에서의 문장 수준 번역에 대한 인간 기초 성능은 13.0 BLEU, 64.6 BLEURT로 측정되었습니다. 제안된 모델은 문장 수준 기초 성능에서 3.7 BLEU를 달성하여 이전의 모델을 능가했습니다. 또한 최근 멀티모달 모델들이 ASL에 대한 이해도가 거의 없다는 것을 시사하며, 수화가 표준 평가 세트에 포함되어야 할 필요성을 강조합니다.



### Topological GCN for Improving Detection of Hip Landmarks from B-Mode Ultrasound Images (https://arxiv.org/abs/2408.13495)
- **What's New**: 이 논문에서는 유아의 고관절 발달 이형성증(DDH) 진단을 위한 B-mode 초음파 기반의 컴퓨터 지원 진단(CAD) 기술이 새로운 모델인 TGCN-ICF를 통합하여 고관절 랜드마크(landmark) 탐지 성능을 향상시킬 수 있음을 제안하고 있습니다.

- **Technical Details**: TGCN-ICF는 Improved Conformer (ICF) 서브네트워크와 Topological Graph Convolutional Network (TGCN) 서브네트워크 두 개로 구성되어 있습니다. ICF 서브네트워크는 히트맵(heatmap)을 생성하고, TGCN 서브네트워크는 랜드마크 탐지를 추가적으로 개선합니다. 또한, Mutual Modulation Fusion (MMF) 모듈을 통해 U-Net과 Transformer 지점에서 추출된 기능(feature)을 깊이 있게 교환하고 융합합니다.

- **Performance Highlights**: 실제 DDH 데이터셋에 대한 실험 결과, 제안된 TGCN-ICF 모델은 비교된 모든 알고리즘보다 우수한 성능을 보입니다.



### Optimal Layer Selection for Latent Data Augmentation (https://arxiv.org/abs/2408.13426)
- **What's New**: 이 연구에서는 데이터 증강(data augmentation, DA) 기법의 효과를 숨겨진 층(hidden layers)에 적용하는 방법에 대해 조사했습니다. 새로운 방법인 AdaLASE(AdaPtive LAyer SElection)를 제안하여, 각 층의 DA 비율을 훈련 중에 동적으로 조정하였습니다.

- **Technical Details**: AdaLASE 방법은 훈련 진행 중에 각 층에 대한 DA 적용 비율을 업데이트하는 방식으로, gradient descent 방식으로 최적의 층을 찾습니다. 이 방법은 입력 데이터에 DA를 적용하는 기존 방법들과 비교하여, 동적이고 효율적인 레이어 선택을 목표로 합니다.

- **Performance Highlights**: 여러 이미지 분류 데이터셋에서 AdaLASE 방법을 적용한 결과, DA를 입력 데이터에서 또는 임의로 선택된 층에 적용하는 것보다 더 높은 전체 테스트 정확도를 기록하였습니다. AdaLASE는 훈련 중 DA에 적합한 층의 비율을 효과적으로 조정했습니다.



### ReCon: Reconfiguring Analog Rydberg Atom Quantum Computers for Quantum Generative Adversarial Networks (https://arxiv.org/abs/2408.13389)
Comments:
          ReCon will appear in the Proceedings of the International Conference on Computer-Aided Design (ICCAD), 2024

- **What's New**: 이 연구는 아날로그 리드버그 원자 양자 컴퓨터에서 최초로 양자 생성적 적대 신경망(Quantum GAN)을 구현한 방법을 제안합니다. 이 기술은 기존의 초전도 큐비트 기술을 넘어 양자 GAN의 가능성을 열어줍니다.

- **Technical Details**: ReCon은 적대적인 기계 학습 모델을 사용하는 양자 GAN의 첫 번째 구현으로, 4개의 큐비트만을 사용하여 이미지 생성을 위해 차원 축소를 적용합니다. 입력 이미지 데이터셋은 주성분 분석(Principal Component Analysis, PCA)을 통해 처리되며, 큐비트 배치와 주파수 프로파일을 조정하여 시스템의 해밀토니안 진화를 제어합니다. 또한, ReCon은 데이터 품질을 개선하기 위해 다층 파라미터 훈련을 실시합니다.

- **Performance Highlights**: ReCon은 MNIST 및 패션-MNIST 데이터셋을 사용해 MoaiQ 기술보다 33% 더 우수한 이미지 품질을 보여줍니다. 또한 QuEra Aquila 컴퓨터에서 실행되어, 실제 노이즈 환경에서도 높은 품질의 이미지를 생성함을 입증하였습니다.



### A systematic review: Deep learning-based methods for pneumonia region detection (https://arxiv.org/abs/2408.13315)
Comments:
          8 pages, 1 figure, published on Applied and Computational Engineering

- **What's New**: 이 논문은 폐렴 질병의 진단 과정에서 컴퓨터 지원 폐렴 검출 방법의 발전을 다룹니다. 특히 최근 10년 동안의 딥러닝(deep learning) 접근 방식이 전통적인 머신러닝 머신(Traditional Machine Learning) 방법보다 더욱 효과적이었다고 밝힙니다.

- **Technical Details**: 논문에서는 다양한 딥러닝 네트워크 구조와 데이터셋(dataset), 데이터 처리 기법(data processing techniques), 일반적인 워크플로우(general workflow)와 그 결과를 포함한 중요한 연구 측면을 분석합니다. 또한 현재의 도전 과제와 향후 연구에서 고려해야 할 사항들을 제안합니다.

- **Performance Highlights**: 딥러닝 방법의 폐렴 탐지에 있어 효율성과 정확성을 강화하면서, 감염된 영역을 탐지, 분류, 로컬리제이션(localization)할 때의 연구 절차를 향상시키기 위한 개선 방향을 모색합니다.



### Multi-modal Intermediate Feature Interaction AutoEncoder for Overall Survival Prediction of Esophageal Squamous Cell Cancer (https://arxiv.org/abs/2408.13290)
Comments:
          Accepted by ISBI 2024

- **What's New**: 이 논문에서는 식도 편평세포암(Esophageal Squamous Cell Cancer, ESCC)의 전체 생존율 예측을 위한 새로운 오토인코더 기반 심층 학습 모델(Multi-modal Intermediate Feature Interaction AutoEncoder, MIFI-AE)을 제안합니다. 기존의 연구에서는 서로 다른 모달리티 간의 생존 예측에 중요한 특징을 제대로 다루지 않았습니다. 또한, 여러 모달의 특징 표현 간의 의미 차이를 고려하지 않았던 점을 보완합니다.

- **Technical Details**: 제안된 MIFI-AE 모델은 두 가지 새로운 모듈인 Cross-modal Multi-step Intermediate Fusion Module (CMIFM)과 Multi-scale Feature map Fusion-Separation Module (MFFSM)을 포함하여 다중 모달 특징의 상호작용을 개선하고 특징 추출 능력을 강화합니다. MFFSM은 인코더의 중간층에서 얻은 다중 스케일 피쳐 맵을 결합하여 인코딩-디코딩 능력을 높이며, CMIFM은 교차 모달 특징을 반복적으로 상호작용하여 최종 위험 점수를 생성합니다.

- **Performance Highlights**: 제안된 모델은 분류 능력, 위험 계층화, 그리고 모듈의 효과성에 대한 비교 및 배제 실험을 통해 만족스러운 결과를 달성하였습니다. 특히, 생존 예측 관련에서 다중 모달 특징 표현의 정렬을 위한 새로운 조인트 손실(Multi-task Joint Loss, MJ-Loss)을 도입하여 성능을 개선했습니다.



### Robust Image Classification: Defensive Strategies against FGSM and PGD Adversarial Attacks (https://arxiv.org/abs/2408.13274)
Comments:
          This is the preprint of the paper that has been accepted for oral presentation and publication in the Proceedings of the IEEE Asian Conference on Intelligent Technologies (ACOIT'2014). The conference will be organized in Kolar, Karnataka, INDIA from September 6 to 7, 2024. The paper is 8 pages long, and it contains 9 Figures and 4 Tables. This is NOT the final version of the paper

- **What's New**: 이 연구는 Deep Learning 모델의 이미지 분류에서 FGSM(Fast Gradient Sign Method) 및 PGD(Projected Gradient Descent)에 대한 방어 메커니즘을 탐구하고 개선합니다.

- **Technical Details**: 입력 데이터의 사전 처리(preprocessing) 기술과 적대적 학습(adversarial training)을 결합하여 적대적 변동(adversarial perturbations)의 영향을 완화하는 방법론을 제시합니다. 다양한 모델 아키텍처(model architectures)와 훈련 전략(training strategies)을 조사하였습니다.

- **Performance Highlights**: 엄격한 평가를 통해, 제안된 방어 전략이 FGSM 및 PGD 공격에 대해 모델의 강인성(model robustness)을 크게 개선함을 보여주었습니다. 이는 실제 응용에서 방어 전략의 가능성을 강조합니다.



### Pediatric TSC-Related Epilepsy Classification from Clinical MR Images Using Quantum Neural Network (https://arxiv.org/abs/2408.12615)
Comments:
          5 pages,4 figures,2 tables,presented at ISBI 2024

- **What's New**: 이번 연구에서는 소아 환자의 결절성 경화증(TSC)을 위한 강력한 분류 모델이 필요함에 따라, 기존의 Convolutional Neural Networks(CNN)와 Quantum Neural Networks(QNN)를 결합한 새로운 딥러닝 모델 QResNet을 소개합니다.

- **Technical Details**: QResNet은 두 개의 양자층(quantum layer, QL)을 통합하여 고전 데이터를 양자 프레임워크 내에서 처리하도록 설계되었습니다. 이 모델은 ZZFeatureMap과 Ansatz 층으로 구성되어 있으며, 기존의 3D-ResNet 모델보다 TSC의 MRI 이미지 분류에서 더 우수한 성능을 보였습니다.

- **Performance Highlights**: QResNet의 결과는 정확도와 곡선 아래 면적(Area Under Curve, AUC) 지표에서 전통적인 CNN을 초과하며, 향후 연구는 실제 의료 이미징 상황에서 양자 알고리즘의 확장성 및 실용적 구현에 초점을 맞출 수 있습니다.



### UMERegRobust - Universal Manifold Embedding Compatible Features for Robust Point Cloud Registration (https://arxiv.org/abs/2408.12380)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 Universal Manifold Embedding (UME) 프레임워크를 사용하여 강체 변환을 추정하고, 부분적으로 겹친 및 서로 다르게 샘플링된 점 구름(point clouds) 시나리오를 수용할 수 있도록 확장합니다. 이를 위해 UME 대조 손실(contrastive loss) 및 샘플링 균형기(Sampling Equalizer)를 도입하여 견고한 등록 파이프라인 UMERegRobust를 구성합니다.

- **Technical Details**: UME는 동일한 객체에 대한 관측값을 단일 저차원 선형 부분공간으로 매핑하는 방법론입니다. 이 방법론은 변환에 불변한 관측값 표현을 제공합니다. 우리는 UME 호환 색칠 방법(coloring method)과 UME 기반 추정기를 통해 변환을 관측값에 연결하는데 유용한 정보를 포함시키는 방식으로 등록 과정에서의 정확도와 견고성을 향상시킵니다.

- **Performance Highlights**: 제안한 방법인 UMERegRobust는 KITTI 벤치마크에서 (1°, 10cm)의 엄밀한 정확도 기준을 고려할 때, 평균 +9%의 향상을 보이며, RotKITTI 벤치마크에서는 최신의 SOTA 방법에 비해 +45%의 성능 향상을 기록하였습니다.



### UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images (https://arxiv.org/abs/2408.11545)
Comments:
          5 pages, 3 figures

- **What's New**: UNetMamba는 Mamba 기반의 UNet 유사 세미나틱 세그멘테이션 모델로, 높은 해상도의 원격 감지 이미지에서 정확도와 효율성 간의 딜레마를 극복하기 위해 설계되었습니다.

- **Technical Details**: UNetMamba는 ResT 백본을 사용하는 인코더와 복잡한 정보를 효율적으로 디코딩하는 mamba segmentation decoder (MSD), 로컬 정보 인식을 향상시키기 위한 local supervision module (LSM)으로 구성됩니다.

- **Performance Highlights**: UNetMamba는 LoveDA에서 mIoU가 0.87% 증가하고 ISPRS Vaihingen에서 0.36% 증가하며, 저비용의 경량 설계를 통해 높은 효율성을 달성합니다.



New uploads on arXiv(cs.AI)

### K-Sort Arena: Efficient and Reliable Benchmarking for Generative Models via K-wise Human Preferences (https://arxiv.org/abs/2408.14468)
Comments:
          Project page: this https URL

- **What's New**: K-Sort Arena는 효율적이고 신뢰할 수 있는 비쥬얼 생성 모델 평가 플랫폼으로, K-wise 비교를 통한 모델 간 경쟁을 가능하게 하여 이전의 Pairwise 비교보다 더 많은 정보를 제공한다.

- **Technical Details**: 이 플랫폼은 K개의 모델이 동시에 자유롭게 경쟁할 수 있도록 하며, 랜덤화된 매칭과 페어와이즈 비교의 단점을 해결합니다. 군집적인 사용자 피드백을 통해 모델 간 ranking을 개선하고, Bayesian 업데이트를 적용하여 모델 능력의 신뢰성을 높입니다.

- **Performance Highlights**: K-Sort Arena는 ELO 시스템에 비해 16.3배 더 빠른 수렴 속도를 보여주며, 적은 투표로도 정확한 leaderboard 업데이트가 가능합니다.



### CHARTOM: A Visual Theory-of-Mind Benchmark for Multimodal Large Language Models (https://arxiv.org/abs/2408.14419)
- **What's New**: CHARTOM은 다중 모달 대형 언어 모델을 위한 시각적 마음 이론 벤치마크로, 차트를 기반으로 FACT 질문과 MIND 질문을 통해 AI의 이해도를 평가합니다.

- **Technical Details**: CHARTOM 벤치마크는 112개의 차트로 구성되며, 기본적인 자료를 시각적으로 조작하여 실제 FACT 질문과 MIND 질문에 답변할 수 있도록 설계되었습니다. 각 차트는 바, 선, 파이, 산점도, 지도 차트 등 5가지 일반적인 유형으로 나뉘며, 조작된 버전은 심리학 문헌에 따른 시각적 오류를 포함합니다.

- **Performance Highlights**: AI는 FACT 질문에 정확히 답변할 수 있지만, MIND 질문의 경우 일반 독자가 차트에 어떻게 반응할지를 예측하는 것이 더 도전적임을 보여줍니다. 이는 AI가 시각적 자료의 진위와 인간의 인지를 고려하는 새로운 방법을 제시합니다.



### Uncovering Knowledge Gaps in Radiology Report Generation Models through Knowledge Graphs (https://arxiv.org/abs/2408.14397)
Comments:
          Code is available at: this https URL

- **What's New**: 최근 인공지능(AI) 기술의 발전으로 방사선 보고서의 자동 생성이 크게 개선되었습니다. 그러나 기존 평가 방법은 모델이 방사선 이미지를 이해하는 수준과 인간 수준의 세밀함(Granularity)에 도달하는 능력을 드러내지 못하고 있습니다. 이를 해결하기 위해, 우리는 ReXKG라는 시스템을 소개하며, 이 시스템은 처리된 보고서에서 구조화된 정보를 추출하여 종합적인 방사선 지식 그래프를 구축합니다.

- **Technical Details**: ReXKG는 처리된 보고서에서 정보를 추출하여 방사선 지식 그래프를 생성하는 시스템입니다. 우리는 다음과 같은 세 가지 평가 지표를 제안합니다: ReXKG-NSC(노드 유사성 계수), ReXKG-AMS(인접 행렬 유사성), ReXKG-SCS(서브그래프 커버리지 점수). 이 지표들은 다양한 지식 그래프 간의 노드 유사성, 엣지 분포, 서브그래프의 커버리지를 평가하는 데 사용됩니다.

- **Performance Highlights**: AI 모델들의 방사선 보고서 생성 성능에 대한 포괄적인 분석을 통해, 일반 모델(generalist)의 경우 80% 가까운 필수 엔티티 커버리지를 보여주지만, 의료 기기 세부사항에서는 방사선 전문의가 작성한 보고서에 미치지 못했습니다. AI 모델들은 개념을 과적합(overfit)하여 훈련 데이터에서 자주 등장하는 특정 개념에 집중하는 경향이 있으며, 결과적으로 세부사항이 부족하거나 비현실적(hallucinated)인 설명을 생성하였습니다. 일반 모델은 다양한 데이터에 노출되어 방사선 지식이 크게 향상되었습니다.



### Machine Learning for Quantifier Selection in cvc5 (https://arxiv.org/abs/2408.14338)
- **What's New**: 이번 연구에서는 1차 양화 문제에 대한 SMT(Satisfiability Modulo Theory) 해결 기법에서 머신러닝(Machine Learning)을 활용하여 양화자(quantifier) 선택을 효율적으로 가이던스하는 방법을 제시하며, 이는 최신 기법을 개선하는 내용을 담고 있습니다.

- **Technical Details**: 기존 cvc5 SMT 솔버에 우리의 방법을 통합하여, 훈련된 효율적인 머신러닝 모델을 통해 어떤 양화자가 인스턴스화(instantiated) 되어야 하고 어떤 것은 그러면 안 되는지를 결정합니다. 이 과정은 전체 해결 과정에서 ML 예측기를 여러 번 호출하여 양화자 선택을 필터링합니다. 또한, 그라디언트 부스팅 결정 트리(gradient boosting decision tree)를 기반으로 한 빠른 머신러닝 모델을 사용하며, 구현의 오버헤드가 최소화되어 있습니다. Mizar Mathematical Library에서 수집된 대규모 1차 문제 집합으로 훈련하게 됩니다.

- **Performance Highlights**: 우리의 최상의 머신러닝 전략은 최신 CASC 대회의 cvc5 포트폴리오보다 10% 이상 성능을 초과하며, MML에 대한 cvc5의 성능을 단일 전략 및 포트폴리오 시나리오 모두에서 20% 이상 향상시킵니다. 이 방식은 한 전략의 샘플로 훈련된 모델이 다른 모든 전략의 성능을 개선할 수 있는 크로스-전략 모델 전이(cross-strategy model transfer)를 특징으로 합니다.



### Fact Probability Vector Based Goal Recognition (https://arxiv.org/abs/2408.14224)
Comments:
          Will be presented at ECAI 2024

- **What's New**: 본 논문에서는 관측된 사실(observed facts)과 해당 사실의 기대 확률(expected probabilities)을 비교하는 새로운 목표 인식(goal recognition) 접근 방식을 제안합니다. 이 확률은 지정된 목표 g와 초기 상태 s0에 의존하며, 이러한 확률과 관측된 사실을 실수(vector space) 벡터 공간에 매핑하여 잠재적 목표에 대한 휴리스틱 값(heuristic values)을 계산합니다.

- **Technical Details**: 제안된 방법은 목표 g의 경우에 대한 사실 관측 확률을, 초기 상태 s0 및 현재 관측된 상태 st와 함께 실수 벡터 공간에 매핑하여 히스토그램을 사용해 각 가능한 목표에 대해 휴리스틱 값을 계산합니다. 이 방법은 FPV(Fact Probability Vector Based Goal Recognition)로 명명되며, 이전 기법들과 비교하여 더욱 효율적이며 낮은 관측 가능성(low observability) 상황에서도 우수한 목표 인식 정밀도를 보여줍니다.

- **Performance Highlights**: FPV는 기존의 최선단 기술(state-of-the-art techniques)에 비해 목표 인식 정밀도를 향상시키면서도 계산 복잡도를 줄이는 결과를 보였습니다. 특히 낮은 관측 가능성의 경우에서 더 나은 성능을 입증하였습니다.



### DynamicRouteGPT: A Real-Time Multi-Vehicle Dynamic Navigation Framework Based on Large Language Models (https://arxiv.org/abs/2408.14185)
Comments:
          This paper is 12 pages long and represents the initial draft, version 1

- **What's New**: 본 논문에서는 복잡한 교통 환경에서의 실시간 동적 경로 계획을 위한 새로운 접근 방식인 DynamicRouteGPT를 제안하고 있습니다. 이 시스템은 전통적인 경로 계획 방법의 한계를 극복하고, 전역 및 지역 최적화를 균형 있게 따릅니다.

- **Technical Details**: DynamicRouteGPT는 마르코프 체인(Markov chains), 베이지안 추론(Bayesian inference), 대규모 사전 학습된 언어 모델(Large-scale pretrained language models)인 Llama3 8B를 통합하여 실시간 차량 경로 결정을 지원합니다. 경로 선택은 정적 Dijkstra 알고리즘을 사용하여 전역 최적 경로를 설정한 후, 동적 교차로에서 실시간으로 결정됩니다.

- **Performance Highlights**: 실험 결과, DynamicRouteGPT는 다중 차량 제어 및 전체 시간 최적화에서 기존 최첨단 기술(SOTA) 방법들을 초월하는 효과를 보였습니다. 이 시스템은 교통 혼잡을 피하고 최적의 경로를 제공하며, 차량들이 목적지에 원활하게 도달할 수 있도록 돕습니다.



### Estimating Causal Effects from Learned Causal Networks (https://arxiv.org/abs/2408.14101)
- **What's New**: 이 논문에서는 관찰 데이터를 기반으로 인과 Bayesian 네트워크(Causal Bayesian Network)와 그 잠재 변수(confounding latent variables)를 직접 학습하여 인과 효과 질의를 해결하는 새로운 패러다임을 제안합니다. 기존에는 관찰 데이터에 대한 확률 표현을 생성하고 이를 평가하는 방식이 널리 사용되었습니다. 그러나 제안한 방법은 model completion 학습 방식을 통해 더욱 효과적일 수 있음을 보여줍니다.

- **Technical Details**: 우리는 Structural Causal Models (SCMs)와 probabilistic graphical models (PGMs)을 기반으로, Expectation-Maximization (EM) 알고리즘을 사용하여 관찰된 변수와 잠재 변수를 포함한 인과 Bayesian 네트워크를 학습합니다. 또한 Bayesian Information Criterion (BIC)을 이용한 모델 선택 기술을 통해 적절한 도메인 크기를 선정할 수 있습니다.

- **Performance Highlights**: 제안한 학습 접근법은 estimand 기반의 방법보다 더 정확한 추정치를 제공합니다. 특히, 높은 차원의 estimand 표현이 있지만 낮은 트리너드(treewidth) 인과 모델을 가진 경우, 제안한 방법은 인과 그래프의 정보를 더 잘 보유하면서 정확한 추정을 가능하게 합니다. 또한 여러 인과 질의를 동일한 모델에서 수행할 때, 학습 시간이 모두 쌓여 비용 효율적인 결과를 제공합니다.



### Revisiting Vacuous Reduct Semantics for Abstract Argumentation (Extended Version) (https://arxiv.org/abs/2408.14069)
Comments:
          The paper has been accepted at ECAI 2024, this is an extended version including proofs of technical results

- **What's New**: 본 논문에서는 추상 논증(framework)에서의 공허한 축소(vacuous reduct) 의미론에 대한 새로운 접근법을 제안합니다. 두 가지 의미론인 {\sigma}와 {\tau}를 결합하여 {\	au}의 비어 있지 않은 확장을 허용하지 않는 {\	au}-확장만을 수용하는 방법을 탐구합니다.

- **Technical Details**: 공허한 축소 의미론은 여러 가지 수용 가능(based on admissibility)하고 갈등 없는(conflict-free) 의미론을 결합하여 발생합니다. 우리는 이러한 공허한 축소 의미론을 기본 조건(base condition)과 공허성 조건(vacuity condition)에 따른 원칙 기반 분석(principle-based analysis)을 통해 체계적으로 살펴봅니다. 또한 약한 논증 의미론(weak argumentation semantics) 내에서 새롭게 제정된 원칙에 대한 기준(criteria)을 제공합니다.

- **Performance Highlights**: 본 연구는 특별한 경우인 분쟁이 없는(undisputed) 의미론에 대한 원칙 기반 분석(principle-based analysis)을 포함하며, 기존의 원칙뿐만 아니라 최신 원칙들의 계승 가능성(inheritance)에 대해 설명합니다.



### MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents (https://arxiv.org/abs/2408.14033)
- **What's New**: 새로운 체계적 프레임워크인 MLR-Copilot을 제안하여 기계 학습 연구의 생산성을 향상시킵니다. 이 프레임워크는 기계 학습 연구 아이디어의 자동 생성 및 구현을 위해 대형 언어 모델(Large Language Model, LLM) 에이전트를 사용합니다.

- **Technical Details**: MLR-Copilot은 세 가지 단계(연구 아이디어 생성, 실험 구현, 실행)로 구성됩니다. IdeaAgent는 문헌에서 연구 질문을 추출하여 가설과 실험 계획을 생성하며, ExperimentAgent는 이 계획을 실행 가능하도록 변환합니다.

- **Performance Highlights**: 다섯 가지 기계 학습 연구 과제를 평가한 결과, MLR-Copilot이 연구 진행과 혁신을 촉진할 수 있는 잠재력을 갖춘 것으로 나타났습니다.



### Geo-Llama: Leveraging LLMs for Human Mobility Trajectory Generation with Spatiotemporal Constraints (https://arxiv.org/abs/2408.13918)
- **What's New**: 이번 논문에서는 Geo-Llama라는 새로운 프레임워크를 제안하여 공간-시간적(spatiotemporal) 제약 조건을 준수하면서 인간 이동 경로를 생성하는 문제를 접근합니다. 이전의 방법들은 일반적으로 훈련 안정성 문제와 데이터 크기가 증가함에 따라 잘 스케일링되지 않는 문제를 가지고 있었고, 구체적인 방문 제약을 설정하는 기능이 부족했습니다.

- **Technical Details**: Geo-Llama는 프리트레인(pre-trained)된 LLMs(대형 언어 모델)을 활용하여 방문별(permutation) 전략으로 훈련하여, 방문을 시간과 위치에 맞추어 생성합니다. 이 방법은 모델이 방문 순서와 관계없이 지역-시간적(spatiotemporal) 패턴을 포착할 수 있게 해주어 유연하고 맥락에 맞는 제약을 생성 과정에서 통합할 수 있게 합니다.

- **Performance Highlights**: 실제 및 합성 데이터셋을 통한 실험 결과, Geo-Llama는 제약 조건을 잘 준수하면서 기존 방법들과 비교하여 보다 현실적인 이동 경로를 생성하는 데 있어 그 유연성과 강력한 성능을 입증하였습니다. 특히, 제한된 훈련 데이터를 사용했음에도 기존 최첨단 접근 방식보다 더 효율적으로 학습할 수 있음을 보여주었습니다.



### Multi-Agent Target Assignment and Path Finding for Intelligent Warehouse: A Cooperative Multi-Agent Deep Reinforcement Learning Perspectiv (https://arxiv.org/abs/2408.13750)
- **What's New**: 이 연구에서는 협력적인 다중 에이전트 딥 강화 학습(cooperative multi-agent deep reinforcement learning) 관점에서 목표 할당(target assignment)과 경로 계획(path planning) 문제를 동시에 해결하는 방법을 제안합니다. 이는 지능형 창고에 대한 TAPF 문제를 모델링한 첫 번째 연구입니다.

- **Technical Details**: TAPF (Task Assignment and Path Finding) 문제는 NP-hard 문제로, 전통적인 방법으로 해결하기 어려운 비선형 분야입니다. 본 연구에서 제안된 방법은 MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 알고리즘을 사용하여, 각 에이전트의 물리적 동역학(dynamics)을 고려하여 경로 계획을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 작업 설정에서 우수한 성능을 보이며, 목표 할당은 합리적으로 해결되고 계획된 경로는 거의 최단 경로에 가깝습니다. 또한, 본 방법은 기존 방법(caller baselines)보다 시간 효율성이 더 높습니다.



### Count-based Novelty Exploration in Classical Planning (https://arxiv.org/abs/2408.13719)
Comments:
          Extended version of paper accepted for publication at ECAI 2024

- **What's New**: 이번 연구에서는 전통적인 탐색 기법과 함께 활용될 수 있는 새로운 novelty 기법인 classical count-based novelty를 제안합니다. 이 기법은 탐색 트리에서 각 tuple의 등장 빈도를 활용하여 일정한 수의 tuple을 사용하여 상태 공간을 탐색할 수 있도록 합니다.

- **Technical Details**: classical count-based novelty 기법은 기존의 novelty 메트릭에서 발생할 수 있는 tuple 수의 기하급수적 증가 문제를 해결합니다. 본 논문에서 제안하는 trimmed open list는 나쁜 novelty 값을 가지는 노드를 가지치기하여 일정한 크기를 유지함으로써 탐색의 효율성을 높입니다.

- **Performance Highlights**: 제안된 기법을 통합한 기존의 classical solver는 최근 국제 계획 대회에서 도전적인 벤치마크에서 경쟁력 있는 성과를 거두었습니다. 또한, 메모리 및 시간 임계값을 모두 활용하는 dual 구성에서, 인스턴스 커버리지에서 유의미한 증가를 보여 현재의 최첨단 solver를 초월하는 성과를 기록했습니다.



### GPT-4 Emulates Average-Human Emotional Cognition from a Third-Person Perspectiv (https://arxiv.org/abs/2408.13718)
Comments:
          submitted to 12th International Conference on Affective Computing & Intelligent Interaction, Glasgow, UK, September 15-18, 2024

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 감정 추론 능력에 대한 최근 연구를 확장한 내용으로, LLM이 타인의 감정 인식과 자기 감정 인식 간의 차이를 평가한다.

- **Technical Details**: 저자들은 인위적으로 설계된 감정 유도 자극(emotion-evoking stimuli)을 사용하여, GPT-4가 그러한 자극에 대한 추론에서 특히 정확성이 높음을 보여준다. 감정 이론(appraisal theory)과 같은 이론적 틀을 통해 LLM의 감정 처리를 탐구하며, LLM이 타인의 감정을 추론하는 방식과 자아 감정을 인식하는 방식의 차이를 분석한다.

- **Performance Highlights**: GPT-4는 감정 유도 자극에 대한 추론에서 인간의 판단과 잘 일치하는 경향이 있으며, 특히 타인의 감정에 대한 해석이 개인의 자기 평가보다 더 신뢰할 수 있는 것으로 나타났다. 기존의 감정 모델은 주로 자기 보고(self-reported) 기반의 진실을 표준으로 삼지만, 이 연구는 관찰자의 관점을 채택하는 것이 더 많은 응용 프로그램에서 유용할 수 있음을 시사한다.



### Evaluating Alternative Training Interventions Using Personalized Computational Models of Learning (https://arxiv.org/abs/2408.13684)
Comments:
          18 pages, 7 figures

- **What's New**: 이 연구에서는 A/B 실험의 한계를 극복하고 교육 개입의 효과를 예측하기 위한 개인화된(computed) 모델을 제안합니다. 기존 A/B 실험에 비해 더 적은 비용으로 다양한 개입의 효과를 평가할 수 있는 방법을 모색하였습니다.

- **Technical Details**: 연구에서는 개인화된 모델을 학습하고, 이러한 모델들이 학생 행동을 예측하는 데 있어 일반 모델보다 우수하다는 것을 시뮬레이션을 통해 입증합니다. 또한, counterfactual prediction(역사적 예측)로 학생들의 성과를 예측하며, 이를 통해 교육 개입의 효과 검증을 제안합니다.

- **Performance Highlights**: 이 모델은 이전에 수집된 데이터를 활용하여 개인화된 에이전트를 만들고, 이 에이전트를 통해 학생의 개별 성향에 맞춘 맞춤형 예측을 제공합니다. 시뮬레이션 결과, 개인화된 모델이 일반화된 모델보다 학생의 반응을 더 정확하게 예측하는 것으로 나타났습니다.



### Uncovering Biases with Reflective Large Language Models (https://arxiv.org/abs/2408.13464)
Comments:
          16 pages, 3 figures, 8 tables

- **What's New**: 본 연구는 여러 개의 대규모 언어 모델(LLMs)을 활용하여 서로의 관점을 탐구하는 반사적(reflective) 방법론을 제안합니다. 이는 기존의 편향된 학습 데이터에 대한 의존성을 극복하는 데 기여하고자 합니다.

- **Technical Details**: 이 방법은 conditional statistics, 정보 이론(information theory), 그리고 divergence metrics를 활용하여 맥락에 따라 언어적 행동을 촉진하고, 편향되지 않은 결과를 생성하는 것을 목표로 합니다.

- **Performance Highlights**: 연구에서는 31개의 뉴스 기사를 분석하였으며, 이 과정에서 편향을 정량적으로 평가하고, 개선 조치를 설명 가능하게 만드는 과정도 포함되었습니다.



### Optimizing Collaboration of LLM based Agents for Finite Element Analysis (https://arxiv.org/abs/2408.13406)
- **What's New**: 이 논문은 다수의 에이전트들이 프로그래밍 및 코딩 작업에서 Large Language Models (LLMs) 내에서 어떻게 상호작용하는지를 조사합니다.

- **Technical Details**: AutoGen 프레임워크를 활용하여 에이전트 간의 통신을 촉진하며, 각 설정에 대해 40회의 무작위 실행에서 성공률을 기반으로 다양한 구성을 평가합니다. 연구는 선형 탄성 문제를 해결하기 위한 Finite Element Method (FEM)를 적용하기 위한 유연한 자동화 프레임워크 개발에 중점을 둡니다.

- **Performance Highlights**: 연구 결과는 에이전트의 역할 최적화와 책임을 명확히 정의하는 것이 중요하다는 점을 강조하며, 단순히 에이전트의 수를 증가시키는 것만으로는 부족하다고 보여줍니다. 에이전트 간의 효과적인 협력은 일반 FEM 과제를 해결하는 데 있어 필수적임을 나타냅니다.



### DrugAgent: Explainable Drug Repurposing Agent with Large Language Model-based Reasoning (https://arxiv.org/abs/2408.13378)
Comments:
          18 pages, 1 figure

- **What's New**: 이번 논문은 기존 약물의 새로운 치료 가능성을 찾기 위해 다중 에이전트 시스템(multi-agent system) 프레임워크를 제안하고 있습니다. 최신 기계 학습(machine learning) 기법과 지식 통합(knowledge integration)을 사용하여 약물 재목적화(drug repurposing) 과정을 향상시키는 데 초점을 둡니다.

- **Technical Details**: 프레임워크는 AI Agent, Knowledge Graph Agent, Search Agent의 세 가지 전문 에이전트로 구성됩니다. AI Agent는 강력한 약물-타겟 상호작용(drug-target interaction, DTI) 모델을 훈련하고, Knowledge Graph Agent는 다양한 데이터베이스를 활용해 DTI를 체계적으로 추출하며, Search Agent는 생물 의학 문헌과 상호작용하여 계산된 예측을 주석 달고 검증합니다. 이 시스템은 외부 데이터베이스에서 얻은 다양한 데이터 소스를 효과적으로 활용합니다.

- **Performance Highlights**: 예비 결과에 따르면, 이 접근법은 약물-질병 상호작용을 예측하는 데 있어 기존 방법들보다 뛰어난 성능을 보이며, 전통적인 약물 발견 과정에 비해 시간과 비용을 줄일 수 있는 가능성을 보여줍니다. 또한, 다중 에이전트 시스템의 확장성을 강조하며, 약물 재목적화 분야에서의 혁신을 촉진하는 역할을 합니다.



### Reduce, Reuse, Recycle: Categories for Compositional Reinforcement Learning (https://arxiv.org/abs/2408.13376)
Comments:
          ECAI 2024

- **What's New**: 이 논문은 강화 학습에서 작업 구성(task composition) 문제를 해결하기 위해 범주론(category theory)의 관점을 도입한 새로운 접근법을 제안합니다. 연구자는 Markov 결정 과정(Markov decision processes)의 범주적 속성을 활용하여 복잡한 작업을 관리 가능한 하위 작업으로 분해합니다.

- **Technical Details**: 논문에서 제안된 방법은 작업의 차원 축소(dimensionality reduction), 보상 구조(reward structures) 최적화, 시스템의 견고성(robustness)을 향상시키는 방식을 포함합니다. 특히, 각 하위 작업에 대한 별도의 보상 구조를 할당하여 드문 보상이 주어지는 상황에서도 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 제안된 범주론적 강화 학습 접근 방식의 유효성을 보여주며, 복잡한 로봇 팔 작업을 학습하는 과정에서 기술 축소(skill reduction), 재사용(reuse), 재활용(recycling)을 가능하게 하였습니다.



### Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning (https://arxiv.org/abs/2408.14472)
Comments:
          Robotics: Science and Systems (RSS), 2024. (Best Paper Award Finalist)

- **What's New**: 이번 연구에서는 Denoising World Model Learning (DWL)이라는 새로운 강화 학습 프레임워크를 소개했습니다. 이를 통해 세계 최초의 휴머노이드 로봇이 복잡한 실제 지형에서 보행을 조절할 수 있게 되었습니다.

- **Technical Details**: DWL은 시뮬레이션과 현실 세계 간의 간극을 줄이기 위한 효과적인 표현 학습 프레임워크를 설정합니다. 2-DoF 발목 제어가 포함된 폐쇄 기구 체인 메커니즘을 사용하여 휴머노이드 로봇의 안정성과 유연성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 연구에서 제안하는 방법은 눈이 쌓인 경사 지대, 계단 및 불규칙한 표면과 같은 복잡한 지형을 안정적으로 탐색할 수 있는 능력을 보여주었습니다. 모든 시나리오에서 동일한 학습된 신경망 정책을 적용하여 강력한 일반화 능력을 입증했습니다.



### Temporal Ensemble Logic (https://arxiv.org/abs/2408.14443)
Comments:
          47 pages, 2 figures

- **What's New**: Temporal Ensemble Logic (TEL)가 처음 소개되었습니다. TEL는 임상 및 인구 건강 연구에서의 엄밀성과 재현성을 요구하는 부분에서 출발하여, 바이오 의학에서의 시간적 추론을 형식화하는 데 필요한 격차를 메우는 것을 목표로 합니다.

- **Technical Details**: TEL은 '항상 $t$ 시간 후' ($\Box_t$), '가끔 $t$ 시간 이전' ($\Diamond_t$), '$t$ 시간 후' ($\varphi_t$) 등의 기본적인 시간적 구조를 포함하고 있습니다. 이 논문에서는 이산적 및 조밀한 시간에 대한 특수 사례와 함께 일반적인 설정에서 TEL을 소개합니다. 그 후, 정수의 시계열 도메인에서 이산 TEL(${\rm TEL}_{\mathbb{N}^{+}}$)의 이론적 개발에 집중하며, 이는 표준 모나딕 제2차 논리보다 표현력이 더 강력합니다.

- **Performance Highlights**: ${\rm TEL}_{\mathbb{N}^{+}}$의 만족 가능성과 관련하여 결정 불가능성을 증명하였으며, TEL의 다양한 조각들이 더 나은 계산 가능성을 지닐 수 있도록 정의할 수 있도록 하는 뛰어난 표현력을 제공합니다. 또한, TEL은 임상 환경에서의 나타나는 표현적 속성을 적절하게 표현할 수 있도록 고안되었습니다.



### Attend-Fusion: Efficient Audio-Visual Fusion for Video Classification (https://arxiv.org/abs/2408.14441)
- **What's New**: 본 연구에서는 Attend-Fusion이라는 오디오-비주얼(AV) 융합 접근법을 제안합니다. 이 접근법은 컴팩트 모델 아키텍처를 통해 비디오 데이터에서 오디오-비주얼 관계를 효과적으로 캡처하며, YouTube-8M 데이터셋에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: Attend-Fusion은 72M 파라미터를 사용하여 75.64%의 F1 점수를 달성합니다. 이는 341M 파라미터를 가진 기존의 Fully-Connected Late Fusion 모델의 75.96% F1 점수와 비슷한 성능을 보여주며, 모델 크기를 80% 이상 줄이는 데 성공하였습니다.

- **Performance Highlights**: Attend-Fusion 모델은 오디오와 비주얼 정보를 효과적으로 결합하여 비디오 분류 작업에서 높은 성능을 달성하며, 리소스가 제한된 환경에서도 고성능 비디오 이해 시스템의 배포가 가능하다는 점에서 큰 의미가 있습니다.



### Sparsity-Aware Hardware-Software Co-Design of Spiking Neural Networks: An Overview (https://arxiv.org/abs/2408.14437)
- **What's New**: 이번 연구는 Sparse Spiking Neural Networks (SNNs)의 하드웨어-소프트웨어 공동 설계(hardware-software co-design)의 중요성을 강조하며, 생물학적 신경 처리의 희소성과 이벤트 기반 통신에서 영감을 받은 SNN의 잠재적인 에너지 효율성을 살펴봅니다.

- **Technical Details**: SNN은 고도로 희소한 신경 네트워크 구조를 사용하여 계산을 수행하며, 이는 사건 중심의 처리(event-driven processing)를 통해 전력을 절약할 수 있게 해줍니다. 연구에서는 정적(static) 및 동적(dynamic) 희소성의 차이를 설명하며, 이는 하드웨어 효율성에 미치는 영향을 논의합니다.

- **Performance Highlights**: SNN을 상용화하기 위해 하드웨어 아키텍처와 알고리즘의 조화를 통한 성능 및 에너지 효율의 개선이 필요하며, 다양한 입력 인코딩 방법과 신경 모델이 하드웨어 구현의 최적화에 미치는 영향을 정량적으로 분석했습니다.



### Social perception of faces in a vision-language mod (https://arxiv.org/abs/2408.14435)
- **What's New**: 이 논문은 CLIP 모델을 통해 인간 얼굴에 대한 사회적 인식을 탐구하고, 다양한 얼굴 속성과 텍스트 프롬프트 간의 유사성을 비교합니다. 연구는 얼굴 이미지를 체계적으로 변형하여 사회적 인식에 미치는 영향을 실험적으로 분석하였습니다.

- **Technical Details**: CLIP(Contrastive Language–Image Pre-training) 모델을 사용하여 텍스트와 이미지 쌍의 임베딩(embedding) 간의 코사인 유사성을 측정하여 사회적 판단을 분석합니다. 연구에서는 나이, 성별, 인종, 표정, 조명, 자세 등의 속성을 독립적으로 조작하여 실험을 진행했습니다. 논문에서 제시된 데이터셋은 CausalFace로, GAN(Generative Adversarial Network)을 이용한 합성 이미지들로 이루어져 있습니다.

- **Performance Highlights**: CLIP 모델은 다양한 속성과 관련하여 강력한 사회적 편향을 보이며, 특히 흑인 여성의 얼굴에 대한 극단적인 사회적 인식 값을 생성합니다. 표정이 나이와 조명보다 사회적 인식에 더 큰 영향을 미친다는 발견은 기존 연구의 편향을 잘못된 결론으로 이끌 수 있음을 내포합니다.



### Contextual Bandit with Herding Effects: Algorithms and Recommendation Applications (https://arxiv.org/abs/2408.14432)
- **What's New**: 이 논문은 추천 시스템 분야에서 사용자의 피드백에 어떤 영향을 미치는 'herding effects'를 고려하여 새로운 개선 방식인 TS-Conf (Thompson Sampling under Conformity) 알고리즘을 제안합니다. 기존의 contextual bandits 알고리즘이 이 피드백 편향을 무시했던 점을 주목하여, 이를 해결하기 위해 사용자 피드백 모델을 개발했습니다. 이는 추천 의사결정에서 탐색(exploration)과 활용(exploitation) 사이의 균형을 맞추는데 기여할 것입니다.

- **Technical Details**: 저자들은 herding effects가 사용자의 피드백에 미치는 바를 수량화하기 위한 모델을 제안하며, 이를 기반으로 하여, TS-Conf 알고리즘을 사용해 posterior sampling 접근 방식을 통해 탐색과 활용의 절충을 효과적으로 구현합니다. 이는 또한 herding effects의 상단 한계를 명시하여, 학습 속도에 미치는 영향을 드러냅니다.

- **Performance Highlights**: TS-Conf 알고리즘은 네 개의 공개 데이터셋에서 실험을 수행한 결과, 하위 선형 하강(regret) 특성을 보여주며, 기존 세 가지 벤치마크 알고리즘보다 월등한 성능을 발휘했습니다. 이를 통해 사용자의 피드백에서 발생하는 부정적인 영향을 효과적으로 완화하여, 더 빠른 학습 속도와 향상된 추천 정확도를 달성했음을 보여줍니다.



### MEDSAGE: Enhancing Robustness of Medical Dialogue Summarization to ASR Errors with LLM-generated Synthetic Dialogues (https://arxiv.org/abs/2408.14418)
- **What's New**: 본 논문은 임상 대화 요약에서 사용되는 노이즈 저항성을 높이기 위해 대규모 언어 모델(LLM)을 이용하여 합성 데이터를 생성하는 MEDSAGE라는 새로운 방법을 제안합니다. 이 방법은 기존의 ASR 시스템의 오류를 시뮬레이션하여 데이터 증강을 가능하게 합니다.

- **Technical Details**: MEDSAGE는 LLM의 맥락 내 학습(in-context learning) 기능을 활용하여 몇 가지 예제를 기반으로 ASR과 유사한 오류를 생성합니다. 이를 통해 실제 ASR 오류 패턴을 반영하는 합성 대화를 생성하고, 오류 유형에 맞춤화된 태그 구문을 도입하여 데이터 증강을 시도합니다.

- **Performance Highlights**: 실험 결과, LLM을 활용하여 생성된 합성 데이터가 ASR 오류를 효과적으로 반영하며, 이러한 합성 데이터를 훈련에 포함시킴으로써 조용한 테스트 세트에서 성능이 최대 16% 향상됨을 보여줍니다.



### Language-specific Calibration for Pruning Multilingual Language Models (https://arxiv.org/abs/2408.14398)
- **What's New**: 본 논문은 다국어 언어 모델(Multilingual Language Models)의 프루닝(pruning) 과정에서 칼리브레이션(calibration) 언어의 효과적인 전략을 탐구합니다. 주목할 만한 점은 대부분의 연구가 영어 데이터에 기반하고 있다는 점에서, 다양한 언어 간 차이를 고려한 최초의 종합적 실험 연구라는 것입니다.

- **Technical Details**: 모델 프루닝을 위한 다양한 칼리브레이션 언어의 성능을 비교하며, 7개 언어(아랍어, 독일어, 영어, 스페인어, 러시아어, 스와힐리어, 중국어)에 대해 실험하였습니다. 사용된 주 기술로는 프루닝을 위한 두 가지 최신 기법인 Wanda와 SparseGPT를 채택했습니다. 실험에서는 각 언어에 대해 128개의 샘플을 기반으로 데이터 세트를 구성했습니다.

- **Performance Highlights**: 목표 언어로 칼리브레이션할 경우 낮은 perplexity를 유지하지만, 하위 작업에서 최적 성능을 보장하지는 않습니다. SparseGPT는 Llama-3 8B 모델에 적합하나, 다른 모델과 작업에서는 mixed performance를 보였습니다. 이는 지식 저장 및 검색 과정에서 다국어 모델이 상당한 영향을 받는다는 것을 보여줍니다.



### Reprogramming Foundational Large Language Models(LLMs) for Enterprise Adoption for Spatio-Temporal Forecasting Applications: Unveiling a New Era in Copilot-Guided Cross-Modal Time Series Representation Learning (https://arxiv.org/abs/2408.14387)
Comments:
          Paper published at the Deployable AI (DAI) workshop at AAAI-2024

- **What's New**: 본 연구는 대규모 및 소규모 언어 모델(LLMs 및 LMs)의 강점을 전통적인 예측 방법과 결합하는 하이브리드 접근 방식을 제안합니다. 이를 통해 시간 시계열 데이터의 비선형적 변화에 따른 intra-series 및 inter-series dependencies를 효과적으로 포착하고, 예측 정확도를 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 Grouped-query Multi-head Attention (GQ-MHA) 메커니즘을 사용하여 시계열 데이터의 동적 의존성을 모델링하며, Low-Rank Adaptation과 Activation Memory Reduction (LoRA-AMR) 기술을 활용하여 작은 오픈 소스 LMs를 fine-tuning 합니다. 또한, on-premise 환경에서 LLM을 사용하는 방안을 통해 데이터 안전성과 맞춤화를 제공합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험 결과, 기존 방법들에 비해 예측 정확도가 유의미하게 향상됨을 입증했습니다. 이 프레임워크는 적응 가능한 프롬프트 메커니즘을 통해 변화하는 MTS 데이터에 보다 유연하게 대응할 수 있으며, 향후 예측 및 리스크 평가의 정확도를 높이는 데 기여할 것으로 기대됩니다.



### Probing Causality Manipulation of Large Language Models (https://arxiv.org/abs/2408.14380)
- **What's New**: 이 논문은 LLMs(대형 언어 모델)의 인과성 조작을 탐색하기 위한 새로운 계층적 접근 방식을 제안합니다. 다양한 단축키를 제공하여 모델의 행동을 관찰하는 방식입니다.

- **Technical Details**: 우리는 RAG(검색 증강 생성) 및 ICL(컨텍스트 학습)을 활용하여 설계된 인과성 분류 작업에서 LLMs의 성능 변화를 관찰합니다. 이를 통해 LLMs가 인과적으로 관련된 개체를 감지하고 직접적인 인과 관계를 인식할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, LLM들은 인과성과 관련된 개체를 인식할 수 있으나, 인과 관계에 대한 전문적인 인지가 부족하여 구문 내에서 이들을 단순히 전반적인 의미로 취급합니다. 이는 LLMs의 훈련 과정에서 인과성에 대한 추가적인 주의가 필요함을 보여줍니다.



### SelEx: Self-Expertise in Fine-Grained Generalized Category Discovery (https://arxiv.org/abs/2408.14371)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 논문에서는 Generalized Category Discovery (GCD)를 다루며, 이 과정에서 아직 알려지지 않은 카테고리를 찾아내고 기존 카테고리를 정확히 분류하려고 합니다. 특히, 기존 방법들이 미세한 카테고리 구분에서 부족한 성능을 보이는 문제를 해결하기 위해 `self-expertise`라는 새로운 개념을 제안합니다.

- **Technical Details**: `self-expertise`는 모델이 미세한 차이를 인식하고 새로운 카테고리를 발견하는 능력을 향상시킵니다. 이 방법은 자율학습(unsupervised)과 감독학습(supervised) `self-expertise` 전략을 결합하여 모델의 판단력과 일반화 능력을 개선합니다. 초기에 계층적 의사 레이블링(hierarchical pseudo-labeling)을 통해 `soft supervision`을 제공하여 self-expertise의 효과를 높입니다. 또한, 감독 기술은 전통적인 방법과 다르게 더 추상적인 긍정(positive) 및 부정(negative) 샘플을 사용하여 클러스터를 형성하고 새로운 카테고리로 일반화하는 데 도움을 줍니다.

- **Performance Highlights**: 실험적으로, 우리는 제안된 방법이 여러 미세한 데이터셋에서 기존의 최첨단 기법보다 더 뛰어난 성능을 보임을 확인했습니다. 이러한 결과는 이론적 통찰에 의해 뒷받침됩니다.



### GR-MG: Leveraging Partially Annotated Data via Multi-Modal Goal Conditioned Policy (https://arxiv.org/abs/2408.14368)
Comments:
          9 pages, 7 figures, letter

- **What's New**: 이번 논문에서는 자연어 지시사항 및 목표 이미지를 기반으로 로봇의 작업 수행 범위를 확장시키기 위한 새로운 모델 GR-MG를 제안합니다. 이 모델은 부분적으로 주석이 달린 데이터(예: 인간의 활동 비디오, 로봇의 플레이 데이터)를 활용하여 로봇 매니퓰레이션의 일반화 능력을 높이고자 합니다.

- **Technical Details**: GR-MG는 두 가지 모듈로 구성되어 있습니다. 첫 번째는 진행 상황 정보를 주입하여 목표 이미지를 생성하는 진척 가이드(goal-image generation) 모델이고, 두 번째는 언어 지시 및 목표 이미지에 기반하여 행동을 예측하는 다중 모달 목표 조건(policy) 모델입니다. 이 모델은 훈련 중 언어 지시가 없을 때에도 목표 이미지를 활용할 수 있습니다.

- **Performance Highlights**: 시뮬레이션 실험에서는 GR-MG가 1회 및 5회의 작업 성공률을 각각 93.8%에서 96.8%, 41.2%에서 64.4%로 개선했습니다. 실제 로봇 실험에서도 GR-MG는 47개의 다양한 작업을 수행하며, 성공률을 62.5%에서 75.0%, 일반화 설정에서는 42.4%에서 57.6%로 개선했습니다.



### SWE-bench-java: A GitHub Issue Resolving Benchmark for Java (https://arxiv.org/abs/2408.14354)
Comments:
          This work is in progress

- **What's New**: 본 논문에서는 다국어 지원을 향한 첫걸음으로 Java 버전인 SWE-bench-java를 개발하였으며, 이는 기존의 Python 중심 SWE-bench의 한계를 보완하기 위한 것입니다. 이는 산업에서 필요한 다양한 프로그래밍 언어 지원을 목표로 합니다.

- **Technical Details**: SWE-bench-java는 53개의 GitHub Java 리포지토리와 Defects4j 데이터베이스에서 수집된 17개의 리포지토리로 구성된 70개의 후보 리포지토리로부터 구축되었습니다. 최종적으로 19개의 오픈소스 Java 리포지토리에서 1,979개의 이슈 인스턴스를 수집하였으며, 이 중 137개의 고급 테스트가 포함된 이슈 인스턴스가 검증되었습니다.

- **Performance Highlights**: SWE-bench-java를 통해 여러 강력한 LLM 모델(GPT-4o, DeepSeek-V2 등)과의 성능 평가가 이루어졌습니다. 이는 다국어 GitHub 이슈 해결 벤치마크를 구축하기 위한 첫 단계를 마련하고, Java 프로젝트에서의 이슈 해결을 보다 잘 이해할 수 있는 기회를 제공합니다.



### Assessing Contamination in Large Language Models: Introducing the LogProber method (https://arxiv.org/abs/2408.14352)
- **What's New**: 이번 논문에서는 LogProber라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 Large Language Models (LLMs)에서 contamination(오염)을 감지하기 위한 도구로, 특정 문장에서 token probability를 활용하여 오염 여부를 평가하는 방식으로 작동합니다.

- **Technical Details**: LogProber는 질문/답변 형식의 짧은 텍스트에 적합하도록 설계되었습니다. 기존 방법들이 noise로 인해 오염을 탐지하기 어려운 짧은 질문들에 대해 효과적으로 작동할 수 있도록 개발되었습니다. 실험을 통해 LLM (LLAMA)에 최근 발표된 인지 테스트 항목을 적용하여 이 알고리즘이 오염을 감지하는 데 효과적임을 보여주었습니다. 그러나 모델의 학습 방식에 따라 모든 유형의 오염을 감지할 수 없음을 발견했습니다.

- **Performance Highlights**: LogProber는 LLM의 오염 여부를 감지하는 데 있어 저렴한 계산 비용과 높은 효율성을 자랑합니다. 그러나 특정 실험에서는 질문 텍스트가 아닌 답변 텍스트에만 맞춘 모델에 대해서는 오염을 인식하지 못했습니다. 이는 모든 오염 유형이 token probability만으로 탐지될 수 없다는 점을 강조합니다.



### Foundation Models for Music: A Survey (https://arxiv.org/abs/2408.14340)
- **What's New**: 최근 몇 년 동안, 대형 언어 모델(LLMs)과 잠재 확산 모델(LDMs)과 같은 foundation models(FMs)이 음악을 포함한 다양한 분야에 심각한 영향을 미쳤습니다. 이 리뷰는 음악의 사전 훈련 모델과 최신 기술을 조사하며, 음악 이해 및 생성 측면에서의 FM의 잠재력을 강조합니다.

- **Technical Details**: 연구에서는 예를 들어, 단일 모달리티 모델과 다중 모달리티 모델을 구분하고, 자가 감독 학습(self-supervised learning, SSL)이 음악 데이터에 어떻게 적용되는지 설명합니다. 또한, 구조적 선택, 토크나이징(tokenisation), 파인튜닝(finetuning) 방법론 및 제어 가능성(controllability)과 같은 모델 사전 훈련 패러다임의 세부 사항을 논의합니다.

- **Performance Highlights**: FMs는 데이터 부족 문제를 해결하고 주석 비용을 줄이는 동시에 음악 정보 검색(music information retrieval) 및 창작에서 일반화(generalization)를 향상시킵니다. 음악 장르, 구조 또는 악기를 이해하는 데 더 뛰어난 성능을 보이며, 문화유산 보호 및 예술 표현의 새로운 형태에 기여할 잠재력을 내포하고 있습니다.



### Equivariant Reinforcement Learning under Partial Observability (https://arxiv.org/abs/2408.14336)
Comments:
          Conference on Robot Learning, 2023

- **What's New**: 이 논문은 로봇 학습에서 샘플 효율성을 높이기 위해 대칭성을 활용하는 새로운 방법을 제안합니다. 특히, 대칭성을 신경망에 인코딩하여 에이전트가 과거의 솔루션을 재사용할 수 있도록 합니다.

- **Technical Details**: 기존의 완전 관측 Markov 결정 프로세스(MDP)에 대한 연구와는 달리, 이 논문은 부분 관측 프로세스인 POMDP에서 대칭성을 활용한 새로운 이론과 솔루션 방법을 제시합니다. 특정 그룹 대칭에 대한 등가성을 신경망에 Embed하여 에이전트가 더 나은 정책을 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 우리의 에이전트는 비대칭(non-equivariant) 접근 방식에 비해 샘플 효율성과 최종 성능에서 현저한 개선을 보여주었습니다. 특히, Advantage Actor-Critic (A2C) 및 Soft Actor-Critic (SAC) 알고리즘을 사용하여 로봇 조작 작업에서 실험한 결과, 대칭적 접근 방식이 더 뛰어난 성과를 기록했습니다.



### PHEVA: A Privacy-preserving Human-centric Video Anomaly Detection Datas (https://arxiv.org/abs/2408.14329)
- **What's New**: PHEVA는 인간 중심의 윤리적 비디오 이상 탐지(VAD)를 위한 가장 큰 데이터셋으로, 사생활을 보호하는 특징을 가지고 있다. 이는 전통적인 데이터셋들과는 달리, 개인 식별 정보를 제거하고 비식별화된 인간 주석만을 제공한다.

- **Technical Details**: PHEVA 데이터셋은 7개의 실내/실외 장면을 포함하며, 6개의 기초 카메라와 법 집행 훈련을 위한 컨텍스트 특정 카메라를 채용하여 다양한 환경을 포괄한다. 또한, 훈련 프레임 수는 이전 최대 데이터셋보다 5배 이상 많으며, 82.14%의 경우에서 기존 방법들을 초월한 결과를 제공한다. 데이터셋은 10% 오류 비율(10ER)과 같은 새로운 메트릭을 포함하여 포즈 기반 VAD 알고리즘을 위한 포괄적 벤치마킹을 수행할 수 있도록 설계되었다.

- **Performance Highlights**: PHEVA는 지속적 학습을 위한 벤치마크를 처음 도입하며, 이는 기존의 전통적인 방법에 비해 성능이 82.14% 더 우수함을 보여준다. 새로운 10ER 메트릭은 실제 환경에서의 잘못된 긍정 및 부정의 비용 차이를 고려하여 비정형 탐지의 정확성을 개선한다.



### Streamline tractography of the fetal brain in utero with machine learning (https://arxiv.org/abs/2408.14326)
- **What's New**: 이 연구는 처음으로 태아의 차도(tractography)를 위한 머신러닝 모델을 개발하여, 태아의 백질 섬유를 재구성하는 혁신적인 접근 방법을 제공합니다.

- **Technical Details**: 모델 입력은 다섯 가지 정보 출처로 구성되어 있습니다: (1) 확산 텐서를 사용하여 추정한 섬유 방향, (2) 최근 전파 단계의 방향, (3) 뇌 피질의 주요 지점까지의 거리로 인코딩된 전역 공간 정보, (4) 조직 분할 정보, (5) 아틀라스를 통해 제공된 예상 지역 섬유 방향에 대한 prior 정보. 이 모델은 컨볼루션 및 어텐션 신경망 모듈을 사용하여 현재 지점 주변의 대규모 공간 문맥을 인코딩합니다.

- **Performance Highlights**: 제안한 방법은 평가된 모든 트랙에서 우수한 성능을 보였으며, 태아의 정상 및 비정상적인 뇌 발달 연구에 대한 dMRI의 능력을 크게 발전시킬 수 있습니다.



### Claim Verification in the Age of Large Language Models: A Survey (https://arxiv.org/abs/2408.14317)
- **What's New**: 이 논문은 최근의 LLM(대형 언어 모델)을 활용한 클레임 검증(claim verification) 프레임워크에 대한 종합적인 조사를 제시하고 있습니다. 기존의 연구들과는 달리 LLM 기반 접근 방식에 주목하며, 클레임 검증 파이프라인의 다양한 구성 요소를 세부적으로 설명합니다.

- **Technical Details**: 클레임 검증 프로세스는 클레임 탐지(claim detection), 증거 검색(evidence retrieval), 진위 예측(veracity prediction)으로 구성됩니다. LLM이 도입되면서 기존 NLP 모델 대비 Misinformation 생성 및 검증에서 더 낮은 오류 전파를 보이며, 더 설명 가능하고 해석 가능한 결과를 제공합니다. RAG(Retrieval Augmented Generation) 기술을 통해 최신 정보에 접근하여 모델의 의사 결정에 도움을 줍니다.

- **Performance Highlights**: LLM 모델은 이전의 수작업 또는 기존 방법론보다 높은 성능과 신뢰도를 자랑하지만, 여전히 허위 정보 생성 가능성(hallucination) 문제로 인해 정확한 진위 라벨을 생성하는 데 한계가 있습니다. 최근 연구들은 자동화된 클레임 검증에서 LLM의 효용성을 보여주며, 앞으로의 연구 방향을 제시하고 있습니다.



### Logic interpretations of ANN partition cells (https://arxiv.org/abs/2408.14314)
- **What's New**: 이번 연구에서는 인공신경망(Artificial Neural Network, ANN)의 해석 가능성을 높이기 위해 신경망과 논리(logic) 간의 다리 역할을 하는 새로운 접근법을 제안합니다. 특히 ReLU 층과 여러 선형 층으로 구성된 ANN의 내부 동작을 보다 명확하게 설명할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 ANN의 입력 공간을 여러 네트워크 파티션 셀로 분해하는 것입니다. 각 파티션 셀은 입력 값을 분류 출력 값으로 매핑하는 선형 결합을 나타냅니다. 이러한 파티션 셀의 선형 맵을 해석하기 위해 논리 표현식을 사용하며, 논리 표현을 이진 논리 트리(binary logic tree) 형태로 제시합니다.

- **Performance Highlights**: 제안된 방법을 통해 ANN의 분류 결과를 해석할 수 있는 접근법이 마련되며, 이는 인공지능의 신뢰성을 높일 수 있는 중요한 단계로 평가됩니다. 이를 통해 신경망의 은닉층에서 발생하는 복잡한 내부 결정 논리를 직관적으로 이해할 수 있는 가능성이 제시됩니다.



### LLM-3D Print: Large Language Models To Monitor and Control 3D Printing (https://arxiv.org/abs/2408.14307)
- **What's New**: 이 연구는 Fused Deposition Modeling (FDM)에서 발생하는 인쇄 결함을 자동으로 감지하고 수정하기 위해 사전 훈련된 Large Language Models (LLMs)를 활용하는 새로운 프로세스 모니터링 및 제어 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 인쇄 품질을 평가하기 위해 각 층 또는 인쇄 세그먼트 후 캡처된 이미지를 분석하며, 실패 모드를 식별하고 관련 파라미터를 쿼리합니다. LLM은 감지된 결함에 대해 교정 조치 계획을 생성하고 실행합니다.

- **Performance Highlights**: 테스트 결과, LLM 기반의 시스템은 3D 인쇄에서 발생하는 일반적인 오류를 정확히 식별하고, 실패를 유발하는 파라미터를 효과적으로 판단하여 인간 개입 없이도 이를 자율적으로 수정할 수 있음을 보여주었습니다.



### May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels (https://arxiv.org/abs/2408.14284)
Comments:
          25 pages, 5 figures. Accepted at the The 35th British Machine Vision Conference 2024 (BMVC 2024), Glasgow, UK

- **What's New**: 본 연구에서는 노이즈가 있는 레이블을 가진 데이터를 처리하기 위한 새로운 접근법인 Alternate Experience Replay (AER)를 소개합니다. 이 방법은 메모리 버퍼에서 클린(깨끗한), 복잡한, 노이즈가 있는 샘플을 명확하게 구분하기 위해 망각(Forgetting)의 특성을 활용합니다.

- **Technical Details**: AER는 클린 및 노이즈 샘플 간의 구분을 강화하기 위해 메모리 버퍼 학습 및 망각 단계를 번갈아 진행하는 새로운 Continual Learning (CL) 최적화 방안을 제시합니다. 이와 함께 Asymmetric Balanced Sampling (ABS)라는 새로운 샘플 선택 전략을 도입하여 현재 작업의 순수성을 유지하면서 과거의 중요한 샘플을 보존합니다.

- **Performance Highlights**: 다양한 노이즈 유형 및 속도에 대한 실험을 통해 AER와 ABS의 효과를 검증했으며, 기존의 손실 기반 정제 전략에 비해 평균 4.71% 높은 정확도를 기록하는 등 성능 향상을 입증했습니다.



### Uncertainties of Latent Representations in Computer Vision (https://arxiv.org/abs/2408.14281)
Comments:
          Doctoral thesis

- **What's New**: 이번 논문은 사전 훈련된 컴퓨터 비전 모델의 잠재적 표현 벡터에 불확실성 추정치를 통합하여 신뢰할 수 있는 머신러닝의 중요한 기초인 불확실성 정량화를 쉽게 접근할 수 있도록 합니다.

- **Technical Details**: 논문에서는 Monte-Carlo InfoNCE (MCInfoNCE)와 같은 확률 및 의사결정 이론에 기반한 새로운 접근 방식을 제안하며, 이론적 및 실증적 질문들을 탐구합니다. 또한, 비가시적인 잠재 표현에 대한 불확실성이 실제로 입증 가능하다는 주장을 제공합니다.

- **Performance Highlights**: 최종적으로, 경량화된 표현 불확실성을 대규모 컴퓨터 비전 모델 플랫폼에서 미리 훈련하여 보지 못한 데이터셋에도 제로샷(zero-shot) 방식으로 전이할 수 있는 능력을 갖춘 결과들을 제시하며, 이는 앞으로의 연구자들이 불확실성 정량화에 더 쉽게 접근할 수 있도록 합니다.



### Text3DAug -- Prompted Instance Augmentation for LiDAR Perception (https://arxiv.org/abs/2408.14253)
Comments:
          Accepted at the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: Text3DAug는 텍스트를 기반으로 인스턴스를 생성하고 주석을 추가하는 최초의 완전 자동화된 방법으로, 데이터에 대한 레이블 없이 3D 모델 생성에 generative models를 활용합니다.

- **Technical Details**: 기존의 인스턴스 증강 방법은 레이블이 필요하고, 시간이 많이 소요되는 CAD 모델링 및 수작업 데이터 주석이 필요했습니다. Text3DAug는 이러한 요구를 없애고, 다양한 LiDAR 센서에서 신뢰할 수 있는 인스턴스 생성과 배치를 자동으로 수행하여 센서에 관여하지 않는 효과적인 데이터 증강을 제공합니다.

- **Performance Highlights**: LiDAR 분할 및 탐지 벤치마크에서 Text3DAug는 기존의 방법들을 보완하거나 독립적으로 사용할 수 있는 효과적인 성능을 보여주었으며, 기존 방법들의 특정 단점을 극복하면서도 동등하거나 더 나은 성능을 발휘했습니다.



### Beyond Few-shot Object Detection: A Detailed Survey (https://arxiv.org/abs/2408.14249)
Comments:
          43 pages, 8 figures

- **What's New**: 이 서베이 논문은 기존의 전통적인 경우보다 적은 양의 데이터로 학습할 수 있는 ‘few-shot object detection (FSOD)’에 대한 포괄적인 검토를 제공합니다. 특히 표준 FSOD, 일반화 FSOD, 증분 FSOD, 오픈셋 FSOD, 도메인 적응 FSOD와 같은 다양한 FSOD 설정에 중점을 두었습니다.

- **Technical Details**: FSOD는 적은 수의 주석이 달린 예제만을 가지고 모델이 새로운 객체 카테고리에 빠르게 적응할 수 있도록 합니다. 이 서베이는 FSOD의 여러 변형들을 고찰하며, 각 접근법의 세부적인 평가 프로토콜을 분석합니다.

- **Performance Highlights**: FSOD 접근법은 의료 영상 진단, 야생동물 보호, 산업 검사, 보안 감시 등 다양한 실제 응용 분야에서 유용하게 사용될 수 있으며, 제한된 데이터 상황에서도 높은 성능을 유지할 수 있는 모델 개발의 중요성을 강조합니다.



### Celtibero: Robust Layered Aggregation for Federated Learning (https://arxiv.org/abs/2408.14240)
- **What's New**: Celtibero는 Federated Learning (FL)에서의 poisoning 공격에 대한 새로운 방어 메커니즘으로, 레이어 집계를 통합하여 적대적 조작에 대한 강인성을 강화합니다. 이를 통해 i.i.d 및 non-i.i.d 조건 모두에서 효과적입니다.

- **Technical Details**: Celtibero는 샘플링된 로컬 모델 업데이트를 통한 다양한 레이어에서의 부분 집계를 수행하며, 각 레이어에서 수집된 benign 업데이트들을 결합하여 글로벌 모델을 형성합니다. 이를 통해 FL 모형에서 공격의 영향력을 최소화합니다. 특히, 여러 유형의 poisoning 공격에 대한 방어 능력이 뛰어납니다.

- **Performance Highlights**: Celtibero는 MNIST와 IMDB 데이터셋에서의 실험을 통해 높은 주요 작업 정확도 (MTA)를 지속적으로 달성하며, 공격 성공률 (ASR)을 최소화했습니다. 기존의 FL-Defender, LFighter, FLAME과 비교해 우수한 성능을 보여줍니다.



### DSTI at LLMs4OL 2024 Task A: Intrinsic versus extrinsic knowledge for type classification (https://arxiv.org/abs/2408.14236)
Comments:
          8 pages, 4 figures, accepted for the LLMs4OL challenge at the International Semantic Web Conference (ISWC) 2024

- **What's New**: 본 연구에서는 semantic towers라는 외부 지식 표현 방법을 소개하고, 이를 대규모 언어 모델에서의 내재적 지식과 비교하여 본다. 연구의 결과는 외부 지식 기반의 모델이 성능과 의미적 기반 간의 트레이드오프를 보임을 발견하였다.

- **Technical Details**: 경량화된 언어 모델(flT5-small 클래스)을 사용하여 WordNet과 GeoNames 데이터셋에서의 태스크 A를 포함한 실험을 수행하였다. 이 연구에서 제안하는 semantic towers는 domain semantic primitive를 활용하여 구축되며, Wikidata Query Service를 통해 각각의 도메인에서 의미 정보를 수집하여 구축된다. 최종적으로 이러한 primitive들은 벡터 임베딩으로 변환되어 MongoDB에 저장된다.

- **Performance Highlights**: 실험 결과, WN1 및 GN1 모델은 각각 WN2 및 GN2에 비해 약 10%의 성능 향상을 보였다. 그러나 외부 지식 기반인 semantic towers의 사용이 모델의 잘못된 응답을 초래할 수 있지만, 특정 의미 개념을 효과적으로 정립할 수 있음을 발견하였다.



### Gallery-Aware Uncertainty Estimation For Open-Set Face Recognition (https://arxiv.org/abs/2408.14229)
- **What's New**: 이 논문은 오픈 세트 얼굴 인식(Open-set Face Recognition, OSFR)에서 불확실성 추정을 위한 새로운 방법론을 제안합니다. 기존 연구는 주로 얼굴 검증(face verification)에 초점을 맞춰, OSFR의 불확실성을 잘 다루지 못했습니다. 본 연구는 이미지 임베딩과 갤러리(gallery) 내 클래스 중첩으로 인한 불확실성을 고려하는 Bayesian 확률 모델을 도입합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 불확실성 출처를 인식합니다: (1) 갤러리 클래스 중첩으로 인한 갤러리 불확실성과 (2) 얼굴 임베딩의 불확실성. 이를 통해 임베딩 분포의 Bayesian 모델을 사용하여 불확실성을 추정합니다. 논문에서는 Laplace 근사를 통해 확률 분포의 후방 확률을 계산하고, vMF(von Mises–Fisher) 및 Power 분포를 사용하여 각 테스트 이미지에 대한 카테고리 분포를 정의합니다. 이로써, 주어진 이미지의 품질과 임베딩 상대적 위치를 함께 고려합니다.

- **Performance Highlights**: 제안된 방법은 IJB-C 데이터셋과 같은 도전적인 오픈 세트 얼굴 인식 데이터셋에서 테스트 되었으며, 이미지 품질 정보만을 기반으로 한 불확실성 추정 방법들보다 오류 인식에서 더 우수한 성능을 보였습니다. 또한 새로운 오픈 세트 인식 프로토콜을 개발하여, 얼굴 외의 범위에서도 적용 가능성을 제시합니다.



### MagicMan: Generative Novel View Synthesis of Humans with 3D-Aware Diffusion and Iterative Refinemen (https://arxiv.org/abs/2408.14211)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문은 MagicMan을 소개하며, 이는 단일 참조 이미지로부터 고품질의 새로운 뷰 이미지를 생성하기 위해 설계된 인체 전용 다중 뷰 확산 모델입니다.

- **Technical Details**: MagicMan은 사전 훈련된 2D diffusion 모델과 SMPL-X 모델을 활용하여 3D 일관성을 증진시킵니다. 혼합 다중 뷰 주의(hybrid multi-view attention) 메커니즘을 도입하여 정보 교환을 간소화하고, 기하학적으로 인식 가능한 이중 분기(dual branch)를 통해 RGB와 노멀 도메인에서 동시 생성을 수행합니다. SMPL-X의 정확성을 개선하기 위한 반복적 정제(iterative refinement) 전략도 제안되었습니다.

- **Performance Highlights**: 광범위한 실험 결과가 MagicMan이 기존 접근 방식보다 새로운 뷰 합성과 3D 인간 재구성 작업에서 현저하게 우수하다는 것을 나타냅니다.



### Representative Arm Identification: A fixed confidence approach to identify cluster representatives (https://arxiv.org/abs/2408.14195)
Comments:
          We analyse a clustered multi-armed bandit formulation, where the learning objective is to identify representative arms from each cluster, in a fixed confidence setting

- **What's New**: 기존의 다수의 손잡이와 보상 분포에 관한 문제를 다루는 연구로, RAI(대표 손잡이 인식) 문제를 정의하고, 이를 해결하기 위한 두 가지 알고리즘을 제안했습니다.

- **Technical Details**: RAI 문제는 손잡이를 미리 정의된 크기의 클러스터로 나누고 각 클러스터의 특정 수의 손잡이를 성공적으로 확인하는 것을 목표로 합니다. 이 연구에서는 각 알고리즘의 샘플 복잡성(sample complexity)의 하한을 제공하고, 신뢰 구간(confidence intervals)을 기반으로 두 가지 알고리즘을 제안합니다. 알고리즘 결과는 이론적 경계와 비교하여 유사한 순서를 따릅니다.

- **Performance Highlights**: 제안된 알고리즘은 합성 데이터 및 실제 데이터 세트를 기반으로 우수한 성능을 보여주었습니다. 특히, 알고리즘은 기존의 다른 메서드에 비해 대부분의 경우에서 더 좋은 성과를 나타냈습니다.



### Robot Navigation with Entity-Based Collision Avoidance using Deep Reinforcement Learning (https://arxiv.org/abs/2408.14183)
Comments:
          14 pages, 5 figures

- **What's New**: 이 연구에서는 동적 환경에서 자율 로봇의 안전하고 효율적인 내비게이션을 위한 새로운 방법론을 제시합니다. 이는 로봇이 다양한 유형의 인물 및 장애물과 상호작용할 때 특정 안전 요구 사항을 기반으로 하여 충돌 회피 능력을 개선합니다.

- **Technical Details**: 제안된 방법은 Entity-Based Collision Avoidance using Deep Reinforcement Learning (EB-CADRL)라는 접근법을 사용하며, 이는 로봇이 환경의 다양한 엔티티에 대해 다르게 반응하도록 합니다. 새로운 보상 함수는 로봇이 위험한 엔티티와의 충돌을 피하고 목표에 가까워지도록 유도합니다. 이 보상 함수는 성인, 자전거 이용자, 어린이 및 고정 장애물과의 충돌에 따라 로봇의 행동에 다르게 패널티를 부여합니다.

- **Performance Highlights**: 시뮬레이션 실험을 통해 제안된 접근법은 기존의 내비게이션 및 충돌 회피 방법들보다 지속적으로 우수한 성능을 보였으며, 복잡한 환경에서도 훈련이 가능하였습니다. 이는 자율 로봇의 안전성과 효율성을 높이는 데 기여합니다.



### I2EBench: A Comprehensive Benchmark for Instruction-based Image Editing (https://arxiv.org/abs/2408.14180)
Comments:
          Tech report, 39 pages, 41 figures

- **What's New**: 이 논문에서는 Instruction-based Image Editing (IIE) 분야에서의 모델 성능 평가를 위한 포괄적인 벤치마크인 I2EBench를 제안합니다. 이 벤치마크는 2000개 이상의 편집 이미지와 4000개 이상의 다양한 원본 지침을 포함하고 있으며, 고급 및 저급 평가 차원을 포함한 총 16개의 평가 차원을 제공합니다.

- **Technical Details**: I2EBench는 사용자의 인식에 맞춘 평가를 위해 대규모 사용자 연구를 수행하였으며, 다양한 편집 모델의 강점과 약점을 분석하여 연구 통찰을 제공합니다. 평가 차원은 고급 편집과 저급 편집으로 구분되며, 이미지의 세부 사항 편집과 특정 영역 편집을 포함합니다. 또한, I2EBench는 인간 평가자에 의한 점수 수집을 통해 높은 상관관계를 발견했습니다.

- **Performance Highlights**: I2EBench는 IIE 모델의 포괄적인 성능 평가를 지원하며, 기존 모델의 장단점을 체계적으로 평가하여 향후 개선 방향을 제시합니다. 이 자료는 오픈 소스로 제공되어 다른 연구자들이 벤치마크를 활용할 수 있도록 하며, 공정한 비교와 커뮤니티 발전을 촉진할 것으로 기대됩니다.



### SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher (https://arxiv.org/abs/2408.14176)
Comments:
          Accepted to ECCV'24

- **What's New**: 이 논문에서는 SwiftBrush, 유명한 일단계 텍스트-이미지 확산 모델의 성능을 향상시켜 다단계 Stable Diffusion 모델과 경쟁할 수 있도록 목표합니다. 본 연구는 SwiftBrush와 SD Turbo 간의 품질-다양성(trade-off)을 탐구하여, 두 모델의 장점 조합과 새로운 훈련 방법론인 clamped CLIP loss의 도입을 통해 하나의 모델이 다단계 모델을 초월할 수 있다는 것을 보여줍니다.

- **Technical Details**: 본 연구에서는 SwiftBrush와 SD Turbo를 비교하면서, SwiftBrush는 더 다양한 출력 이미지를 제공하는 반면 SD Turbo는 높은 품질의 이미지를 생성한다는 사실에 주목했습니다. 또한, LoRA의 효율적인 훈련이 결합된 새로운 훈련 방법론을 제안하여 학생 모델이 교사 모델을 초과하는 성능을 거두도록 했습니다. 극복된 품질 메트릭은 FID (Fréchet Inception Distance)를 포함입니다.

- **Performance Highlights**: 최종적으로, 저희는 FID 점수 8.77을 달성하며, 다단계 모델을 초과한 첫 번째 일단계 모델을 제안했습니다. 이후 추가적인 정규화로 FID 점수가 8.14로 향상되었고, 이는 효율적이고 고품질의 텍스트-이미지 모델 분야에서 새로운 기준을 설정했습니다.



### Dynamic Pricing for Electric Vehicle Charging (https://arxiv.org/abs/2408.14169)
Comments:
          12 pages

- **What's New**: 본 논문은 전기차(EV) 충전소의 가격 책정 문제를 다중 목표 최적화(multi-objective optimization)를 통해 해결하는 새로운 접근 방식을 제안합니다. 이 연구는 수익, 서비스 품질(QoS), 평균 대 피크 비율(PAR)과 같은 여러 상충하는 목표를 동시에 고려하여 최적의 해결책을 찾습니다.

- **Technical Details**: 기존의 연구들은 일반적으로 단일 목표나 선형 결합 방식을 사용하여 가격 책정을 다루었습니다. 본 연구는 Non-dominated Sorting Genetic Algorithms (NSGA) II 및 NSGA III를 통해 효율적으로 Pareto-optimal 솔루션을 찾아 다중 목표 최적화를 수행합니다. 3개의 주요 요소로 구성된 동적 가격 책정 모델을 적용하며, Bayesian 모델을 사용하여 수요와 가격 간의 관계를 정량화합니다.

- **Performance Highlights**: 본 논문의 접근 방식은 California의 두 충전소에서 실증적으로 검증되었습니다. 이 연구는 실세계 데이터를 기반으로 효율적인 가격 책정 솔루션을 제공하고, 전기차 충전소 운영자에게 수익 증대와 그리드 안정성 향상에 도움을 줄 것으로 기대됩니다.



### Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning (https://arxiv.org/abs/2408.14158)
Comments:
          This is the preprint version of the paper accepted for presentation at the 2024 International Conference for High Performance Computing, Networking, Storage, and Analysis (SC'24). \c{opyright} 2024 IEEE. Personal use of this material is permitted. For other uses, permission from IEEE must be obtained. Please refer to IEEE Xplore for the final published version

- **What's New**: 본 논문은 깊은 학습(Deep Learning, DL) 및 대규모 언어 모델(Large Language Models, LLMs)의 발전으로 인한 컴퓨터 자원의 수요 증가 문제를 해결하기 위해 Fire-Flyer AI-HPC 아키텍처를 소개합니다. 새로운 하드웨어-소프트웨어 공동 설계 프레임워크를 통해 DL 훈련을 위한 구축 비용을 절반으로 줄이고 에너지 소비를 40%로 감소시켰습니다.

- **Technical Details**: Fire-Flyer 2는 10,000개의 PCIe A100 GPU를 배치하여 NVIDIA DGX-A100과 비교했을 때 비용 효율성과 낮은 CO2 배출량을 자랑합니다. 다양한 네트워크 조정을 통해 혼잡을 방지하는 두 층의 Fat-Tree Network 구조와 비동기 allreduce 통신을 활용한 HFReduce 소프트웨어를 통해 계산-통신 겹침을 달성했습니다. HaiScale, 3FS 및 HAI-Platform과 같은 소프트웨어 스택을 통해 계산 및 통신의 병렬화를 최적화하였습니다.

- **Performance Highlights**: 실제로 Fire-Flyer AI-HPC 아키텍처는 DL 훈련 시 성능을 크게 향상시켰으며, 향후 AI-HPC 발전에 중요한 통찰을 제공할 것으로 기대됩니다. 강력한 하드웨어 장애 처리 메커니즘과 장애 복구 기능을 통해 시스템의 가용성과 효율성도 높였습니다.



### Explaining Vision-Language Similarities in Dual Encoders with Feature-Pair Attributions (https://arxiv.org/abs/2408.14153)
- **What's New**: 이번 연구에서는 CLIP 모델과 같은 Dual encoder 아키텍처의 예측을 입력 간의 feature-pair 상호작용으로 귀속할 수 있는 방법을 제안하였습니다. 이를 통해 모델이 입력된 두 데이터를 비교하는 방식을 보다 깊이 이해할 수 있습니다.

- **Technical Details**: 제안된 방법은 어떠한 미분 가능 Dual encoder 모델에 대해서도 입력 간의 상호작용을 설명할 수 있는 일반적인 feature-pair 귀속값을 계산하게 합니다. 특히, 이 방법은 훈련된 모델의 수정 없이 적용 가능하며, 시각-언어 모델에 적용되는 과정에서 세밀한 상호작용을 포착할 수 있습니다.

- **Performance Highlights**: 모델은 주어진 데이터 배치 내에서 세부적인 객체 클래스 간의 지식 격차를 식별하고, in-domain 훈련 후 성능 향상을 모니터링할 수 있는 능력을 가지고 있어, CLIP 모델의 시각-언어 기초 능력이 객체 클래스에 따라 다양하게 나타남을 보여줍니다.



### Exploring the Potential of Large Language Models for Heterophilic Graphs (https://arxiv.org/abs/2408.14134)
Comments:
          Under review

- **What's New**: 이 논문은 GNN(그래프 신경망)을 위해 LLM(대형 언어 모델)의 활용을 탐구하며, 이 기술을 통해 이종(heterophilic) 그래프의 특성을 효과적으로 모델링할 수 있는 두 단계 프레임워크인 LLM4HeG를 제안합니다.

- **Technical Details**: 첫 번째 단계에서는 LLM을 미세 조정하여 텍스트 정보 기반으로 동종(homophilic) 및 이종(heterophilic) 엣지를 식별합니다. 두 번째 단계에서는 GNN에서 엣지 유형에 따라 메시지 전파를 관리하여 노드 특성 및 구조에 따른 적응형 접근법을 구현합니다. 또한, LLM을 효율적인 소형 언어 모델(SLM)로 증류하여 계산 비용을 절감합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 실제 이종 데이터셋에서 노드 분류 성능을 향상시키며, 이종 그래프에서 LLM을 효과적으로 활용하는 가능성을 보여주었습니다.



### Contrastive Learning Subspace for Text Clustering (https://arxiv.org/abs/2408.14119)
- **What's New**: 이 논문에서는 텍스트 클러스터링을 위해 클러스터별 관계를 모델링하는 새로운 접근법인 Subspace Contrastive Learning (SCL)을 제안합니다. 기존의 대조 학습 기반 텍스트 클러스터링 기법은 인스턴스 간 의미적 유사성 관계만 모델링하였으나, SCL은 이러한 제한을 극복합니다.

- **Technical Details**: SCL은 두 가지 주요 모듈로 구성되어 있습니다: (1) 가상 긍정 샘플을 구성하는 자기 표현(self-expressive) 모듈과 (2) 텍스트 간의 클러스터별 관계를 캡처하기 위해 더 효과적인 하위 공간(discriminative subspace)을 학습하는 대조 학습 모듈입니다.

- **Performance Highlights**: 실험 결과, SCL 방법은 여러 텍스트 클러스터링 데이터셋에서 기존의 최첨단 방법보다 우수한 성능을 보이는 동시에 긍정 샘플 구성의 복잡성을 줄였습니다.



### Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects (https://arxiv.org/abs/2408.14090)
- **What's New**: 이 논문은 Alps, Leonardo, LUMI라는 세 가지 슈퍼컴퓨터의 멀티 GPU 간의 성능을 처음으로 대규모로 특성화한 연구를 제공합니다. 각 시스템의 서로 다른 아키텍처에서 데이터를 효과적으로 이동시키는 다양한 기술적 문제를 분석합니다.

- **Technical Details**: 이 연구는 각 슈퍼컴퓨터의 노드 아키텍처, 나는 네트워크 연결성, 성능 벤치마크 등을 포함하여 GPU 간의 데이터 이동을 위한 다양한 소프트웨어 솔루션과 상호 연결 방법론을 체계적으로 분석합니다. 특히, CCL, GPU-Aware MPI 등에서 발생하는 비효율성을 강조합니다.

- **Performance Highlights**: 성능 평가 결과, 여전히 활용되지 않은 대역폭이 존재하며, 네트워크 최적화와 소프트웨어 최적화의 기회가 많음을 보여줍니다. 이러한 발견은 시스템 설계자, 연구자 및 소프트웨어 개발자들이 대규모 멀티 GPU 시스템에서 데이터 전송을 최적화하고 최신 시스템을 최대한 활용하는 데 유용한 통찰력을 제공할 것입니다.



### SONICS: Synthetic Or Not -- Identifying Counterfeit Songs (https://arxiv.org/abs/2408.14080)
- **What's New**: 본 논문은 AI로 생성된 노래의 탐지를 위한 SONICS라는 새로운 데이터셋을 소개하며, 이는 97,000개 이상의 노래와 49,000개 이상의 합성(song generated by AI)에 대한 정보를 담고 있다. 또한, AI 생성 노래의 모든 구성 요소가 AI로 만들어질 수 있는 가능성을 탐구한다.

- **Technical Details**: SONICS 데이터셋은 노래의 긴 맥락적 관계를 모델링하는 것이 중요하다는 점을 강조하며, 모든 구성 요소가 AI로 생성된 끝에서 끝까지(end-to-end) 인공 노래 탐지를 위한 데이터셋을 제공한다. 제안된 SpecTTTra 모델은 길고 긴 맥락을 효과적으로 처리할 수 있으며, 기존 CNN 및 Transformer 모델에 비해 3배 빠르고 6배 더 메모리 효율적이다.

- **Performance Highlights**: 제안된 SONICS 데이터셋과 SpecTTTra 모델은 인공 노래 탐지 작업에서 더욱 정확하고 효율적인 성능을 제공하며, 기존 모델과의 비교를 통해 그 유용성을 강조하고 있다.



### HAPM -- Hardware Aware Pruning Method for CNN hardware accelerators in resource constrained devices (https://arxiv.org/abs/2408.14055)
Comments:
          8 pages, 7 figure, thesis for the title of Electronic Engineer attained in 2021 at the Universidad Tecnologica Nacional (UTN), Argentina

- **What's New**: 이 논문은 FPGA(재구성 가능 게이트 배열) 장치에서 구현할 준비가 된 범용 하드웨어 아키텍처를 제안하며, 다양한 신경망 아키텍처를 지원하고, 하드웨어 가속기에서의 스케줄링 속성을 활용하는 맞춤형 가지치기 기법을 도입합니다.

- **Technical Details**: 하드웨어 아키텍처는 Systolic Array 구조를 통해 효율성을 극대화하며, MACC(곱셈 및 누산) 연산을 병렬화하여 컨볼루션 연산의 성능을 향상시킵니다. 알고리즘의 최적화를 위해 피젯 포인트에서 루프 언롤링 및 루프 타일링 기법을 사용하며, 내부 데이터 재사용을 극대화하기 위해 고안된 처리 요소(PE)를 활용します.

- **Performance Highlights**: HAPM(하드웨어 인지 가지치기 방법)을 적용한 경우, 표준 가지치기 기술에 비해 이미지당 추론 시간이 45% 향상되었습니다. CIFAR-10 데이터셋의 이미지를 분류하는 ResNet 신경망 아키텍처에서 최대 7.468 GOPs의 성능을 기록하였습니다.



### Beyond Detection: Leveraging Large Language Models for Cyber Attack Prediction in IoT Networks (https://arxiv.org/abs/2408.14045)
- **What's New**: IoT 기기에서의 사이버 공격 예측을 위한 새로운 접근 방식이 제안되었습니다. 이 방식은 대규모 사이버 공격에 대응하기 위해 LLM과 LSTM 네트워크를 결합한 예측 프레임워크입니다.

- **Technical Details**: 제안된 프레임워크는 Generative Pre-trained Transformer (GPT) 모델과 Bidirectional Encoder Representations from Transformers (BERT) 모델을 사용하여 예측된 네트워크 트래픽을 평가합니다. 이 프레임워크는 CICIoT2023 IoT 공격 데이터셋을 활용하여 평가되었으며, LSTM 분류기를 통해 악성 패킷을 식별합니다.

- **Performance Highlights**: 제안된 접근 방식은 98%의 높은 정확도로 침입을 성공적으로 탐지하는 성능을 보였습니다. 이는 IoT 사이버 보안 문제에 대한 강력한 해결책을 제공합니다.



### PAGE: Parametric Generative Explainer for Graph Neural Network (https://arxiv.org/abs/2408.14042)
- **What's New**: PAGE라는 새로운 파라미터 기반 생성 해석 프레임워크를 소개하며, 이는 그래프 신경망에 대한 신뢰할 수 있는 설명을 제공할 수 있습니다. 이 방법은 사전 지식이나 내부 세부 정보를 필요로 하지 않습니다.

- **Technical Details**: PAGE는 오토인코더(auto-encoder)를 훈련하여 설명 가능한 하위 구조를 생성합니다. 이 과정에서 잠재 공간(latent space)의 특징 차원을 축소하여 모델의 출력을 유도하는 인과적 특징(causal features)을 추출합니다. 이를 위해 추가적인 판별자(discriminator)를 도입해 잠재 인과적 특징과 모델 출력을 연결합니다.

- **Performance Highlights**: PAGE는 샘플 규모에서 작동하여 기존 방법보다 효율성이 크게 향상되었습니다. 실험 결과, 당 연구는 기존 방법들보다 높은 신뢰성과 정확성을 보여주며, 다양한 지표에서 우수한 성과를 나타냅니다.



### SurGen: Text-Guided Diffusion Model for Surgical Video Generation (https://arxiv.org/abs/2408.14028)
- **What's New**: 본 연구에서는 SurGen이라는 텍스트 기반 확산 모델을 소개합니다. 이는 기존의 외과 비디오 생성 모델들 중 가장 높은 해상도(720 x 480 pixels)와 긴 지속시간(49 frames)을 자랑합니다. 이 모델은 외과 절차의 다양한 단계를 지정하는 텍스트 프롬프트에 의해 조건화되어 비디오를 생성합니다.

- **Technical Details**: SurGen은 CogVideoX를 기반으로 하여 텍스트 프롬프트를 사용하여 외과 영상을 생성합니다. 모델 학습에는 200,000개의 독특한 비디오-텍스트 쌍이 사용되며, 각 비디오는 특정 외과 단계에 맞는 프롬프트와 쌍을 이룹니다. 각 비디오는 3D Variational Autoencoder(3D VAE)와 Denoising Video Transformer를 통해 생성됩니다. 또한 2억 개의 파라미터를 가진 텍스트 조건형 비디오 변환기를 사용하여 비디오의 공간 및 시간 정보를 처리합니다.

- **Performance Highlights**: SurGen 모델은 Fréchet Inception Distance(FID) 지표에서 79.9163을 기록하여, 사전 훈련된 모델의 260.0301보다 현저하게 개선된 시각적 품질과 다양성을 나타냅니다. Fréchet Video Distance(FVD)에서도 752의 개선된 성능을 보였습니다.



### Video-CCAM: Enhancing Video-Language Understanding with Causal Cross-Attention Masks for Short and Long Videos (https://arxiv.org/abs/2408.14023)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 연구에서는 비디오-MLLM(Video-MLLM)이라고 불리는 새로운 다중 모달 대형 언어 모델(MLLM)인 Video-CCAM을 제안합니다. Video-CCAM은 비디오와 언어 이해에서 뛰어난 성능을 보여주며, 비디오의 프레임 수에 관계없이 처리할 수 있는 유연한 모델입니다.

- **Technical Details**: 비디오-CCAM은 비주얼 인코더(visual encoder), LLM, 그리고 크로스 어텐션(cross-attention) 메커니즘을 활용한 프로젝트로 구성되어 있습니다. 특히, CCAM(causal cross-attention masks)을 도입하여 비디오의 시간적 순서를 고려하여 모델의 비디오 이해 능력을 향상시킵니다. 이 모델은 차세대 비디오-MLLM으로, 이미지와 16 프레임 비디오만을 사용하여 훈련된 후에도 긴 비디오의 이해에 적응할 수 있는 뛰어난 능력을 가지고 있습니다.

- **Performance Highlights**: Video-CCAM은 표준 비디오 벤치마크인 MVBench 및 VideoChatGPT-QA에서 1위, 2위, 3위의 성과를 달성하였으며, 긴 비디오에 대한 벤치마크에서도 뛰어난 점수를 기록하였습니다. 특히, VideoVista와 MLVU에서는 각각 1위와 2위에 오르며 모든 오픈 소스 비디오-MLLM 중에서 가장 높은 성능을 보였습니다.



### Pixel-Aligned Multi-View Generation with Depth Guided Decoder (https://arxiv.org/abs/2408.14016)
- **What's New**: 이 논문은 이미지를 바탕으로 다중 뷰 생성의 픽셀 정렬 문제를 해결하기 위해 새로운 방법을 제안합니다. 기존의 다중 뷰 생성 모델에서는 VAE와 U-Net을 활용하였으나, 픽셀 정렬 문제로 인해 결과물이 만족스럽지 못했습니다. 본 연구에서는 depth-truncated epipolar attention을 도입하여 고해상도에서의 조합이 가능하도록 하였으며, 이를 통해 다중 뷰 이미지 간의 정렬을 개선하였습니다.

- **Technical Details**: 이 방법은 VAE (Variational Autoencoder) 디코더에 다중 뷰 이미지간의 attention layer를 포함합니다. 구체적으로, depth-truncated epipolar attention을 통해 모델이 공간적으로 인접한 영역에 집중할 수 있도록 하여 메모리를 효율적으로 사용할 수 있게 합니다. 불확실한 깊이 추정으로 인한 일반화를 높이기 위해 훈련 중 깊이 입력을 섞고, 추론 시에는 다중 뷰-3D 재구성 방법인 NeuS를 활용하여 거친 깊이를 얻습니다.

- **Performance Highlights**: 제안된 방법은 기존의 상태-of-the-art 다중 뷰 생성 기법들과 비교하였을 때 PSNR, SSIM, LPIPS 및 재구성된 3D 객체의 일치 수와 같은 수치적 기준에서 높은 성능을 보였습니다. 이로 인해 본 방법이 3D 재구성 작업의 하위 작업에서 효과적임을 입증하였습니다.



### Optimizing TD3 for 7-DOF Robotic Arm Grasping: Overcoming Suboptimality with Exploration-Enhanced Contrastive Learning (https://arxiv.org/abs/2408.14009)
Comments:
          4 pages, 2 figures, IEEE-ICKII-2024

- **What's New**: 본 논문은 TD3(Twin Delayed Deep Deterministic Policy Gradient)와 같은 actor-critic 기반 강화 학습 알고리즘에서 탐색 부족 문제를 해결하기 위해 새로운 Exploration-Enhanced Contrastive Learning (EECL) 모듈을 제안합니다.

- **Technical Details**: EECL 모듈은 탐색된 이전 상태들을 버퍼에 저장하고 K차원 트리(KDTree) 프레임워크를 이용하여 유클리드 거리(Euclidean distance)를 비교함으로써 새로운 상태를 식별합니다. 새로운 상태를 탐색할 때 탐색 보상을 부여하며, 이러한 보상은 TD3 알고리즘에 통합되어 Q-learning 과정이 더 효과적인 전략 최적화를 할 수 있도록 합니다.

- **Performance Highlights**: 우리의 방법을 robosuite panda lift 작업에서 평가한 결과, 테스트된 환경에서 TD3의 기준선과 비교하여 효율성과 수렴 속도에서 상당한 개선을 보여주었습니다.



### LMM-VQA: Advancing Video Quality Assessment with Large Multimodal Models (https://arxiv.org/abs/2408.14008)
- **What's New**: 본 논문에서는 최초의 대규모 다중 모달 비디오 품질 평가 모델(LMM-VQA)을 제안하며, 이는 비디오 품질 평가(VQA) 작업을 해결하기 위해 새로운 시공간 시각 모델링 전략을 도입합니다.

- **Technical Details**: LMM-VQA 모델은 질적 회귀 문제를 Q&A(질문과 응답) 작업으로 재구성하고, 질적 특징을 추출하기 위해 시공간 비전 인코더를 설계합니다. 모델은 비디오의 시공간 특성을 추출하고, 이러한 특징들을 대형 언어 모델(LLM)에 입력하여 품질 점수와 수준을 생성합니다.

- **Performance Highlights**: 다양한 VQA 벤치마크에서 LMM-VQA는 기존 방법보다 평균 5%의 일반화 능력 향상을 보이며, 비디오 이해 작업에서도 우수한 성능을 나타냅니다.



### Dual-CBA: Improving Online Continual Learning via Dual Continual Bias Adaptors from a Bi-level Optimization Perspectiv (https://arxiv.org/abs/2408.13991)
- **What's New**: 온라인 지속 학습(online continual learning)에서 모델은 변화하는 분포에 쉽게 적응하지 못하고 이전에 학습한 지식을 잊어버리기 쉽다. 이를 해결하기 위해, Continual Bias Adaptor(CBA)라는 새로운 생체 수준의 프레임워크를 제안한다. 이 모듈은 훈련 중 분포 변동에 적응하고, 모든 학습 작업의 안정적인 통합을 가능하게 한다.

- **Technical Details**: CBA 모듈은 class-specific 방식으로 분포 변화를 조정하여 안정성 격차 문제를 악화시키며, 동시에 온라인 지속 학습에서의 지속적 테스트 요구를 충족하지 못할 수 있다. 이를 해결하기 위해, class-agnostic CBA 모듈을 제안하여 새로운 작업과 이전 작업의 후방 확률을 개별적으로 집계한다. 두 가지 CBA 모듈을 통합한 Dual-CBA 모듈은 재앙적 분포 변동에 적응하면서 온라인 CL의 실시간 테스트 요구를 충족할 수 있다.

- **Performance Highlights**: 제안된 방법의 효과성을 검증하기 위해, 이론적으로 재앙적 분포 변동을 완화하는 방법에 대한 통찰을 제공하고, 네 가지 리허설 기반의 기준선과 세 가지 공적 지속 학습 벤치마크를 기반으로 광범위한 실험을 통해 그 우수성을 입증하였다.



### Automatic Medical Report Generation: Methods and Applications (https://arxiv.org/abs/2408.13988)
Comments:
          42 pages and 9 figures

- **What's New**: 이 논문은 2021년부터 2024년까지의 자동 의료 보고서 생성(AMRG) 방법을 포괄적으로 검토하여 기존 문제의 해결책을 제시하고 이를 다양한 영상 방식에 적용하며 공개된 데이터셋과 평가 메트릭스를 소개합니다.

- **Technical Details**: AMRG는 컴퓨터 비전(CV) 및 자연어 처리(NLP)를 활용하여 의료 이미지를 해석하고 인간과 유사한 설명적 보고서를 생성하는 연구 분야입니다. 모델은 이미지 특징을 추출하는 visual encoder와 텍스트를 생성하는 text decoder(RNN 또는 Transformer)를 포함합니다.

- **Performance Highlights**: AMRG의 최근 발전은 CNNs와 Transformers를 활용하여 높은 정확도의 병변 탐지 및 질병 분류를 가능하게 하였으며, 특히 딥러닝 기법이 많은 연구에서 성능 향상에 기여하고 있습니다.



### Focused Large Language Models are Stable Many-Shot Learners (https://arxiv.org/abs/2408.13987)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 FocusICL이라는 새로운 방법론을 제안하여 Large Language Models (LLMs)이 다수의 시연으로부터 주의가 분산되는 문제를 해결하고, 더 나은 성능을 이끌어냅니다.

- **Technical Details**: FocusICL은 triviality filtering을 수행하여 중요하지 않은 내용으로 인한 주의 분산을 방지하고, demonstration level에서 계층적 주의 메커니즘을 통해 현재 쿼리에 충분한 주의를 보장합니다. 또한, 모델 perplexity에 기반한 효율적인 하이퍼파라미터 검색 전략도 설계하였습니다.

- **Performance Highlights**: FocusICL은 ICL에 비해 평균 5.2% 향상된 성능을 달성하고, 많은 시연에서도 안정적으로 성능이 개선되는 특징을 보였습니다.



### AgentMove: Predicting Human Mobility Anywhere Using Large Language Model based Agentic Framework (https://arxiv.org/abs/2408.13986)
Comments:
          13 pages

- **What's New**: 본 논문에서는 AgentMove라는 새로운 예측 프레임워크를 소개합니다. 이는 인류의 이동 패턴을 보다 효과적으로 예측할 수 있도록 설계되었습니다. 전통적인 딥러닝 모델의 한계를 극복하고, 제로샷(Zero-shot) 예측을 가능케 합니다.

- **Technical Details**: AgentMove는 이동 예측(task)을 세 가지 서브태스크로 분해합니다: (1) 개별 이동 패턴 탐색(spatial-temporal memory module), (2) 도시 구조의 영향 모델링(world knowledge generator), (3) 인구 간의 공유 패턴 포착(collective knowledge extractor). 이 프레임워크는 메모리와 지식 생성을 결합하여 복잡한 이동 패턴을 더 잘 포착합니다.

- **Performance Highlights**: AgentMove는 12개 도시의 이동 데이터를 기반으로 한 실험에서 기존 최적 모델보다 8% 이상 향상된 성능을 기록했습니다. 또한 다양한 LLMs에 대한 높은 적응성과 안정성을 보여주며, 도시 간 예측의 지리적 편향을 줄였습니다.



### Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models (https://arxiv.org/abs/2408.13979)
Comments:
          Accepted at ICLR 2024 (Spotlight)

- **What's New**: 본 연구는 VLMs(비전-언어 모델)에서 learnable soft-prompt vector의 norm(노름)의 영향을 체계적으로 조사합니다. 특히, Low-Norm Effect를 발견하였으며, 이는 특정 학습된 soft prompt의 norm을 줄이는 것이 성능을 향상시키고, 증가시키는 것이 성능 감소를 초래할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 Nemesis 방법은 VLMs의 soft-prompt 벡터를 정규화하기 위해 Position-Uniform Normalization (PUN) loss를 사용하고, Position-Aware Normalization (PAN) loss를 도입하여 low-norm 효과를 고려한 정규화를 수행합니다. 이 방법은 기존의 soft-prompt 기법에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: Nemesis 방법은 다양한 작업에서 soft-prompt 기반 VLM의 성능을 향상시키는 데 도움이 될 수 있으며, 실험을 통해 이 방법의 효과를 입증하였습니다.



### Time Series Analysis for Education: Methods, Applications, and Future Directions (https://arxiv.org/abs/2408.13960)
Comments:
          24 pages, 3 figures, 6 tables, project page: see this https URL

- **What's New**: 이 논문은 시간 시리즈 분석의 교육적 맥락에서의 적용을 종합적으로 검토한 최초의 논문입니다. 주요 시간 시리즈 기법인 예측(forecasting), 분류(classification), 군집화(clustering), 이상 탐지(anomaly detection)의 다양한 사용 사례를 설정하여 적용하고 다양한 교육 시나리오를 탐색합니다.

- **Technical Details**: 이 논문에서는 교육 데이터 소스와 유형에 대한 심층 탐색을 통해 시간 시리즈 데이터의 특징을 네 가지 주요 카테고리로 조직합니다. 그 후, 시간 시리즈 분석의 주요 방법론을 다루고, 예측, 분류, 군집화 및 이상 탐지 방법이 교육에서 어떻게 적용되는지를 설명합니다. 또한, 다중 모달 융합(multimodal fusion) 및 대규모 언어 모델(LLMs)의 통합 가능성에 대한 미래 연구 방향을 제시합니다.

- **Performance Highlights**: 이 논문은 교육 분야에서 시간 시리즈 분석이 어떻게 학습 성과를 개선하는 데 기여할 수 있는지에 대한 통찰을 제공합니다. 실질적인 교육 시나리오에서 여러 전략을 동시에 적용하여 교육 결과를 향상시키는 방법을 강조하며, 교육 데이터의 구체적인 분류를 통해 현재와 미래의 교육 분석 적용을 위한 유용한 기초 자료를 제공합니다.



### Bridging the Gap between Real-world and Synthetic Images for Testing Autonomous Driving Systems (https://arxiv.org/abs/2408.13950)
Comments:
          Accepted for publication by the International Conference on Automated Software Engineering (ASE 2024)

- **What's New**: 본 논문은 자율 주행 시스템(ADS)의 DNN(Deep Neural Networks) 테스트 과정에서 도메인 간 변환기(translators)의 필요성과 영향력을 조사합니다. 특히, CycleGAN과 neural style transfer, 그리고 제안하는 SAEVAE 변환기를 비교 분석합니다.

- **Technical Details**: 자율 주행 시스템(ADS)의 효과적인 테스트를 위해, 도메인 간 변환기를 사용하여 훈련 데이터와 테스트 데이터의 분포 차이를 줄이는 방법을 모색합니다. 논문에서는 두 가지 주요 ADS 작업인 차선 유지(lane keeping)와 객체 탐지(object detection)에 대해 평가합니다.

- **Performance Highlights**: SAEVAE 변환기는 다른 변환기들에 비해 성능이 우수하고, 테스트 데이터의 다양성과 커버리지(coverage)를 감소시키지 않으며, 결함(revealing faults) 발견 능력도 유지합니다. 이외에도 SAEVAE는 시뮬레이션 시간에 최소한의 오버헤드를 발생시키고, 시뮬레이션 기반 테스트의 비용 절감에 기여할 수 있는 가능성을 제시합니다.



### Learning to Move Like Professional Counter-Strike Players (https://arxiv.org/abs/2408.13934)
Comments:
          The project website is at this https URL

- **What's New**: 이 연구에서는 CS:GO와 같은 멀티플레이어 1인칭 총격전 게임에서 팀 간의 협동 이동을 위한 데이터 기반 접근 방식을 제시합니다. 데이터를 통해 인간과 유사한 동작을 생성하는 이동 모델을 훈련시키는 방법론을 소개합니다.

- **Technical Details**: 연구에서는 총 123시간의 프로 게임 플레이 데이터를 수집하여, 이를 바탕으로 transform 기반의 이동 모델을 훈련시킵니다. 이 모델은 각각의 게임 스텝에서 0.5 ms 미만으로 모든 플레이어의 이동을 예측할 수 있는 효율성을 수치적으로 증명합니다. 새로운 quantitative positioning metrics를 통해 인간과 유사한 이동 행동을 평가하는 시스템도 포함됩니다.

- **Performance Highlights**: 모델은 인간 플레이어와 비교할 때 16%에서 59% 더 인간적인 행동을한다고 평가받았습니다. 모델은 in-game bot 대 bot 자가 플레이 실험을 통해 팀워크를 단순하게 구사하며 일반적인 이동 실수를 줄이고, 프로 CS:GO 경기에서 관찰되는 이동 분포 및 플레이어 생존 시간과 유사한 결과를 보여주었습니다.



### FedGlu: A personalized federated learning-based glucose forecasting algorithm for improved performance in glycemic excursion regions (https://arxiv.org/abs/2408.13926)
- **What's New**: 이 논문에서는 연속 혈당 모니터링 (Continuous Glucose Monitoring, CGM) 장치를 통해 당뇨병 환자의 혈당 조절을 개선할 수 있는 새로운 방법을 제시합니다. 특히, 저혈당 (hypoglycemia)과 고혈당 (hyperglycemia) 예측을 위한 새로운 HH 손실 함수(Hypo-Hyper loss function)를 도입하고, 데이터 프라이버시 문제를 해결하기 위한 연합 학습(Federated Learning, FL) 모델인 FedGlu를 제안합니다.

- **Technical Details**: HH 손실 함수는 평균 제곱 오차(Mean Squared Error, MSE) 손실에 비해 46% 개선된 성능을 보여줍니다. FedGlu는 로컬에서 모델을 훈련하고 모델 매개변수만을 공유하여 데이터 공유 없이 협업 학습을 가능하게 합니다.

- **Performance Highlights**: FedGlu는 local models에 비해 35% 더 우수한 혈당 변동 탐지율을 달성하여 125명의 환자 중 105명의 저혈당 및 고혈당 예측 성능을 향상시킵니다. 이러한 결과는 HH 손실 함수가 혈당 예측의 예측 능력을 상승시킴을 강조합니다.



### LLMs are Superior Feedback Providers: Bootstrapping Reasoning for Lie Detection with Self-Generated Feedback (https://arxiv.org/abs/2408.13915)
Comments:
          19 pages, 18 figures

- **What's New**: 이번 연구에서는 LLM(Large Language Models)의 기만 탐지 능력을 향상시키기 위한 부트스트랩(bootstrapping) 프레임워크를 제안합니다. 이 프레임워크는 제안, 피드백 수집, 수정의 세 가지 단계로 이루어져 있으며, LLM이 스스로 생성한 피드백을 활용하여 추론 능력을 개선합니다.

- **Technical Details**: 부트스트랩 프레임워크는 1) 제안 단계에서 비용 효율적인 LLM이 초기 예측을 생성하고, 2) 피드백 수집 단계에서 LLM이 이러한 예측에 대한 피드백을 제공하며, 3) 수정 단계에서 더 발전된 LLM이 자동 생성된 피드백을 사용하여 초기 예측을 다듬는 구조로 이루어져 있습니다. 이 방법을 Diplomacy 게임의 대화 데이터셋에 적용하였습니다.

- **Performance Highlights**: 제안된 방법은 기본 LLM에 비해 lying-F1 점수를 39% 향상시켰으며, 제로샷(Zero-shot) GPT-4 모델이 이전의 최첨단 감독 학습 모델과 경쟁할 수 있도록 했습니다. 특히, LLM이 생성한 피드백은 전문가의 피드백보다 29% 더 뛰어난 성능을 보였으며, 특히 인간이 불확실한 상황에서 유용함을 증명하였습니다.



### ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models (https://arxiv.org/abs/2408.13906)
Comments:
          First two authors contributed equally. Source code is available at this https URL

- **What's New**: 이번 연구에서는 다중 모달 대형 언어 모델(MLLMs)의 환각(hallucination) 문제를 해결하기 위해 ConVis라는 새로운 비훈련 대조적 디코딩(constrastive decoding) 방법을 제안합니다. ConVis는 텍스트-이미지(text-to-image, T2I) 생성 모델을 활용하여 환각된 캡션으로부터 주어진 이미지를 의미론적으로 재구성합니다.

- **Technical Details**: ConVis는 MLLM의 디코딩 과정에서 오리지널 이미지와 재구성된 이미지 간의 확률 분포를 비교함으로써 시각적 대조 신호(visual contrastive signals)를 포착하고, 환각 생성을 패널티(penalize)하여 이를 감소시킵니다. 주목할 점은 이 방법이 디코딩 과정 내에서만 작동한다는 점이며, 추가적인 데이터나 모델 업데이트가 필요하지 않습니다.

- **Performance Highlights**: 다섯 개의 벤치마크(CHAIR, HallusionBench, POPE, MME, LLaVA-Bench)에서 ConVis의 효과를 검증한 실험 결과, 다양한 MLLM에서 환각을 효과적으로 줄이면서 전반적인 응답 생성 성능을 유지할 수 있음을 보여주었습니다. 이 결과는 LLaVA-1.5, MiniGPT-4, mPLUG-Owl2와 같은 모델에서도 일관되게 나타났습니다.



### Enhancing SQL Query Generation with Neurosymbolic Reasoning (https://arxiv.org/abs/2408.13888)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 연구에서는 SQL 쿼리를 생성하는 신경기호(neurosymbolic) 구조를 제안합니다. 이 방법은 Best-First Search를 사용하여 솔루션 트리를 생성하고 탐색하며, 역추적(backtracking) 기능도 포함되어 있습니다.

- **Technical Details**: 제안된 아키텍처는 Language Model (LM)과 기호적 모듈(symbolic modules)을 통합하여, LM이 SQL 쿼리에서 발생한 오류를 잡고 수정하는 데 도움을 주며 솔루션 트리의 탐색을 안내합니다.

- **Performance Highlights**: Xander라는 툴을 사용하는 이 접근 방식은 작은 오픈소스 LM의 성능을 개선하여 평균 10.9%의 정확도를 증가시키고, 평균 28%의 실행 시간을 단축시킵니다. 그 결과 Xander를 사용할 경우, 네 배 크기가 더 큰 LM보다 우수한 성능을 발휘할 수 있습니다.



### Flexible game-playing AI with AlphaViT: adapting to multiple games and board sizes (https://arxiv.org/abs/2408.13871)
- **What's New**: 본 논문은 AlphaZero 프레임워크에 기반한 새로운 게임 AI 에이전트를 제안하며, Vision Transformers (ViT)을 활용하여 각각 AlphaViT, AlphaViD 및 AlphaVDA를 개발하였습니다. 이러한 에이전트는 고정된 보드 크기의 한계를 극복하고 다양한 사이즈의 보드 게임을 단일 모델로 플레이할 수 있도록 설계되었습니다.

- **Technical Details**: AlphaViT는 트랜스포머 인코더만을 사용하고, AlphaViD와 AlphaVDA는 인코더와 디코더를 포함하고 있습니다. AlphaViD의 디코더는 인코더 출력으로부터 입력을 수신하며, AlphaVDA는 학습 가능한 행렬을 디코더 입력으로 사용합니다. 실험 결과, 이들 에이전트는 Connect4, Gomoku, Othello와 같은 다양한 게임 환경에서 강력한 성능을 보여주었습니다.

- **Performance Highlights**: AlphaViT와 AlphaViD는 전통적인 알고리즘인 Minimax와 Monte Carlo tree search를 능가하며, AlphaZero에 근접하는 성능을 보여줍니다. 특히 AlphaViD는 추가적인 디코더 레이어 덕분에 다양한 행동 공간 및 보드 크기에 적응할 수 있는 능력이 향상되었습니다.



### CodeGraph: Enhancing Graph Reasoning of LLMs with Cod (https://arxiv.org/abs/2408.13863)
Comments:
          In Progress

- **What's New**: 이번 논문에서는 CodeGraph라는 방법을 소개합니다. CodeGraph는 그래프 문제의 솔루션을 코드로 인코딩하여 새로운 문제를 학습하고, 이를 프로그램 인터프리터(Program Interpreter)를 통해 실행합니다. 이 방법은 LLM(대형 언어 모델)에서 그래프 추론(task) 성능을 1.3%에서 58.6%까지 향상시킬 수 있습니다. 특히, 기존 방식들과는 달리, 수학적 문제에서 높은 성능을 보여주며, 추론 과정을 더 잘 제어하고 해석할 수 있는 접근법을 제공합니다.

- **Technical Details**: CodeGraph는 LLM이 파이썬(Python) 프로그램으로 추론 과정을 설명하고, 프로그램 인터프리터를 사용하여 계산을 수행하는 방식입니다. 이 방법은 GraphQA 데이터셋을 활용하여 6개 그래프 인코딩 방법과 함께 6개의 작업(task)을 평가합니다. CodeGraph는 GPT-3.5 Turbo를 포함한 다양한 LLM에서 수행되며, 가장 작은 모델은 140억 개의 활성 매개변수를 사용합니다.

- **Performance Highlights**: CodeGraph의 실험 결과, 6개 기본 그래프 작업의 평균 정확도가 63.3%에서 96.1%로 상승했습니다. 또한, CodeGraph는 다양한 그래프 구조에서도 강건성을 유지하며, 추가적인 예제를 통해 더 작은 모델에서도 높은 성능을 달성할 수 있음을 보여주었습니다.



### Tangram: A Challenging Benchmark for Geometric Element Recognizing (https://arxiv.org/abs/2408.13854)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 연구에서는 Tangram이라는 새로운 벤치마크를 소개하여 기존의 LMM들이 기하학적 요소를 인식하는 능력을 평가합니다. Tangram은 초등 및 중등 학교의 시험, 경쟁, 교과서에서 수집한 1,080개의 다양한 기하학적 도면을 포함하고 있으며, 각 도면에는 4개의 질문이 연결되어 총 4,320개의 시각-질문-답변 쌍을 제공합니다.

- **Technical Details**: Tangram 벤치마크는 세 가지 난이도로 나뉘며, LMM의 기하학적 요소 인식 능력을 평가하기 위해 설계되었습니다. 모델의 성능 평가를 위해서는 '단순하지만 흥미로운' 카운팅 작업이 적용되어, 도면 내의 문자, 삼각형, 원, 선분을 세는 작업이 요구됩니다.

- **Performance Highlights**: 실험 결과, 가장 성능이 좋은 모델조차도 기하학적 요소를 인식하는 데 있어 56.8%의 저조한 정확도를 기록했습니다. 이는 중학생의 93.6%와 전문가의 99.5% 정확도에 비해 현저히 낮은 수치로, LMM의 기하학적 문제 해결 능력이 인간에 비해 상당한 격차가 있음을 드러냅니다.



### Condensed Sample-Guided Model Inversion for Knowledge Distillation (https://arxiv.org/abs/2408.13850)
- **What's New**: 본 논문은 condensed samples를 활용하여 knowledge distillation (KD) 성능을 개선하는 새로운 방법을 제안합니다. 기존 데이터가 불완전할 때 KD의 효율성을 높일 수 있는 접근법을 다룹니다.

- **Technical Details**: 제안된 방식은 teacher 모델과 condensed samples를 활용하여 synthetic data를 생성하고, 모델의 학습 과정을 향상시키기 위해 feature discriminator를 사용합니다. 이 기법은 다양한 데이터 세트에서 KD 정확도를 최대 11.4%까지 개선함을 보여줍니다.

- **Performance Highlights**: 본 연구는 단 1개의 condensed sample로도 성능 개선 효과를 이끌어낼 수 있으며, few-shot 시나리오에서도 효과적인 결과를 보입니다. 기존 방법보다 최소한의 시간 오버헤드로 더 나은 성능을 달성할 수 있습니다.



### PropSAM: A Propagation-Based Model for Segmenting Any 3D Objects in Multi-Modal Medical Images (https://arxiv.org/abs/2408.13836)
Comments:
          26 figures, 6 figures

- **What's New**: 이 논문에서는 3D 의료 이미징에서 효과적으로 사용할 수 있는 새로운 세분화 모델인 PropSAM을 소개합니다. PropSAM은 2D 바운딩 박스 또는 스케치 마스크와 같은 하나의 뷰 프롬프트를 기반으로 세분화를 수행할 수 있는 것이 특징입니다.

- **Technical Details**: PropSAM은 CNN 기반의 UNet 아키텍처와 Transformer 기반의 모듈을 결합하여 슬라이스 내부 정보 처리 및 슬라이스 간 전파(Propagation)를 최적화합니다. 이를 통해 구조적 및 의미적 연속성을 유지하면서 다양한 이미징 모달리티에서의 세분화 효과를 향상합니다.

- **Performance Highlights**: PropSAM은 44개의 의료 데이터셋에서 평균 Dice Similarity Coefficient (DSC)를 18.1% 향상시키는 성능을 보였습니다. 또한, PropSAM은 사람의 프롬프트 효율성을 높이며, 2-view 프롬프트 모델에 비해 사용자 상호작용 시간을 37.8% 줄였습니다.



### Guardians of the Machine Translation Meta-Evaluation: Sentinel Metrics Fall In! (https://arxiv.org/abs/2408.13831)
Comments:
          Presented at ACL 2024 Main Conference. 29 pages

- **What's New**: 이번 연구에서는 기계 번역(Machine Translation, MT) 메트릭스의 메타 평가 프로세스에서 발생하는 두 가지 주요 문제를 조명하고, 이를 해결하기 위한 'sentinel metrics'라는 새로운 개념을 소개합니다. 이 메트릭스는 메타 평가의 정확성, 강건성(robustness), 공정성(fairness)을 검토하기 위해 디자인되었습니다.

- **Technical Details**: 현행 WMT의 메타 평가 프레임워크는 인간의 품질 평가를 모방하기 위해 특별히 훈련된 메트릭스와 연속적인 메트릭스에 유리하며, 이로 인해 발생하는 문제점을 지적하고, 메트릭스의 순위를 검증하기 위해 'sentinel metrics'를 사용합니다. 이 연구는 메트릭의 평가 과정에서 편향(bias)이나 불일치성을 모니터링하는 것을 목표로 합니다.

- **Performance Highlights**: 연구 결과, 현재의 메타 평가 프레임워크는 인간 평가와 잘 일치하지 않는 것으로 보이며, 인공지능(AI) 메트릭스들이 훈련 데이터에서 발생하는 허위 상관관계(spurious correlations)에 근거하여 평가를 수행할 수 있는 우려를 제기합니다.



### RoCP-GNN: Robust Conformal Prediction for Graph Neural Networks in Node-Classification (https://arxiv.org/abs/2408.13825)
Comments:
          12, 5 figures

- **What's New**: 본 논문은 Graph Neural Networks (GNNs)에서 불확실성 추정을 제공하기 위한 새로운 방법인 Robust Conformal Prediction for GNNs (RoCP-GNN)을 제안합니다. 이 방법은 GNN 훈련 과정에 conformal prediction (CP)을 직접 통합하여 사용자 정의 신뢰 수준에서 유효한 예측 집합을 생성합니다.

- **Technical Details**: RoCP-GNN은 예측 집합의 크기를 최소화하고 분류 정확도를 보존하며, 동시에 엔드-투-엔드 훈련 프레임워크를 개발하여 이러한 목표를 달성합니다. 훈련 과정에서 train-calib과 train-pred로 나누어진 훈련 세트를 사용하여 Smooth Calibration을 구현합니다.

- **Performance Highlights**: 다양한 그래프 벤치마크 데이터세트에서 RoCP-GNN을 검증한 결과, GNN 모델의 정확성을 통계적으로 유의미하게 향상시키면서 예측 세트의 정확성을 동시에 개선하였습니다. 이 방법은 모델에 구애받지 않으며 간단한 구현을 통해 일반 연구자와 실무자들이 쉽게 활용할 수 있습니다.



### Localization of Synthetic Manipulations in Western Blot Images (https://arxiv.org/abs/2408.13786)
- **What's New**: 최근의 딥러닝 및 생성 시스템의 발전은 합성 미디어(synthetic media) 창작에 크게 기여했으며, 특히 현실 콘텐츠의 현지 조작(local alteration)에서 매우 사실적인 합성 조작(synthetic manipulation)을 삽입하는 데 있어 심각한 문제를 제기하고 있습니다. 이러한 문제는 멀티미디어 데이터에 국한되지 않고, 과학 출판물에 포함된 생물학적 이미지로도 확장됩니다. 본 연구에서는 웨스턴 블롯(Western blot) 이미지에서 합성 조작을 지역화하는 작업을 다루고 있습니다.

- **Technical Details**: 우리는 이미지에서 추출한 작은 패치를 분석하여 진짜( pristine)와 합성된(synthetic) 픽셀을 구분할 수 있는 합성 탐지기(synthetic detector)를 제안합니다. 패치(patch) 기여도를 집계하여 합성 픽셀을 강조하는 변조 히트맵(tampering heatmap)을 추정합니다. 두 개의 조작된 웨스턴 블롯 이미지 데이터셋을 통해 우리의 방법론을 평가했습니다.

- **Performance Highlights**: 우리의 탐지기는 보편적인 합성 콘텐츠 생성기에 대해 뛰어난 성능을 보여주며, 거짓 알람(false alarm)이 거의 없습니다. 또한, 다양한 의미의 과학 이미지에서 테스트하여 일반화 가능성을 보여주었습니다.



### Analyzing the Impact of Splicing Artifacts in Partially Fake Speech Signals (https://arxiv.org/abs/2408.13784)
Comments:
          Accepted at ASVspoof 5 Workshop (Interspeech2024 Satellite)

- **What's New**: 최근 멀티미디어 포렌식 커뮤니티에서는 음성 딥페이크 탐지에 대한 큰 관심이 쏠리고 있으며, 부분적으로 가짜 신호의 식별에 대한 연구가 진행되고 있습니다. 특히, 실제 음성과 가짜 음성을 결합하여 생성한 스플라이싱(Splicing) 신호에 관한 데이터 및 탐지기 성능을 분석하여, 고품질의 스플라이스된 오디오 생성을 위한 복잡성을 강조합니다.

- **Technical Details**: 논문에서는 스플라이스된 오디오 트랙에서 나타나는 인위적 아티팩트(induced artifacts)에 대해 분석하며, 이러한 아티팩트가 기존 데이터셋에 미치는 영향을 평가합니다. 연구는 PartialSpoof 및 HAD 데이터셋을 활용하여, 간단한 신호 연결(concatenation) 기법으로부터 발생하는 아티팩트를 조사하였으며, 각 데이터셋에서 각각 6.16%와 7.36%의 탐지 EER(EER: Equal Error Rate)을 기록했습니다.

- **Performance Highlights**: 결과적으로, 연구에서는 스플라이싱 아티팩트를 분석함으로써 기존 데이터세트의 신뢰성 문제를 드러내고, 인위적 아티팩트의 존재가 다양한 탐지기 성능에 미치는 영향을 논의합니다. 이로 인해 향후 부분적으로 가짜 음성을 탐지하기 위한 데이터 처리 방식 및 탐지기 개발에 대한 지침을 제공합니다.



### SAB:A Stealing and Robust Backdoor Attack based on Steganographic Algorithm against Federated Learning (https://arxiv.org/abs/2408.13773)
- **What's New**: 이 논문은 Federated Learning(연합 학습)에서의 새로운 backdoor 공격 방법인 SAB(Steganographic Algorithm-based Backdoor attack)를 제안하며, 이는 기존 backdoor 공격보다 더 효과적이고 검출이 어려운 방식입니다. 특히, image steganographic algorithm을 활용하여, benign sample(양성 샘플) 내에 분산된 trigger(트리거)를 생성함으로써, 사람의 눈으로 쉽게 감지되지 않는 특성을 부여합니다.

- **Technical Details**: SAB는 steganographic 기법을 활용하여 전체 크기의 트리거를 생성하고, multiple loss joint computation을 통해 트리거를 생성합니다. 이를 통해 SAB는 benign samples(양성 샘플)와의 거리도 더 작고, 인간의 눈에 감지되기 어려운 높은 불현성(imperceptibility)을 보입니다. 또 다른 특징으로는 bottom-95% method를 적용하여 backdoor 공격의 생존 기간을 연장하며, gradient를 minor value points에서 업데이트함으로써 청소될 확률을 줄입니다.

- **Performance Highlights**: SAB는 기존 backdoor 방어 방법을 회피하거나 완화할 수 있으며, sparse-update 방법을 결합함으로써 backdoor의 일반화를 향상시킵니다. 이러한 방법론을 통해 SAB는 backdoor의 정확성을 개선하며, benign samples(양성 샘플)로 대체되기 어려운 효과를 보입니다. 논문에서 제안한 SAB의 접근법은 backdoor 공격의 지속 시간과 정확성을 크게 향상시키는 결과를 나타냈습니다.



### Lecture Notes on Linear Neural Networks: A Tale of Optimization and Generalization in Deep Learning (https://arxiv.org/abs/2408.13767)
Comments:
          Lecture notes

- **What's New**: 이 논문은 2021년 3월 프린스턴 대학교에서 진행된 심화 과정 강의에 기반하여, 깊은 학습(deep learning)의 수학적 이해에 관한 이론을 발표합니다. 특히, 선형 신경망(linear neural networks)에 대한 이론은 최적화(optimization) 및 일반화(generalization) 연구에 있어 근본적인 모델로 다루어집니다.

- **Technical Details**: 선형 신경망은 활성 함수가 없는 완전 연결(feed-forward) 신경망으로, 깊이는 n≥2이며, 입력 차원(dimension) d0, 출력 차원 dn과 여러 은닉 차원들이 존재합니다. 논문 내에서는 선형 신경망이 비선형적인 학습 문제에서 어떤 해결책으로 수렴하는지를 분석합니다. 또한 역동적(dynamical) 기법을 사용하여 이론적 접근을 통해 최적화 및 일반화를 분석합니다.

- **Performance Highlights**: 이 연구의 결과는 선형 신경망이 비선형 문제에서 뛰어난 최적화 및 일반화를 가능하게 한다는 것을 보여줍니다. 이러한 이론적 분석은 깊은 학습의 실용적인 응용에서 강력한 통찰을 제공합니다.



### Multimodal Ensemble with Conditional Feature Fusion for Dysgraphia Diagnosis in Children from Handwriting Samples (https://arxiv.org/abs/2408.13754)
- **What's New**: 이번 연구에서는 아동의 필기 장애인 developmental dysgraphia 진단을 위한 새로운 다중 모달 머신러닝 접근 방식을 제안합니다. 온라인 및 오프라인 필기 데이터를 모두 활용하여 기존 연구의 한계를 극복하고자 했습니다.

- **Technical Details**: 기존의 온라인 필기 데이터셋을 변환하여 오프라인 필기 이미지로 구성한 새로운 데이터셋을 생성했습니다. 본 연구에서는 단순 단어(simple word), 유사 단어(pseudoword), 어려운 단어(difficult word)의 서로 다른 유형의 단어 데이터를 분석하였고, SVM과 XGBoost 분류기를 온라인 및 오프라인 특징(feature) 각각에 대해 훈련시켰습니다. 또한, 조건부 특징 융합(conditional feature fusion)을 포함한 새로운 앙상블(ensemble) 방법을 제안했습니다.

- **Performance Highlights**: 본 연구의 방법론은 정확도 88.8%를 달성하였으며, 이는 기존의 SVM 단일 모달보다 12-14%, 기존 방법보다 8-9%, 전통적인 다중 모달 접근법보다 각각 3% 및 5% 향상된 성능입니다. 이러한 결과는 다중 모달 학습이 필기 장애 진단 향상에 큰 잠재력을 가지고 있음을 보여줍니다.



### DOCE: Finding the Sweet Spot for Execution-Based Code Generation (https://arxiv.org/abs/2408.13745)
Comments:
          10 pages (32 including appendix), 5 figures, 25 tables. arXiv admin note: text overlap with arXiv:2304.05128 by other authors

- **What's New**: 최근 다양한 decoding 및 reranking 기법이 LLM(대형 언어 모델) 기반 코드 생성에 효과적임을 보여주었습니다. 그러나 이러한 방법들을 연결하고 실험적으로 비교하는 포괄적인 프레임워크는 부족했습니다. 본 연구에서는 코드 실행을 위한 Decoding Objectives(디코딩 목표)라는 포괄적인 프레임워크를 제안하여 후보 생성, n-best reranking, 최소 베이즈 위험(MBR) 디코딩, 자기 디버깅(self-debugging)을 핵심 구성 요소로 포함하고 있습니다.

- **Technical Details**: 프레임워크의 구성 요소에는 후보 생성(candidate generation), reranking, 자기 디버깅(Chen et al., 2024)이 포함되어 있으며, 이를 통해 오라클(oracle) 성능 및 reranking 성능 모두를 개선할 수 있습니다. 실행 기반 메트릭을 통해 평가를 수행하였으며, 여러 후보에 대한 자기 디버깅을 적용한 결과 최첨단 성능을 달성했습니다.

- **Performance Highlights**: 제안된 DOCE(Decoding Objectives for Code Execution) 프레임워크를 통해 생성된 후보들의 실행 기반 reranking이 오라클 성능에 근접한 결과를 보여주었으며, 필터링을 통한 trial unit test를 기반으로 한 간단하고 효과적인 전략의 중요성을 강조하였습니다. 이를 통해 향후 코드 생성 연구에 대한 확고한 지침을 제공할 수 있을 것으로 기대됩니다.



### LogParser-LLM: Advancing Efficient Log Parsing with Large Language Models (https://arxiv.org/abs/2408.13727)
Comments:
          Accepted by ACM KDD 2024

- **What's New**: LogParser-LLM은 기존의 로그 파싱 기술의 한계를 극복하고 LLM(대형 언어 모델)의 능력을 활용하여 로그를 효율적으로 처리하는 새로운 로그 파서입니다.

- **Technical Details**: LogParser-LLM은 LLM 기반의 템플릿 추출기와 prefix tree를 결합하여 구문과 의미의 통찰을 동시에 제공합니다. 이 시스템은 온라인 파싱을 통해 빠르게 적응하며, 인간의 상호작용을 통합하여 사용자가 필요한 세부 수준(Granularity)을 조정할 수 있습니다.

- **Performance Highlights**: LogParser-LLM은 Loghub-2k 및 LogPub 벤치마크에서 검증된 결과, 평균 272.5회의 LLM 호출로 90.6% F1 점수의 그룹화 정확도를 달성하였고, 81.1%에 달하는 파싱 정확도를 기록하며 기존의 최신 기술들을 초월하는 성과를 보였습니다.



### DHP Benchmark: Are LLMs Good NLG Evaluators? (https://arxiv.org/abs/2408.13704)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 자연어 생성(NLG) 작업에서 평가자로 사용될 수 있다는 점을 강조하며, 기존 연구들이 LLM의 NLG 품질 평가 능력에 대한 충분한 탐색을 하지 못했다는 점을 지적합니다. 새로운 DHP(Discernment of Hierarchical Perturbation) 벤치마킹 프레임워크를 제안하여 LLM의 평가 능력을 정량적으로 측정하는 방법을 제시합니다.

- **Technical Details**: DHP 프레임워크는 계층적으로 변형된 텍스트 데이터를 사용하여 LLM의 평가 품질을 측정하는 시스템을 기반으로 하고 있습니다. 이를 통해 Wilcoxon Signed-Rank Test를 사용하여 평가 점수의 차이를 분석하고, 다양한 평가 메트릭을 결합하여 최종 Discernment Scores로 변환합니다. 이로 인해 LLM의 평가 능력을 보다 정밀하게 측정할 수 있습니다.

- **Performance Highlights**: DHP 벤치마크를 통해 다섯 가지 주요 LLM 시리즈에 대한 평가를 실시하였으며, 이를 통해 각 모델의 강점과 한계를 파악할 수 있는 중요한 통찰을 제공하였습니다. 또한, LLM들이 NLG 평가자로서의 역할을 수행하는 데 있어 나타나는 다양한 추세와 패턴을 밝혀내어, 향후 연구 및 개발에 중요한 기초 자료를 제공합니다.



### Submodular Maximization Approaches for Equitable Client Selection in Federated Learning (https://arxiv.org/abs/2408.13683)
Comments:
          13 pages

- **What's New**: 이 논문은 기존의 Federated Learning(FL) 프레임워크에서 클라이언트 선택 방식의 한계를 극복하기 위한 새로운 방법인 SUBTRUNC와 UNIONFL을 제안합니다. 이 방법들은 무작위 샘플링에 의한 불균형성과 공정성 문제를 해결하도록 설계되었습니다.

- **Technical Details**: 제안된 두 알고리즘 SUBTRUNC와 UNIONFL은 서브모듈러 함수(submodular function) 극대화를 활용하여 클라이언트 선택 과정을 최적화합니다. SUBTRUNC는 클라이언트의 손실 정보를 활용하여 해답을 다양화하고, UNIONFL은 이전 클라이언트 선택 데이터를 토대로 공정한 모델 성능을 보장합니다.

- **Performance Highlights**: 다양한 이질적인 시나리오에서의 광범위한 평가를 통해 제안된 방법들이 클라이언트 간의 공정성을 측정하는 디시밀리어리티 지표(client dissimilarity metric)에서 상당한 향상을 보여주었습니다.



### Hierarchical Network Fusion for Multi-Modal Electron Micrograph Representation Learning with Foundational Large Language Models (https://arxiv.org/abs/2408.13661)
Comments:
          Our paper is published at the workshop on Robustness of Few-shot and Zero-shot Learning in Foundation Models at NeurIPS 2023

- **What's New**: 이 연구에서는 전자 마이크로그래프(electron micrographs)의 분석을 위한 혁신적인 네트워크 아키텍처인 Hierarchical Network Fusion (HNF)을 제안합니다. 이 아키텍처는 여러 패치(patch) 시퀀스 및 비전 그래프(vision graphs)를 통해 전자 마이크로그래프의 복잡한 구조를 효과적으로 분석합니다.

- **Technical Details**: HNF는 다계층 네트워크 구조로, 패치 시퀀스와 비전 그래프 간의 정보를 교환하며 다양한 패치 해상도에서 지식을 통합합니다. 이 프레임워크는 Zero-shot Chain-of-Thought (Zero-Shot CoT) 프로세스를 통해 생성된 나노물질(nanomaterials)의 기술적 설명을 보조 정보로 활용하여 최종 작업을 지원합니다. 또한, Cross-modal attention 메커니즘을 사용하여 이미지와 언어적 통찰 간의 지식 융합을 예측합니다.

- **Performance Highlights**: 이 프레임워크는 전통적인 방법들보다 뛰어난 성능을 보이며, 배급 이동(distributional shifts)으로 인한 문제를 극복하고 높은 처리량(screening)을 가능하게 합니다. 이는 고해상도 전자 마이크로그래프의 더 포괄적이고 정확한 표현을 통해 나노 물질 분류 작업을 개선함을 보여줍니다.



### Reactzyme: A Benchmark for Enzyme-Reaction Prediction (https://arxiv.org/abs/2408.13659)
- **What's New**: 이 논문은 카탈리스트로서의 효소 기능 예측을 위한 새로운 접근 방식을 소개합니다. 기존의 효소 주석 방법을 상세하게 다루며, 효소가 촉매하는 특정 반응을 기반으로 한 주석 방법을 제안합니다.

- **Technical Details**: 새로운 방법은 기계 학습 알고리즘을 사용하여 효소 반응 데이터 세트를 분석합니다. Reactzyme 데이터 세트는 SwissProt와 Rhea 데이터베이스에서 파생된 가장 큰 효소-반응 데이터 세트를 사용하여 프레임워크를 구성하며, 효소의 카탈리틱 능력을 기반으로 효소를 순위로 매기는 검색 문제로 정의됩니다.

- **Performance Highlights**: 새로운 모델을 통해 우리는 새로운 반응을 위한 단백질 모집 및 새로운 단백질 내 반응 예측을 할 수 있습니다. 이는 효소 발견과 기능 주석을 촉진하는 데 기여합니다.



### Studying the Effect of Audio Filters in Pre-Trained Models for Environmental Sound Classification (https://arxiv.org/abs/2408.13644)
Comments:
          19 pages, 16 figures

- **What's New**: 이번 연구에서는 Two-Level Classification 방법론을 제안하고, 이를 통해 환경 소리 분류(ESC)의 정확도를 높이는 방법을 탐구합니다. 레벨 1 분류기는 보다 넓은 범주의 소리를 분류하고, 레벨 2 분류기는 레벨 1의 출력에 기반하여 실제 소리의 클래스를 결정합니다. 이 연구는 새로운 오디오 필터인 오디오 크롭(Audio Crop)을 도입하여 높은 정확도를 달성했습니다.

- **Technical Details**: 제안된 방법론은 CNN 모델을 사용하여 VGG, ResNet, EfficientNet 아키텍처를 활용합니다. 레벨 1 분류기는 소리를 동물, 새, 자연 등의 더 넓은 그룹으로 분류하고, 레벨 2 분류기는 해당 그룹 내의 세부 클래스를 식별합니다. 실험에서는 ESC-50 데이터셋을 사용하여 레벨 1에서 78.75%, 레벨 2에서 98.04%의 정확도를 기록했습니다.

- **Performance Highlights**: 연구 결과, 오디오 크롭 필터를 포함한 다양한 오디오 수정 기법이 기존의 소리 분류 정확도를 크게 향상시켰습니다. 기존 문헌에서는 50 클래스 직접 분류를 위한 기법들이 있었으나, 본 연구는 Two-Level Classification으로 접근하여 상대적으로 높은 성능을 기록하며 발전적인 결과를 보여주었습니다.



### Temporal Elections: Welfare, Strategyproofness, and Proportionality (https://arxiv.org/abs/2408.13637)
Comments:
          Appears in the 27th European Conference on Artificial Intelligence (ECAI), 2024

- **What's New**: 본 논문에서는 단일 대안을 하는 순차적 결정 모델을 탐구하며, 공리적 복지(Utilitarian welfare)와 평등적 복지(Egalitarian welfare)의 두 목표를 중점적으로 다루고 있습니다. 특히 이 두 복지 목표에 대한 최대화 문제의 계산 복잡성과 전략 무관성(strategyproofness) 및 비례성(proportionality) 간의 호환성을 고려합니다.

- **Technical Details**: 연구의 주요 발견으로는, 공리적 복지(Util)의 최대화는 O(n^k)와 같은 다항식 시간 내에 해결할 수 있지만, 평등적 복지(Egal)의 결정 문제는 NP-완전(NP-complete)임을 보여주는 것입니다. 또한, 전략 무관성을 고려했을 때, 공리적 결과를 제공하는 메커니즘은 전략 무관성을 만족하지만, 모든 결정론적 메커니즘은 약한 형태의 전략 무관성인 비명백 조작(non-obvious manipulability, NOM)에 실패하는 점도 언급합니다.

- **Performance Highlights**: 마지막으로, 비례적 결과를 산출하는 것은 효율적으로 가능하지만, 공리적 복지를 최대화하면서 비례성을 보장하는 결과를 찾는 것은 NP-하드(NP-hard)임을 보여주며, 공리적 복지와 평등적 복지에 따른 비례성의 가격에 대한 상한 및 하한 값을 도출합니다.



### DeepVoting: Learning Voting Rules with Tailored Embeddings (https://arxiv.org/abs/2408.13630)
- **What's New**: 이 논문에서는 여러 에이전트의 선호를 집합적인 결정으로 집계하는 과정을 개선하기 위한 새로운 접근법을 제안합니다. 특히, 기존의 투표 규칙을 학습하는 데 필요한 복잡한 모델 대신에 더 간단한 신경망과 커스터마이즈된 임베딩을 활용하여 효율적인 학습을 지원합니다.

- **Technical Details**: 연구팀은 Neural Networks를 사용하여 확률적인 사회 선택 함수(probabilistic social choice functions)를 학습하고, 기존의 사회 선택 문헌에서 파생된 선호 프로파일의 임베딩을 통해 더 많은 유권자의 데이터를 효율적으로 처리할 수 있도록 하였습니다. 이 방법을 통해, 모델의 입력 크기와 파라미터 수를 줄여 효율적인 접근이 가능하게 되었습니다.

- **Performance Highlights**: 논문에서 제안된 접근법은 전통적인 규칙들(Plurality, Borda 등)에 비해 참여 공리(Participation axiom) 모순을 방지하는 방법으로 조정할 수 있는 새로운 투표 규칙을 제안하며, 이는 투표의 정확성을 희생하지 않으면서 개선된 공리적 속성을 지닙니다.



### Enhancing Uplift Modeling in Multi-Treatment Marketing Campaigns: Leveraging Score Ranking and Calibration Techniques (https://arxiv.org/abs/2408.13628)
- **What's New**: 이번 논문은 다중 치료 마케팅 캠페인에서의 Uplift Modeling을 최적화하기 위한 새로운 접근법을 제안합니다. 기존의 CausalML 프레임워크를 활용하여 여러 단일 치료 모델을 적용하고, 점수 랭킹(score ranking) 및 보정(calibration) 기술을 통해 마케팅 캠페인의 전반적인 성과를 개선하는 방법을 소개합니다.

- **Technical Details**: 연구에서는 메타 학습자(Meta Learner) 프레임워크(S, T, X)와 현실 세계 사례에서의 적용을 검토합니다. 메타 학습자 보정과 점수 랭크 기반의 제안 선택 전략을 통합하여 다중 치료 캠페인의 조직적 효과성을 향상시키는 방법론을 제시합니다. 실제 데이터셋에 대한 실험 결과를 통해 우리 접근법의 실질적인 이점과 우수성을 입증하였습니다.

- **Performance Highlights**: 실험 결과는 점수 랭킹과 보정 기술의 통합이 Uplift 예측의 성능과 신뢰성을 강화하는 데 중요한 역할을 한다는 점을 강조합니다. 이를 통해 마케팅 분석에서 예측 모델링의 발전을 도모하고 캠페인 전략 최적화를 위한 실행 가능한 통찰력을 제공합니다.



### Towards Case-based Interpretability for Medical Federated Learning (https://arxiv.org/abs/2408.13626)
Comments:
          \c{opyright} 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 논문은 의료 분야에서의 federated learning 설정에서 사례 기반 설명을 생성하기 위해 딥 생성 모델을 탐구합니다. 기존의 AI 모델 결정에 대한 설명은 AI 채택을 장려하는 데 필수적이며, 본 연구는 유례없는 데이터 보호 규정을 준수할 수 있는 방법으로 제안됩니다.

- **Technical Details**: 적용된 방법론은 두 가지 주요 단계를 포함합니다. 첫 번째 단계는 pleural effusion 예측을 위한 차별화된 federated 모델을 훈련시키는 것입니다. 여기에서 DenseNet-121 아키텍처를 기반으로 하고, Privacy-preserving을 위해 differential privacy (DP)를 적용합니다. 두 번째 단계는 Medfusion이라는 생성 모델을 이용하여 사례 기반 설명을 생성하는 것입니다. 생성된 샘플은 유사한 질병 사례로서 판별 모델을 위한 설명으로 사용됩니다.

- **Performance Highlights**: 분류 성능은 F1-score의 형태로 평가되며, 이 모델의 품질은 nDCG를 통해 측정됩니다. 이는 기존 의료 이미징 패러다임에서 AI의 신뢰성을 높이는 데 기여하고자 합니다.



### No Dataset Needed for Downstream Knowledge Benchmarking: Response Dispersion Inversely Correlates with Accuracy on Domain-specific QA (https://arxiv.org/abs/2408.13624)
Comments:
          16 pages, 3 tables, 1 figure

- **What's New**: 이 연구는 LLM의 특정 주제 도메인에 대한 지식을 비교할 때 QA 데이터셋을 생성하고 (챗봇) LLM 응답을 평가할 필요성을 없애고자 합니다. 사용자 중심의 접근 방식을 사용하여 LLM의 내부 작업에 대한 접근 없이도 여러 번 동일한 질문을 던짐으로써 응답 분산(response dispersion)을 정의하고 평가할 수 있는 방법론을 제안합니다.

- **Technical Details**: 응답 분산은 LLM 응답의 임베딩 행렬에서 95%의 분산을 설명하는 데 필요한 단일 값의 수로 정의됩니다. 이 연구에서는 OpenAI API의 임베딩과 '참조 문장 유사성 임베딩'이라는 새로운 임베딩 방법을 사용하여 응답 임베딩을 생성하고 있습니다. 이 연구는 즉각적이고 저렴한 방법으로 최상의 LLM을 선택할 수 있도록 하는 절차를 제공합니다.

- **Performance Highlights**: 응답 분산을 비교하여 두 개의 다른 LLM을 같은 주제 도메인에서 비교할 때, QA 정확도 대신 응답 분산을 사용하는 것이 74%에서 89%의 시간에 적합한 대체 방법으로 나타났습니다. 이 연구는 5%와 10%의 허용 오차(tolerance level)에서 응답 분산을 통해 LLM 성능 평가의 효과성을 입증하였습니다.



### Advancing Enterprise Spatio-Temporal Forecasting Applications: Data Mining Meets Instruction Tuning of Language Models For Multi-modal Time Series Analysis in Low-Resource Settings (https://arxiv.org/abs/2408.13622)
Comments:
          Published at the ICLR 2024 Workshop on Practical ML for Low Resource Settings(PML4LRS)

- **What's New**: 본 논문에서는 시간 시계열 예측(time series forecasting)을 위한 혁신적인 다중 모달 접근 방식을 제안합니다. 이 방법은 전통적인 예측 기법과 소형 언어 모델의 instruction tuning을 통합하여 비정상 데이터의 시계열 트렌드 분석을 개선하는 것을 목표로 합니다.

- **Technical Details**: 제안된 MultiTs Net 프레임워크는 시간-공간(time-then-space) 모델링 접근 방식을 활용하여 비정상 데이터의 intra-series 및 inter-series 의존성을 처리하고, grouped-query attention 메커니즘을 사용하여 장기적인 시간 의존성을 학습합니다. 또한, MoE(Mixture of Experts) 아키텍처와 PEFT(Parameter-Efficient Fine-Tuning) 방법을 결합하여 소비자 하드웨어에서 AI 솔루션을 확장할 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 실제 데이터 세트에 대한 실험 결과, MultiTs Net은 기존의 방법들에 비해 강력하고 정확한 예측 결과를 제공하며, 성능과 지연(latency)의 균형을 유지합니다. 이 프레임워크는 데이터 프라이버시를 보장하면서도 빠른 추론 속도를 자랑합니다.



### Preliminary Investigations of a Multi-Faceted Robust and Synergistic Approach in Semiconductor Electron Micrograph Analysis: Integrating Vision Transformers with Large Language and Multimodal Models (https://arxiv.org/abs/2408.13621)
Comments:
          Published at Deployable AI (DAI) Workshop at AAAI-2024

- **What's New**: 이 연구는 반도체 제조에서 자동화된 나노 소재 식별을 위한 혁신적인 아키텍처를 제안합니다. 이는 대형 언어 모델(LLMs) 및 대형 다중 모달 모델(LMMs)의 생성적 능력을 활용하는 방법입니다.

- **Technical Details**: 이 연구는 'Zero-Shot Chain of Thought (Zero-Shot CoT)' 및 'Few-Shot In-Context Learning' 접근법을 통해 특화된 작업을 위한 최적화된 프롬프트를 생성합니다. 또, Vision Transformers (ViT)를 이용해 입력 이미지를 패치로 나누어 1D 벡터로 전환하고, 위치 정보가 향상된 후 단일 글로벌 이미지 표현을 위한 분류 토큰을 추가합니다.

- **Performance Highlights**: 제안된 방법은 기존의 자동화된 나노 소재 식별 방법을 초월하여 높은 정밀도를 제공하며, 반도체 제조업계에서의 고처리량 스크리닝을 용이하게 합니다.



### Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation (https://arxiv.org/abs/2408.13586)
- **What's New**: 본 논문에서는 샘플링 기반 디코딩 전략의 신뢰성을 향상시키기 위해, 다양성과 위험 간의 균형을 고려한 체계적인 방법을 제안하고 있습니다. 기존의 트렁케이션 샘플링 방법들과 그 추천 파라미터들에 대한 포괄적인 비교를 제공합니다.

- **Technical Details**: 연구팀은 Wikipedia 데이터를 단어 레벨의 접두사 트리 구조로 재구성하여, 'Context-Preserving Trie (CP-Trie)'를 만들었습니다. 이를 통해 각 샘플링 방법의 이론적 용량을 평가할 수 있으며, 다양한 트렁케이션 매개변수 값을 기준으로 토큰의 수를 측정합니다.

- **Performance Highlights**: 새롭게 제안된 평가 벤치마크를 통해, 다양한 샘플링 디코딩 방법의 이론적 용량을 추정할 수 있게 되며, 실제 응용에서의 트렁케이션 샘플링 방법 및 그에 맞는 파라미터 선택에 대한 가이드라인을 제공하여 샘플링 방법의 유용성을 높입니다.



### Synesthesia of Machines (SoM)-Enhanced ISAC Precoding for Vehicular Networks with Double Dynamics (https://arxiv.org/abs/2408.13546)
Comments:
          13 pages, 17 figures, 4 tables

- **What's New**: 본 논문에서는 vehicular networks의 통합 감지 및 통신 기술(Integrated Sensing and Communication, ISAC)을 바탕으로 한 새로운 precoding 방법론을 제안합니다. 특히, double dynamics 상황을 위한 기계의 synesthesia(SoM)를 활용하여 다양한 모달리티를 통합하는 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법론은 deep reinforcement learning (DRL) 기반의 프레임워크를 통해 double dynamics에 적응할 수 있는 precoding을 수행합니다. 또한, parameter-shared actor-critic (PSAC) 아키텍처를 적용하여 복잡한 상태 및 동작 공간 내에서 훈련을 가속화합니다. POMDP(Partially Observed Markov Decision Process)를 통해 상태 공간과 보상 함수를 정의하며, hybrid action space를 통해 효율적인 precoding 업데이트를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 방법들보다 ISAC 성능을 극대화하며, 계산 복잡도를 줄이는 동시에 동적 환경에서도 실시간 반응성을 보이는 것을 확인했습니다. 또한, Doppler shift로 인한 ICI(Inter-Carrier Interference)를 효과적으로 다루며, 복잡한 결정 공간 문제를 해결하는데 성공했습니다.



### Selective Preference Optimization via Token-Level Reward Function Estimation (https://arxiv.org/abs/2408.13518)
Comments:
          Work in progress

- **What's New**: 최근 대규모 언어 모델의 정렬 방법론에서는 토큰 수준(supervision)을 활용하여 세부적인 선호 최적화를 수행하는 것이 주목받고 있습니다. 이와 관련하여 본 연구에서는 Selective Preference Optimization (SePO)이라는 새로운 선택적 정렬 전략을 제안합니다. SePO는 효율적인 키 토큰 선택을 중심으로 하며, Direct Preference Optimization (DPO)을 기반으로 최초의 토큰 선택 방법을 제시합니다.

- **Technical Details**: SePO는 원본 응답 수준의 선호 주석을 기반으로 하는 오라클 모델을 훈련하여 최적의 토큰 수준 보상 함수(reward function)를 파라미터화합니다. 주요 장점은 1) 추가적인 감독 신호 없이 기존 정렬 데이터셋에 직접 적용 가능하다는 점과 2) 오라클 모델의 크기 및 훈련 하위 집합의 크기를 조절함으로써 선택적 정렬의 비용을 감소시킬 수 있다는 점입니다. SePO에서는 선택된 키 토큰을 통해 목표 정책 모델을 최적화합니다.

- **Performance Highlights**: SePO는 세 개의 공공 평가 벤치마크에서 실험을 진행하였으며, 최적화된 30%의 키 토큰으로 경쟁 기초 방법론보다 현저히 우수한 성능을 나타냈습니다. 또한, 약한 오라클 모델이 파라미터 수가 16.8배 더 많은 강력한 정책 모델을 효과적으로 지도하여 성능 향상 및 분포 외(over-optimization) 문제를 해결했습니다.



### AnoPLe: Few-Shot Anomaly Detection via Bi-directional Prompt Learning with Only Normal Samples (https://arxiv.org/abs/2408.13516)
Comments:
          Code is available at this https URL

- **What's New**: 이번 연구에서는 True Anomalies(진짜 이상 패턴)에 대한 사전 정보 없이도 Few-shot Anomaly Detection(소수 샘플 이상 탐지) 문제를 해결하기 위해 AnoPLe이라는 새로운 다중 모달 프로프트 학습 방법론을 제안합니다. AnoPLe은 텍스트와 비주얼 프롬프트 간의 쌍 방향 결합을 통해 두 모달리티 간의 깊은 상호작용을 촉진합니다.

- **Technical Details**: AnoPLe은 시뮬레이션된 이상 패턴을 사용하여 CLIP(Contrastive Language-Image Pretraining)을 통해 정상 샘플과 비정상 샘플을 구분하는 방식으로 작동합니다. 또한, 다중 뷰 신호를 학습 가능한 경량 디코더와 통합하여 지역 의미(semantics)를 향상시킵니다. 실험 결과, AnoPLe은 MVTec-AD와 VisA에서 각각 94.1%와 86.2%의 Image AUROC(Area Under the Receiver Operating Characteristic)을 기록했습니다.

- **Performance Highlights**: AnoPLe는 기존 SoTA(State-of-the-Art)인 PromptAD와 비교해도 떨어지지 않는 성능을 발휘하며, 실제 이상 패턴에 대한 노출 없이도 강력한 FAD 성능을 달성했습니다. 특히, 1-shot 및 4-shot 설정에서 MVTec에서 각각 94.1% 및 96.3%의 I-AUROC을 기록했습니다.



### Thresholded Lexicographic Ordered Multiobjective Reinforcement Learning (https://arxiv.org/abs/2408.13493)
Comments:
          Full version of ECAI 2024 paper

- **What's New**: 본 연구에서는 lexicographic multi-objective 문제에 대한 새로운 접근 방식과 이를 해결할 수 있는 Lexicographic Projection Optimization (LPO) 알고리즘을 제안합니다. 기존의 TLQ 알고리즘의 한계점을 더 깊이 탐구하고 Policy Gradient 알고리즘을 활용한 솔루션을 제시합니다.

- **Technical Details**: Lexicographic Ordering (LO)와 Thresholded Lexicographic Ordering (TLO) 기법을 소개하며, TLO의 단점을 극복하기 위한 새로운 lexicographic projection 알고리즘을 제안합니다. 이 알고리즘은 상위 중요 목표의 값을 보존하면서 불만족 목표를 최적화할 수 있는 방향을 계산합니다. 또한 Lexicographic Markov Decision Processes (LMDPs)에 대해 Policy Gradient 접근 방식을 적용합니다.

- **Performance Highlights**: 제안된 알고리즘은 기존 TLQ 변형에 비해 새로운 문제 클래스에서 성능이 개선되었으며, 하이퍼콘을 통한 프로젝션 기법을 기반으로 더 나은 최적화 성능을 보여주었습니다. 실험을 통해 REINFORCE 적응형 알고리즘의 유효성을 입증하였습니다.



### MPruner: Optimizing Neural Network Size with CKA-Based Mutual Information Pruning (https://arxiv.org/abs/2408.13482)
- **What's New**: 새로운 가지치기 알고리즘인 MPruner를 제안하며, 이는 벡터 유사성을 통해 상호 정보(mutual information)를 활용하여 개별 모델 구성 요소의 기여도를 평가합니다. 이를 통해 보다 정밀하고 효율적으로 층별 가지치기를 수행할 수 있습니다.

- **Technical Details**: MPruner는 Centered Kernel Alignment (CKA) 유사성 수치를 기반으로 층 네트워크를 클러스터링하여 전체 정보를 통합한 후, 불필요한 층을 동시에 제거하고 최적화된 모델을 재학습하여 원래 모델에 비해 크기를 크게 줄입니다. 또한, 가지치기 기준으로 희소성(sparsity)보다 정확도 임계값을 사용하여, 지정된 기준 내에서 안전하게 가지치기할 수 있는 층의 수를 결정합니다.

- **Performance Highlights**: MPruner는 CNN과 transformer 기반 모델에서 최대 50%의 매개변수(Parameter)와 메모리 사용량 감소를 보여주었으며, 정확도 손실은 최소 또는 전무한 결과를 나타냈습니다. 다양한 구조와 환경에서 검증된 MPruner는 신경망을 최적 크기로 가지치기하는 효과적인 도구로 자리잡았습니다.



### Disentangled Generative Graph Representation Learning (https://arxiv.org/abs/2408.13471)
- **What's New**: 본 연구는 Self-supervised Learning(자기지도 학습)에 기반한 Disentangled Generative Graph Representation Learning(DiGGR)을 제안합니다. DiGGR은 학습된 표현의 얽힘을 해소하고 그래프 마스크 모델링을 안내하는 잠재 인자(latent factors)를 학습하여 더 나은 표현을 가능하게 합니다.

- **Technical Details**: DiGGR은 두 가지 주요 접근 방식을 사용합니다. 첫째, Latent Factor Learning 모듈을 통해 노드의 이질적 인자를 모델링하여 그래프를 여러 개의 비판적 서브그래프로 분리합니다. 둘째, Factor-wise Self-supervised Graph Representation Learning 프레임워크를 설계하여 각 서브그래프에 대한 특정 마스킹 전략을 적용하며, 그 결과 각 인자별 그래프 표현을 개선합니다.

- **Performance Highlights**: DiGGR은 11개의 공개 데이터셋을 대상으로 한node 및 graph classification 작업에서 실험을 수행한 결과, 기존의 여러 Self-supervised 방법들보다 일관되게 우수한 성능을 보였으며, 제안한 접근 방식의 유효성을 입증했습니다.



### LlamaDuo: LLMOps Pipeline for Seamless Migration from Service LLMs to Small-Scale Local LLMs (https://arxiv.org/abs/2408.13467)
Comments:
          28 pages, 18 figures, 6 tables

- **What's New**: 이 논문에서는 서비스 지향의 대형 언어 모델(LLM)에서 작고 로컬하게 관리 가능한 모델로 지식과 기능을 매끄럽게 이전하기 위한 LLMOps 파이프라인인 'LlamaDuo'를 소개합니다. 이는 클라우드 기반 모델의 운영 의존성, 개인정보 문제 및 오프라인 요구 사항을 해결하는 데 중점을 둡니다. 

- **Technical Details**: LlamaDuo는 서비스 LLM에 의해 생성된 합성 데이터셋을 사용하여 소형 언어 모델을 미세 조정(fine-tuning)하고, 성능이 기대 이하일 경우 추가적인 유사 데이터로 반복 조정을 진행하는 과정을 포함합니다. 이 접근 방식을 통해 작은 모델이 결국 서비스 LLM의 성능을 일치시키거나 초과할 수 있도록 보장합니다.

- **Performance Highlights**: LlamaDuo는 여러 다운스트림 작업에서의 효과성, 적응성 및 경제성을 입증하기 위한 광범위한 실험을 수행했습니다. GPT4o, Claude 3 Sonnet, Gemini 1.5 Flash와 같은 인기 있는 서비스 LLM을 사용하여 작은 로컬 LLM이 서비스 LLM의 성능에 도달하거나 뛰어넘을 수 있는 잠재력을 보유하고 있음을 확인했습니다.



### Probing the Robustness of Vision-Language Pretrained Models: A Multimodal Adversarial Attack Approach (https://arxiv.org/abs/2408.13461)
- **What's New**: 이 논문에서는 비전-언어 프리트레인 (Vision-Language Pretraining, VLP) 트랜스포머의 적대적 견고성을 연구하고, 새로운 Joint Multimodal Transformer Feature Attack (JMTFA) 방법을 제안합니다. 이 방법은 시각 (visual) 및 텍스트 (textual) 모달리티에서 동시에 적대적 섭동을 도입하여, 모델의 예측 결과를 왜곡하는 데 중점을 둡니다.

- **Technical Details**: JMTFA는 크로스 모달 (cross-modal) 상호작용을 기반으로 시각과 언어 모달리티 간의 깊은 상관관계를 분석하여, 두 모달리티의 중요한 특징을 동시에 교란합니다. 이 연구에서는 비전-언어 이해 및 추론 작업 베이스라인과 비교하여 실험을 수행하였습니다. 트랜스포머의 크기와 적대적 견고성 간의 관계는 뚜렷하지 않다는 것이 발견되었습니다.

- **Performance Highlights**: 제안된 JMTFA 접근법은 VLP 모델에서 높은 공격 성공률을 기록하였으며, 복잡한 네트워크 아키텍처가 시각 모달리티보다 텍스트 모달리티에 더 많이 의존한다는 것을 입증하였습니다. 이러한 결과는 다중모달 AI 시스템의 신뢰할 수 있는 배포에서의 잠재적 위험을 강조합니다.



### Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning (https://arxiv.org/abs/2408.13457)
Comments:
          Preprint

- **What's New**: 이 논문에서는 Difficulty-Adaptive Self-Consistency (DSC)라는 새로운 방법을 제안하여, 기존 Self-consistency (SC) 기법의 한계를 극복하고 비용을 줄이는데 중점을 두었습니다. DSC는 문제 난이도 정보를 활용하여 적절히 추론 자원을 할당합니다.

- **Technical Details**: DSC는 다음의 세 단계로 구성됩니다: 난이도 순위 매기기 (Difficulty Ranking), 문제 파티션 (Problem Partition), 샘플 사이즈 사전 할당 (Sample Size Pre-Allocation). 난이도 순위 매기기 단계에서, LLM을 사용하여 문제 집합의 난이도를 평가하고, 이를 기반으로 문제를 쉬운 문제와 어려운 문제로 나눕니다. 그런 다음, 각 문제에 필요한 샘플 크기를 예측하여, 여러 번의 재샘플링을 줄이고 입력 비용을 절감합니다.

- **Performance Highlights**: DSC는 세 가지 주요 벤치마크에서 ASC 및 ESC보다 비용 효율성이 뛰어난 성능을 보였으며, 비슷한 성능을 유지했습니다. 실험 결과, DSC는 효율성과 성능 모두에서 강력한 기준선인 ASC 및 ESC를 일관되게 초과하는 결과를 나타냈습니다.



### A Law of Next-Token Prediction in Large Language Models (https://arxiv.org/abs/2408.13442)
- **What's New**: 본 논문은 사전 훈련된 대형 언어 모델(LLM)에서 중간 층을 통한 컨텍스트화된 토큰 임베딩(embedding)의 학습을 설명하는 정밀하고 정량적인 법칙을 소개합니다. 이 법칙은 입력 데이터 처리 및 예측의 정확도를 향상시키기 위해 각 층이 동등한 기여를 한다는 점에서의 새로운 통찰을 제공합니다.

- **Technical Details**: LLM은 비선형 모델로, 각 층에서 주의(attention) 메커니즘과 다양한 연산을 통해 토큰 임베딩의 시퀀스를 새 시퀀스로 반복적으로 매핑합니다. 각 층의 예측 능력을 평가하기 위해 예측 잔차(prediction residual, PR)를 사용하여 LLM의 다음 토큰 예측 능력을 정량화하였습니다. 이 연구는 PR 값이 층을 지날 때마다 거의 일정하게 감소하는 '균일학습(equi-learning)' 법칙을 밝혀냈습니다.

- **Performance Highlights**: 각 층의 PR 값은 −0.983에서 −0.997 사이의 Pearson 상관 계수로 나타났으며, 모델 아키텍처, 데이터 세트, 모델 깊이 및 훈련 시간에 따라 법칙의 특성이 달라질 수 있음을 보여줍니다. 이 법칙은 LLM의 설계, 훈련 및 해석을 위한 보다 세분화된 접근 방식으로 활용될 수 있습니다.



### Transforming Location Retrieval at Airbnb: A Journey from Heuristics to Reinforcement Learning (https://arxiv.org/abs/2408.13399)
- **What's New**: 이 논문에서는 Airbnb의 위치 검색 시스템의 발전과 도전 과제를 다루고 있습니다. Airbnb는 다양한 게스트의 요구를 충족하는 효율적인 검색 시스템을 구축하기 위한 방법론을 설명하고, 해당 시스템의 기초부터 기계 학습 기반의 위치 검색 제품을 개발하는 과정에서의 도전 과제를 제시합니다.

- **Technical Details**: 구체적으로, 이 논문에서는 헤리틱스(Heuristics), 통계(Statistics), 기계 학습(Machine Learning), 강화 학습(Reinforcement Learning) 접근 방식을 활용하여 위치 검색 문제를 해결하는 방법을 논의합니다. 특히, 초기 데이터가 부족한 상황에서의 '콜드 스타트(Cold Start)' 문제 및 데이터의 일반화(Generalization)와 차별화(Differentiation), 알고리즘적 편향(Algorithmic Bias) 대응 방법을 중점적으로 다룹니다.

- **Performance Highlights**: 연구 결과, Airbnb 플랫폼은 700만 개 이상의 활성 리스팅을 보유하고 있으며, 기계 학습을 통한 위치 검색 개선이 게스트의 검색 경험을 획기적으로 향상시킬 수 있음을 보여줍니다. 특히, 세분화된 검색 파라미터를 통해 지난 예약 기록에서 학습한 내용을 바탕으로 더 적합한 결과를 제공합니다.



### N-DriverMotion: Driver motion learning and prediction using an event-based camera and directly trained spiking neural networks (https://arxiv.org/abs/2408.13379)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 운전자의 동작을 학습하고 예측하기 위한 새로운 시스템과 이를 교육하기 위해 수집된 고해상도(1280x720) 데이터세트인 N-DriverMotion을 제안합니다. 이 시스템은 스파이크 입력을 나타내는 고해상도 운전 동작 데이터세트를 생성하는 이벤트 기반 카메라로 구성됩니다.

- **Technical Details**: 설계한 네 명의 층으로 이루어진 간소화된 Convolutional Spiking Neural Network (CSNN)은 고해상도 데이터세트를 사용하여 직접 훈련되며, 사전 처리 과정이 필요하지 않습니다. 이는 고해상도 이벤트 기반 스트림에 대한 실시간 추론을 위한 효율적인 SNNs로의 적응을 가능하게 합니다.

- **Performance Highlights**: 제안된 CSNN은 운전자의 동작을 인식하는 데 있어 94.04%의 정확도를 달성하여 자율주행차 또는 효율적인 신경망 아키텍처가 필요한 엣지 디바이스를 위한 안전하고 효율적인 운전 모니터링 시스템 개발에 기여할 수 있습니다.



### Understanding Defects in Generated Codes by Language Models (https://arxiv.org/abs/2408.13372)
- **What's New**: 이 연구는 Large Language Models (LLMs)에 의한 코드 생성의 신뢰성을 조사하며, 생성된 코드에서 발생하는 결함을 식별하고 분석하는 데 중점을 두었습니다. 367개의 결함을 분류하고, 기능 및 알고리즘 오류가 많다는 것을 발견하여 LLMs의 개선 필요성을 강조합니다.

- **Technical Details**: 연구에서는 결함 분류 방법을 사용하여 코드의 종류와 그 복잡성 간의 관계를 분석했습니다. 주요 결함 유형으로는 Functional과 Logic errors가 있으며, Prompt Engineering 기법을 통해 정확도를 높이를 목표로  Scratchpad Prompting, Program of Thoughts Prompting, Chain-of-Thought Prompting, Chain of Code Prompting, Structured Chain-of-Thought Prompting 기법이 적용되었습니다.

- **Performance Highlights**: 이 연구 결과는 Structured Chain-of-Thought (SCoT) 기술이 CodeT5+ 모델에서 약 33.1%, CodeGen 모델에서는 27.8%의 정확도 향상을 가져오는 등, Prompt Engineering 기법이 결함을 크게 줄일 수 있음을 보여줍니다.



### CodeRefine: A Pipeline for Enhancing LLM-Generated Code Implementations of Research Papers (https://arxiv.org/abs/2408.13366)
- **What's New**: 이 논문에서는 연구 방법론을 실행 가능한 코드로 자동 변환하는 CodeRefine라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM (Large Language Models)을 사용하여 코드를 생성하며, 여러 단계로 구성된 접근 방식을 통해 논문에서 주요 텍스트 조각을 추출하고 요약합니다.

- **Technical Details**: CodeRefine는 연구 논문에서 필요한 지식을 지식 그래프(knowledge graph) 형식으로 구조화한 후, 이를 바탕으로 코드를 생성합니다. 생성 과정에는 코드 관련성과 비관련성을 분석하고, ‘Retrospective Retrieval-Augmented Generation (RRAG)’ 접근 방식을 통해 생성된 코드의 정확성을 높입니다.

- **Performance Highlights**: CodeRefine는 다양한 과학 논문을 평가한 결과, 코드 생성의 정확성을 향상시킨 것으로 나타났으며, LLM zero-shot prompting보다 더 신뢰할 수 있는 대안을 제공합니다. 이로 인해 최첨단 알고리즘이 실제 응용 프로그램에 신속하게 채택될 가능성이 높아집니다.



### Reconciling Different Theories of Learning with an Agent-based Model of Procedural Learning (https://arxiv.org/abs/2408.13364)
- **What's New**: 이번 연구에서는 Procedural ABICAP라는 새로운 계산 모델을 제안하며, ICAP, KLI, 그리고 CLT 프레임워크를 통합하여 절차적 지식 학습을 위한 새로운 관점을 제공합니다.

- **Technical Details**: Procedural ABICAP 모델은 ICAP의 네 가지 인지 참여 수준(수동, 능동, 구축적, 상호작용적)을 통해 절차적 학습을 설명하며, 다양한 교육적 설정을 시뮬레이션할 수 있는 실행 가능한 이론으로 제시됩니다.

- **Performance Highlights**: 모델은 기존의 여러 이론적 프레임워크 간의 불일치를 조정하는 데 기여하며, 절차적 학습에서의 더 나은 학습 결과를 도출할 수 있는 잠재력을 가지고 있습니다.



### Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler (https://arxiv.org/abs/2408.13359)
- **What's New**: 본 논문은 새로운 학습률 스케줄러인 PowerLR를 제안하여, 다양한 배치 크기와 훈련 토큰 수에 관계없이 최적의 학습률을 효과적으로 조정할 수 있음을 보여줍니다. 이를 통해 기존의 Cosine 및 WSD 스케줄러가 갖는 단점을 극복하고, 매우 큰 언어 모델에서도 효율적인 훈련을 가능하게 합니다.

- **Technical Details**: PowerLR 스케줄러는 배치 크기(배치 사이즈)와 훈련 토큰 수(트레이닝 토큰)의 영향을 받지 않아, 다양한 모델 크기와 훈련 설정에서 최적의 학습률을 직접 전이할 수 있습니다. 실험을 통해, 최적 학습률은 배치 크기(β) 및 훈련 토큰 수(T)에 대해 파워-로우(power-law) 관계를 따른다는 사실이 밝혀졌습니다.

- **Performance Highlights**: PowerLR와 Maximum Update Parameterization (muP)을 결합한 결과, 훈련 토큰 수, 배치 크기, 모델 크기 및 모델 아키텍처에 관계없이 하나의 하이퍼파라미터 집합으로 지속적으로 인상적인 성능을 달성할 수 있음을 입증했습니다. 특히 3B 밀집 모델(dense model)과 Mixture of Experts(MoE) 모델이 최첨단 소형 언어 모델들과 비교해 상응하는 성능을 보여주었습니다.



### Disentangled Training with Adversarial Examples For Robust Small-footprint Keyword Spotting (https://arxiv.org/abs/2408.13355)
- **What's New**: 이 논문에서는 악의적 예제(adversarial examples)를 효과적으로 적용하여 KWS(Keyword Spotting) 시스템의 견고성을 향상시키는 방법을 탐구하였습니다. 데이터 소스 인식(disentangled learning) 기반의 학습 전략을 통해 원래 데이터와 악의적 데이터 간의 불일치를 줄이는 것을 목표로 하였습니다.

- **Technical Details**: 본 연구의 KWS 모델 아키텍처는 depth-wise separable convolution과 간단한 attention 모듈을 기반으로 합니다. 모델 훈련 시 각 데이터 소스에 대해 별도의 보조 배치 정규화(auxiliary batchnorm)를 사용하여 데이터 소스 간의 보완성을 극대화했습니다. 성능 향상을 위한 제외된 불일치 문제 해결에 초점을 맞추었습니다.

- **Performance Highlights**: 제안된 학습 전략은 내장 데이터셋에서 1%의 잘못된 수락률(false accept rate)에서 40.31%의 잘못된 거부률(false reject rate) 개선을 달성하였으며, Google Speech Commands V1 데이터셋에서는 98.06%의 정확도(accuracy)를 달성했습니다.



### Toward Improving Synthetic Audio Spoofing Detection Robustness via Meta-Learning and Disentangled Training With Adversarial Examples (https://arxiv.org/abs/2408.13341)
Comments:
          IEEE ACCESS 2024

- **What's New**: 이 연구에서는 화자인증(ASV) 시스템의 스푸핑 공격 탐지 성능을 향상시키기 위해 여러 가지 새로운 접근 방식을 제안합니다. 특히, 가중치가 부여된 추가 각 마진 손실(weighted additive angular margin loss)을 통해 데이터 불균형 문제를 해결하고 있으며, 이를 통해 이전에 보지 못한 스푸핑 공격에 대한 일반화 능력을 개선하고자 합니다.

- **Technical Details**: 본 연구는 메타-러닝 손실 함수(meta-learning loss function)를 도입하여 지원 세트(support set)와 질의 세트(query set) 간의 임베딩 차이를 최적화합니다. 우리는 또한 추가적인 데이터 증강(data augmentation) 전략으로 스푸핑 음성에 미세한 교란을 추가하여 적대적 예제(adversarial examples)를 생성하며, 이 과정을 보조 배치 정규화(auxiliary batch normalization)와 결합하여 적용합니다.

- **Performance Highlights**: ASVspoof 2019 데이터셋의 Logical Access 트랙에서 평가한 결과, 제안된 접근 방식은 0.87%의 EER과 0.0277의 최소 t-DCF를 달성하여 스푸핑 공격에 대한 방어 성능이 입증되었습니다.



### LalaEval: A Holistic Human Evaluation Framework for Domain-Specific Large Language Models (https://arxiv.org/abs/2408.13338)
- **What's New**: 이 논문은 LalaEval이라는 도메인 특정 대형 언어 모델(LLMs)을 위한 인간 평가의 포괄적 프레임워크를 제안합니다. LalaEval은 도메인 명세, 기준 설정, 벤치마크 데이터셋 제작, 평가 루브릭 구축 및 평가 결과 분석을 위한 종합적인 프로토콜을 포함하고 있습니다.

- **Technical Details**: LalaEval의 주요 구성 요소들은 다음과 같습니다: (1) Domain Specification, (2) Criteria Establishment, (3) Benchmark Dataset Creation, (4) Construction of Evaluation Rubrics, (5) Analysis and Interpretation of Evaluation Results. 이 프레임워크는 산업의 특정 요구 사항에 맞춘 표준화된 절차를 제공합니다.

- **Performance Highlights**: LalaEval을 물류 산업에 적용한 사례를 통해, 도메인 특정 LLMs의 평가 기준과 데이터셋, 그리고 성능 차이를 비교 분석하여 모델 선택 및 개발을 안내함으로써 프레임워크의 유용성을 입증합니다.



### Mastering the Digital Art of War: Developing Intelligent Combat Simulation Agents for Wargaming Using Hierarchical Reinforcement Learning (https://arxiv.org/abs/2408.13333)
- **What's New**: 이 연구는 군사 분야의 전투 시뮬레이션에서 발생하는 복잡성을 해결하기 위한 포괄적 접근 방식을 제안합니다. 특히, 하이브리드 인공지능 프레임워크와 계층 강화 학습 구조를 통합하여 전투 게임에서의 인공지능 개발을 지원합니다.

- **Technical Details**: 제안된 시스템은 특정 관찰 추상화(targeted observation abstractions), 다중 모델 통합(multi-model integration), 하이브리드 AI 프레임워크(hybrid AI framework), 그리고 계층 강화 학습(HRL) 구조를 포함합니다. 특히, 지역적 관찰 추상화(localized observation abstraction)는 부분 선형 공간 감소(piecewise linear spatial decay)를 사용하여 기존의 전역 관찰(global observation) 방법보다 계산 효율성을 높이고 효율성을 개선합니다.

- **Performance Highlights**: 초기 테스트에서는 향상된 성능이 나타나지 않았지만, 향후 반복(iterations)을 개선하기 위한 인사이트(insights)를 도출할 수 있었습니다. 이러한 연구는 전투 게임에서 AI의 혁신적인 잠재력을 강조하며, 이 분야에서의 지속적인 연구 필요성을 강조합니다.



### Localized Observation Abstraction Using Piecewise Linear Spatial Decay for Reinforcement Learning in Combat Simulations (https://arxiv.org/abs/2408.13328)
- **What's New**: 전투 시뮬레이션 분야에서, 딥 강화 학습 (deep reinforcement learning) 에이전트의 훈련 및 배치에서 기존의 한계를 극복하기 위한 새로운 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법은 부분선형 공간감소 (piecewise linear spatial decay) 를 이용한 국소적 관찰 추상화 (localized observation abstraction) 입니다. 이 기법은 상태 공간 (state space)을 단순화하여 계산 요구 사항을 줄이는 동시에 필수 정보를 유지하여 AI 훈련 효율을 향상시킵니다.

- **Performance Highlights**: 이 연구에서 제안한 국소적 관찰 접근 방식은 증가하는 시나리오 복잡도에서도 전통적인 전역 관찰 접근 방식보다 일관되게 우수한 성능을 보임을 보여줍니다.



### An Overview and Comparison of Axiomatization Structures Regarding Inconsistency Indices' Properties in Pairwise Comparisons Methods (https://arxiv.org/abs/2408.13297)
Comments:
          21 pages, 2 figures

- **What's New**: 이번 논문에서는 분석적 계층 프로세스(Analytic Hierarchy Process, AHP)에서 판단 불일치성을 측정하는 핵심적인 역할을 하는 불일치 지수(Inconsistency Index)에 대한 수학적 분석을 다룹니다. 지난 10년간 불일치 지수의 공리화(axiomatization)에 대한 발전을 종합적으로 검토하고, 다양한 공리적 구조에 대한 비교 및 논의를 제공합니다.

- **Technical Details**: 불일치 지수는 쌍대 비교 행렬(Pairwise Comparison Matrix, PCM)을 실수(real number)로 변환하는 수학적 함수입니다. 연구 커뮤니티에서는 이러한 불일치 지수가 신뢰성을 갖기 위한 적절한 성질(Properties)을 만족해야 한다고 주장하고 있습니다.

- **Performance Highlights**: 본 연구는 독립적으로 제안된 수많은 공리적 프레임워크(axiomatic frameworks)를 종합하여, 불일치 지수에 대한 보다 넓은 프레임워크가 필요함을 강조합니다. 또한 앞으로의 연구 방향에 대한 제언도 포함되어 있습니다.



### Causally-Aware Spatio-Temporal Multi-Graph Convolution Network for Accurate and Reliable Traffic Prediction (https://arxiv.org/abs/2408.13293)
- **What's New**: 본 논문에서는 교통 예측이라는 특수한 공간-시간(spatio-temporal) 학습 문제의 사례를 통해, 명시적(explicit) 및 암시적(implicit) 교통 패턴을 통합하여 예측 성능을 향상시킬 수 있는 고급 딥러닝 모델을 제안합니다.

- **Technical Details**: 연구에서는 세 가지 주요 구성 요소인 동적 인과 구조 학습(dynamic causal structure learning), 인과적으로 인식된 공간-시간 다중 그래프 합성곱 신경망(CASTMGCN), 그리고 예측 불확실성 정량화(conformal prediction)를 활용하여 정확하고 신뢰할 수 있는 교통 예측을 생성하는 최적화된 프레임워크를 구축했습니다. CASTMGCN은 다양한 그래프를 융합하여 교통 네트워크의 다양한 측면을 포착하며, 이러한 그래프는 외부 요인의 영향을 도로 네트워크에 반영합니다.

- **Performance Highlights**: 실제 교통 데이터셋에서 실험한 결과, 제안된 방법이 여러 최신 모델들보다 예측 정확도가 개선되었고, 통계적 유효성을 엄격히 충족하면서도 더 효율적인 예측 범위를 생성함을 확인했습니다.



### Abstract Art Interpretation Using ControlN (https://arxiv.org/abs/2408.13287)
Comments:
          5 pages, 4 figures

- **What's New**: 이 연구는 추상적인 예술 해석과 텍스트-이미지 합성을 융합하는 방법을 탐구하며, 텍스트 프롬프트를 통해 이미지 구성에 대한 정밀한 공간적 제어를 달성하는 문제를 다룹니다. ControlNet의 기능을 활용하여 사용자에게 합성 과정에 대한 더 섬세한 제어를 제공합니다. 특히 우리는 삼각형과 같은 기하학적 원소에서 영감을 받은 새로운 조건을 도입했습니다.

- **Technical Details**: 본 연구에서 사용된 데이터셋은 37.6백만 이미지-텍스트 쌍으로 구성된 WIT(Wikipedia-based Image Text) 데이터셋에서 세심하게 구성되었습니다. ControlNet 모델을 훈련시키기 위해 WIT 데이터셋에서 다운로드된 원본 이미지를 사용했으며, BLIP 모델을 통해 이미지에 대한 캡션을 생성했습니다. ControlNet은 추가적인 조건으로서 세부화된 기하학적 도형을 사용하고 있으며, 이미지 합성의 과정에서 이러한 도형을 통해 더 나은 조작과 해석이 가능합니다.

- **Performance Highlights**: 우리가 제안한 방법의 성능 평가는 정성적 평가를 통해 이루어지며, 생성된 이미지의 질, 조건의 충실도, 그리고 전반적인 이미지 품질이 향상됨을 보여줍니다. ControlNet의 고유한 아키텍처 덕분에, 우리는 텍스트 입력을 통한 이미지 생성에서의 제어 가능성을 극대화시킬 수 있었습니다.



### SIn-NeRF2NeRF: Editing 3D Scenes with Instructions through Segmentation and Inpainting (https://arxiv.org/abs/2408.13285)
Comments:
          Code is available at: this https URL

- **What's New**: 이번 연구에서는 3D 장면에서 객체를 배경과 분리하여 선택적으로 편집할 수 있는 새로운 방법인 SIn-NeRF2NeRF (sn2n)를 제안합니다. 이 방법은 이전의 Instruct-NeRF2NeRF (in2n) 기법을 개선하여 객체를 더욱 정교하게 수정할 수 있도록 합니다.

- **Technical Details**: sn2n은 객체 마스크와 텍스트 프롬프트를 입력으로 받아, 객체와 배경을 분리한 후 3D 장면을 수정합니다. 이를 위해 멀티뷰 분할 기법과 SPIn-NeRF를 사용하여 객체의 편집과 3D 배경 인페인팅을 수행하였습니다. 최종적으로 두 장면을 병합하여 편집된 3D 장면을 생성합니다.

- **Performance Highlights**: sn2n은 3D 장면에서 객체를 효과적으로 분리하고 편집함으로써 기존 방법들보다 더 나은 결과를 보여주며, 다양한 예시를 통해 resizing과 이동이 가능한 것을 입증하였습니다.



### From Radiologist Report to Image Label: Assessing Latent Dirichlet Allocation in Training Neural Networks for Orthopedic Radiograph Classification (https://arxiv.org/abs/2408.13284)
Comments:
          This article is an abridged version of a 2016 master's thesis at the Karolinska Institute. The original is available upon request

- **What's New**: 이 연구는 스웨덴 Danderyd 병원에서 수집한 손목과 발목 X-ray (radiography) 이미지를 활용하여, 기계 학습 (machine learning, ML)과 인공 신경망 (artificial neural networks, ANN)을 통해 정형외과 방사선 사진의 해석을 개선하고자 하였습니다.

- **Technical Details**: 연구 방법으로는 2002년부터 2015년까지의 방사선과 의사 보고서와 함께 제공된 X-ray 이미지를 활용했습니다. LDA (Latent Dirichlet Allocation) 기법을 사용하여 방사선 사진에 대한 라벨을 생성하고, 이를 ANN 훈련에 사용했습니다. 생성된 라벨에 기반하여 ANN의 출력 결과를 수작업으로 검토하여 정확성을 평가했습니다.

- **Performance Highlights**: LDA를 통해 생성된 이미지 라벨은 ANN 훈련에 성공적으로 사용되었으며, ANN의 정확도는 라벨에 따라 60%에서 91% 사이로 나타났습니다. 그러나 LDA는 방사선과 보고서를 기반으로 정형외과 X-ray 라벨링에 높은 정확도를 제공하는 데 적합하지 않았습니다. 그럼에도 불구하고 ANN은 X-ray 이미지에서 특정 특징을 고도로 정확하게 감지할 수 있었습니다.



### Retrieval-Augmented Generation Meets Data-Driven Tabula Rasa Approach for Temporal Knowledge Graph Forecasting (https://arxiv.org/abs/2408.13273)
Comments:
          Paper was accepted at ACM KDD -2024 -- Undergraduate Consortium. Please find the link: this https URL

- **What's New**: 본 논문에서는 sLA-tKGF(Temporal Knowledge Graph Forecasting를 위한 소규모 언어 모델)를 소개하여 기존의 문제점인 부정확한 정보 회상, 환각, 편향 및 데이터 유출 문제를 해결하고 있습니다.

- **Technical Details**: sLA-tKGF는 Retrieval-Augmented Generation (RAG) 방식을 통해, 역사적 데이터와 웹 검색 결과를 활용하여 지식이 포함된 프롬프트를 구성하고, 예측 정확도를 높이기 위해 다층 스택드 바닐라 트랜스포머 아키텍처를 기반으로 하여 처음부터 맞춤형으로 훈련됩니다.

- **Performance Highlights**: 엄격한 실험 결과, 이 프레임워크는 공개된 데이터셋에서 SOTA 성능을 보여주며, 예측의 해석 가능성과 신뢰성을 보장합니다.



### Efficient Task Transfer for HLS DSE (https://arxiv.org/abs/2408.13270)
Comments:
          13 pages, 7 figures, accept to ICCAD'24

- **What's New**: 최근 시중에 나온 고급 합성 (HLS) 도구들을 기반으로 한 모델 기반 최적화 방법을 활용하여 도메인 특화 아키텍처 설계의 생산성을 향상시키기 위한 여러 연구가 진행되고 있습니다. 본 연구에서는 HLS 설계 공간 탐색 (DSE)에서의 도전 과제를 다루며, 새로운 Active-CEM이라는 작업 전이 학습 방식이 도구 체인의 변화에 효율적으로 적응할 수 있도록 설계되었습니다.

- **Technical Details**: Active-CEM은 고품질 설계 구성을 식별하고, 새로운 도구 체인 아래에서 샘플 효율성을 최적화하는 모델 기반 탐색기를 이용합니다. 이 방법론은 도구 체인 불변 모델링을 통합함으로써 QoR(결과 품질)의 변화를 보다 정확하게 예측할 수 있게 해줍니다. 실험에서는 HLSyn 벤치마크에서 수행된 결과로, AutoDSE에 비해 1.58배 향상된 성능을 보였으며 HARP에 대해서도 1.2배 개선된 성과를 거두었습니다.

- **Performance Highlights**: 실험 결과, 새로운 도구 체인으로 전환 시 평균 성능 향상은 2배 (2×), 샘플 효율성은 5.26배 증가하였고, 실행 시간은 2.7배 단축되었습니다.



### Exploiting Formal Concept Analysis for Data Modeling in Data Lakes (https://arxiv.org/abs/2408.13265)
- **What's New**: 이번 연구에서는 데이터 레이크(Data Lake)에서 비정형 데이터를 효율적으로 정리하고 분석하기 위해 Formal Concept Analysis (FCA)를 활용한 새로운 방법론을 소개합니다.

- **Technical Details**: FCA는 데이터 구조를 객체로 표현하고, 이 객체들을 기반으로 개념 격자(concept lattice)를 분석하여 데이터 모델을 통합하는 두 가지 전략, 즉 top-down 및 bottom-up 접근 방식을 사용합니다.

- **Performance Highlights**: 연구 대상 데이터 레이크의 필드 이름 수가 54% 감소(190에서 88로)하면서 80%의 데이터 구조를 34개의 필드 이름으로 충족시킨 결과를 달성했습니다.



### Improving Language Models for Emotion Analysis: Insights from Cognitive Scienc (https://arxiv.org/abs/2406.10265)
- **What's New**: 이번 연구에서는 감정(emotion) 및 소통(communication)에 관한 인지 과학(cognitive science) 연구를 활용하여 감정 분석을 위한 언어 모델(language models)의 향상을 제안합니다.

- **Technical Details**: 심리학(psychology) 및 인지 과학의 주요 감정 이론을 설명하고, 자연어 처리(natural language processing)에서의 감정 주석(annotation) 방법과 그 심리학적 이론과의 연결성을 소개합니다. 또한, 인지 실용주의(cognitive pragmatics)에서의 감정 소통 분석의 두 가지 주요 유형을 다룹니다.

- **Performance Highlights**: 이번 연구는 인간의 감정과 소통의 다양한 측면을 고려하여 감정 이해를 위한 새로운 주석 체계를 구축하고, 가능한 벤치마크(benchmark)를 제안하여 언어 모델 개선 방향을 제시합니다.



### SarcasmBench: Towards Evaluating Large Language Models on Sarcasm Understanding (https://arxiv.org/abs/2408.11319)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)이 풍자(sarcasm) 이해에서 직면한 한계에 대한 종합적인 평가를 제시합니다. 이를 위해 11개의 최신 LLM과 8개의 사전 훈련된 언어 모델(PLMs)의 성능을 다양한 기준 데이터셋에서 비교 분석했습니다.

- **Technical Details**: 연구는 세 가지 프롬프트 방법: zero-shot IO 프롬프트, few-shot IO 프롬프트, chain of thought (CoT) 프롬프트를 사용하여 풍자 감지에서 LLM의 성능을 평가했습니다. 결과적으로, LLM은 감독형 PLM과 비교하여 부족한 성능을 보였으며, GPT-4가 타 모델에 비해 평균 14.0% 이상의 개선을 보여주었습니다.

- **Performance Highlights**: 현재 LLMs는 감독형 PLMs에 비해 풍자 감지 기준에서 부족한 성능을 보이며, 풍자 이해를 개선하기 위한 많은 노력이 필요합니다. GPT-4는 여러 프롬프트 방법에서 지속적이고 유의미하게 다른 LLMs보다 우수한 성능을 발휘했습니다. 또한, few-shot IO 프롬프트 방법이 zero-shot IO 및 few-shot CoT 방식보다 성능이 뛰어남을 확인했습니다.




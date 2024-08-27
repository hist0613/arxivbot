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




New uploads on arXiv(cs.CL)

### Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs (https://arxiv.org/abs/2501.18585)
- **What's New**: 본 연구는 오픈AI의 o1과 같은 대형 언어 모델(LLMs)에서 발견된 'underthinking'(사고 부족) 현상을 조명합니다. 이 현상은 모델이 유망한 추론 경로를 충분히 탐색하지 않고 자주 다른 사고로 전환하여 깊이 있는 사고를 하지 못하게 합니다. 이를 분석하기 위해 다양한 테스트 세트를 사용하여 실험을 수행하였으며, 성공적인 연산 결과와 관련된 통찰력을 제공하고 있습니다.

- **Technical Details**: 연구에서는 잘 알려진 오픈소스 모델인 QwQ-32B-Preview와 DeepSeek-R1-671B를 사용하여 o1 유사 모델의 사고 전환을 분석합니다. 세 가지 테스트 세트(MATH500, GPQA Diamond, AIME2024)에 대한 실험이 진행되었으며, 주로 정답과 오답 간의 토큰 사용과 사고 전환 패턴의 차이를 비교합니다. 연구팀은 새로운 'underthinking' 메트릭을 도입하여 잘못된 응답에서의 토큰 효율성을 정량적으로 평가합니다.

- **Performance Highlights**: 제안하는 'thought switching penalty'(TIP) 전략은 모델이 각 추론 경로를 깊이 탐색하도록 유도하며, 실험 결과 이 접근 방식이 도전적인 데이터셋에서 정확성을 향상시킨다는 것을 보여주었습니다. TIP을 적용한 결과, 모델이 추가적인 미세 조정 없이도 높은 정확도를 달성할 수 있음을 확인했습니다. 연구는 o1과 유사한 LLM의 사고 비효율성을 이해하는 데 기여하며, 문제 해결 능력을 개선할 수 있는 실질적인 솔루션을 제시하고 있습니다.



### R.I.P.: Better Models by Survival of the Fittest Prompts (https://arxiv.org/abs/2501.18578)
- **What's New**: 본 논문에서는 데이터 품질이 모델 성능에 미치는 영향을 측정하는 Rejecting Instruction Preferences (RIP) 방법을 제안합니다. 이 방법은 저품질 프롬프트가 높은 변동성과 낮은 품질의 응답을 초래한다는 가정하에 데이터 무결성을 평가합니다. RIP는 기존의 훈련 데이터에서 프롬프트를 필터링하거나 고품질의 합성 데이터셋을 생성하는 데 사용됩니다.

- **Technical Details**: RIP 방법은 선택된 응답과 거부된 응답 쌍을 기반으로 프롬프트를 필터링하며, 거부 응답 품질과 선택된 응답과의 보상 차이를 주요 메트릭으로 삼습니다. 본 방법은 Direct Preference Optimization (DPO)와 같은 강화학습 방법을 사용하여 모델을 미세 조정하는 데 활용되며, 저품질 프롬프트를 효과적으로 제거할 수 있습니다. 구체적으로, RIP는 Wildchat 프롬프트에 대해 필터링하여 Llama 3.1-8B-Instruct와 Llama 3.3-70B-Instruct 모델에서 성능 향상을 보였습니다.

- **Performance Highlights**: RIP는 AlpacaEval2 LC Win Rate를 9.4%, Arena-Hard를 8.7%, WildBench를 9.9% 향상시켰습니다. Llama 3.3-70B-Instruct 모델에서는 Arena-Hard에서 순위가 18위에서 6위로 올라가는 성과를 보였습니다. 이러한 성과는 RIP 필터링 방법이 모델 성능을 획기적으로 향상시킬 수 있음을 보여줍니다.



### Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method (https://arxiv.org/abs/2501.18539)
- **What's New**: 이 논문은 복잡한 질문에 대한 정보를 검색하는 데 있어 LLM 기반의 새로운 retrieval 방법인 ARM을 제안합니다. ARM은 질문과 데이터 수집의 조직 간의 정렬을 메꿔, 기존의 검색 방법이 간과할 수 있는 필요한 정보에 대한 더 나은 탐색을 가능하게 합니다. 특히, 반복적인 쿼리가 아닌 문맥에 맞는 데이터 오브젝트 간의 관계를 분석함으로써 효율성을 극대화할 수 있습니다.

- **Technical Details**: ARM은 LLM(대형 언어 모델)의 추론 능력을 활용하여 질문에 필요한 데이터를 효율적으로 검색하는 방식으로 설계되었습니다. 이 시스템은 정보 정렬(information alignment)과 구조 정렬(structure alignment) 단계를 통해 데이터를 정리하며, 자가 검증(self-verification) 과정을 통해 최종적으로 적합한 데이터 오브젝트를 선택합니다. 이를 위해 N-gram을 사용하여 각 오브젝트의 주요 정보를 요약하고, 임베딩(embedding)을 통해 의미적 유사성 검색을 지원합니다.

- **Performance Highlights**: 실험 결과 ARM은 Bird 데이터셋에서 기존의 RAG 방법들보다 최대 15.9 포인트의 정확도로 우수한 성능을 보였습니다. OTT-QA 데이터셋에서도 ARM은 이전 방법들에 비해 최대 19.3 포인트 높은 F1 점수를 기록하며, LLM 기반의 질문 대응 문제에서 효과적인 해결책을 제시합니다. 이로 인해 복잡한 질문에 대한 검색 능력이 한층 향상되었습니다.



### Differentially Private Steering for Large Language Model Alignmen (https://arxiv.org/abs/2501.18532)
Comments:
          ICLR 2025; Code: this https URL

- **What's New**: 이 연구는 Large Language Models(LLMs)의 행동을 개인 데이터셋과 정렬하는 최초의 연구를 제시합니다. 	extit{Private Steering for LLM Alignment(PSA)} 알고리즘을 개발하여, 비유출 개인 정보의 보장을 위한 차별적 개인 정보 보호(Differential Privacy, DP)를 사용합니다. 연구를 통해 PSA가 성능 손실을 최소화하면서도 LLM 정렬에 대한 DP 보장을 달성할 수 있음을 보여주었습니다.

- **Technical Details**: 이 연구는 LLM의 활성화(activation)를 편집하는 새로운 방법인 PSA를 통해 여러 공개 데이터셋에서 진행된 광범위한 실험 결과를 보고합니다. PSA는 정렬 목적에 맞는 긍정적(예: 사실적인) 사례의 정보를 보존하고 부정적(예: 환상) 사례의 정보를 최소화하여, LLM의 활성화를 수정합니다. 본 연구는 또한 활성화 편집을 통한 개인정보 보호의 중요성에 대해 논의하며, Membership Inference Attack(MIA)에 대한 첫 번째 연구를 제안합니다.

- **Performance Highlights**: PSA는 0.5B에서 7B 규모의 다양한 오픈 소스 LLM을 사용한 7개의 데이터셋에서 시험되었습니다. 결과에 따르면 PSA는 LLM의 정렬 메트릭 및 열린 텍스트 생성 품질에서 minimal한 성능 손실을 보여주며 DP 보장을 달성했습니다. 또한 PSA는 활성화 편집을 통한 개인 정보 유출 위험을 낮추는 효과를 입증하였습니다.



### Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch (https://arxiv.org/abs/2501.18512)
- **What's New**: 이 논문은 DiLoCo 알고리즘을 개선하는 세 가지 방법을 제안합니다. 첫째, 모든 파라미터를 동시에 동기화하는 대신, 파라미터의 부분 집합을 순차적으로 동기화하여 피크 대역폭을 크게 줄입니다. 둘째, 동기화와 동시에 훈련을 계속할 수 있도록 하여 전체 시간(wall clock time)을 감소시킵니다.

- **Technical Details**: 제안된 방법에서는 파라미터 통신을 최소화하는 방식으로, 각 워커는 서로 간단한 데이터 양자화(quantization)를 통해 교환합니다. 이를 통해 워커 간에 요구되는 대역폭을 두 자릿수만큼 줄일 수 있습니다. 이 방법을 통해 수십억 개의 파라미터를 조정하면서도 예전과 유사한 품질을 유지할 수 있게 됩니다.

- **Performance Highlights**: Streaming DiLoCo는 원래의 DiLoCo보다 성능이 확실히 우수하며, 데이터 병렬(Data-Parallel) 방식과 유사한 성능을 저대역폭으로 달성합니다. 실험을 통해 이 접근 방식이 대역폭 사용을 최소화하면서도 높은 학습 효율성을 유지한다는 점을 확인했습니다.



### CALM: Unleashing the Cross-Lingual Self-Aligning Ability of Language Model Question Answering (https://arxiv.org/abs/2501.18457)
Comments:
          Accepted by NAACL 2025

- **What's New**: 이 논문은 Cross-Lingual Self-Aligning 능력(CALM)을 도입하여 다국어 모델의 지식을 언어 간에 정렬하는 방법을 제안합니다. CALM은 서로 다른 언어로 작성된 여러 응답을 샘플링하고, 가장 일관된 응답을 긍정 샘플로 선택하여 나머지는 부정 샘플로 활용합니다. 또한, 직접 선호 최적화(Direct Preference Optimization, DPO)를 사용하여 모델의 다국어 지식을 정렬합니다. 이를 통해 지식의 일관성을 높이며 다국어 질문 응답 성능을 향상시킵니다.

- **Technical Details**: CALM의 구현은 세 가지 단계로 구성됩니다. 첫째, 다양한 언어에서 Chain-of-Thought(CoT) 응답을 샘플링합니다. 둘째, 다수결을 통해 선택된 응답을 긍정 샘플로 설정한 후, 그에 부합하지 않는 다른 응답과 쌍을 만들어 DPO 훈련에 사용합니다. 또한 이 방법은 외부 지식 소스와 통합되어 Self-supervised Retrieval-Augmented Generation(Self-RAG) 기법을 통해 효율적으로 지식을 검색할 수 있게 합니다.

- **Performance Highlights**: MEDQA와 X-CSQA 데이터셋에서 실험한 결과, CALM은 각각 3.76% 및 5.55%의 정확도를 향상시켰습니다. 다양한 언어를 포함하면 CALM 훈련의 정확성과 일관성이 더욱 향상됩니다. 실험 결과는 CALM이 언어 간 지식 정렬에 있어 효과적임을 보여주고, 내부 및 외부 지식의 호환성을 증진 시킵니다.



### GENIE: Generative Note Information Extraction model for structuring EHR data (https://arxiv.org/abs/2501.18435)
- **What's New**: 이 논문에서는 GENIE라는 새로운 Generative Note Information Extraction 시스템을 소개합니다. GENIE는 대규모 언어 모델(LLM)을 활용해 비구조화된 클리니컬 텍스트를 표준화된 형식으로 변환하여 유용한 데이터로 만드는 데 중점을 둡니다. 기존 방법들과는 달리 하나의 추출 단계에서 전체 단락을 처리하여 높은 정확도로 정보를 추출하는 효율성을 자랑합니다. 이를 통해 클리니컬 노트의 다양한 구조화 과제를 간소화하고 오류를 줄일 수 있습니다.

- **Technical Details**: GENIE는 추출할 수 있는 엔티티(entity), 주장 상태(assertion status), 위치(location), 수정자(modifier), 값(value), 목적(purpose) 등을 신속하고 정확하게 처리합니다. 이 시스템은 강력한 데이터 준비 파이프라인 구조를 사용하고 소규모 LLM을 미세 조정하여 여러 정보 추출 작업에서 경쟁력 있는 성능을 달성합니다. 특히, GENIE는 자동화된 방식으로 추가 속성(extra attributes)을 쉽게 처리할 수 있습니다. 이로 인해 헬스케어 시스템에서의 실제 적용성과 확장성이 크게 강화됩니다.

- **Performance Highlights**: GENIE는 전통적인 도구인 cTAKES 및 MetaMap보다 우수한 성능을 보이며, 고성능 정보 추출 작업에서 성공적인 결과를 나타냅니다. 이 시스템은 인공지능 모델과 테스트 데이터를 오픈 소스 방식으로 공개하여 협업을 촉진하고 EHR 구조화 분야의 Further advancements를 도모할 계획입니다. 이러한 접근은 헬스케어 분야에서의 활용 가능성을 높임과 동시에 기술의 진보를 앞당기는 데 기여할 것입니다.



### RbFT: Robust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects (https://arxiv.org/abs/2501.18365)
- **What's New**: 이 논문은 Retrieval-augmented generation (RAG) 시스템의 신뢰성을 높이기 위해 Robust Fine-Tuning (RbFT)이라는 새로운 방법론을 제안합니다. 기존 RAG 시스템의 정보 검색 과정에서 발생할 수 있는 오류나 불완전한 정보로 인한 문제를 해결하고자 합니다. RbFT는 LLM의 방어 능력을 강화하여 부정확한 정보에도 불구하고 신뢰할 수 있는 응답을 생성하도록 돕습니다.

- **Technical Details**: RbFT는 두 가지 주요 세부 과제로 구성되어 있습니다: Defects Detection(결함 탐지)와 Utility Extraction(유용성 추출)입니다. 이 방법들은 LLM이 결함이 있는 입력을 평가하고, 유용한 정보를 효과적으로 활용할 수 있도록 돕습니다. LLM은 결함이 있는 문서를 사용하여 실제 응답을 생성하는 훈련을 받고, 이는 LAG 시스템의 전반적인 내구성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RbFT는 다양한 검색 조건에서도 RAG 시스템의 견고성을 크게 향상시켰습니다. 기존의 방법들을 초월하며, 높은 추론 효율성과 다른 견고성 강화 기법과의 호환성을 유지합니다. 이 연구는 RAG 시스템의 성능과 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Mining for Species, Locations, Habitats, and Ecosystems from Scientific Papers in Invasion Biology: A Large-Scale Exploratory Study with Large Language Models (https://arxiv.org/abs/2501.18287)
Comments:
          8 pages, 2 figures, accepted to the NLP4Ecology Workshop 2025 (this https URL) co-located with the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies

- **What's New**: 이 논문은 침입 생물학(invasion biology) 문헌에서 핵심 생태학적 개체(ecological entity)를 추출하기 위해 대형 언어 모델(large language models, LLMs)의 능력을 활용한 탐색적 연구를 제시합니다. 종 이름(species names), 위치(locations), 서식지(habitats) 및 생태계(ecosystems)에 대한 정보를 추출하는 것에 중점을 두고 있으며, 이는 종의 확산(spread) 이해와 향후 침입 예측, 보존(conservation) 노력에 필요한 정보입니다. 이 연구는 LLM을 활용한 생태학적 개체 추출의 가능성과 한계를 밝혀내고, 생물 침입 관리 및 이해를 위한 더 정교한 자동화된 지식 추출 도구의 기초를 다지기로 합니다.

- **Technical Details**: 전통적인 텍스트 마이닝(text mining) 접근 방식은 생태학적 용어의 복잡성과 이러한 텍스트에서 나타나는 미묘한 언어 패턴에 어려움을 겪는 경우가 많습니다. 이 논문은 도메인 특화된 미세 조정(fine-tuning) 없이 일반 목적의 LLM을 적용하여 생태학적 개체를 추출한 결과를 제시합니다. 이를 통해 LLM의 활용 가능성을 탐색하며, 생물학적 침입 이해를 위한 도구의 발전 방향을 제시합니다.

- **Performance Highlights**: 이 연구에서는 LLM을 이용한 정보 추출이 생태학적 텍스트에서 어떻게 효과적으로 이루어질 수 있는지를 보여주며, 이를 통해 향후 연구자와 실무자들이 생물 침입을 관리하고 이해하는 데 있어서 활용할 수 있는 기반을 마련합니다. LLM의 탐색 결과는 이 기술이 생태학에서의 정보 추출에 있어 잠재력을 가지지만 동시에 한계도 있음을 나타냅니다. 논문의 결과는 생태학적 연구 분야에서 LLM의 적용 가능성을 더욱 깊이 이해하는 데 기여할 것입니다.



### Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models (https://arxiv.org/abs/2501.18280)
- **What's New**: 최근 큰 언어 모델(LLMs)의 보안 이슈가 많은 주목을 받으면서, 해로운 출력을 방지하기 위한 다양한 방어 메커니즘이 개발되고 있습니다. 본 논문에서는 텍스트 임베딩 모델을 기반으로 한 안전장치가 이러한 방어의 기초가 됨을 발견하였습니다. 특히, 텍스트 임베딩 모델의 출력 분포가 큰 평균을 가지며 상당히 편향되어 있다는 관찰에서 출발해 보편적인 마법 단어(universal magic words)를 검색하는 새로운 효율적인 방법을 제안합니다.

- **Technical Details**: 보편적인 마법 단어는 텍스트 뒤에 붙여져 어떤 텍스트의 임베딩을 편향된 방향으로 이동시키고, 이는 두 텍스트 쌍의 유사성을 조작해 안전장치를 오도할 수 있습니다. 본 연구는 세 가지 방법을 사용하여 이러한 마법 단어를 발견하는 접근 방식을 설명합니다: 1) Brute-force search는 기준선으로 사용되며, 2) Black box 방법은 비대칭 방향과 유사한 텍스트 임베딩 단어를 찾아내고, 3) White box 방법은 임베딩을 편향 방향에 가깝게 만들 마법 단어 접미사를 찾습니다.

- **Performance Highlights**: 실험 결과, 세 가지 방법 모두 최적의 마법 단어를 찾는 것으로 나타났지만, Method 2와 Method 3은 Method 1보다 훨씬 더 효율적이었습니다. 특히, Method 3은 다중 토큰 마법 단어를 검색할 수 있어 유용성이 높습니다. 이를 통해 우리는 LLM 보안 시스템에서의 안전장치가 해로운 콘텐츠를 탐지하는 데 실패하는 취약점을 지적하며, 이러한 공격에 대한 방어 메커니즘도 제안했습니다.



### How to Select Datapoints for Efficient Human Evaluation of NLG Models? (https://arxiv.org/abs/2501.18251)
- **What's New**: 이번 연구에서는 텍스트 생성 모델 평가에서 자주 사용되는 인간 평가의 비용 문제를 해결하기 위한 새로운 선택 기법(selectors)을 개발하였습니다. 기존의 무작위 선택(random selection)보다 훨씬 정보량이 높은 데이터 포인트를 선택할 수 있는 방법을 제시합니다. 이를 통해 평가 비용을 줄이면서도 보다 효율적인 모델 비교가 가능하게 합니다.

- **Technical Details**: 연구에서 제안한 선택 기법은 자동 메트릭 점수의 분산(variance), 모델 출력의 다양성(diversity), 아이템 반응 이론(Item Response Theory)을 기반으로 성능이 검증되었습니다. 또한, 모델 출력이 없는 상황에서도 활용할 수 있는 소스 기반 추정기(source-based estimators)를 도입하여 원본 텍스트만으로 인간 평가에 유용한 아이템을 예측할 수 있습니다.

- **Performance Highlights**: 기계 번역(machine translation)과 요약(summarization)이라는 두 가지 일반적인 자연어 생성(NLG) 작업에서 제안된 선택 기법의 효과성을 입증했습니다. 전체 테스트 데이터의 약 50%만을 사용하여도 동일한 평가 결과를 얻을 수 있음을 보였습니다. 이러한 구현은 subset2evaluate 패키지에서 제공됩니다.



### Contextually Structured Token Dependency Encoding for Large Language Models (https://arxiv.org/abs/2501.18205)
- **What's New**: 본 연구는 Contextually Structured Token Dependency Encoding이라는 새로운 접근법을 제안하며, 이는 토큰 임베딩 내에 계층적이고 상호 의존적인 관계를 직접적으로 통합하는 것을 목표로 합니다. 기존의 자가 주의 메커니즘은 토큰 간의 관계를 명시적으로 표현하는 데 한계를 보였던 반면, 제안된 방식은 토큰 의존성을 구조화된 임베딩으로 명확히 표현합니다.

- **Technical Details**: 제안하는 인코딩 메커니즘은 그래프 기반 접근법을 사용하여 토큰 의존성을 구조화된 임베딩으로 표현하며, 이 과정에서 문맥적 의존성을 고려하여 임베딩 초기화 및 변환을 동적으로 수행합니다. 이는 전통적인 임베딩 기법이 포지션 인코딩과 학습된 문맥적 연관성에만 의존하는 것과 대조적입니다.

- **Performance Highlights**: 실증적인 평가 결과, 제안된 접근법은 다양한 언어적 벤치마크에서 perplexity를 감소시키는 결과를 보여줍니다. 또한, 문맥 일관성 및 예측 일관성이 개선되었음을 시사하며, 더 긴 시퀀스에 대한 의존 관계 정렬에서 특히 강화된 성능을 나타냈습니다.



### Mixed-Precision Graph Neural Quantization for Low Bit Large Language Models (https://arxiv.org/abs/2501.18154)
Comments:
          ICASSP 2025

- **What's New**: 이 논문에서 소개하는 Mixed-precision Graph Neural PTQ (MG-PTQ) 방법은 기존의 Post-Training Quantization (PTQ) 기술의 저비트 환경(<3 bits)에서의 성능 저하 문제를 해결하기 위해 제안되었습니다. 이 방법은 그래프 신경망(GNN)을 활용하여 가중치 간의 의존성을 파악하고, 그 중요도에 따라 양자화 비트 폭을 동적으로 할당합니다. 이로 인해 양자화 성능이 크게 향상되며, 저비트 조건에서도 새로운 벤치마크를 수립할 수 있게 되었습니다.

- **Technical Details**: MG-PTQ 방법은 GNN 모듈을 통해 가중치의 중요도를 보다 정확히 파악하고, 이에 따라 다채로운 비트 폭을 할당하는 방식을 취합니다. GNN은 훈련 중에 양자화 오류를 최소화하면서도 평균 비트 폭을 조절하는 제약 조건을 따릅니다. 이를 통해 모델의 가중치 의존성과 중요도를 반영하여 최적의 양자화 전략을 수립할 수 있습니다.

- **Performance Highlights**: WikiText2와 C4 데이터셋에 대한 광범위한 실험 결과, MG-PTQ 방법은 기존의 최신 PTQ 방법인 GPTQ보다 우수한 성능을 보였으며, 저비트 환경에서 두드러진 양자화 성능을 입증했습니다. 이 연구는 GNN 프레임워크를 활용한 적응형 양자화의 첫 사례로, 계산 효율성과 적응성을 모두 갖춘 방법으로 기존 PTQ 접근 방식을 능가하는 것을 목표로 하고 있습니다.



### Unraveling the Capabilities of Language Models in News Summarization (https://arxiv.org/abs/2501.18128)
- **What's New**: 이 연구에서는 최근 20개의 언어 모델을 종합적으로 벤치마킹하며, 특히 뉴스 요약 작업에 중점을 두었습니다. 소규모 모델들에 대한 성능을 평가하고, 그들의 능력과 효율성을 제로샷(zero-shot) 및 몇 샷(few-shot) 학습 설정에서 분석한 결과, 몇몇 모델이 GPT-3.5-Turbo와 GPT-4에 비해 경쟁력 있는 대안으로 자리 잡을 가능성을 보였습니다. 또한, 데모 예시를 포함한 경우 성능 향상 없이 오히려 품질 저하를 초래한 사례가 있음을 강조하였습니다.

- **Technical Details**: 연구에서는 20개의 최신 언어 모델의 성능을 평가하기 위해 다각적인 평가 방식을 적용했습니다. 자동화된 메트릭, 인간 평가, AI 기반 평가를 결합하여 모델의 요약 품질에 대한 신뢰성 있는 분석을 제공하였습니다. 특히, 뉴스 기사를 다루며 서로 다른 스타일로 작성된 세 가지 데이터 세트를 사용하여 모델의 능력을 체계적으로 테스트했습니다.

- **Performance Highlights**: 결과적으로 GPT-3.5-Turbo와 GPT-4는 매우 뛰어난 성능을 보였으며, 특히 모델 간의 성능 비교를 통해 여러 공공 모델에서 Qwen1.5-7B, SOLAR-10.7B-Instruct-v1.0, Meta-Llama-3-8B 및 Zephyr-7B-Beta와 같은 모델이 주목할 만한 성과를 보여주었습니다. 이는 이들이 대형 모델에 대한 경쟁력 있는 대안으로 나올 수 있음을 시사합니다.



### Self-supervised Quantized Representation for Seamlessly Integrating Knowledge Graphs with Large Language Models (https://arxiv.org/abs/2501.18119)
- **What's New**: 이번 연구에서는 Knowledge Graph (KG) 구조와 대규모 언어 모델 (Large Language Models, LLMs) 간의 통합을 목표로 한 두 단계의 프레임워크를 제안합니다. 첫 번째 단계에서는 Self-Supervised Quantized Representation (SSQR) 방법을 통해 KG의 구조적 및 의미 지식을 압축하여 말뭉치 형식으로 변환된 이산 코드 (discrete codes)를 학습합니다. 이 방법을 통해 우리는 KG와 LLM의 원활한 통합이 가능해짐을 보여줍니다.

- **Technical Details**: 이 연구에서는 그래프 컨볼루션 네트워크 (GCN)를 활용하여 KG의 이웃 구조를 모델링하고, 벡터 양자화 (vector quantization) 기법을 통해 KG 양자화 표현 학습을 수행합니다. 각 엔티티에 대해 학습된 코드 (codes)는 KG 과제에 적합한 지침 데이터를 구축하여 LLMs에 직접 입력할 수 있도록 합니다. 이러한 방식은 LLM의 토크나이저 (tokenizer) 사전의 확장을 통해 실현됩니다.

- **Performance Highlights**: 실험 결과, SSQR 방법은 기존의 비지도 양자화 방법들보다 우수한 성능을 보이며, 더 구별되는 코드를 생성합니다. 또한, 조정된 LLaMA2 및 LLaMA3.1 모델은 KG 링크 예측 및 triple classification 작업에서 뛰어난 성능을 발휘하며, 기존의 수천 개의 토큰 대신 각 엔티티당 단 16개의 토큰만으로 이를 달성하게 됩니다.



### Diverse Preference Optimization (https://arxiv.org/abs/2501.18101)
- **What's New**: 이 연구에서는 표준 파이프라인보다 훨씬 더 다양한 응답을 생성하며 생성의 품질을 유지하는 온라인 최적화 방법인 Diverse Preference Optimization (DivPO)를 소개합니다. 기존의 언어 모델이 단일한 출력에 수렴하는 문제를 깨고 응답의 다양성을 높이는 데 집중하고 있습니다. DivPO는 모델이 높은 품질을 보장하되 다양한 응답을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: DivPO는 기존의 선호 최적화 방법의 한계를 극복하기 위해 두 가지 목표를 설정합니다. 첫째, 높은 보상을 받은 응답이 낮은 보상을 받은 응답보다 더 가능성이 높아야 합니다. 둘째, 모든 높은 보상을 받은 응답이 언어 모델 분포에서 유사한 확률을 가져야 합니다. 이 방법은 보상 기준을 사용하여 응답 쌍을 선택하며, 각 응답의 다양성을 비교하여 선정합니다.

- **Performance Highlights**: 실험 결과 DivPO는 표준 방법에 비해 45.6% 더 다양한 개인 특성을 생성하고, 스토리 다양성에서 74.6% 증가했습니다. 이는 생성된 출력 간의 품질을 유지하면서도 다양성을 높일 수 있음을 보여줍니다. 고정된 품질 목표에 대해 DivPO는 기본 방법들보다 높은 다양성 지표를 기록했습니다.



### Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation (https://arxiv.org/abs/2501.18100)
- **What's New**: 이번 연구는 유해한 파인튜닝 공격(harmful fine-tuning attack)의 보안 위험을 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 방어 기법들이 모델을 효과적으로 보호하지 못하는 문제를 발견하였습니다; 작은 양의 유해 데이터가 모델의 안전 정렬(safety alignment)을 저하할 수 있다는 것입니다. 이를 해소하기 위해, 단순히 랜덤한 섭동(random perturbations)을 추가하여 모델의 유해한 행동을 회복할 수 있음을 입증하고, 이로 인해 발생하는 성능 저하 문제를 해결하기 위한新的 방안인 Panacea를 제안합니다.

- **Technical Details**: Panacea는 파인튜닝 후 모델에 적용할 적응형 섭동(adaptive perturbation)을 최적화하는 방법으로, 최대 손실(maximal loss)을 증가시키는 방식으로 유해한 행동을 회복할 수 있도록 합니다. 이를 위해 구조화된 최적화 문제를 구성하고, 다양한 유해 비율(harmful ratios)에 대해 실험을 진행했습니다. 그 결과, Panacea는 평균 유해 점수를 21.5%까지 감소시키면서도, 파인튜닝 성능은 오히려 0.3% 향상되었습니다.

- **Performance Highlights**: 실험 결과에 따르면 Panacea는 다양한 유해 비율 및 파인튜닝 작업에 대해 유의미한 성과를 보여주었습니다. 특히, 적응형 섭동을 통해 최대 23.2%의 유해 점수를 감소시키면서도 경쟁력 있는 파인튜닝 성능을 유지할 수 있었습니다. 또한, 시각화 실험은 다양한 LLM의 각 레이어(layer)마다 고유한 안전 계수(safety coefficients)가 존재함을 발견하였습니다.



### InnerThoughts: Disentangling Representations and Predictions in Large Language Models (https://arxiv.org/abs/2501.17994)
Comments:
          Accepted at AISTATS 2025

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 성능 향상을 위해 각각의 transformer layer에서 생성된 히든 상태를 활용하여, 마지막 토큰 위치에서 답변 레이블을 예측하는 작은 신경망 예측 모듈을 학습하는 방법을 제안합니다. 이를 통해 LLM의 표현 능력과 예측 능력을 분리하는 구조를 갖습니다. 기존의 방식보다 훨씬 낮은 계산 비용으로, 여러 어렵고 까다로운 벤치마크에서 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 'InnerThoughts'라고 불리며, 각 layer의 히든 상태에서 최종 토큰 위치의 정보를 이용하여 예측을 수행합니다. 이 과정에서 기존의 LLM 파라미터는 고정된 채로 유지되며, 훈련 과정은 오직 단일 전방(pass) 계산으로 진행됩니다. 이를 통해 훈련 비용을 최소화하고, 고정된 파라미터로 범용적인 표현을 유지하면서도 특정 작업에 대한 성능을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 기존의 보정(calibration) 방법들과 비교하여 상당한 성능 향상을 보이며, 특정 벤치마크에서는 파라미터 효율적인 미세 조정 방법인 QLoRA (Dettmers et al., 2023)와 비슷한 수준의 성능에 도달합니다. 또한, 가장 큰 성능 향상을 보이는 벤치마크는 LLM의 답변 신뢰도가 낮은 질문들인 것을 확인하였습니다. 최종적으로, 특정 레이어에서의 기여도를 분석하여 우리의 접근 방식을 정당화하는 결과를 도출하였습니다.



### Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion (https://arxiv.org/abs/2501.17887)
Comments:
          Accepted to AAAI 25: Workshop on Open-Source AI for Mainstream Use

- **What's New**: Docling은 MIT 라이센스 하에 제공되는 오픈소스 문서 변환 툴킷으로, 다양한 문서 포맷을 통합된 구조로 변환할 수 있는 기능을 갖추고 있습니다. 이 툴킷은 최신의 AI 모델인 DocLayNet과 TableFormer를 활용하여 레이아웃 분석과 표 구조 인식을 지원하며, 일반 하드웨어에서도 효율적으로 실행됩니다. Docling은 Python 패키지로 제공되어 API 및 CLI 도구로 사용할 수 있으며, 모듈화된 아키텍처 덕분에 새로운 기능 및 모델의 구현이 용이합니다.

- **Technical Details**: Docling은 파이프라인, 파서 백엔드, DoclingDocument 데이터 모델로 구성된 모듈형 아키텍처를 가지고 있습니다. DoclingDocument 모델은 텍스트, 표, 이미지, 캡션, 목록 등 다양한 문서 기능을 표현하며, 문서 구조와 위치 정보, 출처 정보를 포함합니다. 사용자는 이 데이터 모델을 기반으로 문서를 생성하고 검사하며, JSON 및 HTML과 같은 일반 형식으로 내보낼 수 있는 API에 접근할 수 있습니다.

- **Performance Highlights**: 출시된 이후, Docling은 AI 개발자 커뮤니티에서 큰 관심을 모았고, GitHub의 월간 트렌딩 레포지토리에서 10,000개 이상의 별을 기록하며 1위에 올랐습니다. Docling v2는 새로운 기능 및 개념을 추가하며 성능과 효율성을 더욱 개선했습니다. 이 툴킷은 문서 변환을 위한 신뢰할 수 있는 솔루션으로 자리잡고 있으며, LangChain, LlamaIndex와 같은 주요 AI 개발 프레임워크와의 통합이 용이합니다.



### Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models (https://arxiv.org/abs/2501.18533)
- **What's New**: 본 연구는 Multi-Image Safety (MIS) 데이터셋을 제안하여 안전 관련 비주얼 추론(safety visual reasoning) 및 시각적 인식을 개선하고자 합니다. 대규모 Vision-Language Models (VLMs)의 안전성 문제를 해결하기 위한 새로운 접근 방법을 제공하며, 기존의 안전 조정 방법들의 한계를 강조합니다. MIS 데이터셋은 다중 이미지와 안전 Chain-of-Thought (CoT) 레이블을 통합하여 모델의 성능을 향상시킵니다.

- **Technical Details**: MIS 데이터셋은 훈련 세트와 테스트 세트로 나누어져 있으며, LLMs, VLMs, Text-to-Image 모델을 이용한 자동 데이터 생성 프레임워크를 사용합니다. 특히, MIS는 두 개의 이미지를 결합하여 생기는 위험한 의도를 해석하고 해결해야 하는 과제를 모델에게 부여합니다. 이 과정에서 모델은 효과적인 시각 인식 및 사고 과정을 요구받고, 특히 안전성을 고려한 응답 생성을 목표로 합니다.

- **Performance Highlights**: MIS 데이터셋을 사용하는 InternVL2.5-8B의 미세 조정(fine-tuning)을 통해 강력한 오픈소스 모델 및 API 기반 모델에 비해 우수한 성능을 보여주었습니다. 평균 정확도는 5개의 일반 벤치마크에서 0.83% 증가했으며, 여러 안전 벤치마크에서 Attack Success Rate (ASR)를 크게 감소시켰습니다. 이러한 결과는 MIS 접근 방식이 안전성 및 일반적 능력을 모두 향상시킬 수 있음을 보여줍니다.



### WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training (https://arxiv.org/abs/2501.18511)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델)의 효과적인 후속 학습 방법론을 위한 새로운 데이터셋인 WildChat-50M을 소개합니다. 이 데이터셋은 50개 이상의 다양한 오픈 웨이트 모델로부터 생성된 대화 전사 파일을 포함하고 있으며, 총 1억 2500만 개 이상의 대화 전사를 포함합니다. 연구자들은 이 데이터셋을 바탕으로 RE-WILD라는 새로운 감독된 미세 조정(SFT) 믹스를 선보이며, 이는 Allen AI의 Tulu-3 SFT 믹스를 초과하는 성과를 거두었습니다.

- **Technical Details**: WildChat-50M 데이터셋은 약 두 달에 걸쳐 12x8 H100 공유 연구 클러스터에서 수집되었습니다. 수집 과정에서 사용된 모델들은 VLLM 프레임워크를 통해 다양한 GPU에서 최적화되어 있으며, 현재까지 54개의 체크포인트가 포함되어 있습니다. 이 데이터셋은 0.5B에서 104B 파라미터를 가진 다양한 모델들로 구성되어 있으며, 서로 다른 컨텍스트 윈도우 크기를 설정하여 성능을 극대화하였습니다.

- **Performance Highlights**: RE-WILD 믹스를 사용하여 Llama-3.1 8B Base 모델을 미세 조정한 결과, 기존의 SFT 믹스보다 뛰어난 성능을 보여주었습니다. 이 연구는 비교한 모델들 간의 VRAM 효율성 및 런타임 효율성에 대한 포괄적인 분석을 제공하며, 향후 데이터 수집 및 후속 학습 방법을 더욱 발전시킬 수 있는 기초 자료를 제공합니다.



### MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding (https://arxiv.org/abs/2501.18362)
- **What's New**: 이번 논문에서는 MedXpertQA라는 새로운 벤치마크를 소개합니다. MedXpertQA는 전문가 수준의 의료 지식과 고급 추론을 평가하기 위해 설계된 도전적이고 포괄적인 기준을 제시합니다. 이 벤치마크는 17개 전문 분야와 11개 신체 시스템을 포함하며, 텍스트 평가를 위한 Text와 다중 모드 평가를 위한 MM의 두 가지 하위 집합으로 구성되어 있습니다.

- **Technical Details**: MM 하위 집합은 환자 기록 및 검사 결과를 포함한 다양한 이미지와 풍부한 임상 정보를 가진 전문가 수준의 시험 문제를 도입하여, 간단한 QA 쌍을 생성하는 기존의 의료 다중 모드 벤치마크와 차별화됩니다. MedXpertQA는 기존의 벤치마크인 MedQA와 같은 어려움이 부족한 문제를 해결하기 위해 엄격한 필터링과 증강을 적용하고, 임상 관련성과 포괄성을 높이기 위해 전문 분야 위원회 질문을 포함합니다.

- **Performance Highlights**: 우리는 MedXpertQA에서 16개의 주요 모델을 평가하였습니다. 의료는 실질적인 의사 결정과 깊은 연관이 있기 때문에, 수학 및 코드 외의 추론 능력을 평가하는 데 적합한 풍부하고 대표적인 환경을 제공합니다. 이를 위해 o1과 유사한 모델의 평가를 용이하게 하는 추론 지향 하위 집합을 개발했습니다.



### State Stream Transformer (SST) : Emergent Metacognitive Behaviours Through Latent State Persistenc (https://arxiv.org/abs/2501.18356)
Comments:
          25 pages, 3 figures

- **What's New**: 이번 논문에서는 State Stream Transformer(SST)라는 새로운 LLM 아키텍처를 소개합니다. 이 모델은 전통적인 transformer 모델의 한계를 극복하고, 자가 회귀적 생성 과정에서의 계산적 연속성을 유지함으로써, 학습된 가중치에서 잠재적인 추론 행동을 드러냅니다. SST는 슬라이딩 윈도우 방식의 잠재 상태(FFN) 캐시를 도입해 지속적인 잠재 과정의 유지 및 진화를 가능하게 합니다.

- **Technical Details**: SST는 모든 선형층에서 가중치를 감소시키면서 슬라이딩 윈도우 잠재 상태 캐시를 구현합니다. 이 구조는 토큰 생성 중에 지속적인 '상태 흐름'을 유지하여 모델이 정보를 처리하는 방식에 근본적인 변화를 가져옵니다. 이를 통해 SST는 자가 회귀 모델에 비해 향상된 추론 능력을 보여주며, 이는 메타인지 행동을 통해 증명됩니다.

- **Performance Highlights**: 정량적 평가 결과, SST는 GSM-8K(0-shot)에서 89.01%의 정확도와 ARC Challenge(0-shot CoT)에서 91.04%의 정확도로 기본 모델에 비해 상당한 성과 향상을 이뤘습니다. 이러한 결과는 잠재 상태에서의 지속적 계산이 정보 처리 및 내부 추론 전략에 근본적인 차이를 만든다는 것을 시사합니다. SST 아키텍처는 인공지능 시스템의 능력 및 인공지능 인지에 대한 이해에 중요한 함의를 가집니다.



### A Video-grounded Dialogue Dataset and Metric for Event-driven Activities (https://arxiv.org/abs/2501.18324)
Comments:
          Accepted at AAAI2025

- **What's New**: 이번 논문에서는 사건 주도 활동에 대한 비디오 기반 대화 데이터셋인 VDAct를 소개합니다. VDAct는 3,000개의 대화와 30,000개 이상의 질문-답변 쌍으로 구성되어 있으며, 다양하고 복잡한 비디오 시퀀스를 포함합니다. 게다가 VDAct는 새로운 평가 metric인 VDEval을 제안하여 단일 대화 턴의 맥락에만 의존하지 않고, 대화 세션의 전반적인 맥락을 평가합니다.

- **Technical Details**: VDAct는 다양한 사건-주도 활동이 포함된 긴 비디오 시퀀스를 활용하며, 이에 대한 질문은 설명적 질문, 시간적 질문, 설명적 질문, 정량적 질문 등 여러 범주로 나뉩니다. 데이터셋은 주제와 행동 간의 관계를 연결하는 구조적 정보를 제공하는 지식 그래프(Knowledge Graphs, KGs)로 보강됩니다. 한편, VDEval은 대화 기록과 KGs를 통합해 생성된 응답의 평가를 수행하는 새로운 metric입니다.

- **Performance Highlights**: 기존의 영상 기반 대화 시스템과 비교할 때, VDAct에서 상태 기반 비전 모델들은 복잡한 질문 유형에 대응하는 데 한계를 보였습니다. 새로운 평가 metric인 VDEval은 인간 평가와 높은 상관관계를 보여 VDAct 데이터셋의 효과적인 평가를 가능하게 합니다. 따라서 이 연구는 비디오 기반 대화 시스템의 발전에 기여할 수 있는 중요한 기초 자료로 자리잡을 것입니다.



### Citation Recommendation based on Argumentative Zoning of User Queries (https://arxiv.org/abs/2501.18292)
- **What's New**: 이번 논문에서는 인용 추천(Citation recommendation) 작업을 개선하기 위해 논문에서 주장하는 구조인 argumentative zoning을 활용하는 새로운 접근 방식을 제안합니다. 연구자들이 인용할 필요가 있는 중요한 논문을 찾는 것을 목표로 하며, 각 인용 문장에서의 인용 의도를 파악하고자 합니다. 또한 새로운 argumentative zoning 스키마를 기반으로 PubMed Central에서 주석이 달린 데이터셋을 생성했습니다.

- **Technical Details**: 다중 작업 학습 모델(Multi-task Learning Model)을 구축하여 인용 추천과 argumentative zoning 분류를 동시에 수행합니다. 실험 결과, 인용 문장의 주장 정보를 고려함으로써 인용 추천 모델의 성능이 개선됨을 보여줍니다. 이러한 접근 방식은 인용 분석(Citation Analysis) 분야에서 혼합된 여러 기능을 동시에 처리할 수 있는 가능성을 열어줍니다.

- **Performance Highlights**: 실험적으로, 주장 정보를 포함한 인용 추천 시스템은 더 높은 성능을 기록했습니다. 이는 각기 다른 인용 의도를 분류함으로써 다양한 과학적 문헌에서의 인용을 보다 효과적으로 지원할 수 있음을 의미합니다. 따라서 이 연구는 인용 추천 시스템의 품질을 향상시키는데 중요한 기여를 할 것으로 예상됩니다.



### Collecting Cost-Effective, High-Quality Truthfulness Assessments with LLM Summarized Evidenc (https://arxiv.org/abs/2501.18265)
Comments:
          18 pages; 7 figures; 5 tables

- **What's New**: 이번 연구에서는 온라인 정보의 신뢰성을 평가하는 새로운 접근 방식을 제안합니다. 특히, LLM(대규모 언어 모델)에서 생성된 요약 정보를 바탕으로 한 군중 소싱(crowdsourcing) 방법을 활용하여 효과적인 진위 판별을 시도했습니다. 이를 통해 작업 효율을 높이고, 전문적인 팩트 체크에 대한 의존도를 줄일 수 있는 가능성을 모색하고 있습니다. 연구 결과는 신뢰성 평가의 정확도는 유지하면서도 속도를 크게 증가시켰음을 보여줍니다.

- **Technical Details**: 연구는 두 가지 접근 방식, 즉 웹 페이지의 전체를 평가하는 Standard modality와 LLM이 생성한 요약 증거를 평가하는 Summary modality를 비교합니다. 이 과정에서 A/B 테스트 환경을 설정하고, 다양한 집단의 작업자들이 진위 평가를 수행하게 하여 평가의 질과 효율성을 분석합니다. LLM이 생성한 요약 정보는 필수 정보를 간결히 전달하면서도 사실적 정확성을 유지하려는 목표를 갖고 있습니다.

- **Performance Highlights**: 연구 결과, Summary modality를 통해 변화된 평가 방식이 전체 진위 평가의 정확도에 큰 영향을 미치지 않으면서도 작업자들이 같은 시간 안에 더 많은 평가를 수행할 수 있게 되어 비용 절감 효과를 가져왔습니다. 또한, 요약 정보를 사용했을 때 서로 다른 평가자 간의 동의도가 극대화되며, 증거에 대한 신뢰와 유용성 인식 또한 향상되었습니다. 이는 요약 증거가 평가의 질을 희생하지 않으면서도 유용한 정보를 제공할 수 있음을 나타냅니다.



### Statistical multi-metric evaluation and visualization of LLM system predictive performanc (https://arxiv.org/abs/2501.18243)
- **What's New**: 이번 논문은 Generative 또는 Discriminative Large Language Model (LLM) 기반 시스템의 성능 평가 방법에 대한 새로운 프레임워크를 제시합니다. 이 프레임워크는 여러 데이터셋과 평가 기준을 아우르며, 통계적 검증(testing)을 통해 성과의 유의미성(significance)을 확인하는 과정을 자동으로 수행합니다. 여러 하이퍼파라미터를 가진 시스템 구성의 성능을 비교하고 분석함으로써, 사용자가 보다 효율적으로 LLM 성능을 평가할 수 있게 합니다.

- **Technical Details**: 프레임워크는 여러 데이터셋과 평가 기준을 통합하여 시스템 간의 통계적 비교를 가능하게 합니다. 특히, 이는 쌍 또는 비극형 관측 데이터셋에 대한 통계적 결정을 지원하며, 각 시스템의 통계적 비교 결과를 시각적으로 표현할 수 있는 유틸리티를 포함하고 있습니다. 사용자는 복잡한 통계적 과정을 걱정할 필요 없이, 필요한 데이터를 입력하기만 하면 됩니다.

- **Performance Highlights**: 이 연구는 CrossCodeEval이라는 다국어 코드 보완 벤치마크에서 여러 최신 LLM의 성능을 비교하는 데 성공적으로 적용되었습니다. 기존의 Leaderboard 평가 방식에서는 종종 유의미한 차이를 고려하지 않았으나, 본 프레임워크는 통계적 검정을 통해 그러한 차이를 명확히 할 수 있는 장점을 제공합니다. 또한, 데이터셋 간의 통계적 결과 집계 및 시각화를 통해 최적의 LLM 선택을 지원합니다.



### Scaling Inference-Efficient Language Models (https://arxiv.org/abs/2501.18107)
Comments:
          17 pages, 16 figures

- **What's New**: 이 연구는 대형 언어 모델의 성능 예측에 있어 scaling laws가 인퍼런스 비용을 충분히 반영하지 못한다는 점을 지적합니다. 또한, 동일한 크기의 모델인데도 아키텍처에 따라 인퍼런스 지연(latency)이 3.5배 차이가 날 수 있음을 보여주며, 인퍼런스 효율성을 고려한 새로운 scaling laws를 제안합니다. 이와 함께 Morph-1B 모델을 발표하며, 기존 오픈 소스 모델 대비 1.8배 더 빠른 인퍼런스 지연을 달성했음을 강조합니다.

- **Technical Details**: 기존의 scaling laws는 모델 크기(모델 파라미터 수)와 훈련 토큰 수의 균형을 중시합니다. 본 연구에서 제안하는 새로운 scaling laws는 모델 아키텍처도 반영하며, 인퍼런스 효율성을 최적화합니다. 다양한 모델 파라미터와 재훈련 과정을 통해, 63개의 실험 모델을 개발하고 이는 인퍼런스 손실 예측력에서 Chinchilla scaling law에 비해 더 우수함을 입증합니다.

- **Performance Highlights**: Morph-1B 모델은 본 연구의 인퍼런스 효율성 기반 scaling laws 및 모델 선택 방법을 통해 개발되었으며, 같은 크기의 다른 오픈 소스 모델에 비해 1.8배 빠른 인퍼런스 지연을 보입니다. 이는 모델의 정확도(downstream task 성능)를 유지하면서도 인퍼런스 효율성을 극대화한 결과입니다. 연구 결과는 향후 아키텍처 최적화에 있어 효율성과 정확도 간의 균형을 잘 잡을 수 있는 기준이 될 것입니다.



### Beyond Turn-taking: Introducing Text-based Overlap into Human-LLM Interactions (https://arxiv.org/abs/2501.18103)
Comments:
          16 pages, 9 figures

- **What's New**: 이 연구에서는 전통적인 텍스트 기반 인간-AI 상호 작용의 엄격한 턴 테이킹(turn-taking) 대신 겹치는 메시지를 포함하는 새로운 접근 방식을 제안합니다. 이를 통해 사용자가 자연스럽게 겹치며 대화하는 모습을 관찰하였고, OverlapBot이라는 시제품 챗봇을 개발하여 AI와 사용자 모두 겹치는 방식으로 대화할 수 있도록 하였습니다. 사용자 연구 결과, OverlapBot은 전통적인 턴 테이킹 챗봇보다 더 소통이 활발하고 몰입감이 있으며, 이러한 새로운 접근 방식이 텍스트 기반 상호 작용의 유연성과 참여도를 향상시킬 것이라는 점을 보여줍니다.

- **Technical Details**: 겹치는 상호 작용을 지원하기 위해, OverlapBot은 사용자의 입력과 동시에 응답을 제공하고, 가끔은 자신의 입력을 삭제하는 방식으로 작동합니다. 우리는 이 챗봇을 주제 자유 대화에 참여한 18명을 대상으로 한 사용자 연구를 통해 평가하였고, 사용자가 OverlapBot과의 상호 작용을 통해 더 자연스럽고 효율적인 대화를 경험한다고 보고하였습니다. 연구 참여자들은 챗봇의 사전 응답을 통해 챗봇의 주의 깊음을 확인할 수 있었고, 이는 인간 간의 대화에서 보이는 능동적인 경청 신호와 유사한 결과로 나타났습니다.

- **Performance Highlights**: OverlapBot은 전통적인 턴 테이킹 챗봇과 비교하여 더 빠르고 자연스러운 상호 작용을 가능하게 했습니다. 사용자는 AI의 겹치는 상호 작용에 대해 긍정적으로 반응하며, 이전의 기존 연구 결과를 통해 겹치는 상호 작용이 어떻게 인간-AI 대화의 자연스러움을 증진할 수 있는지를 확인했습니다. 이러한 발견은 겹치는 상호 작용을 설계하는데 있어 중요한 통찰력을 제공하며, AI 시스템에 적용 가능한 겹침 기능을 구현하기 위한 몇 가지 권장 사항을 포함하고 있습니다.



### Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judg (https://arxiv.org/abs/2501.18099)
- **What's New**: 이번 연구에서 제안하는 EvalPlanner는 LLM-as-a-Judge 모델의 평가 계획과 실행을 최적화하는 방법론이다. 난잡한 평가 기준 없이 LLM이 이 평가를 수행할 수 있도록 하여, 더 많은 테스트 시간 동안 코어 사고(Chain-of-Thought, CoT)를 생성한다. 이를 통해 LLM의 최종 판단이 더욱 신뢰성 있고 투명하게 이루어질 수 있도록 한다.

- **Technical Details**: EvalPlanner는 입력 지침에 따라 평가 계획을 작성하고, 이를 단계적으로 실행하여 최종 판단을 도출하는 방식으로 작동한다. 계획 수립 과정에서는 응답 평가에 필요한 모든 단계를 포함하는 세부적인 평가 계획이 생성된다. 이후 실행 단계에서 모델은 생성된 계획을 따르며 입력 응답에 대한 분석 과정을 통해 최종 판단을 수행한다.

- **Performance Highlights**: EvalPlanner는 RewardBench에서 generative reward models의 새로운 최첨단 성능 점수인 93.9를 달성하며, 기존의 많은 데이터로 훈련된 모델들을 능가했다. 또한, 다른 벤치마크에서도 최대 13% 향상된 성능을 나타내며, 개별 평가 기준을 모델이 점진적으로 최적화하도록 학습하면서 평가의 정확성을 개선하고 있다.



### LLMs can see and hear without any training (https://arxiv.org/abs/2501.18096)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 MILS(Multimodal Iterative LLM Solver)를 소개합니다. 이 방법은 훈련이 필요 없는 간단한 접근 방식으로, 기존의 LLM에 멀티모달 기능을 추가할 수 있도록 돕습니다. MILS는 다단계 추론(multi-step reasoning) 능력을 활용해 다양한 응용 프로그램을 가능하게 합니다.

- **Technical Details**: MILS는 후보 출력을 생성하고 각 출력을 점수화(scoring)한 후 피드백(feedback)을 통해 반복(iteratively)적으로 솔루션을 도출하는 방식입니다. 이를 통해, 훈련이 필요했던 전문 모델을 사용하지 않고 특정 작업에 대한 솔루션을 생성할 수 있습니다. 이 방법은 이미지를 캡션(captioning)하는 데 있어 새롭고 최고의 성과를 달성했습니다.

- **Performance Highlights**: MILS는 이미지 생성(text-to-image generation)과 같은 미디어 생성(media generation)에도 효과적으로 적용할 수 있으며, 스타일 전이(style transfer)를 위한 프롬프트 수정(prompt rewrites) 발견에도 기여합니다. 또한, 이 방법은 그래디언트(gradient)가 필요 없는 최적화 방식이기 때문에 멀티모달 임베딩(multimodal embeddings)을 텍스트로 변환하여 크로스모달 산술(cross-modal arithmetic) 응용 프로그램에도 사용할 수 있습니다.



### FinanceQA: A Benchmark for Evaluating Financial Analysis Capabilities of Large Language Models (https://arxiv.org/abs/2501.18062)
Comments:
          10 pages, 7 figures

- **What's New**: FinanceQA는 복잡한 수치 금융 분석 과제를 평가하는 테스트 스위트입니다. 이 시스템은 실제 투자 작업을 반영하며, 최근 LLM의 발전에도 불구하고 현재 모델이 금융 기관의 정확성 요구를 충족하지 못하는 문제를 다룹니다.

- **Technical Details**: 현재 LLM 모델은 헤지펀드, 사모펀드, 투자은행 등에서 실제로 수행되는 업무 분석을 모방한 과제에서 약 60%의 실패율을 보이고 있습니다. 주요 도전 과제는 핸드-스프레딩 메트릭스, 표준 회계 및 기업 가치 평가 관례 준수, 그리고 정보가 불완전한 상황에서의 분석 수행 등입니다.

- **Performance Highlights**: 이러한 성능 차이는 기존 LLM의 기능과 직업 금융 분석의 요구 사이의 괴리를 강조합니다. 더 높은 품질의 훈련 데이터가 필요한 것을 보여주며, OpenAI의 파인튜닝 API를 통해 실험이 수행됩니다.



### From tools to thieves: Measuring and understanding public perceptions of AI through crowdsourced metaphors (https://arxiv.org/abs/2501.18045)
- **What's New**: 본 연구는 미국 내 인공지능(AI)에 대한 공공 인식의 변화를 조사하기 위해 12개월 간 12,000개의 응답을 수집하였습니다. 전통적인 자기보고 조사 방법의 한계를 극복하기 위해 응답자가 AI를 묘사한 은유를 분석했습니다. 이 연구는 AI에 대한 인식의 주요 차원인 인간형성(anthropomorphism), 따뜻함(warmth), 그리고 능력(competence)을 측정하기 위한 확장 가능한 프레임워크를 제공합니다. 이를 통해 우리는 미국인들이 AI를 일반적으로 따뜻하고 유능하다고 바라보며, 이러한 인식이 AI에 대한 신뢰와 수용의도에 강하게 영향을 미친다는 사실을 발견했습니다.

- **Technical Details**: 본 연구에서는 질적 코딩과 정량적 클러스터링을 결합한 혼합 방법론(mixed-methods approach)을 사용하여, 20개의 주도적인 은유를 식별했습니다. 응답자들이 얼마나 AI를 인간으로 형상화했는지를 평가하기 위해 언어모델(language model) 기반의 확률을 이용한 방법론이 사용되었습니다. 또한, AI에 대한 따뜻함과 능력을 측정하기 위해 LM 기반의 임베딩을 사용하여 의미 축을 구축했습니다. 이를 통해 개인의 AI에 대한 경험을 예측하는 중요한 지표로서 인간형성과 따뜻함, 능력을 강조했습니다.

- **Performance Highlights**: 연구 결과, AI에 대한 참여자의 은유는 대체로 따뜻하고 유능했으나 인간형성의 정도는 다양하게 나타났습니다. 특히, 인간형성과 따뜻함은 지난 1년 동안 각각 34%와 41%의 유의미한 증가세를 보였습니다. 또한, '신(god)', '두뇌(brain)', '도둑(thief)'이라는 주도적인 은유가 AI에 대한 신뢰와 수용도를 예측하는 데 중요한 역할을 하며, 사회적 정체성이 AI와의 상호작용에 미치는 영향도 분석되었습니다. 이 연구는 다양한 인구 통계적 차이를 바탕으로 AI에 대한 인식의 변화의 맥락을 제공합니다.



### DReSS: Data-driven Regularized Structured Streamlining for Large Language Models (https://arxiv.org/abs/2501.17905)
- **What's New**: 이 논문에서는Llarge language models (LLMs)의 성능을 유지하면서 모델 크기를 줄일 수 있는 새로운 프루닝(paruning) 패러다임을 제안합니다. 기존의 프루닝 방법이 정보 손실을 초래하는 것과 달리, 새로운 접근 방식에서는 먼저 정규화(regularization)를 적용한 후 프루닝을 수행하고, 마지막으로 미세 조정(finetuning)을 진행합니다. 이 과정을 통해 DReSS(Data-driven Regularized Structured Streamlining)라는 효과적인 방법을 도입하게 되었습니다.

- **Technical Details**: DReSS는 파라미터 행렬에서 선택된 채널에 정규화 과정을 적용하여 중요 정보를 전이하여 제거되는 부분의 정보를 보존합니다. 이 방법은 프로세스가 다음 네 단계로 구성되어 있습니다: 데이터 선택, 정규화 적용, 채널 프루닝 후 RFT 수행입니다. 작은 데이터 세트를 활용함으로써 프루닝 과정에서 발생하는 오버헤드를 최소화하고, 높은 프루닝 비율에서도 성능 저하를 방지합니다.

- **Performance Highlights**: 실험 결과 DReSS는 기존 프루닝 방법보다 크게 향상된 정밀도와 정확도를 보였습니다. 이 방법은 프루닝 비율이 극단적인 상황에서도 성능이 크게 향상되어 상당한 지연 시간 감소와 처리량 증가를 이끌어냈습니다. 따라서 DReSS는 대형 언어 모델에서 정보 손실을 줄이고, 언어 모델링 능력을 향상시키는 데 기여했습니다.



### LLM-AutoDiff: Auto-Differentiate Any LLM Workflow (https://arxiv.org/abs/2501.16673)
- **What's New**: LLM-AutoDiff는 LLM 파이프라인을 최적화하기 위한 새로운 프레임워크로, 텍스트 기울기 기반 방법을 다중 구성 요소와 순환 아키텍처에 확장합니다. 이 시스템은 각 텍스트 입력을 학습 가능한 매개변수로 취급하며, 피드백 신호를 통해 지속적으로 프롬프트 업데이트를 안내합니다. LLM-AutoDiff는 기능 노드를 보존하고 오류를 격리하여 "중간 손실" 문제를 해결함으로써 복잡한 작업에서 효율성을 높입니다.

- **Technical Details**: LLM-AutoDiff는 LLM 애플리케이션을 데이터 흐름과 종속성을 캡처하는 방향 그래프 형태로 모델링합니다. 각 호출에 타임스탬프 기록이 붙여져 있으며, 이를 통해 반복 호출 시 피드백의 정확한 순서가 유지됩니다. 또한, 기능 노드에 대해 통과 기울기(pass-through gradients)를 도입하여, 학습 가능한 파라미터가 없는 중간 작업에서도 기울기가 효과적으로 흐를 수 있도록 합니다.

- **Performance Highlights**: 다양한 벤치마크 실험을 통해 LLM-AutoDiff는 표준 텍스트 기울기 기반 방법에 비해 정확도 및 훈련 비용에서 일관되게 우수한 성능을 보여줍니다. 단일 단계 분류부터 다중 단계 시나리오에 이르기까지, 시스템은 피드백 전달을 간소화하고 오류 신호를 통합하여 효율성을 크게 향상시킵니다. 최종적으로, LLM-AutoDiff는 자동화된 LLM 애플리케이션 최적화(ALAO)의 비전을 구체화하며, 전체 파이프라인을 자동으로 정제할 수 있는 가능성을 제공합니다.



### Context is Key for Agent Security (https://arxiv.org/abs/2501.17070)
- **What's New**: 이 논문은 시스템의 안전성을 보장하고, 다양한 맥락에 적응할 수 있는 새로운 보안 설계에 대한 필요성을 강조합니다. 최신 시스템은 수동으로 작성된 정책이나 사용자 확인에 의존하여 결정을 내리지만, 이러한 방식은 다양한 환경에서 안전성을 보장하기에 부족합니다. 이 연구는 Contextual Security for Agents 프레임워크인 Conseca를 소개하여, 필요한 순간에 적절한 보안 정책을 생성하는 방법을 제안합니다.

- **Technical Details**: Conseca 프레임워크는 사용자 요청과 신뢰할 수 있는 맥락을 기반으로 특정 정책을 생성하는 방식으로 설계되었습니다. 이 시스템은 두 가지 기능을 수행합니다: 요청에 대한 보안 정책을 생성하고, 에이전트의 제안된 행동이 정책을 충족하는지 피드백을 제공합니다. Conseca는 언어 모델을 활용하여 동적으로 정책을 생성하고, 생성된 정책에는 전문가가 감사할 수 있는 인간-가독성의 근거가 포함됩니다.

- **Performance Highlights**: Conseca는 정확하고 컨텍스트에 적합한 보안 정책을 생성하여 다양한 맥락에서의 행동을 안전하게 제어할 수 있도록 합니다. 이 시스템은 악의적인 조작으로부터 보호하기 위해 신뢰할 수 있는 맥락만을 사용하여 정책 생성을 격리합니다. 결론적으로, Conseca는 다목적 에이전트 시스템에 대한 보안을 확장하는 데 필요한 연구와 프로토타입을 제공함으로써 향후 보안 메커니즘 설계에 중요한 기여를 합니다.



New uploads on arXiv(cs.IR)

### Illusions of Relevance: Using Content Injection Attacks to Deceive Retrievers, Rerankers, and LLM Judges (https://arxiv.org/abs/2501.18536)
- **What's New**: 이 연구는 정보 검색(Information Retrieval, IR) 시스템의 취약점, 즉 콘텐츠 주입 공격(content injection attacks)에 대한 새로운 분석을 제시합니다. 연구자들은 임베딩 모델(embedding models), 재정렬기(re-rankers), 대형 언어 모델(large language models, LLMs) 관련 판단자들이 이런 공격에 얼마나 취약한지를 발견했습니다. 특히, 관련성 있는 텍스트에 악의적인 또는 관련 없는 콘텐츠를 삽입하는 공격과 쿼리 키워드를 삽입하여 비관련 문서를 관련 있는 것처럼 보이게 하는 공격을 두 가지 주요 위협으로 지적합니다.

- **Technical Details**: 내용 주입 공격의 성공률은 삽입 위치, 반복 쿼리 및 쿼리 용어, 관련 및 비관련 콘텐츠 간의 균형과 같은 여러 요인에 의해 영향을 받습니다. 본 논문에서는 주로 포인트 와이즈 재정렬기(pointwise rerankers)에 초점을 맞추었으며, 이들은 각 쿼리-패시지 쌍을 독립적으로 평가하여 계산 비용을 줄입니다. 또한 LLM 관련 판단자는 쿼리와 패시지의 관련성을 평가하고 0-3의 스코어를 제공함으로써 검색 결과의 품질을 보장합니다.

- **Performance Highlights**: 연구자들은 다양한 방어 전략도 탐구했으며, 대적 패시지 분류기(adversarial passage classifiers), 검색기 미세 조정(retriever fine-tuning), 더욱 신중하게 LLM 판단자에게 프롬프트를 제공함으로써 콘텐츠 주입 공격에 대한 저항력을 향상시킬 수 있음을 발견했습니다. 그러나 이러한 방어 조치는 종종 공격에 대한 강건성(robustness)과 효과성을 희생하면서 합법적인 문서에 페널티를 부여하는 등 상충 관계를 갖고 있다는 점이 중요합니다.



### Citation Recommendation based on Argumentative Zoning of User Queries (https://arxiv.org/abs/2501.18292)
- **What's New**: 이번 논문에서는 인용 추천(Citation recommendation) 작업을 개선하기 위해 논문에서 주장하는 구조인 argumentative zoning을 활용하는 새로운 접근 방식을 제안합니다. 연구자들이 인용할 필요가 있는 중요한 논문을 찾는 것을 목표로 하며, 각 인용 문장에서의 인용 의도를 파악하고자 합니다. 또한 새로운 argumentative zoning 스키마를 기반으로 PubMed Central에서 주석이 달린 데이터셋을 생성했습니다.

- **Technical Details**: 다중 작업 학습 모델(Multi-task Learning Model)을 구축하여 인용 추천과 argumentative zoning 분류를 동시에 수행합니다. 실험 결과, 인용 문장의 주장 정보를 고려함으로써 인용 추천 모델의 성능이 개선됨을 보여줍니다. 이러한 접근 방식은 인용 분석(Citation Analysis) 분야에서 혼합된 여러 기능을 동시에 처리할 수 있는 가능성을 열어줍니다.

- **Performance Highlights**: 실험적으로, 주장 정보를 포함한 인용 추천 시스템은 더 높은 성능을 기록했습니다. 이는 각기 다른 인용 의도를 분류함으로써 다양한 과학적 문헌에서의 인용을 보다 효과적으로 지원할 수 있음을 의미합니다. 따라서 이 연구는 인용 추천 시스템의 품질을 향상시키는데 중요한 기여를 할 것으로 예상됩니다.



### Collecting Cost-Effective, High-Quality Truthfulness Assessments with LLM Summarized Evidenc (https://arxiv.org/abs/2501.18265)
Comments:
          18 pages; 7 figures; 5 tables

- **What's New**: 이번 연구에서는 온라인 정보의 신뢰성을 평가하는 새로운 접근 방식을 제안합니다. 특히, LLM(대규모 언어 모델)에서 생성된 요약 정보를 바탕으로 한 군중 소싱(crowdsourcing) 방법을 활용하여 효과적인 진위 판별을 시도했습니다. 이를 통해 작업 효율을 높이고, 전문적인 팩트 체크에 대한 의존도를 줄일 수 있는 가능성을 모색하고 있습니다. 연구 결과는 신뢰성 평가의 정확도는 유지하면서도 속도를 크게 증가시켰음을 보여줍니다.

- **Technical Details**: 연구는 두 가지 접근 방식, 즉 웹 페이지의 전체를 평가하는 Standard modality와 LLM이 생성한 요약 증거를 평가하는 Summary modality를 비교합니다. 이 과정에서 A/B 테스트 환경을 설정하고, 다양한 집단의 작업자들이 진위 평가를 수행하게 하여 평가의 질과 효율성을 분석합니다. LLM이 생성한 요약 정보는 필수 정보를 간결히 전달하면서도 사실적 정확성을 유지하려는 목표를 갖고 있습니다.

- **Performance Highlights**: 연구 결과, Summary modality를 통해 변화된 평가 방식이 전체 진위 평가의 정확도에 큰 영향을 미치지 않으면서도 작업자들이 같은 시간 안에 더 많은 평가를 수행할 수 있게 되어 비용 절감 효과를 가져왔습니다. 또한, 요약 정보를 사용했을 때 서로 다른 평가자 간의 동의도가 극대화되며, 증거에 대한 신뢰와 유용성 인식 또한 향상되었습니다. 이는 요약 증거가 평가의 질을 희생하지 않으면서도 유용한 정보를 제공할 수 있음을 나타냅니다.



### Behavior Modeling Space Reconstruction for E-Commerce Search (https://arxiv.org/abs/2501.18216)
- **What's New**: 이번 논문에서는 기존의 검색 시스템을 새로운 프레임워크인 DRP를 통해 재examining하여 사용자 행동을 효과적으로 모델링할 수 있는 방법을 제시합니다. DRP는 사용자 선호(preference)와 관련성(relevance) 사이의 효과를 분리하여 두 가지 구성 요소를 통해 검색의 정확성을 향상시킵니다. 이를 통해 사용자 선호를 왜곡하지 않고 더 깨끗한 예측을 가능하게 합니다.

- **Technical Details**: DRP는 사용자 행동을 모델링하는 과정에서 인과 그래프(causal graphs)와 벤 다이어그램(Venn diagrams)을 활용해 행동 모델링 공간을 재구성하고, 적응형 융합(adaptive fusion) 방식을 통해 다양한 관련성과 선호 패턴에 맞춰 동적으로 조정합니다. 이 모델은 기존의 사용자 행동 신호에 기반한 학습 방식 대신, 인간 레이블 데이터에 의존하지 않고도 동작할 수 있는 점이 특징입니다. 또한, 사용자 데이터는 ‘Activity Degree’와 ‘Clicked Items’로 집계되어 사용됩니다.

- **Performance Highlights**: DRP는 두 개의 공개 데이터셋과 하나의 비공식 검색 데이터셋을 통해 실험적으로 검증되었으며, 기존 방법들보다 월등한 성능 향상을 보여주었습니다. 구체적으로, 사용자 행동의 미세한 차이를 포착할 수 있어, 보다 유연하고 맞춤형의 예측이 가능하다는 점을 입증하였습니다. 따라서 DRP는 e-commerce 검색 최적화에 있어 중요한 기여를 할 것으로 기대됩니다.



### Investigating Tax Evasion Emergence Using Dual Large Language Model and Deep Reinforcement Learning Powered Agent-based Simulation (https://arxiv.org/abs/2501.18177)
- **What's New**: 이 연구는 비공식 경제에서 세금 탈세의 동태 및 비공식 경제 활동의 출현을 새로운 계산적 프레임워크로 분석하는 내용을 다룹니다. 이를 위해, 대규모 언어 모델(LLM)과 심층 강화 학습(Deep Reinforcement Learning)을 기반으로 한 에이전트 기반 시뮬레이션을 사용하여, 비공식 경제 행동이 자율적으로 나타날 수 있도록 설계되었습니다. 이 프레임워크는 세금 탈세의 동태를 탐구하는 데 유용하며 기존 모델과는 달리 비공식 경제의 존재를 전제로 하지 않습니다.

- **Technical Details**: 연구는 AI와 최신 컴퓨터 시뮬레이션 기술을 활용하여 비공식 경제의 동태를 조사합니다. agent-based simulation(ABS)을 통해 다양한 개인 특성을 가진 에이전트들이 상호작용하는 경제 환경을 모델링하며, 여기서 개별 에이전트는 LLM과 DRL 기반으로 운영됩니다. 이 연구는 비공식 경제의 출현, 특히 세금 탈세에 대한 연구를 위해 경제 행동의 기본 특성을 바탕으로 하여 비공식 경제가 어떻게 형성될 수 있는지를 탐구합니다.

- **Performance Highlights**: 연구 결과는 개인의 성격 특성, 외부 내러티브, 단속 가능성, 공공재 제공의 인식된 효율성이 비공식 경제 활동의 시기 및 정도에 큰 영향을 미친다는 것을 보여줍니다. 특히, 공공재의 효율적인 제공과 강력한 단속 메커니즘이 상호 보완적이며, 어느 하나만으로는 비공식 활동을 효과적으로 억제할 수 없다는 사실이 강조되었습니다. 이러한 결과는 향후 경제 정책 수립에서 중요한 시사점을 제공합니다.



### HyperZero: A Customized End-to-End Auto-Tuning System for Recommendation with Hourly Feedback (https://arxiv.org/abs/2501.18126)
- **What's New**: 이 논문에서는 추천 시스템의 두 번째 단계인 가치 모델의 최적화를 위한 자동 조정 기술인 HyperZero를 소개합니다. 기존 자동 조정 방법들은 몇 주 또는 몇 달의 긴 시간이 소요되지만, HyperZero는 2-3일 이내에 적절한 모델을 식별하는 것을 목표로 합니다. 최신 추천 시스템의 고유한 문제를 효과적으로 해결하고, 넓은 조정 작업에도 확장할 수 있는 가능성을 가지고 있습니다.

- **Technical Details**: 이 시스템은 두 주요 모듈인 추정기 모듈과 최적화기 모듈로 구성됩니다. 추정기 모듈은 현재 하이퍼파라미터에 대한 사용자 피드백을 기반으로 목표를 추정하고, 최적화기 모듈은 이 예측을 활용하여 하이퍼파라미터를 업데이트합니다. HyperZero는 제한된 최적화 문제를 풀기 위해 Gaussian 프로세스 추정기와 Thompson 샘플링을 결합하여 여러 목표를 동시에 고려할 수 있도록 합니다.

- **Performance Highlights**: HyperZero는 시간 단위의 피드백을 활용하여 주기적인 조정 사이클을 주에서 일로 단축할 수 있는 엔드투엔드(auto-tuning) 시스템을 개발했습니다. 비독립적으로 시간 신호를 처리하고, 비동기 병렬 샘플링을 구현하여 효율성을 증가시키는데 중점을 두었습니다. 이를 통해 기존의 추천 시스템에서 발생하는 다양한 문제를 해결할 수 있는 잠재력을 가지고 있습니다.



### Improving Minimax Group Fairness in Sequential Recommendation (https://arxiv.org/abs/2501.18117)
Comments:
          This paper has been accepted to the IR for Good track at ECIR 2025

- **What's New**: 이번 연구에서는 Conditional Value at Risk (CVaR) DRO를 활용하여 인기 있는 사용자 그룹과 비주류 사용자 그룹 간의 추천 품질 차이를 줄이기 위한 방법을 제안합니다. 이 방법은 사전 정의된 그룹 주석이 필요 없으며 자연스럽게 중복 그룹을 처리할 수 있습니다. 우리의 실험에서는 CVaR가 전통적인 훈련 방식보다 우수한 성능을 보임을 확인하였습니다.

- **Technical Details**: 리스트가 홀드한 경우, sequential 추천 시스템에서 사용자 u의 아이템 순서 Hu에 기반하여 다음 아이템을 예측합니다. 이 모델들은 일반적으로 empirical risk minimization (ERM)을 사용하여 훈련되며, CGRO와 SDRO는 사전 정의된 그룹 주석을 필요로 합니다. 반면, CVaR 방법은 고손실의 미니 배치를 확인해 모델을 업데이트하며, 이는 그룹 정보를 요구하지 않습니다.

- **Performance Highlights**: 실험 결과, CVaR는 사용자 그룹에 대해 NDCG 점수를 향상시키면서도 전반적으로 더 높은 성능을 발휘했습니다. GDRO와 SDRO는 손실 계산에 사용된 그룹 선택에 매우 민감하게 반응했으며, CVaR는 모든 그룹에 쉽게 확장되고 성능 또한 뛰어났습니다. 이러한 결과는 추천 시스템에서 그룹 의존 방식보다 그룹 비의존 방식을 우선시해야 한다는 중요한 실용적 시사점을 제공합니다.



### RL-based Query Rewriting with Distilled LLM for online E-Commerce Systems (https://arxiv.org/abs/2501.18056)
- **What's New**: 본 논문에서는 온라인 검색의 효율성을 높이고 한계점을 극복하기 위한 새로운 쿼리 리라이트(query rewriting, QR) 파이프라인을 제안합니다. 제안된 방법은 오프라인에서 지식 증류(knowledge distillation)를 통해 효율적인 모델을 생성하고, 온라인에서는 강화 학습(reinforcement learning, RL)을 통해 동적 피드백으로 쿼리 리라이트를 개선합니다. 본 연구에서는 LLM(large language model)을 사용자 피드백의 시뮬레이션으로 활용하여 수동 주석 없이 비용 효율적인 평가를 가능하게 합니다.

- **Technical Details**: 제안하는 파이프라인은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 대형 언어 모델(LLM)로부터 지식 증류를 통해 Mini E-commerce Language Model(MiniELM)을 생성하며, 이는 기존 모델의 의미론적 충실성을 유지하면서 경량화를 이룹니다. 두 번째 단계에서는 MiniELM이 강화 학습을 통해 사용자 피드백을 반영하 여 쿼리 리라이트를 미세 조정하는 방식으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, Amazon ESCI 데이터셋에서 MiniELM은 쿼리의 적합성, 다양성, 적응성에서 유의미한 개선을 보여주었습니다. 또한, LLM 시뮬레이션의 긍정적인 피드백은 제안된 방법의 효과성과 우수성을 뒷받침합니다. 본 연구는 특정 도메인에 대한 LLM의 능력을 향상시키고, 변화무쌍한 전자 상거래 검색 환경에서 강력한 솔루션을 제시합니다.



### Can Generative LLMs Create Query Variants for Test Collections? An Exploratory Study (https://arxiv.org/abs/2501.17981)
Comments:
          Published in the proceedings of SIGIR'23

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 활용하여 정보 요구에 대한 쿼리 및 쿼리 변형을 자동 생성하는 방법을 탐구합니다. LLM이 생성한 쿼리가 인간이 생성한 쿼리와 얼마나 유사한지를 측정하고, 이를 평가하기 위한 다양한 메트릭을 사용하여 문서 풀링(document pooling)에 미치는 영향을 조사합니다. 결과적으로 LLM이 쿼리 변형을 생성하는 데 있어 잠재력을 보여주며, 인간 생성 쿼리의 다양성을 완전히 포괄하지는 않지만, 유사한 관련 문서 세트를 생성할 수 있음을 발견하였습니다.

- **Technical Details**: 논문에서 사용된 실험 설계는 사람과 LLM(GPT-3.5)이 생성한 쿼리 세트를 포함하고 있습니다. 정보 검색(IR)에서 쿼리 변형은 인지의 변화를 고려해 발전해 왔으며, 특히 집단 지성을 통해 생성된 쿼리 변형 세트가 많은 주목을 받아왔습니다. 실험에서는 LLM의 한 가지 학습 방식인 one-shot learning을 채택하여 모델을 프롬프트하고, UQV100 데이터셋의 같은 백스토리를 사용하여 쿼리 변형을 생성했습니다.

- **Performance Highlights**: LLM이 생성한 쿼리는 100개의 백스토리에서 인간의 쿼리와 최대 71.1%의 겹침을 보였습니다. 이는 LLM이 인간과 유사한 문서 풀링 결과를 생성할 수 있음을 나타냅니다. 실험을 통해 쿼리 변형의 중요성이 더욱 뚜렷해졌으며, 이러한 변형들이 정보 검색의 효과성에 긍정적인 영향을 미칠 수 있다는 점도 강조되고 있습니다.



### LLMs can be Fooled into Labelling a Document as Relevant (best caf\'e near me; this paper is perfectly relevant) (https://arxiv.org/abs/2501.17969)
Comments:
          Published in the proceedings of SIGIR-AP'24

- **What's New**: 이 연구는 여러 개의 오픈 소스 및 상용 LLM을 사용하여 짧은 텍스트의 관련성을 평가하는 실험을 보고합니다. 실험 결과, LLM의 문서 관련성 평가가 인간 평가자와 비교해 유사한 일관성을 보여주지만, LLM은 인간보다 더 많이 문서를 관련성 있는 것으로 분류하는 경향이 있음을 발견했습니다. 이는 LLM이 비관련성을 나타내는 라벨이 더욱 신뢰할 수 있다는 점을 시사합니다.

- **Technical Details**: 연구는 TREC 2021 및 TREC 2022의 패시지 검색 작업에서 쿼리 및 패시지를 사용하여 LLM의 성능을 평가합니다. 이를 위해 7개의 정보 검색 시스템 (IR system)을 사용하였고, 다양한 LLM에서 생성된 관련성 라벨의 품질을 평가하기 위해 4점 척도를 사용했습니다. 또한, LLM의 성능에 영향을 미치는 여러 유형의 프롬프트를 사용하여 관련성 라벨의 정확성을 실험적으로 검토했습니다.

- **Performance Highlights**: 실험 결과, LLM은 특히 원래 쿼리 단어가 포함된 문장을 관련성 있는 것으로 잘 분류하는 경향이 있었습니다. 쿼리 단어의 존재가 LLM의 라벨링에 강한 영향을 미치고 있으며, 이는 LLM의 현재 라벨링 기준의 약점을 드러냅니다. 연구는 LLM의 잠재적인 취약점과 관련성 평가의 신뢰성을 높이기 위한 테스트와 메트릭을 제안합니다.



### Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method (https://arxiv.org/abs/2501.18539)
- **What's New**: 이 논문은 복잡한 질문에 대한 정보를 검색하는 데 있어 LLM 기반의 새로운 retrieval 방법인 ARM을 제안합니다. ARM은 질문과 데이터 수집의 조직 간의 정렬을 메꿔, 기존의 검색 방법이 간과할 수 있는 필요한 정보에 대한 더 나은 탐색을 가능하게 합니다. 특히, 반복적인 쿼리가 아닌 문맥에 맞는 데이터 오브젝트 간의 관계를 분석함으로써 효율성을 극대화할 수 있습니다.

- **Technical Details**: ARM은 LLM(대형 언어 모델)의 추론 능력을 활용하여 질문에 필요한 데이터를 효율적으로 검색하는 방식으로 설계되었습니다. 이 시스템은 정보 정렬(information alignment)과 구조 정렬(structure alignment) 단계를 통해 데이터를 정리하며, 자가 검증(self-verification) 과정을 통해 최종적으로 적합한 데이터 오브젝트를 선택합니다. 이를 위해 N-gram을 사용하여 각 오브젝트의 주요 정보를 요약하고, 임베딩(embedding)을 통해 의미적 유사성 검색을 지원합니다.

- **Performance Highlights**: 실험 결과 ARM은 Bird 데이터셋에서 기존의 RAG 방법들보다 최대 15.9 포인트의 정확도로 우수한 성능을 보였습니다. OTT-QA 데이터셋에서도 ARM은 이전 방법들에 비해 최대 19.3 포인트 높은 F1 점수를 기록하며, LLM 기반의 질문 대응 문제에서 효과적인 해결책을 제시합니다. 이로 인해 복잡한 질문에 대한 검색 능력이 한층 향상되었습니다.



### RbFT: Robust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects (https://arxiv.org/abs/2501.18365)
- **What's New**: 이 논문은 Retrieval-augmented generation (RAG) 시스템의 신뢰성을 높이기 위해 Robust Fine-Tuning (RbFT)이라는 새로운 방법론을 제안합니다. 기존 RAG 시스템의 정보 검색 과정에서 발생할 수 있는 오류나 불완전한 정보로 인한 문제를 해결하고자 합니다. RbFT는 LLM의 방어 능력을 강화하여 부정확한 정보에도 불구하고 신뢰할 수 있는 응답을 생성하도록 돕습니다.

- **Technical Details**: RbFT는 두 가지 주요 세부 과제로 구성되어 있습니다: Defects Detection(결함 탐지)와 Utility Extraction(유용성 추출)입니다. 이 방법들은 LLM이 결함이 있는 입력을 평가하고, 유용한 정보를 효과적으로 활용할 수 있도록 돕습니다. LLM은 결함이 있는 문서를 사용하여 실제 응답을 생성하는 훈련을 받고, 이는 LAG 시스템의 전반적인 내구성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RbFT는 다양한 검색 조건에서도 RAG 시스템의 견고성을 크게 향상시켰습니다. 기존의 방법들을 초월하며, 높은 추론 효율성과 다른 견고성 강화 기법과의 호환성을 유지합니다. 이 연구는 RAG 시스템의 성능과 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Hashtag Re-Appropriation for Audience Control on Recommendation-Driven Social Media Xiaohongshu (rednote) (https://arxiv.org/abs/2501.18210)
- **What's New**: 이 연구는 여성들이 Xiaohongshu라는 추천 기반 소셜 플랫폼에서 남성 사용자를 차단하기 위해 해시태그를 재적용하는 방법을 탐구합니다. 특히, #Baby Supplemental Food 해시태그를 사용하여 원래 의미와는 다른 목적을 달성하려고 노력하는 여성들의 사례를 분석하였습니다. 이 연구는 추천 알고리즘의 투명성 부족이 소외된 집단에 미치는 영향을 살펴보고, 사용자들이 알고리즘 기반 플랫폼에서 자율성을 어떻게 회복할 수 있는지를 조명합니다.

- **Technical Details**: 연구에서는 총 5800개의 게시물과 24명의 다양한 배경을 가진 활성 사용자들을 인터뷰하여 해시태그 재적용의 동기와 반응을 조사하였습니다. 믹스드 방법론( mixed-methods approach)을 사용하여 해시태그 사용 패턴을 분석하고, 해시태그 관련 게시물의 관련성 분석, 주제 모델링(topic modeling), 게시물 표현 분석(post expression analysis) 및 해시태그 동시 발생 네트워크 분석을 수행하였습니다. 이를 통해 사용자들이 알고리즘에 대해 보다 통제할 수 있는 방법을 모색하고자 하였습니다.

- **Performance Highlights**: 이 연구에서 발견된 해시태그 재적용 관행은 여성들이 원치 않는 관객을 차단하면서 서로 연결된 게시물을 통해 자신의 내러티브를 형성할 수 있는 방법을 보여줍니다. 해시태그 #BSF의 재적용은 디지털 페미니즘에 대한 일상적인 저항의 형태로 작용하며, 추천 기반 플랫폼에서의 관객 통제 및 사용자 자치의 가능성을 제시합니다. 이러한 결과는 곧바로 알고리즘 중심 구조 내에서 사용자의 자율성 회복을 위한 시사점을 제공합니다.



New uploads on arXiv(cs.CV)

### ROSA: Reconstructing Object Shape and Appearance Textures by Adaptive Detail Transfer (https://arxiv.org/abs/2501.18595)
- **What's New**: 이번 연구에서 우리는 ROSA라는 새로운 역 렌더링(inverse rendering) 기법을 제안하며, 이는 이미지 데이터만을 기반으로 적응형 메쉬 해상도(mesh resolution)를 최적화하여 물체의 형상(shape)과 외관(appearance)을 재구성합니다. 기존의 접근 방식들은 주로 큰 메쉬나 노멀 맵(normal map)을 사용하여 재구성하는 방식을 취했으나, 이는 여러 가지 문제점을 동반했습니다. 특히, 우리는 정밀한 시각화를 위해 재구성을 할 때 메쉬의 곡률(curvature)과 노멀 텍스처(normal texture)를 기반으로 표면의 매끄러움을 조정합니다.

- **Technical Details**: ROSA는 메쉬의 형상을 매개변수화된 삼각형 메쉬로 나타내고, 외관 정보는 재구성 과정 동안 단일 디코더 네트워크(decoder network)를 활용하여 생성된 텍스처 아틀라스(texture atlas)로 표현합니다. 연구진은 고차원 재료 특성을 효과적으로 다루기 위해, 고유의 노멀 손실(normal loss) 기준을 개발하고, 이를 통해 세밀한 디테일을 메쉬 지오메트리(geometry)와 노멀 맵에 적절히 배분합니다. 아울러, 고해상도 텍스처를 생성하기 위해 타일 기반의 기법을 도입하였고, 이는 고해상도 외관 특성을 재구성하는 데에도 효과적입니다.

- **Performance Highlights**: ROSA는 추가적인 후처리(post-processing) 없이 변형이나 왜곡 없이 물체의 형상과 외관을 안정적으로 재구성할 수 있습니다. 이 방법은 기존 노멀 맵의 한계를 극복하고, 상대적으로 컴팩트한 표현을 유지하면서도 고해상도 텍스처를 생성하는 데 강점을 보입니다. 평가 결과, 우리의 방법은 기존 기술들보다 더 나은 시각적 품질을 나타내며, 세밀한 외관 재현에 있어 뛰어난 성능을 발휘했습니다.



### Foundational Models for 3D Point Clouds: A Survey and Outlook (https://arxiv.org/abs/2501.18594)
Comments:
          Initial submission

- **What's New**: 이번 논문은 3D 포인트 클라우드를 이해하기 위한 기초 모델(Foundation Models, FMs)의 최신 발전을 다룹니다. 특히, 2D 지식을 활용하여 3D 작업에 적용하는 방법론을 탐구하고, 언어 모델(LLMs)이 3D 이해를 높이는 데 어떻게 기여할 수 있는지를 설명합니다. 기존 문헌의 부족한 점을 보완하기 위해, 3D 시각 이해를 위한 다양한 방법을 포괄적으로 정리하여 제공합니다.

- **Technical Details**: 3D 포인트 클라우드는 공간 데이터 표현의 기본 패러다임으로, 컴퓨터 비전, 로보틱스, 자율 차량, 증강 현실 등 다양한 분야에서 중요한 역할을 합니다. 그러나 3D 데이터셋은 수집 및 주석 작업이 복잡하고 비용이 많이 들며, 대부분의 데이터가 부족한 상황입니다. 이러한 문제를 해결하기 위해, 저자들은 2D 모델을 기반으로 3D FMs을 개발하고 다양한 3D 작업(예: 분류, 분석 등)에 적용하는 접근 방법을 제시하고 있습니다.

- **Performance Highlights**: 이 논문은 2D 및 3D FMs을 활용하여 3D 작업을 수행하는 최신 기술을 정리하고 성능을 분석합니다. 특히, LLMs와 비전 기반 모델을 통합하여 3D 작업에 대한 텍스트 설명과 3D 모델 간의 정렬을 가능하게 하는 방법론을 강조합니다. 이를 통해, 3D 포인트 클라우드 이해에서의 성과를 높이고 향후 연구 방향에 대한 통찰을 제공하고 있습니다.



### Diffusion Autoencoders are Scalable Image Tokenizers (https://arxiv.org/abs/2501.18593)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 간단한 확산 토크나이저(Diffusion Tokenizer, DiTo)를 소개합니다. 이 토크나이저는 이미지 생성 모델을 위한 컴팩트한 시각적 표현을 학습하는데 초점을 맞추고 있습니다. 기존의 복잡한 비지도 모델 대신, 단일한 학습 목표인 확산 L2 손실(diffusion L2 loss)을 활용하여 확장 가능한 이미지 토크나이저를 효율적으로 훈련할 수 있다는 것이 주요 통찰입니다.

- **Technical Details**: DiTo는 이미지 생성을 위한 잠재 표현 학습을 위한 새로운 접근 방식을 제시합니다. 이는 확산 모델(difussion model)의 이론적 근거를 바탕으로 하여, 단일한 확산 L2 손실을 사용해 훈련하게 됩니다. 제안된 기술은 토크나이저 학습에서 여러 손실 조합을 요구하는 기존 방법들과 대비되어 보다 간단하고 효율적인 모델 학습을 가능하게 합니다.

- **Performance Highlights**: DiTo는 이미지 재구성과 다운스트림 이미지 생성 작업에서 기존 최첨단 모델과 비교하여 경쟁력 있는 혹은 더 나은 품질을 달성합니다. 특히 작은 텍스트나 기호, 구조적 비주얼 부분 처리에서 기존의 GLPTo보다 더 뛰어난 결과를 보여줍니다. 또한, 모델 크기를 늘려도 손실 하이퍼파라미터 조정이 필요 없어 간편하게 확장 가능하다는 장점을 가지고 있습니다.



### Advances in Multimodal Adaptation and Generalization: From Traditional Approaches to Foundation Models (https://arxiv.org/abs/2501.18592)
Comments:
          Project page: this https URL

- **What's New**: 본 연구는 다중 모드(domain adaptation) 적응 및 일반화(multimodal generalization)에 대한 최근의 발전 사항을 포괄적으로 정리했습니다. 전통적인 접근 방식에서부터 다중 모드 기초 모델까지 다양한 방법론을 다루며, 아카이브(arXiv)에서 활발히 업데이트되고 있는 자료들을 포함합니다. 각 주제에 대한 문제 정의와 기존 방법들에 대한 철저한 검토가 이루어지고 있어, 향후 연구 방향에 대한 통찰도 제공합니다.

- **Technical Details**: 다중 모드 적응 및 일반화를 위한 다양한 알고리즘이 제안되었으며, 특히 시각-언어(target domains), 오디오-비디오(audio-video) 및 LiDAR-카메라(LiDAR-camera)와 같은 데이터 소스로부터 출발합니다. 이 논문에서는 MMDA(Multimodal Domain Adaptation), MMDG(Multimodal Domain Generalization), 그리고 MMTTA(Multimodal Test-Time Adaptation) 등의 주요 개념들과 함께 이들을 향상시키기 위한 기초 모델들의 역할도 탐구합니다. 더불어, 각 방법에 대한 체계적인 분석과 기존 데이터셋, 응용 분야를 정리하였습니다.

- **Performance Highlights**: MMDA와 MMDG는 최근 행동 인식(action recognition)과 의미 분할(semantic segmentation) 분야에서 눈에 띄는 성과를 내고 있습니다. 특히, 다중 모드 기초 모델의 활용을 통해 모델들의 일반화 성능이 향상되는 것을 확인하였습니다. 이 연구는 기존 알고리즘의 이해를 도모하며, 다중 모드 적응 수준에서의 보완 정보를 효과적으로 활용하기 위한 새로운 방향을 제시합니다.



### DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models (https://arxiv.org/abs/2501.18590)
Comments:
          Project page: this http URL

- **What's New**: DiffusionRenderer는 복잡한 조명 효과를 모델링하고 시뮬레이션하는 최신의 신경망 기반 접근 방식을 제안합니다. 이 모델은 실세계 비디오로부터 G-buffer를 역 추정하므로 이미지 편집 작업을 지원하며, 물체 삽입, 재조명 등을 가능하게 합니다. 또한, DiffusionRenderer는 노이즈가 있는 G-buffer로부터 포토리얼리스틱 이미지 및 비디오를 합성할 수 있는 기능을 제공합니다.

- **Technical Details**: DiffusionRenderer는 비디오 확산 모델의 강력한 선행 모델을 활용하여 역 렌더링과 전방 렌더링 문제를 함께 처리할 수 있도록 설계되었습니다. 이 모델은 고품질의 데이터 집합을 필요로 하며, 실세계 시나리오에서도 안정적인 일반화를 수행합니다. 비디오로부터 생성된 '의사 레이블들'을 사용하여 강력한 훈련 데이터를 구축하고, 이를 통해 다양한 이미지 및 비디오 편집 기능을 지원합니다.

- **Performance Highlights**: DiffusionRenderer는 최신 기술을 뛰어넘는 성능을 자랑하며, 복잡한 역 렌더링과 전방 렌더링 기능을 효과적으로 근사합니다. 단일 비디오 입력만으로 다양한 씬에서 재조명 및 현실적인 물체 삽입을 가능하게 해, 실제 세계의 신경 렌더링 응용 프로그램의 가능성을 확장합니다. 실험 결과, 제안된 모델은 이미지와 비디오의 편집 작업에서 뛰어난 결과를 보였습니다.



### UDC-VIT: A Real-World Video Dataset for Under-Display Cameras (https://arxiv.org/abs/2501.18545)
Comments:
          Main body (10 pages, 9 Figures, 3 Tables), References (4 pages), Appendix (15 pages, 11 Figures, 6 Tables)

- **What's New**: 이 논문에서는 실제 환경에서 촬영한 UDC 비디오 데이터셋인 UDC-VIT를 제안합니다. 기존의 합성 UDC 데이터셋이 현실의 UDC 왜곡 특성을 반영하지 못하는 것과는 달리, UDC-VIT는 얼굴 인식을 목표로 한 인간 동작만을 포함하고 있습니다. 이 데이터셋은 비디오 캡처 시스템을 통해 비왜곡 비디오와 UDC 왜곡 비디오를 동시에 획득하고, 이를 정밀하게 정렬하여 수집합니다.

- **Technical Details**: UDC-VIT의 비디오 캡처 시스템은 비편광 큐브 빔 스플리터를 사용하여 고정밀의 동기화된 프레임 쌍을 생성합니다. 이 시스템은 Samsung Galaxy Z-Fold 스마트폰의 UDC 영역을 활용하여 구성되며, Arducam Hawk-Eye 카메라 모듈을 통해 고화질의 데이터를 캡처합니다. 이러한 시스템은 기존 데이터셋에서의 현실적인 왜곡 문제를 해결하기 위해 디스크리트 푸리에 변환(DFT)을 사용하여 프레임 간의 픽셀 위치 차이를 보정합니다.

- **Performance Highlights**: UDC-VIT는 여섯 개의 딥러닝 모델을 사용하여 기존 합성 데이터셋과 비교하였고, 그 결과 합성 데이터셋으로 훈련된 모델들이 UDC 왜곡 비디오의 실제 특성을 반영하지 못함을 증명하였습니다. 또한, PSNR, SSIM 및 LPIPS 점수와 관련하여 얼굴 인식 정확도를 평가함으로써 UDC 복원 기술의 중요성을 강조하였습니다. UDC-VIT 데이터셋은 UDC 비디오 복원 분야의 향후 연구를 위한 기초 자료로 제공되며, 프로젝트 사이트에서 다운로드할 수 있습니다.



### Learning Priors of Human Motion With Vision Transformers (https://arxiv.org/abs/2501.18543)
Comments:
          2024 IEEE 48th Annual Computers, Software, and Applications Conference (COMPSAC). IEEE, 2024

- **What's New**: 이 논문은 Vision Transformers (ViTs)를 기반으로 한 신경망 구조를 제안하여 인간의 이동 경로, 속도, 정지 위치를 예측하는 문제를 다룹니다. 이 방식은 Convolutional Neural Networks (CNNs)보다 공간 상관관계를 더 효과적으로 캡처할 수 있음을 주장하고 있습니다. 연구진은 ViT 아키텍처가 공장 자동화와 같은 다양한 로봇 내비게이션 작업에 필요한 인간의 동작에 대한 기초 정보를 제공할 수 있다고 강조합니다.

- **Technical Details**: 제안된 신경망은 세멘틱 맵을 통해 관측된 지역을 작은 구획으로 나누고, 낮은 복잡도로 실시간 예측을 수행할 수 있도록 설계되었습니다. 이 네트워크는 공간적으로 관련된 구획 간의 관계를 이해하는 데 강점을 가지며, 이는 인간이 다양한 공간을 사용하는 방식을 학습하는데 기여합니다. 알고리즘은 신속한 환경 변화에도 불구하고 실시간 계산 비용을 유지하며 성능을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, ViT 기반 모델이 기존 방법(기본선)에 비해 정확도 측면에서 뛰어난 성능을 보였음을 명확하게 입증하였습니다. 이는 모바일 로봇이 복잡하고 동적인 환경에서 내비게이션 할 때 자연스러운 선택이 될 수 있음을 강화합니다. 이 연구는 환경 안에서의 인간 행동을 이해하고 예측할 수 있는 중요한 발전을 이끌어 내고 있습니다.



### Mini-ResEmoteNet: Leveraging Knowledge Distillation for Human-Centered Design (https://arxiv.org/abs/2501.18538)
Comments:
          5 pages with 4 figures

- **What's New**: 이번 연구에서는 사용자 경험(User Experience)에서 점점 더 중요해지고 있는 Facial Emotion Recognition(표정 감정 인식) 기술을 활용하여, Mini-ResEmoteNet 모델을 개발하는 지식 증류(Knowledge Distillation) 프레임워크를 도입했습니다. 이 모델은 경량화된(student) 학생 모델로, 현대의 사용성 테스트(usability testing)에 적합하도록 설계되었습니다.

- **Technical Details**: 연구진은 FER2013 및 RAF-DB 데이터셋을 사용하여 세 가지 학생 모델 아키텍처(Student Model A, B, C)의 효능을 평가했습니다. 각 학생 모델은 교사 모델(teacher model)의 각 레이어에서 특징 채널(feature channel)의 수를 대략 50%, 75%, 87.5%씩 줄이는 방식으로 개발되었습니다.

- **Performance Highlights**: 특히 Student Model A(E1)가 FER2013 데이터셋에서 76.33%의 테스트 정확도로, EmoNeXt보다 0.21%의 절대 향상을 기록하며 뛰어난 성능을 보였습니다. 또한, 제안된 방법은 ResEmoteNet 모델과 비교했을 때 추론(inference) 속도와 메모리 사용(memory usage) 측면에서도 개선된 결과를 보여주었습니다.



### Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models (https://arxiv.org/abs/2501.18533)
- **What's New**: 본 연구는 Multi-Image Safety (MIS) 데이터셋을 제안하여 안전 관련 비주얼 추론(safety visual reasoning) 및 시각적 인식을 개선하고자 합니다. 대규모 Vision-Language Models (VLMs)의 안전성 문제를 해결하기 위한 새로운 접근 방법을 제공하며, 기존의 안전 조정 방법들의 한계를 강조합니다. MIS 데이터셋은 다중 이미지와 안전 Chain-of-Thought (CoT) 레이블을 통합하여 모델의 성능을 향상시킵니다.

- **Technical Details**: MIS 데이터셋은 훈련 세트와 테스트 세트로 나누어져 있으며, LLMs, VLMs, Text-to-Image 모델을 이용한 자동 데이터 생성 프레임워크를 사용합니다. 특히, MIS는 두 개의 이미지를 결합하여 생기는 위험한 의도를 해석하고 해결해야 하는 과제를 모델에게 부여합니다. 이 과정에서 모델은 효과적인 시각 인식 및 사고 과정을 요구받고, 특히 안전성을 고려한 응답 생성을 목표로 합니다.

- **Performance Highlights**: MIS 데이터셋을 사용하는 InternVL2.5-8B의 미세 조정(fine-tuning)을 통해 강력한 오픈소스 모델 및 API 기반 모델에 비해 우수한 성능을 보여주었습니다. 평균 정확도는 5개의 일반 벤치마크에서 0.83% 증가했으며, 여러 안전 벤치마크에서 Attack Success Rate (ASR)를 크게 감소시켰습니다. 이러한 결과는 MIS 접근 방식이 안전성 및 일반적 능력을 모두 향상시킬 수 있음을 보여줍니다.



### Integrating Spatial and Frequency Information for Under-Display Camera Image Restoration (https://arxiv.org/abs/2501.18517)
Comments:
          Main body (10 pages, 9 Figures, 5 Tables), References (3 pages), Appendix (8 pages, 6 Figures, 6 Tables)

- **What's New**: 이 논문에서는 언더 디스플레이 카메라(UDC)의 이미지 복원을 위한 새로운 다중 수준 딥 뉴럴 네트워크 아키텍처인 SFIM(Spatial and Frequency Interactive learning in a Multi-level architecture)을 제안합니다. SFIM은 CNN(Convolutional Neural Network)과 FFT(Fast Fourier Transform) 기반 모델을 통합하여 지역 정보(local information)와 전역 정보(global information)를 효과적으로 결합하는 것을 목표로 합니다. UDC의 왜곡을 복원하는 과정에서 발생하는 다양한 복잡한 패턴을 다루기 위해, 이 연구는 스페셜 도메인(spatial domain)과 주파수 도메인(frequency domain) 모두에서 정보를 활용합니다.

- **Technical Details**: SFIM 아키텍처는 세 가지 주요 구성 요소를 포함합니다: 공간 도메인 블록(SDB), 주파수 도메인 블록(FDB), 그리고 주의(attention) 기반 다중 수준 통합 블록(AMIB)입니다. SDB는 잡음(noise)과 흐림(blur)과 같은 세부 질감을 복원하는 데 중점을 두며, FDB는 전체 영역에서의 플레어(flare)를 다루는 데 중점을 둡니다. AMIB는 효과적인 교차 도메인(inter-domain) 상호작용을 가능하게 하며, 각 수준의 아키텍처를 통합하여 플레어 관련 특징에 네트워크가 집중할 수 있도록 유도합니다.

- **Performance Highlights**: 실험 결과, SFIM은 기존의 최첨단 모델들에 비해 UDC 벤치마크에서 우수한 성능을 보여주며, 특히 넓은 영역에 걸친 불규칙한 텍스처 손실 복원에서 최고의 품질을 제공합니다. 정량적 및 정성적 평가를 통해 SFIM이 UDC 이미지 복원 분야의 기존 접근 방식보다 효과적으로 잡음과 흐림을 제거할 수 있음을 입증하였습니다. 이 논문은 UDC 이미지 복원에 있어 다중 수준 아키텍처의 중요성을 강조하며, 플레어 제거에서의 우수성을 입증합니다.



### Deconstruct Complexity (DeComplex): A Novel Perspective on Tackling Dense Action Detection (https://arxiv.org/abs/2501.18509)
Comments:
          Computer Vision

- **What's New**: 이번 연구에서는 밀집 행동 탐지(dense action detection) 문제를 해결하기 위해 새로운 관점을 제시합니다. 기존의 단일 네트워크 방식 대신, 행동 클래스의 핵심 개념을 감지하는 두 개의 전문화된 네트워크를 제안합니다. 이 접근법은 정적(static) 개념과 동적(dynamic) 개념을 분리하여 탐지함으로써 문제를 더 효과적으로 해결합니다.

- **Technical Details**: 연구에서 제안된 DeComplex 네트워크는 두 개의 스트림으로 구성됩니다. 하나는 정적(static) 개념을 탐지하는 Static 스트림이고, 다른 하나는 동적(dynamic) 개념을 탐지하는 Dynamic 스트림입니다. 각 스트림은 적절한 구조로 설계되어 있으며, 이에 따라 학습 과정에서 Binary Cross-Entropy (BCE)와 밀집(static/dynamic) 레이블을 이용해 최적화됩니다.

- **Performance Highlights**: 이 연구에서는 Charades와 MultiTHUMOS와 같은 도전적인 벤치마크 데이터셋에서 23.4%와 2.5%의 mAP(mean Average Precision) 향상을 달성하였습니다. 이는 제안된 방법이 기존 최첨단 기법보다 성능 향상이 뛰어남을 입증합니다. 또한, 이 연구는 동시 발생 개념(co-occurring concepts)에 대한 명시적 감독(supervision)을 통해 네트워크 최적화를 개선할 수 있음을 보여줍니다.



### CLEAR: Cue Learning using Evolution for Accurate Recognition Applied to Sustainability Data Extraction (https://arxiv.org/abs/2501.18504)
Comments:
          9 pages plus 2 pages of supplemental material

- **What's New**: 이번 논문에서는 이미지에서 데이터 추출을 위한 효과적인 도구인 대형 언어 모델(Large Language Model, LLM) 이미지 인식의 정확도를 높이기 위한 새로운 기법, Cue Learning using Evolution for Accurate Recognition (CLEAR)를 소개합니다. 이 방법은 진화 계산(evolutionary computation)과 LLM의 조합을 활용하여 이미지 내의 전문적인 특징 인식을 향상시키기 위한 단서(cue)를 생성하고 최적화합니다. 이는 도메인 특정 표현 생성과 유전 알고리즘(genetic algorithm)을 통한 적절한 텍스트 단서 최적화를 통해 이루어집니다.

- **Technical Details**: CLEAR는 변동 길이 표현(variable-length representation)을 활용하여 고정 길이 표현(fixed-length representation)과 비교하여 인식의 정확도를 높이는 방법을 연구합니다. 또한, 카테고리 기반 추정(category-based estimates)에서 실수 값(real-valued estimates)으로 리팩토링(refactoring)함으로써 LLM의 일관성을 개선하는 방법에 대해 논의합니다. 이러한 접근은 특정 도메인에 맞는 새로운 표현을 자동 생성하고, 이에 따라 맞춤형 텍스트 단서를 최적화하는 과정을 포함합니다.

- **Performance Highlights**: CLEAR는 빌딩의 내부 및 외부 이미지에서 지속 가능성 데이터(sustainability data)를 인식하는 실제 작업에 적용되었으며, 전문가의 인식 및 사람의 작성된 단서보다 모든 작업에서 더 높은 정확도를 달성하는 결과를 보였습니다. 오류율(error rates)은 최대 두 자릿수 개선을 보였으며, 삭감 연구(ablation study)를 통해 솔루션의 간결성을 입증했습니다.



### HSRMamba: Contextual Spatial-Spectral State Space Model for Single Hyperspectral Super-Resolution (https://arxiv.org/abs/2501.18500)
- **What's New**: 이번 논문은 HSRMamba라는 새로운 모델을 제안합니다. 이 모델은 hyperspectral image super-resolution (HSISR)을 위한 컨텍스트 기반의 공간-스펙트럼 모델링 상태 공간 모델로, 기존 Mamba 모델의 한계를 극복합니다. HSRMamba는 지역적 및 전역적 causal 관계를 효과적으로 설정하여 공간 및 스펙트럼의 상세 복원을 향상시킵니다.

- **Technical Details**: HSRMamba는 지역적 공간-스펙트럼 분할 메커니즘을 통해 인접 픽셀 간의 패치 별 원인 관계를 구축합니다. 또한, 전역 스펙트럼 재정렬 전략을 도입하여 유사한 픽셀 간의 causal representation을 향상시킵니다. 이 두 가지 접근 방식은 고유한 지역적 및 전역적 spatio-spectral 종속성을 효과적으로 포착합니다.

- **Performance Highlights**: 실험 결과, HSRMamba는 기존의 최첨단 방법들과 비교하여 정량적 품질과 시각적 결과 모두에서 우수한 성능을 보였습니다. 특히, 다양한 데이터셋을 통해 HSRMamba의 효과성과 우수성을 입증하였습니다. 곧 코드도 공개될 예정입니다.



### Runway vs. Taxiway: Challenges in Automated Line Identification and Notation Approaches (https://arxiv.org/abs/2501.18494)
Comments:
          Accepted at SysCon 2025

- **What's New**: 이번 논문은 자율 항공 시스템에서의 활주로 및 택시웨이 표식의 정확한 레이블링 필요성을 강조합니다. 기존의 레이블링 알고리즘인 ALINA는 택시웨이 표식 식별에 성공하였으나 활주로 표식에서는 큰 도전에 직면했습니다. 이를 해결하기 위해 ALINA에 Convolutional Neural Network (CNN) 기반의 새로운 분류 단계를 통합하여, 환경 변화와 잘못된 분류에 대한 강인성을 높였습니다.

- **Technical Details**: 이 논문에서는 ALINA의 색상 임계값 조정과 관심 영역(ROI) 선택 개선을 통해 활주로에 특화된 레이블링 프로세스를 발전시켰습니다. 머신러닝 기술을 포함하는 CNN인 AssistNet을 도입하여 활주로 표식 탐지의 정확성과 강인성을 증대시키기 위한 방법론이 소개되었습니다. ALINA는 택시웨이 선 마킹을 탐지하기 위해 기하학적 조정과 색공간 변환을 통해 이진 픽셀 맵을 생성하는 프레임워크로, CIRCLEDAT 알고리즘을 사용하여 택시웨이 표식 픽셀을 식별합니다.

- **Performance Highlights**: 새롭게 제안된 접근 방식은 AssistTaxi 데이터셋에 대한 실험을 통해 활주로 표식 탐지에서 기존 ALINA의 성능을 신뢰성을 높이고 실제 사용 환경에도 적합한 결과를 도출했습니다. 초기 수정 작업을 통해 얻은 성과는 알루미늄의 큰 개선을 도출하지 못했으나, 활주로와 택시웨이를 구분하기 위한 분류 단계의 필요성을 강조하였습니다. 향후 연구는 동적 ROI 조정과 같은 방식을 통해 더욱 신뢰성 있는 활주로 표식 탐지 방법론을 제시할 것으로 기대됩니다.



### Track-On: Transformer-based Online Point Tracking with Memory (https://arxiv.org/abs/2501.18487)
Comments:
          ICLR 2025

- **What's New**: 이번 논문에서는 온라인 방식으로 장기 포인트 추적(long-term point tracking) 문제를 다루고 있습니다. 새로운 모델, Track-On을 소개하며, 이는 기존 방식이 미래 프레임에 의존하는 것과 달리, 현재 프레임에서만 정보를 처리합니다. Track-On은 공간 메모리(spatial memory)와 맥락 메모리(context memory)를 활용하여 시간적 정보를 캡처하고, 높은 정확도로 포인트를 추적할 수 있게 설계되었습니다.

- **Technical Details**: 이 모델은 비디오 프레임을 순차적으로 처리하며, 포인트를 쿼리(query)로 취급하여 변환기(decoder) 구조에서 업데이트합니다. 두 가지 메모리 모듈, 즉 공간 메모리는 최근 프레임으로부터 정보를 업데이트하여 위치 변동을 줄이고, 맥락 메모리는 과거 프레임에서 포인트의 임베딩을 저장하여 시간적 연속성을 보장합니다. 이 구조는 전체 비디오 시퀀스에 대한 완전한 시간 모델링의 높은 계산 및 메모리 비용을 피하면서도 신뢰할 수 있는 포인트 추적을 가능하게 합니다.

- **Performance Highlights**: Track-On은 온라인 모델들 중에서 새로운 최첨단 성능(state-of-the-art performance)을 보여주며, TAP-Vid 벤치마크를 포함한 7개의 데이터셋에서 오프라인 접근 방식보다 우수하거나 경쟁력 있는 결과를 제공합니다. 이 방법은 로봇 기술이나 증강 현실과 같은 다양한 응용 분야에서 실시간 추적을 위한 강력하고 확장 가능한 해결책을 제공합니다.



### SimpleDepthPose: Fast and Reliable Human Pose Estimation with RGBD-Images (https://arxiv.org/abs/2501.18478)
- **What's New**: 이 논문에서는 다중 뷰와 다중 인물의 포즈 추정에서 깊이 정보(depth information)를 활용하는 혁신적인 알고리즘인 SimpleDepthPose를 소개합니다. 이 알고리즘은 RGBD 이미지에서 효율적으로 포즈를 추정할 수 있으며, 추가적인 훈련 없이 다양한 시나리오에 잘 일반화됩니다. 제안된 방법은 공개되어 있어 후속 연구에 기여할 수 있습니다.

- **Technical Details**: SimpleDepthPose는 각 컬러 이미지에 대해 관절 좌표를 예측한 후, 깊이 이미지에서 감지된 각 관절의 거리를 추출합니다. 이를 바탕으로 3D 포즈를 제안하며, 이상치를 필터링하고 최종 3D 포즈를 산출하는 과정을 포함합니다. 이 과정에서 HigherHrNet 모델을 활용하여 오직 가시적인 키포인트만을 예측하도록 덧붙입니다.

- **Performance Highlights**: 제안된 SimpleDepthPose는 OpenPTrack과의 직접 비교에서 우수한 성능을 보였으며, 다중 뷰에서 포즈를 정확하게 예측하는 데 강점을 나타냈습니다. 이 알고리즘은 노이즈에 강하며 빠른 런타임 성능을 제공, 다양한 환경에서도 안정적으로 작동합니다.



### Tuning Vision Foundation Model via Test-Time Prompt-Guided Training for VFSS Segmentations (https://arxiv.org/abs/2501.18474)
- **What's New**: 본 논문에서는 비디오 경기 관찰 데이터(VFSS-5k)를 활용하여 세미 자가 지도 학습을 위한 새로운 Test-Time Training(TTT) 패러다임을 제안합니다. 이 접근법은 완전 주석이 필요 없으므로 의료 영상 분야의 문제를 해결하는 데 효과적입니다. 이 연구는 TTT를 통해 기본 모델이 더 나은 성능을 발휘하도록 하며, 단일 점 프롬프트를 사용하여 교육하는 방법을 중앙에 두고 있습니다.

- **Technical Details**: 제안된 방법론에서는 점 프롬프트를 사용하여 테스트 시간에서 모델 학습을 개선합니다. 이 과정은 모델이 다양한 증강을 통해 점 프롬프트의 모호성을 해결하도록 하여, 학습 과정에서 얻은 모호성 정보를 대조 손실(contrastive loss)을 통해 통합합니다. 이를 통해 데이터가 미비한 의학 이미지 동안 모델의 적응 성능을 증가시키고, 더 효율적인 도메인 적응을 가능하게 합니다.

- **Performance Highlights**: 새롭게 설계된 VFSS-5k 데이터셋을 통해 12개 해부학적 구조에 대해 평균 Dice 계수 0.868을 달성하여, 이전 TTT 방법론보다 우수한 성능을 보여주었습니다. 또한, 전문가 모델과의 성능 격차를 줄이는데 기여하며, 향후 의료 영상 분류에서의 활용 가능성을 밝혔습니다.



### A Benchmark and Evaluation for Real-World Out-of-Distribution Detection Using Vision-Language Models (https://arxiv.org/abs/2501.18463)
- **What's New**: 이 논문에서는 기존의 성능 포화 문제를 해결하기 위해 세 가지 새로운 OOD(Out-of-Distribution) 검출 벤치마크를 소개합니다. 첫 번째는 ImageNet-X로, 도전적인 의미적 변화 하에서의 성능 평가를 목표로 합니다. 두 번째는 ImageNet-FS-X로, 특징 분포의 변화에 대한 강건성을 평가하며, 세 번째는 Wilds-FS-X로 실제 데이터셋을 사용하여 보다 포괄적인 테스트 환경을 제공합니다.

- **Technical Details**: 제안된 세 가지 벤치마크는 (1) ID와 OOD 데이터 간의 의미적 유사성, (2) 의미적 변화와 공변량 변화의 두 가지 분포 변화, (3) 실제 상황과의 일치를 평가합니다. ImageNet-X는 의미적 유사성이 작은 데이터셋으로 ID와 OOD 데이터를 분할하고, ImageNet-FS-X는 공변량 변화를 포함하여 두 분포 변화를 체계적으로 분석하는 벤치마크입니다. Wilds-FS-X는 실제 상황을 반영하여 데이터의 자연적인 분포 변화를 고려합니다.

- **Performance Highlights**: 실험 결과, 최근 CLIP 기반의 OOD 검출 기법은 세 가지 제안된 벤치마크에서 다양한 정도의 어려움에 직면하며, 어떤 방법도 일관되게 최고 성능을 내지 못했습니다. 특히, 공변량 변화가 도입되었을 때 CLIP 기반 OOD 검출 성능은 감소하는 경향이 있었고, 이는 OOD 검출 성능 개선의 필요성을 강조합니다. 이러한 결과들은 이상적인 벤치마크 환경을 넘어 현실 세계에 더 부합하는 상황을 포함해야 함을 보여줍니다.



### Transfer Learning for Keypoint Detection in Low-Resolution Thermal TUG Test Images (https://arxiv.org/abs/2501.18453)
Comments:
          Accepted to AICAS 2025. This is the preprint version

- **What's New**: 본 연구는 저해상도 열 화상 이미지에서 사람의 키포인트(keypoint)를 감지하기 위한 새로운 접근 방식을 제시합니다. 기존의 Mask R-CNN과 ViTPose-Base 모델을 뛰어넘는 성능을 달성하며, 저해상도를 효율적으로 처리할 수 있는 전이 학습(transfer learning) 기법이 활용됩니다. 특히, TUG(Timed Up and Go) 테스트를 열 화상 컴퓨터 비전 분야에 최초로 적용하여 새로운 이동성 평가 기준을 수립하였습니다.

- **Technical Details**: 모델은 MobileNetV3-Small과 ViTPose 디코더를 조합하여 구성되며, 복합 손실 함수(composite loss function)를 통해 학습됩니다. 이 손실 함수는 잠재 표현의 정렬과 히트맵(heatmap)의 정확성을 동시에 최적화합니다. 본 연구는 실제 환경을 모사하는 다양한 시나리오에서의 키포인트 식별 성능을 평가하기 위해 TUG 테스트를 위한 저해상도 열 화상 및 RGB 이미지를 학습 샘플로 사용하였습니다.

- **Performance Highlights**: 제안된 모델은 COCO 키포인트 감지 챌린지의 OKS(Object Keypoint Similarity) 측정 기준으로 평가되었으며, 각각 0.861, 0.942, 0.887의 AP, AP50 및 AP75 점수를 기록하여 전통적인 지도 학습 접근 방식을 초월하는 성능을 보였습니다. 또한, 모델의 파라미터 수와 FLOPS에서 뛰어난 계산 효율성을 입증하여 향후 임상 적용 가능성을 더욱 확장할 수 있는 기반을 마련하였습니다.



### Adaptive Object Detection for Indoor Navigation Assistance: A Performance Evaluation of Real-Time Algorithms (https://arxiv.org/abs/2501.18444)
Comments:
          5 pages, 2 figures, 3 tables

- **What's New**: 이번 연구는 시각 장애인을 위한 보조 기술에서 정확하고 효율적인 객체 탐지(object detection)의 필요성을 다루고 있습니다. YOLO, SSD, Faster R-CNN, Mask R-CNN이라는 네 가지 실시간 알고리즘을 실내 내비게이션 지원의 맥락에서 평가합니다.

- **Technical Details**: Indoor Objects Detection 데이터셋을 사용하여 탐지 정확도(detection accuracy), 처리 속도(processing speed), 실내 환경에 대한 적응성(adaptability) 등을 분석합니다. 연구 결과는 정밀도와 효율성 간의 trade-off를 강조하며, 실시간 보조 내비게이션에 최적의 알고리즘을 선택하는 데 필요한 통찰력을 제공합니다.

- **Performance Highlights**: 이 연구는 적응형 머신러닝 응용 프로그램을 한 단계 발전시키며, 시각 장애인을 위한 실내 내비게이션 솔루션을 개선하고 접근성을 촉진합니다.



### SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer (https://arxiv.org/abs/2501.18427)
- **What's New**: 이번 논문에서는 SANA-1.5라는 새로운 linear Diffusion Transformer 모델을 소개하여, 텍스트-이미지 생성의 효율적인 확장을 구현했습니다. SANA-1.0을 바탕으로, (1) 효율적인 훈련 확장, (2) 모델 깊이 가지치기, (3) 추론 시간 확장이라는 세 가지 혁신을 도입하였습니다. 이러한 전략들을 통해 SANA-1.5는 GenEval에서 텍스트-이미지 정렬 점수 0.72를 달성하였으며, 이는 추론 시간 확장을 통해 0.80으로 개선되어 새로운 SoTA(state of the art)를 기록하였습니다.

- **Technical Details**: SANA-1.5는 모델 성장 전략, 모델 깊이 가지치기 및 추론 확장을 통해 효율적인 모델 확장을 추구합니다. 첫째로, 최대 4.8B 매개변수까지 확장할 수 있도록 설계된 효율적인 모델 성장 전략을 제시하였습니다. 둘째로, 중요 블록 분석 기법을 통해 덜 중요한 블록을 제거함으로써 모델 압축을 구현하였으며, 마지막으로 여러 샘플 생성과 VLM 기반 선택 메커니즘을 활용하여 작은 모델이 큰 모델의 품질을 따라갈 수 있도록 하는 추론 기간 확장을 도입하였습니다.

- **Performance Highlights**: SANA-1.5는 전통적인 방법보다 2.5배 빠른 훈련 수렴 속도를 보였으며, GenEval 점수를 0.66에서 0.72로 향상시켰고, 추론 스케일링을 통해 0.80으로 추가 개선하였습니다. 이 모델은 대규모 모델의 훈련과 미세 조정을 보다 접근 가능하게 만들어 개방형 커뮤니티와 실용적인 응용에 기여할 수 있는 가능성을 보여줍니다. 따라서 비단 모델 크기를 늘리는 것만으로는 더 나은 품질을 보장할 수 없음을 시사하며, 효율적인 확장이 더 나은 최적화를 통해 실현될 수 있음을 강조합니다.



### Efficient Transformer for High Resolution Image Motion Deblurring (https://arxiv.org/abs/2501.18403)
Comments:
          14 pages, 18 figures Submitted as a preprint, no prior journal/conference submission

- **What's New**: 본 연구는 고해상도 이미지 모션 디블러링을 위한 Restormer 아키텍처의 포괄적인 연구 및 개선을 제시합니다. 모델 복잡성을 18.4% 줄이면서 최적화된 attention 메커니즘을 통해 성능을 유지하거나 개선하는 아키텍처 수정 사항을 도입하였습니다. 새로운 훈련 파이프라인에는 색상 지터, 가우시안 블러 및 원근 변환과 같은 추가 변환이 포함되어 있으므로 모델의 강인성이 향상됩니다.

- **Technical Details**: Restormer는 고해상도 이미지 복원을 위해 특별히 설계된 효율적인 Transformer 모델입니다. 이 모델은 멀티-Dconv 헤드 전이 주의 메커니즘(multi-Dconv head transposed attention mechanism)과 게이트드 Dconv 피드 포워드 네트워크(gated-Dconv feed-forward network)를 도입하여 계산 효율성을 유지하면서 장거리 의존성을 모델링합니다. Restormer는 전반적인 이미지 관계를 모델링하면서도 큰 공간 해상도에 대해 계산 비용을 피할 수 있도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 RealBlur-R, RealBlur-J 및 Ultra-High-Definition Motion blurred (UHDM) 데이터셋에서 특성 평가를 실시하였으며, 개선된 아키텍처는 훈련 시간 단축과 더불어 경쟁력 있는 성능을 유지했습니다. 본 논문의 결과는 심층적인 ablation 연구와 함께 아키텍처 단순화 및 개선된 훈련 전략이 모션 디블러링 작업에 효율적이고 동등한 능력을 지닌 모델을 만들어낼 수 있음을 제시합니다.



### MatIR: A Hybrid Mamba-Transformer Image Restoration Mod (https://arxiv.org/abs/2501.18401)
Comments:
          arXiv admin note: text overlap with arXiv:2402.15648 by other authors

- **What's New**: 최근 이미지 복원 분야에서는 Transformers 기반 모델들이 복잡한 맥락적 기능을 포착하는 능력을 활용하여 상당한 성장을 이루었습니다. Mamba 모델은 긴 거리 의존성을 처리하는 능력과 우수한 계산 효율성 덕분에 주목받고 있지만, 현재 Mamba는 Transformer에 비해 맥락 학습 능력에서 뒤처지고 있습니다. 이러한 문제를 해결하기 위해 Hybrid 모델인 MatIR을 제안합니다.

- **Technical Details**: MatIR 모델은 Transformer와 Mamba의 블록을 교차 순환하여 특징을 추출하는 구조로 설계되었습니다. Mamba 모듈에서는 이미지 인페인팅 상태 공간(Image Inpainting State Space, IRSS) 모듈을 도입해 네 가지 스캔 경로를 따라 긴 시퀀스 데이터를 효율적으로 처리합니다. Transformer 모듈에서는 삼각형 윈도우 기반의 지역 주의(local attention)와 채널 기반의 글로벌 주의(global attention)를 결합해 보다 광범위한 이미지 픽셀에 대한 주의 메커니즘을 활성화합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 광범위한 실험 결과와 변별 연구(ablation study)를 통해, MatIR이 기존의 다른 최첨단 방법들에 비해 우수한 성능을 발휘함을 입증하였습니다. MatIR은 지역적 및 글로벌 효과적인 수용 필드(effective receptive field)를 제공하며 효율적인 메모리 관리를 통해 이미지 복원 분야의 강력하고 유망한 솔루션으로 자리 잡았습니다.



### Cracks in concr (https://arxiv.org/abs/2501.18376)
Comments:
          This is a preprint of the chapter: T. Barisin, C. Jung, A. Nowacka, C. Redenbach, K. Schladitz: Cracks in concrete, published in Statistical Machine Learning for Engineering with Applications (LNCS), edited by J. Franke, A. Schöbel, reproduced with permission of Springer Nature Switzerland AG 2024. The final authenticated version is available online at: this https URL

- **What's New**: 이번 연구에서는 콘크리트 이미지에서 균열을 찾고 이를 정확히 분할(segmentation)하는 방법을 제시합니다. 기존의 3D 이미지 데이터가 부족한 문제를 해결하기 위해, 반-합성(semi-synthetic) 이미지 데이터를 생성하여 CNN(Convolutional Neural Networks)과 같은 네트워크를 훈련시키는 방식을 도입합니다. 새로운 네트워크인 RieszNet을 소개하며, 이는 변화하는 스케일(scale)에 불변하는 방식으로 설계되었습니다.

- **Technical Details**: 연구진은 균열 구조를 정의하는 기하학적 모델과 현실적인 CT 이미지에 균열을 삽입하는 방법을 논의합니다. U-Net과 3D U-Net 아키텍처를 활용하여 균열을 분할하는 데 필요한 방법론을 제안합니다. 이 과정에서 클래스 불균형(class imbalance)을 해결하기 위해 균열 픽셀에 더 높은 가중치를 부여하는 전략을 채택합니다.

- **Performance Highlights**: 제안된 방식은 수고로운 수동 주석(annotation) 작업 없이도 효과적인 균열 분할을 가능하게 합니다. 또한, CT 이미지를 통해 균열의 형태와 분포에 대한 더 많은 정보를 제공하며, 섬세한 균열 구조에 대해 높은 정확도의 예측을 달성했습니다. 최종적으로, 이 연구는 다양한 콘크리트 유형에 대해 일반화된 ML 균열 분할 방법을 논의합니다.



### Video-based Surgical Tool-tip and Keypoint Tracking using Multi-frame Context-driven Deep Learning Models (https://arxiv.org/abs/2501.18361)
- **What's New**: 본 연구는 로봇 수술 비디오에서 수술 도구의 주요 포인트를 자동으로 추적하는 새로운 multi-frame context-driven deep learning 프레임워크를 제안합니다. 지금까지 수술 도구의 키포인트 트래킹에 대한 연구는 상대적으로 부족했지만, 이 기술은 수술 도구의 이동 분석 및 기술 평가와 같은 여러 다운스트림 활용 사례에 기여할 것으로 기대됩니다. 실험 결과, 제안된 모델은 90%의 키포인트 검출 정확도와 5.27픽셀의 로컬라이제이션 RMS 오류를 달성했습니다.

- **Technical Details**: 제안된 방법은 두 단계로 이루어져 있으며, 첫 번째 단계에서 각 도구의 키포인트 주위의 작은 영역을 세분화하는 작업을 수행합니다. 이 과정은 다중 클래스 세그멘테이션 문제로 모델링되며, 키포인트 지역을 정의한 후 연속적으로 세분화된 영역을 추정하여 키포인트 좌표를 결정합니다. 특히, optical flow(광학 흐름)와 monocular depth(단안 깊이) 예측을 사용하여 다중 프레임 맥락을 고려한 키포인트 분할을 통해 모델의 정밀도를 높입니다.

- **Performance Highlights**: JIGSAWS 데이터셋의 더 어려운 시나리오에서도 제안된 multi-frame 모델은 도구 끝 및 도구 베이스 키포인트를 정확하게 추적하여 전반적으로 4.2픽셀 이하의 RMS 오류를 기록했습니다. 이러한 성과는 수술 도구의 키포인트를 정확하게 추적할 수 있는 기반을 마련하며, 이는 이후의 비디오 분석 및 의료 기술 평가에 대한 활용 가능성을 확장할 수 있음을 시사합니다.



### CodeBrain: Impute Any Brain MRI via Instance-specific Scalar-quantized Codes (https://arxiv.org/abs/2501.18328)
- **What's New**: 이번 연구에서는 MRI (Magnetic Resonance Imaging) 임프테이션(imputation)을 위한 새로운 통합 모델 CodeBrain을 제안합니다. 이 모델은 다양한 뇌 MRI 임프테이션 시나리오에 적응할 수 있도록 설계되었습니다. CodeBrain은 다양한 모달리티 변환을 전체 모달리티 코드 예측 작업으로 변환하며, 이를 위해 두 가지 단계로 학습됩니다.

- **Technical Details**: CodeBrain은 두 단계로 구성된 훈련 과정을 사용합니다. 첫 번째 단계는 각 MRI 모달리티를 재구성하여 공유 잠재 공간으로 매핑하고, 이를 스칼라 양자화(scalar quantization) 처리하는 것입니다. 두 번째 단계에서는 사용자 정의 그레이딩 손실(customized grading loss)을 통해 무작위로 마스킹된 MRI 샘플로부터 전체 모달리티 코드를 예측합니다. 이를 통해 데이터의 고유한 특성을 보존하면서 다양한 무결점 모달리티 변환을 수행할 수 있습니다.

- **Performance Highlights**: CodeBrain 모델은 IXI 및 BraTS 2023와 같은 두 개의 공개 뇌 MRI 데이터세트에서 평가되었습니다. 실험 결과, CodeBrain은 기존 네 가지 방법보다 우수한 임프테이션 성능을 보여주며, 통합 뇌 MRI 임프테이션을 위한 새로운 최첨단 성능을 설정했습니다. 이러한 결과는 다양한 임프테이션 시나리오에 적합한 모델을 제공합니다.



### A Video-grounded Dialogue Dataset and Metric for Event-driven Activities (https://arxiv.org/abs/2501.18324)
Comments:
          Accepted at AAAI2025

- **What's New**: 이번 논문에서는 사건 주도 활동에 대한 비디오 기반 대화 데이터셋인 VDAct를 소개합니다. VDAct는 3,000개의 대화와 30,000개 이상의 질문-답변 쌍으로 구성되어 있으며, 다양하고 복잡한 비디오 시퀀스를 포함합니다. 게다가 VDAct는 새로운 평가 metric인 VDEval을 제안하여 단일 대화 턴의 맥락에만 의존하지 않고, 대화 세션의 전반적인 맥락을 평가합니다.

- **Technical Details**: VDAct는 다양한 사건-주도 활동이 포함된 긴 비디오 시퀀스를 활용하며, 이에 대한 질문은 설명적 질문, 시간적 질문, 설명적 질문, 정량적 질문 등 여러 범주로 나뉩니다. 데이터셋은 주제와 행동 간의 관계를 연결하는 구조적 정보를 제공하는 지식 그래프(Knowledge Graphs, KGs)로 보강됩니다. 한편, VDEval은 대화 기록과 KGs를 통합해 생성된 응답의 평가를 수행하는 새로운 metric입니다.

- **Performance Highlights**: 기존의 영상 기반 대화 시스템과 비교할 때, VDAct에서 상태 기반 비전 모델들은 복잡한 질문 유형에 대응하는 데 한계를 보였습니다. 새로운 평가 metric인 VDEval은 인간 평가와 높은 상관관계를 보여 VDAct 데이터셋의 효과적인 평가를 가능하게 합니다. 따라서 이 연구는 비디오 기반 대화 시스템의 발전에 기여할 수 있는 중요한 기초 자료로 자리잡을 것입니다.



### Surface Defect Identification using Bayesian Filtering on a 3D Mesh (https://arxiv.org/abs/2501.18315)
Comments:
          Presented at IMEKO2024 World Congress, Hamburg, Germany, 26-29 October 2024

- **What's New**: 이 논문은 CAD 모델을 기반으로 한 자동 표면 결함 감지 방안을 제시합니다. CAD 모델에 내장된 사전 지식을 활용하고, 상용 스테레오 카메라 및 심도 카메라로 수집한 포인트 클라우드 데이터와 통합했습니다. 제안된 방법은 CAD 모델을 고밀도 폴리곤 메쉬로 변환한 후, 가중 최소 제곱 알고리즘을 이용하여 스캔한 작업물의 상태를 추정합니다.

- **Technical Details**: 이 알고리즘은 스테레오 카메라로 획득한 포인트 클라우드 데이터를 CAD 모델과 일치시키는 것을 목표로 하고 있습니다. STL (Stereolithography) 3D 모델을 기준으로 하여 각 정점은 3D 공간에서 상태 변수로 나타나며, 이러한 정보를 결합하여 Bayes 필터링 기법을 적용합니다. 알고리즘은 약 50개의 포인트 클라우드 샘플만으로도 관심 지역에서 서브 밀리미터 수준의 표준 편차로 수렴하는 성능을 나타냅니다.

- **Performance Highlights**: 초기 결과는 스테레오 카메라를 이용한 고정밀 품질 관리 응용에 대한 가능성을 보여줍니다. 이 알고리즘은 기존의 결함 탐지에 비해 높은 정확도를 제시하며, 기존의 인력 의존적인 방법보다 효율적인 자동화 솔루션으로 자리잡을 잠재력이 있습니다. 산업 4.0 시대의 데이터 순환 종속성을 고려할 때, 이 접근 방식은 생산 공정 개선에 중요한 역할을 할 수 있습니다.



### Simulation of microstructures and machine learning (https://arxiv.org/abs/2501.18313)
Comments:
          Preprint of: K. Schladitz, C. Redenbach, T. Barisin, C. Jung, N. Jeziorski, L. Bosnar, J. Fulir, P. Gospodnetić: Simulation of Microstructures and Machine Learning, published in Continuum Models and Discrete Systems by F. Willot, J. Dirrenberger, S. Forest, D. Jeulin, A.V. Cherkaev (eds), 2024, Springer Cham. The final version is this https URL

- **What's New**: 이 논문에서는 머신 러닝(Machine Learning)의 유망한 가능성을 통해 이미지 처리 이미지를 할 수 있는 새로운 솔루션을 제시합니다. 기존의 알고리즘 개발 및 파라미터 조정의 작업을 제거하고, 합성 이미지(Synthetic Images)와 같은 새로운 데이터 생성 방법을 통해 훈련된 CNN(Convolutional Neural Network)과 랜덤 포레스트(Random Forest)를 활용하여 더 나은 일반화(generalization)를 목표로 합니다. 특히, 훈련 데이터의 부족이 주요 병목 현상으로 지적되고 있으며, 이를 해결하기 위한 다양한 사례를 다룹니다.

- **Technical Details**: 제안된 기술적 방법론 중 하나는 확률 기하학 모델(Stochastic Geometry Models)을 기반으로 한 합성 이미지생성입니다. 이 접근 방식은 구조의 다양성을 포착하는데 유용하며, 생성된 합성 이미지는 실제 데이터에 대한 정확한 지도를 제공하여 머신 러닝 모델 훈련 시 유용합니다. 여기서 다양한 모델이 사용되며, Riesz 변환(Riesz Transforms)을 기반으로 한 새로운 스케일 불변 신경망(Scale-Invariant Neural Network)도 소개되어, 훈련에 필요한 매개변수 수를 대폭 줄여줍니다.

- **Performance Highlights**: 논문에서 다루어진 두 가지 주요 사용 사례는 콘크리트 이미지에서 균열 구조(segmenting crack structures) 및 산업 생산에서의 광학 품질 검사(optical quality control)에 대한 것입니다. 콘크리트 이미지는 연결된 단면을 스캔하여 3D 이미지를 생성하며, 동적 데이터를 관리하는 데 있어 도전적입니다. 이를 통해 실시간 처리 및 정확한 결정을 내리는 능력을 향상시키는 다양한 문제와 기회가 제기됩니다.



### A Comprehensive Analysis on Machine Learning based Methods for Lung Cancer Level Classification (https://arxiv.org/abs/2501.18294)
- **What's New**: 본 논문은 폐암 진단을 위한 머신러닝 (ML) 기법의 실질적 활용 가능성을 탐구합니다. 초기 진단의 안정성을 높이고, 모델 성능의 과적합 문제를 해결하기 위한 체계적인 분석이 진행됩니다. 다양한 머신러닝 모델이 비교되며, 목표를 더 정확히 파악할 수 있는 방법이 제시됩니다.

- **Technical Details**: 폐암의 다양한 단계 분류를 위해 XGBoost (XGB), LGBM, Adaboost, Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), CatBoost, k-Nearest Neighbor (k-NN) 등의 ML 모델을 체계적으로 실행하였습니다. 최소 자식 가중치(minimum child weight)와 학습률(learning rate)의 영향을 고려하며, 딥 신경망 (DNN) 모델을 통해 특성과 타겟 간의 상관관계를 분석합니다. 이를 통해 복잡한 패턴 식별에 대한 모델의 능력이 확립됩니다.

- **Performance Highlights**: 연구 결과, 여러 ML 모델이 폐암 단계 분류에서 높은 정확도를 달성할 수 있다는 주장이 제기됩니다. 특히, DNN 아키텍처의 복잡성에도 불구하고, 전통적인 ML 모델인 XGBoost, LGBM, Logistic Regression이 뛰어난 성능을 발휘하였습니다. 정확도, 정밀도(precision), 재현율(recall), F-1 점수(F-1 score) 등 다양한 비교 메트릭에서 뛰어난 예측 성능을 보이고 있습니다.



### MAMS: Model-Agnostic Module Selection Framework for Video Captioning (https://arxiv.org/abs/2501.18269)
Comments:
          Accepted to the AAAI 2025 Main Technical Track. This is an extended version of the original submission

- **What's New**: 이 논문에서는 동영상 캡셔닝에서 적절한 프레임 수를 선택하는 모듈 선택 프레임워크, 즉 Model-Agnostic Module Selection (MAMS) 프레임워크를 제안합니다. 이 프레임워크는 각 비디오에 맞는 캡션 생성 모듈을 선택하고, 중요 시각 토큰의 서브셋을 구성하여 캡션 성능을 향상시킵니다. 또한, 중요한 시각 토큰에 대한 주의를 높이는 새로운 적응형 어텐션 마스킹 기법도 도입하였습니다.

- **Technical Details**: MAMS 프레임워크는 세 가지 주요 모듈로 구성됩니다: 비디오 인코더, 텍스트 인코더, 및 캡션 생성 모듈입니다. 이 프레임워크는 특정 비디오에 적합한 크기의 캡션 생성 모듈을 선택하여 서로 다른 수의 프레임을 사용합니다. 이 과정은 주어진 비디오에서 시각 토큰을 추출하고, 선택된 모듈에 따라 필요에 맞는 시각 토큰의 서브셋을 구성함으로써 수행됩니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에서의 실험 결과, 제안한 MAMS 프레임워크는 SwinBERT, UniVL 및 mPLUG-2와 같은 최신 동영상 캡셔닝 모델의 성능을 크게 향상시켰습니다. 특히, mPLUG-2에 MAMS 프레임워크를 적용함으로써 새로운 최상위 성과 기준을 달성하였습니다. 해당 연구는 매우 동적인 비디오와 비슷한 정보가 적은 비디오를 모두 아우르는 적응형 접근법으로, 기존 방법의 한계를 효과적으로 극복합니다.



### Ground Awareness in Deep Learning for Large Outdoor Point Cloud Segmentation (https://arxiv.org/abs/2501.18246)
Comments:
          This paper has been accepted for presentation at the GRAPP 2025 conference

- **What's New**: 이 논문은 원거리 감지에서의 기존 머신 러닝 네트워크를 통해 야외 포인트 클라우드(Outdoor Point Cloud)의 의미 세분화(Semantic Segmentation)를 지원하기 위한 고도 데이터 활용 분석을 제시합니다. Dense outdoor point clouds를 다룰 때, 머신러닝 모델의 수용 영역(Receptive Field)이 작을 수 있어 주위와 맥락을 정확히 파악하기 어려운 문제를 해결하기 위해 Digital Terrain Model(DTM)을 계산하여 상대 고도 특성을 추출합니다. RandLA-Net을 이용해 대규모 포인트 클라우드의 효율적인 의미 세분화를 수행하며, 다양한 센서 기술과 위치에서 수집한 세 가지 야외 데이터셋에서 성능을 평가합니다.

- **Technical Details**: 딥러닝 기법의 발전으로 인해 야외 포인트 클라우드의 의미 세분화 모델이 널리 사용되고 있으나, 매우 크고 밀집된 포인트 클라우드에 적용 시 어려움이 있을 수 있습니다. 이러한 상황에서 DTM을 사용하여 점과 가장 가까운 지형 점 사이의 수직 거리인 상대 고도를 추가적인 특성으로 통합하여 수용 영역을 확대할 수 있습니다. 연구에서는 RandLA-Net이 상대 고도를 포함한 다양한 포인트 기반 2D 및 3D 로컬 특성과 함께 확장되어 의미 세분화 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 세 가지 다양한 데이터셋에서 상대 고도 데이터 통합이 일관된 성능 향상을 이끈다는 것을 확인했습니다. 특히, Hessigheim 데이터셋에서 평균 F1 점수가 72.35%에서 76.01%로 3.7% 포인트 증가하는 결과를 보였으며, 지면과 객체 간의 장기 의존성을 설정하는 데 기여했습니다. 추가적인 로컬 특성들인 평면도(planarity), 법선 벡터(normal vectors), 2D 특징 등의 효과는 데이터셋의 특성에 따라 다르게 나타났습니다.



### Arbitrary Data as Images: Fusion of Patient Data Across Modalities and Irregular Intervals with Vision Transformers (https://arxiv.org/abs/2501.18237)
- **What's New**: 이번 연구에서 우리는 다중 모달 데이터를 통합하는 새로운 방법을 제안합니다. 이 방법은 다양한 생체 신호 및 처방된 약물 정보를 이미지 형태로 변환하여 Vision Transformer를 학습시키는 것입니다. 기존 연구들과는 달리, 이 접근법은 다중 모달을 단일 모델로 다룰 수 있도록 복잡도를 크게 감소시킵니다. 이를 통해 다중 모달 의료 AI의 진전을 이끌어 내길 기대합니다.

- **Technical Details**: 제안한 모델인 Vision Transformer for irregular sampled Multi-modal Measurements (ViTiMM)은 임상 매개변수, 약물 데이터, ECG 스캔 및 X선 사진 등을 이미지로 표현합니다. 이 과정에서 'visual prompt engineering'을 통해 여러 모달리티를 통합하여 모델링을 간소화합니다. 기존의 방식에서 각 모달리티는 개별적으로 최적화된 모델 아키텍처가 필요하였지만, 우리는 쉽게 데이터 처리를 통합할 수 있는 솔루션을 제공하고 있습니다.

- **Performance Highlights**: 우리의 연구 결과는 MIMIC-IV 데이터셋에서 6,175명의 환자를 분석한 결과 두 가지 벤치마크 과제에서 기존의 최첨단 방법들, MeTra와 MedFuse를 초월하는 성능을 보여주었습니다. 더욱이 이 모델은 해석 가능성을 높여주는 attention map을 시각화할 수 있어, 예측 결과에서 가장 영향력이 있는 입력 영역을 식별할 수 있습니다. 또한 연구 결과는 다중 모달 정보를 통합하여 모델 성능이 향상됨을 명확히 보여주고 있습니다.



### Free-T2M: Frequency Enhanced Text-to-Motion Diffusion Model With Consistency Loss (https://arxiv.org/abs/2501.18232)
- **What's New**: 이번 연구에서는 텍스트-모션 생성(text-to-motion generation)에서 주로 다루어지던 시간적 모델링(temporal modeling)의 한계를 점검하고, 주파수 영역(frequency-domain) 분석을 통합하여 모델의 성능을 향상시킬 새로운 접근 방식을 제시합니다. 제안된 **Free-T2M** 모델은 **semantic planning 단계**와 **fine-grained improving 단계**로 나뉘는 모션 디노이징의 두 가지 주요 창을 구별하고, 각 단계에 맞는 일관성 손실(consistency loss)을 도입합니다. 이러한 방법을 통해 정적 특징의 강건성을 높이고, 정밀도를 개선함으로써 새로운 SOTA 성능을 달성하였습니다.

- **Technical Details**: 연구는 모션 데이터의 디노이징 메커니즘을 주파수 영역(frequency-domain) 관점에서 분석하고, 이를 통해 보다 해석 가능한 솔루션을 제공합니다. 이 모델은 낮은 주파수의 구성 요소(low frequency components)에 집중하여 더욱 안정적이고 정밀한 모션 생성을 도모합니다. 또한, 고주파 정보(high-frequency information)의 과도한 강조가의 학습 불안정성을 초래할 수 있음을 지적하고, 이를 피하기 위한 두 가지 손실 함수를 제안합니다.

- **Performance Highlights**: 실험 결과, **Free-T2M** 모델은 StableMoFusion에서 FID 지수를 0.189에서 0.051로 줄이며, 텍스트-모션 생성 분야에서 새로운 최첨단 성능을 기록하였습니다. 이러한 결과는 주파수 영역 통찰(frequency-domain insights)이 텍스트-모션 생성 과정에 통합됨으로써 더 정확하고 안정적인 결과를 도출할 수 있음을 강조합니다. 모델의 복잡성을 증가시키지 않으면서도 최상의 성능을 달성하여 실질적인 응용에 기여할 수 있는 기반을 마련하였습니다.



### Machine Learning Fairness for Depression Detection using EEG Data (https://arxiv.org/abs/2501.18192)
Comments:
          To appear as part of the International Symposium on Biomedical Imaging (ISBI) 2025 proceedings

- **What's New**: 이 논문은 EEG (electroencephalogram) 데이터를 활용하여 우울증 탐지에서 머신러닝의 공정성을 평가하기 위한 최초의 시도를 제시합니다. 여러 딥러닝 아키텍처, 즉 CNN (Convolutional Neural Networks), LSTM (Long Short-Term Memory) 네트워크, GRU (Gated Recurrent Unit) 네트워크를 사용하여 실험을 수행하였습니다. 세 가지 EEG 데이터셋인 Mumtaz, MODMA, Rest를 대상으로 공정성을 위한 다양한 평가 기준을 사용했습니다.

- **Technical Details**: 다양한 편향 완화 (bias mitigation) 전략을 사전 처리, 처리 중, 사후 처리 단계에서 적용하여 그 효과성을 평가하였습니다. 실험 결과, 기존 EEG 데이터셋과 우울증 탐지를 위한 알고리즘에서 편향이 존재함을 발견하였고, 여러 편향 완화 방법이 서로 다른 공정성 기준에 따라 편향을 해결하는 방식이 다름을 보여주었습니다. 이러한 접근은 EEG 데이터에서 머신러닝의 공정성을 높이는 데 기여할 수 있습니다.

- **Performance Highlights**: 우울증 탐지에 대한 기존 알고리즘의 편향 문제를 강조하며, 다양한 편향 완화 기법이 필요하다는 것을 입증하였습니다. 연구 결과는 각기 다른 레벨에서 편향 문제를 다루는 다양한 방법들을 제시합니다. 따라서 이 연구는 EEG 데이터를 기반으로 한 우울증 탐지의 공정성과 신뢰성을 향상시키기 위한 중요한 기초 자료가 될 것입니다.



### IROAM: Improving Roadside Monocular 3D Object Detection Learning from Autonomous Vehicle Data Domain (https://arxiv.org/abs/2501.18162)
Comments:
          7 pages, 5 figures, ICRA2025

- **What's New**: 이번 논문에서는 IROAM이라는 새로운 대비 학습 프레임워크를 제안하여, 차량측 데이터와 도로측 데이터를 동시에 입력받아 도로측 단안 3D 객체 검출 성능을 향상시키고자 합니다. 기존의 단안 검출 방법들은 도로측 카메라 및 차량측 카메라의 시점 차이로 인해 성능이 저하되는데, IROAM은 이를 해결하기 위해 데이터의 의미적 및 기하학적 부분을 분리하여 학습합니다. 이는 도로측 3D 객체 탐지의 정확도를 높이는 데 기여합니다.

- **Technical Details**: IROAM은 세 가지 주요 모듈로 구성되어 있습니다: Feature Encoder, In-Domain Query Interaction, Cross-Domain Query Enhancement입니다. Feature Encoder는 입력 이미지로부터 내용 특성(content features)과 깊이 특성(depth features)을 추출합니다. In-Domain Query Interaction 모듈에서는 객체 쿼리를 초기화하고 업데이트하여 각 도메인 내 정보를 학습하며, Cross-Domain Query Enhancement에서는 쿼리를 의미적 및 기하학적 부분으로 분리하여 의미적 부분만을 대조 학습에 사용합니다.

- **Performance Highlights**: 실험 결과, IROAM은 도로측 탐지기의 성능을 향상시키는 데 효과적임이 입증되었습니다. 차량측 데이터와 도로측 데이터 간의 학습을 통해 두 도메인 간의 정보 전달이 가능해졌으며, 이는 모델의 성능 향상으로 이어졌습니다. 특히, 이 방법은 다양한 차량측 및 도로측 데이터 세트에서 우수한 성능을 나타내었습니다.



### REMOTE: Real-time Ego-motion Tracking for Various Endoscopes via Multimodal Visual Feature Learning (https://arxiv.org/abs/2501.18124)
- **What's New**: 본 논문에서는 내시경을 위한 실시간 자아 모션 추적(real-time ego-motion tracking)에 대한 새로운 프레임워크를 제안합니다. 이 프레임워크는 다중 모달 비주얼 피쳐 학습 네트워크(multi-modal visual feature learning network)를 통해 상대적 자세 추정(relative pose prediction)을 수행합니다. 또한, 주목 메커니즘(attention mechanism)을 기반으로 한 새로운 피쳐 추출기가 설계되어 두 개의 연속적인 영상에서 수집된 다차원 정보를 통합할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 상대 자세 추정과 절대 자세 계산의 두 단계로 나누어집니다. 첫 단계에서는 현재 관찰값과 이전 프레임을 기반으로 상대 자세 변환을 예측하여, 여러 상대 자세 변환을 통해 내시경의 절대 자세를 계산합니다. 기존의 FastFlowNet을 활용하여 두 개의 인접한 관찰 간의 광학 흐름(optical flow)을 예측하며, 이를 통해 실시간 추적을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 세 가지 엔도스코피 데이터셋에서 최첨단 기법들보다 더 높은 정확도를 보였으며, 초당 30프레임 이상의 추론 속도(inference speed)를 기록하여 실시간 요건을 충족하였습니다. 이는 내시경의 안전하고 정확한 내비게이션을 위해 중요한 기여를 하게 될 것입니다.



### DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification (https://arxiv.org/abs/2501.18116)
Comments:
          27 pages, 8 figures

- **What's New**: 이번 연구에서는 기존의 기능 데이터 분석(FDA) 방법의 비효율성을 해결하기 위해 DeepFRC라는 새로운 심층 학습 프레임워크를 제안합니다. 이 프레임워크는 기능 등록(functional registration)과 분류(classification) 문제를 하나의 통합 모델로 결합하여 동시에 처리합니다. 또한, 시간 왜곡 함수(time warping functions)를 학습하는 정렬 모듈(alignment module)과 정렬된 데이터에 대한 차원 축소를 위한 학습 가능한 기저 표현 모듈(learnable basis representation module)을 통합하여 성능을 향상시킵니다.

- **Technical Details**: DeepFRC는 전반적인 프로세스를 자동화하여 단계적으로 처리하는 대신, 등록과 분류를 simultaneously(동시) 수행합니다. 모델링에서 발생하는 misalignment(불일치)와 generalization error(일반화 오차)에 대한 이론적 분석을 통해 낮은 misalignment을 보장합니다. 이 모델은 elastic functional registration을 사용하여 시간 왜곡을 구현하며, 이를 통해 데이터의 구조적 변화를 효과적으로 반영합니다.

- **Performance Highlights**: 실험 결과, DeepFRC는 다양한 실제 데이터 세트에 대해 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다. 특히, 복잡한 등록 문제를 다루는 데 뛰어난 능력을 갖추고 있으며, 모델이 대칭적이고 누락된 데이터에 대해서도 견고함을 유지합니다. 이 연구는 실제 응용 분야에서의 DeepFRC의 적용 가능성을 시사합니다.



### Disentangling Safe and Unsafe Corruptions via Anisotropy and Locality (https://arxiv.org/abs/2501.18098)
- **What's New**: 본 논문은 	exttt{Projected Displacement} (PD)라는 새로운 위협 모델을 제안하여 기존의 이소트로픽(isotropic) 및 글로벌(global) 위협 모델 이상의 강건성을 연구합니다. 기존의 대부분 위협 모델은 단순히 $	exttt{l_p}$ norm을 사용하여 위협을 정의했으나, 이는 컴퓨터 비전에서의 흔한 왜곡 현상인 블러, 압축 및 가림 등을 잘 포착하지 못합니다. PD 모델은 입력 공간에서 불안전한 방향(unsafe directions)을 정의하고, 이를 통해 입력에 따른 위협을 측정함으로써 비안소트로픽(anisotropic) 및 지역적(local) 특성을 보여줍니다.

- **Technical Details**: PD 위협 모델은 단순히 작은 크기의 변형이 아닌, 특정 입력을 위한 안전한 방향과 불안전한 방향을 고려하여 위협을 평가합니다. 여기서 안전한 변형(safe perturbations)은 실제 레이블을 보존하는 반면, 불안전한 변형(unsafe perturbations)은 레이블을 변경하는 방식으로 정의됩니다. 이러한 평가 방식은 임의의 분류 작업에 대해 사전 훈련(pre-training)이나 미세 조정(fine-tuning) 없이도 쉽게 계산될 수 있습니다.

- **Performance Highlights**: 실험 결과, Imagenet-1k 데이터셋을 통해 PD 위협 모델이 기존 모델과 다르게 작동하여, 실제 레이블을 유지하는 고유의 안전한 변형 집합을 포함하고 있음을 보여줍니다. 이 모델은 또한 이미지의 특정 영역에 대한 민감도나 개념 계층과 같은 추가적인 작업 주석을 쉽게 통합할 수 있어 적용 가능성이 높습니다. PD 모델은 기계 학습 시스템의 안전성 평가에 있어 유연하고 작업 중심의 위협 사양을 제공합니다.



### LLMs can see and hear without any training (https://arxiv.org/abs/2501.18096)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 MILS(Multimodal Iterative LLM Solver)를 소개합니다. 이 방법은 훈련이 필요 없는 간단한 접근 방식으로, 기존의 LLM에 멀티모달 기능을 추가할 수 있도록 돕습니다. MILS는 다단계 추론(multi-step reasoning) 능력을 활용해 다양한 응용 프로그램을 가능하게 합니다.

- **Technical Details**: MILS는 후보 출력을 생성하고 각 출력을 점수화(scoring)한 후 피드백(feedback)을 통해 반복(iteratively)적으로 솔루션을 도출하는 방식입니다. 이를 통해, 훈련이 필요했던 전문 모델을 사용하지 않고 특정 작업에 대한 솔루션을 생성할 수 있습니다. 이 방법은 이미지를 캡션(captioning)하는 데 있어 새롭고 최고의 성과를 달성했습니다.

- **Performance Highlights**: MILS는 이미지 생성(text-to-image generation)과 같은 미디어 생성(media generation)에도 효과적으로 적용할 수 있으며, 스타일 전이(style transfer)를 위한 프롬프트 수정(prompt rewrites) 발견에도 기여합니다. 또한, 이 방법은 그래디언트(gradient)가 필요 없는 최적화 방식이기 때문에 멀티모달 임베딩(multimodal embeddings)을 텍스트로 변환하여 크로스모달 산술(cross-modal arithmetic) 응용 프로그램에도 사용할 수 있습니다.



### Generative AI for Vision: A Comprehensive Study of Frameworks and Applications (https://arxiv.org/abs/2501.18033)
Comments:
          53 pages, 18 figures

- **What's New**: 이 논문은 Generative AI가 이미지 합성의 변화를 이끌어내고 있다는 점에서 새로운 관점을 제공합니다. Generative AI는 고품질의 다양한 포토리얼리스틱 시각 자료를 산업 전반에 걸쳐 생성하는 것을 가능하게 합니다. 이미지 간 변환, 텍스트에서 이미지 생성, 도메인 전이 및 다중 모드 정렬 등 여러 기술의 발전으로 자동화된 시각 자료 생성의 범위가 확대되고 있습니다.

- **Technical Details**: 이 연구는 입력의 성격에 따라 이미지 생성 기술을 체계적으로 분류하였으며, 노이즈 벡터, 잠재 표현, 조건부 입력 등 다양한 입력 모달리티를 식별합니다. Generative Adversarial Networks (GANs), 조건부 프레임워크 및 Stable Diffusion과 같은 확산 기반 접근 방식을 포함하여 다양한 모델을 탐구합니다. 또한 DALL-E, ControlNet, DeepSeek Janus-Pro 등 주요 프레임워크를 소개하고, 계산 비용, 데이터 편향 및 사용자 의도와의 출력 정렬 문제를 다루었습니다.

- **Performance Highlights**: Generative AI는 다양한 산업에서 혁신적 변화를 선도하고 있습니다. E-commerce에서는 개인화된 제품 비주얼을 신속하게 생성해 고객 참여를 증진시키며, 과학 연구에서는 Stable Diffusion을 이용해 복잡한 현상을 고해상도로 묘사해 혁신을 가속화하고 있습니다. 또한, 자율 시스템에서는 현실적인 훈련 환경을 시뮬레이션하는 데 기여하여 자율주행 차량이나 로봇 시스템의 개발 비용을 줄이고 있습니다.



### Anatomy Might Be All You Need: Forecasting What to Do During Surgery (https://arxiv.org/abs/2501.18011)
- **What's New**: 이 논문은 외과 수술 도구의 움직임을 예측하는 새로운 형태의 가이드를 제안합니다. 기존의 수술 내비게이션(system) 및 수술 단계 인식에 이어, 수술 도구의 향후 경로를 예측하여 다음 단계에 대한 명확한 지침을 제공하고자 합니다. 최근의 연구 결과들을 바탕으로 수술 비디오에서 해부학적 특징과 도구의 위치를 분석하여, 도구의 다음 위치를 예측하는 모델을 개발하였습니다.

- **Technical Details**: 우리는 endoscopic video frames의 시퀀스를 나타내는 𝐒_{t}를 정의하고, 주 목표는 미래의 frames에서 수술 도구의 위치 변화를 예측하는 것입니다. 이를 위해, YOLO(You Only Look Once) 알고리즘과 같은 객체 감지 기법을 활용하여 수술 동영상의 프레임에서 해부학적 구조와 수술 도구를 인식하고, 이 정보를 바탕으로 도구의 미래 위치를 예측합니다. 모델은 간단한 신경망(neural network) 아키텍처를 사용하여 과거 64 프레임을 기반으로 다음 8개 또는 16개의 프레임 내에서 도구의 위치를 예측합니다.

- **Performance Highlights**: 실험 결과, 해부학적 구조 탐지가 수술 도구의 움직임 예측 문제를 해결하는 데 중요한 역할을 하는 것으로 나타났습니다. 이러한 접근 방식은 경험이 적은 외과 의사들에게도 도움이 될 수 있는 자율 수술 로봇의 통합된 관리 요소가 될 수 있습니다. 본 연구는 향후 수술 행동에 대한 정확한 가이드를 제공할 수 있는 모델 개발의 초석이 되기를 희망합니다.



### Pressure Field Reconstruction with SIREN: A Mesh-Free Approach for Image Velocimetry in Complex Noisy Environments (https://arxiv.org/abs/2501.17987)
- **What's New**: 이번 연구에서는 이미지 속도계 데이터를 활용한 압력 필드 재구성을 위해 SIREN(Sinusoidal Representation Network)을 도입하였으며, 이는 노이즈 환경에서의 효율성을 강조하고 메쉬가 없는 방식으로 압력 필드를 직접 재구성하는 방법이 됩니다. 기존의 OS-MODI와 GFI 방법들과 비교하여 SIREN은 메쉬 구조의 필요 없이 강력한 압력 재구성 솔루션을 제공합니다. 특히, SIREN의 아키텍처 변경을 통해 속도계 데이터의 내재된 노이즈를 필터링할 수 있다는 점에서 차별성을 가지며, 이는 새로운 적용 가능성을 제시합니다.

- **Technical Details**: SIREN 접근법은 전통적인 메쉬 기반 방법의 단점을 피할 수 있도록 설계되었으며, 메쉬가 아닌 방법으로 압력 필드를 재구성합니다. 압력의 물질적 기여도(물질 미분)와 경계 조건 평가의 중요성이 강조되어 있으며, 이러한 재구성에서 발생하는 오류 전파의 영향에 대한 연구가 필요하다는 점을 지적합니다. 또한, ODI(omni-directional integration) 및 GFI(Green's function integral) 방법이 각각 다루는 방식과 장단점에 대해 논의하며, 제안된 SIREN 방법이 이러한 문제를 해결하는 데 어떻게 기여하는지를 설명합니다.

- **Performance Highlights**: SIREN 방법은 특히 구조가 없는 메쉬 환경에서 압력 필드 재구성을 효과적으로 수행할 수 있으며, 이는 기존의 노이즈가 많은 데이터로부터 신뢰할 수 있는 압력 수치를 추출할 수 있도록 합니다. 기존 메쉬 기반 방법들과 비교할 때, SIREN은 메쉬 구조나 불규칙한 셀에서의 어려움을 피할 수 있어 더욱 유용합니다. 이로 인해 SIREN은 다양한 유체 역학적 시나리오에서 일관된 성능을 나타내며, 실험 데이터 분석에서 새로운 기회를 제공합니다.



### Efficient Feature Fusion for UAV Object Detection (https://arxiv.org/abs/2501.17983)
- **What's New**: 본 논문에서는 UAV(무인 항공기) 객체 탐지를 위해 설계된 새로운 기능 융합 프레임워크를 제안합니다. 이 프레임워크는 하이브리드 업샘플링(hybrid upsampling) 및 다운샘플링 모듈(downsampling modules)을 통합하여 다양한 네트워크 깊이의 기능 맵(feature maps)을 유연하게 조정할 수 있게 합니다. 특히, 소형 객체의 위치 정확도(localization accuracy)와 분류 성능(classification performance)을 강화하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 FMSA(융합 멀티헤드 셀프 어텐션) 모듈은 CNN 기반 네트워크의 보조 구성 요소로 작용합니다. 이 모듈은 FDS(융합 다운샘플링)와 FUS(융합 업샘플링) 두 가지 보조 모듈과 함께 다중 스케일 기능 융합을 수행하고, 장거리 크로스 레이어 연결(cross-layer connections)을 지원합니다. 이러한 구조적 진화를 통해 소형 객체의 탐지에서 중요한 정확도를 확보할 수 있습니다.

- **Performance Highlights**: 제안된 프레임워크는 YOLO-V10 모델에 통합되어 평균 정밀도(AP)가 2% 향상되었습니다. 이는 기존 YOLO-V10 모델 대비 데이터 처리의 효율성을 유지하면서도 성능 개선이 이루어졌음을 보여줍니다. 실험 결과는 제안된 방법이 UAV 객체 탐지에서 정확하고 효율적인 성과를 보여준다는 것을 입증합니다.



### VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting (https://arxiv.org/abs/2501.17978)
- **What's New**: 이번 논문에서는 3D Gaussian Splatting 모델을 확장하여 대칭 행렬을 추가함으로써 각 3D Gaussian의 불투명도 표현을 강화하였습니다. 이 개선을 통해, 관찰자의 시점에 따라 특정 Gaussian을 억제할 수 있게 되어, 보다 장면에 대한 정밀한 표현이 가능해졌습니다. 이는 시각적 반사와 스페큘러 하이라이트를 더 정확하게 재현할 수 있도록 도와줍니다.

- **Technical Details**: 기존 3D Gaussian Splatting 방식이 표면의 반사 효과에서 부족함을 보였던 문제를 해결하기 위해, 새로운 대칭 행렬을 도입하여 각 Gaussian의 스칼라 불투명도를 시점에 의존하는 함수로 변경했습니다. 이로 인해 장면의 완전성을 해치지 않으면서도 스페큘러 조명과 동적 조명 효과를 효과적으로 나타낼 수 있습니다. 이러한 접근은 기존의 렌더링 속도를 크게 저해하지 않으면서도 고급 성능을 달성하는 데 기여합니다.

- **Performance Highlights**: 제안된 모델은 Mip-Nerf, Tanks&Temples, Deep Blending, Nerf-Synthetic 데이터셋에서 최첨단 성능을 보여주며, 렌더링 속도가 >60FPS를 달성했습니다. 메모리 사용량의 작은 증가만으로도 성능을 강화할 수 있었으며, 복잡한 실세계 장면에서도 안정적인 결과를 보여주고 있습니다.



### TransRAD: Retentive Vision Transformer for Enhanced Radar Object Detection (https://arxiv.org/abs/2501.17977)
Comments:
          Accepted by IEEE Transactions on Radar Systems

- **What's New**: TransRAD은 일반적인 레이더 데이터의 한계를 극복하기 위해 Retentive Vision Transformer(RMT)를 활용하여 3D 레이더 객체 감지를 위한 새로운 모델을 제안합니다. 이 모델은 레이더의 Range-Azimuth-Doppler(RAD) 데이터를 정보 밀도가 높은 특징으로 학습하는 데 중점을 두고 있으며, MaSA(Manhattan Self-Attention) 메커니즘을 사용하여 레이더 타겟의 공간적 특성을 정확하게 정렬합니다. 또한, Location-Aware Non-Maximum Suppression(NMS) 기법을 도입하여 딥 레이더 객체 감지에서 중복 바운딩 박스 문제를 효과적으로 해결합니다.

- **Technical Details**: TransRAD 모델은 Backbone, Neck 및 Head의 세 가지 주요 구성요소로 설계되었습니다. Backbone은 RMT를 사용하여 명시적 공간 감쇠 사전 정보를 활용하고, Neck 부분에서는 Feature Pyramid Network(FPN)를 적용하여 더 큰 특징 레이어의 의미적 풍부함을 향상시킵니다. Head 부분에서는 YOLOv8에서 영감을 받은 앵커 프리(Anchor-Free) 감지 헤드를 채택하여 작은 형태의 레이더 객체를 효율적으로 처리하며, 각각의 작업을 분리하여 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, TransRAD는 2D 및 3D 레이더 감지 작업 모두에서 최신 기술을 초월하는 성능을 발휘하며, 더 높은 정확도, 빠른 추론 속도 및 감소된 계산 복잡성을 달성하였습니다. 이 모델은 3D 바운딩 박스를 생성하여 레이더 객체의 거리, 각도 및 속도를 한 번의 단계에서 결정할 수 있으며, 레이더 데이터의 특정 요구 사항에 대한 정확한 중심 탐지를 위해 중심 손실을 도입하였습니다. LA-NMS 기법을 적용하여 레이더 객체 감지의 높은 위치 정확도를 활용하여 분류의 한계를 보완합니다.



### Unsupervised Patch-GAN with Targeted Patch Ranking for Fine-Grained Novelty Detection in Medical Imaging (https://arxiv.org/abs/2501.17906)
- **What's New**: 이번 연구에서는 희귀한 의학적 이상 징후(i.e., anomalies)를 효과적으로 탐지하기 위해 새로운 비지도 학습 기반 Patch-GAN 프레임워크를 제안합니다. 이 프레임워크는 이미지 내의 국소적 세부사항(local details)과 전반적인 구조(global structure)를 모두 활용하여 이상 징후를 정밀하게 감지하고 위치를 찾아냅니다. 기존의 전체 이미지 평가 방식의 단점을 극복하고, 고유한 패치 기반 접근 방식(patch-based approach)을 채택하여 보다 세밀한 분석이 가능합니다.

- **Technical Details**: 제안된 Patch-GAN 프레임워크는 마스크 이미지 재구성(masked image reconstruction)과 패치 평가를 통해 고유한 지역적 특징을 탐지하는 방식으로 작동합니다. 이 프레임워크는 각 패치의 진위를 평가하여 이상 징후를 식별하며, 패치 순위 설정(patch-ranking mechanism) 메커니즘을 통해 이상 점수가 높은 지역을 우선적으로 강조합니다. 이러한 접근은 이미지 내에서 작은 변화를 감지할 수 있는 정확도를 높이며, 보다 세밀한 이상 징후 탐지와 위치 지정이 가능합니다.

- **Performance Highlights**: ISIC 2016 피부 병변 및 BraTS 2019 뇌 종양 데이터셋에서 우리의 프레임워크는 각각 95.79% 및 96.05%의 AUC(Area Under the Curve) 성능을 달성하여 기존의 최첨단 방식들보다 우수한 성능을 입증했습니다. 이러한 결과는 제안된 접근 방식을 통해 의료 영상 분석의 정확성과 민감도를 크게 향상시킬 수 있음을 보여줍니다. 우리의 연구는 의학적 이상 탐지 분야에서 혁신적인 방법을 제시하며, 기존 기술의 한계를 극복하는 데 기여할 것입니다.



### VidSole: A Multimodal Dataset for Joint Kinetics Quantification and Disease Detection with Deep Learning (https://arxiv.org/abs/2501.17890)
Comments:
          Accepted by AAAI 2025 Special Track on AI for Social Impact

- **What's New**: 본 논문은 대규모 및 비용 효율적인 관절 하중 분석을 가능하게 하기 위한 세 가지 주요 기여를 소개합니다. 첫째, 새로운 기구화 인솔(instrumented insoles)의 개발과 배포, 둘째, 대규모 멀티모달 생체역학 데이터셋인 VidSole의 생성, 셋째, 내부 관절 하중 요소(prediction of internal joint loading factors)를 예측하기 위한 기본 심층 학습 파이프라인의 구축이 그것입니다. 이 연구는 관절 관련 질병 진단에 필수적인 정보를 제공하기 위해 노력하고 있습니다.

- **Technical Details**: 기구화 인솔은 발 아래 다섯 가지 고압 지점에서 삼축 힘(tri-axial forces)과 모멘트를 측정합니다. VidSole 데이터셋은 이 인솔로 측정된 힘과 모멘트뿐만 아니라, 두 가지 시점에서의 RGB 비디오, 3D 신체 모션 캡처, 힘 판(force plate) 데이터를 포함하고 있습니다. 이 연구는 관절 하중 예측을 위해 Gated Recurrent Unit(GRU)와 Long Short Term Memory(LSTM) 회귀 네트워크를 이용한 심층 학습 파이프라인을 활용합니다.

- **Performance Highlights**: 활동 분류 정확도는 99.02%에 달하며, Knee Adduction Moment(KAM)의 평균 절대 오차는 0.5%*body weight*height 미만으로, 이는 현재 무릎 골관절염 감지의 기준값에 해당합니다. 이러한 성과는 향후 연구 및 임상 환경에서의 활용 가능성을 보여줍니다. VidSole 데이터셋은 다양한 생체역학적 분석에 대해 실질적인 기여를 할 것으로 기대됩니다.



### Inkspire: Supporting Design Exploration with Generative AI through Analogical Sketching (https://arxiv.org/abs/2501.18588)
Comments:
          Accepted to CHI 2025

- **What's New**: 최근 Text-to-Image (T2I) AI 모델의 발전에 따라 제품 디자이너들이 이를 작업에 적용하기 시작했습니다. 하지만 현재의 T2I 도구는 추상적인 언어를 해석하는데 어려움이 있으며, 디자인 고착화(design fixation)를 유발할 수 있는 사용자 경험 문제를 안고 있습니다. 이를 해결하기 위해 Inkspire라는 스케치 기반 도구를 개발하였으며, 이를 통해 디자이너들은 아날로그적 영감을 바탕으로 프로토타입을 디자인할 수 있는 피드백 루프를 형성할 수 있습니다.

- **Technical Details**: Inkspire는 디자인 프로세스에서 T2I 모델을 지속적으로 탐색하도록 장려하는 워크플로우를 제시합니다. 이 도구는 아날로지적 디자인 개념을 활용하여 추상 테마에서 기발한 아이디어를 도출하고, 디자이너가 AI의 현재 상태를 이해하며 생성된 디자인 위에 구조화된 스케치를 직접 덧붙일 수 있게 합니다. 이를 통해 디자이너는 새로운 아이디어를 탐색할 수 있는 유연한 상호작용을 경험할 수 있습니다.

- **Performance Highlights**: Inkspire를 사용한 사용자들은 ControlNet에 비해 더 많은 영감과 탐색을 경험하였으며, 디자인 생성 프로세스가 더 협력적(co-creative)으로 느껴졌습니다. 연구 결과, Inkspire는 디자이너가 AI와의 협력, 제어 가능성, 커뮤니케이션 및 창작물에 대한 기여도를 크게 향상시켰습니다. 이를 통해 디자인 고착화를 피하고 더 많은 창의적 가능성을 탐구할 수 있게 도움을 주었습니다.



### Task-based Regularization in Penalized Least-Squares for Binary Signal Detection Tasks in Medical Image Denoising (https://arxiv.org/abs/2501.18418)
- **What's New**: 이번 연구에서는 의료 이미지의 노이즈 제거를 위한 혁신적인 과업 기반 정규화 전략을 제안하고 있습니다. 기존의 penalized least-squares (PLS) 방법과 함께 사용되며, 정규화 항은 노이즈 이미지의 테스트 통계량과 복원된 이미지 간의 거리를 측정합니다. 이 기법은 기존의 ground-truth 이미지 데이터에 대한 의존 없이도 동작할 수 있다는 점에서 중요한 발전입니다.

- **Technical Details**: 우리는 신호 탐지 문제를 다루며, 여기에서 의료 이미지는 신호가 없는 가설(H0) 또는 신호가 있는 가설(H1)로 해석됩니다. PLS 최적화 문제는 sparsity-promoting penalty를 포함하여 해결될 수 있으며, 이 과정에서 총 변동(total variation)이나 L1 노름 등의 정규화 항을 사용할 수 있습니다. 그러나 이러한 수작업으로 정의된 페널티는 과업 관련 정보를 보존하기 못하는 경우가 많기 때문에, 이번 연구에서는 새로운 정규화 Ψg(f)를 제안합니다.

- **Performance Highlights**: 컴퓨터 시뮬레이션 연구 결과, 제안된 과업 기반 PLS-TV 방법은 시각적 검토 및 receiver operating characteristic (ROC) 분석을 통해 신호 탐지 능력을 크게 향상시키는 것으로 나타났습니다. 이는 노이즈가 줄어든 이미지에서 신호 탐지 성능을 객관적으로 개선할 수 있다는 점을 강조합니다. 본 연구는 의료 이미지 노이즈 제거 분야에서의 새로운 접근 방식을 제공하며, 임상 응용에 필요한 진전으로 볼 수 있습니다.



### Real Time Scheduling Framework for Multi Object Detection via Spiking Neural Networks (https://arxiv.org/abs/2501.18412)
Comments:
          7 pages

- **What's New**: 본 논문에서는 조정 가능한 TIMESTEP을 갖춘 spiking neural networks (SNN) 기반의 다중 객체 감지(MOD) 시스템인 RT-SNN을 최초로 제안합니다. RT-SNN은 자율 모바일 에이전트(AMA)의 에너지 제약을 해결하며, 시간 보장(R1)과 높은 정확도(R2)를 동시에 달성하는 데 중점을 두고 있습니다. 이 시스템은 멤브레인 잠재력(membrane potential)을 재사용하는 새로운 방법을 통해 R1을 지원하며, 이는 SNN의 독특한 성격을 활용한 것입니다.

- **Technical Details**: RT-SNN은 여러 카메라를 갖춘 MOD 시스템을 대상으로 설계되었으며, 각 카메라는 기능적 중요도에 따라 초당 프레임(FPS)이 다를 수 있습니다. RT-SNN의 주요 구성 요소는 프레임 레벨 스케줄러, 동적 TIMESTEP 실행 파이프라인, 멤브레인 신뢰도 추정기로 이루어져 있으며, 이들은 공유 메모리 아키텍처를 통해 효율적인 상호 통신을 구현합니다. 각 TIMESTEP의 실행 옵션은 동적으로 조정되며, 이전 프레임에서 추출한 멤브레인 잠재력을 재사용할 수 있는 기능이 포함되어 있습니다.

- **Performance Highlights**: RT-SNN은 Spiking-YOLO를 기반으로 한 실험 평가를 통해 기존 방법보다 우수한 정확도, 에너지 소비 및 시간 보장을 나타냈습니다. SNN의 적용으로 인해 Truenorth에서 Spiking-YOLO는 NVIDIA Tesla V100 GPU에서의 Tiny-YOLO보다 280배 더 효율적입니다. 이러한 성능 향상은 R1과 R2를 동시에 충족시키면서도 에너지 효율성을 크게 개선한 결과입니다.



### MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding (https://arxiv.org/abs/2501.18362)
- **What's New**: 이번 논문에서는 MedXpertQA라는 새로운 벤치마크를 소개합니다. MedXpertQA는 전문가 수준의 의료 지식과 고급 추론을 평가하기 위해 설계된 도전적이고 포괄적인 기준을 제시합니다. 이 벤치마크는 17개 전문 분야와 11개 신체 시스템을 포함하며, 텍스트 평가를 위한 Text와 다중 모드 평가를 위한 MM의 두 가지 하위 집합으로 구성되어 있습니다.

- **Technical Details**: MM 하위 집합은 환자 기록 및 검사 결과를 포함한 다양한 이미지와 풍부한 임상 정보를 가진 전문가 수준의 시험 문제를 도입하여, 간단한 QA 쌍을 생성하는 기존의 의료 다중 모드 벤치마크와 차별화됩니다. MedXpertQA는 기존의 벤치마크인 MedQA와 같은 어려움이 부족한 문제를 해결하기 위해 엄격한 필터링과 증강을 적용하고, 임상 관련성과 포괄성을 높이기 위해 전문 분야 위원회 질문을 포함합니다.

- **Performance Highlights**: 우리는 MedXpertQA에서 16개의 주요 모델을 평가하였습니다. 의료는 실질적인 의사 결정과 깊은 연관이 있기 때문에, 수학 및 코드 외의 추론 능력을 평가하는 데 적합한 풍부하고 대표적인 환경을 제공합니다. 이를 위해 o1과 유사한 모델의 평가를 용이하게 하는 추론 지향 하위 집합을 개발했습니다.



### AGAV-Rater: Adapting Large Multimodal Model for AI-Generated Audio-Visual Quality Assessmen (https://arxiv.org/abs/2501.18314)
- **What's New**: 이 논문은 비디오-오디오 변환(Method for Video-to-Audio, VTA) 기술을 활용한 AI 생성 오디오-비주얼 콘텐츠(AI-generated audio-visual content, AGAV)의 품질 평가를 위한 최초의 대규모 데이터셋 AGAVQA를 소개하고 있다. AGAVQA는 총 3,382개의 AGAV를 수집하였으며, 이를 기반으로 다차원 점수를 제공하는 AGAVQA-MOS와 최적의 AGAV 쌍 선택을 위한 AGAVQA-Pair의 두 가지 하위 집합으로 나뉜다. 또한, AGAV-Rater라는 LMM 기반 모델을 제안하여 AGAV의 품질을 다차원적으로 평가하고, 사용자에게 제공할 최상의 AGAV를 선택하는 기능을 수행한다.

- **Technical Details**: AGAVQA 데이터셋은 16개의 VTA 방법에서 생성된 AGAV로 구성되며, 이 데이터셋은 주관적 실험을 통해 수집된 9,264개의 평균 의견 점수(Mean Opinion Scores, MOS)를 포함한다. AGAV-Rater는 LMM을 활용하여 AGAV의 품질을 판단하며, 50,952개의 지침-응답 쌍(instruction-response pairs)으로 사전 훈련되었다. 이러한 모델은 주관적인 인간의 품질 평가 방식을 모방하여 다차원적인 품질 점수를 예측할 수 있도록 설계되었다.

- **Performance Highlights**: AGAV-Rater는 AGAVQA, Text-to-Audio, Text-to-Music 데이터셋에서 최첨단 성과를 달성하였으며, 사용자 경험을 향상시키는 데 기여하고 있다. 실험 결과, 80%의 사용자들은 AGAV-Rater를 통해 선택된 고품질 AGAV가 더 나은 A/V 경험을 제공한다고 인식하였다. 이 결과는 AGAV-Rater가 VTA 방법의 품질 개선에 도움을 줄 수 있다는 점을 뒷받침한다.



### The iToBoS dataset: skin region images extracted from 3D total body photographs for lesion detection (https://arxiv.org/abs/2501.18270)
Comments:
          Article Submitted to Scientific Data

- **What's New**: 이번 연구는 피부암 진단을 위한 새로운 데이터셋 iToBoS를 소개합니다. 대부분의 공개 이미지 데이터셋이 단일 피부 병변에 중점을 두었다면, iToBoS는 3D 전체 신체 사진을 이용해 다양한 피부 영역에서 촬영된 16,954장의 이미지를 포함합니다. 이 데이터셋은 피부 상태와 관련된 다양한 메타데이터를 제공하여 AI 모델의 훈련을 지원합니다.

- **Technical Details**: 데이터셋 생성 과정은 데이터 수집, 주석 작성 및 공개 서브셋 선택의 세 단계로 나누어집니다. 각 이미지는 VECTRA WB360 시스템을 활용해 고해상도로 캡처되며, 주석은 질병을 나타내는 경계 상자로 제공됩니다. 각 이미지는 환자의 나이, 해부학적 위치, 햇빛 손상 점수와 같은 메타데이터와 결합되어 AI 모델이 병변과 건강한 피부를 구별할 수 있도록 돕습니다.

- **Performance Highlights**: iToBoS 데이터셋은 2024년 피부 병변 탐지 챌린지의 주요 요소로, AI 모델의 훈련과 평가를 위한 뛰어난 기회를 제공합니다. 이 대회는 피부병 진단 및 치료의 신속한 진행을 목표로 하며, 다양한 피부 병변을 탐지하는 최신 기계 학습 기법 개발을 촉진합니다. 결과적으로 이러한 노력은 피부암 조기의 발견과 진료 현장에서의 기술 배포 가능성을 높일 것으로 기대됩니다.



### Revisiting $\Psi$DONet: microlocally inspired filters for incomplete-data tomographic reconstructions (https://arxiv.org/abs/2501.18219)
- **What's New**: 이 논문에서는 $	ext{ΨDONet}$이라는 감독 학습 접근 방법을 재조명하며, 이론적 분석을 위한 더 깊은 미세 지역 해석을 제공합니다. 이 연구는 제한된 각도 토모그래피(sparse-angle tomography) 경우로의 연구를 확장하고, 불완전한 데이터에서 생성된 스트릭 아티팩트의 특성을 고안한 특수 필터를 고려하여 원래의 $	ext{ΨDONet}$의 구현을 세련되게 합니다. 이를 통해 학습 가능한 파라미터 수를 줄이면서도 제한된 각도 데이터로부터의 재구성을 위한 동일한 품질을 유지하거나 다소 향상시키는 것을 가능하게 합니다.

- **Technical Details**: 이 논문은 원래의 $	ext{ΨDONet}$ 접근 방식이 보는 각도(basin) 데이터를 사용하여 불완전한 시노그램으로부터 엣지 및 불연속성을 복구할 수 있도록 설계되었다는 점을 강조합니다. 특히, 미세 지역 분석(microlocal analysis)을 활용하여 그런 정보가 어떻게 촉진되는지 설명하고, 단순한 이상점(singularities)에 대한 보다 명확한 접근법을 제시합니다. 여기에 따라 $	ext{ΨDONet}$은 기존의 접근 방법과는 달리 데이터 기반 학습을 통해 스트릭 아티팩트를 완화하는 역할을 하며, 시노그램에서 보이지 않는 정보를 예측하고 상실되는 정보를 추측하는 데 의존하지 않습니다.

- **Performance Highlights**: 이 실험에서는 제한된 각도 및 희소 각도 토모그래피에 대한 새로운 미세 지역 필터의 잠재력을 수치 실험을 통해 논의합니다. 가장 주목할 만한 점은, 이러한 필터가 스트릭 아티팩트를 줄이는데 성공할 뿐만 아니라, 완전히 사라지게 할 수 있다는 점입니다. 전반적으로, 이 연구는 새로운 아키텍처의 구현을 통해 제한된 각도 데이터로부터의 재구성 품질을 보존하면서도 효율성을 높일 수 있음을 시사합니다.



### Scattering approach to diffusion quantifies axonal damage in brain injury (https://arxiv.org/abs/2501.18167)
- **What's New**: 본 연구는 신경 훈련의 조기 진단과 비침습적 모니터링을 위해 micrometer 수준의 축삭(axon) 형태학(morphology) 변화에 대한 높은 민감도를 보여주는 시간 의존적 확산 MRI(dMRI) 기술을 제시합니다. 기존의 의료 영상 기법으로는 확인할 수 없는 세포 수준의 변화를 감지하여, 신경 질환을 폭넓게 적용 가능한 정량적 바이오마커를 제공합니다.

- **Technical Details**: 연구에서는 성인 Sprague-Dawley 쥐를 대상으로 외상을 유도한 후, 세밀한 병리학적 검사를 위해 다양한 심리적, 생리적 환경에서 축삭을 캡처했습니다. 데이터 수집을 위해 SBEM(Sweepable Backscattered Electron Microscopy) 기법을 사용하여, 50 nm³ 분해능으로 세포를 촬영하였고, 딥러닝 기반의 DeepACSON 파이프라인을 통해 조각된 축삭 이미지를 자동으로 세분화하였습니다. 또한, 랜덤하게 배치된 비드를 가진 축삭을 생성하기 위해 몬테카를로 시뮬레이션을 적용하여 세포 내부의 공간 지오메트리를 모사하였습니다.

- **Performance Highlights**: 이 접근법은 대량의 축삭 데이터를 소량의 시간 내에 처리할 수 있는 가능성을 보여주며, 이는 래트 모델에서의 외상성 뇌 손상에 대한 dMRI 지표를 예측하는 데 중요한 의미를 지닙니다. 본 연구는 미세한 수준의 데이터 수집과 분석을 통해 신경 질환 및 생리적 변화 감지에 대한 새로운 기준을 확립하며, 향후 비침습적 진단 기술의 발전에 기여할 것으로 기대됩니다.



### Using Computer Vision for Skin Disease Diagnosis in Bangladesh Enhancing Interpretability and Transparency in Deep Learning Models for Skin Cancer Classification (https://arxiv.org/abs/2501.18161)
Comments:
          18 pages

- **What's New**: 이 연구는 방글라데시에서 피부암 진단과 치료를 위한 딥러닝 모델의 해석 가능성을 향상시키기 위한 새로운 접근법을 제안합니다. 피부암은 매년 200만 건 이상의 새 사례가 확인되는 가장 흔한 암의 유형으로, 방글라데시에서는 유방암에 이어 두 번째로 흔합니다. 현지에서 피부과 의사와 적격한 의료 전문가의 부족으로 인해 많은 사례가 고급 단계에서만 진단되었습니다.

- **Technical Details**: 연구에서는 saliency maps와 attention maps의 조합을 사용하여 딥러닝 모델의 진단에 영향을 미치는 주요 특징을 시각화하는 방법론을 제시합니다. 이러한 기법은 모델의 의사 결정 과정을 이해하는 데 도움이 됩니다. 딥러닝 알고리즘은 피부암 이미지를 효과적으로 분류할 수 있지만, 기존 모델들은 해석 가능성 부족으로 인해 그 활용에 장벽이 있었습니다.

- **Performance Highlights**: 제안된 방법은 피부암 분류의 해석 가능성을 높이며, 이러한 접근은 방글라데시의 피부암 진단 개선에 기여할 것으로 기대됩니다. 연구 결과는 데이터에 대한 명확한 이해를 제공하여 의료 전문가가 더욱 신뢰할 수 있는 진단을 할 수 있도록 지원합니다.



### Efficient Audiovisual Speech Processing via MUTUD: Multimodal Training and Unimodal Deploymen (https://arxiv.org/abs/2501.18157)
- **What's New**: 본 연구는 다양한 모달리티를 활용하여 학습하지만, 추론 과정에서는 단일 또는 축소된 모달리티를 사용하는 접근 방식을 개발했습니다. 이를 위해 제안된 Multimodal Training and Unimodal Deployment (MUTUD) 프레임워크는 TAME 모듈을 포함하여 결측 모달리티의 정보를 추정할 수 있습니다. 이 혁신적인 방법은 서로 다른 모달리티 간의 정보 통합을 촉진하며, 특정 모달리티가 결여된 상태에서도 전반적인 추론 프로세스를 개선합니다.

- **Technical Details**: MUTUD 프레임워크는 다양한 음성 처리 작업에 적용되며, 각각의 특성에 맞춰 교육 및 추론을 효과적으로 지원합니다. TAME(Temporally Aligned Modality feature Estimation) 모듈은 추론 시 존재하는 모달리티의 표현을 사용하여 결측 모달리티의 심층 표현을 추정합니다. 이 과정에서 각 모달리티의 코드북을 사용하여, 다양한 모달리티의 특징을 상호 연결하고 결여된 모달리티의 정보를 재생성하는 방식으로 작동합니다.

- **Performance Highlights**: MUTUD를 활용한 실험에서는 음성 향상, 음성 인식, 활성 화자 검출 과제에서 unimodal 추론이 기존 unimodal 데이터로 교육된 모델에 비해 현저히 높은 성능을 보였습니다. 또한, MUTUD는 전체 다중모달 시스템에 비해 모델 크기와 계산 성능이 거의 80%까지 감소하면서도 경쟁력 있는 성능을 유지합니다. 이는 현실 세계에서 다중모달 시스템의 사용 제약을 극복하는데 큰 기여를 할 것으로 기대됩니다.



### Lifelong 3D Mapping Framework for Hand-held & Robot-mounted LiDAR Mapping Systems (https://arxiv.org/abs/2501.18110)
- **What's New**: 새로운 연구에서는 손에 들 수 있는 LiDAR 및 로봇 장착 LiDAR 매핑 시스템 모두를 지원하는 평생 3D 매핑 프레임워크를 제안합니다. 이 프레임워크는 동적 포인트 제거(dynamic point removal), 다중 세션 맵 정렬(multi-session map alignment), 맵 변화 감지(map change detection), 맵 버전 관리(map version control)로 구성되어 있습니다. 특히 모든 입력 원시 세션 맵을 저장하지 않고도 이전 청정 세션 맵을 재구성하고 사용자가 두 매핑 세션 간의 변화를 쿼리할 수 있는 기능을 제공합니다.

- **Technical Details**: 제안된 시스템은 다양한 센서 설정에 의존하지 않는 동적 포인트 제거 알고리즘을 사용합니다. 첫 번째로 청정 정적 맵을 생성한 후, 다중 세션 맵 정렬 알고리즘이 두 단계 접근법으로 자동으로 정적 맵들을 정렬하여 하나의 참조 프레임으로 통합합니다. 또한 맵 변화 감지 모듈은 정렬된 두 맵 사이에서 긍정적 및 부정적인 변화를 식별하며, 맵 버전 관리 모듈은 환경의 현재 상태를 나타내는 기반 맵을 유지 관리합니다.

- **Performance Highlights**: 이 프레임워크는 상용 LiDAR 매핑 장치 및 오픈 소스 로봇 장착 LiDAR SLAM 알고리즘을 사용하여 각 모듈 및 전체 3D 평생 매핑 프레임워크의 성능을 평가하는 광범위한 실험을 수행하였습니다. 이 시스템은 메모리 효율성이 뛰어나며, 이전 세션 맵을 다운로드하거나 두 개의 세션 간의 변화를 쿼리할 수 있어 장기적인 로봇 작동 및 디지털 트윈 시뮬레이션에 유리합니다.



### Influence of High-Performance Image-to-Image Translation Networks on Clinical Visual Assessment and Outcome Prediction: Utilizing Ultrasound to MRI Translation in Prostate Cancer (https://arxiv.org/abs/2501.18109)
Comments:
          9 pages, 4 figures and 1 table

- **What's New**: 이번 연구는 이미지-투-이미지 변환(I2I) 네트워크의 핵심 특성을 분석하였으며, 특히 일상적인 임상 환경에서의 효과성과 적응성에 집중하였습니다. 794명의 전립선 암 환자 데이터를 활용하여 10개의 주요 2D/3D I2I 네트워크를 사용하여 초음파(US) 이미지를 MRI 스캔으로 변환하는 작업을 수행했습니다. 또한 Spearman 상관 계수를 통한 새로운 Radiomic features(RF) 분석을 도입하였습니다.

- **Technical Details**: 연구에서는 2D-Pix2Pix 네트워크가 7개의 다른 네트워크보다 평균 SSIM~0.855로 크게 우수한 성능을 보인 것을 확인했습니다. 이 과정에서 186개의 RF 중 76개가 2D-Pix2Pix 알고리즘을 통해 감지되었지만, 번역 과정에서 절반의 RF가 소실되었습니다. 또한 7명의 의료 전문가에 의한 질적 검토에서 I2I 작업의 저수준(feature) 특성 인식에 결핍이 발견되었습니다.

- **Performance Highlights**: 합성된 이미지 기반의 분류는 US 이미지 기반 분류보다 우수한 성능을 보였으며, 평균 정확도와 AUC~0.93으로 평가되었습니다. 2D-Pix2Pix 네트워크는 최첨단 네트워크들보다 저수준 특성 발견과 전체 오류 및 유사성 메트릭에서 높은 성능을 기록했지만, 여전히 저수준 특성 성능을 개선할 필요성이 있음이 강조되었습니다.



### Visualization of Organ Movements Using Automatic Region Segmentation of Swallowing C (https://arxiv.org/abs/2501.17897)
Comments:
          8 pages, 5 figures, 1 table

- **What's New**: 본 연구는 4차원 컴퓨터 단층촬영(4D-CT) 이미지를 이용한 자동 지역 분할(automatic region segmentation)을 위한 인공지능(AI) 개발에 대한 첫 번째 보고서입니다. 이 AI는 삼키는 과정 중 촬영된 4D-CT 이미지를 기반으로 트레이닝되었습니다. 또한, 저작 및 삼킴 중 촬영된 4D-CT 이미지를 통해 AI의 실용성을 검증하기 위한 데이터를 수집하였습니다.

- **Technical Details**: AI의 지역 분할을 위한 Ground Truth 데이터는 삼킴 과정의 4D-CT 데이터 셋 5개에서 생성되었습니다. AI 모델로 사용된 nnU-Net은 3D convolutional 모델을 사용하였으며, 학습 방식으로는 Leave-One-Out Cross-Validation(LOOCV)을 채택했습니다. nnU-Net의 훈련을 위한 Epoch 수는 100이었으며, Dice 계수를 통해 AI의 지역 분할 정확도를 평가했습니다.

- **Performance Highlights**: AI의 지역 분할 정확도를 평가한 결과, 중위 Dice 계수가 0.7 이상인 지역으로는 볼루스, 뼈, 혀, 부드러운 입천장이 포함되었습니다. 하지만 갑상연골(thyroid cartilage) 및 후두개(epiglottis)와 같은 지역은 0.7 이하의 Dice 계수를 기록하였습니다. 실용성 검증 과정에서 얼굴 뼈, 턱 뼈, 혀에 대한 오인식은 없었으나, 느린 동작 시에는 혀골(hyoid bone), 갑상연골, 후두개가 완전히 구획화되지 않았습니다. 향후 연구가 AI의 지역 분할 정확도를 향상시키기를 기대하지만, 잘못 인식할 위험은 항상 존재할 것입니다.



### Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion (https://arxiv.org/abs/2501.17887)
Comments:
          Accepted to AAAI 25: Workshop on Open-Source AI for Mainstream Use

- **What's New**: Docling은 MIT 라이센스 하에 제공되는 오픈소스 문서 변환 툴킷으로, 다양한 문서 포맷을 통합된 구조로 변환할 수 있는 기능을 갖추고 있습니다. 이 툴킷은 최신의 AI 모델인 DocLayNet과 TableFormer를 활용하여 레이아웃 분석과 표 구조 인식을 지원하며, 일반 하드웨어에서도 효율적으로 실행됩니다. Docling은 Python 패키지로 제공되어 API 및 CLI 도구로 사용할 수 있으며, 모듈화된 아키텍처 덕분에 새로운 기능 및 모델의 구현이 용이합니다.

- **Technical Details**: Docling은 파이프라인, 파서 백엔드, DoclingDocument 데이터 모델로 구성된 모듈형 아키텍처를 가지고 있습니다. DoclingDocument 모델은 텍스트, 표, 이미지, 캡션, 목록 등 다양한 문서 기능을 표현하며, 문서 구조와 위치 정보, 출처 정보를 포함합니다. 사용자는 이 데이터 모델을 기반으로 문서를 생성하고 검사하며, JSON 및 HTML과 같은 일반 형식으로 내보낼 수 있는 API에 접근할 수 있습니다.

- **Performance Highlights**: 출시된 이후, Docling은 AI 개발자 커뮤니티에서 큰 관심을 모았고, GitHub의 월간 트렌딩 레포지토리에서 10,000개 이상의 별을 기록하며 1위에 올랐습니다. Docling v2는 새로운 기능 및 개념을 추가하며 성능과 효율성을 더욱 개선했습니다. 이 툴킷은 문서 변환을 위한 신뢰할 수 있는 솔루션으로 자리잡고 있으며, LangChain, LlamaIndex와 같은 주요 AI 개발 프레임워크와의 통합이 용이합니다.



New uploads on arXiv(cs.AI)

### Semantic Web and Creative AI -- A Technical Report from ISWS 2023 (https://arxiv.org/abs/2501.18542)
Comments:
          Technical Report

- **What's New**: ISWS 2023에서 진행된 연구는 Semantic Web 기술과 Creative AI의 교차점에 중점을 두고 있습니다. 특히, LLMs (Large Language Models)를 지식 공학의 지원 도구로 활용하는 가능성이 중요한 주제로 다뤄졌습니다. 이 프로그램은 참가자들이 다양한 연구 질문을 통해 Creative AI에 대한 새로운 관점을 제시하도록 독려했습니다.

- **Technical Details**: 참가팀들은 LLMs의 다양한 응용사례를 탐구했습니다. 여기에는 창작 콘텐츠 생산의 법적 측면, 인적 요소의 포함, 다중 모달 생성 AI 모델의 분산 접근 방식, 나노 출판과 개인 과학 지식 그래프를 위한 AI, 자동 스토리와 내러티브 완성을 위한 상식 지식, 그리고 음악 작곡 자동화 등이 포함됩니다. 또한 이들은 프롬프트 엔지니어링(prompt engineering) 및 암묵적 지식의 이끌어내기에 대한 연구도 진행하였습니다.

- **Performance Highlights**: LLMs와 시맨틱 기술의 발전에 따라 창의적 표현과 사실적 지식 사이의 경계가 점점 허물어질 새로운 기회들이 등장하고 있습니다. 이러한 연구는 지식이 정보적이면서도 영감을 주는 세계로 나아가는 가능성을 열어줍니다. ISWS 2023는 참가자들에게 창의적인 AI의 미래에 대한 다양한 시각을 제공하며, 학문적 협업의 중요성을 강조했습니다.



### Conversation Games and a Strategic View of the Turing Tes (https://arxiv.org/abs/2501.18455)
- **What's New**: 이 논문은 언어가 중심이 되는 전략적 상호작용을 다루는 '대화 게임(conversation game)'이라는 다단계 게임 모델을 소개합니다. 특히 판결 게임(verdict game)에 주목하며, 이 게임에서는 두 플레이어가 대화를 나누고 비전략적 심사자가 이를 평가하는 구조입니다. 이러한 framework는 언어 사용의 전략적 본질을 분석하는 데 도움을 주며, Turing test가 판결 게임의 한 예임을 설명합니다.

- **Technical Details**: 대화 게임은 여러 플레이어가 언어로 상호작용하는 다단계의 폭넓은 형태의 게임입니다. 각 단계에서 플레이어는 발언을 통해 대화에 기여하며, private type 및 distinct incentives에 따라 정보를 교환합니다. 판결 게임은 이러한 대화의 결과가 외부 판별자에 의해 결정되며, 플레이어의 유틸리티는 대화의 내용에 의해 영향을 받지만 완전히 정의되지는 않습니다.

- **Performance Highlights**: 시뮬레이션 실험 결과에서 전략적 에이전트는 순진한 에이전트에 비해 상당히 높은 성능을 보여줍니다. 이는 제안된 개념이 실제 세계의 대화 상황에 대해 실제로 어떻게 적용될 수 있는지를 시사합니다. 향후 연구에서 이러한 게임 이론이 AI와의 상호작용에서 어떻게 적용될 수 있는지에 대한 논의가 기대됩니다.



### GBFRS: Robust Fuzzy Rough Sets via Granular-ball Computing (https://arxiv.org/abs/2501.18413)
- **What's New**: 본 논문에서는 다중 세분화(granularity) 구형(ball) 계산을 퍼지 러프 집합(fuzzy rough set) 이론에 통합하는 방법을 제안합니다. 기존 퍼지 러프 집합 모델이 가장 세밀한 세분화에 기반하고 있는 반면, 이 연구는 다양한 크기의 구형을 사용하여 샘플 공간을 적응적으로 표현하고 덮을 수 있도록 합니다. 이러한 접근 방식은 모델의 강인성을 강화하는 데 중요한 역할을 합니다.

- **Technical Details**: 연구에서는 구형 퍼지 러프 집합(granular-ball fuzzy rough set, GBFRS) 프레임워크를 제안하며, 구형의 상하 근사에 대해 엄격하게 정의하고 관련 정리에 대한 공식을 제시합니다. feature selection 방법의 알고리즘 프로세스를 설계하였고, 이를 통해 GBFRS 프레임워크의 이론적 타당성을 확보하였습니다. 또한, GBFRS 방식이 UCI 데이터셋에서 기존 퍼지 러프 집합 방법보다 효율적임을 입증했습니다.

- **Performance Highlights**: 실험 결과는 제안된 GBFRS 방법이 기존의 방법들보다 뛰어난 성능을 보임을 나타냅니다. 특히, 샘플의 노이즈에 대한 강인성(reliability)을 높이며, 그에 따른 feature selection 방식에서도 개선된 결과를 보여줍니다. 이는 고차원 복합 데이터 세트의 분석에서 매우 중요한 발전을 의미합니다.



### Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents (https://arxiv.org/abs/2501.18411)
Comments:
          Technical report - Work in progress

- **What's New**: Gravity-Bench-v1은 AI 에이전트의 과학적 발견 능력을 평가하는 새로운 벤치마크로, 고급 중력역학 시뮬레이션을 통해 동적인 환경에서 물리학을 탐구하는 임무를 제시합니다. 이 벤치마크는 AI가 관찰 데이터를 수집하고 이를 통해 자율적으로 의사결정을 내릴 수 있는 능력을 평가하는 데 중점을 두고 있습니다. 특히, Gravity-Bench-v1은 기존 벤치마크에서 부족했던 과학적 불확실성과 새로운 현상의 발견을 고려한 평가를 제공합니다.

- **Technical Details**: Gravity-Bench의 핵심 설계 원칙은 엄격하게 시뮬레이션된 부분 관찰 가능 환경을 기반으로 합니다. 이 환경은 과학적 주제인 2체 중력역학을 중심으로 하여 AI 에이전트가 특정 물리적 지식을 기반으로 평가될 수 있도록 합니다. 다양한 관찰 프로토콜을 사용하여 환경 내의 데이터를 조절하며, 이는 에이전트가 새로운 상황을 처리하고 일반화 능력을 평가할 수 있게 합니다.

- **Performance Highlights**: Gravity-Bench에서는 AI 에이전트가 데이터를 수집하는 동안의 의사결정 능력과 함께 점진적으로 불확실성이 줄어드는 환경에서의 추론 능력이 평가됩니다. PhD 수준의 솔루션이 제공되며, 이는 AI 성능을 인간 전문성과 비교할 수 있는 기준이 됩니다. 벤치마크의 오픈 엔디드 특성은 다양한 해결 전략을 허용하며, 이는 탐색적 추론과 가설 생성을 장려합니다.



### MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding (https://arxiv.org/abs/2501.18362)
- **What's New**: 이번 논문에서는 MedXpertQA라는 새로운 벤치마크를 소개합니다. MedXpertQA는 전문가 수준의 의료 지식과 고급 추론을 평가하기 위해 설계된 도전적이고 포괄적인 기준을 제시합니다. 이 벤치마크는 17개 전문 분야와 11개 신체 시스템을 포함하며, 텍스트 평가를 위한 Text와 다중 모드 평가를 위한 MM의 두 가지 하위 집합으로 구성되어 있습니다.

- **Technical Details**: MM 하위 집합은 환자 기록 및 검사 결과를 포함한 다양한 이미지와 풍부한 임상 정보를 가진 전문가 수준의 시험 문제를 도입하여, 간단한 QA 쌍을 생성하는 기존의 의료 다중 모드 벤치마크와 차별화됩니다. MedXpertQA는 기존의 벤치마크인 MedQA와 같은 어려움이 부족한 문제를 해결하기 위해 엄격한 필터링과 증강을 적용하고, 임상 관련성과 포괄성을 높이기 위해 전문 분야 위원회 질문을 포함합니다.

- **Performance Highlights**: 우리는 MedXpertQA에서 16개의 주요 모델을 평가하였습니다. 의료는 실질적인 의사 결정과 깊은 연관이 있기 때문에, 수학 및 코드 외의 추론 능력을 평가하는 데 적합한 풍부하고 대표적인 환경을 제공합니다. 이를 위해 o1과 유사한 모델의 평가를 용이하게 하는 추론 지향 하위 집합을 개발했습니다.



### Leveraging LLM Agents for Automated Optimization Modeling for SASP Problems: A Graph-RAG based Approach (https://arxiv.org/abs/2501.18320)
- **What's New**: 이 논문에서는 자동화된 최적화 모델링(Automated Optimization Modeling, AOM) 접근 방식을 기존의 프롬프트 공학(prompt engineering) 대신, 검색 강화 생성(retrieval-augmented generation, RAG) 기술에 기반하여 제안합니다. 새로운 접근 방식은 멀티 에이전트(multi-agent) 구조와 그래프 기반 RAG(Graph-RAG) 과정을 포함하여 SASP(센서 배열 신호 처리) 문제 해결을 위한 도메인 지식을 향상시키는 데 주목적을 둡니다. 여기서 제안된 방식(MAG-RAG)은 기존의 여러 AOM 기준을 초월하는 성능을 나타냅니다.

- **Technical Details**: 제안된 AOM 접근 방식은 두 개의 주요 워크플로로 구성됩니다. 첫 번째는 도메인 특화 문서에서 지식 데이터베이스를 구축하기 위한 Graph-RAG 기법을 활용하며, 두 번째는 여러 에이전트가 참여하는 자동화된 최적화 모델링 프로세스입니다. 이 시스템의 세 가지 주요 에이전트는 정보 추출 에이전트(Extraction Agent), 용어 에이전트(Terminology Agent), 최적화 모델링 에이전트(Optimization Modeling Agent)로 구성되며, 각각의 역할을 통해 문제를 더 세분화하여 보다 효율적으로 해결하는 방식입니다.

- **Performance Highlights**: 실험 결과는 제안된 MAG-RAG 접근 방식이 10개의 고전적인 SASP 문제에서 여러 AOM 기준을 초과하는 성능을 보였음을 보여줍니다. 이 연구는 SASP 문제 해결에 있어 LLM(대형 언어 모델) 지원 AOM의 잠재력을 실현하기 위한 실행 가능한 접근 방식을 개발하였습니다. 또한, 다수의 도전 과제가 식별되고 논의되어 향후 연구 방향을 제시합니다.



### Model-Free RL Agents Demonstrate System 1-Like Intentionality (https://arxiv.org/abs/2501.18299)
- **What's New**: 이 논문은 모델이 없는 강화학습(RL) 에이전트가 명시적인 계획 메커니즘 없이도 인간 인지에서의 시스템 1('빠르게 생각하기') 과정과 유사한 행동을 보인다는 주장을 담고 있습니다. 모델기반 RL 에이전트는 내부 표현을 활용하여 계획함으로써 시스템 2('느리게 생각하기') 추론을 수행하는 반면, 모델 없는 에이전트는 환경 자극에 반응할 뿐입니다. 저자들은 시스템 1과 시스템 2의 이분법을 모델이 없는 RL과 모델기반 RL의 구별에 연결하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 강화학습(RL) 에이전트는 세계를 측정 가능한 상태의 벡터로 인식하며 각 상태에 대해 가능한 행동을 선택할 수 있습니다. 모델기반 에이전트는 세계 모델을 학습하여 행동의 결과를 이해하고 계획을 세우지만, 모델 없는 에이전트는 시행착오를 통해 학습합니다. 특히 최근의 연구는 모델 없는 RL의 성공을 보여주고 있으며, 이러한 경향이 두 가지 학습 방법 사이의 경계가 희미해지고 있음을 나타냅니다.

- **Performance Highlights**: 이 논문은 AI 시스템과 RL 에이전트가 인간의 사고 방식과는 다르게 생각할 수 있으며, 이로 인해 의도(인텐트)(intent)의 개념을 재고해야 한다고 강조합니다. 모델 없는 에이전트가 패턴 인식과 반응적인 행동을 통해 의도를 가진 것으로 간주할 수 있다는 점을 탐구하며, 이는 AI의 윤리적 배치 및 규제에 중요한 시사점을 제공합니다. AI의 행동을 규명하기 위해서는 단순한 시행착오를 넘어서 의도를 이해하고 정의하는 데에 모델이 필요하다고 주장합니다.



### Extending the design space of ontologization practices: Using bCLEARer as an examp (https://arxiv.org/abs/2501.18296)
- **What's New**: 이 논문에서는 현재의 관행보다 더 풍부한 디자인 공간(design space)을 가진 온톨로지화 과정(ontologization process)을 제시하고 있습니다. 엔지니어링 프로세스와 제품이 디자인되어야 하는 필요성을 강조하며, 디자인의 구성 요소들을 식별합니다. 또한, bCLEARer라는 비정형 방법론(outlier methodology)을 통해 지난 30년 간의 새로운 관행을 설계할 가능성을 탐구합니다.

- **Technical Details**: 온톨로지화에 대한 진화적 맥락(evolutionary context)을 설정함으로써 이러한 새로운 관행의 본질을 이해하는 데 도움이 되며, 이는 비옥한 프로세스를 형성하는 개념적 지주(conceptual scaffolding)를 제공합니다. 논문에서는 디지털화(digitalization)를 정보 전환의 긴 진화 과정에서 최신 단계로 재구성하며, 이는 온톨로지화를 디지털화에서 제공하는 기회를 활용하는 전략적 도구로 재정립합니다.

- **Performance Highlights**: 연구 결과, 새로운 디자인 방법론을 통해 기존의 엔지니어링 설계 프로세스를 혁신할 기회를 모색하고 있습니다. 이로 인해 디지털 기술의 발전을 통해 온톨로지화 과정이 어떻게 더 풍부하고 발전적인 형태로 나아갈 수 있는지를 보여줍니다.



### CueTip: An Interactive and Explainable Physics-aware Pool Assistan (https://arxiv.org/abs/2501.18291)
- **What's New**: CueTip은 풀 게임을 위한 인터랙티브하고 설명 가능한 자동 코칭 어시스턴트로, 자연어 인터페이스, 물리적으로 민감한 맥락 추론 능력 및 도메인 전문가에 의해 개발된 규칙에 기반한 설명 기능을 결합한 혁신적인 시스템입니다. 물리 시뮬레이터와 결합된 CueTip은 이벤트 트레이스를 자연어로 생성하여 사용자 질의에 따라 적절한 샷과 함께 설명을 제공합니다. 이를 통해 사용자는 특정 테이블 상태에 맞는 맥락 도움을 받을 수 있습니다.

- **Technical Details**: CueTip은 물리 시뮬레이터와 off-the-shelf 언어 모델(LM)을 폐쇄 루프 방식으로 활용하여 물리적 시스템의 동적 특성을 효과적으로 반영합니다. 사용자는 샷 옵션을 질의할 때, CueTip은 도메인 전문가의 규칙에 기반하여 최적의 전략을 제안하고 설명을 제공합니다. 이 어시스턴트는 샷 선택 전략을 조정할 수 있어 다양한 풀 게임 에이전트를 모방할 수 있는 유연성을 가지고 있습니다.

- **Performance Highlights**: 실험 결과 CueTip은 사용자 질의에 대한 맥락 기반 지원을 제공하며, 이 과정에서 승률을 향상시킬 수 있는 잠재력을 보여줍니다. 설명은 전문가의 규칙에 기반하여 생성되므로 신뢰성이 높고 물리적 사고를 반영합니다. 이러한 특징들을 통해 CueTip은 긴밀한 통합을 이룬 자연어의 이해와 물리적 현실 감각을 효과적으로 결합했습니다.



### On Scaling Neurosymbolic Programming through Guided Logical Inferenc (https://arxiv.org/abs/2501.18202)
- **What's New**: 이번 논문에서는 신경망(neural networks)과 기호 프로그래밍(symbolic programming)을 통합하는 확률적 신경기호 학습(probabilistic neurosymbolic learning)에 대한 새로운 접근법을 제안합니다. 기존 시스템들이 Boolean 공식(complex Boolean formula)인 Probabilistic Weighted Model Counting Problem (PWMC)에 의존해온 반면, 이번 연구는 DPNL이라는 정확한 알고리즘을 통해 이 과정을 우회할 수 있는 방법을 제시합니다. DPNL은 오라클(oracle) 원칙과 DPLL-like 분해 방식을 기반으로 하여, 논리적 과정을 보다 빠르고 효과적으로 처리합니다.

- **Technical Details**: DPNL 접근법은 PWMC와 같은 복잡한 연산을 필요로 하지 않으며, 확률적 추론에서 감마(ε) 또는 (ε, δ) 보장을 제공하는 방법으로 적응할 수 있습니다. 이는 추론 프로세스의 성능을 향상시키고 정확도를 높이는 데 기여합니다. DPNL을 통해 논리적 공식의 계산을 신속하게 처리할 수 있는 구조를 만들고, 이를 통해 확률적 모델링의 효율성을 개선할 수 있습니다.

- **Performance Highlights**: ApproxDPNL 방법은 신경기호 프로그래밍(neurosymbolic programming)의 확장 가능성을 크게 향상시키는 잠재력을 보여줍니다. 이 접근 방식은 근사치를 포함하면서도 추론 과정에 대한 보장을 유지하여, 더 정교하고 정확한 추론을 가능하게 합니다. 따라서 DPNL과 ApproxDPNL은 신경기호 시스템의 다양한 응용 가능성을 더욱 넓혀줄 것으로 기대됩니다.



### Neural Operator based Reinforcement Learning for Control of first-order PDEs with Spatially-Varying State Delay (https://arxiv.org/abs/2501.18201)
Comments:
          6 Pages, 7 Figures

- **What's New**: 이 논문에서는 공간 변동 지연 (spatially-varying delays)의 영향을 받는 불안정한 1차 초월 PDE (hyperbolic PDE) 제어 문제를 다루고 있습니다. 특히, 후방 스텝 제어(backstepping control) 전략과 심층 강화 학습 (deep reinforcement learning, RL)을 결합하여 새로운 NO-SAC 아키텍처를 제안합니다. DeepONet을 활용하여 후방 스텝 컨트롤러를 근사하고 이를 정책 네트워크 (policy network)에 통합함으로써 지연 함수에 대한 가정이 제거됩니다.

- **Technical Details**: NO-SAC 아키텍처는 DeepONet을 기반으로 하여 후방 스텝 제어기의 내재적 지식을 활용하며, 정책 네트워크와 가치 네트워크 (value network) 내에 기능 추출 네트워크로 통합되어 있습니다. 이 접근법은 전통적인 RL 알고리즘의 한계를 극복하고, 지연 함수에 대한 가정 없이 유연한 제어가 가능하도록 설계되었습니다. DeepONet은 고차원 매핑을 근사하는 데 효과적이며, 복잡한 시스템 행동을 적은 데이터와 계산 자원으로 포착할 수 있는 구조를 가지고 있습니다.

- **Performance Highlights**: 시뮬레이션 결과, NO-SAC 알고리즘은 기존의 SAC보다 우수한 성능을 보여주었으며, 폐쇄 루프 시스템에서의 정밀도 오류를 효과적으로 제거했습니다. 또한, 동작 지연 (delay assumption)을 만족하는 동일한 지연 함수 하에서 RL 기반 제어 성능이 후방 스텝 방법보다 우수한 전이 성능을 나타냈습니다. 본 논문의 기여로는 NO-SAC 방법이 지연 함수에 대한 가정을 없애고 성능을 개선하는 데 기여한 점을 강조합니다.



### Economic Rationality under Specialization: Evidence of Decision Bias in AI Agents (https://arxiv.org/abs/2501.18190)
- **What's New**: Chen et al. (2023)의 연구에서는 대형 언어 모델인 GPT가 예산 할당 및 위험 선호와 같은 작업에서 평균적인 인간 수준과 유사하거나 그 이상의 경제적 합리성을 보여주었다. 이 연구는 이러한 발견을 기반으로 바이오테크 전문가 및 경제학자와 같은 전문 에이전트를 포함해 수평 비교를 통해 전문성이 경제적 합리성을 향상시킬 수 있는지 탐구하였다.

- **Technical Details**: 연구 결과, 전문 분야에 더 많은 노력을 투자하는 에이전트는 '합리성 이동(rationality shift)'에 더 취약하여 GARP(Generalized Axiom of Revealed Preference)의 위반이 증가하고, CCEI(Critical Cost Efficiency Index)가 감소하며, 고위험 조건에서 결정 편차가 더 커지는 경향이 있음을 나타냈다. 반면 GPT와 더 일반화된 기본 에이전트는 여러 작업에서 보다 안정적이고 일관된 합리성 수준을 유지하였다.

- **Performance Highlights**: 이 연구는 전문화와 경제적 합리성 간의 본질적인 갈등을 드러내며, 다양한 시나리오에서 전문화와 일반화의 균형을 이룰 수 있는 AI 결정 시스템 구축에 대한 새로운 통찰을 제공한다. 이는 AI의 의사결정 기술 향상에 기여할 것으로 기대된다.



### Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judg (https://arxiv.org/abs/2501.18099)
- **What's New**: 이번 연구에서 제안하는 EvalPlanner는 LLM-as-a-Judge 모델의 평가 계획과 실행을 최적화하는 방법론이다. 난잡한 평가 기준 없이 LLM이 이 평가를 수행할 수 있도록 하여, 더 많은 테스트 시간 동안 코어 사고(Chain-of-Thought, CoT)를 생성한다. 이를 통해 LLM의 최종 판단이 더욱 신뢰성 있고 투명하게 이루어질 수 있도록 한다.

- **Technical Details**: EvalPlanner는 입력 지침에 따라 평가 계획을 작성하고, 이를 단계적으로 실행하여 최종 판단을 도출하는 방식으로 작동한다. 계획 수립 과정에서는 응답 평가에 필요한 모든 단계를 포함하는 세부적인 평가 계획이 생성된다. 이후 실행 단계에서 모델은 생성된 계획을 따르며 입력 응답에 대한 분석 과정을 통해 최종 판단을 수행한다.

- **Performance Highlights**: EvalPlanner는 RewardBench에서 generative reward models의 새로운 최첨단 성능 점수인 93.9를 달성하며, 기존의 많은 데이터로 훈련된 모델들을 능가했다. 또한, 다른 벤치마크에서도 최대 13% 향상된 성능을 나타내며, 개별 평가 기준을 모델이 점진적으로 최적화하도록 학습하면서 평가의 정확성을 개선하고 있다.



### Normative Evaluation of Large Language Models with Everyday Moral Dilemmas (https://arxiv.org/abs/2501.18081)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 도덕적 판단을 복잡한 일상적 윤리 딜레마를 통해 평가하는 새로운 접근 방식을 제시합니다. 특히, Reddit의 'Am I The Asshole' (AITA) 커뮤니티에서 유래한 상황들을 분석하여 LLMs가 어떻게 도덕적으로 판단하고 설명하는지를 살펴보았습니다. 이러한 연구를 통해, LLMs와 인간의 도덕적 평가 사이의 차이를 강조하고, 각 모델에서 나타나는 특정한 도덕적 원칙의 사용 패턴도 밝혀냈습니다.

- **Technical Details**: 연구에서는 10,000개 이상의 AITA 도덕적 딜레마에 대해 7개의 LLMs가 어떻게 비난을 부여하고 설명을 제공하는지를 분석했습니다. 결과적으로 LLMs는 인간 Reddit 사용자들과 비교했을 때 상당히 다른 도덕적 판단 패턴을 보였으며, 모델 간 일치도가 낮았고, 각 모델의 설명에서도 뚜렷한 차이가 나타났습니다. 이러한 결과는 인공지능 시스템에서 일관된 도덕적 판단을 구현하는 복잡성을 강조하며, 다양한 모델이 윤리적 판단에 접근하는 방식에 대한 면밀한 평가의 필요성을 보여줍니다.

- **Performance Highlights**: 연구 결과에 따르면, LLMs는 중간에서 높은 자기 일관성을 보였지만, 모델 간의 합의는 낮은 수준이었습니다. 또한, LLMs의 도덕적 이유 제시에 대한 분석 결과는 각 모델이 다양한 도덕적 원칙을 어떻게 사용하는지에 대한 뚜렷한 패턴을 보여주었습니다. 이 발견들은 인간의 도덕적 판단과 LLM의 도덕적 판단 사이의 복잡한 관계를 탐구하는 데 있어 중요한 통찰력을 제공하며, 지속적인 윤리적 결정이 요구되는 분야에서 LLM의 사용에 따른 잠재적 편향과 한계를 이해하는 데 필수적입니다.



### Large Language Models Think Too Fast To Explore Effectively (https://arxiv.org/abs/2501.18009)
Comments:
          16 pages, 13 figures, under review

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 탐색 능력, 특히 열려 있는 작업에서 인간을 초월할 수 있는지를 조사합니다. 연구는 Little Alchemy 2라는 게임을 사용하여 LLM의 탐색 전략을 분석하는데, 이 게임은 에이전트가 기존의 요소를 조합하여 새로운 것을 발견하는 과제를 제공합니다. 결과적으로 기존의 LLM들은 인간들과 비교할 때 대체로 성능이 떨어지는 것으로 나타났으며, LLM들은 불확실성 기반의 전략에 의존하는 경향이 있음을 보여줍니다.

- **Technical Details**: Little Alchemy 2게임은 기본 요소인 물, 불, 지구, 공기를 조합하여 새로운 요소를 발견하는 것을 목표로 합니다. 데이터는 29,493명의 인간 참가자와 4,691,033회의 시도를 통해 수집되었으며, 이 실험에서 4개의 LLM이 평가되었습니다. 실험에서는 다양한 샘플링 온도를 설정하여 LLM의 탐색 동태를 분석하였으며, 특히 힘과 불확실성의 역할을 결정하는 데 중점을 두었습니다.

- **Performance Highlights**: 대부분의 LLM은 인간의 탐색 능력에 미치지 못했으며, 오직 o1 모델만이 두드러진 성능을 보였습니다. LLM들은 탐색 중 이전의 경험이나 가능성을 적절히 고려하지 않고 너무 빠르게 결정을 내려 효과적인 탐색을 방해할 수 있는 것으로 밝혀졌습니다. 이 연구는 LLM의 제한 사항을 명확히 하여 더 적응력이 뛰어난 인공지능 시스템을 구축하기 위한 향후 논의의 방향을 제시합니다.



### Investigating the Monte-Carlo Tree Search Approach for the Job Shop Scheduling Problem (https://arxiv.org/abs/2501.17991)
- **What's New**: 이 논문에서는 Job Shop Scheduling Problem (JSSP)의 해결에 있어 Monte Carlo Tree Search (MCTS) 기법의 잠재력을 탐구합니다. 특히, recirculation이 있는 대규모 JSSP에 초점을 맞추어 작업 완료 시간을 최소화하는 것을 목표로 합니다. 또한, 실제 제조 데이터를 기반으로 한 새로운 합성 벤치마크(synthetic benchmark)를 도입하여 비정형(instance) 문제의 복잡성을 포착합니다.

- **Technical Details**: MCTS는 강화 학습(reinforcement learning) 기법 중 하나로, JSSP를 해결하기 위해 여러 Markov Decision Process (MDP) 모델링을 제안합니다. MDP는 의사결정 과정에서 상태(state)와 행동(action) 간의 관계를 정의하여 최적의 정책(policy)을 찾는 데 유용합니다. 이 연구는 대규모 JSSP 문제를 해결하는 데 있어 MCTS의 적용 가능성을 구체적으로 보여줍니다.

- **Performance Highlights**: 실험 결과, MCTS는 대규모 JSSP 사례에서 뛰어난 품질의 솔루션을 생성하며, 전통적인 제약 프로그래밍(constraint programming) 접근 방식보다 우수한 성능을 나타냅니다. 이러한 결과는 MCTS가 현실 세계의 복잡한 문제 해결에 효과적일 수 있음을 시사합니다.



### Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization (https://arxiv.org/abs/2501.17974)
- **What's New**: 이번 연구에서는 대형 언어 모델이 수학 문제를 해결하는 능력을 발전시키기 위해 Inference Budget-Constrained Policy Optimization (IBPO) 알고리즘을 제안했습니다. IBPO는 모델이 쿼리의 난이도를 이해하고 어려운 쿼리에 인퍼런스 예산을 할당하도록 학습하게 합니다. 이 방법을 통해 MATH500 데이터셋에서 LLaMA3.1 8B Instruct 모델 대비 4.14% 및 5.74%의 절대 개선을 이끌어냈습니다.

- **Technical Details**: 연구에서 소개된 두 가지 주요 응답 방식인 Sequential Voting (SV)와 Adaptive Sequential Voting (ASV)은 모델의 응답을 위한 체계적인 구조를 제공합니다. SV는 단일 모드 생성 방식으로, 모든 응답이 정해진 규칙에 따라 생성되도록 합니다. 반면 ASV는 출력 응답이 쿼리의 유형에 따라 적절히 결정되도록 하여 자원 할당을 최적화합니다.

- **Performance Highlights**: IBPO로 최적화된 ASV는 주어진 인퍼런스 예산을 더욱 효율적으로 할당할 수 있는 능력을 보여주었습니다. 초기 실험 결과, ASV는 인퍼런스 비용을 줄이면서도 성과를 높였음을 시사합니다. ASV는 초기 응답을 기반으로 하여 결과의 다양성을 높이고, 모델이 동적으로 자원을 할당할 수 있도록 합니다.



### DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights (https://arxiv.org/abs/2501.18596)
- **What's New**: DeltaLLM을 소개하며 이는 LLM의 메모리 발자국을 줄이기 위한 새로운 사후 훈련 압축 기법이다. 우리는 변환기 블록 간의 가중치 공유 및 저순위 차이 행렬을 통해 LLM의 구조를 대안적으로 제안하였다. DeltaLLM은 30M-40M 토큰으로 경량 훈련을 수행하며 기존 LLM과 유사한 성능을 달성할 수 있음을 보여준다.

- **Technical Details**: DeltaLLM은 여러 가지 모델 압축 기술을 체계적으로 탐구하며, 가중치 공유, 저순위 적응 및 점진적 모듈 교체 방식을 포함한다. 이 접근 방식을 통해 DeltaLLM은 메모리 요구 사항을 크게 줄이면서도 경쟁력 있는 성능을 유지한다. 원래 모델의 지식을 활용하여 저순위 행렬을 훈련하고, 부드러운 레이어 대체를 위해 이전 연구에서 효과가 입증된 점진적 모듈 교체 전략을 따른다.

- **Performance Highlights**: DeltaLLM은 Phi-3.5 및 Llama-3.2 모델에서 압축을 수행하여 기존의 SLM보다 더 나은 성능을 기록하였다. DeltaPhi 2.9B는 24%의 매개변수 축소에도 불구하고 유사한 평균 제로샷 정밀도를 달성하며, 모델 크기를 감소시키면서도 원래 모델의 성능을 복원하는데 효과적이다. 우리의 연구는 저장 공간이 중요한 경우 LLM 아키텍처 설계와 압축 기법에 대한 새로운 통찰력을 제공한다.



### Diffusion Autoencoders are Scalable Image Tokenizers (https://arxiv.org/abs/2501.18593)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 간단한 확산 토크나이저(Diffusion Tokenizer, DiTo)를 소개합니다. 이 토크나이저는 이미지 생성 모델을 위한 컴팩트한 시각적 표현을 학습하는데 초점을 맞추고 있습니다. 기존의 복잡한 비지도 모델 대신, 단일한 학습 목표인 확산 L2 손실(diffusion L2 loss)을 활용하여 확장 가능한 이미지 토크나이저를 효율적으로 훈련할 수 있다는 것이 주요 통찰입니다.

- **Technical Details**: DiTo는 이미지 생성을 위한 잠재 표현 학습을 위한 새로운 접근 방식을 제시합니다. 이는 확산 모델(difussion model)의 이론적 근거를 바탕으로 하여, 단일한 확산 L2 손실을 사용해 훈련하게 됩니다. 제안된 기술은 토크나이저 학습에서 여러 손실 조합을 요구하는 기존 방법들과 대비되어 보다 간단하고 효율적인 모델 학습을 가능하게 합니다.

- **Performance Highlights**: DiTo는 이미지 재구성과 다운스트림 이미지 생성 작업에서 기존 최첨단 모델과 비교하여 경쟁력 있는 혹은 더 나은 품질을 달성합니다. 특히 작은 텍스트나 기호, 구조적 비주얼 부분 처리에서 기존의 GLPTo보다 더 뛰어난 결과를 보여줍니다. 또한, 모델 크기를 늘려도 손실 하이퍼파라미터 조정이 필요 없어 간편하게 확장 가능하다는 장점을 가지고 있습니다.



### Advances in Multimodal Adaptation and Generalization: From Traditional Approaches to Foundation Models (https://arxiv.org/abs/2501.18592)
Comments:
          Project page: this https URL

- **What's New**: 본 연구는 다중 모드(domain adaptation) 적응 및 일반화(multimodal generalization)에 대한 최근의 발전 사항을 포괄적으로 정리했습니다. 전통적인 접근 방식에서부터 다중 모드 기초 모델까지 다양한 방법론을 다루며, 아카이브(arXiv)에서 활발히 업데이트되고 있는 자료들을 포함합니다. 각 주제에 대한 문제 정의와 기존 방법들에 대한 철저한 검토가 이루어지고 있어, 향후 연구 방향에 대한 통찰도 제공합니다.

- **Technical Details**: 다중 모드 적응 및 일반화를 위한 다양한 알고리즘이 제안되었으며, 특히 시각-언어(target domains), 오디오-비디오(audio-video) 및 LiDAR-카메라(LiDAR-camera)와 같은 데이터 소스로부터 출발합니다. 이 논문에서는 MMDA(Multimodal Domain Adaptation), MMDG(Multimodal Domain Generalization), 그리고 MMTTA(Multimodal Test-Time Adaptation) 등의 주요 개념들과 함께 이들을 향상시키기 위한 기초 모델들의 역할도 탐구합니다. 더불어, 각 방법에 대한 체계적인 분석과 기존 데이터셋, 응용 분야를 정리하였습니다.

- **Performance Highlights**: MMDA와 MMDG는 최근 행동 인식(action recognition)과 의미 분할(semantic segmentation) 분야에서 눈에 띄는 성과를 내고 있습니다. 특히, 다중 모드 기초 모델의 활용을 통해 모델들의 일반화 성능이 향상되는 것을 확인하였습니다. 이 연구는 기존 알고리즘의 이해를 도모하며, 다중 모드 적응 수준에서의 보완 정보를 효과적으로 활용하기 위한 새로운 방향을 제시합니다.



### Inkspire: Supporting Design Exploration with Generative AI through Analogical Sketching (https://arxiv.org/abs/2501.18588)
Comments:
          Accepted to CHI 2025

- **What's New**: 최근 Text-to-Image (T2I) AI 모델의 발전에 따라 제품 디자이너들이 이를 작업에 적용하기 시작했습니다. 하지만 현재의 T2I 도구는 추상적인 언어를 해석하는데 어려움이 있으며, 디자인 고착화(design fixation)를 유발할 수 있는 사용자 경험 문제를 안고 있습니다. 이를 해결하기 위해 Inkspire라는 스케치 기반 도구를 개발하였으며, 이를 통해 디자이너들은 아날로그적 영감을 바탕으로 프로토타입을 디자인할 수 있는 피드백 루프를 형성할 수 있습니다.

- **Technical Details**: Inkspire는 디자인 프로세스에서 T2I 모델을 지속적으로 탐색하도록 장려하는 워크플로우를 제시합니다. 이 도구는 아날로지적 디자인 개념을 활용하여 추상 테마에서 기발한 아이디어를 도출하고, 디자이너가 AI의 현재 상태를 이해하며 생성된 디자인 위에 구조화된 스케치를 직접 덧붙일 수 있게 합니다. 이를 통해 디자이너는 새로운 아이디어를 탐색할 수 있는 유연한 상호작용을 경험할 수 있습니다.

- **Performance Highlights**: Inkspire를 사용한 사용자들은 ControlNet에 비해 더 많은 영감과 탐색을 경험하였으며, 디자인 생성 프로세스가 더 협력적(co-creative)으로 느껴졌습니다. 연구 결과, Inkspire는 디자이너가 AI와의 협력, 제어 가능성, 커뮤니케이션 및 창작물에 대한 기여도를 크게 향상시켰습니다. 이를 통해 디자인 고착화를 피하고 더 많은 창의적 가능성을 탐구할 수 있게 도움을 주었습니다.



### R.I.P.: Better Models by Survival of the Fittest Prompts (https://arxiv.org/abs/2501.18578)
- **What's New**: 본 논문에서는 데이터 품질이 모델 성능에 미치는 영향을 측정하는 Rejecting Instruction Preferences (RIP) 방법을 제안합니다. 이 방법은 저품질 프롬프트가 높은 변동성과 낮은 품질의 응답을 초래한다는 가정하에 데이터 무결성을 평가합니다. RIP는 기존의 훈련 데이터에서 프롬프트를 필터링하거나 고품질의 합성 데이터셋을 생성하는 데 사용됩니다.

- **Technical Details**: RIP 방법은 선택된 응답과 거부된 응답 쌍을 기반으로 프롬프트를 필터링하며, 거부 응답 품질과 선택된 응답과의 보상 차이를 주요 메트릭으로 삼습니다. 본 방법은 Direct Preference Optimization (DPO)와 같은 강화학습 방법을 사용하여 모델을 미세 조정하는 데 활용되며, 저품질 프롬프트를 효과적으로 제거할 수 있습니다. 구체적으로, RIP는 Wildchat 프롬프트에 대해 필터링하여 Llama 3.1-8B-Instruct와 Llama 3.3-70B-Instruct 모델에서 성능 향상을 보였습니다.

- **Performance Highlights**: RIP는 AlpacaEval2 LC Win Rate를 9.4%, Arena-Hard를 8.7%, WildBench를 9.9% 향상시켰습니다. Llama 3.3-70B-Instruct 모델에서는 Arena-Hard에서 순위가 18위에서 6위로 올라가는 성과를 보였습니다. 이러한 성과는 RIP 필터링 방법이 모델 성능을 획기적으로 향상시킬 수 있음을 보여줍니다.



### Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling (https://arxiv.org/abs/2501.18577)
- **What's New**: 이 논문에서는 머신러닝 모델의 예측 결과를 통계 분석에 활용할 때 발생할 수 있는 오류를 수정하는 새로운 방법을 제안합니다. 특히 Predict-Then-Debias 추정기 방법을 확장하여 bootstrap 신뢰구간을 도입하여 비균일 데이터 샘플에서도 적용 가능하도록 하였습니다. 이를 통해 머신러닝 모델의 정확성에 대한 가정 없이도 유효한 신뢰구간을 생성할 수 있습니다.

- **Technical Details**: 저자들은 머신러닝을 통해 손실된 변수를 보완하는 새로운 신뢰구간을 개발했습니다. 예측 결과가 부분적으로 관측된 데이터와 결합될 때 데이터 샘플의 종류와 관계없이 예측 오류를 보정할 수 있는 강력한 접근 방식을 제공합니다. 이 연구에서는 p차원 데이터 X를 관측 데이터와 결측 데이터로 나누어 처리하며, 머신러닝 예측을 통한 대체 방법과 그 효과를 설명합니다.

- **Performance Highlights**: 제안된 방법은 머신러닝 모델의 성능과는 무관하게 신뢰구간을 생성할 수 있어 강력한 이점을 제공합니다. 기존 방법들보다 신뢰구간이 넓지 않으며, 실질적인 성과를 입증하는 다양한 실험 결과를 포함합니다. 이러한 접근법은 다양한 통계 분석 작업과 데이터 세트에 적용이 가능하여 유용한 도구가 될 것입니다.



### BounTCHA: A CAPTCHA Utilizing Boundary Identification in AI-extended Videos (https://arxiv.org/abs/2501.18565)
Comments:
          22 pages, 15 figures

- **What's New**: 최근 인공지능(AI)의 발전으로, 인간의 주목을 끌 수 있는 새로운 CAPTCHA 방식인 BounTCHA를 제안합니다. 이 시스템은 비디오의 전환 지점을 인간이 인식할 수 있도록 설계되어, 봇의 공격에 효과적으로 대응할 수 있습니다. BounTCHA는 AI를 활용하여 생성된 비디오를 기반으로 하며, 기존 CAPTCHA의 잦은 취약점을 해결하는 데 기여할 것으로 기대됩니다. 또한, 모든 관련 자료는 GitHub에서 오픈소스로 제공되고 있습니다.

- **Technical Details**: BounTCHA는 비디오 프레임의 급격한 변화 및 전환에 대한 인간의 인식을 활용하여, 사용자에게 전환 지점을 식별하도록 요구합니다. 이는 사용자가 비디오를 시청한 후, 예상되는 전환 지점으로 프로그래스 바를 드래그하여 답변을 제출하는 형식입니다. 연구 질문(RQs)을 통해 이 시스템의 실용성, 인간의 경계 인식 능력, 공격 가능성을 평가하며, 다양한 공격 방법에 대한 보안 분석도 진행됩니다.

- **Performance Highlights**: BounTCHA는 랜덤 공격, 데이터베이스 공격, 멀티모달 LLM 공격 등 다양한 위협에 대해 효과적으로 방어할 수 있음을 입증했습니다. 실험을 통해 인간의 경계 인식 능력의 시간 편향 범위를 측정하여, 이 시스템의 feasible한 CAPTCHA의 가능성을 보여주고 있습니다. 이러한 연구는 온라인 보안 강화에 기여하며, AI 주도 시대에서 수많은 웹 애플리케이션을 보호하는 데 중요한 역할을 할 것입니다.



### Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method (https://arxiv.org/abs/2501.18539)
- **What's New**: 이 논문은 복잡한 질문에 대한 정보를 검색하는 데 있어 LLM 기반의 새로운 retrieval 방법인 ARM을 제안합니다. ARM은 질문과 데이터 수집의 조직 간의 정렬을 메꿔, 기존의 검색 방법이 간과할 수 있는 필요한 정보에 대한 더 나은 탐색을 가능하게 합니다. 특히, 반복적인 쿼리가 아닌 문맥에 맞는 데이터 오브젝트 간의 관계를 분석함으로써 효율성을 극대화할 수 있습니다.

- **Technical Details**: ARM은 LLM(대형 언어 모델)의 추론 능력을 활용하여 질문에 필요한 데이터를 효율적으로 검색하는 방식으로 설계되었습니다. 이 시스템은 정보 정렬(information alignment)과 구조 정렬(structure alignment) 단계를 통해 데이터를 정리하며, 자가 검증(self-verification) 과정을 통해 최종적으로 적합한 데이터 오브젝트를 선택합니다. 이를 위해 N-gram을 사용하여 각 오브젝트의 주요 정보를 요약하고, 임베딩(embedding)을 통해 의미적 유사성 검색을 지원합니다.

- **Performance Highlights**: 실험 결과 ARM은 Bird 데이터셋에서 기존의 RAG 방법들보다 최대 15.9 포인트의 정확도로 우수한 성능을 보였습니다. OTT-QA 데이터셋에서도 ARM은 이전 방법들에 비해 최대 19.3 포인트 높은 F1 점수를 기록하며, LLM 기반의 질문 대응 문제에서 효과적인 해결책을 제시합니다. 이로 인해 복잡한 질문에 대한 검색 능력이 한층 향상되었습니다.



### A Hybrid Data-Driven Approach For Analyzing And Predicting Inpatient Length Of Stay In Health Centr (https://arxiv.org/abs/2501.18535)
Comments:
          8 pages, 15 figures

- **What's New**: 이번 연구에서는 병원 경영의 효율성을 평가하는 중요한 지표인 환자의 병원 체류 기간(LoS) 최적화를 위한 포괄적인 프레임워크를 제안합니다. 데이터 기반 기법과 시뮬레이션 방법론을 통합하여 230만 건의 비동의 환자 기록을 분석하였으며, 이를 통해 환자의 흐름 최적화 및 자원 활용 극대화를 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 머신러닝 모델(Decision Tree, Logistic Regression, Random Forest, Adaboost, LightGBM)과 Python 도구(Spark, AWS clusters, 차원 축소)를 활용하여, 환자가 입원 시 예측되는 병원 체류 기간(LoS)을 분석했습니다. 감독 학습 알고리즘을 사용하여 LoS에 영향을 미치는 주요 요인을 식별하고, 이를 통해 환자 흐름과 자원 활용을 개선할 수 있는 강력한 프레임워크를 제공합니다.

- **Performance Highlights**: 연구 결과, 실제 의료 환경에서 환자의 병원 체류 기간이 감소하는 효과가 나타났습니다. 하이브리드 데이터 기반 모델이 병원 경영 관행을 혁신할 가능성을 강조하며, 이는 의료 행정과 환자 만족도 제고에 큰 의미를 가질 것으로 기대됩니다.



### CLEAR: Cue Learning using Evolution for Accurate Recognition Applied to Sustainability Data Extraction (https://arxiv.org/abs/2501.18504)
Comments:
          9 pages plus 2 pages of supplemental material

- **What's New**: 이번 논문에서는 이미지에서 데이터 추출을 위한 효과적인 도구인 대형 언어 모델(Large Language Model, LLM) 이미지 인식의 정확도를 높이기 위한 새로운 기법, Cue Learning using Evolution for Accurate Recognition (CLEAR)를 소개합니다. 이 방법은 진화 계산(evolutionary computation)과 LLM의 조합을 활용하여 이미지 내의 전문적인 특징 인식을 향상시키기 위한 단서(cue)를 생성하고 최적화합니다. 이는 도메인 특정 표현 생성과 유전 알고리즘(genetic algorithm)을 통한 적절한 텍스트 단서 최적화를 통해 이루어집니다.

- **Technical Details**: CLEAR는 변동 길이 표현(variable-length representation)을 활용하여 고정 길이 표현(fixed-length representation)과 비교하여 인식의 정확도를 높이는 방법을 연구합니다. 또한, 카테고리 기반 추정(category-based estimates)에서 실수 값(real-valued estimates)으로 리팩토링(refactoring)함으로써 LLM의 일관성을 개선하는 방법에 대해 논의합니다. 이러한 접근은 특정 도메인에 맞는 새로운 표현을 자동 생성하고, 이에 따라 맞춤형 텍스트 단서를 최적화하는 과정을 포함합니다.

- **Performance Highlights**: CLEAR는 빌딩의 내부 및 외부 이미지에서 지속 가능성 데이터(sustainability data)를 인식하는 실제 작업에 적용되었으며, 전문가의 인식 및 사람의 작성된 단서보다 모든 작업에서 더 높은 정확도를 달성하는 결과를 보였습니다. 오류율(error rates)은 최대 두 자릿수 개선을 보였으며, 삭감 연구(ablation study)를 통해 솔루션의 간결성을 입증했습니다.



### Beyond Prior Limits: Addressing Distribution Misalignment in Particle Filtering (https://arxiv.org/abs/2501.18501)
- **What's New**: 본 논문에서는 Prior Boundary Phenomenon (PBP)이라는 새로운 개념을 도입하여, 초기 prior 분포에 의해 입자 필터링의 제대로 된 상태 추정이 제한되는 현상을 분석합니다. 기존의 방법들이 동적 상황에서 그 효과가 제한적임을 지적하며, Diffusion-Enhanced Particle Filtering Framework (DEPF)를 제안합니다. 이 프레임워크는 입자 필터가 prior 경계를 넘어서 탐색할 수 있도록 세 가지 주요 혁신을 포함합니다.

- **Technical Details**: DEPF 프레임워크는 탐색적 입자를 통한 적응적 확산, 엔트로피 기반 규제, 그리고 동적 지원 확장을 위한 커널 기반의 섭동으로 구성됩니다. 탐색적 입자는 넓은 상태 공간에서 샘플링되어 prior 경계를 넘어 탐색할 수 있게 하며, 엔트로피 기반 규제는 무게 붕괴를 방지합니다. 커널 기반 섭동은 입자의 위치를 불확실하게 변동시켜, 목표 상태 공간을 더 잘 커버할 수 있도록 합니다.

- **Performance Highlights**: 이론적 분석 및 다양한 실험을 통해 DEPF 프레임워크의 효과성을 뒷받침하고, 높은 차원 및 비볼록 시나리오에서 성공률과 추정 정확도가 유의미하게 향상됨을 입증합니다. DEPF는 기존의 한계점을 극복하고 robust한 상태 추정을 가능하게 하여, 안전 및 복잡한 시나리오에서의 활용 가능성을 높입니다.



### GuardReasoner: Towards Reasoning-based LLM Safeguards (https://arxiv.org/abs/2501.18492)
Comments:
          22 pages, 18 figures

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 안전성을 위한 새로운 안전장치인 GuardReasoner를 제안합니다. 기존의 guard models가 갖는 성능, 설명 가능성 및 일반화의 한계를 극복하는 것을 목표로 하며, reasoning SFT와 hard sample DPO 기법을 활용하여 모델의 추론 능력을 강화합니다.

- **Technical Details**: GuardReasoner의 훈련 과정은 두 단계로 나뉘며, 첫 번째 단계에서 GuardReasonerTrain 데이터셋을 구축하여 하루에 약 127K 샘플과 460K의 구체적인 추론 단계를 생성합니다. 두 번째 단계에서는 HS-DPO(하드 샘플 직접 선호 최적화)를 통해 모델이 잘못된 출력에 집중하고, 정확한 출력을 보완하는 방향으로 훈련을 진행하여 추론 능력을 한층 향상시킵니다.

- **Performance Highlights**: GuardReasoner는 13개의 벤치마크에서 실험을 통해 성능을 입증하였으며, GuardReasoner 8B 모델은 GPT-4o+CoT보다 5.74%, LLaMA Guard 3 8B보다 20.84% 더 높은 F1 점수를 기록했습니다. 또한, 이 모델은 다양한 크기(1B, 3B, 8B)로 공개되어 다른 연구자들이 접근할 수 있도록 했습니다.



### Curriculum-based Sample Efficient Reinforcement Learning for Robust Stabilization of a Quadrotor (https://arxiv.org/abs/2501.18490)
Comments:
          8 pages, 7 figures

- **What's New**: 최근 인공지능(AI) 방법론은 무인 항공기(UAV) 제어기 개발에 많은 영향을 미치고 있으며, 강화 학습(Reinforcement Learning, RL)이 특히 쿼드로터 제어에 있어 중요하게 부각되고 있습니다. 본 논문에서는 사전 정의된 성능 기준을 충족하는 안정적인 쿼드로터 제어기를 개발하기 위해 커리큘럼 학습(Curriculum Learning) 접근 방식을 소개합니다. 이 방법은 무작위 초기 상태에서 원하는 위치를 달성하기 위해 세 단계의 학습 목표를 체계적으로 나누어 설정합니다.

- **Technical Details**: 쿼드로터는 세 개의 변환적 자유도((x,y,z)∈ℝ3)와 세 개의 회전적 자유도(ϕ,θ,ψ∈𝕊1×𝕊1×𝕊1)를 가진 저감작동 공중 시스템입니다. 본 논문에서는 Crazyflie 2.0 모델을 사용하며, RL 정책 훈련을 위해 동적 모델링 방법을 적용합니다. 제안된 세 단계 커리큘럼은 고정 초기 위치에서의 안정적인 호버링 학습부터 시작하여 최종 목표에 도달하기까지 점진적으로 난이도를 높이는 방식으로 구성되어 있습니다.

- **Performance Highlights**: 제안된 PPO 기반의 커리큘럼 학습 접근 방식은 같은 보상 함수로 훈련된 단일 단계 PPO 정책에 비해 우수한 성능을 보였으며, 계산 자원 요구 사항과 수렴 시간을 상당히 줄였습니다. 또한, 커리큘럼으로 훈련된 정책의 성능과 강인성은 다양한 초기 조건과 외부 방해가 있는 상황에서도 검증되었습니다. 이는 기존의 RL 방법론에서 문제로 지적된 샘플 효율성 향상에 기여합니다.



### CLoQ: Enhancing Fine-Tuning of Quantized LLMs via Calibrated LoRA Initialization (https://arxiv.org/abs/2501.18475)
- **What's New**: 본 논문에서는 CLoQ (Calibrated LoRA initialization for Quantized LLMs)를 소개합니다. CLoQ는 quantized LLM에서의 LoRA 모듈의 초기화를 위한 데이터 기반 접근법으로, 작은 보정 데이터셋을 활용하여 레이어 간의 차이를 최소화하는 것을 목표로 합니다. 이 방법은 모델의 모든 레이어에서 최적의 LoRA 구성 요소를 결정하여 모델의 성능을 극대화합니다.

- **Technical Details**: CLoQ의 핵심은 두 가지 단계를 포함하는데, 첫 번째는 post-training quantization 단계로 quantized weights를 얻는 것이고, 두 번째는 linear transformation 하에 일반화된 low-rank 근사(th정확한 LoRA 구성 요소를 계산하는 것입니다. 이 과정에서 새로운 폐쇄형 해를 유도하였으며, 이는 두 개의 singular value decompositions (SVD)를 사용하여 효율적으로 계산됩니다. 이 방법은 역전파(back-propagation)를 필요로 하지 않음으로써 quantized 모델의 fine-tuning을 매우 효율적으로 합니다.

- **Performance Highlights**: CLoQ의 효과는 다양한 벤치마크 데이터셋을 통해 검증되었으며, 기존의 LoRA 방법들과 비교할 때 꾸준히 높은 성능을 나타냈습니다. 특히 ultra low-bit width에서 INT2 CLoQ는 Llama2-13B 모델에서 arithmetic reasoning 작업에서 INT4 QLoRA를 초과하는 fine-tuning 정확도를 기록했습니다. 이러한 결과를 통해 CLoQ는 quantized LLM의 성능을 향상시킬 수 있는 강력한 도구임을 입증하였습니다.



### Beyond Instructed Tasks: Recognizing In-the-Wild Reading Behaviors in the Classroom Using Eye Tracking (https://arxiv.org/abs/2501.18468)
Comments:
          24 pages, 16 figures, 6 tables, conference

- **What's New**: 이 연구에서는 교육적 맥락에서의 독서 행동을 이해하기 위해 자연 상태에서의 독서 데이터와 지시된 독서 데이터를 수집하였습니다. 이를 통해 독서 행동의 속도, 밀도, 연속성을 기반으로 한 독서 행동 분류를 위한 혼합 방법론을 개발했습니다. 또한 경량화된 2D CNN을 활용하여 행동 인식의 F1 점수 0.8을 달성하였으며, 이는 교사의 교육 효과를 위한 목표 설정에 기여할 수 있습니다.

- **Technical Details**: 연구에서는 인간 중심의 이론 모델, 통계 분석 및 AI 분류기를 포함한 혼합 방법론을 적용하여 독서 행동을 구분했습니다. 주요 초점은 지시된 환경과 자연 환경에서의 눈 추적 데이터 수집에 기초하여 수집한 독서 행동의 동적 변화를 분석하는 것이었습니다. 2D CNN을 통해 실시간으로 독서 행동을 분류할 수 있는 시스템을 개발하는 데 성공했습니다.

- **Performance Highlights**: 이 연구의 결과물 중 하나는 인스트럭션 기반 독서와 자연적인 독서 환경 간의 행동 차이를 밝혀냈습니다. 이는 독서 과정의 진정한 이해를 높이는 데 기여하며, 교육 의도에 따라 정보 제공의 방식이 달라질 수 있는 가능성을 보여줍니다. 새로운 데이터셋과 행동 분류하기 위한 프레임워크는 독서 연구 커뮤니티에서 향후 연구의 기초가 될 것으로 기대됩니다.



### Clustering Properties of Self-Supervised Learning (https://arxiv.org/abs/2501.18452)
- **What's New**: 자기 지도 학습(self-supervised learning, SSL) 방법은 표식 없음에도 불구하고 강력한 군집(cluster) 속성을 가진 의미론적으로 풍부한 표현을 포착하는 데 매우 효과적임을 보여주고 있습니다. 하지만 이러한 특성을 활용하여 SSL 방법을 향상시키려는 시도는 드물었습니다. 본 논문에서는 인코더의 출력이 다른 구성 요소와 비교하여 더 우수하고 안정적인 군집 속성을 가진다는 것을 다양한 지표를 통해 입증하였고, 이를 바탕으로 새로운 긍정적 피드백 SSL 방법인 Representation Soft Assignment (ReSA)를 제안합니다.

- **Technical Details**: ReSA는 모델의 군집 속성을 활용하여 자기 가이드를 통해 학습을 촉진하는 온라인 자기 군집(self-clustering) 메커니즘을 채택합니다. 이 방법은 인코더의 출력인 encoding에서 군집 속성을 추출하여 시너지 효과를 창출하며, 현재까지 발표된 최첨단 SSL 방법들과 비교하여 성능과 학습 효율성 모두에서 우수함을 보여주었습니다. 또한 ReSA가 군집 속성을 어떻게 개선하는지를 분석하여 세부적(fine-grained)이고 거시적(coarse-grained) 수준에서 표현을 더 구조적이고 의미론적으로 의미 있게 만든다는 것을 실증하였습니다.

- **Performance Highlights**: 다양한 표준 SSL 벤치마크에 대한 광범위한 실험 결과, ReSA로 사전 훈련된 모델은 다른 최신 SSL 방법들보다 현저하게 성능이 우수하다는 발견이 있었습니다. 특히 ReSA는 미세한 군집 인식을 통해 더 나은 성능을 유도하며, 이러한 구조적 표현의 강화는 의미론적 관계 포착에 기여합니다. 전반적으로, ReSA는 SSL의 발전에 중요한 기여를 할 것으로 기대됩니다.



### Autonomy and Safety Assurance in the Early Development of Robotics and Autonomous Systems (https://arxiv.org/abs/2501.18448)
Comments:
          7 pages, 2 figures

- **What's New**: 이 보고서는 2024년 9월 2일 맨체스터 대학교에서 열린 '로봇 및 자율 시스템의 조기 개발에서 자율성과 안전 보증'에 관한 워크숍에 대한 개요를 제공합니다. 이 행사에는 다양한 분야의 6개 규제 및 보증 기관 representatives가 모여 자율 검사 로봇(AIR)의 안전성을 보장하기 위한 도전과제와 증거를 논의했습니다. CRADLE은 신뢰성 있는 자율 시스템 공학에서 보증을 핵심 요소로 만들기 위해 노력하고 있습니다.

- **Technical Details**: 워크숍에서는 AIR의 안전성을 보장하는 데 있어 세 가지 주요 연구 질문에 대해 논의했습니다. 첫 번째 질문은 AIR의 안전성 보증에서의 도전 과제로, 인간-로봇 상호작용과 AI의 의사결정 이해를 위한 투명성과 설명 가능성의 필요성이 강조되었습니다. 두 번째 질문은 안전성 보증을 위한 증거로서 다양한 V&V 방법이 필요하며, 인간의 감독과 개입 가능성에 대한 증거도 요구된다고 설명되었습니다. 마지막으로, 자율 시스템을 위한 보증 사례는 다이내믹한 환경에서의 안전 메커니즘이 중요하다는 점이 강조되었습니다.

- **Performance Highlights**: 이번 워크숍을 통해 산업계, 학계 및 규제 기관 representatives가 자율성 보증에 관한 도전 과제를 논의할 수 있는 귀중한 기회를 제공했습니다. 참가자들은 규제 기대를 충족시키기 위해 디자인-포-어슈어런스(Design-for-Assurance) 프로세스 도입에 대한 강한 의지를 보였습니다. CRADLE은 앞으로 반복 가능한 보증 패턴 개발을 통해 이 기초 위에 지속적으로 발전할 계획을 가지고 있습니다.



### o3-mini vs DeepSeek-R1: Which One is Safer? (https://arxiv.org/abs/2501.18438)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2501.17749

- **What's New**: 이번 연구에서는 DeepSeek-R1와 OpenAI의 o3-mini 두 가지 LLM에 대한 시스템 안전성 평가를 처음으로 시행하였습니다. DeepSeek-R1은 낮은 실행 비용으로 매우 우수한 성능을 제공한다는 점에서 AI 산업의 혁신으로 여겨집니다. 하지만 안전성과 인간의 가치와의 정렬은 필수적이며, 이 연구에서는 이러한 안전성을 평가하는 도구인 ASTRAL을 사용했습니다.

- **Technical Details**: ASTRAL은 LLM의 안전성을 자동으로 생성하고 평가하는 도구로, 총 1260개의 안전하지 않은 테스트 입력을 생성하여 DeepSeek-R1과 o3-mini의 반응을 분석했습니다. 이 도구는 다양한 안전 범주, 작문 스타일 및 설득 기법을 포괄하는 균형 잡힌 입력을 생성해, LLM이 보편적인 사용자 쿼리에 어떻게 반응하는지를 실증합니다. 결과적으로 DeepSeek-R1은 11.98%의 안전하지 않은 응답을 보인 반면, o3-mini는 단 1.19%로 나타났습니다.

- **Performance Highlights**: DeepSeek-R1은 OpenAI의 o3-mini와 비교했을 때 상대적으로 안전하지 않은 것으로 평가되었습니다. 이는 DeepSeek-R1이 자동화된 안전 테스트에서 더 많은 위험한 응답을 제공한다는 것을 의미합니다. 비록 두 모델 모두 사람들에게 중요한 영향을 미칠 것으로 예상되지만, DeepSeek-R1의 안전성 부족은 주요한 우려사항으로 간주됩니다.



### Solving Drone Routing Problems with Quantum Computing: A Hybrid Approach Combining Quantum Annealing and Gate-Based Paradigms (https://arxiv.org/abs/2501.18432)
Comments:
          8 pages, 5 figures. Paper submitted to IEEE Congress on Evolutionary Computation (IEEE CEC 2025)

- **What's New**: 새로운 연구에서는 드론 경로 문제를 해결하기 위해 양자 컴퓨팅의 가능성을 활용한 혁신적인 하이브리드 접근법인 Quantum for Drone Routing (Q4DR)을 제안합니다. 이 방법은 양자 게이트 기반 컴퓨팅과 양자 어닐러를 통합하여 실제 세계의 제약 조건을 반영한 세 가지 사용 사례를 통해 효과성을 입증합니다. Q4DR은 양자 근사 최적화 알고리즘(QAOA)과 D-Wave 시스템의 장치를 활용한 두 단계로 나누어진 알고리즘을 가지고 있습니다.

- **Technical Details**: 본 연구는 양자 컴퓨팅(QC)과 드론 라우팅 간의 상관관계를 설명합니다. 특히, Q4DR 방법론은 초기 클러스터링 단계에서 QAOA를 사용하고, 이어서 경로 설정 단계에서 양자 어닐러를 활용하는 두 단계로 구성되어 있습니다. Eclipse Qrisp라는 프로그래밍 언어를 사용하는 것이 이 연구의 주요 특징이며, 이는 양자 알고리즘의 설계와 구현을 간소화함으로써 효율성을 높이고 있습니다.

- **Performance Highlights**: Q4DR은 실질적인 물류 문제의 해결을 위해 실제 제약 조건을 반영하여 설계되었습니다. 이 연구는 배터리 사용량, 금지 경로, 이동 충전 포인트 최적 선정 등 여러 요소를 고려하여 드론 경로 문제의 복잡성을 높였습니다. 따라서 Q4DR 방법론은 교과서적인 이론을 넘어 실용적인 응용 가능성을 제시하여, 양자 최적화 기술의 발전에 기여하고 있습니다.



### Guaranteed confidence-band enclosures for PDE surrogates (https://arxiv.org/abs/2501.18426)
- **What's New**: 이번 연구는 통계적으로 보장된 신뢰 구간(confidence bands)을 제공하는 새로운 방법을 제안합니다. 이 방법은 예측 오차의 저차원 표현을 활용하여 nested confidence sets를 구성하고, 이를 예측 공간으로 매핑하여 예측의 신뢰성을 향상시킵니다. 이러한 신뢰 구간은 functional surrogate model에 대해서도 적용 가능하여 AI의 신뢰성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 이 방법은 surrogate model의 예측 오차를 계산한 후, 차원 축소 기법(SVD)을 사용하여 저차원 공간으로 투영합니다. 그런 다음 zonotope이라는 집합을 기반으로 신뢰 지역(confidence regions)을 구성하여, 새로운 관측값이 주어진 신뢰 수준에서 포함될 것이라는 보장을 꾀합니다. 이 과정에서 truncation error를 제한하는 기법도 도입하여, 예측의 안전성을 강조합니다.

- **Performance Highlights**: 제안된 방법은 복잡한 Sci-ML 모델, Neural Operators 및 더 간단한 설정에서도 적용할 수 있는 모델 독립적(model agnostic) 특성을 가지고 있습니다. 실험 결과는 제안된 접근 방식이 통계적 보장을 제공하면서도 효율적인 예측을 가능하게 한다는 점을 보여주었습니다. 이러한 특성 덕분에 실제 위험 계산 및 안전-critical 시스템에 적용될 가능성이 높습니다.



### Efficient Transformer for High Resolution Image Motion Deblurring (https://arxiv.org/abs/2501.18403)
Comments:
          14 pages, 18 figures Submitted as a preprint, no prior journal/conference submission

- **What's New**: 본 연구는 고해상도 이미지 모션 디블러링을 위한 Restormer 아키텍처의 포괄적인 연구 및 개선을 제시합니다. 모델 복잡성을 18.4% 줄이면서 최적화된 attention 메커니즘을 통해 성능을 유지하거나 개선하는 아키텍처 수정 사항을 도입하였습니다. 새로운 훈련 파이프라인에는 색상 지터, 가우시안 블러 및 원근 변환과 같은 추가 변환이 포함되어 있으므로 모델의 강인성이 향상됩니다.

- **Technical Details**: Restormer는 고해상도 이미지 복원을 위해 특별히 설계된 효율적인 Transformer 모델입니다. 이 모델은 멀티-Dconv 헤드 전이 주의 메커니즘(multi-Dconv head transposed attention mechanism)과 게이트드 Dconv 피드 포워드 네트워크(gated-Dconv feed-forward network)를 도입하여 계산 효율성을 유지하면서 장거리 의존성을 모델링합니다. Restormer는 전반적인 이미지 관계를 모델링하면서도 큰 공간 해상도에 대해 계산 비용을 피할 수 있도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 RealBlur-R, RealBlur-J 및 Ultra-High-Definition Motion blurred (UHDM) 데이터셋에서 특성 평가를 실시하였으며, 개선된 아키텍처는 훈련 시간 단축과 더불어 경쟁력 있는 성능을 유지했습니다. 본 논문의 결과는 심층적인 ablation 연구와 함께 아키텍처 단순화 및 개선된 훈련 전략이 모션 디블러링 작업에 효율적이고 동등한 능력을 지닌 모델을 만들어낼 수 있음을 제시합니다.



### A Learnable Multi-views Contrastive Framework with Reconstruction Discrepancy for Medical Time-Series (https://arxiv.org/abs/2501.18367)
Comments:
          15 pages,6 figures

- **What's New**: 본 논문에서는 의료 시계열 데이터의 진단 정확도를 향상시키기 위해 두 가지 주요 문제를 해결하는 새로운 접근 방식을 제안합니다. 첫 번째로, AE-GAN을 활용하여 외부 데이터를 통합하고 모델 일반화를 위한 교차 센터 지식 전이를 수행합니다. 두 번째로, LMCF(변화 가능한 다중 관점 대조 프레임워크)를 소개하여 다양한 시점의 정보를 통해 자료를 효과적으로 학습할 수 있도록 합니다.

- **Technical Details**: 의료 시계열 데이터는 클리닉에서 수집한 데이터가 제한적이기 때문에 라벨이 없는 데이터를 사용하여 자가 감독(self-supervised) 대조 학습을 통해 모델을 사전 학습합니다. AE-GAN은 목표 데이터의 불일치를 재구성하여 질병 확률로 변환하며, 이를 대조 학습에 통합하여 목표 표현 학습을 개선합니다. LMCF는 다중 헤드 어텐션(mult-head attention) 메커니즘을 통합하여 인과 및 시간적 대조 학습을 통해 대표성을 높이는 방향으로 설계되었습니다.

- **Performance Highlights**: 세 가지 주요 데이터셋(심근경색, 알츠하이머병, 파킨슨병)에 대해 우리의 방법이 기존 7개 기준 모델을 지속적으로 초과하는 성능을 보여주었습니다. 특히, 데이터의 10%만 라벨이 있는 극단적인 상황에서도 우리의 방법이 우수한 성능을 발휘하는 것으로 나타났습니다. 따라서 본 연구 결과는 해당 분야의 다양한 의료 진단 응용에 중요한 의미를 가집니다.



### State Stream Transformer (SST) : Emergent Metacognitive Behaviours Through Latent State Persistenc (https://arxiv.org/abs/2501.18356)
Comments:
          25 pages, 3 figures

- **What's New**: 이번 논문에서는 State Stream Transformer(SST)라는 새로운 LLM 아키텍처를 소개합니다. 이 모델은 전통적인 transformer 모델의 한계를 극복하고, 자가 회귀적 생성 과정에서의 계산적 연속성을 유지함으로써, 학습된 가중치에서 잠재적인 추론 행동을 드러냅니다. SST는 슬라이딩 윈도우 방식의 잠재 상태(FFN) 캐시를 도입해 지속적인 잠재 과정의 유지 및 진화를 가능하게 합니다.

- **Technical Details**: SST는 모든 선형층에서 가중치를 감소시키면서 슬라이딩 윈도우 잠재 상태 캐시를 구현합니다. 이 구조는 토큰 생성 중에 지속적인 '상태 흐름'을 유지하여 모델이 정보를 처리하는 방식에 근본적인 변화를 가져옵니다. 이를 통해 SST는 자가 회귀 모델에 비해 향상된 추론 능력을 보여주며, 이는 메타인지 행동을 통해 증명됩니다.

- **Performance Highlights**: 정량적 평가 결과, SST는 GSM-8K(0-shot)에서 89.01%의 정확도와 ARC Challenge(0-shot CoT)에서 91.04%의 정확도로 기본 모델에 비해 상당한 성과 향상을 이뤘습니다. 이러한 결과는 잠재 상태에서의 지속적 계산이 정보 처리 및 내부 추론 전략에 근본적인 차이를 만든다는 것을 시사합니다. SST 아키텍처는 인공지능 시스템의 능력 및 인공지능 인지에 대한 이해에 중요한 함의를 가집니다.



### Transfer Learning of Surrogate Models: Integrating Domain Warping and Affine Transformations (https://arxiv.org/abs/2501.18344)
- **What's New**: 이 논문에서는 기존의 surrogate 모델을 새로운 작업으로 전이하는 접근 방식을 확장하여 다루고 있습니다. 특히, 선형 및 비선형 변환을 포함하는 더 넓은 범위의 변환을 다루며, 특히 β 누적 분포 함수(β CDF)를 사용하여 알려지지 않은 입력 왜곡(input warping)을 고려합니다. 이 모델은 적은 수의 데이터 포인트로 최적화를 수행하여 전이 데이터셋에서의 경험적 손실(empirical loss)을 최소화함으로써 트랜스퍼 학습을 실현합니다.

- **Technical Details**: 제안된 방법은 전이 학습을 통해 새로운 문제에 대한 정확한 surrogate 모델을 조정하는 것입니다. 이 과정에서는 선형 및 비선형 변환을 조합하여 도메인 전이를 신속하게 수행합니다. 특히, β CDF를 이용한 비선형 함수 g를 채택하여 자동차 산업의 복잡성을 모델링하며, 이로 인해 실험적으로 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 결과적으로, 제안된 모델은 원래의 surrogate 모델과 단순히 트랜스퍼 데이터셋을 사용하여 새로 구축한 모델보다 월등히 뛰어난 성능을 보였습니다. 특히 데이터가 부족한 상황에서도 더 정확한 예측을 제공하며, Black-Box Optimization Benchmark (BBOB) 테스트베드와 실제 자동차 산업의 전이 학습 과제에서 그 효과를 검증하였습니다. 결과적으로 이 모델은 실제 문제 해결에 있어 유망한 대안으로 주목받고 있습니다.



### Unfaithful Probability Distributions in Binary Triple of Causality Directed Acyclic Graph (https://arxiv.org/abs/2501.18337)
- **What's New**: 이번 논문에서는 세 개의 정점으로 이루어진 이항 인과 구조 다이렉티드 애시클릭 그래프(DAG)에서 발생할 수 있는 비순응성(Non-faithfulness) 확률 분포의 여러 예시를 제시하며, 이는 Robins et al. (2003)에서 설명한 인과 DAG에 대한 비순응성을 증명합니다. 또한, 구간 독립(conditional independence) 및 상태 독립(multiple independence)이 함께 이루어지는 비순응 확률 분포의 일반적인 가족을 설명하고 있습니다.

- **Technical Details**: 이 논문에서는 세 랜덤 변수 (X, Y, Z)에 대해 11개의 대표적인 DAG 구조가 문헌에 소개되고, 이 구조들이 비순응성에 대해 어떻게 영향을 미치는지에 대한 논의가 이루어집니다. 특히, Robins et al. (2003)와 Sadeghi (2017)의 연구를 기반으로 하여 3정점 이항 다각형(DAG)에서 비순응성 확률 분포의 예가 만들어집니다. 또한, 각 확률 분포는 Markov 호환성(Markov compatibility)을 만족하는지를 기반으로 조건부 독립 관계를 분석합니다.

- **Performance Highlights**: 이 논문에서 제시된 예시들은 비순응 확률 분포가 생성되는 다양한 구조를 효과적으로 나타내며, PC 알고리즘(principal component algorithm)과 같은 그래픽 방법을 통해 인과 추론(causal inference) 과정에서 나타나는 문제점을 철저히 탐구합니다. 후속 논문에서 다룰 더 많은 다각적인 예시와 결과들은 이 분야에서의 인과 구조를 더욱 명확히 하는 데 기여할 것으로 기대됩니다.



### CodeBrain: Impute Any Brain MRI via Instance-specific Scalar-quantized Codes (https://arxiv.org/abs/2501.18328)
- **What's New**: 이번 연구에서는 MRI (Magnetic Resonance Imaging) 임프테이션(imputation)을 위한 새로운 통합 모델 CodeBrain을 제안합니다. 이 모델은 다양한 뇌 MRI 임프테이션 시나리오에 적응할 수 있도록 설계되었습니다. CodeBrain은 다양한 모달리티 변환을 전체 모달리티 코드 예측 작업으로 변환하며, 이를 위해 두 가지 단계로 학습됩니다.

- **Technical Details**: CodeBrain은 두 단계로 구성된 훈련 과정을 사용합니다. 첫 번째 단계는 각 MRI 모달리티를 재구성하여 공유 잠재 공간으로 매핑하고, 이를 스칼라 양자화(scalar quantization) 처리하는 것입니다. 두 번째 단계에서는 사용자 정의 그레이딩 손실(customized grading loss)을 통해 무작위로 마스킹된 MRI 샘플로부터 전체 모달리티 코드를 예측합니다. 이를 통해 데이터의 고유한 특성을 보존하면서 다양한 무결점 모달리티 변환을 수행할 수 있습니다.

- **Performance Highlights**: CodeBrain 모델은 IXI 및 BraTS 2023와 같은 두 개의 공개 뇌 MRI 데이터세트에서 평가되었습니다. 실험 결과, CodeBrain은 기존 네 가지 방법보다 우수한 임프테이션 성능을 보여주며, 통합 뇌 MRI 임프테이션을 위한 새로운 최첨단 성능을 설정했습니다. 이러한 결과는 다양한 임프테이션 시나리오에 적합한 모델을 제공합니다.



### Efficient Neural Theorem Proving via Fine-grained Proof Structure Analysis (https://arxiv.org/abs/2501.18310)
- **What's New**: 이번 논문에서는 깊은 학습 모델과 전통적인 자동화 도구의 시너지를 통해 효율적인 신경 정리 증명기(Neural Theorem Provers, NTP)를 개발하는 새로운 방법인 ProofAug를 제안합니다. ProofAug는 모델이 생성하는 증명 제안의 세밀한 구조 분석을 통해 자동화 방식을 다양한 세부 수준에서 적용하여 샘플 효율성을 높입니다. 이 방법은 기존의 트리 검색 알고리즘과 원활하게 통합될 수 있는 유연성을 제공하여 효율적인 재귀 증명 모듈도 구성할 수 있게 합니다.

- **Technical Details**: ProofAug는 LLM의 증명 생성 과정에서 세밀한 구조 분석을 통해 증명 제안의 최대 호환 반증명(maximal compatible semi-proof, MCSP)을 찾습니다. 이 과정에서, ATP가 실패하는 경우에는 더 큰 반증명(coarse semi-proof)으로 재귀적으로 돌아가 문제를 해결하고 있습니다. 이러한 방식은 복잡한 초안 problemas를 줄이고, 복잡한 스마트 방법의 필요성을 감소시켜 자동화 방법의 효율성을 극대화합니다.

- **Performance Highlights**: ProofAug는 miniF2F-test 벤치마크에서 61.9%의 누적 합격률을 기록하였으며, 데이터 세트를 정제한 후 66.0%의 합격률을 달성하여 모든 증명 언어에서 새로운 SOTA를 세웠습니다. 대조적으로, 기존의 기법들과 비교하여 ProofAug는 샘플 비용이 적은 상황에서도 значные 성능 향상을 보여주었습니다. 이를 통해 이 방법은 신경 정리 증명 연구의 발전에 크게 기여할 것으로 기대되고 있습니다.



### A Comprehensive Analysis on Machine Learning based Methods for Lung Cancer Level Classification (https://arxiv.org/abs/2501.18294)
- **What's New**: 본 논문은 폐암 진단을 위한 머신러닝 (ML) 기법의 실질적 활용 가능성을 탐구합니다. 초기 진단의 안정성을 높이고, 모델 성능의 과적합 문제를 해결하기 위한 체계적인 분석이 진행됩니다. 다양한 머신러닝 모델이 비교되며, 목표를 더 정확히 파악할 수 있는 방법이 제시됩니다.

- **Technical Details**: 폐암의 다양한 단계 분류를 위해 XGBoost (XGB), LGBM, Adaboost, Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), CatBoost, k-Nearest Neighbor (k-NN) 등의 ML 모델을 체계적으로 실행하였습니다. 최소 자식 가중치(minimum child weight)와 학습률(learning rate)의 영향을 고려하며, 딥 신경망 (DNN) 모델을 통해 특성과 타겟 간의 상관관계를 분석합니다. 이를 통해 복잡한 패턴 식별에 대한 모델의 능력이 확립됩니다.

- **Performance Highlights**: 연구 결과, 여러 ML 모델이 폐암 단계 분류에서 높은 정확도를 달성할 수 있다는 주장이 제기됩니다. 특히, DNN 아키텍처의 복잡성에도 불구하고, 전통적인 ML 모델인 XGBoost, LGBM, Logistic Regression이 뛰어난 성능을 발휘하였습니다. 정확도, 정밀도(precision), 재현율(recall), F-1 점수(F-1 score) 등 다양한 비교 메트릭에서 뛰어난 예측 성능을 보이고 있습니다.



### Mining for Species, Locations, Habitats, and Ecosystems from Scientific Papers in Invasion Biology: A Large-Scale Exploratory Study with Large Language Models (https://arxiv.org/abs/2501.18287)
Comments:
          8 pages, 2 figures, accepted to the NLP4Ecology Workshop 2025 (this https URL) co-located with the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies

- **What's New**: 이 논문은 침입 생물학(invasion biology) 문헌에서 핵심 생태학적 개체(ecological entity)를 추출하기 위해 대형 언어 모델(large language models, LLMs)의 능력을 활용한 탐색적 연구를 제시합니다. 종 이름(species names), 위치(locations), 서식지(habitats) 및 생태계(ecosystems)에 대한 정보를 추출하는 것에 중점을 두고 있으며, 이는 종의 확산(spread) 이해와 향후 침입 예측, 보존(conservation) 노력에 필요한 정보입니다. 이 연구는 LLM을 활용한 생태학적 개체 추출의 가능성과 한계를 밝혀내고, 생물 침입 관리 및 이해를 위한 더 정교한 자동화된 지식 추출 도구의 기초를 다지기로 합니다.

- **Technical Details**: 전통적인 텍스트 마이닝(text mining) 접근 방식은 생태학적 용어의 복잡성과 이러한 텍스트에서 나타나는 미묘한 언어 패턴에 어려움을 겪는 경우가 많습니다. 이 논문은 도메인 특화된 미세 조정(fine-tuning) 없이 일반 목적의 LLM을 적용하여 생태학적 개체를 추출한 결과를 제시합니다. 이를 통해 LLM의 활용 가능성을 탐색하며, 생물학적 침입 이해를 위한 도구의 발전 방향을 제시합니다.

- **Performance Highlights**: 이 연구에서는 LLM을 이용한 정보 추출이 생태학적 텍스트에서 어떻게 효과적으로 이루어질 수 있는지를 보여주며, 이를 통해 향후 연구자와 실무자들이 생물 침입을 관리하고 이해하는 데 있어서 활용할 수 있는 기반을 마련합니다. LLM의 탐색 결과는 이 기술이 생태학에서의 정보 추출에 있어 잠재력을 가지지만 동시에 한계도 있음을 나타냅니다. 논문의 결과는 생태학적 연구 분야에서 LLM의 적용 가능성을 더욱 깊이 이해하는 데 기여할 것입니다.



### Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models (https://arxiv.org/abs/2501.18280)
- **What's New**: 최근 큰 언어 모델(LLMs)의 보안 이슈가 많은 주목을 받으면서, 해로운 출력을 방지하기 위한 다양한 방어 메커니즘이 개발되고 있습니다. 본 논문에서는 텍스트 임베딩 모델을 기반으로 한 안전장치가 이러한 방어의 기초가 됨을 발견하였습니다. 특히, 텍스트 임베딩 모델의 출력 분포가 큰 평균을 가지며 상당히 편향되어 있다는 관찰에서 출발해 보편적인 마법 단어(universal magic words)를 검색하는 새로운 효율적인 방법을 제안합니다.

- **Technical Details**: 보편적인 마법 단어는 텍스트 뒤에 붙여져 어떤 텍스트의 임베딩을 편향된 방향으로 이동시키고, 이는 두 텍스트 쌍의 유사성을 조작해 안전장치를 오도할 수 있습니다. 본 연구는 세 가지 방법을 사용하여 이러한 마법 단어를 발견하는 접근 방식을 설명합니다: 1) Brute-force search는 기준선으로 사용되며, 2) Black box 방법은 비대칭 방향과 유사한 텍스트 임베딩 단어를 찾아내고, 3) White box 방법은 임베딩을 편향 방향에 가깝게 만들 마법 단어 접미사를 찾습니다.

- **Performance Highlights**: 실험 결과, 세 가지 방법 모두 최적의 마법 단어를 찾는 것으로 나타났지만, Method 2와 Method 3은 Method 1보다 훨씬 더 효율적이었습니다. 특히, Method 3은 다중 토큰 마법 단어를 검색할 수 있어 유용성이 높습니다. 이를 통해 우리는 LLM 보안 시스템에서의 안전장치가 해로운 콘텐츠를 탐지하는 데 실패하는 취약점을 지적하며, 이러한 공격에 대한 방어 메커니즘도 제안했습니다.



### Pre-Trained Vision-Language Model Selection and Reuse for Downstream Tasks (https://arxiv.org/abs/2501.18271)
- **What's New**: 본 논문은 특정 downstream task에 적합한 최상의 VLM을 선택하는 문제를 다루며, Model Label Learning(MLL)이라는 새로운 패러다임을 제안합니다. 이 패러다임은 모델 레이블링, 모델 선택, 모델 재사용의 세 가지 주요 모듈로 구성되어 있습니다. MLL은 VLM을 더욱 효율적으로 사용할 수 있게 하여, 사용자가 다양한 task에 맞춤형 솔루션을 찾을 수 있도록 지원합니다.

- **Technical Details**: 제안된 MLL 방법론의 핵심은 후보 VLM을 모델 허브로 구성하여 각 VLM의 전문성과 유용성을 모델 레이블로 설명하는 것입니다. 모델 레이블링 과정에서는 세멘틱 그래프를 구성하여 각 모델을 테스트하고 레이블을 생성하며, 이를 통해 downstream tasks의 요구사항과 모델 레이블을 매칭하여 선택 과정을 수행합니다. 또, 앙상블 기법을 통해 선택된 모델들의 예측을 결합하여 최종 예측을 도출하는 독창적인 구조를 가지고 있습니다.

- **Performance Highlights**: MLL 방법은 49개의 VLM과 17개의 타겟 데이터셋을 포함하는 새로운 벤치마크를 도입하여 성능을 평가합니다. 실험 결과는 제안된 접근 방식이 VLM 선택 및 재사용에서 효과적임을 명확하게 입증하며, 모델 허브의 확장성도 검증되었습니다. 본 연구는 VLM의 실제적인 적용을 촉진할 수 있는 기초를 제공합니다.



### The iToBoS dataset: skin region images extracted from 3D total body photographs for lesion detection (https://arxiv.org/abs/2501.18270)
Comments:
          Article Submitted to Scientific Data

- **What's New**: 이번 연구는 피부암 진단을 위한 새로운 데이터셋 iToBoS를 소개합니다. 대부분의 공개 이미지 데이터셋이 단일 피부 병변에 중점을 두었다면, iToBoS는 3D 전체 신체 사진을 이용해 다양한 피부 영역에서 촬영된 16,954장의 이미지를 포함합니다. 이 데이터셋은 피부 상태와 관련된 다양한 메타데이터를 제공하여 AI 모델의 훈련을 지원합니다.

- **Technical Details**: 데이터셋 생성 과정은 데이터 수집, 주석 작성 및 공개 서브셋 선택의 세 단계로 나누어집니다. 각 이미지는 VECTRA WB360 시스템을 활용해 고해상도로 캡처되며, 주석은 질병을 나타내는 경계 상자로 제공됩니다. 각 이미지는 환자의 나이, 해부학적 위치, 햇빛 손상 점수와 같은 메타데이터와 결합되어 AI 모델이 병변과 건강한 피부를 구별할 수 있도록 돕습니다.

- **Performance Highlights**: iToBoS 데이터셋은 2024년 피부 병변 탐지 챌린지의 주요 요소로, AI 모델의 훈련과 평가를 위한 뛰어난 기회를 제공합니다. 이 대회는 피부병 진단 및 치료의 신속한 진행을 목표로 하며, 다양한 피부 병변을 탐지하는 최신 기계 학습 기법 개발을 촉진합니다. 결과적으로 이러한 노력은 피부암 조기의 발견과 진료 현장에서의 기술 배포 가능성을 높일 것으로 기대됩니다.



### MAMS: Model-Agnostic Module Selection Framework for Video Captioning (https://arxiv.org/abs/2501.18269)
Comments:
          Accepted to the AAAI 2025 Main Technical Track. This is an extended version of the original submission

- **What's New**: 이 논문에서는 동영상 캡셔닝에서 적절한 프레임 수를 선택하는 모듈 선택 프레임워크, 즉 Model-Agnostic Module Selection (MAMS) 프레임워크를 제안합니다. 이 프레임워크는 각 비디오에 맞는 캡션 생성 모듈을 선택하고, 중요 시각 토큰의 서브셋을 구성하여 캡션 성능을 향상시킵니다. 또한, 중요한 시각 토큰에 대한 주의를 높이는 새로운 적응형 어텐션 마스킹 기법도 도입하였습니다.

- **Technical Details**: MAMS 프레임워크는 세 가지 주요 모듈로 구성됩니다: 비디오 인코더, 텍스트 인코더, 및 캡션 생성 모듈입니다. 이 프레임워크는 특정 비디오에 적합한 크기의 캡션 생성 모듈을 선택하여 서로 다른 수의 프레임을 사용합니다. 이 과정은 주어진 비디오에서 시각 토큰을 추출하고, 선택된 모듈에 따라 필요에 맞는 시각 토큰의 서브셋을 구성함으로써 수행됩니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에서의 실험 결과, 제안한 MAMS 프레임워크는 SwinBERT, UniVL 및 mPLUG-2와 같은 최신 동영상 캡셔닝 모델의 성능을 크게 향상시켰습니다. 특히, mPLUG-2에 MAMS 프레임워크를 적용함으로써 새로운 최상위 성과 기준을 달성하였습니다. 해당 연구는 매우 동적인 비디오와 비슷한 정보가 적은 비디오를 모두 아우르는 적응형 접근법으로, 기존 방법의 한계를 효과적으로 극복합니다.



### PDE-DKL: PDE-constrained deep kernel learning in high dimensionality (https://arxiv.org/abs/2501.18258)
Comments:
          22 pages, 9 figures

- **What's New**: 이 논문은 PDE(부분 미분 방정식) 문제 해결을 위한 새로운 프레임워크인 PDE-제약 딥 커널 학습(PDE-DKL)을 제안합니다. PDE-DKL은 깊은 신경망(DNN)과 Gaussian 프로세스(GP)를 결합하여 높은 차원의 문제를 효과적으로 해결합니다. 이 방법은 불확실성 정량화를 포함하여 적은 데이터를 요구하면서도 높은 정확도를 제공합니다.

- **Technical Details**: PDE-DKL 프레임워크는 신경망이 고차원 PDE 문제의 저차원 잠재 표현을 학습하게 하여 문제의 복잡성을 줄입니다. 이후 Gaussian 프로세스는 주어진 PDE 제약 조건에 따라 커널 회귀를 수행하며, 데이터가 제한적일 때도 정확한 솔루션을 도출할 수 있습니다. 이 접근 방식은 DNN과 GP의 장점을 통합하여 고차원 PDE에 대한 강력한 불확실성 추정과 계산 효율성을 제공하는 것이 특징입니다.

- **Performance Highlights**: numerical experiments에 따르면, PDE-DKL은 적은 데이터 요구 조건으로 높은 정확도를 달성했습니다. 이 방법은 과학과 공학 분야의 복잡한 PDE 기반 응용 프로그램을 위한 실용적이고 신뢰할 수 있으며 확장 가능한 솔버로서의 가능성을 보여줍니다. 특히, 높은 차원의 문제에서의 성능은 기존 방법들에 비해 월등한 효과를 나타냅니다.



### Arbitrary Data as Images: Fusion of Patient Data Across Modalities and Irregular Intervals with Vision Transformers (https://arxiv.org/abs/2501.18237)
- **What's New**: 이번 연구에서 우리는 다중 모달 데이터를 통합하는 새로운 방법을 제안합니다. 이 방법은 다양한 생체 신호 및 처방된 약물 정보를 이미지 형태로 변환하여 Vision Transformer를 학습시키는 것입니다. 기존 연구들과는 달리, 이 접근법은 다중 모달을 단일 모델로 다룰 수 있도록 복잡도를 크게 감소시킵니다. 이를 통해 다중 모달 의료 AI의 진전을 이끌어 내길 기대합니다.

- **Technical Details**: 제안한 모델인 Vision Transformer for irregular sampled Multi-modal Measurements (ViTiMM)은 임상 매개변수, 약물 데이터, ECG 스캔 및 X선 사진 등을 이미지로 표현합니다. 이 과정에서 'visual prompt engineering'을 통해 여러 모달리티를 통합하여 모델링을 간소화합니다. 기존의 방식에서 각 모달리티는 개별적으로 최적화된 모델 아키텍처가 필요하였지만, 우리는 쉽게 데이터 처리를 통합할 수 있는 솔루션을 제공하고 있습니다.

- **Performance Highlights**: 우리의 연구 결과는 MIMIC-IV 데이터셋에서 6,175명의 환자를 분석한 결과 두 가지 벤치마크 과제에서 기존의 최첨단 방법들, MeTra와 MedFuse를 초월하는 성능을 보여주었습니다. 더욱이 이 모델은 해석 가능성을 높여주는 attention map을 시각화할 수 있어, 예측 결과에서 가장 영향력이 있는 입력 영역을 식별할 수 있습니다. 또한 연구 결과는 다중 모달 정보를 통합하여 모델 성능이 향상됨을 명확히 보여주고 있습니다.



### Exploring Large Protein Language Models in Constrained Evaluation Scenarios within the FLIP Benchmark (https://arxiv.org/abs/2501.18223)
- **What's New**: 이 연구에서는 FLIP 벤치를 확장하여, 최신 대형 단백질 언어 모델인 ESM-2와 SaProt의 성능을 평가합니다. FLIP은 제한된 데이터 가용성을 가진 상황에서 단백질 피트니스 예측 모델의 성능을 분석하는 데 중점을 둡니다. 이러한 환경에서 최근의 단백질 언어 모델의 발전이 성능 향상에 기여하는지에 대한 연구 결과는 특히 유용합니다.

- **Technical Details**: FLIP 벤치는 데이터 분할 전략인 'two vs many'와 'low vs high'를 사용하여 제한된 데이터 환경에서 단백질 피트니스 예측을 위한 다양한 활동들을 포함합니다. ESM-2와 SaProt 같은 최신 모델들은 자가 지도 Pretraining(사전 훈련)을 통해 대량의 레이블 없는 단백질 시퀀스 데이터를 활용하고, 이로 인해 모델이 낮은 데이터 조건에서도 잘 일반화될 수 있도록 합니다. SaProt는 구조 정보(Structural Information)를 활용하여 단백질 피트니스를 예측하는 방식으로, 모델링 과정에서 ESMFold를 통해 예측된 단백질 구조를 통합합니다.

- **Performance Highlights**: 연구 결과는 ESM-2가 모델 크기에 따라 학습 성능과 일반화 능력이 어떻게 달라지는지를 분석합니다. SaProt는 구조의 품질을 기반으로 한 예측 효과를 조명하며, 모델 간의 공정한 평가를 위해 학습, 검증, 테스트 세트의 구분을 철저히 유지합니다. 한마디로, 이 연구는 제한된 데이터 환경에서 단백질 피트니스 예측의 정확성과 모델 전반에 걸쳐 성능을 분석할 수 있는 기대감을 제공합니다.



### HKAN: Hierarchical Kolmogorov-Arnold Network without Backpropagation (https://arxiv.org/abs/2501.18199)
Comments:
          13 pages, 9 figures

- **What's New**: 이번 논문에서는 HKAN(Hierarchical Kolmogorov-Arnold Network)라는 새로운 네트워크 아키텍처를 제안합니다. 기존의 KAN(Kolmogorov-Arnold Network)와 달리, HKAN은 고정된 파라미터를 가진 무작위 학습 방식으로 학습하며, 손실 함수의 지역 최소값에 대한 민감성을 없애는 비반복적인 훈련 방법을 사용합니다. 이 방법은 기본 함수의 매개변수를 고정하고, 적은 수의 파라미터로도 비슷한 또는 더 나은 성능을 발휘하게 합니다.

- **Technical Details**: HKAN은 계층적 다중 스태킹( hierarchical multi-stacking ) 프레임워크를 활용하여 이전 층의 예측을 수정합니다. 각 층은 선형 회귀 문제를 해결함으로써 예측을 정제하며, 이를 통해 내부 연산이 단순화됩니다. 이러한 구조는 표준 최소제곱 회귀( least-squares regression ) 방법을 통해 빠른 계산이 가능하게 하며, 비선형 모델링에서도 높은 성능을 유지합니다.

- **Performance Highlights**: 경험적 결과에 따르면, HKAN은 다양한 회귀 작업에서 KAN에 비해 비교 가능한 또는 우수한 정확도와 안정성을 보여주고 있습니다. 또한 변수 중요성을 시각화하는 데 필요한 통찰력을 제공합니다. 이 연구는 이론적 통찰과 실용적 응용을 통합하여 신경망 모델링에 대한 강력하고 효율적인 대안을 제시합니다.



### In-Context Learning of Polynomial Kernel Regression in Transformers with GLU Layers (https://arxiv.org/abs/2501.18187)
- **What's New**: 이번 연구에서는 Transformer 모델의 in-context learning (ICL)에 대한 이론적 이해를 확장합니다. 특히, 기존의 linear self-attention (LSA) 메커니즘이 비선형 함수 클래스에서의 ICL에 적합하지 않음을 보여줍니다. 또한, GLU와 유사한 feed-forward 레이어를 결합하여 비선형 회귀 작업을 위한 경량화를 제안합니다.

- **Technical Details**: 저자들은 LSA가 선형 최소제곱 문제를 해결하기 위해 본질적으로 제한적이라는 점을 지적합니다. GLU-like feed-forward 레이어와 LSA를 결합한 Transformer 구조는 다항회귀에 대한 경량화된 기울기 하강법을 구현할 수 있음을 설명합니다. 그러나 서로 다른 ICL 작업에 대한 요구 사항은 embedding 차원이 입력 차원의 제곱에 비례해야 함을 강조합니다.

- **Performance Highlights**: 연구 결과는 수치 실험을 통해 validation되었으며, 특히 깊은 linear Transformer가 고차 다항식 목표 함수를 효과적으로 학습할 수 있는 사례를 제시합니다. 이러한 연구는 비선형 ICL의 도전 과제를 밝혀내고, 더 긴 프롬프트가 이러한 문제를 극복하는 데 필요할 수 있음을 시사합니다.



### Tensor Completion for Surrogate Modeling of Material Property Prediction (https://arxiv.org/abs/2501.18137)
Comments:
          2 page paper accepted to AAAI KGML 2025 bridge program

- **What's New**: 본 논문에서는 특정 재료 속성을 최적화하기 위한 설계 방법을 개선하기 위해 머신러닝(ML)을 활용하는 새로운 접근 방식을 제안합니다. 특히, 재료 속성 예측 작업을 텐서 완성(tensor completion) 문제로 모델링하여 대량의 재료 조합을 효과적으로 탐색할 수 있는 방법을 소개합니다. 이 방법론을 통해 블라인드 ML 모델보다 10-20% 낮은 오류율을 기록하며, 비슷한 훈련 속도를 유지할 수 있음을 실험적으로 입증하였습니다.

- **Technical Details**: 연구에서는 재료 데이터셋을 생성하기 위해 각 텐서 모드를 고유한 원소나 그 원소의 비율로 설정하고, 텐서 완성을 통해 드문 가치들에서 재료 속성을 추론하는 방법을 사용합니다. CPD(Canonical Polyadic Decomposition) 및 기타 텐서 완성 모델을 통해 주요 정보를 추출하고, 평균 절대 오차(MAE)를 계산하여 연구 결과를 분석합니다. 이러한 접근 방식은 화학식 기반 재료 속성 예측을 가능하게 하며, 데이터셋의 구조를 활용하여 빠르고 정확한 예측을 지원합니다.

- **Performance Highlights**: 실험 결과, 텐서 완성 모델은 다양한 예측 작업에서 평균 절대 오차(MAE) 기준으로 비 텐서 모델들보다 정기적으로 더 우수한 성능을 보였습니다. 특히, 총 자화 예측, 형성 에너지 및 밴드 갭 예측에서 두드러지는 성과를 보였습니다. 텐서 완성을 통해 고속으로 예측할 수 있으며, 기존 값의 작은 비율만으로도 관찰되지 않은 재료 값을 유추할 수 있다는 점에서, 재료 설계 문제를 현저히 가속화할 수 있는 가능성을 보여줍니다.



### Entropy-Synchronized Neural Hashing for Unsupervised Ransomware Detection (https://arxiv.org/abs/2501.18131)
- **What's New**: 이 논문에서는 Entropy-Synchronized Neural Hashing (ESNH)이라는 새로운 접근법을 소개하고 있습니다. 이 접근법은 소프트웨어 바이너리의 엔트로피 특성을 바탕으로 고유한 해시 값을 생성하여 악성 소프트웨어를 분류하는 방식입니다. ESNH는 신경망 아키텍처와 결합하여 폴리모픽 및 메타모픽 변화에도 안정적으로 대응할 수 있는 강력한 해시 값을 만들어냅니다.

- **Technical Details**: ESNH 프레임워크는 실행 파일 내의 엔트로피 분포를 분석하여 고유 식별자를 생성합니다. 엔트로피 프로파일이 신경망에 입력되어 특정 엔트로피 분포를 해시 표현과 연결시키며, 이를 통해 선행되지 않은 랜섬웨어 변종에 대해서도 안정적이고 독창적인 해시 값을 생성합니다. 해시 값의 일관성을 유지하기 위해 검증 메커니즘이 포함되어 있어, 동일한 바이너리에 대해 정확한 동작을 보장합니다.

- **Performance Highlights**: 실험 결과, ESNH 모델은 최신 랜섬웨어 변종에 대해 높은 탐지율을 보여주며, 기존의 정적 서명 기반 접근법에 비해 뛰어난 성능을 보였습니다. 또한, 고정밀도의 분류로 인해 잘 가려진 랜섬웨어 식별에 유리하며, 동적 변화에도 대응할 수 있는 잠재력을 입증했습니다. 이를 통해 ENTH는 전통적인 탐지 방법의 한계를 보완하고 안전성을 더욱 강화할 수 있는 가능성을 보여줍니다.



### Revisiting gender bias research in bibliometrics: Standardizing methodological variability using Scholarly Data Analysis (SoDA) Cards (https://arxiv.org/abs/2501.18129)
Comments:
          33 pg, 7 figures. Soda Cards: this https URL

- **What's New**: 이 연구에서는 학술적 데이터 분석에서 성별 편향을 평가하기 위해 'Scholarly Data Analysis (SoDA) Cards'라는 표준화된 방법론의 개발과 시행을 제안합니다. 이 카드는 저자 이름 이명화(author name disambiguation)와 성별 식별 절차를 포함한 주요 방법적 선택을 문서화하고 보고하는 체계적 프레임워크를 제공합니다. 기초적인 데이터 분석의 투명성과 재현성을 높여 이 연구는 성별 및 기타 사회적 편향에 관한 정책 결정을 지원할 것입니다.

- **Technical Details**: 연구에서는 12년 간 발표된 70개의 관련 논문을 검토하여 저자 이름 이명화와 성별 식별 방법에 대한 주요 불일치와 최근의 관행을 파악했습니다. 저자 이름 이명화는 학술적인 성과의 정확한 평가를 위해 필수적이며, 기존의 방법들은 일반적으로 이름의 유사성을 기반으로 하거나 수작업 검색을 통해 수행됩니다. 그러나 아시아 이름의 이명화 및 성별 라벨 관리를 포함한 여러 도전 과제가 있으며, 보다 강력하고 표준화된 방법론의 필요성을 강조합니다.

- **Performance Highlights**: 여성 연구자들은 학술적 경로의 여러 단계에서 지속적으로 저평가되며, 성별에 따라 인용 수 및 저자 유형에서 차이가 발생하고 있습니다. 예를 들어, 남성 주 저자가 작성한 논문은 평균적으로 더 높은 인용수를 기록하는 반면, 여성 주 저자는 낮은 평균 인용수를 보였습니다. 또한, 고급 학술 직위에서의 여성의 낮은 비율은 성별 편향을 정량적으로 보여주며, 이러한 편향은 시스템적 요인으로 해결해야 함을 시사합니다.



### Unraveling the Capabilities of Language Models in News Summarization (https://arxiv.org/abs/2501.18128)
- **What's New**: 이 연구에서는 최근 20개의 언어 모델을 종합적으로 벤치마킹하며, 특히 뉴스 요약 작업에 중점을 두었습니다. 소규모 모델들에 대한 성능을 평가하고, 그들의 능력과 효율성을 제로샷(zero-shot) 및 몇 샷(few-shot) 학습 설정에서 분석한 결과, 몇몇 모델이 GPT-3.5-Turbo와 GPT-4에 비해 경쟁력 있는 대안으로 자리 잡을 가능성을 보였습니다. 또한, 데모 예시를 포함한 경우 성능 향상 없이 오히려 품질 저하를 초래한 사례가 있음을 강조하였습니다.

- **Technical Details**: 연구에서는 20개의 최신 언어 모델의 성능을 평가하기 위해 다각적인 평가 방식을 적용했습니다. 자동화된 메트릭, 인간 평가, AI 기반 평가를 결합하여 모델의 요약 품질에 대한 신뢰성 있는 분석을 제공하였습니다. 특히, 뉴스 기사를 다루며 서로 다른 스타일로 작성된 세 가지 데이터 세트를 사용하여 모델의 능력을 체계적으로 테스트했습니다.

- **Performance Highlights**: 결과적으로 GPT-3.5-Turbo와 GPT-4는 매우 뛰어난 성능을 보였으며, 특히 모델 간의 성능 비교를 통해 여러 공공 모델에서 Qwen1.5-7B, SOLAR-10.7B-Instruct-v1.0, Meta-Llama-3-8B 및 Zephyr-7B-Beta와 같은 모델이 주목할 만한 성과를 보여주었습니다. 이는 이들이 대형 모델에 대한 경쟁력 있는 대안으로 나올 수 있음을 시사합니다.



### REMOTE: Real-time Ego-motion Tracking for Various Endoscopes via Multimodal Visual Feature Learning (https://arxiv.org/abs/2501.18124)
- **What's New**: 본 논문에서는 내시경을 위한 실시간 자아 모션 추적(real-time ego-motion tracking)에 대한 새로운 프레임워크를 제안합니다. 이 프레임워크는 다중 모달 비주얼 피쳐 학습 네트워크(multi-modal visual feature learning network)를 통해 상대적 자세 추정(relative pose prediction)을 수행합니다. 또한, 주목 메커니즘(attention mechanism)을 기반으로 한 새로운 피쳐 추출기가 설계되어 두 개의 연속적인 영상에서 수집된 다차원 정보를 통합할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 상대 자세 추정과 절대 자세 계산의 두 단계로 나누어집니다. 첫 단계에서는 현재 관찰값과 이전 프레임을 기반으로 상대 자세 변환을 예측하여, 여러 상대 자세 변환을 통해 내시경의 절대 자세를 계산합니다. 기존의 FastFlowNet을 활용하여 두 개의 인접한 관찰 간의 광학 흐름(optical flow)을 예측하며, 이를 통해 실시간 추적을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 세 가지 엔도스코피 데이터셋에서 최첨단 기법들보다 더 높은 정확도를 보였으며, 초당 30프레임 이상의 추론 속도(inference speed)를 기록하여 실시간 요건을 충족하였습니다. 이는 내시경의 안전하고 정확한 내비게이션을 위해 중요한 기여를 하게 될 것입니다.



### VQLTI: Long-Term Tropical Cyclone Intensity Forecasting with Physical Constraints (https://arxiv.org/abs/2501.18122)
- **What's New**: 이 논문에서는 열대 사이클론(Tropical Cyclone, TC) 강도 예보를 개선하기 위해 VQLTI라는 새로운 프레임워크를 제안합니다. VQLTI는 TC 강도 정보를 이산적인 잠재 공간(discrete latent space)으로 전달하면서 공간 정보 간의 차이를 유지하고, 이를 위해 대규모 기상 데이터를 사용합니다. 또한, FengWu 기상 예측 모델의 예측 결과를 통해 물리적 지식을 통합하여 예측 성능을 더욱 향상시킵니다.

- **Technical Details**: VQLTI는 알려진 과거의 TC 강도 및 ERA5 데이터를 입력으로 받아 미래의 TC 강도를 예측합니다. 이 모델은 Conditional Vector Quantized-Variational AutoEncoder (CVQ-VAE) 구조를 사용하여 강도 정보를 이산 잠재 공간으로 매핑하며, 그러한 잠재 변수에 물리적 제약(physical constraints)을 적용하여 예측의 정확도를 높입니다. 예측에 사용된 ERA5 데이터는 기존 대기 상태의 적합 데이터로, 여러 기상 변수들을 포함하고 있습니다.

- **Performance Highlights**: VQLTI 모델은 24시간에서 120시간의 장기 TC 강도 예보에서 뛰어난 성능을 나타내며, ECMWF-IFS와 비교했을 때 최대 지속풍 속도(Maximum Sustained Wind, MSW) 예측 오차를 35.65%에서 42.51%까지 줄였습니다. 이는 기존 딥러닝 기반 방법들이 가지던 장기 예측의 어려움을 크게 개선한 결과입니다. VQLTI가 보여준 최첨단 성능은 TC 강도 예측 분야에서의 중요한 진전을 의미합니다.



### Self-supervised Quantized Representation for Seamlessly Integrating Knowledge Graphs with Large Language Models (https://arxiv.org/abs/2501.18119)
- **What's New**: 이번 연구에서는 Knowledge Graph (KG) 구조와 대규모 언어 모델 (Large Language Models, LLMs) 간의 통합을 목표로 한 두 단계의 프레임워크를 제안합니다. 첫 번째 단계에서는 Self-Supervised Quantized Representation (SSQR) 방법을 통해 KG의 구조적 및 의미 지식을 압축하여 말뭉치 형식으로 변환된 이산 코드 (discrete codes)를 학습합니다. 이 방법을 통해 우리는 KG와 LLM의 원활한 통합이 가능해짐을 보여줍니다.

- **Technical Details**: 이 연구에서는 그래프 컨볼루션 네트워크 (GCN)를 활용하여 KG의 이웃 구조를 모델링하고, 벡터 양자화 (vector quantization) 기법을 통해 KG 양자화 표현 학습을 수행합니다. 각 엔티티에 대해 학습된 코드 (codes)는 KG 과제에 적합한 지침 데이터를 구축하여 LLMs에 직접 입력할 수 있도록 합니다. 이러한 방식은 LLM의 토크나이저 (tokenizer) 사전의 확장을 통해 실현됩니다.

- **Performance Highlights**: 실험 결과, SSQR 방법은 기존의 비지도 양자화 방법들보다 우수한 성능을 보이며, 더 구별되는 코드를 생성합니다. 또한, 조정된 LLaMA2 및 LLaMA3.1 모델은 KG 링크 예측 및 triple classification 작업에서 뛰어난 성능을 발휘하며, 기존의 수천 개의 토큰 대신 각 엔티티당 단 16개의 토큰만으로 이를 달성하게 됩니다.



### Investigating an Intelligent System to Monitor \& Explain Abnormal Activity Patterns of Older Adults (https://arxiv.org/abs/2501.18108)
- **What's New**: 이 연구는 고령자의 독립적인 거주를 지원하기 위한 기술의 설계를 위해 가족 돌봄 제공자와의 포커스 그룹 세션을 실시하였습니다. 이와 함께, 고신뢰 프로토타입을 개발하고, 전문가와 고령자를 대상으로 시스템 기능에 대한 질적 연구를 진행했습니다. 새로운 시스템은 무선 모션 센서를 사용하여 고령자의 비정상적인 활동 패턴을 모니터링하고, 인터랙티브 대화 응답을 통해 돌봄 제공자와 고령자가 정보 공유를 주도할 수 있도록 지원합니다.

- **Technical Details**: 이 시스템은 일상 생활 활동(Activities of Daily Living, ADL)을 모니터링하고, 고령자의 비정상적인 행동 패턴을 감지하여 돌봄 제공자에게 위험 신호를 전달합니다. 이를 위해 다양한 센서와 머신 러닝 모델을 적용하였으며, 시스템의 디자인을 위한 가족 돌봄 제공자의 피드백도 반영하였습니다. 질적 연구를 통해 두 그룹의 피드백을 수집했으며, 시스템의 성능과 사용자 인터페이스에 대해 의견을 모았습니다.

- **Performance Highlights**: 연구 결과, 고령자와 전문가 모두 시스템의 인터랙티브 대화 기능이 개인화된 서비스 제공과 정보 공유의 효율성을 높일 수 있다고 평가했습니다. 그들은 시스템이 보다 빠르고 주의 깊은 돌봄 서비스를 제공할 수 있으며, 고령자가 자신의 상태를 통제할 수 있는 점에서 긍정적인 의견을 나타냈습니다. 하지만 시스템의 성능과 상호작용 방식에 대한 몇 가지 제한 사항에도 불구하고, 이러한 기술이 더 많은 실용성을 갖출 수 있도록 다양한 고려사항을 제시했습니다.



### Scaling Inference-Efficient Language Models (https://arxiv.org/abs/2501.18107)
Comments:
          17 pages, 16 figures

- **What's New**: 이 연구는 대형 언어 모델의 성능 예측에 있어 scaling laws가 인퍼런스 비용을 충분히 반영하지 못한다는 점을 지적합니다. 또한, 동일한 크기의 모델인데도 아키텍처에 따라 인퍼런스 지연(latency)이 3.5배 차이가 날 수 있음을 보여주며, 인퍼런스 효율성을 고려한 새로운 scaling laws를 제안합니다. 이와 함께 Morph-1B 모델을 발표하며, 기존 오픈 소스 모델 대비 1.8배 더 빠른 인퍼런스 지연을 달성했음을 강조합니다.

- **Technical Details**: 기존의 scaling laws는 모델 크기(모델 파라미터 수)와 훈련 토큰 수의 균형을 중시합니다. 본 연구에서 제안하는 새로운 scaling laws는 모델 아키텍처도 반영하며, 인퍼런스 효율성을 최적화합니다. 다양한 모델 파라미터와 재훈련 과정을 통해, 63개의 실험 모델을 개발하고 이는 인퍼런스 손실 예측력에서 Chinchilla scaling law에 비해 더 우수함을 입증합니다.

- **Performance Highlights**: Morph-1B 모델은 본 연구의 인퍼런스 효율성 기반 scaling laws 및 모델 선택 방법을 통해 개발되었으며, 같은 크기의 다른 오픈 소스 모델에 비해 1.8배 빠른 인퍼런스 지연을 보입니다. 이는 모델의 정확도(downstream task 성능)를 유지하면서도 인퍼런스 효율성을 극대화한 결과입니다. 연구 결과는 향후 아키텍처 최적화에 있어 효율성과 정확도 간의 균형을 잘 잡을 수 있는 기준이 될 것입니다.



### Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation (https://arxiv.org/abs/2501.18100)
- **What's New**: 이번 연구는 유해한 파인튜닝 공격(harmful fine-tuning attack)의 보안 위험을 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 방어 기법들이 모델을 효과적으로 보호하지 못하는 문제를 발견하였습니다; 작은 양의 유해 데이터가 모델의 안전 정렬(safety alignment)을 저하할 수 있다는 것입니다. 이를 해소하기 위해, 단순히 랜덤한 섭동(random perturbations)을 추가하여 모델의 유해한 행동을 회복할 수 있음을 입증하고, 이로 인해 발생하는 성능 저하 문제를 해결하기 위한新的 방안인 Panacea를 제안합니다.

- **Technical Details**: Panacea는 파인튜닝 후 모델에 적용할 적응형 섭동(adaptive perturbation)을 최적화하는 방법으로, 최대 손실(maximal loss)을 증가시키는 방식으로 유해한 행동을 회복할 수 있도록 합니다. 이를 위해 구조화된 최적화 문제를 구성하고, 다양한 유해 비율(harmful ratios)에 대해 실험을 진행했습니다. 그 결과, Panacea는 평균 유해 점수를 21.5%까지 감소시키면서도, 파인튜닝 성능은 오히려 0.3% 향상되었습니다.

- **Performance Highlights**: 실험 결과에 따르면 Panacea는 다양한 유해 비율 및 파인튜닝 작업에 대해 유의미한 성과를 보여주었습니다. 특히, 적응형 섭동을 통해 최대 23.2%의 유해 점수를 감소시키면서도 경쟁력 있는 파인튜닝 성능을 유지할 수 있었습니다. 또한, 시각화 실험은 다양한 LLM의 각 레이어(layer)마다 고유한 안전 계수(safety coefficients)가 존재함을 발견하였습니다.



### LLMs can see and hear without any training (https://arxiv.org/abs/2501.18096)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 MILS(Multimodal Iterative LLM Solver)를 소개합니다. 이 방법은 훈련이 필요 없는 간단한 접근 방식으로, 기존의 LLM에 멀티모달 기능을 추가할 수 있도록 돕습니다. MILS는 다단계 추론(multi-step reasoning) 능력을 활용해 다양한 응용 프로그램을 가능하게 합니다.

- **Technical Details**: MILS는 후보 출력을 생성하고 각 출력을 점수화(scoring)한 후 피드백(feedback)을 통해 반복(iteratively)적으로 솔루션을 도출하는 방식입니다. 이를 통해, 훈련이 필요했던 전문 모델을 사용하지 않고 특정 작업에 대한 솔루션을 생성할 수 있습니다. 이 방법은 이미지를 캡션(captioning)하는 데 있어 새롭고 최고의 성과를 달성했습니다.

- **Performance Highlights**: MILS는 이미지 생성(text-to-image generation)과 같은 미디어 생성(media generation)에도 효과적으로 적용할 수 있으며, 스타일 전이(style transfer)를 위한 프롬프트 수정(prompt rewrites) 발견에도 기여합니다. 또한, 이 방법은 그래디언트(gradient)가 필요 없는 최적화 방식이기 때문에 멀티모달 임베딩(multimodal embeddings)을 텍스트로 변환하여 크로스모달 산술(cross-modal arithmetic) 응용 프로그램에도 사용할 수 있습니다.



### DIAL: Distribution-Informed Adaptive Learning of Multi-Task Constraints for Safety-Critical Systems (https://arxiv.org/abs/2501.18086)
Comments:
          16 pages, 14 figures, 6 tables, submission to T-RO in 2024

- **What's New**: 본 논문은 기존의 안전 강화학습(Safe Reinforcement Learning) 개념을 진화시켜 안전 제약 분포를 여러 작업 간에 학습하는 새로운 방법, 즉 DIAL(Distribution-informed Adaptive Learning)을 제시합니다. 이 방법은 다중 작업에서 공유 지식을 활용하여 새로운 작업에 적응하면서 안전성과 샘플 효율성을 개선합니다. DIAL은 모방 학습을 통해 공유 제약을 파악하고 이들 제약의 위험 수준을 조절하여 다양한 작업에 적용할 수 있는 유연성을 제공합니다.

- **Technical Details**: DIAL은 다중 작업 데모에서 위험 분포를 학습하는 것을 중심으로, 조건부 가치 위험(CVaR)과 같은 왜곡된 기준을 통해 동적으로 위험 수준을 조정합니다. 이 접근 방식은 안전 탐색을 극대화하기 위해 엔트로피를 최대화하여 위험을 조절할 수 있는 기능을 부여합니다. 또한, DIAL은 기존의 ICRL 프레임워크에 두 가지 주요 혁신을 추가하여 제약 함수 및 정책을 학습하는 방향으로 접근합니다.

- **Performance Highlights**: 실험 결과 DIAL 방법이 기존 기준선에 비해 뛰어난 안전성을 가지고 있음을 입증하였습니다. 특히, 특정 작업에 대한 제약 정의가 필요 없는 상황에서도 안전성과 성공률이 높았습니다. 이는 DIAL이 다양한 실제 작업에 걸쳐 강력한 실용성을 보여주고 있음을 앞으로의 작업에 적용할 수 있는 기회를 제공합니다.



### Towards Transparent and Accurate Diabetes Prediction Using Machine Learning and Explainable Artificial Intelligenc (https://arxiv.org/abs/2501.18071)
- **What's New**: 이번 연구는 당뇨병 예측을 위한 프레임워크를 제시하며, 머신러닝 (Machine Learning) 모델과 함께 설명 가능한 인공지능 (eXplainable Artificial Intelligence, XAI) 도구를 활용하여 예측 정확성과 해석 가능성을 조사합니다. 당뇨병 이진 건강 지표 데이터셋을 기반으로 설계된 이 시스템은 조기 진단과 관리의 중요성을 강조합니다.

- **Technical Details**: 데이터 전처리는 클래스 불균형 (class imbalance)과 임상 특성의 변동성을 다루기 위해 합성 소수 샘플 오버 샘플링 기법 (Synthetic Minority Oversampling Technique, SMOTE)과 특징 스케일링을 포함합니다. 앙상블 모델은 92.50%의 테스트 정확도(test accuracy)와 0.975의 ROC-AUC를 제공하여 높은 정확성을 입증했습니다.

- **Performance Highlights**: 모델 설명에서 가장 영향력 있는 예측 변수로는 BMI, 나이 (Age), 일반 건강, 소득 (Income), 신체 활동 (Physical Activity) 등이 나타났습니다. 본 연구 결과는 머신러닝과 XAI를 결합하여 의료 시스템에서 사용할 수 있는 정확하고 계산적으로 투명한 도구 개발의 가능성을 제시합니다.



### Learning Metal Microstructural Heterogeneity through Spatial Mapping of Diffraction Latent Space Features (https://arxiv.org/abs/2501.18064)
- **What's New**: 본 연구는 금속 재료 설계를 위한 새로운 데이터 기반 접근 방식을 제안합니다. 기존의 물리 기반의 방법론이 잃어버리는 미세구조적 정보의 손실을 최소화하기 위해, Kikuchi 패턴에서 얻은 데이터의 전체 정보를 유지하면서 공간적 이질성을 포착합니다. 이로 인해 금속 재료의 기계적 특성을 보다 정확하게 예측할 수 있는 방법론을 제공합니다.

- **Technical Details**: 제안된 방법에서는 변환 오토인코더(Variational Autoencoders)와 대비 학습(Contrastive Learning)을 사용하여 Kikuchi 패턴을 인코딩합니다. 본 논문에서는 두 가지 접근 방식, 즉 VAE와 비슷한 수정된 VAE를 채택하고, 각 패턴을 저차원 latent 공간 표현으로 변환하는 과정을 상세히 설명합니다. Encoder와 Decoder로 구성된 기계 학습 구조가 Kikuchi 패턴을 변환할 때 사용되며, 다양한 latent 공간 차원 수가 실험적으로 분석됩니다.

- **Performance Highlights**: 연구는 주조 및 부품 제조(Alloy) 등의 금속 재료에서 미세구조적 이질성을 성공적으로 식별했습니다. 본 연구의 접근 방식은 전통적인 물리 기반 방법보다 미세구조적 특징을 더욱 민감하게 감지할 수 있도록 하면서, 특히 AM 금속에서 포괄적인 미세구조적 이질성의 매핑을 최초로 실시합니다. 이러한 발견은 새로운 미세구조 설계를 위한 기계적 특성 예측 및 가이드를 가능하게 합니다.



### Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probability Ratio Tes (https://arxiv.org/abs/2501.18059)
Comments:
          Accepted to International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이번 논문은 시간에 민감한 기계 학습을 위한 SPRT(Sequential Probability Ratio Test)의 한계를 극복하고자 FIRMBOUND라는 새로운 프레임워크를 도입합니다. FIRMBOUND는 훈련 데이터로부터 역 귀납법(backward induction)의 해결책을 효율적으로 추정하여 이론적인 최적 정지 규칙을 실용적으로 적용할 수 있게 합니다. 이 프레임워크는 조건부 기대값(condition expectation)과 충분 통계량(sufficient statistic)가 필요하며, 이를 통해 Bayes 리스크를 최소화하고 최적성을 달성합니다.

- **Technical Details**: FIRMBOUND는 두 가지 접근 방식을 통해 역 귀납법을 해결합니다. 첫 번째는 조건부 기대값의 오목한 성질을 인식하고, 이를 볼록 함수 학습(convex function learning)으로 설정하여 통계적으로 일관된 추정기를 제공합니다. 두 번째로, Gaussian 프로세스 회귀(Gaussian process regression)를 활용하여 훈련 속도를 30배까지 개선하는 방법을 제안합니다. 두 모델 모두 테스트 단계에서 낮은 배치 비용을 제공하여 실시간 ECTS에 적합하게 설계되었습니다.

- **Performance Highlights**: FIRMBOUND는 Bayes 최적성에 효과적으로 접근하며, 속도-정확도 무기록 최적화 문제의 Pareto-최적점을 도출합니다. 실험 결과, FIRMBOUND는 기존 ECTS 방법들보다 덜 민감한 매개변수를 가지고 높은 성능을 발휘하였습니다. 또한, FIRMBOUND는 정적 기준을 갖는 SPRT 보다도 낮은 오류율을 보여주며, 실용적인 의사 결정 과정에서 신뢰성을 보장합니다.



### Current Pathology Foundation Models are unrobust to Medical Center Differences (https://arxiv.org/abs/2501.18055)
- **What's New**: 이 논문에서는 Pathology Foundation Models (FMs)의 강건성(robustness)을 측정하기 위한 새로운 지표인 Robustness Index를 도입합니다. 현재 사용 가능한 다양한 pathology FM을 평가하여 이들 모델이 생물학적 특징과 혼란스러운 특징의 영향을 얼마나 잘 구별하는지를 분석합니다. 이 연구는 병원 간의 차이와 같은 외부 요인에 얼마나 민감한지를 평가하여, 해당 모델들이 실제 의료 환경에서 신뢰할 수 있는지 확인하는 데 목적이 있습니다.

- **Technical Details**: 이 연구는 생물학적 특징과 혼란스러운 특징(confounding features) 간의 구분을 통해 강건성 개념을 정의하고, Robustness Index를 통해 이것을 정량적으로 측정합니다. 주어진 데이터셋에서 k개의 최근접 이웃(Nearest Neighbors) 간의 비율로 강건성을 평가하며, 유사도 평가를 위해 코사인 거리(cosine distance)를 사용합니다. 이를 통해 매개변수가 어떻게 생물학적 신호를 반영하는가를 시각적으로 분석할 수 있습니다.

- **Performance Highlights**: 10개의 현재 공개된 pathology FM을 평가한 결과, 대부분의 모델이 특정 의료 센터에 의해 강하게 영향을 받는 것으로 나타났습니다. 단 하나의 모델만이 Robustness Index가 1보다 크게 나타나, 생물학적 특징이 혼란스러운 특징보다 우위에 있다는 것을 시사합니다. 이러한 결과는 패스올로지 AI 모델이 실제 임상에서 수용되는 데 중요한 지침을 제공할 수 있습니다.



### SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders (https://arxiv.org/abs/2501.18052)
- **What's New**: 본 논문에서는 텍스트-이미지 확산 모델에서 원치 않는 개념을 제거하기 위한 새로운 방법인 SAeUron을 소개합니다. SAeUron은 스파스 오토인코더(sparse autoencoders, SAE)가 학습한 특징을 활용하여, 기존의 미세조정(fine-tuning) 방식이 아닌 더 명확한 개입을 가능하게 합니다. 이를 통해 개념이 정확히 제거되었는지 또는 단지 마스킹(masking)되었는지를 명확히 알 수 있습니다.

- **Technical Details**: SAE는 여러 디노이징 타임스텝(denoising timesteps)에서 활성화된 특징을 기반으로 비지도 학습 방식으로 훈련되어 특정 개념에 해당하는 희소하고 해석 가능한 특징을 캡처합니다. 본 연구에서는 이러한 특징들 중 개념 특정 특징을 선택하는 방법을 제안하였습니다. 이를 통해 모델의 활성화에 정밀한 개입을 하여 목표 콘텐츠를 차단하면서도 전체 성능은 유지할 수 있습니다.

- **Performance Highlights**: 경쟁력이 있는 UnlearnCanvas 벤치마크(object and style unlearning)에서 SAeUron의 최첨단 성능이 입증되었습니다. 특히 하나의 SAE를 사용하여 동시에 여러 개념을 제거할 수 있으며, 기존의 방법들과 달리 SAeUron은 적대적 공격(adversarial attack) 하에서도 원치 않는 콘텐츠 생성을 방지합니다.



### From tools to thieves: Measuring and understanding public perceptions of AI through crowdsourced metaphors (https://arxiv.org/abs/2501.18045)
- **What's New**: 본 연구는 미국 내 인공지능(AI)에 대한 공공 인식의 변화를 조사하기 위해 12개월 간 12,000개의 응답을 수집하였습니다. 전통적인 자기보고 조사 방법의 한계를 극복하기 위해 응답자가 AI를 묘사한 은유를 분석했습니다. 이 연구는 AI에 대한 인식의 주요 차원인 인간형성(anthropomorphism), 따뜻함(warmth), 그리고 능력(competence)을 측정하기 위한 확장 가능한 프레임워크를 제공합니다. 이를 통해 우리는 미국인들이 AI를 일반적으로 따뜻하고 유능하다고 바라보며, 이러한 인식이 AI에 대한 신뢰와 수용의도에 강하게 영향을 미친다는 사실을 발견했습니다.

- **Technical Details**: 본 연구에서는 질적 코딩과 정량적 클러스터링을 결합한 혼합 방법론(mixed-methods approach)을 사용하여, 20개의 주도적인 은유를 식별했습니다. 응답자들이 얼마나 AI를 인간으로 형상화했는지를 평가하기 위해 언어모델(language model) 기반의 확률을 이용한 방법론이 사용되었습니다. 또한, AI에 대한 따뜻함과 능력을 측정하기 위해 LM 기반의 임베딩을 사용하여 의미 축을 구축했습니다. 이를 통해 개인의 AI에 대한 경험을 예측하는 중요한 지표로서 인간형성과 따뜻함, 능력을 강조했습니다.

- **Performance Highlights**: 연구 결과, AI에 대한 참여자의 은유는 대체로 따뜻하고 유능했으나 인간형성의 정도는 다양하게 나타났습니다. 특히, 인간형성과 따뜻함은 지난 1년 동안 각각 34%와 41%의 유의미한 증가세를 보였습니다. 또한, '신(god)', '두뇌(brain)', '도둑(thief)'이라는 주도적인 은유가 AI에 대한 신뢰와 수용도를 예측하는 데 중요한 역할을 하며, 사회적 정체성이 AI와의 상호작용에 미치는 영향도 분석되었습니다. 이 연구는 다양한 인구 통계적 차이를 바탕으로 AI에 대한 인식의 변화의 맥락을 제공합니다.



### Digital Twin-Enabled Real-Time Control in Robotic Additive Manufacturing via Soft Actor-Critic Reinforcement Learning (https://arxiv.org/abs/2501.18016)
- **What's New**: 본 연구는 Soft Actor-Critic (SAC) 강화 학습을 디지털 트윈 기술과 통합하여 로봇 적층 제조에서 실시간 프로세스 제어를 가능하게 하는 새로운 접근 방식을 소개합니다. Viper X300s 로봇 팔을 사용하여 정적 목표 도달과 동적 경로 추적 두 가지 제어 시나리오를 구현하였습니다. Unity 시뮬레이션 환경과 ROS2를 결합하여 디지털 트윈 동기화를 원활하게 수행하며, 전이 학습을 통해 학습된 모델을 효율적으로 태스크 간에 조정할 수 있습니다.

- **Technical Details**: 연구에서 사용된 Viper X300s 로봇 팔은 6자유도(6 DOF) 구조로, ROS inter-bridging을 통해 Unity의 가상 환경에서 태스크를 시뮬레이션하고 실제 로봇에서 검증할 수 있는 피드백 루프를 제공합니다. 디지털 트윈 방식의 연결을 통해 Unity에서 Viper X300s의 조인트 데이터를 로컬 TCP 서버로 전송하고, 이 데이터는 ROS2에서 구독하여 로봇 팔을 지속적으로 이동시킵니다. 이를 통해 로봇의 실제 동작에 대한 실시간 데이터를 제공하고, 지연 시간은 약 20밀리초로 실시간 반영이 가능합니다.

- **Performance Highlights**: 실험 결과는 신속한 정책 수렴과 튼튼한 작업 실행을 보여주었으며, 누적 보상(cumulative reward), 가치 예측 정확도(value prediction accuracy), 정책 손실(policy loss), 이산 엔트로피 계수(discrete entropy coefficient)와 같은 성능 지표가 연구 접근 방식의 효과성을 입증하고 있습니다. 강화 학습 기반의 제어 시스템에서 Unity와 ROS2 통합을 통해 로봇의 적응성을 향상시키고, 스마트 적층 제조 프로세스에 대한 실시간 제어를 강화하는 데 기여합니다.



### Anatomy Might Be All You Need: Forecasting What to Do During Surgery (https://arxiv.org/abs/2501.18011)
- **What's New**: 이 논문은 외과 수술 도구의 움직임을 예측하는 새로운 형태의 가이드를 제안합니다. 기존의 수술 내비게이션(system) 및 수술 단계 인식에 이어, 수술 도구의 향후 경로를 예측하여 다음 단계에 대한 명확한 지침을 제공하고자 합니다. 최근의 연구 결과들을 바탕으로 수술 비디오에서 해부학적 특징과 도구의 위치를 분석하여, 도구의 다음 위치를 예측하는 모델을 개발하였습니다.

- **Technical Details**: 우리는 endoscopic video frames의 시퀀스를 나타내는 𝐒_{t}를 정의하고, 주 목표는 미래의 frames에서 수술 도구의 위치 변화를 예측하는 것입니다. 이를 위해, YOLO(You Only Look Once) 알고리즘과 같은 객체 감지 기법을 활용하여 수술 동영상의 프레임에서 해부학적 구조와 수술 도구를 인식하고, 이 정보를 바탕으로 도구의 미래 위치를 예측합니다. 모델은 간단한 신경망(neural network) 아키텍처를 사용하여 과거 64 프레임을 기반으로 다음 8개 또는 16개의 프레임 내에서 도구의 위치를 예측합니다.

- **Performance Highlights**: 실험 결과, 해부학적 구조 탐지가 수술 도구의 움직임 예측 문제를 해결하는 데 중요한 역할을 하는 것으로 나타났습니다. 이러한 접근 방식은 경험이 적은 외과 의사들에게도 도움이 될 수 있는 자율 수술 로봇의 통합된 관리 요소가 될 수 있습니다. 본 연구는 향후 수술 행동에 대한 정확한 가이드를 제공할 수 있는 모델 개발의 초석이 되기를 희망합니다.



### Topological Signatures of Adversaries in Multimodal Alignments (https://arxiv.org/abs/2501.18006)
- **What's New**: 최근에 개발된 다중모달 기계 학습 시스템, 특히 텍스트와 이미지 데이터를 정렬하는 CLIP 및 BLIP 모델은 적대적 공격에 취약하다는 점이 연구 중에 발견되었습니다. 기존 연구는 단일 모달 시스템에 대한 적대적 강인성에 초점을 맞추었으나, 다중모달 시스템에 대한 방어 전략은 아직 충분히 탐구되지 않았습니다. 본 연구는 이미지와 텍스트 임베딩 간의 위상적 서명을 조사하며, 적대적 공격이 이들의 정렬을 어떻게 방해하는지 보여줍니다.

- **Technical Details**: 본 연구에서 제안하는 두 가지 새로운 Topological-Contrastive 손실, 즉 Total Persistence (TP)와 Multi-scale Kernel (MK) 손실은 적대적 변동에 의해 생성된 위상적 서명을 분석하는 데 사용됩니다. Persistent Homology라는 위상적 데이터 분석의 기법을 활용하여 이미지-텍스트 정렬의 변화를 추적하고, 이 손실들이 적대적 샘플의 비율에 따라 어떻게 변화하는지를 관찰하였습니다. 이러한 손실은 적대적 샘플이 포함된 데이터 배치에서 이미지와 텍스트 임베딩 간의 불일치를 포착합니다.

- **Performance Highlights**: CIFAR-10과 ImageNet 데이터셋을 사용하여 실험을 수행한 결과, 제안된 TC 손실들이 이미지-텍스트 정렬의 정밀한 특성을 감지하는 데 매우 효과적임을 입증했습니다. 또한, Maximum Mean Discrepancy (MMD) 테스트에 이 손실을 통합하여 기존 MMD 솔루션의 정확성과 1형 오류를 제어하는 데 있어 우수한 성능을 발휘하였습니다. 이 연구는 다중모달 AI 시스템에 대한 적대적 공격 감지의 새로운 접근법을 제공합니다.



### Belief Roadmaps with Uncertain Landmark Evanescenc (https://arxiv.org/abs/2501.17982)
- **What's New**: 본 논문에서는 로봇이 목표 위치까지 이동하는 과정에서 상태 불확실성을 최소화하기 위한 방법을 제시합니다. 이를 위해 로봇은 지도(map)를 활용하여 물체와 관심 지역의 위치에 대한 미리 설정된 신념(prior belief)을 얻습니다. 특히, 지도 제작과 로봇 배치 사이의 시간이 길어질수록 지도에서의 랜드마크가 사라질 수 있다는 점을 다루며, 이를 랜드마크의 소멸(landmark evanescence)이라고 언급합니다.

- **Technical Details**: 로봇의 위치를 지도에서 정확히 파악하기 위해, 로봇은 센서를 사용하여 매핑된 랜드마크를 식별합니다. 랜드마크의 소멸을 고려하여 경로 계획(path planning)을 수행하는 과정에서, 각 랜드마크의 존재 여부를 분석해야 하며, 이는 주어진 모션 계획에 대한 지수(exponential) 수의 가능한 결과를 초래합니다. 이를 해결하기 위해, BRULE라는 벨리프 로드맵(Belief Roadmap)의 확장을 개발하였습니다.

- **Performance Highlights**: BRULE는 로봇의 미래 자세(pose)에 대한 신념을 교환하여 랜드마크 소멸의 영향을 포착할 수 있는 Gaussian 혼합(Gaussian mixture)으로 대체합니다. 또한 신념 갱신(belief updates)을 효율적으로 수행할 수 있음을 보여주며, 혼합 성분(mixture components)의 무작위 하위 집합을 유지함으로써 고품질 솔루션(solution)을 발견할 수 있음을 입증합니다. 이 성능은 시뮬레이션 및 실제 실험을 통해 입증되었습니다.



### Limits to AI Growth: The Ecological and Social Consequences of Scaling (https://arxiv.org/abs/2501.17980)
Comments:
          14 pages

- **What's New**: 이 논문은 AI 스케일링(AI scaling)의 동적 복잡성을 이해하기 위해 기술적, 경제적, 생태적, 사회적 네 가지 관점을 종합적으로 검토합니다. AI 인프라의 확장이 재무적 및 환경적 비용의 증가를 동반하는 가운데, 이러한 요소 간의 관계를 모델링하여 AI의 성장이 직면한 한계를 분석합니다. 이를 통해 빅 테크 업계는 지속가능성을 외부화하며 사회적 및 환경적 손해를 초래하고 있다는 점을 강조합니다.

- **Technical Details**: 시스템 다이나믹스(system dynamics)의 개념을 바탕으로 '성장의 한계(limits to growth)'와 같은 고전적 패턴을 사용하여 AI 스케일링의 동적 복잡성을 모델링합니다. 이 과정에서 기술적 측면, 경제적 함의, 생태적 결과 및 윤리적 문제 간의 상호작용을 조사합니다. 특히, AI 산업의 덩치 큰 기업들이 외부 한계에 어떻게 대응하는지를 분석하여 단기적인 스케일링을 가능하게 하는 방식을 밝혀냅니다.

- **Performance Highlights**: AI의 지속적인 발전을 위해서는 스케일링으로 인한 사회적 및 환경적 손해를 재조정하고 지속 가능한 성장을 우선시해야 한다고 주장합니다. 이 논문은 AI 기술 발전이 지구와 사회에 미치는 영향을 고려하여, '과도한 성장과 붕괴(overshoot and collapse)' 경로를 피하기 위한 대안적 개발 경로를 제시합니다. 결론적으로, AI의 성장은 현재 사회와 환경에 미치는 비용과 이익을 재평가하는 기회를 제공합니다.



### Deep Ensembles Secretly Perform Empirical Bayes (https://arxiv.org/abs/2501.17917)
- **What's New**: 이 논문은 딥 앙상블(deep ensembles)과 베이지안 신경망(Bayesian neural networks, BNNs) 간의 관계를 밝혀내어 두 접근법이 본질적으로 다르지 않다는 것을 보여줍니다. 이는 딥 앙상블이 데이터에 의존하여 학습되는 사전(prior)을 통해 후방 분포(posterior)을 얻어내면서 정확한 베이지안 평균(Bayesian averaging)을 수행한다는 사실을 기반으로 합니다. 이에 따라, 딥 앙상블은 BNN의 일종으로 여겨질 수 있으며, 이로 인해 두 접근법 간의 깊은 연관성을 제공합니다.

- **Technical Details**: 논문에서 제안하는 바에 따르면, 딥 앙상블은 최대 주변 우도(maximum marginal likelihood)를 통해 유연하게 학습 가능할 사전 분포를 도출함으로써 BNNs와 등가입니다. 이 과정에서, 고차원 데이터에 대해 더 강한 사전(explained prior)을 제공하는 혼합(point masses mixture) 방식이 사용됩니다. 이러한 새로운 관점은 여태까지 BNN과 앙상블 간의 오해를 해소하며 두 방법의 상관관계를 명확히 합니다.

- **Performance Highlights**: 이 연구는 딥 앙상블의 성능이 뛰어난 이유에 대한 설명을 제시하며, 향후 UQ(uncertainty quantification)에 대한 더 나은 통찰력을 제공할 것으로 기대됩니다. 또한, 강력한 예측 밀도(predictive density)를 통해 신뢰할 수 있는 불확실성 추정치를 제공하는 데 있어 앙상블의 중요성을 강조하며, 의료 진단, 자율주행 등 안전이 중요한 분야에서의 응용 가능성을 제시합니다. 이러한 이해는 궁극적으로 영감을 줄 방법론의 발전으로 이어질 가능성이 큽니다.



### DReSS: Data-driven Regularized Structured Streamlining for Large Language Models (https://arxiv.org/abs/2501.17905)
- **What's New**: 이 논문에서는Llarge language models (LLMs)의 성능을 유지하면서 모델 크기를 줄일 수 있는 새로운 프루닝(paruning) 패러다임을 제안합니다. 기존의 프루닝 방법이 정보 손실을 초래하는 것과 달리, 새로운 접근 방식에서는 먼저 정규화(regularization)를 적용한 후 프루닝을 수행하고, 마지막으로 미세 조정(finetuning)을 진행합니다. 이 과정을 통해 DReSS(Data-driven Regularized Structured Streamlining)라는 효과적인 방법을 도입하게 되었습니다.

- **Technical Details**: DReSS는 파라미터 행렬에서 선택된 채널에 정규화 과정을 적용하여 중요 정보를 전이하여 제거되는 부분의 정보를 보존합니다. 이 방법은 프로세스가 다음 네 단계로 구성되어 있습니다: 데이터 선택, 정규화 적용, 채널 프루닝 후 RFT 수행입니다. 작은 데이터 세트를 활용함으로써 프루닝 과정에서 발생하는 오버헤드를 최소화하고, 높은 프루닝 비율에서도 성능 저하를 방지합니다.

- **Performance Highlights**: 실험 결과 DReSS는 기존 프루닝 방법보다 크게 향상된 정밀도와 정확도를 보였습니다. 이 방법은 프루닝 비율이 극단적인 상황에서도 성능이 크게 향상되어 상당한 지연 시간 감소와 처리량 증가를 이끌어냈습니다. 따라서 DReSS는 대형 언어 모델에서 정보 손실을 줄이고, 언어 모델링 능력을 향상시키는 데 기여했습니다.



### Free Agent in Agent-Based Mixture-of-Experts Generative AI Framework (https://arxiv.org/abs/2501.17903)
- **What's New**: 이 연구에서는 Multi-agent systems에서 독립적으로 작업을 수행하는 에이전트들 간의 성과를 지속적으로 관리하는 Reinforcement Learning Free Agent (RLFA) 알고리즘을 소개합니다. 이 알고리즘은 성과가 저조한 에이전트를 실시간으로 교체할 수 있는 보상 기반 메커니즘을 도입하여, Major League Baseball의 자유계약 제도에서 영감을 받았습니다.

- **Technical Details**: RLFA 알고리즘은 각 에이전트가 mixture-of-experts (MoE) 접근 방식을 사용하여 작업을 전문화된 하위 모델에 할당합니다. 이때, 에이전트의 성과가 사전 설정된 기준 이하로 떨어지면 즉시 다른 더 나은 에이전트로 교체할 수 있는 방법이 적용됩니다. 새로운 에이전트는 시험 모드에서 성능을 검증받아야 하며, 성과가 우수함을 보여주면 결국 기존 에이전트를 대체하게 됩니다.

- **Performance Highlights**: 이 시스템의 주요 사례는 사기 탐지(fraud detection)로, 신속하게 탐지 정확도가 낮아진 에이전트를 교체하여 지속적인 정확성을 보장합니다. RLFA의 동적 자유계약 사이클은 새로운 위협에 대한 빠른 적응을 가능하게 하며, 운영의 중단을 최소화합니다. 에이전트를 지속적으로 갱신함으로써, 이 시스템은 멀티 에이전트 Generative AI 환경에서 협력의 회복력을 높이고 지속적인 개선을 촉진합니다.



### The Right to AI (https://arxiv.org/abs/2501.17899)
Comments:
          19 pages, 2 figures, 1 table

- **What's New**: 이 논문은 개인과 공동체가 그들의 삶을 형성하는 AI 시스템의 개발 및 관리에 의미 있게 참여할 수 있는 'Right to AI'를 제안합니다. 이는 AI가 단순한 전문가의 디자인 결과물이 아니라 사회 기반 시설(societal infrastructure)로 재구성되어야 한다는 관점에서 접근합니다. 이 논문은 데이터를 사회적으로 생산되는 것으로 보고, 공공 참여의 중요성을 강조합니다.

- **Technical Details**: 제안된 'Right to AI'는 시민 참여 모델을 구축하여 알고리즘 공정성 문제를 해결하고 사회적 반응성을 향상시키기 위한 방법론을 포함합니다. 논문은 또한 Sherry Arnstein의 시민 참여의 사다리를 통해 다수의 사례 연구를 분석하면서 AI에 대한 공동체의 권리를 4단계 모델로 발전시키고, 투명한 데이터 관리 및 디자인 프로세스를 추천합니다. 여기서 AI는 공공 데이터에 기반하여 투명성과 책임이 필수적임을 강조합니다.

- **Performance Highlights**: 시민 참여 방식이 기술적 효율성과 민주적 정당성 간의 균형을 제공할 수 있다고 주장하며, 최근의 정책 제안 및 윤리적 지침들이 대개 상향식(top-down) 모델에 한정되어 있다는 점을 비판합니다. 연구자들은 AI가 사회적 가치와 맞물려야 하며, 보다 포괄적이고 다원적인 구조가 알고리즘 편향 문제를 해결할 수 있다고 지적합니다. 논문은 'Right to AI'가 현대 사회에서 정보 접근과 권력 할당을 크게 변화시킬 수 있는 잠재력을 가지고 있다고 주장합니다.



### Progress in Artificial Intelligence and its Determinants (https://arxiv.org/abs/2501.17894)
- **What's New**: 이 논문은 인공지능(AI)의 장기 발전을 정량적으로 연구하며, 특허와 출판물, 머신러닝 벤치마크, 새로운 ASOTA 지수를 포함한 다양한 측정치를 사용하여 이들의 기하급수적 성장을 보여주고 있습니다. AI 연구자들의 기여를 객관적으로 추정할 수 있으며, 이들 간의 비율을 설명하는 간단한 주장을 제시하고 있습니다. 또한, 기존 문헌의 머신러닝 스케일링 법칙과의 비교도 포함되어 있습니다.

- **Technical Details**: 이 논문에서는 컴퓨팅 자원과 인간의 지적 노동을 포함한 다양한 입력 측정치를 수집하고, 전통적인 출판물 수와 특허 수를 시작으로 여러 출력 측정치를 비교합니다. ASOTA 지수는 머신러닝의 벤치마크를 기반으로 정의되며, 그 기본 성질을 존중하면서도 무한히 미래의 기술 발전을 포함할 수 있도록 설계되었습니다. 논문은 입력과 출력 간의 관계를 설명하기 위해 두 가지 모델을 개발하고 있으며, 이를 통해 AI 발전에 있어 무어의 법칙이 주요한 요인임을 정량적으로 확인하였습니다.

- **Performance Highlights**: ASOTA 지수는 다양한 벤치마크 성과를 집계하여 머신러닝 모델의 성능 향상을 포착하며, 이전 성장률보다 느리지만 일관된 기하급수적 성장을 보여주고 있습니다. 전통적인 연구 성과 측정과 더불어, 지수는 AI 연구에 투자된 총 컴퓨팅 자원의 변화를 분석하여 AI 발전의 요인을 보다 명확히 이해할 수 있도록 도와줍니다. 궁극적으로 논문은 AI 기술 개발의 모니터링과 예측을 위한 정량적 프레임워크를 제공합니다.



### Knoop: Practical Enhancement of Knockoff with Over-Parameterization for Variable Selection (https://arxiv.org/abs/2501.17889)
Comments:
          An earlier version of our paper at Machine Learning

- **What's New**: 이 논문은 변수 선택을 위한 새로운 접근법인 Knockoff with Over-Parameterization (Knoop)을 소개합니다. 기존 Knockoff 방법을 개선하여, 각 원본 변수에 대해 여러 개의 knockoff 변수를 생성하고 이를 Ridgeless 회귀 모델에 통합함으로써 더욱 효과적인 변수 선택을 도모합니다. Knoop 방법은 변수의 회귀 계수 분포를 평가하여, 원본 변수와 knockoff 변수 간의 비교를 통해 유의미성을 검증하는 과정으로 안정적인 변수 선택을 보장합니다.

- **Technical Details**: Knoop은 복수의 계층화된 knockoff 변수를 생성하는 재귀적 접근 방식을 사용하여, 이러한 knockoff 변수들이 원본 변수와 교환 가능한 독립성을 유지하도록 합니다. 이후 원본 변수와 knockoff 변수를 Ridgeless 회귀 모델에 통합하여 회귀 계수의 추정을 개선합니다. 마지막으로 Knoop은 이상치 기반 유의성 검정을 제안하여, 원본 변수를 knockoff 변수를 활용해 비교함으로써 false discovery rate (FDR)를 제어하고, 변수의 중요도에 따라 순위를 매깁니다.

- **Performance Highlights**: 광범위한 실험을 통해 Knoop의 성능이 기존 방법들에 비해 우수함을 입증하였습니다. 특히 Knoop은 시뮬레이션 및 실제 데이터셋에서 신뢰할 수 있는 변수 식별을 위한 평균적으로 더 높은 AUC (Area under the Curve)를 달성했습니다. 다양한 회귀 및 분류 작업에서 예측 정확도를 개선하며, 이론적 분석 결과 또한 관찰된 성능을 뒷받침합니다.



### RadioLLM: Introducing Large Language Model into Cognitive Radio via Hybrid Prompt and Token Reprogrammings (https://arxiv.org/abs/2501.17888)
- **What's New**: 이 논문에서는 Cognitive Radio Technology (CRT)와 Large Language Models (LLMs)를 결합하여 RadioLLM이라는 새로운 프레임워크를 개발했습니다. RadioLLM은 Hybrid Prompt and Token Reprogramming (HPTR) 및 Frequency Attuned Fusion (FAF) 모듈을 포함하여 다양한 CRT 작업에 적합한 성능을 발휘할 수 있도록 설계되었습니다. 이 접근법은 LLMs의 일반화 능력을 활용하여 신호 처리의 정밀도를 높이고, 더욱 효율적인 스펙트럼 할당과 신호 분류를 가능하게 합니다.

- **Technical Details**: RadioLLM은 두 가지 주요 구성 요소로 설계되었습니다. 첫째, 입력 신호를 패치로 나누어 신호 임베딩을 생성하며, 둘째, CNN에서의 높은 주파수 특징과 LLM에서의 낮은 주파수 신호 정보를 결합하여 출력합니다. GPT-2를 백본 모델로 사용하여 입력과 출력 간의 매핑 함수를 학습하고 LoRA 기술을 통해 모델을 미세 조정합니다. 이러한 방법은 다양한 신호 형식과 작업을 수용할 수 있는 통합된 프레임워크를 구조적으로 제시합니다.

- **Performance Highlights**: RadioLLM은 다양한 벤치마크 데이터셋에서 기존의 기준 성능을 초과하는 성능을 보여줍니다. 특히, RSC(Radio Signal Classification) 작업에서 높은 주파수 특성을 효과적으로 모델링하여 정확도를 높였습니다. 다양한 실제 시나리오에서 응용 가능한 범용 CRT 프레임워크로서의 유망성을 demonstrated 합니다.



### Explainable and Robust Millimeter Wave Beam Alignment for AI-Native 6G Networks (https://arxiv.org/abs/2501.17883)
- **What's New**: 본 연구는 6G 네트워크에 통합된 인공지능(AI) 및 통신의 중요성을 강조하며, AI 기반 시스템의 설명 가능성과 강인성이 신뢰를 구축하고 신뢰할 수 있는 성능을 보장하기 위해 필수적임을 설명합니다. 제안된 내용은 밀리미터파(mmWave) 다중 입력 다중 출력(mMIMO) 시스템을 위한 강력하고 설명 가능한 딥러닝(DL) 기반 빔 정렬 엔진(BAE)을 개발하는 것에 초점을 맞추고 있습니다. 이 엔진은 수신 신호 세기 지표(RSSI)를 활용하여 최적의 빔을 예측하게 됩니다.

- **Technical Details**: 본 논문의 시스템 모델은 BS와 단일 안테나 UE 간의 통신을 특별히 다루며, BS는 평균 선형 배열(ULA) 안테나를 통해 여러 UE에게 동시에 정보를 전달합니다. 빔 정렬 문제 해결을 위해, CNN 기반의 BAE는 O-DFT 코드북을 사용하여 초기 접근 과정에서의 빔 훈련 효율성을 크게 향상시킵니다. 또한 딥 k-최근접 이웃(DkNN) 알고리즘을 도입하여 모델의 내부 표현을 평가하고, 불확실성을 줄이고 이해 가능한 예측 및 신뢰도 지표를 제공합니다.

- **Performance Highlights**: 제안된 DL 기반 BAE는 O-DFT 코드북에서의 수색을 통해 기존의 방법보다 75%의 빔 훈련 오버헤드를 줄이면서도 거의 최적의 성능을 유지합니다. 실험 결과, 제안된 프레임워크는 아웃라이어 탐지 강인성을 최대 5배 향상시키며, 전통적인 소프트맥스 기반 분류기보다 더욱 명확한 빔 예측의 통찰력을 제공합니다. 이러한 성과는 AI 기반의 방식이 6G 네트워크의 고도화에 기여할 수 있음을 보여줍니다.



### RayLoc: Wireless Indoor Localization via Fully Differentiable Ray-tracing (https://arxiv.org/abs/2501.17881)
- **What's New**: 이 논문에서는 전통적인 무선 실내 위치 확정의 제한 사항을 극복하기 위해 RayLoc이라는 새로운 접근법을 제안합니다. RayLoc은 무선 레이 트레이싱의 역 문제로 위치 결정을 재정의하여 범위 내의 모든 장치가 없는 감지 장면의 매개 변수를 추출합니다. 이를 통해 CSIs의 정확한 재현이 가능해져 기존 기법들과 비교할 때 더 높은 정확도를 보입니다.

- **Technical Details**: RayLoc은 완전히 차별화된 레이 트레이싱 시뮬레이터를 기반으로 하여 무선 신호 전파를 정확히 시뮬레이션합니다. 이 시스템은 고충실도 배경 모델을 구축하여 물리적 환경의 기하학과 물질 전자기적 속성을 포함하고 있습니다. 또한, 가우스 커널을 사용하여 손실 경관을 부드럽게 하여 최적화의 어려움을 극복하고 위치 정확성을 높입니다.

- **Performance Highlights**: 신뢰할 수 있는 실험 결과를 통해 RayLoc은 기존의 전통적인 위치 추정 방법들보다 월등히 높은 성능을 보여 줍니다. 다양한 환경에서도 높은 일반화를 이루어내며, 이 시스템은 장치 없는 및 장치 기반의 위치결정에 통합된 예방 가능한 솔루션을 제시합니다. 마지막으로 RayLoc은 정밀한 위치 추정이 가능하며, 여러 싸이클을 통해 연결됩니다.



### Assessment of the January 2025 Los Angeles County wildfires: A multi-modal analysis of impact, response, and population exposur (https://arxiv.org/abs/2501.17880)
- **What's New**: 이 연구는 팔리세이즈, 이튼, 케네스, 허스트 등 네 가지 주요 캘리포니아 산불 사건을 종합적으로 분석하여, 땅의 변화, 관리 권한, 구조적 피해 및 인구 Vulnerability 등 다양한 측면에서 영향을 조사했습니다. Chebyshev-Kolmogorov-Arnold network (Cheby-KAN) 모델을 활용한 Sentinel-2 이미지를 통해 화재로 피해를 입은 지역의 범위를 정밀하게 매핑했습니다.

- **Technical Details**: 연구 방법론으로는 Sentinel-2 위성 이미지를 이용하여 화재 발생 전후의 상황을 분석하는 복합적인 접근 방식을 채택했습니다. Cheby-KAN 모델을 적용하여 315.36에서 10,960.98 헥타르에 이르는 화재 범위를 정밀하게 감지하며, 고해상도 지리공간 데이터를 활용하여 인구 조사와 토지 피복 변화, 인프라 Vulnerability를 평가했습니다.

- **Performance Highlights**: 연구 결과는 도시와 농촌 화재 사건 간의 구조적 피해와 인구 노출의 불균형을 보여 주며, 팔리세이즈와 이튼 화재가 각각 20,000명 이상에게 영향을 미친 반면, 농촌 사건은 500명 미만의 인구에 영향을 주었음을 밝혔습니다. 이 연구는 산불 관리 전략을 개발하는 데 있어 중요한 통찰을 제공하며, 특히 도시-야생 지역 인터페이스에서의 연령 및 성별에 대한 인식을 강조했습니다.



### Task and Perception-aware Distributed Source Coding for Correlated Speech under Bandwidth-constrained Channels (https://arxiv.org/abs/2501.17879)
Comments:
          Published at AAAI 2025 Workshop

- **What's New**: 이번 논문에서는 제한된 자원을 가진 장치들로부터 여러 고급 음성을 실시간으로 전송할 수 있는 신규 알고리즘인 NDPCA(Neural Distributed Principal Component Analysis)-지원 분산 소스 코딩 방식을 제안합니다. 기존의 autoencoder 기반 음성 소스 코딩 방법들은 동적 비트 전송 속도를 조정하거나 여러 음성 소스 간의 상관관계를 활용하는 데 한계가 있었습니다. 제안된 방법은 지각적 사실성과 태스크 성능을 균형 있게 고려하는 새로운 손실 함수(perception-aware downstream task loss function)를 포함합니다.

- **Technical Details**: 연구의 핵심은 여러 마이크로폰이 서로 다른 환경에서 동일한 음성을 녹음하고, 이를 중앙 수신기로 전송하는 분산 소스 코딩 상황을 설정하는 것입니다. 수신기에서의 무선 채널 상태 정보(Channel State Information, CSI)를 기반으로, 변동하는 채널 용량을 결정하고 각 소스에 최적의 비트 전송률을 할당합니다. NDPCA를 통해 다수의 상관된 음성 소스를 보다 효율적으로 압축하고 전송하는 방식을 개발했습니다.

- **Performance Highlights**: 실험 결과는 제안된 NDPCA 방식이 대역폭 제약 하에서 평균 PSNR(Peak Signal-to-Noise Ratio)을 기존의 방법보다 19%에서 52% 향상시켰음을 보여주었습니다. 특히 대역폭이 낮은 상황에서도 모든 음성을 단일 인코더를 통해 전송하는 이론적 한계에 근접하는 성능을 보였습니다. 이 논문은 대역폭 제약이 있는 상황에서 압축, 사실성 및 과제 성능을 최적화할 수 있는 방법을 제시합니다.



### An Automatic Sound and Complete Abstraction Method for Generalized Planning with Baggable Types (https://arxiv.org/abs/2501.15249)
- **What's New**: 이번 논문은 일반화된 계획 문제를 해결하기 위한 자동으로 구축된 접근 방법을 제안합니다. 특히, baggable types를 사용하는 제약을 만족하는 QNP 문제의 추상화 방법을 소개하며, 'bounded QNP (BQNP)'라는 변형을 활용합니다. 이 연구는 BQNP 문제의 세분화된 계획이 일반화된 계획 문제의 해법이 됨을 보장할 수 있는 최초의 자동 추상화 방법입니다.

- **Technical Details**: 본 논문에서는 situation calculus를 활용해 동적 세계를 설명할 수 있는 다중 정렬 1차 언어를 기반으로 합니다. 조치(action), 상황(situation) 및 객체(object) 등의 세 가지 종류별로 구분됩니다. BQNP는 정수 변수의 증감이 1로 제한되는 것을 특징으로 하며, 추상화 과정에서 비가시적 객체 튜플의 각 가방에 카운터를 도입함으로써 문제를 단순화합니다.

- **Performance Highlights**: 여러 도메인에 대한 실험을 통해 제안된 접근 방식의 유망함을 입증했습니다. BQNP 문제를 통해 우리는 soundness와 completeness를 모두 보장할 수 있으며, 이는 일반화된 계획 문제에 대한 효과적인 해결책으로 기능합니다. 최종적으로, 이 연구는 자동화된 추상화 방법론의 새로운 가능성을 보여주며, 향후 연구에 기여할 것으로 기대됩니다.



New uploads on arXiv(cs.LG)

### DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights (https://arxiv.org/abs/2501.18596)
- **What's New**: DeltaLLM을 소개하며 이는 LLM의 메모리 발자국을 줄이기 위한 새로운 사후 훈련 압축 기법이다. 우리는 변환기 블록 간의 가중치 공유 및 저순위 차이 행렬을 통해 LLM의 구조를 대안적으로 제안하였다. DeltaLLM은 30M-40M 토큰으로 경량 훈련을 수행하며 기존 LLM과 유사한 성능을 달성할 수 있음을 보여준다.

- **Technical Details**: DeltaLLM은 여러 가지 모델 압축 기술을 체계적으로 탐구하며, 가중치 공유, 저순위 적응 및 점진적 모듈 교체 방식을 포함한다. 이 접근 방식을 통해 DeltaLLM은 메모리 요구 사항을 크게 줄이면서도 경쟁력 있는 성능을 유지한다. 원래 모델의 지식을 활용하여 저순위 행렬을 훈련하고, 부드러운 레이어 대체를 위해 이전 연구에서 효과가 입증된 점진적 모듈 교체 전략을 따른다.

- **Performance Highlights**: DeltaLLM은 Phi-3.5 및 Llama-3.2 모델에서 압축을 수행하여 기존의 SLM보다 더 나은 성능을 기록하였다. DeltaPhi 2.9B는 24%의 매개변수 축소에도 불구하고 유사한 평균 제로샷 정밀도를 달성하며, 모델 크기를 감소시키면서도 원래 모델의 성능을 복원하는데 효과적이다. 우리의 연구는 저장 공간이 중요한 경우 LLM 아키텍처 설계와 압축 기법에 대한 새로운 통찰력을 제공한다.



### Accuracy and Robustness of Weight-Balancing Methods for Training PINNs (https://arxiv.org/abs/2501.18582)
- **What's New**:  이 논문은 Physics-Informed Neural Networks (PINNs)에 대한 새로운 훈련 알고리즘을 제안합니다. Primal-Dual (PD) 최적화 프레임워크를 기반으로 하여 정확성과 견고성의 명확한 정의를 통해 PINNs의 신뢰성을 향상시킵니다. 다양한 손실 함수에서 가중치를 적응적으로 조정하려는 기존 방법에 비해, 제안된 알고리즘은 이와 유사한 성능을 유지하면서 PINNs의 견고성을 증가시킵니다. 이 연구는 몇 가지 숫자 실험을 통하여 PD 방법이 신뢰할 수 있는 솔루션을 지속적으로 달성할 수 있음을 보여줍니다.

- **Technical Details**: PINNs는 물리적 법칙을 신경망의 손실 함수에 직접 통합하여 부분 미분 방정식 (PDEs)와 복잡한 동적 시스템을 모델링하는 데 사용됩니다. 기존 방법들은 초기 조건에 대한 민감성과 불안정성으로 인해 효과적인 훈련을 방해했습니다. 이 연구에서는 PINNs의 훈련 결과의 변동성과 신뢰성을 명시적으로 정량화하기 위해 확률론적 프레임워크를 제안합니다. 이 알고리즘은 매개변수 초기화와 훈련 역학을 최적화하여 안정성을 향상시키기 위한 원칙적인 전략을 통합합니다.

- **Performance Highlights**: 제안된 PD 알고리즘은 확인된 여러 케이스에서 신뢰할 수 있는 솔루션을 지속적으로 달성했습니다. 이 방법은 기존의 가중치 균형 PINN 알고리즘과 비교하여 그 강점과 약점을 명확히 분석합니다. 새로운 교육 절차는 낮은 계산 부담으로도 높은 성능을 유지할 수 있음을 입증하였습니다. 이 연구는 실용적인 적용을 위한 효과적인 방법으로 자리잡을 가능성이 높습니다.



### Bias-variance decompositions: the exclusive privilege of Bregman divergences (https://arxiv.org/abs/2501.18581)
- **What's New**: 본 논문은 머신러닝 모델의 일반화 성능을 이해하는데 있어 편향-분산(bias-variance) 분해의 새로운 접근법을 제시합니다. 특히, 기존의 손실 함수들이 갖는 제한점을 극복하기 위해, 연속적이고 비부정적인 손실 함수에 대해 연구하였습니다. 그 결과, $g$-Bregman divergence가 편향-분산 분해를 갖는 유일한 손실 함수임을 입증했습니다.

- **Technical Details**: 우리는 가벼운 정규성 조건을 만족하는 연속적이고 비부정적인 손실 함수에 대해 분석했습니다. $g$-Bregman divergence는 변환 가능한 변수를 통해 표준 Bregman divergence로 변환될 수 있으며, 이는 특정 대칭 손실 함수의 편향-분산 분해의 청정함을 보장합니다. 이로써, Mahalanobis 거리와 같은 손실 함수가 유일한 대칭 손실 함수로 나타납니다.

- **Performance Highlights**: 이 연구는 Bregman divergence의 폭넓은 범주에 대한 편향-분산 분해의 가능성을 탐구했습니다. 손실 함수에 대한 제한을 완화할 경우 결과의 영향을 살펴보았으며, 이는 이후 머신러닝 모델의 성능 분석에 중요한 통찰을 제공합니다. 논문에서 제공된 결과들은 다양한 손실 함수의 적용에 있어 중요한 지침이 될 수 있습니다.



### Node Classification and Search on the Rubik's Cube Graph with GNNs (https://arxiv.org/abs/2501.18580)
- **What's New**: 본 연구는 3x3x3 루빅스 큐브를 해결하기 위한 딥 기하학 모델의 적용에 초점을 맞추고 있습니다. 특히, 그래프 신경망(Graph Neural Networks, GNN)을 사용하여 거리 근사화 작업을 노드 분류 문제로 재구성하는 방법을 제시하고 있습니다. 이를 통해 모델이 A* 검색 알고리즘에서 사용될 휴리스틱을 학습할 수 있습니다.

- **Technical Details**: 루빅스 큐브 그룹(G,⋅)을 정의하고, 이를 바탕으로 케이리 그래프(Cayley graph)를 구축합니다. 각 노드는 그룹 G의 요소로 구성되며, 해결된 구성을 나타내는 루트 g0이 설정됩니다. 이 그래프는 두 갈래 이동을 보장하기 위해 무방향 그래프로 간주하며, G의 모든 구성에서 g0까지의 경로를 평가할 수 있게 합니다.

- **Performance Highlights**: 연구에서는 GNN을 통해 예측된 클래스가 A* 검색을 위한 일관된 휴리스틱을 구축하는 데 사용된다고 언급합니다. 이 모델은 다양한 노드 구조를 식별해 냈으며, 이를 통해 큐브의 전환 동역학을 이해하고 여러 유형의 노드를 구분할 수 있습니다. 마지막으로, 실험 결과를 통해 제안된 접근 방식의 효과성과 그 의미를 강조하고 있습니다.



### Token-Hungry, Yet Precise: DeepSeek R1 Highlights the Need for Multi-Step Reasoning Over Speed in MATH (https://arxiv.org/abs/2501.18576)
Comments:
          5 pages, 1 figure, 1 table

- **What's New**: 이번 연구는 DeepSeek R1 언어 모델이 MATH 데이터셋에서 가져온 30개의 도전적인 수학 문제에 대한 성능을 조사합니다. 이전 모델들이 시간 제약 속에서 해결할 수 없었던 문제들로, 이 연구는 시간 제한을 제거하여 DeepSeek R1의 아키텍처가 멀티 스텝 프로세스를 통해 정확한 솔루션을 달성할 수 있는지를 탐구합니다. 또한, 연구는 DeepSeek R1을 다른 네 가지 모델과 비교하고 있습니다.

- **Technical Details**: DeepSeek R1은 토큰 기반 추론(token-based reasoning)에 의존하는 구조를 가지고 있으며, 11개의 온도 설정(temperature settings) 하에서 성능을 시험합니다. 연구에서는 DeepSeek R1이 gemini-1.5-flash-8b, gpt-4o-mini-2024-07-18, llama3.1:8b, mistral-8b-latest와 비교됩니다. 결과적으로 DeepSeek R1은 복잡한 문제들에서 뛰어난 정확성을 보여주지만, 다른 모델들보다 많은 토큰을 생성하는 경향이 있습니다.

- **Performance Highlights**: 실험 결과, DeepSeek R1은 이러한 수학 문제들에서 우수한 정확도를 달성하지만, 토큰 생성량이 상당히 많아지는 것을 확인했습니다. 이는 정확도와 효율성 간의 트레이드오프(trade-off)가 존재함을 나타냅니다. 연구는 LLM 선택 시 작업별 요구사항(task-specific requirements)을 고려하는 것이 중요하다고 강조하며, 온도 설정이 성능 최적화에 중요한 역할을 한다고 전합니다.



### No Equations Needed: Learning System Dynamics Without Relying on Closed-Form ODEs (https://arxiv.org/abs/2501.18563)
Comments:
          To appear in the Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025)

- **What's New**: 본 논문은 통상적인 두 단계 모델링 프로세스로부터 벗어나 직접적인 의미론적 모델링 방법을 제안합니다. 이 방법은 기존의 폐쇄형 미분 방정식(ODE) 발견 후 분석하는 과정 없이 데이터를 통해 역동계의 의미론적 표현을 직접 예측합니다. 이러한 접근 방식은 복잡한 후속 분석의 필요성을 없애주며, 사용자에게 직관적인 조정 및 시스템의 행동을 반영하는 제약 조건의 통합이 가능하게 합니다.

- **Technical Details**: 모델링의 성격을 이해하기 위해, 전통적인 ODE 발견 및 분석 방식의 단계적 접근에서 벗어나, 직접적인 데이터 기반의 의미론적 모델링을 통해 모델의 행동을 설명할 수 있습니다. 각 관찰된 궤적 데이터셋은 𝒟로 표현되며, 이는 추정된 ODE의 확률적 특성을 고려하여 회귀 모델로 재구성될 수 있습니다. 이러한 모델은 구체적인 시스템의 의미론적이고 직관적인 표현을 제공하여 여러 도메인에서 활용될 수 있습니다.

- **Performance Highlights**: 직접적 의미론적 모델링 방법은 전통적인 ODE 모델에 비해 투명성과 유연성을 향상시킵니다. 제안된 모델은 과학적 문제 해결에 있어 단순화된 모델링 파이프라인을 제공하며, 최적화를 통해 직관적인 편향을 모델에 통합할 수 있어 원하는 사양을 충족할 수 있게 됩니다. 또한 이 접근법은 특정 요구 사항을 충족하기 위해 발견된 방정식을 수정하는 데 있어 유연성을 제공합니다.



### Bandits with Anytime Knapsacks (https://arxiv.org/abs/2501.18560)
- **What's New**: 이 논문에서는 anytime knapsacks (BwAK)라는 새로운 유형의 bandit 문제를 고려합니다. 총 비용 예산 대신 average cost 제약조건이 도입되어 추후 결정 과정에서 추가적인 복잡성이 발생합니다. 연구자들은 탐색과 활용 간의 균형을 유지하면서 최적의 arm 조합을 식별하는 상위 신뢰 구간 (upper confidence bounds)을 활용하는 SUAK이라는 알고리즘을 제안합니다.

- **Technical Details**: SUAK 알고리즘은 문제를 해결하기 위해 upper confidence bounds를 사용하며, anytime 비용 제약 조건을 만족시키기 위해 사용할 수 있는 예산을 전략적으로 약간 미비하게 활용합니다. 이 알고리즘은 각 라운드에서 예산을 최대한 활용하고, anytime 비용 제약을 위반할 가능성이 있는 경우 라운드를 건너뛰도록 설계되었습니다. SUAK은 이전 연구에서 설정된 O(K log T)라는 문제 의존적인 regret upper bound를 달성함을 보여줍니다.

- **Performance Highlights**: SUAK은 naively BwK 알고리즘을 BwAK 설정에 맞게 수정하여 얻는 경우와 비교하여 유의미한 성능 개선을 발견하였습니다. 성능 검증을 위해 수행된 시뮬레이션 결과를 통해 실제 환경에서도 SUAK의 유용성을 입증하였습니다. 이는 임의의 초기 라운드를 건너뛰지 않고 최소한의 regret을 극대화할 수 있는 가능성을 제시합니다.



### Loss Functions and Operators Generated by f-Divergences (https://arxiv.org/abs/2501.18537)
- **What's New**: 이 연구에서는 전통적인 logisitic loss, 즉 cross-entropy loss의 일반화를 위해 새로운 convex loss function을 제안하고 있다. 제안된 loss function은 KL divergence를 f-divergence로 대체하고, 비균일 기준 분포를 허용함으로써 기존 구조를 확장한다. 우리는 여러 유형의 f-divergence를 사용하여 기존 loss를 재구성하고 새로운 loss를 생성하는 방법을 제시한다. 이러한 접근은 자주 사용되는 softargmax 연산자에 새로운 논리를 통합하는 결과를 가져온다.

- **Technical Details**: 연구에서는 f-divergence를 기반으로 한 편리하고 효과적인 bisection 알고리즘을 개발하였다. 이 알고리즘은 현대 하드웨어에서 잘 병렬화되며, 다양한 f-divergence에 대한 softargmax를 계산하는 데 사용할 수 있다. 우리가 제안하는 loss function은 기존 sparsemax 및 entmax와 같은 기존 구조를 일반화하며, 완전히 새로운 loss function을 생성하는 도구로 작용한다.

- **Performance Highlights**: 제안된 loss function은 이미지 분류, 언어 모델의 post-training 및 distillation 실험에서 효과적인 성능을 보여주었다. 특히, α-divergence를 기반으로 한 loss function이 α=1.5일 때 여러 작업에서 좋은 성능을 발휘하는 것을 확인하였다. 이는 전통적인 cross-entropy를 넘어 더 뛰어난 성능을 달성할 수 있는 가능성을 시사한다.



### A Hybrid Data-Driven Approach For Analyzing And Predicting Inpatient Length Of Stay In Health Centr (https://arxiv.org/abs/2501.18535)
Comments:
          8 pages, 15 figures

- **What's New**: 이번 연구에서는 병원 경영의 효율성을 평가하는 중요한 지표인 환자의 병원 체류 기간(LoS) 최적화를 위한 포괄적인 프레임워크를 제안합니다. 데이터 기반 기법과 시뮬레이션 방법론을 통합하여 230만 건의 비동의 환자 기록을 분석하였으며, 이를 통해 환자의 흐름 최적화 및 자원 활용 극대화를 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 머신러닝 모델(Decision Tree, Logistic Regression, Random Forest, Adaboost, LightGBM)과 Python 도구(Spark, AWS clusters, 차원 축소)를 활용하여, 환자가 입원 시 예측되는 병원 체류 기간(LoS)을 분석했습니다. 감독 학습 알고리즘을 사용하여 LoS에 영향을 미치는 주요 요인을 식별하고, 이를 통해 환자 흐름과 자원 활용을 개선할 수 있는 강력한 프레임워크를 제공합니다.

- **Performance Highlights**: 연구 결과, 실제 의료 환경에서 환자의 병원 체류 기간이 감소하는 효과가 나타났습니다. 하이브리드 데이터 기반 모델이 병원 경영 관행을 혁신할 가능성을 강조하며, 이는 의료 행정과 환자 만족도 제고에 큰 의미를 가질 것으로 기대됩니다.



### Joint Learning of Energy-based Models and their Partition Function (https://arxiv.org/abs/2501.18528)
- **What's New**: 이 논문에서는 에너지 기반 모델(EBMs)에서 파라미터화된 확률 분포를 Learning하는 새로운 접근 방식을 제안합니다. 특히, 조합적으로 큰 이산 공간에서의 최적화 문제를 해결하기 위해, 에너지 모델과 로그 파르티션을 함께 학습하는 방법을 소개합니다. 이러한 접근 방식은 MCMC에 의존하지 않고, 확률적 경량 하강법(SGD)을 통해 tractable한 objective criterion을 제공합니다.

- **Technical Details**: 제안된 방법은 에너지 모델과 로그 파르티션을 협력적으로 학습하는 미니-미니(min-min) 형식을 따릅니다. 네트워크 파라미터를 최적화하는 과정에서 수치적 최소화를 사용하고, 유일한 가정은 주어진 분포에서 샘플링할 수 있는 것입니다. 우리 접근법은 Fenchel-Young 손실 함수의 일반화에도 적용 가능하며, 이는 sparsemax 손실을 사용하는 최초의 tractable한 방법이기도 합니다.

- **Performance Highlights**: 이 방법은 다중 레이블 분류와 레이블 순위 문제에서 실험적으로 검증되었습니다. 이 논문은 또한 이론적 측면에서 우리가 제안한 방법이 연속 함수 공간에서 이상적인 최대우도 추정(MLE) 솔루션을 회복할 수 있음을 보여줍니다. 우리의 접근 방식에서는 학습된 로그 파르티션 네트워크가 신경망의 보편적 근사 특성을 활용할 수 있다는 것을 경험적으로 입증했습니다.



### Neural Discovery in Mathematics: Do Machines Dream of Colored Planes? (https://arxiv.org/abs/2501.18527)
Comments:
          8 pages main paper, 10 pages references and appendix, 17 figures, 1 table

- **What's New**: 이번 연구는 Hadwiger-Nelson 문제라는 고전적인 조합 기하학 문제를 해결하는 데 신경망(Neural Networks)을 활용하는 방법을 제시합니다. 이 문제는 단위 거리 쌍을 피하여 평면을 색칠하는 것이며, 이를 새로운 접근 방식으로 재정의하였습니다. 또한, 30년간의 정체기를 깨고 새로운 여섯 가지 색칠 방법을 발견한 것이 큰 성과입니다.

- **Technical Details**: 연구진은 이 문제를 최적화(task) 문제로 재구성하며, 확률적(probabilistic)인 미분 가능한 손실 함수(differentiable loss function)를 활용합니다. 이를 통해 경량 기반 그래디언트(gradient) 탐색을 가능하게 하고, 허용 가능한 구성(configuration)을 효과적으로 탐색할 수 있게 하였습니다. 이러한 접근 방법은 기하학적 색칠 문제에 대한 혼합 이산-연속(Mixed Discrete-Continuous) 문제를 다루는 혁신적인 기술로 자리매김하고 있습니다.

- **Performance Highlights**: 연구 결과, 두 가지 새로운 여섯 가지 색칠 방법이 제안되었으며, 이는 원문제(original problem)의 비대각선(off-diagonal variant) 변형에 대해 첫 번째 개선을 이룬 것입니다. 이러한 결과는 30년간의 정체를 깬 중요한 발견으로 평가되며, 해당 기법이 다른 연구에도 광범위하게 적용될 수 있는 가능성을 열어줍니다.



### WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training (https://arxiv.org/abs/2501.18511)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델)의 효과적인 후속 학습 방법론을 위한 새로운 데이터셋인 WildChat-50M을 소개합니다. 이 데이터셋은 50개 이상의 다양한 오픈 웨이트 모델로부터 생성된 대화 전사 파일을 포함하고 있으며, 총 1억 2500만 개 이상의 대화 전사를 포함합니다. 연구자들은 이 데이터셋을 바탕으로 RE-WILD라는 새로운 감독된 미세 조정(SFT) 믹스를 선보이며, 이는 Allen AI의 Tulu-3 SFT 믹스를 초과하는 성과를 거두었습니다.

- **Technical Details**: WildChat-50M 데이터셋은 약 두 달에 걸쳐 12x8 H100 공유 연구 클러스터에서 수집되었습니다. 수집 과정에서 사용된 모델들은 VLLM 프레임워크를 통해 다양한 GPU에서 최적화되어 있으며, 현재까지 54개의 체크포인트가 포함되어 있습니다. 이 데이터셋은 0.5B에서 104B 파라미터를 가진 다양한 모델들로 구성되어 있으며, 서로 다른 컨텍스트 윈도우 크기를 설정하여 성능을 극대화하였습니다.

- **Performance Highlights**: RE-WILD 믹스를 사용하여 Llama-3.1 8B Base 모델을 미세 조정한 결과, 기존의 SFT 믹스보다 뛰어난 성능을 보여주었습니다. 이 연구는 비교한 모델들 간의 VRAM 효율성 및 런타임 효율성에 대한 포괄적인 분석을 제공하며, 향후 데이터 수집 및 후속 학습 방법을 더욱 발전시킬 수 있는 기초 자료를 제공합니다.



### CLoQ: Enhancing Fine-Tuning of Quantized LLMs via Calibrated LoRA Initialization (https://arxiv.org/abs/2501.18475)
- **What's New**: 본 논문에서는 CLoQ (Calibrated LoRA initialization for Quantized LLMs)를 소개합니다. CLoQ는 quantized LLM에서의 LoRA 모듈의 초기화를 위한 데이터 기반 접근법으로, 작은 보정 데이터셋을 활용하여 레이어 간의 차이를 최소화하는 것을 목표로 합니다. 이 방법은 모델의 모든 레이어에서 최적의 LoRA 구성 요소를 결정하여 모델의 성능을 극대화합니다.

- **Technical Details**: CLoQ의 핵심은 두 가지 단계를 포함하는데, 첫 번째는 post-training quantization 단계로 quantized weights를 얻는 것이고, 두 번째는 linear transformation 하에 일반화된 low-rank 근사(th정확한 LoRA 구성 요소를 계산하는 것입니다. 이 과정에서 새로운 폐쇄형 해를 유도하였으며, 이는 두 개의 singular value decompositions (SVD)를 사용하여 효율적으로 계산됩니다. 이 방법은 역전파(back-propagation)를 필요로 하지 않음으로써 quantized 모델의 fine-tuning을 매우 효율적으로 합니다.

- **Performance Highlights**: CLoQ의 효과는 다양한 벤치마크 데이터셋을 통해 검증되었으며, 기존의 LoRA 방법들과 비교할 때 꾸준히 높은 성능을 나타냈습니다. 특히 ultra low-bit width에서 INT2 CLoQ는 Llama2-13B 모델에서 arithmetic reasoning 작업에서 INT4 QLoRA를 초과하는 fine-tuning 정확도를 기록했습니다. 이러한 결과를 통해 CLoQ는 quantized LLM의 성능을 향상시킬 수 있는 강력한 도구임을 입증하였습니다.



### Clustering Properties of Self-Supervised Learning (https://arxiv.org/abs/2501.18452)
- **What's New**: 자기 지도 학습(self-supervised learning, SSL) 방법은 표식 없음에도 불구하고 강력한 군집(cluster) 속성을 가진 의미론적으로 풍부한 표현을 포착하는 데 매우 효과적임을 보여주고 있습니다. 하지만 이러한 특성을 활용하여 SSL 방법을 향상시키려는 시도는 드물었습니다. 본 논문에서는 인코더의 출력이 다른 구성 요소와 비교하여 더 우수하고 안정적인 군집 속성을 가진다는 것을 다양한 지표를 통해 입증하였고, 이를 바탕으로 새로운 긍정적 피드백 SSL 방법인 Representation Soft Assignment (ReSA)를 제안합니다.

- **Technical Details**: ReSA는 모델의 군집 속성을 활용하여 자기 가이드를 통해 학습을 촉진하는 온라인 자기 군집(self-clustering) 메커니즘을 채택합니다. 이 방법은 인코더의 출력인 encoding에서 군집 속성을 추출하여 시너지 효과를 창출하며, 현재까지 발표된 최첨단 SSL 방법들과 비교하여 성능과 학습 효율성 모두에서 우수함을 보여주었습니다. 또한 ReSA가 군집 속성을 어떻게 개선하는지를 분석하여 세부적(fine-grained)이고 거시적(coarse-grained) 수준에서 표현을 더 구조적이고 의미론적으로 의미 있게 만든다는 것을 실증하였습니다.

- **Performance Highlights**: 다양한 표준 SSL 벤치마크에 대한 광범위한 실험 결과, ReSA로 사전 훈련된 모델은 다른 최신 SSL 방법들보다 현저하게 성능이 우수하다는 발견이 있었습니다. 특히 ReSA는 미세한 군집 인식을 통해 더 나은 성능을 유도하며, 이러한 구조적 표현의 강화는 의미론적 관계 포착에 기여합니다. 전반적으로, ReSA는 SSL의 발전에 중요한 기여를 할 것으로 기대됩니다.



### MolGraph-xLSTM: A graph-based dual-level xLSTM framework with multi-head mixture-of-experts for enhanced molecular representation and interpretability (https://arxiv.org/abs/2501.18439)
- **What's New**: 이 논문에서는 분자의 특성을 예측하기 위한 새로운 모델 MolGraph-xLSTM을 제안합니다. 이 모델은 graph 기반의 xLSTM 아키텍처를 사용하여 분자 그래프의 장거리 상호작용을 효과적으로 모델링합니다. 두 가지 수준의 그래프, 즉 원자 수준(atom-level)과 모티프 수준(motif-level)을 처리하여 보다 정확한 분자 표현 학습을 가능하게 합니다.

- **Technical Details**: MolGraph-xLSTM은 원자 수준의 GNN 기반 xLSTM 프레임워크를 이용하여 지역적 특성을 추출하고, 모티프 수준의 그래프는 보다 포괄적인 구조 정보를 제공합니다. 이 모델은 각 수준의 임베딩을 다중 헤드 전문가 혼합(multi-head mixture of experts, MHMoE) 모듈을 통해 정제하여 표현력과 성능을 향상시킵니다. xLSTM 아키텍처는 긴 거리 종속성을 효과적으로 처리하도록 설계되었습니다.

- **Performance Highlights**: MolGraph-xLSTM은 10개의 분자 특성 예측 데이터셋에서 검증되었으며, 분류(classification)와 회귀(regression) 작업 모두에서 우수한 성능을 보였습니다. BBBP 데이터셋에서는 분류 작업에서 최대 7.03% 성능 개선, ESOL 데이터셋에서는 회귀 작업에서 7.54% 성능 개선을 기록하였습니다. 평균적으로 MolGraph-xLSTM은 분류 작업에서 AUROC 3.18% 개선, 회귀 데이터셋에서 RMSE 3.83% 감소를 달성하여 모델의 효과성을 확인하였습니다.



### Guaranteed confidence-band enclosures for PDE surrogates (https://arxiv.org/abs/2501.18426)
- **What's New**: 이번 연구는 통계적으로 보장된 신뢰 구간(confidence bands)을 제공하는 새로운 방법을 제안합니다. 이 방법은 예측 오차의 저차원 표현을 활용하여 nested confidence sets를 구성하고, 이를 예측 공간으로 매핑하여 예측의 신뢰성을 향상시킵니다. 이러한 신뢰 구간은 functional surrogate model에 대해서도 적용 가능하여 AI의 신뢰성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 이 방법은 surrogate model의 예측 오차를 계산한 후, 차원 축소 기법(SVD)을 사용하여 저차원 공간으로 투영합니다. 그런 다음 zonotope이라는 집합을 기반으로 신뢰 지역(confidence regions)을 구성하여, 새로운 관측값이 주어진 신뢰 수준에서 포함될 것이라는 보장을 꾀합니다. 이 과정에서 truncation error를 제한하는 기법도 도입하여, 예측의 안전성을 강조합니다.

- **Performance Highlights**: 제안된 방법은 복잡한 Sci-ML 모델, Neural Operators 및 더 간단한 설정에서도 적용할 수 있는 모델 독립적(model agnostic) 특성을 가지고 있습니다. 실험 결과는 제안된 접근 방식이 통계적 보장을 제공하면서도 효율적인 예측을 가능하게 한다는 점을 보여주었습니다. 이러한 특성 덕분에 실제 위험 계산 및 안전-critical 시스템에 적용될 가능성이 높습니다.



### Causal Inference Real-Time Anomaly Detection with Synthetic Anomaly Monitoring (SAM) (https://arxiv.org/abs/2501.18417)
Comments:
          19 pages, 3 figures, submitted for publication

- **What's New**: 이 논문은 Synthetic Anomaly Monitoring (SAM)이라는 새로운 접근 방식을 제안합니다. SAM은 인과 추론(causal inference)에서의 합성 제어 방법(synthetic control methods)을 적용하여 이상 탐지(anomaly detection)의 정확도와 해석 가능성을 개선합니다. 이를 통해 각 특성을 통제 단위(control unit)로 취급하여 정상 행동을 모델링할 수 있습니다.

- **Technical Details**: SAM은 인과적 프레임워크 내에서의 변 deviations을 이상으로 식별합니다. 연구진은 Isolation Forest, Local Outlier Factor (LOF), k-Nearest Neighbors (kNN), One-Class Support Vector Machine (SVM)과 같은 기존 벤치마크 모델들과 SAM을 비교하기 위해 광범위한 실험을 수행했습니다. 사용된 데이터셋은 Credit Card Fraud, HTTP Dataset CSIC 2010, KDD Cup 1999 등 다양한 다섯 개의 데이터셋을 포함합니다.

- **Performance Highlights**: 실험 결과, SAM은 지속적으로 강력한 성능을 제공합니다. 이는 동적이고 복잡한 환경에서 실시간 이상 탐지(real-time anomaly detection) 도구로서의 잠재력을 강조합니다. 연구는 SAM이 이상 탐지 분야에서 기존 모델들보다 더 나은 성과를 나타낸다고 결론짓고 있습니다.



### Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation (https://arxiv.org/abs/2501.18416)
Comments:
          7 pages

- **What's New**: 이 논문은 군사 협력에서의 Federated Learning (FL) 개념을 검토하며, 큰 언어 모델(LLMs)에 대한 안전성 문제를 강조합니다. 특히, 입력 프롬프트를 악의적으로 조작하는 'prompt injection' 공격의 위협을 다루며, 이러한 공격이 작전 보안과 동맹 간 신뢰를 약화할 수 있다는 점을 부각합니다. 저자들은 기술적 및 정책적 대응 방안을 포함한 인간-AI 협력 프레임워크를 제안합니다.

- **Technical Details**: FL은 군사 작전에서 LLMs을 공동으로 훈련하면서 데이터 주권을 유지하기 위한 틀을 제공합니다. 하지만 prompt injection 공격으로 인해 악의적인 조작이 발생할 수 있으며, 이는 기밀 데이터 누출, 시스템 중단, 그리고 허위 정보 유포로 이어질 수 있습니다. 이를 효과적으로 방지하기 위해, 저자들은 red/blue 팀 전쟁 게임과 품질 보증을 통해 적대적 행동을 탐지하고 완화하는 기술적 대책을 마련합니다.

- **Performance Highlights**: FL을 통해 여러 국가가 데이터 보안을 유지하면서도 협력적으로 학습하고 의사결정을 개선할 수 있습니다. 각 국가의 고유한 전투 환경과 공격 정보를 통합함으로써, FL의 활용은 AI 모델의 적응성을 높이고 실전 환경에서의 효율성을 향상시킵니다. 본 연구는 이러한 변화를 이해하고 체계적인 방안을 마련하여 동맹 간 신뢰를 강화하는 것이 필수적임을 강조합니다.



### Segmentation of cracks in 3d images of fiber reinforced concrete using deep learning (https://arxiv.org/abs/2501.18405)
- **What's New**: 이 논문은 콘크리트 구조에 생기는 균열을 분석하기 위한 새로운 접근 방식인 3D U-Net을 사용하여 크랙 구조를 세분화하는 방법을 제안합니다. 기존의 균열 특성을 결정하는 표준화된 테스트 외에도, 논문에서는 콘크리트 표면에서 균열 관측 시 활용할 수 있는 추가 구조적 정보를 강조합니다. 또한, 실시간으로 샘플을 관찰할 수 있는 Computed Tomography 기술을 활용하여 미세 구조를 손상시키지 않고 이미지를 획득하는 방법을 소개합니다.

- **Technical Details**: 본 연구에서 사용한 3D U-Net은 3D 이미지에서 크랙을 효과적으로 탐지하기 위하여 고안되었습니다. 연구진은 실제 콘크리트 샘플과 시뮬레이션된 크랙 구조로 구성된 반합성(semisyntehtic) 3D 이미지에 대해 네트워크를 훈련시켰습니다. 이를 통해, 네트워크가 다양한 유형의 실제 콘크리트, 특히 섬유 강화 콘크리트의 균열 시스템을 탐지할 수 있도록 하는 방법을 설명합니다.

- **Performance Highlights**: 훈련된 3D U-Net은 다양한 실제 콘크리트 샘플에서의 크랙 시스템을 효과적으로 감지할 수 있음을 보여줍니다. 분석 결과, 개발된 알고리즘은 크랙 구조의 세분화 및 특성 분석에서 높은 정확도를 보였습니다. 이를 통해 콘크리트 구조물의 감시와 유지보수에 있어 중요한 데이터를 제공합니다.



### Improved Replicable Boosting with Majority-of-Majorities (https://arxiv.org/abs/2501.18388)
Comments:
          13 pages

- **What's New**: 이번 연구에서 제안된 새로운 복제 가능(boosting) 알고리즘은 이전 알고리즘에 비해 샘플 복잡도(sample complexity)를 크게 개선합니다. 이 알고리즘은 이중 다수결(majority voting) 방식을 활용하여, Impagliazzo et al. (2022)에서 소개된 향상된 복제 가능 알고리즘을 기반으로 하여 하위 계층에서 작동합니다.

- **Technical Details**: 복제 가능성(replicability)은 연구 결과를 재현할 수 없는 문제인 재현성 위기(reproducibility crisis)에 대응하여 소개된 개념입니다. 알고리즘이 복제 가능하다는 것은 같은 분포에서 추출된 서로 다른 입력 데이터를 사용할 때, 높은 확률로 동일한 결과를 출력해야 함을 의미합니다. 본 연구에서는 약한 학습(weak learning)에서 강한 학습(strong learning)으로의 복제 가능성을 개선한 알고리즘을 다룹니다.

- **Performance Highlights**: 제안된 rMetaBoost 알고리즘은 기존 알고리즘보다 높은 정확도를 달성하며, 수치적 성능 수치를 통해 그 효과가 입증되었습니다. 이 알고리즘은 부스팅(Boosting) 과정에서의 샘플 활용도를 극대화하여, 더 나은 학습 성능을 보여 준다. 실험적 결과는 본 알고리즘이 다양한 데이터 분포에서 어떻게 더 높은 신뢰성을 제공하는지를 명확히 보여줍니다.



### Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces (https://arxiv.org/abs/2501.18373)
- **What's New**: 이 논문에서는 transfer learning에 대한 새로운 기하학적 정의를 제시하고, 힐베르트 공간(Hilbert spaces)을 기반으로 한 세 가지 inductive transfer 유형을 정의합니다. 이러한 유형은 격자(GH) 내 보간(interpolation), 선형 공간(linear span)으로의 외삽(extrapolation), 그리고 이 공간 밖으로의 외삽으로 구성됩니다. 제안된 방법은 function encoder 이론에 기초하여 이 세 가지 transfer를 모두 수행할 수 있게 합니다.

- **Technical Details**: 논문에서는 function encoder를 통해 새로운 학습 스킴을 제안하며, 이를 최소 제곱(least-squares) 최적화를 통해 훈련합니다. 또한, function encoders에 대한 보편 근사 정리(universal approximation theorem)를 증명하고, 기존 방법들과의 비교를 통해 이 접근 방식이 Hilbert 공간 이전 학습 설정에서 어떻게 일반화되는지를 설명합니다. 이러한 접근법은 기존의 메타 학습(meta-learning) 및 transformer 대비 우수한 성능을 보입니다.

- **Performance Highlights**: 실험 결과, function encoder는 네 가지 벤치마크 작업에서 기존의 state-of-the-art 방법들보다 우수한 성능을 보여주며, 모든 세 가지 transfer 유형에서 효과적으로 작동합니다. 특히, Type 2 및 Type 3 transfer에서의 성능 향상이 두드러지며, 이로 인해 다양한 새로운 작업에 대한 적응력이 향상됩니다. 종합적으로, 본 연구는 transfer learning의 구조적 접근 방식이 기존 데이터 기반 방법보다 더 효과적임을 입증합니다.



### A Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation (https://arxiv.org/abs/2501.18369)
- **What's New**: 이 논문에서는 Anisotropic Displacement Parameters (ADPs)를 예측하는 새로운 그래프 신경망 모델인 CartNet을 소개합니다. CartNet은 크리스탈 구조의 원자 기하학을 Cartesian 좌표에 인코딩하며, 온도와의 통합을 통해 ADP 예측의 정확도를 높이고 계산 비용을 획기적으로 줄입니다. 또한, SO(3) 회전 방식을 적용한 데이터 증강 전략을 통해 다양한 회전 방향 에 대한 예측 능력을 향상시켰습니다.

- **Technical Details**: CartNet은 원자 간 상호작용과 화학 결합을 강조하는 이웃 동등화 기법을 통합하고, Cholesky 분해 기법을 활용하여 유효한 ADP 예측을 보장하는 출력을 제공합니다. 이 모델은 3D 기하학을 사용하여 정확하게 원자 특성을 예측하며, unit cell에 대한 부가적인 인코딩 없이도 크리스탈의 ADP를 예측할 수 있는 장점을 가지고 있습니다. ADPs는 확률 분포와 관련된 고유한 구조적 특성을 지니며, 이를 이해하는 데 중요한 역할을 합니다.

- **Performance Highlights**: CartNet은 Cambridge Structural Database (CSD)에서 수집된 20만 개 이상의 실험 크리스탈 구조를 기반으로 한 ADP 데이터셋에서 기존 방법보다 10.87% 향상된 결과를 보였습니다. 또한, Jarvis Dataset과 Materials Project Dataset에서도 각각 7.71%와 13.16%의 성능 개선을 달성하여 다양한 크리스탈 속성 예측에서 최첨단 솔루션으로 자리 잡았습니다. 이러한 결과는 크리스탈 구조 분석과 관련된 계산의 효율성을 크게 향상시킬 것입니다.



### A Learnable Multi-views Contrastive Framework with Reconstruction Discrepancy for Medical Time-Series (https://arxiv.org/abs/2501.18367)
Comments:
          15 pages,6 figures

- **What's New**: 본 논문에서는 의료 시계열 데이터의 진단 정확도를 향상시키기 위해 두 가지 주요 문제를 해결하는 새로운 접근 방식을 제안합니다. 첫 번째로, AE-GAN을 활용하여 외부 데이터를 통합하고 모델 일반화를 위한 교차 센터 지식 전이를 수행합니다. 두 번째로, LMCF(변화 가능한 다중 관점 대조 프레임워크)를 소개하여 다양한 시점의 정보를 통해 자료를 효과적으로 학습할 수 있도록 합니다.

- **Technical Details**: 의료 시계열 데이터는 클리닉에서 수집한 데이터가 제한적이기 때문에 라벨이 없는 데이터를 사용하여 자가 감독(self-supervised) 대조 학습을 통해 모델을 사전 학습합니다. AE-GAN은 목표 데이터의 불일치를 재구성하여 질병 확률로 변환하며, 이를 대조 학습에 통합하여 목표 표현 학습을 개선합니다. LMCF는 다중 헤드 어텐션(mult-head attention) 메커니즘을 통합하여 인과 및 시간적 대조 학습을 통해 대표성을 높이는 방향으로 설계되었습니다.

- **Performance Highlights**: 세 가지 주요 데이터셋(심근경색, 알츠하이머병, 파킨슨병)에 대해 우리의 방법이 기존 7개 기준 모델을 지속적으로 초과하는 성능을 보여주었습니다. 특히, 데이터의 10%만 라벨이 있는 극단적인 상황에서도 우리의 방법이 우수한 성능을 발휘하는 것으로 나타났습니다. 따라서 본 연구 결과는 해당 분야의 다양한 의료 진단 응용에 중요한 의미를 가집니다.



### Robust Online Conformal Prediction under Uniform Label Nois (https://arxiv.org/abs/2501.18363)
- **What's New**: 이번 연구에서는 기존의 온라인 변별 예측(Online Conformal Prediction) 방법이 라벨 노이즈(Label Noise)에 대해 얼마나 강인한지를 조사합니다. 특히, 알려진 노이즈 비율을 바탕으로 라벨 노이즈가 존재할 때의 문제점을 다루고 있으며, 기존 알고리즘들이 가정하는 라벨 정확도가 실제로는 잘 유지되지 않음을 강조합니다. 이를 통해 사용자들이 보다 부정확한 데이터를 갖고도 신뢰할 수 있는 예측 집합을 얻을 수 있도록 합니다.

- **Technical Details**: 제안하는 방법은 Noise Robust Online Conformal Prediction(NR-OCP)으로, 새로운 강건 핀볼 손실(Robust Pinball Loss) 업데이트 방식을 이용하여 라벨 노이즈에 대한 강건성을 확보합니다. NR-OCP는 펀드라 불리는 예측 집합의 신뢰도를 증가시키는데, 라벨의 진실성을 요구하지 않고도 신뢰할 수 있는 손실 추정치를 제공합니다. 이론적인 분석에 따르면 NR-OCP는 고정 및 동적 학습률 스케쥴 하에서 커버리지 격차(Coverage Gap)를 제거할 수 있습니다.

- **Performance Highlights**: NR-OCP 방법은 실험에서 높은 정확도의 커버리지와 향상된 효율성을 달성하는 것으로 나타났습니다. 특히, 라벨 노이즈 조건 아래에서도 실제 비커버리지 비율(Mis-coverage Rate)과 원하는 비율(Desired Rate) 사이의 격차를 없애 어 더 나은 커버리지 보장을 제공합니다. 즉, 뉴럴 네트워크 기반의 모델들이 대규모 데이터에서 믿을 수 있는 예측을 제공할 수 있도록 기여합니다.



### Contrastive Learning Meets Pseudo-label-assisted Mixup Augmentation: A Comprehensive Graph Representation Framework from Local to Globa (https://arxiv.org/abs/2501.18357)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문에서는 Comprehensive Graph Representation Learning (ComGRL)라는 새로운 프레임워크를 제안합니다. ComGRL은 지역 정보(local information)를 글로벌 정보(global information)와 통합하여 강력한 표현을 도출합니다. 이 접근은 글로벌 탐색을 위한 신뢰할 수 있는 표현을 보장하기 위해 유연한 그래프 대조 학습(graph contrastive learning)을 통해 암묵적으로 지역 정보를 부드럽게 만드는 방법을 포함합니다.

- **Technical Details**: 이 프레임워크는 MLPs(다층 퍼셉트론)를 기능 추출기로 활용하고, 그래프 대조 학습을 통해 노드 표현을 유연하게 조정합니다. ComGRL은 지역 표현을 다중 헤드(self-attention) 모듈에 전달하여 글로벌 정보 집합을 위한 강력한 입력을 제공합니다. 이를 통해 지역 설명과 글로벌 설명 간 상호작용을 지원하는 자가 감독(pseudo-labels) 방식의 Mixup 증강 메커니즘을 구현합니다.

- **Performance Highlights**: 여섯 개의 널리 사용되는 그래프 데이터셋에서 수행된 실험 결과, ComGRL은 노드 분류 작업에서 뛰어난 성능을 달성하였습니다. 특히, ComGRL은 지역 및 글로벌 정보를 효과적으로 통합하여 모델의 분별력을 높이는 데 기여하고 있습니다. 이러한 성과는 GNN 기반 모델 연구에 중요한 이정표가 될 것으로 예상됩니다.



### State Stream Transformer (SST) : Emergent Metacognitive Behaviours Through Latent State Persistenc (https://arxiv.org/abs/2501.18356)
Comments:
          25 pages, 3 figures

- **What's New**: 이번 논문에서는 State Stream Transformer(SST)라는 새로운 LLM 아키텍처를 소개합니다. 이 모델은 전통적인 transformer 모델의 한계를 극복하고, 자가 회귀적 생성 과정에서의 계산적 연속성을 유지함으로써, 학습된 가중치에서 잠재적인 추론 행동을 드러냅니다. SST는 슬라이딩 윈도우 방식의 잠재 상태(FFN) 캐시를 도입해 지속적인 잠재 과정의 유지 및 진화를 가능하게 합니다.

- **Technical Details**: SST는 모든 선형층에서 가중치를 감소시키면서 슬라이딩 윈도우 잠재 상태 캐시를 구현합니다. 이 구조는 토큰 생성 중에 지속적인 '상태 흐름'을 유지하여 모델이 정보를 처리하는 방식에 근본적인 변화를 가져옵니다. 이를 통해 SST는 자가 회귀 모델에 비해 향상된 추론 능력을 보여주며, 이는 메타인지 행동을 통해 증명됩니다.

- **Performance Highlights**: 정량적 평가 결과, SST는 GSM-8K(0-shot)에서 89.01%의 정확도와 ARC Challenge(0-shot CoT)에서 91.04%의 정확도로 기본 모델에 비해 상당한 성과 향상을 이뤘습니다. 이러한 결과는 잠재 상태에서의 지속적 계산이 정보 처리 및 내부 추론 전략에 근본적인 차이를 만든다는 것을 시사합니다. SST 아키텍처는 인공지능 시스템의 능력 및 인공지능 인지에 대한 이해에 중요한 함의를 가집니다.



### Transfer Learning of Surrogate Models: Integrating Domain Warping and Affine Transformations (https://arxiv.org/abs/2501.18344)
- **What's New**: 이 논문에서는 기존의 surrogate 모델을 새로운 작업으로 전이하는 접근 방식을 확장하여 다루고 있습니다. 특히, 선형 및 비선형 변환을 포함하는 더 넓은 범위의 변환을 다루며, 특히 β 누적 분포 함수(β CDF)를 사용하여 알려지지 않은 입력 왜곡(input warping)을 고려합니다. 이 모델은 적은 수의 데이터 포인트로 최적화를 수행하여 전이 데이터셋에서의 경험적 손실(empirical loss)을 최소화함으로써 트랜스퍼 학습을 실현합니다.

- **Technical Details**: 제안된 방법은 전이 학습을 통해 새로운 문제에 대한 정확한 surrogate 모델을 조정하는 것입니다. 이 과정에서는 선형 및 비선형 변환을 조합하여 도메인 전이를 신속하게 수행합니다. 특히, β CDF를 이용한 비선형 함수 g를 채택하여 자동차 산업의 복잡성을 모델링하며, 이로 인해 실험적으로 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 결과적으로, 제안된 모델은 원래의 surrogate 모델과 단순히 트랜스퍼 데이터셋을 사용하여 새로 구축한 모델보다 월등히 뛰어난 성능을 보였습니다. 특히 데이터가 부족한 상황에서도 더 정확한 예측을 제공하며, Black-Box Optimization Benchmark (BBOB) 테스트베드와 실제 자동차 산업의 전이 학습 과제에서 그 효과를 검증하였습니다. 결과적으로 이 모델은 실제 문제 해결에 있어 유망한 대안으로 주목받고 있습니다.



### Stream-Based Monitoring of Algorithmic Fairness (https://arxiv.org/abs/2501.18331)
Comments:
          31st International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2025)

- **What's New**: 본 연구에서는 알고리즘적 공정성(algorithmic fairness)의 실행 시간(runtime)에서의 검증을 위한 스트림 기반 모니터링(stream-based monitoring) 방법론을 제안합니다. 이를 통해, 대출 신청자나 피고의 재범 위험을 예측하는 시스템들이 반드시 사회 집단에 대한 편견 없이 작동하도록 강제합니다. COMPAS와 같은 실제 사례를 통해 제안한 방법의 효용성을 입증하였습니다.

- **Technical Details**: 저자들은 RTLola라는 명세 언어를 사용해 시간 기반 데이터 스트림에서 알고리즘적 공정성을 정형화하는 방법을 제시합니다. 스트림 이벤트를 독립적인 실험으로 분할하고, 알고리즘적 공정성 명세와 관련된 조건부 확률을 추정하는 방식으로 모니터링을 수행합니다. 이로써 실시간 이벤트에 대한 고유한 논리적 특성을 다룰 수 있으며, 높은 데이터 처리량을 요구하는 환경에서도 효율적으로 작동할 수 있습니다.

- **Performance Highlights**: 실험 결과, RTLola는 전통적인 데이터베이스 접근 방식보다 뛰어난 성능을 보였으며, 알림을 통해 불공정한 시스템의 결정을 사전에 경고할 수 있는 능력을 갖추었습니다. 특히, COMPAS 데이터를 포함한 실제 사례를 평가하여, 불공정한 예측 시스템의 조기 경고 시스템으로서의 가능성을 검증하였습니다. 이러한 접근은 공정성 사양을 문서화하고 검증할 수 있는 중요한 도구로 자리매김할 것입니다.



### A Unified Perspective on the Dynamics of Deep Transformers (https://arxiv.org/abs/2501.18322)
- **What's New**: 이번 논문은 Transformers 모델에서의 동적 과정을 보다 깊이 이해하기 위해, 입력 시퀀스를 확률 측도로 식별하고 그 진화를 Vlasov 방정식으로 모델링합니다. 이를 통해 self-attention의 다양한 변형들에 대한 의미 있는 분석을 제시하며, Gaussian 초기 데이터를 포함한 비압축 초기 조건에 대해서도 처음으로 연구합니다.

- **Technical Details**: 연구는 주로 self-attention 블록만 포함된 단순화된 모델을 고려하며, 여기에 L2, Sinkhorn, Sigmoid 등의 다양한 self-attention 변형에 대한 분석을 포함합니다. Transformer PDE는 상호작용 입자 시스템의 평균장 한계를 가지며, 특히 Gaussian 측정의 공간을 보존하는 특성에 대해 강조합니다.

- **Performance Highlights**: 논문에서 제시된 Gaussian 분석은 딥 Transformer 내에서의 데이터 이방성(인접성) 진화를 포착하며, 이를 통해 클러스터링 현상이라는 흥미로운 행동을 규명합니다. 이러한 결과는 전통적인 상관관계와 더불어 비정규화된 이산 경우에서 나타나는 결과와 유사성을 보여줍니다.



### Efficient Neural Theorem Proving via Fine-grained Proof Structure Analysis (https://arxiv.org/abs/2501.18310)
- **What's New**: 이번 논문에서는 깊은 학습 모델과 전통적인 자동화 도구의 시너지를 통해 효율적인 신경 정리 증명기(Neural Theorem Provers, NTP)를 개발하는 새로운 방법인 ProofAug를 제안합니다. ProofAug는 모델이 생성하는 증명 제안의 세밀한 구조 분석을 통해 자동화 방식을 다양한 세부 수준에서 적용하여 샘플 효율성을 높입니다. 이 방법은 기존의 트리 검색 알고리즘과 원활하게 통합될 수 있는 유연성을 제공하여 효율적인 재귀 증명 모듈도 구성할 수 있게 합니다.

- **Technical Details**: ProofAug는 LLM의 증명 생성 과정에서 세밀한 구조 분석을 통해 증명 제안의 최대 호환 반증명(maximal compatible semi-proof, MCSP)을 찾습니다. 이 과정에서, ATP가 실패하는 경우에는 더 큰 반증명(coarse semi-proof)으로 재귀적으로 돌아가 문제를 해결하고 있습니다. 이러한 방식은 복잡한 초안 problemas를 줄이고, 복잡한 스마트 방법의 필요성을 감소시켜 자동화 방법의 효율성을 극대화합니다.

- **Performance Highlights**: ProofAug는 miniF2F-test 벤치마크에서 61.9%의 누적 합격률을 기록하였으며, 데이터 세트를 정제한 후 66.0%의 합격률을 달성하여 모든 증명 언어에서 새로운 SOTA를 세웠습니다. 대조적으로, 기존의 기법들과 비교하여 ProofAug는 샘플 비용이 적은 상황에서도 значные 성능 향상을 보여주었습니다. 이를 통해 이 방법은 신경 정리 증명 연구의 발전에 크게 기여할 것으로 기대되고 있습니다.



### Update Estimation and Scheduling for Over-the-Air Federated Learning with Energy Harvesting Devices (https://arxiv.org/abs/2501.18298)
Comments:
          6 pages

- **What's New**: 이 논문에서는 무선 환경에서 에너지 수확 장치를 위한 공중 전송(OTA) 연합 학습(FL) 시스템을 연구합니다. 이러한 시스템은 이질적인 데이터 분포에 대해 사용자 스케줄링 전략을 제안하여 글로벌 학습의 성능 향상을 꾀합니다. 특히, 알려진 데이터 분포에 대해 엔트로피 기반 스케줄링과 예측 불가능한 데이터 분포에 대해 최소 제곱 기반 사용자 대표성 추정을 통해 다양한 사용자 선택을 목표로 합니다.

- **Technical Details**: 연합 학습 시스템은 여러 에너지 수확 장치로 구성되며, 모바일 사용자들은 채널 상태 정보(CSI) 없이 데이터를 전송합니다. PS(파라미터 서버)는 K개의 수신 안테나를 사용하여 수신 신호를 정렬하고, 모든 사용자의 업데이트를 동시에 전송함으로써 자원 할당의 효율성을 높입니다. 각 글로벌 반복에서, PS는 최신 글로벌 모델을 방출하고, 선택된 사용자들은 산출된 업데이트를 PS에 전송하게 됩니다.

- **Performance Highlights**: 수치 및 분석적인 결과는 제안된 사용자 선택 전략이 데이터 중복성을 줄이고 에너지를 절약함으로써 학습 성능을 향상시킨다는 것을 보여줍니다. 이러한 접근 방식은 시스템 내에서 중복 정보를 식별하고 그룹화함으로써 학습 성능을 개선하며, 특히 다양한 사용자 업데이트를 활용하는 점에서 혁신적입니다.



### Leveraging Sparsity for Sample-Efficient Preference Learning: A Theoretical Perspectiv (https://arxiv.org/abs/2501.18282)
- **What's New**: 이 논문은 인간의 선택을 비교 판단에 기반하여 모델링하고 예측하는 preference learning의 샘플 효율성을 다룹니다. 전통적인 추정 이론에서 요구되는 최소-최악 최적 추정률 $	heta(d/n)$는 샘플 수 $n$이 특징 공간의 차원 $d$와 선형적으로 스케일해야 함을 의미합니다. 하지만 높은 차원의 특징 공간과 인간 주석 데이터 수집의 높은 비용은 전통적인 추정 방법의 효율성을 저해합니다.

- **Technical Details**: 이를 해결하기 위해 선호 모델의 희소성(sparsity)을 활용하여 날카로운 추정 속도를 설정합니다. $k$-희소인 보상 함수의 매개변수를 갖는 희소 무작위 유틸리티 모델 하에서 최소-최악 최적 속도를 $	heta(k/n 	ext{log}(d/k))$로 줄일 수 있음을 보여줍니다. 또한, $	ext{l}_{1}$-정규화된 추정량에 대한 분석을 통해 Gram 행렬에 대한 완화된 가정 하에서 거의 최적의 속도에 도달함을 입증합니다.

- **Performance Highlights**: 합성 데이터와 LLM 정렬 데이터에 대한 실험을 통해 우리의 이론적 발견을 검증했습니다. 그 결과, 희소성을 고려한 방법들이 샘플 복잡성을 크게 줄이고 예측 정확성을 향상시킴을 보여줍니다.



### ReactEmbed: A Cross-Domain Framework for Protein-Molecule Representation Learning via Biochemical Reaction Networks (https://arxiv.org/abs/2501.18278)
- **What's New**: 본 논문에서는 ReactEmbed라는 새로운 방법론을 제시하여 생화학 반응 데이터를 사전 훈련된 임베딩(embeddings)과 통합하여 단백질 및 분자의 표현력을 강화하고 있습니다. 기존의 단일 모달 데이터는 생화학적 상호작용의 복잡성을 포착하는 데 한계가 있었지만, ReactEmbed는 이를 해결하기 위해 다중 모달 데이터를 효과적으로 활용합니다. 특히 단백질-나노입자 복합체의 혈액-뇌 장벽(BBB) 투과성을 예측하는 데 성공적인 결과를 보였습니다.

- **Technical Details**: ReactEmbed는 첫째로 생화학 반응을 가중치가 부여된 비방향 그래프로 변환하여 단백질과 분자 간의 관계를 포착합니다. 둘째로, 크로스 도메인 대조 학습(cross-domain contrastive learning) 프레임워크를 통해 사전 훈련된 단백질 및 분자 임베딩을 정렬하여 통합된 표현 공간을 만듭니다. 이 접근 방식은 특정 도메인에서 훈련된 모델이 다른 도메인에 대해 제로 샷 예측(zero-shot prediction)을 가능하게 하여 데이터 쌍이 필요하지 않습니다.

- **Performance Highlights**: ReactEmbed는 약물-표적 상호작용, 단백질-단백질 상호작용 등을 포함한 다양한 작업에서 기존의 최첨단 모델들을 일관되게 초월하는 성능을 보였습니다. 실험적으로, LNP(지질 나노입자) 기반 약물 전달 방식을 통해 단백질-나노입자 복합체의 BBB 투과성을 예측하는 데 성공적인 결과를 도출했습니다. 특히, 트랜스페린으로 장식된 뇌 표적 리포솜은 기존 방법에 비해 뇌 세포 내 mAb 농도를 7배 증가시켰으며, 신경퇴행성 질환 치료에 한 걸음 더 나아가는 긍정적인 영향을 미쳤습니다.



### Sebra: Debiasing Through Self-Guided Bias Ranking (https://arxiv.org/abs/2501.18277)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이번 연구에서는 인간의 감독 없이 데이터 포인트에서 spuriousity(불필요한 상관관계의 존재 정도)를 자동으로 분류하는 새로운 방법인 Self-Guided Bias Ranking(Sebra)을 제안합니다. Sebra는 ERM(경험적 위험 최소화) 훈련의 로컬 대칭성을 이용하여 bias를 완화하며, 이 과정을 통해 학습의 순서를 정하고 spuriousity 순위를 자연스럽게 제공합니다.

- **Technical Details**: Sebra 알고리즘은 학습 샘플에서 불필요한 특징의 양과 학습 난이도가 반비례 관계에 있다는 Hardness-Spuriosity Symmetry(난이도-불필요성 대칭)를 기반으로 합니다. 이 대칭성을 바탕으로 샘플들을 순차적으로 학습하여 spuriousity 순위를 도출하고, 이를 통해 bias를 완화하는 방법을 제시합니다. Sebra는 이러한 순위를 위한 별도의 공정성을 위해 ERM의 최적화를 조정합니다.

- **Performance Highlights**: Sebra는 UrbanCars, BAR, CelebA, ImageNet-1K와 같은 여러 데이터셋에서 기존의 비지도 학습 기반의 bias 완화 기술보다 뛰어난 성능을 보여줍니다. 특히 UrbanCars와 CelebA에서는 각각 평균 10%의 성능 향상을 달성하였으며, BAR에서는 6% 향상되었습니다. 이러한 결과는 Sebra가 제안하는 자동화된 spuriousity 분류 방식이 효과적임을 입증합니다.



### Pre-Trained Vision-Language Model Selection and Reuse for Downstream Tasks (https://arxiv.org/abs/2501.18271)
- **What's New**: 본 논문은 특정 downstream task에 적합한 최상의 VLM을 선택하는 문제를 다루며, Model Label Learning(MLL)이라는 새로운 패러다임을 제안합니다. 이 패러다임은 모델 레이블링, 모델 선택, 모델 재사용의 세 가지 주요 모듈로 구성되어 있습니다. MLL은 VLM을 더욱 효율적으로 사용할 수 있게 하여, 사용자가 다양한 task에 맞춤형 솔루션을 찾을 수 있도록 지원합니다.

- **Technical Details**: 제안된 MLL 방법론의 핵심은 후보 VLM을 모델 허브로 구성하여 각 VLM의 전문성과 유용성을 모델 레이블로 설명하는 것입니다. 모델 레이블링 과정에서는 세멘틱 그래프를 구성하여 각 모델을 테스트하고 레이블을 생성하며, 이를 통해 downstream tasks의 요구사항과 모델 레이블을 매칭하여 선택 과정을 수행합니다. 또, 앙상블 기법을 통해 선택된 모델들의 예측을 결합하여 최종 예측을 도출하는 독창적인 구조를 가지고 있습니다.

- **Performance Highlights**: MLL 방법은 49개의 VLM과 17개의 타겟 데이터셋을 포함하는 새로운 벤치마크를 도입하여 성능을 평가합니다. 실험 결과는 제안된 접근 방식이 VLM 선택 및 재사용에서 효과적임을 명확하게 입증하며, 모델 허브의 확장성도 검증되었습니다. 본 연구는 VLM의 실제적인 적용을 촉진할 수 있는 기초를 제공합니다.



### Reducing Aleatoric and Epistemic Uncertainty through Multi-modal Data Acquisition (https://arxiv.org/abs/2501.18268)
- **What's New**: 이 논문은 다중 모달리티(multi-modality) 데이터에서 불확실성을 분리하는 혁신적인 데이터 수집 프레임워크를 제안합니다. 기존의 인식 불확실성(epistemic uncertainty)과 우연적 불확실성(aleatoric uncertainty) 개념은 다양한 모달리티에서 수집된 정보에 따라 도전받고 있습니다. 본 연구는 샘플 크기와 데이터 모달리티를 동시에 고려하여 의사결정을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 액티브 러닝(active learning) 및 액티브 피처 인퀴지션(active feature acquisition)의 개념을 결합하여 두 가지 종류의 불확실성을 낮추는 두 단계의 백트래킹 프로토콜을 구현합니다. 첫 번째 단계에서 새로운 테스트 인스턴스에 대한 인식 불확실성을 평가하고, 두 번째 단계에서 높이가 우연적 불확실성을 체크하여 추가 모달리티를 샘플링합니다. 이러한 접근법은 BIOSCAN-5 데이터셋과 같은 다중 모달 데이터셋에서 검증됩니다.

- **Performance Highlights**: 논문에서는 제안된 프레임워크의 개념 검증을 위해 두 가지 다중 모달 데이터셋에서 실험을 수행하였습니다. 실험 결과는 불확실성 분리의 새로운 통찰을 제공하며 인식 불확실성 측정의 적절한 속성을 강조합니다. 최종적으로 이 프레임워크는 과학 공동체에 공개되었으며, 사전 학습된 모델과의 호환성이 보장되어 쉽게 통합할 수 있습니다.



### PDE-DKL: PDE-constrained deep kernel learning in high dimensionality (https://arxiv.org/abs/2501.18258)
Comments:
          22 pages, 9 figures

- **What's New**: 이 논문은 PDE(부분 미분 방정식) 문제 해결을 위한 새로운 프레임워크인 PDE-제약 딥 커널 학습(PDE-DKL)을 제안합니다. PDE-DKL은 깊은 신경망(DNN)과 Gaussian 프로세스(GP)를 결합하여 높은 차원의 문제를 효과적으로 해결합니다. 이 방법은 불확실성 정량화를 포함하여 적은 데이터를 요구하면서도 높은 정확도를 제공합니다.

- **Technical Details**: PDE-DKL 프레임워크는 신경망이 고차원 PDE 문제의 저차원 잠재 표현을 학습하게 하여 문제의 복잡성을 줄입니다. 이후 Gaussian 프로세스는 주어진 PDE 제약 조건에 따라 커널 회귀를 수행하며, 데이터가 제한적일 때도 정확한 솔루션을 도출할 수 있습니다. 이 접근 방식은 DNN과 GP의 장점을 통합하여 고차원 PDE에 대한 강력한 불확실성 추정과 계산 효율성을 제공하는 것이 특징입니다.

- **Performance Highlights**: numerical experiments에 따르면, PDE-DKL은 적은 데이터 요구 조건으로 높은 정확도를 달성했습니다. 이 방법은 과학과 공학 분야의 복잡한 PDE 기반 응용 프로그램을 위한 실용적이고 신뢰할 수 있으며 확장 가능한 솔버로서의 가능성을 보여줍니다. 특히, 높은 차원의 문제에서의 성능은 기존 방법들에 비해 월등한 효과를 나타냅니다.



### Exploring Large Protein Language Models in Constrained Evaluation Scenarios within the FLIP Benchmark (https://arxiv.org/abs/2501.18223)
- **What's New**: 이 연구에서는 FLIP 벤치를 확장하여, 최신 대형 단백질 언어 모델인 ESM-2와 SaProt의 성능을 평가합니다. FLIP은 제한된 데이터 가용성을 가진 상황에서 단백질 피트니스 예측 모델의 성능을 분석하는 데 중점을 둡니다. 이러한 환경에서 최근의 단백질 언어 모델의 발전이 성능 향상에 기여하는지에 대한 연구 결과는 특히 유용합니다.

- **Technical Details**: FLIP 벤치는 데이터 분할 전략인 'two vs many'와 'low vs high'를 사용하여 제한된 데이터 환경에서 단백질 피트니스 예측을 위한 다양한 활동들을 포함합니다. ESM-2와 SaProt 같은 최신 모델들은 자가 지도 Pretraining(사전 훈련)을 통해 대량의 레이블 없는 단백질 시퀀스 데이터를 활용하고, 이로 인해 모델이 낮은 데이터 조건에서도 잘 일반화될 수 있도록 합니다. SaProt는 구조 정보(Structural Information)를 활용하여 단백질 피트니스를 예측하는 방식으로, 모델링 과정에서 ESMFold를 통해 예측된 단백질 구조를 통합합니다.

- **Performance Highlights**: 연구 결과는 ESM-2가 모델 크기에 따라 학습 성능과 일반화 능력이 어떻게 달라지는지를 분석합니다. SaProt는 구조의 품질을 기반으로 한 예측 효과를 조명하며, 모델 간의 공정한 평가를 위해 학습, 검증, 테스트 세트의 구분을 철저히 유지합니다. 한마디로, 이 연구는 제한된 데이터 환경에서 단백질 피트니스 예측의 정확성과 모델 전반에 걸쳐 성능을 분석할 수 있는 기대감을 제공합니다.



### HKAN: Hierarchical Kolmogorov-Arnold Network without Backpropagation (https://arxiv.org/abs/2501.18199)
Comments:
          13 pages, 9 figures

- **What's New**: 이번 논문에서는 HKAN(Hierarchical Kolmogorov-Arnold Network)라는 새로운 네트워크 아키텍처를 제안합니다. 기존의 KAN(Kolmogorov-Arnold Network)와 달리, HKAN은 고정된 파라미터를 가진 무작위 학습 방식으로 학습하며, 손실 함수의 지역 최소값에 대한 민감성을 없애는 비반복적인 훈련 방법을 사용합니다. 이 방법은 기본 함수의 매개변수를 고정하고, 적은 수의 파라미터로도 비슷한 또는 더 나은 성능을 발휘하게 합니다.

- **Technical Details**: HKAN은 계층적 다중 스태킹( hierarchical multi-stacking ) 프레임워크를 활용하여 이전 층의 예측을 수정합니다. 각 층은 선형 회귀 문제를 해결함으로써 예측을 정제하며, 이를 통해 내부 연산이 단순화됩니다. 이러한 구조는 표준 최소제곱 회귀( least-squares regression ) 방법을 통해 빠른 계산이 가능하게 하며, 비선형 모델링에서도 높은 성능을 유지합니다.

- **Performance Highlights**: 경험적 결과에 따르면, HKAN은 다양한 회귀 작업에서 KAN에 비해 비교 가능한 또는 우수한 정확도와 안정성을 보여주고 있습니다. 또한 변수 중요성을 시각화하는 데 필요한 통찰력을 제공합니다. 이 연구는 이론적 통찰과 실용적 응용을 통합하여 신경망 모델링에 대한 강력하고 효율적인 대안을 제시합니다.



### Fundamental Challenges in Evaluating Text2SQL Solutions and Detecting Their Limitations (https://arxiv.org/abs/2501.18197)
- **What's New**: 이 연구에서는 Text2SQL 솔루션을 평가하는 데 있어 근본적인 도전 과제를 다루고 있으며, 기존 벤치마크에서의 집계 지표에 의존하는 것의 잠재적인 실패 원인과 위험을 강조합니다. 저자들은 두 가지 주목할 만한 한계를 식별했습니다: 첫째, 평가 데이터의 품질 문제로 인해 자연어 설명을 구조화된 질의로 번역하는 확률론적 성질이 제대로 반영되지 못하는 점이며, 둘째, SQL 동등성을 근사하기 위해 사용되는 다양한 매치 함수가 편향을 유발하는 점입니다.

- **Technical Details**: Text2SQL(또는 NL2SQL) 작업은 자연어(NL) 설명이나 질문을 데이터베이스(DB)의 스키마와 함께 구조화된 쿼리 언어(SQL) 쿼리로 변환하는 과정으로 정의됩니다. 이 논문에서는 입력 준비와 결과 추출 및 검증 단계를 포함하여 높은 정확도의 Text2SQL 솔루션을 저해하는 모든 한계를 조망하며, 모델 한계를 넘어서는 다양한 도전 과제를 또한 실질적으로 다룹니다.

- **Performance Highlights**: 대중적인 벤치마크에서는 LLM(대형 언어 모델)의 향상된 자연어 이해 능력 덕택에 Text2SQL 모델이 점점 성공적인 성과를 보이고 있으나, 실제 적용 사례는 여전히 부족합니다. 따라서, 공개적으로 수집된 벤치마크와 최첨단 Text2SQL 솔루션들을 통해 확인된 여러 한계 및 이들이 초래하는 예측 및 평가 오류를 새롭게 정의한 분류법을 통해 연구하고, 실제 사례에 기반한 문제 해결 방안을 제시하는 것을 목표로 합니다.



### GDformer: Going Beyond Subsequence Isolation for Multivariate Time Series Anomaly Detection (https://arxiv.org/abs/2501.18196)
- **What's New**: 본 논문에서는 전체 시퀀스에 공유되는 정상 포인트들의 글로벌 표현(global representations)을 학습하기 위해 글로벌 사전(dictionary)-향상 Transformer(GDformer)를 제안합니다. 기존의 방법이 격리된 하위 시퀀스에서 제한된 시계열 수준의 기준을 제시하는 것과는 달리, GDformer는 모든 정상 포인트 간의 상관관계를 반영하여 더 포괄적인 탐지 기준을 제공합니다. 이를 통해 기계 학습 모델의 성능을 향상시키고 anomaly detection의 경계를 더욱 명확히 할 수 있습니다.

- **Technical Details**: GDformer는 Key 및 Value 벡터의 글로벌 사전을 활용하여 정상 포인트 간의 글로벌 표현을 학습하는 방법을 제안합니다. 이 접근 방식은 Transformer의 self-attention 메커니즘을 개선해 상관관계 가중치를 평가하고, 아울러 프로토타입(prototypes)을 도입하여 정상-비정상 포인트 간의 유사성 차이를 효율적으로 캡처합니다. 이러한 방법으로 인해, 기존의 점별(point-wise) 접근 방식에서 벗어나 전체 시퀀스를 기반으로 한 유사성 기반 탐지 기준을 수립할 수 있습니다.

- **Performance Highlights**: GDformer는 5개의 실제 벤치마크 데이터셋에서 최신 성능을 지속적으로 달성하며, 이를 통해 그 효과성을 입증했습니다. 실험 결과, 글로벌 사전이 다양한 데이터셋 간에 높은 전이 성능(transferability)을 보임을 확인하였습니다. 이러한 결과는 GDformer가 다양한 상황에서도 우수한 anomaly detection 기능을 제공할 수 있음을 시사합니다.



### Neural Network Modeling of Microstructure Complexity Using Digital Libraries (https://arxiv.org/abs/2501.18189)
- **What's New**: 이 연구는 마이크로구조의 진화 과정에서 인공신경망(ANN)과 스파이킹 신경망(SNN)의 성능을 비교 평가하고 있습니다. 특히, SNN는 적은 파라미터와 메모리 사용량으로 보다 우수한 예측 정확성을 나타낸다는 점이 주목됩니다. 본 연구는 이 디지털 라이브러리 접근 방식을 통해 실제 실험 데이터의 제한을 극복하고자 하였습니다.

- **Technical Details**: 마이크로구조의 진화는 다양한 맥락에서 다르게 나타나며, Fatigue Crack Growth(FCG)와 Turing pattern의 개발이 이 연구의 주요 초점입니다. 이를 위해 연구팀은 적절한 데이터 샘플을 수집하기 위해 extended finite element method(XFEM)를 이용하여 경량화된 경량의 FCG 디지털 라이브러리를 구축하였습니다. 각 경계 조건(Boundary Condition) 아래에서의 물리적 시스템을 모델링하기 위해 partial differential equations(PDEs)가 사용됩니다.

- **Performance Highlights**: 신경망 모델의 테스트 결과, SNN이 전통적인 ANN 모델보다 파라미터 수와 메모리 사용량에서 우수한 성능을 보였습니다. 특히 SNN은 복잡한 행동을 처리하는 데 더 뛰어난 적응성을 가질 수 있어, 마이크로구조의 진화와 같은 동적인 시스템에도 효과적으로 적용할 수 있습니다. 이를 통해 연구팀은 기존의 컴퓨터 비전 작업에서의 정확성-비용 간의 균형을 넘는 성과를 거두었습니다.



### In-Context Learning of Polynomial Kernel Regression in Transformers with GLU Layers (https://arxiv.org/abs/2501.18187)
- **What's New**: 이번 연구에서는 Transformer 모델의 in-context learning (ICL)에 대한 이론적 이해를 확장합니다. 특히, 기존의 linear self-attention (LSA) 메커니즘이 비선형 함수 클래스에서의 ICL에 적합하지 않음을 보여줍니다. 또한, GLU와 유사한 feed-forward 레이어를 결합하여 비선형 회귀 작업을 위한 경량화를 제안합니다.

- **Technical Details**: 저자들은 LSA가 선형 최소제곱 문제를 해결하기 위해 본질적으로 제한적이라는 점을 지적합니다. GLU-like feed-forward 레이어와 LSA를 결합한 Transformer 구조는 다항회귀에 대한 경량화된 기울기 하강법을 구현할 수 있음을 설명합니다. 그러나 서로 다른 ICL 작업에 대한 요구 사항은 embedding 차원이 입력 차원의 제곱에 비례해야 함을 강조합니다.

- **Performance Highlights**: 연구 결과는 수치 실험을 통해 validation되었으며, 특히 깊은 linear Transformer가 고차 다항식 목표 함수를 효과적으로 학습할 수 있는 사례를 제시합니다. 이러한 연구는 비선형 ICL의 도전 과제를 밝혀내고, 더 긴 프롬프트가 이러한 문제를 극복하는 데 필요할 수 있음을 시사합니다.



### Genetic Algorithm with Border Trades (GAB) (https://arxiv.org/abs/2501.18184)
- **What's New**: 이 논문은 경계 거래 활동(border trade activities)을 배양 과정에 통합하여 큰 문제 공간에서 유전 알고리즘(Genetic Algorithms, GA)의 성능을 향상시키는 새로운 접근법을 제시합니다. 새로운 염색체 패턴을 도입함으로써 염색체 다양성을 증대시키고 조기 수렴을 방지하며 검색 공간을 더 효과적으로 탐색할 수 있도록 합니다. 실험적 증거는 수렴 행동의 상당한 개선을 입증합니다.

- **Technical Details**: 저자는 두 가지 문제, 즉 플립플롭(Flip-Flop) 문제와 일자리 스케줄링(Job Scheduling) 문제에서 경계 거래 개념의 효과를 조사합니다. 플립플롭 문제에서는 상위 적합성을 달성하는 데 문제가 발생할 수 있으며, 두 부모 염색체가 너무 유사할 경우 Crossover와 Mutation 작업이 동일한 후손을 생성하기 때문에 해결책이 정체될 수 있습니다. 따라서, 부모 염색체의 비트를 뒤집는 경계 거래 메커니즘을 도입하여 다양성과 탐색을 증대시킵니다.

- **Performance Highlights**: 저자들은 경계 거래가 다음세대에서 75% 최적 해결책을 생성할 확률을 제공하는 반면, 일시적인 적합성 감소의 가능성은 25%라는 점을 강조합니다. 일자리 스케줄링 문제에서는 GA의 효율성이 보여지며, 브루트 포스(brute force) 방법이 108 작업의 경우에 비효율적인 반면, GA는 더 빠른 시간 내에 더 나은 결과를 생성합니다. 경계 거래 전략이 전반적으로 성능 개선에 기여하고 있다는 실험 결과가 제시됩니다.



### Advancing Personalized Federated Learning: Integrative Approaches with AI for Enhanced Privacy and Customization (https://arxiv.org/abs/2501.18174)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2501.16758

- **What's New**: 이번 논문은 Personalized Federated Learning (PFL)에 최신 AI 방법론인 adaptive optimization, transfer learning, 및 differential privacy를 통합한 새로운 접근법을 제안합니다. 이 모델은 개별 클라이언트 모델의 성능을 향상시키고, 다양한 네트워크에서 자원 활용을 효율적으로 관리하면서 강력한 개인 정보 보호 메커니즘을 보장합니다. 또한, 기존의 federated learning 모델과 비교하여 모델 정확도와 개인화에서 유의미한 개선을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 로컬 모델 훈련, 매개변수 집합, 개인화 및 제어 이론 원칙에 기반한 동적 학습 속도 조정과 같은 여러 중요한 요소로 구성됩니다. 이 시스템 아키텍처는 IoT 기기들이 분산된 네트워크를 통해 작동하며, 차량 수, 속도 및 흐름 방향과 같은 물 데이터를 수집합니다. Meta-Learning을 통해 모델의 적응성을 향상시키는 방식으로 기여합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 기존의 federated learning 모델에 비해 모델 정확도와 개인화 및 개인 정보 보호 측면에서 현저한 개선을 보여줍니다. 이는 기업들이 엄격한 데이터 보호 규정을 준수해야 하는 상황에서 개인정보 보호를 고려한 AI 시스템의 새로운 시대를 여는 중요한 기회를 제공합니다. 논문은 다양한 환경에서 PFL 프레임워크의 배포를 위한 포괄적인 지침도 제공합니다.



### Continually Evolved Multimodal Foundation Models for Cancer Prognosis (https://arxiv.org/abs/2501.18170)
Comments:
          8 pages, 1 figure

- **What's New**: 이 연구에서는 암 예후 예측을 위한 새로운 접근 방식으로 지속적으로 발전하는 다중 모달 기초 모델을 제안합니다. 기존 모델의 한계를 극복하기 위해 두 가지 주요 구성 요소인 Pseudo Target Generation Module (PTGM)과 Instruction-based Knowledge Distillation (IKD)을 통합하여 고안되었습니다. 이 모델은 다양한 출처의 데이터를 효과적으로 통합할 수 있도록 설계되어, 임상 환경에서의 일반화 가능성과 유용성을 극대화합니다.

- **Technical Details**: 제안된 모델의 PTGM 모듈은 여러 작업에서 배우는 동안 발생하는 재앙적 망각 문제를 완화하며, IKD 모듈은 새로운 모달리티에 적응할 때 모델의 생성 능력을 보존합니다. TCGA 데이터 세트를 사용한 광범위한 실험을 통해 제안된 접근 방식의 유효성이 검증되었습니다. 이 연구는 AI의 다중 모달 통합이 암 예후를 개선하는 데 중요한 역할을 할 수 있음을 강조합니다.

- **Performance Highlights**: 실험 결과는 지속적으로 발전하는 다중 모달 기초 모델이 암 예후 예측의 실제 환경에서 큰 향상을 이루는데 기여할 수 있음을 보여줍니다. 이러한 접근 방식은 제공되는 다양한 모달리티의 상호 의존성을 포착하여 예측 정확도를 높이는 데 매우 효과적입니다. 따라서 이 연구는 암 치료와 관련된 임상 의사결정에 있어 중요한 잠재력을 지니고 있습니다.



### Faster Convergence of Riemannian Stochastic Gradient Descent with Increasing Batch Siz (https://arxiv.org/abs/2501.18164)
- **What's New**: 이번 연구에서는 Riemannian Stochastic Gradient Descent (RSGD)를 통해 배치 크기(batch size)가 증가할 경우 더 빠른 수렴(convergence)이 이루어진다는 것을 이론적으로 분석했습니다. 이를 통해 RSGD는 상수 학습률(learning rate) 또는 감쇠하는 학습률을 사용할 때 보다 빠른 수렴 속도를 보이는 것으로 밝혀졌습니다. 특히, RSGD는 전통적인 방법보다 더 나은 수렴 속도를 보이며, 이는 대규모 모델 훈련에 있어 효율성을 높일 수 있는 가능성을 제시합니다.

- **Technical Details**: RSGD는 유클리드 공간에서의 SGD에 대응하는 Riemannian manifold에서 작동하는 알고리즘으로, 많은 머신 러닝 작업에서 사용되고 있습니다. 본 논문에서는 RSGD의 성능을 높이기 위해 증가하는 배치 크기와 다양한 감쇠 학습률 유형을 도입했습니다. 연구 결과, 다수의 기존 이론적 연구와는 달리, 상수 배치 크기를 사용하는 것보다 더 나은 성과를 기대할 수 있음이 입증되었습니다.

- **Performance Highlights**: 주요 실험은 주성분 분석(principal component analysis) 및 저계수 행렬 완전화(low-rank matrix completion) 문제를 대상으로 진행되었으며, 대부분의 경우 증가하는 배치 크기가 성능을 향상시키는 것으로 나타났습니다. 예외적으로 MovieLens 데이터셋과 상수 학습률의 경우는 제외되었으나, 일반적으로 폴리노미얼 성장 배치 또는 지수 성장 배치를 사용할 경우 상수 배치 크기보다 더 나은 성과를 얻을 수 있었습니다.



### Dual-Bounded Nonlinear Optimal Transport for Size Constrained Min Cut Clustering (https://arxiv.org/abs/2501.18143)
- **What's New**: 이번 논문에서는 min cut 문제를 기존의 비효율성과 간단한 해에 대한 수렴 문제를 해결하기 위해 새로운 접근 방식으로 제안합니다. 특히, 이 논문은 min cut 문제를 dual-bounded constraint로 완화하고, 이를 비선형 optimal transport 문제로 취급하는 첫 번째 연구입니다. 새로운 DNF(dual-bounded nonlinear Frank-Wolfe) 방법은 이러한 문제를 해결하는데 효율적인 접근법으로, 파라미터 없이도 안정적으로 동작합니다.

- **Technical Details**: DNF 방법은 기존 Frank-Wolfe 알고리즘을 기반으로 하여, dual-bounded constraints에 맞춰 비군집성 문제를 해결합니다. DNF 방법은 비선형 목적 함수를 최소화하는 과정에서 제약 조건을 준수하며, 각 반복에서 비례적 gradient를 잘 근사하여 비교적 계산적으로 효율적으로 동작합니다. 또한, convex 문제의 경우 O(1/t)로 수렴 속도를 보이며, Lipschitz 연속 비볼록 함수에서도 우수한 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과 DNF 방법은 이미지, 텍스트 및 그래프 기반 데이터에서 기존 방법보다 우수한 군집 품질을 보였습니다. 또한, 제안된 방법은 크기 제약을 만족하는 min cut 문제에 대해 정밀한 클러스터링 결과를 제공하며, 빠른 속도로 수렴하는 성능을 유지합니다. 이 결과들은 DNF 방법의 실용성과 다양한 분야에서의 적용 가능성을 강조합니다.



### B3C: A Minimalist Approach to Offline Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2501.18138)
- **What's New**: 이 논문에서는 오프라인 강화 학습(offline reinforcement learning)에서 정책 평가 중 보이지 않는 행동 선택으로 발생하는 과대평가(overestimation) 문제를 해결하기 위한 새로운 방법인 B3C(Behavior Cloning regularization with Critic Clipping)를 제안합니다. 이 방법은 정책 평가 시 비율을 클리핑(clipping)하여 과대평가를 완화하고, BC(Behavior Cloning) 정규화를 통해 RL(objective) 목표의 강도를 높입니다. B3C는 오프라인 다중 에이전트 환경에서 기존 최첨단 알고리즘보다 우수한 성능을 보여줍니다.

- **Technical Details**: B3C는 데이터셋의 최대 수익을 기준으로 목표 비평가 가치(critic value)를 클리핑하여 정책 평가의 과대평가를 완화합니다. 이 방법은 RL 목표에 대한 가중치를 높여 성능을 향상시키며, 비선형(value factorization) 분해 기법을 통합하여 성능 개선을 도모합니다. 논문에서는 오프라인 설정에서 비선형 분해 기법이 단조(monotonic) 및 선형(linear) 분해 방식보다 더 효과적임을 실증적으로 보여줍니다.

- **Performance Highlights**: B3C는 다차원 Mujoco 및 입자(particle) 환경을 포함한 다양한 오프라인 다중 에이전트 RL 벤치마크에서 성능이 기준선(baseline)보다 우수하다는 것을 보여줍니다. 이 연구는 다중 에이전트 환경에서의 기존 방법의 한계를 극복하는데 필요한 간결한 해결책을 제공하고 있습니다. B3C의 간단함에도 불구하고, 다양한 데이터셋에서 empirically(경험적으로) 성능이 뛰어난 것으로 나타났습니다.



### Tensor Completion for Surrogate Modeling of Material Property Prediction (https://arxiv.org/abs/2501.18137)
Comments:
          2 page paper accepted to AAAI KGML 2025 bridge program

- **What's New**: 본 논문에서는 특정 재료 속성을 최적화하기 위한 설계 방법을 개선하기 위해 머신러닝(ML)을 활용하는 새로운 접근 방식을 제안합니다. 특히, 재료 속성 예측 작업을 텐서 완성(tensor completion) 문제로 모델링하여 대량의 재료 조합을 효과적으로 탐색할 수 있는 방법을 소개합니다. 이 방법론을 통해 블라인드 ML 모델보다 10-20% 낮은 오류율을 기록하며, 비슷한 훈련 속도를 유지할 수 있음을 실험적으로 입증하였습니다.

- **Technical Details**: 연구에서는 재료 데이터셋을 생성하기 위해 각 텐서 모드를 고유한 원소나 그 원소의 비율로 설정하고, 텐서 완성을 통해 드문 가치들에서 재료 속성을 추론하는 방법을 사용합니다. CPD(Canonical Polyadic Decomposition) 및 기타 텐서 완성 모델을 통해 주요 정보를 추출하고, 평균 절대 오차(MAE)를 계산하여 연구 결과를 분석합니다. 이러한 접근 방식은 화학식 기반 재료 속성 예측을 가능하게 하며, 데이터셋의 구조를 활용하여 빠르고 정확한 예측을 지원합니다.

- **Performance Highlights**: 실험 결과, 텐서 완성 모델은 다양한 예측 작업에서 평균 절대 오차(MAE) 기준으로 비 텐서 모델들보다 정기적으로 더 우수한 성능을 보였습니다. 특히, 총 자화 예측, 형성 에너지 및 밴드 갭 예측에서 두드러지는 성과를 보였습니다. 텐서 완성을 통해 고속으로 예측할 수 있으며, 기존 값의 작은 비율만으로도 관찰되지 않은 재료 값을 유추할 수 있다는 점에서, 재료 설계 문제를 현저히 가속화할 수 있는 가능성을 보여줍니다.



### Battery State of Health Estimation Using LLM Framework (https://arxiv.org/abs/2501.18123)
Comments:
          Accepted at The 26th International Symposium on Quality Electronic Design (ISQED'25)

- **What's New**: 본 연구는 전통적인 배터리 건강 모니터링 방법을 개선하기 위해 트랜스포머 기반의 LLM(대규모 언어 모델) 프레임워크를 도입했습니다. 이 프레임워크는 리튬 타이타네이트(LTO) 배터리 셀의 State of Health(SoH)와 Remaining Useful Life(RUL)를 예측하며, 주기 기반 및 순간 방전 데이터를 모두 활용합니다. 이 연구는 배터리 성능과 에너지 저장 트렌드를 분석하고, DVA(차등 전압 분석)를 통해 용량 변화를 모니터링하는 혁신적인 방법론을 제시합니다.

- **Technical Details**: 본 논문에서는 LTO 배터리 셀의 SoH와 RUL을 예측하기 위해 LLM 프레임워크를 개발했습니다. DVA는 배터리의 성능 변화를 세밀하게 감지하는 데 사용되며, 전압 변화당 배터리의 차등 용량(dQ/dV)을 평가하여 내부 상태 및 용량 열화를 이해하는 데 도움을 줍니다. 향상된 예측 정밀도를 위해 이 프레임워크는 트랜스포머 모델링을 활용하여 주기 기반 및 순간 방전 데이터의 복합 패턴을 추출합니다.

- **Performance Highlights**: 모델의 성과는 MAE(평균 절대 오차)가 0.87%라는 낮은 값으로 나타나며, EV(전기차) 예측 유지보수에 실시간 통합 가능성을 보여줍니다. 본 연구는 배터리의 초기 열화 신호를 감지할 수 있는 기능을 갖추어, 배터리의 갑작스러운 고장을 방지하고 에너지 효율을 향상시키는 데 기여할 수 있습니다. 이로 인해 향후 스마트 운송 시스템의 발전에 중요한 기여를 할 것으로 기대됩니다.



### VQLTI: Long-Term Tropical Cyclone Intensity Forecasting with Physical Constraints (https://arxiv.org/abs/2501.18122)
- **What's New**: 이 논문에서는 열대 사이클론(Tropical Cyclone, TC) 강도 예보를 개선하기 위해 VQLTI라는 새로운 프레임워크를 제안합니다. VQLTI는 TC 강도 정보를 이산적인 잠재 공간(discrete latent space)으로 전달하면서 공간 정보 간의 차이를 유지하고, 이를 위해 대규모 기상 데이터를 사용합니다. 또한, FengWu 기상 예측 모델의 예측 결과를 통해 물리적 지식을 통합하여 예측 성능을 더욱 향상시킵니다.

- **Technical Details**: VQLTI는 알려진 과거의 TC 강도 및 ERA5 데이터를 입력으로 받아 미래의 TC 강도를 예측합니다. 이 모델은 Conditional Vector Quantized-Variational AutoEncoder (CVQ-VAE) 구조를 사용하여 강도 정보를 이산 잠재 공간으로 매핑하며, 그러한 잠재 변수에 물리적 제약(physical constraints)을 적용하여 예측의 정확도를 높입니다. 예측에 사용된 ERA5 데이터는 기존 대기 상태의 적합 데이터로, 여러 기상 변수들을 포함하고 있습니다.

- **Performance Highlights**: VQLTI 모델은 24시간에서 120시간의 장기 TC 강도 예보에서 뛰어난 성능을 나타내며, ECMWF-IFS와 비교했을 때 최대 지속풍 속도(Maximum Sustained Wind, MSW) 예측 오차를 35.65%에서 42.51%까지 줄였습니다. 이는 기존 딥러닝 기반 방법들이 가지던 장기 예측의 어려움을 크게 개선한 결과입니다. VQLTI가 보여준 최첨단 성능은 TC 강도 예측 분야에서의 중요한 진전을 의미합니다.



### ACTGNN: Assessment of Clustering Tendency with Synthetically-Trained Graph Neural Networks (https://arxiv.org/abs/2501.18112)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문에서는 데이터의 클러스터링 경향을 평가하기 위해 고안된 그래프 기반 프레임워크인 ACTGNN을 제안합니다. 이 방법은 Locality-Sensitive Hashing (LSH)를 사용하여 노드 특성을 구성하고, 여러 유사성 메트릭스를 활용하여 에지 특성을 포함합니다. 이러한 구조적 접근 방식은 기존의 전통적인 방법들이 가지는 한계를 극복할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: ACTGNN은 고차원 데이터의 클러스터링 구조 식별을 위해 Graph Neural Network (GNN)를 사용합니다. 훈련 과정에서는 합성 데이터셋만을 사용하며, 이는 클러스터링 구조와 소음을 제어하는 데 유리합니다. 노드 특성은 LSH를 활용하여 주변 이웃 정보를 포착하고, 에지 특성은 Radial Basis Function (RBF) 커널과 같은 다양한 유사성 메트릭스를 통합하여 쌍 간의 관계를 모델링합니다.

- **Performance Highlights**: ACTGNN은 기초 방법들에 비해 합성 및 실제 데이터셋 모두에서 뛰어난 성능을 나타냅니다. 특히, 노이즈가 많고 고차원인 데이터에서도 미세한 클러스터링 구조를 탐지하는 데 우수한 능력을 보입니다. 실험 결과는 제안된 접근 방식의 일반화 가능성과 효과성을 강조하여 클러스터링 경향 평가를 위한 유망한 도구임을 입증합니다.



### Scaling Inference-Efficient Language Models (https://arxiv.org/abs/2501.18107)
Comments:
          17 pages, 16 figures

- **What's New**: 이 연구는 대형 언어 모델의 성능 예측에 있어 scaling laws가 인퍼런스 비용을 충분히 반영하지 못한다는 점을 지적합니다. 또한, 동일한 크기의 모델인데도 아키텍처에 따라 인퍼런스 지연(latency)이 3.5배 차이가 날 수 있음을 보여주며, 인퍼런스 효율성을 고려한 새로운 scaling laws를 제안합니다. 이와 함께 Morph-1B 모델을 발표하며, 기존 오픈 소스 모델 대비 1.8배 더 빠른 인퍼런스 지연을 달성했음을 강조합니다.

- **Technical Details**: 기존의 scaling laws는 모델 크기(모델 파라미터 수)와 훈련 토큰 수의 균형을 중시합니다. 본 연구에서 제안하는 새로운 scaling laws는 모델 아키텍처도 반영하며, 인퍼런스 효율성을 최적화합니다. 다양한 모델 파라미터와 재훈련 과정을 통해, 63개의 실험 모델을 개발하고 이는 인퍼런스 손실 예측력에서 Chinchilla scaling law에 비해 더 우수함을 입증합니다.

- **Performance Highlights**: Morph-1B 모델은 본 연구의 인퍼런스 효율성 기반 scaling laws 및 모델 선택 방법을 통해 개발되었으며, 같은 크기의 다른 오픈 소스 모델에 비해 1.8배 빠른 인퍼런스 지연을 보입니다. 이는 모델의 정확도(downstream task 성능)를 유지하면서도 인퍼런스 효율성을 극대화한 결과입니다. 연구 결과는 향후 아키텍처 최적화에 있어 효율성과 정확도 간의 균형을 잘 잡을 수 있는 기준이 될 것입니다.



### AlphaAdam:Asynchronous Masked Optimization with Dynamic Alpha for Selective Updates (https://arxiv.org/abs/2501.18094)
- **What's New**: 새로운 연구에서는 대형 언어 모델(LLM) 훈련 중에 파라미터 업데이트 효율성과 안정성을 높이는 데 중점을 둡니다. AlphaAdam이라는 최적화 프레임워크를 소개하며 이는 레이어 내(intra-layer) 파라미터 업데이트 관점에서 접근하여 훈련의 수렴 속도를 가속화하고 안정성을 향상시킵니다. 이 방법은 최근 진화한 선택적 파라미터 업데이트 전략을 토대로 하며 큰 차별점을 가지고 있습니다.

- **Technical Details**: AlphaAdam은 파라미터 업데이트를 역전파 과정에서 마스크를 적용하고 이를 동적으로 조정하여 효율적인 최적화를 보장합니다. 과거 모멘텀과 그래디언트 방향의 일관성을 기반으로 만들어진 파라미터 마스크를 활용하며, 이러한 마스크 강도 조정 전략을 통해 이론적 수렴 보장을 추가합니다. 이 새로운 접근 방식은 기존의 많은 모멘텀 기반 최적화 기법에도 적용 가능합니다.

- **Performance Highlights**: 실험 결과, AlphaAdam은 GPT-2 및 RoBERTa, Llama-7B와 같은 다양한 작업에서 AdamW와 같은 최신 기법들보다 수렴 속도와 계산 효율성에서 두각을 나타냅니다. 우리의 방법은 훈련 안정성을 현저히 개선하여 다양한 모델 규모와 작업에서 일반적인 파라미터 업데이트에 비해 동등하거나 우수한 성능을 나타냅니다. 이러한 성과들은 AlphaAdam이 LLM을 위한 현장 최적화 기법으로서 유망함을 지니고 있음을 시사합니다.



### Reward Prediction Error Prioritisation in Experience Replay: The RPE-PER Method (https://arxiv.org/abs/2501.18093)
Comments:
          This paper was accepted for presentation at the 2024 Australasian Conference on Robotics and Automation (ACRA 2024). It consists of 10 pages, including four figures and two tables

- **What's New**: 이 논문에서는 기존의 경험 재생 버퍼(replay buffer)에서 더 나은 경험을 선정하기 위해 '보상 예측 오차 우선 경험 재생'(RPE-PER)이라는 새로운 접근 방식을 소개합니다. RPE-PER는 생물학적 시스템에서의 보상 예측 오차(reward prediction errors, RPE)를 기반으로 경험을 우선 정렬하는 방식으로, 에이전트의 학습 방향성을 인도합니다. 이런 방식을 통해 RPE-PER는 지속적인 제어 작업에서 학습 속도와 성능을 향상시킵니다.

- **Technical Details**: RPE-PER는 '강화 모델 비평가 네트워크'(EMCN)라는 비평가 네트워크를 사용하여 보상을 예측합니다. EMCN은 전통적 비평가 네트워크와 달리 Q-값 뿐만 아니라 다음 상태와 보상도 예측하여 환경을 효율적으로 모델링합니다. 여기서 RPE는 예측된 보상과 실제 보상 간의 차이로 정의되며, 이를 기반으로 경험이 우선 정렬되어 훈련이 이루어집니다.

- **Performance Highlights**: 다양한 MuJoCo 연속 제어 벤치마크에서 RPE-PER의 성능을 평가한 결과, 여러 최첨단 우선순위 버퍼 방법들과 비교했을 때 학습 효율성과 성과에서 주목할 만한 향상을 나타냈습니다. RPE-PER는 TD3 및 SAC와 같은 고급 오프정책 강화 학습 프레임워크와의 통합이 용이하여 실용성과 다재다능성을 더욱 강조합니다.



### Learning Provablely Improves the Convergence of Gradient Descen (https://arxiv.org/abs/2501.18092)
Comments:
          46 pages, 11 figures

- **What's New**: 이번 연구에서는 Learning to Optimize (L2O)라는 딥러닝의 전문적 분야가 최적화 문제를 해결하기 위해 DNN 기반의 솔버를 훈련하는 데 초점을 맞추고 있습니다. 기존의 연구는 엄격한 가정에 의존하고 있어 실제 훈련 과정의 복잡성과 잘 맞지 않다는 점에서 개선이 필요합니다. 이 연구는 L2O의 수렴(convergence)을 훈련 방법론(training methodology)을 통해 확립하는 것을 목표로 합니다.

- **Technical Details**: 우리는 알고리즘의 하이퍼파라미터(hyperparameters)를 학습하는 것이 수렴(convergence)을 크게 향상시킨다는 것을 입증합니다. 특히, 이 연구는 이차 프로그래밍(quadratic programming)에 대한 경량 경량하강법(gradient descent, GD) 알고리즘을 중심으로 하였으며, 신경탄젠트 커널 이론(neural tangent kernel theory)을 통해 L2O의 훈련 수렴을 증명했습니다. 또한 합성 데이터셋(synthetic datasets)을 사용하여 실험적으로 평가를 수행했습니다.

- **Performance Highlights**: 우리의 연구 결과는 GD 방법들에 비해 50	ext{\%} 이상의 성능 향상을 초과하는 것으로 나타났습니다. 이는 L2O 방법론이 전통적인 최적화 알고리즘보다 뛰어난 성능을 보일 수 있음을 보여줍니다. 이 연구는 최적화 문제의 해결책을 찾는 데 있어서 L2O의 잠재력을 드러냅니다.



### DIAL: Distribution-Informed Adaptive Learning of Multi-Task Constraints for Safety-Critical Systems (https://arxiv.org/abs/2501.18086)
Comments:
          16 pages, 14 figures, 6 tables, submission to T-RO in 2024

- **What's New**: 본 논문은 기존의 안전 강화학습(Safe Reinforcement Learning) 개념을 진화시켜 안전 제약 분포를 여러 작업 간에 학습하는 새로운 방법, 즉 DIAL(Distribution-informed Adaptive Learning)을 제시합니다. 이 방법은 다중 작업에서 공유 지식을 활용하여 새로운 작업에 적응하면서 안전성과 샘플 효율성을 개선합니다. DIAL은 모방 학습을 통해 공유 제약을 파악하고 이들 제약의 위험 수준을 조절하여 다양한 작업에 적용할 수 있는 유연성을 제공합니다.

- **Technical Details**: DIAL은 다중 작업 데모에서 위험 분포를 학습하는 것을 중심으로, 조건부 가치 위험(CVaR)과 같은 왜곡된 기준을 통해 동적으로 위험 수준을 조정합니다. 이 접근 방식은 안전 탐색을 극대화하기 위해 엔트로피를 최대화하여 위험을 조절할 수 있는 기능을 부여합니다. 또한, DIAL은 기존의 ICRL 프레임워크에 두 가지 주요 혁신을 추가하여 제약 함수 및 정책을 학습하는 방향으로 접근합니다.

- **Performance Highlights**: 실험 결과 DIAL 방법이 기존 기준선에 비해 뛰어난 안전성을 가지고 있음을 입증하였습니다. 특히, 특정 작업에 대한 제약 정의가 필요 없는 상황에서도 안전성과 성공률이 높았습니다. 이는 DIAL이 다양한 실제 작업에 걸쳐 강력한 실용성을 보여주고 있음을 앞으로의 작업에 적용할 수 있는 기회를 제공합니다.



### Towards Transparent and Accurate Diabetes Prediction Using Machine Learning and Explainable Artificial Intelligenc (https://arxiv.org/abs/2501.18071)
- **What's New**: 이번 연구는 당뇨병 예측을 위한 프레임워크를 제시하며, 머신러닝 (Machine Learning) 모델과 함께 설명 가능한 인공지능 (eXplainable Artificial Intelligence, XAI) 도구를 활용하여 예측 정확성과 해석 가능성을 조사합니다. 당뇨병 이진 건강 지표 데이터셋을 기반으로 설계된 이 시스템은 조기 진단과 관리의 중요성을 강조합니다.

- **Technical Details**: 데이터 전처리는 클래스 불균형 (class imbalance)과 임상 특성의 변동성을 다루기 위해 합성 소수 샘플 오버 샘플링 기법 (Synthetic Minority Oversampling Technique, SMOTE)과 특징 스케일링을 포함합니다. 앙상블 모델은 92.50%의 테스트 정확도(test accuracy)와 0.975의 ROC-AUC를 제공하여 높은 정확성을 입증했습니다.

- **Performance Highlights**: 모델 설명에서 가장 영향력 있는 예측 변수로는 BMI, 나이 (Age), 일반 건강, 소득 (Income), 신체 활동 (Physical Activity) 등이 나타났습니다. 본 연구 결과는 머신러닝과 XAI를 결합하여 의료 시스템에서 사용할 수 있는 정확하고 계산적으로 투명한 도구 개발의 가능성을 제시합니다.



### FinanceQA: A Benchmark for Evaluating Financial Analysis Capabilities of Large Language Models (https://arxiv.org/abs/2501.18062)
Comments:
          10 pages, 7 figures

- **What's New**: FinanceQA는 복잡한 수치 금융 분석 과제를 평가하는 테스트 스위트입니다. 이 시스템은 실제 투자 작업을 반영하며, 최근 LLM의 발전에도 불구하고 현재 모델이 금융 기관의 정확성 요구를 충족하지 못하는 문제를 다룹니다.

- **Technical Details**: 현재 LLM 모델은 헤지펀드, 사모펀드, 투자은행 등에서 실제로 수행되는 업무 분석을 모방한 과제에서 약 60%의 실패율을 보이고 있습니다. 주요 도전 과제는 핸드-스프레딩 메트릭스, 표준 회계 및 기업 가치 평가 관례 준수, 그리고 정보가 불완전한 상황에서의 분석 수행 등입니다.

- **Performance Highlights**: 이러한 성능 차이는 기존 LLM의 기능과 직업 금융 분석의 요구 사이의 괴리를 강조합니다. 더 높은 품질의 훈련 데이터가 필요한 것을 보여주며, OpenAI의 파인튜닝 API를 통해 실험이 수행됩니다.



### Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probability Ratio Tes (https://arxiv.org/abs/2501.18059)
Comments:
          Accepted to International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이번 논문은 시간에 민감한 기계 학습을 위한 SPRT(Sequential Probability Ratio Test)의 한계를 극복하고자 FIRMBOUND라는 새로운 프레임워크를 도입합니다. FIRMBOUND는 훈련 데이터로부터 역 귀납법(backward induction)의 해결책을 효율적으로 추정하여 이론적인 최적 정지 규칙을 실용적으로 적용할 수 있게 합니다. 이 프레임워크는 조건부 기대값(condition expectation)과 충분 통계량(sufficient statistic)가 필요하며, 이를 통해 Bayes 리스크를 최소화하고 최적성을 달성합니다.

- **Technical Details**: FIRMBOUND는 두 가지 접근 방식을 통해 역 귀납법을 해결합니다. 첫 번째는 조건부 기대값의 오목한 성질을 인식하고, 이를 볼록 함수 학습(convex function learning)으로 설정하여 통계적으로 일관된 추정기를 제공합니다. 두 번째로, Gaussian 프로세스 회귀(Gaussian process regression)를 활용하여 훈련 속도를 30배까지 개선하는 방법을 제안합니다. 두 모델 모두 테스트 단계에서 낮은 배치 비용을 제공하여 실시간 ECTS에 적합하게 설계되었습니다.

- **Performance Highlights**: FIRMBOUND는 Bayes 최적성에 효과적으로 접근하며, 속도-정확도 무기록 최적화 문제의 Pareto-최적점을 도출합니다. 실험 결과, FIRMBOUND는 기존 ECTS 방법들보다 덜 민감한 매개변수를 가지고 높은 성능을 발휘하였습니다. 또한, FIRMBOUND는 정적 기준을 갖는 SPRT 보다도 낮은 오류율을 보여주며, 실용적인 의사 결정 과정에서 신뢰성을 보장합니다.



### Current Pathology Foundation Models are unrobust to Medical Center Differences (https://arxiv.org/abs/2501.18055)
- **What's New**: 이 논문에서는 Pathology Foundation Models (FMs)의 강건성(robustness)을 측정하기 위한 새로운 지표인 Robustness Index를 도입합니다. 현재 사용 가능한 다양한 pathology FM을 평가하여 이들 모델이 생물학적 특징과 혼란스러운 특징의 영향을 얼마나 잘 구별하는지를 분석합니다. 이 연구는 병원 간의 차이와 같은 외부 요인에 얼마나 민감한지를 평가하여, 해당 모델들이 실제 의료 환경에서 신뢰할 수 있는지 확인하는 데 목적이 있습니다.

- **Technical Details**: 이 연구는 생물학적 특징과 혼란스러운 특징(confounding features) 간의 구분을 통해 강건성 개념을 정의하고, Robustness Index를 통해 이것을 정량적으로 측정합니다. 주어진 데이터셋에서 k개의 최근접 이웃(Nearest Neighbors) 간의 비율로 강건성을 평가하며, 유사도 평가를 위해 코사인 거리(cosine distance)를 사용합니다. 이를 통해 매개변수가 어떻게 생물학적 신호를 반영하는가를 시각적으로 분석할 수 있습니다.

- **Performance Highlights**: 10개의 현재 공개된 pathology FM을 평가한 결과, 대부분의 모델이 특정 의료 센터에 의해 강하게 영향을 받는 것으로 나타났습니다. 단 하나의 모델만이 Robustness Index가 1보다 크게 나타나, 생물학적 특징이 혼란스러운 특징보다 우위에 있다는 것을 시사합니다. 이러한 결과는 패스올로지 AI 모델이 실제 임상에서 수용되는 데 중요한 지침을 제공할 수 있습니다.



### SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders (https://arxiv.org/abs/2501.18052)
- **What's New**: 본 논문에서는 텍스트-이미지 확산 모델에서 원치 않는 개념을 제거하기 위한 새로운 방법인 SAeUron을 소개합니다. SAeUron은 스파스 오토인코더(sparse autoencoders, SAE)가 학습한 특징을 활용하여, 기존의 미세조정(fine-tuning) 방식이 아닌 더 명확한 개입을 가능하게 합니다. 이를 통해 개념이 정확히 제거되었는지 또는 단지 마스킹(masking)되었는지를 명확히 알 수 있습니다.

- **Technical Details**: SAE는 여러 디노이징 타임스텝(denoising timesteps)에서 활성화된 특징을 기반으로 비지도 학습 방식으로 훈련되어 특정 개념에 해당하는 희소하고 해석 가능한 특징을 캡처합니다. 본 연구에서는 이러한 특징들 중 개념 특정 특징을 선택하는 방법을 제안하였습니다. 이를 통해 모델의 활성화에 정밀한 개입을 하여 목표 콘텐츠를 차단하면서도 전체 성능은 유지할 수 있습니다.

- **Performance Highlights**: 경쟁력이 있는 UnlearnCanvas 벤치마크(object and style unlearning)에서 SAeUron의 최첨단 성능이 입증되었습니다. 특히 하나의 SAE를 사용하여 동시에 여러 개념을 제거할 수 있으며, 기존의 방법들과 달리 SAeUron은 적대적 공격(adversarial attack) 하에서도 원치 않는 콘텐츠 생성을 방지합니다.



### Joint Pricing and Resource Allocation: An Optimal Online-Learning Approach (https://arxiv.org/abs/2501.18049)
- **What's New**: 새로운 동적 가격 책정 및 자원 할당 문제에 대한 연구를 통해 불확실성 하에 가격 및 재고 결정의 온라인 학습 알고리즘을 제시합니다. 본 연구는 가격과 재고가 수요에 미치는 영향을 동시에 고려하여, 제반 비용과 수요를 통합적으로 관리합니다. 기존 연구는 주로 수요 곡선 추정에 집중했지만, 본 연구는 공급 측의 복잡한 결정을 해결하려는 혁신적인 접근법을 도입했습니다.

- **Technical Details**: 본 알고리즘은 Lower-Confidence Bound (LCB) 메타 전략을 활용하여 다수의 온라인 Convex Optimization (OCO) 에이전트 간의 공동 결정을 최적화합니다. 재고 결정은 매 시간 기간의 시작 시점에 이루어지며, 이후의 재고 배분은 가격에 따른 확률적 수요에 기반하여 진행됩니다. 이 알고리즘은 비볼록성 (non-convexity)과 비조화성 (non-smoothness) 문제를 다루면서 효율적으로 운영됩니다.

- **Performance Highlights**: 알고리즘은 전체 $T$ 기간에 대해 $	ilde{O}(	ext{sqrt}(Tmn))$의 후회(regret) 값을 달성하여 최적의 성능을 보입니다. 이는 정보 이론적 하한과 일치하여, 동적 가격 책정 및 재고 관리 분야에서의 효율성을 강조합니다. 이 연구는 불확실성 하에 자원 할당을 고려한 새로운 시사점을 제공하며, 온라인 학습 알고리즘의 실용성을 입증합니다.



### KNN and K-means in Gini Prametric Spaces (https://arxiv.org/abs/2501.18028)
- **What's New**: 이번 논문은 Gini prametric spaces에 기초하여 K-평균 (K-means) 및 K-최근접 이웃 (KNN) 알고리즘의 혁신적인 개선을 소개합니다. Gini 기반의 거리 측정은 값 기반 및 등급 기반 정보를 모두 포함하여 노이즈 및 이상치에 대한 강인성을 개선합니다. 이 연구의 주된 기여는 Gini 기반 측정의 제안, Gini K-means 알고리즘의 수렴성과 노이즈에 대한 회복력을 입증한 것, 그리고 Gini KNN 방법이 것입니다.

- **Technical Details**: K-평균 알고리즘은 클러스터 중심점과 데이터 포인트 간의 거리 측정을 통해 클러스터에 포인트를 반복적으로 할당합니다. 이 과정은 수렴할 때까지 계속되며, Gini K-means는 노이즈 데이터에 대한 회복력도 보여줍니다. KNN은 테스트 데이터 포인트와 주어진 K개의 레이블이 있는 훈련 데이터 포인트 간의 거리를 측정하여 레이블을 할당합니다.

- **Performance Highlights**: 14개의 UCI 데이터셋에 대한 실험 결과, Gini 기반 알고리즘은 클러스터링 및 분류 작업에서 기존 알고리즘보다 우수한 성능과 효율성을 보여주었습니다. 특히 Gini KNN은 주목받는 Hassanat 거리와 비교해도 경쟁력을 가지고 있으며, 노이즈 환경에서도 효과적으로 작동합니다.



### A Proximal Operator for Inducing 2:4-Sparsity (https://arxiv.org/abs/2501.18015)
- **What's New**: 본 논문은 AI 가속기와 GPU의 최신 발전을 활용하여 희소 행렬 곱셈(sparse matrix multiplications)을 효율적으로 수행하는 방법을 제시합니다. 특히, 4개의 연속 가중치 중 2개가 0으로 설정된 2:4 희소성(sparsity)의 적용으로 기존보다 더 나은 효율성을 발휘하였으나, 이는 모델의 정확성을 감소시킬 수 있습니다. 이 문제를 해결하기 위해 지역 특성(local correlation)을 활용한 새로운 정규화 기법을 도출하고, 이를 통해 훈련된 모델에서 더 나은 희소성 마스크(sparsity masks)를 찾으려 합니다.

- **Technical Details**: 우리는 훈련된 모델의 희소성을 정규화하는 새로운 정규화 기법을 제안하며, 이를 통해 지역 제곱 손실(local squared loss)을 최소화합니다. 우리는 proximal operator를 도출하여 이 비선형(noncovex) 문제를 해결하는 효율적인 방법을 제시합니다. 각 반복(iteration) 동안 이 proximal operator를 해석적으로 가능하게 하여, 최적화 마스크(mask)를 사용하여 지역 제곱 손실을 추가로 최소화합니다.

- **Performance Highlights**: 모델 파라미터 수가 최대 70B인 대형 언어 모델에 대한 프루닝(pruning) 작업에 우리의 방법을 적용했습니다. 13B 모델에서는 기존의 최첨단 알고리즘을 초과하는 성능을 달성했으며, 70B 모델에서는 기존의 알고리즘과 동일한 성능을 나타냈습니다. 우리의 접근 방식은 기계 학습 모델의 효율성을 더욱 높이는 데 중요한 영향을 미칠 것으로 기대됩니다.



### When less is more: evolving large neural networks from small ones (https://arxiv.org/abs/2501.18012)
Comments:
          8 pages, 7 figures

- **What's New**: 기존의 인공 신경망과는 달리, 본 연구에서는 학습 중에 노드를 동적으로 추가하거나 제거할 수 있는 소형 적응형 피드포워드 신경망인 Nimble Neural Networks를 제안합니다. 네트워크의 크기는 하나의 가중치에 의해 제어되며, 이는 다른 가중치 및 편향들과 동일한 경량 경량화 알고리즘으로 최적화됩니다. 이러한 접근 방식은 비선형 회귀 및 분류 작업에서 기존의 정적 신경망보다 뛰어난 성능을 보입니다.

- **Technical Details**: Nimble Neural Networks는 훈련 중에 크기를 동적으로 조정할 수 있으며, 목표 함수에 기반하여 크기 의존적 손실 함수로 가중치를 최적화합니다. 이 알고리즘은 기존의 네트워크 pruning 방법론과는 달리 네트워크의 크기를 점진적으로 증가시키며 적절한 크기로 축소할 수 있는 능력을 가지고 있습니다. 본 연구는 간단한 비선형 회귀 및 분류 예제에서 이 네트워크의 성능을 시연합니다.

- **Performance Highlights**: Nimble Neural Networks는 비선형 회귀 및 분류 작업에서 정적 네트워크를 능가하는 성능을 보여줍니다. 특정 손실 함수 및 경량 경량화 알고리즘을 통해 네트워크 크기를 효과적으로 조정하면서, 필요한 최소한의 신경망 구조를 유지할 수 있는 가능성을 열어줍니다. 이러한 접근 방식은 연결성의 변화를 통해 네트워크의 동적 특징을 탐색하고, 보다 효율적이며 에너지 소모가 적은 모델을 생성할 수 있는 기회를 제공합니다.



### Topological Signatures of Adversaries in Multimodal Alignments (https://arxiv.org/abs/2501.18006)
- **What's New**: 최근에 개발된 다중모달 기계 학습 시스템, 특히 텍스트와 이미지 데이터를 정렬하는 CLIP 및 BLIP 모델은 적대적 공격에 취약하다는 점이 연구 중에 발견되었습니다. 기존 연구는 단일 모달 시스템에 대한 적대적 강인성에 초점을 맞추었으나, 다중모달 시스템에 대한 방어 전략은 아직 충분히 탐구되지 않았습니다. 본 연구는 이미지와 텍스트 임베딩 간의 위상적 서명을 조사하며, 적대적 공격이 이들의 정렬을 어떻게 방해하는지 보여줍니다.

- **Technical Details**: 본 연구에서 제안하는 두 가지 새로운 Topological-Contrastive 손실, 즉 Total Persistence (TP)와 Multi-scale Kernel (MK) 손실은 적대적 변동에 의해 생성된 위상적 서명을 분석하는 데 사용됩니다. Persistent Homology라는 위상적 데이터 분석의 기법을 활용하여 이미지-텍스트 정렬의 변화를 추적하고, 이 손실들이 적대적 샘플의 비율에 따라 어떻게 변화하는지를 관찰하였습니다. 이러한 손실은 적대적 샘플이 포함된 데이터 배치에서 이미지와 텍스트 임베딩 간의 불일치를 포착합니다.

- **Performance Highlights**: CIFAR-10과 ImageNet 데이터셋을 사용하여 실험을 수행한 결과, 제안된 TC 손실들이 이미지-텍스트 정렬의 정밀한 특성을 감지하는 데 매우 효과적임을 입증했습니다. 또한, Maximum Mean Discrepancy (MMD) 테스트에 이 손실을 통합하여 기존 MMD 솔루션의 정확성과 1형 오류를 제어하는 데 있어 우수한 성능을 발휘하였습니다. 이 연구는 다중모달 AI 시스템에 대한 적대적 공격 감지의 새로운 접근법을 제공합니다.



### KoopAGRU: A Koopman-based Anomaly Detection in Time-Series using Gated Recurrent Units (https://arxiv.org/abs/2501.17976)
- **What's New**: 이 논문에서는 복잡하고 비선형적인 시간적 동역학(time-series dynamics)을 처리하기 위해 Fast Fourier Transform(FFT), Deep Dynamic Mode Decomposition(DeepDMD), 및 Koopman 이론을 결합한 새로운 딥러닝 모델인 KoopAGRU를 소개합니다. KoopAGRU는 시간-변화 및 시간-불변 구성 요소로 데이터를 분해하여 복잡한 패턴을 정밀하게 모델링합니다. 이 모델은 Gate Recurrent Unit (GRU) 인코더를 활용하여 Koopman 관측값을 학습하여 다양한 시간 눈금에서 이상 탐지 능력을 향상시킵니다.

- **Technical Details**: KoopAGRU는 단일 프로세스에서 훈련되어 빠른 추론 시간을 제공합니다. FFT는 주어진 데이터의 주파수 구성 요소로 분해하며, DeepDMD는 데이터에서 Koopman 연산자와 관측 가능한 함수(observable functions)를 직접 식별합니다. GRU 인코더는 데이터 표현을 효과적으로 학습하여 이상을 식별하는 데 필수적인 시간적 동역학을 캡처하는 데 기여합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트에서 신뢰성 있게 테스트한 결과, KoopAGRU는 다른 주요 방법보다 더 나은 성과를 보이며, 잘 알려진 이상 탐지 작업에 대해 평균 F1-score 90.88%를 달성하였습니다. 이 모델은 적은 매개변수와 자원 사용으로도 높은 강건성과 효과성을 보이며 실제 시나리오에서의 이상 탐지에 효과적입니다.



### Variational Combinatorial Sequential Monte Carlo for Bayesian Phylogenetics in Hyperbolic Spac (https://arxiv.org/abs/2501.17965)
Comments:
          24 pages, 10 figures

- **What's New**: 이 논문에서는 고차원 계통 유추(phylogenetic inference) 작업을 위한 새로운 하이퍼볼릭( hyperbolic ) 공간에서의 시퀀셜 서치 알고리즘을 개발하여 기존의 유클리드(Euclidean) 접근방식보다 향상된 성능을 보인다. 특히, Combinatorial Sequential Monte Carlo (Csmc)와 Nested Combinatorial Sequential Monte Carlo (Ncsmc) 알고리즘의 하이퍼볼릭 확장을 통해 효율적이고 일관된 추정기를 제안하였다. 이 방식은 고차원 데이터에서 초지수적( super-exponential ) 스케일링 문제를 해결하는 데 기여한다.

- **Technical Details**: 하이퍼볼릭 기하학은 고정된 음의 곡률로 정의된 비유클리드 기하학의 일종이다. 특히 포앵까레 원 모델(Poincaré disk model)을 중심으로 하여 하이퍼볼릭 공간의 메트릭 및 기하학적 특성을 중점적으로 다룬다. 이 연구에서는 하이퍼볼릭 공간 내의 제안 분포를 정의하고 일관된 추정기를 입증하여 근사 사후( posterior ) 및 변분 추론 기법을 구축하였다. 또한, 이러한 기법들은 각기 다른 모델에서 효율적으로 적용될 수 있다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 하이퍼볼릭 확장 알고리즘은 기존의 근사 베이지안 추론 방법보다 20배에서 50배까지의 속도 향상을 보였다. 높은 차원에서의 계통 유추 작업에서 빠른 속도, 확장성( scalability ) 및 성능을 입증하였다. 이러한 성과는 Gpu 가속을 활용하여 더욱 개선되었으며, 결과적으로 계통 유추 분야에서의 중요한 기여로 평가된다.



### Deep Ensembles Secretly Perform Empirical Bayes (https://arxiv.org/abs/2501.17917)
- **What's New**: 이 논문은 딥 앙상블(deep ensembles)과 베이지안 신경망(Bayesian neural networks, BNNs) 간의 관계를 밝혀내어 두 접근법이 본질적으로 다르지 않다는 것을 보여줍니다. 이는 딥 앙상블이 데이터에 의존하여 학습되는 사전(prior)을 통해 후방 분포(posterior)을 얻어내면서 정확한 베이지안 평균(Bayesian averaging)을 수행한다는 사실을 기반으로 합니다. 이에 따라, 딥 앙상블은 BNN의 일종으로 여겨질 수 있으며, 이로 인해 두 접근법 간의 깊은 연관성을 제공합니다.

- **Technical Details**: 논문에서 제안하는 바에 따르면, 딥 앙상블은 최대 주변 우도(maximum marginal likelihood)를 통해 유연하게 학습 가능할 사전 분포를 도출함으로써 BNNs와 등가입니다. 이 과정에서, 고차원 데이터에 대해 더 강한 사전(explained prior)을 제공하는 혼합(point masses mixture) 방식이 사용됩니다. 이러한 새로운 관점은 여태까지 BNN과 앙상블 간의 오해를 해소하며 두 방법의 상관관계를 명확히 합니다.

- **Performance Highlights**: 이 연구는 딥 앙상블의 성능이 뛰어난 이유에 대한 설명을 제시하며, 향후 UQ(uncertainty quantification)에 대한 더 나은 통찰력을 제공할 것으로 기대됩니다. 또한, 강력한 예측 밀도(predictive density)를 통해 신뢰할 수 있는 불확실성 추정치를 제공하는 데 있어 앙상블의 중요성을 강조하며, 의료 진단, 자율주행 등 안전이 중요한 분야에서의 응용 가능성을 제시합니다. 이러한 이해는 궁극적으로 영감을 줄 방법론의 발전으로 이어질 가능성이 큽니다.



### DReSS: Data-driven Regularized Structured Streamlining for Large Language Models (https://arxiv.org/abs/2501.17905)
- **What's New**: 이 논문에서는Llarge language models (LLMs)의 성능을 유지하면서 모델 크기를 줄일 수 있는 새로운 프루닝(paruning) 패러다임을 제안합니다. 기존의 프루닝 방법이 정보 손실을 초래하는 것과 달리, 새로운 접근 방식에서는 먼저 정규화(regularization)를 적용한 후 프루닝을 수행하고, 마지막으로 미세 조정(finetuning)을 진행합니다. 이 과정을 통해 DReSS(Data-driven Regularized Structured Streamlining)라는 효과적인 방법을 도입하게 되었습니다.

- **Technical Details**: DReSS는 파라미터 행렬에서 선택된 채널에 정규화 과정을 적용하여 중요 정보를 전이하여 제거되는 부분의 정보를 보존합니다. 이 방법은 프로세스가 다음 네 단계로 구성되어 있습니다: 데이터 선택, 정규화 적용, 채널 프루닝 후 RFT 수행입니다. 작은 데이터 세트를 활용함으로써 프루닝 과정에서 발생하는 오버헤드를 최소화하고, 높은 프루닝 비율에서도 성능 저하를 방지합니다.

- **Performance Highlights**: 실험 결과 DReSS는 기존 프루닝 방법보다 크게 향상된 정밀도와 정확도를 보였습니다. 이 방법은 프루닝 비율이 극단적인 상황에서도 성능이 크게 향상되어 상당한 지연 시간 감소와 처리량 증가를 이끌어냈습니다. 따라서 DReSS는 대형 언어 모델에서 정보 손실을 줄이고, 언어 모델링 능력을 향상시키는 데 기여했습니다.



### Shared DIFF Transformer (https://arxiv.org/abs/2501.17900)
Comments:
          arXiv admin note: text overlap with arXiv:2501.17486

- **What's New**: 이번 연구에서는 Shared DIFF Transformer를 제안하며, 이는 전체 패턴을 모델링하기 위해 공유 기반 행렬을 도입하고, 낮은 차원의 업데이트를 통해 작업 특화된 유연성을 강화한다. 이 구조는 파라미터 중복성을 크게 줄이고 높은 효율성을 제공하며, 노이즈 억제 능력을 유지하는 것에 중점을 두었다.

- **Technical Details**: Shared DIFF Transformer는 디코더 전용 설계를 따른다. 입력 시퀀스를 쿼리, 키, 값 행렬로 프로젝션하는 공통 차별 주의 메커니즘을 사용하며, 각각의 레이어는 shared differential attention 모듈과 피드 포워드 네트워크(FFN) 모듈로 구성된다. 공유된 기반 행렬과 낮은 차원의 매트릭스를 사용하여 모델의 유연성을 높이고 파라미터 효율성을 극대화했다.

- **Performance Highlights**: 실험 결과, Shared DIFF Transformer는 DIFF Transformer와 비교했을 때 언어 모델링 성능이 유사하고, 파라미터 수와 훈련 토큰 수를 크게 줄였다. 다양한 다운스트림 작업에서 성능 우위를 보이며, 우수한 확장성을 갖췄다는 것을 입증했다. 이 결과로, Shared DIFF Transformer는 대규모 언어 모델에 대한 강력하고 효율적인 아키텍처로 자리 잡았다.



### Explainable Machine Learning: An Illustration of Kolmogorov-Arnold Network Model for Airfoil Lift Prediction (https://arxiv.org/abs/2501.17896)
Comments:
          3 pages, 2 tables, 3 figures

- **What's New**: 이번 연구는 데이터 과학이 과학 탐사의 네 번째 패러다임으로 자리잡고 있음을 보여줍니다. 많은 머신러닝 모델들이 블랙 박스로 작동하여 그 예측 뒤에 있는 이유를 제한적으로 설명한다는 문제점이 있었습니다. 최근에 제안된 Kolmogorov-Arnold Network(칸 모델)은 설명 가능한 AI(explainable AI)를 포함하는 대안 모델로 주목받고 있습니다.

- **Technical Details**: 이 연구에서는 KAN 모델과 다섯 가지 인기 있는 감독형 머신러닝 모델들이 항공우주 공학에서의 공기foil 양력(lift) 예측 문제에 적용되었습니다. 2900개의 다양한 공기foil에서 생성된 표준 데이터를 사용하여 테스트를 진행하였습니다. KAN 모델은 테스트 데이터에서 96.17%의 R2 스코어를 기록하며, 기본 모델과 Multi Layer Perceptron를 초월하는 성과를 보였습니다.

- **Performance Highlights**: KAN 모델의 설명 가능성은 모델을 가지치기(pruning)하고 기호화(symbolizing)하여 입력 변수에 따른 양력 계수(coefficient of lift)에 대한 방정식을 생성함으로써 입증되었습니다. KAN 모델에서 추출된 설명 가능한 정보는 공기foil에 의한 양력 발생의 알려진 물리학과 일치하여, 과학 탐사에 기여할 수 있는 잠재력을 보여줍니다.



### Diffusion Autoencoders are Scalable Image Tokenizers (https://arxiv.org/abs/2501.18593)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 간단한 확산 토크나이저(Diffusion Tokenizer, DiTo)를 소개합니다. 이 토크나이저는 이미지 생성 모델을 위한 컴팩트한 시각적 표현을 학습하는데 초점을 맞추고 있습니다. 기존의 복잡한 비지도 모델 대신, 단일한 학습 목표인 확산 L2 손실(diffusion L2 loss)을 활용하여 확장 가능한 이미지 토크나이저를 효율적으로 훈련할 수 있다는 것이 주요 통찰입니다.

- **Technical Details**: DiTo는 이미지 생성을 위한 잠재 표현 학습을 위한 새로운 접근 방식을 제시합니다. 이는 확산 모델(difussion model)의 이론적 근거를 바탕으로 하여, 단일한 확산 L2 손실을 사용해 훈련하게 됩니다. 제안된 기술은 토크나이저 학습에서 여러 손실 조합을 요구하는 기존 방법들과 대비되어 보다 간단하고 효율적인 모델 학습을 가능하게 합니다.

- **Performance Highlights**: DiTo는 이미지 재구성과 다운스트림 이미지 생성 작업에서 기존 최첨단 모델과 비교하여 경쟁력 있는 혹은 더 나은 품질을 달성합니다. 특히 작은 텍스트나 기호, 구조적 비주얼 부분 처리에서 기존의 GLPTo보다 더 뛰어난 결과를 보여줍니다. 또한, 모델 크기를 늘려도 손실 하이퍼파라미터 조정이 필요 없어 간편하게 확장 가능하다는 장점을 가지고 있습니다.



### Advances in Multimodal Adaptation and Generalization: From Traditional Approaches to Foundation Models (https://arxiv.org/abs/2501.18592)
Comments:
          Project page: this https URL

- **What's New**: 본 연구는 다중 모드(domain adaptation) 적응 및 일반화(multimodal generalization)에 대한 최근의 발전 사항을 포괄적으로 정리했습니다. 전통적인 접근 방식에서부터 다중 모드 기초 모델까지 다양한 방법론을 다루며, 아카이브(arXiv)에서 활발히 업데이트되고 있는 자료들을 포함합니다. 각 주제에 대한 문제 정의와 기존 방법들에 대한 철저한 검토가 이루어지고 있어, 향후 연구 방향에 대한 통찰도 제공합니다.

- **Technical Details**: 다중 모드 적응 및 일반화를 위한 다양한 알고리즘이 제안되었으며, 특히 시각-언어(target domains), 오디오-비디오(audio-video) 및 LiDAR-카메라(LiDAR-camera)와 같은 데이터 소스로부터 출발합니다. 이 논문에서는 MMDA(Multimodal Domain Adaptation), MMDG(Multimodal Domain Generalization), 그리고 MMTTA(Multimodal Test-Time Adaptation) 등의 주요 개념들과 함께 이들을 향상시키기 위한 기초 모델들의 역할도 탐구합니다. 더불어, 각 방법에 대한 체계적인 분석과 기존 데이터셋, 응용 분야를 정리하였습니다.

- **Performance Highlights**: MMDA와 MMDG는 최근 행동 인식(action recognition)과 의미 분할(semantic segmentation) 분야에서 눈에 띄는 성과를 내고 있습니다. 특히, 다중 모드 기초 모델의 활용을 통해 모델들의 일반화 성능이 향상되는 것을 확인하였습니다. 이 연구는 기존 알고리즘의 이해를 도모하며, 다중 모드 적응 수준에서의 보완 정보를 효과적으로 활용하기 위한 새로운 방향을 제시합니다.



### R.I.P.: Better Models by Survival of the Fittest Prompts (https://arxiv.org/abs/2501.18578)
- **What's New**: 본 논문에서는 데이터 품질이 모델 성능에 미치는 영향을 측정하는 Rejecting Instruction Preferences (RIP) 방법을 제안합니다. 이 방법은 저품질 프롬프트가 높은 변동성과 낮은 품질의 응답을 초래한다는 가정하에 데이터 무결성을 평가합니다. RIP는 기존의 훈련 데이터에서 프롬프트를 필터링하거나 고품질의 합성 데이터셋을 생성하는 데 사용됩니다.

- **Technical Details**: RIP 방법은 선택된 응답과 거부된 응답 쌍을 기반으로 프롬프트를 필터링하며, 거부 응답 품질과 선택된 응답과의 보상 차이를 주요 메트릭으로 삼습니다. 본 방법은 Direct Preference Optimization (DPO)와 같은 강화학습 방법을 사용하여 모델을 미세 조정하는 데 활용되며, 저품질 프롬프트를 효과적으로 제거할 수 있습니다. 구체적으로, RIP는 Wildchat 프롬프트에 대해 필터링하여 Llama 3.1-8B-Instruct와 Llama 3.3-70B-Instruct 모델에서 성능 향상을 보였습니다.

- **Performance Highlights**: RIP는 AlpacaEval2 LC Win Rate를 9.4%, Arena-Hard를 8.7%, WildBench를 9.9% 향상시켰습니다. Llama 3.3-70B-Instruct 모델에서는 Arena-Hard에서 순위가 18위에서 6위로 올라가는 성과를 보였습니다. 이러한 성과는 RIP 필터링 방법이 모델 성능을 획기적으로 향상시킬 수 있음을 보여줍니다.



### Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling (https://arxiv.org/abs/2501.18577)
- **What's New**: 이 논문에서는 머신러닝 모델의 예측 결과를 통계 분석에 활용할 때 발생할 수 있는 오류를 수정하는 새로운 방법을 제안합니다. 특히 Predict-Then-Debias 추정기 방법을 확장하여 bootstrap 신뢰구간을 도입하여 비균일 데이터 샘플에서도 적용 가능하도록 하였습니다. 이를 통해 머신러닝 모델의 정확성에 대한 가정 없이도 유효한 신뢰구간을 생성할 수 있습니다.

- **Technical Details**: 저자들은 머신러닝을 통해 손실된 변수를 보완하는 새로운 신뢰구간을 개발했습니다. 예측 결과가 부분적으로 관측된 데이터와 결합될 때 데이터 샘플의 종류와 관계없이 예측 오류를 보정할 수 있는 강력한 접근 방식을 제공합니다. 이 연구에서는 p차원 데이터 X를 관측 데이터와 결측 데이터로 나누어 처리하며, 머신러닝 예측을 통한 대체 방법과 그 효과를 설명합니다.

- **Performance Highlights**: 제안된 방법은 머신러닝 모델의 성능과는 무관하게 신뢰구간을 생성할 수 있어 강력한 이점을 제공합니다. 기존 방법들보다 신뢰구간이 넓지 않으며, 실질적인 성과를 입증하는 다양한 실험 결과를 포함합니다. 이러한 접근법은 다양한 통계 분석 작업과 데이터 세트에 적용이 가능하여 유용한 도구가 될 것입니다.



### Differentially Private Steering for Large Language Model Alignmen (https://arxiv.org/abs/2501.18532)
Comments:
          ICLR 2025; Code: this https URL

- **What's New**: 이 연구는 Large Language Models(LLMs)의 행동을 개인 데이터셋과 정렬하는 최초의 연구를 제시합니다. 	extit{Private Steering for LLM Alignment(PSA)} 알고리즘을 개발하여, 비유출 개인 정보의 보장을 위한 차별적 개인 정보 보호(Differential Privacy, DP)를 사용합니다. 연구를 통해 PSA가 성능 손실을 최소화하면서도 LLM 정렬에 대한 DP 보장을 달성할 수 있음을 보여주었습니다.

- **Technical Details**: 이 연구는 LLM의 활성화(activation)를 편집하는 새로운 방법인 PSA를 통해 여러 공개 데이터셋에서 진행된 광범위한 실험 결과를 보고합니다. PSA는 정렬 목적에 맞는 긍정적(예: 사실적인) 사례의 정보를 보존하고 부정적(예: 환상) 사례의 정보를 최소화하여, LLM의 활성화를 수정합니다. 본 연구는 또한 활성화 편집을 통한 개인정보 보호의 중요성에 대해 논의하며, Membership Inference Attack(MIA)에 대한 첫 번째 연구를 제안합니다.

- **Performance Highlights**: PSA는 0.5B에서 7B 규모의 다양한 오픈 소스 LLM을 사용한 7개의 데이터셋에서 시험되었습니다. 결과에 따르면 PSA는 LLM의 정렬 메트릭 및 열린 텍스트 생성 품질에서 minimal한 성능 손실을 보여주며 DP 보장을 달성했습니다. 또한 PSA는 활성화 편집을 통한 개인 정보 유출 위험을 낮추는 효과를 입증하였습니다.



### Graph Learning for Bidirectional Disease Contact Tracing on Real Human Mobility Data (https://arxiv.org/abs/2501.18531)
Comments:
          Accepted into International Workshop on Disaster Network Science for Building Resilient Communities (REINFORCE) held at the Advances in Social Networks Analysis and Mining conference

- **What's New**: 이번 연구는 전염병의 전파 경로를 추적하기 위해 대규모 접촉 네트워크를 활용하는 새로운 접근 방식을 제안합니다. 구체적으로, Infectious Path Centrality라는 새로운 네트워크 메트릭을 도입하여 접촉 추적을 통해 주요 전염 사건을 식별하는 그래프 학습 엣지 분류기를 알립니다. 특히, 양방향 접촉 추적을 탐색하여 전통적인 단방향 추적과 비교 분석합니다.

- **Technical Details**: 연구자는 Foursquare Mobility 데이터셋을 활용하여 사람 간의 접촉 네트워크를 구축합니다. 또한, 전염병 모델을 사용하여 전파 경로를 매핑하고 각 감염된 개인과 연결된 이웃에 대해 Infectious Path Centrality를 계산합니다. 이 메트릭은 접촉 추적 네트워크 내에서 감염된 개인의 중심성을 측정하여 전염 사건을 식별하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 연구 결과, 증상이 있는 인구 중 30%만 검사하더라도 양방향 접촉 추적이 전통적인 방식에 비해 감염성 유효 재생산율을 71%까지 낮출 수 있음을 보여줍니다. 이를 통해 전염병 발생 통제에 있어 양방향 접근 방식이 효과적임을 입증했습니다. F1 점수는 94%에 도달하여 전염 사건 식별 정확도를 나타냅니다.



### Optimal generalisation and learning transition in extensive-width shallow neural networks near interpolation (https://arxiv.org/abs/2501.18530)
Comments:
          8 pages + appendix, 3 figures

- **What's New**: 이 논문은 감독 학습의 교사-학생 모델을 고려하고, 폭이 크고 입력 차원과 비례하는 완전 학습된 2층 신경망의 Bayesian 최적 일반화 오류를 계산합니다. 특히 고유한 가중치 분포에 대한 분석을 통해 이 모델의 예측 성능을 명확히 합니다. 이 연구는 정확한 feature 학습에 대해 상대적으로 적은 양의 데이터를 사용하는 두 가지 상이한 위상 간의 비연속적 상전이를 밝혀내며, '보편'(universal) 및 '전문화'(specialisation) 위상을 구분합니다.

- **Technical Details**: 모델은 훈련 데이터 수 n이 입력 차원 d에 비례하는 경우에 주목하며, n = Θ(d²)와 k (폭) 및 d에 비례합니다. 이 연구는 무작위의 표준 가우시안 입력 벡터와 이를 통해 생성된 라벨에 대한 성능을 분석합니다. 예방 기능을 허용하는 활성화 함수와 비가우시안 이진 가중치에 초점을 맞추며, 이로 인해 학생 신경망의 가중치가 교사 네트워크와 정렬될 수 있는 고유한 상황을 탐구합니다.

- **Performance Highlights**: 학생 네트워크가 가중치에 따라 교사와 정렬하는 비율이 높을 경우, 가중치 분포에 따라 일반화 오류가 다르게 나타납니다. 낮은 비율에서는, 학생은 교사의 가중치와 비선형 조합만 일치시키며 일반화 오류는 느리게 감소합니다. 높은 비율에서는, 학생이 교사 가중치에 더 잘 정렬될 수 있어 오류가 더 빠르게 감소하게 됩니다. 이러한 두 위상의 차이는 일반화 오류를 결정하는 주요 변수와 연결되며, 이를 통해 신경망의 예측 성능을 잘 이해할 수 있는 기초가 됩니다.



### Beyond Prior Limits: Addressing Distribution Misalignment in Particle Filtering (https://arxiv.org/abs/2501.18501)
- **What's New**: 본 논문에서는 Prior Boundary Phenomenon (PBP)이라는 새로운 개념을 도입하여, 초기 prior 분포에 의해 입자 필터링의 제대로 된 상태 추정이 제한되는 현상을 분석합니다. 기존의 방법들이 동적 상황에서 그 효과가 제한적임을 지적하며, Diffusion-Enhanced Particle Filtering Framework (DEPF)를 제안합니다. 이 프레임워크는 입자 필터가 prior 경계를 넘어서 탐색할 수 있도록 세 가지 주요 혁신을 포함합니다.

- **Technical Details**: DEPF 프레임워크는 탐색적 입자를 통한 적응적 확산, 엔트로피 기반 규제, 그리고 동적 지원 확장을 위한 커널 기반의 섭동으로 구성됩니다. 탐색적 입자는 넓은 상태 공간에서 샘플링되어 prior 경계를 넘어 탐색할 수 있게 하며, 엔트로피 기반 규제는 무게 붕괴를 방지합니다. 커널 기반 섭동은 입자의 위치를 불확실하게 변동시켜, 목표 상태 공간을 더 잘 커버할 수 있도록 합니다.

- **Performance Highlights**: 이론적 분석 및 다양한 실험을 통해 DEPF 프레임워크의 효과성을 뒷받침하고, 높은 차원 및 비볼록 시나리오에서 성공률과 추정 정확도가 유의미하게 향상됨을 입증합니다. DEPF는 기존의 한계점을 극복하고 robust한 상태 추정을 가능하게 하여, 안전 및 복잡한 시나리오에서의 활용 가능성을 높입니다.



### Runway vs. Taxiway: Challenges in Automated Line Identification and Notation Approaches (https://arxiv.org/abs/2501.18494)
Comments:
          Accepted at SysCon 2025

- **What's New**: 이번 논문은 자율 항공 시스템에서의 활주로 및 택시웨이 표식의 정확한 레이블링 필요성을 강조합니다. 기존의 레이블링 알고리즘인 ALINA는 택시웨이 표식 식별에 성공하였으나 활주로 표식에서는 큰 도전에 직면했습니다. 이를 해결하기 위해 ALINA에 Convolutional Neural Network (CNN) 기반의 새로운 분류 단계를 통합하여, 환경 변화와 잘못된 분류에 대한 강인성을 높였습니다.

- **Technical Details**: 이 논문에서는 ALINA의 색상 임계값 조정과 관심 영역(ROI) 선택 개선을 통해 활주로에 특화된 레이블링 프로세스를 발전시켰습니다. 머신러닝 기술을 포함하는 CNN인 AssistNet을 도입하여 활주로 표식 탐지의 정확성과 강인성을 증대시키기 위한 방법론이 소개되었습니다. ALINA는 택시웨이 선 마킹을 탐지하기 위해 기하학적 조정과 색공간 변환을 통해 이진 픽셀 맵을 생성하는 프레임워크로, CIRCLEDAT 알고리즘을 사용하여 택시웨이 표식 픽셀을 식별합니다.

- **Performance Highlights**: 새롭게 제안된 접근 방식은 AssistTaxi 데이터셋에 대한 실험을 통해 활주로 표식 탐지에서 기존 ALINA의 성능을 신뢰성을 높이고 실제 사용 환경에도 적합한 결과를 도출했습니다. 초기 수정 작업을 통해 얻은 성과는 알루미늄의 큰 개선을 도출하지 못했으나, 활주로와 택시웨이를 구분하기 위한 분류 단계의 필요성을 강조하였습니다. 향후 연구는 동적 ROI 조정과 같은 방식을 통해 더욱 신뢰성 있는 활주로 표식 탐지 방법론을 제시할 것으로 기대됩니다.



### GuardReasoner: Towards Reasoning-based LLM Safeguards (https://arxiv.org/abs/2501.18492)
Comments:
          22 pages, 18 figures

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 안전성을 위한 새로운 안전장치인 GuardReasoner를 제안합니다. 기존의 guard models가 갖는 성능, 설명 가능성 및 일반화의 한계를 극복하는 것을 목표로 하며, reasoning SFT와 hard sample DPO 기법을 활용하여 모델의 추론 능력을 강화합니다.

- **Technical Details**: GuardReasoner의 훈련 과정은 두 단계로 나뉘며, 첫 번째 단계에서 GuardReasonerTrain 데이터셋을 구축하여 하루에 약 127K 샘플과 460K의 구체적인 추론 단계를 생성합니다. 두 번째 단계에서는 HS-DPO(하드 샘플 직접 선호 최적화)를 통해 모델이 잘못된 출력에 집중하고, 정확한 출력을 보완하는 방향으로 훈련을 진행하여 추론 능력을 한층 향상시킵니다.

- **Performance Highlights**: GuardReasoner는 13개의 벤치마크에서 실험을 통해 성능을 입증하였으며, GuardReasoner 8B 모델은 GPT-4o+CoT보다 5.74%, LLaMA Guard 3 8B보다 20.84% 더 높은 F1 점수를 기록했습니다. 또한, 이 모델은 다양한 크기(1B, 3B, 8B)로 공개되어 다른 연구자들이 접근할 수 있도록 했습니다.



### Resampling Filter Design for Multirate Neural Audio Effect Processing (https://arxiv.org/abs/2501.18470)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 신경망의 입력 및 출력에서 신호 리샘플링(signal resampling)을 사용하여 오디오 처리에서 샘플 레이트 독립성을 탐구하였습니다. 특히, 두 단계의 필터 디자인을 제안하는데, 이는 반대역 IIR 필터와 카이저 윈도우 FIR 필터를 연결하여 더 적은 연산량으로 이전의 모델 조정 방법과 유사하거나 더 좋은 결과를 얻을 수 있음을 보여주었습니다. 이러한 접근은 오디오 효과 모델에서 비선형 활성화 함수로 인한 앨리어싱(aliasing)을 줄이는 데 유용합니다.

- **Technical Details**: 연구팀은 RNN(Recurrent Neural Network) 아키텍처를 사용하여 오디오 신호를 처리하며, LSTM(Long Short-Term Memory) 셀을 기반으로 하였습니다. 입력 신호를 RNN 모델의 샘플 레이트에 맞춰 리샘플링하고, 처리 후 출력 신호를 원래 입력 샘플 레이트로 다시 리샘플링하는 방식으로 구현합니다. 이 과정에서 다양한 리샘플링 필터를 설계하고, 기존의 SRIRNN 샘플 레이트 독립 접근 방식과 비교합니다.

- **Performance Highlights**: 검토 결과, 제안된 두 단계 리샘플링 필터는 오디오 처리에서 높은 효율성을 보이며, 일반적인 오디오 샘플 레이트에서 1ms 미만의 지연으로 작동합니다. 이는 RNN의 정밀도 및 안정성을 개선하고 앨리어싱을 효과적으로 줄일 수 있는 가능성을 열어줍니다. 실험 결과에 따르면, 이 방법은 다양한 왜곡 효과 모델에서 기존보다 우수한 성능을 입증하였습니다.



### adabmDCA 2.0 -- a flexible but easy-to-use package for Direct Coupling Analysis (https://arxiv.org/abs/2501.18456)
- **What's New**: 이 논문에서는 Boltzmann machine learning을 기반으로 한 Direct Coupling Analysis (DCA)의 유연하면서도 사용하기 쉬운 구현방법을 제공합니다. 새로운 패키지인 adabmDCA 2.0은 C++, Julia, Python과 같은 여러 프로그래밍 언어로 제공되며, 다양한 아키텍처(단일 코어, 다중 코어 CPU, GPU)에서 사용할 수 있습니다. 여러 개의 학습 프로토콜을 지원하며, 단백질 및 RNA 서열 데이터를 위한 잔여-잔여 (residue-residue) 접촉 예측, 돌연변이 효과 예측 등의 다운스트림 작업을 직접 수행할 수 있는 기능도 포함되어 있습니다.

- **Technical Details**: Direct Coupling Analysis (DCA) 모델은 확률적 생성 모델로서, MSA (Multiple Sequence Alignment)를 입력으로 하여 서열 공간에서의 확률 분포를 유도합니다. 이 모델의 목적은 훈련 데이터를 기반으로 자연 서열과 유사한 서열에 높은 확률을 부여하고, 크게 벗어난 서열에 낮은 확률을 부여하는 것입니다. 이 패키지는 bmDCA, eaDCA 및 edDCA라는 세 가지 학습 루틴을 제공하며, 각 모델은 서로 다른 방식으로 coupling을 학습합니다.

- **Performance Highlights**: adabmDCA 2.0은 사용자 친화적인 인터페이스를 제공하며, 다양한 하드웨어 및 소프트웨어 제약을 고려한 세 가지 버전(C++, Julia, Python)으로 설치할 수 있습니다. 이 논문에서는 DCA 모델을 통해 단백질 서열의 기능 생성, 삼차 구조 내 접촉 예측, 돌연변이 효과 예측 등을 성공적으로 수행할 수 있는 경량화된 솔루션을 제시합니다. 또한, DCA 모델 학습 후 생성된 서열을 샘플링하고 분석하는 다양한 루틴이 포함되어 있습니다.



### DeepExtractor: Time-domain reconstruction of signals and glitches in gravitational wave data with deep learning (https://arxiv.org/abs/2501.18423)
Comments:
          22 pages, 16 figures, 4 tables

- **What's New**: 본 연구에서는 DeepExtractor라는 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 신호와 'glitches'를 재구성하여 간섭계를 초과하는 전력을 달성하도록 설계되었습니다. DeepExtractor는 신호 및 글리치 원본과 관계없이 간섭계 노이즈를 모델링하는 것을 목표로 합니다.

- **Technical Details**: DeepExtractor는 Gaussian 및 stationary 노이즈 분포를 가정하여 작동합니다. 이 프레임워크는 데이터의 노이즈 성분을 예측하고 빼내어 클린 재구성만 남기는 방식으로 작동합니다. 또한 U-Net 아키텍처를 사용하여 단기 푸리에 변환(STFT)에서 진폭과 위상 스펙트로그램을 동시 분석합니다.

- **Performance Highlights**: DeepExtractor는 시뮬레이션된 글리치에서 0.9%의 중앙 불일치를 기록하며, 여러 딥러닝 기준선을 능가합니다. 또한 BayesWave 알고리즘보다 글리치 회복 성능이 뛰어나며, CPU에서 글리치 샘플을 약 0.1초에 재구성하는 드라마틱한 처리 속도를 자랑합니다. 이는 LIGO 변형 데이터에서 효과적인 글리치 제거를 입증하는 데 성공했습니다.



### Consensus statement on the credibility assessment of ML predictors (https://arxiv.org/abs/2501.18415)
- **What's New**: 이 논문은 머신러닝(ML) 예측기를 의료 데이터에 통합하면서 생긴 변화에 대해 다루고 있습니다. 고위험 의료 결정에 활용되는 예측기의 신뢰성을 평가하기 위한 합의된 성명서가 제시됩니다. 저자들은 ML 예측기의 신뢰성을 평가하는데 꼭 필요한 12가지 핵심 진술을 정리하였습니다.

- **Technical Details**: ML 예측기를 평가하기 위해 인과 관계( causal knowledge ), 철저한 오차 추정( error quantification ) 및 편향( bias )에 대한 강건성( robustness )이 필요하다는 점을 강조합니다. 또한, ML 예측기와 생물 물리학 모델(biophysical models) 간의 비교를 통해 인과적 지식의 내재적 도전과제를 다루고 있습니다.

- **Performance Highlights**: 이 권고사항은 연구자, 개발자 및 규제 기관이 임상 및 생물 의학 분야에서 ML 예측기를 평가하고 배포하는 데 있어 신뢰성과 적용 가능성을 확보할 수 있도록 돕는 방향을 제시합니다. ML 예측기의 신뢰성을 높이기 위한 전략도 함께 제안됩니다.



### GBFRS: Robust Fuzzy Rough Sets via Granular-ball Computing (https://arxiv.org/abs/2501.18413)
- **What's New**: 본 논문에서는 다중 세분화(granularity) 구형(ball) 계산을 퍼지 러프 집합(fuzzy rough set) 이론에 통합하는 방법을 제안합니다. 기존 퍼지 러프 집합 모델이 가장 세밀한 세분화에 기반하고 있는 반면, 이 연구는 다양한 크기의 구형을 사용하여 샘플 공간을 적응적으로 표현하고 덮을 수 있도록 합니다. 이러한 접근 방식은 모델의 강인성을 강화하는 데 중요한 역할을 합니다.

- **Technical Details**: 연구에서는 구형 퍼지 러프 집합(granular-ball fuzzy rough set, GBFRS) 프레임워크를 제안하며, 구형의 상하 근사에 대해 엄격하게 정의하고 관련 정리에 대한 공식을 제시합니다. feature selection 방법의 알고리즘 프로세스를 설계하였고, 이를 통해 GBFRS 프레임워크의 이론적 타당성을 확보하였습니다. 또한, GBFRS 방식이 UCI 데이터셋에서 기존 퍼지 러프 집합 방법보다 효율적임을 입증했습니다.

- **Performance Highlights**: 실험 결과는 제안된 GBFRS 방법이 기존의 방법들보다 뛰어난 성능을 보임을 나타냅니다. 특히, 샘플의 노이즈에 대한 강인성(reliability)을 높이며, 그에 따른 feature selection 방식에서도 개선된 결과를 보여줍니다. 이는 고차원 복합 데이터 세트의 분석에서 매우 중요한 발전을 의미합니다.



### Implicit Riemannian Optimism with Applications to Min-Max Problems (https://arxiv.org/abs/2501.18381)
- **What's New**: 본 논문에서는 Hadamard 다양체(Hadamard manifolds)를 위한 비정확 암묵적(implicit) 업데이트에 기반한 리만 최적화 온라인 학습 알고리즘(Riemannian optimistic online learning algorithm)을 소개합니다. 기존 연구와 달리, 제안된 방법은 다양체 내의 제약 조건(in-manifold constraints)을 처리할 수 있으며, 기하학적 상수(geometric constants)와 무관하게 유클리드(Euclidean) 환경에서 알려진 최고 수준의 후회 경계(regret bounds)를 달성할 수 있습니다.

- **Technical Details**: 이 알고리즘은 Hadamard 다양체에서의 g-볼록(g-convex) 및 g-오목(g-concave) 부드러운(min-max) 문제를 해결하는 방법을 개발하는 데에도 기여합니다. 특히, 제안된 방법 중 하나는 유클리드 문제의 하한(lower bound)에 대한 그래디언트 오라클 복잡도(gradient oracle complexity)를 거의 일치시키는 성과를 보였습니다.

- **Performance Highlights**: 이 연구의 성과는 Hadamard 다양체에서의 최적화 문제 해결을 위한 효율적인 알고리즘 개발에 큰 기여를 하며, 다양한 제약 조건을 수용할 수 있는 능력을 보여줍니다. 이런 접근은 실제 응용 프로그램에서의 성능을 개선시킬 수 있는 가능성을 제시합니다.



### MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding (https://arxiv.org/abs/2501.18362)
- **What's New**: 이번 논문에서는 MedXpertQA라는 새로운 벤치마크를 소개합니다. MedXpertQA는 전문가 수준의 의료 지식과 고급 추론을 평가하기 위해 설계된 도전적이고 포괄적인 기준을 제시합니다. 이 벤치마크는 17개 전문 분야와 11개 신체 시스템을 포함하며, 텍스트 평가를 위한 Text와 다중 모드 평가를 위한 MM의 두 가지 하위 집합으로 구성되어 있습니다.

- **Technical Details**: MM 하위 집합은 환자 기록 및 검사 결과를 포함한 다양한 이미지와 풍부한 임상 정보를 가진 전문가 수준의 시험 문제를 도입하여, 간단한 QA 쌍을 생성하는 기존의 의료 다중 모드 벤치마크와 차별화됩니다. MedXpertQA는 기존의 벤치마크인 MedQA와 같은 어려움이 부족한 문제를 해결하기 위해 엄격한 필터링과 증강을 적용하고, 임상 관련성과 포괄성을 높이기 위해 전문 분야 위원회 질문을 포함합니다.

- **Performance Highlights**: 우리는 MedXpertQA에서 16개의 주요 모델을 평가하였습니다. 의료는 실질적인 의사 결정과 깊은 연관이 있기 때문에, 수학 및 코드 외의 추론 능력을 평가하는 데 적합한 풍부하고 대표적인 환경을 제공합니다. 이를 위해 o1과 유사한 모델의 평가를 용이하게 하는 추론 지향 하위 집합을 개발했습니다.



### Contextual Online Decision Making with Infinite-Dimensional Functional Regression (https://arxiv.org/abs/2501.18359)
Comments:
          30 pages

- **What's New**: 본 논문은 무한 차원 기능 회귀(infinite-dimensional functional regression)를 기반으로 한 일반화된 온라인 의사결정 설정을 다루고 있습니다. 기술적 문제 해결을 위해 여러 종류의 통계적 재분포를 모델링하고, 불확실한 결과를 관리하기 위한 효율적인 알고리즘 프레임워크를 제안합니다. 특히, 각 데이터 포인트를 컨텍스트 의존적인 CDF 기저 함수(combinator of context-dependent CDF basis functions)의 조합으로 모델링하여 무한 차원 회귀 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 알고리즘은 무한 차원에서 작동하며, 현재의 제한된 상수 차원 접근법을 넘어서기 위한 성능도 제공합니다. 이 알고리즘은 선형 컨텍스트 밴딧 알고리즘이 겪는 무한 후회(infinite regret) 문제를 해결하기 위해 고안되었습니다. 특히, 고유값(eigenvalue) 수열의 감소 속도를 기반으로 하는 분석을 통해 회귀 오류 및 효용 후회(rate of utility regret)를 제어할 수 있음을 보여주었습니다.

- **Performance Highlights**: 이 연구에서는 고유값 수열의 다항 감소가 있는 경우, 효용 후회를 제한하는 수식을 제시했습니다. 효능 측면에서, 알고리즘은 𝒪(log⁡T) 호출로 기능 회귀를 수행할 수 있으며, 이는 대규모 심층 학습 모델에 대한 적용 가능성을 높입니다. 최종적으로, 제안된 기법은 모든 문제에 대해 보편적으로 서브선형 후회를 보장하며, 이는 기존의 연구 결과와 비교하여 획기적인 기여로 평가됩니다.



### Unfaithful Probability Distributions in Binary Triple of Causality Directed Acyclic Graph (https://arxiv.org/abs/2501.18337)
- **What's New**: 이번 논문에서는 세 개의 정점으로 이루어진 이항 인과 구조 다이렉티드 애시클릭 그래프(DAG)에서 발생할 수 있는 비순응성(Non-faithfulness) 확률 분포의 여러 예시를 제시하며, 이는 Robins et al. (2003)에서 설명한 인과 DAG에 대한 비순응성을 증명합니다. 또한, 구간 독립(conditional independence) 및 상태 독립(multiple independence)이 함께 이루어지는 비순응 확률 분포의 일반적인 가족을 설명하고 있습니다.

- **Technical Details**: 이 논문에서는 세 랜덤 변수 (X, Y, Z)에 대해 11개의 대표적인 DAG 구조가 문헌에 소개되고, 이 구조들이 비순응성에 대해 어떻게 영향을 미치는지에 대한 논의가 이루어집니다. 특히, Robins et al. (2003)와 Sadeghi (2017)의 연구를 기반으로 하여 3정점 이항 다각형(DAG)에서 비순응성 확률 분포의 예가 만들어집니다. 또한, 각 확률 분포는 Markov 호환성(Markov compatibility)을 만족하는지를 기반으로 조건부 독립 관계를 분석합니다.

- **Performance Highlights**: 이 논문에서 제시된 예시들은 비순응 확률 분포가 생성되는 다양한 구조를 효과적으로 나타내며, PC 알고리즘(principal component algorithm)과 같은 그래픽 방법을 통해 인과 추론(causal inference) 과정에서 나타나는 문제점을 철저히 탐구합니다. 후속 논문에서 다룰 더 많은 다각적인 예시와 결과들은 이 분야에서의 인과 구조를 더욱 명확히 하는 데 기여할 것으로 기대됩니다.



### Random Feature Representation Boosting (https://arxiv.org/abs/2501.18283)
- **What's New**: 이 논문에서는 Random Feature Representation Boosting (RFRBoost)라는 새로운 방법을 소개합니다. RFRBoost는 부스팅 이론(boosting theory)을 이용하여 깊은 잔차 랜덤 피처 신경망(Deep Residual Random Feature Neural Networks, RFNNs)을 구축하는 방법으로, 이로써 성능 향상 및 RFNNs의 고립된 최적화 특성을 유지할 수 있습니다. 신뢰할 수 있는 이론적 보장을 제공하며, 전통적인 RFNNs 및 MLP ResNets보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: RFRBoost는 각 레이어에서 랜덤 피처를 사용하여 네트워크 표현의 함수적 기울기를 학습합니다. MSE 손실함수에 대해 우리는 랜덤 피처를 사용하여 탐욕적 레이어별 부스팅의 폐쇄형 솔루션을 도출했습니다. 일반적인 손실 함수의 경우, 랜덤 피처 잔차 블록을 피팅하는 과정은 제곱 제약 최소 제곱 문제를 해결하는 것과 동등합니다.

- **Performance Highlights**: 91개의 테이블 데이터셋에 대한 회귀 및 분류 실험을 통해 RFRBoost가 전통적인 RFNNs와 end-to-end 훈련된 MLP ResNets보다 상당한 성능 향상을 보임을 입증했습니다. RFRBoost는 계산적 장점이 크고, 부스팅 이론에 기초한 이론적 보장도 제공하여 다양한 작업에서 우수한 결과를 나타냅니다.



### Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models (https://arxiv.org/abs/2501.18280)
- **What's New**: 최근 큰 언어 모델(LLMs)의 보안 이슈가 많은 주목을 받으면서, 해로운 출력을 방지하기 위한 다양한 방어 메커니즘이 개발되고 있습니다. 본 논문에서는 텍스트 임베딩 모델을 기반으로 한 안전장치가 이러한 방어의 기초가 됨을 발견하였습니다. 특히, 텍스트 임베딩 모델의 출력 분포가 큰 평균을 가지며 상당히 편향되어 있다는 관찰에서 출발해 보편적인 마법 단어(universal magic words)를 검색하는 새로운 효율적인 방법을 제안합니다.

- **Technical Details**: 보편적인 마법 단어는 텍스트 뒤에 붙여져 어떤 텍스트의 임베딩을 편향된 방향으로 이동시키고, 이는 두 텍스트 쌍의 유사성을 조작해 안전장치를 오도할 수 있습니다. 본 연구는 세 가지 방법을 사용하여 이러한 마법 단어를 발견하는 접근 방식을 설명합니다: 1) Brute-force search는 기준선으로 사용되며, 2) Black box 방법은 비대칭 방향과 유사한 텍스트 임베딩 단어를 찾아내고, 3) White box 방법은 임베딩을 편향 방향에 가깝게 만들 마법 단어 접미사를 찾습니다.

- **Performance Highlights**: 실험 결과, 세 가지 방법 모두 최적의 마법 단어를 찾는 것으로 나타났지만, Method 2와 Method 3은 Method 1보다 훨씬 더 효율적이었습니다. 특히, Method 3은 다중 토큰 마법 단어를 검색할 수 있어 유용성이 높습니다. 이를 통해 우리는 LLM 보안 시스템에서의 안전장치가 해로운 콘텐츠를 탐지하는 데 실패하는 취약점을 지적하며, 이러한 공격에 대한 방어 메커니즘도 제안했습니다.



### Statistical multi-metric evaluation and visualization of LLM system predictive performanc (https://arxiv.org/abs/2501.18243)
- **What's New**: 이번 논문은 Generative 또는 Discriminative Large Language Model (LLM) 기반 시스템의 성능 평가 방법에 대한 새로운 프레임워크를 제시합니다. 이 프레임워크는 여러 데이터셋과 평가 기준을 아우르며, 통계적 검증(testing)을 통해 성과의 유의미성(significance)을 확인하는 과정을 자동으로 수행합니다. 여러 하이퍼파라미터를 가진 시스템 구성의 성능을 비교하고 분석함으로써, 사용자가 보다 효율적으로 LLM 성능을 평가할 수 있게 합니다.

- **Technical Details**: 프레임워크는 여러 데이터셋과 평가 기준을 통합하여 시스템 간의 통계적 비교를 가능하게 합니다. 특히, 이는 쌍 또는 비극형 관측 데이터셋에 대한 통계적 결정을 지원하며, 각 시스템의 통계적 비교 결과를 시각적으로 표현할 수 있는 유틸리티를 포함하고 있습니다. 사용자는 복잡한 통계적 과정을 걱정할 필요 없이, 필요한 데이터를 입력하기만 하면 됩니다.

- **Performance Highlights**: 이 연구는 CrossCodeEval이라는 다국어 코드 보완 벤치마크에서 여러 최신 LLM의 성능을 비교하는 데 성공적으로 적용되었습니다. 기존의 Leaderboard 평가 방식에서는 종종 유의미한 차이를 고려하지 않았으나, 본 프레임워크는 통계적 검정을 통해 그러한 차이를 명확히 할 수 있는 장점을 제공합니다. 또한, 데이터셋 간의 통계적 결과 집계 및 시각화를 통해 최적의 LLM 선택을 지원합니다.



### Revisiting $\Psi$DONet: microlocally inspired filters for incomplete-data tomographic reconstructions (https://arxiv.org/abs/2501.18219)
- **What's New**: 이 논문에서는 $	ext{ΨDONet}$이라는 감독 학습 접근 방법을 재조명하며, 이론적 분석을 위한 더 깊은 미세 지역 해석을 제공합니다. 이 연구는 제한된 각도 토모그래피(sparse-angle tomography) 경우로의 연구를 확장하고, 불완전한 데이터에서 생성된 스트릭 아티팩트의 특성을 고안한 특수 필터를 고려하여 원래의 $	ext{ΨDONet}$의 구현을 세련되게 합니다. 이를 통해 학습 가능한 파라미터 수를 줄이면서도 제한된 각도 데이터로부터의 재구성을 위한 동일한 품질을 유지하거나 다소 향상시키는 것을 가능하게 합니다.

- **Technical Details**: 이 논문은 원래의 $	ext{ΨDONet}$ 접근 방식이 보는 각도(basin) 데이터를 사용하여 불완전한 시노그램으로부터 엣지 및 불연속성을 복구할 수 있도록 설계되었다는 점을 강조합니다. 특히, 미세 지역 분석(microlocal analysis)을 활용하여 그런 정보가 어떻게 촉진되는지 설명하고, 단순한 이상점(singularities)에 대한 보다 명확한 접근법을 제시합니다. 여기에 따라 $	ext{ΨDONet}$은 기존의 접근 방법과는 달리 데이터 기반 학습을 통해 스트릭 아티팩트를 완화하는 역할을 하며, 시노그램에서 보이지 않는 정보를 예측하고 상실되는 정보를 추측하는 데 의존하지 않습니다.

- **Performance Highlights**: 이 실험에서는 제한된 각도 및 희소 각도 토모그래피에 대한 새로운 미세 지역 필터의 잠재력을 수치 실험을 통해 논의합니다. 가장 주목할 만한 점은, 이러한 필터가 스트릭 아티팩트를 줄이는데 성공할 뿐만 아니라, 완전히 사라지게 할 수 있다는 점입니다. 전반적으로, 이 연구는 새로운 아키텍처의 구현을 통해 제한된 각도 데이터로부터의 재구성 품질을 보존하면서도 효율성을 높일 수 있음을 시사합니다.



### Machine Learning Fairness for Depression Detection using EEG Data (https://arxiv.org/abs/2501.18192)
Comments:
          To appear as part of the International Symposium on Biomedical Imaging (ISBI) 2025 proceedings

- **What's New**: 이 논문은 EEG (electroencephalogram) 데이터를 활용하여 우울증 탐지에서 머신러닝의 공정성을 평가하기 위한 최초의 시도를 제시합니다. 여러 딥러닝 아키텍처, 즉 CNN (Convolutional Neural Networks), LSTM (Long Short-Term Memory) 네트워크, GRU (Gated Recurrent Unit) 네트워크를 사용하여 실험을 수행하였습니다. 세 가지 EEG 데이터셋인 Mumtaz, MODMA, Rest를 대상으로 공정성을 위한 다양한 평가 기준을 사용했습니다.

- **Technical Details**: 다양한 편향 완화 (bias mitigation) 전략을 사전 처리, 처리 중, 사후 처리 단계에서 적용하여 그 효과성을 평가하였습니다. 실험 결과, 기존 EEG 데이터셋과 우울증 탐지를 위한 알고리즘에서 편향이 존재함을 발견하였고, 여러 편향 완화 방법이 서로 다른 공정성 기준에 따라 편향을 해결하는 방식이 다름을 보여주었습니다. 이러한 접근은 EEG 데이터에서 머신러닝의 공정성을 높이는 데 기여할 수 있습니다.

- **Performance Highlights**: 우울증 탐지에 대한 기존 알고리즘의 편향 문제를 강조하며, 다양한 편향 완화 기법이 필요하다는 것을 입증하였습니다. 연구 결과는 각기 다른 레벨에서 편향 문제를 다루는 다양한 방법들을 제시합니다. 따라서 이 연구는 EEG 데이터를 기반으로 한 우울증 탐지의 공정성과 신뢰성을 향상시키기 위한 중요한 기초 자료가 될 것입니다.



### Decentralized Projection-free Online Upper-Linearizable Optimization with Applications to DR-Submodular Optimization (https://arxiv.org/abs/2501.18183)
- **What's New**: 이번 논문에서는 분산형 projection-free 최적화를 위한 새로운 프레임워크를 소개하며, 기존의 DR-submodular 함수 최적화를 보다 넓은 범위의 상부 선형화 가능 함수에까지 확장합니다. 이 접근법은 분산형 최적화 기법과 상부 선형화 함수 프레임워크의 유연성을 활용하여 전통적인 최적화 문제를 일반화합니다. 또한, 이 방법은 일반적인 볼록 제약조건 하에서의 단조 upward-concave 최적화 및 비단조 upward-concave 최적화를 위한 첫 번째 결과를 제공합니다.

- **Technical Details**: 특히, 이 연구에서는 각기 다른 피드백 유형 하에서 상부 선형화 가능 함수 최적화를 위한 후회 경계를 수립합니다. 여기서 통신의 복잡성은 O(T^θ)로, 선형 최적화 오라클 호출 수는 O(T^{2θ})로 제한됩니다. 이 접근법은 저자들이 제안한 메타 알고리즘을 기반으로 하여 여러 피드백 유형에 대한 새로운 결과를 도출합니다.

- **Performance Highlights**: 제안된 알고리즘은 후회 경계 O(T^{1-θ/2})를 달성하며, 실제 적용에 있어서 통신 복잡성 및 선형 최적화 오라클 호출 수와의 균형을 보여줍니다. 특히 θ=1의 경우 최적의 후회 O(T)와 적은 통신 복잡성을 동시에 만족시키며, Zhang et al. (2023a)의 결과와 비교할 때 모든 지표에서 개선된 성능을 보입니다. 이로 인해 최소화 문제의 효율적인 해결 방안이 제시되며, 전문가들의 이목이 집중되고 있습니다.



### Estimating Multi-chirp Parameters using Curvature-guided Langevin Monte Carlo (https://arxiv.org/abs/2501.18178)
- **What's New**: 이 논문은 잡음이 포함된 다중 차원의 chirp 혼합 신호로부터 chirp 매개변수를 추정하는 문제를 다루고 있습니다. 기존 연구들에서 다양한 방법이 제시되었지만, 고차 다항식의 경우에는 여전히 해결 난제가 있습니다. 저자들은 이 문제를 비볼록 최적화 문제로 형성하고, 목표 함수의 평균 곡률을 활용하여 안정적으로 최소값을 찾는 수정된 Langevin Monte Carlo (LMC) 샘플러를 제안합니다.

- **Technical Details**: 이 연구에서는 N𝑁N-샘플 이산 시간 다중 구성 요소 chirp 신호 모델을 수립하고, 주어진 파라미터를 추정하기 위해 Gaussian 노이즈가 포함된 Nc 붕괴(chirp)들의 혼합 모델을 사용합니다. 목표는 아날로그 방정식(1) 및 (3)를 통해 주어진 다항식의 Pᵗʰ 차수와 Aᵗʰ 차수를 사용하여 신호의 파라미터를 추정하는 것입니다. 저자들은 곡률 기반의 Gaussian smoothing 기법을 적용하여 군집 최소값에 대한 수렴성을 개선하였습니다.

- **Performance Highlights**: 저자들이 제안한 Curvature-guided LMC (CG-LMC) 알고리즘은 낮은 신호 대 잡음 비율(SNR) 환경에서도 높은 신뢰성과 낮은 추정 오차를 보였습니다. 실험 결과, CG-LMC는 기존의 Langevin Monte Carlo (LMC) 및 Noise-Annealed Langevin Monte Carlo (NA-LMC) 알고리즘보다 우수한 성능을 보여주었습니다. 이 연구는 다양한 실제 응용, 특히 전자 청진기를 통한 심박 소리 분석 및 진단에 기여할 가능성이 높습니다.



### Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study (https://arxiv.org/abs/2501.18158)
- **What's New**: 이 논문에서는 비트코인 네트워크 내에서 실제 암호화폐 거래 그래프를 LLMs로 분석하는 새로운 접근법을 소개합니다. 저자들은 LLM4TG라는 새로운 그래프 표현 형식과 CETraS라는 샘플링 알고리즘을 도입하여 LLMs가 거래 행동을 해석하는 능력을 평가합니다. 이는 기존의 불투명한 블랙박스 모델의 한계를 극복하고 거래 패턴을 효과적으로 캡처할 수 있도록 돕습니다.

- **Technical Details**: 이 연구는 LLM의 거래 그래프 분석 능력을 세 가지 수준으로 평가합니다: Level 1에서는 기본 메트릭을 측정하고, Level 2에서는 특성 개요를 생성하며, Level 3에서는 문맥적 해석을 수행합니다. LLM4TG는 인간이 읽을 수 있는 텍스트 기반의 그래프 표현 형식으로, 노드와 엣지의 정보를 최적화하여 LLMs가 데이터 처리 시 시각적 잡음을 최소화하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, LLMs는 기본 메트릭에서 뛰어난 성능을 보였으며, 특히 거래 행동의 문맥적 해석에서 유용한 설명을 제공할 수 있음을 입증했습니다. 이러한 성능은 레이블이 제한된 데이터에서도 여전히 유의미한 인사이트를 생성할 수 있음을 보여줍니다.



### HyperZero: A Customized End-to-End Auto-Tuning System for Recommendation with Hourly Feedback (https://arxiv.org/abs/2501.18126)
- **What's New**: 이 논문에서는 추천 시스템의 두 번째 단계인 가치 모델의 최적화를 위한 자동 조정 기술인 HyperZero를 소개합니다. 기존 자동 조정 방법들은 몇 주 또는 몇 달의 긴 시간이 소요되지만, HyperZero는 2-3일 이내에 적절한 모델을 식별하는 것을 목표로 합니다. 최신 추천 시스템의 고유한 문제를 효과적으로 해결하고, 넓은 조정 작업에도 확장할 수 있는 가능성을 가지고 있습니다.

- **Technical Details**: 이 시스템은 두 주요 모듈인 추정기 모듈과 최적화기 모듈로 구성됩니다. 추정기 모듈은 현재 하이퍼파라미터에 대한 사용자 피드백을 기반으로 목표를 추정하고, 최적화기 모듈은 이 예측을 활용하여 하이퍼파라미터를 업데이트합니다. HyperZero는 제한된 최적화 문제를 풀기 위해 Gaussian 프로세스 추정기와 Thompson 샘플링을 결합하여 여러 목표를 동시에 고려할 수 있도록 합니다.

- **Performance Highlights**: HyperZero는 시간 단위의 피드백을 활용하여 주기적인 조정 사이클을 주에서 일로 단축할 수 있는 엔드투엔드(auto-tuning) 시스템을 개발했습니다. 비독립적으로 시간 신호를 처리하고, 비동기 병렬 샘플링을 구현하여 효율성을 증가시키는데 중점을 두었습니다. 이를 통해 기존의 추천 시스템에서 발생하는 다양한 문제를 해결할 수 있는 잠재력을 가지고 있습니다.



### Optimal Survey Design for Private Mean Estimation (https://arxiv.org/abs/2501.18121)
- **What's New**: 이 논문은 차별적 프라이버시(differential privacy, DP) 하에 일반적인 평균 추정을 위한 최초의 프라이버시 인식을 고려한 층화 샘플링(stratified sampling) 방안을 제안합니다. DLap 및 TuLap 메커니즘을 적용하여 정량적 변동성을 최소화하는 최적화된 설계를 개발하였습니다. 기존의 샘플링 전략은 DP의 영향을 무시할 경우 변동성이 크게 증가할 수 있으므로 이 문제를 해결하기 위한 새로운 접근법이 필요하다는 점을 강조합니다.

- **Technical Details**: 제안된 프레임워크는 혼합 로컬/중앙 차별적 프라이버시 체제 하에서 층화 샘플링 계획을 설계하는 새로운 접근 방식을 제공합니다. 이 과정에서 최적의 서브샘플 크기를 결정하기 위해 최적화 문제를 형성하고, 고급 볼록성(strong convexity) 속성을 활용하여 효율적인 정수 최적 설계를 찾는 알고리즘을 제안합니다. 또한, Laplace, Discrete Laplace 및 Truncated-Uniform-Laplace과 같은 세 가지 일반적인 DP 메커니즘 하에서의 추정 변동성 목표를 다룹니다.

- **Performance Highlights**: 실험 결과, 기존의 단순한 설계 방식에 비해 제안된 최적화된 설계 방식이 변동성을 현저히 낮추는 것으로 나타났습니다. 또한, 다양한 샘플 그룹 수에 따라 우리의 알고리즘이 여전히 우수한 성능을 발휘함을 보여줍니다. 이러한 발견은 차별적 프라이버시를 고려한 샘플링 설계의 중요성을 부각시키며, 연구자들이 향후 데이터 수집 및 분석 과정에서 DP를 효과적으로 통합할 수 있는 기반을 마련합니다.



### DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification (https://arxiv.org/abs/2501.18116)
Comments:
          27 pages, 8 figures

- **What's New**: 이번 연구에서는 기존의 기능 데이터 분석(FDA) 방법의 비효율성을 해결하기 위해 DeepFRC라는 새로운 심층 학습 프레임워크를 제안합니다. 이 프레임워크는 기능 등록(functional registration)과 분류(classification) 문제를 하나의 통합 모델로 결합하여 동시에 처리합니다. 또한, 시간 왜곡 함수(time warping functions)를 학습하는 정렬 모듈(alignment module)과 정렬된 데이터에 대한 차원 축소를 위한 학습 가능한 기저 표현 모듈(learnable basis representation module)을 통합하여 성능을 향상시킵니다.

- **Technical Details**: DeepFRC는 전반적인 프로세스를 자동화하여 단계적으로 처리하는 대신, 등록과 분류를 simultaneously(동시) 수행합니다. 모델링에서 발생하는 misalignment(불일치)와 generalization error(일반화 오차)에 대한 이론적 분석을 통해 낮은 misalignment을 보장합니다. 이 모델은 elastic functional registration을 사용하여 시간 왜곡을 구현하며, 이를 통해 데이터의 구조적 변화를 효과적으로 반영합니다.

- **Performance Highlights**: 실험 결과, DeepFRC는 다양한 실제 데이터 세트에 대해 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다. 특히, 복잡한 등록 문제를 다루는 데 뛰어난 능력을 갖추고 있으며, 모델이 대칭적이고 누락된 데이터에 대해서도 견고함을 유지합니다. 이 연구는 실제 응용 분야에서의 DeepFRC의 적용 가능성을 시사합니다.



### A spectral clustering-type algorithm for the consistent estimation of the Hurst distribution in moderately high dimensions (https://arxiv.org/abs/2501.18115)
- **What's New**: 본 논문에서는 고차원 프랙탈 시스템의 Hurst 분포를 통계적으로 식별하는 알고리즘을 개발하였습니다. 특히, 이 알고리즘은 파동렛 랜덤 행렬(wavelet random matrices)과 수정된 스펙트럴 클러스터링(modified spectral clustering)을 기반으로 하며, 클러스터링 정밀도 하이퍼파라미터의 값 선택을 위한 모델 선택 단계를 포함합니다.

- **Technical Details**: 제안된 알고리즘은 차원, 샘플 크기 및 스케일이 무한대로 증가하는 중간 고차원 영역에서 Hurst 분포를 일관되게 추정함을 보여주었습니다. 클러스터링 과정에서 적용되는 몬테 카를로 시뮬레이션(Monte Carlo simulations)은 이 방법론이 실제 샘플 크기에 대해 효과적이며 혼합-가우시안 모델링을 기반으로 한 다른 클러스터링 방법보다 우수한 성능을 발휘함을 입증합니다.

- **Performance Highlights**: 실제 세계의 거시경제 시간 시계열에 대한 분석에 알고리즘을 적용하여 공적분(cointegration)의 증거를 발견하였습니다. 이는 고차원 프랙탈 시스템의 통계적 특성을 이해하는 데 중요한 기초 자료를 제공하며, 경제 데이터 분석에 있어서 새로운 방향성을 제시합니다.



### DCatalyst: A Unified Accelerated Framework for Decentralized Optimization (https://arxiv.org/abs/2501.18114)
- **What's New**: 이 논문은 중앙 서버 없이 그래프 형태로 모델링된 에이전트 네트워크에서 탈중앙화된 최적화를 연구합니다. DCatalyst라는 새로운 블랙박스 프레임워크를 소개하며, 이는 탈중앙화된 최적화 알고리즘에 Nesterov 가속을 통합합니다. DCatalyst는 기존의 탈중앙화 알고리즘에 가속 기능을 제공하여 문제의 선택성과 효율성을 향상시킵니다. 이 접근 방식은 과거에는 가속 솔루션 방법이 부족했던 문제 클래스에서도 널리 적용될 수 있습니다.

- **Technical Details**: DCatalyst는 부정확한 모멘텀 가속화 접근 방식을 기반으로 하며, 탈중앙화 알고리즘과의 통합을 통해 최적의 통신 및 계산 복잡성을 달성합니다. 특히, 이 프레임워크는 탈중앙화 환경에서 복합 손실을 최소화하기 위한 새로운 '불확실 추정 시퀀스(inexact estimating sequences)'를 소개합니다. 이 방법은 기존 모델에서 해결되지 않은 합의 오류 및 에이전트의 서브 문제의 불확실한 솔루션 문제를 효과적으로 처리할 수 있습니다. 이로 인해 DCatalyst는 다양한 문제 인스턴스에서 복잡성을 최적화합니다.

- **Performance Highlights**: DCatalyst는 다양한 탈중앙화 알고리즘에 적용될 수 있으며, 특히 통신 복잡성을 최적화할 수 있는 능력이 돋보입니다. 실험을 통해 이러한 접근 방식이 에이전트 간의 협력 문제를 해결하는 데 효과적임을 보여줍니다. 또한, 기존의 기술들이 대개 특정 문제에 한정되는 반면, DCatalyst는 다양한 문제 유형에 대한 적용 가능성을 높여줍니다. 결과적으로, 이 연구는 탈중앙화 알고리즘의 새로운 연구 방향을 제시합니다.



### Investigating an Intelligent System to Monitor \& Explain Abnormal Activity Patterns of Older Adults (https://arxiv.org/abs/2501.18108)
- **What's New**: 이 연구는 고령자의 독립적인 거주를 지원하기 위한 기술의 설계를 위해 가족 돌봄 제공자와의 포커스 그룹 세션을 실시하였습니다. 이와 함께, 고신뢰 프로토타입을 개발하고, 전문가와 고령자를 대상으로 시스템 기능에 대한 질적 연구를 진행했습니다. 새로운 시스템은 무선 모션 센서를 사용하여 고령자의 비정상적인 활동 패턴을 모니터링하고, 인터랙티브 대화 응답을 통해 돌봄 제공자와 고령자가 정보 공유를 주도할 수 있도록 지원합니다.

- **Technical Details**: 이 시스템은 일상 생활 활동(Activities of Daily Living, ADL)을 모니터링하고, 고령자의 비정상적인 행동 패턴을 감지하여 돌봄 제공자에게 위험 신호를 전달합니다. 이를 위해 다양한 센서와 머신 러닝 모델을 적용하였으며, 시스템의 디자인을 위한 가족 돌봄 제공자의 피드백도 반영하였습니다. 질적 연구를 통해 두 그룹의 피드백을 수집했으며, 시스템의 성능과 사용자 인터페이스에 대해 의견을 모았습니다.

- **Performance Highlights**: 연구 결과, 고령자와 전문가 모두 시스템의 인터랙티브 대화 기능이 개인화된 서비스 제공과 정보 공유의 효율성을 높일 수 있다고 평가했습니다. 그들은 시스템이 보다 빠르고 주의 깊은 돌봄 서비스를 제공할 수 있으며, 고령자가 자신의 상태를 통제할 수 있는 점에서 긍정적인 의견을 나타냈습니다. 하지만 시스템의 성능과 상호작용 방식에 대한 몇 가지 제한 사항에도 불구하고, 이러한 기술이 더 많은 실용성을 갖출 수 있도록 다양한 고려사항을 제시했습니다.



### Disentangling Safe and Unsafe Corruptions via Anisotropy and Locality (https://arxiv.org/abs/2501.18098)
- **What's New**: 본 논문은 	exttt{Projected Displacement} (PD)라는 새로운 위협 모델을 제안하여 기존의 이소트로픽(isotropic) 및 글로벌(global) 위협 모델 이상의 강건성을 연구합니다. 기존의 대부분 위협 모델은 단순히 $	exttt{l_p}$ norm을 사용하여 위협을 정의했으나, 이는 컴퓨터 비전에서의 흔한 왜곡 현상인 블러, 압축 및 가림 등을 잘 포착하지 못합니다. PD 모델은 입력 공간에서 불안전한 방향(unsafe directions)을 정의하고, 이를 통해 입력에 따른 위협을 측정함으로써 비안소트로픽(anisotropic) 및 지역적(local) 특성을 보여줍니다.

- **Technical Details**: PD 위협 모델은 단순히 작은 크기의 변형이 아닌, 특정 입력을 위한 안전한 방향과 불안전한 방향을 고려하여 위협을 평가합니다. 여기서 안전한 변형(safe perturbations)은 실제 레이블을 보존하는 반면, 불안전한 변형(unsafe perturbations)은 레이블을 변경하는 방식으로 정의됩니다. 이러한 평가 방식은 임의의 분류 작업에 대해 사전 훈련(pre-training)이나 미세 조정(fine-tuning) 없이도 쉽게 계산될 수 있습니다.

- **Performance Highlights**: 실험 결과, Imagenet-1k 데이터셋을 통해 PD 위협 모델이 기존 모델과 다르게 작동하여, 실제 레이블을 유지하는 고유의 안전한 변형 집합을 포함하고 있음을 보여줍니다. 이 모델은 또한 이미지의 특정 영역에 대한 민감도나 개념 계층과 같은 추가적인 작업 주석을 쉽게 통합할 수 있어 적용 가능성이 높습니다. PD 모델은 기계 학습 시스템의 안전성 평가에 있어 유연하고 작업 중심의 위협 사양을 제공합니다.



### LLMs can see and hear without any training (https://arxiv.org/abs/2501.18096)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 MILS(Multimodal Iterative LLM Solver)를 소개합니다. 이 방법은 훈련이 필요 없는 간단한 접근 방식으로, 기존의 LLM에 멀티모달 기능을 추가할 수 있도록 돕습니다. MILS는 다단계 추론(multi-step reasoning) 능력을 활용해 다양한 응용 프로그램을 가능하게 합니다.

- **Technical Details**: MILS는 후보 출력을 생성하고 각 출력을 점수화(scoring)한 후 피드백(feedback)을 통해 반복(iteratively)적으로 솔루션을 도출하는 방식입니다. 이를 통해, 훈련이 필요했던 전문 모델을 사용하지 않고 특정 작업에 대한 솔루션을 생성할 수 있습니다. 이 방법은 이미지를 캡션(captioning)하는 데 있어 새롭고 최고의 성과를 달성했습니다.

- **Performance Highlights**: MILS는 이미지 생성(text-to-image generation)과 같은 미디어 생성(media generation)에도 효과적으로 적용할 수 있으며, 스타일 전이(style transfer)를 위한 프롬프트 수정(prompt rewrites) 발견에도 기여합니다. 또한, 이 방법은 그래디언트(gradient)가 필요 없는 최적화 방식이기 때문에 멀티모달 임베딩(multimodal embeddings)을 텍스트로 변환하여 크로스모달 산술(cross-modal arithmetic) 응용 프로그램에도 사용할 수 있습니다.



### ISAM-MTL: Cross-subject multi-task learning model with identifiable spikes and associative memory networks (https://arxiv.org/abs/2501.18089)
- **What's New**: 이 논문은 EEG 분류에서 교차 주제 변동성을 감소시키기 위해 ISAM-MTL이라는 다중 작업 학습 모델을 제안합니다. 이 모델은 각 주제의 EEG 분류를 독립적인 작업으로 간주하여 특징 공유를 촉진하고, 식별 가능한 스파이크 표현과 연관 메모리 네트워크를 통합합니다. 특히, label-guided variational inference를 통해 식별 가능한 스파이크 표현을 구성하여 분류 정확도를 향상시킵니다.

- **Technical Details**: ISAM-MTL 모델은 다른 주제의 샘플에서 공유된 특징을 추출하여 다중 작업 학습을 가능하게 합니다. 이 모델은 Hebbian learning 방식으로 학습되는 주제별 양방향 연관 메모리 네트워크를 활용하여 높은 효율성과 속도를 자랑합니다. 또한, pi-VAE 프레임워크를 통해 잠재적 스파이크 공간의 식별 가능성과 분리 가능성을 향상시키며, 각 작업에 따른 라벨 변수를 보조 변수로 사용하여 신경 활동과 작업 특성 간의 관계를 정량화합니다.

- **Performance Highlights**: ISAM-MTL은 BCI Competition IV IIa 데이터 세트에서 평균 84.1%의 분류 정확도를 기록하며, 주제 간 변동성을 줄이는 데 성공하였습니다. 또한, BCI Competition III Iva 데이터 세트에서는 40개의 샘플로 90% 이상의 정확도를 달성하며, 2개의 클래스 5 샷으로 분류할 수 있는 능력을 보여줍니다. 이러한 특성은 실제 BCI 시스템의 신속한 보정 가능성을 입증합니다.



### U-aggregation: Unsupervised Aggregation of Multiple Learning Algorithms (https://arxiv.org/abs/2501.18084)
- **What's New**: 본 논문에서는 기존의 지도 학습 방법과는 달리, 관찰된 레이블이나 결과 없이도 여러 개의 사전 훈련된 모델들을 집합하여 강력한 성능을 발휘할 수 있는 비지도 모델 집계 방법인 U-aggregation을 제안합니다. 또한, 데이터 이질성을 비롯한 다양한 현실적인 문제를 처리하는 데 초점을 맞추어, 주관적 특성이 있는 다수의 모델에서 유용한 통찰을 제공하고 있습니다. 이를 통해 새로운 인구 집단에서도 효과적인 예측을 가능하게 합니다.

- **Technical Details**: U-aggregation은 랜덤 행렬 이론(random matrix theory)의 통찰력에서 출발하여, 데이터 정규화(data-normalization) 절차와 수정된 동력 반복(power iteration) 알고리즘을 포함한 다단계 프로세스를 통해 모델 집계를 수행합니다. 이 방법은 각 개인의 실제 위험 값을 추정하고, 후보 모델의 상대적인 성능을 평가할 수 있도록 설계되었습니다. 또한, 모델 집계와 개인 예측 간의 관계에 대한 체계적인 이론 분석과 수치 실험 결과를 제공합니다.

- **Performance Highlights**: U-aggregation은 복잡한 특성을 지닌 유전적 위험 예측을 포함한 여러 실제 적용사례에서 기존의 사전 훈련된 모델들로부터 예측 정확도를 높이는 데 강력한 잠재력을 보여주고 있습니다. 공공 데이터베이스인 PGS Catalog의 모델들을 활용하여, 다양한 유전적 특성의 예측 성능을 향상시키는 데 기여할 수 있음을 사실을 통해 입증하고 있습니다. 이러한 접근법은 데이터와 레이블에 대한 접근이 어렵거나 제한적인 상황에서도 실질적인 응용 가능성을 제공합니다.



### Noise-Adaptive Conformal Classification with Marginal Coverag (https://arxiv.org/abs/2501.18060)
- **What's New**: 이 논문은 기계 학습에서 불확실성 정량화를 위한 새로운 적응형 conformal inference 방법을 제안합니다. 본 연구의 주요 기여는 레이블 잡음이 있는 상황에서도 긴밀한 주변 커버리지(marginal coverage)를 보장하는 정보 제공 예측 세트를 효율적으로 구성할 수 있다는 것입니다. 기존의 conformal methods가 잡음 문제로 인해 비효율적인 예측 세트를 생성하는 데 비해, 제안된 방법은 강력한 이론적 보장을 갖추고 있습니다.

- **Technical Details**: 제안된 방법은 (X_i, 	ilde{Y}_i)와 같은 관측값을 활용하여 새로운 테스트 포인트의 실제 레이블 Y_{n+1}을 위해 정보 제공 conformal prediction 세트를 구성하는 것을 목표로 합니다. 이 과정에서 마살르의 보조정리(Massart's lemma), 다들리 정리(Dudley's theorem) 등 다양한 이론적 도구를 활용하여 주어진 잡음으로 인한 커버리지의 편차를 효과적으로 분석합니다. 또한, 레이블 잡음 아래에서의 주변 커버리지 추정의 편차를 직접적으로 분석하여 비효율성을 줄이는 방법을 모색합니다.

- **Performance Highlights**: 다양한 합성 데이터 및 실제 데이터셋인 CIFAR-10H와 BigEarthNet을 통해 이 방법의 실효성을 검증하는 수치 실험을 진행하였습니다. 실험 결과, 제안된 방법은 불균형 클래스 상황에서도 유용한 예측 세트를 생성하며, 기존의 방법들이 초과적으로 보수적인 예측을 할 때 더 정보 제공적인 예측을 만들어냅니다. 이로써, 제안된 방법이 현재의 데이터 환경에서도 실제적인 문제를 해결할 수 있음을 보여주고 있습니다.



### Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks (https://arxiv.org/abs/2501.18018)
Comments:
          23 Pages, 2 Tables, 11 Figures

- **What's New**: 인공 신경망의 뉴런 구조를 생물학적 뉴런과 보다 유사하게 수정한 연구를 소개합니다. 연구에서는 생물학적 가지 돌기가 단순한 활성화 전달 역할을 넘어서 복잡한 비선형 함수(complex non-linear function)를 계산할 수 있다는 점에 주목했습니다. 이를 통해 더 나은 성능을 발휘하도록 설계된 'Perforated' 역전파(backpropagation) 시스템을 제안합니다. 이 시스템은 기존의 아키텍처에서 코드화한 동일한 특징을 위한 성능 향상을 목표로 합니다.

- **Technical Details**: 연구는 초기 네트워크 학습 단계 후, 네트워크에 'Dendrite Nodes'를 추가하고 이를 별도의 목표를 가지고 별도로 학습합니다. Dendrite Nodes는 원래 뉴런의 잔여 오류(remaining error)와 상관관계를 맺는 방향으로 훈련된 후 고정(frozen)됩니다. 이후 원래 뉴런은 Dendrite Nodes에서 제공되는 추가 오류 신호를 고려하여 다시 학습됩니다. 이 과정은 만족할 만한 성능에 도달할 때까지 여러 번 반복될 수 있습니다.

- **Performance Highlights**: 이 알고리즘은 최신의 PyTorch 네트워크에 성공적으로 추가되어 여러 도메인에서 원래의 정확도(original accuracies)를 개선했습니다. 또한, 모델 압축(model compression을 가능하게 하여, 정확도를 잃지 않으면서도 효율성을 높였습니다. 이러한 방법은 인공 신경망의 학습 및 성능 향상에 기여하는 중요한 발전으로 평가됩니다.



### Digital Twin-Enabled Real-Time Control in Robotic Additive Manufacturing via Soft Actor-Critic Reinforcement Learning (https://arxiv.org/abs/2501.18016)
- **What's New**: 본 연구는 Soft Actor-Critic (SAC) 강화 학습을 디지털 트윈 기술과 통합하여 로봇 적층 제조에서 실시간 프로세스 제어를 가능하게 하는 새로운 접근 방식을 소개합니다. Viper X300s 로봇 팔을 사용하여 정적 목표 도달과 동적 경로 추적 두 가지 제어 시나리오를 구현하였습니다. Unity 시뮬레이션 환경과 ROS2를 결합하여 디지털 트윈 동기화를 원활하게 수행하며, 전이 학습을 통해 학습된 모델을 효율적으로 태스크 간에 조정할 수 있습니다.

- **Technical Details**: 연구에서 사용된 Viper X300s 로봇 팔은 6자유도(6 DOF) 구조로, ROS inter-bridging을 통해 Unity의 가상 환경에서 태스크를 시뮬레이션하고 실제 로봇에서 검증할 수 있는 피드백 루프를 제공합니다. 디지털 트윈 방식의 연결을 통해 Unity에서 Viper X300s의 조인트 데이터를 로컬 TCP 서버로 전송하고, 이 데이터는 ROS2에서 구독하여 로봇 팔을 지속적으로 이동시킵니다. 이를 통해 로봇의 실제 동작에 대한 실시간 데이터를 제공하고, 지연 시간은 약 20밀리초로 실시간 반영이 가능합니다.

- **Performance Highlights**: 실험 결과는 신속한 정책 수렴과 튼튼한 작업 실행을 보여주었으며, 누적 보상(cumulative reward), 가치 예측 정확도(value prediction accuracy), 정책 손실(policy loss), 이산 엔트로피 계수(discrete entropy coefficient)와 같은 성능 지표가 연구 접근 방식의 효과성을 입증하고 있습니다. 강화 학습 기반의 제어 시스템에서 Unity와 ROS2 통합을 통해 로봇의 적응성을 향상시키고, 스마트 적층 제조 프로세스에 대한 실시간 제어를 강화하는 데 기여합니다.



### Fault Localization via Fine-tuning Large Language Models with Mutation Generated Stack Traces (https://arxiv.org/abs/2501.18005)
Comments:
          Accepted at ICST 2025

- **What's New**: 이 논문에서는 SAP HANA와 같은 운영 환경에서 발생하는 소프트웨어 충돌을 분석하기 위해 스택 트레이스(stack trace) 정보만을 사용하는 새로운 접근 방식을 제안합니다. 기존의 고전적인 접근법과 달리, 소스 코드나 테스트 실패 없이도 충돌의 원인을 식별할 수 있는 방법을 제시하였습니다. 이를 위해 64,369건의 충돌 데이터와 410만 건의 변형(mutation)을 통해 학습하여, 정확도가 66.9%에 도달함을 보여주었습니다.

- **Technical Details**: 저자들은 스택 트레이스에 기반하여 대형 언어 모델(Large Language Models, LLMs)을 미세 조정(fine-tuning)하는 방법을 사용하였습니다. 이 과정에서 코드 변형 기법을 통해 인위적으로 생성한 충돌 데이터를 사용하여 데이터 세트를 보강하였고, 이를 통해 다양한 코드베이스에서의 일반화 가능성도 검증하였습니다. 다양한 LLM을 사용하여 SAP HANA 외에도 SQLite와 DuckDB에서의 실험을 수행하였습니다.

- **Performance Highlights**: 이 연구의 결과는 미세 조정된 LLM이 충돌 원인을 정밀하게 추적하는 데 있어 비미세 조정된 LLM보다 우수하다는 것을 보여주었습니다. 연구를 통해 여러 데이터베이스에서 각각 63%와 74%의 정확도로 충돌을 예측했으며, 이는 기존의 통계적 방법과 심층 학습 기반 접근 방식보다 탁월한 성능을 발휘합니다. 논문에서는 이 방법이 다양한 소프트웨어 시스템에서의 충돌 분석에 효과적임을 확인하였습니다.



### InnerThoughts: Disentangling Representations and Predictions in Large Language Models (https://arxiv.org/abs/2501.17994)
Comments:
          Accepted at AISTATS 2025

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 성능 향상을 위해 각각의 transformer layer에서 생성된 히든 상태를 활용하여, 마지막 토큰 위치에서 답변 레이블을 예측하는 작은 신경망 예측 모듈을 학습하는 방법을 제안합니다. 이를 통해 LLM의 표현 능력과 예측 능력을 분리하는 구조를 갖습니다. 기존의 방식보다 훨씬 낮은 계산 비용으로, 여러 어렵고 까다로운 벤치마크에서 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 'InnerThoughts'라고 불리며, 각 layer의 히든 상태에서 최종 토큰 위치의 정보를 이용하여 예측을 수행합니다. 이 과정에서 기존의 LLM 파라미터는 고정된 채로 유지되며, 훈련 과정은 오직 단일 전방(pass) 계산으로 진행됩니다. 이를 통해 훈련 비용을 최소화하고, 고정된 파라미터로 범용적인 표현을 유지하면서도 특정 작업에 대한 성능을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 기존의 보정(calibration) 방법들과 비교하여 상당한 성능 향상을 보이며, 특정 벤치마크에서는 파라미터 효율적인 미세 조정 방법인 QLoRA (Dettmers et al., 2023)와 비슷한 수준의 성능에 도달합니다. 또한, 가장 큰 성능 향상을 보이는 벤치마크는 LLM의 답변 신뢰도가 낮은 질문들인 것을 확인하였습니다. 최종적으로, 특정 레이어에서의 기여도를 분석하여 우리의 접근 방식을 정당화하는 결과를 도출하였습니다.



### Reinforcement-Learning Portfolio Allocation with Dynamic Embedding of Market Information (https://arxiv.org/abs/2501.17992)
- **What's New**: 본 논문에서는 고차원, 비정상(non-stationary), 신호 대 잡음비(signal-to-noise ratio)가 낮은 시장 정보로부터 발생하는 문제를 해결하기 위해 딥러닝 기술을 활용한 포트폴리오 할당(framework) 기법을 제안합니다. 특히, 비정상적인 고차원 상태 공간을 낮은 차원으로 줄이는 동적 임베딩(dynamic embedding) 기법을 포함하고 있습니다. 또한, 생성 자동 인코더(generative autoencoders)와 온라인 메타 학습(online meta-learning)을 통합한 강화 학습(reinforcement learning) 프레임워크를 설계하여 시장 정보를 효과적으로 처리합니다.

- **Technical Details**: 이 연구에서는 반복 학습 환경에서 포트폴리오 할당 결정을 위한 상태 공간의 가장 영향력 있는 부분에 집중할 수 있도록uyangRL 에이전트가 정보(정보의 임베딩 및 메타 학습)를 동적으로 파악할 수 있게 합니다. 상위 500개의 미국 주식에 대한 실증 분석을 통해, 제안한 프레임워크가 전통적인 포트폴리오 벤치마크와 머신 러닝을 통한 예측-최적화(predict-then-optimize, PTO) 접근 방법을 초월하는 성능을 보여줍니다. 이 논문은 강화 학습 알고리즘을 사용하여 다양한 조건에서도 성능의 강건함을 입증하는 ablation study를 진행했습니다.

- **Performance Highlights**: 제안된 프레임워크는 특히 시장 스트레스가 발생하는 시기에 일반적인 포트폴리오 벤치마크를 초과하는 성과를 냈습니다. 전통적인 팩터 모델(factor models)은 이 우수한 성과를 완전히 설명하지 못하며, 프레임워크의 변동성 타이밍(timing volatility) 기능이 격변의 시기에 시장 노출을 줄이는 데 기여합니다. 게다가, 고차원, 노이즈가 많은 비정상 금융 데이터를 효과적으로 관리하는 임베딩 및 메타 학습 기법을 통해 포트폴리오 성과와 위험 관리가 향상되었습니다.



### VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting (https://arxiv.org/abs/2501.17978)
- **What's New**: 이번 논문에서는 3D Gaussian Splatting 모델을 확장하여 대칭 행렬을 추가함으로써 각 3D Gaussian의 불투명도 표현을 강화하였습니다. 이 개선을 통해, 관찰자의 시점에 따라 특정 Gaussian을 억제할 수 있게 되어, 보다 장면에 대한 정밀한 표현이 가능해졌습니다. 이는 시각적 반사와 스페큘러 하이라이트를 더 정확하게 재현할 수 있도록 도와줍니다.

- **Technical Details**: 기존 3D Gaussian Splatting 방식이 표면의 반사 효과에서 부족함을 보였던 문제를 해결하기 위해, 새로운 대칭 행렬을 도입하여 각 Gaussian의 스칼라 불투명도를 시점에 의존하는 함수로 변경했습니다. 이로 인해 장면의 완전성을 해치지 않으면서도 스페큘러 조명과 동적 조명 효과를 효과적으로 나타낼 수 있습니다. 이러한 접근은 기존의 렌더링 속도를 크게 저해하지 않으면서도 고급 성능을 달성하는 데 기여합니다.

- **Performance Highlights**: 제안된 모델은 Mip-Nerf, Tanks&Temples, Deep Blending, Nerf-Synthetic 데이터셋에서 최첨단 성능을 보여주며, 렌더링 속도가 >60FPS를 달성했습니다. 메모리 사용량의 작은 증가만으로도 성능을 강화할 수 있었으며, 복잡한 실세계 장면에서도 안정적인 결과를 보여주고 있습니다.



### A Robust Support Vector Machine Approach for Raman COVID-19 Data Classification (https://arxiv.org/abs/2501.17904)
- **What's New**: 본 논문에서는 Raman Spectroscopy(라만 분광법)를 활용하여 COVID-19 샘플을 분류하기 위한 새로운 강인화(robust) 지원 벡터 머신(Support Vector Machine, SVM) 모델을 제안합니다. 기존 모델들이 데이터의 복잡성과 변동성으로 인해 정확한 분석을 보장하지 못하는 문제를 해결하고, 강인 최적화 기법을 통해 분류 성능을 향상시키는 방법을 탐색합니다. 이는 실제 이탈리아 병원에서 제공된 COVID-19 데이터를 통해 검증되었습니다.

- **Technical Details**: Raman Spectroscopy는 분자의 저주파 모드를 관찰하기 위해 단색광의 비탄성 산란을 기반으로 한 기술입니다. SVM의 변형 모델을 사용하여 데이터의 두 개의 평행 초평면으로 분리한 후 최적의 초평면을 탐색하는 두 단계 절차를 통해 데이터를 처리합니다. 이러한 접근 방식은 생물학적 샘플의 노이즈와 불확실성에 적합하도록 강인한 최적화 기법을 적용하여 더욱 향상된 분류 성능을 제공합니다.

- **Performance Highlights**: 제안된 강인 SVM 모델은 기존의 SVM 접근 방식에 비해 대부분의 조건에서 우수한 성능을 보였습니다. 이는 불확실성을 고려한 분류 과정에서의 효율성을 입증하여 Raman Spectroscopy 분석에 적합한 후보임을 시사합니다. 실제 데이터 기반의 실험을 통해 강인 SVM 모델의 효과성을 강조하며, 향후 연구에 있어 이 기술의 가능성을 열어줍니다.



### Molecular Fingerprints Are Strong Models for Peptide Function Prediction (https://arxiv.org/abs/2501.17901)
- **What's New**: 본 연구에서는 분자 핑거프린트를 활용한 펩타이드(Property) 예측의 효과를 탐구하고, 도메인 특화(feature extraction) 방법이 복잡하고 코스트가 높은 모델(GNNs, pretrained sequence-based transformers 등)을 능가할 수 있음을 입증합니다. 특히, ECFP, Topological Torsion, RDKit 분자 핑거프린트를 기반으로 한 모델들이 견고하고, 상태 최고 성능(state-of-the-art)을 달성했습니다. 이 결과는 펩타이드에서 긴 범위(long-range) 상호작용의 중요성에 의문을 제기하며, 복잡한 딥러닝 모델의 대안으로서 분자 핑거프린트의 가능성을 강조합니다.

- **Technical Details**: 펩타이드는 일반적으로 아미노산 사슬로 이루어진 작은 단백질이며, 그 구조는 작은 아미노산 단위에서부터 시작됩니다. 기존의 머신러닝 접근법은 아미노산 시퀀스의 복잡한 피쳐 엔지니어링에 의존하거나, GNNs 및 Graph Transformers와 같은 새로운 방법론을 활용합니다. 본 연구는 LRGB 벤치마크를 포함하여 총 126개 데이터셋에 대해, LightGBM 분류기를 사용하여 해시된 분자 핑거프린트 모델이 뛰어난 성과를 거둘 수 있음을 보여주었습니다.

- **Performance Highlights**: 본 연구에서는 ECFP 및 LightGBM을 통한 펩타이드 기능 예측의 성과를 다섯 개의 벤치마크에서 입증하였으며, 총 126개의 데이터셋에 대한 최고의 기준 성과를 달성했습니다. 특히, 이 연구에서 제시된 카운트 기반의 분자 핑거프린트 모델은 긴 범위 GNN 모델들보다 향상된 성능을 보였습니다. 이러한 성과는 펩타이드 기능 예측의 복잡성을 줄이고, 다양한 머신러닝 파이프라인에서의 효율성을 증대시킬 수 있는 가능성을 보여줍니다.



### Distilling Knowledge for Designing Computational Imaging Systems (https://arxiv.org/abs/2501.17898)
Comments:
          14 figures, 16 pages

- **What's New**: 이 논문에서는 컴퓨터 이미징(Computational Imaging, CI) 시스템의 정확한 이미지 재구성을 위해 물리적 인코더 설계의 중요성을 강조합니다. 기존의 E2E(End-to-End) 최적화 방식은 encoder와 decoder를 동시에 최적화하지만, 물리적 제약으로 인해 성능이 저하됩니다. 저자들은 사전 학습된 덜 제약된 CI 시스템의 지식을 전이하는 지식 증류(Knowledge Distillation, KD) 개념을 재구성하여 이러한 한계를 극복하는 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법론은 세 가지 단계로 구성됩니다: 첫 번째로 학생 시스템(원래의 CI 시스템)에서 제약을 완화한 교사 시스템을 생성합니다. 두 번째로, 교사 시스템은 학생의 문제의 덜 제약된 버전을 해결하도록 최적화됩니다. 세 번째로, 교사는 제안된 두 가지 지식 전달 함수를 통해 학생 훈련을 안내하며, 이 과정에서 encoder와 decoder의 특성 공간을 타겟으로 합니다.

- **Performance Highlights**: 이 접근 방식은 자기 모드에 따라 조정할 수 있기 때문에 모든 이미징 모듈에 적용 가능합니다. MRI, 단일 픽셀 카메라 및 압축 스펙트럼 이미징과 같은 세 가지 대표적인 CI 모드에서 검증된 결과, 교사 시스템을 활용한 최적의 인코더 디자인을 통해 재구성 성능이 크게 향상됨을 보여주었습니다. 제안된 방법은 E2E 최적화와 전통적인 비 데이터 기반 인코더 디자인을 모두 초해 성능이 우수한 것으로 나타났습니다.



### Progress in Artificial Intelligence and its Determinants (https://arxiv.org/abs/2501.17894)
- **What's New**: 이 논문은 인공지능(AI)의 장기 발전을 정량적으로 연구하며, 특허와 출판물, 머신러닝 벤치마크, 새로운 ASOTA 지수를 포함한 다양한 측정치를 사용하여 이들의 기하급수적 성장을 보여주고 있습니다. AI 연구자들의 기여를 객관적으로 추정할 수 있으며, 이들 간의 비율을 설명하는 간단한 주장을 제시하고 있습니다. 또한, 기존 문헌의 머신러닝 스케일링 법칙과의 비교도 포함되어 있습니다.

- **Technical Details**: 이 논문에서는 컴퓨팅 자원과 인간의 지적 노동을 포함한 다양한 입력 측정치를 수집하고, 전통적인 출판물 수와 특허 수를 시작으로 여러 출력 측정치를 비교합니다. ASOTA 지수는 머신러닝의 벤치마크를 기반으로 정의되며, 그 기본 성질을 존중하면서도 무한히 미래의 기술 발전을 포함할 수 있도록 설계되었습니다. 논문은 입력과 출력 간의 관계를 설명하기 위해 두 가지 모델을 개발하고 있으며, 이를 통해 AI 발전에 있어 무어의 법칙이 주요한 요인임을 정량적으로 확인하였습니다.

- **Performance Highlights**: ASOTA 지수는 다양한 벤치마크 성과를 집계하여 머신러닝 모델의 성능 향상을 포착하며, 이전 성장률보다 느리지만 일관된 기하급수적 성장을 보여주고 있습니다. 전통적인 연구 성과 측정과 더불어, 지수는 AI 연구에 투자된 총 컴퓨팅 자원의 변화를 분석하여 AI 발전의 요인을 보다 명확히 이해할 수 있도록 도와줍니다. 궁극적으로 논문은 AI 기술 개발의 모니터링과 예측을 위한 정량적 프레임워크를 제공합니다.



### Language Modelling for Speaker Diarization in Telephonic Interviews (https://arxiv.org/abs/2501.17893)
- **What's New**: 이 논문은 스피커 다이얼라이제이션(speaker diarization) 시스템에서 언어적 및 음향적 모델링의 조합이 주는 이점을 조사합니다. 기존의 시스템은 주로 음향적 특징을 사용했던 반면, 언어적 데이터가 더욱 높은 판별성을 가질 수 있다는 점에 주목하였습니다. 제안하는 시스템은 LSTM(Long Short-Term Memory) 네트워크를 사용하여 스피커를 분류하는 반복 알고리즘에 기반하고 있으며, 이는 전화 인터뷰 오디오에 대한 Call-Center 데이터베이스에서 평가되었습니다.

- **Technical Details**: 제안된 알고리즘은 음향적 및 언어적 양쪽 데이터가 포함된 환경에서 작동하도록 설계되었습니다. 각 녹음은 두 명의 스피커(인터뷰어와 고객) 간의 대화로 구성되어 있으며, 이 시스템은 먼저 음향 특징을 추출한 후 ASR(Automatic Speech Recognition) 시스템을 적용하여 데이터를 전처리합니다. 이후 LSTM 네트워크를 통해 입력 된 단어에 대해 스피커 레이블을 태깅하게 됩니다.

- **Performance Highlights**: 제안된 시스템은 음향적 특징과 언어적 내용을 결합하여 우수한 성과를 나타냈습니다. 연구 결과, HMM/VB 기법 기반 시스템에 비해 84.29% 향상된 단어 수준 DER(word error rate)을 기록하였습니다. 이러한 결과는 언어적 콘텐츠가 특정 스피커 인식 작업에서 효율적으로 활용될 수 있음을 뒷받침합니다.



### Knoop: Practical Enhancement of Knockoff with Over-Parameterization for Variable Selection (https://arxiv.org/abs/2501.17889)
Comments:
          An earlier version of our paper at Machine Learning

- **What's New**: 이 논문은 변수 선택을 위한 새로운 접근법인 Knockoff with Over-Parameterization (Knoop)을 소개합니다. 기존 Knockoff 방법을 개선하여, 각 원본 변수에 대해 여러 개의 knockoff 변수를 생성하고 이를 Ridgeless 회귀 모델에 통합함으로써 더욱 효과적인 변수 선택을 도모합니다. Knoop 방법은 변수의 회귀 계수 분포를 평가하여, 원본 변수와 knockoff 변수 간의 비교를 통해 유의미성을 검증하는 과정으로 안정적인 변수 선택을 보장합니다.

- **Technical Details**: Knoop은 복수의 계층화된 knockoff 변수를 생성하는 재귀적 접근 방식을 사용하여, 이러한 knockoff 변수들이 원본 변수와 교환 가능한 독립성을 유지하도록 합니다. 이후 원본 변수와 knockoff 변수를 Ridgeless 회귀 모델에 통합하여 회귀 계수의 추정을 개선합니다. 마지막으로 Knoop은 이상치 기반 유의성 검정을 제안하여, 원본 변수를 knockoff 변수를 활용해 비교함으로써 false discovery rate (FDR)를 제어하고, 변수의 중요도에 따라 순위를 매깁니다.

- **Performance Highlights**: 광범위한 실험을 통해 Knoop의 성능이 기존 방법들에 비해 우수함을 입증하였습니다. 특히 Knoop은 시뮬레이션 및 실제 데이터셋에서 신뢰할 수 있는 변수 식별을 위한 평균적으로 더 높은 AUC (Area under the Curve)를 달성했습니다. 다양한 회귀 및 분류 작업에서 예측 정확도를 개선하며, 이론적 분석 결과 또한 관찰된 성능을 뒷받침합니다.



### RadioLLM: Introducing Large Language Model into Cognitive Radio via Hybrid Prompt and Token Reprogrammings (https://arxiv.org/abs/2501.17888)
- **What's New**: 이 논문에서는 Cognitive Radio Technology (CRT)와 Large Language Models (LLMs)를 결합하여 RadioLLM이라는 새로운 프레임워크를 개발했습니다. RadioLLM은 Hybrid Prompt and Token Reprogramming (HPTR) 및 Frequency Attuned Fusion (FAF) 모듈을 포함하여 다양한 CRT 작업에 적합한 성능을 발휘할 수 있도록 설계되었습니다. 이 접근법은 LLMs의 일반화 능력을 활용하여 신호 처리의 정밀도를 높이고, 더욱 효율적인 스펙트럼 할당과 신호 분류를 가능하게 합니다.

- **Technical Details**: RadioLLM은 두 가지 주요 구성 요소로 설계되었습니다. 첫째, 입력 신호를 패치로 나누어 신호 임베딩을 생성하며, 둘째, CNN에서의 높은 주파수 특징과 LLM에서의 낮은 주파수 신호 정보를 결합하여 출력합니다. GPT-2를 백본 모델로 사용하여 입력과 출력 간의 매핑 함수를 학습하고 LoRA 기술을 통해 모델을 미세 조정합니다. 이러한 방법은 다양한 신호 형식과 작업을 수용할 수 있는 통합된 프레임워크를 구조적으로 제시합니다.

- **Performance Highlights**: RadioLLM은 다양한 벤치마크 데이터셋에서 기존의 기준 성능을 초과하는 성능을 보여줍니다. 특히, RSC(Radio Signal Classification) 작업에서 높은 주파수 특성을 효과적으로 모델링하여 정확도를 높였습니다. 다양한 실제 시나리오에서 응용 가능한 범용 CRT 프레임워크로서의 유망성을 demonstrated 합니다.



### Heterogeneous Multi-Player Multi-Armed Bandits Robust To Adversarial Attacks (https://arxiv.org/abs/2501.17882)
- **What's New**: 본 논문은 다중 플레이어 다중 무장 강도 설정에서 적대자가 플레이어의 보상에 부정적인 영향을 미치는 상황을 고려합니다. 각 무장에 대한 보상 분포는 플레이어마다 이질적이며, 충돌 시 모든 플레이어는 0의 보상을 받게 됩니다. 적대자는 충돌을 이용하여 플레이어의 보상을 조작할 수 있으며, 플레이어들은 사전 결정된 정책을 따릅니다. 이 시스템에서는 최소 0의 적대적 공격을 수반하는 강력한 적대적 모델을 포함한 알고리즘을 제안하여, 거의 최적의 후회 수치를 달성하는 방법을 제시합니다.

- **Technical Details**: 시스템은 K(𝐾)명의 에이전트를 포함하며, 각 에이전트의 행동 공간은 M(𝑀)개의 무장입니다. 각 시점의 에이전트의 행동은 시그널을 채택하여 이루어지며, 이에 따른 보상은 무장의 보상 분포에 따라 결정됩니다. 적대적 공격이 있을 경우, 공격당한 무장을 선택한 플레이어는 0의 보상을 받게 되며, 이로 인해 플레이어는 충돌과 적대적 공격을 구분할 수 없습니다. 플레이어들은 O(로그 T) 시간 동안 하나의 비트 통신 옵션을 통해 서로 정보를 교환할 수 있습니다.

- **Performance Highlights**: 제안된 알고리즘은 후회수(R regret)의 측면에서 거의 최적의 결과를 달성합니다. 이 알고리즘은 적대적 공격이 발생하는 시점에 대한 후회를 추가적으로 고려하여, 시간 수평선(T horizon)에 대해 O(로그^{1+δ}T + W)의 성능을 보입니다. 여기서 W는 적대적 공격이 발생한 시간 단위 수를 나타냅니다. 본 연구는 다중 플레이어 MAB 설정에서 적대적 공격에 강한 알고리즘을 개발하는 중요한 기여를 합니다.



### RayLoc: Wireless Indoor Localization via Fully Differentiable Ray-tracing (https://arxiv.org/abs/2501.17881)
- **What's New**: 이 논문에서는 전통적인 무선 실내 위치 확정의 제한 사항을 극복하기 위해 RayLoc이라는 새로운 접근법을 제안합니다. RayLoc은 무선 레이 트레이싱의 역 문제로 위치 결정을 재정의하여 범위 내의 모든 장치가 없는 감지 장면의 매개 변수를 추출합니다. 이를 통해 CSIs의 정확한 재현이 가능해져 기존 기법들과 비교할 때 더 높은 정확도를 보입니다.

- **Technical Details**: RayLoc은 완전히 차별화된 레이 트레이싱 시뮬레이터를 기반으로 하여 무선 신호 전파를 정확히 시뮬레이션합니다. 이 시스템은 고충실도 배경 모델을 구축하여 물리적 환경의 기하학과 물질 전자기적 속성을 포함하고 있습니다. 또한, 가우스 커널을 사용하여 손실 경관을 부드럽게 하여 최적화의 어려움을 극복하고 위치 정확성을 높입니다.

- **Performance Highlights**: 신뢰할 수 있는 실험 결과를 통해 RayLoc은 기존의 전통적인 위치 추정 방법들보다 월등히 높은 성능을 보여 줍니다. 다양한 환경에서도 높은 일반화를 이루어내며, 이 시스템은 장치 없는 및 장치 기반의 위치결정에 통합된 예방 가능한 솔루션을 제시합니다. 마지막으로 RayLoc은 정밀한 위치 추정이 가능하며, 여러 싸이클을 통해 연결됩니다.



### Assessment of the January 2025 Los Angeles County wildfires: A multi-modal analysis of impact, response, and population exposur (https://arxiv.org/abs/2501.17880)
- **What's New**: 이 연구는 팔리세이즈, 이튼, 케네스, 허스트 등 네 가지 주요 캘리포니아 산불 사건을 종합적으로 분석하여, 땅의 변화, 관리 권한, 구조적 피해 및 인구 Vulnerability 등 다양한 측면에서 영향을 조사했습니다. Chebyshev-Kolmogorov-Arnold network (Cheby-KAN) 모델을 활용한 Sentinel-2 이미지를 통해 화재로 피해를 입은 지역의 범위를 정밀하게 매핑했습니다.

- **Technical Details**: 연구 방법론으로는 Sentinel-2 위성 이미지를 이용하여 화재 발생 전후의 상황을 분석하는 복합적인 접근 방식을 채택했습니다. Cheby-KAN 모델을 적용하여 315.36에서 10,960.98 헥타르에 이르는 화재 범위를 정밀하게 감지하며, 고해상도 지리공간 데이터를 활용하여 인구 조사와 토지 피복 변화, 인프라 Vulnerability를 평가했습니다.

- **Performance Highlights**: 연구 결과는 도시와 농촌 화재 사건 간의 구조적 피해와 인구 노출의 불균형을 보여 주며, 팔리세이즈와 이튼 화재가 각각 20,000명 이상에게 영향을 미친 반면, 농촌 사건은 500명 미만의 인구에 영향을 주었음을 밝혔습니다. 이 연구는 산불 관리 전략을 개발하는 데 있어 중요한 통찰을 제공하며, 특히 도시-야생 지역 인터페이스에서의 연령 및 성별에 대한 인식을 강조했습니다.



### Performance Analysis of NR Sidelink and Wi-Fi Coexistence Networks in Unlicensed Spectrum (https://arxiv.org/abs/2501.17878)
- **What's New**: 이 논문에서는 손에 넣고 있는 다양한 시스템과의 공정한 공존을 보장하는 데 어려움을 겪고 있는 5G-Advanced의 슬라링크 통신(SL-U) 기술에 대해 소개합니다. 새로운 협력적 채널 접근(CCHA) 메커니즘을 제안하여, 베이스 스테이션(BS)과 SL-U 사용자 간의 협력을 통한 자원 할당을 결합합니다. 이 방법은 기존 시스템과의 공존을 보장하면서 자원 활용성을 높입니다.

- **Technical Details**: 슬라링크 통신의 주요 도전 과제 중 하나는 라이선스가 없는 스펙트럼에서 Wi-Fi와 같은 기존 시스템과의 간섭을 줄이고 공정한 스펙트럼 공유를 보장하는 것입니다. 이를 위해 효과적인 중재 완화 방안을 모색하며, 수학적 모델링을 통해 공정성과 전송 속도 사이의 균형을 분석합니다. 마지막으로 협력적 하위 목표 기반의 계층적 심층 강화 학습(C-GHDRL) 프레임워크를 개발하여 SL-U 사용자들이 최적의 결정들을 내릴 수 있도록 지원합니다.

- **Performance Highlights**: 제안된 방안은 시뮬레이션 결과를 통해 SL-U와 Wi-Fi 사용자 간의 공정한 공존을 보장하면서도 공존 시스템의 성능을 크게 향상시키는 것으로 나타났습니다. 새로운 CCHA 메커니즘을 통해 자원 할당이 이루어지며, 이는 전반적인 성능 개선과 자원 효율성을 높이는 데 기여합니다. 협력적 접근을 통해 SL-U 사용자는 네트워크 내에서 전송 전력을 최적화하고 간섭 문제를 해결할 수 있는 방법을 제시합니다.



### On the challenges of detecting MCI using EEG in the wild (https://arxiv.org/abs/2501.17871)
Comments:
          10 pages

- **What's New**: 이 연구는 일반 진료 환경에서 수집된 새로운 EEG 데이터셋(GENEEG)을 소개하며, 이는 기존의 전문가에 의해 수집된 데이터셋(CAUEEG)과 달리 실제 진단 과정에서의 적용 가능성을 목표로 합니다. 이 데이터셋은 28명의 MCI 환자와 35명의 정상 환자로 이루어져 있으며, 이는 실제적인 MCI 진단을 반영하는 다양한 샘플을 포함하고 있습니다.

- **Technical Details**: 연구에서는 EEG 신호를 통한 MCI 탐지를 위해 여러 모델 아키텍처(MLP, CNN, Transformer)를 평가하고, 신뢰성 지표 및 교차 도메인 테스트를 통해 기존 모델들의 문제점을 보여줍니다. 특히 일반 진료 환경에서의 데이터 수집이 MCI 탐지의 실효성을 높이기 위해 필요하다는 점을 강조합니다.

- **Performance Highlights**: 상당수의 기존 연구가 소규모 데이터셋에서 이루어진 반면, 본 연구에서는 데이터셋 간의 분포 변화가 모델의 신뢰성 및 일반화 가능성에 미치는 영향을 조사합니다. 또한 MCI와 정상 환자 간의 특징이 겹치는 현상이 탐지에서의 근본적인 제한을 초래할 수 있음을 발견하였으며, 이는 향후 비침습적인 MCI 탐지의 발전에 중요한 도전 과제가 됩니다.



### Context is Key for Agent Security (https://arxiv.org/abs/2501.17070)
- **What's New**: 이 논문은 시스템의 안전성을 보장하고, 다양한 맥락에 적응할 수 있는 새로운 보안 설계에 대한 필요성을 강조합니다. 최신 시스템은 수동으로 작성된 정책이나 사용자 확인에 의존하여 결정을 내리지만, 이러한 방식은 다양한 환경에서 안전성을 보장하기에 부족합니다. 이 연구는 Contextual Security for Agents 프레임워크인 Conseca를 소개하여, 필요한 순간에 적절한 보안 정책을 생성하는 방법을 제안합니다.

- **Technical Details**: Conseca 프레임워크는 사용자 요청과 신뢰할 수 있는 맥락을 기반으로 특정 정책을 생성하는 방식으로 설계되었습니다. 이 시스템은 두 가지 기능을 수행합니다: 요청에 대한 보안 정책을 생성하고, 에이전트의 제안된 행동이 정책을 충족하는지 피드백을 제공합니다. Conseca는 언어 모델을 활용하여 동적으로 정책을 생성하고, 생성된 정책에는 전문가가 감사할 수 있는 인간-가독성의 근거가 포함됩니다.

- **Performance Highlights**: Conseca는 정확하고 컨텍스트에 적합한 보안 정책을 생성하여 다양한 맥락에서의 행동을 안전하게 제어할 수 있도록 합니다. 이 시스템은 악의적인 조작으로부터 보호하기 위해 신뢰할 수 있는 맥락만을 사용하여 정책 생성을 격리합니다. 결론적으로, Conseca는 다목적 에이전트 시스템에 대한 보안을 확장하는 데 필요한 연구와 프로토타입을 제공함으로써 향후 보안 메커니즘 설계에 중요한 기여를 합니다.




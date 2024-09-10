New uploads on arXiv(cs.CL)

### MMEvol: Empowering Multimodal Large Language Models with Evol-Instruc (https://arxiv.org/abs/2409.05840)
- **What's New**: 최근 Multimodal Large Language Models (MLLMs)의 중요성이 증가함에 따라 높은 복잡성을 지닌 다중 모달 지침 데이터의 수급이 큰 병목 현상으로 떠오르고 있습니다. 이러한 문제를 해결하기 위해 새로운 프레임워크인 MMEvol을 제안합니다.

- **Technical Details**: MMEvol은 세 가지 진화 방향, 즉 세밀한 인지 진화(fine-grained perception evolution), 인지 추론 진화(cognitive reasoning evolution), 상호작용 진화(interaction evolution)를 결합하여 데이터 품질의 병목 현상을 극복하고 복잡하고 다양한 이미지-텍스트 지침 데이터 세트를 생성합니다.

- **Performance Highlights**: MMEvol을 사용하여 생성된 데이터로 LLaVA-NeXT 모델을 훈련했으며, 기본 데이터로 훈련한 모델에 비해 평균 3.1 포인트의 정확도 향상을 달성하고 13개의 비전-언어 작업 중 9개의 작업에서 최첨단(SOTA) 성능을 기록했습니다.



### Improving Pretraining Data Using Perplexity Correlations (https://arxiv.org/abs/2409.05816)
- **What's New**: 본 논문은 고품질 프리트레인 데이터의 선택을 위한 새로운 프레임워크를 제안하고 있습니다. 기존에 비싼 프리트레인 모델 훈련을 요구하지 않고, LLM(Large Language Model) 훈련 없이도 높은 상관관계를 가진 문서를 선택하는 방법을 찾았습니다.

- **Technical Details**: 논문에서는 perplexity(당혹도)와 benchmark(벤치마크) 성능 간의 상관관계를 기반으로 데이터 선택을 수행하는 통계적 프레임워크를 개발했습니다. 90개의 LLM 샘플을 Open LLM Leaderboard에서 사용하여 다양한 웹 도메인(Texts from tens of thousands of web domains)에서 데이터를 선택했습니다.

- **Performance Highlights**: 160M 파라미터 규모의 통제된 프리트레인 실험에서, 제안한 방법이 각 벤치마크에서 DSIR을 초월하며, DataComp-LM에서 발견된 최상의 데이터 선택기와 일치하는 성능을 보여주었습니다.



### Benchmarking Chinese Knowledge Rectification in Large Language Models (https://arxiv.org/abs/2409.05806)
Comments:
          Ongoing work; code and dataset are available at this https URL

- **What's New**: 본 연구에서는 중국어에 특화된 대규모 언어 모델(LLMs)의 지식을 수정하기 위한 벤치마크인 새로운 데이터셋 CKnowEdit를 소개합니다. 이 데이터셋은 고전 문헌, 속담, 이디엄 등 7종의 지식을 포함하고 있으며, 중국어의 독특한 언어적 특성을 반영하여 수집되었습니다.

- **Technical Details**: CKnowEdit는 1,760개의 사례를 포함하며, 고전 시가, 속담, 이디엄, 음성 표기, 고전 중국어, 지리 지식, 그리고 Ruoziba라는 7개의 중국어-specific 종류의 지식을 조직하고 수집합니다. 현재의 지식 수정 방법을 평가하기 위해, 단어 수준의 중첩 및 의미 벡터 유사성을 사용하여 수정된 모델의 성능을 평가합니다.

- **Performance Highlights**: 상태-of-the-art 지식 수정 기법들의 실증 결과를 통해, 중국 문헌에 적용했을 때의 한계를 드러냈으며, 이는 미래의 더 정교한 중국어 지식 수정 접근 방식의 필요성을 강조합니다.



### Evidence from fMRI Supports a Two-Phase Abstraction Process in Language Models (https://arxiv.org/abs/2409.05771)
Comments:
          Equal contribution from both authors. Submitted to NeurIPS NeuroAI workshop 2024

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 중간 은닉 상태(intermediate hidden states)가 자연어 자극에 대한 뇌 반응을 예측할 수 있다는 기존 연구 결과를 바탕으로, 이러한 예측 성능을 가능하게 하는 표현적 특성에 대해 탐구합니다.

- **Technical Details**: LLMs의 중간 은닉 레이어가 선형 전이 작업에 가장 최적이라는 현상을 조사하면서, 학습 과정에서 자연스럽게 발생하는 두 단계의 추상화 과정(abstraction process)을 제안합니다. 우리는 manifold learning 방법을 사용하여 이 과정을 분석하고, 학습이 계속됨에 따라 첫 번째 '구성(composition)' 단계가 적은 레이어로 압축된다는 것을 보여줍니다.

- **Performance Highlights**: 이 연구는 LLM의 레이어별 인코딩 성능(encoded performance)과 LLM의 표현의 고유 차원(intrinsic dimensionality) 간에 강한 상관관계(correspondence)가 있음을 입증합니다. 이 관계는 LLM의 본질적인 조합성(compositionality)에서 기인한다는 초기 증거를 제공합니다.



### Towards Democratizing Multilingual Large Language Models For Medicine Through A Two-Stage Instruction Fine-tuning Approach (https://arxiv.org/abs/2409.05732)
Comments:
          Technical Report v1, work in progress

- **What's New**: 본 논문에서는 6개 언어로 된 20만 개 이상의 고품질 의료 샘플을 포함한 두 개의 다국어 지침 미세 조정 데이터 세트(MMed-IFT와 MMed-IFT-MC)를 소개합니다. 이를 통해 의료 LLM(대형 언어 모델)이 다양한 언어와 상황에서 최적의 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: MMed-IFT 데이터 세트는 일반 의료 지식을 주입하는 첫 번째 단계와 특정 작업에 대한 다중 선택 질문을 미세 조정하는 두 번째 단계로 구성된 2단계 훈련 패러다임을 따릅니다. 이 과정에서 LoRA(저비용 어댑터)를 사용하여 파라미터 효율적인 미세 조정을 수행하며, 영어와 다국어 벤치마크에서 경쟁력 있는 결과를 달성합니다.

- **Performance Highlights**: 본 방법은 다국어 의료 벤치마크에서 좋은 성능을 유지하면서도 컴퓨팅 효율성을 고려하여 설계되었습니다. 의료 전문가들이 안전하게 사용할 수 있는 지식이 풍부한 모델을 제공할 수 있는 가능성을 보여줍니다.



### Referring Expression Generation in Visually Grounded Dialogue with Discourse-aware Comprehension Guiding (https://arxiv.org/abs/2409.05721)
Comments:
          Accepted for publication at INLG 2024

- **What's New**: 이 논문에서는 시각적 기반 대화에서의 참조 표현 생성(REF)의 새로운 접근법을 제안합니다. 제안된 방법은 식별적이며 대화에 적합한 RE를 생성하기 위한 두 단계의 과정으로 구성됩니다.

- **Technical Details**: 첫 번째 단계에서는 REG를 텍스트 및 이미지 조건부의 다음 토큰 예측 작업으로 모델링합니다. 이후 RE는 이전 언어적 맥락과 참조 대상의 시각적 표현을 바탕으로 자가 회귀적으로 생성됩니다. 두 번째 단계에서는 대화 맥락을 보고 후보 RE의 선별을 위해 대화 인식(comprehension guiding)을 적용합니다.

- **Performance Highlights**: 인간 평가 결과, 제안된 두 단계 접근법이 더 높은 성능을 보여주었으며, 선택된 RE의 텍스트-이미지 검색 정확도가 더 높았습니다. 이는 기존의 탐욕적 디코딩 방법으로 생성된 RE에 비해 월등히 우수한 성과를 나타냅니다.



### RegNLP in Action: Facilitating Compliance Through Automated Information Retrieval and Answer Generation (https://arxiv.org/abs/2409.05677)
- **What's New**: 본 논문은 RegNLP(Regulatory Natural Language Processing) 분야에 기여하며, 자동화된 질문-단락 생성(Automated Question-Passage Generation) 작업을 정의하고, 27,869개의 질문을 포함한 ObliQA 데이터셋을 만들고, 규제 정보 검색 및 답변 생성 시스템을 설계하여 성능을 평가합니다.

- **Technical Details**: RegNLP는 규제 규칙과 의무의 접근과 해석을 단순화하기 위한 다학제 분야입니다. 오히려 복잡한 규제 문체에서 정확하게 정보를 추출하기 위한 자동화된 질문-단락 생성 프레임워크를 도입하고, 자연어 추론(Natural Language Inference, NLI)을 검증 단계에 통합하였습니다. ObliQA 데이터셋은 금융 규제 문서에서 파생된 질문과 관련 단락들을 포함합니다. RePASs 평가 메트릭은 생성된 답변이 모든 관련 의무를 정확히 반영하는지 평가합니다.

- **Performance Highlights**: 이 연구는 RegNLP의 데이터셋 및 평가 메트릭의 한계를 넘어선 포괄적이고 간결한 답변 생성이 가능하다는 점에서 향후 규제 문서의 이해 및 접근성을 크게 향상시킬 수 있습니다. 특히, 잘 설계된 질문 생성 및 정보 검색 단계는 규정 준수 오류를 줄이고, 운영 효율성을 증대시킬 수 있음을 보여줍니다.



### Revisiting English Winogender Schemas for Consistency, Coverage, and Grammatical Cas (https://arxiv.org/abs/2409.05653)
- **What's New**: 이 연구는 성별 편향을 평가하기 위한 Winogender schemas의 문제점을 식별하고 이를 해결하여 새 데이터셋 Winogender 2.0을 만들었다. 이 데이터셋은 원래 데이터셋의 한계를 극복하고, 대명사 편향을 평가하는 새로운 방법론도 제시하였다.

- **Technical Details**: Winogender 2.0은 총 1,440개의 문장으로 구성되어 있으며, 문법적 경우를 균형 있게 배분했다. 데이터셋은 각 문법적 경우에 따른 시스템 성능과 편향 특성을 측정할 수 있도록 설계되었다.

- **Performance Highlights**: Winogender 2.0은 Winogender 1.0 대비 모든 시스템에서 평균 0.1 F1 점수 하락을 보였다. 특히, 대명사에 따라 성능 차이가 있으며, 주격 대명사를 해결하는 데 가장 우수한 성능을 보였다.



### ExDDI: Explaining Drug-Drug Interaction Predictions with Natural Languag (https://arxiv.org/abs/2409.05592)
Comments:
          17 pages, 4 figures

- **What's New**: 이 연구는 약물-약물 상호작용(DDI) 예측에서 보다 신뢰성을 높일 수 있는 자연어 기반의 설명 생성 기법을 제안합니다.

- **Technical Details**: 연구에서는 DDInter와 DrugBank에서 수집한 약물 상호작용 설명을 활용하여, 예측을 수행하면서 약물의 약리 동력학(pharmacodynamics) 및 약물 약리학(pharmacokinetics) 메커니즘을 동시에 드러낼 수 있는 여러 모델을 개발했습니다.

- **Performance Highlights**: 제안된 모델은 알려진 약물 간의 알려지지 않은 DDI에 대한 정확한 설명을 제공할 수 있으며, 이는 DDI 예측 분야에 새로운 도구를 제공하여 추가 연구를 위한 기초를 마련합니다.



### MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery (https://arxiv.org/abs/2409.05591)
Comments:
          Codes and models are in this https URL

- **What's New**: 이번 연구에서는 MemoRAG라고 불리는 새로운 retrieval-augmented generation 파라다임을 제안합니다. MemoRAG는 장기 기억(long-term memory)을 통해 외부 데이터베이스에 접근하여 정보를 더 잘 활용하고, 불명확한 정보 요구를 처리할 수 있는 기능을 강화하였습니다.

- **Technical Details**: MemoRAG는 이중 시스템 아키텍처를 채택하여, 가벼운 LLM(light LLM)과 비싼 LLM(expensive LLM)을 결합합니다. 가벼운 LLM은 전체 데이터베이스의 글로벌 메모리를 형성하는 역할을 하며, 주어진 작업에 대해 초안을 생성하여 유용한 정보를 검색 후 제공하는 기능을 합니다. 반면, 비싼 LLM은 검색된 정보를 바탕으로 최종 답변을 생성합니다. 성능 향상을 위해 MemoRAG는 클루 생성 및 기억 용량을 최적화합니다.

- **Performance Highlights**: MemoRAG는 다양한 평가 작업에서 뛰어난 성능을 보여줍니다. 특히 기존 RAG 시스템이 실패하는 복잡한 작업에서도 주문형 고품질 답변을 생성할 수 있으며, 전통적인 질문-응답 작업에서도 유의미한 이점을 제공합니다. 실험에 사용된 UltraDomain 벤치마크는 법률, 금융, 교육 등 다양한 도메인에서 복잡한 RAG 작업들을 포함하고 있으며, MemoRAG는 그러한 작업에서 높은 성과를 달성하였습니다.



### Spatially-Aware Speaker for Vision-and-Language Navigation Instruction Generation (https://arxiv.org/abs/2409.05583)
- **What's New**: 이 논문은 Embodied AI를 위한 새로운 instruction generation 모델인 SAS (Spatially-Aware Speaker)를 소개합니다. SAS는 구조적 및 의미적 지식을 활용하여 더 다양하고 정보가 풍부한 내비게이션 지시를 생성합니다.

- **Technical Details**: SAS는 Encoder-Decoder 아키텍처 기반으로 구성되어 있으며, 시각적 입력으로부터 객체 카테고리, 객체 간의 공간적 관계, 주요 랜드마크 등을 추출하여 내비게이션 경로에 대한 명령을 생성합니다. 이를 통해 학습된 모델은 언어 평가 지표의 시스템적 편향을 피하기 위해 적대적 보상 학습(adversarial reward learning) 방법을 사용합니다.

- **Performance Highlights**: SAS 모델은 VLN 데이터셋에서 기존의 instruction generation 모델보다 향상된 성능을 보여주며, 여러 표준 언어 평가 지표에서 더 좋은 평가 결과를 기록하였습니다.



### QiBERT -- Classifying Online Conversations Messages with BERT as a Featur (https://arxiv.org/abs/2409.05530)
- **What's New**: 최근 온라인 커뮤니케이션의 발전은 짧은 텍스트(data) 형식으로 새로운 데이터 장르가 폭발적으로 증가하게 만들었습니다. 본 논문은 포르투갈 학교의 온라인 대화 데이터를 사용하여 학생들이 토론 주제에 대한 참여를 지속하는지를 관찰하는 것을 목표로 하고 있습니다.

- **Technical Details**: 본 연구는 BERT 모델을 기반으로 한 최신 기계 학습(ML) 알고리즘을 사용하여 학생들이 논의 주제에 대해 발언하였는지를 분류합니다. SBERT 임베딩을 특성(feature)으로 사용하여, 지도 학습(supervised learning)을 통해 온라인 메시지 분류에서 평균 0.95 이상의 정확도를 달성하였습니다.

- **Performance Highlights**: 이 모델은 짧은 텍스트에 대한 기존 텍스트 분류 툴의 한계를 극복하며, 사회과학자들이 인간의 커뮤니케이션, 행동, 그리고 설득을 이해하는 데 도움을 줄 수 있습니다. 이상적인 특성 선택 및 차원 축소 기법을 결합하여 모델의 성능을 향상시키고 있습니다.



### Harmonic Reasoning in Large Language Models (https://arxiv.org/abs/2409.05521)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 음악적 과제를 해결하는 능력을 조사했습니다. 특히 비슷한 모델인 GPT-3.5와 GPT-4o를 비교하여 음악의 노트 간격을 이해하고 코드 및 음계를 인식하는 방식에서의 차이를 분석했습니다. 연구 결과, LLM이 노트 간격에는 뛰어난 성능을 보였지만 더 복잡한 코드와 음계 인식에는 어려움을 겪는다는 점이 발견되었습니다.

- **Technical Details**: 대형 언어 모델(LLMs)은 음악 이론을 이해하기 위해 두 가지 실험을 수행했습니다. 첫 번째 실험에서는 음악 노트와 관련된 간격을 적용하고, 두 번째 실험에서는 코드와 음계를 인식했습니다. 이를 통해 LLM의 음악적 추론 능력을 테스트하고, 500개의 문제가 포함된 데이터 세트를 자동 생성했습니다. 모델의 성능은 각 실험에서 기대치와 비교하여 평가되었습니다.

- **Performance Highlights**: GPT-4o는 upward intervals에 대해 거의 100%의 정확도를 기록했지만, downward intervals 및 다중 옥타브를 포함한 가장 어려운 설정에서는 50% 이하의 정확도를 보였습니다. 이는 LLM이 훈련된 정보를 기억하고 재생산하는 능력은 있지만, 새로운 상황에 적합하게 사고하는 능력은 여전히 제한적임을 나타냅니다.



### Elsevier Arena: Human Evaluation of Chemistry/Biology/Health Foundational Large Language Models (https://arxiv.org/abs/2409.05486)
Comments:
          11 pages, 5 tables, 6 figures

- **What's New**: 이 논문에서는 Elsevier에서 수행된 생물 의학 분야를 위한 인공지능(AI) LLM(human evaluation experiment)을 소개합니다. 기존의 자동화된 벤치마크 평가의 한계를 극복하기 위해 A/B 테스트(A/B testing) 프레임워크를 적용하여 인간 평가자의 모델 선호도를 측정한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 Elsevier의 데이터로부터 훈련된 8.8B 파라미터의 LLM 모델과 OpenAI의 GPT-3.5-turbo, Meta의 Llama 2 모델을 평가하였습니다. 총 141개의 질문을 만들어 LLM의 출력 결과를 사실성(factuality), 일관성(coherence), 관련성(relevance), 개요(overview) 기준으로 평가하였습니다. A/B 테스트는 기본 LLM을 비교 기준으로 설정하여 수행하였으며, 모델은 온도(t=0.7) 기반 샘플링 방법으로 생성되었습니다.

- **Performance Highlights**: 결과는 GPT-3.5-turbo 모델이 대체로 더 높은 선호도를 보인 것으로 나타났지만, 상대적으로 작은 모델도 잘 curating된 데이터셋에서 훈련될 경우 생물 의학 분야에서 경쟁력 있는 대안이 될 수 있음을 시사합니다. 그러나 모든 모델의 IRR 점수는 일반적으로 낮았으며, 이는 인간 평가의 주관성과 불확실성을 반영합니다.



### Representational Analysis of Binding in Large Language Models (https://arxiv.org/abs/2409.05448)
- **What's New**: 이번 연구는 Binding ID 메커니즘에 대한 새로운 관점을 제시하며, 언어 모델(LM)이 엔티티-속성(예: Box Z와 coffee) 쌍을 어떻게 내부적으로 표시하는지에 대한 통찰을 제공합니다. 연구팀은 저차원 서브스페이스에서 BI 정보를 지역화하고, 이를 통해 속성과 엔티티의 순서를 인코딩하고 있음을 밝혔습니다.

- **Technical Details**: 연구에서는 언어 모델의 은닉 상태에서 BI 정보를 효과적으로 포착하기 위해 주성분 분석(Principle Component Analysis, PCA)을 사용하였습니다. 이 분석을 통해 BI 서브스페이스가 존재함을 밝혀내고, 이를 통해 LMs가 속성에 대한 인식을 어떻게 변경할 수 있는지를 설명합니다. 특정 방향으로 표현을 편집함으로써, LMs는 주어진 엔티티에 다른 속성을 바인딩할 수 있습니다.

- **Performance Highlights**: 연구 결과는 언어 모델이 BI 정보를 엔티티 활성화의 저차원 서브스페이스에 저장하고, 이 공간을 통해 속성 바인딩의 인과 관계를 조정할 수 있음을 보여주었습니다. 예를 들어, 'Box Z가 stone을 포함한다'는 추론을 할 수 있는 가능성을 추가로 입증했습니다.



### STLM Engineering Report: Dropou (https://arxiv.org/abs/2409.05423)
Comments:
          6 pages, 3 figures, For code base see this https URL

- **What's New**: 본 연구는 100M 파라미터 이하의 현대 언어 모델에서 dropout의 유효성을 탐구하며, 작은 고품질 데이터셋에서 샘플 효율성을 높이는 것과 더 큰 데이터셋에서의 적합도를 개선하는 두 가지 맥락에서 dropout의 역할을 분석합니다.

- **Technical Details**: dropout은 특정 레이어의 활성화를 확률적으로 비활성화하는 방법으로, 대규모 네트워크의 서브샘플링에 해당합니다. 연구에서는 dropout 비율로 0.1을 사용하여 덜 적합한 모델의 성능을 향상하는 새로운 스케줄링 방식도 제안합니다. 실험에서는 다양한 최적화 방법과 모델에 대해 dropout을 적용한 결과를 분석하였습니다.

- **Performance Highlights**: early dropout 사용은 긍정적인 성능 향상을 보여주고, dropout 스케줄링에 따른 성능 개선 효과를 관찰할 수 있었습니다. 특히 과적합 문제를 방지하기 위한 dropout의 사용이 성공적으로 입증되었습니다.



### Towards Building a Robust Knowledge Intensive Question Answering Model with Large Language Models (https://arxiv.org/abs/2409.05385)
Comments:
          This paper has been accepted by NLPCC-2024

- **What's New**: 본 논문에서는 LLM의 견고성을 평가하기 위해 기계 독해(Machine Reading Comprehension, MRC) 데이터셋을 기반으로 다양한 시나리오를 시뮬레이션하는 새로운 데이터셋을 구축하였습니다. 추가로, 노이즈와 정보 결여를 해결하기 위한 데이터 증강(data augmentation) 기반의 파인튜닝(fine-tuning) 방법을 제안하였습니다.

- **Technical Details**: 연구의 핵심은 LLM을 다양한 외부 정보에 노출시키고, 이를 통해 모델의 정확성을 평가하는 것입니다. 이를 위해 Single Source (SS), Single-Source-Incomplete (SSIncomp), Multi-Source-Consistent (MSCons) 및 Multi-Source-Inconsistent (MSIncons)와 같은 다양한 데이터셋을 구축하였습니다. 또한, 대조 학습(contrastive learning) 접근 방식을 통해 모델의 외부 정보 및 내부 지식의 활용 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법이 모델의 견고성을 개선하고, 외부 정보에 대한 차별화(discrimination) 능력을 강화함을 확인하였습니다. GPT-4를 활용한 성능 평가에서 개선된 결과를 나타내었습니다.



### Application Specific Compression of Deep Learning Models (https://arxiv.org/abs/2409.05368)
Comments:
          Accepted in the Proceedings of the 8th Joint International Conference on Data Science & Management of Data (12th ACM IKDD CODS and 30th COMAD) for the Short Research Paper track, 5 pages

- **What's New**: 이 연구에서는 특정 응용 프로그램에 최적화된 모델 압축 프로세스를 제안합니다. 기존 압축 방법들이 응용 프로그램의 정보를 활용하지 않는 문제를 해결하기 위해, Application Specific Compression (ASC) 방식을 사용하여 주어진 응용 프로그램에 대해 불필요한 네트워크 구성 요소를 가지치기(pruning)하여 모델을 최적화합니다.

- **Technical Details**: ASC는 훈련 데이터셋을 기반으로 모델을 최적화합니다. 모델의 각 층에서 데이터 표현의 변화를 분석하여, Quasar 프로세스를 통해 코사인 유사성(cosine similarity)을 측정합니다. 만약 어떤 모델 부분이 데이터 표현에 유의미한 변화를 주지 않는다면, 이는 해당 응용 프로그램에 대해 중복된 부분으로 간주됩니다. ASC 방법은 훈련 데이터에 대해 단일의 전방 패스(forward pass)를 통해 중복된 모델 부분을 식별할 수 있습니다.

- **Performance Highlights**: BERT 모델과 그 압축 버전을 사용하여 Extractive QA, Natural Language Inference, Paraphrase Identification의 세 가지 임무를 수행한 실험 결과, ASC 방식이 기존의 모델 압축 방법과 상용 compressed 모델에 비해 모든 작업에서 최상의 품질의 압축 모델을 생성했습니다.



### Diagnostic Reasoning in Natural Language: Computational Model and Application (https://arxiv.org/abs/2409.05367)
- **What's New**: 본 논문은 자연어(Natural Language) 기반 진단 유추 추론(Diagnostic Abductive Reasoning, DAR)에 대한 새로운 모델링 프레임워크를 제안합니다. 이 프레임워크는 Pearl의 구조적 인과 모델(Structural Causal Models, SCM)을 기반으로 하며, 생물 의학 분야의 과학 논문 평가에서 이를 실제로 구현하여 연구했습니다.

- **Technical Details**: 우리의 접근법은 업무 프로세스를 분석하고 진단 추론을 단계별로 분해하는 워크플로(workflow)를 도입합니다. 우리는 전문가 상호작용에 기초하여 세 단계의 프로세스를 통해 인과 모델(causal models)을 구축하고, 이를 통해 LLMs(대형 언어 모델)과 협력하여 진단 유추 작용을 효과적으로 지원하는 방법을 탐구합니다.

- **Performance Highlights**: 대규모 실험 결과, LLMs가 진단 추론의 개별 단계를 수행할 수 있는 능력이 있으나, 오류 전파(error propagation) 문제가 나타났습니다. 하지만, 인간의 감독이 이 문제를 효과적으로 완화시켜 LLMs의 성공적인 사용 가능성을 보여주었습니다. 이번 연구는 인과성, 인간-AI 협업, LLMs 분야의 연구 간의 격차를 메우고, 복잡한 진단 추론이 필요로 하는 전문 작업에 LLMs를 적용하기 위한 기초를 마련합니다.



### IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS (https://arxiv.org/abs/2409.05356)
- **What's New**: 이번 연구는 인도 언어에 대한 고품질 TTS(텍스트-음성 변환) 데이터 부족 문제를 해결하기 위해, 저품질 환경에서 수집된 자연 대화를 포함한 기존의 ASR(자동 음성 인식) 데이터셋을 개선하여 새로운 TTS 데이터셋인 IndicVoices-R(IV-R)을 개발했습니다. IV-R은 22개 인도 언어에서 1,704시간의 고품질 음성을 포함하며, 여러 인도 언어에 대한 TTS 모델의 제로샷(zero-shot) 일반화 능력을 평가하는 IV-R Benchmark도 함께 도입되었습니다.

- **Technical Details**: IV-R은 10,496명의 화자로부터 수집된 데이터를 기반으로 하며, 기존 TTS 데이터셋인 LJSpeech, LibriTTS, IndicTTS와 유사한 품질을 달성합니다. 특히, 엔지니어링 파이프라인을 통해 ASR 데이터의 노이즈를 제거하고 음성을 향상시키는 과정에서 영어로 훈련된 모델을 활용하여 크로스-링구얼(generalization) 효과를 극대화했습니다. 새로운 TTS 모델은 모든 22개 공식 인도 언어를 지원하며, 코드와 데이터는 오픈소스로 제공됩니다.

- **Performance Highlights**: IV-R을 사용하여 기존의 영어로 사전 훈련된 VoiceCraft 모델이 다수의 인도 언어와 화자를 지원할 수 있음을 보여주었으며, 다양한 인도 언어로 훈련된 데이터셋에 비해 제로샷 일반화 성능이 향상되었습니다. 이로 인해 인도 언어 TTS 시스템의 발전에 기여할 수 있을 것으로 기대됩니다.



### Seek and Solve Reasoning for Table Question Answering (https://arxiv.org/abs/2409.05286)
- **What's New**: 이 논문은 Table-based Question Answering (TQA) 성능을 향상시키기 위해 LLMs(대형 언어 모델)의 추론 능력을 활용하는 새로운 "Seek-and-Solve" 파이프라인을 제안합니다. 이 방법은 인간의 문제 해결 방식을 모방하여 두 단계로 나누어 질문 관련 정보를 찾고, 이후 질문을 해결하는 방식을 통합합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계인 "Seek"에서 LLM은 테이블 내 질문 관련 정보를 탐색하고, 이 정보를 바탕으로 추론의 단계를 나타내는 Seek Chain of Thought (CoT)를 생성합니다. 두 번째 단계인 "Solve"에서는 Seek-CoT를 사용하여 추론을 진행합니다. 이 과정에서 테이블 구조를 이해하기 위해 노드 기반 트리 구조를 모델링하여 복잡한 테이블에서 정보를 쉽게 찾아낼 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SS-CoT 경로를 사용하는 샘플을 시연으로 활용하면 LLM이 복잡한 TQA 작업을 효과적으로 해결하고 성능과 신뢰성이 향상됩니다. 이 방법은 기존 TQA 태스크 단순화 과정보다 더 높은 오류 수용성을 제공합니다.



### On the Relationship between Truth and Political Bias in Language Models (https://arxiv.org/abs/2409.05283)
- **What's New**: 이번 연구에서는 언어 모델 정렬(language model alignment)에서 중심적인 두 개념인 진실성(truthfulness)과 정치적 편향(political bias)의 관계를 분석하였습니다. 연구 결과, 진실성에 중점을 둔 보상 모델(reward models)을 훈련시키면 왼쪽으로 치우친 정치적 편향이 발생하는 경향이 있음이 밝혀졌습니다.

- **Technical Details**: 연구팀은 다양한 진실성 관련 데이터셋에 대한 보상 모델을 훈련하고, 결과적으로 이 모델들이 정치적 편향을 평가하였습니다. 기존의 오픈 소스 보상 모델들은 이미 비슷한 편향을 보였으며, 특히 모델 크기가 클수록 더 큰 편향을 드러내었습니다.

- **Performance Highlights**: 보상 모델 훈련 결과, 진실성 데이터셋에서 훈련된 모델이 왼쪽 편향을 보이며, 기후, 에너지, 노동 조합 관련 주제에서 편향이 특히 강하게 나타났습니다. 반면, 세금 및 사형과 관련된 주제에서는 편향이 약해지거나 심지어 반대 방향으로 나타났습니다.



### RexUniNLU: Recursive Method with Explicit Schema Instructor for Universal NLU (https://arxiv.org/abs/2409.05275)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2304.14770

- **What's New**: 새로운 연구는 정보 추출(IE)과 텍스트 분류(CLS) 작업을 통합할 수 있는 최초의 인코더 기반 모델인 RexUniNLU를 제시했습니다. 이 모델은 명시적 스키마 제약을 사용하여 다양한 NLU 작업을 효과적으로 수행합니다.

- **Technical Details**: 이 연구는 UIE(Universal Information Extraction)의 형식을 재정의하여 최근 UIE 모델이 해결하지 못했던 사중쌍(quadroples) 및 오중쌍(quintuples) 등의 스키마를 포함합니다. RexUniNLU는 모든 스키마 유형에 대한 쿼리를 재귀적으로 실행하며, 세 가지 통합된 토큰 링크 작업을 활용하여 결과를 계산합니다. 모델은 명시적 스키마 지시어(Explicit Schema Instructor, ESI)를 통해 레이블 의미 정보를 제공합니다.

- **Performance Highlights**: RexUniNLU는 정보 추출, 텍스트 분류 및 다중 모달 작업에서 이전의 소스 기준(State-of-the-art) 성능을 초과하며, 높은 자원 환경과 낮은 자원 환경 모두에서 우수한 성능을 보입니다. 이 모델은 PCLUE 리더보드에서 상위 두 자리를 차지하며, 복잡한 작업인 사중쌍 및 오중쌍 추출에서도 뛰어난 성과를 보입니다.



### UPCS: Unbiased Persona Construction for Dialogue Generation (https://arxiv.org/abs/2409.05257)
- **What's New**: 본 연구에서는 개인화된 대화 시스템에서 발생할 수 있는 편향 문제를 해결하기 위해 UPCS(비편향 페르소나 구축 시스템) 프레임워크를 도입하였습니다.

- **Technical Details**: UPCS는 캐릭터 설명을 8개의 차원으로 분류하여 편향 완화 전략을 통합하고, 자동화된 기능과 수동 점검을 결합하여 편향이 제거된 페르소나 세트를 생성합니다. 이 프레임워크는 초기 페르소나의 정보를 바탕으로 GPT-3.5를 이용하여 페르소나를 생성하고, 자동화 도구를 통해 편향이 제거됩니다.

- **Performance Highlights**: UPCS는 기존 페르소나 구축 방법들과 비교하여 정확성, 다양성, 편향 제거 및 사용자 만족도에서 우수한 성능을 보여주었습니다.



### Socially Responsible Data for Large Multilingual Language Models (https://arxiv.org/abs/2409.05247)
- **What's New**: 이번 논문에서는 다국어 대규모 언어 모델(LLMs)의 필요성과 도전 과제를 다룹니다. 특히, 역사적으로 디지털 영역에 충분히 대표되지 않은 저자원언어(low-resource languages)와 비서구 언어들에 대한 연구와 자료 수집의 윤리적 고려사항을 강조합니다.

- **Technical Details**: LLMs의 훈련 데이터는 주로 영어 텍스트로 구성되어 있으며, 다국어 처리(NLP) 작업을 위해 약 7,000개 언어에 대한 모델 개발이 필요합니다. 그러나 대부분의 세계 언어는 LLMs에서 대표되지 않으며, 특히 저자원언어의 경우 성능이 일반적으로 열악합니다. 이 논문은 이러한 언어에 대한 소셜, 문화적, 윤리적 고려사항을 개략적으로 설명합니다.

- **Performance Highlights**: 이 논문은 LLMs 개발자가 사회적으로 책임 있는 접근 방식을 취해야 하는 도전 과제를 6가지 제시하며, 특히 개발자와 지역 사회 커뮤니티 간의 파트너십을 통한 질적 연구 및 참여적 설계 접근 방식을 조명합니다. 또한 데이터 수집 시 고려해야 할 12가지 권장 사항도 제시합니다.



### Exploring Intrinsic Language-specific Subspaces in Fine-tuning Multilingual Neural Machine Translation (https://arxiv.org/abs/2409.05224)
- **What's New**: 이번 연구에서는 다국어 신경 기계 번역 모델(Multilingual Neural Machine Translation, MNMT)에 대한 파인튜닝이 전체 매개변수의 일부만으로도 수행 가능하다는 것을 보여줍니다. 또한, 언어별 LoRA(Language-specific LoRA) 방법을 제안하여 각 언어의 내재적 특성을 고려한 파인튜닝이 이루어질 수 있도록 합니다.

- **Technical Details**: 제안된 방법론에서는 스파스한 언어 특정 활성화를 갖춘 여러 개의 LoRA 모듈을 사용하는 언어별 LoRA (LSLo)를 활용하여 각 언어의 특수한 하위 공간에서 파인튜닝이 이루어지도록 합니다. 이를 위해 고-자원(high-resource) 언어의 경우 파라미터 수를 줄이는 방법과 점진적 다듬기(Gradual Pruning Schedule)를 도입합니다.

- **Performance Highlights**: 실험 결과, FLORES-101의 12개 및 30개 언어 부분 집합에 대한 결과에서, 제안된 방법은 전체 매개변수 파인튜닝에 비해 최대 2.25 spBLEU 점수 향상을 보여주었으며, 고-자원 언어에 대해서는 0.4%, 저-자원 언어에 대해서는 1.6%의 학습 가능한 파라미터로 효율적인 파인튜닝이 가능함을 시연했습니다.



### Interactive Machine Teaching by Labeling Rules and Instances (https://arxiv.org/abs/2409.05199)
Comments:
          Accepted to TACL 2024

- **What's New**: 이 연구는 약한 감독 학습(weakly supervised learning)에서 전문가의 제한된 시간을 효율적으로 활용하는 방법을 탐구하고 있습니다. INTERVAL이라는 대화형 학습 프레임워크를 제안하며, 이는 후보 규칙을 자동으로 추출하고 전문가의 피드백을 통해 규칙과 개별 예시의 효과를 극대화합니다.

- **Technical Details**: INTERVAL 프레임워크는 n-그램(n-grams), 구문(syntactic) 피처 및 프롬프트 기반 피처를 사용해 후보 규칙을 추출합니다. 이 방법은 기존의 약한 감독 방법보다 높은 F1 성능을 보여주며, 10회의 전문가 피드백이 필요할 뿐만 아니라, 기존의 능동 학습(active learning) 방법보다 현저히 적은 피드백으로 더 높은 성능을 달성합니다.

- **Performance Highlights**: 이 연구의 성과는 6개의 데이터셋을 통해 검증되었으며, INTERVAL은 최첨단 약한 감독 방법보다 7% 높은 F1 점수를 기록했습니다. 전문가 피드백에 소요되는 쿼리 수가 100회에 달하는 기존 방법들에 비해, 단 10회로도 동일한 수준의 F1 값을 달성할 수 있음을 보여주었습니다.



### Seemingly Plausible Distractors in Multi-Hop Reasoning: Are Large Language Models Attentive Readers? (https://arxiv.org/abs/2409.05197)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문은 최신 대형 언어 모델(LLMs)의 다중 홉 추론(multi-hop reasoning) 능력을 조사합니다. 기존의 다중 홉 추론 벤치마크에서 발견된 단순화된 단서가 모델이 추론 요구 사항을 회피하게 한다는 우려를 반영하여, LLM이 이를 이용할 가능성이 있는지 살펴보았습니다.

- **Technical Details**: 연구를 통해, LLM들은 멀티 홉 추론을 수행해야 하는 요구를 보다 미묘한 방식으로 회피하는 경향이 있다는 것을 발견했습니다. 저자들은 헷갈리게 하는 추론 경로가 LLMs에 큰 도전 과제가 된다는 점을 강조하며, 오류가 있는 답변으로 이어지는 그럴듯한 다중 홉 추론 체인을 생성하는 새로운 벤치마크를 제안합니다.

- **Performance Highlights**: 최신 LLM들을 평가한 결과, 그들은 그럴듯한 대안을 제시받았을 때 F1 점수가 최대 45% 감소하는 등 다중 홉 추론 수행에 영향받는 것으로 나타났습니다. 이 분석을 통해, LLM이 오해를 일으키는 어휘적 단서(lexical cues)를 무시하는 경향이 있지만, 헷갈리게 하는 추론 경로는 상당한 도전을 제시한다는 것을 알 수 있었습니다.



### OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs (https://arxiv.org/abs/2409.05152)
Comments:
          Work in progress; code is available at this https URL

- **What's New**: 이번 논문에서는 기존의 Large Language Models (LLMs)에서 발생하는 생성(generation)과 검색(retrieval) 작업의 통합 문제를 해결하기 위해 새로운 One-pass Generation and retrieval 프레임워크인 OneGen을 소개합니다. 이 프레임워크는 생성과 검색을 통합하여 동시에 처리할 수 있도록 설계되었습니다.

- **Technical Details**: OneGen 프레임워크는 autoregressively 생성된 retrieval tokens를 포함하여 전통적인 생성과 검색의 훈련 방식을 결합합니다. 이를 통해 단일 LLM이 통합된 forward pass 내에서 두 가지 작업을 동시에 처리할 수 있게 됩니다. 우리는 RAG와 Entity Linking이라는 두 가지 복합 작업에서 OneGen의 pluggability, 효과성, 효율성을 검증하기 위한 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 생성과 검색을 같은 맥락에서 통합하면 LLMs의 생성 능력을 유지하면서도 검색 성능이 향상됨을 확인했습니다. 또한, OneGen은 LLMs가 생성 중에 벡터 검색(vector retrieval)을 수행할 수 있도록 최초로 가능하게 하였습니다.



### READoc: A Unified Benchmark for Realistic Document Structured Extraction (https://arxiv.org/abs/2409.05137)
- **What's New**: 이 논문은 Document Structured Extraction (DSE)의 평가를 위한 새로운 벤치마크인 READoc을 소개합니다. 기존의 DSE 시스템들이 단편적으로 평가되고 있어 발전이 어려운 상황에서, READoc은 비구조화된 PDF 문서를 의미론적으로 풍부한 Markdown으로 변환하는 현실적인 작업으로 DSE를 정의합니다.

- **Technical Details**: READoc 데이터셋은 arXiv와 GitHub에서 수집된 2,233개의 다양한 실세계 문서에서 유래되었습니다. 이를 통해 PDF에서 Markdown으로의 전환을 목표로 하는 DSE 시스템의 과정을 평가하기 위한 DSE Evaluation S$^3$uite (Standardization, Segmentation, Scoring)를 개발했습니다.

- **Performance Highlights**: 다양한 최신 DSE 접근 방식을 평가하는 과정에서, 기존 연구들과 현실적인 DSE 목표 간의 격차를 찾아내고, READoc이 DSE 연구의 기반을 제공할 것이라고 기대하고 있습니다.



### MHS-STMA: Multimodal Hate Speech Detection via Scalable Transformer-Based Multilevel Attention Framework (https://arxiv.org/abs/2409.05136)
- **What's New**: 이번 논문에서는 소셜 미디어에서의 증오 발언(hate speech) 탐지를 위해 새로운 확장 가능 아키텍처인 STMA(transformer-based multilevel attention)를 제안합니다.

- **Technical Details**: STMA는 세 가지 주요 구성 요소로 이루어져 있습니다: 통합된 attention 기반 딥러닝 메커니즘, 비전 비디오(vision attention) 메커니즘 인코더, 자막(caption) attention 메커니즘 인코더. 각 구성 요소는 다양한 attention 프로세스를 사용하여 multimodal(다중모드) 데이터를 독특하게 처리합니다.

- **Performance Highlights**: STMA는 Hateful memes, MultiOff, MMHS150K 등 세 가지 증오 발언 데이터셋에서 여러 평가 기준을 이용한 연구를 통해, 모든 데이터셋에서 기존의 baseline 접근 방식보다 우수한 성능을 발휘함을 보여줍니다.



### Hate Content Detection via Novel Pre-Processing Sequencing and Ensemble Methods (https://arxiv.org/abs/2409.05134)
- **What's New**: 본 논문은 소셜 미디어에서의 증오 발언(hate speech) 식별을 위한 혁신적인 컴퓨팅 프레임워크를 소개합니다. 특히, 텍스트 전처리(text pre-processing) 작업의 순서를 변경할 때의 영향을 연구하여 효과적인 전처리 접근법을 제시합니다.

- **Technical Details**: 연구에서는 Support Vector Machine, Random Forest, Decision Tree, Logistic Regression 및 K-Neighbor와 같은 인기 있는 분류(classification) 방법들과 함께 가장 성능이 좋은 전처리 순서를 구현했습니다. 또한, bagging, boosting, stacking과 같은 다양한 앙상블 방법(ensemble methods)과 결합하여 성능을 더욱 향상시켰습니다.

- **Performance Highlights**: 제안된 접근법은 세 가지의 공공 벤치마크 데이터 세트(WZ-LS, DT, FOUNTA)를 사용하여 평가되었으며, 최대 95.14%의 정확도를 달성하여 독창적인 전처리 접근법과 앙상블 분류기(ensemble classifier)의 효과성을 강조합니다.



### WaterSeeker: Efficient Detection of Watermarked Segments in Large Documents (https://arxiv.org/abs/2409.05112)
Comments:
          18 pages, 5 figures, 4 tables

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 생성된 텍스트의 워터마크(watermark) 감지를 개선하기 위해 새로운 방법인 WaterSeeker를 제시합니다. 기존 방법들이 전체 워터마크와 비워터마크 텍스트를 구분하는 데 집중했다면, WaterSeeker는 긴 문서 내에서의 워터마크된 구간을 효율적으로 탐지하고 위치를 파악하는 데 초점을 맞추고 있습니다.

- **Technical Details**: WaterSeeker는 먼저 효율적인 이상치(anomaly) 추출 방법을 사용하여 의심스러운 워터마크 영역을 초기 탐지합니다. 이후에는 지역 탐색(local traversal)을 통해 전체 텍스트 감지를 수행하여 더 정밀한 확인을 진행합니다. 이 방법은 복잡도를 줄이면서도 감지 정확도를 최적화합니다.

- **Performance Highlights**: 실험 결과, WaterSeeker는 다양한 워터마크 강도와 구간 길이에 대한 적응력을 가지고 있으며, 기존의 기준 방법들보다 시간 복잡성과 감지 성능의 균형이 뛰어난 것으로 나타났습니다. 또한 WaterSeeker의 로컬라이제이션(localization) 기능은 해석 가능한 AI 감지 시스템의 개발에 기여할 수 있습니다.



### EdaCSC: Two Easy Data Augmentation Methods for Chinese Spelling Correction (https://arxiv.org/abs/2409.05105)
Comments:
          18 pages, 2 figures

- **What's New**: 이번 연구에서는 Chinese Spelling Correction(CSC)에서 발생하는 문제를 해결하기 위해 두 가지 데이터 증가(data augmentation) 방법, 즉 긴 문장을 짧은 문장으로 나누거나 여러 개의 오타를 포함하고 있는 문장의 오타를 줄이는 방법을 제안합니다.

- **Technical Details**: 제안된 방법인 EdaCSC는 (1) 데이터 세트를 두 가지 데이터 증강 방법으로 증강하고, (2) 최적의 모델을 선택하기 위해 다양한 훈련 절차를 적용하는 구조를 가지고 있습니다. 첫 번째 방법은 문장을 구두점 기준으로 나누어 모델의 과도 수정(overcorrection) 경향을 줄이고, 두 번째 방법은 다수의 오타를 포함한 문장의 오타를 줄여 성능 저하 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SIGHAN14와 SIGHAN15 데이터 세트에서 모든 비교 방법들을 초월하여 최우수 성능을 달성하였으며, SIGHAN13 데이터 세트에서도 두 번째로 우수한 성능을 기록하였습니다.



### Vision-fused Attack: Advancing Aggressive and Stealthy Adversarial Text against Neural Machine Translation (https://arxiv.org/abs/2409.05021)
Comments:
          IJCAI 2024

- **What's New**: 본 논문에서는 Adversarial attacks (적대 공격)의 새로운 개념인 Vision-fused Attack (비전 융합 공격, VFA) 프레임워크를 제안하여, 기존의 자연어 기계 번역(NMT) 모델에 대한 공격 능력을 높이고 사람의 인지에 대한 불가명성을 향상시켰습니다.

- **Technical Details**: VFA는 먼저 한정된 의미구역을 확장하기 위해 vision-merged solution space enhancement (VSSE) 전략을 사용하며, 이는 텍스트-이미지 변환 블록을 통해 시각적 특성과 의미적 특성을 결합한 후보 단어를 탐색하여 더 강력한 공격 능력을 발휘하게 합니다. 또한 perception-retained adversarial text selection (PATS) 전략을 통해 인간의 읽기 메커니즘에 맞춰 적대적 텍스트를 최적화하여 인지 불가성을 높입니다.

- **Performance Highlights**: 다양한 모델에서의 실험 결과 VFA가 기존 방법에 비해 ASR에서 81%, SSIM에서 14%의 성능 향상을 demonstrated했으며, 인간 인지에 대한 불가명성 또한 14% 개선되었습니다.



### Evaluation of Google Translate for Mandarin Chinese translation using sentiment and semantic analysis (https://arxiv.org/abs/2409.04964)
- **What's New**: 본 연구는 기계 번역 모델의 자동 평가를 인간 전문가와 함께 감정 및 의미 분석을 통해 수행합니다.经典文本인 '아 Q의 진짜 이야기'를 선택해, 구글 번역과 전문가 번역 간의 차이를 분석하여 새로운 통찰력을 제공합니다.

- **Technical Details**: 기계 번역(Machine Translation)과 자연어 처리(Natural Language Processing, NLP)에 대한 이론적 배경을 정립하며, 기계 번역 성능 비교를 위한 정량적 접근 방식을 사용합니다. 연구에서는 구글 번역과 전문가의 번역 결과를 감정 분석(Sentiment Analysis) 및 의미 분석(Semantic Analysis)으로 평가합니다.

- **Performance Highlights**: 구글 번역은 역사적 지식 및 맥락적 중요성의 결여로 인해 중국어의 특정 단어나 구절을 번역하는 데 어려움을 겪으며, 전문가 번역과 비교할 때 정확성과 감정 전달 측면에서 차이를 보입니다.



### Maximizing Relation Extraction Potential: A Data-Centric Study to Unveil Challenges and Opportunities (https://arxiv.org/abs/2409.04934)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 복잡한 관계 추출에 대한 데이터 중심 성능 분석을 최초로 수행한 연구로, 15개 최신 신경망 기반 관계 추출 알고리즘과 7개의 대규모 데이터셋을 사용하여 현재 알고리즘의 한계를 조명합니다.

- **Technical Details**: 이 연구에서는 현대의 관계 추출 알고리즘들이 복잡한 데이터 및 관계 특징에 견디지 못한다는 것을 입증하며, 중요한 이슈들로는 맥락 모호성(contextual ambiguity), 상관 관계(correlating relations), 긴 꼬리 데이터(long-tail data) 및 세분화된 관계 분포(fine-grained relation distributions)가 included됩니다.

- **Performance Highlights**: 이 논문은 관계 추출 알고리즘의 성능 격차를 강조하기 위해 15개의 알고리즘을 포괄적으로 비교하고, 현재의 상치 및 향후 방향에 대한 세부 논의를 제공합니다. 또한, 실천을 위한 데이터셋과 알고리즘 구현코드를 github에 제공하여 참조할 수 있게 하였습니다.



### Just ASR + LLM? A Study on Speech Large Language Models' Ability to Identify and Understand Speaker in Spoken Dialogu (https://arxiv.org/abs/2409.04927)
Comments:
          Accepted to IEEE SLT 2024

- **What's New**: 최근 음성 대화 모델(Speech LLMs)의 놀라운 발전으로 인해 인간의 청취 및 추론 능력과 비슷한 수준에 도달했습니다. 특히, Gaokao와 같은 벤치마크에서의 성과가 인상적이며, 대화의 음성과 내용의 이해가 결합된 질문응답(SQA) 능력이 주목받고 있습니다.

- **Technical Details**: 이 논문에서는 음성 대화 이해를 위한 Speech LLM의 성능을 이해하기 위해 context-based questions (CBQs) 및 identity-critical questions (ICQs)라는 두 가지 질문 유형으로 분류했습니다. 우리의 연구에서는 Qwen-Audio와 WavLLM 모델을 사용하여 두 가지 질문 유형에 대한 성능을 평가했습니다.

- **Performance Highlights**: 결과적으로, Speech LLM 모델들은 ICQs에서 상당히 저조한 성능을 보였으며, CBQs에 비해 음성 인식능력 부족이 드러났습니다. 이는 현재의 Speech LLM들이 대화 전사에서의 내용을 기반으로 추론하는 듯한 경향이 있으며, 실제로 음성 특성을 인식하지 못한다는 것을 시사합니다.



### Achieving Peak Performance for Large Language Models: A Systematic Review (https://arxiv.org/abs/2409.04833)
Comments:
          34 pages, 7 figures, 8 tables. Journal Article: IEEE Access

- **What's New**: 최근 대규모 언어 모델(LLMs)에 대한 성능 최적화 및 가속화 방법들을 다룬 체계적인 문헌 검토가 진행되었습니다.

- **Technical Details**: 이 논문은 2017년부터 2023년 12월까지 983개의 자료 중 65편의 문헌을 리뷰하였으며, LLM의 학습 최적화, 하드웨어 최적화, 그리고 시스템 서비스 개선을 위한 세 가지 카테고리로 분류하였습니다. 각 최적화 전략과 관련된 최근의 방법론들을 정리하였고, 다양한 프레임워크와 라이브러리의 효율성을 평가했습니다.

- **Performance Highlights**: 논문은 LLMs의 훈련 및 추론 효율성을 높이는 실제 사례를 두 가지 포함하고 있으며, 최첨단 성능을 유지하면서도 자원 제약 문제를 해결할 수 있는 실용적인 접근을 제시합니다.



### Exploring Straightforward Conversational Red-Teaming (https://arxiv.org/abs/2409.04822)
- **What's New**: 대규모 언어 모델(LLMs)이 비즈니스 대화 시스템에서 점점 더 많이 사용되고 있지만, 이들은 보안 및 윤리적 위험을 초래합니다. 이 논문에서는 공격 공격자 LLM을 사용하여 타겟 LLM에서 원하지 않는 출력을 유도하는 직관적인 레드팀(red-teaming) 접근 방식의 효과를 조사합니다.

- **Technical Details**: 우리는 다양한 단일 및 다중 턴(red-teaming) 전략을 비교하며, 공격자의 각기 다른 전술을 평가합니다. 이 연구에서는 사전 훈련된 LLM이 추가 훈련 없이도 공격자 모델로 효과적으로 사용될 수 있는지, 또한 대화 설정에서 공격 표면이 확대되는지에 대해 탐구합니다.

- **Performance Highlights**: 실험 결과, 사용자가 과거 시도의 경험에 따라 공격 전략을 조정할 수 있도록 하며, 비즈니스 애플리케이션에서의 활용 가능성을 확인했습니다. 이 연구는 성공적인 공격을 위한 대화 회수 수와 같은 중요한 변수에 대한 통찰력을 제공합니다.



### Phrase-Level Adversarial Training for Mitigating Bias in Neural Network-based Automatic Essay Scoring (https://arxiv.org/abs/2409.04795)
- **What's New**: 이번 연구에서는 자동 에세이 채점 시스템(AES)의 모델 불변(agnostic) 문구 수준 방법을 제안하여 AES 모델의 편향(bias)과 강건함을 개선하기 위한 적대적(Adversarial) 에세이 세트를 생성했습니다. 이로 인해 편향이 줄어들고 모델의 강건성이 향상되었습니다.

- **Technical Details**: 제안된 접근법은 원본 테스트 세트와 적대적으로 생성된 샘플을 포함하는 공격 테스트 세트를 구성합니다. 기계 학습 모델을 통해 에세이를 평가하고, 여러 신경망 점수 모델을 활용하여 공격 전략과 데이터 증강의 효과를 평가합니다. 주요 과정은 문장 추출기, 문구 추출, 변화된 문구 생성, 라벨 보존 필터 적용 순으로 진행됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 적대적 예시가 있는 경우와 없는 경우 모두에서 AES 모델의 성능을 크게 향상시킴을 보여주었습니다. 특히, 적대적인 샘플들에 대해 더 강건한 채점 결과를 제공하게 되었습니다.



### Selective Self-Rehearsal: A Fine-Tuning Approach to Improve Generalization in Large Language Models (https://arxiv.org/abs/2409.04787)
Comments:
          14 pages, 8 figures

- **What's New**: 이 논문은 Selective Self-Rehearsal (SSR)이라는 새로운 미세 조정(fine-tuning) 방법론을 소개하며, 이를 통해 모델의 일반화 능력을 개선하면서도 표준 감독 미세 조정(SFT)과 유사한 성능을 달성한다고 보고하였습니다.

- **Technical Details**: SSR은 질문에 대해 여러 개의 유효한 응답이 존재할 수 있다는 사실을 활용합니다. 모델의 올바른 응답을 식별한 후, SSR은 이러한 응답과 나머지 샘플에 대한 금본(gold) 응답을 이용해 모델을 미세 조정합니다. 이 과정은 적절한 LLM을 판별자로 사용하여 이루어지며, SSR은 모델이 자신의 생성된 출력(output)을 활용하여 학습하게 합니다.

- **Performance Highlights**: 여러 데이터셋에서 수행된 실험 결과에 따르면, 표준 SFT는 MMLU와 TruthfulQA와 같은 여러 벤치마크에서 평균 성능이 최대 16.7% 감소한 반면, SSR을 적용한 경우 평균적으로 2% 감소에 그쳐, SSR이 표준 SFT보다 더 나은 일반화 능력을 보여주었습니다.



### LoCa: Logit Calibration for Knowledge Distillation (https://arxiv.org/abs/2409.04778)
Comments:
          Accepted by ECAI 2024

- **What's New**: 최근 연구에 따르면, Knowledge Distillation (KD)의 한 문제인 mis-instruction을 발견했습니다. 이는 teacher logits에 기반한 예측이 라벨을 따르지 않을 때 발생하며, 학생 모델이 잘못된 방향으로 학습하게 됩니다. 이를 해결하기 위해 Logit Calibration (LoCa) 방법을 제안합니다.

- **Technical Details**: LoCa는 teacher 모델에서 ground-truth 라벨을 근거로 logits를 보정하여 mis-instruction 문제를 해결하고, 동시에 중요한 dark knowledge를 유지합니다. 이 방법은 추가적인 파라미터 없이 작동합니다. LoCa의 보정 과정은 세 가지 관점(확률 분포, 예측 정확성, 비타겟 비율 불변성)을 고려한 최적화 문제로 모델링됩니다.

- **Performance Highlights**: 실험 결과, LoCa는 CIFAR-100 및 ImageNet에서의 이미지 분류와 Dolly, S-NI 및 UnNI 데이터셋에서의 텍스트 생성 작업에서 기존 방법들보다 성능을 크게 개선했습니다. 또한, hyperparameter alpha의 범위가 0.9에서 1.0일 때 높은 활용성과 견고성을 보여주었습니다.



### Untie the Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models (https://arxiv.org/abs/2409.04774)
- **What's New**: 이 논문에서는 'Untie the Knots' (UtK)라는 새로운 데이터 증강 전략을 소개합니다. 이 방법은 기존 데이터 혼합을 수정하지 않고도 LLM이 긴 컨텍스트 처리 능력을 효율적으로 향상시키도록 설계되었습니다.

- **Technical Details**: UtK는 입력 문서를 청크(chunk) 단위로 나누고, 이를 무작위로 섞은 후 복잡한 구조의 긴 텍스트를 생성하여, 모델이 무질서한 토큰 시퀀스 내의 관련 세그먼트를 식별하도록 학습합니다. 이 과정에서 백트레이싱(Backtracing) 작업을 도입해, 모델이 모든 세그먼트를 올바른 순서로 찾도록 합니다.

- **Performance Highlights**: 7B 및 72B 매개변수 모델을 사용하여 200억 개의 토큰으로 훈련한 결과, UtK는 128K 컨텍스트 길이에서 RULER에서 75% 및 84.5% 정확도를 달성하며, 기존의 데이터 전략을 크게 초월하는 성능을 보여줍니다. 이 모델들은 오픈 소스로 제공되어 향후 연구를 촉진할 예정입니다.



### Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models (https://arxiv.org/abs/2409.04701)
Comments:
          4 pages, early draft

- **What's New**: 이 논문에서는 'late chunking'이라는 새로운 방식의 텍스트 조각 인코딩 방법을 제안합니다. 기존의 방식에서는 문서의 텍스트를 작은 조각으로 나누어 개별적으로 인코딩하는 것에서 발생하는 컨텍스트 정보 손실 문제를 해결합니다.

- **Technical Details**: Late chunking 방법은 긴 텍스트 임베딩 모델을 활용하여 먼저 문서의 모든 토큰을 임베딩한 후, Mean Pooling 적용 직전에 조각으로 나누는 방식입니다. 이로 인해 전체 텍스트의 컨텍스트 정보를 보존하며, 이를 통해 텍스트 내의 의미와 관련성을 더 잘 캡처할 수 있습니다.

- **Performance Highlights**: Late chunking은 다양한 검색 작업에서 기존의 조각화 방법보다 우수한 성능을 보였으며, 추가적인 학습 없이도 다양한 긴 컨텍스트 임베딩 모델에 적용할 수 있는 범용성을 가지고 있습니다.



### Sparse Rewards Can Self-Train Dialogue Agents (https://arxiv.org/abs/2409.04617)
Comments:
          Minor but nontrivial changes likely

- **What's New**: 최근 대규모 언어 모델(LLM) 에이전트의 성능이 높아짐에 따라, 더 이상 외부 인간 피드백 없이 자율적으로 성능을 개선할 수 있는 새로운 패러다임이 제안되었습니다. 이 출처에서는 특정 도메인에서 LLM이 인간의 능력을 초과할 수 있다는 가능성에 주목하며, Juxtaposed Outcomes for Simulation Harvesting (JOSH)라는 자체 정렬 알고리즘을 소개합니다.

- **Technical Details**: JOSH는 희소 보상( sparse reward) 시뮬레이션 환경을 활용하여 이상적인 행동을 추출하고 LLM을 자체 출력에 대해 추가로 훈련시키는 혁신적인 방법입니다. ToolWOZ는 MultiWOZ에서 파생된 희소 보상 도구 호출 시뮬레이션 환경으로, 도구 기반 상호작용에서 LLM의 성능을 크게 향상시킵니다.

- **Performance Highlights**: JOSH를 통해 훈련된 모델은 ToolWOZ에서 74% 성공률 증가를 기록했으며, 다른 벤치마크에서도 일반 모델 성능을 유지하며 전반적인 능력을 저하시키지 않습니다. 또한, JOSH의 사용으로 인해 GPT-4o는 8%의 대화 성공률 증가를 달성하여, 최신 성과를 보였습니다.



### BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training (https://arxiv.org/abs/2409.04599)
Comments:
          9 pages

- **What's New**: Picky BPE라는 수정된 BPE 알고리즘이 소개되었습니다. 이 알고리즘은 tokenizer 훈련 중 어휘 정제를 수행하여 어휘의 효율성을 개선하고, 중간에 불필요한 토큰을 제거합니다.

- **Technical Details**: Picky BPE는 각 토큰의 개별 빈도 및 더 큰 토큰 내의 빈도를 기준으로 중간 토큰을 식별합니다. 이 과정에서 Intersection over Self (IoS)라는 지표를 도입하여, 특정 토큰의 빈도를 분석합니다.

- **Performance Highlights**: 실험 결과, Picky BPE는 다운스트림 번역 작업에서 동등하거나 더 나은 성능을 보여주었고, 낮은 빈도의 언더 트레인(token) 수를 줄이며, 품질 높은 초기 단어에 대한 공간을 확보했습니다.



### Paper Copilot: A Self-Evolving and Efficient LLM System for Personalized Academic Assistanc (https://arxiv.org/abs/2409.04593)
- **What's New**: 본 논문은 연구자들이 대량의 문헌을 효율적으로 탐색할 수 있도록 돕기 위해 설계된 self-evolving LLM 시스템인 Paper Copilot을 제안합니다. 이 시스템은 사용자 프로필을 기반으로 개인화된 연구 서비스를 제공하며, 최신 Arxiv 논문을 바탕으로 매일 갱신되는 데이터베이스를 유지합니다.

- **Technical Details**: Paper Copilot은 사용자 프로필로부터 연구 이력을 추출하고, 최신 트렌드 분석 및 아이디어 제안을 통해 개인화된 연구 지원을 제공합니다. 또한, Thought Retrieval 방법을 통해 이전 사용자 쿼리에 기반한 응답 개선을 수행하며, 효율적인 배포를 위해 멀티스레딩 방식과 캐시 시스템을 도입하여 API 비용 및 응답 시간을 69.92% 감소시킵니다.

- **Performance Highlights**: Paper Copilot은 필요한 정보를 얻는 데 있어 최소 20분을 절약할 수 있으며, 효율성과 사용자 경험에서 우수성을 입증했습니다. 데이터에 따르면, Paper Copilot은 효율적인 배포 후 69.92%의 시간 비용을 줄일 수 있음을 보여줍니다.



### Customizing Large Language Model Generation Style using Parameter-Efficient Finetuning (https://arxiv.org/abs/2409.04574)
- **What's New**: 이 논문에서는 파라미터 효율적인 미세 조정 방법인 PEFT(Parametric Efficient Finetuning)를 통해 LLM(대규모 언어 모델)의 스타일 생성을 사용자 맞춤형으로 조정하는 방법을 탐구합니다. LLaMA-2 모델을 기반으로 하여 10명의 다양한 저자를 위한 맞춤형 텍스트 생성을 위해 LoRA(Low-Rank Adaptation)를 사용하였습니다.

- **Technical Details**: 이 연구에서 우리는 스타일 맞춤화를 위해 LoRA 어댑터를 Llama-2-7b 모델에 미세 조정했습니다. 이는 특정 저자의 비구조적 데이터셋을 통해 이루어졌으며, 출력의 스타일 성격을 반영하면서 기존 훈련에서 학습한 능력을 유지하는 것을 목표로 합니다. 저자의 이름과 관련된 'content' 단어를 제외하고 스타일에만 초점을 맞출 수 있는 것도 이 방법의 장점입니다.

- **Performance Highlights**: 실험 결과, StyleTunedLM은 전통적인 프롬프트 엔지니어링 및 몇 예시 학습(few-shot learning) 접근법에 비해 스타일 포착 능력이 뛰어난 것으로 나타났습니다. 이를 통해 사용자의 기존 작업에 적합한 텍스트 생성을 더 효율적으로 지원할 수 있음을 밝혀냈습니다.



### How Does Code Pretraining Affect Language Model Task Performance? (https://arxiv.org/abs/2409.04556)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 사전 훈련 과정에서 자연어와 코드 데이터를 혼합해 훈련할 경우의 성능 향상을 다룹니다. 코드 포함이 비프로그래밍 작업에 미치는 영향을 명확하게 설정한 최초의 연구로서, 이는 코드와 언어 데이터의 연관성을 탐구합니다.

- **Technical Details**: 연구진은 자연어와 소스 코드 데이터를 다양한 비율로 혼합한 데이터셋을 구성하였으며, 두 가지 다른 실험 설정(경쟁적 및 추가적)에서 언어 모델을 사전 훈련했습니다. 이 과정에서 코드 비율의 변화가 (a) BigBench 벤치마크의 다양한 작업에서 성능에 미치는 영향을 분석하였고, (b) 의미 구문 분석과 구문 변환에 대한 일반화 정확성으로 측정되는 구성 가능성(compositionality)을 평가했습니다.

- **Performance Highlights**: 코드 비율이 높아질수록 수학 및 구성적 작업에서 성능이 향상되었으며, 반면 언어 구조(구문 또는 형태소)를 필요로 하는 작업에서는 성능 저하가 발생했습니다. 코드 사전 훈련이 하위 작업 성능에 변동성을 증가시키는 반면, 최상위 4분위 작업에서는 성능 향상이 관찰되었습니다.



### Chain-of-Translation Prompting (CoTR): A Novel Prompting Technique for Low Resource Languages (https://arxiv.org/abs/2409.04512)
- **What's New**: 이 논문은 저자들이 언급한 저자원 언어의 성능을 높이기 위해 새로운 전략인 Chain of Translation Prompting (CoTR)을 도입한 것입니다. CoTR은 입력 문맥을 저자원 언어에서 더 높은 자원의 언어인 영어로 번역한 후, 그 번역된 텍스트에서 특정 NLP 기능을 수행하는 방식입니다.

- **Technical Details**: 이 방법은 주어진 작업을 수행하기 위해 LLM(대형 언어 모델)을 활용하며, 단계는 단일 프롬프트에서 모두 진행됩니다. 특히, 감정 분석, 증오 발언 분류, 주제 분류 및 텍스트 생성을 포함하는 다양한 작업에 적용됩니다. 이 연구는 CoTR 방법이 기존의 직접적인 프롬프트 방법에 비해 우수한 성능을 보여준다는 점을 강조합니다.

- **Performance Highlights**: 연구 결과, CoTR 방법이 특히 감정 분석 및 증오 발언 탐지 작업에서 가장 높은 정확도 개선을 보여 주며, 다양한 LLM 모델에서 일관되게 더 나은 성능을 발휘하여 모델에 따른 성능을 비교할 때, 폐쇄형 LLM이 더 높은 점수를 기록했습니다.



### Leveraging Large Language Models for Solving Rare MIP Challenges (https://arxiv.org/abs/2409.04464)
- **What's New**: 이 논문에서는 Mixed Integer Programming (MIP) 문제 해결을 위한 새로운 접근 방식인 재귀적 동적 온도 방법을 제안합니다. 이 방법은 체인 오브 씽크(Chain-of-Thought) 접근 방식과 통합되어 있습니다.

- **Technical Details**: 기존의 온도가 낮고 일정한 환경에서는 LLMs (대형 언어 모델)이 다양한 해결책을 탐색하는 데 제한이 있었으나, 높은 온도에서 시작해 점차 낮추는 재귀적 방법을 통해 보다 나은 해를 제공함을 입증했습니다.

- **Performance Highlights**: LLMs는 Gurobi와의 비교를 통해 전통적인 솔버의 가지치기 과정을 가속화하고 전반적인 효율성을 개선하여 보완적인 결과를 도출해낼 수 있음을 보여주었습니다.



### PDAF: A Phonetic Debiasing Attention Framework For Speaker Verification (https://arxiv.org/abs/2409.05799)
Comments:
          Accepted to SLT

- **What's New**: 이 논문은 화자의 음성 인증을 위해 기존의 방식인 특징 벡터(feature vector) 비교에 도전하며, 음소의 우세(phonetic dominance)를 강조하여 음성 내용의 중요성을 제시합니다.

- **Technical Details**: PDAF(Phoneme Debiasing Attention Framework)는 음소의 비편향화(demising) 수단으로, 기존의 attention framework와 결합하여 음소의 가중치를 조정하고 특징 추출에 영향을 미칩니다. 이는 음성 분석을 보다 미세하게 할 수 있게 합니다.

- **Performance Highlights**: 여러 가중치 전략을 통해 음소의 특성이 화자 인증 시스템의 효율성에 미치는 영향을 평가하고, 실험을 통해 음소의 결합 기여도가 개별 기여도보다 더 높다는 것을 발견했습니다.



### Evaluation of real-time transcriptions using end-to-end ASR models (https://arxiv.org/abs/2409.05674)
Comments:
          15 pages, 4 figures

- **What's New**: 본 논문에서는 실시간 자동 음성 인식(ASR) 시스템을 위한 오디오 분할 알고리즘의 성능을 평가합니다. 오디오를 짧은 조각으로 나누어 ASR 시스템에서 실시간으로 처리할 수 있게 하여 지연(latency)을 최소화하는 방법을 제안합니다.

- **Technical Details**: 세 가지 오디오 분할 알고리즘인 고정 간격 조각화(fragmentation at fixed intervals), 음성 활동 감지(Voice Activity Detection, VAD), 피드백 조각화(fragmentation with feedback)를 다양한 ASR 모델과 결합하여 평가합니다. 이 연구는 GigaSpeech 데이터셋을 사용하여 모델 성능과 지연을 측정합니다.

- **Performance Highlights**: 실험 결과, VAD 분할 방식이 가장 높은 품질을 제공하였으나 지연이 가장 길었습니다. 고정 간격 조각화는 가장 낮은 품질과 짧은 지연을 보였고, 새로운 피드백 알고리즘은 VAD 분할에 비해 2-4%의 오류율 증가로 1.5-2초의 지연 감소를 이루었습니다.



### Longer is (Not Necessarily) Stronger: Punctuated Long-Sequence Training for Enhanced Speech Recognition and Translation (https://arxiv.org/abs/2409.05601)
Comments:
          Accepted at SLT 2024

- **What's New**: 이 논문은 음성 인식 및 번역 작업을 위한 sequence-to-sequence 모델 훈련의 새로운 방법을 제안합니다. 전통적인 방식인 짧은 세그먼트로 훈련하는 대신 완전한 문장과 문장 부호 및 대문자를 포함하는 긴 발화에 대해 훈련하는 방법을 제시합니다.

- **Technical Details**: FastConformer 아키텍처를 사용하여 1억 개의 파라미터를 가진 모델을 60초 길이의 시퀀스로 훈련할 수 있으며, 전통적인 PnC(Partial punctuation and Capitalization) 방식에서는 성능 향상이 있었으나 40초 이상의 시퀀스에서 정확도가 정체되는 현상을 발견했습니다. 우리는 TDT-CTC 아키텍처를 채택하여 훈련하였으며, GTP-CTC 혼합 모델을 통해 더 빠른 수렴 속도를 이루었습니다.

- **Performance Highlights**: Earnings-21 및 Earnings-22 데이터 세트에서 단어 오류율(WER)이 25% 개선되었으며, MuST-C 테스트 세트에서 BLEU 점수 15% 향상이 있었습니다. 우리의 방법은 낮은 대문자 기준 평가 세트에서도 모델 정확도를 향상시켜 HF 리더보드에서 최고 성과를 기록했습니다.



### SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning (https://arxiv.org/abs/2409.05556)
- **What's New**: 이 연구는 자율적으로 과학적 이해를 진전시킬 수 있는 시스템의 개발을 목표로 하는 SciAgents 접근법을 제시합니다.

- **Technical Details**: (1) 대규모 온톨로지 지식 그래프(ontological knowledge graphs)를 활용하여 다양한 과학적 개념을 조직하고 연결합니다. (2) 대형 언어 모델(large language models, LLMs) 및 데이터 검색 도구(data retrieval tools) 모음을 사용합니다. (3) 현장 학습(in-situ learning) 기능을 갖춘 다중 에이전트 시스템(multi-agent systems)을 통합합니다.

- **Performance Highlights**: SciAgents는 생물 영감을 받은 재료 재료에 적용되어 새로운 학제 간 관계를 드러내며 기존의 연구 방법을 초월한 규모, 정밀도 및 탐사 능력을 달성합니다. 이 프레임워크는 자율적으로 연구 가설을 생성 및 정제하고, 기본 메커니즘 및 디자인 원칙을 설명하며, 예기치 않은 물질 속성을 밝혀냅니다.



### NLLB-E5: A Scalable Multilingual Retrieval Mod (https://arxiv.org/abs/2409.05401)
- **What's New**: 이번 논문에서는 다국어 정보 검색을 위한 혁신적인 모델인 NLLB-E5를 소개합니다. 이 모델은 다양한 언어에 대한 지원이 부족한 전통적인 방법들의 한계를 극복하기 위해, 여러 언어에 대한 다중 언어 학습 데이터 없이 제로샷(Zero-shot) 검색을 가능하게 합니다.

- **Technical Details**: NLLB-E5 모델은 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 서로 다른 언어의 단어와 구문을 공유 벡터 공간으로 맵핑하는 교차 언어 임베딩 모델, 2) 이 임베딩 벡터를 활용하여 여러 언어에서 관련 정보를 검색하는 검색 모델. 이 모델은 Knowledge-Distillation 기법을 활용하여 학습됩니다.

- **Performance Highlights**: 종합적인 기준 데이터세트인 Hindi-BEIR에서의 평가 결과, NLLB-E5는 다양한 언어 및 작업에서 강력한 성능을 보여주었습니다. 특히 저자원(低資源) 언어 지원을 위한 정보 검색의 정확도와 효율성에서 기존의 다국어 검색 모델을 초월하는 성과를 보였습니다.



### Mpox Narrative on Instagram: A Labeled Multilingual Dataset of Instagram Posts on Mpox for Sentiment, Hate Speech, and Anxiety Analysis (https://arxiv.org/abs/2409.05292)
- **What's New**: 이번 연구는 WHO가 국제적인 공중보건 비상사태로 선언한 mpox 발생과 관련된 인스타그램 게시물의 데이터셋을 최초로 개발했습니다. 이 데이터셋은 2022년 7월 23일부터 2024년 9월 5일 사이에 작성된 60,127개의 인스타그램 게시물을 포함하고 있으며, 52개 언어로 제공됩니다.

- **Technical Details**: 개발된 데이터셋은 각 게시물에 대한 Post ID, 게시물 설명, 게시물 날짜, 언어, 번역된 버전(구글 번역 API를 사용하여 영어로 번역됨)을 포함하는 여러 속성으로 구성되어 있습니다. 이후에는 감정 분석(sentiment analysis), 혐오 발언 탐지(hate speech detection), 불안 또는 스트레스 탐지(anxiety or stress detection)가 수행되었습니다.

- **Performance Highlights**: 감정 클래스의 분포는 두려움(fear) 27.95%, 놀람(surprise) 2.57%, 기쁨(joy) 8.69%, 슬픔(sadness) 5.94%, 분노(anger) 2.69%, 혐오(disgust) 1.53%, 중립(neutral) 50.64%로 나타났습니다. 혐오 발언 탐지 결과, 95.75%의 게시물이 혐오 발언을 포함하지 않았고, 나머지 4.25%만이 혐오 발언을 포함했습니다. 또한, 72.05%의 게시물은 불안/스트레스가 없었고, 27.95%의 게시물이 어떤 형태의 불안/스트레스를 나타냈습니다.



### Better Spanish Emotion Recognition In-the-wild: Bringing Attention to Deep Spectrum Voice Analysis (https://arxiv.org/abs/2409.05148)
- **What's New**: 본 연구는 감정 인식이 중요한 사회적 지원 로봇(Socially Assistive Robots, SAR)의 발전에 기여하는 것을 목표로 하며, 스페인어 음성 데이터셋인 ELRA-S0329와 EmoMatchSpanishDB를 분석하였습니다. 이 연구는 패럴랭귀지(paralanguage)와 딥러닝 기법인 DeepSpectrum을 활용하여 음성의 시각적 표현을 추출하고 사전 훈련된 CNN 모델에 입력합니다.

- **Technical Details**: DeepSpectrum 방법론은 오디오 트랙의 시각적 표현을 추출하고 이를 SVC(Support Vector Classifier) 또는 FC(Fully-Connected deep-learning classifier)에 입력하여 감정을 분류합니다. 연구에서는 Attention Mechanism을 기반으로 한 새로운 분류기인 DS-AM을 제안하였으며, ELRA-S0329과 EmoMatchSpanishDB 두 데이터셋에서 비교 실험을 수행했습니다.

- **Performance Highlights**: DS-AM 모델은 모든 SOTA(state-of-the-art) 모델과 비교하여 우수한 성능을 보였으며, 데이터셋 간의 훈련 및 테스트를 통해 모델의 편향성을 확인했습니다. 두 데이터셋 비교 결과, EmoMatchSpanishDB가 더 적절한 감정 인식 성능을 보였습니다.



### LLM-based Abstraction and Concretization for GUI Test Migration (https://arxiv.org/abs/2409.05028)
- **What's New**: 이 논문에서는 GUI 테스트 마이그레이션을 위한 새로운 패러다임인 abstraction-concretization paradigm을 제안합니다. 이 패러다임은 특정 기능에 대한 일반 테스트 로직을 추출한 후, 이를 기반으로 구체적인 GUI 테스트 케이스를 생성하는 방식입니다.

- **Technical Details**: 이 연구는 MACdroid라는 도구를 소개합니다. MACdroid는 소스 테스트 케이스에서 추출한 일반 테스트 로직을 활용하여 LLM(large language model)을 통해 타겟 앱의 GUI 테스트 케이스를 자동으로 생성합니다. 이 과정은 추상화 단계와 구체화 단계로 나뉩니다.

- **Performance Highlights**: MACdroid는 FrUITeR 데이터셋에서 타겟 기능의 64%를 성공적으로 테스트하였으며, 이는 기존 방법보다 191% 향상된 결과입니다. Lin 데이터셋에서도 75%의 타겟 기능을 성공적으로 테스트하여, 기본 성능 대비 42% 향상된 결과를 보였습니다.



### Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Detector (https://arxiv.org/abs/2409.05005)
Comments:
          Under review in ICASSP 2025

- **What's New**: 이 논문에서는 취약한 집단을 겨냥한 Patronizing and Condescending Language (PCL)의 첫 다중 모달 데이터셋인 PCLMM을 소개합니다. 이 데이터셋은 Bilibili에서 수집된 715개의 주석이 달린 비디오로 구성되어 있으며, PCL 인식을 위한 얼굴 표정 감지 모듈을 갖춘 MultiPCL 탐지기도 제안하였습니다.

- **Technical Details**: PCLMM 데이터셋은 21시간 이상의 비디오를 포함하고 있으며, 취약한 커뮤니티의 얼굴 표정과 비디오, 텍스트, 오디오의 특징을 통합하여 차별적인 언어의 탐지 정확도를 향상시키는 MultiPCL 탐지기를 사용합니다. PCLMM은 비디오 플랫폼에서 마이크로어그레션 탐지를 자동화하기 위한 작업을 지원합니다.

- **Performance Highlights**: PCLMM 데이터셋은 PCL 샘플의 독성 점수가 비-PCL 샘플보다 높으며, PCL 탐지기가 명확한 비율로 탐지 정확도를 향상시키는 것을 보여줍니다. PCL 비디오는 약 27.4%가 박해성으로 분류되었으며, 이는 인터넷 플랫폼에서 PCL 데이터의 분포와 일치합니다.



### InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inferenc (https://arxiv.org/abs/2409.04992)
- **What's New**: 이 논문에서는 LLM (Large Language Models) 추론 과정을 최적화하기 위해 새로운 시스템인 InstInfer를 제안합니다. 이 시스템은 매우 중요한 계산과 데이터 처리를 CSD (Computational Storage Drives)로 오프로드하여 KV 캐시의 전송 오버헤드를 최소화합니다.

- **Technical Details**: InstInfer는 특히 긴 시퀀스 추론을 위해 GPU와 CSD를 통합하여 수행합니다. 이 시스템은 데이터 이동의 병목 현상을 줄이기 위해 PCIe 피어 투 피어 DMA를 사용하며, 메모리 집약적인 디코딩 단계의 주 계산을 CSD로 옮깁니다. 또한 전용 flash-aware 저장 장치 내 계산 엔진 및 KV 캐시 접근을 위한 FTL (Flash Translation Layer) 설계를 제안합니다.

- **Performance Highlights**: 실험 결과, InstInfer는 NVIDIA A6000 GPU를 사용할 때 13B 모델에 대해 긴 시퀀스 추론의 처리량을 FlexGen과 비교하여 최대 11.1배 향상시킵니다.



### MILE: A Mutation Testing Framework of In-Context Learning Systems (https://arxiv.org/abs/2409.04831)
- **What's New**: 본 연구에서는 In-context Learning (ICL) 시스템에 대해 고품질 테스트 데이터를 특징화하는 mutation testing 프레임워크인 MILE을 제안합니다. 이 프레임워크는 ICL 테스트의 신뢰성과 품질을 평가하기 위한 새로운 시각과 기술적 방법을 도입합니다.

- **Technical Details**: MILE 프레임워크는 ICL 시스템에 특화된 mutation operators와 mutation scores를 제안합니다. 특히 우리는 demonstration-level 및 prompt-level mutators를 설계하고, 다양한 결함을 특성화할 수 있는 group-wise mutation score도 포함하였습니다. 이러한 접근 방식은 ICL 시스템에서의 신뢰성 문제를 해결하는 데 중점을 두고 있습니다.

- **Performance Highlights**: MILE 프레임워크는 여러 기준 데이터세트와 인기 있는 LLM을 사용하여 평가되었으며, 실험 결과는 우리의 mutation scores가 테스트 세트의 품질과 강한 상관관계를 가지며, ICL 테스트 슈트의 품질을 측정하는 데 효과적임을 입증했습니다.



### QueryBuilder: Human-in-the-Loop Query Development for Information Retrieva (https://arxiv.org/abs/2409.04667)
- **What's New**: 새로운 시스템인 QueryBuilder는 사용자가 많은 노력을 들이지 않고도 정보 검색 쿼리를 생성할 수 있도록 돕는 상호작용 시스템입니다. 이 시스템은 최소한의 노력으로 크로스링구얼 정보 검색 쿼리를 신속하게 개발할 수 있도록 설계되었습니다.

- **Technical Details**: QueryBuilder는 사용자가 영어 개발 코퍼스를 효율적으로 탐색하여 정보 검색 시스템이 처리할 수 있는 쿼리로 변환하는 데 도움을 줍니다. 시스템은 사용자가 검색어를 입력하여 문서를 검색하고, 관련 문장을 선택하여 쿼리의 정교함을 개선하도록 돕습니다. 이 과정은 반복하여 진행되며, 최종 결과로 다각적인 쿼리가 생성됩니다. 시스템은 신속한 확률적 정보 검색(IR) 시스템과 BERT 기반의 신경망 IR 시스템을 활용하여 사용자 피드백을 반영합니다.

- **Performance Highlights**: 실험 결과, 초보 사용자들은 최대 10분 내에 유용한 세부 쿼리를 개발할 수 있고, QueryBuilder에서 생성된 쿼리는 단순한 전반적인 작업을 기반으로 한 검색보다 약 12% 향상된 성능을 보였습니다.



### 3D Data Long-Term Preservation in Cultural Heritag (https://arxiv.org/abs/2409.04507)
- **What's New**: 이번 보고서는 문화유산의 3D 디지털 데이터 보존에 대한 도전 과제와 전략을 탐구합니다. 특히 기술의 노후화(technological obsolescence) 문제를 강조하며, 지속 가능한 저장 솔루션(sustainable storage solutions)과 지속적인 데이터 관리(data management) 전략의 필요성을 강조하고 있습니다.

- **Technical Details**: 주요 주제로는 기술적 노후화 이해, 디지털 콘텐츠의 생애주기(lifecycle of digital content), 디지털 연속성(digital continuity), 데이터 관리 계획(Data Management Plans, DMP), FAIR 원칙, 공공 저장소(public repositories)의 사용 등이 포함됩니다. 또한 장기 디지털 보존에서 메타데이터(metadata)의 중요성과 유용한 메타데이터 구축 전략을 다루며, 3D 형식 보존에서의 진화하는 표준과 상호 운용성(interoperability), 메타데이터와 파라데이터(paradata) 관리의 중요성을 논의합니다.

- **Performance Highlights**: 이 문서는 3D 문화 유산 데이터의 장기 보존을 위한 도전 과제와 솔루션에 대한 종합적인 개요를 제공합니다.



### WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermarks (https://arxiv.org/abs/2409.04459)
Comments:
          Work in Progress

- **What's New**: 이번 연구에서는 Embeddings-as-a-Service (EaaS)watermark 기술의 새로운 결점인 패러프레이징(paraphrasing) 공격을 소개하고, 이를 방어할 수 있는 새로운 watermarking 방법인 WET(Linear Transformation을 이용한 EaaS Watermarking)를 제안합니다.

- **Technical Details**: 기존의 EaaS watermarks가 패러프레이징에 의해 제거될 수 있음을 보여주고, WET는 원본 임베딩에 선형 변환(linear transformation)을 적용하여 watermark를 주입합니다. 이 방법은 패러프레이징 공격에 대해 이론적으로 및 경험적으로 강건함을 입증하였습니다.

- **Performance Highlights**: WET는 여러 번의 실험을 통해 거의 완벽한 검증 가능성을 보여주며, 하나의 샘플로도 효과적으로 Watermark를 검증할 수 있음을 입증하였습니다. 또한, 단순 선형 변환을 사용하여 임베딩의 유용성도 대부분 보존됩니다.



### Towards Generative Class Prompt Learning for Fine-grained Visual Recognition (https://arxiv.org/abs/2409.01835)
Comments:
          Accepted in BMVC 2024

- **What's New**: 이번 연구는 Generative Class Prompt Learning (GCPL)과 Contrastive Multi-class Prompt Learning (CoMPLe)이라는 두 가지 새로운 방법을 제안하여 기존의 VLM(Visual Language Model)들의 세밀한 분류 성능을 향상시킵니다. 이는 text-to-image diffusion 모델을 활용하여 적은 예제에 대한 클래스 프롬프트를 학습 가능한 형태로 조건화하는 방식입니다.

- **Technical Details**: GCPL은 클래스 임베딩에서 비주얼-언어 시너지를 크게 향상시키며, CoMPLe는 생성 최적화 과정에서 클래스 간 분리를 촉진하는 대조 학습(Contrastive Learning) 요소를 도입합니다. 이들 방식은 세밀한 이미지 인식의 도전 과제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 여기서 제안한 생성적 클래스 프롬프트 학습 접근법이 기존의 방법들을 상당히 초월하는 성능을 보여주었으며, 적은 샷(Few-shot) 이미지 인식 과제에서도 효과적인 대안이 될 수 있음을 입증했습니다.



New uploads on arXiv(cs.IR)

### Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation (https://arxiv.org/abs/2409.05633)
- **What's New**: 이번 연구에서는 GNN 기반의 협업 필터링에서 한계를 극복하기 위해 CoGCL이라는 새로운 그래프 대조 학습 프레임워크를 제안합니다. 이 프레임워크는 사용자의 아이템 상호작용 정보를 바탕으로 강력한 협업 정보를 담은 대조적 뷰를 생성하는 것을 목표로 합니다.

- **Technical Details**: CoGCL은 다층 벡터 양자화(multi-level vector quantizer)를 사용하여 사용자와 아이템의 표현을 이산 코드(discrete codes)로 양자화합니다. 이를 통해 이웃 구조(neighborhood structure) 및 의미적 관련성(semantic relevance)을 고려하여 대조적 뷰를 강화합니다. 가상 이웃 증강(virtual neighbor augmentation) 방법을 통해 대조적 뷰의 이웃 정보를 확장하고, 공통 이산 코드를 공유하는 사용자/아이템의 유사성을 활용하여 정보의 관련성을 증가시킵니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에서 실시한 실험 결과 CoGCL이 기존의 기초 모델들보다 일관되게 더 우수한 성능을 보였음을 입증하였습니다. 제안된 구성 요소가 그래프 대조 학습을 통해 추천을 개선하는 중요한 역할을 한다는 추가 분석도 수행되었습니다.



### Rs4rs: Semantically Find Recent Publications from Top Recommendation System-Related Venues (https://arxiv.org/abs/2409.05570)
- **What's New**: Rs4rs는 추천 시스템(Recommendation Systems) 관련 최신 논문들을 위해 semantic search(의미 기반 검색)를 수행하는 웹 애플리케이션입니다. 기존의 학술 검색 엔진들이 제공하는 광범위한 검색 결과의 단점을 보완하여 사용자가 관심 주제를 입력하면 최상위 회의 및 저널으로부터 최근의 관련 논문 목록을 제공하는 사용자 친화적인 플랫폼입니다.

- **Technical Details**: Rs4rs는 기존 추천 시스템 관련 회의 및 저널에서의 최근 논문들을 대상으로 인덱싱합니다. 최신 논문에 중점을 두면서, batch processing(배치 처리)과 online processing(온라인 처리)을 활용하여 신속하게 사용자 쿼리에 응답합니다. 다양한 언어 모델을 도입하여 모델 앙상블을 통해 결과 순위를 개선하고, SBERT(Sentence-BERT) 모델을 사용해 코사인 유사도의 평균을 통해 검색 결과를 순위 매깁니다.

- **Performance Highlights**: Rs4rs는 높은 품질의 소스 선택을 보장하여 가장 관련성이 높은 논문 결과를 제공합니다. 사용자는 맞춤형 필터링 옵션을 통해 특정 저널이나 회의에 따라 결과를 세분화할 수 있으며, 이는 연구자들이 가치 있는 콘텐츠를 발견하는 데 집중할 수 있도록 돕습니다. 또한, 사용자 피드백을 통해 지속적으로 개선할 수 있는 가능성을 가지고 있습니다.



### End-to-End Learnable Item Tokenization for Generative Recommendation (https://arxiv.org/abs/2409.05546)
- **What's New**: 새로운 생성 추천 패러다임인 ETEGRec이 제안되었으며, 이는 아이템 토크나이제이션(item tokenization)과 생성 추천(generative recommendation)을 통합하는 혁신적인 방법입니다.

- **Technical Details**: ETEGRec은 듀얼 인코더-디코더(dual encoder-decoder) 아키텍처를 기반으로 하며, Residual Quantization Variational Autoencoder (RQ-VAE)와 Transformer 모델을 사용합니다. 이 프레임워크는 두 가지 특정 최적화 목표인 시퀀스-아이템 정렬(sequence-item alignment)과 선호-의미 정렬(preference-semantic alignment)을 통해 두 구성 요소 간의 상호 강화를 달성합니다.

- **Performance Highlights**: 다양한 추천 벤치마크에서 ETEGRec의 성능을 검증하였으며, 전통적인 순차 추천 모델과 생성 추천 기준선에 비해 월등한 성능을 보여주었습니다.



### RBoard: A Unified Platform for Reproducible and Reusable Recommender System Benchmarks (https://arxiv.org/abs/2409.05526)
- **What's New**: RBoard라는 혁신적인 프레임워크가 추천 시스템 연구의 재현성과 알고리즘 비교의 부족한 표준화된 벤치마크 문제를 해결하기 위해 도입되었습니다. 이 플랫폼은 CTR 예측, Top-N 추천 등의 다양한 추천 작업을 벤치마킹할 수 있는 포괄적인 플랫폼을 제공합니다.

- **Technical Details**: RBoard는 여러 데이터세트에서 알고리즘을 평가하며, 실험에 대한 재현성과 재사용성을 보장하는 데 중점을 둡니다. 이를 위해, 데이터 처리 및 전처리, 작업별 평가, 사용자 코드 통합을 위해 표준화된 접근 방식을 사용합니다.

- **Performance Highlights**: RBoard는 공개된 리더보드를 통해 알고리즘 효과성을 다양한 데이터 상황에서 종합적으로 평가합니다. 연구자들은 최신 알고리즘과 비교하여 자신의 알고리즘을 평가할 수 있으며, 업계 실무자들은 필요에 맞는 최적의 알고리즘을 식별할 수 있습니다.



### Federated Transfer Learning Based Cooperative Wideband Spectrum Sensing with Model Pruning (https://arxiv.org/abs/2409.05462)
- **What's New**: 본 논문에서는 WSS(광대역 스펙트럼 센싱)를 위한 WSSNet이라는 신경망 구조를 제안합니다. 이는 멀티코셋(Multicoset) 전처리를 통해 서브-나이퀴스트 샘플링(Sub-Nyquist Sampling)을 가능하게 하여 높은 정확도로 스펙트럼 홀을 탐지합니다.

- **Technical Details**: WSSNet은 먼지 쉬운 배치 적응과 모델 추론을 위해 선택적 가중치 가지치기(Selective Weight Pruning) 전략을 통해 경량화되었습니다. 또한, 펠리케이트 전송 학습(FTL) 기반의 다중 사용자 연합 학습을 도입해 다양한 상황에서 강력한 성능을 유지할 수 있도록 하였습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 FTL-WSSNet은 로컬 적응 샘플 없이도 여러 목표 시나리오에서 좋은 성능을 보였습니다. 이는 SUs가 다양한 주파수 대역에서 활동하는 PUs와의 동적 활동 환경에서도 적용 가능함을 의미합니다.



### Recommender Systems Algorithm Selection for Ranking Prediction on Implicit Feedback Datasets (https://arxiv.org/abs/2409.05461)
Comments:
          Accepted for presentation at the 18th ACM Conference on Recommender Systems in the Late-Breaking Results Track

- **What's New**: 이 연구는 암묵적 피드백 데이터셋에 대한 추천 시스템 알고리즘 선택 문제를 처음으로 해결하기 위한 시도를 제안합니다. 전통적인 연구는 주로 명시적 피드백 데이터셋에 초점을 맞춰 랭킹 예측에 대한 연구가 부족했습니다.

- **Technical Details**: 총 72개의 추천 시스템 데이터셋에서 24개의 추천 알고리즘을 평가한 후, 각 알고리즘은 두 가지 하이퍼파라미터 구성으로 설정됩니다. 또한, 4개의 최적화된 기계 학습 메타 모델과 1개의 자동화된 기계 학습 메타 모델을 세 가지 설정으로 훈련하였습니다. 메타 모델의 예측은 기준 값과의 중위 스피어맨 상관계수가 0.857에서 0.918까지 나타났습니다.

- **Performance Highlights**: 최적화된 전통적인 메타 모델인 XGBoost는 48.6%의 재현율(Recall)을 보였고, 자동화된 기계 학습 메타 모델인 AutoGluon은 47.2%의 재현율로 뒤따랐습니다. 이는 전통적인 메타 학습 알고리즘이 자동화된 기계 학습 알고리즘보다 더 효과적일 수 있음을 시사합니다.



### Replicability Measures for Longitudinal Information Retrieval Evaluation (https://arxiv.org/abs/2409.05417)
Comments:
          Experimental IR Meets Multilinguality, Multimodality, and Interaction - 15th International Conference of the CLEF Association, CLEF 2024, Grenoble, France, September 9-12, 2024, Proceedings. arXiv admin note: text overlap with arXiv:2308.10549

- **What's New**: 이 연구는 정보 검색 (Information Retrieval, IR) 시스템의 지속적인 효과성을 측정하는 새로운 접근법을 제시합니다. 특히 LongEval 데이터셋을 사용하여 다양한 시간 지점에서 IR 시스템의 효과성이 어떻게 변화하는지 살펴봅니다.

- **Technical Details**: LongEval 공유 과제를 통해 IR 시스템의 시간적 지속성을 조사하며, 시스템의 효과성을 재현성 (replicability) 과제로 변환하여 효과성이 시간이 지남에 따라 어떻게 진화하는지를 분석합니다. 다양한 시스템의 효과성을 nDCG(normalized Discounted Cumulative Gain)와 같은 지표를 통해 평가하며, 결과 델타(Result Delta, ℛeΔ)와 같은 새로운 비교 전략을 도입합니다.

- **Performance Highlights**: 연구 결과, 가장 효과적인 시스템이 가장 지속적인 성능을 가진 시스템과 반드시 일치하지 않음을 발견하였습니다. 여러 비교 지표와 시간이 지남에 따라 시스템의 성능이 다양하게 나타나는 것으로 나타났습니다.



### NLLB-E5: A Scalable Multilingual Retrieval Mod (https://arxiv.org/abs/2409.05401)
- **What's New**: 이번 논문에서는 다국어 정보 검색을 위한 혁신적인 모델인 NLLB-E5를 소개합니다. 이 모델은 다양한 언어에 대한 지원이 부족한 전통적인 방법들의 한계를 극복하기 위해, 여러 언어에 대한 다중 언어 학습 데이터 없이 제로샷(Zero-shot) 검색을 가능하게 합니다.

- **Technical Details**: NLLB-E5 모델은 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 서로 다른 언어의 단어와 구문을 공유 벡터 공간으로 맵핑하는 교차 언어 임베딩 모델, 2) 이 임베딩 벡터를 활용하여 여러 언어에서 관련 정보를 검색하는 검색 모델. 이 모델은 Knowledge-Distillation 기법을 활용하여 학습됩니다.

- **Performance Highlights**: 종합적인 기준 데이터세트인 Hindi-BEIR에서의 평가 결과, NLLB-E5는 다양한 언어 및 작업에서 강력한 성능을 보여주었습니다. 특히 저자원(低資源) 언어 지원을 위한 정보 검색의 정확도와 효율성에서 기존의 다국어 검색 모델을 초월하는 성과를 보였습니다.



### A Survey on Diffusion Models for Recommender Systems (https://arxiv.org/abs/2409.05033)
Comments:
          Under Review

- **What's New**: 본 논문은 추천 시스템을 위한 확산 모델(diffusion models, DMs)에 대한 최초의 포괄적인 서베이를 제시하고 다양한 추천 파이프라인 관점에서 이 모델들이 어떻게 활용되는지를 조망합니다.

- **Technical Details**: 이 논문은 추천 시스템에서의 확산 모델을 세 가지 주요 도메인으로 체계적으로 분류합니다: (1) 데이터 공학 및 인코딩을 위한 확산 - 데이터 증강 및 표현 향상; (2) 추천 모델로서의 확산 - 사용자 선호도를 직접 추정하고 아이템을 순위 매기는 데 사용; (3) 콘텐츠 프리젠테이션을 위한 확산 - 패션 및 광고 창작물과 같은 개인화된 콘텐츠 생성.

- **Performance Highlights**: 확산 모델은 복잡한 데이터 분포를 캡처하고 사용자 선호도에 맞춘 고품질의 다양한 샘플을 생성하는 데 강력한 능력을 보이며, 추천 시스템의 성능을 크게 향상시키는데 기여합니다.



### Sequential Recommendation via Adaptive Robust Attention with Multi-dimensional Embeddings (https://arxiv.org/abs/2409.05022)
- **What's New**: 이 논문은 sequential recommendation 모델의 정확도를 높이기 위해 mix-attention 메커니즘과 layer-wise noise injection (LNI) 정규화를 도입한 adaptive robust sequential recommendation framework (ADRRec) 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 multi-dimensional kernel embedding을 통해 사용자의 행동 패턴을 캡처하고, 절대 및 상대적 mix-attention 메커니즘을 통해 각 사용자 행동의 고유 패턴을 학습합니다. noise injection regularization (NIR)을 통해 모델의 강인성과 일반화를 강화합니다.

- **Performance Highlights**: 네 개의 유명한 데이터셋을 이용한 실험 결과 제안된 ADRRec 모델이 기존 self-attention 아키텍처를 능가하는 성능을 보임을 입증하였습니다.



### Incorporate LLMs with Influential Recommender System (https://arxiv.org/abs/2409.04827)
Comments:
          5 pages, 1 figure

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)을 활용한 새로운 접근 방법인 LLM 기반 영향 경로 계획(LLM-IPP)을 제안합니다. 이는 사용자의 흥미를 효과적으로 유도하는 일관성 있는 추천 아이템(영향 경로) 시퀀스를 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: LLM-IPP는 사용자의 인구 통계학적 특징과 이전의 행동 데이터를 기반으로 LLM에게 지시를 제공하여 영향 경로를 생성합니다. 이 방법은 추천된 항목 간의 일관성을 유지하며, 각 항목이 인접한 항목과 관련성이 있도록 합니다. 또한, 경로의 마지막에 목표 항목을 포함하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, LLM-IPP는 전통적인 적극적인 추천 시스템보다 훨씬 더 나은 성능을 보였습니다. 특히 사용자 수용성과 경로 일관성에서 현저한 개선이 있었습니다.



### Debias Can be Unreliable: Mitigating Bias Issue in Evaluating Debiasing Recommendation (https://arxiv.org/abs/2409.04810)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구에서는 recommendation 모델의 평가에서 전통적인 평가 방식의 한계를 지적하고, Unbiased Recall Evaluation (URE) 스킴을 제안하여 랜덤하게 노출된 데이터셋을 통해 사용자 선호도의 정확한 Recall 성능을 추정할 수 있도록 합니다.

- **Technical Details**: 기존의 계산 방식인 Recall@K를 랜덤 노출 데이터셋에서 사용하는 것은 부적절하며, URE 스킴은 모델의 예측 점수를 정렬한 후 특정 기준 값(K+1)-th 항목의 점수를 사용하여 Recall@K를 unbiased하게 추정합니다.

- **Performance Highlights**: URE 스킴은 랜덤 노출 데이터셋의 긍정적인 비율을 사용하여 Fully Exposed Dataset에서 Recall@K의 의도된 값을 비편향적으로 추정할 수 있음을 이론적 및 실험적으로 검증하며, 이는 기존의 방법들보다 더 신뢰할 수 있는 결과를 제공합니다.



### QueryBuilder: Human-in-the-Loop Query Development for Information Retrieva (https://arxiv.org/abs/2409.04667)
- **What's New**: 새로운 시스템인 QueryBuilder는 사용자가 많은 노력을 들이지 않고도 정보 검색 쿼리를 생성할 수 있도록 돕는 상호작용 시스템입니다. 이 시스템은 최소한의 노력으로 크로스링구얼 정보 검색 쿼리를 신속하게 개발할 수 있도록 설계되었습니다.

- **Technical Details**: QueryBuilder는 사용자가 영어 개발 코퍼스를 효율적으로 탐색하여 정보 검색 시스템이 처리할 수 있는 쿼리로 변환하는 데 도움을 줍니다. 시스템은 사용자가 검색어를 입력하여 문서를 검색하고, 관련 문장을 선택하여 쿼리의 정교함을 개선하도록 돕습니다. 이 과정은 반복하여 진행되며, 최종 결과로 다각적인 쿼리가 생성됩니다. 시스템은 신속한 확률적 정보 검색(IR) 시스템과 BERT 기반의 신경망 IR 시스템을 활용하여 사용자 피드백을 반영합니다.

- **Performance Highlights**: 실험 결과, 초보 사용자들은 최대 10분 내에 유용한 세부 쿼리를 개발할 수 있고, QueryBuilder에서 생성된 쿼리는 단순한 전반적인 작업을 기반으로 한 검색보다 약 12% 향상된 성능을 보였습니다.



### A Unified Framework for Cross-Domain Recommendation (https://arxiv.org/abs/2409.04540)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 Cross-Domain Recommendation (CDR) 기술을 활용하여 데이터 부족(data-sparsity) 및 콜드 스타트(cold-start) 문제를 해결하는 새로운 모델 UniCDR+을 제안합니다. 기존의 SOTA(State-Of-The-Art) 모델인 UniCDR을 확장하여 다양한 시나리오에서의 적응력을 향상시켰습니다.

- **Technical Details**: UniCDR+ 모델은 5가지의 서로 다른 측면에서 개선되었습니다. 특징 조작(feature engineering) 및 item-item 다중-hop 신호를 추가하고, 다양한 크기의 아이템 지원을 처리하기 위해 여러 효율적인 상호 작용 시퀸스 집합기(aggregators)를 제공합니다. 또한, 도메인 공유 표현(domain-shared representation)을 배우기 위해 부드러운 목표(soft objective)를 도입하며, MMoE 스타일의 예측 모듈을 포함시켰습니다.

- **Performance Highlights**: UniCDR+은 Kuaishou Living-Room Fullrank 프로세스에 성공적으로 배포되어 매일 수백만 명의 활성 사용자들에게 서비스를 제공합니다. 이 모델은 5가지 CDR 시나리오에서 철저한 테스트를 진행하였고, 기존의 다양한 SOTA 작업과 비교했을 때 독특한 전이 능력(unique transferring ability)을 입증하였습니다.



### Benchmarking Chinese Knowledge Rectification in Large Language Models (https://arxiv.org/abs/2409.05806)
Comments:
          Ongoing work; code and dataset are available at this https URL

- **What's New**: 본 연구에서는 중국어에 특화된 대규모 언어 모델(LLMs)의 지식을 수정하기 위한 벤치마크인 새로운 데이터셋 CKnowEdit를 소개합니다. 이 데이터셋은 고전 문헌, 속담, 이디엄 등 7종의 지식을 포함하고 있으며, 중국어의 독특한 언어적 특성을 반영하여 수집되었습니다.

- **Technical Details**: CKnowEdit는 1,760개의 사례를 포함하며, 고전 시가, 속담, 이디엄, 음성 표기, 고전 중국어, 지리 지식, 그리고 Ruoziba라는 7개의 중국어-specific 종류의 지식을 조직하고 수집합니다. 현재의 지식 수정 방법을 평가하기 위해, 단어 수준의 중첩 및 의미 벡터 유사성을 사용하여 수정된 모델의 성능을 평가합니다.

- **Performance Highlights**: 상태-of-the-art 지식 수정 기법들의 실증 결과를 통해, 중국 문헌에 적용했을 때의 한계를 드러냈으며, 이는 미래의 더 정교한 중국어 지식 수정 접근 방식의 필요성을 강조합니다.



### Extracting the U.S. building types from OpenStreetMap data (https://arxiv.org/abs/2409.05692)
- **What's New**: 이 논문에서는 미국 전역의 주거 및 비주거 건물 분류를 포함한 포괄적인 건물 데이터셋을 구축하였으며, 비지도 학습 방법을 활용하여 이 데이터를 생성했습니다.

- **Technical Details**: 주된 데이터 출처로는 OSM(OpenStreetMap)을 사용하였으며, 건물 발자국(building footprints)과 관련된 태그(tags) 정보를 통해 건물 분류를 수행했습니다. 이 방법론은 OSM의 건물 정보와 그 태그들, 그리고 POI(Point-Of-Interest)와 같은 보조 데이터를 결합하여 비주거 건물과 주거 건물을 구분합니다. 검증을 위해 미니애폴리스 및 세인트폴과 같은 특정 카운티의 공식 데이터를 활용했습니다.

- **Performance Highlights**: 결과적으로 총 67,705,475 개의 건물이 분류되었으며, 비주거 건물 분류에서 높은 정밀도(precision)를 얻었고, 주거 건물에서 높은 재현율(recall)을 달성했습니다. 또한, 자신의 방법론을 통해 OSM의 메타데이터 부족으로 인한 분류 오류를 분석하고, 더 나은 분류 품질을 위한 접근 방식도 제시하였습니다.



### RegNLP in Action: Facilitating Compliance Through Automated Information Retrieval and Answer Generation (https://arxiv.org/abs/2409.05677)
- **What's New**: 본 논문은 RegNLP(Regulatory Natural Language Processing) 분야에 기여하며, 자동화된 질문-단락 생성(Automated Question-Passage Generation) 작업을 정의하고, 27,869개의 질문을 포함한 ObliQA 데이터셋을 만들고, 규제 정보 검색 및 답변 생성 시스템을 설계하여 성능을 평가합니다.

- **Technical Details**: RegNLP는 규제 규칙과 의무의 접근과 해석을 단순화하기 위한 다학제 분야입니다. 오히려 복잡한 규제 문체에서 정확하게 정보를 추출하기 위한 자동화된 질문-단락 생성 프레임워크를 도입하고, 자연어 추론(Natural Language Inference, NLI)을 검증 단계에 통합하였습니다. ObliQA 데이터셋은 금융 규제 문서에서 파생된 질문과 관련 단락들을 포함합니다. RePASs 평가 메트릭은 생성된 답변이 모든 관련 의무를 정확히 반영하는지 평가합니다.

- **Performance Highlights**: 이 연구는 RegNLP의 데이터셋 및 평가 메트릭의 한계를 넘어선 포괄적이고 간결한 답변 생성이 가능하다는 점에서 향후 규제 문서의 이해 및 접근성을 크게 향상시킬 수 있습니다. 특히, 잘 설계된 질문 생성 및 정보 검색 단계는 규정 준수 오류를 줄이고, 운영 효율성을 증대시킬 수 있음을 보여줍니다.



### DatAasee -- A Metadata-Lake as Metadata Catalog for a Virtual Data-Lak (https://arxiv.org/abs/2409.05512)
- **What's New**: 이번 연구는 분산 데이터 소스에 대한 메타데이터 관리를 개선하기 위해 메타데이터-레이크(metadata-lake)라는 새로운 데이터 아키텍처를 제안합니다. 이는 기존의 데이터 레이크(data lake) 개념에서 파생된 것으로, 연구 데이터와 도서관 환경에서의 메타데이터 중심 시스템을 구축합니다.

- **Technical Details**: 메타데이터-레이크는 다양한 출처의 메타데이터를 중앙집중화하여 일관된 형식으로 제공하는 시스템입니다. 이는 데이터 파이프라인(data pipelining) 기술을 통해 이루어지며, 데이터 해양(data swamp)을 지양하기 위해 메타데이터 카탈로그가 저장된 데이터를 메타데이터로 색인합니다. 이를 통해 인터디서플리너리(data discovery)와 FAIR(Findable, Accessible, Interoperable, Reusable) 원칙을 지원하게 됩니다.

- **Performance Highlights**: 제안된 메타데이터-레이크 시스템인 DatAasee는 연구 데이터의 메타데이터를 집중화하고, 이를 통해 연구 데이터의 접근성과 재사용성을 높이는데 기여합니다. 또한, 이 시스템은 다양한 메타데이터 변경의 추적과 현재 메타데이터 기록의 집계를 가능하게 하며, 여러 응용 프로그램에 대한 사용 사례와 혜택을 명확히 제시하고 있습니다.



### A Survey of Multimodal Composite Editing and Retrieva (https://arxiv.org/abs/2409.05405)
Comments:
          22 pages, 3 figures, and 11 tables

- **What's New**: 이번 설문조사는 다중 모달 복합 검색(multimodal composite retrieval)에 대한 포괄적인 리뷰를 제공하는 첫 번째 논문으로, 이미지-텍스트 복합 편집 및 검색 방법을 심도 있게 탐구합니다. 다중 모달 데이터 유형의 통합을 통해 개인화된 정보 검색의 개선을 꾀하고 있습니다.

- **Technical Details**: 다중 모달 복합 검색은 텍스트, 이미지, 오디오 등 다양한 모달리티를 통합하여 더 정확하고 맞춤형 결과를 제공합니다. 최근에는 Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) 네트워크, Visual Transformer (ViT) 등을 사용하여 이미지 검색 성능을 향상시키는 방법들이 제안되고 있습니다.

- **Performance Highlights**: 다중 모달 복합 검색 방법들은 사용자 질의와 맥락을 더 잘 이해함으로써 검색 성능과 사용자 만족도를 개선하는 데 기여하고 있습니다. 다양한 모달리티를 활용하는 이러한 기술들은 지난 몇 년간 많은 연구가 진행되어오며, 특히 패션 산업, 의료 진단 등 여러 분야에 활발히 응용되고 있습니다.



### OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs (https://arxiv.org/abs/2409.05152)
Comments:
          Work in progress; code is available at this https URL

- **What's New**: 이번 논문에서는 기존의 Large Language Models (LLMs)에서 발생하는 생성(generation)과 검색(retrieval) 작업의 통합 문제를 해결하기 위해 새로운 One-pass Generation and retrieval 프레임워크인 OneGen을 소개합니다. 이 프레임워크는 생성과 검색을 통합하여 동시에 처리할 수 있도록 설계되었습니다.

- **Technical Details**: OneGen 프레임워크는 autoregressively 생성된 retrieval tokens를 포함하여 전통적인 생성과 검색의 훈련 방식을 결합합니다. 이를 통해 단일 LLM이 통합된 forward pass 내에서 두 가지 작업을 동시에 처리할 수 있게 됩니다. 우리는 RAG와 Entity Linking이라는 두 가지 복합 작업에서 OneGen의 pluggability, 효과성, 효율성을 검증하기 위한 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 생성과 검색을 같은 맥락에서 통합하면 LLMs의 생성 능력을 유지하면서도 검색 성능이 향상됨을 확인했습니다. 또한, OneGen은 LLMs가 생성 중에 벡터 검색(vector retrieval)을 수행할 수 있도록 최초로 가능하게 하였습니다.



### Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models (https://arxiv.org/abs/2409.04701)
Comments:
          4 pages, early draft

- **What's New**: 이 논문에서는 'late chunking'이라는 새로운 방식의 텍스트 조각 인코딩 방법을 제안합니다. 기존의 방식에서는 문서의 텍스트를 작은 조각으로 나누어 개별적으로 인코딩하는 것에서 발생하는 컨텍스트 정보 손실 문제를 해결합니다.

- **Technical Details**: Late chunking 방법은 긴 텍스트 임베딩 모델을 활용하여 먼저 문서의 모든 토큰을 임베딩한 후, Mean Pooling 적용 직전에 조각으로 나누는 방식입니다. 이로 인해 전체 텍스트의 컨텍스트 정보를 보존하며, 이를 통해 텍스트 내의 의미와 관련성을 더 잘 캡처할 수 있습니다.

- **Performance Highlights**: Late chunking은 다양한 검색 작업에서 기존의 조각화 방법보다 우수한 성능을 보였으며, 추가적인 학습 없이도 다양한 긴 컨텍스트 임베딩 모델에 적용할 수 있는 범용성을 가지고 있습니다.



### Preserving Individuality while Following the Crowd: Understanding the Role of User Taste and Crowd Wisdom in Online Product Rating Prediction (https://arxiv.org/abs/2409.04649)
Comments:
          Preprint

- **What's New**: 이 논문은 온라인 제품 평점 예측에서 사용자 및 제품 정보의 구체적인 영향을 탐구하며, 개별 사용자 취향이 집단의 지혜보다 더 중요한 요소임을 밝혔습니다.

- **Technical Details**: 연구는 동적 트리 표현을 사용하여 사용자 및 제품 수준에서의 역사적인 평점을 캡슐화하고, 이를 통해 사용자와 제품의 시간적 역동성을 효과적으로 포착합니다. 이 방법은 콜드 스타트 문제를 자연스럽게 해결하고, 확장성과 배포 용이성을 제공합니다.

- **Performance Highlights**: 다양한 모델 타입에서 개별 사용자 취향이 집단의 지혜를 초월하는 경향이 지속적으로 확인되었으며, 이는 온라인 제품 평점 예측의 정확성을 높이는 데 중요한 시사점을 제공합니다.



### Understanding Fairness in Recommender Systems: A Healthcare Perspectiv (https://arxiv.org/abs/2409.03893)
Comments:
          Accepted to the 18th ACM Conference on Recommender Systems

- **What's New**: 이 논문은 의료 추천 시스템에서의 공정성(fairness)에 대한 대중의 이해를 탐구하며, 다양한 공정성 지표를 통해 이 개념이 어떻게 이해되고 있는지 조사합니다. 결과적으로, 공정성은 복잡하고 오해를 받기 쉬운 개념이라는 사실을 밝혀냈습니다.

- **Technical Details**: 연구에서는 Demographic Parity, Equal Accuracy, Equalized Odds, Positive Predictive Value의 네 가지 공정성 지표를 사용하여 참가자들이 이해하는 공정성을 평가하였습니다. 참가자는 의료 시나리오에 따라 적합한 공정성 지표를 선택하게 됩니다.

- **Performance Highlights**: 연구 결과, 대중은 공정성 개념에 대한 이해도가 낮고, 상황의 맥락에 따라 인식이 달라진다는 것을 보여줍니다. 특히, 공정성에 대한 단일 접근 방식은 충분치 않으며, 맥락에 적합한 설계가 필요하다는 것을 강조했습니다.



New uploads on arXiv(cs.CV)

### Promptable Closed-loop Traffic Simulation (https://arxiv.org/abs/2409.05863)
Comments:
          Accepted to CoRL 2024. Website available at this https URL

- **What's New**: 이 논문에서는 ProSim이라는 새로운 다중모드 프롬프트 가능 폐쇄형 트래픽 시뮬레이션 프레임워크를 제안합니다. ProSim은 사용자가 각 에이전트의 행동 및 의도를 지시하기 위해 복잡한 수치적, 범주적 또는 텍스트 프롬프트를 제공할 수 있게 하며, 폐쇄형 방식으로 트래픽 시나리오를 전개합니다.

- **Technical Details**: ProSim은 장면 초기화와 다중모드 프롬프트를 입력으로 받아, 에이전트의 정책 집합을 생성하고, 폐쇄형으로 반응형 시나리오 롤아웃을 생성합니다. 이렇게 생성된 프롬프트는 ProSim-Instruct-520k 데이터셋에 포함되어 있으며, 이 데이터셋은 520K paired 프롬프트-시나리오의 세트를 포함하고 있습니다.

- **Performance Highlights**: ProSim은 다양한 사용자 프롬프트를 제공할 때 높은 조작 가능성을 보여주며, 프롬프트가 없는 경우에도 Waymo Sim Agents Challenge에서 경쟁력 있는 성과를 달성합니다.



### Evaluating Multiview Object Consistency in Humans and Image Models (https://arxiv.org/abs/2409.05862)
Comments:
          Project page: https:/tzler.github.io/MOCHI/ Code: this https URL Huggingface dataset: this https URL

- **What's New**: 이 연구에서는 인간 관찰자와 컴퓨터 비전 모델 간의 정렬(alignment)을 평가하는 새로운 벤치마크를 소개합니다. 3D 물체 추론(task) 작업을 위해, 피험자들은 여러 이미지 세트에서 어떤 이미지들이 동일한 물체인지 구별합니다.

- **Technical Details**: 실험 설계는 'zero-shot visual inference'를 기반으로 하며, 피험자들은 동일한 물체를 포함하는 이미지를 식별해야 합니다. 다양한 물체와 추상적 형태의 2000개 이상의 고유한 이미지 세트를 구성하고 35,000회의 행동 데이터를 500명 이상의 참가자로부터 수집했습니다. 주요 비전 모델(DINOv2, MAE, CLIP)의 성능을 평가했습니다.

- **Performance Highlights**: 연구 결과, 인간은 모든 모델에 비해 현저히 우수한 성능을 보였습니다. 다중 스케일 평가 접근법을 사용하여 모델과 인간 간의 유사성과 차이점을 확인했습니다. 인간은 특히 어려운 시험에서 더 많은 시간과 처리를 할당하는 경향이 있습니다.



### LSVOS Challenge Report: Large-scale Complex and Long Video Object Segmentation (https://arxiv.org/abs/2409.05847)
Comments:
          ECCV 2024 LSVOS Challenge Report: this https URL

- **What's New**: 이번 논문에서는 2024 ECCV 워크숍과 함께 실시되는 6번째 대규모 비디오 객체 분할(LSVOS) 챌린지를 소개합니다. 이 챌린지는 비디오 객체 분할(Video Object Segmentation, VOS)과 언급된 비디오 객체 분할(Referring Video Object Segmentation, RVOS) 두 가지 작업을 포함합니다. 기존의 YouTube-VOS 데이터셋을 MOSE, LVOS, MeViS와 같은 최신 데이터셋으로 대체하여 복잡한 환경에서의 VOS 성능을 평가합니다.

- **Technical Details**: VOS는 비디오 프레임에서 특정 객체 인스턴스를 분할하는 문제를 다루며, MOSE 데이터셋은 2,149개의 비디오와 5,200개의 객체에 대한 주석을 포함하고 있습니다. LVOS 데이터셋은 평균 1.14분 길이의 720개 시퀀스를 포함합니다. RVOS는 언어 표현을 바탕으로 비디오에서 객체를 분할하며, MeViS 데이터셋은 2,006개의 비디오와 8,171개의 객체에 대한 주석을 제공합니다. 각 챌린지 트랙은 Jaccard 및 F-measure 메트릭을 사용하여 성능을 평가합니다.

- **Performance Highlights**: 올해 챌린지에는 8개국 20개 이상의 기관에서 129팀이 등록하여 참여했으며, 최종 상위 6개 해결책이 도출되었습니다. VOS 트랙에서 최고의 팀은 yahooo, yuanjie, Sch89.89로, 각각 𝒥&ℱ 최고 점수를 기록했습니다. 더불어 새로운 방법론과 영상 이해의 발전을 위해 이 작업들이 이바지했습니다.



### Vision-Driven 2D Supervised Fine-Tuning Framework for Bird's Eye View Perception (https://arxiv.org/abs/2409.05834)
- **What's New**: 이 논문에서는 LiDAR에 의존하지 않고, 2D 영상 정보를 활용하여 BEV(Top-down, Bird's Eye View) 모델을 세밀하게 조정하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 이 방법은 주위 카메라에서 수집된 2D 정보를 기반으로 BEV 모델을 최적화하므로, 다양한 복잡한 시나리오에 대한 적응력이 향상됩니다. BEVFormer 프레임워크를 사용하여 2D 주석과 3D 인식 결과 간의 정밀한 매칭을 수행하는 손실 함수를 설계하였습니다. 3D 결과는 BEV 쿼리를 통해 공간 및 시간적으로 상호 작용하며, 객체의 위치, 범주, 속도 및 방향을 예측합니다.

- **Performance Highlights**: nuScenes 및 Waymo와 같은 다양한 공개 데이터셋에서 실시한 실험 결과, 제안된 방법으로 조정된 BEV 모델이 전통적인 방법에 비해 다양한 시나리오에서 우수한 성능을 보이고, 자율주행 응용 분야에서 큰 잠재력을 가진 것으로 나타났습니다.



### GASP: Gaussian Splatting for Physic-Based Simulations (https://arxiv.org/abs/2409.05819)
- **What's New**: 최근 연구에서 물리 기반의 시뮬레이션을 위한 Gaussian Splatting for Physics-Based Simulations (GASP) 모델이 제안되었습니다. 이 모델은 기존의 Gaussian Splatting(GS)을 수정 없이 활용하며, 물리 엔진과의 통합이 용이합니다.

- **Technical Details**: GASP는 3D Gaussian 분포를 사용하여 시뮬레이션을 수행하며, 물체의 지점들을 독립된 개체로 간주합니다. 이는 Material Point Method (MPM)와 결합하여 Flat Gaussian 분포를 통해 구현됩니다. GASP는 추가적인 메싱 전략 없이 Gaussian 분포를 직접 처리합니다.

- **Performance Highlights**: GASP 모델은 다양한 3D 객체 렌더링을 위한 벤치마크 데이터셋에서 우수한 성능을 보여주며, 정적 및 동적 장면을 효과적으로 시뮬레이션할 수 있습니다.



### VFA: Vision Frequency Analysis of Foundation Models and Human (https://arxiv.org/abs/2409.05817)
- **What's New**: 본 연구는 대규모 컴퓨터 비전 모델의 다양한 특성이 인간 인지와의 정합성 및 견고성에 미치는 영향을 조사하였습니다. 연구 결과, 모델 및 데이터 크기 증가와 풍부한 의미 정보를 포함하며 다양한 모달리티를 결합할 경우 모델이 인간 인지에 더 잘 정렬되고 전반적인 견고성이 향상된다는 것을 발견하였습니다.

- **Technical Details**: 이 연구에서는 1200개 이상의 판별 모델을 분석하며, 인간과 모델 간의 OOD(Out-Of-Distribution) 정확도 및 형태 편향 사이의 관계를 평가하였습니다. 더불어 주파수 대역폭과 모델 크기 간의 관계를 분석하여, 대역폭이 감소하면 OOD 정확도가 증가하는 반비례 관계를 발견했습니다. 주요 연구 방법론으로는 주파수 대역 분석과 Gaussian curve 피팅을 활용했습니다.

- **Performance Highlights**: 결과적으로, 특정 모델(예: BEiTv2, CoAtNet, ConvFormer 등)은 인간과 유사한 성능을 보였으며, 일부 모델은 인간 성능을 초과하기도 하였습니다. 특히, 대역폭이 작은 모델들은 OOD 정확도가 높은 경향을 보였으며, 이로 인해 대역폭이 OOD 일반화에 있어 중요한 예측 변수라는 점이 강조되었습니다.



### Leveraging Object Priors for Point Tracking (https://arxiv.org/abs/2409.05786)
Comments:
          ECCV 2024 ILR Workshop

- **What's New**: 이 논문에서는 Point Tracking을 위한 혁신적인 Objectness 정규화 기법을 제안하여 포인트가 대상 물체의 경계 내에 머물도록 유도하고, 이로 인해 장기 추적 성능의 향상을 도모합니다.

- **Technical Details**: 제안된 방법은 훈련 시 Objectness Loss를 도입하여 각 포인트가 속한 객체의 속성을 인식하도록 하며, Testing 시 객체 마스크를 계산하지 않고도 Spatial Continuity를 직접 통합하는 방식입니다. 또한, Contextual Attention을 활용하여 포인트 추적 중 각 지역의 Feature Representation을 향상시킵니다.

- **Performance Highlights**: 우리의 방식은 PointOdyssey, TAP-Vid-DAVIS, CroHD의 세 가지 벤치마크에서 최신 기술(features)인 방법들을 초월하는 성능을 보여주며, Objectness 정규화 기법을 통해 추론 시간 동안 추가적인 계산 비용 없이 효율성을 유지합니다.



### ReL-SAR: Representation Learning for Skeleton Action Recognition with Convolutional Transformers and BYOL (https://arxiv.org/abs/2409.05749)
Comments:
          8 pages, 4 figures, 6 tables

- **What's New**: 본 연구에서는 비지도(self-supervised) 표현 학습을 통해 스켈레톤 행동 인식(skeleton action recognition) 기술을 연구합니다. 이를 위해 매우 가벼운 Convolutional Transformer 프레임워크인 ReL-SAR(Representation Learning for Skeleton Action Recognition)를 개발하였으며, 합성곱(convolution) 및 주의(attention) 레이어의 보완적 특성을 활용하여 스켈레톤 시퀀스의 공간적(spatial) 및 시간적(temporal) 힌트를 공동으로 모델링합니다.

- **Technical Details**: ReL-SAR 모델은 스켈레톤 관절의 Selection-Permutation 전략을 사용하여 더 많은 정보가 포함된 설명을 제공합니다. Bootstrap Your Own Latent(BYOL) 접근법을 통해 레이블이 없는 스켈레톤 시퀀스 데이터로부터 강력한 표현을 학습합니다. 이 모델은 YOLOv5x를 사용하여 사람을 감지하고 ViTPose를 통해 자세를 추정하여 스켈레톤 시퀀스를 생성합니다.

- **Performance Highlights**: 제안된 ReL-SAR 모델은 MCAD, IXMAS, JHMDB, NW-UCLA와 같은 제한된 크기 데이터셋에서 매우 경쟁력 있는 결과를 달성하였으며, 성능과 계산 효율성 모두에서 최첨단(state-of-the-art) 방법에 대해 우수성을 입증하였습니다.



### Boosting CNN-based Handwriting Recognition Systems with Learnable Relaxation Labeling (https://arxiv.org/abs/2409.05699)
Comments:
          26 pages, 3 figures

- **What's New**: 이 논문에서는 필기 인식(handwriting recognition) 시스템을 위한 새로운 접근법을 제안합니다. 이 방식은 Relaxation Labelling (RL) 프로세스와 다양한 신경망(neural architectures)의 강점을 통합하며, 알고리즘의 수렴(convergence)을 가속화하고 전체 시스템의 성능을 향상시키는 희소화(sparsification) 기법을 도입합니다.

- **Technical Details**: RL 프로세스는 컨텍스트에 맞는 라벨 지정 문제를 해결하고, 변분 불평등(variational inequality) 및 게임 이론(game theory)에 기초한 이론적 기초를 가지고 있습니다. RL 프로세스는 정보의 상호 작용(message-passing)을 통해 문맥 내에서 최적의 라벨링을 결정하며, 하이브리드 네트워크인 CRNN, FCN, GCNN 등과 결합되어 사용됩니다.

- **Performance Highlights**: 여러 데이터셋을 대상으로 한 실험 결과, RL 프로세스는 일반화 능력을 향상시킬 수 있으며, 일부 경우에는 최신 transformer 기반 아키텍처를 초월하는 성능을 보였습니다.



### Segmentation by Factorization: Unsupervised Semantic Segmentation for Pathology by Factorizing Foundation Model Features (https://arxiv.org/abs/2409.05697)
- **What's New**: 논문에서 제안한 Segmentation by Factorization (F-SEG) 방법은 기존의 단일 데이터셋에 대해 전용으로 훈련된 세그멘테이션 모델 없이, 사전 훈련된 심층 신경망을 활용하여 병리학적 이미지의 세그멘테이션 마스크를 생성하는 비지도 학습 방법입니다.

- **Technical Details**: F-SEG는 Non-Negative Matrix Factorization (NMF)를 사용하여 심층 신경망에서 추출된 공간 활성화를 세그멘테이션 마스크와 개념 피처로 나누어 식별합니다. H&E 이미지에 대한 클러스터링 모델을 훈련하여 일반적인 조직 표현형을 생성하고, 이 클러스터를 활용하여 세그멘테이션 마스크를 분해합니다.

- **Performance Highlights**: F-SEG는 H&E 병리학적 이미지에 대해 강력한 비지도 세그멘테이션 능력을 제공하며, 병리학 기초 모델을 활용함으로써 세그멘테이션 품질이 크게 향상되는 결과를 보여줍니다.



### LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow (https://arxiv.org/abs/2409.05688)
Comments:
          Accepted to ECCV 2024

- **What's New**: LayeredFlow라는 새로운 벤치마크 데이터세트를 도입하여 비-Lambertian(non-Lambertian) 객체의 다층 3D 이해를 위한 경향성과 성능을 개선했습니다. 이 데이터세트는 고품질의 광학 흐름(optical flow) 및 스테레오 쌍(stereo pairs) 주석을 포함하고 있으며, 다층 3D 주석을 갖춘 실제 환경을 기반으로 구축되었습니다.

- **Technical Details**: LayeredFlow는 185개의 실내 및 실외 환경에서 360개의 독창적인 객체로부터 수집된 150k개의 고품질 광학 흐름 주석을 포함하는 실세계 다층 벤치마크입니다. AprilTag 시스템을 활용하여 투명 장면에서의 다층 3D 지오메트리를 캡처하며, 신경망 학습을 위한 대규모 합성 데이터셋(60k 이미지, 30장면)을 제공합니다.

- **Performance Highlights**: 기존의 광학 흐름 방법을 LayeredFlow와 함께 미세 조정(fine-tuning)했을 때 비-Lambertian 객체에 대한 성능이 향상되었으며, 과거의 데이터세트를 사용한 경우보다 훨씬 나은 결과를 얻었습니다. 또한 새로운 다층 광학 흐름 추정 작업을 제안하였고, RAFT 기반의 기초 방법을 제공하여 비-Lambertian 객체 인식을 위한 길잡이 역할을 합니다.



### SX-Stitch: An Efficient VMS-UNet Based Framework for Intraoperative Scoliosis X-Ray Image Stitching (https://arxiv.org/abs/2409.05681)
- **What's New**: 이번 논문에서는척추 측만증 수술 중 사용되는 C-arm X선 기계의 제한된 시야(FOV)를 극복하기 위한 효율적이고 견고한 수술 중 X선 이미지 스티칭 방법인 SX-Stitch를 제안합니다. 이 방법은 세분화(segmentation) 및 스티칭(stitching)의 두 단계로 나뉘며, 강력한 세분화 모델인 Vision Mamba of Spine-UNet(VMS-UNet)을 사용하여 직렬 이미지의 보다 나은 스티칭을 실현합니다.

- **Technical Details**: 세분화 단계에서는, VMS-UNet을 통해 이미지에서 중요한 구조적 정보를 캡처하며, 이 모델은 SimAM 주의(attention) 메커니즘을 포함하여 세분화 성능을 크게 향상시킵니다. 스티칭 단계에서는 레지스트레이션 에너지 함수를 최소화하는 방식으로 이미지 정렬을 단순화하여 비정렬된 이미지를 정렬하고, 하이브리드 에너지 함수를 도입하여 최적의 이음새(seam)를 최적화합니다.

- **Performance Highlights**: SX-Stitch는 임상 데이터셋에서 기존 최첨단 기법들(SOTA)보다 질적 및 양적으로 우수한 성능을 보이며, 특히 척추 수술 전반에 걸쳐 실질적인 도움을 줄 수 있는 가능성을 보여줍니다.



### AnomalyCD: A benchmark for Earth anomaly change detection with high-resolution and time-series observations (https://arxiv.org/abs/2409.05679)
Comments:
          remote sensing benchmark

- **What's New**: 이번 연구에서는 고해상도 원격 감지 이미지를 위한 새로운 비정상 변화 감지 기술인 AnomalyCD(Anomaly Change Detection)를 제안합니다. 이 기술은 시간 시계열 관측치를 받아들이고, 과거의 정상 변화 패턴을 학습하여 비정상 변화를 식별합니다.

- **Technical Details**: AnomalyCD는 고정된 비정상 범주에 의존하지 않고, 다양한 비정상을 통합적으로 로컬라이즈할 수 있는 능력이 있습니다. 이를 위해 AnomalyCDD라는 고해상도 시계열 이미지 데이터셋을 구축했습니다. 또한, 제로샷(baseline) 모델인 AnomalyCDM을 개발하여, SAM(Segment Anything Model)으로부터 일반적인 표현을 추출하고 시간적 비교를 수행합니다.

- **Performance Highlights**: AnomalyCDM은 기존 기술과 비교하여 비정상 이미지들을 직접 처리할 수 있는 능력을 가지고 있으며, 각 장면을 위해 재교육 없이 효율적으로 작동합니다.



### Real-Time Human Action Recognition on Embedded Platforms (https://arxiv.org/abs/2409.05662)
- **What's New**: 이 논문은 컴퓨터 비전과 딥러닝 기술의 발전으로 비디오 기반 인간 행동 인식(HAR)이 실용화된 가운데, 실시간 퍼포먼스 문제를 해결하기 위한 4가지 기여를 제안합니다.

- **Technical Details**: 하나, 기존의 Optical Flow(OF) 추출 기술이 최신 HAR 파이프라인의 지연 병목임을 확인하는 실험적 연구, 둘, 전통적인 방식과 딥러닝 기반 OF 추출 방법 간의 지연-정확도 트레이드오프를 탐구하는 것, 셋, 효율성과 정확성을 충족하는 Integrated Motion Feature Extractor(IMFE) 설계, 넷, 임베디드 플랫폼에 최적화된 실시간 HAR 시스템 RT-HARE의 개발입니다.

- **Performance Highlights**: Nvidia Jetson Xavier NX 플랫폼에서 RT-HARE는 30 FPS의 비디오 프레임 속도로 실시간 HAR을 처리하고 높은 인식 정확도를 유지하는 성능이 입증되었습니다.



### Replay Consolidation with Label Propagation for Continual Object Detection (https://arxiv.org/abs/2409.05650)
- **What's New**: 이 논문에서는 객체 탐지(Object Detection) 분야에서의 지속 학습(Continual Learning, CL) 문제를 다루며, 레플레이 방법론의 한계를 개선하기 위해 새로운 기술인 리플레이 통합과 라벨 전파(Replay Consolidation with Label Propagation, RCLPOD)를 제안합니다.

- **Technical Details**: RCLPOD는 레플레이 메모리의 샘플을 개선하여, 클래식 배포 최적화(Optimizing Class Distribution in Memory, OCDM) 및 라벨 전파(Label Propagation) 기법을 통합하여 작업 간 간섭 문제(task interference issue)를 줄이는 방식을 따릅니다. 또한, 마스킹 손실(Masking Loss) 기술과 피처 증류(Feature Distillation) 기법을 사용하여 모델의 효과성을 극대화합니다.

- **Performance Highlights**: 이 방법은 VOC 및 COCO와 같은 기존 CLOD 벤치마크에서 다른 방법론에 비해 우수한 성능을 보여주며, YOLOv8 모델에서 테스트되었습니다. 이는 현재의 YOLO 아키텍처와 CLOD 프레임워크에서의 연구를 위한 중요한 기준점을 제공합니다.



### Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification (https://arxiv.org/abs/2409.05642)
Comments:
          7 pages

- **What's New**: 본 논문에서는 Prototype-Driven Multi-feature Generation framework (PDM)을 제안하여 가시적-적외선 인물 재식별(VI-ReID)에서의 모드 간 차이를 줄이는 방법을 소개합니다.

- **Technical Details**: PDM은 Multi-Feature Generation Module (MFGM)과 Prototype Learning Module (PLM)으로 구성됩니다. MFGM은 다양한 피처를 생성하여 두 모드 간 유사하게 분포하도록 하며, PLM은 학습 가능한 프로토타입을 활용하여 가시적(VIS)과 적외선(IR) 모드 간의 잠재적 의미적 유사성을 발굴하여 모드 정렬을 촉진합니다. 또한, cosine heterogeneity loss를 도입하여 프로토타입 다양성을 강화합니다.

- **Performance Highlights**: SYSU-MM01 및 LLCM 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 최신 성능을 보여주었으며, 코드 또한 공개되었습니다.



### 3D-SAR Tomography and Machine Learning for High-Resolution Tree Height Estimation (https://arxiv.org/abs/2409.05636)
- **What's New**: 이 연구는 Synthetic Aperture Radar (SAR) 기술을 활용하여 숲의 생물량을 정확히 추정하는 방법을 제공합니다. 특히, Tomographic SAR (TomoSAR) 데이터를 사용하여 숲의 높이를 측정하는 기계를 학습하는 새로운 접근법을 제시합니다.

- **Technical Details**: 연구에서는 SAR 이미지와 LiDAR 데이터를 결합하여 숲의 구조를 3D 모델링합니다. TomoSense 데이터세트를 활용하여 L 및 P 밴드의 SAR 데이터를 사용하고, 다양한 기계 학습 기법, 특히 deep learning을 위한 3D U-Net 아키텍처를 사용합니다. 또한, 공간의 자율 상관관계를 분석하기 위해 다양한 지리적 분할 방법을 평가합니다.

- **Performance Highlights**: 연구의 최종 결과, 모델은 30m 높이의 수관에서 평균 절대 오차(Mean Absolute Error)가 2.82m로 측정되어, 지구의 탄소 저장량을 정확히 측정하고 기후 행동을 지원하는 능력이 향상되었습니다.



### Renormalized Connection for Scale-preferred Object Detection in Satellite Imagery (https://arxiv.org/abs/2409.05624)
Comments:
          24 pages, 14 figures Journal

- **What's New**: 본 논문에서는 Knowledge Discovery Network (KDN)을 설계하여 작은 객체 탐지에서 효율적인 피처 추출을 구현하는 방법을 제안합니다. KDN의 renormalized connection (RC)을 통해 다중 규모 피처의 'synergistic focusing'을 가능하게 합니다.

- **Technical Details**: KDN은 단일 브랜치 탐지기에 적용되어 서로 다른 단계의 피처 사이의 관계를 찾고, 각 단계의 기능을 탐지 작업에 맞게 조정합니다. n21C라는 클래스의 renormalized connections를 일반화하여 FPN 기반의 다중 브랜치 탐지기에 적용합니다. 이 방법을 통해 FPN에서 발생하는 간섭 활성화를 줄이고, 학습 방향을 올바르게 설정할 수 있습니다.

- **Performance Highlights**: 17개의 다양한 탐지 아키텍처에 n21s를 삽입하여 실시한 광범위한 실험 결과, E421C가 모든 작업에서 우수한 성능을 보였고, RGT의 스케일 속성을 만족시키는 것을 확인하였습니다. 이 연구는 컴퓨터 비전에서 설계된 탐지기를 원거리 감지 커뮤니티에 효과적으로 이전할 수 있는 가능성을 보여줍니다.



### G-NeLF: Memory- and Data-Efficient Hybrid Neural Light Field for Novel View Synthesis (https://arxiv.org/abs/2409.05617)
- **What's New**: 이번 논문에서는 G-NeLF라는 최신 경량 NeLF 접근법을 제안합니다. G-NeLF는 공간 인식(feature-aware) 그리드를 혼합하여 R2L의 훈련 어려움을 극복할 수 있는 방법을 제시합니다.

- **Technical Details**: G-NeLF는 Ray의 색상을 예측하기 위해 공간 인식 피쳐 시퀀스를 사용하며, 0.95 MB의 모델 크기로 100뷰의 데이터만으로 훈련이 가능합니다. 또한, LSTM 기반의 Ray 색상 디코더를 디자인하여 Ray 전파 과정에 따라 효율적으로 색상을 예측합니다.

- **Performance Highlights**: G-NeLF는 기존의 NeLF 방법들보다 성능이 뛰어나며, 100뷰와 0.95 MB 모델 크기로 R2L보다 0.04dB 뛰어난 성능을 보입니다. 격자 기반 NeRF 방법들(Benchmark Instant-NGP)과 비교하여 1/10의 파라미터만 사용하지만 더 높은 성능을 달성합니다.



### Adapted-MoE: Mixture of Experts with Test-Time Adaption for Anomaly Detection (https://arxiv.org/abs/2409.05611)
- **What's New**: 본 논문에서는 Adapted-MoE라는 새로운 방법을 제안하여, 같은 범주 내에서의 다양한 샘플 분포를 다루는 문제를 해결합니다. 이 방법은 여러 전문가 모델과 라우팅 네트워크를 사용하여 다양한 정상 샘플의 피처(Feature) 분포에 대응하여 독립적인 결정 경계를 생성합니다.

- **Technical Details**: Adapted-MoE는 라우팅 네트워크를 통해 같은 범주 샘플을 서브클래스 피처 공간으로 라우팅하고, 여러 전문가 모델이 다양한 정상 샘플의 표현을 학습하여 여러 독립적인 결정 경계를 형성합니다. 또한 테스트 시간에 적응(Test-Time Adaption)을 통해 미지의 정상 샘플 표현과 전문가 모델이 학습한 피처 분포 간의 편향을 제거합니다.

- **Performance Highlights**: 실험 결과, Adapted-MoE는 Texture AD benchmark에서 I-AUROC와 P-AUROC에서 각각 2.18%-7.20% 및 1.57%-16.30%의 성능 향상을 보여주었으며, 기존 최신 방법들보다 우수한 성능을 기록했습니다.



### CustomContrast: A Multilevel Contrastive Perspective For Subject-Driven Text-to-Image Customization (https://arxiv.org/abs/2409.05606)
- **What's New**: CustomContrast라는 새로운 프레임워크를 제안하며, 기존의 self-reconstructive 관점의 한계를 극복하기 위해 cross-differential 관점을 기반으로 한다.

- **Technical Details**: CustomContrast는 Multimodal Feature Injection (MFI) Encoder와 Multilevel Contrastive Learning (MCL) 패러다임을 포함하여, 고수준 의미에서 저수준 외양까지의 주제 내적 속성을 추출한다. CSCL과 MACL을 통해 intra-consistency와 inter-distinctiveness를 보장한다.

- **Performance Highlights**: CustomContrast는 기존 방법보다 텍스트 제어 가능성을 3.8% 및 5.4% 각각 향상시키고, 주제 유사성(E-DI)을 5.9% 및 2.4% 향상시키는 성능을 보여준다.



### SynMorph: Generating Synthetic Face Morphing Dataset with Mated Samples (https://arxiv.org/abs/2409.05595)
- **What's New**: 본 연구에서는 2450개의 아이디와 10만 개 이상의 변형된 이미지를 포함한 합성 얼굴 변형 데이터셋을 생성하는 새로운 방법을 제안합니다. 이 데이터셋은 고품질 샘플, 다양한 변형 알고리즘 및 단일 및 차별 변형 공격 탐지 알고리즘에 대한 일반화를 특징으로 합니다.

- **Technical Details**: 연구팀은 StyleGAN 2 모델을 사용하여 1024 × 1024 해상도의 고품질 합성 얼굴 이미지를 생성합니다. 이를 통해 감정 표현과 조명 조건을 조정하여 다양한 변형 이미지를 생성하며, GAN 기반과 랜드마크 기반의 변형 알고리즘을 결합하여 샘플을 생성합니다. 합성 데이터는 S-MAD와 D-MAD 두 경우를 모두 지원합니다.

- **Performance Highlights**: 검증 작업을 통해 제안된 합성 데이터셋은 기존 SOTA 합성 데이터셋과 비교하여 향상된 성능을 보여주며, 비오메트릭 샘플 품질 및 얼굴 인식 시스템에 대한 변형 공격 가능성을 평가하였습니다. 연구 결과는 MAD 알고리즘의 훈련 및 평가에서 새로운 가능성을 열어줍니다.



### DSDFormer: An Innovative Transformer-Mamba Framework for Robust High-Precision Driver Distraction Identification (https://arxiv.org/abs/2409.05587)
- **What's New**: 이번 연구에서는 DSDFormer라는 새로운 프레임워크를 제안하여 드라이버 분산 탐지의 정확도를 높이고, Temporal Reasoning Confident Learning (TRCL)이라는 비지도 방식으로 노이즈 레이블을 개선했습니다.

- **Technical Details**: DSDFormer는 Transformer와 Mamba 아키텍처의 장점을 통합한 Dual State Domain Attention (DSDA) 메커니즘을 통해 긴 거리 의존성 및 상세한 기능 추출 간의 균형을 유지할 수 있도록 설계되었습니다. TRCL은 비디오 시퀀스의 시공간 상관관계를 활용하여 노이즈가 있는 레이블을 정제하는 방식입니다.

- **Performance Highlights**: 우리 모델은 AUC-V1, AUC-V2 및 100-Driver 데이터셋에서 최신 성능을 기록하였으며, NVIDIA Jetson AGX Orin 플랫폼에서 실시간 처리 효율성을 보여주었습니다. 이 연구를 통해 DSDFormer와 TRCL이 드라이버 분산 탐지의 정확성과 견고성을 크게 향상시킬 수 있음을 입증하였습니다.



### Latent 3D Brain MRI Counterfactua (https://arxiv.org/abs/2409.05585)
- **What's New**: 이번 논문에서는 고차원 데이터에서 인과성을 정확하게 모델링하는 데 어려움이 있는 기존의 대체 모델들 대신, 구조적 인과 모델 (Structural Causal Model, SCM)을 사용하여 고해상도 MRI 이미지를 생성하는 새로운 2단계 방법을 제안합니다.

- **Technical Details**: 우리의 접근법은 첫째 단계에서 VQ-VAE (Vector Quantized Variational Autoencoder)를 사용하여 MRI 볼륨의 압축된 임베딩을 학습합니다. 이후 인과 모델을 latent space에 통합하고, 일반화 선형 모델 (Generalized Linear Model, GLM)을 사용하여 세 단계의 반사실적 절차 (counterfactual procedure)를 수행합니다.

- **Performance Highlights**: 실제 고해상도 MRI 데이터(1mm)에 대한 실험 결과, 제안된 방법이 높은 품질의 3D MRI 반사실적 이미지를 생성할 수 있음을 입증했습니다.



### LEROjD: Lidar Extended Radar-Only Object Detection (https://arxiv.org/abs/2409.05564)
Comments:
          Accepted for publication as ECCV 2024

- **What's New**: 이 논문에서는 3D 객체 감지(3D object detection)의 성능을 향상시키기 위한 두 가지 전략을 제시합니다. 기존의 LiDAR 데이터의 이점을 활용해 3+1D 이미징 레이더(sensor) 객체 감지 모델을 개선하는 방법을 탐구합니다.

- **Technical Details**: 제안된 전략은 1) LiDAR 포인트 클라우드(thin-out) 기법을 활용한 다단계 훈련(multi-stage training) 및 2) 크로스 모달 지식 증류(cross-modal knowledge distillation)입니다. 다단계 훈련 과정에서는 세 가지 thin-out 방법을 비교 분석하였습니다.

- **Performance Highlights**: 다단계 훈련을 통해 평균 정밀도(mean Average Precision)가 최대 4.2% 향상되었으며, 지식 증류를 통해 최대 3.9%의 성능 향상을 기록했습니다. 이 접근 방식은 다른 3D 객체 감지 네트워크에도 적용 가능함을 입증했으며, 코드가 해당 URL에서 제공됩니다.



### Seeing Through the Mask: Rethinking Adversarial Examples for CAPTCHAs (https://arxiv.org/abs/2409.05558)
Comments:
          Under review

- **What's New**: 이 논문은 머신 비전 모델이 CAPTCHAs에서 이미지를 혼란스럽게 만드는 새로운 방법을 제안합니다. 기존의 고유한 노이즈 추가 방식 대신, 다양한 강도의 마스크를 추가하여 인간에게는 인식 가능하지만 머신에는 인식되지 않도록 하는 방식으로써, 최신 이미지 분류기를 속일 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 CAPTCHAs의 새로운 형태로 hCaptcha의 마스크 신호를 사용하여 이미지에 대한 공격을 평가합니다. 연구는 CAPTCHAs와 관련된 공격을 위해 인간의 시각 인식을 활용하고, 이미지의 의미 정보를 보존하는 방법을 탐구합니다. 또한, ConvNeXt, EVA, ViT와 같은 최신 비전 모델을 사용하여 다양한 이미지 필터 적용 후 정확도가 얼마나 감소하는지를 측정합니다.

- **Performance Highlights**: 모든 모델에서 Accuracy @ 1 (Acc@1)이 50% 이상 감소하며, 특히 비전 변환기(vision transformers)에서는 80% 감소하는 결과를 보였습니다. 이러한 발견은 최신 비전 모델들이 여전히 인간의 인식 능력을 따라잡지 못하고 있음을 강조합니다.



### Seeing is Believing? Enhancing Vision-Language Navigation using Visual Perturbations (https://arxiv.org/abs/2409.05552)
Comments:
          5 pages, 2 figures, submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 다양한 시각 정보를 통합하기 위해 Multi-Branch Architecture (MBA)를 제안합니다. 이 구조는 원본 RGB 관측치에 과적합(overfitting)되는 것을 방지하면서도, 다양한 노이즈가 섞인 시각 정보를 탐색하고 활용합니다.

- **Technical Details**: MBA는 여러 가지 서로 다른 시각 입력을 처리하는 다중 가지(branch) 구조로, 각 가지는 서로 다른 시각 입력을 처리합니다. 이 연구에서는 ground-truth depth images, 시각적으로 불일치한 뷰와 통합된 입력, 무작위 노이즈가 주입된 입력의 세 가지 변형을 도입하여 실험하였습니다.

- **Performance Highlights**: 폭넓은 실험 결과, 제안된 방법은 REVERIE, R2R 및 SOON 데이터셋에서 기존 최첨단(State-of-the-art) 결과와 동등하거나 이를 초과하는 성과를 보였습니다. 허용되는 과적합이지 묵시적으로 시각적 입력이 풍부해져서 일반화를 개선했다는 것이 확인되었습니다.



### Exploring Rich Subjective Quality Information for Image Quality Assessment in the Wild (https://arxiv.org/abs/2409.05540)
- **What's New**: 이 논문에서는 MOS(mean opinion score) 이상의 주관적 품질 정보인 SOS(standard deviation of opinion scores) 및 DOS(distribution of opinion scores)를 활용하여 이미지 품질을 평가하는 새로운 방법론 RichIQA를 제안합니다.

- **Technical Details**: RichIQA는 (1) Convolutional vision Transformer (CvT)의 강력한 특징 표현 능력을 활용하고, 인간의 단기 및 장기 기억 메커니즘을 모방하는 3단계의 이미지 품질 예측 네트워크를 채택하였으며, (2) MOS, SOS 및 DOS와 같은 다양한 주관적 품질 정보를 동시에 사용하여 품질 예측 네트워크를 훈련하는 다중 레이블 훈련 전략을 특징으로 합니다.

- **Performance Highlights**: RichIQA는 대규모의 실제 IQA 데이터베이스에서 최신 경쟁 모델들보다 우수한 성능을 발휘하며, 주관적 품질 평가를 완전히 활용하여 네트워크의 예측 성능과 일반화 능력을 향상시킵니다. RichIQA의 코드는 GitHub에서 공개될 예정입니다.



### HMAFlow: Learning More Accurate Optical Flow via Hierarchical Motion Field Alignmen (https://arxiv.org/abs/2409.05531)
Comments:
          11 pages, 6 figures

- **What's New**: 본 연구에서 제안된 HMAFlow는 Hierarchical Motion Field Alignment (HMA) 모듈과 Correlation Self-Attention (CSA) 모듈을 통해 작은 물체가 포함된 복잡한 장면에서의 optical flow 추정 성능을 향상시키는 새로운 방법입니다. 다중 스케일 상관 관계 검색(Multi-Scale Correlation Search, MCS) 레이어를 통해 4D 비용 볼륨을 재구성함으로써 기존 방법보다 효과적으로 경량 물체의 움직임을 캡처합니다.

- **Technical Details**: HMAFlow 모델은 두 개의 핵심 모듈로 구성됩니다: HMA 모듈은 다중 스케일 모션 특징을 통합하고 CSA 모듈은 글로벌 모션 특징의 신뢰성과 강건성을 향상시킵니다. 비용 볼륨 계산에서 평균 풀링 대신 MCS 레이어를 사용하여 동적 검색 범위를 통해 현재 모션 특징을 검색합니다. 이는 기존 RAFT와는 다른 접근 방식으로, 전반적인 흐름 필드의 정확한 추정을 위한 전역 의존성을 더 잘 모델링할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, HMAFlow는 Sintel 온라인 벤치마크에서 RAFT에 비해 상대적으로 14.2% 및 3.4%의 오류 감소를 달성하며, KITTI 벤치마크에서는 각각 RAFT와 GMA에 대해 6.8% 및 7.7%의 우위를 보입니다. 이러한 성과는 제안된 모델의 효과성과 우수성을 증명합니다.



### An Atmospheric Correction Integrated LULC Segmentation Model for High-Resolution Satellite Imagery (https://arxiv.org/abs/2409.05494)
- **What's New**: 이번 연구는 세가지 주요사항을 다루고 있습니다: 첫째, 대기 보정을 위한 고해상도 다중 스펙트럼 이미지(CARTOSAT-3 MX)를 사용하여 토지 이용 및 토지 피복(LULC) 분류의 정확성을 향상시켰습니다. 둘째, Cross Pseudo Supervision(CPS) 접근 방식을 통해 적게 라벨된 데이터의 정확성을 개선하였습니다. 셋째, 깊이 학습 모델을 통한 LULC 세분화의 안정성을 입증하였습니다.

- **Technical Details**: 본 연구에서는 다중 스펙트럼 이미지의 대기 보정을 위해 Look-Up Table 기반의 방사 전이 시뮬레이션을 수행하였습니다. 이를 통해 대기 경로 반사 및 투과율을 추정하였으며, 그런 다음 보정된 표면 반사 데이터를 지도 학습(supervised) 및 반지도 학습(semi-supervised) 세분화 모델에 사용했습니다. 이를 통해 여러 클래스(건물, 도로, 나무, 수역)의 분류 정확성을 비교했습니다. 사용된 모델에는 U-Net 아키텍처와 DeepLab v3+가 포함됩니다.

- **Performance Highlights**: 고해상도 멀티스펙트럼(CARTOSAT-3 MX) 이미지의 대기 보정 후, 각 도시의 토지 이용 분류의 다중 클래스 정확성이 우수한 결과를 보였으며, 특히 라벨이 부족한 데이터 상황에서도 모델 안정성이 확보되었습니다.



### PVP-Recon: Progressive View Planning via Warping Consistency for Sparse-View Surface Reconstruction (https://arxiv.org/abs/2409.05474)
- **What's New**: PVP-Recon은 스파스 뷰(sparse view)에서 최적의 관찰 지점을 계획하여 고품질의 3D 메쉬를 재구성하는 혁신적인 방법입니다. 기존의 스파스 뷰 재구성 방식과 달리, 입력 이미지의 미리 정해진 집합을 사용하지 않고, 필요한 이미지를 점진적으로 추가하면서 재구성을 수행합니다.

- **Technical Details**: PVP-Recon은 두 가지 주요 모듈로 구성됩니다: (1) 정보가 가장 많은 관측 지점을 동적으로 식별하는 뷰 계획 모듈(view planning module)과 (2) 추가된 관측을 활용하여 재구성 품질을 점진적으로 개선하는 신경면 재구성 모듈(neural surface reconstruction module)입니다. 뷰 계획은 'warping score'라는 새로운 척도를 사용하여, 각 관측 방법의 정보 이득을 반영합니다.

- **Performance Highlights**: 정량적 및 정성적 실험에서 PVP-Recon은 제한된 입력 예산 내에서 고품질 재구성을 달성하였으며, 기존의 기준선(baselines)을 초월하는 성과를 보였습니다. 평균적으로, PVP-Recon은 8장의 이미지 이하로도 객체를 성공적으로 재구성할 수 있습니다.



### Proto-OOD: Enhancing OOD Object Detection with Prototype Feature Similarity (https://arxiv.org/abs/2409.05466)
Comments:
          14pages

- **What's New**: 최근 연구에서 OOD (Out-of-Distribution) 객체 탐지의 정확도를 높이기 위한 Proto-OOD라는 새로운 네트워크 아키텍처가 제안되었습니다. 이 모델은 프로토타입 학습을 활용하여 입력 특징과 프로토타입의 유사성을 평가합니다.

- **Technical Details**: Proto-OOD는 다양한 클래스에 대한 특징 벡터를 수집하여 프로토타입으로 사용하며, 대조 손실(contrastive loss)을 통해 프로토타입의 대표성을 향상시킵니다. 또한, negative embedding generator를 활용하여 부정 임베딩을 생성하고, 이를 유사성 모듈을 훈련시키는 데 사용합니다.

- **Performance Highlights**: Proto-OOD는 MS-COCO 데이터셋에서 FPR95를 현저히 낮추고, Pascal VOC 데이터셋에서는 더 높은 mAP를 달성합니다. 기존 평가 메트릭의 한계를 분석하고, False Positive를 필터링하여 성과를 재평가하는 개선된 평가 프로토콜을 제안하였습니다.



### DriveScape: Towards High-Resolution Controllable Multi-View Driving Video Generation (https://arxiv.org/abs/2409.05463)
- **What's New**: 이 논문은 DriveScape라는 새로운 생성 프레임워크를 제안하여 다중 뷰 비디오 생성을 위한 3D 조건 유도를 수행합니다. 이 방법은 기존 방식의 단점을 극복하여, 실시간으로 고해상도 및 고프레임 비디오를 생성할 수 있는 효율적인 프로세스를 보장합니다.

- **Technical Details**: DriveScape는 Bi-Directional Condition Alignment 모듈을 통해 3D 도로 구조 정보를 효과적으로 정렬시키며, 카메라 데이터를 통합하여 공간-시간적 범위를 포괄적으로 커버합니다. 이는 빠른 처리 과정에서 고급 해상도와 고프레임 비디오 생성을 가능하게 합니다. DriveScape는 LDM 파이프라인을 기반으로 하며, 다중 뷰 프레임의 세밀한 제어를 허용합니다.

- **Performance Highlights**: nuScenes 데이터셋에서 최고 수준의 성능을 달성하였으며, FID 점수 8.34, FVD 점수 76.39를 기록하여 생성 품질이 우수합니다. 또한 다양한 인식 작업에서 뛰어난 성능을 보이며, 자율 주행 기술 향상에 기여할 것으로 보입니다.



### EndoOmni: Zero-Shot Cross-Dataset Depth Estimation in Endoscopy by Robust Self-Learning from Noisy Labels (https://arxiv.org/abs/2409.05442)
- **What's New**: 본 연구에서는 EndoOmni를 소개합니다. EndoOmni는 내시경을 위한 제로샷 크로스 도메인 깊이 추정의 첫 번째 기초 모델입니다.

- **Technical Details**: EndoOmni는 대규모 레이블이 있는 데이터와 레이블이 없는 데이터를 활용하여 학생 모델을 훈련시키는 고급 자기 학습(self-learning) 패러다임을 정제하는 방법론을 제안합니다. 또한, 레이블 신뢰도(estimated confidence)에 기반하여 학습 가중치를 조정하는 가중치 조정 학습 손실을 도입하여 노이즈가 있는 레이블로 인한 훈련 방해를 해결합니다.

- **Performance Highlights**: 제로샷 상대 깊이 추정에서 EndoOmni는 의료 영상 분야에서 기존 방법보다 41%, 기존 기초 모델보다 25% 개선된 성능을 보여주었습니다. 또한, 메트릭 깊이 추정으로 미세 조정(fine-tuning)했을 때도 인 도메인 및 아웃 오브 도메인 시나리오에서 우수한 성능을 유지합니다.



### TextToucher: Fine-Grained Text-to-Touch Generation (https://arxiv.org/abs/2409.05427)
- **What's New**: 이번 연구에서는 문자 기반의 텍스처에서 접촉 감각으로의 변환(Text-to-Touch generation) 문제를 탐구하여, 기존의 비주얼 기반 방법들보다 더 나은 결과를 얻을 수 있다는 점을 보여주는 첫 번째 연구이다.

- **Technical Details**: 이 논문에서는 접촉 이미지를 객체 수준(tactile texture, tactile shape)과 센서 수준(gel status)의 두 가지 세분화로 분석하며, 이를 통해 고품질 접촉 샘플을 생성하기 위해 정교한 문서 기반의 생성을 제안하는 TextToucher 방법을 소개한다. 멀티모달 대규모 언어 모델을 활용하여 객체 수준의 텍스트 정보를 구축하고, 학습 가능한 텍스트 프롬프트를 통해 센서 수준의 정보를 대표하도록 설계하였다. 여기에서 Diffusion Transformer 아키텍처 내에서 다양한 이중 세분화 텍스트 조건화 방법들을 탐색한다.

- **Performance Highlights**: 모든 실험 결과에서 TextToucher 방법의 우수성이 입증되었으며, 고급 텍스트 설명을 통해 효과적으로 고품질의 접촉 이미지를 생성할 수 있음을 보여주었다.



### Distribution Discrepancy and Feature Heterogeneity for Active 3D Object Detection (https://arxiv.org/abs/2409.05425)
Comments:
          Accepted to CoRL 2024

- **What's New**: 이 논문에서는 LiDAR 기반 3D 객체 탐지의 성능을 개선하기 위한 새로운 능동 학습(Active Learning, AL) 방법인 DDFH(Distribution Discrepancy and Feature Heterogeneity)를 제안합니다. DDFH는 기하학적 특성과 모델 포함 정보를 동시에 고려하여, 학습 시 데이터 비용을 최소화할 수 있도록 설계되었습니다.

- **Technical Details**: DDFH 방법은 세 가지 주요 요소로 구성되어 있습니다. 첫째, Distribution Discrepancy를 통해 레이블이 없는 데이터와 레이블이 있는 데이터의 분포 차이를 평가함으로써 효율적인 학습이 가능합니다. 둘째, Feature Heterogeneity는 내부 프레임 인스턴스의 특성을 다양하게 유지하여 중복 데이터의 생성을 방지합니다. 마지막으로, Quantile Transform을 이용하여 다양한 지표를 통합하여 정보의 균일한 측정을 제공합니다.

- **Performance Highlights**: 다양한 실험 결과에서 DDFH는 KITTI와 Waymo 데이터셋에서 기존의 SOTA(state-of-the-art) 방법을 초월하여, 바운딩 박스 주석 비용을 56.3% 줄이고, 동일한 데이터 양으로 3D mAP에서 평균 1.8%의 성능 향상을 도달했습니다. 또한, DDFH는 1단계 및 2단계 탐지 모델 모두에서 우수한 일반화 성능을 보였습니다.



### AD-Net: Attention-based dilated convolutional residual network with guided decoder for robust skin lesion segmentation (https://arxiv.org/abs/2409.05420)
- **What's New**: 본 연구는 피부 암 진단 및 치료를 위한 컴퓨터 지원 진단 도구에서 피부 병변의 정확한 세분화(segmentation)를 다룸. 이 논문에서는 dilated convolutional residual network를 활용하여 공간적 특징을 향상시키는 attenzione 기반의 ASFEB(block)와 guided decoder 전략을 제안한다.

- **Technical Details**: 연구의 주요 기법은 다양한 dilation rate을 사용하는 dilated convolution을 통해 receptive field를 확장하는 residual block 기반의 신경망인 AD-Net이다. ASFEB는 average 및 maximum pooling 작업으로부터 얻은 feature map을 결합하여 공간적 feature 정보를 개선하며, guided decoder는 각 decoder block을 개별적인 손실 함수로 최적화하여 feature 학습을 향상시킨다.

- **Performance Highlights**: 제안된 AD-Net은 다른 최신 방법들과 비교했을 때, 적은 모델 파라미터를 요구하며 더 빠른 수렴(convergence)을 보였다. 4개의 공공 벤치마크 데이터셋을 통해 AD-Net의 성능이 평가되었으며, 연구 결과는 데이터 증강(data augmentation) 전략을 사용하지 않더라도 우수한 성능을 보임을 시사한다.



### From Words to Poses: Enhancing Novel Object Pose Estimation with Vision Language Models (https://arxiv.org/abs/2409.05413)
- **What's New**: 이 논문에서는 언어 임베딩을 활용한 새로운 프롬프트 기반 제로샷 6D 객체 자세 추정 프레임워크를 제안합니다. 이를 통해 로봇이 이전 지식 없이도 새로운 객체를 인식하고 그 위치를 파악할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 NeRF(Neural Radiance Fields)와 LERF(Language-Embedded Radiance Fields)를 기반으로 객체의 모호한 위치를 추정한 후, 포인트 클라우드 등록 방법(예: TEASER++)을 사용하여 6D 자세를 계산합니다. 또한, 활성화 임계값과 같은 하이퍼파라미터를 분석하여 제로샷 능력을 최적화할 필요한 조건들을 확인합니다.

- **Performance Highlights**: 이 연구는 다양한 하이퍼파라미터에 기반한 제로샷 능력의 성과를 평가하며, HouseCat6D 데이터세트에 대한 실험을 통해 제안된 방식이 가정용 로봇에 적합함을 입증합니다. 향후 연구는 산업 환경에서의 어플리케이션도 탐구할 계획입니다.



### KRONC: Keypoint-based Robust Camera Optimization for 3D Car Reconstruction (https://arxiv.org/abs/2409.05407)
Comments:
          Accepted at ECCVW

- **What's New**: KRONC는 차량 장면에 특화된 새로운 카메라 프레임 등록(Registration) 알고리즘을 제안하며, 기존의 구조물 기반 모션 추정(Structure-from-Motion, SfM) 방법에 대한 의존성을 줄입니다. 이 방법은 차량 점검에 필요한 고품질 이미지를 위한 새로운 데이터셋(KRONC-dataset)을 제공합니다.

- **Technical Details**: KRONC는 시맨틱 키포인트(semantic keypoints)로부터 카메라 시점(view poses)을 추정하고, 이를 경량화된 최적화 문제로 해결하여 키포인트의 백프로젝션(back-projection)을 수렴하는 방향으로 진행합니다. Keypoint 기반의 카메라 등록 방법은 매우 낮은 계산 오버헤드를 유지하며, 차량 장면에서의 좋은 카메라 배치를 찾는데 효과적입니다.

- **Performance Highlights**: KRONC는 실제 차량 시나리오에서 카메라 포즈를 초기화 단계에서 매우 간단하게 설정함으로써, 최종적으로 입력 이미지 수를 75% 줄이더라도 효과적인 카메라 위치 구성을 찾을 수 있습니다. 이를 통해 COLMAP에 비해 1배 이상의 속도 개선을 달성하며, 성능은 최신 상태의 방법과 비교할 수 있습니다.



### A Survey of Multimodal Composite Editing and Retrieva (https://arxiv.org/abs/2409.05405)
Comments:
          22 pages, 3 figures, and 11 tables

- **What's New**: 이번 설문조사는 다중 모달 복합 검색(multimodal composite retrieval)에 대한 포괄적인 리뷰를 제공하는 첫 번째 논문으로, 이미지-텍스트 복합 편집 및 검색 방법을 심도 있게 탐구합니다. 다중 모달 데이터 유형의 통합을 통해 개인화된 정보 검색의 개선을 꾀하고 있습니다.

- **Technical Details**: 다중 모달 복합 검색은 텍스트, 이미지, 오디오 등 다양한 모달리티를 통합하여 더 정확하고 맞춤형 결과를 제공합니다. 최근에는 Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) 네트워크, Visual Transformer (ViT) 등을 사용하여 이미지 검색 성능을 향상시키는 방법들이 제안되고 있습니다.

- **Performance Highlights**: 다중 모달 복합 검색 방법들은 사용자 질의와 맥락을 더 잘 이해함으로써 검색 성능과 사용자 만족도를 개선하는 데 기여하고 있습니다. 다양한 모달리티를 활용하는 이러한 기술들은 지난 몇 년간 많은 연구가 진행되어오며, 특히 패션 산업, 의료 진단 등 여러 분야에 활발히 응용되고 있습니다.



### Sequential Posterior Sampling with Diffusion Models (https://arxiv.org/abs/2409.05399)
Comments:
          5 pages, 4 figures, preprint

- **What's New**: 본 연구에서는 sequential한 diffusion posterior sampling의 효율성을 높이기 위해 새로운 접근 방식을 제안합니다. 심장 초음파 이미지 데이터에 대한 사례를 통해 이 접근법이 어떻게 real-time posterior sampling을 가능하게 하는지 보여줍니다.

- **Technical Details**: 이 방법은 Video Vision Transformer (ViViT)를 사용하여 이전 diffusion 출력에 기초한 전이 동역학을 모델링하여 이루어집니다. SeqDiff와 SeqDiff+ 두 가지 변형을 제안하며, SeqDiff는 이전 프레임의 출력을 사용하고, SeqDiff+는 프레임 간의 전이를 모델링하여 정확한 초기화를 수행합니다.

- **Performance Highlights**: 이 연구는 고속 심장 초음파 이미지 데이터셋을 사용하여 검증한 결과, 제안된 방식이 같은 성능을 유지하면서 추론을 25배 가속화하며, 심한 움직임이 있는 경우 PSNR을 최대 8% 향상시키는 데 성공했습니다.



### FacialFlowNet: Advancing Facial Optical Flow Estimation with a Diverse Dataset and a Decomposed Mod (https://arxiv.org/abs/2409.05396)
Comments:
          ACMMM2024

- **What's New**: 본 논문은 FacialFlowNet(FFN)라는 새로운 대규모 얼굴 광학 흐름(optical flow) 데이터셋과, 얼굴 흐름을 분해하는 최초의 방법인 Decomposed Facial Flow 모델(DecFlow)을 제안합니다. FFN은 9,635명의 다양한 정체성과 105,970개의 이미지 쌍으로 구성되어 있어, 얼굴 및 머리 움직임 분석의 다양성을 제공합니다.

- **Technical Details**: FFN 데이터셋은 얼굴 움직임을 나타내는 이미지 쌍과 머리 흐름(head flow) 라벨을 제공합니다. DecFlow는 얼굴 의미 중심 인코더(facial semantic-aware encoder)와 분해 흐름 디코더(decomposed flow decoder)를 특징으로 하며, 얼굴 흐름을 머리 흐름과 표현 흐름으로 정확하게 분해할 수 있습니다. 이 접근 방식은 미세한 표정을 정확하게 분석하는 데 용이합니다.

- **Performance Highlights**: FFN은 다양한 광학 흐름 방법에서 얼굴 흐름 추정의 정확성을 크게 향상시키며, 최대 11%의 Endpoint Error(EPE) 감소(3.91에서 3.48로)를 달성합니다. DecFlow는 FFN과 결합하여 기존 방법들보다 우수한 성능을 보이며, 미세 표정 인식에서 18%의 정확도 향상(69.1%에서 82.1%로)을 달성했습니다.



### Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling (https://arxiv.org/abs/2409.05395)
- **What's New**: 이 연구는 최근의 구조적 상태 공간 모델(SSM)인 Mamba를 비주얼 언어 모델(VLM)에서 Transformer와 대체하여 성능을 비교합니다. Mamba가 Transformer 기반 VLM보다 캡션 작성, 질의 응답 및 독서 이해에서 더 나은 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: 모델은 최대 30억 개의 매개변수로 테스트되었으며, Mamba 기반 VLM이 Transformer 기반 VLM을 네 개의 기준 제시 작업에서 초과한 결과가 나왔습니다. 그러나 Transformer는 시각적 기초 작업에서 더 나은 성과를 보였으며, 이는 모델 크기가 커질수록 더욱 두드러졌습니다.

- **Performance Highlights**: Mamba는 이미지의 요약에 의존하는 작업에서는 유망한 성능을 보여주지만, 문맥에서 명시적 정보를 검색해야 하는 작업에서는 어려움을 겪었습니다. 연구 결과 Mamba는 캡션 작성, 질문 응답 및 독해에서 Transformer를 초과했지만, grounding 작업에서 Transformer가 더 높은 성능을 보여주었습니다.



### TAVP: Task-Adaptive Visual Prompt for Cross-domain Few-shot Segmentation (https://arxiv.org/abs/2409.05393)
- **What's New**: 이 논문은 Segment Anything Model (SAM)을 기반으로 다분야 적응 프롬프트 프레임워크를 제안하여, Cross-domain Few-shot Segmentation (CD-FSS)에서의 성능 향상을 목표로 합니다. 전통적인 딥 네트워크가 대량의 레이블 데이터에 의존하는 반면, 이번 연구는 기존 SAM의 주민지식(knowledge)을 유지하면서 새로운 응용 프로그램으로의 전이와 학습 능력 보존의 중요성을 강조합니다.

- **Technical Details**: Multi-level Feature Fusion (MFF) 기법을 통해 통합된 특징 추출을 수행하며, Class Domain Task-Adaptive Auto-Prompt (CDTAP) 모듈을 도입하여 클래스-도메인 무관한 특징 추출 및 고품질의 학습 가능한 프롬프트 생성을 지원합니다. 또한 대비 학습(contrastive learning)을 적용하여 클래스를 기준으로 저수준 및 고수준 특징 정보를 효율적으로 분리하여 CD-FSS의 정확성을 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 논문에서는 SAM 기반의 새로운 CD-FSS 접근 방식이 세 가지 벤치마크에서 이전의 최첨단(SOTA) 방법들과 비교해 최고의 결과를 달성했다고 보고하며, 실험 결과를 통해 SAM의 풍부한 특징 정보가 CD-FSS에 더 잘 학습되었다고 밝혔습니다.



### A Novel Representation of Periodic Pattern and Its Application to Untrained Anomaly Detection (https://arxiv.org/abs/2409.05389)
- **What's New**: 이 논문에서는 주기적인 텍스처를 가진 산업 제품의 품질 검사에서 기존의 이미지 기반 방법의 한계를 극복하기 위해 새로운 접근법을 제시합니다.

- **Technical Details**: 저자들은 주기적인 이미지를 연속적인 파라미터 집합에 정의된 새로운 자기 표현(self-representation)을 통해 학습하여 주기적 스파스 분해(periodic-sparse decomposition)라는 공동 최적화(framework)를 제안합니다. 이 방법은 희소한(anomalies) 이상과 가우시안 노이즈를 동시에 모델링합니다.

- **Performance Highlights**: 모의(simulated) 및 실제(real-world) 사례 연구를 통해 제안된 방법론이 주기적인 패턴 학습과 이상 탐지에서 효과적임을 보여줍니다.



### Decoupling Contact for Fine-Grained Motion Style Transfer (https://arxiv.org/abs/2409.05387)
- **What's New**: 이 논문에서는 동작 스타일 전송(motion style transfer)에서 접촉(contact)을 세밀하게 제어할 수 있는 혁신적인 방법을 제안합니다. 이 방법은 모션의 자연스러움과 스타일의 시공간적 변화를 동시에 달성합니다.

- **Technical Details**: 접촉 제어는 고관절 속도(hip velocity)를 통해 간접적으로 이루어지며, 이를 궤적(trajectory)과 접촉 타이밍(contact timing)으로 분해합니다. 또한, 접촉과 스타일 간의 관계를 모델링하는 새로운 신경망을 도입해 고차원 및 저차원 피쳐를 각각 학습합니다. 마지막으로, 변환기(decoder) 기반의 모델을 통해 최종 모션 합성이 가능합니다.

- **Performance Highlights**: 종합적인 평가를 통해, 제안된 방법이 기존의 방법들에 비해 스타일 표현력(style expressivity)과 모션 품질(motion quality) 면에서 우수한 성능을 보임을 입증했습니다.



### Look One and More: Distilling Hybrid Order Relational Knowledge for Cross-Resolution Image Recognition (https://arxiv.org/abs/2409.05384)
Comments:
          Accepted by AAAI 2020

- **What's New**: 이번 논문에서는 저해상도 이미지 인식을 위한 하이브리드 오더 관계 지식 증류(hybrid order relational knowledge distillation) 접근법을 제안합니다. 이 방법은 고해상도 이미지에서의 인식 능력을 줄임으로써 저해상도 이미지 인식을 향상시킵니다.

- **Technical Details**: 제안된 방법은 세 가지 스트림, 즉 튜터 스트림(teacher stream), 학생 스트림(student stream), 보조 스트림(assistant stream)으로 구성됩니다. 튜터 스트림은 고해상도 이미지를 매우 정확하게 인식할 수 있도록 사전 훈련(pre-trained)됩니다. 학생 스트림은 튜터의 행동을 모방하여 저해상도 이미지를 인식하게 되며, 보조 스트림은 지식 전이를 돕기 위해 튜터와 학생 사이의 다리 역할을 수행합니다.

- **Performance Highlights**: 다양한 작업(metric learning, 저해상도 이미지 분류, 저해상도 얼굴 인식)을 통해 실험한 결과, 제안된 접근법이 메모리 사용량이 줄어들고 속도가 빨라지면서 저해상도 이미지 인식에서 인상적인 성능을 보여주었습니다.



### Deep Learning for Video Anomaly Detection: A Review (https://arxiv.org/abs/2409.05383)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 연구는 VAD(비디오 비정상 탐지)에 대한 포괄적인 리뷰를 제공하며, 다양한 감독 신호에 기반한 다섯 가지 VAD 작업을 다룹니다. 또한 최신 연구 동향인 오픈 세트 감독 VAD와 사전 훈련된 대형 모델 기반 VAD 방법을 재조명합니다.

- **Technical Details**: 연구는 VAD 작업을 반지도학습(semi-supervised), 약지도학습(weakly supervised), 완전 지도학습(fully supervised), 비지도학습(unsupervised), 오픈 세트 감독(open-set supervised) VAD로 분류하여 체계적인 분류 체계를 제안합니다. 각 방법의 특징 및 성능 비교에 대해 심층적으로 논의합니다.

- **Performance Highlights**: 최근 7년간의 성과 통계를 통해 VAD 방법의 AUC(Area Under the Curve)가 CUHK Avenue 데이터셋에서 70.2%에서 90.1%로 향상된 것을 보여줍니다. 다양한 데이터셋에서 SOTA(state-of-the-art) 방법의 성능이 지속적으로 상승하고 있음을 강조합니다.



### Boosting CLIP Adaptation for Image Quality Assessment via Meta-Prompt Learning and Gradient Regularization (https://arxiv.org/abs/2409.05381)
- **What's New**: 본 논문에서는 이미지 품질 평가(Image Quality Assessment, IQA) 분야에서 새로운 Gradient-Regulated Meta-Prompt IQA Framework (GRMP-IQA)를 제안합니다. 이 프레임워크는 강력한 비주얼-언어 모델인 CLIP을 활용하여 데이터가 제한된 하향식 IQA 작업에 빠르게 적응하도록 돕습니다.

- **Technical Details**: GRMP-IQA는 두 개의 주요 모듈로 구성되어 있습니다: Meta-Prompt Pre-training Module과 Quality-Aware Gradient Regularization. 첫 번째 모듈은 다양한 왜곡을 고려한 메타 학습(paradigm)을 활용하여 소프트 프롬프트를 사전 훈련(Pre-training)하고, 두 번째로 품질 관련 특징에 집중하도록 업데이트된 기울기를 조정하여 오버피팅(overfitting)을 방지합니다.

- **Performance Highlights**: GRMP-IQA는 단 200개의 데이터 샘플을 사용하여, LIVEC 데이터셋에서 SOTA(State-Of-The-Art) 모델을 능가하였고, KonIQ 데이터셋에서도 경쟁력 있는 성능을 나타냈습니다. 특히, 20%의 훈련 데이터를 활용하여 SRCC 값 0.836을 달성하였습니다.



### Prim2Room: Layout-Controllable Room Mesh Generation from Primitives (https://arxiv.org/abs/2409.05380)
- **What's New**: 이 논문에서는 2D 레이아웃 조건과 3D 프리미티브(primitive) 검색을 활용하여 제어 가능한 실내 메쉬 생성을 위한 새로운 프레임워크인 Prim2Room을 제안합니다. 이는 사용자가 정밀한 3D 레이아웃을 지정할 수 있도록 돕는 방식을 채택하고 있습니다.

- **Technical Details**: Prim2Room은 서로 다른 뷰포인트를 사용하여 개체의 질감(texture)과 기하학(geometry)을 생성하는 적응형 뷰포인트 선택 알고리즘을 도입합니다. 이 시스템은 비강직(depth) 등록(non-rigid depth registration)을 통해 생성된 객체들과 해당 프리미티브 간의 정렬을 보장하고, 형태 변화를 허용하여 다양성을 유지합니다.

- **Performance Highlights**: 제안된 방법은 생성된 3D 장면의 정확성과 미적 매력을 향상시키고, 사용자에게 실내 디자인에서의 상세한 작업을 수행할 수 있는 직관적인 플랫폼을 제공합니다.



### PersonaTalk: Bring Attention to Your Persona in Visual Dubbing (https://arxiv.org/abs/2409.05379)
Comments:
          Accepted at SIGGRAPH Asia 2024 (Conference Track)

- **What's New**: 이 논문에서는 audio-driven visual dubbing(오디오 기반 비주얼 더빙) 분야에서 speaker의 개성과 정확한 lip synchronization(립 동기화)을 유지하는 데 중점을 둔 새로운 프레임워크인 PersonaTalk를 제시합니다. 기존 방법들은 speaker의 특정 스타일이나 얼굴의 세부 정보를 잘 반영하지 못하는 문제가 있었습니다.

- **Technical Details**: PersonaTalk는 attention 기반의 두 단계 프레임워크로 구성되어 있으며, 첫 번째 단계에서는 style-aware audio encoding module(스타일 인식 오디오 인코딩 모듈)을 사용하여 오디오 특징에 speaking style(말하기 스타일)을 주입합니다. 두 번째 단계에서는 dual-attention face renderer(이중 주의 얼굴 렌더러)를 통해 target geometries(대상 기하형상)에 대한 텍스처를 렌더링합니다. 이 렌더러는 Lip-Attention(립 어텐션)과 Face-Attention(얼굴 어텐션)의 두 개의 병렬 크로스 어텐션 레이어로 구성되어 있습니다.

- **Performance Highlights**: Comprehensive experiments(포괄적인 실험)과 user studies(사용자 연구)를 통해 PersonaTalk는 시각 퀄리티(visual quality), lip-sync accuracy(립 동기화 정확도), persona preservation(개인화 보존)의 측면에서 최고의 방법들에 비해 우수한 성능을 보여주었다. 또한, 이 방법은 person-generic(개인 일반화) 프레임워크이면서도 state-of-the-art person-specific(최첨단 개인화) 방법들과 경쟁할 수 있는 성능을 달성할 수 있습니다.



### Memoryless Multimodal Anomaly Detection via Student-Teacher Network and Signed Distance Learning (https://arxiv.org/abs/2409.05378)
Comments:
          14 pages, 4 figures, 2 tables, to be published in PRCV-2024

- **What's New**: 본 연구에서는 RGB 이미지와 3D 포인트 클라우드를 기반으로 한 새롭고 메모리 없는 다중 모드 이상 탐지 방법인 MDSS(Memoryless multimodal anomaly Detection method)를 제안합니다.

- **Technical Details**: MDSS는 경량화된 학생-교사 네트워크(Student-Teacher Network)와 서명 거리 함수(Signed Distance Function)를 사용하여 RGB 이미지와 3D 포인트 클라우드로부터 이상 정보를 학습합니다. 이 방법은 RGB 이미지로부터의 이상 점수 맵과 3D 포인트 클라우드로부터의 서명 거리를 결합하여 최종 이상 점수 맵을 생성합니다.

- **Performance Highlights**: 실험 결과 MDSS는 최신 메모리 뱅크 기반 방법인 Shape-guided보다 안정적이며, 또한 다른 기초 방법들보다 우수한 성능을 보였습니다. MDSS는 I-AUROC 측면에서 SOTA 이미지 수준 다중 모드 이상 탐지 성능을 달성했습니다.



### KARGEN: Knowledge-enhanced Automated Radiology Report Generation Using Large Language Models (https://arxiv.org/abs/2409.05370)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)을 활용하여 자동화된 방사선 보고서 생성(R2Gen)을 향상시키기 위한 새로운 프레임워크인 KARGEN을 제시합니다. 이는 질병 관련 지식을 활성화시키기 위해 특정 지식 그래프를 통합한 최초의 시도입니다.

- **Technical Details**: KARGEN은 대규모 언어 모델의 보고서 생성 능력을 활용하는 동시에, 의학적 지식 그래프를 통합하여 방사선 이미지를 분석합니다. 이 과정에서 이미지에서 지역적 특징을 추출하고, 지식 그래프를 사용하여 질병 관련 정보를 '증류'하여 모델에게 제공합니다. 최종 보고서는 이러한 통합된 특징을 기반으로 LLaMA 기반의 리포트 생성기를 통해 작성됩니다.

- **Performance Highlights**: KARGEN은 IU-Xray 및 MIMIC-CXR 데이터셋에서 검증되었으며, 여러 최신 방법들과 비교하여 다양한 평가 지표에서 뛰어난 성능을 보였습니다. 이는 LLM 내의 직관적인 지식을 효과적으로 활용했음을 보여줍니다.



### FedBrain-Distill: Communication-Efficient Federated Brain Tumor Classification Using Ensemble Knowledge Distillation on Non-IID Data (https://arxiv.org/abs/2409.05359)
- **What's New**: 본 논문에서는 뇌종양 분류를 위한 새로운 접근 방식인 FedBrain-Distill을 제안합니다. 이 방법은 Knowledge Distillation (KD)을 결합하여 Federated Learning (FL) 환경에서 개인 정보를 보호하고 모델 아키텍처 독립성을 보장합니다.

- **Technical Details**: FedBrain-Distill은 여러 개의 복잡한 teacher 모델에서 간단한 student 모델로 지식을 증류하는 방법입니다. Teacher 모델로는 VGGNet16을 사용하며, Dirichlet 분포를 활용하여 IID 및 non-IID 데이터를 생성합니다. FedBrain-Distill은 통신 비용을 최소화하면서도 정확한 분류 성능을 보장합니다.

- **Performance Highlights**: FedBrain-Distill의 평가 결과, IID 및 non-IID 데이터 모두에서 높은 정확도 결과를 나타냈습니다. 실제 Figshare 뇌종양 데이터셋에서 저렴한 통신 비용으로도 역량을 발휘했습니다.



### Driving with Prior Maps: Unified Vector Prior Encoding for Autonomous Vehicle Mapping (https://arxiv.org/abs/2409.05352)
- **What's New**: PriorDrive 프레임워크는 선행 지도를 활용하여 HD 지도 생성을 향상시키는 새로운 접근 방식을 제공합니다. 이 프레임워크는 OpenStreetMap의 Standard Definition Maps (SD maps), 기존 HD 지도 데이터 및 역사적인 차량 데이터를 통합하여 데이터 부족 문제를 극복합니다.

- **Technical Details**: PriorDrive는 Hybrid Prior Representation (HPQuery)와 Unified Vector Encoder (UVE)를 활용하여 벡터 데이터를 인코딩합니다. UVE는 이너-벡터 인코더와 인터-벡터 인코더를 통해 세밀한 지역 특성과 전역 맥락을 동시에 처리하며, 세그먼트 및 포인트 수준의 프리트레이닝 전략을 통해 벡터 데이터의 사전 분포를 학습합니다.

- **Performance Highlights**: nuScenes 데이터셋에서의 광범위한 테스트 결과, PriorDrive는 다양한 온라인 맵핑 모델과 높은 호환성을 보이며, HDMapNet에서 +2.2 mIoU 및 MapTRv2에서 +4.2 mAP의 성능 개선을 달성했습니다.



### Early-exit Convolutional Neural Networks (https://arxiv.org/abs/2409.05336)
- **What's New**: 이 논문은 추론(Inference) 과정에서 합성곱 신경망(CNN)의 계산 비용을 줄이는 방법을 개발하는 데 초점을 맞추고 있습니다. "Early-exit CNNs"(EENets)라는 새로운 아키텍처를 도입하여, 입력값에 따라 특정 exit 지점에서 추론 과정을 중단함으로써 계산 비용을 조절합니다.

- **Technical Details**: EENets는 여러 개의 exit 블록으로 구성되어 있으며, 각각의 블록은 confidence branch와 softmax branch로 이루어져 있습니다. Confidence branch는 해당 위치에서 추론 과정을 종료할 수 있는 신뢰 점수를 계산하는 반면, softmax branch는 분류 확률 벡터를 출력합니다. 이 두 가지 블록은 학습 가능하며 서로 독립적입니다. 훈련 중에 EENets는 전통적인 분류 손실 이외에도 계산 비용을 고려하여, 쉬운 예제에 대해 더 적은 계산 리소스를 사용하도록 조정됩니다.

- **Performance Highlights**: EENets는 ResNets와 유사한 정확도를 유지하면서도 SVHN 데이터세트에서 30%, CIFAR10에서 20%, Tiny-ImageNet에서 42%의 상대적 계산 비용 절감을 달성했습니다. 이를 통해 EENets는 초기 종료 네트워크의 효과를 잘 보여줍니다.



### Lagrangian Hashing for Compressed Neural Field Representations (https://arxiv.org/abs/2409.05334)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 Lagrangian Hashing이라는 새로운 표현 방식을 발표합니다. 이는 빠른 학습 속도를 자랑하는 NeRF 방법들과 점 기반 특성(point-based representation)의 조합을 통해 신호를 더 컴팩트하게 재구성할 수 있도록 합니다.

- **Technical Details**: Lagrangian Hashing는 InstantNGP의 계층적 해시 테이블에 점 기반 표현을 통합하며, 각 점은 영향력을 가지는 필드를 장착하여 해시 테이블 내에서 가우시안 혼합체로 해석될 수 있습니다. 본 기술은 모델이 요구하는 지역에 따라 가우시안을 이동시키는 손실 함수를 도입하여 더 나은 표현을 유도합니다.

- **Performance Highlights**: 본 연구의 주된 발견은 Lagrangian Hashing를 통해 고주파 정보를 효과적으로 저장할 수 있어, 더 작고 효율적인 모델을 통해 비슷한 시각적 성능을 달성할 수 있다는 점입니다.



### KAN-Based Fusion of Dual-Domain for Audio-Driven Facial Landmarks Generation (https://arxiv.org/abs/2409.05330)
- **What's New**: 본 논문에서는 오디오 기반의 말하는 얼굴 생성 문제를 해결하기 위한 KFusion의 이중 도메인 모델을 제안합니다. 이 모델은 오디오에서 감정 정보와 얼굴 맥락을 학습하기 위해 오디오를 두 가지 이질적인 도메인으로 분리하고, KAN 기반의 융합 메커니즘을 사용합니다.

- **Technical Details**: 제안하는 모델은 오디오 신호와 신원 이미지를 입력으로 받아, 이를 바탕으로 실제 비디오의 각 프레임에 해당하는 랜드마크 시퀀스를 생성합니다. 아울러 LSTM과 Transformer를 결합하여 감정 정보를 최대한 효과적으로 추출하고, KFusion 블록을 통해 두 데이터 도메인의 정보를 융합합니다.

- **Performance Highlights**: 우리의 모델은 최근 모델과 비교하여 높은 효율성을 보여주며, 오디오와 일치하는 움직임의 랜드마크 시퀀스를 구축하는 데 기초한 새로운 가능성을 열어줍니다.



### ICPR 2024 Competition on Safe Segmentation of Drive Scenes in Unstructured Traffic and Adverse Weather Conditions (https://arxiv.org/abs/2409.05327)
Comments:
          15 pages, 7 figures, ICPR Competition Paper

- **What's New**: ICPR 2024 Competition은 비구조적 교통 환경 및 악천후에서 운전 장면의 안전한 세분화(Safe Segmentation)를 평가하기 위한 플랫폼으로, 최첨단 semantic segmentation 모델들을 테스트하였습니다. 참가자는 IDD-AW 데이터셋을 활용하여 5000개의 고품질 RGB-NIR 이미지 쌍을 제공받아 세부 픽셀 수준으로 주석을 달았습니다.

- **Technical Details**: 경쟁의 주요 초점은 Safe mean Intersection over Union(Safe mIoU) 지표를 최적화하는 것이었습니다. 이는 기존의 mIoU가 간과할 수 있는 안전과 관련된 잘못된 예측에 패널티를 부여하도록 설계되었습니다. 데이터셋 IDD-AW은 악천후 조건에서의 운전 장면을 담아냈으며, 4개 카테고리(비, 안개, 낮은 조명, 눈)로 나뉘어서 제공됩니다.

- **Performance Highlights**: 참가자들은 안전성과 강건성을 우선으로 하는 세분화 기술에서 두드러진 성과를 보였습니다. 이 경쟁을 통해 모형의 성능은 새로운 기준을 설정하였고, 안전성이 자율 주행차의 실제 적용에서 중요하다는 것을 강조했습니다. 결과적으로, 이 대회는 자율주행 기술의 혁신을 촉진할 것으로 기대됩니다.



### FIF-UNet: An Efficient UNet Using Feature Interaction and Fusion for Medical Image Segmentation (https://arxiv.org/abs/2409.05324)
- **What's New**: FIF-UNet라는 새로운 U자형 모델이 소개되었으며, 이는 의료 이미지 분할에서 프리트레인(pre-trained) 인코더의 장점을 극대화하고 고유한 세 가지 모듈을 갖추고 있습니다.

- **Technical Details**: FIF-UNet는 Channel Spatial Interaction (CSI) 모듈, Cascaded Conv-SE (CoSE) 모듈, Multi-Level Fusion (MLF) 모듈을 포함하여 인코더와 디코더 간의 정보 상호작용 및 다양한 수준의 특징 융합을 통해 성능을 개선합니다. CSI 모듈은 인코더와 디코더 단계 간의 상호작용을 통해 중요한 특징을 재조정하고, CoSE 모듈은 채널 주의(channel attention)를 사용하여 중요한 특징을 강조합니다. MLF 모듈은 다중 스케일 특징을 효과적으로 융합하여 세부정보 손실을 완화합니다.

- **Performance Highlights**: FIF-UNet은 Synapse 및 ACDC 데이터셋에서 실험을 통해 평균 DICE 점수 86.05% 및 92.58%를 기록하며 기존의 최첨단 방법들을 초월하는 성능을 보여주었습니다.



### Open-World Dynamic Prompt and Continual Visual Representation Learning (https://arxiv.org/abs/2409.05312)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 동적이며 진화하는 오픈 월드(Context) 환경에서의 지속적 학습(Continual Learning, CL)의 새로운 설정을 제안합니다. 그리고, 이를 해결하기 위한 간단하면서도 효과적인 방법인 Dynamic Prompt and Representation Learner (DPaRL)를 소개합니다.

- **Technical Details**: DPaRL은 고정된 프롬프트 풀(Prompt Pool)에 의존하지 않고 테스트 시 동적 프롬프트를 생성합니다. 이 방법은 학습 단계에서 감별적 표현(Discriminative Representation)을 동시에 업데이트하며, 이전의 PCL 방법과의 차별성을 가집니다.

- **Performance Highlights**: 실험 결과, DPaRL은 기존의 오픈 월드 이미지 검색 벤치마크에서 상태 최적(methods)에 비해 4.7% 향상된 Recall@1 성능을 보이며, 10 CL 단계에서 76.1%에서 80.8%로, 100 CL 단계에서 68.0%에서 77.1%로 향상되었습니다.



### Fitting Skeletal Models via Graph-based Learning (https://arxiv.org/abs/2409.05311)
Comments:
          This paper was presented at the 2024 IEEE International Symposium on Biomedical Imaging (ISBI)

- **What's New**: 이 연구에서는 그래프 컨볼루션 네트워크(Graph Convolutional Networks)를 활용하여 밀집(segmentation masks)에서 스켈레탈 표현(skeletal representations, s-reps)을 생성하는 새로운 스켈레타이제이션(skeletonization) 방법을 제안합니다. 이 방법은 합성 데이터와 실제 해마(hippocampus) 분할에서 평가되어 유망한 결과를 보여주었습니다.

- **Technical Details**: 제안된 방법은 HybridVNet이라고 하며, 이는 3D 이미지에서 볼륨 그래프 표현을 직접 예측하기 위해 내장된 3D 컨볼루션 인코더와 스펙트럴 그래프 컨볼루션 층을 사용합니다. 이 모델은 인코더-디코더 아키텍처를 기반으로 하며, 입력 이진 분할 마스크로부터 s-rep을 생성합니다.

- **Performance Highlights**: 제안된 알고리즘은 5000개의 합성 s-rep 데이터셋과 해마 s-rep 데이터셋에서 벤치마킹되었으며, 좋은 일관성과 빠른 추론 속도를 달성했습니다.



### RAL:Redundancy-Aware Lipreading Model Based on Differential Learning with Symmetric Views (https://arxiv.org/abs/2409.05307)
Comments:
          5 pages, 4 figures

- **What's New**: 이 논문은 왼쪽과 오른쪽 입술의 비대칭적 움직임이 포함된 미세한 차이가 의미 있는 정보를 포함하고 있다는 점에 착안하여, 비대칭 정보를 효과적으로 학습할 수 있는 새로운 모델인 RAL(Lip Reading Model)을 제안합니다.

- **Technical Details**:  제안된 RAL 모델은 DLSV(Differential Learning Strategy with Symmetric Views)와 RAO(Redundancy-Aware Operation)를 사용하여, 왼쪽과 오른쪽 입술의 유사성과 차별성을 강조하며, 학습 과정에서 불필요한 중복 정보를 필터링합니다. 또한, ACVI(Adaptive Cross-View Interaction) 모듈을 도입하여 유사한 수치적 특성을 갖는 다양한 입술 부분 간의 관계를 학습할 수 있습니다.

- **Performance Highlights**: LRW 및 LRW-1000 데이터셋에서 실험한 결과, 제안된 방법이 이전 모델에 비해 성능이 현저히 향상된 것을 확인할 수 있으며, 특히 다양한 언어 데이터를 처리할 수 있는 능력이 증가했습니다.



### RotCAtt-TransUNet++: Novel Deep Neural Network for Sophisticated Cardiac Segmentation (https://arxiv.org/abs/2409.05280)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 연구에서는 심장 구조의 세밀한 세분화(segmentation)를 위한 새로운 아키텍처인 RotCAtt-TransUNet++를 도입했습니다. 이 아키텍처는 멀티스케일 특징 집합(feature aggregation) 및 중첩 스킵 연결(nested skip connections)을 통해 글로벌 컨텍스트 모델링을 강화하였습니다.

- **Technical Details**: RotCAtt-TransUNet++는 CNN 기반 및 Transformer 기반 접근 방식을 결합하여, 다양한 해상도와 깊이에서 4개의 특징 맵(X1, X2, X3, X4)을 사용합니다. 이 아키텍처는 회전식 주의 메커니즘(rotatory attention mechanism)을 통해 슬라이스 간 연결성을 처리하고, 채널-와이즈 크로스 주의 게이트(channel-wise cross-attention gate)를 통해 멀티스케일 정보를 통합합니다.

- **Performance Highlights**: 여러 데이터 세트를 통한 실험 결과, RotCAtt-TransUNet++는 기존 방법보다 뛰어난 성능을 보여 주었으며, 관상 동맥(coronal arteries) 및 심근(myocardium)의 거의 완벽한 주석(annotation)을 달성하였습니다. 또한, 회전식 주의 메커니즘이 세분화 정확도를 크게 향상시킴을 입증했습니다.



### BrainDecoder: Style-Based Visual Decoding of EEG Signals (https://arxiv.org/abs/2409.05279)
Comments:
          5 pages, 4 figures, 2 tables

- **What's New**: 본 논문은 EEG(뇌전도) 신호를 기반으로 시각적 자극을 복원하는 새로운 방법인 BrainDecoder를 소개합니다. 기존의 방법과 달리, 이 연구는 이미지의 스타일(색상 및 질감 등) 복원에도 중점을 둡니다.

- **Technical Details**: 이 방법은 EEG 신호를 CLIP(image 및 text) 임베딩 공간과 각각 정렬하고, 이를 통해 스타일과 의미 정보를 동시에 추출하는 'style-based' 접근 방식을 사용합니다. BrainDecoder는 LSTM 기반의 인코더 아키텍처를 이용해 EEG 신호를 효과적으로 인코딩하며, CLIP 이미지 및 텍스트 인코더와의 연결을 통해 뇌 측정 중에 인지된 시각적 자극의 스타일과 콘텐츠를 복원합니다.

- **Performance Highlights**: 정량적 및 정성적 평가 모두에서 BrainDecoder는 기존 최선의 방법들보다 더 우수한 스타일 보존과 정교한 의미 정보를 추출하며, 데이터셋인 Brain2Image에서 새로운 최첨단 성능을 달성했습니다.



### Disentangled Representations for Short-Term and Long-Term Person Re-Identification (https://arxiv.org/abs/2409.05277)
Comments:
          arXiv admin note: substantial text overlap with arXiv:1910.12003

- **What's New**: 이 논문에서는 person re-identification (reID) 문제를 다루며, 쿼리 이미지에 대한 검색을 수행합니다. 새로운 기법인 identity shuffle GAN (IS-GAN)을 제안하여, 개별 신원과 관련 없는 특징을 분리하여 학습하는 접근법을 소개합니다.

- **Technical Details**: IS-GAN은 identity shuffling 기법을 통해 신원 관련 및 비관련 특징을 분리합니다. 신원 관련 특징은 특정 개인을 정의하는 데 유용한 정보를 포함하며(예: 의상), 신원 비관련 특징은 다른 요인(예: 인간 자세)을 포함합니다. 이 방법은 보조적인 감독 신호 없이도 작동합니다.

- **Performance Highlights**: IS-GAN은 Market-1501, CUHK03, DukeMTMC-reID와 같은 표준 reID 벤치마크에서 최고의 성능을 보여주었으며, Celeb-reID 데이터셋에서 긴 기간 reID 작업에 대한 새로운 최첨단 결과를 발표합니다.



### Scalable Frame Sampling for Video Classification: A Semi-Optimal Policy Approach with Reduced Search Spac (https://arxiv.org/abs/2409.05260)
- **What's New**: 본 논문은 비디오 분류기를 위한 프레임 샘플링 문제를 다루며, 기존의 높은 시간 복잡도를 가진 방법 대신 \(O(T^N)\)에서 \(O(T)\)로 검색 공간을 줄이는 새로운 반 최적 정책을 제안합니다.

- **Technical Details**: 제안된 반 최적 정책(\(\pi_s\))은 각 프레임의 신뢰도(confidence)를 기반으로 독립적으로 평가된 값을 통해 상위 N프레임을 선택하며, 이는 계산 복잡도를 효과적으로 줄입니다.

- **Performance Highlights**: 여러 데이터셋과 모델 아키텍처에서 실험을 통해, 제안된 반 최적 정책은 크기에 관계없이 안정적이고 높은 성능을 보이는 것을 입증했습니다.



### MRStyle: A Unified Framework for Color Style Transfer with Multi-Modality Referenc (https://arxiv.org/abs/2409.05250)
- **What's New**: 이 논문에서 제안하는 MRStyle은 이미지와 텍스트를 포함한 다중 모달리티(reference) 참조를 사용하여 색상 스타일 전이를 가능하게 하는 포괄적인 프레임워크입니다. 독창적인 점은 IRStyle이라는 신경망 모델을 개발하여 스타일화된 3D lookup tables (LUTs) 를 생성하는 것입니다.

- **Technical Details**: MRStyle은 이미지와 텍스트에서 오는 프롬프트를 모두 수용하는 일반적인 다중 모달리티 색상 스타일 전이 아키텍처입니다. IRStyle은 매핑 기반 방법을 본따 고해상도 이미지에서 저렴한 메모리 사용으로 아티팩트를 제거하며 색상 스타일의 일관성을 유지하도록 설계되었습니다. 텍스트 참조를 위한 TRStyle은 사전 훈련된 스테이블 디퓨전에서의 텍스트 피처를 IRStyle의 스타일 피처와 정렬하여 색상 스타일 전이를 수행합니다.

- **Performance Highlights**: 이 방법은 정성적 및 정량적인 평가 모두에서 최첨단 방법들보다 뛰어난 결과를 보였습니다. 실험 결과, MRStyle을 이용한 색상 스타일 전이에서 놀라운 성능을 달성하였으며, 훈련과 추론의 효율성을 크게 향상시켰습니다.



### Mamba-Enhanced Text-Audio-Video Alignment Network for Emotion Recognition in Conversations (https://arxiv.org/abs/2409.05243)
- **What's New**: Emotion Recognition in Conversations (ERC) 분야에서 새로운 Mamba-enhanced Text-Audio-Video alignment network (MaTAV) 제안. 기존 unimodal 방법의 한계를 극복하고 여러 modality 간의 일관성을 확보하기 위한 혁신적인 접근법을 제공함.

- **Technical Details**: MaTAV는 4개의 핵심 구성 요소로 이루어져 있으며, 각각 text-audio-video encoders (TAV-encoders), text-audio-video alignment (TAV-Alignment), multimodal fusion module, emotion classifier를 포함. MEC-Loss라는 새로운 multimodal emotion contrastive loss를 도입하여 unimodal features 간의 정렬을 최적화함. Mamba 네트워크 아키텍처를 활용하여 긴 시퀀스에서의 문맥 정보를 효과적으로 캡처.

- **Performance Highlights**: MELD와 IEMOCAP 데이터셋에서 MaTAV가 기존의 state-of-the-art 방법들보다 상당히 우수한 성능을 보여줌. MaTAV는 특히 긴 대화에서의 감정 변화 포착에 강점을 발휘함.



### A Low-Computational Video Synopsis Framework with a Standard Datas (https://arxiv.org/abs/2409.05230)
Comments:
          13 pages, 8 figures

- **What's New**: 이 논문은 비디오 요약 작업을 위한 표준 데이터셋인 SynoClip을 도입하여, 다양한 비디오 synopsis 모델의 비교를 용이하게 하고자 합니다.

- **Technical Details**: 논문에서 제안하는 FGS(Fast Greedy Synopsis) 모델은 객체 감지를 위한 empty-frame object detector와 두브 그룹화 알고리즘을 포함하여, 영상에서 불필요한 객체 프레임을 효율적으로 처리하고, 두브 간의 관계를 유지하며 시작 시간을 효율적으로 결정하는 greedy tube rearrangement 알고리즘을 사용합니다. 이 모델은 비디오의 저장 공간을 최적화하며, 영상의 중요한 정보를 보존합니다.

- **Performance Highlights**: 제안된 모델은 빠른 계산 비용으로 비디오를 재배치할 수 있으며, SynoClip 데이터셋을 기준으로 효과적으로 평가되는 성과를 보여 줍니다.



### Comparison of Two Augmentation Methods in Improving Detection Accuracy of Hemarthrosis (https://arxiv.org/abs/2409.05225)
- **What's New**: 이번 연구에서는 의료 이미징 분야에서 드문 질병인 혈우병(hemophilia) 진단을 위한 머신러닝 모델의 정확성을 높이기 위해 데이터 증강(data augmentation) 기술을 활용하는 방안을 탐구했습니다.

- **Technical Details**: 연구에서는 미리 훈련된 VGG-16 모델을 통해 초음파(ultrasound) 이미지의 특징을 추출하고, 실제 이미지(real images), 합성 이미지(synthetic images), 전통적 증강 이미지(augmentation images) 간의 유사성을 코사인 유사성(cosine similarity) 측정을 통해 비교했습니다. 이를 위해 EfficientNet-B4 모델을 이용해 '혈액(blood)' 이미지를 인식하고, 두 가지 증강 방법을 적용해 성능을 테스트했습니다. 또한, Grad-CAM(gradient-weighted class activation mapping) 시각화를 통해 예기치 않은 정확도 감소 원인을 해석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 합성 이미지와 실제 이미지 간의 평균 유사도 점수는 0.4737로 높지 않으며, 수평 뒤집기(horizontal flip) 기법으로 생성된 합성 이미지가 원본 이미지와의 유사도가 더 높았습니다. 전통적 증강 기법과 데이터 합성 모두 모델의 정확도를 개선할 수 있으며, 전통적인 증강 기법이 합성 데이터보다 더 나은 성능을 보였습니다. Grad-CAM 히트맵은 정확도 손실이 도메인 이동(domain shift) 때문임을 밝혀냈습니다.



### Lung-DETR: Deformable Detection Transformer for Sparse Lung Nodule Anomaly Detection (https://arxiv.org/abs/2409.05200)
- **What's New**: 이번 연구에서는 CT 스캔 이미지를 통한 폐 결절 탐지를 새로운 관점에서 접근하고 있습니다. 기존의 방안과 달리, 결절이 드문 경우를 단순한 이상 탐지(anomaly detection) 작업으로 전환하여 드문 데이터에 대한 탐지를 극대화했습니다.

- **Technical Details**: Deformable Detection Transformer (Deformable-DETR)를 사용하여 폐 결절을 탐지하며, 이 모델은 7.5mm의 Maximum Intensity Projection (MIP)을 통해 인접한 CT 슬라이스를 결합하여 단일 이미지로 만듭니다. 또한, 사용자 정의 포컬 로스(custom focal loss)를 적용하여 불균형 데이터셋을 보다 효과적으로 처리합니다. 이 방법론은 LUNA16 데이터셋을 기반으로 하여 엄선된 데이터 전처리 과정을 포함합니다.

- **Performance Highlights**: 이 모델은 LUNA16 데이터셋에서 F1 점수 94.2% (재현율 95.2%, 정밀도 93.3%)를 기록하며, 임상 데이터에 반영된 실제 시나리오에서 완전한 탐지 성능을 자랑합니다. 이 연구는 의료 분야에서의 실제 적용 가능성을 높이기 위해 결절 sparsity 문제를 해결하는 데 주력하고 있습니다.



### Advanced Machine Learning Framework for Efficient Plant Disease Prediction (https://arxiv.org/abs/2409.05174)
- **What's New**: 최근 스마트 농업 플랫폼에서 머신 러닝 (Machine Learning, ML) 방법이 중요한 구성 요소로 자리 잡고 있습니다. 본 논문에서는 농부들이 공공 또는 전문가 집단에서 도움을 받을 수 있는 스마트 농업 플랫폼을 만들기 위한 새로운 고급 ML 방법의 조합을 탐구합니다.

- **Technical Details**: 본 시스템은 드문드문하게 나타나는 식물 질병을 감지하기 위해 심층 학습 (Deep Learning) 기술을 사용하여 영향을 받은 이미지에서 질병을 식별합니다. 이후 사용자 커뮤니티에서 게시된 해결책을 순위화하기 위해 자연어 처리 (Natural Language Processing, NLP) 기술을 활용합니다. 또한, Twitter와 같은 인기 있는 소셜 미디어 플랫폼에서 농부들 간의 원활한 소통을 위해 메시지 채널을 구축합니다.

- **Performance Highlights**: 제안된 프레임워크는 벤치마크 데이터셋에서 테스트하였으며, 정확하고 신뢰할 수 있는 결과를 생성했습니다.



### CD-NGP: A Fast Scalable Continual Representation for Dynamic Scenes (https://arxiv.org/abs/2409.05166)
Comments:
          23 pages, full version

- **What's New**: CD-NGP는 동적 장면의 3D 재구성과 새로운 뷰 합성을 위한 빠르고 확장 가능한 표현 방식을 제안합니다. 기존의 offline 방식과는 달리, 입력 비디오를 여러 청크로 분할하여 계속해서 학습하고, 각 모델 가지의 특징을 융합하여 기억리 소모를 줄입니다.

- **Technical Details**: CD-NGP는 비디오를 타임스탬프에 따라 여러 청크로 나누고, 각 청크를 MLP 회귀기 및 공간 인코딩을 사용하여 처리합니다. 해시 테이블을 통해 각 청크를 인코딩하고 메모리 점유율을 줄이기 위해 청크 별로 모델을 계속해서 학습합니다. 각 청크는 기존의 모델 가지에 연결되고, 공간적 및 시간적 특징이 결합되어 MLP 회귀기로 전달되어 최적화 및 랜더링을 수행합니다.

- **Performance Highlights**: 실험 결과, CD-NGP는 기존 오프라인 방법보다 훈련 중 85% 낮은 메모리 소모(<14GB)를 기록하며, 온라인 대안보다는 스트리밍 대역폭이 40% 적게(<0.4MB/프레임) 소모됩니다. 이번 연구는 동적 장면 재구성을 위한 효과적이고 효율적인 솔루션을 제공합니다.



### Can OOD Object Detectors Learn from Foundation Models? (https://arxiv.org/abs/2409.05162)
Comments:
          19 pages, 4 figures

- **What's New**: 이번 연구는 대규모 오픈 세트 데이터에 기반한 텍스트-이미지 생성 모델을 활용하여 Out-of-Distribution (OOD) object detection의 효율성을 향상시키기 위한 방법인 SyncOOD를 제안합니다. SyncOOD는 생성 모델의 능력을 적극 활용하여 의미 있는 OOD 데이터를 자동으로 추출하는 데이터 큐레이션 방법을 도입합니다.

- **Technical Details**: SyncOOD는 Foundation 모델을 활용하여 텍스트-이미지 생성 모델에서 OOD 데이터를 수집하는 자동화된 데이터 큐레이션 과정을 포함합니다. 이 과정은 Hard OOD 샘플을 ID 데이터와 가까운 위치에서 생성하는 것을 목표로 하며, 컨텍스트가 OOD 탐지에 불리하게 작용하는 것을 방지합니다. OOD 샘플들은 대체로 Box-Conditioned 이미지 인페인팅(Box-Conditioned image in-painting) 방식으로 생성됩니다.

- **Performance Highlights**: SyncOOD를 통한 실험 결과, 기존 방법들에 비해 현저하게 향상된 성능을 보이며, 최소한의 합성 데이터 사용으로도 새로운 최첨단 성능을 달성했습니다. 이는 생성 모델의 잠재력을 활용한 OOD object detection의 새로운 방향을 제시합니다.



### Image color consistency in datasets: the Smooth-TPS3D method (https://arxiv.org/abs/2409.05159)
- **What's New**: 이 논문에서는 데이터셋을 생성할 때 디지털 이미징의 일관성을 유지하기 위한 컬러 일관성 문제를 해결하기 위해 개선된 3D Thin-Plate Splines (TPS3D) 색상 보정 방법을 제안합니다. 이 방법은 색상 차트와 함께 사용되어 후처리를 통해 이미지 일관성을 달성합니다.

- **Technical Details**: TPS3D 방법은 RGB 공간에서 이미지 데이터셋의 색상을 보정하기 위해 thin-plate spline 색상 보정을 적용합니다. 또한, 이 연구에서는 Smooth-TPS3D 방법을 제안하여 원래의 방법과 동등한 결과를 보여주며 이전 방법이 실패했던 ill-conditioned 시나리오를 11-15%에서 1%로 감소시킵니다. Smooth-TPS3D는 원래 방법보다 20% 빠릅니다.

- **Performance Highlights**: 연구 결과 TPS3D는 이미지 일관성을 달성하기 위한 최적의 방법으로 나타났으며, Smooth-TPS3D는 원본 방법과 유사한 성능을 보이며 보다 효율적입니다. 또한, 다양한 방법이 품질, 보정 정확도 및 계산 부하 사이에서 다른 타협점을 제공한다는 점을 논의했습니다.



### Ultron: Enabling Temporal Geometry Compression of 3D Mesh Sequences using Temporal Correspondence and Mesh Deformation (https://arxiv.org/abs/2409.05151)
- **What's New**: 이 논문은 임의의 망 구조를 가진 동적 메쉬 시퀀스를 압축하기 위한 새로운 방법을 제안합니다. 특히 시간적 상관관계(temporal correspondence)와 메쉬 변형(mesh deformation)을 활용하여 연속된 프레임 간의 관계를 설정합니다.

- **Technical Details**: 제안된 방법은 두 개의 주요 구성 요소로 나뉩니다. 첫째, 연속된 프레임 간의 상관관계를 설정하는 것이며, 둘째, 이 상관관계를 활용하여 메쉬 시퀀스를 압축합니다. 이 과정에서 비강체 3D 정합(non-rigid 3D registration)을 사용하고, 각 프레임의 품질을 평가하여 기준을 충족하는 경우 변형된 메쉬로 대체합니다. 마지막으로 엔트로피 기반 인코딩(encoding) 및 코너 테이블(corner table) 방법을 통해 정점 및 연결성을 압축합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법은 압축 성능 측면에서 최첨단 성능(state-of-the-art performance)을 달성했습니다. 이는 동적 3D 재구성 및 컴퓨터 비전 분야에서 중요한 기여를 할 것으로 기대됩니다.



### TanDepth: Leveraging Global DEMs for Metric Monocular Depth Estimation in UAVs (https://arxiv.org/abs/2409.05142)
- **What's New**: TanDepth는 UAV 애플리케이션을 위한 실시간 스케일 복구 방법으로, 상대 깊이 추정에서 메트릭 깊이 결과를 얻기 위해 설계되었습니다.

- **Technical Details**: TanDepth는 Global Digital Elevation Models (GDEM)의 희소 측정치를 사용하여 카메라 뷰로 투영한 다음, Cloth Simulation Filter를 통해 예측된 깊이 맵에서 지면 포인트를 선택하고 이를 기준 점과 연관시킵니다.

- **Performance Highlights**: TanDepth는 3개의 공공 데이터셋에서 다양한 장면에 대해 실험을 수행하여, 기존의 UAV 스케일 복구 방법과 비교할 때 효과적인 결과를 보여주었습니다.



### PdfTable: A Unified Toolkit for Deep Learning-Based Table Extraction (https://arxiv.org/abs/2409.05125)
Comments:
          19 pages, 4 figures

- **What's New**: PdfTable 툴킷은 다양한 오픈소스 모델을 통합하여 PDF 테이블 추출의 다양한 응용 시나리오에 대한 적응성을 향상시킵니다. 이 툴킷은 데이터 전처리, 레이아웃 분석, 테이블 구조 인식, 콘텐츠 추출 등으로 구성된 모듈화된 구조를 제공합니다.

- **Technical Details**: PdfTable은 총 7개의 테이블 구조 인식 모델, 4개의 OCR (Optical Character Recognition) 도구 및 3개의 레이아웃 분석 모델을 포함하여 다양한 오픈소스 알고리즘을 통합했습니다. 이 툴킷은 디지털 및 스캔된 PDF 문서의 배치 변환을 지원하고, PDF 테이블을 Excel로 직접 추출할 수 있도록 설계되었습니다.

- **Performance Highlights**: PdfTable은 중국 금융 분야의 소규모 자기 레이블링된 테이블 데이터셋과 PubTabNet의 무선 테이블 데이터셋에서 실험을 통해 효과성과 정확성이 검증되었습니다.



### PMT: Progressive Mean Teacher via Exploring Temporal Consistency for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2409.05122)
Comments:
          Accepted by ECCV2024

- **What's New**: 본 논문은 의료 영상 분할을 위한 새로운 semi-supervised learning 프레임워크인 Progressive Mean Teachers (PMT)를 제안합니다. 이 프레임워크는 훈련 과정에서 강력하고 다양한 특징을 학습하여 높은 품질의 의사 레이블을 생성하는 것을 목표로 합니다.

- **Technical Details**: PMT는 표준 Mean Teacher 구조를 사용하여 모델의 일관성을 저하시키고, 동시 훈련을 위해 두 개의 MT 아키텍처 세트를 이용합니다. 이 두 세트는 안정적인 모델 다양성을 유지하기 위해 각 입력 업데이트를 독립적으로 수행하며, Discrepancy Driven Alignment (DDA) 정규화기를 사용하여 리드 모델과 레그 모델 간의 격차를 신속하게 조정합니다. 또한 Pseudo Label Filtering (PLF) 알고리즘을 사용하여 고성능 모델에서 생성된 고품질 의사 레이블을 평가하고 선택합니다.

- **Performance Highlights**: CT 및 MRI와 같은 다양한 모드에서의 실험 결과, PMT는 여러 차원에서 최신 의료 영상 분할 접근 방식보다 뛰어난 성과를 보였습니다.



### DreamMapping: High-Fidelity Text-to-3D Generation via Variational Distribution Mapping (https://arxiv.org/abs/2409.05099)
Comments:
          15 pages, 14 figures

- **What's New**: 본 논문에서는 Score Distillation Sampling(SDS)을 철저히 분석하고 이를 개선한 Variational Distribution Mapping(VDM) 전략을 중심으로 새로운 텍스트-3D 생성 프레임워크를 제안합니다. 이 연구는 기존 SDS 모델의 색상 과다 및 부드러움 문제를 해결하기 위해 새로운 방법론을 도입하였습니다.

- **Technical Details**: VDM은 렌더링된 이미지를 확산 모델에 의해 생성된 이미지의 저하된 형태로 간주하고, 이로써 변동 분포(variational distribution) 모델링 과정을 간소화합니다. 변동 분포를 동적으로 모델링하기 위해 Distribution Coefficient Annealing(DCA)을 도입하여 생성 품질을 개선합니다. 이러한 기술적 접근은 U-Net의 복잡한 Jacobian 계산을 피할 수 있게 해줍니다.

- **Performance Highlights**: VDM과 DCA를 활용하여 갱신된 텍스트-3D 생성 프레임워크는 Gaussian Splatting 기법을 사용하며, 기존 방법들에 비해 고해상도의 사실적인 3D 자산을 효율적으로 생성할 수 있음을 다양한 실험을 통해 입증하였습니다.



### Leveraging WaveNet for Dynamic Listening Head Modeling from Speech (https://arxiv.org/abs/2409.05089)
- **What's New**: 이 논문은 대화 중 청취자의 반응을 시뮬레이션하는 영상을 생성하는 새로운 방법을 제안합니다. 이를 통해 청취자가 한 명의 화자에 맞춰 진정성 있는 반응을 생산할 수 있습니다.

- **Technical Details**: 본 연구는 WaveNet과 Long Short-Term Memory (LSTM) 네트워크의 조합을 사용한 Sequence-to-Sequence 모델을 기반으로 합니다. 이 모델은 청취자의 피드백의 미세한 뉘앙스를 캡처하는 데 중점을 두며, 각 청취자의 개성을 유지하면서 적절한 태도와 관점을 표현하게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 ViCo 벤치마크 데이터셋에서 기존의 베이스라인 모델을 능가하는 성능을 보였습니다.



### Transformer with Leveraged Masked Autoencoder for video-based Pain Assessmen (https://arxiv.org/abs/2409.05088)
- **What's New**: 본 논문은 Facial Video Analysis를 활용하여 통증 인식을 개선하고자 하며, Transformer 기반의 딥러닝 모델을 사용하여 강력한 Masked Autoencoder와 Transformer 기반 분류기를 결합했습니다.

- **Technical Details**: 이러한 모델은 얼굴 표정과 미세 표정에서 통증 수준 지표를 효과적으로 캡처하며, AI4Pain 데이터셋을 사용하여 실험을 진행했습니다. 이 과정에서는 Temporal Feature Extraction을 위한 MARLIN 기반 Masked Autoencoder와 Multivariate Time Series Classification을 위한 Transformer 기반 분류기를 활용하여, 시간적 의존성을 캡처하는 기술을 적용했습니다. 또한, Absolute Position Encoding (tAPE) 및 Relative Position Encoding (eRPE)을 배합하여 성과를 극대화했습니다.

- **Performance Highlights**: 실험 결과는 유망하며, 이는 보다 포괄적이고 객관적인 혁신적인 헬스케어 솔루션을 여는 길이 될 것입니다.



### PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions (https://arxiv.org/abs/2409.05076)
Comments:
          Accepted by ACM Multimedia 2024 BNI track (Oral)

- **What's New**: 이 연구에서는 LVLMs의 적대적 예제를 탐지하기 위한 새로운 방법인 PIP를 소개합니다. 이 방법은 무관한 프로브 질문의 주의 패턴을 활용하여 적대적 예제를 효과적으로 구별합니다.

- **Technical Details**: PIP는 무관한 프로브 질문(예: '시계가 있나요?')의 주의 패턴을 사용하여 클린 이미지와 적대적 이미지를 구분하는 비전통적인 방법입니다. 이 방법은 검증할 이미지와 프로브 질문에 대해 추가적인 추론을 한 번만 수행해야 하며, SVM(Classifier)와 결합하여 98% 이상의 재현율(recall)과 90% 이상의 정밀도(precision)를 달성합니다.

- **Performance Highlights**: PIP는 블랙박스 공격과 열린 데이터셋 시나리오에서도 효율적으로 적대적 예제를 탐지하며, 클린 예제와 적대적 예제를 무관한 질문에 대한 주의 패턴으로 구별합니다. 이 방법은 LVLM에 대한 적대적 공격을 탐지하는 최초의 시도로, LVLM 내에서의 더 깊은 이해를 위한 통찰력을 제공합니다.



### Sight View Constraint for Robust Point Cloud Registration (https://arxiv.org/abs/2409.05065)
Comments:
          9 pages

- **What's New**: 이 논문에서는 Partial to Partial Point Cloud Registration (부분 대 부분 포인트 클라우드 정합)의 새로운 접근 방식인 Sight View Constraint (SVC)를 제안합니다. 기존 방법들의 문제점을 해결하기 위해 잘못된 변환을 확실히 식별할 수 있는 방법을 제시합니다.

- **Technical Details**: 부분 대 부분 포인트 클라우드 정합은 낮은 중첩 비율(overlap rate)에서 특히 도전적입니다. 이 연구에서는 일관된 변환을 정의하는 데 필요한 메트릭이 부족하다는 점을 강조하며, SVC를 통해 유효한 변환 가설을 좁힐 수 있습니다. 이는 입력 포인트 클라우드 정렬을 위한 6자유도(6-DoF) 변환을 찾는 것이 목적입니다.

- **Performance Highlights**: 3DLoMatch 데이터셋에서 SVC를 적용하여 정합 회수율(registration recall)을 78%에서 82%로 증가시켜 최첨단 성능을 달성했습니다. SVC는 기존 Partial PCR 방법의 강건성을 현저히 향상시킵니다.



### Unsupervised Multimodal 3D Medical Image Registration with Multilevel Correlation Balanced Optimization (https://arxiv.org/abs/2409.05040)
Comments:
          Method description for MICCAI Learn2Reg 2024 challenge

- **What's New**: 이 논문에서는 Learn2Reg 2024에서 다중 모드 이미지 등록의 문제를 해결하기 위해 multilevel correlation balanced optimization (MCBO) 기반의 비지도식(multimodal) 의학 이미지 등록 방법을 제안합니다. 이 방법은 사전 운영 이미지와 운영 중 이미지 간의 효과적인 등록을 구현하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 MCBO 방법은 convexAdam 기반으로, 다음과 같은 일련의 개선 사항을 포함합니다: (1) 결합(convex) 최적화에서 부드러운 변형 최적화를 달성하기 위해 가중치 균형 항을 도입. (2) 다양한 스케일에 대한 최적화 결과를 융합하여 밀집 변형 필드의 정제를 달성하는 다단계 피라미드 융합 최적화 메커니즘 설계. (3) 다중 모드 운영 중 이미지와 사전 운영 이미지를 정렬 및 스택하여 변수 최대 융합을 실현.

- **Performance Highlights**: ReMIND2Reg와 COMULIS3DCLEM 두 개의 하위 도전 과제에서 검증 결과 모두 두 번째 순위를 달성하였습니다. 제안된 MCBO 방법은 대 변형이 있는 다중 모드 의학 이미지의 신속하고 정확한 등록을 실현할 수 있음을 보여줍니다.



### Deep Self-cleansing for Medical Image Segmentation with Noisy Labels (https://arxiv.org/abs/2409.05024)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 의학 이미지 세분화에서 노이즈가 있는 레이블을 처리하기 위한 새로운 접근 방식인 딥 셀프 클렌징(segmentation) 프레임워크를 제안합니다. 이 방법은 훈련 과정에서 노이즈가 있는 레이블을 정화하고 깨끗한 레이블을 보존합니다.

- **Technical Details**: 우리의 방법은 가우시안 혼합 모델(Gaussian Mixture Model, GMM)을 기반으로 한 레이블 필터링 모듈(Label Filtering Module, LFM)과 픽셀 수준의 클렌징 모듈(Label Cleansing Module, LCM)으로 구성됩니다. LFM은 샘플별 손실 분포를 기반으로 노이즈 레이블과 깨끗한 레이블을 구분하고, LCM은 네트워크 출력에서 타겟 영역의 대표 기반을 추출하여 저노이즈(pseudo low-noise) 레이블을 생성합니다.

- **Performance Highlights**: 임상 간암 데이터셋과 공공 심장 진단 데이터셋에 대한 실험에서 제안된 방법이 노이즈 레이블의 영향을 효과적으로 억제하고 뛰어난 세분화(segmentation) 성능을 달성하는 것을 확인했습니다.



### Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Detector (https://arxiv.org/abs/2409.05005)
Comments:
          Under review in ICASSP 2025

- **What's New**: 이 논문에서는 취약한 집단을 겨냥한 Patronizing and Condescending Language (PCL)의 첫 다중 모달 데이터셋인 PCLMM을 소개합니다. 이 데이터셋은 Bilibili에서 수집된 715개의 주석이 달린 비디오로 구성되어 있으며, PCL 인식을 위한 얼굴 표정 감지 모듈을 갖춘 MultiPCL 탐지기도 제안하였습니다.

- **Technical Details**: PCLMM 데이터셋은 21시간 이상의 비디오를 포함하고 있으며, 취약한 커뮤니티의 얼굴 표정과 비디오, 텍스트, 오디오의 특징을 통합하여 차별적인 언어의 탐지 정확도를 향상시키는 MultiPCL 탐지기를 사용합니다. PCLMM은 비디오 플랫폼에서 마이크로어그레션 탐지를 자동화하기 위한 작업을 지원합니다.

- **Performance Highlights**: PCLMM 데이터셋은 PCL 샘플의 독성 점수가 비-PCL 샘플보다 높으며, PCL 탐지기가 명확한 비율로 탐지 정확도를 향상시키는 것을 보여줍니다. PCL 비디오는 약 27.4%가 박해성으로 분류되었으며, 이는 인터넷 플랫폼에서 PCL 데이터의 분포와 일치합니다.



### Visual Grounding with Multi-modal Conditional Adaptation (https://arxiv.org/abs/2409.04999)
Comments:
          Accepted by ACM MM 2024 [Oral]

- **What's New**: 이번 연구에서는 Multi-modal Conditional Adaptation (MMCA) 방법을 제안하고, 시각 인코더의 가중치를 적응적으로 업데이트하여 텍스트 관련 영역에 집중하도록 유도합니다. 이런 접근 방식은 기존의 독립적 시각 인코더가 동일한 이미지에 대해 일반적인 시각적 특징만을 생성하는 문제를 해결하려고 합니다.

- **Technical Details**: MMCA는 다양한 모달리티에서 정보를 통합하여 다중 모달 임베딩을 가져오는 과정으로 시작됩니다. 이후 언급된 임베딩으로부터 생성된 가중치 계수를 사용하여 시각 인코더에 적용할 가중치 업데이트 행렬을 재구성합니다. 이 과정에 의해 비주얼 그라운딩 모델의 시각 인코더는 이미지 입력에 대해 텍스트와 관련된 전경 영역에 더 집중할 수 있습니다.

- **Performance Highlights**: 네 가지 일반적인 데이터셋을 기반으로 광범위한 실험을 수행한 결과, MMCA는 탁월한 성능 향상을 달성하고 최신 기술 기준(state-of-the-art)과 비교할 만한 결과를 보였습니다. 또한, 다양한 강력한 베이스라인 모델에 MMCA를 적용했을 때 일관된 성능 개선을 보였으며, 경량성과 효율성을 갖춘 접근 방식임을 입증했습니다.



### 2DSig-Detect: a semi-supervised framework for anomaly detection on image data using 2D-signatures (https://arxiv.org/abs/2409.04982)
- **What's New**: 본 논문은 2D-signature-embedded semi-supervised framework를 기반으로 하는 이미지의 이상 감지 방법인 2DSig-Detect를 도입하였습니다. 이 방법은 이미지에서 발생하는 adversarial 공격에 대한 방어 기능을 향상시키기 위해 설계되었습니다.

- **Technical Details**: 2DSig-Detect 알고리즘은 이미지에서 얻은 2D-signature 특징을 반영하여 semi-supervised anomaly detection을 수행합니다. 이 방법은 anomaly score를 정의하고, conformance score와 covariance norm 두 가지 거리 측정을 통해 테스트 인스턴스가 이상치인지 판단합니다. 또한, rough path theory를 바탕으로 하여 multi-modal streamed data에 적합한 feature transform을 제공합니다.

- **Performance Highlights**: 실험 결과, 2DSig-Detect는 기존 방법에 비해 false negative rate, F1-score, AUC-ROC 등 여러 지표에서 우수한 성능을 나타냈으며, 이상 행동 탐지 시 계산 시간 또한 단축되었습니다.



### Multi-V2X: A Large Scale Multi-modal Multi-penetration-rate Dataset for Cooperative Perception (https://arxiv.org/abs/2409.04980)
Comments:
          9 pages, 4 figures, 5 tables

- **What's New**: Multi-V2X 데이터셋은 V2X 인식에서 다양한 CAV 침투율을 지원하는 첫 번째 다중 침투율 데이터셋으로, 최대 86.21%의 CAV 침투율을 기록하여 새로운 협업 탐색을 가능하게 합니다.

- **Technical Details**: Multi-V2X는 RGB 프레임 549k, LiDAR 프레임 146k, 3D 경계 상자 4,219k를 포함하며, 차량, 오토바이, 보행자 등 6개 카테고리에 대해 다양한 데이터를 수집합니다. 데이터셋은 SUMO와 CARLA를 공동 시뮬레이션하여 생성되었으며, 차량과 도로변 장치(RSU)로 구성된 현실적인 환경에서 데이터를 수집합니다.

- **Performance Highlights**: 최대 31개의 통신 가능 에이전트를 지원할 수 있으며, 데이터셋을 통해 협업 3D 물체 탐지 작업을 위한 포괄적인 벤치마크를 제공합니다. 기존 데이터셋에 비해 CAV 침투율의 다양성을 확보함으로써, 실제 환경에 더 가깝고 신뢰할 수 있는 데이터를 제공합니다.



### RCBEVDet++: Toward High-accuracy Radar-Camera Fusion 3D Perception Network (https://arxiv.org/abs/2409.04979)
Comments:
          The extended work of RCBEVDet (CVPR2024)

- **What's New**: 이 논문에서는 레이더와 카메라의 정보를 결합하기 위한 새로운 3D 객체 탐지 프레임워크인 RCBEVDet을 제시합니다. 기존의 카메라 기반 3D 객체 탐지기를 기반으로 하며, 레이더 전용 특징 추출기(RadarBEVNet)와 크로스 어텐션 다층 융합 모듈(CAMF)을 추가하여 성능을 강화합니다.

- **Technical Details**: RCBEVDet은 레이더 점을 조밀한 조감도(BEV) 특징으로 인코딩하기 위해 듀얼 스트림 레이더 백본과 RCS(Radar Cross Section) 인식 BEV 인코더를 활용합니다. 또한, CAMF 모듈은 변형 가능한 어텐션 메커니즘을 사용해 레이더와 카메라 BEV 특징을 정렬하고 채널 및 공간 융합 층을 통해 결합합니다. RCBEVDet++는 쿼리 기반 다중 뷰 카메라 인식을 지원하도록 CAMF를 향상시킵니다.

- **Performance Highlights**: nuScenes 벤치마크에서 RCBEVDet는 카메라 기반 3D 객체 탐지 모델의 성능을 향상시키고 다양한 인식 작업에서 우수한 성능을 보여줍니다. RCBEVDet++는 상태-of-the-art 결과를 달성하며, ViT-L을 백본으로 사용할 경우 3D 객체 탐지에서 72.73 NDS 및 67.34 mAP를 기록했습니다.



### Time-independent Spiking Neuron via Membrane Potential Estimation for Efficient Spiking Neural Networks (https://arxiv.org/abs/2409.04978)
- **What's New**: 이 논문에서는 스파이킹 신경망(Spiking Neural Networks, SNNs)의 비효율적인 계산 문제를 해결하기 위해 Membrane Potential Estimation Parallel Spiking Neurons (MPE-PSN)라는 병렬 계산 방법을 제안합니다. 이 방법은 SNN의 고유한 동적 특성을 유지하면서 병렬 처리 가능성을 높여 계산 효율을 향상시킵니다.

- **Technical Details**: MPE-PSN은 스파이크 확률을 통해 활성화 출력을 추정하여 SNN의 시간 의존성을 디커플링합니다. 이 과정에서 최소 제곱 오차(MSE) 손실 함수를 사용하여 실제와 추정된 막 전위 간의 정합성을 최적화합니다. 이러한 접근 방식은 SNN의 긴 시간 의존성의 계산 효율을 크게 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 제안된 MPE-PSN 방법은 추가적인 훈련 매개변수 없이도 neuromorphic 데이터셋에서 최첨단(SOTA) 정확도와 효율성을 달성했습니다. 이는 높은 뉴런 밀도 조건에서도 계산 효율을 높일 수 있는 가능성을 보여줍니다.



### PatchAlign:Fair and Accurate Skin Disease Image Classification by Alignment with Clinical Labels (https://arxiv.org/abs/2409.04975)
Comments:
          MICCAI 2024. Early Accept Paper (amongst the top 11% of 2869 papers submitted)

- **What's New**: 본 논문에서는 피부 병변 진단에서의 예측 정확성과 공정성을 향상시키기 위한 새로운 방법인 PatchAlign을 제안합니다. 이 방법은 임상 텍스트 표현과의 정렬을 통해 피부 상태 이미지 분류 정확성을 강화합니다.

- **Technical Details**: PatchAlign은 Graph Optimal Transport (GOT) Loss를 정규화 항으로 사용하여 cross-domain alignment를 수행합니다. Masked Graph Optimal Transport (MGOT) 기술을 통해 노이즈와 아티팩트의 영향을 줄이며, 효율적으로 피부 병변 클래스를 구분합니다. 또한, 이를 통해 학습 이미지 패치 중 질병과 관련된 패치를 우선적으로 정렬하여 정확성을 최적화합니다.

- **Performance Highlights**: PatchAlign은 Fitzpatrick17k에서 2.8% (in-domain) 및 6.2% (out-domain), DDI에서 4.2% (in-domain)의 정확성을 FairDisCo 모델에 비해 향상시켰으며, 여러 피부 톤에 걸쳐 true positive rate의 공정성도 지속적으로 개선되었습니다.



### Natias: Neuron Attribution based Transferable Image Adversarial Steganography (https://arxiv.org/abs/2409.04968)
Comments:
          Accepted by IEEE TIFS

- **What's New**: 이 논문에서는 기존의 적대적 스테가노그래피 방법이 전이 가능성(transferability)을 향상시키지 못한 문제를 해결하기 위해 Natias라는 새로운 적대적 스테가노그래피 방안을 제안하고 있습니다. 이 방법은 스테가놀리틱 모델의 중간 층에서의 특징을 공격하여 다양한 스테가놀리틱 모델에서의 검출을 회피하려고 합니다.

- **Technical Details**: Natias는 스테가놀리틱 모델의 중간 계층에서 각 뉴런의 출력을 속성을 부여하여 중요 특징을 식별하고, 이들이 여러 스테가놀리틱 모델에서 채택될 수 있는 비판적 특징을 손상시킵니다. 이를 통해 적대적 스테가노그래피의 전이 가능성을 증진시킵니다. 이 방식은 기존의 적대적 스테가노그래피 프레임워크와 원활하게 통합될 수 있으며, 통합 경량 기울기(attribution) 기법을 활용하여 스테가노그래피 신호의 미세한 성격에 적응하고 기울기 소실 문제를 완화합니다.

- **Performance Highlights**: 실험 결과, 제안된 Natias 방법은 기존 접근 방식에 비해 향상된 전이 가능성을 가지며, 목표 스테가놀리틱 모델에 대해 높은 성능과 재훈련 시 유사한 보안 성능을 달성하였습니다.



### GS-PT: Exploiting 3D Gaussian Splatting for Comprehensive Point Cloud Understanding via Self-supervised Learning (https://arxiv.org/abs/2409.04963)
- **What's New**: 본 논문은 자가 지도 학습(Self-Supervised Learning)에서 3D Gaussian Splatting (3DGS)을 통합한 GS-PT를 처음으로 제안합니다. 이 방식은 3D 포인트 클라우드 데이터에서 의미 있는 표현을 학습하고, 데이터 증강과 크로스 모달 대조 학습을 효율적으로 수행할 수 있습니다.

- **Technical Details**: GS-PT는 트랜스포머(transformer) 네트워크를 기본으로 사용하여 자가 지도 사전 학습을 수행하며, 3DGS를 활용하여 마스크된 포인트 클라우드를 재구성하는 새로운 대조 학습 과제를 도입합니다. 또한, 깊이 맵(depth maps)에서의 피처를 결합하여 3D 포인트 클라우드와 2D 이미지 간의 상관관계를 활용합니다.

- **Performance Highlights**: GS-PT는 3D 객체 분류, 현실 세계 분류, Few-Shot Learning 및 분할(segmentation) 등 다양한 다운스트림 작업에서 기존의 자가 지도 학습 방법들보다 우수한 성능을 보여주었습니다.



### DDNet: Deformable Convolution and Dense FPN for Surface Defect Detection in Recycled Books (https://arxiv.org/abs/2409.04958)
- **What's New**: 이번 연구에서는 재활용 및 순환 도서의 표면 결함 탐지 정확도를 향상시키기 위해 DDNet이라는 혁신적인 모델을 제안합니다. 이 모델은 변형된 합성곱 연산자(DC)와 밀접하게 연결된 특징 피라미드 네트워크(DFPN)를 기반으로 합니다.

- **Technical Details**: DDNet의 주요 구성 요소로는 특징 추출 네트워크, DC 모듈, DFPN 모듈, 분리된 검출 헤드가 있습니다. DC 모듈은 대상의 윤곽에 맞게 합성곱 그리드를 동적으로 조정하여 미세한 형태 변화를 포착하고, DFPN은 밀집 스킵 연결을 활용하여 다중 해상도의 고충실도 특징 맵을 생성합니다.

- **Performance Highlights**: DDNet은 독자적인 데이터셋에서 46.7%의 mAP 값을 기록하며, 기존 모델보다 14.2% 향상된 결과를 보여줍니다. 이는 재활용 도서에서의 표면 결함 탐지에 있어 뛰어난 성능을 나타냅니다.



### Deep Bayesian Active Learning-to-Rank with Relative Annotation for Estimation of Ulcerative Colitis Severity (https://arxiv.org/abs/2409.04952)
Comments:
          14 pages, 8 figures, accepted in Medical Image Analysis 2024

- **What's New**: 본 논문은 Bayesian CNN을 활용한 활성 학습(active learning) 방법을 통해 상대 주석(relative annotation)에서의 쌍 선택(pair selection) 문제를 해결하는 방법을 제안합니다. 상대 주석은 강도 판단을 더 간편하게 만들며, 주석의 변동성을 줄이는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 Bayesian CNN을 기반으로 한 심층 활성 학습을 통해 모델의 불확실성을 활용하여 높은 학습 효율성을 가진 쌍을 자동으로 선택합니다. MC dropout을 통해 쌍별 학습을 수행하며, 이 과정에서 시아미즈 네트워크(Siamese network) 구조를 적용합니다. 이 방식은 감염성 대장염과 같은 의료 이미지에서의 쌍의 불확실성을 효과적으로 추정합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 적은 수의 쌍으로도 높은 성능을 발휘하는 것을 보여주었으며, 클래스 불균형(class imbalance) 상황에서도 소수 클래스의 중요한 샘플을 우선 선택하여 성능 향상에 기여하는 것으로 나타났습니다. 이 방법은 일반적인 의료 이미지 진단에서 발생할 수 있는 분류(classification) 작업에도 유용함을 입증했습니다.



### Fast Deep Predictive Coding Networks for Videos Feature Extraction without Labels (https://arxiv.org/abs/2409.04945)
- **What's New**: 이 논문은 효과적인 sparsification 기법의 부족이라는 문제를 해결하고, 동적 시스템 학습을 더 빠르고 정확하게 수행할 수 있는 DPCN(Deep Predictive Coding Network)을 제안합니다. 이 DPCN은 라벨이 없는 데이터에서도 작동할 수 있으며, 빠른 추론(inference) 및 모델 변수를 위한 개선된 절차를 통해 높은 sparsity와 정확성을 달성합니다.

- **Technical Details**: 논문은 adaptive dynamic programming에 기반한 majorization-minimization 프레임워크를 통해 DPCN의 내부 모델 변수(상태 및 원인)의 빠른 추론을 달성하며, rigorously 분석된 수렴 성질을 포함합니다. MM-DPCN(majorization-minimization DPCN)은 비지도 학습 절차를 사용하여 sparsity와 정확성을 보장하고, 이전 DPCN들과 비교하여 높은 성능을 보여줍니다.

- **Performance Highlights**: CIFAR-10, Super Mario Bros 비디오 게임 및 Coil-100 데이터 세트에서 시행된 실험에서 MM-DPCN이 학습 속도, sparsity 비율 및 feature clustering 정확도에서 이전 DPCN 버전보다 뛰어난 성능을 보였습니다.



### MoistNet: Machine Vision-based Deep Learning Models for Wood Chip Moisture Content Measuremen (https://arxiv.org/abs/2409.04920)
- **What's New**: 이번 연구에서는 깊은 학습(deep learning)과 기계 비전(machine vision) 기술을 활용하여 나무 칩의 RGB 이미지에서 수분 함량(moisture content) 클래스를 예측하는 방법을 제안합니다. 기존의 시간 소모적이고 파괴적인 전통적인 방법을 대체할 수 있는 즉각적이고 비파괴적인 접근 방법을 제시합니다.

- **Technical Details**: 연구팀은 1,600개의 RGB 나무 칩 이미지를 포함한 대규모 이미지 데이터 세트를 수집하고, 오븐 건조 기법의 결과를 바탕으로 정답 레이블을 달았습니다. 두 가지 고성능 신경망인 MoistNetLite와 MoistNetMax를 개발하였으며, Neural Architecture Search (NAS)와 하이퍼파라미터 최적화(hyperparameter optimization)를 이용해 최적의 성능을 추구했습니다. 이 모델들은 기존의 최첨단 깊은 학습 모델들과 비교 평가되었습니다.

- **Performance Highlights**: MoistNetLite는 87%의 정확도로 계산 비용이 적고, MoistNetMax는 91%의 정확도로 수분 함량 클래스를 훌륭하게 예측하였습니다. 개선된 정확성과 빠른 예측 속도를 갖춘 MoistNet 모델들은 나무 칩 가공 산업에 큰 가능성을 보여줍니다.



### Training-free ZS-CIR via Weighted Modality Fusion and Similarity (https://arxiv.org/abs/2409.04918)
Comments:
          13 pages, 3 figures

- **What's New**: 본 연구에서는 새로운 이미지 검색 방식인 composed image retrieval (CIR)에서 훈련이 필요 없는 접근 방식인 WeiMoCIR을 제시합니다. WeiMoCIR은 참조 이미지와 수정된 텍스트의 조합으로 쿼리 표현을 직접 구축할 수 있는 방식을 적용합니다. 이를 통해 수작업으로 수집해야 했던 (참조 이미지, 텍스트 수정자, 목표 이미지) 삼중항의 필요성을 없앱니다.

- **Technical Details**: WeiMoCIR은 pretrained vision-language models (VLMs) 및 multimodal large language models (MLLMs)를 활용하여 작동합니다. 이 접근 방식에서는 VLM을 이용해 참조 이미지에서 시각적 특징을 추출하고, 텍스트 수정자에서 텍스트 특징을 추출합니다. 이렇게 생성된 시각적 및 텍스트 표현을 단순한 가중 평균으로 결합하여 쿼리 표현을 형성하고, MLLM을 통해 생성된 데이터베이스 이미지의 캡션을 사용해 유사도를 계산합니다.

- **Performance Highlights**: WeiMoCIR의 성능은 FashionIQ 및 CIRR 데이터셋에서의 실험을 통해 검증되었으며, 기존 방법들과 비교하여 동등하거나 더 나은 결과를 나타냈습니다. 이 방법은 단순하고 적용이 용이하여 이미지 검색 영역에서의 활용 가능성을 확대할 수 있습니다.



### Activation Function Optimization Scheme for Image Classification (https://arxiv.org/abs/2409.04915)
- **What's New**: 이 연구에서는 심층 신경망에서의 활성화 함수의 성능을 최적화하기 위한 진화적 접근법을 제안하며, 기존의 최첨단 활성화 함수들을 넘어서는 새로운 함수를 개발했습니다.

- **Technical Details**: 우리는 Exponential Error Linear Unit (EELU)라는 고성능 활성화 함수 시리즈를 개발하였으며, ResNet50, AlexNet, VGG16, MobileNet, Compact Convolutional Transformer와 같은 다섯 가지 최첨단 신경망 아키텍처 및 CIFAR10, Imagenette, MNIST, Fashion MNIST, Beans, Colorectal Histology, CottonWeedID15, TinyImageNet을 포함한 여덟 가지 표준 데이터셋에 대해 평가했습니다.

- **Performance Highlights**: 최적화된 활성화 함수가 28개의 다양한 케이스에서 92.8%의 경우 기존 표준 함수보다 뛰어난 성능을 보여주었으며, 최적화 스킴을 통해 생성된 활성화 함수 중 $-x\cdot erf(e^{-x})$가 이미지 분류에 가장 우수한 결과를 보였습니다.



### A Quantitative Approach for Evaluating Disease Focus and Interpretability of Deep Learning Models for Alzheimer's Disease Classification (https://arxiv.org/abs/2409.04888)
- **What's New**: 이번 연구에서는 심층 학습(Deep Learning, DL) 모델의 해석가능성 향상을 위한 정량적 질병 집중 전략을 개발하였으며, 이는 알츠하이머 병(Alzheimer's Disease, AD)의 병리학적 연관성이 있는 뇌 영역에 대한 DL 모델의 집중도를 정량화하는 질병 집중(DF) 점수를 제안합니다.

- **Technical Details**: 연구에 사용된 주요 모델로는 3D ResNet, 미세 조정된 MedicalNet 및 데이터 증강을 사용하는 MedicalNet이 포함되며, MRI 데이터의 AD 분류 성능을 비교 평가하였습니다. saliency maps와 뇌 분할(brain segmentations) 기술을 결합하여 DL 모델 해석 가능성을 개선하는 방법을 제안하였습니다.

- **Performance Highlights**: 결과적으로, 미세 조정된 pretrained 모델이 질병 관련 지역에 집중하는 능력을 증가시키고, 데이터 증강은 모델의 일반화 능력을 향상시키는 것으로 나타났습니다. 이러한 정량적 평가 접근법이 AD 분류의 해석 가능성을 향상시키고 임상적 진단에 있어 DL 모델의 채택을 촉진할 수 있음을 보여줍니다.



### Contrastive Disentangling: Fine-grained representation learning through multi-level contrastive learning without class priors (https://arxiv.org/abs/2409.04867)
- **What's New**: 이 논문은 클래스 정보를 의존하지 않고 표현을 학습하는 새로운 프레임워크인 Contrastive Disentangling (CD)을 제안합니다. 기존의 방법들과 달리, CD는 고유하게 설계된 다중 수준의 contrastive learning 전략을 통해 feature extraction과 clustering의 성능을 향상시킵니다.

- **Technical Details**: CD 프레임워크는 인스턴스 수준(instace-level)과 특징 수준(feature-level)의 contrastive loss를 사용하여 표현 학습을 진행합니다. (1) 인스턴스 수준 contrastive loss는 서로 다른 샘플의 feature representations를 분리하고, (2) 특징 수준 contrastive loss는 예측된 각 특징 헤드(feature head) 간의 독립성을 촉진하며, (3) 정규화된 엔트로피 손실(normalized entropy loss)은 특징 헤드가 데이터의 의미 있는 속성을 포착하도록 유도합니다.

- **Performance Highlights**: CD 프레임워크는 CIFAR-10, CIFAR-100, STL-10, ImageNet-10 등 여러 벤치마크 데이터셋에서 기존의 방법들보다 크게 개선된 성능을 보여주었으며, 특히 클래스 정보가 결여된 상황에서 뛰어난 성과를 보였습니다.



### AdaptiveFusion: Adaptive Multi-Modal Multi-View Fusion for 3D Human Body Reconstruction (https://arxiv.org/abs/2409.04851)
- **What's New**: AdaptiveFusion은 다양한 환경에서 신뢰할 수 있는 3D 인간 신체 재구성을 위한 최초의 일반화된 적응형 다중 모달 다중 뷰 융합 프레임워크입니다. 이 프레임워크는 비교적 유연한 융합 접근 방식을 제공하며, 평가에 필요한 단일 네트워크로 다양한 조합의 센서 입력을 처리할 수 있습니다.

- **Technical Details**: AdaptiveFusion은 다양한 모달 데이터의 로컬 특성을 기반으로 적응적으로 선택하는 Fusion Transformer Module을 사용합니다. 각기 다른 모달의 특징을 동등한 토큰(token)으로 간주하여, 특정 모달이 불완전한 경우 다른 모달의 데이터를 활용하여 결합합니다. 주목할 점은 이 모델이 모든 모달 조합을 경험할 수 있도록 하는 혁신적인 모달 샘플링 모듈을 포함하고 있다는 것입니다.

- **Performance Highlights**: AdaptiveFusion은 다양한 날씨 조건과 환경에서 광범위한 실험을 통해 기존의 융합 방법들을 능가하는 우수한 성능을 보여주었으며, 특히 3D 인간 신체 재구성에서 높은 품질을 기록했습니다. 대규모 인간 데이터셋에서 실시한 평가에서는 기존의 최첨단 방식들보다 현저히 높은 정확도를 기록했습니다.



### Deep Computer Vision for Solar Physics Big Data: Opportunities and Challenges (https://arxiv.org/abs/2409.04850)
- **What's New**: 최근 태양 물리학 분야의 혁신적인 변화인 태양 물리학 빅 데이터(Solar Physics Big Data, SPBD) 시대에 접어들었습니다. 심화된 컴퓨터 비전 기술과 결합된 SPBD의 데이터 양, 속도, 다양성은 새로운 연구 기회를 창출하고 있으며, 이를 통해 손쉽게 복잡한 연구 문제를 해결할 수 있는 가능성이 열리고 있습니다.

- **Technical Details**: SPBD는 고해상도 이미지 및 지속적인 태양 활동 모니터링을 제공하는 여러 우주 및 지상 기반 관측소를 통해 수집됩니다. 특히, SDO 데이터는 약 6.5 테라바이트(TB), DKIST의 우수한 관측일로는 페타바이트(PB)에 이를 수 있습니다. 심층 학습 기반의 컴퓨터 비전(deep learning-based computer vision)을 활용함으로써 자동 특징 추출/공학이 가능해졌고, 이로써 태양의 복잡한 현상 탐지 및 분류가 신속하고 정확하게 이루어질 수 있습니다.

- **Performance Highlights**: 딥 컴퓨터 비전 모델은 복잡한 태양 패턴을 효과적으로 탐지할 수 있으며, Hα𝛼와 같은 다양한 스펙트럼 라인을 이용한 이미지 세분화(image segmentation) 및 초해상도(super-resolution) 기술을 통해 태양 관측 이미지를 자동으로 처리하여 방대한 데이터베이스를 구축할 수 있는 기회를 제공합니다.



### Rethinking The Training And Evaluation of Rich-Context Layout-to-Image Generation (https://arxiv.org/abs/2409.04847)
- **What's New**: 최근 생성 모델의 발전으로 이미지 생성 능력이 크게 향상되었으며, 본 연구에서는 layout-to-image (L2I) 생성에서 레이아웃의 각 영역을 풍부하게 표현하기 위한 새로운 regional cross-attention 모듈을 제안하고 있습니다.

- **Technical Details**: 이 연구에서는 bounding box 기반의 레이아웃을 사용하여 복잡하고 상세한 텍스트 설명을 바탕으로 객체를 생성하고 있으며, 기존 self-attention 메커니즘 대신 cross-attention 메커니즘을 활용하여 텍스트 특징을 유지하고 객체 생성을 개선합니다.

- **Performance Highlights**: 제안된 방법은 rich-context 설명에 대해 높은 정확도를 보이며, 평가 기준을 통해 open-vocabulary 환경에서도 모델의 성능을 신뢰할 수 있음을 입증했습니다.



### POINTS: Improving Your Vision-language Model with Affordable Strategies (https://arxiv.org/abs/2409.04828)
- **What's New**: 이번 연구에서는 비전-언어 모델(Vision-Language Models)의 최신 발전을 반영한 강력한 베이스라인 모델을 구축했습니다. 또한, 데이터셋을 쉽고 효율적으로 필터링하기 위해 perplexity를 활용하여 훈련 데이터셋을 선택하는 방안과 다양한 데이터셋에서 모델을 통합하는 model soup 전략을 도입했습니다.

- **Technical Details**: 제안된 모델, POINTS는 9B 파라미터로 구성되어 있으며, Consistent Aspect Ratio Dynamic High Resolution (CATTY) 기법을 통해 이미지 왜곡 문제를 해결합니다. 또한, perplexity를 활용하여 1M 규모의 큐레이션된 데이터셋을 선택하여 훈련했습니다. 이 전략은 기존 모델보다 성능이 우수합니다.

- **Performance Highlights**: 필터링된 데이터셋으로 학습한 모델은 원래 5배 큰 데이터셋으로 학습한 모델보다 뛰어난 성능을 보였습니다. 특히, 데이터셋 선택의 한계에 도달한 후에도 model soup 기법을 통해 성능 향상을 이끌어낼 수 있었습니다.



### Metadata augmented deep neural networks for wild animal classification (https://arxiv.org/abs/2409.04825)
- **What's New**: 본 연구는 카메라 트랩 이미지 데이터를 사용하여 야생 동물을 분류하는 새로운 접근 방식을 제안합니다. 기존의 이미지 데이터만을 기반으로 한 분류 방법을 넘어, 온도, 위치, 시간 등의 메타데이터(metadata)를 결합함으로써 개선된 결과를 도출하였습니다.

- **Technical Details**: 노르웨이 기후에 초점을 맞춘 데이터셋을 사용하여 모델을 훈련하였으며, 새로운 접근 방식이 기존 방법에 비해 정확도가 98.4%에서 98.9%로 향상됨을 보여줍니다. 특히, 메타데이터만으로도 높은 정확도를 달성함으로써 이미지 품질에 대한 의존도를 줄일 수 있는 가능성을 시사합니다.

- **Performance Highlights**: 이 연구는 야생 동물 분류 기술의 발전을 위한 통합 시스템을 위한 기초를 마련하고 있으며, 이미지 데이터와 메타데이터를 결합한 접근 방식이 실질적인 성과를 낳고 있음을 보여줍니다.



### FreeAugment: Data Augmentation Search Across All Degrees of Freedom (https://arxiv.org/abs/2409.04820)
Comments:
          Accepted by ECCV 2024

- **What's New**: 새로운 접근 방식인 FreeAugment는 데이터 증강(Data Augmentation) 정책의 네 가지 자유도(all degrees of freedom)를 전역적으로 최적화하는 첫 번째 방법으로, 완전히 미분 가능한(differentiable) 방법을 사용하여 비최적화 문제에서 자동으로 최적의 이미지 변환을 찾습니다.

- **Technical Details**: FreeAugment는 (1) 변환의 수를 학습할 수 있는 확률 분포를 도입하고, (2) 변환의 유형 및 순서를 최적화하여 중복 변환을 내재적으로 방지하며, (3) 각 증강에 대해 확률 분포에서 확률적 크기를 샘플링합니다. 이러한 모든 구성은 검증 세트에 대해 end-to-end 방식으로 최적화됩니다.

- **Performance Highlights**: FreeAugment는 CIFAR10/100 및 ImageNet-100과 같은 자연 이미지 벤치마크에서 최첨단의 성능을 달성하며, 다양한 도메인에서 적용 가능성과 견고성을 입증하였습니다.



### Top-GAP: Integrating Size Priors in CNNs for more Interpretability, Robustness, and Bias Mitigation (https://arxiv.org/abs/2409.04819)
Comments:
          eXCV Workshop at ECCV 2024

- **What's New**: 이 논문에서는 Top-GAP이라는 새로운 정규화 기술이 소개되었습니다. 이 기술은 CNN의 설명가능성과 강건성을 향상시키는 데 중점을 둡니다.

- **Technical Details**: Top-GAP은 학습된 특징 표현의 공간 크기를 제한하여 네트워크가 가장 중요한 이미지 영역에 집중할 수 있도록 합니다. 이는 배경의 영향을 줄이고, 적대적 공격(adversarial attacks)과 Effective Receptive Field(ERF)를 통해 검증되었습니다.

- **Performance Highlights**: 이 방법을 사용하여 CIFAR-10 데이터셋에서 PGD 공격에 대해 50% 이상의 강건 정확도를 달성하였고, 객체 국소화(지역화)에서 IOU(Intersection over Union)가 최대 25% 향상되었습니다.



### SSFam: Scribble Supervised Salient Object Detection Family (https://arxiv.org/abs/2409.04817)
Comments:
          Accepted by TMM 2024

- **What's New**: 이 논문에서는 Scribble Supervised Salient Object Detection (SSSOD) 방법을 개발하여 다양한 모달리티(modality) 조합에서 돋보이는 물체를 탐지합니다. 특히 최근에 제안된 Segment Anything Model (SAM)을 기반으로 하여 SSSOD 가족을 구축한 SSFam을 소개합니다.

- **Technical Details**: SSFam는 RGB, RGB-D, RGB-T 및 비주얼-깊이-열화(V-D-T) 조합 입력을 위해 특별히 설계된 modal-aware modulators와 siamese decoder를 포함합니다. 이 모델은 SAM의 frozen encoder에서 추출한 modal-agnostic 정보와 modal-specific 정보를 결합하여 더 나은 feature ensemble을 구축합니다. Training 단계에서는 scribble 프롬프트를 기반으로 모델을 학습시키고, 테스트 단계에서는 프롬프트 없이 수행됩니다.

- **Performance Highlights**: 모델은 다양한 모달리티 조합에서 뛰어난 성능을 보여주며, 기존의 scribble supervised 방법들 가운데 최고의 성과를 달성하고, 일부 fully supervised 방법들과도 유사한 성능을 보입니다. V-D-T SOD 데이터셋에 처음으로 scribble 레이블을 적용하여 연구에 편의성을 제공합니다.



### Power Line Aerial Image Restoration under dverse Weather: Datasets and Baselines (https://arxiv.org/abs/2409.04812)
- **What's New**: 전력선 자율 점검(PLAI)을 위한 새로운 작업인 악천후 하의 전력선 항공 이미지 복원(PLAIR-AW)이 제안되었습니다. 이 작업은 UAV가 촬영한 악천후 상태의 저하된 이미지를 깨끗하고 고품질의 이미지로 복구함으로써 PLAI의 탐지 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: PLAIR-AW 작업은 세 가지 하위 작업으로 세분화되며, 각각은 전력선 항공 이미지 탈안개(dehazing), 오염 방지(deraining), 제설(desnowing)을 포함합니다. 이를 위해 CPLID, TTPLA, InsPLAD 데이터셋을 기반으로 다양한 기상 조건에서의 합성 데이터셋이 생성되었습니다. 각각의 데이터셋은 대기 분산 모델(ASM) 및 종합 강수 모델(CRM)을 기반으로 구성되었습니다.

- **Performance Highlights**: 다양한 최첨단 이미지 복원 방법이 PLAIR-AW의 기준 방법으로 선택되었으며, 대규모 실험을 통해 이들 방법의 성능이 제안된 데이터셋에서 평가되었습니다. 이 연구는 악천후 속에서 전력선 검사의 효율성을 크게 향상시키는 데 기여할 것으로 기대됩니다.



### SpotActor: Training-Free Layout-Controlled Consistent Image Generation (https://arxiv.org/abs/2409.04801)
- **What's New**: 이번 연구에서 우리는 레이아웃 기반의 일관된 이미지 생성(Task: Layout-to-Consistent-Image, L2CI)이라는 새로운 작업을 제안합니다. 이는 주어진 레이아웃 조건 및 텍스트 프롬프트에 따라 일관되며 조합 가능한 이미지를 생성하는 것을 목표로 합니다.

- **Technical Details**: SpotActor라는 새로운 파이프라인을 제안하며, 이는 훈련이 필요 없는 방식으로 두 개의 주요 단계로 구성됩니다: 배치 조건 하의 역 업데이트 단계 및 일관성 있는 전방 샘플링 단계입니다. 우리는 레이아웃 에너지 함수를 혁신적으로 설계하고, RISA(Regional Interconnection Self-Attention) 및 SFCA(Semantic Fusion Cross-Attention) 메커니즘을 통해 이미지 간 상호작용을 가능하게 합니다.

- **Performance Highlights**: ActorBench라는 최초의 L2CI 검증 벤치마크를 통해, SpotActor가 레이아웃 정렬, 주제 일관성, 프롬프트 일치성 및 배경 다양성에서 기대치를 충족함을 입증하였습니다. 다양한 실험을 통해 우리의 방법론이 효과적임을 실증적으로 입증했습니다.



### Enhancing Outlier Knowledge for Few-Shot Out-of-Distribution Detection with Extensible Local Prompts (https://arxiv.org/abs/2409.04796)
- **What's New**: 최근 연구에서는 OOD (Out-of-Distribution) 탐지 기술에 대한 관심이 높아지고 있으며, 비전-언어 모델(Vision-Language Models, VLM)의 발전을 통해 Few-shot tuning 기법을 활용하는 방법이 주목받고 있습니다. 기존의 기법들이 글로벌 프롬프트(Global Prompts)의 최적화에 중점을 두는 데 비해, 본 연구는 국소적 정보의 세밀한 활용이 부족하다는 문제를 인식하고 새로운 코스 투 파인 튜닝(coarse-to-fine tuning) 패러다임을 제안합니다.

- **Technical Details**: 이 연구에서는 글로벌 프롬프트를 동결(freeze)하고 지역 강화(local enhancement)를 강조하는 새로운 접근 방식을 제안합니다. 이 방법은 두 가지 주요 요소로 구성됩니다: 글로벌 프롬프트 가이드 부정 증강(global prompt guided negative augmentation)과 로컬 프롬프트 강화 지역 정규화(local prompt enhanced regional regularization)입니다. 글로벌 프롬프트는 동결되어 부정 증강을 생성하는 데 도움을 주고, 로컬 프롬프트는 지역 정보 캡처를 위해 학습 가능합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 이미지넷(ImageNet-1k) 데이터셋에 대해 4-shot 튜닝에서 기존 최첨단 방법에 비해 평균 FPR95를 5.17% 감소시키며, 16-shot 결과를 초과하는 성능을 보였습니다. 이는 OOD 탐지의 강화를 위한 로컬 프롬프트 활용의 잠재력과 효과성을 입증합니다.



### Medical Image Segmentation via Single-Source Domain Generalization with Random Amplitude Spectrum Synthesis (https://arxiv.org/abs/2409.04768)
Comments:
          11 pages, 4 figures, Medical Image Computing and Computer Assisted Intervention 2024

- **What's New**: 이번 연구에서는 Random Amplitude Spectrum Synthesis (RASS)를 제안하며, 의료 이미지의 분할 성능을 향상시키기 위해 주파수 관점에서 분포 변화를 모사합니다. 기존의 단일원천 도메인 일반화(SSDG) 방법의 한계를 극복하기 위한 새로운 접근법입니다.

- **Technical Details**: RASS는 높은 주파수 대역에서 더 집중적인 변형을 적용해 의료 이미지의 주파수 스펙트럼에 무작위로 변형을 유도합니다. 추가적으로, 랜덤 마스크 셔플(Random Mask Shuffle) 및 재구성 구조(Reconstruction Design)를 도입하여 구조적 정보를 효과적으로 처리할 수 있도록 모델의 저항력을 키웁니다. 이 방법은 CT 및 MR 이미지 등 다양한 도메인에서 도메인 간 불일치를 줄이는 데 효과적입니다.

- **Performance Highlights**: RASS는 3D 태아 뇌 이미지와 2D 홍채 사진에서 검증되었으며, 기존의 SSDG 모델과 비교했을 때 분할 성능이 향상됨을 입증하였습니다.



### Cross-Dataset Gaze Estimation by Evidential Inter-intra Fusion (https://arxiv.org/abs/2409.04766)
Comments:
          This paper was previously submitted to ACM MM 2024

- **What's New**: 본 논문은 'cross-dataset gaze estimation' 문제를 해결하기 위해 새로운 'Evidential Inter-intra Fusion (EIF)' 프레임워크를 제안합니다. 이는 다양한 소스 도메인에서의 성능 저하 없이 일반화 능력을 향상시키는 데 초점을 맞춥니다.

- **Technical Details**: EIF 프레임워크는 각 데이터 세트에 대해 독립적인 단일 데이터 세트 브랜치를 구축하고, 부분적으로 겹치는 하위 공간으로 데이터 공간을 분할해 지역 회귀를 수행합니다. 주요 기술 요소로는 Normal and Inverse-Gamma (NIG) 분포를 기반으로 한 증거 회귀기(evidential regressors)와 Mixture of Normal Inverse-Gamma (MoNIG) 분포를 사용한 다중 회귀기의 융합이 포함됩니다.

- **Performance Highlights**: 제안된 EIF 프레임워크는 소스 도메인 및 보이지 않는 도메인 모두에서 일관된 성능 향상을 달성했습니다. 실험 결과는 기존의 단일 소스 도메인 일반화 방법들보다 우수한 성능을 보여주었습니다.



### Training-Free Point Cloud Recognition Based on Geometric and Semantic Information Fusion (https://arxiv.org/abs/2409.04760)
- **What's New**: 이번 연구에서는 훈련이 필요 없는(training-free) 포인트 클라우드 인식(Point Cloud Recognition) 방법에서 기하학적 특성과 의미적 특성을 통합한 혁신적인 접근 방식을 제안합니다. 기존의 방법들은 일반적으로 이러한 특성 중 하나만을 추출하는 제한이 있었으나, 우리의 방법은 두 가지를 모두 포함하여 포인트 클라우드 데이터에 대한 포괄적인 이해를 제공합니다.

- **Technical Details**: 제안된 접근 방식은 기하학 인코더(Geometric Encoder)와 의미 인코더(Semantic Encoder)로 구성된 IF-인코더(IF-encoder)를 통해 특성을 추출합니다. 기하학 인코더는 최솟값 점 샘플링(Farthest Point Sampling), k-최근접 이웃(k-NN) 및 풀링(Pooling) 작용을 사용하여 비모수적(non-parametric) 방법으로 특성을 추출합니다. 의미 인코더는 대조 학습(Contrastive Learning)을 통해 미리 학습된 모델을 활용하여 의미 특성을 추출합니다. 이후, 데이터베이스에 저장된 특성과의 코사인 유사도(Cosine Similarity)를 계산하여 분류 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 ModelNet40 및 ScanObjectNN을 포함한 여러 기준 벤치마크 데이터셋에서 기존의 최고 성능을 보이는 훈련이 필요 없는 접근 방식을 초월하는 성능을 기록하였습니다. 또한, 제안한 방법은 몇 장의 샘플만을 사용하여도 높은 분류 정확도를 달성하는 효과적인 프레임워크를 제공합니다.



### Adaptative Context Normalization: A Boost for Deep Learning in Image Processing (https://arxiv.org/abs/2409.04759)
Comments:
          arXiv admin note: text overlap with arXiv:2403.16798

- **What's New**: 이번 연구에서는 Adaptative Context Normalization (ACN)이라는 새로운 정규화 방법을 도입합니다. ACN은 데이터의 유사한 특성을 가진 집합인 "context"를 도입하여 데이터 간의 정규화를 수행합니다. 이번 연구는 이미지 처리 분야에서 ACN이 Batch Normalization (BN) 및 Mixture Normalization (MN)보다 더 나은 성능과 빠른 수렴을 제공함을 입증합니다.

- **Technical Details**: ACN은 Gaussian Mixture Model (GMM)을 기반으로 하여 activations를 정규화하는 방식을 사용합니다. 각 context에 속한 데이터들은 동일한 정규화 파라미터를 이용하여 처리되며, 이는 백프로포게이션(backpropagation) 과정에서 학습됩니다. 이를 통해 ACN은 데이터의 분포 변화에 효과적으로 대응하며, 다음과 같은 다양한 딥 뉴럴 네트워크 아키텍처에 적용될 수 있습니다: Vision Transformer와 Convolutional Neural Networks.

- **Performance Highlights**: 다양한 실험을 통해 ACN은 훈련 과정을 가속화하고, 일반화 성능을 개선하여 이미지 분류 성능을 향상시킴을 보여주었습니다. 또한, ACN은 이미지 도메인 적응과 같은 다양한 데이터 분포에 대한 모델 성능을 극대화하는 데에도 효과적입니다.



### SGSeg: Enabling Text-free Inference in Language-guided Segmentation of Chest X-rays via Self-guidanc (https://arxiv.org/abs/2409.04758)
Comments:
          This preprint has not undergone peer review or any post-submission improvments or corrections

- **What's New**: 이번 연구에서는 언어 가이드 세분화(language-guided segmentation)에서 텍스트 없는 추론(text-free inference)을 가능하게 하는 첫 번째 자가 가이드 세분화 프레임워크(SGSeg)를 제안합니다. 이 접근 방식은 기존의 다중 모달(multi-modal) 방법들이 임상 보고서와 이미지를 함께 요구하는 한계를 극복합니다.

- **Technical Details**: SGSeg는 언어 기반 U-Net과 새롭게 제안된 약한 지도 학습(weakly-supervised) 기반의 위치 강화 보고서 생성(Localization-Enhanced Report Generation, LERG) 모듈로 구성됩니다. LERG는 객체 탐지기와 위치 기반 어텐션 집계기를 통합하여 세분화 가이드를 위한 임상 보고서를 생성합니다. 이 과정에서 가짜 레이블 추출을 통해 약한 감독을 제공합니다.

- **Performance Highlights**: QaTa-COV19 데이터셋을 사용한 광범위한 실험에서 SGSeg는 기존 단일 모달(segmentation methods)보다 우수한 성능을 보여주었으며, 다중 모달 언어 가이드 세분화 방법의 최신 성능과 유사한 결과를 달성했습니다.



### Fisheye-GS: Lightweight and Extensible Gaussian Splatting Module for Fisheye Cameras (https://arxiv.org/abs/2409.04751)
- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS) 기술을 바탕으로 fisheye 카메라를 지원하는 Fisheye-GS라는 새로운 방법을 제안합니다. 기존의 3DGS 기법은 fisheye 렌즈의 고유한 3D-2D 투사 수식을 처리하는 데 어려움이 있었으나, 이 연구는 이를 해결하고 새로운 가능성을 열어줍니다.

- **Technical Details**: Fisheye-GS는 equidistant projection transformation을 직접 계산하고, fisheye 렌즈에 따른 가우시안의 평균 및 공분산을 도출하여 이를 통해 왜곡 없는 이미지를 생성합니다. 또한, CUDA 구현을 통해 효율적인 학습과 최적화가 가능합니다. 이 방법은 3DGS의 성능을 확장가능하고 경량화된 모듈로 통합할 수 있습니다.

- **Performance Highlights**: Fisheye-GS는 기존 방법과 비교하였을 때 높은 시각 품질을 유지하면서, 왜곡 없는 이미지를 생성하는 성능을 보여주었습니다. 이는 가상 현실, 보안 및 기타 실시간 렌더링 애플리케이션에 적합합니다. Fisheye-GS는 fisheye 카메라에 최적화된 최초의 오픈소스 프로젝트로, 높은 속도와 성능을 갖추고 있습니다.



### Training-Free Style Consistent Image Synthesis with Condition and Mask Guidance in E-Commerc (https://arxiv.org/abs/2409.04750)
- **What's New**: 이 연구에서는 e-commerce(전자상거래) 분야에서 스타일 일관성이 있는 이미지를 생성하기 위해 QKV(쿼리/키/값) 레벨 개념을 도입하였습니다.

- **Technical Details**: 이 방법은 UNet과 이미지 조건을 통합할 때 self-attention(자기 주의) 및 cross-attention(교차 주의)에서의 주의 맵을 수정하는 것을 포함합니다. 제품의 주요 구성을 방해하지 않으면서 사전 설정된 조건에 의해 가이드되는 train-free(훈련 필요 없음) 방법을 사용합니다. 또한 공유된 KV를 이용하여 교차 주의에서 유사성을 향상시키고, 주의 맵에서 생성된 마스크 가이드를 통해 스타일 일관성을 갖춘 이미지 생성을 효과적으로 유도합니다.

- **Performance Highlights**: 이 방법은 실제 애플리케이션에서 유망한 결과를 보여주었습니다.



### Explicit Mutual Information Maximization for Self-Supervised Learning (https://arxiv.org/abs/2409.04747)
- **What's New**: 최근 자가 감독 학습(self-supervised learning, SSL)이 주목받고 있으며, 본 연구는 상호 정보 극대화(mutual information maximization, MIM)를 SSL에 적용하는 새로운 방안을 제시합니다.

- **Technical Details**: MIM의 불변 특성을 바탕으로, 일반 데이터 분포 가정 하에 SSL에 대해 명시적인 MI 극대화가 가능하다는 것을 보이며, 이는 2차 통계량만을 사용하여 새로운 손실 함수를 도출합니다.

- **Performance Highlights**: 제안된 기법은 CIFAR-10/100 및 ImageNet-100/1K 데이터셋에서 기존의 최첨단 방법들과 비교하여 우수성을 입증하였습니다.



### Enhancing Image Authenticity Detection: Swin Transformers and Color Frame Analysis for CGI vs. Real Images (https://arxiv.org/abs/2409.04742)
Comments:
          7 pages, 5 figures, 3 tables

- **What's New**: 본 연구에서는 Swin Transformers와 RGB 및 CbCrY 색상 프레임 분석 기법을 적용하여 CGI(Computer-Generated Images)와 ADI(Authentic Digital Images)를 분류하는 새로운 접근 방식을 제안합니다. 이 방법은 수동으로 설계된 특징을 사용하지 않고, 원시 픽셀 데이터를 활용하여 훈련되고 있습니다.

- **Technical Details**: Swin Transformers는 이미지 인식 작업을 효과적으로 수행할 수 있도록 설계된 최신 딥러닝 아키텍처입니다. 이 연구에서는 색상 프레임 분석을 통해 RGB와 CbCrY 색상 프레임에서 특징을 추출하고, t-SNE 플롯을 통해 특징 공간을 시각화하였습니다. Self-attention 메커니즘을 활용하여 대규모 비주얼 데이터에서 복잡한 패턴을 학습하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 제안된 방법은 RGB 형식 이미지를 대상으로 98%의 최첨단 정확도를 달성하며, 노이즈 추가, 블러링, JPEG 압축과 같은 이미지 조작에 대한 강인성을 제공합니다. 다양한 테스트를 통해 이 방법이 기존의 전통적인 방법들보다 처리 속도와 안정성에서 현저한 향상을 보임을 확인하였습니다.



### Swin Transformer for Robust Differentiation of Real and Synthetic Images: Intra- and Inter-Dataset Analysis (https://arxiv.org/abs/2409.04734)
Comments:
          12 pages, 4 figures, 3 tables

- **What's New**: 본 연구에서는 Swin Transformer 기반 모델을 제안하여 컴퓨터 생성 이미지(CGI)와 실제 이미지(NIs) 간의 효과적인 구분을 목표로 하였습니다. 기존의 분류 방법들에 비해 성능을 향상시키려는 시도가 있습니다.

- **Technical Details**: Swin Transformer의 계층적 구조는 local과 global feature를 포착하는 데 유용하며, RGB 색상 공간에서 데이터 처리 후 세 가지 데이터셋(CiFAKE, JSSSTU, Columbia)에 대해 intra-dataset 및 inter-dataset 테스트가 수행되었습니다. 모델은 고급 전처리 기술을 사용하였고, 특성 시각화를 위해 t-SNE 기법이 채택되었습니다.

- **Performance Highlights**: 모델은 모든 데이터셋에서 97-99%의 높은 정확도를 달성하며 CGI를 감지하는 데 탁월한 성능을 보여주었습니다. 이는 디지털 이미지 포렌식 분야에서 활용 가능성을 의미합니다.



### VidLPRO: A $\underline{Vid}$eo-$\underline{L}$anguage $\underline{P}$re-training Framework for $\underline{Ro}$botic and Laparoscopic Surgery (https://arxiv.org/abs/2409.04732)
- **What's New**: VidLPRO는 로봇 및 복강경 수술을 위해 특별히 설계된 새로운 비디오-언어(VL) 사전 학습 프레임워크입니다. 이 모델은 기존의 대조 학습(constrastive learning) 방법을 넘어, 보다 포괄적인 접근 방식을 통해 비디오와 언어를 정렬하는 복잡한 시간적 동역학을 캡처합니다.

- **Technical Details**: VidLPRO는 Video-Text Contrastive Learning (VTC), Video-Text Matching (VTM), Masked Language Modeling (MLM) 목표를 결합하여 풍부한 VL 표현을 학습합니다. 또한 GenSurg+ 데이터세트를 통해 17,000개의 수술 비디오 클립과 GPT-4에 의해 생성된 캡션이 결합되어 있습니다. 이 데이터세트는 수술 영역에 필요한 대규모 고품질 VL 데이터를 충족시킵니다.

- **Performance Highlights**: VidLPRO는 zero-shot 수술 단계 인식에서 최첨단 성능을 달성하며, Cholec80에서 57.1%의 정확도와 32.1%의 F1 점수를 기록하여 기존 모델보다 21.5%의 정확도 향상과 15.7%의 F1 점수 향상을 보였습니다. 단일 프레임 추론에서도 견고한 성능을 나타내며, 시간적 컨텍스트가 증가함에 따라 효과적으로 확장됩니다.



### Cross-Organ Domain Adaptive Neural Network for Pancreatic Endoscopic Ultrasound Image Segmentation (https://arxiv.org/abs/2409.04718)
- **What's New**: 새로운 방법론인 COTS-Nets가 제안되었으며, 이는 범위 손실(boundary loss)과 일관성 손실(consistency loss)을 통해 췌장 EUS 이미지에서 종양의 정확한 분할을 돕습니다. 또한, PCEUS 데이터셋이 개발되어 501개의 경과적으로 확인된 EUS 이미지를 제공합니다.

- **Technical Details**: COTS-Nets는 보편적 네트워크와 보조 네트워크로 구성됩니다. 보편적 네트워크는 종양의 경계 정보를 학습하기 위해 경계 손실을 사용하며, 고급 특징 추출을 목적으로 다양한 기관에서 동질적 특성을 통합합니다. 보조 네트워크는 다양한 기관의 다중 스케일 특징을 통합하여 도메인 불변 지식을 확보하는 데 기여합니다.

- **Performance Highlights**: COTS-Nets는 췌장암 진단의 정확성을 상당히 향상시키며, EUS 이미지의 품질이 낮고 데이터가 제한적인 상황에서도 효과적으로 작동합니다.



### Unleashing the Power of Generic Segmentation Models: A Simple Baseline for Infrared Small Target Detection (https://arxiv.org/abs/2409.04714)
Comments:
          ACM MM'24

- **What's New**: 이번 연구는 Segment Anything Model (SAM)와 같은 일반 분할 모델을 적응시켜 적외선 소형 물체 탐지(IRSTD) 작업에 적용하는 방안을 제안한다. 이를 통해 일반 분할 모델들이 기존의 최첨단 IRSTD 기법들과 유사한 성능을 보일 수 있음을 발견하였다. 이 접근법은 기존의 데이터 기반 IRSTD 방법들이 직면한 한계를 극복하고, 더욱 효율적으로 소형 적외선 물체를 분할할 수 있는 새로운 모델의 가능성을 제시한다.

- **Technical Details**: 연구에서는 SAM 및 그 변형 모델들을 활용하여 IRSTD 작업의 성능을 향상시키기 위한 경량 모델을 제안하였다. 이 모델은 지식 증류(knowledge distillation) 기법을 통해 성능을 극대화하며, 밀집(dense) 및 희소(sparse) 쿼리 설계를 도입하여 다중 스케일 특징을 효과적으로 인코딩할 수 있도록 한다. 실험에서는 세분화된 피라미드 네트워크(feature pyramid network, FPN)와 수정된 SAM 디코더를 결합하여 고해상도 마스크를 생성한다.

- **Performance Highlights**: 제안된 모델은 NUDT 데이터셋에서 97.0 mIoU를 달성하며, 기존의 SAM 및 Semantic-SAM보다 14 IoU 이상 개선된 성능을 보였다. 이는 적외선 소형 물체 인식 분야에서 높은 정확성과 처리량을 동시에 개선했음을 나타낸다.



### Dual-stream Feature Augmentation for Domain Generalization (https://arxiv.org/abs/2409.04699)
- **What's New**: 이 논문은 domain generalization (DG) 과제에서 feature augmentation 방식을 개선하기 위해 이중 스트림(feature) 증강 방식인 Dual-stream Feature Augmentation (DFA) 방법을 제안합니다. 이 방법은 모델의 일반화 능력을 증가시키고, unseen 도메인에서의 성능을 개선하는데 초점을 맞춥니다.

- **Technical Details**: DFA 방법은 두 가지 주요 관점으로부터 유도된 hard features를 생성합니다. 첫째, uncertainty를 활용하여 cross-domain fictitious features를 생성하고, 둘째, adversarial mask를 이용해 비인과 causal(non-causal) 정보를 분리합니다. 이러한 방법을 통해 보다 discriminative한 features를 추출하게 됩니다. 이 과정에서 contrastive learning을 사용하여 semantics 일관성을 유지합니다.

- **Performance Highlights**: 여러 공개 데이터셋에서의 Extensive experiments 결과, 제안된 방법이 domain generalization을 위한 최신 성능을 달성하는 것으로 확인되었습니다.



### C2F-CHART: A Curriculum Learning Approach to Chart Classification (https://arxiv.org/abs/2409.04683)
Comments:
          This paper has been accepted for publication in the proceedings of the 2024 International Conference on Pattern Recognition (ICPR)

- **What's New**: 이 연구에서는 차트 분류(Chart Classification) 부문을 최적화하기 위해 새로운 교육 접근 방식을 소개합니다. 특히, 인간의 학습 과정을 바탕으로 한 커리큘럼 러닝(Curriculum Learning)을 활용하여 차트 분류 작업에 적용한 점이 눈에 띕니다.

- **Technical Details**: 새로운 방법론인 C2F-CHART(Coarse-to-Fine CHART)을 개발하였습니다. 이 방법은 클래스 간 유사성을 활용하여 다양한 난이도의 학습 작업을 구성하여 차트 분류를 위한 기존 모델 아키텍처인 Swin-Chart를 학습시킬 때 적용됩니다.

- **Performance Highlights**: ICP 2022 CHART-Infographics UB UNITEC PMC 데이터셋에서 최고 성과(State-of-the-Art)를 달성하였으며, 전통적인 coarse-to-fine 커리큘럼 러닝을 초월하는 결과를 보였습니다.



### Neural Augmentation Based Panoramic High Dynamic Range Stitching (https://arxiv.org/abs/2409.04679)
Comments:
          11 pages

- **What's New**: 이 논문에서는 HDR(High Dynamic Range) 장면을 위한 새로운 파노라마 HDR 스티칭(panoramic HDR stitching) 알고리즘을 제안합니다. 이 알고리즘은 기존의 LDR(Low Dynamic Range) 이미지 스티칭 방법의 한계를 극복하고, 다양한 노출의 좌표 정렬된 LDR 이미지를 사용하여 정보가 풍부한 파노라마 LDR 이미지를 생성하는데 초점을 맞춥니다.

- **Technical Details**: 제안된 알고리즘은 'neural augmentation' 기반으로, 물리 기반 접근 방식과 데이터 기반 접근 방식을 통합하여 OFOV(Overlapping Fields of View)와 서로 다른 노출을 가진 LDR 이미지의 스티칭을 수행합니다. 초기적으로는 물리 기반 접근을 통해 서로 다른 노출의 이미지를 생성하고, 이후 학습 기반 접근을 통해 이 이미지를 정제하여 최종적으로 파노라마 LDR 이미지를 생성합니다. 새로운 IMFs(Intensity Mapping Functions)를 이용하여 인접 이미지의 정보를 바탕으로 노출된 이미지를 정확히 복원하는 방법론이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존의 파노라마 스티칭 알고리즘보다 뛰어난 성능을 보여주며, 특히 어두운 영역과 밝은 영역의 정보를 잘 보존하는 것으로 확인되었습니다. 이 방식은 노출 범위가 넓은 HDR 장면에 대해 우수한 시각적 품질을 제공합니다.



### Multi-Conditioned Denoising Diffusion Probabilistic Model (mDDPM) for Medical Image Synthesis (https://arxiv.org/abs/2409.04670)
- **What's New**: 이 논문에서는 의료 이미징(application)에서 고해상도 주석이 달린 이미지를 효과적으로 생성할 수 있는 방법을 소개합니다. Denoising Diffusion Probabilistic Model (DDPM)을 활용하여 폐 CT 이미지 생성을 수행하고, 여러 조건부 주석을 동시에 적용할 수 있는 새롭고 통제된 생성 프레임워크를 개발했습니다.

- **Technical Details**: 제안된 방법은 DDPM을 이용하여 저용량 CT 이미지 데이터 세트를 대상으로 훈련 및 샘플링하는 과정을 통해 다수의 조건 및 주석을 동시에 적용할 수 있습니다. DDPM은 본질적으로 Markov Chain 모델이며, 부분적으로 발생하는 노화 과정의 역학습을 통해 이미지를 생성합니다. 이렇게 생성된 개별 폐 CT 이미지는 해부학적 정확성을 유지하면서 치료 분야에서 유의미한 Hounsfield Unit (HU) 평균을 포함할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 생성 프레임워크는 최신 이미지 생성 모델보다 해부학적 일관성에서 우수한 성과를 보였으며 실 전문가들을 속일 수 있을 정도로 사실적인 이미지를 생성합니다. 또한 비주석 이미지를 생성하는 기존 방법들과 비교해 동적인 골격에 기반하는 이러한 접근 노선은 의료 데이터 세트의 정확성과 다양성을 크게 향상시킬 수 있음을 입증하였습니다.



### Self-Supervised Contrastive Learning for Videos using Differentiable Local Alignmen (https://arxiv.org/abs/2409.04607)
Comments:
          Accepted in 2nd Workshop on Video Understanding and its Applications, held in conjunction with the British Machine Vision Conference (BMVC) 2024

- **What's New**: 최근 제안된 Local-Alignment Contrastive (LAC) 손실 함수를 통해 비디오 분석 및 이해의 중요한 요소인 프레임 단위 특징 추출에서 유의미한 향상을 이끌어냈습니다. 이 방법은 비디오 시퀀스를 정렬하여 추상화 학습을 위한 새로운 접근 방식을 제공합니다.

- **Technical Details**: 본 연구는 transformer 기반 인코더를 활용하여 프레임 수준의 특징을 추출하고, Smith-Waterman (SW) 기법을 채택하여 지역 정렬(local alignment)을 통해 비디오 시퀀스 간 최적 정렬 경로를 도출합니다. LAC 손실함수는 지역적 시간 의존성을 포착하기 위한 미분 가능한 지역 정렬 손실과 판별 학습을 강화하기 위한 대조적 손실을 결합하여 성능을 개선합니다.

- **Performance Highlights**: 제안된 방법은 Pouring 및 PennAction 데이터셋에서 진행된 여러 행동 인식 작업에서 기존의 최첨단 방법보다 우수한 성능을 보였습니다.



### Multi-scale Feature Fusion with Point Pyramid for 3D Object Detection (https://arxiv.org/abs/2409.04601)
Comments:
          12 pages

- **What's New**: 본 논문에서는 POP-RCNN(Point Pyramid RCNN)이라는 새로운 프레임워크를 제안하여, 3D 포인트 클라우드에서의 물체 탐지 성능을 개선합니다. 이 프레임워크는 여러 공간적(scale) 특성과 의미적(depth) 특성을 결합하여 정보 전송을 원활하게 합니다.

- **Technical Details**: POP-RCNN은 PPFE(Point Pyramid Feature Enhancement) 모듈을 통해 다양한 공간적 및 의미적 깊이에서 정보를 교환합니다. 이 모듈은 효율적인 멀티스케일(feature) 융합을 통해 기본적으로 복잡성을 증가시키지 않으면서도 풍부한 정보를 제공합니다. 또한, 점 밀도 신뢰성 모듈(point density confidence module)을 통합하여 거리 차이와 관련된 특성을 고려합니다.

- **Performance Highlights**: POP-RCNN은 Waymo Open Dataset 및 KITTI 데이터셋에서 기존의 Voxel-RCNN과 PV-RCNN을 각각 2.88% 및 1.12% 향상시키는 성능 결과를 보여줍니다. 특히, 장거리 탐지에서 POP-RCNN은 평균 정밀도(mAP) 2.02%, 3.32%, 1.02%의 향상을 보였습니다.



### A Novel Dataset for Video-Based Autism Classification Leveraging Extra-Stimulatory Behavior (https://arxiv.org/abs/2409.04598)
- **What's New**: 이 논문은 자폐 스펙트럼 장애(ASD)의 진단을 개선하기 위해 비디오 기반 데이터셋인 Video ASD dataset을 발표합니다. 이 데이터셋은 아동들이 여러 감각 자극에 반응하는 장면을 담고 있으며, 총 2,467개의 비디오와 약 140만 개의 프레임으로 구성되어 있습니다.

- **Technical Details**: 기존 연구들과 달리, 본 데이터셋은 아동들이 동일한 감각 자극(맛과 냄새 등)에 대해 다양한 반응을 보여주는 비디오 프레임의 특징 데이터와 함께 제공됩니다. 이 데이터셋에는 얼굴 표정 변화와 머리 자세 각도를 고려한 다양한 텍스트 레이블이 포함되어 있어, 깊이 있는 ASD 분류 연구에 기여할 것으로 기대됩니다.

- **Performance Highlights**: 논문에서는 제안된 데이터셋을 기반으로 기초 모델의 분류 성능을 시험했고, ASD 및 일반 아동(NT) 간의 반응 차이를 분석하여 더 복잡한 접근 방법과 추가 데이터의 필요성을 강조합니다. 이를 통해 ASD 분류를 위한 딥러닝 모델의 가능성을 제시하고 있습니다.



### Influence of Early through Late Fusion on Pancreas Segmentation from Imperfectly Registered Multimodal MRI (https://arxiv.org/abs/2409.04563)
Comments:
          13.5 pages of manuscript content

- **What's New**: 이 논문은 췌장(segmentation) 분할을 위한 다중 모달(fusion) 융합에서의 최적 위치를 탐구하고 있으며, 서로 완벽하게 정렬되지 않은 이미지를 분석할 때 가장 효과적인 융합 지점이 무엇인지에 대한 논의를 제공합니다.

- **Technical Details**: 총 353쌍의 T2 가중(T2-weighted) 및 T1 가중(T1-weighted) 복부 MR 이미지를 사용하여, 이미지 쌍의 정합(alignment)을 위해 이미지 등록(image registration) 기법을 적용하였습니다. 기본 UNet 모델을 활용하여 초기부터 후기까지 다양한 융합 지점을 평가하였으며, nnUNet의 경우에도 융합 포인트(generalization)를 분석하였습니다.

- **Performance Highlights**: 기본 UNet의 단일 모달 T2w 기초 모델은 Dice 점수 0.73을 기록했으며, nnUNet 모델에서는 0.80을 달성했습니다. 기본 UNet에서의 최적 융합 접근법은 인코더 중간(early/mid fusion)에서 이루어졌으며, 이는 기초 모델 대비 Dice 점수를 0.0125 개선했습니다. nnUNet의 경우, 모델 이전에 이미지를 연결(concatenation)하는 단순한 융합(early fusion)이 가장 효과적이었으며, 이는 기초 모델 대비 0.0021의 Dice 점수 향상을 가져왔습니다.



### Dual-Level Cross-Modal Contrastive Clustering (https://arxiv.org/abs/2409.04561)
Comments:
          10 pages,4 figures

- **What's New**: 새로운 이미지 클러스터링 방법인 Dual-level Cross-Modal Contrastive Clustering (DXMC)를 제안합니다. 이 방법은 외부 텍스트 정보를 활용하여 이미지-텍스트 쌍을 생성하고, 이를 사용해 대조 학습을 진행합니다.

- **Technical Details**: DXMC는 외부 지식을 활용하여 세 가지 주요 단계를 포함합니다: 1) 세멘틱 스페이스(semantic space) 구축과 이미지-텍스트 쌍 생성, 2) 이미지 및 텍스트 임베딩을 생성하기 위해 사전에 훈련된 인코더 사용, 3) 다양한 네트워크를 통해 다양한 레벨의 크로스 모달 대조 학습 수행입니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에서 실험을 통해 DXMC가 기존의 최첨단 깊은 클러스터링 방법보다 우수한 성능을 보였음을 입증하였습니다.



### Multi-Modal Diffusion for Hand-Object Grasp Generation (https://arxiv.org/abs/2409.04560)
Comments:
          8-page paper, 7-page appendix and 10 pages

- **What's New**: 이 연구는 손과 객체의 형태를 동시에 수용할 수 있는 단일 모델을 통해 손잡이를 생성하는 새로운 접근 방식을 제안합니다. Multi-modal Grasp Diffusion (MGD)라는 방법은 이질적인 데이터 소스에서 두 모달리티의 조건부 후분포와 사전분포를 학습합니다.

- **Technical Details**: MGD 모델은 초기에 대규모 3D 객체 데이터셋으로부터 객체 형상을 인코딩하고 디코딩하는 네트워크를 활용해 훈련됩니다. 이는 기존 모델의 한계를 극복할 수 있도록 돕고, 모달리티의 독립적인 인코더 및 디코더 네트워크를 통해 다양한 데이터 리소스를 융합할 수 있도록 설계되었습니다.

- **Performance Highlights**: 정성적 및 정량적 평가에 따르면, MGD는 조건부 및 비조건부 손잡이 생성을 모두 성공적으로 수행하였으며, 이전에 보지 못한 객체 형태에 대해서도 높은 안정성과 다양성을 보여 주목받고 있습니다. 이는 '유니버설 그랩 생성'을 위한 중요한 발전입니다.



### Thinking Outside the BBox: Unconstrained Generative Object Compositing (https://arxiv.org/abs/2409.04559)
- **What's New**: 본 논문에서는 제약 없는 생성적 객체 합성(unconstrained generative object compositing)의 새로운 문제를 정의하고, 기존의 마스크에 의존하지 않는 방법으로 객체 효과(그림자 및 반사)를 생성할 수 있는 디퓨전 기반 모델을 제안합니다. 이는 그림에 자연스럽고 리얼리틱한 합성을 제공합니다.

- **Technical Details**: 제안된 모델은 객체를 빈 마스크(empty mask)와 함께 사용할 수 있으며, 이 경우 자동으로 객체를 적절한 위치와 크기로 배치합니다. 이 과정에서 이미지 인페인팅(image inpainting)을 사용하여 훈련 데이터를 생성하고, 다중 스케일 이미지 임베딩(multi-scale image embeddings)을 통합하여 다양한 크기의 객체를 생성합니다.

- **Performance Highlights**: 모델은 다양한 품질 지표와 사용자 연구에서 기존의 이미지 합성 및 객체 배치 예측 모델을 초월한 성능을 보여주었습니다. 효과적인 그림자와 반사 생성에 대한 모델의 능력은 이미지의 리얼리즘을 크게 향상시킵니다.



### SCARF: Scalable Continual Learning Framework for Memory-efficient Multiple Neural Radiance Fields (https://arxiv.org/abs/2409.04482)
- **What's New**: 이번 논문에서는 새로운 지속적 학습 프레임워크인 SCARF(Scalable Continual Learning Framework for Memory-efficient Multiple Neural Radiance Fields)를 소개합니다. 이 프레임워크는 여러 3D 장면의 새로운 시점을 합성하고, 3D 장면을 점진적으로 학습하며, 새로운 장면에 대한 훈련 데이터로만 네트워크의 파라미터를 업데이트합니다.

- **Technical Details**: SCARF는 다양한 장면을 표현하기 위해 교차 장면 가중치 행렬(cross-scene weight matrix)과 장면별 가중치 행렬(scene-specific weight matrices)로 구성된 선형 조합을 사용합니다. 이 구조는 메모리 요구 사항(memory requirements)을 크게 줄여주며, 불확실한 표면 지식 증류(uncertain surface knowledge distillation) 전략을 통해 이전 장면의 방사장(field of radiance) 지식을 새로운 모델로 전이합니다. 이를 통해 캐츠트로픽 포게팅(catastrophic forgetting) 문제를 극복하고 이전 장면의 포토 리얼리스틱 렌더링 품질을 유지합니다.

- **Performance Highlights**: SCARF는 NeRF-Synthetic, LLFF 및 TanksAndTemples 데이터셋에서 지속적 학습을 통해 최첨단 렌더링 품질을 달성하였으며, 추가적으로 저렴한 저장 비용(low storage cost)을 유지합니다.



### Flash Cache: Reducing Bias in Radiance Cache Based Inverse Rendering (https://arxiv.org/abs/2409.05867)
Comments:
          Website: this https URL

- **What's New**: 이 논문은 기존의 radiance cache와 volumetric inverse rendering 방법을 개선하여, 최적화 과정에서 발생하는 편향(bias)을 최소화하는 방법을 제안합니다. 새로운 occlusion-aware importance sampler와 빠른 cache 아키텍처를 활용하여 높은 품질의 렌더링을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 volumetric geometry를 따라 카메라 레이에서 반사된 빛을 효율적으로 추정하기 위해 neural field를 활용합니다. 또한, 라디언스 캐시에서 들어오는 조명을 질의(query)할 수 있는 고용량의 경량 아키텍처를 도입하여 렌더링 편향을 피하고 고주파 및 근거리 조명 효과를 효과적으로 모델링합니다.

- **Performance Highlights**: 제안된 시스템은 이전 방법들과 비교하여 개선된 기하학(geometry) 및 물질(material) 복원을 보여주며, 특히 근거리 조명 효과가 있는 영역에서의 물질 매개변수 회복이 더욱 정확합니다.



### Neural MP: A Generalist Neural Motion Planner (https://arxiv.org/abs/2409.05864)
Comments:
          Website at this http URL. Main paper: 7 pages, 4 figures, 2 tables. Appendix: 9 pages, 5 figures, 6 tables

- **What's New**: 본 연구는 모션 플래닝의 문제를 다루며, 기존의 방법들이 문제마다 솔루션을 처음부터 생성해야 하는데 비해, 데이터 기반의 학습을 스케일으로 적용하여 복잡한 환경에서도 빠른 해결책을 제시하는 새로운 접근법을 제안합니다.

- **Technical Details**: 시뮬레이션을 통해 다양한 복잡한 장면을 구축하고, 최첨단 모션 플래너로부터 전문가 데이터를 수집하여 반응형 일반화 정책으로 증류합니다. 이 정책은 100만 개의 장면에서 유래되어, 이전에 본 적 없는 장애물과 장면 구성에 대해 일반화할 수 있습니다. 또한, 경량화된 최적화 기법을 활용하여 안전한 경로를 보장합니다.

- **Performance Highlights**: 64개의 모션 플래닝 작업에 대한 평가에서 샘플링 기반 방법 대비 23%, 최적화 기반 방법 대비 17%, 신경망 모션 플래닝 방법 대비 79% 향상된 성공률을 달성했습니다.



### A Flexible Framework for Universal Computational Aberration Correction via Automatic Lens Library Generation and Domain Adaptation (https://arxiv.org/abs/2409.05809)
- **What's New**: 이 논문에서는 새로운 Universal Computational Aberration Correction (CAC) 프레임워크인 OmniLens를 제안합니다. 이 프레임워크는 다양한 렌즈 설계에 대한 적응을 위해 반복적인 데이터 준비 및 모델 훈련 없이도 고품질 이미징을 가능하게 합니다.

- **Technical Details**: OmniLens는 zero-shot CAC, few-shot CAC 및 domain adaptive CAC의 세 가지 상황으로 일반화된 CAC를 확장합니다. 또한, Evolution-based Automatic Optical Design (EAOD) 파이프라인을 사용하여 다양한 실제 왜곡 변화를 반영하는 LensLib을 자동으로 생성합니다. 이를 통해 훈련된 베이스 모델은 강력한 일반화 성능을 보여주며, 고품질 코드북 우선 (HQCP)를 활용하여 모델의 수렴 속도를 증가시킵니다.

- **Performance Highlights**: OmniLens는 4가지 수동 설계된 저가 렌즈를 통해 검증되었습니다. 특히, AODLib에서 훈련된 기본 모델은 zero-shot 환경에서 렌즈 특화 성능의 97%를 달성하며, FS-CAC에서는 훈련 데이터의 5%로도 렌즈 특화 모델보다 우수한 결과를 냅니다.



### Input Space Mode Connectivity in Deep Neural Networks (https://arxiv.org/abs/2409.05800)
- **What's New**: 이번 연구에서는 손실 경량 모드 연결성(mode connectivity) 개념을 깊은 신경망(deep neural networks)의 입력 공간(input space)으로 확장했습니다. 연구진은 특정 솔루션들이 어떻게 저손실(low-loss) 경로로 연결되어 있는지를 이론적 및 실증적으로 증명하며, 입력 이미지 간의 연결성을 조명합니다.

- **Technical Details**: 이 논문에서는 모델의 파라미터가 고정된 상태에서 다양한 입력을 최적화(input optimization)하여 생성된 실제, 보간(interpolated), 합성(synthetic) 입력을 활용했습니다. 입력 모드 간의 경로는 대개 단순하며 선형 경로에 가까운 형태임을 관찰했습니다.

- **Performance Highlights**: 이 연구는 주어진 입력에 대해 적대적 예제(adversarial examples)를 감지하는 새로운 방법을 제시하며, 고급 공격(예: DeepFool, Carlini-Wagner)에 대해 기존 방법보다 우수한 성능을 보입니다. 또한 입력 공간 모드 연결성이 높은 차원의 기하학적 현상으로 설명될 수 있음을 제안함으로써 딥러닝 모델의 해석 가능성도 높입니다.



### Creativity and Visual Communication from Machine to Musician: Sharing a Score through a Robotic Camera (https://arxiv.org/abs/2409.05773)
- **What's New**: 이 논문은 "Guided Harmony" 음악 게임 내에서 로봇 카메라를 구현하여 시각적 소통과 음악적 상호작용의 통합을 탐구합니다. 이는 인간 뮤지션과 로봇 시스템 간의 공동 창작 행동을 조사하는 것을 목표로 합니다.

- **Technical Details**: 로봇 시스템은 뮤지션의 비언어적 신호를 해석하고 반응하여 협력적이고 적응적인 음악적 경험을 창출합니다. PTZ(팬-틸트-줌) 카메라를 사용하여 시각적 신호를 해석하고, AI 즉흥 연주 시스템의 복잡성을 더할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 초기 사례 연구는 직관적인 시각 소통 채널의 중요성을 강조하며, 향후 연구 방향으로는 시각 신호 툴킷 개선을 위한 매개변수 설정 및 인간-기계 공동 창의성을 더 깊이 이해할 데이터 수집 방법론을 제안합니다.



### Consensus-based Distributed Quantum Kernel Learning for Speech Recognition (https://arxiv.org/abs/2409.05770)
- **What's New**: 이 논문에서는 분산 양자 커널 학습(Consensus-based Distributed Quantum Kernel Learning, CDQKL) 프레임워크를 제안하여 분산 양자 컴퓨팅을 통해 음성 인식을 개선하는 방법을 소개합니다. CDQKL은 중앙 집중식 양자 커널 학습의 확장성과 데이터 프라이버시 문제를 해결합니다.

- **Technical Details**: CDQKL은 양자 단말기 간의 모델 파라미터 교환을 통해 데이터 프라이버시를 유지하며, 계산 효율성을 높이는 분산 아키텍처를 구성합니다. 각 양자 단말기는 클래식 채널을 통해 연결되어 있으며, 지역 훈련 데이터를 공유하지 않고도 기능적인 모델 학습을 수행할 수 있습니다. 분산 구조는 음성 인식 등 민감한 데이터가 필요한 분야에서의 응용 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, CDQKL은 중앙 집중식 및 지역 양자 커널 학습 모델에 비해 경쟁력 있는 분류 정확도와 확장성을 달성하였습니다. CDQKL은 데이터 프라이버시를 유지하면서 효율적인 모델 학습을 가능하게 하여 통신, 자동차, 금융과 같은 데이터 민감한 분야에 적합합니다.



### Robust Loss Functions for Object Grasping under Limited Ground Truth (https://arxiv.org/abs/2409.05742)
- **What's New**: 이 논문은 로봇의 물체 grasping(잡기) 기술에서 누락되거나 노이즈가 있는 ground truth(기준 진실) 문제를 해결하기 위한 새로운 방법론을 제시합니다. 저자는 새로운 손실 함수(loss function)를 도입하여 정확한 학습을 지원하며, 이는 기존의 grasping 신경망에 쉽게 통합될 수 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 문제를 해결합니다. 첫째, 누락된 ground truth를 다루기 위해 새로운 예측 카테고리 확률 방법을 도입하여 레이블이 없는 샘플에 대해 더 나은 pseudo-label(유사 레이블)을 생성합니다. 둘째, 노이즈가 있는 ground truth에 대해 Симметричный 손실 함수(symmetrical loss function)를 소개하여 레이블 노이즈의 영향을 줄입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 grasping 신경망의 성능을 2%에서 13%까지 향상시킬 수 있음을 보여주었습니다. 이러한 성과는 로봇이 산업 환경에서 물체를 효과적으로 잡을 수 있도록 기여할 것입니다.



### Referring Expression Generation in Visually Grounded Dialogue with Discourse-aware Comprehension Guiding (https://arxiv.org/abs/2409.05721)
Comments:
          Accepted for publication at INLG 2024

- **What's New**: 이 논문에서는 시각적 기반 대화에서의 참조 표현 생성(REF)의 새로운 접근법을 제안합니다. 제안된 방법은 식별적이며 대화에 적합한 RE를 생성하기 위한 두 단계의 과정으로 구성됩니다.

- **Technical Details**: 첫 번째 단계에서는 REG를 텍스트 및 이미지 조건부의 다음 토큰 예측 작업으로 모델링합니다. 이후 RE는 이전 언어적 맥락과 참조 대상의 시각적 표현을 바탕으로 자가 회귀적으로 생성됩니다. 두 번째 단계에서는 대화 맥락을 보고 후보 RE의 선별을 위해 대화 인식(comprehension guiding)을 적용합니다.

- **Performance Highlights**: 인간 평가 결과, 제안된 두 단계 접근법이 더 높은 성능을 보여주었으며, 선택된 RE의 텍스트-이미지 검색 정확도가 더 높았습니다. 이는 기존의 탐욕적 디코딩 방법으로 생성된 RE에 비해 월등히 우수한 성과를 나타냅니다.



### Cherenkov Imaged Bio-morphological Features Verify Patient Positioning with Deformable Tissue Translocation in Breast Radiotherapy (https://arxiv.org/abs/2409.05680)
Comments:
          25 pages, 4 figures, 1 table, journal under review

- **What's New**: 이번 연구는 유방암 방사선 치료 중의 지역 조직 변형을 Cherenkov 이미지 분석을 통해 추적하는 새로운 방법을 제시합니다.

- **Technical Details**: 연구의 주된 목표는 Cherenkov 기반의 지역 위치 정확도 정량화 알고리즘을 개발하고 검증하는 것이었으며, 이를 통해 혈관 탐지 및 세분화 기술이 개발되었습니다. 이 기술은 전체 유방 방사선 치료에서 얻은 이미지에 적용되었습니다. 또한, 결합된 강체(rigid) 및 비강체(non-rigid) 등록 기법을 사용하여 치료 간 및 치료 내 위치 변이를 감지했습니다.

- **Performance Highlights**: 실험 결과, 알려진 치료 소파 번역 및 호흡 운동을 시뮬레이션한 인간형 흉부 팬텀 실험에서 평균 정확도가 0.83 mm로 나타났습니다. 10명의 유방암 환자의 임상 Cherenkov 데이터 분석 결과, 첫 번째 치료에 비해 평균 3.7 ± 2.4 mm의 치료 간 설치 변동과 지역 변형이 3.3 ± 1.9 mm로 관찰되었습니다.



### Robust Real-time Segmentation of Bio-Morphological Features in Human Cherenkov Imaging during Radiotherapy via Deep Learning (https://arxiv.org/abs/2409.05666)
Comments:
          9 pages, 7 figures, 1 table, journal under review

- **What's New**: 이 연구는 방사선 치료(Radiation Therapy, RT) 중 메가볼트 X선 또는 전자 빔의 실시간 시각화를 위한 최초의 딥러닝 프레임워크를 제안했습니다.

- **Technical Details**: 전통적인 이미지 처리의 느린 속도와 정확성을 극복하기 위해, 20,529개의 패치 망막 이미지로 구성된 fundus photography 데이터 세트를 사용하여 ResNet 세분화(segmentation) 프레임워크를 사전 훈련(pre-train)했습니다. 이후, 19명의 유방암 환자로부터 수집된 1,483개의 Cherenkov 이미지 데이터 세트를 사용하여 모델을 미세 조정(fine-tune)하였습니다.

- **Performance Highlights**: 모델은 0.85의 Dice 점수를 달성했으며, 각 인스턴스당 0.7 밀리초(0.7 milliseconds)보다 짧은 처리 시간을 요구했습니다. 이 접근법은 전통적인 수동 세분화 방법에 비해 일관성 및 속도 면에서 뛰어난 성능을 보였습니다.



### A Taxonomy of Miscompressions: Preparing Image Forensics for Neural Compression (https://arxiv.org/abs/2409.05490)
Comments:
          6 pages, 6 figures

- **What's New**: 이 논문은 Neural compression(신경 압축)의 특성과 그에 따른 miscompression(잘못된 압축) 현상을 탐구합니다. 이들은 고품질의 시각적 재현을 제공하나, 원래 의미와 다르게 해석될 수 있어 중요한 문제로 떠오르고 있습니다.

- **Technical Details**: 논문에서는 Neural compression의 원리와 그것이 이미지 압축 파이프라인에서 어떻게 구현되는지를 비교합니다. 주요 기술에는 Deep Convolutional Networks(CNNs), 변형 생성 모델인 GANs(Generative Adversarial Networks), 그리고 DDPM(Denoising Diffusion Probabilistic Models)이 포함되어 있습니다. 또한, miscompressions의 분류 체계를 개발하여 잘못된 압축에 대한 연구의 필요성을 강조합니다.

- **Performance Highlights**: Neural compression 시스템은 압축 속도와 품질 면에서 혁신적인 성능을 보여주고 있지만, 이미지에서의 semantic fidelity(의미적 충실도)의 손실이 발생할 수 있습니다. 이는 고품질 이미지가 인간의 이해와 정체성을 왜곡할 수 있음을 의미하며, 다양한 사회적 및 법의학적 영향에 대한 논의가 필요합니다.



### CipherDM: Secure Three-Party Inference for Diffusion Model Sampling (https://arxiv.org/abs/2409.05414)
- **What's New**: 이번 논문에서는 Diffusion Models (DMs)의 개인 정보 보호 문제를 해결하기 위해 CipherDM이라는 새로운 프레임워크를 제안합니다. 이는 Secure Multi-Party Computation (MPC) 기술을 활용하여 DM의 샘플링 과정을 안전하게 수행할 수 있도록 합니다. 이러한 접근은 기존 DMs에서 발생할 수 있는 사용자 프라이버시 침해를 방지하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: CipherDM은 ABY3 프로토콜을 기반으로 하는 최초의 MPC 기반 DM 샘플링 프레임워크로, 3PC (Three-Party Computation) 복제 비밀 공유 방식을 사용하여 세 개의 클라우드 서버에서 샘플링을 수행합니다. 샘플링 과정에서 개인 데이터나 모델 파라미터가 외부에 노출되지 않도록 보장합니다. 추가적으로, 이 연구는 SoftMax, SiLU, Mish와 같은 비선형 활성화 기능에 대한 안전한 MPC 프로토콜을 설계합니다.

- **Performance Highlights**: CipherDM은 MNIST 데이터셋과 diffusers에 의해 배포된 Stable Diffusion (SD)에서 DDPM 및 DDIM 아키텍처를 사용하여 평가되었습니다. SPU에 직접 구현한 것에 비해 실행 시간은 약 1.084배에서 2.328배 개선되었으며, 통신 비용은 약 1.212배에서 1.791배 감소했습니다. 이는 DM 안전한 샘플링을 위한 MPC의 유효성을 입증하는 결과입니다.



### A Multi-Modal Deep Learning Based Approach for House Price Prediction (https://arxiv.org/abs/2409.05335)
Comments:
          22 pages

- **What's New**: 이 논문은 주택 가격 예측 시스템에서 텍스트와 비주얼 특성을 포함한 다양한 속성을 종합적으로 통합하여 향상된 정확도를 목표로 합니다. 특히, 다중 모달 딥러닝 접근법을 통해 집의 속성, 지리적 특성, 텍스트 설명 및 이미지의 조합을 활용합니다.

- **Technical Details**: 제안하는 Multi-Modal House Price Predictor (MHPP)는 집의 속성, 지리적 이웃, 텍스트 설명, 이미지로부터의 임베딩을 학습합니다. 이를 위해 BERT와 CLIP 모델을 활용하여 세 가지 유형의 데이터에서의 최적 조합 임베딩을 학습하여 다운스트림 회귀 모델에 입력합니다.

- **Performance Highlights**: 실험 결과, 텍스트 설명에 대한 임베딩과 집 사진의 임베딩을 포함한 접근이 주택 가격 예측의 정확도를 크게 향상시킴을 보여줍니다. 52,851개의 멜버른 주택 거래 데이터셋을 활용하여 이러한 결과를 확인했습니다.



### Neural Surface Reconstruction and Rendering for LiDAR-Visual Systems (https://arxiv.org/abs/2409.05310)
- **What's New**: 이 논문은 LiDAR-비주얼 시스템을 위한 통합된 surface reconstruction과 rendering 프레임워크를 제안합니다. Neural Radiance Fields (NeRF)와 Neural Distance Fields (NDF)를 통합하여 포즈가 부여된 이미지와 포인트 클라우드로부터 외관과 구조 정보를 모두 복구하는 방법을 다룹니다. 또한, 가시성을 고려한 occupancy map을 사용하여 공간을 분류하여 전체 외관과 구조를 복원할 수 있습니다.

- **Technical Details**: 제안된 방법은 spatial-varying scale SDF-to-density 변환을 사용하여 NDF와 NeRF의 훈련을 통합합니다. NDF는 NeRF 훈련을 위해 구조 인식 비구조적 채집 전략을 활용하여 구조 렌더링을 정확하게 수행하며, NeRF는 NDF 내 누락되거나 흐릿한 구조를 복원하는 데 기여합니다. 이러한 방법으로, 시각적으로 가시한 오프닝은 occupancy map을 통해 가구분할(categorize)되어 구조적 일관성을 강화하고 렌더링 품질을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법의 뛰어난 품질과 다양한 시나리오에서의 유연성을 입증했습니다. 이 방법은 높은 세밀도를 갖춘 surface reconstruction과 photorealistic rendering을 가능하게 하며, LiDAR 및 비주얼 데이터의 결합을 통해 실용적인 3D 재구성 데이터 수집 상황에서 효과적인 성능을 보여줍니다.



### Rethinking the Atmospheric Scattering-driven Attention via Channel and Gamma Correction Priors for Low-Light Image Enhancemen (https://arxiv.org/abs/2409.05274)
- **What's New**: 이 논문에서는 저조도 이미지 개선 문제를 해결하기 위해 CPGA-Net의 확장 버전인 CPGA-Net+를 제안하고 있으며, 이는 Atmospheric Scattering Model을 기반으로 한 주의(attention) 메커니즘을 통합하여 전 세계적(global) 및 지역적(local) 이미지 처리에서 우수한 성능을 발휘합니다.

- **Technical Details**: CPGA-Net+는 플러그인(pug-in) 주의(attention)와 감마 보정(gamma correction)을 통해 채널 우선 정보(channel prior information)를 개선하고 이미지 구조와 세부 정보를 잘 보존합니다. 또한 저조도 환경에서의 효율성을 극대화하기 위한 경량화(lightweight) 설계가 특징입니다.

- **Performance Highlights**: CPGA-Net+는 이미지 개선 작업에서 최신 경량 기술들에 비해 우수한 성능을 보여주며, 자원이 제한된 환경에서도 효과적으로 활용될 수 있는 가능성을 나타냅니다.



### Towards Automated Machine Learning Research (https://arxiv.org/abs/2409.05258)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 도움으로 기계 학습 연구의 점진적 발전을 자동화하는 탑다운(top-down) 접근 방식이 제시됩니다. 이 프레임워크는 새롭고 실행 가능한 구성 요소를 체계적으로 생성하고, 이를 기존 기준선과 비교하여 평가합니다.

- **Technical Details**: 이 방법은 전통적인 AutoML(Automated Machine Learning) 및 NAS(Neural Architecture Search)와는 달리 정의된 하드코딩된 기본 구성 요소에 대한 조합 검색에 의존하지 않고, LLMs에 내재된 도메인 간 지식을 활용하여 새로운 구성 요소를 제안합니다. 이를 통해 가설 생성 및 검증 과정을 효율적으로 개선하며, 성공적인 가설의 공통 패턴을 확인하는 보상 모델을 학습합니다.

- **Performance Highlights**: 이 연구는 실현 가능성 있는 구성 요소를 제안하고 평가하여, 기존 대안과 경쟁하는 성능을 달성하는 것을 목표로 합니다. 이 프레임워크는 자동화된 연구를 통해 혁신을 가속화하고 다양한 과학적 발견 작업에 적용될 수 있는 가능성을 탐구합니다.



### Label-free evaluation of lung and heart transplant biopsies using virtual staining (https://arxiv.org/abs/2409.05255)
Comments:
          21 Pages, 5 Figures

- **What's New**: 이 연구에서는 전통적인 조직 염색(process) 과정을 대체할 수 있는 가상 염색 신경망(virtual staining neural networks)를 제시합니다. 이는 폐와 심장 이식 생검(transplant biopsies)의 자가형광(autofluorescence) 미세 이미지(microscopic images)를 디지털 방식으로 변환하여 생리학적 염색(histologically stained) 이미지로 전환하는 혁신적인 방법입니다.

- **Technical Details**: 이 신경망은 Hematoxylin and Eosin (H&E), Masson's Trichrome (MT), Elastic Verhoeff-Van Gieson (EVG) 염색을 가상으로 생성하며, 염색이 필요 없는(label-free) 조직을 사용합니다. 세 명의 board-certified 병리학자(pathologists)에 의해 진행된 블라인드 평가(blind evaluations)에서는 가상 염색 이미지가 전통적으로 염색된 조직과 높은 색상 균일성(color uniformity)을 유지하는 것으로 확인되었습니다.

- **Performance Highlights**: 가상 염색 이미지를 이용한 이식 생검의 평가 결과는 전통적인 염색 방식과 유사한 진단 성능을 보였고, 폐 샘플의 일치율은 82.4%, 심장 샘플에서는 91.7%에 달했습니다. 이러한 가상 염색 모델은 동일한 자가형광 입력으로부터 여러 가지 염색을 생성할 수 있어 조직 손실(tissue loss), 전문가 시간(consumption of expert time), 염색 비용(staining costs)을 절감할 수 있는 이점을 제공합니다.



### A Survey on Mixup Augmentations and Beyond (https://arxiv.org/abs/2409.05202)
Comments:
          Preprint V1 with 27 pages main text. Online project at this https URL

- **What's New**: 딥러닝(Deep Learning) 분야에서 Mixup 방법이 데이터 증강(Data Augmentation) 기법으로 주목받고 있으며, 본 논문에서는 Mixup의 다양한 적용 사례와 이론적 배경을 포괄적으로 정리하였습니다.

- **Technical Details**: Mixup은 두 개의 샘플(sample)과 해당 레이블(label)을 선형 조합하여 새로운 가상 샘플을 생성하는 방식입니다. 이 방법은 대량의 훈련 데이터를 요구하지 않으면서도 다양한 도메인에 효과적으로 적용 가능합니다.

- **Performance Highlights**: Mixup은 기존의 데이터 증강 기법들과 비교해 모델의 일반화 성능을 향상시키며, Supervised Learning(SL), Self-Supervised Learning(SSL), Semi-Supervised Learning(Semi-SL) 등 다양한 훈련 패러다임에 성공적으로 적용되고 있습니다.



### Exploring Fungal Morphology Simulation and Dynamic Light Containment from a Graphics Generation Perspectiv (https://arxiv.org/abs/2409.05171)
Comments:
          Siggraph Asia 2024 Art Paper

- **What's New**: 본 연구는 생물 예술(Bio-Art)에서 균류(mold) 시뮬레이션과 제어를 위한 새로운 접근 방식을 제안합니다. 이는 프로그래밍 코드 없이도 딥러닝(Deep Learning) 기반의 셀룰러 오토마타(cellular automaton)를 활용하여 균류의 확산 패턴을 시뮬레이션할 수 있도록 합니다.

- **Technical Details**: 본 연구에서는 E-ViT 이미지 분할 모델 및 TCN(Temporal Convolutional Network) 모델을 사용하여 균류의 확산 패턴을 학습합니다. 이 과정에서 생성된 모델은 시뮬레이션을 통한 실제 형태 복제를 가능하게 하며, 레이저를 통해 균류의 경계를 동적으로 조절하여 설계된 복잡한 형태로 확산을 유도합니다.

- **Performance Highlights**: 이 연구의 접근 방식은 아티스트들이 프로그래밍이나 복잡한 알고리즘에 대한 지식 없이도 균류 형태 시뮬레이션을 수행할 수 있게 하여, 진정성 높은 결과를 얻을 수 있도록 합니다. 또한, E-ViT 모델은 다양한 조명과 색상 조건에서도 균류의 윤곽을 정확히 식별하고, 경량화된 설계로 인해 실행 장벽을 낮췄습니다.



### Better Spanish Emotion Recognition In-the-wild: Bringing Attention to Deep Spectrum Voice Analysis (https://arxiv.org/abs/2409.05148)
- **What's New**: 본 연구는 감정 인식이 중요한 사회적 지원 로봇(Socially Assistive Robots, SAR)의 발전에 기여하는 것을 목표로 하며, 스페인어 음성 데이터셋인 ELRA-S0329와 EmoMatchSpanishDB를 분석하였습니다. 이 연구는 패럴랭귀지(paralanguage)와 딥러닝 기법인 DeepSpectrum을 활용하여 음성의 시각적 표현을 추출하고 사전 훈련된 CNN 모델에 입력합니다.

- **Technical Details**: DeepSpectrum 방법론은 오디오 트랙의 시각적 표현을 추출하고 이를 SVC(Support Vector Classifier) 또는 FC(Fully-Connected deep-learning classifier)에 입력하여 감정을 분류합니다. 연구에서는 Attention Mechanism을 기반으로 한 새로운 분류기인 DS-AM을 제안하였으며, ELRA-S0329과 EmoMatchSpanishDB 두 데이터셋에서 비교 실험을 수행했습니다.

- **Performance Highlights**: DS-AM 모델은 모든 SOTA(state-of-the-art) 모델과 비교하여 우수한 성능을 보였으며, 데이터셋 간의 훈련 및 테스트를 통해 모델의 편향성을 확인했습니다. 두 데이터셋 비교 결과, EmoMatchSpanishDB가 더 적절한 감정 인식 성능을 보였습니다.



### READoc: A Unified Benchmark for Realistic Document Structured Extraction (https://arxiv.org/abs/2409.05137)
- **What's New**: 이 논문은 Document Structured Extraction (DSE)의 평가를 위한 새로운 벤치마크인 READoc을 소개합니다. 기존의 DSE 시스템들이 단편적으로 평가되고 있어 발전이 어려운 상황에서, READoc은 비구조화된 PDF 문서를 의미론적으로 풍부한 Markdown으로 변환하는 현실적인 작업으로 DSE를 정의합니다.

- **Technical Details**: READoc 데이터셋은 arXiv와 GitHub에서 수집된 2,233개의 다양한 실세계 문서에서 유래되었습니다. 이를 통해 PDF에서 Markdown으로의 전환을 목표로 하는 DSE 시스템의 과정을 평가하기 위한 DSE Evaluation S$^3$uite (Standardization, Segmentation, Scoring)를 개발했습니다.

- **Performance Highlights**: 다양한 최신 DSE 접근 방식을 평가하는 과정에서, 기존 연구들과 현실적인 DSE 목표 간의 격차를 찾아내고, READoc이 DSE 연구의 기반을 제공할 것이라고 기대하고 있습니다.



### Enhancing Convolutional Neural Networks with Higher-Order Numerical Difference Methods (https://arxiv.org/abs/2409.04977)
- **What's New**: 딥 러닝 기술의 발전과 함께 Convolutional Neural Networks (CNNs)가 실제 문제 해결에 큰 도움을 주고 있습니다. 본 논문은 모델 크기와 환경적 제약을 극복하면서 ResNet의 성능을 향상시키는 새로운 stacking scheme을 제안합니다.

- **Technical Details**: 이 연구는 일반적인 미분 방정식을 이산화(discretization)하는 방식으로 많은 CNN 구조를 설명합니다. 저자들은 선형 다단계 수치 차분 방법(linear multi-step numerical difference methods)을 활용하여 이론적으로 지지된 딥 네트워크 구조를 설계할 수 있음을 강조하고 있습니다. 또한, 제안된 stacking scheme은 기존의 Runge-Kutta 방법과 비교하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 stacking scheme이 ResNet 및 HO-ResNet과 같은 기존 stacking scheme에 비해 우수한 성능을 보이며, 다른 유형의 신경망에도 확장 가능성을 가지고 있음을 보여줍니다.



### Attention-Based Efficient Breath Sound Removal in Studio Audio Recordings (https://arxiv.org/abs/2409.04949)
- **What's New**: 본 연구에서는 음성 녹음에서 비음성 발화 소음, 특히 호흡 소음을 자동으로 감지하고 제거하는 혁신적이고 파라미터 효율적인 모델을 제안합니다. 이 모델은 attention U-Net 아키텍처를 기반으로 하여, 기존 방법의 한계를 극복하고 더 나은 정확성을 제공합니다.

- **Technical Details**: 우리 연구의 중심은 attention 메커니즘을 강조한 특별한 U-Net 모델입니다. 이 모델은 Short Time Fourier Transform (STFT)으로부터 생성된 스펙트로그램을 입력으로 사용합니다. 모델은 3872x2048x1의 입력 형태를 가지며, 16개의 필터와 배치 정규화를 포함하고, 20%의 드롭아웃 비율을 사용합니다. 훈련을 위해 Mean Absolute Error (MAE) 및 음성을 보존하는 추가 항목을 포함한 맞춤형 손실 함수를 설계했습니다.

- **Performance Highlights**: 우리 모델은 1.9M의 파라미터와 3.2시간의 훈련 시간으로 최고 성능 모델보다 훨씬 효율적입니다. 이전 모델과 동일한 출력을 생성하면서도 정확도가 크게 향상되었습니다. 이로 인해 음향 엔지니어가 귀중한 시간을 절약하고 오디오 생산의 품질 및 일관성을 높일 수 있습니다.



### Structure-Invariant Range-Visual-Inertial Odometry (https://arxiv.org/abs/2409.04633)
Comments:
          IEEE/RSJ International Conference on Intelligent Robots (IROS), 2024

- **What's New**: 이번 연구는 Mars Science Helicopter (MSH) 임무를 위한 새로운 range-visual-inertial odometry 시스템을 소개합니다. 기존 임무와 달리 MSH는 복잡한 지형 조건에서 작동할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 시스템은 xVIO 프레임워크를 확장하여 1D-LRF 측정값을 시각 및 관성 측정값과 융합함으로써, 비슷한 시각-관성 자극이 없을 때에도 메트릭 스케일 드리프트를 방지합니다. 이를 통해 다양한 지형에 안전하게 착륙할 수 있는 기능을 제공합니다.

- **Performance Highlights**: 시뮬레이션을 포함한 extensive 테스트 결과, 제안된 range-VIO 접근 방식은 지형 상대 속도를 정확히 추정하며 목표 요구 사항을 충족하였고, 기존의 방법들보다 우수한 성능을 보여주었습니다.



### Zero-Shot Whole Slide Image Retrieval in Histopathology Using Embeddings of Foundation Models (https://arxiv.org/abs/2409.04631)
Comments:
          This paper will be updated with more results

- **What's New**: 본 논문에서는 최근 발표된 기초 모델(base models)을 사용하여 조직병리학(histopathology) 이미지 검색을 테스트하였습니다. 이전에 발표되지 않은 제로샷(zero-shot) 검색 방식을 적용하였으며, 이는 임베딩(embeddings)을 변경하지 않고 어떠한 분류기(classifier)도 훈련시키지 않았음을 의미합니다.

- **Technical Details**: 테스트 데이터로는 TCGA(The Cancer Genome Atlas)에서 제공하는 진단 슬라이드를 사용하였으며, 23개 장기와 117개 암 하위 유형을 포함하고 있습니다. 검색 플랫폼으로는 패치를 이용한 WSI(Whole Slide Imaging) 검색을 수행할 수 있는 Yottixel을 사용하였습니다.

- **Performance Highlights**: 최상위 5개 검색에 대한 F1 점수는 27% +/- 13% (Yottixel-DenseNet), 42% +/- 14% (Yottixel-UNI), 40% +/- 13% (Yottixel-Virchow), 41% +/- 13% (Yottixel-GigaPath)로 나타났습니다. GigaPath WSI의 결과는 처리에 필요한 상당한 계산 자원으로 인해 지연될 예정입니다.



### A Short Survey on Set-Based Aggregation Techniques for Single-Vector WSI Representation in Digital Pathology (https://arxiv.org/abs/2409.04615)
- **What's New**: 이 논문은 디지털 병리학에서 Whole Slide Images (WSIs)의 집합 기반 접근 방법을 통해 단일 벡터 형태로 WSIs를 표현하는 혁신적인 방법을 다룹니다.

- **Technical Details**: WSIs는 기가픽셀(gigapixel) 파일로, 조직 샘플의 세부 정보를 정밀하게 캡처하며, 이를 효과적으로 처리하기 위해 'patch-oriented' 방법론을 탈피하고 단일 벡터 표현 방식으로 전환하는 것이 필요합니다. 이 접근 방식은 더 효율적이고 효과적인 디지털 병리학적 분석을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 대용량 데이터를 저장하고 분석하는 데 필요한 고성능 스토리지를 요구하지 않으면서도, 병원들 간의 헬스케어 품질 및 접근성의 격차를 해소할 수 있는 잠재력을 가지고 있습니다.



### NeCA: 3D Coronary Artery Tree Reconstruction from Two 2D Projections by Neural Implicit Representation (https://arxiv.org/abs/2409.04596)
Comments:
          16 pages, 10 figures, 6 tables

- **What's New**: 본 논문에서는 두 개의 2D 프로젝션을 기반으로 3D 관상동맥 나무를 재구성하기 위한 자기 지도 학습(self-supervised learning) 방법인 NeCA를 제안합니다. NeCA는 3D 지상 진실(ground truth)이나 대규모 학습 데이터셋이 필요하지 않으며, 이는 임상 적용에서 큰 장점을 지닙니다.

- **Technical Details**: NeCA 방법은 implicit neural representation에 기초하여 작동하며, multiresolution hash encoder와 differentiable cone-beam forward projector layer를 활용하여 3D 재구성을 수행합니다. 이 모델은 모든 이미지에 대한 신경 표현을 최적화하며, 주어진 환자의 프로젝션 데이터만을 입력으로 사용하여 재구성을 반복적으로 최적화합니다.

- **Performance Highlights**: NeCA 모델은 실제 데이터셋에 대해 벤치마크 평가를 수행하여 관상동맥 구조의 토폴로지를 유지하고 가지 연결성을 유지하는 데 있어 기존의 감독된 학습 모델에 비해 비교적 우수한 성능을 보였습니다. 이는 3D 지상 진실이 없고 대규모 데이터셋이 필요하지 않기 때문에 임상 환경에서 유용하게 활용될 수 있습니다.



### Diff-INR: Generative Regularization for Electrical Impedance Tomography (https://arxiv.org/abs/2409.04494)
- **What's New**: 본 연구에서는 Electrical Impedance Tomography (EIT)의 재구성을 개선하기 위해 Diff-INR이라는 새로운 방법을 제안합니다. 이 방법은 Diffusion 모델과 Implicit Neural Representations (INR)을 결합하여 기존의 정규화 기법의 한계를 극복할 수 있도록 합니다.

- **Technical Details**: Diff-INR는 Generative regularization과 geometric priors를 통합하여 EIT 재구성에서 발생하는 ill-posed nonlinear inverse problem을 효과적으로 해결합니다. 이 방법은 이전에 훈련된 diffusion regularizer를 INR과 결합하여 정확한 결과를 도출할 수 있도록 합니다.

- **Performance Highlights**: Diff-INR은 시뮬레이션 및 실험 데이터를 통해 state-of-the-art 수준의 재구성 정확성을 달성하였으며, 다양한 메쉬 밀도와 하이퍼파라미터 설정에서도 강력한 성능을 보였습니다. 이로 인해 EIT의 ill-posed 특성을 관리하는 데 중요한 발전을 이루었습니다.



### Pattern based learning and optimisation through pricing for bin packing problem (https://arxiv.org/abs/2409.04456)
- **What's New**: 본 논문에서는 데이터 마이닝의 패턴 식별 과정에서 패턴의 동적 가치에 대한 체계적인 분석을 진행합니다. 특히 랜덤 변수의 분포와 같은 문제 조건이 변화할 때 패턴의 효과가 떨어질 수 있음을 강조합니다.

- **Technical Details**: 본 연구는 운영 연구(operations research)의 쌍대성 이론(duality theory)과 데이터 마이닝을 연결하여 패턴을 효율적으로 식별하고 조건에 따라 동적으로 그 가치를 정량화하는 새로운 방법론을 제안합니다. 이 방법은 패턴이 확률적 제약(stochastic constraints)을 만족시키는 능력과 목적 함수(objective value) 미치는 영향을 바탕으로 패턴의 가치를 정량화합니다.

- **Performance Highlights**: 제안된 알고리즘은 온라인 빈 포장 문제(online bin packing problem)를 통해 기존의 최첨단 방법들(state-of-the-art methods)보다 우수한 성과를 보이며, 제안된 방법의 고유 특성과 성능 개선 사항을 상세히 분석하였습니다.



### A Greedy Hierarchical Approach to Whole-Network Filter-Pruning in CNNs (https://arxiv.org/abs/2409.03777)
Comments:
          Accepted in TMLR 2024

- **What's New**: 본 논문에서는 필터 가지치기(filter pruning)를 위한 효율적인 계층적 접근법을 제안하며, CNN 모델의 전체 네트워크에서 필터를 제거하는 새로운 방법론을 소개합니다. 이 방법은 최종 기준으로 분류 손실(classification loss)을 사용하여 더 빠르고 정확한 필터 가지치기를 수행합니다.

- **Technical Details**: 제안된 방법은 두 가지 단계로 구성됩니다. 하위 단계인 필터 가지치기(Filter-Pruning)는 필터 가중치의 선형 근사(linear approximation)를 기반으로 희소 근사(sparse-approximation) 공식을 사용합니다. 상위 단계인 레이어 선택(Layer-Selection)은 최적의 가지치기 레이어를 선택하는 알고리즘으로, 두 가지 전역 가지치기 기준(global pruning criterion)을 사용합니다: (1) 레이어-wise 상대적 오차(HBGS)와 (2) 최종 분류 오차(HBGTS).

- **Performance Highlights**: 이 방법은 ResNet18, ResNet32, ResNet56, VGG16 및 ResNext101 모델에서 최신 가지치기 방법들보다 높은 성능을 보였습니다. 특히 ResNext101의 경우 RAM 요구량을 7.6GB에서 1.5GB로 줄였으며, CIFAR-10에서 정확도 손실 없이 FLOPS를 94% 감소시켰습니다.



### Estimating Indoor Scene Depth Maps from Ultrasonic Echoes (https://arxiv.org/abs/2409.03336)
Comments:
          ICIP 2024

- **What's New**: 이번 연구에서는 인식 가능한 울림(echo) 대신 인식할 수 없는 초음파 기반 깊이 추정(deep estimation)을 적용하였습니다. 이는 조용한 공간에서 사용할 수 있는 대안으로 제안되었습니다.

- **Technical Details**: 초음파를 사용한 깊이 추정에서 발생하는 정확도 감소 문제를 해결하기 위해 합성 데이터(synthetic data)를 생성하여 훈련 중 식별 가능한 울림을 보조 데이터로 사용합니다. 이를 통해 깊이 추정 네트워크의 강건성을 높이고 이 방법의 성능 개선을 도모하였습니다.

- **Performance Highlights**: 실험 결과, 우리 방법은 Replica 데이터셋을 사용하여 초음파를 기반으로 한 깊이 추정의 정확성을 개선했습니다. 구체적으로, 다양한 주파수 설정에 대해 평균 RMSE(root mean square error)가 개선되었습니다.



### Towards Generative Class Prompt Learning for Fine-grained Visual Recognition (https://arxiv.org/abs/2409.01835)
Comments:
          Accepted in BMVC 2024

- **What's New**: 이번 연구는 Generative Class Prompt Learning (GCPL)과 Contrastive Multi-class Prompt Learning (CoMPLe)이라는 두 가지 새로운 방법을 제안하여 기존의 VLM(Visual Language Model)들의 세밀한 분류 성능을 향상시킵니다. 이는 text-to-image diffusion 모델을 활용하여 적은 예제에 대한 클래스 프롬프트를 학습 가능한 형태로 조건화하는 방식입니다.

- **Technical Details**: GCPL은 클래스 임베딩에서 비주얼-언어 시너지를 크게 향상시키며, CoMPLe는 생성 최적화 과정에서 클래스 간 분리를 촉진하는 대조 학습(Contrastive Learning) 요소를 도입합니다. 이들 방식은 세밀한 이미지 인식의 도전 과제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 여기서 제안한 생성적 클래스 프롬프트 학습 접근법이 기존의 방법들을 상당히 초월하는 성능을 보여주었으며, 적은 샷(Few-shot) 이미지 인식 과제에서도 효과적인 대안이 될 수 있음을 입증했습니다.



New uploads on arXiv(cs.AI)

### Applying Attribution Explanations in Truth-Discovery Quantitative Bipolar Argumentation Frameworks (https://arxiv.org/abs/2409.05831)
Comments:
          This paper has been accepted at ArgXAI Workshop 2024

- **What's New**: 본 논문에서는 Truth Discovery QBAFs (TD-QBAFs)에서 Argument Attribution Explanations (AAEs)와 Relation Attribution Explanations (RAEs)의 적용 가능성을 조사하였습니다. 이 연구는 주로 순환 구조를 가진 QBAFs를 다루며, 기존 연구와는 달리 AAEs와 RAEs의 직접 비교를 시도합니다.

- **Technical Details**: Quantitative Bipolar Argumentation Frameworks (QBAFs)는 전통적인 인수 기반 논증 프레임워크의 확장으로, 주장 간의 지지 관계와 공격 관계를 고려합니다. AAEs는 주제 주장의 기여도를 평가하는 반면, RAEs는 관계에 대한 기여도를 평가합니다. 일반적으로 제거 기반 기술과 Shapley 기반 기술이 사용됩니다. QE(Quadratic Energy) 점진적 의미론을 통해 주장들의 강도를 평가합니다.

- **Performance Highlights**: 연구 결과, AAEs와 RAEs 모두 TD-QBAFs에서 흥미로운 설명을 제공하고, 신뢰 등급의 비선형적인 통찰을 제시할 수 있음을 발견했습니다. 이는 다양한 응용 프로그램에서의 신뢰성 평가 및 주장 검증에 기여할 수 있는 가능성을 보여줍니다.



### CauseJudger: Identifying the Cause with LLMs for Abductive Logical Reasoning (https://arxiv.org/abs/2409.05559)
- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)을 위한 새로운 추론 프레임워크, CauseJudger (CJ)를 제안합니다. 이는 역추론(reverse reasoning)에서 정방향 추론(forward reasoning)으로 전환하고 무관한 정보(irrelavant information)를 제거하여 가능성 있는 원인의 진위(authenticity)를 판단하는 방식입니다.

- **Technical Details**: CauseJudger는 세 가지 모듈로 구성됩니다: (1) Logic Reverse Module (LRM) - 역사고를 정방향 사고로 변환합니다. (2) Information Pruning Module (IPM) - 무관한 정보의 간소화(pruning)를 통해 방해 요소를 제거합니다. (3) Forward Reasoning Module (FRM) - LLM을 이용해 정방향으로 추론을 수행하여 최종 답안을 출력합니다.

- **Performance Highlights**: CJ는 gpt-3.5를 사용할 때 최대 41%의 정확도 향상을 보였으며 Zero-Shot-CoT와 비교했습니다. 또한 gpt-4를 사용할 경우 모든 데이터셋에서 90% 이상의 정확도를 달성했습니다.



### SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning (https://arxiv.org/abs/2409.05556)
- **What's New**: 이 연구는 자율적으로 과학적 이해를 진전시킬 수 있는 시스템의 개발을 목표로 하는 SciAgents 접근법을 제시합니다.

- **Technical Details**: (1) 대규모 온톨로지 지식 그래프(ontological knowledge graphs)를 활용하여 다양한 과학적 개념을 조직하고 연결합니다. (2) 대형 언어 모델(large language models, LLMs) 및 데이터 검색 도구(data retrieval tools) 모음을 사용합니다. (3) 현장 학습(in-situ learning) 기능을 갖춘 다중 에이전트 시스템(multi-agent systems)을 통합합니다.

- **Performance Highlights**: SciAgents는 생물 영감을 받은 재료 재료에 적용되어 새로운 학제 간 관계를 드러내며 기존의 연구 방법을 초월한 규모, 정밀도 및 탐사 능력을 달성합니다. 이 프레임워크는 자율적으로 연구 가설을 생성 및 정제하고, 기본 메커니즘 및 디자인 원칙을 설명하며, 예기치 않은 물질 속성을 밝혀냅니다.



### Semifactual Explanations for Reinforcement Learning (https://arxiv.org/abs/2409.05435)
Comments:
          9 pages, 2 figures, 4 tables

- **What's New**: 이번 연구에서는 강화학습(RL) 에이전트의 결정을 설명하기 위한 새로운 접근법인 semifactual 설명(sSémi factual explanations)을 제안합니다. 이는 RL 시스템의 결정을 설명하기 위해 최초로 사용된 방식입니다.

- **Technical Details**: 강화학습의 효용성을 높이기 위해 SGRL-Rewind 및 SGRL-Advance라는 두 가지 알고리즘을 개발하였습니다. SGRL-Rewind는 과거의 변화를 탐색하여 결과를 유지하는 반면, SGRL-Advance는 미래의 행동을 탐색하여 결과를 유지하는 방법을 제안합니다.

- **Performance Highlights**: 이 알고리즘들은 두 가지 RL 환경에서 평가되었으며, 그 결과 semifactual 설명들이 보다 높은 충실도를 유지하며 다양한 결과를 제공하는 것으로 나타났습니다. 또한, 사용자 연구를 통해 참가자들이 RL 에이전트의 행동에 대한 semifactual 설명을 긍정적으로 인식했음을 확인하였습니다.



### Dynamic Demand Management for Parcel Lockers (https://arxiv.org/abs/2409.05061)
- **What's New**: 이 논문에서는 다양한 크기의 구획을 가진 소포 락커에 대한 수요 관리를 최초로 다루고, 이를 순차적 결정 문제로 형식화합니다. 즉, 소포 락커의 유한한 용량을 최대한 활용하면서 고객 만족을 유지하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Sequential Decision Analytics와 Reinforcement Learning 기술을 통합하여 소포 락커의 수요 관리 문제를 해결합니다. 핵심은 수요 제어와 할당을 동시에 고려하는 cost function approximation (CFA)와 parametric value function approximation (VFA)입니다.

- **Performance Highlights**: Computational study 결과, 제안된 방법은 myopic benchmark보다 13.7%, 산업 기반 정책보다 12.6% 성능이 우수함을 보였습니다.



### A $\Delta$-evaluation function for column permutation problems (https://arxiv.org/abs/2409.04926)
Comments:
          technical report

- **What's New**: 이번 연구에서는 sparse binary matrix에서 정의된 column permutation 문제를 해결하기 위한 새로운 Δ-evaluation 방법이 소개되었습니다. 이 방법은 그래프 이론 및 산업 제조와 같은 다양한 NP-hard 문제를 모델링합니다.

- **Technical Details**: Δ-evaluation 방법은 consecutive ones property를 따르는 sparse binary matrix에서 최대 열 합을 최소화하는 permutation matrix를 찾는 Column Permutation Problem (CPP)에 적용됩니다. 이 방법은 기존의 지역 탐색(Local Search) 절차에서 사용되는 두 가지 방법과 비교되어 시간 효율성을 입증합니다.

- **Performance Highlights**: Δ-evaluation 방법은 Gate Matrix Layout 및 Minimization of Open Stacks와 같은 문제에 적용될 때, 특히 크고 밀집된 인스턴스에서 처리 속도가 빠르며, 솔루션을 개선하는 데 매우 유용하게 사용될 수 있습니다.



### Defeasible Reasoning on Concepts (https://arxiv.org/abs/2409.04887)
- **What's New**: 이 논문에서는 KLM 프레임워크에서 개념에 대한 반박 가능한 추론(defeasible reasoning)의 기초를 다룹니다. 이 연구는 누적 추론 시스템 C와 루프가 있는 누적 추론 시스템 CL의 개념 설정으로의 일반화를 정의하고, 누적 모델, 누적 정렬 모델 및 선호 모델을 확장하여 해당 모델들의 일관성과 완전성을 보여줍니다.

- **Technical Details**: 반박 가능한 추론을 형식화하기 위해 KLM 프레임워크를 활용하였습니다. 논문은 개념 누적 추론 시스템 CC와 개념 누적 정렬 시스템 CCL을 정의하고, 각각의 모델이 어떻게 정당성과 완전성을 가지는지에 대해 논의합니다. 또한, 선호 모델의 개념적 대응체를 정의하여, 기존의 KLM 설정과의 차이점을 설명합니다.

- **Performance Highlights**: 이 연구에서는 실용적인 애플리케이션에서 개념 간의 비모노토닉(non-monotonic) 관계를 수립할 필요성을 강조하며, `mammals`와 같은 개념에 대한 구체적인 예시를 들고, 증명된 모델들이 개념적 설정에서 어떻게 잘 작동하는지를 보여줍니다.



### HULLMI: Human vs LLM identification with explainability (https://arxiv.org/abs/2409.04808)
Comments:
          17 pages, 8 figures

- **What's New**: 이 연구는 전통적인 ML 모델(예: Naive-Bayes, MLP, Random Forests, XGBoost)이 현대의 NLP 탐지기(예: T5-Sentinel, RoBERTa-Sentinel)와 동일한 성능을 보여준다는 포괄적인 분석을 제공합니다. 우리는 다양한 데이터셋에서 강력한 테스트 절차를 구현했습니다.

- **Technical Details**: LIME(Locally Interpretable Model-agnostic Explanations) 기술을 사용하여 각 모델의 예측에 가장 크게 기여하는 입력 부분을 밝혀내고 탐지 프로세스에 대한 통찰력을 제공합니다. 전통적인 ML 방법과 현대 NLP 모델을 결합하여 이러한 탐지 도구의 해석 가능성을 높이는 방법에 대해 논의합니다.

- **Performance Highlights**: 이 연구는 데이터를 커스터마이즈하여 탐지 모델의 강인성과 일반화 능력을 평가하며, 여러 도메인(예: 교육, 헬스케어, 미디어)에서 AI 기반 텍스트의 신뢰성과 유효성을 강조합니다.



### Action is the primary key: a categorical framework for episode description and logical reasoning (https://arxiv.org/abs/2409.04793)
Comments:
          26 pages, 18 figures, 4 tables

- **What's New**: 이 연구는 episodes (에피소드) 묘사 및 인식을 위한 새로운 계산 프레임워크인 cognitive-logs를 제시합니다. 이 프레임워크는 relational (관계형) 및 graph (그래프) 데이터베이스로 구성되어 있으며, 인간과 유사한 사고를 하는 데이터베이스 기반 인공지능의 발전을 목표로 하고 있습니다.

- **Technical Details**: cognitive-logs는 자연 언어로 표현된 '행동'(verbs)과 행동을 수행하는 '참여자'(participants)로 구성된 에피소드를 기록합니다. 카테고리 이론(category theory)에 기반한 연산을 통해 에피소드 간 비교 및 연역적 추론(deductive inference)이 가능하며, 이야기의 추상화도 수행할 수 있습니다.

- **Performance Highlights**: cognitive-logs는 데이터베이스의 방대한 용량을 활용하여 신경망 기반 인공지능보다 더 많은 지식을 저장할 수 있는 잠재력을 지니고 있습니다. 이 모델은 사람의 인지를 모델링하며, 다양한 인간 정신 활동을 시뮬레이션할 수 있는 가능성을 가지고 있습니다.



### Algorithmic Scenario Generation as Quality Diversity Optimization (https://arxiv.org/abs/2409.04711)
- **What's New**: 이 논문은 복잡한 로봇 시스템을 배포하기 전에 체계적으로 테스트할 필요성을 강조하며, 이를 위한 일반적인 프레임워크를 제시합니다.

- **Technical Details**: 이 연구는 Quality Diversity Optimization (QD 최적화) 문제를 알고리즘 시나리오 생성 문제로 설정하고, 이를 해결하기 위한 일반적인 프레임워크를 제안합니다. QD 알고리즘은 단일 최적 솔루션을 찾는 것이 아니라, 다양한 특성을 가진 여러 솔루션을 생성하는 것을 목표로 합니다. 시나리오 매개변수는 로봇과 인간의 상호작용을 시뮬레이션에서 검토하기 위해 구성됩니다.

- **Performance Highlights**: 이 통합 프레임워크는 인간-로봇 상호작용에서 다양한 현실적이고 도전적인 엣지 케이스 실패를 발견하는 데 기여하며, 미래 연구를 위한 열린 도전 과제를 제시합니다.



### MuAP: Multi-step Adaptive Prompt Learning for Vision-Language Model with Missing Modality (https://arxiv.org/abs/2409.04693)
- **What's New**: 본 논문은 Vision-Language (VL) 작업에서 불완전한 모달리티(모드) 환경에서의 프롬프트 학습(learn) 행동에 대한 최초의 포괄적 조사 결과를 제시합니다. 이를 통해 프롬프트 기반 모델이 결측된 모달리티에 대해 높은 민감도를 보이는 것을 보여줍니다.

- **Technical Details**: 우리는 Multi-step Adaptive Prompt Learning (MuAP) 프레임워크를 제안하여 다중 모달 프롬프트를 생성하고 다단계 프롬프트 튜닝을 수행하며, 각 모달리티를 반복적으로 정렬하여 지식을 적응적으로 학습합니다. 이 방법은 Transformer 모델에 통합될 수 있도록 프롬프트 전략을 설정합니다.

- **Performance Highlights**: 실험 결과, MuAP를 통해 본 모델은 모든 벤치마크 데이터 세트에서 최첨단 상태보다 현저한 성능 향상을 달성했습니다.



### Neurosymbolic Methods for Dynamic Knowledge Graphs (https://arxiv.org/abs/2409.04572)
- **What's New**: 이 논문에서는 Knowledge Graphs (KGs)의 동적(dynamics) 및 시간적(temporal) 차원을 다루며, Dynamic Knowledge Graphs (DKGs)에 대해 형식적으로 정의하고, DKG와 Temporal Knowledge Graphs (TKGs)의 차별성을 제시합니다.

- **Technical Details**: DKGs는 시간이 지남에 따라 변화하는 정보를 수용할 수 있도록 설계되어 있으며, 이러한 시스템에서의 엔티티 정합(entity alignment)은 시간이 지남에 따라 변화하는 속성을 고려해야 합니다. DKG의 동적 정보는 기초 정의, 사실 집합 및 로직 트리플의 형태로 표현됩니다.

- **Performance Highlights**: 이 논문에서는 DKG에 대한 neurosymbolic 방법들을 다루며, KG Completion 및 동적 엔티티 정합 작업에 제시된 방법들이 미래 예측 및 이상 탐지와 같은 다양한 응용 프로그램에서 어떻게 사용될 수 있는지를 설명합니다.



### Intensional FOL: Many-Sorted Extension (https://arxiv.org/abs/2409.04469)
Comments:
          21 pages

- **What's New**: 이 논문에서는 비정렬 IFOL (Intensional First-Order Logic)로부터 다중 정렬 IFOL로의 확장을 제안합니다. 이는 자연어가 암묵적으로 다중 정렬로 구성되어 있기 때문에, 자연어를 처리하는 애플리케이션에 적용하기 위한 것입니다.

- **Technical Details**: 비정렬 intension과 extension 개념을 다루며, intension은 개념의 속성을, extension은 개념이 적용되는 주체의 집합으로 정의됩니다. 고유의 논리 문장이 열리고 닫히는 구조 영역에서 개념을 사용하며, 다양한 유형의 관계, 속성 및 명제 등 intension을 나타내는 방법을 정의합니다.

- **Performance Highlights**: 이 연구는 자연어 처리에 있어 다중 정렬의 필요성을 강조하며, 기존 논리의 개념을 새로운 방식으로 확장함으로써, 더욱 효과적인 자연어 지원 시스템을 위한 기반을 마련합니다.



### Here's Charlie! Realising the Semantic Web vision of Agents in the age of LLMs (https://arxiv.org/abs/2409.04465)
Comments:
          The 23rd International Semantic Web Conference, November 11--15, 2024, Hanover, MD - Posters and Demos track

- **What's New**: 이 논문은 개인 및 조직과 같은 법적 주체가 온라인 상호작용을 수행할 수 있도록 하기 위해 반자율 (semi-autonomous) AI 에이전트를 개발하는 연구를 다룹니다. 이 연구는 사용자가 충분한 맥락이나 신뢰를 갖지 못할 경우에만 사용자와 상담하는 하이브리드 웹 에이전트를 통해 사용자와 에이전트 간의 대화를 생성합니다.

- **Technical Details**: 논문에서는 반자율 에이전트가 웹에서 법적 주체를 식별하고, 데이터 사용 통제를 설명하며, 교환하는 데이터의 출처를 명확하게 설명할 수 있는 비기능적 요구사항에 대해 논의합니다. 또한, Notation3 규칙을 사용해 신뢰성 및 안전성을 확보하며, LLMs(대규모 언어 모델)을 통해 자연어 상호작용이 가능합니다.

- **Performance Highlights**: 이 연구는 제너릭 개인 비서의 샘플 사용 사례를 통해 사용자 요청에 대한 흐름을 구현하였으며, Jun의 에이전트가 Nigel과의 미팅을 예약하는 과정에서 자동화된 확인을 수행하는 구조로 설계되었습니다. 이를 통해 개인 에이전트를 통해 신뢰성과 사용성의 균형을 달성하는 목표를 이루고 있습니다.



### Neural MP: A Generalist Neural Motion Planner (https://arxiv.org/abs/2409.05864)
Comments:
          Website at this http URL. Main paper: 7 pages, 4 figures, 2 tables. Appendix: 9 pages, 5 figures, 6 tables

- **What's New**: 본 연구는 모션 플래닝의 문제를 다루며, 기존의 방법들이 문제마다 솔루션을 처음부터 생성해야 하는데 비해, 데이터 기반의 학습을 스케일으로 적용하여 복잡한 환경에서도 빠른 해결책을 제시하는 새로운 접근법을 제안합니다.

- **Technical Details**: 시뮬레이션을 통해 다양한 복잡한 장면을 구축하고, 최첨단 모션 플래너로부터 전문가 데이터를 수집하여 반응형 일반화 정책으로 증류합니다. 이 정책은 100만 개의 장면에서 유래되어, 이전에 본 적 없는 장애물과 장면 구성에 대해 일반화할 수 있습니다. 또한, 경량화된 최적화 기법을 활용하여 안전한 경로를 보장합니다.

- **Performance Highlights**: 64개의 모션 플래닝 작업에 대한 평가에서 샘플링 기반 방법 대비 23%, 최적화 기반 방법 대비 17%, 신경망 모션 플래닝 방법 대비 79% 향상된 성공률을 달성했습니다.



### Promptable Closed-loop Traffic Simulation (https://arxiv.org/abs/2409.05863)
Comments:
          Accepted to CoRL 2024. Website available at this https URL

- **What's New**: 이 논문에서는 ProSim이라는 새로운 다중모드 프롬프트 가능 폐쇄형 트래픽 시뮬레이션 프레임워크를 제안합니다. ProSim은 사용자가 각 에이전트의 행동 및 의도를 지시하기 위해 복잡한 수치적, 범주적 또는 텍스트 프롬프트를 제공할 수 있게 하며, 폐쇄형 방식으로 트래픽 시나리오를 전개합니다.

- **Technical Details**: ProSim은 장면 초기화와 다중모드 프롬프트를 입력으로 받아, 에이전트의 정책 집합을 생성하고, 폐쇄형으로 반응형 시나리오 롤아웃을 생성합니다. 이렇게 생성된 프롬프트는 ProSim-Instruct-520k 데이터셋에 포함되어 있으며, 이 데이터셋은 520K paired 프롬프트-시나리오의 세트를 포함하고 있습니다.

- **Performance Highlights**: ProSim은 다양한 사용자 프롬프트를 제공할 때 높은 조작 가능성을 보여주며, 프롬프트가 없는 경우에도 Waymo Sim Agents Challenge에서 경쟁력 있는 성과를 달성합니다.



### An Introduction to Quantum Reinforcement Learning (QRL) (https://arxiv.org/abs/2409.05846)
Comments:
          Accepted by The 15th International Conference on ICT Convergence - ICTC 2024

- **What's New**: 최근 양자 컴퓨팅(Quantum Computing, QC)과 머신러닝(Machine Learning, ML) 통합에 대한 관심이 증가하고 있습니다. 특히 강화 학습(Reinforcement Learning, RL)을 양자 컴퓨팅의 원리를 통해 개선하려는 양자 강화 학습(Quantum Reinforcement Learning, QRL) 분야가 신흥 분야로 부각되고 있습니다.

- **Technical Details**: 양자 컴퓨팅은 특정 문제에서 고전 컴퓨터에 비해 상당한 계산 이점을 제공할 수 있는 잠재력을 가지고 있습니다. 양자 머신러닝(Quantum Machine Learning, QML) 알고리즘은 하이브리드 양자-고전적인 방식으로 구성되어 있으며, 양자 컴퓨터는 양자 계산에서 이점을 얻는 부분을 처리하고, 고전 컴퓨터는 그들이 잘하는 부분을 처리합니다. 양자 신경망(Quantum Neural Networks)에서는 큐비트(Qubit)가 기본 정보 처리 단위로 사용되며, 양자 게이트(Quantum Gates)는 양자 상태를 변환하는 데 활용됩니다.

- **Performance Highlights**: 양자 강화 학습(QRL)은 강화 학습 에이전트의 성능을 향상시키기 위해 양자 컴퓨팅 원리를 적용하는 최신 연구 동향을 보이고 있습니다. VQA(Variational Quantum Algorithms) 및 VQC(Variational Quantum Circuits)를 통해 여러 머신러닝 작업에서 성과를 보여주는 QML의 성공 사례를 통해 QRL의 가능성을 엿볼 수 있습니다.



### The Future of Software Testing: AI-Powered Test Case Generation and Validation (https://arxiv.org/abs/2409.05808)
Comments:
          24 Pages

- **What's New**: 이 논문에서는 소프트웨어 개발 생명 주기(SDLC)에서 중요한 단계인 소프트웨어 테스트의 혁신적인 AI 기반 방법론을 제시합니다. AI는 테스트 케이스 생성 및 검증을 더욱 효율적으로 개선하고, 정확성과 확장성을 높이는 잠재력을 가지고 있습니다.

- **Technical Details**: AI 기반의 테스트 방법은 테스트 케이스를 자동으로 생성하고, 변경 사항에 동적으로 적응하며, 기계 학습(machine learning)을 활용하여 코드베이스에서 높은 위험 영역을 식별합니다. 이러한 접근 방식은 회귀 테스트(regression testing)의 효율성을 높이고 전체 테스트 커버리지를 확장합니다. 또한 AI 도구는 지속적인 테스트(continuous testing)와 자체 복구(self-healing) 테스트 케이스를 가능하게 합니다.

- **Performance Highlights**: AI의 활용을 통해 전통적인 방법에서 발생하는 긴 기간, 인적 오류, 불완전한 테스트 범위 및 높은 수작업 개입 비용 등 여러 가지 문제를 해결할 수 있으며, 더 빠르고 신뢰성 높은 소프트웨어 출시가 가능합니다.



### Benchmarking Chinese Knowledge Rectification in Large Language Models (https://arxiv.org/abs/2409.05806)
Comments:
          Ongoing work; code and dataset are available at this https URL

- **What's New**: 본 연구에서는 중국어에 특화된 대규모 언어 모델(LLMs)의 지식을 수정하기 위한 벤치마크인 새로운 데이터셋 CKnowEdit를 소개합니다. 이 데이터셋은 고전 문헌, 속담, 이디엄 등 7종의 지식을 포함하고 있으며, 중국어의 독특한 언어적 특성을 반영하여 수집되었습니다.

- **Technical Details**: CKnowEdit는 1,760개의 사례를 포함하며, 고전 시가, 속담, 이디엄, 음성 표기, 고전 중국어, 지리 지식, 그리고 Ruoziba라는 7개의 중국어-specific 종류의 지식을 조직하고 수집합니다. 현재의 지식 수정 방법을 평가하기 위해, 단어 수준의 중첩 및 의미 벡터 유사성을 사용하여 수정된 모델의 성능을 평가합니다.

- **Performance Highlights**: 상태-of-the-art 지식 수정 기법들의 실증 결과를 통해, 중국 문헌에 적용했을 때의 한계를 드러냈으며, 이는 미래의 더 정교한 중국어 지식 수정 접근 방식의 필요성을 강조합니다.



### Enhancing Preference-based Linear Bandits via Human Response Tim (https://arxiv.org/abs/2409.05798)
- **What's New**: 본 연구에서는 일반적인 이진 선택 기반의 인간 피드백 대신 반응 시간(response time)을 활용하여 선호(preference)의 강도를 추정하는 새로운 방법을 제안합니다. 이는 간편하면서도 중요한 추가 정보를 제공합니다.

- **Technical Details**: EZ-diffusion 모델을 통합하여 인간의 선택과 반응 시간을 동시에 모델링한 후, 이를 선호 기반의 선형 밴딧(preference-based linear bandits) 문제에 적용했습니다. 반응 시간을 선택과 함께 사용하는 새로운 유틸리티 추정기를 제안하며, 이를 선형 회귀 문제로 재구성합니다.

- **Performance Highlights**: 실험 결과, 반응 시간을 통합한 방식이 기존의 선택만을 기반으로 한 방법보다 더 효과적임을 보여줍니다. 특히, '쉬운' 질문의 경우, 선택 정보만으로는 한계가 있지만 반응 시간을 활용하면 선호 강도에 대한 유용한 정보를 제공하여 학습 효율성을 높입니다.



### Leveraging Object Priors for Point Tracking (https://arxiv.org/abs/2409.05786)
Comments:
          ECCV 2024 ILR Workshop

- **What's New**: 이 논문에서는 Point Tracking을 위한 혁신적인 Objectness 정규화 기법을 제안하여 포인트가 대상 물체의 경계 내에 머물도록 유도하고, 이로 인해 장기 추적 성능의 향상을 도모합니다.

- **Technical Details**: 제안된 방법은 훈련 시 Objectness Loss를 도입하여 각 포인트가 속한 객체의 속성을 인식하도록 하며, Testing 시 객체 마스크를 계산하지 않고도 Spatial Continuity를 직접 통합하는 방식입니다. 또한, Contextual Attention을 활용하여 포인트 추적 중 각 지역의 Feature Representation을 향상시킵니다.

- **Performance Highlights**: 우리의 방식은 PointOdyssey, TAP-Vid-DAVIS, CroHD의 세 가지 벤치마크에서 최신 기술(features)인 방법들을 초월하는 성능을 보여주며, Objectness 정규화 기법을 통해 추론 시간 동안 추가적인 계산 비용 없이 효율성을 유지합니다.



### NeurLZ: On Systematically Enhancing Lossy Compression Performance for Scientific Data based on Neural Learning with Error Contro (https://arxiv.org/abs/2409.05785)
- **What's New**: NeurLZ는 과학 데이터용 새로운 cross-field learning 기반의 error-controlled 압축 프레임워크로, 기존의 기술적 한계를 극복하고 압축 성능을 개선합니다.

- **Technical Details**: NeurLZ는 lightweight skipping DNN 모델을 이용하여 복원 품질을 향상시키며, cross-field learning을 통해 다양한 데이터 필드 간의 관계를 학습합니다. 이 방식은 계량적으로 중요한 정보를 손실하지 않고도 압축을 수행합니다.

- **Performance Highlights**: NeurLZ는 실제 시뮬레이션 데이터 세트에 대해 최대 90%의 비트 비율 감소를 달성했으며, 복원 품질을 현저히 개선하는 데 기여합니다.



### Creativity and Visual Communication from Machine to Musician: Sharing a Score through a Robotic Camera (https://arxiv.org/abs/2409.05773)
- **What's New**: 이 논문은 "Guided Harmony" 음악 게임 내에서 로봇 카메라를 구현하여 시각적 소통과 음악적 상호작용의 통합을 탐구합니다. 이는 인간 뮤지션과 로봇 시스템 간의 공동 창작 행동을 조사하는 것을 목표로 합니다.

- **Technical Details**: 로봇 시스템은 뮤지션의 비언어적 신호를 해석하고 반응하여 협력적이고 적응적인 음악적 경험을 창출합니다. PTZ(팬-틸트-줌) 카메라를 사용하여 시각적 신호를 해석하고, AI 즉흥 연주 시스템의 복잡성을 더할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 초기 사례 연구는 직관적인 시각 소통 채널의 중요성을 강조하며, 향후 연구 방향으로는 시각 신호 툴킷 개선을 위한 매개변수 설정 및 인간-기계 공동 창의성을 더 깊이 이해할 데이터 수집 방법론을 제안합니다.



### Evidence from fMRI Supports a Two-Phase Abstraction Process in Language Models (https://arxiv.org/abs/2409.05771)
Comments:
          Equal contribution from both authors. Submitted to NeurIPS NeuroAI workshop 2024

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 중간 은닉 상태(intermediate hidden states)가 자연어 자극에 대한 뇌 반응을 예측할 수 있다는 기존 연구 결과를 바탕으로, 이러한 예측 성능을 가능하게 하는 표현적 특성에 대해 탐구합니다.

- **Technical Details**: LLMs의 중간 은닉 레이어가 선형 전이 작업에 가장 최적이라는 현상을 조사하면서, 학습 과정에서 자연스럽게 발생하는 두 단계의 추상화 과정(abstraction process)을 제안합니다. 우리는 manifold learning 방법을 사용하여 이 과정을 분석하고, 학습이 계속됨에 따라 첫 번째 '구성(composition)' 단계가 적은 레이어로 압축된다는 것을 보여줍니다.

- **Performance Highlights**: 이 연구는 LLM의 레이어별 인코딩 성능(encoded performance)과 LLM의 표현의 고유 차원(intrinsic dimensionality) 간에 강한 상관관계(correspondence)가 있음을 입증합니다. 이 관계는 LLM의 본질적인 조합성(compositionality)에서 기인한다는 초기 증거를 제공합니다.



### ReL-SAR: Representation Learning for Skeleton Action Recognition with Convolutional Transformers and BYOL (https://arxiv.org/abs/2409.05749)
Comments:
          8 pages, 4 figures, 6 tables

- **What's New**: 본 연구에서는 비지도(self-supervised) 표현 학습을 통해 스켈레톤 행동 인식(skeleton action recognition) 기술을 연구합니다. 이를 위해 매우 가벼운 Convolutional Transformer 프레임워크인 ReL-SAR(Representation Learning for Skeleton Action Recognition)를 개발하였으며, 합성곱(convolution) 및 주의(attention) 레이어의 보완적 특성을 활용하여 스켈레톤 시퀀스의 공간적(spatial) 및 시간적(temporal) 힌트를 공동으로 모델링합니다.

- **Technical Details**: ReL-SAR 모델은 스켈레톤 관절의 Selection-Permutation 전략을 사용하여 더 많은 정보가 포함된 설명을 제공합니다. Bootstrap Your Own Latent(BYOL) 접근법을 통해 레이블이 없는 스켈레톤 시퀀스 데이터로부터 강력한 표현을 학습합니다. 이 모델은 YOLOv5x를 사용하여 사람을 감지하고 ViTPose를 통해 자세를 추정하여 스켈레톤 시퀀스를 생성합니다.

- **Performance Highlights**: 제안된 ReL-SAR 모델은 MCAD, IXMAS, JHMDB, NW-UCLA와 같은 제한된 크기 데이터셋에서 매우 경쟁력 있는 결과를 달성하였으며, 성능과 계산 효율성 모두에서 최첨단(state-of-the-art) 방법에 대해 우수성을 입증하였습니다.



### A Novel Idea Generation Tool using a Structured Conversational AI (CAI) System (https://arxiv.org/abs/2409.05747)
Comments:
          21 pages, 16 figures, AIEDAM Journal Article

- **What's New**: 본 논문은 초급 디자이너가 초기 아이디어 생성에서 흔히 발생하는 지연(latency) 문제와 아이디어 생성 병목 현상을 극복할 수 있도록 지원하는 새로운 Conversational AI 기반의 액티브 아이데이션 인터페이스를 제안합니다.

- **Technical Details**: 제안된 도구는 크게 언어 처리 분야의 대규모 언어 모델(Large Language Model, LLM)을 통합하여 다양한 디자인 문제에 대한 아이디어 진술을 생성하는 동적, 상호작용적 접근 방식을 제공합니다. 이를 통해 설계 단계에서 연속적인 대화 및 맥락 민감성(conversational context)을 반영한 대화를 통해 풍부하고 다양한 아이디어를 생성할 수 있도록 합니다.

- **Performance Highlights**: 파일럿 연구에서는 30명의 초급 디자이너가 전통적인 방법과 CAI 기반 인터페이스를 사용하여 아이디어를 생성하였으며, 전문 패널을 통해 정의된 유창성(fluecy), 참신함(novelty), 다양성(variety) 등의 핵심 매개변수를 질적으로 비교하였습니다. 연구 결과, 제안된 도구는 풍부하고 다양한 새로운 아이디어를 생성하는 데 효과적임이 입증되었습니다.



### A System and Benchmark for LLM-based Q\&A on Heterogeneous Data (https://arxiv.org/abs/2409.05735)
- **What's New**: 이번 논문에서는 사내 환경에서 다수의 이질적인 데이터 소스를 통합하여 자연어 질문에 대한 응답을 제공할 수 있는 siwarex 플랫폼을 소개합니다. 이 플랫폼은 데이터베이스와 API에 seamless하게 접근할 수 있는 기능을 제공합니다.

- **Technical Details**: siwarex는 다양한 데이터베이스와 API 호출을 포함하는 이질적인 데이터 소스를 처리하는 질문 응답 시스템입니다. 이 시스템은 LLM(Large Language Model)을 사용해 자연어 질문을 이해하고 SQL 문장을 생성하며, 사용자가 정의한 함수(User-Defined Function, UDF)를 활용하여 API 호출을 처리합니다. 또한, siwarex는 혼합된 데이터 액세스를 지원하는 최초의 벤치마크를 제공합니다.

- **Performance Highlights**: siwarex는 확장된 Spider 데이터셋을 기반으로 여러 산업 환경에서 성공적으로 배포되었으며, 데이터 소스의 이질성을 잘 처리하는 성능을 보였습니다. 논문에서 제시된 결과는 siwarex가 DB 호출과 API 호출 간의 비율에 따라 Q&A 정확성을 유지하며, 연구 커뮤니티에 수정된 Spider 벤치마크를 제공하여 업계의 LLM 기반 Q&A 시스템 개발을 장려하고자 한다는 점에서 중요합니다.



### What Did My Car Say? Autonomous Vehicle Explanation Errors, Context, and Personal Traits Impact Comfort, Reliance, Satisfaction, and Driving Confidenc (https://arxiv.org/abs/2409.05731)
Comments:
          23 pages, 4 figures

- **What's New**: 이 연구는 자율주행차(AV)의 설명 오류가 승객의 신뢰, 안전한 의존, 만족도 및 자신감에 어떤 영향을 미치는지를 조사했습니다. 오류는 모든 결과에 부정적인 영향을 미쳤으며, 맥락적 요소가 이러한 관계에 중요한 역할을 한다는 것을 발견했습니다.

- **Technical Details**: 연구에서는 232명이 참가한 시뮬레이션 주행 연구를 통해 AV 설명 오류, 주행 맥락의 특성(인지된 해악과 주행 난이도), 개인 특성(이전의 신뢰 및 전문성)이 승객의 경험에 미치는 영향을 평가했습니다. 실험은 세 가지 정확도 수준의 AV 설명을 제시하며, 설명 오류가 AV에 대한 신뢰, 통제 선호, AV의 주행 능력에 대한 자신감, 설명 만족도에 미치는 영향을 측정했습니다.

- **Performance Highlights**: 이 연구 결과는 AV 설명의 정확성, 맥락에 적합성 및 개인화의 필요성을 강조합니다. 특히, 해악의 인식 및 주행 난이도는 결과에 직접적인 영향을 미치고 설명 오류와 사용자 인식 간의 관계를 조절하는 역할을 했습니다. 높은 전문성을 가진 참가자는 AV에 대한 신뢰가 높았고 이는 긍정적인 결과와 관련이 있었습니다.



### Referring Expression Generation in Visually Grounded Dialogue with Discourse-aware Comprehension Guiding (https://arxiv.org/abs/2409.05721)
Comments:
          Accepted for publication at INLG 2024

- **What's New**: 이 논문에서는 시각적 기반 대화에서의 참조 표현 생성(REF)의 새로운 접근법을 제안합니다. 제안된 방법은 식별적이며 대화에 적합한 RE를 생성하기 위한 두 단계의 과정으로 구성됩니다.

- **Technical Details**: 첫 번째 단계에서는 REG를 텍스트 및 이미지 조건부의 다음 토큰 예측 작업으로 모델링합니다. 이후 RE는 이전 언어적 맥락과 참조 대상의 시각적 표현을 바탕으로 자가 회귀적으로 생성됩니다. 두 번째 단계에서는 대화 맥락을 보고 후보 RE의 선별을 위해 대화 인식(comprehension guiding)을 적용합니다.

- **Performance Highlights**: 인간 평가 결과, 제안된 두 단계 접근법이 더 높은 성능을 보여주었으며, 선택된 RE의 텍스트-이미지 검색 정확도가 더 높았습니다. 이는 기존의 탐욕적 디코딩 방법으로 생성된 RE에 비해 월등히 우수한 성과를 나타냅니다.



### pFedGPA: Diffusion-based Generative Parameter Aggregation for Personalized Federated Learning (https://arxiv.org/abs/2409.05701)
- **What's New**: 이번 논문은 Federated Learning (FL)에서의 모델 집합을 개선하기 위해 새로운 생성적 매개변수 집합 프레임워크인 pFedGPA를 제안합니다. 이 프레임워크는 고차원 매개변수 공간의 복잡성을 효과적으로 처리하며, 개인화된 매개변수를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: pFedGPA는 서버에서 diffusion model을 배포하여 다양한 매개변수 분포를 통합합니다. 또한, 매개변수 역전환 방법을 통해 각 클라이언트에 대한 개인화된 매개변수 집합을 효율적으로 생성합니다. 이 과정에서 업로드된 매개변수는 잠재 코드로 변환되고, 그 후 denoising sampling을 통해 최종 개인화된 매개변수가 생성됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법 pFedGPA는 여러 데이터셋에서 기존의 baseline 접근 방식을 초과하는 우수한 성능을 지속적으로 나타냈습니다. 이 방법은 클라이언트의 모델 매개변수가 특정 데이터 분포에 의존성을 효과적으로 인코딩하는 강력한 diffusion model을 사용하여 모델 집합의 복잡성을 감소시킬 수 있습니다.



### MANA-Net: Mitigating Aggregated Sentiment Homogenization with News Weighting for Enhanced Market Prediction (https://arxiv.org/abs/2409.05698)
Comments:
          Accepted by CIKM 24

- **What's New**: 이번 논문에서는 금융 뉴스 데이터를 통해 시장 예측을 개선할 수 있는 방법을 제시합니다. 특히, 기존의 단순한 집계 방식으로 인한 'Aggregated Sentiment Homogenization' 문제를 해결하기 위해, Market Attention-weighted News Aggregation Network (MANA-Net)이라는 새로운 모델을 도입합니다.

- **Technical Details**: MANA-Net은 시장 뉴스와의 동적 주의 메커니즘을 활용하여 뉴스 아이템의 중요도에 따라 가중치를 부여합니다. 이를 통해, 뉴스 감정을 정교하게 집계하고, 다량의 뉴스 데이터를 효과적으로 처리하여 예측에 가장 중요한 정보를 포착할 수 있도록 합니다. 특히, MANA-Net의 학습 과정에서 뉴스 집계와 시장 예측이 통합되어 모델의 구성 요소가 훈련 기간 동안 동적으로 조정됩니다.

- **Performance Highlights**: MANA-Net은 S&P 500 및 NASDAQ 100 지수를 활용한 실험에서 기존의 다양한 시장 예측 방법들보다 뛰어난 성능을 보였습니다. 수익성 측면에서 1.1% 향상된 Profit & Loss와 0.252 증가된 일일 Sharpe 비율을 달성했습니다.



### RegNLP in Action: Facilitating Compliance Through Automated Information Retrieval and Answer Generation (https://arxiv.org/abs/2409.05677)
- **What's New**: 본 논문은 RegNLP(Regulatory Natural Language Processing) 분야에 기여하며, 자동화된 질문-단락 생성(Automated Question-Passage Generation) 작업을 정의하고, 27,869개의 질문을 포함한 ObliQA 데이터셋을 만들고, 규제 정보 검색 및 답변 생성 시스템을 설계하여 성능을 평가합니다.

- **Technical Details**: RegNLP는 규제 규칙과 의무의 접근과 해석을 단순화하기 위한 다학제 분야입니다. 오히려 복잡한 규제 문체에서 정확하게 정보를 추출하기 위한 자동화된 질문-단락 생성 프레임워크를 도입하고, 자연어 추론(Natural Language Inference, NLI)을 검증 단계에 통합하였습니다. ObliQA 데이터셋은 금융 규제 문서에서 파생된 질문과 관련 단락들을 포함합니다. RePASs 평가 메트릭은 생성된 답변이 모든 관련 의무를 정확히 반영하는지 평가합니다.

- **Performance Highlights**: 이 연구는 RegNLP의 데이터셋 및 평가 메트릭의 한계를 넘어선 포괄적이고 간결한 답변 생성이 가능하다는 점에서 향후 규제 문서의 이해 및 접근성을 크게 향상시킬 수 있습니다. 특히, 잘 설계된 질문 생성 및 정보 검색 단계는 규정 준수 오류를 줄이고, 운영 효율성을 증대시킬 수 있음을 보여줍니다.



### Evaluation of real-time transcriptions using end-to-end ASR models (https://arxiv.org/abs/2409.05674)
Comments:
          15 pages, 4 figures

- **What's New**: 본 논문에서는 실시간 자동 음성 인식(ASR) 시스템을 위한 오디오 분할 알고리즘의 성능을 평가합니다. 오디오를 짧은 조각으로 나누어 ASR 시스템에서 실시간으로 처리할 수 있게 하여 지연(latency)을 최소화하는 방법을 제안합니다.

- **Technical Details**: 세 가지 오디오 분할 알고리즘인 고정 간격 조각화(fragmentation at fixed intervals), 음성 활동 감지(Voice Activity Detection, VAD), 피드백 조각화(fragmentation with feedback)를 다양한 ASR 모델과 결합하여 평가합니다. 이 연구는 GigaSpeech 데이터셋을 사용하여 모델 성능과 지연을 측정합니다.

- **Performance Highlights**: 실험 결과, VAD 분할 방식이 가장 높은 품질을 제공하였으나 지연이 가장 길었습니다. 고정 간격 조각화는 가장 낮은 품질과 짧은 지연을 보였고, 새로운 피드백 알고리즘은 VAD 분할에 비해 2-4%의 오류율 증가로 1.5-2초의 지연 감소를 이루었습니다.



### Zero-shot Outlier Detection via Prior-data Fitted Networks: Model Selection Bygone! (https://arxiv.org/abs/2409.05672)
Comments:
          preprint

- **What's New**: 본 논문은 FoMo-0D라는 새로운 아웃라이어 탐지(Outlier Detection, OD) 방법을 제안합니다. 이 방법은 모델 선택의 필요성을 완전히 제거하며, 기존의 알고리즘 및 하이퍼파라미터 최적화를 우회하는 혁신적인 접근 방식을 보여줍니다.

- **Technical Details**: FoMo-0D는 Priord-data Fitted Networks를 기반으로 하며, 이는 대량의 합성 데이터에서 생성된 데이터를 바탕으로 Transformer 모델을 훈련합니다. 이를 통해 FoMo-0D는 새로운 OD 데이터셋에 대해 단일 포워드 패스를 통해 (아웃라이어/인라이어) 레이블을 예측할 수 있는 사전 훈련된 Foundation Model입니다.

- **Performance Highlights**: 57개의 공공 벤치마크 데이터셋에서 26개의 기본 방법과 비교한 결과, FoMo-0D는 상위 2위 기본 방법과 통계적으로 유사한 성과를 나타내며, 대부분의 다른 방법들을 상당히 초과 달성하였습니다. 평균 추론 시간은 샘플 당 7.7ms로 매우 빠릅니다.



### Real-Time Human Action Recognition on Embedded Platforms (https://arxiv.org/abs/2409.05662)
- **What's New**: 이 논문은 컴퓨터 비전과 딥러닝 기술의 발전으로 비디오 기반 인간 행동 인식(HAR)이 실용화된 가운데, 실시간 퍼포먼스 문제를 해결하기 위한 4가지 기여를 제안합니다.

- **Technical Details**: 하나, 기존의 Optical Flow(OF) 추출 기술이 최신 HAR 파이프라인의 지연 병목임을 확인하는 실험적 연구, 둘, 전통적인 방식과 딥러닝 기반 OF 추출 방법 간의 지연-정확도 트레이드오프를 탐구하는 것, 셋, 효율성과 정확성을 충족하는 Integrated Motion Feature Extractor(IMFE) 설계, 넷, 임베디드 플랫폼에 최적화된 실시간 HAR 시스템 RT-HARE의 개발입니다.

- **Performance Highlights**: Nvidia Jetson Xavier NX 플랫폼에서 RT-HARE는 30 FPS의 비디오 프레임 속도로 실시간 HAR을 처리하고 높은 인식 정확도를 유지하는 성능이 입증되었습니다.



### Interactive incremental learning of generalizable skills with local trajectory modulation (https://arxiv.org/abs/2409.05655)
Comments:
          21 pages, 16 figures

- **What's New**: 본 연구는 로컬(local) 및 글로벌(global) 경로(modulation)를 동시에 활용하는 인터랙티브 모방 학습(interactive imitation learning) 프레임워크를 제안합니다. 이는 보다 나은 일반화(generalization)를 위한 새로운 접근법을 제공하며, 인간의 직접적인 피드백을 통해 개발된 기술(modulation) 범위를 확대하고 있습니다.

- **Technical Details**: 이 프레임워크는 커널화된 움직임 원리(kernelized movement primitives, KMP)에 기반하고 있으며, 사용자의 물리적 피드백으로부터 로컬 모델을 조절하여 학습한 기술의 일반화를 개선합니다. 이를 통해 새로운 작업 파라미터를 동적으로 정의할 수 있고, 새로운 환경으로의 기술 확장이 가능해집니다.

- **Performance Highlights**: DLR SARA 로봇을 사용하여 베어링 링 로딩(task) 실험에서 평가한 결과, 사용자가 초기 모델을 점진적으로 수정하고 새로운 행동을 추가할 수 있는 가능성을 보여주었습니다. 이는 인터랙티브한 오류 수정과 기술 확장을 통한 성능 향상을 강조합니다.



### Replay Consolidation with Label Propagation for Continual Object Detection (https://arxiv.org/abs/2409.05650)
- **What's New**: 이 논문에서는 객체 탐지(Object Detection) 분야에서의 지속 학습(Continual Learning, CL) 문제를 다루며, 레플레이 방법론의 한계를 개선하기 위해 새로운 기술인 리플레이 통합과 라벨 전파(Replay Consolidation with Label Propagation, RCLPOD)를 제안합니다.

- **Technical Details**: RCLPOD는 레플레이 메모리의 샘플을 개선하여, 클래식 배포 최적화(Optimizing Class Distribution in Memory, OCDM) 및 라벨 전파(Label Propagation) 기법을 통합하여 작업 간 간섭 문제(task interference issue)를 줄이는 방식을 따릅니다. 또한, 마스킹 손실(Masking Loss) 기술과 피처 증류(Feature Distillation) 기법을 사용하여 모델의 효과성을 극대화합니다.

- **Performance Highlights**: 이 방법은 VOC 및 COCO와 같은 기존 CLOD 벤치마크에서 다른 방법론에 비해 우수한 성능을 보여주며, YOLOv8 모델에서 테스트되었습니다. 이는 현재의 YOLO 아키텍처와 CLOD 프레임워크에서의 연구를 위한 중요한 기준점을 제공합니다.



### 3D-SAR Tomography and Machine Learning for High-Resolution Tree Height Estimation (https://arxiv.org/abs/2409.05636)
- **What's New**: 이 연구는 Synthetic Aperture Radar (SAR) 기술을 활용하여 숲의 생물량을 정확히 추정하는 방법을 제공합니다. 특히, Tomographic SAR (TomoSAR) 데이터를 사용하여 숲의 높이를 측정하는 기계를 학습하는 새로운 접근법을 제시합니다.

- **Technical Details**: 연구에서는 SAR 이미지와 LiDAR 데이터를 결합하여 숲의 구조를 3D 모델링합니다. TomoSense 데이터세트를 활용하여 L 및 P 밴드의 SAR 데이터를 사용하고, 다양한 기계 학습 기법, 특히 deep learning을 위한 3D U-Net 아키텍처를 사용합니다. 또한, 공간의 자율 상관관계를 분석하기 위해 다양한 지리적 분할 방법을 평가합니다.

- **Performance Highlights**: 연구의 최종 결과, 모델은 30m 높이의 수관에서 평균 절대 오차(Mean Absolute Error)가 2.82m로 측정되어, 지구의 탄소 저장량을 정확히 측정하고 기후 행동을 지원하는 능력이 향상되었습니다.



### Joint Input and Output Coordination for Class-Incremental Learning (https://arxiv.org/abs/2409.05620)
Comments:
          11 pages, 4 figues. Accepted by IJCAI 2024

- **What's New**: 이번 연구는 점진적 학습(incremental learning)에서의 클래스 불균형(class imbalance) 문제와 기존 및 새로운 작업 간의 상호 간섭을 해소하기 위한 새로운 접근법을 제안합니다. 이를 위해 JIOC(joint input and output coordination) 메커니즘을 도입하여 데이터의 각 카테고리에 서로 다른 가중치를 부여하고, Knowledge Distillation(KD)을 활용해 출력 간섭을 줄입니다.

- **Technical Details**: 제안한 JIOC 메커니즘은 입력 데이터의 가중치를 각 클래스의 출력 점수의 그래디언트(gradient)에 따라 동적으로 조정합니다. 또한, 새로운 작업의 분류 헤드(classification head)에서의 이전 작업 데이터의 출력을 명확히 억제하며, KD를 통해 출력 점수를 조정하는 전략을 수행합니다. 이는 다양한 점진적 학습 접근법에 적용 가능하여 유연성을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 메커니즘은 여러 유명한 데이터 세트(CIFAR10-LT, CIFAR100-LT 등)에서 기존 접근법보다 성능이 10% 이상 향상되는 결과를 보여주었습니다.



### Adapted-MoE: Mixture of Experts with Test-Time Adaption for Anomaly Detection (https://arxiv.org/abs/2409.05611)
- **What's New**: 본 논문에서는 Adapted-MoE라는 새로운 방법을 제안하여, 같은 범주 내에서의 다양한 샘플 분포를 다루는 문제를 해결합니다. 이 방법은 여러 전문가 모델과 라우팅 네트워크를 사용하여 다양한 정상 샘플의 피처(Feature) 분포에 대응하여 독립적인 결정 경계를 생성합니다.

- **Technical Details**: Adapted-MoE는 라우팅 네트워크를 통해 같은 범주 샘플을 서브클래스 피처 공간으로 라우팅하고, 여러 전문가 모델이 다양한 정상 샘플의 표현을 학습하여 여러 독립적인 결정 경계를 형성합니다. 또한 테스트 시간에 적응(Test-Time Adaption)을 통해 미지의 정상 샘플 표현과 전문가 모델이 학습한 피처 분포 간의 편향을 제거합니다.

- **Performance Highlights**: 실험 결과, Adapted-MoE는 Texture AD benchmark에서 I-AUROC와 P-AUROC에서 각각 2.18%-7.20% 및 1.57%-16.30%의 성능 향상을 보여주었으며, 기존 최신 방법들보다 우수한 성능을 기록했습니다.



### SynMorph: Generating Synthetic Face Morphing Dataset with Mated Samples (https://arxiv.org/abs/2409.05595)
- **What's New**: 본 연구에서는 2450개의 아이디와 10만 개 이상의 변형된 이미지를 포함한 합성 얼굴 변형 데이터셋을 생성하는 새로운 방법을 제안합니다. 이 데이터셋은 고품질 샘플, 다양한 변형 알고리즘 및 단일 및 차별 변형 공격 탐지 알고리즘에 대한 일반화를 특징으로 합니다.

- **Technical Details**: 연구팀은 StyleGAN 2 모델을 사용하여 1024 × 1024 해상도의 고품질 합성 얼굴 이미지를 생성합니다. 이를 통해 감정 표현과 조명 조건을 조정하여 다양한 변형 이미지를 생성하며, GAN 기반과 랜드마크 기반의 변형 알고리즘을 결합하여 샘플을 생성합니다. 합성 데이터는 S-MAD와 D-MAD 두 경우를 모두 지원합니다.

- **Performance Highlights**: 검증 작업을 통해 제안된 합성 데이터셋은 기존 SOTA 합성 데이터셋과 비교하여 향상된 성능을 보여주며, 비오메트릭 샘플 품질 및 얼굴 인식 시스템에 대한 변형 공격 가능성을 평가하였습니다. 연구 결과는 MAD 알고리즘의 훈련 및 평가에서 새로운 가능성을 열어줍니다.



### ExDDI: Explaining Drug-Drug Interaction Predictions with Natural Languag (https://arxiv.org/abs/2409.05592)
Comments:
          17 pages, 4 figures

- **What's New**: 이 연구는 약물-약물 상호작용(DDI) 예측에서 보다 신뢰성을 높일 수 있는 자연어 기반의 설명 생성 기법을 제안합니다.

- **Technical Details**: 연구에서는 DDInter와 DrugBank에서 수집한 약물 상호작용 설명을 활용하여, 예측을 수행하면서 약물의 약리 동력학(pharmacodynamics) 및 약물 약리학(pharmacokinetics) 메커니즘을 동시에 드러낼 수 있는 여러 모델을 개발했습니다.

- **Performance Highlights**: 제안된 모델은 알려진 약물 간의 알려지지 않은 DDI에 대한 정확한 설명을 제공할 수 있으며, 이는 DDI 예측 분야에 새로운 도구를 제공하여 추가 연구를 위한 기초를 마련합니다.



### MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery (https://arxiv.org/abs/2409.05591)
Comments:
          Codes and models are in this https URL

- **What's New**: 이번 연구에서는 MemoRAG라고 불리는 새로운 retrieval-augmented generation 파라다임을 제안합니다. MemoRAG는 장기 기억(long-term memory)을 통해 외부 데이터베이스에 접근하여 정보를 더 잘 활용하고, 불명확한 정보 요구를 처리할 수 있는 기능을 강화하였습니다.

- **Technical Details**: MemoRAG는 이중 시스템 아키텍처를 채택하여, 가벼운 LLM(light LLM)과 비싼 LLM(expensive LLM)을 결합합니다. 가벼운 LLM은 전체 데이터베이스의 글로벌 메모리를 형성하는 역할을 하며, 주어진 작업에 대해 초안을 생성하여 유용한 정보를 검색 후 제공하는 기능을 합니다. 반면, 비싼 LLM은 검색된 정보를 바탕으로 최종 답변을 생성합니다. 성능 향상을 위해 MemoRAG는 클루 생성 및 기억 용량을 최적화합니다.

- **Performance Highlights**: MemoRAG는 다양한 평가 작업에서 뛰어난 성능을 보여줍니다. 특히 기존 RAG 시스템이 실패하는 복잡한 작업에서도 주문형 고품질 답변을 생성할 수 있으며, 전통적인 질문-응답 작업에서도 유의미한 이점을 제공합니다. 실험에 사용된 UltraDomain 벤치마크는 법률, 금융, 교육 등 다양한 도메인에서 복잡한 RAG 작업들을 포함하고 있으며, MemoRAG는 그러한 작업에서 높은 성과를 달성하였습니다.



### Interpretable Responsibility Sharing as a Heuristic for Task and Motion Planning (https://arxiv.org/abs/2409.05586)
- **What's New**: 본 논문에서는 Interpretable Responsibility Sharing (IRS)라는 새로운 휴리스틱(heuristic)을 통합한 Task and Motion Planning (TAMP) 방법론을 소개합니다. 이 접근법은 인간이 만든 환경과 기존의 편향을 이용하여, 가정 로봇의 계획 효율성을 높이고자 합니다.

- **Technical Details**: IRS는 Responsibility Sharing (RS)라는 개념을 기반으로 하여, 보조 개체(auxiliary objects, 예: 쟁반, 주전자)를 통해 복잡한 작업을 관리 가능한 하위 문제로 나누어 작업을 최적화합니다. Optimized Rule Synthesis (ORS)를 통해 의사결정을 최적화하며, 로봇이 이들 보조 개체를 전략적이고 상황 인지적으로 활용할 수 있도록 합니다.

- **Performance Highlights**: 여러 가정 작업에서 수행된 실험은 IRS가 전통적인 방법들보다 현저히 우수한 성능을 보였음을 보여줍니다. IRS는 작업 실행에 필요한 노력을 줄이고 전반적인 의사결정 과정을 개선하였습니다. 이 접근법은 인간의 직관적인 방식에 부합하며, 다양한 가정 환경에 적응 가능성이 있는 확장 가능한 솔루션을 제공합니다.



### Latent 3D Brain MRI Counterfactua (https://arxiv.org/abs/2409.05585)
- **What's New**: 이번 논문에서는 고차원 데이터에서 인과성을 정확하게 모델링하는 데 어려움이 있는 기존의 대체 모델들 대신, 구조적 인과 모델 (Structural Causal Model, SCM)을 사용하여 고해상도 MRI 이미지를 생성하는 새로운 2단계 방법을 제안합니다.

- **Technical Details**: 우리의 접근법은 첫째 단계에서 VQ-VAE (Vector Quantized Variational Autoencoder)를 사용하여 MRI 볼륨의 압축된 임베딩을 학습합니다. 이후 인과 모델을 latent space에 통합하고, 일반화 선형 모델 (Generalized Linear Model, GLM)을 사용하여 세 단계의 반사실적 절차 (counterfactual procedure)를 수행합니다.

- **Performance Highlights**: 실제 고해상도 MRI 데이터(1mm)에 대한 실험 결과, 제안된 방법이 높은 품질의 3D MRI 반사실적 이미지를 생성할 수 있음을 입증했습니다.



### Learning to Model Graph Structural Information on MLPs via Graph Structure Self-Contrasting (https://arxiv.org/abs/2409.05573)
- **What's New**: 본 논문에서는 Graph Neural Networks (GNNs)에서 전통적으로 사용되는 메시지 패싱(message passing) 방식 대신에, 그래프 구조 정보(Structural Information)를 효율적으로 학습할 수 있는 새로운 프레임워크인 Graph Structure Self-Contrasting (GSSC)를 제안합니다.

- **Technical Details**: GSSC 프레임워크는 두 개의 주요 네트워크로 구성되어 있습니다: (i) Structural Sparsification (STR-Sparse) 및 (ii) Structural Self-Contrasting (STR-Contrast). STR-Sparse 네트워크는 이웃에서 노이즈가 있는 엣지를 제거하고, STR-Contrast는 그러한 희소화된 이웃에서 강건한 노드 표현을 학습합니다. 이 구조는 Multi-Layer Perceptrons (MLPs)를 기반으로 하며, GNN과는 다르게 명시적 메시지 전파 없이 구조적 정보를 암묵적으로 활용합니다.

- **Performance Highlights**: GSSC는 다른 최신 경쟁 모델들과 비교하여 우수한 일반화(generalization)와 강건성(robustness)을 보여줍니다. 많은 실험 결과는 GSSC가 예기치 못한 성능 향상을 가져올 수 있음을 입증하였습니다.



### On the Convergence of Sigmoid and tanh Fuzzy General Grey Cognitive Maps (https://arxiv.org/abs/2409.05565)
- **What's New**: 본 논문은 Fuzzy General Grey Cognitive Map (FGGCM)의 수렴성에 대한 심층적인 탐구를 진행하며, 일반 회색 수(Grey Number) 처리의 새로운 가능성을 제시합니다. 이전 연구들에서 FCM과 FGCM의 수렴성은 다루어졌지만, FGGCM의 수렴성은 충분히 다루어지지 않았습니다. 이를 통해 FGGCM의 정확한 고정점을 설계하는 데 필요한 학습 알고리즘의 이론적 기초를 제공합니다.

- **Technical Details**: FGGCM은 여러 구간을 가진 일반 회색 수를 처리할 수 있도록 Fuzzy Cognitive Map (FCM)의 확장에서 발전된 모델입니다. Minkowski 부등식을 이용해 일반 회색 수 공간과 벡터 공간의 지표를 제시 및 증명하였으며, Cauchy 수열의 수렴 특성을 활용해 두 공간의 완전성을 입증했습니다. 그 후 Banach 고정점 정리와 Browder-Gohde-Kirk 고정점 정리를 활용하여, tanh와 sigmoid 함수가 활성화 함수로 사용될 때 FGGCM이 고유 고정점에 수렴하기 위한 충분 조건을 도출하였습니다.

- **Performance Highlights**: FGGCM 모델은 웹 경험과 토목 공학 분야의 FCM을 기반으로 하여 설계되었으며, sigmoid 및 tanh 활성화 함수를 사용하는 고정점 수렴 조건을 제공함으로써 기존의 수렴 정리를 개선합니다. 실험 결과, FCM 및 FGCM의 수렴 정리와 비교함으로써 본 논문의 정리들이 효과적임을 검증하였으며, FCM의 수렴 정리가 본 논문에서 제시하는 정리의 특별한 경우임을 보여줍니다.



### LEROjD: Lidar Extended Radar-Only Object Detection (https://arxiv.org/abs/2409.05564)
Comments:
          Accepted for publication as ECCV 2024

- **What's New**: 이 논문에서는 3D 객체 감지(3D object detection)의 성능을 향상시키기 위한 두 가지 전략을 제시합니다. 기존의 LiDAR 데이터의 이점을 활용해 3+1D 이미징 레이더(sensor) 객체 감지 모델을 개선하는 방법을 탐구합니다.

- **Technical Details**: 제안된 전략은 1) LiDAR 포인트 클라우드(thin-out) 기법을 활용한 다단계 훈련(multi-stage training) 및 2) 크로스 모달 지식 증류(cross-modal knowledge distillation)입니다. 다단계 훈련 과정에서는 세 가지 thin-out 방법을 비교 분석하였습니다.

- **Performance Highlights**: 다단계 훈련을 통해 평균 정밀도(mean Average Precision)가 최대 4.2% 향상되었으며, 지식 증류를 통해 최대 3.9%의 성능 향상을 기록했습니다. 이 접근 방식은 다른 3D 객체 감지 네트워크에도 적용 가능함을 입증했으며, 코드가 해당 URL에서 제공됩니다.



### Seeing Through the Mask: Rethinking Adversarial Examples for CAPTCHAs (https://arxiv.org/abs/2409.05558)
Comments:
          Under review

- **What's New**: 이 논문은 머신 비전 모델이 CAPTCHAs에서 이미지를 혼란스럽게 만드는 새로운 방법을 제안합니다. 기존의 고유한 노이즈 추가 방식 대신, 다양한 강도의 마스크를 추가하여 인간에게는 인식 가능하지만 머신에는 인식되지 않도록 하는 방식으로써, 최신 이미지 분류기를 속일 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 CAPTCHAs의 새로운 형태로 hCaptcha의 마스크 신호를 사용하여 이미지에 대한 공격을 평가합니다. 연구는 CAPTCHAs와 관련된 공격을 위해 인간의 시각 인식을 활용하고, 이미지의 의미 정보를 보존하는 방법을 탐구합니다. 또한, ConvNeXt, EVA, ViT와 같은 최신 비전 모델을 사용하여 다양한 이미지 필터 적용 후 정확도가 얼마나 감소하는지를 측정합니다.

- **Performance Highlights**: 모든 모델에서 Accuracy @ 1 (Acc@1)이 50% 이상 감소하며, 특히 비전 변환기(vision transformers)에서는 80% 감소하는 결과를 보였습니다. 이러한 발견은 최신 비전 모델들이 여전히 인간의 인식 능력을 따라잡지 못하고 있음을 강조합니다.



### HMAFlow: Learning More Accurate Optical Flow via Hierarchical Motion Field Alignmen (https://arxiv.org/abs/2409.05531)
Comments:
          11 pages, 6 figures

- **What's New**: 본 연구에서 제안된 HMAFlow는 Hierarchical Motion Field Alignment (HMA) 모듈과 Correlation Self-Attention (CSA) 모듈을 통해 작은 물체가 포함된 복잡한 장면에서의 optical flow 추정 성능을 향상시키는 새로운 방법입니다. 다중 스케일 상관 관계 검색(Multi-Scale Correlation Search, MCS) 레이어를 통해 4D 비용 볼륨을 재구성함으로써 기존 방법보다 효과적으로 경량 물체의 움직임을 캡처합니다.

- **Technical Details**: HMAFlow 모델은 두 개의 핵심 모듈로 구성됩니다: HMA 모듈은 다중 스케일 모션 특징을 통합하고 CSA 모듈은 글로벌 모션 특징의 신뢰성과 강건성을 향상시킵니다. 비용 볼륨 계산에서 평균 풀링 대신 MCS 레이어를 사용하여 동적 검색 범위를 통해 현재 모션 특징을 검색합니다. 이는 기존 RAFT와는 다른 접근 방식으로, 전반적인 흐름 필드의 정확한 추정을 위한 전역 의존성을 더 잘 모델링할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, HMAFlow는 Sintel 온라인 벤치마크에서 RAFT에 비해 상대적으로 14.2% 및 3.4%의 오류 감소를 달성하며, KITTI 벤치마크에서는 각각 RAFT와 GMA에 대해 6.8% 및 7.7%의 우위를 보입니다. 이러한 성과는 제안된 모델의 효과성과 우수성을 증명합니다.



### An encoding of argumentation problems using quadratic unconstrained binary optimization (https://arxiv.org/abs/2409.05524)
- **What's New**: 본 논문에서는 여러 NP-Complete 문제를 Abstract Argumentation에서 Quadratic Unconstrained Binary Optimization (QUBO) 문제로 인코딩하는 방법을 개발하였습니다. 이 방법을 통해 QUBO 문제의 해결책은 이진 변수(0/1)로 이루어진 2차 함수의 최소화와 관련이 있으며, 이를 위해 대칭 정사각형 행렬 또는 동등한 상삼각형 형태의 계수 반복이 가능합니다.

- **Technical Details**: 이 연구에서는 Argumentation의 일부 고전 문제를 QUBO로 인코딩하여, 예를 들어 주어진 인자가 적어도 하나의 완전하고, 선호하며 안정된 확장에서 수용되는지에 대한 확인, 안정된 확장의 존재 여부, 완전하고 선호하는, 반안정 및 안정적 의미론을 사용할 때 비어 있지 않은 확장의 존재 여부, 특정 인자 집합 T가 주어졌을 때 AF를 수정하는 최소화 등의 문제를 고려하였습니다. 이러한 문제들은 모두 Argumentation에서 잘 알려진 NP-Complete 문제로, QUBO 문제의 해결과 동일한 복잡도 클래스를 보여줍니다.

- **Performance Highlights**: 제안된 해결책은 AFGCN 및 Harper++라는 두 기존 근사 솔버와 비교되었습니다. 최종 실험에서는 Simulated Annealing 알고리즘을 지역 컴퓨터에서 사용하였으며, D-Wave Ocean SDK를 이용하여 Quantum Annealer에서도 테스트를 진행하였습니다. 이러한 모형의 성능을 검증하기 위해 여러 가지 조사를 수행하였고, Classical 문제들에 대한 적용 가능성을 입증하였습니다.



### Harmonic Reasoning in Large Language Models (https://arxiv.org/abs/2409.05521)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 음악적 과제를 해결하는 능력을 조사했습니다. 특히 비슷한 모델인 GPT-3.5와 GPT-4o를 비교하여 음악의 노트 간격을 이해하고 코드 및 음계를 인식하는 방식에서의 차이를 분석했습니다. 연구 결과, LLM이 노트 간격에는 뛰어난 성능을 보였지만 더 복잡한 코드와 음계 인식에는 어려움을 겪는다는 점이 발견되었습니다.

- **Technical Details**: 대형 언어 모델(LLMs)은 음악 이론을 이해하기 위해 두 가지 실험을 수행했습니다. 첫 번째 실험에서는 음악 노트와 관련된 간격을 적용하고, 두 번째 실험에서는 코드와 음계를 인식했습니다. 이를 통해 LLM의 음악적 추론 능력을 테스트하고, 500개의 문제가 포함된 데이터 세트를 자동 생성했습니다. 모델의 성능은 각 실험에서 기대치와 비교하여 평가되었습니다.

- **Performance Highlights**: GPT-4o는 upward intervals에 대해 거의 100%의 정확도를 기록했지만, downward intervals 및 다중 옥타브를 포함한 가장 어려운 설정에서는 50% 이하의 정확도를 보였습니다. 이는 LLM이 훈련된 정보를 기억하고 재생산하는 능력은 있지만, 새로운 상황에 적합하게 사고하는 능력은 여전히 제한적임을 나타냅니다.



### Using machine learning for fault detection in lighthouse light sensors (https://arxiv.org/abs/2409.05495)
- **What's New**: 본 논문은 등대의 고장 감지를 위한 머신러닝 기반의 혁신적인 접근 방식을 소개합니다. 특히, 멀티레이어 퍼셉트론이 10-15분의 시간 차이를 감지할 수 있음을 입증하였습니다.

- **Technical Details**: 연구에서는 의사결정 나무(decision trees), 랜덤 포레스트(random forest), 극단적인 그래디언트 부스팅(extreme gradient boosting), 멀티레이어 퍼셉트론(multi-layer perceptron) 등 4가지 알고리즘을 평가하였습니다. 시뮬레이션된 센서 데이터에 대한 성능 평가는 정확도(accuracy)와 F1 점수를 기반으로 진행되었습니다.

- **Performance Highlights**: 멀티레이어 퍼셉트론은 다른 알고리즘에 비해 탁월한 성능을 보였습니다. 이 모델은 등대 선박의 안전을 위한 정밀한 유지 관리를 가능하게 합니다.



### Elsevier Arena: Human Evaluation of Chemistry/Biology/Health Foundational Large Language Models (https://arxiv.org/abs/2409.05486)
Comments:
          11 pages, 5 tables, 6 figures

- **What's New**: 이 논문에서는 Elsevier에서 수행된 생물 의학 분야를 위한 인공지능(AI) LLM(human evaluation experiment)을 소개합니다. 기존의 자동화된 벤치마크 평가의 한계를 극복하기 위해 A/B 테스트(A/B testing) 프레임워크를 적용하여 인간 평가자의 모델 선호도를 측정한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 Elsevier의 데이터로부터 훈련된 8.8B 파라미터의 LLM 모델과 OpenAI의 GPT-3.5-turbo, Meta의 Llama 2 모델을 평가하였습니다. 총 141개의 질문을 만들어 LLM의 출력 결과를 사실성(factuality), 일관성(coherence), 관련성(relevance), 개요(overview) 기준으로 평가하였습니다. A/B 테스트는 기본 LLM을 비교 기준으로 설정하여 수행하였으며, 모델은 온도(t=0.7) 기반 샘플링 방법으로 생성되었습니다.

- **Performance Highlights**: 결과는 GPT-3.5-turbo 모델이 대체로 더 높은 선호도를 보인 것으로 나타났지만, 상대적으로 작은 모델도 잘 curating된 데이터셋에서 훈련될 경우 생물 의학 분야에서 경쟁력 있는 대안이 될 수 있음을 시사합니다. 그러나 모든 모델의 IRR 점수는 일반적으로 낮았으며, 이는 인간 평가의 주관성과 불확실성을 반영합니다.



### CRADLE-VAE: Enhancing Single-Cell Gene Perturbation Modeling with Counterfactual Reasoning-based Artifact Disentanglemen (https://arxiv.org/abs/2409.05484)
- **What's New**: 본 논문에서는 CRADLE-VAE라는 새로운 인과적 생성 프레임워크를 제안하여, 단일 세포 유전자 교란 모델링을 위해 Counterfactual Reasoning(상대적 추론)을 기반으로 기술적 아티팩트를 분리하는 방법을 제시합니다. 이는 단일 세포 데이터셋의 질적 문제를 해결하기 위한 첫 번째 시도입니다.

- **Technical Details**: CRADLE-VAE는 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터에서의 기술적 아티팩트를 모델링하고 이를 효과적으로 분리하여 강력한 세포 반응 데이터를 생성합니다. 이 모델은 논문 내에서 제안된 보조 손실 목표(auxiliary loss objective)를 사용하여 아티팩트를 분리합니다. Counterfactual reasoning을 활용하여 QC를 통과한 데이터에서 발생 가능한 아티팩트를 식별하고 이와 관련된 세포의 자연스러운 변화를 모델링합니다.

- **Performance Highlights**: 실험 결과 CRADLE-VAE는 이전 방법들 보다 높은 품질의 유전자 발현 프로파일을 생성하면서 세포 반응 예측에서 우수한 상관관계와 생성 품질을 보여주었습니다. 특히 보지 못한 교란에 대한 입력을 제공할 때 더욱 뛰어난 성능을 발휘했습니다.



### Proto-OOD: Enhancing OOD Object Detection with Prototype Feature Similarity (https://arxiv.org/abs/2409.05466)
Comments:
          14pages

- **What's New**: 최근 연구에서 OOD (Out-of-Distribution) 객체 탐지의 정확도를 높이기 위한 Proto-OOD라는 새로운 네트워크 아키텍처가 제안되었습니다. 이 모델은 프로토타입 학습을 활용하여 입력 특징과 프로토타입의 유사성을 평가합니다.

- **Technical Details**: Proto-OOD는 다양한 클래스에 대한 특징 벡터를 수집하여 프로토타입으로 사용하며, 대조 손실(contrastive loss)을 통해 프로토타입의 대표성을 향상시킵니다. 또한, negative embedding generator를 활용하여 부정 임베딩을 생성하고, 이를 유사성 모듈을 훈련시키는 데 사용합니다.

- **Performance Highlights**: Proto-OOD는 MS-COCO 데이터셋에서 FPR95를 현저히 낮추고, Pascal VOC 데이터셋에서는 더 높은 mAP를 달성합니다. 기존 평가 메트릭의 한계를 분석하고, False Positive를 필터링하여 성과를 재평가하는 개선된 평가 프로토콜을 제안하였습니다.



### Visualizing Extensions of Argumentation Frameworks as Layered Graphs (https://arxiv.org/abs/2409.05457)
- **What's New**: 이번 논문에서는 새로운 시각화 기술을 소개하여 argumentation frameworks (AFs)를 3-layer 그래프 레이아웃으로 그릴 수 있게 하였습니다. 이는 AF와 extension을 함께 시각화하도록 설계되어 사용자가 더욱 쉽게 AF를 탐색하고 이해할 수 있도록 지원합니다.

- **Technical Details**: 이 기술은 edge crossings를 최소화하여 시각적 선명도와 미적 요소를 최적화합니다. 정확한 Integer Linear Programming (ILP) 방법과 빠른 heuristic 파이프라인이 제안되었으며, 후자는 큰 인스턴스에서도 효과적임을 나타내었습니다.

- **Performance Highlights**: 정량적 평가를 통해 제안된 heuristic 방식이 최적 드로잉에 비해 최대 두 배의 edge crossings를 생성하며, 대부분의 경우 효과적인 성능을 보임을 실증하였습니다. 이 기술은 법률, 의학, 전자 민주주의 등 다양한 분야에서의 적용 가능성을 제시합니다.



### State-Novelty Guided Action Persistence in Deep Reinforcement Learning (https://arxiv.org/abs/2409.05433)
Comments:
          Under review

- **What's New**: 이 논문에서는 Deep Reinforcement Learning (DRL)에서 샘플 비효율성 문제를 해결하기 위해 동적으로 행동 지속성(action persistence)을 조정하는 새로운 방법을 제안합니다.

- **Technical Details**: 기존의 연구들은 고정된 전략을 사용하거나 반복 수를 선택하기 위해 추가적인 가치 함수(value function) 또는 정책(policy)을 학습하는 방식을 취했습니다. 본 방법은 상태 공간의 현재 탐색 상태에 기반하여 행동 지속성을 동적으로 조정하며, 추가적인 학습이 필요 없습니다. 반복 확률의 부드러운 스케줄링을 통해 탐색과 활용(exploitation) 간의 더 효과적인 균형을 이룰 수 있습니다.

- **Performance Highlights**: DMControl 작업에 대한 광범위한 실험 결과, 제안된 상태-신선도(state-novelty) 기반 행동 지속성 방법이 샘플 효율성을 크게 향상시킨다는 것을 보여줍니다.



### AD-Net: Attention-based dilated convolutional residual network with guided decoder for robust skin lesion segmentation (https://arxiv.org/abs/2409.05420)
- **What's New**: 본 연구는 피부 암 진단 및 치료를 위한 컴퓨터 지원 진단 도구에서 피부 병변의 정확한 세분화(segmentation)를 다룸. 이 논문에서는 dilated convolutional residual network를 활용하여 공간적 특징을 향상시키는 attenzione 기반의 ASFEB(block)와 guided decoder 전략을 제안한다.

- **Technical Details**: 연구의 주요 기법은 다양한 dilation rate을 사용하는 dilated convolution을 통해 receptive field를 확장하는 residual block 기반의 신경망인 AD-Net이다. ASFEB는 average 및 maximum pooling 작업으로부터 얻은 feature map을 결합하여 공간적 feature 정보를 개선하며, guided decoder는 각 decoder block을 개별적인 손실 함수로 최적화하여 feature 학습을 향상시킨다.

- **Performance Highlights**: 제안된 AD-Net은 다른 최신 방법들과 비교했을 때, 적은 모델 파라미터를 요구하며 더 빠른 수렴(convergence)을 보였다. 4개의 공공 벤치마크 데이터셋을 통해 AD-Net의 성능이 평가되었으며, 연구 결과는 데이터 증강(data augmentation) 전략을 사용하지 않더라도 우수한 성능을 보임을 시사한다.



### CipherDM: Secure Three-Party Inference for Diffusion Model Sampling (https://arxiv.org/abs/2409.05414)
- **What's New**: 이번 논문에서는 Diffusion Models (DMs)의 개인 정보 보호 문제를 해결하기 위해 CipherDM이라는 새로운 프레임워크를 제안합니다. 이는 Secure Multi-Party Computation (MPC) 기술을 활용하여 DM의 샘플링 과정을 안전하게 수행할 수 있도록 합니다. 이러한 접근은 기존 DMs에서 발생할 수 있는 사용자 프라이버시 침해를 방지하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: CipherDM은 ABY3 프로토콜을 기반으로 하는 최초의 MPC 기반 DM 샘플링 프레임워크로, 3PC (Three-Party Computation) 복제 비밀 공유 방식을 사용하여 세 개의 클라우드 서버에서 샘플링을 수행합니다. 샘플링 과정에서 개인 데이터나 모델 파라미터가 외부에 노출되지 않도록 보장합니다. 추가적으로, 이 연구는 SoftMax, SiLU, Mish와 같은 비선형 활성화 기능에 대한 안전한 MPC 프로토콜을 설계합니다.

- **Performance Highlights**: CipherDM은 MNIST 데이터셋과 diffusers에 의해 배포된 Stable Diffusion (SD)에서 DDPM 및 DDIM 아키텍처를 사용하여 평가되었습니다. SPU에 직접 구현한 것에 비해 실행 시간은 약 1.084배에서 2.328배 개선되었으며, 통신 비용은 약 1.212배에서 1.791배 감소했습니다. 이는 DM 안전한 샘플링을 위한 MPC의 유효성을 입증하는 결과입니다.



### A Survey of Multimodal Composite Editing and Retrieva (https://arxiv.org/abs/2409.05405)
Comments:
          22 pages, 3 figures, and 11 tables

- **What's New**: 이번 설문조사는 다중 모달 복합 검색(multimodal composite retrieval)에 대한 포괄적인 리뷰를 제공하는 첫 번째 논문으로, 이미지-텍스트 복합 편집 및 검색 방법을 심도 있게 탐구합니다. 다중 모달 데이터 유형의 통합을 통해 개인화된 정보 검색의 개선을 꾀하고 있습니다.

- **Technical Details**: 다중 모달 복합 검색은 텍스트, 이미지, 오디오 등 다양한 모달리티를 통합하여 더 정확하고 맞춤형 결과를 제공합니다. 최근에는 Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) 네트워크, Visual Transformer (ViT) 등을 사용하여 이미지 검색 성능을 향상시키는 방법들이 제안되고 있습니다.

- **Performance Highlights**: 다중 모달 복합 검색 방법들은 사용자 질의와 맥락을 더 잘 이해함으로써 검색 성능과 사용자 만족도를 개선하는 데 기여하고 있습니다. 다양한 모달리티를 활용하는 이러한 기술들은 지난 몇 년간 많은 연구가 진행되어오며, 특히 패션 산업, 의료 진단 등 여러 분야에 활발히 응용되고 있습니다.



### HyperSMOTE: A Hypergraph-based Oversampling Approach for Imbalanced Node Classifications (https://arxiv.org/abs/2409.05402)
- **What's New**: 본 논문에서는 Hypergraphs(하이퍼그래프)의 클래스 불균형(class imbalance) 문제를 해결하기 위해 새로운 방법인 HyperSMOTE를 제안합니다. 기존의 GraphSMOTE 기술은 하이퍼그래프의 독특한 구조를 처리하지 못하는 한계가 있습니다.

- **Technical Details**: HyperSMOTE는 두 단계로 이루어져 있습니다: 첫째, 소수 클래스(minority class) 노드를 합성(synthesize)하고, 둘째, 이러한 노드를 기존 하이퍼그래프에 통합(integrate)하는 것입니다. 이 과정에서 소수 클래스 샘플과 이웃 샘플을 기반으로 새로운 노드를 생성하며, 원래 하이퍼그래프의 incidence matrix를 이용하여 Augmented 노드를 하이퍼엣지와 연관시키기 위한 디코더를 학습합니다.

- **Performance Highlights**: HyperSMOTE는 여러 단일 모달 데이터셋(Cora, Cora-CA, Citeseer)과 다중 모달 대화 데이터셋(MELD)에서 평가되었으며, 각각 평균적으로 3.38% 및 2.97%의 정확도 향상을 보였습니다.



### FacialFlowNet: Advancing Facial Optical Flow Estimation with a Diverse Dataset and a Decomposed Mod (https://arxiv.org/abs/2409.05396)
Comments:
          ACMMM2024

- **What's New**: 본 논문은 FacialFlowNet(FFN)라는 새로운 대규모 얼굴 광학 흐름(optical flow) 데이터셋과, 얼굴 흐름을 분해하는 최초의 방법인 Decomposed Facial Flow 모델(DecFlow)을 제안합니다. FFN은 9,635명의 다양한 정체성과 105,970개의 이미지 쌍으로 구성되어 있어, 얼굴 및 머리 움직임 분석의 다양성을 제공합니다.

- **Technical Details**: FFN 데이터셋은 얼굴 움직임을 나타내는 이미지 쌍과 머리 흐름(head flow) 라벨을 제공합니다. DecFlow는 얼굴 의미 중심 인코더(facial semantic-aware encoder)와 분해 흐름 디코더(decomposed flow decoder)를 특징으로 하며, 얼굴 흐름을 머리 흐름과 표현 흐름으로 정확하게 분해할 수 있습니다. 이 접근 방식은 미세한 표정을 정확하게 분석하는 데 용이합니다.

- **Performance Highlights**: FFN은 다양한 광학 흐름 방법에서 얼굴 흐름 추정의 정확성을 크게 향상시키며, 최대 11%의 Endpoint Error(EPE) 감소(3.91에서 3.48로)를 달성합니다. DecFlow는 FFN과 결합하여 기존 방법들보다 우수한 성능을 보이며, 미세 표정 인식에서 18%의 정확도 향상(69.1%에서 82.1%로)을 달성했습니다.



### Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling (https://arxiv.org/abs/2409.05395)
- **What's New**: 이 연구는 최근의 구조적 상태 공간 모델(SSM)인 Mamba를 비주얼 언어 모델(VLM)에서 Transformer와 대체하여 성능을 비교합니다. Mamba가 Transformer 기반 VLM보다 캡션 작성, 질의 응답 및 독서 이해에서 더 나은 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: 모델은 최대 30억 개의 매개변수로 테스트되었으며, Mamba 기반 VLM이 Transformer 기반 VLM을 네 개의 기준 제시 작업에서 초과한 결과가 나왔습니다. 그러나 Transformer는 시각적 기초 작업에서 더 나은 성과를 보였으며, 이는 모델 크기가 커질수록 더욱 두드러졌습니다.

- **Performance Highlights**: Mamba는 이미지의 요약에 의존하는 작업에서는 유망한 성능을 보여주지만, 문맥에서 명시적 정보를 검색해야 하는 작업에서는 어려움을 겪었습니다. 연구 결과 Mamba는 캡션 작성, 질문 응답 및 독해에서 Transformer를 초과했지만, grounding 작업에서 Transformer가 더 높은 성능을 보여주었습니다.



### Towards Building a Robust Knowledge Intensive Question Answering Model with Large Language Models (https://arxiv.org/abs/2409.05385)
Comments:
          This paper has been accepted by NLPCC-2024

- **What's New**: 본 논문에서는 LLM의 견고성을 평가하기 위해 기계 독해(Machine Reading Comprehension, MRC) 데이터셋을 기반으로 다양한 시나리오를 시뮬레이션하는 새로운 데이터셋을 구축하였습니다. 추가로, 노이즈와 정보 결여를 해결하기 위한 데이터 증강(data augmentation) 기반의 파인튜닝(fine-tuning) 방법을 제안하였습니다.

- **Technical Details**: 연구의 핵심은 LLM을 다양한 외부 정보에 노출시키고, 이를 통해 모델의 정확성을 평가하는 것입니다. 이를 위해 Single Source (SS), Single-Source-Incomplete (SSIncomp), Multi-Source-Consistent (MSCons) 및 Multi-Source-Inconsistent (MSIncons)와 같은 다양한 데이터셋을 구축하였습니다. 또한, 대조 학습(contrastive learning) 접근 방식을 통해 모델의 외부 정보 및 내부 지식의 활용 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법이 모델의 견고성을 개선하고, 외부 정보에 대한 차별화(discrimination) 능력을 강화함을 확인하였습니다. GPT-4를 활용한 성능 평가에서 개선된 결과를 나타내었습니다.



### Look One and More: Distilling Hybrid Order Relational Knowledge for Cross-Resolution Image Recognition (https://arxiv.org/abs/2409.05384)
Comments:
          Accepted by AAAI 2020

- **What's New**: 이번 논문에서는 저해상도 이미지 인식을 위한 하이브리드 오더 관계 지식 증류(hybrid order relational knowledge distillation) 접근법을 제안합니다. 이 방법은 고해상도 이미지에서의 인식 능력을 줄임으로써 저해상도 이미지 인식을 향상시킵니다.

- **Technical Details**: 제안된 방법은 세 가지 스트림, 즉 튜터 스트림(teacher stream), 학생 스트림(student stream), 보조 스트림(assistant stream)으로 구성됩니다. 튜터 스트림은 고해상도 이미지를 매우 정확하게 인식할 수 있도록 사전 훈련(pre-trained)됩니다. 학생 스트림은 튜터의 행동을 모방하여 저해상도 이미지를 인식하게 되며, 보조 스트림은 지식 전이를 돕기 위해 튜터와 학생 사이의 다리 역할을 수행합니다.

- **Performance Highlights**: 다양한 작업(metric learning, 저해상도 이미지 분류, 저해상도 얼굴 인식)을 통해 실험한 결과, 제안된 접근법이 메모리 사용량이 줄어들고 속도가 빨라지면서 저해상도 이미지 인식에서 인상적인 성능을 보여주었습니다.



### Deep Learning for Video Anomaly Detection: A Review (https://arxiv.org/abs/2409.05383)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 연구는 VAD(비디오 비정상 탐지)에 대한 포괄적인 리뷰를 제공하며, 다양한 감독 신호에 기반한 다섯 가지 VAD 작업을 다룹니다. 또한 최신 연구 동향인 오픈 세트 감독 VAD와 사전 훈련된 대형 모델 기반 VAD 방법을 재조명합니다.

- **Technical Details**: 연구는 VAD 작업을 반지도학습(semi-supervised), 약지도학습(weakly supervised), 완전 지도학습(fully supervised), 비지도학습(unsupervised), 오픈 세트 감독(open-set supervised) VAD로 분류하여 체계적인 분류 체계를 제안합니다. 각 방법의 특징 및 성능 비교에 대해 심층적으로 논의합니다.

- **Performance Highlights**: 최근 7년간의 성과 통계를 통해 VAD 방법의 AUC(Area Under the Curve)가 CUHK Avenue 데이터셋에서 70.2%에서 90.1%로 향상된 것을 보여줍니다. 다양한 데이터셋에서 SOTA(state-of-the-art) 방법의 성능이 지속적으로 상승하고 있음을 강조합니다.



### PersonaTalk: Bring Attention to Your Persona in Visual Dubbing (https://arxiv.org/abs/2409.05379)
Comments:
          Accepted at SIGGRAPH Asia 2024 (Conference Track)

- **What's New**: 이 논문에서는 audio-driven visual dubbing(오디오 기반 비주얼 더빙) 분야에서 speaker의 개성과 정확한 lip synchronization(립 동기화)을 유지하는 데 중점을 둔 새로운 프레임워크인 PersonaTalk를 제시합니다. 기존 방법들은 speaker의 특정 스타일이나 얼굴의 세부 정보를 잘 반영하지 못하는 문제가 있었습니다.

- **Technical Details**: PersonaTalk는 attention 기반의 두 단계 프레임워크로 구성되어 있으며, 첫 번째 단계에서는 style-aware audio encoding module(스타일 인식 오디오 인코딩 모듈)을 사용하여 오디오 특징에 speaking style(말하기 스타일)을 주입합니다. 두 번째 단계에서는 dual-attention face renderer(이중 주의 얼굴 렌더러)를 통해 target geometries(대상 기하형상)에 대한 텍스처를 렌더링합니다. 이 렌더러는 Lip-Attention(립 어텐션)과 Face-Attention(얼굴 어텐션)의 두 개의 병렬 크로스 어텐션 레이어로 구성되어 있습니다.

- **Performance Highlights**: Comprehensive experiments(포괄적인 실험)과 user studies(사용자 연구)를 통해 PersonaTalk는 시각 퀄리티(visual quality), lip-sync accuracy(립 동기화 정확도), persona preservation(개인화 보존)의 측면에서 최고의 방법들에 비해 우수한 성능을 보여주었다. 또한, 이 방법은 person-generic(개인 일반화) 프레임워크이면서도 state-of-the-art person-specific(최첨단 개인화) 방법들과 경쟁할 수 있는 성능을 달성할 수 있습니다.



### KARGEN: Knowledge-enhanced Automated Radiology Report Generation Using Large Language Models (https://arxiv.org/abs/2409.05370)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)을 활용하여 자동화된 방사선 보고서 생성(R2Gen)을 향상시키기 위한 새로운 프레임워크인 KARGEN을 제시합니다. 이는 질병 관련 지식을 활성화시키기 위해 특정 지식 그래프를 통합한 최초의 시도입니다.

- **Technical Details**: KARGEN은 대규모 언어 모델의 보고서 생성 능력을 활용하는 동시에, 의학적 지식 그래프를 통합하여 방사선 이미지를 분석합니다. 이 과정에서 이미지에서 지역적 특징을 추출하고, 지식 그래프를 사용하여 질병 관련 정보를 '증류'하여 모델에게 제공합니다. 최종 보고서는 이러한 통합된 특징을 기반으로 LLaMA 기반의 리포트 생성기를 통해 작성됩니다.

- **Performance Highlights**: KARGEN은 IU-Xray 및 MIMIC-CXR 데이터셋에서 검증되었으며, 여러 최신 방법들과 비교하여 다양한 평가 지표에서 뛰어난 성능을 보였습니다. 이는 LLM 내의 직관적인 지식을 효과적으로 활용했음을 보여줍니다.



### BAMDP Shaping: a Unified Theoretical Framework for Intrinsic Motivation and Reward Shaping (https://arxiv.org/abs/2409.05358)
- **What's New**: 이번 연구에서는 강화 학습(RL)에서 내부 동기화(Intrinsic Motivation, IM)와 보상 형성(Reward Shaping) 기법이 어떻게 탐색을 유도하는지를 다루며, 이를 Bayes-Adaptive Markov Decision Processes (BAMDPs)로 모델링하여 탐색의 가치를 포매틱하게 정의합니다.

- **Technical Details**: BAMDPs가 경험을 통해 가능한 MDP들에 대한 prior를 업데이트하는 과정으로 강화 학습 프로세스를 형식화합니다. RL 알고리즘은 BAMDP 정책으로 볼 수 있으며, 이 연구는 가상의 보상이 어떻게 비최적 알고리즘의 탐색을 안내하는지 이해하기 위한 이론적 틀로써 BAMDP를 사용합니다.

- **Performance Highlights**: Pseudo-rewards가 BAMDP Potential-based shaping Functions (BAMPFs)로 설계되었을 때, RL 알고리즘의 최적 혹은 근사 최적 행동을 유지한다고 제안합니다. 반면, 잘못 설계된 경우 최적 학습자를 손상시킬 수 있음을 강조합니다.



### TriplePlay: Enhancing Federated Learning with CLIP for Non-IID Data and Resource Efficiency (https://arxiv.org/abs/2409.05347)
- **What's New**: 이 연구는 CLIP와 같은 대형 pretrained 모델을 Federated Learning (FL) 프레임워크에 통합하는 방법을 탐구하여 프라이버시, 효율성 및 적응성을 향상시킵니다. 간단한 주의(attention) 기반 어댑터를 통하여 특정 작업에 빠르게 적응할 수 있도록 합니다.

- **Technical Details**: TriplePlay는 CLIP 모델을 어댑터로 통합하여 FL 시스템의 적응성과 성능을 강화합니다. 이 접근법은 'long-tail distribution' 문제를 해결하여 공정성을 보장하고, quantization 및 low-rank adaptation 기술을 통해 자원 요구를 감소시키도록 설계되었습니다. 또한 FL에서 높은 대역폭과 통신 오버헤드를 최소화합니다.

- **Performance Highlights**: 시뮬레이션 결과, TriplePlay는 GPU 사용 비용을 효과적으로 줄이고 학습 과정을 가속화하며 통신 오버헤드 감소와 함께 수렴을 달성함을 보여줍니다.



### GDFlow: Anomaly Detection with NCDE-based Normalizing Flow for Advanced Driver Assistance System (https://arxiv.org/abs/2409.05346)
- **What's New**: 본 연구는 전기차의 첨단 운전 보조 시스템(ADAS)에서 주행 패턴을 효과적으로 모델링하기 위해 Normalizing Flow(NF)와 Neural Controlled Differential Equations(NCDE)를 활용한 Graph Neural Controlled Differential Equation Normalizing Flow(GDFlow) 모델을 제안합니다. 기존의 기법들과 달리, GDFlow는 다변량 시계열(MTS) 센서 데이터로부터의 시공간(spatio-temporal) 정보를 효과적으로 포착합니다.

- **Technical Details**: GDFlow는 NCDE 기반의 NF를 사용하여 정상 주행 패턴의 분포를 지속적으로 학습하며, 새로운 사용자 주행 패턴 변화에 적응할 수 있도록 설계되었습니다. 또한, 배포의 경계 근처에서 정상 데이터에 대한 우도 추정(likelihood estimate)을 향상시키기 위해 양자 기반의 최대 우도 객체를 도입하고 있습니다.

- **Performance Highlights**: 현실 세계에서 수집한 현대 아이오닉5 및 GV80EV 전기차 주행 데이터를 사용하여 GDFlow의 성능을 검증한 결과, 4가지 차량 유형과 운전자의 데이터 세트 구성에 대해 6개의 기준선(baselines)과 비교하여 최첨단 성능(State-of-the-Art performance)을 기록했습니다. 또한, GDFlow는 4개의 시계열 벤치마크 데이터셋에서 최신 이상 탐지 기법보다 높은 성능을 보였습니다.



### Early-exit Convolutional Neural Networks (https://arxiv.org/abs/2409.05336)
- **What's New**: 이 논문은 추론(Inference) 과정에서 합성곱 신경망(CNN)의 계산 비용을 줄이는 방법을 개발하는 데 초점을 맞추고 있습니다. "Early-exit CNNs"(EENets)라는 새로운 아키텍처를 도입하여, 입력값에 따라 특정 exit 지점에서 추론 과정을 중단함으로써 계산 비용을 조절합니다.

- **Technical Details**: EENets는 여러 개의 exit 블록으로 구성되어 있으며, 각각의 블록은 confidence branch와 softmax branch로 이루어져 있습니다. Confidence branch는 해당 위치에서 추론 과정을 종료할 수 있는 신뢰 점수를 계산하는 반면, softmax branch는 분류 확률 벡터를 출력합니다. 이 두 가지 블록은 학습 가능하며 서로 독립적입니다. 훈련 중에 EENets는 전통적인 분류 손실 이외에도 계산 비용을 고려하여, 쉬운 예제에 대해 더 적은 계산 리소스를 사용하도록 조정됩니다.

- **Performance Highlights**: EENets는 ResNets와 유사한 정확도를 유지하면서도 SVHN 데이터세트에서 30%, CIFAR10에서 20%, Tiny-ImageNet에서 42%의 상대적 계산 비용 절감을 달성했습니다. 이를 통해 EENets는 초기 종료 네트워크의 효과를 잘 보여줍니다.



### A Multi-Modal Deep Learning Based Approach for House Price Prediction (https://arxiv.org/abs/2409.05335)
Comments:
          22 pages

- **What's New**: 이 논문은 주택 가격 예측 시스템에서 텍스트와 비주얼 특성을 포함한 다양한 속성을 종합적으로 통합하여 향상된 정확도를 목표로 합니다. 특히, 다중 모달 딥러닝 접근법을 통해 집의 속성, 지리적 특성, 텍스트 설명 및 이미지의 조합을 활용합니다.

- **Technical Details**: 제안하는 Multi-Modal House Price Predictor (MHPP)는 집의 속성, 지리적 이웃, 텍스트 설명, 이미지로부터의 임베딩을 학습합니다. 이를 위해 BERT와 CLIP 모델을 활용하여 세 가지 유형의 데이터에서의 최적 조합 임베딩을 학습하여 다운스트림 회귀 모델에 입력합니다.

- **Performance Highlights**: 실험 결과, 텍스트 설명에 대한 임베딩과 집 사진의 임베딩을 포함한 접근이 주택 가격 예측의 정확도를 크게 향상시킴을 보여줍니다. 52,851개의 멜버른 주택 거래 데이터셋을 활용하여 이러한 결과를 확인했습니다.



### Sample-Efficient Bayesian Optimization with Transfer Learning for Heterogeneous Search Spaces (https://arxiv.org/abs/2409.05325)
- **What's New**: 이 논문은 새롭고 강력한 Bayesian optimization (BO) 접근 방식으로, 이질적인 탐색 공간에서 이전 실험의 정보를 전이하는 방법을 제안합니다.

- **Technical Details**: 첫 번째 접근 방식은 조건부 커널을 가진 Gaussian process (GP) 모델을 활용하여 서로 다른 탐색 공간 간의 정보를 전이합니다. 두 번째 접근 방식은 누락된 매개변수를 GP 모델의 하이퍼파라미터로 취급하여 다른 GP 하이퍼파라미터와 함께 추론하거나 고정값으로 설정합니다.

- **Performance Highlights**: 이 두 가지 방법은 여러 벤치마크 문제에서 우수한 성능을 보이는 것으로 나타났습니다.



### Machine Anomalous Sound Detection Using Spectral-temporal Modulation Representations Derived from Machine-specific Filterbanks (https://arxiv.org/abs/2409.05319)
- **What's New**: 이번 논문에서는 기계 이상 소리 탐지(Anomalous Sound Detection, ASD)를 위한 새로운 접근법을 제안합니다. 이 접근법은 인간의 청각 시스템의 계산 모델과 기계 특정 속성을 통합하여, 독특한 진동 주파수 범위를 갖는 다양한 기계에서의 이상 소리를 효과적으로 탐지하는 것을 목표로 합니다.

- **Technical Details**: 연구자들은 먼저 Fisher 비율(F-ratio)을 사용하여 네 가지 종류의 기계의 주파수 중요성을 정량화했습니다. 이 중요성 데이터는 비균일 필터뱅크(Non-Uniform Filterbanks, NUFBs)를 설계하는 데 사용되었으며, 이 필터뱅크는 로그 비균일 스펙트럼(Log Non-Uniform Spectrum, LNS) 특징을 추출합니다. LNS 특징은 자가 인코더 신경망(Autoencoder Neural Network)을 기반으로 한 탐지기에 입력됩니다.

- **Performance Highlights**: 테스트 결과, Malfunctioning Industrial Machine Investigation and Inspection 데이터셋의 학습 세트에서, 기계 간의 정상 및 이상 소리가 주파수 영역에서 비균일하게 인코딩된 정보를 나타내는 것으로 나타났습니다. 설계된 NUFBs를 사용함으로써, 다양한 신호 대 잡음 비율(signal-to-noise ratio, SNR) 조건에서 수혜(Gain, AUC) 성능이 크게 향상되었습니다. 특히 시간 변조(Temporal Modulation) 및 주파수 변조(Spectral Modulation) 표현을 통합하는 것이 기계의 이상 탐지 성능을 더 개선했습니다.



### Tele-LLMs: A Series of Specialized Large Language Models for Telecommunications (https://arxiv.org/abs/2409.05314)
- **What's New**: 이 논문은 통신 분야를 위한 대형 언어 모델(LLM)의 필요성을 강조하며, 통신 전문화가 결여된 일반 모델의 한계를 분석합니다. 저자들은 Tele-Data라는 통신 관련 데이터셋과 Tele-Eval이라는 질문-답변 데이터셋을 생성하여 이 분야에서 진정한 적용 가능성을 탐구합니다.

- **Technical Details**: 통신 분야의 데이터셋은 LLM 기반 필터링을 통해 수집된 Material로 구성되며, 총 750,000개의 질문-답변 쌍으로 이루어진 Tele-Eval은 통신 분야 최초의 대규모 오픈-엔드 QnA 데이터셋입니다. 또한, LLM의 적응을 위해 파라미터 효율적 미세 조정(Parameter-Efficient Fine-Tuning) 기법의 적합성을 검토합니다. 실험은 1B에서 8B 파라미터의 LLM 시리즈를 만들고, 그 성능을 기존 모델과 비교합니다.

- **Performance Highlights**: Tele-Eval에 대한 평가 결과, 해당 모델은 일반 모델보다 평균 25% 향상된 성능을 보여주며, 적응된 소형 모델이 큰 일반 모델과 경쟁할 수 있는 능력을 갖추고 있음을 입증합니다. 이 모델들은 원래의 능력을 유지하며 CAT(재앙적 망각) 현상을 피하는 데 성공합니다.



### Closed-Form Interpretation of Neural Network Latent Spaces with Symbolic Gradients (https://arxiv.org/abs/2409.05305)
- **What's New**: 이 논문에서는 인공 신경망의 잠재 공간에서 뉴런의 개념을 해석하기 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 기호 검색 공간에서 정의된 인간이 읽을 수 있는 방정식과의 교차점을 찾아 정보를 추출합니다.

- **Technical Details**: 이 해석 프레임워크는 같은 개념을 인코딩하는 함수의 동등 클래스(equivalence class)로 학습된 신경망을 삽입하는 방식으로 구성됩니다. 이를 통해 시암 신경망(Siamese networks)의 잠재 공간에서 행렬 불변량(matrix invariants) 및 역학 시스템의 보존 물리량(conserved quantities)을 검색하는 과정을 보여줍니다.

- **Performance Highlights**: 이 연구는 인공 신경망의 잠재 공간에서 저차원 뉴런을 해석할 수 있는 새로운 가능성을 제시하며, 이전에는 알지 못했던 과학적 통찰력을 발견할 기회를 제공합니다. 이러한 접근 방식은 Neural networks의 해석 가능성(interpretable AI)을 높이고, 과학 및 의학 분야에서의 실용성을 강화합니다.



### Resource-Efficient Generative AI Model Deployment in Mobile Edge Networks (https://arxiv.org/abs/2409.05303)
- **What's New**: 인공지능 생성 콘텐츠(AIGC)의 빠른 발전은 콘텐츠 제작 및 생산의 혁신적인 시대를 알리고 있습니다. 본 논문에서는 엣지 서버를 활용하여 생성 AI 모델 배치 관리의 복잡성을 해결하는 방법을 제안합니다.

- **Technical Details**: 본 연구는 생성 AI 모델의 리소스 소비 및 지연 요구 사항을 파악하여 모델 배치 문제를 최적화 문제로 공식화합니다. 이와 함께 리소스 소비와 지연 간의 균형을 최적화하는 모델 수준 결정 선택 알고리즘을 제안합니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 시스템 설정에서 다른 기준선과 비교하여 자원 소비를 줄이고 지연을 최소화하는 성과를 보였습니다. 시뮬레이션 결과는 AIGC 서비스의 효율적인 엣지 모델 배치를 가능하게 함을 확인시켜줍니다.



### TERD: A Unified Framework for Safeguarding Diffusion Models Against Backdoors (https://arxiv.org/abs/2409.05294)
- **What's New**: 이 논문은 이미지 생성에서의 Diffusion Models의 백도어 공격(backdoor attacks)에 대한 방어 방안을 제안합니다. 새로운 방어 프레임워크인 TERD(Trigger Estimation and Refinement for Diffusion)를 통해 기존 공격에 대한 통합 모델링을 구축하고, 역 손실(reversed loss)을 도출할 수 있도록 하였습니다. 또한, 트리거 반환(triggers reversion) 전략을 도입하여 보다 안전한 방어를 진행합니다.

- **Technical Details**: TERD는 초기 추정을 기반으로 시뮬레이션된 노이즈를 사용하여 트리거를 추정하고, 다단계 샘플러(differential multi-step samplers)를 통해 이를 정제합니다. 또한, KL 다이버전스(KL divergence)를 계산하는 새로운 모델 탐지 알고리즘을 제안하여 노이즈 공간에서 백도어를 감지합니다. 이 과정은 백도어 입력을 식별하고 무력화하는 데 중점을 둡니다.

- **Performance Highlights**: TERD는 다양한 해상도의 데이터셋에서 100%의 진정 양성률(True Positive Rate, TPR) 및 진정 음성률(True Negative Rate, TNR)을 달성했습니다. 또한, SDE(Stochastic Differential Equation) 기반 모델에 대한 방어력도 뛰어난 적응성을 보여주었습니다.



### Mpox Narrative on Instagram: A Labeled Multilingual Dataset of Instagram Posts on Mpox for Sentiment, Hate Speech, and Anxiety Analysis (https://arxiv.org/abs/2409.05292)
- **What's New**: 이번 연구는 WHO가 국제적인 공중보건 비상사태로 선언한 mpox 발생과 관련된 인스타그램 게시물의 데이터셋을 최초로 개발했습니다. 이 데이터셋은 2022년 7월 23일부터 2024년 9월 5일 사이에 작성된 60,127개의 인스타그램 게시물을 포함하고 있으며, 52개 언어로 제공됩니다.

- **Technical Details**: 개발된 데이터셋은 각 게시물에 대한 Post ID, 게시물 설명, 게시물 날짜, 언어, 번역된 버전(구글 번역 API를 사용하여 영어로 번역됨)을 포함하는 여러 속성으로 구성되어 있습니다. 이후에는 감정 분석(sentiment analysis), 혐오 발언 탐지(hate speech detection), 불안 또는 스트레스 탐지(anxiety or stress detection)가 수행되었습니다.

- **Performance Highlights**: 감정 클래스의 분포는 두려움(fear) 27.95%, 놀람(surprise) 2.57%, 기쁨(joy) 8.69%, 슬픔(sadness) 5.94%, 분노(anger) 2.69%, 혐오(disgust) 1.53%, 중립(neutral) 50.64%로 나타났습니다. 혐오 발언 탐지 결과, 95.75%의 게시물이 혐오 발언을 포함하지 않았고, 나머지 4.25%만이 혐오 발언을 포함했습니다. 또한, 72.05%의 게시물은 불안/스트레스가 없었고, 27.95%의 게시물이 어떤 형태의 불안/스트레스를 나타냈습니다.



### Seek and Solve Reasoning for Table Question Answering (https://arxiv.org/abs/2409.05286)
- **What's New**: 이 논문은 Table-based Question Answering (TQA) 성능을 향상시키기 위해 LLMs(대형 언어 모델)의 추론 능력을 활용하는 새로운 "Seek-and-Solve" 파이프라인을 제안합니다. 이 방법은 인간의 문제 해결 방식을 모방하여 두 단계로 나누어 질문 관련 정보를 찾고, 이후 질문을 해결하는 방식을 통합합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계인 "Seek"에서 LLM은 테이블 내 질문 관련 정보를 탐색하고, 이 정보를 바탕으로 추론의 단계를 나타내는 Seek Chain of Thought (CoT)를 생성합니다. 두 번째 단계인 "Solve"에서는 Seek-CoT를 사용하여 추론을 진행합니다. 이 과정에서 테이블 구조를 이해하기 위해 노드 기반 트리 구조를 모델링하여 복잡한 테이블에서 정보를 쉽게 찾아낼 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SS-CoT 경로를 사용하는 샘플을 시연으로 활용하면 LLM이 복잡한 TQA 작업을 효과적으로 해결하고 성능과 신뢰성이 향상됩니다. 이 방법은 기존 TQA 태스크 단순화 과정보다 더 높은 오류 수용성을 제공합니다.



### On the Relationship between Truth and Political Bias in Language Models (https://arxiv.org/abs/2409.05283)
- **What's New**: 이번 연구에서는 언어 모델 정렬(language model alignment)에서 중심적인 두 개념인 진실성(truthfulness)과 정치적 편향(political bias)의 관계를 분석하였습니다. 연구 결과, 진실성에 중점을 둔 보상 모델(reward models)을 훈련시키면 왼쪽으로 치우친 정치적 편향이 발생하는 경향이 있음이 밝혀졌습니다.

- **Technical Details**: 연구팀은 다양한 진실성 관련 데이터셋에 대한 보상 모델을 훈련하고, 결과적으로 이 모델들이 정치적 편향을 평가하였습니다. 기존의 오픈 소스 보상 모델들은 이미 비슷한 편향을 보였으며, 특히 모델 크기가 클수록 더 큰 편향을 드러내었습니다.

- **Performance Highlights**: 보상 모델 훈련 결과, 진실성 데이터셋에서 훈련된 모델이 왼쪽 편향을 보이며, 기후, 에너지, 노동 조합 관련 주제에서 편향이 특히 강하게 나타났습니다. 반면, 세금 및 사형과 관련된 주제에서는 편향이 약해지거나 심지어 반대 방향으로 나타났습니다.



### RotCAtt-TransUNet++: Novel Deep Neural Network for Sophisticated Cardiac Segmentation (https://arxiv.org/abs/2409.05280)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 연구에서는 심장 구조의 세밀한 세분화(segmentation)를 위한 새로운 아키텍처인 RotCAtt-TransUNet++를 도입했습니다. 이 아키텍처는 멀티스케일 특징 집합(feature aggregation) 및 중첩 스킵 연결(nested skip connections)을 통해 글로벌 컨텍스트 모델링을 강화하였습니다.

- **Technical Details**: RotCAtt-TransUNet++는 CNN 기반 및 Transformer 기반 접근 방식을 결합하여, 다양한 해상도와 깊이에서 4개의 특징 맵(X1, X2, X3, X4)을 사용합니다. 이 아키텍처는 회전식 주의 메커니즘(rotatory attention mechanism)을 통해 슬라이스 간 연결성을 처리하고, 채널-와이즈 크로스 주의 게이트(channel-wise cross-attention gate)를 통해 멀티스케일 정보를 통합합니다.

- **Performance Highlights**: 여러 데이터 세트를 통한 실험 결과, RotCAtt-TransUNet++는 기존 방법보다 뛰어난 성능을 보여 주었으며, 관상 동맥(coronal arteries) 및 심근(myocardium)의 거의 완벽한 주석(annotation)을 달성하였습니다. 또한, 회전식 주의 메커니즘이 세분화 정확도를 크게 향상시킴을 입증했습니다.



### BrainDecoder: Style-Based Visual Decoding of EEG Signals (https://arxiv.org/abs/2409.05279)
Comments:
          5 pages, 4 figures, 2 tables

- **What's New**: 본 논문은 EEG(뇌전도) 신호를 기반으로 시각적 자극을 복원하는 새로운 방법인 BrainDecoder를 소개합니다. 기존의 방법과 달리, 이 연구는 이미지의 스타일(색상 및 질감 등) 복원에도 중점을 둡니다.

- **Technical Details**: 이 방법은 EEG 신호를 CLIP(image 및 text) 임베딩 공간과 각각 정렬하고, 이를 통해 스타일과 의미 정보를 동시에 추출하는 'style-based' 접근 방식을 사용합니다. BrainDecoder는 LSTM 기반의 인코더 아키텍처를 이용해 EEG 신호를 효과적으로 인코딩하며, CLIP 이미지 및 텍스트 인코더와의 연결을 통해 뇌 측정 중에 인지된 시각적 자극의 스타일과 콘텐츠를 복원합니다.

- **Performance Highlights**: 정량적 및 정성적 평가 모두에서 BrainDecoder는 기존 최선의 방법들보다 더 우수한 스타일 보존과 정교한 의미 정보를 추출하며, 데이터셋인 Brain2Image에서 새로운 최첨단 성능을 달성했습니다.



### Disentangled Representations for Short-Term and Long-Term Person Re-Identification (https://arxiv.org/abs/2409.05277)
Comments:
          arXiv admin note: substantial text overlap with arXiv:1910.12003

- **What's New**: 이 논문에서는 person re-identification (reID) 문제를 다루며, 쿼리 이미지에 대한 검색을 수행합니다. 새로운 기법인 identity shuffle GAN (IS-GAN)을 제안하여, 개별 신원과 관련 없는 특징을 분리하여 학습하는 접근법을 소개합니다.

- **Technical Details**: IS-GAN은 identity shuffling 기법을 통해 신원 관련 및 비관련 특징을 분리합니다. 신원 관련 특징은 특정 개인을 정의하는 데 유용한 정보를 포함하며(예: 의상), 신원 비관련 특징은 다른 요인(예: 인간 자세)을 포함합니다. 이 방법은 보조적인 감독 신호 없이도 작동합니다.

- **Performance Highlights**: IS-GAN은 Market-1501, CUHK03, DukeMTMC-reID와 같은 표준 reID 벤치마크에서 최고의 성능을 보여주었으며, Celeb-reID 데이터셋에서 긴 기간 reID 작업에 대한 새로운 최첨단 결과를 발표합니다.



### Learning Submodular Sequencing from Samples (https://arxiv.org/abs/2409.05265)
- **What's New**: 이 논문에서는 대부분의 기존 연구에서 유틸리티 함수에 대한 접근을 가정하는 것과 달리, 표본(sample) 집합만을 주어진 상황에서 연속적(submodular) 최적화를 다룹니다. 특히, 두 단계의 균일 분포에서 추출된 폴리노미얼한 표본을 사용하여 알고리즘을 제안하고, 개별 submodular 함수의 곡률(curvature)에 의존하는 근사 비율을 달성하는 방법을 보여줍니다.

- **Technical Details**: 본 연구는 연속적(submodular) 최대화 문제에 대한 새로운 접근법을 제시합니다. 주어진 표본으로부터 순서에 따라 종속적인 함수 최적화를 수행하며, 이는 추천 시스템(recommendation systems) 및 순차적(active) 학습에 활용될 수 있습니다. 특히, 이 문제는 선택한 항목의 순서가 결과에 미치는 영향을 고려하며, 매개변수 F(π)는 고객의 구매 가능성을 해석하는 데 사용됩니다. 여기서 F(π)는 구매 가능성을 높이기 위해 제품을 순서대로 배열하려는 목표를 반영합니다.

- **Performance Highlights**: 실험을 통해 제안된 알고리즘은 제한된 데이터에도 불구하고 실질적으로 유용한 결과를 제공하며, 데이터 부족에도 불구하고 구매 가능성을 극대화하기 위한 효율적인 순서 결정을 지원함을 증명합니다. 이러한 결과는 온라인 상거래 플랫폼의 제품 순위 결정 같은 다양한 실제 응용 분야에 적용될 수 있음을 보여줍니다.



### Towards Automated Machine Learning Research (https://arxiv.org/abs/2409.05258)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 도움으로 기계 학습 연구의 점진적 발전을 자동화하는 탑다운(top-down) 접근 방식이 제시됩니다. 이 프레임워크는 새롭고 실행 가능한 구성 요소를 체계적으로 생성하고, 이를 기존 기준선과 비교하여 평가합니다.

- **Technical Details**: 이 방법은 전통적인 AutoML(Automated Machine Learning) 및 NAS(Neural Architecture Search)와는 달리 정의된 하드코딩된 기본 구성 요소에 대한 조합 검색에 의존하지 않고, LLMs에 내재된 도메인 간 지식을 활용하여 새로운 구성 요소를 제안합니다. 이를 통해 가설 생성 및 검증 과정을 효율적으로 개선하며, 성공적인 가설의 공통 패턴을 확인하는 보상 모델을 학습합니다.

- **Performance Highlights**: 이 연구는 실현 가능성 있는 구성 요소를 제안하고 평가하여, 기존 대안과 경쟁하는 성능을 달성하는 것을 목표로 합니다. 이 프레임워크는 자동화된 연구를 통해 혁신을 가속화하고 다양한 과학적 발견 작업에 적용될 수 있는 가능성을 탐구합니다.



### FedFT: Improving Communication Performance for Federated Learning with Frequency Space Transformation (https://arxiv.org/abs/2409.05242)
- **What's New**: 이 논문에서는 Federated Learning (FL) 환경에서 모델 파라미터를 효과적으로 전달할 수 있는 방법론인 FedFT를 소개합니다. FedFT는 Discrete Cosine Transform (DCT)을 사용하여 모델 파라미터를 주파수 영역(frequency space)에서 표현하며, 데이터 전송을 압축하여 커뮤니케이션 오버헤드를 줄입니다.

- **Technical Details**: FedFT는 DCT 변환을 통해 모델 파라미터를 주파수 공간에서 효율적으로 압축하고 이는 기존 FL 방법론 및 다양한 신경망 아키텍처와 호환됩니다. 이 방법론은 서버에서 모델 집합을 수행하는 동안 필요한 추가 변환을 줄여주며, 모델의 정확도를 유지합니다. FedFT는 데이터 프라이버시, 상호운용성(interoperability), 에너지 효율성 등의 문제를 해결하는 데 중요한 도구입니다.

- **Performance Highlights**: FedFT 방법론을 네 가지 데이터셋을 사용하여 평가한 결과, 모델 파라미터의 차이를 주파수 공간에서 표현할 경우 전체 모델을 주파수 공간에서 표현할 때보다 더 압축된 표현을 얻을 수 있음을 보여주었습니다. 데이터셋에 따라 클라이언트당 5%에서 30%까지 커뮤니케이션 오버헤드를 줄이는 동시에 정확도를 유지하거나 개선하는 결과를 보였습니다.



### Synthetic Tabular Data Generation for Class Imbalance and Fairness: A Comparative Study (https://arxiv.org/abs/2409.05215)
Comments:
          Accepted at the ECML PKDD 2024, 4th Workshop on Bias and Fairness in AI

- **What's New**: 이 논문은 기계 학습(ML) 모델에서 발생하는 클래스 불균형(class imbalance)과 그룹 불균형(group imbalance)의 문제를 해결하기 위해 최신 합성 데이터 생성 모델을 활용한 비교 분석을 실시합니다. 기존의 오버샘플링(over-sampling) 기법에 대한 보다 발전된 접근법을 제안하고, 이를 통해 공정성(fairness) 측면에서도 효과성을 입증합니다.

- **Technical Details**: 논문에서는 다섯 가지 합성 데이터 생성 방법을 다루며, 각각의 방법은 다음과 같습니다: SDV-GC(Probabilistic Gaussian Copula), CTGAN(Conditional Generative Adversarial Network), TVAE(Tabular Variational Autoencoder), CART(Classification and Regression Trees), SMOTE-NC(Synthetic Minority Over-sampling Technique for Nominal and Continuous). 이 방법들은 주어진 테이블 데이터에서 클래스와 그룹 불균형을 해소하기 위해 다양한 샘플링 전략과 함께 평가됩니다.

- **Performance Highlights**: 실험 결과는 네 가지 실제 테이블 데이터셋에서의 성능을 분석하여, 기계 학습의 유용성을 유지하면서 불균형을 완화하는 데 있어 생성 모델이 큰 효과를 발휘함을 보여줍니다. 연구진은 또한 다수의 보호 속성이 동시 존재하는 경우의 교차 공정성(intersectional fairness)에 대한 실험도 진행하여, 향후 연구의 가능성을 제시합니다.



### ICML Topological Deep Learning Challenge 2024: Beyond the Graph Domain (https://arxiv.org/abs/2409.05211)
Comments:
          Proceedings of the Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM) at ICML 2024

- **What's New**: 이 논문은 ICML 2024에서 주최된 제2회 Topological Deep Learning Challenge를 다루고 있습니다. 이 도전 과제는 Topological Deep Learning(TDL)와 다양한 구조적 데이터셋(예: point clouds, graphs) 간의 간극을 해소하기 위해 서로 다른 이산(topological) 도메인에서 데이터를 표현하는 문제에 중점을 두었습니다.

- **Technical Details**: 참여자들은 서로 다른 데이터 구조와 topological 도메인 간의 매핑인 topological lifting을 설계하고 구현해야 했습니다. 제출물은 point-cloud/graph, hypergraph, simplicial complex, cell complex, combinatorial complex 간의 유효한 lifting transformation을 포함해야 했습니다. 56개의 제출물 중 52개가 유효하였으며, 모든 데이터는 GitHub Action workflow를 준수해야 했습니다.

- **Performance Highlights**: 52개의 유효 제출물은 31개 팀에서 출품되었으며, 각 카테고리별로 24, 28, 25, 27개의 제출물이 있었습니다. 제출물의 평가 기준은 lifting의 올바른 구현, 코드의 가독성 및 문서화, 유닛 테스트의 강건성 등을 포함하였습니다.



### Influence-based Attributions can be Manipulated (https://arxiv.org/abs/2409.05208)
- **What's New**: 이 연구는 Influence Functions를 기반으로 한 데이터 귀속 예측을 조작할 수 있는 가능성을 제시하며, 이를 통해 적대적 상황에서의 신뢰성 문제에 대한 새로운 질문을 제기한다.

- **Technical Details**: 이 논문에서는 Influence Function 파이프라인을 세 가지 요소로 형식화 하였고, Targeted 및 Untargeted 공격을 수행할 수 있는 알고리즘을 제공한다. Targeted 공격은 특정 데이터 샘플의 influence score를 조작하는 것을 목표로 하며, 이를 위해 computationally efficient한 알고리즘을 제안한다. Untargeted 공격은 특정 샘플을 목표로 하지 않고 임의로 influence scores를 조작하고, 모델 가중치를 조정하는 것이 효과적임을 발견하였다.

- **Performance Highlights**: Targeted 공격의 성공률은 최대 94%에 달하며, 3개의 데이터셋에서 큰 정확도 저하 없이 수행되었다. Untargeted 공격의 경우, 공정성 측정에서 최대 16%의 차이를 초래하여, 모델의 공정성을 심각하게 저해하였다.



### SEF: A Method for Computing Prediction Intervals by Shifting the Error Function in Neural Networks (https://arxiv.org/abs/2409.05206)
Comments:
          The paper has been accepted at the 2024 International Conference on Computer and Applications (ICCA24), Cairo, Egypt, December 17-19, 2024. this https URL

- **What's New**: 본 논문에서는 SEF (Shifting the Error Function)라는 새로운 방법을 제안하여 신경망의 예측 불확실성을 효과적으로 정량화하는 기술을 개발했습니다. 기존의 방법들이 가진 문제점을 해결하기 위해 단일 신경망을 세 번 훈련하여 예측값과 그에 따른 상한 및 하한을 동시에 생성합니다.

- **Technical Details**: SEF 방법은 초기 신경망의 추정값에서 계산된 매개변수를 이용하여 다른 두 신경망의 손실 함수에 통합하는 방식으로 작동합니다. 이 과정은 예측 구간 (Prediction Interval, PI)을 생성하며, γ (신뢰수준) 조건을 만족하는 최적의 상한과 하한을 제공합니다. 이 방법은 복잡한 아키텍처나 복잡한 손실 함수 없이도 쉽게 적용 가능하고, 비용적인 측면에서도 효율적입니다.

- **Performance Highlights**: 실험 결과, SEF 방법은 두 개의 합성 데이터셋에서 PI3NN 및 PIVEN 방법과 비교하여 성공적인 예측 구간 생성을 보였으며, 다른 방법들에 비해 시간 및 자원 소모가 적다는 점에서 강력한 성능을 입증했습니다.



### A Survey on Mixup Augmentations and Beyond (https://arxiv.org/abs/2409.05202)
Comments:
          Preprint V1 with 27 pages main text. Online project at this https URL

- **What's New**: 딥러닝(Deep Learning) 분야에서 Mixup 방법이 데이터 증강(Data Augmentation) 기법으로 주목받고 있으며, 본 논문에서는 Mixup의 다양한 적용 사례와 이론적 배경을 포괄적으로 정리하였습니다.

- **Technical Details**: Mixup은 두 개의 샘플(sample)과 해당 레이블(label)을 선형 조합하여 새로운 가상 샘플을 생성하는 방식입니다. 이 방법은 대량의 훈련 데이터를 요구하지 않으면서도 다양한 도메인에 효과적으로 적용 가능합니다.

- **Performance Highlights**: Mixup은 기존의 데이터 증강 기법들과 비교해 모델의 일반화 성능을 향상시키며, Supervised Learning(SL), Self-Supervised Learning(SSL), Semi-Supervised Learning(Semi-SL) 등 다양한 훈련 패러다임에 성공적으로 적용되고 있습니다.



### Seemingly Plausible Distractors in Multi-Hop Reasoning: Are Large Language Models Attentive Readers? (https://arxiv.org/abs/2409.05197)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문은 최신 대형 언어 모델(LLMs)의 다중 홉 추론(multi-hop reasoning) 능력을 조사합니다. 기존의 다중 홉 추론 벤치마크에서 발견된 단순화된 단서가 모델이 추론 요구 사항을 회피하게 한다는 우려를 반영하여, LLM이 이를 이용할 가능성이 있는지 살펴보았습니다.

- **Technical Details**: 연구를 통해, LLM들은 멀티 홉 추론을 수행해야 하는 요구를 보다 미묘한 방식으로 회피하는 경향이 있다는 것을 발견했습니다. 저자들은 헷갈리게 하는 추론 경로가 LLMs에 큰 도전 과제가 된다는 점을 강조하며, 오류가 있는 답변으로 이어지는 그럴듯한 다중 홉 추론 체인을 생성하는 새로운 벤치마크를 제안합니다.

- **Performance Highlights**: 최신 LLM들을 평가한 결과, 그들은 그럴듯한 대안을 제시받았을 때 F1 점수가 최대 45% 감소하는 등 다중 홉 추론 수행에 영향받는 것으로 나타났습니다. 이 분석을 통해, LLM이 오해를 일으키는 어휘적 단서(lexical cues)를 무시하는 경향이 있지만, 헷갈리게 하는 추론 경로는 상당한 도전을 제시한다는 것을 알 수 있었습니다.



### Insights from Benchmarking Frontier Language Models on Web App Code Generation (https://arxiv.org/abs/2409.05177)
- **What's New**: 본 논문은 16개의 최신 대형 언어 모델(LLMs)을 WebApp1K 벤치마크에서 평가한 결과를 보고합니다. 이 벤치마크는 LLM이 웹 애플리케이션 코드를 생성하는 능력을 측정하기 위해 설계되었습니다. 모든 모델은 유사한 지식을 가지고 있지만, 실수 빈도에 따라 성능이 차별화되는 것으로 나타났습니다.

- **Technical Details**: 모델의 성능을 평가하기 위해 각 모델이 1000개의 문제를 해결하는 데 10회를 시도하였으며, 총 160,000개 솔루션을 생성하였습니다. 172개의 솔루션에서 구문 오류가 발견되었고, 이는 0.1%의 실패율을 나타냅니다. WebApp1K의 난이도는 문제별 실패율에 따라 분석되었으며, 모델별 LOC (lines of code) 분포를 통해 코드 생성의 복잡성을 추가적으로 분석하였습니다.

- **Performance Highlights**: 주요 발견 중 하나는 성공적인 솔루션의 LOC 분포가 항상 실패한 솔루션보다 더 복잡하다는 것이며, 이는 올바른 코드를 작성하는 데 내재된 복잡성을 의미합니다. 또한, 상위 성능 모델일수록 LOC 분포에서 여러 개의 피크를 보여주며, 이는 다양한 코드 길이를 생성할 수 있는 능력을 시사합니다.



### OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs (https://arxiv.org/abs/2409.05152)
Comments:
          Work in progress; code is available at this https URL

- **What's New**: 이번 논문에서는 기존의 Large Language Models (LLMs)에서 발생하는 생성(generation)과 검색(retrieval) 작업의 통합 문제를 해결하기 위해 새로운 One-pass Generation and retrieval 프레임워크인 OneGen을 소개합니다. 이 프레임워크는 생성과 검색을 통합하여 동시에 처리할 수 있도록 설계되었습니다.

- **Technical Details**: OneGen 프레임워크는 autoregressively 생성된 retrieval tokens를 포함하여 전통적인 생성과 검색의 훈련 방식을 결합합니다. 이를 통해 단일 LLM이 통합된 forward pass 내에서 두 가지 작업을 동시에 처리할 수 있게 됩니다. 우리는 RAG와 Entity Linking이라는 두 가지 복합 작업에서 OneGen의 pluggability, 효과성, 효율성을 검증하기 위한 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 생성과 검색을 같은 맥락에서 통합하면 LLMs의 생성 능력을 유지하면서도 검색 성능이 향상됨을 확인했습니다. 또한, OneGen은 LLMs가 생성 중에 벡터 검색(vector retrieval)을 수행할 수 있도록 최초로 가능하게 하였습니다.



### QuantFactor REINFORCE: Mining Steady Formulaic Alpha Factors with Variance-bounded REINFORCE (https://arxiv.org/abs/2409.05144)
Comments:
          15 pages, 7 figures

- **What's New**: 본 논문에서는 formulaic alpha factors(포뮬러 알파 팩터) 발굴을 위한 새로운 강화 학습 알고리즘인 QuantFactor REINFORCE(QFR)를 제안합니다. 이 방법은 기존의 Proximal Policy Optimization(PPO) 방식 대신 사용되며, 높은 변동성과 수렴 속도의 문제를 해결합니다.

- **Technical Details**: QFR 알고리즘은 deterministic state transition(결정론적 상태 전환)을 특징으로 하며, Dirac distribution(디락 분포)에 따라 설계된 Markov Decision Process(MDP)를 이용합니다. 새로운 기반(baseline)을 도입하여 REINFORCE의 높은 변동성을 감소시키고, 정보 비율(Information Ratio, IR)을 보상 형성 메커니즘(reward shaping mechanism)으로 도입하여 시장 변동성에 잘 적응하는 안정적인 alpha factors 생성을 유도합니다.

- **Performance Highlights**: QFR 알고리즘은 여러 실질 자산 데이터셋에서 실험을 수행한 결과, 기존의 방법에 비해 자산 수익률과의 상관관계를 3.83% 개선하였으며, 초과 수익을 효과적으로 창출하는 더 강력한 능력을 보여 주었습니다.



### EdaCSC: Two Easy Data Augmentation Methods for Chinese Spelling Correction (https://arxiv.org/abs/2409.05105)
Comments:
          18 pages, 2 figures

- **What's New**: 이번 연구에서는 Chinese Spelling Correction(CSC)에서 발생하는 문제를 해결하기 위해 두 가지 데이터 증가(data augmentation) 방법, 즉 긴 문장을 짧은 문장으로 나누거나 여러 개의 오타를 포함하고 있는 문장의 오타를 줄이는 방법을 제안합니다.

- **Technical Details**: 제안된 방법인 EdaCSC는 (1) 데이터 세트를 두 가지 데이터 증강 방법으로 증강하고, (2) 최적의 모델을 선택하기 위해 다양한 훈련 절차를 적용하는 구조를 가지고 있습니다. 첫 번째 방법은 문장을 구두점 기준으로 나누어 모델의 과도 수정(overcorrection) 경향을 줄이고, 두 번째 방법은 다수의 오타를 포함한 문장의 오타를 줄여 성능 저하 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SIGHAN14와 SIGHAN15 데이터 세트에서 모든 비교 방법들을 초월하여 최우수 성능을 달성하였으며, SIGHAN13 데이터 세트에서도 두 번째로 우수한 성능을 기록하였습니다.



### Adaptive $k$-nearest neighbor classifier based on the local estimation of the shape operator (https://arxiv.org/abs/2409.05084)
Comments:
          18 pages, 4 figures

- **What's New**: 본 논문에서는 k-최근접 이웃(k-NN) 알고리즘의 새로운 적응형 버전인 kK-NN 알고리즘을 소개합니다. 이 알고리즘은 샘플의 지역 곡률을 기반으로 이웃 크기를 적응적으로 정의합니다. 낮은 곡률을 가진 포인트는 보다 큰 이웃을 가질 수 있으며, 높은 곡률을 가진 포인트는 더 작은 이웃을 가집니다.

- **Technical Details**: kK-NN 알고리즘은 로컬 모양 연산자(local shape operator)를 사용하여 로컬 공분산 행렬(local covariance matrix)과 로컬 헤시안 행렬(local Hessian matrix)으로 Gaussian 곡률을 추정합니다. 이 알고리즘은 모든 샘플에 대해 같은 k 값을 사용하는 기존의 k-NN과 달리, 각 샘플에 대해 동적으로 k 값을 조정하여 결정 경계(decision boundary)의 복잡성을 제어합니다.

- **Performance Highlights**: 30개의 실제 데이터셋에 대한 실험 결과, kK-NN 알고리즘은 기존의 k-NN 및 다른 적응형 k-NN 알고리즘에 비해 보다 우수한 밸런스 정확도(balanced accuracy)를 나타냅니다. 이는 특히 제한된 훈련 데이터 상황에서 두드러지며, kK-NN이 적은 데이터로 더 많은 판별 함수를 학습할 수 있음을 시사합니다.



### PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions (https://arxiv.org/abs/2409.05076)
Comments:
          Accepted by ACM Multimedia 2024 BNI track (Oral)

- **What's New**: 이 연구에서는 LVLMs의 적대적 예제를 탐지하기 위한 새로운 방법인 PIP를 소개합니다. 이 방법은 무관한 프로브 질문의 주의 패턴을 활용하여 적대적 예제를 효과적으로 구별합니다.

- **Technical Details**: PIP는 무관한 프로브 질문(예: '시계가 있나요?')의 주의 패턴을 사용하여 클린 이미지와 적대적 이미지를 구분하는 비전통적인 방법입니다. 이 방법은 검증할 이미지와 프로브 질문에 대해 추가적인 추론을 한 번만 수행해야 하며, SVM(Classifier)와 결합하여 98% 이상의 재현율(recall)과 90% 이상의 정밀도(precision)를 달성합니다.

- **Performance Highlights**: PIP는 블랙박스 공격과 열린 데이터셋 시나리오에서도 효율적으로 적대적 예제를 탐지하며, 클린 예제와 적대적 예제를 무관한 질문에 대한 주의 패턴으로 구별합니다. 이 방법은 LVLM에 대한 적대적 공격을 탐지하는 최초의 시도로, LVLM 내에서의 더 깊은 이해를 위한 통찰력을 제공합니다.



### Deep Generic Representations for Domain-Generalized Anomalous Sound Detection (https://arxiv.org/abs/2409.05035)
- **What's New**: 본 논문에서는 기존의 방법들과 달리 레이블이 없는 데이터만으로도 신뢰할 수 있는 이상 소음 탐지(anomalous sound detection, ASD) 시스템을 개발할 수 있는 새로운 접근 방식인 GenRep을 소개합니다.

- **Technical Details**: GenRep은 BEATs에서 추출한 일반적인 특징 표현을 활용하며, k-nearest neighbors (kNN) 기법을 결합하여 도메인 일반화된 ASD를 구현합니다. 이는 세밀한 조정을 필요로 하지 않습니다. MemMixup을 통해 소스 샘플을 사용하여 목표 메모리 뱅크(target memory bank)를 증강하고, 도메인 균형을 맞추기 위해 도메인 정규화(Domain Normalization, DN)를 적용합니다.

- **Performance Highlights**: DCASE2023T2 평가 세트에서 73.79%의 공식 점수를 기록하며, 기존의 Outlier-Exposure(OE) 기반 접근방식을 초월하는 성능을 보였습니다. 또한, 데이터가 제한된 상황에서도 강력한 성능을 유지하였습니다.



### A Survey on Diffusion Models for Recommender Systems (https://arxiv.org/abs/2409.05033)
Comments:
          Under Review

- **What's New**: 본 논문은 추천 시스템을 위한 확산 모델(diffusion models, DMs)에 대한 최초의 포괄적인 서베이를 제시하고 다양한 추천 파이프라인 관점에서 이 모델들이 어떻게 활용되는지를 조망합니다.

- **Technical Details**: 이 논문은 추천 시스템에서의 확산 모델을 세 가지 주요 도메인으로 체계적으로 분류합니다: (1) 데이터 공학 및 인코딩을 위한 확산 - 데이터 증강 및 표현 향상; (2) 추천 모델로서의 확산 - 사용자 선호도를 직접 추정하고 아이템을 순위 매기는 데 사용; (3) 콘텐츠 프리젠테이션을 위한 확산 - 패션 및 광고 창작물과 같은 개인화된 콘텐츠 생성.

- **Performance Highlights**: 확산 모델은 복잡한 데이터 분포를 캡처하고 사용자 선호도에 맞춘 고품질의 다양한 샘플을 생성하는 데 강력한 능력을 보이며, 추천 시스템의 성능을 크게 향상시키는데 기여합니다.



### Sequential Recommendation via Adaptive Robust Attention with Multi-dimensional Embeddings (https://arxiv.org/abs/2409.05022)
- **What's New**: 이 논문은 sequential recommendation 모델의 정확도를 높이기 위해 mix-attention 메커니즘과 layer-wise noise injection (LNI) 정규화를 도입한 adaptive robust sequential recommendation framework (ADRRec) 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 multi-dimensional kernel embedding을 통해 사용자의 행동 패턴을 캡처하고, 절대 및 상대적 mix-attention 메커니즘을 통해 각 사용자 행동의 고유 패턴을 학습합니다. noise injection regularization (NIR)을 통해 모델의 강인성과 일반화를 강화합니다.

- **Performance Highlights**: 네 개의 유명한 데이터셋을 이용한 실험 결과 제안된 ADRRec 모델이 기존 self-attention 아키텍처를 능가하는 성능을 보임을 입증하였습니다.



### Audio-Guided Fusion Techniques for Multimodal Emotion Analysis (https://arxiv.org/abs/2409.05007)
- **What's New**: 이번 논문에서는 MER2024의 반전이 없는 학습(semisupervised learning) 분야에서 효과적인 솔루션을 제안합니다. 클립 비전 모델(CLIP-vit-large) 및 Baichuan-13B 모델을 레이블이 붙은 데이터를 활용해 조정하고, 오디오를 기반으로 한 변환기(Audio-Guided Transformer, AGT) 결합 기법을 통해 전 채널 및 내 채널 정보를 효과적으로 융합하는 방법을 제시했습니다. 또한, 자가 지도 학습을 사용하여 높은 신뢰도를 가진 레이블 없는 데이터를 파라미터로 사용하고, 불균형한 데이터 분포를 해결하기 위해 사전 지식 기반 투표 메커니즘을 도입했습니다.

- **Technical Details**: 이 연구에서는 감정 분류 작업을 위한 피쳐 추출기(feature extractor)로 CLIP-vit-large 및 Baichuan-13B를 세밀하게 조정하고, Hubert-large를 활용하는 AGT 융합 메커니즘을 제안합니다. 또한, 신뢰도가 높은 소프트 의사 레이블을 선택하여 자가 감독 학습을 반복적으로 적용했으며, 블랙 박스 프로빙을 통해 학습 및 테스트 세트 간의 불균형한 데이터 분포를 확인했습니다. 특히, 오디오 모달리티의 기여가 시각 및 텍스트 모달리티에 비해 더 중요하다는 점을 강조합니다.

- **Performance Highlights**: 해당 연구는 MER-SEMI 트랙에서 3위를 기록하며 제안한 전략의 효과성을 입증하였습니다. 각 모달리티의 감정 정보를 최대한 보존하면서도 불필요한 정보의 간섭을 최소화하는 방법으로 감정 인식을 향상시키는 데 성공했습니다.



### A Pair Programming Framework for Code Generation via Multi-Plan Exploration and Feedback-Driven Refinemen (https://arxiv.org/abs/2409.05001)
Comments:
          Accepted in the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE 2024)

- **What's New**: 이 논문에서는 기존의 LLM 기반 코드 생성 방법의 한계를 극복하기 위해 PairCoder라는 새로운 프레임워크를 제안합니다. PairCoder는 두 개의 협동적인 LLM 에이전트, 즉 고수준 계획을 담당하는 Navigator와 구체적인 구현을 담당하는 Driver로 구성됩니다.

- **Technical Details**: PairCoder는 스페인 소프트웨어 공학에서의 페어 프로그래밍 관행을 기반으로 하여 설계되었습니다. Navigator는 여러 가능한 해결책을 제안하고, 실행 피드백에 따라 최적의 계획을 선택하여 Driver에게 지시합니다. Driver는 Navigator의 방향에 따라 코드를 생성하고 테스트합니다.

- **Performance Highlights**: 실험 결과에 따르면 PairCoder는 여러 코드 생성 벤치마크에서 기존 LLM 접근 방식보다 상대적으로 12.00%에서 162.43%까지 높은 정확도를 달성했습니다. 이 성능은 GPT-3.5-Turbo 및 DeepSeek-Coder와 같은 다양한 LLM에서 확인되었습니다.



### Enhancing Convolutional Neural Networks with Higher-Order Numerical Difference Methods (https://arxiv.org/abs/2409.04977)
- **What's New**: 딥 러닝 기술의 발전과 함께 Convolutional Neural Networks (CNNs)가 실제 문제 해결에 큰 도움을 주고 있습니다. 본 논문은 모델 크기와 환경적 제약을 극복하면서 ResNet의 성능을 향상시키는 새로운 stacking scheme을 제안합니다.

- **Technical Details**: 이 연구는 일반적인 미분 방정식을 이산화(discretization)하는 방식으로 많은 CNN 구조를 설명합니다. 저자들은 선형 다단계 수치 차분 방법(linear multi-step numerical difference methods)을 활용하여 이론적으로 지지된 딥 네트워크 구조를 설계할 수 있음을 강조하고 있습니다. 또한, 제안된 stacking scheme은 기존의 Runge-Kutta 방법과 비교하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 stacking scheme이 ResNet 및 HO-ResNet과 같은 기존 stacking scheme에 비해 우수한 성능을 보이며, 다른 유형의 신경망에도 확장 가능성을 가지고 있음을 보여줍니다.



### HYDRA: Hybrid Data Multiplexing and Run-time Layer Configurable DNN Accelerator (https://arxiv.org/abs/2409.04976)
- **What's New**: 이 논문은 HYDRA라는 새로운 하이브리드 데이터 멀티플렉싱 및 실행 가능한 DNN(Deep Neural Network) 가속기를 제안합니다. 이 아키텍처는 레이어 재사용과 Fused-Multiply-Accumulate(FMA)의 향상된 기능을 활용하여 에너지 효율적인 DNN 실행을 가능하게 합니다.

- **Technical Details**: HYDRA 아키텍처는 레이어-멀티플렉스(Layer-Multiplexed) 구성으로 다양한 DNN의 깊이를 정의하여 재구성 가능한 아키텍처를 제공합니다. 이 구조는 각 레이어에서 단일 활성화 함수(AFF)를 재사용하고, PISO(Parallel-In-Serial-Out) 구성을 통해 데이터 흐름을 병렬화합니다. 기존 DNN 수치 구성의 64:32:32:10을 사용하여 성능을 벤치마킹하고 평가하였습니다.

- **Performance Highlights**: 제안된 HYDRA 아키텍처는 90% 이상의 전력 소모 감소 및 자원 활용 개선을 달성하였으며, 35.21 TOPS(Trillions of Operations Per Second)의 성능을 기록했습니다. 이 아키텍처는 자원이 제한된 엣지 기기에서 최적의 DNN 계산을 지원하며, 국가 수준의 가장 진보된 기술과 비교하여 상당한 효율성을 보여줍니다.



### Soft Actor-Critic with Beta Policy via Implicit Reparameterization Gradients (https://arxiv.org/abs/2409.04971)
Comments:
          10 pages

- **What's New**: 최근 딥 강화 학습의 발전은 복잡한 작업에서 인상적인 결과를 달성했지만, 샘플 효율성(sample efficiency) 부족이 현실 세계에서의 배치를 제한하는 주요 장애물로 남아 있습니다. 본 논문에서는 Soft Actor-Critic (SAC)을 사용하여 beta 정책을 적용함으로써 이 문제를 해결하고, 더 나아가 기존의 정상 정책(normal policy)보다 나은 성능을 보이는 것을 보여줍니다.

- **Technical Details**: SAC는 확률적 정책 최적화와 오프 정책 학습을 결합하여 상태의 최대 엔트로피(maximum entropy) 원칙을 따릅니다. 하지만, 재매개변수화 기법(reparameterization trick)을 이용할 수 없는 분포는 SAC의 적용에 제한이 있으며, 이를 극복하기 위해 암묵적 재매개변수화(implicit reparameterization) 기법을 사용하여 beta 정책을 훈련합니다. 이러한 방식은 고차원 연속 제어 문제에서 actor-critic 알고리즘의 수렴 속도를 개선합니다.

- **Performance Highlights**: 실험 결과, beta 정책이 정상 정책을 능가하는 성능을 보이며, SAC의 일반적인 선택지인 squashed normal policy와 유사한 성능을 발휘했습니다. 이는 beta 정책이 샘플 효율성을 높이면서도, 사전 지식을 효과적으로 주입할 수 있음을 보여줍니다.



### Evaluation of Google Translate for Mandarin Chinese translation using sentiment and semantic analysis (https://arxiv.org/abs/2409.04964)
- **What's New**: 본 연구는 기계 번역 모델의 자동 평가를 인간 전문가와 함께 감정 및 의미 분석을 통해 수행합니다.经典文本인 '아 Q의 진짜 이야기'를 선택해, 구글 번역과 전문가 번역 간의 차이를 분석하여 새로운 통찰력을 제공합니다.

- **Technical Details**: 기계 번역(Machine Translation)과 자연어 처리(Natural Language Processing, NLP)에 대한 이론적 배경을 정립하며, 기계 번역 성능 비교를 위한 정량적 접근 방식을 사용합니다. 연구에서는 구글 번역과 전문가의 번역 결과를 감정 분석(Sentiment Analysis) 및 의미 분석(Semantic Analysis)으로 평가합니다.

- **Performance Highlights**: 구글 번역은 역사적 지식 및 맥락적 중요성의 결여로 인해 중국어의 특정 단어나 구절을 번역하는 데 어려움을 겪으며, 전문가 번역과 비교할 때 정확성과 감정 전달 측면에서 차이를 보입니다.



### DDNet: Deformable Convolution and Dense FPN for Surface Defect Detection in Recycled Books (https://arxiv.org/abs/2409.04958)
- **What's New**: 이번 연구에서는 재활용 및 순환 도서의 표면 결함 탐지 정확도를 향상시키기 위해 DDNet이라는 혁신적인 모델을 제안합니다. 이 모델은 변형된 합성곱 연산자(DC)와 밀접하게 연결된 특징 피라미드 네트워크(DFPN)를 기반으로 합니다.

- **Technical Details**: DDNet의 주요 구성 요소로는 특징 추출 네트워크, DC 모듈, DFPN 모듈, 분리된 검출 헤드가 있습니다. DC 모듈은 대상의 윤곽에 맞게 합성곱 그리드를 동적으로 조정하여 미세한 형태 변화를 포착하고, DFPN은 밀집 스킵 연결을 활용하여 다중 해상도의 고충실도 특징 맵을 생성합니다.

- **Performance Highlights**: DDNet은 독자적인 데이터셋에서 46.7%의 mAP 값을 기록하며, 기존 모델보다 14.2% 향상된 결과를 보여줍니다. 이는 재활용 도서에서의 표면 결함 탐지에 있어 뛰어난 성능을 나타냅니다.



### Evaluating Neural Networks Architectures for Spring Reverb Modelling (https://arxiv.org/abs/2409.04953)
Comments:
          8 pages, 7 figures, 2 tables

- **What's New**: 본 연구는 스프링 리버브 효과를 효과적으로 복제하기 위한 5가지 신경망 아키텍처를 비교합니다.

- **Technical Details**: 연구에서는 16kHz와 48kHz의 샘플링 레이트로 두 개의 데이터셋에서 신경망 모델의 효과를 평가하며, Convolutional과 Recurrent 모델을 포함한 다양한 신경망 아키텍처의 성능을 분석합니다.

- **Performance Highlights**: 신경 오디오 아키텍처가 제공하는 파라메트릭 제어를 통해 현재의 블랙박스 모델링 기술의 경계를 넘기 위한 진전을 이루는 것을 목표로 하고 있습니다.



### Attention-Based Efficient Breath Sound Removal in Studio Audio Recordings (https://arxiv.org/abs/2409.04949)
- **What's New**: 본 연구에서는 음성 녹음에서 비음성 발화 소음, 특히 호흡 소음을 자동으로 감지하고 제거하는 혁신적이고 파라미터 효율적인 모델을 제안합니다. 이 모델은 attention U-Net 아키텍처를 기반으로 하여, 기존 방법의 한계를 극복하고 더 나은 정확성을 제공합니다.

- **Technical Details**: 우리 연구의 중심은 attention 메커니즘을 강조한 특별한 U-Net 모델입니다. 이 모델은 Short Time Fourier Transform (STFT)으로부터 생성된 스펙트로그램을 입력으로 사용합니다. 모델은 3872x2048x1의 입력 형태를 가지며, 16개의 필터와 배치 정규화를 포함하고, 20%의 드롭아웃 비율을 사용합니다. 훈련을 위해 Mean Absolute Error (MAE) 및 음성을 보존하는 추가 항목을 포함한 맞춤형 손실 함수를 설계했습니다.

- **Performance Highlights**: 우리 모델은 1.9M의 파라미터와 3.2시간의 훈련 시간으로 최고 성능 모델보다 훨씬 효율적입니다. 이전 모델과 동일한 출력을 생성하면서도 정확도가 크게 향상되었습니다. 이로 인해 음향 엔지니어가 귀중한 시간을 절약하고 오디오 생산의 품질 및 일관성을 높일 수 있습니다.



### Maximizing Relation Extraction Potential: A Data-Centric Study to Unveil Challenges and Opportunities (https://arxiv.org/abs/2409.04934)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 복잡한 관계 추출에 대한 데이터 중심 성능 분석을 최초로 수행한 연구로, 15개 최신 신경망 기반 관계 추출 알고리즘과 7개의 대규모 데이터셋을 사용하여 현재 알고리즘의 한계를 조명합니다.

- **Technical Details**: 이 연구에서는 현대의 관계 추출 알고리즘들이 복잡한 데이터 및 관계 특징에 견디지 못한다는 것을 입증하며, 중요한 이슈들로는 맥락 모호성(contextual ambiguity), 상관 관계(correlating relations), 긴 꼬리 데이터(long-tail data) 및 세분화된 관계 분포(fine-grained relation distributions)가 included됩니다.

- **Performance Highlights**: 이 논문은 관계 추출 알고리즘의 성능 격차를 강조하기 위해 15개의 알고리즘을 포괄적으로 비교하고, 현재의 상치 및 향후 방향에 대한 세부 논의를 제공합니다. 또한, 실천을 위한 데이터셋과 알고리즘 구현코드를 github에 제공하여 참조할 수 있게 하였습니다.



### Nearest Neighbor CCP-Based Molecular Sequence Analysis (https://arxiv.org/abs/2409.04922)
- **What's New**: 이 연구에서는 Nearest Neighbor Correlated Clustering and Projection (CCP-NN) 기법을 제안하여 고차원 분자의 서열 데이터를 효율적으로 전처리하고, 관련 분자 서열을 그룹화하여 대표적인 슈퍼서열을 생성합니다.

- **Technical Details**: CCP-NN 기법은 기존의 Correlated Clustering and Projection (CCP) 방법의 개선 버전으로, Nearest Neighbor (NN) 검색 기법을 사용하여 데이터의 밀도 맵을 추정하고 상관 관계를 계산합니다. 이 알고리즘은 데이터 포인트 간의 거리를 계산하는 대신 최근접 이웃을 통해 고차원 데이터의 차원 축소 및 특징 선택을 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, CCP-NN 방법이 기존의 CCP 방법보다 분류 작업의 정확성을 크게 향상시켰으며, 계산 런타임에서도 유의미한 개선을 보였습니다.



### MoistNet: Machine Vision-based Deep Learning Models for Wood Chip Moisture Content Measuremen (https://arxiv.org/abs/2409.04920)
- **What's New**: 이번 연구에서는 깊은 학습(deep learning)과 기계 비전(machine vision) 기술을 활용하여 나무 칩의 RGB 이미지에서 수분 함량(moisture content) 클래스를 예측하는 방법을 제안합니다. 기존의 시간 소모적이고 파괴적인 전통적인 방법을 대체할 수 있는 즉각적이고 비파괴적인 접근 방법을 제시합니다.

- **Technical Details**: 연구팀은 1,600개의 RGB 나무 칩 이미지를 포함한 대규모 이미지 데이터 세트를 수집하고, 오븐 건조 기법의 결과를 바탕으로 정답 레이블을 달았습니다. 두 가지 고성능 신경망인 MoistNetLite와 MoistNetMax를 개발하였으며, Neural Architecture Search (NAS)와 하이퍼파라미터 최적화(hyperparameter optimization)를 이용해 최적의 성능을 추구했습니다. 이 모델들은 기존의 최첨단 깊은 학습 모델들과 비교 평가되었습니다.

- **Performance Highlights**: MoistNetLite는 87%의 정확도로 계산 비용이 적고, MoistNetMax는 91%의 정확도로 수분 함량 클래스를 훌륭하게 예측하였습니다. 개선된 정확성과 빠른 예측 속도를 갖춘 MoistNet 모델들은 나무 칩 가공 산업에 큰 가능성을 보여줍니다.



### Activation Function Optimization Scheme for Image Classification (https://arxiv.org/abs/2409.04915)
- **What's New**: 이 연구에서는 심층 신경망에서의 활성화 함수의 성능을 최적화하기 위한 진화적 접근법을 제안하며, 기존의 최첨단 활성화 함수들을 넘어서는 새로운 함수를 개발했습니다.

- **Technical Details**: 우리는 Exponential Error Linear Unit (EELU)라는 고성능 활성화 함수 시리즈를 개발하였으며, ResNet50, AlexNet, VGG16, MobileNet, Compact Convolutional Transformer와 같은 다섯 가지 최첨단 신경망 아키텍처 및 CIFAR10, Imagenette, MNIST, Fashion MNIST, Beans, Colorectal Histology, CottonWeedID15, TinyImageNet을 포함한 여덟 가지 표준 데이터셋에 대해 평가했습니다.

- **Performance Highlights**: 최적화된 활성화 함수가 28개의 다양한 케이스에서 92.8%의 경우 기존 표준 함수보다 뛰어난 성능을 보여주었으며, 최적화 스킴을 통해 생성된 활성화 함수 중 $-x\cdot erf(e^{-x})$가 이미지 분류에 가장 우수한 결과를 보였습니다.



### Efficient Training of Transformers for Molecule Property Prediction on Small-scale Datasets (https://arxiv.org/abs/2409.04909)
- **What's New**: 본 논문은 Self Attention을 활용하여 저데이터 환경에서도 우수한 성능을 발휘하는 GPS Transformer 아키텍처를 제안합니다.

- **Technical Details**: 제안된 모델은 BBBP 데이터셋을 사용하여 혈뇌장벽(BBB) 투과성을 예측하는 작업에서 기존 모델보다 우수한 성능을 기록했으며, ROC-AUC 점수는 78.8%로 기존 최첨단을 5.5% 초과했습니다.

- **Performance Highlights**: 제안된 접근법은 GPS Transformer와 표준 Self Attention을 결합하여 다른 주의 메커니즘과 결합한 GPS Transformer보다 뛰어난 성능을 발휘했습니다.



### Reinforcement Learning-Based Adaptive Load Balancing for Dynamic Cloud Environments (https://arxiv.org/abs/2409.04896)
Comments:
          Length: 6 pages (including references) Figures: 3 figures Submission Type: Conference paper Keywords: Reinforcement Learning, Load Balancing, Cloud Computing, Adaptive Algorithms, AI-driven Load Management

- **What's New**: 본 논문에서는 클라우드 컴퓨팅 환경을 위한 새로운 적응형 로드 밸런싱 프레임워크를 제안합니다. 이 프레임워크는 강화 학습(Reinforcement Learning, RL)을 기반으로 하여 실시간 시스템 성능을 관찰하고 트래픽 패턴 및 리소스 가용성을 바탕으로 작업을 분배합니다.

- **Technical Details**: 제안된 프레임워크는 RL 알고리즘을 활용하여 서버 응답 시간, CPU 활용도, 네트워크 처리량과 같은 실시간 메트릭을 모니터링하고, 이를 기반으로 서버 간 작업을 효율적으로 분배합니다. 전통적인 로드 밸런싱 방식과의 차별점은 시스템이 동적으로 작업 부하와 인프라의 변화에 적응할 수 있다는 점입니다.

- **Performance Highlights**: 실험 결과, 제안된 RL 기반 로드 밸런서가 응답 시간, 리소스 활용도, 적응성 면에서 기존의 알고리즘들보다 우수한 성과를 보였습니다. 이러한 결과는 AI 주도의 솔루션이 클라우드 인프라의 효율성과 확장성을 향상시킬 잠재력을 갖추고 있음을 시사합니다.



### Learning to Open and Traverse Doors with a Legged Manipulator (https://arxiv.org/abs/2409.04882)
- **What's New**: 이번 논문에서는 레그드 매니퓰레이터(robot manipulator)가 문을 자동으로 열고 통과하는 새로운 학습 기반 제어 정책을 제안합니다. 이 정책은 도어의 특성을 추정하고 다양한 도어 형태에 적응하는 능력을 갖추고 있습니다.

- **Technical Details**: 제어 정책은 teacher-student 접근 방식을 통해 훈련되며, 강화 학습(Reinforcement Learning, RL)과 도메인 랜덤화(domain randomization) 기법을 활용하여 다양하게 변하는 도어의 특성을 인식합니다. 이 정책은 푸시(push)와 풀(pull) 도어 모두를 거치고 열 수 있는 단일 제어 정책으로 설계되었습니다.

- **Performance Highlights**: ANYmal 레그드 로봇에 정책을 배포했을 때, 실험 환경에서 반복 테스트 중 95.0%의 성공률을 기록했습니다. 다양한 도어와 방해 요소에 대한 정책의 효과성과 강건성을 검증하는 추가 실험도 수행되었습니다.



### Towards identifying Source credibility on Information Leakage in Digital Gadget Mark (https://arxiv.org/abs/2409.04880)
- **What's New**: 소셜 미디어에서의 정보 공유가 증가하면서, 스마트폰 및 디지털 기기에 관한 '정보 유출'의 신뢰성 검토가 중요해졌습니다. 본 연구에서는 54,495개의 유출 기사와 공식 보도 자료의 헤드라인을 수집하고 분석함으로써, 정보 유출의 신뢰도를 평가할 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구에서는 커스텀 NER (Named Entity Recognition) 모델을 훈련시켜 스마트폰 이름의 변화를 82.14%의 정확도로 포착하고, 허위 및 진짜 스마트폰 유출 게시물 수에 기반한 신뢰성 점수 메트릭을 제안합니다.

- **Performance Highlights**: 스마트폰 유출과 관련된 웹 블로그의 트래픽이 공식 보도 자료보다 높은 것을 확인했으며, 이는 유출 정보의 블로그가 소비자에게 더 큰 영향을 미친다는 것을 보여줍니다.



### FedModule: A Modular Federated Learning Framework (https://arxiv.org/abs/2409.04849)
- **What's New**: 이 논문은 다양한 Federated Learning (FL) 패러다임을 지원하는 유연하고 확장 가능한 FL 실험 프레임워크인 FedModule을 소개합니다. FedModule은 '하나의 코드, 모든 시나리오' 원칙을 준수하며 모듈 방식으로 FL 프로세스를 개별 구성 요소로 나누어 서로 다른 패러다임을 원활하게 통합할 수 있는 기능을 제공합니다.

- **Technical Details**: FedModule은 동기식, 비동기식 및 개인화된 Federated Learning을 지원하며, 20개 이상의 알고리즘을 구현했습니다. 프레임워크는 선형, 스레드, 프로세스 기반 및 분산 실행 모드를 포함하여 사용자가 실험 필요에 맞게 설정을 조정할 수 있도록 여러 실행 모드를 제공합니다. FedModule은 메시지 큐를 통해 서버와 클라이언트 간의 통신을 분리하여 구성 요소의 유연성을 높이고, 다양한 모듈을 통해 새로운 FL 패러다임을 지원하도록 설계되었습니다.

- **Performance Highlights**: FedModule은 TensorFlow Federated, PySyft, Flower 및 FLGo와 같은 기존 FL 툴킷과 비교 평가를 통해 뛰어난 확장성, 유연성 및 포괄적인 벤치마크 지원을 입증하였습니다. 공공 데이터셋에서 수행된 실험 결과는 FedModule의 유연성과 확장성을 보여줍니다.



### Sample- and Oracle-Efficient Reinforcement Learning for MDPs with Linearly-Realizable Value Functions (https://arxiv.org/abs/2409.04840)
- **What's New**: 이 논문에서는 Markov Decision Processes (MDPs)에서 샘플 효율적(sample-efficient)이고 계산적으로 적합한 강화 학습(rl) 알고리즘을 제시합니다. 특히 무한한 상태와 행동 공간을 가진 환경을 효과적으로 모델링할 수 있는 알고리즘을 발전시킵니다.

- **Technical Details**: 우리의 알고리즘은 주어진 feature map에서 어떤 정책의 state-action value function이 선형(linear)인 조건 하에 작동합니다. 이 새로운 RL 알고리즘은 문제 매개변수에 다항식(polynomial)으로 의존하여 near-optimal policy를 찾습니다. 또한, cost-sensitive classification (CSC) oracle을 활용하여 효율적인 구현이 가능합니다.

- **Performance Highlights**: 특히, 우리의 CSC oracle은 feature 차원이 일정할 때 효율적으로 구현되며, 이는 비선형(non-convex) 문제를 해결해야 했던 기존 최고 수준의 방법들에 비해 중요한 개선을 나타냅니다. 기존 방법들은 horizon 많은 변수들로 구성될 때 계산 비용이 지수(exponential)적이었습니다.



### Reducing Events to Augment Log-based Anomaly Detection Models: An Empirical Study (https://arxiv.org/abs/2409.04834)
Comments:
          Accepted By ESEM'24

- **What's New**: 이 논문에서는 로그의 양을 줄이는 것이 이상 탐지에 미치는 영향을 조사하고, 이를 기반으로 자동 로그 정리를 위한 LogCleaner라는 방법론을 제안합니다.

- **Technical Details**: LogCleaner는 소프트웨어 시스템과 이상 탐지 모델 간의 미들웨어 역할을 하며, 원시 로그에서 anti-events와 duplicative-events를 필터링합니다. 주요 구성 요소로는 프로파일링과 온라인 컴포넌트가 있으며, TF-IDF와 그래프 기반 클러스터링 접근 방식이 사용됩니다.

- **Performance Highlights**: 실험 결과에 따르면, LogCleaner는 이상 탐지에서 70% 이상의 로그 이벤트를 줄이고, 모델의 추론 속도를 약 300% 가량 증가시키며 전반적인 모델 성능을 향상시킵니다.



### Achieving Peak Performance for Large Language Models: A Systematic Review (https://arxiv.org/abs/2409.04833)
Comments:
          34 pages, 7 figures, 8 tables. Journal Article: IEEE Access

- **What's New**: 최근 대규모 언어 모델(LLMs)에 대한 성능 최적화 및 가속화 방법들을 다룬 체계적인 문헌 검토가 진행되었습니다.

- **Technical Details**: 이 논문은 2017년부터 2023년 12월까지 983개의 자료 중 65편의 문헌을 리뷰하였으며, LLM의 학습 최적화, 하드웨어 최적화, 그리고 시스템 서비스 개선을 위한 세 가지 카테고리로 분류하였습니다. 각 최적화 전략과 관련된 최근의 방법론들을 정리하였고, 다양한 프레임워크와 라이브러리의 효율성을 평가했습니다.

- **Performance Highlights**: 논문은 LLMs의 훈련 및 추론 효율성을 높이는 실제 사례를 두 가지 포함하고 있으며, 최첨단 성능을 유지하면서도 자원 제약 문제를 해결할 수 있는 실용적인 접근을 제시합니다.



### Reward-Directed Score-Based Diffusion Models via q-Learning (https://arxiv.org/abs/2409.04832)
- **What's New**: 본 논문은 생성 AI를 위한 continuous-time score-based diffusion models의 강화 학습 (RL) 훈련을 위한 새로운 접근 방식을 제안합니다. 기존의 사전 훈련된 모델을 사용하지 않고, 보상 함수를 최대화하는 샘플을 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 저자들은 entropy-regularized continuous-time RL 문제를 제시하며, 최적의 확률적 정책은 알려진 공분산 행렬을 가지는 Gaussian 분포로 나타난다고 말합니다. 이 방법을 사용하여 생성된 Gaussian 정책의 평균을 매개변수화하고 actor-critic 타입의 (little) q-learning 알고리즘을 개발합니다.

- **Performance Highlights**: 제안된 알고리즘은 합성 훈련 데이터에 대해 실험을 수행한 결과, 기존의 두 가지 최신 RL 방법과 비교했을 때 우수한 성과를 보였습니다. 또한, 연속 시간 설정 덕분에 ODE 기반 모델로의 확장이 용이하다는 장점이 있습니다.



### MILE: A Mutation Testing Framework of In-Context Learning Systems (https://arxiv.org/abs/2409.04831)
- **What's New**: 본 연구에서는 In-context Learning (ICL) 시스템에 대해 고품질 테스트 데이터를 특징화하는 mutation testing 프레임워크인 MILE을 제안합니다. 이 프레임워크는 ICL 테스트의 신뢰성과 품질을 평가하기 위한 새로운 시각과 기술적 방법을 도입합니다.

- **Technical Details**: MILE 프레임워크는 ICL 시스템에 특화된 mutation operators와 mutation scores를 제안합니다. 특히 우리는 demonstration-level 및 prompt-level mutators를 설계하고, 다양한 결함을 특성화할 수 있는 group-wise mutation score도 포함하였습니다. 이러한 접근 방식은 ICL 시스템에서의 신뢰성 문제를 해결하는 데 중점을 두고 있습니다.

- **Performance Highlights**: MILE 프레임워크는 여러 기준 데이터세트와 인기 있는 LLM을 사용하여 평가되었으며, 실험 결과는 우리의 mutation scores가 테스트 세트의 품질과 강한 상관관계를 가지며, ICL 테스트 슈트의 품질을 측정하는 데 효과적임을 입증했습니다.



### NASH: Neural Architecture and Accelerator Search for Multiplication-Reduced Hybrid Models (https://arxiv.org/abs/2409.04829)
- **What's New**: 본 논문에서는 NASH라는 새로운 Neural architecture and Accelerator Search 프레임워크를 제안하여 multiplication-reduced hybrid 모델을 위한 최적의 모델과 가속기 조합을 자동으로 찾아내는 방법을 보여줍니다. 이 방법은 기존의 방법들보다 더 효율적이고 정확한 하드웨어 효율성을 제공합니다.

- **Technical Details**: NASH는 Neural Architecture Search (NAS)에서 tailored zero-shot metric을 사용하여 훈련 전에 유망한 하이브리드 모델을 식별함으로써 검색 효율성을 향상시키고 gradient conflicts를 완화합니다. 또한, 가속기 검색 수준에서 coarse-to-fine search 전략을 도입하여 검색 과정을 간소화하고, 이 두 검색 단계를 통합하여 최적의 모델과 가속기 조합을 찾습니다.

- **Performance Highlights**: 실험 결과, CIFAR-100에서 기존의 multiplication 기반 시스템에 비해 throughput은 2.14배, FPS는 2.01배 증가하고 정확도는 0.25% 상승했습니다. Tiny-ImageNet에서도 0.56%의 정확도 향상을 달성했습니다.



### POINTS: Improving Your Vision-language Model with Affordable Strategies (https://arxiv.org/abs/2409.04828)
- **What's New**: 이번 연구에서는 비전-언어 모델(Vision-Language Models)의 최신 발전을 반영한 강력한 베이스라인 모델을 구축했습니다. 또한, 데이터셋을 쉽고 효율적으로 필터링하기 위해 perplexity를 활용하여 훈련 데이터셋을 선택하는 방안과 다양한 데이터셋에서 모델을 통합하는 model soup 전략을 도입했습니다.

- **Technical Details**: 제안된 모델, POINTS는 9B 파라미터로 구성되어 있으며, Consistent Aspect Ratio Dynamic High Resolution (CATTY) 기법을 통해 이미지 왜곡 문제를 해결합니다. 또한, perplexity를 활용하여 1M 규모의 큐레이션된 데이터셋을 선택하여 훈련했습니다. 이 전략은 기존 모델보다 성능이 우수합니다.

- **Performance Highlights**: 필터링된 데이터셋으로 학습한 모델은 원래 5배 큰 데이터셋으로 학습한 모델보다 뛰어난 성능을 보였습니다. 특히, 데이터셋 선택의 한계에 도달한 후에도 model soup 기법을 통해 성능 향상을 이끌어낼 수 있었습니다.



### Exploring Straightforward Conversational Red-Teaming (https://arxiv.org/abs/2409.04822)
- **What's New**: 대규모 언어 모델(LLMs)이 비즈니스 대화 시스템에서 점점 더 많이 사용되고 있지만, 이들은 보안 및 윤리적 위험을 초래합니다. 이 논문에서는 공격 공격자 LLM을 사용하여 타겟 LLM에서 원하지 않는 출력을 유도하는 직관적인 레드팀(red-teaming) 접근 방식의 효과를 조사합니다.

- **Technical Details**: 우리는 다양한 단일 및 다중 턴(red-teaming) 전략을 비교하며, 공격자의 각기 다른 전술을 평가합니다. 이 연구에서는 사전 훈련된 LLM이 추가 훈련 없이도 공격자 모델로 효과적으로 사용될 수 있는지, 또한 대화 설정에서 공격 표면이 확대되는지에 대해 탐구합니다.

- **Performance Highlights**: 실험 결과, 사용자가 과거 시도의 경험에 따라 공격 전략을 조정할 수 있도록 하며, 비즈니스 애플리케이션에서의 활용 가능성을 확인했습니다. 이 연구는 성공적인 공격을 위한 대화 회수 수와 같은 중요한 변수에 대한 통찰력을 제공합니다.



### FreeAugment: Data Augmentation Search Across All Degrees of Freedom (https://arxiv.org/abs/2409.04820)
Comments:
          Accepted by ECCV 2024

- **What's New**: 새로운 접근 방식인 FreeAugment는 데이터 증강(Data Augmentation) 정책의 네 가지 자유도(all degrees of freedom)를 전역적으로 최적화하는 첫 번째 방법으로, 완전히 미분 가능한(differentiable) 방법을 사용하여 비최적화 문제에서 자동으로 최적의 이미지 변환을 찾습니다.

- **Technical Details**: FreeAugment는 (1) 변환의 수를 학습할 수 있는 확률 분포를 도입하고, (2) 변환의 유형 및 순서를 최적화하여 중복 변환을 내재적으로 방지하며, (3) 각 증강에 대해 확률 분포에서 확률적 크기를 샘플링합니다. 이러한 모든 구성은 검증 세트에 대해 end-to-end 방식으로 최적화됩니다.

- **Performance Highlights**: FreeAugment는 CIFAR10/100 및 ImageNet-100과 같은 자연 이미지 벤치마크에서 최첨단의 성능을 달성하며, 다양한 도메인에서 적용 가능성과 견고성을 입증하였습니다.



### Top-GAP: Integrating Size Priors in CNNs for more Interpretability, Robustness, and Bias Mitigation (https://arxiv.org/abs/2409.04819)
Comments:
          eXCV Workshop at ECCV 2024

- **What's New**: 이 논문에서는 Top-GAP이라는 새로운 정규화 기술이 소개되었습니다. 이 기술은 CNN의 설명가능성과 강건성을 향상시키는 데 중점을 둡니다.

- **Technical Details**: Top-GAP은 학습된 특징 표현의 공간 크기를 제한하여 네트워크가 가장 중요한 이미지 영역에 집중할 수 있도록 합니다. 이는 배경의 영향을 줄이고, 적대적 공격(adversarial attacks)과 Effective Receptive Field(ERF)를 통해 검증되었습니다.

- **Performance Highlights**: 이 방법을 사용하여 CIFAR-10 데이터셋에서 PGD 공격에 대해 50% 이상의 강건 정확도를 달성하였고, 객체 국소화(지역화)에서 IOU(Intersection over Union)가 최대 25% 향상되었습니다.



### Generalized Learning of Coefficients in Spectral Graph Convolutional Networks (https://arxiv.org/abs/2409.04813)
- **What's New**: 본 논문에서는 G-Arnoldi-GCN이라는 새로운 알고리즘을 제안합니다. 이 알고리즘은 주어진 필터 함수를 효율적이고 효과적으로 근사하기 위해 Arnoldi 정규 직교화(orthonormalization) 기반의 접근 방식을 사용합니다. 또한, 필터 함수와 다항식 근사(polygonal approximation) 간의 관계를 명확히 하고 이를 통해 효율적인 필터 설계를 가능하게 합니다.

- **Technical Details**: G-Arnoldi-GCN은 주어진 필터를 비너몬드(Vandermonde) 선형 시스템을 사용하여 다항식으로 근사하는 방식으로 작동합니다. 이 알고리즘은 혁신적인 QR 분해(QR decomposition) 접근법을 활용하여 정확한 다항식 근사를 제공합니다. 또한, Arnoldi-GCN 알고리즘은 현재 방법에서 다항식 업데이트 규칙이 필요 없도록 하여 새로운 전파 매커니즘을 제시합니다.

- **Performance Highlights**: G-Arnoldi-GCN은 10개의 다양한 토폴로지 특성을 가진 데이터셋을 통해 다중 클래스 노드 분류(multi-class node classification) 성능을 평가한 결과, 적합한 필터 함수를 사용할 때 기존의 최첨단 방법들보다 일관되게 우수한 성능을 보였습니다. 15개의 벤치마크 데이터셋에서의 엄격한 실험 결과는 이 알고리즘이 노드 분류 성능에서 획기적인 성과를 이룬 것을 보여줍니다.



### Phrase-Level Adversarial Training for Mitigating Bias in Neural Network-based Automatic Essay Scoring (https://arxiv.org/abs/2409.04795)
- **What's New**: 이번 연구에서는 자동 에세이 채점 시스템(AES)의 모델 불변(agnostic) 문구 수준 방법을 제안하여 AES 모델의 편향(bias)과 강건함을 개선하기 위한 적대적(Adversarial) 에세이 세트를 생성했습니다. 이로 인해 편향이 줄어들고 모델의 강건성이 향상되었습니다.

- **Technical Details**: 제안된 접근법은 원본 테스트 세트와 적대적으로 생성된 샘플을 포함하는 공격 테스트 세트를 구성합니다. 기계 학습 모델을 통해 에세이를 평가하고, 여러 신경망 점수 모델을 활용하여 공격 전략과 데이터 증강의 효과를 평가합니다. 주요 과정은 문장 추출기, 문구 추출, 변화된 문구 생성, 라벨 보존 필터 적용 순으로 진행됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 적대적 예시가 있는 경우와 없는 경우 모두에서 AES 모델의 성능을 크게 향상시킴을 보여주었습니다. 특히, 적대적인 샘플들에 대해 더 강건한 채점 결과를 제공하게 되었습니다.



### Improving Deep Reinforcement Learning by Reducing the Chain Effect of Value and Policy Churn (https://arxiv.org/abs/2409.04792)
- **What's New**: 이 논문에서는 Deep Reinforcement Learning (DRL)에서 발생하는 'churn' (예측값의 급격한 변화) 현상을 이해하고 이를 제어하기 위한 새로운 방법인 Churn Approximated ReductIoN (CHAIN)을 제안합니다.

- **Technical Details**: churn 현상은 RL의 비정상적 특성으로 인해 발생하며, 이는 정책 개선과 가치 추정 모두에 영향을 미쳐 학습 동적에 편향을 초래합니다. 특히, 정책 churn (정책 개선의 churn)과 가치 churn (가치 추정의 churn)의 상호작용으로 인한 연쇄 효과를 분석합니다.

- **Performance Highlights**: 실험결과, CHAIN 방법은 다양한 DRL 설정에서 churn을 효과적으로 줄이고 최종 성능을 개선하는 것으로 나타났습니다. 또한 CHAIN은 넓은 네트워크나 깊은 네트워크를 사용할 때 성능이 크게 향상됨을 보여주었습니다.



### Selective Self-Rehearsal: A Fine-Tuning Approach to Improve Generalization in Large Language Models (https://arxiv.org/abs/2409.04787)
Comments:
          14 pages, 8 figures

- **What's New**: 이 논문은 Selective Self-Rehearsal (SSR)이라는 새로운 미세 조정(fine-tuning) 방법론을 소개하며, 이를 통해 모델의 일반화 능력을 개선하면서도 표준 감독 미세 조정(SFT)과 유사한 성능을 달성한다고 보고하였습니다.

- **Technical Details**: SSR은 질문에 대해 여러 개의 유효한 응답이 존재할 수 있다는 사실을 활용합니다. 모델의 올바른 응답을 식별한 후, SSR은 이러한 응답과 나머지 샘플에 대한 금본(gold) 응답을 이용해 모델을 미세 조정합니다. 이 과정은 적절한 LLM을 판별자로 사용하여 이루어지며, SSR은 모델이 자신의 생성된 출력(output)을 활용하여 학습하게 합니다.

- **Performance Highlights**: 여러 데이터셋에서 수행된 실험 결과에 따르면, 표준 SFT는 MMLU와 TruthfulQA와 같은 여러 벤치마크에서 평균 성능이 최대 16.7% 감소한 반면, SSR을 적용한 경우 평균적으로 2% 감소에 그쳐, SSR이 표준 SFT보다 더 나은 일반화 능력을 보여주었습니다.



### Leveraging LLMs, Graphs and Object Hierarchies for Task Planning in Large-Scale Environments (https://arxiv.org/abs/2409.04775)
Comments:
          8 pages, 6 figures

- **What's New**: 본 연구에서는 LLMs(대형 언어 모델)가 인코딩한 상식 지식을 활용하여 로봇의 작업 수준에서의 계획 문제를 해결하는 새로운 방법을 제안합니다. 주된 목표는 대규모 환경에서의 계획 문제를 간소화하는 것입니다.

- **Technical Details**: 복잡한 계획 문제의 상태 공간에서 의미 없는 구성 요소를 제거하기 위해 LLM을 효율적으로 활용합니다. 이러한 접근 방식은 작업 목표 함수에 따라 상태-행동 공간을 줄이는 방법론을 통해 이루어집니다. 그래프 기반의 상태 표현을 사용하여 환경의 상태를 모델링하며 노드와 에지를 통해 물체 간의 관계를 시각적으로 표현합니다. 특히, 객체와의 관계에서 발생하는 알고리즘 복잡도를 감소시키는 것이 핵심입니다.

- **Performance Highlights**: 가정용 시뮬레이션 환경에서 수행된 광범위한 실험을 통해 제안된 시스템의 효율성을 입증하였고, 7-DoF 조작기를 활용하여 실제 세계에서도 검증하였습니다.



### Untie the Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models (https://arxiv.org/abs/2409.04774)
- **What's New**: 이 논문에서는 'Untie the Knots' (UtK)라는 새로운 데이터 증강 전략을 소개합니다. 이 방법은 기존 데이터 혼합을 수정하지 않고도 LLM이 긴 컨텍스트 처리 능력을 효율적으로 향상시키도록 설계되었습니다.

- **Technical Details**: UtK는 입력 문서를 청크(chunk) 단위로 나누고, 이를 무작위로 섞은 후 복잡한 구조의 긴 텍스트를 생성하여, 모델이 무질서한 토큰 시퀀스 내의 관련 세그먼트를 식별하도록 학습합니다. 이 과정에서 백트레이싱(Backtracing) 작업을 도입해, 모델이 모든 세그먼트를 올바른 순서로 찾도록 합니다.

- **Performance Highlights**: 7B 및 72B 매개변수 모델을 사용하여 200억 개의 토큰으로 훈련한 결과, UtK는 128K 컨텍스트 길이에서 RULER에서 75% 및 84.5% 정확도를 달성하며, 기존의 데이터 전략을 크게 초월하는 성능을 보여줍니다. 이 모델들은 오픈 소스로 제공되어 향후 연구를 촉진할 예정입니다.



### LMGT: Optimizing Exploration-Exploitation Balance in Reinforcement Learning through Language Model Guided Trade-offs (https://arxiv.org/abs/2409.04744)
- **What's New**: 본 논문에서는 탐사(exploration)와 활용(exploitation) 사이의 균형을 최적화하기 위해 대규모 언어 모델(Large Language Models, LLMs)의 지식을 활용한 새로운 샘플 효율적인 프레임워크인 ‘LMGT(Language Model Guided Trade-offs)’를 도입했습니다. 이 프레임워크는 RL(강화 학습) 작업의 샘플 효율성을 향상시키고, 다양한 환경에서의 성능을 높이기 위해 보상 변동(reward shifts) 기법을 사용하여 에이전트의 탐사 노력을 유도합니다.

- **Technical Details**: LMGT은 다양한 출처에서의 사전 지식을 활용하여 RL 모델의 학습을 효율적으로 지원합니다. 언어 모델은 환경 정보를 처리하고 에이전트 행동에 점수를 매겨 탐사 및 활용을 조절하는 방식으로, 이는 통계적 정보에 기반하여 Q-함수의 초기화를 변경하는 것과 같은 효과를 가집니다. 이 접근 방식은 에이전트를 사전 지식 없이 독립적으로 운용할 수 있도록 하여, LLM의 환각(hallucinations)으로 인한 위험을 최소화합니다.

- **Performance Highlights**: 실험 결과, LMGT는 다양한 RL 작업에서 성능을 일관되게 향상시키며, 구글의 산업용 추천 알고리즘인 SlateQ에 적용했을 때도 훈련 비용이 현저히 감소함을 보여주었습니다. 본 프레임워크는 학습 효율성을 극대화하며, 산업 실무자들에게 RL 모델의 훈련 비용을 줄일 수 있는 실제적인 솔루션을 제공합니다.



### Up-sampling-only and Adaptive Mesh-based GNN for Simulating Physical Systems (https://arxiv.org/abs/2409.04740)
- **What's New**: 이 논문에서는 전통적인 복잡한 기계 시스템의 시뮬레이션을 개선하기 위해 UA-MGN(Up-sampling-only and Adaptive Message Propagation 기술을 기반으로 한 계층 구조의 Mesh Graph Network)을 제안합니다.

- **Technical Details**: 전통적인 시뮬레이션 방법은 PDE(Partial Differential Equations)를 수치적으로 해결하는 FEM(Finite Element Method)에 의존합니다. UA-MGN은 효과적인 메커니즘을 통해 복잡한 기계 시스템의 비효율적인 표현과 메시지 전파(MP)의 문제를 해결합니다.

- **Performance Highlights**: 두 개의 합성 데이터셋과 하나의 실제 데이터셋에서 UA-MGN은 MS-MGN(state-of-the-art)보다 40.99% 낮은 오류를 기록하며, 네트워크 매개변수는 43.48% 적고 FLOPs(floating point operations)는 4.49% 감소시켰습니다.



### VidLPRO: A $\underline{Vid}$eo-$\underline{L}$anguage $\underline{P}$re-training Framework for $\underline{Ro}$botic and Laparoscopic Surgery (https://arxiv.org/abs/2409.04732)
- **What's New**: VidLPRO는 로봇 및 복강경 수술을 위해 특별히 설계된 새로운 비디오-언어(VL) 사전 학습 프레임워크입니다. 이 모델은 기존의 대조 학습(constrastive learning) 방법을 넘어, 보다 포괄적인 접근 방식을 통해 비디오와 언어를 정렬하는 복잡한 시간적 동역학을 캡처합니다.

- **Technical Details**: VidLPRO는 Video-Text Contrastive Learning (VTC), Video-Text Matching (VTM), Masked Language Modeling (MLM) 목표를 결합하여 풍부한 VL 표현을 학습합니다. 또한 GenSurg+ 데이터세트를 통해 17,000개의 수술 비디오 클립과 GPT-4에 의해 생성된 캡션이 결합되어 있습니다. 이 데이터세트는 수술 영역에 필요한 대규모 고품질 VL 데이터를 충족시킵니다.

- **Performance Highlights**: VidLPRO는 zero-shot 수술 단계 인식에서 최첨단 성능을 달성하며, Cholec80에서 57.1%의 정확도와 32.1%의 F1 점수를 기록하여 기존 모델보다 21.5%의 정확도 향상과 15.7%의 F1 점수 향상을 보였습니다. 단일 프레임 추론에서도 견고한 성능을 나타내며, 시간적 컨텍스트가 증가함에 따라 효과적으로 확장됩니다.



### NapTune: Efficient Model Tuning for Mood Classification using Previous Night's Sleep Measures along with Wearable Time-series (https://arxiv.org/abs/2409.04723)
Comments:
          Accepted at ICMI 2024

- **What's New**: 이 연구에서는 수면 관련 데이터를 웨어러블 기반 기분 인식에 통합하는 새로운 프레임워크인 NapTune을 제안합니다. NapTune은 미리 훈련된 웨어러블 시계열 인코더에 수면 데이터를 추가 입력으로 사용하여 기분 인식 성능을 향상시킵니다.

- **Technical Details**: NapTune은 고정된 Transformer 인코더와 각 레이어에 경량 프로프트 매개변수를 학습하여 기분 인식을 수행합니다. 이 접근은 ECG, PPG, EDA 신호를 통해 수집된 웨어러블 시계열 데이터와 함께 수면 관련 측정값을 사용하여 더 나은 성능을 입증합니다.

- **Performance Highlights**: NapTune은 기존 기법들에 비해 F1 점수에서 최대 11% 향상을 보였으며, 수면 관련 데이터를 사용함으로써 기분 인식의 샘플 효율성을 크게 증대시켰습니다. 또한, 수면 패턴이 기분 인식에 미치는 영향을 분석하여 각기 다른 기분을 효과적으로 구별할 수 있음을 입증했습니다.



### A Comprehensive Survey on Evidential Deep Learning and Its Applications (https://arxiv.org/abs/2409.04720)
- **What's New**: Evidential Deep Learning (EDL) is introduced as a novel paradigm for reliable uncertainty estimation with minimal computational overhead, particularly useful in high-risk applications like autonomous driving and medical diagnosis. EDL leverages subjective logic to enhance uncertainty estimation in a single forward pass, contrasting mainstream methods that require heavier computational resources.

- **Technical Details**: EDL is built on subjective logic, a variant of Dempster-Shafer Evidence Theory, which assigns belief masses to potential outcomes to represent uncertainties. It models the posterior predictive distribution using a Dirichlet distribution and minimizes loss functions integrated over this distribution, facilitating effective evidence collection without added complexity. The paper discusses theoretical advancements in evidence collection, out-of-distribution (OOD) sample estimation, training strategies, and evidential regression networks.

- **Performance Highlights**: The survey highlights EDL's broad applicability across various machine learning paradigms and its extensive use in downstream tasks, aiming for improved performance and wider adoption. Future research directions are suggested to enhance EDL capabilities and explore potential applications.



### Enhancing Deep Learning with Optimized Gradient Descent: Bridging Numerical Methods and Neural Network Training (https://arxiv.org/abs/2409.04707)
- **What's New**: 이번 논문은 최적화 이론(optimization theory)과 딥 러닝(deep learning) 간의 깊은 관계를 탐색하며, 딥 러닝에서의 최적화 문제의 보편성을 강조합니다.

- **Technical Details**: 주요 기술적 내용으로는 신경망(neural networks) 최적화를 위한 기울기 하강법(gradient descent) 알고리즘과 그 변형들이 소개됩니다. 또한, 수치 최적화(numerical optimization) 기법에서 영감을 얻은 SGD 최적화기(optimizer)의 개선 방안이 제안됩니다.

- **Performance Highlights**: 다양한 딥 러닝(Task)에서 수행된 실험 결과, 개선된 알고리즘의 효율성이 입증되었음을 보고합니다. 이 논문은 최적화 이론의 지속적인 발전과 복잡한 문제 해결(NLP, 컴퓨터 비전 등)에서의 역할 확대를 강조합니다.



### A Multi-scenario Attention-based Generative Model for Personalized Blood Pressure Time Series Forecasting (https://arxiv.org/abs/2409.04704)
Comments:
          5 pages, 2 figures

- **What's New**: 이 논문에서는 환자의 개인 생리학에 맞춘 개인화된 혈압(BP) 예측 모델을 제안하며, ECG(심전도)와 PPG(광용적맥파측정기) 신호를 활용하여 다양한 의료 시나리오에서 높은 정확도를 달성합니다.

- **Technical Details**: 이 연구는 개인화된 BP 예측 모델을 개발하며, 이를 위해 2D 표현 학습을 사용하여 복잡한 생리학적 관계를 포착합니다. TABNet라는 새로운 Time-series Attention-based Blood Pressure Forecast Network를 제안하며, multi-domain feature fusion을 적용하여 시간 도메인 및 비선형 특징을 포함합니다. 데이터 전처리 및 특징 추출 과정에서 심전도와 PPG 신호로부터 38가지의 다양한 특징을 추출합니다.

- **Performance Highlights**: 제안된 모델은 60명의 피험자로부터 수집된 세 가지 다양한 시나리오에서 AAMI 기준을 충족하는 높은 정확도와 강인한 BP 예측 결과를 보여줍니다. 이 연구는 수술이나 집중 치료를 받는 고위험 환자의 조기 이상 탐지를 가능하게 합니다.



### Solving Stochastic Orienteering Problems with Chance Constraints Using a GNN Powered Monte Carlo Tree Search (https://arxiv.org/abs/2409.04653)
Comments:
          8 pages, 6 figures

- **What's New**: 본 논문은 그래프 신경망(Graph Neural Network, GNN)과 메시지 패싱(message passing) 기술을 활용한 랜덤 오리엔티어링 문제(stochastic orienteering problem, SOP)를 해결하기 위한 몬테 카를로 트리 탐색(Monte Carlo Tree Search, MCTS) 방법론을 제안합니다. 이 방법론은 할당된 여행 예산을 준수하면서 수집된 보상을 극대화하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 MCTS 알고리즘은 온라인 방식으로 작동하며, 경로 계획(planning)과 실행(execution)을 반복하면서 남은 여행 예산을 모니터링하여 방문할 다음 정점을 결정합니다. 또한 GNN을 사용하여 각 가능 동작의 유용성(utility)과 실패 확률(failure probability)을 예측함으로써 롤아웃(rollout) 단계를 혁신적으로 구현합니다. 이중 예측 방식은 SOP 인스턴스 해결에 필수적인 두 가지 주요 메트릭(누적 보상과 예산 제약 위반 확률)을 동시에 예측합니다.

- **Performance Highlights**: 제안된 방법과 아키텍처를 통해 복잡한 문제 인스턴스를 효율적으로 해결할 수 있으며, 수집된 보상 면에서는 중간 정도의 손실을 감수합니다. 실험 결과 이 방법이 학습 데이터를 넘어 일반화 가능하다는 점도 입증되었습니다. 수치적으로 MCTS의 대부분의 계획 시간을 소모하는 롤아웃 단계를 기계학습 기반으로 최적화하여 실시간 속도(real-time speeds)로 복잡한 오리엔티어링 솔루션을 계획하고 실행할 수 있습니다.



### Stacked Universal Successor Feature Approximators for Safety in Reinforcement Learning (https://arxiv.org/abs/2409.04641)
Comments:
          13 pages

- **What's New**: 이 논문에서는 안전한 강화학습 환경을 위해 스택된, 연속 제어 변형인 보편적 후속 특성 근사(USFA)를 사전 조건자와 결합한 새로운 방법인 Stacked Universal Successor Feature Approximation for Safety (SUSFAS)를 제안합니다.

- **Technical Details**: SUSFAS는 연속 제어 환경에서 개별화된 후속 특성(SF)을 예측하는 스택된 변형을 사용하며, 인간 시스템에서 안전을 보장하고자 하는 다양한 목표를 조화롭게 최적화합니다. 이 방법은 부가적인 안전 제어기와의 협력을 통해 성능을 개선시키는데, 특히 런타임 보증(RTA) 제어기를 활용합니다.

- **Performance Highlights**: 우리의 방법을 통해 임무 비판 환경에서 연료 사용량을 최대 18배까지 줄일 수 있으며, 안전 제어기 행동을 효과적으로 학습하여 안전-critical 응용에서 능률적인 주요 제어기로서의 역할을 수행할 수 있음을 수치적으로 입증했습니다.



### Enhancing Quantum Security over Federated Learning via Post-Quantum Cryptography (https://arxiv.org/abs/2409.04637)
Comments:
          Submission for IEEE 2024 IEEE Workshop on Quantum IntelLigence, Learning & Security (QUILLS), this https URL

- **What's New**: 이 연구에서는 양자 컴퓨터 시대에 대응하기 위한 디지털 서명 알고리즘의 Post-Quantum Cryptography(PQC) 알고리즘인 Dilithium, FALCON, SPHINCS+의 영향을 탐구합니다. 특히, Dilithium이 연합 학습(federated learning)에서 가장 효율적인 PQC 알고리즘으로 입증되었습니다.

- **Technical Details**: 연구는 연합 학습의 다양한 모델 및 설정에서 세 가지 NIST 표준 PQC 알고리즘의 실증적 영향을 조사하였습니다. 이를 통해 연합 학습 환경에서 PQC DSA 알고리즘을 적용할 때의 효율성 특성을 분석하였습니다.

- **Performance Highlights**: Dilithium은 연합 학습에서의 데이터 변조 공격 방어에 가장 효과적인 PQC DSA 알고리즘으로 부각되었습니다. 이 결과는 향후 연합 학습의 보안성을 향상시키기 위한 중요한 통찰을 제공합니다.



### Zero-Shot Whole Slide Image Retrieval in Histopathology Using Embeddings of Foundation Models (https://arxiv.org/abs/2409.04631)
Comments:
          This paper will be updated with more results

- **What's New**: 본 논문에서는 최근 발표된 기초 모델(base models)을 사용하여 조직병리학(histopathology) 이미지 검색을 테스트하였습니다. 이전에 발표되지 않은 제로샷(zero-shot) 검색 방식을 적용하였으며, 이는 임베딩(embeddings)을 변경하지 않고 어떠한 분류기(classifier)도 훈련시키지 않았음을 의미합니다.

- **Technical Details**: 테스트 데이터로는 TCGA(The Cancer Genome Atlas)에서 제공하는 진단 슬라이드를 사용하였으며, 23개 장기와 117개 암 하위 유형을 포함하고 있습니다. 검색 플랫폼으로는 패치를 이용한 WSI(Whole Slide Imaging) 검색을 수행할 수 있는 Yottixel을 사용하였습니다.

- **Performance Highlights**: 최상위 5개 검색에 대한 F1 점수는 27% +/- 13% (Yottixel-DenseNet), 42% +/- 14% (Yottixel-UNI), 40% +/- 13% (Yottixel-Virchow), 41% +/- 13% (Yottixel-GigaPath)로 나타났습니다. GigaPath WSI의 결과는 처리에 필요한 상당한 계산 자원으로 인해 지연될 예정입니다.



### A Short Survey on Set-Based Aggregation Techniques for Single-Vector WSI Representation in Digital Pathology (https://arxiv.org/abs/2409.04615)
- **What's New**: 이 논문은 디지털 병리학에서 Whole Slide Images (WSIs)의 집합 기반 접근 방법을 통해 단일 벡터 형태로 WSIs를 표현하는 혁신적인 방법을 다룹니다.

- **Technical Details**: WSIs는 기가픽셀(gigapixel) 파일로, 조직 샘플의 세부 정보를 정밀하게 캡처하며, 이를 효과적으로 처리하기 위해 'patch-oriented' 방법론을 탈피하고 단일 벡터 표현 방식으로 전환하는 것이 필요합니다. 이 접근 방식은 더 효율적이고 효과적인 디지털 병리학적 분석을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 대용량 데이터를 저장하고 분석하는 데 필요한 고성능 스토리지를 요구하지 않으면서도, 병원들 간의 헬스케어 품질 및 접근성의 격차를 해소할 수 있는 잠재력을 가지고 있습니다.



### Decentralized Learning in General-sum Markov Games (https://arxiv.org/abs/2409.04613)
Comments:
          16 pages, 1 figure

- **What's New**: 이 논문은 일반형 마르코프 게임에서의 분산 학습 알고리즘 설계를 탐구하며, 근사 내쉬 균형(approximate Nash equilibria)으로의 수렴을 보장하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 마르코프 근사 잠재 함수(Markov Near-Potential Function, MNPF)를 사용하여 알고리즘이 정확한 내쉬 균형에 수렴하는 문제를 다룹니다. 이 설정은 두 가지 시간 척도를 활용하여, Q-function 추정이 정책 업데이트보다 빠르게 업데이트되고, 최종적으로 MNPF의 수준 집합(level set)으로 수렴함을 보여줍니다. MNPF는 연속 시간 동역학 시스템의 리야푼 함수(Lyapunov functions)로 작용하여 안정성을 보장합니다.

- **Performance Highlights**: 이 연구는 알고리즘이 유한한 내쉬 균형 집합을 가정할 경우, 최종적으로 단일 균형의 이웃으로 수렴할 것임을 입증하였으며, 이는 근본적으로 실제 다중 에이전트 환경에서의 복잡한 상호작용을 보다 효과적으로 모델링하는 데 기여합니다.



### Detection of False Data Injection Attacks (FDIA) on Power Dynamical Systems With a State Prediction Method (https://arxiv.org/abs/2409.04609)
Comments:
          Under review

- **What's New**: 이번 연구는 전력 시스템에서의 잘못된 데이터 주입 공격(FDIA)을 탐지하기 위한 새로운 방법을 제안하며, 특히 동적 전력 시스템 모델의 상태 예측 방법을 효율적으로 활용합니다.

- **Technical Details**: Long Short-Term Memory (LSTM) 및 Graph Neural Networks (GNN)와 LSTM을 결합한 상태 예측 모델이 사용되었습니다. 이 모델들은 전력 시스템의 주파수 동역학을 예측하고, 이를 통해 FDIA 이벤트를 식별하는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 상태 예측 모델은 다양한 공격 및 배치 설정에서도 높은 탐지 정확도를 유지할 수 있음을 보여줍니다.



### Training quantum machine learning model on cloud without uploading the data (https://arxiv.org/abs/2409.04602)
Comments:
          6 pages, 1 figure

- **What's New**: 이 논문은 양자 유니터리(quantum unitary) 연산의 선형성을 기반으로 하여, 입력 데이터를 인코딩하기 전에 파라미터화된 양자 회로를 실행하는 방법을 제안하고 있습니다. 이를 통해 데이터 소유자는 자신의 데이터를 누출할 위험 없이 양자 클라우드 컴퓨테이션 플랫폼에서 머신러닝 모델을 훈련할 수 있습니다.

- **Technical Details**: 제안된 방법은 중간 단계에서 모든 연산이 유니터리 변환(unitary transformation)이며, 마지막 단계에서만 측정이 이루어지는 양자 회로에 적용됩니다. 이 방법은 양자 머신러닝에서 널리 사용되는 변동 양자 회로(variational quantum circuit, VQC) 아키텍처에 적합합니다.

- **Performance Highlights**: 이 방법은 인코딩의 병목 현상을 완화시키고 회로 깊이(circuit depth)를 O(2^n)에서 n/2로 줄이는 결과를 보여줍니다. 이를 통해 양자 및 양자 영감을 받은(machine learning) 모델이 기존의 고전 신경망에 비해 가지는 또 다른 이점을 입증하고 데이터 보안 측면에서 접근 방식을 다양화합니다.



### The emergence of Large Language Models (LLM) as a tool in literature reviews: an LLM automated systematic review (https://arxiv.org/abs/2409.04600)
Comments:
          18 main pages with 5 figures and 1 table, references, followed by supplementary materials

- **What's New**: 이번 연구는 과학 리뷰 작성 과정에서 Large Language Models (LLMs)의 사용을 요약하고, 리뷰 자동화의 다양한 단계와 최신 연구 프로젝트의 현 상태를 평가합니다.

- **Technical Details**: 연구는 2024년 6월에 PubMed, Scopus, Dimensions, Google Scholar 데이터베이스에서 인간 리뷰어에 의해 진행되었습니다. Covidence에서 LLM 추가 기능을 사용하여 스크리닝 및 추출 과정을 거쳤으며, OpenAI GPT-4o 모델을 활용했습니다. ChatGPT는 추출된 데이터를 정리하고, 논문 내 그림을 위한 코드를 생성하는 데 사용되었습니다.

- **Performance Highlights**: 3,788개의 논문이 검색되었고, 172개의 연구가 최종 리뷰에 적합하다고 판단되었습니다. ChatGPT 및 GPT 기반 LLM이 리뷰 자동화에서 가장 우세한 아키텍처로 나타났으며 (n=126, 73.2%), LLM을 활용하여 작성된 실제 리뷰는 단 26편 (15.1%)에 불과했습니다. GPT 기반 모델은 데이터 추출에서 평균 정밀도 83.0% (SD=10.4), 재현율 86.0% (SD=9.8)로 BERT 기반 모델보다 성능이 우수하였지만, 제목 및 초록 스크리닝 단계에서는 약간 낮은 정확도를 보였습니다 (Maccuracy=77.3%, SD=13.0).



### CubicML: Automated ML for Distributed ML Systems Co-design with ML Prediction of Performanc (https://arxiv.org/abs/2409.04585)
- **What's New**: 이번 논문에서는 분산 머신러닝 시스템의 훈련 성능을 자동으로 최적화하는 CubicML을 제안합니다. CubicML은 머신러닝 모델을 사용하여 훈련 성능을 예측하고, 이를 통해 성능 최적화에 필요한 하이퍼파라미터를 효율적으로 탐색할 수 있도록 합니다.

- **Technical Details**: CubicML은 다섯 가지 핵심 구성 요소로 이루어져 있습니다: (1) ML 시스템: 훈련 작업을 시작하는 클러스터 스택입니다. (2) 역사적 작업 데이터: 완료된 작업의 메타데이터를 저장하고, 이를 통해 회귀 모델(예: ‘예측기’)을 학습합니다. (3) 탐색 공간: CubicML이 조정할 수 있는 하이퍼파라미터 집합 및 값 범위를 정의합니다. (4) 예측기: 시스템 성능을 예측하는 경량 회귀 모델입니다. (5) 탐색기: 탐색 알고리즘을 사용하여 하이퍼파라미터를 샘플링합니다. CubicML은 RL(강화학습) 알고리즘을 사용하여 탐색합니다.

- **Performance Highlights**: CubicML은 대규모 광고 추천 모델에서 훈련 속도를 10.3% 향상시켰으며, 이 과정은 완전히 자동화되어 있어 인력 자원을 절약하고, MegaWatts 단위의 전력 절약 효과를 가져옵니다.



### ActionFlow: Equivariant, Accurate, and Efficient Policies with Spatially Symmetric Flow Matching (https://arxiv.org/abs/2409.04576)
- **What's New**: 이번 연구에서는 로봇 작업에서의 공간적 이해(spatial understanding)를 향상시키기 위해 ActionFlow라는 새로운 정책 클래스(policy class)를 도입합니다. 이는 복잡한 조작 작업(manipulation tasks)에서의 일반화(generalization)를 촉진하는 데 도움을 줍니다.

- **Technical Details**: ActionFlow는 공간 대칭 유도 편향(spatial symmetry inductive biases)을 통합하여 표현력 있는(action sequences) 행동 시퀀스를 생성합니다. SE(3) 불변 변환기(SE(3) Invariant Transformer) 아키텍처를 도입하여 관측 및 행동 간의 상대적인 SE(3) 자세(pose)에 기반한 정보적인 공간 추론(spatial reasoning)을 가능하게 합니다. 또한, Flow Matching이라는 최첨단(advanced) 딥 생성 모델(deep generative model)을 이용하여 고품질 샘플(samples)을 빠른 추론(inference)으로 생성합니다.

- **Performance Highlights**: 실험 결과, ActionFlow 정책은 강력한 공간 및 지역성(biases) 편향을 보이며, SE(3)-공변적인(equivariant) 행동 생성을 수행합니다. 여러 시뮬레이션 및 실제 로봇 조작 작업에서 ActionFlow의 효과성과 그 두 가지 주요 구성 요소의 성과가 입증되었습니다.



### Thinking Outside the BBox: Unconstrained Generative Object Compositing (https://arxiv.org/abs/2409.04559)
- **What's New**: 본 논문에서는 제약 없는 생성적 객체 합성(unconstrained generative object compositing)의 새로운 문제를 정의하고, 기존의 마스크에 의존하지 않는 방법으로 객체 효과(그림자 및 반사)를 생성할 수 있는 디퓨전 기반 모델을 제안합니다. 이는 그림에 자연스럽고 리얼리틱한 합성을 제공합니다.

- **Technical Details**: 제안된 모델은 객체를 빈 마스크(empty mask)와 함께 사용할 수 있으며, 이 경우 자동으로 객체를 적절한 위치와 크기로 배치합니다. 이 과정에서 이미지 인페인팅(image inpainting)을 사용하여 훈련 데이터를 생성하고, 다중 스케일 이미지 임베딩(multi-scale image embeddings)을 통합하여 다양한 크기의 객체를 생성합니다.

- **Performance Highlights**: 모델은 다양한 품질 지표와 사용자 연구에서 기존의 이미지 합성 및 객체 배치 예측 모델을 초월한 성능을 보여주었습니다. 효과적인 그림자와 반사 생성에 대한 모델의 능력은 이미지의 리얼리즘을 크게 향상시킵니다.



### The role of data embedding in quantum autoencoders for improved anomaly detection (https://arxiv.org/abs/2409.04519)
Comments:
          8 pages, 5 figures, 4 tables

- **What's New**: 이 연구는 Quantum Autoencoders (QAEs)의 이상 탐지 성능에 대한 데이터 임베딩 기술의 영향을 조사합니다. 데이터 재업로드(data re-uploading), 병렬 임베딩(parallel embedding), 대체 임베딩(alternate embedding) 등의 방법이 QAEs의 효과성과 표현력에 미치는 영향을 비교 분석합니다.

- **Technical Details**: QAEs는 양자 회로(quantum circuits)를 통해 정상 데이터의 압축된 표현을 학습하며, 두 가지 중요한 구성요소인 ansatz 아키텍처와 데이터 임베딩 방법의 선택이 모델의 성능에 큰 영향을 미친다는 것을 강조합니다. 이 연구는 다양한 임베딩 기술과 ansatz 아키텍처의 비교를 통해 QAEs의 성능을 최적화하는 방법을 설명합니다.

- **Performance Highlights**: 기본적인 변형 회로(variational circuits)를 사용하면서도 향상된 데이터 임베딩 전략을 통해 여러 데이터셋에 대해 이상 탐지 정확도를 상당히 개선할 수 있음을 보여주었습니다. 2D 및 고차원 데이터셋 모두에서의 성능 차이를 명확히 시각적으로 증명하였습니다.



### Learning to Solve Combinatorial Optimization under Positive Linear Constraints via Non-Autoregressive Neural Networks (https://arxiv.org/abs/2409.04495)
Comments:
          English version of the same paper published on Scientia Sinica Informationis

- **What's New**: 본 논문에서는 비자율 회귀 신경망(non-autoregressive neural networks)을 활용하여 조합 최적화(Combinatorial Optimization, CO) 문제를 정밀하게 해결하는 방법을 제시하고 있습니다. 이 방법은 기존의 자율 회귀 모델과의 차별점을 강조하며, 특정한 양의 선형 제약조건 아래에서 작동하는 솔버를 개발하였습니다.

- **Technical Details**: 제안된 방법은 Linear Satisfiability Network (LinSATNet)을 통해 신경망 출력을 양의 선형 제약 조건에 맞추어 프로젝션함으로써, 다양한 CO 문제를 다룰 수 있습니다. 문제는 그래프 신경망(Graph Neural Network, GNN)을 통해 처리되며, Gumbel 재매개화(Gumbel reparameterization)를 활용하여 네트워크 출력을 근사적으로 이산화(discrete)하고 해결 가능한(solution feasible) 형태로 합니다.

- **Performance Highlights**: 제안된 비자율 회귀 신경 솔버는 시설 위치(Facility Location), 최대 집합 커버링(Max-Set Covering), 외판원 문제(Traveling Salesman Problem) 등의 대표적인 CO 문제에서 기존의 SCP 및 Gurobi와 비교하여 높은 효율성과 솔루션 품질을 보여주었습니다. 이 솔버는 특히 효율성과 효과성을 동시에 고려할 때 우수한 성능을 발휘합니다.



### The Current and Future Perspectives of Zinc Oxide Nanoparticles in the Treatment of Diabetes Mellitus (https://arxiv.org/abs/2409.04486)
Comments:
          21 pages, 1 figure. Includes comprehensive review of synthesis methods and biological evaluations of ZnO nanoparticles in diabetes treatment

- **What's New**: 이번 리뷰 논문은 당뇨병(diabetes mellitus) 치료에 사용되는 아연 산화물 나노입자(zinc oxide nanoparticles, ZnO NPs)의 합성, 특성 분석 및 치료적 응용을 탐구합니다.

- **Technical Details**: 이 연구는 화학적 합성(chemical synthesis)과 친환경 합성(green synthesis) 방법을 비교하여 나노입자 특성에 미치는 영향을 분석합니다. XRD, FTIR, UV-Vis spectroscopy, SEM과 같은 주요 특성 결정 기법(characterization techniques)이 나노입자의 결정 구조(crystalline structure), 광학적 특성(optical properties), 그리고 형태(morphology)를 확인합니다.

- **Performance Highlights**: ZnO NPs는 항균성(antibacterial), 항염증성(anti-inflammatory), 그리고 항당뇨 효과(antidiabetic effects)를 포함한 유의미한 생물학적 활성을 입증하였습니다. 이 나노입자들은 포도당 조절(glucose regulation) 개선, 인슐린 감수성(insulin sensitivity) 증대 및 세포 내 포도당 흡수(glucose uptake) 촉진에 효과적입니다. 그러나 ZnO NPs의 잠재적 독성(toxicity) 및 장기적 효과(long-term effects)에 대한 추가 연구가 필요합니다.



### Large Language Models in Drug Discovery and Development: From Disease Mechanisms to Clinical Trials (https://arxiv.org/abs/2409.04481)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 약물 발견 및 개발 분야에서의 변화된 역할을 다루고 있습니다. LLM이 질병 메커니즘을 이해하고, 약물 발견을 촉진하며, 임상 시험 과정을 최적화하는 등 다각적인 기여를 하고 있다는 점이 강조되었습니다.

- **Technical Details**: LLM은 약물 발견 프로세스의 여러 단계를 혁신하는 잠재력을 지니고 있으며, 특정 과학적 언어로 훈련된 전문화된 언어 모델과 일반적인 텍스트 언어로 훈련된 범용 언어 모델의 두 가지 주요 패러다임이 존재합니다. 이러한 모델들은 정보를 추출하고, 약물 효능 및 안전성을 예측하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: LLM은 질병 모델링, 화학 실험 자동화, 임상 시험 데이터 분석 등을 통해 약물 개발 프로세스를 가속화할 수 있는 가능성을 보여주고 있습니다. 예를 들어, Med-PaLM 모델은 임상 지식을 바탕으로 미국 의사 면허시험(USMLE)에서 인간 전문가와 동등한 성과를 기록하였습니다.



### Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Sma (https://arxiv.org/abs/2409.04478)
- **What's New**: 이번 연구에서는 고차원 희소 오토인코더(Sparse Autoencoders, SAEs)가 GPT-2 small의 숨겨진 표현을 학습하면서 도시가 어느 나라에 속하는지와 대륙에 대한 지식을 별도로 매개하는 특성을 평가하기 위해 RAVEL 벤치마크를 사용합니다.

- **Technical Details**: 연구자들은 여러 공개된 GPT-2 small용 SAEs(Open AI SAE, Apollo SAEs, Bloom SAE)를 실험하고, 각 SAEs가 특징으로 선택된 이진 마스크를 학습하여 중재(interventions) 실험을 통해 국가와 대륙 관련 지식을 평가합니다. 인터벤션 과정에서는 도시명과 관련된 특징값을 조작하여 예측 출력이 어떻게 변하는지 관찰합니다.

- **Performance Highlights**: 실험 결과, 모든 SAEs가 뉴런 기본치에 비해 성능이 저조하며, DAS 스카이라인과 비교 시에도 크게 뒤처짐을 보여줍니다. 이로 인해 SAEs가 모델 내부의 개념 매개를 찾는 데 있어 뉴런보다 유용하지 않음을 증명하였습니다.



### Revolutionizing Database Q&A with Large Language Models: Comprehensive Benchmark and Evaluation (https://arxiv.org/abs/2409.04475)
Comments:
          12 pages

- **What's New**: DQA는 데이터베이스 Q&A를 위한 최초의 포괄적인 벤치마크로, LLM 기반 방법론으로 240,000개 이상의 Q&A 쌍을 생성하였으며, 데이터베이스 지식의 다양한 측면을 포괄합니다.

- **Technical Details**: DQA는 Question Classification Routing (QCR), Retrieval-Augmented Generation (RAG), Tool Invocation Generation (TIG), Prompt Template Engineering (PTE) 등의 모듈형 구성 요소로 이루어진 테스트베드를 제공합니다. 이 테스트베드는 LLM을 데이터베이스 Q&A 작업에 적응시키기 위한 기본 및 고급 모듈을 포함하고 있습니다.

- **Performance Highlights**: DQA를 이용한 평가 결과, 여러 LLM 기반 Q&A 봇의 강점과 한계, 그리고 다양한 서비스 구성 요소 (QCR, RAG, TIG)의 성능 영향 및 개선 가능성이 도출되었습니다.



### Learning in Order! A Sequential Strategy to Learn Invariant Features for Multimodal Sentiment Analysis (https://arxiv.org/abs/2409.04473)
- **What's New**: 이 연구는 다중 모달 감정 분석을 위한 비디오 및 텍스트에서 모델을 학습하는 새로운 순차적 학습 전략을 제안합니다. 이 전략은 도메인 불변 기능을 학습한 후 비디오에서 희소 도메인 요건 없는 기능을 학습하도록 설계되었습니다.

- **Technical Details**: 우리는 'S2LIF'라는 순차적 불변 기능 학습 전략을 사용해, 텍스트에서 불변 기능을 먼저 선택하고 이를 기반으로 비디오에서 관련 기능을 추출합니다. 이 방법은 복잡한 모달 간 상호작용 네트워크 없이 도메인 일반화 모델을 구축합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 단일 출처 및 다중 출처 환경 모두에서 기존 최첨단 접근 방식보다 유의미하게 더 나은 성능을 보였습니다. 선정된 기능이 상호 의존하지 않고 감정 레이블과 강한 상관관계를 가지는 특징을 보여줍니다.



### Leveraging Large Language Models for Solving Rare MIP Challenges (https://arxiv.org/abs/2409.04464)
- **What's New**: 이 논문에서는 Mixed Integer Programming (MIP) 문제 해결을 위한 새로운 접근 방식인 재귀적 동적 온도 방법을 제안합니다. 이 방법은 체인 오브 씽크(Chain-of-Thought) 접근 방식과 통합되어 있습니다.

- **Technical Details**: 기존의 온도가 낮고 일정한 환경에서는 LLMs (대형 언어 모델)이 다양한 해결책을 탐색하는 데 제한이 있었으나, 높은 온도에서 시작해 점차 낮추는 재귀적 방법을 통해 보다 나은 해를 제공함을 입증했습니다.

- **Performance Highlights**: LLMs는 Gurobi와의 비교를 통해 전통적인 솔버의 가지치기 과정을 가속화하고 전반적인 효율성을 개선하여 보완적인 결과를 도출해낼 수 있음을 보여주었습니다.



### Pattern based learning and optimisation through pricing for bin packing problem (https://arxiv.org/abs/2409.04456)
- **What's New**: 본 논문에서는 데이터 마이닝의 패턴 식별 과정에서 패턴의 동적 가치에 대한 체계적인 분석을 진행합니다. 특히 랜덤 변수의 분포와 같은 문제 조건이 변화할 때 패턴의 효과가 떨어질 수 있음을 강조합니다.

- **Technical Details**: 본 연구는 운영 연구(operations research)의 쌍대성 이론(duality theory)과 데이터 마이닝을 연결하여 패턴을 효율적으로 식별하고 조건에 따라 동적으로 그 가치를 정량화하는 새로운 방법론을 제안합니다. 이 방법은 패턴이 확률적 제약(stochastic constraints)을 만족시키는 능력과 목적 함수(objective value) 미치는 영향을 바탕으로 패턴의 가치를 정량화합니다.

- **Performance Highlights**: 제안된 알고리즘은 온라인 빈 포장 문제(online bin packing problem)를 통해 기존의 최첨단 방법들(state-of-the-art methods)보다 우수한 성과를 보이며, 제안된 방법의 고유 특성과 성능 개선 사항을 상세히 분석하였습니다.



### Process Trace Querying using Knowledge Graphs and Notation3 (https://arxiv.org/abs/2409.04452)
- **What's New**: 본 논문에서는 이벤트 로그(Event Log)를 기반으로 한 지식 그래프(Knowledge Graph, KG)를 생성하고 이를 활용하여 이벤트 탐색(log exploration)을 지원하는 방법을 제시합니다. 이 방법은 RDF(리소스 기술 프레임워크)와 N3(표기법 3)를 사용하여 그래프 데이터의 쿼리를 가능하게 합니다.

- **Technical Details**: 이벤트 로그를 기반으로 하는 이벤트 로그 지식 그래프(Event Log KG, ELKG)는 케이스 중심(event logs) 및 객체 중심(event logs) 로그를 트레이스 기반의 세멘틱 KG로 변환합니다. 이 방법은 N3를 활용하여 이벤트 간의 관계(관계 제약조건)를 정의하고, 다수의 쿼리 패턴을 지원합니다. OCEL2 로그는 객체 경로를 통해 트레이스로 ‘평탄화(flatten)’되어 표현됩니다.

- **Performance Highlights**: 제안된 방법은 표현력(expressivity), 유연성(flexibility), 확장성(extensibility)을 갖추고 있습니다. 따라서 다양한 방식으로 쿼리를 생성할 수 있으며, 사용자 맞춤형 제약 조건을 구현할 수 있는 유용성을 제공합니다.



### Leveraging Contrastive Learning and Self-Training for Multimodal Emotion Recognition with Limited Labeled Samples (https://arxiv.org/abs/2409.04447)
Comments:
          Accepted by ACM MM Workshop 2024

- **What's New**: 2024년 멀티모달 감정 인식 대회(MER2024)는 오디오, 언어 및 시각 신호를 사용하여 감정을 인식하는 데 초점을 맞추고 있습니다. 본 논문에서는 한정된 주석 데이터 문제를 해결하기 위한 준지도 학습 서브챌린지(MER2024-SEMI)에 대한 제출 솔루션을 제시합니다.

- **Technical Details**: 우리는 클래스 불균형 문제를 해결하기 위해 오버샘플링(oversampling) 전략을 채택하고, 세 가지 모달리티(모드: trimodal) 입력 데이터를 활용하는 모달리티 표현 조합 대비 학습 프레임워크(MR-CCL)를 제안합니다. 또한, 자기 훈련(self-training) 방법을 탐색하여 훈련 세트를 확장하고, 다중 분류기를 사용하는 가중 소프트 보팅(soft voting) 전략으로 예측의 강인성을 향상시킵니다.

- **Performance Highlights**: 우리의 제안된 방법은 MER2024-SEMI 챌린지에서 효과적으로 검증되었으며, 가중 평균 F-점수(weighted average F-score)에서 88.25%를 달성하였고, 리더보드에서 6위에 랭크되었습니다.



### Intelligent tutoring systems by Bayesian nets with noisy gates (https://arxiv.org/abs/2409.04102)
- **What's New**: 이 논문에서는 Bayesian 네트워크를 기반으로 한 지능형 튜터링 시스템(ITS)의 조건부 확률 테이블(parametrization)을 간단하게 하기 위해 불확실성을 가진 논리 게이트(logical gates with uncertainty)를 사용하는 방법을 제안합니다.

- **Technical Details**: 주요 내용으로는, ITS에서 많은 수의 파라미터를 요구하는 문제를 해결하기 위해 noisy-OR 게이트를 사용하여 파라미터 수를 줄이고, 이들 사이의 연결을 그래픽적으로 표현하는 방법이 있습니다. 논문은 이러한 모델 파라미터의 의미와 이들에 적용하기 위한 가정들을 논의하고, 고속 계산을 위한 전용 추론(inference) 방법도 도출합니다.

- **Performance Highlights**: 이러한 접근법은 파라미터의 수를 기하급수적으로 줄여 선형적으로 만들며, 그에 따라 추론의 복잡성도 감소하여 실시간 피드백(real-time feedback)이 가능하게 됩니다.



### Real-time Speech Enhancement on Raw Signals with Deep State-space Modeling (https://arxiv.org/abs/2409.03377)
Comments:
          7 pages, 2 figures

- **What's New**: 이번 연구에서 우리는 aTENNuate라는 단순한 딥 상태 공간 자동 인코더를 제시합니다. 이 모델은 효율적인 온라인 원시 음성 향상을 위해 설계되었습니다. aTENNuate는 원시 음성 잡음 제거의 성능 평가를 주요 대상으로 하고 있으며, 추가적으로 슈퍼 해상도(super-resolution) 및 양자화 해제(de-quantization) 작업에 대한 평가도 포함되어 있습니다.

- **Technical Details**: aTENNuate 네트워크는 Temporal Neural Networks (TENNs) 클래스에 속하며, 원시 음성 파형의 실시간 잡음 제거를 위해 최적화된 딥 상태공간 모델입니다. 이 모델은 음성 신호에 존재하는 장기 Temporal 관계(long-range temporal relationships)를 캡처할 수 있는 안정적인 선형 반복 유닛(linear recurrent units)을 사용합니다. 훈련 동안에는 SSM 레이어의 무한 임펄스 응답(IIR) 커널을 활용하여 입력 기능에 대한 긴 합성곱 커널을 사용할 수 있으며, FFT 합성곱 등과 같은 기법을 통해 병렬화할 수 있습니다.

- **Performance Highlights**: aTENNuate는 VoiceBank + DEMAND 및 Microsoft DNS1 합성 테스트 세트에서 이전의 실시간 잡음 제거 모델보다 우수한 PESQ 점수를 기록했습니다. 이 모델은 잡음이 4000Hz 및 4비트로 압축되어도 높은 충실도를 유지하며, 최적의 성능을 발휘합니다. 또한, 모델의 코드가 공개되어 있어 연구 및 개발에 활용될 수 있습니다.




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




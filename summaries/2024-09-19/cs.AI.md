New uploads on arXiv(cs.CL)

### Gender Representation and Bias in Indian Civil Service Mock Interviews (https://arxiv.org/abs/2409.12194)
- **What's New**: 이번 연구는 888개의 YouTube 모의 인터뷰 비디오에서 수집한 51,278개의 질문을 통해 남녀 지원자에게 제시되는 질문의 성차별적 편향을 드러낸다. 또한, 대규모 언어 모델(LLMs)을 활용한 성별 추론 실험을 통해 성별 추론 작업에서 LLM이 보이는 성차별적 편향을 분석하였다.

- **Technical Details**: 본 연구는 UPSC(Union Public Services Commission) 모의 인터뷰 질문을 분석하여 성별에 따른 질문의 차이를 평가하였다. 분석 근거는 14개의 유명한 코칭 기관에서 발표한 모의 인터뷰 비디오에 기반하며, 질문의 데이터셋은 4.5백만 토큰으로 구성되어 있다.

- **Performance Highlights**: 여성 지원자는 성평등이나 가족에 대한 질문을 받을 가능성이 남성보다 거의 세 배 더 높은 반면, 남성은 국제 문제, 세계 정치, 스포츠에 대한 질문을 더 많이 받는다. 이는 성별에 따른 스테레오타입이 뚜렷하게 나타나는 결과이다.



### Qwen2.5-Coder Technical Repor (https://arxiv.org/abs/2409.12186)
- **What's New**: Qwen2.5-Coder 시리즈는 CodeQwen1.5의 주요 업그레이드로, 두 가지 모델 Qwen2.5-Coder-1.5B 및 Qwen2.5-Coder-7B로 구성된 코드 특정 모델입니다. 이 모델은 5.5 조 개 이상의 토큰으로 사전 학습되었으며, 뛰어난 코드 생성 기능을 보여주고 있습니다.

- **Technical Details**: Qwen2.5-Coder 모델은 Qwen2.5 아키텍처에 기반하며, 1.5B와 7B 파라미터를 가진 두 가지 모델로 제공됩니다. 28개의 레이어와 128의 헤드 크기를 공유하지만, 1.5B 모델은 1,536의 숨겨진 크기를, 7B 모델은 3,584의 숨겨진 크기를 갖습니다. 또한 다양한 코드 데이터를 처리하기 위해 특수 토큰이 추가되었습니다.

- **Performance Highlights**: Qwen2.5-Coder 모델은 10개 이상의 벤치마크에서 SOTA 성능을 달성하였으며, 코드 생성, 완성, 추론 및 수리를 포함한 여러 작업에서 동일한 모델 크기의 더 큰 모델들보다 일관되게 우수한 성능을 보였습니다.



### To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning (https://arxiv.org/abs/2409.12183)
- **What's New**: 본 논문에서는 Chain-of-thought (CoT) 기법이 대형 언어 모델(LLMs)에서의 추론 능력을 이끌어내는 데 어떤 작업에서 효과적인지 분석하기 위해 100편 이상의 문헌에 대한 메타 분석을 수행하였으며, 14개 모델과 20개 데이터셋에서 자체 평가를 진행했습니다.

- **Technical Details**: 논문 결과에 따르면, CoT는 주로 수학적 또는 논리적 추론과 관련된 작업에서 성능 향상을 가져오며, 기타 작업에서는 성과가 미미합니다. CoT는 두 가지 처리 단계인 계획 단계와 실행 단계로 나누어 분석하였고, CoT는 주로 실행 단계에서 효용이 있음을 발견했습니다. 그러나 CoT는 도구를 사용한 LLM보다 성능이 떨어지는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 CoT의 유용성이 도구 보완에 의해 한정되어 있음을 보여주며, CoT가 매우 복잡한 문제에는 필요한 경우가 있지만, 전반적으로 더 효율적인 프롬프트 전략이 존재함을 강조합니다. 아울러 CoT를 넘어서는 새로운 패러다임을 탐색할 필요성을 제기하였습니다.



### A Controlled Study on Long Context Extension and Generalization in LLMs (https://arxiv.org/abs/2409.12181)
- **What's New**: 이 연구는 전체 문서 컨텍스트를 활용하는 언어 모델의 긴 컨텍스트 성능을 평가하기 위한 통제된 프로토콜을 구현하였습니다. 다양한 컨텍스트 확장 방법을 표준화된 평가를 통해 비교함으로써, 긴 컨텍스트 작업에서의 모델 성능에 대한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 표준화된 기준 모델을 사용하며, 세 가지 주요 확장 방법인 exact attention, approximate attention, context compression을 다룹니다. 평가 지표로는 perplexity와 downstream task performance을 사용하여, 효과적인 컨텍스트 확장을 위한 방법론을 제시합니다.

- **Performance Highlights**: 정확한 fine-tuning 기반 방법들이 일반적으로 효과적이며, 현재의 approximate attention 방식은 긴 컨텍스트 작업에서 일관되게 성능이 저하되는 경향이 있습니다. Dynamic NTK를 사용한 방법이 특히 효과적이며, 기존의 방법들에 비해 더 나은 성능을 보였습니다.



### Finetuning Language Models to Emit Linguistic Expressions of Uncertainty (https://arxiv.org/abs/2409.12180)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 생성 예측에 불확실성 언어 표현을 추가하는 방법에 대해 탐구합니다. LLM이 특정 예측에 대한 자신감을 표현하지 않기 때문에 사용자들은 모델의 정확성과 자신감을 일치시키기 어려운 문제를 해결하고자 합니다.

- **Technical Details**: 연구자들은 다양한 질문-응답 데이터셋에서 실험을 통해 사전 훈련된 모델의 보정(calibration)을 측정하고, 감독된 마무리 조정(supervised finetuning)을 통해 불확실성 언어 표현을 생성하는 방법을 제시합니다. 결과적으로, 이 모델들은 신뢰도가 높고 잘 보정된 불확실성 표현을 생성합니다.

- **Performance Highlights**: 연구 결과, LLMs는 자신의 예측을 평가하는 데에 있어서 잘 보정된 것으로 나타났으며, 모델의 자체 신뢰도를 기반으로 한 감독된 마무리 조정이 특히 단일 주장(single-claim) 응답에 대해 잘 보정된 불확실성 표현을 이끌어내었습니다.



### You Only Read Once (YORO): Learning to Internalize Database Knowledge for Text-to-SQL (https://arxiv.org/abs/2409.12172)
- **What's New**: 새로운 패러다임인 You Only Read Once (YORO)를 소개하여, 데이터베이스 스키마를 인코딩할 필요 없이 텍스트-투-SQL 모델에서 데이터베이스 지식을 직접 내재화하여 학습을 진행하고, 이로 인해 입력 토큰 길이를 획기적으로 단축시킴. 이 접근법은 각 질문에 대해 동일한 스키마를 반복하여 인코딩하는 전통적 방식의 비효율성을 해결.

- **Technical Details**: YORO는 데이터베이스 이해 및 지식 습득 단계를 포함하여, 질문 인식 및 SQL 생성에 초점을 맞춘 인퍼런스 단계로 구성됨. 이 모델은 대상 데이터베이스에 대한 고품질 NLQ-SQL 쌍을 합성하여 교육이 이루어지며, 이러한 방식으로 생성된 모델을 사용하면, 입력에서 스키마 정보를 필요로 하지 않음.

- **Performance Highlights**: YORO는 기존 모델보다 입력 길이를 66%-98% 단축시키면서도 다양한 벤치마크에서 경쟁력 있는 성능을 발휘. 특히, YORO는 대규모 데이터베이스에서 뛰어난 성능을 보여주며, 약어와 같은 어려운 값 검색이 필요한 질문에서도 우수한 성과를 나타냄.



### MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning (https://arxiv.org/abs/2409.12147)
Comments:
          22 pages, code: this https URL

- **What's New**: 본 논문에서는 MAgICoRe라는 새로운 프레임워크를 제안하고 있으며, 이는 대규모 언어 모델(Large Language Models, LLM)이 정답률을 높이고, 과도한 수정, 오류 식별의 어려움, 그리고 수정 횟수 부족 문제 등을 해결하기 위한 방법론입니다.

- **Technical Details**: MAgICoRe는 문제의 난이도에 따라 쉬운 문제는 대강의 집계를 사용하고, 어려운 문제는 세밀한 다중 에이전트 수정을 사용하여 해결합니다. 외부 리워드 모델(Reward Model, RM)의 단계별 점수를 통합하여 오류를 더욱 효과적으로 식별하고 개선합니다. 세 가지 에이전트인 Solver, Reviewer, Refiner를 통해 원활한 피드백과 수정을 진행합니다.

- **Performance Highlights**: MAgICoRe는 Llama-3-8B 및 GPT-3.5에서 5개의 수학 데이터셋에 대해 평균적으로 Self-Consistency와 Best-of-k 샘플링 방법보다 각각 3.4% 및 3.2% 더 높은 성과를 보였습니다. 또한, 수정 횟수가 증가함에 따라 성능이 꾸준히 개선되었으며, 전체 성과에서 6.4%까지 향상된 결과를 기록했습니다.



### GRIN: GRadient-INformed MoE (https://arxiv.org/abs/2409.12136)
Comments:
          58 pages

- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 모델의 훈련을 개선하기 위해 새로운 방법론인 GRIN (GRadient-INformed MoE training)을 도입하였습니다. 이는 expert routing을 위한 희소 그라디언트 추정과 모델 병렬성을 지원하여 토큰 드롭 문제를 해결합니다.

- **Technical Details**: GRIN 방법론은 SparseMixer-v2 아키텍처를 활용하여 Top1 및 TopK MoE의 훈련을 최적화합니다. 특히, MaskedSoftmax 함수를 사용하여 explicit expert sampling을 수행하며, 이 과정에서 소개된 추가 하이퍼파라미터 r은 샘플링 공간의 무작위성 및 희소성을 조절합니다. 이러한 기법들은 기존의 jitter noise를 대체하여 MoE의 성능을 향상시킵니다.

- **Performance Highlights**: GRIN을 활용하여 개발된 16x3.8B MoE 모델은 7B 밀집 모델보다 뛰어난 성능을 발휘하였으며, 동일한 데이터를 학습한 14B 밀집 모델과 동등한 성취를 보였습니다. MMLU에서 79.4, HellaSwag에서 83.7, HumanEval에서 74.4, MATH에서 58.9의 성과를 달성하여 MoE의 효율성 향상을 입증하였습니다.



### BERT-VBD: Vietnamese Multi-Document Summarization Framework (https://arxiv.org/abs/2409.12134)
Comments:
          10 pages

- **What's New**: 이 논문은 베트남어 다문서 요약(Multi-Document Summarization, MDS) 문제를 해결하기 위해 추출 기반(extractive) 및 생성 기반(abstractive) 요약 기법을 통합하는 새로운 프레임워크를 제안합니다. 기존의 접근 방식이 갖고 있는 한계를 극복하기 위해 두 개의 구성 요소로 이루어진 파이프라인 아키텍처를 활용하여, 성능을 향상시키는 가능성을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 구성 요소로 나뉩니다. 첫 번째는 수정된 사전 학습 BERT(Bidirectional Encoder Representations from Transformers) 네트워크를 활용하여 각 문서 내에서 핵심 문장을 찾아내는 추출적 요약입니다. 두 번째는 VBD-LLaMA2-7B-50b 모델을 사용하여 이 추출된 문장을 기반으로 생성적 요약을 수행하여 최종 요약 문서를 생성합니다. 이 과정에서 Sentence-BERT(SBERT)와 VBD-LLaMA2-7B-50b 같은 딥러닝 모델을 사용합니다.

- **Performance Highlights**: 제안된 프레임워크는 VN-MDS 데이터셋에서 ROUGE-2 점수 39.6%를 달성하며, 기존의 최첨단 모델들보다 성능이 우수함을 보여주었습니다. BET와 같은 모델을 활용한 기존 연구들과 비교할 때, 제안된 방법이 베트남어에서의 요약 품질을 현저히 향상시킨다는 것을 입증했습니다.



### Linguini: A benchmark for language-agnostic linguistic reasoning (https://arxiv.org/abs/2409.12126)
- **What's New**: 언어 모델의 언어적 추론 능력을 측정할 새로운 벤치마크인 Linguini를 제안합니다. 이 테스트는 75개의 (대부분) 자원이 거의 없는 언어로 그룹화된 894개의 질문을 포함하며, 국제 언어 올림피아드(IOL) 데이터셋에서 추출되었습니다.

- **Technical Details**: Linguini 벤치마크는 기존 언어 지식 없이도 언어 문제를 해결할 수 있는 언어 모델의 메타 언어 인식과 추론 능력을 측정합니다. 모든 문제는 주어지는 맥락 내에서 해결할 수 있도록 구성되어 있으며, 분석된 모델들은 모두 25% 이하의 정확도를 보였습니다. 최상의 성능을 기록한 독점 모델은 24.05%, 오픈 모델은 8.84%의 정확도를 보였습니다.

- **Performance Highlights**: Linguini 벤치마크에서 언어 모델의 성능은 저조하며, 오픈 모델과 독점 모델 간의 성능 차이가 뚜렷합니다. 독점 모델이 훨씬 더 높은 성능을 보였으나 모든 모델이 주어진 문제를 해결하는 데 필요한 정보를 요청된 언어에 대한 사전 지식 없이도 사용할 수 있도록 설계되었습니다.



### Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvemen (https://arxiv.org/abs/2409.12122)
- **What's New**: 이번 논문에서는 Qwen2.5-Math 및 Qwen2.5-Math-Instruct 1.5B/7B/72B의 수학 전용 대형 언어 모델 시리즈를 소개합니다. Qwen2.5 시리즈의 핵심 혁신은 자가 개선(self-improvement) 철학을 전 과정에 통합했다는 점입니다.

- **Technical Details**: 이 모델들은 사전 학습(pre-training), 후속 학습(post-training) 및 추론(inference) 과정에서 Qwen2-Math-Instruct를 사용하여 대규모 수학 데이터 생성을 지원합니다. 후속 학습 단계에서는 Qwen2-Math-Instruct에서 대규모 샘플링을 통해 보상 모델(reward model, RM)을 개발하고, 이는 감독 세부 조정(Supervised Fine-Tuning, SFT) 과정에서 데이터의 반복적인 진화를 주도합니다. 마지막으로 Qwen2.5-Math-Instruct는 강화 학습에 최적의 RM을 사용하여 재학습하며, 영어와 중국어 모두 지원하여 고급 수학적 추론 능력을 갖추고 있습니다.

- **Performance Highlights**: Qwen2.5-Math-7B 모델은 GSM8K, MATH, GaoKao Math Cloze에서 각각 91.6, 55.4, 57.6의 점수를 기록하며, Qwen2-72B 모델을 초월했습니다. 특히 Qwen2.5-Math-72B 모델은 MATH 벤치마크에서 66.8점을 달성하여 새로운 최고 기록을 세우며, CoT 모드에서는 Qwen2.5-Math-1.5B-Instruct 모델이 현재 사용 가능한 모든 오픈 소스 모델의 성능을 초과합니다.



### Measuring Human and AI Values based on Generative Psychometrics with Large Language Models (https://arxiv.org/abs/2409.12106)
- **What's New**: 이 논문에서는 Generative Psychometrics for Values (GPV)라는 LLM 기반의 데이터 구동 가치 측정 패러다임을 소개합니다. 기존의 심리 측정 도구들을 개선하여 LLM의 고급 의미 이해를 활용하여 텍스트에서 맥락적이고 가치가 있는 인식을 추출할 수 있습니다.

- **Technical Details**: GPV는 Llama 3(즉, ValueLlama)로 미세 조정되어 있으며, 텍스트를 인식으로 분해하는 LLM의 능력을 검증합니다. 이 과정은 GPV의 핵심 파이프라인을 형성하며 대량의 인간 작성 블로그에 적용하여 안정성 및 타당성을 검증합니다.

- **Performance Highlights**: GPV는 이전의 심리 도구들에 비해 더 우수한 성능을 보여주며, 17개의 LLM과 4개의 가치 이론에 대한 광범위한 평가를 통해 유효성 및 효용성 면에서 훌륭한 측정 결과를 얻었습니다.



### Skill matching at scale: freelancer-project alignment for efficient multilingual candidate retrieva (https://arxiv.org/abs/2409.12097)
- **What's New**: 이 논문은 다국어 환경에서 프리랜서와 프로젝트 간의 매칭을 개선하기 위한 새로운 신경망 검색 아키텍처를 제안합니다. 기존 시스템의 한계를 극복하고 보다 효율적인 매칭을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 다국어 언어 모델을 활용하여 프로젝트 설명 및 프리랜서 프로필을 인코딩합니다. 사용자 맞춤형 transformer 아키텍처를 통해 프로필과 프로젝트 구조를 유지하며, 과거 데이터를 활용하여 대조 손실(contrastive loss)로 훈련합니다. 이 모델은 프리랜서의 기술 매칭 유사성을 신뢰성 있게 포착합니다.

- **Performance Highlights**: 여러 실험 결과, 제안된 접근 방식이 기존의 전통적인 방법들보다 우수한 성능을 보여주었으며, 더 나은 기술 매칭 효율성을 제공함을 입증했습니다.



### PARAPHRASUS : A Comprehensive Benchmark for Evaluating Paraphrase Detection Models (https://arxiv.org/abs/2409.12060)
- **What's New**: 이 논문은 paraphrasus라는 새로운 벤치마크를 제안하여, 다양한 측면에서 패러프레이즈 감지 모델을 평가하고 세분화된 모델 선택을 가능하게 합니다. 이를 통해 기존의 단일 분류 데이터셋으로는 포착할 수 없는 성능의 트레이드오프를 드러냅니다.

- **Technical Details**: paraphrasus는 인간이 작성한 문장 쌍을 포함한 데이터셋을 사용하여 패러프레이즈 감지 모델을 평가합니다. 이 벤치마크에는 의미적 및 어휘적 유사성이 다양한 두 가지 새로운 데이터셋이 포함되어 있습니다. 첫 번째 데이터셋은 패러프레이즈 분류를 위한 338개의 테스트 집합을 주석 처리하였고, 두 번째는 진정한 패러프레이즈의 전문가 예시를 추출한 것입니다.

- **Performance Highlights**: 패러프러스 벤치마크를 통해 LLMs와 훈련된 모델을 다양한 설정 하에 테스트하여, 기존 패러프레이즈 데이터에 대한 평가의 한계를 명확히 하였습니다. 특히, 단일 패러프레이즈 데이터셋에서 평가하는 것이 실제 일반화 성능에 대한 오해를 초래할 수 있음을 발견했습니다.



### Dual-Layer Training and Decoding of Large Language Model with Simultaneously Thinking and Speaking (https://arxiv.org/abs/2409.12059)
Comments:
          9 pages, 5 figures

- **What's New**: 이 논문은 기존의 데이터 기반 또는 훈련 기반 접근 방식이 아닌, 자연 세계의 인지 기제를 영감을 받아 TaS라는 새로운 모델 아키텍처를 설계하였습니다. 이 모델은 먼저 생각을 고려한 후 질의에 대한 응답을 생성합니다.

- **Technical Details**: TaS 모델은 프롬프트-응답 샘플에서 생각의 내용을 주석 달거나 생성하는 여러 파이프라인을 설계하며, 중간 계층에 언어 헤드를 추가하여 사고 계층(thinking layer)으로 작동합니다. 이 모델은 생각이 증가된 데이터(thoughts-augmented data)로 언어 모델을 훈련시킵니다.

- **Performance Highlights**: TaS의 효과성과 성능을 정성적 예제(qualitative examples)와 정량적 결과(quantitative results)로 입증하였습니다. 모델의 코드도 공개되었습니다.



### Using Large Language Models to Generate Clinical Trial Tables and Figures (https://arxiv.org/abs/2409.12046)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)을 활용하여 임상 시험 데이터를 자동으로 요약하기 위한 TFL(테이블, 그림, 및 리스트) 생성을 탐구하였습니다. 특히 프롬프트 엔지니어링 및 몇 가지 샷 전달 학습을 통해 LLM이 효율적으로 TFL을 생성할 수 있음을 보여주며, 이 분야에서의 가능성을 제시하였습니다.

- **Technical Details**: 연구진은 ClinicalTrials.gov에서 제공하는 암 시험의 적격성 기준에 대한 자동화된 분류 모델을 개발하였습니다. 총 764개의 주석이 달린 시험을 분석하여 키워드 매칭을 사용해 특정 기준 요구 사항을 전달하는 문장을 추출하였고, 다섯 가지의 공개 도메인 특정 모델로 훈련 데이터를 보강하여 학습한 결과, 최고 F1 점수가 0.95 이상으로 나타났습니다.

- **Performance Highlights**: 연구 결과, 자동화된 분류 모델을 통해 암 임상 시험의 적격성 기준을 효과적으로 식별할 수 있으며, 일부 기준에서는 기존 모델보다 향상된 성능을 보였습니다. 이는 향후 임상 시험의 적격성 기준을 연구하는 데 있어 큰 도움이 될 것입니다.



### ASR Benchmarking: Need for a More Representative Conversational Datas (https://arxiv.org/abs/2409.12042)
- **What's New**: 이번 연구에서는 TalkBank에서 유래된 다국어 대화 데이터셋을 소개합니다. 이 데이터셋은 성인 간의 비구조적 전화 대화로 구성되어 있으며, 실제 대화 환경의 복잡성을 반영하고 있습니다.

- **Technical Details**: TalkBank 데이터셋을 기반으로 하여 실시간 대화에서의 Auitomatic Speech Recognition (ASR) 모델의 성능을 평가합니다. 주요 전처리 과정에는 음성 활동 탐지(Voice Activity Detection, VAD), 전사 정렬, 및 일치하지 않는 녹음을 필터링하는 과정이 포함됩니다.

- **Performance Highlights**: 대화 환경에서 실험한 결과, 다양한 최신 ASR 모델들이 기존 벤치마크에서 경험했던 성능을 크게 떨어뜨렸습니다. 예를 들어, CANARY 모델은 LibriSpeech에서 0.19의 Word Error Rate (WER)를 기록했으나 TalkBank에서 0.54로 증가하였습니다.



### Sampling Latent Material-Property Information From LLM-Derived Embedding Representations (https://arxiv.org/abs/2409.11971)
Comments:
          10 pages, 7 figures

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 생성하는 벡터 임베딩이 소재의 성질에 대한 정보를 얼마나 잘 캡처할 수 있는지를 탐구합니다. LLM에서 파생된 임베딩을 활용하여 산업 분석 및 예측에 유용한 정보를 얻는 가능성을 조사합니다.

- **Technical Details**: 연구는 Llama 2 모델을 사용하여 13억 개의 매개변수를 가진 대규모 언어 모델의 출력 임베딩을 분석합니다. 이 모델은 고차원 벡터 임베딩을 생성하는 언어 모델링 작업을 통해 구축됩니다. 화학적 조성을 기반으로 한 컴파운드-specific 임베딩을 생성하기 위한 최소한의 전략을 제안합니다.

- **Performance Highlights**: 임베딩의 품질을 평가하기 위해 코사인 유사성을 기반으로 실험을 수행하였으며, LLM이 생성한 임베딩이 화학 물질의 특성을 예측하는 데 의미 있는 역할을 할 수 있음을 시사합니다.



### Efficacy of Synthetic Data as a Benchmark (https://arxiv.org/abs/2409.11968)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)을 사용해 생성한 합성 데이터가 다양한 자연어 처리(NLP) 작업의 벤치마크로서 어떻게 효과적인지를 조사합니다. 특히, 합성 데이터가 간단한 작업에서는 성능을 잘 나타내지만, 복잡한 작업에서는 그 효력에 한계가 있음을 보여주고 있습니다.

- **Technical Details**: 여섯 개의 데이터셋과 세 가지 작업(의도 탐지, 텍스트 유사성, 명명된 개체 인식(NER))을 통해 합성 데이터의 효과성을 분석했습니다. 또한, 동일한 LLM이 데이터를 생성하고 작업을 수행할 때 발생할 수 있는 편향을 측정하기 위한 새로운 지표인 'bias factor'를 제안했습니다. 실험 결과, 더 작은 LLM은 자신이 생성한 데이터에서 편향을 보이는 반면, 더 큰 모델은 그러한 경향을 보이지 않습니다.

- **Performance Highlights**: 합성 데이터는 의도 탐지와 같은 간단한 작업에서는 효과적인 반면, 명명된 개체 인식(NER)과 같은 복잡한 작업에서는 그 대표성이 떨어진다는 결과를 얻었습니다. 여러 LLM에서 생성한 합성 데이터를 평균화하면 보다 강력하고 대표적인 벤치마크를 제공할 수 있으며, 연구에 따르면 큰 LLM은 적은 편향을 보이는 반면, 작은 모델은 자신이 생성한 데이터에서 상대적으로 더 나은 성능을 나타냅니다.



### LLMs in Education: Novel Perspectives, Challenges, and Opportunities (https://arxiv.org/abs/2409.11917)
Comments:
          COLING 2025 Tutorial

- **What's New**: 이 논문은 교육에서 대규모 언어 모델(LLMs)의 역할과 가능성을 탐구하며, LLMs가 국면을 전환한 교육 응용 분야에 초점을 맞추고 있습니다. 이는 특히 독서, 쓰기 및 말하기 능력, 그리고 지능형 튜터링 시스템(ITS) 등 네 가지 주요 교육 응용 프로그램에 대한 논의를 포함하고 있습니다.

- **Technical Details**: 본 튜토리얼은 LLMs와 자연어 처리(NLP)의 융합을 통해 교육적 응용의 기술적 세부 사항을 다룹니다. 특히, 문법 오류 수정(Grammatical Error Correction, GEC), 가독성 평가, 텍스트 단순화와 같은 가장 최신 연구를 체계적으로 설명하며, LLMs 사용의 주요 접근 방법 및 평가 방법론을 논의합니다. 또한, 자동화된 말하기 평가 방법에 대해서도 다룹니다.

- **Performance Highlights**: 대규모 언어 모델은 글쓰기 지원과 피드백 개선, 그리고 읽기 및 말하기 교육의 혁신적인 접근 방식을 제공하며, 사용자 맞춤형 학습 기회를 제공합니다. GEC 및 발음 평가에서 LLMs의 성능은 전반적으로 향상되었으며, 이는 교육과 평가에 있어 실질적인 이점을 더욱 확장하고 있습니다.



### LLMs + Persona-Plug = Personalized LLMs (https://arxiv.org/abs/2409.11901)
- **What's New**: 이 논문에서는 사용자별 언어 모델 개인화를 위한 새로운 접근 방식을 제안합니다. 기존의 방법들이나 Fine-tuning이 아닌, 사용자 행동을 반영한 사용자 특화 임베딩을 생성하여 LLMs에게 입력합니다.

- **Technical Details**: 이 모델은 Plug-and-Play 방식으로 작동하며, 사용자의 모든 역사적 맥락을 모델링하여 능률적인 Plug-in 사용자 임베더 모듈을 통해 사용자별 임베딩을 생성합니다. 이 임베딩은 LLM의 입력에 추가되어 사용자의 습관과 선호를 이해하도록 도와줍니다.

- **Performance Highlights**: 실험을 통해 본 모델은 LaMP 벤치마크에서 기존의 개인화된 LLM 기법보다 1.4%에서 35.8% 향상된 성과를 보여주며, 사용자 전반의 패턴을 캡처할 수 있음을 증명했습니다.



### DocMamba: Efficient Document Pre-training with State Space Mod (https://arxiv.org/abs/2409.11887)
- **What's New**: DocMamba라는 새로운 프레임워크를 발표하여 길이가 긴 문서를 효율적으로 처리할 수 있는 선형 계산 복잡성을 달성했습니다.

- **Technical Details**: DocMamba는 상태공간모델(State Space Model, SSM) 기반으로 구성되어 있으며, 세그먼트 우선 양방향 스캔(Segment-First Bidirectional Scan, SFBS) 기법을 사용하여 연속적인 의미 정보를 포착합니다. 이 모델은 입력 길이에 비례하여 선형 복잡성(linear complexity)을 유지하면서도 글로벌 모델링 능력을 보존합니다.

- **Performance Highlights**: 실험 결과, DocMamba는 FUNSD, CORD, SORIE 등의 데이터셋에서 기존의 LayoutLMv3보다 뛰어난 성능을 보여주었으며, GPU 메모리 사용량을 최대 88.3% 절감하고 2.4배 빠른 처리 속도를 달성했습니다.



### MEOW: MEMOry Supervised LLM Unlearning Via Inverted Facts (https://arxiv.org/abs/2409.11844)
- **What's New**: 본 논문에서 제안하는 MEOW는 LLM(Unlearning) 기술을 개선하기 위해 새로운 방법으로, 메모리에서 민감한 정보를 효과적으로 제거하는 동시에 유용성을 유지할 수 있도록 설계되었습니다. 기존의 접근 방식들이 직면했던 주요 도전 과제를 극복하기 위해 gradient descent 기반의 방식을 채택하였으며, MEMO라는 새로운 메트릭을 도입하여 메모리화를 정량화합니다.

- **Technical Details**: MEOW는 gradient descent를 기반으로 하여 LLM에서 특정 데이터를 잊도록 훈련하는 과정에서 생기는 문제점을 해결합니다. 메트릭 MEMO는 LLM에서의 메모리화를 측정하기 위해 설계되었으며, 사용자는 사용 사례 기반으로 특정 반대 사실(inverted facts)을 생성하고, 이를 통해 모델을 미세 조정(fine-tune)함으로써 필요 없는 정보를 잊도록 합니다.

- **Performance Highlights**: MEOW는 NLU(Natural Language Understanding) 및 NLG(Natural Language Generation) 벤치마크에서 기존 방법들보다 뛰어난 결과를 보였으며, 모델의 유용성에서 큰 손실 없이 기억해버리는 정보의 질이 크게 향상되었음을 입증했습니다. 또한, NLU 성능이 일부 데이터셋에서 약간 개선되는 결과도 나타났습니다.



### Extract-and-Abstract: Unifying Extractive and Abstractive Summarization within Single Encoder-Decoder Framework (https://arxiv.org/abs/2409.11827)
- **What's New**: 이 논문에서는 Extract-then-Abstract(추출-요약) 패러다임을 개선하기 위해 Extractive와 Abstractive summarization(추출적 및 추상적 요약) 작업을 단일 encoder-decoder 모델 내에서 함께 수행하는 새로운 ExtAbs 패러다임을 제안합니다.

- **Technical Details**: ExtAbs는 encoder의 attention mask를 non-padded token mask 대신에 saliency mask(주목성 마스크)로 교체하여 decoder가 입력의 중요한 부분에만 집중할 수 있도록 합니다. 이 접근 방식은 기존의 독립적으로 훈련된 extractor와 abstractor의 오류 누적 문제를 해결합니다.

- **Performance Highlights**: CNN/DM, Reddit, PubMed 데이터셋에서 실험을 진행한 결과, ExtAbs는 CNN/DM에서 기존 모델보다 더 나은 abstractive 성능을 보이고, Reddit와 PubMed에서는 최고의 extractive 성능을 달성하였으며, 추상적 작업에서도 기존 모델과 유사한 성능을 유지했습니다.



### The Factuality of Large Language Models in the Legal Domain (https://arxiv.org/abs/2409.11798)
Comments:
          CIKM 2024, short paper

- **What's New**: 이번 연구는 법률 분야에서 대규모 언어 모델(LLMs)을 지식 기반(KB)으로 사용하는 사실성을 검토합니다. 특히, 모델이 불확실할 때 답변을 보류할 수 있도록 허용한 현실적인 사용 시나리오에서 연구가 진행되었습니다.

- **Technical Details**: 이 연구에서는 다양한 사례법 및 입법에 대한 사실 질문의 데이터셋을 설계하고, 여러 LLM을 동일한 데이터셋으로 다양한 평가 방법(정확 매칭, 별칭 매칭, 퍼지 매칭 등)으로 평가하였습니다. 또한, 모델의 응답 보류 전략과 인-컨텍스트 예시가 사실 정확성을 어떻게 향상시키는지 탐구하였으며, 법률 문서에 대한 추가 사전 훈련이 63%에서 81%로 정확성을 향상시킨 것을 보여주었습니다.

- **Performance Highlights**: 단순 정확 매칭에서 벗어나 별칭 및 퍼지 매칭 방법에서 성능이 크게 향상되었으며, 응답을 보류하거나 구체적인 예시를 제공하는 것이 LLM의 정확성을 높였습니다. 이러한 연구 결과는 법률 기술 응용 프로그램의 실질적인 활용 가능성을 보여줍니다.



### Development and bilingual evaluation of Japanese medical large language model within reasonably low computational resources (https://arxiv.org/abs/2409.11783)
Comments:
          18 pages, 9 tables

- **What's New**: 최근 대형 언어 모델(LLM)의 성공과 스케일링 법칙(scaling law)의 발전으로 인해 의료 분야에서도 보안 문제를 고려하여 로컬로 운영 가능한 모델에 대한 수요가 증가하고 있습니다. 본 논문에서는 7B 모델 기반의 의료 적응형 LLM을 제시하며, 낮은 계산 자원에서도 운영할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 이 연구에서는 Qwen2 7B 모델을 활용하여, 의료 전문 지식의 증대 및 추출을 목표로 하는 MFPT (Medical Full-Parameter Training)와 MPEFT (Medical Parameter-Efficient Fine-Tuning)의 방법을 적용합니다. 주문형 데이터셋을 메디컬 퀘스천-앤서 데이터셋으로 변환하여 일본어를 포함한 이중 언어 벤치마크를 통해 평가를 진행하였습니다.

- **Performance Highlights**: 일본어와 영어로 된 의료 질문-응답 벤치마크에서 7B 모델의 성능이 기존의 70B 모델과 동등하거나 이를 초월함을 보여주었습니다. 이러한 결과는 더 작은 모델의 사용이 경제적 부담을 덜 수 있다는 점에서 높은 의미를 가집니다.



### Human-like Affective Cognition in Foundation Models (https://arxiv.org/abs/2409.11733)
- **What's New**: 이 연구는 AI의 감정 이해 능력을 평가하기 위한 새로운 평가 프레임워크를 도입합니다. 이를 통해 다양한 시나리오를 생성하고 기초 모델들이 인간의 직관과 얼마나 일치하는지 비교 분석합니다.

- **Technical Details**: 총 1,280개의 다양한 시나리오를 생성하여 평가하고, GPT-4, Claude-3, Gemini-1.5-Pro와 같은 기초 모델의 감정 인지 능력을 검토합니다. 이를 위해 심리학 이론에 기반하여 감정, 평가(appraisal), 표현(expressions) 및 결과 간의 인과관계를 설명하는 추상적 인과 그래프를 정의하고 사용합니다.

- **Performance Highlights**: 연구 결과, 기초 모델들은 인간의 직관과 일치하거나 초월하는 예측 능력을 보이며, 일부 조건에서는 '슈퍼휴먼(superhuman)' 성능을 나타냅니다. 모든 모델은 chain-of-thought reasoning에서 이점을 얻으며, 이는 기초 모델들이 감정과 그들이 미치는 영향에 대한 인간과 유사한 이해를 갖추었다는 것을 시사합니다.



### Enabling Real-Time Conversations with Minimal Training Costs (https://arxiv.org/abs/2409.11727)
Comments:
          7pages, 6 figures, 1 table

- **What's New**: 본 논문은 대화 시스템에서 실시간 상호작용의 제약을 해결하기 위한 새로운 DUO (DUplex decOding) 접근법을 제안합니다. 이 방법은 대화 중에 쿼리와 응답을 병렬로 디코딩하여 duplex 능력을 향상시키며, 추가적인 훈련 비용을 최소화합니다.

- **Technical Details**: DUO는 입력 및 출력을 주기적으로 처리하며, 각 주기에서 출력 채널은 새로운 토큰을 자가회귀적으로 생성하고, 입력 채널은 키-값 캐시를 미리 채우고 다음 토큰을 예측합니다. 주의 마스크가 수정되어 입력과 출력 토큰이 서로에게 주의를 기울이지 않도록 하여 언어 모델링의 기초를 유지합니다.

- **Performance Highlights**: 실험 결과, DUO 방법은 사용자-AI 상호작용의 자연스러움과 인간 유사성을 크게 향상시키며, 훈련 비용은 최소화하여 효율적인 인간-AI 인터페이스를 지원합니다.



### Revealing the Challenge of Detecting Character Knowledge Errors in LLM Role-Playing (https://arxiv.org/abs/2409.11726)
Comments:
          22 pages, 14 figures

- **What's New**: 본 논문은 대형 언어 모델(LLM)의 역할 수행 방식에서 캐릭터의 지식 오류를 탐지하는 능력의 유효성에 대해 평가하는 새로운 접근 방식을 제안하고 있습니다. 특히, 알려진 지식 오류(KKE)와 알 수 없는 지식 오류(UKE)에 대한 탐지를 중점적으로 다루며, 이를 통한 RPA의 자동화된 코퍼스 구축의 품질을 높이고자 합니다.

- **Technical Details**: 연구진은 KKE와 UKE를 탐지하기 위한 probing dataset을 구축하였고, 네 가지 메모리 유형(사건, 관계, 태도 및 정체성 메모리)에 따라 지식을 분류했습니다. 다양한 추론 전략을 실험하여 Self-Recollection and Self-Doubt (S2RD)이라는 에이전트 기반 추론 방법을 제안하며, 이를 통해 LLMS의 오류 탐지 능력을 개선하려고 했습니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들도 KKE와 UKE 두 가지 오류를 효과적으로 탐지하는 데 어려움을 겪으며, 최고 정확도는 65%에 미치지 못하며, 특히 KKE는 UKE보다 약 20% 더 낮은 탐지 능력을 보였습니다. S2RD 방법을 도입하면 LLM의 오류 탐지 능력이 향상되지만, 여전히 지속적인 연구가 필요함을 강조합니다.



### TART: An Open-Source Tool-Augmented Framework for Explainable Table-based Reasoning (https://arxiv.org/abs/2409.11724)
Comments:
          technical report

- **What's New**: 본 논문에서는 Tool-Augmented Reasoning framework for Tables (TART)를 제안하여 기존의 Large Language Models (LLMs)가 테이블 구조를 이해하고 정확한 수치적 추론을 수행하는 한계를 극복하고자 합니다. TART는 테이블 포맷터, 도구 생성기 및 설명 생성기의 세 가지 주요 구성 요소를 통합합니다.

- **Technical Details**: TART는 입력 테이블을 정리하고 포매팅하는 테이블 포맷터, 테이블 조작 및 수치적 추론을 위한 특화된 도구를 호출하는 도구 생성기, 그리고 외부 도구 호출을 통합하여 설명을 생성하는 설명 생성기로 구성됩니다. TART는 ToolTab 데이터셋을 활용하여 LLMs를 테이블-도구 통합에 맞춰 훈련시킵니다.

- **Performance Highlights**: TART는 기존 Chain-of-Thought 방식을 초월하여 수치적 처리의 정확성과 추론 과정을 명확하게 개선하였으며, CodeLlama와의 조합으로 GPT-3.5-turbo의 90.0% 정확도를 달성하였습니다. 이를 통해 다양한 실제 세계 시나리오에서 강력한 성능을 입증하였습니다.



### From Lists to Emojis: How Format Bias Affects Model Alignmen (https://arxiv.org/abs/2409.11704)
Comments:
          Working in progress

- **What's New**: 본 논문은 인간 피드백 기반 강화 학습(RLHF)에서 포맷 편향(format biases) 문제를 다룹니다. 다양한 선호 모델이 특정 포맷 패턴(예: 목록, 링크, 강조 텍스트, 이모지)에 대한 강한 편향을 가지고 있음을 발견했습니다. 또한 이러한 편향을 활용하여 인기 있는 벤치마크에서 높은 순위를 차지할 수 있음을 보여주었습니다.

- **Technical Details**: 연구에서는 RLHF를 통해 형성된 정책이 보상 모델(RM)을 "게임화(gaming)"하여 비효율적인 응답을 생성하는 문제, 즉 불필요하게 긴 응답을 생성하려는 경향을 조사했습니다. 연구자들은 형식 편향을 정의하고 이를 통해 인간 평가자, GPT-4 및 여러 오픈소스 모델의 행동을 분석하였습니다. 기존의 연구에서는 길이 편향만을 다루었으나, 이 연구에서는 다양한 형식 편향을 포괄적으로 분석하였습니다. 이 논문에서 소량의 편향된 데이터를 사용한 실험 결과, 1% 미만의 편향된 데이터가 보상 모델에 중대한 영향을 미친다는 것을 발견하였습니다.

- **Performance Highlights**: 논문에서 제시된 결과는 AlpacaEval 및 LMSYS Chatbot Arena와 같은 벤치마크에서 형식 편향이 존재함을 보였으며, 이러한 편향이 모델의 실제 능력을 정확하게 반영하지 못한다는 점을 강조합니다. 특히, 많은 고위 모델들이 특정 편향된 포맷의 응답을 생성함으로써 평가자를 속이고 있다고 지적합니다. 연구는 형식과 내용을 분리하는 것이 RLHF 알고리즘 설계 및 모델 평가에 필수적임을 강조합니다.



### Harnessing LLMs for API Interactions: A Framework for Classification and Synthetic Data Generation (https://arxiv.org/abs/2409.11703)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)를 활용하여 자연어 입력을 해당 API 호출로 분류하고, 특정 API 기능에 맞춘 샘플 데이터셋을 자동으로 생성하는 시스템을 제안합니다. 이 시스템은 사용자들이 단순한 입력으로 복잡한 소프트웨어 기능을 호출할 수 있도록 하여 상호작용 효율성을 향상시키고, 소프트웨어 활용 장벽을 낮추는 것을 목표로 합니다.

- **Technical Details**: 제안된 시스템은 두 가지 핵심 기능을 통합합니다. 첫째, 자연어 입력을 해석하고 올바른 API 호출로 분류하는 LLM의 활용입니다. 둘째, 특정 API 기능에 맞춘 샘플 데이터셋 생성 자동화입니다. 이 접근 방식은 다양한 API 호출의 정확한 분류와 LLM 성능의 체계적인 평가를 위한 실용적인 도구를 제공합니다.

- **Performance Highlights**: 실험 결과, GPT-4는 0.996의 높은 분류 정확도를 기록했으며, LLaMA-3-8B는 0.759로 낮은 성능을 보였습니다. 이 결과는 API 관리에 있어 LLM의 가능성을 입증하며, 다양한 응용 분야에서 모델 테스트와 선택을 안내하는 시스템의 효과성을 강조합니다.



### Enhancing Complex Formula Recognition with Hierarchical Detail-Focused Network (https://arxiv.org/abs/2409.11677)
Comments:
          Submitted to the 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이 논문은 복잡한 수학 공식을 인식하는 데 높은 정확도를 제공하는 HDR (Hierarchical Detail-Focused Recognition) 데이터셋과 새로운 네트워크 구조 HDNet을 제안합니다. HDR 데이터셋은 1억 개의 훈련 샘플로 구성된 대규모 데이터셋이며, 여러 해석을 포함하는 테스트 세트를 제공합니다.

- **Technical Details**: HDNet은 계층적 서브 포뮬라 모듈을 가진 인코더-디코더 기반의 MER (Mathematical Expression Recognition) 프레임워크입니다. 이 모델은 고해상도의 서브 포뮬라 이미지로 분해하여 세밀한 계층적 특징을 포착하고, 띄어쓰기를 고려하여 복잡한 공식을 보다 정확하게 해석할 수 있도록 디자인되었습니다.

- **Performance Highlights**: 실험 결과 HDNet은 기존 MER 모델들보다 뛰어난 성능을 보였으며, 복잡한 수학 공식 처리를 위한 새로운 기준을 제시했습니다. 이를 통해 복잡한 공식을 인식하는데 있어 기존 방법들에 비해 개선된 정확도를 확보하였습니다.



### RUIE: Retrieval-based Unified Information Extraction using Large Language Mod (https://arxiv.org/abs/2409.11673)
Comments:
          14 pages, 3 figures

- **What's New**: RUIE( Retrieval-based Unified Information Extraction) 프레임워크를 제안하여 정보 추출(Information Extraction) 작업을 위해 신속한 일반화(Generalization)를 가능하게 하면서 계산 비용을 줄였습니다.

- **Technical Details**: RUIE는 인컨텍스트 학습(In-context Learning) 방법을 활용하여 LLM의 정보 추출 성능을 최적화하고, 후보 데모의 랭킹과 관련한 LLM의 선호도를 통합한 키워드 강화 리워드 모델(Keyword-enhanced Reward Model)을 설계했습니다.

- **Performance Highlights**: 8개의 검증 데이터셋에 대한 실험 결과, RUIE는 평균 F1-스코어가 instruction-tuning 방법과 기타 검색기보다 각각 19.22 및 3.13 향상되어 보이지 않는 작업에 대한 일반화에서 효과적임을 입증했습니다.



### BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla (https://arxiv.org/abs/2409.11638)
- **What's New**: 이번 연구는 방글라어의 다언어 LLM(대형 언어 모델)의 고정관념 사회 편향을 평가하기 위해 설계된 BanStereoSet 데이터셋을 소개합니다. 기존의 영어 중심 데이터셋을 넘어 방글라어 사용 커뮤니티에서 발생하는 편향을 포착하기 위한 리소스를 개발하였습니다.

- **Technical Details**: BanStereoSet 데이터셋은 인종(race), 직업(profession), 성별(gender), 연령차별(ageism), 미(beauty), 직업 내 미(beauty in profession), 지역(region), 카스트(caste), 종교(religion)의 9가지 편향 카테고리를 포함하여 1,194개의 문장으로 구성되어 있습니다. 이 데이터셋은 LLM의 편향 측정 도구로 사용될 뿐만 아니라, 다양한 사회적 범주에서의 고정관념 편향 탐색을 가능하게 합니다.

- **Performance Highlights**: 기존의 다언어 LLM 모델(GPT-4, Mistral-7B, llama3.1-70B, Gemma2-27B)을 사용한 분석 결과, 방글라어에 특화된 편향이 발견되었습니다. 이는 문화적 및 언어적 맥락에 따라 조정된 데이터셋의 필요성을 강조하며, 보다 공정한 언어 기술 개발을 위한 기초 자료로 작용할 수 있습니다.



### "A Woman is More Culturally Knowledgeable than A Man?": The Effect of Personas on Cultural Norm Interpretation in LLMs (https://arxiv.org/abs/2409.11636)
Comments:
          Preprint, Under Review

- **What's New**: 이 연구는 대형 언어 모델(LLM)의 개인화에 대한 필요성이 증가하는 가운데, LLM의 출력 결과를 특정 페르소나(persona)를 통해 안내하는 방법을 탐구합니다. 연구 목적은 지정된 페르소나에 따라 사회적 규범의 이해가 얼마나 다르게 나타나는지를 살펴보는 것입니다.

- **Technical Details**: 연구에서는 12개의 사회인구통계학적 범주에서 36개의 페르소나를 정의하고, 4개의 다른 LLM을 통해 두 개의 문화적 규범 데이터셋을 활용하여 LLM의 문화적 규범 해석을 분석했습니다. 여기서 사용된 LLM은 GPT-4o-mini, Llama3-8B, Gemma2-27B, Mistral-7B입니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, LLM은 사회적으로 선호되는 페르소나가 사용될 때(예: 마른 사람) 사회적 규범을 더 정확하게 해석하는 경향이 있습니다. 또한, 다양한 사회인구통계 그룹 내에서도 문화적 규범 해석의 차이를 발견하였습니다. 예를 들어, '여성' 페르소나는 '남성' 페르소나보다 규범을 더 정확하게 해석하는 결과를 나타냈습니다.



### ProSLM : A Prolog Synergized Language Model for explainable Domain Specific Knowledge Based Question Answering (https://arxiv.org/abs/2409.11589)
Comments:
          Accepted at NeSy 2024

- **What's New**: 본 논문에서는 기존의 신경 기계 학습 모델의 불투명성을 해결하기 위해 기호 기반의 신뢰할 수 있는 접근 방식을 제안합니다. 특히, \\systemname{}라는 새로운 신경-기호 프레임워크를 통해 질문-응답 시스템에서 LLM의 신뢰성과 견고성을 향상시키는 방법을 제시합니다.

- **Technical Details**: \\systemname{}는 도메인 특화된 지식 기반(knowledge base), 논리적 추론 시스템, 기존의 LLM과 통합된 구조로 이루어져 있습니다. 이 프레임워크는 (1) 맥락 수집(context gathering): 주어진 질의에 대해 설명 가능하고 관련된 맥락을 생성하고, (2) 검증(validation): 지식 기반에 따라 진술의 사실적 정확성을 확증하고 검증하는 기능을 포함하고 있습니다.

- **Performance Highlights**: 이 프레임워크는 사용자에게 시스템이 특정 출력을 도출한 과정을 이해할 수 있도록 돕고, LLM의 추가적인 학습이나 파인 튜닝 없이도 계산 효율성을 제공합니다. 실험 결과는 향후 LLM의 활용 방안에 관한 새로운 방향성을 제시합니다.



### HEARTS: A Holistic Framework for Explainable, Sustainable and Robust Text Stereotype Detection (https://arxiv.org/abs/2409.11579)
Comments:
          Submitted to NeurIPS 2024 SoLaR Workshop

- **What's New**: 이 논문은 HEARTS(설명 가능한, 지속 가능한 및 강력한 텍스트 고정관념 감지 프레임워크)를 도입하며, 이것은 고정관념 감지의 정확성을 높이고, 환경적 영향을 최소화하며, 투명하고 해석 가능한 설명을 제공합니다.

- **Technical Details**: HEARTS는 Expanded Multi-Grain Stereotype Dataset(EMGSD)을 구성하여 57,201개의 라벨이 부착된 텍스트를 사용할 수 있도록 하며, ALBERT-V2 모델을 이용해 고정관념 감지를 수행합니다. SHAP을 사용하여 토큰 수준의 중요 값을 생성하고, SHAP과 LIME의 출력을 비교하여 설명 가능성 신뢰 점수를 계산합니다.

- **Performance Highlights**: EMGSD를 기반으로 세부 조정을 수행한 BERT 모델이 단일 구성 요소로 훈련된 모델보다 성능이 뛰어난 것으로 확인되었으며, ALBERT-V2 모델은 80% 이상의 정확성을 달성하였습니다. 또한 LLM의 출력에서 고정관념 편향의 변화가 시간이 지남에 따라 점진적으로 감소하는 것을 나타냈습니다.



### Preference Tuning with Human Feedback on Language, Speech, and Vision Tasks: A Survey (https://arxiv.org/abs/2409.11564)
Comments:
          Survey paper

- **What's New**: 본 논문은 인간의 피드백과의 통합을 통해 딥 생성 모델을 인간의 선호도에 aligned(정렬)시키는 preference tuning(선호 조정)에 관한 최신 발전을 포괄적으로 검토합니다.

- **Technical Details**: 논문은 preference tuning을 reinforcement learning(강화 학습) 프레임워크 내에서 세 가지 주요 섹션으로 구성하였습니다. 여기에는 강화 학습을 이용한 모델과 데이터셋, 그리고 다양한 정책 접근 방법에 대한 설명이 포함되어 있습니다.

- **Performance Highlights**: LLM(대형 언어 모델)에서 preference tuning은 작업 특화 기술을 개선하고 불필요한 출력(undesired outputs)을 피하는 데 기여하며, 다국어 LLM에서도 이점이 있는 것으로 나타났습니다.



### Small Language Models can Outperform Humans in Short Creative Writing: A Study Comparing SLMs with Humans and LLMs (https://arxiv.org/abs/2409.11547)
- **What's New**: 이 연구는 소규모 언어 모델(SLM)인 BART Large의 창작 허구 글쓰기 능력을 평가하고, 이를 인간 창작자 및 대규모 언어 모델(LLM)인 GPT-3.5와 GPT-4o와 비교했습니다. 실험을 통해 BART Large는 대부분의 평가 기준에서 인간보다 우수한 성과를 나타냈으며, 창의성 측면에서만 열세를 보였습니다.

- **Technical Details**: 연구는 두 가지 실험으로 구성되어 있습니다. 첫 번째는 인간 평가로, 68명의 참가자들이 생성된 짧은 이야기를 평가했습니다. 평가 기준에는 문법, 관련성, 창의성 및 매력이 포함됩니다. 두 번째 실험은 생성한 텍스트의 언어적 특성을 비교하는 질적 분석으로, GPT-4o가 높은 일관성(coherence)을 보였지만 예측 가능한 내러티브를 생성했다는 결과를 냈습니다.

- **Performance Highlights**: BART Large는 전체 점수에서 2.11을 기록하며 인간 점수 1.85보다 14% 향상된 성과를 보였고, 창의성 측면에서 BART의 15%의 이야기가 창의적으로 새롭다고 평가된 반면, GPT-4o는 3%만이 창의적으로 새로운 것으로 간주되었습니다. 이는 모델의 크기가 창의성에 미치는 영향을 보여줍니다.



### Chain-of-Thought Prompting for Speech Translation (https://arxiv.org/abs/2409.11538)
- **What's New**: 최근 연구에서는 speech embeddings를 활용한 Speech-LLM 모델이 자동 음성 인식(ASR)과 자동 음성 번역(AST)에서 강력한 성능을 보이고 있다는 점에 착안해 ASR 전사를 AST의 프롬프트로 활용하는 새로운 접근법을 제안하고 있다.

- **Technical Details**: 제안된 Speech-LLM 모델은 speech encoder와 encoder-decoder 구조인 Megatron-T5로 구성된다. ASR 전사를 먼저 디코딩하고, 이후 이러한 전사와 인코딩된 음성을 사용하여 프롬프트를 생성함으로써, 두 단계의 체인 오브 사고(Chain-of-Thought, CoT) 방식으로 음성 번역을 이끌어낸다. 저자들은 Low-rank adaptation(LoRA)을 Megatron-T5 LLM에 적용하여 모델을 조정하며, 전통적인 전체 모델 튜닝(full model fine-tuning)보다 우수한 성능을 보인다.

- **Performance Highlights**: 제안된 CoT 프롬프트 방식은 6개의 En->X 또는 X->En AST 작업에 대해 평균 2.4 BLEU 포인트를 개선하는 성과를 보여주었고, ASR 전사를 사용하지 않은 기본 모델과 비교할 때 통계적으로 유의미한 성과를 거두었다. 또한, 관련 CoT 예측 방법과 비교했을 때 평균 2 BLEU 포인트 더 나은 성능을 보여 주었다.



### Egalitarian Language Representation in Language Models: It All Begins with Tokenizers (https://arxiv.org/abs/2409.11501)
Comments:
          Content - 8 pages, References - 3 pages

- **What's New**: 최근 연구에서는 Tokenization(토크나이제이션)이 언어 모델과 인간 언어 사이에서 중요한 역할을 한다는 점을 강조하고 있습니다. 특히, 복잡한 스크립트 언어인 타밀어, 신할라어, 힌디어에 대한 공정한 표현을 달성하기 위한 새로운 접근법인 Grapheme Pair Encoding (GPE)을 소개합니다.

- **Technical Details**: 우리는 기존의 Byte Pair Encoding (BPE) 알고리즘을 개선하여 Grapheme(그라페임) 정보를 통합하는 방법을 모색했습니다. 이 연구는 사전 토크나이제이션(pre-tokenization) 방법의 선택이 복잡한 스크립트 언어에 대한 표현성에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험 결과, 그라페임 기반의 문자 추출 방식이 바이트 단위 토크나이저보다 복잡한 스크립트에 대해 우수한 성능을 보이는 것으로 나타났습니다. 타밀어, 신할라어, 힌디어에서 이 접근법을 검증하였습니다.



### Multi-Document Grounded Multi-Turn Synthetic Dialog Generation (https://arxiv.org/abs/2409.11500)
- **What's New**: 이번 연구에서는 다수의 문서에 기반한 복수 턴의 합성 대화 생성(new synthetic dialog generation) 기법을 소개합니다. 이 기법은 1) Chain-of-Thought (CoT) 프롬프트를 사용하여 생성된 사용자 쿼리를 통해 전체 대화 흐름을 제어하고, 2) 사용자의 대화 턴마다 기초 문서를 업데이트하는 방식을 모방하여 다수의 문서에 기반한 대화 생성을 지원하며, 3) LLM-as-a-Judge를 적용하여 잘못된 답변이 포함된 쿼리를 필터링합니다.

- **Technical Details**: 이 연구는 synthetic dialog의 품질을 평가하기 위해 사용자-에이전트 간 대화를 인공지능 모델을 통해 생성하고, 이 데이터가 얼마나 다양하고 일관성 있으며 주로 정확한 답변을 포함하는지를 평가하는 데 초점을 맞추었습니다. 평가 기준은 쿼리의 응답 가능성과 응답의 정확성 간의 관계를 고려하여 설정되었습니다. 평가 과정에서 단일 절(clause)과 여러 절로 구성된 쿼리를 분석해야 합니다.

- **Performance Highlights**: 인간 평가 및 자동 평가를 통해 합성 대화에 최적화된 모델이 기존 인간 생성 훈련 데이터에 비해 네 개의 공개적인 복수 턴 문서 기반 벤치마크 테스트 세트에서 일관되게 성능을 발휘한다는 결과가 도출되었습니다.



### Enriching Datasets with Demographics through Large Language Models: What's in a Name? (https://arxiv.org/abs/2409.11491)
Comments:
          8 pages, 7 Tables, 5 Figures

- **What's New**: 본 논문은 Large Language Models(LLMs)를 사용하여 이름으로부터 인구 통계 데이터를 예측하는 방법을 최초로 탐구합니다. 이는 의료, 사회 과학 및 공공 정책과 같은 분야에서 중요하게 여겨집니다.

- **Technical Details**: Zero-shot LLM prompting 기법을 활용하여 개인의 이름을 단일 입력 변수로 사용하여 인구 통계학적 정보(성별, 인종, 연령 등)를 추론합니다. 연구는 홍콩의 금융 전문가 이름 데이터를 분석하여, 비서구적인 인구 통계학적 데이터셋의 부족 문제를 해결합니다.

- **Performance Highlights**: 최신 LLM이 이전의 감독 학습 방식보다 뛰어난 성능을 보여주는 것을 입증하며, 이는 특히 연령 예측에 있어 다소의 편향이 있음을 발견하였습니다. 연구 결과는 LLM이 인구 통계학적 데이터의 질을 높일 수 있는 잠재력을 지니고 있음을 시사합니다.



### Optimizing Performance: How Compact Models Match or Exceed GPT's Classification Capabilities through Fine-Tuning (https://arxiv.org/abs/2409.11408)
- **What's New**: 이 논문에서는 FinBERT 및 FinDRoBERTa와 같은 비생성형 소형 모델이 감정 분석에서 GPT-3.5 및 GPT-4 모델을 초과하는 성능을 보일 수 있음을 입증했습니다. 특히 금융 뉴스에 대한 제로샷 학습(zero-shot learning) 환경에서 이러한 모델들은 시장 심리를 평가하는 데 있어 일관된 성과를 나타냈습니다.

- **Technical Details**: 새로운 데이터베이스를 생성하여 뉴스에 시장 점수를 할당하고 회사의 주가가 상승, 하락 또는 중립인지 체계적으로 분석했습니다. Condorcet의 배심원 정리에 대한 가정이 성립하지 않음을 보여주며, 이는 미세 조정된 소형 모델이 GPT 모델과 독립적이지 않음을 나타냅니다.

- **Performance Highlights**: FinBERT 및 FinDRoBERTa는 GPT-3.5와의 비교에서 경쟁력 있는 성과를 보였으며, 이를 통해 대형 언어 모델의 크기 이점에 대한 재고를 촉발했습니다. 이러한 결과는 HuggingFace 플랫폼에서 공개되어, 추후 연구에 기여할 수 있는 중요한 자원으로 활용될 것입니다.



### Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (https://arxiv.org/abs/2409.12191)
Comments:
          Code is available at this https URL. arXiv admin note: text overlap with arXiv:2408.15262 by other authors

- **What's New**: Qwen2-VL 시리즈는 이전 Qwen-VL 모델의 업그레이드 버전으로, 동적 해상도(naive dynamic resolution) 처리 메커니즘을 도입하여 다양한 해상도의 이미지를 처리하고, 시각적 토큰(visual tokens)을 동적으로 생성할 수 있도록 개선되었습니다.

- **Technical Details**: Qwen2-VL 모델은 Multimodal Rotary Position Embedding (M-RoPE)을 활용하여 텍스트, 이미지 및 비디오 간의 위치 정보를 효과적으로 융합합니다. 또한, 2B, 8B 및 72B 매개변수를 갖춘 버전으로 확장 법칙을 탐색하며, 다양한 해상도와 종횡비에서의 이해력을 향상시킵니다.

- **Performance Highlights**: Qwen2-VL-72B 모델은 DocVQA, InfoVQA, RealWorldQA와 같은 다양한 멀티모달 벤치마크에서 GPT-4o 및 Claude3.5-Sonnet과 유사한 성능을 보여주며, 고해상도 영상 이해 및 다국어 지원 기능을 제공합니다.



### Low Frame-rate Speech Codec: a Codec Designed for Fast High-quality Speech LLM Training and Inferenc (https://arxiv.org/abs/2409.12117)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 Low Frame-rate Speech Codec (LFSC)라는 새로운 신경 오디오 코덱을 소개합니다. 이 코덱은 1.89 kbps의 비트 전송률로 21.5 FPS(Frames Per Second)의 속도로 오디오를 압축하면서도 높은 오디오 품질을 유지합니다.

- **Technical Details**: LFSC는 풀리 합성곱 생성기 신경망과 세 개의 구별자(discriminator)로 구성되어 있습니다. 생성기는 인코더, 벡터 정량화(vector quantization), HiFi-GAN 기반의 디코더로 구성됩니다. 인코더는 5개의 잔차 블록(residual blocks)으로 이루어져 있으며, 각 블록은 단계(steps)에 따라 [2, 2, 4, 8, 8]의 커널 스트라이드를 가진 1D 합성곱 층이 뒤따릅니다. 또한, 세 개의 구별자는 각각의 네트워크가 제곱-GAN(squared-GAN)과 특징 매칭 손실(feature-matching loss)을 이용하고 있습니다.

- **Performance Highlights**: LFSC는 LLM 기반의 텍스트-음성 변환 모델의 추론 속도를 약 3배 개선하면서도 음성 인식 크기를 높이고, 품질 또한 이전 모델들과 유사한 수준을 보여줍니다. 실험 결과, LFSC는 고품질 오디오를 제공하며, Speech LLM을 위한 추론 시간을 효과적으로 줄이는 데 기여합니다.



### Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation (https://arxiv.org/abs/2409.11860)
Comments:
          13 pages, 5 figures, 4 Tables

- **What's New**: 본 논문에서는 대규모 e-commerce 환경에서 제품 검색 엔진을 평가하는 프레임워크를 제안합니다. 이 프레임워크는 Multimodal LLMs(대형 다중모달 언어 모델)을 활용하여 개별 쿼리에 맞춤화된 주석 가이드를 생성하고 주석 작업을 수행합니다.

- **Technical Details**: 저자들은 (1) 사용자 쿼리 및 그 맥락에 따라 LLM이 쿼리 요구 사항 목록과 쿼리 구체적 주석 가이드를 생성하며, (2) 검색 엔진에서 제품을 검색하고, (3) 검색된 제품에 대한 텍스트 설명 및 이미지를 기반으로 시각적 설명을 생성하고, (4) 결합된 설명을 입력으로 LLM에 제공하여 해당 쿼리-제품 쌍에 대한 관련 점수를 배정합니다. 이 과정은 데이터베이스에 저장되어 다양한 검색 엔진 간의 평가 일관성을 보장합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 인간 주석과 유사한 품질을 보여 주며, 시간과 비용을 크게 절감하고 문제 발견 속도를 향상시키는 것으로 나타났습니다. 또한, LLMs가 대량의 주석 작업을 효율적으로 처리할 수 있는 반면, 인간 전문가는 더 복잡한 사례에 대해 더 잘 활용될 수 있다고 결론지었습니다.



### FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancemen (https://arxiv.org/abs/2409.11699)
- **What's New**: 본 논문은 하이브리드 추천 시스템의 새로운 모델인 Flare(협업 필터링 모델과 언어 모델을 통합한 구조)를 소개합니다. Flare는 언어 모델(mT5)과 협업 필터링 모델(Bert4Rec)을 Perceiver 네트워크를 통해 결합하여 더 향상된 추천을 제공합니다.

- **Technical Details**: Flare는 텍스트 기반 아이템 컨텍스트와 아이템 임베딩을 합성하여 richer representation을 생성하는 하이브리드 추천 모델입니다. 두 개의 추천 모델(Bert4Rec의 재구현 및 Flare)을 사용하여 성능을 평가하며, 이 과정에서 두 단계의 평가를 진행합니다.

- **Performance Highlights**: Flare는 작은 데이터 세트에서 경쟁력 있는 성능을 보였으며, 이는 기존의 Bert4Rec 모델보다 뛰어난 결과를 보여줍니다. 또한, Flare는 대규모 데이터 세트에서도 새로운 기준선을 설정하여, 텍스트를 추가함으로써 성능이 개선됨을 입증하였습니다.



### Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation (https://arxiv.org/abs/2409.11598)
- **What's New**: 이번 논문은 Retrieval-Augmented Generation (RAG) 시스템에 공정한 순위 평가를 통합한 최초의 체계적인 평가를 제시합니다. 연구는 RAG 시스템에서 각 관련 항목의 공정한 노출을 측정하여 공정한 순위의 중요성을 강조합니다.

- **Technical Details**: RAG 시스템에서 공정한 순위란, 각각의 관련된 항목이 얼마나 공정하게 노출되는지를 측정하는 것을 의미합니다. 연구는 공정성을 고려하여 세 가지 주제에 대한 결과를 분석하며, 공정한 순위 시스템을 채택하는 RAG 모델이 전통적인 RAG보다 높은 생성 품질을 유지할 수 있음을 발견했습니다. 해당 연구는 7개 데이터셋을 이용하여 9가지 RAG 시스템을 분석하였습니다.

- **Performance Highlights**: 공정한 순위를 포함한 RAG 시스템이 높은 생성 품질을 유지하면서도 기존 RAG 시스템을 초월할 수 있는 사례가 있음을 발견했습니다. 이는 공정성을 보장하면서도 시스템의 효과성을 훼손하지 않고 양질의 서비스를 제공할 수 있음을 나타냅니다.



### Augment, Drop & Swap: Improving Diversity in LLM Captions for Efficient Music-Text Representation Learning (https://arxiv.org/abs/2409.11498)
Comments:
          To appear in the Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: 본 연구에서는 오디오-텍스트 대비 모델의 디자인 선택이 음악-텍스트 표현의 품질에 미치는 영향을 정량적으로 분석합니다. 특히, 데이터 제약과 컴퓨팅 예산의 한계를 고려한 연구를 통해 데이터 큐레이션이 가장 중요한 영향을 미친다는 사실을 발견했습니다.

- **Technical Details**: 우리는 두 가지 주요 요소, 즉 아키텍처와 데이터에 중점을 두어 음악-텍스트 임베딩 모델의 디자인 공간을 탐구합니다. 이 논문에서는 Augmented View Dropout 및 TextSwap이라는 두 가지 새로운 기술을 도입해 훈련 중 텍스트 입력의 다양성과 설명성을 증가시킵니다.

- **Performance Highlights**: 실험을 통해, 제안한 기술들이 다양한 프리트레이닝 체계, 모델 아키텍처 및 하위 데이터 분포에서 성능을 향상시키는 데 효과적임을 보여줍니다. 이전 작업에 비해 새로운 최첨단 성능을 달성했습니다.



### Jailbreaking Large Language Models with Symbolic Mathematics (https://arxiv.org/abs/2409.11445)
- **What's New**: 이번 연구에서는 MathPrompt라는 새로운 jailbreaking 기술을 소개합니다. 이 기술은 LLM의 기호 수학(Symbolic Mathematics) 능력을 활용하여 안전 메커니즘을 우회합니다. 이는 유해한 자연어 프롬프트를 수학 문제로 인코딩하여 현재 AI 안전 조치의 중요한 취약점을 드러냅니다.

- **Technical Details**: MathPrompt는 두 단계로 구성됩니다. 첫째, 유해한 자연어 프롬프트를 기호 수학 문제로 변환하고, 둘째, 이러한 수학적으로 인코딩된 프롬프트를 목표 LLM에 제시합니다. 실험 결과, 13개의 최첨단 LLM에서 평균 73.6%의 공격 성공률을 기록하였습니다.

- **Performance Highlights**: 현재 안전 훈련 메커니즘은 수학적으로 인코딩된 입력에 일반화되지 않는다는 것을 강조합니다. LLM이 기호 수학을 이해함에도 불구하고, 기존의 안전 조치들은 이와 같은 수학적 표현을 잘 처리하지 못하고 있습니다.



### AIvril: AI-Driven RTL Generation With Verification In-The-Loop (https://arxiv.org/abs/2409.11411)
- **What's New**: AIvril은 RTL-aware LLM의 정확도와 신뢰성을 향상시키기 위한 고급 프레임워크로, 자동 문법 수정 및 기능 검증을 위한 다중 에이전트 시스템을 도입합니다. 이를 통해 코드 생성에서 오류를 크게 줄이고, 하드웨어 설계의 자동화와 최적화를 위한 중요한 단계를 제공합니다.

- **Technical Details**: AIvril 프레임워크는 두 가지 핵심 구성 요소, 즉 AutoReview와 AutoDV(Automatic Design Verification)를 포함하여 RTL 코드의 문법적 정확성과 기능적 정확성을 보장합니다. 이 프레임워크는 전자 설계 자동화(EDA) 도구의 피드백을 활용하여 생성된 코드를 정제하고 디버깅하는 지능형 에이전트의 네트워크를 구성합니다.

- **Performance Highlights**: AIvril은 VerilogEval-Human 데이터셋에서 실험을 진행한 결과, 코드 질이 이전 작업에 비해 1.32배와 2배 향상되었으며, 검증 목표를 달성하는 평균 성공률이 88.46%입니다. 이는 보다 강력하고 준수하는 RTL 구현으로 이어집니다.



### Autoregressive + Chain of Thought $\simeq$ Recurrent: Recurrence's Role in Language Models' Computability and a Revisit of Recurrent Transformer (https://arxiv.org/abs/2409.09239)
- **What's New**: 이 논문은 기존 Transformer 아키텍처의 한계를 극복하기 위해, Chain of Thought (CoT) 프롬프트를 활용하여 반복 구조의 중요성과 자가 회귀(autoregression)의 역할을 심층적으로 분석합니다.

- **Technical Details**: 이 연구는 신경망의 반복 구조가 추론 능력과 계산 가능성에 미치는 영향을 탐구하며, CoT 접근법이 반복 계산을 어떻게 모방할 수 있는지 설명합니다. 또한, 최근의 반복 기반 Transformer 모델 설계를 재조명하고, 'recurrence-completeness'의 개념을 통해 이들의 계산 능력을 분석합니다.

- **Performance Highlights**: CoT 접근을 통해 Transformer 모델의 성능을 높일 수 있음을 입증하며, 다양한 과제를 해결하는 데 있어 CoT의 기여를 효과적으로 강조합니다.



### ReflectDiffu:Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework (https://arxiv.org/abs/2409.10289)
- **What's New**: 이 논문에서는 감정적 응답 생성을 위한 경량화된 종합 프레임워크인 ReflectDiffu를 제안합니다. 기존 연구는 감정과 의도 간의 복잡한 상호작용을 간과하거나, 큰 언어 모델(LLMs)로 인해 계산 부담이 큽니다.

- **Technical Details**: ReflectDiffu는 감정 전염(emotion contagion)을 통합하여 감정 표현(emotional expressiveness)을 향상시키고, 감정 추론 마스크(emotion-reasoning mask)를 사용하여 중요한 감정 요소를 강조합니다. 또한 강화 학습(reinforcement learning) 내에서 의도 모방(intent mimicry)을 포함시켜 확산(diffusion) 중 개선을 이룹니다. Exploring-Sampling-Correcting 메커니즘을 활용하여 감정적 의사결정을 정밀한 의도 행동으로 전환합니다.

- **Performance Highlights**: ReflectDiffu는 관련성(relevance), 제어 가능성(controllability), 정보성(informativeness) 면에서 기존 모델들을 능가하며, 자동 평가 및 인간 평가 모두에서 최첨단 성과를 달성했습니다.



New uploads on arXiv(cs.IR)

### Decoding Style: Efficient Fine-Tuning of LLMs for Image-Guided Outfit Recommendation with Preferenc (https://arxiv.org/abs/2409.12150)
Comments:
          CIKM 2024

- **What's New**: 이 논문은 개인화된 의상 추천의 어려움을 해결하기 위해 대형 언어 모델(LLMs)을 활용하는 새로운 프레임워크를 제안합니다. 기존 LLM의 '블랙 박스'(black box) 및 정적 특성을 극복하기 위해 세밀하게 조정하고 직접적인 피드백을 통합합니다.

- **Technical Details**: 이 프레임워크는 다중 모달 대형 언어 모델(MLLM)을 활용하여 아이템 설명의 시각적-텍스트적 갭을 연결합니다. 또한, LLM을 오픈 소스 폴리보어(Polyvore) 데이터세트에서 효율적으로 미세 조정하여 세련된 의상 추천 능력을 최적화합니다.

- **Performance Highlights**: 평가 결과, 제안된 프레임워크는 기본 LLM보다 상당히 향상된 성능을 나타내며, 트렌드에 맞춘 스타일리시한 의상 제안을 지속적으로 생성합니다. 이 결과는 개인화된 추천 경험을 향상시킬 잠재력을 보여줍니다.



### Understanding the Effects of the Baidu-ULTR Logging Policy on Two-Tower Models (https://arxiv.org/abs/2409.12043)
Comments:
          Accepted at the CONSEQUENCES '24 workshop, co-located with ACM RecSys '24

- **What's New**: 이 논문은 Baidu-ULTR 데이터셋을 사용하여 최근의 두 개의 타워 모델(two-tower model)이 산업 응용에서의 confounding 문제를 어떻게 영향을 미치는지를 분석합니다. 이 연구의 결과는 두 타워 모델이 이러한 문제에 영향을 받지 않는다는 것을 나타냅니다.

- **Technical Details**: 두 타워 모델은 쿼리-문서 특징을 이용해 unbiased relevance를 예측하는 타워와, 위치 혹은 디바이스 타입 같은 bias 관련 특징을 이용해 examination behavior를 예측하는 타워로 구성됩니다. 본 논문은 logging policy confounding 문제의 존재 여부를 baidu-ULTR 데이터셋을 통해 확인하고, 기존 두 개의 제안된 방법이 모델 성능을 개선하는지 평가합니다.

- **Performance Highlights**: 모든 두 타워 모델이 랜덤 기반라인을 초과하여 성능을 발휘했습니다. 그러나 confounding correction 방법들은 유의미한 성능 향상을 보이지 않았습니다.



### Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation (https://arxiv.org/abs/2409.11860)
Comments:
          13 pages, 5 figures, 4 Tables

- **What's New**: 본 논문에서는 대규모 e-commerce 환경에서 제품 검색 엔진을 평가하는 프레임워크를 제안합니다. 이 프레임워크는 Multimodal LLMs(대형 다중모달 언어 모델)을 활용하여 개별 쿼리에 맞춤화된 주석 가이드를 생성하고 주석 작업을 수행합니다.

- **Technical Details**: 저자들은 (1) 사용자 쿼리 및 그 맥락에 따라 LLM이 쿼리 요구 사항 목록과 쿼리 구체적 주석 가이드를 생성하며, (2) 검색 엔진에서 제품을 검색하고, (3) 검색된 제품에 대한 텍스트 설명 및 이미지를 기반으로 시각적 설명을 생성하고, (4) 결합된 설명을 입력으로 LLM에 제공하여 해당 쿼리-제품 쌍에 대한 관련 점수를 배정합니다. 이 과정은 데이터베이스에 저장되어 다양한 검색 엔진 간의 평가 일관성을 보장합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 인간 주석과 유사한 품질을 보여 주며, 시간과 비용을 크게 절감하고 문제 발견 속도를 향상시키는 것으로 나타났습니다. 또한, LLMs가 대량의 주석 작업을 효율적으로 처리할 수 있는 반면, 인간 전문가는 더 복잡한 사례에 대해 더 잘 활용될 수 있다고 결론지었습니다.



### Active Reconfigurable Intelligent Surface Empowered Synthetic Aperture Radar Imaging (https://arxiv.org/abs/2409.11728)
- **What's New**: 본 논문은 unmanned aerial vehicle (UAV)에 장착된 active reconfigurable intelligent surface (ARIS)를 이용해 정지 레이더 시스템을 위한 SAR 이미징을 실현하는 방법을 조사합니다. UAV의 이동성을 활용하여 가상 송수신 경로를 생성하고, SAR 시스템의 성능을 향상시키는 데 중점을 두었습니다.

- **Technical Details**: 제안된 시스템은 ARIS 기반의 Mono-static 레이더 시스템으로, ARIS는 레이더 신호를 증폭하고 반사하여 이미징 지역으로 전송하고, 에코 신호는 같은 경로로 레이더 스테이션으로 돌아옵니다. 본 연구에서는 range-Doppler (RD) 이미징 알고리즘을 제안하며, ARIS의 반사 계수를 최적화하여 신호 대 잡음 비율 (SNR)을 극대화합니다. 비선형 최적화 문제를 해결하기 위해 fractional programming (FP) 및 majorization minimization (MM) 방법을 사용합니다.

- **Performance Highlights**: 시뮬레이션 결과는 ARIS 지원 SAR 이미징의 효과성을 입증하였으며, 제안된 RD 이미징 및 ARIS 최적화 알고리즘이 SAR 시스템의 성능을 성공적으로 향상시키는 것을 보여주었습니다.



### FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancemen (https://arxiv.org/abs/2409.11699)
- **What's New**: 본 논문은 하이브리드 추천 시스템의 새로운 모델인 Flare(협업 필터링 모델과 언어 모델을 통합한 구조)를 소개합니다. Flare는 언어 모델(mT5)과 협업 필터링 모델(Bert4Rec)을 Perceiver 네트워크를 통해 결합하여 더 향상된 추천을 제공합니다.

- **Technical Details**: Flare는 텍스트 기반 아이템 컨텍스트와 아이템 임베딩을 합성하여 richer representation을 생성하는 하이브리드 추천 모델입니다. 두 개의 추천 모델(Bert4Rec의 재구현 및 Flare)을 사용하여 성능을 평가하며, 이 과정에서 두 단계의 평가를 진행합니다.

- **Performance Highlights**: Flare는 작은 데이터 세트에서 경쟁력 있는 성능을 보였으며, 이는 기존의 Bert4Rec 모델보다 뛰어난 결과를 보여줍니다. 또한, Flare는 대규모 데이터 세트에서도 새로운 기준선을 설정하여, 텍스트를 추가함으로써 성능이 개선됨을 입증하였습니다.



### Basket-Enhanced Heterogenous Hypergraph for Price-Sensitive Next Basket Recommendation (https://arxiv.org/abs/2409.11695)
- **What's New**: 이번 연구에서는 가격 정보와 사용자 행동 간의 중요한 상호작용을 고려한 새로운 추천 시스템, Basket-augmented Dynamic Heterogeneous Hypergraph (BDHH)를 제안합니다. 기존의 Next Basket Recommendation (NBR) 모델들이 가격을 간과하고 사용자 간의 상호작용을 완벽하게 반영하지 못하는 한계를 해결하기 위해, 이 방법은 다양한 요소들을 포함하는 복잡한 관계를 탐색할 수 있도록 설계되었습니다.

- **Technical Details**: BDHH 모델은 이질적인 다중 관계 그래프를 활용하여 아이템 특징과 가격 같은 다양한 노드 간의 상관관계를 포착합니다. 이 모델은 아이템-바구니-사용자 간의 관계를 강화하기 위해 바구니 가이드 동적 증대 네트워크를 포함합니다. 이 접근법은 사용자 행동을 풍부하게 나타내고 세계적인 특징 정보의 손실 없이 개인화된 아이템 표현 향상을 달성하는 데 주력합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험 결과, BDHH는 기존의 NBR 모델에 비해 추천 정확성을 크게 향상시켰습니다. 이 연구는 가격을 NBR 문제에 통합한 최초의 시도로, 사용자 행동을 보다 포괄적으로 이해할 수 있는 가능성을 제공합니다.



### LLM-Powered Text Simulation Attack Against ID-Free Recommender Systems (https://arxiv.org/abs/2409.11690)
Comments:
          12 pages

- **What's New**: 이 연구에서는 ID가 없는 추천 시스템의 효과성을 입증함과 동시에, 이러한 시스템이 'Text Simulation attack (TextSimu)'라는 새로운 형태의 공격에 취약하다는 사실을 밝혔습니다. TextSimu는 대규모 언어 모델(LLM)을 활용하여 타겟 아이템의 텍스트 정보를 변조하는 텍스트 독성 공격으로, 인기가 있는 아이템의 특성을 시뮬레이션합니다.

- **Technical Details**: ID 없는 추천 시스템은 사용자 및 아이템의 텍스트 정보를 언어 모델(LLM)을 사용하여 의미적 관계를 구축합니다. 그러나 TextSimu 공격은 통합 인기 추출 모듈과 N-persona 일관성 시뮬레이션 전략을 이용하여 공격자가 인기가 있는 아이템의 특성을 모방하여 타겟 아이템의 텍스트를 조작하는 과정을 포함합니다. 이를 방어하기 위해, 'RewriteDetection'이라는 새로운 탐지 방법을 제안하며, LLM을 사용하여 텍스트의 진위를 검사하는 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, TextSimu는 기존의 독성 공격보다 더 심각한 위협을 제기하며, 제안된 방어 방법은 텍스트의 악성 여부를 효과적으로 탐지할 수 있음을 보여주었습니다. 세 가지 데이터셋에서 진행된 실험을 통해 우리의 접근 방식이 유의미한 방어 성능을 발휘함을 확인했습니다.



### An Enhanced-State Reinforcement Learning Algorithm for Multi-Task Fusion in Large-Scale Recommender Systems (https://arxiv.org/abs/2409.11678)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2404.17589

- **What's New**: 이 논문에서는 추천 시스템(Recommender Systems, RSs)에서 멀티 태스크 융합(Multi-Task Fusion, MTF)에 대한 새로운 접근법인 Enhanced-State RL을 제안합니다. 기존의 방법들이 사용자(feature)만을 상태(state)로 활용하는 반면, 이 방법은 사용자, 아이템(item), 그리고 기타 유용한 특성을 결합한 향상된 상태(enhanced state)를 정의하여 이를 이용합니다.

- **Technical Details**: Enhanced-State RL 방법은 사용자 특징, 아이템 특징, 그리고 기타 유용한 특징들을 하나의 향상된 상태(enhanced state)로 정의하여, 액터(actor)와 크리틱(critic) 학습 프로세스를 개선하였습니다. 이를 통해 사용자-아이템 쌍에 대한 보다 효과적인 액션(action)을 생성할 수 있습니다. 이 새로운 모델링 패턴은 RL-MTF 분야에서 최초로 제안되었습니다.

- **Performance Highlights**: 대규모 추천 시스템에서 수행한 오프라인 및 온라인 실험 결과, Enhanced-State RL 모델은 다른 모델들보다 우수한 성능을 보였습니다. 또한, 이 모델은 이미 6개월 이상 우리의 추천 시스템에 완전히 배포되어 있으며, 기존 기본 모델(baseline) 대비 +3.84%의 사용자 유효 소비(valid consumption) 증가와 +0.58%의 사용자 지속 시간(duration time) 개선을 기록하였습니다.



### Designing Interfaces for Multimodal Vector Search Applications (https://arxiv.org/abs/2409.11629)
Comments:
          12 pages, 8 figures, CIKM 2024 MMSR Workshop

- **What's New**: 이 논문에서는 전통적인 텍스트 검색 시스템 대신 사용 가능한 multimodal vector search의 새로운 기능을 탐구하고 있습니다. CLIP 모델을 이용하여 사용자 요구를 효과적으로 반영하는 방법론과 디자인 패턴을 제시하며, 향상된 정보 검색 체험을 제공합니다.

- **Technical Details**: ClIP 모델은 이미지와 텍스트를 통합하는 응용 프로그램을 위한 벡터 표현 생성에 사용되며, 이는 정보 검색을 위한 쿼리 리파인먼트, 의미 필터링, 맥락화, 무작위 추천 탐색과 같은 개념을 포함하고 있습니다. 논문에서는 lerp(선형 보간법)와 slerp(구면 선형 보간법)를 통해 서로 다른 모달리티를 통합한 벡터를 생성하는 방법도 설명합니다.

- **Performance Highlights**: 저자들은 CLIP 모델 기반의 multimodal vector search의 신속한 프로토타입을 통해 비전문가 사용자가 정보를 명확하게 요청할 수 있는 사용자 인터페이스의 유효성을 입증하고, 실제 검색 시나리오에서 사용자 경험을 향상시키는 방법론을 제안합니다.



### Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation (https://arxiv.org/abs/2409.11598)
- **What's New**: 이번 논문은 Retrieval-Augmented Generation (RAG) 시스템에 공정한 순위 평가를 통합한 최초의 체계적인 평가를 제시합니다. 연구는 RAG 시스템에서 각 관련 항목의 공정한 노출을 측정하여 공정한 순위의 중요성을 강조합니다.

- **Technical Details**: RAG 시스템에서 공정한 순위란, 각각의 관련된 항목이 얼마나 공정하게 노출되는지를 측정하는 것을 의미합니다. 연구는 공정성을 고려하여 세 가지 주제에 대한 결과를 분석하며, 공정한 순위 시스템을 채택하는 RAG 모델이 전통적인 RAG보다 높은 생성 품질을 유지할 수 있음을 발견했습니다. 해당 연구는 7개 데이터셋을 이용하여 9가지 RAG 시스템을 분석하였습니다.

- **Performance Highlights**: 공정한 순위를 포함한 RAG 시스템이 높은 생성 품질을 유지하면서도 기존 RAG 시스템을 초월할 수 있는 사례가 있음을 발견했습니다. 이는 공정성을 보장하면서도 시스템의 효과성을 훼손하지 않고 양질의 서비스를 제공할 수 있음을 나타냅니다.



### A Framework for Ranking Content Providers Using Prompt Engineering and Self-Attention Network (https://arxiv.org/abs/2409.11511)
- **What's New**: 이 연구에서는 콘텐츠 추천 시스템을 위한 콘텐츠 제공자의 순위를 매기는 문제를 다루고 있습니다. 클릭 및 반응과 같은 명시적 사용자의 피드백과 글쓰기 스타일, 게시 빈도와 같은 콘텐츠 기반 기능을 활용하여 주제별 콘텐츠 제공자를 평가하는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 언어 모델을 사용하여 비지도 학습 과제를 위한 데이터셋을 생성함으로써, 셀프 어텐션(self-attention) 네트워크와 함께 Learning to Rank ListWise 작업을 훈련합니다. 이를 통해 콘텐츠 품질, 신뢰성 및 다양성을 높이는 성과를 이룹니다.

- **Performance Highlights**: 온라인 실험을 통해, 제안된 프레임워크는 추천되는 콘텐츠의 품질 및 신뢰성을 개선할 수 있음을 보여주며, 이를 통해 사용자에게 더 나은 추천 서비스를 제공합니다.



### Perceptions of Edinburgh: Capturing Neighbourhood Characteristics by Clustering Geoparsed Local News (https://arxiv.org/abs/2409.11505)
Comments:
          Preprint - paper under submission

- **What's New**: 이번 연구에서는 지역 뉴스 기사를 활용하여 이웃의 특성을 정량화하는 방법론을 제안합니다. 또한, 자연어 처리(Natural Language Processing, NLP)를 통해 뉴스 기사를 분석하고, 지리적 파싱(geoparsing)과 클러스터링(clustering)을 수행하여 이웃에 대한 더욱 심층적인 정보를 얻는 방식을 보여줍니다.

- **Technical Details**: 우리는 도로 수준의 지리적 파싱을 지역에 맞게 조정하고, 전체 뉴스 기사를 클러스터링하여 더욱 세부화된 이웃 특성을 분석합니다. 이를 통해 뉴스 기사에서 추출한 주제들이 현실 세계의 여러 특성을 반영하고 있음을 질적 및 정량적 관점에서 확인하였습니다.

- **Performance Highlights**: 뉴스 데이터를 활용한 이웃 특성 분석 결과는 건강에 미치는 이웃의 영향을 이해하는 데 중요한 기초 자료를 제공합니다. 이 연구 결과는 장소 기반 연구의 새로운 세대를 지원하여, 건강에 영향을 미치는 보다 광범위한 공간적 프로세스(spatial processes) 분석을 가능하게 합니다.



### Generalized compression and compressive search of large datasets (https://arxiv.org/abs/2409.12161)
- **What's New**: 이번 연구는 panCAKES라는 새로운 압축 검색 알고리즘을 제안합니다. 이 알고리즘은 압축된 데이터에서 $k$-NN 및 $ho$-NN 검색을 수행할 수 있도록 설계되어 있습니다. 특징적으로, panCAKES는 단지 관련성이 높은 데이터의 소량만을 복원하여 동작합니다.

- **Technical Details**: panCAKES는 manifold 가설을 활용하여 데이터의 저차원 구조를 기반으로 효과적으로 압축 및 검색할 수 있게 설계되었습니다. 이 알고리즘은 쿼리에 대해 가장 유사한 k개의 데이터 포인트를 찾는 $k$-NN 검색 및 주어진 유사성 임계값 $ho$ 내에 있는 모든 포인트를 찾는 $ho$-NN 검색을 포함합니다. 각 거리 함수에 대해 일반성을 유지하며, 이를 통해效果적인 압축이 가능합니다.

- **Performance Highlights**: panCAKES는 gzip과 유사한 압축 비율을 달성하면서도 $k$-NN 및 $ho$-NN 검색에서 서브 선형 시간 성능을 제공합니다. 다양한 데이터셋에서 벤치마킹을 통해, panCAKES는 기존의 압축 및 검색 방법보다 더욱 효율적임을 demonstrated 하였습니다. 이 논문의 구현코드는 Rust 프로그래밍 언어로 되어 있으며, 오픈 소스 라이센스 하에 제공됩니다.



### Skill matching at scale: freelancer-project alignment for efficient multilingual candidate retrieva (https://arxiv.org/abs/2409.12097)
- **What's New**: 이 논문은 다국어 환경에서 프리랜서와 프로젝트 간의 매칭을 개선하기 위한 새로운 신경망 검색 아키텍처를 제안합니다. 기존 시스템의 한계를 극복하고 보다 효율적인 매칭을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 다국어 언어 모델을 활용하여 프로젝트 설명 및 프리랜서 프로필을 인코딩합니다. 사용자 맞춤형 transformer 아키텍처를 통해 프로필과 프로젝트 구조를 유지하며, 과거 데이터를 활용하여 대조 손실(contrastive loss)로 훈련합니다. 이 모델은 프리랜서의 기술 매칭 유사성을 신뢰성 있게 포착합니다.

- **Performance Highlights**: 여러 실험 결과, 제안된 접근 방식이 기존의 전통적인 방법들보다 우수한 성능을 보여주었으며, 더 나은 기술 매칭 효율성을 제공함을 입증했습니다.



### AlignBot: Aligning VLM-powered Customized Task Planning with User Reminders Through Fine-Tuning for Household Robots (https://arxiv.org/abs/2409.11905)
- **What's New**: 이 논문에서는 사용자 알림과 효과적으로 조정하여 가정용 로봇의 맞춤형 작업 계획을 최적화하기 위해 설계된 새롭고 독창적인 프레임워크인 AlignBot을 소개합니다.

- **Technical Details**: AlignBot은 GPT-4o의 어댑터로 기능하는 세밀하게 조정된 LLaVA-7B 모델을 사용하여 사용자의 다양한 알림(개인화된 선호, 수정 가이드라인 및 상황적 지원 등)을 구조화된 지침 형식의 큐로 변환하여 맞춤형 작업 계획 생성을 안내합니다. 또한 동적 검색 메커니즘을 통합하여 GPT-4o에 대한 프로세스 프롬프트로서 작업과 관련된 과거 성공 사례를 선택합니다.

- **Performance Highlights**: AlignBot은 실제 가정 환경에서 수행된 실험을 통해 맞춤형 작업 계획 품질을 향상시키며, 기존의 LLM 및 VLM 기반 플래너보다 유의미하게 개선된 성과를 나타냈습니다. 결과적으로 AlignBot은 86.8%의 성공률을 기록하여 기본 GPT-4o 모델의 21.6%와 비교하여 65% 향상된 성과를 보였으며, 효과성 또한 4배 이상 증가하였습니다.



### The Factuality of Large Language Models in the Legal Domain (https://arxiv.org/abs/2409.11798)
Comments:
          CIKM 2024, short paper

- **What's New**: 이번 연구는 법률 분야에서 대규모 언어 모델(LLMs)을 지식 기반(KB)으로 사용하는 사실성을 검토합니다. 특히, 모델이 불확실할 때 답변을 보류할 수 있도록 허용한 현실적인 사용 시나리오에서 연구가 진행되었습니다.

- **Technical Details**: 이 연구에서는 다양한 사례법 및 입법에 대한 사실 질문의 데이터셋을 설계하고, 여러 LLM을 동일한 데이터셋으로 다양한 평가 방법(정확 매칭, 별칭 매칭, 퍼지 매칭 등)으로 평가하였습니다. 또한, 모델의 응답 보류 전략과 인-컨텍스트 예시가 사실 정확성을 어떻게 향상시키는지 탐구하였으며, 법률 문서에 대한 추가 사전 훈련이 63%에서 81%로 정확성을 향상시킨 것을 보여주었습니다.

- **Performance Highlights**: 단순 정확 매칭에서 벗어나 별칭 및 퍼지 매칭 방법에서 성능이 크게 향상되었으며, 응답을 보류하거나 구체적인 예시를 제공하는 것이 LLM의 정확성을 높였습니다. 이러한 연구 결과는 법률 기술 응용 프로그램의 실질적인 활용 가능성을 보여줍니다.



### Evaluation of pretrained language models on music understanding (https://arxiv.org/abs/2409.11449)
- **What's New**: 이 논문에서는 음악 정보 검색(Music Information Research, MIR) 애플리케이션에서의 대형 언어 모델(Large Language Model, LLM)의 음악 지식 평가에 초점을 맞추고 있습니다. 특히 대형 언어 모델이 가지는 프롬프트 민감성과 부정 표현 모델링의 한계 등을 지적하고, 이를 바탕으로 새로운 평가 방법론을 제안합니다.

- **Technical Details**: Audioset의 계층적 온톨로지를 사용하여, 각기 다른 장르와 악기에 대해 ‘앵커(anchor)’, ‘양성 긍정(label)’, ‘부정적(label)’ 형식의 트리플릿(triplet)을 생성하고, LLM의 상대적 유사성을 평가합니다. 이 방법은 음악적 지식을 정량화하기 위해 13,633개의 음악 장르와 37,640개의 음악 악기 트리플릿을 수집하였습니다.

- **Performance Highlights**: 리포트에 따르면, 모든 모델에서 높은 정확도가 나타났으나, 일관성이 부족하다는 결과가 나타났습니다. 이는 기존 LLM이 음악 관련 작업을 위한 적절한 사전 훈련 없이 사용될 경우, 성능 향상을 위해 조정이 필요함을 시사합니다.



New uploads on arXiv(cs.CV)

### Vista3D: Unravel the 3D Darkside of a Single Imag (https://arxiv.org/abs/2409.12193)
Comments:
          ECCV'2024

- **What's New**: 새로운 연구인 Vista3D는 단일 이미지에서 3D 객체의 숨겨진 측면을 효율적으로 드러내는 프레임워크입니다. 이는 5분 만에 다양한 3D 객체를 생성할 수 있게 합니다.

- **Technical Details**: Vista3D는 두 단계(phase)로 구성된 접근 방식을 채택하고 있습니다. 첫 번째 단계인 coarse phase에서는 Gaussian Splatting 기술을 사용해 초기 기하학을 빠르게 생성합니다. 두 번째 단계인 fine phase에서는 학습한 Gaussian Splatting에서 직접 Signed Distance Function (SDF)을 추출하고, 이를 differentiable isosurface representation으로 최적화하여 품질을 높입니다. 또한 두 개의 독립적인 implicit 함수를 사용하여 가시적인 부분과 가려진 부분을 포착하는 disentangled representation을 사용합니다.

- **Performance Highlights**: Vista3D는 생성된 3D 객체의 일관성과 다양성을 효과적으로 유지하며, 종합적인 평가를 통해 이 균형을 성공적으로 달성하고 있음을 입증하였습니다.



### Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (https://arxiv.org/abs/2409.12191)
Comments:
          Code is available at this https URL. arXiv admin note: text overlap with arXiv:2408.15262 by other authors

- **What's New**: Qwen2-VL 시리즈는 이전 Qwen-VL 모델의 업그레이드 버전으로, 동적 해상도(naive dynamic resolution) 처리 메커니즘을 도입하여 다양한 해상도의 이미지를 처리하고, 시각적 토큰(visual tokens)을 동적으로 생성할 수 있도록 개선되었습니다.

- **Technical Details**: Qwen2-VL 모델은 Multimodal Rotary Position Embedding (M-RoPE)을 활용하여 텍스트, 이미지 및 비디오 간의 위치 정보를 효과적으로 융합합니다. 또한, 2B, 8B 및 72B 매개변수를 갖춘 버전으로 확장 법칙을 탐색하며, 다양한 해상도와 종횡비에서의 이해력을 향상시킵니다.

- **Performance Highlights**: Qwen2-VL-72B 모델은 DocVQA, InfoVQA, RealWorldQA와 같은 다양한 멀티모달 벤치마크에서 GPT-4o 및 Claude3.5-Sonnet과 유사한 성능을 보여주며, 고해상도 영상 이해 및 다국어 지원 기능을 제공합니다.



### Massively Multi-Person 3D Human Motion Forecasting with Scene Contex (https://arxiv.org/abs/2409.12189)
Comments:
          14 pages, 6 figures

- **What's New**: 이 논문에서는 장기(10초) 인간 모션 예측을 위해 새로운 모델인 Scene-Aware Social Transformer (SAST)를 제안합니다. 이 모델은 다양한 수의 사람들과 객체의 상호작용을 모델링할 수 있으며, 장기 예측에서 환경의 맥락을 고려한 첫 번째 접근방식입니다.

- **Technical Details**: SAST는 시간적 합성곱 인코더-디코더 아키텍처와 Transformer 기반의 병목 구조를 결합하여 모션과 씬 정보를 효율적으로 통합합니다. 이 모델은 denoising diffusion models를 사용하여 조건부 모션 분포를 모델링합니다. 사용된 데이터셋은 'Humans in Kitchens'로, 1~16명과 29~50개의 객체가 동시에 존재하는 환경을 포함합니다.

- **Performance Highlights**: 제안된 모델은 현실감과 다양성 측면에서 다른 접근 방식들을 초월합니다. 다양한 메트릭(지표)과 사용자 연구를 통해 그 성능이 검증되었습니다.



### NSSR-DIL: Null-Shot Image Super-Resolution Using Deep Identity Learning (https://arxiv.org/abs/2409.12165)
- **What's New**: 본 연구는 이미지 데이터에 의존하지 않는 혁신적이고 계산적으로 효율적인 이미지 슈퍼 해상도( ISR ) 알고리즘을 제안합니다. 이 알고리즘은 슈퍼 해상도 이미지(SR)를 생성하는 대신, 열화 공간을 포괄하는 커널의 역을 계산하는 작업으로 ISR을 재정의합니다. 이 방법은 Deep Identity Learning(DIL)을 도입하여 열악한 모델과 그 역 모델 간의 정체 관계를 활용합니다.

- **Technical Details**: 제안된 NSSR-DIL 모델은 기본적으로 10배 이상의 계산 효율성을 보여주며, X2, X3, X4와 같은 다양한 스케일 팩터에 대해 모델 재훈련이 필요 없습니다. 또한, 기존의 자가 감독 방식의 ZSSR와는 달리, LR 이미지 데이터가 없이도 ISR 작업을 학습할 수 있게 합니다. 본 연구는 경량의 CNN인 Linear-CNN(L-CNN)을 사용하여 열화 모델을 훈련합니다.

- **Performance Highlights**: 제안된 NSSR-DIL 모델은 벤치마크 ISR 데이터 세트에서 경쟁력 있는 성능을 시연하며, 실제 응용 프로그램에 더 적합한 매우 효율적인 ISR 모델로 자리매김하고 있습니다.



### Precise Forecasting of Sky Images Using Spatial Warping (https://arxiv.org/abs/2409.12162)
- **What's New**: 이 연구에서는 깊은 학습(deep learning) 기술을 활용하여 더 높은 해상도의 미래 하늘 이미지 프레임을 예측하는 새로운 접근 방식을 제안합니다. 특히, 수평선(horizon) 근처의 구름의 영향을 줄이기 위해 최적의 왜곡(warping) 방법을 도출하였습니다.

- **Technical Details**: 제안된 SkyNet 모델은 4개의 입력 이미지(t-5, t-3, t-1, t)를 사용하여 보다 큰 시간적 맥락을 고려하며, 이는 구름의 진화 예측을 향상하도록 돕습니다. 또한, 기존의 방법들보다 높은 해상도의 하늘 이미지 프레임을 더 정확하게 예측하기 위해 많은 양의 하늘 이미지 데이터셋을 활용하여 훈련 및 검증을 수행하였습니다.

- **Performance Highlights**: SkyNet의 예측 정확도는 이전 구름 예측 방법보다 더 높은 해상도의 메트릭을 보여주며, 예측된 하늘 이미지를 기반으로 미래 GHI(Global Horizontal Irradiance) 값을 추정하는 데 성공했습니다. 통계적으로 예측 가능한 시간이 최대 2분 반(t+5)까지 확장되었습니다.



### JEAN: Joint Expression and Audio-guided NeRF-based Talking Face Generation (https://arxiv.org/abs/2409.12156)
Comments:
          Accepted by BMVC 2024. Project Page: this https URL

- **What's New**: 이 논문에서는 얼굴 표정과 오디오에 기반한 말하는 얼굴 생성의 통합 방법을 제안합니다. 기존 방법은 화자의 정체성 보존에 어려움을 겪거나 얼굴 표정을 충실하게 생성하지 못하는 문제가 있었습니다. 이를 해결하기 위해 NeRF(Neural Radiance Fields) 기반 네트워크를 도입했습니다.

- **Technical Details**: 제안된 방법에서는 단안 비디오(mono videos)를 기반으로 훈련하며, 오디오와 표정에 대한 분리된 표현(disentangled representations)을 학습합니다. 자가 지도 학습(self-supervised learning) 방법을 통해 오디오 특징을 학습하고, 대조 학습(contrastive learning) 기법을 포함시켜 lip motion과 얼굴의 다른 근육 운동을 분리합니다. 그 후, 변환기(transformer) 기반 아키텍처를 통해 표정 특징을 학습하여 장거리 표정을 포착하고, 발화에 특화된 입 움직임과 분리합니다.

- **Performance Highlights**: 정량적 및 정성적 평가를 통해 제안된 방법이 고충실도 말하는 얼굴 비디오를 합성할 수 있음을 보여주었으며, 여기서 얼굴 표정 전이와 입술 동기화(lip synchronization)를 성능적으로 우수하게 달성했습니다.



### MoRAG -- Multi-Fusion Retrieval Augmented Generation for Human Motion (https://arxiv.org/abs/2409.12140)
- **What's New**: MoRAG는 텍스트 기반 인간 동작 생성을 위한 새로운 다중 파트 융합 기반 검색 증강 생성 전략을 소개합니다. 이 방법은 개선된 동작 검색 프로세스를 통해 얻은 추가 지식을 활용하여 동작 확산 모델의 성능을 향상시킵니다. 특히, 대규모 언어 모델(LLMs)을 효과적으로 활용하여 검색 중의 철자 오류 및 재구성 문제를 해결합니다.

- **Technical Details**: 제안된 MoRAG 방법은 다중 파트 검색 전략을 활용하여 제공된 텍스트 설명에 정렬된 각 부분의 동작과 관련된 동작 시퀀스를 검색하기 위한 부분별 독립 동작 검색 모델을 훈련합니다. 검색된 동작 시퀀스는 다음으로 전체 몸 동작 시퀀스를 구성하는 데 결합됩니다. 이 프레임워크는 추가 지식으로서 동작 생성 모델에 통합되어 일반화 가능성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, MoRAG를 통해 구성된 동작 시퀀스는 텍스트 설명의 의미와 더 잘 정렬되며, 생성된 시퀀스의 다양성 또한 개선되었습니다. MoRAG는 확산 기반 동작 생성 모델의 성능을 향상시키는 플러그 앤 플레이 모듈로 작용할 수 있음을 입증했습니다.



### Applications of Knowledge Distillation in Remote Sensing: A Survey (https://arxiv.org/abs/2409.12111)
Comments:
          50 pages, 11 figures and 9 tables

- **What's New**: 본 논문은 원격 감지(Remote Sensing, RS) 분야에서 모델 정확도와 계산 효율성을 균형 있게 맞출 필요성에 대해 논의하고, 이를 위한 중요한 기술인 지식 증류(Knowledge Distillation, KD)의 혁신적인 응용에 대한 포괄적인 검토를 제공합니다.

- **Technical Details**: KD는 복잡한 모델(teacher)에서 간단하고 효율적인 모델(student)로 지식을 전이하는 기법입니다. 이 논문에서는 KD 방법의 기본 개념과 역사적 발전을 소개하며, 모델 압축(model compression), 계산 효율성(computational efficiency), 성능 향상(performance improvement) 등의 장점에 대해 강조합니다.

- **Performance Highlights**: KD 기법은 RS 작업에서 객체 탐지(object detection), 인스턴스 분할(instance segmentation) 등의 여러 실제 적용 사례를 통해 그 유용성을 보여줍니다. KD를 통해 복잡한 네트워크의 지식을 소형 모델로 성공적으로 이전하여 배포할 때 필요한 계산 자원을 줄이면서도 유사한 정확도를 달성할 수 있음을 밝혔습니다.



### SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba (https://arxiv.org/abs/2409.12108)
- **What's New**: ESD(Endoscopic Submucosal Dissection) 수술 단계 인식을 위한 새로운 알고리즘 SPRMamba가 제안되었습니다. 이 알고리즘은 Mamba 기반의 접근 방식을 사용하여 정밀성을 향상시킵니다.

- **Technical Details**: SPRMamba는 시간적인 정보의 장기 모델링을 가능하게 하는 Mamba의 기능을 활용합니다. Scale Residual TranMamba(SRTM) 모듈을 도입하여 세밀한 정보를 더 잘 포착하고, Temporal Sample Strategy(TSS)를 통해 실시간 인식이 가능하도록 처리 속도를 가속화했습니다.

- **Performance Highlights**: ESD385 및 Cholec80 데이터셋에서의 실험 결과, SPRMamba는 기존의 최첨단 기술들보다 뛰어난 성능과 강 robustness를 보여줍니다.



### Brain-Streams: fMRI-to-Image Reconstruction with Multi-modal Guidanc (https://arxiv.org/abs/2409.12099)
- **What's New**: 이번 연구에서는 뇌의 시각 정보 처리 메커니즘 이해를 위한 새로운 방법인 Brain-Streams를 제안합니다. 이 시스템은 fMRI 데이터로부터 시각 자극을 재구성하는 데 있어 텍스트와 시각 가이드를 활용하여 구조적 및 의미론적 이미지 생성을 달성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: Brain-Streams는 다양한 뇌 영역에서의 fMRI 신호를 기반으로 고수준(high-level), 중간 수준(mid-level), 저수준(low-level)의 가이드를 추출합니다. 이와 같은 방식으로, Brain-Streams는 다중 모드(multi-modal) 가이드를 LDM(Latent Diffusion Model)에 제공하여 시각 자극의 정확한 재구성이 가능해집니다.

- **Performance Highlights**: Brain-Streams는 NSD 데이터셋에서 최첨단(State-of-the-art, SOTA) 성능을 기록하며, 복잡한 자연 이미지에서의 시각 자극 재구성에 성공했습니다. 이 시스템은 특히 의미론적 세부사항을 정확히 복원하여 재구성 품질을 향상시킵니다.



### SFDA-rPPG: Source-Free Domain Adaptive Remote Physiological Measurement with Spatio-Temporal Consistency (https://arxiv.org/abs/2409.12040)
- **What's New**: 본 논문은 원천 데이터에 접근하지 않고도 효과적인 도메인 적응을 가능하게 하는 첫 번째 Source-free Domain Adaptation benchmark인 SFDA-rPPG를 제안합니다. 기존 방법들의 한계인 소스 도메인 데이터에 대한 접근성 부족 및 개인 정보 보호 문제를 해결합니다.

- **Technical Details**: 제안된 SFDA-rPPG 프레임워크는 Three-Branch Spatio-Temporal Consistency Network (TSTC-Net)을 통합하여 다양한 도메인 간의 특징 일관성을 개선합니다. 또한 Frequency-domain Wasserstein Distance (FWD)에 기반한 새로운 rPPG 분포 정렬 손실을 제안하여 도메인 간의 전력 스펙트럼 분포를 효과적으로 정렬하고 세 가지 브랜치의 정렬을 강제합니다.

- **Performance Highlights**: 광범위한 교차 도메인 실험 및 블라인드 연구를 통해 SFDA-rPPG 방법의 효과를 입증하였으며, FWD 손실이 분포 정렬에서 기여한 바가 크다는 것을 강조합니다. 제공되는 코드와 함께 성능이 우수하다는 것을 확인하였습니다.



### Multi-Sensor Deep Learning for Glacier Mapping (https://arxiv.org/abs/2409.12034)
Comments:
          This article will be a chapter of the book Deep Learning for Multi-Sensor Earth Observation, to be published by Elsevier

- **What's New**: 이번 연구는 20만 개 이상의 빙하(glaciers)가 해수면 상승(sea-level rise)과 물 자원 관리(water resource management)에 미치는 영향을 강조하고, 위성 기반 지구 관측(satellite-based Earth Observation) 기법을 활용하여 빙하를 효과적으로 맵핑(mapping)하는 방법을 제시합니다. 특히, 심층 학습(deep learning) 기법이 결합된 다중 센서(remote sensing) 데이터를 이용한 새로운 접근 방식을 소개합니다.

- **Technical Details**: 연구에서는 다중 센서(remote sensing) 데이터를 결합하여 심층 학습(deep learning) 기술을 활용함으로써 빙하를 더 잘 특정하고(즉, 맵핑(maps)할 수 있는 방법을 탐구합니다. 다양한 지역 및 전세계 빙하 인벤토리(glacier inventories)를 활용하여 더 정확한 분석이 가능함을 설명합니다.

- **Performance Highlights**: 특히, 밝히기 어려운 이물질이 덮인(below-covered) 빙하와 해양과 접촉하는 빙하(calding glaciers)에 대한 다중 센서 원거리 감지 기술의 가능성을 강조합니다. 시각적 이미지를 통해 계절적 눈 덮임(seasonal snow cover), 변화하는 이물질(깎인 것) 덮임(changing debris coverage), 그리고 빙하 전선(glacier fronts)과 주변 바다 얼음(sea ice)을 구별하는 것과 같은 도전 과제를 시각적으로 나타내며, 빙하 변화 탐지와 관련된 몇 가지 중요한 장점과 도전 과제를 함께 제시합니다.



### PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba (https://arxiv.org/abs/2409.12031)
Comments:
          Accepted by CCBR 2024

- **What's New**: PhysMamba는 얼굴 비디오에서 장거리 생리학적 의존성을 효율적으로 표현하기 위한 Mamba 기반 프레임워크입니다. Temporal Difference Mamba 블록을 도입하여 지역 동적 차이를 향상시키고, 장거리 시공간 컨텍스트를 모델링합니다.

- **Technical Details**: Temporal Difference Mamba (TD-Mamba) 블록을 통해 세밀한 지역 시간 역학(shape) 기반으로 장거리 시공간 의존성을 캡쳐합니다. 또한, 다중 스케일 생리학적 특성을 융합하기 위해 Dual-stream SlowFast 아키텍처를 활용합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에서 PhysMamba의 우수성과 효율성을 입증하기 위한 실험을 광범위하게 수행했습니다. PhysMamba는 이전의 CNN 및 Transformer 기반 방법들과 비교해 뛰어난 성능을 보여줍니다.



### On Vision Transformers for Classification Tasks in Side-Scan Sonar Imagery (https://arxiv.org/abs/2409.12026)
- **What's New**: 본 연구는 Side-scan sonar (SSS) 이미지를 활용한 이진 분류 작업에서 Vision Transformers (ViTs)와 일반적으로 사용되는 Convolutional Neural Networks (CNN) 아키텍처를 비교하여 ViTs의 우수성을 보여주고 있습니다.

- **Technical Details**: 이 연구에서는 ResNet, ConvNext와 같은 CNN 아키텍처와 ViT 모델을 종합적으로 평가하여 SSS 이미지의 이진 분류 효과를 측정하였습니다. 데이터셋은 다양한 해양 바닥 유형을 포함하고 있으며, 수동 객체의 존재 여부에 따라 균형을 이루고 있습니다. ViT 모델은 f1-score, precision, recall, accuracy 메트릭에서 우수한 성능을 보여주었으나, 더 많은 계산 자원을 요구합니다.

- **Performance Highlights**: ViT 기반 모델은 CNN 아키텍처에 비해 이진 이미지 분류 성능이 뛰어나며, 주로 rocky 또는 ripple sand와 같은 다양한 해저 질감을 처리하는 데 효과적입니다. 그러나 계산 효율성 측면에서는 CNNs가 더 나은 성능을 보여주며, 자원 제약 환경에서의 배포에 적합합니다.



### LEMON: Localized Editing with Mesh Optimization and Neural Shaders (https://arxiv.org/abs/2409.12024)
- **What's New**: 새로운 연구인 LEMON에서는 텍스트 지시 사항을 바탕으로 다각형 메시를 편집하는 효율적인 파이프라인을 소개합니다. 이 방법은 기존의 장르에서 벗어나, 메시와 뷰 간의 일관성을 유지하는 새로운 접근 방식을 채택하고 있습니다.

- **Technical Details**: LEMON은 neural deferred shading과 localized mesh optimization을 결합한 방식으로 작동합니다. 이 파이프라인에서는 중요 정점을 식별하기 위해 segmentation 모델을 사용하며, 입력 이미지를 텍스트-이미지 diffusion 모델과 결합하여 메시를 동적으로 변형합니다.

- **Performance Highlights**: DTU 데이터셋을 통해 평가된 결과, LEMON은 현재의 최첨단 방법들보다 더 빠르게 세밀하게 편집된 메시를 생성하는 성능을 보여주었습니다.



### Computational Imaging for Long-Term Prediction of Solar Irradianc (https://arxiv.org/abs/2409.12016)
- **What's New**: 이 논문에서는 클라우드에 의한 태양의 폐색을 예측하기 위해 새로운 카타디옵트릭 시스템을 설계 및 배치하였습니다. 이 시스템은 하늘을 균일한 해상도로 촬영할 수 있는 장비로, 기존의 비디오 이미지와 달리 지평선 근처의 클라우드를 정확하게 감지할 수 있습니다.

- **Technical Details**: 본 연구는 고유의 하이퍼볼로이달 미러(hyperboloidal mirror)를 이용하여 하늘 이미지를 촬영하는 새로운 스카이 이미저(sky imager)를 구현했습니다. 이 시스템은 풍향과 속도를 고려하여 클라우드의 운동 예측을 위한 2D 시공간(slice of spatio-temporal) 데이터를 선택하는 알고리즘을 활용합니다.

- **Performance Highlights**: 이 시스템은 최대 30분의 태양 폐색 및 일사량 예측을 성공적으로 수행하였으며, 이전 연구들의 2-3분 예측 시간과 비교했을 때 월등한 성능 향상을 보여주었습니다.



### BRDF-NeRF: Neural Radiance Fields with Optical Satellite Images and BRDF Modelling (https://arxiv.org/abs/2409.12014)
- **What's New**: 본 논문에서는 복잡한 지구 표면의 비등방성 반사율을 위성 이미지를 통해 이해하는 데 중점을 두고 있습니다. 특히, Neural Radiance Fields (NeRF)를 활용하여 Bidirectional Reflectance Distribution Function (BRDF)을 효과적으로 추정하는 BRDF-NeRF를 제안합니다.

- **Technical Details**: BRDF-NeRF는 Rahman-Pinty-Verstraete (RPV) 모델을 명시적으로 추정하는 알고리즘으로, 위성 이미지에서 단 3~4장의 이미지로부터 새로운 뷰를 합성할 수 있는 능력을 가지고 있습니다. 이 방법은 Djibouti와 Lanzhou의 다양한 데이터를 활용하여 검증되었습니다.

- **Performance Highlights**: 실험 결과, BRDF-NeRF는 훈련 데이터와 멀리 떨어진 방향에서 고품질의 디지털 표면 모델 (DSM)을 생성할 수 있으며, 이는 환경 모니터링 및 지구 관측에 중요한 기여를 할 것으로 기대됩니다.



### Mixture of Prompt Learning for Vision Language Models (https://arxiv.org/abs/2409.12011)
- **What's New**: 이 연구는 기존의 단일 소프트 프롬프트에서의 두 가지 주요 과제를 해결하기 위해 라우팅 모듈이 포함된 혼합 소프트 프롬프트 학습 방법을 제안합니다. 이 방법은 데이터셋의 다양한 스타일을 포착하고, 각 인스턴스에 가장 적합한 프롬프트를 동적으로 선택합니다.

- **Technical Details**: 제안된 방법은 소프트 프롬프트의 초기화를 수동으로 설계된 템플릿의 토큰 임베딩을 사용하여 실행하며, 대조 손실(constrastive loss)을 적용하여 소프트 프롬프트의 텍스트 피쳐가 하드 프롬프트의 피쳐 근처에 있도록 합니다. 최적의 프롬프트를 선택하기 위해 하드 프롬프트 유도 게이팅 손실을 도입하여 신뢰성을 높입니다.

- **Performance Highlights**: 이 방법은 11개의 데이터셋에서 검증되었으며, 기존의 기준 대비 소수 샷 학습(few-shot learning), 도메인 일반화(domain generalization), 기본-신규 일반화(base-to-new generalization) 시나리오에서 뚜렷한 성능 개선을 보여주었습니다.



### ChefFusion: Multimodal Foundation Model Integrating Recipe and Food Image Generation (https://arxiv.org/abs/2409.12010)
- **What's New**: 새롭고 혁신적인 음식 컴퓨팅 모델인 ChefFusion을 소개합니다. 이 모델은 t2t, t2i, i2t, it2t, t2ti와 같은 다양한 멀티미디어 작업을 수행할 수 있는 진정한 멀티모달리티를 구현하였습니다.

- **Technical Details**: ChefFusion은 선행 학습된 Transformer 기반의 LLM(large language model)과 이미지 인코더 및 디코더 모델을 활용하여 음식 이해, 인식, 레시피 생성 및 음식 이미지 생성을 포함한 여러 작업을 처리합니다. 이 모델은 기존 모델들보다 더 넓은 기능 범위를 제공하며, 특히 음식 이미지 생성 및 레시피 생성 작업에서 뛰어난 성능을 보입니다.

- **Performance Highlights**: ChefFusion은 다른 저명한 음식 컴퓨팅 방법과 비교하여 멀티모달 능력과 기능에서 우수한 성능을 보여주며, 특히 음식 이미지 생성 및 레시피 생성 작업에서 현저히 높은 성능으로 주목받고 있습니다.



### Panoptic-Depth Forecasting (https://arxiv.org/abs/2409.12008)
- **What's New**: 본 연구는 기존의 semantic 및 panoptic scene forecasting 방법의 한계를 극복하고, panoptic-depth forecasting 작업을 제안하였다. 이 작업은 monocular 카메라 이미지를 기반으로 미래의 panoptic segmentation 및 depth maps를 공동으로 예측한다.

- **Technical Details**: panoptic-depth forecasting는 카메라 이미지의 과거 시퀀스를 입력으로 받아서 일련의 미래 프레임에서 각 픽셀에 대해 semantic class, instance ID 및 depth 값을 예측하는 작업이다. PDcast 아키텍처는 transformer 기반의 인코더 및 다양한 작업별 디코더를 포함하여 rich spatio-temporal representations를 학습한다.

- **Performance Highlights**: PDcast 아키텍처는 KITTI-360과 Cityscapes 데이터셋에서 세 가지 forecasting 작업을 통해 효과적임을 입증하였다. 평가 지표 PDC-Q는 panoptic 예측 품질과 depth 정확성을 일관되게 정량화 하여, 제안된 방법의 효율성을 잘 나타낸다.



### Intraoperative Registration by Cross-Modal Inverse Neural Rendering (https://arxiv.org/abs/2409.11983)
Comments:
          Accepted at MICCAI 2024

- **What's New**: 본 논문에서는 신경외과 수술 중 3D/2D 내부 등록을 위한 새로운 접근법, 즉 cross-modal inverse neural rendering을 제시합니다. 이 방법은 해부학적 구조와 외관을 사전 및 수술 중에 분리하여 처리하며, 다중 스타일 하이퍼 네트워크를 통해 Neural Radiance Field의 외관을 제어하는 방식으로 이 분리를 달성합니다.

- **Technical Details**: 제안된 방법은 Neural Radiance Field (NeRF)를 사용하여 6도의 자유도 (6-DoF) 포즈 추정 문제를 해결합니다. 이 방법은 사전 수술 이미지와 수술 중 이미지를 정렬하는 과정을 통해 최적화를 수행하며, 임플리트 신경 표현을 통해 수술 카메라의 위치를 정량화합니다. 크라니오토미 후 수술 중에 얻은 이미지와의 차이를 최소화하여 포즈를 계산합니다.

- **Performance Highlights**: 실험 결과, 이 방법이 기존의 최첨단 기법들보다 우수한 성능을 보이는 것으로 확인되었습니다. 임상 사례에서 수집한 환자의 데이터에 대한 후향적 테스트 결과가 포함되어 있으며, 현재의 임상 기준을 충족하고 있습니다.



### MitoSeg: Mitochondria Segmentation Too (https://arxiv.org/abs/2409.11974)
- **What's New**: 이 논문에서는 미토콘드리아 분리(segmentation)를 자동으로 수행할 수 있는 소프트웨어 솔루션인 MitoSeg를 소개합니다. 이 소프트웨어는 전자현미경 단층촬영(EMT) 이미지의 미토콘드리아 경계를 강조하고, 3D 메쉬를 생성하는 기능을 갖추고 있습니다.

- **Technical Details**: MitoSeg는 주로 구성 요소로 OpenCV4, Boost 및 yaml-cpp 라이브러리를 사용하여 C++로 개발되었습니다. 이 소프트웨어는 전처리, 경계 검출 및 분할을 포함하는 세 가지 주요 단계를 포함하여 작동합니다. 전처리 단계에서는 이미지의 ROI (Region of Interest)를 자동 정의하고 대조를 조정한 후, Hessian matrix 기반의 능선 검출을 수행하여 미토콘드리아와 내부 구조를 식별합니다.

- **Performance Highlights**: MitoSeg는 기본적으로 CNN을 사용한 기존 방식과 달리 선행 학습(learning) 과정 없이 자동으로 미토콘드리아를 분할할 수 있습니다. 높은 해상도의 세포 내 이미지 데이터셋을 요구하며, 사용자 지정 가능한 설정을 통해 연구자가 소프트웨어를 유연하게 조정할 수 있습니다. 또한, Docker 환경에서 다양한 운영 체제에서 실행될 수 있어 접근성과 활용성이 높습니다.



### Unveiling the Black Box: Independent Functional Module Evaluation for Bird's-Eye-View Perception Mod (https://arxiv.org/abs/2409.11969)
- **What's New**: 이번 연구에서는 BEV (Bird's-Eye-View) 인식 모델에 대한 독립적인 기능 모듈 평가를 위한 새로운 프레임워크인 BEV-IFME (Independent Functional Module Evaluation for Bird's-Eye-View Perception Model)를 제안합니다. 이 프레임워크는 모듈의 특징 맵을 Ground Truth와 비교하여 유사성을 정량화하며, 개별 기능 모듈의 학습 성숙도를 평가합니다.

- **Technical Details**: BEV-IFME의 핵심은 두 단계 Alignment AutoEncoder를 활용하여 특징 맵 인코딩과 표현 정렬 과정을 통해 명시적인 정보를 보존하고 특성 구조의 일관성을 유지하는 것입니다. 기능 모듈의 학습 성숙도를 평가하는 메트릭은 Similarity Score로, BEV 메트릭과 강한 양의 상관관계를 보여주며, 평균 상관계수는 0.9387입니다.

- **Performance Highlights**: BEV-IFME는 기능 모듈의 독립적인 평가 및 계층 최적화를 가능하게 하여, 전통적인 평가 방법에서 탈피합니다. 또한, 다양한 기능 모듈에 쉽게 적용 가능하고 안정성을 유지하며, 학습 매개변수를 조정하거나 최적 모델을 선택하는 데 사용할 수 있습니다.



### A Chinese Continuous Sign Language Dataset Based on Complex Environments (https://arxiv.org/abs/2409.11960)
Comments:
          11 pages, 3 figures

- **What's New**: 본 연구는 실제 환경에서 수집된 복잡한 배경을 반영한 중국 연속 수화 데이터셋 CE-CSL을 새롭게 구축하였다. 이는 기존의 데이터셋들이 가지는 단일 배경과 제한된 조명 조건의 한계를 극복하고자 한다.

- **Technical Details**: 연구에서 제안된 TFNet 모델은 프레임 수준의 특징을 추출한 후, 시각적 및 시간적 (temporal) 정보를 사용하여 순차적 특징을 따로 추출하고 융합하는 방식으로, 연속 수화 인식의 효율성과 정확성을 높인다.

- **Performance Highlights**: 실험 결과, CE-CSL 데이터셋에서 제안된 방법이 현저한 성능 향상을 보여주었으며, 이는 복잡한 배경에서도 효과적으로 작동함을 입증했다. TFNet 모델은 세 가지 다른 공개 CSL 데이터셋에서도 경쟁력 있는 결과를 보였다.



### Tracking Any Point with Frame-Event Fusion Network at High Frame Ra (https://arxiv.org/abs/2409.11953)
- **What's New**: 이번 연구에서는 이미지 프레임과 이벤트 데이터를 융합하여 어떤 포인트든 추적할 수 있는 데이터 기반 추적기 FE-TAP을 제안합니다. 이 시스템은 이상적인 성능을 제공하기 위해 이미지 생성 과정에서 이벤트에 의해 유도된 정보를 활용하는 진화 융합 모듈(EvoFusion)을 설계했습니다.

- **Technical Details**: EvoFusion 모듈을 통해 다양한 프레임 속도를 가진 이미지와 이벤트를 융합하여 점의 궤적과 특성을 반복적으로 업데이트하는 트랜스포머 기반 정제 전략을 도입했습니다. 슬라이딩 윈도우 방식으로 작동하여 스페이셜(Spatial) 및 시간적(Temporal) 관계를 포착하며, 모든 비트가 아닌 점을 추적할 수 있는 강력한 모듈을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FE-TAP은 기존의 최첨단 접근 방식보다 더 나은 성능을 보여주었고, EDS 데이터 세트에서 기대되는 특성 연령을 24% 향상시켰습니다. 또한 실제 드라이브 시나리오에서 알고리즘의 강건성을 검증했습니다.



### GaussianHeads: End-to-End Learning of Drivable Gaussian Head Avatars from Coarse-to-fine Representations (https://arxiv.org/abs/2409.11951)
Comments:
          ACM Transaction on Graphics (SIGGRAPH Asia 2024); Project page: this https URL

- **What's New**: 이 논문에서는 다중 시점 이미지에서 실시간으로 동적이고 변형 가능한 인간 얼굴 아바타를 생성하는 새로운 방법을 제안합니다. 기존의 방법들보다 더 복잡한 표정과 머리의 포즈 변화를 효율적으로 모델링합니다.

- **Technical Details**: 논문은 변형 가능한 템플릿 메쉬와 3D Gaussian primitives를 활용하여 코스 투 파인(coarse-to-fine) 전략으로 얼굴의 복잡한 동역학을 모델링합니다. 입력 프레임에서 풍부한 얼굴 특징을 추출하여 조잡한 얼굴 기하부를 변형한 후, 이를 기반으로 3D Gaussian을 초기화及精炼 합니다. 글로벌 리지드(head pose) 변환 매개변수와 함께 전반적인 모델을 학습합니다.

- **Performance Highlights**: 저자들은 다양한 데이터세트에서 기존 방법과 비교하여 성능을 입증하며, 75 FPS에서 960x540 해상도로 고품질 렌더링을 달성할 수 있음을 보여주었습니다. 이 방법은 특히 큰 변형이 있는 표정 생성에 높은 정확도를 자랑합니다.



### Differentiable Collision-Supervised Tooth Arrangement Network with a Decoupling Perspectiv (https://arxiv.org/abs/2409.11937)
Comments:
          16 pages, 13 figures

- **What's New**: 본 연구에서는 DTAN이라는 차별화된 충돌 감독 치아 배열 네트워크를 제안하며, 이 네트워크가 치아 배열 작업을 독립적으로 처리할 수 있도록 함수적 모형을 강화합니다.

- **Technical Details**: DTAN은 최종 치아 자세에 대한 숨겨진 특징을 먼저 예측하고, 이 정보를 사용하여 시작과 목표 치아 간의 모션을 회귀하는 방법을 도입합니다. 이를 통해 기하학적 및 위치적 특성을 포함하는 특징의 일관성을 유지하며, 점군 데이터(Point Cloud Data)에 대한 차별화 가능한 충돌 손실 함수를 적용합니다.

- **Performance Highlights**: 비교 연구에 따라, 제안된 방법이 이전 최첨단(State-of-the-Art) 방법들에 비해 정확성과 속도에서 현저히 개선된 성능을 보였습니다. 또한, 다양한 결과 조절 기능을 제공하여 치아 배열 과정을 더욱 효율적으로 수행할 수 있습니다.



### Agglomerative Token Clustering (https://arxiv.org/abs/2409.11923)
Comments:
          ECCV 2024. Project webpage at this https URL

- **What's New**: Agglomerative Token Clustering (ATC)는 이미지 분류, 이미지 생성, 객체 탐지 및 분할 작업에서 기존의 토큰 병합 및 가지 치기 방법을 일관되게 초월하는 새로운 토큰 병합 방법입니다. ATC는 추가 학습 가능한 파라미터 없이 하향식 계층적 클러스터링을 통해 클러스터를 병합합니다.

- **Technical Details**: ATC는 상향식 계층적 클러스터링(agglomerative clustering) 알고리즘을 사용하여 유사한 클러스터를 병합합니다. 이 과정에서, 다양한 작업에 걸쳐 우수한 성능을 보이며 특히 낮은 유지율을 적용할 때 효과적입니다. 이는 전통적인 메서드와 비교하여 정보 중복을 방지하고 성능을 보존하는데 도움을 줍니다.

- **Performance Highlights**: ATC는 이미지 분류, 이미지 생성 및 객체 탐지 및 분할 작업의 모든 테스트에서 최신 성능을 달성하며, 사전 조정 없이 사용하는 것만으로도 우수한 성능을 발휘합니다. 이는 기존의 수정된 방법들과 비슷한 성능을 보여줍니다.



### Generation of Complex 3D Human Motion by Temporal and Spatial Composition of Diffusion Models (https://arxiv.org/abs/2409.11920)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 논문에서는 훈련 단계에서 본 적이 없는 행동 클래스를 위한 현실적인 3D 인간 동작 생성의 도전 과제를 다룹니다. 저희의 방법은 복잡한 동작을 훈련 중 관찰된 단순한 동작으로 분해하여, GPT 모델에 내재된 인간 동작 지식을 활용합니다. 단순한 동작은 확산 모델의 특성을 이용하여 하나의 현실적인 애니메이션으로 결합됩니다.

- **Technical Details**: 저희 방법(MCD)은 생성할 동작을 설명하는 텍스트 주석과 그 지속 시간을 입력 받아, GPT를 통해 훈련 세트에 있는 하나 이상의 알려진 하위 동작으로 분해합니다. 추론(Inference) 단계에서 각 하위 동작을 개별적으로 처리하고, 해당 시작 및 종료 시간에 따라 결합합니다.

- **Performance Highlights**: 기본 동작 및 복잡한 동작으로 나눈 데이터 세트를 평가하여, 저희 방법이 단순한 텍스트 조건화 생성 및 문헌의 다른 조합 전략에 비해 더 나은 결과를 달성했음을 보여주었습니다.



### LLM-wrapper: Black-Box Semantic-Aware Adaptation of Vision-Language Foundation Models (https://arxiv.org/abs/2409.11919)
Comments:
          EVAL-FoMo workshop, ECCV 2024

- **What's New**: 본 연구에서는 LLM-wrapper라는 새로운 접근 방식을 제안합니다. 이는 Vision Language Models (VLMs)를 '블랙박스' 방식으로 조정하는 방법으로, 대규모 언어 모델(LLMs)을 이용하여 VLM의 출력을 이유를 분석합니다.

- **Technical Details**: LLM-wrapper는 주어진 복잡한 텍스트 쿼리에 대해 VLM이 예측한 결과를 자연어로 변환한 후 LLM에 제공하여 최상의 상자를 선택하도록 요청하는 방식입니다. 이 과정에서 LLM은 공간적 및 의미적 추론 능력을 활용하여 최적의 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, LLM-wrapper는 모든 VLM-LLM 조합에서 P@1 점수를 9.5점 이상 향상시키며, VLM 성능을 크게 개선했습니다. 이를 통해 모델에 대한 무관한 접근 방식을 통해 VLM의 결과를 개선할 수 있음을 보여줍니다.



### Finding the Subjective Truth: Collecting 2 Million Votes for Comprehensive Gen-AI Model Evaluation (https://arxiv.org/abs/2409.11904)
- **What's New**: 이번 연구는 Rapidata 기술을 활용하여 텍스트-이미지 모델의 성과를 효율적으로 평가할 수 있는 새로운 주석(annotation) 프레임워크를 제시합니다. 이를 통해 4,512개의 이미지에 대해 200만 건 이상의 주석을 수집하였으며, DALL-E 3, Flux.1, MidJourney, Stable Diffusion과 같은 4개의 주요 모델을 스타일 선호도, 일관성, 텍스트-이미지 정합성에 대해 평가했습니다.

- **Technical Details**: 주요 기준은 스타일(style), 일관성(coherence), 텍스트-이미지 정합성(text-to-image alignment)으로 나뉘어 있으며, 주석자는 각기 다른 모델에서 생성된 이미지 쌍을 제시받고, 선호하는 이미지를 선택하게 됩니다. 이 연구에는 282개의 이미지 생성 프롬프트가 포함되어 있으며, 다양한 문화적 배경을 가진 글로벌 주석자 풀을 활용하여 더 많은 사용자 피드백을 수집할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이번 연구는 기존의 텍스트-이미지 생성 벤치마크와 비교했을 때 200만 건의 투표를 바탕으로 진행된 가장 포괄적인 평가를 제공하며, 다양한 주석자 인구통계학적 특성을 반영하여 편향의 위험을 감소시켰습니다.



### ABHINAW: A method for Automatic Evaluation of Typography within AI-Generated Images (https://arxiv.org/abs/2409.11874)
- **What's New**: 본 논문은 AI가 생성한 이미지 내 텍스트 및 타이포그래피의 성능을 정량화하기 위한 새로운 평가 매트릭스를 소개합니다. 기존의 텍스트 정확성을 평가하는 방법들이 존재하지만, AI 생성 이미지 내의 텍스트 평가에는 한계가 있었으며, 이를 해결하기 위해 발음치기(“zero-shot”) 방식을 도입하였습니다.

- **Technical Details**: 우리는 레터 바이 레터 매칭(letter by letter matching) 전략을 사용하여 참조 텍스트와 AI 생성 텍스트 간의 정확한 매칭 점수를 계산합니다. 또한, 단어 반복, 대소문자 구분, 단어 혼합, 불규칙적 문자 조합 등과 같은 여러 중복성을 처리하기 위해 점수를 계산하는 새로운 접근 방식을 제안합니다. 과도한 텍스트를 처리하기 위해 브레비티 조정(brevity adjustment)이라는 새로운 방법도 개발하였습니다.

- **Performance Highlights**: 기존의 CLIP SCORE 및 T2I-CompBench++와 같은 벤치마크 방법들이 존재하나, 여전히 AI 생성 이미지 내 텍스트 및 타이포그래피 평가에서 체계적인 접근이 부족합니다. 우리의 접근 방식은 신뢰성을 높이고, AI 생성 이미지의 의미를 풍부하게 하는 동시에 그래픽 디자인 산업을 민주화하는 데 기여할 것입니다.



### SpheriGait: Enriching Spatial Representation via Spherical Projection for LiDAR-based Gait Recognition (https://arxiv.org/abs/2409.11869)
- **What's New**: 이 논문에서는 LiDAR(대기 탐지 및 거리 측정 기술)를 기반으로 한 3D 포인트 클라우드에서 동적 특징을 추출하고 강화하는 새로운 방법, SpheriGait를 제안합니다. 이 방법은 기존의 평면 프로젝션 방식 대신 구형 프로젝션을 사용하여 3D 공간 특성을 직접적으로 캡처하고 조명 조건의 영향을 줄이며 개인 정보 보호를 보장합니다.

- **Technical Details**: SpheriGait는 LiDAR 범위 뷰의 3D 포인트 클라우드를 구형 프로젝션을 통해 깊이 이미지로 변환한 후, 이 깊이 이미지에서 동적 주요 특징을 추출하는 합성곱 신경망을 활용합니다. 또한, 새로운 네트워크 블록인 DAM-L을 도입해 동적 표현을 향상시킵니다. 이를 통해 관절 움직임과 같은 고유한 특성을 더 잘 강조할 수 있습니다.

- **Performance Highlights**: SpheriGait는 SUSTech1K 데이터셋에서 최첨단 성능을 기록했으며, 구형 프로젝션 방법이 다른 LiDAR 기반 보행 인식 방법의 성능을 향상시키는 보편적인 데이터 전처리 기술로서의 가능성을 입증하였습니다. 이 방법은 유연성과 실용성을 갖추고 있어 다양한 실세계 응용 분야에서 효과적으로 활용될 수 있습니다.



### Distillation-free Scaling of Large SSMs for Images and Videos (https://arxiv.org/abs/2409.11867)
- **What's New**: 본 논문에서는 Mamba 기반 아키텍처의 확장성 문제를 해결하기 위해 Mamba-Attention 간섭 아키텍처인 StableMamba를 제안합니다. 이는 이미지 및 비디오 인식에서 스케일 문제를 해결하며, 추가 기술(예: knowledge distillation) 없이도 향상된 성능을 이룹니다.

- **Technical Details**: Mamba 모델은 S6 selective-scan 알고리즘을 사용하여 데이터 의존적인 변형을 도입한 상태 공간 모델(State-Space Models, SSM)로, 긴 시퀀스의 문맥 모델링을 개선합니다. 그러나 기존 Mamba 아키텍처는 파라미터 수 증가에 따라 성능 개선이 미미하며, JPEG 압축과 같은 일반적인 손상에 둔감합니다. StableMamba는 이러한 두 가지 문제를 모두 해결하며, 강력하고 효율적인 간섭 아키텍처를 제시합니다.

- **Performance Highlights**: 제안된 StableMamba 아키텍처는 ImageNet-1K, Kinetics-400 및 Something-Something-v2 기준에서 기존 Mamba 기반 아키텍처 대비 최대 1.7의 정확도 증가를 보여줍니다.



### End-to-End Probabilistic Geometry-Guided Regression for 6DoF Object Pose Estimation (https://arxiv.org/abs/2409.11819)
- **What's New**: 본 논문에서는 EPRO-GDR (End-to-End Probabilistic Geometry-Guided Regression)이라는 새로운 알고리즘을 도입하여 6D 객체 자세 추정 문제를 해결하고자 합니다. 기존의 방법에서 단일 자세를 예측하는 것과 달리, 여러 유의미한 자세 후보를 확률 밀도 함수로 예측합니다.

- **Technical Details**: EPRO-GDR은 GDRNPP의 개선된 버전으로, 단일 객체 관측에 대해 여러 자세를 샘플링할 수 있는 능력을 가지고 있습니다. 이 방법은 특히 씬 레벨 최적화에서 유용하며, 개별 객체 자세 추정의 정확성을 향상시키는 데에도 기여합니다. 또한, 평가 절차는 BOP (Benchmark for 6D Object Pose Estimation) Challenge를 기반으로 하여 수행되었습니다.

- **Performance Highlights**: EPRO-GDR은 BOP의 핵심 데이터셋 중 3개인 LM-O, YCB-V, 및 ITODD에서 기존의 GDRNPP보다 우수한 정량적 결과를 보여주었습니다. 이를 통해 자세 분포 예측이 단일 뷰 자세 추정을 개선할 수 있음을 확인하였습니다.



### EFCM: Efficient Fine-tuning on Compressed Models for deployment of large models in medical image analysis (https://arxiv.org/abs/2409.11817)
- **What's New**: 최근 의료 분야에서 대규모 딥러닝 모델이 발전하면서 의료 이미지 분석과 진단에서 뛰어난 성과를 보여주고 있습니다. 하지만 이러한 모델의 거대한 파라미터 수로 인해 메모리 및 추론 지연 문제를 겪고 있습니다. 본 연구에서는 효율적인 압축 모델에 대한 미세 조정(EFCM) 프레임워크를 제안하며, 이에는 비지도 학습을 통한 특성 증류와 미세 조정의 두 단계가 포함됩니다.

- **Technical Details**: EFCM 프레임워크는 비지도 특성 증류(unsupervised feature distillation)와 미세 조정(fine-tuning) 두 단계를 갖고 있습니다. 증류 단계에서는 특성 투영 증류(Feature Projection Distillation, FPD) 방법이 제안되며, TransScan 모듈을 활용하여 수용 필드 크기를 적응적으로 조정하여 모델의 지식 흡수 능력을 향상시킵니다. 슬라이드 수준의 미세 조정 단계에서는 재사용(CLAM), 재훈련(CLAM), End2end Train CLAM을 비교하여 성능을 최적화합니다.

- **Performance Highlights**: 본 연구의 실험 결과, EFCM 프레임워크는 TCGA-NSCLC와 TCGA-BRCA 데이터셋에서 BROW 대규모 모델에 비해 4.33%의 정확도(ACC) 향상과 5.2%의 AUC 향상을 이루어내며, 슬라이드 수준 병리 이미지 문제를 처리하는 데 있어 정확도와 효율성을 크게 개선합니다. 또한, 모델 추론 효율성을 분석한 결과, 증류 미세 조정 방법의 높은 효율성이 강조되었습니다.



### SymFace: Additional Facial Symmetry Loss for Deep Face Recognition (https://arxiv.org/abs/2409.11816)
Comments:
          11 Pages, 6 Figures, 5 Tables, Submitted for WACV 2025

- **What's New**: 이 논문에서는 얼굴 인식 알고리즘 개선을 위해 얼굴 대칭(symmetry)을 활용한 새로운 손실 함수인 SymFace loss를 제안합니다. 이 손실 함수는 수직으로 분할된 얼굴의 임베딩 벡터들이 가까이 있도록 네트워크를 훈련시키며, 대칭 쌍의 차이를 처벌하여 얼굴 인식의 신뢰성을 높입니다.

- **Technical Details**: 제안된 SymFace loss는 3-Point Symmetric Split (3PSS) 알고리즘을 기반으로 하며, 이 알고리즘은 얼굴 이미지의 대칭성을 평가하는 데 사용됩니다. 분할된 얼굴의 두 절반의 임베딩 벡터가 임베딩 공간에서 가까워지도록 유도합니다. 이 방법은 얼굴의 대칭적 특성을 활용하여, 얼굴 인식의 클래스 간 분산(inter-class variance)을 증가시키고, 기존 손실 함수에 추가되어 성능을 개선합니다.

- **Performance Highlights**: 본 연구의 SymFace loss는 LFW, CFP-FP, CP-LFW, AgeDB, CA-LFW 데이터셋에서 성능을 평가하였으며, 기존의 얼굴 인식 벤치마크 데이터셋을 초월하는 SoTA 결과를 달성했습니다. 이를 통해 얼굴 인식 알고리즘의 전반적인 성능을 높일 수 있는 잠재력을 입증하였습니다.



### EventAug: Multifaceted Spatio-Temporal Data Augmentation Methods for Event-based Learning (https://arxiv.org/abs/2409.11813)
- **What's New**: 본 논문에서는 EventAug라는 체계적인 데이터 증강 기법을 소개하고, 다중 스케일 시간 통합(MSTI), 공간-유의 미분(mask) (SSEM), 그리고 시간-유의 미분(TSEM)의 세 가지 새로운 방법을 제안해 이벤트 카메라 데이터의 시공간 다양성을 증대시키고자 합니다.

- **Technical Details**: 이벤트 카메라는 저지연(low latency)과 높은 동적 범위(high dynamic range)로 인해 다양한 분야에서 성공적으로 사용되어 왔습니다. 그러나 데이터 부족과 제한된 다양성으로 인해 오버피팅(overfitting)과 불충분한 특징 학습이 발생합니다. 본 연구에서는 이벤트 데이터의 희소성(sparsity)와 불균등한 시공간 분포(spatio-temporal distribution)를 고려하여 효율적으로 학습 데이터를 다양화하는 전용 증강 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 증강 방법이 다양한 작업(task)과 네트워크 구조(backbones)에서 일관되게 성능을 향상시켜, DVS128 제스처에서 4.87%의 정확도 향상을 달성하는 등의 성과를 보였습니다.



### Latent fingerprint enhancement for accurate minutiae detection (https://arxiv.org/abs/2409.11802)
- **What's New**: 이 논문에서는 지문 인식의 주요 과제로 여겨지는 부분적이고 흐려진 지문(latent fingerprints)의 식별 문제를 해결하기 위해 생성적 적대 신경망(Generative Adversarial Networks, GANs)을 활용한 새로운 접근 방식을 제안합니다. 이 연구의 주요 목표는 기존의 지문 인식 기법에서 미세한 세부 사항을 개선하여 포렌식 조사에서 신뢰할 수 있는 식별을 보장하는 것입니다.

- **Technical Details**: 본 연구는 미세한 정보(minutiae information)를 직접 최적화하는 방식을 통해 지문 생성을 구조적으로 접근하여 향상된 latent fingerprints를 생성합니다. 이 과정에서 경량의 구조와 방향 필드를 통합하여 지문 이미지의 걸러진 구조를 보존하며, 다양한 지문 센서와 호환되는 범용 지문 표현을 개발하여 적용 가능한 모든 분야에서 통일성과 상호 운영성을 보장합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 공개 데이터셋을 대상으로 광범위한 평가를 실시하여 기존의 최신 기술들과 비교했을 때 높은 성능을 입증하였습니다. 특히 NIST SD27 데이터셋에 대한 rank-1 식별률이 65%에 불과했던 것과 비교해, 제안된 방법의 성능은 이보다 현저하게 향상된 것으로 나타났습니다.



### Efficient Low-Resolution Face Recognition via Bridge Distillation (https://arxiv.org/abs/2409.11786)
Comments:
          This paper is published in IEEE TIP 2020

- **What's New**: 이 논문은 고해상도 얼굴 모델을 저해상도 얼굴 인식을 위해 경량화하는 브릿지 디스틸레이션(bridge distillation) 방법을 제안합니다. 이를 통해 고해상도 얼굴에서의 지식을 저해상도 얼굴에 효과적으로 전이할 수 있습니다.

- **Technical Details**: 브릿지 디스틸레이션 접근법은 두 단계의 지식 전이를 통해 진행됩니다. 첫 번째 단계에서는 고해상도 데이터셋 간의 교차 전이를 수행하여 저해상도 얼굴에 필요한 특성을 생성합니다. 두 번째 단계에서는 다중 작업 학습(multi-task learning)을 통해 전이된 지식을 저해상도 얼굴에 맞게 조정합니다.

- **Performance Highlights**: 실험 결과, 제안된 학생 모델은 0.21M의 파라미터 수를 가지고 저해상도 얼굴 인식에서 뛰어난 성능을 나타냈으며, GPU, CPU, 모바일에서 각각 14,705, ~934, 763 얼굴/초의 속도를 달성했습니다.



### Distilling Channels for Efficient Deep Tracking (https://arxiv.org/abs/2409.11785)
Comments:
          Published by IEEE TIP 2020

- **What's New**: 이 논문은 딥 트래커의 효율성을 개선하기 위해 새로운 프레임워크인 '채널 증류(channel distillation)'을 제안합니다. 이 방법은 잡음이 있는 채널의 영향을 완화하고, 유용한 채널을 선택하여 특정 이동 물체에 대한 트래킹 성능을 향상시킵니다.

- **Technical Details**: 채널 증류는 디스크리미네이티브 상관 필터(DCF)와 ECO를 예시로 하여, 피처 압축(feature compression), 응답 맵 생성(response map generation), 모델 업데이트(model update)를 통합하여 에너지 최소화 문제로 формulieren합니다. 이는 정보 전달이 효율적으로 이루어지도록 합니다. 이 프레임워크의 통합된 공식화 방식이 트래킹 효과성을 개선합니다.

- **Performance Highlights**: 실험적으로 다양한 벤치마크에서 테스트된 결과, 채널 증류 방식이 정확도, 속도 및 메모리 저장 요구 사항을 개선함을 입증했습니다. 결과적으로, 제안된 딥 트래커는 정확하고, 빠르며, 메모리 요구 사항이 낮습니다.



### Knowledge Adaptation Network for Few-Shot Class-Incremental Learning (https://arxiv.org/abs/2409.11770)
Comments:
          13 pages;6 figures

- **What's New**: 본 논문에서는 Few-Shot Class-Incremental Learning (FSCIL) 문제를 해결하기 위해 'Knowledge Adaptation Network (KANet)'라는 새로운 방법을 제안합니다. 이 방법은 CLIP 모델을 활용하여 각 클래스에 대한 일반적인 표현을 제공하고, Knowledge Adapter (KA) 모듈을 통해 데이터 특화 지식을 통합함으로써 성능을 향상시킵니다.

- **Technical Details**: KANet은 먼저 CLIP 모델을 기반으로 하여, 적은 수의 샘플을 이용해 새로운 클래스를 점진적으로 인식하는 기능을 수행합니다. KA 모듈은 학습 데이터로부터의 데이터 특화 지식을 요약하여 일반적 표현에 융합시키고, Incremental Pseudo Episode Learning (IPEL) 메커니즘을 도입하여 이전 클래스의 지식을 새로운 클래스에 맞게 조정합니다. KA는 QKF(질문 기반 지식 융합) 메커니즘을 적용하여 데이터의 특화된 지식을 표기합니다.

- **Performance Highlights**: KANet은 CIFAR100, CUB200, ImageNet-R 등 다양한 데이터셋에서 경쟁력 있는 성능을 달성했습니다. 즉, 제안된 방법은 랜덤한 조건에서도 이전 클래스에 대한 성능을 유지하며 새로운 클래스를 효과적으로 인정하는 데 성공했습니다.



### Neural Encoding for Image Recall: Human-Like Memory (https://arxiv.org/abs/2409.11750)
Comments:
          5 pages, 7 figures

- **What's New**: 이번 논문은 인공지능 시스템이 인간과 유사한 메모리 회상(memory recall)을 달성하기 위한 방법론을 제안합니다. 기존의 방법이 아닌 인간의 기억 과정을 토대로 한 새로운 접근법을 통해 인공 시스템과 생물학적 시스템 간의 격차를 줄이기 위해 노력했습니다.

- **Technical Details**: 우리는 이미지의 고수준 정보를 암호화하는 과정에 중점을 두고, 이미지를 인코딩하기 전에 노이즈를 추가하여 인간 기억의 비결정론적(non-deterministic) 특성을 모방합니다. 다양한 아키텍처를 활용하여 이미지가 어떻게 인코딩되고 기억 회상에 미치는 영향을 탐구했습니다.

- **Performance Highlights**: 자연 이미지에서 97%의 정확도를 달성했으며, 텍스처에서는 거의 무작위 성능(52%)을 보였습니다. CLIP 모델에 Gaussian noise를 적용했을 때, 자연 이미지에 대한 정확도가 98%에 달하지만 텍스처에서는 무작위에 가까운 성능을 보였습니다.



### RockTrack: A 3D Robust Multi-Camera-Ken Multi-Object Tracking Framework (https://arxiv.org/abs/2409.11749)
Comments:
          RockTrack establishes a new state-of-the-art with 59.1% AMOTA on the nuScenes vision-only test leaderboard with ResNet50-level backbone

- **What's New**: 이번 연구에서 제안하는 RockTrack는 다중 카메라 감지기를 위한 3D Multi-Object Tracking (MOT) 방법으로, 기존의 단일 센서 기반 트래커를 뛰어넘는 유연성과 견고함을 제공합니다. 또한, 새로운 다중 뷰 외관 유사성 지표(multi-view appearance similarity metric)와 신뢰도 기반 전처리 모듈(confidence-guided preprocessing module)을 도입하여 신뢰할 수 있는 관측값의 회수를 극대화합니다.

- **Technical Details**: RockTrack는 Tracking-By-Detection (TBD) 프레임워크에 기반하여 설계되었으며, 다중 카메라 감지기를 지원하도록 최적화되었습니다. 이 시스템은 신뢰성을 높이기 위해 특정 필터를 사용하여 2D 및 3D/BEV 공간에서 관련 관측값을 선택적으로 회수합니다. 데이터 연관(data association) 모듈은 기하학적 특성과 외관 정보를 조합하여 신뢰할 수 있는 트래킹을 구현합니다.

- **Performance Highlights**: RockTrack는 nuScenes 비전 전용 트래킹 리더보드에서 59.1%의 AMOTA를 기록하며, 뛰어난 연산 효율성을 발휘합니다. 이 시스템은 단일 CPU만으로도 경쟁력 있는 실행 성능을 유지하여 다중 카메라 3D MOT의 새로운 기준을 설정합니다.



### Exploring Gaze Pattern in Autistic Children: Clustering, Visualization, and Prediction (https://arxiv.org/abs/2409.11744)
- **What's New**: 이 연구는 자폐 스펙트럼 장애(ASD) 아동의 주시 행동을 자동으로 분석하기 위해 최적화된 7가지 클러스터링 알고리즘을 적용한 새로운 방법을 제안합니다. 이 방법은 ASD 아동의 주의 패턴을 सू적하여 진단에 활용할 수 있는 63가지 유의미한 특징을 추출합니다.

- **Technical Details**: 연구에서는 K-Means, K-Medoids, Agglomerative Clustering, BIRCH, DBSCAN, OPTICS, GMM과 같은 다양한 클러스터링 알고리즘을 사용하여 주시 데이터를 군집화한 후, 각 군집의 질을 평가하기 위해 9가지 유효성 지수를 계산합니다. 이를 통해 ASD 아동과 정상 아동 간의 주시 패턴 차이를 분석합니다. 이러한 특징을 통해 예측 모델을 훈련하여 ASD를 진단합니다.

- **Performance Highlights**: 이 방법은 ASD 진단을 위한 자동 생성된 주시 포인트 특성에 대해 81% AUC를 기록하며 최신 성과를 달성하였습니다. 실험 결과 클러스터링 알고리즘이 ASD 아동의 독특한 주시 패턴 분석에서 개선됨을 보여줍니다.



### InverseMeetInsert: Robust Real Image Editing via Geometric Accumulation Inversion in Guided Diffusion Models (https://arxiv.org/abs/2409.11734)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문에서는 Geometry-Inverse-Meet-Pixel-Insert (GEO)라는 새로운 이미지 편집 기법을 소개합니다. GEO는 사용자 요구에 맞춘 유연한 이미지 편집 기능을 제공하며, 텍스트 프롬프트와 이미지 프롬프트를 결합해 다양한 편집 결과를 생성합니다. 특히 이 방법은 사전 학습이 필요하지 않으며, 새로운 기하학적 누적 손실(geometric accumulation loss)과 향상된 이미지 프롬프트 기술을 통해 편집 결과의 정확성과 세밀함을 보장합니다.

- **Technical Details**: GEO 기법은 Denoising Diffusion Implicit Model (DDIM)의 역전환 과정을 개선하는 새로운 기하학적 누적 손실을 포함하고 있습니다. 이 손실은 이미지의 기하학적 특성을 보존하기 위해 입력-인코딩된 이미지 잠재(latent)를 기준점으로 활용합니다. 편집 과정에서는 초기 픽셀 수준 편집 후 기하학적 누적 손실을 이용한 역전환이 진행되며, 이는 기존 DDIM 방법보다 세부 사항의 보존을 향상시킵니다.

- **Performance Highlights**: GEO는 다양한 이미지 유형 및 복잡한 프롬프트 편집 시나리오에서 고충실도 편집 결과를 제공합니다. 사용자는 원하는 길이의 텍스트 프롬프트를 입력하여 세밀하고 다중 영역 편집을 수행할 수 있으며, 배경 세부 사항도 효과적으로 보존합니다. 따라서 이 기법은 시각적 세부 조정에서 특히 뛰어난 성능을 보여줍니다.



### Free-VSC: Free Semantics from Visual Foundation Models for Unsupervised Video Semantic Compression (https://arxiv.org/abs/2409.11718)
Comments:
          ECCV2024

- **What's New**: 이 논문은 비지도 비디오 시맨틱 압축(Unsupvised Video Semantic Compression, UVSC)에 대한 새로운 접근 방식을 제안합니다. 기존 방법들이 가진 한계를 극복하기 위해, 다양한 비주얼 파운데이션 모델(Visual Foundation Models, VFM)에서 파생된 풍부한 시맨틱 정보를 흡수하는 방식을 사용합니다.

- **Technical Details**: 제안된 프레임워크인 Free-VSC는 VFMs의 시맨틱 정렬 레이어인 Prom-SAL(promo-based semantic alignment layer)을 도입하여 압축된 비디오와 VFMs 간의 시맨틱 정렬을 유연하게 수행합니다. 또한, 동적 궤적 기반의 인터 프레임 압축 방식을 통해 전체 비트 비용을 줄이고 압축 효율성을 향상시킵니다.

- **Performance Highlights**: 이 연구는 3개의 주요 작업과 6개의 데이터셋에서 이전 코딩 방법들보다 뛰어난 성능을 보여주어, 시맨틱 압축 효율성에서 두드러진 개선을 입증합니다.



### RopeBEV: A Multi-Camera Roadside Perception Network in Bird's-Eye-View (https://arxiv.org/abs/2409.11706)
- **What's New**: 이 논문에서는 다중 카메라 Bird's-Eye-View (BEV) 인식 기술을 도로 측 시나리오에 적용하기 위한 첫 번째 방법인 RopeBEV를 소개합니다. 이 방법은 기존의 차량 측 인식과 차별화된 다양한 도전과제를 체계적으로 분석하고 해결점을 제시합니다.

- **Technical Details**: RopeBEV는 4가지 주요 개선점을 포함하고 있습니다: (1) BEV Augmentation, (2) CamMask, (3) ROIMask 및 (4) 카메라 회전 임베딩. 이러한 개선은 다양한 카메라 포즈와 카메라 수에 대한 불확실성, 인식 지역의 희소성 및 방향 각도의 모호성을 해결하는 데 도움을 줍니다. RopeBEV는 BEVFormer 아키텍처를 기반으로 하는 구조로, 여러 카메라에서 얻은 2D 이미지를 3D 특징으로 변환하여 밀집형 BEV 특징 맵을 생성합니다.

- **Performance Highlights**: RopeBEV는 실제 도로 고속 데이터셋인 RoScenes에서 1위를 차지하였으며, 50개 이상의 교차로와 600대의 카메라를 포함하는 대규모 도시 데이터셋에서도 우수한 성능을 입증했습니다.



### ORB-SfMLearner: ORB-Guided Self-supervised Visual Odometry with Selective Online Adaptation (https://arxiv.org/abs/2409.11692)
- **What's New**: 제안된 방법인 ORB-SfMLearner는 ORB(Oriented FAST and Rotated BRIEF) 기능을 활용하여 학습 기반의 ego-motion 추정을 통한 깊은 시각 오도메트리 문제에 접근하였다. 또한 크로스 어텐션 메커니즘을 도입하여 PoseNet의 설명 가능성을 높였으며, 자동차의 주행 방향을 어텐션 가중치를 통해 설명할 수 있음을 밝혔다.

- **Technical Details**: ORB-SfMLearner는 ORB 기능을 활용해 시각 오도메트리를 수행하며, 선택적 온라인 적응(selective online adaptation) 메커니즘을 통해 다양한 도메인 간 최적 파라미터로 빠르게 조정할 수 있다. 이 시스템은 두 개의 연속된 프레임을 입력받고, 각각 독립된 인코더를 통해 RGB 데이터와 ORB 기능을 처리하여 6D 카메라 포즈를 예측한다.

- **Performance Highlights**: KITTI 및 vKITTI 데이터셋에서 실험 결과, 제안된 방법이 기존 최첨단 딥 비주얼 오도메트리 방법들에 비해 ego-motion 정확도와 일반화(generalizability) 면에서 뛰어난 성능을 보였다.



### GUNet: A Graph Convolutional Network United Diffusion Model for Stable and Diversity Pose Generation (https://arxiv.org/abs/2409.11689)
- **What's New**: PoseDiffusion은 자연어를 기반으로 2D 인간 포즈 스켈레톤을 생성하는 데 있어서 첫 번째로 제안된 생성적 프레임워크입니다. 이 모델은 GANs (Generative Adversarial Networks) 대신에 디퓨전 모델(Diffusion Model)을 사용하여 더 높은 다양성과 구조적 정확성을 구현합니다.

- **Technical Details**: PoseDiffusion의 주 모델인 GUNet는 그래픽 컨볼루션 신경망(Graphical Convolutional Neural Network)을 포함하여, 포즈 스켈레톤의 공간적 관계를 학습합니다. 각 키 포인트를 개별적으로 예측할 수 있도록 2D 인간 포즈 스켈레톤을 별도의 특징 채널로 표현하며, 교차 주의(cross-attention) 기법을 통해 텍스트 조건을 도입합니다.

- **Performance Highlights**: PoseDiffusion은 기존의 최첨단(SOTA) 알고리즘보다 텍스트 기반 포즈 스켈레톤 생성에서 안정성과 다양성이 뛰어납니다. 정성적 분석은 Stable Diffusion에서 제어 가능한 생성에 대한 우수성을 입증합니다.



### Detecting Underdiagnosed Medical Conditions with Deep Learning-Based Opportunistic CT Imaging (https://arxiv.org/abs/2409.11686)
- **What's New**: 이번 연구는 복부 컴퓨터 단층 촬영(CT)을 통한 기회를 활용한 진단(Opportunistic CT)의 효과를 조사하였으며, 이를 통해 저진단 상태인 사코펜리아(sarcopenia), 간지방증(hepatic steatosis), 복수(ascites)를 발견할 수 있는 가능성을 제시합니다.

- **Technical Details**: 2,674개의 병원 입원 환자의 CT 스캔을 분석하여, 기회를 활용한 CT 이미지에서 파생된 이미징 표현형(imaging phenotypes)과 방사선 보고서 및 ICD 코드 간의 불일치를 식별하였습니다.

- **Performance Highlights**: 사코펜리아는 0.5%, 간지방증은 3.2%, 복수는 30.7%의 비율로 ICD 코드화되었으며, 연구 결과는 기회를 활용한 CT가 진단의 정확성을 높이고 위험 조정 모델의 정확성을 향상시킬 수 있는 잠재력을 지니고 있음을 보여줍니다.



### SRIF: Semantic Shape Registration Empowered by Diffusion-based Image Morphing and Flow Estimation (https://arxiv.org/abs/2409.11682)
- **What's New**: 본 논문에서는 diffusion 기반의 이미지 형태 변경(image morphing) 및 흐름(flow) 추정에 기초한 새로운 Semantic shape Registration 프레임워크인 SRIF를 제안합니다. SRIF는 서로 외부 정렬(extrinsically aligned)된 형태 쌍에서 이미지를 렌더링하고, 다중 시점에서 생성된 중간 이미지를 생성하여, 중간 포인트 클라우드를 구성합니다.

- **Technical Details**: SRIF 프레임워크는 다중 시점에서 이미지 렌더링을 시작으로, diffusion 모델에 기반한 이미지 보간(image interpolation) 프레임워크를 통해 중간 이미지를 생성합니다. 이후, 동적 3D Gaussian splatting reconstruction 프레임워크를 이용해 이들 이미지를 기반으로 중간 포인트 클라우드를 재구성합니다. 마지막으로, 이러한 포인트 클라우드를 통해 타겟(target) 형태로 변형(deform)할 흐름(flow)을 추정하는 등록 모듈을 제안합니다.

- **Performance Highlights**: SRIF는 SHREC ’07 데이터셋과 EBCM 데이터셋에 있는 다양한 형태 쌍에 대해 평가되었으며, 모든 테스트 세트에서 기존 방법들과 비교해 뛰어난 성능을 보였습니다. SRIF는 고품질의 밀집 대응(dense correspondences)을 생성하며, 매끄럽고 의미 있는 형태 변경 과정을 제공합니다.



### Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks (https://arxiv.org/abs/2409.11681)
Comments:
          Preprint, Under review for ICRA 2025

- **What's New**: 이번 연구는 2D 세분화 모델을 3D Gaussian splatting(3DGS)으로 확장하기 위한 새로운 투표 기반 방법을 소개합니다. 이 방법은 마스크된 그래디언트를 활용하여 더욱 정확한 세분화를 가능하게 합니다.

- **Technical Details**: 연구 결과, 2D 세분화 마스크에 의해 생성된 그래디언트를 사용하여 3D 세분화의 정확도를 향상시키고, 훈련된 가우시안을 가지치기(pruning)하는 데 사용할 수 있다는 점이 밝혀졌습니다. 또한, 몇 개의 샷을 통해 2D 이미지의 주석을 3D Gaussian splatting에 효과적으로 전이할 수 있는 방법이 제안되었습니다.

- **Performance Highlights**: 이 접근 방식은 고성능을 자랑하며, 향후 증강 현실(AR), 물체 편집, 로보틱스 등의 다양한 다운스트림 애플리케이션에 효과적으로 활용될 가능성이 높습니다. 이 연구를 통해 최대 21%의 압축 효과도 발견되었습니다.



### Agent Aggregator with Mask Denoise Mechanism for Histopathology Whole Slide Image Analysis (https://arxiv.org/abs/2409.11664)
- **What's New**: 이 논문은 AMD-MIL이라는 새로운 방법론을 제안하여 병리학적 이미지 분석에서의 성능을 향상시키는 것을 목표로 합니다. 특히, agent aggregator와 mask denoise 메커니즘을 도입하여 WSI(classification of Whole Slide Images)의 분류 정확성을 높입니다.

- **Technical Details**: AMD-MIL은 query와 key 간의 instance 중요도를 계산하기 위한 intermediate variable 역할을 하는 agent token을 사용합니다. masking 및 denoising 매트릭스는 동적으로 low-contribution representation을 필터링하고 noise를 제거합니다. 이 방법은 self-attention의 연산 복잡성을 줄이면서도 각 instance 간의 정보를 잘 포착할 수 있도록 해줍니다.

- **Performance Highlights**: AMD-MIL은 CAMELYON-16, CAMELYON-17, TCGA-KIDNEY, TCGA-LUNG datasets에서 extensive한 실험을 통해 기존의 최신 기법들에 비해 우수한 성능을 보였습니다.



### Bridging Domain Gap for Flight-Ready Spaceborne Vision (https://arxiv.org/abs/2409.11661)
Comments:
          Submitted to Journal of Spacecraft and Rockets; Appeared as Chapter 4 of Tae Ha Park's PhD thesis

- **What's New**: SPNv3는 비협조적 우주선의 단안 위치 추정을 위한 신경망으로 설계되었으며, 오프라인 훈련과 검증 중 관찰하지 않은 우주 이미지에 대한 강건성을 제공하면서도 계산적으로 효율적입니다. 이러한 특성은 우주 등급 엣지 장치에서 NNs(Neural Networks)의 배치를 위해 필수적입니다.

- **Technical Details**: SPNv3는 Vision Transformer(ViT) 아키텍처를 기반으로 하며, 데이터 증강(data augmentation), 전이 학습(transfer learning) 및 향상된 입력 이미지 해상도를 활용하여 OOD(Out-Of-Distribution) 강건성을 극대화합니다.

- **Performance Highlights**: SPNv3는 SPEED+ 데이터셋의 HIL(하드웨어-인-더-루프) 이미지에서 최첨단 위치 정확도를 달성하며, 단독으로 합성 이미지에 대한 훈련만을 진행했습니다. SPEC2021 대회에서도 특출난 성능을 보여 주었습니다.



### VL-Reader: Vision and Language Reconstructor is an Effective Scene Text Recognizer (https://arxiv.org/abs/2409.11656)
Comments:
          Accepted by ACM-MM2024

- **What's New**: VL-Reader는 비전(vision)과 언어(language) 간의 상호작용을 기반으로 하는 새로운 장면 텍스트 인식(scene text recognition) 접근법으로, 마스크(Masked) 비주얼-언어 재구성(Masked Visual-Linguistic Reconstruction, MVLR) 목표를 도입합니다. 이 모델은 시각적 정보와 언어적 정보를 동시에 학습하고, 재구성을 통해 이러한 정보를 효과적으로 결합합니다.

- **Technical Details**: VL-Reader에는 'Masked Visual-Linguistic Decoder' (MVLD)가 설계되어 비주얼-언어 컨텍스트를 활용하고 양-modal 특성 상호작용(bi-modal feature interaction)을 촉진합니다. 이 아키텍처는 사전 훈련(pre-training) 단계에서 마스크된 시각적 및 텍스트 토큰을 재구성하고, 미세 조정(fine-tuning) 단계에서는 모든 문자(character)를 재구성합니다.

- **Performance Highlights**: VL-Reader는 6개의 일반적인 데이터셋에서 평균 97.1%의 정확도를 달성하여 SOTA(state-of-the-art)를 1.1% 초과했습니다. 특히 더 도전적인 데이터셋에서 성능 향상이 두드러지며, 전반적으로 모델의 강력한 효과성을 입증했습니다.



### Relax DARTS: Relaxing the Constraints of Differentiable Architecture Search for Eye Movement Recognition (https://arxiv.org/abs/2409.11652)
Comments:
          Accepted By CCBR 2024

- **What's New**: 이번 논문에서는 Eye movement biometrics(안구 움직임 생체 인식) 분야에 자동화된 네트워크 검색(Neural Architecture Search, NAS) 알고리즘을 도입하고, 특히 완화된 DARTS(Relax DARTS)라는 향상된 알고리즘을 제안하였습니다.

- **Technical Details**: Relax DARTS는 DARTS의 한계를 극복하고, 눈 움직임 인식 작업을 위해 네트워크 구조를 효율적으로 검색 및 훈련할 수 있도록 설계되었습니다. 이 알고리즘은 각 셀의 아키텍처 파라미터(α)를 독립적으로 훈련하여 목표 아키텍처를 더 정확하게 실현하게 합니다. 또한, 모듈 입력 가중치(β)를 도입하여 입력 선택의 유연성을 제공하고, 과적합(overfitting) 문제를 완화하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: Relax DARTS는 네 개의 공개 데이터베이스에서 실험을 통해 기존 연구들보다 우수한 인식 성능을 달성하였으며, 특히 다양한 다중 특징(tempural classification) 분류 작업에 대한 적응성을 강조하였습니다.



### DAF-Net: A Dual-Branch Feature Decomposition Fusion Network with Domain Adaptive for Infrared and Visible Image Fusion (https://arxiv.org/abs/2409.11642)
Comments:
          5pages,4figures

- **What's New**: 본 논문에서는 적외선 및 가시 이미지 융합을 위한 새로운 네트워크인 DAF-Net을 제안합니다. 이 네트워크는 Multi-Kernel Maximum Mean Discrepancy (MK-MMD)를 기반 인코더에 도입하여 적외선과 가시 이미지의 잠재 특성을 효과적으로 정렬합니다.

- **Technical Details**: DAF-Net은 Restormer 네트워크에 기반한 기본 인코더와 인버터블 신경망(Invertible Neural Networks, INN)을 사용한 세부 인코더로 구성됩니다. 이를 통해 글로벌 구조 정보를 캡처하고 세부 텍스처 정보를 추출하여 두 모달리티 간의 융합 과정에서 특징 손실을 최소화합니다. MK-MMD는 기본 인코더에서만 적용되어 글로벌 특성 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, DAF-Net은 다양한 데이터셋에서 기존 기술보다 우수한 성능을 보이며 시각적 품질과 융합 성능 모두에서 큰 향상을 이루어냈습니다.



### PainDiffusion: Can robot express pain? (https://arxiv.org/abs/2409.11635)
Comments:
          Under reviewing

- **What's New**: PainDiffusion 모델은 통증 자극에 반응하여 얼굴 표정을 생성하도록 설계되었습니다. 이는 로봇 아바타의 통증 표현력을 조절하고 감정을 반영할 수 있게 해줍니다.

- **Technical Details**: PainDiffusion는 조건부 시간 U-Net을 활용하며, EMOCA의 얼굴 표정 잠재 공간 내에서 작동하는 잠재적인 확산(diffusion) 모델입니다. 이 모델은 데이터의 효율적인 표현과 신속한 렌더링을 보장합니다.

- **Performance Highlights**: PainDiffusion는 자기 회귀(autoregressive) 방법보다 질적, 양적으로 더 우수한 성능을 보여주며, 통증 표현의 품질과 정확성을 평가하기 위한 새로운 메트릭스를 제안합니다.



### Multimodal Generalized Category Discovery (https://arxiv.org/abs/2409.11624)
- **What's New**: 이 논문에서는 Generalized Category Discovery (GCD) 개념을 확장하여 다양한 모달리티에서 데이터를 처리할 수 있는 Multimodal GCD 문제를 소개합니다. 이는 기존의 GCD가 주로 단일 모달리티 데이터에만 초점을 맞춘 점을 보완하며, 보다 풍부한 정보를 활용합니다.

- **Technical Details**: Multimodal GCD는 여러 모달리티(예: 이미지, 텍스트)에서 수집된 정보를 통합하여 분류하는 문제입니다. 메인 프레임워크인 MM-GCD는 대조 학습(contrastive learning) 및 증류(distillation) 기술을 사용하여 서로 다른 모달리티의 특성과 출력 공간을 정렬합니다. 이로 인해 분류 성능이 개선되고, 다양한 모달리티에서 수집된 정보를 효과적으로 통합할 수 있습니다.

- **Performance Highlights**: MM-GCD는 UPMC-Food101 및 N24News 데이터셋에서 각각 11.5%와 4.7% 성과 향상을 이루며 최신 최첨단 성능을 달성했습니다. 단일 모달리티 접근 방식보다 각각 6.8%와 3.4% 향상된 결과를 보였으며, 이는 모달리티 간의 상호 보완성이 있음을 강조합니다.



### VALO: A Versatile Anytime Framework for LiDAR-based Object Detection Deep Neural Networks (https://arxiv.org/abs/2409.11542)
- **What's New**: 본 논문은 LiDAR 객체 감지 딥 뉴럴 네트워크(DNNs)의 동적인 마감 기한 요구사항을 적응시키는 문제를 다룹니다. 기존 DNN들은 실시간 성능이 제한된 리소스 제약 환경에서 상당한 지연(latency)을 보이기 때문에, 런타임에서 정확도와 지연 사이의 균형을 동적으로 관리해야 합니다. 이를 해결하기 위해 VALO(Versatile Anytime algorithm for LiDAR Object detection)라는 새로운 데이터 중심 접근 방식을 제안하고 있습니다.

- **Technical Details**: VALO는 데드라인 인식 스케줄러(deadline-aware scheduler)를 사용하여 입력 데이터의 특정 영역을 선택적으로 처리합니다. 이 과정에서 VALO는 과거 감지 결과를 효율적으로 예측하여 입력 데이터를 부분적으로 처리할 때 발생할 수 있는 정확도 손실을 줄입니다. 또한, 탐지 헤드 내에 있는 입력 축소 기술을 활용하여 계산량을 줄이면서도 정확도를 유지합니다. 이를 위해 두 가지 최첨단 LiDAR 객체 감지 네트워크인 CenterPoint와 VoxelNext에 VALO를 구현하였습니다.

- **Performance Highlights**: VALO는 광범위한 시간 제약 조건에 대해 동적 적응력을 보이며, 이전의 최신 기술에 비해 높은 정확도를 달성하는 것을 보여주었습니다. 또한, VALO는 물체 감지 DNN이 지연과 정확도 간의 거래를 런타임에서 수행할 수 있도록 하는 새로운 데이터 스케줄링 프레임워크를 제안합니다.



### Obfuscation Based Privacy Preserving Representations are Recoverable Using Neighborhood Information (https://arxiv.org/abs/2409.11536)
- **What's New**: 본 논문은 기하학적 흐리게 하는 기술(geometry obfuscation)이 원래의 포인트 위치를 회복할 수 있는 취약점을 지적하며, 이러한 방법들이 실제로는 개인 정보 보호를 보장하지 않는다는 새로운 발견을 제공한다.

- **Technical Details**: 연구는 주변 정보(neighborhood information)를 활용하여 흐리게 된 포인트의 원래 위치를 근사하는 방법을 제안한다. 이 방법은 현재 제안된 모든 기하학적 흐리게 하는 기술에 적용 가능하며, 지역적으로 가까운 원래 포인트들과의 관계를 수학적으로 모델링한다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 흐리게 하는 기술에 대한 역공격(inversion attacks)을 효과적으로 가능하게 하며, 현재 개인 정보 보호를 주장하는 방법들이 실제로는 그러한 보호를 제공하지 않음을 보여준다.



### Mamba Fusion: Learning Actions Through Questioning (https://arxiv.org/abs/2409.11513)
- **What's New**: MambaVL은 선택적 상태 공간 모달리티 융합(selective state space modality fusion)을 활용하여 긴 거리 의존성을 효율적으로 포착하고 비전과 언어 데이터의 공동 표현(joint representations)을 학습하는 혁신적인 모델입니다.

- **Technical Details**: MambaVL은 비전(Vision)과 언어(Language) 브랜치에 대해 공유 상태 전이 행렬(shared state transition matrix)을 사용하여 두 가지 모달리티 간의 정보를 교환하고, 이러한 구조가 선택 메커니즘을 확장할 수 있도록 합니다. 이를 통해 모델은 양쪽 모달리티에서 관련 정보를 효과적으로 선택하고 융합할 수 있습니다.

- **Performance Highlights**: MambaVL은 Epic-Kitchens-100 데이터셋에서 액션 인식(action recognition) 및 액션 예측(action anticipation)에서 최첨단 성능(state-of-the-art performance)을 달성하며, 기존 방법들을 초월하는 결과를 보여주고 있습니다.



### Two Stage Segmentation of Cervical Tumors using PocketN (https://arxiv.org/abs/2409.11456)
- **What's New**: 이 논문에서는 자궁경부암 치료에서의 방사선치료 계획을 개선하기 위해 새로운 딥러닝 모델(PocketNet)을 제안합니다. 해당 모델은 MRI 영상을 통해 자궁, 질, 자궁 및 종양을 성공적으로 세분화할 수 있습니다.

- **Technical Details**: PocketNet은 T2 가중(Magnetic Resonance Imaging) 자기 공명 영상에서 자궁경부, 질, 자궁 및 종양을 분할하는 데 사용되는 딥러닝 모델입니다. 5-폴드 교차 검증(5-fold cross validation)을 통해 데이터에 대해 학습하였으며, 종양 분할의 평균 Dice-Sorensen 유사도 계수(DSC)는 70%를 초과하고, 장기 분할의 경우 80%를 초과했습니다.

- **Performance Highlights**: PocketNet은 대비 프로토콜의 변동성에 강한 내성을 보여주며, ROI(Region of Interest)의 신뢰할 수 있는 세분화를 제공합니다. 이는 방사선치료 계획의 품질과 일관성을 높이는 데 기여할 수 있습니다.



### Continual Learning of Conjugated Visual Representations through Higher-order Motion Flows (https://arxiv.org/abs/2409.11441)
Comments:
          Currently under review

- **What's New**: 이 논문에서는 비지도 연속 학습(unmonitored continual learning) 환경에서 픽셀 단위(feature) 특성을 motion-induced constraints의 영향을 받아 학습하는 방법을 제안합니다. 특히, 기존 접근 방법들과는 달리, 운동(motion)은 주어진 신호가 아니라 점진적이고 자율적인 학습 과정의 결과로 간주합니다.

- **Technical Details**: 모델은 여러 수준(feature hierarchy)의 운동 흐름(motion flows)을 추정하고, 이를 전통적인 optical flow부터 higher-order motions에 이르는 다양한 수준의 추상화(abstraction)로 분류합니다. 또한, self-supervised contrastive loss를 도입하여 흐름(flow) 기반 유사성(similarity)에 따라 공간적으로 인식하는 방식을 사용하여 단순한 해법을 방지합니다.

- **Performance Highlights**: 모델은 사진 현실적인 합성 스트림(synthetic streams)과 실제 비디오(real-world videos)에서 평가되었으며, 사전 학습된 최신(feature extractors) 모델 및 최근 비지도 학습 모델에 비해 우수한 성능을 보여주었습니다.



### DynaMo: In-Domain Dynamics Pretraining for Visuo-Motor Contro (https://arxiv.org/abs/2409.12192)
- **What's New**: DynaMo는 비주얼 데이터를 기반으로 한 새로운 자기 지도 학습 방법으로, 프리트레인(pretrained)된 데이터 없이도 자기 스스로 비주얼 표현을 학습할 수 있는 접근 방식을 제안합니다.

- **Technical Details**: DynaMo는 전문가의 시연 데이터를 사용하여 이미지 임베딩(sequence of image embeddings)에서 역 동역학 모델(inverse dynamics model)과 순방향 동역학 모델(forward dynamics model)을 공동 학습하여 다음 프레임을 예측합니다.

- **Performance Highlights**: DynaMo를 사용한 경우, 이전의 프리트레인된 및 자기 지도 학습 방법보다 다운스트림 모방 학습 성능이 39% 향상되었으며, 다양한 정책 클래스에서 효과적으로 작동합니다.



### Bundle Adjustment in the Eager Mod (https://arxiv.org/abs/2409.12190)
- **What's New**: 이 연구에서는 PyTorch와 원활하게 통합된 새로운 eager-mode 번들 조정(Bundle Adjustment, BA) 프레임워크를 소개합니다. 이는 기존의 C++ 기반 BA 프레임워크가 지닌 현대 딥러닝 라이브러리와의 통합 부족 문제를 해결하는 데 목표를 두고 있습니다.

- **Technical Details**: 제안된 BA 프레임워크는 PyPose를 기반으로 하며, GPU 가속화, 차분 가능성, 희소 스프레드시트 최적화 및 Lie 그룹 및 Lie 대수 작업을 지원합니다. 이 프레임워크는 2차 최적화와 희소 문제를 위한 AutoDiff 및 선형 대수 연산을 포함하고 있어, 고도의 유연성과 효율성을 제공합니다.

- **Performance Highlights**: GPU에서 실행되는 eager-mode BA는 GTSAM, g2o, Ceres에 비해 각각 18.5배, 22배, 23배로 속도 향상을 달성했습니다. 이러한 성능 향상은 기존 BA의 복잡한 Factor Graph를 Python에서 간편하게 수행할 수 있게 한 덕분입니다.



### multiPI-TransBTS: A Multi-Path Learning Framework for Brain Tumor Image Segmentation Based on Multi-Physical Information (https://arxiv.org/abs/2409.12167)
- **What's New**: 이번 연구에서는 다중 물리적 정보를 통합하여 분할 정확도를 향상시키는 새로운 Transformer 기반 프레임워크인 multiPI-TransBTS를 제안합니다.

- **Technical Details**: multiPI-TransBTS 프레임워크는 인코더, Adaptive Feature Fusion (AFF) 모듈, 다중 소스 및 다중 스케일 기능 디코더로 구성되어 있습니다. 인코더는 다양한 MRI 시퀀스에서 모달리티 특정 기능을 각각 추출하기 위해 다중 브랜치 아키텍처를 통합하였고, AFF 모듈은 채널 및 요소 기반 주의를 사용하여 다수의 출처로부터 정보를 융합합니다.

- **Performance Highlights**: multiPI-TransBTS는 BraTS2019 및 BraTS2020 데이터셋에서 기존의 최첨단 방법들에 비해 뛰어난 성능을 보여주었으며, 더 나은 Dice 계수, Hausdorff 거리, Sensitivity 점수를 지속적으로 달성하였습니다.



### Autopet III challenge: Incorporating anatomical knowledge into nnUNet for lesion segmentation in PET/C (https://arxiv.org/abs/2409.12155)
Comments:
          AutoPET III challenge submission

- **What's New**: autoPET III 챌린지는 PET/CT 영상에서의 자동 종양 병변(segmentation) 세분화 방법을 개선하는 데 중점을 두고 있으며, 새로운 FDG와 PSMA 트레이서 데이터셋을 도입하여 다양한 임상 환경에서의 적용 가능성을 확대했습니다.

- **Technical Details**: PET(Positron Emission Tomography)와 CT(Computed Tomography) 이미지를 활용한 병변 세분화는 종양의 특성 파악과 진단 정밀도를 향상시키기 위해 매우 중요합니다. 최신 연구에서는 nnUNet(Neural Network U-Net) 프레임워크와 ResNet18 모델을 사용하여 FDG와 PSMA 트레이서를 구분하고, 생리적 및 병리적 흡수 패턴을 학습하기 위한 해부학적 지식을 통합했습니다. MIP(Maximum Intensity Projection) 기법을 활용하여 PET 이미지를 간소화하고 분석 시간을 단축했습니다.

- **Performance Highlights**: 최종 제출된 모델은 FDG 데이터셋에서 76.90%의 Dice score, PSMA 데이터셋에서 61.33%의 Dice score를 기록하며 높은 정확도를 보였습니다. 이는 다양한 데이터셋에서의 자동 세분화 성능을 개선했음을 나타냅니다.



### Optimal Visual Search with Highly Heuristic Decision Rules (https://arxiv.org/abs/2409.12124)
- **What's New**: 이번 연구는 인간이 짧게 제시된 화면에서 분리된 잠재적 목표 객체 위치를 검색할 때 사용하는 의사 결정 과정에 대한 통찰을 제공합니다. 일반적인 Bayesian-optimal 결정 과정과 비교했을 때, 놀랍게도 인간의 성능이 최적 성능을 약간 초과하는 결과를 보였습니다.

- **Technical Details**: 연구에서는 세 가지 요인이 인간의 시각 검색 성능이 최적을 초과하게 하는 이유를 정량적으로 설명합니다. 첫째, 간단하고 고정된 heuristic 결정 규칙이 거의 최적의 검색 성능에 도달할 수 있다는 점입니다. 둘째, 중심 위치에서의 감각적 무시(foveal neglect)는 주로 중앙 잠재 목표 위치에만 영향을 미친다는 것입니다. 마지막으로, 공간적으로 상관된 신경 소음(neural noise)이 독립적 소음에 대해 예측되는 성능을 초과하게 만들어, 보다 나은 검색 성능을 보이게 합니다.

- **Performance Highlights**: 이 연구의 결과는 인간과 다른 동물의 시각 검색 작업 및 다른 식별 작업에 대한 이해에 중요한 의미를 갖습니다. 특히, 인간의 뇌가 최적의 계산을 복제할 수 없다는 주장에도 불구하고, 특정 조건에서 인간이 최적의 성능을 초과할 수 있는 이유를 제시합니다.



### Denoising diffusion models for high-resolution microscopy image restoration (https://arxiv.org/abs/2409.12078)
- **What's New**: 본 연구에서는 저해상도 정보에 조건화하여 고해상도 이미지를 예측할 수 있는 Denoising Diffusion Probabilistic Model (DDPM)을 훈련시키는 방법을 제안한다. DDPM의 확률적 특성을 활용하여 신호 대 잡음 비율(signal-to-noise ratio)을 증가시키는 이미지를 반복적으로 생성할 수 있다.

- **Technical Details**: Denoising Diffusion Probabilistic Models (DDPM)은 노이즈가 있는 입력으로부터 이미지를 생성하는 강력한 모델이며, 이 연구에서는 다양한 형광 현미경 데이터셋의 복잡한 노이즈 특성을 모델링하여 고품질 이미지 복원에 사용된다. 다양한 유형의 현미경 이미지에서 성능을 검증하기 위해 네 가지 다양한 데이터셋을 사용할 예정이다.

- **Performance Highlights**: DDPM 모델은 네 개의 잘 다양화된 데이터셋에서 기존의 최선 모델들과 비슷하거나 더 나은 성과를 보여주었다. 본 방법은 다양한 형광 현미경 응용 프로그램에서 높은 성능을 균일하게 달성하여 더 정확하고 덜 침습적인 이미징 관행을 위한 가능성을 제시한다.



### Online Refractive Camera Model Calibration in Visual Inertial Odometry (https://arxiv.org/abs/2409.12074)
Comments:
          Accepted at the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024), 8 pages

- **What's New**: 이 논문은 일반적인 굴절 카메라 모델과 미지의 매체의 굴절률을 온라인으로 공동 추정하는 방법을 제시합니다. 이 방법은 공기 중에서의 카메라 보정만으로 다양한 굴절 매체에서 작업할 수 있게 해줍니다.

- **Technical Details**: 제안된 카메라 모델을 사용하여 단일 시각-관성 오도메트리(Visual-Inertial Odometry; VIO) 프레임워크 내에서 굴절률을 상태 변수로 온라인 추정합니다. 실험은 수조 안에서 이동하는 수중 로봇을 통해 수행되었습니다. 수집된 데이터는 다양한 조명 조건을 포함하며, 자세히 검증된 굴절률 추정 및 VIO 정확성 평가가 포함됩니다.

- **Performance Highlights**: 제안된 방법은 초기 설정의 큰 섭동에도 불구하고 물의 이상적인 굴절률로 수렴함을 보여주었으며, 매체에 대한 사전 지식이나 매체별 카메라 보정 없이 굴절 매체에서 VIO 성능을 동등하게 유지할 수 있는 능력을 강조합니다.



### PAD-FT: A Lightweight Defense for Backdoor Attacks via Data Purification and Fine-Tuning (https://arxiv.org/abs/2409.12072)
- **What's New**: 이 논문에서는 PAD-FT라는 새로운 경량의 방어 메커니즘을 제안하고 있습니다. 기존의 방어 방법들과 달리 추가적인 깨끗한 데이터셋 없이도 방어할 수 있는 방법입니다.

- **Technical Details**: PAD-FT는 먼저 간단한 데이터 정제 과정을 도입하여 오염된 훈련 데이터셋에서 가장 깨끗한 데이터를 식별하고 선택합니다. 이후 자기 정제된 깨끗한 데이터셋을 사용하여 마지막 분류 레이어만을 세밀하게 조정하여 방어력을 향상시킵니다.

- **Performance Highlights**: PAD-FT는 다양한 백도어 공격 방법과 데이터셋에서 우수한 효율성을 보여주며, 엄청난 실험 평가를 통해 그 효과가 입증되었습니다.



### Towards Global Localization using Multi-Modal Object-Instance Re-Identification (https://arxiv.org/abs/2409.12002)
Comments:
          8 pages, 5 figures, 3 tables. Submitted to ICRA 2025

- **What's New**: 본 연구는 새로운 이중 경로 오브젝트 인스턴스 재식별(transformer architecture) 아키텍처를 제안하여 RGB 및 깊이 정보의 멀티모달 통합을 통해 효율적인 재식별을 가능하게 합니다. 이 접근 방식은 다양한 조명 조건에서 복잡한 장면에서도 성능 향상을 보여줍니다.

- **Technical Details**: Dual Path Attention Transformer for Object Re-identification(DATOR)라는 구조를 제안하였으며, 이 구조는 RGB 및 깊이 영상 정보의 특징을 추출하여 최종 임베딩을 생성합니다. DATOR는 깊이 센서와 RGB 센서를 활용하여 다양한 환경에서 높은 정확도를 유지합니다.

- **Performance Highlights**: 제안된 모델 DATOR는 객체 인스턴스 재식별에서 mAP 75.18을 기록하였으며, TUM-RGBD 데이터셋에서는 83%의 성공률로 정확한 카메라 로컬라이제이션 및 포즈 식별을 수행하는 성능을 나타냈습니다.



### Tumor aware recurrent inter-patient deformable image registration of computed tomography scans with lung cancer (https://arxiv.org/abs/2409.11910)
Comments:
          Minor revision under the journal of Medical Physics

- **What's New**: 본 논문에서는 TRACER라는 새로운 딥러닝(DL) 기반의 종양 인식을 고려한 반복 등록(tumor-aware recurrent registration) 방법을 개발하였습니다. 이 방법은 다양한 환자 스캔 간의 topology를 보존하며 종양을 정확히 정렬할 수 있습니다.

- **Technical Details**: TRACER는 3D 합성곱 장기 단기 기억 네트워크(3D-CLSTM)로 구현된 인코더 계층과 밀집 변형 벡터 필드(dense deformation vector field, DVF)를 계산하는 디코더 및 공간 변형 계층으로 구성됩니다. 각종 손실 함수(예: 이미지 유사도, 변형의 부드러움)를 사용하여 네트워크를 비지도 학습 방식으로 최적화합니다.

- **Performance Highlights**: TRACER는 정상 조직을 정확히 정렬하였으며, 종양 보존 측면에서 가장 우수한 결과를 보여주었습니다. 데이터셋 I, II, III에서 종양 부피 차이는 각각 0.24%, 0.40%, 0.13%로 측정되었고, CT 강도에서 평균 제곱 오차는 0.005, 0.005, 0.004로 나타났습니다. 계획된 방사선 치료의 종양 용량 차이는 각각 0.01 Gy 및 0.013 Gy로 집계되었습니다.



### Physically-Based Photometric Bundle Adjustment in Non-Lambertian Environments (https://arxiv.org/abs/2409.11854)
Comments:
          Accepted to 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 본 논문에서는 비라면체(non-Lambertian) 환경에서의 카메라 포즈와 3D 기하학 추정을 위한 새로운 물리 기반의 Photometric Bundle Adjustment (PBA) 방법을 제안합니다. 기존 PBA 방법들이 가진 광도 불일치의 문제를 해결하기 위해 물체의 재질, 조명, 및 빛의 경로에 대한 물리 기반의 가중치를 도입하였습니다.

- **Technical Details**: 제안된 방법은 순차 이미지와 점 구름을 바탕으로 하는 재질 추정 및 조명 추정을 위한 새로운 모델을 설계하였습니다. 이 과정에서 재질과 조명 정보를 사전 정보로 활용하여 비라면체 표면으로 인한 광도 오류의 영향을 최소화하는 가중 포토메트릭 오류를 식별합니다. 또한, SpecularRooms라는 새로운 SLAM 관련 데이터셋을 구축하여 비라면체 장면에서의 조명 및 재질의 진실 값을 제공합니다.

- **Performance Highlights**: 광도 일관성이 도전적인 환경에서 제안된 방법은 기존 접근 방식들보다 정밀도에서 우수한 성능을 보였습니다. 실험 결과, 다양한 복잡한 조명 조건에서 우리의 방법이 신뢰도를 높이고, 기존의 PBA 방법들보다 더욱 견고하게 작동함을 입증하였습니다.



### NT-ViT: Neural Transcoding Vision Transformers for EEG-to-fMRI Synthesis (https://arxiv.org/abs/2409.11836)
Comments:
          ECCV24 Workshop on Synthetic Data for Computer Vision

- **What's New**: 본 논문에서는 높은 해상도의 기능적 자기공명영상(fMRI) 샘플을 동시에 수집된 뇌파(EEG) 데이터로부터 생성하는 신경 변환 비전 트랜스포머(Neural Transcoding Vision Transformer, NT-ViT)라는 생성 모델을 소개합니다. NT-ViT는 도메인 매칭(Domain Matching, DM) 하위 모듈을 통해 EEG 표현과 fMRI 볼륨을 효과적으로 정렬하여 정확성과 신뢰성을 향상시킵니다.

- **Technical Details**: NT-ViT는 EEG 신호를 fMRI 볼륨으로 변환하기 위해 Transformer 아키텍처를 기반으로 하며, DM 모듈을 통합하여 변환 능력을 향상시킵니다. 이 모델은 두 가지 주요 하위 모듈인 생성기(Generator)와 DM 모듈로 구성됩니다. 생성기는 EEG 파형을 fMRI 볼륨으로 변환하고, DM 모듈은 훈련 중에만 작동하여 생성기의 재구성 성능을 증가시킵니다.

- **Performance Highlights**: NT-ViT는 두 개의 벤치마크 데이터셋(NODDI 및 Oddball)에서 현재 최첨단 기술을 상당한 차이로 초과하며, Oddball 데이터셋에서 RMSE가 $10	imes$ 줄어들고 SSIM에서 $3.14	imes$ 향상된 성과를 달성했습니다. 이는 높은 해상도의 뇌 이미징에 있어 시간 및 재정적 제약을 줄이는 새로운 접근 방식을 제공합니다.



### RaggeDi: Diffusion-based State Estimation of Disordered Rags, Sheets, Towels and Blankets (https://arxiv.org/abs/2409.11831)
- **What's New**: 본 논문은 Cloth state estimation 문제를 RGB 이미지를 생성하는 방식으로 공식화하고, 이를 위해 Diffusion model 기반의 새로운 파이프라인(RaggeDi)을 제안합니다. 이는 기존의 방법들과 비교하여 높은 정확성과 속도로 우수한 성능을 보였으며, 시뮬레이션과 실제 환경 모두에서 실험이 수행되었습니다.

- **Technical Details**: RaggeDi 알고리즘은 미리 정의된 평면 메쉬와 변형된 메쉬 간의 점 대 점 변환(translation map)을 표현하는 RGB 이미지를 활용하여 Cloth state를 추정하는 조건부 이미지 생성 문제로 해결합니다. 여기서는 Conditional Diffusion 모델을 통해 관측 기반의 변환 맵을 예측합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최근의 두 가지 방법에 비해 정확한 추정 및 빠른 속도를 자랑하며, Zero-shot sim-to-real 전이 능력을 검증하였습니다.



### Cross-Organ and Cross-Scanner Adenocarcinoma Segmentation using Rein to Fine-tune Vision Foundation Models (https://arxiv.org/abs/2409.11752)
- **What's New**: 최근 디지털 병리학 분야에서 종양 세분화(tumor segmentation) 기술이 크게 발전했습니다. 그러나 기관, 조직 준비 방법 및 이미지 수집 과정의 차이로 인해 도메인 불일치(domain discrepancies)가 발생하고 있습니다. 이를 해결하기 위해, 본 논문에서는 MICCAI 2024 Cross-Organ and Cross-Scanner Adenocarcinoma Segmentation (COSAS2024) 과제를 위해 다양한 비전 기초 모델(Vision Foundation Models, VFMs)을 관개적으로 조정하는 방법인 Rein을 사용하였습니다.

- **Technical Details**: Rein은 인스턴스(instance)와 직접 연결된 학습 가능 토큰(learnable tokens) 집합으로 구성되어 각 레이어의 인스턴스 수준에서 기능성을 향상시킵니다. ConvNeXt와 DINOv2 모델을 조정하는 데 사용했으며, Mask2Former를 세분화(semantic segmentation) 헤드로 채택하였습니다. 이 방법은 MMsegmentation 코드베이스를 바탕으로 구현되었습니다.

- **Performance Highlights**: 데이터세트에서 두 가지 과제가 있으며, task1에서 ConvNeXt를 조정하여 preliminary test에서 0.7719, final test에서 0.7557의 점수를 달성했습니다. task2에서 DINOv2로 preliminary test에서 0.8848, final test에서 0.8192의 점수를 기록했습니다. 다양한 도메인에서 강력한 일반화(generalization) 능력을 입증하였습니다.



### Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing (https://arxiv.org/abs/2409.11738)
Comments:
          30 pages, Accepted to ECCV 2024

- **What's New**: 본 논문에서는 전통적인 조합 최적화 방식의 샘플링-재구성 방법의 한계를 극복하기 위한 새로운 적응형 샘플링-재구성 선택 프레임워크(H1.5)를 제안합니다. 이 방법은 각 입력 데이터를 위한 최적의 샘플링 마스크와 재구성 네트워크를 선택하여 기존 방법들(H1, H2)에 비해 효과적인 재구성을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 고주파 고베이즈 불확실성을 정량화하기 위해 슈퍼해상도(Super-resolution) 공간 생성 모델을 활용하여, 샘플링 마스크 선택을 최적화합니다. 또한, 서로 다른 샘플링 마스크에 대해 별도의 재구성 네트워크를 사용함으로써 Pareto 최적성을 달성합니다.

- **Performance Highlights**: 제안 방법은 얼굴 이미지 복원과 다중 코일 MRI 재구성에서 각각 SSIM 지표의 평균 0.04, 0.004의 향상을 보여주며, 여러 Fourier CS 문제에서 현저한 성능 개선을 달성했습니다.



### DETECLAP: Enhancing Audio-Visual Representation Learning with Object Information (https://arxiv.org/abs/2409.11729)
Comments:
          under review

- **What's New**: DETECLAP은 오디오-비주얼 표현 학습에서 객체 정보를 통합하여 더 세밀한 범주 인식 능력을 향상시키는 새롭고 혁신적인 방법입니다. 이 방법은 기존의 Contrastive Audio-Visual Masked AutoEncoder (CAV-MAE)에 오디오-비주얼 레이블 예측 손실을 추가하여 객체 인식을 개선합니다.

- **Technical Details**: DETECLAP은 오디오와 비디오 입력을 통해 객체 레이블을 자동으로 생성하기 위해 최첨단 언어-오디오 모델과 객체 탐지기를 활용합니다. 이를 통해 CAV-MAE의 훈련 중 오디오-비주얼 예측 손실을 도입하여 모델이 수백 개의 오디오-비주얼 객체 레이블을 예측할 수 있도록 합니다.

- **Performance Highlights**: DETECLAP은 VGGSound 및 AudioSet20K 데이터셋에서 오디오-비주얼 검색 및 분류 작업의 성능을 평가한 결과, audio-to-visual 검색에서 +1.5% 및 visual-to-audio 검색에서 +1.2%의 recall@10 개선을 달성하였고, audio-visual 분류 정확도에서 +0.6%의 향상을 보였습니다.



### LFIC-DRASC: Deep Light Field Image Compression Using Disentangled Representation and Asymmetrical Strip Convolution (https://arxiv.org/abs/2409.11711)
- **What's New**: 이번 논문에서는 LF 이미지를 효과적으로 압축하기 위해 Disentangled Representation과 Asymmetrical Strip Convolution을 활용한 LF 이미지 압축 방법(LFIC-DRASC)을 제안합니다.

- **Technical Details**: LF 이미지를 압축 문제를 Disentangled LF representation 네트워크와 이미지 인코딩-디코딩 네트워크 학습으로 구성했습니다. 두 개의 혁신적인 feature extractor를 제안하여 LF 데이터의 구조적 정보를 통합하였습니다. 또한, LF 특징의 분리 및 디커플링을 향상시키기 위한 disentangled LF representation 네트워크를 도입했습니다.

- **Performance Highlights**: 실험 결과, 제안하는 LFIC-DRASC 방법이 기존 최첨단 방법들과 비교했을 때 평균 20.5%의 비트 전송률 감소를 달성했습니다.



### Discovering Conceptual Knowledge with Analytic Ontology Templates for Articulated Objects (https://arxiv.org/abs/2409.11702)
- **What's New**: 이번 연구는 기계 지능이 새로운 형태의 조작 대상을 인식하고 상호작용할 수 있도록 기본적인 개념적 지식을 활용할 수 있는 방법을 제시합니다. 이를 통해, 복잡한 기하학적 구조와 다양한 조인트 타입을 가진 조작 대상을 이해하고 상호작용하는 문제를 해결하고자 합니다.

- **Technical Details**: 연구진은 Analytic Ontology Template (AOT)를 제안하여 기계 지능이 조작 대상에 대한 개념적 설명을 구체화할 수 있는 프레임워크를 제공합니다. AOT는 구조(structure), 파라미터(parameter), 수용 기능(affordance), 렌더러(renderer)의 네 가지 기본 요소로 구성됩니다. 특히 AOT는 변수를 조작하여 다양한 인스턴스를 생성할 수 있으며, 이는 신경망(neural networks)에서도 활용될 수 있도록 설계되었습니다.

- **Performance Highlights**: AOT 기반의 AOTNet 접근법은 실험을 통해 조작 대상을 이해하고 상호작용하는 데 기존 방법들보다 우수한 성과를 보여주었습니다. 이 방법은 실제 훈련 데이터에 의존하지 않으면서도 새로운 조작 대상 범주에 대해 효과적으로 일반화되어 높은 성공률을 기록했습니다.



### SLAM assisted 3D tracking system for laparoscopic surgery (https://arxiv.org/abs/2409.11688)
Comments:
          Demo: this https URL

- **What's New**: 본 연구는 최소 침습 수술(minimally invasive surgery)에서 내부 해부학적 구조를 정확하게 찾아내는 데 있어 Augmented Reality(AR)를 활용한 실시간 단안 3D 추적 알고리즘을 제안합니다.

- **Technical Details**: ORB-SLAM2 프레임워크를 바탕으로 기존의 3D 추적 방법을 수정하였으며, 기본 3D 형태(primitive 3D shape)를 통해 단안 SLAM의 빠른 초기화를 구현합니다. 추적을 위한 목표 기관과 배경을 분리하는 pseudo-segmentation 전략을 사용하고, 3D 형태의 기하학적 선행 정보(geometric prior)를 포즈 그래프(pose graph)에서 추가 제약조건으로 통합합니다.

- **Performance Highlights**: 이 시스템은 in-vivo 및 ex-vivo 테스트를 통해 빠른 움직임(fast motion), 시야 밖(out-of-field-of-view) 상황, 부분 가시성(partial visibility) 및 "기관-배경(organ-background)" 상대적 움직임 문제를 효과적으로 처리하며, 견고한 3D 추적 성능을 보여주었습니다.



### Enhancing Semi-Supervised Learning via Representative and Diverse Sample Selection (https://arxiv.org/abs/2409.11653)
Comments:
          Under Review

- **What's New**: 이 논문은 Semi-Supervised Learning (SSL)에서 레이블이 없는 데이터에서의 샘플 선택이 모델 성능에 미치는 영향을 강조하고, Representative and Diverse Sample Selection (RDSS) 접근법을 제안합니다.

- **Technical Details**: RDSS는 새로운 기준인 α-Maximum Mean Discrepancy (α-MMD)를 최소화하는 방식으로 레이블을 매기기 위한 대표적이고 다양한 서브셋을 추출합니다. Modified Frank-Wolfe Algorithm을 이용하여 이 최적화 문제의 근사 해결책인 Generalized Kernel Herding without Replacement (GKHR)를 사용합니다.

- **Performance Highlights**: 실험 결과, RDSS는 여러 인기 있는 SSL 프레임워크에서 일관되게 성능을 향상시키며, Active Learning (AL) 및 Semi-Supervised Active Learning (SSAL)에서 사용되는 최신 샘플 선택 방식보다 뛰어난 성능을 보였습니다.



### Few-Shot Learning Approach on Tuberculosis Classification Based on Chest X-Ray Images (https://arxiv.org/abs/2409.11644)
Comments:
          6 pages. Pre-print

- **What's New**: 이번 연구에서는 결핵(***Tuberculosis***)의 조기 발견을 위해 인공지능(***AI***)을 활용하여 흉부 X-선 이미지를 분류하는 방법을 제안합니다. 특히, 데이터 불균형 문제를 해결하기 위해 few-shot learning (***FSL***) 접근 방식을 Prototypical Network 알고리즘을 통해 구현했습니다.

- **Technical Details**: 연구에서는 TBX11K Chest X-ray 데이터세트에서 특징 추출을 위해 ResNet-18, ResNet-50, VGG16 모델을 비교했습니다. 이를 통해 각각 98.93%, 98.60%, 33.33%의 분류 정확도를 기록하였습니다. 이 연구는 데이터 불균형을 완화하는 데 있어 제안된 방법이 우수함을 보여줍니다.

- **Performance Highlights**: ResNet-18과 ResNet-50 모델의 성능이 가장 뛰어나며, VGG16의 성능은 상대적으로 낮은 결과를 보입니다. 이는 결핵 분류 응용에서 데이터 불균형 문제를 해결하는 데 있어 제안된 방법이 매우 유용하다는 것을 나타냅니다.



### Hyperspectral Image Classification Based on Faster Residual Multi-branch Spiking Neural Network (https://arxiv.org/abs/2409.11619)
Comments:
          15pages,12figures

- **What's New**: 이번 연구에서는 하이퍼스펙트럼 이미지(HSI) 분류를 위한 스파이킹 신경망(SNN)을 기반으로 한 새로운 접근 방식을 제안합니다. 이 네트워크는 누수 적분 화재(LIF) 뉴런 모델을 기반으로 하며, 스파이킹 폭 혼합 잔여(SWMR) 모듈을 사용하여 특징 추출 작업을 수행합니다.

- **Technical Details**: SNN-SWMR 네트워크는 스파이킹 혼합 합성곱(SMC) 모듈을 통해 공간-스펙트럼 특징을 효과적으로 추출합니다. 또한 간단하고 효율적인 arcsine approximate derivative (AAD) 함수를 설계하여 스파이크 발화의 비미분 가능 문제를 해결합니다. 아울러 AAD를 통해 감독된 스파이크 신경망을 직접 훈련할 수 있습니다.

- **Performance Highlights**: 실험 결과, AAD 함수는 강력한 강건성과 우수한 적합 효과를 보여줍니다. SNN-SWMR은 다른 알고리즘과 비교하여 약 84%의 시간 단계를 줄이며, 동일한 정확도에서 훈련 시간과 테스트 시간을 각각 약 63% 및 70% 감소시킬 수 있습니다.



### Self-Contrastive Forward-Forward Algorithm (https://arxiv.org/abs/2409.11593)
- **What's New**: 본 논문에서는 Self-Contrastive Forward-Forward (SCFF) 메소드를 소개하며, 이는 Self-Supervised Contrastive Learning에서 영감을 받아 기존의 Forward-Forward (FF) 알고리즘의 성능 한계를 극복하는 것을 목표로 합니다. SCFF는 다양한 데이터셋에 적용 가능한 긍정적인 예와 부정적인 예를 생성하여 FF 알고리즘의 효용성을 확장합니다.

- **Technical Details**: SCFF는 각 데이터 샘플을 상호 대비를 통해 학습하며, 입력을 자신과 연결하여 긍정적인 예를, 훈련 세트에서 임의 선택된 예와 연결하여 부정적인 예를 만듭니다. 이를 통해 FF 알고리즘을 사용한 이미지 분류 및 시퀀스 데이터 문제 해결을 위한 새로운 가능성을 열어줍니다.

- **Performance Highlights**: SCFF는 MNIST에서 MLP를 사용해 98.7%, CIFAR-10에서 CNN을 통해 80.75%, STL-10에서 77.3%의 정확도를 기록하며 기존의 알고리즘들을 능가합니다. SCFF는 또한 시퀀스 데이터에서 첫 번째 성공적인 FF 훈련을 시연하며, Free Spoken Digit Dataset (FSDD)에서 10% 이상의 정확도 향상을 보여줍니다.



### Preference Tuning with Human Feedback on Language, Speech, and Vision Tasks: A Survey (https://arxiv.org/abs/2409.11564)
Comments:
          Survey paper

- **What's New**: 본 논문은 인간의 피드백과의 통합을 통해 딥 생성 모델을 인간의 선호도에 aligned(정렬)시키는 preference tuning(선호 조정)에 관한 최신 발전을 포괄적으로 검토합니다.

- **Technical Details**: 논문은 preference tuning을 reinforcement learning(강화 학습) 프레임워크 내에서 세 가지 주요 섹션으로 구성하였습니다. 여기에는 강화 학습을 이용한 모델과 데이터셋, 그리고 다양한 정책 접근 방법에 대한 설명이 포함되어 있습니다.

- **Performance Highlights**: LLM(대형 언어 모델)에서 preference tuning은 작업 특화 기술을 개선하고 불필요한 출력(undesired outputs)을 피하는 데 기여하며, 다국어 LLM에서도 이점이 있는 것으로 나타났습니다.



### Open-Set Semantic Uncertainty Aware Metric-Semantic Graph Matching (https://arxiv.org/abs/2409.11555)
- **What's New**: 이번 연구는 해양 환경에서의 비정형 객체 탐지를 위한 개방형 객체 매핑(Open-Set Object Mapping)에서의 시각적 기초 모델(Visual Foundation Models)의 통합을 다룬다. 또한, 객체 수준의 불확실성 추적(Object-Level Uncertainty Tracking) 프레임워크에 의미적 불확실성(Semantic Uncertainty) 척도를 포함시키는 방법을 제안한다.

- **Technical Details**: 해당 연구에서는 비정형 객체 탐지 문제를 그래프 매칭(Graph Matching) 문제로 수식화하여 다중 객체의 상대적 레이아웃을 활용한다. 불확실성 메트릭을 계산하는 방식으로 Kirchof et al.의 방법을 사용하며, 이를 통해 물체 간의 기하학적 관계를 활용하고, 그래프 편집 문제(Graph Editing Problem)로 해결 가능성을 도출한다.

- **Performance Highlights**: 실험 결과는 해양 환경에서의 실시간 다중 객체 탐지 및 의미적 불확실성을 고려한 루프 클로저(loop closure) 감지가 가능한 것으로 나타났다. KITTI 데이터셋에서도 실험을 진행하여, 제안한 방법이 육상의 대규모 장면에도 적용 가능함을 입증하였다.



### Multi-Domain Data Aggregation for Axon and Myelin Segmentation in Histology Images (https://arxiv.org/abs/2409.11552)
Comments:
          12 pages, 8 figures

- **What's New**: 이번 연구는 다양한 이미징 모달리티(전송 전자현미경, 주사 전자현미경, 밝은장면 광학 현미경 등)와 종(쥐, 쥐, 인간 등)에서 수집된 데이터를 통합하여 신경 영상에서 축삭(axon)과 미엘린(myelin) 분할(segmentation)을 위한 공용 다중 도메인(segmentation) 모델을 제공합니다. 이를 통해 특정 이미징 데이터 세트에 맞춘 모델 구축에 따른 비효율성을 줄이고 공통적으로 사용할 수 있는 모델을 제시합니다.

- **Technical Details**: 이 연구에서는 CNN(합성곱 신경망) 아키텍처를 기반으로 한 다중 도메인 분할 모델을 개발했습니다. 이 모델은 다양한 데이터를 집계하여 훈련되었으며, 각 연구 그룹의 이미지 데이터를 포함하여 서로 다른 원인에 대한 능동 학습(active learning) 전략을 활용하였습니다. 또한, 이 모델은 특정 도메인에 특화된 기존 모델보다 더 나은 일반화 성능을 보여주고, 사용이 용이하며 유지 관리가 용이하다는 장점을 가지고 있습니다.

- **Performance Highlights**: 제안된 다중 도메인 모델은 단일 모달리티 전용 학습자보다 우수한 성능을 발휘하며(p=0.03077), 다양한 도메인에서 충분한 성능 향상을 보여줍니다. 또한, 연구자들이 사용하는 데 더욱 간편하며, 오픈 소스 소프트웨어에 잘 통합되어 접근성을 높입니다.



### NCT-CRC-HE: Not All Histopathological Datasets Are Equally Usefu (https://arxiv.org/abs/2409.11546)
- **What's New**: 이 논문에서는 NCT-CRC-HE-100K 대장암 데이터셋의 분석을 통해 임상 관련성이 없는 데이터 특정 편향(data-specific biases)이 결과에 미치는 영향을 조사합니다. 색상 정규화의 부적절성, 클래스 간 심각한 JPEG 아티팩트, 잘못된 이미지 동적 범위 처리로 인해 완전히 손상된 조직 샘플 등이 주요 문제로 밝혀졌습니다.

- **Technical Details**: NCT-CRC-HE 데이터셋은 100,000개의 훈련 패치와 7,180개의 테스트 패치를 포함하며, 9개 조직 클래스에 속합니다. 저자들은 이 데이터셋이 여러 혁신적인 이미지 전처리와 깊이 있는 학습 모델을 사용한 많은 연구에서 사용되었다고 강조합니다. 간단한 EfficientNet-B0 모델이 97.7%의 정확도로 기존의 모든 솔루션을 초월했음을 보여주었습니다.

- **Performance Highlights**: 가장 단순한 모델이 3개의 특징(빨강, 초록, 파랑 색상 강도)만으로도 50% 이상의 정확도를 보였고, 색상 히스토그램을 사용했을 때 82%의 정확도를 달성했습니다. 이 논문은 NCT-CRC-HE 데이터셋과 관련된 모든 결과는 이전의 모든 전용 솔루션보다 뛰어난 성과를 나타내었다고 결론짓습니다.



### Unsupervised Hybrid framework for ANomaly Detection (HAND) -- applied to Screening Mammogram (https://arxiv.org/abs/2409.11534)
- **What's New**: 본 논문에서는 HAND (Hybrid ANomaly Detection) 라는 새로운 구조를 개발하여 대규모 디지털 스크리닝 유방촬영 데이터에서 OOD(Out-of-distribution) 데이터를 탐지하는 방법을 제시합니다. 이는 CNN(Convolutional Neural Network)과 Vision Transformer의 하이브리드 아키텍처를 결합하여 OOD 데이터를 효과적으로 식별하고, 이에 대한 학습 효율성을 높이기 위해 합성 OOD 샘플과 병렬 판별기를 도입하였습니다.

- **Technical Details**:  HAND는 입력 이미지에 대해 ID(In-distribution) 샘플을 정확히 재구성하는 것을 목표로 하며, OOD 샘플은 정상과의 차이로 인해 재구성 품질이 떨어지는 것으로 가정합니다. 또한, 기하학적 증강(geometric augmentation)을 활용한 합성 ODD 생성 기법을 구현하고, 모델의 OOD 재구성을 학습하는 데 패널티를 부여하는 Gradient Reversal 기법을 적용하였습니다.

- **Performance Highlights**: HAND 모델은 내부 RSNA 유방촬영 테스트 세트 및 외부 Mayo Clinic에서 수집된 데이터 셋에서 기존 encoder 기반 및 GAN 기반 기본 모델보다 뛰어난 성능을 보여주었으며, hybrid CNN+transformer 기준선(benchmark)보다도 우수한 결과를 달성했습니다.



### Robot Manipulation in Salient Vision through Referring Image Segmentation and Geometric Constraints (https://arxiv.org/abs/2409.11518)
- **What's New**: 본 논문에서는 로봇의 인식 모듈에 컴팩트한 참조 이미지 세분화 모델을 통합하여 언어 컨텍스트와 함께 실제 환경에서 로봇 조작 활동을 수행합니다. 주요 기여로는 CLIPU²Net이라는 새로운 경량의 참조 이미지 세분화 모델을 소개하며, 이 모델은 명세된 언어 표현으로부터 섬세한 구조 및 경계 세분화를 제공합니다.

- **Technical Details**: CLIPU²Net은 세 가지 구성 요소로 이루어져 있습니다: CLIP 인코더, 학습 가능한 마스크 멀티모달 융합 블록, 그리고 세분화용 U-squared 디코더 블록으로 구성된 주목 모듈입니다. 이 모델은 6.6MB의 소형 디코더 사이즈를 가지며, 시각적 정보의 지리적 제약을 모델링하여 로봇의 시각적 인식과 실행 가능한 명령을 연결합니다.

- **Performance Highlights**: 46개의 실제 로봇 조작 작업에서 실험 결과, 본 연구의 방법이 전통적인 시각 서보 방법보다 우수한 성능을 발휘했으며, 섬세한 참조 이미지 세분화에 뛰어난 성능을 자랑합니다. 본 방법은 다양한 환경에서도 로봇 제어를 지원하여 코스트의 효율성을 높이고, 계산 요구 사항을 줄입니다.



### Good Grasps Only: A data engine for self-supervised fine-tuning of pose estimation using grasp poses for verification (https://arxiv.org/abs/2409.11512)
Comments:
          8 pages, 7 figures, 3 tables

- **What's New**: 이 논문에서는 bin-picking을 위한 pose estimation의 self-supervised fine-tuning을 위한 새로운 방법을 제안합니다. 우리의 접근 방식은 zero-shot pose estimation을 활용하여 로봇이 수동 라벨링 없이 자동으로 훈련 데이터를 얻을 수 있게 합니다.

- **Technical Details**: 이 방법은 객체를 grasp한 후 in-hand pose estimation을 사용하여 데이터 검증을 수행하고, 시스템이 운영되고 있는 동안 fine-tuning을 가능하게 하여 학습 단계를 제거합니다. 우리의 방법은 로봇 작업 셀에서 구현되었고, 네 가지 다른 객체로 테스트되었습니다.

- **Performance Highlights**: 모든 객체에서 우리의 방법은 성능을 향상시켰고 객체의 CAD 모델로 훈련된 최신 기법을 초월하였습니다.



### Retinal Vessel Segmentation with Deep Graph and Capsule Reasoning (https://arxiv.org/abs/2409.11508)
- **What's New**: GCC-UNet은 캡슐 합성곱(capsule convolutions)과 CNN을 통합하여 망막 혈관 분할에서 로컬 및 글로벌 특성을 모두 포착하는 새로운 접근 방식을 제안합니다. 본 연구는 그래프(concept)의 힘을 활용하여 혈관 연속성을 향상시키고, 기존 방법보다 우수한 성능을 입증했습니다.

- **Technical Details**: GCC-UNet의 아키텍처는 U-Net을 기반으로 하며, 지역 특징 추출(Local Feature Extractor), 글로벌 특징 추출(Global Feature Extractor), 선택적 그래프 주의 융합(Selective Graph Attention Fusion) 모듈이 통합되어 있습니다. Graph Capsule Convolution(GC-Conv) 연산자는 캡슐 네트워크 안에 그래프 추론을 통합하여 혈관 구조의 글로벌 표현을 향상시킵니다. 각 구성 요소는 아브레이션 연구(ablation studies)를 통해 그 유효성이 입증되었습니다.

- **Performance Highlights**: GCC-UNet은 공용 데이터셋을 통한 실험에서 기존 방법들에 비해 뛰어난 성능을 보였으며, 망막 혈관 분할의 새로운 기준을 설정했습니다. 본 연구는 기존의 방법들과 비교하여 혈관 연속성을 유지하기 위해 채널 및 공간 그래프 주의 메커니즘을 도입하였습니다.



### Machine Learning for Analyzing Atomic Force Microscopy (AFM) Images Generated from Polymer Blends (https://arxiv.org/abs/2409.11438)
Comments:
          39 pages, 13 figures, 4 tables

- **What's New**: 이번 논문에서는 폴리머 필름에서 얻은 원자 힘 현미경(Atomic Force Microscopy, AFM) 이미지에서 도메인을 식별하기 위한 새로운 머신러닝(ML) 워크플로우를 소개합니다. 이 워크플로우의 목표는 두 가지 유형의 폴리머 도메인의 공간적 위치를 수동介入(Intervention) 없이 식별하고, 도메인 크기 분포를 계산하는 것입니다.

- **Technical Details**: 본 연구에서는 컴퓨터 비전(Computer Vision)과 신호 처리(Signal Processing)에서 사용되는 기존 접근 방식을 간략히 검토하고, AFM 이미지 데이터셋에 대해 이러한 접근 방식을 테스트합니다. 도메인 분할(Domain Segmentation) 작업에서는 이산 푸리에 변환(Discrete Fourier Transform, DFT) 또는 이산 코사인 변환(Discrete Cosine Transform, DCT)과 분산 통계(Variance Statistics)를 특징으로 하는 워크플로우가 가장 효과적임을 발견했습니다. 반면, 컴퓨터 비전 분야의 인기 있는 ResNet50 딥러닝(Deep Learning) 접근 방식은 상대적으로 저조한 성능을 보였습니다.

- **Performance Highlights**: 144개의 입력 AFM 이미지 각각에 대해 기존의 porespy 파이썬 패키지를 사용하여 DFT 기반 워크플로우의 출력에서 도메인 크기 분포를 계산했습니다. 이 논문에서 공유하는 정보와 오픈 소스 코드(Open Source Codes)는 크리스탈리노(Crystalline) 또는 비정질(Amorphous) 도메인, 도메인 간의 뚜렷한 또는 거친 경계, 미세 또는 거대 분리(Micro or Macrophase Separation) 도메인을 가진 폴리머 샘플의 AFM 이미지를 자동 분석하기 위한 ML 모델링 및 워크플로우에 필요한 연구자들에게 유용한 가이드가 될 수 있습니다.



### CUNSB-RFIE: Context-aware Unpaired Neural Schr\"odinger Bridge in Retinal Fundus Image Enhancemen (https://arxiv.org/abs/2409.10966)
- **What's New**: 이 연구에서는 신규의 Context-aware Unpaired Neural Schrödinger Bridge (CUNSB-RFIE) 프레임워크를 소개하여 저품질 망막 이미지를 고품질 이미지로 변환하는 새로운 딥러닝 기반 접근 방식을 제안합니다. 이 접근법은 이전의 GAN 기반 방법보다 훈련의 안정성과 이미지 품질을 개선합니다.

- **Technical Details**: CUNSB-RFIE는 Optimal Transport (OT) 이론을 활용하여 두 개의 임의 분포 간의 확률적 변환을 모델링하는 Schrödinger Bridge (SB) 프레임워크를 통해 저품질 망막 이미지를 개선하는 이미지-투-이미지 (I2I) 변환 파이프라인을 구성합니다. 또한, Dynamic Snake Convolution (DSC)를 도입하여 혈관과 같은 미세구조 세부정보를 보존하는 데 중점을 두고 PatchNCE 및 SSIM 정규화를 적용하여 개선된 결과를 보여줍니다.

- **Performance Highlights**: 대규모 데이터셋에 대한 실험 결과, 제안된 방법이 여러 최신 감독 및 비감독 방법에 비해 이미지 품질과 다운스트림 작업의 성능 모두에서 우수한 결과를 보였습니다. 이 연구는 망막 영상 향상을 위한 SB 접근 방식을 최초로 사용한 것이며, 특히 병리학적 세부 사항을 보다 효과적으로 캡처하는 데 성공했습니다.



New uploads on arXiv(cs.AI)

### LifeGPT: Topology-Agnostic Generative Pretrained Transformer Model for Cellular Automata (https://arxiv.org/abs/2409.12182)
- **What's New**: 이 논문에서는 토로이드 그리드(toroidal grid)에서의 Game of Life 시뮬레이션을 위한 decoder-only generative pretrained transformer 모델인 LifeGPT를 개발했습니다. 이는 시스템의 기본 토폴로지에 대한 사전 지식 없이도 작동합니다.

- **Technical Details**: LifeGPT는 특정 크기나 주기적 경계 조건에 대한 선험적 지식 없이 Game of Life의 결정론적 규칙을 거의 완벽하게 캡처할 수 있는 GPT 모델입니다. 또한, 'autoregressive autoregressor' 개념을 도입하여 LifeGPT를 사용한 재귀적 구현을 가능하게 합니다.

- **Performance Highlights**: LifeGPT는 다양한 훈련 데이터를 기반으로 하여 Game of Life의 복잡한 동작을 매우 높은 정확도로 예측할 수 있습니다. 이 결과는 대규모 언어 모델(LLM)의 보편적 계산 가능성에 대한 기초를 제공합니다.



### Abductive explanations of classifiers under constraints: Complexity and properties (https://arxiv.org/abs/2409.12154)
Comments:
          Full version with proofs of Martin C. Cooper and Leila Amgoud, Abductive explanations of classifiers under constraints: Complexity and properties, ECAI 2023, 469-476

- **What's New**: 이 논문에서는 Abductive explanations (AXp)를 Refinements 하여, feature 간의 제약을 고려한 세 가지 새로운 설명 유형을 제안합니다.

- **Technical Details**: 새로운 AXp는 integrity constraints (IC) 및 dependency constraints (DC)를 포함하여 생성됩니다. 설명의 핵심 개념인 coverage를 활용하여, 제약을 고려한 설명의 독립성을 보장합니다. 본 연구에서는 각 설명 유형의 복잡성을 분석하고, 샘플 기반 설명을 검토하여 계산 비용을 줄이는 방법도 모색하였습니다.

- **Performance Highlights**: Invalid instances를 배제함으로써, 중복되거나 불필요한 AXp를 제거할 수 있으며, 설명 기능들이 다양한 복잡도와 보장 기능을 가지고 있다는 점을 분석합니다.



### Optimizing Job Shop Scheduling in the Furniture Industry: A Reinforcement Learning Approach Considering Machine Setup, Batch Variability, and Intralogistics (https://arxiv.org/abs/2409.11820)
Comments:
          18 pages, 8 pages

- **What's New**: 이번 논문은 가구 산업에서 Deep Reinforcement Learning (DRL)의 응용 가능성을 탐구하고 있습니다. 특히 Job Shop Scheduling Problem (JSSP)의 복잡성을 훨씬 잘 표현하는 모델을 제안하며, 생산 계획을 위한 DRL 적용을 강조하고 있습니다.

- **Technical Details**: 제안된 모델은 기존 JSSP 접근 방식을 확장하여 작업량, 버퍼 관리, 운송 시간, 기계 세팅 시간을 포함시킵니다. 이를 통해 생산 흐름과 프로세스의 예측 및 분석을 보다 정밀하게 수행할 수 있으며, RL 에이전트는 세부적인 관찰에 기반하여 최적의 스케줄링 결정을 내립니다.

- **Performance Highlights**: RL 에이전트는 에피소드 계획 및 지속적 계획 구현 전략을 통해 생산 스케줄의 실시간 조정을 가능하게 하여, 가구 산업에서의 스케줄링 정확성과 효율성을 향상시킵니다. 이를 통해 생산 마감 기한을 충족시키는 효율적인 스케줄링을 달성할 수 있습니다.



### Explaining Non-monotonic Normative Reasoning using Argumentation Theory with Deontic Logic (https://arxiv.org/abs/2409.11780)
Comments:
          13 pages

- **What's New**: 이번 연구에서는 디자이너가 법적 관련 디자인 결정을 내릴 수 있도록 효과적인 설명을 제공하는 방법을 탐구했습니다. 특히, 이전 연구에서 개발한 LeSAC 시스템을 확장하여 법적 또는 윤리적 원칙을 위한 규범을 명시했습니다.

- **Technical Details**: 이 논문은 제1차 법적 명세를 도입하여 디온틱 논리 (deontic logic)를 활용합니다. 이 시스템은 의무 (obligation), 허가 (permission) 및 우선순위 기반의 추론을 통해 법적 결정의 정당성을 지원합니다.

- **Performance Highlights**: 자율주행 (autonomous driving) 사례 연구를 통해 LeSAC가 복잡한 디자인 결정에 대한 일관된 법적 설명을 제공할 수 있음을 입증했습니다. LeSAC는 규칙 기반의 논증 프레임워크에 대한 합리성 공리를 만족시킬 수 있어, 최종적으로 설계자에게 더 명확한 이유를 제시합니다.



### Synthesizing Evolving Symbolic Representations for Autonomous Systems (https://arxiv.org/abs/2409.11756)
- **What's New**: 이 연구는 인공지능 시스템이 개방형 학습 능력을 향상하기 위한 새로운 아키텍처를 제안합니다. 이 시스템은 사전 정의된 목표가 없이도 환경을 탐험하고 경험을 반영하여 동적 계획을 수립할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 이 시스템은 Intrinsic Motivation (IM)을 활용하여 에이전트가 환경을 자율적으로 탐색할 수 있도록 합니다. 에이전트는 자신의 경험으로부터 PPDDL (Planning Domain Definition Language) 표현을 생성 및 업데이트하며, 이를 통해 저수준(low-level) 기술과 고수준(high-level) 추상화를 통합합니다.

- **Performance Highlights**: 이 접근법은 에이전트가 저수준의 데이터를 활용하여 점진적으로 복잡한 목표를 달성하는 계획을 수립할 수 있게 합니다. 에이전트는 반복적으로 옵션을 발견하고 환경을 탐색하며, 수집한 지식을 추상화하고 계획을 수립함으로써 높은 수준의 자율성을 보장합니다.



### Towards Explainable Goal Recognition Using Weight of Evidence (WoE): A Human-Centered Approach (https://arxiv.org/abs/2409.11675)
- **What's New**: 이 논문은 Goal Recognition (GR) 과정에서 인간이 이해할 수 있는 방식으로 설명을 제공하는 새로운 접근 방식을 소개합니다. XGR 모델을 통해 인간의 기대에 부합하는 설명을 생성합니다.

- **Technical Details**: 본 연구는 eXplainable Goal Recognition (XGR) 모델을 개발하여 설명 가능성이 강조된 GR 시스템을 구축합니다. 이 모델은 Weight of Evidence (WoE) 개념을 사용하여 관찰된 설명의 선택 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, XGR 모델은 사용자 이해도, 신뢰도 및 의사 결정 지원 측면에서 기존 모델들을 능가하였으며, 특히 Sokoban 게임 도메인에서 73% 이상의 유사성으로 인간과 유사한 설명을 생성했습니다.



### Anticipating Oblivious Opponents in Stochastic Games (https://arxiv.org/abs/2409.11671)
- **What's New**: 이 논문에서는 여러 에이전트가 존재하고 불확실한 환경에서 동시 확률적 게임 (concurrent stochastic games)에서의 정책을 예측하는 새로운 접근 방식을 제시합니다. 특히, 행동을 결정할 수 없는 환경에서 에이전트가 자신의 보상을 최대화하는 방안을 탐구합니다.

- **Technical Details**: 주요 기여는 환경의 행동에 대한 정보를 제공하는 유한한 정보 상태 기계 (finite information state machine)의 합성입니다. 각각의 상태는 환경이 사용하는 정책에 대한 신념 상태 (belief state)와 매핑됩니다. 제안된 시스템은 고정된 거리 내에서 정확한 신념 상태를 유지하는 일관성 (consistency)의 개념을 도입하고, 이를 확인하는 방법과 성공적으로 종료될 경우 자동 생성될 수 있는 접근 방식을 제공합니다.

- **Performance Highlights**: 실험 평가 결과, 카타락트 수술 및 가구 조립과 같은 작업을 포함하는 기준 사례를 통해, 제안된 접근 방식이 환경의 정책과 행동을 성공적으로 예측하며 보상을 극대화하는 데 기여함을 보여줍니다.



### A Metric Hybrid Planning Approach to Solving Pandemic Planning Problems with Simple SIR Models (https://arxiv.org/abs/2409.11631)
- **What's New**: 본 연구에서는 전염병 전파 문제를 해결하기 위해 SIR 모델의 솔루션 방정식을 기반으로 하는 상태 전이 모델을 구축하였습니다. 이 모델을 통해 효율적인 봉쇄 조치 및 전염병 완화 전략을 수립할 수 있는 방식으로 문제를 공식화하였습니다.

- **Technical Details**: 연구는 SIR 모델에 기초한 상태 전이 모델을 통해 metric hybrid planning 문제를 형식화하고, 자동 계획 도구를 활용하여 연속 시간에서의 계획 수립을 수행합니다. 이 과정에서 유효 부등식(valid inequalities)을 도입하여 계산의 효율성을 향상시켰습니다. 이는 부분 미분 방정식을(PDEs) 기반으로 한 복잡한 시스템을 다루며, 제약 조건을 포함한 조합적 순차적 추론(combinatorial sequential reasoning)을 필요로 합니다.

- **Performance Highlights**: 자동 계획 도구의 이론적 결과와 다양한 실험 환경에서의 성공적인 적용 사례를 통해 우리의 접근 방식의 유효성을 입증하였습니다. 이는 감염자의 수를 특정 임계값 이하로 유지하면서도 의료 시스템의 과부하를 방지하는 전략을 개발하는 데 기여할 수 있습니다.



### Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agen (https://arxiv.org/abs/2409.11527)
- **What's New**: 이번 논문에서는 Multi-agent strategies와 Tree of Thoughts (ToT) 방법을 조합하여 LLM의 추론 능력을 향상시키는 새로운 접근 방식을 제시합니다. 이를 통해 Reasoner agent의 피상적인 추론 경로 탐색 문제를 해결하고, 불완전한 추론 경로를 체계적으로 검증하는 Thought Validator agent를 도입함으로써 신뢰성을 높였습니다.

- **Technical Details**: 제안된 방법은 여러 개의 Reasoner agent가 동시에 ToT를 활용하여 다양한 추론 경로를 탐색하는 구조입니다. Thought Validator agent는 다양한 Reasoner의 추론 결과를 평가하여 논리적으로 유효한 경로만 최종 결정에 반영합니다. 이 구조는 문제 공간의 탐색과 답변의 신뢰성을 동시에 높이는 데 기여합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 GSM8K 데이터셋에서 평가되었으며, 기존 기법들보다 평균 5.6% 더 높은 성능을 보였습니다. 이는 복잡한 수리적 추론 작업에서 정확성과 신뢰성을 확대함을 보여줍니다.



### Beyond Algorithmic Fairness: A Guide to Develop and Deploy Ethical AI-Enabled Decision-Support Tools (https://arxiv.org/abs/2409.11489)
- **What's New**: 이 논문은 인공지능(AI)과 최적화의 통합을 통해 엔지니어링 시스템의 효율성과 신뢰성을 향상시키는 방법을 제시합니다. 특히, AI 기반 최적화 도구의 윤리적 사용이 중요하며, 공정성 이외의 다양한 윤리적 고려사항을 다루는 필요성을 강조합니다.

- **Technical Details**: AI와 최적화의 융합은 복잡한 네트워크 시스템에서의 의사결정에 새로운 윤리적 차원을 추가합니다. 논문은 데이터 수집, 모델 개발, 결과 해석, 최적화 도구 구현의 각 단계에서 발생하는 윤리적 고려사항에 대해 논의하며, 전력이용사례와 물류사례를 통한 실증적 검토가 포함됩니다.

- **Performance Highlights**: 사례 연구를 통해 AI 기반 최적화 도구가 공정성, 투명성 및 책임을 강화하는 방법을 탐색하고, 각 단계에서 윤리적 함의를 고려하여 최적화가 이루어져야 함을 강조합니다. 궁극적으로, 이러한 윤리적 지침은 다양한 분야에서 AI 기술이 긍정적으로 작용하도록 보장하기 위해 필요하다고 결론짓습니다.



### AIvril: AI-Driven RTL Generation With Verification In-The-Loop (https://arxiv.org/abs/2409.11411)
- **What's New**: AIvril은 RTL-aware LLM의 정확도와 신뢰성을 향상시키기 위한 고급 프레임워크로, 자동 문법 수정 및 기능 검증을 위한 다중 에이전트 시스템을 도입합니다. 이를 통해 코드 생성에서 오류를 크게 줄이고, 하드웨어 설계의 자동화와 최적화를 위한 중요한 단계를 제공합니다.

- **Technical Details**: AIvril 프레임워크는 두 가지 핵심 구성 요소, 즉 AutoReview와 AutoDV(Automatic Design Verification)를 포함하여 RTL 코드의 문법적 정확성과 기능적 정확성을 보장합니다. 이 프레임워크는 전자 설계 자동화(EDA) 도구의 피드백을 활용하여 생성된 코드를 정제하고 디버깅하는 지능형 에이전트의 네트워크를 구성합니다.

- **Performance Highlights**: AIvril은 VerilogEval-Human 데이터셋에서 실험을 진행한 결과, 코드 질이 이전 작업에 비해 1.32배와 2배 향상되었으며, 검증 목표를 달성하는 평균 성공률이 88.46%입니다. 이는 보다 강력하고 준수하는 RTL 구현으로 이어집니다.



### Vista3D: Unravel the 3D Darkside of a Single Imag (https://arxiv.org/abs/2409.12193)
Comments:
          ECCV'2024

- **What's New**: 새로운 연구인 Vista3D는 단일 이미지에서 3D 객체의 숨겨진 측면을 효율적으로 드러내는 프레임워크입니다. 이는 5분 만에 다양한 3D 객체를 생성할 수 있게 합니다.

- **Technical Details**: Vista3D는 두 단계(phase)로 구성된 접근 방식을 채택하고 있습니다. 첫 번째 단계인 coarse phase에서는 Gaussian Splatting 기술을 사용해 초기 기하학을 빠르게 생성합니다. 두 번째 단계인 fine phase에서는 학습한 Gaussian Splatting에서 직접 Signed Distance Function (SDF)을 추출하고, 이를 differentiable isosurface representation으로 최적화하여 품질을 높입니다. 또한 두 개의 독립적인 implicit 함수를 사용하여 가시적인 부분과 가려진 부분을 포착하는 disentangled representation을 사용합니다.

- **Performance Highlights**: Vista3D는 생성된 3D 객체의 일관성과 다양성을 효과적으로 유지하며, 종합적인 평가를 통해 이 균형을 성공적으로 달성하고 있음을 입증하였습니다.



### DynaMo: In-Domain Dynamics Pretraining for Visuo-Motor Contro (https://arxiv.org/abs/2409.12192)
- **What's New**: DynaMo는 비주얼 데이터를 기반으로 한 새로운 자기 지도 학습 방법으로, 프리트레인(pretrained)된 데이터 없이도 자기 스스로 비주얼 표현을 학습할 수 있는 접근 방식을 제안합니다.

- **Technical Details**: DynaMo는 전문가의 시연 데이터를 사용하여 이미지 임베딩(sequence of image embeddings)에서 역 동역학 모델(inverse dynamics model)과 순방향 동역학 모델(forward dynamics model)을 공동 학습하여 다음 프레임을 예측합니다.

- **Performance Highlights**: DynaMo를 사용한 경우, 이전의 프리트레인된 및 자기 지도 학습 방법보다 다운스트림 모방 학습 성능이 39% 향상되었으며, 다양한 정책 클래스에서 효과적으로 작동합니다.



### Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (https://arxiv.org/abs/2409.12191)
Comments:
          Code is available at this https URL. arXiv admin note: text overlap with arXiv:2408.15262 by other authors

- **What's New**: Qwen2-VL 시리즈는 이전 Qwen-VL 모델의 업그레이드 버전으로, 동적 해상도(naive dynamic resolution) 처리 메커니즘을 도입하여 다양한 해상도의 이미지를 처리하고, 시각적 토큰(visual tokens)을 동적으로 생성할 수 있도록 개선되었습니다.

- **Technical Details**: Qwen2-VL 모델은 Multimodal Rotary Position Embedding (M-RoPE)을 활용하여 텍스트, 이미지 및 비디오 간의 위치 정보를 효과적으로 융합합니다. 또한, 2B, 8B 및 72B 매개변수를 갖춘 버전으로 확장 법칙을 탐색하며, 다양한 해상도와 종횡비에서의 이해력을 향상시킵니다.

- **Performance Highlights**: Qwen2-VL-72B 모델은 DocVQA, InfoVQA, RealWorldQA와 같은 다양한 멀티모달 벤치마크에서 GPT-4o 및 Claude3.5-Sonnet과 유사한 성능을 보여주며, 고해상도 영상 이해 및 다국어 지원 기능을 제공합니다.



### Democratizing MLLMs in Healthcare: TinyLLaVA-Med for Efficient Healthcare Diagnostics in Resource-Constrained Settings (https://arxiv.org/abs/2409.12184)
- **What's New**: 이 논문은 고급 진단이 필요한 원격 의료 환경에서 자원 제한이 있는 장치에서 다중 모달 대규모 언어 모델(MLLM)인 TinyLLaVA를 의료 데이터셋에 맞춰 최적화한 TinyLLaVA-Med를 소개합니다.

- **Technical Details**: TinyLLaVA-Med는 18.9W에서 작동하고 11.9GB의 메모리를 사용하며, VQA-RAD에서 64.54%, SLAKE에서 70.70%의 정확도를 달성합니다. 모델은 instruction-tuning과 fine-tuning을 통해 의료 데이터셋에 적합하게 조정되었습니다.

- **Performance Highlights**: TinyLLaVA-Med는 제한된 컴퓨팅 자원으로도 뛰어난 정확성을 유지하며, 기본적인 기능을 유지하면서 최신 모델과 유사한 정확도에 도달할 수 있는 가능성을 보여줍니다.



### To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning (https://arxiv.org/abs/2409.12183)
- **What's New**: 본 논문에서는 Chain-of-thought (CoT) 기법이 대형 언어 모델(LLMs)에서의 추론 능력을 이끌어내는 데 어떤 작업에서 효과적인지 분석하기 위해 100편 이상의 문헌에 대한 메타 분석을 수행하였으며, 14개 모델과 20개 데이터셋에서 자체 평가를 진행했습니다.

- **Technical Details**: 논문 결과에 따르면, CoT는 주로 수학적 또는 논리적 추론과 관련된 작업에서 성능 향상을 가져오며, 기타 작업에서는 성과가 미미합니다. CoT는 두 가지 처리 단계인 계획 단계와 실행 단계로 나누어 분석하였고, CoT는 주로 실행 단계에서 효용이 있음을 발견했습니다. 그러나 CoT는 도구를 사용한 LLM보다 성능이 떨어지는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 CoT의 유용성이 도구 보완에 의해 한정되어 있음을 보여주며, CoT가 매우 복잡한 문제에는 필요한 경우가 있지만, 전반적으로 더 효율적인 프롬프트 전략이 존재함을 강조합니다. 아울러 CoT를 넘어서는 새로운 패러다임을 탐색할 필요성을 제기하였습니다.



### Computational Dynamical Systems (https://arxiv.org/abs/2409.12179)
Comments:
          46+14 pages, 6 figures; accepted to FOCS 2024

- **What's New**: 본 논문은 매끄럽고 유한 차원인 동역학 시스템의 계산 복잡성 이론을 연구하며, 이런 시스템이 튜링 머신(Turing machine)을 어떻게 시뮬레이션(simulate)하는지를 정의합니다. 특히 '혼돈적(chaotic)' 동역학 시스템과 '적분 가능(integrable)' 동역학 시스템이 보편 튜링 머신(universal Turing machines)을 강건하게 시뮬레이션할 수 없다는 점을 강조합니다.

- **Technical Details**: 저자는 매끄러운 동역학 시스템이 튜링 머신을 시뮬레이션 할 수 있는 조건을 제시하고, 이를 기반으로 여러 수학적 결과를 도출합니다. 특히, 구조적으로 안정적인 1차원 동역학 시스템으로 인코딩(encoding)될 수 있는 튜링 머신은 결정 가능한 멈춤 문제(decidable halting problem)를 가지고 있으며, 특정 시간 복잡도(time complexity)의 한계가 있음을 보여줍니다.

- **Performance Highlights**: 저자에 의하면, 계산 동역학 시스템(computational dynamical systems) 이론을 통해 복잡성 이론과 동역학 시스템 이론, 그리고 실대수기하학(real algebraic geometry) 간의 교차점에서 다양한 흥미로운 질문들이 발생할 수 있음을 강조합니다. 또한 이 논문은 다양한 동역학 모델이 고전적인 튜링 머신 개념과 어떻게 연결될 수 있는지를 탐구함으로써 해당 분야의 학문적 기여를 하고 있습니다.



### Expanding Expressivity in Transformer Models with M\"obiusAttention (https://arxiv.org/abs/2409.12175)
- **What's New**: 이번 연구에서는 Transformer 모델의 attention 메커니즘에 Möbius 변환(Möbius transformations)을 통합하는 혁신적인 접근법인 MöbiusAttention을 제안합니다. 이를 통해 모델은 복잡한 기하학적 관계를 학습할 수 있으며, 토큰 간의 의존성을 보다 효과적으로 포착할 수 있습니다.

- **Technical Details**: MöbiusAttention은 섭동 기하학적 공간에서 비선형(non-linear) 연산을 활용하여, 전통적인 linear methods보다 더 복잡한 inter-token 의존성을 포착할 수 있습니다. 우리는 MöbiusAttention으로 개선된 BERT와 RoFormer 모델을 구축하고, C4 데이터셋에서 사전 학습(pre-trained)한 후 GLUE 벤치마크에서 미세 조정(fine-tuning) 하였습니다.

- **Performance Highlights**: MöbiusAttention을 적용한 모델은 여러 NLP 작업에서 baseline 모델들(BERT, RoFormer)보다 더 좋은 성능을 보였으며, 적은 수의 매개변수(parameter)로도 개선된 표현력을 나타냈습니다. 이는 MöbiusAttention의 잠재력과 성능 향상에 대한 새로운 연구 방향을 열어줍니다.



### Semantic Interoperability on Blockchain by Generating Smart Contracts Based on Knowledge Graphs (https://arxiv.org/abs/2409.12171)
- **What's New**: 이번 연구에서는 헬스케어 분야에서의 결정을 지원하기 위해 여러 기관의 장기 데이터를 기반으로 하는 Health 3.0의 개념과 블록체인 스마트 계약을 활용한 신뢰할 수 있는 의사 결정을 제안하고 있습니다.

- **Technical Details**: 제안된 방법에서는 HL7 FHIR과 같은 표준을 사용하여 데이터의 의미적 상호운용성을 확보하며, 이를 위해 스마트 계약은 블록체인 언어(예: Solidity)로 개발됩니다. 또한, 고급 의미론적 지식 그래프(Semantic Knowledge Graph)를 사용하여 스마트 계약의 논리를 인코딩합니다. 최종적으로, 오프체인(off-chain)에서 구성된 코드 생성 파이프라인을 통해 구체적인 스마트 계약으로 변환되어 온체인(on-chain)에 배포됩니다.

- **Performance Highlights**: 우리는 Medicare의 3가지 건강 보험 사례에 대해 코드 생성 방법을 적용하여, 생성된 계약들이 블록체인 상에서 정확성과 실행 비용('gas') 측면에서 우수한 성능을 보임을 확인하였습니다.



### The Unreliability of Acoustic Systems in Alzheimer's Speech Datasets with Heterogeneous Recording Conditions (https://arxiv.org/abs/2409.12170)
Comments:
          5 pages, 1 figure, 1 table

- **What's New**: 이 논문은 알츠하이머병(AD)의 초기 징후를 감지하기 위한 자동화된 음성 분석 방법의 유효성을 조사합니다. 기존 데이터셋의 녹음 조건의 이질성이 문제점을 드러내며, 이로 인해 음향 특징의 신뢰성이 의심받고 있습니다.

- **Technical Details**: ADreSSo 데이터셋과 스페인어 구어체 데이터셋을 통해 MFCCs와 Wav2vec 2.0 임베딩을 이용한 두 가지 음향 특징 기반 시스템을 연구했습니다. 이 시스템은 음성을 포함한 신호뿐만 아니라 비음성 부분에서도 높은 분류 성능을 보였으며, 이는 녹음 조건이 클래스 예측에 영향을 미칠 수 있음을 시사합니다.

- **Performance Highlights**: 비음성 부분만 이용했을 때에도 AD 환자와 대조군을 분류할 수 있는 능력이 있는 것으로 나타났습니다. 이는 표준화되지 않은 녹음을 바탕으로 한 자동 판별 시스템 사용에 대한 경고로, 데이터 수집 시 엄격한 음향 조건의 통제가 필요함을 강조합니다.



### NSSR-DIL: Null-Shot Image Super-Resolution Using Deep Identity Learning (https://arxiv.org/abs/2409.12165)
- **What's New**: 본 연구는 이미지 데이터에 의존하지 않는 혁신적이고 계산적으로 효율적인 이미지 슈퍼 해상도( ISR ) 알고리즘을 제안합니다. 이 알고리즘은 슈퍼 해상도 이미지(SR)를 생성하는 대신, 열화 공간을 포괄하는 커널의 역을 계산하는 작업으로 ISR을 재정의합니다. 이 방법은 Deep Identity Learning(DIL)을 도입하여 열악한 모델과 그 역 모델 간의 정체 관계를 활용합니다.

- **Technical Details**: 제안된 NSSR-DIL 모델은 기본적으로 10배 이상의 계산 효율성을 보여주며, X2, X3, X4와 같은 다양한 스케일 팩터에 대해 모델 재훈련이 필요 없습니다. 또한, 기존의 자가 감독 방식의 ZSSR와는 달리, LR 이미지 데이터가 없이도 ISR 작업을 학습할 수 있게 합니다. 본 연구는 경량의 CNN인 Linear-CNN(L-CNN)을 사용하여 열화 모델을 훈련합니다.

- **Performance Highlights**: 제안된 NSSR-DIL 모델은 벤치마크 ISR 데이터 세트에서 경쟁력 있는 성능을 시연하며, 실제 응용 프로그램에 더 적합한 매우 효율적인 ISR 모델로 자리매김하고 있습니다.



### Decoding Style: Efficient Fine-Tuning of LLMs for Image-Guided Outfit Recommendation with Preferenc (https://arxiv.org/abs/2409.12150)
Comments:
          CIKM 2024

- **What's New**: 이 논문은 개인화된 의상 추천의 어려움을 해결하기 위해 대형 언어 모델(LLMs)을 활용하는 새로운 프레임워크를 제안합니다. 기존 LLM의 '블랙 박스'(black box) 및 정적 특성을 극복하기 위해 세밀하게 조정하고 직접적인 피드백을 통합합니다.

- **Technical Details**: 이 프레임워크는 다중 모달 대형 언어 모델(MLLM)을 활용하여 아이템 설명의 시각적-텍스트적 갭을 연결합니다. 또한, LLM을 오픈 소스 폴리보어(Polyvore) 데이터세트에서 효율적으로 미세 조정하여 세련된 의상 추천 능력을 최적화합니다.

- **Performance Highlights**: 평가 결과, 제안된 프레임워크는 기본 LLM보다 상당히 향상된 성능을 나타내며, 트렌드에 맞춘 스타일리시한 의상 제안을 지속적으로 생성합니다. 이 결과는 개인화된 추천 경험을 향상시킬 잠재력을 보여줍니다.



### Takin: A Cohort of Superior Quality Zero-shot Speech Generation Models (https://arxiv.org/abs/2409.12139)
- **What's New**: Takin AudioLLM은 오디오북 제작을 위한 혁신적인 기술 및 모델 집합으로, Takin TTS, Takin VC, Takin Morphing을 포함합니다. 이 모델들은 개인 맞춤형 스피치 생산을 가능하게 하며 고품질의 음성을 생성할 수 있습니다.

- **Technical Details**: Takin TTS는 향상된 신경 코덱 언어 모델로, 낮은 대역폭의 고충실도 음성 코덱을 기반으로 하여 고품질 자연 음성을 생성하는 능력을 가지고 있습니다. Takin VC는 효과적인 내용 및 음색 결합 모델링 접근 방식을 통해 화자 유사성을 개선합니다. 마지막으로 Takin Morphing 시스템은 고급 음색 및 프라소디 모델링 접근 방식을 통해 사용자가 원하는 음색 및 프라소디를 조절할 수 있도록 지원합니다.

- **Performance Highlights**: Takin AudioLLM 시리즈 모델들은 다양한 응용 프로그램에서 고품질의 자연스러운 음성을 생성할 수 있는 능력을 입증했으며, 사용자 경험을 크게 향상시키고 음성 변환 기술의 발전에 기여할 잠재력을 보여줍니다.



### GRIN: GRadient-INformed MoE (https://arxiv.org/abs/2409.12136)
Comments:
          58 pages

- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 모델의 훈련을 개선하기 위해 새로운 방법론인 GRIN (GRadient-INformed MoE training)을 도입하였습니다. 이는 expert routing을 위한 희소 그라디언트 추정과 모델 병렬성을 지원하여 토큰 드롭 문제를 해결합니다.

- **Technical Details**: GRIN 방법론은 SparseMixer-v2 아키텍처를 활용하여 Top1 및 TopK MoE의 훈련을 최적화합니다. 특히, MaskedSoftmax 함수를 사용하여 explicit expert sampling을 수행하며, 이 과정에서 소개된 추가 하이퍼파라미터 r은 샘플링 공간의 무작위성 및 희소성을 조절합니다. 이러한 기법들은 기존의 jitter noise를 대체하여 MoE의 성능을 향상시킵니다.

- **Performance Highlights**: GRIN을 활용하여 개발된 16x3.8B MoE 모델은 7B 밀집 모델보다 뛰어난 성능을 발휘하였으며, 동일한 데이터를 학습한 14B 밀집 모델과 동등한 성취를 보였습니다. MMLU에서 79.4, HellaSwag에서 83.7, HumanEval에서 74.4, MATH에서 58.9의 성과를 달성하여 MoE의 효율성 향상을 입증하였습니다.



### Almost Sure Convergence of Linear Temporal Difference Learning with Arbitrary Features (https://arxiv.org/abs/2409.12135)
Comments:
          30 pages, 0 figures

- **What's New**: 이번 연구는 선형 함수 근사(linear function approximation)를 사용하는 Temporal Difference (TD) 학습이 선형 독립성(linearly independent) 가정을 필요로 하지 않고도 거의 확실히 수렴한다는 것을 제시합니다. 이는 많은 실제 시나리오에서 선형 독립성 가정이 성립하지 않음에도 불구하고, 이러한 결과를 최초로 도출했습니다.

- **Technical Details**: 연구에서는 TD 고정점 집합으로 수렴하는 가중치 반복(weight iterates)의 경향과 지역 안정성(local stability)에 대한 개념을 확립했습니다. 분석의 핵심은 선형 TD의 평균 ODE(Ordinary Differential Equation)의 경계 불변 집합(bounded invariant sets)을 이론적으로 새롭게 정리한 것입니다.

- **Performance Highlights**: 이 연구는 선형 TD 알고리즘에 대해 추가적인 가정이나 수정 없이도 수렴성을 확립한다는 점에서 의미가 있으며, 실제로 많은 RL 문제에서 활용될 수 있는 기반을 마련했습니다.



### BERT-VBD: Vietnamese Multi-Document Summarization Framework (https://arxiv.org/abs/2409.12134)
Comments:
          10 pages

- **What's New**: 이 논문은 베트남어 다문서 요약(Multi-Document Summarization, MDS) 문제를 해결하기 위해 추출 기반(extractive) 및 생성 기반(abstractive) 요약 기법을 통합하는 새로운 프레임워크를 제안합니다. 기존의 접근 방식이 갖고 있는 한계를 극복하기 위해 두 개의 구성 요소로 이루어진 파이프라인 아키텍처를 활용하여, 성능을 향상시키는 가능성을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 구성 요소로 나뉩니다. 첫 번째는 수정된 사전 학습 BERT(Bidirectional Encoder Representations from Transformers) 네트워크를 활용하여 각 문서 내에서 핵심 문장을 찾아내는 추출적 요약입니다. 두 번째는 VBD-LLaMA2-7B-50b 모델을 사용하여 이 추출된 문장을 기반으로 생성적 요약을 수행하여 최종 요약 문서를 생성합니다. 이 과정에서 Sentence-BERT(SBERT)와 VBD-LLaMA2-7B-50b 같은 딥러닝 모델을 사용합니다.

- **Performance Highlights**: 제안된 프레임워크는 VN-MDS 데이터셋에서 ROUGE-2 점수 39.6%를 달성하며, 기존의 최첨단 모델들보다 성능이 우수함을 보여주었습니다. BET와 같은 모델을 활용한 기존 연구들과 비교할 때, 제안된 방법이 베트남어에서의 요약 품질을 현저히 향상시킨다는 것을 입증했습니다.



### Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvemen (https://arxiv.org/abs/2409.12122)
- **What's New**: 이번 논문에서는 Qwen2.5-Math 및 Qwen2.5-Math-Instruct 1.5B/7B/72B의 수학 전용 대형 언어 모델 시리즈를 소개합니다. Qwen2.5 시리즈의 핵심 혁신은 자가 개선(self-improvement) 철학을 전 과정에 통합했다는 점입니다.

- **Technical Details**: 이 모델들은 사전 학습(pre-training), 후속 학습(post-training) 및 추론(inference) 과정에서 Qwen2-Math-Instruct를 사용하여 대규모 수학 데이터 생성을 지원합니다. 후속 학습 단계에서는 Qwen2-Math-Instruct에서 대규모 샘플링을 통해 보상 모델(reward model, RM)을 개발하고, 이는 감독 세부 조정(Supervised Fine-Tuning, SFT) 과정에서 데이터의 반복적인 진화를 주도합니다. 마지막으로 Qwen2.5-Math-Instruct는 강화 학습에 최적의 RM을 사용하여 재학습하며, 영어와 중국어 모두 지원하여 고급 수학적 추론 능력을 갖추고 있습니다.

- **Performance Highlights**: Qwen2.5-Math-7B 모델은 GSM8K, MATH, GaoKao Math Cloze에서 각각 91.6, 55.4, 57.6의 점수를 기록하며, Qwen2-72B 모델을 초월했습니다. 특히 Qwen2.5-Math-72B 모델은 MATH 벤치마크에서 66.8점을 달성하여 새로운 최고 기록을 세우며, CoT 모드에서는 Qwen2.5-Math-1.5B-Instruct 모델이 현재 사용 가능한 모든 오픈 소스 모델의 성능을 초과합니다.



### Pareto Data Framework: Steps Towards Resource-Efficient Decision Making Using Minimum Viable Data (MVD) (https://arxiv.org/abs/2409.12112)
- **What's New**: 이번 논문에서는 'Pareto Data Framework'를 소개하며, 머신러닝 응용 프로그램을 위한 Minimum Viable Data (MVD) 선택의 전략을 제안합니다. 이 접근법은 제한된 플랫폼, 즉 임베디드 시스템, 모바일 장치, IoT 장치에서 발생할 수 있는 데이터 비용을 줄이면서도 성능을 유지하는 방법을 다룹니다.

- **Technical Details**: 이 프레임워크는 자원 제한 환경에서 성능 저하 없이 효율성을 최적화하기 위해 MVD를 식별합니다. 연구는 다운 샘플링, 양자화 및 절단을 통해 데이터의 축소가 성능에 미치는 영향을 실험적으로 보여줍니다. 결과적으로 샘플 속도는 75% 감소하더라도 성능은 95% 이상 유지할 수 있음을 입증했습니다.

- **Performance Highlights**: 이 연구 결과는 데이터 저장 및 전송 비용을 동시에 줄이면서 머신러닝 모델의 효율성을 향상시킬 수 있는 가능성을 보여줍니다. 또한 농업, 교통, 제조와 같은 다양한 분야에서 IoT 응용 프로그램과 고급 AI 기술의 민주화를 통해 데이터 기반 통찰력의 혜택을 multiplied할 수 있는 기회를 제공합니다.



### Measuring Human and AI Values based on Generative Psychometrics with Large Language Models (https://arxiv.org/abs/2409.12106)
- **What's New**: 이 논문에서는 Generative Psychometrics for Values (GPV)라는 LLM 기반의 데이터 구동 가치 측정 패러다임을 소개합니다. 기존의 심리 측정 도구들을 개선하여 LLM의 고급 의미 이해를 활용하여 텍스트에서 맥락적이고 가치가 있는 인식을 추출할 수 있습니다.

- **Technical Details**: GPV는 Llama 3(즉, ValueLlama)로 미세 조정되어 있으며, 텍스트를 인식으로 분해하는 LLM의 능력을 검증합니다. 이 과정은 GPV의 핵심 파이프라인을 형성하며 대량의 인간 작성 블로그에 적용하여 안정성 및 타당성을 검증합니다.

- **Performance Highlights**: GPV는 이전의 심리 도구들에 비해 더 우수한 성능을 보여주며, 17개의 LLM과 4개의 가치 이론에 대한 광범위한 평가를 통해 유효성 및 효용성 면에서 훌륭한 측정 결과를 얻었습니다.



### IMRL: Integrating Visual, Physical, Temporal, and Geometric Representations for Enhanced Food Acquisition (https://arxiv.org/abs/2409.12092)
- **What's New**: 이번 연구에서는 로봇 보조 급식에서 다양한 식품을 획득하는 데 있어 기존 방법들의 한계를 극복하기 위한 새로운 접근법인 IMRL (Integrated Multi-Dimensional Representation Learning)을 제안합니다. IMRL은 시각적, 물리적, 시간적 및 기하학적 표현을 통합하여 강력하고 일반화 가능한 정책을 학습합니다.

- **Technical Details**: IMRL은 로봇이 다양한 식품 유형과 물리적 특성을 이해하고, 식습관의 시간적 동적을 모델링하며, 최적의 스쿱 포인트를 결정하는 데 기하학적 정보를 활용합니다. 이 방법은 주어진 환경에 따라 스쿱 전략을 적응적으로 조정하여 다양한 식품을 처리하는 능력을 향상합니다.

- **Performance Highlights**: 실제 UR3 로봇 실험에서 IMRL을 이용한 접근법은 성공률에서 최상의 기초 모델에 비해 최대 35% 향상을 달성했습니다. 또한 보이지 않는 환경에서도 제로샷 일반화(zero-shot generalization)를 통한 강인한 성능을 보여주었습니다.



### Towards Interpretable End-Stage Renal Disease (ESRD) Prediction: Utilizing Administrative Claims Data with Explainable AI Techniques (https://arxiv.org/abs/2409.12087)
Comments:
          10pages, 4 figures, AMIA 2024

- **What's New**: 이 연구는 행정 청구 데이터(Administrative Claims Data)와 고급 머신러닝(Machine Learning) 및 딥러닝(Deep Learning) 기법을 조합하여 만성 신장 질환(Chronic Kidney Disease, CKD)에서 말기 신부전(End-Stage Renal Disease, ESRD)로의 진행을 예측하는 가능성을 탐구합니다.

- **Technical Details**: 연구에서는 주요 건강 보험 기관이 제공한 10년의 데이터셋을 분석하여 전통적인 머신러닝 기법인 Random Forest 및 XGBoost와 딥러닝 접근법인 Long Short-Term Memory (LSTM) 네트워크를 사용하여 여러 관찰(window) 모델을 개발하였습니다. 관찰 기간이 24개월인 LSTM 모델이 ESRD 진행 예측에서 최고의 성능을 보였으며, SHapley Additive exPlanations (SHAP) 분석을 통해 개별 환자 수치의 결정 요소를 해석할 수 있는 인사이트를 제공하였습니다.

- **Performance Highlights**: LSTM 모델은 24개월의 관찰 창(window)에서 기존 문헌의 모델들을 초월하는 성능을 보이며, CKD 관리 및 ESRD 진행 예측을 위한 행정 청구 데이터 활용의 가치를 강조합니다.



### PAD-FT: A Lightweight Defense for Backdoor Attacks via Data Purification and Fine-Tuning (https://arxiv.org/abs/2409.12072)
- **What's New**: 이 논문에서는 PAD-FT라는 새로운 경량의 방어 메커니즘을 제안하고 있습니다. 기존의 방어 방법들과 달리 추가적인 깨끗한 데이터셋 없이도 방어할 수 있는 방법입니다.

- **Technical Details**: PAD-FT는 먼저 간단한 데이터 정제 과정을 도입하여 오염된 훈련 데이터셋에서 가장 깨끗한 데이터를 식별하고 선택합니다. 이후 자기 정제된 깨끗한 데이터셋을 사용하여 마지막 분류 레이어만을 세밀하게 조정하여 방어력을 향상시킵니다.

- **Performance Highlights**: PAD-FT는 다양한 백도어 공격 방법과 데이터셋에서 우수한 효율성을 보여주며, 엄청난 실험 평가를 통해 그 효과가 입증되었습니다.



### Generalized Robot Learning Framework (https://arxiv.org/abs/2409.12061)
Comments:
          6 pages, 2 figures. cs.RO

- **What's New**: 이번 논문에서는 저비용으로 쉽게 재현이 가능하고 여러 로봇 및 환경에 이식 가능한 로봇 학습 프레임워크를 제안합니다. 이 imitation learning이 산업용 로봇에도 성공적으로 적용될 수 있음을 증명하였습니다.

- **Technical Details**: 제안된 프레임워크는 10가지의 명확히 정의된 로봇 작업을 기반으로 하여 설계되었고, 각 작업의 데이터는 4,000건 이상 수집되었습니다. 모델의 일반화(generalization)를 통해 단일 체크포인트로 여러 작업을 수행할 수 있도록 만들어졌습니다. Perception module과 action prediction module로 구성된 로봇 제어 정책이 도입되었습니다.

- **Performance Highlights**: Voting Positive Rate (VPR)이라는 새로운 평가 전략을 제안하여 실제 조작 작업에서의 성과를 보다 객관적으로 평가할 수 있게 하였습니다. 수집된 데이터는 공개되어 로봇 학습 커뮤니티에서의 협업 및 연구를 촉진할 예정입니다.



### PARAPHRASUS : A Comprehensive Benchmark for Evaluating Paraphrase Detection Models (https://arxiv.org/abs/2409.12060)
- **What's New**: 이 논문은 paraphrasus라는 새로운 벤치마크를 제안하여, 다양한 측면에서 패러프레이즈 감지 모델을 평가하고 세분화된 모델 선택을 가능하게 합니다. 이를 통해 기존의 단일 분류 데이터셋으로는 포착할 수 없는 성능의 트레이드오프를 드러냅니다.

- **Technical Details**: paraphrasus는 인간이 작성한 문장 쌍을 포함한 데이터셋을 사용하여 패러프레이즈 감지 모델을 평가합니다. 이 벤치마크에는 의미적 및 어휘적 유사성이 다양한 두 가지 새로운 데이터셋이 포함되어 있습니다. 첫 번째 데이터셋은 패러프레이즈 분류를 위한 338개의 테스트 집합을 주석 처리하였고, 두 번째는 진정한 패러프레이즈의 전문가 예시를 추출한 것입니다.

- **Performance Highlights**: 패러프러스 벤치마크를 통해 LLMs와 훈련된 모델을 다양한 설정 하에 테스트하여, 기존 패러프레이즈 데이터에 대한 평가의 한계를 명확히 하였습니다. 특히, 단일 패러프레이즈 데이터셋에서 평가하는 것이 실제 일반화 성능에 대한 오해를 초래할 수 있음을 발견했습니다.



### Dual-Layer Training and Decoding of Large Language Model with Simultaneously Thinking and Speaking (https://arxiv.org/abs/2409.12059)
Comments:
          9 pages, 5 figures

- **What's New**: 이 논문은 기존의 데이터 기반 또는 훈련 기반 접근 방식이 아닌, 자연 세계의 인지 기제를 영감을 받아 TaS라는 새로운 모델 아키텍처를 설계하였습니다. 이 모델은 먼저 생각을 고려한 후 질의에 대한 응답을 생성합니다.

- **Technical Details**: TaS 모델은 프롬프트-응답 샘플에서 생각의 내용을 주석 달거나 생성하는 여러 파이프라인을 설계하며, 중간 계층에 언어 헤드를 추가하여 사고 계층(thinking layer)으로 작동합니다. 이 모델은 생각이 증가된 데이터(thoughts-augmented data)로 언어 모델을 훈련시킵니다.

- **Performance Highlights**: TaS의 효과성과 성능을 정성적 예제(qualitative examples)와 정량적 결과(quantitative results)로 입증하였습니다. 모델의 코드도 공개되었습니다.



### A Unified Framework for Neural Computation and Learning Over Tim (https://arxiv.org/abs/2409.12038)
- **What's New**: 이 논문은 Hamiltonian Learning이라는 새로운 통합 프레임워크를 제안합니다. 이 프레임워크는 신경망을 사용하여 "시간에 따라" 학습할 수 있도록 설계되었으며, 이는 데이터를 수신하는 무한 스트림에서 온라인 방식으로 이루어집니다. 기존 연구들은 데이터의 저명한 유한 길이를 가진 스트림 또는 작은 시퀀스로 세분화된 경우에 초점을 맞추었습니다.

- **Technical Details**: Hamiltonian Learning은 최적 제어 이론(optimal control theory)의 도구를 활용하여 신경 계산 및 학습을 위한 통합 모델을 제시합니다. 이 프레임워크는 미분 방정식(differential equations)에 기반하며, 이는 외부 소프트웨어 솔버 없이도 통합될 수 있고, 피드 포워드 및 순환 네트워크에서의 그래디언트 기반 학습의 개념을 일반화합니다.

- **Performance Highlights**: 이 논문에서는 Hamiltonian Learning이 어떻게 인기 있는 그래디언트 기반 학습 방법인 BackPropagation 및 BackPropagation Through Time을 회복하는지 실험적으로 입증합니다. 또한 이 프레임워크는 병렬화(parallelization) 및 분산 컴퓨팅(distributed computation)에 대한 사용자 맞춤형 접근 방식을 제공하며, 활성화 저장 없이 메모리 효율적인 BackPropagation을 일반화합니다.



### Topological Deep Learning with State-Space Models: A Mamba Approach for Simplicial Complexes (https://arxiv.org/abs/2409.12033)
- **What's New**: 이 연구는 Simplicial Complexes에 기반한 새로운 아키텍처를 제안합니다. 이를 통해 여러 차수의 구조 간에 직접 통신이 가능하여, n-body 관계를 더 잘 모델링할 수 있게 됩니다.

- **Technical Details**: Mamba state-space model을 백본으로 사용하여 이웃 셀을 기반으로 노드의 시퀀스를 생성합니다. 이 시퀀스는 모든 차수의 구조 간 정보 전파를 단일 단계에서 가능하게 합니다. 이를 통해 정보 전파 방법을 최적화하고, 배치 처리를 통해 메모리 사용 및 훈련 시간을 효율적으로 관리합니다.

- **Performance Highlights**: 제안된 모델은 Simplicial Complexes를 위한 최신 모델들과 경쟁력 있는 성능을 달성함을 입증했습니다. 다양한 그래프 데이터셋에 대한 실험을 통해 사용의 유용성을 확인했습니다.



### Promise and Peril of Collaborative Code Generation Models: Balancing Effectiveness and Memorization (https://arxiv.org/abs/2409.12020)
Comments:
          Paper accepted to the ASE 2024 Conference Research Track

- **What's New**: 이 연구는 여러 기업에서 소속되는 데이터셋을 활용한 협업 훈련 방식의 효과를 평가한 것으로, 코드 생성 모델의 다음 토큰 예측과 메모리 패턴을 조사하여 유용성과 정확성을 분석합니다.

- **Technical Details**: 연구는 중앙 집중식(centralized), 연합(federated), 점진적(incremental) 훈련 방식을 사용하는 협업 훈련에 대한 다양한 리스크와 효과성을 탐구하며, 코드 데이터셋의 크기와 다양성이 모델 성능에 미치는 영향을 강조합니다.

- **Performance Highlights**: 연합 학습은 중앙 집중식 학습에 비해 데이터 노출과 메모리 비율이 낮으면서 경쟁력 있는 성능을 보였고, 점진적 학습에서의 성능은 데이터셋의 순서에 따라 크게 달라졌습니다. 또한, 훈련 데이터가 보이더라도 여전히 노출 위험이 존재함을 강조합니다.



### Representing Positional Information in Generative World Models for Object Manipulation (https://arxiv.org/abs/2409.12005)
- **What's New**: 본 논문은 객체 조작(task)에서 발생하는 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 세계 모델(world model)의 객체 중심(object-centric) 표현을 강화하여 조작 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: 제안한 방법에는 두 가지 접근 방식이 포함됩니다: position-conditioned (PCP)와 latent-conditioned (LCP) 정책 학습입니다. LCP는 목표 설정을 위해 객체의 위치 정보를 명시적으로 캡처하는 객체 중심 잠재 표현(object-centric latent representations)을 활용합니다. 이로 인해 공간 좌표나 시각적 목표를 통해 목표를 정의하는 멀티모달(multi-modal) 기능이 자연스럽게 나타납니다.

- **Performance Highlights**: 여러 조작 환경을 통해 평가한 결과, 제안된 방법은 기존의 모델 기반 제어 접근 방식보다 유리한 성능을 보였습니다. 특히, 목표가 공간 좌표로 표현될 때 성능이 크게 개선된 점이 두드러집니다.



### Putting Data at the Centre of Offline Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2409.12001)
- **What's New**: 이 논문은 오프라인 다중 에이전트 강화 학습(MARL) 분야에서 데이터 사용의 중요성을 강조하며, 현재의 연구 방향이 데이터의 본질을 간과하고 있다는 주장을 제기합니다.

- **Technical Details**: 논문에서는 세 가지 주요 기여를 포함합니다: (1) 새로운 데이터셋 생성을 위한 명확한 가이드라인; (2) 80개 이상의 기존 데이터셋을 표준화하여 공공 리포지토리에 호스팅; (3) 데이터셋을 더 잘 이해할 수 있는 분석 도구 제공. 이는 MARL의 데이터 인식(data awareness) 개선에 기여하고자 합니다.

- **Performance Highlights**: 레퍼런스에 따라, 기존의 MARL 연구는 데이터의 품질과 내용을 무시하고 알고리즘적인 혁신에만 집중해왔습니다. 이는 학습 성능에 부정적인 영향을 미칠 수 있으며, 이 논문은 이러한 문제점을 명확하게 보여 줍니다.



### Additive-feature-attribution methods: a review on explainable artificial intelligence for fluid dynamics and heat transfer (https://arxiv.org/abs/2409.11992)
- **What's New**: 최근 유체역학 분야에서 데이터 기반 방법의 활용이 급증하고 있으며, 이는 복잡하고 다중 스케일의 난류 흐름에 적응하고 대규모 시뮬레이션 또는 실험에서 패턴을 감지하는 능력 때문입니다.

- **Technical Details**: 본 논문에서는 SHAP (SHapley Additive exPlanations) 값을 포함한 여러 additive-feature-attribution 방법의 개요와 이를 유체역학에 적용하는 방법이 설명됩니다. 보다 구체적으로, kernel SHAP, tree SHAP, gradient SHAP 및 deep SHAP의 네 가지 일반적인 구현 방법을 제시합니다. 이러한 방법들은 deep-learning 모델이 예측을 수행할 때 사용되는 입력 피처들의 중요도를 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 이 논문은 XAI (eXplainable Artificial Intelligence)의 사용을 통해 유체역학에서 해석 가능한 물리 법칙 준수의 딥러닝 모델을 구현하는 데 additive-feature-attribution 방법들이 중요한 역할을 한다는 점을 강조합니다. 이 연구는 난류 모델링, 유체 역학의 기초 및 유체 역학과 열 전달의 응용 문제로 나누어 주요 응용 분야를 제공합니다.



### AlignBot: Aligning VLM-powered Customized Task Planning with User Reminders Through Fine-Tuning for Household Robots (https://arxiv.org/abs/2409.11905)
- **What's New**: 이 논문에서는 사용자 알림과 효과적으로 조정하여 가정용 로봇의 맞춤형 작업 계획을 최적화하기 위해 설계된 새롭고 독창적인 프레임워크인 AlignBot을 소개합니다.

- **Technical Details**: AlignBot은 GPT-4o의 어댑터로 기능하는 세밀하게 조정된 LLaVA-7B 모델을 사용하여 사용자의 다양한 알림(개인화된 선호, 수정 가이드라인 및 상황적 지원 등)을 구조화된 지침 형식의 큐로 변환하여 맞춤형 작업 계획 생성을 안내합니다. 또한 동적 검색 메커니즘을 통합하여 GPT-4o에 대한 프로세스 프롬프트로서 작업과 관련된 과거 성공 사례를 선택합니다.

- **Performance Highlights**: AlignBot은 실제 가정 환경에서 수행된 실험을 통해 맞춤형 작업 계획 품질을 향상시키며, 기존의 LLM 및 VLM 기반 플래너보다 유의미하게 개선된 성과를 나타냈습니다. 결과적으로 AlignBot은 86.8%의 성공률을 기록하여 기본 GPT-4o 모델의 21.6%와 비교하여 65% 향상된 성과를 보였으며, 효과성 또한 4배 이상 증가하였습니다.



### Finding the Subjective Truth: Collecting 2 Million Votes for Comprehensive Gen-AI Model Evaluation (https://arxiv.org/abs/2409.11904)
- **What's New**: 이번 연구는 Rapidata 기술을 활용하여 텍스트-이미지 모델의 성과를 효율적으로 평가할 수 있는 새로운 주석(annotation) 프레임워크를 제시합니다. 이를 통해 4,512개의 이미지에 대해 200만 건 이상의 주석을 수집하였으며, DALL-E 3, Flux.1, MidJourney, Stable Diffusion과 같은 4개의 주요 모델을 스타일 선호도, 일관성, 텍스트-이미지 정합성에 대해 평가했습니다.

- **Technical Details**: 주요 기준은 스타일(style), 일관성(coherence), 텍스트-이미지 정합성(text-to-image alignment)으로 나뉘어 있으며, 주석자는 각기 다른 모델에서 생성된 이미지 쌍을 제시받고, 선호하는 이미지를 선택하게 됩니다. 이 연구에는 282개의 이미지 생성 프롬프트가 포함되어 있으며, 다양한 문화적 배경을 가진 글로벌 주석자 풀을 활용하여 더 많은 사용자 피드백을 수집할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이번 연구는 기존의 텍스트-이미지 생성 벤치마크와 비교했을 때 200만 건의 투표를 바탕으로 진행된 가장 포괄적인 평가를 제공하며, 다양한 주석자 인구통계학적 특성을 반영하여 편향의 위험을 감소시켰습니다.



### DocMamba: Efficient Document Pre-training with State Space Mod (https://arxiv.org/abs/2409.11887)
- **What's New**: DocMamba라는 새로운 프레임워크를 발표하여 길이가 긴 문서를 효율적으로 처리할 수 있는 선형 계산 복잡성을 달성했습니다.

- **Technical Details**: DocMamba는 상태공간모델(State Space Model, SSM) 기반으로 구성되어 있으며, 세그먼트 우선 양방향 스캔(Segment-First Bidirectional Scan, SFBS) 기법을 사용하여 연속적인 의미 정보를 포착합니다. 이 모델은 입력 길이에 비례하여 선형 복잡성(linear complexity)을 유지하면서도 글로벌 모델링 능력을 보존합니다.

- **Performance Highlights**: 실험 결과, DocMamba는 FUNSD, CORD, SORIE 등의 데이터셋에서 기존의 LayoutLMv3보다 뛰어난 성능을 보여주었으며, GPU 메모리 사용량을 최대 88.3% 절감하고 2.4배 빠른 처리 속도를 달성했습니다.



### Learning Task Planning from Multi-Modal Demonstration for Multi-Stage Contact-Rich Manipulation (https://arxiv.org/abs/2409.11863)
- **What's New**: 이번 논문에서는 인적용 시연에서 수집된 촉각(tactile) 및 힘-토크(force-torque) 정보를 LLM(대규모 언어 모델)의 작업 계획 생성에 통합하는 새로운 인컨텍스트 러닝(in-context learning) 프레임워크를 제안합니다. 이는 LLM이 새로운 작업 시나리오에 대한 계획을 생성하는 능력을 향상시키기 위해 고안되었습니다.

- **Technical Details**: 제안하는 방법은 여러 센서 모달리티를 갖춘 부트스트랩 reasoning 파이프라인을 통해 시연을 기술적으로 단편화하고, 각 모달리티를 통합하여 포괄적인 작업 계획을 메이킹합니다. 시연 동영상은 촉각 감지(ViTac) 및 힘/토크 신호와 함께 카메라 비디오로 수집되며, 이를 통해 LLM 분석기를 통해 기술 순서를 세분화하고 기술 간의 신뢰할 수 있는 전이 조건을 설정합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 새로운 작업 구성을 계획할 때 LLM의 이해도를 향상시키고 전반적인 계획 성능을 높이는 것으로 나타났습니다. 두 가지 단계적 조작 작업(케이블 조립 및 병뚜껑 조이기)에 대한 실제 실험 결과는 다양한 작업 구성에 대한 높은 성공률을 보여주며, 촉각 기반 기술 조건의 중요성과 다중 모달 시연의 활용 필요성을 강조합니다.



### Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation (https://arxiv.org/abs/2409.11860)
Comments:
          13 pages, 5 figures, 4 Tables

- **What's New**: 본 논문에서는 대규모 e-commerce 환경에서 제품 검색 엔진을 평가하는 프레임워크를 제안합니다. 이 프레임워크는 Multimodal LLMs(대형 다중모달 언어 모델)을 활용하여 개별 쿼리에 맞춤화된 주석 가이드를 생성하고 주석 작업을 수행합니다.

- **Technical Details**: 저자들은 (1) 사용자 쿼리 및 그 맥락에 따라 LLM이 쿼리 요구 사항 목록과 쿼리 구체적 주석 가이드를 생성하며, (2) 검색 엔진에서 제품을 검색하고, (3) 검색된 제품에 대한 텍스트 설명 및 이미지를 기반으로 시각적 설명을 생성하고, (4) 결합된 설명을 입력으로 LLM에 제공하여 해당 쿼리-제품 쌍에 대한 관련 점수를 배정합니다. 이 과정은 데이터베이스에 저장되어 다양한 검색 엔진 간의 평가 일관성을 보장합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 인간 주석과 유사한 품질을 보여 주며, 시간과 비용을 크게 절감하고 문제 발견 속도를 향상시키는 것으로 나타났습니다. 또한, LLMs가 대량의 주석 작업을 효율적으로 처리할 수 있는 반면, 인간 전문가는 더 복잡한 사례에 대해 더 잘 활용될 수 있다고 결론지었습니다.



### MEOW: MEMOry Supervised LLM Unlearning Via Inverted Facts (https://arxiv.org/abs/2409.11844)
- **What's New**: 본 논문에서 제안하는 MEOW는 LLM(Unlearning) 기술을 개선하기 위해 새로운 방법으로, 메모리에서 민감한 정보를 효과적으로 제거하는 동시에 유용성을 유지할 수 있도록 설계되었습니다. 기존의 접근 방식들이 직면했던 주요 도전 과제를 극복하기 위해 gradient descent 기반의 방식을 채택하였으며, MEMO라는 새로운 메트릭을 도입하여 메모리화를 정량화합니다.

- **Technical Details**: MEOW는 gradient descent를 기반으로 하여 LLM에서 특정 데이터를 잊도록 훈련하는 과정에서 생기는 문제점을 해결합니다. 메트릭 MEMO는 LLM에서의 메모리화를 측정하기 위해 설계되었으며, 사용자는 사용 사례 기반으로 특정 반대 사실(inverted facts)을 생성하고, 이를 통해 모델을 미세 조정(fine-tune)함으로써 필요 없는 정보를 잊도록 합니다.

- **Performance Highlights**: MEOW는 NLU(Natural Language Understanding) 및 NLG(Natural Language Generation) 벤치마크에서 기존 방법들보다 뛰어난 결과를 보였으며, 모델의 유용성에서 큰 손실 없이 기억해버리는 정보의 질이 크게 향상되었음을 입증했습니다. 또한, NLU 성능이 일부 데이터셋에서 약간 개선되는 결과도 나타났습니다.



### DPI-TTS: Directional Patch Interaction for Fast-Converging and Style Temporal Modeling in Text-to-Speech (https://arxiv.org/abs/2409.11835)
Comments:
          Submitted to ICASSP2025

- **What's New**: 최근 음성 확산 모델(Speech Diffusion Model)의 발전과 함께, 잘 알려진 U-Net 아키텍처 외에도 Transformer 기반의 모델인 Diffusion Transformer (DiT)가 주목받고 있습니다. 하지만 기존 DiT 음성 모델들은 Mel 스펙트로그램을 일반 이미지로 처리하여 음성의 고유한 음향 특성을 간과하고 있습니다. 이를 해결하기 위해, 저자들은 Directional Patch Interaction for Text-to-Speech (DPI-TTS)라는 새로운 방법을 제안하였으며, 이는 디지털 학습 속도를 향상시키면서도 정확성을 유지하는 특징을 가지고 있습니다.

- **Technical Details**: DPI-TTS는 Mel 스펙트로그램을 패치(각 영역)로 나누고 각 패치와 그 이전 프레임, 저주파 성분 간의 상호작용에 집중하는 방향성 패치 상호작용(Directional Patch Interaction)을 사용합니다. 이 방법은 훈련 속도를 두 배로 증가시키며 합성된 음성의 자연스러움과 시간적 일관성을 향상시킵니다. 또한, 제안된 방법은 Transformer 아키텍처에서 MSH 및 RoPE를 적용하여 더 정교한 스타일 모델링을 가능하게 합니다.

- **Performance Highlights**: DPI-TTS는 훈련 속도를 두 배로 증가시키며, 기존의 기준 모델들에 비해 상당한 성능 향상을 보여줍니다. 제안된 방법은 음성 합성의 자연스러움과 시간적 일관성을 개선하는데 기여하며, 음성 스타일 표현의 정밀한 조절을 가능하게 합니다.



### EFCM: Efficient Fine-tuning on Compressed Models for deployment of large models in medical image analysis (https://arxiv.org/abs/2409.11817)
- **What's New**: 최근 의료 분야에서 대규모 딥러닝 모델이 발전하면서 의료 이미지 분석과 진단에서 뛰어난 성과를 보여주고 있습니다. 하지만 이러한 모델의 거대한 파라미터 수로 인해 메모리 및 추론 지연 문제를 겪고 있습니다. 본 연구에서는 효율적인 압축 모델에 대한 미세 조정(EFCM) 프레임워크를 제안하며, 이에는 비지도 학습을 통한 특성 증류와 미세 조정의 두 단계가 포함됩니다.

- **Technical Details**: EFCM 프레임워크는 비지도 특성 증류(unsupervised feature distillation)와 미세 조정(fine-tuning) 두 단계를 갖고 있습니다. 증류 단계에서는 특성 투영 증류(Feature Projection Distillation, FPD) 방법이 제안되며, TransScan 모듈을 활용하여 수용 필드 크기를 적응적으로 조정하여 모델의 지식 흡수 능력을 향상시킵니다. 슬라이드 수준의 미세 조정 단계에서는 재사용(CLAM), 재훈련(CLAM), End2end Train CLAM을 비교하여 성능을 최적화합니다.

- **Performance Highlights**: 본 연구의 실험 결과, EFCM 프레임워크는 TCGA-NSCLC와 TCGA-BRCA 데이터셋에서 BROW 대규모 모델에 비해 4.33%의 정확도(ACC) 향상과 5.2%의 AUC 향상을 이루어내며, 슬라이드 수준 병리 이미지 문제를 처리하는 데 있어 정확도와 효율성을 크게 개선합니다. 또한, 모델 추론 효율성을 분석한 결과, 증류 미세 조정 방법의 높은 효율성이 강조되었습니다.



### EventAug: Multifaceted Spatio-Temporal Data Augmentation Methods for Event-based Learning (https://arxiv.org/abs/2409.11813)
- **What's New**: 본 논문에서는 EventAug라는 체계적인 데이터 증강 기법을 소개하고, 다중 스케일 시간 통합(MSTI), 공간-유의 미분(mask) (SSEM), 그리고 시간-유의 미분(TSEM)의 세 가지 새로운 방법을 제안해 이벤트 카메라 데이터의 시공간 다양성을 증대시키고자 합니다.

- **Technical Details**: 이벤트 카메라는 저지연(low latency)과 높은 동적 범위(high dynamic range)로 인해 다양한 분야에서 성공적으로 사용되어 왔습니다. 그러나 데이터 부족과 제한된 다양성으로 인해 오버피팅(overfitting)과 불충분한 특징 학습이 발생합니다. 본 연구에서는 이벤트 데이터의 희소성(sparsity)와 불균등한 시공간 분포(spatio-temporal distribution)를 고려하여 효율적으로 학습 데이터를 다양화하는 전용 증강 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 증강 방법이 다양한 작업(task)과 네트워크 구조(backbones)에서 일관되게 성능을 향상시켜, DVS128 제스처에서 4.87%의 정확도 향상을 달성하는 등의 성과를 보였습니다.



### Latent fingerprint enhancement for accurate minutiae detection (https://arxiv.org/abs/2409.11802)
- **What's New**: 이 논문에서는 지문 인식의 주요 과제로 여겨지는 부분적이고 흐려진 지문(latent fingerprints)의 식별 문제를 해결하기 위해 생성적 적대 신경망(Generative Adversarial Networks, GANs)을 활용한 새로운 접근 방식을 제안합니다. 이 연구의 주요 목표는 기존의 지문 인식 기법에서 미세한 세부 사항을 개선하여 포렌식 조사에서 신뢰할 수 있는 식별을 보장하는 것입니다.

- **Technical Details**: 본 연구는 미세한 정보(minutiae information)를 직접 최적화하는 방식을 통해 지문 생성을 구조적으로 접근하여 향상된 latent fingerprints를 생성합니다. 이 과정에서 경량의 구조와 방향 필드를 통합하여 지문 이미지의 걸러진 구조를 보존하며, 다양한 지문 센서와 호환되는 범용 지문 표현을 개발하여 적용 가능한 모든 분야에서 통일성과 상호 운영성을 보장합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 공개 데이터셋을 대상으로 광범위한 평가를 실시하여 기존의 최신 기술들과 비교했을 때 높은 성능을 입증하였습니다. 특히 NIST SD27 데이터셋에 대한 rank-1 식별률이 65%에 불과했던 것과 비교해, 제안된 방법의 성능은 이보다 현저하게 향상된 것으로 나타났습니다.



### The Factuality of Large Language Models in the Legal Domain (https://arxiv.org/abs/2409.11798)
Comments:
          CIKM 2024, short paper

- **What's New**: 이번 연구는 법률 분야에서 대규모 언어 모델(LLMs)을 지식 기반(KB)으로 사용하는 사실성을 검토합니다. 특히, 모델이 불확실할 때 답변을 보류할 수 있도록 허용한 현실적인 사용 시나리오에서 연구가 진행되었습니다.

- **Technical Details**: 이 연구에서는 다양한 사례법 및 입법에 대한 사실 질문의 데이터셋을 설계하고, 여러 LLM을 동일한 데이터셋으로 다양한 평가 방법(정확 매칭, 별칭 매칭, 퍼지 매칭 등)으로 평가하였습니다. 또한, 모델의 응답 보류 전략과 인-컨텍스트 예시가 사실 정확성을 어떻게 향상시키는지 탐구하였으며, 법률 문서에 대한 추가 사전 훈련이 63%에서 81%로 정확성을 향상시킨 것을 보여주었습니다.

- **Performance Highlights**: 단순 정확 매칭에서 벗어나 별칭 및 퍼지 매칭 방법에서 성능이 크게 향상되었으며, 응답을 보류하거나 구체적인 예시를 제공하는 것이 LLM의 정확성을 높였습니다. 이러한 연구 결과는 법률 기술 응용 프로그램의 실질적인 활용 가능성을 보여줍니다.



### Efficient Low-Resolution Face Recognition via Bridge Distillation (https://arxiv.org/abs/2409.11786)
Comments:
          This paper is published in IEEE TIP 2020

- **What's New**: 이 논문은 고해상도 얼굴 모델을 저해상도 얼굴 인식을 위해 경량화하는 브릿지 디스틸레이션(bridge distillation) 방법을 제안합니다. 이를 통해 고해상도 얼굴에서의 지식을 저해상도 얼굴에 효과적으로 전이할 수 있습니다.

- **Technical Details**: 브릿지 디스틸레이션 접근법은 두 단계의 지식 전이를 통해 진행됩니다. 첫 번째 단계에서는 고해상도 데이터셋 간의 교차 전이를 수행하여 저해상도 얼굴에 필요한 특성을 생성합니다. 두 번째 단계에서는 다중 작업 학습(multi-task learning)을 통해 전이된 지식을 저해상도 얼굴에 맞게 조정합니다.

- **Performance Highlights**: 실험 결과, 제안된 학생 모델은 0.21M의 파라미터 수를 가지고 저해상도 얼굴 인식에서 뛰어난 성능을 나타냈으며, GPU, CPU, 모바일에서 각각 14,705, ~934, 763 얼굴/초의 속도를 달성했습니다.



### Distilling Channels for Efficient Deep Tracking (https://arxiv.org/abs/2409.11785)
Comments:
          Published by IEEE TIP 2020

- **What's New**: 이 논문은 딥 트래커의 효율성을 개선하기 위해 새로운 프레임워크인 '채널 증류(channel distillation)'을 제안합니다. 이 방법은 잡음이 있는 채널의 영향을 완화하고, 유용한 채널을 선택하여 특정 이동 물체에 대한 트래킹 성능을 향상시킵니다.

- **Technical Details**: 채널 증류는 디스크리미네이티브 상관 필터(DCF)와 ECO를 예시로 하여, 피처 압축(feature compression), 응답 맵 생성(response map generation), 모델 업데이트(model update)를 통합하여 에너지 최소화 문제로 формulieren합니다. 이는 정보 전달이 효율적으로 이루어지도록 합니다. 이 프레임워크의 통합된 공식화 방식이 트래킹 효과성을 개선합니다.

- **Performance Highlights**: 실험적으로 다양한 벤치마크에서 테스트된 결과, 채널 증류 방식이 정확도, 속도 및 메모리 저장 요구 사항을 개선함을 입증했습니다. 결과적으로, 제안된 딥 트래커는 정확하고, 빠르며, 메모리 요구 사항이 낮습니다.



### Smart Data-Driven GRU Predictor for SnO$_2$ Thin films Characteristics (https://arxiv.org/abs/2409.11782)
Comments:
          19 pages, 14 figures. Baltica Journal, Special Issues, September 2024

- **What's New**: 이 논문에서는 SnO$_2$(110) 얇은 필름의 구조적 특성을 예측하기 위한 Smart GRU(Gated Recurrent Unit) 모델을 제안합니다. 이 모델은 여러 샘플에 대한 데이터를 수집하여 AI 모델을 생성하는데 활용됩니다.

- **Technical Details**: X-Ray diffraction은 결정질 재료의 구조적 특성을 이해하고 추출하는 데 필수적인 방법입니다. 연구자들은 실험 데이터를 수집한 후, AI GRU 모델에서 성능을 높이기 위한 데이터 정제, 정규화 및 특성 감소 과정을 거칩니다. 또한, Scherrer 관계식에 따라 입자 크기(D)를 계산합니다.

- **Performance Highlights**: 제안된 AI 기반 GRU 모델은 재료 특성 예측의 효용을 보여주었으며, 향후 더 복잡한 AI 모델을 적용하여 재료 특성화 분야를 더욱 발전시킬 수 있는 잠재력을 보유하고 있습니다.



### Knowledge Adaptation Network for Few-Shot Class-Incremental Learning (https://arxiv.org/abs/2409.11770)
Comments:
          13 pages;6 figures

- **What's New**: 본 논문에서는 Few-Shot Class-Incremental Learning (FSCIL) 문제를 해결하기 위해 'Knowledge Adaptation Network (KANet)'라는 새로운 방법을 제안합니다. 이 방법은 CLIP 모델을 활용하여 각 클래스에 대한 일반적인 표현을 제공하고, Knowledge Adapter (KA) 모듈을 통해 데이터 특화 지식을 통합함으로써 성능을 향상시킵니다.

- **Technical Details**: KANet은 먼저 CLIP 모델을 기반으로 하여, 적은 수의 샘플을 이용해 새로운 클래스를 점진적으로 인식하는 기능을 수행합니다. KA 모듈은 학습 데이터로부터의 데이터 특화 지식을 요약하여 일반적 표현에 융합시키고, Incremental Pseudo Episode Learning (IPEL) 메커니즘을 도입하여 이전 클래스의 지식을 새로운 클래스에 맞게 조정합니다. KA는 QKF(질문 기반 지식 융합) 메커니즘을 적용하여 데이터의 특화된 지식을 표기합니다.

- **Performance Highlights**: KANet은 CIFAR100, CUB200, ImageNet-R 등 다양한 데이터셋에서 경쟁력 있는 성능을 달성했습니다. 즉, 제안된 방법은 랜덤한 조건에서도 이전 클래스에 대한 성능을 유지하며 새로운 클래스를 효과적으로 인정하는 데 성공했습니다.



### One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation (https://arxiv.org/abs/2409.11764)
- **What's New**: 이번 논문은 제로샷(Zero-shot) 다중 물체 탐색을 위한 새로운 벤치마크를 소개하며, 이전 검색에서 수집한 정보를 활용하여 로봇이 새로운 물체를 보다 효율적으로 찾을 수 있도록 합니다. 또한, 실시간 물체 검색을 위해 재사용 가능한 오픈-어휘(Open-vocabulary) 기능 맵을 구축하고, 일반적인 오류를 완화하는 확률적-의미 맵 업데이트를 제안합니다.

- **Technical Details**: 우리의 방법은 비전-언어 모델(Vision-Language Models, VLMs)을 활용하여 개방형 어휘(open vocabulary)의 공간-의미 지도(spatial-semantic map)를 생성하고 탐색합니다. 이 접근 방식은 CLIP 정렬 기능을 활용하여 2차원 오픈-어휘 신념 맵을 실시간으로 생성하며, 각 맵 위치에 대한 분산을 포함합니다. 확률적 관찰 모델은 의미 기능의 매핑에 대한 일반적인 오류를 완화하는 데 기여합니다.

- **Performance Highlights**: 우리는 Jetson Orin AGX에서 실시간으로 작동하는 실제 로봇 실험을 통해, 단일 및 다중 물체 탐색 작업 모두에서 기존 최첨단 접근 방식을 능가하는 성능을 입증했습니다. 평가한 결과는 우리 방법론이 HM3D 씬 세트에서 다른 선행 연구보다 우수함을 보여줍니다.



### NPAT Null-Space Projected Adversarial Training Towards Zero Deterioration (https://arxiv.org/abs/2409.11754)
- **What's New**: 이번 연구에서는 신경망의 적대적 공격에 대한 취약성을 완화하기 위한 적대적 훈련(adversarial training)의 새로운 접근 방식을 제안하고 있습니다. 특히, null-space projection을 적대적 훈련에 통합하여 정확성과 강건성 간의 균형을 맞추는 혁신적인 두 가지 알고리즘인 NPDA(Null-space Projected Data Augmentation)와 NPGD(Null-space Projected Gradient Descent)를 도입했습니다.

- **Technical Details**: 제안된 방법에서는 null-space projector(영공간 프로젝터)를 활용하여 적대적 샘플과 변동을 결정 경계의 null-space 내로 제한합니다. 이를 통해 신뢰할 수 없는 특성에서 발생하는 공격의 위협을 효과적으로 완화합니다. 실험은 CIFAR10 및 SVHN 데이터셋에서 수행되었으며, 적대적 훈련 방법과 원활하게 결합할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: NPDA 및 NPGD 방법은 추가적인 합성 데이터 또는 모델 용량 없이도 일반화 성능의 거의 0% 손실로 강건성을 크게 향상시키는 성과를 달성하였습니다.



### Exploring Gaze Pattern in Autistic Children: Clustering, Visualization, and Prediction (https://arxiv.org/abs/2409.11744)
- **What's New**: 이 연구는 자폐 스펙트럼 장애(ASD) 아동의 주시 행동을 자동으로 분석하기 위해 최적화된 7가지 클러스터링 알고리즘을 적용한 새로운 방법을 제안합니다. 이 방법은 ASD 아동의 주의 패턴을 सू적하여 진단에 활용할 수 있는 63가지 유의미한 특징을 추출합니다.

- **Technical Details**: 연구에서는 K-Means, K-Medoids, Agglomerative Clustering, BIRCH, DBSCAN, OPTICS, GMM과 같은 다양한 클러스터링 알고리즘을 사용하여 주시 데이터를 군집화한 후, 각 군집의 질을 평가하기 위해 9가지 유효성 지수를 계산합니다. 이를 통해 ASD 아동과 정상 아동 간의 주시 패턴 차이를 분석합니다. 이러한 특징을 통해 예측 모델을 훈련하여 ASD를 진단합니다.

- **Performance Highlights**: 이 방법은 ASD 진단을 위한 자동 생성된 주시 포인트 특성에 대해 81% AUC를 기록하며 최신 성과를 달성하였습니다. 실험 결과 클러스터링 알고리즘이 ASD 아동의 독특한 주시 패턴 분석에서 개선됨을 보여줍니다.



### HARP: Human-Assisted Regrouping with Permutation Invariant Critic for Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2409.11741)
Comments:
          7 pages, 6 figures

- **What's New**: 이번 논문에서는 HARP(Human-Assisted Regrouping with Permutation Invariant Critic)라는 새로운 다중 에이전트 강화 학습 프레임워크를 제안합니다. 이 프레임워크는 그룹 지향적 작업을 위해 설계되었으며, 자동 에이전트 재조정을 통합하여 비전문가가 최소한의 개입으로 효과적인 지침을 제공할 수 있도록 합니다.

- **Technical Details**: HARP는 에이전트가 협력적 작업을 최적화할 수 있도록 동적으로 그룹 구성과 재조정을 수행합니다. 에이전트는 배치(Batch) 단계에서 인간의 지원을 적극적으로 요청하고, Permutation Invariant Group Critic을 통해 인간이 제안한 그룹 구성을 평가하고 개선합니다. 또한, 이 과정에서 인간의 지침을 활용하여 에이전트의 의사결정 능력을 향상시킵니다.

- **Performance Highlights**: StarCraft II의 세 가지 난이도에서 협력 문제에 대한 실험 결과, 배치 단계에서 제한된 인간의 지침을 활용할 때 HARP는 에이전트의 성능을 10% 이상 향상시킬 수 있음을 증명하였습니다.



### InverseMeetInsert: Robust Real Image Editing via Geometric Accumulation Inversion in Guided Diffusion Models (https://arxiv.org/abs/2409.11734)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문에서는 Geometry-Inverse-Meet-Pixel-Insert (GEO)라는 새로운 이미지 편집 기법을 소개합니다. GEO는 사용자 요구에 맞춘 유연한 이미지 편집 기능을 제공하며, 텍스트 프롬프트와 이미지 프롬프트를 결합해 다양한 편집 결과를 생성합니다. 특히 이 방법은 사전 학습이 필요하지 않으며, 새로운 기하학적 누적 손실(geometric accumulation loss)과 향상된 이미지 프롬프트 기술을 통해 편집 결과의 정확성과 세밀함을 보장합니다.

- **Technical Details**: GEO 기법은 Denoising Diffusion Implicit Model (DDIM)의 역전환 과정을 개선하는 새로운 기하학적 누적 손실을 포함하고 있습니다. 이 손실은 이미지의 기하학적 특성을 보존하기 위해 입력-인코딩된 이미지 잠재(latent)를 기준점으로 활용합니다. 편집 과정에서는 초기 픽셀 수준 편집 후 기하학적 누적 손실을 이용한 역전환이 진행되며, 이는 기존 DDIM 방법보다 세부 사항의 보존을 향상시킵니다.

- **Performance Highlights**: GEO는 다양한 이미지 유형 및 복잡한 프롬프트 편집 시나리오에서 고충실도 편집 결과를 제공합니다. 사용자는 원하는 길이의 텍스트 프롬프트를 입력하여 세밀하고 다중 영역 편집을 수행할 수 있으며, 배경 세부 사항도 효과적으로 보존합니다. 따라서 이 기법은 시각적 세부 조정에서 특히 뛰어난 성능을 보여줍니다.



### GUNet: A Graph Convolutional Network United Diffusion Model for Stable and Diversity Pose Generation (https://arxiv.org/abs/2409.11689)
- **What's New**: PoseDiffusion은 자연어를 기반으로 2D 인간 포즈 스켈레톤을 생성하는 데 있어서 첫 번째로 제안된 생성적 프레임워크입니다. 이 모델은 GANs (Generative Adversarial Networks) 대신에 디퓨전 모델(Diffusion Model)을 사용하여 더 높은 다양성과 구조적 정확성을 구현합니다.

- **Technical Details**: PoseDiffusion의 주 모델인 GUNet는 그래픽 컨볼루션 신경망(Graphical Convolutional Neural Network)을 포함하여, 포즈 스켈레톤의 공간적 관계를 학습합니다. 각 키 포인트를 개별적으로 예측할 수 있도록 2D 인간 포즈 스켈레톤을 별도의 특징 채널로 표현하며, 교차 주의(cross-attention) 기법을 통해 텍스트 조건을 도입합니다.

- **Performance Highlights**: PoseDiffusion은 기존의 최첨단(SOTA) 알고리즘보다 텍스트 기반 포즈 스켈레톤 생성에서 안정성과 다양성이 뛰어납니다. 정성적 분석은 Stable Diffusion에서 제어 가능한 생성에 대한 우수성을 입증합니다.



### Detecting Underdiagnosed Medical Conditions with Deep Learning-Based Opportunistic CT Imaging (https://arxiv.org/abs/2409.11686)
- **What's New**: 이번 연구는 복부 컴퓨터 단층 촬영(CT)을 통한 기회를 활용한 진단(Opportunistic CT)의 효과를 조사하였으며, 이를 통해 저진단 상태인 사코펜리아(sarcopenia), 간지방증(hepatic steatosis), 복수(ascites)를 발견할 수 있는 가능성을 제시합니다.

- **Technical Details**: 2,674개의 병원 입원 환자의 CT 스캔을 분석하여, 기회를 활용한 CT 이미지에서 파생된 이미징 표현형(imaging phenotypes)과 방사선 보고서 및 ICD 코드 간의 불일치를 식별하였습니다.

- **Performance Highlights**: 사코펜리아는 0.5%, 간지방증은 3.2%, 복수는 30.7%의 비율로 ICD 코드화되었으며, 연구 결과는 기회를 활용한 CT가 진단의 정확성을 높이고 위험 조정 모델의 정확성을 향상시킬 수 있는 잠재력을 지니고 있음을 보여줍니다.



### Hypergraph-based Motion Generation with Multi-modal Interaction Relational Reasoning (https://arxiv.org/abs/2409.11676)
- **What's New**: 이번 연구는 자율주행차(Autonomous Vehicles, AVs)의 모션 예측을 위한 통합 프레임워크를 소개하며, 복잡한 주행 환경에서의 차량 간 상호작용을 모델링하고 예측의 불확실성을 처리하는 것을 목표로 한다.

- **Technical Details**: 소개된 프레임워크는 Relational Hypergraph Interaction-informed Neural mOtion generator (RHINO)와 Graph-based Interaction-awaRe Anticipative Feasible Future Estimator (GIRAFFE) 두 부분으로 구성되어 있다. RHINO는 다중 스케일 하이퍼그래프(hypergraph) 신경망을 활용하여 여러 차량의 집단 간 상호작용 및 다중 모드의 주행 행동을 모델링한다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 기존의 모델에 비해 예측 정확성과 신뢰성을 높이며, 동적 교통 시나리오에서 사회적 인식을 높이는 데 기여하는 것으로 나타났다.



### Agent Aggregator with Mask Denoise Mechanism for Histopathology Whole Slide Image Analysis (https://arxiv.org/abs/2409.11664)
- **What's New**: 이 논문은 AMD-MIL이라는 새로운 방법론을 제안하여 병리학적 이미지 분석에서의 성능을 향상시키는 것을 목표로 합니다. 특히, agent aggregator와 mask denoise 메커니즘을 도입하여 WSI(classification of Whole Slide Images)의 분류 정확성을 높입니다.

- **Technical Details**: AMD-MIL은 query와 key 간의 instance 중요도를 계산하기 위한 intermediate variable 역할을 하는 agent token을 사용합니다. masking 및 denoising 매트릭스는 동적으로 low-contribution representation을 필터링하고 noise를 제거합니다. 이 방법은 self-attention의 연산 복잡성을 줄이면서도 각 instance 간의 정보를 잘 포착할 수 있도록 해줍니다.

- **Performance Highlights**: AMD-MIL은 CAMELYON-16, CAMELYON-17, TCGA-KIDNEY, TCGA-LUNG datasets에서 extensive한 실험을 통해 기존의 최신 기법들에 비해 우수한 성능을 보였습니다.



### GReDP: A More Robust Approach for Differential Privacy Training with Gradient-Preserving Noise Reduction (https://arxiv.org/abs/2409.11663)
- **What's New**: 본 논문에서는 심층 학습(Deep Learning) 모델의 교육 과정에서 데이터 프라이버시(Differential Privacy, DP)를 보장하는 새로운 방법인 GReDP를 제안합니다. GReDP는 기존 방법들보다 적은 노이즈(noise)로 모든 그래디언트 정보를 유지하여 모델 유용성을 향상시킵니다.

- **Technical Details**: GReDP는 모델 그래디언트를 주파수 영역(frequency domain)에서 계산하고, 고속 푸리에 변환(Fast Fourier Transformation, FFT)을 통해 가우시안 노이즈(Gaussian noise) 메커니즘을 적용합니다. 본 방법은 노이즈를 시간 영역(time domain)에서 처리하여 그라디언트 정보를 유지하면서 노이즈 수준을 낮추는 새로운 접근법을 사용합니다.

- **Performance Highlights**: GReDP는 기존의 DP 방법들과 비교하여 동일한 프라이버시 예산하에서도 노이즈 수준을 절반으로 줄이며, 여러 다양한 아키텍처에서 실험 결과를 통해 모든 설정에서 consistently 더 나은 모델 유용성을 보여주었습니다.



### Few-Shot Class-Incremental Learning with Non-IID Decentralized Data (https://arxiv.org/abs/2409.11657)
- **What's New**: 본 논문은 Federated Few-Shot Class-Incremental Learning (F2SCIL)이라는 개념을 도입하여, 분산된 클라이언트에서 제한된 데이터로 새로운 클래스를 학습하는 분산 머신러닝 패러다임을 탐구합니다. 이는 데이터 프라이버시를 유지하면서도 클라이언트가 새로운 클래스를 지속적으로 업데이트할 수 있도록 합니다.

- **Technical Details**: F2SCIL은 클라이언트가 로컬 모델을 업데이트하고, 해당 업데이트를 중앙 서버에 전송해 글로벌 모델을 학습하는 구조로 되어 있습니다. 이 과정에서 Synthetic Data-Driven (SDD) 프레임워크를 활용하여 기존 지식을 유지하고 새로운 지식을 습득하는데 도움을 줍니다. Noise-Aware Generative Replay (NAGR) 모듈은 로컬 모델을 미세 조정하기 위해 새로운 데이터와 재생 데이터의 균형을 유지합니다. 또한, Class-Specific Weighted Aggregation (CSWA) 전략을 적용해 클라이언트 모델의 집합을 효율적으로 수행합니다.

- **Performance Highlights**: 세 가지 유명한 데이터셋에 대한 포괄적인 실험을 통해 제안한 프레임워크의 효과성과 그 우수성이 강조되었습니다. 이 연구는 FSCIL 문제를 해결하기 위한 혁신적인 접근 방식을 제공하면서, 제한된 데이터와 데이터 이질성으로 인한 여러 도전에 대응합니다.



### How to Build the Virtual Cell with Artificial Intelligence: Priorities and Opportunities (https://arxiv.org/abs/2409.11654)
- **What's New**: 이 논문에서는 인공지능(AI) 기반의 가상 세포(AI Virtual Cells)를 제안합니다. 이는 다양한 조건에서 세포 및 세포 시스템의 강력한 표현을 생물학적 데이터로부터 직접 학습하여 예측하는 새로운 접근 방식을 의미합니다.

- **Technical Details**: AI Virtual Cells는 대규모, 다중 모드의 신경망 모델로 설계되며, 데이터로부터 직접 학습하여 세포의 행동을 시뮬레이션하고 구성하는 데 필요한 기능을 갖춰야 합니다. 이는 분자, 세포, 다세포 수준에서의 기능적 특성을 통합하면서 실험적 데이터를 기반으로 한 예측실험을 허용합니다.

- **Performance Highlights**: AI Virtual Cells는 신약 타겟 탐색, 세포의 외부 자극에 대한 반응 예측 및 가설 탐색의 규모를 확대하는 데 도움을 줄 수 있습니다. 이러한 가상 세포 모델을 통해 세포 메커니즘에 대한 포괄적인 예측 이해가 가능해질 것입니다.



### Art and Science of Quantizing Large-Scale Models: A Comprehensive Overview (https://arxiv.org/abs/2409.11650)
- **What's New**: 이 논문은 대규모 신경망 모델의 양자화(quantization)와 관련된 원칙, 도전 과제 및 방법론에 대한 포괄적인 개요를 제공합니다. 신경망이 점점 더 복잡한 아키텍처로 발전함에 따라 생기는 계산 및 에너지 비용의 증가를 다룹니다.

- **Technical Details**: 주요 초점은 모델 양자화가 모델 크기를 줄이고 효율성을 향상시키면서 정확도를 크게 저하시키지 않는 핵심 접근 방식임을 강조합니다. 포스트 훈련 양자화(post-training quantization, PTQ)와 양자화 인식 훈련(quantization-aware training, QAT) 등 다양한 양자화 기법을 탐구하며, LLM-QAT, PEQA(L4Q), ZeroQuant, SmoothQuant 등의 최신 알고리즘을 분석합니다.

- **Performance Highlights**: 각 양자화 기법이 아웃라이어(outliers), 중요도 가중치(importance weighting), 활성화 양자화(activation quantization) 같은 문제를 어떻게 해결하는지를 비교 분석하며, 대규모 모델의 지속 가능하고 접근 가능한 배포에 기여합니다.



### Few-Shot Learning Approach on Tuberculosis Classification Based on Chest X-Ray Images (https://arxiv.org/abs/2409.11644)
Comments:
          6 pages. Pre-print

- **What's New**: 이번 연구에서는 결핵(***Tuberculosis***)의 조기 발견을 위해 인공지능(***AI***)을 활용하여 흉부 X-선 이미지를 분류하는 방법을 제안합니다. 특히, 데이터 불균형 문제를 해결하기 위해 few-shot learning (***FSL***) 접근 방식을 Prototypical Network 알고리즘을 통해 구현했습니다.

- **Technical Details**: 연구에서는 TBX11K Chest X-ray 데이터세트에서 특징 추출을 위해 ResNet-18, ResNet-50, VGG16 모델을 비교했습니다. 이를 통해 각각 98.93%, 98.60%, 33.33%의 분류 정확도를 기록하였습니다. 이 연구는 데이터 불균형을 완화하는 데 있어 제안된 방법이 우수함을 보여줍니다.

- **Performance Highlights**: ResNet-18과 ResNet-50 모델의 성능이 가장 뛰어나며, VGG16의 성능은 상대적으로 낮은 결과를 보입니다. 이는 결핵 분류 응용에서 데이터 불균형 문제를 해결하는 데 있어 제안된 방법이 매우 유용하다는 것을 나타냅니다.



### Combating Phone Scams with LLM-based Detection: Where Do We Stand? (https://arxiv.org/abs/2409.11643)
Comments:
          2 pages, 1 figure

- **What's New**: 이번 연구는 전통적인 전화 사기 탐지 방법을 넘어서, 대규모 언어 모델(LLM)을 활용하여 사기성 전화의 실시간 감지를 탐구하고 있습니다.

- **Technical Details**: 연구팀은 다양한 대화 데이터셋을 분석하여 사기 대화를 식별하는 데 필요한 특성 및 구조를 연구하였으며, TF-IDF 분석을 통해 주요 단어를 추출하였습니다. 이후 SVM, RF, KNN 등의 전통적 머신러닝 분류기를 평가했으며, LLM을 사용한 실험에서 GPT-4와 다른 모델들이 우수한 정확도를 보여주었습니다.

- **Performance Highlights**: LLM 기반의 탐지기는 0.98 이상의 정확도를 기록했지만, 재현율(recall)은 최대 0.72로, 이는 여전히 많은 사기 전화를 탐지하지 못하고 있음을 나타냅니다. 또한, 모델 간 일관성이 결여되어 있어, 신뢰성 점수를 도입하여 평가하였습니다.



### HRA: A Multi-Criteria Framework for Ranking Metaheuristic Optimization Algorithms (https://arxiv.org/abs/2409.11617)
Comments:
          13 pages, 1 figure

- **What's New**: 이 논문에서는 메타휴리스틱 알고리즘의 성능을 효율적으로 평가하기 위한 새로운 Hierarchical Rank Aggregation (HRA) 알고리즘을 소개합니다. HRA 알고리즘은 다양한 기준 및 차원에서 알고리즘의 성능을 기반으로 계층적으로 순위를 매기고, 메타휴리스틱 알고리즘의 비교 및 선택을 단순화하는 방법을 제안합니다.

- **Technical Details**: HRA 알고리즘은 여러 벤치마크 함수와 차원에서 성능 메트릭을 수집하여 시작하며, 각 성능 측정에 대해 rank-based normalization을 적용하여 비교 가능성을 보장합니다. 또한, 여러 계층 수준에서 이러한 순위를 결합하기 위해 강력한 TOPSIS 집계를 이용하여 알고리즘의 포괄적인 순위를 생성합니다.

- **Performance Highlights**: 본 연구는 CEC 2017 대회의 데이터를 활용하여 HRA 프레임워크의 강건성과 효율성을 입증했습니다. 30개의 벤치마크 함수에 대해 13개의 메타휴리스틱 알고리즘을 4개의 차원에서 평가하였으며, 이 시스템은 알고리즘의 상대적인 장점과 단점을 해석하는 데 기여하여 특정 최적화 문제에 가장 적합한 알고리즘 선택을 용이하게 합니다.



### Harnessing AI data-driven global weather models for climate attribution: An analysis of the 2017 Oroville Dam extreme atmospheric river (https://arxiv.org/abs/2409.11605)
Comments:
          This Work has been submitted to Artificial Intelligence for the Earth Systems

- **What's New**: AI 데이터 기반 모델(예: Graphcast, Pangu Weather, Fourcastnet, SFNO)이 스토리라인 기반의 기후 귀속 분석에 활용되었습니다. 이 모델들은 짧은 추론 시간 덕분에 연구할 수 있는 사건 수를 증가시키고 공적 관심이 고조될 때 실시간 귀속을 제공합니다.

- **Technical Details**: 2017년 2월 극단적인 대기 강 흐름(Atmospheric River) 사건을 분석하였으며, 이는 북캘리포니아의 오로빌 댐(Oroville dam) 방류 사건에 기여했습니다. 초기 조건을 전 산업시대(pre-industrial)와 21세기 후반 온도 기후 변화 신호로 변동시켜 과거와 미래의 시뮬레이션을 생성했습니다. AI 모델들의 결과는 동적 모델(Dynamical Model)과 비교되어 두 기후 환경 하의 그럴듯한 의사 현실(pseudo-reality)을 나타냈습니다. 각 AI 모델 테스트를 통해 기후 귀속 반응의 물리적성을 이해하는 데 유용한 정보를 제공합니다.

- **Performance Highlights**: AI 모델은 현재와 전 산업 시대 비교 시 오로빌 댐 위의 수증기가 5-6% 증가한다고 예측하였으며, 이는 동적 모델과 일치하는 결과입니다. 그러나 AI 모델은 동적 모델이 상상하는 의사 현실보다 약한 귀속 값을 시뮬레이션하는 경향이 있어, 특히 21세기 후반 환경에서 과소 추정할 수 있는 점이 지적되었습니다. AI 모델로 생성된 대규모 앙상블(>500 멤버)은 통계적으로 유의미한 현재와 전 산업 시대 귀속 결과를 낳았으나, 동적 모델의 >20 멤버 앙상블과는 차이를 보였습니다.



### No Saved Kaleidosope: an 100% Jitted Neural Network Coding Language with Pythonic Syntax (https://arxiv.org/abs/2409.11600)
Comments:
          12 pages, 3 figures and 3 tables

- **What's New**: 이 논문에서는 C++, LLVM 및 Cuda를 이용하여 인공지능 신경망(Artificial Neural Networks) 훈련을 위한 JIT 컴파일러를 개발했습니다. 이 컴파일러는 객체 지향 특성, 강한 타입, 데이터 전처리를 위한 병렬 작업자(parallel workers), 파이썬 스타일의 표현식(syntax), PyTorch와 유사한 모델 선언 및 자동 미분(Automatic Differentiation) 기능을 제공합니다.

- **Technical Details**: 이 논문은 JIT 언어의 초기 버전인 No Saved Kaleidoscope (NSK) 언어를 설명합니다. CUDA 및 cuDNN을 사용하여 VRAM 관리, 고성능 행렬 곱셈을 위한 cuBLAS 구현, 컨볼루셔널 레이어 지원 등의 메커니즘을 구현했습니다. 컴파일러는 파라미터의 다중 스코프 지원 및 병렬처리 구현을 위해 인덴테이션 기반의 문법 구조를 사용하여 단순하면서도 강력한 병렬 처리 코드를 허용합니다.

- **Performance Highlights**: ImageNet에서의 Residual Convolutional Neural Networks 실험에서는 비슷한 속도를 유지하였으나 성능이 저하된 것으로 나타났습니다. GRU 네트워크 실험에서는 유사한 정확도가 나왔지만 속도는 저하되었습니다. 그러나 CIFAR-10 벤치마크에서는 PyTorch와 비슷한 성능과 속도를 달성하여 유망한 결과를 보여주었습니다.



### Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation (https://arxiv.org/abs/2409.11598)
- **What's New**: 이번 논문은 Retrieval-Augmented Generation (RAG) 시스템에 공정한 순위 평가를 통합한 최초의 체계적인 평가를 제시합니다. 연구는 RAG 시스템에서 각 관련 항목의 공정한 노출을 측정하여 공정한 순위의 중요성을 강조합니다.

- **Technical Details**: RAG 시스템에서 공정한 순위란, 각각의 관련된 항목이 얼마나 공정하게 노출되는지를 측정하는 것을 의미합니다. 연구는 공정성을 고려하여 세 가지 주제에 대한 결과를 분석하며, 공정한 순위 시스템을 채택하는 RAG 모델이 전통적인 RAG보다 높은 생성 품질을 유지할 수 있음을 발견했습니다. 해당 연구는 7개 데이터셋을 이용하여 9가지 RAG 시스템을 분석하였습니다.

- **Performance Highlights**: 공정한 순위를 포함한 RAG 시스템이 높은 생성 품질을 유지하면서도 기존 RAG 시스템을 초월할 수 있는 사례가 있음을 발견했습니다. 이는 공정성을 보장하면서도 시스템의 효과성을 훼손하지 않고 양질의 서비스를 제공할 수 있음을 나타냅니다.



### Self-Contrastive Forward-Forward Algorithm (https://arxiv.org/abs/2409.11593)
- **What's New**: 본 논문에서는 Self-Contrastive Forward-Forward (SCFF) 메소드를 소개하며, 이는 Self-Supervised Contrastive Learning에서 영감을 받아 기존의 Forward-Forward (FF) 알고리즘의 성능 한계를 극복하는 것을 목표로 합니다. SCFF는 다양한 데이터셋에 적용 가능한 긍정적인 예와 부정적인 예를 생성하여 FF 알고리즘의 효용성을 확장합니다.

- **Technical Details**: SCFF는 각 데이터 샘플을 상호 대비를 통해 학습하며, 입력을 자신과 연결하여 긍정적인 예를, 훈련 세트에서 임의 선택된 예와 연결하여 부정적인 예를 만듭니다. 이를 통해 FF 알고리즘을 사용한 이미지 분류 및 시퀀스 데이터 문제 해결을 위한 새로운 가능성을 열어줍니다.

- **Performance Highlights**: SCFF는 MNIST에서 MLP를 사용해 98.7%, CIFAR-10에서 CNN을 통해 80.75%, STL-10에서 77.3%의 정확도를 기록하며 기존의 알고리즘들을 능가합니다. SCFF는 또한 시퀀스 데이터에서 첫 번째 성공적인 FF 훈련을 시연하며, Free Spoken Digit Dataset (FSDD)에서 10% 이상의 정확도 향상을 보여줍니다.



### ProSLM : A Prolog Synergized Language Model for explainable Domain Specific Knowledge Based Question Answering (https://arxiv.org/abs/2409.11589)
Comments:
          Accepted at NeSy 2024

- **What's New**: 본 논문에서는 기존의 신경 기계 학습 모델의 불투명성을 해결하기 위해 기호 기반의 신뢰할 수 있는 접근 방식을 제안합니다. 특히, \\systemname{}라는 새로운 신경-기호 프레임워크를 통해 질문-응답 시스템에서 LLM의 신뢰성과 견고성을 향상시키는 방법을 제시합니다.

- **Technical Details**: \\systemname{}는 도메인 특화된 지식 기반(knowledge base), 논리적 추론 시스템, 기존의 LLM과 통합된 구조로 이루어져 있습니다. 이 프레임워크는 (1) 맥락 수집(context gathering): 주어진 질의에 대해 설명 가능하고 관련된 맥락을 생성하고, (2) 검증(validation): 지식 기반에 따라 진술의 사실적 정확성을 확증하고 검증하는 기능을 포함하고 있습니다.

- **Performance Highlights**: 이 프레임워크는 사용자에게 시스템이 특정 출력을 도출한 과정을 이해할 수 있도록 돕고, LLM의 추가적인 학습이나 파인 튜닝 없이도 계산 효율성을 제공합니다. 실험 결과는 향후 LLM의 활용 방안에 관한 새로운 방향성을 제시합니다.



### Uncertainty Decomposition and Error Margin Detection of Homodyned-K Distribution in Quantitative Ultrasound (https://arxiv.org/abs/2409.11583)
Comments:
          4 pages, 2 figures

- **What's New**: 이번 연구에서는 Bayesian Neural Networks (BNNs)를 사용하여 Homodyned K-distribution (HK-distribution)의 매개변수를 추정하고, 예측 불확실성을 에피스테믹(epistemic)과 알레아토릭(aletorik)으로 분해하는 방법론을 제안합니다.

- **Technical Details**: 본 연구는 BNN을 이용하여 HK-distribution 매개변수(α, k)의 에피스테믹 및 알레아토릭 불확실성을 계산하는 방법을 제시하며, 시뮬레이션 및 실험 데이터를 사용하여 예측 오차와 불확실성 간의 관계를 탐구합니다. 에피스테믹 불확실성은 네트워크 가중치에 대한 불확실성을 의미하고 알레아토릭 불확실성은 관측 데이터의 고유한 노이즈를 포착합니다.

- **Performance Highlights**: BNNs를 통해 HK-distribution 매개변수의 추정 정확도를 향상시키고, 모델 출력의 신뢰성 평가를 위한 불확실성 정량화를 가능하게 했습니다.



### Automating proton PBS treatment planning for head and neck cancers using policy gradient-based deep reinforcement learning (https://arxiv.org/abs/2409.11576)
- **What's New**: 본 논문은 두 번째로, 심층 강화 학습(Deep Reinforcement Learning, DRL)에 기반한 자동 치료 계획 수립 모델을 제안하고 있습니다. 기존 Q-learning 방법들을 벗어나 근접 정책 최적화(Proximal Policy Optimization, PPO) 알고리즘과 용량 분포 기반 보상 함수를 활용하여 머리와 목(Head and Neck, H&N) 암의 치료 계획 수립을 자동화하는 방법을 보여줍니다.

- **Technical Details**: 이 모델은 실험적 규칙을 사용하여 목표 부피와 위험 장기(Organ-at-Risk, OAR)로부터 보조 계획 구조를 생성합니다. 그런 다음, 이러한 계획 목표는 내부 최적화 엔진에 공급되어 스팟 모니터 단위(Spot Monitor Unit, MU) 값을 생성하며, PPO를 통해 의사결정 정책 네트워크가 계속 변화하는 액션 공간에서 계획 목표 매개변수를 조정하여 최종 PBS 치료 계획을 정제합니다.

- **Performance Highlights**: 이 모델이 생성한 양성자 H&N 치료 계획은 인간이 생성한 계획과 비교하여 OAR을 더 잘 보호하면서도 동등하거나 우수한 목표 커버리지를 보여주었습니다. 또한, 간암 치료에 대한 추가 실험을 통해 이 방법이 다른 치료 부위에도 성공적으로 일반화될 수 있음을 입증했습니다.



### Preference Tuning with Human Feedback on Language, Speech, and Vision Tasks: A Survey (https://arxiv.org/abs/2409.11564)
Comments:
          Survey paper

- **What's New**: 본 논문은 인간의 피드백과의 통합을 통해 딥 생성 모델을 인간의 선호도에 aligned(정렬)시키는 preference tuning(선호 조정)에 관한 최신 발전을 포괄적으로 검토합니다.

- **Technical Details**: 논문은 preference tuning을 reinforcement learning(강화 학습) 프레임워크 내에서 세 가지 주요 섹션으로 구성하였습니다. 여기에는 강화 학습을 이용한 모델과 데이터셋, 그리고 다양한 정책 접근 방법에 대한 설명이 포함되어 있습니다.

- **Performance Highlights**: LLM(대형 언어 모델)에서 preference tuning은 작업 특화 기술을 개선하고 불필요한 출력(undesired outputs)을 피하는 데 기여하며, 다국어 LLM에서도 이점이 있는 것으로 나타났습니다.



### Multi-Domain Data Aggregation for Axon and Myelin Segmentation in Histology Images (https://arxiv.org/abs/2409.11552)
Comments:
          12 pages, 8 figures

- **What's New**: 이번 연구는 다양한 이미징 모달리티(전송 전자현미경, 주사 전자현미경, 밝은장면 광학 현미경 등)와 종(쥐, 쥐, 인간 등)에서 수집된 데이터를 통합하여 신경 영상에서 축삭(axon)과 미엘린(myelin) 분할(segmentation)을 위한 공용 다중 도메인(segmentation) 모델을 제공합니다. 이를 통해 특정 이미징 데이터 세트에 맞춘 모델 구축에 따른 비효율성을 줄이고 공통적으로 사용할 수 있는 모델을 제시합니다.

- **Technical Details**: 이 연구에서는 CNN(합성곱 신경망) 아키텍처를 기반으로 한 다중 도메인 분할 모델을 개발했습니다. 이 모델은 다양한 데이터를 집계하여 훈련되었으며, 각 연구 그룹의 이미지 데이터를 포함하여 서로 다른 원인에 대한 능동 학습(active learning) 전략을 활용하였습니다. 또한, 이 모델은 특정 도메인에 특화된 기존 모델보다 더 나은 일반화 성능을 보여주고, 사용이 용이하며 유지 관리가 용이하다는 장점을 가지고 있습니다.

- **Performance Highlights**: 제안된 다중 도메인 모델은 단일 모달리티 전용 학습자보다 우수한 성능을 발휘하며(p=0.03077), 다양한 도메인에서 충분한 성능 향상을 보여줍니다. 또한, 연구자들이 사용하는 데 더욱 간편하며, 오픈 소스 소프트웨어에 잘 통합되어 접근성을 높입니다.



### Small Language Models can Outperform Humans in Short Creative Writing: A Study Comparing SLMs with Humans and LLMs (https://arxiv.org/abs/2409.11547)
- **What's New**: 이 연구는 소규모 언어 모델(SLM)인 BART Large의 창작 허구 글쓰기 능력을 평가하고, 이를 인간 창작자 및 대규모 언어 모델(LLM)인 GPT-3.5와 GPT-4o와 비교했습니다. 실험을 통해 BART Large는 대부분의 평가 기준에서 인간보다 우수한 성과를 나타냈으며, 창의성 측면에서만 열세를 보였습니다.

- **Technical Details**: 연구는 두 가지 실험으로 구성되어 있습니다. 첫 번째는 인간 평가로, 68명의 참가자들이 생성된 짧은 이야기를 평가했습니다. 평가 기준에는 문법, 관련성, 창의성 및 매력이 포함됩니다. 두 번째 실험은 생성한 텍스트의 언어적 특성을 비교하는 질적 분석으로, GPT-4o가 높은 일관성(coherence)을 보였지만 예측 가능한 내러티브를 생성했다는 결과를 냈습니다.

- **Performance Highlights**: BART Large는 전체 점수에서 2.11을 기록하며 인간 점수 1.85보다 14% 향상된 성과를 보였고, 창의성 측면에서 BART의 15%의 이야기가 창의적으로 새롭다고 평가된 반면, GPT-4o는 3%만이 창의적으로 새로운 것으로 간주되었습니다. 이는 모델의 크기가 창의성에 미치는 영향을 보여줍니다.



### NCT-CRC-HE: Not All Histopathological Datasets Are Equally Usefu (https://arxiv.org/abs/2409.11546)
- **What's New**: 이 논문에서는 NCT-CRC-HE-100K 대장암 데이터셋의 분석을 통해 임상 관련성이 없는 데이터 특정 편향(data-specific biases)이 결과에 미치는 영향을 조사합니다. 색상 정규화의 부적절성, 클래스 간 심각한 JPEG 아티팩트, 잘못된 이미지 동적 범위 처리로 인해 완전히 손상된 조직 샘플 등이 주요 문제로 밝혀졌습니다.

- **Technical Details**: NCT-CRC-HE 데이터셋은 100,000개의 훈련 패치와 7,180개의 테스트 패치를 포함하며, 9개 조직 클래스에 속합니다. 저자들은 이 데이터셋이 여러 혁신적인 이미지 전처리와 깊이 있는 학습 모델을 사용한 많은 연구에서 사용되었다고 강조합니다. 간단한 EfficientNet-B0 모델이 97.7%의 정확도로 기존의 모든 솔루션을 초월했음을 보여주었습니다.

- **Performance Highlights**: 가장 단순한 모델이 3개의 특징(빨강, 초록, 파랑 색상 강도)만으로도 50% 이상의 정확도를 보였고, 색상 히스토그램을 사용했을 때 82%의 정확도를 달성했습니다. 이 논문은 NCT-CRC-HE 데이터셋과 관련된 모든 결과는 이전의 모든 전용 솔루션보다 뛰어난 성과를 나타내었다고 결론짓습니다.



### Mamba Fusion: Learning Actions Through Questioning (https://arxiv.org/abs/2409.11513)
- **What's New**: MambaVL은 선택적 상태 공간 모달리티 융합(selective state space modality fusion)을 활용하여 긴 거리 의존성을 효율적으로 포착하고 비전과 언어 데이터의 공동 표현(joint representations)을 학습하는 혁신적인 모델입니다.

- **Technical Details**: MambaVL은 비전(Vision)과 언어(Language) 브랜치에 대해 공유 상태 전이 행렬(shared state transition matrix)을 사용하여 두 가지 모달리티 간의 정보를 교환하고, 이러한 구조가 선택 메커니즘을 확장할 수 있도록 합니다. 이를 통해 모델은 양쪽 모달리티에서 관련 정보를 효과적으로 선택하고 융합할 수 있습니다.

- **Performance Highlights**: MambaVL은 Epic-Kitchens-100 데이터셋에서 액션 인식(action recognition) 및 액션 예측(action anticipation)에서 최첨단 성능(state-of-the-art performance)을 달성하며, 기존 방법들을 초월하는 결과를 보여주고 있습니다.



### FedNE: Surrogate-Assisted Federated Neighbor Embedding for Dimensionality Reduction (https://arxiv.org/abs/2409.11509)
- **What's New**: 이번 연구에서는 Federated Learning(FL) 환경에서의 차원 축소 문제를 해결하기 위한 새로운 접근법인 FedNE를 제시합니다. FedNE는 데이터 공유 없이 다양한 클라이언트 간의 모델 학습을 가능하게 하며, neighborhood structure를 효과적으로 시각화하는 방법입니다.

- **Technical Details**: FedNE는 FedAvg 프레임워크와 contrastive Neighbor Embedding(NE) 기술을 통합합니다. 이를 위해 각 클라이언트가 그들의 local repulsive loss function을 요약하는 surrogate model을 학습하고 공유합니다. 또한, local data를 증강하기 위해 data-mixing 전략을 도입하여 인접 이웃 문제를 해결합니다.

- **Performance Highlights**: FedNE는 다양한 synthetic 및 real-world 데이터세트에서 실험하여 neighborhood data 구조를 잘 보존하고, global embedding 공간에서의 정렬을 개선함을 보여주었습니다. 실험 결과, FedNE는 기존의 baseline 방법보다 우수한 성능을 나타냈습니다.



### Egalitarian Language Representation in Language Models: It All Begins with Tokenizers (https://arxiv.org/abs/2409.11501)
Comments:
          Content - 8 pages, References - 3 pages

- **What's New**: 최근 연구에서는 Tokenization(토크나이제이션)이 언어 모델과 인간 언어 사이에서 중요한 역할을 한다는 점을 강조하고 있습니다. 특히, 복잡한 스크립트 언어인 타밀어, 신할라어, 힌디어에 대한 공정한 표현을 달성하기 위한 새로운 접근법인 Grapheme Pair Encoding (GPE)을 소개합니다.

- **Technical Details**: 우리는 기존의 Byte Pair Encoding (BPE) 알고리즘을 개선하여 Grapheme(그라페임) 정보를 통합하는 방법을 모색했습니다. 이 연구는 사전 토크나이제이션(pre-tokenization) 방법의 선택이 복잡한 스크립트 언어에 대한 표현성에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험 결과, 그라페임 기반의 문자 추출 방식이 바이트 단위 토크나이저보다 복잡한 스크립트에 대해 우수한 성능을 보이는 것으로 나타났습니다. 타밀어, 신할라어, 힌디어에서 이 접근법을 검증하였습니다.



### Multi-Document Grounded Multi-Turn Synthetic Dialog Generation (https://arxiv.org/abs/2409.11500)
- **What's New**: 이번 연구에서는 다수의 문서에 기반한 복수 턴의 합성 대화 생성(new synthetic dialog generation) 기법을 소개합니다. 이 기법은 1) Chain-of-Thought (CoT) 프롬프트를 사용하여 생성된 사용자 쿼리를 통해 전체 대화 흐름을 제어하고, 2) 사용자의 대화 턴마다 기초 문서를 업데이트하는 방식을 모방하여 다수의 문서에 기반한 대화 생성을 지원하며, 3) LLM-as-a-Judge를 적용하여 잘못된 답변이 포함된 쿼리를 필터링합니다.

- **Technical Details**: 이 연구는 synthetic dialog의 품질을 평가하기 위해 사용자-에이전트 간 대화를 인공지능 모델을 통해 생성하고, 이 데이터가 얼마나 다양하고 일관성 있으며 주로 정확한 답변을 포함하는지를 평가하는 데 초점을 맞추었습니다. 평가 기준은 쿼리의 응답 가능성과 응답의 정확성 간의 관계를 고려하여 설정되었습니다. 평가 과정에서 단일 절(clause)과 여러 절로 구성된 쿼리를 분석해야 합니다.

- **Performance Highlights**: 인간 평가 및 자동 평가를 통해 합성 대화에 최적화된 모델이 기존 인간 생성 훈련 데이터에 비해 네 개의 공개적인 복수 턴 문서 기반 벤치마크 테스트 세트에서 일관되게 성능을 발휘한다는 결과가 도출되었습니다.



### Augment, Drop & Swap: Improving Diversity in LLM Captions for Efficient Music-Text Representation Learning (https://arxiv.org/abs/2409.11498)
Comments:
          To appear in the Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: 본 연구에서는 오디오-텍스트 대비 모델의 디자인 선택이 음악-텍스트 표현의 품질에 미치는 영향을 정량적으로 분석합니다. 특히, 데이터 제약과 컴퓨팅 예산의 한계를 고려한 연구를 통해 데이터 큐레이션이 가장 중요한 영향을 미친다는 사실을 발견했습니다.

- **Technical Details**: 우리는 두 가지 주요 요소, 즉 아키텍처와 데이터에 중점을 두어 음악-텍스트 임베딩 모델의 디자인 공간을 탐구합니다. 이 논문에서는 Augmented View Dropout 및 TextSwap이라는 두 가지 새로운 기술을 도입해 훈련 중 텍스트 입력의 다양성과 설명성을 증가시킵니다.

- **Performance Highlights**: 실험을 통해, 제안한 기술들이 다양한 프리트레이닝 체계, 모델 아키텍처 및 하위 데이터 분포에서 성능을 향상시키는 데 효과적임을 보여줍니다. 이전 작업에 비해 새로운 최첨단 성능을 달성했습니다.



### Two Stage Segmentation of Cervical Tumors using PocketN (https://arxiv.org/abs/2409.11456)
- **What's New**: 이 논문에서는 자궁경부암 치료에서의 방사선치료 계획을 개선하기 위해 새로운 딥러닝 모델(PocketNet)을 제안합니다. 해당 모델은 MRI 영상을 통해 자궁, 질, 자궁 및 종양을 성공적으로 세분화할 수 있습니다.

- **Technical Details**: PocketNet은 T2 가중(Magnetic Resonance Imaging) 자기 공명 영상에서 자궁경부, 질, 자궁 및 종양을 분할하는 데 사용되는 딥러닝 모델입니다. 5-폴드 교차 검증(5-fold cross validation)을 통해 데이터에 대해 학습하였으며, 종양 분할의 평균 Dice-Sorensen 유사도 계수(DSC)는 70%를 초과하고, 장기 분할의 경우 80%를 초과했습니다.

- **Performance Highlights**: PocketNet은 대비 프로토콜의 변동성에 강한 내성을 보여주며, ROI(Region of Interest)의 신뢰할 수 있는 세분화를 제공합니다. 이는 방사선치료 계획의 품질과 일관성을 높이는 데 기여할 수 있습니다.



### Evaluation of pretrained language models on music understanding (https://arxiv.org/abs/2409.11449)
- **What's New**: 이 논문에서는 음악 정보 검색(Music Information Research, MIR) 애플리케이션에서의 대형 언어 모델(Large Language Model, LLM)의 음악 지식 평가에 초점을 맞추고 있습니다. 특히 대형 언어 모델이 가지는 프롬프트 민감성과 부정 표현 모델링의 한계 등을 지적하고, 이를 바탕으로 새로운 평가 방법론을 제안합니다.

- **Technical Details**: Audioset의 계층적 온톨로지를 사용하여, 각기 다른 장르와 악기에 대해 ‘앵커(anchor)’, ‘양성 긍정(label)’, ‘부정적(label)’ 형식의 트리플릿(triplet)을 생성하고, LLM의 상대적 유사성을 평가합니다. 이 방법은 음악적 지식을 정량화하기 위해 13,633개의 음악 장르와 37,640개의 음악 악기 트리플릿을 수집하였습니다.

- **Performance Highlights**: 리포트에 따르면, 모든 모델에서 높은 정확도가 나타났으나, 일관성이 부족하다는 결과가 나타났습니다. 이는 기존 LLM이 음악 관련 작업을 위한 적절한 사전 훈련 없이 사용될 경우, 성능 향상을 위해 조정이 필요함을 시사합니다.



### Volvo Discovery Challenge at ECML-PKDD 2024 (https://arxiv.org/abs/2409.11446)
Comments:
          ECML/PKDD 2024, Discovery Challenge

- **What's New**: 본 논문은 ECML-PKDD 2024 컨퍼런스 중 개최된 Volvo Discovery Challenge에 대한 개요를 제공합니다. 이 챌린지의 목표는 새로운 데이터셋을 사용하여 Volvo 트럭의 비공식 구성 요소의 고장 위험을 예측하는 것이었습니다. 참가자들은 두 개의 세대(gen1 및 gen2)에서 관찰된 데이터를 토대로 예측 모델을 개발하였고, 총 52명의 데이터 과학자가 791개의 제출물을 제출하였습니다.

- **Technical Details**: 챌린지는 Codabench 플랫폼에서 진행되었으며, 두 단계로 나뉘었습니다. 개발 단계에서는 하루에 5개의 예측을 제출할 수 있었고, 최종 단계에서는 총 3개의 예측만 제출할 수 있었습니다. 데이터셋은 train_gen1.csv, public_X_test.csv 및 variants.csv 여러 파일로 구성되어 있으며, 각 파일은 특정 형식으로 데이터를 포함하고 있습니다. F1-score는 본 대회의 주요 평가 지표로 사용되었습니다.

- **Performance Highlights**: 이 대회에서 상위 3위는 각자의 예측 방법론을 제시하였으며, 첫 번째 변수가 제출 시점에 대한 성능 향상을 보여줍니다. 첫 번째 우승자의 초기 점수는 0.53이었으나, 7주 동안 118개의 예측 제출을 통해 점수를 0.89로 개선했습니다. 대회 기간 동안 총 791개의 제출물 중 763개는 개발 단계에서 나왔고, 모델의 성능은 두 단계 사이에 약 2% 감소하는 수준이었습니다.



### Jailbreaking Large Language Models with Symbolic Mathematics (https://arxiv.org/abs/2409.11445)
- **What's New**: 이번 연구에서는 MathPrompt라는 새로운 jailbreaking 기술을 소개합니다. 이 기술은 LLM의 기호 수학(Symbolic Mathematics) 능력을 활용하여 안전 메커니즘을 우회합니다. 이는 유해한 자연어 프롬프트를 수학 문제로 인코딩하여 현재 AI 안전 조치의 중요한 취약점을 드러냅니다.

- **Technical Details**: MathPrompt는 두 단계로 구성됩니다. 첫째, 유해한 자연어 프롬프트를 기호 수학 문제로 변환하고, 둘째, 이러한 수학적으로 인코딩된 프롬프트를 목표 LLM에 제시합니다. 실험 결과, 13개의 최첨단 LLM에서 평균 73.6%의 공격 성공률을 기록하였습니다.

- **Performance Highlights**: 현재 안전 훈련 메커니즘은 수학적으로 인코딩된 입력에 일반화되지 않는다는 것을 강조합니다. LLM이 기호 수학을 이해함에도 불구하고, 기존의 안전 조치들은 이와 같은 수학적 표현을 잘 처리하지 못하고 있습니다.



### A Green Multi-Attribute Client Selection for Over-The-Air Federated Learning: A Grey-Wolf-Optimizer Approach (https://arxiv.org/abs/2409.11442)
- **What's New**: 이 논문에서는 Multi-Attribute client selection framework을 통해 Federated Learning (FL)에서 클라이언트를 전략적으로 선택하고, Grey Wolf Optimizer (GWO)를 활용하여 OTA-FL 절차를 최적화하는 방법을 제안합니다. 이 방식은 정확성, 에너지, 지연, 신뢰성, 공정성과 같은 여러 제약 조건을 고려하여 참여하는 장치의 수를 조정합니다.

- **Technical Details**: OTA-FL(Over-the-Air Federated Learning)에서는 장치 간의 직접 연결 없이 무선 통신을 사용하여 모델 업데이트를 전송합니다. Grey Wolf Optimizer (GWO)는 Grey Wolf의 사회적 행동을 기반으로 한 메타휴리스틱 최적화 알고리즘으로, 클라이언트 선택에 적용되어 모델 손실 최소화, 수렴 시간 단축 및 에너지 효율을 극대화합니다.

- **Performance Highlights**: GWO 기반 클라이언트 선택 접근 방식은 기존의 최첨단 방법과 비교하여 주목할 만한 모델 손실 감소, 수렴 시간 가속화 및 에너지 효율을 달성했습니다. 실험 결과에서도 공정성과 신뢰성을 유지하면서 성능이 개선된 것을 보여주었습니다.



### MARCA: Mamba Accelerator with ReConfigurable Architectur (https://arxiv.org/abs/2409.11440)
Comments:
          9 pages, 10 figures, accepted by ICCAD 2024. arXiv admin note: text overlap with arXiv:2001.02514 by other authors

- **What's New**: MARCA는 재구성 가능한 아키텍처를 가진 새 Mamba 가속기로, 선형 및 요소별 연산을 위한 개선된 PE 배열 아키텍처, 재사용 가능한 비선형 함수 유닛 및 버퍼 관리 전략을 제안합니다.

- **Technical Details**: MARCA는 (1) 선형 연산을 위한 PE 배열과 연결된 축소 트리를 활성화하고, 요소별 연산에서는 축소 트리를 비활성화하여 결과를 우회하는 축소 대체 PE 배열 아키텍처를 제공합니다. (2) 빠른 편향 지수 알고리즘을 사용하여 지수 함수를 요소별 연산과 이동 연산으로 분해하고, 조각별 근사 알고리즘으로 활성화 함수(SiLU)를 범위 감지 및 요소별 연산으로 분해하여 비선형 함수를 정확도 손실 없이 실행합니다. (3) 선형 연산 내 데이터 공유를 극대화하는 내부 연산 버퍼 관리 전략과 요소별 연산 간 데이터 공유를 극대화하는 외부 연산 전략을 제안합니다.

- **Performance Highlights**: MARCA는 Intel Xeon 8358P CPU 및 NVIDIA Tesla A100 GPU 대비 최대 463.22×/11.66× 속도 향상과 최대 9761.42×/242.52× 에너지 효율성을 달성합니다.



### Machine listening in a neonatal intensive care un (https://arxiv.org/abs/2409.11439)
- **What's New**: 이 논문은 병원 환경에서 흔하게 발생하는 소음원의 탐지라는 새로운 접근방법을 제시합니다. 기존의 음파 녹음 대신, 음향 센서를 통해 제3옥타브 스펙트로그램을 실시간으로 계산하여 프라이버시 보호 문제를 해결하려 했습니다.

- **Technical Details**: 논문에서는 엣지 컴퓨팅(edge computing)과 클라우드 컴퓨팅(cloud computing)을 결합하여 소리 탐지 문제를 해결합니다. 프리트레인(pretrained) 오디오 뉴럴 네트워크(neural network)인 PANN를 활용하여 스펙트럴 트랜스코딩(spectral transcoding)과 레이블 스페이스 어댑테이션(label space adaptation)을 통해 샘플 효율적인 머신 러닝(machine learning)을 구현하였습니다.

- **Performance Highlights**: 신생아 집중 치료실(NICU)에서의 작은 규모의 연구에서 탐지된 이벤트의 시계열(time series)이 전자 배지를 통해 측정된 데이터와 일치함을 확인했습니다. 이로써 병원 병동에서 다성(multi-voiced) 기계 청취의 가능성을 보여주며, 설계 단계부터 프라이버시를 보장합니다.



### Analysis of flexible traffic control method in SDN (https://arxiv.org/abs/2409.11436)
- **What's New**: 이 논문은 SDN(Software Defined Networking) 네트워크에서 유연한 제어 방법을 분석하고, SDN 컨트롤러의 성능을 지능적으로 적응시키는 자가 개발된 솔루션을 제안합니다.

- **Technical Details**: 이 연구는 현대적인 기계 학습(Machine Learning) 방법 중 하나인 강화 학습(Reinforcement Learning)을 사용하여 네트워크가 동적으로 변화하는 환경에서 스스로 선택한 사항에 기반하여 학습하며 자율적인 결정을 내리게 합니다.

- **Performance Highlights**: 제안된 솔루션은 네트워크 성능을 향상시킬 뿐만 아니라 유연성과 실시간 적응성(flexible traffic control)을 제공하여 네트워크 관리의 효율성과 적응성을 증가시키는 목표를 가지고 있습니다.



### A hybrid solution for 2-UAV RAN slicing (https://arxiv.org/abs/2409.11432)
Comments:
          9 pages, 11 figures

- **What's New**: 이 논문은 드론을 이용하여 사용자에게 인터넷을 분배하는 방법을 제안하며, 특히 5G 네트워크에서의 최적 배치 및 대역폭 분할 문제를 해결하기 위한 하이브리드 AI 최적화 접근법을 다루고 있습니다.

- **Technical Details**: 5G New Radio (NR) 기술을 기반으로 하여, 네트워크 슬라이싱(network slicing)을 통해 세 가지 사용자 클래스를 위한 맞춤형 서비스를 제공합니다. 각 드론은 eMBB(강화된 모바일 브로드밴드), mMTC(대규모 기계 통신), URLLC(초신뢰 저지연 통신) 각각에 대역폭을 할당해야 합니다. 최적화 문제는 수학적 모델(예: SINR 계산)을 통해 표현됩니다.

- **Performance Highlights**: 하이브리드 솔루션을 통해 AI만 사용하는 방법에 비해 더 나은 성과를 달성하였지만 계산 시간이 다소 증가하는 결과를 보여주었습니다. 사용자 만족도를 최대화하기 위한 신뢰할 수 있는 방법과 비교 분석을 진행하였습니다.



### Federated Learning with Quantum Computing and Fully Homomorphic Encryption: A Novel Computing Paradigm Shift in Privacy-Preserving ML (https://arxiv.org/abs/2409.11430)
- **What's New**: 이번 논문에서는 데이터 개인 정보 보호 및 정보 보안 문제를 해결하기 위해 Federated Learning(연합 학습)과 Fully Homomorphic Encryption(완전 동형 암호화, FHE) 접근 방식을 결합한 새로운 방안을 제시합니다.

- **Technical Details**: 연구는 Federated Learning과 FHE를 통합하여 클래식 및 양자(quantum) 층을 포함하는 Neural Network(신경망) 아키텍처를 구현하고, 이를 통해 개인 정보를 보호하는 기계 학습 시스템을 설계합니다.

- **Performance Highlights**: 본 연구는 양자 안전(quantum-safe) 암호 시스템을 활용하여 개인 데이터 유출 없이 모델 지식을 공유할 수 있는 가능성을 보이며, 성능 저하를 최소화할 수 있는 새로운 컴퓨팅 패러다임의 가능성을 탐구합니다.



### Towards Opinion Shaping: A Deep Reinforcement Learning Approach in Bot-User Interactions (https://arxiv.org/abs/2409.11426)
Comments:
          5 pages, 3 figures, 2 tables

- **What's New**: 이 연구는 Stochastic Bounded Confidence Model (SBCM)을 통해 사용자-봇 상호작용의 간섭이 사회 네트워크 알고리즘에 미치는 영향을 조사합니다. DDPG(Deep Deterministic Policy Gradient) 알고리즘을 이용하여 의견 형성을 위한 두 가지 접근법, 즉 봇의 배치와 균형 잡힌 예산 하의 타겟 광고를 제안합니다.

- **Technical Details**: 이 연구는 SBCM의 사용자-봇 상호작용 시나리오를 통합하여, DDPG 알고리즘을 통해 타겟 광고를 한정된 예산 내에서 수행하는 방법을 모색합니다. DDPG는 연속 행동 공간을 처리하는 최적화된 정책 학습을 가능하게 하며, 복잡한 고차원 환경에서 효과적으로 작동합니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식이 효과적인 의견 형성을 이끌어낼 수 있음을 입증하였으며, 광고 자원 배치의 새로운 가능성을 제시합니다.



### The Unseen AI Disruptions for Power Grids: LLM-Induced Transients (https://arxiv.org/abs/2409.11416)
Comments:
          21 pages, 18 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 AI 관련 데이터 센터로의 다중 백억 달러 투자를 촉진하였습니다. 하지만, AI 모델의 효율성 못지않게 중요한 전력 소비의 동적 행동이 간과되고 있는 문제가 제기되었습니다.

- **Technical Details**: AI 인프라는 초저관성(ultra-low inertia)과 급격한 전력 피크 및 디프 동작을 특징으로 하며, 전력 소비는 수백 와트에서 메가와트, 기가와트까지 이릅니다. 본 연구는 AI 전력 소비 규모를 분석하고, AI 부하 행동을 나타내는 수학적 모델을 개발했습니다.

- **Performance Highlights**: AI 산업의 에너지 수요는 10개월마다 두 배로 증가하고 있으며, 2028년까지 AI 관련 전력 소비는 14 GW에서 18.7 GW에 이를 것으로 예측됩니다. AI의 변동적으로 불안정한 행동이 전력망 안정성에 미치는 영향이 강조되었습니다.



### RTLRewriter: Methodologies for Large Models aided RTL Code Optimization (https://arxiv.org/abs/2409.11414)
Comments:
          ICCAD2024

- **What's New**: 이 논문은 RTL (Register Transfer Level) 코드 최적화를 위한 혁신적인 프레임워크인 RTLRewriter를 소개합니다. 기존의 수작업 방식과 컴파일러 기반 방법의 한계를 극복하기 위해 대규모 모델을 활용하여 RTL 코드를 최적화합니다. 또한, 복잡한 설계를 효과적으로 관리하는 회로 분할 파이프라인과 시각적 다이어그램 정보를 포함한 다중 모드(program) 분석을 제안합니다.

- **Technical Details**: RTLRewriter 프레임워크는 다음과 같은 주요 구성 요소로 이루어져 있습니다: 1) 회로 분할(circuit partition): 전체 회로를 작은 부분으로 나누어 빠른 합성을 가능하게 함; 2) 다중 모드 프로그램 분석(multi-modal program analysis): 시맨틱 정보를 강화하여 최적화 패턴 생성; 3) 문서 활용(documentation utilization): 관련 문서와 노트를 검색하는 전용 검색 엔진을 통해 모델 생성을 지원; 4) 비용 인식 몬테카를로 트리 탐색(C-MCTS): 최적의 재작성 전략 결정 및 다양한 콘텐츠 관리; 5) 검증 비용 절감: 프로그램 분석을 통해 적절한 검증 솔버를 선택하여 검증 비용을 줄입니다.

- **Performance Highlights**: RTLRewriter는 Yosys 및 E-graph와 같은 기존 컴파일러 대비 상당한 최적화 개선을 보여줍니다. 새로운 두 개의 벤치마크인 Large Rewriter Benchmark와 Small Rewriter Benchmark를 통해 다양한 복잡성과 시나리오를 다루며 실제 산업 환경에서 이러한 프레임워크가 중요한 역할을 할 수 있음을 증명합니다.



### CyberNFTs: Conceptualizing a decentralized and reward-driven intrusion detection system with ML (https://arxiv.org/abs/2409.11409)
Comments:
          9 pages, 6 figures, 1 table, 1 algorithm, 1 listing, journal article

- **What's New**: 이 논문은 분산형 협동 침입 탐지 네트워크(CIDN)를 구현하기 위한 새로운 개념적 접근 방식을 제안합니다. Web3 기술과 정보 보안 간의 시너지를 분석하며, 블록체인 개념, 사이버 NFT 보상, 기계 학습 알고리즘 및 게시/구독 구조를 포함하는 모델을 제안합니다.

- **Technical Details**: 제안된 시스템은 네트워크 트래픽을 모니터링하고, 기계 학습 모델을 통해 분류하며, 블록체인에 분산 저장하는 기능을 갖추고 있습니다. 또한, 침입 탐지자가 고유한 NFT로 보상을 받는 구조를 채택하고 있습니다. 이 접근 방식은 기존의 침입 탐지 시스템과 비교하여 분산화, 독립 작동, 점진적 개선의 특성을 갖추고 있습니다.

- **Performance Highlights**: 기계 학습 알고리즘을 통합하여 자동화된 시스템의 탐지 및 예방 한계를 명확하게 드러내고 있으며, 연구 결과는 CIDN의 운영과 기존 시스템 간의 차이점을 도출해낼 수 있는 가능성을 제시합니다.



### Towards Signal Processing In Large Language Models (https://arxiv.org/abs/2406.10254)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM) 내부에 신호 처리(signal processing)를 적용하는 새로운 아이디어를 소개합니다. 생성 AI의 폭발적 성장 가운데, 이 연구는 신호 처리와 대형 언어 모델 간의 연결 고리를 마련하는 데 기여할 것입니다.

- **Technical Details**: 우리는 고전적인 푸리에 변환(Fourier Transform)과 대형 언어 모델의 중간 활성화 신호에 대한 푸리에 변환 유사한 학습 가능한 시간-주파수 표현(time-frequency representation) 간의 유사성을 그립니다. 모든 활성화 신호를 시간-주파수 표현으로 분해하고, 이를 학습하여 다음 토큰을 예측합니다. 우리의 접근 방식은 추가적인 매개변수를 최소화하면서도 성능을 크게 증가시킵니다.

- **Performance Highlights**: 이 연구의 결과, GPT와 유사한 아키텍처에서 더 빠른 수렴을 달성하였고, 기존 방법에 비해 성능이 현저히 향상되었습니다. 이를 통해 대형 언어 모델의 신경 아키텍처 내에서의 신호 처리 탐구를 위한 알고리즘 개발에 기여할 것으로 기대합니다.



### ReflectDiffu:Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework (https://arxiv.org/abs/2409.10289)
- **What's New**: 이 논문에서는 감정적 응답 생성을 위한 경량화된 종합 프레임워크인 ReflectDiffu를 제안합니다. 기존 연구는 감정과 의도 간의 복잡한 상호작용을 간과하거나, 큰 언어 모델(LLMs)로 인해 계산 부담이 큽니다.

- **Technical Details**: ReflectDiffu는 감정 전염(emotion contagion)을 통합하여 감정 표현(emotional expressiveness)을 향상시키고, 감정 추론 마스크(emotion-reasoning mask)를 사용하여 중요한 감정 요소를 강조합니다. 또한 강화 학습(reinforcement learning) 내에서 의도 모방(intent mimicry)을 포함시켜 확산(diffusion) 중 개선을 이룹니다. Exploring-Sampling-Correcting 메커니즘을 활용하여 감정적 의사결정을 정밀한 의도 행동으로 전환합니다.

- **Performance Highlights**: ReflectDiffu는 관련성(relevance), 제어 가능성(controllability), 정보성(informativeness) 면에서 기존 모델들을 능가하며, 자동 평가 및 인간 평가 모두에서 최첨단 성과를 달성했습니다.



### Autoregressive + Chain of Thought $\simeq$ Recurrent: Recurrence's Role in Language Models' Computability and a Revisit of Recurrent Transformer (https://arxiv.org/abs/2409.09239)
- **What's New**: 이 논문은 기존 Transformer 아키텍처의 한계를 극복하기 위해, Chain of Thought (CoT) 프롬프트를 활용하여 반복 구조의 중요성과 자가 회귀(autoregression)의 역할을 심층적으로 분석합니다.

- **Technical Details**: 이 연구는 신경망의 반복 구조가 추론 능력과 계산 가능성에 미치는 영향을 탐구하며, CoT 접근법이 반복 계산을 어떻게 모방할 수 있는지 설명합니다. 또한, 최근의 반복 기반 Transformer 모델 설계를 재조명하고, 'recurrence-completeness'의 개념을 통해 이들의 계산 능력을 분석합니다.

- **Performance Highlights**: CoT 접근을 통해 Transformer 모델의 성능을 높일 수 있음을 입증하며, 다양한 과제를 해결하는 데 있어 CoT의 기여를 효과적으로 강조합니다.




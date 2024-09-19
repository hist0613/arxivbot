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




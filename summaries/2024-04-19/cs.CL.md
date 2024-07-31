### Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language  Models (https://arxiv.org/abs/2404.12387)
- **What's New**: Reka 사의 새로운 다중모드 언어 모델 시리즈인 Reka Core, Flash, Edge는 텍스트, 이미지, 비디오, 오디오 입력을 처리하고 추론할 수 있는 능력을 갖추었습니다. 이들은 다양한 벤치마크에서 상태 기술 기준을 뛰어넘는 성능을 보여주고 있으며, Reka Core는 자동 평가 및 블라인드 인간 평가에서 최고의 모델들과 경쟁적인 성능을 보여줍니다.

- **Technical Details**: Reka Edge와 Flash는 각각 7B와 21B의 매개변수를 가진 밀집 모델입니다. 이 모델들은 엄선된 데이터 세트를 사용하여 훈련되었으며, 다양한 언어와 형식의 데이터를 처리할 수 있습니다. Core 및 Flash 모델은 128K 컨텍스트 길이를 지원하며, 대규모 문서와 장문의 정보 처리에 특화된 훈련을 받았습니다.

- **Performance Highlights**: Reka Core는 이미지 질문 응답 벤치마크 (MMMU, VQAv2)에서 GPT4-V와 경쟁적이며, 다중모드 채팅에서는 Claude 3 Opus를 능가합니다. 비디오 질문 응답에서는 Gemini Ultra를 능가하며, 텍스트 벤치마크에서는 GPT4-0613보다 더 높은 인간 평가 점수를 받았습니다. 또한, Edge 모델은 같은 계산 클래스에서 최고의 모델들을 능가하는 성능을 보여주었고, Flash 모델은 GPT-3.5 Turbo 및 기타 대형 모델들을 능가했습니다.



### When LLMs are Unfit Use FastFit: Fast and Effective Text Classification  with Many Classes (https://arxiv.org/abs/2404.12365)
Comments: Accepted to NAACL

- **What's New**: 새롭게 제시된 FastFit은 수많은 의미상 유사한 클래스가 있는 시나리오에서 빠르고 정확한 few-shot classification (최소샘플 분류)을 제공하는 Python 패키지입니다. FastFit은 batch contrastive learning (배치 대조 학습)과 token-level similarity score (토큰 수준 유사성 점수)를 통합하는 새로운 접근 방식을 사용합니다.

- **Technical Details**: FastFit은 다양한 실험을 통해 기존의 few-shot learning 패키지들보다 훨씬 빠른 속도로 훈련이 가능하며, SetFit, Transformers, 및 대규모 언어 모델의 few-shot prompting에 비해 우수한 성능을 보여주었습니다. 이는 FastFit이 token-level 텍스트 유사성 측정과 텍스트 증강 기술을 활용하여 정밀한 정보를 활용하기 때문입니다.

- **Performance Highlights**: FastFit은 기존 방법들에 비해 훈련 속도에서 3-20배 개선을 보였으며, 단 몇 초 내에 훈련을 완료할 수 있습니다. 또한, 영어와 다국어 데이터셋 모두에서 우수한 다중 클래스 분류 성능을 입증하였습니다.



### Large Language Models in Targeted Sentiment Analysis (https://arxiv.org/abs/2404.12342)
Comments: Fine-tuned Flan-T5-xl outperforms the top #1 results of transformer-based classifier in RuSentNE-2023 competition, to appear in Lobachevskii Journal of Mathematics No.8/2024 proceedings

- **What's New**: 이 논문에서는 러시아 뉴스 기사에서 명명된 엔티티(Entity)에 대한 감정을 추출하기 위해 디코더 기반 생성 트랜스포머(Generative Transformers)의 사용을 조사했습니다. Instruction-tuned 대규모 언어 모델(LLMs, Large Language Models)의 감정 분석 능력을 연구하였으며, RuSentNE-2023 데이터셋을 사용했습니다.

- **Technical Details**: 실험은 두 그룹으로 나뉘었습니다. 첫 번째 그룹은 폐쇄형 및 개방형 투명성을 가진 LLM의 제로-샷(Zero-Shot) 능력 평가에 초점을 맞췄습니다. 두 번째 그룹은 '사고 연쇄'(Chain of Thought, CoT)와 세 단계 추론 프레임워크(Three-hop Reasoning Framework, THoR)를 사용하여 Flan-T5를 미세 조정하는 것을 다룹니다.

- **Performance Highlights**: Zero-shot 접근 방식의 결과는 baseline으로 미세 조정된 인코더 기반 트랜스포머(BERT-base)가 달성한 결과와 유사했습니다. THoR를 사용한 미세 조정된 Flan-T5 모델은 Zero-shot 실험의 결과에 비해 최소 5% 향상된 추론 능력을 보였습니다. RuSentNE-2023에 대한 감정 분석에서 가장 좋은 결과는 미세 조정된 Flan-T5-xl에서 나타났으며, 이는 이전 최고의 트랜스포머 기반 분류기(Classifiers)의 결과를 뛰어넘었습니다.



### Reuse Your Rewards: Reward Model Transfer for Zero-Shot Cross-Lingual  Alignmen (https://arxiv.org/abs/2404.12318)
- **What's New**: 언어 모델(LM: Language Models)을 다국어 환경으로 확장하는 데 필수적인 인간이 주석을 단 선호도 데이터를 기반으로 하는 언어 모델의 정렬(alignment)은 매우 중요한 단계입니다. 이 연구에서는 단일 소스 언어에서 선호도 데이터를 훈련시킨 보상 모델(reward model)을 이용하여 다른 목표 언어(target languages)에 대해 제로-샷(zero-shot) 교차 언어 정렬(cross-lingual alignment) 방법을 평가합니다.

- **Technical Details**: 제로-샷 교차 언어 정렬 방법을 이용해, 한 소스 언어에서 훈련된 보상 모델을 다양한 목표 언어에 직접 적용하였습니다. 이 방법은 요약과 개방형 대화 생성(open-ended dialog generation) 작업에서 폭넓은 평가 설정하에 일관되게 성공적임을 보여주었습니다. 특히, 언어 특정 데이터(language-specific data)가 없는 경우, 감독된 미세 조정(supervised finetuning)을 위한 최선의 방법도 도출하였습니다.

- **Performance Highlights**: 인간 평가를 포함한 포괄적 평가 설정에서, 교차 언어로 정렬된 모델들이 정렬되지 않은 모델들보다 평가 인스턴스의 70% 이상에서 선호되었습니다. 또한 때때로 다른 언어의 보상 모델이 같은 언어의 것보다 더 잘 정렬된 모델을 제공하기도 하였습니다.



### Simultaneous Interpretation Corpus Construction by Large Language Models  in Distant Language Pair (https://arxiv.org/abs/2404.12299)
Comments: 23 pages, 9 figures

- **What's New**: 이 논문은 기존의 음성 번역(Speech Translation, ST) 코퍼스를 동시 통역(Simultaneous Interpretation, SI) 스타일 데이터로 변환하는 새로운 방법을 제안합니다. 이 방법은 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 원본의 단어 순서를 유지하고 전체 소스 콘텐츠를 보존합니다. 이를 통해 생성된 LLM-SI-Corpus는 기존 SI 및 SiMT(Simultaneous Machine Translation) 코퍼스보다 더 자연스럽고 충실한 번역을 제공합니다.

- **Technical Details**: LLM을 활용하여 ST 코퍼스를 SI 스타일 데이터로 변환하는 과정은 Chunk-Wise Monotonic Translation (CWMT) 지침을 따르는 것을 포함합니다. 이는 입력을 청크로 나누고 이를 순차적으로 번역하여 최소한의 지연을 유지하면서 번역 품질을 보장하는 전략입니다. 또한, SiMT 모델을 LLM-SI-Corpus로 미세 조정할 때 기존 코퍼스로 훈련된 모델과 동일한 품질을 유지하면서 지연 시간을 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: LLM-SI-Corpus를 사용하여 훈련된 SiMT 모델은 텍스트 간 번역(Text-to-Text)과 음성 간 번역(Speech-to-Text) 설정에서 기존 SI 코퍼스에 비해 우수한 성능을 보입니다. 이 모델은 의미론적 측면에서 향상된 번역 품질을 제공하며, 기존 오프라인 데이터셋으로 훈련된 모델과 동일한 품질 수준을 유지하면서 지연 시간을 줄입니다.



### Augmenting emotion features in irony detection with Large language  modeling (https://arxiv.org/abs/2404.12291)
Comments: 11 pages, 3 tables, 2 figures. Submitted to the 25th Chinese Lexical Semantics Workshop

- **What's New**: 이 연구는 아이러니 감지를 위해 Large Language Models (LLMs)와 프롬프트 기반 학습을 적용한 새로운 방법을 소개합니다. 이 방법은 텍스트 감정 중심의 보강을 통해 전통적인 아이러니 감지 기술의 한계를 극복하고자 합니다. 기존 방법들은 정적인 언어적 특성과 사전 정의된 지식 기반에 의존하며, 아이러니에 필수적인 섬세한 감정적 차원을 종종 간과합니다.

- **Technical Details**: 본 방법론은 세 가지 벤치마크 NLP 모델(BERT, T5, GPT-2)에 감정적 단서를 통합하여 아이러니 감지 과정을 보강하는 것을 특징으로 합니다. 이러한 감정 단서는 LLMs를 통해 보강됩니다. 이 연구는 SemEval-2018 Task 3 데이터셋을 사용하여 방법을 평가했으며, 아이러니 감지 기능에서 상당한 향상을 관찰하였습니다.

- **Performance Highlights**: 통합된 감정적 단서와 텍스트 보강 기법을 활용함으로써 BERT, T5, GPT-2 모델들은 아이러니를 훨씬 정확하게 감지할 수 있었습니다. 이는 SemEval-2018 Task 3 데이터셋에서 측정된 결과에 의해 입증되었습니다.



### Resilience through Scene Context in Visual Referring Expression  Generation (https://arxiv.org/abs/2404.12289)
- **What's New**: 이 논문은 장면 맥락(scene context)이 이미지에서 객체를 참조하는 표현(Referring Expression Generation, REG) 생성에 어떻게 도움이 되는지에 대한 새로운 관점을 제시합니다. 기존 연구가 대상과 비슷한 교란 요소(distractor)에 초점을 맞춘 데 반해, 이 연구는 맥락 정보가 REG 모델을 더욱 강력하게 만들고, 객체 설명, 특히 객체 유형의 생성을 용이하게 한다는 가설을 세웁니다.

- **Technical Details**: 연구팀은 Transformer 기반 REG 모델을 사용하여, 다양한 정도로 노이즈로 인해 모호하게 된 대상 표현(target representations)에 대해 훈련하고 테스트했습니다. 이 모델들은 다른 맥락 표현(context representations)을 제공받고, 일반적인 품질 지표(quality metrics) 및 참조 유형을 판단하는 데 초점을 맞춘 인간 평가를 통해 성능을 비교했습니다. 신경 생성 모델(neural generation models)은 저수준의 시각적 표현에서 객체의 속성을 추출해야 합니다.

- **Performance Highlights**: 실험 결과, 심지어 단순한 장면 맥락조차도 모델이 타겟 표현의 교란에 대해 놀랍도록 강건함을 보여주었습니다. 그 결과, 대상에 대한 시각적 정보가 전혀 없는 경우에도 참조 유형을 식별할 수 있었습니다. 이는 장면 맥락이 시각적 REG에서의 역할에 대한 새로운 시각을 제공합니다.



### Enhancing Embedding Performance through Large Language Model-based Text  Enrichment and Rewriting (https://arxiv.org/abs/2404.12283)
- **What's New**: 이 논문은 언어 모델(ChatGPT 3.5)을 사용하여 입력 텍스트를 풍부하게 하고 재작성함으로써 embedding 성능을 향상시키는 새로운 접근 방식을 제안합니다. 기존의 embedding 모델들이 직면한 한계, 예를 들어 제한된 어휘력, 문맥의 부재, 문법 오류 등을 해결하기 위해 대규모 언어 모델의 지식과 문맥 이해를 활용합니다.

- **Technical Details**: 이 방법은 ChatGPT 3.5를 활용하여 입력 텍스트에 추가적인 문맥을 제공하고, 철자 및 문법 오류를 정정하며, 도메인 특정 용어를 표준화합니다. 또한, 다의어의 의도된 의미를 명확히 하고, 약어를 전체 형태로 확장하여 입력 텍스트의 품질과 정보성을 개선합니다. 사용된 embedding 모델은 OpenAI가 개발한 text-embedding-3-large 모델로, 이는 다양한 NLP 작업에서 강력한 성능을 보여주었습니다.

- **Performance Highlights**: 특히 TwitterSemEval 2015 데이터셋에서 이전 최고 점수였던 81.52보다 향상된 85.34라는 점수를 달성하여, Massive Text Embedding Benchmark(MTEB) 리더보드에서 상당한 개선을 보였습니다. 그러나 Banking77Classification과 Amazon Counter-factual Classification 데이터셋에서는 덜 인상적인 성능을 보여, 도메인 특화 특성의 고려가 중요함을 강조했습니다.



### Advancing the Robustness of Large Language Models through Self-Denoised  Smoothing (https://arxiv.org/abs/2404.12274)
Comments: Accepted by NAACL 2024. Jiabao, Bairu, Zhen, Guanhua contributed equally. This is an updated version of the paper: arXiv:2307.07171

- **What's New**: LLM (Large Language Models)의 취약성 문제를 해결하려는 새로운 접근 방법으로, 우리는 자체 제거(smoothing) 기법을 사용하여 잡음이 있는 입력을 처리하는 SelfDenoise 방법을 제안합니다. 이 방법은 다운스트림 작업 및 인간 정렬(jailbreak attacks)을 포함한 적대적 공격에 대한 방어력을 향상시키기 위해 개발되었습니다.

- **Technical Details**: SelfDenoise는 LLM에 무작위로 마스킹된 단어를 포함한 잡음이 있는 입력을 제공하고, LLM 스스로 이를 재구성하여 고쳐진 문장을 생성하게 합니다. 이후, 이 고쳐진 문장들을 다시 모델로 입력하여 작업 수행을 요청합니다. 이 과정은 컴퓨터 비전에서 볼 수 있는 별도의 denoising 모듈이 필요 없어 효율적입니다. 이 방법은 LLM의 다작업 수행 능력을 활용하여, 자체적인 denoising을 통한 강화된 로버스트(robustness)를 제공합니다.

- **Performance Highlights**: 실험 결과, SelfDenoise는 기존 방법들보다 우수한 경험적(empirical) 및 인증된( certified) 로버스트성을 보여주며, 다운스트림 작업과 인간 값과의 정렬에서 더 자연스럽고 안전한 결과를 생성합니다. 이 방법은 추가적인 denoiser 훈련의 비용 없이 LLM의 로버스트성을 높이는 데 효과적입니다.



### Toward Self-Improvement of LLMs via Imagination, Searching, and  Criticizing (https://arxiv.org/abs/2404.12253)
- **What's New**: AlphaLLM은 LLM의 자가 개선을 위해 모방 검색(imagination-searching) 및 비판 평가(criticizing) 프레임워크를 융합한 새로운 모델입니다. 이는 몬테카를로 트리 검색(Monte Carlo Tree Search, MCTS)과 결합되어, 추가적인 주석 없이 LLM의 능력을 향상시킬 수 있는 자체 개선 루프(self-improving loop)를 설정합니다.

- **Technical Details**: AlphaLLM은 세 가지 주요 구성 요소를 포함합니다: 1) 데이터 부족 문제를 완화하기 위한 상상력 컴포넌트(imagination component)로 프롬프트를 합성, 2) 언어 과제에 적합한 MCTS 접근 방식인 η-MCTS를 통한 효율적인 검색을 제안, 3) 정확한 피드백을 위한 비평가 모델(trio of critic models) 구성. η-MCTS는 마르코프 결정 과정(Markov Decision Process, MDP)을 통해 텍스트 생성 과정을 다양한 하위 과제로 나누어 처리 효율을 높입니다.

- **Performance Highlights**: 실험 결과, AlphaLLM은 수학적 추론 과제에서 뛰어난 성능을 보였으며, LLaMA-2 70b를 사용하여 GSM8K에서 57.8에서 92.0으로, MATH에서 20.7에서 51.0으로 성능이 향상되었습니다. 이는 GPT-4와 비슷한 수준입니다. 이러한 결과는 AlphaLLM이 LLM의 성능을 자체적으로 향상시킬 수 있는 효과적인 루프를 형성함을 시사합니다.



### CMNEE: A Large-Scale Document-Level Event Extraction Dataset based on  Open-Source Chinese Military News (https://arxiv.org/abs/2404.12242)
Comments: 13 pages, 7 figures, accepted to LREC-COLING 2024

- **What's New**: CMNEE (Chinese Military News Event Extraction) 데이터셋이 새롭게 제안되었습니다. 이 데이터셋은 17,000개의 문서와 29,223개의 사건을 포함하고 있으며, 군사 분야의 이벤트 추출에 특화된 것입니다. CMNEE는 높은 품질의 데이터 제공을 목표로 하며, 군사 뉴스에서 사건의 트리거(trigger), 인자(arguments) 및 해당 역할을 추출하기 위한 것입니다.

- **Technical Details**: 이 데이터셋은 자동화된 이벤트 유형 예측 및 두 단계의 다중 회전 어노테이션 전략(annotation strategy)을 사용하여 구축되었습니다. 데이터는 군사 관련 웹사이트에서 크롤링되며, 사전 정의된 트리거 사전(trigger dictionary)과 도메인 전문가의 지식을 통해 미리 레이블링되었습니다. CMNEE에는 8가지 이벤트 유형과 11개의 인자 역할 유형이 포함되어 있으며, 이는 고도로 구조화된 이벤트 스키마(event schema)를 통해 정의됩니다.

- **Performance Highlights**: CMNEE 데이터셋을 사용하여 여러 최신 이벤트 추출 모델을 재현하고 체계적으로 평가했습니다. 그러나 실험 결과는 다른 도메인 데이터셋에 비해 낮게 나타났으며, 군사 분야에서의 이벤트 추출이 고유의 도전을 안고 있음을 보여 줍니다. 이는 향후 연구와 노력이 더 필요함을 시사합니다.



### Introducing v0.5 of the AI Safety Benchmark from MLCommons (https://arxiv.org/abs/2404.12241)
- **What's New**: 이 논문에서는 MLCommons AI Safety Working Group이 만든 AI Safety Benchmark v0.5를 소개합니다. AI Safety Benchmark는 채팅-조율 언어 모델(chat-tuned language models)을 사용하는 AI 시스템의 안전성 위험을 평가하기 위해 설계되었습니다. v0.5 버전은 영어로 일반 목적의 도우미와 대화하는 어른이라는 한 가지 유스 케이스(use case)와 제한된 페르소나(personas) 집합(일반 사용자, 악의적 사용자, 취약한 사용자)만을 다룹니다.

- **Technical Details**: 새로운 13개의 위험 범주(hazard categories) 분류 체계를 도입하고, 이 중 7개 범주에 대한 테스트를 포함하였습니다. 벤치마크 구성 요소는 사용 사례, 테스트 대상 시스템(types of systems under test, SUTs), 언어 및 맥락, 페르소나, 테스트, 테스트 항목들로 구성되어 있습니다. 총 43,090개의 테스트 항목이 템플릿을 이용해 생성되었습니다. 이와 함께 AI 시스템을 평가하는 데 사용할 수 있는 개방적이고 다운로드 가능한 도구인 ModelBench 플랫폼을 제공합니다.

- **Performance Highlights**: v0.5 벤치마크는 현재 AI 시스템의 안전성 평가에 사용되어서는 안 되며, 주로 접근 방식을 설명하고 피드백을 요청하기 위해 공개되었습니다. 다수의 공개적으로 사용 가능한 채팅-조율 언어 모델들의 성능을 벤치마킹한 예시 평가 보고서를 포함하며, 모든 모델은 익명으로 처리되었습니다. AI Safety Benchmark v1.0은 2024년 말에 출시될 예정이며, AI 시스템의 안전성에 대한 의미 있는 통찰력을 제공할 것입니다.



### Length Generalization of Causal Transformers without Position Encoding (https://arxiv.org/abs/2404.12224)
- **What's New**: 이 논문에서는 위치 인코딩 없는 Transformer(NoPE)의 길이 일반화 특성을 분석하고 구현했습니다. 일반적으로 사용되는 명시적 위치 인코딩보다 더 긴 시퀀스로 성공적으로 확장할 수 있음을 발견했지만, 일정 범위를 넘어서면 성능이 제한된다는 것을 확인했습니다. 이 연구는 NoPE에 대한 새로운 매개변수 효율적 튜닝 방법을 제안하여, 주의 집중의 분포(distraction of attention distributions)와의 관계를 식별하고, 이를 통해 NoPE의 컨텍스트 크기를 크게 확장할 수 있었습니다.

- **Technical Details**: NoPE 모델은 위치 인코딩을 사용하지 않고 훈련되었습니다. 연구팀은 주의 집중 메커니즘의 온도 하이퍼-파라미터(temperature hyper-parameters)를 조정하여 NoPE의 길이 일반화 성능을 개선하는 방법을 개발했습니다. 이 접근 방식은 각 attention head에 대해 다른 매개변수를 탐색하며, 이는 전체 모델 파라미터에 비해 극히 적은 수의 조정 가능한 매개변수를 필요로 합니다. 이러한 최적화를 통해 NoPE는 4K 토큰 이상으로 일반화할 수 있었습니다.

- **Performance Highlights**: 이 논문에서 제안된 NoPE 및 그 튜닝 방법은 긴 시퀀스 언어 모델링, 합성 키 검색 작업(passkey retrieval task), 실제 긴 컨텍스트 작업에서 최첨단 길이 일반화 알고리즘과 경쟁할 수 있는 성능을 달성했습니다. 예를 들어, NoPE는 훈련 길이에서 20% 늘렸을 때 유의미한 복잡도(perplexity) 증가 없이 확장할 수 있었습니다. 이와 대조적으로, 로터리 위치 인코딩(Rotary Position Encoding, RoPE)은 10%만 확장 가능했습니다.



### OpenBezoar: Small, Cost-Effective and Open Models Trained on Mixes of  Instruction Data (https://arxiv.org/abs/2404.12195)
Comments: 25 pages, 27 Figures, 8 Tables

- **What's New**: 이 연구에서는 OpenLLaMA 3Bv2를 기반 모델로 사용하여 OpenBezoar 시리즈 모델을 정밀 튜닝하는 방법을 제시합니다. 다양한 데이터셋과 튜닝 기법을 통합하여 고성능 모델을 개발하였습니다. 특히, Falcon-40B, LaMini-LM, WizardLM/Evol-Instruct, Orca를 이용한 데이터 생성과 GPT-4를 활용한 필터링 과정을 거쳐, 비용 효율적인 QLoRA 기반 순차적 교육 방식을 적용하였습니다.

- **Technical Details**: 첫 단계로 Falcon-40B 모델을 사용하여 synthetic instruction fine-tuning data를 생성하고, LaMini-LM, WizardLM/Evol-Instruct 및 Orca를 사용하였습니다. 이후 생성된 데이터는 GPT-4로 필터링하여 인간 선호도에 맞춥니다. 다음으로, QLoRA(Quality-of-Life Regressive Adjustment) 기법을 이용해 효과적인 순차적 미세 조정을 수행하고, HH-RLHF(Human Hedonic Reinforcement Learning from Human Feedback) 데이터셋의 일부를 사용하여 배포 이전의 분포 차이를 최소화합니다. 최종적으로 DPO(Differential Privacy Optimization) 손실을 통해 최종 체크포인트를 완성합니다.

- **Performance Highlights**: 최종적으로 개발된 'OpenBezoar-HH-RLHF-DPO' 모델은 3B (billion) 파라미터 규모에서 매우 우수한 성능을 보여주며, 몇몇 범주에서는 Huggingface Open LLM Leaderboard 상의 최상위 모델보다 우수한 성능을 나타냈습니다. LM Eval Harness 작업 및 MT-Bench를 사용하여 'LLM-as-a-judge' 프레임워크 내에서 평가되었습니다. 이 연구 결과들은 Huggingface에서 'OpenBezoar-SFT', 'OpenBezoar-HH-RLHF-SFT', 'OpenBezoar-HH-RLHF-DPO' 체크포인트와 함께 제공된 데이터셋과 함께 공개되었습니다.



### EuSQuAD: Automatically Translated and Aligned SQuAD2.0 for Basqu (https://arxiv.org/abs/2404.12177)
Comments: Under review in the journal of Procesamiento de Lenguaje Natural

- **What's New**: 이 연구는 SQuAD2.0을 바스크어로 자동 번역 및 정렬하여 생성된 첫 번째 QA(Question Answering, 질의 응답) 데이터셋인 EuSQuAD를 소개합니다. 142,000개가 넘는 QA 예시를 포함하며, 이는 소수 언어를 위한 NLP(Natural Language Processing, 자연어 처리) 자원의 부족 문제를 해결하는 데 중요한 발전입니다.

- **Technical Details**: 이 작업은 SQuAD2.0 데이터셋을 바스크어로 번역하고 정렬하는 과정을 포함하며, 자동화된 도구를 사용하여 효율성을 높였습니다. EuSQuAD는 기존의 잉글리시 QA 데이터를 활용해 다양한 NLP 모델을 훈련시킬 수 있는 기반을 마련했습니다.

- **Performance Highlights**: EuSQuAD 데이터셋은 새로운 인간 주석이 달린 데이터셋을 사용하여 평가된 QA 실험을 통해 그 가치를 입증했습니다. 이 실험 결과는 EuSQuAD를 훈련 데이터로 사용했을 때 상당한 질의 응답 성능을 보여줌으로써, 바스크어 NLP 개발에 크게 기여할 것으로 예상됩니다.



### Claim Check-Worthiness Detection: How Well do LLMs Grasp Annotation  Guidelines? (https://arxiv.org/abs/2404.12174)
- **What's New**: 이 연구에서는 문서의 신뢰성과 가치 판단(claim check-worthiness detection, CD/CW)을 위한 자동화 기법으로 '제로샷-퓨샷' 기반 대형 언어 모델(LLM) 프롬프트(prompt)를 활용한 접근 방식을 소개하였습니다. 이는 레이블링된 데이터셋의 필요성을 우회하고 직접적인 문구를 사용하여 모델에 적용할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: 연구진은 다양한 분야에서 오는 다섯 개의 데이터셋을 사용하여 문서 신뢰성과 가치 판단 기준의 정의를 어느 정도 프롬프트에 통합할 것인지, 그리고 각 주장에 얼마나 많은 맥락을 제공할 것인지를 실험해 보았습니다. 프롬프트의 자세한 정도(prompt verbosity)와 맥락 정보의 양을 변화시켜가며 이를 조사하였으며, 그 결과는 분야에 따라 최적의 프롬프트 자세한 정도가 다르다는 것을 보여주었습니다. 특히, 추가 맥락이 성능을 향상시키지 않았고, 신뢰도 점수(confidence scores)를 직접 사용하여 믿을 수 있는 가치 판단 순위를 생성할 수 있다는 것을 입증하였습니다.

- **Performance Highlights**: OpenAI의 gpt-4-turbo 모델을 사용하여 기존의 CD/CW 방법들과 비교할 때, 정확성과 순위 부여 점수에서 비슷하거나 뛰어난 성능을 보였습니다. 이는 LLM을 사용하여 제한된 프롬프트 엔지니어링으로 기존의 CD/CW 메소드를 효과적으로 대체할 수 있는 가능성을 보여줍니다.



### Stance Detection on Social Media with Fine-Tuned Large Language Models (https://arxiv.org/abs/2404.12171)
- **What's New**: 이 연구는 자연어 처리(Natural Language Processing, NLP)에서 중요한 작업 중 하나인 입장 탐지(stance detection) 방법의 발전을 평가합니다. 초기의 기계 학습 접근 방식에서부터 혁신적인 BERT 모델, 그리고 최근의 대규모 언어 모델(Large Language Models, LLMs)인 ChatGPT, LLaMa-2, Mistral-7B로의 전환을 다루고 있습니다. 특히, 이 연구는 ChatGPT, LLaMa-2, Mistral-7B를 다양한 데이터셋을 사용하여 미세 조정(fine-tuning)하고, zero-shot 및 few-shot 학습 시나리오에서 이들의 성능을 평가하는 것을 목표로 합니다.

- **Technical Details**: LLM들은 대규모 데이터셋으로 훈련되어 인간의 언어의 미묘함을 놀라울 정도로 정확하게 모방하는 능력을 가졌습니다. 이 연구에서는 ChatGPT, LLaMa-2, Mistral-7B 모델을 트위터(Twitter, 현재는 X로 알려짐) 데이터셋을 사용하여 미세 조정하였고, 이를 통해 사용자의 관점을 더 잘 이해할 수 있도록 하였습니다. LLaMa-2와 Mistral-7B는 오픈소스(open-source)이며, 비교적 작은 크기에도 불구하고 입장 탐지에서 탁월한 효율성과 가능성을 보여주었습니다.

- **Performance Highlights**: LLMs는 입장을 정확하게 탐지하는 뛰어난 능력을 보여주었으며, 모든 테스트된 모델이 기존 벤치마크를 초과했습니다. 특히 LLaMa-2와 Mistral-7B는 뛰어난 성능을 보여주었고, 입장 탐지에서 뛰어난 잠재력을 가지고 있다고 평가되었습니다. 이 연구는 미세 조정(fine-tuning), thought chaining, zero-shot 및 few-shot 학습 같은 고급 기술을 활용하여 복잡한 온라인 담론을 탐색하는 LLMs의 유연성을 강조합니다.



### FecTek: Enhancing Term Weight in Lexicon-Based Retrieval with Feature  Context and Term-level Knowledg (https://arxiv.org/abs/2404.12152)
- **What's New**: 본 논문에서는 기존 어휘 기반 검색의 성능을 높이기 위해 '특징 맥락 모듈(Feature Context Module, FCM)'과 '단어 수준 지식 안내 모듈(Term-level Knowledge Guidance Module, TKGM)'을 도입한 새로운 방법인 FecTek을 소개합니다. FecTek은 BERT의 표현을 활용하여 용어 가중치의 특징 맥락 표현을 풍부하게 하고, 단어 수준 지식을 활용하여 용어 가중치 모델링 과정을 지능적으로 안내합니다.

- **Technical Details**: FecTek은 두 가지 주요 모듈로 구성됩니다. 첫 번째인 '특징 맥락 모듈(FCM)'은 BERT의 표현력을 활용하여 임베딩의 각 요소에 대한 동적 가중치를 결정하고, 이를 통해 얻은 특징 맥락 표현으로 용어 가중치를 풍부하게 합니다. 두 번째 모듈인 '단어 수준 지식 안내 모듈(TKGM)'은 쿼리와 패시지 간의 공유 용어에 더 큰 중요성을 부여하여, 용어 가중치 계산 시 이를 활용하고, 크로스 엔트로피 손실을 사용하여 단어 수준 지식을 활용해 모델링 과정을 안내합니다.

- **Performance Highlights**: FecTek은 MS Marco 벤치마크에서 기존의 최고 성능을 가진 방법론들을 능가하며 새로운 최고 성과(state-of-the-art)로 자리매김하였습니다. 이를 통해 복잡한 어휘 기반 검색 문제에 대한 접근 방식과 성능 개선에 기여할 수 있음을 입증하였습니다.



### From Form(s) to Meaning: Probing the Semantic Depths of Language Models  Using Multisense Consistency (https://arxiv.org/abs/2404.12145)
- **What's New**: 이 연구에서는 언어 모델(LM)의 이해가 어떻게 현실적인 이해와 다른지를 파악하기 위해, 다양한 제시 방식에서 일관된 세계 이해를 평가하는 새로운 방법론을 제안했습니다. 이는 Frege의 '의미(Sinn)' 개념과 Wittgenstein의 언어의 의미에 관한 이론을 기반으로, GPT-3.5 모델을 사용하여 다국어 및 다양한 NLU 벤치마크를 통해 테스트되었습니다.

- **Technical Details**: 연구진은 '멀티센스 일관성(multisense consistency)'이라는 새로운 매트릭을 사용하여, GPT-3.5가 텍스트만을 기반으로 학습했음에도 불구하고 여러 언어와 파라프레이징(paraphrasing)을 통해 같은 의미를 일관되게 표현할 수 있는지 평가했습니다. 이를 통해 모델이 언어의 형태(form)가 아닌 실제 '의미(meaning)'를 얼마나 이해하고 있는지를 분석했습니다.

- **Performance Highlights**: GPT-3.5는 다양한 언어와 NLU 벤치마크에서 멀티센스 일관성이 부족한 것으로 나타났습니다. 이는 언어 모델이 여전히 인간처럼 일관된 이해를 달성하기에는 거리가 멀다는 것을 시사합니다. 추가 분석을 통해 이 일관성 부족이 과제에 대한 '감각 의존적(sense-dependent)' 이해 때문임을 확인했습니다.



### LongEmbed: Extending Embedding Models for Long Context Retrieva (https://arxiv.org/abs/2404.12096)
- **What's New**: 새로운 논문에서는 기존 임베딩(embedding) 모델의 문맥 창(context window)을 32k까지 확장하는 새로운 전략을 제시하며, 이를 통해 법률 문서와 같은 긴 입력을 필요로 하는 애플리케이션에 적용할 수 있게 되었습니다. 특히, 절대 위치 인코딩(Absolute Position Encoding, APE)과 회전 위치 인베딩(Rotary Position Embedding, RoPE)을 사용하는 모델의 성능 향상 가능성을 탐구하였으며, RoPE가 APE보다 문맥 창 확장에 더 유리함을 입증하였습니다.

- **Technical Details**: 이 연구에서는 기존 임베딩 모델의 능력을 평가하기 위해 LongEmbed 벤치마크를 구축하고, 문맥 확장 전략인 위치 보간(position interpolation), 병렬 문맥 창(parallel context windows), 위치 ID 재구성(reorganizing position ids) 등을 실험했습니다. 또한 APE를 사용하는 모델을 위한 추가 조정과 RoPE를 사용하는 모델을 위한 RoPE-특정 메소드인 NTK와 SelfExtend의 적용을 통해 상당한 성능 향상을 관찰했습니다.

- **Performance Highlights**: LongEmbed 벤치마크에서의 실험 결과는 기존 임베딩 모델들이 크게 개선될 여지가 있다는 것을 보여줍니다. 특히, RoPE를 활용한 E5-Mistral 모델은 문맥 창을 32k로 확장하고, 키 리트리벌(key retrieval) 및 LongEmbed에서 최고 성능을 달성했습니다. 이 연구는 또한 E5Base-4k와 E5-RoPEBase 모델을 비롯하여 LongEmbed 벤치마크 자체를 공개하여, 앞으로의 연구를 촉진하고자 합니다.



### RAGAR, Your Falsehood RADAR: RAG-Augmented Reasoning for Political  Fact-Checking using Multimodal Large Language Models (https://arxiv.org/abs/2404.12065)
Comments: 8 pages, submitted to ACL Rolling Review

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)과 검색-증강 생성(Retrieval-augmented Generation, RAG)을 통합하여 다모달 사실 확인의 신뢰성과 효율성을 향상시키는 두 가지 새로운 방법론, RAG의 사슬(Chain of RAG, CoRAG)과 RAG의 나무(Tree of RAG, ToRAG)를 제안합니다. 이들 접근법은 이미지와 텍스트를 모두 분석할 수 있는 다모달 LLM을 활용하여 정치적 주장의 사실성을 판단합니다.

- **Technical Details**: 이 연구는 텍스트와 이미지를 모두 처리할 수 있는 다모달 LLM을 사용하여, 검증해야 할 주장에 대한 다음 질문을 추론함으로써 다모달 주장을 처리합니다. CoRAG와 ToRAG 접근 방식은 기존의 사실 확인 접근 방식을 개선하여 사실 확인 정확도와 설명 생성을 향상시킵니다. 또한, RAG-augmented Reasoning (RAGAR) 방법을 사용하여 정치 주장의 사실성을 결정하기 위한 다음 단계를 촉진합니다.

- **Performance Highlights**: RAGAR 접근 방식은 다양한 사실 확인 작업에서 효과적인 성능을 보여주며, 특히 복잡한 정치적 주장에 대한 사실성 평가에 유용합니다. MOCHEG 데이터셋을 사용하여 실시한 실험에서 300개의 다모달 주장 중 150개를 지지하고 150개를 반박하는 레이블을 정확하게 예측하는 능력을 입증하였습니다.



### Constituents Correspond to Word Sequence Patterns among Sentences with  Equivalent Predicate-Argument Structures: Unsupervised Constituency Parsing  by Span Matching (https://arxiv.org/abs/2404.12059)
- **What's New**: 이 연구는 우수하다(Predicate-Argument Structure, PAS) 동등한 문장 세트에서 동사구 구성체를 확인하기 위한 빈도 기반 방법인 'span-overlap'을 제안합니다. 이 방법은 계산 무감독 파싱(computational unsupervised parsing)에서 처음으로 단어 시퀀스 패턴의 적용을 도입하며, 10개 언어 중 8개에서 최신 기술의 파서를 능가하는 성능을 보여줍니다.

- **Technical Details**: span-overlap 방법은 대상 문장의 PAS-동등한 문장 세트에서 받침구 패턴(span pattern)을 발견하여 이를 활용합니다. 이 방법은 대규모 언어 모델(Large Language Model, LLM)을 사용하여 PAS-동등 문장을 생성하고, 이를 토대로 구문 분석을 수행합니다. 연구는 구성요소와 비구성요소(non-constituents)를 분리하는 데 있어 중요하게 작용하는 단어 시퀀스 패턴의 유틸리티를 입증합니다.

- **Performance Highlights**: span-overlap 파서는 다국어 실험에서 10개국어 중 8개국어에서 최고 성능을 달성했습니다. 구성 요소를 비구성 요소에서 구분하는 능력에서도 무작위 기준(random baseline)보다 훨씬 우수한 결과를 보여주었습니다. 또한, 참여자를 나타내는 구성 요소가 사건을 나타내는 구성 요소보다 빈번하게 등장한다는 다국어 현상을 발견했으며, 이는 두 구성 요소 유형 간의 행동적 차이를 시사합니다.



### emrQA-msquad: A Medical Dataset Structured with the SQuAD V2.0  Framework, Enriched with emrQA Medical Information (https://arxiv.org/abs/2404.12050)
Comments: The dataset is available in this https URL

- **What's New**: 이 연구는 의료 질문 응답 시스템(Medical Question Answering Systems, QAS)의 효율성을 강화하기 위해 특수 의료 데이터셋을 통합하고 새로운 데이터셋을 구축하는 새로운 접근 방식을 제시합니다. 특히, emrQA에서 파생되어 재구성된 'emrQA-msquad' 데이터셋과 헌신적으로 만들어진 의료 분야 스팬 추출(Span extraction) 태스크 전용 데이터셋이 소개됩니다.

- **Technical Details**: BERT, RoBERTa, 그리고 Tiny RoBERTa 모델들이 의료 분야에 맞게 미세 조정되면서, 더 정밀한 답변 추출이 가능해졌습니다. 이 데이터셋은 SQuAD V2.0의 체계를 의료 내용의 emrQA 데이터셋과 통합하여 구축되었습니다. 또한, 특정 의학적 문맥에서 모델의 성능을 최적화하기 위해 필수적인 세밀한 조정이 필요했습니다.

- **Performance Highlights**: 미세 조정된 모델들은 처음 베이스라인에서 보여준 F1 점수 범위(0.75-1.00)에서 각각 10.1%에서 37.4%, 18.7%에서 44.7%, 그리고 16.0%에서 46.8%로 개선되었습니다. 이러한 개선은 의료 컨텍스트에서의 정확성 향상을 나타냅니다.



### Exploring Boundaries and Intensities in Offensive and Hate Speech:  Unveiling the Complex Spectrum of Social Media Discours (https://arxiv.org/abs/2404.12042)
- **What's New**: 이 연구에서는 트윗에 대한 3가지 고유한 작업: 카테고리 분류, 증오 대상 식별 및 공격성 및 증오 강도 평가를 포함하여 아마릭어로 구성된 광범위한 벤치마크 데이터세트를 제시합니다. 이러한 고급 분석을 통해 증오와 공격적 언행이 단순히 이분법적 범주로만 처리될 수 없으며, 지속적인 다양성을 띄고 있음을 드러냈습니다.

- **Technical Details**: Afro-XLMR-large 모델은 카테고리, 대상, 회귀 작업에서 각각 75.30%, 70.59%, 29.42%의 F1-score를 달성하여 우수한 성능을 보였습니다. 80.22%의 상관 계수는 모델의 정확한 정렬을 나타냅니다. 데이터셋 수집은 X(전 Twitter)로부터 이루어졌으며, 아마릭어를 모국어로 하는 5명의 주석자(annotator)가 각 트윗에 대한 주석을 제공했습니다.

- **Performance Highlights**: 데이터셋의 다수 트윗은 덜 공격적이고 증오 강도가 낮은 수준에 속함을 강조하여 초기 개입의 필요성을 강조합니다. 아울러 Afro-XLMR-large 모델은 강력한 정렬을 보여주는 높은 상관계수를 기록하였습니다.



### Can We Catch the Elephant? The Evolvement of Hallucination Evaluation on  Natural Language Generation: A Survey (https://arxiv.org/abs/2404.12041)
Comments: 19 pages in total, with 9 pages as main body. Under review as a conference paper at CoLM 2024

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 텍스트 생성에서 '환각'(hallucination) 문제에 주목하고 있습니다. 환각 현상을 평가하기 위한 여러 방법을 종합적으로 조사하여, 다양한 정의와 사실의 세밀성(fact granularity), 자동 평가기(evaluators)의 유형 및 적용 가능성, 그리고 해결되지 않은 문제와 미래의 방향성에 대해 논의합니다.

- **Technical Details**: 환각을 평가하는 주요 개념은 '출처 충실도'(Source Faithfulness, SF)와 '세계 사실성'(World Factuality, WF)로 구분됩니다. SF는 생성된 텍스트가 원본 입력과 얼마나 일치하는지를 측정하는 반면, WF는 생성된 텍스트가 일반적인 세계 지식과 사실에 얼마나 부합하는지를 평가합니다. 또한 평가 방법은 시대에 따라 이른바 전통적인 방식에서 LLM을 이용한 새로운 방법으로 진화해왔으며, 종류에 따라 세분화하여 분류하는 새로운 분류 방법을 제안합니다.

- **Performance Highlights**: 이 연구는 기존의 탐구들과는 달리 더 폭넓은 관점에서 환각 평가 방법을 분석하며, LLM의 최신 발전에 따른 새로운 문제들에 주목합니다. 여러 자동 평가 방법들을 비교 분석하고, 실제 적용될 수 있는 분류 방식을 제안함으로써, 텍스트 생성의 신뢰성과 안전성을 높이는 데 기여할 것으로 기대됩니다.



### Uncovering Safety Risks in Open-source LLMs through Concept Activation  Vector (https://arxiv.org/abs/2404.12038)
- **What's New**: 이 연구는 LLaMA-2와 같은 원래 잘 정렬된 LLM에 대한 공격을 가능하게 하는 새로운 방법론을 제시합니다. 이는 안전 개념 활성 벡터(Safety Concept Activation Vectors, SCAVs)를 이용하여 LLM의 안전 취약점을 탐지하고, 공격의 성공률을 거의 100%로 높이는 방법을 개발하였습니다. 이 방법은 기존의 공격 기술들이 보여주지 못한 높은 효율과 정확성을 제공합니다.

- **Technical Details**: 이 연구는 LLM의 계산 흐름에서 안전 개념 활성 벡터(SCAVs)를 추출하고 이를 조작하여 LLM을 공격하는 메커니즘을 설명합니다. SCAV는 LLM의 활성화 공간에서 안전 관련 개념을 대표하는 벡터로서, 이를 조정함으로써 LLM의 출력물을 통제할 수 있습니다. 이 방법은 복잡한 계산 없이도 빠르고 간단하게 실행될 수 있으며, 다른 오픈소스 LLM으로의 전이성도 관찰되었습니다.

- **Performance Highlights**: SCAV 기반 공격 방법은 성공률이 거의 100%에 가깝고, 출력의 해로움을 평가하는 새로운 방법론을 제안함으로써 공격이 실제로 유해하다는 것을 추가적으로 검증했습니다. 이는 ASR(attacks success rate), GPT-4 등급, 그리고 사람 기반 평가를 포함한 종합적인 평가 방법을 통해 이루어졌습니다. 또한, 이 연구는 다양한 LLM에 대한 공격 전략과 그 결과의 해로움을 체계적으로 평가하고, 안전성을 강화하는 데 중요한 기여를 할 수 있습니다.



### Parallel Decoding via Hidden Transfer for Lossless Large Language Model  Acceleration (https://arxiv.org/abs/2404.12022)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 추론 시간당(latency) 문제를 해결하기 위해 'Hidden Transfer'라는 새로운 병렬 디코딩(parallel decoding) 방법을 제안합니다. 이 방법은 이전 컨텍스트의 중간 숨겨진 상태(intermediate hidden states)를 미래 토큰의 '가상(pseudo) 숨겨진 상태'로 전달하여, 한 번의 포워드 패스(forward pass)에서 여러 개의 연속 토큰을 동시에 디코딩할 수 있습니다. 추가로, 'tree attention mechanism'을 사용하여 출력 시퀀스의 여러 후보를 동시에 생성하고 검증함으로써, 우리의 방법이 이전의 단일 모델 가속 기술보다 우수한 성능을 보임을 입증합니다.

- **Technical Details**: 'Hidden Transfer'는 입력 토큰의 숨겨진 상태를 특정 중간 레이어의 미래 토큰의 가상 숨겨진 상태로 변환하는 훈련 가능한 선형 투영(linear projection)을 사용합니다. 이 가상 숨겨진 상태는 이후 레이어를 통과하면서 전체 시퀀스의 숨겨진 상태와 상호 작용하고, 최종 레이어에서 원래의 언어 모델 헤드(lm-head)를 사용하여 미래 위치의 초안 토큰을 디코드합니다. 훈련 단계에서는 KL-분산(KL-divergence)을 감독 신호(supervised signal)로 사용하여, 가상 숨겨진 상태에 의해 예측된 토큰의 분포와 실제 토큰의 분포 간의 차이를 최소화합니다.

- **Performance Highlights**: 실험 결과, 'Hidden Transfer' 방법은 가장 높은 초안 토큰 예측 정확도(draft token prediction accuracy)와 추론 가속 비율(inference acceleration ratio)을 달성하여 다른 단일 모델 설정 하에서의 방법들과 비교하여 가장 우수한 성능을 보였습니다. 또한, 본 논문에서 제안하는 'tree attention mechanism'는 출력 시퀀스의 여러 후보를 동시에 생성하고 검증함으로써 무손실 생성(lossless generation)을 보장합니다.



### Enhance Robustness of Language Models Against Variation Attack through  Graph Integration (https://arxiv.org/abs/2404.12014)
Comments: 12 pages, 4 figures, accepted by COLING 2024

- **What's New**: 이 연구에서는 중국어 콘텐츠에서 문자 변형 공격에 대한 언어 모델의 강인성을 향상시키는 새로운 방법, CHANGE (CHinese vAriatioN Graph Enhancement)를 제안합니다. CHANGE는 중국어 문자 변형 그래프를 PLMs(pre-trained language models)에 통합하는 새로운 접근 방식을 제시하여 광범위한 어택 시도에 대해 모델의 이해력과 반응성을 강화합니다.

- **Technical Details**: CHANGE 방법은 중국어 문자 변형 지식 그래프를 활용하여 다중 태스크 프레임워크를 통해 PLMs의 어텍 텍스트에 대한 리터럴 이해를 개선합니다. 첫째, Chinese Variation Graph Integration(CVGI) 방식은 변형 그래프를 사용하여 fine-tuning 과정에서 PLM의 강인성을 강화합니다. 둘째, Variation Graph Instructed Pre-training 방식은 추가적인 사전 훈련 태스크를 그래프를 통해 PLMs에 적용하여 변형 그래프의 잠재력을 극대화합니다.

- **Performance Highlights**: 실험 결과 CHANGE는 다양한 NLP 작업에서 기존 언어 모델보다 뛰어난 성능을 보였으며, 특히 적대적 공격에 대한 강건성이 크게 향상되었습니다. 이는 그래프 가이드 사전 훈련 전략이 실제 응용 프로그램에 큰 잠재력을 가질 수 있음을 강조합니다.



### Sequential Compositional Generalization in Multimodal Models (https://arxiv.org/abs/2404.12013)
Comments: Accepted to the main conference of NAACL (2024) as a long paper

- **What's New**: 이 연구는 대규모 다중모달 모델을 활용하여 복잡한 시퀀스의 구성 일반화(compositional generalization) 능력을 탐구합니다. 특히, 이 연구에서는 새로운 CompAct (Compositional Activities) 데이터셋을 도입하여, 주방 활동을 중심으로 한 인지적 활동들에 대한 일반화 가능성을 평가합니다.

- **Technical Details**: 연구팀은 여러 가지 단일모달(uni-modal) 및 다중모달(multimodal) 모델들을 평가하였으며, 이는 텍스트만을 사용하는 모델보다 이중모달(bi-modal) 및 삼중모달(tri-modal) 모델이 더 낫다는 것을 발견하였습니다. 특히, CompAct 데이터셋은 EPIC KITCHENS-100 데이터셋에서 추출되었으며, 각 비디오 인스턴스는 원시 비디오, 자연스러운 소리, 그리고 군중 소싱된 단계별 설명을 조합하여 구성됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 다양한 입력 스트림의 조합(예: 비디오 + 언어, 비디오 + 오디오, 비디오 + 언어 + 오디오)을 처리할 수 있는 다중모달 모델들이 강력한 구성 일반화 능력을 보여줄 수 있다고 합니다. 특히, IDEFICS 모델이 다중모달 순차적 구성 일반화에 매우 유용할 수 있음을 시사합니다.



### ParaFusion: A Large-Scale LLM-Driven English Paraphrase Dataset Infused  with High-Quality Lexical and Syntactic Diversity (https://arxiv.org/abs/2404.12010)
- **What's New**: 이 연구는 자연어 처리(Natural Language Processing, NLP) 분야에서 중요한 '패러프레이즈 생성(paraphrase generation)' 작업을 향상시키기 위해 'ParaFusion'이라는 새로운 대규모, 고품질의 영어 패러프레이즈 데이터셋을 소개합니다. ParaFusion은 기존 데이터셋을 확장하여 어휘적 및 문법적 다양성을 크게 향상시키고, 혐오 발언(hate speech)의 존재를 완화하며, 노이즈를 줄이는 것을 목표로 합니다. 이는 Large Language Models (LLM)를 사용하여 개발되었습니다.

- **Technical Details**: ParaFusion은 기존의 패러프레이즈 데이터셋을 확장하고, ChatGPT (gpt-3.5-turbo)와 같은 LLM을 이용하여 고품질의 데이터를 생성합니다. 문법적 정확성, 어휘 다양성, 문법적 다양성, 그리고 의미적 유사성을 유지하면서 데이터의 품질을 향상시킨다는 점에서 기존의 데이터 조사 방법을 크게 개선합니다. 또한, 이 데이터셋의 평가를 위한 새로운 표준을 설정하여 다양한 메트릭(metric)으로 데이터 소스별로 측정하였습니다.

- **Performance Highlights**: ParaFusion은 문법적 및 어휘적 다양성 측면에서 기존 데이터셋 대비 최소 25% 이상의 개선을 보여줍니다. 이는 여러 평가 메트릭을 통해 측정된 결과이며, 이러한 향상은 NLP 응용 프로그램에서의 성능 개선 가능성을 시사합니다. 또한, ParaFusion은 기존 데이터셋과 비교하여 훨씬 높은 품질의 패러프레이즈를 제공한다는 점이 인간 평가(human evaluation)를 통해 확인되었습니다.



### Variational Multi-Modal Hypergraph Attention Network for Multi-Modal  Relation Extraction (https://arxiv.org/abs/2404.12006)
- **What's New**: 본 논문에서는 멀티모달 관계 추출(Multi-modal relation extraction, MMRE)에 대한 새로운 접근 방식인 변이형 멀티모달 하이퍼그래프 어텐션 네트워크(Variational Multi-Modal Hypergraph Attention Network, VM-HAN)를 제안합니다. 이 모델은 텍스트와 이미지 정보를 활용하여 문장 내의 다른 엔티티 쌍 간의 상위 수준(high-order) 내/외부모달 상관관계를 구축합니다.

- **Technical Details**: VM-HAN은 각 문장과 해당 이미지에 대해 다중 모달 하이퍼그래프(multi-modal hypergraph)를 구성하여 다양한 엔티티 쌍을 위한 상호 모달 상관 관계를 설정합니다. 변이 하이퍼그래프 어텐션 네트워크(Variational Hypergraph Attention Networks, V-HAN)를 통해 가우스 분포를 사용하여 다른 엔티티 쌍 간의 표현 다양성을 확보하고, 변이 주의(variational attention)를 통해 더 나은 하이퍼그래프 구조를 학습합니다.

- **Performance Highlights**: VM-HAN은 MORE 데이터셋에서 이미지 특징을 사용하는 실험을 통해 기존 방법들에 비해 정확도와 효율성 면에서 최신의 성능(state-of-the-art performance)을 달성하였습니다. 또한, 이미지 정보의 중요성과 서로 다른 모달 간의 상위 수준 상관관계를 캡처하는 것의 중요성을 강조하는 결과를 보여줍니다. 다양한 엔티티 반복 수에 따른 실험에서도 멀티모달 관계 추출 작업에서의 성능 차이를 그래픽적으로 입증하였습니다.



### Token-level Direct Preference Optimization (https://arxiv.org/abs/2404.11999)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)을 인간의 선호도에 맞추기 위한 새로운 방법으로 Token-level Direct Preference Optimization (TDPO)을 소개했습니다. TDPO는 기존의 DPO 방식을 개선하여 토큰 수준에서의 KL 발산을 제어함으로써 모델의 정렬 성능(Alignment Performance)과 생성 다양성(Generation Diversity) 사이의 균형을 더 잘 맞출 수 있습니다.

- **Technical Details**: TDPO는 각 토큰에 대해 forward KL divergence 제약을 통합하여 KL 발산을 조절합니다. 이는 Bradley-Terry 모델을 토큰 기반 보상 시스템으로 활용하여 달성되며, 정책 최적화 단계에서 명시적인 보상 모델 학습이나 정책 샘플링을 요구하지 않습니다. 또한, 문장 수준의 보상과 토큰 수준 생성 사이의 연결은 Bellman 방정식을 사용하여 설정됩니다.

- **Performance Highlights**: 실험 결과, TDPO는 제어된 감정 생성 및 단일 턴 대화 데이터셋에서 DPO보다 향상된 균형을 보여주었으며, DPO 및 PPO 기반 RLHF 방법보다 생성된 응답의 품질을 크게 개선하였습니다. 이러한 결과는 TDPO가 과도한 KL 발산 문제뿐만 아니라 발산 효율성을 크게 향상시킬 수 있음을 시사합니다.



### EVIT: Event-Oriented Instruction Tuning for Event Reasoning (https://arxiv.org/abs/2404.11978)
- **What's New**: 이 연구에서는 사건 추론(Event reasoning)의 능력을 향상시키기 위한 새로운 접근법, 'Event-Oriented Instruction Tuning (EvIT)'를 제안합니다. EvIT는 사건의 구조와 의미를 명확하게 모델링하고, 이를 지시어(instruction) 튜닝과 결합하여 언어 모델이 사건 관련 지식을 효과적으로 이해하고 추론할 수 있도록 합니다. 특히, 사건 사중체(event quadruple)라는 새로운 구조를 도입하여 사건들의 상호 관계를 학습하는 기반을 마련하였습니다.

- **Technical Details**: EvIT 방법론은 대규모 텍스트 코퍼스에서 사건 사중체를 구축하고, 이를 활용하여 사건 관계 학습(event-relation learning)을 수행합니다. 이 사중체는 두 사건, 그들의 관계, 그리고 사실이 유효한 배경 정보를 포함합니다. 학습 과정은 생성 및 판별 방식으로 진행되며, 특히 지시어 튜닝(instruction tuning) 과정에 통합되어 모델이 사건 추론을 보다 효과적으로 수행할 수 있도록 설계되었습니다. Llama 모델을 EvIT로 파인튜닝하여, 여러 데이터셋에서의 사건 추론 작업에 대한 실험을 수행하였습니다.

- **Performance Highlights**: EvIT는 학습 중 보지 못한 8가지 사건 추론 작업에서의 평가를 수행했습니다. 이 중 4가지는 훈련 중 명시적으로 다루어진 관계를 포함하는 태스크이며, 나머지 4가지는 훈련 중 다루어지지 않은 태스크입니다. 자동 및 인간 평가 결과, EvIT는 기타 지시어 튜닝 모델들에 비해 경쟁력 있는 성능을 보여줬습니다. 결과적으로, EvIT는 사건 구조와 의미의 이해를 향상시키며, 관련 사건 지식을 효과적으로 통합할 수 있는 능력을 향상시킵니다.



### Aligning Language Models to Explicitly Handle Ambiguity (https://arxiv.org/abs/2404.11972)
- **What's New**: 이 논문은 모델이 본질적으로 모호한 입력을 명확히 처리할 수 있도록 Large Language Models (LLMs)를 조정하는 새로운 방법을 제안합니다. 특히, 입력의 모호함을 자체적으로 해석할 수 있도록 LLMs를 유도하는 프록시 작업을 소개합니다. 이는 모델이 입력을 얼마나 모호하게 인지하는지를 측정하기 위한 정보 획득을 정량화하는 데 사용됩니다.

- **Technical Details**: 이 연구는 대규모 언어 모델(LLMs)의 본질적인 지식을 이용하여 주어진 질문을 자체 해석하도록 유도하는 프록시 작업을 설계합니다. 이어서, 해석 과정에서의 정보 획득을 암시적 척도로 이용하여 모델이 입력을 모호하다고 판단하는 샘플을 선별하고 이를 학습에 활용합니다. 또한, AmbigTriviaQA라는 새로운 데이터셋을 구축하여 모델의 모호성 처리 능력을 평가합니다.

- **Performance Highlights**: 모호한 입력을 처리하기 위해 조정된 LLMs는 여러 QA 데이터셋에 대한 실험 결과에서 모호한 질문을 명확히 해석하는 능력이 향상되었음을 보여주었습니다. 또한, 명확한 질문에 대해서도 경쟁력 있는 성능을 유지하면서, 모델이 입력을 어떻게 인지하는지를 평가하는 방법으로서의 잠재력을 강조합니다.



### P-NAL: an Effective and Interpretable Entity Alignment Method (https://arxiv.org/abs/2404.11968)
Comments: 13 pages, 2 figures

- **What's New**: 이 논문에서는 엔티티 정렬(Entity Alignment, EA)의 새로운 접근 방식인 P-NAL을 소개합니다. P-NAL은 비공리적 논리(Non-Axiomatic Logic, NAL)를 사용하여 두 유형의 논리적 추론 경로를 포착합니다. 이 방법은 기존의 임베딩 기반 방식이나 경로 기반 방식의 한계를 극복하고, 높은 해석 가능성(interpretability)과 확장성을 제공합니다.

- **Technical Details**: P-NAL은 향상된 엔티티와 관계의 정렬을 위해 비공리적 논리(NAL)의 개정 추론 규칙을 활용하여 두 유형의 인퍼런스 경로의 증거를 취합합니다. 첫 번째 유형(Type I)은 'align-bridge'로 명명된 브리지 형태의 경로이며, 두 관계/속성 트리플과 다른 두 엔티티 간의 유사성 문장을 포함합니다. 두 번째 유형(Type II)은 엔티티 쌍을 그들의 이름이나 설명의 임베딩을 통해 연결합니다. P-NAL은 반복적인 정렬 전략을 채택하여, 각 반복에서 유사성 추론을 수행한 후 수정된 rBMat 알고리즘을 사용하여 EA 결과를 도출합니다.

- **Performance Highlights**: P-NAL은 벤치마크 데이터셋 DBP15K에서 상태 최고의 감독(supervised) 및 비감독(unsupervised) 방법들을 포함한 12개의 기존 EA 방법들을 능가하는 성능을 보여주었습니다. 특히, 모든 데이터셋에 대해 0.98 이상의 Hits@1 성능을 달성했습니다. 이러한 결과는 P-NAL의 유사성 추론이 현재의 EA 작업의 본질을 효과적으로 포착하고 있음을 시사합니다.



### CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual  Knowledge Alignmen (https://arxiv.org/abs/2404.11932)
Comments: 11 pages

- **What's New**: 이 논문은 비영어권 언어에서의 대규모 언어 모델(Large Language Models, LLMs)의 성능 향상을 위한 새로운 접근법인 CrossIn을 제안합니다. 이 방법은 다양한 언어에서 효율적으로 모델의 작업 해결 능력 및 다국어 능력을 향상시키기 위해 교차 언어 지시 튜닝(Cross-lingual Instruction Tuning) 데이터의 혼합 구성을 활용합니다. 또한, 다국어 일관성과 정확성을 평가하기 위한 다면적 벤치마크를 소개합니다.

- **Technical Details**: CrossIn은 다양한 언어의 압축 표현을 공유하여 LLMs의 언어 간 지식 정렬을 지원합니다. 이 연구는 문제해결, 상식 질문 응답(Commonsense Question-Answering), 논리 추론(Logic Reasoning)과 같은 세 가지 작업에 대해 다국적 데이터셋을 사용하여 모델의 다국어 성능을 평가하는 벤치마크를 구축합니다. 이 벤치마크는 국가 간 일치성을 측정하고 다양한 난이도 수준을 포괄합니다.

- **Performance Highlights**: 실험 결과는 CrossIn을 통한 다국어 튜닝이 모든 테스트 측면에서 최대 40%까지 상대적으로 성능 향상을 보였으며, 다국어 데이터의 양과 번역 데이터의 통합이 언어 일관성 및 지식 정확성 향상에 미치는 영향에 대해 상세한 분석을 제공합니다.



### SKIP: Skill-Localized Prompt Tuning for Inference Speed Boost-Up (https://arxiv.org/abs/2404.11916)
Comments: 6 pages

- **What's New**: 이 연구에서는 새로운 방식인 SKIll-localized Prompt tuning (SKIP)이 제안되었습니다. 이 방법은 추론 속도를 최대 160% 향상시키고, 52%의 파라미터를 가지치기(pruning)하면서도 효율적인 추론이 가능합니다. SKIP은 기존의 프롬프트 튜닝 방식과 기술 위치화(skill localization) 방식을 통합하여 언어 모델에서 특정 기술과 관련된 뉴런만을 유지함으로써, 추론 효율을 크게 향상시킵니다.

- **Technical Details**: SKIP 방식은 언어 모델에서 각 뉴런의 기술 관련성을 정량화하기 위해 설명 가능성(explainability) 방법을 사용하는 attribution을 도입합니다. 이 기법으로, 특정 작업에 필요하지 않은 뉴런은 구조적 가지치기(structured pruning)를 통해 제거됩니다. 언어 이해 벤치마크에서 널리 사용되는 여러 변형된 트랜스포머(transformer) 아키텍처에서 이 방식의 유용성을 증명하였습니다. 트랜스포머 기반의 어떠한 아키텍처에도 적용 가능하여 실용성과 확장성을 보유하고 있음을 보여줍니다.

- **Performance Highlights**: 이 연구는 기존 PEFT 방법들보다 우수한 추론 효율을 달성하였으며, 원래의 성능을 손상시키지 않는 동시에 추론 속도를 최대 160% 향상시킨 것으로 나타났습니다. 또한, SKIP을 사용함으로써 52%의 파라미터 가지치기율을 달성했습니다. 이는 언어 모델 처리에 중요한 층을 이해하고 기술 지식을 처리하는 데 중요한 레이어 유형을 조사하여 튜닝 방법을 사용할 때의 지침을 제공합니다.



### TriForce: Lossless Acceleration of Long Sequence Generation with  Hierarchical Speculative Decoding (https://arxiv.org/abs/2404.11912)
- **What's New**: TriForce는 대규모 언어 모델(Large Language Models, LLMs)의 긴 시퀀스 추론을 위한 계층적인 추측 디코딩 시스템을 도입합니다. 효율적인 긴 시퀀스 생성을 위해 원본 모델 가중치와 동적 희소 key-value (KV) 캐시를 활용하는 새로운 접근 방식을 제안합니다. 이 시스템은 빠른 추론 속도와 높은 확장성을 제공합니다.

- **Technical Details**: TriForce는 원본 모델의 가중치를 활용하지만, KV 캐시의 작은 비율(예: 3%)만을 이용하여 병목 현상을 해결합니다. 계층적으로, draft model이 더 가벼운 모델(예: Llama-68M)에 의해 추측되어 모델 가중치의 병목 현상을 해결합니다. 특히, KV 캐시 선택 방법으로는 retrieval-based drafting 전략을 사용하여 필요한 컨텍스트 정보를 검색하고, 이는 정보 손실을 최소화합니다. 또한, StreamingLLM 캐시를 사용하여 초기 추측을 더 빠르게 처리할 수 있습니다.

- **Performance Highlights**: TriForce는 단일 A100 GPU에서 Llama2-7B-128K 모델을 사용하여 최대 2.31배의 속도 향상을 달성했습니다. 또한, 두 대의 RTX 4090 GPU를 사용하는 offloading 설정에서는 0.108s/token의 인상적인 속도를 기록하여, A100에서의 자동 회귀 기준 대비 속도가 두 배 빠르게 나타났습니다. 이 시스템은 다양한 온도 설정에서도 안정적인 성능을 유지합니다.



### Challenging Negative Gender Stereotypes: A Study on the Effectiveness of  Automated Counter-Stereotypes (https://arxiv.org/abs/2404.11845)
Comments: LREC-COLING2024

- **What's New**: 이 연구는 온라인 커뮤니케이션에서 성별 스테레오타입(gender stereotypes)의 부정적인 영향을 인식하고, 이러한 관점들을 자동으로 반박하고 도전하는 11가지 전략을 조사합니다. 성 기반 반(反) 스테레오타입을 생성하기 위해 AI를 사용하고, 참가자들에게 그들의 불쾌감, 가능성, 그리고 효과성을 평가하도록 요청했습니다.

- **Technical Details**: 연구에서는 ChatGPT를 사용하여 부정적인 스테레오타입에 맞서는 데 효과적인 반론을 생성할 수 있는지를 평가했습니다. 대표적으로 반-사실(counter-facts)과 보편적 확장(broadening universals) 같은 전략이 효과적인 것으로 나타났으며, 유머(humour), 관점 취하기(perspective-taking), 반례(counter-examples), 발언자에 대한 공감(empathy) 등 다른 전략들은 상대적으로 덜 효과적인 것으로 평가되었습니다. 또한, 스테레오타입의 대상에 따라 평가가 다른 것으로 나타났지만, 평가자의 성별에 따라서는 그 차이가 덜 두드러졌습니다.

- **Performance Highlights**: 연구 결과에 따르면, 차별적 내용은 일반적으로 온라인 플랫폼에서 삭제될 수 있으나, 스테레오타입과 미세 공격(microaggressions)은 그대로 남아있을 가능성이 더 높아, 이에 대응하는 전략 개발이 중요합니다. AI가 생성한 많은 반-스테레오타입은 불쾌하거나 비현실적으로 받아들여졌지만, 일부 전략은 온라인 상호작용에서 성별 스테레오타입에 효과적으로 도전할 가능성을 보여줍니다.



### AdvisorQA: Towards Helpful and Harmless Advice-seeking Question  Answering with Collective Intelligenc (https://arxiv.org/abs/2404.11826)
Comments: 19 pages, 11 figures

- **What's New**: AdvisorQA는 개인적이고 주관적인 문제에 대한 조언을 제공하는 데 효과적인 LLM(Large Language Models)의 능력을 평가하는 최초의 벤치마크입니다. 이 벤치마크는 LifeProTips 서브레딧(Subreddit) 포럼에서 실제 사용자가 작성한 질문과 답변을 기반으로, 각 질문에 대해 평균 8.9개의 답변과 평균 164.2개의 투표를 받아 순위를 매기고 있습니다. AdvisorQA는 개별적인 조언을 제공할 때 고려해야 할 ‘도움됨(helpfulness)’과 ‘해롭지 않음(harmlessness)’의 두 가지 품질 지표로 평가합니다.

- **Technical Details**: AdvisorQA는 각 QA에 대해 사용자 선호에 근거한 순위를 기반으로 하며, Plackett-Luce (PL) 모델을 사용하여 ‘도움됨’ 지표를 구성하였습니다. 도움됨에 대한 평가는 객관성만이 아니라 주관적 경험을 반영해야 하기 때문에, 피드백을 반영하는 강화 학습(Reinforcement Learning with Human Feedback, RLHF)을 사용하여, 모델이 실제 사용자의 반응을 효과적으로 예측할 수 있도록 조정했습니다. 한편 '해롭지 않음'은 LifeTox 모델을 사용하여 평가하였으며, 이는 서브레딧 데이터에 기반하여 훈련되었습니다.

- **Performance Highlights**: AdvisorQA를 통해 테스트된 LLM은 도움됨과 해롭지 않음의 균형을 이루며 주관적 조언을 제공할 수 있는 능력을 보여줍니다. 특히 GPT 모델(OpenAI, 2023)은 크기가 더 큰 만큼 높은 도움됨을 보여주었으며, 실험 결과에 따르면 강화 학습이 도움됨을 향상시키는 동시에 해롭지 않음을 유지하는데 기여했습니다. 이러한 결과는 LLM이 개인의 주관적 경험을 이해하고 조언하는 능력이 향상되었음을 시사합니다.



### Sharing Parameter by Conjugation for Knowledge Graph Embeddings in  Complex Spac (https://arxiv.org/abs/2404.11809)
Comments: 8 pages, 1 figure, 6 tables, accepted at TextGraphs-16 workshop held in conjunction with COLING 2022

- **What's New**: 이 논문에서는 지식 그래프 임베딩(Knowledge Graph Embedding, KGE) 모델의 메모리 효율성을 개선하기 위한 새로운 방법을 제안합니다. 특히, 복소수(complex numbers)를 활용한 KGE 모델에서 공약 파라미터(conjugate parameters)를 공유하는 방식을 도입하였습니다. 이는 관계 임베딩(relation embedding)의 메모리 사용량을 기존 모델 대비 2배 감소시키면서도, 성능은 기존의 비공약(non-conjugate) 모델과 비교하여 경쟁력 있는 결과를 보여주었습니다.

- **Technical Details**: 제안된 방식은 KGE 모델에서 복소수를 사용할 때, 각 변환 함수(transformation functions)의 적절한 차원에서 공약 파라미터를 형성하여 관계 파라미터(relation parameters)의 수를 감소시킵니다. 이로 인해 공간 복잡도(space complexity)가 O(ne*de + nr*dr/2)로 감소하게 됩니다. 연구는 5가지 벤치마크 데이터셋에서 두 가지 성능이 뛰어난 KGE 모델, $5^{igstar}	ext{E}$와 $	ext{ComplEx}$를 사용하여 일반화 가능성을 입증하였습니다.

- **Performance Highlights**: 이 방법은 관계 임베딩의 메모리 효율성을 두 배 향상시키며, 훈련 시간도 기존 모델과 비교하여 빠르거나 최소한 비슷한 수준을 유지합니다. 결과적으로, 제안된 방식은 지식 그래프의 규모 확장성을 개선하고, 실제 애플리케이션에서의 메모리 부담을 상당히 완화할 수 있는 가능성을 보여줍니다.



### Enhancing Argument Summarization: Prioritizing Exhaustiveness in Key  Point Generation and Introducing an Automatic Coverage Evaluation Metric (https://arxiv.org/abs/2404.11793)
Comments: NAACL 2024 Main Conference

- **What's New**: 본 논문에서는 서로 다른 견해를 가진 사람들 사이의 종종 잘 다루어지지 않는 온라인 토론과 토론에서 'Key Point Generation (KPG)'에 초점을 맞추고, 첨단 기술을 활용한 새로운 추출 방식의 접근법을 소개합니다. 이 방법은 이전의 최신 기술보다 우수한 성능을 보이며, 키 포인트(Key Point) 생성을 통한 효과적인 요약을 가능하게 합니다.

- **Technical Details**: 이 연구의 핵심은 'extractive clustering' 접근 방식을 사용하여 중요한 키 포인트를 생성하는 것입니다. 이는 의미 기반 클러스터링을 통해 유사한 주장을 그룹화하고 매칭 모델을 사용하여 클러스터를 가장 잘 대표하는 주장을 찾습니다. 이렇게 생성된 요약은 덜 인기 있는 키 포인트까지도 포함할 수 있으며, 필터링 단계가 필요 없어 처리 과정을 개선합니다.

- **Performance Highlights**: 제안된 모델은 기존 방법보다 더 적은 중복성과 더 높은 범위를 가진 고품질의 요약을 생성함으로써 성능이 뛰어납니다. 또한, 전통적인 'ROUGE'와 같은 요약 평가 메트릭스가 KPG 작업을 평가하는 데 한계가 있음을 보여주고, 새로운 평가 척도를 제안하여 실제 키 포인트 커버리지와 더 상관관계가 높은 결과를 제공합니다.



### REQUAL-LM: Reliability and Equity through Aggregation in Large Language  Models (https://arxiv.org/abs/2404.11782)
- **What's New**: 이 연구에서는 REQUAL-LM을 도입하여 LLM (Large Language Models)의 신뢰성과 공정성을 개선합니다. LLM의 출력이 데이터의 내재된 편견과 역사적인 고정관념으로부터 영향을 받는 문제를 해결하기 위해, REQUAL-LM은 재샘플링(Monte Carlo method) 기반의 방법론을 사용하여 다수의 출력 중에서 가장 신뢰할 수 있는 결과를 선택합니다. 특히, 이 방법은 효율적이며 별도의 하드웨어나 복잡한 계산 부담 없이 LLM을 '블랙박스(blackbox)'로 사용하며, 다양한 LLM에 적용 가능합니다.



### Language Models Still Struggle to Zero-shot Reason about Time Series (https://arxiv.org/abs/2404.11757)
- **What's New**: 이 연구는 시계열 데이터(time series data)를 언어 모델(language model, LM)로 처리하고 이해할 수 있는지 분석한 최초의 평가 프레임워크(evaluation framework)를 개발했습니다. 여러 세부 분야에서 시계열 데이터와 텍스트 캡션을 짝지어 구성된 새로운 데이터셋과 형식적인 과제를 포함하여 언어 모델의 이유 추론(reasoning) 능력을 평가합니다.

- **Technical Details**: 이 연구는 세 가지 주요 추론 형태를 평가합니다. 첫 번째는 '원인 추론(Etiological Reasoning)'으로, 주어진 시계열 데이터로부터 가장 가능성 있는 시나리오를 식별합니다. 두 번째는 '질문 응답(Question Answering)'으로, 시계열 데이터에 관한 사실적 질문에 답할 수 있는지를 조사합니다. 마지막으로 '맥락 보조 예측(Context-Aided Forecasting)'은 관련 텍스트 맥락이 시계열 예측을 개선하는지 여부를 검토합니다.

- **Performance Highlights**: 연구 결과에 따르면, 인간은 언어 모델보다 월등히 높은 성능을 보여주었으며, 특히 GPT-4와 같은 강력한 언어 모델조차 무작위 선택 수준의 성능을 넘지 못하는 경우가 많았습니다. 원인 추론과 질문 응답 과제에서 모델은 무작위 선택보다 약간 나은 정도로 밖에 수행하지 못했으며, 맥락 보조 예측에서는 약간의 성공을 보였지만 이는 여전히 제한적이었습니다.



### Mapping Violence: Developing an Extensive Framework to Build a Bangla  Sectarian Expression Dataset from Social Media Interactions (https://arxiv.org/abs/2404.11752)
- **What's New**: 남아시아의 온라인 포럼에서 공동체 간 폭력이 급증하고 있으며, 다양한 문화의 커뮤니티들이 자원을 공유하면서 갈등이 자주 발생합니다. 이러한 문제에 대응하기 위해, 연구팀은 온라인 벵골어(Bangla) 콘텐츠에서 공동체 간 폭력 표식을 자동으로 감지하는 최초의 포괄적 프레임워크를 개발하였고, 13,000개의 원문장(raw sentences) 데이터셋을 구축했습니다. 이 데이터셋은 네 가지 주요 폭력 유형과 각각의 16가지 거친 표현들을 포함하고 있습니다.

- **Technical Details**: 이 연구에서는 사회 과학자, 언어학자, 심리학자들의 통찰력을 활용하여 7단계 전문가 주석 과정(expert annotation process)을 도입하였습니다. 이를 통해 벵골어(Bangla) 텍스트에서 비공동체 폭력(Non-communal violence)을 제외하고, 종교 공동체 간 폭력(Religio-communal violence)이 특히 만연하다는 것을 밝혀냈습니다.

- **Performance Highlights**: 연구팀은 최신 벵골어 딥 러닝 모델(Bangla deep learning model)을 사용하여 폭력적 코멘트를 식별하는 데 있어 언어 모델의 미세 조정(fine-tuning)의 효과를 검증하였고, 이 프레임워크의 벤치마킹 성과도 제시했습니다. 초기 벤치마킹 결과에서 이 데이터셋을 이용한 폭력 감지가 상당히 유효함을 입증했습니다.



### Missed Connections: Lateral Thinking Puzzles for Large Language Models (https://arxiv.org/abs/2404.11730)
Comments: 8 pages, 3 figures

- **What's New**: 이 연구는 New York Times에서 매일 발행되는 Connections 퍼즐을 자동 AI 시스템이 플레이할 수 있는지 살펴보고 이 게임을 추상적 사고(abstract reasoning)와 언어 시스템에서의 의미 정보(semantic information) 측정을 위한 자동 벤치마크로서의 잠재력을 탐구합니다. 특히, 문장 임베딩(sentence-embedding) 기반 접근 방식과 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 퍼즐 해결 능력을 조사하였습니다.

- **Technical Details**: 연구팀은 BERT, RoBERTa, MPNet, MiniLM 등 여러 SentenceTransformers 라이브러리의 임베딩 모델을 사용했습니다. 각 단어 그룹의 평균 쌍별 코사인 유사도(cosine similarity)를 계산하여 가장 높은 그룹을 반환하는 방식으로 추측을 생성합니다. 또한, LLMs에 대해 chain-of-thought prompting를 사용하여 그 영향을 측정하고, 더 도전적인 퍼즐 변형을 검토하여 기본 퍼즐보다 훨씬 어려운 버전을 구현했습니다.

- **Performance Highlights**: 연구 결과, 이 접근 방식들은 퍼즐을 상당 부분 시간 동안 해결할 수 있었으나, 정확도는 완벽하지 않았습니다. 도전적인 퍼즐 변형에서는 모든 그룹을 동시에 제출해야 하며, 올바른 그룹만이 피드백으로 제공되어 기본 게임보다 훨씬 더 어려운 것으로 나타났습니다. 이 성능 격차는 현대 시스템이 의미 정보를 인코딩하고 검색하는 데 성공과 실패를 연구하는데 풍부한 기회를 제공합니다.



### Investigating Gender Bias in Turkish Language Models (https://arxiv.org/abs/2404.11726)
Comments: arXiv admin note: text overlap with arXiv:1903.10561 by other authors

- **What's New**: 이 연구에서는 특히 터키어와 같이 문법적으로 성 중립적인 언어에서 성 편견(gender bias)의 중요성을 조사합니다. 이는 기존의 성 편견 연구가 주로 영어에 집중되어 있었던 것과 대비됩니다. 또한 터키어 모델들의 쿠르드인에 대한 인종적 편견도 평가하였습니다.

- **Technical Details**: 저자들은 기존의 bias 평가 프레임워크를 확장하여 터키어로 번역하고 새로운 테스트를 만들어 터키어(context of Türkiye)에서의 성 편견을 측정합니다. 주로 사용된 평가 도구로는 WEAT(Word Embeddings Association Test)이며, 이는 단어 임베딩 사이의 코사인 유사도(Cosine Similarity)를 사용합니다. 또한, 여러 터키어 모델들을 평가했으며, 그 중에는 multilingual BERT, BERTurk, mT5가 포함됩니다.

- **Performance Highlights**: 연구 결과에 따르면 모델의 크기, 다국어 처리 능력(multilingualism), 그리고 훈련 데이터의 성질이 편견 생성에 영향을 미치는 것으로 나타났습니다. 이 연구는 터키어 성 편견 데이터셋을 공개하였으며, 이는 다른 연구자들이 이를 활용하여 추가적인 연구를 할 수 있게 합니다.



### How often are errors in natural language reasoning due to paraphrastic  variability? (https://arxiv.org/abs/2404.11717)
Comments: accepted to TACL 2024 (pre-MIT Press publication version)

- **What's New**: 이 연구에서는 자연 언어 추론 모델의 패러프레이직 일관성 (paraphrastic consistency)을 평가하는 새로운 지표를 제안합니다. 이 지표는 동일한 문제에 대한 두 가지 패러프레이즈 (paraphrases)가 모델에 의해 동일하게 해결될 확률을 기반으로 합니다. 연구팀은 ParaNLU라는 새로운 데이터셋을 구축하여 이 지표를 사용하여 다양한 모델의 성능을 분석했습니다.

- **Technical Details**: ParaNLU 데이터셋은 기존 벤치마크 데이터셋 위에 구축된 7,782개의 인간 작성 추론 문제와 그 패러프레이즈를 포함하고 있습니다. 이 데이터셋을 사용하여 연구팀은 다양한 자연 언어 추론(NLI) 모델의 패러프레이직 일관성을 측정하였습니다. 이 연구는 세밀한 튜닝(finetuning)보다는 사전 학습(pretraining)이 모델의 패러프레이직 일관성을 향상시키는 데 더 효과적임을 보여줍니다.

- **Performance Highlights**: 연구 결과에 따르면, 모든 테스트된 모델들은 패러프레이직 일관성에서 개선의 여지가 있습니다. 특히, 사전 학습을 거친 모델들은 미세 조정을 거친 모델들보다 일관성이 크게 향상되었습니다. 하지만 아직 고정밀도와 높은 패러프레이직 일관성을 동시에 달성한 모델은 없습니다.



### Improvement in Semantic Address Matching using Natural Language  Processing (https://arxiv.org/abs/2404.11691)
Comments: 5 pages, 7 tables, 2021 2nd International Conference for Emerging Technology (INCET)

- **What's New**: 이 논문에서는 주소 매칭(address matching)의 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 기존의 문자열 유사성에 기반한 방법과 달리, 의미적 유사성(semantic similarity)을 평가하여 주소를 매칭하는 기법을 사용합니다. 이는 깊은 자연어 처리(NLP) 및 딥 러닝(deep learning) 기술을 사용하여 주소 데이터의 구조적이지 않거나 불완전한 문제를 효과적으로 극복합니다.

- **Technical Details**: 이 방법은 첫 번째로 주소 정보를 수집하고 OCR(optical character recognition)을 사용하여 주소를 추출합니다. 추출된 데이터는 BM-25 알고리즘으로 초기 평가를 하여 가장 유사한 주소 후보를 선정한 후, BERT(Bidirectional Encoder Representations from Transformers) 모델을 사용하여 최종적인 주소 매칭을 수행합니다. BM-25는 로그 기간과 용어 질의 빈도를 고려하여 각 문서에 점수를 매기는 반면, BERT는 문맥적 이해를 바탕으로 보다 정교한 유사성 평가를 제공합니다.

- **Performance Highlights**: 이 연구는 기존의 주소 매칭 방법들과 비교하여 상당한 정확도와 재현율의 향상을 이뤘습니다. 특히, 구조화되지 않았거나, 복잡한 문법을 가진 주소 정보에서도 높은 매칭 성능을 보여, 실제 비즈니스 환경에서의 적용 가능성을 크게 향상시켰습니다.



### How Well Can You Articulate that Idea? Insights from Automated Formative  Assessmen (https://arxiv.org/abs/2404.11682)
- **What's New**: 이 연구는 과학 설명 에세이에 대한 자동화된 피드백의 효과를 분석합니다. 특히, 물리학의 에너지와 질량에 대해 설명하는 에세이를 작성한 중학생들을 대상으로, 수정된 에세이의 질을 향상시키기 위해 제공된 자동 피드백의 정확성을 조사합니다.

- **Technical Details**: 자동화된 피드백 시스템은 PyrEval (PyrEval) 소프트웨어 패키지를 사용하여 구현되었습니다. 이 시스템은 rubric (루브릭)을 기반으로 하여 학생들의 에세이에서 주요 아이디어를 식별합니다. 벡터화 방법(vectorization method)을 선택하고, 학생 에세이 데이터를 분석하여 코사인 유사도(cosine similarity)를 측정함으로써 에세이에서 주요 아이디어의 존재 여부를 판단합니다. 본 연구는 자동화된 도구의 결정 과정을 추적하여 학생의 진술이 충분한 명확성을 가지고 있지 않을 때 이를 진단하고, 이러한 정보를 교사와 동료에게 제공하여 학생이 자신의 아이디어를 더 명확하게 표현할 수 있도록 돕습니다.

- **Performance Highlights**: 학생들은 대체로 자동 피드백을 통해 수정된 버전의 에세이에서 개선을 보였습니다. 그러나 주요 아이디어의 구별성(distinctiveness)과 학생들의 진술 명확성(clarity)은 피드백의 정확성에 영향을 미치는 주요 요소로 밝혀졌습니다. 따라서 일부 아이디어는 보다 일관된 표현을 사용함으로써 더 정확하게 감지되었습니다. 예를 들어, 에너지 보존 법칙은 공식적인 표현 덕분에 더 높은 정확성을 보였습니다.



### MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory (https://arxiv.org/abs/2404.11672)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)에 구조화되고 명시적인 읽기 및 쓰기 메모리 모듈을 통합하여 성능을 향상시키는 새로운 방법인 MemLLM을 소개합니다. 이 메모리 모듈은 LLM이 저장된 지식을 사용하는 능력을 개선하며, 자주 발생하지 않는 지식 및 시간에 따른 지식의 열화 문제를 해결하는 데 도움이 됩니다. 또한, MemLLM은 기존의 RAG(Retrieval Augmented Generation) 방식의 한계를 극복하고, 메모리의 구조화를 통한 효율적인 지식 관리와 해석 가능성을 제공합니다.

- **Technical Details**: MemLLM은 LLM에 명시적 메모리 컴포넌트를 추가하여 언어 모델이 텍스트를 처리하거나 사용자와 상호작용하는 동안 정보를 메모리에 저장하고 필요할 때 검색할 수 있도록 합니다. 이 메모리는 데이터베이스 스키마와 유사한 명시적 구조를 가지며, 인간에 의해 해석 및 편집이 가능하고 스케일이 용이합니다. LLM은 메모리에 접근하기 위해 API를 통한 읽기 및 쓰기 명령을 사용하고, 이러한 인터페이스를 위한 트레이닝 데이터셋을 제공하여 어떤 언어 모델도 명시적 메모리 기능을 통해 파인튜닝할 수 있습니다.

- **Performance Highlights**: MemLLM은 DOCRED 데이터셋에서 기준 모델 대비 낮은 혼란도(perplexity)를 보이며, 특히 명명된 개체에 대한 성능 향상을 보여 줍니다. 이는 MemLLM이 메모리 컴포넌트를 활용해 지식 중심 작업에서 기존 LLM들이 겪는 문제들을 우수하게 해결할 수 있음을 시사합니다.



### BLINK: Multimodal Large Language Models Can See but Not Perceiv (https://arxiv.org/abs/2404.12390)
Comments: Multimodal Benchmark, Project Url: this https URL

- **What's New**: 새로운 멀티모달 언어 모델(LLMs) 벤치마크 'Blink'가 소개되었습니다. Blink는 다른 평가에서는 찾아볼 수 없는 핵심 시각 인식 능력에 초점을 맞추고 있습니다. Blink는 컴퓨터 비전의 14가지 클래식한 태스크를 3,807개의 다중 선택형 문제로 재포매팅하며, 하나 또는 여러 이미지와 시각적 프롬프팅을 사용합니다.

- **Technical Details**: Blink는 시각적 해당 관계, 포렌식 감지(forensics detection), 멀티뷰 추론(multi-view reasoning)과 같이 인간이 '한순간에' 해결할 수 있는 태스크로 구성되어 있습니다. 그러나 현재의 멀티모달 LLMs는 이러한 시각 인식을 요구하는 작업에 상당한 도전을 받고 있으며, 자연 언어를 통한 중재에 저항합니다.

- **Performance Highlights**: 인간은 평균 95.70%의 정확도를 달성하는 반면, Blink는 기존 멀티모달 LLMs에게 놀랍게도 도전적인 과제였습니다. 가장 성능이 좋은 GPT-4V와 Gemini는 각각 51.26%와 45.72%의 정확도를 달성했습니다. 이는 무작위 추측(random guessing)보다 각각 13.17%와 7.63% 높은 수치에 불과합니다. 이는 최신 멀티모달 LLMs에서 해당 인식 능력이 아직 '등장하지 않았음(emerged)'을 시사합니다. 전문 컴퓨터 비전(CV) 모델이 이 문제들을 훨씬 더 잘 해결할 수 있음을 나타내며, 미래 개선을 위한 잠재적인 경로를 제시합니다.



### FedEval-LLM: Federated Evaluation of Large Language Models on Downstream  Tasks with Collective Wisdom (https://arxiv.org/abs/2404.12273)
Comments: In Progress

- **What's New**: FedEval-LLM은 협력적인 학습을 위해 설계된 분산 학습(FL, Federated Learning) 환경에서 대규모 언어 모델(LLM, Large Language Models)의 성능을 평가하는 새로운 프레임워크입니다. 이 프레임워크는 레이블이 붙은 테스트 세트나 외부 도구에 의존하지 않고, 참가자들의 지역 데이터와 최고 품질의 평가 데이터를 활용하여 하위 태스크에 대한 신뢰성 있는 성능 측정을 제공합니다. FedEval-LLM은 개인화된 평가 모델을 사용하여 도메인 지식을 제공하고, 다수의 심판 모델을 통해 평가 능력을 향상시킵니다.

- **Technical Details**: FedEval-LLM 프레임워크는 참여자들의 개인화된 LLM을 활용하여 도메인 지식을 통합하고, 평가 기준을 각 하위 태스크에 맞춰 조정합니다. 이는 로컬 데이터를 사용하여 고도의 개인화된 평가 데이터를 부트스트랩하여, 평가 모델들이 특정 평가 기준을 따를 수 있도록 합니다. 또한, 이 프레임워크는 다중 심판 모델을 도입하여, 참가자들의 지역적 관점을 종합적으로 반영하고, 특정 심판에 대한 의존성과 편향을 줄여줍니다.

- **Performance Highlights**: FedEval-LLM은 하위 태스크에서 우수한 평가 성능을 보여주었으며, 높은 RougeL 점수와 인간의 선호도와 강한 일치성을 나타냈습니다. 8개의 참가자 클라이언트가 포함된 연합 학습 시나리오에서 테스트한 결과, 기존의 로컬 평가 모델과 비교하여 경쟁력 있는 성과를 보였습니다. 이는 전통적인 평가 방법과 외부 서비스에 대한 의존성을 극복하는 데 효과적임을 시사합니다.



### Aligning language models with human preferences (https://arxiv.org/abs/2404.12150)
Comments: PhD thesis

- **What's New**: 이 논문은 언어 모델(LMs)이 사람의 선호도를 반영하도록 조정하는 새로운 접근 방법들을 탐색합니다. 특히, 전통적인 강화 학습을 기반으로 한 조정 방법과는 다른 방법들을 제안하여 언어 모델의 윤리적 측면과 사회적 편향 문제를 해결하고자 합니다.

- **Technical Details**: 저자는 기본 언어 모델(base, pretrained LM)을 사람의 선호도에 관한 증거와 조건을 부여하는 베이지안 추론(Bayesian inference)으로 조정하는 과정을 설명합니다. 이를 토대로, 강화 학습을 통한 피드백(Reinforcement Learning from Human Feedback, RLHF)과 분포 매칭(distribution matching)을 비교 분석하며, RLHF는 분포 매칭의 특수한 경우로 볼 수 있음을 주장합니다. 또한, 분산 매칭을 조건부 언어 모델에 확장하는 방법을 제시하고, 사전 학습(pretraining) 단계부터 인간의 피드백을 포함시키는 것이 단순히 감독 학습(supervised finetuning) 동안 사용하는 것보다 더 효과적임을 보여줍니다.

- **Performance Highlights**: 이 연구는 인간의 피드백을 사전 학습 단계에 포함시킴으로써 사회적 편향이나 거짓 정보의 문제를 미리 방지할 수 있음을 강조합니다. 특히, 분포 매칭을 적용하여 언어 모델이 사람의 선호도를 보다 정확하게 반영할 수 있는 기반을 마련합니다.



### Enhancing Suicide Risk Assessment: A Speech-Based Automated Approach in  Emergency Medicin (https://arxiv.org/abs/2404.12132)
- **What's New**: 이 연구는 긴급 상황에서 자살 위험을 평가하기 위해 비침습적, 음성 기반 접근 방식을 제시합니다. 이는 응급실에서 자살 경향이 있는 환자들에게 전문적인 정신건강 평가가 지연됨으로 인한 문제를 해결하기 위해 도입된 새로운 시도입니다. 특히, 음성 데이터와 환자의 메타데이터를 결합하여 자살 위험성을 평가하는 복합 모델을 개발하였습니다.

- **Technical Details**: 연구팀은 독일 아우크스부르크 대학의 정신과 응급실에서 20명의 환자로부터 음성 녹음 데이터를 수집했습니다. 이 데이터는 세 가지 유형의 음성 활동(짧은 이야기 읽기, 그림 설명, 모음 발음)으로 구성됩니다. 음성에서는 wav2vec, 해석 가능한 음성 및 음향 특성, 딥러닝 기반 스펙트럼 표현 등 세 가지 특징 세트를 추출합니다. 이 특성들을 기반으로 하여, leave-one-subject-out 방식으로 이진 분류를 수행하여 자살 위험을 평가합니다. 또한, 환자의 과거 자살 시도 이력이나 총기 접근성 등의 메타데이터를 통합함으로써 모델의 성능을 향상시켰습니다.

- **Performance Highlights**: 음성 기반 모델만 사용했을 때의 균형 정확도는 66.2%였으며, 환자의 메타데이터를 결합한 경우 균형 정확도는 94.4%로 상당히 향상되었습니다. 이는 단독 음성 모델 대비 28.2%의 절대적인 개선을 나타냅니다. 이러한 결과는 음성 데이터와 메타데이터의 결합이 자살 위험 평가의 정확도를 높일 수 있음을 보여줍니다.



### Ethical-Lens: Curbing Malicious Usages of Open-Source Text-to-Image  Models (https://arxiv.org/abs/2404.12104)
Comments: 42 pages, 17 figures, 29 tables

- **What's New**: 이 연구에서는 Ethical-Lens, 텍스트-이미지(text-to-image) 도구에 대한 새로운 윤리적 조정 프레임워크를 제안합니다. 이는 내부적 모델 개정 없이도 사용 가능하며, 독성(toxicity)과 편향(bias)과 같은 가치 조정을 보장할 수 있도록 설계되었습니다. Ethical-Lens는 사용자 명령을 정제하고 모델 출력을 수정하여 문제를 해결합니다.

- **Technical Details**: Ethical-Lens는 큰 언어 모델(LLM)에 의해 제공되는 Ethical Text Scrutiny와 CLIP 모델을 기반으로 한 Ethical Image Scrutiny로 구성되며, 이 두 가지 접근 방법은 각각 텍스트와 이미지 공간에서 윤리적 조정을 지원합니다. 또한, 시스템 평가(metric)는 GPT4-V, HEIM, 그리고 FairFace 점수를 조합하여 사용하며, 이는 모델의 가치 정렬 능력을 측정합니다.

- **Performance Highlights**: 실험 결과에 따르면, Ethical-Lens는 상업 모델인 DALLE 3와 같거나 더 우수한 수준의 가치 정렬 능력을 제공함을 확인하였습니다. 특히, Stable Diffusion와 같은 오픈 소스 도구 사용 시 Ethical-Lens를 통해 윤리적 기준에 부합하는 사용자 생성 콘텐츠를 생성할 수 있었으며, 이미지 품질을 유지하면서도 윤리적 정렬(ethical alignment) 능력에 뚜렷한 개선이 있었습니다.



### TIMIT Speaker Profiling: A Comparison of Multi-task learning and  Single-task learning Approaches (https://arxiv.org/abs/2404.12077)
- **What's New**: TIMIT 데이터셋을 이용하여 발화자 프로필링 작업을 위한 멀티태스크 학습과 싱글태스크 학습의 효과를 비교 연구한다. 특히, 성별 분류(Gender Classification), 연령 추정(Age Estimation), 방언 분류(Accent Classification), 그리고 발화자 식별(Speaker Identification) 네 가지 과제를 다룬다. 발화자 인식 작업에서 특징(JSON 영문명: feature) 공학의 중요성과, 시퀀셜(Sequential) 및 비시퀀셜(non-sequential) 특징에 따른 학습 효율의 차이점을 제시한다.

- **Technical Details**: 이 연구에서는 멀티태스크 학습(Multi-task Learning, MTL)과 싱글태스크 학습(Single-task Learning, STL)을 비교 분석한다. 멀티태스크 모델은 여러 관련 작업을 통해 일반화를 개선하기 위해 공유된 정보를 활용하며, MultiTask MLP와 MultiTask CNN+LSTM 모델을 실험한다. 특히 CNN+LSTM 모델은 MFCC(Mel Frequency Cepstral Coefficients) 특징을 처리하기 위해 설계되었으며, 오디오 시퀀스에서 스펙트럼 및 시간적 패턴을 모두 추출한다.

- **Performance Highlights**: 멀티태스크 모델은 비슷한 복잡성을 가진 작업에서 이점을 보여준다. 성별 분류 작업에서는 다양한 특징-모델 조합으로 유연성을 보였으며, 연령 추정에서는 CNN이 시퀀셜 MFCC 특징을 이용하여 가장 우수한 성능을 나타냈다. 방언 분류는 여전히 도전적인 과제로, 다양한 특징과 모델을 시도한 결과 일부 성공을 거두었다. 이러한 실험 결과는 MTL이 STL에 비해 일부 과제에 있어서 특정 측면에서 유리할 수 있음을 입증하지만, 모든 작업에 대해 일괄적으로 우세하다고 단정짓기는 힘들다. 결과적으로, MTL과 STL의 상황에 따른 적절한 사용이 중요하다.



### Enhancing Length Extrapolation in Sequential Models with  Pointer-Augmented Neural Memory (https://arxiv.org/abs/2404.11870)
Comments: Preprint

- **What's New**: Pointer-Augmented Neural Memory(PANM) 도입으로 신경망이 기호 처리(symbol processing)를 이해하고 새롭고 더 긴 데이터 시퀀스에 적용할 수 있게 되었습니다. PANM은 외부 신경 메모리를 통합하여 물리적 주소와 포인터 조작 기술을 사용하여 인간과 컴퓨터의 기호 처리 능력을 모방합니다. 이는 특히 알고리즘 이유화(algorithmic reasoning) 및 Dyck 언어 인식(Dyck language recognition)과 같이 기호 처리를 필요로 하는 작업에서 뛰어난 성능을 보입니다.

- **Technical Details**: PANM은 물리적 주소(physical addresses)를 명시적으로 모델링하고, input data로부터 포인터 조작을 엄격하게 분리하는 두 가지 원칙에 기반을 두고 있습니다. 메모리는 slot-based RAM 형태로서, 각 슬롯은 데이터와 주소로 구성됩니다. Pointer Unit이라는 신경망을 사용하여 초기 주소(address bank에서 제공)에서 포인터를 변환하고, 이를 통해 데이터에 접근하는 두 가지 모드(pointer dereference: Mode-1, relational access: Mode-2)를 제공합니다. PANM은 LSTM 또는 Transformer와 같은 일반적인 encoder-decoder 백본과 연동할 수 있습니다.

- **Performance Highlights**: PANM은 복잡한 학습(compositional learning) 작업에서 Transformer 모델의 일반화 능력을 향상시켜 최대 100%의 일반화 정확도를 달성했습니다. 또한, 수학적 추론(mathematical reasoning), 질문 답변(question answering), 기계 번역(machine translation) 작업에서도 뛰어난 결과를 보였습니다. 이는 PANM이 실제 시퀀스 및 기호 처리 작업에서 기존 딥 러닝 모델의 한계를 극복하고 더 좋은 일반화 능력을 제공할 수 있음을 보여줍니다.



### Advancing Speech Translation: A Corpus of Mandarin-English  Conversational Telephone Speech (https://arxiv.org/abs/2404.11619)
Comments: 2 pages

- **What's New**: 이 논문은 중국어(만다린)-영어 번역을 위한 대화형 말하기 데이터 corpus의 새로운 세트를 소개합니다. 외부벽 아닌 train, development, test set 등으로 세분화된 훈련 데이터의 필요성을 강조하며, 막강한 학습 자료인 CallHome Mandarin Chinese data와 HKUST Mandarin Telephone Speech data에 대한 영어 번역을 제공합니다.

- **Technical Details**: 관련 데이터는 총 123.5시간 분량의 중국어 대화 음성(CallHome 33.5시간, HKUST 90시간)으로 구성되어 있으며, 이는 두 데이터셋(CallHome, HKUST)에서 추출되었습니다. 번역은 Appen을 통해 Mandarin-English 이중 언어 번역자가 수행하였고, 전체 영역에 걸쳐 일관된 번역을 위해 다수의 annotators(본문 주석자)로부터 번역이 제공되었습니다. 실험에서는 ASR 시스템의 Decoder 출력을 MT 시스템의 입력으로 사용하는 Cascade speech translation systems를 활용하였고, 주요 ASR 모델로는 hybrid TDNN-LSTM을 사용하여 Sage 시스템에 의해 훈련되었습니다.

- **Performance Highlights**: 초기 NLLB 모델은 CTS test set에서 BLEU 점수가 5.98이었으나, CTS train set에 대한 fine-tuning 후 BLEU 점수가 137% 향상되었습니다. 이는 맥락이 잘 맞는(training data matched) 훈련 데이터가 대화형 스피치 번역 시스템 구축에 매우 중요하다는 것을 강조합니다.




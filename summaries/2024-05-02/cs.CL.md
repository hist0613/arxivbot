### Is Bigger Edit Batch Size Always Better? -- An Empirical Study on Model  Editing with Llama-3 (https://arxiv.org/abs/2405.00664)
- **What's New**: 이 연구는 최신 대형 언어 모델인 Llama-3를 대상으로 한 특정 모델 편집 분석을 제시합니다. 시퀀셜 편집(sequential editing), 배치 편집(batch editing) 및 우리가 순차-배치 편집(sequential-batch editing)이라고 부르는 혼합 접근 방식을 포함하여 다양한 전략을 통해 최대 4096개의 편집을 평가합니다. 이 결과는 배치 크기를 늘릴 때 모델 성능이 더 큰 폭으로 저하될 가능성이 있음을 시사하며, 이는 작은 배치를 순차적으로 사용하는 것이 동일한 편집 수량에 대해 더 효과적일 수 있음을 제안합니다.

- **Technical Details**: 이 연구에서는 ROME, MEMIT, EMMET와 같은 인기 있는 모델 편집 기술을 사용하여 Llama-3의 특정 레이어(layer)에 대한 개입을 분석합니다. 이를 통해 가장 효과적인 레이어를 찾아내고, 서로 다른 편집 전략들의 효과를 비교합니다. 배치 크기가 클수록 편집 효율성이 저하되는 경향이 확인되었으며, 이 연구는 순차적인 방식과 배치 방식을 조합하는 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: 본 연구는 소규모 배치를 사용하여 순차적으로 편집을 할 때 모델 성능이 개선되었다는 것을 발견했습니다. 큰 배치 크기로 편집할 때 발생하는 성능 저하는, 편집 방식을 최적화하고 성능 개선을 위한 향후 연구로 이어질 수 있는 중요한 관찰 결과로 제시되었습니다.



### NLU-STR at SemEval-2024 Task 1: Generative-based Augmentation and  Encoder-based Scoring for Semantic Textual Relatedness (https://arxiv.org/abs/2405.00659)
- **What's New**: 이 논문은 의미적 텍스트 관련성(Semantic Textual Relatedness, STR) 영역에서의 새로운 연구 결과를 제시하며, 특히 다양한 아랍 방언을 포함한 다국어 데이터셋에 대한 연구를 포함하고 있습니다. SemRel-2024 공유 작업에 참여하여 알제리와 모로코 방언(Track A)과 현대 표준 아랍어(MSA, Track B)에서의 성과를 보고합니다.

- **Technical Details**: 연구팀은 BERT 기반 모델을 활용하여 아랍어 방언의 데이터셋에 대해 미세 조정(fine-tuning)함으로써 의미적 관련성 점수를 회귀 분석(regression) 방식으로 산출했습니다. 또한, 지도 학습(supervised) 트랙에서는 데이터 증강(data augmentation)을 통해 다양한 문장 쌍을 추가 생성하여 모델의 성능을 강화하였고, 무지도 학습(unsupervised) 트랙에서는 코사인 유사도(cosine similarity)를 이용하여 평가를 수행했습니다.

- **Performance Highlights**: 이 시스템은 현대 표준 아랍어(MSA)에서 스피어먼 상관 관계(Spearman correlation) 0.49로 1위를 기록하였습니다. 모로코 방언에서는 0.83으로 5위, 알제리 방언에서는 0.53으로 12위를 기록하였습니다. 이 결과는 여러 아랍 방언에서 의미적 텍스트 관련성 평가 기술의 유효성을 입증하며, 다양한 언어와 방언에 대한 데이터셋의 중요성을 강조합니다.



### RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document  Abstractive Summarization (https://arxiv.org/abs/2405.00657)
Comments: NAACL 2024 Main & Long Conference Paper (Oral Presentation)

- **What's New**: 이 논문은 수사 구조 이론(Rhetorical Structure Theory, RST)을 장문 문서 요약에 효과적으로 통합하기 위한 새로운 전략을 제안합니다. RST-LoRA라는 모델을 도입하고, RST를 명시적으로 통합하는 네 가지 변형을 제안하여, 수사 관계의 형태와 불확실성이 LoRA 모델의 성능을 향상시킬 수 있음을 실험적으로 검증합니다.

- **Technical Details**: 이 연구는 기존 LoRA(Low-Rank Adaptation) 모델에 RST를 통합하여, 장문 문서 요약의 효율성과 정확성을 높이고자 합니다. 제안된 RST-LoRA 모델은 수사 구조를 활용해 문장 간의 중요도를 판단하고, 요약 생성 과정에서 핵심 내용을 더 잘 추출할 수 있도록 합니다. 모델은 Seq2Seq 및 GPT 변형기반(transformer-based) 구조와 호환됩니다.

- **Performance Highlights**: RST-LoRA는 기존의 LoRA 모델 및 전체 매개 변수를 미세 조정하는 모델들을 능가하는 성능을 보여주었습니다. 자동 및 인간 평가를 통해 검증된 결과, 이 모델은 이전의 최첨단 방법들보다도 우수한 결과를 달성하였으며, 요약의 사실 일관성 검사에서도 더 높은 점수를 얻었습니다. 이는 RST 지식을 통합한 접근 방식이 장문 문서 요약 작업에 있어 중요한 영향을 미칠 수 있음을 시사합니다.



### When Quantization Affects Confidence of Large Language Models? (https://arxiv.org/abs/2405.00632)
Comments: Accepted to NAACL 2024 Findings

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 양자화(quantization)가 모델의 신뢰도(confidence) 및 보정(calibration)에 미치는 영향을 조사합니다. GPTQ와 같은 후학습 양자화(post-training quantization) 방법을 사용하여 4비트로 양자화할 때 진정성 있는 레이블에 대한 신뢰도가 감소하는 것을 발견했습니다. 특히, 다양한 언어 모델과 규모에서 양자화의 영향이 달라지며, 신뢰도 손실(confidence loss)에 대한 새로운 설명을 제안합니다.

- **Technical Details**: 연구진은 여러 언어 모델과 규모에서 양자화가 정확도, 예측 신뢰도에 미치는 영향을 분석하고, 양자화된 모델(quantized models)과 전체 정밀도 모델(full-precision models) 간의 신뢰도 일치(confidence alignment)를 평가했습니다. 이를 위해, GPTQ 방법을 사용하여 학습된 가중치를 4비트로 줄였으며, C4 데이터셋에서 무작위로 선택된 128개의 시퀀스를 사용하여 양자화를 진행했습니다.

- **Performance Highlights**: 양자화 후 모델의 정확도(accuracy)와 보정 오류(calibration error)를 평가한 결과, 양자화는 특히 낮은 신뢰도에서 원래 모델이 예측한 샘플에서 더 큰 영향을 미치는 것으로 나타났습니다. 이는 양자화 과정에서 발생하는 신뢰도 손실을 설명하는 데 중요한 통찰을 제공합니다. 또한, 이러한 결과는 양자화가 모델의 성능을 유지하면서도 보정 수준을 보존하는 데 있어 일부 도전을 제기한다는 것을 보여줍니다.



### Causal Evaluation of Language Models (https://arxiv.org/abs/2405.00622)
Comments: 315 pages, 230 figures, 21 tables. Project website: this https URL

- **What's New**: 본 연구에서는 인간 수준의 기계 지능 달성에 필수적으로 간주되는 인과 추론(cause-and-effect reasoning) 능력을 평가하는 최초의 포괄적 벤치마크인 Causal evaluation of Language Models (CaLM)을 소개하고 있습니다. CaLM은 인과 추론 기능을 평가하기 위해 특별히 설계된 표준화된 도구 및 데이터셋 프레임워크를 제공합니다.

- **Technical Details**: CaLM 프레임워크는 인과 목표(causal target), 적응(adaptation), 지표(metric), 오류 분석(error analysis)의 네 가지 핵심 모듈로 구성되어 있으며, 이는 폭넓은 평가 디자인 공간을 정의합니다. 제안된 프레임워크는 126,334개의 데이터 샘플을 포함하는 CaLM 데이터셋을 구성하여 연구의 다양성을 지원합니다.

- **Performance Highlights**: 28개의 선도적인 언어 모델(language models)을 사용하여 92개의 인과 목표, 9가지 적응 사례, 7가지 평가 지표(metric), 12가지 오류 유형(error type)에 걸쳐 광범위한 평가를 수행하였습니다. 결과는 다양한 차원(예: 적응, 스케일)에서 상세하게 분석되었고, 9가지 차원의 50가지 고차원(empirical) 소견을 제시하여 향후 언어 모델 개발에 귀중한 가이드를 제공합니다.



### Addressing Topic Granularity and Hallucination in Large Language Models  for Topic Modelling (https://arxiv.org/abs/2405.00611)
- **What's New**: 이 연구에서는 큰 언어모델(Large Language Models, LLM)을 사용한 주제 모델링에 대해 기존의 확률적 주제 모델링 접근 방식과 달리 사람의 선호에 기반한 피드백으로 파인튜닝(Fine-tuning)하였습니다. 이를 통해 주제의 세밀함과 정확한 명명 그리고 환각(hallucination) 문제를 해결하기 위한 새로운 접근 방식을 소개했습니다. 주제 생성에 있어서 중복과 불필요한 주제의 축소에 초점을 맞추어, 재구성 파이프라인 및 직접적인 선호 최적화(Direct Preference Optimisation, DPO)를 이용한 학습 방법이 제안되었습니다.

- **Technical Details**: 개발된 시스템은 문서 집합에서 주제를 추출하는 데 LSTM과 같은 기존 모델이 아닌, Mistral-7B 같은 오픈 소스 대형 언어 모델을 사용합니다. 특히, 주제의 그래뉼라리티(Granularity)와 명명을 제어하기 위한 프롬프팅 전략(prompting strategies)이 사용되었습니다. 또한, 이러한 LLM 기반 시스템은 사람의 주석없이 효율적으로 학습할 수 있도록 재구성 파이프라인(reconstruction pipeline)을 통해 원시 주제를 수정합니다.

- **Performance Highlights**: 실험 결과, 파인튜닝된 LLM은 기존 LLM보다 더 일관성 있고 관련성 높으며 정밀한 주제를 생성할 수 있는 능력이 크게 향상되었음을 보여줍니다. 또한, 환각 주제의 수를 줄이는데도 효과적이었습니다. 추가적으로, 이 연구는 학습된 모델의 주제 추출 품질을 평가하기 위한 플러그 앤 플레이 평가 프로토콜(plug-and-play evaluation protocols)도 제시하고 있습니다.



### Investigating Automatic Scoring and Feedback using Large Language Models (https://arxiv.org/abs/2405.00602)
- **What's New**: 이 연구는 대형 언어 모델(LLM: Large Language Models)의 양자화된 모델을 사용하여 자동 등급 매기기 및 피드백 생성의 효과를 탐구합니다. LLaMA-2와 같은 최신 LLM을 이용한 전략은 기존의 머신러닝(ML: Machine Learning) 및 딥러닝(DL: Deep Learning) 기술보다 더 향상된 성과를 보여주고 있으며, 이번 연구는 특히 파라미터 효율적 파인튜닝(PEFT: Parameter Efficient Fine-tuning) 방법을 적용하여 계산 자원 사용을 크게 줄이는 것에 중점을 둡니다.

- **Technical Details**: 이 논문에서는 LoRA와 QLoRA와 같은 PEFT 기술을 사용하여 모델의 메모리 및 계산 요구 사항을 줄이면서 LLM을 세밀하게 조정합니다. 4비트 양자화 LLaMA-2 13B 모델을 사용하여 숫자 점수를 자동으로 할당하고 해당 피드백을 생성하는 방식으로 사용하였으며, 이는 회귀 또는 분류 작업에 적용됩니다. 연구는 고유한 데이터셋과 오픈소스 데이터셋에서 수행되었으며, 높은 BLEU 및 ROUGE 점수를 달성하고 주제별 전문가 피드백과 높은 유사성을 보여줍니다.

- **Performance Highlights**: 양자화된 LLM은 평균 3% 미만의 오류율로 등급 점수를 예측하는 데 매우 정확했으며, 4비트 양자화된 LLaMA-2 13B 모델은 기존의 기본 모델들을 능가하여 평가된 피드백을 제공하는데 있어 높은 성능을 보였습니다. 이는 감독 학습 및 인스트럭션 파인튜닝(Supervised Instruction Fine-tuning)이 LLM의 성능을 크게 향상시키는 데 기여한 것으로 보여집니다.



### Are Models Biased on Text without Gender-related Language? (https://arxiv.org/abs/2405.00588)
Comments: In International Conference on Learning Representations 2024

- **What's New**: 이 논문은 대규모 언어 모델의 성 편향 연구에 있어 새로운 시각을 제시합니다. 기존의 성 편향은 훈련 데이터에서 성별 관련 단어의 사용으로 인한 것으로 보았지만, 이 연구는 성별에 대한 직접적 언급이 없는 상황에서도 편향이 나타나는지를 조사합니다. 이를 위해 'UnStereoEval (USE)'라는 새로운 프레임워크를 도입하여, 스테레오타입이 없는 시나리오에서의 성 편향을 평가합니다.

- **Technical Details**: USE 프레임워크는 문장 수준에서 성별과 관련된 단어의 연관성을 최소화하는 벤치마크를 자동으로 생성하는 능력을 갖추고 있습니다. 이는 Pointwise Mutual Information (PMI) 기술을 활용하여 단어 수준에서 성별 연관성을 측정하고, 이를 종합한 문장 수준 점수로 성 편향을 평가합니다. 또한, 기존의 성 편향 벤치마크인 Winobias와 Winogender를 새 기준에 맞게 수정하여 사용합니다. 이 연구를 통해 28개의 언어 모델을 평가하였으며, 그 결과 대부분의 모델에서 스테레오타입이 제거된 시나리오에서도 낮은 공정성을 보임을 확인하였습니다.

- **Performance Highlights**: 평가된 28개 모델 중 어느 것도 스테레오타입이 없는 문장에서 41% 이상의 공정함을 보이지 않았습니다. 특히, 모델은 Winobias와 Winogender 벤치마크에서 남성 문장을 선호하는 경향을 보였습니다. 이는 LMs가 성 관련 언어의 직접적 사용 없이도 성별 편향을 나타낼 수 있음을 시사하며, 모델의 성 편향 근원에 대한 더 깊은 조사가 필요함을 강조합니다.



### The Real, the Better: Aligning Large Language Models with Online Human  Behaviors (https://arxiv.org/abs/2405.00578)
Comments: 11 pages, 6 figures

- **What's New**: 이 논문은 Reinforcement Learning with Human Behavior (RLHB)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 실제 온라인에서의 인간 행동을 직접 활용하여 대규모 언어 모델(Large Language Models, LLMs)을 정렬하는 방법을 다룹니다. 기존의 긴 학습 과정과 사전 정의된 선호도 편향을 극복하고자 하는 시도로, 다양한 인간의 선호에 적응할 수 있는 모델을 개발하고자 합니다.

- **Technical Details**: RLHB 프레임워크는 생성적 적대 신경망(generative adversarial framework)을 사용하여, 생성기(generator)는 기대되는 인간 행동을 따르는 응답을 생성하도록 훈련받습니다. 한편, 판별자(discriminator)는 쿼리, 응답, 그리고 인간 행동의 삼중 요소가 실제 온라인 환경에서 왔는지를 판별하려고 시도합니다. 자연 언어 형태로의 행동 모델링과 다중 모델 공동 학습 메커니즘은 지속 가능하고 적극적인 온라인 정렬을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 인간의 평가와 자동 평가 모두를 통해 제안한 방법의 효과를 확인하였습니다. 이 방법은 실제 온라인 환경에서의 인간 행동을 모델링하고, 이를 통한 LLM의 적절한 정렬에 성공적으로 기여하였음을 보여줍니다.



### Mixture of insighTful Experts (MoTE): The Synergy of Thought Chains and  Expert Mixtures in Self-Alignmen (https://arxiv.org/abs/2405.00557)
- **What's New**: 새로운 연구에서는 인간의 가치와 일치하는 대규모 언어 모델(Large Language Models, LLMs)의 자기조정(self-alignment) 방법론을 제시하고 있습니다. 이번 연구는 'Chain of Thought'(CoT) 접근 방식을 기반으로 'AlignCoT' 방법을 도입하여 질문 분석(Question Analysis), 답변 안내(Answer Guidance), 안전한 답변 생성(Safe Answer production)의 단계를 포함합니다. 또한, 각 단계를 강화하기 위해 전문가의 혼합을 적용하는 새로운 구조인 'Mixture of insighTful Experts'(MoTE)를 소개하며, 이는 기존 방법보다 우수한 성능을 보여 줍니다.

- **Technical Details**: AlignCoT은 모델이 발전하는 여러 단계에서 고품질의 안전한 응답을 생성할 수 있도록 설계되었습니다. 이 과정은 질문의 안전하지 않은 의도를 식별하고, 전략적으로 답변을 구성함으로써 대응합니다. MoTE 아키텍처는 'mixture of experts'(MoE)를 적용하여 각 구성 요소의 효율성을 높이고, 학습 데이터의 중복을 줄이며, 훈련 시간을 상당히 단축시키는 설계로 이루어져 있습니다. 이러한 접근 방식은 LLMs가 인간의 피드백 없이도 윤리적으로 타당한 답변을 독립적으로 생성할 수 있도록 합니다.

- **Performance Highlights**: MoTE 방법론은 기존의 자기 조정 기법(Supervised Fine-Tuning, SFT; Reinforcement Learning from Human Feedback, RLHF; Critique-Revise 및 Mistake Analysis)과 비교할 때 LLMs의 인간 가치와의 일치성에서 높은 효율성을 보여 주었습니다. 또한, 자체 생성 데이터(self-generated data)를 사용함으로써 학습 효율성이 크게 향상되었다는 이점을 강조하고 있습니다.



### New Benchmark Dataset and Fine-Grained Cross-Modal Fusion Framework for  Vietnamese Multimodal Aspect-Category Sentiment Analysis (https://arxiv.org/abs/2405.00543)
- **What's New**: 최근 소셜 미디어 플랫폼에서 다양한 모달리티(Multimodal) 데이터가 증가하면서 이를 통해 사용자들의 감성을 파악하는 데 큰 기회가 열렸습니다. ViMACSA 데이터셋과 Fine-Grained Cross-Modal Fusion Framework (FCMF)을 소개하여 호텔 분야에 대한 텍스트와 이미지 모두에서 미세 범주화된 주석을 포함하여 다양한 모달리티를 효과적으로 활용하는 새로운 방법을 제안합니다. 이 연구는 비트남어(Vietnamese)로 된 다중 모달 감성 분석에 특화된 것이 특징입니다.

- **Technical Details**: 이 연구에서는 텍스트-이미지 쌍 4,876개와 14,618개의 미세 주석이 포함된 ViMACSA 벤치마크 데이터셋을 제공하여 Aspect-Category Sentiment Analysis (ACSA)를 위한 토대를 마련합니다. 또한, FCMF 프레임워크는 내부 및 상호 모달리티 상호작용(inter-modality interactions)을 학습하고 이를 통합하여 통합된 다중 모달 표현을 생성합니다. 주목할만한 기술은 Attention Mechanisms를 활용하여 텍스트와 이미지 사이의 미세한 요소 간의 상호작용을 파악하는 것입니다.

- **Performance Highlights**: 연구 결과에 따르면 FCMF 프레임워크는 기존의 상태보다 높은 수행(State-of-the-Art, SOTA) 모델들을 능가하여 ViMACSA 데이터셋에서 최고 F1 점수 79.73%를 달성했습니다. 이로써 본 프레임워크는 다양한 비교 베이스라인을 위한 효과적인 지원 도구가 될 가능성을 보여줍니다.



### A Legal Framework for Natural Language Processing Model Training in  Portuga (https://arxiv.org/abs/2405.00536)
Comments: LEGAL2024 Legal and Ethical Issues in Human Language Technologies, LREC 2024

- **What's New**: 인공 지능(AI) 및 자연어 처리(NLP)의 최근 발전에 따라, 이와 관련된 법적, 윤리적 우려가 증가하고 있습니다. 본 논문은 포르투갈 법 체계 하에서 NLP 연구 및 개발에 수반될 수 있는 법적 문제를 조명하고, 이를 해결하기 위한 다학제적 접근을 제안합니다. 특히, 포르투갈 법은 EU 법의 영향을 받기 때문에 다른 EU 국가들에도 유용한 정보를 제공할 것입니다.

- **Technical Details**: 이 논문은 첨단 NLP(Large Language Models, LLMs) 모델들이 생성하는 성능과 이들이 직면한 법적, 윤리적 도전을 조명합니다. 여기서 분석된 주요 모델로는 ChatGPT와 같은 최신 generative models가 있습니다. 데이터 프라이버시, 저작권 침해 및 인공 지능의 부정적 사용에 대한 우려가 초점을 이룹니다. 포르투갈의 법적 맥락을 설명하고, NLP 연구에서의 리스크 관리 및 컴플라이언스(compliance) 필요성을 강조합니다.

- **Performance Highlights**: 기존의 NLP 모델과 비교하여 최신 LLM들은 상당히 향상된 텍스트 생성 능력을 보여 주었지만, 이러한 성능 향상은 높은 훈련 비용과 방대한 텍스트 데이터의 필요성을 수반합니다. 이와 동시에, EU 내 GDPR(General Data Protection Regulation) 같은 법령과 AI Act의 역할을 중심으로 NLP에 대한 새로운 규제 접근 방식이 필요함을 강조합니다.



### Is Temperature the Creativity Parameter of Large Language Models? (https://arxiv.org/abs/2405.00492)
Comments: To be published in the Proceedings of the 15th International Conference on Computational Creativity (ICCC'24), 8 pages, 2 figures, 2 tables

- **What's New**: 이 연구는 큰 언어 모델(LLMs: Large Language Models)이 창의성을 발휘하는 방식과 온도 매개변수가 스토리텔링의 창의성에 어떻게 영향을 미치는지에 대해 탐구합니다. 온도가 높아질수록 소설적 출력이 약간 증가한다는 결과를 발견하였지만, 창의성 매개변수로서 온도의 영향은 예상보다 훨씬 미묘하고 약합니다.

- **Technical Details**: 연구진은 고정된 상황, 모델, 프롬프트를 사용하여 이야기 생성 작업에서 온도 값이 다양한 LLM 출력을 실증적으로 분석했습니다. 연구는 노벨티(Novelty), 타이피칼리티(Typicality), 코히전(Cohesion), 그리고 코히런스(Coherence)의 네 가지 창의성에 필요한 조건을 사용했습니다. LLM은 Llama 2-Chat 모델을 사용하며, 창의성 평가는 인간 참가자를 통해 수행되었습니다.

- **Performance Highlights**: 온도가 증가함에 따라 노벨티는 약한 긍정적 상관관계를 보이며, 코히런스와는 부적 상관관계가 있음을 관찰하였습니다. 그러나 코히전과 타이피칼리티와는 관련이 없음이 밝혀졌습니다. 이 결과는 온도가 창의적 결과를 생성하는 데 중요한 역할을 한다기보다는, 더 복잡한 다차원적 접근이 필요함을 시사합니다.



### Harnessing the Power of Multiple Minds: Lessons Learned from LLM Routing (https://arxiv.org/abs/2405.00467)
Comments: Accepted to Workshop on Insights from Negative Results in NLP 2024 (co-located with NAACL 2024)

- **What's New**: 이 연구는 입력 쿼리를 가장 적합한 단일 LLM(Large Language Model, 대형 언어 모델)에 직접 라우팅하는 LLM 라우팅 모델을 제안합니다. 이 모델은 LLM들 간의 상이한 전문 지식을 활용하여 입력 쿼리에 가장 적합한 모델을 선택함으로써 전체적인 성능을 향상시키고자 합니다. 또한, 이 연구는 분류(classification)와 군집화(clustering) 두 가지 접근 방식을 통해 LLM 라우팅을 다룹니다.

- **Technical Details**: 제안된 LLM 라우팅 모델은 트랜스포머(Transformer) 인코더 모델을 미세조정하여 각 입력 쿼리에 대해 최적의 LLM을 선택합니다. 이 모델은 GSM8K와 MMLU와 같은 도전적인 추론 과제 벤치마크에서 테스트되었습니다. 분류 기반 라우팅은 여러 LLM의 성능을 예측하고, 군집화 기반 접근 방식은 쿼리 간의 유사성을 기반으로 합니다.

- **Performance Highlights**: 실제 실험에서 이론적 상한선과 비교했을 때, 제안된 라우팅 모델의 성능은 개별 모델보다는 높지만 가장 성능이 우수한 LLM들과 비슷하거나 약간 낮은 결과를 보였습니다. 그러나 이 연구는 LLM 라우팅 모델링의 실행 가능성을 보여주었으며, 효율적인 LLM 사용에 대한 새로운 연구 방향을 제시합니다.



### BiomedRAG: A Retrieval Augmented Large Language Model for Biomedicin (https://arxiv.org/abs/2405.00465)
- **What's New**: BiomedRAG는 기존의 검색 강화 언어 모델(retrieval-augmented language models)과 다르게 검색된 문서를 직접 LLM에 입력하는 간단한 접근 방식을 채택합니다. 이 모델은 특히 정보가 많이 포함된 태스크에서 검색된 문서의 잡음 정보를 효과적으로 우회하며, 생물의학(biomedical) 분야에서 LLM을 사용하여 검색 모델을 감독하고 LM의 예측을 향상시키는 데 도움을 주는 문서를 검색하도록 합니다.

- **Technical Details**: BiomedRAG는 특수한 교차주의 메커니즘(cross-attention mechanisms)을 사용하는 대신, 검색된 덩어리 기반 문서(chunk-based documents)를 LLM에 직접 입력하는 방식을 사용합니다. 이러한 설계는 기존의 검색 및 언어 모델에 쉽게 적용될 수 있습니다. 또한, 생물의학 분야에서 LLM을 사용하여 검색 모델을 감독하는 잠재력을 보여주며, 이는 검색 모델이 LM의 예측을 개선하는 데 도움이 되는 문서를 검색하도록 합니다.

- **Performance Highlights**: BiomedRAG는 조율된 점수판(tuned scorer)을 통해 9개 이상의 데이터셋을 활용하여 5가지 생물의학 NLP 태스크(biomedical NLP tasks)에서 우수한 성능을 달성합니다. 예를 들어, 트리플 추출(tiple extraction) 태스크에서는 GIT와 ChemProt 코퍼스에서 각각 81.42와 88.83의 마이크로-F1 스코어(micro-F1 scores)로 다른 트리플 추출 시스템을 능가합니다.



### Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models (https://arxiv.org/abs/2405.00402)
- **What's New**: 이 논문에서는 작은 언어 모델(Small Language Models, SLMs)이 스스로 능력을 세련화(Refine)할 수 있도록 돕는 새로운 방법인 'Self-refine Instruction-tuning'을 제안합니다. 이 방법은 기존의 Supervised Fine-Tuning (SFT) 및 Instruction-tuning과는 다르게, 큰 언어 모델(Large Language Models, LLMs)에서 제공한 데모를 기반으로 첫 단계에서는 지시 튜닝을 수행하고, 두 번째 단계에서는 선호 최적화(Preference Optimization) 전략으로 모델의 능력을 세련화합니다.

- **Technical Details**: 우리의 접근법은 두 단계로 구성됩니다. 첫 번째 단계에서는 LLM에서 SLM으로 CoT(Chain-of-Thought) 추론 능력을 전달합니다. 이는 LLM이 생성한 데모스트레이션(파라미터 이름, 문제, 기대 결과 등을 포함한 튜플)을 기반으로 지시 튜닝(Instruction-tuning)을 통해 이루어집니다. 두 번째 단계에서는 '직접 선호 최적화(Direct Preference Optimization, DPO)' 알고리즘을 활용, 이전 단계에서 지시된 모델이 자가 생성된 반응을 샘플링하고, LLM에서 제공하는 ground truths를 사용하여 보상을 제공함으로써 자체 CoT 추론 모델을 세련화(self-refine)합니다.

- **Performance Highlights**: 실험 결과, 이 새로운 접근법은 기존의 Instruction-tuning 방식보다 상당히 우수한 성능을 보였으며, 특히 상식 및 수학 추론 과제에서 in-domain 및 out-domain 시나리오 모두에서 강력한 일반화 능력을 보여주었습니다. 다양한 크기의 LLMs와 SLMs 조합을 사용하여 강화된 학습법의 효과를 검증하였고, 기존의 교육-학생(teacher-student) 모델 정렬 방법과 비교하여 성능 차이를 극복하고 효율성을 최대화했습니다.



### CofiPara: A Coarse-to-fine Paradigm for Multimodal Sarcasm Target  Identification with Large Multimodal Models (https://arxiv.org/abs/2405.00390)
Comments: 25 pages, 7 figures, and 18 tables

- **What's New**: 이 논문은 멀티모달 (Multimodal) 비꼬기 표적 식별 (MSTI)을 위한 새로운 프레임워크를 제안합니다. 기존의 방법들이 텍스트와 이미지 모두를 통한 비꼬기의 미묘한 이해를 간과하는 반면, 이 연구는 이를 극복하기 위해 대규모 멀티모달 모델(LMMs)의 이점을 활용하여 세밀한 (fine-grained) 비꼬기 표적 식별을 진행합니다.

- **Technical Details**: 제안된 프레임워크는 큰 거친 (Coarse) 단계와 세밀한 (Fine) 단계의 패러다임을 결합하여 설계되었습니다. 첫 단계에서, LMMs를 사용하여 비꼬기 탐지를 위한 소규모 언어 모델의 사전 학습을 위한 다양한 근거를 생성합니다. 그 후, 모델을 미세 조정 (Fine-tuning)하여 비꼬기 표적의 구체적인 식별을 수행합니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 모델은 기존의 최고수준 MSTI 방법들을 크게 앞서며, 비꼬기를 해독하는 데 있어 높은 설명 가능성 (Explainability)을 보여줌으로써 더욱 향상된 성능을 제공합니다.



### AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of  Low-Rank Adaptation Experts (https://arxiv.org/abs/2405.00361)
- **What's New**: AdaMoLE는 대규모 언어 모델을 보다 효율적으로 조정하기 위해 새롭게 소개된 방식입니다. 이는 Adaptive Mixture of Low-Rank Adaptation (LoRA) Experts라는 새로운 접근법을 사용하여, 다양한 작업의 복잡성에 적응적으로 반응합니다. 기존의 고정된 top-k 전략 대신, AdaMoLE은 동적으로 전문가(Experts) 활성화 임계값을 조정하는 전용 임계 네트워크를 활용합니다.

- **Technical Details**: AdaMoLE은 LoRA 기술을 활용하여 단일 LoRA 대신 여러 LoRA 전문가를 층(layer)에 통합하고, 입력 컨텍스트에 따라 가장 적절한 전문가를 선택하고 활성화하는 게이팅 기능과 임계값 메커니즘을 결합합니다. 이를 통해 모델은 입력에 따라 가장 유효한 전문가를 동적으로 선택할 수 있습니다.

- **Performance Highlights**: AdaMoLE은 상식적 추론(Commonsense Reasoning) 및 자연어 처리(Natural Language Processing) 작업들에 걸쳐 광범위한 평가를 거쳐, 기존 기준 모델들을 능가하는 성능을 보여줍니다. 이는 AdaMoLE이 전문가 수를 증가시키지 않고도 LoRA 전문가의 적응적 선택을 통해 모델 효율성을 증진시킬 수 있음을 강조합니다.



### A Careful Examination of Large Language Model Performance on Grade  School Arithmetic (https://arxiv.org/abs/2405.00332)
- **What's New**: 이 연구는 최신 대용량 언어 모델(LLM: Large Language Model)의 수학적 추론 능력을 평가하기 위해 새롭게 제작된 Grade School Math 1000 (GSM1k) 벤치마크를 사용하여 이전의 GSM8k 벤치마크와의 성능 차이를 조사합니다. 이는 모델이 실제 수학적 추론 능력을 보유하고 있는지, 아니면 단지 훈련 데이터에서 유사 데이터를 암기한 결과인지를 판별하고자 하는 목표를 가집니다.

- **Technical Details**: GSM1k는 총 1250개의 초등학교 수준의 수학 문제를 포함하며, 모든 문제는 기본 산술 연산(덧셈, 뺄셈, 곱셈, 나눗셈)으로 해결 가능합니다. GSM1k는 인간 주석자(annotators)에 의해 생성되었고, 언어 모델(language models)이나 기타 인공 생성 데이터 소스의 도움 없이 작성되었습니다. 이 연구에서는 공개된(open-source) 및 비공개(closed-source) LLM들에 대한 벤치마킹을 수행하였으며, 최고의 모델(GPT-4, Gemini, Claude 등)과 최악의 모델(Mistral, Phi 등)의 성능이 상당히 차별화되는 것을 관찰했습니다.

- **Performance Highlights**: GSM1k에서 평가된 몇몇 LLM은 GSM8k 대비 최대 13%의 정확도 하락을 보였으며, 특히 Mistral과 Phi 모델군은 모델 크기에 관계없이 일관되게 과적합(overfitting)의 증거를 보였습니다. 반면, Gemini, GPT, Claude와 같은 선두 모델들은 과적합의 징후를 거의 보이지 않았습니다. 추가 분석에 따르면, 모델이 GSM8k에서 예제를 생성할 확률과 GSM8k와 GSM1k 간의 성능 격차 사이에는 양의 상관관계(Spearman's r^2=0.32)가 있음을 확인했습니다.



### DFKI-NLP at SemEval-2024 Task 2: Towards Robust LLMs Using Data  Perturbations and MinMax Training (https://arxiv.org/abs/2405.00321)
- **What's New**: NLI4CT 작업에서는 대규모 언어 모델(LLMs: Large Language Models)을 사용하여 임상 시험 보고서(CTRs: Clinical Trial Reports)에서의 자연어 추론(NLI: Natural Language Inference)을 위한 견고한 모델 개발에 중점을 두고 있습니다. 이번 대회에서는 CTRs의 숫자, 어휘, 의미적 측면을 특별히 대상으로 하는 개입이 도입되었습니다.

- **Technical Details**: 우리가 제안하는 시스템은 최신 모델인 Mistral 레벨을 활용하며, 복잡한 입력 공간에 초점을 맞추기 위해 보조 모델을 보완합니다. 데이터에 숫자와 약어 기반의 교란(perturbations)을 적용함으로써, 의미 변화와 수치적 모순 개입을 처리할 수 있는 견고한 시스템을 훈련합니다.

- **Performance Highlights**: 분석을 통해 CTRs 내에서 추론에 대한 도전적인 섹션들을 조명했습니다. 이를 통해 우리의 시스템은 특정 의미적 변화와 수치적 모순을 감지하고 대응하는 능력을 입증하였습니다.



### Generating Feedback-Ladders for Logical Errors in Programming using  Large Language Models (https://arxiv.org/abs/2405.00302)
- **What's New**: 프로그래밍 과제에서 논리적 오류에 대한 피드백 생성에 대한 연구가 수행되었습니다. 이 연구에서는 LLM (Large Language Model)을 사용하여 한 문제에 대해 여러 단계의 피드백을 생성하는 '피드백-사다리(feedback-ladder)' 기법을 도입했습니다. 이는 학습자의 학습 맥락을 고려하고, 일반적인 단일 피드백 방법의 한계를 극복하고자 합니다.

- **Technical Details**: 본 연구에서는 문제 진술과 학생의 잘못된 제출물을 입력으로 받아 LLM을 활용해 다양한 레벨의 피드백을 생성합니다. 이러한 다층적 피드백 방법은 학생의 이전 제출물, 현재 지식 등 학생의 학습 맥락을 고려했을 때 실질적으로 유용할 수 있습니다. 또한, 교사는 학생의 개인적 학습 맥락에 따라 적절한 피드백 레벨을 선택하거나, 더 높은 레벨의 피드백이 효과적이지 않을 경우 더 상세한 피드백으로 진행할 수 있습니다.

- **Performance Highlights**: 사용자 연구를 통해 생성된 피드백-사다리의 품질을 평가했습니다. 연구 결과, 피드백의 레벨이 높아질수록 및 높은 점수를 받은 제출물에서 그 효과가 감소하는 경향을 관찰했습니다. 이는 더 높은 레벨의 피드백이 모든 학생들에게 동일하게 효과적이지 않을 수 있음을 시사합니다. 실제 교육 현장에서 이 방법은 교사가 학생 개별의 필요에 맞춰 피드백을 조정할 수 있게 해주어, 보다 맞춤화된 학습 경험을 제공할 수 있습니다.



### LITO: Learnable Intervention for Truthfulness Optimization (https://arxiv.org/abs/2405.00301)
Comments: 14 pages, 5 figures

- **What's New**: 새롭게 제안된 LITO(Learnable Intervention method for Truthfulness Optimization, 진실성 최적화를 위한 학습 가능한 개입 방법)는 대규모 언어 모델(LLM)이 각 질문 맥락에 맞추어 최적의 개입 강도를 자동으로 식별할 수 있도록 설계되었습니다. 이 방법을 통해 LLM이 생성하는 내용의 진실성을 향상시킬 수 있습니다.

- **Technical Details**: LITO는 모델의 생성물을 기반으로 점진적으로 개입 강도를 증가시키면서 가장 정확한 반응을 선택하거나 불확실할 때 답변을 거부하는 방식으로 작동합니다. 기존의 '진실된 방향(truthful directions)'을 적용하는 방식과 달리, LITO는 맥락에 따라 최적화된 개입 강도를 결정하는 학습 가능한 구조를 탑재하고 있습니다.

- **Performance Highlights**: 다양한 LLM과 질의응답(question-answering) 데이터셋에서 이루어진 실험을 통해 LITO가 진실성을 개선하는 동시에 작업의 정확성을 유지하였다는 결과가 나타났습니다. 이는 표준적인 일률적인 개입 방식의 문제점들을 해결하는 적응형 접근법이라 할 수 있습니다.



### How Can I Improve? Using GPT to Highlight the Desired and Undesired  Parts of Open-ended Responses (https://arxiv.org/abs/2405.00291)
Comments: 11 pages, full research paper, EDM 2024

- **What's New**: 이 연구는 자동 설명 피드백 시스템의 개발을 통해 튜터 훈련 과정의 피드백을 개선하려는 새로운 시도입니다. 특히, GPT (Generative Pre-Trained Transformers) 대형 언어 모델을 활용하여, 실시간으로 효과적인 설명 피드백을 제공하기 위한 시퀀스 라벨링 방법을 탐색합니다. 새롭게 도입된 Modified Intersection over Union (M-IoU) 점수는 시퀀스 품질 평가에서 인간의 판단과 유효하게 상관관계를 보여주며, 더 나아가 튜터의 찬사 구성 요소를 식별하는 데 GPT 모델의 유효성을 입증합니다.

- **Technical Details**: GPT 모델을 사용한 주요 기술적 접근 방식은 프롬프팅(prompting)과 파인 튜닝(fine-tuning)입니다. 프롬프팅은 모델이 이미 보유한 지식을 활용하여 원하는 출력을 생성하도록 입력 쿼리를 설계하는 방식이며, 파인 튜닝은 특정 작업이나 도메인에서 모델의 성능을 최적화하기 위해 모델의 파라미터를 조정합니다. 연구 결과, GPT-3.5 모델은 노력 기반 (effort-based) 찬사에서 M-IoU 점수 0.64, 결과 기반 (outcome-based) 찬사에서 0.84를 달성하여 인간 코더의 만족도와 일치하는 높은 성능을 보였습니다.

- **Performance Highlights**: 이 연구에서 GPT-3.5는 프롬프팅을 통해 노력 기반 찬사에서 M-IoU 0.46, 결과 기반 찬사에서 M-IoU 0.68의 성과를 보였습니다. 또한, 파인 튜닝을 통해 최적화된 GPT-3.5 모델은 노력 기반 찬사에서 M-IoU 0.64, 결과 기반 찬사에서 M-IoU 0.84까지 성능을 향상시켰습니다. 이는 인간 평가자의 판단과 일치하는 수준으로, 특히 튜터 훈련에서 매우 유용한 피드백을 제공할 수 있는 가능성을 시사합니다.



### Adversarial Attacks and Defense for Conversation Entailment Task (https://arxiv.org/abs/2405.00289)
- **What's New**: 이 연구에서는 다양한 NLP(Natural Language Processing, 자연어 처리) 작업에서 매우 강력함을 입증한 대규모 언어 모델Large language models (LLMs)의 새로운 도전에 대해 다룹니다. 특히, 모델을 방어하는 방법인 '적대적 공격(adversarial attack)'에 중점을 두고, 이를 모델의 새로운 도메인으로 간주하고 모델의 강건성을 향상시키는 방법을 모색합니다.

- **Technical Details**: 본 연구는 대화 추론(conversation entailment) 작업에 초점을 맞추어 여러 차례의 자연어 대화를 전제로 하고, 주어진 가설이 해당 대화에 대해 참인지 거짓인지를 예측하도록 Transformer 모델을 세밀하게 조정합니다. 적대적 공격자는 가설을 공격하여 모델이 잘못된 예측을 하도록 속입니다. 공격 방법으로는 동의어 교체(synonym-swapping)를 적용합니다. 모델의 강건성을 보여주기 위해 몇 가지 미세 조정 전략을 구현하고, 강건성을 높이기 위한 방법으로 embedding perturbation loss를 제안합니다.

- **Performance Highlights**: 이 연구는 NLP에서의 적대적 공격이 실제 세계에서 얼마나 중요한지를 논의함으로써 우리 작업의 중요성을 보여줍니다. 이는 모델을 보다 강건하게 만들어 적대적 환경에서도 효과적으로 작동할 수 있게 해줍니다.



### Social Life Simulation for Non-Cognitive Skills Learning (https://arxiv.org/abs/2405.00273)
- **What's New**: 새로운 대화 기반 플랫폼 'SimuLife++'가 소개되었습니다. 이 플랫폼은 대규모 언어 모델(LLM: Large Language Model)을 활용하여 사용자가 여러 AI 캐릭터와 함께 다양한 사회적 시나리오에서 이야기를 창조하고 주인공으로 활동할 수 있게 합니다. 특히, 인간-AI 상호작용에 ‘sage agent’라 불리는 새로운 캐릭터가 추가되어 사용자의 선택과 대화에 대해 더 심도 깊은 통찰을 제공함으로써, 인간-AI-AI 협력이 가능하게 되었습니다.

- **Technical Details**: SimuLife++는 사용자가 AI 기반 캐릭터와 상호 작용하면서 사회적 시나리오를 경험할 수 있도록 설계되었습니다. 이 플랫폼은 여러 AI 캐릭터와의 그룹 채팅을 통해 이야기를 진행할 수 있으며, sage agent가 관찰자로서 참여하여 사용자의 전략적 사고와 의사결정을 돕습니다. 이 플랫폼은 narrative transportation scale을 사용하여 narrative immersion을 측정하였고, 사용자의 비인지적(non-cognitive) 기술을 향상시키는데 중점을 두었습니다.

- **Performance Highlights**: SimuLife++에서 sage agent의 포함은 narrative immersion을 유의미하게 증가시켜, 특히 그룹 채팅에서의 메시지 수가 증가했습니다. 사용자들은 sage agent와의 상호작용을 통해 동기 부여, 자아 인식, 회복탄력성 및 대처 능력과 관련된 높은 점수를 보고했습니다. 이 결과들은 비인지능력의 반영에 긍정적인 영향을 미쳤다고 할 수 있습니다. 한편, 사용자 인터뷰 결과를 통해 sage agent가 의사 결정, 윤리적 딜레마 해결, 문제 해결을 돕는 데 유용하다는 평가를 받았으나, 사용자 제어 및 다중 캐릭터로부터의 균형 잡힌 반응에 대한 개선이 필요하다는 피드백도 있었습니다.



### Clover: Regressive Lightweight Speculative Decoding with Sequential  Knowledg (https://arxiv.org/abs/2405.00263)
- **What's New**: 이 논문에서는 큰 언어 모델(Large Language Models, LLMs)의 효율 문제를 해결하기 위해 Clover라는 새로운 추측적 디코딩 알고리즘을 제안합니다. Clover는 병렬 디코딩 과정에 연속적 지식을 통합하여 후보 토큰의 적중률을 높이고 전체 효율을 향상시킵니다.

- **Technical Details**: Clover는 Regressive Connection을 통해 사전에 추측된 토큰에서 연속적 지식을 전송하고, Attention Decoder를 사용하여 이러한 추측된 토큰을 통합합니다. 또한, Clover는 Augmenting Block을 포함하여 은닉 상태(hidden states)를 조정함으로써 다음 토큰 예측 대신 추측적 생성(speculative generation)의 목적에 더 잘 부합하도록 합니다.

- **Performance Highlights**: Clover는 Baichuan-Small에서 기준 모델을 최대 91%, Baichuan-Large에서는 최대 146%까지 능가했습니다. 또한, 이전에 가장 성능이 좋았던 Medusa 방법보다 Baichuan-Small에서 최대 37%, Baichuan-Large에서 최대 57%까지 성능이 우수함을 입증했습니다.



### CodeHalu: Code Hallucinations in LLMs Driven by Execution-based  Verification (https://arxiv.org/abs/2405.00253)
- **What's New**: 코드 생성 분야에서 대규모 언어 모델(Large Language Models, LLMs)이 크게 발전하고 있음에도 불구하고, 코드 생성 중 '환각 현상(code hallucinations)'이라는 새로운 문제점이 지적되었습니다. 이 연구는 코드 환각을 정의하고 분류하는 첫 번째 시도를 소개하며, LLMs가 생성하는 코드의 기능적 정확성과 안전성을 보장하기 위해 필요한 연구 방향을 제시합니다.

- **Technical Details**: 이 논문은 코드 환각을 실행 검증을 통해 정의하고, 네 가지 주요 유형(매핑(mapping), 명명(naming), 자원(resource), 논리(logic) 환각)으로 분류하여 각각의 고유한 도전을 이해하고 대응할 수 있는 방법을 제안합니다. 동적 감지 알고리즘(dynamic detection algorithm)을 제안하여 코드 생성 중에 발생하는 환각 현상을 체계적으로 평가하고, 8,883개의 샘플과 699개의 과제를 포함한 CodeHalu 벤치마크를 구축하여 LLMs의 환각 현상을 활발히 탐지합니다.

- **Performance Highlights**: 16개의 인기 있는 LLMs를 CodeHalu 벤치마크에서 테스트한 결과, 코드 생성 중 환각 발생 빈도와 성격에 있어서 현저한 차이가 나타났습니다. 이는 LLMs의 코딩 정확성과 신뢰성에 중요한 변화를 강조하며, 자동 생성된 코드의 기능적 정확성과 안전성을 확보하기 위해 모델과 훈련 방법을 개선할 필요성을 강조합니다. 이 연구는 코드 환각을 분류하고 정량화할 뿐만 아니라 LLM 기반 코드 생성 연구의 미래 개선을 위한 통찰력을 제공합니다.



### Graphical Reasoning: LLM-based Semi-Open Relation Extraction (https://arxiv.org/abs/2405.00216)
- **What's New**: 이 논문은 Chain of Thought (CoT, 사고 연쇄) 및 Graphical Reasoning (GRE, 그래픽 추론) 기법을 사용한 고급 언어 모델을 활용하여 관계 추출에 대한 종합적인 탐구를 제시합니다. 특히, in-context learning을 활용하여 GPT-3.5를 통한 추출 프로세스를 크게 향상시키는 방법을 보여줍니다. 또한, 관계 추출을 순차적인 하위 작업으로 분해하는 새로운 그래픽 추론 접근 방식을 도입하여 복잡한 관계 데이터 처리의 정밀도와 적응성을 향상시킵니다.

- **Technical Details**: 이 논문은 GPT-3.5의 in-context learning 기능을 활용하여 Chain of Thought 기법과 함께 사용합니다. 제시된 사례를 통해 텍스트에서 관계를 추출하는 단계별 사고 과정을 모델에 제시함으로써, 문제 해결 행동을 모방하고 모델의 예측을 더 해석 가능하고 신뢰할 수 있게 만듭니다. Graphical Reasoning (그래피컬 추론) 방법론은 관계 추출 작업을 엔티티 인식, 인식된 엔티티를 사용한 텍스트 재구성, 및 재구성된 텍스트를 바탕으로 한 관계 추출로 분해합니다. 이 모듈식 접근 방식은 각 하위 작업에서의 최적화 및 성능 향상을 가능하게 하며, 수학적 공식화를 통해 이론적 기초와 실제 구현을 상세히 설명합니다.

- **Performance Highlights**: 다양한 데이터셋을 사용한 실험을 통해, 우리의 방법론이 전통적인 접근 방식에 비해 눈에 띄는 성능 향상을 이루었다는 점이 확인되었습니다. ADE, CoNLL04, NYT 등의 잘 알려진 데이터셋과 함께, 원본 주석의 문제를 해결하고 더 신뢰할 수 있는 시험대를 제공하기 위해 수동으로 주석이 달린 CoNLL04 데이터셋 버전을 사용하였습니다. 구조화된 추론과 문제 분해의 통합이 NLP 작업의 성능과 신뢰성을 향상시킬 수 있는 가능성을 보여줍니다.



### A Primer on the Inner Workings of Transformer-based Language Models (https://arxiv.org/abs/2405.00208)
- **What's New**: 이 연구는 복잡한 언어 모델의 내부 작동 방식을 설명하기 위한 기술을 다루는 새로운 접근 방식을 제공합니다. 특히, Transformer 기반의 생성적 디코더 전용 아키텍처(generative decoder-only architecture)에 초점을 맞추고 있으며, 이 모델들의 내부 메커니즘에 대한 깊은 이해를 제공합니다.

- **Technical Details**: 이 논문은 Transformer 기반 언어 모델의 각종 구성 요소와 작동 원리를 자세히 설명합니다. 특히, Embedding layer, Attention mechanism(어텐션 메커니즘), Feed-forward network(피드-포워드 네트워크), Layer normalization(LayerNorm, 레이어 노멀라이제이션), 그리고 Softmax layer 등이 구체적으로 논의됩니다. 각 레이어에서의 데이터 처리 과정도 상세히 소개되어, 이를 통한 언어 모델의 이해도를 높일 수 있습니다.

- **Performance Highlights**: 이 연구는 복잡한 언어 모델의 예측에 책임 있는 입력이나 모델 컴포넌트를 식별하는 방법(i)과 학습된 표상(learned representations)에서 정보를 해석하여 네트워크 전체에서의 용도를 이해하는 방법(ii)으로 언어 모델의 해석 가능성을 두 가지 차원으로 분류합니다. 이로 인해, 언어 모델이 어떻게 작동하는지에 대한 통찰력을 제공하며, AI 안전성을 향상시키는 데 기여할 수 있습니다.



### General Purpose Verification for Chain of Thought Prompting (https://arxiv.org/abs/2405.00204)
Comments: 22 pages, preprint

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 추론 능력을 개선하는 새로운 방법을 제안합니다. 기존의 문제점을 해결하기 위해, 추론 과정에서 발생하는 오류를 초기에 포착하고 수정하는 problem-agnostic verification 기법을 도입하였습니다. 또한, 세 가지 기본 원칙을 사용하여 모델이 추론을 수행하도록 했습니다: 적절성(Relevance), 수학적 정확성(Mathematical Accuracy), 그리고 논리적 일관성(Logical Consistency).

- **Technical Details**: 저자들은 chain-of-thought (사고의 연쇄) 방식을 통해 생성된 단계별 추론 과정에 대해 모델 스스로 검증(verifiers)을 수행하게 하여 각 단계가 위 세 가지 원칙을 준수하는지 평가합니다. 추가적으로, 각 추론 단계의 perplexity를 이용하여 추론의 질을 높이기 위한 추가적인 검증자로 활용합니다. 이러한 검증 방법을 통해, 다양한 유형의 추론 작업에 걸쳐 다수의 데이터셋(총 9개)에서 모델의 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 이 방법은 기본적인 추론 생성 방법보다 항상 우수하였으며, 9개의 데이터셋 중 6개에서는 최고의 N 샘플링(best-of N sampling) 방법 보다도 더 나은 결과를 보였습니다. 이는 대형 언어 모델이 자체적으로 자신의 오류를 인지하고 수정할 수 있는 능력을 향상시킬 수 있다는 점을 시사합니다.



### SPAFIT: Stratified Progressive Adaptation Fine-tuning for Pre-trained  Large Language Models (https://arxiv.org/abs/2405.00201)
- **What's New**: 새로운 매개변수 효율적인 미세조정(PEFT, Parameter-Efficient Fine-Tuning) 방법으로 'Stratified Progressive Adaptation Fine-tuning (SPAFIT)'이 제안되었습니다. 이 방법은 Transformer 모델의 각 레이어에 특정한 언어 지식을 위치시키는 기반으로 개발되었습니다.

- **Technical Details**: SPAFIT는 다양한 언어지식을 각 레이어별로 배치하여 더 효율적인 미세조정을 가능하게 합니다. 이는 트랜스포머(Transformer) 아키텍처의 과다 매개 변수화와 재앙적 망각(catastrophic forgetting) 문제를 해결하고자 제안된 방법입니다. SPAFIT은 전통적인 방법인 LoRA와 BitFit처럼 모델의 모든 레이어에 적용되는 것이 아니라 특정 레이어에 집중적으로 적용됩니다.

- **Performance Highlights**: GLUE 벤치마크의 9가지 작업에서 수행된 실험 결과, SPAFIT 방법은 다른 PEFT 방법들에 비해 우수한 성능을 보였으며, 다른 방법들보다 훨씬 적은 매개 변수를 조정함에도 불구하고 더 나은 결과를 도출했습니다.



### In-Context Learning with Long-Context Models: An In-Depth Exploration (https://arxiv.org/abs/2405.00200)
Comments: 27 pages; preprint

- **What's New**: 이 논문은 맥락 길이(context length)가 증가함에 따라 맥락 내에서 제공할 수 있는 데모수가 전체 훈련 데이터셋의 크기에 근접하게 됨을 조사합니다. 특히 대규모 레이블 공간을 가진 데이터셋에서 수백 또는 수천 개의 데몬스트레이션을 사용할 때 성능이 계속 향상됨을 보여줍니다. 이는 맥락 내 학습(In-context learning, ICL)과 장맥락 모델(long-context models)의 여러 속성을 조사하는 데 사용됩니다.

- **Technical Details**: 연구팀은 Llama2-7b 및 그 변형 모델들과 같은 장맥락 모델을 사용하여 인-컨텍스트 학습의 성능을 평가했습니다. 이러한 모델들은 극단적인 맥락 길이를 다룰 수 있도록 조정되었으며, 이를 통해 데몬스트레이션의 숫자를 2000개 이상까지 확장할 수 있습니다. 연구팀은 샘플링, 관련 데이터 검색(retrieving relevant data), 풀 데이터셋에 대한 파인튜닝(finetuning) 등 다양한 방법으로 대규모 데이터셋을 사용하여 실험을 수행했습니다.

- **Performance Highlights**: 장맥락 인-컨텍스트 학습(long-context ICL)은 인퍼런스 시 데몬스트레이션을 그룹화하는 것이 성능에 부정적인 영향을 미칠 수 있다는 점을 밝혀냈습니다. 그러나 적절한 예시의 검색을 통해서는 훨씬 관련성 높은 예시들을 검색해내어 인퍼런스 성능을 크게 향상시킬 수 있음을 확인했습니다. 실험 결과 전체 5개의 데이터셋에 대해 평균 36.8포인트의 정확도 향상(10에서 1000 데몬스트레이션까지)을 보였고, 이는 관련 예시 검색이 무작위 선택된 예시 집합을 사용하는 것보다 단기 맥락에서 훨씬 우수한 성능을 나타냄을 의미합니다.



### Towards a Search Engine for Machines: Unified Ranking for Multiple  Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2405.00175)
- **What's New**: 이 논문은 다양한 하위 검색 강화 생성(Retrieval-Augmented Generation, RAG) 시스템을 지원하는 통합 검색 엔진을 제공하는 uRAG 프레임워크를 소개합니다. 각 RAG 시스템은 검색 결과를 독특한 목적으로 사용하며, 이는 오픈 도메인 질문 응답, 사실 검증, 엔티티 연결, 관계 추출 등을 포함합니다. uRAG는 검색 엔진과 하위 RAG 시스템 간의 통신을 표준화하는 일반적인 훈련 지침을 도입하여 검색 모델을 최적화하는 데 기여합니다.

- **Technical Details**: uRAG는 18개의 RAG 시스템과 함께 대규모 실험 생태계를 구현하여 중요한 연구 질문에 답할 수 있습니다. 이들 시스템은 다양한 대규모 언어 모델(Large Language Model, LLM) 아키텍처를 사용하며, 검색된 문서 수, 훈련 데이터 세트가 다릅니다. 이 실험을 통해 검색 엔진 개발의 약속과 도전을 더 잘 이해할 수 있습니다.

- **Performance Highlights**: 통합된 재순위화(Unified Reranking)는 각각의 하위 RAG 모델을 위해 개별적으로 훈련된 재순위화 모델과 비교하여 동등하거나 훨씬 우수한 성능을 보여, 훈련 프로세스에 참여하는 18개의 RAG 모델 중 61%에서 통계적으로 유의미한 성능 향상을 보였습니다. 신규(미지의) RAG 시스템에 대한 효율성과 일반화 가능성에 대한 연구 질문도 다루었으며, 이는 통합된 재순위화 모델이 BM25를 크게 능가하고 동일한 데이터 세트를 사용하는 다른 RAG 시스템에서 훈련된 재순위화와 비교할 수 있음을 보여줍니다.



### HistNERo: Historical Named Entity Recognition for the Romanian Languag (https://arxiv.org/abs/2405.00155)
Comments: Accepted at the International Conference on Document Analysis and Recognition (ICDAR 2024)

- **What's New**: 이 연구는 역사적 신문에 대한 이름을 지정된 엔티티 인식(Named Entity Recognition, NER)을 위한 첫 번째 루마니아어 코퍼스인 HistNERo를 소개합니다. 전반적으로 1817년부터 1990년까지의 기간을 포괄하며, 베사라비아(Bessarabia), 몰다비아(Moldavia), 트란실바니아(Transylvania), 왈라키아(Wallachia) 등 루마니아의 네 역사적 지역을 대표하는 문서를 수집하였습니다.

- **Technical Details**: HistNERo 데이터셋은 10,026개 문장과 323,865개 토큰으로 구성되어 있으며, PERSON, ORGANIZATION, LOCATION, PRODUCT, DATE 등 다섯 가지 이름 지정된 엔티티로 주석이 달렸습니다. 데이터셋의 주석 작업은 RELATE 플랫폼에서 이루어졌으며, 총 9,601개의 엔티티가 주석처리되었습니다. 또한, 이 연구는 여러 루마니아어 사전 훈련된 언어 모델(pre-trained language models)의 성능을 평가하여, RoBERT-base 모델이 55.69%의 엄격한 F1-score(strict F1-score)를 달성한 것을 발견하였습니다.

- **Performance Highlights**: 역사적 문서의 언어적 특성 때문에, 도메인 적응 기술(domain adaptation technique)인 'loss reversal'을 적용하여 모델의 성능을 향상시켰습니다. 이 기술은 RoBERT-base 모델의 F1-score를 63.85%로 향상시켰고, RoBERT-large 모델은 66.80%의 F1-score를 달성하여 더 높은 성능을 보였습니다. 이 기술은 도메인 차별자의 손실 신호를 피처 추출기에 역전시켜서 (the sign of the domain discriminator loss is reversed before entering the feature extractor) 지역 간의 편차를 줄이는 데 중점을 두고 있습니다.



### Transforming Dutch: Debiasing Dutch Coreference Resolution Systems for  Non-binary Pronouns (https://arxiv.org/abs/2405.00134)
Comments: 22 pages, 2 figures. Accepted at the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)

- **What's New**: 이 연구는 네덜란드어의 성 중립 대명사 'hen'과 'die'의 대용법 시스템(Corefence Resolution) 성능을 평가합니다. 이는 성 중립 대명사가 2016년에 도입된 이후 처음으로 시행되는 평가이며, 성별 이분법(gender binary)에 국한되지 않은 새로운 평가 척도인 프로논 스코어(Pronoun Score)를 도입하여, 이러한 대명사의 처리 정확도를 직접적으로 반영합니다.

- **Technical Details**: 본 논문은 대용법 시스템에서 성 중립 대명사 처리의 편견을 감소시키기 위해 두 가지 탈편견 기법(Debiasing Techniques), 카운터팩추얼 데이터 증강(Counterfactual Data Augmentation, CDA)과 단어 제거(Delexicalisation)를 비교 평가합니다. CDA는 성별 특정 대명사와 성 중립 대명사 사이의 성능 격차를 줄이는 데 효과적인 것으로 나타났지만, 단어 제거 방법은 개선에 실패하였습니다. 추가로, CDA는 리소스가 제한된(low-resource) 환경 및 이전에 보지 못한 새로운 밈 명사(neopronouns)에도 효과적인 것으로 관찰되었습니다.

- **Performance Highlights**: 성 중립 대명사 'hen'과 'die'를 포함한 네덜란드어 대용법 시스템은 성별이 구체화된 대명사에 비해 낮은 성능을 보였습니다. 그러나 CDA를 적용한 결과, 성능 격차가 상당히 줄어들었으며, 특히 자원이 부족한 환경에서도 그 효과가 지속됨을 확인할 수 있었습니다. 이는 NLP 시스템에서 성 중립성 확보를 위한 효과적인 접근법을 제시하며, 향후 성 중립 대명사의 사용 빈도가 증가할 것을 고려할 때 중요한 발견입니다.



### Self-Play Preference Optimization for Language Model Alignmen (https://arxiv.org/abs/2405.00675)
Comments: 25 pages, 4 figures, 5 tables

- **What's New**: 이 논문에서는 인간의 선호에 기반한 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 방식을 대체하는 새로운 자기 대결(Self-Play) 기반 최적화 방법, '자기 대결 선호 최적화(Self-Play Preference Optimization, SPPO)'를 제안합니다. 이 방법은 큰 언어 모델(Large Language Models, LLMs)을 보다 인간의 선호와 일치하도록 미세 조정하는 데 사용됩니다. SPPO는 전통적인 선호 기반 강화 학습 방법과 다르게, 내쉬 균형(Nash Equilibrium)을 찾아내는 데 초점을 맞춥니다.

- **Technical Details**: SPPO는 이차원 상수합 게임(two-player constant-sum game)으로 언어 모델 정렬 문제를 접근하고, 내쉬 균형에 근사하는 반복 정책 업데이트를 통해 이를 해결합니다. 이러한 접근은 이론적 수렴 보장을 제공하며, 대응 반응의 로그 가능도를 증가시키고 거부된 반응의 로그 가능도를 감소시키는 효과를 보입니다. SPPO는 직접 선호 최적화(Direct Preference Optimization, DPO) 및 정체성 선호 최적화(Identity Preference Optimization, IPO)와 같은 대칭 쌍대 손실 방식과는 차별화됩니다.

- **Performance Highlights**: SPPO는 UltraFeedback 데이터셋에서 60k 프롬프트만을 사용함에도 불구하고, Mistral-7B-Instruct-v0.2 모델을 미세 조정하여 AlpacaEval 2.0에서 GPT-4-Turbo에 대한 상태에서 최첨단 길이-제어 승률(state-of-the-art length-controlled win-rate) 28.53%를 달성했습니다. 또한, MT-Bench 및 Open LLM Leaderboard에서 (반복적인) DPO 및 IPO를 능가하는 성능을 보여주었습니다.



### NumLLM: Numeric-Sensitive Large Language Model for Chinese Financ (https://arxiv.org/abs/2405.00566)
- **What's New**: 이 논문에서는 중국 금융 분야에 특화된 새로운 대형 언어 모델(Numeric-sensitive Large Language Model, NumLLM)을 제안한다. 이 모델은 금융 텍스트에서 숫자 변수를 포함하는 질문을 이해하는 능력을 향상시키기 위해 개발되었다.

- **Technical Details**: NumLLM은 금융 교과서에서 수집한 금융 코퍼스(Fin-Textbooks)를 기반으로 두 개의 Low-Rank Adaptation (LoRA) 모듈을 훈련하여 개발되었다. 하나의 모듈은 일반적인 LLM을 금융 도메인에 적합하게 조정하고, 다른 하나는 숫자 변수를 포함한 금융 텍스트 이해능력을 강화하는 데 사용된다. 이 두 모듈은 기초 모델에 통합되어 NumLLM을 구성하며, 이를 통해 추론이 수행된다.

- **Performance Highlights**: 금융 질의응답 벤치마크에서 NumLLM은 기초 모델의 성능을 크게 향상시키고, 숫자 및 비숫자 질문 모두에서 모든 기준 모델보다 가장 우수한 전반적인 성능을 달성했다. 이는 NumLLM이 금융 분야에서 다른 모델들과 비교할 때 탁월한 수치 이해능력을 제공함을 시사한다.



### CookingSense: A Culinary Knowledgebase with Multidisciplinary Assertions (https://arxiv.org/abs/2405.00523)
Comments: LREC-COLING 2024 Accepted

- **What's New**: 이 논문에서는 요리 분야의 다양한 지식 연구를 위해 구축된 새로운 지식 베이스인 CookingSense와 새로운 벤치마크인 FoodBench를 소개합니다. CookingSense는 웹 데이터, 과학 논문, 레시피에서 추출한 지식을 바탕으로 건설되었으며, FoodBench는 요리 결정 지원 시스템의 성능을 평가하기 위해 개발되었습니다.

- **Technical Details**: CookingSense는 사전 기반 필터링(Dictionary-based filtering)과 언어 모델 기반 의미적 필터링(Language model-based semantic filtering) 기술을 사용하여 구축되었습니다. 이는 다양한 분야의 음식 관련 주장을 포함하는 풍부한 지식 베이스를 형성합니다. FoodBench는 맛 예측, 재료 분류, 요리 질문 응답 등 다양한 평가 작업을 포함합니다.

- **Performance Highlights**: CookingSense는 검색 강화 언어 모델(Retrieval augmented language models)의 성능을 향상시키는 것으로 나타났습니다. 또한, FoodBench를 사용한 평가에서 CookingSense가 일반적인 조리 지식 베이스와 비교하여 더 넓은 요리 개념 이해를 제공하는 풍부한 설명과 다양한 의미를 제공한다는 것을 증명했습니다.



### Navigating WebAI: Training Agents to Complete Web Tasks with Large  Language Models and Reinforcement Learning (https://arxiv.org/abs/2405.00516)
Comments: ACM 2024, Avila Spain. 9 pages

- **What's New**: 이 논문에서는 감독 학습(Supervised Learning, SL)과 강화 학습(Reinforcement Learning, RL) 기법을 결합하여 MiniWoB 벤치마크에서 두 방법의 강점을 활용하는 새로운 접근 방식을 제안합니다. 이전 모델들이 HTML 내용의 이해에 있어 한계를 드러냄으로써, 이를 극복하고 진정한 이해를 향상시키기 위한 방법을 제시합니다.

- **Technical Details**: 우리는 T5 기반 모델을 분석하고, 계층적 계획 기술(hierarchical planning techniques)을 사용하여 한계를 극복합니다. 이후, 최적화된 모델을 다중 모드 신경망(multimodal neural network)과 통합하여 SL과 RL을 모두 사용하여 성능과 적응성을 향상시킵니다. 또한, Miniwob++ 벤치마크와 인간 제작 시연을 기반으로 하는 새로운 데이터셋을 통해 모델을 훈련시킵니다.

- **Performance Highlights**: 제안된 접근 방식은 SL에서 평균 43.58%의 정확도를 달성하고, SL과 RL을 결합한 멀티모드 접근 방식에서는 36.69%의 정확도를 보여 주어, 기존 SL 방법들을 능가하고 RL 모델과의 성능 격차를 줄였습니다. 이러한 결과는 향후 웹 탐색과 관련된 언어 모델링의 한계와 잠재력에 대한 통찰력을 제공합니다.



### GOLD: Geometry Problem Solver with Natural Language Description (https://arxiv.org/abs/2405.00494)
Comments: Accepted in NAACL 2024 Findings

- **What's New**: 인공지능(AI)에서 자동 기하학 문제 해결에 대한 새로운 접근법으로, 'Geometry problem sOlver with natural Language Description' (GOLD) 모델을 소개합니다. GOLD 모델은 기하학 다이어그램 내의 기호와 기하학적 기본 요소를 별도로 처리함으로써 기하학적 관계 추출을 향상시킵니다. 이를 통해 추출된 관계를 자연어 설명으로 변환하고, 큰 언어 모델을 효율적으로 활용하여 기하학 문제를 해결합니다.

- **Technical Details**: GOLD 모델은 기하학 다이어그램에서 기호(symbol)와 기하학적 기본 요소(geometric primitives)를 분리하여 처리하는 방식으로 기하학적 관계를 파악하고 이를 자연어 설명(natural language description)로 변환합니다. 이렇게 변환된 자연어를 활용하여 대규모 언어 모델(large language models)로 문제를 해결합니다.

- **Performance Highlights**: GOLD 모델은 이전 최고 방법이었던 Geoformer 모델을 UniGeo 데이터셋에서 계산(calculation) 부분에서 12.7% 향상, 증명(proving) 부분에서는 42.1% 향상의 정확도를 달성하여 능가했습니다. 또한, PGPS9K 및 Geometry3K 데이터셋에서 이전 최고 모델인 PGPSNet보다 각각 1.8% 및 3.2%의 정확도 향상을 이뤄냈습니다.



### Explainable Automatic Grading with Neural Additive Models (https://arxiv.org/abs/2405.00489)
- **What's New**: 이 연구에서는 자동 단답형 평가 (ASAG) 모델의 새로운 접근 방식인 신경 첨가 모델 (Neural Additive Model, NAM)을 사용하여 평가의 투명성과 예측 성능을 향상시키려고 시도했습니다. NAM은 신경망 (Neural Networks, NN)의 고성능을 유지하면서 학부형 모델의 설명 가능성을 제공합니다. 학습 과학에서의 지식 통합 (Knowledge Integration, KI) 프레임워크를 활용하여 학생들이 응답에 특정 아이디어를 포함했는지를 반영하는 입력 항목을 생성하여 NAM의 예측력과 해석 가능성을 높였습니다.

- **Technical Details**: NAM은 각 특성이 최종 예측 점수에 미치는 기여도를 시각적으로 검토할 수 있게 해주며, 입력으로 쓰인 특성을 엔지니어링해야 합니다. 이 연구에서는 KI 프레임워크에 기반한 하나의 문항으로 NAM의 성능을 시험하였고, 동일한 특성을 사용하는 설명 가능한 모델인 로지스틱 회귀 (Logistic Regression, LR) 및 설명이 불가능한 모델인 DeBERTa와 비교 분석하였습니다. DeBERTa는 특성 공학 없이 텍스트에서 스스로 특성을 생성합니다.

- **Performance Highlights**: 예측 성능 측면에서 NAM은 LR보다 우수하며, 대형 언어 모델 (Large Language Model, LLM) 분류자와 비슷한 수준을 보였습니다. 또한, NAM을 사용하면 이해관계자들이 어떤 특성이 응답의 예측에 중요한지 이해할 수 있도록 도와줍니다. 이는 특히 교육 분야에서 ASAG 시스템의 실용적 활용을 위해 더 많은 설명 가능한 모델의 필요성을 강조합니다.



### Enhancing Surgical Robots with Embodied Intelligence for Autonomous  Ultrasound Scanning (https://arxiv.org/abs/2405.00461)
Comments: ICRA 2024 Full-day Workshop: C4SR+: Continuum, Compliant, Cooperative, Cognitive

- **What's New**: 이 연구에서 새롭게 제안된 Ultrasound Embodied Intelligence 시스템은 초음파 로봇에 대한 인간의 의도와 지시를 이해할 수 있는 능력을 부여하여 자율적인 초음파 스캐닝을 가능하게 합니다. 이 시스템은 큰 언어 모델(Large Language Model, LLM)과 도메인 지식을 결합하여 초음파 로봇의 효율성을 개선합니다.

- **Technical Details**: 본 시스템은 초음파 운영 지식 데이터베이스(Ultrasound Operation Knowledge Database)를 설계하여 LLM에 초음파 스캐닝 전문 지식을 추가하고, 정밀한 모션 플래닝(Precise Motion Planning)을 수행할 수 있도록 합니다. 또한, 'think-observe-execute' 프롬프트 엔지니어링을 기반으로 동적 초음파 스캐닝 전략(Dynamic Ultrasound Scanning Strategy)을 개발하여 스캐닝 절차 중 모션 플래닝 전략을 동적으로 조정할 수 있습니다.

- **Performance Highlights**: 실시된 실험들은 본 시스템이 구어 명령에서 초음파 스캔 효율성과 품질을 크게 향상시킴을 보여줍니다. 이는 비침습적 진단(Non-invasive Diagnostics) 및 의료 워크플로우의 효율성을 높이는 데 기여합니다.



### RAG-based Explainable Prediction of Road Users Behaviors for Automated  Driving using Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2405.00449)
- **What's New**: 이 논문에서는 자율 주행 차량의 환경에서 도로 사용자의 행동을 예측하기 위해 지식 그래프(Knowledge Graphs, KG)와 대규모 언어 모델(Large Language Models, LLM)의 결합을 통한 새로운 접근 방식을 제안합니다. 이 시스템은 Retrieval Augmented Generation (RAG) 기술을 활용하여 설명 가능한 예측을 제공하며, 실시간으로 수집된 증거와 지식 그래프에 포함된 레거시 정보를 기반으로 예측을 발행합니다.

- **Technical Details**: 이 시스템은 지식 그래프 임베딩(Knowledge Graph Embeddings, KGE)과 베이지안 추론(Bayesian inference)을 결합하여 완전히 유도적인 추론 시스템을 사용합니다. 두 가지 사용 사례가 구현되었습니다: 1) 보행자의 도로 횡단 행동 예측, 2) 차선 변경 기동 예측. 이 시스템은 도로 사용자의 키네마틱 정보뿐만 아니라 사회적 상호작용과 환경적 요소를 포함한 맥락적 정보를 고려합니다.

- **Performance Highlights**: 제안된 시스템은 현재까지의 연구보다 더 높은 선행성(anticipation)과 F1 점수를 달성하여 상태 보존 기술(state of the art)을 능가합니다. 특히, 도로 사용자의 행동을 예측하는 분야에서 매우 유망한 연구 방향을 보여 주며, 자율 주행 차량의 안전성과 효율성을 높이는 데 기여할 수 있을 것으로 기대됩니다.



### MetaRM: Shifted Distributions Alignment via Meta-Learning (https://arxiv.org/abs/2405.00438)
Comments: 11 pages, 6 figures. arXiv admin note: text overlap with arXiv:2401.06080

- **What's New**: 새롭게 소개되는 MetaRM은 Meta-Learning(메타러닝)을 활용해 강화학습에서의 보상 모델(Reward Model, RM)의 한계를 극복하고자 하는 방법론입니다. 기존의 RM은 훈련 데이터 분포에 익숙해져 외부 분포에 대한 일반화가 어려웠으나, MetaRM은 이러한 분포의 변화에도 불구하고 정확한 판별 능력을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: MetaRM은 강화 학습(RL) 과정에서 환경의 분포가 변화할 때 RM의 효과성이 저하되는 문제를 해결하기 위해 메타러닝을 적용합니다. 이 방법은 RM이 변화된 데이터 분포를 감지하고 이에 대응할 수 있도록 데이터 손실을 최소화하는 트레이닝을 포함합니다. 결과적으로, RM은 게재된 환경 내에서 더 정밀하게 반응을 구분할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 연구진이 실시한 광범위한 실험 결과에 따르면, MetaRM은 반복적인 강화학습(RLHF) 최적화 과정에서 RM의 구분 능력을 눈에 띄게 향상시켰습니다. 또한, MetaRM은 분포 밖(out-of-distribution) 샘플에서 미묘한 차이를 식별하는 능력을 제공함으로써 RM의 적용 범위와 효율성을 크게 개선하였습니다.



### Graph Neural Network Approach to Semantic Type Detection in Tables (https://arxiv.org/abs/2405.00123)
- **What's New**: 이 연구는 관계형 테이블에서의 의미 있는 컬럼 유형 감지라는 과제를 해결합니다. 신규 접근 방식으로 Graph Neural Networks (GNN, 그래프 신경망)을 사용하여 테이블 내 컬럼의 상호 의존성을 모델링하면서 언어 모델은 테이블 간 정보에 집중할 수 있게 하였습니다. 이러한 방식은 기존의 최첨단 알고리즘을 능가할 뿐만 아니라 다양한 GNN 유형의 유용성과 기능에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: GAIT라 불리는 새로운 프레임워크는 단일 컬럼 예측 프레임워크에서 그래프 신경망 (GNN) 모듈을 사용하여 테이블 간(inter-table) 및 테이블 내(intra-table) 정보를 동시에 통합합니다. 각 테이블은 컬럼을 노드로 하고 컬럼 간의 의존성을 에지로 하는 그래프로 모델링되며, GNN의 메시지 패싱(Message Passing)을 사용하여 이 의존성을 처리합니다. 이 접근방식은 특히 넓은 테이블을 효과적으로 처리할 수 있으며, 다양한 시나리오에서 경쟁력 있는 모델로 자리 잡을 수 있습니다.

- **Performance Highlights**: GAIT 모델은 RECA와 같은 최신의 언어 모델 기반 접근 방식을 기반으로 하면서, Doduo의 단점을 극복하고, 컬럼간 의존성을 포괄적으로 고려하여 성능을 유지합니다. 이 복합 데이터 접근 방식을 통해 GAIT는 단일 테이블 정보만을 고려하는 기존 모델들과 비교하여 뛰어난 성능을 보입니다.



### Creative Beam Search (https://arxiv.org/abs/2405.00099)
- **What's New**: 이 논문은 창의적 빔 검색(Creative Beam Search, CBS)이라는 새로운 샘플링 스키마를 제안합니다. 이 방법은 다양한 빔 검색(Diverse Beam Search, DBS)과 LLM-as-a-Judge를 이용하여 창의적 과정을 모방하고, 응답 생성 및 검증 단계를 통해 더 나은 결과를 도출할 수 있도록 합니다.

- **Technical Details**: CBS는 두 가지 주요 단계로 구성됩니다: 응답 생성과 응답 검증. 첫 번째 단계에서는 DBS를 사용하여 다양한 해결책을 생성하며, 두 번째 단계에서는 생성된 응답을 LLM-as-a-Judge 기법을 통해 검증합니다. 이 모델은 각 후보의 위치 편향을 조정하기 위해 균형 위치 보정 체계를 사용하며, 평가 과정에서 최종 출력을 결정합니다.

- **Performance Highlights**: CBS는 기존의 샘플링 기법보다 우수한 결과를 제공하며, 특히 창의성 평가에서 높은 선호도를 얻었습니다. 이는 CBS가 LLMs의 창의력을 향상시키는 데 효과적임을 시사합니다.



### SIMPLOT: Enhancing Chart Question Answering by Distilling Essentials (https://arxiv.org/abs/2405.00021)
- **What's New**: 최근에 SIMPLOT은 복잡한 차트(chart)를 해석하는 데 사용되는 새로운 방법을 제안했습니다. 기존의 Deplot 방법과 달리, SIMPLOT은 차트로부터 필수 정보만을 추출하여 이를 데이터 테이블로 변환하는 방법을 사용합니다. 이러한 접근 방식은 표 추출(chart-to-table extraction)에서 중요하지 않은 정보를 배제함으로써 보다 정확한 차트 추론(chart reasoning)을 가능하게 합니다. 또한, SIMPLOT은 시각적 속성(visual attributes)을 고려하는 새로운 프롬프트(novel prompt)를 제안하여 기존 모델들이 간과한 색상(color) 등의 속성을 처리할 수 있습니다.

- **Technical Details**: SIMPLOT의 기술적인 디테일을 살펴보면, 두 단계의 접근 방식을 포함합니다. 첫 번째는 복잡한 차트에서 필수 정보만을 포함하는 간단한 플롯(simple plot)을 모방(mimic)하여 훈련하는 것이고, 두 번째는 이를 바탕으로 데이터 테이블을 생성하고, 그 테이블을 사용하여 추론(reasoning)을 수행하는 것입니다. 이 방법은 추가적인 주석(annotations)이나 데이터셋(datasets)이 필요 없이 정확한 차트 추론을 가능하게 합니다.

- **Performance Highlights**: SIMPLOT의 효과는 다양한 실험을 통해 입증되었습니다. 특히, 필수 정보의 정확한 추출을 통해 차트 추론의 정확도가 향상되었으며, 색상과 같은 시각적 속성을 고려한 프롬프트 사용으로 모델의 적용 범위가 확대되었습니다. 또한, 기존의 대규모 언어 모델(Large Language Models, LLMs)을 통한 추론과 비교하여 상당한 성능 향상이 보고되었습니다.




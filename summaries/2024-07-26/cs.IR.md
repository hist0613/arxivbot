New uploads on arXiv(cs.CL)

### Self-Training with Direct Preference Optimization Improves Chain-of-Thought Reasoning (https://arxiv.org/abs/2407.18248)
Comments:
          ACL 2024. Code and data are available at this https URL

- **What's New**: 연구자들은 소규모 언어 모델(Language Models, LMs)을 자가 학습(self-training)을 통해 수학적 추론 능력을 향상시키는 방법을 제안했습니다. 기존 자가 학습 방법에 Direct Preference Optimization (DPO) 알고리즘을 통합하여 더 정확하고 다양한 chain-of-thought 추론을 유도합니다. 이 방법은 높은 비용과 예측 불가능한 행동을 보이는 대형 LMs에 의존하는 대신, 소규모 LMs의 성능을 더 경제적이고 안정적으로 향상시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 자가 학습 프레임워크를 확장하여 Direct Preference Optimization (DPO)을 통합합니다. DPO는 선호 데이터(preference data)를 활용하여 LMs가 더 정확하고 다양한 추론 경로를 따르도록 합니다. 특히 수학적 추론에서 명확한 풀이가 가능한 점을 활용하여 모델 출력을 검증하고 선호 데이터셋을 생성합니다. 이 접근 방식은 기존 자가 학습 방법의 한계를 보완하면서도 높은 성능을 유지합니다.

- **Performance Highlights**: 다양한 수학적 추론 작업에 대해 여러 기본 모델을 사용하여 방법을 평가한 결과, 제안된 접근 방식은 LMs의 추론 성능을 크게 향상시켰습니다. 또한 대형 LMs에 의존하는 방법과 비교해 비용 효율적이며 확장성이 뛰어납니다. 실험 결과는 GSM8K 데이터셋에서 성능과 계산 비용 간의 관계를 시각적으로 보여주며 경쟁력 있는 성능을 달성하면서도 연산 요구량을 상당히 줄였음을 확인했습니다.



### The FIGNEWS Shared Task on News Media Narratives (https://arxiv.org/abs/2407.18147)
Comments:
          18 pages, 10 tables, 1 figure, accepted to ArabicNLP 2024 co-located with ACL 2024

- **What's New**: 최근에 발표된 FIGNEWS 공유 과제는 ArabicNLP 2024 컨퍼런스의 일환으로 ACL 2024와 공동 개최되었습니다. 이 과제는 다국어 뉴스 포스트에서의 편향(bias) 및 선전(propaganda) 주석 작업을 다루며, 특히 이스라엘-가자 전쟁 초기 기간을 사례로 분석합니다. 이 공유 과제는 다섯 개 언어 (영어, 프랑스어, 아랍어, 히브리어, 힌디어)를 포함한 다국어 관점에서 문제를 접근하여 다양한 서사를 분석하는 프레임워크를 만듭니다.

- **Technical Details**: 총 17개 팀이 두 가지 주석 하위 과제에 참여했습니다: 편향 (16개 팀)과 선전 (6개 팀). 평가 트랙은 가이드라인 개발, 주석 품질, 주석 양, 그리고 일관성 등 네 가지로 구성되었습니다. FIGNEWS는 다양한 언어와 서사를 동시에 살펴보며 뉴스 기사에서 잠재적인 편향과 선전을 밝히는 작업을 진행하고, 복잡하고 주관적인 작업에 대한 주석 가이드를 개발하는 것에 중점을 둡니다.

- **Performance Highlights**: 참여한 팀들은 총 129,800개의 데이터 포인트를 생성했습니다. 이 공유 과제는 편향 및 선전 탐지에 대한 강력한 방법론과 지표를 개발하고, 미래의 NLP와 관련 분야에서 사용될 소중한 자원을 제공하게 됩니다. 이를 통해 미디어 리터러시 향상과 더욱 정보에 근거한 비판적 대중의 형성을 목표로 합니다.



### Dallah: A Dialect-Aware Multimodal Large Language Model for Arabic (https://arxiv.org/abs/2407.18129)
- **What's New**: 이 논문은 주로 영어에 한정된 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 한계를 넘기 위해, 고급 언어 모델 LLaMA-2를 기반으로 한 새로운 아랍어 멀티모달 어시스턴트인 Dallah를 소개합니다. Dallah는 아랍어의 다양한 사투리를 포함한 복잡한 언어 및 시각적 요소를 효과적으로 처리할 수 있습니다.

- **Technical Details**: Dallah는 LLaVA라는 고급 멀티모달 언어 모델 프레임워크를 사용하여 구축되었으며, AraLLaMA의 언어 능력을 통합했습니다. Dallah는 현대 표준 아랍어(Modern Standard Arabic, MSA) 및 여러 아랍어 방언에 대해 미세 조정되었습니다. 또한 새로운 데이터 필터링 방법을 도입하여 고품질, 관련성 높은 멀티모달 데이터셋을 선택하고 사용하여 모델을 최적화했습니다.

- **Performance Highlights**: Dallah는 아랍어 멀티모달 벤치마크 테스트에서 최첨단 성능을 보여주며, 특히 MSA 평가와 방언 응답 평가에서 뛰어난 실력을 발휘합니다. 이 모델은 MSA 뿐만 아니라, 여섯 가지 주요 아랍어 사투리로 성공적으로 미세 조정되었습니다. 또한, Dallah-Bench를 도입하여 실생활 애플리케이션에서 멀티모달 언어 모델의 효능을 평가할 수 있게 합니다.



### Tracking linguistic information in transformer-based sentence embeddings through targeted sparsification (https://arxiv.org/abs/2407.18119)
Comments:
          12 pages, 9 figures, 1 table, published in RepL4NLP 2024

- **What's New**: 이번 연구는 transformer 기반 모델이 문장 표현(embeddings)에서 다양한 형태의 언어 정보를 어떻게 인코딩하는지 조사합니다. 특히 명사구, 동사구, 전치사구와 같은 구(chunk) 정보가 문장 표현 내 특정 영역에 국한되어 인코딩되는지를 분석합니다.

- **Technical Details**: 데이터셋을 사용하여 알려진 구조를 가진 문장에서 구 정보 및 의미 역할이 문장 표현에서 어느 정도 지역화될 수 있는지를 테스트합니다. 이 연구는 변종 인코더-디코더 시스템을 사용하여 문장 표현의 각 레이어(layer)가 다양한 정보를 인코딩하는 방식을 분석합니다. 이를 통해 문장 표현에 대한 내부 신호를 추적하고, 구 속성의 변동이 문장 표현에 어떻게 반영되는지 조사합니다.

- **Performance Highlights**: 연구 결과는 transformer 모델이 문장 표현에서 언어 정보를 특정 부분에만 국한하여 인코딩하는 경향이 있음을 보여줍니다. 이는 향후 설명 가능한 신경 모델 개발에 중요할 수 있는 통찰을 제공합니다. 또한, 문장 표현의 로버스트니스(robustness)를 검증하고, 데이터의 얕은 규칙성에 의존하는지 여부를 확인할 수 있습니다.



### PEFT-U: Parameter-Efficient Fine-Tuning for User Personalization (https://arxiv.org/abs/2407.18078)
- **What's New**: 최근 Large Language Models (LLMs)의 등장으로 인간-AI 상호작용이 새로운 국면을 맞았습니다. 특히 Chat-GPT와 같은 모델들은 언어 이해에서 놀라운 성능을 보여주고 있습니다. 하지만 LLM의 급성장에도 불구하고, 개인화(personalization) 측면은 아직 충분히 연구되지 않았습니다. 이에 대응하여 우리는 PEFT-U Benchmark를 소개합니다. 이는 사용자 개인화를 위한 새로운 데이터셋으로, 사용자의 다채롭고 개별적인 표현을 포함한 유저 중심의 다양한 태스크를 제공합니다.

- **Technical Details**: PEFT-U Benchmark는 15,000명 이상의 사용자와 증오 발언, 감정/감성, 유머 등의 도메인에서 13개 이상의 개인화된 태스크로 구성되어 있습니다. 일반적인 NLP 리소스가 다수결 투표에 기반하여 데이터셋을 구성하는 것과 달리, PEFT-U는 각 주석자를 고유의 사용자로 간주하여 개별적인 관점을 포착합니다. 이를 통해 모델이 동일한 입력에 대해서도 다양한 사용자 관점을 반영할 수 있게 합니다. 또한, Krippendorff’s alpha 가장 높은 값이 0.5 이하인 데이터셋만 선택하여, 서로 다른 사용자 관점을 캡처하는 모델의 능력을 평가합니다.

- **Performance Highlights**: 우리는 PEFT-U 벤치마크에 대해 파라미터 효율적인(personalized) 방법과 zero/few-shot prompting 방법을 비교 평가했습니다. Flan-T5 모델을 사용하여 다음과 같은 방법들을 테스트했습니다: Zero-shot/Few-shot Prompting, LoRa, Adapters, Prompt Tuning 등. 결과적으로 personalized 모델들이 실제 사용자 관점을 더 정확하게 반영하는 결과를 보여주었습니다. 우리의 코드와 모델, 벤치마크는 모두 공개되어 있습니다.



### Difficulty Estimation and Simplification of French Text Using LLMs (https://arxiv.org/abs/2407.18061)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문은 생성적 대형 언어 모델(generative large language models)을 활용하여 외국어 학습 응용 프로그램에서 텍스트 난이도를 추정하고 이를 더 낮은 난이도로 단순화하는 방법을 제시합니다. 이는 기존 접근 방식보다 더 높은 정확성을 보여줍니다.

- **Technical Details**: 이 작업은 두 가지 예측 문제로 구성되며, 난이도 분류 모델은 레이블이 있는 예제를 사용하여 전이 학습(transfer learning)과 대형 언어 모델을 활용해 개발되었습니다. 단순화를 위해서는 단순화 품질과 의미 보존 간의 트레이드 오프(trade-off)를 평가하였으며, zero-shot과 fine-tuned 성능을 비교하였습니다.

- **Performance Highlights**: 프랑스어 텍스트에 대한 실험에서 제한된 fine-tuning으로도 의미 있는 텍스트 단순화가 가능함을 보여주었습니다. 또한, 제안된 방법은 특정 언어에 종속되지 않으며 다른 외국어에도 직접 적용할 수 있습니다.



### Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption (https://arxiv.org/abs/2407.18003)
Comments:
          to be published in CoLM 2024

- **What's New**: 최근 ChatGPT와 같은 대형 언어 모델(LLMs)의 발전으로 인해 여러 산업에서 언어 이해 능력이 눈에 띄게 향상되었습니다. 그러나 Transformer 아키텍처가 긴 텍스트를 처리하는 데 어려움을 겪으며, 특히 토큰 생성을 위한 시간 복잡성이 이슈가 됩니다. 이를 해결하기 위한 중요한 솔루션으로 KV-Cache가 등장했으며, 이 리뷰 논문에서는 KV-Cache의 다양한 최적화 방법을 분석하고 평가합니다.

- **Technical Details**: KV-Cache는 주목할 만한 솔루션으로, 주어진 텍스트의 길이에 비례해 GPU 메모리 오버헤드가 증가하는 것을 제외하면 토큰 생성의 시간 복잡성을 선형으로 줄일 수 있습니다. 이 리뷰는 모델의 사전 학습 단계, 배포 단계 및 추론 단계에서 사용되는 다양한 KV-Cache 최적화 방법을 다룹니다. 이러한 방법에는 사전 훈련 단계에서의 모델 압축, 배포 단계에서의 다양한 프레임워크 도입, 추론 단계에서의 에비션(Eviction) 및 양자화(Quantization) 방법이 포함됩니다.

- **Performance Highlights**: 효율성 측면에서는 모델 생성 속도와 공간 점유율 개선이 주요 평가 지표가 되며, 성능 측면에서는 모델의 능력 손실이 최소화 되는지를 중점적으로 평가합니다. KV-Cache 최적화는 특히 대규모 모델에서 메모리 사용을 줄이면서 동일한 성능을 유지하도록 도와줍니다.



### On the Effect of Purely Synthetic Training Data for Different Automatic Speech Recognition Architectures (https://arxiv.org/abs/2407.17997)
Comments:
          Accepted at the SynData4GenAI 2024 workshop

- **What's New**: 이 연구에서는 자동 음성 인식(Automatic Speech Recognition, ASR) 훈련에 합성 데이터의 유용성을 평가했습니다. FastSpeech-2와 유사한 텍스트-음성 변환(Text-to-Speech, TTS) 시스템을 통해 원본 데이터 세트를 재생산하고, 세 가지 ASR 아키텍처에 대해 합성 데이터만으로 훈련된 모델을 분석했습니다. 다양한 구조적 모델의 합성 데이터 민감도를 평가하고, 합성 및 실제 데이터 훈련의 차이를 연구했습니다.

- **Technical Details**: 연구에서는 비자기회귀(non-autoregressive) TTS 모델을 사용하여 합성 데이터를 생성했으며, GMM-HMM, Hybrid DNN-HMM, AED의 세 가지 ASR 아키텍처를 사용했습니다. TTS 모델에는 BLSTM 디코더와 혼합 합성(convex combination)의 인코더가 포함되어 있습니다. 합성 음성 데이터를 생성하기 위해 Griffin & Lim(G&L) vocoding을 사용했습니다. 모든 모델은 LibriSpeech-100h 데이터 셋을 기반으로 훈련되었습니다.

- **Performance Highlights**: 실험 결과, 합성 데이터로만 훈련된 ASR 모델의 성능은 상대적으로 낮았지만, 세부 조정을 통해 합성 데이터의 유용성을 증가시킬 수 있음을 보여주었습니다. 특히, 모델 크기를 단순히 증가시켜도 성능이 향상되었으며, 고성능 화자 임베딩 시스템을 도입함으로써 더 나은 결과를 얻을 수 있음을 확인할 수 있었습니다. 또한, GMM-HMM 모델은 상대적으로 합성 데이터에 덜 민감하다는 것이 밝혀졌습니다.



### What does Kiki look like? Cross-modal associations between speech sounds and visual shapes in vision-and-language models (https://arxiv.org/abs/2407.17974)
Comments:
          Appeared at the 13th edition of the Workshop on Cognitive Modeling and Computational Linguistics (CMCL 2024)

- **What's New**: 이번 연구에서는 VLM(Vision-and-Language Models)이 인간의 대표적인 교차 감각 선호도인 부바-키키 효과(bouba-kiki effect)를 인식하는지 조사했습니다. 이 연구를 통해 교차 감각 선호도와 관련된 비언어적-시각적 연상이 인간과 유사한지를 탐구했습니다.

- **Technical Details**: 이 연구는 Transformer 기반 아키텍처를 사용하는 네 개의 VLM을 대상으로 실험을 진행했습니다. 특정 교차 감각 패러다임을 사용하여 모델이 부바-키키 효과를 인식하는지 여부를 확인했습니다. 결과적으로 모델의 아키텍처 디자인, 크기, 그리고 학습 세부 사항이 부바-키키 효과의 존재 여부에 영향을 미칠 수 있음을 발견했습니다. 사용된 VLM 중에는 CLIP와 Stable Diffusion 모델도 포함되었습니다.

- **Performance Highlights**: 실험 결과, 부바-키키 효과가 모든 모델에서 일관되게 관찰되지 않았습니다. 이는 앞선 연구와는 다른 결과로, VLM의 특정 특성과 실험의 세부 사항에 따라 이 효과가 달라질 수 있음을 시사합니다. 이를 통해 앞으로 더 인간과 유사한 교차 감각 연상을 인식하는 VLM 개발에 기여할 수 있는 중요한 통찰을 제시했습니다.



### The Curious Case of Representational Alignment: Unravelling Visio-Linguistic Tasks in Emergent Communication (https://arxiv.org/abs/2407.17960)
Comments:
          Appeared at the 13th edition of the Workshop on Cognitive Modeling and Computational Linguistics (CMCL 2024)

- **What's New**: 이 논문은 언어 발생 시뮬레이션에서 나타나는 대리 표현 정렬(alignment) 문제를 다룹니다. 연구진은 에이전트 이미지 표현과 입력 이미지 간의 정렬을 평가하며, 정렬이 에이전트 간의 효과적인 소통과 어떤 관련이 있는지를 밝혀내었습니다. 새로운 발견으로는, 이미지 대리 표현의 정렬을 유지하면서도 구성적 판별 과제에서 성과를 향상시키지 못한다는 것입니다.

- **Technical Details**: 연구팀은 레퍼런셜 게임(referential game)에서 에이전트 간의 정렬을 평가하기 위해, 최신 비전 모듈(DinoV2)을 탑재한 강화 학습(Reinforcement Learning) 에이전트를 훈련시켰습니다. 주요히 사용된 데이터셋은 MS COCO와 Winoground였습니다. 연구진은 에이전트 간 정렬(inter-agent alignment)과 입력 이미지 정렬(image-agent alignment) 간의 상관관계를 확인했습니다. 또한, RSA(Representational Similarity Analysis)를 이용해 에이전트 간 이미지 해석의 정렬도를 측정했습니다.

- **Performance Highlights**: 에이전트 간 정렬은 높은 톱시밀러리티(topographic similarity)와 큰 연관이 있는 것으로 확인되었습니다. 정렬 페널티를 도입하여 임의적 이미지 특징 대신 이미지 자체에 대한 소통을 유도함으로써 높은 톱시밀러리티와 동등한 소통 성공률을 유지할 수 있었습니다. 이러한 새로운 접근법은 에이전트가 보다 인간과 유사한 개념적 속성을 포착하는데 도움이 될 수 있습니다.



### Positive Text Reframing under Multi-strategy Optimization (https://arxiv.org/abs/2407.17940)
- **What's New**: 이 논문은 기존의 감정 이전(senti transfer)과는 달리, 부정적인 관점을 긍정적인 표현으로 대체하며 원래 의미를 보존하는 '긍정적 재구성(positive reframing)'을 목표로 합니다. 이를 위해 다중 전략 최적화 프레임워크(MSOF)를 제안하였습니다. MSOF은 긍정적 감정 보상과 내용 보존 보상을 통해 모델이 부정적인 표현을 변환하면서 의미와 일관성을 유지하도록 설계되었습니다.

- **Technical Details**: 제안된 프레임워크는 BART 및 T5와 같은 Seq2Seq(pre-trained language models) PLM을 사용하여 긍정적 재구성 작업에서 성능을 개선하는 것을 목표로 합니다. 먼저 긍정적 감정 보상과 내용 보존 보상을 통해 원래 텍스트의 부정적 표현을 변환하면서 의미와 일관성을 유지하도록 모델을 훈련시킵니다. 그런 다음 다양한 디코딩 최적화 접근 방식을 도입하여 텍스트 생성의 질을 높이고, 마지막으로 전략 일관성, 텍스트 유사성, 유창성의 세 가지 차원을 기반으로 후보 문장을 재평가하는 다차원 재순위(re-ranking) 방법을 제안합니다.

- **Performance Highlights**: 제안된 MSOF 프레임워크는 비제한적 및 제한적 긍정적 재구성 작업 모두에서 중요한 개선을 보여주었습니다. BART 및 T5를 사용한 광범위한 실험 결과가 이 프레임워크의 효과를 입증하였습니다.



### Modelling Multimodal Integration in Human Concept Processing with Vision-and-Language Models (https://arxiv.org/abs/2407.17914)
- **What's New**: 이번 연구에서는 시각-언어 DNN 모델들(VLM)이 인간의 뇌 활동과 얼마나 잘 일치하는지를 조사했습니다. 참가자들이 문장 혹은 사진과 함께 단어를 읽을 때 fMRI 응답을 분석해, VLM을 통해 얻은 표현이 언어 전용 혹은 시각 전용 DNN보다 뇌 활성화와 더 강하게 상관성이 있다는 결과를 도출했습니다.

- **Technical Details**: 이번 연구는 참가자들이 전체 문장이나 동반되는 사진과 함께 개념 단어를 읽을 때 기록된 fMRI 응답을 분석했습니다. VLM, 언어 전용 DNN, 시각 전용 DNN의 표현을 비교하여 뇌와의 상관성을 조사했습니다. 특히, 최근의 생성적 VLM이 이전의 아키텍처보다 다운스트림 응용 성능이 낮음에도 불구하고 덜 뇌와 일치하는 경향이 있다는 점이 발견되었습니다.

- **Performance Highlights**: 추가 분석 결과, 대부분의 경우 행동 평가와 강하게 일치하는 표현은 뇌 반응과 높은 상관성을 가지지 않는다는 것을 보여주었습니다. 이는 행동적 유사성과 뇌 유사성이 반드시 일치하지 않는다는 것을 의미합니다.



### The Power of Combining Data and Knowledge: GPT-4o is an Effective Interpreter of Machine Learning Models in Predicting Lymph Node Metastasis of Lung Cancer (https://arxiv.org/abs/2407.17900)
- **What's New**: 최근에 발표된 논문에서는 대형 언어 모델(LLMs)과 기계 학습 모델을 결합하여 폐암 환자의 림프절 전이에 대한 예측 정확도를 높이는 새로운 앙상블 방법을 제안했습니다. LLMs의 의학적 지식을 기반으로 한 예측과 기계 학습 모델의 잠재적 패턴을 통합하여 성능을 개선합니다.

- **Technical Details**: 이 연구에서는 처음에 환자 데이터를 이용해 기계 학습 모델을 개발했으며, 환자 데이터와 예측 확률을 통합하는 프롬프트 템플릿을 설계했습니다. 이후 가장 최신의 GPT-4o를 이용하여 환자 데이터를 바탕으로 림프절 전이 가능성을 추정하고 기계 학습 모델의 출력을 조정했습니다. GPT-4o의 동일한 프롬프트를 세 번 실행하여 최종 예측으로 앙상블화했습니다.

- **Performance Highlights**: 제안된 방법을 통해 모델들은 LNM(Lymph Node Metastasis) 예측에 대한 AUC 값 0.765 및 AP 값 0.415를 달성하여, 기존 기계 학습 모델과 비교했을 때 예측 성능이 크게 향상되었습니다.



### A Large-Scale Sensitivity Analysis on Latent Embeddings and Dimensionality Reductions for Text Spatializations (https://arxiv.org/abs/2407.17876)
Comments:
          To be published at IEEE VIS 2024 conference

- **What's New**: 이 논문은 문서 간 의미 유사성을 2차원 산점도(scatterplot) 형태로 시각화할 때, 입력 데이터와 하이퍼파라미터의 변동에 따라 레이아웃이 얼마나 안정적인지에 대한 감도 분석(sensitivity study)을 제시합니다. 이를 통해 문서 시각화에서 일관성을 유지할 수 있는 가이드라인을 제공합니다.

- **Technical Details**: 이 연구는 두 단계로 구성됩니다. 첫째, 세 가지 텍스트 코퍼스와 여섯 가지 텍스트 임베딩(text embeddings)에 대해 그리드 검색(grid-search) 기반 하이퍼파라미터 선택을 통해 레이아웃을 도출했습니다. 그런 다음, 10가지 메트릭을 사용해 지역(local) 및 글로벌(global) 구조와 클래스 분리를 평가했습니다. 두 번째로, 생성된 42817개의 데이터 포인트를 기술 통계 분석(descriptive statistical analysis)을 통해 분석했습니다.

- **Performance Highlights**: 이 연구는 하이퍼파라미터 설정에 따른 민감도를 분석하여 사용자가 문서 시각화를 더 일관성 있게 유지할 수 있는 지침을 제공합니다. 주요한 발견 중 일부는 특정 하이퍼파라미터 설정이 레이아웃의 안정성에 긍정적인 영향을 미친다는 것입니다. 이로써 사용자는 텍스트 코퍼스 시각화를 보다 신뢰성 있게 수행할 수 있게 됩니다.



### Improving Domain-Specific ASR with LLM-Generated Contextual Descriptions (https://arxiv.org/abs/2407.17874)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 최신 연구는 end-to-end 자동 음성 인식 (ASR) 시스템의 한계를 극복하기 위한 방법을 제안합니다. 새로운 접근 방식은 Whisper를 사용하면서 아키텍처를 변경하지 않고 설명을 효과적으로 활용할 수 있도록 합니다. 또한, 두 가지 추가 훈련 기술인 디코더 세분화(decoder fine-tuning) 및 컨텍스트 간섭(context perturbation)을 통해 도메인 특정 ASR 성능을 향상시킵니다. 기술 설명이 없는 경우에는 대형 언어 모델(LLM)을 사용하여 메타 데이터를 기반으로 설명을 생성하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서 제안한 방법은 Whisper의 아키텍처를 변경하지 않고 설명을 입력 프롬프트(prompt)로 사용하여 도메인 특정 ASR 성능을 개선합니다. Whisper는 트랜스포머(Transformer) 기반의 인코더와 디코더를 사용하는데, 인코더는 오디오 입력을 처리하고 디코더는 텍스트를 생성합니다. 제안된 방식은 디코더에 설명을 프롬프트로 입력하여 설명을 활용할 수 있도록 합니다. 추가적인 모듈 없이 Whisper를 미세 조정(fine-tuning)하는 방식으로 도메인 특정 데이터를 사용하여 성능을 향상시킬 수 있습니다. 또 다른 기술적 접근으로 컨텍스트 간섭(context perturbation)을 제안하며, 이는 도메인 특정 단어가 포함되지 않은 발화를 처리할 때 설명을 무시할 수 있도록 모델을 훈련시킵니다. 마지막으로, LLM을 사용하여 음성 파일의 간단한 메타 데이터를 기반으로 설명을 생성하여 세부적인 설명을 자동으로 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 실제 데이터 세트에서 도메인 특정 ASR 정확도를 현저히 향상시킴을 보여줍니다. 특히, LLM으로 생성된 설명이 사람이 작성한 설명보다 더 효과적으로 도메인 특정 ASR 성능을 높이는 것으로 나타났습니다. Earnings Call 데이터셋과 MIT OpenCourseWare 데이터셋을 사용한 실험에서 이 방식의 우수성을 입증하였습니다.



### factgenie: A Framework for Span-based Evaluation of Generated Texts (https://arxiv.org/abs/2407.17863)
Comments:
          Accepted to INLG 2024 (System Demonstrations)

- **What's New**: factgenie는 텍스트 모델 출력에서 단어 범위를 주석(annotation)하고 시각화하기 위한 프레임워크입니다. 주석은 의미적 부정확성이나 관련 없는 텍스트와 같은 다양한 현상을 캡처할 수 있습니다. 인간 크라우드 워커(crowdworkers)와 대형 언어 모델(LLMs) 모두에서 주석을 수집할 수 있습니다.

- **Technical Details**: factgenie는 데이터 시각화와 텍스트 주석 수집을 위한 웹 인터페이스로 구성되어 있으며, 쉽게 확장 가능한 코드베이스를 갖추고 있습니다. Flask 백엔드와 HTML 기반 프론트엔드로 구성되며, Boostrap 5.3, jQuery, YPet 라이브러리 및 Highcharts.JS를 사용합니다. 다양한 데이터 집합형식을 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: factgenie는 입력 데이터와 모델 출력을 몇 줄의 코드로 시각화할 수 있으며, 클라우드 소스 서비스로부터 주석을 수집하기 위한 웹 인터페이스를 제공합니다. 여러 LLM API에서 모델 기반 주석을 수집할 수 있으며, 수집된 주석을 관리 및 시각화할 도구도 제공합니다. Ollama API와 OpenAI API를 사용하여 클라우드 및 셀프 호스티드 LLM에서 주석을 수집할 수 있습니다.



### Exploring Description-Augmented Dataless Intent Classification (https://arxiv.org/abs/2407.17862)
Comments:
          Accepted to the 6th NLP for Conversational AI Workshop at ACL 2024(NLP4ConvAI)

- **What's New**: 이번 연구에서는 최신 텍스트 임베딩 모델(State-of-the-Art, SOTA)을 활용한 데이터리스(데이터가 없는) 의도 분류(Dataless Intent Classification) 방법을 여러 가지 도입하였습니다. 이전 연구들과 비교하여 네 가지 일반적으로 사용되는 의도 분류 데이터셋에서 우리 방법의 결과를 보고합니다. 라벨링된 데이터나 특정 작업을 위한 데이터에 대한 학습 없이 강력한 Zero-shot 기준점에 비해 평균 6.12%의 성능 향상을 포함한 유의미한 개선 성과를 나타냈습니다. 또한, 이 방법론의 단점을 보여주는 정성적 오류 분석을 제공하여 향후 연구를 위한 가이드를 제시합니다.

- **Technical Details**: 최근 텍스트 임베딩 모델의 발전을 활용하여, 의도 설명(intent descriptions)을 사용하여 의도의 중간 텍스트 표현을 생성하는 여러 가지 접근법을 도입했습니다. 이 방법을 공식화하고, 대화 시스템(TODS)의 의도 분류 구성 요소에서 새로운 의도 설명 생성 방식을 도입하여 인간 전문가의 입력을 최소화하려 했습니다. 주어진 발화(utterance)에 대해 최근 유사도 계산(task간 용어로 Cosine Similarity를 사용) 기술을 사용하여 가장 유사한 의도 클래스를 예측합니다.

- **Performance Highlights**: 우리의 의도 설명 접근법은 라벨 토큰화(label tokenization)보다 유의미한 개선 결과를 보여줍니다. 발화 파라프레이징(utterance paraphrasing) 및 마스킹 접근법을 도입하여 추가 개선을 이루었으며, 이는 다양한 모델에서 일관되게 나타났습니다. 네 가지 널리 사용되는 의도 분류 데이터셋에 대해 광범위한 평가를 수행하여 경쟁력 있는 성과를 달성했습니다.



### Scaling A Simple Approach to Zero-Shot Speech Recognition (https://arxiv.org/abs/2407.17852)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 무라벨(unlabeled) 텍스트 데이터만을 사용하여 zero-shot 접근 방식을 통해 1078개 언어에 대한 음향 모델을 제안하는 MMS Zero-shot을 소개합니다. 이 모델은 직관적으로 단순하며 매핑 과정에서 흔히 발생하는 오류를 줄이기 위해 로마자화(romanization)를 사용합니다.

- **Technical Details**: MMS Zero-shot은 uber-rom이라는 로마자화 도구를 활용하여 모든 텍스트를 하나의 통합된 라틴 문자로 변환합니다. 이를 바탕으로 다중언어에 대해 사전 학습된 wav2vec 2.0 모델을 fine-tuning하여 새로운 언어에도 일반화할 수 있도록 했습니다. 또한, lexicon 기반의 단순한 단어 매핑 방식을 사용하여 단어를 모델 출력과 일치시킵니다.

- **Performance Highlights**: MMS Zero-shot은 이전의 최고 성능 모델에 비해 평균 문자 오류율(Character Error Rate, CER)을 100개 미지의 언어에 대해 상대적 46% 감소시켰습니다. 또한, in-domain supervised 모델에 비해 오류율이 2.5배 높은 수준을 유지하면서도 평가 언어에 대해 무라벨 데이터만을 사용하여 평가를 진행했습니다.



### Demystifying Verbatim Memorization in Large Language Models (https://arxiv.org/abs/2407.17817)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 정확한 초문을 외우는 능력에 대한 분석을 진행하였습니다. 연구팀은 Pythia 체크포인트에서 훈련 데이터를 지속적으로 주입하여 이를 통제된 환경에서 연구하는 프레임워크를 개발했습니다. 이를 통해 특정한 여러 반복이 있어야만 초문 암기가 일어나며, 나중(더 성숙한) 체크포인트들이 이러한 초문을 더 많이 기억한다는 점과 암기된 시퀀스의 생성이 고급 언어 모델링 능력에 크게 의존한다는 것을 발견했습니다. 이러한 통찰에 기반해, 연구팀은 '비학습(unlearning)' 방법을 스트레스 테스트하여 제거하려 시도하는 암기된 정보를 제거하지 못하거나 모델 성능을 저하시키는 경우가 많음을 발견했습니다.

- **Technical Details**: 연구진은 Pythia 모델군을 이용하여, 원본 훈련 데이터에 새로운 시퀀스를 주입하여 실험을 수행했습니다. 실험 결과 (1) 시퀀스가 암기되기 위해서는 비트리비얼(non-trivial) 양의 반복이 필요하다는 것을 확인했고, (2) 나중에 혹은 더 성숙한 체크포인트들이 초문을 더 많이 암기하는 경향이 있으며, 이는 도메인 외 시퀀스에서도 마찬가지였습니다. (3) 암기된 시퀀스의 일부 토큰은 고급 의미적 특징을 인코딩하는 분산된 트리거 상태에 의존하며, 나머지는 일반적인 언어 모델 디코딩을 통해 생성됩니다.

- **Performance Highlights**: 모델의 특정 가중치나 메커니즘이 초문 암기를 담당한다는 기존의 가설에 도전하여, 초문 암기가 데이터와 언어 모델링의 일반적인 기능과 관련된 많은 상호작용 요인에 의해 발생된다는 것을 발견했습니다. 따라서, 암기된 정보를 성능 저하 없이 제거하는 것은 매우 어려운 과제로 남아있습니다.



### Banyan: Improved Representation Learning with Explicit Structur (https://arxiv.org/abs/2407.17771)
Comments:
          First Draft

- **What's New**: Banyan은 데이터에 대한 명시적 구조를 유도하여 의미 표현을 학습하는 향상된 모델입니다. Banyan은 여러 구성 요소 구조를 공유된 구조로 결합하여 글로벌 컨텍스트를 명시적으로 통합합니다. Griffin에서 영감을 얻은 향상된 메시지 패싱(schema)을 결합하여, Banyan은 더 나은 표현을 학습하고, 대조 학습(contrastive learning)을 통해 잘못된 부정 오류를 피하며, 메모리 효율성을 크게 향상합니다. Self-StrAE 프레임워크를 사용하여 Banyan은 다양한 설정에서 전통적인 모델을 능가하며, 제한된 자원으로 사전 훈련된 GloVe 및 RoBERTa와 같은 비구조적 기준 모델을 능가하거나 일치합니다.

- **Technical Details**: Banyan은 Self-StrAE 모델을 변형한 것으로, 초기 자극을 토큰으로 변환하여 시작합니다. 그 후, 코사인 유사성(cosine similarity)을 사용하여 토큰을 쌍으로 병합하는 절차를 포함하며, 이는 루트 임베딩에 도달할 때까지 반복됩니다. 그런 다음, 각 노드에서 임베딩을 분해하는 함수(DTheta)를 사용하여 임베딩을 복구합니다. 이 모델은 또한 새로운 형태의 구조인 '엉킨 나무(entangled trees)'와 SSM을 반영한 메시지 패싱 기능을 도입하여 성능과 계산 효율성을 개선했습니다.

- **Performance Highlights**: Banyan 모델은 다양한 설정에서 기존의 Self-StrAE와 기타 비구조적 기준 모델을 능가합니다. 구체적으로, Asian 및 African과 같은 저자원 언어에서의 SemRel 과제에서 효과적인 표현을 학습했습니다. 또한 상대적으로 적은 수의 매개변수로도 효과적인 성능을 발휘하여 데이터와 메모리 효율성을 유지했습니다.



### BotEval: Facilitating Interactive Human Evaluation (https://arxiv.org/abs/2407.17770)
Comments:
          ACL 2024 SDT, 10 pages

- **What's New**: 자연어 처리(NLP) 모델의 최근 발전으로 인해 협상, 대화 중재 등의 복잡한 상호작용 작업에 NLP 모델이 점점 더 많이 적용되고 있습니다. BotEval이라는 새로운 평가 도구가 개발돼 이러한 상호작용 작업에서 인간 평가자가 NLP 모델과 직접 상호작용할 수 있는 환경을 제공합니다. 이는 기존의 정적 입력 평가 방식과는 다르게 인간-봇 상호작용을 통한 평가를 가능하게 합니다.

- **Technical Details**: BotEval은 다음과 같은 주요 기능을 제공합니다: 1) 대화 패널, 지침 패널, 설문 패널 등 구성요소를 HTML 및 YAML 파일을 통해 쉽게 커스터마이즈 가능. 2) Amazon Mechanical Turk(AMT) 및 Prolific과 같은 크라우드소싱 플랫폼과의 내장 호환성. 3) 평가 작업을 관리하고 모니터링할 수 있는 관리자 대시보드 제공. 4) 다수의 인간 및 봇 간 상호작용 설정을 위한 동적 구성 가능.

- **Performance Highlights**: BotEval을 통해 다양한 챗봇의 대화 중재 성능을 평가한 사례 연구에서 BotEval의 유용한 기능들이 잘 드러났습니다. 특히, 직접 상호작용을 통해 평가된 모델이 더욱 효과적인 대화 중재자로 인식된 반면, 모델과 이용자 간의 완성된 상호작용을 평가한 경우는 다른 시각의 결과가 도출되었습니다.



### Beyond Entity Alignment: Towards Complete Knowledge Graph Alignment via Entity-Relation Synergy (https://arxiv.org/abs/2407.17745)
- **What's New**: 최근 지식 그래프 정렬(Knowledge Graph Alignment, KGA) 분야에서는 새로운 접근법인 EREM이 제안되었습니다. EREM은 기존 모델들이 주로 엔티티(entity) 정렬에 중점을 두고 있었던 반면, 이 접근법은 엔티티 정렬과 관계 정렬(relation alignment)을 별도로 다루어, 완전한 KGA를 이루는 것을 목표로 합니다. 이 모델은 두 하위 작업을 기대-최대화(Expectation-Maximization) 프레임워크를 통해 상호 강화하며 최적화합니다.

- **Technical Details**: EREM 모델은 두 개의 주요 모듈로 구성됩니다: 엔티티 정렬(Entity Matching, E-step) 모듈과 관계 정렬(Relation Matching, M-step) 모듈입니다. 이 두 모듈은 변분 기대-최대화(Variational EM) 모델로 정의되어 있으며, 두 작업의 상호 강화 상관관계를 포착하도록 설계되었습니다. E-step에서는 관계 정렬 앵커를 이용해 엔티티 정렬을 최적화하고, M-step에서는 예측된 엔티티 앵커를 이용하여 관계 정렬을 최적화합니다. 이 과정은 최적 수송(Optimal Transport, OT) 정렬 작업으로 공식화되어 있으며, Sinkhorn 알고리즘을 이용해 효율적으로 해결됩니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험 결과, EREM은 엔티티 정렬과 관계 정렬 작업에서 최첨단(SOTA) 모델들보다 일관되게 뛰어난 성능을 보였습니다. 특히, EREM은 7개의 SOTA 엔티티 정렬 방법과 자연스럽게 통합되어 성능 향상을 일관되게 보여주었습니다. 이는 EREM의 통합된 접근 방식이 엔티티와 관계 정렬 간의 시너지 효과를 극대화한다는 것을 입증합니다.



### Are Large Language Models Possible to Conduct Cognitive Behavioral Therapy? (https://arxiv.org/abs/2407.17730)
- **What's New**: 최근 연구는 감정 장애 인식 및 개입에 대한 대형 언어 모델(LLMs)의 활용 가능성을 증명하며 새로운 심리적 지원 치료 가능성을 제시했습니다. 실제 CBT 코퍼스를 이용하여 LLM의 CBT 수행 능력을 평가하기 위해 감정 경향 점수, 구조화된 대화 패턴, 그리고 PQA(Proactive Questioning Ability) 평가 메트릭스를 사용한 종합적인 자동 평가 프레임워크를 설계 및 수행했습니다.

- **Technical Details**: 본 연구에서는 온라인 비디오 웹사이트에서 실제 CBT 코퍼스를 수집하고, 개별 모델이 생성한 텍스트의 감정 경향 점수를 계산했습니다. 구조화된 대화 패턴 평가를 위해 다양한 자동 평가 메트릭스를 사용했으며, PQA 메트릭스를 통해 환자를 유도하는 능력을 평가했습니다. 또한, CBT 지식 베이스를 통합하여 모델의 CBT 상담 능력을 향상시키는 추가적인 연구도 수행했습니다.

- **Performance Highlights**: 실험 결과, 자연어 처리에서 뛰어난 성능을 보이는 네 가지 LLM 변형이 평가되었으며, 특히 추가적인 그 기술과의 융합 후에 심리상담 분야에서 LLM의 큰 잠재력이 확인되었습니다.



### Examining the Influence of Political Bias on Large Language Model Performance in Stance Classification (https://arxiv.org/abs/2407.17688)
Comments:
          Accepted at ICWSM 2025

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM: Large Language Models)이 정치적 입장 분류 작업에서 어떻게 편향을 보이는지를 조사했습니다. LLM들은 잘 훈련된 데이터셋으로부터 학습되었지만, 이러한 모델들은 인종, 국적 및 성별 편향을 내포하고 있습니다. 이 연구는 LLM들이 정치적으로 민감한 입장 분류 작업에서 더 정확하게 분류하는 경향을 보이는지에 대해 분석했습니다.

- **Technical Details**: 연구진은 세 가지 데이터셋과 일곱 가지 LLM, 네 가지 다른 프롬프트(prompting) 방식을 사용하여 정치적으로 중요한 발언과 대상에 대한 LLM의 성능을 평가했습니다. 사용된 데이터셋은 BASIL, SemEval2016, Elections2016였으며, 각 데이터셋에는 정치적으로 명확한 입장이 포함된 텍스트가 담겨 있습니다. LLM은 각 발언이 특정 정치적 입장과 관련이 있는지를 분류하는 작업을 수행했습니다. 사용된 프롬프트 방식에는 'task-only prompt,' 'task + context prompt,' 'zero-shot chain-of-thought (CoT)' 등이 포함되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 정치적으로 명확한 발언을 분류하는 데 있어 LLM의 성능에 통계적으로 유의미한 차이가 있음을 발견했습니다. 특히 데이터셋 수준에서 이러한 차이가 두드러졌으며, 모델과 프롬프트 방식은 서로 다른 입장 분류 데이터셋에서 유사한 성능을 보였습니다. 또한, 발언이 향하는 대상이 더 모호할 때, LLM의 입장 분류 정확도는 낮아지는 경향을 보였습니다.



### Efficient LLM Training and Serving with Heterogeneous Context Sharding among Attention Heads (https://arxiv.org/abs/2407.17678)
Comments:
          10 pages

- **What's New**: 최근 LLM(Large Language Model) 훈련 및 추론 프레임워크는 희소성을 통해 효율성을 높이면서도 컨텍스트와 모델 아키텍처의 무결성을 유지하는 데 어려움을 겪고 있습니다. 데이터베이스의 샤딩(sharding) 개념과 주의(attention)가 가속기에서 헤드별로 병렬 처리된다는 사실에 영감을 받아, S2-Attention이라는 새로운 주의 알고리즘을 제안합니다. S2-Attention은 서로 다른 주의 헤드에 이질적인 컨텍스트 파티션을 할당하여 문제를 분산 처리합니다.

- **Technical Details**: S2-Attention은 각각의 주의 헤드가 스트라이드 희소성 패턴에 따라 컨텍스트의 일부만을 참조하도록 강제하여, 풀 컨텍스트는 모든 샤드의 합집합으로 보존됩니다. 주의 헤드는 별도의 스레드 블록에서 처리되므로 각 헤드의 컨텍스트 감소가 전체 속도와 메모리 사용량을 줄입니다. 추론 시, S2-Attention으로 훈련된 LLM은 리프리젠테이션 유지와 함께 KV 캐시(KV cache) 감소의 이점을 누릴 수 있습니다. 실험 결과, S2-Attention은 FlashAttention-2 대비 최대 25.3배의 워크 클락 속도 향상, 6배의 훈련 시간 단축, 10배의 추론 지연 감소를 보여줍니다.

- **Performance Highlights**: S2-Attention은 FlashAttention-2 대비 128K 시퀀스 길이에서 1.3B 모델에 대해 최대 8.8배의 워크 클락 속도 향상을, 70B 모델에 대해 최대 25.3배의 속도 향상을 가져옵니다. 추론 속도는 1M 컨텍스트 윈도우에서 최대 12배 향상됩니다. S2-Attention은 DKernel이라는 최적화된 커널 라이브러리와 함께 제공되며, Megatron, Pytorch, vLLM 등과 호환됩니다.



### Time Matters: Examine Temporal Effects on Biomedical Language Models (https://arxiv.org/abs/2407.17638)
Comments:
          Accept to AMIA 2024 Annual Symposium

- **What's New**: 이번 연구는 언어 모델을 바이오메디컬(의생명) 응용 분야에 적용할 때 발생하는 시간적 영향을 조사합니다. 특히, 모델은 역사적인 데이터를 기반으로 훈련되며 새로운 또는 미래의 데이터에 배포될 예정인데, 이 데이터는 훈련 데이터와 다를 수 있습니다. 본 연구에서는 바이오메디컬 모델 성능과 데이터 변동 간의 관계를 통계적으로 분석하여 이 격차를 메워줍니다.

- **Technical Details**: 연구는 세 가지 바이오메디컬 작업 전반에서 언어 모델의 성능과 데이터 변동 측정을 위해 다양한 지표(metric)를 사용합니다. 또한, 거리 방법(distance methods)을 통해 데이터 드리프트를 측정하고, 통계적 방법(statistical methods)을 통해 시간적 영향을 정량화합니다. 이는 바이오메디컬 언어 모델 배포 시 시간의 중요성을 보여주고, 각 작업 및 통계적 계량화 접근 방식에 따라 성능 저하의 정도가 달라짐을 나타냅니다.

- **Performance Highlights**: 연구 결과에 따르면, 바이오메디컬 언어 모델을 배포할 때 시간적 요소가 중요한 역할을 한다는 결론을 얻었습니다. 특히, 모델 성능 저하의 정도는 바이오메디컬 작업별로, 그리고 통계적 계량화 방법별로 다릅니다. 이 연구는 바이오메디컬 언어 모델의 시간적 영향을 평가하고 검토하기 위한 견고한 벤치마크를 확립할 수 있을 것으로 기대됩니다.



### IgnitionInnovators at "Discharge Me!": Chain-of-Thought Instruction Finetuning Large Language Models for Discharge Summaries (https://arxiv.org/abs/2407.17636)
Comments:
          Accepted by BioNLP2024 Workshop

- **What's New**: 이번 논문에서는 BioNLP 워크숍에서 열리는 'Discharge Me!' 공유 과제에 대한 우리의 접근 방식을 제안합니다. 이 연구에서는 병원 퇴원 요약서 문서화(Discharge Summary Documentation, DSD) 작업을 해결하기 위해 LLM(대규모 언어 모델) 기반 프레임워크를 개발했습니다. 최근의 인스트럭션 파인튜닝을 통해 LLM을 최적의 프롬핑 전략으로 특정 생성 작업인 DSD에 적응시키는 방법을 탐구했습니다. 실험 결과, 명확한 출력 구조와 함께 종합적인 Chain-of-Thoughts(CoT) 질문 세트를 제공하는 것이 모델의 추론 능력을 효과적으로 향상시켜 생성된 텍스트의 구조적 정확성과 임상 정보의 신뢰성을 높이는 것으로 나타났습니다.

- **Technical Details**: Discharge-LLM이라는 프레임워크를 제안하며, Section Extraction, Radiology Report Selection, Target Section Generation의 세 가지 단계로 구성됩니다. 이를 통해 환자의 병원 방문 정보와 방사선 보고 내용을 바탕으로 'Brief Hospital Course'와 'Discharge Instructions' 섹션을 생성합니다. 이를 위해 LLM을 인스트럭션 파인튜닝하여 각 생성 작업에 적응시켰습니다. 파라미터 효율적 파인튜닝을 위해 LoRA(Low-Rank Adaptation)를 사용했습니다. 프롬프트는 다섯 부분 구조로 나뉘며, CoT 질문을 통합해 LLM이 중요한 정보를 효과적으로 생성하도록 유도합니다.

- **Performance Highlights**: 세 가지 베이스라인을 설정하여 성능을 평가했습니다: Discharge_LLMBase, Discharge_LLMContext, Discharge_LLMCoT. Mistral 모델을 사용하여 각각의 생성 작업에 대해 인스트럭션 파인튜닝을 수행했으며, NVIDIA RTX 4090 GPU에서 10시간이 소요되었습니다. BLEU-4 등 여러 지표를 사용해 텍스트 유사성과 사실 정확성을 측정했습니다. 실험 결과, CoT 질문을 포함한 프롬프트 설계가 모델의 성능을 크게 향상시켰습니다.



### Papilusion at DAGPap24: Paper or Illusion? Detecting AI-generated Scientific Papers (https://arxiv.org/abs/2407.17629)
Comments:
          to appear in DAGPAP 2024 proceedings

- **What's New**: Papilusion은 자동 생성된 과학 텍스트를 탐지하기 위한 AI 시스템입니다. 이 시스템은 DAGPap24에서 6등을 차지하였으며, 이후 F1 점수를 99.46점으로 개선하여 큰 성과를 이뤘습니다.

- **Technical Details**: Papilusion은 토큰 레벨(token-level) 분류 문제를 해결하기 위한 시스템으로, 여러 개의 인코더 전용 시퀀스 태그 모델들을 앙상블(ensemble)하여 정확성을 높였습니다. 모델 파인튜닝(fine-tuning)과 다수결 투표(majority voting) 방법을 사용하여 텍스트 분류의 정확성과 신뢰성을 높였습니다. 참고로, 사용된 모델은 DeBERTaV3입니다.

- **Performance Highlights**: 경쟁 후 약간의 수정과 최적화를 통해 공식 테스트 세트에서 F1 점수를 99.46 (+9.63)로 향상시켰으며, 이 점수는 DAGPap24 경쟁에서 2등과 비교될 만한 성과입니다. 특히 가장 작은 DeBERTa 모델도 F1 점수 98 이상을 기록하였고, 최대 모델에서는 99.21의 점수를 기록하였습니다. 이러한 성과는 문제가 상대적으로 복잡하지 않음을 시사합니다.



### Coupling Speech Encoders with Downstream Text Models (https://arxiv.org/abs/2407.17605)
- **What's New**: 이번 연구에서는 새로운 'exporter' layer를 도입하여 기존의 1-best cascade를 성능 저하 없이 개선한 모듈식 cascade 음성 번역(Automatic Speech Translation, AST) 모델을 제시합니다. 이 'exporter' layer는 ASR 임베딩과 MT 토큰 임베딩 사이의 일치성을 L2-loss를 통해 보장하며, 이를 통해 ASR과 MT 모델 간의 백프로파게이션이 가능해집니다.

- **Technical Details**: ASR 모델은 Conformer 인코더(Gulati et al., 2020) 구조를 사용하며, 많은 음성 데이터로 사전 훈련된 Unified Speech Model (USM)(Zhang et al., 2023)을 기반으로 합니다. Connectionist Temporal Classification (CTC)(Graves et al., 2006)를 활용해 1-best 레이블과 정렬을 계산합니다. MT 모델은 표준 인코더-디코더 트랜스포머 구조를 사용하며, 회전 위치 임베딩(rotary position embeddings)(Su et al., 2022)을 포함합니다. 'exporter' layer는 L2-loss로 훈련되어 1-best 시퀀스에 대해 ASR 임베딩과 MT 토큰 임베딩 간의 강한 일치성을 보장합니다.

- **Performance Highlights**: 이 접근법은 WMT 데이터로 훈련된 MT 백엔드를 사용할 때 AST 성능 향상을 보여주었으나, MuST-C 백엔드에서 그 이점은 사라졌습니다. 이는 'exporter' layer가 오류가 있는 ASR 전사를 MT 모델에 피드하는 영향을 완화하기보다는 작업 적응(task adaptation)을 수행함을 보여줍니다. 이 연구는 ASR 인코더와 불변 텍스트 모델(immutable text models), 예를 들어 대규모 언어 모델(LLM)을 결합하려는 시나리오에서도 유용할 가능성을 제시합니다.



### Generative artificial intelligence in dentistry: Current approaches and future challenges (https://arxiv.org/abs/2407.17532)
- **What's New**: 이 논문은 생성형 AI (GenAI) 모델이 치과 분야에 미치는 영향을 다루고 있습니다. 생성형 AI 모델은 자연어 인터페이스를 통해 복잡한 모델과 상호 작용할 수 있게 하여 텍스트 생성에서 영상 생성까지 다양한 기능을 제공합니다. 이러한 기술 발전은 치과 교육, 환자 치료 개선, 그리고 치과 연구에 다양한 응용 가능성을 열어줍니다.

- **Technical Details**: 논문은 먼저 생성형 AI 모델의 정의와 다양한 생성 모달리티에 대해 설명합니다. 텍스트 생성, 이미지 및 비디오 생성 등 사용자가 입력한 텍스트 기반으로 다양한 결과물을 생성할 수 있는 모델의 기능을 소개합니다. 그리고 이 모델들이 치과 교육, 환자 치료, 연구 분야에서 어떻게 활용되는지 논의합니다.

- **Performance Highlights**: 치과 교육에서는 학생들이 다양한 질문에 대해 매우 신속하게 답을 얻을 수 있습니다. 또한, 치과 의사들이 지식 습득을 보다 빠르고 효율적으로 할 수 있어 더 나은 환자 치료를 제공하는데 도움이 됩니다. 연구 분야에서는 신약 개발부터 학술 논문 작성 지원까지 다양한 응용이 가능합니다.



### The #Somos600M Project: Generating NLP resources that represent the diversity of the languages from LATAM, the Caribbean, and Spain (https://arxiv.org/abs/2407.17479)
Comments:
          16 pages and 10 figures in total, includes English and Spanish version, to be published in proceedings North American Chapter of the Association for Computational Linguistics Conference: LatinX in AI (LXAI) Research Workshop, Mexico City, Mexico, June 2024

- **What's New**: 600백만 명의 스페인어 사용자를 대표하기 위해 #Somos600M 프로젝트가 시작되었습니다. 이 프로젝트는 인공지능(AI) 시스템에서 라틴 아메리카(LATAM), 카리브해, 스페인의 다양한 언어를 대표하는 것을 목표로 합니다. 첫 번째 버전의 명령 및 평가 데이터셋을 국제 오픈 소스 커뮤니티로서 만들었습니다.

- **Technical Details**: 프로젝트의 주요 목표는 다음과 같습니다. 첫째, LLMs(대규모 언어 모델)을 명령 지침(instruction)으로 미세 조정(fine-tuning)할 수 있는 오픈 명령 데이터셋을 만드는 것입니다. 둘째, 스페인어와 공동 공용어로 생성된 LLMs를 평가하기 위한 표준화된 리더보드를 구축하는 것입니다. 이를 위해 해커톤, 데이터셋 수집 캠페인, 영어 평가 데이터셋의 번역 및 검증 작업이 수행되었습니다.

- **Performance Highlights**: 해커톤에서 18개의 프로젝트가 제출되었고 총 2,333,052개의 예제가 만들어졌습니다. 프로젝트 주제는 콜롬비아 항공 규정, 난민 법률 지원, 페루 헌법 등 다양한 주제를 다루었습니다. 첫 번째 데이터 수집 캠페인에서 5개의 평가 데이터셋이 전문가에 의해 수동으로 주석이 달렸고, 두 번째 캠페인에서는 14개가 추가되어 새로운 언어들이 포함되었습니다. Okapi 번역 검증에는 총 61명이 참여하며, ARC-C, HellaSwag 및 MMLU의 다양한 퍼센트가 커버되었습니다. DIBT의 프롬프트는 100% 검증되었으며, 스페인어가 첫 번째로 완전히 검증된 언어가 되었습니다.



### LoRA-Pro: Are Low-Rank Adapters Properly Optimized? (https://arxiv.org/abs/2407.18242)
- **What's New**: 이번 논문은 Low-Rank Adaptation (LoRA) 방식의 성능 격차를 줄이기 위한 새로운 방법으로 LoRA-Pro를 제안합니다. LoRA를 사용한 파라미터 효율성 튜닝에서 원본 행렬을 두 개의 저차원 행렬의 곱으로 재매개변수화 하지만, 기존의 최적화 과정을 제대로 반영하지 못하여 성능이 떨어지는 문제를 해결합니다.

- **Technical Details**: LoRA-Pro는 '동등 기울기 (equivalent gradient)'라는 개념을 도입하여, LoRA의 저차원 근사해가 전체 파인튜닝에 가까운 최적화 과정을 따르도록 합니다. 이 과정에서 매개변수 A와 B의 기울기를 통해 LoRA와 전체 파인튜닝 간의 차이를 최소화합니다. 이 문제를 최적화 문제로 설정하고, 이론적인 해결책을 구해 LoRA의 성능 향상을 실현합니다.

- **Performance Highlights**: 자연어 처리 작업에 대한 광범위한 실험을 통해 LoRA-Pro의 효과가 검증되었고, 기존 LoRA 대비 성능 격차가 줄어들어 더욱 효율적이고 정확한 파인튜닝이 가능함을 보였습니다.



### Recursive Introspection: Teaching Language Model Agents How to Self-Improv (https://arxiv.org/abs/2407.18219)
- **What's New**: 이번 연구의 핵심은 LLMs (대형 언어 모델)가 반복적 자기 성찰(recursive introspection)을 통해 자신의 답변을 개선하고 오류를 수정할 수 있도록 하는 RISE(Recursive IntroSpEction)라는 새로운 방법론을 개발한 것입니다. 이를 통해 모델은 어려운 문제에 여러 번 도전하면서 서서히 정답에 다가가는 능력을 갖추게 됩니다.

- **Technical Details**: RISE는 온라인 모방 학습(online imitation learning)과 강화 학습(reinforcement learning)의 원리에 영감을 받아 다중 턴 데이터 수집과 훈련 전략을 제안합니다. 모델은 단일 턴 프롬프트(single-turn prompt)를 다중 턴 마코프 결정 프로세스(MDP)로 처리하여 초기 상태를 프롬프트로 간주하고, 여러 번의 시도 끝에 올바른 응답에 도달할 수 있도록 합니다. 모델은 처음에 잘못된 응답을 점진적으로 수정하여 최종적으로 정확한 응답을 만들어내도록 학습합니다.

- **Performance Highlights**: 실험 결과, RISE는 Llama2, Llama3, Mistral 모델이 수학적 추론 작업에서 반복적인 자기 개선을 통해 성능을 향상시켰음을 보여줍니다. RISE를 적용한 모델들은 동일한 양의 추론 시간(inference-time computation)을 사용한 단일 턴 전략들보다 우수한 성능을 보였습니다. 또한, 더 능력 있는 모델에서 더욱 큰 성능 향상을 보여주며 RISE의 강력한 확장성을 입증했습니다. 특히, 약한 모델에서 생성된 데이터를 사용하여 더 강력한 모델을 훈련시키는 경우에도 성능 향상이 가능함을 확인했습니다.



### Exploring Scaling Trends in LLM Robustness (https://arxiv.org/abs/2407.18213)
Comments:
          31 pages

- **What's New**: 이 논문은 대규모 언어 모델의 확장이 모델의 적대적 훈련(adversarial training)에 대한 반응을 개선하는지에 대해 연구합니다. 이전 연구에서는 모델 크기와 데이터 확장이 컴퓨터 비전(computer vision) 모델의 강건성을 개선한다는 것을 밝혔으나, 언어 모델에서도 유사한 현상이 나타나는지에 대한 것은 명확하지 않았습니다.

- **Technical Details**: 논문에서는 Pythia 모델(Pythia models)을 사용하여 다양한 크기(14M ~ 12B 파라미터)로 조정된 모델을 대상으로 두 가지 공격 기법(랜덤 토큰 기반 및 최신 greedy coordinate gradient 공격)에 대한 강건성을 평가하였습니다. 평가에 사용된 작업(task)은 이진 분류(binary classification)로, 스팸 이메일 필터링과 영화 리뷰 감정 분석을 포함합니다.

- **Performance Highlights**: 실험 결과, 적대적 훈련을 받지 않은 경우 큰 규모의 모델에서는 비약적인 강건성 향상을 기대하기 어렵다는 점이 확인되었습니다. 하지만 적대적 훈련을 받으면, 큰 모델일수록 적은 수의 예제만으로도 더 높은 강건성을 확보할 수 있으며, 충분한 수의 예제를 통해 더욱 높은 강건성에 도달할 수 있음을 보여주었습니다. 또한, 한 가지 공격에 대한 적대적 훈련이 유사한 다른 공격에 대한 방어력도 증가시키는 것으로 나타났습니다.



### I can listen but cannot read: An evaluation of two-tower multimodal systems for instrument recognition (https://arxiv.org/abs/2407.18058)
Comments:
          Accepted to ISMIR 2024

- **What's New**: 이 논문은 음악 두-타워 멀티모달(two-tower multimodal) 시스템에서 제로-샷(Zero-shot) 악기 인식을 평가하고 분석합니다. 오디오 및 텍스트 모달리티를 결합하여 새로운 방식의 분류 및 검색을 가능하게 하며, 특히 악기의 경우 본격적인 제로-샷 학습 특성을 살펴보았습니다.

- **Technical Details**: 두-타워 시스템은 사전 학습된 오디오 모델과 언어 모델(LM)을 각각 오디오 및 텍스트 인코더로 사용하여, 각각의 표현(embeddings)을 합동 오디오-텍스트 공간으로 매핑(mappings)하여 시스템을 최적화합니다. 본 연구에서는 MusCALL, CLAP 모델(음악 및 음성 데이터 기반과 음악 데이터 기반)을 포함해 총 3개의 시스템을 평가하였습니다.

- **Performance Highlights**: 음악 두-타워 시스템의 경우 제로-샷 학습에 유망한 결과를 보였지만, 텍스트 인코더 또는 합동 공간 투영에서 여전히 문제가 있습니다. 특히, 두-타워 시스템은 특정 단어에 민감하게 반응하며, 더 구체적인 음악 정보를 담은 프롬프트(prompt)보다 일반적인 프롬프트를 선호하는 경향이 있습니다. 텍스트 인코더의 대규모 크기와는 달리 추가적인 텍스트 맥락을 활용하거나 악기에 대한 정확한 설명을 추론하지 못하는 한계가 있습니다.



### RestoreAgent: Autonomous Image Restoration Agent via Multimodal Large Language Models (https://arxiv.org/abs/2407.18035)
- **What's New**: 모바일 기기로 촬영된 자연 이미지는 노이즈, 블러, 저조도 등 여러 가지 열화(degradation)로 인해 품질 저하가 발생할 수 있습니다. 이를 해결하기 위해 RestoreAgent라는 지능형 이미지 복원 시스템이 도입되었습니다. RestoreAgent는 멀티모달 대형 언어 모델(multi-modal large language model, MLLM)을 활용하여 입력 이미지의 열화 유형과 정도를 자동으로 평가하고, 적절한 복원 작업을 결정하고, 작업 순서를 최적화하며, 가장 적합한 모델을 선택하고, 복원을 실행합니다.

- **Technical Details**: RestoreAgent는 4단계 프로세스를 통해 이미지 복원을 수행합니다: 1) 열화 유형 식별(Degradation Type Identification), 2) 적응형 복원 순서 결정(Adaptive Restoration Sequence), 3) 최적 모델 선택(Optimal Model Selection), 4) 자동 실행(Automated Execution). 이 시스템은 다중 작업 데이터세트로 훈련된 MLLM을 활용하여, 광범위한 데이터에 대해 우수한 일반화 성능과 논리적 추론 능력을 발휘합니다. 훈련 데이터는 열화 이미지와 그라운드 트루스(평가용), 최적 작업 실행 순서, 사용자 선호에 따른 최적 모델 선택을 포함합니다.

- **Performance Highlights**: RestoreAgent는 실험 결과 복잡한 열화를 처리하는 데 있어 인간 전문가를 능가하는 탁월한 성능을 보여주었습니다. 또한, 모듈형 설계를 통해 새로운 작업과 모델의 빠른 통합을 촉진하여 다양한 응용 분야에 대한 유연성과 확장성을 강화시킵니다. 더욱이, 새로운 기능 추가에 단 몇 시간밖에 걸리지 않는다는 점에서 빠른 적응력을 자랑합니다.



### GermanPartiesQA: Benchmarking Commercial Large Language Models for Political Bias and Sycophancy (https://arxiv.org/abs/2407.18008)
Comments:
          12 pages

- **What's New**: LLMs의 도입이 콘텐츠 생성 및 소비 방식을 변화시키면서, 시민들의 정치적 의견과 투표 결정에 영향을 줄 수 있는 잠재력을 탐구하는 연구입니다. 이 논문은 OpenAI, Anthropic, Cohere에서 개발한 6개의 LLM을 독일 정당의 입장과 일치시켜보고, 프롬프트 실험을 통해 비율 세렌시 (sycophancy)의 정도를 평가했습니다.

- **Technical Details**: 연구진은 Voting Advice Application Wahl-o-Mat를 기반으로 한 GermanPartiesQA 벤치마크 dataset을 개발하여, 2021년부터 2023년까지의 10개의 주선거와 1개의 전국 선거를 다룹니다. LLM은 'I am [politician X],...' 및 'You are [politician X],...' 프롬프트를 사용하여 정치적 페르소나에 따른 응답 변화를 평가했습니다. 이 벤치마크 데이터는 주요 독일 정치인의 사회인구학적 데이터를 활용하여 제작되었습니다.

- **Performance Highlights**: 모든 LLM에서 좌-녹색 성향이 나타났으며, 'I am'과 'You are' 프롬프트 사이에 눈에 띄는 차이는 없었습니다. 이는 LLM의 응답이 아첨보다는 맥락에 대한 개인화로 설명될 수 있음을 시사합니다.



### Is the Digital Forensics and Incident Response Pipeline Ready for Text-Based Threats in LLM Era? (https://arxiv.org/abs/2407.17870)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 논문은 대규모 언어모델(LLMs)을 활용한 생성 AI(Generating AI)가 디지털 포렌직과 사건 대응(DFIR) 분야에 가져오는 새로운 사이버 보안 도전과제를 다룹니다. 특히, 사람과 NTG(Neural Text Generator)가 공동으로 작성한 텍스트를 탐지하고 기여도를 확인하는 CS-ACT 공격을 통해 전통적인 DFIR 기법의 취약점을 밝힙니다.

- **Technical Details**: 본 연구는 텍스트 기반 보안 시스템에 대한 DFIR 파이프라인을 평가하며, 특히 NTG-작성 텍스트의 탐지와 저자귀속(Attribution) 문제에 집중합니다. 14개의 다양한 데이터셋과 GPT-4를 포함한 43개의 NTG를 사용하여 상당한 포렌직 프로파일링의 취약성을 식별합니다. 이 연구는 대립 학습(adversarial learning), NTG 스타일화, 계층적 저자귀속 등 다양한 전략을 통해 보안을 강화할 수 있는 방안을 제안합니다.

- **Performance Highlights**: 연구 결과, NTG의 모델 정교화와 특정한 스타일의 부족 등이 주요 취약점으로 작용함을 밝혔습니다. 22개의 알고리즘을 이용해 이상적인 시나리오와 실제 공격 시나리오 모두에서 평가를 진행하였고, 현실적인 조건 아래서도 포렌직 프로파일링이 큰 도전을 받고 있음을 확인했습니다. NTG와 인간이 공동으로 작성한 텍스트의 경우, 전통적인 탐지 기법이 효과적이지 않다는 점에서 더 발전된 접근법이 필요함을 강조합니다.



### Financial Statement Analysis with Large Language Models (https://arxiv.org/abs/2407.17866)
Comments:
          Previously posted on SSRN (May 21, 2024). See this http URL

- **What's New**: 최근 연구에 따르면 GPT-4와 같은 대규모 언어 모델(LLM)이 금융 분석가와 유사하게 금융 재무제표를 분석할 수 있는지 조사합니다. 이 연구에서는 표준화되고 익명화된 재무제표를 LLM에 제공하고, 모델이 금융 성과를 예측하도록 지시합니다. 놀랍게도, LLM은 내러티브나 산업 특화 정보가 없더라도 금융 애널리스트보다 성과 예측 능력이 뛰어납니다.

- **Technical Details**: 연구는 GPT-4 Turbo 모델을 활용하여 표준화된 대차대조표와 손익계산서를 제공하고, 회사의 실적이 지속 가능한지 여부를 분석하도록 했습니다. 중요한 설계 방식으로는 재무제표에 동반되는 텍스트 정보를 제공하지 않은 점입니다. 모델의 예측 성능을 다양한 벤치마크(로지스틱 회귀 및 최첨단 인공 신경망(ANN) 설계)와 비교했습니다. 또한, 체인 오브 생각(Chain-of-Thought, CoT) 프롬프트를 사용하여 모델이 금융 분석가처럼 금융 비율을 계산하고 정보를 종합하도록 유도했습니다.

- **Performance Highlights**: LLM은 인간 분석가보다 어려운 상황에서 상대적 우위를 보였습니다. LLM의 예측 정확도는 최첨단 ML 모델과 동등했으며, 이는 모델의 학습 메모리가 아닌 실제로 유용한 내러티브 통찰력을 생성했음을 시사합니다. LLM 기반의 거래 전략은 Sharpe 비율과 알파가 더 높은 수익을 보여주었고, 이는 LLM이 금융 의사결정에 중요한 역할을 할 수 있음을 시사합니다.



### Shapley Value-based Contrastive Alignment for Multimodal Information Extraction (https://arxiv.org/abs/2407.17854)
Comments:
          Accepted at ACM Multimedia 2024

- **What's New**: 소셜 미디어와 멀티모달 (multimodal) 데이터 통신의 급속한 성장이 멀티모달 정보 추출 (MIE: Multimodal Information Extraction)의 발전을 요구하면서, 본 논문은 이미지, 텍스트, 그리고 새로운 텍스트 기반 컨텍스트(Context)를 연결하는 새로운 접근방식을 제안합니다. 이를 위해 Shapley Value 기반의 Contrastive Alignment(Shap-CA)를 도입하여 컨텍스트-텍스트 및 컨텍스트-이미지 페어를 정렬합니다.

- **Technical Details**: Shap-CA는 협력 게임 이론의 Shapley value 개념을 활용하여 각 요소(이미지, 텍스트, 컨텍스트)의 기여도를 평가하고, 대조 학습 전략을 통해 상호 작용을 강화합니다. 또한 선택적 크로스모달 융합을 위한 적응형 모듈을 설계하여 멀티모달 데이터를 효과적으로 결합합니다.

- **Performance Highlights**: 네 개의 MIE 데이터셋에서 실험을 통해 제안된 방법이 기존 최첨단 방법들을 현저히 능가하는 성능을 보였습니다. 이 방법은 이미지-텍스트 상호작용에서 발생하는 의미적 및 모달리티 갭을 효과적으로 해결하여 더 정확한 엔티티 및 관계 인식을 가능하게 합니다.



### Innovative Speech-Based Deep Learning Approaches for Parkinson's Disease Classification: A Systematic Review (https://arxiv.org/abs/2407.17844)
Comments:
          Submitted in Applied Sciences - peer reviewed Open Access journal. This research was funded by the NWO research programme AiNed Fellowship Grants under the project Responsible AI for Voice Diagnostics (RAIVD) - grant number NGF.1607.22.013

- **What's New**: 본 논문은 2020년부터 2024년 3월까지 발표된 33개의 과학 연구를 토대로, Parkinson's disease (파킨슨병)의 음성 데이터를 이용한 진단을 위한 최신의 인공지능(AI) 접근법을 검토합니다. 특히, 딥러닝(DL) 기반의 분류 방법에 대해서 집중적으로 다루고 있으며, End-to-End(E2E) 학습, 전이 학습(Transfer Learning, TL) 및 딥 음향 특성(Deep Acoustic Features, DAF) 추출로 분류하여 설명하고 있습니다.

- **Technical Details**: E2E 학습 접근법에서는 주로 Convolutional Neural Networks (CNNs)이 사용되지만, 최근 들어 Transformers도 인기를 끌고 있습니다. E2E 접근법은 제한된 데이터 및 컴퓨팅 자원 문제를 겪고 있지만, TL은 더 강력한 진단 및 언어 전반에 걸친 우수한 일반화 성능을 제공합니다. DAF 추출은 결과의 설명 가능성과 해석 가능성을 높이는 것을 목표로 하지만, 전통적인 기계 학습(ML) 방법과 다른 DL 접근법들에 비해 성능이 떨어지는 경향이 있습니다.

- **Performance Highlights**: 전반적으로, E2E와 TL 접근법이 더 나은 성능을 보이고 있으며, 특히 CNNs 및 Transformers와 같은 DL 모델이 주를 이루고 있습니다. 하지만, 여전히 편향성, 설명 가능성 및 프라이버시와 관련된 해결되지 않은 문제들이 남아 있으며, 이는 앞으로의 연구에서 다뤄져야 할 부분입니다.



### Unified Lexical Representation for Interpretable Visual-Language Alignmen (https://arxiv.org/abs/2407.17827)
- **What's New**: 이번 논문에서는 CLIP의 비단순화된 잠재특징 정렬의 해석가능성 문제를 해결하며, LexVLA라는 새로운 프레임워크를 소개합니다. 이는 시각 데이터와 언어 데이터를 통합된 어휘 표현으로 학습하여, 직관적이고 적용하기 쉽게 구현되었습니다. 특히, DINOv2와 Llama 2를 사용하여 어휘 예측 능력을 최적화합니다.

- **Technical Details**: LexVLA는 시각 모델로 DINOv2를, 텍스트 모델로 Llama 2를 사용합니다. DINOv2는 로컬 특성을 잘 학습하는 반면, Llama 2는 문맥 내 어휘 예측 능력을 갖추고 있습니다. 불필요한 어휘 활성화를 방지하기 위해 오버유즈 페널티(overuse penalty)도 도입되었습니다. 두 프리트레인된 단일 모달 모델을 적절한 훈련 데이터를 사용하여 정렬하였고, 복잡한 훈련 설정 없이 효과적인 학습을 실현했습니다.

- **Performance Highlights**: LexVLA는 CC-12M 다중 모달 데이터셋에서 훈련하여 더 큰 데이터셋(YF100m-CLIP + CC-12M, 1.1B data 포함)에 맞춰진 모델보다 우수한 성능을 보여주었습니다. 이 모델은 교차 모달 검색(Cross-Modal Retrieval) 벤치마크에서 높은 성능을 발휘하며, 이전 모델들보다 작은 데이터셋에서도 좋은 결과를 보였습니다.



### KiVA: Kid-inspired Visual Analogies for Testing Large Multimodal Models (https://arxiv.org/abs/2407.17773)
Comments:
          9 pages. For the KiVA benchmark, see this https URL

- **What's New**: 최근 연구에서는 대형 멀티모달 모델(Large Multimodal Models, LMMs)이 인간 성인과 어린이의 시각적 유추 추론(visual analogical reasoning) 능력과 비교되었습니다. 본 연구는 일상 물체의 1,400가지 시각적 변형을 포함하는 새로운 벤치마크를 제안하며, 이를 통해 LMMs가 시각적 유추 추론을 얼마나 잘 수행할 수 있는지를 평가합니다.

- **Technical Details**: 평가는 세 가지 단계로 나눠집니다: 1) 무엇이 변경되었는지 식별(예: 색상, 수량 등), 2) 어떻게 변경되었는지 이해(예: 객체 하나 추가됨), 3) 이 규칙을 새로운 시나리오에 적용. 이 벤치마크는 개발 심리학에 영감을 받아 어린이들이 초기 단계에서 경험하는 기본적인 시각적 유추 추론 능력을 중점으로 구성되었습니다.

- **Performance Highlights**: GPT-4V, LLaVA-1.5 및 MANTIS와 같은 모델들은 '무엇이' 변경되었는지를 효과적으로 식별하지만, '어떻게' 변경되었는지를 양적으로 이해하고 이를 새로운 객체에 적용하는 데 있어서는 어려움을 겪습니다. 특히, GPT-4V는 색상과 크기와 같은 간단한 시각적 속성에서는 더 나은 성능을 보이지만 숫자, 회전, 반사와 같은 복잡한 과제에서는 어려움을 겪습니다. 이 연구는 2D 이미지와 텍스트 위주의 데이터로 훈련된 모델의 한계를 강조합니다.



### ERIT Lightweight Multimodal Dataset for Elderly Emotion Recognition and Multimodal Fusion Evaluation (https://arxiv.org/abs/2407.17772)
- **What's New**: ERIT는 고령자의 다양한 상황에 대한 반응을 기록한 텍스트와 이미지 데이터를 포함하는 새로운 멀티모달(multimodal) 데이터셋으로, 감정 인식을 위한 중요한 연구 자원을 제공합니다. 특히, 기계 학습에서 상대적으로 잘 다뤄지지 않은 고령자 그룹의 감정 인식을 돕고자 합니다.

- **Technical Details**: ERIT 데이터셋은 ElderReact 비디오 데이터셋에서 프레임을 추출하여 생성되었습니다. 7가지 기본 감정(분노, 혐오, 공포, 기쁨, 슬픔, 놀람, 중립)에 대해 감정 레이블을 제공하며, DeepFace 프레임워크와 Whisper ASR(automatic speech recognition)를 사용해 텍스트와 표정을 분석했습니다. 경량 처리와 멀티모달 연구를 위해 텍스트와 이미지 데이터를 선택했습니다.

- **Performance Highlights**: ERIT 데이터셋을 사용하여 다양한 모델의 멀티모달 융합 성능을 평가했습니다. 특히, LMMs(large multimodal models)인 GPT-4와 LLaMA-7B가 텍스트 기반 감정 인식에서 경쟁력 있는 성능을 보였습니다. 그러나 ERIT 데이터셋의 복잡한 감정 표현과 연령대에 따른 분석에서는 멀티모달 입력의 통합이 필요합니다.



### Cost-effective Instruction Learning for Pathology Vision and Language Analysis (https://arxiv.org/abs/2407.17734)
- **What's New**: CLOVER라는 새로운 비용 효율적인 학습 프레임워크가 소개되었습니다. 이 프레임워크는 경량 모듈을 훈련하고, 대규모 언어 모델(Large Language Model, LLM)의 파라미터를 고정시킨 상태에서 학습을 진행합니다. CLOVER는 GPT-4 대신 GPT-3.5로 훈련 데이터를 생성하여 비용을 절감했습니다.

- **Technical Details**: CLOVER의 아키텍처는 BLIP-2를 기반으로 하며, 시각 언어 사전 학습에서는 경량 변환기(Trainable Query Transformer, Q-former)를 사용하여 시각 인코더와 LLM을 고정시킵니다. 원형 템플릿 질문과 원본 텍스트 설명을 일치시켜 '하이브리드형 명령어(Instructions)'를 구축했습니다. 이러한 하이브리드형 명령어는 모델의 튜닝을 강화시킵니다.

- **Performance Highlights**: CLOVER는 두 개의 벤치마크 데이터셋과 하나의 독립적인 임상 데이터셋에서 실험 결과를 통해 그 유효성이 입증되었습니다. 주요 경쟁 방법과의 비교에서는 최대 26.13%의 정확도 향상을 보였으며, 37배 더 많은 파라미터를 가진 LLaVA-Med에 비해 비슷한 성능을 보였습니다. 이는 매우 제한된 리소스로도 효과적인 학습이 가능함을 의미합니다.



### Describe Where You Are: Improving Noise-Robustness for Speech Emotion Recognition with Text Description of the Environmen (https://arxiv.org/abs/2407.17716)
- **What's New**: 이번 논문에서는 실제 환경에서의 음성 감정 인식(SER) 성능을 강화하기 위해 새로운 접근 방식을 제안합니다. 제안된 방식은 텍스트 기반 환경 기술을 활용하여 노이즈가 포함된 음성 데이터에 대한 환경 인식 학습을 통해 SER 모델의 성능을 극대화합니다. MSP-Podcast 코퍼스와 Freesound 저장소에서 수집한 실제 노이즈 샘플을 사용해 실험을 수행한 결과, 텍스트 기반 환경 기술이 SER 시스템의 노이즈 내성을 향상시킨다는 것을 확인했습니다.

- **Technical Details**: 논문에서는 사전 학습된 텍스트 인코더를 사용하여 텍스트 기반 환경 임베딩을 추출하고, 이를 트랜스포머 기반 SER 모델에 결합합니다. 'Text-Guided Environment-Aware Training (TG-EAT)' 전략을 통해 노이즈가 포함된 음성과 해당 환경 기술을 사용하여 SER 모델을 학습시킵니다. TG-EAT는 진폭(arousal), 지배(dominance), 평가(valence)를 예측하는 데 중점을 둡니다. 또, Large Language Model(LLM)을 활용하여 더 나은 표현을 생성하고 노이즈 발생 조건에서도 높은 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, -5dB 신호대잡음비(SNR) 조건에서 제안된 방법이 기존 모델 대비 arousal에서 31.8%, dominance에서 23.5%, valence에서 9.5% 성능 향상을 보였습니다. 또한, CLAP 모델의 텍스트 인코더를 미세 조정하여 -5dB SNR 조건에서 arousal 72.2%, dominance 91.6%, valence 21.0%의 성능 향상을 달성하였습니다. 제안된 접근 방식은 여러 다양한 환경에서의 SER 성능을 크게 개선하여 실제 환경에서의 응용 가능성을 높였습니다.



### Enhancing Agent Learning through World Dynamics Modeling (https://arxiv.org/abs/2407.17695)
- **What's New**: 이 논문에서는 기존의 대형 언어 모델(LLMs)이 환경에 대한 포괄적이고 깊이 있는 지식을 이미 갖추고 있다는 가정의 문제를 지적하고, 이를 해결하기 위해 Discover, Verify, and Evolve (DiVE) 프레임워크를 제안합니다. DiVE는 소수의 데모를 통해 전 세계적 동태를 발견하고, 이러한 동태를 검증하며, 현재 상황에 맞게 새로운 고급 동태를 발전시키는 과정을 거칩니다.

- **Technical Details**: DiVE는 세 가지 주요 구성 요소로 나누어집니다. 첫째, Discoverer는 제공된 데모를 기반으로 환경의 동태를 반복적으로 발견하는 역할을 합니다. 둘째, Verifier는 LLMs의 잘못된 정보 생성 경향을 걸러내어 정확하고 신뢰할 수 있는 정보만을 남깁니다. 셋째, Evolver는 전략적 플레이나 맥락에 맞는 결정을 내리는 등의 상태 적합 동태를 심층적으로 추론합니다.

- **Performance Highlights**: Crafter 환경에서 기능 평가를 통해 DiVE는 다양한 동태를 학습하고 인간의 플레이어와 유사한 수준의 보상을 달성했습니다. 이는 DiVE가 제공하는 심층 지식이 LLM의 의사 결정 능력을 현저히 향상시킨다는 것을 입증합니다.



### Transformers on Markov Data: Constant Depth Suffices (https://arxiv.org/abs/2407.17686)
Comments:
          29 pages, 10 figures

- **What's New**: 이 논문에서는 다양한 도메인과 방식에서 생성 과정을 모델링하는 데 성공한 Attention 기반 Transform을 \\kth 마르코프 과정(Markov Processes)에서의 동작을 분석합니다. 이전 연구 결과와는 달리, 특정 조건 하에 Transform이 마르코프 소스에서 낮은 테스트 손실을 달성할 수 있는 현상을 관찰했습니다.

- **Technical Details**: Transform에서 한 층당 하나의 헤드(head)를 가지고 고정된 깊이로 구성된 형태로 충분히 오랜 기간 동안 훈련되었을 때도 \\kth 마르코프 소스에서 유효함을 보였습니다. 이들은 in-context conditional empirical distribution을 잘 표현하고 학습할 수 있었으며 이를 이론적으로도 단일 헤드와 세 층으로 구성된 Transform이 이를 표현할 수 있음을 입증했습니다.

- **Performance Highlights**: 실험 결과, 고정된 깊이를 가진 Transform이 \\kth 마르코프 소스에서 기대 이상의 낮은 테스트 손실을 달성했으며, $O(\log_2(k))$ layers를 가진 attention-only Transform이 in-context conditional empirical distribution을 잘 표현할 수 있음을 보여줬습니다.



### Traditional Methods Outperform Generative LLMs at Forecasting Credit Ratings (https://arxiv.org/abs/2407.17624)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 사용하여 기업 신용 등급을 예측하는 작업의 성능을 조사합니다. 저자들은 LLMs가 텍스트 정보 인코딩에는 뛰어나지만 숫자 및 다중 모드(multimodal) 데이터를 인코딩하는 데 있어서는 전통적인 방법에 비해 경쟁력이 떨어진다고 주장합니다. 이는 LLMs가 텍스트 기반 예측 작업에서 강력한 도구로 떠오르지만, 모든 데이터 유형에서 우수한 성능을 보이는 것은 아님을 시사합니다.

- **Technical Details**: 이 연구는 금융 문맥에서의 LLMs의 능력을 시험하기 위해, 주로 텍스트 기반 예측을 다루고 있습니다. 전통적인 XGBoost 아키텍처와 비교하여 LLMs는 고밀도 텍스트 기반 임베딩 기능과 함께 기초 경제 데이터와 거시 경제 데이터를 결합한 방식에서 더 나은 성능을 보여주지 못했습니다. 이 논문은 현대 언어 모델링 기법을 신용 등급 예측 작업에 처음으로 적용한 연구로, 학술 WRDS 라이센스를 통해 재현 가능한 금융 데이터셋을 사용합니다.

- **Performance Highlights**: 본 연구 결과는 LLMs가 텍스트 정보를 매우 잘 인코딩하지만 숫자 정보를 포함하는 부분에서는 전통적인 부스팅 트리 기반 방식보다 성능이 떨어진다는 것을 보여줍니다. 이는 신용 등급 예측 작업과 같은 복잡한 금융 예측 문맥에서 LLMs의 현재 한계를 드러냅니다. 또한, 긴 텍스트 시퀀스를 처리하기 위한 효과적인 모델링 방법을 제안하며, 이는 향후 연구에 중요한 시사점을 제공합니다.



### Big5PersonalityEssays: Introducing a Novel Synthetic Generated Dataset Consisting of Short State-of-Consciousness Essays Annotated Based on the Five Factor Model of Personality (https://arxiv.org/abs/2407.17586)
Comments:
          13 pages, 1 table, 2 figures

- **What's New**: 대규모 언어 모델(LLM)의 급격한 발전에 따라 이 모델들의 행동을 연구하고 이를 다양한 과학 분야에 적용하는 것이 매우 중요해졌습니다. 하지만 심리학 분야는 최근 몇 년간 새로운 컴퓨팅 도구를 충분히 활용하지 못했습니다. 그 이유 중 하나는 올바른 분석을 위해 필요한 데이터의 복잡성이 높기 때문입니다. 또한 심리학, 특히 심리측정학(psychometry)은 분석 및 인공지능 활용에 필요한 데이터셋이 거의 없습니다. 이러한 문제를 해결하기 위해 이번 연구에서는 성격특질의 5요인 모델(Five Factor Model, FFM)을 기준으로 레이블이 지정된 짧은 에세이의 합성 데이터베이스를 소개합니다.

- **Technical Details**: 연구에서는 FFM을 기반으로 데이터셋을 생성하여 심리학적 분석에 활용할 수 있게 했습니다. 이 데이터베이스는 대규모 언어 모델을 통해 생성된 단문 에세이들로 구성되어 있으며, 성격 특질 중 5가지 요소(외향성, 신경증, 개방성, 친화성 및 성실성)로 라벨링(Labeling)이 되어 있습니다.

- **Performance Highlights**: 이번 연구의 주요 성과는 심리학 세부 분야에 활용할 수 있는 새로운 합성 데이터베이스를 제공함으로써, 심리측정학 분야에서도 대규모 언어 모델을 효과적으로 사용할 수 있는 가능성을 열었습니다. 이 데이터베이스를 통해 보다 복잡하고 종합적인 심리학 분석이 가능해질 것으로 기대됩니다.



### Exploring Domain Robust Lightweight Reward Models based on Router Mechanism (https://arxiv.org/abs/2407.17546)
Comments:
          This paper is accepted for ACL 2024

- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전은 인간의 피드백에 의한 강화 학습(이하 RLHF)의 보상 모델에 크게 의존하고 있습니다. 그러나 다양한 도메인에서 단일 보상 모델을 사용하는 것은 항상 최적의 결과를 보장하지 못하며, 새로운 도메인 데이터가 도입될 때마다 처음부터 다시 훈련해야 하는 문제가 있습니다. 이러한 문제를 해결하기 위해, 우리는 라우터 메커니즘을 기반으로 도메인별로 작동하는 소형 언어 모델을 활용하는 방법을 탐구했습니다.

- **Technical Details**: 세 가지 접근 방법을 제안했습니다. 첫째, 내부 라우터와 전문가들(Experts)을 모듈화하여 단일 보상 모델을 구성하는 **Mixture of Reward Experts (MoRE)**를 사용합니다. 둘째, 외부 라우터를 통해 여러 도메인별 모델 중에서 적절한 보상 모델을 선택하는 **Router for DOmain-specific reward modelS (RODOS)**를 도입했습니다. 셋째, **Adapter Router Lightweight Integrated rewardS Switching (ARLISS)** 프레임워크를 적용해 단일 소형 언어 모델에 보상 모델과 라우터 어댑터를 적재하여 파라미터 크기를 줄였습니다.

- **Performance Highlights**: 실험 결과, 다섯 가지 도메인 데이터셋에서 제안된 방법들이 기본 방법과 비교해 뛰어난 성능을 보여주었습니다. 특히, RODOS는 최고의 성능을 기록했으며, MoRE는 약 52%, ARLISS는 약 55%의 파라미터 크기 감소를 달성했습니다.



### Large Language Models for Anomaly Detection in Computational Workflows: from Supervised Fine-Tuning to In-Context Learning (https://arxiv.org/abs/2407.17545)
Comments:
          12 pages, 14 figures, paper is accepted by SC'24, source code, see: this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 컴퓨팅 워크플로우(Computational Workflows)에서 이상 탐지를 시도한 연구입니다. 새로운 이상(anomaly)을 탐지하는 데 어려움을 겪는 기존의 규칙 기반 방법을 보완하기 위해 LLMs의 복잡한 데이터 패턴 학습 능력을 이용합니다. 이 논문에서는 두 가지 접근 방식을 조사합니다: 1) 지도 학습 기반 미세 조정(Supervised Fine-Tuning, SFT)이며, 사전 훈련된 LLM을 레이블된 데이터로 미세 조정하여 문장 분류를 통해 이상을 식별하는 방법과 2) 맥락 내 학습(In-Context Learning, ICL)으로, 몇 가지 예제와 작업 설명을 포함하는 프롬프트를 사용하여 LLM이 소수의 예제만으로 이상을 탐지하는 방법입니다.

- **Technical Details**: SFT는 사전 훈련된 모델을 소규모의 레이블이 지정된 데이터셋으로 재학습시켜 특정 작업의 성능을 향상시키는 방법입니다. SFT의 문제점으로는 데이터의 편향 생성, 그리고 한 작업을 학습한 후 새로운 작업을 학습할 때 기존 작업을 잊어버리는 'Catastrophic Forgetting' 현상이 있습니다. 반면, ICL은 레이블이 지정된 데이터로 미세 조정을 필요로 하지 않고, 프롬프트 엔지니어링을 통해 쿼리의 맥락 내에서 몇 가지 예제를 제공하여 작업을 수행하는 emerging 패러다임입니다. 이 연구는 또한 프롬프트에 'Chain-of-Thought' 접근을 사용해 LLM의 해석 가능성을 향상시킵니다.

- **Performance Highlights**: 다양한 워크플로우 데이터셋을 대상으로 한 실험에서, LLMs가 복잡한 실행 환경에서 효과적인 이상 탐지를 위한 유망한 잠재력을 보여주었습니다. 제로샷(zero-shot) 및 소수의 예제만을 사용한 소수샷(few-shot) 학습에서의 ICL 프롬프트와 해석 가능성을 위한 'Chain-of-Thought Prompting'이 우수한 성능을 입증했습니다.



### Mapping the Technological Future: A Topic, Sentiment, and Emotion Analysis in Social Media Discours (https://arxiv.org/abs/2407.17522)
- **What's New**: 이번 연구는 2021년부터 2023년까지의 150만 개의 트윗을 분석하여, 기술 중심의 미래에 대한 기대와 감정을 파악했습니다. 특히, 인공지능(AI)과 같은 기술의 변동성과 예측 불가능성에 대한 무게가 긍정적인 기대(emotions)와 부정적인 정서를 얼마나 압도하는지를 확인했습니다. 또한, 주요 의견 리더들(KOLs)의 감정적 영향을 분석하여, 이들이 긍정적인 기대와 낙관주의를 어떻게 강조하는지를 밝혔습니다.

- **Technical Details**: 연구에서는 BERTopic 모델링을 사용하여 트위터 데이터의 주제를 분석하고, 감정 및 정서 분석을 통해 트위터 상의 감정 변화를 추적했습니다. 데이터는 snsscrape 라이브러리를 통해 약 400명의 기술 인플루언서의 트윗을 스크래핑하여 수집했습니다. 분석 결과, '희망' 점수가 '불안' 점수보다 약 10.33% 높게 나타났습니다.

- **Performance Highlights**: 연구 결과, 긍정적인 감정이 부정적인 감정보다 압도적으로 우세함을 확인하였으며, 주요 의견 리더들(KOLs)은 '낙관주의'와 기술의 이점을 '비관주의'와 도전 과제보다 더 강조하는 경향이 있었습니다. 이는 기술의 불확실성 속에서도 긍정적인 미래상을 제시하는 역할을 하는 KOLs의 중요성을 나타냅니다.



### Survey on biomarkers in human vocalizations (https://arxiv.org/abs/2407.17505)
- **What's New**: 최근 몇 년간 음성(voice)을 활용하여 화자의 건강 상태를 감지하는 기술이 증가하고 있습니다. 이 설문 논문은 이러한 기술들에 대한 일반적인 분류와 현재의 진보 및 도전을 폭넓게 개관합니다. 음성 바이오마커(vocal biomarkers)는 종종 다른 센서의 신호를 근사하거나 정신적, 인지적, 생리적 상태를 식별하는 이차적 측정 수단으로 사용됩니다.

- **Technical Details**: 음성 바이오마커의 측정은 여러 혼란(disturbance)과 불확실성(uncertainty)을 포함하며, 이는 노이즈원의 형태로 간주될 수 있습니다. 이 바이오마커는 이러한 노이즈원의 다양한 출처를 기준으로 조잡하게 평가됩니다. 제안된 일부 바이오마커에서 오차 수준이 높아 보이지만, 오차가 낮아 건강관리 응용 프로그램에서 채택될 가능성이 높은 음성 바이오마커도 존재합니다.

- **Performance Highlights**: 많은 음성 바이오마커가 높은 오차를 보이지만, 일부는 더 낮은 오차로 성능이 뛰어나며, 이는 건강관리 응용 프로그램에서 더 신뢰할 수 있는 후보로 채택될 가능성을 지니고 있습니다. 이러한 저오차 음성 바이오마커는 특히 정신적, 인지적, 생리적 상태 모니터링에 유용하게 사용될 수 있습니다.



### Challenges and Considerations in Annotating Legal Data: A Comprehensive Overview (https://arxiv.org/abs/2407.17503)
- **What's New**: 법적 문서 주석화의 복잡성을 다룬 새로운 연구가 발표되었습니다. 이 논문은 법률 텍스트의 고유한 표현 구조와 용어의 복잡성을 이해하고 처리하기 위한 기초적인 지침과 방법론을 제시합니다. 이를 통해 연구자와 전문가들이 법률 데이터 주석화 프로젝트를 효과적으로 수행할 수 있도록 돕고자 합니다.

- **Technical Details**: 법적 문서의 주석화 과정은 적절한 데이터 세트를 선택하는 것에서 시작합니다. 이 과정은 법률 텍스트의 복잡성을 잘 포착할 수 있는 데이터를 선택하는 것이 핵심입니다. 텍스트 추출은 법적 문서가 종종 포함하고 있는 각주, 참조, 독특한 용어 등의 복잡성으로 인해 어렵습니다. 데이터 정제(data cleaning)는 불필요한 정보를 제거하면서도 중요한 법률 세부 사항과 맥락을 유지하는 것을 목표로 합니다. 주석 가이드라인(annotation guidelines)은 법률 용어의 미세한 뉘앙스를 고려해 일관성을 유지하는 데 필수적입니다. 주석 작업에는 법률 전문가가 직접 참여하여 법적 기준과 해석을 정확히 반영하도록 해야 합니다.

- **Challenges of Legal Annotations**: 독일 법률 텍스트 주석화 과정에서 발생하는 주요 과제는 법적 용어의 미묘한 뉘앙스와 문서 구조로 인해 발생하는 복잡성입니다. 특히, 법적 사례 또는 특정 명명 엔티티(named entities)만을 포함하는 기존 데이터 세트의 한계를 지적하며, 모든 법적 참고 자료와 해당 단락의 전체 텍스트를 제공하는 데이터 세트의 필요성을 강조합니다. 데이터 정제 과정에서 HTML 형식의 비일관성을 해결하기 위해 특정 가정을 통해 데이터를 처리했습니다. 규칙 기반(regex) 접근 방식을 사용했지만, 다양한 법적 규정 표현 방식으로 인해 추가적인 어려움이 존재했습니다.

- **Performance Highlights**: 논문에서 소개된 데이터 세트는 약 1.1 GB의 크기로, 43337개의 행과 12개의 속성으로 구성되어 있으며, Zenodo를 통해 공개되었습니다. 이 데이터 세트는 법적 텍스트와 해당 법률 간의 의미적 유사성을 계산하는 연구에 활용되었습니다. 법률 전문을 포함한 참조 법률 정의를 정확히 추출하는 것은 여전히 해결해야 할 과제입니다.



### Explainable Natural Language Processing for Corporate Sustainability Analysis (https://arxiv.org/abs/2407.17487)
- **What's New**: 이 논문에서는 기업지속가능성 분석 방식을 크게 향상시킬 수 있는 Explainable Natural Language Processing (XNLP) 방법을 제안합니다. XNLP는 사람의 분석을 보조해 데이터를 신뢰성 있게 이해하고 해석할 수 있는 능력을 갖추며, 현재 존재하는 주관성 문제를 완화할 수 있습니다.

- **Technical Details**: 기업지속가능성은 회사의 운영 복잡성(지리적 위치, 규모, 비즈니스 활동 등)과 데이터의 본래 특성(불완전성, 모호성, 신뢰성 부족 등)으로 인해 평가하기 어렵습니다. 이 논문은 기업이 자발적으로 제공하는 지속가능성 보고서가 주된 데이터 소스임을 강조하면서, 데이터와 분석가 차원의 주관성이 평가 과정을 복잡하게 만든다고 지적합니다. 이러한 문제를 해결하기 위해 XNLP가 제안되며, 이는 언어적 이해 알고리즘(lexical, semantic, syntactic)과 해석 가능 및 설명 가능 인공지능(XAI) 기능을 통합하여 주관성 문제를 개선할 수 있습니다.

- **Performance Highlights**: 이 논문은 XNLP가 분석가의 주관성과 데이터의 주관성 문제를 동시에 해결할 수 있다고 주장합니다. XNLP는 다양한 기업지속가능성 자료를 빠르고 일관되게 처리할 수 있고, 이를 통해 분석가는 더 많은 인사이트를 얻어 효율적으로 지속가능성 평가를 수행할 수 있습니다.



### Reinforcement Learning from Human Feedback: Whose Culture, Whose Values, Whose Perspectives? (https://arxiv.org/abs/2407.17482)
- **What's New**: 이 논문은 인간의 피드백을 통해 강화 학습(Reinforcement Learning from Human Feedback, RLHF)에 있어서 다원주의의 인식론적(에피스템적) 및 윤리적 이점을 주장하고 있습니다. 특히, 대형 언어 모델(LLM)의 문맥에서 이러한 장점을 논의하고 있습니다.

- **Technical Details**: 논문은 사회 인식론(social epistemology)과 과학의 다원주의 철학(pluralist philosophy of science)을 바탕으로, RLHF를 인간의 필요에 더 잘 반응할 수 있도록 만드는 방법과 그 과정에서 직면하게 될 도전에 대해 제안하고 있습니다. 이를 통해 다각적인 접근법이 어떻게 LLM 개발을 향상시킬 수 있는지 고찰합니다.

- **Performance Highlights**: 논문은 LLM 개발을 개선하기 위해 구체적이고 실현 가능한 단계들로 변화의 의제를 결론부에 제시하고 있습니다. 이는 명확한 행동 지침을 통해, 다양하고 다양한 의견을 수용할 수 있는 더 포용적인 AI 시스템을 목표로 합니다.



### LLM4PM: A case study on using Large Language Models for Process Modeling in Enterprise Organizations (https://arxiv.org/abs/2407.17478)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 연구에서는 대형 언어 모델 (LLM)을 활용하여 조직 내 프로세스 모델 생성 지원 여부를 탐구하였습니다. Hilti Group라는 다국적 기업을 대상으로 LLM 기반 챗봇 PRODIGY (PROcess moDellIng Guidance for You)를 개발하고 테스트하였습니다. 이는 LLM이 인간 모델러들이 프로세스 플로우 다이어그램을 만들 때 어떻게 도울 수 있는지 이해하기 위한 첫 사례 연구입니다.

- **Technical Details**: 연구의 첫 단계로, Hilti의 전문 프로세스 모델러 10명을 대상으로 사전 사용자 조사를 실시하여 그들이 일상에서 겪는 여러 가지 고충점을 물었습니다. 이 반응을 바탕으로 PRODIGY를 설계 및 구현하였으며, 최종적으로 사용자 조사 참가자들로 하여금 PRODIGY를 사용해 본 후 장단점을 평가하도록 요청했습니다. 이렇게 모은 결과를 바탕으로 실용적 인사이트를 도출하였습니다.

- **Performance Highlights**: 이번 연구는 대형 언어 모델이 프로세스 모델링에 실제로 활용된 첫 사례임을 강조하며, LLM이 비즈니스 프로세스 관리 활동을 개선하는 데 어떻게 기여할 수 있는지에 대해 산업계에 새로운 통찰을 제시합니다.



### Toward Automated Detection of Biased Social Signals from the Content of Clinical Conversations (https://arxiv.org/abs/2407.17477)
Comments:
          Accepted by AMIA 2024 Annual Symposium

- **What's New**: 이번 연구는 자동 음성 인식(ASR)과 자연어 처리(NLP)를 사용하여 환자-의료 제공자 상호작용에서 사회적 신호(social signals)를 식별하는 방법을 탐구했습니다. 이 연구는 782차례 1차 진료 방문의 오디오 녹음을 분석하여, 백인 환자와 비백인 환자에 대한 의료 제공자의 의사 소통 행동의 통계적으로 유의미한 차이를 발견했습니다.

- **Technical Details**: 연구팀은 자동화된 파이프라인을 구축하여 오디오 녹음에서 사회적 신호를 예측했습니다. 이 파이프라인은 평균 90.1%의 정확도를 달성했으며, 백인 환자와 비백인 환자에 대해 공정성을 보였습니다. 이러한 파이프라인을 통해, 의료 제공자는 백인 환자에게 더 많은 환자 중심 행동을 표현함을 확인했습니다. 이는 더 높은 수준의 따뜻함(warmth), 참여도(engagement), 그리고 주의력(attentiveness)을 포함합니다.

- **Performance Highlights**: 연구 결과, 구성된 파이프라인은 다양한 사회적 신호의 예측에서 뛰어난 성능을 보여주었습니다. 평균 90.1%의 예측 정확도는 임상 환경에서의 적용 가능성을 높이고, 백인 환자와 비백인 환자 모두에게 일관된 성능을 제공할 수 있는 잠재력을 제시합니다. 이는 의료 품질과 형평성에 영향을 미칠 가능성이 있는 미묘한 의사소통 신호를 식별하는 데 중요한 도구가 될 수 있습니다.



### OpenDevin: An Open Platform for AI Software Developers as Generalist Agents (https://arxiv.org/abs/2407.16741)
Comments:
          Code: this https URL

- **What's New**: OpenDevin은 인간 개발자처럼 코드 작성, 명령어 실행, 웹 브라우징 등을 통해 세상과 상호작용하는 강력하고 유연한 AI 에이전트를 개발할 수 있는 플랫폼을 소개합니다. 이 플랫폼은 새로운 에이전트 구현, 코드 실행을 위한 안전한 샌드박스 환경, 다중 에이전트 간의 조정, 평가 벤치마크 통합 등을 제공하며, MIT 라이선스 하에 공개된 커뮤니티 프로젝트입니다.

- **Technical Details**: OpenDevin은 LLMs(대형 언어 모델)의 발전을 기반으로 구축되었으며, 에이전트의 구현을 용이하게 하고, 안전한 샌드박스 환경에서 코드를 실행할 수 있게 설계되었습니다. 또한, 여러 에이전트 간의 조정(Coordinate)과 성능 평가(evaluation)를 위한 벤치마크를 통합하는 기능을 제공합니다.

- **Performance Highlights**: OpenDevin을 사용한 에이전트들은 소프트웨어 엔지니어링(SWE-Bench) 및 웹 브라우징(WebArena) 등 15개의 도전적인 작업에서 평가를 받았습니다. 현재까지 이 플랫폼은 학계와 산업계의 160명 이상의 기여자로부터 1.3K 이상의 기여를 받았습니다.



New uploads on arXiv(cs.IR)

### Sample Enrichment via Temporary Operations on Subsequences for Sequential Recommendation (https://arxiv.org/abs/2407.17802)
Comments:
          12 pages, 6 figures

- **What's New**: 최근의 논문에서는 연속 추천 시스템(sequential recommendation)에서 사용자의 고유한 선호를 예측하기 위한 혁신적인 프레임워크인 'SETO(sample enrichment via temporary operations on subsequences)'를 제안합니다. 이 프레임워크는 모델의 구조나 추가 정보를 부가하지 않고, 기존 데이터와 사용자 선호 사이의 변환 공간을 임시적으로 강화하는 방식을 채택하여 데이터 희소성 문제를 해결합니다.

- **Technical Details**: SETO는 주어진 데이터와 사용자 선호 간의 변환 과정을 다양성 있는 시퀀스 증강(sequence enhancement) 작업으로 채워나가는 모델 불가지론적(generic model-agnostic) 프레임워크입니다. 논문의 저자들은 두 가지 주요 작업, 즉 Swap과 Removal을 이용하여 시퀀스를 강화합니다. 이 작업들은 확률적이며 무작위적인 선택을 통해 이루어지며, 각각의 반복 훈련(iterative training) 과정에서 입력 및 목표 서브시퀀스(subsequences)에 일시적으로 적용됩니다.

- **Performance Highlights**: SETO는 6개의 단일 도메인 시계열 모델과 2개의 크로스 도메인 시계열 모델을 포함한 여러 첨단 시계열 추천 모델들에 대해 높은 효과성과 범용성을 입증했습니다. 이 연구는 3개의 단일 도메인 데이터셋, 3개의 크로스 도메인 데이터셋, 그리고 대규모 산업 데이터셋을 사용한 광범위한 실험을 통해 수행되었습니다. 이러한 실험 결과는 SETO가 기존 방법들에 비해 추천 정확도를 크게 향상시킬 수 있음을 보여줍니다.



### Text-Driven Neural Collaborative Filtering Model for Paper Source Tracing (https://arxiv.org/abs/2407.17722)
Comments:
          KDD CUP 2024 OAG-Challenges, Paper Source Tracing, Technical Report of Team AoboSama @ KDD CUP 2024. August 25--29, 2024. Barcelona, Spain

- **What's New**: 이 논문에서는 Paper Source Tracing (PST) 작업을 자동화하기 위한 새로운 추천 시스템 기반 프레임워크를 제안합니다. 이 프레임워크는 Neural Collaborative Filtering (NCF) 모델과 사전 학습된 언어 모델인 SciBERT를 이용하여 학술 논문의 중요한 참고 문헌을 식별합니다. 이 연구는 KDD CUP 2024에서 0.37814의 Mean Average Precision (MAP) 점수를 기록하며 11위를 차지했습니다.

- **Technical Details**: PST 작업은 논문 간의 인용, 저자, 키워드 등 다양한 관계를 통해 연결된 복잡한 관계 그래프를 분석하여 중요한 참고 문헌을 식별하는 과제입니다. 제안된 프레임워크는 Neural Collaborative Filtering (NCF) 모델을 사용하여 논문과 참고 문헌 간의 유사성을 예측합니다. 텍스트 속성을 처리하기 위해 SciBERT 모델을 사용하여 논문 본문에서 추출된 특징을 모델에 입력으로 제공합니다. 데이터셋은 DBLP-citation-network에서 가져왔으며, 검색된 텍스트 속성을 SciBERT를 통해 처리합니다.

- **Performance Highlights**: 제안된 NCF-SciBERT 모델은 다양한 비교 실험에서 높은 성능을 보여주었습니다. PST 데이터를 사용하여 학습된 모델은 11위에 올랐으며, 0.37814의 MAP 점수를 기록하여 베이스라인 모델들을 크게 능가했습니다. 또한, 이 모델은 NVIDIA A100-PCIE-40GB GPU에서 훈련되었으며, 약 23억 개의 매개변수를 가지며 36GB의 메모리를 사용하여 효율적으로 학습되었습니다.



### Challenges and Considerations in Annotating Legal Data: A Comprehensive Overview (https://arxiv.org/abs/2407.17503)
- **What's New**: 법적 문서 주석화의 복잡성을 다룬 새로운 연구가 발표되었습니다. 이 논문은 법률 텍스트의 고유한 표현 구조와 용어의 복잡성을 이해하고 처리하기 위한 기초적인 지침과 방법론을 제시합니다. 이를 통해 연구자와 전문가들이 법률 데이터 주석화 프로젝트를 효과적으로 수행할 수 있도록 돕고자 합니다.

- **Technical Details**: 법적 문서의 주석화 과정은 적절한 데이터 세트를 선택하는 것에서 시작합니다. 이 과정은 법률 텍스트의 복잡성을 잘 포착할 수 있는 데이터를 선택하는 것이 핵심입니다. 텍스트 추출은 법적 문서가 종종 포함하고 있는 각주, 참조, 독특한 용어 등의 복잡성으로 인해 어렵습니다. 데이터 정제(data cleaning)는 불필요한 정보를 제거하면서도 중요한 법률 세부 사항과 맥락을 유지하는 것을 목표로 합니다. 주석 가이드라인(annotation guidelines)은 법률 용어의 미세한 뉘앙스를 고려해 일관성을 유지하는 데 필수적입니다. 주석 작업에는 법률 전문가가 직접 참여하여 법적 기준과 해석을 정확히 반영하도록 해야 합니다.

- **Challenges of Legal Annotations**: 독일 법률 텍스트 주석화 과정에서 발생하는 주요 과제는 법적 용어의 미묘한 뉘앙스와 문서 구조로 인해 발생하는 복잡성입니다. 특히, 법적 사례 또는 특정 명명 엔티티(named entities)만을 포함하는 기존 데이터 세트의 한계를 지적하며, 모든 법적 참고 자료와 해당 단락의 전체 텍스트를 제공하는 데이터 세트의 필요성을 강조합니다. 데이터 정제 과정에서 HTML 형식의 비일관성을 해결하기 위해 특정 가정을 통해 데이터를 처리했습니다. 규칙 기반(regex) 접근 방식을 사용했지만, 다양한 법적 규정 표현 방식으로 인해 추가적인 어려움이 존재했습니다.

- **Performance Highlights**: 논문에서 소개된 데이터 세트는 약 1.1 GB의 크기로, 43337개의 행과 12개의 속성으로 구성되어 있으며, Zenodo를 통해 공개되었습니다. 이 데이터 세트는 법적 텍스트와 해당 법률 간의 의미적 유사성을 계산하는 연구에 활용되었습니다. 법률 전문을 포함한 참조 법률 정의를 정확히 추출하는 것은 여전히 해결해야 할 과제입니다.



### I can listen but cannot read: An evaluation of two-tower multimodal systems for instrument recognition (https://arxiv.org/abs/2407.18058)
Comments:
          Accepted to ISMIR 2024

- **What's New**: 이 논문은 음악 두-타워 멀티모달(two-tower multimodal) 시스템에서 제로-샷(Zero-shot) 악기 인식을 평가하고 분석합니다. 오디오 및 텍스트 모달리티를 결합하여 새로운 방식의 분류 및 검색을 가능하게 하며, 특히 악기의 경우 본격적인 제로-샷 학습 특성을 살펴보았습니다.

- **Technical Details**: 두-타워 시스템은 사전 학습된 오디오 모델과 언어 모델(LM)을 각각 오디오 및 텍스트 인코더로 사용하여, 각각의 표현(embeddings)을 합동 오디오-텍스트 공간으로 매핑(mappings)하여 시스템을 최적화합니다. 본 연구에서는 MusCALL, CLAP 모델(음악 및 음성 데이터 기반과 음악 데이터 기반)을 포함해 총 3개의 시스템을 평가하였습니다.

- **Performance Highlights**: 음악 두-타워 시스템의 경우 제로-샷 학습에 유망한 결과를 보였지만, 텍스트 인코더 또는 합동 공간 투영에서 여전히 문제가 있습니다. 특히, 두-타워 시스템은 특정 단어에 민감하게 반응하며, 더 구체적인 음악 정보를 담은 프롬프트(prompt)보다 일반적인 프롬프트를 선호하는 경향이 있습니다. 텍스트 인코더의 대규모 크기와는 달리 추가적인 텍스트 맥락을 활용하거나 악기에 대한 정확한 설명을 추론하지 못하는 한계가 있습니다.



### BLAZE: Cross-Language and Cross-Project Bug Localization via Dynamic Chunking and Hard Example Learning (https://arxiv.org/abs/2407.17631)
- **What's New**: 최근 arxiv에 게재된 논문에서 BLAZE라는 새로운 버그 로케이션(bug localization) 기법이 제안되었습니다. 개발자는 소프트웨어 버그를 찾고 수정하는데 많은 시간을 소모합니다. BLAZE는 이러한 문제를 해결하기 위해 소스 코드 파일을 동적으로 분할하고 어려운 예제(hard example) 학습을 활용합니다. 이를 통해 다중 프로젝트 및 다중 언어 환경에서도 개선된 성능을 제공합니다.

- **Technical Details**: BLAZE는 소스 코드를 동적으로 분할하는 'Dynamic Chunking' 기법을 사용하여 소스 코드의 연속성 손실을 최소화하고, GPT 기반 모델을 도전적인 버그 사례로 미세 조정(fine-tuning)합니다. BEETLEBOX 데이터셋을 통해 5 가지 프로그래밍 언어(Java, C++, Python, Go, JavaScript)에서 29개의 오픈 소스 프로젝트와 26,321 개 버그를 수집하여 평가에 활용했습니다. BLAZE는 또한 세 가지 벤치마크 데이터셋(SWE-Bench, Ye et al., BEETLEBOX)을 사용하여 평가되었습니다.

- **Performance Highlights**: BLAZE는 기존의 최첨단 기반의 여섯 가지 기법 대비 주요 성능 지표(Top 1 accuracy, MAP, MRR)에서 최대 120%, 144%, 100% 향상을 기록했습니다. 광범위한 ablation 스터디를 통해 BLAZE의 파이프라인 구성 요소가 전체 성능 향상에 기여함을 입증했습니다.



### Unveiling Legitimacy in the unexpected events context : An Inquiry into Information System Consultancy companies and international organizations through Topic Modeling Analysis (https://arxiv.org/abs/2407.17509)
Comments:
          Includes translation of the abstract in French language. 29e Conf{é}rence de l'Association Information et Management 27-29 mai 2024 {à} Montpellier - La Grande-Motte, May 2024, Montpellier, France

- **What's New**: 이 연구는 정보 시스템(IS) 정당화에 대한 영역을 탐구하며, 특히 예상치 못한 사건들에 대한 IS 컨설팅 기업과 국제 조직의 커뮤니케이션에 주목합니다. 연구는 다양한 출판물을 조사하여 두 주요 이해관계자의 정당화 방법을 분석합니다.

- **Technical Details**: 이 연구는 topic modeling methodology(주제 모델링 방법론)을 활용하여 문서들을 분석하고, IS 컨설팅 회사와 국제 기관이 불확실한 사건에 어떻게 대응하고 정당성을 부여하는지를 이해하려고 합니다. 이를 통해 두 주요 IS 이해관계자의 정당화 전략을 심도 있게 살펴봅니다.

- **Performance Highlights**: 연구를 통해 획득한 통찰은 예상치 못한 사건에 대응하는 IS 컨설팅 회사와 국제 기관의 행동 및 전략에 대한 문헌에 기여할 것으로 기대됩니다. 이 연구는 정보 시스템 이해관계자들의 정당화 담론에 중요한 기여를 합니다.




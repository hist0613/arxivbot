New uploads on arXiv(cs.CL)

### Understanding Synthetic Context Extension via Retrieval Heads (https://arxiv.org/abs/2410.22316)
- **What's New**: 이 논문은 합성 데이터(synthetic data)로 훈련된 Long-context LLMs의 성능을 평가하고 해석하는 새로운 통찰을 제공합니다. 특히, 합성 데이터가 Long-context 작업에 미치는 영향을 분석하였습니다.

- **Technical Details**: 저자들은 'needle' 개념의 현실성과 주변 'haystack' 맥락의 다양성을 변화시키면서, LLM을 활용해 합성 문서를 생성하고, 템플릿 관계(templated relations) 및 기호 데이터셋(symbolic datasets)을 만드는 방법을 비교했습니다. 이 과정에서 합성 데이터에 기반한 모델들의 주의 헤드(attention heads)가 실제 데이터에 기반한 모델들의 헤드와 어떻게 연결되어 있는지를 조사했습니다.

- **Performance Highlights**: 합성 데이터에서 훈련된 모델들은 실제 데이터보다 성능이 떨어지는 경향을 보였지만, 특정 주의 헤드의 존재 및 회수(recall)에 기반하여 모델 성능을 예측할 수 있음을 발견했습니다. 또한, 주의 노크아웃(attention knockout)과 활성화 패칭(activation patching)을 통해 주의 헤드가 긴 문맥에서의 검색(retrieval)에 필수적이라는 점을 기계적으로 입증했습니다.



### Natural Language Inference Improves Compositionality in Vision-Language Models (https://arxiv.org/abs/2410.22315)
Comments:
          Project page: this https URL

- **What's New**: 본 연구는 자연어 추론(Natural Language Inference, NLI)을 활용하여 서로 모순되거나 함의가 있는 문장을 생성함으로써 이미지와 텍스트 간의 관계를 보다 정교하게 이해하는 새로운 접근 방식인 캡션 확장(Caption Expansion with Contradictions and Entailments, CECE)을 제안합니다. 이를 통해 모델의 해석 가능성(interpretability)을 높이며, 기존 방법들이 갖는 편향(bias)이나 피상적인 특성에 의존하는 문제를 해결하고자 합니다.

- **Technical Details**: CECE는 텍스트로부터 원래 의미를 유지하면서도 어휘적으로 다양한 문장을 생성합니다. 자연어 추론을 통해 주어진 전제(premise)에 대한 함의(entailments) 및 모순(contradictions)을 생성하며, 이 과정은 LLM(대형 언어 모델)을 활용합니다. CECE는 문장을 다루는 방식에서 기존의 문장 분해 방법(Sentence Decomposition via Semantics, SDS)을 넘어, 보다 풍부하고 다양한 문맥 정보를 제공하여 모델의 성능을 향상시키는 방식입니다.

- **Performance Highlights**: CECE를 적용한 결과, 이미지-텍스트 정렬(agreement with human judgments) 기준에서 이전 방법들보다 월등한 성과를 보였으며, Winoground에서 +19.2%, EqBen에서 +12.9%의 성능 향상을 이루었습니다. CECE는 추가적인 파인튜닝 없이도 이미지와 텍스트 간의 정렬을 개선하며, 기존의 편향 문제를 완화하여 보다 통합적인 결과를 생성합니다.



### Flow-DPO: Improving LLM Mathematical Reasoning through Online Multi-Agent Learning (https://arxiv.org/abs/2410.22304)
Comments:
          5 pages, 4 figures, 1 table

- **What's New**: 이 논문은 온라인 학습 기법인 Flows를 활용하여 고품질의 수학적 추론 추적을 생성하는 새로운 접근 방식을 제안합니다. 이를 통해 LLM(대형 언어 모델)의 성능을 향상시키는 데 필요한 세부적인 추론 단계를 정확하게 제공하는 것이 목표입니다.

- **Technical Details**: 제안된 방법은 두 개의 LLM(응답 LLM과 정지 LLM)을 사용하는 점진적 출력 생산 흐름을 통해 이루어지며, 이러한 LLM 각각은 자신의 아키텍처에 따라 협력하여 솔루션을 구성합니다. 온라인 DPO(Direct Preference Optimization) 학습을 통해 각 훈련 예시에 대해 DPO 쌍을 생성하고 실시간으로 모델을 업데이트하는 방식입니다.

- **Performance Highlights**: MetaMath 데이터 세트를 사용한 실험 결과, 제안된 방법이 기존 모델 추론을 통한 추론 품질보다 우수하다는 것을 입증하였습니다. 제안된 흐름 접근 방식은 다양한 LLM 모델에 걸쳐 일관된 향상을 보여주었습니다.



### From melodic note sequences to pitches using word2vec (https://arxiv.org/abs/2410.22285)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문에서는 언어 모델링에 일반적으로 사용되는 word2vec 기법을 멜로디에 적용하는 새로운 접근 방식을 소개합니다. 음표가 문장에서 단어로 간주되며, 이를 통해 피치(pitch) 정보를 효과적으로 캡처합니다.

- **Technical Details**: 본 연구는 20개의 어린이 노래와 바흐 소나타의 발췌본, 두 개의 데이터셋을 사용하였습니다. 그들의 임베딩(embeddings)은 2차원으로 정의된 매우 작은 차원의 의미 공간(semantic space)을 사용하며, 2, 3 또는 4개의 이전 음표에 기반하여 음표를 예측합니다.

- **Performance Highlights**: 다변량 분석(multivariate analysis) 결과, 음표를 나타내는 의미 벡터는 피치와 약 0.80의 다중 상관 계수를 보여주며, 이는 모델이 음표의 피치를 잘 예측하고 있음을 시사합니다.



### FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation (https://arxiv.org/abs/2410.22257)
Comments:
          25 pages, 10 figures

- **What's New**: 본 논문에서는 LANGUAGE MODELS (LMs)의 사실성을 검증하기 위해 VERIFY(Verification and Evidence RetrIeval for FactualitY evaluation)라는 새로운 파이프라인을 제안합니다. VERIFY는 LM이 생성한 콘텐츠의 검증 가능성을 고려하고, 웹에서 검색된 증거를 기반으로 콘텐츠 단위를 지원, 비지원 또는 결정할 수 없는 것으로 분류합니다. 이는 기존 방법보다 인간 평가와 더 높은 상관관계를 보입니다.

- **Technical Details**: VERIFY는 LM의 응답에서 콘텐츠 단위를 추출하고 그 유형을 식별한 후, 웹 기반 증거를 사용하여 검증 가능한 단위만 평가합니다. 이 과정에서 hallucination(환각) 프롬프트를 식별하고, FactBench라는 데이터셋을 구축하여 150개의 세분화된 주제에 걸쳐 1,000개의 프롬프트를 포함합니다. 프롬프트는 검증 가능성과 유용성이 평가된 후 최종적으로 포함됩니다.

- **Performance Highlights**: 일반적으로 사용되는 LM을 FactBench에서 평가한 결과, (i) 소유 모델이 더 나은 사실성을 보이며, (ii) Llama3.1-405B-Instruct는 subjectivity(주관성)로 인해 Llama3.1-70B-Instruct보다 낮은 정확도를 기록했습니다. (iii) Gemini1.5-Pro는 25% 이상의 경우에 과도한 거부율을 보였습니다. 우리는 VERIFY의 결과가 기존의 방법들보다 인간의 판단과 더 높은 상관관계를 보여줍니다.



### DISCERN: Decoding Systematic Errors in Natural Language for Text Classifiers (https://arxiv.org/abs/2410.22239)
Comments:
          20 pages, 9 figures, 15 tables; Accepted to EMNLP 2024

- **What's New**: DISCERN 프레임워크는 텍스트 분류기의 체계적인 오류를 해석하기 위한 자연어 설명을 제공하며, 이를 통해 분류 성능을 향상시킵니다.

- **Technical Details**: DISCERN은 두 개의 대형 언어 모델(LLMs)을 사용하여 반복적으로 체계적인 오류에 대한 자연어 설명을 생성합니다. 설명 모델은 각 클러스터의 특성을 서술하는 간결한 프레디케이트 스타일 설명을 생성하고, 평가 모델은 이 설명이 해당 클러스터의 예제에만 해당하는지를 판단합니다.

- **Performance Highlights**: 세 가지 텍스트 분류 데이터세트에서 DISCERN의 설명을 통해 10% 이상의 오분류율 감소와 25% 이상의 사용자 해석 효율성을 보여줍니다.



### ProMQA: Question Answering Dataset for Multimodal Procedural Activity Understanding (https://arxiv.org/abs/2410.22211)
Comments:
          18 pages, 11 figures

- **What's New**: 본 논문에서는 Procedural Multimodal Question Answering(ProMQA)라는 새로운 평가 데이터셋을 제시하여, 다중 모달 시스템이 절차적 활동을 이해하고 지원하는 능력을 측정합니다. ProMQA 데이터셋은 절차적 활동과 관련된 사용자 기록 및 그에 상응하는 지침을 포함한 401개의 다중 모달 QA 쌍으로 구성되어 있습니다.

- **Technical Details**: ProMQA의 QA 주석은 비용 효율적인 인간-LLM 협업 접근 방식을 통해 진행됩니다. 여기서 LLM(대형 언어 모델)이 QA 쌍을 생성한 후, 인간이 이를 검증하여 품질을 확보합니다. 이 경우, LLM이 생성한 500개 QA 쌍 중 약 80%가 추가 인간 작성 답변으로 유지되었습니다.

- **Performance Highlights**: 본 연구의 벤치마크 실험 결과, 현재 시스템의 성능은 여전히 인간의 성능과 비교할 때 상당한 격차가 있음을 보여주었습니다. 특히, 현재의 경쟁력 있는 상업용 다중 모달 모델조차도 프로시저 활동 이해에 있어 어려운 성능을 보였습니다.



### Class-Aware Contrastive Optimization for Imbalanced Text Classification (https://arxiv.org/abs/2410.22197)
Comments:
          10 pages, 3 figures, accepted for publication in CODS-COMAD 2024

- **What's New**: 이 논문에서는 클래스 인식을 강화한 대조 최적화(class-aware contrastive optimization)와 잡음 제거 오토인코더(denoising autoencoders)를 결합하여 불균형 텍스트 분류(imblanced text classification) 문제를 성공적으로 해결할 수 있음을 보여주고 있습니다.

- **Technical Details**: CAROL(클래스 인식 대조 손실)은 불균형 텍스트 이진 분류(imblanced text binary classification) 작업을 해결하기 위해 클래스 분리 기반 손실 함수(class separation-based loss function)와 오토인코더 모델(autoencoder-based model)을 결합하여 설계되었습니다. 이러한 접근은 임베딩 공간에서 클래스 분리를 개선하고 생성된 임베딩의 진실성과 다양한 클래스를 분리하는 모델의 능력 간의 균형을 맞춥니다.

- **Performance Highlights**: CAROL은 전통적인 방법 및 최신(state-of-the-art) 경쟁 방법과 비교했을 때 다양한 텍스트 데이터셋에서 눈에 띄는 성능 향상을 보여주었습니다. 특히, CAROL은 클래스 분리와 재구성을 균형 있게 최적화하여 성능을 손실 없이 유지하는 특징이 있습니다.



### Natural Language Processing for Analyzing Electronic Health Records and Clinical Notes in Cancer Research: A Review (https://arxiv.org/abs/2410.22180)
- **What's New**: 이번 리뷰는 전자 건강 기록(EHR)과 임상 노트를 사용하여 암 연구에서 자연어 처리(NLP) 기술의 적용을 분석합니다. 이전 연구들보다 더 포괄적인 시각을 제공하여 특정 암 유형이나 응용 분야에 국한되지 않은 연구의 공백을 해결합니다.

- **Technical Details**: 이번 연구는 2019년부터 2024년 사이에 발표된 94개의 관련 연구를 스코퍼스(Scopus) 데이터베이스를 통해 분석했습니다. 연구들은 암 유형과 NLP 응용 분야에 따라 분류되었습니다. NLP 작업으로는 정보 추출(information extraction)과 텍스트 분류(text classification)가 두드러졌으며, 규칙 기반(rule-based)에서 변환기 기반(transformer-based) 기계 학습(machine learning) 기술로의 전환이 관찰되었습니다.

- **Performance Highlights**: 암 연구에서 NLP의 적용이 증가하고 있으며, 특히 유방암(breast cancer), 폐암(lung cancer), 대장암(colorectal cancer)이 가장 많이 연구되었습니다. 그러나 제안된 솔루션의 일반화(generalizability) 제한 및 임상 워크플로우(clinical workflows)로의 통합 필요성이 주요 도전 과제로 지적되었습니다. 향후 연구는 복잡한 임상 언어 처리의 강화를 목표로 해야 하며, 충분히 연구되지 않은 암 유형으로의 응용 확장도 필요합니다.



### Very Attentive Tacotron: Robust and Unbounded Length Generalization in Autoregressive Transformer-Based Text-to-Speech (https://arxiv.org/abs/2410.22179)
Comments:
          Submitted to NAACL

- **What's New**: 본 논문에서는 오토 리그레시브(AR) Transformer 기반의 텍스트-음성 변환(Text-to-Speech, TTS) 모델의 안정성과 길이 일반화 문제를 해결하기 위한 개선안을 제시합니다. 이를 위해 상대적 위치 정보를 제공하는 정렬 메커니즘을 사용하여 크로스 어텐션(cross-attention) 작업을 수행합니다. 이 정렬 위치는 모델의 잠재적 특성으로 학습되며, 학습 중 외부 정렬 정보가 필요하지 않습니다.

- **Technical Details**: 제안된 시스템인 Very Attentive Tacotron (VAT)은 기본 T5 기반 TTS 아키텍처에 정렬 메커니즘을 추가하여 텍스트-음성 변환 작업의 단조로운 특성을 활용합니다. VAT는 멀티-헤드(Multi-head) 자기-어텐션(self-attention)과 크로스 어텐션의 조합을 통해 강력한 모델링 능력을 유지하면서도 길이 일반화를 가능하게 합니다. VQ-VAE(Variational Quantization Autoencoder)를 사용하여 음성 데이터의 분산화를 수행하며, GAN 기반의 신경 보코더(neural vocoder)를 이용하여 최종 음성을 생성합니다.

- **Performance Highlights**: VAT는 자연스러움과 표현력이 뛰어난 T5-TTS 기본 시스템과 동등한 성능을 발휘하면서도 중복되거나 삭제된 단어 문제를 제거하고, 실제 발화 길이에 대해 일반화할 수 있는 능력을 보여줍니다.



### Benchmarking LLM Guardrails in Handling Multilingual Toxicity (https://arxiv.org/abs/2410.22153)
- **What's New**: 이번 연구에서는 7개의 데이터셋과 10개 이상의 언어를 아우르는 포괄적인 다국어 테스트 수트를 소개하여 최첨단 guardrail의 성능을 벤치마킹하였습니다. 또한 최근의 jailbreaking 기법에 대한 guardrail의 저항력을 조사하고, 안전 정책 및 언어 리소스의 가용성이 guardrail 성능에 미치는 영향을 평가했습니다.

- **Technical Details**: 이 연구에서는 다국어 토크스(Toxicity) 데이터셋을 통해 다국어 환경에서의 guardrail 성능을 분석하고, LlaMa, Mistral 같은 사전 훈련된 LLM을 사용하여 안전 vs. 위험한 콘텐츠 판별을 수행했습니다. 또한, 기존 연구에서 다뤄지지 않았던 다국어 jailbreaking prompt의 탐지 성능을 평가하여 그 한계점을 제시했습니다.

- **Performance Highlights**: 기존의 guardrail는 다국어 토크스 처리에 비효율적이며, jailbreaking 공격에 대한 강인성에서도 부족한 것으로 나타났습니다. 실험 결과, 모든 guardrail 모델은 비영어 데이터에서 일관된 성능 저하를 보여주었고, 낮은 리소스 언어에 대한 높은 오탐률(FPR)을 기록했습니다.



### AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts (https://arxiv.org/abs/2410.22143)
- **What's New**: 본 논문에서는 AmpleGCG의 향상된 버전인 AmpleGCG-Plus를 소개합니다. 이 모델은 맞춤형 gibberish adversarial suffixes를 더 빠르고 효과적으로 생성하여 jailbreak 공격 성능을 강화합니다.

- **Technical Details**: AmpleGCG-Plus는 pre-trained 모델을 기반으로 하며, 100배 더 많은 성공적인 suffix 데이터로 훈련되고, 더 엄격한 harmfulness 분류기를 사용하여 모델의 성능을 향상시킵니다. 이 모델은 오픈-웨이트와 클로즈드-소스 모델 모두에서 높은 attack success rate (ASR)을 기록합니다.

- **Performance Highlights**: AmpleGCG-Plus는 Llama-2-7B-chat 모델에서 17% 이상 ASR을 개선하였고, GPT-4 블랙박스 설정에서 ASR을 세 배 이상 증가시켰습니다. 또한 새로운 GPT-4o 시리즈 모델에 대해서도 비슷한 비율로 성공적인 jailbreak 공격을 수행할 수 있습니다.



### The Impact of Inference Acceleration Strategies on Bias of LLMs (https://arxiv.org/abs/2410.22118)
- **What's New**: 최근 몇 년간 대규모 언어 모델(LLMs)의 성능이 비약적으로 발전했습니다. 그러나 이러한 모델의 inference(추론) 효율성을 높이기 위한 많은 전략들(quantization, pruning 등)이 도입되었음에도 불구하고, 이 과정에서 데이터의 인구 통계학적 편향이 유의미하게 바뀔 수 있다는 점이 주목됩니다.

- **Technical Details**: 우리는 5가지 일반적인 inference acceleration techniques(추론 가속 기술)과 3개의 널리 사용되는 LLM을 통해 편향의 변화를 평가하였습니다. 특히, 모델 출력의 편향을 측정하기 위해 6가지 다른 bias metrics(편향 지표)를 활용하였으며, 각 모델에서 가속화 전략의 영향을 분석했습니다.

- **Performance Highlights**: 분석 결과, inference acceleration 전략이 모델의 편향에 중대한 영향을 미친다는 것을 확인했습니다. 예를 들어, 4-bit AWQ Quantization 방식이 일부 모델의 편향을 크게 변화시켰으나 KV-cache quantization 방식은 상대적으로 견고한 결과를 보여주었습니다. 따라서 가속화를 적용할 때 편향에 대한 신중한 평가가 필요하다는 점이 강조됩니다.



### Protecting Privacy in Multimodal Large Language Models with MLLMU-Bench (https://arxiv.org/abs/2410.22108)
Comments:
          30 pages

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLM)과 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 개인의 기밀 데이터 및 개인 정보를 기억하고 노출할 수 있다는 법적 및 윤리적 문제를 다루고 있습니다. 특히, MLLMs에 대한 머신 언러닝(machine unlearning) 적용이 미비하였음을 강조하며, 이를 해결하기 위해 새로운 벤치마크인 MLLMU-Bench를 도입합니다.

- **Technical Details**: MLLMU-Bench는 500개의 가상 프로필과 153개의 공인 프로필로 구성되어 있으며, 각 프로필에는 14개 이상의 맞춤형 질문-답변 쌍이 포함되어 있습니다. 이 벤치마크는 다중 모달(이미지+텍스트) 및 단일 모달(텍스트) 관점에서 평가됩니다. 또한, 효과성(efficacy), 일반화 가능성(generality), 모델 유용성(utility)을 기준으로 네 개의 세트로 나누어 언러닝 알고리즘을 평가합니다.

- **Performance Highlights**: 실험 결과, 단일 모달 언러닝 알고리즘이 생성(generation) 및 클로즈(cloze) 작업에서 우수한 성능을 보인 반면, 다중 모달 언러닝 도구는 다중 모달 입력을 가진 분류(classification) 작업에서 더 뛰어난 성능을 나타냈습니다.



### Joint Extraction and Classification of Danish Competences for Job Matching (https://arxiv.org/abs/2410.22103)
- **What's New**: 이 연구는 덴마크의 구직 공고에서 능력을 자동으로 추출하고 분류하는 첫 번째 모델을 제시합니다. 기존 연구와 달리, 이 모델은 대규모의 주석이 달린 덴마크 말코퍼스를 기반으로 훈련되었습니다.

- **Technical Details**: 모델은 BERT-like 아키텍처를 기반으로 하며, 직무 공고에서의 능력 추출 및 분류를 단일 아키텍처로 수행합니다. 또한 명명된 개체 인식(NER) 라벨을 생성하여 직무 능력을 추출합니다. 데이터는 20만 개의 문장에서 유럽의 기술, 역량, 자격 및 직업(ESCO) 주석이 추가되어 있습니다.

- **Performance Highlights**: 이 모델은 덴마크 능력 추출 및 분류에서 최첨단 모델들을 초과하는 성능을 보였으며, 추론 기간을 50% 이상 단축시켰습니다.



### Choosy Babies Need One Coach: Inducing Mode-Seeking Behavior in BabyLlama with Reverse KL Divergenc (https://arxiv.org/abs/2410.22081)
- **What's New**: 이번 연구에서는 2차 BabyLM 챌린지의 Strict-Small 트랙에 대한 제출물을 소개합니다. BabyLLaMa 모델을 기반으로 하는 teacher-student distillation 설정을 사용하여, 학생 모델의 학습이 더욱 집중될 수 있도록 objective function을 reverse Kullback-Leibler divergence로 변경하였습니다. 이는 mode-seeking(모드 추적) 행동을 유도하여 학생 모델이 효과적으로 학습할 수 있도록 합니다.

- **Technical Details**: 제안된 ChooBaCa 접근법은 knowledge distillation에서 출발했습니다. Baby Llama 모델을 기반으로 하여, reverse KL divergence를 objective function으로 사용하고, 두 명의 teacher 대신 단일 teacher 모델을 사용하여 distillation 프로세스를 최적화했습니다. 이러한 설정을 통해 학생 모델은 teacher 모델의 주요 모드를 포착할 수 있으며, 이로 인해 학습 과정이 더욱 효율적이고 빠르게 진행됩니다.

- **Performance Highlights**: 실험 결과, reverse KL divergence 환경에서 단일 teacher 모델이 여러 teacher 모델보다 대부분의 작업에서 성능이 더 뛰어나거나 동일한 결과를 나타냈습니다. 특정 최적화 기법을 통합함으로써 모델 성능이 더욱 향상되어 제안된 접근 방식의 효과성과 견고성을 입증했습니다.



### Distinguishing Ignorance from Error in LLM Hallucinations (https://arxiv.org/abs/2410.22071)
- **What's New**: 이 논문에서는 Close-book Question Answering (CBQA) 환경에서 발생하는 환각(hallucination)의 두 가지 유형을 명확히 구분하는 데에 초점을 맞추고 있습니다. 이러한 구분은 LLM의 출력 신뢰성을 높이는 데 중요한 역할을 합니다.

- **Technical Details**: 이 연구의 주요 기여 중 하나는 'Wrong Answer despite having Correct Knowledge (WACK)' 방법론을 소개하는 것입니다. WACK는 모델의 내부 상태를 기반으로 두 가지 유형의 환각(HK- 및 HK+)을 구별하여 모델 특화 데이터셋을 구축합니다. HK-는 지식 부족으로 인한 환각을, HK+는 올바른 지식이 있음에도 잘못된 응답을 생성하는 경우를 나타냅니다.

- **Performance Highlights**: WACK 데이터셋을 사용하여 훈련된 프로브(probe)는 HK+ 환각을 보다 효과적으로 탐지할 수 있음을 보여주었습니다. 특히, WACK 데이터셋은 공유 지식을 가지고 있는 모델 간에도 환각이 발생하는 방식에 차이가 있음을 나타냅니다.



### Sing it, Narrate it: Quality Musical Lyrics Translation (https://arxiv.org/abs/2410.22066)
- **What's New**: 이번 연구는 뮤지컬 번역에 대한 새로운 접근 방식을 제안하고, 번역 품질 향상과 음가성(singability) 유지 간의 균형을 맞추는 것을 목표로 합니다. 이 과정에서 새로운 데이터셋인 MusicalTransEval이 개발되었으며, 이를 통해 뮤지컬 번역 평가를 위한 기초 자료로 활용될 수 있습니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 자동 번역 품질 평가를 위한 보상 모델(reward models) 학습을 위한 데이터셋 구축입니다. 둘째, 번역 품질과 음가성을 동시에 향상시키기 위한 두 단계 훈련 과정입니다. 마지막으로, 전체 노래 번역을 위한 추론 시간 최적화(inference-time optimization) 프레임워크를 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 기준 방법들보다 현저한 개선을 보였으며, 각각의 구성 요소가 본 연구의 효과성을 검증하는 데 기여했습니다. 그 결과, 뮤지컬 번역의 품질이 크게 향상되었음을 확인했습니다.



### Are VLMs Really Blind (https://arxiv.org/abs/2410.22029)
Comments:
          2 pages, 1 figure

- **What's New**: 이 연구는 Visual Question Answering (VQA)에서 시각적 캡션 생성과 질문 응답을 결합하는 새로운 방법을 제안합니다. 기존 접근 방식보다 성능을 향상시키는 동시에 VLMs(사물 시각 언어 모델)이 기하학적 추론에 대한 인식을 증진할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서, Llama 3.1 모델에서 추출된 키워드를 기반으로 이미지 캡션을 생성합니다. 두 번째 단계에서, 생성된 캡션을 VLM(Geminiflash)에 입력하고 해당 질문을 제출하여 답변을 받습니다. 이 방식은 VQA 태스크에서 키워드 기반 캡션 생성의 효과를 증명하기 위해 설계되었습니다.

- **Performance Highlights**: 평균 정확도는 키워드 기반 캡션 방식이 적용될 때 전통적인 VQA 접근 방식에 비해 항상 더 높았습니다. 특히 기하학적 비계수 태스크의 경우, 성능 향상이 두드러진 반면, 계수 관련 태스크에서는 개선이 일관되지 않고 랜덤한 경향을 보였습니다.



### Not All Languages are Equal: Insights into Multilingual Retrieval-Augmented Generation (https://arxiv.org/abs/2410.21970)
- **What's New**: 본 논문은 다양한 언어를 처리하는 RALMs(Retrieval-Augmented Language Models)의 필요성을 강조하며, 다국어 텍스트에서의 효율적인 활용법을 제안합니다. 특히, 'Futurepedia'라는 새로운 벤치마크를 소개하며, 8개 언어를 아우르는 병렬 텍스트 시대를 다룹니다.

- **Technical Details**: Futurepedia는 197개의 병렬 문서와 Q&A 쌍으로 구성되며, 언어별로 Monolingual Knowledge Extraction, Cross-lingual Knowledge Transfer, Multilingual Knowledge Selection의 세 가지 작업을 제공합니다. 이 벤치마크는 Wikipedia에서 수집된 데이터로 구축되었고, GPT-4로 QA 쌍을 생성하여 정확한 답변을 제공합니다.

- **Performance Highlights**: 실험 결과, 고자원 언어에서의 Monolingual Knowledge Extraction 성능이 두드러지며, 인도유럽 언어가 RALMs의 성능을 높이는 경향이 관찰됩니다. 영어는 다국어 환경에서의 선택 편향으로 인해 타 언어보다 훨씬 높은 영향력을 행사하며, 적은 수의 영어 문서만으로도 다른 언어 문서보다 큰 영향력을 미칩니다. 다국어 지식 선택에서 비영어 문서를 더 많이 포함하는 것이 필요합니다.



### SG-Bench: Evaluating LLM Safety Generalization Across Diverse Tasks and Prompt Types (https://arxiv.org/abs/2410.21965)
Comments:
          Accepted by NeurIPS2024 (Dataset and Benchmark Track)

- **What's New**: SG-Bench라는 새로운 벤치마크를 제안하여 LLM의 안전성 평가를 위한 포괄적인 접근 방식을 제공합니다.

- **Technical Details**: SG-Bench는 생성(generative) 및 식별(discriminative) 과제를 통합하고, 다양한 프롬프트(prompts) 유형의 영향을 평가할 수 있도록 설계되었습니다. 이 벤치마크는 3가지 고급 LLM과 10개의 오픈소스 LLM을 평가하였으며, 각 LLM의 안전성 일반화(safety generalization)를 연구합니다.

- **Performance Highlights**: 연구 결과, LLM은 생성 작업에서 뛰어난 성능을 보였지만, 식별 작업에서는 안전성이 크게 저하되었습니다. 이 연구는 프롬프트 유형의 변화가 LLM의 안전성 성능에 미치는 영향을 조사하고, 역할 지향 프롬프트(role-oriented prompts)가 jailbreak 공격에 대한 방어에 도움이 될 수 있음을 발견했습니다.



### Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications (https://arxiv.org/abs/2410.21943)
- **What's New**: 이 논문은 산업 도메인에서 Retrieval Augmented Generation (RAG) 시스템에 멀티모달 모델을 통합하는 실험을 다루고 있으며, 이미지와 텍스트를 함께 사용하여 RAG 성능을 향상시키는 최적 구성을 찾는 데 중점을 두고 있습니다.

- **Technical Details**: 연구에서는 두 가지 이미지 처리 접근 방식과 답변 합성을 위해 두 가지 대형 언어 모델 (LLMs), 즉 GPT-4-Vision과 LLaVA를 사용합니다. 이미지 처리 전략은 멀티모달 임베딩 (multimodal embeddings)과 이미지에서 생성된 텍스트 요약을 포함합니다. 또한, LLM-as-a-Judge 방법론을 사용하여 실험 결과를 평가합니다.

- **Performance Highlights**: 멀티모달 RAG 시스템은 단일 모달 RAG 설정보다 우수한 성능을 보이며, 이미지 검색은 텍스트 검색보다 더 많은 도전 과제가 있음을 보여주었습니다. 이미지에서 생성된 텍스트 요약은 멀티모달 임베딩보다 더 나은 잠재력을 제공하여 미래 발전의 기회를 제시합니다.



### SceneGenAgent: Precise Industrial Scene Generation with Coding Agen (https://arxiv.org/abs/2410.21909)
- **What's New**: 이 논문에서는 산업 제조의 시뮬레이션을 위한 산업 장면 모델링의 중요성을 강조하며, 새로운 LLM 기반 에이전트인 SceneGenAgent를 소개합니다.

- **Technical Details**: SceneGenAgent는 C# 코드를 통해 산업 장면을 생성하며, 정확한 배치 계획(layout planning)을 위해 구조적이고 계산 가능한 형식을 사용합니다. 레이아웃 검증(layout verification)과 반복적 개선(iterative refinement)을 통해 정량적 요구 사항을 충족합니다.

- **Performance Highlights**: SceneGenAgent는 산업 장면 생성 작업에서 최대 81.0% 성공률을 기록하며, 대부분의 장면 생성 요구를 효과적으로 충족합니다. 또한, SceneInstruct 데이터셋을 통해 오픈 소스 LLM을 미세 조정(fine-tuning)하면 성능이 크게 향상되는 것을 보여주었습니다.



### Improving In-Context Learning with Small Language Model Ensembles (https://arxiv.org/abs/2410.21868)
Comments:
          Accepted to NeurIPS 2024 Workshop on Adaptive Foundation Models

- **What's New**: 이 논문에서 제안하는 Ensemble SuperICL은 여러 개의 소형 언어 모델(SLMs)의 전문 지식을 활용하여 In-Context Learning (ICL) 방법을 향상시키고 있습니다. 이 접근법은 최신 기술을 사용하여자연어 이해(NLU) 벤치마크에서 최첨단 성능을 달성하고 있으며, 대량 데이터 라벨링에서 높은 정확도를 보여줍니다.

- **Technical Details**: Ensemble SuperICL은 여러 개의 fine-tuned SLM으로부터 예측값과 신뢰도 점수를 조합하여 LLM의 ICL 능력을 향상시키는 방식입니다. 입력 및 실제 라벨의 쌍을 포함하는 In-Context 예시를 선택하고, 두 개 이상의 SLM이 이 예시에 대해 예측 라벨과 신뢰도 점수를 생성하여 이를 LLM의 입력으로 사용합니다.

- **Performance Highlights**: Ensemble SuperICL은 GLUE 벤치마크와 MedMCQA 데이터셋에서 ICL, SLM, SuperICL의 성능을 초월했습니다. 특히, 의료 라벨링 작업에서 대규모 도메인 특화 데이터를 기존의 모든 기초 모델보다 정확하게 라벨링할 수 있음을 입증했습니다.



### Joint Beamforming and Speaker-Attributed ASR for Real Distant-Microphone Meeting Transcription (https://arxiv.org/abs/2410.21849)
- **What's New**: 본 논문에서는 기존의 SA-ASR 아키텍처가 갖고 있는 멀티채널 노이즈 및 리버베레이션 감소 기능의 부족을 보완하기 위해, 공동 빔포밍(joint beamforming)과 SA-ASR의 결합 방식을 제안합니다.

- **Technical Details**: 데이터 정렬(data alignment) 및 증강(augmentation) 방법을 사용하여 실제 회의 데이터에 대한 신경망 빔포머(neural beamformer) 사전 훈련을 실시합니다. 그런 다음, SA-ASR 모델에 대한 전처리기로서 고정(fixed), 하이브리드(hybrid), 완전 신경망(neural) 빔포머를 비교합니다.

- **Performance Highlights**: 실제 AMI 코퍼스에서 진행된 실험 결과, 최신 멀티프레임 크로스채널 주의 기반(channel attention) 채널 융합이 ASR 성능을 향상시키지 못하는 반면, 고정 빔포머의 출력에서 SA-ASR을 미세 조정(fine-tune)하고 신경망 빔포머와 SA-ASR을 공동 미세 조정했을 때, 각각 8% 및 9%의 단어 오류율(word error rate) 감소를 달성했습니다.



### Multi-aspect Depression Severity Assessment via Inductive Dialogue System (https://arxiv.org/abs/2410.21836)
- **What's New**: 이 논문에서는 다중 측면 우울증 중증도 평가를 위한 새로운 작업인 MaDSA( multi-aspect Depression Severity Assessment via inductive dialogue system)를 제안합니다. 이 시스템은 심리적 대화 응답을 생성하고 감정 분류 작업을 통해 우울증의 다양한 기준으로 환자의 상태를 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: MaDSA는 환자의 우울증 수준을 Interest, Fatigue, Self-esteem과 같은 다양한 기준으로 평가하며, 질의응답(PHQ-8)과 함께 감정 분류를 통해 평가를 지원합니다. 데이터 세트는 DailyDialog 및 EmpatheticDialogues를 기반으로 생성되며, 전체 95,287개의 샘플로 트레이닝하고 13,078개 샘플로 개발하였습니다. T5 모델을 활용하여 사용자 응답 생성을 통해 대화의 맥락에 적합한 응답을 만들어냅니다.

- **Performance Highlights**: 실험 결과, MaDSA의 초기 작업이 효과적임을 보여주었으며, DAIC-WOZ 데이터셋을 이용한 우울증 식별 작업에서 2.48%의 정확도 향상을 달성했습니다. 이러한 결과는 여러 측면을 통한 중증도 측정의 가치와 함께 우울증 진단에 도움을 줄 수 있음을 시사합니다.



### Self-Preference Bias in LLM-as-a-Judg (https://arxiv.org/abs/2410.21819)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 자기 선호 편향(self-preference bias)을 정량적으로 측정하기 위한 새로운 메트릭을 도입합니다. 자가 평가에서 발생하는 편향은 LLM들이 자기 생성 결과에 대해 과대 평가하는 경향을 보이는 것을 의미하며, 이는 특정 스타일이나 정책을 조장할 수 있는 중요한 위험 요소로 지적됩니다.

- **Technical Details**: 연구진은 8개의 LLM에서 자기 선호 편향을 측정하기 위해 새로운 메트릭을 제안하고 이를 통해 GPT-4의 자기 선호 편향 정도를 평가했습니다. 이 메트릭은 알고리즘적 공정성(algorithmic fairness)의 개념을 바탕으로 하며, 텍스트의 perplexity(혼란도)와 결과 평가 간의 관계를 분석했습니다. 평가에서 LLM들은 낮은 perplexity를 가진 출력에 대해 인간 평가자보다 더 높은 평가를 부여하는 경향이 있었습니다.

- **Performance Highlights**: 연구 결과, GPT-4는 유의미한 자기 선호 편향을 보였습니다. 이는 GPT-4가 특정 스타일과 정책에 대해 과도한 영향을 미칠 수 있음을 암시합니다. 또한, 자기 생성된 텍스트가 아닌 경우에도 LLM이 낮은 perplexity를 가진 텍스트에 더 높은 평가를 부여한다는 점에서 LLM의 편향 가능성이 있음을 시사합니다.



### SimSiam Naming Game: A Unified Approach for Representation Learning and Emergent Communication (https://arxiv.org/abs/2410.21803)
- **What's New**: 본 논문에서는 SimSiam+VAE라는 새로운 접근 방식을 제안하여 자기 지도 학습(self-supervised learning, SSL)과 긴급 통신(emergent communication)을 통합합니다. 이는 기존의 SimSiam 모델에 변분 오토인코더(Variational Autoencoder, VAE)를 통합하여 표현 학습을 강화하고 불확실성을 포착합니다.

- **Technical Details**: SimSiam+VAE는 SimSiam 네트워크의 예측기(prediction) 내에 VAE를 통합하여 두 개의 프로세스를 결합합니다: 긍정 쌍을 통한 비교로 정렬하는 과정과 VAE의 인코딩-디코딩 과정을 통해 표현을 다듬는 과정입니다. 이 모델은 에이전트 간의 상호작용을 통해 새로운 언어를 개발할 수 있는 SimSiam Naming Game (SSNG) 구조로 확장됩니다.

- **Performance Highlights**: 실험 결과, SimSiam+VAE는 SimSiam 및 VI-SimSiam 모델보다 뛰어난 성능을 보였으며, SSNG는 참조 게임(referential game)과 메트로폴리스-헤이스팅스(Metropolis-Hastings) 네이밍 게임과 비교했을 때 비슷한 성능을 보여주었습니다.



### Enhancing Adversarial Attacks through Chain of Though (https://arxiv.org/abs/2410.21791)
- **What's New**: 이 논문은 정렬된 대규모 언어 모델(LLM)에서 적대적 공격의 강인성을 향상시키기 위해 Chain of Thought (CoT) 프롬프트와 Greedy Coordinate Gradient (GCG) 기법을 통합하는 새로운 접근 방식을 제안하고 있습니다. 이 방법은 CoT 트리거를 사용하여 LLM의 추론 능력을 자극하며, 적대적 공격의 전이성과 보편성을 개선합니다.

- **Technical Details**: 제안된 방법은 CoT 프롬프트를 사용하여 원하는 응답을 유도하기 위해 공격 목표 대신 GCG 기법을 활용하여 LLM의 추론 단계를 활성화합니다. CoT는 두 가지 형태로 분류되며, 첫 번째는 사용자의 프롬프트 뒤에 '단계별로 생각해 보자' 같은 트리거를 추가하는 것입니다. 이러한 방법을 통해 LLM의 안전성 제약을 유지하면서도 적대적 공격의 유효성을 높일 수 있습니다.

- **Performance Highlights**: 실험 결과, CoT-GCG 접근 방식이 기존의 GCG 공격 및 CoT 프롬프팅 방식보다 뛰어난 성능을 보였으며, Amazon Web Services의 Auto-CoT과 비교하는 절삭 연구에서 더 나은 결과를 기록했습니다. 또한, Llama Guard를 활용하여 LLM의 출력 평가를 보다 객관적으로 수행하여 대화의 위험 수준을 보다 면밀하게 분석하였습니다.



### Leveraging LLMs for Hypothetical Deduction in Logical Inference: A Neuro-Symbolic Approach (https://arxiv.org/abs/2410.21779)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)이 외부 논리적 기호 해결기에 의존하지 않고도 충실한 논리적 추론을 수행할 수 있도록 돕는 LINA라는 신경-기호(neuro-symbolic) 접근법을 소개합니다. LINA는 전제 추출에서 복잡한 논리적 추론으로의 자율적 전환을 가능하게 함으로써, 기존 기법들이 가진 일반화 문제와 정보 손실 문제를 해결합니다.

- **Technical Details**: LINA는 정보 추출 모듈과 LLM 기반 신경-기호 추론 모듈로 구성됩니다. 정보 추출 모듈은 1차 논리(FOL)를 이용해 정보를 손실 없이 유지하며, LLM 기반 신경-기호 추론 모듈은 외부 자원에 의존하지 않는 귀납적 추론 방법을 통해 일반화 문제를 해결합니다.

- **Performance Highlights**: LINA는 다양한 데이터셋에서 기존 신경-기호 방법 및 프롬프트 방법들과 비교하여 최대 35.24% 향상된 성능을 보였으며, FOLIO 데이터셋에서는 LINC에 비해 24.34% 개선된 성과를 달성했습니다. M대비 CoT 및 CoT-SC와 같은 프롬프트 방법에서도 최대 24.02%의 정확도 향상을 이끌었습니다.



### RELATE: A Modern Processing Platform for Romanian Languag (https://arxiv.org/abs/2410.21778)
- **What's New**: 이 논문은 RELATE 플랫폼의 설계 및 진화를 다룹니다. RELATE는 주로 루마니아어를 위한 자연어 처리(NLP) 활동을 위한 고성능 환경을 제공하며, 최근에는 오디오 처리 도구를 통합하였습니다.

- **Technical Details**: RELATE 플랫폼은 모듈식이며, REST 웹 서비스를 통해 다른 시스템과의 상호작용을 지원합니다. 이 플랫폼은 문서 업로드, 다운로드, 편집 및 병렬 처리를 지원하여 대규모 코퍼스 관리를 효율적으로 수행합니다. 또한, 플랫폼은 여러 NLP 기능을 통합하여 다중 수준의 주석을 자동으로 처리하고 시각화합니다.

- **Performance Highlights**: RELATE는 여러 국가 및 국제 연구 프로젝트에서 활발히 사용되고 있으며, 고유의 텍스트 및 오디오 처리 기능을 통해 루마니아어 코퍼스 처리에 있어 현대적이고 성숙한 플랫폼으로 자리잡고 있습니다.



### Learning and Unlearning of Fabricated Knowledge in Language Models (https://arxiv.org/abs/2410.21750)
- **What's New**: 이 논문에서는 대규모 언어 모델(LM)에 새로운 지식이 주입될 때 발생하는 현상과 이러한 지식이 얼마나 오래 지속되는지를 연구합니다. 이를 위해, 다양한 사실 유형을 테스트할 수 있도록 설계된 새로운 probing 데이터셋인 'Outlandish'를 사용합니다.

- **Technical Details**: 연구 결과, 일반 지식과 상충하는 사실(Knowledge-Conflicting Facts, KCFs)은 수만 번의 훈련 단계 동안 강력하게 기억되며, 일반적인 혹은 무작위로 섞인 사실에 비해 더 오래 지속된다는 것을 보여주었습니다. KCFs는 언어 모델이 논리적으로 관련이 없는 프롬프트에서 환각을 유도할 때 더 많이 작용합니다. 이러한 사실들은 새로운 다단계 희소 업데이트(multistep sparse updates)를 적용하여 지워질 수 있음이 드러났습니다.

- **Performance Highlights**: 이 연구는 데이터 포이즈닝(data poisoning)의 영향 완화 방법에 대한 중요한 시사를 제공합니다. KCFs는 상당히 오랜 기억 지속성을 가지지만, 적절한 업데이트 희소화 기법을 통해 이러한 독성이 제거될 수 있습니다.



### Enhancing Financial Question Answering with a Multi-Agent Reflection Framework (https://arxiv.org/abs/2410.21741)
Comments:
          Accepted by ICAIF 24

- **What's New**: 이 연구는 재무 질문응답(QA) 작업을 위한 다중 에이전트 프레임워크를 제안하며, 각 질문에 대해 사고 과정을 반영하는 비평가 에이전트를 포함하여 다중 비평가 에이전트를 추가로 통합하여 성능을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 LLM(대형 언어 모델) 코어 기반의 전문가 에이전트와 비평가 에이전트를 구현하며, 체인 오브 사고(Chain-of-Thought) 프로세스를 통해 데이터를 추출하고 수학적 추론을 수행합니다. 에이전트는 두 개의 하위 작업으로 분리되고, 각 비평가의 피드백이 첫 번째 에이전트의 초기 응답 재평가에 전달됩니다.

- **Performance Highlights**: 이 연구 결과는 LLaMA3-8B 모델에서 평균 15%, LLaMA3-70B 모델에서 5%의 성능 향상을 보여주며, 단일 에이전트 프레임워크와 비교해 통계적으로 유의미한 개선을 나타냅니다. 또한 제안된 프레임워크는 GPT-4o-mini 및 Claude-3.5 Sonnet과 같은 대형 모델과 유사한 성능을 보이면서 비용 효율적인 솔루션을 제공합니다.



### Let's Be Self-generated via Step by Step: A Curriculum Learning Approach to Automated Reasoning with Large Language Models (https://arxiv.org/abs/2410.21728)
- **What's New**: 본 연구에서는 LBS3라는 새로운 자동 추론 프로세스를 제안합니다. 이 방법은 커리큘럼 학습(curriculum learning)에서 영감을 받아 LLMs가 쉽게-어렵게 구조화된 프록시 쿼리를 생성하도록 유도합니다.

- **Technical Details**: LBS3는 두 단계의 프록시 쿼리 생성을 통한 접근 방식을 채택하여, 먼저 쉬운 프록시 쿼리를 생성한 후, 이를 바탕으로 어려운 프록시 쿼리를 생성합니다. 이를 통해 각 쿼리의 난이도를 적절하게 조절합니다.

- **Performance Highlights**: 다양한 추론 집약적 작업에서 LBS3는 기존의 SOTA(Sate-of-the-Art) 기준과 비교하여 강력하게 경쟁하는 성능을 보여주었습니다.



### A Bayesian Approach to Harnessing the Power of LLMs in Authorship Attribution (https://arxiv.org/abs/2410.21716)
- **What's New**: 이 연구는 사전 훈련된 대형 언어 모델(Large Language Models, LLMs)을 활용하여 단회 저자 귀속(one-shot authorship attribution) 문제를 해결하는 방안을 제시하며, 베이esian 접근법과 LLM의 확률 출력을 사용하여 저자 추정을 정교하게 수행합니다.

- **Technical Details**: 저자는 LLM으로부터 얻은 로그 확률을 활용하여 특정 저자가 작성했을 확률을 추정하는 Bayesian 접근 방식을 사용합니다. 'Logprob method'라는 새로운 방법론을 제안하며, 이 방식은 사전 훈련의 목표와 분류 작업을 일치시켜, 복잡한 수작업 특성 엔지니어링 없이 언어의 미세한 뉘앙스를 자동으로 포착합니다.

- **Performance Highlights**: 이 방법론은 IMDb와 블로그 데이터셋에서 10명의 저자에 대해 85%의 정확도를 달성하며, LLMs를 이용한 단회 저자 분석의 새로운 기준을 설정하고 법언어학(forensic linguistics) 분야에서의 적용 가능성을 확장합니다.



### CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs (https://arxiv.org/abs/2410.21695)
- **What's New**: 새로운 CFSafety 벤치마크를 소개하며, 5개의 고전적인 안전 시나리오와 5가지의 명령 공격 유형을 포함한 총 10개 카테고리의 안전 질문을 결합하여 25,000개의 프롬프트로 테스트 세트를 구성했습니다.

- **Technical Details**: CFSafety는 자연어 생성(NLG) 능력을 평가하기 위해 0-5의 안전 등급 척도와 간단한 도덕적 판단을 결합하여 사용합니다. 이 벤치마크를 통해 OpenAI의 GPT 시리즈를 포함한 8개의 인기 있는 LLM의 안전성을 테스트했습니다.

- **Performance Highlights**: GPT-4는 비교 우수한 안전 성능을 보여주었으나, 다른 LLM 모델들의 안전성은 여전히 개선이 필요하다는 결과를 도출했습니다.



### $f$-PO: Generalizing Preference Optimization with $f$-divergence Minimization (https://arxiv.org/abs/2410.21662)
- **What's New**: 이 논문은 언어 모델의 인간 선호 최적화를 위한 새로운 프레임워크인 $f$-divergence Preference Optimization ($f$-PO)을 소개합니다. 이 방법은 기존의 여러 접근 방식을 일반화하고 확장하여 다양한 $f$-다이버전스(f-divergence)를 사용하는 정렬 방법의 광범위한 가족을 포함합니다.

- **Technical Details**: $f$-PO는 정책 모델과 최적 모델 간의 $f$-다이버전스 최소화를 통해 선호 최적화를 분포 일치(problem)로 모델링합니다. 이는 DPO(Direct Preference Optimization)와 EXO를 재현하며, 다양한 $f$-다이버전스를 선택하여 새로운 변형을 제공합니다. 또한, 다양한 $f$-다이버전스의 영향에 대한 세부 분석을 통해 오프라인 선호 최적화에서 정규화와 성능 간의 트레이드오프(insights)를 탐구합니다.

- **Performance Highlights**: 실험 결과에 따르면 $f$-PO는 AlpacaEval 2, Arena-Hard, MT-Bench와 같은 인기 있는 벤치마크에서 기존 방법에 비해 우수한 성능을 보여주었습니다. 특히 $	ext{α-divergence}$를 사용할 때, Llama3-8B-instruct에서 최대 45.3%의 승리율로 성능이 개선되었습니다.



### Are Paraphrases Generated by Large Language Models Invertible? (https://arxiv.org/abs/2410.21637)
- **What's New**: 이 논문에서는 패러프레이즈(Paraphrase) 역전(paraphrase inversion) 문제를 다루고 있습니다. 이는 재작성된 문서가 주어졌을 때 원본 텍스트를 복원하는 과제를 목표로 합니다.

- **Technical Details**: 연구진은 원본 텍스트와 그에 상응하는 패러프레이즈 쌍을 기반으로 한 지도학습(Supervised Learning)을 통해 패러프레이즈 역전 모델을 훈련했습니다. 저자 특정 정보(author-specific context)를 활용한 두 가지 접근 방식이 제안되었습니다: 타겟 저자의 글쓰기 예시를 사용하는 방법과 저자의 독특한 스타일을 포착하는 학습된 스타일 표현을 사용하는 방법입니다.

- **Performance Highlights**: 실험 결과, 패러프레이즈된 기계 생성 텍스트에서 원본 문서의 의미를 상당 부분 복원할 수 있었으며, 저자 특정 정보가 있을 때 효과적임을 보여주었습니다. 비슷한 스타일의 텍스트가 생성된 소스 문서에서는 복원이 더 어려워질 수 있지만, 여전히 스타일 기반의 표절 탐지기(plagiarism detectors) 및 저자 식별 시스템에서 성능이 크게 향상되었습니다.



### MCPDial: A Minecraft Persona-driven Dialogue Datas (https://arxiv.org/abs/2410.21627)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)을 활용하여 플레이어와 비플레이어 캐릭터(NPC) 간의 페르소나(펄소나) 기반 대화를 생성하는 혁신적인 접근법을 제안합니다. 이를 통해 Minecraft와 같은 인기 게임에서 사용할 수 있는 MCPDial 프레젠스를 소개합니다. MCPDial은 기본 대화를 넘어 깊이 있는 상호작용을 가능하게 합니다.

- **Technical Details**: MCPDial 데이터셋은 250개의 NPC 페르소나 설명과 관련된 대화로 구성되어 있으며, LLM을 활용하여 대화 예시를 생성합니다. 이 데이터셋은 NPC와 플레이어 간의 페르소나 일관성을 유지하며 게임 특정 기능 호출을 통합하여 상호작용성을 증대시킵니다. 데이터 생성 과정에는 gold-standard 페르소나 및 대화 수집, 그리고 LLM을 이용한 데이터 증강이 포함됩니다.

- **Performance Highlights**: MCPDial 데이터셋은 유창성, 페르소나 일관성 및 기능 호출 정확성 측면에서 높은 품질의 대화를 포함하고 있으며, Minecraft 도메인 내에서 대화형 에이전트를 교육하고 평가하는 데 있어 매우 유용한 자원으로 작용합니다.



### Reducing the Scope of Language Models with Circuit Breakers (https://arxiv.org/abs/2410.21597)
- **What's New**: 이 논문은 언어 모델들이 특정한 용도로 특정 쿼리에만 답변하게 하려는 새로운 방법, 즉 'scoping'을 제안합니다. 기존 시스템 프롬프트를 이용한 방법이 불완전하다는 것을 발견하고, 'Circuit Breakers' (CB)라는 새로운 메소드를 제시합니다.

- **Technical Details**: CB 방법은 언어 모델의 출력을 특정 작업(예: 감정 분석, 요약하기)에 맞춰 scoping하는 데 사용됩니다. 또한, Supervised Fine Tuning (SFT)과 결합하여 정확한 쿼리에서의 성능을 향상시키고, 비관련 쿼리는 거부할 수 있는 구조를 제공합니다.

- **Performance Highlights**: CB 방법은 기존의 미세 조정(fine-tuning) 방식이나 선호 학습(preference learning)에 비해 아웃 오브 디스트리뷰션(out of distribution) 작업에 대한 강건성이 뛰어나며, 다수의 허용 작업을 지원하여 보다 정교한 범위를 제공합니다. SFT와 CB를 결합할 때 쿼리 성능 향상과 비관련 쿼리 거부에서 모두 긍정적인 결과를 보여줍니다.



### Thank You, Stingray: Multilingual Large Language Models Can Not (Yet) Disambiguate Cross-Lingual Word Sens (https://arxiv.org/abs/2410.21573)
- **What's New**: 이번 연구에서는 여러 언어의 감정을 평가하기 위해 새로운 벤치마크인 StingrayBench를 도입했습니다. 특히, false friends(의미가 다른 유사한 단어들)를 활용하여 다국어 대형 언어 모델(LLMs)의 한계를 분석합니다.

- **Technical Details**: StingrayBench는 인도네시아-말레이(Indonesian-Malay), 인도네시아-타갈로그(Indonesian-Tagalog), 중국어-일본어(Chinese-Japanese), 영어-독일어(English-German) 등 네 가지 언어 쌍에 대한 cross-lingual sense disambiguation(언어 간 의미 구분) 평가를 위해 설계되었습니다. 새로운 metrics(측정 기준)인 cognate bias(유사어 편향)와 cognate comprehension score(유사어 이해 점수)를 도입하여 LLM의 성능을 평가합니다.

- **Performance Highlights**: 분석 결과, LLM들이 자원(RResources)이 풍부한 언어에 편향되어 있다는 것을 발견했습니다. 이러한 연구는 더 다양한 다국어 커뮤니티를 위한 공정한 접근성을 촉진하는데 기여합니다.



### MultiTok: Variable-Length Tokenization for Efficient LLMs Adapted from LZW Compression (https://arxiv.org/abs/2410.21548)
- **What's New**: 이번 논문은 Lempel-Ziv-Welch (LZW) 데이터 압축을 기반으로 한 새로운 토크나이징 방법인 MultiTok을 제안합니다. 이는 반복적인 구문을 다중 단어 토큰으로 압축하여 더 효율적인 LLM 훈련을 가능하게 만듭니다.

- **Technical Details**: MultiTok은 입력된 데이터를 인코더를 사용하여 토크나이즈하고, 의미와 맥락 정보를 나타내는 임베딩 벡터로 추출한 후, 변환기 모델로 전달합니다. 기존의 토큰을 적은 수의 토큰으로 변환하여 크기를 줄이는 방식으로 작동합니다. 이 방법은 훈련 데이터의 33%를 동적으로 압축하고, 훈련 속도를 2.5배 가량 증가시킵니다.

- **Performance Highlights**: MultiTok은 BERT에 비견되는 성능을 유지하면서도 30% 이상의 훈련 데이터 감소를 달성하였습니다. 이러한 접근은 정보 이론적 방법을 활용하여 효율적이고 안정적인 LLM 시스템을 제공할 가능성이 있습니다.



### Unveiling Context-Aware Criteria in Self-Assessing LLMs (https://arxiv.org/abs/2410.21545)
- **What's New**: 본 논문에서는 기존의 정적이고 인간이 정의한 기준에 의존하는 LLM 평가방법의 한계를 극복하기 위해 Context-Aware Criteria (SALC)라는 새로운 Self-Assessing LLM 프레임워크를 제안합니다. 이 프레임워크는 각 평가 인스턴스에 맞춘 동적인 지식을 통합하여 평가 성능을 높이고 다양한 과제에 유연하게 적응할 수 있습니다.

- **Technical Details**: 제안된 SALC 프레임워크는 LLM이 자동으로 평가 기준을 생성하여 동적이고 맥락에 맞는 평가를 가능하게 합니다. SALC는 절대적 설정(Instruction, Reference Answer, Response를 기반으로 기준 생성) 및 상대적 설정(여러 응답 비교) 두 가지 방식으로 작동합니다. 또한 SALC-Tune을 사용하여 작은 모델들을 GPT-4의 기준으로 미세 조정하여 평가 비용을 절감하며 성능을 향상시킵니다.

- **Performance Highlights**: SALC는 다양한 데이터셋에 대해 평균 4.8% 향상된 성능을 보여주었고, Direct Preference Optimization (DPO)에서는 LC Win-Rate가 최대 12% 개선되었습니다. 이 결과들은 SALC가 LLM 평가 및 선호 기반 학습 작업 모두에서 유망한 효과를 지닌 평가 프레임워크임을 입증합니다.



### Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups (https://arxiv.org/abs/2410.21508)
- **What's New**: 본 논문에서는 Sparse Autoencoders (SAEs)의 훈련 전략을 혁신적으로 개선하여 동일한 층 그룹에 대해 하나의 SAE만 훈련하도록 하는 방법을 제안합니다. 이로 인해 기존의 층별 훈련에서 연속적인 층 그룹 훈련으로 전환하게 됩니다.

- **Technical Details**: SAEs는 대규모 언어 모델(LLM)의 활성화를 해석 가능한 특징의 희소 선형 조합으로 재구성하는 비지도 학습 방법입니다. 다층의 LLM에서 각각의 층마다 하나의 SAE를 훈련하는 전통적인 접근법을 대신하여, 연속된 층들에 대해 하나의 SAE를 훈련함으로써 비약적인 계산 효율성을 제공합니다. 실험 결과, Pythia 160M 모델에서 훈련 속도가 6배 향상됨을 보여주었습니다.

- **Performance Highlights**: 모델의 재구성 품질이나 하위 작업의 성능을 저하시키지 않고, SAE 훈련의 효율성을 크게 향상시켰습니다. 이로 인해 현대 LLM에서의 SAE 훈련이 보다 효율적으로 진행될 수 있게 되었습니다.



### SandboxAQ's submission to MRL 2024 Shared Task on Multi-lingual Multi-task Information Retrieva (https://arxiv.org/abs/2410.21501)
Comments:
          MRL 2024 Shared Task on Multi-lingual Multi-task Information Retrieval; 4th Multilingual Representation Learning (MRL) Workshop; EMNLP 2024

- **What's New**: 본 연구는 다양한 언어에서 질문 응답(Question Answering) 및 개체명 인식(Named Entity Recognition) 문제를 탐구합니다. 우리는 여러 가지 프롬프트 기법을 사용하여 다섯 개의 대형 언어 모델(Large Language Models)을 테스트하였으며, 그 결과 일부 모델이 일관되게 다른 모델보다 우수한 성능을 보였지만, 작업 및 언어에 따라 효과가 다르게 나타났습니다.

- **Technical Details**: 고급 프롬프트 기술은 일반적으로 QA 성능을 개선하지만 NER에는 혼합된 결과를 보였으며, 언어의 난이도 패턴이 작업에 따라 다르게 나타났습니다. 특히, 영문을 중간 단계로 활용하는 것이 저자원 언어에 대한 성능을 향상시키는 데 도움이 되는지를 조사하였고, DSPy를 통해 강화된 프롬프트 기법을 활용했습니다.

- **Performance Highlights**: 모델 구조, 프롬프트 전략 및 언어적 특성 간의 복잡한 상호작용은 향후 발전이 보다 미세화된 언어별 접근법을 요구할 수 있음을 시사합니다. 번역 및 고급 프롬프트 기법의 효과는 언어와 모델에 따라 크게 달라지므로 다양한 언어 및 모델에 이러한 방법을 적용할 때 신중한 고려가 필요합니다.



### RoBIn: A Transformer-Based Model For Risk Of Bias Inference With Machine Reading Comprehension (https://arxiv.org/abs/2410.21495)
- **What's New**: 이 연구에서는 새로운 데이터셋인 Risk of Bias Inference (RoBIn)를 소개하고 머신러닝을 이용하여 임상 시험의 Risk of Bias (RoB)를 자동으로 평가할 수 있는 모델을 개발했습니다.

- **Technical Details**: RoBIn은 이중 작업 접근법을 사용하여 주어진 맥락에서 증거를 추출하고 수집된 증거를 기반으로 RoB를 평가합니다. 이 연구는 Cochrane Database of Systematic Reviews (CDSR) 데이터를 활용하여 PubMed에서 clinical trial 게시물을 라벨링하고, RoBInExt (extractive)와 RoBInGen (generative) 두 가지 Transformer 기반 접근법을 구현했습니다.

- **Performance Highlights**: RoBIn은 다양한 설정에서 평가되었으며, 대규모 언어 모델을 포함한 기존 RoB 추론 방법들과 비교했습니다. 대부분의 경우, RoBIn의 최상의 변형이 전통적인 기계 학습 및 LLM 기반 접근법을 초과하여 ROC AUC 0.83을 달성했습니다.



### Can Large Language Models Act as Symbolic Reasoners? (https://arxiv.org/abs/2410.21490)
Comments:
          18 pages, currently under review

- **What's New**: 본 논문은 Large Language Models (LLMs)의 기호적 추론(Symbolic Reasoning) 수행 능력에 대한 최신 연구를 탐구하고, LLM이 본질적으로 추론을 제공할 수 있는지에 대한 질문을 제기합니다. 또한 연구의 현재 격차와 미래 동향을 파악하고 있습니다.

- **Technical Details**: LLM은 주로 transformer architecture를 기반으로 하여 자연어 처리(NLP) 응용 프로그램에서 우수한 성능을 보여주고 있습니다. 이 논문은 LLM의 구조와 기호적 추론 간의 관계, 그리고 LLM의 결정 과정에 대한 대안적 접근인 prompt engineering, chain-of-thought, tree-of-thought, knowledge graph 등의 기법들에 대해 논의하고 있습니다.

- **Performance Highlights**: LLMs는 출력 생성 과정에서 기호적 추론을 효과적으로 수행하지 못하지만, 외부에서 제공되는 가이드라인이나 프롬프트를 통해 그들의 출력을 향상시킬 수 있음을 보여주었습니다. 또한 LLM의 결정에 대한 설명 가능성을 증대시키는 전략이나 계획 능력이 중요한 미래의 AI 시스템 개발을 위한 핵심 요소로 제시되었습니다.



### SpeechQE: Estimating the Quality of Direct Speech Translation (https://arxiv.org/abs/2410.21485)
Comments:
          EMNLP2024

- **What's New**: 최근 머신 번역의 자동 품질 평가에서 음성 모달리티가 충분히 탐구되지 않았습니다. 본 연구에서는 음성 번역 품질 추정(SpeechQE) 문제를 정의하고 벤치마크를 구축하여 다양한 시스템을 평가했습니다. 또한, 사전 훈련된 텍스트 LLM을 활용한 새로운 엔드 투 엔드(end-to-end) 시스템을 도입하였습니다.

- **Technical Details**: 이 작업에서는 음성 번역 품질(E2E \\speechQE) 추정 작업을 정의하고, 기존 ASR 및 텍스트 품질 추정 모듈을 사용한 계단식(cascaded) 시스템과 새로운 E2E 시스템을 탐구합니다. 우리의 모델 아키텍처는 프리트레인(pre-trained)된 음성 인코더와 대형 언어 모델(large language model, LLM)을 포함하여 고품질 오디오 특징을 추출하고 번역 관련 작업을 효과적으로 처리하도록 설계되었습니다.

- **Performance Highlights**: E2E 모델이 인간의 직접 평가 및 측정 기준 점수와의 상관관계에서 SOTA ASR 모듈을 기반으로 한 계단식 시스템보다 더 나은 성과를 보였습니다. E2E 모델은 오류 범위를 탐지할 수 있으며, qualitatively 분석을 통해 E2E 모델이 잘못된 음성 표현에 대해 강건성을 보인다는 것이 확인되었습니다.



### TransformLLM: Adapting Large Language Models via LLM-Transformed Reading Comprehension Tex (https://arxiv.org/abs/2410.21479)
- **What's New**: 이 연구에서는 법률 애플리케이션에 특화된 Phi-2-Legal 및 Mistral-Legal-7B라는 언어 모델을 개발하였습니다. 이 모델들은 500백만 토큰 이상의 법률 텍스트로 계속해서 사전 학습을 진행하였으며, 법률 작업에서의 성능을 크게 향상시켰습니다.

- **Technical Details**: 이 모델들은 이전에 학습한 대형 언어 모델(LLM)을 기반으로 하며, 법률 관련 원시 데이터를 읽기 이해 자료로 변환하여 지속적 사전 학습을 수행합니다. AdaptLLM 기법을 사용하여 법률 텍스트를 읽기 이해 형식으로 변환하고, 이를 통해 발생한 데이터의 고품질화에 중점을 두었습니다. LoRA(저전력 재학습)와 같은 파라미터 효율적인 미세 조정 기법이 적용되었습니다.

- **Performance Highlights**: 새로운 법률 LLM들은 기존에 더 많은 자원으로 훈련된 모델들보다 우수한 성능을 보여주며, 법률 기준에서도 최신 상태의 성과를 달성했습니다. 이 연구는 도메인 특화 텍스트에서의 지속적 사전 학습의 효과를 증명하며, 다양한 작업에서 개선을 가져오는 가능성을 강조합니다.



### Estimating Causal Effects of Text Interventions Leveraging LLMs (https://arxiv.org/abs/2410.21474)
- **What's New**: 이 논문에서는 사회 시스템에서의 언어적 개입의 효과를 정량화할 수 있는 새로운 방법론인 CausalDANN을 제안합니다. 전통적인 인과 추론 방법이 텍스트 데이터를 처리하는 데 한계가 있는 문제를 해결하는 데 초점을 맞추었습니다.

- **Technical Details**: CausalDANN은 대규모 언어 모델(LLM)을 사용하여 텍스트 변환을 통해 인과 효과를 추정하는 프레임워크입니다. 이 방법은 임의의 텍스트 개입을 처리할 수 있으며, 관찰된 데이터와 변환된 데이터의 결과를 예측하여 인과 효과를 추정합니다. 또한, 도메인 적응(neural DANN) 모델을 결과 예측에 적용하여 데이터 이동(data shift) 문제를 해결합니다.

- **Performance Highlights**: CausalDANN의 성능을 검증하기 위해 세 가지 실험을 통해 언어와 관련된 속성의 인과성을 정확하게 추정하였습니다. 첫 번째는 아마존 리뷰의 긍정적인 감정이 판매량에 미치는 영향, 두 번째는 소셜 미디어 포스트에 대한 인기 있는 반응이 도덕적 판단에 미치는 영향, 세 번째는 소셜 미디어 포스트의 분노가 도덕적 판단을 변화시키는 영향을 추정했습니다. 이 방법은 근거 기반 의사 결정을 지원하고 복잡한 사회적 현상을 깊이 이해하는 데 기여합니다.



### UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function (https://arxiv.org/abs/2410.21438)
- **What's New**: 이 논문에서는 Unified Fine-Tuning (UFT)라는 새로운 방법론을 제안하여 Supervised Fine-Tuning (SFT)와 alignment을 하나의 훈련 단계로 통합합니다. 이 접근 방식은 catastrophic forgetting 문제를 효과적으로 방지하며, downstream task에서 SFT에 비해 성능 향상을 보입니다.

- **Technical Details**: UFT는 이전의 SFT와 alignment의 목표와 손실 함수를 동일하게 사용하여 훈련합니다. 이는 implicit reward function을 통해 이루어지며, instruction-tuning 데이터를 alignment 데이터로 고려하여 두 데이터 세트를 혼합함으로써 훈련합니다. 이 방법은 강화학습(RL) 기반의 불안정성 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, UFT는 instruction-following에 대한 ifeval task 및 사실성(factuality)에 대한 truthful-qa task에서 상당한 성능 개선을 보여주었습니다. UFT는 각 단계를 순차적으로 적용하는 것에 비해 명확한 장점을 가지고 있습니다.



### CT2C-QA: Multimodal Question Answering over Chinese Text, Table and Char (https://arxiv.org/abs/2410.21414)
Comments:
          10 pages, 6 figures

- **What's New**: 본 연구는 텍스트, 표, 차트를 포함한 새로운 중국어 기반의 멀티모달 질문 응답 데이터 세트인 C\text{T}^{2}C-QA를 소개합니다. 이는 현재의 한계와 다양한 모달리티를 사용하는 질문에 대한 응답을 통합적으로 분석하는 새로운 기준을 제시합니다.

- **Technical Details**: C\text{T}^{2}C-QA 데이터 세트는 200개의 웹페이지에서 수집한 9,981개의 질문-답변 쌍을 포함하고 있습니다. 새로운 multi-agent 시스템인 AED(Allocating, Expert and Decision)는 다양한 모달리티(텍스트, 표, 차트)를 처리하는 전문가 에이전트를 할당하고 상호작용하여 결정을 내리는 기능을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, 우리의 데이터 세트에서 AED의 성능은 KM = 33.9 및 CLKM = 34.3에 도달했으며, 이는 아직 개선이 필요함을 나타냅니다. 현재의 방법론(GPT-4 포함)보다 더 높은 성능 목표를 가지고 있음을 보여줍니다.



### A Survey on Automatic Credibility Assessment of Textual Credibility Signals in the Era of Large Language Models (https://arxiv.org/abs/2410.21360)
- **What's New**: 본 논문은 자동 신뢰도 평가에 관한 시스템적이고 포괄적인 문헌 검토를 제공하며, 텍스트 신뢰도 신호와 자연어 처리(NLP)에 중점을 두고 있습니다. 특히, 대형 언어 모델(LLMs)의 발전에 따른 신뢰도 평가 접근법을 탐구합니다.

- **Technical Details**: 저자는 175개의 연구 논문을 분석하고, 9가지 신뢰도 신호 범주를 제시합니다. 그 중 3개의 주요 범주(사실성, 주관성 및 편향; 설득 기술 및 논리적 오류; 주장 및 진위)에 대해 심층 분석을 수행합니다. 신뢰도 신호는 AI를 통해 자동으로 탐지될 수 있는 텍스트에 관련된 신호들입니다. 이 연구는 신뢰도 평가 알고리즘을 통해 신뢰도 점수로 집계됩니다.

- **Performance Highlights**: 이 논문은 자동 신뢰도 평가 및 신뢰도 신호 탐지의 현재 연구 동향 및 발전을 반영하며, 신뢰도 신호의 해석 가능성과 사용자 중심 AI 접근 방식을 지원하는 가치가 있음을 강조합니다. 저자는 다양한 분야의 연구에서 채택된 신뢰도 신호의 다각적인 활용 가능성을 논의합니다.



### Can Machines Think Like Humans? A Behavioral Evaluation of LLM-Agents in Dictator Games (https://arxiv.org/abs/2410.21359)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)을 기반으로 한 에이전트의 친사회적 행동을 유도하는 다양한 페르소나(persona)와 실험적 환경의 영향을 조사하였습니다. 또한, LLM의 복잡한 의사결정 시나리오에서 성과를 평가하기 위한 행동적 접근 방식을 제안하였습니다.

- **Technical Details**: 이 연구는 권위자의 게임(dicator game)을 기반으로 하여 LLM 에이전트의 이타적 행동을 평가했습니다. 다양한 LLM 가족 간과 인간 행동과 비교한 결과, LLM 사이에서 행동의 변동성과 불일치가 크게 나타났습니다. LLM에 인간과 유사한 정체성을 부여하는 것만으로는 인간과 유사한 행동을 나타내지 않았습니다.

- **Performance Highlights**: LLM 에이전트는 인간의 의사결정을 정확하게 예측하지 못하고, 이러한 일치도는 모델 아키텍처와 프롬프트 형식에 따라 달라지는 highly variable한 특성을 보였습니다. 이는 LLM의 복잡한 사회적 상호작용에서 그들의 행동을 평가하기 위한 더 깊은 이해와 견고한 방법론의 필요성을 강조합니다.



### Energy-Based Diffusion Language Models for Text Generation (https://arxiv.org/abs/2410.21357)
- **What's New**: 이 연구는 기존의 discrete diffusion models의 한계를 극복하기 위해 Energy-based Diffusion Language Model (EDLM)을 제안합니다. EDLM은 디퓨전 과정의 각 단계에서 전체 시퀀스를 동시에 디노이즈할 수 있도록 설계된 에너지 기반 모델입니다.

- **Technical Details**: EDLM은 잔여 형태(residual form)로 구성되어 있으며, pretrained autoregressive 모델의 매개변수를 활용하거나 noise contrastive estimation을 통해 bidirectional transformer를 fine-tuning하여 매개변수를 얻습니다. 이 모델은 기존 diffusion models에서 발생하는 훈련 및 샘플링 분포의 불일치를 해결합니다.

- **Performance Highlights**: EDLM은 언어 모델링 벤치마크에서 기존의 최첨단 diffusion 모델을 일관되게 초과 달성하며, autoregressive 모델과 유사한 perplexity를 접근합니다. 또한, 기존 디퓨전 모델보다 1.3배 빠른 샘플링 속도를 제공하면서도 생성 성능의 저하 없이 결과를 낼 수 있습니다.



### Causal Interventions on Causal Paths: Mapping GPT-2's Reasoning From Syntax to Semantics (https://arxiv.org/abs/2410.21353)
Comments:
          12 pages

- **What's New**: 이 연구는 변형자 기반의 LLMs(large language models)가 자연어에서 인과적 사고를 어떻게 처리하는지를 초기 단계로 분석합니다. 특히, 명확한 원인과 결과 문장들을 분석함으로써 LLM의 인과적 구조를 이해하고자 합니다.

- **Technical Details**: 연구진은 GPT-2 small 모델을 사용하여 명확한 인과 문장을 확인했습니다. 이 모델은 12개의 레이어를 가지고 있으며, 자기 주의(attention) 메커니즘을 포함합니다. 연구에서는 문장의 구문적 신호와 의미적 관계를 분석하기 위해 인과 문장을 포함한 합성 데이터 셋을 사용했습니다.

- **Performance Highlights**: 결과적으로, GPT-2 모델은 주로 첫 2-3 레이어에서 구문적 구조를 파악하며, 일부 주의 헤드는 비논리적인 변형에 대한 민감도를 높게 나타냈습니다. 이러한 발견은 모델이 신택틱 큐(syntactic cues)를 탐지하고 최종 레이어의 특정 헤드가 의미적 관계를 중시함을 시사합니다.



### LLMCBench: Benchmarking Large Language Model Compression for Efficient Deploymen (https://arxiv.org/abs/2410.21352)
- **What's New**: 대형 언어 모델(LLMs)의 효율성을 높이기 위해 여러 모델 압축 기술이 제안되었으나, 이번 연구는 다양한 가정 하에 고도화된 압축 알고리즘의 효과를 평가할 수 있는 새로운 벤치마크인 LLMCBench를 발표하였다.

- **Technical Details**: LLMCBench는 LLM 압축 알고리즘의 깊이 있는 분석을 제공하며, 6개의 평가 트랙과 11개의 데이터셋, 18개의 네트워크 아키텍처, 3개의 배포 플랫폼에서 실험을 수행한다. 주요 압축 기술로는 sparsification(희소화), quantization(양자화), knowledge distillation(지식 증류) 등이 포함된다.

- **Performance Highlights**: LLMCBench는 기존 연구에서의 평가 한계를 극복하고, 모델 압축 알고리즘에 대한 통합적인 비교를 제공함으로써 경량화된 LLM의 실제 운용 가능성을 높일 수 있는 통찰력을 제공한다.



### Large Language Model Benchmarks in Medical Tasks (https://arxiv.org/abs/2410.21348)
Comments:
          25 pages, 5 tables

- **What's New**: 이 논문은 의료 분야에서 대규모 언어 모델(LLMs)의 성능 평가를 위한 다양한 벤치마크 데이터세트에 대한 포괄적인 조사를 제공합니다. 특히, 전자 건강 기록(EHR), 의사-환자 대화, 의료 질문-응답 및 의료 이미지 캡션 생성을 포함한 여러 모달리티를 다룹니다.

- **Technical Details**: 논문에서는 MIMIC-III, MIMIC-IV, BioASQ, PubMedQA, CheXpert와 같은 주요 벤치마크 데이터세트를 소개하며, 이는 의료 보고서 생성, 임상 요약 및 합성 데이터 생성과 같은 작업의 발전을 촉진시켰습니다. 데이터세트의 카테고리는 텍스트, 이미지 및 다중 모달리티에 따라 나뉘며, 각 데이터단의 구조와 임상 작업에 미치는 영향을 설명합니다.

- **Performance Highlights**: LLMs의 발전에 따라, 환자의 건강 관리 및 의사결정 지원을 최적화하기 위한 의학적 데이터 세트의 중요성이 강조되며, 데이터의 언어 다양성과 구조화된 오믹스 데이터의 필요성이 제기됩니다. 또한 이러한 벤치마크를 활용하여 다중모달 의료 지능을 발전시키기 위한 도전과 기회도 요약했습니다.



### FinTeamExperts: Role Specialized MOEs For Financial Analysis (https://arxiv.org/abs/2410.21338)
- **What's New**: 완전한 LLM 적용을 위한 금융 전문 모델, FinTeamExperts 소개

- **Technical Details**: FinTeamExperts는 Mixture of Experts (MOEs) 구조를 가진 LLM 프레임워크로, 각 모델은 매크로(Macro), 마이크로(Micro), 정량(Quantitative) 분석에 특화되어 있음. 세 가지의 80억 파라미터 모델이 각기 특화된 데이터를 기반으로 훈련됨. 두 단계의 훈련 과정으로 먼저 역할 기반 데이터로 사전 훈련한 후, 실질적인 금융 작업과 기대에 맞춰 추가 조정(instruct-tuning) 진행.

- **Performance Highlights**: FinTeamExperts는 4개의 데이터셋에서 동일한 규模 및 더 큰 모델들보다 우수한 성능을 발휘함. 특히, 복잡한 작업을 요구하는 데이터셋에서도 각각의 역할 기반 전문가 모델의 효용성을 강조함.



### Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection (https://arxiv.org/abs/2410.21337)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 보안 취약성 중 하나인 프롬프트 인젝션(prompt injection) 공격을 탐구합니다. 이는 LLM을 사용한 응용 프로그램이 신뢰할 수 없는 입력 프롬프트에 의해 조작될 수 있다는 문제를 다루고 있습니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식을 사용하여 프롬프트의 취약성을 탐지합니다: 1) 사전 훈련된 LLM, 2) 파인 튜닝(fine-tuning)된 LLM. 사전 훈련된 XLM-RoBERTa 모델을 사용하여 초기 테스트 데이터셋에서 제로샷 분류(zero-shot classification)를 통해 프롬프트 인젝션을 탐지하였고, 그 후 특정 태스크에 맞춘 레이블이 지정된 데이터셋을 활용하여 파인 튜닝을 실시했습니다.

- **Performance Highlights**: 파인 튜닝된 모델은 99.13%의 정확도, 100%의 정밀도, 98.33%의 재현율(recall), 그리고 99.15%의 F1-score를 기록하며 뛰어난 성과를 보였습니다.



### LLM Robustness Against Misinformation in Biomedical Question Answering (https://arxiv.org/abs/2410.21330)
- **What's New**: 이 논문에서는 바이오메디컬 질문에 대한 답변 정확도를 평가하기 위해 Gemma 2, GPT-4o-mini, Llama 3.1, Mixtral 네 가지 대형 언어 모델(LLMs)을 비교했습니다. 특히, 잘못된 정보를 주입하여 생성된 답변이 얼마나 정확한지를 분석하는 데 중점을 두었습니다.

- **Technical Details**: 우리는 세 가지 시나리오에서 LLM의 질문 응답 정확도를 평가했습니다: (1) 기본 LLM 답변 (맥락 없음), (2) '완벽한' 보강 생성 (정확한 맥락 제공), (3) 프롬프트 주입 공격 (잘못된 맥락 제공). Llama 3.1 모델은 기본 및 '완벽한' RAG 시나리오에서 가장 높은 정확도를 기록했습니다.

- **Performance Highlights**: Llama 3.1(70B 파라미터)은 기본 답변에서 0.651의 정확도와 '완벽한' RAG에서 0.802의 정확도를 기록했습니다. 그러나 '완벽한' RAG에서 모델 간의 정확도 차이가 거의 사라져 LLM의 크기와 관련된 효과 차이를 완화할 가능성을 보여줍니다. Llama는 최대 0.48의 정확도 하락을 초래하여 가장 효과적인 공격자로 나타났습니다.



### Mathematical Derivation Graphs: A Task for Summarizing Equation Dependencies in STEM Manuscripts (https://arxiv.org/abs/2410.21324)
Comments:
          10 pages, 4 figures

- **What's New**: 이 연구는 STEM(Science, Technology, Engineering, Mathematics) 분야의 학술 문서에서 수학적 표현 간의 의존 관계를 이해하기 위한 최초의 단계로, 새로운 객체인 'derivation graph'를 제안합니다. 이 그래프는 논문의 수학 내용을 요약화합니다.

- **Technical Details**: 연구자는 arXiv 데이터베이스에서 수집한 107개의 STEM 논문을 기반으로 수학적 유도 그래프를 수작업으로 레이블링했습니다. 다양한 알고리즘을 사용하여 이 그래프를 자동으로 추출하는 방법을 평가했습니다. 실험적으로, 분석적 및 NLP 모델(LLM 포함)을 평가하여 각각의 문서에서 유도 그래프를 추출하는 능력을 비교했습니다.

- **Performance Highlights**: 연구 결과, 분석적 모델과 NLP 모델 모두 약 40-50%의 F1 점수를 기록하여, 최신 NLP 기법이 단순 분석 모델에 비해 수학적 텍스트를 이해하는데 큰 진전을 이루지 못했음을 보여주었습니다. 따라서 보다 정확하고 심도 있는 수학적 정보 처리를 위한 추가 연구가 필요합니다.



### GraphLSS: Integrating Lexical, Structural, and Semantic Features for Long Document Extractive Summarization (https://arxiv.org/abs/2410.21315)
Comments:
          Short paper submitted to ACL ARR November cycle

- **What's New**: 본 논문에서는 GraphLSS라는 이질적인 그래프 구조를 제안하여 긴 문서의 추출적 요약을 위한 새로운 접근 방식을 소개합니다. 이 구조는 Lexical, Structural, Semantic 특징을 포괄하며, 외부 학습 모델 없이도 문서를 노드(문장과 단어)와 엣지(문장 의미 유사성, 문장 발생 순서 등)로 구성합니다.

- **Technical Details**: GraphLSS는 두 가지 노드(문장 및 단어)와 네 가지 엣지(문장 간 유사성, 순서, 단어 간 유사성 등)를 정의합니다. 이 모델은 GAT(Veličković et al., 2018)와 같은 그래프 신경망을 통해 처리됩니다. 또한, 무거운 교차 엔트로피 손실(weighted cross-entropy loss)을 최적화하여 불균형한 장문서에 대한 추출적 진실 레이블을 생성합니다.

- **Performance Highlights**: GraphLSS는 PubMed와 arXiv 두 개의 벤치마크 데이터셋에서 실험을 수행하였으며, 최근 비그래프 모델을 능가하는 성과를 기록하였습니다. 또한, GitHub에 코드를 공개하여 재현성과 협업을 지원합니다.



### Decoding Diffusion: A Scalable Framework for Unsupervised Analysis of Latent Space Biases and Representations Using Natural Language Prompts (https://arxiv.org/abs/2410.21314)
- **What's New**: 최근의 이미지 생성 알고리즘 발전으로 인해 순방향 확산 모델(Diffusion Models)이 고 품질의 이미지를 생성하는 강력한 도구로 자리잡았습니다. 본 논문은 이러한 모델의 의미론적 잠재 공간(semantic latent space)을 자동으로 탐색할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 자연어 프롬프트와 이미지 캡션을 직접 활용하여 잠재 방향(latent direction)을 매핑합니다. 이 접근법은 특정 벡터를 훈련할 필요 없이, 의미론적 정보(semantic information)를 활용하여 잠재 공간의 자동 이해를 가능하게 합니다. 또한, Latent Consistency Model(LCM)을 활용하여 중간 U-Net 레이어의 출력인 h-space를 샘플링합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 도메인에서 숨겨진 패턴과 연관성을 발견하며, 순방향 확산 모델의 의미론적 지식(semantic knowledge)과 편향(bias)을 분석하는 데 뛰어난 성능을 보였습니다. 이로 인해 해석할 수 있는 더욱 투명한 생성 모델을 위한 가능성을 열었습니다.



### Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models, and Challenges (https://arxiv.org/abs/2410.21306)
Comments:
          35 pages

- **What's New**: 이번 설문조사는 법률 분야에서의 자연어 처리(Natural Language Processing, NLP) 기술의 현재 응용 현황을 종합적으로 조사합니다. 148개의 연구를 검토하여 127개의 연구를 최종 선정하여 법률 텍스트 처리와 관련된 독특한 측면과 도전 과제를 조명합니다.

- **Technical Details**: 자연어 처리의 다양한 법률 응용 프로그램, 예를 들어 법률 문서 요약(Legal Document Summarization, LDS), 법률 개체 인식(Named Entity Recognition, NER), 법률 질의 응답(Legal Question Answering, LQA), 법률 텍스트 분류(Legal Text Classification, LTC), 법률 판결 예측(Legal Judgment Prediction, LJP) 등을 다룹니다. 이 설문조사는 15개의 열려 있는 연구 문제가 포함되어 있으며, 인공지능(AI) 응용 프로그램의 편향, 강력하고 해석 가능한 모델의 필요성, 법률 언어의 복잡성을 처리하기 위한 설명 가능성 개선 등을 포함합니다.

- **Performance Highlights**: NLP의 발전으로 법률 분야에서 문서 요약, 질의 응답 및 판결 예측과 같은 복잡한 작업이 단순화되었으며, 변호사와 법률 전문가의 업무 효율성을 높이고 오류를 최소화하는 데 기여하고 있습니다. 그러나 긴 문서를 처리하고 복잡한 언어를 이해하는 데 여전히 어려움이 존재하며, 대형 언어 모델(Large Language Models, LLMs)의 출현이 법률 서비스의 효율성과 접근성을 높일 수 있는 잠재력을 보유하고 있습니다.



### Task Vectors are Cross-Moda (https://arxiv.org/abs/2410.22330)
- **What's New**: 이번 연구에서는 시각-언어 모델(VLMs)의 내부 표현과 이들이 과제 표현(task representations)을 어떻게 인코딩하는지를 조사합니다. 새로운 점은, 유사한 개념의 과제가 어떻게 명세되었는지에 관계없이 유사한 과제 벡터 표현(task vector representations)으로 매핑된다는 것입니다.

- **Technical Details**: 우리는 예시나 지시를 통해 지정된 과제를 고려하며, 텍스트 또는 이미지 입력을 사용합니다. VLM에서 발견된 과제 벡터는 한 모달리티(예: 텍스트)에서 파생되어 다른 모달리티(예: 이미지)로 전이될 수 있을 만큼 일반적입니다. 또한, 예시 기반(exemplar)과 지시 기반(instruction based)의 과제 벡터를 집계(ensembling)하면 더 나은 과제 표현을 제공합니다.

- **Performance Highlights**: 이 연구 결과는 VLM의 설계 메커니즘 및 서로 다른 모달리티와 과제 명세를 통해 과제를 공유하는 방식에 대한 통찰을 제공합니다. VLM이 입력(input), 과제(task), 응답(answer)의 세 가지 독특한 단계로 토큰을 처리하는 과정을 밝혀냈습니다.



### SVIP: Towards Verifiable Inference of Open-source Large Language Models (https://arxiv.org/abs/2410.22307)
Comments:
          20 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 검증 가능한 추론 문제를 형식화하고, 이용자와 컴퓨팅 제공자 간의 신뢰성을 높이기 위해 중간 출력을 활용한 비밀 기반의 검증 가능한 LLM 추론 프로토콜(SVIP)을 제안합니다.

- **Technical Details**: SVIP는 LLM의 중간 출력(hidden state representations)을 고유한 모델 식별자로 사용하여, 컴퓨팅 제공자가 생성한 출력이 실제 요청된 LLM에서 온 것인지 검증할 수 있는 mecanismos입니다. 이 방법은 컴퓨팅 제공자에게 생성된 텍스트와 프로세스된 중간 출력을 반환하도록 요구하여, 사용자에게 신뢰할 수 있는 검증 수단을 제공합니다.

- **Performance Highlights**: SVIP는 평균 3.49%의 잘못된 부정률(false negative rate) 및 3% 미만의 잘못된 긍정률(false positive rate)을 유지하며, 한 쿼리에 대해 0.01초 이하의 검증 시간을 요구합니다. 본 프로토콜은 약 80에서 120만 건의 프롬프트 쿼리를 안전하게 처리할 수 있도록 설계되었습니다.



### Fourier Head: Helping Large Language Models Learn Complex Probability Distributions (https://arxiv.org/abs/2410.22269)
Comments:
          Project page and code are at this https URL

- **What's New**: 본 논문에서는 연속적인 구조를 고려하여, 기존 LLM(linear language model) 구조를 개선하기 위해 푸리에 변환(Fourier transformation) 기반의 새로운 신경망 레이어인 푸리에 헤드(Fourier head)를 제안합니다. 이 레이어는 고질적 소음(high-frequency noise)을 무시하면서 데이터를 통해 신호(signal)를 더 잘 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 푸리에 헤드는 입력 벡터를 받아 신경망의 선형 계층(linear layer)을 통해 푸리에 계수를 학습하고, 이를 통해 연속적인 확률 밀도 함수(continuous probability density function)를 학습합니다. 입력값은 [-1, 1] 범위에서 m개의 동등한 구간으로 양자화되며, 각 구간에서의 확률을 카테고리 분포로 반환합니다.

- **Performance Highlights**: 푸리에 헤드는 Atari Seaquest 게임에서 결정 트랜스포머(Decision Transformer)의 성과를 46% 향상시켰으며, 20개의 비훈련 데이터셋에서 3.5%의 예측 개선을 보여주어 최신 시간 시계열 모델의 성능을 초월했습니다.



### Cora: Accelerating Stateful Network Applications with SmartNICs (https://arxiv.org/abs/2410.22229)
- **What's New**: Cora는 SmartNICs에 상태 기반 네트워크 애플리케이션을 오프로드하는 새로운 컴파일러와 런타임 시스템을 제안합니다. 기존의 솔루션과 비교할 때 더 효율적인 오프로드 계획을 제시하며 CPU 사용량을 크게 줄일 수 있습니다.

- **Technical Details**: Cora 컴파일러는 각 SmartNIC에 대한 정확한 성능 모델을 도입하고 효율적인 컴파일 알고리즘을 통해 오프로드 계획을 탐색합니다. Cora 런타임은 트래픽 동태를 모니터링하고 적응하여 CPU 사용량을 최소화합니다. Netronome Agilio와 BlueField 2 SmartNICs 위에서 구축되었습니다.

- **Performance Highlights**: Cora는 동일한 처리량 목표에 대해 CPU 코어를 최대 94.0% 절약할 수 있으며, 이는 기준 솔루션보다 1.9배 향상된 결과입니다. 동일한 자원 제약 하에서도 Cora는 네트워크 기능을 44.9%에서 82.3%까지 가속화할 수 있습니다. 또한 Cora 런타임은 트래픽 변화에 적응하며 낮은 CPU 사용량을 유지합니다.



### ADAM: An Embodied Causal Agent in Open-World Environments (https://arxiv.org/abs/2410.22194)
- **What's New**: 본 논문에서는 Minecraft 환경에서 자율적으로 탐색하고, 다중 모달 컨텍스트를 인식하며, 인과 세계 지식을 학습하여 복잡한 작업을 해결할 수 있는 ADAM(An emboDied causal Agent in Minecraft)을 소개합니다. ADAM은 기존 모델의 한계를 극복하고 더 나은 해석 가능성과 일반화 능력을 제공합니다.

- **Technical Details**: ADAM은 네 가지 주요 모듈로 구성됩니다: 1) Interaction Module - 에이전트가 행동을 실행하고 상호작용 과정을 문서화합니다. 2) Causal Model Module - 인과 그래프를 구성하여 해석 가능성을 높이고 사전 지식에 대한 의존도를 줄입니다. 3) Controller Module - planner, actor 및 memory pool로 구성되어, 학습한 인과 그래프를 기반으로 작업을 수행합니다. 4) Perception Module - 다중 모달 대형 언어 모델(MLLMs)을 사용하여 ADAM이 인간 플레이어처럼 주변을 인식하도록 합니다.

- **Performance Highlights**: ADAM은 기존 최첨단(SOTA) 방법과 비교하여 다이아몬드 획득 작업에서 2.2배의 속도를 보였으며, 공예 레시피가 수정된 경우에도 안정적으로 다이아몬드를 획득하는 유일한 방법으로, 4.6배의 속도를 기록했습니다. ADAM은 거의 완벽한 기술 트리를 구성하여 해석 가능성을 높이고, 인간 플레이 방식에 근접한 성능을 유지했습니다.



### RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier (https://arxiv.org/abs/2410.22124)
Comments:
          Accepted at NeurIPS 2024 (Poster)

- **What's New**: 이번 논문에서는 RushUp이라는 새로운 반감독 학습 프레임워크를 제안합니다. 이는 기존의 반감독 분류 기법을 회귀 문제에 적용하여 성능을 향상시키기 위해 개발되었습니다. RankUp은 원래의 회귀 문제를 순위 문제로 변환하여 보조 순위 분류기를 동시에 훈련시킴으로써 성능을 높입니다.

- **Technical Details**: RankUp은 보조 순위 분류기를 사용하여 원래의 회귀 작업과 함께 순위 작업을 동시에 해결합니다. 이를 통해 기존의 반감독 분류 기법을 활용하여 훈련할 수 있으며, 새로운 방법인 Regression Distribution Alignment (RDA)를 통해 의사 라벨의 품질을 더욱 개선합니다.

- **Performance Highlights**: RankUp은 RDA 없이도 다양한 회귀 벤치마크에서 SOTA 결과를 달성했습니다. 예를 들어, RankUp은 UTKFace 데이터셋에서 50개의 레이블 샘플을 사용하여 MAE에서 최소 13% 개선을 보여주었으며, RDA 통합 시 추가적인 6% 개선을 기록했습니다.



### Unlearning as multi-task optimization: A normalized gradient difference approach with an adaptive learning ra (https://arxiv.org/abs/2410.22086)
- **What's New**: 이 논문에서는 기계 잊기를 최적화 관점에서 조사하고, 망각(memorization) 목표 최적화와 모델 성능을 최적화하는 것을 다중 작업 최적화(multi-task optimization) 문제로 공식화합니다. 특히, 우리는 목표 간의 균형 조절을 향상시키기 위한 새로운 정규화된 그래디언트 차이(NGDiff) 알고리즘을 도입했습니다.

- **Technical Details**: 기계 잊기를 다중 작업 최적화 문제로 공식화하여 두 가지 목표를 동시에 최소화 및 최대화하는 방법을론의 명확화와 NGDiff 알고리즘을 통해 제시합니다. 이 알고리즘은 전달(set)과 망각(forgetting) 작업 간의 균형을 동적으로 조절하며, Hessian 기반 학습률 선택을 통합하여 안정적인 수렴(convergence)을 이룰 수 있습니다.

- **Performance Highlights**: NGDiff는 TOFU 및 MUSE 데이터셋에서 최첨단(unlearning) 방법들보다 뛰어난 성능을 보여주며, Llama2-7B 모델과 비교했을 때 모델 유틸리티를 40% 향상시켰습니다. 또한 안정적인 학습을 유지하는 특징이 있습니다.



### An Actor-Critic Approach to Boosting Text-to-SQL Large Language Mod (https://arxiv.org/abs/2410.22082)
- **What's New**: 이 논문에서는 Actor-Critic (AC) 접근 방식을 제안하여 LLM 기반 Text-to-SQL (T2S) 성능을 개선하였습니다. 이 접근 방식은 두 개의 역할을 포함하며, 둘 다 동일한 LLM을 사용하여 SQL 쿼리를 생성하고 평가합니다. Critic이 SQL의 오류를 평가하면 Actor에게 다시 생성하도록 지시합니다. 이는 T2S 전환을 위한 일반적인 개선 접근 방식으로 작용합니다.

- **Technical Details**: AC-SQL 접근 방식은 두 개의 에이전트인 Actor와 Critic을 사용하며, 각각 SQL 후보를 생성하고 평가하는 기능을 수행합니다. 이 과정은 평과 기준인 Execution Critic과 LLM Critic에 따라 수행됩니다. 이 반복 과정은 솔루션의 성능을 이론적으로 유도할 수 있는 가능성을 제공합니다. 이 연구는 Spider 데이터셋을 사용하여 AC-SQL 방법에 대한 고광범위한 실험을 수행하였다.

- **Performance Highlights**: AC-SQL 방법은 다양한 오픈소스 모델에 대해 일관되게 성능을 향상시켰으며, 상업적 모델인 GPT-3.5 및 GPT-4o에서도 효과적인 접근 방식임을 입증하였습니다. 또한 이 방법은 일반적인 적용 가능성을 띠며, 기대 성능이 이론적으로 보장된다는 점에서 중요한 기여를 하고 있습니다.



### A Longitudinal Analysis of Racial and Gender Bias in New York Times and Fox News Images and Articles (https://arxiv.org/abs/2410.21898)
Comments:
          13 pages, and 11 figures

- **What's New**: 본 연구는 온라인 뉴스 기사에서 다양한 인종 및 성별 집단의 등장 빈도와 그 맥락을 분석하는 최초의 종단적(longitudinal) 연구로, 머신러닝(classifier)을 활용하여 인종과 나이를 감지하는 두 가지 새로운 모델을 제안합니다.

- **Technical Details**: 연구는 123,337개의 이미지와 441,321개의 온라인 뉴스 기사를 뉴욕 타임스(New York Times)와 폭스 뉴스(Fox News)에서 수집하여 분석합니다. 연구에 사용된 머신러닝 모델은 최신 기술을 도입하여 인종 및 나이를 정확하게 분류하며, 텍스트 기반의 분석 방법과 결합하여 이중으로 인종 및 성별의 표현을 다룹니다.

- **Performance Highlights**: 결과적으로, 인종 및 성별 소수자 집단은 전체적으로 나타나는 빈도가 매우 낮았으며, 나타날 때에도 남자 집단에 비해 상대적으로 덜 두드러지게 표현되었습니다. 뉴욕 타임스는 폭스 뉴스에 비해 소수자 집단의 이미지를 더 자주 포함하고 있으며, 감정적 측면에서 보았을 때 두 언론은 담아내는 감정의 차이가 있음을 알 수 있습니다.



### Evaluating K-Fold Cross Validation for Transformer Based Symbolic Regression Models (https://arxiv.org/abs/2410.21896)
- **What's New**: 본 연구에서는 Transformer 기반의 Symbolic Regression 모델이 작은 데이터셋(15,000 데이터 포인트)을 활용하여 훈련되었을 때의 K-fold cross-validation을 적용하여 모델의 일반화 가능성 및 과적합 문제를 완화하는 방법을 제안합니다.

- **Technical Details**: K-Fold Cross-Validation (KFCV)을 통해 5개의 fold로 데이터셋을 나누어 각 fold마다 모델을 훈련하고 검증하여 평균 훈련 손실과 검증 손실을 산출했습니다. 이 모델의 경우, 기존 500,000 포인트 데이터셋에서 97% 줄어든 15,000 포인트로 훈련되었습니다.

- **Performance Highlights**: KFCV 기법을 적용한 결과, 모델의 검증 손실이 53.31% 개선되었으며, 이는 모델의 출력 일관성과 일반화를 현저히 향상시켰음을 보여줍니다.



### Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Models (https://arxiv.org/abs/2410.21815)
- **What's New**: 본 논문에서는 self-interpretable 모델과 black-box 모델을 위한 post-hoc 설명 사이의 간극을 메우기 위한 새로운 방법인 *AutoGnothi*를 제안합니다. 이 방법은 블랙박스 모델이 예측 정확성을 손상시키지 않으면서 자가 해석 가능성을 이론적으로 보장하는 것을 목표로 합니다.

- **Technical Details**: *AutoGnothi*는 parameter-efficient transfer learning (PETL)을 활용하여 블랙박스 모델에 작은 사이드 네트워크를 통합합니다. 이를 통해 Shapley value 기반의 설명을 생성할 수 있으며, 원래 네트워크의 매개변수를 변경하지 않고도 메모리, 훈련 및 추론 비용을 크게 줄일 수 있습니다. 이 방법은 두 개의 모델로 구성된 기존 방식과 달리 예측과 설명을 동시에 수행하며, 단일 추론 단계에서 작업을 완료합니다.

- **Performance Highlights**: 실험 결과 *AutoGnothi*는 매우 높은 훈련 및 추론 효율성을 보여 주었습니다. 예를 들어, ViT-base 모델에 대해 97%의 훈련 가능 매개변수 감소, 72%의 훈련 메모리 감소를 기록하며, 설명 생성에 있어서 54%의 추론 비용 감소와 44%의 추론 시간 감소를 달성했습니다. 이러한 결과는 *AutoGnothi*가 블랙박스 모델의 예측 성능을 유지하면서 자기 해석 가능성을 강화한다는 것을 일증합니다.



### MARCO: Multi-Agent Real-time Chat Orchestration (https://arxiv.org/abs/2410.21784)
Comments:
          EMNLP 2024 Industry Track

- **What's New**: MARCO는 LLM을 이용한 다중 에이전트 실시간 채팅 조정 프레임워크로, 복잡한 작업을 자동화하는 새로운 접근 방식을 제시합니다. 특히 여러 도구와의 상호작용 및 다중 단계의 작업 실행을 지원하여 실시간으로 사용자와 상호작용할 수 있는 기능을 탑재하고 있습니다.

- **Technical Details**: MARCO는 LLM의 불확실성과 오류를 극복하기 위해 견고한 가드레일(guardrails)을 도입하였으며, 신뢰성 높은 작업 실행을 위해 자연어로 표현된 작업 실행 절차(TEP)를 사용합니다. 에이전트 간 장기 메모리를 공유하는 구조로 설계되어, 동적인 정보와 대화 상황을 포함한 완전한 컨텍스트 정보를 저장합니다. 다양한 LLM 모델을 비교하여 최적의 성능을 도출할 수 있도록 구성되어 있습니다.

- **Performance Highlights**: 테스트 결과, MARCO는 Digital Restaurant Service Platform 대화 및 Retail 대화 데이터셋에서 각각 94.48%와 92.74%의 높은 정확도를 기록했으며, 44.91%의 지연 개선(latency improvement)과 33.71%의 비용 감소(cost reduction)를 달성했습니다. 가드레일의 효과를 통해 성능 향상이 이루어졌으며, 여러 LLM 모델에 대한 비교 분석도 포함되어 있습니다.



### Sequential choice in ordered bundles (https://arxiv.org/abs/2410.21670)
- **What's New**: 본 논문에서는 경험 재화(Experience Goods) 구성을 위해 고안된 여러 예측 모델을 사용하여, 순서가 정해진 번들에서 개인이 다음 항목을 소비할 가능성을 예측하는 방법을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 커스텀 Transformers(디코더 전용 및 인코더-디코더 아키텍처), 미세 조정된 GPT-3, 커스텀 LSTM 모델, 강화 학습 모델, 두 개의 Markov 모델 그리고 제로 순서 모델을 평가합니다. 주목할 점은 Transformers의 주의(attention) 가중치를 분석하여 소비자가 이전 선택을 기반으로 다음 항목을 고를 때의 패턴을 보여줍니다.

- **Performance Highlights**: Spotify 데이터 분석 결과, 디코더 전용 아키텍처를 가진 커스텀 Transformer가 개별 선택과 총 수요 예측에서 가장 높은 정확도를 달성합니다. 또한, 이 모델은 개인 맞춤형 프로모션을 통해 수요를 증가시키는 데 도움을 줄 수 있습니다.



### Can Language Models Replace Programmers? REPOCOD Says 'Not Yet' (https://arxiv.org/abs/2410.21647)
- **What's New**: RL과 DNN의 조합, 특정 영역에 대한 성능 향상 보장 및 새로운 코드 생성 기준 REPOCOD 제시.

- **Technical Details**: REPOCOD는 11개의 인기 있는 실세계 프로젝트에서 수집된 980개의 코드 생성 문제로 구성됨. 이 중 58% 이상은 파일 수준이나 레포 수준의 맥락 정보가 필요함. 평균 canonical solution 길이는 331.6 tokens, cyclomatic complexity는 9.00으로 기존 벤치마크보다 복잡도가 높음.

- **Performance Highlights**: 10개의 LLM을 평가한 결과, 어떤 모델도 REPOCOD에서 30 pass@1 이상의 성능을 보이지 않음. 이는 LLM이 복잡한 문제를 해결하는 데 한계를 나타내며, 개발자들을 돕기 위한 더 강력한 LLM의 필요성을 강조함.



### Can Large Language Models Replace Data Scientists in Clinical Research? (https://arxiv.org/abs/2410.21591)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 활용하여 임상 연구의 데이터 과학 태스크를 자동화하는 가능성을 평가하는 새로운 데이터 셋을 개발했습니다. 이 데이터 셋은 39개의 임상 연구를 기반으로 하여 293개의 실제 데이터 과학 코딩 작업으로 구성되어 있습니다.

- **Technical Details**: 데이터 셋은 Python과 R에서 128개의 작업과 165개의 작업을 포함하며, 임상 데이터 분석을 위한 실제 시나리오를 시뮬레이션합니다. 연구에서 확인된 고급 적응 방법으로는 chain-of-thought prompting과 self-reflection이 있으며, 각각 코드 정확성을 60%와 38% 향상시켰습니다.

- **Performance Highlights**: 사용자 연구 결과, 의료 전문가들이 제출한 코드 솔루션의 80%가 LLM이 생성한 코드에서 가져온 것이었고, 일부 경우 최대 96%의 재사용이 발견되었습니다. 이 결과는 LLMs가 전문가 워크플로우와 통합될 때 데이터 과학 효율성을 향상시키는 잠재력을 가진다는 것을 강조합니다.



### Semantic Search Evaluation (https://arxiv.org/abs/2410.21549)
Comments:
          Accepted by 3rd International Workshop on Industrial Recommendation Systems (at CIKM 2024)

- **What's New**: 이 논문은 콘텐츠 검색 시스템의 성능을 평가하는 새로운 방법을 제안합니다. 이 방법은 쿼리와 검색 결과 간의 의미적 일치를 측정합니다. 새로운 메트릭인 'on-topic rate'를 도입하여 쿼리와 관련된 결과의 비율을 측정합니다.

- **Technical Details**: 제안된 평가 파이프라인은 금색 쿼리 세트를 정의하고 각 쿼리에 대해 상위 K 개의 결과를 검색하며, GPT 3.5와의 프롬프트를 통해 평가합니다. 이 과정은 쿼리와 문서 간의 관련성을 판단하기 위해 Generative AI(GAI)를 활용합니다. OTR(온-토픽 레이트)은 검색 결과가 쿼리와 얼마나 관련성이 있는지를 측정합니다.

- **Performance Highlights**: OTR을 통해 검색 엔진의 성능을 평가하고 개선할 수 있는 패턴을 확인할 수 있습니다. 특히 OTR@K 메트릭을 통해 모델의 성능을 평가할 수 있으며, 이는 검색 엔진의 품질을 지속적으로 측정할 수 있는 도구로서 구실합니다.



### L3Ms -- Lagrange Large Language Models (https://arxiv.org/abs/2410.21533)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 감독하에 세밀한 조정(Supervised Fine-Tuning, SFT)과 정렬(Alignment)을 제약 최적화 문제로 공식화하였습니다. 이는 애플리케이션의 특정 요구 사항을 충족하도록 하여 휴리스틱(Heuristic) 방법에 의한 최적화를 피할 수 있게 합니다.

- **Technical Details**: 제안된 Lagrange Large Language Models (L3Ms)는 로그 장벽(Logarithmic Barriers)을 사용하여 제약을 시행합니다. 이를 통해 L3Ms는 다양한 애플리케이션에 맞춰 커스터마이징 가능하며, 기존의 단계적 접근 방식에서 벗어나 SFT와 정렬 단계를 통합합니다.

- **Performance Highlights**: 실험을 통해 L3Ms가 다양한 애플리케이션에서 사용자 지정 요구 사항을 충족하는 데 어떻게 유용한지를 보여줍니다. 예를 들어, 두 가지 L3Ms를 사용하여 동일한 지시를 따르는 작업을 수행하되, 하나는 간결하도록 제약하고 다른 하나는 장황하도록 제약하는 결과를 비교하였습니다.



### Not All LLM-Generated Data Are Equal: Rethinking Data Weighting in Text Classification (https://arxiv.org/abs/2410.21526)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문은 기존의 LLM(대형 언어 모델)에서 생성된 합성 데이터를 실제 분포와 정렬시키기 위한 효율적인 가중 손실(Weighted-loss) 접근 방식을 제안합니다. 특히, 고품질 및 다양한 데이터를 강조하는 방법을 통해 적은 양의 실제 데이터를 사용하여 모델 성능을 향상시킵니다.

- **Technical Details**: 저자들은 IMP-Loss와 DIMP-Loss라는 두 가지 새로운 자동 가중 손실 접근 방식을 소개합니다. 이들 방법은 합성 데이터와 실제 데이터의 분포를 밀접하게 정렬하고, 높은 품질과 다양성을 가진 데이터 포인트에 더 높은 가중치를 부여합니다. 이를 통해 불필요한 데이터의 영향을 줄이고 모델 성능을 개선합니다.

- **Performance Highlights**: 다양한 텍스트 분류 작업을 통해 이 방법이 BERT 수준의 모델에서 표준 크로스 엔트로피(Cross-entropy)와 다른 데이터 가중치 접근 방식보다 효과적임을 입증하였습니다. 특히 DIMP-Loss는 모델 크기, 데이터 요구 사항 및 계산 자원 측면에서 효율적이며, 합성 데이터에서 실제 데이터를 효과적으로 활용할 수 있는 잠재력을 제공합니다.



### LLM-Forest for Health Tabular Data Imputation (https://arxiv.org/abs/2410.21520)
- **What's New**: 이번 연구에서는 기존의 데이터 보간(method) 방식의 한계를 극복하기 위해 LLM-Forest라는 새로운 프레임워크를 제안합니다. 이는 few-shot learning을 활용한 LLM '나무'를 여러 개 결합하여 보다 robust한 결과를 제공할 수 있도록 설계되었습니다.

- **Technical Details**: LLM-Forest는 bipartite 정보 그래프(bipartite information graphs)를 기반으로 하여 고품질의 이웃 데이터를 식별하는 그래프 알고리즘을 사용합니다. 이 알고리즘은 feature와 value 수준에서의 granularity를 통해 데이터의 복잡한 관계를 포착하며, O(n)으로 계산 복잡도를 줄일 수 있습니다. 여러 LLM '나무'로 구성된 이 프레임워크는 ensemble learning에 기반하여 hallucination(환각) 문제를 감소시킵니다.

- **Performance Highlights**: 실제 건강 데이터셋을 포함한 네 개의 데이터셋에 대한 실험을 통해 LLM-Forest의 효과성과 효율성을 입증하였습니다. 전통적인 방법들과 비교했을 때, LLM-Forest는 결측값 보간(imputation)에서 뛰어난 성능을 보이며, 실제 의료 데이터 분석에 대한 응용 가능성을 강조합니다.



### FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks (https://arxiv.org/abs/2410.21492)
- **What's New**: 이 논문은 FATH(Formatting AuThentication with Hash-based tags)라는 새로운 테스트 시점 방어 전략을 소개합니다. 이는 외부 텍스트 정보의 악성 지시문으로부터 LLM(대규모 언어 모델)을 보호하는 것을 목표로 합니다.

- **Technical Details**: FATH는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 보안 입력 형식화(Secure Input Formatting): 사용자의 지시문과 외부 데이터의 구분을 위한 동적 태그를 사용합니다; (2) 보안 정책에 따른 프롬프트(Prompting with Security Policy): LLM이 응답 내에서 비밀 인증 키를 생성하도록 합니다; (3) 인증 검증(Authentication Verification): LLM 출력에서 인증 키를 추출하고 검증합니다.

- **Performance Highlights**: FATH 방어 방법은 GPT3.5와 Llama3 모델 모두에서 다양한 공격 방식에 대해 공격 성공률(ASR)을 거의 0%로 감소시키며, 이전의 모든 방어 방법을 초월하는 성능을 보였습니다.



### AiSciVision: A Framework for Specializing Large Multimodal Models in Scientific Image Classification (https://arxiv.org/abs/2410.21480)
- **What's New**: AiSciVision은 과학 연구에서의 인공지능(AI) 사용을 위한 신뢰성과 해석 가능성을 제공하는 새로운 프레임워크입니다. 이 프레임워크는 대형 다중모달 모델(LMM)을 인터랙티브 연구 파트너로 변환하고 이미지 분류 작업을 위한 모델로 특화되었습니다.

- **Technical Details**: AiSciVision은 두 가지 주요 구성 요소로 이루어져 있습니다: (1) Visual Retrieval-Augmented Generation (VisRAG)과 (2) 도메인 특정 도구입니다. 이 시스템은 이미지 분류를 위해 유사한 이미지(양성 및 음성 레이블이 붙은)를 검색하고, LMM 에이전트가 이를 기반으로 도구를 선택하여 여러 번에 걸쳐 이미지를 조작하고 검토합니다.

- **Performance Highlights**: AiSciVision은 세 가지 실제 과학 이미지 분류 데이터셋에서 평가되었으며, 수조(aquaculture) 연못, 병든 이름풀이(eelgrass), 태양광 패널의 존재를 탐지하는 작업에서 완전 감독 모델보다 더 나은 성능을 보였습니다.



### ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inferenc (https://arxiv.org/abs/2410.21465)
- **What's New**: ShadowKV 시스템은 긴 문맥을 처리하는 LLM 추론 성능을 향상시키기 위해 저랭크( Low-rank ) 키 캐시를 저장하고 값 캐시( Value Cache )를 오프로드하는 방식을 도입합니다. 이를 통해 메모리 사용량을 줄이고 높은 배치 크기와 긴 시퀀스를 지원합니다.

- **Technical Details**: ShadowKV는 CUDA 멀티 스트림을 활용하여 선택된 키 캐시의 복구와 해당 값 캐시의 로드를 겹쳐서 수행하여 디코딩 지연을 최소화합니다. 또한, 사전 RoPE 키가 낮은 차원에 집중되어 제대로 된 KV 쌍을 동적으로 선택하여 정확성을 유지합니다.

- **Performance Highlights**: ShadowKV는 A100 GPU에서 최대 6배의 배치 크기를 지원하며, 최대 3.04배의 처리량을 증가시킵니다. Llama-3.1-8B 모델을 사용했을 때, 문맥 길이가 122K인 샘플에서 이 성능을 실현하였습니다.



### Large Language Models for Manufacturing (https://arxiv.org/abs/2410.21418)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 제조 산업에 통합될 잠재력과 그로 인해 얻을 수 있는 혜택을 종합적으로 탐구합니다. LLMs는 생산 설계부터 품질 관리, 공급망 최적화, 인재 관리까지 다양한 제조 과정의 자동화 및 향상을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 최첨단 LLM인 GPT-4V의 성능을 여러 제조 작업에서 평가하며, 텍스트 처리, 데이터 분석, 코드 생성 및 제로샷 학습(Zero-shot learning) 등에서의 장점을 강조합니다. LLMs는 맞춤화된 데이터 세트로 훈련시켜 특정 제조 지식을 더 잘 습득할 수 있도록 할 수 있습니다.

- **Performance Highlights**: LLMs는 데이터 처리 및 분석에서 매우 유용하며, 심층 인사이트 및 실행 가능한 권장 사항을 생성할 수 있는 도구로서의 가치를 점차 입증하고 있습니다. 그러나 제조 설계와 같은 직접 계산 역할에서는 한계를 가지고 있으며 지원적인 역할이 주를 이룹니다. LLM의 적용은 지속적인 기술 발전을 통해 향상될 수 있습니다.



### Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Wors (https://arxiv.org/abs/2410.21333)
- **What's New**: 본 논문은 Chain-of-thought(CoT) 프롬프트 기법이 대규모 언어 및 다중 모달 모델의 성능에 미치는 영향을 체계적으로 분석합니다. 특히, CoT가 성능을 저하시키는 특정 작업의 특성을 확인하고, 인지 심리학에서 영감을 받아 CoT의 부정적 영향을 이해하기 위한 새로운 틀을 제시합니다.

- **Technical Details**: CoT는 모델이 '단계별 사고'(think step-by-step)를 하도록 유도하는 프롬프트 기법으로, 여러 작업에서 성능을 향상시키는 것으로 알려져 있습니다. 그러나 본 연구에서는 CoT가 OpenAI o1-preview 등 최신 모델의 성능을 최대 36.3%까지 감소시킬 수 있는 작업을 확인했습니다. 실험은 암묵적 통계 학습(implied statistical learning), 시각적 인식(visual recognition), 예외가 포함된 패턴으로 분류하는 작업에서 수행되었습니다.

- **Performance Highlights**: CoT는 특정 조건(i와 ii를 동시에 만족하는 경우)에서 모델 성능을 크게 저하시켰습니다. 예를 들어, CoT를 사용할 때 OpenAI o1-preview의 성능은 GPT-4o와 비교하여 절대 정확도가 최대 36.3% 감소했습니다. 반면, 다른 세 가지 작업 카테고리에서는 CoT가 모델 성능에 부정적인 영향을 미치지 않는 것으로 나타났습니다.



### Building, Reusing, and Generalizing Abstract Representations from Concrete Sequences (https://arxiv.org/abs/2410.21332)
- **What's New**: 이 논문에서는 비모수 계층 변수 학습 모델(HVM)을 도입하여, 시퀀스에서 청크(Chunk)를 학습하고, 맥락적으로 유사한 청크를 변수로 추상화하는 방법을 제안합니다. 이는 메모리를 효율적으로 조직하고 압축된 시퀀스 표현을 가능하게 합니다.

- **Technical Details**: HVM은 관찰 시퀀스에서 스스로 객체와 카테고리를 발견하는 비지도학습 원칙을 사용하며, 청크 제안(Chunk Proposal)과 변수 발견(Variable Discovery)이라는 두 가지 메커니즘을 내세웁니다. HVM은 기존 모델들보다 더 효율적인 딕셔너리를 학습하며, 압축과 일반화 사이의 정밀한 균형을 실현합니다.

- **Performance Highlights**: HVM은 babyLM 언어 데이터셋에서 일반적인 압축 알고리즘인 Lempel-Ziv보다 더 효율적인 성능을 보여주며, 시퀀스를 상기하는 작업에서 HVM의 시퀀스 가능도가 인간의 회상 시간과 관련이 있음을 증명합니다. 대형 언어 모델(LLM)들과의 비교를 통해, HVM은 인간과 유사한 추상화 기제를 보여주고, LLM은 추상 변수를 효과적으로 전이하지 못함을 강조합니다.



### User-Aware Multilingual Abusive Content Detection in Social Media (https://arxiv.org/abs/2410.21321)
- **What's New**: 본 연구에서는 다국어 인디크 언어에서의 모욕적인 콘텐츠 탐지에 대한 새로운 방법론을 제안합니다. 특히 자원이 부족한 저자원 언어에 중점을 두어 사회적 맥락과 사용자 이력의 중요성을 강조합니다.

- **Technical Details**: 제안된 방법은 두 개의 별도 모듈에서 사회적 및 텍스트 맥락 기능을 학습하고, 이들 모듈의 통합 표현을 통해 최종 예측을 수행합니다. 이 방법은 1,500,000개 및 665,000개의 다국어 댓글로 구성된 SCIDN 및 MACI 데이터셋에서 성능을 평가하였습니다.

- **Performance Highlights**: 제안된 방법은 SCIDN 데이터셋에서 4.08%, MACI 데이터셋에서 9.52%의 F1-score 향상을 보이며, 최신 방법 대비 성능이 우수함을 입증하였습니다.



### MatExpert: Decomposing Materials Discovery by Mimicking Human Experts (https://arxiv.org/abs/2410.21317)
- **What's New**: 이번 연구에서는 MatExpert라는 새로운 프레임워크를 도입했습니다. 이 프레임워크는 Large Language Models(LLMs)와 contrastive learning을 활용하여 솔리드 스테이트(고체) 소재의 발견 및 설계를 가속화합니다. MatExpert는 인간 소재 설계 전문가의 작업 흐름을 바탕으로 검토(retrieval), 전환(transition), 생성(generation)이라는 세 가지 주요 단계를 통합합니다.

- **Technical Details**: MatExpert 프레임워크는 다음의 세 가지 단계로 구성됩니다: 1) Retrieval: 주어진 기준과 가장 잘 일치하는 기존 소재를 식별합니다. 2) Transition: 이 소재를 특정 요구 사항에 맞추기 위해 필요한 수정을 개략적으로 설명합니다. 3) Generation: 제공된 정보를 바탕으로 새로운 소재를 생성합니다. 이 과정은 텍스트-구조 검색(text-structure retrieval) 방식을 결합한 chain-of-thought reasoning을 활용하여 이루어집니다.

- **Performance Highlights**: 실험 결과, MatExpert는 소재 생성 작업에서 최신 방법들보다 우수한 성능을 발휘하며, 유효성(validity), 분포(distribution), 안정성(stability) 등의 다양한 메트릭에서 탁월한 결과를 보였습니다. MatExpert는 기존의 데이터 세트인 NOMAD에서 2,886,120 개의 소재를 기반으로 광범위한 평가를 실시하여 해당 분야에서의 일반화 가능성과 강건성을 입증하였습니다.



### $\texttt{PatentAgent}$: Intelligent Agent for Automated Pharmaceutical Patent Analysis (https://arxiv.org/abs/2410.21312)
Comments:
          7 pages

- **What's New**: 이번 연구에서는 제약 산업에서 특허 분석의 통합을 목표로 하는 인공지능 에이전트인 \\texttt{PatentAgent}를 소개합니다. 이는 약물 연구에 영향을 미치는 특허 데이터를 읽고, 화합물 구조를 인식하는 과정을 단순화하며, 연구자들에게 필수적인 도구가 될 것으로 보입니다.

- **Technical Details**: \\texttt{PatentAgent}는 세 가지 주요 모듈로 구성되어 있습니다: \\textit{PA-QA} (특허 질문-응답), \\textit{PA-Img2Mol} (이미지-화합물 구조 변환), \\textit{PA-CoreId} (핵심 화학 구조 식별). 이 시스템은 자연어 쿼리를 처리하고, 화학 이미지에서 분자 구조를 추출하며, 여러 화학 화합물 간의 중심 화학 구조를 정의하는 데 중점을 둡니다.

- **Performance Highlights**: \\textit{PA-Img2Mol}은 CLEF, JPO, UOB, USPTO 특허 벤치마크에서 2.46%에서 8.37%의 정확도를 개선하였고, \\textit{PA-CoreId}는 PatentNetML 벤치마크에서 7.15%에서 7.62%의 향상된 성능을 보여주었습니다. 이러한 결과는 \\texttt{PatentAgent}가 기존 도구보다 얼마나 효과적인지를 강조하고 있습니다.



New uploads on arXiv(cs.IR)

### Synthetic Data Generation with Large Language Models for Personalized Community Question Answering (https://arxiv.org/abs/2410.22182)
Comments:
          Accepted in WI-IAT '24

- **What's New**: 이 연구는 Personalized Information Retrieval (PIR) 분야에서 대규모 데이터셋 부족 문제를 해결하기 위해 Large Language Models (LLMs)를 사용하여 생성된 합성 데이터셋 Sy-SE-PQA를 소개합니다. 이 데이터셋은 StackExchange에서 수집된 질문을 바탕으로 생성된 100,000개의 답변을 포함하고 있습니다.

- **Technical Details**: 이 논문에서는 SE-PQA 데이터셋을 기반으로 사용자가 선호하는 정보에 맞춰 합성 답변을 생성하기 위해 GPT-3.5 및 Phi-3와 같은 최신 LLMs를 사용합니다. 그런 다음, LLM이 생성한 데이터로 Neural Retrieval 모델을 학습하여 그 효과를 평가합니다.

- **Performance Highlights**: LLM이 생성한 데이터로 훈련된 모델이 인간이 작성한 데이터로 훈련된 모델보다 더 나은 성능을 보였으며, 35% 이상의 생성된 답변에서 'hallucination'이 발생한 것으로 확인되었습니다. 이는 LLM 기반의 훈련 데이터가 사실적으로 정확하지 않아도 개인화된 질문-응답 작업에 유용할 수 있음을 시사합니다.



### SimRec: Mitigating the Cold-Start Problem in Sequential Recommendation by Integrating Item Similarity (https://arxiv.org/abs/2410.22136)
Comments:
          ACM RecSys 2024 Workshop on Context-Aware Recommender Systems

- **What's New**: SimRec은 연속 추천 시스템에서 콜드 스타트 문제를 완화하기 위한 새로운 접근 방식을 제안합니다. 이 모델은 아이템 간의 유사성을 활용하여 훈련 과정에 특화된 손실 함수를 포함시킵니다. 이를 통해 훈련 중 명시적으로 보지 못한 희귀 아이템에 대해서도 성능을 향상시킬 수 있습니다.

- **Technical Details**: SimRec은 맞춤형 손실 함수를 통해 훈련 과정에서 아이템 유사성을 통합합니다. 이 접근법은 추가적인 학습 가능한 파라미터 없이 동일한 모델 아키텍처와 훈련 가능한 파라미터 수를 유지하며, 비슷한 인퍼런스 시간을 요구합니다. 아이템 유사성은 텍스트 임베딩 모델을 사용하여 측정됩니다.

- **Performance Highlights**: SimRec은 SASRec과 비교하여 HR@10에서 최대 78%의 향상을 보여주며, 희귀 데이터셋에서 우수한 성능을 발휘하지만 밀집 데이터셋에서도 비슷한 성능을 유지합니다.



### Modeling Temporal Positive and Negative Excitation for Sequential Recommendation (https://arxiv.org/abs/2410.22013)
- **What's New**: 이 논문에서는 Static-Dynamic Interest Learning (SDIL) 프레임워크를 제안하여 사용자의 정적(static) 및 동적(dynamic) 관심을 모두 모델링함으로써 순차 추천(sequential recommendation) 성능을 향상시키고자 한다.

- **Technical Details**: 제안된 SDIL 프레임워크는 (1) Static Interest Modeling (SIM) 모듈, (2) Dynamic Interest Modeling (DIM) 모듈, 그리고 (3) 다음 아이템 예측(next-item prediction) 모듈로 구성된다. 또한 Temporal Positive and Negative Excitation Modeling (TPNE) 모듈이 DIM 모듈과 결합되어 positive 및 negative excitation을 모두 모델링하는 데 중점을 둔다.

- **Performance Highlights**: 세 가지 실제 데이터셋에 대한 광범위한 실험 결과, SDIL은 기존의 최첨단 방법들보다 효과적으로 정적 및 동적 관심을 포착하고 추천 성능이 뛰어난 것으로 나타났다.



### Dual Conditional Diffusion Models for Sequential Recommendation (https://arxiv.org/abs/2410.21967)
- **What's New**: 최근 확산 모델(Diffusion Models) 기술의 발전이 연속 추천(Sequential Recommendation) 분야에서 주목할 만한 성과를 보여주고 있습니다. 본 연구에서는 기존 방식의 두 가지 주요 한계를 극복하기 위해 이산-연속 추천 프레임워크를 제안합니다. 이 방법은 마르코프 체인(Markov Chain)을 도입하여 이산 대상 아이템 지수로의 전환을 모델링하여 추천 프로세스의 일관성을 보장합니다.

- **Technical Details**: 제안된 이중 조건부 확산 모델(Dual Conditional Diffusion Models, DCRec)은 이산 아이템과 연속 확산 모델의 간극을 다리며, 이중 조건부 확산 변환기(Dual Conditional Diffusion Transformer, DCDT)를 통해 사용자 행동을 반영하는 방식으로 개발되었습니다. 이 모델은 임계 조건(implicit condition)과 명시적 조건(explicit condition)을 통합하여 추천의 정확도를 높이고, 샘플링 단계를 줄여 연산 효율성을 증가시킵니다.

- **Performance Highlights**: 공개된 벤치마크 데이터셋에서 실시한 실험 결과, DCRec은 최신 SR 모델 대비 우수한 성능을 나타냈습니다. 특히 소수의 샘플링 단계로 최적의 결과를 달성하여 실시간 애플리케이션에 적합한 다양한 응용 가능성을 확인하였습니다.



### Guided Diffusion-based Counterfactual Augmentation for Robust Session-based Recommendation (https://arxiv.org/abs/2410.21892)
- **What's New**: 본 논문에서는 세션 기반 추천(sesion-based recommendation) 시스템의 인기 편향(popularity bias) 문제를 해결하기 위해 가이드 확산 모델(guided diffusion model)을 이용한 대안 데이터 증강(framework for counterfactual data augmentation) 방법론을 제안합니다. 기존의 데이터 생성 모델과는 달리, 저자는 최첨단 확산 모델을 활용하여 대안 아이템을 생성합니다.

- **Technical Details**: 제안된 DCASR(가이드 확산 기반의 대안 증강 프레임워크)은 SR 모델과 GNN(그래프 신경망), 인과 응답 모델(temporal Structural causal model)을 포함합니다. 이 프레임워크는 실제 및 시뮬레이션 데이터셋에서 함께 작동하여 신뢰성 있는 대안 세션을 생성하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 제안된 DCASR 모델은 기존의 SR 모델 및 다른 증강 프레임워크에 비해 현저한 성능 향상을 보여주었습니다. 특히, 적게 선호되는 타겟 아이템에서 최대 20% 향상된 Recall과 13% 향상된 CTR을 달성하였습니다.



### Application of Audio Fingerprinting Techniques for Real-Time Scalable Speech Retrieval and Speech Clusterization (https://arxiv.org/abs/2410.21876)
- **What's New**: 본 논문은 음성 통신 및 클라우드 통신 플랫폼에서 음성 검색 최적화를 위한 새로운 통찰을 제시합니다. 기존의 음악 중심의 오디오 지문 기술을 음성 검색에 맞게 조정하여, 배치 처리에서의 신속하고 정확한 오디오 검색을 목표로 하고 있습니다.

- **Technical Details**: 이 논문은 Session Initiation Protocol (SIP) 내에서 조기 미디어(early media) 분석을 다루고, 이들 오디오 파일을 오디오 지문 인식 기술을 통해 분류합니다. 여기에 필요한 것은 실시간 처리이며, 음성-텍스트 변환 없이 음성 전사에 기반한 오디오 클러스터링을 지원할 수 있는 방법도 제시합니다.

- **Performance Highlights**: 오디오 지문 기술을 통해 실시간으로 조기 미디어 파일을 분류함으로써, 일반적으로 실시간 작동을 위해 필요한 GPU 컴퓨팅 없이도 상당한 빠른 처리가 가능하다는 점을 강조하고 있습니다. 이 최적화는 통신 네트워크 성능 분석에도 유용하게 활용될 수 있습니다.



### PerSRV: Personalized Sticker Retrieval with Vision-Language Mod (https://arxiv.org/abs/2410.21801)
- **What's New**: 본 논문에서는 개인 맞춤형 스티커 검색을 위한 새로운 프레임워크인 PerSRV(개인화된 스티커 검색 및 비전-언어 모델)를 제안합니다.

- **Technical Details**: PerSRV는 오프라인 계산과 온라인 처리 모듈로 구성되며, 온라인 검색 부분은 관련 검색 및 개인화된 순위 매기기 패러다임을 따릅니다. 오프라인 부분에서는 스티커의 의미 이해, 유용성 평가 및 개인화 모듈을 포함합니다. LLaVA-1.5-7B 모델을 통해 인간과 유사한 스티커 의미를 생성하고, 크라우드 소스 메트릭스를 통해 스티커 유용성을 평가하며, 사용자 개인 선호 모델링을 위해 스타일 중심을 클러스터링합니다.

- **Performance Highlights**: PerSRV는 WeChat의 공개 스티커 검색 데이터셋에서 기존 방법들에 비해 다중 모달 스티커 검색에서 유의미한 성능 향상을 보여줍니다. 또한, 미세 조정된 VLM은 스티커의 의미 이해에 있어 눈에 띄는 개선을 기록했습니다.



### Semantic Search Evaluation (https://arxiv.org/abs/2410.21549)
Comments:
          Accepted by 3rd International Workshop on Industrial Recommendation Systems (at CIKM 2024)

- **What's New**: 이 논문은 콘텐츠 검색 시스템의 성능을 평가하는 새로운 방법을 제안합니다. 이 방법은 쿼리와 검색 결과 간의 의미적 일치를 측정합니다. 새로운 메트릭인 'on-topic rate'를 도입하여 쿼리와 관련된 결과의 비율을 측정합니다.

- **Technical Details**: 제안된 평가 파이프라인은 금색 쿼리 세트를 정의하고 각 쿼리에 대해 상위 K 개의 결과를 검색하며, GPT 3.5와의 프롬프트를 통해 평가합니다. 이 과정은 쿼리와 문서 간의 관련성을 판단하기 위해 Generative AI(GAI)를 활용합니다. OTR(온-토픽 레이트)은 검색 결과가 쿼리와 얼마나 관련성이 있는지를 측정합니다.

- **Performance Highlights**: OTR을 통해 검색 엔진의 성능을 평가하고 개선할 수 있는 패턴을 확인할 수 있습니다. 특히 OTR@K 메트릭을 통해 모델의 성능을 평가할 수 있으며, 이는 검색 엔진의 품질을 지속적으로 측정할 수 있는 도구로서 구실합니다.



### Can Users Detect Biases or Factual Errors in Generated Responses in Conversational Information-Seeking? (https://arxiv.org/abs/2410.21529)
Comments:
          Extended version of the paper that appeared in the Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP '24)

- **What's New**: 이번 연구에서는 정보 검색 대화 시스템에서의 응답 생성의 한계를 조사하며, 사용자가 인식할 수 있는 오류들(부정확성, 편향 등)을 중심으로 사용자 경험에 미치는 영향을 분석합니다.

- **Technical Details**: 정보 검색 대화 시스템(CIS)에서의 쿼리의 응답 가능성(query answerability)과 응답 불완전성(response incompleteness)에 대한 두 가지 크라우드소싱 관련 연구를 진행하였으며, 사용자의 능력 및 응답의 다양성(variability)이 만족도와 관련이 있는지를 분석하였습니다. 응답은 정보의 입장(viewpoint) 및 측면(facet)에서 다양성을 고려하여 설계되었습니다.

- **Performance Highlights**: 연구 결과, 사용자는 응답의 부정확성을 인식하는 데 어려움을 겪지만, 응답의 다양성 및 균형 있는 관점을 식별하는 데는 상대적으로 용이하다는 것을 발견했습니다. 전반적으로 사용자의 만족도는 응답의 사실적 정확성(factual correctness)보다 응답의 다양성에 더 많은 연관이 있음을 보여줍니다.



### Enhancing CTR Prediction in Recommendation Domain with Search Query Representation (https://arxiv.org/abs/2410.21487)
Comments:
          Accepted by CIKM 2024 Full Research Track

- **What's New**: 이 논문은 사용자 검색 쿼리의 임베딩을 추천 도메인에서의 사용자 선호도와 연관 지어 학습하는 프레임워크를 제안합니다. 이러한 접근은 기존의 방법에서 간과된 사용자 의도 전이를 다루며, 검색 도메인에서의 사용자 행동 데이터를 활용하여 추천 시스템의 클릭률(CTR) 예측 정확도를 향상시킵니다.

- **Technical Details**: 제안된 프레임워크인 QueryRec는 사용자 쿼리 히스토리를 추천 도메인에 보강된 특성으로 추가하여 사용자 관심사를 보다 효과적으로 반영합니다. 이 프레임워크는 또한 자기 주의 기반의 순차 인코더를 사용하여 쿼리 리스트와 다음 클릭된 아이템의 임베딩을 정렬하며, 대조 학습 모듈을 통해 검색 도메인에서의 쿼리-아이템 관계를 캡처합니다. 확산 모델(diffusion model)을 도입하여 긍정적인 아이템의 예측을 강화하고, 노이즈 제거 방식을 통해 잘못된 긍정 예측을 방지합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 최신 모델들과 비교하여 추천 도메인에서 뛰어난 성능을 보였으며, 사용자 선호도 전이를 효과적으로 반영하여 정확한 클릭률 예측을 가능하게 하였습니다.



### Pushing the Performance Envelope of DNN-based Recommendation Systems Inference on GPUs (https://arxiv.org/abs/2410.22249)
Comments:
          This work has been accepted in the 57th MICRO (this https URL). Please check appendix for details on reproducing our work including codebase and steps

- **What's New**: 이 논문에서는 Deep Learning Recommendation Models (DLRMs)의 추론(인퍼런스) 성능 개선을 위한 새로운 방법을 제안하며, 주로 GPU에서의 임베딩 단계가 병목 현상(Bottleneck)을 일으킨다는 점을 강조합니다.

- **Technical Details**: DLRM의 네 가지 주요 단계는 임베딩, 바닥 다층 퍼셉트론(Multi-Layer Perceptron), 피처 상호 작용, 그리고 최상위 MLP입니다. 임베딩 단계는 메모리 집약적인 문제로, CUDA 기반의 최신 DLRM을 사용해 테스트한 결과, '무작위' 접근 방식에서 '단일 항목' 접근 방식으로 성능이 3.2배 느려지는 병목 현상이 발견되었습니다. 연구 결과, 레지스터 압박(Register Pressure)으로 인해 최적의 다중 처리(Parallel Processing)를 달성하지 못하는 문제가 있었습니다. 또한, 메모리 지연 문제를 숨기기 위한 소프트웨어 프리패칭(Prefetching)과 L2 핀(Pinning) 방법을 제안했습니다.

- **Performance Highlights**: 제안한 기술들은 A100 GPU와 대형 모델을 사용한 실험에서 임베딩 단계에서 최대 103%, 전체 DLRM 인퍼런스 파이프라인에서 최대 77%의 성능 개선을 나타냈습니다. 각기 독립적으로 프리패칭과 L2 핀은 각각 최대 97%와 62%의 성능 개선을 보여주었으며, 두 방법을 결합할 경우 성능 향상이 더욱 두드러졌습니다.



### ContextIQ: A Multimodal Expert-Based Video Retrieval System for Contextual Advertising (https://arxiv.org/abs/2410.22233)
Comments:
          Accepted at WACV 2025

- **What's New**: ContextIQ는 맥락 광고(contextual advertising)를 위해 특별히 설계된 다중 모달 전문가 기반 비디오 검색 시스템입니다. 이를 통해 비디오, 오디오, 자막 등 다양한 모달리티에 기초하여 의미적으로 풍부한 비디오 표현을 생성합니다.

- **Technical Details**: ContextIQ는 전문가 모델을 활용하여 비디오 검색을 수행하며, 모달리티별 전문가(비디오, 오디오, 자막, 메타데이터)를 사용하여 비디오 콘텐츠를 이해합니다. 이 시스템은 대규모 공동 학습 없이도 최신 모델들과 비교하여 유사하거나 더 나은 성능을 보입니다.

- **Performance Highlights**: ContextIQ는 MSR-VTT, Condensed Movies 및 신규 공개된 Val-1 데이터셋에서 경쟁력 있는 성능을 보였으며, 광고 생태계에서 브랜드 안전 및 부적절한 콘텐츠 필터링 문제를 해결하며 효과적으로 통합될 수 있음을 보여줍니다.



### Testing Identity of Distributions under Kolmogorov Distance in Polylogarithmic Spac (https://arxiv.org/abs/2410.22123)
- **What's New**: 이번 논문은 연속 분포의 테스트에서 공간 복잡도를 획기적으로 줄일 수 있는 새로운 알고리즘을 제시합니다. 특히, Kolmogorov 거리에서 필요로 하는 공간을 O(log⁴ 1/ε)로 감소시킴으로써 기존의 높은 공간 복잡도를 저감하였습니다.

- **Technical Details**: Kolmogorov 거리(Kolmogorov distance)는 누적 분포 함수의 L∞ 거리로 정의됩니다. 이전 작업에서는 Kolmogorov–Smirnov 테스트에 따라 Θ(log(1/δ)/ε²) 샘플이 필요했으나, 본 연구에서는 O(log(1/δ)/ε²) 샘플을 유지하면서 공간 복잡도를 줄이는 알고리즘을 소개합니다.

- **Performance Highlights**: 제안된 알고리즘은 O(log⁴ 1/ε) 공간에서 작업하며, 이는 연속 분포의 정체성 테스트(identity testing)에서 매우 낮은 공간 요구 사항을 보여줍니다. 또한, 이는 디스크리트(discrete) 분포에서의 표준 총 변이 거리(total variation distance)와 비교할 때 비현실적인 공간 감소를 가능하게 하였습니다.



### A Dual Adaptive Assignment Approach for Robust Graph-Based Clustering (https://arxiv.org/abs/2410.21745)
- **What's New**: 이번 논문에서는 RDSA(Robust Deep Graph Clustering via Dual Soft Assignment)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 그래프 기반 클러스터링의 강인성을 향상시키기 위해 두 가지 종류의 soft assignment를 활용합니다.

- **Technical Details**: RDSA는 다음의 세 가지 주요 구성 요소로 이루어져 있습니다: (i) 노드 임베딩 모듈: 그래프의 구조적 특성과 노드 속성을 효과적으로 통합합니다. (ii) 구조 기반 soft assignment 모듈: 노드 할당을 위한 affinity matrix를 이용하여 그래프의 모듈러리티(modularity)를 개선합니다. (iii) 노드 기반 soft assignment 모듈: 커뮤니티 랜드마크를 식별하고 노드 할당을 정제하여 모델의 강인성을 향상시킵니다.

- **Performance Highlights**: 다양한 실제 데이터셋에서 RDSA의 성능을 평가한 결과, 기존의 최첨단 방법과 비교해 뛰어난 성능을 보여주었습니다. RDSA는 클러스터링 효과성 및 강인성, 노이즈에 대한 적응성, 훈련 안정성, 그리고 대규모 데이터셋에 대한 확장성을 포함하여 뛰어난 클러스터링 성능을 제공합니다.



### A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions (https://arxiv.org/abs/2410.21484)
- **What's New**: 이번 연구는 스포츠 베팅에서 머신러닝(Machine Learning) 기술의 최신 동향과 도전 과제를 살펴보고 있습니다. 연구에 따르면, 머신러닝은 베팅 전략 최적화와 예측의 정확성을 향상시키며, 이 과정에서 다양한 장르의 스포츠에 적용됩니다.

- **Technical Details**: 연구에서는 서포트 벡터 머신(Support Vector Machines), 랜덤 포레스트(Random Forests), 신경망(Neural Networks) 등 다양한 머신러닝 기법이 활용됩니다. 이 모델들은 역사적 데이터, 경기 중 통계 및 실시간 정보를 사용하여 베팅 전략을 최적화하고 가치 있는 베팅 기회를 식별합니다.

- **Performance Highlights**: 머신러닝 모델들은 베팅 수익성을 향상시키는 데 중요한 역할을 하며, 베팅업체는 역동적인 배당 조정(dynamic odds adjustment)을 통해 리스크를 관리합니다. 또한, 의심스러운 베팅 패턴을 감지하기 위한 이상 탐지(anomaly detection) 모델의 활용이 강조됩니다.



New uploads on arXiv(cs.CV)

### Task Vectors are Cross-Moda (https://arxiv.org/abs/2410.22330)
- **What's New**: 이번 연구에서는 시각-언어 모델(VLMs)의 내부 표현과 이들이 과제 표현(task representations)을 어떻게 인코딩하는지를 조사합니다. 새로운 점은, 유사한 개념의 과제가 어떻게 명세되었는지에 관계없이 유사한 과제 벡터 표현(task vector representations)으로 매핑된다는 것입니다.

- **Technical Details**: 우리는 예시나 지시를 통해 지정된 과제를 고려하며, 텍스트 또는 이미지 입력을 사용합니다. VLM에서 발견된 과제 벡터는 한 모달리티(예: 텍스트)에서 파생되어 다른 모달리티(예: 이미지)로 전이될 수 있을 만큼 일반적입니다. 또한, 예시 기반(exemplar)과 지시 기반(instruction based)의 과제 벡터를 집계(ensembling)하면 더 나은 과제 표현을 제공합니다.

- **Performance Highlights**: 이 연구 결과는 VLM의 설계 메커니즘 및 서로 다른 모달리티와 과제 명세를 통해 과제를 공유하는 방식에 대한 통찰을 제공합니다. VLM이 입력(input), 과제(task), 응답(answer)의 세 가지 독특한 단계로 토큰을 처리하는 과정을 밝혀냈습니다.



### Multi-Class Textual-Inversion Secretly Yields a Semantic-Agnostic Classifier (https://arxiv.org/abs/2410.22317)
Comments:
          Accepted in WACV 2025. Code link: this https URL

- **What's New**: 이번 논문에서는 텍스트-이미지(Text-to-Image, T2I) 모델의 개인화에서 의미 불감성(semantic-agnostic) 분류를 탐구하고, 기존의 단일 개념 텍스트 연결(Textual Inversion, TI) 방법론에서 한 걸음 나아가 다중 클래스 텍스트 연결(Multi-Class Textual Inversion, MC-TI) 방법론을 제안합니다.

- **Technical Details**: MC-TI 방법론은 분류 불확실성을 줄이기 위해 특화된 정규화(discriminative regularization) 항을 포함하여 학습 과정을 개선합니다. 각 훈련 단계에서는 각 범주에 대해 임의로 선택한 이미지 특징을 샘플링하여 현재 텍스트 프롬프트와의 분류 확률을 계산하며, 여기에서 코사인 유사도(cosine similarity)를 기반으로 한 교차 엔트로피 손실(cross-entropy loss)을 추가하여 분류 성능을 높입니다.

- **Performance Highlights**: MC-TI는 12개의 데이터셋에 걸쳐 평가되었으며, 소수 샘플로도 우수한 성능을 보여줍니다. 특히, CLIP과 유사한 제로샷(zero-shot) 방법론과 비교하여 적은 샘플(최소 1샷, 최대 8샷)으로도 뛰어난 분류와 생성 성능을 달성하였습니다.



### Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving (https://arxiv.org/abs/2410.22313)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 Senna라는 자율 주행 시스템을 소개하고 있습니다. Senna는 대형 비전-언어 모델(LVLM)과 엔드투엔드(end-to-end) 모델을 통합한 시스템으로, 고수준의 계획 결정과 저수준의 궤적 예측을 분리하여 더 나은 계획 성능을 제공합니다.

- **Technical Details**: Senna는 LVLM을 통해 자연어로 계획 결정을 생성하고, 엔드투엔드 모델(Senna-E2E)은 이를 바탕으로 정밀한 궤적을 예측합니다. Senna-VLM은 다중 이미지 인코딩과 멀티 뷰 프롬프트를 활용하여 장면 이해를 향상시킵니다. 또한, 계획 지향적인 QAs(질문-답변)와 삼단계 훈련 전략을 도입하여 계획 성능을 개선합니다.

- **Performance Highlights**: Senna는 큰 규모의 데이터셋 DriveX에서 사전 훈련(pre-training) 후 nuScenes 데이터셋에서 세밀한 조정(fine-tuning) 과정을 거친 결과, 평균 계획 오차를 27.12% 줄이고, 충돌률을 33.33% 감소시켰습니다. 이는 Senna의 다양한 상황에 대한 일반화 및 이전 가능성(transferability)에 중요한 기여를 합니다.



### Effective Guidance for Model Attention with Simple Yes-no Annotations (https://arxiv.org/abs/2410.22312)
Comments:
          10 pages, 5 figures, IEEE BigData 2024 Paper

- **What's New**: 최근 딥러닝 모델들은 예측 시 irrelevant areas에 집중하여 성능 저하와 일반화 제한을 초래하는 경향이 있습니다. 이러한 문제를 해결하기 위해 CRAYON을 제안합니다. CRAYON은 간단한 yes-no 주석을 활용하여 모델의 주의를 수정하는 효과적이고 실용적인 방법입니다.

- **Technical Details**: CRAYON은 두 가지 모드로 작동합니다: CRAYON-ATTENTION은 saliency maps를 기반으로 모델 해석을 돕고, CRAYON-PRUNING은 irrelevant neurons를 제거하여 모델 성능을 개선합니다. CRAYON은 10%의 훈련 데이터에 대한 주석만으로도 최첨단 성능을 달성할 수 있습니다.

- **Performance Highlights**: CRAYON은 3개의 벤치마크 데이터셋에서 12개의 기존 방법보다 우수한 성능을 보이며, 약 6,000명의 참여자가 포함된 대규모 인간 주석을 바탕으로 효과성과 확장성을 입증했습니다.



### Multi-Object 3D Grounding with Dynamic Modules and Language-Informed Spatial Attention (https://arxiv.org/abs/2410.22306)
Comments:
          NeurIPS 2024

- **What's New**: D-LISA는 multi-object 3D grounding을 위한 혁신적인 두 단계 방법을 제안하며, 동적 비전 모듈, 동적 카메라 포지셔닝 및 언어 기반 공간 주의 모듈을 포함하여 기존 방식의 한계를 극복합니다.

- **Technical Details**: D-LISA는 3D 포인트 클라우드로부터 개체를 찾기 위해 동적 제안 생성과 다양한 시점에서의 기능 추출을 결합합니다. 언어 정보를 활용하여 객체 간의 공간적 관계를 효과적으로 추론할 수 있는 모듈을 도입합니다.

- **Performance Highlights**: 복합화된 실험 결과, D-LISA는 기존의 M3DRef-CLIP 방법에 비해 12.8%의 성능 향상을 보였으며, 단일 객체 3D grounding에서도 경쟁력 있는 결과를 도출했습니다.



### Motion Graph Unleashed: A Novel Approach to Video Prediction (https://arxiv.org/abs/2410.22288)
Comments:
          Accepted by NeurIPS 2024, 19 pages, 12 figures

- **What's New**: 본 논문에서는 비디오 예측 문제를 위한 새로운 접근 방식인 motion graph를 소개합니다. 이는 제한된 과거 데이터로부터 미래의 비디오 프레임을 예측합니다.

- **Technical Details**: motion graph는 비디오 프레임의 패치를 상호 연결된 그래프 노드로 변환하여 공간-시간 관계를 종합적으로 설명합니다. 이 방법은 기존의 motion representations의 한계를 극복하며, 다양한 동작 패턴을 보다 효과적으로 포착합니다.

- **Performance Highlights**: UCF Sports 데이터세트에서 본 방법은 SOTA(State-of-the-art) 방법과 일치하거나 이를 초과하며, 모델 크기를 78% 줄이고 GPU 메모리 사용량을 47% 감소시켰습니다.



### Active Event Alignment for Monocular Distance Estimation (https://arxiv.org/abs/2410.22280)
- **What's New**: 이 논문은 event camera(이벤트 카메라) 데이터에서 객체별 거리 추정(Object-wise distance estimation)을 위한 행동 기반 접근법을 제안합니다. 제안된 방법은 인간 시각 시스템에서 영감을 받아 대상을 안정화하여 상대적 거리를 추정하는 방식입니다.

- **Technical Details**: 이 연구는 이벤트 카메라의 자연스러운 데이터 표현을 활용하여 영상의 고속 처리를 가능하게 합니다. 행동 기반의 회전 조정(rotational adjustments)을 통해 입력된 이벤트를 정렬하고, 지역 내의 각도 속도(angular velocity)를 추정하여 물체의 거리와 깊이를 추정합니다. 이 방법은 기존의 깊이 추정(detection)을 정적인 이미지 전역에서가 아니라 특정 관심 영역(region of interest) 내에서 수행합니다.

- **Performance Highlights**: 제안된 방법은 EVIMO2 데이터 세트에서 거리 추정에서 16%의 성능 향상을 기록하며, 이전의 방법보다 더 높은 정확도로 객체 간 상대적 깊이를 효율적으로 추정했습니다.



### NCA-Morph: Medical Image Registration with Neural Cellular Automata (https://arxiv.org/abs/2410.22265)
- **What's New**: NCA-Morph는 Deep Learning (DL) 기술과 생물학적 기법인 Neural Cellular Automata (NCA)를 결합하여 의료 이미지 등록의 효율을 극대화하는 새로운 접근방식을 제시합니다. 이 방법은 기존 DL 방법들보다 적은 리소스를 요구하며, 빠른 처리 속도를 자랑합니다.

- **Technical Details**: NCA-Morph는 3D 의료 이미지 등록을 위한 경량의 아키텍처를 제공하며, VoxelMorph 대비 60%, TransMorph 대비 99.7% 적은 매개변수를 사용합니다. 또한, 이 방법은 세포 간의 지역적 통신 네트워크를 생성하여 생명체의 상호작용을 모방합니다. NCA는 약 13,000개의 매개변수로 구성되어 있어 계산 효율성이 뛰어납니다.

- **Performance Highlights**: NCA-Morph의 성능은 Brain, Prostate 및 Hippocampus와 같은 세 가지 3D 등록 작업에서 평가되었으며, 최신 기술에 필적하는 정확도를 달성했습니다. 이 방법은 GPU를 사용하여 몇 초 만에 등록을 완료할 수 있으며, 경량 구조로 Raspberry Pi와 같은 자원 제한 환경에서도 실행 가능합니다.



### ContextIQ: A Multimodal Expert-Based Video Retrieval System for Contextual Advertising (https://arxiv.org/abs/2410.22233)
Comments:
          Accepted at WACV 2025

- **What's New**: ContextIQ는 맥락 광고(contextual advertising)를 위해 특별히 설계된 다중 모달 전문가 기반 비디오 검색 시스템입니다. 이를 통해 비디오, 오디오, 자막 등 다양한 모달리티에 기초하여 의미적으로 풍부한 비디오 표현을 생성합니다.

- **Technical Details**: ContextIQ는 전문가 모델을 활용하여 비디오 검색을 수행하며, 모달리티별 전문가(비디오, 오디오, 자막, 메타데이터)를 사용하여 비디오 콘텐츠를 이해합니다. 이 시스템은 대규모 공동 학습 없이도 최신 모델들과 비교하여 유사하거나 더 나은 성능을 보입니다.

- **Performance Highlights**: ContextIQ는 MSR-VTT, Condensed Movies 및 신규 공개된 Val-1 데이터셋에서 경쟁력 있는 성능을 보였으며, 광고 생태계에서 브랜드 안전 및 부적절한 콘텐츠 필터링 문제를 해결하며 효과적으로 통합될 수 있음을 보여줍니다.



### Towards Unifying Understanding and Generation in the Era of Vision Foundation Models: A Survey from the Autoregression Perspectiv (https://arxiv.org/abs/2410.22217)
- **What's New**: 이번 리뷰 논문은 비전 기초 모델(vision foundation models)에서 오토회귀(autoregression)의 발전을 다루고 있습니다. 특히, 이해(interpreting)와 생성(generating)을 통합하는 차세대 비전 기초 모델을 제시하고, 이러한 모델이 직면한 한계와 오토회귀의 장점을 설명합니다.

- **Technical Details**: 오토회귀 비전 기초 모델에서는 이미지가 일련의 토큰(token)으로 토크나이즈(tokenize)되어 다음 토큰 예측(next-token prediction)이 이루어집니다. 이 모델들은 Transformer 아키텍처를 기반으로 하며, 이해와 생성이라는 두 개의 비전 작업을 통합하기 위한 방법을 모색합니다. 또한, 비전 토크나이저와 오토회귀 Transformer의 분류를 통해 이러한 모델을 분석합니다.

- **Performance Highlights**: 비전 이해와 생성의 통합은 서로를 증진시키는 시너지를 창출합니다. 오토회귀 비전 기초 모델은 대규모 데이터에서 효과적인 표현을 학습하고, Transformer의 확장성을 최대한 활용함으로써 뛰어난 성능을 보여줍니다. 이 논문은 오토회귀 비전 모델의 현재 발전과 앞으로의 연구 방향을 포괄적으로 탐구하고 있습니다.



### LiVisSfM: Accurate and Robust Structure-from-Motion with LiDAR and Visual Cues (https://arxiv.org/abs/2410.22213)
Comments:
          18 pages, 9 figures, 2 tables

- **What's New**: 이번 논문에서는 LiDAR(라이다)와 시각적 단서를 통합한 새로운 구조-모션(SfM) 파이프라인인 LiVisSfM을 소개합니다. 기존의 LiDAR 기반 방법들과는 달리, IMU(관성 측정 장치)에 의존하지 않고 정확하고 강력한 LiDAR 자세 추정을 구현하였습니다.

- **Technical Details**: LiVisSfM는 Point-to-Gaussian residual metrics를 활용하여 LiDAR 프레임을 voxel 맵에 등록하며, LiDAR-visual BA(Bundle Adjustment)와 명시적 루프 클로저를 포함한 방법을 적용하여 정밀한 LiDAR 위치 추정을 가능하게 합니다. 이 시스템은 LiDAR와 비주얼 단서를 결합하여 효율적인 최적화를 수행합니다.

- **Performance Highlights**: 실험 결과, LiVisSfM은 KITTI 벤치마크와 다양한 자가 캡처 데이터셋에서 기존의 LIO 및 LIVO 방법들에 비해 더 정확하고 견고한 LiDAR 자세 복구 및 밀집 포인트 클라우드 재구성을 보여주었습니다.



### Active Learning for Vision-Language Models (https://arxiv.org/abs/2410.22187)
Comments:
          Accepted in WACV 2025

- **What's New**: 이 논문에서는 Vision-Language 모델(VLM)의 Zero-shot 분류 성능을 개선하기 위해 새로운 Active Learning(활성 학습) 프레임워크인 Calibrated Entropy-weighted Clustering(CEC)를 제안합니다. 이 방법은 레이블이 없는 데이터를 학습하는 과정에서 정보가 풍부한 샘플만을 선택하여 성능 격차를 줄이는 데 초점을 맞춥니다.

- **Technical Details**: 제안된 CEC 방법은 VLM에서 예측된 엔트로피(Entropy)를 보정(calibrate)하고, Self-uncertainty와 Neighbor-aware uncertainty를 결합하여 샘플 선택을 위한 신뢰할 수 있는 불확실성 측정치를 계산합니다. 이를 통해 VLM에서 정보가 풍부한 샘플을 선택하여 Prompt Tuning의 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 이미지 분류 데이터셋에서 기존의 최첨단 Active Learning 방법을 초월하며 VLM의 Zero-shot 성능을 유의미하게 개선하는 것임을 입증하였습니다.



### Multi-Level Feature Distillation of Joint Teachers Trained on Distinct Image Datasets (https://arxiv.org/abs/2410.22184)
Comments:
          Accepted at WACV 2025

- **What's New**: 이번 논문에서는 여러 개의 서로 다른 데이터 세트로 훈련된 여러 교사로부터 지식을 증류하는 새로운 teacher-student 프레임워크를 제안합니다. 각 교사는 자신의 데이터 세트로부터 처음부터 훈련받고, 이후 모든 교사의 기능이 결합되어 공동 아키텍처를 형성합니다. 최종적으로, 이 공동 교사로부터의 지식을 student 모델로 전달하기 위해 multi-level feature distillation (MLFD) 절차를 사용합니다.

- **Technical Details**: 제안된 MLFD 방법은 세 가지 단계로 구성됩니다: 1단계에서는 각 교사가 자신의 데이터 세트에서 처음부터 학습됩니다. 2단계에서는 개별 교사들을 결합하여 공동 교사를 생성하며, 이는 모든 데이터 세트에 대해 훈련됩니다. 3단계에서는 집합적 지식을 각 데이터 세트에 맞는 student 모델로 증류합니다. 이러한 방법은 다중 수준의 표현을 사용하여 지식을 추출하고, 학습된 모델의 성능을 높입니다.

- **Performance Highlights**: 이미지 분류 및 동작 인식 실험에서 MLFD 방법이 같은 모델 아키텍처를 가진 기존 모델보다 상당한 성능 향상을 보여줍니다. 또한, 데이터 세트별 학생 아키텍처로부터의 지식 증류가 성능 향상을 가져온다는 점을 입증하였습니다.



### Shining a Light on Hurricane Damage Estimation via Nighttime Light Data: Pre-processing Matters (https://arxiv.org/abs/2410.22150)
- **What's New**: 이번 연구는 허리케인 피해 예측에 대한 기존 연구의 한계를 극복하기 위해서 다양한 야간 조명 데이터(NTL) 전처리 기법을 체계적으로 비교하고 분석했습니다. 특히, 'built masking'과 'quality filtering and imputation' 기법의 조합을 신규로 도입하여 허리케인 피해와의 상관관계를 강화했습니다.

- **Technical Details**: 연구에서는 VIIRS 데이터에서 야간 조명(NTL) 값에 대해 세 가지 전처리 기법을 적용했습니다: value thresholding, built masking, quality filtering and imputation. 이 기법들은 허리케인 피해 데이터를 보다 정확하게 분석하기 위해 zip 코드 수준에서 적용되었습니다. 두 가지 데이터셋인 VSC-NTL와 VNP46A2에서 실험을 수행하며, 특히 VNP46A2에 적용한 품질 마스킹 및 임퓨테이션 기법이 경제적 피해와의 강한 상관관계를 나타냈습니다.

- **Performance Highlights**: 실험 결과, 품질 마스킹 및 임퓨테이션 기법이 적용된 VNP46A2 데이터셋에서 허리케인에 의한 경제적 피해와 높은 상관관계를 보였습니다. 이를 통해, 야간 조명 데이터의 전처리 방법의 선택이 결과의 신뢰성과 일관성에 중대한 영향을 미친다는 것을 강조하게 되었습니다.



### Capacity Control is an Effective Memorization Mitigation Mechanism in Text-Conditional Diffusion Models (https://arxiv.org/abs/2410.22149)
Comments:
          Accepted at the GenLaw (Generative AI + Law) workshop at ICML'24

- **What's New**: 본 논문에서는 diffusion models에서 memorization을 효과적으로 억제할 수 있는 모델 용량 조절의 중요성을 제시합니다. 특히, 전통적인 full fine-tuning 접근 방식과 비교하여 Parameter-Efficient Fine-Tuning(PEFT) 기법을 채택했을 때 memorization이 크게 감소함을 보여줍니다.

- **Technical Details**: PEFT 방법은 기존의 사전 훈련된 모델의 대부분 파라미터를 동결하고, 특정 파라미터 서브셋만 조정하는 기법으로, Selective와 Additive 전략으로 구분됩니다. 실험에서는 Stable Diffusion(v1.5) 모델과 MIMIC 데이터셋을 활용하여, U-Net만 fine-tune 하였으며 다른 구성 요소는 동결되었습니다.

- **Performance Highlights**: 실험 결과, PEFT 방법이 이미지 생성 품질(FID, BioViL-T Score 증가)과 memorization(평균 최소 코사인 거리 감소, 추출된 이미지 수 감소) 모두에서 우수한 성능을 보였습니다. 결과적으로, 모델 용량을 조절하는 것이 memorization 억제와 생성 품질 향상에 기여한다는 것을 나타냅니다.



### Lighten CARAFE: Dynamic Lightweight Upsampling with Guided Reassemble Kernels (https://arxiv.org/abs/2410.22139)
Comments:
          Accepted at ICPR 2024

- **What's New**: 이 논문에서는 기존의 Content-aware Reassembly of Features (CARAFE) 방법의 복잡성과 필요 파라미터 수를 줄인 새로운 방법인 Dynamic Lightweight Upsampling (DLU)을 제안합니다. DLU는 소규모 소스 커널 공간에서 대규모 커널을 샘플링하는 과정을 통해 성능을 향상시킵니다.

- **Technical Details**: DLU는 학습 가능한 가이던스 오프셋을 도입하여 소규모 소스 커널 공간에서 대규모 커널을 샘플링합니다. 이러한 방식을 통해 DLU는 CARAFE에 비해 91% 적은 파라미터와 최소 63% 적은 FLOPs (Floating Point Operations)를 필요로 하며, 복잡성을 크게 줄입니다.

- **Performance Highlights**: 실험 결과 DLU는 객체 검출에서 CARAFE보다 0.3% mAP 향상된 성능을 보이며, FPN과 Libra RCNN 모델에 DLU를 통합했을 때 각각 1.2%와 0.7% mAP 성능이 개선되었습니다.



### Lightweight Frequency Masker for Cross-Domain Few-Shot Semantic Segmentation (https://arxiv.org/abs/2410.22135)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이번 연구에서 제안된 Cross-domain few-shot segmentation (CD-FSS) 방법은 대량의 데이터셋에서 사전 학습된 모델을 사용하여 데이터가 부족한 타겟 도메인에 대한 픽셀 수준의 분할을 수행합니다. 특히, 주목할 만한 점은 주파수 구성 요소를 필터링함으로써 성능을 14%까지 향상시킬 수 있다는 것입니다.

- **Technical Details**: 이 연구에서는 Amplitude-Phase Masker (APM)와 Adaptive Channel Phase Attention (ACPA) 모듈을 통해 경량화된 주파수 마스커를 제안합니다. APM 모듈은 추가 매개변수가 0.01%만 증가하면서 평균 mIoU 성능을 10% 이상 개선하고, ACPA는 추가 매개변수 2.5%로 평균 성능을 1.5% 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 기존의 CD-FSS 방법보다 현저하게 우수하며, 네 가지 타겟 데이터셋에서의 광범위한 실험 결과 평균 mIoU 성능이 11% 향상되었습니다. 이 연구는 간단하고 효과적인 설계를 통해 최신 CD-FSS 기법을 초월하는 성능을 구현하였습니다.



### PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting (https://arxiv.org/abs/2410.22128)
Comments:
          project page: this https URL

- **What's New**: 본 연구에서는 PF3plat(포즈 프리 피드 포워드 3D Gaussian Splatting)이라는 새로운 프레임워크를 제안하여, 비포즈 이미지로부터 빠르고 사실적인 새로운 뷰 합성을 제공합니다. 이 방법은 고속 및 고품질의 3D 재구성과 뷰 합성을 위한 3D Gaussian Splatting(3DGS)의 효율성을 활용하면서 밀집 이미지 뷰, 정확한 카메라 포즈 및 상당한 이미지 중첩과 같은 일반적인 가정을 완화합니다.

- **Technical Details**: PF3plat 프레임워크는 정밀한 깊이 및 카메라 포즈 추정에 의존하지 않고 3D Gaussians의 조정 문제를 해결합니다. 이를 위해 사전 훈련된 단안(depth) 추정 및 시각적 대응 모델을 활용하여 3D Gaussian의 대략적인 정렬을 수행하고, 이는 안정적인 학습 프로세스를 촉진합니다. 이후, 경량의 학습 가능 모듈을 도입하여 대략적 정렬에서 깊이 및 포즈 추정을 정제하여 3D 재구성과 뷰 합성의 품질을 개선합니다.

- **Performance Highlights**: 대규모 실제 실내 및 실외 데이터셋에서 PF3plat는 모든 벤치마크에서 새로운 최첨단 성능을 달성하였으며, 종합적인 개별 분석 결과는 우리의 설계 선택을 검증하였습니다. 이 프레임워크는 희소하고 비포즈 이미지를 활용하여 빠르고 고성능의 3D 재구성과 새로운 뷰 합성을 가능하게 합니다.



### Hyperspectral Imaging-Based Perception in Autonomous Driving Scenarios: Benchmarking Baseline Semantic Segmentation Models (https://arxiv.org/abs/2410.22101)
Comments:
          Accepted at IEEE WHISPERS 2024

- **What's New**: 최근 Hyperpectral Imaging (HSI) 기술이 전통적인 RGB 이미징에 비해 원거리 감지, 농업, 의료 분야에서의 장점으로 주목 받고 있습니다. 특히, Advanced Driving Assistance Systems (ADAS) 인식 향상을 위한 연구가 이루어졌으며, 다양한 HSI 데이터셋이 공개되었습니다.

- **Technical Details**: 본 연구에서는 다양한 주석(annotation)이 포함된 HSI 데이터셋을 기반으로 DeepLab v3+, HRNet, PSPNet, U-Net 및 두 가지 변형인 Coordinate Attention (UNet-CA)와 Convolutional Block-Attention Module (UNet-CBAM)와 같은 네 가지 딥러닝 기반의 Semantic Segmentation Models (SSM)에 대해 평가합니다. 데이터셋의 다양한 공간적(spatial) 및 스펙트럴(spectral) 차원을 처리하도록 모델 아키텍처를 조정하였으며, 클래스 가중 손실 함수(class-weighted loss function)를 사용하여 학습하였습니다.

- **Performance Highlights**: UNet-CBAM 모델이 채널별 특징(channel-wise features)을 추출하여 다른 SSM들보다 뛰어난 성능을 보였으며, 향상된 시맨틱 세그멘테이션을 위한 스펙트럴 정보를 활용할 가능성이 있는 것으로 나타났습니다. 그러나 현재 HSI 데이터셋의 제한된 크기, 높은 클래스 불균형, 미세 주석 부족 등은 ADAS 응용을 위한 안정적인 SSM 개발에 주요한 제약으로 남아 있습니다.



### TractShapeNet: Efficient Multi-Shape Learning with 3D Tractography Point Clouds (https://arxiv.org/abs/2410.22099)
Comments:
          10 pages, 2 figures, 4 tables. This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구에서는 뇌의 백질 연결의 기하학적 형태 측정값을 깊이 있는 학습 모델을 통해 계산할 수 있는 가능성을 탐구합니다. 새로운 프레임워크인 TractShapeNet을 소개하며, 이는 트랙토그래피의 포인트 클라우드 표현을 활용하여 다섯 가지 형태 측정값(길이, 범위, 부피, 총 표면적, 불규칙성)을 계산합니다.

- **Technical Details**: TractShapeNet은 1065명의 건강한 젊은 성인으로 구성된 대규모 데이터셋을 사용하여 평가되며, 저자들은 딥러닝 모델이 전통적인 DSI-Studio 도구에 비해 빠르고 효율적으로 형태 측정값을 계산할 수 있음을 보여줍니다. 이 모델은 마치 포인트 클라우드를 처리하는 것처럼 각 섬유 클러스터를 포인트 클라우드 형태로 나타내며, Siamese 네트워크 아키텍처를 기반으로 하여 구조적 정보를 보존하면서 최적화합니다.

- **Performance Highlights**: 실험 결과, TractShapeNet은 Pearson 상관 계수 및 정규화된 오류 메트릭에서 기존의 포인트 클라우드 기반 신경망 모델을 초월하는 성능을 보였습니다. 또한, TractShapeNet에서 계산된 형태 측정값이 DSI-Studio 방법과 유사한 성과를 내면서 언어 인지 예측 작업에서도 잘 작동함을 입증했습니다.



### HRPVT: High-Resolution Pyramid Vision Transformer for medium and small-scale human pose estimation (https://arxiv.org/abs/2410.22079)
Comments:
          under review

- **What's New**: 본 논문에서는 중소 규모의 인체 포즈 추정을 위한 HRPVT 모델을 제안합니다. 이 모델은 PVT v2를 기반으로 하여 고해상도 피쳐 맵에서 CNN의 고유한 유도 편향을 통합하여 중소 규모 인체의 키포인트를 보다 정확히 로컬라이즈하는 성능을 향상시킵니다. 또한, 기존의 heatmap 기반 방법 대신 SimCC 접근법을 사용하여 비싼 업샘플링 레이어를 제거하고 계산 자원을 보다 효율적으로 분배하는 방식을 도입합니다.

- **Technical Details**: HRPVT 모델은 High-Resolution Pyramid Module (HRPM)을 포함하고 있으며, 이 모듈은 HRPM v1과 HRPM v2로 구성됩니다. 이 두 서브모듈은 CNN의 스케일 불변성과 지역성을 모델링하여 고해상도 피쳐 맵에서 키포인트 로컬라이제이션을 개선합니다. HRPM v2의 삽입 위치에 따라 Layer-wise Insertion과 Stage-wise Insertion의 두 가지 삽입 전략이 개발되어 다양한 복잡성의 베이스라인을 지원합니다.

- **Performance Highlights**: HRPVT 모델은 MS COCO와 MPII 데이터 세트에서 뛰어난 성능을 보여주었으며, MS COCO 데이터 세트에서 HRNet-W48에 비해 40%의 파라미터 수와 37%의 GFLOPS로도 더 나은 결과를 달성했습니다.



### FreeGaussian: Guidance-free Controllable 3D Gaussian Splats with Flow Derivatives (https://arxiv.org/abs/2410.22070)
- **What's New**: 본 논문에서는 수동 주석 없이 동적으로 제어 가능한 Gaussian splats를 복원하는 새로운 방법인 FreeGaussian을 소개합니다. 이 방법은 광학 흐름(optical flow)과 카메라 모션(camera motion)을 기반으로 하여 동적 Gaussian 운동을 수학적으로 유도합니다.

- **Technical Details**: FreeGaussian은 2D 이미지의 광학 흐름과 카메라에 의해 유도된 장면 흐름을 기반으로 동적 Gaussian을 직접 도출하고, 이를 통해 상호작용하는 동적 3D Gaussian 구조를 추적합니다. 또한, 3D 구형 벡터 제어 방식(3D spherical vector controlling scheme)을 도입하여 복잡한 1D 제어 신호 계산을 피하고, Gaussian 동적 모션을 보다 쉽게 모델링할 수 있게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FreeGaussian은 새로운 관점 합성과 장면 제어에서 기존 방법보다 우수한 성능을 보여주며, 수동 주석 없이도 상호작용 가능한 콘텐츠를 보다 정확하고 효율적으로 모델링할 수 있음을 입증하였습니다.



### Benchmarking Human and Automated Prompting in the Segment Anything Mod (https://arxiv.org/abs/2410.22048)
- **What's New**: 본 논문에서는 Segment Anything Model (SAM)의 이미지 세분화(image segmentation) 작업에 있어 자동화된 포인트 프롬프트(point prompt) 선택 전략의 효과성을 탐구하고 있습니다. 연구 결과, 인간이 생성한 프롬프트가 자동화된 방법에 비해 평균적으로 약 29% 높은 세분화 성능을 보이며, 프롬프트 점 간 거리와 같은 해석 가능한 요소가 세분화 성능에 미치는 영향 또한 연구했습니다.

- **Technical Details**: PointPrompt라는 최신 비주얼 프롬프트 데이터셋을 사용하여, 인간과 자동화 방법 간의 프롬프트 생성 차이를 3가지 벤치마크 태스크를 통해 평가했습니다. 태스크 1에서는 인간 프롬프트와 자동화된 방법을 비교했고, 태스크 2에서는 SAM의 프롬프트 인코더를 특정 전략에 맞춰 파인튜닝(finetuning)하여 성능 차이를 줄일 수 있음을 보여주었습니다. 마지막으로, 태스크 3에서는 간단한 특성이 성능 예측에 도움이 된다는 것을 입증했습니다.

- **Performance Highlights**: 실험 결과, 기존의 자동화된 방법은 인간 프롬프트에 비해 평균적으로 29% 낮은 성능을 보였습니다. 자동화된 방법의 성능을 향상시키기 위한 파인튜닝 접근법을 통해 최대 68%까지 성능을 개선할 수 있음을 발견했습니다. 또한, 포함(inclusion) 포인트가 성능 차이에 더 큰 기여를 한다는 점을 확인했습니다.



### Feature distribution Adaptation Network for Speech Emotion Recognition (https://arxiv.org/abs/2410.22023)
- **What's New**: 본 논문에서는 다중 모달 음성 감정 인식 문제를 해결하기 위한 새로운 심층 유도 전이 학습 프레임워크인 'feature distribution adaptation network (FDAN)'을 제안합니다. 이 방법은 시각 및 오디오 피처 분포를 정렬하여 감정의 일관된 표현을 얻고, 이를 통해 음성 감정 인식 성능을 개선하는데 초점을 맞추고 있습니다.

- **Technical Details**: 이 모델은 사전 훈련된 ResNet-34를 사용하여 얼굴 표정 이미지와 음향 Mel 스펙트로그램의 피처를 추출하고, 그 뒤에 교차 주의 메커니즘을 도입하여 다중 모달 피처의 내재적 유사성 관계를 모델링합니다. 피드포워드 네트워크를 사용하여 효율적으로 다중 모달 피처 분포 적응을 수행하며, 이는 로컬 최대 평균 차이 손실(local maximum mean discrepancy loss)을 통해 확장됩니다.

- **Performance Highlights**: 실험은 두 개의 벤치마크 데이터셋에서 수행되었으며, 결과는 제안된 모델이 기존 방법들과 비교하여 우수한 성능을 달성할 수 있음을 보여줍니다.



### A Machine Learning-Based Secure Face Verification Scheme and Its Applications to Digital Surveillanc (https://arxiv.org/abs/2410.21993)
Comments:
          accepted by International Conference on Digital Image and Signal Processing (DISP) 2019

- **What's New**: 이 논문은 얼굴 이미지의 개인 식별 보호와 보안 강화를 위한 새로운 얼굴 인증 시스템을 제안합니다. 기존 시스템의 단점으로 지적된 개인 정보 보호 문제를 해결하기 위해, DeepID2을 이용한 CNN 모델과 EM 알고리즘을 적용하여 안전하게 얼굴 인증을 수행합니다.

- **Technical Details**: 논문에서 제안한 방법은 DeepID2 (Deep Identification-verification network version two)라는 심층 신경망을 활용하여 얼굴 이미지의 특징을 추출하고, EM (Expectation-Maximization) 알고리즘을 사용하여 두 이미지가 동일 인물로부터 촬영되었는지를 검증합니다. 또한, 개인 정보를 보호하기 위해 동형 암호화 (homomorphic encryption) 기술을 적용하여 암호화된 상태에서 얼굴 특징을 계산합니다.

- **Performance Highlights**: 연구팀은 세 가지 개인 정보 보호 수준에 따라 지역 사회의 출입 통제를 위한 세 가지 얼굴 인증 시스템을 개발했습니다. 실험 결과는 제안된 시스템의 실용성을 입증하며, 암호화된 데이터에 대한 효율적인 연산이 가능함을 보여줍니다.



### From Explicit Rules to Implicit Reasoning in an Interpretable Violence Monitoring System (https://arxiv.org/abs/2410.21991)
Comments:
          12 pages,7 figures

- **What's New**: 이 논문은 violence surveillance (폭력 감시) 작업을 위한 새로운 패러다임인 Rule base Violence monitoring (RuleVM)를 제안합니다. 기존의 블랙박스 시스템 대신, 명시적 지식이 포함된 해석 가능한 모델을 설계하는 방안을 모색합니다.

- **Technical Details**: RuleVM은 이미지와 텍스트에 대해 서로 다른 디자인을 가진 dual-branch structure(이중 가지 구조)를 사용합니다. implicit branch(암묵적 가지)는 시각적 특징만을 활용하여 coarse-grained binary classification(조잡한 이진 분류)을 수행하고, explicit branch(명시적 가지)는 언어-이미지 정렬을 통해 fine-grained classification(세밀한 분류)을 실행합니다.

- **Performance Highlights**: RuleCLIP은 다양한 벤치마크에서 실험을 통해 조잡한 분류와 세밀한 분류 모두에서 가장 뛰어난 성능을 보여주며, 기존의 최첨단 방법들을 크게 능가했습니다. 또한, 해석 가능성 실험에서 사람의 수가 증가할수록 폭력 행동의 위험 수준이 상승한다는 흥미로운 규칙이 발견되었습니다.



### A Survey on RGB, 3D, and Multimodal Approaches for Unsupervised Industrial Anomaly Detection (https://arxiv.org/abs/2410.21982)
Comments:
          28 pages, 18 figures

- **What's New**: 본 논문은 Unsupervised Industrial Anomaly Detection (UIAD) 분야의 RGB, 3D 및 Multimodal 접근 방식을 체계적으로 분류하고, 최근의 연구 동향을 포괄적으로 요약하였습니다. 특히, 기존 리뷰에서 다루지 않은 3D 및 Multimodal 설정에 대한 분석을 포함하고 있습니다.

- **Technical Details**: 논문에서는 UIAD 기술의 발전을 다루며, 전통적인 감지 방법과 비교할 때, 딥 러닝 기반의 비지도 방식이 더 적합하다는 이유를 제시합니다. 특히 RGB, 3D 포인트 클라우드 및 복합 모달 정보를 이용한 감지 방법을 비교하여 그 장점을 설명합니다. UIAD의 주요 도전 과제와 미래 연구 방향도 제안하고 있습니다.

- **Performance Highlights**: UIAD 기술은 복잡한 산업 환경에서 여러 종류의 이상 탐지의 정확성을 향상시키며, 다양한 모달 정보를 통합하여 탐지 성능의 안정성을 증가시키는 효과가 있습니다. 특히 MVTec 3D-AD 데이터셋은 Multimodal 이상 탐지 방법의 발전에 크게 기여하였으며, 다양한 산업 시스템에서의 활용 가능성을 제시하고 있습니다.



### BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays (https://arxiv.org/abs/2410.21969)
Comments:
          Accepted to NeurIPS24 Datasets and Benchmarks Track

- **What's New**: 의료 비전-언어 사전학습(MedVLP)을 위한 새로운 벤치마크 프레임워크인 BenchX를 제안합니다. 이 프레임워크는 공공 흉부 X-레이 데이터셋을 이용해 MedVLP 방법 간의 체계적인 분석과 대결을 가능하게 합니다.

- **Technical Details**: BenchX는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 아홉 개의 데이터셋과 네 개의 의료 작업을 포괄하는 포괄적인 데이터셋; 2) 데이터 전처리, 학습-테스트 분할 및 파라미터 선택을 표준화하는 벤치마크 슈트; 3) 분류, 분할 및 보고서 생성을 위한 일관된 작업 적응을 위한 통합된 미세 조정 프로토콜.

- **Performance Highlights**: BenchX를 활용하여 아홉 가지 최신 MedVLP 방법의 기준선을 설정했습니다. 특히, 일부 초기 MedVLP 방법의 성능이 적절한 훈련 전략을 통해 최신 방법을 초과할 수 있음을 발견했습니다.



### PrefPaint: Aligning Image Inpainting Diffusion Model with Human Preferenc (https://arxiv.org/abs/2410.21966)
- **What's New**: 이 논문에서는 이미지 인페인팅(image inpainting)을 위해 확산 모델(diffusion models)을 인간의 미적 기준(aesthetic standards)과 정렬하기 위한 시도를 최초로 진행하였으며, 강화 학습(reinforcement learning) 프레임워크를 통해 인페인팅 이미지의 품질과 시각적 매력을 크게 향상시켰습니다.

- **Technical Details**: 우리는 약 51,000개의 인간 선호가 주석(annotation)된 이미지를 포함하는 데이터셋을 구축하여 보상 모델(reward model)을 훈련했습니다. 기존의 쌍이미지와의 편차(divergence)를 직접 측정하는 대신, 강화 학습 과정을 통해 사전 훈련된 확산 모델의 분포를 보상이 높은 방향으로 미세 조정합니다. 또한, 우리는 보상 모델의 오류에 대한 상한을 이론적으로 도출하여 강화 정렬 과정에서 보상 추정의 신뢰도를 중심으로 정확한 정규화를 촉진합니다.

- **Performance Highlights**: 인페인팅 비교 및 이미지 확장(image extension), 3D 재구성(3D reconstruction) 등의 다운스트림 작업에서 진행한 광범위한 실험은 우리의 접근 방식의 효과성을 검증했으며, 최첨단(state-of-the-art) 방법들에 비해 인페인팅 이미지의 인간 선호와의 정렬에서 중요한 개선을 보여주었습니다. 이 연구는 이미지 인페인팅 분야를 발전시킬 뿐만 아니라, 보상 정확성(recognizing reward accuracy) 모델링에 기반한 생성 모델의 반복적 개선(iterative refinement)을 통해 인간의 선호를 통합할 수 있는 프레임워크를 제공합니다.



### FakeFormer: Efficient Vulnerability-Driven Transformers for Generalisable Deepfake Detection (https://arxiv.org/abs/2410.21964)
- **What's New**: 이 논문에서는 Vision Transformers (ViTs)의 deepfake 탐지에서의 저조한 성능 원인을 분석하고, 이를 개선하기 위한 새로운 프레임워크인 FakeFormer를 제안합니다.

- **Technical Details**: 기존의 ViT 아키텍처는 CNN에 비해 제한된 지역적 왜곡 아티팩트 모델링 성능을 보입니다. 이를 해결하기 위해 FakeFormer는 artifact-giving patches를 활용하여 지역적 주의(Learning-based Local Attention, L2-Att) 메커니즘을 통합하여 성능을 개선합니다.

- **Performance Highlights**: FakeFormer는 유명한 데이터셋인 FF++, Celeb-DF, WildDeepfake 등에서 state-of-the-art 성능을 기록하며, 대규모 학습 데이터셋 없이도 뛰어난 일반화 능력과 계산 비용 효율성을 보여줍니다.



### Spatio-temporal Transformers for Action Unit Classification with Event Cameras (https://arxiv.org/abs/2410.21958)
Comments:
          Under review at CVIU. arXiv admin note: substantial text overlap with arXiv:2409.10213

- **What's New**: 이번 연구에서는 Event 카메라를 활용하여 인식된 얼굴 Action Unit의 정확도를 높이기 위한 새로운 Spatiotemporal Vision Transformer 모델을 제안합니다. FACEMORPHIC 데이터셋을 발표하며, 이는 RGB 영상과 Event 스트림으로 구성된 다중 모드 얼굴 데이터셋입니다.

- **Technical Details**: 우리가 제안하는 모델은 Shifted Patch Tokenization (SPT) 및 Locality Self-Attention (LSA)를 활용하여 Event 스트림에서 Action Unit 분류의 정확성을 향상시킬 수 있습니다. 네트워크는 RGB와 Event 카메라를 통해 수집된 데이터의 비디오 수준 및 프레임 수준 교차 도메인 감독을 통하여 훈련됩니다.

- **Performance Highlights**: 제안된 모델은 FACEMORPHIC 데이터셋의 도움으로 Action Unit 분류에서 기존의 방법들보다 뛰어난 성능을 보이며, NEFER 데이터셋 감정 인식 벤치마크에서도 최첨단 성능을 달성했습니다.



### ReMix: Training Generalized Person Re-identification on a Mixture of Data (https://arxiv.org/abs/2410.21938)
Comments:
          Accepted by WACV 2025

- **What's New**: 본 연구는 제한된 라벨이 있는 다중 카메라 데이터와 대규모 비라벨 단일 카메라 데이터를 혼합하여 공동 학습하는 새로운 방법인 ReMix를 제안합니다. 이 방법은 이전의 self-supervised pre-training 접근법의 한계를 극복하고, 더 나은 generalization 능력을 달성하도록 설계되었습니다.

- **Technical Details**: ReMix는 두 가지 유형의 데이터: 라벨이 있는 다중 카메라 데이터와 비라벨 단일 카메라 이미지를 효율적으로 혼합하고 공동 학습하기 위해 새로운 데이터 샘플 전략과 세 가지 새로운 손실 함수 (Instance Loss, Augmentation Loss, Centroids Loss)를 제안합니다. 이러한 손실 함수들은 두 종류의 데이터에 맞게 조정되어 훈련의 질을 높입니다.

- **Performance Highlights**: 실험 결과, ReMix는 cross-dataset 및 multi-source cross-dataset 시나리오에서 기존의 최첨단 방법보다 성능이 우수하며, generalizable person Re-ID에서 높은 generalization 능력을 보여줍니다.



### Structured Analysis and Comparison of Alphabets in Historical Handwritten Ciphers (https://arxiv.org/abs/2410.21913)
Comments:
          Acccepted at ECCV24 Workshop AI4DH

- **What's New**: 이 논문에서는 역사적 암호 문서의 유사성을 분석하는 새로운 방법인 Cipher Similarity Index (CSI) 메트릭을 제안합니다. 이는 비지도 학습 방법을 사용하여 암호 알파벳을 자동으로 식별하고 비교할 수 있게 해줍니다.

- **Technical Details**: CSI 메트릭은 SIFT, 사전 훈련된 임베딩 및 OCR 설명자를 포함한 시각적 특징을 활용하여 암호 문서 쌍의 유사성을 정량화합니다. 본 연구에서는 클러스터링 알고리즘인 k-means를 사용하여 시각적으로 유사한 기호들을 그룹화하고, Shannon Entropy를 통해 클러스터의 이질성을 평가합니다.

- **Performance Highlights**: CSI 메트릭은 기존 문헌보다 적은 의존도로 더 정확한 유사성 측정 값을 생성하며, 비주얼 기반 방법을 통해 암호 문서의 분류 및 분석에서 효과적인 성능을 보입니다.



### Multi-step feature fusion for natural disaster damage assessment on satellite images (https://arxiv.org/abs/2410.21901)
Comments:
          10 pages, for associated Github repository: this https URL

- **What's New**: 이 논문은 자연 재해 후 건축물 상태의 피해를 신속하고 정확하게 평가하기 위해 새로운 다단계 기능 융합 네트워크(multi-step feature fusion network)를 제안합니다. 이 방법은 대규모 위성 이미지에서 재해 이전 및 이후의 데이터를 비교하여 피해 상태를 분류합니다.

- **Technical Details**: 제안된 방법은 특별히 설계된 Fuse Module을 사용하여 CNN 모델이 이미지 쌍을 분석할 수 있도록 조정합니다. 이 모듈은 CNN의 여러 네트워크 레벨 사이에서 수평적 및 수직적 기능 융합을 수행하며, 기존의 CNN 아키텍처에 대한 성능을 개선할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 Vision Transformer 모델의 정확도를 3%포인트 향상시키며, IDA-BD 및 xView2와 같은 대규모 공개 데이터셋에서 검증되었습니다.



### Self-Relaxed Joint Training: Sample Selection for Severity Estimation with Ordinal Noisy Labels (https://arxiv.org/abs/2410.21885)
Comments:
          Accepted at WACV2025

- **What's New**: 본 연구에서는 'ordinal' noisy labels을 이용한 새로운 프레임워크인 self-relaxed joint training을 제안합니다. 이 프레임워크는 청정 샘플 선택 및 이중 네트워크 아키텍처를 사용하여 noisy labels의 부정적 영향을 줄입니다.

- **Technical Details**: 제안된 방법은 noisy hard labels에서 파생된 soft labels를 사용하여 샘플 선택 및 네트워크 훈련의 정확성을 높이며, ordinal 관계를 활용하여 모델을 훈련합니다. 결국 이것은 모델 업데이트에서 잘못된 라벨의 부정적 영향을 감소시킵니다.

- **Performance Highlights**: 제안된 방법은 두 개의 endoscopic ulcerative colitis (UC) 데이터셋과 한 개의 retinal Diabetic Retinopathy (DR) 데이터셋을 사용한 실험에서 다양한 최첨단 방법들을 초월하는 성과를 보여줍니다.



### Advancing Efficient Brain Tumor Multi-Class Classification -- New Insights from the Vision Mamba Model in Transfer Learning (https://arxiv.org/abs/2410.21872)
- **What's New**: 이번 연구는 뇌 종양 분류를 위한 사전 훈련된 모델의 적용을 조사하고, 특히 Mamba 모델을 활용하여 진행되었습니다. 특히, 새로운 네트워크 아키텍처인 Vision Mamba(Vim)를 소개하고 이를 통한 최초의 뇌 종양 분류 적용을 보여주었습니다.

- **Technical Details**: 연구진은 여러 주류 전이 학습(transfer learning) 모델을 미세 조정(fine-tuning)하고 이를 다중 클래스 분류(multi-class classification) 문제에 적용하였습니다. Vim 모델은 독립 테스트 세트에서 100% 분류 정확성을 달성하였으며, 기존의 최신 모델들과 비교했을 때 경량( lightweight) 및 효율적(efficient)으로 나타났습니다.

- **Performance Highlights**: Vim 모델은 의료 이미징(medical imaging) 분야에서 레이블이 붙은 데이터가 제한적인 상황에서도 전이 학습의 우수성을 입증하며, Tumor classification를 위한 새로운 아이디어를 제공합니다. 이 연구의 전이 학습 기반 프레임워크는 다른 의료 이미징 분류 문제에도 널리 적용될 수 있습니다.



### HRGR: Enhancing Image Manipulation Detection via Hierarchical Region-aware Graph Reasoning (https://arxiv.org/abs/2410.21861)
- **What's New**: 이번 논문에서는 기존의 격자 기반 방법 대신에, 내용 일관성(feature coherence) 있는 비정형적 특성 영역을 기반으로 한 새로운 방법인 Hierarchical Region-aware Graph Reasoning (HRGR)을 제안합니다.

- **Technical Details**: HRGR은 Differentiable Feature Partition 전략을 활용하여 비overlapped한 내용 일관성을 가진 영역(Feature Region)을 생성합니다. 이 영역들은 계층적으로 구성된 그래프 구조를 통해 서로의 관계를 모델링하여 manipulations를 탐지하도록 설계되었습니다. 이 방법은 end-to-end 방식으로 기존 네트워크에 통합될 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과는 HRGR 방법이 이미지 조작 탐지에서 효과적임을 보여주며, 기존 아키텍처에 쉽게 추가할 수 있는 뛰어난 잠재력을 지니고 있음을 입증합니다.



### Micro-Structures Graph-Based Point Cloud Registration for Balancing Efficiency and Accuracy (https://arxiv.org/abs/2410.21857)
- **What's New**: 본 논문에서는 포인트 클라우드 등록(Point Cloud Registration, PCR)을 위한 새로운 마이크로 구조 그래프 기반의 전역 포인트 클라우드 등록 방법을 제안합니다. 이 방법은 두 단계로 나뉘어 있습니다.

- **Technical Details**: 1) Coarse registration (CR): 마이크로 구조를 포함한 그래프를 개발하고, 효율적인 그래프 기반 계층 전략을 사용하여 외부값을 제거하고 최대 합의 집합을 얻습니다. 외부값 프로세스에 대해 강력한 GNC-Welsch-estimator를 최적화하는 데 사용합니다. 2) Fine registration (FR): 옥트리 접근법을 활용하여 마이크로 구조의 평면 특성을 적응형으로 검색하고, 점-평면 간의 거리를 최소화하여 더 정밀한 지역 정렬을 달성합니다.

- **Performance Highlights**: 실제 데이터에 대한 광범위한 실험 결과, 제안된 방법은 3DMatch 및 ETH 데이터셋에서 최신 방법들과 비교하여 높은 정확성 메트릭을 달성하고, 시간 비용을 최소한 1/3까지 줄이는 성과를 보였습니다.



### Diffusion as Reasoning: Enhancing Object Goal Navigation with LLM-Biased Diffusion Mod (https://arxiv.org/abs/2410.21842)
- **What's New**: 본 논문에서는 Object Goal Navigation (ObjectNav) 작업을 해결하기 위해 새로운 접근 방식인 Diffusion as Reasoning (DAR) 기법을 제안합니다. 이 방법은 에이전트의 이동 중 누적된 메모리를 기반으로 하는 통계적 분포 패턴을 학습하는 diffusion model을 훈련하여 보이지 않는 환경에서의 목표 객체 위치 추정을 가능하게 합니다.

- **Technical Details**: DAR는 Denoising Diffusion Probabilistic Models (DDPM)을 활용하여 에이전트의 환경 기억을 조건으로 삼아 알려지지 않은 지역의 신뢰할 수 있는 의미론적 콘텐츠를 생성합니다. DDPM은 RGB 데이터 대신 클래스 채널의 의미론적 맵으로 학습시키며, LLM의 일반 지식을 활용하여 생성화의 일반화를 향상시킵니다.

- **Performance Highlights**: Gibson과 MP3D 환경에서 실험 결과, DAR 기법이 기존의 일반적인 방법보다 높은 성공률을 나타내어, 목표 객체 위치 추정 및 효율적인 내비게이션을 가능하게 함을 보여주었습니다.



### Enhanced Survival Prediction in Head and Neck Cancer Using Convolutional Block Attention and Multimodal Data Fusion (https://arxiv.org/abs/2410.21831)
Comments:
          Accepted to [ACCV 2024 Workshop]

- **What's New**: 이번 연구는 두 가지 이미징 모달리티인 CT와 PET를 활용한 딥 러닝 기반 생존 예측 모델을 제안합니다. 기존의 Cox 비례 위험 모델과 같은 전통적인 방법들이 복잡한 다중 모달 데이터를 다루는 데에 한계가 있음에 따라, 본 연구에서는 Convolutional Block Attention Module (CBAM)과 다중 모달 데이터 융합 레이어를 통합했습니다.

- **Technical Details**: 본 연구는 HECKTOR와 HEAD-NECK-RADIOMICS-HN1 데이터셋을 활용하여 CT 및 PET 이미징에서의 기능을 추출하고, 이를 완전한 매개변수 이산 시간 생존 모델을 통해 예측하는 방법론을 사용합니다. 이 과정에서 다중 모달 데이터를 통합하여 Compact feature representation을 생성하고, 전통적인 생존 모델의 한계를 극복할 수 있는 유연한 위험 함수를 제공합니다.

- **Performance Highlights**: 본 연구의 딥 러닝 모델은 기존의 통계적 및 기계 학습 모델에 비해 생존 예측 정확도를 유의미하게 향상시켰습니다. HECKTOR 및 HEAD-NECK-RADIOMICS-HN1 데이터셋에서의 실험 결과는 개인 맞춤형 치료 계획을 위한 강력한 도구로서의 가능성을 보여줍니다.



### Volumetric Conditioning Module to Control Pretrained Diffusion Models for 3D Medical Images (https://arxiv.org/abs/2410.21826)
Comments:
          17 pages, 18 figures, accepted @ WACV 2025

- **What's New**: 본 논문은 3D 의료 이미지 생성을 위한 새로운 공간 제어 방법, 즉 Volumetric Conditioning Module (VCM)을 제안합니다. VCM은 비대칭 U-Net 아키텍처를 활용하여 복잡한 3D 조건으로부터 정보를 효과적으로 인코딩하며, 적은 훈련 데이터로 조건부 생성을 가능하게 합니다.

- **Technical Details**: VCM은 훈련된 대규모 diffusion 모델의 매개변수를 동결하며, 각 시간 단계에서 노이즈 제거 과정을 안내하는 공간 제어를 학습합니다. 다양한 입력 조건으로부터 세밀한 안내를 제공하며, 단일 및 다중 모드 조건에서 실험을 수행하여 데이터 세트 크기에 따라 모델의 효율성을 평가합니다.

- **Performance Highlights**: 실험 결과, VCM은 조건부 생성에서 효과적이며 더 적은 훈련 데이터와 계산 리소스를 요구합니다. VCM은 T1w 뇌 MRI와 같은 여러 응용 프로그램에 효과적으로 적용될 수 있어 의료 이미징 분야의 데이터 증강, 초해상도 및 변환 작업에서 활용 가능성을 보여줍니다.



### PK-YOLO: Pretrained Knowledge Guided YOLO for Brain Tumor Detection in Multiplanar MRI Slices (https://arxiv.org/abs/2410.21822)
Comments:
          Accepted by WACV 2025

- **What's New**: 이 논문에서는 다중 평면 (multiplane) 자기공명영상 (MRI) 슬라이스에서 뇌종양 탐지를 개선하기 위해 Pretrained Knowledge (PK)를 통합한 YOLO(You Only Look Once) 기반 탐지 모델인 PK-YOLO를 제안합니다.

- **Technical Details**: PK-YOLO는 경량의 합성곱 신경망 (convolutional neural network)을 기반으로 한 백본과 소형 객체 탐지를 개선하기 위한 회귀 손실 함수 (regression loss function)를 포함합니다. 이 모델은 각각의 MRI 슬라이스에서 특징 전이 가능성을 높이고, 학습된 도메인 지식 (domain knowledge base)이 탐지 성능을 향상시킵니다.

- **Performance Highlights**: PK-YOLO는 기존의 최신 YOLO 계열 및 DETR 계열 객체 탐지기와 비교했을 때 경쟁력 있는 성능을 보여주며, 2차원 다중 평면 MRI 슬라이스에서의 소형 뇌종양 탐지 성능이 크게 개선되었습니다.



### SAM-Swin: SAM-Driven Dual-Swin Transformers with Adaptive Lesion Enhancement for Laryngo-Pharyngeal Tumor Detection (https://arxiv.org/abs/2410.21813)
- **What's New**: SAM-Swin은 두 가지 주요 혁신을 통해 인후-목 Cancer (LPC) 탐지의 새로운 가능성을 제시합니다. 첫째, Segment Anything Model 2 (SAM2)를 통합하여 정밀한 병변 세분화를 실현하고, 둘째, MS-LAEM 모듈로 다양한 스케일에서의 보완적 특성을 효과적으로 강화합니다.

- **Technical Details**: SAM-Swin은 SAM2를 활용하여 병변 위치를 정확하게 파악하고, Whole Image Branch (WIB) 및 Lesion Region Branch (LRB)를 통해 다양한 스케일의 정보를 통합하여 최적화된 특성 학습이 가능합니다. Multi-scale Class-aware Guidance (CAG) 손실 함수는 클래스별 특성 추출에 대한 목표 감독을 제공합니다.

- **Performance Highlights**: SAM-Swin은 FAHSYSU, SAHSYSU, NHSMU의 세 가지 데이터셋에 대해 광범위한 실험을 수행하였으며 기존의 최첨단 방법들을 일관되게 초과하는 성능을 기록하였습니다. 이를 통해 SAM-Swin은 LPC 탐지를 위한 혁신적인 접근 방식임을 입증하였습니다.



### A Fresh Look at Generalized Category Discovery through Non-negative Matrix Factorization (https://arxiv.org/abs/2410.21807)
Comments:
          13 pages, 8 figures,Submitted to IEEE Transactions on Image Processing

- **What's New**: 논문은 비부정적인 일반화 범주 발견(Non-Negative Generalized Category Discovery, NN-GCD) 프레임워크를 제안하여 기존의 GCD 접근법의 한계를 극복하고, 코사인 유사성에 기반한 공존 행렬의 최적화를 다룰 수 있는 새로운 방법론을 제공합니다. 이 프레임워크는 대칭 비부정적 행렬 분해(Symmetric Non-negative Matrix Factorization, SNMF)를 수학적 매개체로 사용하여 최적의 K-평균과 SNMF 간의 동등성을 증명합니다.

- **Technical Details**: NN-GCD는 비부정적 활성화 뉴런 메커니즘과 NMF NCE 손실을 도입하며, 이를 통해 GCD 모델이 글로벌 최적 지역으로 수렴할 수 있도록 돕습니다. 이론적으로 K-평균 클러스터링을 SNMF와 비부정적 대조 학습(Non-negative Contrastive Learning, NCL) 최적화와 동등한 문제로 재구성합니다. 또한 혼합 희소 정규화 방법을 제안하여 A의 희소성 제약을 부과합니다.

- **Performance Highlights**: NN-GCD는 일반화 범주 발견(GCD) 벤치마크에서 최첨단 방법을 초과하는 성능을 보여주며, Semantic Shift Benchmark에서 평균 66.1%의 정확도를 기록하여 이전 방법들보다 4.7% 향상된 결과를 나타냈습니다.



### Text-Guided Attention is All You Need for Zero-Shot Robustness in Vision-Language Models (https://arxiv.org/abs/2410.21802)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 Text-Guided Attention for Zero-Shot Robustness (TGA-ZSR)라는 새로운 전략을 제안합니다. 이 방법은 CLIP 모델의 일반화 능력을 유지하면서 적대적 공격에 대한 강인성을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: TGA-ZSR는 두 가지 주요 구성 요소로 이루어져 있습니다: Attention Refinement 모듈과 Attention-based Model Constraint 모듈입니다. Attention Refinement 모듈은 원래의 깨끗한 예제에서 얻은 text-guided attention과 적대적 예제에서 얻은 text-guided attention을 정렬시켜 모델의 강인성을 높이는 역할을 합니다. Attention-based Model Constraint 모듈은 깨끗한 예제를 이용하여 두 모델의 text-guided attention을 획득하고, 이는 오염된 데이터에서 모델의 성능을 유지하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, TGA-ZSR 방법은 현재의 최신 기술들에 비해 9.58%의 향상된 zero-shot 강인성을 보여주며, 16개 데이터셋에서 검증되었습니다.



### HairDiffusion: Vivid Multi-Colored Hair Editing via Latent Diffusion (https://arxiv.org/abs/2410.21789)
- **What's New**: 본 논문은 Latent Diffusion Models (LDMs)를 활용하여 헤어 스타일 수정 작업을 위한 새로운 접근 방식을 제안합니다. 특히 Multi-stage Hairstyle Blend (MHB) 방법을 도입하여 헤어 색상과 스타일의 제어를 확실히 분리하여 다채로운 헤어 스타일을 편집할 수 있습니다.

- **Technical Details**: 제안된 방식은 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 머리와 무관한 마스크(hair-agnostic masks)를 얻어 얼굴 표현을 독립적으로 획득하고, 두 번째 단계에서는 참조 이미지를 사용하여 헤어 색상을 전송합니다. 이 과정에서 사전 훈련된 워핑 모듈(‘warping module’)을 사용하여 원본 이미지와 참조 이미지의 머리를 맞추어 헤어 색상 정보(‘color proxy’)를 확보합니다.

- **Performance Highlights**: 저자들은 정량적 및 정성적 평가를 통해 제안된 방법이 텍스트 설명과 참조 이미지를 바탕으로 복잡한 다채로운 헤어 스타일 편집에서 우수한 성능을 보임을 입증하였습니다. 얼굴 속성의 보존을 강조하면서도 다양한 헤어 색상 변환을 동시에 처리할 수 있습니다.



### Fast-OMRA: Fast Online Motion Resolution Adaptation for Neural B-Frame Coding (https://arxiv.org/abs/2410.21763)
- **What's New**: 본 논문에서는 B-frame 코덱의 도메인 변환 문제를 해결하기 위한 경량화된 분류기를 도입합니다. 이 분류기는 모션 추정을 위한 최적 다운샘플링 계수를 예측하며, 기존의 완전 탐색 방법에 비해 계산 복잡도를 크게 줄입니다.

- **Technical Details**: 제안된 두 가지 분류기, Bi-Class Fast-OMRA와 Mu-Class Fast-OMRA는 각각 이진 및 다중 클래스 분류기로, 간단한 상태 신호를 관찰하여 다운샘플링 계수를 결정합니다. Bi-Class Fast-OMRA는 고해상도와 저해상도의 모션 추정을 구분하며, 다중 클래스 분류기는 다양한 다운샘플링 계수의 rate-distortion 비용을 포함한 소프트 라벨로 학습됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 완전 탐색 방법과 비슷한 코딩 성능을 보이면서도 계산 복잡도를 현저히 줄임을 확인하였습니다.



### IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models (https://arxiv.org/abs/2410.21759)
Comments:
          Technical Report

- **What's New**: 본 논문에서는 IntLoRA를 제안하여 텍스트-이미지 확산 모델의 효율성을 높이는 방법을 소개합니다. 이 방법은 양자화된 모델을 적응시키기 위해 정수형(low-rank) 파라미터를 활용합니다.

- **Technical Details**: IntLoRA는 세 가지 주요 이점을 제공합니다: (i) 미세 조정을 위해 사전 훈련된 가중치는 양자화되어 메모리 사용량을 줄인다; (ii) 저장을 위해 사전 훈련된 가중치와 저차원 가중치 모두 INT 형식으로 저장되어 디스크 공간을 절약할 수 있다; (iii) 추론 과정에서 IntLoRA 가중치는 효율적인 정수 곱셈이나 비트 시프트를 통해 양자화된 사전 훈련 가중치에 자연스럽게 통합될 수 있다. 이 과정에서 추가적인 후처리 양자화가 필요 없다.

- **Performance Highlights**: IntLoRA는 광범위한 실험을 통해 기존의 LoRA와 동등하거나 더 나은 성능을 달성할 수 있으며, 효율성 또한 상당히 개선되었습니다.



### DOFS: A Real-world 3D Deformable Object Dataset with Full Spatial Information for Dynamics Model Learning (https://arxiv.org/abs/2410.21758)
Comments:
          5 pages, 6 figures, 2024 CoRL Workshop on Learning Robot Fine and Dexterous Manipulation: Perception and Control

- **What's New**: 본 연구에서는 3D 탄성-플라스틱 객체(Deformable Objects, DOs)의 파일럿 데이터셋인 DOFS를 제안합니다. 이 데이터셋은 새로운 저비용 데이터 수집 플랫폼을 통해 모든 공간 정보를 포함하여 수집되었습니다. 이 플랫폼은 투명한 운영 평면을 사용하여 3D 변형 메쉬, 액션, RGB-D 이미지 및 포인트 클라우드를 포함하는 데이터를 제공합니다.

- **Technical Details**: DOFS 데이터셋을 구축하기 위해 알루미늄 프레임과 투명 아크릴 보드를 포함한 새로운 데이터 수집 플랫폼을 설계하였습니다. 이 시스템은 6개의 Intel RealSense D435i 카메라로 구성되어 있으며, 상단과 하단에서 다양한 각도로 다중 시점 RGB-D 이미지를 캡처하여 밀집 포인트 클라우드를 생성합니다. 이 데이터를 사용하여 Iterative Closest Point (ICP) 알고리즘을 이용해 포인트 클라우드를 정합(Registration)하여 전체 공간 정보를 확보합니다.

- **Performance Highlights**: 본 연구에서 구축한 DOFS 데이터셋을 바탕으로 훈련한 동역학 모델은 고무 수젯 모형의 변형 상태를 잘 예측할 수 있음을 보여주었습니다. 현재 DOFS 데이터셋은 150개의 핀치 작업으로 구성되어 있으며, 각 프레임은 3D 변형 메쉬, 두 손가락의 3D 위치, 다중 시점 RGB-D 이미지 및 의미론적으로 태깅된 3D 점유 데이터를 포함합니다. 이러한 데이터는 로봇의 정밀하고 섬세한 조작 연구에 큰 기여를 할 것으로 기대됩니다.



### Memory-Efficient Point Cloud Registration via Overlapping Region Sampling (https://arxiv.org/abs/2410.21753)
Comments:
          accepted for IEEE International Conference on Visual Communications and Image Processing 2024 (VCIP2024)

- **What's New**: 최근 심층 학습의 발전으로 3D 포인트 클라우드 등록 기술이 향상되었지만, 이에 따라 GPU 메모리 사용량도 증가하고 있습니다. 본 논문은 메모리 사용량을 줄이면서도 정확도를 유지할 수 있는 오버랩 지역 샘플링 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 k-최근접 이웃(kNN) 기반의 포인트 압축 메커니즘을 활용하여 다층 퍼셉트론(MLP) 및 트랜스포머 아키텍처와 함께 오버랩 지역을 추정하고, 그 지역에서 집중적으로 샘플링합니다. 이 방법을 통해 등록 정확도를 유지하면서 메모리 사용량을 효율적으로 줄이고 있습니다.

- **Performance Highlights**: 3DMatch 및 3DLoMatch 데이터셋 평가에서 제안된 방법은 다른 샘플링 방법에 비해 등록 리콜 성능이 뛰어났습니다. 특히 GPU 메모리 사용량이 낮을 때 더 좋은 성능을 보였으며, 3DMatch에서는 33% 메모리 사용량 감소로 94%의 리콜을 달성했습니다. 이는 자원 제약 환경에서 대규모 포인트 클라우드 등록을 효율적으로 수행할 수 있게 합니다.



### MotionGPT-2: A General-Purpose Motion-Language Model for Motion Generation and Understanding (https://arxiv.org/abs/2410.21747)
- **What's New**: MotionGPT-2는 다양한 모션 관련 작업을 처리할 수 있는 통합된 Large Motion-Language Model (LMLM)으로서 기존의 한계를 극복합니다. 이는 다중 모션 통제 신호를 수용하고, 텍스트 및 단일 키프레임 포즈와 같은 여러 입력을 통합하여 인체 운동을 생성할 수 있게 합니다.

- **Technical Details**: MotionGPT-2는 Vector Quantized Variational-AutoEncoder (VQ-VAE)와 Part-Aware VQ-VAE 구조를 사용하여 정밀한 인체 운동 토큰화를 실현합니다. 이는 개별 손 동작까지 학습할 수 있는 두 단계 디스크리트 코드북을 통해 인체의 세밀한 움직임을 캡처합니다. 또한, LLM의 어휘를 보강하며, 멀티모달 사전 훈련 및 지침 조정 접근법을 통해 효율적으로 훈련됩니다.

- **Performance Highlights**: MotionGPT-2는 HumanML3D, KIT-ML, Motion-X 데이터셋에서 실험을 통해 다양한 모션 관련 작업에서 강력한 적응성과 효율성을 입증했습니다. 또한 전체 신체 모션 생성 작업에서도 최신 성과 기준을 세우며, 훈련 시간은 기존 방법의 10%에 불과하고, 1%의 추가 파라미터로 경쟁력을 유지합니다.



### EI-Nexus: Towards Unmediated and Flexible Inter-Modality Local Feature Extraction and Matching for Event-Image Data (https://arxiv.org/abs/2410.21743)
Comments:
          Accepted to WACV 2025. The source code and benchmarks will be made publicly available at this https URL

- **What's New**: 이 논문에서는 이벤트 카메라(event cameras)의 데이터와 이미지 데이터를 활용하여 두 가지 모드 특이 키포인트 추출기(keypoint extractors)와 피처 매처(feature matcher)를 통합한 EI-Nexus라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: Local Feature Distillation (LFD) 기법을 통해 이미지 추출기로부터 이벤트 추출기로 뷰포인트 일관성을 전달합니다. 또한 Context Aggregation (CA)을 활용하여 피처 매칭에서의 성능 향상을 도모하며, MVSEC-RPE와 EC-RPE라는 두 가지 새로운 인터모달리티 피처 매칭 벤치마크를 설정하여 이벤트-이미지 데이터에서의 상대 위치 추정을 평가합니다.

- **Performance Highlights**: 제안된 접근 방식은 명시적인 모달 변환에 의존하는 기존 방법들보다 더 나은 성능을 보이며, MVSEC-RPE와 EC-RPE 벤치마크에서 보다 좋고 최신의 결과들을 달성합니다.



### SS3DM: Benchmarking Street-View Surface Reconstruction with a Synthetic 3D Mesh Datas (https://arxiv.org/abs/2410.21739)
Comments:
          NeurIPS 2024, Track on Datasets and Benchmarks

- **What's New**: 본 연구에서는 디지털 엔터테인먼트 및 자율주행 시뮬레이션과 같은 응용 프로그램에서 중요한 정확한 3D 표면 재구성을 위한 새로운 데이터셋 SS3DM을 소개합니다. 이 데이터셋은 CARLA 시뮬레이터에서 추출한 정밀한 Synthetic Street-view 3D Mesh 모델로 구성되어 있으며, 표면 노멀을 평가할 수 있는 정상 벡터를 포함하고 있습니다.

- **Technical Details**: SS3DM 데이터셋은 촬영된 다수의 RGB 카메라와 LiDAR 센서를 장착한 가상의 차량을 통해 다양한 야외 장면에서 데이터를 수집하여 3D 표면 재구성을 위한 입력 데이터를 생성합니다. SS3DM은 6개의 RGB 카메라와 5개의 LiDAR 센서를 장착하여 360도 주변을 점검하며, 다양한 건축물, 보행자 다리 및 다른 구조물들을 포함한 데이터로 구성되어 있습니다.

- **Performance Highlights**: 이 연구는 SS3DM을 활용하여 최첨단 표면 재구성 방법을 벤치마킹하고, F-score, Chamfer Distance, Normal Chamfer Distance 등 다양한 기하학적 평가 메트릭스를 포함하여 기존 방법의 한계와 거리보기 표면 재구성과 관련된 도전 과제를 분석했습니다.



### DiffSTR: Controlled Diffusion Models for Scene Text Remova (https://arxiv.org/abs/2410.21721)
Comments:
          11 Pages, 6 Figures, 3 Tables

- **What's New**: 본 논문에서는 Scene Text Removal (STR) 문제를 해결하기 위해 ControlNet diffusion 모델을 도입합니다. STR 과제를 inpainting 작업으로 보고, Masked Autoencoder (MAE)를 활용하여 모델의 강건성을 향상시키는 마스크 사전 훈련 파이프라인을 개발하였습니다.

- **Technical Details**: DiffSTR는 diffusion 모델을 이용하여 STR 문제를 해결하는 최초의 연구입니다. 모델은 마스크 이미지 모델링과 같은 기술을 통해 전처리된 인페인팅 이미지를 생성하고, 세그멘테이션 기반 마스크 정제 프레임워크를 통해 마스크의 정확도를 개선합니다. SLIC와 Hierarchical Feature Selection (HFS) 알고리즘을 사용하여 초기 마스크를 반복적으로 정제합니다.

- **Performance Highlights**: SCUT-EnsText 및 SCUT-Syn 데이터셋에서의 실험 결과, DiffSTR는 기존 최첨단 기술보다 현저하게 높은 성능을 보였습니다. 이는 인페인팅 마스크의 예측 품질 향상과 텍스처 정보의 효과적인 활용 덕분입니다.



### Unsupervised Modality Adaptation with Text-to-Image Diffusion Models for Semantic Segmentation (https://arxiv.org/abs/2410.21708)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 다양한 시각적 모달리티(visual modalities)간의 도메인 적응을 위한 새로운 방법인 Modality Adaptation with text-to-image Diffusion Models (MADM)을 제안합니다. 기존 방법들이 이미지 도메인 간의 적응에 치중했던 반면, MADM은 깊이(depth), 적외선(infrared), 이벤트(event)와 같은 다양한 시각적 모달리티를 활용하여 성능을 개선하고자 합니다.

- **Technical Details**: MADM은 두 가지 주요 구성요소로 구성됩니다. 첫째, Diffusion-based Pseudo-Label Generation (DPLG)은 잠재적(noise) 노이즈를 추가하여 pseudo-labels의 안정성을 높이고 정확성을 향상시킵니다. 둘째, Label Palette and Latent Regression (LPLR)은 one-hot 인코딩된 라벨을 RGB 형식으로 변환하고 잠재 공간(latent space)에서 회귀(regression)를 수행하여 고해상도 특징(feature)을 추출할 수 있도록 합니다.

- **Performance Highlights**: MADM은 이미지(image)에서 깊이(depth), 적외선(infrared), 이벤트(event) 모달리티로의 적응 성능에서 최첨단(State-of-the-art) 성과를 달성했습니다. 실험 결과는 MADM이 다양한 모달리티 작업에서 우수한 전이가 이루어짐을 입증합니다.



### AdaptGCD: Multi-Expert Adapter Tuning for Generalized Category Discovery (https://arxiv.org/abs/2410.21705)
- **What's New**: 이번 연구에서는 Generalized Category Discovery (GCD) 작업에 adapter tuning을 최초로 도입하여, 기존 pretrained 모델의 일반 지식을 보존하면서 새로운 카테고리 발견에 대한 적응성을 향상시키는 AdaptGCD라는 새로운 방법을 제안합니다.

- **Technical Details**: AdaptGCD는 ViT(Visual Transformer) 블록의 피드포워드 레이어에 평행하게 학습 가능한 adapter들을 통합하고, 기존 백본의 파라미터를 동결하여 pretrained 지식을 보존합니다. 또한, old class와 new class 간의 감독 정보 불균형을 문제로 삼아, 다수의 adapter 전문가를 포함하는 Multi-Expert Adapter 구조를 도입하고, 데이터의 클래스를 구분하기 위해 route assignment 제약을 설계하였습니다.

- **Performance Highlights**: 7개의 널리 사용되는 데이터셋에서 수행된 광범위한 실험을 통해, adapter tuning이 GCD 성능을 크게 향상시키는 것으로 나타났습니다. 또한, route assignment 제약이 적용된 multi-expert adapter 구조의 효과도 확인되었습니다.



### Investigating Memorization in Video Diffusion Models (https://arxiv.org/abs/2410.21669)
Comments:
          Preprint

- **What's New**: 이번 연구는 비디오 확산 모델(VDMs)에서의 기억화(memorization) 문제를 다루며, 콘텐츠 기억화(content memorization)와 모션 기억화(motion memorization)의 두 가지 유형을 명확히 정의했습니다. 이를 통해 개인 프라이버시 보호에 대한 새로운 평가 지표를 제시하고, 기존의 평가 방법의 한계를 극복하고자 합니다.

- **Technical Details**: 연구에서 제안된 새로운 평가 지표는 콘텐츠와 모션 기억화를 개별적으로 측정하는 데 초점을 맞추고 있으며, 비디오 생성 과정에서의 다양한 데이터셋에 대해 체계적으로 분석하고 합니다. 이를 통해, 기존의 이미지 기반 방법에서 한계를 겪고 있던 VDMs의 메모리 문제를 해결하고자 합니다.

- **Performance Highlights**: 이 연구는 다수의 오픈 소스 VDMs를 이용해 다양한 비디오를 생성하였으며, 모든 테스트된 모델에서 교육 비디오를 효과적으로 추출했습니다. 또한, 콘텐츠와 모션 기억화에 대한 탐지 전략을 제안하여 VDMs의 개인 정보 보호 개선을 위한 기초를 마련했습니다.



### Revisiting Multi-Granularity Representation via Group Contrastive Learning for Unsupervised Vehicle Re-identification (https://arxiv.org/abs/2410.21667)
- **What's New**: 본 논문은 Vehicle Re-identification (Vehicle ReID) 문제를 해결하기 위한 새로운 비지도 학습 기반의 프레임워크인 MGR-GCL을 제안합니다. 이 프레임워크는 다중 세분성 CNN (Multi-Granularity CNN) 표현을 통해 전이 가능한 분별 적 특징을 학습하고, 비지도 동안 효율적인 도메인 적응을 위한 대조 학습 모듈을 통합합니다.

- **Technical Details**: MGR-GCL은 레이블이 있는 출처 데이터 세트에서 Multi-Granularity Representation (MGR)을 훈련한 후, 비 레이블 목표 데이터 세트에 대한 의사 레이블을 생성하기 위해 그룹 대조 학습 모듈 (Group Contrastive Learning, GCL)을 사용합니다. 이 방식은 DBSCAN 클러스터링 방법을 활용하여 목표 데이터 세트에 대한 그룹 레이블을 생성하며, 점진적인 대조 손실을 통해 네트워크를 업데이트합니다.

- **Performance Highlights**: VeRi, VehicleID, VehicleX 데이터 세트에서 포괄적인 실험을 수행하였고, 제안한 방법이 기존의 최첨단 기법들보다 우수하다는 것을 입증하였습니다. 이를 통해 비지도 차량 재식별 성능이 크게 향상된 결과를 보였습니다.



### Exploring Local Memorization in Diffusion Models via Bright Ending Attention (https://arxiv.org/abs/2410.21665)
Comments:
          Preprint

- **What's New**: 논문에서는 'bright ending' (BE)라는 새로운 이상현상을 통해 메모리 이미지를 기록하는 확산 모델의 문제를 해결하고 지역화된 메모리 영역을 찾는 새로운 작업을 제안합니다. BE는 확산 모델의 텍스트-이미지 생성 중 최종 추론 단계에서 메모리 이미지 조각이 끝 토큰에 높은 주의를 보이는 패턴을 의미합니다.

- **Technical Details**: BE 현상은 텍스트-이미지 확산 모델의 교차 주의 맵에서 관찰되는 특징으로, 모방된 이미지 조각에서 비모방 조각에 비해 특정 이미지 패치에서 끝 토큰에 대한 주의 점수가 비정상적으로 높게 나타납니다. 이를 통해 훈련 데이터와 유사한 생성 이미지를 메모리 하는 장소를 강조하는 마스크를 제시합니다. 이 과정은 단일 추론 패스를 사용하며, 훈련 데이터 접근 없이 이루어질 수 있습니다.

- **Performance Highlights**: 새롭게 제안한 BE와 지역화된 메모리 마스크를 기존의 평가, 탐지, 경감 전략에 통합함으로써 기존 작업에서 성능을 크게 향상시켰습니다. 이러한 통합은 특히 지역 메모리에 의한 성능 격차를 감소시키며, 새로운 최신 성능 기준을 수립합니다.



### Discriminative Pedestrian Features and Gated Channel Attention for Clothes-Changing Person Re-Identification (https://arxiv.org/abs/2410.21663)
Comments:
          The article has been accepted by IEEE International Conference on Multimedia and Expo 2024

- **What's New**: 본 논문에서는 옷 변경으로 인한 외모 변화 문제를 해결하기 위해 디스엔탱글드(Disentangled) 특징 추출 방법을 제안합니다. 또한, 게이트드 채널 주목 메커니즘(Gated Channel Attention Mechanism)을 도입하여 보행자 재식별 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 ResNet을 백본 네트워크로 사용하며, 보행자 파싱 기법을 통해 의상에 영향을 받지 않는 특징을 효과적으로 추출합니다. 모델은 두 단계의 최적화 전략을 사용하여 훈련되며, 첫 번째 단계에서는 정체성 손실만 학습하고, 그 후에 삼중 손실을 통합하여 복잡한 CC-ReID 작업에서 성능을 향상시킵니다.

- **Performance Highlights**: PRCC 및 VC-Clothes 데이터셋에서 옷 변경 시나리오에서 각각 64.8% 및 83.7%의 Top-1 정확도를 기록하며, 기존의 최첨단 방법들보다 우수한 성능을 나타냅니다.



### Fingerprints of Super Resolution Networks (https://arxiv.org/abs/2410.21653)
Comments:
          Published in Transactions on Machine Learning Research (2022)

- **What's New**: 본 연구는 GAN 기반의 이미지 생성 모델 외에도 Single Image Super-Resolution (SISR) 네트워크의 지문(fingerprints)을 분석하여, 이러한 모델들이 생성하는 이미지에서 고유한 특성을 어떻게 드러내는지 탐구합니다.

- **Technical Details**: 연구팀은 205개의 서로 다른 SISR 모델을 사용하여 205,000개의 초해상도(super-resolved) 이미지를 생성하고, 이 모델들이 생성하는 이미지에서 지문을 추출하고 분석하였습니다. 또한, 아키텍처, 배율(scaling factor), 훈련 데이터셋, 손실 함수(loss function)와 같은 하이퍼파라미터(hyperparameters)가 지문의 독창성에 미치는 영향을 논의하였습니다.

- **Performance Highlights**: 고배율(upscaling factor) 사용 또는 적대적 손실(adversarial loss)로 훈련된 SISR 네트워크는 뚜렷한 지문을 남기며, 특정 조건에서 출력 이미지에서 하이퍼파라미터를 역설계(reverse-engineer) 할 수 있다는 것을 보여주었습니다.



### Predicting the Encoding Error of SIRENs (https://arxiv.org/abs/2410.21645)
Comments:
          Published in Transactions on Machine Learning Research (2024)

- **What's New**: 이 논문에서는 인공신경망의 새로운 응용 분야로서 Implicit Neural Representations (INRs)와 그 중의 하나인 SIREN 네트워크의 인코딩 오류를 예측하는 모델을 제안합니다. 이 방법은 30만 개의 SIREN 모델을 훈련한 데이터셋을 기반으로 하여 신호 압축 속도를 개선하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: SIREN 네트워크는 다층 퍼셉트론 (Multilayer Perceptron)으로, 주기적인 활성 함수 (periodic activation function)를 사용하여 2D 이미지, 3D 형상 등을 인코딩합니다. 모델은 hyperparameter와 타겟 이미지에 기반하여 PSNR (peak signal-to-noise ratio)로 인코딩 오류를 예측합니다. 이 예측 모델은 훈련 도메인 내에서 밀리초 단위로 인코딩 오류를 정확히 예측할 수 있으며, 30만 개의 SIREN과 10만 장의 이미지를 포함한 데이터셋을 사용하여 훈련되었습니다.

- **Performance Highlights**: 제안된 예측 모델은 기존의 모델보다 훨씬 빠른 시간 안에 인코딩 오류를 예측하여 SIREN 네트워크의 성능을 효과적으로 관리할 수 있도록 도와줍니다. 실험 결과, 깊고 좁은 SIREN 구조는 인코딩 오류의 큰 변동성을 보이며, SIREN의 성능과 JPEG 압축 간의 강한 상관관계가 발견되었습니다.



### On filter design in deep convolutional neural network (https://arxiv.org/abs/2410.21644)
- **What's New**: 이 논문은 딥 컨볼루션 신경망(DCNN)의 필터(또는 가중치) 초기화, 크기-형태 선택 및 필터 수가 학습 및 최적화에 미치는 영향을 별도로 조사한 최초의 연구입니다. 이는 과거에 비해 좀 더 명확한 방법론적 접근을 제공합니다.

- **Technical Details**: 본 연구에서는 필터의 특정 물리적 매개변수 선택과 초기화 및 학습 기술에 대한 주장을 중심으로 구성되어 있으며, 반 자가 학습(semi-supervised), 자가 학습(self-supervised), 비자 지도 학습(unsupervised) 방안이 평가됩니다.

- **Performance Highlights**: 논문에서는 현재의 제약조건, 도전 과제 및 미래 가능성에 대해 논의하여 DCNN의 학습 과정을 이해하는 것이 실제 응용에서의 한계를 극복하는 데 중요하다고 강조합니다.



### Neural Experts: Mixture of Experts for Implicit Neural Representations (https://arxiv.org/abs/2410.21643)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 기존의 Implicit Neural Representations(INRs)에 Mixture of Experts(MoE) 아키텍처를 적용하여 신호 복구 성능을 개선하는 새로운 접근법이 소개됩니다. 이를 통해 지역적 특성 학습 및 공간 분할이 가능해졌습니다.

- **Technical Details**: Mixture of Experts 아키텍처는 여러 전문가 전문가 네트워크와 관리 네트워크로 구성되어 다양한 입력 데이터를 지역적으로 학습합니다. 이 새로운 네트워크 구조는 매니저 네트워크에서 전문가 네트워크로 직접적인 정보 공유로 최적화를 향상시킵니다.

- **Performance Highlights**: 제안된 Neural Experts 접근법은 이미지, 오디오 및 3D 표면 복구 작업에서 비 Mixture of Experts 방법에 비해 성능이 개선되었음을 보여주며, 적은 파라미터로 우수한 복구 성능을 달성했습니다.



### Investigation of moving objects through atmospheric turbulence from a non-stationary platform (https://arxiv.org/abs/2410.21639)
- **What's New**: 이번 연구에서는 대기 난류가 발생하는 장면에서 움직이는 물체의 optical flow field를 추출하며, 움직이는 카메라로 캡처한 이미지 시퀀스를 사용합니다. 카메라 모션에 의해 유도된 flow field를 보상하기 위해 모션 모델을 생성하고, 이를 바탕으로 대기 난류와 객체 움직임을 분리하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 optical flow 기반 방법을 사용하여 카메라 모션 모델을 구축합니다. 먼저, optical flow field를 계산한 후 카메라 모션 모델과 차감하여 대기 난류와 객체의 motion을 분리합니다. 이 과정에서 Horn-Schunck(HS) 알고리즘과 TV-L1 알고리즘을 사용하여 optical flow를 계산합니다. 또한, Apache Spark 등 데이터 처리 도구들을 활용하여 효과적인 데이터 세트를 구성합니다.

- **Performance Highlights**: 연구에서 제작된 데이터 세트는 다양한 기상 조건 하의 대기 난류와 물체 움직임을 실험적으로 보여주며, 기존의 방법론 대비 향상된 성능을 달성하였습니다. 모든 실험 데이터와 코드가 오픈 소스로 제공되어 연구자들의 접근을 용이하게 하고 있습니다.



### Adapting Diffusion Models for Improved Prompt Compliance and Controllable Image Synthesis (https://arxiv.org/abs/2410.21638)
Comments:
          Accepted to NeurIPS 2024 conference. Project Page: this https URL

- **What's New**: 본 연구에서는 최근 이미지 합성에서의 발전을 이룬 diffusion processes (DPs)를 활용하여 	extit{Factor Graph Diffusion Models} (FG-DMs)라는 새로운 모델을 제안합니다. 이는 이미지와 조건 변수를 결합한 분포를 모델링하여 기존 모델의 한계를 극복하고자 합니다.

- **Technical Details**: FG-DMs는 semantic, sketch, depth, normal maps와 같은 다양한 조건 변수를 factor graph decomposition을 통해 모델링합니다. 이를 통해 prompt compliance(프롬프트 준수)를 지원하는 효율적인 샘플링 기법과 반자동화된 세밀한 편집을 가능하게 합니다. 또한, attention distillation loss를 도입하여 모든 factor의 attention maps의 일관성을 보장합니다.

- **Performance Highlights**: 제안된 FG-DM은 기존의 Stable Diffusion (SD) 모델을 기반으로 하여 COCO dataset을 사용해 구현되었으며, 기존 SD보다 15% 높은 recall(기억률)로 이미지를 생성하는데 성공했습니다. 또한, 새로운 조건 변수를 추가할 때 기존 구조에 최소한의 수정으로도 가능하며, 지속적인 학습이 가능합니다.



### OFER: Occluded Face Expression Reconstruction (https://arxiv.org/abs/2410.21629)
- **What's New**: 이번 논문에서는 한 장의 이미지로부터 3D 얼굴 모델을 복원하는 새로운 접근 방식, OFER (Occlusion-aware Face Reconstruction) 를 소개합니다. 이 방법은 강한 occlusion (가림 현상) 아래에서도 그럴듯하고 다양한 3D 얼굴을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: OFER는 입력 이미지에 조건화된 두 개의 denoising diffusion probabilistic models (DDPMs)를 훈련시켜 얼굴 매개변수 모델의 형태 및 표정 계수를 생성합니다. 또한, 예측된 형태 정확도 점수에 따라 생성된 결과를 정렬하여 최적의 일치를 선택하는 새로운 ranking mechanism (순위 매김 메커니즘)을 제안합니다. 이 방법은 FLAME (Faces Learned with an Articulated Model) 3D 얼굴 모델과 결합되어 작동합니다.

- **Performance Highlights**: OFER는 CO-545라는 새로운 데이터 셋을 통해 검증되었으며, 기존 occlusion 기반 방법들보다 더 나은 성능을 보여주었습니다. 특히, 다수의 표정을 생성할 수 있는 능력에서 두각을 나타내었습니다.



### NYC-Event-VPR: A Large-Scale High-Resolution Event-Based Visual Place Recognition Dataset in Dense Urban Environments (https://arxiv.org/abs/2410.21615)
- **What's New**: 본 논문은 NYC-Event-VPR 데이터셋을 소개하여 이벤트 기반 시각 장소 인식(Visual Place Recognition, VPR)의 데이터 부족 문제를 해결하고자 합니다. 이 데이터셋은 뉴욕시에서 수집된 13.5시간 이상의 이벤트 데이터를 포함하고 있으며, 다양 한 조명 및 날씨 조건을 아우릅니다.

- **Technical Details**: NYC-Event-VPR 데이터셋은 Prophesee IMX636 HD 이벤트 센서를 사용하여 1280x720 해상도로 캡처되었으며, 이벤트 카메라는 1MHz의 높은 시간 해상도와 μs 단위의 낮은 지연시간, 120dB 이상의 높은 동적 범위를 제공합니다. 각각의 픽셀은 독립적으로 조명 변화에 반응하여, 모션 블러에 덜 민감하고 조명 변화에도 더 강한 특성을 보입니다.

- **Performance Highlights**: 이 연구는 이벤트 기반 VPR 분야의 일반화 성능 평가를 위해 세 가지 프레임워크를 활용하여 초기 벤치마크 실험을 진행하였으며, 이는 이벤트 기반 VPR의 혁신 및 로봇 응용 통합을 촉진하고자 하는 목표를 지니고 있습니다.



### Topological numbers and their use to characterize simple points for 2D binary images (https://arxiv.org/abs/2410.21588)
- **What's New**: 이번 논문에서는 3D 이진 이미지에서 특정 이웃 내의 간단한 점을 효율적으로 특성화하는 두 개의 topological numbers를 2D 이진 이미지에 적용합니다. 2D의 경우 단일 이웃만을 사용하여 이 두 가지 topological number를 정의합니다.

- **Technical Details**: 2D 이진 이미지는 픽셀로 구성되며, 각 픽셀은 배경을 나타내는 검은색과 형태를 나타내는 흰색으로 구분됩니다. 본 논문에서는 sequential removal을 통해 topology를 보존하면서 간단한 점들을 제거하는 thinning algorithm에 대해 논의하며, 또한 topological numbers를 사용한 간단한 점의 특성과 Hilditch crossing number, Yokoi number와 비교 분석합니다.

- **Performance Highlights**: 논문에서는 간단한 점에 대한 다양한 특성화 방법을 제안하고, thinning algorithm이 삭제할 수 있는 최대 지역 구성의 수와 관련하여 개선된 접근 방식을 제시합니다. 이를 통해 topology 보존의 중요성을 강조하며, 더 강력한 thinning algorithm의 개념도 소개합니다.



### ImageNet-RIB Benchmark: Large Pre-Training Datasets Don't Guarantee Robustness after Fine-Tuning (https://arxiv.org/abs/2410.21582)
- **What's New**: 이 논문에서는 새로운 강건한 미세 조정 벤치마크인 ImageNet-RIB(Robustness Inheritance Benchmark)를 도입하여, 대규모 사전 훈련 모델이 특수 작업에 대해 어떻게 미세 조정(fine-tuning) 되는지를 평가합니다.

- **Technical Details**: 이 벤치마크는 관련 있지만 구별되는 특수 작업들로 구성되어 있으며, 사전 훈련된 모델은 집합의 하나의 작업에 대해 미세 조정되고 나머지 작업에서의 강건성(robustness)을 평가하게 됩니다. EWC, LwF와 같은 지속적 학습(continual learning) 방법은 미세 조정 이후에도 강건성을 유지하는 반면, 일반적으로 미세 조정은 관련 하류 작업에 대한 일반화(generalization) 성능을 감소시킵니다.

- **Performance Highlights**: 사전 훈련된 데이터셋의 다양성과 풍부함에 따라 성능 저하의 정도를 예측할 수 있으며, 특수 작업에 대한 성능을 위해 가장 강력한 모델로 시작하는 것이 항상 최선의 접근 방식은 아닐 수 있다는 것을 보여줍니다.



### MVSDet: Multi-View Indoor 3D Object Detection via Efficient Plane Sweeps (https://arxiv.org/abs/2410.21566)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: MVSDet라는 새로운 방법을 제안하여 다중 뷰 이미지만을 활용하여 정확한 기하학 정보를 추출함으로써 3D 객체 감지를 개선합니다.

- **Technical Details**: MVSDet는 plane sweep을 활용하여 기하학 정보를 학습하는 방식으로, 확률적 샘플링과 소프트 가중치 메커니즘을 도입하여 각 픽셀의 깊이 위치를 결정합니다. 이를 통해 연산량을 줄이고, Gaussian Splatting을 사용하여 깊이 예측을 강화합니다.

- **Performance Highlights**: ScanNet 및 ARKitScenes 데이터셋에서의 실험 결과, MVSDet는 mAP@0.5 지표에서 각각 +3.9와 +5.1의 개선을 기록하며 우수한 성능을 입증했습니다.



### Empirical curvelet based Fully Convolutional Network for supervised texture image segmentation (https://arxiv.org/abs/2410.21562)
- **What's New**: 본 논문에서는 감독 학습(Supervised Learning)을 통해 텍스처 분류(Texture Classification) 및 세분화(Segmentation)를 수행하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 특정 텍스처 설명자(Texture Descriptors)를 사용하여 Fully Convolutional Network(FCN)에 데이터를 입력하는 방식입니다.

- **Technical Details**: 본 연구에서는 이미지를 대상으로 경험적 커브렛 변환(Empirical Curvelet Transform)을 통해 텍스처 피쳐(Features)를 추출하고, 이러한 출력이 신경망에 입력될 수 있는 효율적인 텍스처 설명자를 형성하는 데 사용됩니다. 특히, 주어진 텍스처 집합에 적합한 독특한 경험적 커브렛 필터 뱅크(Empirical Curvelet Filter Bank)를 구축하는 방법을 논의합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 평가를 통해 제안된 방법이 여러 최신 알고리즘에 비해 현저히 우수한 성능을 보임을 확인하였습니다. 이는 텍스처 분석 분야에서의 기존 문제들을 효과적으로 해결할 수 있는 가능성을 보여줍니다.



### Going Beyond H&E and Oncology: How Do Histopathology Foundation Models Perform for Multi-stain IHC and Immunology? (https://arxiv.org/abs/2410.21560)
Comments:
          Accepted at Workshop on Advancements In Medical Foundation Models (NeurIPS 2024)

- **What's New**: 이 연구는 최신 histopathology foundation 모델이 다중 염색(autoimmune) IHC 데이터셋에 대해 가지는 일반화 능력을 평가합니다. 13개의 feature extractor 모델을 비교하여 류마티스 관절염 하위 유형 분석과 쇼그렌 질병 감별 작업에 대한 성과를 분석하였습니다.

- **Technical Details**: H&E 염색된 암 이미지에서 autoimmunity IHC 이미지로 학습된 representation의 전이 가능성을 평가하기 위해 Attention-Based Multiple Instance Learning (ABMIL) 분류기를 사용했습니다. 연구는 autoimmunity 특유의 염색 패턴을 정확히 인식하는 데 어려움이 있었음을 나타냅니다.

- **Performance Highlights**: 연구 결과, histopathology pretrained 모델이 ImageNet pretrained 모델에 비해 유의미한 성과 차이를 보여주지 않았으며, autoimmunity 특성의 오해석과 편향된 feature 중요성이 발견되었습니다.



### Detection of moving objects through turbulent media. Decomposition of Oscillatory vs Non-Oscillatory spatio-temporal vector fields (https://arxiv.org/abs/2410.21551)
- **What's New**: 이번 논문에서는 대기 난류가 영향을 미치는 이미지에서 이동 객체를 탐지하는 방법을 제안합니다. 기하학적(spatial-temporal) 관점에서 문제를 다루어 난류로 인한 이동과 실제 이동 객체를 구별할 수 있는 기법을 제시합니다.

- **Technical Details**: 제안된 알고리즘은 2D 만화+텍스처 분해(cartoon+texture decomposition) 알고리즘을 3D 벡터 필드로 확장한 것입니다. 이 알고리즘은 커브렛(curvelet) 공간에 기반하여 이동 흐름의 기하학적 구조를 더 잘 특성화합니다.

- **Performance Highlights**: 실제 데이터를 기반으로 한 실험을 통해 제안된 방법의 효율성을 입증하였습니다. 따라서, 대기 난류와 이동 객체를 효과적으로 분리할 수 있는 가능성을 보여줍니다.



### ECMamba: Consolidating Selective State Space Model with Retinex Guidance for Efficient Multiple Exposure Correction (https://arxiv.org/abs/2410.21535)
Comments:
          Accepted by NeurIPS 2024. Retinex-theory, Mamba, Exposure Correction

- **What's New**: 본 연구에서는 Exposure Correction (EC)을 위한 새로운 두 가지 경로를 가진 구조를 소개합니다, 즉 ECMamba로 알려진 프레임워크입니다. 이 구조는 Retinex 이론을 완전히 통합하여 효율성과 성능의 균형을 맞추고 있습니다.

- **Technical Details**: ECMamba는 각각 반사 반응과 조명 지도를 복원하는 데 전념하는 두 개의 경로를 가지며, Retinex 추정기를 통해 입력을 두 개의 중간 공간으로 변환합니다. Retinex-SS2D라는 새로운 2D Selective State-space 계층을 핵심 연산자로 개발하여 기능 집합 전략을 개선합니다.

- **Performance Highlights**: 다양한 실험 결과와 해석적 분석을 통해, 제안된 ECMamba의 놀라운 성능을 보여줍니다. 새로운 아키텍처의 각 구성 요소가 미치는 중요성도 강조하고 있습니다.



### Towards Multi-dimensional Explanation Alignment for Medical Classification (https://arxiv.org/abs/2410.21494)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 새로운 프레임워크 Med-MICN(의료 다차원 해석 가능한 개념 네트워크)을 제안하여 의료 이미지 분석의 해석 가능성 문제를 해결합니다.

- **Technical Details**: Med-MICN은 신경 기호 추론(neural symbolic reasoning), 개념 의미론(concept semantics), 및 주목도 맵(saliency maps)을 통한 다각적 해석 가능성을 제공합니다. 이 프레임워크는 대규모 다중 모달(LMM)을 활용하여 개념 세트를 생성하고 의료 이미지에 대한 자동 주석(auto-annotation)을 수행합니다.

- **Performance Highlights**: 네 가지 벤치마크 데이터셋에서 테스트한 결과, Med-MICN은 기존의 개념 기반 모델과 블랙박스 모델에 비해 우수한 성능과 해석 가능성을 보였습니다.



### AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models (https://arxiv.org/abs/2410.21471)
- **What's New**: 이 연구는 Image-to-Image (I2I) diffusion 모델에서 발생하는 새로운 유형의 적대적 이미지 공격을 공개합니다. 또한 AdvI2I라는 새로운 프레임워크를 제안하여 입력 이미지를 조작하여 diffusion 모델이 NSFW (Not Safe for Work) 콘텐츠를 생성하도록 유도합니다.

- **Technical Details**: AdvI2I는 입력 이미지를 최적화하여 적대적 이미지를 생성하는 방법을 포함합니다. 이 프레임워크는 기존의 방어 메커니즘인 Safe Latent Diffusion (SLD)을 우회할 수 있는 특성을 가지고 있으며, 적대적 이미지와 NSFW 개념 임베딩 간의 유사성을 최소화하도록 설계된 AdvI2I-Adaptive라는 향상된 버전도 소개합니다.

- **Performance Highlights**: 법적 및 윤리적 요구사항에 대한 안전 장치가 현재의 공격에도 효과를 가지지 않음을 보여주며, 실험을 통해 AdvI2I 및 AdvI2I-Adaptive가 현존하는 안전 장치를 효과적으로 우회함을 입증했습니다. 이는 I2I diffusion 모델의 보안을 강화하기 위한 강력한 대책 마련의 필요성을 강조합니다.



### Constrained Transformer-Based Porous Media Generation to Spatial Distribution of Rock Properties (https://arxiv.org/abs/2410.21462)
Comments:
          24 pages

- **What's New**: 본 연구에서는 VQVAE와 transformer 모델을 결합하여 공간 확장 및 임의 크기의 3D 다공성 매체를 재구성하는 새로운 두 단계 모델링 프레임워크를 제안합니다.

- **Technical Details**: VQVAE는 소규모 훈련 이미지를 저차원 토큰으로 압축 및 양자화하며, transformer는 이러한 토큰을 특정 공간적 순서에 따라 조합하여 큰 이미지를 생성합니다. 이 접근 방식은 다중 토큰 생성 전략을 통해 소규모 무결성과 공간적 관계를 보존합니다.

- **Performance Highlights**: 실제 테스트 우물에서 데이터를 사용하여 우리의 다중 토큰 transformer 생성 접근 방식의 유효성을 입증하였으며, 이는 공간적 다공성 모델을 통해 우물 스케일에서의 다공성 매체 모델을 생성할 수 있는 잠재력을 보여주었습니다.



### TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors (https://arxiv.org/abs/2410.21443)
- **What's New**: 논문에서 소개된 Truck Adversarial Camouflage Optimization (TACO)은 3D 차량 모델에 대한 적대적인 위장 패턴을 생성하여 최신 모델인 YOLOv8을 속이기 위한 새로운 프레임워크입니다.

- **Technical Details**: TACO는 Unreal Engine 5를 사용하여 포토리얼리스틱(Photorealistic) 렌더링과 차별화된 렌더링을 결합하여 YOLOv8을 목표로 하는 텍스처를 최적화합니다. 또한 이를 위해 Convolutional Smooth Loss 함수를 도입하여 생성된 텍스처가 감지기를 속이는 데 효과적이면서도 시각적으로 그럴듯하도록 합니다.

- **Performance Highlights**: 실험 결과, TACO는 YOLOv8의 탐지 성능을 크게 저하시키며, 테스트 데이터에서 AP@0.5의 수치가 0.0099를 기록했습니다. 이러한 적대적 패턴은 다른 객체 탐지 모델인 Faster R-CNN 및 이전 YOLO 버전에서도 강한 전이 가능성을 보였습니다.



### SocialGPT: Prompting LLMs for Social Relation Reasoning via Greedy Segment Optimization (https://arxiv.org/abs/2410.21411)
Comments:
          Accepted by NeurIPS 2024. Project page: this https URL

- **What's New**: 소셜 관계 추론을 위한 새로운 모듈형 프레임워크 SocialGPT를 소개합니다. 이 프레임워크는 Vision Foundation Models (VFMs)와 Large Language Models (LLMs)의 인식 및 추론 능력을 결합하여 소셜 관계 인식을 위한 강력한 기준선을 제공합니다.

- **Technical Details**: SocialGPT는 VFMs를 사용하여 이미지 내용을 텍스트 소셜 스토리로 변환하고, 이후 LLMs를 통해 텍스트 기반 추론을 수행합니다. 이 프레임워크는 VFMs와 LLMs 각각을 조정할 수 있는 체계적인 설계 원칙을 도입합니다. Greedy Segment Prompt Optimization (GSPO) 알고리즘은 자동화된 프롬프트 조정 방법으로, 긴 프롬프트 최적화 문제를 해결합니다.

- **Performance Highlights**: 문화 및 다른 이미지 스타일에 일반화하고 추가 모델 훈련 없이 두 데이터베이스에서 경쟁력 있는 제로샷(zero-shot) 결과를 달성하였습니다. 실험 결과, GSPO는 LLM의 성능을 크게 개선하며, 우리의 방법은 기존 최첨단 기법들을 상당히 초월합니다.



### Domain Adaptation with a Single Vision-Language Embedding (https://arxiv.org/abs/2410.21361)
Comments:
          Under review

- **What's New**: 이번 논문에서는 도메인 적응(Domain Adaptation)을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 전체 타겟 데이터를 필요로 하지 않고, 단일 Vision-Language (VL) 잠재 임베딩에 의존하여 이루어집니다. 기존 방법과의 차별점은 CLIP 모델을 활용하여 여러 시각적 스타일을 추출하는 기능 향상 방법인 Prompt/photo-driven instance normalization (PIN)을 적용하는 것입니다.

- **Technical Details**: 이 연구는 언어 프롬프트나 단일 비표시 타겟 이미지를 기반으로 하는 단일 VL 잠재 임베딩을 통해 도메인 적응을 수행합니다. PIN 방법을 사용해 저수준 소스 특징의 affine 변환을 최적화하여 여러 시각적 스타일을 생성하며, 이를 통해 제로샷(Zero-Shot)과 원샷(One-Shot) 비지도 도메인 적응을 가능하게 합니다. 이 방식은 CLIP의 잠재 공간을 활용하여 소스 도메인 임베딩을 타겟 도메인 임베딩으로 변환합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 제로샷 및 원샷 설정에서 관련 기준선보다 뛰어난 성능을 보였습니다. PØDA와 PIDA 방식으로 이미지 분할 성능이 향상되었으며, 이는 공공 데이터 사용이 제한된 산업 환경에서도 유용함을 나타냅니다.



### Multi-path Exploration and Feedback Adjustment for Text-to-Image Person Retrieva (https://arxiv.org/abs/2410.21318)
- **What's New**: 이번 논문에서는 Multi-Pathway Exploration, Feedback, and Adjustment (MeFa) 프레임워크를 제안하여 텍스트 기반 인물 검색에서의 성능을 개선한다. 기존의 VLP 모델의 한계를 극복하고자 하는 노력이 돋보인다.

- **Technical Details**: MeFa는 여러 경로를 통한 탐색, 피드백 및 조정 과정을 포함하여 intra-modal(단일 모달) 및 inter-modal(다중 모달) 피드백을 활용하여 텍스트와 인물 간의 정밀한 연관성을 이룬다. 세 가지 주요 경로로 조직되며, 각각의 경로는 하드 네거티브 샘플을 생성하거나, 글로벌 및 로컬 정보의 정제를 통해 면밀한 세부 사항을 개선한다.

- **Performance Highlights**: CUHK-PEDES, ICFG-PEDES, RSTPReid의 세 가지 공개 벤치마크에서 MeFa는 선행 기술들보다 뛰어난 인물 검색 성능을 달성하여 추가 데이터나 복잡한 구조 없이도 우수한 결과를 나타냈다.



### Towards Robust Out-of-Distribution Generalization: Data Augmentation and Neural Architecture Search Approaches (https://arxiv.org/abs/2410.21313)
Comments:
          Hong Kong University of Science and Technology Thesis

- **What's New**: 이 논문에서는 Deep Learning (딥러닝) 분야에서 Out-of-Distribution (OoD) 데이터에 대해 보다 강력한 일반화 방법론을 제안합니다. 기존 모델이 수집된 분포와 다를 경우의 성능 저하 문제를 해결하기 위해 spurious correlation (가짜 상관관계)을 분리하고, context-related features (맥락 관련 특성)에 대한 gradient-based augmentation (그래디언트 기반 증강)을 수행하여 학습된 표현의 강인성을 향상시킵니다.

- **Technical Details**: 논문에서 제안하는 DecAug 방법은 category-related features (범주 관련 특성)과 context-related features (맥락 관련 특성)를 직교화하여 분리합니다. 또한, NAS-OoD(Neural Architecture Search-out-of-distribution)는 가짜 OoD 데이터를 생성하는 조건부 생성기를 배우면서 아키텍처 파라미터를 최적화합니다. 이를 통해 더 강력한 네트워크 아키텍처를 발견하고 있는 것으로 보입니다.

- **Performance Highlights**: DecAug는 다양한 OoD 데이터셋에서 최첨단 방법들을 초월하는 성능을 보여주었으며, NAS-OoD 방법은 업계 데이터셋에서 70% 이상의 오류율 감소를 보이며 실제 애플리케이션에서도 높은 유용성을 입증했습니다.



### MMDocBench: Benchmarking Large Vision-Language Models for Fine-Grained Visual Document Understanding (https://arxiv.org/abs/2410.21311)
Comments:
          Under review

- **What's New**: 대형 비전-언어 모델(LVLMs)의 세밀한 시각적 이해 능력을 평가할 새로운 벤치마크인 MMDocBench를 제안합니다. 기존 벤치마크의 한계를 보완하기 위해 다양한 문서 이미지를 사용하여 평가합니다.

- **Technical Details**: MMDocBench는 15개의 주요 작업과 4,338개의 QA 쌍, 11,353개의 지원 영역을 포함하는 문서 이해 작업으로 구성되어 있습니다. 작업에는 텍스트 인식, 테이블 인식, 문서 위조 탐지 등이 포함되며, 다양한 문서 이미지 유형을 평가할 수 있습니다.

- **Performance Highlights**: 실험을 통해 현재 LVLMs가 지역 예측에서 매우 낮은 성능을 보이며, 문서 이미지에서의 로컬화와 탐지가 다른 작업들보다 더 어려운 것으로 나타났습니다. 또한, 개방형 및 폐쇄형 LVLM 간에 답변 예측에서 차이가 발견되었지만 지역 예측 성능은 유사했습니다.



### ArCSEM: Artistic Colorization of SEM Images via Gaussian Splatting (https://arxiv.org/abs/2410.21310)
Comments:
          presented and published at AI for Visual Arts Workshop and Challenges (AI4VA) in conjunction with European Conference on Computer Vision (ECCV) 2024, Milano, Italy

- **What's New**: 본 연구에서는 주사 전자 현미경(Scanning Electron Microscopes, SEM)에서 캡처한 단일 또는 일부 색칠된 이미지를 기반으로 다수의 색상화된 이미지를 자동으로 생성하는 새로운 접근 방식을 제안합니다. 이는 기존의 수동 색칠 작업을 대체할 수 있으며, 예술가가 한 번의 색칠로 전체 장면에 색상을 전파할 수 있도록 합니다.

- **Technical Details**: 우리의 접근 방식은 2D Gaussian Splatting(2DGS) 기반의 정확한 3D 표현을 활용하여 SEM 장면의 색상화를 수행합니다. 우리는 이미지 특정의 affine color transformation(ACT)을 적용하여 조명 변동 문제를 해결했습니다. 더불어 전문 예술가인 Martin Oeggerli가 생성한 최대 5개의 색상 이미지를 사용하여 씬에 색상을 도입합니다.

- **Performance Highlights**: 제안된 ArCSEM 방법을 통해 SEM 장면의 표현적이고 색칠된 새로운 뷰를 생성할 수 있으며, 이전 방법들과 비교하여 뛰어난 성능을 보여줍니다. 실험을 통해 우리의 방법이 기존 시스템에 비해 효과적임을 입증했습니다.



### A Robust Anchor-based Method for Multi-Camera Pedestrian Localization (https://arxiv.org/abs/2410.21308)
- **What's New**: 이 논문은 이미지와 카메라 매개변수를 사용하여 보행자의 위치를 추정하는 시각 기반(localization) 방법을 다룹니다. 기존 방법이 카메라 매개변수의 오류로 인해 부정확해지는 문제를 해결하기 위한 anchor(앵커) 기반 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 고정 위치의 앵커를 활용하여 카메라 매개변수의 오류 영향을 줄이고, 이론적 분석을 통해 견고성을 입증했습니다. 앵커의 재투영 오차를 통해 카메라 매개변수의 오류를 보정하는 방식으로, 카메라의 정확한 위치 추정에 기여합니다.

- **Performance Highlights**: 실시된 실험들은 제안한 방법이 소음이 있는 카메라 매개변수 상황에서도 높은 정확성을 유지하면서, 기존 앵커 없이 접근한 방법들에 비해 유의미한 향상을 보인다는 것을 보여줍니다.



### VideoSAM: A Large Vision Foundation Model for High-Speed Video Segmentation (https://arxiv.org/abs/2410.21304)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문에서는 Segment Anything Model (SAM)을 기반으로 한 VideoSAM을 소개합니다. VideoSAM은 다양한 고속 비디오 (HSV) 데이터 셋에 맞춰 조정되어 복잡한 기포 형성을 구분할 수 있도록 발전되었습니다.

- **Technical Details**: VideoSAM은 두 단계 접근 방식으로 작동합니다. 먼저 각 데이터 모달리티(아르곤, 질소, FC-72, 물)에 대해 전문화된 U-Net CNN을 구축하여 초기 분할 마스크를 생성합니다. 이후 이 이미지-마스크 쌍을 VideoSAM의 트랜스포머 아키텍처에 공급하여 최종 출력에 대한 정제된 분할 마스크를 생성합니다.

- **Performance Highlights**: VideoSAM은 복잡한 유체 환경에서 U-Net보다 현저하게 뛰어난 성능을 보였습니다. 실험 결과 VideoSAM이 다양한 과학적 응용 분야에서 더 강력하고 정확한 HSV 분할을 제공할 가능성을 강조합니다.



### VEMOCLAP: A video emotion classification web application (https://arxiv.org/abs/2410.21303)
Comments:
          Accepted to 2024 IEEE International Symposium on Multimedia (ISM), Tokyo, Japan

- **What's New**: VEMOCLAP을 소개합니다: 사용자 제공 비디오의 감정 내용을 분석할 수 있는 최초의 오픈 소스 웹 애플리케이션입니다. 이전의 연구를 개선하여 멀티헤드 크로스-어텐션(multi-head cross-attention)을 사용하여 비디오 프레임과 오디오로부터 추출된 사전 훈련된 특징을 효율적으로 융합합니다.

- **Technical Details**: VEMOCLAP은 Google Colab에서 호스트되며, 무료 GPU 런타임으로 모든 사용자 수준에서 접근 가능하고 몇 번의 클릭만으로 사용할 수 있습니다. 사용자는 비디오를 업로드하거나 YouTube 링크를 제공할 수 있으며, 예상 감정을 출력하고 자동 음성 인식(ASR), 광학 문자 인식(OCR), 얼굴 감지 및 표정 분류, 오디오 분류, 이미지 캡셔닝과 같은 추가 분석을 포함합니다. 모델은 Ekman-6 비디오 감정 데이터셋을 기반으로 훈련되었으며, 고급 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 모델은 Ekman-6 비디오 감정 데이터셋에서 분류 정확도를 4.3% 향상시켰으며, 데이터 정제를 통해 비디오 감정 모델의 훈련을 개선했습니다. 이 웹 애플리케이션을 통해 사용자는 자신의 비디오나 YouTube 비디오를 분석하고 감정을 분류할 수 있습니다.



### Domain-Adaptive Pre-training of Self-Supervised Foundation Models for Medical Image Classification in Gastrointestinal Endoscopy (https://arxiv.org/abs/2410.21302)
- **What's New**: 이번 연구에서는 대규모의 위장내시경(VCE) 데이터셋인 EndoExtend24를 소개합니다. 이 데이터셋은 10개의 기존 공개 및 비공식 데이터셋을 병합하고 재구성하여 만들어졌으며, 226,000개 이상의 라벨링된 이미지를 포함하고 있습니다. 또한, 123개의 병리학적 소견을 지원하는 동적 클래스 매핑을 제공하여 다양한 레이블링 세분화에 따라 통일된 학습을 가능하게 합니다.

- **Technical Details**: EndoExtend24 데이터셋은 영상 캡슐 내시경, 위내시경, 대장내시경 등의 세 가지 검사 방법을 포함합니다. EVA-02 모델은 Vision Transformer 아키텍처를 기반으로 하며, EndoExtend24 데이터셋에서 사전 훈련(pre-training) 후, Capsule Endoscopy 2024 Challenge 데이터셋에서 미세 조정(fine-tuning)됩니다. 이 과정에서 SwiGLU 활성화(SwiGLU activations), Rotary Position Embeddings (ROPE), 추가적인 Layer Normalization (LN) 기능이 적용됩니다.

- **Performance Highlights**: 실험 결과, 챌린지 검증 세트에서 AUC 매크로 점수 0.993과 균형 잡힌 정확도 89.3%를 기록하는 등 우수한 성능을 보였습니다.



### Contrastive Learning with Auxiliary User Detection for Identifying Activities (https://arxiv.org/abs/2410.21300)
Comments:
          Accepted in ICMLA 2024

- **What's New**: 본 논문에서는 User Identification(UI)와 Context-Aware Human Activity Recognition(CA-HAR)를 통합한 새로운 프레임워크인 CLAUDIA를 제안합니다. CLAUDIA는 사용자 특성과 맥락 정보를 동시에 고려하여 개인화된 활동 인식을 향상시킵니다.

- **Technical Details**: CLAUDIA 프레임워크는 Contrastive Learning을 통한 사용자 식별과 주요 CA-HAR 과업을 통합하며, supervised contrastive loss를 도입하여 사용자 간의 변이를 정규화합니다. 이 모델은 다양한 모달리티 센서 데이터를 통합하여 높은 성능을 얻도록 설계되었습니다.

- **Performance Highlights**: 실제 CA-HAR 데이터셋을 통해 CLAUDIA는 Matthew의 상관 계수(Matthew's Correlation Coefficient)에서 평균 5.8%에서 14.1%, Macro F1 점수에서 3.0%에서 7.2% 향상된 성능을 보였습니다.



### TV-3DG: Mastering Text-to-3D Customized Generation with Visual Promp (https://arxiv.org/abs/2410.21299)
- **What's New**: 이 논문에서는 최근 Text-to-3D 생성의 발전을 다루고 있으며, 기존의 Score Distillation Sampling (SDS) 기술의 한계를 극복하기 위한 새로운 알고리즘인 Classifier Score Matching (CSM)을 제안합니다.

- **Technical Details**: CSM은 SDS에서 차이 항(difference term)을 제거하고, 최적화 과정에서 노이즈를 결정적(deterministic)으로 추가하여 노이즈를 줄이는 과정을 포함합니다. 또한, 시각적 정보 통합을 위해 주의력 융합(attention fusion) 메커니즘과 샘플링 가이던스(sampling guidance) 기술을 통합하여 Visual Prompt CSM (VPCSM) 알고리즘을 형성합니다.

- **Performance Highlights**: TV-3DG로 명명된 이 새로운 접근법은 안정적이며 고품질의 맞춤형 3D 생성을 성공적으로 달성하는 성능을 보여줍니다.



### Local Policies Enable Zero-shot Long-horizon Manipulation (https://arxiv.org/abs/2410.22332)
Comments:
          Main paper 7 pages, 3 tables, 3 figures. Appendix 6 pages, 2 figures, 6 tables

- **What's New**: ManipGen을 통해 로봇 조작을 위한 새로운 접근 방식을 제시하며, 시뮬레이션에서 실제 환경으로의 전이를 개선하는 방법론을 개발했습니다.

- **Technical Details**: 로컬 정책(local policies)을 도입하여, 객체와 로봇 간의 상대적인 포즈에 대한 불변성과 다양한 장면 구성에 대한 적응성을 확보했습니다.

- **Performance Highlights**: ManipGen은 50개의 실제 조작 과제에서 76%의 성공률을 기록하며 기존 방법들보다 36%에서 최대 76%까지 성능 향상을 보여주었습니다.



### Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datas (https://arxiv.org/abs/2410.22325)
- **What's New**: 이번 연구는 로봇 비주얼 표현에 대한 사전 학습(pre-training)에서 새로운 접근 방식을 제안하며, 매니퓰레이션 중심의 표현(Manipulation Centric Representation, MCR) 프레임워크를 통해 로봇 조작 작업의 성능을 향상시키는 방법을 탐구합니다.

- **Technical Details**: MCR은 로봇의 시각적 특징과 조작 작업의 행동 및 고유 감각(proprioceptive) 정보를 모두 포착하는 기반 표현 학습 프레임워크입니다. 이 방법은 DROID 로봇 데이터셋에서 비주얼 인코더를 사전 학습하고, 로봇의 고유 상태 및 행동 데이터를 활용하여 새로운 대조 손실(contrastive loss)을 도입합니다. 이 손실은 시각적 관찰을 로봇의 고유 상태-행동 동역학과 정렬시킵니다. 

- **Performance Highlights**: 실험 결과, MCR은 약 20개의 다양한 로봇 작업을 포함하는 4개의 시뮬레이션 도메인에서 가장 강력한 기준 방법보다 14.8% 성능을 향상시켰으며, 3개의 실제 로봇 작업에서는 76.9%의 성능 향상을 보였습니다.



### Natural Language Inference Improves Compositionality in Vision-Language Models (https://arxiv.org/abs/2410.22315)
Comments:
          Project page: this https URL

- **What's New**: 본 연구는 자연어 추론(Natural Language Inference, NLI)을 활용하여 서로 모순되거나 함의가 있는 문장을 생성함으로써 이미지와 텍스트 간의 관계를 보다 정교하게 이해하는 새로운 접근 방식인 캡션 확장(Caption Expansion with Contradictions and Entailments, CECE)을 제안합니다. 이를 통해 모델의 해석 가능성(interpretability)을 높이며, 기존 방법들이 갖는 편향(bias)이나 피상적인 특성에 의존하는 문제를 해결하고자 합니다.

- **Technical Details**: CECE는 텍스트로부터 원래 의미를 유지하면서도 어휘적으로 다양한 문장을 생성합니다. 자연어 추론을 통해 주어진 전제(premise)에 대한 함의(entailments) 및 모순(contradictions)을 생성하며, 이 과정은 LLM(대형 언어 모델)을 활용합니다. CECE는 문장을 다루는 방식에서 기존의 문장 분해 방법(Sentence Decomposition via Semantics, SDS)을 넘어, 보다 풍부하고 다양한 문맥 정보를 제공하여 모델의 성능을 향상시키는 방식입니다.

- **Performance Highlights**: CECE를 적용한 결과, 이미지-텍스트 정렬(agreement with human judgments) 기준에서 이전 방법들보다 월등한 성과를 보였으며, Winoground에서 +19.2%, EqBen에서 +12.9%의 성능 향상을 이루었습니다. CECE는 추가적인 파인튜닝 없이도 이미지와 텍스트 간의 정렬을 개선하며, 기존의 편향 문제를 완화하여 보다 통합적인 결과를 생성합니다.



### Emotion-Guided Image to Music Generation (https://arxiv.org/abs/2410.22299)
Comments:
          2024 6th Asian Digital Image Processing Conference

- **What's New**: 이 논문은 감정에 기반하여 이미지를 음악으로 변환하는 새로운 프레임워크를 소개합니다. 기존 모델이 대조 학습(contrastive learning)에 의존하는 것과 달리, 제안된 모델은 감정 일치도를 향상시키기 위해 Valence-Arousal (VA) 손실 함수(loss function)를 직접 통합하였습니다.

- **Technical Details**: 모델은 CNN-Transformer 아키텍처를 사용하며, 미리 학습된 CNN 이미지 특징 추출기와 MIDI 음악에서 복잡하고 고급 감정 특징을 캡처하기 위한 세 개의 Transformer 인코더를 포함합니다. 이전 프로세스를 거쳐 변형된 세 개의 Transformer 디코더는 음악적으로 및 감정적으로 일관된 MIDI 시퀀스를 생성하도록 특징을 정제합니다.

- **Performance Highlights**: 실험 결과, 새롭게 구성된 감정적으로 쌍을 이룬 이미지-MIDI 데이터셋에서 제안된 모델이 Polyphony Rate, Pitch Entropy, Groove Consistency 및 손실 수렴(loss convergence)과 같은 여러 메트릭에서 우수한 성능을 보여주었습니다.



### Guide3D: A Bi-planar X-ray Dataset for 3D Shape Reconstruction (https://arxiv.org/abs/2410.22224)
Comments:
          Accepted to ACCV 2024

- **What's New**: Guide3D 데이터셋의 도입으로 3D 재구성의 발전을 위한 새로운 기회를 제공합니다. 이 데이터셋은 고해상도 이중 평면(dual-plane) 영상으로 구성되어 있으며, 실제 환경에서 수동으로 주석을 추가한 비디오가 포함되어 있습니다. 이 연구는 임상 환경을 반영한 시뮬레이션 환경에서 데이터셋의 유효성을 검증하여 실질적인 적용 가능성을 입증합니다.

- **Technical Details**: Guide3D 데이터셋은 이중 평면 X-ray 시스템을 통해 촬영된 상세 영상을 포함합니다. 특히, 깊이 인식을 향상시키기 위해 다중 시각 이미징 시스템을 활용하며, 이때 적용되는 에피폴라 기하학(epipolar geometry) 기반 재구성 방법을 사용합니다. 데이터셋은 8,746개의 고해상도 샘플로 구성되어 있으며, 이는 자동화된 가이드와 추적을 지원하는 견고한 기반을 제공합니다.

- **Performance Highlights**: 이 연구에서는 가이드와 형태 예측을 위한 새로운 기준을 제안하며, 향후 연구에 대한 강력한 기준을 제공합니다. Guide3D는 세분화(segmentation)와 3D 재구성 기술의 발전을 위한 플랫폼을 제공하여 보다 정확하고 효율적인 내혈관 수술 개입의 발전에 이바지합니다.



### MAPUNetR: A Hybrid Vision Transformer and U-Net Architecture for Efficient and Interpretable Medical Image Segmentation (https://arxiv.org/abs/2410.22223)
- **What's New**: 본 논문에서는 의료 영상 분할을 위한 새로운 아키텍처인 MAPUNetR을 소개합니다. 이 모델은 transformer 모델의 강점과 입증된 U-Net 프레임워크를 결합하여 영상 해상도를 보존하면서도 세밀한 정보 처리를 가능하게 합니다.

- **Technical Details**: MAPUNetR은 attention maps를 포함하여 분할된 영역을 강조하며, 의료 이미지의 세밀한 부분을 감지하는 데 필수적인 해상도 보존 문제를 해결합니다. 이 모델은 BraTS 2020 데이터셋에서 테스트되었으며, ISIC 2018 데이터셋에서 dice score 0.88과 dice coefficient 0.92를 기록했습니다.

- **Performance Highlights**: MAPUNetR은 안정적인 성능을 유지하며, 임상 실무에서 의료 영상 분할을 위한 강력한 도구로서의 잠재력을 보여줍니다.



### ADAM: An Embodied Causal Agent in Open-World Environments (https://arxiv.org/abs/2410.22194)
- **What's New**: 본 논문에서는 Minecraft 환경에서 자율적으로 탐색하고, 다중 모달 컨텍스트를 인식하며, 인과 세계 지식을 학습하여 복잡한 작업을 해결할 수 있는 ADAM(An emboDied causal Agent in Minecraft)을 소개합니다. ADAM은 기존 모델의 한계를 극복하고 더 나은 해석 가능성과 일반화 능력을 제공합니다.

- **Technical Details**: ADAM은 네 가지 주요 모듈로 구성됩니다: 1) Interaction Module - 에이전트가 행동을 실행하고 상호작용 과정을 문서화합니다. 2) Causal Model Module - 인과 그래프를 구성하여 해석 가능성을 높이고 사전 지식에 대한 의존도를 줄입니다. 3) Controller Module - planner, actor 및 memory pool로 구성되어, 학습한 인과 그래프를 기반으로 작업을 수행합니다. 4) Perception Module - 다중 모달 대형 언어 모델(MLLMs)을 사용하여 ADAM이 인간 플레이어처럼 주변을 인식하도록 합니다.

- **Performance Highlights**: ADAM은 기존 최첨단(SOTA) 방법과 비교하여 다이아몬드 획득 작업에서 2.2배의 속도를 보였으며, 공예 레시피가 수정된 경우에도 안정적으로 다이아몬드를 획득하는 유일한 방법으로, 4.6배의 속도를 기록했습니다. ADAM은 거의 완벽한 기술 트리를 구성하여 해석 가능성을 높이고, 인간 플레이 방식에 근접한 성능을 유지했습니다.



### RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier (https://arxiv.org/abs/2410.22124)
Comments:
          Accepted at NeurIPS 2024 (Poster)

- **What's New**: 이번 논문에서는 RushUp이라는 새로운 반감독 학습 프레임워크를 제안합니다. 이는 기존의 반감독 분류 기법을 회귀 문제에 적용하여 성능을 향상시키기 위해 개발되었습니다. RankUp은 원래의 회귀 문제를 순위 문제로 변환하여 보조 순위 분류기를 동시에 훈련시킴으로써 성능을 높입니다.

- **Technical Details**: RankUp은 보조 순위 분류기를 사용하여 원래의 회귀 작업과 함께 순위 작업을 동시에 해결합니다. 이를 통해 기존의 반감독 분류 기법을 활용하여 훈련할 수 있으며, 새로운 방법인 Regression Distribution Alignment (RDA)를 통해 의사 라벨의 품질을 더욱 개선합니다.

- **Performance Highlights**: RankUp은 RDA 없이도 다양한 회귀 벤치마크에서 SOTA 결과를 달성했습니다. 예를 들어, RankUp은 UTKFace 데이터셋에서 50개의 레이블 샘플을 사용하여 MAE에서 최소 13% 개선을 보여주었으며, RDA 통합 시 추가적인 6% 개선을 기록했습니다.



### 4D-based Robot Navigation Using Relativistic Image Processing (https://arxiv.org/abs/2410.22087)
Comments:
          AAAI Fall Symposia 2024

- **What's New**: 이 논문은 동적 환경에서 로봇 내비게이션을 위한 4D 기반 접근 방법을 제안합니다. 상대론적 이미지 처리(Relativistic Image Processing)가 결합된 4D 정보 사용을 통해 로봇의 인지 및 반응 능력을 향상시키는 방법을 탐구합니다.

- **Technical Details**: 4D 내비게이션은 시간에 관련된 여러 회전과 변환을 활용하여 로봇의 위치를 할당할 수 있도록 해줍니다. 전통적인 6자유도(Euclidean 6 DoFs)의 움직임 표현과 비교하여 비유클리드(Non-Euclidean) 4D 공간은 10자유도(DoFs)를 포함함으로써 더 복잡한 환경에서의 내비게이션을 가능케 합니다.

- **Performance Highlights**: 4D 기반 로봇 내비게이션의 효과성을 입증하였으며, 상대론적 이미지 처리 원칙을 통해 센서 인식의 가능성을 확장하였습니다. 이 접근 방식은 로봇이 동적 환경에서 안전하게 상호작용하고 이동할 수 있도록 돕습니다.



### DINeuro: Distilling Knowledge from 2D Natural Images via Deformable Tubular Transferring Strategy for 3D Neuron Reconstruction (https://arxiv.org/abs/2410.22078)
Comments:
          9 pages, 3 figures, and 2 tables. This work has been submitted to the IEEE for possible publication

- **What's New**: 본 연구에서는 2D 자연 이미지에서 얻은 선행 지식을 활용하여 3D 신경 세포의 형태학적 학습을 개선하는 혁신적인 프레임워크, DINeuro를 제안합니다. 이 프레임워크는 깊은 학습 기법을 활용하여 기존의 세그멘테이션 모델들이 간과했던 신경 세포의 내재적 구조적 특성을 통합합니다.

- **Technical Details**: DINeuro는 사전 훈련된 2D 비전 트랜스포머(DINO)의 지식을 3D 비전 트랜스포머에 전이하여 3D 신경 세포 이미지를 효과적으로 포착합니다. 이 과정에서 우리는 변형 가능한 튜블 전송 전략(deformable tubular transferring strategy)을 도입하여 2D 자연 지식을 신경 구조의 원형 특성에 적응시킵니다. 2D에서 3D로의 지식 전이는 평균 및 중심 전략을 통해 이루어지며, 이를 통해 수신(field)의 확장을 가능하게 합니다.

- **Performance Highlights**: 대규모 데이터세트인 Janelia 데이터셋에서 실험한 결과, DINeuro는 일반적인 Dice 지표에서 4.53% 향상 및 95% 하슈도르프 거리 지표에서 3.56% 향상을 달성하였습니다. 이는 3D 신경 세포 이미지의 복잡한 나뭇가지 구조를 보다 효율적으로 학습하고 효과적으로 세그멘테이션하는 결과를 보여줍니다.



### PACA: Perspective-Aware Cross-Attention Representation for Zero-Shot Scene Rearrangemen (https://arxiv.org/abs/2410.22059)
Comments:
          Accepted by WACV2025

- **What's New**: 이번 논문에서는 로봇 조작에서의 장면 재배치(scene rearrangement) 문제를 해결하기 위해 PACA라는 제로샷(zero-shot) 파이프라인을 제안합니다. 이 시스템은 Stable Diffusion으로부터 파생된 관점 인식 크로스 어텐션(perspective-aware cross-attention) 표현을 활용하여 단일 단계에서 객체 수준의 표현을 생성합니다.

- **Technical Details**: PACA는 객체 수준의 표현을 생성, 분할(segmentation), 특성 인코딩(feature encoding)을 하나의 단계에서 통합합니다. 또한, 시점 제어(perspective control)를 도입하여 6-DoF 설정에서 실제 장면과 생성된 목표를 맞출 수 있도록 합니다. 이 연구에서는 기존의 3-DoF 한계를 넘어 6-DoF 카메라 뷰를 적합하게 만드는 데 중점을 두었습니다.

- **Performance Highlights**: PACA는 다양한 장면에서 실제 로봇 실험을 통해 제로샷 성능을 입증하였으며, 평균 매칭 정확도는 87%, 실행 성공률은 67%를 기록했습니다.



### FANCL: Feature-Guided Attention Network with Curriculum Learning for Brain Metastases Segmentation (https://arxiv.org/abs/2410.22057)
- **What's New**: 본 논문에서는 Feature-guided Attention Network with Curriculum Learning (FANCL)이라는 새로운 모델을 제안하였습니다. 이 모델은 뇌 전이암(BMs)의 세분화를 개선하기 위해 독특한 접근 방식을 사용합니다.

- **Technical Details**: FANCL은 CNN 기반으로, 다양한 크기의 전이암 사이의 본질적인 상관관계를 구축하여 작은 종양의 세분화에서 발생하는 고급 특징 손실을 보완합니다. 또한 voxel-level curriculum learning 전략을 적용하여 모델이 전이암 구조와 세부 사항을 점진적으로 학습하도록 도와줍니다.

- **Performance Highlights**: BraTS-METS 2023 데이터셋에 대한 평가 결과, FANCL은 세분화 성능을 크게 향상시켜 제안된 방법의 효과성을 입증하였습니다.



### Are VLMs Really Blind (https://arxiv.org/abs/2410.22029)
Comments:
          2 pages, 1 figure

- **What's New**: 이 연구는 Visual Question Answering (VQA)에서 시각적 캡션 생성과 질문 응답을 결합하는 새로운 방법을 제안합니다. 기존 접근 방식보다 성능을 향상시키는 동시에 VLMs(사물 시각 언어 모델)이 기하학적 추론에 대한 인식을 증진할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서, Llama 3.1 모델에서 추출된 키워드를 기반으로 이미지 캡션을 생성합니다. 두 번째 단계에서, 생성된 캡션을 VLM(Geminiflash)에 입력하고 해당 질문을 제출하여 답변을 받습니다. 이 방식은 VQA 태스크에서 키워드 기반 캡션 생성의 효과를 증명하기 위해 설계되었습니다.

- **Performance Highlights**: 평균 정확도는 키워드 기반 캡션 방식이 적용될 때 전통적인 VQA 접근 방식에 비해 항상 더 높았습니다. 특히 기하학적 비계수 태스크의 경우, 성능 향상이 두드러진 반면, 계수 관련 태스크에서는 개선이 일관되지 않고 랜덤한 경향을 보였습니다.



### ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting (https://arxiv.org/abs/2410.21955)
- **What's New**: 이번 연구에서는 Gaussian splatting에 기반한 새로운 자율 고화질 재구성 시스템인 ActiveSplat을 제안합니다. 이 시스템은 온라인 맵핑, 시점 선택(viewpoint selection), 경로 계획(path planning)을 통합하는 프레임워크를 설정하여 효율적이고 현실적인 렌더링을 가능하게 합니다.

- **Technical Details**: ActiveSplat은 혼합 지도 표현(hybrid map representation)을 사용하여 환경에 대한 밀집 정보(dense information)와 작업 공간의 희소한 추상화(sparse abstraction)를 통합합니다. 이 시스템은 희소 토폴로지(sparse topology)를 활용하여 효율적인 시점 샘플링(viewpoint sampling) 및 경로 계획을 수행하며, 뷰 의존적인 밀집 예측(view-dependent dense prediction)을 통해 시점 선택을 용이하게 합니다.

- **Performance Highlights**: 지속적인 실험 및 블라인드 실험(제외 실험)을 통해 이 시스템은 재구성 정확도(reconstruction accuracy), 데이터 커버리지(data coverage), 및 탐색 효율성(exploration efficiency) 측면에서 제안된 방법의 효용성을 입증했습니다.



### Analyzing Noise Models and Advanced Filtering Algorithms for Image Enhancemen (https://arxiv.org/abs/2410.21946)
- **What's New**: 본 논문은 이미지의 노이즈 제거에 대한 다양한 필터링 기법의 효과를 평가합니다. 특히, 여덟 가지 타입의 노이즈를 가진 이미지에 대해 Wiener, Median, Gaussian, Mean, Low pass, High pass, Laplacian 및 bilateral filtering과 같은 방법론을 검토합니다.

- **Technical Details**: 본 연구는 Peak signal to noise ratio (PSNR)라는 성능 지표를 사용하여 다양한 필터가 각각의 노이즈 모델에 미치는 영향을 분석합니다. 여러 필터를 적용하여 노이즈 모델에 대한 각 필터링 전략의 적합성을 판단합니다.

- **Performance Highlights**: 연구 결과는 다양한 필터링 기법이 각기 다른 노이즈 모델에 적용될 때의 효과를 보여주며, 노이즈 제거 후 이미지의 선명도를 개선하여 의료 이미징, 위성 이미지 및 레이더 응용 분야에서 더 나은 해석과 분석이 가능해짐을 강조합니다.



### CT to PET Translation: A Large-scale Dataset and Domain-Knowledge-Guided Diffusion Approach (https://arxiv.org/abs/2410.21932)
Comments:
          IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이번 연구에서는 CT 이미지를 기반으로 PET 이미지를 생성하기 위해 조건부 확산 모델(CPDM)을 도입하였습니다. 이는 CT-PET 변환을 위한 확산 모델의 초기 시도 중 하나로, 기존의 PET 이미징 방법의 제약을 해결하고자 합니다.

- **Technical Details**: CPDM 모델은 Attention map 및 Attenuation map이라는 두 가지 조건부 맵을 통합하여 diffusion 과정의 방향성을 제공합니다. 이 과정에서 Attention map은 PET 관심 영역을 강조하고, Attenuation map은 511 keV의 광자 감쇠를 계산하여 PET 데이터 정정을 향상시킵니다.

- **Performance Highlights**: CPDM은 여러 메트릭에서 기존의 최신 기술(SOTA) 방법을 초월하여 높은 품질의 PET 이미지를 생성할 수 있음을 실험적으로 입증했습니다. 이 모델은 2,028,628 쌍의 CT-PET 이미지를 포함하는 대규모 데이터셋을 기반으로 테스트되었습니다.



### A Longitudinal Analysis of Racial and Gender Bias in New York Times and Fox News Images and Articles (https://arxiv.org/abs/2410.21898)
Comments:
          13 pages, and 11 figures

- **What's New**: 본 연구는 온라인 뉴스 기사에서 다양한 인종 및 성별 집단의 등장 빈도와 그 맥락을 분석하는 최초의 종단적(longitudinal) 연구로, 머신러닝(classifier)을 활용하여 인종과 나이를 감지하는 두 가지 새로운 모델을 제안합니다.

- **Technical Details**: 연구는 123,337개의 이미지와 441,321개의 온라인 뉴스 기사를 뉴욕 타임스(New York Times)와 폭스 뉴스(Fox News)에서 수집하여 분석합니다. 연구에 사용된 머신러닝 모델은 최신 기술을 도입하여 인종 및 나이를 정확하게 분류하며, 텍스트 기반의 분석 방법과 결합하여 이중으로 인종 및 성별의 표현을 다룹니다.

- **Performance Highlights**: 결과적으로, 인종 및 성별 소수자 집단은 전체적으로 나타나는 빈도가 매우 낮았으며, 나타날 때에도 남자 집단에 비해 상대적으로 덜 두드러지게 표현되었습니다. 뉴욕 타임스는 폭스 뉴스에 비해 소수자 집단의 이미지를 더 자주 포함하고 있으며, 감정적 측면에서 보았을 때 두 언론은 담아내는 감정의 차이가 있음을 알 수 있습니다.



### Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Models (https://arxiv.org/abs/2410.21815)
- **What's New**: 본 논문에서는 self-interpretable 모델과 black-box 모델을 위한 post-hoc 설명 사이의 간극을 메우기 위한 새로운 방법인 *AutoGnothi*를 제안합니다. 이 방법은 블랙박스 모델이 예측 정확성을 손상시키지 않으면서 자가 해석 가능성을 이론적으로 보장하는 것을 목표로 합니다.

- **Technical Details**: *AutoGnothi*는 parameter-efficient transfer learning (PETL)을 활용하여 블랙박스 모델에 작은 사이드 네트워크를 통합합니다. 이를 통해 Shapley value 기반의 설명을 생성할 수 있으며, 원래 네트워크의 매개변수를 변경하지 않고도 메모리, 훈련 및 추론 비용을 크게 줄일 수 있습니다. 이 방법은 두 개의 모델로 구성된 기존 방식과 달리 예측과 설명을 동시에 수행하며, 단일 추론 단계에서 작업을 완료합니다.

- **Performance Highlights**: 실험 결과 *AutoGnothi*는 매우 높은 훈련 및 추론 효율성을 보여 주었습니다. 예를 들어, ViT-base 모델에 대해 97%의 훈련 가능 매개변수 감소, 72%의 훈련 메모리 감소를 기록하며, 설명 생성에 있어서 54%의 추론 비용 감소와 44%의 추론 시간 감소를 달성했습니다. 이러한 결과는 *AutoGnothi*가 블랙박스 모델의 예측 성능을 유지하면서 자기 해석 가능성을 강화한다는 것을 일증합니다.



### Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging (https://arxiv.org/abs/2410.21804)
- **What's New**: 이번 연구에서는 Weight-Ensembling Mixture of Experts (WEMoE) 방법을 제안하여 멀티태스크 모델 병합을 동적으로 수행합니다. 이 방법은 Transformer 기반 모델의 중요 모듈을 식별하고, 비중요 모듈은 정적 병합하고 중요 모듈은 Mixture-of-Experts (MoE) 구조로 전환합니다.

- **Technical Details**: WEMoE는 비중요 모듈을 정적으로 병합하고, 중요 모듈은 입력 샘플에 기반하여 MoE 전문가 모듈의 가중치를 동적으로 할당하는 방식으로 작동합니다. 특히, MLP 모듈의 매개변수 변화를 분석하여 전문 지식의 전이와 태스크 간의 충돌을 줄이는 데 중점을 둡니다. 이어서 E-WEMoE를 소개하여 비핵심 요소를 제거하고 모듈 간 라우팅을 공유함으로써 전체 매개변수 수와 계산 오버헤드를 줄입니다.

- **Performance Highlights**: WEMoE와 E-WEMoE는 다양한 아키텍처와 태스크에서 SOTA 성능을 달성하여 멀티태스크 학습 성능, 일반화 능력 및 강건성을 향상시켰습니다. 실험 결과, 두 방법 모두 많은 태스크에서 독립적으로 조정된 모델과 유사한 성능을 보여주었습니다.



### Super-resolution in disordered media using neural networks (https://arxiv.org/abs/2410.21556)
- **What's New**: 이 논문에서는 강한 산란 매체(strongly scattering media)에서 대규모 및 다양한 데이터 세트를 활용하여 주변 매질의 그린 함수(Green's functions)를 정확하게 추정하는 방법론을 제안합니다.

- **Technical Details**: 제안된 방법은 신경망(neural networks)을 사용하거나 사용하지 않고도 그린 함수를 추정합니다.

- **Performance Highlights**: 이 방법으로 얻은 이미징 결과는 동질 매질(homogeneous medium)에서보다 더 나은 해상도를 보여주는 우수한 성과를 달성하였습니다. 이 현상은 슈퍼 해상도(super-resolution)로도 알려져 있으며, 주변 산란 매질이 물리적 이미징 아펠쳐(physical imaging aperture)를 효과적으로 향상시키기 때문에 발생합니다.



### AiSciVision: A Framework for Specializing Large Multimodal Models in Scientific Image Classification (https://arxiv.org/abs/2410.21480)
- **What's New**: AiSciVision은 과학 연구에서의 인공지능(AI) 사용을 위한 신뢰성과 해석 가능성을 제공하는 새로운 프레임워크입니다. 이 프레임워크는 대형 다중모달 모델(LMM)을 인터랙티브 연구 파트너로 변환하고 이미지 분류 작업을 위한 모델로 특화되었습니다.

- **Technical Details**: AiSciVision은 두 가지 주요 구성 요소로 이루어져 있습니다: (1) Visual Retrieval-Augmented Generation (VisRAG)과 (2) 도메인 특정 도구입니다. 이 시스템은 이미지 분류를 위해 유사한 이미지(양성 및 음성 레이블이 붙은)를 검색하고, LMM 에이전트가 이를 기반으로 도구를 선택하여 여러 번에 걸쳐 이미지를 조작하고 검토합니다.

- **Performance Highlights**: AiSciVision은 세 가지 실제 과학 이미지 분류 데이터셋에서 평가되었으며, 수조(aquaculture) 연못, 병든 이름풀이(eelgrass), 태양광 패널의 존재를 탐지하는 작업에서 완전 감독 모델보다 더 나은 성능을 보였습니다.



### Decoding Diffusion: A Scalable Framework for Unsupervised Analysis of Latent Space Biases and Representations Using Natural Language Prompts (https://arxiv.org/abs/2410.21314)
- **What's New**: 최근의 이미지 생성 알고리즘 발전으로 인해 순방향 확산 모델(Diffusion Models)이 고 품질의 이미지를 생성하는 강력한 도구로 자리잡았습니다. 본 논문은 이러한 모델의 의미론적 잠재 공간(semantic latent space)을 자동으로 탐색할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 자연어 프롬프트와 이미지 캡션을 직접 활용하여 잠재 방향(latent direction)을 매핑합니다. 이 접근법은 특정 벡터를 훈련할 필요 없이, 의미론적 정보(semantic information)를 활용하여 잠재 공간의 자동 이해를 가능하게 합니다. 또한, Latent Consistency Model(LCM)을 활용하여 중간 U-Net 레이어의 출력인 h-space를 샘플링합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 도메인에서 숨겨진 패턴과 연관성을 발견하며, 순방향 확산 모델의 의미론적 지식(semantic knowledge)과 편향(bias)을 분석하는 데 뛰어난 성능을 보였습니다. 이로 인해 해석할 수 있는 더욱 투명한 생성 모델을 위한 가능성을 열었습니다.



### Evaluating the Posterior Sampling Ability of Plug&Play Diffusion Methods in Sparse-View C (https://arxiv.org/abs/2410.21301)
- **What's New**: 이 논문은 Plug&Play (PnP) 확산 모델의 사후 샘플링 능력을 평가하는 데 초점을 맞추고 있으며, 측정된 sinogram이 충분한 정보를 포함하지 않을 때의 문제를 다룹니다.

- **Technical Details**: Sparse-View Computed Tomography (SVCT) 모델을 사용하여 sinogram으로부터 이미지를 재구성하는 과정에서, 적은 수의 투사(≤ 180)로 인해 사후 분포가 피크를 이루지 않거나 다중 모드(multi-modal)가 되는 상황을 평가합니다. 기존의 PnP 확산 모델은 종종 피크가 있는 분포를 가정하였고, 이에 대한 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: 실험 결과, 각 PnP 모델은 투사 수가 줄어들 때마다 약한 사후 분포와 실제 사후 분포 간의 차이가 발생하는 경향을 보였습니다. 이는 PnP 모델의 성능 평가를 위한 새로운 기준의 필요성을 강조합니다.



### Grounded GUI Understanding for Vision Based Spatial Intelligent Agent: Exemplified by Virtual Reality Apps (https://arxiv.org/abs/2409.10811)
- **What's New**: 이 논문에서는 가상 현실(VR) 애플리케이션에서의 상호작용 가능한 GUI 요소(IGE) 인식을 위한 새로운 접근 방식인 Orienter를 제안합니다. Orienter는 사람의 행동을 모방하여 VR 앱 장면의 의미적 맥락을 이해한 후 감지를 수행합니다.

- **Technical Details**: 이 프레임워크는 세 가지 주요 모듈로 구성됩니다: 1) 의미적 맥락 이해(Semantic context comprehension), 2) 반영 지향 IGE 후보 탐지(Reflection-directed IGE candidate detection), 3) 맥락 감수성 상호 작용 가능성 분류(Context-sensitive interactability classification). 이 시스템은 LMM(사전 훈련된 대규모 다중 모달 모델)을 활용하여 각 VR 장면의 전반적 및 국소적 맥락을 결합합니다.

- **Performance Highlights**: Orienter는 기존의 GUI 요소 탐지 방법들보다 뛰어난 성능을 보이며, VR 앱에서의 IGE 탐지를 획기적으로 개선할 수 있음을 실험을 통해 입증합니다.



New uploads on arXiv(cs.AI)

### A Methodology for Gradual Semantics for Structured Argumentation under Incomplete Information (https://arxiv.org/abs/2410.22209)
- **What's New**: 이 논문에서는 구조화된 논증 프레임워크(structured argumentation frameworks)에 대한 점진적 의미론(gradual semantics)을 도출하기 위한 새로운 방법론을 제시합니다.

- **Technical Details**: 제안된 방법론은 논증과 그 간의 관계가 알려진 구조화된 논증 프레임워크에 대해 점진적 의미론을 구축하는 것을 목표로 하며, 주어진 과제와 맥락에 따라 점진적 의미론을 구성할 수 있는 모듈형 특성을 갖추고 있습니다. 이 방법론에서는 먼저 주장(statement) 그래프를 통해 진술의 전제와 주장을 구성하고, 기존 점진적 의미론을 활용하여 중간 평가를 수행한 후, 이를 최종 힘(strength)으로 집계합니다.

- **Performance Highlights**: 제안된 방법론을 통해 두 가지 구체적인 점진적 의미론 구현을 소개하고, 이들이 기존 QBAFs에 대한 점진적 의미론 및 구조화된 논증에 비해 가지는 강점과 한계를 이론적으로 분석하였습니다. 이는 실제 정보가 불완전한 경우에도 유용하게 적용될 수 있는 가능성을 보여줍니다.



### Democratizing Reward Design for Personal and Representative Value-Alignmen (https://arxiv.org/abs/2410.22203)
Comments:
          19 pages, 16 figures

- **What's New**: AI 에이전트의 행동을 인간의 가치와 조화롭게 맞추는 방법을 제시합니다. 주요 기법으로는 사용자가 자신의 가치 정의를 반영하고 명시하도록 하는 Interactive-Reflective Dialogue Alignment를 도입했습니다.

- **Technical Details**: 사용자가 대화형 인터페이스를 통해 원하는 행동 패턴을 설명하고, AI 에이전트의 행동 예시를 보내 피드백을 요청하는 방식으로 작동합니다. 이 시스템은 사용자의 피드백을 바탕으로 언어 기반 보상 모델을 생성합니다.

- **Performance Highlights**: 30명의 참여자를 대상으로 한 두 가지 연구에서 각 개인의 가치 정립을 정확하게 반영하는 결과를 보였습니다. 또한, 가치 정렬 행동에 대한 다양성을 보여주면서 각 개인의 고유한 이해를 접수할 수 있는 능력을 입증했습니다.



### ADAM: An Embodied Causal Agent in Open-World Environments (https://arxiv.org/abs/2410.22194)
- **What's New**: 본 논문에서는 Minecraft 환경에서 자율적으로 탐색하고, 다중 모달 컨텍스트를 인식하며, 인과 세계 지식을 학습하여 복잡한 작업을 해결할 수 있는 ADAM(An emboDied causal Agent in Minecraft)을 소개합니다. ADAM은 기존 모델의 한계를 극복하고 더 나은 해석 가능성과 일반화 능력을 제공합니다.

- **Technical Details**: ADAM은 네 가지 주요 모듈로 구성됩니다: 1) Interaction Module - 에이전트가 행동을 실행하고 상호작용 과정을 문서화합니다. 2) Causal Model Module - 인과 그래프를 구성하여 해석 가능성을 높이고 사전 지식에 대한 의존도를 줄입니다. 3) Controller Module - planner, actor 및 memory pool로 구성되어, 학습한 인과 그래프를 기반으로 작업을 수행합니다. 4) Perception Module - 다중 모달 대형 언어 모델(MLLMs)을 사용하여 ADAM이 인간 플레이어처럼 주변을 인식하도록 합니다.

- **Performance Highlights**: ADAM은 기존 최첨단(SOTA) 방법과 비교하여 다이아몬드 획득 작업에서 2.2배의 속도를 보였으며, 공예 레시피가 수정된 경우에도 안정적으로 다이아몬드를 획득하는 유일한 방법으로, 4.6배의 속도를 기록했습니다. ADAM은 거의 완벽한 기술 트리를 구성하여 해석 가능성을 높이고, 인간 플레이 방식에 근접한 성능을 유지했습니다.



### Solving Epistemic Logic Programs using Generate-and-Test with Propagation (https://arxiv.org/abs/2410.22130)
- **What's New**: 이 논문은 에피스테믹 논리 프로그램(epistemic logic programs)을 위해 생성-테스트 기반(solvers for generate-and-test-based) 솔버의 일반적인 프레임워크를 도입합니다. 또한 이 프레임워크를 사용하여 구축된 솔버의 정확성을 보장하는 충분 조건을 증명합니다.

- **Technical Details**: 새로운 생성기 프로그램(generator program)을 도입하여 에피스테믹 결과(epistemic consequences)의 전파(propogation)를 포함합니다. 이로 인해 테스트해야 할 후보(candidates)의 수를 기하급수적으로 줄일 수 있으며, 단지 선형적인 오버헤드(overhead)가 발생합니다.

- **Performance Highlights**: 새로운 솔버는 이론적 발견에 기반하여 구현되었으며, 기존 솔버 대비 약 3.3배 속도 향상(speed-up)을 달성하고, 잘 알려진 벤치마크(benchmarks)에서 91% 더 많은 인스턴스(instances)를 해결하는 성능을 보여주었습니다.



### Mapping the Neuro-Symbolic AI Landscape by Architectures: A Handbook on Augmenting Deep Learning Through Symbolic Reasoning (https://arxiv.org/abs/2410.22077)
Comments:
          57 pages

- **What's New**: 이번 논문에서는 neuro-symbolic AI의 기법을 아키텍처 기반의 여러 프레임워크 패밀리와 연계하여 처음으로 체계적으로 정리했습니다. 이러한 통합은 각 프레임워크의 강점을 구조와 연결하고, 심볼릭 기법을 블랙박스처럼 사용하면서 신경망을 확장할 수 있는 방법을 제시합니다.

- **Technical Details**: Neuro-symbolic AI는 신경망(neural networks)과 논리 모델(logical models)의 결합을 통해 기계 학습(machine learning) 모델의 한계를 극복하는 것을 목표로 합니다. 이 연구에서는 Structured Relational Learning(SRL)과 같은 확률론적(framework) 방법론과의 통합을 통해 신경망의 데이터 필요성(data need), 해석 가능성(explainability), 그리고 안전 보장(guarantees)의 한계를 다루고 있습니다.

- **Performance Highlights**: 여러 neuro-symbolic 프레임워크가 기존 모델과 비교하여 정확도가 높아지고 모델 복잡성이 낮아지며, 백그라운드 지식을 통합하여 데이터 요구량이 줄어들었습니다. 그러나 machine learning 실무자들 사이에서는 아직까지 널리 채택되지 않고 있는 상황입니다.



### Path-based summary explanations for graph recommenders -- extended version (https://arxiv.org/abs/2410.22020)
Comments:
          Extended Version - This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문에서는 사용자 또는 사용자 그룹이 특정 항목 추천 세트를 받는 이유와 항목 또는 항목 그룹이 특정 사용자 집단에게 추천되는 이유를 강조하는 요약 설명(summary explanations)을 제안합니다. 이는 추천 시스템의 집단 행동에 대한 통찰을 제공하는 효과적인 방법으로 소개됩니다.

- **Technical Details**: 제안된 방법은 효율적인 그래프 알고리즘, 특히 Steiner Tree 및 Prize-Collecting Steiner Tree를 사용하여 요약 설명을 생성하는 새로운 방법입니다. 이를 통해 요약 설명의 크기와 복잡성을 줄이면서도 필수 정보를 유지할 수 있습니다.

- **Performance Highlights**: 여러 메트릭을 통해 평가한 결과, 제안한 요약 방법이 대부분의 시나리오에서 기존 설명 방법보다 우수한 성능을 보임을 보여주었습니다. 특히, 이해도(comprehensibility), 실행 가능성(actionability), 중복성(redundancy), 관련성(relevance) 등의 지표에서 개선된 성능을 나타냈습니다.



### Building Altruistic and Moral AI Agent with Brain-inspired Affective Empathy Mechanisms (https://arxiv.org/abs/2410.21882)
- **What's New**: 이번 연구에서는 AI의 의사결정 과정을 윤리적이고 이타적으로 전환하기 위해, 감정적 공감(affective empathy) 메커니즘을 활용하는 새로운 접근법을 제안합니다. 이를 통해 AI가 인간과 유사한 방식으로 도덕적 행동을 자발적으로 수행할 수 있도록 하는 모델을 구성하였습니다.

- **Technical Details**: 제안하는 모델은 거울 뉴런 시스템(mirror neuron system)을 기반으로 하여, AI가 타인의 감정을 경험하고 이를 통해 내재적 이타적 동기를 발전시킬 수 있도록 합니다. 도파민(dopamine) 분비가 공감의 수준에 영향을 미치고, 이는 도덕적 결정을 내리는 데 있어 중요한 역할을 하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 높은 수준의 감정적 공감을 가진 에이전트는 타인의 고통을 덜기 위해 자신의 이해관계를 희생하는 경향이 있음을 확인하였습니다. 이는 심리적 행동 실험에서 보고된 결과와 일치하며, 제안된 모델의 유효성을 뒷받침합니다.



### Robot Policy Learning with Temporal Optimal Transport Reward (https://arxiv.org/abs/2410.21795)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 Temporal Optimal Transport (TemporalOT) 보상을 도입하여 강화 학습에서의 보상 명세 문제를 해결하는 방법을 제시합니다. 기존 연구들이 Temporal Order 정보의 중요성을 간과하고 있다는 점을 지적하였으며, 이를 보완하기 위한 알고리즘을 설계하였습니다.

- **Technical Details**: TemporalOT 보상은 전문가 비디오 시연과 로봇 궤적간의 정렬을 측정하여 프로시된 보상을 생성하는 데 Optimal Transport를 바탕으로 하며, 컨텍스트 임베딩과 마스크 메커니즘을 통해 시간 순서 정보를 통합합니다. 이 방법은 강화 학습의 표준 Markov Decision Process (MDP) 설정을 기반으로 합니다.

- **Performance Highlights**: Meta-world 벤치마크 과제를 통한 광범위한 실험을 통해 제안된 방법이 기존의 SOTA(State-Of-The-Art) 알고리즘 보다 우수한 성능을 보임을 확인하였습니다.



### Inverse Attention Agent for Multi-Agent System (https://arxiv.org/abs/2410.21794)
- **What's New**: 이번 논문에서는 다중 에이전트 시스템에서 에이전트가 동적으로 다양한 환경에 적응할 수 있도록 하는 새로운 방법을 제안합니다. 특히, Inverse Attention Agents라는 개념을 도입하여 Theory of Mind의 원리를 활용합니다.

- **Technical Details**: Inverse Attention Agents는 attention mechanism을 사용하여 알고리즘적으로 구현되며, end-to-end 방식으로 학습됩니다. 에이전트의 최종 행동을 결정하는 데 중요한 attention model의 가중치는 다양한 목표에 대한 주의를 명확히 나타냅니다. 또한 관찰 및 이전 행동을 기반으로 다른 에이전트의 ToM을 추론하는 inverse attention network를 제안합니다.

- **Performance Highlights**: 실험 결과, inverse attention network는 다른 에이전트의 주의를 성공적으로 추론하여 에이전트의 성능을 향상시켰습니다. 추가적인 인간 실험을 통해, baseline 에이전트 모델에 비해 Inverse Attention Agents가 인간과의 협력에서 우수한 결과를 나타내고 인간 행동을 더 잘 모방함을 보여주었습니다.



### MARCO: Multi-Agent Real-time Chat Orchestration (https://arxiv.org/abs/2410.21784)
Comments:
          EMNLP 2024 Industry Track

- **What's New**: MARCO는 LLM을 이용한 다중 에이전트 실시간 채팅 조정 프레임워크로, 복잡한 작업을 자동화하는 새로운 접근 방식을 제시합니다. 특히 여러 도구와의 상호작용 및 다중 단계의 작업 실행을 지원하여 실시간으로 사용자와 상호작용할 수 있는 기능을 탑재하고 있습니다.

- **Technical Details**: MARCO는 LLM의 불확실성과 오류를 극복하기 위해 견고한 가드레일(guardrails)을 도입하였으며, 신뢰성 높은 작업 실행을 위해 자연어로 표현된 작업 실행 절차(TEP)를 사용합니다. 에이전트 간 장기 메모리를 공유하는 구조로 설계되어, 동적인 정보와 대화 상황을 포함한 완전한 컨텍스트 정보를 저장합니다. 다양한 LLM 모델을 비교하여 최적의 성능을 도출할 수 있도록 구성되어 있습니다.

- **Performance Highlights**: 테스트 결과, MARCO는 Digital Restaurant Service Platform 대화 및 Retail 대화 데이터셋에서 각각 94.48%와 92.74%의 높은 정확도를 기록했으며, 44.91%의 지연 개선(latency improvement)과 33.71%의 비용 감소(cost reduction)를 달성했습니다. 가드레일의 효과를 통해 성능 향상이 이루어졌으며, 여러 LLM 모델에 대한 비교 분석도 포함되어 있습니다.



### Asynchronous Tool Usage for Real-Time Agents (https://arxiv.org/abs/2410.21620)
- **What's New**: 최신 논문에서는 현재 대형 언어 모델(LLM)의 비동기적(Aynchronous) 작동을 통해 사용자와의 상호작용 및 도구 사용을 개선하는 접근 방식을 제시합니다. 기존의 AI 시스템이 순차적으로 운영되는 갇힌 상태에서 벗어나 실시간으로 다중 작업을 수행할 수 있는 에이전트를 구현한 것이 주요 관점입니다.

- **Technical Details**: 주요 기여는 이벤트 기반의 유한 상태 기계(Event-Driven Finite-State Machine) 아키텍처를 통해 비동기적 에이전트를 설계하는 것입니다. 이 아키텍처는 자동 음성 인식(Automatic Speech Recognition) 및 텍스트 음성 변환(Text-to-Speech)과 통합되어 있으며, 어떤 LLM과도 호환되어 모듈화된 시스템을 구현할 수 있습니다. Llama 3.1과 GPT-4o를 통해 최적화하고 성능을 평가하였습니다.

- **Performance Highlights**: 논문에서는 이 시스템이 통해 사용자 경험이 크게 향상될 것이라는 점을 강조하며, 에이전트와의 상호작용 중 지연을 감소시키고, 전반적인 응답 속도를 높일 것으로 보입니다. 비동기적인 도구 사용을 통해 AI 에이전트는 사용자의 요청에 실제시간으로 반응할 수 있으며, 이는 기존의 AI 시스템의 한계를 극복하는 데 큰 도움이 됩니다.



### Can Large Language Models Replace Data Scientists in Clinical Research? (https://arxiv.org/abs/2410.21591)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 활용하여 임상 연구의 데이터 과학 태스크를 자동화하는 가능성을 평가하는 새로운 데이터 셋을 개발했습니다. 이 데이터 셋은 39개의 임상 연구를 기반으로 하여 293개의 실제 데이터 과학 코딩 작업으로 구성되어 있습니다.

- **Technical Details**: 데이터 셋은 Python과 R에서 128개의 작업과 165개의 작업을 포함하며, 임상 데이터 분석을 위한 실제 시나리오를 시뮬레이션합니다. 연구에서 확인된 고급 적응 방법으로는 chain-of-thought prompting과 self-reflection이 있으며, 각각 코드 정확성을 60%와 38% 향상시켰습니다.

- **Performance Highlights**: 사용자 연구 결과, 의료 전문가들이 제출한 코드 솔루션의 80%가 LLM이 생성한 코드에서 가져온 것이었고, 일부 경우 최대 96%의 재사용이 발견되었습니다. 이 결과는 LLMs가 전문가 워크플로우와 통합될 때 데이터 과학 효율성을 향상시키는 잠재력을 가진다는 것을 강조합니다.



### Large Language Models for Manufacturing (https://arxiv.org/abs/2410.21418)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 제조 산업에 통합될 잠재력과 그로 인해 얻을 수 있는 혜택을 종합적으로 탐구합니다. LLMs는 생산 설계부터 품질 관리, 공급망 최적화, 인재 관리까지 다양한 제조 과정의 자동화 및 향상을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 최첨단 LLM인 GPT-4V의 성능을 여러 제조 작업에서 평가하며, 텍스트 처리, 데이터 분석, 코드 생성 및 제로샷 학습(Zero-shot learning) 등에서의 장점을 강조합니다. LLMs는 맞춤화된 데이터 세트로 훈련시켜 특정 제조 지식을 더 잘 습득할 수 있도록 할 수 있습니다.

- **Performance Highlights**: LLMs는 데이터 처리 및 분석에서 매우 유용하며, 심층 인사이트 및 실행 가능한 권장 사항을 생성할 수 있는 도구로서의 가치를 점차 입증하고 있습니다. 그러나 제조 설계와 같은 직접 계산 역할에서는 한계를 가지고 있으며 지원적인 역할이 주를 이룹니다. LLM의 적용은 지속적인 기술 발전을 통해 향상될 수 있습니다.



### Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datas (https://arxiv.org/abs/2410.22325)
- **What's New**: 이번 연구는 로봇 비주얼 표현에 대한 사전 학습(pre-training)에서 새로운 접근 방식을 제안하며, 매니퓰레이션 중심의 표현(Manipulation Centric Representation, MCR) 프레임워크를 통해 로봇 조작 작업의 성능을 향상시키는 방법을 탐구합니다.

- **Technical Details**: MCR은 로봇의 시각적 특징과 조작 작업의 행동 및 고유 감각(proprioceptive) 정보를 모두 포착하는 기반 표현 학습 프레임워크입니다. 이 방법은 DROID 로봇 데이터셋에서 비주얼 인코더를 사전 학습하고, 로봇의 고유 상태 및 행동 데이터를 활용하여 새로운 대조 손실(contrastive loss)을 도입합니다. 이 손실은 시각적 관찰을 로봇의 고유 상태-행동 동역학과 정렬시킵니다. 

- **Performance Highlights**: 실험 결과, MCR은 약 20개의 다양한 로봇 작업을 포함하는 4개의 시뮬레이션 도메인에서 가장 강력한 기준 방법보다 14.8% 성능을 향상시켰으며, 3개의 실제 로봇 작업에서는 76.9%의 성능 향상을 보였습니다.



### An Efficient Approach to Generate Safe Drivable Space by LiDAR-Camera-HDmap Fusion (https://arxiv.org/abs/2410.22314)
- **What's New**: 본 논문에서는 자율주행차량(Autonomous Vehicles, AVs)을 위해 정확하고 견고한 인식 모듈을 제안합니다. 이 모듈은 LiDAR, 카메라, 고화질(HD) 지도 데이터의 융합을 통해 다양한 기상 조건에서도 안전하고 신뢰할 수 있는 주행 가능 공간을 제공합니다.

- **Technical Details**: 이 연구는 다양한 환경에서의 신뢰성 향상을 목표로 하며, 적응형 바닥 제거 및 실행 가능한 도로 경계선(커브) 감지 방식을 통합하여 장애물 탐지의 신뢰성을 높였습니다. 또한, 강수 노이즈에 최적화된 적응형 DBSCAN 클러스터링 알고리즘과 교정 불일치에 강한 LiDAR-카메라 프러스텀 연결 방법을 제안하였습니다.

- **Performance Highlights**: 이 접근 방식은 실제 데이터셋을 기반으로 테스트되었으며, 자율주행 셔틀 WATonoBus의 일상적인 운영에서 신뢰성을 입증하였습니다. 이러한 방법은 자율주행차량 운전의 안전성을 크게 향상시키며, 도로 규정 및 차량 크기와의 호환성을 보장합니다.



### Effective Guidance for Model Attention with Simple Yes-no Annotations (https://arxiv.org/abs/2410.22312)
Comments:
          10 pages, 5 figures, IEEE BigData 2024 Paper

- **What's New**: 최근 딥러닝 모델들은 예측 시 irrelevant areas에 집중하여 성능 저하와 일반화 제한을 초래하는 경향이 있습니다. 이러한 문제를 해결하기 위해 CRAYON을 제안합니다. CRAYON은 간단한 yes-no 주석을 활용하여 모델의 주의를 수정하는 효과적이고 실용적인 방법입니다.

- **Technical Details**: CRAYON은 두 가지 모드로 작동합니다: CRAYON-ATTENTION은 saliency maps를 기반으로 모델 해석을 돕고, CRAYON-PRUNING은 irrelevant neurons를 제거하여 모델 성능을 개선합니다. CRAYON은 10%의 훈련 데이터에 대한 주석만으로도 최첨단 성능을 달성할 수 있습니다.

- **Performance Highlights**: CRAYON은 3개의 벤치마크 데이터셋에서 12개의 기존 방법보다 우수한 성능을 보이며, 약 6,000명의 참여자가 포함된 대규모 인간 주석을 바탕으로 효과성과 확장성을 입증했습니다.



### SVIP: Towards Verifiable Inference of Open-source Large Language Models (https://arxiv.org/abs/2410.22307)
Comments:
          20 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 검증 가능한 추론 문제를 형식화하고, 이용자와 컴퓨팅 제공자 간의 신뢰성을 높이기 위해 중간 출력을 활용한 비밀 기반의 검증 가능한 LLM 추론 프로토콜(SVIP)을 제안합니다.

- **Technical Details**: SVIP는 LLM의 중간 출력(hidden state representations)을 고유한 모델 식별자로 사용하여, 컴퓨팅 제공자가 생성한 출력이 실제 요청된 LLM에서 온 것인지 검증할 수 있는 mecanismos입니다. 이 방법은 컴퓨팅 제공자에게 생성된 텍스트와 프로세스된 중간 출력을 반환하도록 요구하여, 사용자에게 신뢰할 수 있는 검증 수단을 제공합니다.

- **Performance Highlights**: SVIP는 평균 3.49%의 잘못된 부정률(false negative rate) 및 3% 미만의 잘못된 긍정률(false positive rate)을 유지하며, 한 쿼리에 대해 0.01초 이하의 검증 시간을 요구합니다. 본 프로토콜은 약 80에서 120만 건의 프롬프트 쿼리를 안전하게 처리할 수 있도록 설계되었습니다.



### $\mathsf{OPA}$: One-shot Private Aggregation with Single Client Interaction and its Applications to Federated Learning (https://arxiv.org/abs/2410.22303)
Comments:
          To appear at the NeurIPS 2024 FL@FM workshop

- **What's New**: 이번 연구에서는 단일 서버 노드에서 안전한 집합체 집합(Secure Aggregation) 문제를 재검토하여, 많은 클라이언트가 참여하는 시나리오에서 통신 비용을 최소화하기 위한 새로운 방법을 제시합니다.

- **Technical Details**: 우리의 주요 기여는 One-shot Private Aggregation (OPA)입니다. 이 방법에서는 각 클라이언트가 집합체 평가(in aggregation evaluation)당 한 번만 통신하거나 아예 통신하지 않을 수 있습니다. OPA는 LWR, LWE, class groups, DCR를 기반으로 구축되었으며, 클라이언트가 한 번만 통신하는 프라이버시 보호 연합 학습(Federated Learning)에도 적용됩니다.

- **Performance Highlights**: OPA는 기존의 다중 라운드 프라이버시 보호 연합 학습 프로토콜과 비교하여 비약적으로 간편하며 효율성을 극대화합니다. 우리는 두 개의 데이터셋을 대상으로 로지스틱 회귀(classifiers for logistic regression) 성능을 평가하고, MNIST, CIFAR-10, CIFAR-100 데이터셋에 대한 MLP 분류기를 훈련시키며 OPA의 우수성을 입증합니다.



### From melodic note sequences to pitches using word2vec (https://arxiv.org/abs/2410.22285)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문에서는 언어 모델링에 일반적으로 사용되는 word2vec 기법을 멜로디에 적용하는 새로운 접근 방식을 소개합니다. 음표가 문장에서 단어로 간주되며, 이를 통해 피치(pitch) 정보를 효과적으로 캡처합니다.

- **Technical Details**: 본 연구는 20개의 어린이 노래와 바흐 소나타의 발췌본, 두 개의 데이터셋을 사용하였습니다. 그들의 임베딩(embeddings)은 2차원으로 정의된 매우 작은 차원의 의미 공간(semantic space)을 사용하며, 2, 3 또는 4개의 이전 음표에 기반하여 음표를 예측합니다.

- **Performance Highlights**: 다변량 분석(multivariate analysis) 결과, 음표를 나타내는 의미 벡터는 피치와 약 0.80의 다중 상관 계수를 보여주며, 이는 모델이 음표의 피치를 잘 예측하고 있음을 시사합니다.



### Leveraging Reverberation and Visual Depth Cues for Sound Event Localization and Detection with Distance Estimation (https://arxiv.org/abs/2410.22271)
- **What's New**: 본 보고서는 DCASE2024 Task 3 챌린지에 제출된 시스템을 설명하며, 오디오-비주얼 (Audio-Visual) Conformer 모델을 중심으로 구성되어 있습니다. 이 모델은 ResNet50으로 추출된 비디오 및 오디오 임베딩을 처리하며, SELD에 대해 사전 훈련된 오디오 인코더를 사용하고 있습니다.

- **Technical Details**: 주요 모델은 오디오-비주얼 Conformer를 사용하고, 이를 통해 비디오 프레임과 오디오 신호를 동시에 처리하여 사운드 이벤트를 식별합니다. 이 시스템은 N=3개의 트랙 각각에 대해 4개의 포지셔널 값 (x, y, z, 거리)을 예측하며, 13개의 사운드 클래스를 포함하여 156차원을 생성합니다. 또한, 시스템은 DCASE 챌린지에서의 성능을 향상시키기 위해 새로운 거리 예측 기능을 추가하였습니다.

- **Performance Highlights**: 제출된 첫 번째 시스템은 STARSS23 개발 세트에서 기존의 오디오-비주얼 기준을 크게 초과하여 DOAE를 절반으로 줄이고 F1 점수를 3배 이상 개선했습니다. 두 번째 시스템은 거리 예측 기능을 강화했지만 F1 점수는 낮아졌으며, 이 문제를 해결하기 위해 모든 변형의 예측을 결합하는 앙상블 전략으로 네 번째 시스템이 구현되었습니다.



### Fourier Head: Helping Large Language Models Learn Complex Probability Distributions (https://arxiv.org/abs/2410.22269)
Comments:
          Project page and code are at this https URL

- **What's New**: 본 논문에서는 연속적인 구조를 고려하여, 기존 LLM(linear language model) 구조를 개선하기 위해 푸리에 변환(Fourier transformation) 기반의 새로운 신경망 레이어인 푸리에 헤드(Fourier head)를 제안합니다. 이 레이어는 고질적 소음(high-frequency noise)을 무시하면서 데이터를 통해 신호(signal)를 더 잘 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 푸리에 헤드는 입력 벡터를 받아 신경망의 선형 계층(linear layer)을 통해 푸리에 계수를 학습하고, 이를 통해 연속적인 확률 밀도 함수(continuous probability density function)를 학습합니다. 입력값은 [-1, 1] 범위에서 m개의 동등한 구간으로 양자화되며, 각 구간에서의 확률을 카테고리 분포로 반환합니다.

- **Performance Highlights**: 푸리에 헤드는 Atari Seaquest 게임에서 결정 트랜스포머(Decision Transformer)의 성과를 46% 향상시켰으며, 20개의 비훈련 데이터셋에서 3.5%의 예측 개선을 보여주어 최신 시간 시계열 모델의 성능을 초월했습니다.



### ContextIQ: A Multimodal Expert-Based Video Retrieval System for Contextual Advertising (https://arxiv.org/abs/2410.22233)
Comments:
          Accepted at WACV 2025

- **What's New**: ContextIQ는 맥락 광고(contextual advertising)를 위해 특별히 설계된 다중 모달 전문가 기반 비디오 검색 시스템입니다. 이를 통해 비디오, 오디오, 자막 등 다양한 모달리티에 기초하여 의미적으로 풍부한 비디오 표현을 생성합니다.

- **Technical Details**: ContextIQ는 전문가 모델을 활용하여 비디오 검색을 수행하며, 모달리티별 전문가(비디오, 오디오, 자막, 메타데이터)를 사용하여 비디오 콘텐츠를 이해합니다. 이 시스템은 대규모 공동 학습 없이도 최신 모델들과 비교하여 유사하거나 더 나은 성능을 보입니다.

- **Performance Highlights**: ContextIQ는 MSR-VTT, Condensed Movies 및 신규 공개된 Val-1 데이터셋에서 경쟁력 있는 성능을 보였으며, 광고 생태계에서 브랜드 안전 및 부적절한 콘텐츠 필터링 문제를 해결하며 효과적으로 통합될 수 있음을 보여줍니다.



### Drone Acoustic Analysis for Predicting Psychoacoustic Annoyance via Artificial Neural Networks (https://arxiv.org/abs/2410.22208)
Comments:
          20 Pages, 10 Figures, 4 Tables

- **What's New**: 본 연구는 무인 항공기(UAV)의 소음 문제를 해결하기 위해 다양한 Deep Learning 모델을 활용하여 Psychoacoustic Annoyance(심리음향적 불쾌감)를 예측하는 효능을 검토합니다. 이를 통해 드론의 여러 특성을 입력으로 사용하여 소음의 영향을 이해하고 노이즈 저감 기술 개발에 기여하고자 합니다.

- **Technical Details**: 연구에서는 다양한 드론 모델의 정밀 측정 데이터와 비행 데이터, 드론의 물리적 특성 및 현실적 조건에서의 심리적 불쾌감을 분석하여 훈련 데이터셋을 구성하였습니다. Linear, Support Vector Machine (SVM), Deep Neural Network (DNN), Convolutional Neural Network (CNN)의 네 가지 회귀 모델을 사용하여 DNN이 가장 효과적임을 보여줍니다.

- **Performance Highlights**: 본 연구 결과, DNN 모델이 수집된 데이터를 기반으로 드론의 소음으로 인한 불쾌감을 예측하는 데 가장 높은 성능을 보였으며, 이 접근 방식은 청중의 공공 장소에서의 드론 사용 수용을 촉진할 수 있는 기반을 마련합니다.



### Multi-Level Feature Distillation of Joint Teachers Trained on Distinct Image Datasets (https://arxiv.org/abs/2410.22184)
Comments:
          Accepted at WACV 2025

- **What's New**: 이번 논문에서는 여러 개의 서로 다른 데이터 세트로 훈련된 여러 교사로부터 지식을 증류하는 새로운 teacher-student 프레임워크를 제안합니다. 각 교사는 자신의 데이터 세트로부터 처음부터 훈련받고, 이후 모든 교사의 기능이 결합되어 공동 아키텍처를 형성합니다. 최종적으로, 이 공동 교사로부터의 지식을 student 모델로 전달하기 위해 multi-level feature distillation (MLFD) 절차를 사용합니다.

- **Technical Details**: 제안된 MLFD 방법은 세 가지 단계로 구성됩니다: 1단계에서는 각 교사가 자신의 데이터 세트에서 처음부터 학습됩니다. 2단계에서는 개별 교사들을 결합하여 공동 교사를 생성하며, 이는 모든 데이터 세트에 대해 훈련됩니다. 3단계에서는 집합적 지식을 각 데이터 세트에 맞는 student 모델로 증류합니다. 이러한 방법은 다중 수준의 표현을 사용하여 지식을 추출하고, 학습된 모델의 성능을 높입니다.

- **Performance Highlights**: 이미지 분류 및 동작 인식 실험에서 MLFD 방법이 같은 모델 아키텍처를 가진 기존 모델보다 상당한 성능 향상을 보여줍니다. 또한, 데이터 세트별 학생 아키텍처로부터의 지식 증류가 성능 향상을 가져온다는 점을 입증하였습니다.



### Natural Language Processing for Analyzing Electronic Health Records and Clinical Notes in Cancer Research: A Review (https://arxiv.org/abs/2410.22180)
- **What's New**: 이번 리뷰는 전자 건강 기록(EHR)과 임상 노트를 사용하여 암 연구에서 자연어 처리(NLP) 기술의 적용을 분석합니다. 이전 연구들보다 더 포괄적인 시각을 제공하여 특정 암 유형이나 응용 분야에 국한되지 않은 연구의 공백을 해결합니다.

- **Technical Details**: 이번 연구는 2019년부터 2024년 사이에 발표된 94개의 관련 연구를 스코퍼스(Scopus) 데이터베이스를 통해 분석했습니다. 연구들은 암 유형과 NLP 응용 분야에 따라 분류되었습니다. NLP 작업으로는 정보 추출(information extraction)과 텍스트 분류(text classification)가 두드러졌으며, 규칙 기반(rule-based)에서 변환기 기반(transformer-based) 기계 학습(machine learning) 기술로의 전환이 관찰되었습니다.

- **Performance Highlights**: 암 연구에서 NLP의 적용이 증가하고 있으며, 특히 유방암(breast cancer), 폐암(lung cancer), 대장암(colorectal cancer)이 가장 많이 연구되었습니다. 그러나 제안된 솔루션의 일반화(generalizability) 제한 및 임상 워크플로우(clinical workflows)로의 통합 필요성이 주요 도전 과제로 지적되었습니다. 향후 연구는 복잡한 임상 언어 처리의 강화를 목표로 해야 하며, 충분히 연구되지 않은 암 유형으로의 응용 확장도 필요합니다.



### Analyzing Multimodal Interaction Strategies for LLM-Assisted Manipulation of 3D Scenes (https://arxiv.org/abs/2410.22177)
Comments:
          under review

- **What's New**: 해당 논문은 LLM(대형 언어 모델) 보조 3D 장면 편집 시스템에서 사용자의 행동을 연구하여 상호작용 패턴과 잠재적 장애물을 식별하였습니다. 이를 통해 LLM 통합 3D 콘텐츠 생성 시스템의 미래 디자인에 대한 빛을 제공합니다.

- **Technical Details**: 본 연구에서 개발한 AssistVR 시스템은 Microsoft Azure Conversational Language Understanding (CLU) 서비스와 GPT-4o를 통합하여 사용자의 자연어 질의를 처리합니다. AssistVR은 사용자의 음성 명령과 레이캐스트 선택을 사용하는 다중 모드 상호작용을 통해 3D 장면을 편집하는 시스템입니다.

- **Performance Highlights**: 12명의 참가자를 대상으로 한 연구 결과, LLM 보조 시스템이 사용자의 상호작용 전략을 더 효율적으로 식별하는 데 도움을 줄 수 있음을 보여주었으며, 앞으로의 LLM 지원 인터랙티브 시스템에 대한 디자인 제안을 제공합니다.



### Standardization Trends on Safety and Trustworthiness Technology for Advanced AI (https://arxiv.org/abs/2410.22151)
Comments:
          13 pages, 2 figures, 4 tables

- **What's New**: 최근 인공지능(AI) 기술이 급격히 발전하면서 언어 이해, 이미지 및 비디오 인식, 프로그래밍, 과학적 추론 등 여러 분야에서 진전을 이뤘습니다. 대규모 언어 모델과 기초 모델을 기반으로 한 AI는 인공 일반 지능(AGI)에 근접하거나 이를 초과하는 성능을 보이고 있습니다.

- **Technical Details**: AI의 안전성과 신뢰성 확보를 위한 국제 표준화 노력이 진행되고 있으며, 연구에서는 안전성 및 신뢰성 확보를 위한 기술 영역 식별과 함께 표준화의 미래 방향성 및 전략에 대해 제안합니다. 여기서 중요하게 다루어지는 모델에는 대규모 언어 모델(LLM), 다중 모달 모델(MLLM), 일반 목적 인공지능(AGI), 초지능(ASI) 등이 포함됩니다.

- **Performance Highlights**: 이 연구는 글로벌 차원에서 AI의 안전성과 신뢰성 표준화 기술의 동향을 분석하며, 안전성과 신뢰성을 증진하기 위한 기술적 분야를 확인합니다. 궁극적으로 AI의 안전하고 신뢰할 수 있는 발전을 지원하고 국제 경쟁력을 향상시키기 위한 정책적 시사점을 도출합니다.



### Lightweight Frequency Masker for Cross-Domain Few-Shot Semantic Segmentation (https://arxiv.org/abs/2410.22135)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이번 연구에서 제안된 Cross-domain few-shot segmentation (CD-FSS) 방법은 대량의 데이터셋에서 사전 학습된 모델을 사용하여 데이터가 부족한 타겟 도메인에 대한 픽셀 수준의 분할을 수행합니다. 특히, 주목할 만한 점은 주파수 구성 요소를 필터링함으로써 성능을 14%까지 향상시킬 수 있다는 것입니다.

- **Technical Details**: 이 연구에서는 Amplitude-Phase Masker (APM)와 Adaptive Channel Phase Attention (ACPA) 모듈을 통해 경량화된 주파수 마스커를 제안합니다. APM 모듈은 추가 매개변수가 0.01%만 증가하면서 평균 mIoU 성능을 10% 이상 개선하고, ACPA는 추가 매개변수 2.5%로 평균 성능을 1.5% 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 기존의 CD-FSS 방법보다 현저하게 우수하며, 네 가지 타겟 데이터셋에서의 광범위한 실험 결과 평균 mIoU 성능이 11% 향상되었습니다. 이 연구는 간단하고 효과적인 설계를 통해 최신 CD-FSS 기법을 초월하는 성능을 구현하였습니다.



### ProMoE: Fast MoE-based LLM Serving using Proactive Caching (https://arxiv.org/abs/2410.22134)
- **What's New**: 이 논문에서는 Mixture-of-Experts (MoE) 기반의 대형 언어 모델에서 성능 문제를 해결하기 위한 새로운 시스템인 ProMoE를 제안합니다. ProMoE는 중간 모델 결과를 활용해 향후 사용될 매개변수를 예측하여 전문가들을 미리 가져오는 프로액티브(Active) 캐싱 시스템입니다.

- **Technical Details**: ProMoE는 'GOODPRED' 메트릭을 도입해 예측 정확도와 리드 타임을 동시에 고려합니다. 또한, ProMoE는 전문가 선정을 위한 학습된 예측기를 슬라이딩 윈도우 방식으로 활용하여 고품질의 예측을 달성합니다. 이 시스템은 전문가 요청과 추론 과정을 조정하여 최대한의 겹침을 만들어내며, 청크로 된 프리패치(chunked prefetch), 조기 중단(early preemption), 재정렬된 추론(reordered inference) 등의 기법을 사용합니다.

- **Performance Highlights**: ProMoE를 채택한 결과, 기존 오프로드 솔루션에 비해 평균적으로 2.13배 및 2.84배의 속도 향상을 이뤘습니다. 또한, 손수 작성한 캐싱 기반라인에 비해 프리필(pre-fill) 및 디코드(decode) 단계에서 각각 1.83배와 1.36배의 속도 향상을 기록하였습니다. 이러한 성능 향상은 오프로드의 주요 경로에서의 로드 시간 감소에 기인합니다.



### Improving Performance of Commercially Available AI Products in a Multi-Agent Configuration (https://arxiv.org/abs/2410.22129)
Comments:
          7 pages, 8 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)의 빠른 발전으로 다중 에이전트 시스템이 실용적인 응용 가능성을 갖추게 되었습니다. 특히, 소프트웨어 개발 생명 주기(SDLC)에서의 다중 에이전트 시스템의 역할에 대한 관심이 증가하고 있습니다.

- **Technical Details**: 이번 실험에서는 Crowdbotics PRD AI(소프트웨어 요구 사항 생성 도구)와 GitHub Copilot(AI 쌍 프로그래밍 도구) 간의 컨텍스트 공유를 테스트하였습니다. PRD AI에서의 비즈니스 요구 사항을 공유함으로써 GitHub Copilot의 코드 제안 능력을 13.8% 향상시키고 개발자 작업 성공률을 24.5% 개선했습니다.

- **Performance Highlights**: Crowdbotics PRD AI와 GitHub Copilot의 협업으로 실제 상용 AI 시스템이 함께 작업하여 가시적인 결과를 만들어내는 사례를 보여주었습니다.



### Vision Paper: Designing Graph Neural Networks in Compliance with the European Artificial Intelligence Ac (https://arxiv.org/abs/2410.22120)
- **What's New**: 유럽연합의 인공지능법(AI Act)은 인공지능(Artificial Intelligence) 및 머신러닝(Machine Learning) 시스템의 개발 및 감독을 위한 포괄적인 지침을 제시합니다. 본 논문은 복잡한 그래프 구조 데이터에서 운영되는 그래프 신경망(Graph Neural Networks, GNNs)에 대한 AI Act의 독특한 도전 과제를 다룹니다.

- **Technical Details**: AI Act는 데이터 관리(data management), 데이터 거버넌스(data governance), 견고성(robustness), 인간 감독(human oversight), 개인정보 보호(privacy)를 포함한 여러 요구사항을 제시합니다. 이 논문은 이러한 요구사항이 GNN 학습(training) 및 준수를 보장하기 위한 방법론에 미치는 영향을 탐구합니다.

- **Performance Highlights**: 편향(bias), 견고성(robustness), 설명 가능성(explainability), 개인정보 보호(privacy)와 같은 GNN의 측면에 대한 심층 분석을 제공합니다. 공정한 샘플링 전략(fair sampling strategies)과 효과적인 해석 기술(interpretability techniques)의 필요성을 강조하며, 새로운 법적 체제 하에서 GNN에 대한 구체적인 지침을 제공하고, 향후 연구 방향을 제시합니다.



### The Impact of Inference Acceleration Strategies on Bias of LLMs (https://arxiv.org/abs/2410.22118)
- **What's New**: 최근 몇 년간 대규모 언어 모델(LLMs)의 성능이 비약적으로 발전했습니다. 그러나 이러한 모델의 inference(추론) 효율성을 높이기 위한 많은 전략들(quantization, pruning 등)이 도입되었음에도 불구하고, 이 과정에서 데이터의 인구 통계학적 편향이 유의미하게 바뀔 수 있다는 점이 주목됩니다.

- **Technical Details**: 우리는 5가지 일반적인 inference acceleration techniques(추론 가속 기술)과 3개의 널리 사용되는 LLM을 통해 편향의 변화를 평가하였습니다. 특히, 모델 출력의 편향을 측정하기 위해 6가지 다른 bias metrics(편향 지표)를 활용하였으며, 각 모델에서 가속화 전략의 영향을 분석했습니다.

- **Performance Highlights**: 분석 결과, inference acceleration 전략이 모델의 편향에 중대한 영향을 미친다는 것을 확인했습니다. 예를 들어, 4-bit AWQ Quantization 방식이 일부 모델의 편향을 크게 변화시켰으나 KV-cache quantization 방식은 상대적으로 견고한 결과를 보여주었습니다. 따라서 가속화를 적용할 때 편향에 대한 신중한 평가가 필요하다는 점이 강조됩니다.



### Policy Gradient for Robust Markov Decision Processes (https://arxiv.org/abs/2410.22114)
- **What's New**: 이 논문은 강건한 마르코프 결정 과정(Robust Markov Decision Processes, RMDPs)을 해결하기 위한 새로운 정책 그래디언트 방법인 DRPMD(Double-Loop Robust Policy Mirror Descent)를 개발하였습니다. 이는 기존의 정책 그래디언트 방법들이 모델 불확실성에 적응하기 어려운 문제를 해결하는 데 기여하는 내용을 다루고 있습니다.

- **Technical Details**: DRPMD는 각 반복마다 적응형 허용오차(adaptive tolerance)를 사용하여 정책 최적화를 위한 일반적인 미러 강하 업데이트 규칙을 적용합니다. 이는 전역 최적 정책(global optimal policy)으로의 수렴을 보장하며, 직접 및 소프트맥스 파라미터화에 대한 새로운 수렴 결과를 포함한 포괄적인 분석을 제공합니다. 또한 RMDP의 신속한 수렴 속도를 보장합니다.

- **Performance Highlights**: 실험 결과 DRPMD는 다양한 도전적인 RMDP 설정에서 강건성과 전역 수렴성을 검증하며, 정책 그라디언트 방법 중에서 최고의 반복 복잡도(iteration complexity)를 기록하고 있습니다.



### Protecting Privacy in Multimodal Large Language Models with MLLMU-Bench (https://arxiv.org/abs/2410.22108)
Comments:
          30 pages

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLM)과 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 개인의 기밀 데이터 및 개인 정보를 기억하고 노출할 수 있다는 법적 및 윤리적 문제를 다루고 있습니다. 특히, MLLMs에 대한 머신 언러닝(machine unlearning) 적용이 미비하였음을 강조하며, 이를 해결하기 위해 새로운 벤치마크인 MLLMU-Bench를 도입합니다.

- **Technical Details**: MLLMU-Bench는 500개의 가상 프로필과 153개의 공인 프로필로 구성되어 있으며, 각 프로필에는 14개 이상의 맞춤형 질문-답변 쌍이 포함되어 있습니다. 이 벤치마크는 다중 모달(이미지+텍스트) 및 단일 모달(텍스트) 관점에서 평가됩니다. 또한, 효과성(efficacy), 일반화 가능성(generality), 모델 유용성(utility)을 기준으로 네 개의 세트로 나누어 언러닝 알고리즘을 평가합니다.

- **Performance Highlights**: 실험 결과, 단일 모달 언러닝 알고리즘이 생성(generation) 및 클로즈(cloze) 작업에서 우수한 성능을 보인 반면, 다중 모달 언러닝 도구는 다중 모달 입력을 가진 분류(classification) 작업에서 더 뛰어난 성능을 나타냈습니다.



### DAGE: DAG Query Answering via Relational Combinator with Logical Constraints (https://arxiv.org/abs/2410.22105)
- **What's New**: 이 논문에서는 기존의 쿼리 임베딩 방법의 한계를 극복하기 위해 DAG 쿼리를 도입하고, 이를 위해 DAGE라는 새로운 쿼리 임베딩 방법을 제안합니다.

- **Technical Details**: DAGE는 DAG 쿼리의 계산 그래프를 기반으로 하여 두 노드 간의 가능한 여러 경로를 합쳐 하나의 경로로 표현합니다. 이를 통해 쿼리 임베딩을 효과적으로 학습할 수 있으며, DAG 쿼리에 대한 새로운 벤치마크도 제안됩니다.

- **Performance Highlights**: DAGE는 기존 tree-form 쿼리에서 사용되는 기본 쿼리 임베딩 방법보다 성능을 개선하였으며, DAG 쿼리 벤치마크에서의 성능을 실험적으로 측정하였습니다.



### Hyperspectral Imaging-Based Perception in Autonomous Driving Scenarios: Benchmarking Baseline Semantic Segmentation Models (https://arxiv.org/abs/2410.22101)
Comments:
          Accepted at IEEE WHISPERS 2024

- **What's New**: 최근 Hyperpectral Imaging (HSI) 기술이 전통적인 RGB 이미징에 비해 원거리 감지, 농업, 의료 분야에서의 장점으로 주목 받고 있습니다. 특히, Advanced Driving Assistance Systems (ADAS) 인식 향상을 위한 연구가 이루어졌으며, 다양한 HSI 데이터셋이 공개되었습니다.

- **Technical Details**: 본 연구에서는 다양한 주석(annotation)이 포함된 HSI 데이터셋을 기반으로 DeepLab v3+, HRNet, PSPNet, U-Net 및 두 가지 변형인 Coordinate Attention (UNet-CA)와 Convolutional Block-Attention Module (UNet-CBAM)와 같은 네 가지 딥러닝 기반의 Semantic Segmentation Models (SSM)에 대해 평가합니다. 데이터셋의 다양한 공간적(spatial) 및 스펙트럴(spectral) 차원을 처리하도록 모델 아키텍처를 조정하였으며, 클래스 가중 손실 함수(class-weighted loss function)를 사용하여 학습하였습니다.

- **Performance Highlights**: UNet-CBAM 모델이 채널별 특징(channel-wise features)을 추출하여 다른 SSM들보다 뛰어난 성능을 보였으며, 향상된 시맨틱 세그멘테이션을 위한 스펙트럴 정보를 활용할 가능성이 있는 것으로 나타났습니다. 그러나 현재 HSI 데이터셋의 제한된 크기, 높은 클래스 불균형, 미세 주석 부족 등은 ADAS 응용을 위한 안정적인 SSM 개발에 주요한 제약으로 남아 있습니다.



### TractShapeNet: Efficient Multi-Shape Learning with 3D Tractography Point Clouds (https://arxiv.org/abs/2410.22099)
Comments:
          10 pages, 2 figures, 4 tables. This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구에서는 뇌의 백질 연결의 기하학적 형태 측정값을 깊이 있는 학습 모델을 통해 계산할 수 있는 가능성을 탐구합니다. 새로운 프레임워크인 TractShapeNet을 소개하며, 이는 트랙토그래피의 포인트 클라우드 표현을 활용하여 다섯 가지 형태 측정값(길이, 범위, 부피, 총 표면적, 불규칙성)을 계산합니다.

- **Technical Details**: TractShapeNet은 1065명의 건강한 젊은 성인으로 구성된 대규모 데이터셋을 사용하여 평가되며, 저자들은 딥러닝 모델이 전통적인 DSI-Studio 도구에 비해 빠르고 효율적으로 형태 측정값을 계산할 수 있음을 보여줍니다. 이 모델은 마치 포인트 클라우드를 처리하는 것처럼 각 섬유 클러스터를 포인트 클라우드 형태로 나타내며, Siamese 네트워크 아키텍처를 기반으로 하여 구조적 정보를 보존하면서 최적화합니다.

- **Performance Highlights**: 실험 결과, TractShapeNet은 Pearson 상관 계수 및 정규화된 오류 메트릭에서 기존의 포인트 클라우드 기반 신경망 모델을 초월하는 성능을 보였습니다. 또한, TractShapeNet에서 계산된 형태 측정값이 DSI-Studio 방법과 유사한 성과를 내면서 언어 인지 예측 작업에서도 잘 작동함을 입증했습니다.



### Enhance Hyperbolic Representation Learning via Second-order Pooling (https://arxiv.org/abs/2410.22026)
- **What's New**: 이 논문에서는 hyperbolic representation learning에서 계층적 정보의 캡처를 개선하기 위해 second-order pooling을 도입합니다. 이는 입력 특성의 일반화 능력을 해치지 않으면서 샘플 간의 거리를 자연스럽게 증가시키는데 중점을 두었습니다.

- **Technical Details**: 본 연구는 hyperbolic space에서의 feature projection을 통해 전통적인 backbone이 계층 구조 정보를 효과적으로 캡처하도록 하는 방법을 제안합니다. 특히, low-dimensional bilinear pooling 기법이 hyperbolic representation learning에 효과적으로 적용되지 않는 이유와 이를 해결하기 위한 kernel approximation regularization을 도입하여 저차원 bilinear features가 저차원 공간에서 kernel function를 잘 근사화할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 그래프 구조 데이터셋에서 효과적임을 보여줍니다. 특히, 제안된 second-order pooling이 backbone의 Lipschitz 상수를 낮추어 일반화 능력을 향상시킴을 입증하였습니다.



### Modeling Temporal Positive and Negative Excitation for Sequential Recommendation (https://arxiv.org/abs/2410.22013)
- **What's New**: 이 논문에서는 Static-Dynamic Interest Learning (SDIL) 프레임워크를 제안하여 사용자의 정적(static) 및 동적(dynamic) 관심을 모두 모델링함으로써 순차 추천(sequential recommendation) 성능을 향상시키고자 한다.

- **Technical Details**: 제안된 SDIL 프레임워크는 (1) Static Interest Modeling (SIM) 모듈, (2) Dynamic Interest Modeling (DIM) 모듈, 그리고 (3) 다음 아이템 예측(next-item prediction) 모듈로 구성된다. 또한 Temporal Positive and Negative Excitation Modeling (TPNE) 모듈이 DIM 모듈과 결합되어 positive 및 negative excitation을 모두 모델링하는 데 중점을 둔다.

- **Performance Highlights**: 세 가지 실제 데이터셋에 대한 광범위한 실험 결과, SDIL은 기존의 최첨단 방법들보다 효과적으로 정적 및 동적 관심을 포착하고 추천 성능이 뛰어난 것으로 나타났다.



### From Explicit Rules to Implicit Reasoning in an Interpretable Violence Monitoring System (https://arxiv.org/abs/2410.21991)
Comments:
          12 pages,7 figures

- **What's New**: 이 논문은 violence surveillance (폭력 감시) 작업을 위한 새로운 패러다임인 Rule base Violence monitoring (RuleVM)를 제안합니다. 기존의 블랙박스 시스템 대신, 명시적 지식이 포함된 해석 가능한 모델을 설계하는 방안을 모색합니다.

- **Technical Details**: RuleVM은 이미지와 텍스트에 대해 서로 다른 디자인을 가진 dual-branch structure(이중 가지 구조)를 사용합니다. implicit branch(암묵적 가지)는 시각적 특징만을 활용하여 coarse-grained binary classification(조잡한 이진 분류)을 수행하고, explicit branch(명시적 가지)는 언어-이미지 정렬을 통해 fine-grained classification(세밀한 분류)을 실행합니다.

- **Performance Highlights**: RuleCLIP은 다양한 벤치마크에서 실험을 통해 조잡한 분류와 세밀한 분류 모두에서 가장 뛰어난 성능을 보여주며, 기존의 최첨단 방법들을 크게 능가했습니다. 또한, 해석 가능성 실험에서 사람의 수가 증가할수록 폭력 행동의 위험 수준이 상승한다는 흥미로운 규칙이 발견되었습니다.



### Automated Vulnerability Detection Using Deep Learning Techniqu (https://arxiv.org/abs/2410.21968)
Comments:
          4 pages, 1 figures; Presented at The 30st International Conference on Computational & Experimental Engineering and Sciences (ICCES2024)

- **What's New**: 이번 연구는 CodeBERT 모델을 활용하여 Python 애플리케이션의 SQL Injection 취약점을 탐지하는 코드 보안 테스트의 효율성을 높이는 방법을 탐구합니다.

- **Technical Details**: 소스 코드를 벡터 표현으로 변환하고 Long Short-Term Memory (LSTM) 모델을 훈련시켜 취약한 패턴을 식별합니다. 감지 기법에는 깊은 학습(deep learning) 기술과 CodeBERT의 고급 컨텍스트 이해가 포함됩니다.

- **Performance Highlights**: 기존의 정적 애플리케이션 보안 테스트(Static Application Security Testing, SAST) 도구와 비교했을 때, 우리 모델은 더 높은 정밀도(precision), 재현율(recall), F1 점수를 달성하며 우수한 성능을 보여줍니다.



### Dual Conditional Diffusion Models for Sequential Recommendation (https://arxiv.org/abs/2410.21967)
- **What's New**: 최근 확산 모델(Diffusion Models) 기술의 발전이 연속 추천(Sequential Recommendation) 분야에서 주목할 만한 성과를 보여주고 있습니다. 본 연구에서는 기존 방식의 두 가지 주요 한계를 극복하기 위해 이산-연속 추천 프레임워크를 제안합니다. 이 방법은 마르코프 체인(Markov Chain)을 도입하여 이산 대상 아이템 지수로의 전환을 모델링하여 추천 프로세스의 일관성을 보장합니다.

- **Technical Details**: 제안된 이중 조건부 확산 모델(Dual Conditional Diffusion Models, DCRec)은 이산 아이템과 연속 확산 모델의 간극을 다리며, 이중 조건부 확산 변환기(Dual Conditional Diffusion Transformer, DCDT)를 통해 사용자 행동을 반영하는 방식으로 개발되었습니다. 이 모델은 임계 조건(implicit condition)과 명시적 조건(explicit condition)을 통합하여 추천의 정확도를 높이고, 샘플링 단계를 줄여 연산 효율성을 증가시킵니다.

- **Performance Highlights**: 공개된 벤치마크 데이터셋에서 실시한 실험 결과, DCRec은 최신 SR 모델 대비 우수한 성능을 나타냈습니다. 특히 소수의 샘플링 단계로 최적의 결과를 달성하여 실시간 애플리케이션에 적합한 다양한 응용 가능성을 확인하였습니다.



### Fast and High-Quality Auto-Regressive Speech Synthesis via Speculative Decoding (https://arxiv.org/abs/2410.21951)
Comments:
          5 pages, 3 figures, 3 tables. Submitted to ICASSP 2025

- **What's New**: 이 연구에서는 VADUSA라는 새로운 접근 방식을 제시하며, 이는 추측적인 디코딩(speculative decoding)을 통해 자동 회귀 텍스트 음성 변환(TTS) 시스템의 속도를 높이는 최초의 접근 중 하나입니다.

- **Technical Details**: VADUSA는 MEDUSA라는 효율적인 추측 디코딩 방법을 기반으로 하여, 자동 회귀 모델인 VALL-E와 통합하여 홀수 대칭 토큰 변환을 실현합니다. VADUSA는 다양한 응답 토큰의 조합을 처리할 수 있는 메커니즘을 도입하여 디코딩 과정을 가속화하고 품질을 높입니다. 또한, 샘플링 동안 여유 메커니즘을 포함시켜 디코딩 품질을 유지하면서 추론 속도를 더욱 향상시킵니다.

- **Performance Highlights**: VADUSA는 inferencing 속도를 크게 향상시킬 뿐만 아니라, 다양한 대규모 데이터셋과 여러 형태의 음성 토큰을 활용하여 일반화 성능을 강화합니다.



### Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications (https://arxiv.org/abs/2410.21943)
- **What's New**: 이 논문은 산업 도메인에서 Retrieval Augmented Generation (RAG) 시스템에 멀티모달 모델을 통합하는 실험을 다루고 있으며, 이미지와 텍스트를 함께 사용하여 RAG 성능을 향상시키는 최적 구성을 찾는 데 중점을 두고 있습니다.

- **Technical Details**: 연구에서는 두 가지 이미지 처리 접근 방식과 답변 합성을 위해 두 가지 대형 언어 모델 (LLMs), 즉 GPT-4-Vision과 LLaVA를 사용합니다. 이미지 처리 전략은 멀티모달 임베딩 (multimodal embeddings)과 이미지에서 생성된 텍스트 요약을 포함합니다. 또한, LLM-as-a-Judge 방법론을 사용하여 실험 결과를 평가합니다.

- **Performance Highlights**: 멀티모달 RAG 시스템은 단일 모달 RAG 설정보다 우수한 성능을 보이며, 이미지 검색은 텍스트 검색보다 더 많은 도전 과제가 있음을 보여주었습니다. 이미지에서 생성된 텍스트 요약은 멀티모달 임베딩보다 더 나은 잠재력을 제공하여 미래 발전의 기회를 제시합니다.



### Human-Readable Programs as Actors of Reinforcement Learning Agents Using Critic-Moderated Evolution (https://arxiv.org/abs/2410.21940)
Comments:
          Accepted in BNAIC/BeNeLearn 2024 conference proceedings

- **What's New**: 이 논문은 Deep Reinforcement Learning (DRL)의 투명성과 설명 가능성의 부족 문제를 해결하기 위한 새로운 접근법을 제안합니다. 기존의 Programmatic Reinforcement Learning (PRL) 방법이 정책을 코드 형태로 증류하는데 집중하는 반면, 저자들은 RL 에이전트의 정책으로 프로그램을 직접 학습하는 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 TD3(Twin Delayed Deep Deterministic Policy Gradient) 알고리즘을 기반으로 하며, 유전 알고리즘(genetic algorithm)에 의한 프로그램 합성을 위한 목표 함수로 TD3의 critic을 사용합니다. 이 접근법은 학습 단계에서 프로그램을 생성함으로써 평균 제곱 오차(Mean Squared Error) 대신 실제 높은 보상(high reward)을 추구하게 됩니다.

- **Performance Highlights**: 실험은 제안한 방법이 간단한 Gridworld 환경에서 우수한 샘플 효율성(sample efficiency), 정책 품질(policy quality), 그리고 설명 가능성(explainability)을 보임을 입증하였습니다.



### Benchmarking OpenAI o1 in Cyber Security (https://arxiv.org/abs/2410.21939)
- **What's New**: OpenAI의 o1-preview 및 o1-mini 모델이 기존의 GPT-4o 모델과 비교하여 실제 소프트웨어에서의 취약점 감지를 평가하는 데 탁월한 성능을 보임을 입증하였다.

- **Technical Details**: 이 연구는 DARPA의 AI Cyber Challenge (AIxCC)와 Nginx challenge 프로젝트를 활용하여 자동화된 취약점 탐지(AVD) 및 자동 프로그램 수리(APR) 태스크를 위한 테스트 환경을 설계하였다. 평가 방법론에서는 reflexion loop를 사용하여 각 모델이 입력을 생성하고 실패한 시도를 분석한 후 자가 피드백을 통해 출력 개선을 도모하도록 하였다.

- **Performance Highlights**: o1-preview 모델은 GPT-4o 모델에 비해 성공률과 효율성이 뛰어났으며, 특히 복잡한 시나리오에서 그 성능을 크게 발휘하였다.



### ReMix: Training Generalized Person Re-identification on a Mixture of Data (https://arxiv.org/abs/2410.21938)
Comments:
          Accepted by WACV 2025

- **What's New**: 본 연구는 제한된 라벨이 있는 다중 카메라 데이터와 대규모 비라벨 단일 카메라 데이터를 혼합하여 공동 학습하는 새로운 방법인 ReMix를 제안합니다. 이 방법은 이전의 self-supervised pre-training 접근법의 한계를 극복하고, 더 나은 generalization 능력을 달성하도록 설계되었습니다.

- **Technical Details**: ReMix는 두 가지 유형의 데이터: 라벨이 있는 다중 카메라 데이터와 비라벨 단일 카메라 이미지를 효율적으로 혼합하고 공동 학습하기 위해 새로운 데이터 샘플 전략과 세 가지 새로운 손실 함수 (Instance Loss, Augmentation Loss, Centroids Loss)를 제안합니다. 이러한 손실 함수들은 두 종류의 데이터에 맞게 조정되어 훈련의 질을 높입니다.

- **Performance Highlights**: 실험 결과, ReMix는 cross-dataset 및 multi-source cross-dataset 시나리오에서 기존의 최첨단 방법보다 성능이 우수하며, generalizable person Re-ID에서 높은 generalization 능력을 보여줍니다.



### LogSHIELD: A Graph-based Real-time Anomaly Detection Framework using Frequency Analysis (https://arxiv.org/abs/2410.21936)
- **What's New**: 이번 연구에서 제안하는 LogSHIELD는 효율적인 탐지 모델로, 주로 깊은 학습(Deep Learning)을 활용하여 사이버 공격(attack)을 탐지하는 새로운 기법입니다. 이는 그래프 기반의 이상(anomaly) 탐지 모델을 사용하며, 고주파 분석을 통해 실시간으로 위협을 탐지할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: LogSHIELD는 두 가지 접근 방식을 제안합니다: 첫째, Graph Neural Network (GNN)을 사용한 접근 방식인 LogGNN과 둘째, 주파수 도메인 분석(Frequency Domain Analysis, FDA)을 사용하는 접근 방식입니다. 이 두 방법 모두 통계적 클러스터링 알고리즘을 사용하여 이상 탐지를 수행하며, LogSHIELD는 774M의 정상 로그와 375K의 악성 로그를 포함한 대규모 데이터셋에서 평가되었습니다.

- **Performance Highlights**: LogSHIELD는 평균 98% AUC 및 F1 점수를 기록하며, 평균 탐지 대기시간이 0.13초에 불과하고 기존 모델에 비해 우수한 성능을 나타냅니다. 이 모델은 높은 처리량(throughput)을 자랑하며, 실시간 탐지 애플리케이션에 적합합니다.



### Differentiable Inductive Logic Programming for Fraud Detection (https://arxiv.org/abs/2410.21928)
- **What's New**: 본 논문에서는 설명 가능한 AI(Explainable AI) 접근법인 미분 유도 논리 프로그래밍(Differentiable Inductive Logic Programming, DILP)을 사기 탐지(Fraud Detection) 분야에 적용하는 연구를 수행하였습니다. DILP의 스케일 문제를 해결하기 위한 데이터 정제(data curation) 기법을 제안하며, 기존 전통적 방법들과 비교하여 DILP의 성능을 평가합니다.

- **Technical Details**: DILP는 인덕티브 논리 프로그래밍(Inductive Logic Programming, ILP)의 변형으로, 데이터셋에서 규칙을 도출하는 머신러닝 방법입니다. 연구에서는 합성 데이터셋을 생성하여 DILP 성능을 평가하고, 이후 PaySim 데이터셋에서 훈련 및 평가를 진행하였습니다. DILP는 의사결정 나무(Decision Trees, DTs) 및 심볼릭 분류(Deep Symbolic Classification)와 같은 기존 방법들과 비교하여 성능을 분석합니다.

- **Performance Highlights**: DILP는 기존의 전통적인 방법들에 비해 규칙 생성에 강점을 가지며, 작은 데이터셋에서도 일반화할 수 있는 능력을 가지고 있습니다. 하지만, 데이터 전처리(data preprocessing) 작업이 필요한 점은 DILP의 적용에 있어 도전 요소로 작용합니다. 연구 결과, DILP는 전통적인 방법들과 유사한 성능을 보였으며, 특히 규칙 수를 늘리고 관계를 탐지하는 데 있어 유용하다는 점에서 가능성을 보여주었습니다.



### Reliable Semantic Understanding for Real World Zero-shot Object Goal Navigation (https://arxiv.org/abs/2410.21926)
Comments:
          16 pages, 7 figures, 2 tables

- **What's New**: 본 연구는 제로샷 객체 목표 내비게이션(Zero-Shot Object Goal Navigation, ZS-OGN)에서 의미적 이해를 강화하는 혁신적인 접근 방식을 제시하고 있습니다. 기존의 라벨이 달린 데이터에 의존하는 방식의 한계를 극복하여, GLIP 비전 언어 모델과 InstructionBLIP 모델의 이중 구성 프레임워크를 사용해 초기 탐지 및 검증 프로세스를 결합하였습니다.

- **Technical Details**: 제안된 시스템은 GLIP Vision Language Model (VLM)을 사용하여 초기 객체 및 맥락 탐지를 수행하고, InstructionBLIP을 통해 GLIP의 탐지 결과를 검증합니다. 이 접근 방식은 복합적인 환경에서의 객체 인식 및 주변 상황 이해를 강화하여, 인간과 같은 탐색 및 내비게이션 결정을 내릴 수 있게 합니다.

- **Performance Highlights**: 시뮬레이션과 실제 상황 모두에서 실시된 테스트에서 내비게이션의 정확성과 신뢰성이 눈에 띄게 향상되었습니다. 특히, 제안된 'Doubly Right' 프레임워크는 제로샷 객체 목표 내비게이션에서 객체 탐지의 신뢰성과 정확성을 개선하는데 효과적임을 보여주었습니다.



### Semi-Supervised Self-Learning Enhanced Music Emotion Recognition (https://arxiv.org/abs/2410.21897)
- **What's New**: 본 논문에서는 음악 감정 인식(Music Emotion Recognition, MER) 분야에서 세그먼트 기반(segment-based) 방법을 제안하여 감정 관련 태스크의 데이터 부족 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 반감독(Self-supervised) 학습 프레임워크를 기반으로 하며, 클립(clip) 수준의 레이블을 각 세그먼트에 직접 계승하지 않고, 노이즈(label noise)가 있는 레이블을 다루기 위해 세미-슈퍼바이즈드(semi-supervised) 학습을 사용합니다. 주요 기술로는 Mixup 데이터 증강(data augmentation) 기법과 일관성 정규화(consistency regularization)가 포함됩니다.

- **Performance Highlights**: 세 가지 공개 감정 데이터셋에서 실험한 결과, 제안된 방법이 기존 모델보다 더 나은 성능을 나타내거나 비슷한 수준의 성능을 달성하였습니다.



### Bayesian Optimization for Hyperparameters Tuning in Neural Networks (https://arxiv.org/abs/2410.21886)
Comments:
          Bachelor Thesis in Optimization for Machine Learning, 57 pages

- **What's New**: 이번 연구에서는 이미지 분류 작업을 위한 Convolutional Neural Networks (CNN)의 하이퍼파라미터 튜닝에 Bayesian Optimization (BO)을 적용하고 그 효율성을 조사합니다.

- **Technical Details**: Bayesian Optimization은 연속 입력과 제한된 평가 예산이 있는 복잡한 블랙박스 함수에 적합한 도함수 비사용 글로벌 최적화 방법입니다. BO 알고리즘은 Gaussian Process 회귀 및 Upper Confidence Bound (UCB), Expected Improvement (EI)과 같은 수집 함수를 활용하여 최적 구성을 효과적으로 식별합니다.

- **Performance Highlights**: Ax와 BOTorch 프레임워크를 사용한 실험 결과, BO는 하이퍼파라미터 튜닝 시도 횟수를 줄이면서도 경쟁력 있는 모델 성능을 달성하는 효율성을 보여주었습니다. BO는 탐색과 착취의 균형을 효과적으로 맞추어 CNN 아키텍처에 대한 최적 설정으로 신속하게 수렴합니다.



### Advancing Efficient Brain Tumor Multi-Class Classification -- New Insights from the Vision Mamba Model in Transfer Learning (https://arxiv.org/abs/2410.21872)
- **What's New**: 이번 연구는 뇌 종양 분류를 위한 사전 훈련된 모델의 적용을 조사하고, 특히 Mamba 모델을 활용하여 진행되었습니다. 특히, 새로운 네트워크 아키텍처인 Vision Mamba(Vim)를 소개하고 이를 통한 최초의 뇌 종양 분류 적용을 보여주었습니다.

- **Technical Details**: 연구진은 여러 주류 전이 학습(transfer learning) 모델을 미세 조정(fine-tuning)하고 이를 다중 클래스 분류(multi-class classification) 문제에 적용하였습니다. Vim 모델은 독립 테스트 세트에서 100% 분류 정확성을 달성하였으며, 기존의 최신 모델들과 비교했을 때 경량( lightweight) 및 효율적(efficient)으로 나타났습니다.

- **Performance Highlights**: Vim 모델은 의료 이미징(medical imaging) 분야에서 레이블이 붙은 데이터가 제한적인 상황에서도 전이 학습의 우수성을 입증하며, Tumor classification를 위한 새로운 아이디어를 제공합니다. 이 연구의 전이 학습 기반 프레임워크는 다른 의료 이미징 분류 문제에도 널리 적용될 수 있습니다.



### Cross-Entropy Is All You Need To Invert the Data Generating Process (https://arxiv.org/abs/2410.21869)
- **What's New**: 이번 연구는 지도 학습(supervised learning)의 효과성을 설명하는 포괄적인 이론을 제시하기 위해, 비선형 독립 성분 분석(nonlinear Independent Component Analysis) 기법을 활용하여 잠재 구조(latent structures)를 회복(establish)하는 방법을 다룹니다.

- **Technical Details**: 연구진은 매개변수 인스턴스 차별화(parametric instance discrimination)를 통해 식별 가능성(identifiability) 결과를 확장하고, 교차 엔트로피(minimization)로 최적화를 하는 지도 학습 환경에서 어떻게 통찰(insights)을 적용하는지를 보여줍니다. 이들은 표준 분류 작업(standard classification tasks)에서도 모델들이 진실(factors of variation)의 표현(representations)을 선형 변환(linear transformation)까지 배우는 것을 증명하였습니다.

- **Performance Highlights**: 시뮬레이션 데이터(시뮬레이트된 데이터) 및 DisLib라는 대표적인 디스럽란틀 벤치마크(disentanglement benchmark)에서 이론적 가정을 충족하는 결과를 보이며, 단순한 분류 작업이 잠재 구조를 회복할 수 있음을 입증했습니다. 또한, ImageNet에서 훈련된 모델이 변형(factors of variation)의 선형 디코딩을 허용하는 표현을 인코딩(encoding)함을 밝혔습니다.



### Learning Infinitesimal Generators of Continuous Symmetries from Data (https://arxiv.org/abs/2410.21853)
Comments:
          Neurips 2024

- **What's New**: 이 논문은 데이터에서 연속적인 대칭을 학습하기 위한 새로운 알고리즘을 제안합니다. 기존의 방법들이 사전 정의된 Lie 군에 의존하는 반면, 본 연구에서는 비선형 생성기를 포함하여 다양한 대칭을 파라미터화하고 학습할 수 있도록 합니다.

- **Technical Details**: 연속 대칭을 학습하기 위해, 본 논문에서는 Neural Ordinary Differential Equation (Neural ODE)를 모델링하여 학습 가능한 infinitesimal generator를 설정합니다. 이 방법은 주어진 작업에 따라 정의된 기준에 대한 불변성을 위반하는 정도를 측정하기 위해 validity score 함수를 설계합니다. 이 함수는 풀리 미분 가능하며, end-to-end로 gradient descent를 통해 대칭을 학습할 수 있도록 합니다.

- **Performance Highlights**: CIFAR-10 분류 데이터셋과 편미분 방정식(PDE)에서 실험을 통해, 제안한 방법이 약한 대칭과 비선형 대칭을 효과적으로 발견하는 것을 보여주었습니다. 또한, 학습된 생성기를 이용하여 자동 증강 기법을 개발함으로써, 전통적인 변환과 비교하여 경쟁력 있는 성과를 나타냄을 입증했습니다.



### Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning (https://arxiv.org/abs/2410.21845)
- **What's New**: 본 연구에서는 인간-루프(human-in-the-loop) 비전 기반 강화 학습 시스템을 제안하며, 다양한 정교한 조작 작업에서 뛰어난 성능을 보여줍니다. 이 시스템은 연속적인 동적 조작, 정밀 조립 및 이중 팔 협동 작업을 포함한 작업에서 높은 성공률을 달성하며, 1시간에서 2.5시간의 훈련 시간 동안 훈련합니다.

- **Technical Details**: 강화 학습 시스템 이름은 HIL-SERL(Human-in-the-Loop Sample-Efficient Robotic Reinforcement Learning)로, 미리 훈련된 비전 백본(visual backbone)을 사용해 정책 학습 시 최적화 안정성 문제를 해결합니다. 인간의 시연(demonstrations)과 수정(corrections)을 통합한 샘플 효율적인 오프-정책(off-policy) 강화 학습 알고리즘을 활용하며, 안전성을 보장하기 위해 저수준(low-level) 제어기가 포함됩니다.

- **Performance Highlights**: 본 시스템은 동일한 양의 인간 데이터를 훈련에 사용하여 평균 101%의 성공률 향상 및 1.8배 빨라진 사이클 타임(cycle time)을 달성하였습니다. RL 시스템은 다양한 복잡한 비전 기반 조작 정책을 직접 학습할 수 있음을 보여주며, 이로 인해 산업 분야 및 연구 발전에 큰 기여를 할 것으로 기대됩니다.



### Diffusion as Reasoning: Enhancing Object Goal Navigation with LLM-Biased Diffusion Mod (https://arxiv.org/abs/2410.21842)
- **What's New**: 본 논문에서는 Object Goal Navigation (ObjectNav) 작업을 해결하기 위해 새로운 접근 방식인 Diffusion as Reasoning (DAR) 기법을 제안합니다. 이 방법은 에이전트의 이동 중 누적된 메모리를 기반으로 하는 통계적 분포 패턴을 학습하는 diffusion model을 훈련하여 보이지 않는 환경에서의 목표 객체 위치 추정을 가능하게 합니다.

- **Technical Details**: DAR는 Denoising Diffusion Probabilistic Models (DDPM)을 활용하여 에이전트의 환경 기억을 조건으로 삼아 알려지지 않은 지역의 신뢰할 수 있는 의미론적 콘텐츠를 생성합니다. DDPM은 RGB 데이터 대신 클래스 채널의 의미론적 맵으로 학습시키며, LLM의 일반 지식을 활용하여 생성화의 일반화를 향상시킵니다.

- **Performance Highlights**: Gibson과 MP3D 환경에서 실험 결과, DAR 기법이 기존의 일반적인 방법보다 높은 성공률을 나타내어, 목표 객체 위치 추정 및 효율적인 내비게이션을 가능하게 함을 보여주었습니다.



### Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Models (https://arxiv.org/abs/2410.21815)
- **What's New**: 본 논문에서는 self-interpretable 모델과 black-box 모델을 위한 post-hoc 설명 사이의 간극을 메우기 위한 새로운 방법인 *AutoGnothi*를 제안합니다. 이 방법은 블랙박스 모델이 예측 정확성을 손상시키지 않으면서 자가 해석 가능성을 이론적으로 보장하는 것을 목표로 합니다.

- **Technical Details**: *AutoGnothi*는 parameter-efficient transfer learning (PETL)을 활용하여 블랙박스 모델에 작은 사이드 네트워크를 통합합니다. 이를 통해 Shapley value 기반의 설명을 생성할 수 있으며, 원래 네트워크의 매개변수를 변경하지 않고도 메모리, 훈련 및 추론 비용을 크게 줄일 수 있습니다. 이 방법은 두 개의 모델로 구성된 기존 방식과 달리 예측과 설명을 동시에 수행하며, 단일 추론 단계에서 작업을 완료합니다.

- **Performance Highlights**: 실험 결과 *AutoGnothi*는 매우 높은 훈련 및 추론 효율성을 보여 주었습니다. 예를 들어, ViT-base 모델에 대해 97%의 훈련 가능 매개변수 감소, 72%의 훈련 메모리 감소를 기록하며, 설명 생성에 있어서 54%의 추론 비용 감소와 44%의 추론 시간 감소를 달성했습니다. 이러한 결과는 *AutoGnothi*가 블랙박스 모델의 예측 성능을 유지하면서 자기 해석 가능성을 강화한다는 것을 일증합니다.



### A Fresh Look at Generalized Category Discovery through Non-negative Matrix Factorization (https://arxiv.org/abs/2410.21807)
Comments:
          13 pages, 8 figures,Submitted to IEEE Transactions on Image Processing

- **What's New**: 논문은 비부정적인 일반화 범주 발견(Non-Negative Generalized Category Discovery, NN-GCD) 프레임워크를 제안하여 기존의 GCD 접근법의 한계를 극복하고, 코사인 유사성에 기반한 공존 행렬의 최적화를 다룰 수 있는 새로운 방법론을 제공합니다. 이 프레임워크는 대칭 비부정적 행렬 분해(Symmetric Non-negative Matrix Factorization, SNMF)를 수학적 매개체로 사용하여 최적의 K-평균과 SNMF 간의 동등성을 증명합니다.

- **Technical Details**: NN-GCD는 비부정적 활성화 뉴런 메커니즘과 NMF NCE 손실을 도입하며, 이를 통해 GCD 모델이 글로벌 최적 지역으로 수렴할 수 있도록 돕습니다. 이론적으로 K-평균 클러스터링을 SNMF와 비부정적 대조 학습(Non-negative Contrastive Learning, NCL) 최적화와 동등한 문제로 재구성합니다. 또한 혼합 희소 정규화 방법을 제안하여 A의 희소성 제약을 부과합니다.

- **Performance Highlights**: NN-GCD는 일반화 범주 발견(GCD) 벤치마크에서 최첨단 방법을 초과하는 성능을 보여주며, Semantic Shift Benchmark에서 평균 66.1%의 정확도를 기록하여 이전 방법들보다 4.7% 향상된 결과를 나타냈습니다.



### Text-Guided Attention is All You Need for Zero-Shot Robustness in Vision-Language Models (https://arxiv.org/abs/2410.21802)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 Text-Guided Attention for Zero-Shot Robustness (TGA-ZSR)라는 새로운 전략을 제안합니다. 이 방법은 CLIP 모델의 일반화 능력을 유지하면서 적대적 공격에 대한 강인성을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: TGA-ZSR는 두 가지 주요 구성 요소로 이루어져 있습니다: Attention Refinement 모듈과 Attention-based Model Constraint 모듈입니다. Attention Refinement 모듈은 원래의 깨끗한 예제에서 얻은 text-guided attention과 적대적 예제에서 얻은 text-guided attention을 정렬시켜 모델의 강인성을 높이는 역할을 합니다. Attention-based Model Constraint 모듈은 깨끗한 예제를 이용하여 두 모델의 text-guided attention을 획득하고, 이는 오염된 데이터에서 모델의 성능을 유지하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, TGA-ZSR 방법은 현재의 최신 기술들에 비해 9.58%의 향상된 zero-shot 강인성을 보여주며, 16개 데이터셋에서 검증되었습니다.



### Online Mirror Descent for Tchebycheff Scalarization in Multi-Objective Optimization (https://arxiv.org/abs/2410.21764)
Comments:
          27 pages, 7 figures, 2 tables

- **What's New**: 이번 논문은 Tchebycheff scalarization을 기반으로 하는 새로운 온라인 미러 하강 알고리즘인 OMD-TCH를 제안합니다. 이 알고리즘은 여러 목표를 최적화하려는 다목표 최적화(Multi-Objective Optimization, MOO)의 단점을 보완합니다. 특히, 기존의 linear scalarization 방식의 한계를 극복하고는 OMD-TCH의 수렴 속도를 보여줍니다.

- **Technical Details**: OMD-TCH는 확률적 최적화 문제에 대한 효율적인 솔루션을 제공합니다. 이 알고리즘은 Tchebycheff scalarization을 활용하여 최악의 목표를 최적화합니다. OMD-TCH는 $O(\sqrt{\log m / T})$의 수렴 속도를 가지고 있으며, 여기서 m은 목표의 수, T는 반복 횟수를 의미합니다. 또한, 새로운 적응형 온라인-배치 변환 방식인 AdaOMD-TCH를 제안하여 성능을 개선합니다.

- **Performance Highlights**: OMD-TCH와 AdaOMD-TCH는 합성 문제와 페더레이티드 러닝(Federated Learning) 작업에서 공정성 제약을 만족시키면서 최첨단의 성능을 보여주었습니다. AdaOMD-TCH는 OMD-TCH에 비해 정확도와 공정성 지표 모두에서 현저한 성능 향상을 이루었으며, 기존의 공정 페더레이티드 러닝 방법들과 유사한 성능을 달성했습니다.



### Learning and Unlearning of Fabricated Knowledge in Language Models (https://arxiv.org/abs/2410.21750)
- **What's New**: 이 논문에서는 대규모 언어 모델(LM)에 새로운 지식이 주입될 때 발생하는 현상과 이러한 지식이 얼마나 오래 지속되는지를 연구합니다. 이를 위해, 다양한 사실 유형을 테스트할 수 있도록 설계된 새로운 probing 데이터셋인 'Outlandish'를 사용합니다.

- **Technical Details**: 연구 결과, 일반 지식과 상충하는 사실(Knowledge-Conflicting Facts, KCFs)은 수만 번의 훈련 단계 동안 강력하게 기억되며, 일반적인 혹은 무작위로 섞인 사실에 비해 더 오래 지속된다는 것을 보여주었습니다. KCFs는 언어 모델이 논리적으로 관련이 없는 프롬프트에서 환각을 유도할 때 더 많이 작용합니다. 이러한 사실들은 새로운 다단계 희소 업데이트(multistep sparse updates)를 적용하여 지워질 수 있음이 드러났습니다.

- **Performance Highlights**: 이 연구는 데이터 포이즈닝(data poisoning)의 영향 완화 방법에 대한 중요한 시사를 제공합니다. KCFs는 상당히 오랜 기억 지속성을 가지지만, 적절한 업데이트 희소화 기법을 통해 이러한 독성이 제거될 수 있습니다.



### Enhancing Financial Question Answering with a Multi-Agent Reflection Framework (https://arxiv.org/abs/2410.21741)
Comments:
          Accepted by ICAIF 24

- **What's New**: 이 연구는 재무 질문응답(QA) 작업을 위한 다중 에이전트 프레임워크를 제안하며, 각 질문에 대해 사고 과정을 반영하는 비평가 에이전트를 포함하여 다중 비평가 에이전트를 추가로 통합하여 성능을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 LLM(대형 언어 모델) 코어 기반의 전문가 에이전트와 비평가 에이전트를 구현하며, 체인 오브 사고(Chain-of-Thought) 프로세스를 통해 데이터를 추출하고 수학적 추론을 수행합니다. 에이전트는 두 개의 하위 작업으로 분리되고, 각 비평가의 피드백이 첫 번째 에이전트의 초기 응답 재평가에 전달됩니다.

- **Performance Highlights**: 이 연구 결과는 LLaMA3-8B 모델에서 평균 15%, LLaMA3-70B 모델에서 5%의 성능 향상을 보여주며, 단일 에이전트 프레임워크와 비교해 통계적으로 유의미한 개선을 나타냅니다. 또한 제안된 프레임워크는 GPT-4o-mini 및 Claude-3.5 Sonnet과 같은 대형 모델과 유사한 성능을 보이면서 비용 효율적인 솔루션을 제공합니다.



### Efficient Reprogramming of Memristive Crossbars for DNNs: Weight Sorting and Bit Stucking (https://arxiv.org/abs/2410.21730)
Comments:
          5 pages, 10 figures

- **What's New**: 이 논문에서는 메모리스터(membristor)의 재프로그래밍 횟수를 줄이기 위한 새로운 접근 방식을 소개합니다. 이 방법은 DNN(deep neural networks)에서의 메모리스터의 비휘발성 메모리 내구성 한계를 해결하는 데 초점을 맞추고 있습니다. 주목할 만한 두 가지 기술을 사용하여 재프로그래밍 요구 사항을 감소시킵니다.

- **Technical Details**: (1) 정렬된 가중치 섹션(sorted weight sectioning)을 사용하여 비슷한 크로스바(crossbar)의 재프로그래밍을 스케줄링하고 메모리스터 상태 재사용을 극대화합니다. (2) 낮은 차수 열의 랜덤하게 선택된 메모리스터의 일부만 재프로그래밍하여 비트 수준 분포를 활용하고 모델 정확도에 미치는 작은 영향을 인식합니다.

- **Performance Highlights**: ImageNet-1K 데이터셋에서 ResNet-50의 경우 3.7배, ViT-Base의 경우 21배의 크로스바 재프로그래밍 횟수 감소를 보이며 모델 정확도는 1% 이하의 범위 내에서 유지됩니다.



### On the Statistical Complexity of Estimating VENDI Scores from Empirical Data (https://arxiv.org/abs/2410.21719)
- **What's New**: 이 논문은 기계 학습 커뮤니티에서 최근 연구된 참조가 없는 생성 모델 평가 지표인 VENDI score에 대해 다루고 있습니다. VENDI score는 정보 이론의 행렬 기반 엔트로피를 사용하여 생성 모델의 다양성을 정량화합니다. 이 논문에서는 유한한 샘플 크기에서 VENDI score의 통계적 수렴성을 탐구하고, t-truncated VENDI 통계량이라는 대체 통계량을 도입합니다.

- **Technical Details**: VENDI score는 n×n 커널 행렬의 고유분해를 통해 계산되며, 여기서 n은 생성된 샘플의 수입니다. 그러나 n이 클 경우, 고유분해의 계산 비용이 급격히 증가하여 수천 개로 제한됩니다. 이 논문에서는 VENDI score의 계산에서 고유값 중 상위 t개만을 사용하는 t-truncated VENDI 통계량을 제안하고, 이를 통해 더 적은 샘플 크기로 통계량을 계산할 수 있음을 보입니다. 나아가 Nyström 방법과 FKEA 근사 방법이 정의된 절단 VENDI 통계량으로 수렴한다는 것을 증명합니다.

- **Performance Highlights**: 본 논문은 실험을 통해 제한된 n 크기의 샘플로 계산된 VENDI score와 우리의 정의된 t-truncated VENDI 통계량 간의 관계를 검증하며, 특히 유한 차원 커널 함수의 경우 VENDI score가 VENDI 통계량으로 효율적으로 수렴함을 확인했습니다. 무한 차원 Gaussian 커널 함수의 경우, 샘플 수가 10,000을 초과한 경우에도 점점 증가하는 VENDI score의 특성을 관찰하였습니다.



### Generating Realistic Tabular Data with Large Language Models (https://arxiv.org/abs/2410.21717)
Comments:
          To appear at ICDM 2024

- **What's New**: 이 논문에서는 tabular data (표 형식 데이터) 생성을 위한 새로운 LLM 기반의 방법을 제안합니다. 기존의 방법들이 특정 feature와 target variable 간의 올바른 상관관계를 포착하지 못하는 문제를 해결하기 위해 세 가지 중요한 개선 방안을 도입하였습니다.

- **Technical Details**: 제안된 방법은 fine-tuning(미세 조정) 단계에서의 새로운 permutation strategy, feature-conditional sampling(특징 조건 샘플링) 접근 방식, 그리고 생성된 샘플을 바탕으로 한 라벨 생성 방식을 포함합니다. 이러한 개선은 LLM이 실제 데이터에서의 feature-class correlation(특징-클래스 상관관계)을 올바르게 포착할 수 있도록 도와줍니다.

- **Performance Highlights**: 제안된 Pred-LLM 방법은 20개의 데이터셋에서 10개의 기존 SOTA (State-Of-The-Art) 기법을 초월하며, 품질과 다양성 면에서 매우 현실적인 합성 샘플을 생성합니다. 더 나아가, 합성 데이터로 훈련된 분류기는 원본 데이터로 훈련된 분류기와 비교해도 경쟁력을 가지는 성과를 보여, tabular data generation(표 데이터 생성)에서 중요한 진전을 이룹니다.



### A Bayesian Approach to Harnessing the Power of LLMs in Authorship Attribution (https://arxiv.org/abs/2410.21716)
- **What's New**: 이 연구는 사전 훈련된 대형 언어 모델(Large Language Models, LLMs)을 활용하여 단회 저자 귀속(one-shot authorship attribution) 문제를 해결하는 방안을 제시하며, 베이esian 접근법과 LLM의 확률 출력을 사용하여 저자 추정을 정교하게 수행합니다.

- **Technical Details**: 저자는 LLM으로부터 얻은 로그 확률을 활용하여 특정 저자가 작성했을 확률을 추정하는 Bayesian 접근 방식을 사용합니다. 'Logprob method'라는 새로운 방법론을 제안하며, 이 방식은 사전 훈련의 목표와 분류 작업을 일치시켜, 복잡한 수작업 특성 엔지니어링 없이 언어의 미세한 뉘앙스를 자동으로 포착합니다.

- **Performance Highlights**: 이 방법론은 IMDb와 블로그 데이터셋에서 10명의 저자에 대해 85%의 정확도를 달성하며, LLMs를 이용한 단회 저자 분석의 새로운 기준을 설정하고 법언어학(forensic linguistics) 분야에서의 적용 가능성을 확장합니다.



### AdaptGCD: Multi-Expert Adapter Tuning for Generalized Category Discovery (https://arxiv.org/abs/2410.21705)
- **What's New**: 이번 연구에서는 Generalized Category Discovery (GCD) 작업에 adapter tuning을 최초로 도입하여, 기존 pretrained 모델의 일반 지식을 보존하면서 새로운 카테고리 발견에 대한 적응성을 향상시키는 AdaptGCD라는 새로운 방법을 제안합니다.

- **Technical Details**: AdaptGCD는 ViT(Visual Transformer) 블록의 피드포워드 레이어에 평행하게 학습 가능한 adapter들을 통합하고, 기존 백본의 파라미터를 동결하여 pretrained 지식을 보존합니다. 또한, old class와 new class 간의 감독 정보 불균형을 문제로 삼아, 다수의 adapter 전문가를 포함하는 Multi-Expert Adapter 구조를 도입하고, 데이터의 클래스를 구분하기 위해 route assignment 제약을 설계하였습니다.

- **Performance Highlights**: 7개의 널리 사용되는 데이터셋에서 수행된 광범위한 실험을 통해, adapter tuning이 GCD 성능을 크게 향상시키는 것으로 나타났습니다. 또한, route assignment 제약이 적용된 multi-expert adapter 구조의 효과도 확인되었습니다.



### How Does Critical Batch Size Scale in Pre-training? (https://arxiv.org/abs/2410.21676)
- **What's New**: 이 논문은 대규모 모델을 훈련할 때의 병렬 처리 전략(Parallelism strategies)의 효율성을 분석하며, critical batch size(CBS)라는 새로운 지표를 제안합니다.

- **Technical Details**: 저자들은 CBS를 측정하기 위해 8500만에서 12억 개의 매개변수를 가지는 일련의 자동 회귀 모델(autogressive models)을 C4 데이터셋에서 사전 훈련(pre-train)했습니다. 배치 크기(batch size), 모멘텀(momentum), 학습률(learning rate) 및 스케줄링(scheduling)과 같은 하이퍼파라미터(hyper-parameter)를 조정하여 스케일(scale)이 CBS에 미치는 영향을 체계적으로 조사했습니다.

- **Performance Highlights**: 결과는 CBS가 모델 크기보다 데이터 크기에 주로 비례하여 증가한다는 것을 보여줍니다. 이 발견은 신경망(Neural networks)의 무한 너비 한계(infinite-width limits)와 무한 차원 최소 제곱 회귀(infinite-dimensional least squares regression)에 대한 이론적 분석을 통해 정당화됩니다.



### BF-Meta: Secure Blockchain-enhanced Privacy-preserving Federated Learning for Metavers (https://arxiv.org/abs/2410.21675)
- **What's New**: 메타버스(metaverse)는 사회적 및 경제적 활동을 위한 혁신적인 플랫폼으로 떠오르고 있으며, 사용자 프라이버시를 보호하면서 지능형 서비스를 제공하기 위해 연합 학습(federated learning, FL)을 활용하여 로컬 웨어러블 장치에서 모델을 학습하는 BF-Meta라는 안전한 블록체인 기반 FL 프레임워크를 제안합니다.

- **Technical Details**: BF-Meta는 중앙 집중식 모델 집계를 탈피하여, 악의적인 사용자로부터의 부정적인 영향을 완화하고 메타버스에서 안전한 가상 서비스를 제공합니다. 이 프레임워크는 하드웨어, 데이터, 네트워크, 합의(Consensus), 인센티브(Incentive), 그리고 애플리케이션 레이어로 구성되어 있으며, 악의적인 모델을 집계 전에 감지하고 사용자 행동에 기반한 피드백을 제공하는 인센티브 메커니즘을 설계하였습니다.

- **Performance Highlights**: 다섯 개의 데이터셋을 사용한 실험에서 BF-Meta의 효과성과 적용 가능성을 입증하였으며, 사용자 행동을 모니터링하고 비정상적인 행동을 감지하여 FL 훈련 동안 악의적 사용자를 제외하는 시스템의 안정성을 보여주었습니다.



### Knowledge-Guided Prompt Learning for Request Quality Assurance in Public Code Review (https://arxiv.org/abs/2410.21673)
Comments:
          28 pages, 7 images, 12 tables, Manuscript submitted to a journal (2024)

- **What's New**: 이 논문에서는 개발자 중심의 코드 리뷰 요청 품질 보증을 달성하기 위해 Knowledge-guided Prompt learning for Public Code Review (KP-PCR)라는 새로운 방법을 제안합니다.

- **Technical Details**: KP-PCR은 1) text prompt tuning을 사용하여 Masked Language Model (MLM)으로 변환하여 리뷰 요청의 필요성과 태그 추천 두 가지 subtasks를 재구성하고, 2) Knowledge와 code prefix tuning을 통해 외부 지식을 도입하며, 데이터 흐름 다이어그램을 사용하여 코드 스니펫을 특성화합니다.

- **Performance Highlights**: 2011-2023년 기간 동안 PCR 데이터셋에 대한 실험 결과, KP-PCR은 요청 필요성 예측에서 8.3%~28.8%, 태그 추천에서 0.1%~29.5%의 향상된 성능을 보여주었습니다.



### Sequential choice in ordered bundles (https://arxiv.org/abs/2410.21670)
- **What's New**: 본 논문에서는 경험 재화(Experience Goods) 구성을 위해 고안된 여러 예측 모델을 사용하여, 순서가 정해진 번들에서 개인이 다음 항목을 소비할 가능성을 예측하는 방법을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 커스텀 Transformers(디코더 전용 및 인코더-디코더 아키텍처), 미세 조정된 GPT-3, 커스텀 LSTM 모델, 강화 학습 모델, 두 개의 Markov 모델 그리고 제로 순서 모델을 평가합니다. 주목할 점은 Transformers의 주의(attention) 가중치를 분석하여 소비자가 이전 선택을 기반으로 다음 항목을 고를 때의 패턴을 보여줍니다.

- **Performance Highlights**: Spotify 데이터 분석 결과, 디코더 전용 아키텍처를 가진 커스텀 Transformer가 개별 선택과 총 수요 예측에서 가장 높은 정확도를 달성합니다. 또한, 이 모델은 개인 맞춤형 프로모션을 통해 수요를 증가시키는 데 도움을 줄 수 있습니다.



### PACE: Physics Informed Uncertainty Aware Climate Emulator (https://arxiv.org/abs/2410.21657)
- **What's New**: 본 논문에서는 PACE 라는 저용량의 물리 기반 기후 에뮬레이터를 제안합니다. PACE는 684K의 파라미터를 가진 모델로, 온도와 강수량을 86년 동안 안정적으로 모사할 수 있습니다.

- **Technical Details**: PACE는 온실가스 배출 데이터를 기반으로 훈련되어, 대류-확산 (advection-diffusion) 의 기본 물리 법칙을 통합하여 경계 조건을 고려하고 확산 계수 및 흐름 속도를 추정합니다. 이것은 기후 변수에 대한 정확한 예측을 가능하게 합니다. 또한 PACE는 15개의 기후 모델에서 벤치마크를 초과하여 우수한 성능을 보여주며, Neural ODE (Ordinary Differential Equation) 구조를 기반으로 하여 기후 시스템 내에서의 물리적 과정을 모델링합니다.

- **Performance Highlights**: PACE는 ClimateSet에서 제공하는 15가지 기후 모델을 통해 일반화된 높은 성능을 보여줍니다. 모델은 저비용으로 GHG 배출 데이터만으로 86년 동안의 기후 변수를 안정적으로 에뮬레이트 할 수 있습니다.



### RDSinger: Reference-based Diffusion Network for Singing Voice Synthesis (https://arxiv.org/abs/2410.21641)
- **What's New**: RDSinger는 고품질의 오디오를 생성하기 위한 참조 기반의 denoising diffusion 네트워크로, 전통적인 Singing Voice Synthesis(SVS) 방법의 한계를 극복하려고 합니다. 이 모델은 FastSpeech2 mel-spectrogram을 참조로 사용하여 denoising 과정에서의 아티팩트를 줄이고, pitch 전환 동안 문제를 해결하기 위해 부분적인 mel-spectrogram에 Gaussian blur를 적용합니다.

- **Technical Details**: RDSinger는 노이즈를 mel-spectrogram으로 변환하는 diffusion 기반의 acoustic 모델로, 음악 점수와 참조 mel-spectrogram에 조건을 두어 작동합니다. 기존의 L1 또는 L2 손실 함수 대신, Gaussian blur를 적용하고 손실 가중치를 조정하여 멜-스펙트로그램의 중요한 영역에 집중하여 품질을 높입니다.

- **Performance Highlights**: OpenCpop 데이터셋에 대한 평가에서, RDSinger는 현재의 최첨단 SVS 방법들과 비교하여 더 나은 성능을 보였으며, 자연스럽고 고감도의 보컬 사운드를 생성하는 데 성공하였습니다.



### A Tutorial on Clinical Speech AI Development: From Data Collection to Model Validation (https://arxiv.org/abs/2410.21640)
Comments:
          76 pages, 24 figures

- **What's New**: 이 논문은 임상Speech AI의 강력한 개발을 위한 필수 요소에 대한 포괄적인 안내를 제공합니다. 특히 음성을 통해 건강 상태를 평가하는 데 필요한 맞춤형 태스크 디자인 및 데이터 수집 방안에 대해 다루고 있습니다.

- **Technical Details**: 정확한 음성 데이터를 수집하고 임상적 구성 요소를 측정하기 위한 음성 표현의 개발 및 검증, 신뢰할 수 있는 예측 모델 구축, 그리고 윤리적 고려사항을 포함하여 임상Speech AI 모델 개발의 다양한 기술적 요소를 설명합니다.

- **Performance Highlights**: 이 연구는 감독 학습( supervised learning)의 사용이 실제 임상 환경에서의 일반화 문제를 초래할 수 있음을 강조하며, 기존 모델들이 환자 데이터와의 상상적 연관성을 통해 임상적 예측 결과를 정확히 반영할 수 없다는 점을 지적합니다.



### MCPDial: A Minecraft Persona-driven Dialogue Datas (https://arxiv.org/abs/2410.21627)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)을 활용하여 플레이어와 비플레이어 캐릭터(NPC) 간의 페르소나(펄소나) 기반 대화를 생성하는 혁신적인 접근법을 제안합니다. 이를 통해 Minecraft와 같은 인기 게임에서 사용할 수 있는 MCPDial 프레젠스를 소개합니다. MCPDial은 기본 대화를 넘어 깊이 있는 상호작용을 가능하게 합니다.

- **Technical Details**: MCPDial 데이터셋은 250개의 NPC 페르소나 설명과 관련된 대화로 구성되어 있으며, LLM을 활용하여 대화 예시를 생성합니다. 이 데이터셋은 NPC와 플레이어 간의 페르소나 일관성을 유지하며 게임 특정 기능 호출을 통합하여 상호작용성을 증대시킵니다. 데이터 생성 과정에는 gold-standard 페르소나 및 대화 수집, 그리고 LLM을 이용한 데이터 증강이 포함됩니다.

- **Performance Highlights**: MCPDial 데이터셋은 유창성, 페르소나 일관성 및 기능 호출 정확성 측면에서 높은 품질의 대화를 포함하고 있으며, Minecraft 도메인 내에서 대화형 에이전트를 교육하고 평가하는 데 있어 매우 유용한 자원으로 작용합니다.



### Identifying Selections for Unsupervised Subtask Discovery (https://arxiv.org/abs/2410.21616)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문은 고수익 태스크를 해결하기 위해 태스크를 서브태스크로 분해하는 방법에 대해 논의합니다. 특히, 서브태스크가 데이터 생성 과정에서의 선택 메커니즘으로부터 도출된다는 점을 강조하며, 이를 통해 강화 학습(RL) 및 모방 학습(IL) 문제 해결에 있어 데이터 효율성을 향상시킬 수 있다는 점에 주목합니다.

- **Technical Details**: 논문에서는 서브태스크를 파악하기 위한 새로운 이론을 제안합니다. 제안된 방법은 순차적 비음수 행렬 분해(sequential non-negative matrix factorization, seq-NMF)를 통해 서브목표를 학습하고, 이를 바탕으로 의미 있는 행동 패턴을 추출합니다. 이 과정에서 선택 변수를 서브목표로 활용하여 서브태스크를 발견합니다.

- **Performance Highlights**: 다양한 Kitchen 환경에서의 실험 결과, 학습된 서브태스크가 새로운 태스크에 대한 일반화 성능을 효과적으로 향상시킴을 보여주었습니다. 이 연구는 서브태스크 발견 문제에 대한 심층적인 이해를 제공하며, 데이터 효율성을 높이는 기회를 마련합니다.



### Reducing the Scope of Language Models with Circuit Breakers (https://arxiv.org/abs/2410.21597)
- **What's New**: 이 논문은 언어 모델들이 특정한 용도로 특정 쿼리에만 답변하게 하려는 새로운 방법, 즉 'scoping'을 제안합니다. 기존 시스템 프롬프트를 이용한 방법이 불완전하다는 것을 발견하고, 'Circuit Breakers' (CB)라는 새로운 메소드를 제시합니다.

- **Technical Details**: CB 방법은 언어 모델의 출력을 특정 작업(예: 감정 분석, 요약하기)에 맞춰 scoping하는 데 사용됩니다. 또한, Supervised Fine Tuning (SFT)과 결합하여 정확한 쿼리에서의 성능을 향상시키고, 비관련 쿼리는 거부할 수 있는 구조를 제공합니다.

- **Performance Highlights**: CB 방법은 기존의 미세 조정(fine-tuning) 방식이나 선호 학습(preference learning)에 비해 아웃 오브 디스트리뷰션(out of distribution) 작업에 대한 강건성이 뛰어나며, 다수의 허용 작업을 지원하여 보다 정교한 범위를 제공합니다. SFT와 CB를 결합할 때 쿼리 성능 향상과 비관련 쿼리 거부에서 모두 긍정적인 결과를 보여줍니다.



### ImageNet-RIB Benchmark: Large Pre-Training Datasets Don't Guarantee Robustness after Fine-Tuning (https://arxiv.org/abs/2410.21582)
- **What's New**: 이 논문에서는 새로운 강건한 미세 조정 벤치마크인 ImageNet-RIB(Robustness Inheritance Benchmark)를 도입하여, 대규모 사전 훈련 모델이 특수 작업에 대해 어떻게 미세 조정(fine-tuning) 되는지를 평가합니다.

- **Technical Details**: 이 벤치마크는 관련 있지만 구별되는 특수 작업들로 구성되어 있으며, 사전 훈련된 모델은 집합의 하나의 작업에 대해 미세 조정되고 나머지 작업에서의 강건성(robustness)을 평가하게 됩니다. EWC, LwF와 같은 지속적 학습(continual learning) 방법은 미세 조정 이후에도 강건성을 유지하는 반면, 일반적으로 미세 조정은 관련 하류 작업에 대한 일반화(generalization) 성능을 감소시킵니다.

- **Performance Highlights**: 사전 훈련된 데이터셋의 다양성과 풍부함에 따라 성능 저하의 정도를 예측할 수 있으며, 특수 작업에 대한 성능을 위해 가장 강력한 모델로 시작하는 것이 항상 최선의 접근 방식은 아닐 수 있다는 것을 보여줍니다.



### A Generative Model Based Honeypot for Industrial OPC UA Communication (https://arxiv.org/abs/2410.21574)
Comments:
          This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is accepted and will be published in Computer Aided Systems Theory - EUROCAST 2024

- **What's New**: 본 논문은 산업 OPC UA 통신을 모방한 생성 모델 기반의 honeypot을 소개합니다. 이는 사전 녹화된 상태 공간 궤적(state space trajectories)을 학습하여 동적인 메카트로닉 시스템의 특성을 반영합니다.

- **Technical Details**: Long Short-Term Memory (LSTM) 네트워크를 활용하여, 생성 모델 기반의 honeypot이 주기적인 산업 프로세스를 OPC UA 통신을 통해 복제할 수 있음을 증명합니다. 본 연구는 기존의 Brownfield 시스템을 위해 현실적인 honeypot을 생성하는 도전 과제를 해결합니다.

- **Performance Highlights**: 효율적인 하드웨어에서 낮은 계산 자원을 요구하며, 단기적으로 안정적이고 그럴 듯한 궤적 생성이 가능하나, 장기적인 시간에서는 편차가 발생합니다. 향후 작업은 모델 정확도와 상호작용 능력 향상 및 데이터셋 확장을 목표로 하고 있습니다.



### Thank You, Stingray: Multilingual Large Language Models Can Not (Yet) Disambiguate Cross-Lingual Word Sens (https://arxiv.org/abs/2410.21573)
- **What's New**: 이번 연구에서는 여러 언어의 감정을 평가하기 위해 새로운 벤치마크인 StingrayBench를 도입했습니다. 특히, false friends(의미가 다른 유사한 단어들)를 활용하여 다국어 대형 언어 모델(LLMs)의 한계를 분석합니다.

- **Technical Details**: StingrayBench는 인도네시아-말레이(Indonesian-Malay), 인도네시아-타갈로그(Indonesian-Tagalog), 중국어-일본어(Chinese-Japanese), 영어-독일어(English-German) 등 네 가지 언어 쌍에 대한 cross-lingual sense disambiguation(언어 간 의미 구분) 평가를 위해 설계되었습니다. 새로운 metrics(측정 기준)인 cognate bias(유사어 편향)와 cognate comprehension score(유사어 이해 점수)를 도입하여 LLM의 성능을 평가합니다.

- **Performance Highlights**: 분석 결과, LLM들이 자원(RResources)이 풍부한 언어에 편향되어 있다는 것을 발견했습니다. 이러한 연구는 더 다양한 다국어 커뮤니티를 위한 공정한 접근성을 촉진하는데 기여합니다.



### Mitigating Gradient Overlap in Deep Residual Networks with Gradient Normalization for Improved Non-Convex Optimization (https://arxiv.org/abs/2410.21564)
- **What's New**: 최근 딥러닝(DL) 분야에서 Residual Networks(ResNets)의 효과적인 사용이 찾아졌습니다. 그러나, skip connection으로 인해 발생하는 gradient overlap 문제를 해결하기 위해 Z-score Normalization(ZNorm) 기법이 제안되었습니다.

- **Technical Details**: ZNorm은 gradient scale을 조정하여 layers 간의 gradient를 표준화함으로써 gradient overlap의 부정적인 영향을 줄입니다. 이를 통해 비선형 최적화(non-convex optimization) 환경에서도 훈련 과정이 개선됩니다.

- **Performance Highlights**: 실험 결과, ZNorm을 적용한 경우 최적 솔루션을 찾기 어려운 비선형 환경에서의 학습 성능이 향상되었습니다. 이러한 결과는 대규모 데이터 처리 시 정확도가 중요한 영역에서의 성능 개선을 시사합니다.



### Going Beyond H&E and Oncology: How Do Histopathology Foundation Models Perform for Multi-stain IHC and Immunology? (https://arxiv.org/abs/2410.21560)
Comments:
          Accepted at Workshop on Advancements In Medical Foundation Models (NeurIPS 2024)

- **What's New**: 이 연구는 최신 histopathology foundation 모델이 다중 염색(autoimmune) IHC 데이터셋에 대해 가지는 일반화 능력을 평가합니다. 13개의 feature extractor 모델을 비교하여 류마티스 관절염 하위 유형 분석과 쇼그렌 질병 감별 작업에 대한 성과를 분석하였습니다.

- **Technical Details**: H&E 염색된 암 이미지에서 autoimmunity IHC 이미지로 학습된 representation의 전이 가능성을 평가하기 위해 Attention-Based Multiple Instance Learning (ABMIL) 분류기를 사용했습니다. 연구는 autoimmunity 특유의 염색 패턴을 정확히 인식하는 데 어려움이 있었음을 나타냅니다.

- **Performance Highlights**: 연구 결과, histopathology pretrained 모델이 ImageNet pretrained 모델에 비해 유의미한 성과 차이를 보여주지 않았으며, autoimmunity 특성의 오해석과 편향된 feature 중요성이 발견되었습니다.



### Bayesian Regression for Predicting Subscription to Bank Term Deposits in Direct Marketing Campaigns (https://arxiv.org/abs/2410.21539)
- **What's New**: 이번 연구는 포르투갈 은행의 직접 마케팅 데이터를 사용하여 term deposit 구독을 예측하는 logit 모델과 probit 모델의 효과를 조사하였습니다. 특히 Bayesian 접근법을 통해 고객 행동 예측의 정밀도를 향상시키려는 노력이 주목받고 있습니다.

- **Technical Details**: 데이터셋은 41,188개의 샘플과 21개의 변수를 포함하며, 고객의 term deposit 구독 행동을 예측하기 위해 Bayesian Logistic Regression과 Bayesian Probit Regression 모델이 비교되었습니다. 이를 위해 Leave-One-Out Cross-Validation (LOO-CV)과 Bayesian 기법을 활용하여 모델의 예측 능력을 평가하였습니다.

- **Performance Highlights**: 연구 결과 logit 모델이 classification 문제를 처리하는 데 있어서 probit 모델보다 더 우수한 성능을 보였습니다. 이러한 결과는 복잡한 의사 결정 과정을 처리할 때 모델 선택의 중요성을 강조하며, 은행이 고객 세분화 및 마케팅 캠페인을 최적화할 수 있는 방안을 제시합니다.



### L3Ms -- Lagrange Large Language Models (https://arxiv.org/abs/2410.21533)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 감독하에 세밀한 조정(Supervised Fine-Tuning, SFT)과 정렬(Alignment)을 제약 최적화 문제로 공식화하였습니다. 이는 애플리케이션의 특정 요구 사항을 충족하도록 하여 휴리스틱(Heuristic) 방법에 의한 최적화를 피할 수 있게 합니다.

- **Technical Details**: 제안된 Lagrange Large Language Models (L3Ms)는 로그 장벽(Logarithmic Barriers)을 사용하여 제약을 시행합니다. 이를 통해 L3Ms는 다양한 애플리케이션에 맞춰 커스터마이징 가능하며, 기존의 단계적 접근 방식에서 벗어나 SFT와 정렬 단계를 통합합니다.

- **Performance Highlights**: 실험을 통해 L3Ms가 다양한 애플리케이션에서 사용자 지정 요구 사항을 충족하는 데 어떻게 유용한지를 보여줍니다. 예를 들어, 두 가지 L3Ms를 사용하여 동일한 지시를 따르는 작업을 수행하되, 하나는 간결하도록 제약하고 다른 하나는 장황하도록 제약하는 결과를 비교하였습니다.



### A Multi-Agent Reinforcement Learning Testbed for Cognitive Radio Applications (https://arxiv.org/abs/2410.21521)
Comments:
          Accepted to IEEE CCNC 2025

- **What's New**: 이번 논문에서는 멀티 에이전트 강화 학습(MARL) 기능이 추가된 RFRL Gym의 업데이트된 환경을 소개합니다. 이전에는 단일 RL 에이전트만 훈련할 수 있었던 RFRL Gym이 이제는 여러 에이전트를 동시에 훈련할 수 있게 개선되었습니다.

- **Technical Details**: RFRL Gym은 OpenAI Gym 프레임워크를 기반으로 구축되어 있으며, RF 스펙트럼 내에서 시뮬레이션 시나리오를 커스터마이징할 수 있는 기능을 제공합니다. 이번 업데이트를 통해 멀티 에이전트 강화 학습(MARL)을 지원하여 여러 인공지능 에이전트가 협력적, 경쟁적 또는 혼합 환경에서 훈련될 수 있게 되었습니다.

- **Performance Highlights**: 이 논문에서는 다양한 RF 시나리오에서의 MARL 환경 테스트 결과와 함께 향후 추가될 기능도 소개합니다. RFRL Gym은 다양한 협력적 및 경쟁적인 상황을 지원할 수 있는 유연성을 제공하며, 이는 향후 IoT 디바이스의 증가와 RF 스펙트럼 활용 증가에 따른 필수 요구 사항을 충족시킬 것입니다.



### Sabotage Evaluations for Frontier Models (https://arxiv.org/abs/2410.21514)
- **What's New**: 이번 연구는 AI 모델의 사보타주(sabotage) 능력과 이에 따른 잠재적 위험에 대한 새로운 평가 프레임워크를 개발했습니다. 특히, 모델이 자율적으로 다른 시스템의 측정 및 의사 결정을 방해할 수 있는 능력에 중점을 두고 있습니다.

- **Technical Details**: 연구팀은 모델의 사보타주 능력을 3가지 임계값(threshold)으로 구분했습니다: 배포 전 행동 숨기기, 감독 undermining, 조직적 행동에 대한 사보타지. 각 임계값에 대한 검증 평가를 설계하고, Anthropic의 Claude 3 Opus 및 Claude 3.5 Sonnet 모델을 통해 실험했습니다.

- **Performance Highlights**: Claude 3 Opus와 Claude 3.5 Sonnet 모델은 기본적인 감독 하에서는 사보타주 임계값을 충족하지 않았으나, 모든 작업에서 비트리비얼(trivial) 성능을 보였습니다. 이는 현재 minimal mitigations이 사보타주 위험을 다루기에 충분하지만, 앞으로는 더 강력한 mitigations이 필요할 것임을 나타냅니다.



### Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups (https://arxiv.org/abs/2410.21508)
- **What's New**: 본 논문에서는 Sparse Autoencoders (SAEs)의 훈련 전략을 혁신적으로 개선하여 동일한 층 그룹에 대해 하나의 SAE만 훈련하도록 하는 방법을 제안합니다. 이로 인해 기존의 층별 훈련에서 연속적인 층 그룹 훈련으로 전환하게 됩니다.

- **Technical Details**: SAEs는 대규모 언어 모델(LLM)의 활성화를 해석 가능한 특징의 희소 선형 조합으로 재구성하는 비지도 학습 방법입니다. 다층의 LLM에서 각각의 층마다 하나의 SAE를 훈련하는 전통적인 접근법을 대신하여, 연속된 층들에 대해 하나의 SAE를 훈련함으로써 비약적인 계산 효율성을 제공합니다. 실험 결과, Pythia 160M 모델에서 훈련 속도가 6배 향상됨을 보여주었습니다.

- **Performance Highlights**: 모델의 재구성 품질이나 하위 작업의 성능을 저하시키지 않고, SAE 훈련의 효율성을 크게 향상시켰습니다. 이로 인해 현대 LLM에서의 SAE 훈련이 보다 효율적으로 진행될 수 있게 되었습니다.



### Towards Multi-dimensional Explanation Alignment for Medical Classification (https://arxiv.org/abs/2410.21494)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 새로운 프레임워크 Med-MICN(의료 다차원 해석 가능한 개념 네트워크)을 제안하여 의료 이미지 분석의 해석 가능성 문제를 해결합니다.

- **Technical Details**: Med-MICN은 신경 기호 추론(neural symbolic reasoning), 개념 의미론(concept semantics), 및 주목도 맵(saliency maps)을 통한 다각적 해석 가능성을 제공합니다. 이 프레임워크는 대규모 다중 모달(LMM)을 활용하여 개념 세트를 생성하고 의료 이미지에 대한 자동 주석(auto-annotation)을 수행합니다.

- **Performance Highlights**: 네 가지 벤치마크 데이터셋에서 테스트한 결과, Med-MICN은 기존의 개념 기반 모델과 블랙박스 모델에 비해 우수한 성능과 해석 가능성을 보였습니다.



### Trustworthiness of Stochastic Gradient Descent in Distributed Learning (https://arxiv.org/abs/2410.21491)
- **What's New**: 이번 논문에서는 분산 학습(distributed learning)에서 컴프레스트 SGD(compressed SGD) 기법의 신뢰성을 평가합니다. 기존 연구들은 주로 단일 공격자(singular attackers) 상황에서 강도를 낮춘 기법에 집중했으나, 이 연구는 두 가지 머신러닝 공격인 Gradient Inversion(GradInv)과 Membership Inference Attack(MIA)에 대한 포괄적인 분석을 제공하고 있습니다.

- **Technical Details**: 실험은 PowerSGD와 Top-K SGD라는 두 가지 주요 기법을 통해 진행되며, 이 두 방법 모두 그래디언트 압축(gradient compression) 기술을 사용하여 통신 비용을 줄이는 데 초점을 맞추고 있습니다. 연구 결과에 따르면, 컴프레스트 SGD는 비압축 SGD에 비해 프라이버시 유출에 대한 저항력이 더 높습니다.

- **Performance Highlights**: 연구 결과, 세 가지 데이터셋에서 압축된 SGD 알고리즘이 그래디언트 인버전 공격에 대해 더 강력한 저항성을 보였으며, MIA는 두 SGD 방식에 대해 낮은 민감도를 나타내어 프라이버시 위험 평가에 있어 MIA의 신뢰성이 낮음을 시사합니다.



### Can Large Language Models Act as Symbolic Reasoners? (https://arxiv.org/abs/2410.21490)
Comments:
          18 pages, currently under review

- **What's New**: 본 논문은 Large Language Models (LLMs)의 기호적 추론(Symbolic Reasoning) 수행 능력에 대한 최신 연구를 탐구하고, LLM이 본질적으로 추론을 제공할 수 있는지에 대한 질문을 제기합니다. 또한 연구의 현재 격차와 미래 동향을 파악하고 있습니다.

- **Technical Details**: LLM은 주로 transformer architecture를 기반으로 하여 자연어 처리(NLP) 응용 프로그램에서 우수한 성능을 보여주고 있습니다. 이 논문은 LLM의 구조와 기호적 추론 간의 관계, 그리고 LLM의 결정 과정에 대한 대안적 접근인 prompt engineering, chain-of-thought, tree-of-thought, knowledge graph 등의 기법들에 대해 논의하고 있습니다.

- **Performance Highlights**: LLMs는 출력 생성 과정에서 기호적 추론을 효과적으로 수행하지 못하지만, 외부에서 제공되는 가이드라인이나 프롬프트를 통해 그들의 출력을 향상시킬 수 있음을 보여주었습니다. 또한 LLM의 결정에 대한 설명 가능성을 증대시키는 전략이나 계획 능력이 중요한 미래의 AI 시스템 개발을 위한 핵심 요소로 제시되었습니다.



### Enhancing CTR Prediction in Recommendation Domain with Search Query Representation (https://arxiv.org/abs/2410.21487)
Comments:
          Accepted by CIKM 2024 Full Research Track

- **What's New**: 이 논문은 사용자 검색 쿼리의 임베딩을 추천 도메인에서의 사용자 선호도와 연관 지어 학습하는 프레임워크를 제안합니다. 이러한 접근은 기존의 방법에서 간과된 사용자 의도 전이를 다루며, 검색 도메인에서의 사용자 행동 데이터를 활용하여 추천 시스템의 클릭률(CTR) 예측 정확도를 향상시킵니다.

- **Technical Details**: 제안된 프레임워크인 QueryRec는 사용자 쿼리 히스토리를 추천 도메인에 보강된 특성으로 추가하여 사용자 관심사를 보다 효과적으로 반영합니다. 이 프레임워크는 또한 자기 주의 기반의 순차 인코더를 사용하여 쿼리 리스트와 다음 클릭된 아이템의 임베딩을 정렬하며, 대조 학습 모듈을 통해 검색 도메인에서의 쿼리-아이템 관계를 캡처합니다. 확산 모델(diffusion model)을 도입하여 긍정적인 아이템의 예측을 강화하고, 노이즈 제거 방식을 통해 잘못된 긍정 예측을 방지합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 최신 모델들과 비교하여 추천 도메인에서 뛰어난 성능을 보였으며, 사용자 선호도 전이를 효과적으로 반영하여 정확한 클릭률 예측을 가능하게 하였습니다.



### AiSciVision: A Framework for Specializing Large Multimodal Models in Scientific Image Classification (https://arxiv.org/abs/2410.21480)
- **What's New**: AiSciVision은 과학 연구에서의 인공지능(AI) 사용을 위한 신뢰성과 해석 가능성을 제공하는 새로운 프레임워크입니다. 이 프레임워크는 대형 다중모달 모델(LMM)을 인터랙티브 연구 파트너로 변환하고 이미지 분류 작업을 위한 모델로 특화되었습니다.

- **Technical Details**: AiSciVision은 두 가지 주요 구성 요소로 이루어져 있습니다: (1) Visual Retrieval-Augmented Generation (VisRAG)과 (2) 도메인 특정 도구입니다. 이 시스템은 이미지 분류를 위해 유사한 이미지(양성 및 음성 레이블이 붙은)를 검색하고, LMM 에이전트가 이를 기반으로 도구를 선택하여 여러 번에 걸쳐 이미지를 조작하고 검토합니다.

- **Performance Highlights**: AiSciVision은 세 가지 실제 과학 이미지 분류 데이터셋에서 평가되었으며, 수조(aquaculture) 연못, 병든 이름풀이(eelgrass), 태양광 패널의 존재를 탐지하는 작업에서 완전 감독 모델보다 더 나은 성능을 보였습니다.



### TransformLLM: Adapting Large Language Models via LLM-Transformed Reading Comprehension Tex (https://arxiv.org/abs/2410.21479)
- **What's New**: 이 연구에서는 법률 애플리케이션에 특화된 Phi-2-Legal 및 Mistral-Legal-7B라는 언어 모델을 개발하였습니다. 이 모델들은 500백만 토큰 이상의 법률 텍스트로 계속해서 사전 학습을 진행하였으며, 법률 작업에서의 성능을 크게 향상시켰습니다.

- **Technical Details**: 이 모델들은 이전에 학습한 대형 언어 모델(LLM)을 기반으로 하며, 법률 관련 원시 데이터를 읽기 이해 자료로 변환하여 지속적 사전 학습을 수행합니다. AdaptLLM 기법을 사용하여 법률 텍스트를 읽기 이해 형식으로 변환하고, 이를 통해 발생한 데이터의 고품질화에 중점을 두었습니다. LoRA(저전력 재학습)와 같은 파라미터 효율적인 미세 조정 기법이 적용되었습니다.

- **Performance Highlights**: 새로운 법률 LLM들은 기존에 더 많은 자원으로 훈련된 모델들보다 우수한 성능을 보여주며, 법률 기준에서도 최신 상태의 성과를 달성했습니다. 이 연구는 도메인 특화 텍스트에서의 지속적 사전 학습의 효과를 증명하며, 다양한 작업에서 개선을 가져오는 가능성을 강조합니다.



### Knowledge Distillation for Real-Time Classification of Early Media in Voice Communications (https://arxiv.org/abs/2410.21478)
- **What's New**: 본 논문은 음성 통화 초기화 단계에서의 초기 미디어의 실시간 분류를 위한 산업 환경을 조사합니다. 최신 오디오 태깅 모델을 적용하며, 이러한 모델이 초기 미디어 분류에 적용될 때의 몇 가지 한계를 강조합니다.

- **Technical Details**: 기존 접근 방식은 대개 Convolutional Neural Networks(CNNs)를 활용하지만, 본 논문은 Gradient-Boosted Trees(GBT)를 기반으로 하는 저전력 소모 방식의 새로운 접근 방식을 제안합니다. 또한 지식 증류(Knowledge Distillation)와 클래스 집합(Class Aggregation) 기법을 활용하여 더 간단하고 작은 모델을 훈련시킵니다.

- **Performance Highlights**: 제안된 모델은 실행 성능이 크게 개선되었음을 입증하며, 비교 가능한 정확도를 보여줍니다. 또한 인도의 지역 데이터 센터에서의 성능 향상 사례 연구를 포함하여, 결과의 정확도와 실행 성능에 대한 상세한 분석을 제공합니다.



### Estimating Causal Effects of Text Interventions Leveraging LLMs (https://arxiv.org/abs/2410.21474)
- **What's New**: 이 논문에서는 사회 시스템에서의 언어적 개입의 효과를 정량화할 수 있는 새로운 방법론인 CausalDANN을 제안합니다. 전통적인 인과 추론 방법이 텍스트 데이터를 처리하는 데 한계가 있는 문제를 해결하는 데 초점을 맞추었습니다.

- **Technical Details**: CausalDANN은 대규모 언어 모델(LLM)을 사용하여 텍스트 변환을 통해 인과 효과를 추정하는 프레임워크입니다. 이 방법은 임의의 텍스트 개입을 처리할 수 있으며, 관찰된 데이터와 변환된 데이터의 결과를 예측하여 인과 효과를 추정합니다. 또한, 도메인 적응(neural DANN) 모델을 결과 예측에 적용하여 데이터 이동(data shift) 문제를 해결합니다.

- **Performance Highlights**: CausalDANN의 성능을 검증하기 위해 세 가지 실험을 통해 언어와 관련된 속성의 인과성을 정확하게 추정하였습니다. 첫 번째는 아마존 리뷰의 긍정적인 감정이 판매량에 미치는 영향, 두 번째는 소셜 미디어 포스트에 대한 인기 있는 반응이 도덕적 판단에 미치는 영향, 세 번째는 소셜 미디어 포스트의 분노가 도덕적 판단을 변화시키는 영향을 추정했습니다. 이 방법은 근거 기반 의사 결정을 지원하고 복잡한 사회적 현상을 깊이 이해하는 데 기여합니다.



### AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models (https://arxiv.org/abs/2410.21471)
- **What's New**: 이 연구는 Image-to-Image (I2I) diffusion 모델에서 발생하는 새로운 유형의 적대적 이미지 공격을 공개합니다. 또한 AdvI2I라는 새로운 프레임워크를 제안하여 입력 이미지를 조작하여 diffusion 모델이 NSFW (Not Safe for Work) 콘텐츠를 생성하도록 유도합니다.

- **Technical Details**: AdvI2I는 입력 이미지를 최적화하여 적대적 이미지를 생성하는 방법을 포함합니다. 이 프레임워크는 기존의 방어 메커니즘인 Safe Latent Diffusion (SLD)을 우회할 수 있는 특성을 가지고 있으며, 적대적 이미지와 NSFW 개념 임베딩 간의 유사성을 최소화하도록 설계된 AdvI2I-Adaptive라는 향상된 버전도 소개합니다.

- **Performance Highlights**: 법적 및 윤리적 요구사항에 대한 안전 장치가 현재의 공격에도 효과를 가지지 않음을 보여주며, 실험을 통해 AdvI2I 및 AdvI2I-Adaptive가 현존하는 안전 장치를 효과적으로 우회함을 입증했습니다. 이는 I2I diffusion 모델의 보안을 강화하기 위한 강력한 대책 마련의 필요성을 강조합니다.



### TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors (https://arxiv.org/abs/2410.21443)
- **What's New**: 논문에서 소개된 Truck Adversarial Camouflage Optimization (TACO)은 3D 차량 모델에 대한 적대적인 위장 패턴을 생성하여 최신 모델인 YOLOv8을 속이기 위한 새로운 프레임워크입니다.

- **Technical Details**: TACO는 Unreal Engine 5를 사용하여 포토리얼리스틱(Photorealistic) 렌더링과 차별화된 렌더링을 결합하여 YOLOv8을 목표로 하는 텍스처를 최적화합니다. 또한 이를 위해 Convolutional Smooth Loss 함수를 도입하여 생성된 텍스처가 감지기를 속이는 데 효과적이면서도 시각적으로 그럴듯하도록 합니다.

- **Performance Highlights**: 실험 결과, TACO는 YOLOv8의 탐지 성능을 크게 저하시키며, 테스트 데이터에서 AP@0.5의 수치가 0.0099를 기록했습니다. 이러한 적대적 패턴은 다른 객체 탐지 모델인 Faster R-CNN 및 이전 YOLO 버전에서도 강한 전이 가능성을 보였습니다.



### Deploying Ten Thousand Robots: Scalable Imitation Learning for Lifelong Multi-Agent Path Finding (https://arxiv.org/abs/2410.21415)
Comments:
          Submitted to ICRA 2025

- **What's New**: 이번 연구는 Lifelong Multi-Agent Path Finding (LMAPF) 문제 해결을 위한 새로운 접근법인 Scalable Imitation Learning for LMAPF (SILLM)를 제안합니다. SILLM은 최신 GPU의 도움을 받아 빠른 추론 속도를 유지하면서도 고품질 솔루션을 제공합니다.

- **Technical Details**: SILLM은 Spatially Sensitive Communication (SSC) 모듈을 통해 에이전트 간의 의사소통을 강화하고, 단일 단계 충돌 해결 및 글로벌 가이드를 체계적으로 통합하는 새로운 접근법입니다. 이러한 설계를 통해 최대 10,000명의 에이전트를 효율적으로 관리할 수 있습니다.

- **Performance Highlights**: SILLM은 10,000명의 에이전트를 포함한 6개의 대규모 맵에서 평균 137.7%와 16.0%의 처리량 개선을 달성하며, 2023 League of Robot Runners 대회에서 우승 솔루션을 초월하였습니다. 또한, 실제 로봇 10대 및 가상 로봇 100대를 사용한 검증에서도 효과성을 입증하였습니다.



### CT2C-QA: Multimodal Question Answering over Chinese Text, Table and Char (https://arxiv.org/abs/2410.21414)
Comments:
          10 pages, 6 figures

- **What's New**: 본 연구는 텍스트, 표, 차트를 포함한 새로운 중국어 기반의 멀티모달 질문 응답 데이터 세트인 C\text{T}^{2}C-QA를 소개합니다. 이는 현재의 한계와 다양한 모달리티를 사용하는 질문에 대한 응답을 통합적으로 분석하는 새로운 기준을 제시합니다.

- **Technical Details**: C\text{T}^{2}C-QA 데이터 세트는 200개의 웹페이지에서 수집한 9,981개의 질문-답변 쌍을 포함하고 있습니다. 새로운 multi-agent 시스템인 AED(Allocating, Expert and Decision)는 다양한 모달리티(텍스트, 표, 차트)를 처리하는 전문가 에이전트를 할당하고 상호작용하여 결정을 내리는 기능을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, 우리의 데이터 세트에서 AED의 성능은 KM = 33.9 및 CLKM = 34.3에 도달했으며, 이는 아직 개선이 필요함을 나타냅니다. 현재의 방법론(GPT-4 포함)보다 더 높은 성능 목표를 가지고 있음을 보여줍니다.



### Exploring reinforcement learning for incident response in autonomous military vehicles (https://arxiv.org/abs/2410.21407)
Comments:
          DIGILIENCE 2024

- **What's New**: 무인 차량의 자율 사이버 방어를 위한 강화 학습 활용에 관한 연구입니다. 이 연구는 기존 문헌에서 다루지 않았던 군사 작전에서의 자율 방어 시스템을 다룬 점이 특징입니다.

- **Technical Details**: 이 논문은 무인 육상 차량(UGV)에서 사이버 공격에 자율적으로 대응할 수 있는 에이전트를 강화 학습을 이용하여 훈련하는 방법을 탐구합니다. 초기에는 간단한 시뮬레이션 환경에서 프로토타입 에이전트를 개발하고, 이를 더 현실적인 환경과 실제 UGV로 테스트했습니다. 핵심 기여는 간단한 시뮬레이션 환경에서도 강화 학습이 실제 UGV에 사용할 수 있는 에이전트를 훈련하기에 적합하다는 것을 보여주는 것입니다.

- **Performance Highlights**: 강화 학습 기반 에이전트는 사이버 공격에 대한 자율적 대응을 가능하게 하며, 군사 작전에서 자율 차량의 보안 요구를 충족하는 데 필요한 방안을 제시합니다. 이 연구에서는 다양한 시뮬레이션 환경에서 에이전트를 적용하고 비교하여 유의미한 결과를 도출했습니다.



### Unveiling the Role of Expert Guidance: A Comparative Analysis of User-centered Imitation Learning and Traditional Reinforcement Learning (https://arxiv.org/abs/2410.21403)
Comments:
          Published as CEUR Workshop Proceedings in Proceedings of the 1st International Workshop on Human-in-the-Loop Applied Machine Learning (HITLAML 2023). Awarded Best Paper. this https URL

- **What's New**: 본 연구는 전통적인 강화 학습 방법과 비교하여 모방 학습(imitation learning)의 성과, 강인성(robustness), 그리고 한계에 대해 깊이 있는 분석을 제공합니다. 특히, 인간 피드백(human feedback)을 통해 지식이 어떻게 효과적으로 통합되는지를 살펴봅니다.

- **Technical Details**: 연구는 Unity 플랫폼을 사용한 시뮬레이션 환경에서 진행되었으며, 모방 학습 기술의 기존 기법과 최신 강화 학습 방법(Proximal Policy Approximation (PPO), Soft-Actor Critic (SAC))을 비교합니다. 2D Bird Hunter 게임을 통해 에이전트를 훈련시키고, 여러 복잡한 환경을 설정하여 모델 성능을 평가하였습니다.

- **Performance Highlights**: 모방 학습과 전통적인 강화 학습 간의 비교 결과, 모방 학습이 특히 복잡한 환경에서 효율적임이 입증되었습니다. 이 연구는 복잡한 실세계 문제에 효과적으로 대처할 수 있는 지능형 시스템 개발에 기여할 것입니다.



### Can Machines Think Like Humans? A Behavioral Evaluation of LLM-Agents in Dictator Games (https://arxiv.org/abs/2410.21359)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)을 기반으로 한 에이전트의 친사회적 행동을 유도하는 다양한 페르소나(persona)와 실험적 환경의 영향을 조사하였습니다. 또한, LLM의 복잡한 의사결정 시나리오에서 성과를 평가하기 위한 행동적 접근 방식을 제안하였습니다.

- **Technical Details**: 이 연구는 권위자의 게임(dicator game)을 기반으로 하여 LLM 에이전트의 이타적 행동을 평가했습니다. 다양한 LLM 가족 간과 인간 행동과 비교한 결과, LLM 사이에서 행동의 변동성과 불일치가 크게 나타났습니다. LLM에 인간과 유사한 정체성을 부여하는 것만으로는 인간과 유사한 행동을 나타내지 않았습니다.

- **Performance Highlights**: LLM 에이전트는 인간의 의사결정을 정확하게 예측하지 못하고, 이러한 일치도는 모델 아키텍처와 프롬프트 형식에 따라 달라지는 highly variable한 특성을 보였습니다. 이는 LLM의 복잡한 사회적 상호작용에서 그들의 행동을 평가하기 위한 더 깊은 이해와 견고한 방법론의 필요성을 강조합니다.



### Causal Interventions on Causal Paths: Mapping GPT-2's Reasoning From Syntax to Semantics (https://arxiv.org/abs/2410.21353)
Comments:
          12 pages

- **What's New**: 이 연구는 변형자 기반의 LLMs(large language models)가 자연어에서 인과적 사고를 어떻게 처리하는지를 초기 단계로 분석합니다. 특히, 명확한 원인과 결과 문장들을 분석함으로써 LLM의 인과적 구조를 이해하고자 합니다.

- **Technical Details**: 연구진은 GPT-2 small 모델을 사용하여 명확한 인과 문장을 확인했습니다. 이 모델은 12개의 레이어를 가지고 있으며, 자기 주의(attention) 메커니즘을 포함합니다. 연구에서는 문장의 구문적 신호와 의미적 관계를 분석하기 위해 인과 문장을 포함한 합성 데이터 셋을 사용했습니다.

- **Performance Highlights**: 결과적으로, GPT-2 모델은 주로 첫 2-3 레이어에서 구문적 구조를 파악하며, 일부 주의 헤드는 비논리적인 변형에 대한 민감도를 높게 나타냈습니다. 이러한 발견은 모델이 신택틱 큐(syntactic cues)를 탐지하고 최종 레이어의 특정 헤드가 의미적 관계를 중시함을 시사합니다.



### LLMCBench: Benchmarking Large Language Model Compression for Efficient Deploymen (https://arxiv.org/abs/2410.21352)
- **What's New**: 대형 언어 모델(LLMs)의 효율성을 높이기 위해 여러 모델 압축 기술이 제안되었으나, 이번 연구는 다양한 가정 하에 고도화된 압축 알고리즘의 효과를 평가할 수 있는 새로운 벤치마크인 LLMCBench를 발표하였다.

- **Technical Details**: LLMCBench는 LLM 압축 알고리즘의 깊이 있는 분석을 제공하며, 6개의 평가 트랙과 11개의 데이터셋, 18개의 네트워크 아키텍처, 3개의 배포 플랫폼에서 실험을 수행한다. 주요 압축 기술로는 sparsification(희소화), quantization(양자화), knowledge distillation(지식 증류) 등이 포함된다.

- **Performance Highlights**: LLMCBench는 기존 연구에서의 평가 한계를 극복하고, 모델 압축 알고리즘에 대한 통합적인 비교를 제공함으로써 경량화된 LLM의 실제 운용 가능성을 높일 수 있는 통찰력을 제공한다.



### LinFormer: A Linear-based Lightweight Transformer Architecture For Time-Aware MIMO Channel Prediction (https://arxiv.org/abs/2410.21351)
- **What's New**: 6세대(6G) 이동통신 네트워크의 출현으로 인해 고속 이동 통신 지원과 채널 노화(channel aging) 문제 해결을 위한 새로운 도전 과제가 제기되었습니다. 기존 채널 예측 방법은 더 높은 정확도를 제공하지만, 계산 복잡성이 증가하여 모바일 네트워크에서의 실용적인 적용을 제한합니다. 이러한 문제를 해결하기 위해 스케일이 가능한 LinFormer라는 혁신적인 채널 예측 프레임워크를 제안합니다.

- **Technical Details**: LinFormer는 자연어 처리(NLP) 모델인 BERT에서 영감을 받아 설계된 인코더 전용 Transformer 모델 기반의 채널 예측 프레임워크입니다. 본 접근법에서는 Transformer에서 일반적으로 사용되는 계산 집약적인 주의(attention) 메커니즘을 시간 인식 다층 퍼셉트론(TMLP)으로 대체하여 계산 요구 사항을 크게 줄입니다. TMLP 모듈은 시간 인식을 내재하고 있어 채널 예측 업무에 특히 적합합니다.

- **Performance Highlights**: LinFormer는 균형잡힌 손실 함수인 WMSELoss와 데이터 증가(data augmentation) 기술을 이용하여 통신 데이터셋을 활용한 훈련 과정을 강화했습니다. 실험 결과, LinFormer는 다양한 이동 시나리오에서 기존 방법보다 뛰어난 성능을 보여 주어, 향후 무선 통신 시스템에 대한 유망한 솔루션을 제공하고 있습니다.



### FALCON: Feedback-driven Adaptive Long/short-term memory reinforced Coding Optimization system (https://arxiv.org/abs/2410.21349)
Comments:
          20 pages, 7 figures

- **What's New**: 최근 대형 자연어 처리 모델(LLMs)이 자동화된 코드 생성에서 큰 발전을 이루었지만, 사용자 의도와의 정렬 문제를 겪고 있습니다. 본 논문에서는 피드백 기반의 적응형 장기/단기 메모리 강화 코드 최적화(FALCON)를 제안하여 이러한 문제를 해결하고 코딩 성능을 향상시키고자 합니다.

- **Technical Details**: FALCON은 두 가지 계층적 구조로 되어 있으며, 장기 메모리는 코드 품질을 향상시키기 위해 학습된 지식을 지속적으로 적용하고, 단기 메모리는 컴파일러와 AI 시스템으로부터의 즉각적인 피드백을 통합합니다. 또한 피드백 보상을 통한 메타 강화 학습을 도입하여 범지구적-국소적 최적화 문제를 해결하고 다양한 코드 생성 과제에 대한 모델의 적응성을 강화합니다.

- **Performance Highlights**: 광범위한 실험 결과, FALCON은 MBPP 벤치마크에서 4.5% 이상, Humaneval 벤치마크에서 6.1% 이상 다른 강화 학습 방법들을 능가하며 최첨단 성능을 달성하였습니다.



### Large Language Model Benchmarks in Medical Tasks (https://arxiv.org/abs/2410.21348)
Comments:
          25 pages, 5 tables

- **What's New**: 이 논문은 의료 분야에서 대규모 언어 모델(LLMs)의 성능 평가를 위한 다양한 벤치마크 데이터세트에 대한 포괄적인 조사를 제공합니다. 특히, 전자 건강 기록(EHR), 의사-환자 대화, 의료 질문-응답 및 의료 이미지 캡션 생성을 포함한 여러 모달리티를 다룹니다.

- **Technical Details**: 논문에서는 MIMIC-III, MIMIC-IV, BioASQ, PubMedQA, CheXpert와 같은 주요 벤치마크 데이터세트를 소개하며, 이는 의료 보고서 생성, 임상 요약 및 합성 데이터 생성과 같은 작업의 발전을 촉진시켰습니다. 데이터세트의 카테고리는 텍스트, 이미지 및 다중 모달리티에 따라 나뉘며, 각 데이터단의 구조와 임상 작업에 미치는 영향을 설명합니다.

- **Performance Highlights**: LLMs의 발전에 따라, 환자의 건강 관리 및 의사결정 지원을 최적화하기 위한 의학적 데이터 세트의 중요성이 강조되며, 데이터의 언어 다양성과 구조화된 오믹스 데이터의 필요성이 제기됩니다. 또한 이러한 벤치마크를 활용하여 다중모달 의료 지능을 발전시키기 위한 도전과 기회도 요약했습니다.



### Towards Trustworthy Machine Learning in Production: An Overview of the Robustness in MLOps Approach (https://arxiv.org/abs/2410.21346)
- **What's New**: 이 논문은 MLOps 시스템의 신뢰성 속성에 대한 포괄적인 개요를 제공하고, robust ML 솔루션 채택을 위한 기술적 실행 방법을 강조합니다.

- **Technical Details**: MLOps는 ML 시스템을 운영하기 위해 DevOps 원칙(continuous integration, continuous delivery)을 채택하고, 데이터 수집, 모델 배포 등 ML 작업의 모든 단계에서 시스템 개발을 지원합니다. 이 논문은 robust ML 시스템의 사양을 formalize하고, MLOps의 기술적 견고성을 위한 기존 접근 방식과 도구들을 리뷰합니다.

- **Performance Highlights**: 이 연구는 robust ML 솔루션의 세 가지 구성 요소를 정의하고, ML 솔루션의 작업 흐름과 MLOps 시스템 요소를 반영하여 신뢰할 수 있는 AI 시스템의 주된 속성으로서의 견고성을 정량화합니다. 또한, 향후 연구 방향 및 기회를 제안합니다.



### Absorb & Escape: Overcoming Single Model Limitations in Generating Genomic Sequences (https://arxiv.org/abs/2410.21345)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 최근 면역학 및 합성 생물학의 발전은 DNA 서열 설계에 대한 심층 생성 방법의 발전을 가속화하고 있습니다. 본 논문은 AutoRegressive (AR) 모델과 Diffusion Models (DMs)의 한계를 분석하고, 두 가지 방식을 조합한 새로운 샘플링 방법인 Absorb & Escape (A&E)를 제안합니다.

- **Technical Details**: DNA 서열은 다양한 기능적 영역 (예: Promoter Regions, Exons, Introns)로 구성되어 있는 비동질적(heterogeneous) 특성을 가지고 있습니다. 현재의 AR 모델은 데이터의 전이 확률을 학습하나 DNA 서열의 전체적인 특성을 포착하지 못하고, DMs는 전반적인 분포를 회복하는 데 유리하나 베이스 페어(base pair) 수준에서 오류를 발생시킬 수 있습니다. A&E 접근 방식은 DMs로 생성된 샘플을 AR 모델을 통해 정제하여 품질을 개선합니다.

- **Performance Highlights**: 15종의 생물체에 대한 광범위한 실험을 통해, A&E 방법은 모티프 분포 및 전체 유전자 유사성까지 고려했을 때 최신 AR 모델과 DMs보다 뛰어난 성능을 보였습니다. 이는 기능적 특성과 다양성에서 기존 방식들이 가지고 있는 한계를 넘어서는 결과를 보여줍니다.



### Heterogeneous Interaction Modeling With Reduced Accumulated Error for Multi-Agent Trajectory Prediction (https://arxiv.org/abs/2410.21342)
Comments:
          20 pages, accepted by IEEE TNNLS

- **What's New**: 이 논문은 복잡한 동적 시스템에서 이질적인 상호작용 모델링을 통해 다중 에이전트 궤적 예측을 개선하는 방법을 제안합니다. 이 방법은 누적 오류를 줄이기 위한 새로운 그래프 엔트로피(graph entropy)와 믹스업 훈련(mixup training) 전략을 도입합니다.

- **Technical Details**: 제안된 방법은 히스토리컬 궤적을 바탕으로 에이전트 간의 동적 상호작용 그래프를 추정합니다. 이를 통해 유도된 상호작용 관계와 영향을 설명하는 heterogeneous attention 메커니즘을 적용하여 이질적인 이웃으로부터의 정보를 집계합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋을 통해 실험한 결과, 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보임을 확인했습니다.



### Retrieval-Retro: Retrieval-based Inorganic Retrosynthesis with Expert Knowledg (https://arxiv.org/abs/2410.21341)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 기계 학습을 활용하여 무기 화합물의 구조를 계획하는 새로운 방법인 Retrieval-Retro를 제안했습니다. 기존의 유기 화합물 합성 경로를 계획하는 방법과는 달리, 무기 화합물에 대한 레트로 합성 경로를 보다 효과적으로 학습할 수 있도록 다양한 주의(attention) 레이어를 통해 참고 자료의 전구체 정보를 암묵적으로 추출합니다.

- **Technical Details**: Retrieval-Retro는 도메인 전문성을 기반으로 하여, 타겟 물질과 전구체 간의 열역학적 관계를 고려함으로써 전구체 세트를 보다 효과적으로 찾을 수 있게 합니다. 이를 통해 기계 학습 모델이 기존 합성 레시피를 초월하는 새로운 합성 레시피를 발견하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 Retrieval-Retro의 효과성을 입증했으며, 특히 새로운 합성 레시피를 발견하는 데 있어 그 우수성을 보여주었습니다. 이 방법은 실제 무기 물질 발견의 응용 가능성을 강조합니다.



### Meta-Learning for Speeding Up Large Model Inference in Decentralized Environments (https://arxiv.org/abs/2410.21340)
- **What's New**: 대규모 모델(large-scale models) 배포에서 발생하는 높은 비용 문제를 해결하기 위해 분산 시스템(decentralized systems)으로의 전환이 이루어지고 있으며, 이 과정에서 효율적인 추론 가속화(inference acceleration) 필요성이 강조되고 있습니다. 본 논문에서는 메타 학습(meta-learning) 기반 프레임워크를 도입하여 최적의 가속화 방법 선택 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 메타 학습 프레임워크(MetaInf)는 다양한 과제(task)에서 여러 가속화 기술의 성능 데이터를 학습하여, 각 과제의 특성에 맞는 최상의 가속화 전략을 식별합니다. 기존의 임의 선택(random selection) 방식이나 전문가의 직관에 의존하는 방법과 달리, 체계적이고 자동화된 방식으로 가속화 방법을 결정합니다. 이 프레임워크는 BSNS 모델에 통합되어 분산 네트워크에서의 효율성을 높입니다.

- **Performance Highlights**: 본 연구는 제안된 메타 학습 기반 프레임워크가 전통적 방법에 비해 효율성과 성능 면에서 일관되게 우수함을 보여주며, 대규모 모델 배포와 AI 민주화(democratization)에 기여함을 강조합니다.



### FinTeamExperts: Role Specialized MOEs For Financial Analysis (https://arxiv.org/abs/2410.21338)
- **What's New**: 완전한 LLM 적용을 위한 금융 전문 모델, FinTeamExperts 소개

- **Technical Details**: FinTeamExperts는 Mixture of Experts (MOEs) 구조를 가진 LLM 프레임워크로, 각 모델은 매크로(Macro), 마이크로(Micro), 정량(Quantitative) 분석에 특화되어 있음. 세 가지의 80억 파라미터 모델이 각기 특화된 데이터를 기반으로 훈련됨. 두 단계의 훈련 과정으로 먼저 역할 기반 데이터로 사전 훈련한 후, 실질적인 금융 작업과 기대에 맞춰 추가 조정(instruct-tuning) 진행.

- **Performance Highlights**: FinTeamExperts는 4개의 데이터셋에서 동일한 규模 및 더 큰 모델들보다 우수한 성능을 발휘함. 특히, 복잡한 작업을 요구하는 데이터셋에서도 각각의 역할 기반 전문가 모델의 효용성을 강조함.



### Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection (https://arxiv.org/abs/2410.21337)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 보안 취약성 중 하나인 프롬프트 인젝션(prompt injection) 공격을 탐구합니다. 이는 LLM을 사용한 응용 프로그램이 신뢰할 수 없는 입력 프롬프트에 의해 조작될 수 있다는 문제를 다루고 있습니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식을 사용하여 프롬프트의 취약성을 탐지합니다: 1) 사전 훈련된 LLM, 2) 파인 튜닝(fine-tuning)된 LLM. 사전 훈련된 XLM-RoBERTa 모델을 사용하여 초기 테스트 데이터셋에서 제로샷 분류(zero-shot classification)를 통해 프롬프트 인젝션을 탐지하였고, 그 후 특정 태스크에 맞춘 레이블이 지정된 데이터셋을 활용하여 파인 튜닝을 실시했습니다.

- **Performance Highlights**: 파인 튜닝된 모델은 99.13%의 정확도, 100%의 정밀도, 98.33%의 재현율(recall), 그리고 99.15%의 F1-score를 기록하며 뛰어난 성과를 보였습니다.



### E(3)-invaraint diffusion model for pocket-aware peptide generation (https://arxiv.org/abs/2410.21335)
- **What's New**: 이 연구에서는 생물학자들이 필요로 하는 새로운 단백질 억제제 발견 방법을 제시합니다. 기존 연구들이 일반적인 구조나 생물학적 속성에 국한된 반면, 본 연구는 특정 수용체 포켓에 주목하여 구조와 서열 생성을 할 수 있는 새로운 기계학습 접근법을 사용합니다.

- **Technical Details**: 본 연구의 방법론은 두 개의 순차적인 diffusion 모델로 구성됩니다: 1) conditional structure diffusion model과 2) conditional sequence diffusion model. 이 모델들은 수용체 포켓의 정보에 기반하여 효소 단백질의 구조와 아미노산 서열을 생성합니다. E(3)-invariant 표현 방식을 통해 백본 원자 간의 각도와 디헤드럴 관계를 반영하여 펩타이드 구조의 일관성을 유지합니다.

- **Performance Highlights**: 제안된 방법은 최신 모델들과 비교하여 경쟁력 있는 성능을 보여주었으며, 특정 수용체 포켓을 고려한 펩타이드 디자인의 잠재력을 강조합니다. 이 연구는 수용체 특이적인 펩타이드 생성을 통한 정밀한 약물 발견의 새로운 접근법을 제공합니다.



### Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Wors (https://arxiv.org/abs/2410.21333)
- **What's New**: 본 논문은 Chain-of-thought(CoT) 프롬프트 기법이 대규모 언어 및 다중 모달 모델의 성능에 미치는 영향을 체계적으로 분석합니다. 특히, CoT가 성능을 저하시키는 특정 작업의 특성을 확인하고, 인지 심리학에서 영감을 받아 CoT의 부정적 영향을 이해하기 위한 새로운 틀을 제시합니다.

- **Technical Details**: CoT는 모델이 '단계별 사고'(think step-by-step)를 하도록 유도하는 프롬프트 기법으로, 여러 작업에서 성능을 향상시키는 것으로 알려져 있습니다. 그러나 본 연구에서는 CoT가 OpenAI o1-preview 등 최신 모델의 성능을 최대 36.3%까지 감소시킬 수 있는 작업을 확인했습니다. 실험은 암묵적 통계 학습(implied statistical learning), 시각적 인식(visual recognition), 예외가 포함된 패턴으로 분류하는 작업에서 수행되었습니다.

- **Performance Highlights**: CoT는 특정 조건(i와 ii를 동시에 만족하는 경우)에서 모델 성능을 크게 저하시켰습니다. 예를 들어, CoT를 사용할 때 OpenAI o1-preview의 성능은 GPT-4o와 비교하여 절대 정확도가 최대 36.3% 감소했습니다. 반면, 다른 세 가지 작업 카테고리에서는 CoT가 모델 성능에 부정적인 영향을 미치지 않는 것으로 나타났습니다.



### Building, Reusing, and Generalizing Abstract Representations from Concrete Sequences (https://arxiv.org/abs/2410.21332)
- **What's New**: 이 논문에서는 비모수 계층 변수 학습 모델(HVM)을 도입하여, 시퀀스에서 청크(Chunk)를 학습하고, 맥락적으로 유사한 청크를 변수로 추상화하는 방법을 제안합니다. 이는 메모리를 효율적으로 조직하고 압축된 시퀀스 표현을 가능하게 합니다.

- **Technical Details**: HVM은 관찰 시퀀스에서 스스로 객체와 카테고리를 발견하는 비지도학습 원칙을 사용하며, 청크 제안(Chunk Proposal)과 변수 발견(Variable Discovery)이라는 두 가지 메커니즘을 내세웁니다. HVM은 기존 모델들보다 더 효율적인 딕셔너리를 학습하며, 압축과 일반화 사이의 정밀한 균형을 실현합니다.

- **Performance Highlights**: HVM은 babyLM 언어 데이터셋에서 일반적인 압축 알고리즘인 Lempel-Ziv보다 더 효율적인 성능을 보여주며, 시퀀스를 상기하는 작업에서 HVM의 시퀀스 가능도가 인간의 회상 시간과 관련이 있음을 증명합니다. 대형 언어 모델(LLM)들과의 비교를 통해, HVM은 인간과 유사한 추상화 기제를 보여주고, LLM은 추상 변수를 효과적으로 전이하지 못함을 강조합니다.



### Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness (https://arxiv.org/abs/2410.21331)
- **What's New**: 이 논문은 deep learning 모델의 해석 가능성에서의 polysemanticity 문제를 다루며, monosemanticity가 해석 가능성 뿐만 아니라 모델의 정확성도 향상시킬 수 있음을 제시합니다. 특히, 모델의 성능을 강화할 수 있는 다양한 상황에서 monosemantic feature의 장점을 발견하였습니다.

- **Technical Details**: 모델이 monosemantic features를 활용할 경우, noisy input 및 label, few-shot learning, 그리고 out-of-domain generalization과 같은 다양한 상황에서 성능이 향상됩니다. 특히, monosemanticity는 feature 표현의 더 나은 분리를 촉진하여 더 견고한 결정 경계를 형성하는 데 기여합니다.

- **Performance Highlights**: 모델은 label noise 하에 13.7%의 top-1 정확도 향상, few-shot fine-tuning에서는 3.9%의 상향을 보여줍니다. 또한, LLM의 fine-tuning에서 MonoLoRA라는 접근법을 사용하여 모델 정렬을 더욱 잘 유지하며 작업 성능을 개선할 수 있음을 보여주었습니다.



### LLM Robustness Against Misinformation in Biomedical Question Answering (https://arxiv.org/abs/2410.21330)
- **What's New**: 이 논문에서는 바이오메디컬 질문에 대한 답변 정확도를 평가하기 위해 Gemma 2, GPT-4o-mini, Llama 3.1, Mixtral 네 가지 대형 언어 모델(LLMs)을 비교했습니다. 특히, 잘못된 정보를 주입하여 생성된 답변이 얼마나 정확한지를 분석하는 데 중점을 두었습니다.

- **Technical Details**: 우리는 세 가지 시나리오에서 LLM의 질문 응답 정확도를 평가했습니다: (1) 기본 LLM 답변 (맥락 없음), (2) '완벽한' 보강 생성 (정확한 맥락 제공), (3) 프롬프트 주입 공격 (잘못된 맥락 제공). Llama 3.1 모델은 기본 및 '완벽한' RAG 시나리오에서 가장 높은 정확도를 기록했습니다.

- **Performance Highlights**: Llama 3.1(70B 파라미터)은 기본 답변에서 0.651의 정확도와 '완벽한' RAG에서 0.802의 정확도를 기록했습니다. 그러나 '완벽한' RAG에서 모델 간의 정확도 차이가 거의 사라져 LLM의 크기와 관련된 효과 차이를 완화할 가능성을 보여줍니다. Llama는 최대 0.48의 정확도 하락을 초래하여 가장 효과적인 공격자로 나타났습니다.



### Deconfounding Time Series Forecasting (https://arxiv.org/abs/2410.21328)
- **What's New**: 이번 연구에서는 기존의 시계열 예측 방법의 한계를 극복하기 위해 잠재적인 혼란 변수(latent confounders)의 영향을 고려하여 예측 품질을 향상시키는 새로운 접근법을 제안합니다. 이 방법은 과거 데이터에서 추출한 혼란 변수의 표현을 통합하여 시계열 예측의 정확성과 강건성을 높이고자 합니다.

- **Technical Details**: 제안된 방법은 Time Series Deconfounder를 기반으로 하며, 인과 추론(causal inference) 기법을 사용하여 혼란 변수의 영향을 식별하고 예측 모델에 통합합니다. 이 모델은 과거 변수의 데이터를 활용하여 현재의 예측 변수를 개선하는 데 초점을 둡니다.

- **Performance Highlights**: 기후 과학 데이터에 대한 적용 결과, 기존의 혼란 변수를 고려하지 않은 전통적인 방법들에 비해 유의미한 예측 정확도 향상을 보여주었습니다.



### Self-Supervised Learning and Opportunistic Inference for Continuous Monitoring of Freezing of Gait in Parkinson's Diseas (https://arxiv.org/abs/2410.21326)
Comments:
          11 pages

- **What's New**: 이 논문에서는 Parkinson's disease (PD) 환자의 Freezing of Gait (FoG) 감지를 위한 LIFT-PD라는 새로운 self-supervised learning 프레임워크를 제안합니다. 이는 라벨이 없는 데이터에서 self-supervised pre-training과 differential hopping windowing 기술을 결합하여 제한된 라벨 인스턴스에서 배울 수 있도록 설계되었습니다.

- **Technical Details**: LIFT-PD는 self-supervised learning 방식을 사용하여 non-FoG 활동과 비교하여 sparse한 FoG 사건을 감지하는 robust한 deep learning 모델을 훈련합니다. 또한, Differential Hopping Windowing(DHW) 기법을 통해 데이터를 전처리하고, opportunistic model activation 모듈을 통해 전력 소비를 줄여줍니다. 이 모델은 3차원 accelerometer 센서를 통해 raw 신호를 수집하여 multivariate time-series classification으로 FrG 사건을 탐지합니다.

- **Performance Highlights**: LIFT-PD는 supervised 모델에 비해 precision이 7.25% 증가하고 accuracy가 4.4% 향상되며, supervised learning에서 사용되는 라벨 데이터를 40%만 사용하여 동일한 성능을 보여줍니다. 또한, inference 시간이 연속 추론에 비해 최대 67% 단축됩니다.



### Just Propagate: Unifying Matrix Factorization, Network Embedding, and LightGCN for Link Prediction (https://arxiv.org/abs/2410.21325)
- **What's New**: 정확한 링크 예측(link prediction)을 위한 통합 프레임워크를 제안함으로써 기존의 다양한 그래프 기반 머신러닝 모델의 이해도를 높이고 새로운 설계를 위한 영감을 제공하고자 함.

- **Technical Details**: 제안하는 프레임워크는 행렬 분해(matrix factorization), 네트워크 임베딩(network embedding) 방법, 그래프 신경망(graph neural network) 방법을 아우름. 이를 통해 링크 예측을 위한 기존 방법의 학습 과정과 성능에 영향을 미치는 여러 가지 설계 요소를 탐색함.

- **Performance Highlights**: 기존 방식들이 프레임워크 내에서 어떻게 표현이 전파(propagation)되는지를 보여주고, 높은 차수 이웃(neighbors) 정보를 어떻게 활용하는가에 대한 통찰력을 제공함.



### Mathematical Derivation Graphs: A Task for Summarizing Equation Dependencies in STEM Manuscripts (https://arxiv.org/abs/2410.21324)
Comments:
          10 pages, 4 figures

- **What's New**: 이 연구는 STEM(Science, Technology, Engineering, Mathematics) 분야의 학술 문서에서 수학적 표현 간의 의존 관계를 이해하기 위한 최초의 단계로, 새로운 객체인 'derivation graph'를 제안합니다. 이 그래프는 논문의 수학 내용을 요약화합니다.

- **Technical Details**: 연구자는 arXiv 데이터베이스에서 수집한 107개의 STEM 논문을 기반으로 수학적 유도 그래프를 수작업으로 레이블링했습니다. 다양한 알고리즘을 사용하여 이 그래프를 자동으로 추출하는 방법을 평가했습니다. 실험적으로, 분석적 및 NLP 모델(LLM 포함)을 평가하여 각각의 문서에서 유도 그래프를 추출하는 능력을 비교했습니다.

- **Performance Highlights**: 연구 결과, 분석적 모델과 NLP 모델 모두 약 40-50%의 F1 점수를 기록하여, 최신 NLP 기법이 단순 분석 모델에 비해 수학적 텍스트를 이해하는데 큰 진전을 이루지 못했음을 보여주었습니다. 따라서 보다 정확하고 심도 있는 수학적 정보 처리를 위한 추가 연구가 필요합니다.



### Angel or Devil: Discriminating Hard Samples and Anomaly Contaminations for Unsupervised Time Series Anomaly Detection (https://arxiv.org/abs/2410.21322)
Comments:
          14 pages, 9 figures, 5 tables

- **What's New**: 이 논문에서는 전통적인 손실 기반 접근 방법 대신 '파라미터 행동(parameter behavior)'을 보완하여 이상 패턴의 세밀한 특성을 담아내는 새로운 기법, 즉 듀얼 파라미터-손실 데이터 증강 방법(PLDA)을 제안합니다.

- **Technical Details**: PLDA는 강화 학습(reinforcement learning) 패러다임 내에서 구현되며, 학습 데이터의 증대를 동적으로 수행합니다. 이 방법은 파라미터 민감도(parameter sensitivity)를 기반으로 한 보상 함수를 사용하여 정상 샘플(hard normal samples, HS)과 이상 상태 스팸(anomaly contaminations, AC)을 효과적으로 구분합니다.

- **Performance Highlights**: 10개 데이터세트에 대한 광범위한 실험 결과, PLDA는 기존의 데이터 증강 방법들보다 최대 8% 향상된 성능을 보였으며, 다양한 이상 감지 모델에 원활하게 통합될 수 있음을 입증하였습니다.



### User-Aware Multilingual Abusive Content Detection in Social Media (https://arxiv.org/abs/2410.21321)
- **What's New**: 본 연구에서는 다국어 인디크 언어에서의 모욕적인 콘텐츠 탐지에 대한 새로운 방법론을 제안합니다. 특히 자원이 부족한 저자원 언어에 중점을 두어 사회적 맥락과 사용자 이력의 중요성을 강조합니다.

- **Technical Details**: 제안된 방법은 두 개의 별도 모듈에서 사회적 및 텍스트 맥락 기능을 학습하고, 이들 모듈의 통합 표현을 통해 최종 예측을 수행합니다. 이 방법은 1,500,000개 및 665,000개의 다국어 댓글로 구성된 SCIDN 및 MACI 데이터셋에서 성능을 평가하였습니다.

- **Performance Highlights**: 제안된 방법은 SCIDN 데이터셋에서 4.08%, MACI 데이터셋에서 9.52%의 F1-score 향상을 보이며, 최신 방법 대비 성능이 우수함을 입증하였습니다.



### Towards Continuous Skin Sympathetic Nerve Activity Monitoring: Removing Muscle Nois (https://arxiv.org/abs/2410.21319)
Comments:
          4 pages, 5 figures, 1 table, IEEE-EMBS International Conference on Body Sensor Networks: NextGen Health: Sensor Innovation, AI, and Social Responsibility (IEEE BSN 2024)

- **What's New**: 이 연구에서는 비침습적인 피부 자율 신경 활동(SKNA)의 지속적인 모니터링 방법을 제안하며, 심장 관련 질환의 원인을 이해하는 데 기여할 것으로 기대된다.

- **Technical Details**: 연구팀은 심전도(ECG) 전극을 통해 수집된 SKNA 신호에서 근육 소음을 감지하고 제거하기 위해 2D CNN(Convolutional Neural Network) 모델을 활용하였다. 데이터 세트는 다양한 실험 상황에서 수집되었으며, 최종 모델은 baseline, 스트레스 유도 SKNA 및 근육 소음이 포함된 기간을 분류하는 데 있어 평균 89.85%의 정확도를 보였다.

- **Performance Highlights**: 이 연구 결과는 신뢰할 수 있는 SKNA 모니터링의 필요성을 강조하며, 궁극적으로 실제 적용을 위한 웨어러블 SKNA 센서 개발로 나아가는 중요한 기초가 될 것이다.



### Multi-path Exploration and Feedback Adjustment for Text-to-Image Person Retrieva (https://arxiv.org/abs/2410.21318)
- **What's New**: 이번 논문에서는 Multi-Pathway Exploration, Feedback, and Adjustment (MeFa) 프레임워크를 제안하여 텍스트 기반 인물 검색에서의 성능을 개선한다. 기존의 VLP 모델의 한계를 극복하고자 하는 노력이 돋보인다.

- **Technical Details**: MeFa는 여러 경로를 통한 탐색, 피드백 및 조정 과정을 포함하여 intra-modal(단일 모달) 및 inter-modal(다중 모달) 피드백을 활용하여 텍스트와 인물 간의 정밀한 연관성을 이룬다. 세 가지 주요 경로로 조직되며, 각각의 경로는 하드 네거티브 샘플을 생성하거나, 글로벌 및 로컬 정보의 정제를 통해 면밀한 세부 사항을 개선한다.

- **Performance Highlights**: CUHK-PEDES, ICFG-PEDES, RSTPReid의 세 가지 공개 벤치마크에서 MeFa는 선행 기술들보다 뛰어난 인물 검색 성능을 달성하여 추가 데이터나 복잡한 구조 없이도 우수한 결과를 나타냈다.



### MatExpert: Decomposing Materials Discovery by Mimicking Human Experts (https://arxiv.org/abs/2410.21317)
- **What's New**: 이번 연구에서는 MatExpert라는 새로운 프레임워크를 도입했습니다. 이 프레임워크는 Large Language Models(LLMs)와 contrastive learning을 활용하여 솔리드 스테이트(고체) 소재의 발견 및 설계를 가속화합니다. MatExpert는 인간 소재 설계 전문가의 작업 흐름을 바탕으로 검토(retrieval), 전환(transition), 생성(generation)이라는 세 가지 주요 단계를 통합합니다.

- **Technical Details**: MatExpert 프레임워크는 다음의 세 가지 단계로 구성됩니다: 1) Retrieval: 주어진 기준과 가장 잘 일치하는 기존 소재를 식별합니다. 2) Transition: 이 소재를 특정 요구 사항에 맞추기 위해 필요한 수정을 개략적으로 설명합니다. 3) Generation: 제공된 정보를 바탕으로 새로운 소재를 생성합니다. 이 과정은 텍스트-구조 검색(text-structure retrieval) 방식을 결합한 chain-of-thought reasoning을 활용하여 이루어집니다.

- **Performance Highlights**: 실험 결과, MatExpert는 소재 생성 작업에서 최신 방법들보다 우수한 성능을 발휘하며, 유효성(validity), 분포(distribution), 안정성(stability) 등의 다양한 메트릭에서 탁월한 결과를 보였습니다. MatExpert는 기존의 데이터 세트인 NOMAD에서 2,886,120 개의 소재를 기반으로 광범위한 평가를 실시하여 해당 분야에서의 일반화 가능성과 강건성을 입증하였습니다.



### Deep Optimizer States: Towards Scalable Training of Transformer Models Using Interleaved Offloading (https://arxiv.org/abs/2410.21316)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 효율적인 훈련을 위한 새로운 방법인 Deep Optimizer States를 제안합니다. 이 방법은 GPU 메모리와 호스트 메모리를 동적으로 관리하여 메모리 한계를 극복하고 훈련 속도를 향상시킵니다.

- **Technical Details**: Deep Optimizer States는 모델 파라미터와 옵티마이저 상태를 작은 부분집합으로 분할하여 각 변화를 GPU와 CPU 간에 인터리빙 방식으로 오프로드하도록 설계되었습니다. 이러한 방법으로 GPU와 CPU의 기계성이 최적화되어 훈련 성능을 극대화합니다.

- **Performance Highlights**: 연구 결과, Deep Optimizer States를 적용하여 기존의 최첨단 기술보다 2.5배 더 빠른 훈련 속도를 달성했습니다. 특히, 20B 파라미터의 모델을 대상으로 한 실험에서 유의미한 시간 단축과 모델 업데이트 속도의 3배 향상을 보여주었습니다.



### GraphLSS: Integrating Lexical, Structural, and Semantic Features for Long Document Extractive Summarization (https://arxiv.org/abs/2410.21315)
Comments:
          Short paper submitted to ACL ARR November cycle

- **What's New**: 본 논문에서는 GraphLSS라는 이질적인 그래프 구조를 제안하여 긴 문서의 추출적 요약을 위한 새로운 접근 방식을 소개합니다. 이 구조는 Lexical, Structural, Semantic 특징을 포괄하며, 외부 학습 모델 없이도 문서를 노드(문장과 단어)와 엣지(문장 의미 유사성, 문장 발생 순서 등)로 구성합니다.

- **Technical Details**: GraphLSS는 두 가지 노드(문장 및 단어)와 네 가지 엣지(문장 간 유사성, 순서, 단어 간 유사성 등)를 정의합니다. 이 모델은 GAT(Veličković et al., 2018)와 같은 그래프 신경망을 통해 처리됩니다. 또한, 무거운 교차 엔트로피 손실(weighted cross-entropy loss)을 최적화하여 불균형한 장문서에 대한 추출적 진실 레이블을 생성합니다.

- **Performance Highlights**: GraphLSS는 PubMed와 arXiv 두 개의 벤치마크 데이터셋에서 실험을 수행하였으며, 최근 비그래프 모델을 능가하는 성과를 기록하였습니다. 또한, GitHub에 코드를 공개하여 재현성과 협업을 지원합니다.



### Decoding Diffusion: A Scalable Framework for Unsupervised Analysis of Latent Space Biases and Representations Using Natural Language Prompts (https://arxiv.org/abs/2410.21314)
- **What's New**: 최근의 이미지 생성 알고리즘 발전으로 인해 순방향 확산 모델(Diffusion Models)이 고 품질의 이미지를 생성하는 강력한 도구로 자리잡았습니다. 본 논문은 이러한 모델의 의미론적 잠재 공간(semantic latent space)을 자동으로 탐색할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 자연어 프롬프트와 이미지 캡션을 직접 활용하여 잠재 방향(latent direction)을 매핑합니다. 이 접근법은 특정 벡터를 훈련할 필요 없이, 의미론적 정보(semantic information)를 활용하여 잠재 공간의 자동 이해를 가능하게 합니다. 또한, Latent Consistency Model(LCM)을 활용하여 중간 U-Net 레이어의 출력인 h-space를 샘플링합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 도메인에서 숨겨진 패턴과 연관성을 발견하며, 순방향 확산 모델의 의미론적 지식(semantic knowledge)과 편향(bias)을 분석하는 데 뛰어난 성능을 보였습니다. 이로 인해 해석할 수 있는 더욱 투명한 생성 모델을 위한 가능성을 열었습니다.



### Towards Robust Out-of-Distribution Generalization: Data Augmentation and Neural Architecture Search Approaches (https://arxiv.org/abs/2410.21313)
Comments:
          Hong Kong University of Science and Technology Thesis

- **What's New**: 이 논문에서는 Deep Learning (딥러닝) 분야에서 Out-of-Distribution (OoD) 데이터에 대해 보다 강력한 일반화 방법론을 제안합니다. 기존 모델이 수집된 분포와 다를 경우의 성능 저하 문제를 해결하기 위해 spurious correlation (가짜 상관관계)을 분리하고, context-related features (맥락 관련 특성)에 대한 gradient-based augmentation (그래디언트 기반 증강)을 수행하여 학습된 표현의 강인성을 향상시킵니다.

- **Technical Details**: 논문에서 제안하는 DecAug 방법은 category-related features (범주 관련 특성)과 context-related features (맥락 관련 특성)를 직교화하여 분리합니다. 또한, NAS-OoD(Neural Architecture Search-out-of-distribution)는 가짜 OoD 데이터를 생성하는 조건부 생성기를 배우면서 아키텍처 파라미터를 최적화합니다. 이를 통해 더 강력한 네트워크 아키텍처를 발견하고 있는 것으로 보입니다.

- **Performance Highlights**: DecAug는 다양한 OoD 데이터셋에서 최첨단 방법들을 초월하는 성능을 보여주었으며, NAS-OoD 방법은 업계 데이터셋에서 70% 이상의 오류율 감소를 보이며 실제 애플리케이션에서도 높은 유용성을 입증했습니다.



### $\texttt{PatentAgent}$: Intelligent Agent for Automated Pharmaceutical Patent Analysis (https://arxiv.org/abs/2410.21312)
Comments:
          7 pages

- **What's New**: 이번 연구에서는 제약 산업에서 특허 분석의 통합을 목표로 하는 인공지능 에이전트인 \\texttt{PatentAgent}를 소개합니다. 이는 약물 연구에 영향을 미치는 특허 데이터를 읽고, 화합물 구조를 인식하는 과정을 단순화하며, 연구자들에게 필수적인 도구가 될 것으로 보입니다.

- **Technical Details**: \\texttt{PatentAgent}는 세 가지 주요 모듈로 구성되어 있습니다: \\textit{PA-QA} (특허 질문-응답), \\textit{PA-Img2Mol} (이미지-화합물 구조 변환), \\textit{PA-CoreId} (핵심 화학 구조 식별). 이 시스템은 자연어 쿼리를 처리하고, 화학 이미지에서 분자 구조를 추출하며, 여러 화학 화합물 간의 중심 화학 구조를 정의하는 데 중점을 둡니다.

- **Performance Highlights**: \\textit{PA-Img2Mol}은 CLEF, JPO, UOB, USPTO 특허 벤치마크에서 2.46%에서 8.37%의 정확도를 개선하였고, \\textit{PA-CoreId}는 PatentNetML 벤치마크에서 7.15%에서 7.62%의 향상된 성능을 보여주었습니다. 이러한 결과는 \\texttt{PatentAgent}가 기존 도구보다 얼마나 효과적인지를 강조하고 있습니다.



### MMDocBench: Benchmarking Large Vision-Language Models for Fine-Grained Visual Document Understanding (https://arxiv.org/abs/2410.21311)
Comments:
          Under review

- **What's New**: 대형 비전-언어 모델(LVLMs)의 세밀한 시각적 이해 능력을 평가할 새로운 벤치마크인 MMDocBench를 제안합니다. 기존 벤치마크의 한계를 보완하기 위해 다양한 문서 이미지를 사용하여 평가합니다.

- **Technical Details**: MMDocBench는 15개의 주요 작업과 4,338개의 QA 쌍, 11,353개의 지원 영역을 포함하는 문서 이해 작업으로 구성되어 있습니다. 작업에는 텍스트 인식, 테이블 인식, 문서 위조 탐지 등이 포함되며, 다양한 문서 이미지 유형을 평가할 수 있습니다.

- **Performance Highlights**: 실험을 통해 현재 LVLMs가 지역 예측에서 매우 낮은 성능을 보이며, 문서 이미지에서의 로컬화와 탐지가 다른 작업들보다 더 어려운 것으로 나타났습니다. 또한, 개방형 및 폐쇄형 LVLM 간에 답변 예측에서 차이가 발견되었지만 지역 예측 성능은 유사했습니다.



### Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models, and Challenges (https://arxiv.org/abs/2410.21306)
Comments:
          35 pages

- **What's New**: 이번 설문조사는 법률 분야에서의 자연어 처리(Natural Language Processing, NLP) 기술의 현재 응용 현황을 종합적으로 조사합니다. 148개의 연구를 검토하여 127개의 연구를 최종 선정하여 법률 텍스트 처리와 관련된 독특한 측면과 도전 과제를 조명합니다.

- **Technical Details**: 자연어 처리의 다양한 법률 응용 프로그램, 예를 들어 법률 문서 요약(Legal Document Summarization, LDS), 법률 개체 인식(Named Entity Recognition, NER), 법률 질의 응답(Legal Question Answering, LQA), 법률 텍스트 분류(Legal Text Classification, LTC), 법률 판결 예측(Legal Judgment Prediction, LJP) 등을 다룹니다. 이 설문조사는 15개의 열려 있는 연구 문제가 포함되어 있으며, 인공지능(AI) 응용 프로그램의 편향, 강력하고 해석 가능한 모델의 필요성, 법률 언어의 복잡성을 처리하기 위한 설명 가능성 개선 등을 포함합니다.

- **Performance Highlights**: NLP의 발전으로 법률 분야에서 문서 요약, 질의 응답 및 판결 예측과 같은 복잡한 작업이 단순화되었으며, 변호사와 법률 전문가의 업무 효율성을 높이고 오류를 최소화하는 데 기여하고 있습니다. 그러나 긴 문서를 처리하고 복잡한 언어를 이해하는 데 여전히 어려움이 존재하며, 대형 언어 모델(Large Language Models, LLMs)의 출현이 법률 서비스의 효율성과 접근성을 높일 수 있는 잠재력을 보유하고 있습니다.



### VEMOCLAP: A video emotion classification web application (https://arxiv.org/abs/2410.21303)
Comments:
          Accepted to 2024 IEEE International Symposium on Multimedia (ISM), Tokyo, Japan

- **What's New**: VEMOCLAP을 소개합니다: 사용자 제공 비디오의 감정 내용을 분석할 수 있는 최초의 오픈 소스 웹 애플리케이션입니다. 이전의 연구를 개선하여 멀티헤드 크로스-어텐션(multi-head cross-attention)을 사용하여 비디오 프레임과 오디오로부터 추출된 사전 훈련된 특징을 효율적으로 융합합니다.

- **Technical Details**: VEMOCLAP은 Google Colab에서 호스트되며, 무료 GPU 런타임으로 모든 사용자 수준에서 접근 가능하고 몇 번의 클릭만으로 사용할 수 있습니다. 사용자는 비디오를 업로드하거나 YouTube 링크를 제공할 수 있으며, 예상 감정을 출력하고 자동 음성 인식(ASR), 광학 문자 인식(OCR), 얼굴 감지 및 표정 분류, 오디오 분류, 이미지 캡셔닝과 같은 추가 분석을 포함합니다. 모델은 Ekman-6 비디오 감정 데이터셋을 기반으로 훈련되었으며, 고급 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 모델은 Ekman-6 비디오 감정 데이터셋에서 분류 정확도를 4.3% 향상시켰으며, 데이터 정제를 통해 비디오 감정 모델의 훈련을 개선했습니다. 이 웹 애플리케이션을 통해 사용자는 자신의 비디오나 YouTube 비디오를 분석하고 감정을 분류할 수 있습니다.



### Evaluating the Posterior Sampling Ability of Plug&Play Diffusion Methods in Sparse-View C (https://arxiv.org/abs/2410.21301)
- **What's New**: 이 논문은 Plug&Play (PnP) 확산 모델의 사후 샘플링 능력을 평가하는 데 초점을 맞추고 있으며, 측정된 sinogram이 충분한 정보를 포함하지 않을 때의 문제를 다룹니다.

- **Technical Details**: Sparse-View Computed Tomography (SVCT) 모델을 사용하여 sinogram으로부터 이미지를 재구성하는 과정에서, 적은 수의 투사(≤ 180)로 인해 사후 분포가 피크를 이루지 않거나 다중 모드(multi-modal)가 되는 상황을 평가합니다. 기존의 PnP 확산 모델은 종종 피크가 있는 분포를 가정하였고, 이에 대한 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: 실험 결과, 각 PnP 모델은 투사 수가 줄어들 때마다 약한 사후 분포와 실제 사후 분포 간의 차이가 발생하는 경향을 보였습니다. 이는 PnP 모델의 성능 평가를 위한 새로운 기준의 필요성을 강조합니다.



### Explainable Artificial Intelligent (XAI) for Predicting Asphalt Concrete Stiffness and Rutting Resistance: Integrating Bailey's Aggregate Gradation Method (https://arxiv.org/abs/2410.21298)
Comments:
          The link to web app this https URL this https URL

- **What's New**: 이번 연구는 다양한 집합체(aggregate) 구성을 가진 아스팔트 콘크리트에 대해 해석 가능한 인공지능(XAI) 기법을 활용하여 복원 모듈러스(resilience modulus, MR)와 동적 안정성(dynamic stability, DS)을 분석하였습니다.

- **Technical Details**: 연구는 다층 퍼셉트론(multi-layer perceptron) 아키텍처를 갖춘 딥러닝 모델을 사용하여 Bailey의 방법으로부터 유도된 집합체 구성을 기준으로 MR과 DS를 예측하였으며, 모델 성능은 k-폴드 교차 검증(k-fold cross-validation)을 통해 검증되었습니다. 또한, SHAP(SHapley Additive exPlanations) 값을 사용하여 모델의 예측을 해석하였습니다.

- **Performance Highlights**: 주요 발견으로는 크리티컬 집합체 크기 임계값을 식별하였고, 특히 0.6 mm 체크 사이즈가 MR과 DS에 중요한 영향을 미치는 것으로 나타났습니다. 연구는 굵은 집합체가 러트 저항에 주로 영향을 미치고 중간-미세 집합체가 강성에 영향을 준다는 것을 밝혀냈습니다. 실용성을 위해 웹 기반 인터페이스가 개발되어 MR과 DS 예측이 가능하게 되었으며, 결과의 해석을 위한 해석 가능한 기능이 포함되었습니다.



### Large-scale Multi-objective Feature Selection: A Multi-phase Search Space Shrinking Approach (https://arxiv.org/abs/2410.21293)
- **What's New**: 이 논문에서는 Feature Selection에 대한 새로운 접근 방식인 LMSSS(제목: Large-scale Multi-objective Shrinking Subspace)라는 대규모 다목적 진화 알고리즘을 제안합니다. 이 알고리즘은 불필요한 특징을 제거하여 검색 공간의 차원을 줄임으로써 Sparse Optimization 문제를 해결하는 데 도움을 줍니다.

- **Technical Details**: LMSSS 방법은 특징과 클래스 레이블 간의 상관관계 및 초기 비용 효율적인 진화 과정에서의 빈도를 기반으로 한 순위 기반 필터링 방법을 포함합니다. 또한, 더 나은 분류 정확도를 가진 부모 솔루션에 더 높은 가중치를 부여하는 스마트 교차 방식과, 인구에서 조기에 제외된 특징을 대상으로 하는 지능형 돌연변이 과정을 설계하였습니다. 이러한 통합 기법을 통해 LMSSS는 대규모 Feature Selection 문제의 탐색을 보다 효율적이고 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: 15개의 대규모 데이터 세트에 대한 포괄적인 실험을 통해 제안된 알고리즘의 효과성을 입증하였으며, 기존의 최첨단 대규모 Feature Selection 알고리즘보다 더 정확한 특징 집합을 식별할 수 있는 잠재력을 보여주었습니다. 이 결과는 LMSSS가 모델 성능과 계산 효율성을 개선할 수 있는 능력을 강조하며, 이 분야에서 새로운 벤치마크를 설정하는 것을 목표로 하고 있습니다.



### A Systematic Assessment of OpenAI o1-Preview for Higher Order Thinking in Education (https://arxiv.org/abs/2410.21287)
Comments:
          An assessment of OpenAI o1-Preview for Higher Order Thinking in Education

- **What's New**: 이 연구는 OpenAI o1-preview의 고차원 인지 작업 수행 능력을 14개 차원에서 평가하고, 인간과의 성과 비교를 통해 AI의 교육적 잠재력을 규명했습니다.

- **Technical Details**: 연구에서는 Ennis-Weir Critical Thinking Essay Test, Biological Systems Thinking Test 등의 유효한 도구를 사용해 o1-preview의 성과를 분석했습니다. o1-preview는 시스템 사고, 계산적 사고, 데이터 리터러시 등 여러 분야에서 인간 평균을 뛰어넘는 성과를 보였지만, 논리적 사고와 비판적 사고에서는 약 25% 낮은 성과를 기록했습니다.

- **Performance Highlights**: 대부분의 카테고리에서 o1-preview는 인간보다 150% 향상된 결과를 나타냈으며, 특히 시스템 사고와 창의적 사고에서 매우 두드러진 성과를 기록했습니다. 그러나 추상적 추론에서 인간 심리학 전공 학생들에 비해 열악한 성과를 보였고, 이로 인해 고차원적인 추론이 필요한 작업에서 인간의 감독이 여전히 중요하다는 점이 강조되었습니다.



### OpenCity: A Scalable Platform to Simulate Urban Activities with Massive LLM Agents (https://arxiv.org/abs/2410.21286)
- **What's New**: OpenCity는 LLM(대형 언어 모델) 에이전트의 도시 활동 시뮬레이션을 위한 확장 가능한 플랫폼으로, 시스템 및 프롬프트 효율성을 최적화하여 고성능 도시 모형을 제공한다. 최근 LLM의 발달에 따라, 이 플랫폼은 도시 공간 내에서의 복잡한 사회적 현상을 시뮬레이션할 수 있게 되었다.

- **Technical Details**: OpenCity는 LLM 요청 스케줄러를 사용하여 IO 다중화( multiplexing) 기법을 통해 LLM 요청의 통신 오버헤드를 줄인다. ‘group-and-distill’ 프롬프트 최적화 전략을 도입하여 유사한 정적 속성을 가진 에이전트를 클러스터링하여 중복성을 최소화한다. 이 플랫폼은 10,000명의 에이전트의 일일 활동을 1시간 안에 시뮬레이션할 수 있으며, 일반 하드웨어에서도 동작 가능하다.

- **Performance Highlights**: OpenCity는 10,000 LLM 에이전트의 시뮬레이션에서 평균 635배 가속화를 달성하며, LLM 요청 수는 73.7%, 사용된 토큰은 45.5% 줄어든다. 이는 도시 활동의 실질성을 높이며, 6개 도시의 실제 데이터와의 비교를 통해 LLM 에이전트의 능력을 최초로 검증하는 기준을 마련하는 데 기여한다.



### AI-driven innovation in medicaid: enhancing access, cost efficiency, and population health managemen (https://arxiv.org/abs/2410.21284)
- **What's New**: 이 논문은 미국의 Medicaid 프로그램에서 인공지능(AI)의 사용이 어떻게 의료 시스템을 변화시킬 수 있는지를 조사합니다.

- **Technical Details**: Medicaid 프로그램은 의료 비용 증가, 접근성 불균형, 인구 건강 요구의 다양성과 같은 여러 도전에 직면해 있습니다. 본 논문은 AI가 predictive analytics, care coordination, fraud detection, personalized medicine과 같은 분야에서 중요한 역할을 할 수 있음을 설명합니다.

- **Performance Highlights**: AI의 도입을 통해 운영 효율성을 개선하고, 공공 건강 결과를 개선하며, 모든 수혜자를 위한 더 접근 가능하고 공평한 의료 시스템을 구축할 수 있는 긴급성을 강조합니다.



### pLDDT-Predictor: High-speed Protein Screening Using Transformer and ESM2 (https://arxiv.org/abs/2410.21283)
Comments:
          6 pages main topic, 8 pages including citiation, 4 figures

- **What's New**: pLDDT-Predictor는 미리 훈련된 ESM2 단백질 임베딩과 Transformer 아키텍처를 활용하여 AlphaFold2의 pLDDT 점수를 빠르고 정확하게 예측하는 도구이다. 이 모델은 단백질 구조 품질 평가의 필요성을 충족시키며 동시에 컴퓨팅 리소스를 절약할 수 있도록 설계되었다.

- **Technical Details**: pLDDT-Predictor는 ESM2(진화적 규모 모델링) 임베딩 레이어와 Transformer 인코더를 결합한 고급 딥러닝 모델이다. ESM2는 각 아미노산의 진화적 특성을 포착하고, Transformer는 문자열 내 용접(dependencies)을 학습하여 단백질 구조 품질을 예측한다. 최종 예측 점수는 Huber 손실 함수에 기반하여 훈련되며, Adam 옵티마이저와 CosineAnnealingLR 스케줄러를 사용하여 최적화된다.

- **Performance Highlights**: 실험 결과, pLDDT-Predictor는 150만 개의 단백질 서열에서 70 이상의 pLDDT 점수를 가지는 단백질의 90% 이상을 분류할 수 있으며, 이는 AlphaFold2의 신뢰성과 밀접하게 일치한다.



### Logic Error Localization in Student Programming Assignments Using Pseudocode and Graph Neural Networks (https://arxiv.org/abs/2410.21282)
- **What's New**: 본 논문에서는 프로그래밍 기초 과정에서 학생들이 알고리즘 설계를 배우는 데 도움을 주기 위해 고안된 시스템을 소개합니다. 이 시스템은 pseudocode를 활용하여 학생 코드에서 logic error(논리 오류)를 로컬라이즈하고 수정 제안을 제공합니다.

- **Technical Details**: 학생들의 프로그래밍 과제에서 논리 오류를 줄이기 위해 source code (소스 코드)와 pseudocode (의사 코드)의 그래프를 구축하고, Graph Neural Network (GNN)를 활용하여 오류를 정확히 찾아내고 수정을 제안합니다. 또한, DrRepair 모델을 바탕으로 syntax error (구문 오류) 수정 과정에서 논리 오류 취약 프로그램을 수집하여 효율적인 데이터셋을 생성합니다.

- **Performance Highlights**: 실험 결과, 논리 오류의 로컬라이제이션 정확도가 99.2%에 이르며, 이는 학생들의 코딩 능향상 및 오류 수정 기술 향상에 큰 기여를 증명합니다.



### The Social Impact of Generative LLM-Based AI (https://arxiv.org/abs/2410.21281)
Comments:
          34 pages, 3 figures, 2 tables

- **What's New**: 인공지능(AI) 혁명은 경제 생산과 사회 생활을 지배할 새로운 시대에 접어들게 됨을 암시합니다. 이 논문에서는 생성적 LLM 기반 AI(GELLMAI)가 사회에 미치는 영향을 논의합니다.

- **Technical Details**: 논문은 기술 발전에 기여하는 사회적 요인과 GELLMAI의 두 가지 주요 역할, 즉 국가 간 및 국가 내 사회적 불평등을 심화시키는 가능성을 중점적으로 다룹니다.

- **Performance Highlights**: 미국과 중국이 AI 분야의 주요 경쟁자가 될 것으로 예상되며, AI 혁명은 지식의 중요성보다 개인 관계 및 사회적 정체성이 더 중요해질 '포스트-지식 사회(post-knowledge society)'의 출현을 이끌 것으로 보입니다.



### TraderTalk: An LLM Behavioural ABM applied to Simulating Human Bilateral Trading Interactions (https://arxiv.org/abs/2410.21280)
Comments:
          4 pages

- **What's New**: 본 연구는 Agent-Based Models (ABMs)과 Large Language Models (LLMs)를 결합하여 인간 거래 상호작용을 시뮬레이션하는 새로운 하이브리드 접근 방식을 소개합니다. 이 모델을 TraderTalk이라고 부르며, LLM을 사용하여 금융 거래에서의 양방향 대화를 세밀하게 포착합니다.

- **Technical Details**: TraderTalk은 Generative Agent-Based Model (GABM)이며, Concordia라는 오픈 소스 소프트웨어를 활용하여 LLM을 ABM에 통합합니다. CoT(Chain-of-Thought) 방법론을 적용하여 에이전트 간의 대화에서 의사 결정 및 협상 과정을 시뮬레이션합니다.

- **Performance Highlights**: 300회의 시뮬레이션 중 LLM은 60%의 정확도로 거래하지 않기로 결정한 경우를 올바르게 추론하였으며, 이는 Agens의 거래 의도를 성공적으로 이해하고 실행할 수 있는 가능성을 보여줍니다.



### Comparative Global AI Regulation: Policy Perspectives from the EU, China, and the US (https://arxiv.org/abs/2410.21279)
Comments:
          36 pages, 11 figures and tables

- **What's New**: 이번 논문은 AI(Artificial Intelligence)의 규제를 다루고 있으며, 유럽연합(EU), 중국, 미국의 세 가지 접근 방식을 비교합니다. 특히 미국의 캘리포니아주에서 진행 중인 상원 법안 1047에 대해 중점적으로 분석하고 있습니다.

- **Technical Details**: AI 규제는 각국의 문화적, 정치적 및 경제적 관점을 반영합니다. EU는 상향식 위험 기반 접근 방식, 미국은 시장 주도 접근 방식을 강조하며, 중국은 중앙 집중식 규제를 위장한 분산 혁신과 지역 경쟁을 목표로 하는 접근 방식을 제공합니다. 각국의 규제 체계는 AI 기술의 사회적, 경제적, 지정학적 영향에 대한 우려를 공통적으로 가지고 있습니다.

- **Performance Highlights**: EU의 AI 규제는 법률적 및 규제적 체계를 통해 경쟁력을 강조하며 윤리적 고려사항도 배려하지만, 지나치게 엄격한 규제가 혁신을 저해할 수 있는 위험이 있습니다. 블룸버그의 보고서는 미국의 AI 규제가 더욱 중앙집중적이고 제재적인 방향으로 나아갈 가능성을 제기하고 있습니다.



### Grounded GUI Understanding for Vision Based Spatial Intelligent Agent: Exemplified by Virtual Reality Apps (https://arxiv.org/abs/2409.10811)
- **What's New**: 이 논문에서는 가상 현실(VR) 애플리케이션에서의 상호작용 가능한 GUI 요소(IGE) 인식을 위한 새로운 접근 방식인 Orienter를 제안합니다. Orienter는 사람의 행동을 모방하여 VR 앱 장면의 의미적 맥락을 이해한 후 감지를 수행합니다.

- **Technical Details**: 이 프레임워크는 세 가지 주요 모듈로 구성됩니다: 1) 의미적 맥락 이해(Semantic context comprehension), 2) 반영 지향 IGE 후보 탐지(Reflection-directed IGE candidate detection), 3) 맥락 감수성 상호 작용 가능성 분류(Context-sensitive interactability classification). 이 시스템은 LMM(사전 훈련된 대규모 다중 모달 모델)을 활용하여 각 VR 장면의 전반적 및 국소적 맥락을 결합합니다.

- **Performance Highlights**: Orienter는 기존의 GUI 요소 탐지 방법들보다 뛰어난 성능을 보이며, VR 앱에서의 IGE 탐지를 획기적으로 개선할 수 있음을 실험을 통해 입증합니다.



New uploads on arXiv(cs.LG)

### Optimizing Posterior Samples for Bayesian Optimization via Rootfinding (https://arxiv.org/abs/2410.22322)
- **What's New**: 본 논문은 베이지안 최적화(Bayesian Optimization, BO)에 있어 포스터리어 샘플(posterior samples)을 이용한 acquisition functions의 전역 최적화(global optimization)를 위한 새로운 전략을 제시합니다. 특히, 포스터리어 샘플에 기반한 여기서 제안된 획기적인 방법은 기존의 EI 및 GP-UCB 방법보다 우수한 성능을 보였습니다.

- **Technical Details**: 이 알고리즘은 포스터리어 샘플 기반의 acquisition function을 효율적으로 최적화하기 위해 global rootfinding을 기반으로 합니다. 특히, gradient-based optimizers에 적절하게 선택된 시작점을 제공하여 exploitation(착취)와 exploration(탐색)을 결합할 수 있도록 설계되었습니다. 본 논문에서는 Gaussian process Thompson sampling (GP-TS)와 같은 포스터리어 샘플 기반 acquisition functions의 최적화를 제안하며 이 방법론이 고차원에서도 실제로 선형적으로 확장될 수 있음을 보여줍니다.

- **Performance Highlights**: 다양한 입력 차원을 가진 문제 세트에 대해 실험한 결과, 본 방법은 inner-loop 뿐만 아니라 outer-loop 최적화 성능에서도 상당한 개선을 보였습니다. 특히, 포스터리어 샘플을 기반으로 한 acquisition function의 수렴 속도가 다른 일반적 acquisition function보다 빠른 것으로 나타났습니다.



### Online Detecting LLM-Generated Texts via Sequential Hypothesis Testing by Betting (https://arxiv.org/abs/2410.22318)
- **What's New**: 최근 몇 년 간 LLM(대형 언어 모델)에 의해 생성된 텍스트와 인간이 작성한 텍스트를 구별하는 알고리즘 개발에 주목할 만한 관심이 집중되고 있습니다. 본 논문에서는 온라인(online) 시나리오에서 신속하고 정확하게 소스가 LLM인지 인간인지 판별할 수 있는 알고리즘을 제안합니다. 이는 통계적 보장이 필요하며, 기존의 오프라인 접근 방법보다는 실시간으로 텍스트가 생성되는 상황을 처리하는 데 중점을 둡니다.

- **Technical Details**: 제안된 알고리즘은 sequential hypothesis testing을 기반으로 하며, 텍스트가 연속적으로 유입되는 온라인 환경에서 LLM을 신속하게 식별합니다. 각 라운드 t에서 관찰된 텍스트에 대해 이는 H_0(Null Hypothesis)를 설정하여, 이 가설을 올바르게 기각할 경우 해당 소스가 LLM임을 확인할 수 있습니다. 알고리즘은 타입 I 오류(허위 긍정율)를 제어하고 타입 II 오류(허위 부정율)를 최소화하며, LLM으로 판별하는 데 필요한 전체 라운드의 기대값 상한을 설정하는 것을 목표로 합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법의 효과성을 입증하였으며, 연구결과는 추후 발표와 함께 코드와 데이터세트가 공개될 예정입니다. 이는 신뢰할 수 있는 기초 통계적 보장을 제공하면서, 오프라인에서 개발된 기존 방법의 성능을 보완하는 방식으로 설계되었습니다.



### Convex Formulations for Training Two-Layer ReLU Neural Networks (https://arxiv.org/abs/2410.22311)
- **What's New**: 본 논문에서는 비행렬(width가 무한한) 두 층 ReLU 신경망(Neural Network)을 훈련하기 위한 문제를 완전 양의 프로그램(convex completely positive program)으로 재구성하는 방법을 제시합니다. 기존의 비선형 활성화 함수(non-linear activation functions)와 관련하여 신경망과 볼록 최적화(convex optimization) 간의 연결을 촉진합니다.

- **Technical Details**: 제안된 방법은 고정된 입력 및 출력 차원과 특정 수의 데이터 포인트를 제공하며, 네트워크의 풍부함이 포화된 이후 훈련 문제가 제안된 완전 양의 프로그램과 동등해집니다. 원래의 구성은 NP-hard 문제이므로, 이를 해결하기 위한 반양자화(semidefinite relaxation)를 도입하여 다항 시간 내에 해결할 수 있도록 합니다.

- **Performance Highlights**: 실험적으로 제안된 반양자화의 구속성을 평가하고, 여러 분류 작업에서 매우 경쟁력 있는 테스트 정확도(test accuracy)를 달성합니다. 방법은 Neural Network Gaussian Process(NNGP) 및 Neural Tangent Kernel(NTK) 방법과 비교 시 유의미한 성능으로 나타났습니다.



### SVIP: Towards Verifiable Inference of Open-source Large Language Models (https://arxiv.org/abs/2410.22307)
Comments:
          20 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 검증 가능한 추론 문제를 형식화하고, 이용자와 컴퓨팅 제공자 간의 신뢰성을 높이기 위해 중간 출력을 활용한 비밀 기반의 검증 가능한 LLM 추론 프로토콜(SVIP)을 제안합니다.

- **Technical Details**: SVIP는 LLM의 중간 출력(hidden state representations)을 고유한 모델 식별자로 사용하여, 컴퓨팅 제공자가 생성한 출력이 실제 요청된 LLM에서 온 것인지 검증할 수 있는 mecanismos입니다. 이 방법은 컴퓨팅 제공자에게 생성된 텍스트와 프로세스된 중간 출력을 반환하도록 요구하여, 사용자에게 신뢰할 수 있는 검증 수단을 제공합니다.

- **Performance Highlights**: SVIP는 평균 3.49%의 잘못된 부정률(false negative rate) 및 3% 미만의 잘못된 긍정률(false positive rate)을 유지하며, 한 쿼리에 대해 0.01초 이하의 검증 시간을 요구합니다. 본 프로토콜은 약 80에서 120만 건의 프롬프트 쿼리를 안전하게 처리할 수 있도록 설계되었습니다.



### LLMs are Highly-Constrained Biophysical Sequence Optimizers (https://arxiv.org/abs/2410.22296)
Comments:
          Supercedes arXiv:2407.00236v1

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)을 고도로 제약된 이층 최적화기(bilevel optimizer)로 활용하는 방법인 언어 모델 최적화와 여유 기대(Margin Expectation)를 제안합니다. 이를 통해 생물학적 제약을 충족하면서 효율적으로 시퀀스를 생성을 시도합니다.

- **Technical Details**: 제안된 방법론은 오프라인(offline) 및 온라인(online) 최적화를 결합하여 LLM이 생성한 시퀀스를 반복적으로 개선하고, 새로운 훈련 목표인 여유 정렬 기대(MargE)를 도입하여 입력과 목표 간의 보상 여유를 원활하게 보간합니다.

- **Performance Highlights**: LLMs는 유전자 알고리즘(genetic algorithm) 기초선과 비교해 훨씬 낮은 후회(regret) 솔루션을 얻으면서 적은 테스트 함수 평가를 요구하는 성과를 보였습니다. 그러나 LLM들은 중간 정도의 잘못된 보정(miscalibration)과 생성기 붕괴(generator collapse)에 취약합니다.



### Fourier Head: Helping Large Language Models Learn Complex Probability Distributions (https://arxiv.org/abs/2410.22269)
Comments:
          Project page and code are at this https URL

- **What's New**: 본 논문에서는 연속적인 구조를 고려하여, 기존 LLM(linear language model) 구조를 개선하기 위해 푸리에 변환(Fourier transformation) 기반의 새로운 신경망 레이어인 푸리에 헤드(Fourier head)를 제안합니다. 이 레이어는 고질적 소음(high-frequency noise)을 무시하면서 데이터를 통해 신호(signal)를 더 잘 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 푸리에 헤드는 입력 벡터를 받아 신경망의 선형 계층(linear layer)을 통해 푸리에 계수를 학습하고, 이를 통해 연속적인 확률 밀도 함수(continuous probability density function)를 학습합니다. 입력값은 [-1, 1] 범위에서 m개의 동등한 구간으로 양자화되며, 각 구간에서의 확률을 카테고리 분포로 반환합니다.

- **Performance Highlights**: 푸리에 헤드는 Atari Seaquest 게임에서 결정 트랜스포머(Decision Transformer)의 성과를 46% 향상시켰으며, 20개의 비훈련 데이터셋에서 3.5%의 예측 개선을 보여주어 최신 시간 시계열 모델의 성능을 초월했습니다.



### Meta-Learning Adaptable Foundation Models (https://arxiv.org/abs/2410.22264)
Comments:
          Preprint

- **What's New**: 본 논문은 다양한 작업에 쉽게 적용할 수 있는 모델을 학습하기 위해 PEFT(파라미터 효율적인 미세조정)와 메타 학습(meta-learning) 프레임워크를 결합했습니다. 이를 통해 훈련과 미세조정 단계의 독립성 문제를 해결하고자 합니다.

- **Technical Details**: 모델은 선형 모델을 기반으로 하며, 저랭크 적응(low-rank adaptations)을 사용합니다. 이 메타 학습 프레임워크는 초보 단계에서 적합한 파라미터 세트를 찾는 일반적인 재훈련 방법의 비효율성을 해결합니다.

- **Performance Highlights**: ConvAI2 데이터셋에서 서로 다른 페르소나 간의 대화가 지속되는 예측 작업을 통해 로버타(RoBERTa) 모델을 재훈련한 결과, 제안된 메타 학습 기법을 사용한 경우 전통적인 접근 방식 대비 성능 향상이 뚜렷하게 나타났습니다.



### LipKernel: Lipschitz-Bounded Convolutional Neural Networks via Dissipative Layers (https://arxiv.org/abs/2410.22258)
- **What's New**: 이 논문은 합성곱 신경망(Convolutional Neural Networks, CNNs)을 위한 새로운 레이어별 매개변수화 방법을 제안합니다. 이 방법은 정해진 Lipschitz 경계를 강제하여 내장된 강건성 보장을 포함하는 계층을 설계합니다.

- **Technical Details**: 각 레이어는 선형 행렬 부등식(Linear Matrix Inequality, LMI)을 만족하도록 설계되어, 이는 특정 공급률에 대한 소산성(dissipativity)을 의미합니다. 새로운 방법인 LipKernel은 Roesser 유형의 상태 공간 모델을 사용하여 소산적인 합성곱 커널을 직접 파라미터화합니다. 이 방식은 합성곱 층이 훈련 후 표준 형태로 제공되며, 계산 비용이 줄어듭니다.

- **Performance Highlights**: 수치 실험 결과, 제안한 방법은 Fourier 영역에서 합성곱을 매개변수화하는 최신 Lipschitz 경계 네트워크보다 수 배 빠른 처리 시간을 보였습니다. 이로 인해 로봇, 자율주행차, 자동화 시스템에서의 실시간 인식 또는 제어의 강건성을 개선하는 데 특히 매력적인 접근법입니다.



### Hypergraph-based multi-scale spatio-temporal graph convolution network for Time-Series anomaly detection (https://arxiv.org/abs/2410.22256)
- **What's New**: 이번 연구에서 제안하는 모델은 다변량 시계열 데이터의 이상 탐지를 위한 STGCN_Hyper로, 고차원 및 복합 데이터 세트에서 효과적이고 정확한 이상 탐지를 가능하게 합니다.

- **Technical Details**: STGCN_Hyper 모델은 하이퍼그래프(hypergraph) 기반의 동적 그래프 구조 학습 모듈을 통해 다수의 변수 간의 고차, 다중 홉 상관관계를 명시적으로 포착합니다. 또한, 다중 스케일 TCN(Temporal Convolutional Network) 확장 합성 모듈을 도입하여 시간 차원에서 다양한 스케일의 특징 종속성을 캡처할 수 있습니다.

- **Performance Highlights**: 실험 결과, STGCN_Hyper 모델은 여러 시계열 데이터셋에서 precision, recall, F1-score 등에서 기존의 대부분의 기준 모델을 초월하는 성능을 보였습니다.



### Abrupt Learning in Transformers: A Case Study on Matrix Completion (https://arxiv.org/abs/2410.22244)
Comments:
          NeurIPS 2024 Poster

- **What's New**: 이번 연구에서는 Transformer의 훈련 동역학에 대한 분석을 통해 훈련 손실이 상당한 수의 훈련 단계에서 정체(plateau)된 후, 갑작스럽고 급격하게 근접 최적 값으로 떨어지는 흥미로운 특성을 제시합니다.

- **Technical Details**: 저자들은 저순위 행렬 완성(low-rank matrix completion) 문제를 마스크 언어 모델링(masked language modeling, MLM) 작업으로 공식화하고, 이를 해결하기 위해 BERT 모델을 성공적으로 훈련시켜 낮은 오류를 달성할 수 있음을 보여줍니다. 훈련 초기에는 손실 곡선이 평탄하게 유지되지만, 갑작스럽게 근접 최적 값으로 떨어지는 현상을 관찰하였습니다. 이는 훈련 절차나 하이퍼파라미터(hyper-parameters)에 변화가 없음에도 불구하고 발생합니다.

- **Performance Highlights**: 가장 두드러진 관찰 중에는 (a) 모델이 마스크 처리된 입력을 단순히 복사하는 것을 넘어서 마스크된 항목을 정확하게 예측하는 단계로 전환하며, (b) 주의(attention) 헤드가 작업에 관련된 해석 가능한 패턴으로 변화하고, (c) 임베딩(embeddings)과 은닉 상태(hidden states)가 문제와 관련된 정보를 인코딩하는 모습을 보여줍니다. 또한 각 모델 구성 요소의 훈련 동역학을 분석하여 손실의 급격한 감소를 이해하고자 하였습니다.



### Auditing $f$-Differential Privacy in One Run (https://arxiv.org/abs/2410.22235)
- **What's New**: 본 논문에서는 개인 정보 보호 알고리즘의 이행에서 발생하는 결함을 발견하는 경험적 감사 절차 및 분석 방법을 제안합니다. 기존 감사 메커니즘들은 다수의 기계 학습 알고리즘 실행을 요구하거나 경험적 개인 정보 보호를 계산하는 데 비효율적이었습니다. 본 연구의 감사 절차는 효율적이며 단일 실행으로도 개인 정보 보호를 효과적으로 평가할 수 있습니다.

- **Technical Details**: 우리는 새롭게 제안된 감사 절차가 f-DP(f-differential privacy) 곡선을 활용하여 기존의 ε, δ(딜타) 차별 개인 정보 보호 파라미터보다 더욱 정확한 개인 정보 보호 추정을 가능하게 한다는 점에서 차별화됩니다. 이 방법은 한 번의 메커니즘 실행만으로 개인 정보 보호를 분석할 수 있도록 돕습니다. 또한, 우리 감사 절차는 Steinke et al. (2023)와 유사한 런던 데이터의 무작위성을 활용합니다.

- **Performance Highlights**: 실험 결과, 우리는 단순한 가우시안 메커니즘 및 실 데이터로 훈련된 모델에 대해 감사 절차의 성능이 Steinke et al. (2023)보다 훨씬 우수함을 보였습니다. 이는 대규모 모델 훈련 맥락에서 컴퓨팅 효율성을 유지하면서도 보다 엄밀한 개인 정보 보호 추정이 가능하다는 것을 의미합니다.



### Subgraph Aggregation for Out-of-Distribution Generalization on Graphs (https://arxiv.org/abs/2410.22228)
- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)에서 Out-of-Distribution (OOD) 일반화를 다룬 새로운 프레임워크인 SubGraph Aggregation (SuGAr)을 제안합니다. 기존 방법들이 단일 인과 서브그래프(syungraph)에 의존하는 반면, SuGAr는 여러 개의 인과 서브그래프를 학습하여 보다 일반화된 예측을 가능하게 합니다.

- **Technical Details**: SuGAr는 맞춤형 서브그래프 샘플러(subgraph sampler)와 다양성 정규화기(diversity regularizer)를 사용하여 다양한 불변 서브그래프(invariant subgraphs)를 추출합니다. 이 불변 서브그래프들은 평균을 내어 집계되어, 데이터에 포함된 인과 구조(causal structures)를 더욱 풍부하게 반영합니다. 결국 SuGAr는 여러 개의 인과 서브그래프를 독립적으로 학습하고, 이를 앙상블하여 OOD 일반화를 개선합니다.

- **Performance Highlights**: 실험 결과, SuGAr는 15개의 다양한 데이터셋에서 기존 최첨단 방법들보다 우수한 성능을 보여주며 OOD 일반화에서 최대 24%의 성능 향상을 달성하였습니다. 이는 SuGAr가 단일 서브그래프뿐만 아니라 다수의 서브그래프 학습을 통해 OOD 일반화를 성공적으로 개선한 첫 사례임을 의미합니다.



### $r$Age-$k$: Communication-Efficient Federated Learning Using Age Factor (https://arxiv.org/abs/2410.22192)
- **What's New**: 이 논문에서는 Federated Learning(FL)의 두 가지 주요 문제, 즉 데이터 이질성과 통신 오버헤드를 동시에 해결하는 새로운 커뮤니케이션 효율적인 알고리즘인 rAge-k을 제안합니다.

- **Technical Details**: rAge-k 알고리즘은 Age of Information (AoI) 메트릭을 사용하여 Gradient Index의 업데이트를 선택적으로 요청합니다. 파라미터 서버(PS)는 각 클라이언트로부터 업데이트된 Gradient Indices의 나이를 추적하는 Age 벡터를 도입하며, 이 벡터를 통해 통계적으로 유사한 데이터를 갖는 클라이언트들을 클러스터링합니다.

- **Performance Highlights**: MNIST 및 CIFAR-10 데이터셋에서 수행된 실험 결과, 제안된 방법이 다른 커뮤니케이션 효율적인 전략들보다 더 빠른 학습 속도와 향상된 정확도를 달성함을 보여줍니다.



### Standardization Trends on Safety and Trustworthiness Technology for Advanced AI (https://arxiv.org/abs/2410.22151)
Comments:
          13 pages, 2 figures, 4 tables

- **What's New**: 최근 인공지능(AI) 기술이 급격히 발전하면서 언어 이해, 이미지 및 비디오 인식, 프로그래밍, 과학적 추론 등 여러 분야에서 진전을 이뤘습니다. 대규모 언어 모델과 기초 모델을 기반으로 한 AI는 인공 일반 지능(AGI)에 근접하거나 이를 초과하는 성능을 보이고 있습니다.

- **Technical Details**: AI의 안전성과 신뢰성 확보를 위한 국제 표준화 노력이 진행되고 있으며, 연구에서는 안전성 및 신뢰성 확보를 위한 기술 영역 식별과 함께 표준화의 미래 방향성 및 전략에 대해 제안합니다. 여기서 중요하게 다루어지는 모델에는 대규모 언어 모델(LLM), 다중 모달 모델(MLLM), 일반 목적 인공지능(AGI), 초지능(ASI) 등이 포함됩니다.

- **Performance Highlights**: 이 연구는 글로벌 차원에서 AI의 안전성과 신뢰성 표준화 기술의 동향을 분석하며, 안전성과 신뢰성을 증진하기 위한 기술적 분야를 확인합니다. 궁극적으로 AI의 안전하고 신뢰할 수 있는 발전을 지원하고 국제 경쟁력을 향상시키기 위한 정책적 시사점을 도출합니다.



### Learning Successor Features the Simple Way (https://arxiv.org/abs/2410.22133)
Comments:
          Main Paper: 10 pages and 8 figures. Accepted at Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이 논문은 픽셀 수준의 관찰에서 Successor Features (SFs)를 학습하기 위한 새로운 간단한 방법을 소개합니다. 이 방법은 Temporal-difference (TD) 손실과 보상 예측 손실을 결합하여 SFs의 기본 수학적 정의를 포착합니다.

- **Technical Details**: 제안된 방법은 SFs를 학습하는데 TD 손실을 사용하며, 이는 에이전트가 환경과 상호작용하는 동안 손실을 최적화하여 대표적인 손실 공간에서 SFs를 직접적으로 학습하게 합니다. 이렇게 함으로써 기존의 복잡한 손실 함수 없이도 효율적으로 SF를 학습할 수 있습니다.

- **Performance Highlights**: 최신 SF 학습 기법과 비교할 때, 저자들의 방법은 2D (Minigrid), 3D (Miniworld) 미로 및 Mujoco에서 단일 및 지속적 학습 시나리오 모두에서 성능이 동등하거나 우수한 결과를 보여줍니다. 또한, 계산 비용이 낮고 빠른 시간 안에 높은 성능을 달성할 수 있는 장점이 있습니다.



### RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier (https://arxiv.org/abs/2410.22124)
Comments:
          Accepted at NeurIPS 2024 (Poster)

- **What's New**: 이번 논문에서는 RushUp이라는 새로운 반감독 학습 프레임워크를 제안합니다. 이는 기존의 반감독 분류 기법을 회귀 문제에 적용하여 성능을 향상시키기 위해 개발되었습니다. RankUp은 원래의 회귀 문제를 순위 문제로 변환하여 보조 순위 분류기를 동시에 훈련시킴으로써 성능을 높입니다.

- **Technical Details**: RankUp은 보조 순위 분류기를 사용하여 원래의 회귀 작업과 함께 순위 작업을 동시에 해결합니다. 이를 통해 기존의 반감독 분류 기법을 활용하여 훈련할 수 있으며, 새로운 방법인 Regression Distribution Alignment (RDA)를 통해 의사 라벨의 품질을 더욱 개선합니다.

- **Performance Highlights**: RankUp은 RDA 없이도 다양한 회귀 벤치마크에서 SOTA 결과를 달성했습니다. 예를 들어, RankUp은 UTKFace 데이터셋에서 50개의 레이블 샘플을 사용하여 MAE에서 최소 13% 개선을 보여주었으며, RDA 통합 시 추가적인 6% 개선을 기록했습니다.



### Vision Paper: Designing Graph Neural Networks in Compliance with the European Artificial Intelligence Ac (https://arxiv.org/abs/2410.22120)
- **What's New**: 유럽연합의 인공지능법(AI Act)은 인공지능(Artificial Intelligence) 및 머신러닝(Machine Learning) 시스템의 개발 및 감독을 위한 포괄적인 지침을 제시합니다. 본 논문은 복잡한 그래프 구조 데이터에서 운영되는 그래프 신경망(Graph Neural Networks, GNNs)에 대한 AI Act의 독특한 도전 과제를 다룹니다.

- **Technical Details**: AI Act는 데이터 관리(data management), 데이터 거버넌스(data governance), 견고성(robustness), 인간 감독(human oversight), 개인정보 보호(privacy)를 포함한 여러 요구사항을 제시합니다. 이 논문은 이러한 요구사항이 GNN 학습(training) 및 준수를 보장하기 위한 방법론에 미치는 영향을 탐구합니다.

- **Performance Highlights**: 편향(bias), 견고성(robustness), 설명 가능성(explainability), 개인정보 보호(privacy)와 같은 GNN의 측면에 대한 심층 분석을 제공합니다. 공정한 샘플링 전략(fair sampling strategies)과 효과적인 해석 기술(interpretability techniques)의 필요성을 강조하며, 새로운 법적 체제 하에서 GNN에 대한 구체적인 지침을 제공하고, 향후 연구 방향을 제시합니다.



### Policy Gradient for Robust Markov Decision Processes (https://arxiv.org/abs/2410.22114)
- **What's New**: 이 논문은 강건한 마르코프 결정 과정(Robust Markov Decision Processes, RMDPs)을 해결하기 위한 새로운 정책 그래디언트 방법인 DRPMD(Double-Loop Robust Policy Mirror Descent)를 개발하였습니다. 이는 기존의 정책 그래디언트 방법들이 모델 불확실성에 적응하기 어려운 문제를 해결하는 데 기여하는 내용을 다루고 있습니다.

- **Technical Details**: DRPMD는 각 반복마다 적응형 허용오차(adaptive tolerance)를 사용하여 정책 최적화를 위한 일반적인 미러 강하 업데이트 규칙을 적용합니다. 이는 전역 최적 정책(global optimal policy)으로의 수렴을 보장하며, 직접 및 소프트맥스 파라미터화에 대한 새로운 수렴 결과를 포함한 포괄적인 분석을 제공합니다. 또한 RMDP의 신속한 수렴 속도를 보장합니다.

- **Performance Highlights**: 실험 결과 DRPMD는 다양한 도전적인 RMDP 설정에서 강건성과 전역 수렴성을 검증하며, 정책 그라디언트 방법 중에서 최고의 반복 복잡도(iteration complexity)를 기록하고 있습니다.



### Where Do Large Learning Rates Lead Us? (https://arxiv.org/abs/2410.22113)
Comments:
          Published in NeurIPS 2024. First three authors contributed equally, last two authors share senior authorship

- **What's New**: 본 연구에서는 대규모 학습률(Learning Rate, LR)로 신경망 트레이닝을 시작하는 것이 일반화(Generalization)를 향상시키는 원리를 실험적으로 조사했습니다. 특히, 최적 품질을 얻기 위해 요구되는 초기 LR의 크기와 다양한 LR로 훈련한 모델 간의 주요 차이점을 밝혔습니다.

- **Technical Details**: 연구에서는 처음에 너무 낮거나 너무 높은 LR으로 훈련하는 경우의 결과를 비교했습니다. 최적 결과를 얻기 위해서는 수렴(threshold) 이상의 좁은 범위의 초기 LR 값을 사용해야 하며, 이 범위를 'subregime 2A'로 정의했습니다. 최적의 결과를 얻기 위해 작은 LR로 미세 조정(fine-tuning)하는 것이 효과적입니다.

- **Performance Highlights**: 실험 결과, subregime 2A에서 훈련한 모델이 높은 품질의 국소 최소(minima)에 도달하며, 학습한 특성(feature)들도 더 적고 해당 작업에 가장 관련이 높은 것으로 나타났습니다. 그러나 너무 낮은 LR로 시작한 경우 불안정한 최소에 머물면서 모든 특성을 동시에 학습하고, 너무 높은 LR로 시작한 경우 유용한 패턴을 감지하지 못하는 경향을 보였습니다.



### Data Generation for Hardware-Friendly Post-Training Quantization (https://arxiv.org/abs/2410.22110)
- **What's New**: 본 논문에서는 Zero-shot quantization (ZSQ)을 위한 새로운 데이터 생성 방법인 Data Generation for Hardware-friendly quantization (DGH)를 제안합니다. DGH는 기존 데이터 생성 방법의 주요 결점을 해결하며, 모든 생성된 이미지를 동시에 최적화하여 양질의 이미지를 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: DGH는 모든 생성 이미지에 대해 batch normalization (BN) 통계를 계산함으로써 이전 생성 문제를 해결합니다. 또한, 데이터 증강(data augmentation) 프로세스를 모방하는 전처리 단계를 포함하여 자연 이미지의 특성을 더합니다. 마지막으로, output distribution-stretching loss를 도입하여 실제 데이터와 합성 데이터 간의 feature map 분포를 일치시킵니다.

- **Performance Highlights**: DGH는 이미지넷(ImageNet-1k) 분류 및 COCO 객체 감지에서 30%의 정확도 향상을 보여주며, 기존의 최첨단 방법들과 동일한 수준의 성능을 냅니다. 이 연구는 Sony의 오픈 소스 Model-Compression-Toolkit 라이브러리에서도 확인할 수 있습니다.



### InLINE: Inner-Layer Information Exchange for Multi-task Learning on Heterogeneous Graphs (https://arxiv.org/abs/2410.22089)
- **What's New**: 이 논문에서는 이종 그래프에서의 다중 작업 학습(Multi-Task Learning, MTL)에 의한 부정적 전이(negative transfer) 문제를 다룹니다. 이를 해결하기 위해 제안된 Inner-Layer Information Exchange (InLINE) 모델은 각 그래프 레이어 내에서 세밀한 정보 교환을 촉진합니다.

- **Technical Details**: InLINE 모델은 각 레이어의 구조를 분리하여 정보의 중요성을 조정하는 Structure Disentangled Experts 모듈과, 각 작업에 맞는 정보를 할당하는 Gate 모듈로 구성됩니다. 이 접근 방식은 기존의 외부 임베딩 교환 방식보다 정보를 보다 정교하게 처리할 수 있습니다.

- **Performance Highlights**: 두 개의 공공 데이터셋(DBLP)과 대규모 산업 데이터셋에 대한 평가에서 InLINE 모델은 기존 SoA 방법에 비해 DBLP 데이터셋에서 Macro F1 점수를 6.3% 향상시키고, 산업 데이터셋에서 AUC 점수를 3.6% 개선하는 성능을 보여주었습니다.



### Unlearning as multi-task optimization: A normalized gradient difference approach with an adaptive learning ra (https://arxiv.org/abs/2410.22086)
- **What's New**: 이 논문에서는 기계 잊기를 최적화 관점에서 조사하고, 망각(memorization) 목표 최적화와 모델 성능을 최적화하는 것을 다중 작업 최적화(multi-task optimization) 문제로 공식화합니다. 특히, 우리는 목표 간의 균형 조절을 향상시키기 위한 새로운 정규화된 그래디언트 차이(NGDiff) 알고리즘을 도입했습니다.

- **Technical Details**: 기계 잊기를 다중 작업 최적화 문제로 공식화하여 두 가지 목표를 동시에 최소화 및 최대화하는 방법을론의 명확화와 NGDiff 알고리즘을 통해 제시합니다. 이 알고리즘은 전달(set)과 망각(forgetting) 작업 간의 균형을 동적으로 조절하며, Hessian 기반 학습률 선택을 통합하여 안정적인 수렴(convergence)을 이룰 수 있습니다.

- **Performance Highlights**: NGDiff는 TOFU 및 MUSE 데이터셋에서 최첨단(unlearning) 방법들보다 뛰어난 성능을 보여주며, Llama2-7B 모델과 비교했을 때 모델 유틸리티를 40% 향상시켰습니다. 또한 안정적인 학습을 유지하는 특징이 있습니다.



### Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks (https://arxiv.org/abs/2410.22069)
- **What's New**: 이 논문은 심층 동질 신경망(deep homogeneous neural networks)에서의 여러 최적화 알고리즘, 특히 steepest descent(최대 기울기 하강) 알고리즘의 암묵적 편향(implicit bias)을 연구합니다. 최근 교육 정확도가 완벽해지면 네트워크의 구조적인 기하학적 여유(geometric margin)가 증가하기 시작한다는 것을 증명했습니다.

- **Technical Details**: 우리는 최적화 문제에 대한 일반화된 정지상태(stationarity) 개념을 정의하고 알고리즘이 (일반화된) Bregman divergence를 점진적으로 축소하는 과정을 보여줍니다. 이는 여유 최대화(margin-maximization) 문제의 정지점에 대한 접근성을 정량화합니다. 다양한 steepest descent 알고리즘으로 최적화된 신경망의 궤적(trajectories)을 실험적으로 분석합니다.

- **Performance Highlights**: 특히, Adam 알고리즘과의 연관성을 강조하며, 최적화 알고리즘의 후반부 편향(late-stage bias)을 특징화합니다. 이 논문은 각 알고리즘이 여유 극대화 문제를 해결하는 방식에 대한 통찰을 제공합니다.



### Enhance Hyperbolic Representation Learning via Second-order Pooling (https://arxiv.org/abs/2410.22026)
- **What's New**: 이 논문에서는 hyperbolic representation learning에서 계층적 정보의 캡처를 개선하기 위해 second-order pooling을 도입합니다. 이는 입력 특성의 일반화 능력을 해치지 않으면서 샘플 간의 거리를 자연스럽게 증가시키는데 중점을 두었습니다.

- **Technical Details**: 본 연구는 hyperbolic space에서의 feature projection을 통해 전통적인 backbone이 계층 구조 정보를 효과적으로 캡처하도록 하는 방법을 제안합니다. 특히, low-dimensional bilinear pooling 기법이 hyperbolic representation learning에 효과적으로 적용되지 않는 이유와 이를 해결하기 위한 kernel approximation regularization을 도입하여 저차원 bilinear features가 저차원 공간에서 kernel function를 잘 근사화할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 그래프 구조 데이터셋에서 효과적임을 보여줍니다. 특히, 제안된 second-order pooling이 backbone의 Lipschitz 상수를 낮추어 일반화 능력을 향상시킴을 입증하였습니다.



### On the Robustness of Adversarial Training Against Uncertainty Attacks (https://arxiv.org/abs/2410.21952)
- **What's New**: 이번 연구에서는 불확실성 공격에 대항하는 효율적인 방어 전략을 제시합니다. 특히, Adversarial Training (AT)을 통해 비정상적인 입력에 대한 모델의 예측 정확성을 확인합니다. AT는 공격에 대한 저항력을 키우면서, 동시에 불확실성에 대한 신뢰할 수 있는 추정치를 보장합니다.

- **Technical Details**: 연구는 Uncertainty Quantification (UQ) 기법과 Adversarial Machine Learning (ML) 기술을 결합하여 불확실성이 높은 상황에서도 안전한 결정을 돕는 모델을 제안합니다. 이를 위해, 다수의 공개 벤치마크인 RobustBench에서의 다양한 모델을 평가하여 AT를 통한 방어 기능의 성공을 입증했습니다.

- **Performance Highlights**: CIFAR-10 및 ImageNet 데이터셋에서 다양한 adversarial-robust 모델의 실험을 통해, 공격 시나리오에서의 성능이 크게 향상된 것을 보여주었습니다. 방어 모델은 불확실성 공격에도 효과적으로 대응함을 입증했습니다.



### Human-Readable Programs as Actors of Reinforcement Learning Agents Using Critic-Moderated Evolution (https://arxiv.org/abs/2410.21940)
Comments:
          Accepted in BNAIC/BeNeLearn 2024 conference proceedings

- **What's New**: 이 논문은 Deep Reinforcement Learning (DRL)의 투명성과 설명 가능성의 부족 문제를 해결하기 위한 새로운 접근법을 제안합니다. 기존의 Programmatic Reinforcement Learning (PRL) 방법이 정책을 코드 형태로 증류하는데 집중하는 반면, 저자들은 RL 에이전트의 정책으로 프로그램을 직접 학습하는 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 TD3(Twin Delayed Deep Deterministic Policy Gradient) 알고리즘을 기반으로 하며, 유전 알고리즘(genetic algorithm)에 의한 프로그램 합성을 위한 목표 함수로 TD3의 critic을 사용합니다. 이 접근법은 학습 단계에서 프로그램을 생성함으로써 평균 제곱 오차(Mean Squared Error) 대신 실제 높은 보상(high reward)을 추구하게 됩니다.

- **Performance Highlights**: 실험은 제안한 방법이 간단한 Gridworld 환경에서 우수한 샘플 효율성(sample efficiency), 정책 품질(policy quality), 그리고 설명 가능성(explainability)을 보임을 입증하였습니다.



### Evaluating K-Fold Cross Validation for Transformer Based Symbolic Regression Models (https://arxiv.org/abs/2410.21896)
- **What's New**: 본 연구에서는 Transformer 기반의 Symbolic Regression 모델이 작은 데이터셋(15,000 데이터 포인트)을 활용하여 훈련되었을 때의 K-fold cross-validation을 적용하여 모델의 일반화 가능성 및 과적합 문제를 완화하는 방법을 제안합니다.

- **Technical Details**: K-Fold Cross-Validation (KFCV)을 통해 5개의 fold로 데이터셋을 나누어 각 fold마다 모델을 훈련하고 검증하여 평균 훈련 손실과 검증 손실을 산출했습니다. 이 모델의 경우, 기존 500,000 포인트 데이터셋에서 97% 줄어든 15,000 포인트로 훈련되었습니다.

- **Performance Highlights**: KFCV 기법을 적용한 결과, 모델의 검증 손실이 53.31% 개선되었으며, 이는 모델의 출력 일관성과 일반화를 현저히 향상시켰음을 보여줍니다.



### Bayesian Optimization for Hyperparameters Tuning in Neural Networks (https://arxiv.org/abs/2410.21886)
Comments:
          Bachelor Thesis in Optimization for Machine Learning, 57 pages

- **What's New**: 이번 연구에서는 이미지 분류 작업을 위한 Convolutional Neural Networks (CNN)의 하이퍼파라미터 튜닝에 Bayesian Optimization (BO)을 적용하고 그 효율성을 조사합니다.

- **Technical Details**: Bayesian Optimization은 연속 입력과 제한된 평가 예산이 있는 복잡한 블랙박스 함수에 적합한 도함수 비사용 글로벌 최적화 방법입니다. BO 알고리즘은 Gaussian Process 회귀 및 Upper Confidence Bound (UCB), Expected Improvement (EI)과 같은 수집 함수를 활용하여 최적 구성을 효과적으로 식별합니다.

- **Performance Highlights**: Ax와 BOTorch 프레임워크를 사용한 실험 결과, BO는 하이퍼파라미터 튜닝 시도 횟수를 줄이면서도 경쟁력 있는 모델 성능을 달성하는 효율성을 보여주었습니다. BO는 탐색과 착취의 균형을 효과적으로 맞추어 CNN 아키텍처에 대한 최적 설정으로 신속하게 수렴합니다.



### SCGNet-Stacked Convolution with Gated Recurrent Unit Network for Cyber Network Intrusion Detection and Intrusion Type Classification (https://arxiv.org/abs/2410.21873)
- **What's New**: 본 연구에서는 사이버 네트워크 침입 탐지(IMDS)를 위한 새로운 딥러닝 아키텍처 SCGNet(스택드 컨볼루션과 게이티드 리커런트 유닛 네트워크)을 제안합니다. SCGNet은 NSL-KDD 데이터셋에서 99.76%와 98.92%의 정확도로 네트워크 공격 탐지 및 공격 유형 분류에서 우수한 결과를 보였습니다.

- **Technical Details**: SCGNet은 스택드 컨볼루션(또는 Stacked Convolution)과 GRU(게이티드 리커런트 유닛, Gated Recurrent Unit) 네트워크 구조를 결합한 새로운 딥러닝 구조입니다. 이 모델은 전통적인 머신러닝 기법과 함께 데이터 처리 파이프라인을 실험하여 성능을 평가하였으며, 일반적인 데이터 전처리(Preprocessing) 파이프라인을 소개하였습니다.

- **Performance Highlights**: 본 연구의 SCGNet은 네트워크 공격 탐지와 공격 유형 분류에서 각각 99.76%와 98.92%의 정확도를 달성하여 기존 기술보다 우수한 성능을 보여줍니다. 또한, 전통적인 머신러닝 기법들과 비교하여도 효율적인 데이터 전처리를 통해 성능 향상을 입증함으로써 다양한 데이터셋에 적용할 수 있는 가능성을 확인하였습니다.



### Cross-Entropy Is All You Need To Invert the Data Generating Process (https://arxiv.org/abs/2410.21869)
- **What's New**: 이번 연구는 지도 학습(supervised learning)의 효과성을 설명하는 포괄적인 이론을 제시하기 위해, 비선형 독립 성분 분석(nonlinear Independent Component Analysis) 기법을 활용하여 잠재 구조(latent structures)를 회복(establish)하는 방법을 다룹니다.

- **Technical Details**: 연구진은 매개변수 인스턴스 차별화(parametric instance discrimination)를 통해 식별 가능성(identifiability) 결과를 확장하고, 교차 엔트로피(minimization)로 최적화를 하는 지도 학습 환경에서 어떻게 통찰(insights)을 적용하는지를 보여줍니다. 이들은 표준 분류 작업(standard classification tasks)에서도 모델들이 진실(factors of variation)의 표현(representations)을 선형 변환(linear transformation)까지 배우는 것을 증명하였습니다.

- **Performance Highlights**: 시뮬레이션 데이터(시뮬레이트된 데이터) 및 DisLib라는 대표적인 디스럽란틀 벤치마크(disentanglement benchmark)에서 이론적 가정을 충족하는 결과를 보이며, 단순한 분류 작업이 잠재 구조를 회복할 수 있음을 입증했습니다. 또한, ImageNet에서 훈련된 모델이 변형(factors of variation)의 선형 디코딩을 허용하는 표현을 인코딩(encoding)함을 밝혔습니다.



### Learning Infinitesimal Generators of Continuous Symmetries from Data (https://arxiv.org/abs/2410.21853)
Comments:
          Neurips 2024

- **What's New**: 이 논문은 데이터에서 연속적인 대칭을 학습하기 위한 새로운 알고리즘을 제안합니다. 기존의 방법들이 사전 정의된 Lie 군에 의존하는 반면, 본 연구에서는 비선형 생성기를 포함하여 다양한 대칭을 파라미터화하고 학습할 수 있도록 합니다.

- **Technical Details**: 연속 대칭을 학습하기 위해, 본 논문에서는 Neural Ordinary Differential Equation (Neural ODE)를 모델링하여 학습 가능한 infinitesimal generator를 설정합니다. 이 방법은 주어진 작업에 따라 정의된 기준에 대한 불변성을 위반하는 정도를 측정하기 위해 validity score 함수를 설계합니다. 이 함수는 풀리 미분 가능하며, end-to-end로 gradient descent를 통해 대칭을 학습할 수 있도록 합니다.

- **Performance Highlights**: CIFAR-10 분류 데이터셋과 편미분 방정식(PDE)에서 실험을 통해, 제안한 방법이 약한 대칭과 비선형 대칭을 효과적으로 발견하는 것을 보여주었습니다. 또한, 학습된 생성기를 이용하여 자동 증강 기법을 개발함으로써, 전통적인 변환과 비교하여 경쟁력 있는 성과를 나타냄을 입증했습니다.



### Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Models (https://arxiv.org/abs/2410.21815)
- **What's New**: 본 논문에서는 self-interpretable 모델과 black-box 모델을 위한 post-hoc 설명 사이의 간극을 메우기 위한 새로운 방법인 *AutoGnothi*를 제안합니다. 이 방법은 블랙박스 모델이 예측 정확성을 손상시키지 않으면서 자가 해석 가능성을 이론적으로 보장하는 것을 목표로 합니다.

- **Technical Details**: *AutoGnothi*는 parameter-efficient transfer learning (PETL)을 활용하여 블랙박스 모델에 작은 사이드 네트워크를 통합합니다. 이를 통해 Shapley value 기반의 설명을 생성할 수 있으며, 원래 네트워크의 매개변수를 변경하지 않고도 메모리, 훈련 및 추론 비용을 크게 줄일 수 있습니다. 이 방법은 두 개의 모델로 구성된 기존 방식과 달리 예측과 설명을 동시에 수행하며, 단일 추론 단계에서 작업을 완료합니다.

- **Performance Highlights**: 실험 결과 *AutoGnothi*는 매우 높은 훈련 및 추론 효율성을 보여 주었습니다. 예를 들어, ViT-base 모델에 대해 97%의 훈련 가능 매개변수 감소, 72%의 훈련 메모리 감소를 기록하며, 설명 생성에 있어서 54%의 추론 비용 감소와 44%의 추론 시간 감소를 달성했습니다. 이러한 결과는 *AutoGnothi*가 블랙박스 모델의 예측 성능을 유지하면서 자기 해석 가능성을 강화한다는 것을 일증합니다.



### Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging (https://arxiv.org/abs/2410.21804)
- **What's New**: 이번 연구에서는 Weight-Ensembling Mixture of Experts (WEMoE) 방법을 제안하여 멀티태스크 모델 병합을 동적으로 수행합니다. 이 방법은 Transformer 기반 모델의 중요 모듈을 식별하고, 비중요 모듈은 정적 병합하고 중요 모듈은 Mixture-of-Experts (MoE) 구조로 전환합니다.

- **Technical Details**: WEMoE는 비중요 모듈을 정적으로 병합하고, 중요 모듈은 입력 샘플에 기반하여 MoE 전문가 모듈의 가중치를 동적으로 할당하는 방식으로 작동합니다. 특히, MLP 모듈의 매개변수 변화를 분석하여 전문 지식의 전이와 태스크 간의 충돌을 줄이는 데 중점을 둡니다. 이어서 E-WEMoE를 소개하여 비핵심 요소를 제거하고 모듈 간 라우팅을 공유함으로써 전체 매개변수 수와 계산 오버헤드를 줄입니다.

- **Performance Highlights**: WEMoE와 E-WEMoE는 다양한 아키텍처와 태스크에서 SOTA 성능을 달성하여 멀티태스크 학습 성능, 일반화 능력 및 강건성을 향상시켰습니다. 실험 결과, 두 방법 모두 많은 태스크에서 독립적으로 조정된 모델과 유사한 성능을 보여주었습니다.



### Online Mirror Descent for Tchebycheff Scalarization in Multi-Objective Optimization (https://arxiv.org/abs/2410.21764)
Comments:
          27 pages, 7 figures, 2 tables

- **What's New**: 이번 논문은 Tchebycheff scalarization을 기반으로 하는 새로운 온라인 미러 하강 알고리즘인 OMD-TCH를 제안합니다. 이 알고리즘은 여러 목표를 최적화하려는 다목표 최적화(Multi-Objective Optimization, MOO)의 단점을 보완합니다. 특히, 기존의 linear scalarization 방식의 한계를 극복하고는 OMD-TCH의 수렴 속도를 보여줍니다.

- **Technical Details**: OMD-TCH는 확률적 최적화 문제에 대한 효율적인 솔루션을 제공합니다. 이 알고리즘은 Tchebycheff scalarization을 활용하여 최악의 목표를 최적화합니다. OMD-TCH는 $O(\sqrt{\log m / T})$의 수렴 속도를 가지고 있으며, 여기서 m은 목표의 수, T는 반복 횟수를 의미합니다. 또한, 새로운 적응형 온라인-배치 변환 방식인 AdaOMD-TCH를 제안하여 성능을 개선합니다.

- **Performance Highlights**: OMD-TCH와 AdaOMD-TCH는 합성 문제와 페더레이티드 러닝(Federated Learning) 작업에서 공정성 제약을 만족시키면서 최첨단의 성능을 보여주었습니다. AdaOMD-TCH는 OMD-TCH에 비해 정확도와 공정성 지표 모두에서 현저한 성능 향상을 이루었으며, 기존의 공정 페더레이티드 러닝 방법들과 유사한 성능을 달성했습니다.



### Reliable and Compact Graph Fine-tuning via GraphSparse Prompting (https://arxiv.org/abs/2410.21749)
- **What's New**: 최근 그래프 프롬프트 학습(Graph Prompt Learning)에 대한 관심이 높아지고 있으며, 이는 사전 학습된 그래프 신경망(GNN) 모델을 다운스트림 그래프 학습 작업에 적응시키는 데 기여하고 있습니다. 기존 방법들은 일반적으로 모든 그래프 요소에 대해 프롬프트를 수행하는데, 이는 비효율적이고 중복성이 있습니다. 이를 해결하기 위해, 우리는 희소 표현 이론(Sparse Representation Theory)을 활용하여 최적의 프롬프트 요소를 선택하는 Graph Sparse Prompting(GSP) 접근 방식을 제안합니다.

- **Technical Details**: GSP는 특정 노드 속성과 같은 최적의 요소를 능동적으로 선택하여 다운스트림 작업을 위한 간결한 프롬프트를 달성하는 것을 목표로 합니다. 두 가지 GSP 모델인 Graph Sparse Feature Prompting(GSFP)와 Graph Sparse multi-Feature Prompting(GSmFP)를 제안하며, 이들은 사전 학습된 GNN을 조정할 수 있는 일반적인 프레임워크를 제공합니다. 이러한 방식은 추가적인 학습 가능 파라미터가 필요하지 않습니다.

- **Performance Highlights**: 16개의 널리 사용되는 벤치마크 데이터셋에 대한 실험 결과는 제안된 GSFP와 GSmFP의 효과성과 장점을 검증합니다. 이 방법들은 그래프 학습 작업에서 라벨 데이터에 대한 의존도를 줄이고, 더욱 효율적인 학습을 가능하게 합니다.



### A Dual Adaptive Assignment Approach for Robust Graph-Based Clustering (https://arxiv.org/abs/2410.21745)
- **What's New**: 이번 논문에서는 RDSA(Robust Deep Graph Clustering via Dual Soft Assignment)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 그래프 기반 클러스터링의 강인성을 향상시키기 위해 두 가지 종류의 soft assignment를 활용합니다.

- **Technical Details**: RDSA는 다음의 세 가지 주요 구성 요소로 이루어져 있습니다: (i) 노드 임베딩 모듈: 그래프의 구조적 특성과 노드 속성을 효과적으로 통합합니다. (ii) 구조 기반 soft assignment 모듈: 노드 할당을 위한 affinity matrix를 이용하여 그래프의 모듈러리티(modularity)를 개선합니다. (iii) 노드 기반 soft assignment 모듈: 커뮤니티 랜드마크를 식별하고 노드 할당을 정제하여 모델의 강인성을 향상시킵니다.

- **Performance Highlights**: 다양한 실제 데이터셋에서 RDSA의 성능을 평가한 결과, 기존의 최첨단 방법과 비교해 뛰어난 성능을 보여주었습니다. RDSA는 클러스터링 효과성 및 강인성, 노이즈에 대한 적응성, 훈련 안정성, 그리고 대규모 데이터셋에 대한 확장성을 포함하여 뛰어난 클러스터링 성능을 제공합니다.



### Generating Realistic Tabular Data with Large Language Models (https://arxiv.org/abs/2410.21717)
Comments:
          To appear at ICDM 2024

- **What's New**: 이 논문에서는 tabular data (표 형식 데이터) 생성을 위한 새로운 LLM 기반의 방법을 제안합니다. 기존의 방법들이 특정 feature와 target variable 간의 올바른 상관관계를 포착하지 못하는 문제를 해결하기 위해 세 가지 중요한 개선 방안을 도입하였습니다.

- **Technical Details**: 제안된 방법은 fine-tuning(미세 조정) 단계에서의 새로운 permutation strategy, feature-conditional sampling(특징 조건 샘플링) 접근 방식, 그리고 생성된 샘플을 바탕으로 한 라벨 생성 방식을 포함합니다. 이러한 개선은 LLM이 실제 데이터에서의 feature-class correlation(특징-클래스 상관관계)을 올바르게 포착할 수 있도록 도와줍니다.

- **Performance Highlights**: 제안된 Pred-LLM 방법은 20개의 데이터셋에서 10개의 기존 SOTA (State-Of-The-Art) 기법을 초월하며, 품질과 다양성 면에서 매우 현실적인 합성 샘플을 생성합니다. 더 나아가, 합성 데이터로 훈련된 분류기는 원본 데이터로 훈련된 분류기와 비교해도 경쟁력을 가지는 성과를 보여, tabular data generation(표 데이터 생성)에서 중요한 진전을 이룹니다.



### Sliced-Wasserstein-based Anomaly Detection and Open Dataset for Localized Critical Peak Rebates (https://arxiv.org/abs/2410.21712)
- **What's New**: 본 논문에서는 sliced-Wasserstein metric을 이용한 새로운 비지도적 이상 탐지 방법을 제안합니다. 이 필터링 기술은 에너지와 같은 중요 분야에서 신뢰할 수 있는 머신러닝 모델을 배포하는 MLOps 파이프라인 통합에 흥미로운 개념적 가치가 있습니다. 또한, 북부 기후의 지역화된 주요 피크 리베이트 수요 반응을 보여주는 첫 번째 데이터셋도 공개합니다.

- **Technical Details**: 우리는, localization된 CPR(critical peak rebates) 프로그램에 참여하는 고객을 위한 2년의 집계 소비 데이터를 공개합니다. 새로운 단순한 비지도적 이상 필터를 제안하며, 이 필터는 Sliced-Wasserstein 거리(Sliced-Wasserstein distance)를 활용합니다. 이러한 필터는 표준 이상 탐지 데이터셋에서 성능을 검증하고, 지역화된 CPR 참가자의 에너지 소비 예측에 대한 벤치마크를 제시하는 데 사용됩니다.

- **Performance Highlights**: 우리는 synthesized datasets와 표준 이상 탐지 데이터셋에서 이 방법의 성능을 입증하며, 향후 연구에 신뢰할 수 있는 머신러닝 모델을 위한 기반을 제공합니다. 이 새로운 이상 탐지 방법은 기존의 여러 비지도적 AD 방법들과 비교할 때 더 높은 성과를 낼 것으로 기대됩니다.



### Multi-view clustering integrating anchor attribute and structural information (https://arxiv.org/abs/2410.21711)
Comments:
          18 pages, 7 figures

- **What's New**: 이 논문에서는 전통적인 데이터 클러스터링 알고리즘의 한계를 극복하기 위해, 조리된 (directed) 구조를 포함한 새로운 다중 관점 클러스터링 알고리즘 AAS를 소개합니다.

- **Technical Details**: AAS는 각 관점(view)에서 앵커(anchors)를 통해 두 단계의 근접성 접근 방식을 사용하여, 속성(attribute) 정보와 조리된 구조 정보를 통합합니다. 이 방식은 유사성 행렬(similarity matrices)에서 카테고리 특성의 명확성을 개선합니다. 앵커 구조 유사성 행렬은 조리 그래프의 강한 연결 요소(strongly connected components)를 활용합니다. 이 전체 과정은 유사성 행렬의 구축에서 클러스터링까지 통합 최적화 프레임워크(unified optimization framework)로 구성됩니다.

- **Performance Highlights**: 수정된 Attribute SBM 데이터 세트를 기반으로 8개의 알고리즘과의 비교 실험에서 AAS의 효과성과 우수성이 입증되었습니다.



### Stochastic Approximation with Unbounded Markovian Noise: A General-Purpose Theorem (https://arxiv.org/abs/2410.21704)
- **What's New**: 이번 연구에서는 네트워크의 자원 배분 및 재고 시스템과 같은 공학적 응용을 동기로 삼아, 무한 상태 공간 및 보상 함수가 있는 평균 보상 강화 학습(Average-reward Reinforcement Learning)을 탐구했습니다. 최근의 연구들은 액터-크리틱(actor-critic) 프레임워크에서 이 문제를 다루었고, 특정 오류 보증을 갖춘 크리틱에 접근할 때의 유한 샘플 경계를 수립했습니다.

- **Technical Details**: Temporal Difference (TD) 학습과 선형 함수 근사를 사용하여, 최적 $	ext{O}(1/	ext{ε}^2)$ 샘플 복잡도를 갖는 유한 시간 경계를 수립합니다. 이러한 결과는 특정 변동 조건을 갖는 비선형 확률 근사(Stochastic Approximation, SA)를 위한 일반 목적 정리를 통해 도출되었습니다. 어떤 비선형 SA에 대한 Lyapunov 함수를 구성할 경우, 적합한 조건에서 무한형 마르코프 노이즈에 의해 구동될 때 유한 시간 경계를 수립할 수 있습니다.

- **Performance Highlights**: 우리는 두 가지 시스템을 연구하여 우리의 정리의 강력함을 입증하였습니다: (i) $Q$-learning의 유한 시간 경계를 개선하여 오류 경계를 조정하고 더 넓은 행동 정책 클래스에 대해 허용했습니다. (ii) 고차원 매끄럽고 강하게 볼록한 함수에 대한 분산 확률 최적화(distributed stochastic optimization)에 대한 최초의 유한 시간 경계를 수립했습니다.



### On the Role of Depth and Looping for In-Context Learning with Task Diversity (https://arxiv.org/abs/2410.21698)
- **What's New**: 이 연구는 Transformer 모델의 in-context learning 능력에 대한 새로운 시각을 제시하며, 다 task-linear regression 문제에 대해 Transformer가 얼마나 잘 학습할 수 있는지를 규명합니다. 연구는 특히 각기 다른 데이터 분포를 처리하는 Transformer의 깊이(depth)가 중요한 역할을 한다고 강조합니다.

- **Technical Details**: 이 논문은 다양한 task를 위한 in-context linear regression을 다루며, covariance matrix의 고유값이 [1, \kappa]의 범위를 가질 때, Transformer의 깊이가 얼마나 필요한지를 이론적으로 규명합니다. 이 연구는 특히 multilayer Transformer와 Looped Transformer의 성능의 기초를 다집니다. Looped Transformer는 서로 다른 레이어 간 weight sharing을 통해 robust한 generalization을 보여줍니다.

- **Performance Highlights**: 멀티 레이어 Transformer는 task-diverse linear regression에서 log(\kappa) 또는 √(\kappa)의 lower bound를 요구하며, 이는 이 모델이 특정한 데이터 분포에서도 높은 표현력을 갖추게 함을 나타냅니다. 반면에, Looped Transformer는 뛰어난 out-of-distribution generalization 능력을 갖추고 있으며, 이로 인해 집중적으로 다루어야 할 필요가 있는 robust한 일반화 문제를 해결할 수 있습니다.



### Pushing the Limits of All-Atom Geometric Graph Neural Networks: Pre-Training, Scaling and Zero-Shot Transfer (https://arxiv.org/abs/2410.21683)
Comments:
          15 pages, 4 figures, supporting information appended

- **What's New**: 이 연구는 사전 학습된 Geom-GNNs를 활용하여 분자 구조 표현에 있어 성공적인 전이 학습(transferrable learning)을 가능하게 했으며, OOD(Out-of-Distribution) 시나리오에서도 효과적인 특징 추출을 입증했습니다.

- **Technical Details**: Geom-GNN(Geometric Graph Neural Networks)의 사전 학습은 자가 감독(self-supervised) 및 감독(supervised) 학습 설정에서 성능 스케일링 행동을 조사하여, 다양한 아키텍처의 표현 능력이 다르게 나타난다는 것을 발견했습니다. 또한, Geom-GNNs는 전통적인 파워-로우 스케일링(power-law scaling)을 따르지 않으며, 고품질 레이블 데이터를 필요로 한다는 점을 강조하였습니다.

- **Performance Highlights**: 사전 학습된 그래프 임베딩(embeddings)은 효율적으로 분자 구성(conformation)을 표현하며, 전통적인 로타머(rotamer) 특징보다도 뛰어난 표현력을 보여주어 복잡한 펩타이드(peptide)와 단백질(protein) 시스템에 대해 최대 50%의 성능 향상을 달성했습니다.



### How Does Critical Batch Size Scale in Pre-training? (https://arxiv.org/abs/2410.21676)
- **What's New**: 이 논문은 대규모 모델을 훈련할 때의 병렬 처리 전략(Parallelism strategies)의 효율성을 분석하며, critical batch size(CBS)라는 새로운 지표를 제안합니다.

- **Technical Details**: 저자들은 CBS를 측정하기 위해 8500만에서 12억 개의 매개변수를 가지는 일련의 자동 회귀 모델(autogressive models)을 C4 데이터셋에서 사전 훈련(pre-train)했습니다. 배치 크기(batch size), 모멘텀(momentum), 학습률(learning rate) 및 스케줄링(scheduling)과 같은 하이퍼파라미터(hyper-parameter)를 조정하여 스케일(scale)이 CBS에 미치는 영향을 체계적으로 조사했습니다.

- **Performance Highlights**: 결과는 CBS가 모델 크기보다 데이터 크기에 주로 비례하여 증가한다는 것을 보여줍니다. 이 발견은 신경망(Neural networks)의 무한 너비 한계(infinite-width limits)와 무한 차원 최소 제곱 회귀(infinite-dimensional least squares regression)에 대한 이론적 분석을 통해 정당화됩니다.



### Sequential choice in ordered bundles (https://arxiv.org/abs/2410.21670)
- **What's New**: 본 논문에서는 경험 재화(Experience Goods) 구성을 위해 고안된 여러 예측 모델을 사용하여, 순서가 정해진 번들에서 개인이 다음 항목을 소비할 가능성을 예측하는 방법을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 커스텀 Transformers(디코더 전용 및 인코더-디코더 아키텍처), 미세 조정된 GPT-3, 커스텀 LSTM 모델, 강화 학습 모델, 두 개의 Markov 모델 그리고 제로 순서 모델을 평가합니다. 주목할 점은 Transformers의 주의(attention) 가중치를 분석하여 소비자가 이전 선택을 기반으로 다음 항목을 고를 때의 패턴을 보여줍니다.

- **Performance Highlights**: Spotify 데이터 분석 결과, 디코더 전용 아키텍처를 가진 커스텀 Transformer가 개별 선택과 총 수요 예측에서 가장 높은 정확도를 달성합니다. 또한, 이 모델은 개인 맞춤형 프로모션을 통해 수요를 증가시키는 데 도움을 줄 수 있습니다.



### Minimum Entropy Coupling with Bottleneck (https://arxiv.org/abs/2410.21666)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024) - Spotlight

- **What's New**: 이 논문은 로그 손실(logarithmic loss) 기반의 새로운 lossy compression 프레임워크를 조사합니다. 이 프레임워크는 원본 분포로부터 재구성 분포가 일치하지 않는 상황에서도 적용 가능하며, 압축과 검색이 공동으로 필요한 응용 분야에 적합합니다.

- **Technical Details**: 제안된 프레임워크는 고전적인 최소 엔트로피 결합(minimum entropy coupling) 프레임워크를 확장하며, 여기서 bottleneck을 통합하여 결합에서 통계적 제어 가능성을 도입합니다. 주요 최적화 문제는 인코더를 위한 Entropy-Bounded Information Maximization (EBIM)과 디코더를 위한 Minimum Entropy Coupling (MEC)으로 분해됩니다.

- **Performance Highlights**: 실험을 통해 마르코프 코딩 게임(Markov Coding Games, MCGs)에서 MEC-B의 실제 적용 가능성을 보여주며, 다양한 압축 비율에서 MDP 보상과 수신 정확성 간의 균형을 강조합니다. 전통적인 압축 기법에 비해 제안된 방법의 효과성을 입증했습니다.



### Dimensionality-induced information loss of outliers in deep neural networks (https://arxiv.org/abs/2410.21656)
Comments:
          This preprint has not undergone peer review (when applicable) or any post-submission improvements or corrections. The Version of Record of this contribution is published in ECML PKDD 2024, and is available online at this https URL

- **What's New**: 이번 연구에서는 딥 뉴럴 네트워크(DNN)에서 OOD(out-of-distribution) 샘플과 ID(in-distribution) 샘플 간의 차이를 생성하는 방법에 대해 실험적으로 조사하고, OOD 탐지 문제를 해결하기 위한 최적의 차원 인식을 통해 높은 성능을 달성하는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 DNN에서의 OOD 샘플과 ID 샘플 간의 관계를 이해하기 위해 계층별(feature representation) 의존성을 분석합니다. 특히 DNN의 내재적 저차원화가 OOD 샘플이 ID 샘플과 더 뚜렷하게 구분되는 이유를 설명합니다. 또한, 차원 인식 OOD 탐지 방법(dimensionality-aware OOD detection method)을 제안하여 데이터셋의 편향에 대한 과도한 주의로 인해 잘못된 분류가 발생하는 원인을 규명합니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터셋에 대해 높은 성능을 지속적으로 달성하며, 더 낮은 계산 비용으로 OOD 탐지의 효용을 증명합니다.



### Faster Local Solvers for Graph Diffusion Equations (https://arxiv.org/abs/2410.21634)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문은 그래프 확산 방정식(GDEs)을 근사적으로 해결하기 위한 새로운 프레임워크를 제안합니다. 이 방법은 기존의 로컬 솔버의 비효율성을 밝혀내고, 표준 반복 솔버를 효과적으로 지역화합니다.

- **Technical Details**: 제안된 프레임워크는 로컬 확산 과정(local diffusion process)을 사용하여 GDEs를 해결합니다. 새로운 로컬 솔버는 단순하고 증명 가능한 하한 시간 알고리즘으로 설계되어 있으며, 고성능 GPU 구현에 적합하도록 매우 병렬화(parallelizable) 가능합니다.

- **Performance Highlights**: 이 프레임워크는 근사 확산 벡터를 신속하게 얻는 데 매우 효과적이며, 최대 100배의 속도 향상을 달성할 수 있습니다. 또한, 대규모 동적 그래프(dynamic graphs)에 적용할 수 있습니다.



### Graph Sparsification for Enhanced Conformal Prediction in Graph Neural Networks (https://arxiv.org/abs/2410.21618)
- **What's New**: 본 연구에서는 SparGCP라는 새로운 프레임워크를 소개하며, 이는 그래프 신경망(GNN) 훈련 과정에서의 conformal prediction을 개선하는 데 중점을 둡니다.

- **Technical Details**: SparGCP는 파라미터화된 그래프 희소화 모듈과 CP 기반 손실 함수를 통합하여 기본 GNN 모델에 적용됩니다. 이 모듈은 각 엣지가 CP 작업에 미치는 기여도를 평가하고, 낮은 점수를 받은 노이즈 엣지를 제거합니다.

- **Performance Highlights**: SparGCP는 실세계 그래프 데이터셋에서 기존 방법들과 비교했을 때 예측 집합의 크기를 평균 32% 줄이며, 일반 GPU에서 대규모 네트워크로도 원활하게 확장됩니다.



### Identifying Selections for Unsupervised Subtask Discovery (https://arxiv.org/abs/2410.21616)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문은 고수익 태스크를 해결하기 위해 태스크를 서브태스크로 분해하는 방법에 대해 논의합니다. 특히, 서브태스크가 데이터 생성 과정에서의 선택 메커니즘으로부터 도출된다는 점을 강조하며, 이를 통해 강화 학습(RL) 및 모방 학습(IL) 문제 해결에 있어 데이터 효율성을 향상시킬 수 있다는 점에 주목합니다.

- **Technical Details**: 논문에서는 서브태스크를 파악하기 위한 새로운 이론을 제안합니다. 제안된 방법은 순차적 비음수 행렬 분해(sequential non-negative matrix factorization, seq-NMF)를 통해 서브목표를 학습하고, 이를 바탕으로 의미 있는 행동 패턴을 추출합니다. 이 과정에서 선택 변수를 서브목표로 활용하여 서브태스크를 발견합니다.

- **Performance Highlights**: 다양한 Kitchen 환경에서의 실험 결과, 학습된 서브태스크가 새로운 태스크에 대한 일반화 성능을 효과적으로 향상시킴을 보여주었습니다. 이 연구는 서브태스크 발견 문제에 대한 심층적인 이해를 제공하며, 데이터 효율성을 높이는 기회를 마련합니다.



### CaloChallenge 2022: A Community Challenge for Fast Calorimeter Simulation (https://arxiv.org/abs/2410.21611)
Comments:
          204 pages, 100+ figures, 30+ tables

- **What's New**: 2022년 'Fast Calorimeter Simulation Challenge'의 결과를 발표하며, 다양한 generative 모델을 비교하고 평가하는 심층적인 조사를 진행했습니다.

- **Technical Details**: 본 연구에서는 Variational AutoEncoders (VAEs), Generative Adversarial Networks (GANs), Normalizing Flows, Diffusion models, Conditional Flow Matching 기반의 모델을 포함한 31개의 제출물에 대해 분석했습니다. 테스트 데이터는 수백 개에서 수만 개의 voxel로 다양한 차원을 가진 calorimeter shower 데이터셋에서 수집되었습니다. 평가 지표로는 1차원 히스토그램 차이, KPD/FPD 점수, binary classifier의 AUC, multi-class classifier의 log-posterior를 사용했습니다.

- **Performance Highlights**: CaloChallenge의 결과는 최신 calorimeter fast simulation 접근 방식에 대한 가장 포괄적인 설문조사를 제공하며, generative 모델 평가의 중요 문제를 독특하게 조망했습니다. 이는 다른 generative AI가 필요로 하는 분야에도 적용될 수 있습니다.



### The Limits of Transfer Reinforcement Learning with Latent Low-rank Structur (https://arxiv.org/abs/2410.21601)
- **What's New**: 본 연구에서는 latent low rank 구조를 가진 강화 학습의 전이(Transfer RL) 접근 방식을 제안합니다. 이전의 문제와 신뢰할 수 있는 연결 구조를 배우는 방법을 새롭게 소개하여, 대규모 상태 및 행동 공간에 대한 의존도를 줄입니다.

- **Technical Details**: 저자들은 다양한 Tucker rank 설정을 고려하여 전이 가능성 계수(transfer-ability coefficient) α를 도입합니다. 이 계수는 소스 MDP와 타겟 MDP 간의 유사도를 측정하며, 각 상태 및 행동 공간의 크기에 대한 의존성을 제거하는 알고리즘을 학습합니다.

- **Performance Highlights**: 제안된 알고리즘은 정보 이론적으로 최적이며, 타겟 MDP의 회귀 경계에서 상태 및 행동 공간 크기(|S|, |A|)에 대한 의존성을 제거할 수 있습니다. 본 연구의 결과는 기존 연구보다 더 나은 효율성을 보이며, 다양한 전이 설정에서의 최적성을 입증합니다.



### Deep Trees for (Un)structured Data: Tractability, Performance, and Interpretability (https://arxiv.org/abs/2410.21595)
Comments:
          Submitted to Machine Learning. Authors are listed in alphabetical order

- **What's New**: 본 연구에서는 Generalized Soft Trees (GSTs)라는 새로운 머신러닝 모델을 소개합니다. 이는 기존의 Soft Decision Trees (STs)의 확장으로, 비정형 데이터를 직접 처리할 수 있는 능력을 갖추고 있으며, 해석 가능성과 성능의 장점을 가지고 있습니다.

- **Technical Details**: GST는 하이퍼플레인 분할과 컨볼루션 분할을 포함한 더 일반적인 함수들을 노드에서 사용하여 구성됩니다. GST는 backpropagation 및 stochastic gradient descent를 사용하여 학습되며, DeepTree 알고리즘에 의해 성장하는 과정이 제안됩니다. 이 알고리즘은 손실 감소를 기반으로 한 새로운 분할을 선택하고 추가하는 방식을 사용하여 트리 구조를 효율적으로 성장시킵니다.

- **Performance Highlights**: GST는 MIMIC-IV, MNIST, Fashion MNIST, CIFAR-10, Celeb-A와 같은 다양한 벤치마크 데이터셋에서 다른 트리 기반 방법들과 비교할 때 뛰어난 성능을 보입니다. 특히 CIFAR-10 및 Fashion MNIST와 같은 어려운 데이터셋에서 상대적으로 우수한 성능을 나타내며, 깊은 저의 신경망에 비해 해석 가능성이 훨씬 우수하다는 것을 보여줍니다.



### Mitigating Gradient Overlap in Deep Residual Networks with Gradient Normalization for Improved Non-Convex Optimization (https://arxiv.org/abs/2410.21564)
- **What's New**: 최근 딥러닝(DL) 분야에서 Residual Networks(ResNets)의 효과적인 사용이 찾아졌습니다. 그러나, skip connection으로 인해 발생하는 gradient overlap 문제를 해결하기 위해 Z-score Normalization(ZNorm) 기법이 제안되었습니다.

- **Technical Details**: ZNorm은 gradient scale을 조정하여 layers 간의 gradient를 표준화함으로써 gradient overlap의 부정적인 영향을 줄입니다. 이를 통해 비선형 최적화(non-convex optimization) 환경에서도 훈련 과정이 개선됩니다.

- **Performance Highlights**: 실험 결과, ZNorm을 적용한 경우 최적 솔루션을 찾기 어려운 비선형 환경에서의 학습 성능이 향상되었습니다. 이러한 결과는 대규모 데이터 처리 시 정확도가 중요한 영역에서의 성능 개선을 시사합니다.



### Super-resolution in disordered media using neural networks (https://arxiv.org/abs/2410.21556)
- **What's New**: 이 논문에서는 강한 산란 매체(strongly scattering media)에서 대규모 및 다양한 데이터 세트를 활용하여 주변 매질의 그린 함수(Green's functions)를 정확하게 추정하는 방법론을 제안합니다.

- **Technical Details**: 제안된 방법은 신경망(neural networks)을 사용하거나 사용하지 않고도 그린 함수를 추정합니다.

- **Performance Highlights**: 이 방법으로 얻은 이미징 결과는 동질 매질(homogeneous medium)에서보다 더 나은 해상도를 보여주는 우수한 성과를 달성하였습니다. 이 현상은 슈퍼 해상도(super-resolution)로도 알려져 있으며, 주변 산란 매질이 물리적 이미징 아펠쳐(physical imaging aperture)를 효과적으로 향상시키기 때문에 발생합니다.



### Exploring the Design Space of Diffusion Bridge Models via Stochasticity Contro (https://arxiv.org/abs/2410.21553)
Comments:
          23 pages, 9 figures

- **What's New**: 이번 논문에서는 Stochasticity-controlled Diffusion Bridge (SDB)라는 새로운 이론적 프레임워크를 제안하여 기존의 Diffusion Bridge Models (DBMs)의 한계를 극복하고, 샘플링 효율성과 이미지 품질, 다양성을 향상시키기 위한 방법을 제시합니다.

- **Technical Details**: SDB는 두 개의 분포 사이를 연결하여 이미지-투-이미지(I2I) 변환을 위한 새로운 디자인 공간을 제공하며, Stochasticity Control (SC) 메커니즘을 도입하여 샘플링 SDEs의 불확실성을 조절합니다. 이 방식을 통해 훈련 및 샘플링 중 특이점을 완화하고, 전통적인 DBMs보다 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과에 따르면, SDB는 기존 DDBM 샘플러보다 최대 5배 빠르며, 같은 사전 훈련된 모델을 사용해도 더 낮은 FID 점수를 기록했습니다. 또한, 새로운 벤치마크로서 이미지 품질을 설정하였고, 더 많은 색상과 질감을 가진 다양한 합성 이미지를 생성할 수 있음을 보여주었습니다.



### Personalized Federated Learning with Mixture of Models for Adaptive Prediction and Model Fine-Tuning (https://arxiv.org/abs/2410.21547)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 실시간 예측을 위한 개인화된 연합 학습 알고리즘인 Fed-POE(Federated Learning with Personalized Online Ensemble)를 제안합니다. 이 알고리즘은 각 클라이언트가 자신의 데이터를 기반으로 로컬 또는 연합 모델을 조합하여 개인화된 모델을 생성할 수 있도록 합니다.

- **Technical Details**: Fed-POE는 클라이언트가 로컬 학습 모델과 연합 학습 모델의 조합을 통해 개인화된 예측 모델을 만들도록 돕습니다. 각 클라이언트는 서버에 저장된 여러 연합 모델 중 성능에 기반하여 적절한 모델을 선택하여 앙상블 모델을 구성하며, 이를 통해 온라인 예측의 메모리 및 계산 복잡성을 제어합니다.

- **Performance Highlights**: 광범위한 회귀(regression) 및 이미지 분류(image classification) 데이터셋을 대상으로 한 실험 결과, Fed-POE는 기존의 연합 학습 알고리즘보다 더 높은 실시간 예측 정확도를 달성하였습니다. 이 연구는 연합 학습과 로컬 모델 둘 다의 장점을 활용하는 방법론을 제시합니다.



### Bayesian Regression for Predicting Subscription to Bank Term Deposits in Direct Marketing Campaigns (https://arxiv.org/abs/2410.21539)
- **What's New**: 이번 연구는 포르투갈 은행의 직접 마케팅 데이터를 사용하여 term deposit 구독을 예측하는 logit 모델과 probit 모델의 효과를 조사하였습니다. 특히 Bayesian 접근법을 통해 고객 행동 예측의 정밀도를 향상시키려는 노력이 주목받고 있습니다.

- **Technical Details**: 데이터셋은 41,188개의 샘플과 21개의 변수를 포함하며, 고객의 term deposit 구독 행동을 예측하기 위해 Bayesian Logistic Regression과 Bayesian Probit Regression 모델이 비교되었습니다. 이를 위해 Leave-One-Out Cross-Validation (LOO-CV)과 Bayesian 기법을 활용하여 모델의 예측 능력을 평가하였습니다.

- **Performance Highlights**: 연구 결과 logit 모델이 classification 문제를 처리하는 데 있어서 probit 모델보다 더 우수한 성능을 보였습니다. 이러한 결과는 복잡한 의사 결정 과정을 처리할 때 모델 선택의 중요성을 강조하며, 은행이 고객 세분화 및 마케팅 캠페인을 최적화할 수 있는 방안을 제시합니다.



### L3Ms -- Lagrange Large Language Models (https://arxiv.org/abs/2410.21533)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 감독하에 세밀한 조정(Supervised Fine-Tuning, SFT)과 정렬(Alignment)을 제약 최적화 문제로 공식화하였습니다. 이는 애플리케이션의 특정 요구 사항을 충족하도록 하여 휴리스틱(Heuristic) 방법에 의한 최적화를 피할 수 있게 합니다.

- **Technical Details**: 제안된 Lagrange Large Language Models (L3Ms)는 로그 장벽(Logarithmic Barriers)을 사용하여 제약을 시행합니다. 이를 통해 L3Ms는 다양한 애플리케이션에 맞춰 커스터마이징 가능하며, 기존의 단계적 접근 방식에서 벗어나 SFT와 정렬 단계를 통합합니다.

- **Performance Highlights**: 실험을 통해 L3Ms가 다양한 애플리케이션에서 사용자 지정 요구 사항을 충족하는 데 어떻게 유용한지를 보여줍니다. 예를 들어, 두 가지 L3Ms를 사용하여 동일한 지시를 따르는 작업을 수행하되, 하나는 간결하도록 제약하고 다른 하나는 장황하도록 제약하는 결과를 비교하였습니다.



### Not All LLM-Generated Data Are Equal: Rethinking Data Weighting in Text Classification (https://arxiv.org/abs/2410.21526)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문은 기존의 LLM(대형 언어 모델)에서 생성된 합성 데이터를 실제 분포와 정렬시키기 위한 효율적인 가중 손실(Weighted-loss) 접근 방식을 제안합니다. 특히, 고품질 및 다양한 데이터를 강조하는 방법을 통해 적은 양의 실제 데이터를 사용하여 모델 성능을 향상시킵니다.

- **Technical Details**: 저자들은 IMP-Loss와 DIMP-Loss라는 두 가지 새로운 자동 가중 손실 접근 방식을 소개합니다. 이들 방법은 합성 데이터와 실제 데이터의 분포를 밀접하게 정렬하고, 높은 품질과 다양성을 가진 데이터 포인트에 더 높은 가중치를 부여합니다. 이를 통해 불필요한 데이터의 영향을 줄이고 모델 성능을 개선합니다.

- **Performance Highlights**: 다양한 텍스트 분류 작업을 통해 이 방법이 BERT 수준의 모델에서 표준 크로스 엔트로피(Cross-entropy)와 다른 데이터 가중치 접근 방식보다 효과적임을 입증하였습니다. 특히 DIMP-Loss는 모델 크기, 데이터 요구 사항 및 계산 자원 측면에서 효율적이며, 합성 데이터에서 실제 데이터를 효과적으로 활용할 수 있는 잠재력을 제공합니다.



### Diffusion-nested Auto-Regressive Synthesis of Heterogeneous Tabular Data (https://arxiv.org/abs/2410.21523)
- **What's New**: 이 논문은 표형 데이터(tabular data) 생성에 대한 새로운 접근 방식인 Diffusion-nested Autoregressive 모델(TabDAR)을 제안합니다. 기존의 자기회귀 모델(Autoregressive models)이 표형 데이터에 미흡하게 적용되었던 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: TabDAR는 먼저 연속적인 열의 조건부 확률 분포를 모델링하기 위해 디퓨전 모델(Diffusion model)을 사용합니다. 또한, 양방향 주의(attention)가 포함된 마스킹된 트랜스포머(Masked Transformers) 구조를 통해 임의의 열 순서를 시뮬레이션하며, 이를 통해 표형 데이터의 다양한 열의 순서를 자유롭게 생성할 수 있도록 합니다.

- **Performance Highlights**: TabDAR는 10개의 데이터 세트를 이용한 실험에서 기존의 최첨단 방법에 비해 18%에서 45% 이상의 성능 향상을 달성하였습니다. 통계적 충실도(Statistical Fidelity), 데이터 유용성(Data Utility), 그리고 프라이버시 보호(Privacy Protection) 측면에서 월등한 성능을 나타냈고, 결측값 대체(missing value imputation) 작업에서도 놀라운 성능을 보였습니다.



### A Multi-Agent Reinforcement Learning Testbed for Cognitive Radio Applications (https://arxiv.org/abs/2410.21521)
Comments:
          Accepted to IEEE CCNC 2025

- **What's New**: 이번 논문에서는 멀티 에이전트 강화 학습(MARL) 기능이 추가된 RFRL Gym의 업데이트된 환경을 소개합니다. 이전에는 단일 RL 에이전트만 훈련할 수 있었던 RFRL Gym이 이제는 여러 에이전트를 동시에 훈련할 수 있게 개선되었습니다.

- **Technical Details**: RFRL Gym은 OpenAI Gym 프레임워크를 기반으로 구축되어 있으며, RF 스펙트럼 내에서 시뮬레이션 시나리오를 커스터마이징할 수 있는 기능을 제공합니다. 이번 업데이트를 통해 멀티 에이전트 강화 학습(MARL)을 지원하여 여러 인공지능 에이전트가 협력적, 경쟁적 또는 혼합 환경에서 훈련될 수 있게 되었습니다.

- **Performance Highlights**: 이 논문에서는 다양한 RF 시나리오에서의 MARL 환경 테스트 결과와 함께 향후 추가될 기능도 소개합니다. RFRL Gym은 다양한 협력적 및 경쟁적인 상황을 지원할 수 있는 유연성을 제공하며, 이는 향후 IoT 디바이스의 증가와 RF 스펙트럼 활용 증가에 따른 필수 요구 사항을 충족시킬 것입니다.



### LLM-Forest for Health Tabular Data Imputation (https://arxiv.org/abs/2410.21520)
- **What's New**: 이번 연구에서는 기존의 데이터 보간(method) 방식의 한계를 극복하기 위해 LLM-Forest라는 새로운 프레임워크를 제안합니다. 이는 few-shot learning을 활용한 LLM '나무'를 여러 개 결합하여 보다 robust한 결과를 제공할 수 있도록 설계되었습니다.

- **Technical Details**: LLM-Forest는 bipartite 정보 그래프(bipartite information graphs)를 기반으로 하여 고품질의 이웃 데이터를 식별하는 그래프 알고리즘을 사용합니다. 이 알고리즘은 feature와 value 수준에서의 granularity를 통해 데이터의 복잡한 관계를 포착하며, O(n)으로 계산 복잡도를 줄일 수 있습니다. 여러 LLM '나무'로 구성된 이 프레임워크는 ensemble learning에 기반하여 hallucination(환각) 문제를 감소시킵니다.

- **Performance Highlights**: 실제 건강 데이터셋을 포함한 네 개의 데이터셋에 대한 실험을 통해 LLM-Forest의 효과성과 효율성을 입증하였습니다. 전통적인 방법들과 비교했을 때, LLM-Forest는 결측값 보간(imputation)에서 뛰어난 성능을 보이며, 실제 의료 데이터 분석에 대한 응용 가능성을 강조합니다.



### Predicting sub-population specific viral evolution (https://arxiv.org/abs/2410.21518)
- **What's New**: 이 논문에서는 서로 다른 지역의 바이러스 단백질 분포의 변화를 예측하는 하위 집단 특정 단백질 진화 모델을 제안합니다. 이 모델은 전염률을 명시적으로 모델링하고 데이터로부터 그 상호 의존성을 학습합니다.

- **Technical Details**: 단백질 서열의 발생 변화를 시간에 따라 설명하기 위해 전파율 행렬을 매개로 하는 일반 미분 방정식(ODE)을 사용합니다. ODE를 풀어서 특정 하위 집단에 특정 단백질이 발생할 확률을 계산합니다. 주요 구성 요소로는 언어 모델을 사용하여 전파율 행렬과 경계 조건을 모델링하는 방법이 포함됩니다.

- **Performance Highlights**: SARS-CoV-2와 인플루엔자 A/H3N2에 대한 다년간 평가에서 우리의 모델은 다양한 대륙과 국가에서 단백질 분포를 정확하게 예측하는 데 있어서 기존의 모델들을 초월합니다. 모델이 생성한 단백질 서열의 상위 500개는 여러 나라에서 보고된 순환 서열의 92.8%를 커버할 수 있음을 보여줍니다.



### Sabotage Evaluations for Frontier Models (https://arxiv.org/abs/2410.21514)
- **What's New**: 이번 연구는 AI 모델의 사보타주(sabotage) 능력과 이에 따른 잠재적 위험에 대한 새로운 평가 프레임워크를 개발했습니다. 특히, 모델이 자율적으로 다른 시스템의 측정 및 의사 결정을 방해할 수 있는 능력에 중점을 두고 있습니다.

- **Technical Details**: 연구팀은 모델의 사보타주 능력을 3가지 임계값(threshold)으로 구분했습니다: 배포 전 행동 숨기기, 감독 undermining, 조직적 행동에 대한 사보타지. 각 임계값에 대한 검증 평가를 설계하고, Anthropic의 Claude 3 Opus 및 Claude 3.5 Sonnet 모델을 통해 실험했습니다.

- **Performance Highlights**: Claude 3 Opus와 Claude 3.5 Sonnet 모델은 기본적인 감독 하에서는 사보타주 임계값을 충족하지 않았으나, 모든 작업에서 비트리비얼(trivial) 성능을 보였습니다. 이는 현재 minimal mitigations이 사보타주 위험을 다루기에 충분하지만, 앞으로는 더 강력한 mitigations이 필요할 것임을 나타냅니다.



### Trustworthiness of Stochastic Gradient Descent in Distributed Learning (https://arxiv.org/abs/2410.21491)
- **What's New**: 이번 논문에서는 분산 학습(distributed learning)에서 컴프레스트 SGD(compressed SGD) 기법의 신뢰성을 평가합니다. 기존 연구들은 주로 단일 공격자(singular attackers) 상황에서 강도를 낮춘 기법에 집중했으나, 이 연구는 두 가지 머신러닝 공격인 Gradient Inversion(GradInv)과 Membership Inference Attack(MIA)에 대한 포괄적인 분석을 제공하고 있습니다.

- **Technical Details**: 실험은 PowerSGD와 Top-K SGD라는 두 가지 주요 기법을 통해 진행되며, 이 두 방법 모두 그래디언트 압축(gradient compression) 기술을 사용하여 통신 비용을 줄이는 데 초점을 맞추고 있습니다. 연구 결과에 따르면, 컴프레스트 SGD는 비압축 SGD에 비해 프라이버시 유출에 대한 저항력이 더 높습니다.

- **Performance Highlights**: 연구 결과, 세 가지 데이터셋에서 압축된 SGD 알고리즘이 그래디언트 인버전 공격에 대해 더 강력한 저항성을 보였으며, MIA는 두 SGD 방식에 대해 낮은 민감도를 나타내어 프라이버시 위험 평가에 있어 MIA의 신뢰성이 낮음을 시사합니다.



### A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions (https://arxiv.org/abs/2410.21484)
- **What's New**: 이번 연구는 스포츠 베팅에서 머신러닝(Machine Learning) 기술의 최신 동향과 도전 과제를 살펴보고 있습니다. 연구에 따르면, 머신러닝은 베팅 전략 최적화와 예측의 정확성을 향상시키며, 이 과정에서 다양한 장르의 스포츠에 적용됩니다.

- **Technical Details**: 연구에서는 서포트 벡터 머신(Support Vector Machines), 랜덤 포레스트(Random Forests), 신경망(Neural Networks) 등 다양한 머신러닝 기법이 활용됩니다. 이 모델들은 역사적 데이터, 경기 중 통계 및 실시간 정보를 사용하여 베팅 전략을 최적화하고 가치 있는 베팅 기회를 식별합니다.

- **Performance Highlights**: 머신러닝 모델들은 베팅 수익성을 향상시키는 데 중요한 역할을 하며, 베팅업체는 역동적인 배당 조정(dynamic odds adjustment)을 통해 리스크를 관리합니다. 또한, 의심스러운 베팅 패턴을 감지하기 위한 이상 탐지(anomaly detection) 모델의 활용이 강조됩니다.



### AiSciVision: A Framework for Specializing Large Multimodal Models in Scientific Image Classification (https://arxiv.org/abs/2410.21480)
- **What's New**: AiSciVision은 과학 연구에서의 인공지능(AI) 사용을 위한 신뢰성과 해석 가능성을 제공하는 새로운 프레임워크입니다. 이 프레임워크는 대형 다중모달 모델(LMM)을 인터랙티브 연구 파트너로 변환하고 이미지 분류 작업을 위한 모델로 특화되었습니다.

- **Technical Details**: AiSciVision은 두 가지 주요 구성 요소로 이루어져 있습니다: (1) Visual Retrieval-Augmented Generation (VisRAG)과 (2) 도메인 특정 도구입니다. 이 시스템은 이미지 분류를 위해 유사한 이미지(양성 및 음성 레이블이 붙은)를 검색하고, LMM 에이전트가 이를 기반으로 도구를 선택하여 여러 번에 걸쳐 이미지를 조작하고 검토합니다.

- **Performance Highlights**: AiSciVision은 세 가지 실제 과학 이미지 분류 데이터셋에서 평가되었으며, 수조(aquaculture) 연못, 병든 이름풀이(eelgrass), 태양광 패널의 존재를 탐지하는 작업에서 완전 감독 모델보다 더 나은 성능을 보였습니다.



### ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inferenc (https://arxiv.org/abs/2410.21465)
- **What's New**: ShadowKV 시스템은 긴 문맥을 처리하는 LLM 추론 성능을 향상시키기 위해 저랭크( Low-rank ) 키 캐시를 저장하고 값 캐시( Value Cache )를 오프로드하는 방식을 도입합니다. 이를 통해 메모리 사용량을 줄이고 높은 배치 크기와 긴 시퀀스를 지원합니다.

- **Technical Details**: ShadowKV는 CUDA 멀티 스트림을 활용하여 선택된 키 캐시의 복구와 해당 값 캐시의 로드를 겹쳐서 수행하여 디코딩 지연을 최소화합니다. 또한, 사전 RoPE 키가 낮은 차원에 집중되어 제대로 된 KV 쌍을 동적으로 선택하여 정확성을 유지합니다.

- **Performance Highlights**: ShadowKV는 A100 GPU에서 최대 6배의 배치 크기를 지원하며, 최대 3.04배의 처리량을 증가시킵니다. Llama-3.1-8B 모델을 사용했을 때, 문맥 길이가 122K인 샘플에서 이 성능을 실현하였습니다.



### Inverting Gradient Attacks Naturally Makes Data Poisons: An Availability Attack on Neural Networks (https://arxiv.org/abs/2410.21453)
Comments:
          8 pages, 10 figures

- **What's New**: 이 논문은 비볼록(non-convex) 설정에서 데이터 중독(data poisoning) 공격이 그라디언트 공격(gradient attack)과 유사한 방식으로 가능하다는 것을 보여줍니다. 특히 데이터 중독을 통해 신경망(neural networks)에 대한 가용성 공격(availability attack)을 수행할 수 있음을 처음으로 입증했습니다.

- **Technical Details**: 저자들은 그라디언트 역전(gradient inversion) 방법을 이용하여 악의적인 그라디언트를 통해 데이터 포인팅을 재구성(reconstruct)하는 방식으로 데이터 중독이 가능하다는 것을 설명합니다. 실험을 통해 다양한 신경망 아키텍처에 대해 데이터 중독 공격을 수행하였으며, 상태-of-the-art 방어 메커니즘에도 불구하고 성공적인 가용성 공격을 수행했습니다.

- **Performance Highlights**: 데이터 중독 공격이 모델 성능을 1%의 악성 데이터 포인트로 무작위 수준(random-level)으로 저하시키는 가용성 공격을 성공적으로 수행할 수 있음을 보여주었습니다. 이들은 선택한 최적화 알고리즘을 통해 이미지 분류 작업에 사용되었으며, 데이터 중독이 그라디언트 공격보다 효과적이지 않더라도 가용성 공격 범위는 동일한 효과를 할 수 있음을 증명했습니다.



### A Temporal Linear Network for Time Series Forecasting (https://arxiv.org/abs/2410.21448)
- **What's New**: 이번 논문에서는 복잡한 딥러닝 아키텍처가 시간 시계열 예측에 반드시 필요하지 않다는 점을 강조하며, 간단한 선형 모델들이 종종 더 정교한 접근법보다 뛰어난 성능을 발휘할 수 있음을 보여줍니다. 이에 기반하여 새로운 아키텍처인 Temporal Linear Net (TLN)을 소개합니다.

- **Technical Details**: TLN은 선형 모델의 해석 가능성과 계산 효율성을 유지하면서 다변량 시간 시계열 데이터의 시간적 및 특성 의존성을 효과적으로 포착하도록 설계되었습니다. TLN은 TSMixer의 변형으로서, 아키텍처 전반에 걸쳐 엄격한 선형성을 유지합니다. 이 모델은 활성화 함수 없이 특수한 커널 초기화를 도입하고, 다양한 시간 스케일을 처리하기 위해 dilated convolutions를 사용합니다.

- **Performance Highlights**: TLN은 다양한 시퀀스 길이와 예측 지평선에서 강력한 성능을 보이며, 더 복잡한 아키텍처와 비교해도 우수하거나 동등한 성능을 달성합니다. 또한 TLN은 완전한 해석 가능성을 유지하고, 다양한 입력 구성에서도 탁월한 안정성을 보여줍니다.



### Sum-of-squares lower bounds for Non-Gaussian Component Analysis (https://arxiv.org/abs/2410.21426)
- **What's New**: 이 논문에서는 Non-Gaussian Component Analysis (NGCA)의 복잡성을 Sum-of-Squares (SoS) 프레임워크에서 연구하였습니다. 특히, 기존에 알려진 알고리즘들이 요구하는 샘플 수보다 현저히 적은 수의 샘플로도 방향 $v$의 존재를 규명하지 못함을 보여줍니다.

- **Technical Details**: NGCA는 고차원 데이터 집합에서 비율정규(Non-Gaussian) 방향을 찾는 통계적 작업입니다. 이 작업의 목표는 분포 $P^A_{v}$에서 $A$의 첫 $k-1$ 모멘트가 표준 가우시안과 같고 $k$-번째 모멘트가 다른 방향을 근사하는 것입니다. 우리는 SoS 프레임워크에서 NGCA에 대한 첫 번째 초상수( супер-constant) 차수 하한을 제시합니다.

- **Performance Highlights**: 이 연구는 이전 연구를 통해 제시된 정보-계산 무역의 초다항적( super-polynomial) 하한을 세우는 성과를 달성하였습니다. 이 결과는 여러 문제에 대해 SoS 하한을 도출할 수 있게 하여, 보다 넓은 범위의 알고리즘에 대한 제약을 강하게 합니다.



### Bayesian Collaborative Bandits with Thompson Sampling for Improved Outreach in Maternal Health Program (https://arxiv.org/abs/2410.21405)
- **What's New**: 이번 연구에서는 mHealth 프로그램에서 최적의 건강 정보 발신 시점을 자동으로 조정하기 위한 새로운 베이지안 접근 방식을 제안합니다. 여러 수혜자를 대상으로 하는 다중 무장 밴딧(Multi-armed Bandit) 문제에 대해 Thompson Sampling을 활용하여 저랭크 보상 행렬을 효과적으로 학습합니다.

- **Technical Details**: 제안된 방법은 Gibbs 샘플링을 통해 저랭크 행렬 요소에 대한 후행 추론을 수행하며, 이로 인해 빠른 수렴을 가능하게 합니다. 기존의 방법들과 비교하여, 신속한 수신 시점을 학습할 수 있는 장점을 가지고 있습니다. 또한 Stochastic Gradient Langevin Dynamics(SGLD)를 사용하여 샘플링을 효과적으로 처리합니다.

- **Performance Highlights**: Kilkari 프로그램의 실제 데이터셋을 활용하여 테스트한 결과, 제안된 방법이 기존 시스템 대비 전화를 시도하는 횟수를 47% 감소시키고, 최신 기법 대비 16%의 감소를 보여 프로그램 규모를 더욱 확장할 수 있는 가능성을 지니고 있음을 증명했습니다. 또한 수혜자 이탈률을 29% 감소시키는 성과를 얻어 mHealth 프로그램의 성과를 크게 개선했습니다.



### Unveiling the Role of Expert Guidance: A Comparative Analysis of User-centered Imitation Learning and Traditional Reinforcement Learning (https://arxiv.org/abs/2410.21403)
Comments:
          Published as CEUR Workshop Proceedings in Proceedings of the 1st International Workshop on Human-in-the-Loop Applied Machine Learning (HITLAML 2023). Awarded Best Paper. this https URL

- **What's New**: 본 연구는 전통적인 강화 학습 방법과 비교하여 모방 학습(imitation learning)의 성과, 강인성(robustness), 그리고 한계에 대해 깊이 있는 분석을 제공합니다. 특히, 인간 피드백(human feedback)을 통해 지식이 어떻게 효과적으로 통합되는지를 살펴봅니다.

- **Technical Details**: 연구는 Unity 플랫폼을 사용한 시뮬레이션 환경에서 진행되었으며, 모방 학습 기술의 기존 기법과 최신 강화 학습 방법(Proximal Policy Approximation (PPO), Soft-Actor Critic (SAC))을 비교합니다. 2D Bird Hunter 게임을 통해 에이전트를 훈련시키고, 여러 복잡한 환경을 설정하여 모델 성능을 평가하였습니다.

- **Performance Highlights**: 모방 학습과 전통적인 강화 학습 간의 비교 결과, 모방 학습이 특히 복잡한 환경에서 효율적임이 입증되었습니다. 이 연구는 복잡한 실세계 문제에 효과적으로 대처할 수 있는 지능형 시스템 개발에 기여할 것입니다.



### LinFormer: A Linear-based Lightweight Transformer Architecture For Time-Aware MIMO Channel Prediction (https://arxiv.org/abs/2410.21351)
- **What's New**: 6세대(6G) 이동통신 네트워크의 출현으로 인해 고속 이동 통신 지원과 채널 노화(channel aging) 문제 해결을 위한 새로운 도전 과제가 제기되었습니다. 기존 채널 예측 방법은 더 높은 정확도를 제공하지만, 계산 복잡성이 증가하여 모바일 네트워크에서의 실용적인 적용을 제한합니다. 이러한 문제를 해결하기 위해 스케일이 가능한 LinFormer라는 혁신적인 채널 예측 프레임워크를 제안합니다.

- **Technical Details**: LinFormer는 자연어 처리(NLP) 모델인 BERT에서 영감을 받아 설계된 인코더 전용 Transformer 모델 기반의 채널 예측 프레임워크입니다. 본 접근법에서는 Transformer에서 일반적으로 사용되는 계산 집약적인 주의(attention) 메커니즘을 시간 인식 다층 퍼셉트론(TMLP)으로 대체하여 계산 요구 사항을 크게 줄입니다. TMLP 모듈은 시간 인식을 내재하고 있어 채널 예측 업무에 특히 적합합니다.

- **Performance Highlights**: LinFormer는 균형잡힌 손실 함수인 WMSELoss와 데이터 증가(data augmentation) 기술을 이용하여 통신 데이터셋을 활용한 훈련 과정을 강화했습니다. 실험 결과, LinFormer는 다양한 이동 시나리오에서 기존 방법보다 뛰어난 성능을 보여 주어, 향후 무선 통신 시스템에 대한 유망한 솔루션을 제공하고 있습니다.



### FALCON: Feedback-driven Adaptive Long/short-term memory reinforced Coding Optimization system (https://arxiv.org/abs/2410.21349)
Comments:
          20 pages, 7 figures

- **What's New**: 최근 대형 자연어 처리 모델(LLMs)이 자동화된 코드 생성에서 큰 발전을 이루었지만, 사용자 의도와의 정렬 문제를 겪고 있습니다. 본 논문에서는 피드백 기반의 적응형 장기/단기 메모리 강화 코드 최적화(FALCON)를 제안하여 이러한 문제를 해결하고 코딩 성능을 향상시키고자 합니다.

- **Technical Details**: FALCON은 두 가지 계층적 구조로 되어 있으며, 장기 메모리는 코드 품질을 향상시키기 위해 학습된 지식을 지속적으로 적용하고, 단기 메모리는 컴파일러와 AI 시스템으로부터의 즉각적인 피드백을 통합합니다. 또한 피드백 보상을 통한 메타 강화 학습을 도입하여 범지구적-국소적 최적화 문제를 해결하고 다양한 코드 생성 과제에 대한 모델의 적응성을 강화합니다.

- **Performance Highlights**: 광범위한 실험 결과, FALCON은 MBPP 벤치마크에서 4.5% 이상, Humaneval 벤치마크에서 6.1% 이상 다른 강화 학습 방법들을 능가하며 최첨단 성능을 달성하였습니다.



### Towards Trustworthy Machine Learning in Production: An Overview of the Robustness in MLOps Approach (https://arxiv.org/abs/2410.21346)
- **What's New**: 이 논문은 MLOps 시스템의 신뢰성 속성에 대한 포괄적인 개요를 제공하고, robust ML 솔루션 채택을 위한 기술적 실행 방법을 강조합니다.

- **Technical Details**: MLOps는 ML 시스템을 운영하기 위해 DevOps 원칙(continuous integration, continuous delivery)을 채택하고, 데이터 수집, 모델 배포 등 ML 작업의 모든 단계에서 시스템 개발을 지원합니다. 이 논문은 robust ML 시스템의 사양을 formalize하고, MLOps의 기술적 견고성을 위한 기존 접근 방식과 도구들을 리뷰합니다.

- **Performance Highlights**: 이 연구는 robust ML 솔루션의 세 가지 구성 요소를 정의하고, ML 솔루션의 작업 흐름과 MLOps 시스템 요소를 반영하여 신뢰할 수 있는 AI 시스템의 주된 속성으로서의 견고성을 정량화합니다. 또한, 향후 연구 방향 및 기회를 제안합니다.



### Retrieval-Retro: Retrieval-based Inorganic Retrosynthesis with Expert Knowledg (https://arxiv.org/abs/2410.21341)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 기계 학습을 활용하여 무기 화합물의 구조를 계획하는 새로운 방법인 Retrieval-Retro를 제안했습니다. 기존의 유기 화합물 합성 경로를 계획하는 방법과는 달리, 무기 화합물에 대한 레트로 합성 경로를 보다 효과적으로 학습할 수 있도록 다양한 주의(attention) 레이어를 통해 참고 자료의 전구체 정보를 암묵적으로 추출합니다.

- **Technical Details**: Retrieval-Retro는 도메인 전문성을 기반으로 하여, 타겟 물질과 전구체 간의 열역학적 관계를 고려함으로써 전구체 세트를 보다 효과적으로 찾을 수 있게 합니다. 이를 통해 기계 학습 모델이 기존 합성 레시피를 초월하는 새로운 합성 레시피를 발견하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 Retrieval-Retro의 효과성을 입증했으며, 특히 새로운 합성 레시피를 발견하는 데 있어 그 우수성을 보여주었습니다. 이 방법은 실제 무기 물질 발견의 응용 가능성을 강조합니다.



### Meta-Learning for Speeding Up Large Model Inference in Decentralized Environments (https://arxiv.org/abs/2410.21340)
- **What's New**: 대규모 모델(large-scale models) 배포에서 발생하는 높은 비용 문제를 해결하기 위해 분산 시스템(decentralized systems)으로의 전환이 이루어지고 있으며, 이 과정에서 효율적인 추론 가속화(inference acceleration) 필요성이 강조되고 있습니다. 본 논문에서는 메타 학습(meta-learning) 기반 프레임워크를 도입하여 최적의 가속화 방법 선택 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 메타 학습 프레임워크(MetaInf)는 다양한 과제(task)에서 여러 가속화 기술의 성능 데이터를 학습하여, 각 과제의 특성에 맞는 최상의 가속화 전략을 식별합니다. 기존의 임의 선택(random selection) 방식이나 전문가의 직관에 의존하는 방법과 달리, 체계적이고 자동화된 방식으로 가속화 방법을 결정합니다. 이 프레임워크는 BSNS 모델에 통합되어 분산 네트워크에서의 효율성을 높입니다.

- **Performance Highlights**: 본 연구는 제안된 메타 학습 기반 프레임워크가 전통적 방법에 비해 효율성과 성능 면에서 일관되게 우수함을 보여주며, 대규모 모델 배포와 AI 민주화(democratization)에 기여함을 강조합니다.



### E(3)-invaraint diffusion model for pocket-aware peptide generation (https://arxiv.org/abs/2410.21335)
- **What's New**: 이 연구에서는 생물학자들이 필요로 하는 새로운 단백질 억제제 발견 방법을 제시합니다. 기존 연구들이 일반적인 구조나 생물학적 속성에 국한된 반면, 본 연구는 특정 수용체 포켓에 주목하여 구조와 서열 생성을 할 수 있는 새로운 기계학습 접근법을 사용합니다.

- **Technical Details**: 본 연구의 방법론은 두 개의 순차적인 diffusion 모델로 구성됩니다: 1) conditional structure diffusion model과 2) conditional sequence diffusion model. 이 모델들은 수용체 포켓의 정보에 기반하여 효소 단백질의 구조와 아미노산 서열을 생성합니다. E(3)-invariant 표현 방식을 통해 백본 원자 간의 각도와 디헤드럴 관계를 반영하여 펩타이드 구조의 일관성을 유지합니다.

- **Performance Highlights**: 제안된 방법은 최신 모델들과 비교하여 경쟁력 있는 성능을 보여주었으며, 특정 수용체 포켓을 고려한 펩타이드 디자인의 잠재력을 강조합니다. 이 연구는 수용체 특이적인 펩타이드 생성을 통한 정밀한 약물 발견의 새로운 접근법을 제공합니다.



### Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Wors (https://arxiv.org/abs/2410.21333)
- **What's New**: 본 논문은 Chain-of-thought(CoT) 프롬프트 기법이 대규모 언어 및 다중 모달 모델의 성능에 미치는 영향을 체계적으로 분석합니다. 특히, CoT가 성능을 저하시키는 특정 작업의 특성을 확인하고, 인지 심리학에서 영감을 받아 CoT의 부정적 영향을 이해하기 위한 새로운 틀을 제시합니다.

- **Technical Details**: CoT는 모델이 '단계별 사고'(think step-by-step)를 하도록 유도하는 프롬프트 기법으로, 여러 작업에서 성능을 향상시키는 것으로 알려져 있습니다. 그러나 본 연구에서는 CoT가 OpenAI o1-preview 등 최신 모델의 성능을 최대 36.3%까지 감소시킬 수 있는 작업을 확인했습니다. 실험은 암묵적 통계 학습(implied statistical learning), 시각적 인식(visual recognition), 예외가 포함된 패턴으로 분류하는 작업에서 수행되었습니다.

- **Performance Highlights**: CoT는 특정 조건(i와 ii를 동시에 만족하는 경우)에서 모델 성능을 크게 저하시켰습니다. 예를 들어, CoT를 사용할 때 OpenAI o1-preview의 성능은 GPT-4o와 비교하여 절대 정확도가 최대 36.3% 감소했습니다. 반면, 다른 세 가지 작업 카테고리에서는 CoT가 모델 성능에 부정적인 영향을 미치지 않는 것으로 나타났습니다.



### Building, Reusing, and Generalizing Abstract Representations from Concrete Sequences (https://arxiv.org/abs/2410.21332)
- **What's New**: 이 논문에서는 비모수 계층 변수 학습 모델(HVM)을 도입하여, 시퀀스에서 청크(Chunk)를 학습하고, 맥락적으로 유사한 청크를 변수로 추상화하는 방법을 제안합니다. 이는 메모리를 효율적으로 조직하고 압축된 시퀀스 표현을 가능하게 합니다.

- **Technical Details**: HVM은 관찰 시퀀스에서 스스로 객체와 카테고리를 발견하는 비지도학습 원칙을 사용하며, 청크 제안(Chunk Proposal)과 변수 발견(Variable Discovery)이라는 두 가지 메커니즘을 내세웁니다. HVM은 기존 모델들보다 더 효율적인 딕셔너리를 학습하며, 압축과 일반화 사이의 정밀한 균형을 실현합니다.

- **Performance Highlights**: HVM은 babyLM 언어 데이터셋에서 일반적인 압축 알고리즘인 Lempel-Ziv보다 더 효율적인 성능을 보여주며, 시퀀스를 상기하는 작업에서 HVM의 시퀀스 가능도가 인간의 회상 시간과 관련이 있음을 증명합니다. 대형 언어 모델(LLM)들과의 비교를 통해, HVM은 인간과 유사한 추상화 기제를 보여주고, LLM은 추상 변수를 효과적으로 전이하지 못함을 강조합니다.



### Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness (https://arxiv.org/abs/2410.21331)
- **What's New**: 이 논문은 deep learning 모델의 해석 가능성에서의 polysemanticity 문제를 다루며, monosemanticity가 해석 가능성 뿐만 아니라 모델의 정확성도 향상시킬 수 있음을 제시합니다. 특히, 모델의 성능을 강화할 수 있는 다양한 상황에서 monosemantic feature의 장점을 발견하였습니다.

- **Technical Details**: 모델이 monosemantic features를 활용할 경우, noisy input 및 label, few-shot learning, 그리고 out-of-domain generalization과 같은 다양한 상황에서 성능이 향상됩니다. 특히, monosemanticity는 feature 표현의 더 나은 분리를 촉진하여 더 견고한 결정 경계를 형성하는 데 기여합니다.

- **Performance Highlights**: 모델은 label noise 하에 13.7%의 top-1 정확도 향상, few-shot fine-tuning에서는 3.9%의 상향을 보여줍니다. 또한, LLM의 fine-tuning에서 MonoLoRA라는 접근법을 사용하여 모델 정렬을 더욱 잘 유지하며 작업 성능을 개선할 수 있음을 보여주었습니다.



### Deconfounding Time Series Forecasting (https://arxiv.org/abs/2410.21328)
- **What's New**: 이번 연구에서는 기존의 시계열 예측 방법의 한계를 극복하기 위해 잠재적인 혼란 변수(latent confounders)의 영향을 고려하여 예측 품질을 향상시키는 새로운 접근법을 제안합니다. 이 방법은 과거 데이터에서 추출한 혼란 변수의 표현을 통합하여 시계열 예측의 정확성과 강건성을 높이고자 합니다.

- **Technical Details**: 제안된 방법은 Time Series Deconfounder를 기반으로 하며, 인과 추론(causal inference) 기법을 사용하여 혼란 변수의 영향을 식별하고 예측 모델에 통합합니다. 이 모델은 과거 변수의 데이터를 활용하여 현재의 예측 변수를 개선하는 데 초점을 둡니다.

- **Performance Highlights**: 기후 과학 데이터에 대한 적용 결과, 기존의 혼란 변수를 고려하지 않은 전통적인 방법들에 비해 유의미한 예측 정확도 향상을 보여주었습니다.



### Self-Supervised Learning and Opportunistic Inference for Continuous Monitoring of Freezing of Gait in Parkinson's Diseas (https://arxiv.org/abs/2410.21326)
Comments:
          11 pages

- **What's New**: 이 논문에서는 Parkinson's disease (PD) 환자의 Freezing of Gait (FoG) 감지를 위한 LIFT-PD라는 새로운 self-supervised learning 프레임워크를 제안합니다. 이는 라벨이 없는 데이터에서 self-supervised pre-training과 differential hopping windowing 기술을 결합하여 제한된 라벨 인스턴스에서 배울 수 있도록 설계되었습니다.

- **Technical Details**: LIFT-PD는 self-supervised learning 방식을 사용하여 non-FoG 활동과 비교하여 sparse한 FoG 사건을 감지하는 robust한 deep learning 모델을 훈련합니다. 또한, Differential Hopping Windowing(DHW) 기법을 통해 데이터를 전처리하고, opportunistic model activation 모듈을 통해 전력 소비를 줄여줍니다. 이 모델은 3차원 accelerometer 센서를 통해 raw 신호를 수집하여 multivariate time-series classification으로 FrG 사건을 탐지합니다.

- **Performance Highlights**: LIFT-PD는 supervised 모델에 비해 precision이 7.25% 증가하고 accuracy가 4.4% 향상되며, supervised learning에서 사용되는 라벨 데이터를 40%만 사용하여 동일한 성능을 보여줍니다. 또한, inference 시간이 연속 추론에 비해 최대 67% 단축됩니다.



### Just Propagate: Unifying Matrix Factorization, Network Embedding, and LightGCN for Link Prediction (https://arxiv.org/abs/2410.21325)
- **What's New**: 정확한 링크 예측(link prediction)을 위한 통합 프레임워크를 제안함으로써 기존의 다양한 그래프 기반 머신러닝 모델의 이해도를 높이고 새로운 설계를 위한 영감을 제공하고자 함.

- **Technical Details**: 제안하는 프레임워크는 행렬 분해(matrix factorization), 네트워크 임베딩(network embedding) 방법, 그래프 신경망(graph neural network) 방법을 아우름. 이를 통해 링크 예측을 위한 기존 방법의 학습 과정과 성능에 영향을 미치는 여러 가지 설계 요소를 탐색함.

- **Performance Highlights**: 기존 방식들이 프레임워크 내에서 어떻게 표현이 전파(propagation)되는지를 보여주고, 높은 차수 이웃(neighbors) 정보를 어떻게 활용하는가에 대한 통찰력을 제공함.



### Angel or Devil: Discriminating Hard Samples and Anomaly Contaminations for Unsupervised Time Series Anomaly Detection (https://arxiv.org/abs/2410.21322)
Comments:
          14 pages, 9 figures, 5 tables

- **What's New**: 이 논문에서는 전통적인 손실 기반 접근 방법 대신 '파라미터 행동(parameter behavior)'을 보완하여 이상 패턴의 세밀한 특성을 담아내는 새로운 기법, 즉 듀얼 파라미터-손실 데이터 증강 방법(PLDA)을 제안합니다.

- **Technical Details**: PLDA는 강화 학습(reinforcement learning) 패러다임 내에서 구현되며, 학습 데이터의 증대를 동적으로 수행합니다. 이 방법은 파라미터 민감도(parameter sensitivity)를 기반으로 한 보상 함수를 사용하여 정상 샘플(hard normal samples, HS)과 이상 상태 스팸(anomaly contaminations, AC)을 효과적으로 구분합니다.

- **Performance Highlights**: 10개 데이터세트에 대한 광범위한 실험 결과, PLDA는 기존의 데이터 증강 방법들보다 최대 8% 향상된 성능을 보였으며, 다양한 이상 감지 모델에 원활하게 통합될 수 있음을 입증하였습니다.



### Towards Continuous Skin Sympathetic Nerve Activity Monitoring: Removing Muscle Nois (https://arxiv.org/abs/2410.21319)
Comments:
          4 pages, 5 figures, 1 table, IEEE-EMBS International Conference on Body Sensor Networks: NextGen Health: Sensor Innovation, AI, and Social Responsibility (IEEE BSN 2024)

- **What's New**: 이 연구에서는 비침습적인 피부 자율 신경 활동(SKNA)의 지속적인 모니터링 방법을 제안하며, 심장 관련 질환의 원인을 이해하는 데 기여할 것으로 기대된다.

- **Technical Details**: 연구팀은 심전도(ECG) 전극을 통해 수집된 SKNA 신호에서 근육 소음을 감지하고 제거하기 위해 2D CNN(Convolutional Neural Network) 모델을 활용하였다. 데이터 세트는 다양한 실험 상황에서 수집되었으며, 최종 모델은 baseline, 스트레스 유도 SKNA 및 근육 소음이 포함된 기간을 분류하는 데 있어 평균 89.85%의 정확도를 보였다.

- **Performance Highlights**: 이 연구 결과는 신뢰할 수 있는 SKNA 모니터링의 필요성을 강조하며, 궁극적으로 실제 적용을 위한 웨어러블 SKNA 센서 개발로 나아가는 중요한 기초가 될 것이다.



### Deep Optimizer States: Towards Scalable Training of Transformer Models Using Interleaved Offloading (https://arxiv.org/abs/2410.21316)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 효율적인 훈련을 위한 새로운 방법인 Deep Optimizer States를 제안합니다. 이 방법은 GPU 메모리와 호스트 메모리를 동적으로 관리하여 메모리 한계를 극복하고 훈련 속도를 향상시킵니다.

- **Technical Details**: Deep Optimizer States는 모델 파라미터와 옵티마이저 상태를 작은 부분집합으로 분할하여 각 변화를 GPU와 CPU 간에 인터리빙 방식으로 오프로드하도록 설계되었습니다. 이러한 방법으로 GPU와 CPU의 기계성이 최적화되어 훈련 성능을 극대화합니다.

- **Performance Highlights**: 연구 결과, Deep Optimizer States를 적용하여 기존의 최첨단 기술보다 2.5배 더 빠른 훈련 속도를 달성했습니다. 특히, 20B 파라미터의 모델을 대상으로 한 실험에서 유의미한 시간 단축과 모델 업데이트 속도의 3배 향상을 보여주었습니다.



### $\texttt{PatentAgent}$: Intelligent Agent for Automated Pharmaceutical Patent Analysis (https://arxiv.org/abs/2410.21312)
Comments:
          7 pages

- **What's New**: 이번 연구에서는 제약 산업에서 특허 분석의 통합을 목표로 하는 인공지능 에이전트인 \\texttt{PatentAgent}를 소개합니다. 이는 약물 연구에 영향을 미치는 특허 데이터를 읽고, 화합물 구조를 인식하는 과정을 단순화하며, 연구자들에게 필수적인 도구가 될 것으로 보입니다.

- **Technical Details**: \\texttt{PatentAgent}는 세 가지 주요 모듈로 구성되어 있습니다: \\textit{PA-QA} (특허 질문-응답), \\textit{PA-Img2Mol} (이미지-화합물 구조 변환), \\textit{PA-CoreId} (핵심 화학 구조 식별). 이 시스템은 자연어 쿼리를 처리하고, 화학 이미지에서 분자 구조를 추출하며, 여러 화학 화합물 간의 중심 화학 구조를 정의하는 데 중점을 둡니다.

- **Performance Highlights**: \\textit{PA-Img2Mol}은 CLEF, JPO, UOB, USPTO 특허 벤치마크에서 2.46%에서 8.37%의 정확도를 개선하였고, \\textit{PA-CoreId}는 PatentNetML 벤치마크에서 7.15%에서 7.62%의 향상된 성능을 보여주었습니다. 이러한 결과는 \\texttt{PatentAgent}가 기존 도구보다 얼마나 효과적인지를 강조하고 있습니다.



### Explainable Artificial Intelligent (XAI) for Predicting Asphalt Concrete Stiffness and Rutting Resistance: Integrating Bailey's Aggregate Gradation Method (https://arxiv.org/abs/2410.21298)
Comments:
          The link to web app this https URL this https URL

- **What's New**: 이번 연구는 다양한 집합체(aggregate) 구성을 가진 아스팔트 콘크리트에 대해 해석 가능한 인공지능(XAI) 기법을 활용하여 복원 모듈러스(resilience modulus, MR)와 동적 안정성(dynamic stability, DS)을 분석하였습니다.

- **Technical Details**: 연구는 다층 퍼셉트론(multi-layer perceptron) 아키텍처를 갖춘 딥러닝 모델을 사용하여 Bailey의 방법으로부터 유도된 집합체 구성을 기준으로 MR과 DS를 예측하였으며, 모델 성능은 k-폴드 교차 검증(k-fold cross-validation)을 통해 검증되었습니다. 또한, SHAP(SHapley Additive exPlanations) 값을 사용하여 모델의 예측을 해석하였습니다.

- **Performance Highlights**: 주요 발견으로는 크리티컬 집합체 크기 임계값을 식별하였고, 특히 0.6 mm 체크 사이즈가 MR과 DS에 중요한 영향을 미치는 것으로 나타났습니다. 연구는 굵은 집합체가 러트 저항에 주로 영향을 미치고 중간-미세 집합체가 강성에 영향을 준다는 것을 밝혀냈습니다. 실용성을 위해 웹 기반 인터페이스가 개발되어 MR과 DS 예측이 가능하게 되었으며, 결과의 해석을 위한 해석 가능한 기능이 포함되었습니다.



### Local Policies Enable Zero-shot Long-horizon Manipulation (https://arxiv.org/abs/2410.22332)
Comments:
          Main paper 7 pages, 3 tables, 3 figures. Appendix 6 pages, 2 figures, 6 tables

- **What's New**: ManipGen을 통해 로봇 조작을 위한 새로운 접근 방식을 제시하며, 시뮬레이션에서 실제 환경으로의 전이를 개선하는 방법론을 개발했습니다.

- **Technical Details**: 로컬 정책(local policies)을 도입하여, 객체와 로봇 간의 상대적인 포즈에 대한 불변성과 다양한 장면 구성에 대한 적응성을 확보했습니다.

- **Performance Highlights**: ManipGen은 50개의 실제 조작 과제에서 76%의 성공률을 기록하며 기존 방법들보다 36%에서 최대 76%까지 성능 향상을 보여주었습니다.



### Task Vectors are Cross-Moda (https://arxiv.org/abs/2410.22330)
- **What's New**: 이번 연구에서는 시각-언어 모델(VLMs)의 내부 표현과 이들이 과제 표현(task representations)을 어떻게 인코딩하는지를 조사합니다. 새로운 점은, 유사한 개념의 과제가 어떻게 명세되었는지에 관계없이 유사한 과제 벡터 표현(task vector representations)으로 매핑된다는 것입니다.

- **Technical Details**: 우리는 예시나 지시를 통해 지정된 과제를 고려하며, 텍스트 또는 이미지 입력을 사용합니다. VLM에서 발견된 과제 벡터는 한 모달리티(예: 텍스트)에서 파생되어 다른 모달리티(예: 이미지)로 전이될 수 있을 만큼 일반적입니다. 또한, 예시 기반(exemplar)과 지시 기반(instruction based)의 과제 벡터를 집계(ensembling)하면 더 나은 과제 표현을 제공합니다.

- **Performance Highlights**: 이 연구 결과는 VLM의 설계 메커니즘 및 서로 다른 모달리티와 과제 명세를 통해 과제를 공유하는 방식에 대한 통찰을 제공합니다. VLM이 입력(input), 과제(task), 응답(answer)의 세 가지 독특한 단계로 토큰을 처리하는 과정을 밝혀냈습니다.



### Flow-DPO: Improving LLM Mathematical Reasoning through Online Multi-Agent Learning (https://arxiv.org/abs/2410.22304)
Comments:
          5 pages, 4 figures, 1 table

- **What's New**: 이 논문은 온라인 학습 기법인 Flows를 활용하여 고품질의 수학적 추론 추적을 생성하는 새로운 접근 방식을 제안합니다. 이를 통해 LLM(대형 언어 모델)의 성능을 향상시키는 데 필요한 세부적인 추론 단계를 정확하게 제공하는 것이 목표입니다.

- **Technical Details**: 제안된 방법은 두 개의 LLM(응답 LLM과 정지 LLM)을 사용하는 점진적 출력 생산 흐름을 통해 이루어지며, 이러한 LLM 각각은 자신의 아키텍처에 따라 협력하여 솔루션을 구성합니다. 온라인 DPO(Direct Preference Optimization) 학습을 통해 각 훈련 예시에 대해 DPO 쌍을 생성하고 실시간으로 모델을 업데이트하는 방식입니다.

- **Performance Highlights**: MetaMath 데이터 세트를 사용한 실험 결과, 제안된 방법이 기존 모델 추론을 통한 추론 품질보다 우수하다는 것을 입증하였습니다. 제안된 흐름 접근 방식은 다양한 LLM 모델에 걸쳐 일관된 향상을 보여주었습니다.



### $\mathsf{OPA}$: One-shot Private Aggregation with Single Client Interaction and its Applications to Federated Learning (https://arxiv.org/abs/2410.22303)
Comments:
          To appear at the NeurIPS 2024 FL@FM workshop

- **What's New**: 이번 연구에서는 단일 서버 노드에서 안전한 집합체 집합(Secure Aggregation) 문제를 재검토하여, 많은 클라이언트가 참여하는 시나리오에서 통신 비용을 최소화하기 위한 새로운 방법을 제시합니다.

- **Technical Details**: 우리의 주요 기여는 One-shot Private Aggregation (OPA)입니다. 이 방법에서는 각 클라이언트가 집합체 평가(in aggregation evaluation)당 한 번만 통신하거나 아예 통신하지 않을 수 있습니다. OPA는 LWR, LWE, class groups, DCR를 기반으로 구축되었으며, 클라이언트가 한 번만 통신하는 프라이버시 보호 연합 학습(Federated Learning)에도 적용됩니다.

- **Performance Highlights**: OPA는 기존의 다중 라운드 프라이버시 보호 연합 학습 프로토콜과 비교하여 비약적으로 간편하며 효율성을 극대화합니다. 우리는 두 개의 데이터셋을 대상으로 로지스틱 회귀(classifiers for logistic regression) 성능을 평가하고, MNIST, CIFAR-10, CIFAR-100 데이터셋에 대한 MLP 분류기를 훈련시키며 OPA의 우수성을 입증합니다.



### Emotion-Guided Image to Music Generation (https://arxiv.org/abs/2410.22299)
Comments:
          2024 6th Asian Digital Image Processing Conference

- **What's New**: 이 논문은 감정에 기반하여 이미지를 음악으로 변환하는 새로운 프레임워크를 소개합니다. 기존 모델이 대조 학습(contrastive learning)에 의존하는 것과 달리, 제안된 모델은 감정 일치도를 향상시키기 위해 Valence-Arousal (VA) 손실 함수(loss function)를 직접 통합하였습니다.

- **Technical Details**: 모델은 CNN-Transformer 아키텍처를 사용하며, 미리 학습된 CNN 이미지 특징 추출기와 MIDI 음악에서 복잡하고 고급 감정 특징을 캡처하기 위한 세 개의 Transformer 인코더를 포함합니다. 이전 프로세스를 거쳐 변형된 세 개의 Transformer 디코더는 음악적으로 및 감정적으로 일관된 MIDI 시퀀스를 생성하도록 특징을 정제합니다.

- **Performance Highlights**: 실험 결과, 새롭게 구성된 감정적으로 쌍을 이룬 이미지-MIDI 데이터셋에서 제안된 모델이 Polyphony Rate, Pitch Entropy, Groove Consistency 및 손실 수렴(loss convergence)과 같은 여러 메트릭에서 우수한 성능을 보여주었습니다.



### Batch, match, and patch: low-rank approximations for score-based variational inferenc (https://arxiv.org/abs/2410.22292)
- **What's New**: 본 논문에서는 Black-box Variational Inference (BBVI)의 한계를 극복하기 위해 Batch-and-Match (BaM) 프레임워크를 확장하였습니다. 특히 고차원 문제에 적합한 새로운 접근 방식을 제안하여, 풀 공분산 행렬을 저장하거나 추정하는 것이 비싸고 어려운 경우에도 효과적으로 사용할 수 있도록 하였습니다.

- **Technical Details**: BaM 알고리즘을 통해 목표 밀도(target density)와 다변량 가우시안 근사(Gaussian approximation)의 점수를 일치시키는 데 집중합니다. 이 과정에서, EM 알고리즘에서의 아이디어를 도입하여 각 반복(iteration)마다 새롭게 업데이트된 공분산 행렬을 더 효율적으로 매개변수화된 대각행렬과 저순위 행렬의 합으로 투영하는 패치(patch) 단계를 추가합니다. 이를 통해 pBaM(patched batch-and-match)이라는 새로운 알고리즘을 제안합니다.

- **Performance Highlights**: pBaM 알고리즘은 다양한 합성 타겟 분포(synthetic target distributions)와 고차원 추론(real-world problems in high-dimensional inference) 문제에서 평가되었으며, 기존의 BBVI 접근 방식보다 훨씬 더 나은 성능을 보였습니다.



### Embedding-based classifiers can detect prompt injection attacks (https://arxiv.org/abs/2410.22284)
- **What's New**: 본 논문은 Large Language Models (LLMs)이 출력할 수 있는 유해한 콘텐츠에 대한 독창적인 보호 방법을 제안합니다. 특히 prompt injection 공격을 완화하기 위해 embedding 기반의 머신 러닝(Machine Learning) 분류기를 사용하는 접근 방식을 설명합니다.

- **Technical Details**: 저자들은 OpenAI의 텍스트 임베딩 모델(text-embedding-3-small), GTE 및 MiniLM 등 3가지 임베딩 모델을 활용하여 악의적(prompt injection)과 무해한(benign) 프롬프트의 임베딩을 생성합니다. 이를 통해 ML 분류기를 구축하여 프롬프트를 악의적인지 판단할 수 있게 합니다. 여러 전통적인 ML 방법 중에서 Random Forest와 XGBoost 분류기를 사용하여 가장 우수한 성능을 보였습니다.

- **Performance Highlights**: 제안한 분류기는 오픈 소스 구현에서 사용 가능한 최신의 프롬프트 인젝션 분류기를 초과하는 성능을 보였습니다. 약 467,057개의 프롬프트 데이터 세트를 기반으로 독창적인 방식의 분류기를 훈련시켜 효과적인 결과를 이루어냈습니다.



### Leveraging Recurrent Neural Networks for Predicting Motor Movements from Primate Motor Cortex Neural Recordings (https://arxiv.org/abs/2410.22283)
- **What's New**: 이 논문에서는 비인간 영장류의 신경 기록(neural recordings)에서 운동(motor) 움직임을 해독하기 위한 효율적인 딥러닝(deep learning) 솔루션을 제시합니다. 이 연구에서 사용된 Autoencoder Gated Recurrent Unit (AEGRU) 모델은 훈련 과정에서만 오토인코더(autoencoder)를 사용하여 더 나은 일반화를 달성했습니다.

- **Technical Details**: AEGRU 모델 아키텍처는 Gated Recurrent Unit (GRU)을 기반으로 하며, 신경망의 메모리 발자국(memory footprint)과 계산량을 줄이는 방법이 포함되어 있습니다. 이 모델은 Neurobench를 활용하여 평가되었으며, 입력 피처(input feature)의 시간 지속성이 정확도에 중대한 영향을 미친다고 강조하고 있습니다.

- **Performance Highlights**: 모델은 0.71 $R^2$ 점수를 달성하였으며, 이는 Neurobench의 기본 모델을 초과하여 IEEE BioCAS 2024 Grand Challenge에서 $R^2$ 면에서 1위를 기록했습니다. 또한 모델 가지치기(model pruning)를 통해 41.4%의 multiply-accumulate (MAC) 연산을 줄였으나 $R^2$ 점수는 거의 변화가 없었습니다.



### Pushing the Performance Envelope of DNN-based Recommendation Systems Inference on GPUs (https://arxiv.org/abs/2410.22249)
Comments:
          This work has been accepted in the 57th MICRO (this https URL). Please check appendix for details on reproducing our work including codebase and steps

- **What's New**: 이 논문에서는 Deep Learning Recommendation Models (DLRMs)의 추론(인퍼런스) 성능 개선을 위한 새로운 방법을 제안하며, 주로 GPU에서의 임베딩 단계가 병목 현상(Bottleneck)을 일으킨다는 점을 강조합니다.

- **Technical Details**: DLRM의 네 가지 주요 단계는 임베딩, 바닥 다층 퍼셉트론(Multi-Layer Perceptron), 피처 상호 작용, 그리고 최상위 MLP입니다. 임베딩 단계는 메모리 집약적인 문제로, CUDA 기반의 최신 DLRM을 사용해 테스트한 결과, '무작위' 접근 방식에서 '단일 항목' 접근 방식으로 성능이 3.2배 느려지는 병목 현상이 발견되었습니다. 연구 결과, 레지스터 압박(Register Pressure)으로 인해 최적의 다중 처리(Parallel Processing)를 달성하지 못하는 문제가 있었습니다. 또한, 메모리 지연 문제를 숨기기 위한 소프트웨어 프리패칭(Prefetching)과 L2 핀(Pinning) 방법을 제안했습니다.

- **Performance Highlights**: 제안한 기술들은 A100 GPU와 대형 모델을 사용한 실험에서 임베딩 단계에서 최대 103%, 전체 DLRM 인퍼런스 파이프라인에서 최대 77%의 성능 개선을 나타냈습니다. 각기 독립적으로 프리패칭과 L2 핀은 각각 최대 97%와 62%의 성능 개선을 보여주었으며, 두 방법을 결합할 경우 성능 향상이 더욱 두드러졌습니다.



### DISCERN: Decoding Systematic Errors in Natural Language for Text Classifiers (https://arxiv.org/abs/2410.22239)
Comments:
          20 pages, 9 figures, 15 tables; Accepted to EMNLP 2024

- **What's New**: DISCERN 프레임워크는 텍스트 분류기의 체계적인 오류를 해석하기 위한 자연어 설명을 제공하며, 이를 통해 분류 성능을 향상시킵니다.

- **Technical Details**: DISCERN은 두 개의 대형 언어 모델(LLMs)을 사용하여 반복적으로 체계적인 오류에 대한 자연어 설명을 생성합니다. 설명 모델은 각 클러스터의 특성을 서술하는 간결한 프레디케이트 스타일 설명을 생성하고, 평가 모델은 이 설명이 해당 클러스터의 예제에만 해당하는지를 판단합니다.

- **Performance Highlights**: 세 가지 텍스트 분류 데이터세트에서 DISCERN의 설명을 통해 10% 이상의 오분류율 감소와 25% 이상의 사용자 해석 효율성을 보여줍니다.



### GRINNs: Godunov-Riemann Informed Neural Networks for Learning Hyperbolic Conservation Laws (https://arxiv.org/abs/2410.22193)
Comments:
          29 pages, 6 figures

- **What's New**: 본 연구에서는 GRINNs(Gradient-Recurrent Informed Neural Networks)를 제안하여 비선형 보존법칙 시스템의 역문제를 해결하기 위한 수치 해석 기반의 신경망을 개발하였습니다. 이는 기존 머신러닝 기법과 다른 점으로, 물리적인 유속 함수(physical flux function)를 직접 학습합니다.

- **Technical Details**: GRINNs는 고해상도의 Godunov 스킴을 기반으로 하여 Riemann 문제를 해결하며, Rankine-Hugoniot 조건을 만족하는 근사 Riemann 해법에 기반한 솔루션 연산자를 학습합니다. 이 방법은 불연속성과 연속 영역 모두에서 높은 정확성을 제공하는 특성을 가지고 있습니다.

- **Performance Highlights**: GRINNs는 Burgers', Shallow Water, Lighthill-Whitham-Richards, 그리고 Payne-Whitham 교통 흐름 모델을 포함한 네 가지 벤치마크 문제를 통해 검증되었으며, 이들 모델에서 나타나는 충격파, 희박화, 또는 접촉 불연속성을 매우 높은 정확도로 재현합니다.



### Multi-Level Feature Distillation of Joint Teachers Trained on Distinct Image Datasets (https://arxiv.org/abs/2410.22184)
Comments:
          Accepted at WACV 2025

- **What's New**: 이번 논문에서는 여러 개의 서로 다른 데이터 세트로 훈련된 여러 교사로부터 지식을 증류하는 새로운 teacher-student 프레임워크를 제안합니다. 각 교사는 자신의 데이터 세트로부터 처음부터 훈련받고, 이후 모든 교사의 기능이 결합되어 공동 아키텍처를 형성합니다. 최종적으로, 이 공동 교사로부터의 지식을 student 모델로 전달하기 위해 multi-level feature distillation (MLFD) 절차를 사용합니다.

- **Technical Details**: 제안된 MLFD 방법은 세 가지 단계로 구성됩니다: 1단계에서는 각 교사가 자신의 데이터 세트에서 처음부터 학습됩니다. 2단계에서는 개별 교사들을 결합하여 공동 교사를 생성하며, 이는 모든 데이터 세트에 대해 훈련됩니다. 3단계에서는 집합적 지식을 각 데이터 세트에 맞는 student 모델로 증류합니다. 이러한 방법은 다중 수준의 표현을 사용하여 지식을 추출하고, 학습된 모델의 성능을 높입니다.

- **Performance Highlights**: 이미지 분류 및 동작 인식 실험에서 MLFD 방법이 같은 모델 아키텍처를 가진 기존 모델보다 상당한 성능 향상을 보여줍니다. 또한, 데이터 세트별 학생 아키텍처로부터의 지식 증류가 성능 향상을 가져온다는 점을 입증하였습니다.



### Very Attentive Tacotron: Robust and Unbounded Length Generalization in Autoregressive Transformer-Based Text-to-Speech (https://arxiv.org/abs/2410.22179)
Comments:
          Submitted to NAACL

- **What's New**: 본 논문에서는 오토 리그레시브(AR) Transformer 기반의 텍스트-음성 변환(Text-to-Speech, TTS) 모델의 안정성과 길이 일반화 문제를 해결하기 위한 개선안을 제시합니다. 이를 위해 상대적 위치 정보를 제공하는 정렬 메커니즘을 사용하여 크로스 어텐션(cross-attention) 작업을 수행합니다. 이 정렬 위치는 모델의 잠재적 특성으로 학습되며, 학습 중 외부 정렬 정보가 필요하지 않습니다.

- **Technical Details**: 제안된 시스템인 Very Attentive Tacotron (VAT)은 기본 T5 기반 TTS 아키텍처에 정렬 메커니즘을 추가하여 텍스트-음성 변환 작업의 단조로운 특성을 활용합니다. VAT는 멀티-헤드(Multi-head) 자기-어텐션(self-attention)과 크로스 어텐션의 조합을 통해 강력한 모델링 능력을 유지하면서도 길이 일반화를 가능하게 합니다. VQ-VAE(Variational Quantization Autoencoder)를 사용하여 음성 데이터의 분산화를 수행하며, GAN 기반의 신경 보코더(neural vocoder)를 이용하여 최종 음성을 생성합니다.

- **Performance Highlights**: VAT는 자연스러움과 표현력이 뛰어난 T5-TTS 기본 시스템과 동등한 성능을 발휘하면서도 중복되거나 삭제된 단어 문제를 제거하고, 실제 발화 길이에 대해 일반화할 수 있는 능력을 보여줍니다.



### EconoJax: A Fast & Scalable Economic Simulation in Jax (https://arxiv.org/abs/2410.22165)
Comments:
          8 pages

- **What's New**: EconoJax라는 새로운 다중 에이전트 경제 환경을 소개하며, 이는 JAX로 완전히 작성되어 빠른 훈련 속도와 확장성을 제공합니다. 이전의 AI economist 프레임워크와 비교하여 훈련에 필요한 시간이 15분으로 단축되었습니다.

- **Technical Details**: EconoJax는 리인포스먼트 러닝 (Reinforcement Learning) 에이전트를 위한 벡터 기반 환경으로, 자원 수집, 거래 및 세금 부과로 경제 행동을 시뮬레이션합니다. JAX를 사용하여 코드 작성 및 훈련을 통합하여, CPU, GPU, TPU에서의 효율적인 연산이 가능합니다. 이 시스템은 실시간으로 경제 행동을 모델링하며, 에이전트들이 훈련을 통해 세금 일정 및 생산성 균형 등의 현실적인 경제 행동을 나타내도록 합니다.

- **Performance Highlights**: EconoJax는 100명의 에이전트로 구성된 실험에서 15분 이내에 경제 행동을 나타내며, 기존 AI economist 프레임워크에서는 며칠이 소요되었습니다. EconoJax는 대규모 실험을 가능하게 하여 연구자들이 더 풍부하고 동적인 경제 시뮬레이션을 구축할 수 있도록 지원합니다.



### Deep Q-Exponential Processes (https://arxiv.org/abs/2410.22119)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문은 Q-Exponential Process (Q-EP)를 심층적으로 확장하여 Deep Q-EP 모델을 제안합니다. Deep Q-EP는 불균형 데이터셋을 더 효과적으로 처리할 수 있는 정규화 기능을 제공합니다.

- **Technical Details**: Deep Q-EP는 여러 층의 Q-EP를 쌓아 올려 더욱 복잡한 잠재 표현을 학습할 수 있도록 설계되었습니다. 또한, 변량 추론(variational inference)을 위해 sparse approximation과 inducing points를 활용합니다.

- **Performance Highlights**: Deep Q-EP는 기존의 여러 최신 딥 확률 모델들과 비교하여 데이터를 더 잘 처리하는 수치적 우수성을 보여줍니다.



### The Impact of Inference Acceleration Strategies on Bias of LLMs (https://arxiv.org/abs/2410.22118)
- **What's New**: 최근 몇 년간 대규모 언어 모델(LLMs)의 성능이 비약적으로 발전했습니다. 그러나 이러한 모델의 inference(추론) 효율성을 높이기 위한 많은 전략들(quantization, pruning 등)이 도입되었음에도 불구하고, 이 과정에서 데이터의 인구 통계학적 편향이 유의미하게 바뀔 수 있다는 점이 주목됩니다.

- **Technical Details**: 우리는 5가지 일반적인 inference acceleration techniques(추론 가속 기술)과 3개의 널리 사용되는 LLM을 통해 편향의 변화를 평가하였습니다. 특히, 모델 출력의 편향을 측정하기 위해 6가지 다른 bias metrics(편향 지표)를 활용하였으며, 각 모델에서 가속화 전략의 영향을 분석했습니다.

- **Performance Highlights**: 분석 결과, inference acceleration 전략이 모델의 편향에 중대한 영향을 미친다는 것을 확인했습니다. 예를 들어, 4-bit AWQ Quantization 방식이 일부 모델의 편향을 크게 변화시켰으나 KV-cache quantization 방식은 상대적으로 견고한 결과를 보여주었습니다. 따라서 가속화를 적용할 때 편향에 대한 신중한 평가가 필요하다는 점이 강조됩니다.



### Joint Extraction and Classification of Danish Competences for Job Matching (https://arxiv.org/abs/2410.22103)
- **What's New**: 이 연구는 덴마크의 구직 공고에서 능력을 자동으로 추출하고 분류하는 첫 번째 모델을 제시합니다. 기존 연구와 달리, 이 모델은 대규모의 주석이 달린 덴마크 말코퍼스를 기반으로 훈련되었습니다.

- **Technical Details**: 모델은 BERT-like 아키텍처를 기반으로 하며, 직무 공고에서의 능력 추출 및 분류를 단일 아키텍처로 수행합니다. 또한 명명된 개체 인식(NER) 라벨을 생성하여 직무 능력을 추출합니다. 데이터는 20만 개의 문장에서 유럽의 기술, 역량, 자격 및 직업(ESCO) 주석이 추가되어 있습니다.

- **Performance Highlights**: 이 모델은 덴마크 능력 추출 및 분류에서 최첨단 모델들을 초과하는 성능을 보였으며, 추론 기간을 50% 이상 단축시켰습니다.



### Variational inference for pile-up removal at hadron colliders with diffusion models (https://arxiv.org/abs/2410.22074)
Comments:
          19 pages, 13 figures

- **What's New**: 이 논문에서는 pile-up 제거를 위한 새로운 방법인 Vipr(Variational Inference for Pile-up Removal)를 소개합니다. 기존의 분류(classification) 방법과 달리, 이 방법은 생성 모델(generative model)을 훈련하여 하드 스캐터 제트의 성분을 예측합니다.

- **Technical Details**: Vipr는 확산 모델(diffusion models)을 사용하여 관측된 제트에서 pile-up에 의해 오염된 하드 스캐터 제트의 진짜 성분을 추론합니다. 또한, Vipr는 제트 성분에 대한 전체 posterior distribution을 근사하여 다양한 관측 가능한 값들을 생성할 수 있습니다.

- **Performance Highlights**: Vipr는 SoftDrop과 비교할 때 하드 스캐터 제트의 서브스트럭처(substructure)를 예측하는 데 있어 다양한 pile-up 시나리오에서 뛰어난 성능을 보여줍니다.



### FreeGaussian: Guidance-free Controllable 3D Gaussian Splats with Flow Derivatives (https://arxiv.org/abs/2410.22070)
- **What's New**: 본 논문에서는 수동 주석 없이 동적으로 제어 가능한 Gaussian splats를 복원하는 새로운 방법인 FreeGaussian을 소개합니다. 이 방법은 광학 흐름(optical flow)과 카메라 모션(camera motion)을 기반으로 하여 동적 Gaussian 운동을 수학적으로 유도합니다.

- **Technical Details**: FreeGaussian은 2D 이미지의 광학 흐름과 카메라에 의해 유도된 장면 흐름을 기반으로 동적 Gaussian을 직접 도출하고, 이를 통해 상호작용하는 동적 3D Gaussian 구조를 추적합니다. 또한, 3D 구형 벡터 제어 방식(3D spherical vector controlling scheme)을 도입하여 복잡한 1D 제어 신호 계산을 피하고, Gaussian 동적 모션을 보다 쉽게 모델링할 수 있게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FreeGaussian은 새로운 관점 합성과 장면 제어에서 기존 방법보다 우수한 성능을 보여주며, 수동 주석 없이도 상호작용 가능한 콘텐츠를 보다 정확하고 효율적으로 모델링할 수 있음을 입증하였습니다.



### Hamiltonian Monte Carlo on ReLU Neural Networks is Inefficien (https://arxiv.org/abs/2410.22065)
Comments:
          Paper published at NeurIPS 2024

- **What's New**: 새로운 연구에서는 ReLU 기반의 신경망에 대한 Hamiltonian Monte Carlo (HMC) 알고리즘의 오류율을 분석하여, 기존의 smooth한 에너지 함수와 비교했을 때 HMC의 비효율성을 입증했습니다.

- **Technical Details**: HMC 알고리즘은 leapfrog integrator를 사용하며, ReLU 활성화 함수의 비미분 가능성으로 인해 오류율이 Ω(ε)로 증가하여 제안의 거부율이 높아진다고 합니다. 이 연구에서는 HMC의 이론적 조건 및 Hamiltonian dynamics의 오류를 𝒪(ε)와 Ω(ε) 형태로 제한하는 것을 고찰합니다.

- **Performance Highlights**: 실험을 통해 ReLU 기반 신경망의 HMC 추론이 기존의 분석적 신경망과 비교하여 비효율적임을 수치적으로 확인했습니다. 새롭게 제안된 HMC 튜닝 가이드라인에서는 단계 크기를 d^{-1/2}로 설정하고, 최적의 수용 확률을 0.45로 설정하는 것을 추천합니다.



### CHORDONOMICON: A Dataset of 666,000 Songs and their Chord Progressions (https://arxiv.org/abs/2410.22046)
- **What's New**: Chordonomicon 데이터셋을 소개하고 있으며, 이는 666,000곡 이상의 노래와 그에 따른 코드 진행을 포함합니다. 이 데이터셋은 구조적 부분, 장르 및 출시 날짜와 함께 주석이 달려 있으며, 다양한 사용자 생성 코드 진행과 관련 메타데이터를 수집하여 작성되었습니다.

- **Technical Details**: Chordonomicon 데이터셋은 기존의 대규모 데이터셋에 비해 20배 이상의 코드 진행 정보를 포함하고 있으며, 음악 이론 지식을 포괄하는 형식적 표현을 통해 머신 러닝 및 지식 기반 시스템의 결합을 탐구할 수 있는 기회를 제공합니다. 코드 진행은 text와 graph 같은 다양한 포맷으로 표현될 수 있습니다.

- **Performance Highlights**: Chordonomicon 데이터셋은 분류 및 생성 작업에 실질적인 유용성을 보여주며, 고급 머신 러닝 기법을 탐구하기 위한 이상적인 테스트베드로 평가됩니다. 이 데이터셋은 실험의 재현성 및 비교를 위한 공통 기준을 제공하여 기술적 발전에 기여할 것으로 기대됩니다.



### On uniqueness in structured model learning (https://arxiv.org/abs/2410.22009)
- **What's New**: 이 논문은 부분 미분 방정식(Partial Differential Equations, PDE)에 대한 물리 법칙 학습의 유일성 문제를 다루고 있습니다. 기존의 접근 방식과는 달리, 기존의 약간 정확한 물리 모델이 데이터에서 학습된 구성 요소로 보강되는 구조화된 모델 학습 프레임워크를 고려하고 있습니다.

- **Technical Details**: 주요 결과는 특정 클래스의 PDE 및 알려지지 않은 모델 구성 요소를 근사하는 데 사용되는 적절한 클래스의 신경망에 대한 유일성 결과입니다. 논문은 문제가 되는 측정이 완전하고 무잡음인 경우, PDE 시스템의 정규화 최소화 솔루션으로서 알려지지 않은 모델 구성 요소의 유일한 식별이 가능할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 논문은 불완전하고 노이즈가 있는 측정 기반으로 학습된 모델 구성 요소가 한계에서 진실 모델 구성 요소에 근접하게 된다는 수렴 결과도 제공합니다. 이러한 결과는 근사 신경망의 특정 속성과 정규화를 조정하여 가능합니다.



### A Machine Learning-Based Secure Face Verification Scheme and Its Applications to Digital Surveillanc (https://arxiv.org/abs/2410.21993)
Comments:
          accepted by International Conference on Digital Image and Signal Processing (DISP) 2019

- **What's New**: 이 논문은 얼굴 이미지의 개인 식별 보호와 보안 강화를 위한 새로운 얼굴 인증 시스템을 제안합니다. 기존 시스템의 단점으로 지적된 개인 정보 보호 문제를 해결하기 위해, DeepID2을 이용한 CNN 모델과 EM 알고리즘을 적용하여 안전하게 얼굴 인증을 수행합니다.

- **Technical Details**: 논문에서 제안한 방법은 DeepID2 (Deep Identification-verification network version two)라는 심층 신경망을 활용하여 얼굴 이미지의 특징을 추출하고, EM (Expectation-Maximization) 알고리즘을 사용하여 두 이미지가 동일 인물로부터 촬영되었는지를 검증합니다. 또한, 개인 정보를 보호하기 위해 동형 암호화 (homomorphic encryption) 기술을 적용하여 암호화된 상태에서 얼굴 특징을 계산합니다.

- **Performance Highlights**: 연구팀은 세 가지 개인 정보 보호 수준에 따라 지역 사회의 출입 통제를 위한 세 가지 얼굴 인증 시스템을 개발했습니다. 실험 결과는 제안된 시스템의 실용성을 입증하며, 암호화된 데이터에 대한 효율적인 연산이 가능함을 보여줍니다.



### Node Regression on Latent Position Random Graphs via Local Averaging (https://arxiv.org/abs/2410.21987)
- **What's New**: 이번 논문은 무작위 그래프에서 노드 회귀(node regression) 문제를 다루며, Latent Position Model (LPM)에 기반한 이론적 연구를 수행합니다.

- **Technical Details**: 그래프 회귀를 위한 가장 간단한 추정기인 인접 노드의 레이블을 평균내는 방법을 설명합니다. 이 추정기는 Latent Position Models에서 Nadaraya-Watson 추정기로 수렴하며, 그 수렴 속도 또한 동일합니다. 기존의 추정기와 대안 방법을 제시하여, 잠재적인 거리 추정을 통한 성능 향상을 도모합니다. 이를 통해 그래프 이웃보다 더 작거나 큰 영역에서 평균화가 가능합니다.

- **Performance Highlights**: 제안된 방법은 그래프 구조에 대한 영향을 분리하여, 특정 상황에서는 표준 비모수(nonparametric) 속도를 달성할 수 있음을 보입니다.



### Individualised recovery trajectories of patients with impeded mobility, using distance between probability distributions of learnt graphs (https://arxiv.org/abs/2410.21983)
- **What's New**: 이 논문은 개인의 신체 재활과정을 반영하는 회복 궤적(recovery trajectory) 학습 방법을 제안합니다. 특히 케어가 필요한 환자들에게 개인 맞춤형 운동 루틴을 제시할 수 있는 기술적인 툴을 개발하였습니다.

- **Technical Details**: 환자의 관절 위치에 대한 다변량 시계열 데이터로부터 Bayesian 방법론을 활용하여 무작위 그래프(random graph)를 학습하고, 이를 기반으로 한 Movement Recovery Scores (MRS)를 계산합니다. 쿼리 간의 posterior 확률(env) 간의 통계적 거리(divergence)를 통해 회복 정도를 측정합니다.

- **Performance Highlights**: 본 연구는 환자 개개인의 운동 능력 회복을 평가하는 새로운 방법을 제시하며, 이를 통해 환자의 운동 재활 과정에서 실시간 피드백을 제공할 수 있게 되어, 신체 재활치료의 질을 향상시키는 효과를 기대할 수 있습니다.



### ReMix: Training Generalized Person Re-identification on a Mixture of Data (https://arxiv.org/abs/2410.21938)
Comments:
          Accepted by WACV 2025

- **What's New**: 본 연구는 제한된 라벨이 있는 다중 카메라 데이터와 대규모 비라벨 단일 카메라 데이터를 혼합하여 공동 학습하는 새로운 방법인 ReMix를 제안합니다. 이 방법은 이전의 self-supervised pre-training 접근법의 한계를 극복하고, 더 나은 generalization 능력을 달성하도록 설계되었습니다.

- **Technical Details**: ReMix는 두 가지 유형의 데이터: 라벨이 있는 다중 카메라 데이터와 비라벨 단일 카메라 이미지를 효율적으로 혼합하고 공동 학습하기 위해 새로운 데이터 샘플 전략과 세 가지 새로운 손실 함수 (Instance Loss, Augmentation Loss, Centroids Loss)를 제안합니다. 이러한 손실 함수들은 두 종류의 데이터에 맞게 조정되어 훈련의 질을 높입니다.

- **Performance Highlights**: 실험 결과, ReMix는 cross-dataset 및 multi-source cross-dataset 시나리오에서 기존의 최첨단 방법보다 성능이 우수하며, generalizable person Re-ID에서 높은 generalization 능력을 보여줍니다.



### Differentiable Inductive Logic Programming for Fraud Detection (https://arxiv.org/abs/2410.21928)
- **What's New**: 본 논문에서는 설명 가능한 AI(Explainable AI) 접근법인 미분 유도 논리 프로그래밍(Differentiable Inductive Logic Programming, DILP)을 사기 탐지(Fraud Detection) 분야에 적용하는 연구를 수행하였습니다. DILP의 스케일 문제를 해결하기 위한 데이터 정제(data curation) 기법을 제안하며, 기존 전통적 방법들과 비교하여 DILP의 성능을 평가합니다.

- **Technical Details**: DILP는 인덕티브 논리 프로그래밍(Inductive Logic Programming, ILP)의 변형으로, 데이터셋에서 규칙을 도출하는 머신러닝 방법입니다. 연구에서는 합성 데이터셋을 생성하여 DILP 성능을 평가하고, 이후 PaySim 데이터셋에서 훈련 및 평가를 진행하였습니다. DILP는 의사결정 나무(Decision Trees, DTs) 및 심볼릭 분류(Deep Symbolic Classification)와 같은 기존 방법들과 비교하여 성능을 분석합니다.

- **Performance Highlights**: DILP는 기존의 전통적인 방법들에 비해 규칙 생성에 강점을 가지며, 작은 데이터셋에서도 일반화할 수 있는 능력을 가지고 있습니다. 하지만, 데이터 전처리(data preprocessing) 작업이 필요한 점은 DILP의 적용에 있어 도전 요소로 작용합니다. 연구 결과, DILP는 전통적인 방법들과 유사한 성능을 보였으며, 특히 규칙 수를 늘리고 관계를 탐지하는 데 있어 유용하다는 점에서 가능성을 보여주었습니다.



### Online Test of a Neural Network Deep Convection Parameterization in ARP-GEM1 (https://arxiv.org/abs/2410.21920)
Comments:
          10 pages, 5 figures, submitted to Artificial Intelligence for the Earth Systems

- **What's New**: 이 연구는 ARP-GEM1이라는 글로벌 대기 모델에 신경망 기반의 파라미터화를 통합하는 방법을 제시합니다. 이 방식은 Python 인터페이스를 활용하여 Fortran 기반의 ARP-GEM1 모델과 신경망 추론을 담당하는 Python 구성 요소 간의 필드 교환을 용이하게 합니다.

- **Technical Details**: 신경망(Neural Network)을 사용하여 ARP-GEM1의 깊은 대류(deep convection) 파라미터화를 모방하는 실험을 진행하였으며, Fortran/Python 인터페이스를 통해 성공적으로 신경망 에뮬레이터로 대체했습니다. 여러 해의 ARP-GEM1 시뮬레이션 결과, 신경망 기반의 대류 모델이 기존 물리 기반 대류 모델과 좋은 일치를 보였습니다.

- **Performance Highlights**: 평균 필드의 평가 결과, 신경망 심층 대류 모델이 기존의 물리 기반 대류 모델과 거의 일치함을 보여주었습니다. 또한, 신경망의 추론 속도를 높이기 위해 GPU를 사용하여 모델의 온라인 성능을 평가했습니다.



### Identifiability Analysis of Linear ODE Systems with Hidden Confounders (https://arxiv.org/abs/2410.21917)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문은 숨겨진 혼란 변수가 포함된 선형 상미분 방정식(ODE) 시스템의 가시성 분석을 제시합니다. 특히, 잠재적인 혼란 변수가 시스템과 상호작용할 때의 가시성 조건을 탐구하여 이 분야에서의 공백을 해소하고자 합니다.

- **Technical Details**: 연구에서는 두 가지 경우의 시스템을 다루고 있습니다. 첫 번째 경우는 잠재 혼란 변수가 인과 관계가 없지만, 그 진화는 특정 함수 형식(예: 시간 t에 대한 다항식 함수)을 따릅니다. 두 번째 경우는 숨겨진 혼란 변수가 인과적 종속성을 갖는 더 복잡한 상황을 분석합니다. 이를 위해 유도 비순환 그래프(DAG)를 사용하여 혼란 변수의 인과 구조를 설명합니다. 각 사례에 대하여 다양한 관측 상태에서 가시성 분석을 수행하고, 시뮬레이션을 통해 이론적 결과를 검증하였습니다.

- **Performance Highlights**: 시뮬레이션 결과, 가시성이 확보된 경우 샘플 수가 증가할수록 평균 제곱 오차(MSE)가 감소하여 0에 접근하였습니다. 반면, 가시성이 확보되지 않은 경우에는 MSE가 높고 샘플 수와 무관한 결과를 보였습니다. 이러한 결과는 제시한 가시성 조건의 유효성을 강력히 뒷받침합니다.



### SceneGenAgent: Precise Industrial Scene Generation with Coding Agen (https://arxiv.org/abs/2410.21909)
- **What's New**: 이 논문에서는 산업 제조의 시뮬레이션을 위한 산업 장면 모델링의 중요성을 강조하며, 새로운 LLM 기반 에이전트인 SceneGenAgent를 소개합니다.

- **Technical Details**: SceneGenAgent는 C# 코드를 통해 산업 장면을 생성하며, 정확한 배치 계획(layout planning)을 위해 구조적이고 계산 가능한 형식을 사용합니다. 레이아웃 검증(layout verification)과 반복적 개선(iterative refinement)을 통해 정량적 요구 사항을 충족합니다.

- **Performance Highlights**: SceneGenAgent는 산업 장면 생성 작업에서 최대 81.0% 성공률을 기록하며, 대부분의 장면 생성 요구를 효과적으로 충족합니다. 또한, SceneInstruct 데이터셋을 통해 오픈 소스 LLM을 미세 조정(fine-tuning)하면 성능이 크게 향상되는 것을 보여주었습니다.



### Hierarchical mixtures of Unigram models for short text clustering: the role of Beta-Liouville priors (https://arxiv.org/abs/2410.21862)
Comments:
          47 pages, 4 figures. Submitted

- **What's New**: 이 연구에서는 짧은 텍스트 데이터의 비지도 분류(unsupervised classification)를 위해 조정된 다항 혼합 모델(Multinomial mixture model)의 변형을 소개합니다. 이를 통해 더 유연한 상관 관계 구조를 제공하는 베타-리우빌 분포(Beta-Liouville distribution)를 탐구합니다.

- **Technical Details**: 베타-리우빌 분포는 다항 likelihood와의 공액(conjugate) 관계를 가집니다. 이를 통해 CAVI(좌표 상승 변분 추론) 알고리즘에 대한 업데이트 방정식을 도출할 수 있으며, 모델 파라미터의 근사 후속 추정을 용이하게 합니다. 또한, 확장성이 향상된 CAVI 알고리즘의 확률적 변형(stochastic variant)을 제안합니다.

- **Performance Highlights**: 짧은 텍스트로 구성된 데이터 세트를 사용하여 베타-리우빌 혼합 모델과 디리클레-다항 혼합 모델(Dirichlet-Multinomial) 간의 비교를 수행하였으며, 비지도 분류의 정확도를 측정하여 베타-리우빌 모델의 상대적 장점을 확인했습니다.



### Joint Estimation of Conditional Mean and Covariance for Unbalanced Panels (https://arxiv.org/abs/2410.21858)
- **What's New**: 본 논문에서는 대규모 비균형 패널을 위한 새로운 비모수적 커널 기반 추정기를 제안합니다. 이 추정기는 조건부 평균과 공분산 행렬을 동시에 추정할 수 있으며, 기존의 방법들에 비해 구체적인 샘플 개수에 대한 보장을 제공합니다.

- **Technical Details**: 이 모델은 자산의 특성을 함수로 하는 조건부 순간(conditional moments)의 비모수적 특징을 가지고 있으며, 모든 상태에서도 대칭적이고 양수 준정적(positive semidefinite) 공분산 행렬을 유지합니다. 논문에서 제안된 방법은 개인용 컴퓨터에서 몇 시간 내에 모든 실증적 연습을 완료할 수 있을 만큼 빠르고 효율적입니다.

- **Performance Highlights**: 제안된 모델은 1962년부터 2021년까지의 미국 주식 월수익률을 기반으로 실증적 포트폴리오를 생성하며, 연간 Sharpe 비율이 균형 포트폴리오를 초과하는 뛰어난 성과를 보여줍니다. 이러한 성과는 Fama-French 5 요인 모델과의 관계에서 그 요인 수가 증가함에 따라 점차 약해진다는 점도 주목할 만합니다.



### Exponentially Consistent Statistical Classification of Continuous Sequences with Distribution Uncertainty (https://arxiv.org/abs/2410.21799)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.01161

- **What's New**: 이번 연구에서는 연속적인 시퀀스의 분포 불확실성을 고려하여 다중 분류(multiple classification) 문제를 다뤘습니다. 기존 연구들이 이산값 시퀀스의 완벽한 분포 일치에 초점을 맞춘 반면, 본 논문은 실제 애플리케이션에서의 불확실성을 해결하기 위해 분포에 구애받지 않는 테스트(distribution free tests)를 제안합니다.

- **Technical Details**: 연구는 세 가지 테스트 설계인 고정 길이(fixed-length), 순차(sequential), 그리고 두 단계(two-phase) 테스트에 대해 오차 확률이 신속하게 지수적으로 감소함을 증명합니다. 고정 길이 테스트는 각 훈련 및 테스트 시퀀스의 샘플 크기를 고정하며, 순차 테스트는 테스트 시퀀스에서 샘플을 순차적으로 수집하여 안정적인 결정을 내립니다. 두 단계 테스트는 두 고정 값 사이에서 훈련 및 테스트 시퀀스의 샘플 크기를 조정합니다.

- **Performance Highlights**: 행렬 분석을 통해 세 가지 테스트 설계 중 순차 테스트가 고정 길이 테스트보다 더 큰 지수를 달성하는 것을 보여주었습니다. 또한 두 단계 테스트는 고정 길이 테스트의 성능과 순차 테스트에 가까운 성능을 제공하면서 테스트 설계 복잡도를 줄일 수 있는 좋은 균형을 나타냅니다.



### MARCO: Multi-Agent Real-time Chat Orchestration (https://arxiv.org/abs/2410.21784)
Comments:
          EMNLP 2024 Industry Track

- **What's New**: MARCO는 LLM을 이용한 다중 에이전트 실시간 채팅 조정 프레임워크로, 복잡한 작업을 자동화하는 새로운 접근 방식을 제시합니다. 특히 여러 도구와의 상호작용 및 다중 단계의 작업 실행을 지원하여 실시간으로 사용자와 상호작용할 수 있는 기능을 탑재하고 있습니다.

- **Technical Details**: MARCO는 LLM의 불확실성과 오류를 극복하기 위해 견고한 가드레일(guardrails)을 도입하였으며, 신뢰성 높은 작업 실행을 위해 자연어로 표현된 작업 실행 절차(TEP)를 사용합니다. 에이전트 간 장기 메모리를 공유하는 구조로 설계되어, 동적인 정보와 대화 상황을 포함한 완전한 컨텍스트 정보를 저장합니다. 다양한 LLM 모델을 비교하여 최적의 성능을 도출할 수 있도록 구성되어 있습니다.

- **Performance Highlights**: 테스트 결과, MARCO는 Digital Restaurant Service Platform 대화 및 Retail 대화 데이터셋에서 각각 94.48%와 92.74%의 높은 정확도를 기록했으며, 44.91%의 지연 개선(latency improvement)과 33.71%의 비용 감소(cost reduction)를 달성했습니다. 가드레일의 효과를 통해 성능 향상이 이루어졌으며, 여러 LLM 모델에 대한 비교 분석도 포함되어 있습니다.



### Efficient Reprogramming of Memristive Crossbars for DNNs: Weight Sorting and Bit Stucking (https://arxiv.org/abs/2410.21730)
Comments:
          5 pages, 10 figures

- **What's New**: 이 논문에서는 메모리스터(membristor)의 재프로그래밍 횟수를 줄이기 위한 새로운 접근 방식을 소개합니다. 이 방법은 DNN(deep neural networks)에서의 메모리스터의 비휘발성 메모리 내구성 한계를 해결하는 데 초점을 맞추고 있습니다. 주목할 만한 두 가지 기술을 사용하여 재프로그래밍 요구 사항을 감소시킵니다.

- **Technical Details**: (1) 정렬된 가중치 섹션(sorted weight sectioning)을 사용하여 비슷한 크로스바(crossbar)의 재프로그래밍을 스케줄링하고 메모리스터 상태 재사용을 극대화합니다. (2) 낮은 차수 열의 랜덤하게 선택된 메모리스터의 일부만 재프로그래밍하여 비트 수준 분포를 활용하고 모델 정확도에 미치는 작은 영향을 인식합니다.

- **Performance Highlights**: ImageNet-1K 데이터셋에서 ResNet-50의 경우 3.7배, ViT-Base의 경우 21배의 크로스바 재프로그래밍 횟수 감소를 보이며 모델 정확도는 1% 이하의 범위 내에서 유지됩니다.



### On the Statistical Complexity of Estimating VENDI Scores from Empirical Data (https://arxiv.org/abs/2410.21719)
- **What's New**: 이 논문은 기계 학습 커뮤니티에서 최근 연구된 참조가 없는 생성 모델 평가 지표인 VENDI score에 대해 다루고 있습니다. VENDI score는 정보 이론의 행렬 기반 엔트로피를 사용하여 생성 모델의 다양성을 정량화합니다. 이 논문에서는 유한한 샘플 크기에서 VENDI score의 통계적 수렴성을 탐구하고, t-truncated VENDI 통계량이라는 대체 통계량을 도입합니다.

- **Technical Details**: VENDI score는 n×n 커널 행렬의 고유분해를 통해 계산되며, 여기서 n은 생성된 샘플의 수입니다. 그러나 n이 클 경우, 고유분해의 계산 비용이 급격히 증가하여 수천 개로 제한됩니다. 이 논문에서는 VENDI score의 계산에서 고유값 중 상위 t개만을 사용하는 t-truncated VENDI 통계량을 제안하고, 이를 통해 더 적은 샘플 크기로 통계량을 계산할 수 있음을 보입니다. 나아가 Nyström 방법과 FKEA 근사 방법이 정의된 절단 VENDI 통계량으로 수렴한다는 것을 증명합니다.

- **Performance Highlights**: 본 논문은 실험을 통해 제한된 n 크기의 샘플로 계산된 VENDI score와 우리의 정의된 t-truncated VENDI 통계량 간의 관계를 검증하며, 특히 유한 차원 커널 함수의 경우 VENDI score가 VENDI 통계량으로 효율적으로 수렴함을 확인했습니다. 무한 차원 Gaussian 커널 함수의 경우, 샘플 수가 10,000을 초과한 경우에도 점점 증가하는 VENDI score의 특성을 관찰하였습니다.



### Minimax optimality of deep neural networks on dependent data via PAC-Bayes bounds (https://arxiv.org/abs/2410.21702)
- **What's New**: 이번 논문에서는 Schmidt-Hieber (2020)의 결과를 확장하여 i.i.d. (독립 동분포) 가정을 제거하고, 비독립적인 관측치를 사용하는 설정에서의 딥 신경망(DNN) 기반의 최소 제곱 회귀 추정에 대한 새로운 결과를 제시합니다.

- **Technical Details**: 관측치는 비어 있는 pseudo-spectral gap을 가진 마코프 체인으로 가정하며, 최소 제곱 회귀와 로지스틱 회귀를 포함하는 보다 일반화된 기계 학습 문제를 연구합니다. PAC-Bayes 오라클 불평등 및 Bernstein 불평등을 활용하여 일반화된 베이지안 추정기의 추정 리스크에 대한 상한을 도출합니다.

- **Performance Highlights**: 최소 제곱 회귀의 경우, 도출된 상한은 Schmidt-Hieber (2020)의 하한과 일치하며, 로지스틱 손실 분류에서도 유사한 하한을 established합니다. proposed DNN 추정기가 minimax 차원에서 최적임을 증명합니다.



### The Effects of Multi-Task Learning on ReLU Neural Network Functions (https://arxiv.org/abs/2410.21696)
- **What's New**: 이 논문은 다중 작업(multi-task) 얕은 ReLU 신경망 학습 문제의 해에 대한 속성을 연구하며, 각 작업에 대해 학습된 해결책이 커널 방법(kernel method)을 통해 얻은 것과 유사한 특성을 지닌다는 점에서 신경망과 커널 방법之间의 새로운 연결성을 밝혔습니다.

- **Technical Details**: 우리는 다변량 입력(multi-variable input) 및 단변량 입력(univariate-input)의 다중 작업 신경망 보간(interpolation) 문제에 대한 해결책이 거의 항상 유일하다는 것을 증명하였으며, 이들은 소벨(Sobolev) 힐베르트 공간(Hilbert Space)에서 최소 노름 보간(minimum-norm interpolation) 문제의 해와 일치합니다. 또한, 다양한 작업이 많은 신경망 학습 문제는 최적의 뉴런에 의해 결정된 고정 커널(fixed kernel)에 대한 ℓ^2(Hilbert space) 최소화 문제와 대략적으로 동등하다는 사실을 보여주었습니다.

- **Performance Highlights**: 우리는 멀티태스크 신경망 학습 문제가 단일 작업 학습 문제와는 다르게 작동하며, 각 작업이 서로 유사하지 않더라도 전혀 다른 해결책을 제공한다는 것을 입증하였습니다. 이는 멀티태스크 학습에 대한 기존 이론과의 차별점으로, 단일 작업 간의 유사성에 기반하지 않는 성과입니다.



### CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs (https://arxiv.org/abs/2410.21695)
- **What's New**: 새로운 CFSafety 벤치마크를 소개하며, 5개의 고전적인 안전 시나리오와 5가지의 명령 공격 유형을 포함한 총 10개 카테고리의 안전 질문을 결합하여 25,000개의 프롬프트로 테스트 세트를 구성했습니다.

- **Technical Details**: CFSafety는 자연어 생성(NLG) 능력을 평가하기 위해 0-5의 안전 등급 척도와 간단한 도덕적 판단을 결합하여 사용합니다. 이 벤치마크를 통해 OpenAI의 GPT 시리즈를 포함한 8개의 인기 있는 LLM의 안전성을 테스트했습니다.

- **Performance Highlights**: GPT-4는 비교 우수한 안전 성능을 보여주었으나, 다른 LLM 모델들의 안전성은 여전히 개선이 필요하다는 결과를 도출했습니다.



### Revisiting Reliability in Large-Scale Machine Learning Research Clusters (https://arxiv.org/abs/2410.21680)
- **What's New**: 이 논문은 대규모 머신러닝 인프라의 안정성 문제를 다루며, 11개월간의 데이터를 분석하여 다양한 GPU 규모에서의 고장 패턴과 신뢰성 메트릭을 소개합니다.

- **Technical Details**: 저자들은 고성능 컴퓨팅(High-Performance Computing, HPC) 스택 위에 Slurm 스케줄러를 사용하는 두 개의 대형 ML 클러스터(RSC-1, RSC-2)를 운영하며, 전체 GPU 자원에서 큰 작업의 비율이 낮지만 GPU 사용량은 높다는 점을 강조합니다. 또한, Mean Time to Failure (MTTF) 예측 모델과 Effective Training Time Ratio를 도출하여 훈련 효율성을 평가합니다.

- **Performance Highlights**: RSC-1과 RSC-2 클러스터는 각각 평균 83%, 85%의 활용도를 기록하며, 4천 개 이상의 GPU를 사용하는 작업의 안정성을 높이기 위한 다양한 소프트웨어 완화 방법을 제안합니다.



### $f$-PO: Generalizing Preference Optimization with $f$-divergence Minimization (https://arxiv.org/abs/2410.21662)
- **What's New**: 이 논문은 언어 모델의 인간 선호 최적화를 위한 새로운 프레임워크인 $f$-divergence Preference Optimization ($f$-PO)을 소개합니다. 이 방법은 기존의 여러 접근 방식을 일반화하고 확장하여 다양한 $f$-다이버전스(f-divergence)를 사용하는 정렬 방법의 광범위한 가족을 포함합니다.

- **Technical Details**: $f$-PO는 정책 모델과 최적 모델 간의 $f$-다이버전스 최소화를 통해 선호 최적화를 분포 일치(problem)로 모델링합니다. 이는 DPO(Direct Preference Optimization)와 EXO를 재현하며, 다양한 $f$-다이버전스를 선택하여 새로운 변형을 제공합니다. 또한, 다양한 $f$-다이버전스의 영향에 대한 세부 분석을 통해 오프라인 선호 최적화에서 정규화와 성능 간의 트레이드오프(insights)를 탐구합니다.

- **Performance Highlights**: 실험 결과에 따르면 $f$-PO는 AlpacaEval 2, Arena-Hard, MT-Bench와 같은 인기 있는 벤치마크에서 기존 방법에 비해 우수한 성능을 보여주었습니다. 특히 $	ext{α-divergence}$를 사용할 때, Llama3-8B-instruct에서 최대 45.3%의 승리율로 성능이 개선되었습니다.



### PACE: Physics Informed Uncertainty Aware Climate Emulator (https://arxiv.org/abs/2410.21657)
- **What's New**: 본 논문에서는 PACE 라는 저용량의 물리 기반 기후 에뮬레이터를 제안합니다. PACE는 684K의 파라미터를 가진 모델로, 온도와 강수량을 86년 동안 안정적으로 모사할 수 있습니다.

- **Technical Details**: PACE는 온실가스 배출 데이터를 기반으로 훈련되어, 대류-확산 (advection-diffusion) 의 기본 물리 법칙을 통합하여 경계 조건을 고려하고 확산 계수 및 흐름 속도를 추정합니다. 이것은 기후 변수에 대한 정확한 예측을 가능하게 합니다. 또한 PACE는 15개의 기후 모델에서 벤치마크를 초과하여 우수한 성능을 보여주며, Neural ODE (Ordinary Differential Equation) 구조를 기반으로 하여 기후 시스템 내에서의 물리적 과정을 모델링합니다.

- **Performance Highlights**: PACE는 ClimateSet에서 제공하는 15가지 기후 모델을 통해 일반화된 높은 성능을 보여줍니다. 모델은 저비용으로 GHG 배출 데이터만으로 86년 동안의 기후 변수를 안정적으로 에뮬레이트 할 수 있습니다.



### Are Paraphrases Generated by Large Language Models Invertible? (https://arxiv.org/abs/2410.21637)
- **What's New**: 이 논문에서는 패러프레이즈(Paraphrase) 역전(paraphrase inversion) 문제를 다루고 있습니다. 이는 재작성된 문서가 주어졌을 때 원본 텍스트를 복원하는 과제를 목표로 합니다.

- **Technical Details**: 연구진은 원본 텍스트와 그에 상응하는 패러프레이즈 쌍을 기반으로 한 지도학습(Supervised Learning)을 통해 패러프레이즈 역전 모델을 훈련했습니다. 저자 특정 정보(author-specific context)를 활용한 두 가지 접근 방식이 제안되었습니다: 타겟 저자의 글쓰기 예시를 사용하는 방법과 저자의 독특한 스타일을 포착하는 학습된 스타일 표현을 사용하는 방법입니다.

- **Performance Highlights**: 실험 결과, 패러프레이즈된 기계 생성 텍스트에서 원본 문서의 의미를 상당 부분 복원할 수 있었으며, 저자 특정 정보가 있을 때 효과적임을 보여주었습니다. 비슷한 스타일의 텍스트가 생성된 소스 문서에서는 복원이 더 어려워질 수 있지만, 여전히 스타일 기반의 표절 탐지기(plagiarism detectors) 및 저자 식별 시스템에서 성능이 크게 향상되었습니다.



### Learning the structure of any Hamiltonian from minimal assumptions (https://arxiv.org/abs/2410.21635)
Comments:
          44 pages

- **What's New**: 이번 연구에서는 시간 진화 연산자 e^{-iHt}에 대한 블랙 박스 쿼리를 통해 알려지지 않은 quantum many-body Hamiltonian H를 효율적으로 학습하는 알고리즘을 제시합니다. 기존 방법들은 Hamiltonian H에 대한 구조적 가정이나 다량의 컴퓨팅 후처리를 요구하였으나, 본 논문은 단지 Hamiltonian 항의 개수에 대한 제한만을 가정하고 효율적인 방법으로 H를 학습할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 두 가지 시간 진화 모델을 고려합니다. 첫 번째 모델은 시간 역전(t < 0) 접근을 제공하며, 이는 시간 진화에 대해 총 \\\widetilde{O}(m/\epsilon) 시간을 요구합니다. 두 번째 모델은 전통적인 접근으로, Forward-time만 가능하며, 이에 대해 알고리즘은 \\\widetilde{O}(||H||^3/\epsilon^4) 진화 시간을 요구합니다. Pseudo-Choi 상태 개념이 이 알고리즘의 중심이며, 이를 통해 Hamiltonian H의 Fourier 스펙트럼을 학습하는 데 도움을 줍니다.

- **Performance Highlights**: 이 알고리즘은 효율적으로 n-큐빗 Hamiltonian을 학습할 수 있으며, 특정 구조적 가정 없이도 높은 정밀도로 작동합니다. 특히, 시간 진화에 대한 쿼리 수를 줄이면서도 얻을 수 있는 정확도를 보장합니다. 이는 양자 메트롤로지, 양자 장치 공학 등 다양한 분야에서 중요하게 활용될 수 있습니다.



### Refined Risk Bounds for Unbounded Losses via Transductive Priors (https://arxiv.org/abs/2410.21621)
- **What's New**: 이 논문에서는 전통적인 선형 회귀와 분류 문제에 대한 새로운 접근 방식을 제시하며, 특히 트랜스덕티브(Transductive) 온라인 학습 설정에서 손실을 최소화하는 알고리즘을 제안합니다. 이는 고정된 디자인 회귀(fixed design regression)와 다르게, 디자인 벡터의 집합을 미리 알고 있을 때의 알고리즘을 다룹니다.

- **Technical Details**: 제안된 알고리즘은 지수 가중치(exponential weights) 알고리즘을 기반으로 하며, 트랜스덕티브 사전 확률(transductive priors)을 사용하여 디자인 벡터의 전체 히스토리를 활용합니다. 이를 통해 불확실한 분포에 대해 추가적인 가정 없이도 통계적 위험 경계(statistical risk bounds)를 도출할 수 있습니다.

- **Performance Highlights**: 분류 문제에서 손실 경계는 오직 파라미터 공간의 차원과 라운드 수에만 의존하며, 디자인 벡터나 최적 해결의 노름(norm)에 독립적입니다. 선형 회귀의 경우, 희소성에 기반한 경계도 제시하며, 이는 트랜스덕티브 설정에 특화되어 있습니다.



### Accelerated, Robust Lower-Field Neonatal MRI with Generative Models (https://arxiv.org/abs/2410.21602)
Comments:
          5 pages, 3 figures, submitted to ISBI 2025

- **What's New**: 본 연구는 Neonatal Magnetic Resonance Imaging (MRI)에서 저자기장(1.5 Tesla 이하) 시스템의 활용을 개선하기 위한 Diffusion-based generative modeling과 motion modeling 기법을 도입하여 새로운 방법론을 제시합니다.

- **Technical Details**: 연구팀은 오랜 시간 소요되는 투영 및 움직임 아티팩트를 줄이기 위해 임상 Neonatal MRI 이미지를 바탕으로 training dataset을 수집하였으며, Diffusion 모델을 활용하여 낮은 신호 대 잡음 비율(Signal-to-Noise Ratio, SNR) 문제를 해결하였습니다. 자유로운 매트릭스 크기를 지원하도록 diffusion network 아키텍처를 수정하였고, 여러 종류의 영상 대비와 방향을 하나의 모델에서 훈련하는 방법을 사용했습니다. 또한 자가 감독 기법을 통해 SNR을 증가시킨 후 training을 수행했습니다.

- **Performance Highlights**: 제안된 방법은 single-coil Fast Spin Echo 및 Spin Echo 시퀀스의 스캔 시간을 평균 1.5배 단축하는 효과를 보였으며, 전통적인 재구성을 비교할 때, 제안된 방법이 독립적으로 획득한 데이터에서의 움직임 아티팩트를 현저히 줄이는 성과를 입증하였습니다.



### Reducing the Scope of Language Models with Circuit Breakers (https://arxiv.org/abs/2410.21597)
- **What's New**: 이 논문은 언어 모델들이 특정한 용도로 특정 쿼리에만 답변하게 하려는 새로운 방법, 즉 'scoping'을 제안합니다. 기존 시스템 프롬프트를 이용한 방법이 불완전하다는 것을 발견하고, 'Circuit Breakers' (CB)라는 새로운 메소드를 제시합니다.

- **Technical Details**: CB 방법은 언어 모델의 출력을 특정 작업(예: 감정 분석, 요약하기)에 맞춰 scoping하는 데 사용됩니다. 또한, Supervised Fine Tuning (SFT)과 결합하여 정확한 쿼리에서의 성능을 향상시키고, 비관련 쿼리는 거부할 수 있는 구조를 제공합니다.

- **Performance Highlights**: CB 방법은 기존의 미세 조정(fine-tuning) 방식이나 선호 학습(preference learning)에 비해 아웃 오브 디스트리뷰션(out of distribution) 작업에 대한 강건성이 뛰어나며, 다수의 허용 작업을 지원하여 보다 정교한 범위를 제공합니다. SFT와 CB를 결합할 때 쿼리 성능 향상과 비관련 쿼리 거부에서 모두 긍정적인 결과를 보여줍니다.



### ATLAS: Adapting Trajectory Lengths and Step-Size for Hamiltonian Monte Carlo (https://arxiv.org/abs/2410.21587)
Comments:
          Code available at this https URL

- **What's New**: ATLAS는 Hamiltonian Monte-Carlo (HMC) 샘플링의 새로운 접근 방식으로, 복잡한 기하학을 가진 분포를 효과적으로 샘플링하기 위해 적응형 단계 크기(step size)와 경로 길이(trajectory length)를 조정하는 전략을 개발했습니다.

- **Technical Details**: ATLAS는 local Hessian의 저차원 근사를 평가하여 단계 크기의 패러미터를 매 iteration마다 조정합니다. 또한 no U-turn 조건을 모니터링하며 경로 길이를 조정하여 복합 샘플러를 만듭니다. 이 과정에서 지연 거부(deferred rejection) 프레임워크를 사용하여 샘플링의 계산 효율을 개선하고, 하이퍼파라미터를 자동으로 조정하는 방법을 개발했습니다.

- **Performance Highlights**: ATLAS는 복잡한 기하학을 가진 어려운 분포를 정확하게 샘플링할 수 있으며, 간단한 분포에서는 NUTS에 비해 계산적으로 경쟁력을 가지면서도 하이퍼파라미터 조정에 더 견고한 성능을 보입니다.



### Audio Classification of Low Feature Spectrograms Utilizing Convolutional Neural Networks (https://arxiv.org/abs/2410.21561)
- **What's New**: 이 논문은 저특징(low-feature) 오디오 신호 분류를 위한 새로운 기계 학습 방법론을 제안하고, 이를 통해 저특징 오디오 스펙트로그램의 분류 성능을 향상시키는 구조를 발전시켰습니다.

- **Technical Details**: 저자는 binary, one-class, 그리고 siamese 접근 방식을 사용하는 여러 맞춤형 convolutional 네트워크 아키텍처를 디자인하여 오디오 신호의 스펙트로그래픽 서명을 인식하는 방법을 제시합니다. 실험은 Fast Fourier Transform (FFT)으로 처리된 시간-주파수 관계를 반영한 스펙트로그램을 사용하여 진행되었습니다.

- **Performance Highlights**: 제안된 방법론은 전통적인 오디오 분류 기법보다 우수한 성능을 보여주며, 저특징 신호의 정확한 분류를 통한 최첨단 예측 클래스 모델을 제공합니다.



### A Novel Score-CAM based Denoiser for Spectrographic Signature Extraction without Ground Truth (https://arxiv.org/abs/2410.21557)
- **What's New**: 본 논문에서는 유음 탐지 시스템에서의 스펙트로그램 기반 오디오 분류를 위한 새로운 접근 방식을 제안합니다. 특히, 점진적인 노이즈 제거와 시그니처 추출을 위해 Score-CAM 기반의 새로운 denoiser와 GAN(Generative Adversarial Network) 구조를 개발하였습니다.

- **Technical Details**: 제안된 방법론은 GAN을 사용하여 유사한 분포의 합成 스펙트로그램 훈련 데이터를 생성하여 CNN과 같은 머신러닝 모델의 훈련을 돕습니다. 또한, score-CAM 기반의 클래스 활성화 맵을 활용하여 특정 클래스의 음성과 관련된 음향 데이터를 선택적으로 추출합니다.

- **Performance Highlights**: 본 실험에서는 최신 기준보다 높은 노이즈 저감 정확도와 오디오 분류 정확도를 보였으며, 이 접근법은 다양한 데이터 분포에서 MATLAB와 같은 머신러닝 기법의 활용 가능성을 제시합니다.



### MultiTok: Variable-Length Tokenization for Efficient LLMs Adapted from LZW Compression (https://arxiv.org/abs/2410.21548)
- **What's New**: 이번 논문은 Lempel-Ziv-Welch (LZW) 데이터 압축을 기반으로 한 새로운 토크나이징 방법인 MultiTok을 제안합니다. 이는 반복적인 구문을 다중 단어 토큰으로 압축하여 더 효율적인 LLM 훈련을 가능하게 만듭니다.

- **Technical Details**: MultiTok은 입력된 데이터를 인코더를 사용하여 토크나이즈하고, 의미와 맥락 정보를 나타내는 임베딩 벡터로 추출한 후, 변환기 모델로 전달합니다. 기존의 토큰을 적은 수의 토큰으로 변환하여 크기를 줄이는 방식으로 작동합니다. 이 방법은 훈련 데이터의 33%를 동적으로 압축하고, 훈련 속도를 2.5배 가량 증가시킵니다.

- **Performance Highlights**: MultiTok은 BERT에 비견되는 성능을 유지하면서도 30% 이상의 훈련 데이터 감소를 달성하였습니다. 이러한 접근은 정보 이론적 방법을 활용하여 효율적이고 안정적인 LLM 시스템을 제공할 가능성이 있습니다.



### Deep Learning Methods for the Noniterative Conditional Expectation G-Formula for Causal Inference from Complex Observational Data (https://arxiv.org/abs/2410.21531)
- **What's New**: 이번 연구에서는 NICE g-formula 추정기를 위한 통합 딥 러닝 프레임워크를 제안합니다. 이 프레임워크는 멀티태스크 순환 신경망(multi-task recurrent neural networks)을 사용하여 시간에 따라 변하는 치료, 교란 변수(confounder), 그리고 결과의 결합 조건 분포(joint conditional distribution)를 추정합니다.

- **Technical Details**: g-formula는 관찰 데이터로부터 지속적인 치료 전략의 인과 효과를 추정하기 위한 방법론입니다. NICE 추정기는 조건부 분포를 정확히 추정하는 것을 요구하며, 전통적으로 사용되는 모형(parametric models)은 모델의 명세 오류(model misspecification)에 취약합니다. 본 연구는 LSTM(Long Short-Term Memory) 모델을 적용하여 이러한 문제를 해결하려고 하며, 실험적으로 생성된 데이터를 사용하여 깊은 학습 기반의 추정기를 평가합니다.

- **Performance Highlights**: 딥 러닝 기반의 NICE g-formula 추정기는 기존의 파라메트릭 NICE 추정기보다 인과 효과 추정의 추정 편향(bias)이 낮음을 보였습니다. 또한 간단하고 복잡한 시간 의존성 설정 모두에서 더욱 안정적인 성능을 보여주었습니다. 이 연구 결과는 깊은 학습 방식이 복잡한 관찰 데이터에서 지속적인 치료 전략의 인과 효과를 추정하는 데 있어 모델 명세 오류에 덜 민감함을 시사합니다.



### Diagnosis of Knee Osteoarthritis Using Bioimpedance and Deep Learning (https://arxiv.org/abs/2410.21512)
- **What's New**: 이 논문에서는 무릎 골관절염(knee osteoarthritis) 조기 진단을 위한 생체 임피던스(bioimpedance) 기반 진단 도구를 제안합니다. 이 도구는 정밀한 하드웨어와 딥러닝(deep learning)을 결합하여 비침습적(non-invasive)으로 효과적인 진단을 제공하도록 설계되었습니다.

- **Technical Details**: 제안된 시스템은 릴레이 기반 회로와 전략적으로 배치된 전극(electrodes)을 사용하여 포괄적인 생체 임피던스 데이터를 캡쳐합니다. 데이터는 컨볼루션(convolutional) 레이어, 드롭아웃(regularization) 및 아담 옵티마이저(optimizer)를 활용하여 최적화된 신경망 모델(neural network model)에 의해 처리됩니다.

- **Performance Highlights**: 이 접근법은 98%의 테스트 정확도(test accuracy)를 달성하여, 무릎 골관절염 관련 근골격계 질환(musculoskeletal disorders)을 감지하기 위한 유망한 도구로 평가됩니다.



### Towards Multi-dimensional Explanation Alignment for Medical Classification (https://arxiv.org/abs/2410.21494)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 새로운 프레임워크 Med-MICN(의료 다차원 해석 가능한 개념 네트워크)을 제안하여 의료 이미지 분석의 해석 가능성 문제를 해결합니다.

- **Technical Details**: Med-MICN은 신경 기호 추론(neural symbolic reasoning), 개념 의미론(concept semantics), 및 주목도 맵(saliency maps)을 통한 다각적 해석 가능성을 제공합니다. 이 프레임워크는 대규모 다중 모달(LMM)을 활용하여 개념 세트를 생성하고 의료 이미지에 대한 자동 주석(auto-annotation)을 수행합니다.

- **Performance Highlights**: 네 가지 벤치마크 데이터셋에서 테스트한 결과, Med-MICN은 기존의 개념 기반 모델과 블랙박스 모델에 비해 우수한 성능과 해석 가능성을 보였습니다.



### Enhancing CTR Prediction in Recommendation Domain with Search Query Representation (https://arxiv.org/abs/2410.21487)
Comments:
          Accepted by CIKM 2024 Full Research Track

- **What's New**: 이 논문은 사용자 검색 쿼리의 임베딩을 추천 도메인에서의 사용자 선호도와 연관 지어 학습하는 프레임워크를 제안합니다. 이러한 접근은 기존의 방법에서 간과된 사용자 의도 전이를 다루며, 검색 도메인에서의 사용자 행동 데이터를 활용하여 추천 시스템의 클릭률(CTR) 예측 정확도를 향상시킵니다.

- **Technical Details**: 제안된 프레임워크인 QueryRec는 사용자 쿼리 히스토리를 추천 도메인에 보강된 특성으로 추가하여 사용자 관심사를 보다 효과적으로 반영합니다. 이 프레임워크는 또한 자기 주의 기반의 순차 인코더를 사용하여 쿼리 리스트와 다음 클릭된 아이템의 임베딩을 정렬하며, 대조 학습 모듈을 통해 검색 도메인에서의 쿼리-아이템 관계를 캡처합니다. 확산 모델(diffusion model)을 도입하여 긍정적인 아이템의 예측을 강화하고, 노이즈 제거 방식을 통해 잘못된 긍정 예측을 방지합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 최신 모델들과 비교하여 추천 도메인에서 뛰어난 성능을 보였으며, 사용자 선호도 전이를 효과적으로 반영하여 정확한 클릭률 예측을 가능하게 하였습니다.



### A Mathematical Analysis of Neural Operator Behaviors (https://arxiv.org/abs/2410.21481)
Comments:
          24 pages

- **What's New**: 본 논문은 신경 연산자(neural operators)의 행동을 분석하기 위한 수학적 프레임워크를 엄격하게 제시하며, 이들의 안정성(stability), 수렴(convergence), 군집 동역학(clustering dynamics), 보편성(universality), 그리고 일반화 오차(generalization error)에 중점을 둡니다.

- **Technical Details**: 신경 연산자는 Sobolev 공간에서 특정 리프시츠 조건 하에 안정된 매핑임을 증명하여 입력의 작은 변화에 대한 제어된 전파를 보장합니다. 또한, 이들은 축소 매핑으로 모델링될 때 고정점으로의 지수적 수렴을 보이며, 이는 반복적인 메서드에 대한 이론적 보장을 제공합니다. 이를 통해 신경 연산자의 군집 행동을 그레디언트 흐름(gradient flow)의 관점에서 분석합니다.

- **Performance Highlights**: 논문은 신경 연산자가 유한 차원 데이터에서 무한 차원 함수 공간으로 확장된 신경망의 역량을 통해 PDE(부분 미분 방정식)의 해를 근사할 수 있는 가능성을 강조합니다. 이로 인해 신경 연산자는 복잡한 연산자를 효율적으로 근사할 수 있는 방법으로 주목받고 있습니다.



### TransformLLM: Adapting Large Language Models via LLM-Transformed Reading Comprehension Tex (https://arxiv.org/abs/2410.21479)
- **What's New**: 이 연구에서는 법률 애플리케이션에 특화된 Phi-2-Legal 및 Mistral-Legal-7B라는 언어 모델을 개발하였습니다. 이 모델들은 500백만 토큰 이상의 법률 텍스트로 계속해서 사전 학습을 진행하였으며, 법률 작업에서의 성능을 크게 향상시켰습니다.

- **Technical Details**: 이 모델들은 이전에 학습한 대형 언어 모델(LLM)을 기반으로 하며, 법률 관련 원시 데이터를 읽기 이해 자료로 변환하여 지속적 사전 학습을 수행합니다. AdaptLLM 기법을 사용하여 법률 텍스트를 읽기 이해 형식으로 변환하고, 이를 통해 발생한 데이터의 고품질화에 중점을 두었습니다. LoRA(저전력 재학습)와 같은 파라미터 효율적인 미세 조정 기법이 적용되었습니다.

- **Performance Highlights**: 새로운 법률 LLM들은 기존에 더 많은 자원으로 훈련된 모델들보다 우수한 성능을 보여주며, 법률 기준에서도 최신 상태의 성과를 달성했습니다. 이 연구는 도메인 특화 텍스트에서의 지속적 사전 학습의 효과를 증명하며, 다양한 작업에서 개선을 가져오는 가능성을 강조합니다.



### Flow Matching for Atmospheric Retrieval of Exoplanets: Where Reliability meets Adaptive Noise Levels (https://arxiv.org/abs/2410.21477)
Comments:
          Accepted for publication in Astronomy & Astrophysics

- **What's New**: 이번 연구는 ML(머신 러닝) 기반의 대기 회수(atmospheric retrieval) 방법을 개선하기 위해 FMPE(Flow Matching Posterior Estimation)라는 새로운 ML 알고리즘을 도입했습니다. 이는 기존의 NPE(Neural Posterior Estimation)의 장점을 유지하면서도 더 나은 아키텍처적 유연성과 확장성을 제공합니다.

- **Technical Details**: FMPE는 MCMC(Markov Chain Monte Carlo) 및 nested sampling의 대안으로 제안되며, 주의 깊은 모델링을 통해 noise level(잡음 수준)에 따라 ML 모델을 조건화하여 다양한 noise 모델에 적응할 수 있도록 합니다. 이 방법은 또한 importance sampling(IS)을 사용하여 ML 결과를 검증하고 수정합니다.

- **Performance Highlights**: FMPE는 NPE보다 약 3배 빠른 교육 속도를 보이며, IS는 부정확한 ML 결과를 수정하고, 모델 오류를 식별하며, Bayesian evidence에 대한 정확한 추정치를 제공합니다. 이 두 가지 방법 모두 다양한 noise 수준에 대해 nested sampling과 유사한 성능을 보여줍니다.



### TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors (https://arxiv.org/abs/2410.21443)
- **What's New**: 논문에서 소개된 Truck Adversarial Camouflage Optimization (TACO)은 3D 차량 모델에 대한 적대적인 위장 패턴을 생성하여 최신 모델인 YOLOv8을 속이기 위한 새로운 프레임워크입니다.

- **Technical Details**: TACO는 Unreal Engine 5를 사용하여 포토리얼리스틱(Photorealistic) 렌더링과 차별화된 렌더링을 결합하여 YOLOv8을 목표로 하는 텍스처를 최적화합니다. 또한 이를 위해 Convolutional Smooth Loss 함수를 도입하여 생성된 텍스처가 감지기를 속이는 데 효과적이면서도 시각적으로 그럴듯하도록 합니다.

- **Performance Highlights**: 실험 결과, TACO는 YOLOv8의 탐지 성능을 크게 저하시키며, 테스트 데이터에서 AP@0.5의 수치가 0.0099를 기록했습니다. 이러한 적대적 패턴은 다른 객체 탐지 모델인 Faster R-CNN 및 이전 YOLO 버전에서도 강한 전이 가능성을 보였습니다.



### UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function (https://arxiv.org/abs/2410.21438)
- **What's New**: 이 논문에서는 Unified Fine-Tuning (UFT)라는 새로운 방법론을 제안하여 Supervised Fine-Tuning (SFT)와 alignment을 하나의 훈련 단계로 통합합니다. 이 접근 방식은 catastrophic forgetting 문제를 효과적으로 방지하며, downstream task에서 SFT에 비해 성능 향상을 보입니다.

- **Technical Details**: UFT는 이전의 SFT와 alignment의 목표와 손실 함수를 동일하게 사용하여 훈련합니다. 이는 implicit reward function을 통해 이루어지며, instruction-tuning 데이터를 alignment 데이터로 고려하여 두 데이터 세트를 혼합함으로써 훈련합니다. 이 방법은 강화학습(RL) 기반의 불안정성 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, UFT는 instruction-following에 대한 ifeval task 및 사실성(factuality)에 대한 truthful-qa task에서 상당한 성능 개선을 보여주었습니다. UFT는 각 단계를 순차적으로 적용하는 것에 비해 명확한 장점을 가지고 있습니다.



### High-Dimensional Gaussian Process Regression with Soft Kernel Interpolation (https://arxiv.org/abs/2410.21419)
Comments:
          14 pages, 6 figures

- **What's New**: 이번 논문에서는 고차원 데이터 세트에서 확장 가능한 Gaussian Process (GP) 회귀를 위한 Soft Kernel Interpolation (SoftKI) 기법을 도입합니다. SoftKI는 구조적 격자에서 GP 커널을 보간(interpolation)하여 근사하는 Structured Kernel Interpolation (SKI)에서 영감을 받아 개별 포인트에서 학습된 보간점(인듀싱 포인트)를 통해 커널을 근사합니다.

- **Technical Details**: SoftKI는 SKI의 장점을 유지하면서도 인듀싱 포인트 접근 방식을 통해 보간 포인트의 위치를 학습합니다. 커널의 구조를 방해함으로써 데이터 차원(dimension)과의 비용 의존성을 제거하며, softmax를 이용한 효율적이고 정확한 추론 방법을 제시하여 시간 복잡도가 𝒪(m²n)로 유지됩니다. 공간 복잡도는 𝒪(mn)입니다.

- **Performance Highlights**: SoftKI는 UCI 리포지토리의 다양한 데이터셋에서 실험하였으며, 데이터 차원이 약 10일 때 인덕팅 포인트를 기반으로 한 인기 있는 확장 가능 GP 방법보다 테스트 Root Mean Square Error (RMSE)에서 더 높은 정확도를 보였습니다. 또한, SoftKI는 수백에서 수천 차원의 고차원 분자 데이터셋에서도 우수한 성능을 보였습니다.



### Deploying Ten Thousand Robots: Scalable Imitation Learning for Lifelong Multi-Agent Path Finding (https://arxiv.org/abs/2410.21415)
Comments:
          Submitted to ICRA 2025

- **What's New**: 이번 연구는 Lifelong Multi-Agent Path Finding (LMAPF) 문제 해결을 위한 새로운 접근법인 Scalable Imitation Learning for LMAPF (SILLM)를 제안합니다. SILLM은 최신 GPU의 도움을 받아 빠른 추론 속도를 유지하면서도 고품질 솔루션을 제공합니다.

- **Technical Details**: SILLM은 Spatially Sensitive Communication (SSC) 모듈을 통해 에이전트 간의 의사소통을 강화하고, 단일 단계 충돌 해결 및 글로벌 가이드를 체계적으로 통합하는 새로운 접근법입니다. 이러한 설계를 통해 최대 10,000명의 에이전트를 효율적으로 관리할 수 있습니다.

- **Performance Highlights**: SILLM은 10,000명의 에이전트를 포함한 6개의 대규모 맵에서 평균 137.7%와 16.0%의 처리량 개선을 달성하며, 2023 League of Robot Runners 대회에서 우승 솔루션을 초월하였습니다. 또한, 실제 로봇 10대 및 가상 로봇 100대를 사용한 검증에서도 효과성을 입증하였습니다.



### Exploring reinforcement learning for incident response in autonomous military vehicles (https://arxiv.org/abs/2410.21407)
Comments:
          DIGILIENCE 2024

- **What's New**: 무인 차량의 자율 사이버 방어를 위한 강화 학습 활용에 관한 연구입니다. 이 연구는 기존 문헌에서 다루지 않았던 군사 작전에서의 자율 방어 시스템을 다룬 점이 특징입니다.

- **Technical Details**: 이 논문은 무인 육상 차량(UGV)에서 사이버 공격에 자율적으로 대응할 수 있는 에이전트를 강화 학습을 이용하여 훈련하는 방법을 탐구합니다. 초기에는 간단한 시뮬레이션 환경에서 프로토타입 에이전트를 개발하고, 이를 더 현실적인 환경과 실제 UGV로 테스트했습니다. 핵심 기여는 간단한 시뮬레이션 환경에서도 강화 학습이 실제 UGV에 사용할 수 있는 에이전트를 훈련하기에 적합하다는 것을 보여주는 것입니다.

- **Performance Highlights**: 강화 학습 기반 에이전트는 사이버 공격에 대한 자율적 대응을 가능하게 하며, 군사 작전에서 자율 차량의 보안 요구를 충족하는 데 필요한 방안을 제시합니다. 이 연구에서는 다양한 시뮬레이션 환경에서 에이전트를 적용하고 비교하여 유의미한 결과를 도출했습니다.



### Model-agnostic basis functions for the 2-point correlation function of dark matter in linear theory (https://arxiv.org/abs/2410.21374)
Comments:
          20 pages, 9 figures, to be submitted to JCAP. The implementation of the BiSequential architecture, along with a simple example notebook, is publicly available as part of the MLFundas repository at this https URL

- **What's New**: 본 연구에서는 정형적 분석을 넘어 바리온 음향 진동(BAO) 기능을 탐색하고 예측하기 위해 다차원 데이터에서 최소한의 기초 함수 집합을 발견하는 머신러닝 프레임워크를 개발하였다. 이 접근 방식은 확장성과 정확성을 가진 새로운 방법론을 제시한다.

- **Technical Details**: 연구진은 정규화된 신경망 아키텍처인 	exttt{BiSequential}을 사용하여, 분리된 cosmological parameters와 분리된 거리 변수를 동시에 처리할 수 있는 구조를 개발하였다. 기초 함수 집합은 복잡한 우주론적 모델의 BAO 특성을 수치적으로 설명할 수 있도록 설계되어, 매우 높은 정확도를 달성하였다.

- **Performance Highlights**: 최적의 신경망은 9개의 함수로 구성된 기초 집합을 생성하였고, 이는 $	extit{curved}$ $w$CDM 모델에서 약 0.6% 정확도로 $	extxi_{m lin}(r)$를 설명할 수 있는 것으로 나타났다. 이러한 성과는 BAO 분석에서 통계적 이점을 더할 수 있음을 시사한다.



### Domain Adaptation with a Single Vision-Language Embedding (https://arxiv.org/abs/2410.21361)
Comments:
          Under review

- **What's New**: 이번 논문에서는 도메인 적응(Domain Adaptation)을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 전체 타겟 데이터를 필요로 하지 않고, 단일 Vision-Language (VL) 잠재 임베딩에 의존하여 이루어집니다. 기존 방법과의 차별점은 CLIP 모델을 활용하여 여러 시각적 스타일을 추출하는 기능 향상 방법인 Prompt/photo-driven instance normalization (PIN)을 적용하는 것입니다.

- **Technical Details**: 이 연구는 언어 프롬프트나 단일 비표시 타겟 이미지를 기반으로 하는 단일 VL 잠재 임베딩을 통해 도메인 적응을 수행합니다. PIN 방법을 사용해 저수준 소스 특징의 affine 변환을 최적화하여 여러 시각적 스타일을 생성하며, 이를 통해 제로샷(Zero-Shot)과 원샷(One-Shot) 비지도 도메인 적응을 가능하게 합니다. 이 방식은 CLIP의 잠재 공간을 활용하여 소스 도메인 임베딩을 타겟 도메인 임베딩으로 변환합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 제로샷 및 원샷 설정에서 관련 기준선보다 뛰어난 성능을 보였습니다. PØDA와 PIDA 방식으로 이미지 분할 성능이 향상되었으며, 이는 공공 데이터 사용이 제한된 산업 환경에서도 유용함을 나타냅니다.



### Can Machines Think Like Humans? A Behavioral Evaluation of LLM-Agents in Dictator Games (https://arxiv.org/abs/2410.21359)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)을 기반으로 한 에이전트의 친사회적 행동을 유도하는 다양한 페르소나(persona)와 실험적 환경의 영향을 조사하였습니다. 또한, LLM의 복잡한 의사결정 시나리오에서 성과를 평가하기 위한 행동적 접근 방식을 제안하였습니다.

- **Technical Details**: 이 연구는 권위자의 게임(dicator game)을 기반으로 하여 LLM 에이전트의 이타적 행동을 평가했습니다. 다양한 LLM 가족 간과 인간 행동과 비교한 결과, LLM 사이에서 행동의 변동성과 불일치가 크게 나타났습니다. LLM에 인간과 유사한 정체성을 부여하는 것만으로는 인간과 유사한 행동을 나타내지 않았습니다.

- **Performance Highlights**: LLM 에이전트는 인간의 의사결정을 정확하게 예측하지 못하고, 이러한 일치도는 모델 아키텍처와 프롬프트 형식에 따라 달라지는 highly variable한 특성을 보였습니다. 이는 LLM의 복잡한 사회적 상호작용에서 그들의 행동을 평가하기 위한 더 깊은 이해와 견고한 방법론의 필요성을 강조합니다.



### Energy-Based Diffusion Language Models for Text Generation (https://arxiv.org/abs/2410.21357)
- **What's New**: 이 연구는 기존의 discrete diffusion models의 한계를 극복하기 위해 Energy-based Diffusion Language Model (EDLM)을 제안합니다. EDLM은 디퓨전 과정의 각 단계에서 전체 시퀀스를 동시에 디노이즈할 수 있도록 설계된 에너지 기반 모델입니다.

- **Technical Details**: EDLM은 잔여 형태(residual form)로 구성되어 있으며, pretrained autoregressive 모델의 매개변수를 활용하거나 noise contrastive estimation을 통해 bidirectional transformer를 fine-tuning하여 매개변수를 얻습니다. 이 모델은 기존 diffusion models에서 발생하는 훈련 및 샘플링 분포의 불일치를 해결합니다.

- **Performance Highlights**: EDLM은 언어 모델링 벤치마크에서 기존의 최첨단 diffusion 모델을 일관되게 초과 달성하며, autoregressive 모델과 유사한 perplexity를 접근합니다. 또한, 기존 디퓨전 모델보다 1.3배 빠른 샘플링 속도를 제공하면서도 생성 성능의 저하 없이 결과를 낼 수 있습니다.



### Absorb & Escape: Overcoming Single Model Limitations in Generating Genomic Sequences (https://arxiv.org/abs/2410.21345)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 최근 면역학 및 합성 생물학의 발전은 DNA 서열 설계에 대한 심층 생성 방법의 발전을 가속화하고 있습니다. 본 논문은 AutoRegressive (AR) 모델과 Diffusion Models (DMs)의 한계를 분석하고, 두 가지 방식을 조합한 새로운 샘플링 방법인 Absorb & Escape (A&E)를 제안합니다.

- **Technical Details**: DNA 서열은 다양한 기능적 영역 (예: Promoter Regions, Exons, Introns)로 구성되어 있는 비동질적(heterogeneous) 특성을 가지고 있습니다. 현재의 AR 모델은 데이터의 전이 확률을 학습하나 DNA 서열의 전체적인 특성을 포착하지 못하고, DMs는 전반적인 분포를 회복하는 데 유리하나 베이스 페어(base pair) 수준에서 오류를 발생시킬 수 있습니다. A&E 접근 방식은 DMs로 생성된 샘플을 AR 모델을 통해 정제하여 품질을 개선합니다.

- **Performance Highlights**: 15종의 생물체에 대한 광범위한 실험을 통해, A&E 방법은 모티프 분포 및 전체 유전자 유사성까지 고려했을 때 최신 AR 모델과 DMs보다 뛰어난 성능을 보였습니다. 이는 기능적 특성과 다양성에서 기존 방식들이 가지고 있는 한계를 넘어서는 결과를 보여줍니다.



### Combining Incomplete Observational and Randomized Data for Heterogeneous Treatment Effects (https://arxiv.org/abs/2410.21343)
Comments:
          10 pages, 4 figures, Accepted By CIKM2024

- **What's New**: 이 논문에서는 결측(缺失) 관찰 데이터와 무작위화된 데이터(RCTs)를 결합하여 이질적 치료 효과(Heterogeneous Treatment Effects, HTEs)를 추정하는 새로운 접근 방식인 CIO(Combine Incomplete Observational data and randomized data)를 제안합니다. 기존 방법들이 완전한 관찰 데이터를 필요로 하는 반면, CIO는 부분적 데이터로도 효과적으로 HTE를 추정할 수 있습니다.

- **Technical Details**: CIO 접근법은 먼저 관찰 연구(Observational Studies, OSs)에서 생성된 유사 실험군(pseudo-experimental group)과 RCT에서의 유사 대조군(pseudo-control group)을 통해 혼란 Bias 함수(confounding bias function)를 도출합니다. 이 함수를 통해 관찰 데이터의 관측된 결과를 수정하여 HTE 추정에 활용합니다. 이를 위해, 사용 가능한 모든 관찰 데이터와 무작위화된 데이터의 조합을 통해 HTE 추정기를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 CIO 방법은 합성 데이터와 두 개의 반합성 데이터세트에서 HTE 추정 시 기존 방법들보다 더 효과적으로 관찰 데이터와 무작위화된 데이터를 통합하는 것으로 나타났습니다. 이는 관찰 데이터가 부분적으로 결여된 경우에도 유효성을 유지함을 보여줍니다.



### FinTeamExperts: Role Specialized MOEs For Financial Analysis (https://arxiv.org/abs/2410.21338)
- **What's New**: 완전한 LLM 적용을 위한 금융 전문 모델, FinTeamExperts 소개

- **Technical Details**: FinTeamExperts는 Mixture of Experts (MOEs) 구조를 가진 LLM 프레임워크로, 각 모델은 매크로(Macro), 마이크로(Micro), 정량(Quantitative) 분석에 특화되어 있음. 세 가지의 80억 파라미터 모델이 각기 특화된 데이터를 기반으로 훈련됨. 두 단계의 훈련 과정으로 먼저 역할 기반 데이터로 사전 훈련한 후, 실질적인 금융 작업과 기대에 맞춰 추가 조정(instruct-tuning) 진행.

- **Performance Highlights**: FinTeamExperts는 4개의 데이터셋에서 동일한 규模 및 더 큰 모델들보다 우수한 성능을 발휘함. 특히, 복잡한 작업을 요구하는 데이터셋에서도 각각의 역할 기반 전문가 모델의 효용성을 강조함.



### CloudCast -- Total Cloud Cover Nowcasting with Machine Learning (https://arxiv.org/abs/2410.21329)
Comments:
          27 pages, 12 figures

- **What's New**: 이 논문에서는 CloudCast라는 새로운 예측 모델을 소개합니다. 이 모델은 U-Net 아키텍처에 기반한 합성곱 신경망(CNN)으로, 위성 데이터를 통해 최대 5시간 선행 예측을 가능하게 합니다. 기존의 수치 기상 예측(NWP) 모델에 비해 크게 향상된 성능을 보여줍니다.

- **Technical Details**: CloudCast는 5년간의 위성 데이터를 훈련하여 만들어졌으며, TCC(Total Cloud Cover) 예측에 특화되어 있습니다. 이 네트워크는 훈련 중 다양한 특징 선택과 손실 함수의 최적 조합에 대한 ablation study를 포함하고 있습니다. 모델은 MAE(Mean Absolute Error) 기반의 손실 함수에서 최고의 성능을 나타냅니다.

- **Performance Highlights**: CloudCast는 기존의 NWP 모델 대비 24% 낮은 평균 절대 오차(MAE)를 달성하며, 다중 카테고리 예측 오류는 46% 감소했습니다. 특히 첫 몇 시간 동안 구름 덮개 구조를 잘 포착하는 반면, 나중 예측은 흐릿해지거나 구름 형성을 과소평가하게 됩니다.



### Mathematical Derivation Graphs: A Task for Summarizing Equation Dependencies in STEM Manuscripts (https://arxiv.org/abs/2410.21324)
Comments:
          10 pages, 4 figures

- **What's New**: 이 연구는 STEM(Science, Technology, Engineering, Mathematics) 분야의 학술 문서에서 수학적 표현 간의 의존 관계를 이해하기 위한 최초의 단계로, 새로운 객체인 'derivation graph'를 제안합니다. 이 그래프는 논문의 수학 내용을 요약화합니다.

- **Technical Details**: 연구자는 arXiv 데이터베이스에서 수집한 107개의 STEM 논문을 기반으로 수학적 유도 그래프를 수작업으로 레이블링했습니다. 다양한 알고리즘을 사용하여 이 그래프를 자동으로 추출하는 방법을 평가했습니다. 실험적으로, 분석적 및 NLP 모델(LLM 포함)을 평가하여 각각의 문서에서 유도 그래프를 추출하는 능력을 비교했습니다.

- **Performance Highlights**: 연구 결과, 분석적 모델과 NLP 모델 모두 약 40-50%의 F1 점수를 기록하여, 최신 NLP 기법이 단순 분석 모델에 비해 수학적 텍스트를 이해하는데 큰 진전을 이루지 못했음을 보여주었습니다. 따라서 보다 정확하고 심도 있는 수학적 정보 처리를 위한 추가 연구가 필요합니다.



### Decoding Diffusion: A Scalable Framework for Unsupervised Analysis of Latent Space Biases and Representations Using Natural Language Prompts (https://arxiv.org/abs/2410.21314)
- **What's New**: 최근의 이미지 생성 알고리즘 발전으로 인해 순방향 확산 모델(Diffusion Models)이 고 품질의 이미지를 생성하는 강력한 도구로 자리잡았습니다. 본 논문은 이러한 모델의 의미론적 잠재 공간(semantic latent space)을 자동으로 탐색할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 자연어 프롬프트와 이미지 캡션을 직접 활용하여 잠재 방향(latent direction)을 매핑합니다. 이 접근법은 특정 벡터를 훈련할 필요 없이, 의미론적 정보(semantic information)를 활용하여 잠재 공간의 자동 이해를 가능하게 합니다. 또한, Latent Consistency Model(LCM)을 활용하여 중간 U-Net 레이어의 출력인 h-space를 샘플링합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 도메인에서 숨겨진 패턴과 연관성을 발견하며, 순방향 확산 모델의 의미론적 지식(semantic knowledge)과 편향(bias)을 분석하는 데 뛰어난 성능을 보였습니다. 이로 인해 해석할 수 있는 더욱 투명한 생성 모델을 위한 가능성을 열었습니다.



### Towards Robust Out-of-Distribution Generalization: Data Augmentation and Neural Architecture Search Approaches (https://arxiv.org/abs/2410.21313)
Comments:
          Hong Kong University of Science and Technology Thesis

- **What's New**: 이 논문에서는 Deep Learning (딥러닝) 분야에서 Out-of-Distribution (OoD) 데이터에 대해 보다 강력한 일반화 방법론을 제안합니다. 기존 모델이 수집된 분포와 다를 경우의 성능 저하 문제를 해결하기 위해 spurious correlation (가짜 상관관계)을 분리하고, context-related features (맥락 관련 특성)에 대한 gradient-based augmentation (그래디언트 기반 증강)을 수행하여 학습된 표현의 강인성을 향상시킵니다.

- **Technical Details**: 논문에서 제안하는 DecAug 방법은 category-related features (범주 관련 특성)과 context-related features (맥락 관련 특성)를 직교화하여 분리합니다. 또한, NAS-OoD(Neural Architecture Search-out-of-distribution)는 가짜 OoD 데이터를 생성하는 조건부 생성기를 배우면서 아키텍처 파라미터를 최적화합니다. 이를 통해 더 강력한 네트워크 아키텍처를 발견하고 있는 것으로 보입니다.

- **Performance Highlights**: DecAug는 다양한 OoD 데이터셋에서 최첨단 방법들을 초월하는 성능을 보여주었으며, NAS-OoD 방법은 업계 데이터셋에서 70% 이상의 오류율 감소를 보이며 실제 애플리케이션에서도 높은 유용성을 입증했습니다.



### VideoSAM: A Large Vision Foundation Model for High-Speed Video Segmentation (https://arxiv.org/abs/2410.21304)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문에서는 Segment Anything Model (SAM)을 기반으로 한 VideoSAM을 소개합니다. VideoSAM은 다양한 고속 비디오 (HSV) 데이터 셋에 맞춰 조정되어 복잡한 기포 형성을 구분할 수 있도록 발전되었습니다.

- **Technical Details**: VideoSAM은 두 단계 접근 방식으로 작동합니다. 먼저 각 데이터 모달리티(아르곤, 질소, FC-72, 물)에 대해 전문화된 U-Net CNN을 구축하여 초기 분할 마스크를 생성합니다. 이후 이 이미지-마스크 쌍을 VideoSAM의 트랜스포머 아키텍처에 공급하여 최종 출력에 대한 정제된 분할 마스크를 생성합니다.

- **Performance Highlights**: VideoSAM은 복잡한 유체 환경에서 U-Net보다 현저하게 뛰어난 성능을 보였습니다. 실험 결과 VideoSAM이 다양한 과학적 응용 분야에서 더 강력하고 정확한 HSV 분할을 제공할 가능성을 강조합니다.



### VEMOCLAP: A video emotion classification web application (https://arxiv.org/abs/2410.21303)
Comments:
          Accepted to 2024 IEEE International Symposium on Multimedia (ISM), Tokyo, Japan

- **What's New**: VEMOCLAP을 소개합니다: 사용자 제공 비디오의 감정 내용을 분석할 수 있는 최초의 오픈 소스 웹 애플리케이션입니다. 이전의 연구를 개선하여 멀티헤드 크로스-어텐션(multi-head cross-attention)을 사용하여 비디오 프레임과 오디오로부터 추출된 사전 훈련된 특징을 효율적으로 융합합니다.

- **Technical Details**: VEMOCLAP은 Google Colab에서 호스트되며, 무료 GPU 런타임으로 모든 사용자 수준에서 접근 가능하고 몇 번의 클릭만으로 사용할 수 있습니다. 사용자는 비디오를 업로드하거나 YouTube 링크를 제공할 수 있으며, 예상 감정을 출력하고 자동 음성 인식(ASR), 광학 문자 인식(OCR), 얼굴 감지 및 표정 분류, 오디오 분류, 이미지 캡셔닝과 같은 추가 분석을 포함합니다. 모델은 Ekman-6 비디오 감정 데이터셋을 기반으로 훈련되었으며, 고급 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 모델은 Ekman-6 비디오 감정 데이터셋에서 분류 정확도를 4.3% 향상시켰으며, 데이터 정제를 통해 비디오 감정 모델의 훈련을 개선했습니다. 이 웹 애플리케이션을 통해 사용자는 자신의 비디오나 YouTube 비디오를 분석하고 감정을 분류할 수 있습니다.



### Domain-Adaptive Pre-training of Self-Supervised Foundation Models for Medical Image Classification in Gastrointestinal Endoscopy (https://arxiv.org/abs/2410.21302)
- **What's New**: 이번 연구에서는 대규모의 위장내시경(VCE) 데이터셋인 EndoExtend24를 소개합니다. 이 데이터셋은 10개의 기존 공개 및 비공식 데이터셋을 병합하고 재구성하여 만들어졌으며, 226,000개 이상의 라벨링된 이미지를 포함하고 있습니다. 또한, 123개의 병리학적 소견을 지원하는 동적 클래스 매핑을 제공하여 다양한 레이블링 세분화에 따라 통일된 학습을 가능하게 합니다.

- **Technical Details**: EndoExtend24 데이터셋은 영상 캡슐 내시경, 위내시경, 대장내시경 등의 세 가지 검사 방법을 포함합니다. EVA-02 모델은 Vision Transformer 아키텍처를 기반으로 하며, EndoExtend24 데이터셋에서 사전 훈련(pre-training) 후, Capsule Endoscopy 2024 Challenge 데이터셋에서 미세 조정(fine-tuning)됩니다. 이 과정에서 SwiGLU 활성화(SwiGLU activations), Rotary Position Embeddings (ROPE), 추가적인 Layer Normalization (LN) 기능이 적용됩니다.

- **Performance Highlights**: 실험 결과, 챌린지 검증 세트에서 AUC 매크로 점수 0.993과 균형 잡힌 정확도 89.3%를 기록하는 등 우수한 성능을 보였습니다.



### Evaluating the Posterior Sampling Ability of Plug&Play Diffusion Methods in Sparse-View C (https://arxiv.org/abs/2410.21301)
- **What's New**: 이 논문은 Plug&Play (PnP) 확산 모델의 사후 샘플링 능력을 평가하는 데 초점을 맞추고 있으며, 측정된 sinogram이 충분한 정보를 포함하지 않을 때의 문제를 다룹니다.

- **Technical Details**: Sparse-View Computed Tomography (SVCT) 모델을 사용하여 sinogram으로부터 이미지를 재구성하는 과정에서, 적은 수의 투사(≤ 180)로 인해 사후 분포가 피크를 이루지 않거나 다중 모드(multi-modal)가 되는 상황을 평가합니다. 기존의 PnP 확산 모델은 종종 피크가 있는 분포를 가정하였고, 이에 대한 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: 실험 결과, 각 PnP 모델은 투사 수가 줄어들 때마다 약한 사후 분포와 실제 사후 분포 간의 차이가 발생하는 경향을 보였습니다. 이는 PnP 모델의 성능 평가를 위한 새로운 기준의 필요성을 강조합니다.



### Contrastive Learning with Auxiliary User Detection for Identifying Activities (https://arxiv.org/abs/2410.21300)
Comments:
          Accepted in ICMLA 2024

- **What's New**: 본 논문에서는 User Identification(UI)와 Context-Aware Human Activity Recognition(CA-HAR)를 통합한 새로운 프레임워크인 CLAUDIA를 제안합니다. CLAUDIA는 사용자 특성과 맥락 정보를 동시에 고려하여 개인화된 활동 인식을 향상시킵니다.

- **Technical Details**: CLAUDIA 프레임워크는 Contrastive Learning을 통한 사용자 식별과 주요 CA-HAR 과업을 통합하며, supervised contrastive loss를 도입하여 사용자 간의 변이를 정규화합니다. 이 모델은 다양한 모달리티 센서 데이터를 통합하여 높은 성능을 얻도록 설계되었습니다.

- **Performance Highlights**: 실제 CA-HAR 데이터셋을 통해 CLAUDIA는 Matthew의 상관 계수(Matthew's Correlation Coefficient)에서 평균 5.8%에서 14.1%, Macro F1 점수에서 3.0%에서 7.2% 향상된 성능을 보였습니다.



### Large-scale Multi-objective Feature Selection: A Multi-phase Search Space Shrinking Approach (https://arxiv.org/abs/2410.21293)
- **What's New**: 이 논문에서는 Feature Selection에 대한 새로운 접근 방식인 LMSSS(제목: Large-scale Multi-objective Shrinking Subspace)라는 대규모 다목적 진화 알고리즘을 제안합니다. 이 알고리즘은 불필요한 특징을 제거하여 검색 공간의 차원을 줄임으로써 Sparse Optimization 문제를 해결하는 데 도움을 줍니다.

- **Technical Details**: LMSSS 방법은 특징과 클래스 레이블 간의 상관관계 및 초기 비용 효율적인 진화 과정에서의 빈도를 기반으로 한 순위 기반 필터링 방법을 포함합니다. 또한, 더 나은 분류 정확도를 가진 부모 솔루션에 더 높은 가중치를 부여하는 스마트 교차 방식과, 인구에서 조기에 제외된 특징을 대상으로 하는 지능형 돌연변이 과정을 설계하였습니다. 이러한 통합 기법을 통해 LMSSS는 대규모 Feature Selection 문제의 탐색을 보다 효율적이고 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: 15개의 대규모 데이터 세트에 대한 포괄적인 실험을 통해 제안된 알고리즘의 효과성을 입증하였으며, 기존의 최첨단 대규모 Feature Selection 알고리즘보다 더 정확한 특징 집합을 식별할 수 있는 잠재력을 보여주었습니다. 이 결과는 LMSSS가 모델 성능과 계산 효율성을 개선할 수 있는 능력을 강조하며, 이 분야에서 새로운 벤치마크를 설정하는 것을 목표로 하고 있습니다.



### Achilles, Neural Network to Predict the Gold Vs US Dollar Integration with Trading Bot for Automatic Trading (https://arxiv.org/abs/2410.21291)
- **What's New**: 이 논문에서는 Achilles라는 새로운 주식 시장 예측 모델을 소개합니다. LSTM(Long Short Term Memory) 신경망을 사용하여 Gold와 USD 사이의 상품 가격을 예측합니다.

- **Technical Details**: Achilles 모델은 LSTM 아키텍처를 기반으로 하며, 매분마다 예측을 수행합니다. 이 모델은 23일 동안의 테스트를 통해 성능을 검증하였고, 주말을 제외한 실시간 데이터로 학습 및 예측합니다.

- **Performance Highlights**: 테스트 기간 동안 $1623.52의 이익을 생성하며, 기계 학습(Machine Learning)을 활용한 경제적 예측의 가능성을 입증했습니다.



### pLDDT-Predictor: High-speed Protein Screening Using Transformer and ESM2 (https://arxiv.org/abs/2410.21283)
Comments:
          6 pages main topic, 8 pages including citiation, 4 figures

- **What's New**: pLDDT-Predictor는 미리 훈련된 ESM2 단백질 임베딩과 Transformer 아키텍처를 활용하여 AlphaFold2의 pLDDT 점수를 빠르고 정확하게 예측하는 도구이다. 이 모델은 단백질 구조 품질 평가의 필요성을 충족시키며 동시에 컴퓨팅 리소스를 절약할 수 있도록 설계되었다.

- **Technical Details**: pLDDT-Predictor는 ESM2(진화적 규모 모델링) 임베딩 레이어와 Transformer 인코더를 결합한 고급 딥러닝 모델이다. ESM2는 각 아미노산의 진화적 특성을 포착하고, Transformer는 문자열 내 용접(dependencies)을 학습하여 단백질 구조 품질을 예측한다. 최종 예측 점수는 Huber 손실 함수에 기반하여 훈련되며, Adam 옵티마이저와 CosineAnnealingLR 스케줄러를 사용하여 최적화된다.

- **Performance Highlights**: 실험 결과, pLDDT-Predictor는 150만 개의 단백질 서열에서 70 이상의 pLDDT 점수를 가지는 단백질의 90% 이상을 분류할 수 있으며, 이는 AlphaFold2의 신뢰성과 밀접하게 일치한다.




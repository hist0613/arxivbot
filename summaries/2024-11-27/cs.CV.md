New uploads on arXiv(cs.CL)

### Adaptive Deployment of Untrusted LLMs Reduces Distributed Threats (https://arxiv.org/abs/2411.17693)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 비신뢰 모델에 대한 안전성 평가를 위한 새로운 접근 방식을 제안합니다. 특히 "분산 위협 설정(distributed threat setting)"에서 안전 및 유용성을 동시에 고려하여 동적으로 프로토콜을 선택하는 두 단계 배포 전략을 활용합니다. 이를 통해 비신뢰 모델의 안전성을 평가하며, 작은 문제들에서 발생할 수 있는 누적 영향을 최소화할 수 있습니다.

- **Technical Details**: 제안된 접근 방식은 '마이크로 프로토콜(micro-protocols)'과 '매크로 프로토콜(macro-protocol)'을 사용하여 작업 순서에서 비신뢰 모델의 출력을 안전하게 모니터링합니다. 마이크로 프로토콜은 단일 작업 수준에서 신뢰 가능한 모델을 사용하여 비신뢰 모델을 감시하며, 매크로 프로토콜은 전체 작업 시퀀스에 대한 신뢰도를 지속적으로 업데이트하여 더욱 안전한 선택을 하도록 합니다. 실험을 통해 각각의 프로토콜이 서로 다른 수준의 안전성과 유용성을 제공함을 보여줍니다.

- **Performance Highlights**: 실험 결과, 새로운 배포 전략은 더 즉각적으로 비신뢰 모델의 잠재적 위험을 줄이는 데 효과적임이 입증되었습니다. 10,000개의 작업 시퀀스에서 < 1%의 작업에서만 뒷문(backdoor)을 허용하며, 비신뢰 모델의 APPS 성능을 98.3% 유지합니다. 이 연구는 안전성과 유용성의 파레토 경계를 크게 향상시킨다는 결과도 제시하여, 향후 연구에 기여할 수 있는 가능성이 큽니다.



### Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning (https://arxiv.org/abs/2411.17679)
- **What's New**: 본 논문에서는 Token Internal Position Awareness (TIPA)라는 새로운 접근 방식을 소개합니다. 이 방법은 큰 언어 모델(LLMs)이 토큰의 내부 구조에 대한 이해를 향상시키기 위해, 토크나이저의 자체 어휘를 사용하여 역문자 예측 작업(reverse character prediction tasks)에서 학습하도록 합니다. TIPA는 모델이 문자 위치 및 내부 구조를 효과적으로 학습하고 일반화할 수 있도록 돕습니다.

- **Technical Details**: TIPA는 토큰이 문자로 세분화된 구조의 이해 부족을 해결하기 위한 방법으로 제안되었습니다. 기존의 Byte-Pair Encoding (BPE) 및 Byte-Level BPE (BBPE) 기술은 텍스트를 토큰으로 세분화하지만, 이 과정에서 문자 구조와 순서가 가려지는 문제가 있었습니다. TIPA는 이러한 문제를 극복하기 위해 문자 예측을 기반으로 토큰 내부의 다양한 상태를 학습하는데 초점을 맞춥니다.

- **Performance Highlights**: 실험 결과, TIPA로 훈련된 LLM은 토큰 수준의 문자 위치 예측에서 기존의 벤치마크 모델보다 우수한 성능을 보였습니다. 또한, 중국어 철자 교정(CSC)과 같은 다운스트림 작업에 적용했을 때, TIPA는 모델의 수렴 속도를 증가시키고, 작업 성능을 현저히 향상시켰습니다.



### Push the Limit of Multi-modal Emotion Recognition by Prompting LLMs with Receptive-Field-Aware Attention Weighting (https://arxiv.org/abs/2411.17674)
- **What's New**: 최근의 AI 발전은 인간의 감정을 인식하는 데 있어 필수적입니다. 그러나 감정 인식은 대화의 내용 이해, 음향 신호 해석, 비디오 표현 포착 등을 필요로 하는 복잡한 작업입니다. 본 논문에서는 대화에서 감정을 더 잘 이해하기 위해 LLM과 멀티미디어 모달리티의 보조적 특징을 결합한 새로운 프레임워크인 Lantern을 제시합니다.

- **Technical Details**: Lantern은 감정 클래스와 차원 점수를 생성하는 멀티태스크 바닐라 모델(CORECT 및 SDT)을 사용하여, 사전 훈련된 LLM(GPT-4 및 Llama-3.1-405B)로 조정합니다. 각 대화 샘플은 특정 수 (t)의 수용 영역에 포함되고, 반응 필드 인식 주의 가중치 모듈을 통해 LLM의 예측 결과가 합쳐집니다. 이 방법은 LLM의 한계인 컨텍스트 윈도우를 극복하고, 음향 신호를 통해 감정 차원 점수를 보다 잘 예측할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, Lantern은 기존의 바닐라 모델(CORECT 및 SDT)의 인식 정확도를 1.23%에서 1.80%까지 향상시켰습니다. 본 연구는 LLM과 함께 멀티모달 샘플에서 감정 분류 메트릭과 차원 점수를 추출하는 프레임워크를 제시하며, 멀티미디어 모달리티를 효율적으로 지원하는 방안을 모색합니다.



### Linguistic Laws Meet Protein Sequences: A Comparative Analysis of Subword Tokenization Methods (https://arxiv.org/abs/2411.17669)
Comments:
          8 pages, 9 figures

- **What's New**: 이번 연구는 단백질 서열(tokenization) 처리에서 BPE, WordPiece, SentencePiece와 같은 세 가지 하위 단어 토크나이저(subword tokenizers)의 효과를 비교하여 베이직한 정보를 제공합니다. 이들은 전통적인 NLP에서 사용되던 방법이지만, 단백질 서열의 독특한 특성을 고려할 때 적합하지 않을 수 있으며, 연구를 통해 이들의 성능을 분석하였습니다. 실험 결과는 단백질 서열 처리에서의 토크나이저 선택이 중요함을 강조하고 있습니다.

- **Technical Details**: 단백질 서열에서 최적의 벡터 표현(vector representation)을 유지하기 위해 BPE는 더 작은 어휘(vocabulary) 내에서 도메인 경계를 더 잘 보존하는 경향이 있으며, SentencePiece는 인코딩 효율성을 높이는데 최고점을 기록했습니다. 각 토크나이저는 고유한 패턴과 특징을 가지고 있으며, 어휘 크기의 변화에 따라 성능이 달라지는 것으로 나타났습니다. 이 연구에서는 Zipf의 법칙, 간결성 법칙, Menzerath의 법칙과 같은 언어적 법칙을 분석하여 단백질 서열의 언어적 특성을 이해하고자 하였습니다.

- **Performance Highlights**: BPE는 적은 어휘에서 더 나은 도메인 경계 보존과 컨텍스트 전문성을 보여주며, SentencePiece는 더 효율적인 인코딩을 제공하여 적은 착수도(fertility score)를 나타냈습니다. 연구 결과, 세 가지 토크나이저 모두 단백질 도메인 무결성을 유지하는 데 제한이 있으며, 어휘 크기가 증가할수록 이러한 한계가 더욱 두드러졌습니다. 최종적으로, 이 연구는 전통적인 NLP 방법이 단백질 서열에 맞지 않음을 강조하며, 단백질의 특성에 최적화된 새로운 토크나이저 전략 개발의 필요성을 요구합니다.



### How do Multimodal Foundation Models Encode Text and Speech? An Analysis of Cross-Lingual and Cross-Modal Representations (https://arxiv.org/abs/2411.17666)
Comments:
          Under review

- **What's New**: 이 연구는 최근의 다중 모달( multimodal ) 기초 모델들에서 언어 및 모달리티의 격차를 탐구합니다. 저자들은 텍스트와 음성 데이터에서 의미적으로 동등한 문장을 분석하여 이들 모델의 내부 표현을 비교합니다. 연구 결과, 대부분의 모델 계층에서 크로스 모달 표현이 수렴하는 경향을 보이는 반면, 초기 계층에서는 서로 다른 모달리티의 전처리 기능이 나타났습니다.

- **Technical Details**: 이 연구는 교차 모달( cross-modal ) 및 교차 언어( cross-lingual ) 표현의 유사성을 조사하기 위해 특수화된 마르코프 체인 인과 행렬, 즉 SVCCA( Singular Vector Canonical Correlation Analysis )를 사용합니다. 저자들은 음성과 텍스트 양쪽의 모델 활성화를 추출하고 이들 간의 유사성을 측정하여 서로 다른 모델 아키텍처의 영향을 검토합니다. 특히, 음성과 텍스트의 길이 차이가 중요한 요소로 여겨지며, 이를 최소화하는 여러 메커니즘이 존재함을 강조합니다.

- **Performance Highlights**: 저자들은 다양한 언어 자원 수준에 따라 음성과 텍스트 간의 유사성을 평가했으며, 자원 수준이 낮은 언어에서 모델이 효과적으로 공유 표현 공간으로 매핑되지 않는 경향을 발견했습니다. 또한, SONAR 모델은 모달리티 간의 격차를 줄이는 데 가장 효과적이며, 특정 길이 적응 메커니즘이 고자원 언어에 국한되어 제한적인 영향력을 발휘한다고 보고합니다. 마지막으로, 음성의 발화 변동성으로 인해 텍스트 모달리티가 더 효과적으로 통합된 교차 언어 공간을 생성한다는 점을 강조합니다.



### BERT or FastText? A Comparative Analysis of Contextual as well as Non-Contextual Embeddings (https://arxiv.org/abs/2411.17661)
- **What's New**: 이 연구는 마라티어와 같은 저자원 언어에 대한 NLP 작업의 성능을 향상시키기 위해 다양한 임베딩 기법의 영향을 분석합니다. 특히 Contextual BERT 기반, Non-Contextual BERT 기반, FastText 기반 임베딩을 비교하여, 이들 각각의 임베딩이 특정 NLP 과제에서 어떻게 수행되는지를 검토합니다. 또한 압축된 임베딩과 비압축된 임베딩의 성능 차이도 평가하여, 저자원 언어 처리에서의 임베딩 선택이 얼마나 중요한지를 강조합니다.

- **Technical Details**: 마라티어 데이터 세트인 MahaSent, MahaHate 및 MahaNews를 사용하여 감정 분류, 혐오 발언 탐지 및 뉴스 기사의 분류 작업을 수행했습니다. FastText와 BERT 임베딩 모두를 활용하며, BERT 모델은 MahaBERT와 MuRIL을 사용했습니다. 또한, 임베딩을 Multiple Logistic Regression (MLR) 분류기에 적용하여 성능을 평가하고, TSNE 시각화로 임베딩의 공간 분포를 관찰했습니다.

- **Performance Highlights**: 연구 결과, Contextual 임베딩이 Non-Contextual 임베딩보다 전반적으로 더 나은 성능을 보였습니다. 특히, BERT 기반의 Non-Contextual 임베딩이 FastText 기반 임베딩보다 더 나은 결과를 도출했으며, 이는 BERT 임베딩의 효용성이 저자원 언어의 특정 NLP 작업에서도 확립됨을 의미합니다. 해당 연구는 저자원 언어에 대한 BERT와 FastText의 비교 연구가 필요함을 제언하며, 향후 연구에 중요한 기초 자료를 제공합니다.



### On Limitations of LLM as Annotator for Low Resource Languages (https://arxiv.org/abs/2411.17637)
- **What's New**: 이번 연구는 Low-resource 언어인 마라티어에 대한 주목을 다루고 있습니다. 대형 언어 모델(LLMs)의 성능을 평가하여, 인간 언어 전문가가 아닌 시스템이 어떻게 주석 작업을 수행할 수 있는지를 탐구합니다. 연구 결과에 따르면, LLM은 고자원 언어에 적합한 주석 작업에는 뛰어난 성능을 보였지만, 낮은 자원의 언어인 마라티어에 대해서는 여전히 한계를 보이고 있음을 밝혔습니다.

- **Technical Details**: 이 연구에서는 3가지 주요 작업 범주를 사용하여 마라티어의 LLM 생성 주석과 인간 생성 주석 간의 차이를 비교 분석하였습니다. 사용된 데이터셋은 감정 분석(MahaSent), 혐오 발언 감지(MahaHate), 뉴스 분류(MahaNews)로, 각 작업에 대해 다양한 LLM(예: GPT-4o, Gemini 1.0 Pro, Llama 3.1 등)과 BERT 기반 모델을 비교했습니다. 결과는 0-shot 및 few-shot 학습 기법을 통해 수집되었습니다.

- **Performance Highlights**: 연구 결과, LLM들이 고자원 언어(예: 영어)에서는 높은 정확도를 기록했지만, 마라티어와 같은 저자원 언어에서는 성능이 크게 하락했습니다. LLM 역시 BERT 기반 모델과 비교 시 기대에 미치지 못하는 성능을 보였으며, 자동 주석 기술이 여전히 인간 전문가의 주석을 대체할 수 없음을 시사합니다. 특히, 진행된 다양한 실험에서도 LLM의 성능은 BERT 기반 모델의 기준 성능에 미치지 못했습니다.



### Scaling Speech-Text Pre-training with Synthetic Interleaved Data (https://arxiv.org/abs/2411.17607)
- **What's New**: 이 논문은 Speech language models (SpeechLMs)가 음성 입력을 수용하고 음성 출력을 생성하는 혁신적인 기법을 제안합니다. 기존의 SpeechLM 개발 접근법은 비감독 음성 데이터와 평행 음성-텍스트 데이터의 제한된 가용성으로 인해 확장성이 떨어졌습니다. 그러나 저자들은 텍스트 코퍼스로부터 파생된 대규모 합성(interleaved) 데이터를 활용하여 평행 음성-텍스트 데이터셋의 필요성을 없애는 방법을 소개합니다.

- **Technical Details**: 이 방법은 기존 텍스트 코퍼스에서 텍스트 범위를 샘플링하고 텍스트-토큰 모델을 사용하여 해당하는 음성 범위를 합성(synthesize)함으로써 음성-텍스트 합성 데이터를 효율적으로 구성합니다. 또한 자동 음성 인식(ASR) 모델에서 파생된 감독 음성 토크나이저를 사용하여 인코더에 벡터 양자화(bottleneck) 기법을 통합하여 사용합니다. 이 감독 학습 접근법은 적은 샘플링 주파수(예: 12.5Hz)에서도 강력한 의미 보존을 가진 이산 음성 토큰을 생성합니다.

- **Performance Highlights**: 이 연구는 사전 학습된 언어 모델을 시작으로 1조 개의 토큰을 사용하여 사전 학습을 확대하고, 음성 언어 모델링과 음성 질문 응답에서 최첨단(performance) 성능을 달성하였습니다. 음성 질문 작업에 대한 성능은 이전의 SOTA(13%)에서 31%로 향상되었습니다. 또한 사전 학습된 모델을 음성 대화 데이터로 미세 조정하여 경쟁력 있는 성능을 발휘하는 엔드-투-엔드 음성 챗봇을 개발했으며, 이는 대화 능력 및 음질 면에서 기존 기준과 동등한 성능을 보여주었습니다.



### What Differentiates Educational Literature? A Multimodal Fusion Approach of Transformers and Computational Linguistics (https://arxiv.org/abs/2411.17593)
- **What's New**: 최근 문헌을 교육 과정에 통합하는 것은 여전히 어려운 문제입니다. 본 연구는 Transformer 기반 텍스트 분류와 언어 특징 분석을 결합한 다중 모달(multimodal) 접근 방식을 제안하여 이러한 격차를 해결하고자 합니다. 특히, 연구결과 8개의 최첨단 Transformers가 세분화된 텍스트 데이터에 대해 미세 조정(fine-tuned)되었으며, BERT는 최고 F1 점수인 0.75를 달성했습니다.

- **Technical Details**: 연구에서는 500개의 딥 뉴럴 네트워크(topologies) 구성을 검색하여 언어적 특성을 분류하고, 그 결과 F1 점수 0.392를 획득했습니다. 이 모달리티의 융합은 상당한 개선을 보였으며, 모든 다중 모달 접근 방식이 단일 모달 모델을 초월했습니다. 특히, ELECTRA Transformer와 뉴럴 네트워크가 융합되어 F1 점수 0.996을 달성하였습니다.

- **Performance Highlights**: 제안된 접근 방식은 비기술적 이해관계자에게 실시간 텍스트 복잡성(text complexity), 독서 난이도(reading difficulty), 교육과정 정렬(curriculum alignment), 그리고 학습 연령 범위에 대한 추천을 제공하는 웹 애플리케이션으로 묶였습니다. 이 응용 프로그램은 데이터 기반 의사결정을 가능하게 하고, 영어 문학 수업의 계획에서 AI 기반 추천을 통합함으로써 수작업 부담을 줄이는 데 기여합니다.



### Natural Language Understanding and Inference with MLLM in Visual Question Answering: A Survey (https://arxiv.org/abs/2411.17558)
- **What's New**: 이 논문에서는 Visual Question Answering (VQA)의 발전을 종합적으로 분석하고 최신 모델들에 대한 상세한 설명을 제공합니다. VQA는 자연어 처리(NLP)와 컴퓨터 비전의 교차점에서 중요한 작업으로, 최신 경향과 지식 추론 모듈에 대해서도 강조합니다. 특히, 이미지-질문 정보 기반의 인지 능력 강화를 위한 진전을 다룹니다.

- **Technical Details**: VQA 추론 과정에는 시각과 텍스트 정보에서 특성을 추출하고, 이들을 정렬하여 융합하는 과정이 포함됩니다. 최근의 VQA 모델들은 Graph Neural Networks(GNN)와 Transformer 아키텍처를 활용하여 시각-언어의 상호 작용을 극대화하고 있으며, Attention 메커니즘을 통해 인식과 정보를 보완하는 역할도 수행하고 있습니다. 이러한 모델들은 지식 추론 및 멀티모달 대형 언어 모델(MLLM)과 통합되어 더욱 발전하고 있습니다.

- **Performance Highlights**: 최신 VQA 모델들은 제로샷 질문 응답에 있어 뛰어난 성과를 보여주며, 실제 어플리케이션에서도 큰 영향을 미치고 있습니다. VQA는 시각 장애인을 돕는 다양한 방식과 이미지 검색, 자율 주행, 의료 진단 등 여러 분야에 활용됩니다. 앞으로 VQA는 비주얼 대화로 진화할 가능성도 있으며, 이에 대한 연구 및 발전도 기대됩니다.



### Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search (https://arxiv.org/abs/2411.17538)
- **What's New**: 본 연구는 코드 검색 성능에 대한 isotropy(등방성)의 영향을 조사하였으며, 이를 완화하기 위한 사후 처리(post-processing) 기술을 탐구합니다. 이전의 코드 언어 모델에 대한 분석을 통해 isotropy가 각 모델의 검색 효율성에 미치는 영향을 평가하였습니다. 특히, Soft-ZCA whitening 기법을 제안하여 임베딩(embedding) 공간에서의 anisotropy(비등방성) 문제를 해결하고자 하였습니다.

- **Technical Details**: 임베딩의 whitening 과정은 변수나 특징을 직교성(orthogonality)으로 변환하는 일반적인 처리 단계입니다. ZCA whitening은 원본 데이터와 가장 높은 상관관계를 유지하며, 임베딩 공간에 가장 적합한 방법으로 알려져 있습니다. 본 논문에서는 Soft-ZCA라는 새로운 기술을 도입하여 whitening 정도를 조절하는 방식으로 ZCA whitening을 수정하였습니다.

- **Performance Highlights**: Soft-ZCA whitening 기술의 적용은 사전 훈련(pre-trained) 및 미세 조정(fine-tuned)된 코드 언어 모델에서 코드 검색 성능을 개선하는 것으로 확인되었습니다. 실험 결과, 다양한 프로그래밍 언어에 대한 성능 향상을 보여주며, 또한 낮은 자원(low-resource) 프로그래밍 언어 데이터셋에 대한 일반화 능력을 평가하였습니다. 이 연구는 효과적인 사후 처리 기술이 코드 검색 작업에 미치는 긍정적인 영향을 입증합니다.



### "Stupid robot, I want to speak to a human!" User Frustration Detection in Task-Oriented Dialog Systems (https://arxiv.org/abs/2411.17437)
- **What's New**: 이 논문에서는 사용자 불만족을 식별하는 것이 현대의 작업 지향 대화 시스템(TOD)에서 전체적인 사용자 만족도와 참여도, 유지를 위해 얼마나 중요한지를 강조합니다. 특히, 기존 연구에서는 주로 감정 감지에 초점을 맞추고 있으나, 실제 사용자 데이터를 고려하지 못하고 있는 문제를 지적합니다. 이를 해결하기 위해 우리는 사용자 불만족을 탐지하기 위한 다양한 접근 방식을 비교하고, 특히 LLM 기반의 방법이 기존의 오픈 소스 방법들보다 우수한 성능을 보임을 보여줍니다.

- **Technical Details**: 이 연구에서는 두 가지 접근 방법을 제안합니다. 첫 번째는 키워드 기반의 사용자 불만족 탐지 방법이며, 이는 감정 분석(sentiment analysis) 원칙에 기반하고 있습니다. 두 번째는 최근에 등장한 LLM 기반의 인컨텍스트 학습(in-context learning) 방법으로, 이는 실제 대화 시스템에서의 사용자 불만족 감지를 위해 설계되었습니다. 우리는 이러한 방법들이 사용자 불만족을 성공적으로 탐지하는 데 있어서 얼마나 효과적인지를 분석합니다.

- **Performance Highlights**: 논문에서 제안한 LLM 기반의 접근 방식은 내부 벤치마크에서 F1 점수가 16% 개선된 성능을 보여주었습니다. 기존의 오픈 소스 방법들과 비교했을 때, 이 연구에서는 사용자 불만족을 감지하는 데 있어 LLM이 가지는 장점을 부각시킵니다. 추후 연구 방향으로는 이러한 방법들이 실제 산업에서 어떻게 활용될 수 있을지를 제시하고 있습니다.



### One Mind, Many Tongues: A Deep Dive into Language-Agnostic Knowledge Neurons in Large Language Models (https://arxiv.org/abs/2411.17401)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 지식 저장 메커니즘을 탐구하는 새로운 벤치마크인 Rephrased Multilingual LAMA (RML-LAMA)를 제안합니다. 이 벤치마크는 4개 언어 계통, 7개 언어, 총 7,849개의 사실을 포함하여 다국어로 된 높은 품질의 쿼리를 제공합니다. 또한, 지식 로컬라이제이션을 위한 새로운 방법인 Multilingual Integrated Gradients with Uncertainty Estimation (MATRICE)를 개발하여 언어 비의존적인 지식 뉴런을 정확하게 위치시키는 데 기여합니다.

- **Technical Details**: RML-LAMA는 모든 언어에서 쿼리의 불확실성을 정량화하는 모듈을 설계하여 각 뉴런의 기여도를 평가합니다. 이 연구는 LLM의 아키텍처에 따라 적응형으로 베이스라인 벡터를 설정하고, 여러 언어와 쿼리에 걸쳐 지식 기여도의 불확실성을 측정하여 언어 비의존적인 기여 점수를 도출합니다. 나아가, Knowledge Neurons Selection 모듈을 통해 다양한 사실에 대해 동적 임계값을 설정하여 언어 비의존적인 지식 뉴런을 선택합니다.

- **Performance Highlights**: 실험 결과, 언어 비의존적인 지식 뉴런의 상당 부분이 마지막 몇 층에 분포하며, 이를 조작함으로써 여러 언어에서 해당 지식 표현에 영향을 미친다는 것을 보여주었습니다. 또한, 언어 비의존적인 지식 뉴런을 활용한 연구를 통해 교차 언어 지식 편집, 지식 증대, 새로운 지식 주입의 가능성을 탐구할 수 있었습니다. 이 결과는 LLM의 낮은 리소스 언어 성능 향상과 함께, 새로운 지식을 효과적으로 학습하고 이전 지식을 잊어버리는 문제를 해결하는 데 기여할 수 있음을 나타냅니다.



### Can LLMs be Good Graph Judger for Knowledge Graph Construction? (https://arxiv.org/abs/2411.17388)
- **What's New**: 이 연구에서는 비구조화된 데이터를 구조화된 지식 그래프(KGs)로 변환하기 위한 새로운 프레임워크인 GraphJudger를 제안합니다. GraphJudger는 세 가지 혁신적인 모듈인 entity-centric iterative text denoising, knowledge aware instruction tuning, graph judgement를 포함하여, 지식 그래프 생성에서 발생할 수 있는 여러 문제를 해결하는 데 중점을 두고 있습니다. 특히, 대규모 언어 모델(LLMs)의 역할을 단순한 예측자가 아닌 그래프 판단자로 확장하고자 하며, 이를 통해 더 높은 품질의 KGs를 구축할 수 있도록 합니다.

- **Technical Details**: GraphJudger는 다음 세 가지 도전과제를 해결하기 위해 세분화된 접근법을 제안합니다. 첫째, Entity-Centric Iterative Text Denoising 모듈을 통해 실제 문서에서 불필요한 정보를 제거하여 LLMs가 필요한 정보를 효과적으로 추출할 수 있도록 합니다. 둘째, Knowledge Aware Instruction Tuning 모듈은 LLM을 조정하여 도메인 지식의 이해를 높이고 지식 그래프 완성 작업의 성능을 향상시킵니다. 셋째, Graph Judgement 모듈은 생성된 triples의 올바름을 평가하여 잘못된 아이템을 걸러냄으로써 전체 KGs의 품질을 개선합니다.

- **Performance Highlights**: GraphJudger는 두 개의 일반 텍스트-그래프 쌍 데이터셋과 하나의 도메인 특정 텍스트-그래프 쌍 데이터셋에서 수행된 실험에서 기존 방법들보다 우수한 성능을 보였습니다. 이 연구는 지식 그래프 구축 과제에서 LLMs의 효율성을 높이고, 필수적인 노이즈 제거와 도메인 지식 강화 방안을 통해 실제 세계의 데이터를 더 잘 활용할 수 있는 방법을 제안합니다. GraphJudger의 성공적인 성능 향상은 KG 구축이 더욱 신뢰할 수 있는 정보를 포함할 수 있도록 해줍니다.



### The Extractive-Abstractive Spectrum: Uncovering Verifiability Trade-offs in LLM Generations (https://arxiv.org/abs/2411.17375)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)과 검색 엔진 간의 비교를 통해 정보의 검증 가능성과 유용성을 동시에 만족할 수 있는 시스템을 탐구합니다. 저자들은 특히 인용의 품질과 사용자의 선호도에 대한 인간 평가를 바탕으로, 다섯 가지의 운영 포인트를 정의하여 서로 다른 정보 검색 맥락에서 최적의 시스템 설계를 위한 추천을 제공합니다. 또한 사용자들이 신뢰도가 낮은 LLM 대신 검색 엔진을 선호하는 이유를 파악하여, 정보 품질 향상을 위한 기회를 제시합니다.

- **Technical Details**: 이 연구에서 저자들은 정보 검색 도구 간의 상호작용을 연구하는 과정에서 '추출-요약 스펙트럼(extractive-abstractive spectrum)'을 도입했습니다. 검색 엔진은 원본 웹 페이지에 대한 링크와 함께 정보를 제공하는 추출 방식(extractive)을 사용하고, LLM은 다양한 출처의 정보를 종합하여 인용 없이 응답하는 요약 방식(abstractive)입니다. 실험 결과는 변별력 있는 평가 지표로써 각 운영 포인트의 유용성과 검증 가능성을 평가하였으며, 특히 구글 제미니의 인용 정확도가 15%에 불과하다는 점을 강조합니다.

- **Performance Highlights**: 실험 결과, 추출 기반 시스템에서 요약 기반 시스템으로 이동함에 따라 유용성은 최대 200%까지 증가하는 반면, 올바르게 인용된 문장의 비율은 50% 감소하고 정보 검증을 위해 소요되는 시간은 최대 3배 증가하는 경향이 있었습니다. 이 연구는 고유한 성능을 지닌 여러 운영 포인트에 대한 대안적 접근을 제시하며, 향후 LLM 시스템이 검증 가능한 정보를 제공하는 데 기여할 수 있는 가능성을 탐구합니다.



### Fairness And Performance In Harmony: Data Debiasing Is All You Need (https://arxiv.org/abs/2411.17374)
- **What's New**: 이 연구는 870개의 프로필을 포함한 실제 대학 입학 데이터셋을 사용하여 머신 러닝(ML) 모델의 공정성을 조사합니다. XGB, Bi-LSTM, KNN 세 가지 ML 모델을 활용하여 성별 언어 편향을 제거하는 파이프라인을 제안하며, ML 모델이 인간 전문가보다 14.08%에서 18.79% 더 높은 공정성을 기록함을 보여줍니다. 또한, 성별 언어 편향 제거 후 모든 모델의 분류 정확도가 유지되거나 향상되는 성과를 보였습니다.

- **Technical Details**: 이 연구에서는 BERT(Bidirectional Encoder Representations from Transformers) 임베딩을 사용하여 고차원 텍스트 특징을 인코딩하고, 인간 전문가의 결정과 ML 모델의 일관성을 평가하기 위해 일관성 점수를 사용합니다. 성별 편향을 완화하기 위해 설계된 디바이빙 파이프라인은 입력 기능에서 성별 특정 언어를 제거하고, 사전 훈련된 BERT 모델을 활용하여 성별 추론을 수행합니다. 데이터 준비 과정에서는 BERT 임베딩을 생성하는 과정을 통해 각 프로필에 대해 결합된 표현을 형성합니다.

- **Performance Highlights**: 실험 결과, ML 모델이 인간 전문가의 결정보다 더 높은 일관성 점수를 기록하며 공정한 결정에서의 신뢰성을 진단합니다. 성별 편향 제거 후, 모델들의 분류 정확도가 향상되거나 유지되는 결과를 도출했습니다. 이러한 결과는 공정성과 성능이 동시에 존재할 수 있음을 검증하며, ML 모델이 높은 정확도를 유지하면서 입학 공정성을 향상시킬 수 있다는 가능성을 제시합니다.



### Different Bias Under Different Criteria: Assessing Bias in LLMs with a Fact-Based Approach (https://arxiv.org/abs/2411.17338)
Comments:
          Accepted in NeurIPS 2024 Workshop on Socially Responsible Language Modelling Research (SoLaR)

- **What's New**: 이번 연구는 기존의 동등성 기반 접근 방식과 다른 사실 기반 기준을 사용한 새로운 편향 평가 메트릭을 제안합니다. 연구진은 LLM의 출력을 실제 세계의 인구 통계 분포와 연관시키는 방식으로 편향을 측정하고자 했습니다. 이 연구는 여러 LLM 모델의 출력이 서로 다르게 평가됨을 보여주며, 다양한 관점에서의 평가가 필요하다는 점을 강조합니다.

- **Technical Details**: 편향 측정에는 동등성과 사실 기반 기준을 모두 고려한 새로운 메트릭, '통계적 정렬'이 도입되었습니다. 연구진은 성별과 연령 도메인을 포함하여 다양한 이항 그룹을 설정하고, 다양한 질문에 대한 응답을 통해 내재된 편향을 평가합니다. 편향 평가를 위해 '균형', '거부', '통계적으로 정렬된' 세 가지 비편향 상태를 정의하였습니다.

- **Performance Highlights**: 실험 결과, 응답자들은 LLM의 출력이 실제 세계의 인구 통계 분포와 가까울수록 긍정적으로 인식하는 경향이 있음을 보여주었습니다. 평가된 여러 LLM의 편향은 적용된 기준에 따라 달라졌으며, 이는 다양한 관점에서 편향을 평가하는 중요성을 강조합니다. 이 결과는 사실 기반 기준이 LLM 편향 평가에서 보다 일관되고 객관적인 기준을 제공할 수 있음을 시사합니다.



### Meaningless is better: hashing bias-inducing words in LLM prompts improves performance in logical reasoning and statistical learning (https://arxiv.org/abs/2411.17304)
- **What's New**: 이 논문에서는 'hashing'이라는 새로운 방법을 소개합니다. 이 방법은 대규모 언어 모델(LLMs)에서 편향을 유발할 수 있는 단어를 해시와 유사한 무의미한 식별자로 마스킹함으로써 인지적 편향(cognitive biases)과 외부 지식에 대한 의존도를 줄이는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구는 총 490개의 프롬프트를 포함하는 세 가지 실험 세트에서 테스트되었습니다. 카이제곱 검정(chi-square tests)을 사용한 통계 분석에서 모든 테스트된 시나리오에서 유의미한 개선이 나타났으며, LLama, ChatGPT, Copilot, Gemini 및 Mixtral 모델이 포함되었습니다.

- **Performance Highlights**: 해싱은 '린다(Linda)' 문제의 수정된 버전에서는 오류율을 감소시켰으며, 아이템 집합 추출(task)에서도 성능을 향상시켰습니다. 세 번째 실험에서는 린다 문제가 텍스트보다 표 형식으로 제공될 때에도 해싱이 효과적이라는 것을 발견했습니다. 전반적으로 이 방법은 편향 감소와 외부 지식 통합을 개선하는 것으로 나타났으나, 환각(hallucination) 비율은 모델 유형에 따라 일관되게 감소하지 않았습니다.



### ER2Score: LLM-based Explainable and Customizable Metric for Assessing Radiology Reports with Reward-Control Loss (https://arxiv.org/abs/2411.17301)
- **What's New**: 이번 연구에서는 자동 방사선 보고 생성(Automated Radiology Report Generation, R2Gen)의 평가 지표로 ER2Score를 제안합니다. ER2Score는 사용자 정의 기준에 맞춰 평가 기준을 조정할 수 있는 자동 평가 지표로, 기계 학습 모델의 보상 메커니즘을 활용하여 설계되었습니다. 이 평가 지표는 보고서의 전반적인 점수와 각각의 평가 항목에 대한 세부 점수를 제공하여 해석 가능성을 높입니다.

- **Technical Details**: ER2Score는 GPT-4를 활용하여 LLM(대형 언어 모델) 기반의 보상 모델을 훈련시키며, 보고서의 품질을 높이기 위해 '수용' 및 '거부' 샘플을 쌍으로 만들어 학습합니다. 이 과정에서, 정밀한 보상을 제공하는 맞춤형 손실 함수를 사용하여 모델이 다수의 평가 기준에 대해 개별 보상을 동시에 출력하도록 합니다. 특히, 다양한 평가 시스템을 지원하기 위한 유연한 훈련 방법론을 채택했습니다.

- **Performance Highlights**: 실험 결과 ER2Score는 기존 평가 지표들보다 사람의 평가와의 상관 관계가 더 높으며, 모델 선택에서 더 우수한 성능을 보였습니다. 이번 연구에서 제안한 ER2Score는 다양한 평가 기준에 맞춰 맞춤 분석이 가능하여 보고서 생성 품질을 향상시킬 수 있는 잠재력을 가지고 있음을 보여줍니다. 사용자가 특정 기준을 통해 평가 결과의 특정 측면을 파악할 수 있도록 지원하는 점도 중요한 장점입니다.



### An Attempt to Develop a Neural Parser based on Simplified Head-Driven Phrase Structure Grammar on Vietnames (https://arxiv.org/abs/2411.17270)
Comments:
          Accepted at SoICT 2024

- **What's New**: 이 논문에서는 단순화된 Head-Driven Phrase Structure Grammar (HPSG)를 기반으로 한 베트남어 신경 파서를 개발하는 것을 목표로 했습니다. 기존의 베트남어 데이터셋인 VietTreebank와 VnDT는 약 15%의 구문 및 의존성 트리 쌍이 HPSG 규칙을 따르지 않았습니다. 저자들은 트레이닝과 개발 세트의 샘플을 무작위로 치환하여 HPSG 규칙에 준수하도록 만든 후, PhoBERT 및 XLM-RoBERTa 모델을 활용한 HPSG 신경 파서를 제안했습니다.

- **Technical Details**: 이 연구에서는 VietTreebank와 VnDT 데이터셋의 불일치를 해결하고자 하였으며, PhoBERT와 XLM-RoBERTa 모델을 통합하여 베트남어 구문 분석기의 성능을 향상시키려 하였습니다. 실험 결과, 동일한 파트-오브-스피치 (POS) 태그를 사용했을 때, 단순화된 HPSG 신경 파서가 82%의 F-score를 기록하며 세계 선두의 성능을 보여주었습니다. 하지만 Labeled Attachment Score (LAS)는 낮게 나타났으며, 이는 언어 전문가와의 협의 없이 원래 레이블을 변경하지 않고 아크 치환에 집중한 결과로 보입니다.

- **Performance Highlights**: 이 연구는 VLSP 2023 베트남어 구성 구문 분석 챌린지를 통해 제안된 헤드 규칙을 이용하여 구성 트리를 의존성 트리로 변환하는 연구도 포함하고 있습니다. HPSG 신경 파서는 89.04%의 F-score를 기록하며 기존의 Stanza 구성 파서(88.73%)를 초과하는 성과를 보였습니다. 이 결과는 베트남어 자연어 처리 도구 개발에 있어 언어 전문 지식을 통합하는 것이 유망하다는 것을 시사합니다.



### A Topic-level Self-Correctional Approach to Mitigate Hallucinations in MLLMs (https://arxiv.org/abs/2411.17265)
- **What's New**: 이 논문에서는 인간의 선호와 더 잘 일치하도록 다중 모달 대형 언어 모델(MLLM)의 행동을 조정하는 것이 필수적임을 강조합니다. 최근에는 전문가나 보조 AI 시스템을 활용하여 선호 피드백을 더욱 정확하게 제공하려는 시도가 있었으나, 이러한 방식은 자원이 많이 소모되어 확장성에 문제가 있었습니다. 이에 대해 'Topic-level Preference Overwriting (TPO)'라는 자가 교정 방식을 도입하여 모델 자체가 주제 수준에서 오차를 줄일 수 있도록 하였습니다.

- **Technical Details**: TPO는 모델이 스스로의 환각(hallucination)을 감지하고 수정할 수 있도록 하여 피드백 수집의 자동화 및 확장성을 목표로 합니다. 이 방식은 복잡한 응답을 구성하는 다양한 주제를 분리하고, 각 주제에 대해 모델이 생성한 최적의 대안으로 교체하여 더 뚜렷한 선호 쌍을 만들어 냅니다. 주제 수준의 수정으로 응답의 의미와 구조가 단순화되어 보다 정확한 후보 재샘플링이 가능해지고, 피드백 수집 과정에서 일관성을 유지할 수 있습니다.

- **Performance Highlights**: 'TPO'는 기존의 환각 벤치마크에서 신뢰성 면에서 최첨단의 성능을 보여줍니다. ObjectHal-Bench에서 모델의 환각을 약 92% 감소시켰고, MMHal-Bench에서는 38% 감소에 성공했습니다. 이러한 성과는 모델 자체가 선호 쌍을 생성하도록 설정하여 이루어진 것으로, 인간 피드백이나 독점 모델의 개입 없이도 가능했습니다.



### Strategic Prompting for Conversational Tasks: A Comparative Analysis of Large Language Models Across Diverse Conversational Tasks (https://arxiv.org/abs/2411.17204)
Comments:
          37 pages, 12 tables

- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Models, LLM)인 Llama, OPT, Falcon, Alpaca 및 MPT의 성능을 종합적으로 평가한 연구를 제시합니다. 다양한 대화 과제를 포함한 이 연구에서는 특히 예약, 공감 반응 생성, 정신 건강 및 법률 상담, 설득, 협상 등의 분야에서 모델의 기능과 한계를 분석합니다. 여러 평가 기준을 통해 철저하게 검증하여 특정 작업에 따라 모델의 성능이 다르게 나타난다는 결과를 도출하였습니다.

- **Technical Details**: 이 연구에서는 음성 인식 및 자연어 처리(Natural Language Processing, NLP) 작업에서 LLM의 성능을 높은 정확도로 평가하기 위해 자동 및 인적 평가 기준을 포함한 정교한 테스트 설정을 사용했습니다. 각 모델의 성능을 올바르게 게이지하기 위해 일반적인 메트릭과 특정 작업에 적합한 메트릭을 활용합니다. 검증을 통해 특정 작업에 맞춰 적절한 LLM을 선택하는 것이 중요함을 강조하고 있습니다.

- **Performance Highlights**: 각 LLM은 특정 과제에서 최상의 성능을 보였으나, 모든 과제에서 보편적으로 최적의 모델은 없으며, 성능은 작업의 요구 사항에 따라 크게 달라진다는 점이 시사됩니다. 실험 결과, 일부 모델이 특정 작업에서 뛰어난 성능을 나타냈지만, 다른 작업에서는 상대적으로 낮은 성능을 보이는 경향이 있었습니다. 이는 대화 애플리케이션에서 가장 적합한 LLM을 선택할 때 작업의 특정 요구 사항과 특성을 고려해야 함을 강조합니다.



### A Novel Word Pair-based Gaussian Sentence Similarity Algorithm For Bengali Extractive Text Summarization (https://arxiv.org/abs/2411.17181)
Comments:
          Submitted to ACM Transaction on Asian and Low-resource Language Information Processing

- **What's New**: 이번 연구에서는 새로운 Word pair-based Gaussian Sentence Similarity (WGSS) 알고리즘을 제안하여, 두 문장 간의 의미적 관계를 계산합니다. 이 방법은 개별 단어 임베딩 벡터의 기하 평균을 사용하여 문장 간 의미적 유사성을 평가합니다. 기존의 단어 평균 방식에서 발생하는 문장 표현 문제를 수정하여 보다 정교한 요약을 가능하게 합니다.

- **Technical Details**: 추출적 요약의 핵심 접근 방식은 문장을 선택하는 방법입니다. 클러스터링 기법을 통해 의미적으로 유사한 문장들을 그룹화한 후 TF-IDF 순위를 통해 각 클러스터에서 최상의 문장을 선택합니다. 제안된 방법은 4개의 다양한 데이터 세트에서 검증되었으며, 평균 ROUGE 점수에서 43.2% 더 우수한 결과를 보였습니다.

- **Performance Highlights**: 제안된 방법은 벵골어를 포함한 여러 저자원 언어에서 비슷한 성능을 보여줍니다. 또한, 고품질의 벵골어 데이터세트를 새롭게 수집하였으며, 이는 250개의 기사와 각 기사의 요약 쌍을 포함합니다. 연구의 결과는 자동화된 텍스트 요약 작업에서 특정 언어에 국한되지 않고 일반화할 수 있는 가능성을 제시하고 있습니다.



### Learning Monotonic Attention in Transducer for Streaming Generation (https://arxiv.org/abs/2411.17170)
Comments:
          Codes: this https URL

- **What's New**: 이 논문은 Streaming generation 분야에서 Transducer 구조를 개선하기 위한 새로운 접근 방식을 제안합니다. 입력 스트림의 역사(history)를 통해 디코딩을 조정하는 learnable monotonic attention 메커니즘을 통합하여, 비순차적인 정렬(non-monotonic alignment) 문제를 해결하는 것을 목표로 합니다. 이 방법론은 forward-backward 알고리즘을 활용하여 예측자 상태(predictor states)와 입력 타임스탬프 간의 posterior probability를 추론합니다.

- **Technical Details**: 논문에서 제안하는 MonoAttn-Transducer는 Transducer 모델의 디코딩 과정에서 비순차적 정렬을 처리하는 성능을 크게 향상시키는 기술을 포함합니다. 즉, 예측자가 실시간으로 소스 히스토리를 활용할 수 있게 하여, attention의 범위를 예측에 따라 적응적으로 조정합니다. 이러한 방식으로 학습하는 동안 기하급수적으로 큰 정렬 공간을 열거할 필요가 없으며, 이는 훈련 과정의 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, MonoAttn-Transducer는 음성-텍스트 및 음성-음성 동시 번역 작업에서 생성 품질을 눈에 띄게 개선하면서도 지연(latency) 증가를 피하는 놀라운 성능을 보여주었습니다. 특히 높은 비순차성 수준을 가진 샘플을 처리하는 데 강력한 효과를 나타내며, 이를 통해 종합적인 Streaming generation 작업의 복잡성을 효과적으로 다룰 수 있는 가능성을 제시합니다.



### Star Attention: Efficient LLM Inference over Long Sequences (https://arxiv.org/abs/2411.17116)
Comments:
          Code: this https URL

- **What's New**: Star Attention은 두 단계의 블록 희소 근사를 통해 긴 시퀀스에서의 추론을 향상시키는 새로운 알고리즘입니다. 이 방법은 여러 호스트에 걸쳐 주의를 분산시키며 통신 비용을 최소화하여 계산 효율성을 개선합니다. Star Attention은 기존 Transformer 기반 LLM과의 호환성을 가지며, 추가적인 모델 파인튜닝 없이 사용 가능합니다.

- **Technical Details**: Star Attention은 두 단계로 되어 있습니다: 첫 번째 단계에서는 컨텍스트를 블록으로 나누어 여러 호스트에 분산 처리하고, 두 번째 단계에서는 쿼리와 응답 토큰이 모든 이전 캐시된 토큰을 대상으로 주의를 기울입니다. 이 알고리즘은 각 호스트의 할당된 블록에 대한 self-attention만을 계산하여 복잡도를 제곱에서 선형으로 감소시킵니다. 또한 쿼리 호스트는 전 세계적으로 결과를 집계하여 기존 KV 캐시를 업데이트합니다.

- **Performance Highlights**: Star Attention은 Llama3.1-8B 및 Llama3.1-70B 모델에서 95-100%의 정확도를 유지하면서 최대 11배 빠른 추론을 달성합니다. 또한 Flash Attention이나 KV 캐시 압축과 같은 다른 LLM 최적화 기법과 결합하여 추가적인 성능 향상을 가능하게 합니다.



### Don't Command, Cultivate: An Exploratory Study of System-2 Alignmen (https://arxiv.org/abs/2411.17075)
Comments:
          Preprint version, more results will be updated

- **What's New**: 이 연구는 OpenAI의 o1 모델이 빠른, 직관적 사고(System-1)에서 느린, 더 신중한 사고(System-2)로의 발전을 통해 더 안전하게 작동할 수 있는 가능성을 보여준다. 연구진은 o1의 안전성을 평가하며 복잡한 jailbreak 공격 시나리오를 포함한 다양한 조건에서 모델의 반응을 분석하였다. 결과적으로 o1 모델은 상대적으로 향상된 안전 성능을 나타내지만, 여전히 수학적 인코딩을 활용한 공격에 대해 취약함을 보였다.

- **Technical Details**: 인간의 사고 패턴을 System-1과 System-2로 구분할 수 있으며, System-2의 사고를 활용하는 것은 모델의 안전성을 증가시키는 중요한 요소로 작용한다. o1 모델의 안전성 평가를 위해 adversarial natural language prompts 및 mathematically encoded prompts를 활용하여 연구를 진행하였다. 실험 결과, 느린 사고 메커니즘을 이용하는 모델이 긍정적인 안전 성과를 나타내지만, 여전히 수학적 인코딩 공격에 특히 취약한 것으로 확인되었다.

- **Performance Highlights**: 안전성 평가에서 o1 모델은 adversarial harmful 및 benign prompts를 처리하면서 높은 내구성을 보였으나, 가끔씩 과잉 거부(overrefusal)를 나타내기도 하였다. 수학적 인코딩으로 구성된 jailbreak 공격에 대해서는 o1 모델이 뚜렷한 취약점을 드러내었으며, 향후 연구는 사고 체계의 안전성과 강건성을 보장하는 방법을 모색하는 데 중점을 둬야 할 것이다. 이에 따라 모델이 사용자 요청을 신중하게 분석토록 유도하는 간단한 개입이 안전성 향상에 도움이 된다는 것을 확인하였다.



### Tree Transformers are an Ineffective Model of Syntactic Constituency (https://arxiv.org/abs/2411.16993)
- **What's New**: 본 논문은 현대 자연어 처리(NLP) 모델들이 자연어의 계층적 구조를 효과적으로 인식하고 활용하는지에 대해 탐구합니다. 특히, Tree Transformer 구조를 사용하여 기존의 Transformer 모델과 비교하고, 해당 모델이 의미 있는 구성 요소를 얼마나 잘 학습하는지를 분석합니다. 연구 결과, Tree Transformer 모델이 약간의 성능 향상을 보였지만, 언어 구조의 유의미성을 향상시키는 데에는 한계가 있음을 발견하였습니다.

- **Technical Details**: 이 연구는 Tree Transformer 아키텍처를 사용하여 사전 학습된 대규모 언어 모델을 분석하며, 여기서 구성 요소 주의(attention) 메커니즘을 통합하여 구문 구조를 학습합니다. 실험을 통해 구문 유닛, 예를 들어 수사(detectors), 형용사(adjectives), 관계절(relative clauses) 등을 다루는 방식과 모델의 구문 분석(syntactic parse) 결과를 비교합니다. 그럼에도 불구하고, 모델이 도출한 구문 구조는 기존의 언어학적 믿음과는 상당히 다름을 보여주었습니다.

- **Performance Highlights**: Tree Transformer 모델은 구문 오류 탐지 작업에서 이전 모델보다 약간 향상된 결과를 보였지만, 기대했던 것만큼 효과적인 결과를 제공하지 못했습니다. 이 연구는 비계층적(prehierarchical) 모델의 사전 훈련이 반드시 필요한 계층적 편향을 부여하는 것은 아니란 점을 강조합니다. 전반적으로, Tree Transformer가 언어 모델링에서 더 효과적이라는 증거는 부족하다는 결론을 내렸습니다.



### Dynamic Self-Distillation via Previous Mini-batches for Fine-tuning Small Language Models (https://arxiv.org/abs/2411.16991)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 기존의 Knowledge Distillation (KD) 기법의 한계를 극복하기 위해 새로운 Self-distillation 방법인 Dynamic SelfD from the previous minibatch (DynSDPB)를 제안합니다. 이 기법은 간단한 아키텍처 수정 없이도 언어 모델이 스스로 학습할 수 있도록 하며, 작은 언어 모델(SLMs)의 성능을 극대화하는 데 중점을 둡니다. 또한, DynSDPB는 여러 최신 언어 모델에서 효과적으로 적용될 수 있도록 설계되었습니다.

- **Technical Details**: DynSDPB는 마지막 미니배치(mini-batch)에서 생성된 logits를 활용하여 현재 미니배치의 학습을 안내하는 혁신적인 SelfD 기법입니다. 이 방법은 언어 모델이 예측 불확실성과 판별 능력에 따라 SelfD 설정(distillation factor α 및 temperature τ)을 동적으로 조정할 수 있도록 지원합니다. 또한, Vocabulary Map Matching (VMM)이라는 새로운 방식을 통해 동일 입력에 대한 출력 차원 불일치를 해결합니다.

- **Performance Highlights**: DynSDPB는 NLU(자연어 이해)와 NLG(자연어 생성) 벤치마크에서 encoder-only (예: BERT) 및 decoder-only (예: LLaMA) 모델 모두에서 우수한 성능을 보여줍니다. 주요 한계없이 LLMs에 접근하지 않고도 작은 언어 모델의 미세 조정을 효과적으로 수행할 수 있는 능력을 갖추고 있습니다. 실험 결과, DynSDPB는 기존 Self-Training/Correction 기술과의 원활한 통합이 가능하다는 점에서도 주목받고 있습니다.



### Teaching Smaller Language Models To Generalise To Unseen Compositional Questions (Full Thesis) (https://arxiv.org/abs/2411.16985)
- **What's New**: 이 연구에서는 Pretrained large Language Models (LLMs)의 한계를 넘어서, 네트워크 연결이 없는 상황에서 로컬 컴퓨팅 자원을 이용하여 주어진 질문에 답할 수 있는 Reasoning Models을 개발하였다. 특히, 초기 훈련 시 접하지 않은 질문에 대처할 수 있도록 설계된 이 모델은 유연한 질문 응답 성능을 보임을 목표로 한다. 연구진은 다양한 지식 출처에서 맥락을 도출해내어 훈련함으로써 모델의 고유한 기능을 강조하였다.

- **Technical Details**: 이 모델은 Wikipedia와 더 큰 Language Model에서 생성한 Rationale를 활용하여 정보 맥락을 끌어낸다. 연구팀은 novel retrieval-augmented training datasets (RATD)를 도입하여 Reasoning Model의 훈련을 보강하였고, 이로 인해 효과적인 상황에서 답변 가능성을 크게 향상시킬 수 있었다. Rationale Ranking 모델을 구축하여 생성된 Rationale와 가져온 컨텍스트의 관련성과 진실성을 평가하는 또한 중요한 방법론으로 사용되었다.

- **Performance Highlights**: 모델의 성능 개선은 unseen evaluation datasets에서 체계적으로 검증되었으며, novel RATD의 추가는 결과를 극적으로 증가시켰다. 두 출처에서 지식을 결합하는 기법을 통해 모델의 대처 능력도 더욱 향상되었다. 이러한 연구 결과는 자원 제약이 있는 환경에서도 LLM의 장점을 활용할 수 있는 가능성을 보여준다.



### Harnessing LLMs for Educational Content-Driven Italian Crossword Generation (https://arxiv.org/abs/2411.16936)
Comments:
          This paper has been accepted for presentation at this http URL 2024

- **What's New**: 이번 연구에서는 Italian crossword puzzles(크로스워드 퍼즐)을 텍스트에서 생성하는 혁신적인 도구를 소개합니다. 이 도구는 GPT-4o, Mistral-7B-Instruct-v0.3, Llama3-8b-Instruct와 같은 고급 언어 모델을 활용하며, 30,000개 이상의 항목이 포함된 Italian-Clue-Instruct 데이터셋을 사용하여 교육용 어플리케이션을 목적으로 합니다. 이 연구는 교육 경험을 개선하는 데 기여하고, 상호작용적인 플랫폼을 통해 인지적 발전을 도울こと을 목표로 하고 있습니다.

- **Technical Details**: 이 연구에서는 LLMs(대형 언어 모델)을 사용하여 이탈리아어 교육용 크로스워드 퍼즐을 생성하는 자동화 시스템을 개발했습니다. 데이터는 이탈리아어 Wikipedia에서 주요 키워드를 추출하여 수집하였으며, 이를 통해 다양한 메타데이터를 포함한 포괄적인 데이터 레포지토리를 구축했습니다. 또한 각기 다른 구조의 클루를 생성하기 위해 세 가지 유형의 프롬프트를 작성하고, 특정 구조를 명시하지 않는 프롬프트도 추가하여 총 네 가지 접근 방식을 설정했습니다.

- **Performance Highlights**: 본 연구의 도구는 이탈리아어 교육 자료에서 동적으로 크로스워드 퍼즐을 생성하여 학습자가 즐겁고 상호작용적인 학습 환경을 경험할 수 있도록 합니다. 이를 통해 언어 교육을 위한 도구 키트를 풍부하게 하며, 크로스워드 퍼즐 생성에 있어 기존의 한계를 뛰어넘는 성능을 보여주고 있습니다. 이 새로운 접근법은 효과적인 교육 툴로 자리매김하며, 이탈리아어 교육의 새로운 기준을 설정하는 데 기여할 것입니다.



### Integrating Geodesic Interpolation and Flow Matching for Non-Autoregressive Text Generation in Logit Spac (https://arxiv.org/abs/2411.16821)
- **What's New**: 비자율 언어 모델(Non-autoregressive Language Models)은 자연어 처리 분야에서 자율 모델(Autoregressive Models)의 효과적인 대안으로 떠오르고 있습니다. 본 연구에서는 Kullback-Leibler(KL) 발산을 이용한 새로운 흐름 매칭(Flow Matching) 접근법을 제안하여, 초기 분포와 목표 분포 사이의 보간을 수행합니다. 이 방법은 이산 시퀀스에 대한 정확한 최대 조건 우도를 예측하는 손실 함수를 수식적으로 정립하고 있습니다.

- **Technical Details**: 흐름 매칭의 핵심 개념은 알려진 시작 분포와 타겟 분포 사이의 확률 경로를 설정하는 것입니다. 모델은 비유적(One-hot) 벡터로 표현된 토큰을 사용하여 이산적인 텍스트의 복잡한 의존성을 포착합니다. 우리는 KL 발산 하에 지오데식을 활용하여 단순한 형태의 보간 방법을 제시하며, 최근 연구 결과를 기반으로 다양한 손실 함수 중 조건부 우도를 최대화하는 방식을 선택하였습니다.

- **Performance Highlights**: 비록 TinyStories 데이터셋에서 초기 실험 결과가 다소 비효율적이었으나, 미리 훈련된 노이즈 제거기를 사용한 새로운 샘플링 방식을 통해 성능이 상당히 향상되었습니다. 또한, Fine Web과 Lamini Instruction과 같은 더 복잡한 데이터셋에서 강력한 성능을 확인한 하이브리드 접근법을 제시하였습니다. 이는 이론적 기반은 부족하지만 실험 결과에서 두드러진 향상을 보여주고 있습니다.



### Enhancing In-Hospital Mortality Prediction Using Multi-Representational Learning with LLM-Generated Expert Summaries (https://arxiv.org/abs/2411.16818)
- **What's New**: 이번 연구에서는 ICU(중환자실) 환자의 병원 내 사망률(IHM) 예측을 위해 구조화된 생리학적 데이터와 비구조화된 임상 노트를 통합하여 개선된 예측 모델을 제안합니다. 특히, 이 모델에는 대형 언어 모델(LLM)인 Med42-v2 70B로 생성된 전문가 요약이 포함됩니다. 이러한 통합 접근 방식은 예측 정확성을 높이고, 예측 모델의 적용 가능성을 모두 고려한 결과를 도출합니다.

- **Technical Details**: 연구팀은 MIMIC-III 데이터베이스를 사용하여 ICU 입원 첫 48시간 동안의 생리학적 데이터와 임상 노트를 분석했습니다. 임상 노트는 환자별로 연대순으로 연결되었고, LLM을 활용하여 전문가 요약으로 변환되었습니다. 연구에서는 LLM이 제공하는 도메인 지식을 통해 구조화된 데이터와 비구조화된 데이터의 연결을 강화하는 다중 표현 학습 프레임워크를 개발했습니다.

- **Performance Highlights**: 제안된 모델은 AUPRC 0.6156(+36.41%) 및 AUROC 0.8955(+7.64%)를 달성하여 단일 시간 시계열 데이터 모델보다 우수한 성능을 보였습니다. LLM에서 생성된 전문가 요약이 임상 노트나 생리학적 데이터 단독보다 나은 성능을 발휘하며, 인구 집단 간의 성능 향상 또한 일정했습니다. 이 연구는 LLM이 중환자 예측 모델에서 실질적인 기여를 할 수 있음을 입증했습니다.



### Fine-Tuning LLMs with Noisy Data for Political Argument Generation (https://arxiv.org/abs/2411.16813)
- **What's New**: 본 연구는 정치적 민감성을 가진 콘텐츠 생성에 있어 소셜 미디어의 비례성과 비인격화를 줄이기 위한 새로운 방법으로, GPT-3.5 Turbo 모델의 미세 조정(fine-tuning)과 프롬프트(prompting) 전략을 탐색합니다. 특히 CLAPTON 데이터셋을 사용하여 Twitter와 Reddit의 정치적 논의 게시물을 분석하였으며, 이들 데이터셋의 특성에 따라 다양한 결과를 도출했습니다.

- **Technical Details**: 연구에서는 Reddit 데이터에 대한 미세 조정이 논의의 질을 향상시키는 데 가장 효과적이라는 점을 확인했습니다. 반면에 결합된 잡음이 많은 데이터는 지속적인 독성을 초래하는 경향이 있었고, 특정 독성 특성(예: 개인 공격)에 대한 프롬프트 전략의 영향은 제한적임을 알게 되었습니다. 고품질 데이터와 잘 구성된 프롬프트가 자동 정치 담론 생성에서 비례성을 줄이고 수사적 질을 향상시키는 데 필수적이라는 주장을 하고 있습니다.

- **Performance Highlights**: Reddit 데이터로 미세 조정된 모델은 논의 품질에서 가장 높은 점수를 기록하였으며, 이는 정확한 사전 조정 및 데이터 선택의 중요성을 보여줍니다. 반면에 다양한 소셜 미디어 출처의 잡음이 많은 데이터는 전반적인 독성을 증가시키는 결과를 초래했습니다. 본 연구의 결과는 정치적 담론의 자동 생성 과정에서 비례성을 줄이고 품질을 높이기 위해 더욱 깊이 있는 접근이 필요함을 시사합니다.



### Enhancing Answer Reliability Through Inter-Model Consensus of Large Language Models (https://arxiv.org/abs/2411.16797)
Comments:
          15 pages, 2 figures

- **What's New**: 본 연구는 최근의 혁신적인 언어 모델 상호작용 시스템을 탐색하였으며, GPT-4-0125-preview, Meta-LLaMA-3-70B-Instruct, Claude-3-Opus, 그리고 Gemini-1.5-Flash와 같은 고급 모델들이 포함됩니다. 이 모델들은 정확한 정답이 없는 복잡한 PhD 수준의 통계 질문을 생성하고 응답할 수 있는 능력을 보여줍니다. 연구는 모델 간 합의가 응답의 신뢰성과 정확성을 향상시킨다는 점을 조사하고, 이를 통해 AI 시스템의 자율적 협력적 추론과 검증에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 우리는 카이제곱 검정(chi-square tests), Fleiss' Kappa, 신뢰 구간 분석(confidence interval analysis)과 같은 통계적 방법을 사용하여 협력적 출력의 신뢰성을 정량화했습니다. 핵심 결과에 따르면 Claude와 GPT-4가 가장 높은 신뢰성과 일관성을 보여주었으며, 이는 그들의 좁은 신뢰 구간과 질문 생성 모델과의 높은 정렬에서 확인되었습니다. 반면, Gemini와 LLaMA는 더 큰 변동성을 보였으며, 이는 넓은 신뢰 구간과 낮은 신뢰성 비율로 나타났습니다.

- **Performance Highlights**: 이 연구는 다양한 LLMs 간의 협업이 응답의 신뢰성을 유의미하게 향상시킨다는 것을 증명했습니다. 또한, 협업 검증 접근 방식의 효율성과 효과성에 대한 실증적 증거를 제공하며, 향후 연구를 위한 기준을 정립했습니다. 이러한 발견은 교육 기술, 자동 평가 시스템, 전문 학문 영역의 연구 검증 및 AI 기반 지식 시스템의 개선 등 다양한 분야에 중요한 시사점을 제공합니다.



### What can LLM tell us about cities? (https://arxiv.org/abs/2411.16791)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)이 도시와 지역에 대한 글로벌 정보를 제공하는 능력을 조사합니다. 연구진은 두 가지 방법을 활용하여 LLM에 대한 직접 쿼리 및 목표 변수와 상관관계가 있는 명시적 및 암시적 특징을 추출했습니다. 실험 결과, LLM이 글로벌 도시 전반에 걸쳐 다양한 정도의 지식을 포함하고 있으며, LLM 파생 특징으로 학습된 ML 모델이 일관되게 예측 정확도를 향상시킴을 확인했습니다.

- **Technical Details**: 실험은 LLM을 통해 도시와 지역에 대한 정보를 쿼리하고 해당 정보를 기반으로 특징을 추출하는 두 가지 접근 방식을 사용하였습니다. 연구에서는 금방 자주 나타나는 특정 지식 부족을 분석하며, 이러한 경우 LLM이 생성하는 출력이 일반적이거나 무작위적임을 발견했습니다. LLM의 이러한 특성은 데이터 기반 의사결정의 새로운 기회를 제공할 수 있음을 시사합니다.

- **Performance Highlights**: LLM은 전 세계 모든 대륙의 도시들에 대한 고유한 수준의 지식을 보여주며, 특히 특정 도시나 지역에 대한 정보가 부족할 때 일반적이거나 무작위적인 결과를 생성함을 보여주었습니다. 이러한 결과는 LLM을 활용하여 도시 연구에서 데이터 중심의 결정을 내리는 데 기여할 수 있음을 강조합니다.



### Contrastive Multi-graph Learning with Neighbor Hierarchical Sifting for Semi-supervised Text Classification (https://arxiv.org/abs/2411.16787)
Comments:
          16 pages, 6 figures

- **What's New**: 이번 논문에서는 반지도 학습(semi-supervised learning)에서 텍스트 분류를 개선하기 위해 새로운 방법인 ConNHS를 제안합니다. 기존의 그래프 증강(graph augmentation) 방식을 피하고, 이 새로운 방법은 멀티 관계형 텍스트 그래프를 구성하여 텍스트 간 의미적 연결을 강화합니다. Neighbor Hierarchical Sifting (NHS) 손실 함수를 도입하여 부정 샘플(false negatives) 문제를 해결하고, 텍스트의 의미 정보를 최적화하여 더 안정적인 분류 결과를 제공합니다.

- **Technical Details**: 제안된 방법은 멀티-관계 텍스트 그래프를 기반으로 하며, RW-GCN(Relation-Aware Graph Convolutional Network)와 CGAN(Cross-Graph Attention Network)을 활용하여 노드 간의 다양한 상관관계와 엣지 특징(edge features)을 고려합니다. NHS 손실 함수는 고차 이웃을 배제하여 부정 샘플이 발생하는 것을 막고, 노드의 유사성을 기반으로 하여 적절한 샘플링을 수행합니다. 이 방식은 그래프 간 객체 간 정보 융합을 조화롭게 이루어지도록 돕습니다.

- **Performance Highlights**: 실험 결과, 저자들은 ThuCNews, SogouNews, 20 Newsgroups, Ohsumed 데이터셋에서 각각 95.86%, 97.52%, 87.43%, 70.65%라는 성능을 달성했습니다. 이는 반지도 텍스트 분류에서 경쟁력 있는 결과를 나타내며, 기존 방법들에 비해 더 높은 정확도를 보여줍니다. 따라서 ConNHS는 그래프 기반 텍스트 분류의 새로운 가능성을 제시하는 연구로 평가받을 수 있습니다.



### Parameter Efficient Instruction Tuning: An Empirical Study (https://arxiv.org/abs/2411.16775)
Comments:
          7 pages, 7 figures

- **What's New**: 본 연구는 수많은 PEFT(파라미터 효율적 미세 조정) 방법을 체계적으로 조사하여 각 방법이 성능에 미치는 영향을 분석합니다. 연구진은 LoRA와 Adapter가 최상의 성능을 발휘하며, 본 연구는 이상적인 훈련 설정에서 이들 방법의 강점을 강조합니다. 특히, 훈련 하이퍼파라미터와 모델 크기가 성능에 미치는 상관관계를 구체적으로 조사하였습니다.

- **Technical Details**: 효과적인 PEFT 방법을 식별하기 위해 연구팀은 SuperNI 데이터셋에서 훈련하였으며, 다양한 하이퍼파라미터와 모델 크기를 설정하여 실험을 진행했습니다. LoRA와 Adapter는 훈련 안정성이 상대적으로 낮은 것으로 나타났으며, 이는 더 높은 학습률과 큰 롤라 랭크와 연관이 있습니다. 다양한 PEFT 방법들의 훈련 설정을 통해 이들의 효율성도 분석하였습니다.

- **Performance Highlights**: 실험 결과, LoRA와 Adapter는 복잡한 추론, 코딩 및 긴 형태의 생성에서 약한 성능을 보였습니다. 그러나 open instruction tuning 환경에서는 LoRA가 Adapter보다 더 나은 기능을 발휘하며, 이는 PEFT가 기존의 전통적인 미세 조정 대안으로 적합함을 보여줍니다. LoRA의 경우, 제한된 데이터 환경에서 일반화 능력에 있어 상대적으로 낮은 성과를 보였고, 훈련 안정성 이슈도 존재합니다.



### SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction (https://arxiv.org/abs/2411.16765)
Comments:
          17 pages

- **What's New**: SHuBERT(사인 히든 유닛 BERT)는 자동 수화 처리 시스템의 발전을 가속화할 수 있는 자기 지도 학습 기반의 transformer 인코더입니다. 약 1,000시간의 미국 수화(ASL) 비디오 콘텐츠에서 강력한 표현을 학습하며, 기존의 테스크 특화 모델을 넘어서 다중 테스크에서의 이전 학습 가능성을 높입니다. SHuBERT는 다양한 비디오 스트림(손, 얼굴, 몸 자세)의 클러스터 할당을 예측하여 자가 감독 학습을 적용합니다.

- **Technical Details**: SHuBERT는 비디오 프레임의 문맥적 표현을 학습하기 위해 multi-stream 접근 방식을 채택하고, 손, 얼굴, 몸 자세 피처의 여러 masked 비디오 스트림을 사용하여 훈련합니다. 사용된 비디오 피처는 MediaPipe Hand Landmarker 모델을 통해 손의 특징을 검출하며, 검출의 정확도가 95%에 달합니다. SHuBERT는 이렇게 학습된 표현이 여러 하위 작업에 효과적으로 전달될 수 있음을 입증하였습니다.

- **Performance Highlights**: SHuBERT는 sign language translation(SLT)과 isolated sign language recognition(ISLR) 벤치마크에서 최첨단 성능을 달성하였습니다. 예를 들어, How2Sign 데이터셋에서는 +0.7 BLEU, OpenASL에서는 +10.0 BLEU, ASL-Citizen에서는 +5% 정확도를 초과하는 성과를 보였습니다. 이러한 결과는 전문화된 아키텍처에 비해 적은 테스크 특정 주석으로도 뛰어난 성과를 나타낼 수 있음을 보여줍니다.



### ChemSafetyBench: Benchmarking LLM Safety on Chemistry Domain (https://arxiv.org/abs/2411.16736)
- **What's New**: 이 논문에서는 ChemSafetyBench라는 새로운 벤치마크를 소개합니다. 이는 화학 분야에서 대규모 언어 모델(LLM)의 안전성과 정확성을 평가하기 위해 설계되었습니다. ChemSafetyBench는 화학 물질 질의, 화학 사용의 합법성 평가, 합성 방법 설명 등 3가지 주요 작업을 포함하며, 이러한 작업들은 점차적으로 깊은 화학 지식을 요구합니다.

- **Technical Details**: 데이터셋은 30,000개 이상의 샘플을 포함하고 있으며, 화학 물질의 속성, 용도 및 주요 합성 반응 정보를 포괄합니다. 또한, 수작업으로 수집한 데이터와 자동 평가 프레임워크를 통해 LLM의 안전성, 정확성, 적절성을 종합적으로 평가합니다. GHS(Globally Harmonized System)의 분류 시스템을 채택하여 화학 물질의 속성을 표현하며, 국제적인 협력 및 준수의 기준을 강화합니다.

- **Performance Highlights**: 최신 LLM과의 광범위한 실험을 통해 이 모델들의 강점과 주요 취약점을 발견하였습니다. 그러나 안전성 증진이 필수적이며, 이 연구는 화학 분야에서 화학 정보를 다룰 때의 안전성을 높이는 데 기여할 것입니다. ChemSafetyBench는 AI 기술을 보다 안전하게 발전시키는 데 중요한 도구로 자리 잡을 것으로 기대됩니다.



### Multi-Reranker: Maximizing performance of retrieval-augmented generation in the FinanceRAG challeng (https://arxiv.org/abs/2411.16732)
- **What's New**: 이번 연구에서는 금융 분야의 특정 문제들을 해결하기 위해 개발된 고성능의 Retrieval-Augmented Generation (RAG) 시스템을 소개합니다. 이 시스템은 ACM-ICAIF '24 FinanceRAG 대회에서 2위에 자리하며, LLMs (Large Language Models)를 활용하여 복잡한 금융 데이터를 효과적으로 처리하고 분석할 수 있는 가능성을 보여줍니다. 주요 기여로는 사전 검색 단계에서의 ablation 분석과 개선된 검색 알고리즘이 포함됩니다.

- **Technical Details**: 실행 성능을 최적화하기 위해 쿼리 확장과 코퍼스 정제를 통한 ablation 연구가 수행되었습니다. 검색 정확도를 높이기 위해 여러 reranker 모델을 채택하였으며, 생성 단계에서는 긴 컨텍스트 관리에 대한 효율적인 방법을 도입하였습니다. 이러한 접근은 응답 품질을 크게 향상시키면서도 성능 저하를 방지했습니다.

- **Performance Highlights**: 이번 연구는 FinanceRAG Challenge에서 2위를 기록하였으며, 이는 금융 데이터에 대한 정확하고 가치 있는 통찰력을 생성하기 위한 LLMs의 잠재력을 보여줍니다. 모든 세부 사항은 미리 준비된 소스 코드와 함께 제공되며, 이를 통해 다른 연구자들이 이 시스템을 재현하고 발전시킬 수 있도록 지원합니다.



### Enhancing LLMs for Power System Simulations: A Feedback-driven Multi-agent Framework (https://arxiv.org/abs/2411.16707)
- **What's New**: 이 논문은 대형 언어 모델(LLM)과 실험 기술을 결합하여 과학 연구를 변혁하고, AI를 단순한 문제 해결 도구가 아닌 다재다능한 연구 보조자로 자리매김하는 방법을 제시합니다. 특히 전력 시스템 분야에서 시뮬레이션 관리는 LLM의 제한적인 도메인 지식과 추론 능력으로 인해 여전히 도전 과제가 되고 있습니다. 이를 해결하기 위해 본 연구에서는 강화된 검색-보강 생성(RAG) 모듈과 개선된 추론 모듈, 동적 환경 작동 모듈을 포함하는 피드백 기반 다중 에이전트 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 LLM이 전력 시스템 시뮬레이션을 수행하기 위한 여러 모듈을 통합하고 있습니다. 강화된 RAG 모듈은 적응형 쿼리 계획 전략과 삼중 구조를 통해 LLM이 시뮬레이션의 기능 및 옵션을 더 잘 인식하고 해석할 수 있도록 돕습니다. 또한, 강화된 추론 모듈은 시뮬레이션에 특화된 전문 지식과 연결된 사고를 통해 LLM의 추론 능력을 향상시키고, 환경 상호작용 및 오류 교정 메커니즘을 통하여 시뮬레이션 결과의 신뢰성을 높입니다.

- **Performance Highlights**: 이 프레임워크는 69개의 다양한 작업을 통해 93.13%와 96.85%의 성공률을 달성하며, 최근 LLM인 ChatGPT 4o와 o1-preview의 27.77%와 0%와 비교할 때 눈에 띄게 우수한 성능을 보였습니다. 각 시뮬레이션은 평균 30초 이내에 완료되며, 비용은 약 0.014 USD로 매우 경제적입니다. 전체적으로 이 프레임워크는 인공지능 기반 연구 보조 개발의 기초를 마련하고, 전력 시스템 연구 및 그 외 분야에서 활용될 수 있는 가능성을 보여줍니다.



### Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens (https://arxiv.org/abs/2411.17691)
Comments:
          Work in progress; Please note that Figure 1's gray areas may not be displayed properly using Chrome (maybe due to bugs in Chrome)

- **What's New**: 본 연구에서는 낮은 비트 양자화가 언더트레인된 대형 언어 모델(LLM)에 유리하다는 사실을 밝혀냈습니다. 더 큰 모델이거나 훈련 토큰이 적은 경우, 낮은 비트 양자화를 적용해도 양자화 유도 저하(QiD)가 덜 발생하는 반면, 훈련 토큰이 많은 더 작은 모델은 상당한 QiD를 경험합니다. 이 연구에서 1500개 이상의 양자화된 LLM 체크포인트를 분석하였으며, 이 결과를 바탕으로 QiD를 활용하여 LLM의 훈련 수준을 측정할 수 있는 새로운 관점을 제시하였습니다.

- **Technical Details**: 본 연구에서는 다양한 크기의 LLM이 저비트 양자화에 어떻게 영향을 받는지를 실험적으로 분석하기 위해 훈련 토큰 수, 모델 크기, 비트 너비 등의 요소를 고려한 스케일링 법칙(scaling laws)을 도출하였습니다. 이 스케일링 법칙을 통해 QiD와 훈련 수준 사이의 관계를 규명할 수 있었으며, 이를 통해 LLM의 최적 훈련 토큰 수를 추정하는 방법을 제안하였습니다. 향후 100조 개의 훈련 토큰을 이용하여 훈련된 모델의 저비트 양자화 성능을 예측할 수 있는 데이터도 제공하였습니다.

- **Performance Highlights**: 미래의 LLM, 즉 100조 개의 훈련 토큰으로 훈련될 모델들은 저비트 양자화 성능이 바람직하지 않을 수 있다는 점을 강조하였습니다. 이는 저비트 양자화가 미래의 LLM에서 어떤 도전을 안고 있는지를 나타냅니다. 또한, 본 연구에서 제시하는 모든 1500개 이상의 양자화된 체크포인트는 향후 연구에 도움을 줄 수 있도록 공개되었습니다.



### Attamba: Attending To Multi-Token States (https://arxiv.org/abs/2411.17685)
- **What's New**: Attamba는 최신 아키텍처로, State-Space Models (SSM)을 활용하여 토큰 덩어리를 압축하고 이러한 압축된 키-값( key-value ) 표현에 대해 주의를 적용합니다. 기존의 Transformer 구조에서 키와 값의 프로젝션을 SSM으로 대체함으로써 모델 품질이 향상되고 유연한 토큰 청킹(token chunking) 기능이 구현되었습니다. 결과적으로 비슷한 KV-Cache와 주의적 공간을 가지면서도 24%의 혼란도(perplexity)가 개선되었고, KV-Cache 및 Attention FLOPs는 약 4배 더 작아졌습니다.

- **Technical Details**: Attamba는 변형된 Attention 메커니즘으로, SSM을 통해 여러 토큰을 하나의 의미 있는 상태로 압축하여 처리합니다. 이 구조는 토큰 덩어리를 동적으로 처리할 수 있도록 하며, 이는 L^2 주의 연산을 청크 크기만큼 줄이는 결과를 가져옵니다. 또한, 주기적인 토큰 청킹을 통해 고정된 경계를 줄이고, 가변 길이의 토큰 청킹을 가능하게 해 메모리 및 계산 비용을 효과적으로 절감합니다.

- **Performance Highlights**: Attamba는 고정된 경계를 사용하지 않고 청크 경계만 캐싱함으로써 KV-cache 메모리와 attention 계산 비용을 크게 줄일 수 있습니다. 이를 통해 최소한의 혼란도 손실로 최대 8배 이상의 KV-cache 절감 효과를 보여줍니다. 또한, Attamba는 quadratic에서 linear로의 매끄러운 전환을 가능하게 하여, 환경에 따라 효율성을 조절할 수 있는 장점을 제공합니다.



### ShowUI: One Vision-Language-Action Model for GUI Visual Agen (https://arxiv.org/abs/2411.17465)
Comments:
          Technical Report. Github: this https URL

- **What's New**: 본 연구에서는 GUI(그래픽 사용자 인터페이스) 작업을 수행하기 위한 혁신적인 비전-언어-액션 모델인 ShowUI를 개발하였습니다. ShowUI는 스크린샷을 UI 연결 그래프로 구성하여 시각적 토큰 선택을 UI 지침에 따라 조정하는 새로운 방법론을 채택했습니다. 또한, 다양한 GUI 작업을 통합하는 Interleaved Vision-Language-Action Streaming 구조를 통해 훈련 효율성을 극대화했습니다.

- **Technical Details**: UI 가이드를 제공하는 시각적 토큰 선택 방법은 스크린샷의 패치를 그래프의 노드로 표현하고, 연결된 구성 요소를 통해 시각적 중복성을 모델링합니다. 이 방법은 self-attention 블록에서의 토큰 선택을 최적화하여 계산 비용을 줄이고 있습니다. 또한, Action 해석을 돕기 위해 GUI 작업 공간을 JSON 형식으로 구조화하여, 다양한 시각-언어-액션 데이터를 효율적으로 관리합니다.

- **Performance Highlights**: ShowUI는 256K의 데이터로 경량화된 2B 모델을 구성하여 제로샷 스크린샷 그라운딩에서 75.1%의 높은 정확도를 달성했습니다. UI 지침에 따른 토큰 선택 방식을 통해 훈련 중 시각적 중복 토큰을 33% 감소시키고 성능을 1.4배 가속화하는 성과를 보였습니다. 이 모델은 웹, 모바일, 온라인 환경에서의 내비게이션 실험에서도 뛰어난 효과를 입증하였습니다.



### FLEX-CLIP: Feature-Level GEneration Network Enhanced CLIP for X-shot Cross-modal Retrieva (https://arxiv.org/abs/2411.17454)
- **What's New**: 본 논문에서는 FLEX-CLIP이라는 새로운 Feature-Level Generation Network을 제안하고, 이는 CLIP(feature extractor) 기능을 향상시켜 X-shot Cross-modal Retrieval(CMR) 문제에 효과적으로 대응합니다. FLEX-CLIP은 두 가지 주요 단계를 포함하는데, 하나는 다중 모달(feature) 생성, 또 하나는 공통 공간(projection)이다. 이 접근 방식은 데이터 불균형 문제와 CLIP 기능의 저하 문제를 해결합니다.

- **Technical Details**: FLEX-CLIP의 첫 번째 단계인 다중 모달(feature) 생성에서는 VAE와 GAN 네트워크의 강점을 활용하여 데이터의 불균형 문제를 해결하고 CLIP 기능을 향상시키기 위해 복합적인 크로스 모달 생성 아키텍처를 설계했습니다. 두 번째 단계에서는 원본 샘플과 생성된 샘플을 공통 특징 공간으로 투영하여, 게이트 잔차 네트워크를 통해 CLIP 기능과 투영된 기능을 선택적으로 융합하여 특징 저하 문제를 크게 줄였습니다.

- **Performance Highlights**: FLEX-CLIP은 4개의 벤치마크 데이터셋에서 실험을 진행했으며, 기존의 최첨단 방법보다 7%에서 15% 향상된 성능을 보였습니다. 특히, X-shot 시나리오에서 (0, 1, 3, 5, 7-shot) FLEX-CLIP이 강력한 기준선보다 최대 7.9%의 성능 향상을 달성함을 입증했습니다.



### VLRewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models (https://arxiv.org/abs/2411.17451)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 멀티모달 AI 시스템을 평가하고 정렬하는 데 중요한 역할을 하는 Vision-language generative reward models (VL-GenRMs)의 평가 방법이 잘 탐구되지 않았음을 지적합니다. 기존의 평가 방법들은 전통적인 VL 작업에서 AI 주석을 기반으로 하는 선호 레이블에 의존하고 있으며, 이는 편향을 초래하고 최신 모델을 효과적으로 도전하지 못하는 경우가 많습니다. 이를 해결하기 위해 저자들은 VL-RewardBench라는 포괄적인 벤치마크를 소개합니다.

- **Technical Details**: VL-RewardBench는 일반 멀티모달 쿼리, 비주얼 환각 탐지, 복잡한 추론 작업을 포함하며, AI 지원 주석 파이프라인을 통해 인력의 검증을 결합한 샘플 선택 방식으로 1,250개의 고품질 예제를 큐레이션합니다. 16개의 선도하는 대형 비전-언어 모델을 대상으로 한 종합 평가 결과, VL-RewardBench는 도전적인 테스트베드로서 효과적이며, GPT-4o의 정확도는 65.4%에 불과합니다. 또한 Qwen2-VL-72B와 같은 최신 오픈소스 모델은 무작위 추측을 초과하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: VL-RewardBench에서의 성능은 MMMU-Pro 정확도와 강한 상관관계를 보이며 (Pearson's r > 0.9), VL-GenRMs로 Best-of-N 샘플링을 적용했을 때 나타납니다. 분석 실험을 통해 VL-GenRMs 개선을 위한 세 가지 주요 통찰이 발견되었습니다. 첫째, 모델은 추론 작업보다 기본적인 비주얼 지각 작업에서 주로 실패하며, 둘째, 추론 시간 확장 이점은 모델 용량에 따라 크게 달라지고, 셋째, VL-GenRMs의 판단을 학습하면 판단 능력이 크게 향상됩니다 (+14.7% 정확도). 이러한 연구 결과는 VL-GenRMs의 발전에 귀중한 자원이 될 것입니다.



### 2D Matryoshka Training for Information Retrieva (https://arxiv.org/abs/2411.17299)
- **What's New**: 2D Matryoshka Training은 여러 레이어-차원 설정에서 인코더 모델을 동시에 훈련시키기 위해 설계된 혁신적인 임베딩 표현 훈련 방법입니다. 이 방법은 Semantic Text Similarity (STS) 작업에서 기존 훈련 방식보다 더 높은 효율성을 보여주고 있습니다. 그러나 두 가지 구현 간의 차이로 인해 비교 결과가 다르게 나타나는 문제도 존재합니다. 본 연구에서는 두 가지 버전의 2D Matryoshka Training을 구현하고 평가하여 이 차이를 분석했습니다.

- **Technical Details**: 2D Matryoshka 훈련은 'Matryoshka Representation Learning'에서 영감을 받아 서로 다른 차원 크기를 가지는 여러 임베딩을 동시에 학습합니다. 이 방법은 전체 모델을 통과하지 않고 직접 서브 레이어에서 임베딩을 추출함으로써 텍스트 인코딩에 필요한 시간을 대폭 줄이고 있습니다. 이를 통해 훈련된 임베딩은 전통적인 전체 크기 모델보다 STS 작업에서 더 높은 효과를 나타냅니다. 본 연구는 두 가지 버전의 2D Matryoshka 훈련 설계를 통해 이 기술의 효과를 더욱 심도 있게 살펴봅니다.

- **Performance Highlights**: 2D Matryoshka 훈련의 두 가지 버전은 모두 전통적인 Matryoshka 훈련 및 전체 모델 훈련 방식보다 높은 효과를 보였습니다. 그러나 특정 서브 레이어 및 서브 차원 설정으로 별도로 훈련된 모델에는 미치지 못하는 것을 발견했습니다. 이러한 연구 결과는 감독 학습 및 제로샷 환경에서도 정보 검색 작업에 잘 일반화되었습니다. 추가적으로, 손실 계산 방식을 수정하여 검색 효과성을 향상시킬 수 있는 방법도 모색하였으며, 이를 통해 최적의 검색 효과성을 달성하기 위한 새로운 전략을 제안하였습니다.



### Interleaved Scene Graph for Interleaved Text-and-Image Generation Assessmen (https://arxiv.org/abs/2411.17188)
- **What's New**: 이번 연구에서는 텍스트와 이미지를 교차 생성하는 시스템의 평가를 위한 새로운 프레임워크인 ISG를 소개합니다. 이 프레임워크는 요리책처럼 사용자가 요구하는 정보를 텍스트와 이미지로 함께 제공할 수 있는 모델의 일관성을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: ISG는 장면 그래프(scene graph) 구조를 활용하여 텍스트와 이미지 블록 간의 관계를 포착합니다. 이는 전체적인 일관성(holistic consistency), 구조적 일관성(structural consistency), 블록 수준(block-level), 이미지 특정(image-specific) 네 가지 레벨에서 평가가 가능합니다.

- **Performance Highlights**: ISG-Bench라는 벤치마크 데이터셋을 통해 최근의 통합 비전-언어 모델이 교차 콘텐츠 생성에서 낮은 성능을 보임을 입증했습니다. 형태적 접근(compositional approaches)을 사용할 경우 통합 모델보다 111% 성능 개선을 달성했으며, ISG-Agent를 통해 '계획-실행-정제(plan-execute-refine)' 파이프라인을 적용해 122% 성능 개선을 달성했습니다.



### Relations, Negations, and Numbers: Looking for Logic in Generative Text-to-Image Models (https://arxiv.org/abs/2411.17066)
- **What's New**: 본 연구는 최신 이미지 생성 AI인 DALL·E 3을 활용하여 논리 연산자의 신뢰성 있는 배치에 대해 인간 피험자들의 평가 결과를 분석합니다. 특히, 관계(relations), 부정(negations), 및 정수(numbers)라는 세 가지 형태의 논리 연산자를 탐구하였으며, 생성된 이미지가 50% 이상의 인간 일치 점수를 얻지 못한다는 점이 주목할 만합니다. 이와 더불어 ‘grounded diffusion’ 기법을 사용한 실험에서도 DALL·E 3보다 나쁜 성능을 보였음을 발견하였습니다.

- **Technical Details**: 연구는 DALL·E 3를 대상으로 하며, 청사진에 따른 물리적 관계와 부정, 정수 등을 포함한 간단한 프롬프트를 다룹니다. 연구진은 총 35개의 다른 프롬프트를 생성하여 178명의 인간 평가자에게 평가받았으며, 각각의 프롬프트가 생성된 이미지와 얼마나 잘 일치하는지를 수치적으로 분석하였습니다. 이를 통해 논리 연산자가 상황을 이해하는 데 중요한 역할을 한다는 점에 주목하고, 이를 다룬 기존 문헌과 연결지어 설명하고 있습니다.

- **Performance Highlights**: 연구 결과, DALL·E 3이 생성한 이미지들은 인간이 기대하는 논리 연산자의 사용에 비해 현저히 부족한 성능을 보였습니다. 특히, 부정 프롬프트에 대한 성공률이 매우 낮았으며, 정수 프롬프트에서도 고질적인 문제를 드러냈습니다. 이러한 결과는 AI가 여전히 기본적인 논리 개념을 처리하는 데 어려움을 겪고 있음을 시사하며, 향후 더 나은 성능을 위한 알고리즘 개선의 필요성을 강조합니다.



### Boundless Socratic Learning with Language Games (https://arxiv.org/abs/2411.16905)
- **What's New**: 본 논문에서는 폐쇄 시스템 내에서 에이전트가 원하는 능력을 마스터하기 위한 세 가지 조건을 제시하고 이를 정당화합니다. 첫 번째 조건은 충분히 정보적이고 정렬된 피드백을 받는 것이며, 두 번째는 경험/데이터의 폭이 넓어야 한다는 점입니다. 마지막으로, 충분한 용량과 자원이 필요합니다. 이 조건들이 충족될 경우, 에이전트는 스스로 학습하여 성능을 개선할 수 있는 가능성이 높아집니다.

- **Technical Details**: 자기 개선(self-improvement)은 에이전트의 출력이 미래 학습에 영향을 미치는 과정을 의미합니다. 이러한 과정은 강화 학습(reinforcement learning)과 관련이 깊으며, 시스템 내에서 에이전트가 수행하는 행동에 의해 학습 데이터 분포가 변화하는 구조를 형성합니다. 특별히 제안된 '소크라틱 학습(Socratic learning)'은 에이전트가 입력과 출력을 서로 맞추어 지속적으로 성능을 향상시키는 방식입니다.

- **Performance Highlights**: 이 연구는 순수한 자기 개선을 통해 초기 데이터나 지식 이상으로 성능을 극대화할 수 있음을 보여줍니다. 이를 위한 구체적인 구현 프레임워크는 언어 게임(language games)이라는 개념을 기반으로 할 수 있습니다. 상호 작용하는 도구를 통해 에이전트는 피드백을 받고 이를 학습하여 시간에 따라 점진적으로 성능을 개선할 수 있습니다.



### Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering (https://arxiv.org/abs/2411.16863)
- **What's New**: 이번 연구에서는 Multimodal LLMs(MLLMs)의 적응성을 높이기 위해 외부 지식 소스를 통합하는 새로운 방법을 제안합니다. 제안된 모델인 Reflective LLaVA(ReflectiVA)는 외부 지식의 필요성을 동적으로 판단하고 외부 데이터베이스에서 검색한 정보의 관련성을 예측하기 위해 reflective tokens를 활용합니다. 이는 MLLM이 외부 지식을 관리하는 동시에 외부 지식이 필요 없는 작업에서 유창성과 성능을 유지할 수 있도록 합니다.

- **Technical Details**: ReflectiVA 모델은 두 단계의 두 개 모델 훈련 방법을 통해 reflective tokens를 학습합니다. 모델의 어휘를 확장하여 검색의 필요성을 판단하고 검색한 샘플이 입력 쿼리에 대한 적절한지 여부를 결정합니다. 본 연구에서는 Encyclopedic-VQA 및 InfoSeek 데이터셋을 사용하여 제안된 접근 방식의 성능을 검증하며, 이는 외부 지식을 필요로 하는 질문-답변 쌍을 포함하고 있습니다.

- **Performance Highlights**: 제안된 모델은 기존의 방법들과 비교하여 지식 기반 시각 질문 응답에서 우수한 성능을 보였습니다. 광범위한 실험을 통해, ReflectiVA는 모든 고려된 데이터셋과 설정에서 정답 정확도를 증가시키며, 표준 MLLM 벤치마크와 외부 지식이 필요하지 않은 전통적인 VQA 데이터셋에서도 높은 성능을 유지함을 보여주었습니다.



### Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation (https://arxiv.org/abs/2411.16789)
- **What's New**: 이번 논문에서는 Multimodal Sign Language Translation (MMSLT)라는 새로운 접근법을 제안합니다. MMSLT는 사전 훈련된 다중 모달 대형 언어 모델(MLLM)을 활용하여 수화를 점괄할 수 있는 텍스트 설명을 생성하고, 이를 문장 공간에 정렬하여 통역 정확성을 개선합니다. 이 접근법은 고전적인 글로스 기반 모델과 달리, 수화의 비주얼 요소를 직접적으로 분석하여 글로스 어노테이션의 필요성을 제거합니다.

- **Technical Details**: MMSLT는 두 가지 주 모듈로 구성되어 있습니다: Generating Sign Language Description via MLLM (GSD-MLLM)과 Multimodal-Language Pre-training (MMLP)입니다. GSD-MLLM 모듈은 MLLM을 통해 SL 설명을 생성하고, MMLP 모듈은 이 설명의 특징을 수화 이미지와 통합하여 텍스트로 변환합니다. 이를 통해 SL 구성 요소에 대한 세밀한 이해를 가능하게 하고, 변환 과정에서 발생할 수 있는 모달리티 간 간극 (modality gap)을 줄입니다.

- **Performance Highlights**: MMSLT는 두 개의 벤치마크 데이터셋인 PHOENIX14T와 CSL-Daily에서 탁월한 성능을 보여주었습니다. 특히 CSL-Daily 데이터셋에서는 prior 모델들에 비해 현저하게 향상된 BLEU-4 및 ROUGE 점수를 기록하여 SLT (Sign Language Translation) 분야에서 SOTA(최첨단) 성능을 달성했습니다. 이는 MMSLT가 복잡한 구문과 긴 맥락에서 효과적인 번역을 가능하게 함을 시사합니다.



### In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models (https://arxiv.org/abs/2411.16769)
- **What's New**: 이번 연구에서 우리는 텍스트-이미지(T2I) 모델의 안전 메커니즘을 평가하기 위한 혁신적인 레드 팀잉 프레임워크인 ICER를 도입합니다. ICER는 과거에 성공적인 레드 팀잉 시도로부터 배우고 이를 기반으로 해석 가능한 문제성 프롬프트를 생성하는 혁신적인 접근 방식으로, 기존 프롬프트 공격 방법들에 비해 뛰어난 성능을 보여줍니다. 이 프레임워크는 LLM(대형 언어 모델)을 활용하여 추가 훈련 없이 다양한 T2I 모델의 안전 메커니즘을 효율적으로 조사할 수 있도록 설계되었습니다.

- **Technical Details**: ICER 프레임워크는 과거의 경험을 기록하고 이를 참고하여 새로운 레드 팀잉 전략을 탐색하는 방식으로 구성됩니다. 핵심적으로, 우리는 LLM을 활용하여 해석 가능한 적대적 프롬프트를 생성하며, 이 과정에서 밴딧 최적화 알고리즘을 사용하여 중요성을 고려함으로써 다양한 안전 메커니즘을 효율적으로 테스트합니다. 연구에서는 성공적이고 실패한 공격의 경험을 바탕으로 프롬프트 효과성을 향상시키기 위해 Bayesian Optimization(베이지안 최적화) 프레임워크 내에서 세 가지 주요 요소를 통합합니다.

- **Performance Highlights**: 실험 결과, ICER는 기존의 프롬프트 공격 방법들에 비해 T2I 모델의 취약성을 식별하는 데 있어 우수한 성능을 보였습니다. 발견된 문제성 프롬프트는 원래 입력과 높은 의미적 유사성을 유지하며 의도된 콘텐츠를 효과적으로 탈옥하는 결과를 가져왔습니다. 이러한 접근 방법은 T2I 모델의 안전성을 평가하는 데 있어 현실적이고 도전적인 평가를 가능하게 하여, 향후 잠재적 악용에 대한 보다 강력한 안전 장치를 개발하는 데 기여할 것으로 기대됩니다.



### PriorDiffusion: Leverage Language Prior in Diffusion Models for Monocular Depth Estimation (https://arxiv.org/abs/2411.16750)
- **What's New**: 이번 연구는 텍스트-이미지 확산 모델을 활용하여 단안(depth estimation)에서 발생하는 모호성(ambiguity)과 시각적 혼란(visual nuisance)을 해결할 수 있는 가능성을 탐구하고 있습니다. 특히, 기존의 단안 깊이 추정은 입체(depth cues)나 다중 시점(multi-view) 깊이 안내의 부재로 인해 본질적인 모호성에 시달리고 있습니다. 연구진은 확산 모델에서 학습된 언어 선험(prior)을 통해 기하학적 정보를 확보함으로써 이러한 모호성을 해소할 수 있다고 주장합니다.

- **Technical Details**: 연구에서는 PriorDiffusion이라는 새로운 접근법을 제안하여, 텍스트와 이미지를 조합하여 심층 추론을 수행하는 방식을 채택하였습니다. 구체적으로, 선행 학습된 텍스트-이미지 확산 모델을 사용하여 장면에 맞춘 이미지와 텍스트 설명을 입력으로 받아들이고, 이를 통해 깊이 추정을 위한 노이즈 제거(denoising)를 진행합니다. 이러한 방법은 3D 장면을 사용자 의도에 맞춰 인식할 수 있도록 모델의 주의를 특정 영역에 집중시키는 방식으로 작동합니다.

- **Performance Highlights**: HyperSim 및 Virtual KITTI 데이터셋으로 훈련한 결과, NYUv2, KITTI, ETH3D 및 ScanNet 데이터셋에서 다른 확산 기반 깊이 추정기들과 비교하여 최신 기술 수준의 성과를 달성하였습니다. 연구진은 실제 데이터셋에서도 우수한 정성적 및 정량적 결과를 도출하며, 빠른 수렴 속도를 보여줍니다. 이는 언어 선험을 활용한 접근 방식을 통해 단안 깊이 추정 모델의 성능이 크게 향상될 수 있음을 보여줍니다.



### "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks (https://arxiv.org/abs/2411.16730)
Comments:
          This paper has been submitted to ICLR 2025 BlogPosts and OpenReview preprints. It has 9 pages of text, 4 figures, and 3 tables

- **What's New**: 이 연구는 대규모 언어 모델의 유해 콘텐츠 생성을 식별하고 억제하는 데 있어 새로운 도전을 제시합니다. 특히, 다단계 탈옥 프롬프트에 대해 가드레일의 효과성을 평가한 것이 특징입니다. 연구자는 GPT-4o, Grok-2 Beta, Llama 3.1, Gemini 1.5, Claude 3.5 Sonnet 모델을 실험 대상으로 선택하여 검증을 수행했습니다.

- **Technical Details**: 연구에서 사용된 시나리오는 "기업 중간 관리자들이 승진하기 위해 경쟁한다"는 설정으로, 연구자는 다단계 프롬프트를 통해 모델의 반응을 관찰했습니다. 실험 결과, 모든 모델의 가드레일이 우회되었으며, 이로 인해 유해한 언어 공격 콘텐츠가 생성되었습니다. 특히 Claude 3.5 Sonnet이 탈옥 프롬프트를 식별하는 데 가장 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 가드레일이 단순한 콘텐츠 필터 역할을 넘어서 예방 기능을 수행해야 한다고 강조합니다. 연구자는 GitHub에 실험 과정과 코드를 공개하여 개발자 및 연구자들 간의 협력을 촉진하고자 합니다. 이러한 노력은 언어 모델의 안전성과 윤리를 향상시키는데 기여할 것으로 기대됩니다.



### A Brief Summary of Explanatory Virtues (https://arxiv.org/abs/2411.16709)
Comments:
          10 pages, 2 tables

- **What's New**: 기술적 설명의 미덕(Explanatory Virtues, EVs)에 대한 철학, 심리학 및 인지 과학 문헌을 종합하여 설명합니다. 이 보고서에서는 Keas(2018)의 분류 체계를 따르며, 증거적(evidential), 일관성(coherential), 미적(aesthetic), 그리고 시간적(diachronic) 등의 네 가지 주요 유형을 도출합니다. 추가적으로, 이론의 넓은 적용 가능성을 고려하는 범주(coverage)를 포함하였습니다.

- **Technical Details**: EV는 귀납적 추론(abductive reasoning)에서 중요하며, 설명은 관찰을 정당화하는 이론으로 설명됩니다. 이론에서의 EV 차원은 인식론적(epistemic) 및 실용적(pragmatic) 차원으로 나뉘며, 각각의 차원은 내재적 일관성(internal coherence) 및 경험적 강도(empirical strength) 등 여러 요소로 구성됩니다. 이러한 개념은 기계 학습(Machine Learning, ML) 모델의 전역(global) 및 국소(local) 설명과도 연결됩니다.

- **Performance Highlights**: EVs는 이론이 관찰을 얼마나 잘 설명하는지를 측정하는 여러 측면을 나타냅니다. 각 EV는 데이터를 설명하는데 기여하며, 그들 간의 상관관계를 통해 이론의 유용성을 판단할 수 있습니다. 또한, 이론의 일관성, 유사성, 그리고 기존 신념들과의 호환성 등 다양한 요소들이 EV에 의해 평가됩니다.



### Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts? (https://arxiv.org/abs/2411.16679)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 잠재적 다단계 추론(latent multi-hop reasoning) 능력에 대한 평가를 진행하고, 새로운 평가 데이터셋 SOCRATES를 소개합니다. 기존 연구에서 LLM들이 추론을 위해 단축키(shortcuts)를 사용할 가능성에 대한 문제를 다루지 않았던 점을 보완하고, 이러한 단축키를 피하기 위한 데이터셋 구성 및 평가 절차를 제안합니다. 또한, 특정 쿼리에 대해 LLM들의 잠재적 합성(latent composability) 능력이 상당히 다름을 발견했습니다.

- **Technical Details**: 논문에서는 LLM들이 응답을 생성하는 과정에서 주체-객체(entity) 단축키를 피하기 위해, 사전 훈련(pretraining) 시 어려운 쿼리들을 제외하고 새로운 SOCRATES 데이터셋을 구성하였습니다. 이 데이터셋은 7,232개의 쿼리 쌍을 포함하고 있으며, 다양한 관계 조합을 평가할 수 있는 구조로 설계되었습니다. 두 가지 주요 기준을 통해 정확한 테스트 결과를 얻기 위해, 대표적인 다단계 쿼리에서 잠재적 결합 능력을 측정합니다.

- **Performance Highlights**: 연구 결과, 최첨단 LLM들이 잠재적 합성에서 80% 이상의 성능을 보였으나, 이는 다리 개체(bridge entity)의 유형에 따라 크게 달라졌습니다. 특히, '나라'를 다리 개체로 사용할 경우 80% 이상에 달하지만, '연도'를 사용할 경우 그 비율은 5%로 크게 떨어졌습니다. 이 결과는 다단계 추론 능력 평가에 있어 다양한 관계 조합 유형을 고려해야 할 필요성을 강조합니다.



### Self-Generated Critiques Boost Reward Modeling for Language Models (https://arxiv.org/abs/2411.16646)
Comments:
          20 pages

- **What's New**: 이번 논문에서는 Critic-RM이라는 새로운 프레임워크를 소개합니다. Critic-RM은 기존의 보상 모델이 단일 스칼라 점수를 생성하는 한계를 극복하고, 자가 생성된 비평을 활용하여 보상 모델을 향상시키는 방법을 제시합니다. 이 프레임워크는 추가적인 감독 없이 자가 비평 생성력(critique generation ability)을 활용하여 보상 모델의 정확성을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: Critic-RM은 비평을 생성하고 이를 필터링하는 두 단계를 통해 작동합니다. 이 과정에서 품질이 높은 비평만을 선택하여 보상 예측과 비평 생성의 공동 세부 조정을 실시합니다. 또한, LLM이 생성한 여러 후보 비평을 평가하여 인간 주석과의 일치성을 기반으로 품질을 개선하는 일관성 유도 필터링(consistency-guided filtering) 기법을 적용합니다.

- **Performance Highlights**: 실험 결과 Critic-RM은 기존 보상 모델 및 LLM 평가자에 비해 보상 모델링 정확도가 3.7%-7.3% 향상되었음을 보여줍니다. 추가 연구를 통해生成된 비평이 LLM의 결함 있는 추론 단계를 정정하는 데에도 효과적이었으며, 추론 정확도를 2.5%-3.2% 향상시키는 데 기여했습니다.



### Do Automatic Factuality Metrics Measure Factuality? A Critical Evaluation (https://arxiv.org/abs/2411.16638)
- **What's New**: 현대의 LLMs(대형 언어 모델)는 높은 품질의 요약을 생성할 수 있지만, 여전히 원하는 정보를 요약에 추가하는 오류를 범할 수 있습니다. 이러한 오류를 자동으로 측정하는 것이 어려웠으며, 이에 따라 생성된 요약의 사실적 일관성을 평가하기 위한 다양한 메트릭이 개발되었습니다. 본 연구는 이러한 자동화된 사실 정확성 메트릭을 심층 분석하고, 그 신뢰성에 관한 의문을 제기합니다.

- **Technical Details**: 이 연구에서는 사실 적합성(factuality)을 평가하기 위해 다양한 메트릭을 스트레스 테스트했습니다. 기본적인 속성만으로도 사실 적합성을 예측할 수 있음을 발견하였으며, 간단한 MLP(다층 퍼셉트론) 모델이 복잡한 SOTA(최신 기술 동향) 메트릭과 경쟁할 수 있음을 제시했습니다. 또한, 일부 메트릭들은 사실적인 수정에 반응하지만, 무관한 수정에 더 민감하게 반응하는 경향이 있습니다.

- **Performance Highlights**: 자동화된 사실 적합성 메트릭들은 무의미한 수정에 민감해 표면적인 신호에 의존하는 경향이 있습니다. 이는 '게임 내기'(game-making)를 통해 메트릭 값을 인위적으로 증가시킬 수 있음을 시사합니다. 결과적으로, 기존 메트릭에 대한 신뢰성을 확립하는 데에 심각한 문제가 있음을 나타내며, 향후 이 메트릭들에 대한 신뢰성 향상이 필요합니다.



### StructFormer: Document Structure-based Masked Attention and its Impact on Language Model Pre-Training (https://arxiv.org/abs/2411.16618)
- **What's New**: 이번 연구에서는 Language Models (LMs)의 pre-training 과정에서 global attention의 영향을 실증적으로 평가하는 데 중점을 두고 있습니다. 특히, arXiv 데이터로부터 구조 인식 텍스트를 생성하고, 이를 활용하여 pre-training을 진행한 후 attention 패턴의 변화를 분석합니다. 문서 구조를 LM 모델에 통합함으로써 문서 이해와 같은 더 추상적인 작업에서의 성능 향상이 가능함을 보여줍니다.

- **Technical Details**: Language Model의 목표는 특정 단어 시퀀스의 분포를 학습하여 다음 단어의 발생 확률을 계산하는 것입니다. 기존에는 LSTM과 BiLSTM 같은 메모리 인식 딥러닝 방법이 선호되었으나, 현재는 Transformer 기반의 모델이 주류를 이루고 있습니다. 이 연구에서는 global token을 전체 문서의 pre-training 과정에 통합하는 방법을 제시하며, LaTeX 파일로부터 구조 인식 문서를 생성하여 글로벌 토큰을 식별하고 이를 포함시킵니다.

- **Performance Highlights**: 이 연구는 BERT와 같은 Language Model이 pre-training 과정에서 문서 구조를 통합함으로써 downstream tasks에서 현저한 성과를 보여줄 수 있음을 입증합니다. 실험 결과, 구조 정보를 도입함으로써 핵심 키워드와 섹션 헤더 사이의 관계를 더 강하게 잡을 수 있었으며, 이는 자연어 이해를 넘어선 학습이 가능하다는 것을 시사합니다. 이러한 접근 방법은 LM이 복잡한 문서 이해 및 요약 작업을 효과적으로 수행할 수 있는 가능성을 열어줍니다.



### Recent Trends in Linear Text Segmentation: a Survey (https://arxiv.org/abs/2411.16613)
- **What's New**: 이 논문은 Linear Text Segmentation(선형 텍스트 분할) 분야의 현재 진행 상황을 종합적으로 조사합니다. 기존 연구에서는 Transformer 기반의 언어 모델 사용에 대한 심층 탐구가 부족했는데, 본 연구에서는 이러한 격차를 메우고 있습니다. 특히, 최근의 LLMs(대형 언어 모델) 사용을 강조하며, NLP(Natural Language Processing)의 다른 영역과의 연계를 제안합니다.

- **Technical Details**: 논문은 선형 텍스트 분할 시스템을 설계할 때 텍스트의 기본 단위를 선택하는 것에서 시작합니다. 연구 결과에 따르면 단어, 문장, 단락 단위 모두 사용할 수 있으며, 다양한 알고리즘이 이러한 단위에 따라 기능합니다. 예를 들어, TextTiling과 C99 알고리즘이 존재하며, 최근에는 BERT와 같은 Transformer 기반 모델을 활용한 접근법이 두드러집니다.

- **Performance Highlights**: 이 연구는 LLM을 활용하여 선형 텍스트 분할 문제를 자연어 생성(NLG) 작업으로 다루며, 검증된 여러 전통적 방법보다 우수한 성과를 보였습니다. 특히, 문맥상에서 연속 단어의 일관성을 평가하는 데 있어, 단어 임베딩을 활용한 GraphSeg와 같은 알고리즘이 주목받고 있습니다. 최근 LLM들은 신뢰성 있는 결과를 도출하고 있으며, 향후 연구 방향으로 유망한 성과를 기대하게 합니다.



### Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision (https://arxiv.org/abs/2411.16579)
Comments:
          Preprint

- **What's New**: 이 논문에서는 두 플레이어 파라다임을 제안하여, reasoning (추론) 모형과 critique (비판) 모형의 역할을 분리합니다. AutoMathCritique라는 자동화된 프레임워크를 통해 76,321개의 응답과 단계별 피드백 데이터셋을 생성하였고, 이는 언어 모델을 fine-tuning (미세조정) 하는 데 사용됩니다. 이러한 비판 모형은 복잡한 질문에 대한 actor (행위자) 모델의 성능을 개선하는 데 효과적임을 입증하였습니다.

- **Technical Details**: AutoMathCritique 프레임워크는 오류가 있는 추론 경로를 구성하고, 비판을 생성하며, 데이터를 필터링하는 세 가지 주요 단계로 이루어져 있습니다. 이 프레임워크의 결과로 생성된 데이터셋인 MathCritique-76k는 76,321개의 샘플로 언어 모델을 fine-tune하는 데 활용되었습니다. 논문은 비판 모형이 actor 모델의 탐색 효율성 및 문제 해결 다양성을 어떻게 향상시키는지를 심층적으로 분석하였습니다.

- **Performance Highlights**: 비판 모형은 특히 어려운 질문에 대해 actor 모형의 성능을 일관되게 향상시켰으며, 탐색 과정에서의 효과적인 감독을 통해 더 나은 결과를 가져왔습니다. 결과적으로, critique-in-the-loop self-improvement method (비판 루프 자가 개선 방법)은 actor 모델이 스스로 개선할 수 있는 과정에서 비판 모델과의 협업을 통해 더 강력한 추론 모델을 만드는 데 기여하였습니다. 마지막으로, self-talk-via-critique 방법론을 통해 단계별 자기 반성 및 수정이 가능하다는 가능성을 보여주었습니다.



### Profiling Bias in LLMs: Stereotype Dimensions in Contextual Word Embeddings (https://arxiv.org/abs/2411.16527)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 편향을 강조하기 위해 고안된 편향 프로파일(bias profiles)을 제안합니다. 사회 심리학 연구의 사전(dictionaries)을 기반으로 한 여러 고정관념 차원을 바탕으로 LLM의 성별 편향을 분석했습니다. 연구는 총 12개의 LLM에서 고정관념 프로파일을 생성하고, 이를 통해 편향을 시각화하는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 Fiske et al.(2002)의 고정관념 내용 모델(stereotype content model, SCM)에 기반하여 LLM의 내재 벡터(embeddings)를 변환합니다. 연구의 원리는 온도(warmth)와 능력(competence)이라는 두 가지 주요 차원에 걸쳐 이루어지며, 성별에 따라 여성은 높은 온도와, 남성은 높은 능력과 연결되는 경향을 보여줍니다. 또한, 연구에서는 7차원(bias profiles)을 제공하여 LLM의 편향을 보다 세분화하여 분석합니다.

- **Performance Highlights**: 이 연구는 LLM이 생성한 예제들이 성별 특정 지명과 대명사를 포함하지 않도록 하여 추가적인 성별 편향이 도입되지 않도록 합니다. 다양한 실험 설정을 통해, 연구자들은 LLM의 내재 벡터가 인지하는 고정관념 차원의 성별 관련 특성을 명확히 분석했으며, 생성한 데이터 예제가 자연어 이해(natural language understanding, NLU) 작업에서 유사한 성능을 나타냈습니다. 마지막으로, 고정관념의 시각화와 편향 노출을 통해, 연구는 모델에 내재된 편향에 대한 새로운 통찰을 제공합니다.



### AtomR: Atomic Operator-Empowered Large Language Models for Heterogeneous Knowledge Reasoning (https://arxiv.org/abs/2411.16495)
- **What's New**: 이 논문에서는 AtomR이라는 새로운 이질적 지식 추론 프레임워크를 제안합니다. AtomR은 복잡한 질문을 세 가지 원자적 지식 연산자의 조합으로 분해하여 다양한 지식 소스에서 동적 지식 검색을 수행합니다. 이를 통해 기존의 렌더링된 질문 응답 시스템에서 나타나는 추론 계획의 비효율성 문제와 다양한 지식 소스로부터의 동적 검색 수용 부족 문제를 해결합니다.

- **Technical Details**: AtomR은 입력 질문을 세부적인 하위 질문의 트리로 분해하고, 각 리프 노드에서 원자적 지식 연산자를 통해 질문을 해결합니다. 이 과정에서 LLM을 사용하여 지식 소스 간 중복 검색을 피하고, 필요할 경우 동적으로 여러 지식 소스에서 자료를 검색하여 최적의 추론 경로를 도출합니다. AtomR은 구조적 트리와 포스트오더 순회를 설정하여 복잡한 질문을 효율적으로 관리합니다.

- **Performance Highlights**: 실험 결과, AtomR은 세 개의 단일 소스 및 두 개의 다중 소스 데이터셋에서 업계 최고 수준의 성과를 나타냈습니다. 특히, 2WikiMultihop에서 9.4%, BlendQA에서 9.5%의 성능 향상을 보였으며, 전반적으로 원자적 지식 연산자를 통해 제안된 접근 방식이 기존 모델들보다 탁월한 성능을 발휘함을 보여주었습니다.



### O1 Replication Journey -- Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson? (https://arxiv.org/abs/2411.16489)
Comments:
          16 pages

- **What's New**: 이 논문은 OpenAI의 O1 모델 복제와 관련된 현재 접근 방식에 대한 비판적 검토를 제공합니다. 특히, 지식 증류(knowledge distillation) 기술의 널리 퍼졌지만 종종 공개되지 않은 사용에 집중하고 있습니다. 이전 연구에서는 O1 복제의 기본 기술 경로를 탐구했으며, 이번 연구는 단순한 증류와 감독된 파인 튜닝(supervised fine-tuning)이 복잡한 수학적 추론 작업에서 우수한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 이)의 시도들을 평가하는 포괄적인 벤치마크 프레임워크를 소개합니다. 프레임워크는 기술적 투명성과 재현 가능성의 기준으로 O1 복제 시도를 평가하기 위한 명확한 메트릭을 제공합니다. 또한, 지식 증류 기술에 대한 논의와 이 접근 방식의 잠재적 위험도 다룹니다. 이러한 분석 결과는 단순히 성능 향상을 추구하는 것 이상의 의미를 가지고 있음을 강조합니다.

- **Performance Highlights**: 연구 결과, 적은 표본(수만 개)의 O1 증류 샘플을 사용하고 표준 감독 파인 튜닝을 통해, 기본 모델이 American Invitational Mathematics Examination (AIME)에서 O1-preview의 성능을 초과할 수 있음을 입증했습니다. 이는 기술적 복잡성이 적게 드는 방법으로, 수학 문제 해결 데이터만으로도 다양한 작업에서 강력한 일반화 능력을 보여주었습니다. 이러한 성과에도 불구하고, 증류 접근 방식의 과도한 의존성은 장기적으로 혁신의 정체를 초래할 수 있다는 점에서 우려를 표합니다.



### When Babies Teach Babies: Can student knowledge sharing outperform Teacher-Guided Distillation on small datasets? (https://arxiv.org/abs/2411.16487)
Comments:
          Accepted to BabyLM challenge, CoNLL Workshop, EMNLP 2024

- **What's New**: 이번 연구는 BabyLM 챌린지에 대한 우리의 제출물을 소개하며, 데이터 효율적인 언어 모델 프리트레이닝의 경계를 확장하는 것을 목표로 합니다. 본 방법은 심층 상호 학습(deep mutual learning)을 기반으로 하며, 다양한 초기화를 위한 학생 모델 탐색(student model search)을 도입합니다. 여기에 학생을 동등하게 대하는 한계를 극복하기 위해 가중치 상호 학습(weighted mutual learning)을 이차 최적화 문제(bi-level optimization problem)로 형성하였습니다.

- **Technical Details**: 본 연구 방법론은 RoBERTa-base 모델(125M parameters)을 절반 이하의 크기로 증류하면서 성능을 유지하는 것에 중점을 두고 있습니다. 베이지안 최적화(Bayesian optimization)를 통해 학생 모델의 아키텍처를 선택하였으며, 기존의 교사-학생 증류 대신 가중치 상호 학습을 탐구합니다. 이 과정은 두 개의 루프로 구성되어 있으며, 내부 루프는 온라인 증류(online distillation)를 통해 압축된 학생 모델을 학습하고, 외부 루프는 다양한 학생으로부터의 지식을 효과적으로 증류하기 위해 가중치를 최적화합니다.

- **Performance Highlights**: 우리의 평가 결과, 교사 모델 없이도 기존의 교사-지도 접근 방법과 유사하거나 우수한 성능을 나타냅니다. 이 연구는 전통적인 자율 학습(independent learning)보다 여러 대규모 네트워크 간의 성능을 향상시키는 것을 확인하였으며, 교사 모델의 필요성을 줄이면서도 상당한 성능 향상이 가능하다는 것을 보여줍니다. 가중치 상호 학습 전략이 네트워크의 학습 효율을 높임으로써, 데이터 효율성과 메모리 효율성이 중요한 경계 시스템에서 특히 큰 장점을 가질 수 있음을 시사합니다.



### Learning by Analogy: Enhancing Few-Shot Prompting for Math Word Problem Solving with Computational Graph-Based Retrieva (https://arxiv.org/abs/2411.16454)
- **What's New**: 새로운 논문에서는 기존의 방법들이 수학 문제(학문적 단어 문제, MWP)를 해결하기 위해 무작위 선택 또는 의미 기반 검색에 의존했던 반면, 계산 그래프를 기반으로 한 예시 검색 방식을 제안합니다. 이 방법은 주어진 문제와 구조적으로 유사한 예시를 검색하여 LLM(대형 언어 모델)의 문제 해결 성능을 향상시킵니다. 특히, 이 연구는 6개의 MWP 데이터셋에서 제안한 방법이 기존의 방법보다 미세한 성과 향상을 이루고 있음을 입증했습니다.

- **Technical Details**: 제안된 방법은 구조적 유사성을 활용하여 주어진 문제와 유사한 계산 그래프를 지닌 문제를 검색하는 과정을 포함합니다. 간단한 구조를 갖춘 검색 모델을 설계하여 유사한 예시를 우선적으로 선택할 수 있도록 하였습니다. 이러한 검색 모델은 대조 학습(contrastive learning) 방식으로 훈련되며 LLM의 파라미터 변동 없이 통합될 수 있는 모듈형 접근 방식을 제공합니다.

- **Performance Highlights**: 연구 결과, 제안한 계산 그래프 기반의 검색 방법은 의미 기반 검색 및 무작위 선택 방법보다 6.7% 및 19.5%의 평균 정확도(Exact Matching, EM) 개선을 달성했습니다. 또한 많은 케이스 스터디를 통해 이 방법이 MWP 해결에 필수적인 구조적 뉘앙스를 잘 포착하는 능력을 검증했습니다. 이로 인해 LLM의 응용 분야가 교육 및 복잡한 추론 과제로 확대될 가능성이 드러났습니다.



### Finding Structure in Language Models (https://arxiv.org/abs/2411.16433)
Comments:
          PhD Thesis at ILLC, University of Amsterdam

- **What's New**: 이번 연구에서는 언어 모델이 인간과 유사한 깊은 문법 구조 이해를 가지고 있는지를 탐구합니다. 최근 몇 년간 사회에 미친 언어 모델의 영향력을 감안할 때, 이러한 질문은 자연어 처리(natural language processing), 언어학(linguistics), 그리고 해석 가능성(interpretability) 사이의 교차점에 위치하고 있습니다. 우리는 대규모 언어 모델의 복잡한 본질을 이해하기 위해 새로운 해석 가능성 기법을 개발할 것입니다.

- **Technical Details**: 우리는 세 가지 방향에서 연구 질문에 접근합니다. 첫째, 심리언어학(psycholinguistics)의 핵심 패러다임인 구조적 프라이밍(structural priming)을 통해 추상적인 언어 정보를 조사합니다. 둘째, 형용사 순서(adjective order)와 부정 극성 항목(negative polarity items)과 같은 다양한 언어 현상을 분석하고 이러한 현자에 대한 모델의 이해도를 훈련된 데이터 분포와 연결합니다. 셋째, 점진적으로 복잡성이 증가하는 합성 언어(synthetic languages)를 사용해 언어 모델의 계층적 구조(hierarchical structure)를 연구하기 위한 제어된 테스트베드를 소개합니다.

- **Performance Highlights**: 우리의 연구 결과는 언어 모델의 표현에 내재된 문법 지식에 대한 자세한 설명을 제공합니다. 이 결과는 계산 방법(computational methods)을 사용하여 근본적인 언어학적 질문(fundamental linguistic questions)을 조사하기 위한 여러 방향성을 제시합니다. 연구는 언어 처리에 필요한 문법 구조의 복잡성을 파악하는 데 기여하며, 언어 모델의 이해도를 한층 더 끌어올리는 데 도움을 줍니다.



### Adapter-based Approaches to Knowledge-enhanced Language Models -- A Survey (https://arxiv.org/abs/2411.16403)
Comments:
          12 pages, 4 figures. Published at KEOD24 via SciTePress

- **What's New**: 최근 KELMs(지식 강화 언어 모델)가 대규모 언어 모델과 도메인 특화 지식 간의 격차를 메우는 유망한 도구로 주목받고 있습니다. 본 연구는 어댑터 기반 접근법에 대한 체계적인 문헌 리뷰(SLR)를 수행하며, 다양한 방법론의 강점과 잠재적 단점을 탐구합니다. 특히 생물 의학 영역에 집중하여 기존 KELMs의 성능 비교를 제공합니다.

- **Technical Details**: KELMs는 지식 그래프(KGs)를 활용하여 사실 정확도를 높이고 생성 텍스트 내 환각을 줄이는 데 기여합니다. 어댑터 모듈은 계산 부하를 줄이고 재학습 시의 삭제 위험을 최소화하는 데 효과적입니다. 본 연구에서는 KELMs에 대한 기존의 다양한 연구를 정량적, 정성적으로 분석하고, 어댑터 기반의 지식 강화를 위한 주요 동향을 정리합니다.

- **Performance Highlights**: 연구 결과, KELMs의 일반 지식과 도메인 특화 접근법이 자주 탐구되고 있으며, 다양한 어댑터 아키텍처와 하위 작업에서 긍정적인 성능을 보였습니다. 어댑터를 통한 경량화 및 효율적 해결책이 존재함에도 불구하고, KELMs에 대한 포괄적인 리뷰가 부족했음을 확인하였습니다. 본 연구는 어댑터 기반 KELM의 접근법에 대한 새로운 통찰을 제공합니다.



### Human-Calibrated Automated Testing and Validation of Generative Language Models (https://arxiv.org/abs/2411.16391)
- **What's New**: 이 논문에서는 고위험 분야, 특히 은행 산업에서 사용되는 Retrieval-Augmented Generation (RAG) 시스템을 위한 생성 언어 모델(GLM)의 평가 및 검증을 위한 포괄적인 프레임워크인 Human-Calibrated Automated Testing (HCAT)을 제안합니다. HCAT은 자동화된 테스트 생성을 통한 효율적인 검증, 임베딩 기반의 메트릭을 사용한 기능성 및 안전성 평가, 그리고 인간 판단과의 정렬을 위한 이중 단계 보정 접근 방식을 통합합니다. 이 프레임워크는 모델의 성능을 다양한 입력 조건에 대해 평가할 수 있는 강건성 테스트를 포함하여 GLM을 특정 분야에 맞게 안전하고 정확하게 배치할 수 있는 실질적이고 신뢰할 수 있는 솔루션을 제공합니다.

- **Technical Details**: HCAT 프레임워크는 자동 테스트 생성, 설명 가능한 평가 메트릭 및 인간 보정 기준을 결합하여 RAG 시스템에서 GLM을 평가하는 복잡성을 해소합니다. 자동화된 테스트 생성은 주제 모델링과 계층화 샘플링을 사용하여 모델 평가를 위한 다양한 쿼리를 생성합니다. 또한, HCAT은 임베딩 기반 메트릭을 통해 기능성 평가, 리스크 평가, 안전성 특성을 체계적으로 평가할 수 있는 방안을 제시합니다.

- **Performance Highlights**: HCAT의 성능 강조점은 제안된 접근 방식이 GLM의 작동 신뢰성을 보장하며, 사용자가 요구하는 정보에 대해 정확하고 관련성 있는 응답을 생성할 수 있도록 함을 보여줍니다. 또한, HCAT은 모델 성능에 대한 지속적인 모니터링과 개선점을 식별할 수 있도록 해주며, 특히 금융 산업에서 규제 준수를 위한 엄격한 기준을 충족하기 위한 체계적인 검증 절차를 제공합니다.



### FineWeb-zhtw: Scalable Curation of Traditional Chinese Text Data from the Web (https://arxiv.org/abs/2411.16387)
- **What's New**: 이번 논문에서는 전통 중국어(Traditional Chinese) 사용자를 위한 새로운 데이터셋 FineWeb-zhtw를 소개합니다. 기존의 영어 데이터셋이 많이 구축된 것과는 달리, 전통 중국어를 위한 유사한 노력은 상대적으로 부족했던 점을 지적하고 있습니다.

- **Technical Details**: FineWeb-zhtw는 언어적 차이를 반영하기 위해 여러 단계로 구성된 필터(Filtering)를 사용하여 품질(Quality)과 포괄성(Comprehensiveness)을 보장합니다. 이 과정에서는 데이터셋 샘플을 질의(Querying)하여 효과성을 평가하는 세 가지 주요 목표(Objectives)를 설정했습니다.

- **Performance Highlights**: 연구에서 사용된 코드와 데이터셋은 공개(Available)되어 있어 연구자들이 자유롭게 이용할 수 있습니다. 이는 전통 중국어 LLM(large language model) 개발에 기여할 것으로 기대됩니다.



### Multi-modal Retrieval Augmented Multi-modal Generation: A Benchmark, Evaluate Metrics and Strong Baselines (https://arxiv.org/abs/2411.16365)
- **What's New**: 본 논문은 Multi-modal Retrieval Augmented Multi-modal Generation (M2RAG)이라는 흥미로운 작업을 조사하고 있습니다. 이 작업은 기본 모델이 혼합된 텍스트와 이미지를 포함한 다중 모드 웹 페이지를 탐색하고 사용자의 질문에 대한 답변을 생성하는 것을 요구합니다. 이는 정보 밀도와 가독성을 향상시키며, 체계적인 연구 및 분석의 부족을 해결하기 위해 새로운 기준점을 제시하고 있습니다.

- **Technical Details**: M2RAG 작업을 완료하기 위한 여러 방법을 제안하며, 두 가지 접근법인 Single-stage Approach와 Multi-stage Approach를 소개합니다. Single-stage Approach는 응답을 한 번에 생성하며, Multi-stage Approach는 텍스트 응답을 먼저 생성한 후 관련 이미지를 추가하여 보다 응집력 있는 최종 출력을 제공합니다. 이 논문은 Large Language Models (LLMs)와 Multi-modal Large Language Models (MLLMs) 존재하는 모델의 능력을 평가하기 위해 설계된 종합적인 벤치마크를 구축하여, 텍스트 모달 및 멀티 모달 측면에서 생성된 콘텐츠를 평가합니다.

- **Performance Highlights**: 광범위한 실험 결과는 M2RAG 작업에서 LLM이 MLLM보다 상당히 우수한 성능을 보인다는 것을 보여주었습니다. 특히, Multi-stage Approach는 Single-stage Approach보다 관련 이미지를 더 잘 통합하여 응답 품질을 크게 향상시킵니다. 또한, 기존 MLLM은 M2RAG 작업에서 중요한 한계를 보였으며, 이미지와 텍스트 간의 관계를 통합하여 모델 성능을 개선해야 할 필요성을 강조합니다.



### The Two-Hop Curse: LLMs trained on A->B, B->C fail to learn A-->C (https://arxiv.org/abs/2411.16353)
- **What's New**: 이번 논문에서는 LLMs(대규모 언어 모델)가 두 단계 추론(two-hop reasoning)에서 겪는 문제를 새로운 설정에서 조사합니다. 연구자들은 Llama 3 8B Instruct 및 GPT-4o와 같은 모델을 허구의 사실로 미세 조정(fine-tuning)한 후, CoT(Chain-of-Thought) 방법을 이용해 질문에 답하는 일반화를 확인했습니다. 그러나 CoT 없이 서로 다른 문서에서 학습된 사실이 포함된 경우에는 두 단계 추론이 완전히 실패하는 "Two-Hop Curse"라는 개념을 제시했습니다.

- **Technical Details**: 이 연구는 두 단계 추론을 위한 통제된 설정(controlled setting)을 도입하여, 모델이 학습 시 자료를 함께 다루는 경우에는 잠재적 추론(latent reasoning)이 가능하다는 것을 발견했습니다. 하지만 두 가지 사실이 서로 다른 문서에서 학습된 경우, 모델은 기회를 넘어서는 성과를 보이지 못하고 무작위 수준의 정확도(chance-level accuracy)와 손실(test loss)를 기록했습니다. 이는 LLMs가 질문 범주와 무관하게 잠재적 다단계 추론에 대한 일반적인 능력을 결여하고 있음을 나타냅니다.

- **Performance Highlights**: 9개의 최전선 LLMs 모델을 실제 사실에 대해 평가한 결과, 대부분의 질문 범주에서 CoT를 사용하지 않는 경우 두 단계 추론에서 완전히 실패하는 경향을 발견했습니다. 반면, 대부분 범주에서 CoT를 사용할 경우에는 부분적인 성공을 유지했습니다. 이러한 결과는 LLMs가 질문 유형에 관계없이 다단계 추론에 필요한 일반적인 능력이 부족하다는 것을 시사합니다.



### Preference Optimization for Reasoning with Pseudo Feedback (https://arxiv.org/abs/2411.16345)
Comments:
          28 pages, 11 figures

- **What's New**: 본 논문에서는 Direct Preference Optimization (DPO)와 같은 선호 최적화 기법을 통해 대규모 언어 모델(LLMs)의 추론 능력을 향상시키는 새로운 접근 방법을 제안합니다. 이를 위해 사람의 검증된 레이블이 부족한 추론 과제를 해소하기 위한 방법으로, 테스트 케이스에 대한 평가를 통한 의사 피드백(pseudo feedback) 생성을 탐구합니다. 특히, 전선 LLMs로부터 생성된 테스트 케이스와 자기 일관성(self-consistency)을 확장하여 여러 테스트 케이스를 활용한 두 가지 형태의 의사 피드백을 소개합니다.

- **Technical Details**: 연구에서는 Mathstral-7B 모델을 기반으로 하여 수학적 추론 및 코딩 작업에서 의사 피드백을 사용하여 DPO의 성능을 최적화하는 실험을 진행했습니다. 실험 결과, MATH 결과는 58.3에서 68.6으로 개선되었으며, GSM8K와 College Math에서는 각각 85.6에서 90.3, 34.3에서 42.3으로 증가했습니다. 또한 Deepseek-coder-7B-v1.5를 활용하여 LiveCodeBench에서도 21.1에서 24.6으로 점수가 상승하였습니다.

- **Performance Highlights**: 실험 결과는 제안한 두 가지 형태의 의사 피드백이 함께 사용될 때 서로 보완적인 효과를 가져올 수 있음을 보여줍니다. 이러한 방법들은 사람에 의한 레이블링 없이도 확장 가능하게 적용될 수 있으며, 최종적으로 LLM의 추론 정확성과 성능 향상에 기여합니다. 특히, 이 연구는 추론 문제의 솔루션 레이블링을 테스트 케이스 평가 과정으로 정립하여 DPO 최적화를 용이하게 만드는 기초를 제공하였습니다.



### Can AI grade your essays? A comparative analysis of large language models and teacher ratings in multidimensional essay scoring (https://arxiv.org/abs/2411.16337)
Comments:
          Accepted at LAK '25

- **What's New**: 이 연구에서는 독일 학생들의 에세이를 평가하기 위해 오픈소스 및 클로즈드 소스 LLM(대형 언어 모델)의 성능과 신뢰성을 평가했습니다. 37명의 교사들이 정의한 10개의 평가 기준에 따라 실제 에세이를 기준으로 실험을 진행하였으며, LLM들이 언어 관련 기준에서 특히 우수한 성능을 보였습니다. 특히, o1 모델은 모든 LLM 중에서 가장 높은 성과를 기록하였으며, 이는 교사의 평가와의 상관관계에서 $r = .74$, 내부 신뢰성에서는 $ICC=.80$이라는 수치를 보여 줍니다.

- **Technical Details**: 연구에서는 7학년과 8학년 학생들의 20개 실제 에세이를 대상으로 GPT-3.5, GPT-4, o1, LLaMA 3-70B, Mixtral 8x7B의 5개의 LLM을 분석하였습니다. 각 모델의 평가 결과를 37명의 교사들의 평가와 비교하여, 이들의 강점과 한계를 여러 평가 범주에서 철저히 분석하였습니다. 특히, 평가 범주 간 상관 관계를 조사하여 LLM의 추론 과정을 이해하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 결과적으로, 클로즈드 소스 GPT 모델들은 오픈소스 모델들보다 높은 내부 일관성과 교사 평가와의 일치를 보였으며, 언어 관련 기준에서 특히 두드러진 성과를 나타냈습니다. LLM 기반의 평가가 교사의 작업 부담을 줄일 수 있는 유용한 도구가 될 수 있다는 점을 보여 주었으나, 모델들이 점수가 높게 나오는 경향이 있어 콘텐츠 품질을 더 잘 반영하기 위한 추가적인 개선이 필요함을 시사합니다.



### Learning from Relevant Subgoals in Successful Dialogs using Iterative Training for Task-oriented Dialog Systems (https://arxiv.org/abs/2411.16305)
- **What's New**: 본 연구에서는 Task-oriented Dialog (ToD) 시스템의 성능을 향상시키기 위한 새로운 접근 방식인 SUIT(SUbggoal-aware ITerative Training)를 제안한다. SUIT는 모델에서 샘플링된 대화를 사용하여 대화 성공에 기여하는 서브목표(subgoal)를 식별하고, 이를 기반으로 고품질의 훈련 샘플을 생성한다. 이 방법은 기존의 고정된 데이터 세트에 의존하기보다 반복적으로 더 많은 데이터를 생성할 수 있는 능력을 갖추고 있다.

- **Technical Details**: SUIT는 초기 LLM(대형 언어 모델)을 사용하여 사용자의 목표에 대한 대화를 생성하고, 성공적인 대화에 대한 서브목표를 식별하기 위해 'distant supervision'(원거리 감시) 방식을 적용한다. 이렇게 결정된 서브목표를 통해 추가적인 훈련 샘플을 구축하고, 이를 통해 ToD 시스템의 supervised fine-tuning(SFT) 또는 preference learning을 개선한다. 이 반복적인 과정은 오류를 최소화하고 대화 관련 모델 성능을 최대화하는 데 중점을 둔다.

- **Performance Highlights**: SUIT의 접근 방식은 대화 품질을 크게 향상시키며, 인기 있는 ToD 벤치마크에서 새로운 최첨단 성능(state-of-the-art performance)을 기록하였다. 기존 E2E(End-to-End) 시스템과는 달리 SUIT는 모델 맞춤화 없이 대규모 응용 프로그램에서도 쉽게 설정하고 사용할 수 있다. 특히, 이 방법은 DPO(Direct Preference Optimization)를 기반으로 한 선호 학습을 적용하여 효율성과 안정성을 높인다.



### BayLing 2: A Multilingual Large Language Model with Efficient Language Alignmen (https://arxiv.org/abs/2411.16300)
Comments:
          BayLing 2's online demo: this http URL. BayLing 2's code and models: this https URL

- **What's New**: BayLing 2는 고자원의 언어에서 저자원의 언어로의 생성 능력과 지식을 효율적으로 전이하는 혁신적인 다국어 대형 언어 모델(LLM)입니다. 320만 개의 지시사항 데이터셋을 사용하여 100개 이상의 언어에 대한 다국어 이해 능력을 향상시키고, 고자원 언어에서 저자원 언어로의 지식 전이 처치를 가능하게 합니다. BayLing은 다국어 번역과 지식 전이 벤치마크에서 기존 오픈 소스 모델보다 월등한 성능을 보였습니다.

- **Technical Details**: BayLing 2는 Llama 모델을 기반으로 하여 설계되었으며, 이 모델은 고자원 언어인 중국어와 영어로 이루어진 지시사항으로 훈련되었습니다. 이를 통해 100개 이상의 언어에 대한 교차 언어 지시사항을 포함하여 보다 효율적인 언어 정합을 수행합니다. BayLing은 두 가지 주요 모델인 BayLing-2-7B와 BayLing-2-13B를 포함하며, 다양한 언어 간의 능력 전이를 위한 체계적인 평가가 이루어졌습니다.

- **Performance Highlights**: BayLing의 성능 평가는 100개 이상의 언어에서 우수한 번역 결과를 나타냈으며, 특히 20개 이상의 저자원 언어에서 지식 전이의 효과가 두드러졌습니다. 예를 들어, 바밤바(Bambara), 루간다(Luganda), 스와힐리(Swahili), 줄루(Zulu) 등의 언어에서 뚜렷한 성능 향상이 이루어졌습니다. 또한, BayLing은 고자원 언어에서의 성능을 유지하면서 저자원 언어에서의 응답 능력을 효율적으로 개선하였습니다.



### NormXLogit: The Head-on-Top Never Lies (https://arxiv.org/abs/2411.16252)
- **What's New**: 이 논문에서는 최근의 언어 모델(Large Language Models, LLMs)의 해석 가능성을 향상시키기 위한 새로운 접근법, NormXLogit을 제안합니다. 기존의 기법들은 특정 모델 설계와 복잡한 계산 비용에 의존하지만, NormXLogit은 그러한 제약 없이 다양한 아키텍처에 적용할 수 있습니다. 이 방법은 입력 토큰의 중요도를 평가하기 위해 단어 임베딩의 노름을 활용하며, 모델의 최종 예측과의 관계를 통해 각 토큰의 중요성을 판단합니다.

- **Technical Details**: NormXLogit은 자연어 처리(Natural Language Processing, NLP) 작업에 쉽게 적용할 수 있는 아키텍처 독립적인 접근법으로, 모델의 최종 레이어 표현에서 인코딩된 풍부한 의미 및 구문 정보를 활용합니다. 이 방법은 중간 결과의 집합이 필요 없으며, 대규모 언어 모델의 프리트레인 과정에서 임베딩의 노름을 기반으로 입력 토큰의 중요도를 판단합니다. 더불어, NormXLogit은 태스크에 특화된 해석을 제공하기 위해 사전 훈련된 모델의 헤드-온-탑을 통합하고, 계산 비용을 대폭 줄입니다.

- **Performance Highlights**: 실험을 통해 NormXLogit은 기존의 경량 기법보다 더 높은 정확성을 보여주며, 특히 점수 부여(faithfulness) 측면에서 개선된 성과를 나타냅니다. 회귀 설정에서는 아키텍처 특정 기법보다 뛰어난 성능을 보였으며, 다양한 언어적 현상에 대한 층별 설명(layer-wise explanations) 평가에서도 효과적인 결과를 입증했습니다. 결과적으로 이 방법은 LLMs의 변화를 반영하여 다양한 응용에서 유용성을 제공할 수 있는 가능성이 큽니다.



### Transparent Neighborhood Approximation for Text Classifier Explanation (https://arxiv.org/abs/2411.16251)
Comments:
          IEEE DSAA'24

- **What's New**: 최근 문헌에서는 모델 비의존적 설명을 도출하는 데 있어 이웃 구성의 중요한 역할이 강조되고 있습니다. 본 논문에서는 텍스트 분류기를 설명하기 위해 생성 모델을 통해 합성 인스턴스의 품질을 개선하는 방법을 제시합니다. 이 방식은 텍스트의 비구조적 특성으로 인한 이웃 구성의 문제를 해결하며, 설명의 품질을 높입니다. 이를 통해 기존의 블랙박스 생성기 대안으로 확률 기반 편집 방법을 도입합니다.

- **Technical Details**: 논문에서는 설명 가능한 AI(XAI)의 개념과 함께 모델의 의사 결정 과정을 이해하기 위한 다양한 접근 방식을 논의합니다. XPROB라는 새로운 설명 방법은 주어진 텍스트 주변에 이웃을 구성하고, 구현된 문맥에 기반한 조작으로 이웃 텍스트를 생성합니다. 이 방법은 블랙박스 생성기 대신에 명확하고 통제 가능한 이웃 구성을 가능하게 하며, 그 과정에서 불투명성을 제거합니다. XPROB는 확률 기반 편집을 통해 경쟁력 있는 성능을 입증합니다.

- **Performance Highlights**: XPROB의 실험 결과는 두 개의 실제 데이터셋을 통해 가치 있는 성과를 보여줍니다. XPROB는 생성기 기반 설명기와 비교할 때보다 더 높은 안정성을 보이고, 그론한 이유는 불확실한 잠재공간의 교란 효과를 제거하기 때문입니다. 이러한 특성 덕분에 XTROB는 설명의 품질을 크게 향상시키며, 향후 설명 가능성 향상에 중요한 기여를 할 것으로 기대됩니다.



### DoubleCCA: Improving Foundation Model Group Robustness with Random Sentence Embeddings (https://arxiv.org/abs/2411.16236)
Comments:
          18 pages, 6 figures, 2 tables

- **What's New**: 본 논문은 Foundation 모델의 그룹 기반 편향에 대한 견고성을 향상시키기 위한 새로운 방법인 DoubleCCA를 제안합니다. 이 방법은 랜덤 문장과 Canonical Correlation Analysis (CCA)를 활용하여 텍스트 임베딩을 보강합니다. 기존의 모델에 비해 구현이 간단하고 효과적으로 성능과 견고성을 향상시킬 수 있음을 입증합니다.

- **Technical Details**: DoubleCCA 메서드는 원래의 프롬프트를 랜덤 단어로 확장하여 텍스트 임베딩의 표현을 풍부하게 만듭니다. 추가적인 문장 임베딩 모델을 사용하여 이 랜덤 문장들에 대한 다양한 텍스트 임베딩을 생성하고, CCA를 두 번 사용하여 이러한 임베딩을 정렬하고 원래의 표현 공간으로 재구성합니다. 이는 CLIP 모델과 통합되어 그룹 기반 편향에 대한 견고성을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 DoubleCCA 방법은 다양한 데이터셋에서 기존의 방법들보다 성능과 그룹 견고성 모두에서 더 우수한 결과를 보였습니다. 특히, 그룹 속성과 관련된 잡음을 회피하는 능력을 높여주어, 모델의 도메인 일반화 성능도 향상되었습니다. 이 접근 방식은 다양한 비전-언어 모델에 쉽게 적용할 수 있어 실용적인 솔루션이 됩니다.



### MH-MoE: Multi-Head Mixture-of-Experts (https://arxiv.org/abs/2411.16205)
Comments:
          7 pages, 0 figures

- **What's New**: 이번 논문에서는 Multi-Head Mixture-of-Experts (MH-MoE)라는 새로운 모델을 도입하고 있습니다. MH-MoE는 다양한 전문가들로부터 정보를 집합적으로 주목하는 멀티헤드 메커니즘을 활용하여 놀라운 성능을 발휘하고 있습니다. 또한, 새로운 구현은 sparse Mixture of Experts (MoE) 모델과 파라미터 및 FLOPs(operations per second) 균형을 맞춰 유지합니다.

- **Technical Details**: Sparse Mixture-of-Experts (SMoE)는 효율적인 신경망 훈련 방법으로, 모델이 입력에 따라 사용해야 할 파라미터를 동적으로 선택하는 방식입니다. 이 방법은 파라미터 수를 대폭 늘릴 수 있는 동시에 각 토큰당 FLOPs 수는 거의 일정하게 유지하는 것이 가능합니다. MH-MoE는 기존의 SMoE와 비교해 헤드의 차원을 추가하고 MoE 레이어의 양쪽에 선형 투영 레이어를 포함한 두 가지 주요 수정사항이 적용되었습니다.

- **Performance Highlights**: 언어 모델 실험 결과, 새로운 MH-MoE 구현이 기존의 MoE 및 세분화된 MoE 모델들보다 더 향상된 품질을 보여주고 있습니다. 이 모델은 BitNet와 같은 1비트 대형 언어 모델(LLMs)과의 호환성도 보여주어, 실제 활용 가능성을 높이고 있습니다. 향후 이러한 성능 개선을 활용한 다양한 응용 분야에 대한 기대가 모아집니다.



### LLM Augmentations to support Analytical Reasoning over Multiple Documents (https://arxiv.org/abs/2411.16116)
Comments:
          2024 IEEE International Conference on Big Data (IEEE BigData 2024)

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 능력을 통해 정보 분석의 깊이 있는 추론을 향상시키는 방안을 모색합니다. 정보 분석가는 방대한 보고서를 분석하여 서로 관련이 없는 정보 간의 연결을 찾고, 적의 계획과 동기를 파악합니다. LLMs의 사용을 통하여 이러한 작업이 얼마나 개선될 수 있는지, 그리고 어떤 방식으로 도움을 줄 수 있는지를 탐구합니다.

- **Technical Details**: 연구에서는 다이나믹 증거 트리(Dynamic Evidence Trees, DETs)라는 메모리 모듈을 통해 LLM의 기능을 확장하는 아키텍처를 개발하였습니다. 이 시스템은 여러 조사 분석 루트를 개발하고 추적하는 데 도움이 되며, LLMs의 잠재력을 극대화하기 위한 세 가지 단계 보강과정을 제안합니다. 또한, LLMs가 정보 수집, 가치 있는 출처의 조직화 및 설득력 있는 주장의 생성을 지원하는 방법을 실험합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, LLMs는 정보 분석에 있어 여전히 부족한 부분이 있으며, 특히 설득력 있는 주장을 구성하는 창의적인 분석 능력이 떨어진다고 보고되었습니다. 이 연구는 LLMs를 정보 분석 태스크에 적용하기 위한 구체적인 개선 방안을 제시하고, 구체적인 하위 작업인 증거 조정 및 내러티브 생성을 지원하는 방법을 강조합니다. DKTs를 사용하여 LLM의 성능을 보강할 이후 전망을 제시하였습니다.



### SAGEval: The frontiers of Satisfactory Agent based NLG Evaluation for reference-free open-ended tex (https://arxiv.org/abs/2411.16077)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)을 기반으로 여러 애플리케이션에서 고급 자연어 생성을 평가하기 위한 새로운 프레임워크인 'SAGEval'을 소개합니다. 이 프레임워크는 기존에 비해 참조 데이터가 없는 복잡한 상황에서도 LLM의 평가 품질을 개선하는 데 초점을 맞추고 있습니다. 특히, SAGEval은 크리틱 에이전트(critique agent)를 활용하여 LLM 평가자들이 생성한 점수를 교정하고, 이를 통해 주어진 태스크에 대한 전반적인 이해도를 높입니다.

- **Technical Details**: SAGEval 프레임워크는 컨텍스트 학습과 역할 기반 에이전트를 활용하여 LLM이 생성한 텍스트를 평가합니다. 이를 통해 LLM 평가자들이 점수를 비판하고 필요한 경우 피드백을 제공함으로써, 참조 문서가 없는 경우에도 LLM의 평가를 더욱 정교화합니다. 이 과정에서 LLM은 새로운 점수 기준을 제안하고, 이를 통해 평가 시스템의 적응성을 높이는 데 기여하게 됩니다.

- **Performance Highlights**: SAGEval의 성능을 검증하기 위해 연구진은 다양한 NLG(자연어 생성) 작업에서 프레임워크의 효과를 보여 주었습니다. LLM 평가자들이 human annotators의 주관적인 평가와 비교하여 더 정확한 평가를 수행할 수 있도록 돕는 것을 목표로 하고 있습니다. 또한, 본 논문에서는 데이터세트와 관련된 인간 주석도 공개하여 재현성을 쉽게 확보할 수 있는 기틀을 마련했습니다.



### TransCompressor: LLM-Powered Multimodal Data Compression for Smart Transportation (https://arxiv.org/abs/2411.16020)
Comments:
          6 pages

- **What's New**: 이번 연구는 Large Language Models(LLMs)을 이용한 새로운 프레임워크인 TransCompressor를 소개합니다. 이 기술은 멀티모달(multi-modal) 교통 센서 데이터를 효율적으로 압축하고 복원하는 방법을 제시합니다. 다양한 교통 수단인 버스, 택시, 대중교통에 걸쳐 엄격한 평가가 이루어졌으며, LLMs의 방대한 지식 기반이 데이터 압축 프로세스에 기여할 수 있음을 입증하였습니다.

- **Technical Details**: TransCompressor는 센서 데이터 압축을 위한 최초의 LLM 기반 시스템으로, 각종 센서 데이터의 압축 및 복원을 조사합니다. 이 연구는 바람압(barometer), 속도(speed), 고도(altitude)와 같은 세 가지 유형의 센서 데이터를 활용하여 교통 체계에서 LLM이 어떻게 제로샷(zero-shot) 복원을 수행하는지를 분석하였습니다. LLM은 특정 데이터 세트에 대한 재교육이나 미세 조정 없이도 높은 복원 정확도를 유지하는 것으로 나타났습니다.

- **Performance Highlights**: 종합적인 성능 평가 결과, TransCompressor는 다양한 압축 비율에서 교통 센서 데이터를 효과적으로 재구성할 수 있음을 보여주었습니다. 연구에서는 세 가지 시나리오를 통해 LLM이 데이터의 맥락을 이해하고 효과적으로 해석 및 복원할 수 있는 능력을 입증하였습니다. 향후 스마트 교통 시스템에 LLM을 통합하는 연구를 위한 기초를 마련했습니다.



### Exploring Performance Contrasts in TableQA: Step-by-Step Reasoning Boosts Bigger Language Models, Limits Smaller Language Models (https://arxiv.org/abs/2411.16002)
- **What's New**: 이 논문은 TableQA 작업에서 더 큰 언어 모델과 작은 언어 모델의 성능 차이를 단계별 추론 방법을 사용하여 조사하기 위해 Table-Logic이라 불리는 세부 프롬프트 흐름을 제안합니다. 이 방법은 질문과 테이블을 기반으로 중요한 열과 행을 식별하고, 필요한 집계(aggregations), 계산(calculations) 또는 비교(comparisons)를 결정하며, 최종적으로 결과를 추론하여 정확한 예측을 생성하는 것을 목표로 합니다. 실험 결과에 따르면 Llama-3-70B와 같은 큰 모델에서 7.8%의 정확도 향상을 확인할 수 있었고, Llama-2-7B와 같은 작은 모델은 11%의 성능 저하가 있음을 관찰했습니다.

- **Technical Details**: Table-Logic 방법은 여러 단계를 거쳐 TableQA 작업을 처리하도록 설계되었습니다. 각 단계에서 모델은 이전 단계의 출력을 참조하여 주요 열, 행 및 필요한 집계, 계산 또는 비교를 식별합니다. 또한, 우리는 7개의 하위 작업을 설계하여 TableQA 작업에 필요한 서로 다른 모델의 능력을 비교합니다. 이 하위 작업은 테이블 구조 이해, 주요 열과 행 식별, 집계, 계산 또는 비교 식별로 나뉘어 있으며, 이를 통해 두 모델 간의 성능 차이를 평가합니다.

- **Performance Highlights**: 실험을 통해 큰 모델과 작은 모델 모두에서 Table-Logic을 포함한 다양한 방법을 사용하여 성능을 평가했습니다. Llama-3-70B, GPT-3.5-Turbo 및 Qwen1.5-72B 등의 큰 모델이 잘 작동하는 반면, Llama-2-7B와 같은 작은 모델에서 성능 저하가 발생했습니다. 이러한 차이는 작은 모델이 정확한 중간 정보를 생성하는 데 어려움을 겪으며, 잘못된 결과로 이어질 수 있음을 강조합니다. 전체적으로 이 연구는 작은 모델의 성능을 향상시키기 위한 특이한 개선점을 제시합니다.



### Multi-ToM: Evaluating Multilingual Theory of Mind Capabilities in Large Language Models (https://arxiv.org/abs/2411.15999)
- **What's New**: 이 논문에서는 언어 모델(LLMs)의 다국적 사고(Theory of Mind, ToM) 능력을 평가하기 위해 두 가지 주요 요소를 포함한 포괄적인 연구를 소개합니다. 첫째, 기존 ToM 데이터 세트를 여러 언어로 번역하여 다국어 ToM 데이터 세트를 구축하고, 둘째, 문화적으로 특화된 요소를 추가하여 다양한 인구에 관련된 사회적 및 인지적 시나리오를 반영합니다. 이를 통해 LLM들이 다양한 언어적 및 문화적 맥락에서 ToM이 어떻게 나타나는지를 평가합니다.

- **Technical Details**: 데이터 준비 과정에서 MultiToM 데이터 세트를 구성하고, 이를 위해 기존의 ToMBench 데이터 세트를 바탕으로 체계적으로 샘플링하였습니다. 영어를 포함하여 아랍어, 프랑스어, 힌디어, 방글라데시어, 러시아어, 중국어로 번역된 샘플을 생성하였으며, 번역 정확도를 보장하기 위해 다단계 검증 절차를 도입했습니다. 또한, Llama-3.1 및 GPT-3.5 Turbo를 활용하여 특정 문화적 맥락을 반영함으로써, LLM의 ToM 평가에 있어 문화적 요소의 중요성을 강조하고 있습니다.

- **Performance Highlights**: 실험 결과, Faux-pas Recognition Test (FRT)에서 상대적으로 높은 성능을 보였고, 이는 단순한 true/false 형식 때문으로 분석됩니다. 반면 Scalar Implicature Task (SIT)에서는 낮은 성능을 나타내어, 수학적 추론에 대한 LLM의 한계를 반영합니다. 여러 언어를 통해 수행된 ToM 작업 성능은 작업의 특성에 더 많은 영향을 받는다고 결론지으며, 일반적으로 Non-literal Communication (NLC) 작업이 가장 높은 성능을 보였습니다.



### Investigating Factuality in Long-Form Text Generation: The Roles of Self-Known and Self-Unknown (https://arxiv.org/abs/2411.15993)
- **What's New**: 이번 연구에서는 다양한 대형 언어 모델(LLMs)의 긴 형식 텍스트 생성에서 사실성을 조사하였습니다. GPT-4와 Gemini-1.5-Pro와 같은 최신 모델들도 그들의 출력의 사실성을 정확히 판단하는 데 한계를 보이고 있습니다. 구조적으로 LLM들이 초기 문장에서는 더 높은 사실성을 나타내지만, 후속 문장에서는 사실성이 떨어지고 지원되지 않은 주장(unsupported claims)이 증가하는 경향을 보입니다.

- **Technical Details**: 연구에서는 LLM의 출력에서 아토믹 클레임(atomic claims)으로 세분화된 세 가지 유형의 주장을 분석했습니다. Self-Known 점수는 LLM이 판단한 지원된 클레임의 비율을 측정하고, Self-Unknown 점수는 LLM이 잘못된 것으로 판단한 지원되지 않은 클레임의 비율을 나타냅니다. 이러한 두 점수를 통해 LLM이 자가 평가를 얼마나 잘 수행하는지를 분석하였습니다.

- **Performance Highlights**: 연구 결과, 현재 LLM들은 후반부 문장에서 더 많은 비지원 주장을 생성하며, Self-Known 점수와 사실성 간의 상관관계도 발견되었습니다. 예를 들어, Self-Known 점수가 높을수록 사실성도 향상되지만, Self-Unknown 점수가 높으면 오히려 사실성이 낮아지는 경향이 있음을 확인했습니다. 이는 LLM의 현재 성능과 한계를 확인하는 중요한 통찰을 제공합니다.



### Generative Context Distillation (https://arxiv.org/abs/2411.15927)
- **What's New**: 최근의 대규모 언어 모델(LLM) 기반 애플리케이션에서 사용되는 프롬프트는 고정적이고 길어 컴퓨팅 오버헤드가 크다는 문제를 해결하기 위해 Generative Context Distillation (GCD)라는 경량화된 프롬프트 내재화(method) 방법이 제안되었습니다. 이 방법은 프롬프트 입력의 행동을 복제함과 동시에 프롬프트의 내용 및 모델 행동 변경의 이유를 생성하는 방식으로 작동합니다. GCD는 다양한 자동화된 애플리케이션 시나리오에서 복잡한 프롬프트를 효과적으로 내재화할 수 있음을 입증합니다.

- **Technical Details**: GCD는 프롬프트를 단순히 입력으로 사용하는 대신, 목표 프롬프트를 생성하도록 학습됩니다. 이 방법은 1) 컨텍스트 증류(context distillation) 접근 방식을 활용하여 행동을 안내하고, 2) 프롬프트에 기반하여 결과가 왜 변경되어야 하는지를 유추하는 과정을 포함하는 결합 손실 훈련을 사용합니다. 또한, 훈련 데이터셋이 없는 시나리오를 위해 대화 데이터를 자동으로 수집하는 데이터 합성(data synthesis) 방법을 도입하여, 두 역할을 전환하여 멀티 턴 의사소통 데이터셋을 생성합니다.

- **Performance Highlights**: GCD는 AgentBench를 사용한 길이 있는 프롬프트의 에이전트 애플리케이션 시나리오에서 평가되었으며, 프롬프트 입력 없이도 뛰어난 성능을 유지했습니다. OS 상호작용 에이전트 작업에서 100% 성능 유지를 달성했으며, 웹 기반 에이전트 작업에서는 1,000개 이상의 토큰에 대해 최소 82%의 성능을 유지했습니다. GCD는 길이 있는 프롬프트를 다룰 때 39%의 효율성 향상을 보이며, 기존의 압축 기반 방법을 초월하는 성능을 보여주었습니다.



### Evaluating Large Language Models for Causal Modeling (https://arxiv.org/abs/2411.15888)
Comments:
          13 pages, 6 figutrd, 4 tabels

- **What's New**: 이 논문에서는 인과적 도메인 지식을 인과적 데이터 과학의 지침과 더욱 밀접하게 일치하는 표현으로 변환하는 과정에 대해 다룹니다. LLMs (Large Language Models)를 사용하여 인과적 변수를 증류하는 두 가지 새로운 작업을 도입하고, 상호작용 엔티티를 감지하는 방법을 제안합니다. GPT-4-turbo와 Llama3-70b와 같은 최신 LLM이 인과적 도메인 지식을 인과적 변수로 변환하는 데 탁월하다는 점을 강조합니다.

- **Technical Details**: 인과 모델링은 자원 집약적이고 오류가 발생하기 쉬운 데이터 분석을 신뢰할 수 있고 강력하게 만드는 데 필수적인 역할을 합니다. 본 연구는 제공된 텍스트 엔티티로부터 인과적 관계를 추출하여 특정 인과적 변수로 변환할 수 있는 LLM의 능력을 실험적으로 조사하였습니다. 다양한 LLM과 도메인을 포함한 실험 설계를 통해 인과 변수의 값과 그 상호작용에 관련된 텍스트 엔티티를 생성하는 작업을 수행했습니다.

- **Performance Highlights**: 결과적으로, 인과 모델링 작업에서 현재의 접근법이 자원 집약적이고 오류가 발생하기 쉬운 중복을 최소화하는 데 LLM의 잠재력을 보여주었습니다. 인과 변수와 상호작용 엔티티를 대표할 수 있는 능력을 평가하여, 선택한 LLM에 따라 인과 모델링 성능이 달라질 수 있음을 발견했습니다. 이는 LLM의 한계와 향후 LLM 벤치마크에 이러한 작업을 포함할 수 있는 가능성을 조명합니다.



### LLMs Do Not Think Step-by-step In Implicit Reasoning (https://arxiv.org/abs/2411.15862)
- **What's New**: 이번 연구는 Chain-of-Thought (CoT) 방법의 명시적 생성 없이도 LLMs의 성능을 향상시키려는 시도를 다루고 있습니다. 그러나 연구에 따르면 암묵적 CoT의 효과는 전통적인 CoT 방법에 비해 여전히 부족하며, LLMs는 중간 단계에 대한 계산을 거의 수행하지 않고 있다고 강조합니다. 이는 LLMs가 경험에 의존하는 경향이 있음을 보여줍니다.

- **Technical Details**: 연구진은 Qwen2.5-72B-Instruct Team 모델을 사용하여 단순한 산술 문제에 대한 실험을 수행했습니다. 이 모델은 중간 단계의 결과를 출력하지 않고 최종적인 답변만을 제공하도록 요구받았으며, 각 단계에 대한 숨겨진 상태(hidden states)를 조사했습니다. 결과적으로, 모델은 2단계 문제에서 유일하게 두 번의 추론을 수행할 수 있었지만, 중간 단계의 계산을 수행하지 않았습니다.

- **Performance Highlights**: 이번 실험을 통해 LLMs가 암묵적 추론 과정에서 안정적이지 않으며, 직관적이고 직접적인 사고 방식에 의존한다는 사실이 확인되었습니다. 결국, LLMs는 특히 큰 모델일수록 산술 문제에서 단지 정답을 제시하는 것에 그치지 않고, 명시적 CoT 없이는 단계별 추론을 수행하지 않는다는 것을 발견했습니다. 연구는 명시적 CoT 방법론의 지속적 필요성을 강조하며 복잡한 작업에서 LLM의 능력을 향상시키는 데 중요한 통찰력을 제공합니다.



### Is Training Data Quality or Quantity More Impactful to Small Language Model Performance? (https://arxiv.org/abs/2411.15821)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구는 작은 언어 모델(Small Language Models, SLMs)의 성능에 대한 훈련 데이터의 질과 양의 상대적 영향을 조사합니다. TinyStories 데이터셋을 사용하여 실증 분석을 수행했으며, 데이터셋의 크기와 중복에 따른 변화 분석도 포함되어 있습니다.

- **Technical Details**: 연구에서는 데이터셋 크기를 원래의 25% 및 50%로 조정하고, 중복 비율을 25%, 50%, 75%, 100%로 통제하여 실험을 진행했습니다. 모델 성능은 validation loss, accuracy, perplexity 지표를 기반으로 평가되었으며, 훈련 데이터의 질이 SLMs의 전반적인 성능에 더 큰 역할을 한다는 결과가 나타났습니다.

- **Performance Highlights**: 최소한의 중복은 모델의 정확도에 긍정적인 영향을 미쳐 25% 중복 시 정확도가 0.87% 증가했습니다. 반면, 과도한 중복은 성능 저하를 초래하여 100% 중복시 정확도가 40% 감소하는 결과를 보였습니다. 이러한 결과는 대규모 모델 교육이 재정적 및 계산적 부담을 초래하며, 에너지 소비 문제와 함께 AI 기술의 보다 민주화된 접근을 모색할 필요성을 제기합니다.



### LoRA-Mini : Adaptation Matrices Decomposition and Selective Training (https://arxiv.org/abs/2411.15804)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 Low-Rank Adaptation(LoRA) 방법을 최적화한 LoRA-Mini를 제안합니다. LoRA-Mini는 저랭크 매트릭스를 네 부분으로 나누어 두 개의 내부 매트릭스만 훈련 가능하게 하여 파라미터 효율성을 크게 향상시킵니다. 이 접근방식은 표준 LoRA에 비해 훈련 가능한 파라미터 수를 최대 20배까지 줄여주며, 성능 또한 유지할 수 있습니다.

- **Technical Details**: LoRA-Mini는 각 LoRA 매트릭스를 훈련 가능 매트릭스와 고정 매트릭스로 분할하여 선택적 훈련을 가능하게 합니다. 이를 통해 업데이트 공간을 제한하고, 고정된 외부 매트릭스가 훈련 과정에서 가이드를 제공합니다. 수학적으로는 LoRA 매트릭스 A와 B를 보조 및 훈련 가능한 구성 요소로 분해하며, 중간 매트릭스만 훈련하도록 설정됩니다.

- **Performance Highlights**: 다양한 모델(BERT, RoBERTa, T5 등)에 대한 실험을 통해 LoRA-Mini가 전체 파인튜닝 접근법과 비교해 비슷한 정확도를 달성하면서 메모리 요구 사항을 대폭 줄일 수 있음을 입증했습니다. 본 연구의 주요 기여는 매트릭스 내 선택적 파라미터 동결 평가, 이전 PEFT 방법 대비 훈련 가능한 파라미터 수 최소화, 다양한 태스크 및 모델에 대한 확장성 평가를 포함합니다.



### A Method for Building Large Language Models with Predefined KV Cache Capacity (https://arxiv.org/abs/2411.15785)
- **What's New**: 이번 논문은 사전 정의된 Key-Value (KV) 캐시 용량을 갖춘 대형 언어 모델을 구축하는 방법을 제안합니다. 이 방법은 Transformer decode-only 아키텍처에서 주의(Attention) 계층에 적합하며, 기존 KV 캐시의 불필요한 메모리 소모 문제를 해결합니다. 고정 길이 KV 캐시를 사용하여 무한 컨텍스트 처리를 지원하면서 메모리 사용량을 줄이는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 Transformer 아키텍처의 주의 계층에 적용됩니다. 초기 Key-Value 벡터 시퀀스는 0 또는 랜덤 값으로 초기화되며, 입력 벡터를 매핑하여 쓰기 쿼리 벡터와 첫 번째 쓰기 Key-Value 벡터를 생성합니다. 이후 유사도 점수를 계산하여 쓰기 가중치 벡터를 얻고, 과거 Key-Value 벡터 시퀀스를 업데이트하여 새로운 정보로 갱신합니다. 이러한 동적 업데이트는 모델의 추론(Inference) 품질 유지를 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 메모리 사용량을 상당히 감소시키면서도 모델의 추론 품질을 유지하는 것으로 나타났습니다. 이 방법은 전통적인 KV 캐시 메커니즘 대비 효율성을 높이며, 시스템의 처리 성능도 향상시킵니다. 따라서 대형 언어 모델의 효율적인 활용이 기대됩니다.



### Detecting Turkish Synonyms Used in Different Time Periods (https://arxiv.org/abs/2411.15768)
Comments:
          published at Innovations in Intelligent Systems and Applications Conference (Akıllı Sistemlerde Yenilikler ve Uygulamaları Konferansı - ASYU) 2024

- **What's New**: 이번 연구에서는 20세기 언어 개혁의 영향을 받은 터키어의 시기별 동의어 탐지 방법을 제안합니다. 제안된 두 가지 방법은 Orthogonal Procrustes (OP)와 Spearman rank correlation을 결합한 방법입니다. 이런 접근은 역사적 문서에서 단어의 의미를 더 효과적으로 이해하는 데 도움이 됩니다. 특히 1960년대에서 1980년대까지의 시점에서 일관된 성능을 보여주지만, 그 이후의 시점에서는 약간의 성능 저하가 발생합니다.

- **Technical Details**: 제안된 방법은 두 개의 서로 다른 시기의 문서에서 발생한 임베딩 공간을 정렬하는 데 중점을 둡니다. 첫 번째 방법인 OP는 두 개의 시간이 다른 문서에서 임베딩 공간 간의 변환 행렬을 찾습니다. 두 번째 방법인 OP+SC는 첫 번째 방법의 결과를 기반으로 Spearman의 순위 상관관계를 활용하여 선별하는 신규 개선 방법입니다. 실험은 1920년대부터 2020년대까지의 터키어 다이아크로닉 코퍼스인 Turkronicles를 기반으로 진행되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들이 Zhang et al.의 기준 방법보다 우수한 성능을 보이는 것으로 나타났습니다. OP와 SVD 임베딩을 사용할 때, Spearman의 상관계수를 포함할 경우 OP 모델의 성능이 대부분의 경우 향상됩니다. 또한, 각 방법은 서로 다른 시점 간의 시간적 거리 변화에 대해 강인성을 유지하는 것으로 평가되었습니다.



### Development of Pre-Trained Transformer-based Models for the Nepali Languag (https://arxiv.org/abs/2411.15734)
- **What's New**: 이 논문에서는 네팔어( Nepali )에 대해 지금까지 가장 큰 규모의 텍스트 데이터셋인 27.5GB를 수집하여 BERT, RoBERTa, GPT-2 모델을 사전 훈련했습니다. 이는 지금까지의 네팔어 코퍼스보다 약 2.4배 큰 데이터입니다. 연구 초점은 기존의 인코더 기반 모델에 국한되지 않고 디코더 기반 아키텍처에 대한 탐구의 필요성을 강조합니다. 또한, 본 연구는 모노링구얼(monolingual) 네팔어 데이터의 가능성을 탐색하기 위한 지시 조정(instruction tuning)을 수행하였습니다.

- **Technical Details**: 자연어 처리(NLP) 분야에서 트랜스포머(Transformer) 기반 사전 훈련 언어 모델이 지배해왔습니다. 하지만 네팔어는 약 3200만 명이 사용하는 언어임에도 불구하고 자원 부족으로 인해 크게 저평가되고 있습니다. 이 연구에서는 세 가지 모델을 사전 훈련하였으며, 이는 기존 최대 모델들보다 Nep-gLUE 기준에서 2점 높은 95.60점을 기록했습니다. 실험적으로 지시 조정 기술을 사용하여 모노링구얼 네팔어에 대한 성능을 향상시켰습니다.

- **Performance Highlights**: 본 연구의 모델은 텍스트 생성(task) 성능에서도 기존 모델들을 초월했습니다. 기존 언어 모델에 비해 이해 및 생성 분야에서 누락된 부분을 메우고 있습니다. 저자들은 성능 개선이 네팔어 NLP 분야의 새로운 기준을 세우고 낮은 자원 언어에 대한 연구에 큰 기여를 할 것으로 기대하고 있습니다. 연구 결과는 향후 네팔어 및 유사한 자원 부족 언어를 위한 더 나은 모델 개발에 기초가 될 것입니다.



### LLaMA-MoE v2: Exploring Sparsity of LLaMA from Perspective of Mixture-of-Experts with Post-Training (https://arxiv.org/abs/2411.15708)
Comments:
          Technical report,13 pages

- **What's New**: 최근 Mixture-of-Experts (MoE) 모델이 큰 모델의 크기를 확장하는 데 있어 인기를 끌고 있습니다. 이 연구에서는 LLaMA 모델의 희소성(sparsity)를 자세히 살펴보고, transformer 블록의 attention 모듈과 MLP 모듈에 대해 MoE를 구축합니다. 특히, 모델의 희소성으로 인한 성능 저하를 보완하기 위해 두 단계의 후속 훈련 전략을 설계하였습니다. 실험 결과는 LLaMA3 모델이 instructed MoE 모델의 미래 발전 가능성을 보여줍니다.

- **Technical Details**: MoE 모델의 주요 기능은 전문가(expert)들을 선택적으로 활성화하여 모델 파라미터의 수를 줄이는 것입니다. 이 연구에서는 attention 모듈과 MLP 모듈에서의 전문가 구축 전략을 탐색하고, 다양한 성격을 지닌 attention head들을 하나의 전문가로 통합하는 방법을 제안합니다. 모델의 파라미터 일부만 활성화되도록 변환한 MoE 모델을 사용하여, 대화, 코드, 수학 등 다양한 분야의 작업을 수행할 수 있습니다. 또한, instruction tuning(IT) 기법을 적용하여 후속 훈련을 통해 모델 성능을 향상시키는 두 단계 훈련 패러다임을 마련하였습니다.

- **Performance Highlights**: 실험을 통해 LLaMA3-8B 모델의 MLP-MoE와 Attention-MoE 모델을 평가하였고, 성능 개선을 위한 후속 훈련 데이터에서 모델 성능 회복에 대한 효과를 확인하였습니다. 10개의 기준 벤치마크에서의 성능 향상 결과는 제안된 프레임워크가 효과적임을 입증합니다. 이 연구는 MoE 모델의 활용 가능성 및 후속 훈련 프로세스를 통해 다양한 측면에서 모델 성능을 개선할 수 있음을 보여줍니다.



### RAMIE: Retrieval-Augmented Multi-task Information Extraction with Large Language Models on Dietary Supplements (https://arxiv.org/abs/2411.15700)
- **What's New**: 본 연구에서는 RAMIE(Retrieval-Augmented Multi-task Information Extraction) 프레임워크를 개발하여 잘 정리되지 않은 임상 기록에서 다양한 정보 추출 작업을 동시에 처리할 수 있도록 하였습니다. 4가지 주요 작업인 명명된 개체 인식(NER), 관계 추출(RE), 삼중 추출(TE), 사용 분류(UC)를 통해 다목적 문제를 해결하는 방법을 제시했습니다. 본 프레임워크는 기존의 LLM(large language model)을 활용하여 보다 정확한 정보 추출을 가능하게 합니다.

- **Technical Details**: RAMIE는 MTL(multi-task learning), RAG(retrieval-augmented generation), 그리고 설명 조정(instruction fine-tuning) 기법을 결합하여 효율성을 높이고 정확한 정보 추출을 위해 최적화된 모델입니다. 각 작업은 비정형 임상 텍스트에서 구조화된 데이터를 추출하는 데 필수적이며, 효율적인 정보 추출을 통해 실제 애플리케이션에서 활용도를 높일 수 있습니다. 이 프레임워크는 다양한 도메인에서도 적용될 가능성을 가지고 있습니다.

- **Performance Highlights**: RAMIE 프레임워크를 활용하여 Llama2-13B 모델은 NER 작업에서 F1 스코어 87.39를 기록하며 3.51% 향상된 성과를 보여주었습니다. 관계 추출(RE) 작업에서 93.74(F1 스코어, 1.15% 향상)를 달성하였으며, 사용 분류(UC)에서는 MedAlpaca-7B가 93.45의 최고 점수를 기록했습니다. VC와 RAG의 조합이 전체 정확도를 크게 향상시켰음을 보여주는 실험 결과도 도출되었습니다.



### Deep Sparse Latent Feature Models for Knowledge Graph Completion (https://arxiv.org/abs/2411.15694)
- **What's New**: 최근 지식 그래프 완성(Knowledge Graph Completion, KGC) 연구가 텍스트 기반 접근법에 중점을 두고 있지만, 이러한 방법들은 종종 지식 그래프의 핵심 구성 요소인 엔티티 간의 복잡한 상호 연결 관계를 간과하는 경향이 있습니다. 본 논문에서는 희소한 잠재 특성 모델(sparse latent feature models) 프레임워크를 제안하여, 깊은 변분 오토인코더(deep variational autoencoder, VAE)를 통해 최적화했습니다. 이 접근법은 누락된 트리플을 효과적으로 보완할 뿐만 아니라, 텍스트 정보를 활용해 잠재 구조를 명확하게 해석할 수 있게 합니다.

- **Technical Details**: 희소 잠재 특성 모델 DSLFM-KGC는 지식 그래프에서의 덴시티 기반 접근법을 확장하여 엔티티 간의 관계 연결에 초점을 맞춘 확률적 모델입니다. 이를 통해 추가적인 희소 군집 기능을 트리플 표현에 통합하였고, 대규모 그래프에서의 효율적인 추론을 가능하게 하였습니다. 이 모델은 숨겨진 커뮤니티 구조를 탐색하여 누락된 트리플 완성 문제를 해결합니다.

- **Performance Highlights**: 다양한 데이터셋(WN18RR, FB15k-237, Wikidata5M)에서 수행된 종합 실험은 DSLFM-KGC가 KGC 작업을 관리하고 해석 가능한 잠재 구조를 공개하는 데 있어 우수한 성능을 입증했습니다. 본 모델은 관련 커뮤니티 간의 상호 연결성을 활용하여 성능을 높이고 있으며, KGC 분야에서 해석 가능성과 스케일링의 장점을 제공합니다.



### Ontology-Constrained Generation of Domain-Specific Clinical Summaries (https://arxiv.org/abs/2411.15666)
Comments:
          24th International Conference on Knowledge Engineering and Knowledge Management (EKAW 2024), November 26-28, 2024, Amsterdam, The Netherlands

- **What's New**: 이 연구는 온톨로지(ontology)를 활용하여 특정 도메인에 맞춘 요약을 생성하는 새로운 방법론을 제안합니다. 기존의 대형 언어 모델(LLM)은 텍스트 요약에 대한 잠재력을 가지고 있지만, 도메인에 대한 적합성과 비정상적 정보 생성을 줄이는 데 한계가 있습니다. 해당 연구는 의학 분야에서 전자 건강 기록(EHRs) 요약에서 효과적인 결과를 보여주고 있습니다.

- **Technical Details**: 제안된 방법은 온톨로지에 기반한 제한된 디코딩(process) 과정을 통해 요약 생성 시 비현실적인 정보 생성을 감소시키고, 더 많은 관련 정보를 포함하도록 합니다. 특히, LLM의 출력이 온톨로지에서 정의된 관계와 개념에 맞도록 제한하여 생성된 내용의 신뢰성을 높입니다. 이는 의학적 특성에 맞춘 요약을 생성하는 데 필수적입니다.

- **Performance Highlights**: MIMIC-III 데이터셋을 활용한 평가를 통해, 본 연구 방법이 임상 노트의 도메인 맞춤 요약과 환각 감소를 효과적으로 달성하는 것을 보여줍니다. 이러한 결과는 의사들이 중요한 정보에 집중할 수 있는 기반을 마련해주며, 의료 환경에서의 사용 가능성을 더욱 높입니다.



### Improving Next Tokens via Second-Last Predictions with Generate and Refin (https://arxiv.org/abs/2411.15661)
- **What's New**: 이 논문에서는 GPT와 BERT 모델의 장점을 결합하기 위한 새로운 접근 방식인 'generate-then-refine'을 제안합니다. 저자들은 상대적으로 간단한 구조를 통해 토큰을 마스킹하여 훈련 효율성을 높이며, 표준 GPT의 다음 토큰 예측 성능을 뚜렷하게 개선하는 방법을 보여줍니다. 특히, 두 번째 마지막 토큰을 예측하는 데 집중하여, 기존의 다음 토큰 예측보다 15% 이상 높은 정확도를 달성했습니다.

- **Technical Details**: 저자들은 두 가지 모델을 사용하여 이 방법을 구현합니다. 첫 번째는 GPT-2 변형을 사용하는 단방향 자율회귀 언어 모델이며, 두 번째는 두 번째 마지막 토큰을 예측하도록 훈련된 양방향 인코딩 모델입니다. 이 구성은 이전 토큰 예측을 피드백으로 사용하여 다음 토큰 예측의 신뢰성과 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 저자들은 그들이 제안한 방법이 여러 데이터셋에서 일관되고 통계적으로 유의미한 향상을 보여준다는 것을 입증합니다. 비록 다음 토큰 예측에서의 개선이 다소 작을 수 있지만, 두 모델의 조합을 통해 얻는 성능 향상은 상당한 가치를 지니고 있습니다. 저자들은 이 접근 방식이 LLM들이 추론 중에 자기 수정할 수 있는 가능성에 대한 논의에서도 중요한 기여를 할 것으로 보고 있습니다.



### AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Datas (https://arxiv.org/abs/2411.15640)
- **What's New**: 이 논문에서는 아프리카 대륙의 의료 접근성을 높이기 위한 첫 번째 파나프리칸 영어 멀티스페셜티 의료 질의응답(QA) 데이터셋, AfriMed-QA를 소개합니다. 이 데이터셋은 16개 국가의 60개 이상 의과대학에서 수집한 15,000개의 질문을 담고 있으며, 32개의 의료 전문 분야를 포괄하고 있습니다. 이는 저소득 및 중간 소득 국가(LMICs)의 의사 부족 문제를 해소하기 위한 중요한 자원으로 작용할 수 있습니다.

- **Technical Details**: AfriMed-QA는 전문가 MCQ(다지선다형 질문) 4,000개, 개방형 단답형 질문(SAQ) 1,200개, 소비자 질문(CQ) 10,000개 등 총 15,275개의 질문을 포함하고 있습니다. 데이터셋은 621명의 기여자로부터 모집되었으며, 동아프리카 지역의 다양한 의료 데이터를 통합하여 LLM(대형 언어 모델)의 성능을 평가합니다. 연구는 LLM의 정확성과 인구 통계적 편향(demographic bias)을 평가하는 양적 및 질적 접근 방식을 이용합니다.

- **Performance Highlights**: 조사 결과, 특정 전문 분야와 지역에 따라 LLM 성능에 상당한 차이가 있음을 발견했습니다. LLM은 전반적으로 기존의 의료 시험인 USMLE(MedQA)와 비교하여 성능이 떨어졌고, 특정 의학 분야에서도 성능이 일관되지 않았습니다. 인간 평가에서는 LLM의 답변과 설명이 전문가의 답변에 비해 지속적으로 더 선호된다는 결과를 도출하였습니다.



### "All that Glitters": Approaches to Evaluations with Unreliable Model and Human Annotations (https://arxiv.org/abs/2411.15634)
Comments:
          20 pages, 15 figures, 58 pages with references and appendices

- **What's New**: 이번 연구는 인간의 레이블(annotations)에서 발생하는 오류를 분석하고, 이러한 오류가 모델 평가 과정에서 정확성(accuracy), 편향(bias), 공정성(fairness)과 유용성(usefulness)에 미치는 영향을 다룹니다. 특히, K12 교육의 교실 수업 품질 자동 평가를 주제로, 두 가지 대규모 언어 모델(LLM) 아키텍처를 사용하여 인간 레이블 품질을 다양한 차원에서 평가하는 새로운 방법을 제시합니다.

- **Technical Details**: 이 연구는 교육 분야에서의 레이블 품질을 평가하기 위한 여섯 가지 차원인 일관성(concordance), 자신감(confidence), 유효성(validity), 편향(bias), 공정성(fairness), 도움됨(helpfulness) 등을 이용하여, 인간 전문가의 불확실한 레이블을 통해 얻은 데이터를 평가하고 분석합니다. 또한, 모델 사용이 인간 레이블 품질에 미치는 변화를 추정하고, 현재 데이터의 일반화 가능성 내에서 일부 LLM이 교육용 레이블의 품질을 향상시킬 수 있는 방법을 식별합니다.

- **Performance Highlights**: 연구 결과, 일반적인 평가 지표에 의존할 경우 레이블과 모델 품질이 가려질 수 있으며, 일부 모델은 '슈퍼 인간'(super-human) 성과를 달성할 수 있지만, 더 엄격한 평가 기준을 적용하면 비합리적 상관관계 및 인종 편향이 드러납니다. 이 연구는 노이즈가 많은 레이블 환경에서도 정확하고 신뢰할 수 있는 평가 정보를 제공할 수 있도록 하는 더욱 견고한 평가 기법의 필요성을 강조하며 향후 연구에 대한 제언도 포함하고 있습니다.



### Multi-label Sequential Sentence Classification via Large Language Mod (https://arxiv.org/abs/2411.15623)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 이 논문은 LLM-SSC라는 새로운 프레임워크를 제안하며, 이를 통해 단일 및 다중 레이블의 순차 문장 분류(SSC) 작업을 지원합니다. 이 프레임워크는 LLM(large language model)을 활용하여 설계된 프롬프트를 통해 SSC 레이블을 생성함으로써 작업 이해를 향상시킵니다. 또한, 자동 가중치를 적용한 다중 레이블 대조 학습 손실 함수를 도입하여 기존의 SSC 방법들이 갖고 있던 여러 한계를 극복하고자 합니다.

- **Technical Details**: LLM-SSC는 조건부 텍스트 생성(task of conditional text generation)으로 SSC를 접근하며, 문맥 정보를 완전하게 통합한 프롬프트를 사용합니다. 이를 위해 각 문장의 SSC 레이블을 생성하기 위한 확률을 모델링하며, 문장 간 유사성을 기반으로 데모 샘플을 선택합니다. 이 과정에서 SimCSE 사전 훈련 모델의 임베딩을 활용합니다.

- **Performance Highlights**: 실험 결과, LLM-SSC는 인컨텍스트 학습(in-context learning) 및 파라미터 효율적 미세 조정(parameter-efficient fine-tuning) 상황에서 강력한 성능을 보여주었습니다. 새로 포함된 biorc800 데이터셋은 생물 의학 분야의 비정형 초록에서 수작업으로 주석이 달린 다중 레이블 SSC 데이터셋으로, 이 연구의 유용성을 더욱 높입니다. LLM 기반 프레임워크는 앞으로의 과학 논문 정보 검색 및 요약 작업을 한층 개선할 것으로 기대됩니다.



### A Survey on LLM-as-a-Judg (https://arxiv.org/abs/2411.15594)
Comments:
          33 pages, 9 figures. arXiv admin note: text overlap with arXiv:2310.05470 by other authors

- **What's New**: 이번 논문은 LLM(as-a-Judge) 시스템을 구축하는 방법론을 제시하는 포괄적인 설문을 제공하며, 신뢰할 수 있는 평가 시스템의 개발을 위한 전략들을 탐구합니다. LLM의 다양한 데이터 타입 처리 능력을 활용하여 전통적인 전문가 평가 방식의 대안으로 자리 잡고 있으며, 평가 일관성 향상 및 편향 완화 등의 신뢰성 강화를 위한 방법을 모색하고 있습니다.

- **Technical Details**: LLM은 과거 인공지능 및 소프트웨어 공학, 사회 과학에 걸쳐 성공적으로 활용되어 왔으며, 이는 LLM을 평가자로 활용하는 'LLM-as-a-Judge' 모델의 채택 증가로 이어졌습니다. 이 프레임워크는 평가 기준의 명확한 정의와 다양한 평가 시나리오에 적합한 적응 능력을 포함하여, 평가 과정에서 발생할 수 있는 편향을 탐구하고 해결책을 제시합니다. 또한, 새로운 벤치마크를 통해 다양한 신뢰성 향상 전략의 효과성을 분석할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLM-as-a-Judge는 기존 인간 평가 방식에 비해 평가의 효율성을 향상시킬 잠재력이 있으며, LLM이 제공하는 평가의 일관성 및 비용 효율성은 혁신적입니다. LLMs는 다양한 입력 유형을 처리할 수 있는 유연성을 제공하고, 주요 평가 메트릭에서는 놓칠 수 있는 미세한 qualitative insights를 통합할 수 있는 가능성을 가지고 있습니다. 본 논문은 LLM으로서의 평가자 시스템의 신뢰성, 확장성 및 적용 가능성을 높이기 위한 미래 연구 방향에 대한 통찰을 제시합니다.



### Transparent but Powerful: Explainability, Accuracy, and Generalizability in ADHD Detection from Social Media Data (https://arxiv.org/abs/2411.15586)
Comments:
          12 pages (including references and appendix)

- **What's New**: 본 논문은 소셜 미디어 데이터를 활용하여 ADHD의 선별적 검사를 위한 머신러닝과 딥러닝 기법을 통합한 새로운 접근 방식을 소개합니다. 특히 BiLSTM 및 transformer 기반 모델을 사용하여 언어적 패턴을 분석하고 ADHD 관련 텍스트 데이터를 처리합니다. 연구 결과는 다양한 모델 간의 해석 가능성과 성능 간의 trade-off를 제시하며, 이에 따른 최적의 접근 방식을 모색합니다.

- **Technical Details**: 논문에서는 비선형 기계학습 모델과 딥러닝 기법을 적용하여 ADHD 탐지를 위한 모델을 개발하였습니다. 이 모델은 8가지 차원의 언어적 행동을 포착하는 사람의 해석이 가능한 특성을 포함하여 학습됩니다. 평가 방법론으로는 Reddit 데이터로 학습하고 Twitter 데이터로 성능을 평가하는 out-of-distribution 실험을 실시하였습니다. 마지막으로, feature ablation 실험을 통해 ADHD와 강하게 연관된 가장 유익한 특성 그룹을 식별하였습니다.

- **Performance Highlights**: 이 연구는 비선형 기계학습 모델과 딥러닝 모델 간의 균형을 강조하며, BiLSTM이 최적의 해석 가능성과 정확도를 제공하는 모델로 평가되었습니다. 각 모델의 성능을 비교한 결과, 특정 언어적 특징들이 ADHD 탐지에 중요한 역할을 한다는 사실이 발견되었습니다. 전체 데이터베이스는 12,000명의 진단된 ADHD 사용자와 조정된 수의 대조군 사용자를 포함하고 있으며, 이는 향후 디지털 선별 도구 개발에 기여할 수 있는 토대를 마련합니다.



### From MTEB to MTOB: Retrieval-Augmented Classification for Descriptive Grammars (https://arxiv.org/abs/2411.15577)
Comments:
          submitted to COLING 2025

- **What's New**: 최근 언어 모델링의 발전으로 인해 매우 자원 부족한 언어에서 인-context learning, instruction following, 기계 번역의 제로샷(zero-shot) 능력이 크게 향상되었습니다. 본 논문에서는 복잡한 언어학적 문법의 기술적 설명에서 정보를 추출하고 분류하는 모델의 성능을 평가하기 위한 벤치마크 세트를 소개합니다. 이 연구는 248개 언어와 142개 언어 가계를 아우르는 언어학적 설명을 포함하여 언어 모델의 in-context 능력을 평가하는 최초의 포괄적인 자원을 제공합니다.

- **Technical Details**: 우리는 Retrieval-Augmented Generation (RAG) 기반 접근법을 제시하며, 이를 통해 저자원 언어를 위한 기계 번역과 언어 모델링의 성능을 향상시키고자 합니다. 이 시스템은 문법의 특정 유형적 특성(예: 문장 성분의 순서) 기반으로 정보를 추출하고, 대형 언어 모델(LLM)에게 이러한 정보를 기반으로 의미를 결정할 수 있도록 돕습니다. 또한, 700개의 문단과 14개의 설명 문법을 포함하는 벤치마크를 포함하여 LLM의 기계 독서 능력을 평가합니다.

- **Performance Highlights**: 본 논문에서는 언어 자원이 부족한 국가의 언어들을 위한 능력 향상에 중점을 두고 있으며, 실험을 통해 RAG 파이프라인이 언어 모델의 성능을 어떻게 개선하는지를 폭넓게 평가합니다. 따라서 우리는 LLM이 주어진 언어 문법을 통해 유형적 특성을 결정할 수 있는 능력을 강화하고, 저자원 언어에 대한 NPL의 접근성을 높이는 데 기여하고자 합니다.



### Enhancing Grammatical Error Detection using BERT with Cleaned Lang-8 Datas (https://arxiv.org/abs/2411.15523)
Comments:
          10 pages, 6 tables, 20 references

- **What's New**: 이 논문은 문법 오류 탐지(Grammatical Error Detection, GED)를 위한 개선된 LLM 기반 모델을 제안합니다. 전통적인 GED 접근 방식은 수작업으로 설계된 특징(feature)을 사용했으나, 최근에는 신경망(Neural Networks, NN)이 이러한 특징을 자동으로 발견하여 성능을 향상시키고 있습니다. BERT-base-uncased 모델은 학습 데이터에서 98.49%의 정확도와 0.91의 F1 점수를 기록하며, 데이터 정제의 중요성을 보여주었습니다.

- **Technical Details**: 이 연구는 Lang-8 데이터셋의 품질을 개선하기 위해 이를 철저히 정제하였으며, 여러 Transformer 기반 모델인 BERT와 RoBERTa를 비교했습니다. 이전의 모델과 비교해 BERT와 RoBERTa의 정제된 데이터셋에서의 성능 향상을 입증했습니다. 또한 GPT-4와 Llama-3-70B-instruct 같은 생성 모델을 활용한 평가를 통해, fine-tuning 없이도 문법 오류 탐지 작업에서 효과를 보여주었습니다.

- **Performance Highlights**: 이 연구의 실험 결과, BERT-base-uncased 모델이 가장 높은 성능을 보였으며, 기존 모델보다 월등한 결과를 나타냈습니다. 특히, 대규모 모델인 BERT-large-uncased와 RoBERTa-large가 성능 향상에 유의미하지 않음을 강조하여, 가장 큰 모델이 항상 가장 좋은 성능을 내는 것은 아님을 보여줍니다. 데이터 정제와 간단한 Transformer 기반 모델들이 GED 품질 향상에 매우 크게 기여할 수 있음을 증명했습니다.



### Traditional Chinese Medicine Case Analysis System for High-Level Semantic Abstraction: Optimized with Prompt and RAG (https://arxiv.org/abs/2411.15491)
- **What's New**: 이 논문은 전통 중국 의학(Traditional Chinese Medicine, TCM) 임상 사례 데이터베이스 구축을 위한 웹 스크래핑(web scraping) 기술을 이용한 계획을 제시하고 있습니다. 360doc를 포함한 여러 플랫폼에서 5,000개 이상의 TCM 임상 사례를 수집하고, 데이터 정화(data cleaning)와 구조화 과정을 통해 데이터셋을 준비하였습니다. 저자들은 Baidu의 ERNIE Speed 128K API를 사용하여 불필요한 정보를 제거하고, DeepSeekv2 API를 통해 최종 답변을 생성하여 표준 JSON 형식으로 결과를 출력했습니다.

- **Technical Details**: 데이터 검색 및 추론을 위해 저자들은 벡터 유사성(vetor similarity) 기반의 하이브리드 매칭 기법을 구현하였습니다. 여기서는 Jieba 분리를 통해 키워드 매칭을 결합한 두 단계 검색 방법을 적용했습니다. RAG(Retrieval Augmented Generation)와 재순위(re-ranking) 기술을 이용하여 초기 검색 결과와 재정렬 전략을 통합하여 모델 응답의 품질을 향상시켰습니다. 전반적으로 높은 품질의 TCM 임상 사례 데이터베이스를 구축하여 연구자들과 실무자들에게 유용한 자원을 제공합니다.

- **Performance Highlights**: TCM의 복잡한 추론 프로세스를 모델링하는 데 AI의 잠재력 탐구와 함께, 기존 데이터셋에 기반한 진단 정확도를 높이는 슈퍼바이즈드 학습(supvervised learning)이 가능해졌습니다. Baidu ERNIE Speed 128K API를 활용한 데이터 정화 및 구조화 과정과 함께 gte-Qwen2-1.5B-instruct API를 이용한 하이브리드 검색 기법이 특히 주목받습니다. 이 시스템은 TCM 연구 및 진료에 있어 정확성을 높이고, 더 나아가 임상적 일관성을 유지하는 데 도움을 줄 것입니다.



### Automatic Evaluation for Text-to-image Generation: Task-decomposed Framework, Distilled Training, and Meta-evaluation Benchmark (https://arxiv.org/abs/2411.15488)
- **What's New**: 이 논문은 텍스트-이미지 생성 모델의 평가 방법을 개선하기 위해, GPT-4o 기반의 작업 분해 평가 프레임워크를 제안합니다. 이를 통해 복잡한 평가 작업을 단순한 서브 작업으로 나누어 처리함으로써 자동으로 새로운 학습 데이터셋을 구성하게 됩니다. 이 프레임워크는 7B 개방형 MLLM, MiniCPM-V-2.6의 효율적인 자동 평가 모델로의 변환을 촉진합니다.

- **Technical Details**: 제안된 프레임워크는 입력 텍스트 프롬프트에서 сущности(entity)와 내재적 속성(intrinsic properties), 그리고 관계 속성(relational attributes)을 추출하기 위해 GPT-4o를 사용합니다. 이러한 정보를 바탕으로 세 가지 차원(시각적 외형, 내재적 특성, 관계 속성)을 통해 평가 질문을 구성하고, 이를 통해 이미지와 캡션 간의 품질 점수를 산출합니다. 또한, 개별 평가 차원에 대해 예측된 결과를 통합하여 종합적인 판단을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 MLLM은 기존의 최첨단 모델인 GPT-4o-베이스 기준 및 VIEScore와 비교하여 Spearman 및 Kendall의 상관 관계에서 4.6% 이상의 개선을 달성했습니다. 또한, 새롭게 수동으로 주석이 달린 메타 평가 벤치마크를 통해 기존의 평가 방법들과 비교하여 더욱 신뢰성 있는 평가를 제공합니다. 이 모델은 고품질의 생성된 이미지를 평가하는 데 더욱 효과적임을 입증했습니다.



### Seed-Free Synthetic Data Generation Framework for Instruction-Tuning LLMs: A Case Study in Tha (https://arxiv.org/abs/2411.15484)
Comments:
          ACL-SRW 2024. Our code and dataset are publicly available at this https URL

- **What's New**: 본 논문에서는 저자들이 태국어와 같은 저자원 언어의 대형 언어 모델(LLMs)을 위한 효과적인 instruction-tuning을 위한 합성 데이터 접근 방식을 제안합니다. 특히, 이들은 유창성(fluency), 다양성(diversity), 문화적 맥락(cultural context)의 세 가지 주요 특성이 instruction-tuning 데이터셋의 효과에 기여한다고 주장합니다. 이 연구는 이러한 특성을 포함한 합성 데이터셋을 생성하는 'seed-data-free' 프레임워크를 개발하여, 비용 효율적인 데이터 생성이 가능함을 보여줍니다.

- **Technical Details**: 저자들은 LLaMa-3 8B 모델을 기반으로 하는 신경망을 사용하여 다양한 주제를 생성하고, 위키백과에서 관련 맥락을 검색한 후 질문 응답이나 요약 및 대화와 같은 작업을 위한 지침을 만듭니다. 실험 결과, 저자들은 5,000개의 지침만으로도 세 가지 특성을 모두 포함한 합성 데이터셋이 최첨단 아이디어와 경쟁력 있는 성능을 나타낸다고 보고합니다. 이는 기존 데이터셋이 10배에서 100배 더 많은 예시를 사용하는 것과 비교하여, 데이터 효율성을 크게 향상시킵니다.

- **Performance Highlights**: 연구 결과, 세 가지 특성을 모두 포함한 데이터로 교육한 LLM이 저자원 언어, 특히 태국어에서 현저한 성능 향상을 보여주었습니다. 저자들은 5,000개의 지침만으로도 다른 방법들에 비해 유사한 성능을 달성할 수 있었으며, 이로 인해 대규모 데이터셋에 대한 필요성이 감소함을 입증했습니다. 논문에서 제안된 방법은 실제 응용 프로그램에서 저자원 언어의 LLM 개발에 중요한 진전을 이룰 것으로 기대됩니다.



### Towards Robust Evaluation of Unlearning in LLMs via Data Transformations (https://arxiv.org/abs/2411.15477)
Comments:
          Accepted at EMNLP 2024 Findings; 21 pages (5 page main content + references + appendix)

- **What's New**: 이번 연구에서는 기존의 Machine Unlearning (MUL) 기술의 강인성을 살펴보며, LLMs가 특정 정보를 잊도록 강제하는 방법에 대해 다룹니다. 연구 팀은 TOFU 데이터셋을 사용하여 다양한 데이터 포맷에서 잊어버리기 성능을 측정하는 중요성을 강조합니다. 특히, 서로 다른 포맷으로 입력이 제공되었을 때 잊어진 정보를 다시 회상할 수 있는지에 대한 실험을 진행하였습니다.

- **Technical Details**: LLMs는 대량의 텍스트 데이터로 훈련되며, 이 과정에서 개인 식별 정보(PII)를 포함한 원하지 않는 정보가 포함될 수 있습니다. 따라서 MUL 기법이 LLM에서 어떤 영향을 미치는지를 조사하며, 잊어버리기 성능이 다양한 데이터 포맷에 따라 어떻게 달라지는지를 탐구합니다. 연구 방법으로는 TOFU 벤치마크를 확장하는 방식으로, 같은 정보가 표현되는 다양한 형식을 도입하고, 이에 대한 평가 지표를 개발합니다.

- **Performance Highlights**: 연구 결과는 서로 다른 데이터 포맷에서 목표 모델(target model)과 잊은 모델(unlearned model) 사이의 성능 격차를 보여줍니다. 이를 통해 다양한 포맷을 고려하는 것이 MUL 알고리즘의 평가에서 신뢰성과 강인함을 확보하는 데 필요하다는 점을 강조합니다. 또한, 이번 연구에서 발표된 새로운 데이터셋과 평가 메트릭스는 MUL 분야의 발전을 도울 것으로 기대합니다.



### HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter (https://arxiv.org/abs/2411.15462)
- **What's New**: 이 논문에서는 온라인 혐오발언 문제를 해결하기 위해 HateDay라는 새로운 글로벌 데이터셋을 소개합니다. 이 데이터셋은 2022년 9월 21일에 게시된 모든 트윗에서 무작위로 샘플링한 240,000개의 트윗으로 구성되어 있으며, 8개 언어와 4개 영어 사용 국가를 포괄합니다. 이를 통해 혐오발언의 분포와 내용이 언어와 국가에 따라 어떻게 변하는지를 보여줍니다.

- **Technical Details**: HateDay 데이터셋은 TwitterDay 데이터셋을 기반으로 하여 8개 주요 언어에 대한 트윗 및 4개 영어 사용 국가(미국, 인도, 나이지리아, 케냐)에서 샘플링됩니다. 모든 트윗은 혐오 발언 감지를 위한 주석이 달리며, 각 트윗은 세 가지 범주(혐오, 공격적, 중립)로 분류됩니다. 주석 작업은 36명의 주석자를 통해 진행되며, 모든 언어에 대해 체계적인 주의사항을 준수하도록 교육받습니다.

- **Performance Highlights**: 연구 결과, 기존 학술 데이터셋에서의 성능 평가가 실제 소셜 미디어 환경에서의 혐오발언 감지 성능을 과대평가하고 있음을 발견했습니다. 실질적인 감지 성능은 특히 비유럽 언어에서 매우 낮으며, 이는 모델이 혐오 발언과 공격 발언을 구분하지 못하는 한계를 포함합니다. 따라서 이러한 낮은 성능은 공공 감지 모델을 통한 혐오발언 조정의 실현 가능성을 낮춘다고 주장하며, 인간 검증을 포함한 조정 방식이 비용 측면에서 비효율적임을 보여줍니다.



### Efficient Ternary Weight Embedding Model: Bridging Scalability and Performanc (https://arxiv.org/abs/2411.15438)
Comments:
          Technical Report

- **What's New**: 이 논문은 임베딩 모델에서 삼원 가중치(ternary-weight) 변환을 위한 혁신적인 미세 조정 프레임워크를 제안합니다. 이를 통해 메모리 사용량을 줄이고 계산 오버헤드를 최소화하며, 고성능을 유지합니다. 사전 훈련된 모델에서 삼원화 작업을 적용하기 위해 자가 학습 지식 증류(self-taught knowledge distillation) 기법을 도입하여 선형 계층의 삼원 가중치를 조정합니다. 이 접근법은 리소스 제약이 있는 환경에서 임베딩 모델을 배포하는 데 적합합니다.

- **Technical Details**: 삼원 가중치 네트워크는 가중치를 -1, 0, +1로 설정하여, 모델의 정보 표현 능력을 향상시키면서도 계산 효율성을 유지합니다. 이는 특히 32비트 부동소수점 연산을 대체할 수 있는 간단한 덧셈 및 뺄셈操作로 대체할 수 있어 논리적입니다. 이러한 구조는 메모리 사용량을 현저히 줄이고 대기 시간(inference latency)을 개선하는데 기여하며, 모델의 강건성을 높이는 데에도 도움을 줍니다.

- **Performance Highlights**: 논문의 실험 결과는 삼원 임베딩 모델이 ANN(Approximate Nearest Neighbor) 검색과 통합될 때, 정확도 및 계산 효율성에서 뛰어난 개선을 이뤄냈음을 보여줍니다. MTEB 벤치마크와 실세계 이미지 데이터셋에 대한 평가에서, 제안된 모델은 32비트 모델과 유사한 성능을 보이면서도 지연 시간과 저장 공간을 획기적으로 줄였습니다. 이로 인해 실시간 추천 시스템에서의 활용 가능성을 높였습니다.



### Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts (https://arxiv.org/abs/2411.15432)
- **What's New**: 본 논문에서는 Lifelong Vision language model Editing을 위한 새로운 프레임워크인 LiveEdit를 제안합니다. 이 프레임워크는 기존의 LLM 편집 기술과 VLLM의 차이를 극복하기 위해 설계되었으며, 특히 지속적 학습 상황에서 VLLMs의 지식 수정을 가능하게 합니다. LiveEdit는 저랭크 전문가( low-rank experts)를 생성하고 이를 하드 및 소프트 라우팅 메커니즘을 통해 결합하여 VLLM의 응답을 조정합니다.

- **Technical Details**: LiveEdit는 두 단계의 라우팅 전략을 활용하여 전문가를 라우팅합니다. 첫 번째 단계에서는 입력 샘플과 관련된 시각적 특징을 추출하여 시각적으로 관련성이 없는 전문가를 필터링합니다. 이 하드 라우팅 단계 이후, 다수의 전문가를 융합하여 입력 쿼리와 편집 텍스트 간의 유사성을 고려한 소프트 라우팅 단계가 진행됩니다. 이를 통해 입력 샘플에 최적화된 전문가 집합을 선택하고 조정된 응답을 생성합니다.

- **Performance Highlights**: LiveEdit는 E-VQA, E-IC, VLKEB 등 다양한 벤치마크 데이터셋에서 1, 10, 100, 1,000개의 편집을 통해 테스트되었습니다. 그 결과, 기존의 강력한 편집기와 비교했을 때 LiveEdit는 뛰어난 성능을 보였으며, 각 모듈 설계의 합리성과 효율성도 검증되었습니다.



### Exploring Large Language Models for Multimodal Sentiment Analysis: Challenges, Benchmarks, and Future Directions (https://arxiv.org/abs/2411.15408)
- **What's New**: 이번 연구에서는 Multimodal Aspect-Based Sentiment Analysis (MABSA) 작업에 대한 대형 언어 모델 (LLMs)의 적합성을 탐구합니다. MABSA는 텍스트와 이미지를 포함한 다중 모달 정보에서 다양한 측면 용어와 그에 따른 감정 폴라리티를 추출하는 것을 목표로 합니다. 우리는 LLM을 활용한 새로운 평가 벤치마크를 설계하였으며, 기존의 지도 학습 방법과 비교하여 그 성능을 평가하였습니다.

- **Technical Details**: MABSA에서 LLM의 활용을 위해 LLM For Sentiment Analysis (LLM4SA) 프레임워크를 설계하였습니다. 이 프레임워크는 텍스트와 시각적 기능을 공동으로 처리하며, 잘 확립된 LLM인 Llama2, ChatGPT 및 LLaVA를 평가 모델로 사용합니다. 또한, 이미지 기능을 추출하기 위해 예측된 비전 변환기 (ViT)를 적용하고, 이를 통해 생성된 시각적 임베딩을 텍스트 기능과 정렬하여 LLM에 통합하는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과, LLM은 다중 모달 이해의 잠재력을 보여주지만 MABSA의 복잡한 요건을 충족하는 데 있어 상당한 도전에 직면하고 있음을 알 수 있었습니다. 특히, LLM들은 SLM 기반 방법보다 더 높은 계산 비용을 요구하여 실용성에 제약이 있는 것으로 나타났습니다. 이러한 결과는 LLM의 현재 한계를 부각시키며, 복잡한 다중 모달 감정 분석 작업에 대한 적응력을 향상시키기 위해 추가적인 최적화가 필요함을 강조합니다.



### ML-SPEAK: A Theory-Guided Machine Learning Method for Studying and Predicting Conversational Turn-taking Patterns (https://arxiv.org/abs/2411.15405)
Comments:
          64 pages, 9 figures

- **What's New**: 이 논문에서는 팀원 성격 특성에 기반하여 팀 동태를 예측하는 문제를 다룹니다. 기존의 Input-Process-Output (IPO) 모델에 비해 팀원 간의 상호작용의 복잡성을 반영하기 위해 더 역동적인 접근 방식을 요구합니다. 저자들은 자기 조직화(self-organized)된 팀 내에서 대화의 턴 테이킹(turn-taking) 패턴을 모델링하여 팀원 성격과 의사소통 동태 사이의 관계를 분석합니다.

- **Technical Details**: 모델은 대화 데이터(conversational data)를 기반으로 훈련되어, 성격 조합에 따라 팀 내 의사소통 패턴(group-wide patterns of communication)을 예측할 수 있습니다. 특히, 내용에 관계없이 턴 테이킹 패턴(turn-taking patterns)을 분석하는 데 중점을 두어, 팀의 emergent states와 결과에 미치는 영향을 객관적으로 측정할 수 있습니다. 이 모델은 시뮬레이션 데이터를 사용하여 성능을 평가하고 실생활 데이터에 적용됩니다.

- **Performance Highlights**: 모델은 베이스라인과 비교했을 때, 말하기 턴 시퀀스를 더욱 정확하게 예측할 수 있으며, 팀원 성격 특성과 의사소통 패턴 간의 새로운 관계를 밝혀낼 수 있습니다. 데이터 기반(data-driven)이고 역동적인 팀 프로세스 이해를 제공함으로써, 팀 편성 및 교육 최적화에 대한 통찰력을 제공합니다.



### From Jack of All Trades to Master of One: Specializing LLM-based Autoraters to a Test S (https://arxiv.org/abs/2411.15387)
- **What's New**: 본 논문에서는 주어진 테스트 세트에 특화된 LLM 기반 Autorater를 구축하는 새로운 방법을 제안합니다. 이 방법은 오직 멀티 샷 프롬프팅(multi-shot prompting)을 사용하여 구성되며, 파인 튜닝(fine-tuning)을 필요로 하지 않습니다. 이를 통해 'Specialist AutoMQM'이라는 최신 자동 메트릭을 만들어 내며, 기존 XCOMET 메트릭에 비해 WMT'23 및 WMT'24 테스트 세트에서 각각 54% 및 119%의 성능 향상을 이루었습니다.

- **Technical Details**: 이 연구에서 제안된 Specialist 방법은 역사적인 평가 결과를 활용하여 인컨텍스트 학습(in-context learning) 예제를 생성합니다. 이 방법은 다양한 ICL 예제 수, LLM 백본 구조, 평가할 시스템 및 평가 작업에 대해 일반화 가능성과 내구성을 검증했습니다. 특히 이 모델은 ICL 예제를 통해 학습된 표현들이 비트리비얼(non-trivial)하며 견고함을 보증합니다.

- **Performance Highlights**: Specialist AutoMQM 모델은 기계 번역(machin translation) 평가에 있어 기존 메트릭을 뛰어넘는 성능을 발휘하여, 세팅된 테스트 세트에서 보다 정확하고 구체적인 평가를 가능하게 합니다. 또한, 이 방법은 단일 평가자(singe rater)에 특화될 수 있어, 여러 평가자의 주관적인 편차 문제를 줄일 수 있는 장점이 있습니다. 모델 성능은 ICL 예제의 수에 따라 향상되며, LLM 선택 및 평가 시스템에 구애받지 않고 안정성을 유지합니다.



### On the Impact of Fine-Tuning on Chain-of-Thought Reasoning (https://arxiv.org/abs/2411.15382)
Comments:
          This paper is a work in progress with findings based on limited evidence. Please exercise discretion when interpreting the findings

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 추론 능력에 대한 파인 튜닝의 영향을 조사합니다. 특히 Chain-of-Thought (CoT) 추론의 품질과 LLMs의 전반적인 추론 기능에 미치는 영향을 중점적으로 다룹니다. 기존 연구에서는 파인 튜닝이 성능을 향상시킬 수 있지만, 추론 능력에 미치는 영향에 대한 조사는 부족했습니다.

- **Technical Details**: 이 논문에서는 CoT 추론을 이용하여 대형 언어 모델의 성능을 분석합니다. CoT는 모델이 문제를 해결하기 위해 단계별로 추론 경로를 생성하도록 유도하는 기법입니다. 연구는 파인 튜닝 전후의 CoT 추론의 정확성과 신뢰성을 비교하며, 다양한 데이터셋에서의 성과를 평가합니다.

- **Performance Highlights**: 연구 결과, LLMs의 파인 튜닝은 전반적으로 CoT 추론 성능을 감소시키며, 특히 작은 모델에서 이 효과가 더 뚜렷하게 나타났습니다. 또한 비추론 데이터셋에서 파인 튜닝한 경우 모델의 신뢰성이 더욱 떨어지는 경향을 보였습니다. 이는 LLMs의 내재적 메커니즘에 변화가 있을 수 있음을 시사합니다.



### Transforming NLU with Babylon: A Case Study in Development of Real-time, Edge-Efficient, Multi-Intent Translation System for Automated Drive-Thru Ordering (https://arxiv.org/abs/2411.15372)
Comments:
          12 pages, 3 figures, 2 tables

- **What's New**: 본 논문에서는 Babylon이라는 새로운 transformer 기반 아키텍처를 소개합니다. Babylon은 자연어 이해(NLU)를 인텐트 번역(task) 문제로 전환하여, 자연어 입력을 인텐트와 슬롯 정보를 인코딩하는 일반 언어 단위인 'transcodes'로 변환합니다. 이러한 접근 방식을 통해 복잡한 다중 인텐트 시나리오 처리가 가능하며, LSTM 기반의 토큰 풀링 메커니즘을 통합하여 사용자 쿼리의 입력 길이를 줄이는 동시에 낮은 지연 시간과 메모리 요구 사항을 최적화합니다.

- **Technical Details**: Babylon 모델은 주요 구성 요소로 자동 음성 인식(ASR), 자연어 이해(NLU), 대화 관리(Dialog Management), 및 텍스트-음성 변환(TTS) 합성을 갖추고 있습니다. NLU는 음성을 실시간으로 전사하여 생성된 음소(phoneme)를 일반 언어 단위인 'transcodes'로 변환하는 기능을 수행합니다. 이 모델은 여러 사용자 쿼리를 단일 대화 턴으로 처리할 수 있는 능력을 가지며, 이를 통해 Transform의 계산 부하를 줄이고 ASR 오류를 완화합니다.

- **Performance Highlights**: 일련의 실험 결과, Babylon은 Flan-T5 및 BART와 같은 전통적인 NMT 모델 대비 정확도-지연 시간-메모리 풋프린트 수준에서 유의미한 성능 향상을 보여주었습니다. 특히, Babaylon 모델은 단일 CPU 코어에서 작동하며, 현실 세계의 배포에서 실용성을 입증하였습니다. 이러한 최적화는 모든 지연 시간 민감 애플리케이션에 필수적인 요소로, 드라이브 스루 시스템과 같은 환경에서 큰 장점을 제공합니다.



### PPLqa: An Unsupervised Information-Theoretic Quality Metric for Comparing Generative Large Language Models (https://arxiv.org/abs/2411.15320)
- **What's New**: 이번 논문에서는 PPLqa라는 새로운 정보 이론 기반의 평가 지표를 제안합니다. 이 지표는 인간의 주관적인 평가 없이도 생성된 응답의 품질을 평가할 수 있으며, 다양한 언어에 독립적이고 쉽게 계산할 수 있는 장점을 가지고 있습니다. PPLqa는 생성된 언어 모델의 응답을 효율적으로 비교할 수 있게 해주며, 이는 기존 지표들보다 더 나은 성과를 보입니다.

- **Technical Details**: PPLqa는 정보 이론적인 접근을 통해 프롬프트와 응답 쌍의 구조를 정량화합니다. 이는 응답의 일관성(coherence), 유창성(fluency), 적합성(relevance), 일관성(consistency)을 모두 포함하면서도 기존의 여러 메트릭에 비해 더 직관적입니다. 실험에서는 다양한 주제 영역에서 LLM으로부터 생성된 응답을 평가하여 PPLqa가 인간 또는 다른 LLM의 평가 결과와 높은 상관관계를 나타낸다는 것을 보여주었습니다.

- **Performance Highlights**: PPLqa는 긴 형식의 질의응답(Q&A)에서 다른 관련 지표들과 비슷한 성과를 보이며, MT-Bench 데이터셋에서도 우수성을 입증했습니다. 특히, 인간 평가와의 상관관계가 뛰어나고, Anthropic의 Claude 3 LLM 평가와 비교했을 때 더 강한 일관성을 보여줍니다. 이를 통해 PPLqa가 다양한 상황에서 신뢰할 수 있는 평가 도구임을 확인할 수 있습니다.



### Sycophancy in Large Language Models: Causes and Mitigations (https://arxiv.org/abs/2411.15287)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 보이는 아첨적(sycophantic) 행동의 원인, 영향, 완화 전략에 대한 기술적 조사 결과를 제시합니다. 기존 연구를 바탕으로 아첨적 경향을 측정하고 정량화하는 방법에 대한 논의가 포함되어 있으며, 이는 모델 성능을 유지하면서 아첨적 행동을 줄이기 위한 전략을 평가하는 데 기여합니다.

- **Technical Details**: 대형 언어 모델은 텍스트 데이터를 기반으로 다음 토큰을 예측하는 신경망(neural network)입니다. 아첨적 행동을 측정하는 방법으로는 여러 가지가 있으며, 정확도(accuracy), 동의율(agreement rate), 뒤집기율(flip rate)과 같은 지표가 사용됩니다. 인간 평가(human evaluation)와 자동화된 메트릭을 통해 모델의 아첨적 경향을 정량화하며, 적대적 테스트(adversarial testing)를 통해 모델의 잠재적 취약성을 드러내는 방법도 모색합니다.

- **Performance Highlights**: 논문은 아첨적 행동이 LLM의 신뢰성과 윤리적 적용에 미치는 영향에 대해 분석하고 있으며, 정보의 사실 정확성을 보장하고 사용자 신뢰를 유지하는 데 있어 아첨적 행동을 완화하는 것이 중요하다는 점을 강조합니다. 다양한 완화 기술(techniques)인 훈련 데이터 개선, 새로운 세부 조정 방법(novel fine-tuning methods), 배포 후 제어 메커니즘(post-deployment control mechanisms)을 평가하며, 아첨적 행동을 줄이면서도 모델 성능을 유지하기 위한 경로를 제시합니다.



### BanglaEmbed: Efficient Sentence Embedding Models for a Low-Resource Language Using Cross-Lingual Distillation Techniques (https://arxiv.org/abs/2411.15270)
Comments:
          Accepted in ACAI 2024

- **What's New**: 본 연구에서는 방글라어(Bengali)와 같은 저자원 언어를 위한 두 가지 경량 문장 변환기를 제안하였습니다. 이 모델은 최신의 cross-lingual knowledge distillation 방식을 활용하여 사전 학습된 영어 문장 변환기로부터 지식을 증류하는 방법을 사용합니다. 제안된 모델은 여러 다운스트림 작업에서 평가되어 기존 방글라 문장 변환기보다 뛰어난 성능을 보였습니다.

- **Technical Details**: 이 연구는 영어-방글라 기계 번역 데이터셋을 활용하여 방글라어 문장 변환기를 훈련하는 접근 방식을 소개합니다. 두 가지 손실 함수인 mean squared error (MSE)와 multiple negatives ranking loss를 적용하였으며, 사전 학습된 영어 문장 변환기를 teacher 모델로 사용하여 방글라어의 lightweight student 모델을 훈련합니다. 이 과정에서 방글라어 문장과 해당 영어 번역이 동일한 맥락 의미를 전달하기 때문에 서로 같은 임베딩 공간으로 매핑될 수 있습니다.

- **Performance Highlights**: 제안된 BanglaEmbed-MSE 모델은 패러프레이즈 탐지(paraphrase detection) 작업에서 기존의 방글라 문장 변환기보다 성능이 우수하며, 계산 자원도 적게 소모합니다. 또한, 가벼운 아키텍처와 짧은 추론 시간은 자원이 제한된 환경에서의 배포에 매우 적합합니다. 이로 인해 저자원 언어에 대한 실제 NLP 애플리케이션에 유용한 가치를 제공합니다.



### Graph Neural Network-Based Entity Extraction and Relationship Reasoning in Complex Knowledge Graphs (https://arxiv.org/abs/2411.15195)
- **What's New**: 이번 연구에서는 그래프 신경망(Graph Neural Network)을 기반으로 하는 지식 그래프(Knowledge Graph) 엔티티 추출과 관계 추론 알고리즘을 제안하였습니다. 그래프 합성곱 네트워크(Graph Convolutional Network)와 그래프 주의 네트워크(Graph Attention Network)를 사용하여 지식 그래프의 복잡한 구조를 모델링하였습니다. 이 논문은 엔드 투 엔드(End-to-End) 공동 모델을 구축하여 엔티티와 관계의 효율적인 인식 및 추론을 달성했습니다.

- **Technical Details**: 제안된 모델은 다양한 딥러닝 알고리즘과 비교되어 AUC(Area Under Curve), Recall Rate, Precision Rate, F1 Value 등의 지표를 통해 우수성을 검증받았습니다. 실험 결과는 특히 복잡한 지식 그래프에 대한 강한 일반화 능력과 안정성을 보여주었습니다. 이는 지식 그래프에 대한 추가 연구에 강력한 지원을 제공하며, 그래프 신경망의 엔티티 추출 및 관계 추론에서의 응용 가능성을 입증합니다.

- **Performance Highlights**: 제안된 모델은 모든 지표에서 뛰어난 성능을 보였으며, 특히 복잡한 지식 그래프에서 더 강력한 일반화와 안정성을 나타냈습니다. 이러한 성능은 실제 상황에서의 응용 가능성을 더욱 높여 줍니다. 따라서, 그래프 신경망의 활용 범위를 넓히는 데 이 연구가 기여할 것으로 기대됩니다.



### Can Open-source LLMs Enhance Data Augmentation for Toxic Detection?: An Experimental Study (https://arxiv.org/abs/2411.15175)
- **What's New**: 이번 연구에서는 오픈 소스 LLMs(대규모 언어 모델)에 대한 프롬프트 엔지니어링과 파인튜닝 기술을 통해 유해 데이터 증강의 효과를 높이는 방법을 탐구하였습니다. 특히, 모델들이 생성하는 유해 데이터의 품질과 다양성을 현저히 개선하는 데 성공했습니다. Mistral 모델이 최소한의 환각(hallucination)으로 유해 데이터를 생성할 수 있음을 발견했습니다. 또한, 파인튜닝이 데이터 품질 향상에 효과적이지만, 데이터 중복 및 과적합(overfitting) 문제는 여전히 존재합니다.

- **Technical Details**: 방법론은 두 단계로 구성되며, 첫 번째 단계는 프롬프트 엔지니어링을 통해 유해 데이터를 생성하기 위한 프롬프트를 설계하는 것입니다. 요구 사항을 명확히 하고 다양한 샘플 예제를 포함하는 프롬프트 템플릿을 개발했습니다. 그러나 모델의 내부 안전 정렬로 인해 유해 데이터 품질과 다양성이 저하되는 문제에 직면하게 되었습니다. 그래서 파인튜닝을 통해 모델의 가중치를 업데이트하여 유해 데이터 생성 능력을 향상시키기로 결정했습니다.

- **Performance Highlights**: 우리는 다양한 오픈 소스 LLMs를 평가하여 Mistral 모델이 기존 기준선 모델들과 비교하여 최상의 성능을 보여준다는 것을 확인했습니다. 실험 결과, 파인튜닝된 모델은 유해 데이터 생성의 품질과 다양성을 크게 개선했습니다. 이 연구는 향후 유해 물질 탐지 시스템의 발전 방향에 대한 통찰력을 제공하며, 자동화된 콘텐츠 모더레이션 도구의 효과성을 실질적으로 향상시키는 데 기여할 것으로 기대됩니다.



### DreamRunner: Fine-Grained Storytelling Video Generation with Retrieval-Augmented Motion Adaptation (https://arxiv.org/abs/2411.16657)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 DreamRunner라는 새로운 스토리 비디오 생성 방법을 제안합니다. 이 방법은 길고 멀티 모션, 멀티 씬 비디오를 생성할 수 있으며, 입력 텍스트 스크립트에서 설명된 스토리를 일관되게 표현하는데 중점을 둡니다. DreamRunner는 LLM(대규모 언어 모델)을 사용해 입력 스크립트를 구조화하고, 리트리벌(검색) 기반의 적응 방법을 통해 비디오 생성 과정에서 동작을 커스터마이즈(purpose)할 수 있습니다.

- **Technical Details**: DreamRunner 프레임워크는 세 가지 주요 프로세스로 구성됩니다: 1) 이중 수준 비디오 계획 생성, 2) 동작 리트리벌 및 주제/동작 사전 학습, 3) 공간-시간 기반의 3D 주의 및 사전 주입 모듈(SR3AI). 첫 번째 단계에서는 사용자 제공의 스토리 내러티브를 기반으로 고수준 및 세부적인 계획을 수립하며, 두 번째 단계에서는 비디오 데이터베이스에서 동작과 관련된 비디오를 검색하여 동작 사전을 학습합니다. 마지막 단계에서는 상세한 프레임 제어와 원활한 동작 전환을 가능하게 하는 SR3AI 모듈을 도입합니다.

- **Performance Highlights**: DreamRunner는 T2V-ComBench에서 기존의 최첨단 방식들에 비해 캐릭터 일관성(CLIP 점수)에서 13.1%의 상대적 개선을 보이고, 텍스트 추적 능력(ViCLIP 점수)에서도 8.56%의 향상을 기록했습니다. 또한, 동일 씬 내에서의 사건 전환의 매끄러움(DINO 점수)에서도 27.2% 개선을 보이며 효과성을 입증했습니다. DreamRunner는 오픈 소스 모델을 기반으로 하면서도 폐쇄형 모델에 비해 동적 속성 바인딩에서 최고의 성과를 달성했고, 캐릭터 상호작용에서도 경쟁력 있는 결과를 보여주며 오픈 소스의 가능성을 보여주고 있습니다.



### Preventing Jailbreak Prompts as Malicious Tools for Cybercriminals: A Cyber Defense Perspectiv (https://arxiv.org/abs/2411.16642)
- **What's New**: 이 논문은 AI와 사이버 보안에서 중요한 위협인 'jailbreak prompts'를 분석합니다. 이러한 프롬프트는 대규모 언어 모델의 윤리적 보호 장치를 우회하도록 만들어져 사이버 범죄자들에 의해 악용될 수 있는 가능성을 가지고 있습니다. 저자들은 프롬프트 주입 및 컨텍스트 조작과 같은 기술을 탐구하여 유해한 콘텐츠 생성 및 민감한 정보 추출을 가능하게 합니다.

- **Technical Details**: 논문에서는 'jailbreak prompts'의 본질과 이들이 악용될 수 있는 잠재력을 분석하고, 그 위험을 완화하기 위한 다층 방어 전략을 제안합니다. 예를 들어, 여러 사례研究를 통해 1회성 방어를 우회하는 멀티턴 공격 프롬프트의 사용 및 생물무기 합성을 위한 LLM의 악용 취약성을 조명합니다. 저자들은 키워드 탐지 및 컨텍스트 인식 필터링과 같은 기술로 적대적인 사용을 저지할 수 있음을 강조합니다.

- **Performance Highlights**: 제안된 방어 접근 방식은 프롬프트 수준의 필터링, 모델 수준의 메커니즘, 그리고 사고 방지 전략을 포함합니다. 이 논문은 사용자가 겪는 보안 위험을 줄이기 위한 지속적인 모델 미세 조정 및 동적 안전 프로토콜의 필요성을 강조합니다. 효과적인 방어 조치를 마련하기 위해 AI 연구자, 사이버 보안 전문가, 정책 입안자 간의 협력을 강조하고 있습니다.



### From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judg (https://arxiv.org/abs/2411.16594)
Comments:
          32 pages, 5 figures

- **What's New**: 본 논문은 LLM(대형 언어 모델)을 활용하여 평가 및 판단을 수행하는 새로운 패러다임, 즉 'LLM-as-a-judge'에 대해 포괄적으로 조사합니다. LLM을 통해 다양한 작업과 응용 프로그램에서 점수 매기기, 순위 매기기, 선택을 수행할 수 있는 가능성을 제시합니다. 특히, 입력과 출력 관점에서의 정의를 상세하게 설명합니다.

- **Technical Details**: 논문은 평가의 잠재력과 한계를 이해하기 위해 LLM-as-a-judge의 세 가지 차원, 즉 무엇을 판단하는가(what to judge), 어떻게 판단하는가(how to judge), 어디서 판단하는가(where to judge)를 제시합니다. 그러한 평가를 위한 주요 기준과 방법론을 분류하고, 각 기준에 대해 다양한 접근법과 기술적 내용을 다룹니다. 또한, 인공지능 모델들이 자동으로 좋아요 및 차별적 특성을 평가할 수 있는 방법론적인 기초를 마련합니다.

- **Performance Highlights**: 평가를 위해 기존의 여러 모델과 접근 방법들을 정리하고, 현재 LLM을 통한 평가는 유망하지만 여전히 여러 문제와 도전에 직면해 있음을 강조합니다. 향후 연구 방향으로는 LLM의 상용화 및 평가 방법의 개선이 필요하다고 제시하며, 이를 통해 AI와 NLP 분야에서의 혁신을 촉진할 수 있는 가능성을 제공합니다.



### EnStack: An Ensemble Stacking Framework of Large Language Models for Enhanced Vulnerability Detection in Source Cod (https://arxiv.org/abs/2411.16561)
Comments:
          Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData 2024)

- **What's New**: 본 논문에서는 자연어 처리(NLP) 기술을 활용하여 소프트웨어 취약점을 탐지하는 새로운 앙상블 스태킹 프레임워크인 EnStack을 소개합니다. EnStack은 코드 이해에 특화된 여러 개의 사전 학습된 대형 언어 모델(LLM)인 CodeBERT, GraphCodeBERT, UniXcoder를 결합하여 복잡한 코드 패턴과 취약점을 효과적으로 인식합니다. 기존 방법들과 비교할 때, EnStack은 정확도, 정밀도, 재현율, F1 점수에서 현저한 개선을 보여줍니다.

- **Technical Details**: EnStack은 CodeBERT의 의미적 분석, GraphCodeBERT의 구조적 표현 및 UniXcoder의 크로스 모달 능력을 통합하여 취약점 탐지의 정밀도를 높이고자 합니다. 각 모델은 Draper VDISC 데이터셋에서 개별적으로 미세 조정되며, 로지스틱 회귀(Logistic Regression), 서포트 벡터 머신(SVM), 랜덤 포레스트(Random Forest), XGBoost와 같은 메타 분류기를 통해 통합됩니다. 이 방식은 각 모델의 고유한 강점을 활용하여 소스 코드 내 취약점을 더 정교하게 식별할 수 있는 시스템을 개발합니다.

- **Performance Highlights**: 실험 결과는 EnStack이 기존 방법보다 우수한 성능을 발휘하여 더 미세하고 복잡한 취약점을 효과적으로 탐지함을 입증합니다. 본 프레임워크는 다양한 프로그래밍 환경에서의 취약점 탐지에 있어 포괄적인 모델을 제공하며, 향후 자동화된 취약점 탐지 기술 발전에 중요한 기초자료를 제공합니다. 특히, 메타 분류기가 어떻게 기본 모델의 예측을 통합하는지를 연구함으로써 모델 앙상블 전략에 대한 통찰력을 제공합니다.



### RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics (https://arxiv.org/abs/2411.16537)
- **What's New**: 이 논문은 로봇 분야에서의 공간 이해 능력 향상을 위해 RoboSpatial이라는 대규모 데이터셋을 소개합니다. 이 데이터셋은 3D 스캔과 에고세닉(ego-centric) 이미지로 구성된 실내 및 테이블 씬을 기반으로 하며, 로봇과 관련된 풍부한 공간 정보를 포함하고 있습니다. RoboSpatial은 100만 개의 이미지, 5,000개의 3D 스캔, 300만 개의 주석 공간 관계를 포함하여 로봇이 더 잘 이해할 수 있도록 다양한 질문-답변 쌍을 제공합니다.

- **Technical Details**: RoboSpatial 데이터셋은 세 가지 유형의 질문을 제공합니다: (1) 공간 구성, (2) 공간 맥락, (3) 공간 호환성. 각 질문 유형은 서로 다른 관점에서 공간 관계를 이해하도록 설계되어 있으며, 관점은 에고세닉, 객체 중심(object-centric), 세계 중심(world-centric)으로 나뉩니다. 이러한 다각적인 접근 방식은 로봇이 복잡한 공간 지침을 더욱 유연하게 처리할 수 있도록 합니다.

- **Performance Highlights**: RoboSpatial을 사용하여 훈련된 비전-언어 모델(VLM)은 기존 모델보다 공간 추론 능력이 상당히 향상되었습니다. 실험 결과, RoboSpatial에서 학습한 모델은 다양한 로봇 조작 작업 및 실내 씬 질문 응답에서 우수한 성능을 보였습니다. 이 데이터셋의 3D 준비 디자인은 VLM의 공간 추론 능력을 높이는 데 기여하며, 실제 공간 작업에서의 차별적인 성능을 보여줍니다.



### Fundamental Limits of Prompt Tuning Transformers: Universality, Capacity and Efficiency (https://arxiv.org/abs/2411.16525)
- **What's New**: 이번 연구에서는 transformer 기반의 foundation 모델을 위한 prompt tuning의 통계적 및 계산적 한계를 조사하였습니다. 주요 기여로는 단일 헤드(single-head) transformer에 단일 self-attention layer만을 사용하는 prompt tuning 이론을 제시하는 것입니다. 이 과정에서 우리는 이 방식이 보편적(universal)이며, Strong Exponential Time Hypothesis (SETH) 하에서 효율적인 알고리즘을 지원한다고 주장합니다.

- **Technical Details**: 우리는 단순한 transformer에서의 prompt tuning이 시퀀스-투-시퀀스 Lipschitz 함수에 대한 보편적 근사기(universal approximators)임을 증명하였습니다. 또한 1-layer 및 1-head transformer로 데이터셋을 기억하기 위해 필요한 soft-prompt 토큰에 대한 하한선(lower bound)을 제공했습니다. 이를 통해 prompt tuning의 효율성에서의 phase transition을 규명하고, soft-prompt-induced 키(key)와 쿼리(query)의 노름(norm)에 의해 결정된 최대 경계 조건을 설정했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 설정한 경계 조건을 초과하는 경우 SETH 하에서 효율적인 알고리즘이 존재하지 않음을 보여주고 있습니다. 그 경계 조건 내에서 우리는 거의 선형 시간(almost-linear time) prompt tuning 추론 알고리즘의 존재를 입증하여 이론을 구체화하였습니다. 이러한 기본 한계는 실무자를 위한 표현력 높고 효율적인 prompt tuning 방법 설계 시 필수적인 조건을 제공한다는 점에서 중요합니다.



### LaB-RAG: Label Boosted Retrieval Augmented Generation for Radiology Report Generation (https://arxiv.org/abs/2411.16523)
- **What's New**: 이번 논문에서는 LaB-RAG(Label Boosted Retrieval Augmented Generation)라는 새로운 이미지 캡셔닝 방법을 제안했습니다. 이 방법은 고차원 latent features를 대신하여 카테고리 레이블을 사용하여 이미지 설명을 반영함으로써, 표준 retrieval augmented generation을 향상시키는 데 초점을 맞추고 있습니다. 특히, 이 접근법은 의료 영상의 보고서 생성을 위한 연구에 중점을 두고 있습니다.

- **Technical Details**: LaB-RAG는 간단한 선형 분류기를 사용하여 이미지에서 카테고리 레이블을 추출하고, 이를 기반으로 일반적인 LLM(여기서는 대규모 언어 모델)과 결합하여 라디오로지 보고서를 생성합니다. 이 방법은 고급 딥러닝 모델의 훈련 없이, 기존 모델의 frozen 상태에서 작동합니다. 따라서, LaB-RAG는 어떤 이미지 피처 인코더 모델을 훈련시키지 않으면서도 경쟁력 있는 성과를 나타냅니다.

- **Performance Highlights**: LaB-RAG는 기존의 retrieval 기반 RRG 방법들보다 더 나은 성능을 보여주었고, 다른 세밀하게 조정된 시각-언어 RRG 모델들과도 비교할 만한 결과를 얻었습니다. 또한, 다양한 구성 요소에 대한 실험을 통해 이 방법의 효과를 분석했습니다. 최종적으로, 일반적으로 사용되는 RRG 메트릭의 문제점을 비판하면서, LaB-RAG를 사용하여 결과를 인위적으로 부풀릴 수 있는 가능성을 언급했습니다.



### All Languages Matter: Evaluating LMMs on Culturally Diverse 100 Languages (https://arxiv.org/abs/2411.16508)
Comments:
          A Multilingual Multimodal cultural benchmark for 100 languages

- **What's New**: 이번 연구에서 발표된 All Languages Matter Benchmark (ALM-bench)는 100개 언어에 걸쳐 대규모 다중모달 모델(LMM)을 평가하기 위한 포괄적이고 문화적으로 다양한 벤치마크입니다. ALM-bench는 기존 모델들이 다양한 문화적 맥락을 이해하는 능력을 시험하며, 부족 자원 언어(low-resource languages)에 대한 평가를 포함합니다. 이 벤치마크는 정교한 질문 형식을 제공하여 다양한 시각적 및 언어적 추론에서 모델의 능력을 검증합니다.

- **Technical Details**: ALM-bench는 22,763개의 질문 및 답변을 포함하며, 19개의 일반 및 문화 특정 도메인에서 다양한 유형의 질문을 제공합니다. 이 데이터셋은 73개국의 문화적 뉘앙스를 포착하며, 15개의 언어 계통과 24개의 언어 스크립트를 포함하고 있습니다. 기존 벤치마크들과 비교할 때 ALM-bench는 3배 더 많은 언어와 다양한 질문 형식을 제공하여 다문화 및 다언어 환경에서 LMM의 성능을 종합적으로 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 16개의 최신 LMM을 대상으로 한 평가 결과, 그들이 부족 자원 언어 및 문화에 대한 이해에서 상당한 성능 격차가 있음을 보여주었습니다. 특히, GPT-4o는 최고의 오픈소스 모델인 GLM-4V보다 27% 더 뛰어난 성능을 보였습니다. ALM-bench를 통해 저자원 지역인 동남아시아 및 서부 아프리카의 이해를 개선할 필요성을 강조했습니다.



### Unraveling Arithmetic in Large Language Models: The Role of Algebraic Structures (https://arxiv.org/abs/2411.16260)
- **What's New**: 이 논문은 대형 언어 모델(LLM)이 복잡한 산술 문제를 해결하는 과정에서 대수 구조를 학습한다는 새로운 관점을 제시합니다. LLM은 Commutativity(교환 법칙)와 Identity(항등 법칙)와 같은 대수적 속성을 통해 학습하며, 이는 입력과 출력 간의 관계를 통해 관찰할 수 있습니다. 이를 통해 LLM의 산술 성능을 향상시킬 수 있는 통찰을 제공합니다.

- **Technical Details**: 저자들은 LLM이 숫자와 연산 기호만을 사용하여 표현된 산술 문제를 푸는 데 집중하고 있습니다. 연구의 초점은 유한 가벨 군 내에서의 산술 연산에 두어지며, 유한 가벨 군의 정의를 제시합니다. 주어진 원소들에 대해 닫힘, 결합법칙, 교환법칙, 그리고 항등원을 정의함으로써 이론적인 기초를 마련합니다.

- **Performance Highlights**: 실험 결과, LLM이 훈련 데이터에서 대수적 구조를 학습하고 이를 통해 미지의 데이터를 일반화하여 산술 문제를 해결할 수 있음을 보여줍니다. 연구진은 Commutativity와 Identity 속성을 훈련 세트에서 학습하고 이를 테스트 세트에 일반화했음을 입증합니다. 이러한 발견은 알고리즘 설계를 개선할 수 있는 귀중한 통찰을 제공합니다.



### Video-Text Dataset Construction from Multi-AI Feedback: Promoting Weak-to-Strong Preference Learning for Video Large Language Models (https://arxiv.org/abs/2411.16201)
- **What's New**: 이 논문에서는 Multimodal Large Language Models (MLLMs) alignment에 필요한 고품질의 비디오-텍스트 선호 데이터 세트를 제안합니다. 기존의 비디오 질문-응답(VQA) 선호 데이터는 수집이 어렵고 비용이 많이 들며, 수동 주석이 신뢰성이 떨어지는 문제가 있습니다. 오히려 AI-generated 응답은 온도 조절에 의해 다양성이 부족해 선호 학습을 저해합니다. 이를 해결하기 위해 우리는 Multiple Multimodal Artificial Intelligence Preference Datasets in VQA (MMAIP-V)라는 새로운 데이터 세트를 구축했습니다.

- **Technical Details**: MMAIP-V는 응답 분포 집합에서 샘플링하고, 외부 스코어링 기능을 사용하여 응답 품질을 평가하여 구성된 고품질 VQA 선호 데이터 세트입니다. 또한, Iterative Weak-to-Strong Reinforcement Learning from AI Feedback (Iter-W2S-RLAIF)라는 프레임워크를 통해 참조 모델을 점진적으로 업데이트하며 MLLMs의 alignment 능력을 향상시킵니다. 이러한 접근방식은 선호 데이터의 잠재력을 최대한 활용하고, 모델의 성능을 개선하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, MMAIP-V의 높은 품질의 긍정적 응답과 다양성 있는 부정적 응답이 MLLMs의 선호 학습에 유익하다는 것이 입증되었습니다. 또한, Iter-W2S-RLAIF는 MMAIP-V에 내재된 AI 피드백을 효과적으로 활용하여 VQA 생성 능력을 향상시킵니다. 마지막으로, 비전 기반의 공정한 평가 방식을 도입하여 이전 평가의 편향을 최소화하고 비전 정보를 포함한 다각적 평가를 수행합니다.



### Enhancing Multi-Agent Consensus through Third-Party LLM Integration: Analyzing Uncertainty and Mitigating Hallucinations in Large Language Models (https://arxiv.org/abs/2411.16189)
- **What's New**: 이 논문에서는 복잡한 추론 작업을 수행할 때 대형 언어 모델(LLMs)이 직면하는 문제인 환각(hallucination)을 완화하기 위한 새로운 방법을 제안합니다. 이 방법은 여러 LLM을 통합하여 지식의 경계를 확장하고, 특정 모델에 대한 의존도를 줄이며, 에이전트들 간의 심층적인 논의를 촉진하는 데 초점을 맞추고 있습니다. 실험 결과, 제안한 방법이 기존의 다중 에이전트 기반선보다 우수함을 입증하였습니다.

- **Technical Details**: 연구에서 제안된 방법론은 세 번째 대형 언어 모델(LLM)을 도입하여 다중 에이전트 시스템에서 불확실성 추정과 신뢰도 분석을 통해 주의(attention) 가중치를 조절합니다. 이 과정에서 각 에이전트는 다른 에이전트의 응답을 학습하여 합의(consensus) 형성을 최적화합니다. 또한, 대화의 정보를 정확하게 파악하기 위해 에이전트 간의 상호 작용을 기반으로 하는 미세한 추론 방법이 개발되었습니다.

- **Performance Highlights**: 제안된 연구는 산술 데이터셋(arithmetic dataset)에 대한 실험으로, 높은 신뢰도를 가진 에이전트에게 더 많은 비중을 두어 승률 기반의 답변을 도출하는 방식으로 성과를 달성하였습니다. 결과적으로, 제안된 방법은 기존의 다중 에이전트 시스템에서 나타나는 제한 사항을 극복하고, 고차원적인 문제 해결에 도움을 주어 LLM의 적용범위를 확장할 수 있습니다.



### Adaptive Circuit Behavior and Generalization in Mechanistic Interpretability (https://arxiv.org/abs/2411.16105)
Comments:
          10 pages, 8 figures

- **What's New**: 이 논문은 Mechanistic interpretability의 새로운 관점을 제시하며, GPT-2 small에서 간접 목적어 식별(I/OI) 회로의 일반성을 조사합니다. 저자들은 다양한 프롬프트 변형에서도 동일한 회로가 작동하는지를 확인하며, 이 회로가 기존 알고리즘의 전제 조건을 어떻게 극복하는지를 탐구합니다. 특히, S2 Hacking이라는 새로운 메커니즘을 발견하여 기존 알고리즘의 실패에도 불구하고 회로가 잘 작동하는 원인을 설명합니다.

- **Technical Details**: 간접 목적어 식별(I/OI) 알고리즘은 이전 이름을 식별하고 중복된 이름을 제거하여 최종적으로 남은 이름을 출력하는 세 단계의 프로세스를 포함합니다. 이 알고리즘은 특정 프롬프트 구조에 거의 무관하다고 여겨지지만, 중복된 이름이 사용되는 프롬프트에서는 원래 알고리즘이 실패해야 합니다. 이에 반해, 저자들은 DoubleIO 및 TripleIO와 같은 새로운 프롬프트 변형에서도 이 회로가 여전히 유효하다는 것을 발견하고, IOI 회로의 다양한 기능을 이해하는 데 중요한 통찰을 제공합니다.

- **Performance Highlights**: IOI 회로는 IOI 알고리즘이 완전히 실패할 것으로 예상되는 프롬프트 변형에서 전체 모델보다 우수한 성능을 발휘했습니다. 모든 구성 요소를 재사용하면서 추가 입력 엣지만 추가하는 방식으로 회로가 잘 일반화된다는 것을 발견했습니다. 제시된 결과는 대규모 신경망의 일반적인 능력을 이해하는 데 중요한 진전을 나타내며, Mechanistic interpretability의 가능성을 더욱 부각시킵니다.



### Cautious Optimizers: Improving Training with One Line of Cod (https://arxiv.org/abs/2411.16085)
- **What's New**: 본 논문에서는 transformer 사전학습을 위한 기본 최적화 알고리즘인 AdamW에 대한 새로운 접근 방식을 제안합니다. 기존의 여러 최적화 알고리즘보다 더 빠르고 안정적인 성능을 추구하는 과정에서 기존의 모멘텀 기반 최적화 알고리즘에 단 하나의 간단한 수정만으로 새로운 'Cautious Optimizer'를 만들어냈습니다. 이 수정된 최적화 알고리즘은 C-AdamW 및 C-Lion과 같은 형태로 나타납니다.

- **Technical Details**: 이론적인 결과에 따르면, 제안된 수정은 Adam의 해밀토니안 함수(Hamiltonian function)를 보존하며, Lyapunov 분석 하에서도 수렴 보장을 깨뜨리지 않습니다. 이러한 이론적 통찰을 바탕으로 새로운 최적화 알고리즘군을 밝혀내었으며, 이 중 가장 단순한 형태의 알고리즘을 선택하여 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, Llama와 MAE 사전 학습에서 최대 1.47배 더 빠른 속도를 기록하는 것으로 나타났습니다. 제안된 알고리즘은 기존 기법들에 비해 뚜렷한 성능 향상을 보였으며, 해당 코드는 GitHub와 같은 플랫폼에서 공개될 예정입니다.



### Predicting Emergent Capabilities by Finetuning (https://arxiv.org/abs/2411.16035)
- **What's New**: 이번 논문에서는 LLM(대규모 언어 모델)의 발전에 있어 'emergent capabilities'의 예측 문제를 다룹니다. 언어 모델 사전 훈련 손실(pretraining loss)은 예측 가능한 패턴을 따르지만, 후속 작업에서의 성능은 예측하기 어려운 특성을 지니고 있습니다. 저자들은 현재 LLM을 기반으로 향후 모델의 성능을 예측할 수 있는 간단한 통찰력을 발견하였으며, 이를 'emergence prediction'이라고 합니다.

- **Technical Details**: 연구팀은 LLM을 특정 작업에 맞게 파인튜닝(finetuning)할 때 발생하는 'emergence' 현상을 연구하였습니다. 모델의 사전 훈련 손실을 통해 'emergence point'를 정의하고, 이 시점이 얼마나 많은 파인튜닝 데이터로 shift되는지 분석합니다. 이 과정을 통해 'emergence laws'라는 파라메트릭 함수로 emergence 지점을 예측할 수 있는 방법을 개발하였습니다.

- **Performance Highlights**: 제안한 접근 방식을 MMLU, GSM8K, CommonsenseQA, CoLA와 같은 표준 NLP 벤치마크를 사용하여 검증하였습니다. 작은 규모의 모델을 사용하여도 향후 더 강력한 모델의 emergence 발생 시점을 정확히 예측할 수 있음을 보였습니다. 이를 통해 파인튜닝 없이도 모델의 성능을 평가할 수 있는 새로운 방법을 제시하고 있습니다.



### Kleene algebra with commutativity conditions is undecidab (https://arxiv.org/abs/2411.15979)
Comments:
          Published at CSL 2025

- **What's New**: 이번 연구는 Kleene 대수에서 원자 항목에 대한 교환 가능성 조건이 있는 경우의 방정식 이론이 결정을 내릴 수 없음을 입증하여, Kleene 대수 이론에서 오랜 시간을 기다려온 문제를 해결했습니다. 이 연구는 또한 Kuznetsov에 의해 최근 독립적으로 해결된 결과와 일치하지만, 우리의 결과는 Kleene 대수의 귀납 공리(induction axioms)를 지원하지 않는 약한 이론에 대해서도 유효합니다.

- **Technical Details**: Kleene 대수는 정규 언어의 대수 구조를 일반화한 것으로, 유용한 성질들을 보존하고 있습니다. 다양한 적용을 위해 Kleene 대수에 교환 가능성 조건을 추가하는 경우가 빈번하지만, 이는 결정 가능성에 문제를 일으킬 수 있습니다. 특히, 이러한 조건이 포함된 경우 두 정규 언어의 동등성 여부를 결정하는 문제는 Π10-complete로, 이는 정지 문제의 여집합과 동등합니다.

- **Performance Highlights**: 이 연구에서는 주어진 기계 M과 입력 x에 대해 Kleene 대수 항 사이의 불평등을 정의하여, M이 x에서 수용할 때 이 불평등이 성립하고, 반대로 M이 거부할 때 성립하지 않음을 보여주었습니다. 이러한 불평등이 결정 가능하다면 두 시나리오를 컴퓨터적으로 구분할 수 있지만, 이는 불가능하다는 것을 입증하였습니다.



### TableTime: Reformulating Time Series Classification as Zero-Shot Table Understanding via Large Language Models (https://arxiv.org/abs/2411.15737)
- **What's New**: 이 논문에서는 다변량 시계열 분류(MTSC)를 위한 새로운 접근법인 TableTime을 제안합니다. TableTime은 MTSC 문제를 표 이해(task) 문제로 재구성하여 정보 손실을 최소화하고, LLM의 의미 공간과의 자연스러운 정렬을 이룹니다. 이 방법은 시계열을 표 형식으로 변환하고, 고유한 추론 프레임워크를 설계해 LLM의 추론 능력을 극대화합니다.

- **Technical Details**: TableTime은 세 가지 주요 전략을 포함합니다: 첫째, 다변량 시계열 데이터를 표 형식으로 변환하여 시간적 및 채널 특정 정보를 최대한 보존합니다. 둘째, 표 형식 시계열을 텍스트로 변환하여 LLM의 의미 공간과 자연스럽게 정렬합니다. 마지막으로, LLM의 추론 가능성을 극대화하기 위해 이웃 강화, 다중 경로 추론(multi-path inference)을 포함하는 프롬프트를 개발했습니다.

- **Performance Highlights**: 우리는 10개의 공개적으로 대표적인 데이터셋에서 TableTime을 광범위하게 실험하여 기존 LLM 기반 방법의 한계를 극복하고 우수성을 검증했습니다. 이 실험은 TableTime이 다변량 시계열 분류에 있어 효율적이고 뛰어난 성능을 발휘한다는 것을 보여줍니다. 특히, 제안한 방법은 제로샷 분류(zero-shot classification) 능력을 잘 실현하여 실질적인 응용 가능성을 높였습니다.



### Do LLMs Agree on the Creativity Evaluation of Alternative Uses? (https://arxiv.org/abs/2411.15560)
Comments:
          19 pages, 7 figures, 15 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 대안 사용 테스트(Alternative Uses Test, AUT)에 대한 응답의 창의성을 평가하는 데 동의하는지를 조사합니다. 이전 연구들은 주로 단일 모델이 동일한 모델이나 인간이 생성한 응답을 평가하는 데 초점을 맞추었지만, 본 연구에서는 모델들이 자신의 응답과 타 모델의 응답을 공정하고 정확하게 평가할 수 있는지를 탐구합니다. 이를 통해 LLM들이 창의성 평가에서의 편향성을 가지지 않음을 확인할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: 연구에서는 AUT 응답을 공통, 창의적, 매우 창의적인 세 가지 카테고리로 분류하여 LLM들이 이 응답들을 점수화(score)하고 순위를 매기는 실험적 프레임워크를 적용했습니다. 이 연구에서는 4개의 최신 LLM을 사용하며, 두 가지 평가 설정(포괄적 및 분할적)을 활용하여 모델들이 대안 사용의 창의성 평가에서 얼마나 합의하는지를 분석합니다. 연구 결과, 모델 간의 Spearman 상관계수는 평균 0.7 이상으로 높은 합의도를 보여주었고, 오라클과 비교 시에도 0.77을 초과하였습니다.

- **Performance Highlights**: 모델들은 자기 자신의 응답을 선호하지 않으며, 대신 다른 모델이 생성한 대안 사용에 대해 유사한 창의성 평가 점수나 순위를 제공합니다. 이러한 발견은 LLM들이 창의성 평가에서 공정성과 높은 일치를 보임을 시사하며, 이는 자동화된 창의성 평가에 대한 기대를 높입니다. 연구는 LLM들이 대안 사용의 창의성 평가에서 신뢰할 수 있는 평가자로 작용할 수 있음을 보여줍니다.



### QEQR: An Exploration of Query Expansion Methods for Question Retrieval in CQA Services (https://arxiv.org/abs/2411.15530)
- **What's New**: 이번 연구에서는 커뮤니티 기반 Q&A 서비스에서 발생하는 lexical gap 문제를 해결하기 위해 질문 확장(query expansion) 방법을 사용합니다. 특히, 가장 유사한 질문과 단어를 기반으로 질문을 확장하여 기존 질문 아카이브에서 더 관련성 높은 질문을 찾고자 합니다. 또한 기존의 방법 외에 질문 유사도 기반의 새로운 방법과 선택적 확장을 도입하여 질문의 의도가 변하지 않도록 신경 써서 접근합니다.

- **Technical Details**: 레퍼런스에 따르면, 질문 확장의 첫 번째 방법은 유사한 의미를 지닌 단어들을 사용하여 입력 질문을 확장하는 것입니다. 이를 위해 KL-Divergence를 이용해 확장된 질문과 후보 질문들의 관련성을 계산합니다. 또한, 선택적 확장을 도입하여 모든 단어를 확장하지 않고, 특정 단어만 확장할 때 질문의 본래 의도를 유지할 수 있도록 하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 방법의 최적화 모델은 쿼리 확장 없이 최상의 기준선에 비해 1.8%의 상대적 성능 향상을 보여주었습니다. 이는 플랫폼 사용자가 보다 빠르게 정보 필요에 맞는 답변을 찾을 수 있도록 도와주는 중요한 결과입니다. 이 성과는 주어진 질문과 유사한 질문을 찾는 질문 검색의 효과를 높이는 데 기여합니다.



### MolMetaLM: a Physicochemical Knowledge-Guided Molecular Meta Language Mod (https://arxiv.org/abs/2411.15500)
- **What's New**: 본 연구에서 제안한 MolMetaLM은 기존의 분자 언어 모델의 제한점을 극복하기 위한 새로운 물리화학 지식 기반의 메타 언어 프레임워크입니다. 이는 <S,P,O> (주어, 술어, 목적어) 형태의 지식을 활용하여 분자 간의 의미적 관계 학습을 강화합니다. 특히, MolMetaLM은 대량의 사전 훈련 작업을 생성함으로써 모델의 일반성(generality)을 높였습니다.

- **Technical Details**: MolMetaLM에서는 분자 정보를 <S,P,O> 형태로 표현함으로써 분자 관련 태스크를 통합하여 다룰 수 있는 일반적인 프레임워크를 설계하였습니다. 이 프레임워크는 노이즈 제거(pre-training) 과정을 통해 MLM, GLM 기반 분자 언어 모델을 통합하였으며, 특정 물리화학적 속성과 함께 다양한 잡음을 추가하여 분자 표현 능력을 향상시킵니다. 모델 학습 과정에서 18개의 기본 노이즈 제거 사전 훈련 작업이 설계되어 있습니다.

- **Performance Highlights**: MolMetaLM은 대규모 벤치마크 평가에서 우수한 성능을 보였으며, 분자 생성 및 속성 예측에 있어 그 능력과 다재다능성을 입증하였습니다. 본 연구의 접근 방식은 분자 분야에 대한 새로운 통찰력을 제공하며, 분자 모델링의 발전을 위한 토대를 마련합니다.



### Transition Network Analysis: A Novel Framework for Modeling, Visualizing, and Identifying the Temporal Patterns of Learners and Learning Processes (https://arxiv.org/abs/2411.15486)
Comments:
          Accepted at Learning Analytics & Knowledge (LAK '25)

- **What's New**: 본 논문에서는 Transition Network Analysis (TNA)라는 새로운 분석 프레임워크를 제안합니다. 이 접근법은 Stochastic Process Mining (SPM)과 확률적 그래프 표현을 통합하여 학습 과정 데이터에서 전이 패턴을 모델링하고 시각화하는 데 중점을 둡니다. TNA는 학습 이벤트에서 중앙성을 캡처하고, 행동 패턴을 식별하며, 시간적 패턴을 드러내는 기능을 갖추고 있어 학습 분석 및 학습 과학의 중요한 질문들을 해결할 수 있도록 설계되었습니다.

- **Technical Details**: TNA는 학습 이벤트의 전이 행렬을 Markov 모델을 이용해 그래프로 모델링하며, 그래프 표현을 통해 네트워크 분석의 장점을 활용합니다. 이 방법은 일반적인 네트워크 측정(centrality measures) 및 커뮤니티 발견(community finding)과 같은 기술을 활용하여 학습 이벤트 간의 관계를 식별합니다. 또한, 부트스트랩 검증 기술을 통해 허위 전이(Spurious Transitions)를 제거하고 중요한 전이를 식별할 수 있어 결과의 신뢰성을 높입니다.

- **Performance Highlights**: TNA의 사례 연구를 통해 학생들 간의 협력 학습 패턴을 분석한 결과, TNA는 규제 프로세스 및 중요한 이벤트, 시간적 패턴과 클러스터를 성공적으로 드러낼 수 있다는 것을 보여주었습니다. TNA의 기능은 학습 동적 요소를 포착하는 데 유용하며 기존 방법들과 비교하여 효율성을 더불어 신뢰성을 제공합니다. 향후 TNA는 장기적인 데이터 분석(Longitudinal Analysis)과 추정 방법 개선 등 추가적인 연구 방향을 모색할 것입니다.



### A Comparative Analysis of Transformer and LSTM Models for Detecting Suicidal Ideation on Redd (https://arxiv.org/abs/2411.15404)
Comments:
          23rd IEEE International Conference on Machine Learning and Applications, ICMLA 2024 (camera-ready)

- **What's New**: 이번 연구는 Reddit 사용자 게시물에서 자살 사고를 감지하기 위한 최첨단 딥러닝 모델의 효과를 평가합니다. BERT, RoBERTa와 같은 transformer 기반 모델에 대한 성능 비교와 LSTM 모델을 포함하여 자살 의도를 탐지하기 위한 다양한 기계 학습 접근법이 다루어집니다. 특히, RoBERTa가 93.22%의 정확도를 기록하며 가장 우수한 성능을 나타내고 있습니다.

- **Technical Details**: 연구팀은 Reddit에서 많은 서브레딧(subreddit)에서 수집된 데이터셋을 기반으로 자살 사고를 탐지하기 위한 기계 학습 모델을 평가했습니다. LSTM 모델과 다양한 word embeddings도 사용되어 BERT 모델과 비교되었습니다. 이 연구는 자살 사고 탐지를 위한 사실 기반의 언어 처리 기술이 중요함을 강조하며, 통계적 분석 및 LDA(topic modeling)를 통해 데이터의 질을 높였습니다.

- **Performance Highlights**: 연구 결과, 모든 모델은 높은 정확도와 F1 점수를 달성했으나, RoBERTa 모델이 가장 뛰어난 성과를 보였습니다. 두 번째로 LSTM 모델이 BERT 임베딩을 사용하여 92.65% 정확도로 두 번째 최고의 성과를 기록했습니다. 이러한 모델들은 사회적 미디어로부터 자살 사고 탐지를 개선할 수 있는 가능성을 제시하며, 정신 건강 모니터링 도구 개발을 위한 앞길을 열어줍니다.



### ChatBCI: A P300 Speller BCI Leveraging Large Language Models for Improved Sentence Composition in Realistic Scenarios (https://arxiv.org/abs/2411.15395)
- **What's New**: 본 논문에서는 ChatBCI를 소개하며, 이는 P300 스펠러 BCI에 LLM(대형 언어 모델)인 GPT-3.5를 통합하여 사용자에게 단어 완성과 여러 단어 예측 기능을 제공합니다. ChatBCI는 원거리 쿼리를 통해 단어 제안을 누리며, 키보드 GUI를 새롭게 디자인하여 사용자에게 더 효율적이고 개인화된 타이핑 경험을 제공합니다. 이를 통해 문장 작성을 가속화하고, 기존 P300 스펠러의 한계를 극복할 수 있는 potential이 제시됩니다.

- **Technical Details**: ChatBCI는 P300 speller BCI 기술에 LLM의 제로샷 학습을 활용하여, 사용자가 입력한 초기 글자를 기반으로 단어를 제안하고 다음 단어를 예측합니다. 새로운 GUI는 제안된 단어를 추가 키로 표시하며, P300 분류에는 SWLDA가 사용됩니다. 본 연구에서 수행된 두 가지 온라인 스펠링 과제는 ChatBCI의 실용성을 평가하는데 중요한 역할을 하게 됩니다.

- **Performance Highlights**: 실험 결과, Task 1에서 ChatBCI는 기존의 글자 단위 BCI 스펠러보다 평균 62.14%의 시간 단축과 53.22%의 키 입력 수 감소를 달성하며, 정보 전송 속도는 198.96% 증가했습니다. Task 2에선 80.68%의 키 입력 절약을 발생시키고, 8.53자/분의 타이핑 속도를 기록했습니다. 이러한 결과는 ChatBCI가 전통적인 스펠러 BCI에 비해 훨씬 효율적임을 보여주며, 특히 사용자 맞춤의 통신 도구 개발 가능성을 제시합니다.



### Exploring Facets of Language Generation in the Lim (https://arxiv.org/abs/2411.15364)
Comments:
          24 pages

- **What's New**: 이 논문은 언어 생성 알고리즘의 새로운 접근 방식을 제시합니다. 기존 연구(Kleinberg and Mullainathan [KM24])는 모든 셈할 수 있는 언어 컬렉션에 대해 언어 생성 문제의 긍정적인 결과를 보여주었습니다. 후속 연구(Raman and Tewari [RT24])에서는 언어 생성에 필요한 고유 입력의 수를 다루어, 이를 균일 생성과 비균일 생성으로 나눴습니다.

- **Technical Details**: 이 논문은 각 셈할 수 있는 언어 컬렉션에 대해 비균일 생성(non-uniform generation)을 가능하게 하는 생성기를 제시합니다. 여기서 생성기는 고유 문자열의 고정된 상수를 수신한 후에만 유효한 문자열을 생성합니다. 그러나, 알고리즘이 입력된 언어 컬렉션의 두 언어만을 이용해 비균일 생성을 수행할 수 없다는 것도 증명하였습니다.

- **Performance Highlights**: 결과적으로, 이 논문은 유효성과 범위 간의 트레이드오프가 생성 알고리즘에서 내재되어 있음을 보여줍니다. 또한, 피드백을 포함한 균일 생성 모델을 연구하여 피드백이 가능한 언어 컬렉션의 특성을 규명하였습니다. 이 연구는 생성 알고리즘의 새로운 가능성을 제시하며, 다양한 응용 프로그램에서 활용될 수 있습니다.



### MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs (https://arxiv.org/abs/2411.15296)
Comments:
          Produced by MME+MMBench+LLaVA Teams. Project Page: this https URL

- **What's New**: 이번 논문에서는 인공지능 일반화(AI General Intelligence, AGI)의 중요한 방향으로 여겨지는 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 평가 방법에 대한 포괄적인 조사 결과를 제시합니다. 기존의 LLM들을 기반으로 하여 MLLMs는 시각과 오디오를 포함한 다양한 정보 형태를 처리하는 능력을 추가로 개발하였습니다. 이에 따라 MLLMs의 성능을 평가하기 위한 새로운 기준과 방법들이 요구되고 있음을 설명합니다.

- **Technical Details**: MLLMs는 모달리티 인코더(modality encoder), LLM, 그리고 이들을 연결하는 커넥터(connector)로 구성되어 있습니다. 비전-언어 모델의 예를 들면, 텍스트 쿼리와 비전 샘플을 입력으로 받아 비전 인코더가 특징(feature)을 추출하고, 커넥터는 비전 특징과 텍스트 임베딩을 정렬합니다. 이후 이 정렬된 비전 특징은 사용자 쿼리의 텍스트 임베딩과 결합되어 LLM에 의해 자연어 응답으로 생성됩니다.

- **Performance Highlights**: MLLM 평가의 예상 방향으로는 기능별 분류, 기능 중심 평가, 태스크 지향 평가, 그리고 더욱 다양한 모달리티의 포함 등이 있습니다. 본 논문은 다양한 평가 기준을 계층적으로 분류하고, 평가 제작 과정에서 필요한 주의 사항을 정리하였으며, 성능 측정 방법으로 인간 기반, LLM 기반, 그리고 스크립트 기반의 세 가지 주요 방식을 제시합니다. 이로써 연구자들이 적절한 평가 기준을 쉽게 찾고 효과적인 평가 방안을 탐색할 수 있도록 돕고자 합니다.



### ICT: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models (https://arxiv.org/abs/2411.15268)
- **What's New**: 이 논문은 LVLMs (Large Vision Language Models)의 고질적인 허위 현상(hallucination)을 줄이기 위한 새로운 접근 방식을 제안합니다. ICT (Image-Object Cross-Level Trusted Intervention)라는 경량의 훈련이 필요 없는 방법을 소개하며, 이 방법은 모델이 시각적 정보에 대한 집중을 높이는 데 도움을 줍니다. 기존의 방법들과 달리, ICT는 언어 프라이어를 제거하지 않고도 모델의 주의를 조정하면서 이 문제를 해결합니다.

- **Technical Details**: ICT는 모델의 주의를 조정하는 개입(intervention) 방향을 계산하는 프로세스를 통하여 동작합니다. 이 과정에서는 모델이 visual information과 fine-grained object details를 모두 활성화할 수 있도록 attention heads의 활성화 패턴을 분석합니다. 이때, attention heads의 활성화 값을 조정하여 모델이 신뢰할 수 있는 정보를 강화하도록 합니다.

- **Performance Highlights**: 실험 결과, ICT는 LLaVA-v1.5와 Qwen-VL 모델에서 각각 POPE 벤치마크에서 평균 6.27%의 성능 향상을 보였으며, MME 벤치마크에서 67.37포인트의 향상을 달성했습니다. 더불어 ICT는 다양한 데이터셋과 모델에 대해 일반화되는 성능을 보이며, 기존의 decoding 전략과 병행하여 사용할 수 있는 장점을 가지고 있습니다.



### TPLogAD: Unsupervised Log Anomaly Detection Based on Event Templates and Key Parameters (https://arxiv.org/abs/2411.15250)
- **What's New**: 이번 연구에서는 TPLogAD라는 새로운 보편적 비지도형(log anomaly detection) 로그 분석 방법을 제안합니다. TPLogAD는 이벤트 템플릿과 핵심 매개변수를 기반으로 하는 비지도형 접근 방식을 통해 로그에서의 이례성을 효과적으로 탐지합니다. 기존 방법들은 템플릿 인덱스나 특정 문자열 임베딩을 사용했으나, TPLogAD는 더 풍부한 의미 분석을 가능하게 합니다.

- **Technical Details**: TPLogAD는 itemplate2vec와 para2vec이라는 두 가지 효율적이고 구현이 용이한 의미 표현 방식을 포함하고 있습니다. itemplate2vec는 이벤트 템플릿에서의 의미 정보를 추출하는 데 사용되며, Bert를 이용해 로그 내 단어 간의 의미적 관계를 학습합니다. 반면, para2vec는 로그 내 매개변수에서 의미 정보를 정확하게 추출하기 위해 적응형 클러스터링 알고리즘을 사용합니다.

- **Performance Highlights**: TPLogAD는 네 개의 공개 로그 데이터셋에 대해 기존 로그 이례성 탐지 방법들보다 월등한 성능을 보여주었습니다. 실험 결과, TPLogAD는 로그 탐지 정확성을 크게 향상시켰으며, 다양한 로그 포맷과 동적 데이터를 처리하는 데 있어 탁월한 적응력을 발휘합니다.



### Bio-inspired AI: Integrating Biological Complexity into Artificial Intelligenc (https://arxiv.org/abs/2411.15243)
- **What's New**: 이 논문은 인공지능(AI)의 발전이 생물학적 계산에서의 기본 원칙을 어떻게 응용할 수 있는지 탐구합니다. 특히, 맥락에 따라 달라지는 계층적 정보 처리(hierarchical information processing), 시행착오 발견(trial-and-error heuristics), 그리고 다중 스케일 조직(multi-scale organization)의 중요성을 강조합니다. 생물학적 지능의 미세한 메커니즘을 조사하여 인공지능 시스템 설계에서의 가능성과 한계를 조명하고자 합니다. 결과적으로 생물학적 시스템에서 영감을 받은 더 적응적이고 강력한 AI 시스템을 설계하는 프레임워크를 제안합니다.

- **Technical Details**: AI의 역사적 맥락에서 저자들은 토마스 홉스와 블레즈 파스칼의 기계적 사유 이론(mechanical theory)과 같은 초기 사상가들이 지능형 기계의 꿈을 어떻게 펼쳤는지를 분석합니다. 생물학적 계산 원칙에 따르면, 생물학적 지능은 신경계에 국한되지 않고 단세포 생물이나 식물에서도 정보 처리 및 적응 행동을 보여 줍니다. 자연에서 발견되는 지능은 계층적 상호작용과 다중 스케일에서의 과정들로 형성되며, 이는 AI 시스템 설계에서 중요한 참고자료가 됩니다.

- **Performance Highlights**: 현재 AI 시스템 개발에서 생물학적 영감을 받아 RoboBee와 RoboRay 프로젝트와 같은 새로운 접근 방식이 주목받고 있습니다. 이러한 시스템들은 경량화와 유연성을 추구하며, 뇌와 유사한 에너지 효율(enery efficiency)을 가지려는 뉴로모픽 컴퓨팅(neuromorphic computing) 기술에 의존합니다. 그러나 여전히 많은 생물학적 문제 해결은 신경망을 기반으로 한 접근 방식이 아닌 다른 생물학적 메커니즘에서도 발생하고 있습니다. 이 논문은 그러한 다양한 생물학적 전략들이 AI 설계에 더 많은 통찰력을 제공할 수 있음을 강조합니다.



### The Zamba2 Suite: Technical Repor (https://arxiv.org/abs/2411.15242)
Comments:
          21/11/24 initial upload

- **What's New**: Zamba2 시리즈는 1.2B, 2.7B, 7.4B 파라미터를 가진 하이브리드 Mamba2-Transformer 모델의 집합으로, 최첨단 성능을 제공하면서 추론 지연(inference latency), 처리량(throughput), 메모리 효율성(memory efficiency)에서도 큰 향상을 이루었습니다. 이 모델들은 3조 토큰까지 훈련되었으며, 이전의 Zamba1-7B 작업을 바탕으로 아키텍처와 훈련 데이터를 최적화했습니다. Zamba2 시리즈 모델에 대한 오픈 소스 가중치(weights)와 훈련 데이터셋인 Zyda-2도 공개되었습니다.

- **Technical Details**: Zamba2 아키텍처는 이전 Zamba1-7B 모델에서 도입된 혁신을 바탕으로 아키텍처 개선 과정을 거쳤습니다. Mamba1에서 Mamba2로의 전환을 통해 더 높은 처리량을 달성하였고, 두 개의 교차 공유 어텐션 블록(shared attention blocks)을 도입하여 성능을 개선했습니다. 또한, 저순위 어댑터(Low-Rank Adapters)와 Rotary Position Embeddings를 적용하여 모델의 표현력을 증가시켰습니다.

- **Performance Highlights**: Zamba2 시리즈 모델은 언어 모델링 평가에서 최첨단 성능을 달성하며, 인퍼런스와 메모리 효율성 면에서도 우수합니다. 동급의 트랜스포머 모델과 비교했을 때, 최대 30-50%의 첫 토큰 처리 시간 단축과 6배의 KV 캐시 메모리 요구량 감소를 자랑합니다. 이러한 성능 향상은 특히 리소스가 제한된 장치에서 모델을 실행하는 데 매우 유리합니다.



### BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models (https://arxiv.org/abs/2411.15232)
Comments:
          18 pages, 5 figures, 10 tables

- **What's New**: 이 논문은 BiomedCoOp이라는 새로운 프롬프트 학습 프레임워크를 제안하여 BiomedCLIP을 효율적으로 적응시키고, 적은 수의 데이터로 생물 의학 이미지를 분류하는 정확성을 높이는 방법을 소개합니다. 기존의 프롬프트 학습 기법들은 일반화에 한계가 있었지만, BiomedCoOp은 대형 언어 모델(LLMs)을 활용하여 컨텍스트 학습을 효과적으로 수행합니다. 또한, 통계 기반의 프롬프트 선택 전략을 통해 제거된 프롬프트로 인한 문제를 해결합니다.

- **Technical Details**: BiomedCoOp는 CoOp 방식을 기반으로 하여, LLM에서 유도된 프롬프트 앙상블을 통해 의미론적 일관성을 강화합니다. 이 시스템은 대형 생물 의학 데이터셋을 포함한 11개 의료 데이터셋에서 검증되어, 다양한 모달리티와 신체 기관에 걸쳐 성능 개선을 보여주었습니다. 특히, 우리는 일반 지식 CLIP 모델과 비교하여 BiomedCLIP을 활용함으로써 다양한 임상 작업에서 이점을 입증했습니다.

- **Performance Highlights**: 이 연구는 BiomedCoOp의 성능이 기존의 CLIP 프롬프트 학습 기술과 비교하여, 다양한 의료 조건 및 이미징 모달리티에서 우수한 일반화 능력과 강건성을 발휘함을 나타냅니다. 실험 결과는 BiomedCoOp가 적은 샷 학습 구조에서 높은 정확도를 달성했음을 보여줍니다. 이로 인해, 생물 의학 이미지 분석의 효율적인 프롬프트 학습 방법으로서의 가능성을 제시합니다.



### Uni-Mlip: Unified Self-supervision for Medical Vision Language Pre-training (https://arxiv.org/abs/2411.15207)
Comments:
          15 pages, 2 figures, accepted by BMVC'24

- **What's New**: 최근 비전-언어 사전 학습(Vision-and-Language Pre-training) 기술의 발전이 컴퓨터 비전 분야의 성능을 크게 향상시켰습니다. 그러나 의료 분야에서는 다중 모달 데이터를 얻는 것이 비용이 많이 들고 복잡하며, 개인정보 보호 등 여러 어려움이 있습니다. 이를 해결하기 위해 Uni-Mlip라는 새로운 프레임워크를 소개하며, 이 프레임워크는 의료 비전-언어 사전 학습을 위한 통합 자기 지도화(self-supervision) 접근 방식을 제공합니다.

- **Technical Details**: Uni-Mlip은 데이터 수준과 특징 수준 모두에서 교차 모달(cross-modality), 단일 모달(uni-modality), 융합 모달(fused-modality) 자기 지도화 기법을 통합하여 의료 이미지를 효과적으로 처리합니다. 특히, 의료 이미지의 특성에 맞춘 단일 모달 이미지 자기 지도화 기법을 조정하여 높은 정밀도와 자세한 감도를 제공합니다. 이러한 접근 방법을 통해 의료 데이터를 더 효과적으로 발굴하고, 모델의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 다양한 규모의 데이터세트에 대한 실험 결과, Uni-Mlip는 이미지-텍스트 검색(image-text retrieval), 이미지 분류(image classification), 시각적 질의 응답(Visual Question Answering, VQA)과 같은 주요 다운스트림 작업에서 현재의 최첨단 방법들을 능가하는 뛰어난 성능을 보여줍니다. 이 성능 향상은 모델의 특징과 데이터 간의 정렬 및 전이 가능성을 크게 개선하는 데 기여하였습니다.



### Multimodal large language model for wheat breeding: a new exploration of smart breeding (https://arxiv.org/abs/2411.15203)
- **What's New**: 이 연구는 UAV 원격 감지 기술을 활용하여 작물 육종에 필요한 데이터 수집의 비파괴적이고 고속화를 달성하는 새로운 접근 방식을 제시합니다. 특히, 다양한 사전 훈련된 오픈 소스 다중 모달 대형 언어 모델(MLLMs)을 기반으로 하여 스마트 육종 목표 도구를 개발했습니다. 이 도구는 다중 분야의 데이터 마이닝을 가능하게 합니다.

- **Technical Details**: 이 연구는 감독된 세밀조정(Supervised Fine-tuning, SFT), 검색 증강 생성(Retrieval-augmented Generation, RAG), 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 기술을 이용하여 MLLMs에 다중 분야의 지식을 주입했습니다. 이를 통해 밀 생산량 예측을 위한 다수의 다중 모달 대형 언어 모델(Wheat Breeding Language Models, WBLMs)을 구축하였고, 새롭게 생성된 평가 기준을 사용하여 이 모델을 평가했습니다.

- **Performance Highlights**: 연구 결과, InternVL2-8B를 사용하여 SFT, RAG 및 RLHF 기술로 구축된 WBLM은 매우 우수한 성능을 보였습니다. 특히, WBLM은 원격 감지, 표현형 데이터, 날씨, 유전자원 데이터를 동시에 활용하여 밀 생산량을 예측할 때 R2 값이 0.821, RMSE가 489.254 kg/ha로 나타났습니다. 또한, WBLM은 환경 스트레스 평가, 목표 유전자원 선별, 재배 기술 추천 등 다양한 전문적 의사결정 지원 답변을 생성할 수 있습니다.



### Guiding Word Equation Solving using Graph Neural Networks (Extended Technical Report) (https://arxiv.org/abs/2411.15194)
- **What's New**: 본 논문은 단어 방정식을 해결하기 위한 그래프 신경망(Graphic Neural Network) 기반의 알고리즘을 제안합니다. 이 알고리즘은 방정식을 분할하는 잘 알려진 Nielsen 변환을 토대로 발전되었습니다. 각 방정식 변형의 첫 번째 항을 반복적으로 재작성하여 트리와 같은 검색 공간을 생성하며, 분할 지점에서의 경로 선택이 문제 해결 시간에 미치는 영향을 다룹니다.

- **Technical Details**: 제안된 알고리즘은 DragonLi라는 이름의 솔버로 구현되었으며, 멀티 분류 작업으로서 분할 결정을 인코딩합니다. 또한 다섯 가지의 단어 방정식 그래프 표현을 도입하여 GNN의 구조적 정보를 인코딩합니다. GNN은 일반적으로 비유클리드(Non-Euclidean) 데이터 구조에 적합한 딥 러닝 알고리즘으로, 이 논문에서 처음으로 단어 방정식 문제에 적용되었습니다.

- **Performance Highlights**: 실험 결과, DragonLi는 특히 만족 가능한 문제에서 높은 성능을 발휘하며, 단일 단어 방정식의 경우 기존의 문자열 솔버보다 현저히 많은 문제를 해결할 수 있습니다. 여러 단어 방정식의 결합에 대한 경우에도 DragonLi는 최신 문자열 솔버와 경쟁력을 유지합니다. GNN 기반의 지침이 DragonLi의 성능을 균일하게 향상시켜 주며, 특정 벤치마크에서는 모든 다른 솔버를 초월하는 결과를 보였습니다.



### Information Extraction from Heterogeneous Documents without Ground Truth Labels using Synthetic Label Generation and Knowledge Distillation (https://arxiv.org/abs/2411.14957)
Comments:
          Accepted to WACV 2025

- **What's New**: 이 논문에서는 직원들이 제출하는 인보이스와 영수증과 같은 비주얼리 리치 문서(Visually Rich Documents, VRDs)의 정보 추출 문제를 다루고 있습니다. 저자들은 라벨이 없는 VRD 데이터셋에서 합성 라벨을 생성할 수 있는 Task Aware Instruction-based Labelling (TAIL)이라는 새로운 방법을 제안하며, 이를 활용하여 다중 모달 VRD 이해 모델인 LlaVA-Net을 조정합니다. 이 모델은 고급 멀티모달 모델인 Claude 3 Sonnet과 유사한 성능을 보이는 동시에 비용을 85% 줄이고 속도는 약 5배 증가시킬 수 있음을 입증합니다.

- **Technical Details**: 저자들은 TAIL 메소드를 통해 라벨이 없는 VRD에 대한 합성 소프트 라벨을 생성합니다. 이 과정에서, 기존의 교사 모델의 가중치나 훈련 데이터셋을 사용하지 않고 지식 증류(response-based knowledge distillation)를 통해 작은 다중 모달 모델을 훈련하여 효율적인 정보 추출을 가능하게 합니다. 이 방법론은 고급 OCR 기술을 넘어, 텍스트의 기하학적 구조와 문맥적 단서를 바탕으로 문서의 정보를 추출하는 지능적인 추론(Reasoning)을 요구합니다.

- **Performance Highlights**: LlaVA-Net은 비용 대 성능 비율이 뛰어나며, 기존의 레이아웃 인식 모델보다 10% 이상의 성능 향상을 보였습니다. 또한, 잘 알려진 기준 데이터셋을 활용하여 자신의 접근 방식이 Claude 3 Sonnet과 유사한 수준에 도달함을 입증했습니다. 이 모델은 기업의 내부 지출 문서에 대해 잘 작동하며, 잠재적인 남용, 낭비 또는 사기를 탐지하는 데 사용될 수 있음을 보여주었습니다.



### KBAlign: Efficient Self Adaptation on Specific Knowledge Bases (https://arxiv.org/abs/2411.14790)
- **What's New**: 이번 논문에서는 KBAlign이라는 새로운 접근法을 제안하여, 대규모 언어 모델(LLM)을 효율적으로 지식 베이스(KB)에 적응시키는 방법을 소개합니다. 이 방법은 자가 주석(self-annotation) 데이터를 활용하여 모델이 지식 내용을 신속하게 이해하고, 실제 작업에서 성능을 향상시키는 데 중점을 둡니다. 실험 결과, KBAlign을 사용함으로써 GPT-4-turbo 주석을 이용했을 때와 유사한 성능 향상을 달성할 수 있음을 보여줍니다.

- **Technical Details**: KBAlign은 Q&A 쌍 및 수정 제안을 포함한 자기 주석 데이터를 반복적으로 훈련하는 방식을 사용합니다. 이 과정을 통해 모델은 자신의 응답을 점검하고, 학습이 진행될수록 기존의 오류를 수정하게 됩니다. 또한, 쿼리 확장(query expansion) 및 확신 검증(confidence verification)과 같은 전략을 사용하여 응답을 개선하는 과정이 포함되어 있습니다.

- **Performance Highlights**: KBAlign은 사실 Q&A, 긴 형식 QA, 전문 분야 테스트 등 다양한 데이터 세트에서 실험을 수행하여 그 효율성을 입증하였습니다. 이를 통해 일반 지식 내용을 효과적으로 파악하고, 특정 지식이 필요한 하위 작업에서 성능을 크게 향상시켰습니다. 부가적인 실험 결과로 가장 효율적인 자가 주석 데이터 양과 최적의 훈련 볼륨을 찾아내어 실용적인 적용에 대한 귀중한 지침을 제공합니다.



New uploads on arXiv(cs.IR)

### Exploring Structural Dynamics in Retracted and Non-Retracted Author's Collaboration Networks: A Quantitative Analysis (https://arxiv.org/abs/2411.17447)
- **What's New**: 이 연구는 두 가지 주요 유형의 데이터셋, 즉 철회된(co-authorship) 및 철회되지 않은(non-retracted) 공저 기록을 분석하였으며, 이들 간의 네트워크 구조의 비교를 중심으로 진행되었습니다. 연구에서는 철회된 논문의 공저자 간 협업 네트워크에서의 위험 요소를 식별하기 위해 상관 관계를 평가하고, 중앙성(degree centrality) 및 가중 중앙성(weighted degree)과 같은 주요 메트릭을 분석하였습니다. 이를 통해 연구의 진실성과 신뢰성을 높이기 위한 정책 개선 방안을 모색할 수 있는 기초 자료를 제공하였습니다.

- **Technical Details**: 연구는 CrossRef, Retraction Watch, Scopus에서 데이터셋을 수집하여, 철회된 논문과 비철회 논문 간의 협업 네트워크의 구조적 차이를 분석하였습니다. 주된 수학적 도구로는 $t$-테스트(t-test)와 Cohen's $d$가 포함되어 있으며, 이를 통해 네트워크의 중앙성 및 연결성 메트릭에 대한 통계적 유의성을 평가하였습니다. 연구에서 확인된 협업 네트워크의 계층적 및 집중적 구조는 향후 연구에서 중요한 시사점을 제공할 수 있습니다.

- **Performance Highlights**: 통계적 분석을 통해 철회된 논문의 네트워크가 집단적인 특성을 가지며, 비철회 네트워크에 비해 더 많은 중앙 집중성을 보인다는 결과를 도출하였습니다. 이러한 발견은 향후 철회 경향성을 예측하고, 저자 및 연구기관의 신뢰성을 강화하는데 필요한 기반을 마련합니다. 궁극적으로, 이러한 네트워크 분석은 연구의 무결성을 높이는 데 중대한 기여를 할 것으로 예상됩니다.



### Towards Robust Cross-Domain Recommendation with Joint Identifiability of User Preferenc (https://arxiv.org/abs/2411.17361)
Comments:
          12 pages, 6 figures, under review

- **What's New**: 최근의 Cross-Domain Recommendation (CDR) 연구들은 도메인 공유(user representations)와 도메인 특정에 따라 분리된 사용자 표현을 통해 도메인 간의 격차를 완화하고 효과적인 지식 전이를 촉진하는 방안을 모색하고 있습니다. 그러나 이러한 분리(disentanglement)는 실제로 도전적이며, 사용자 행동이 복잡하게 얽혀 있어서 관찰된 사용자-아이템 상호작용으로는 진정한 사용자 선호를 완전히 포착할 수 없습니다. 이 논문에서는 사용자 표현들이 도메인 간에 정확하게 대응될 수 있도록 하는 joint identifiability 모델을 제안합니다.

- **Technical Details**: 본 연구에서는 계층적 사용자 선호 모델링 프레임워크를 도입하여 신경망 인코더의 깊이에 따라 사용자 표현을 정리하였습니다. 이 프레임워크는 얕은 서브스페이스(shallow subspace)에서 사용자마다 관심 중심(interest centroids)을 모델링하고, 이러한 중심들을 도메인 간에 선택적으로 정렬하여 도메인 무관한 특성에서의 미세한 일관성을 보장합니다. 또한, 깊은 서브스페이스(deep subspace) 표현은 공유된 공통 컴포넌트(shared component)와 도메인 변동 컴포넌트를 분해하여 joint identifiability를 강제합니다.

- **Performance Highlights**: 실제 CDR 작업에서 다양한 도메인 상관관계를 고려한 실험을 통해 본 방법이 최신 기술 수준(state-of-the-art)보다 일관되게 우수한 성능을 발휘함을 입증하였습니다. 특히, 약한 상관관계의 작업에서도 뛰어난 성능을 나타내어, robust CDR을 달성하기 위한 joint identifiability의 중요성을 강조합니다.



### 2D Matryoshka Training for Information Retrieva (https://arxiv.org/abs/2411.17299)
- **What's New**: 2D Matryoshka Training은 여러 레이어-차원 설정에서 인코더 모델을 동시에 훈련시키기 위해 설계된 혁신적인 임베딩 표현 훈련 방법입니다. 이 방법은 Semantic Text Similarity (STS) 작업에서 기존 훈련 방식보다 더 높은 효율성을 보여주고 있습니다. 그러나 두 가지 구현 간의 차이로 인해 비교 결과가 다르게 나타나는 문제도 존재합니다. 본 연구에서는 두 가지 버전의 2D Matryoshka Training을 구현하고 평가하여 이 차이를 분석했습니다.

- **Technical Details**: 2D Matryoshka 훈련은 'Matryoshka Representation Learning'에서 영감을 받아 서로 다른 차원 크기를 가지는 여러 임베딩을 동시에 학습합니다. 이 방법은 전체 모델을 통과하지 않고 직접 서브 레이어에서 임베딩을 추출함으로써 텍스트 인코딩에 필요한 시간을 대폭 줄이고 있습니다. 이를 통해 훈련된 임베딩은 전통적인 전체 크기 모델보다 STS 작업에서 더 높은 효과를 나타냅니다. 본 연구는 두 가지 버전의 2D Matryoshka 훈련 설계를 통해 이 기술의 효과를 더욱 심도 있게 살펴봅니다.

- **Performance Highlights**: 2D Matryoshka 훈련의 두 가지 버전은 모두 전통적인 Matryoshka 훈련 및 전체 모델 훈련 방식보다 높은 효과를 보였습니다. 그러나 특정 서브 레이어 및 서브 차원 설정으로 별도로 훈련된 모델에는 미치지 못하는 것을 발견했습니다. 이러한 연구 결과는 감독 학습 및 제로샷 환경에서도 정보 검색 작업에 잘 일반화되었습니다. 추가적으로, 손실 계산 방식을 수정하여 검색 효과성을 향상시킬 수 있는 방법도 모색하였으며, 이를 통해 최적의 검색 효과성을 달성하기 위한 새로운 전략을 제안하였습니다.



### Scholar Name Disambiguation with Search-enhanced LLM Across Languag (https://arxiv.org/abs/2411.17102)
- **What's New**: 본 논문은 학술명 구분(Disambiguation) 과제를 해결하기 위한 새로운 접근 방식을 제안하고 있습니다. 기존 방법이 복잡한 헤테로지너스 데이터로 인해 한계를 겪는 가운데, 여러 언어의 검색 향상 언어 모델(search-enhanced language models)을 활용하여 성능을 개선하고자 했습니다. 검색엔진의 쿼리 재작성(query rewriting)과 데이터 인덱싱(data indexing) 기능을 사용하는 이번 방안은 다양한 데이터를 수집하고 학자의 프로파일을 추출하는 데 있어 더 풍부한 정보를 제공합니다.

- **Technical Details**: 우리는 LLM(대형 언어 모델)과 검색 엔진의 협력을 통해 성과를 극대화하는 방법론을 적용했습니다. 이 방법론은 학술 프로파일 추출, 이름 일치(name matching) 및 논문 이름 구분(paper name disambiguation) 등의 여러 모듈로 구성되어 있습니다. 특히, 비영어권 학자의 경우, 그들의 정보는 원주의(native) 언어에서 더 풍부하게 제공되므로, 검색 엔진의 기능을 활용하여 다양한 데이터 원천을 연결하고 해석하는 데 중점을 두었습니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식은 특히 지리적 다양성이 큰 학자들 간의 이름 구분(disambiguation) 성능을 현저히 향상시킴을 보여주었습니다. 논문에서 제안하는 방안은 다국어 및 검색 향상 방법론을 통해 적극적인 학술 이름 구분의 효율성을 높이는 가능성을 보여줍니다. 이러한 접근 방식은 특히, 데이터가 복잡하고 heterogeneous한 환경에서 그 효과를 발휘하며, 향후 연구와 실용적 응용에 큰 의미를 가질 것으로 기대됩니다.



### Enabling Adoption of Regenerative Agriculture through Soil Carbon Copilots (https://arxiv.org/abs/2411.16872)
- **What's New**: 이번 연구에서는 기후 변화 완화를 위한 새로운 접근 방식을 소개합니다. AI 기반의 Soil Organic Carbon Copilot을 통해 농업의 환경적 영향을 최소화하고 기후 회복력을 구축할 수 있는 통찰을 제공합니다. 이 시스템은 여러 종류의 데이터(예: 극단적 날씨 사건, 농장 관리 데이터)를 자동으로 수집하고 분석하여 대규모로 토양 건강과 재생 농업 관행을 조사할 수 있도록 지원합니다.

- **Technical Details**: SOC Copilot은 텍스트 및 표 형식의 다양한 데이터를 수집합니다. 이를 통해 최근의 토양 과학 문서를 포함한 비구조적 데이터와 극단적인 날씨 사건 데이터를 통합하여 SOC에 영향을 미치는 다양한 환경적 및 농장 관리 관행에 대한 통찰을 제공합니다. 또한, 컴포스터 역할에 대해 맞춤형 데이터를 제공하여 사용자 요구에 맞는 정보 제공을 보장합니다.

- **Performance Highlights**: 연구 결과, SOC Copilot은 극단적 날씨 사건이 SOC 변화에 미치는 영향을 정밀하게 분석할 수 있는 능력을 보였습니다. 예를 들어, 소노마 카운티의 SOC 증가 원인과 관련된 정보 및 몬터레이 카운티에서의 SOC 감소 이유를 비교 분석하여, 재생 농업 관행이 극단적인 환경 조건에서 SOC 손실을 방지하는 데 효과적임을 확인하였습니다. 이러한 통찰은 정책 결정에 필요한 증거 기반 전략을 구현하는 데 기여할 것입니다.



### Automating Chapter-Level Classification for Electronic Theses and Dissertations (https://arxiv.org/abs/2411.17614)
- **What's New**: 이 논문은 전통적인 전자 석사논문 및 박사논문(ETD) 보관 방식의 한계를 극복하기 위해 AI와 머신 러닝을 활용하여 각chapter에 대한 메타데이터를 자동으로 분류하는 방법을 제안합니다. 기존의 고수준 메타데이터는 ETD의 깊이와 복잡성을 제대로 담아내지 못해 연구자들이 특정 정보를 찾는 데 어려움을 겪고 있었습니다. 이를 해결하기 위해 chapter-level metadata를 도입하여 각 장의 내용을 보다 쉽게 찾을 수 있도록 하여 학술 연구의 접근성과 활용성을 높이고자 합니다.

- **Technical Details**: 이 연구는 두 가지 주요 작업인 segmentation과 classification으로 구성됩니다. 우선, segmentation 작업을 통해 PDF 내 chapter 경계를 식별하고, 이후 classification 작업을 통해 각 장에 대한 구체적인 메타데이터를 할당합니다. 이를 통해 연구자들은 전통적인 문서 수준의 메타데이터에서 벗어나 각각의 chapter를 더 효과적으로 탐색할 수 있습니다. 우리는 전통적인 머신 러닝 분류기, BERT 기반의 모델, 대형 언어 모델(LLMs) 간의 효과를 비교하며, 각 장의 특정 권한 레이블을 생성하는 접근법을 탐구합니다.

- **Performance Highlights**: 개발된 프로토타입 시스템을 통해 chapter-level classification labels가 제공되면 연구자들은 자신의 관심사에 맞는 chapters를 신속하게 검색할 수 있습니다. 또한, AI 향상 접근 방식을 통해 아카이브가 연구자들에게 더 나은 서비스를 제공하고 정보 접근성을 증가시키며 ETD와의 깊은 상호 작용을 지원할 수 있도록 합니다. 이를 통해 ETD의 학문적 도구로서의 영향력 강화, 학제 간 탐색 촉진 및 학술 커뮤니케이션에서 아카이브의 역할을 재강조할 수 있습니다.



### Making History Readab (https://arxiv.org/abs/2411.17600)
- **What's New**: 본 논문은 버지니아 공대 도서관의 디지털 라이브러리 플랫폼(DLP)이 역사적 및 문화적 중요성이 있는 다양한 문서에 대한 접근을 제공하는 방법을 제시합니다. AI를 활용하여 손글씨 인식, 텍스트 추출, 요약 등을 통해 복잡한 디지털 자료에 대한 온라인 접근성을 개선하고 있습니다. 이러한 접근은 특히 1860년대의 남북전쟁 편지, 역사적인 신문, 디지털 지형도와 같은 세 가지 컬렉션을 통해 구체화됩니다.

- **Technical Details**: DLP는 클라우드 기반 솔루션으로, 텍스트 인식과 메타데이터 생성을 위한 Optical Character Recognition (OCR) 기술을 통합하고 있습니다. Pytesseract 및 AWS Textract와 같은 도구를 사용해 손글씨와 복잡한 레이아웃의 문서에서 텍스트를 추출하며, LLM(large language models)은 요약 서비스를 제공합니다. 특히, Llama-3.1-8B-Instruct 모델을 활용하여 손글씨 텍스트에서 요약을 생성하는 방법이 소개됩니다.

- **Performance Highlights**: 손글씨 문서의 텍스트 추출 및 요약 처리 결과, 이전에 접근할 수 없었던 콘텐츠가 검색 가능하도록 변환되었습니다. 신문 컬렉션에서는 복잡한 열 구성으로 인한 인식 문제를 해결하고, 지형도 컬렉션의 경우 텍스트 비스듬한 배치 문제를 해결하기 위해 다각 회전 전략을 사용하였습니다. 이러한 방법들은 디지털 라이브러리 플랫폼의 이용자 경험을 향상시키며, 더 많은 사용자들이 자료를 쉽게 탐색하고 이해할 수 있도록 지원합니다.



### Agentic AI for Improving Precision in Identifying Contributions to Sustainable Development Goals (https://arxiv.org/abs/2411.17598)
- **What's New**: 이 연구는 연구 기관들이 유엔의 지속 가능한 발전 목표(SDGs)에 대한 연구 성과를 정확히 평가해야 하는 필요성을 강조합니다. 기존의 키워드 기반 접근 방식을 넘어 자가 회귀형 대형 언어 모델(LLMs)을 이용한 평가 방법론을 제안합니다. LLMs는 여러 문서 간의 의미상 관련성을 구분할 수 있어 SDG 목표에 대한 진정한 기여도를 파악하는 데 도움을 줍니다.

- **Technical Details**: 연구는 Scopus에서 선택한 쿼리를 통해 17개 SDGs 각각에 대해 20,000개의 학술 초록을 수집하여 이를 분석합니다. 이 과정에서 Phi-3.5-mini, Mistral-7B, Llama-3.2와 같은 소형 LLMs를 활용하여 추출한 데이터의 관련성을 평가합니다. 각 모델은 구체적인 SDG 표적과의 일치를 기준으로 초록을 'Relevant' 또는 'Non-Relevant'로 분류하며, 이를 통해 연구 모델 간의 차이를 드러내고자 합니다.

- **Performance Highlights**: 각 LLM의 'Relevant'와 'Non-Relevant' 초록의 비율은 상이하며, 이는 서로 다른 평가 기준을 적용한 결과를 나타냅니다. Phi-3.5-mini는 52%를 'Relevant'로, Mistral-7B는 70%, Llama-3.2는 15%의 'Relevant' 비율을 보였습니다. 이러한 결과는 다양한 모델의 분류 전략 통합의 중요성을 시사하며, 향후 다중 에이전트 프레임워크를 통해 SDG 기여도 분류의 신뢰성을 강화할 수 있는 가능성을 보여줍니다.



### Fairness And Performance In Harmony: Data Debiasing Is All You Need (https://arxiv.org/abs/2411.17374)
- **What's New**: 이 연구는 870개의 프로필을 포함한 실제 대학 입학 데이터셋을 사용하여 머신 러닝(ML) 모델의 공정성을 조사합니다. XGB, Bi-LSTM, KNN 세 가지 ML 모델을 활용하여 성별 언어 편향을 제거하는 파이프라인을 제안하며, ML 모델이 인간 전문가보다 14.08%에서 18.79% 더 높은 공정성을 기록함을 보여줍니다. 또한, 성별 언어 편향 제거 후 모든 모델의 분류 정확도가 유지되거나 향상되는 성과를 보였습니다.

- **Technical Details**: 이 연구에서는 BERT(Bidirectional Encoder Representations from Transformers) 임베딩을 사용하여 고차원 텍스트 특징을 인코딩하고, 인간 전문가의 결정과 ML 모델의 일관성을 평가하기 위해 일관성 점수를 사용합니다. 성별 편향을 완화하기 위해 설계된 디바이빙 파이프라인은 입력 기능에서 성별 특정 언어를 제거하고, 사전 훈련된 BERT 모델을 활용하여 성별 추론을 수행합니다. 데이터 준비 과정에서는 BERT 임베딩을 생성하는 과정을 통해 각 프로필에 대해 결합된 표현을 형성합니다.

- **Performance Highlights**: 실험 결과, ML 모델이 인간 전문가의 결정보다 더 높은 일관성 점수를 기록하며 공정한 결정에서의 신뢰성을 진단합니다. 성별 편향 제거 후, 모델들의 분류 정확도가 향상되거나 유지되는 결과를 도출했습니다. 이러한 결과는 공정성과 성능이 동시에 존재할 수 있음을 검증하며, ML 모델이 높은 정확도를 유지하면서 입학 공정성을 향상시킬 수 있다는 가능성을 제시합니다.



### Efficient Data-aware Distance Comparison Operations for High-Dimensional Approximate Nearest Neighbor Search (https://arxiv.org/abs/2411.17229)
Comments:
          Accepted by VLDB 2025

- **What's New**: 본 연구에서는 기존의 근접 이웃 검색 알고리즘에서 주로 시간 비용을 지배하는 거리 비교 작업(Distance Comparison Operations, DCOs)을 가속화하는 데 중점을 두고 있습니다. 이를 위해 우리는 데이터 분포에 대한 편향 없는 거리 추정(Distance Estimation) 방법인 DADE(Data-Aware Distance Estimation)를 제안합니다. DADE를 통해 낮은 차원 공간에서 정확한 거리를 근사할 수 있으며, 비슷한 거리 추정 방법들보다 더 나은 성능을 제공합니다.

- **Technical Details**: DADE는 정교한 거리 추정을 위해 직교 변환(Orthogonal Transformation)을 활용하여 차원 수(d)가 적응적으로 결정됩니다. 연구팀은 작은 데이터셋에 적합한 거리 추정 문제를 해결하기 위해 통계적 가설 검정(Hypothesis Testing) 접근법을 도입했습니다. 이 방법은 거리 추정과 관련하여 불편함이 없고, 데이터 분포에 기반하여 점진적으로 최적화된 차원의 수를 판단할 수 있도록 합니다.

- **Performance Highlights**: DADE는 HNSW와 IVF와 같은 널리 사용되는 AKNN 검색 알고리즘에 통합되어 기존의 방법론보다 뛰어난 성능을 보여줍니다. 실험 결과, DADE는 불편한 거리 추정에서의 성능 저하 없이 기존 알고리즘보다 검색 속도를 개선할 수 있음을 입증하였습니다. DADE를 사용함으로써 딥러닝과 추천 시스템과 같은 다양한 활용 분야에서 더 효과적인 KNN 검색이 가능해질 것입니다.



### Recommender Systems for Good (RS4Good): Survey of Use Cases and a Call to Action for Research that Matters (https://arxiv.org/abs/2411.16645)
- **What's New**: 이 논문에서는 추천 시스템(recommender systems) 연구 커뮤니티가 사회적 선(good)을 증진하는 영역, 즉 RS4Good에 더 중점을 두어야 한다고 주장하고 있습니다. 많은 추천 알고리즘이 영화 및 전자상거래와 같은 좁은 응용 분야에 집중되었고, 사용자와의 실제 평가가 부족하다는 점이 강조됩니다. 이와 함께, 추천 시스템이 건강한 생활 습관이나 에너지 절약 행동을 촉진할 수 있는 가능성을 제시합니다.

- **Technical Details**: 추천 시스템은 보통 최적의 사용자를 대상으로 정보 과부하를 피하거나 기업의 매출을 증가시키기 위해 설계됩니다. 최근 몇 년 동안 이러한 시스템이 부정적인 영향을 미칠 수 있다는 인식이 높아지며, 다양한 연구 및 이니셔티브가 '책임 있는 추천(responsible recommendation)'이라는 개념 하에 이루어지고 있습니다. 이에 따라 공정성(fairness), 개인 정보 보호(privacy), 해석 가능성(interpretability) 등의 요소가 고려되어야 합니다.

- **Performance Highlights**: 기존의 추천 시스템이 사용자 참여를 극대화하려는 목표로 인해 논란이 있는 콘텐츠를 추천할 수 있다는 우려가 제기되었습니다. 이러한 시스템은 단기적인 비즈니스 지표를 높일 수 있지만 사회적인 문제를 악화시킬 수 있습니다. RS4Good는 단순히 문제를 피하기보다는 사회적 가치를 창출하는 것을 목표로 하며, 경제적 가치를 동시에 추구할 수 있습니다.



### Low-Data Classification of Historical Music Manuscripts: A Few-Shot Learning Approach (https://arxiv.org/abs/2411.16408)
Comments:
          6 pages, The Sixth IEEE international conference on Image Processing Applications and Systems

- **What's New**: 이 논문에서는 역사적 악보의 음악 기호 분류를 위한 자기 지도 학습(framework) 프레임워크를 개발하여 기술과 문화 보존의 교차점을 탐구합니다. 전통적인 Optical Music Recognition (OMR) 방법은 레이블이 있는 데이터 부족으로 어려움을 겪고 있는데, 본 연구는 레이블이 없는 데이터에서 훈련된 신경망 기반(feature extractor) 특징 추출기를 통해 이 문제를 극복하려 합니다. 주요 기여로는 자기 지도 CNN의 크롭 전처리를 최적화하고 SVM, 다층 퍼셉트론(MLP), 프로토타입 네트워크(prototypical networks) 등의 분류 방법을 평가한 점이 있습니다.

- **Technical Details**: 본 연구는 자기 지도 CNN을 기반으로 하여 음악 기호 분류를 위한 특징을 추출하는 방법을 제시합니다. 역사적 악보의 기호들은 크기, 간격, 문서 노후화 등 다양한 변동성을 보이므로, 문서를 세부 섹션(크롭)으로 나누고 이 과정을 슬라이딩 윈도우(sliding window) 방법으로 처리하여 진행합니다. VICReg 방법을 적용하여 CNN은 각 크롭을 두 번 왜곡하여 특징 공간에서 동일한 포인트에 매핑하도록 훈련되며, 이를 통해 역사적 문서에서 발생하는 기호의 가변성에 효과적으로 대처합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제시한 방법은 87.66%의 분류 정확도를 기록하여 AI 기반 방법들이 역사적 음악의 보존 가능성을 확인했습니다. MLP는 다양한 분류 알고리즘 중 가장 우수한 성능을 보였으며, 고차원 데이터와 제한된 라벨 데이터 처리에 있어 강력한 일반화 능력을 나타냅니다. 데이터 증강(data augmentation) 기술을 통해 모델 학습 과정에서 일반화 능력을 더욱 향상시키는 결과를 도출했습니다.



### Stop Playing the Guessing Game! Target-free User Simulation for Evaluating Conversational Recommender Systems (https://arxiv.org/abs/2411.16160)
Comments:
          Work in progress

- **What's New**: 최근 회화 추천 시스템(Conversational Recommender Systems) 분야에서는 CRSs가 인간 사용자와의 대화를 보다 현실적으로 모사하기 위해 노력하고 있습니다. 본 논문에서는 PEPPER라는 새로운 평가 프로토콜을 소개하여, 기존의 타겟 중심 사용자 시뮬레이터의 한계를 극복하고 사용자의 선호도를 보다 정확히 반영할 수 있는 방법을 제시합니다. 이를 통해 CRS와 사용자 간의 복잡한 상호작용을 실현하고, 단순한 추측 게임에서 벗어나 사용자 개인의 선호도를 점진적으로 발견할 수 있는 기회를 제공합니다.

- **Technical Details**: PEPPER는 실제 사용자 상호작용 기록과 리뷰를 활용한 타겟 없는 사용자 시뮬레이터로 구성되어 있습니다. 이 프로토콜은 정량적 및 정성적 측정을 통해 CRS의 선호도 유도 능력을 포괄적으로 평가합니다. 특히, PEPPER에서는 선호도 유도 과정의 네 가지 독특한 측면을 평가할 수 있는 세부적인 측정 기준을 제공합니다. 이는 CRS와 사용자의 대화에서 발생할 수 있는 다양한 변화를 보다 사실적으로 반영할 수 있도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 PEPPER의 검증된 신뢰성을 입증하며, 기존 CRS들이 선호도 유도 및 추천에서 얼마나 효과적으로 성능을 발휘하는지를 철저히 분석합니다. PEPPER는 사용자의 진화하는 선호도를 포착할 수 있는 평가 시스템으로, CRSs의 여러 성능을 분석하는 데 유용한 도구로 자리매김할 수 있습니다. 이로써, CRS의 실제적인 응용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### Ensemble Learning via Knowledge Transfer for CTR Prediction (https://arxiv.org/abs/2411.16122)
- **What's New**: 이 논문에서는 Click-through rate (CTR) 예측을 위한 더 큰 ensemble networks의 활용 가능성을 탐구합니다. 기존 방법들이 보통 2~3개의 서브 네트워크를 사용하는 반면, 본 연구는 더 큰 ensemble의 한계점을 세 가지로 정리합니다: 네트워크 수 증가에 따른 성능 저하, 서브 네트워크 성능의 높은 분산, 그리고 서브 네트워크와 ensemble 예측 간의 큰 차이입니다. 이를 극복하기 위해 Knowledge Distillation (KD)과 Deep Mutual Learning (DML) 기반의 새로운 Ensemble Knowledge Transfer Framework (EKTF)를 제안합니다.

- **Technical Details**: 본 연구는 EKTF를 통해 각 학생 모델(서브 네트워크)의 집단적 의사 결정을 추상적인 교사 모델로 활용하여 보다 효과적인 학습을 유도합니다. 또한, 학생 네트워크 간의 상호 학습을 유도하여 다양한 관점에서 지식을 습득하도록 합니다. 이를 통해 KD와 DML의 개념을 통합하여 교육 및 학습 손실의 비율을 조절하는 새로운 '검토 메커니즘'(examination mechanism)을 설계하였습니다.

- **Performance Highlights**: 다섯 개의 실제 데이터셋에서 실시한 실험 결과, EKTF는 기존 방법들에 비해 효과성, 호환성, 유연성을 입증하였습니다. 실험을 통해 EKTF가 CTR 예측 작업에서의 한계를 극복하고, 보다 나은 성능 향상을 가져올 수 있음을 보여주었습니다. 제공된 코드와 상세한 하이퍼파라미터 구성은 코드가 포함된 URL에서 접근할 수 있습니다.



### ScalingNote: Scaling up Retrievers with Large Language Models for Real-World Dense Retrieva (https://arxiv.org/abs/2411.15766)
- **What's New**: 이 논문에서는 ScalingNote라는 새로운 두 단계의 방법론을 제안하여 LLMs를 통한 밀집 검색(dense retrieval)에서의 확장 가능성을 어떻게 활용할 수 있는지를 탐구합니다. 기존의 검색 모델들은 실시간 쿼리에 대한 응답 속도를 보장하기 위해 모델 크기를 제한하는 경우들이 많았으나, 연구팀은 이 문제를 해결하기 위한 새로운 접근 방식을 제시하였습니다. ScalingNote는 LLM에 기반하여 쿼리 및 문서 타워를 각각 최적화하여 온라인 쿼리 지연 시간을 최소화하면서 성능을 극대화합니다.

- **Technical Details**: ScalingNote는 두 단계로 구성된 학습 방법으로, 첫 번째 단계에서는 동일한 LLM에서 초기화된 두 개의 타워를 훈련하여 밀집 검색의 잠재력을 최대한 활용합니다. 두 번째 단계에서는 평균 제곱 오차 손실(mean squared error loss)과 코사인 유사성(cosine similarity)을 사용하여 쿼리 타워만 증류(distill)하여 온라인 비용을 줄이라는 방법론을 강조합니다. 이 방법론은 실제 산업 환경에서 LLM의 스케일링 법칙을 검증하고 있습니다.

- **Performance Highlights**: ScalingNote는 오프라인 및 온라인 실험을 통해 그 효율성과 효과성을 입증하였으며, 기존의 end-to-end 모델보다 성능이 향상되었습니다. 연구진은 ScalingNote 방법이 효율적으로 더 높은 쿼리 적합도를 달성한다고 보고하였으며, 모든 실험 조건 하에서도 낮은 온라인 지연 시간을 유지한다고 설명하였습니다. 이 방법은 밀집 검색 시스템의 비용 효과적인 확장을 가능하게 하여 산업 응용 분야에서의 실질적인 가치가 기대됩니다.



### Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models (https://arxiv.org/abs/2411.15731)
Comments:
          Accepted by WSDM 2025

- **What's New**: 새로운 연구에서는 CTR(Click-Through Rate) 모델의 융합 설계(fusion design)는 여전히 주목받지 못하고 있으며, 기존의 두 가지 단순한 융합 디자인인 stacked와 parallel이 널리 사용되고 있다고 강조합니다. 이러한 발견에 기반하여, OptFusion이라는 방법이 도입되어 융합 학습(fusion learning) 프로세스를 자동화하고, 연결 학습(connection learning) 및 연산 선택(operation selection)을 통합적으로 수행합니다. 이 논문의 새로운 접근법은 CTR 모델의 성능을 상당히 향상시킬 가능성을 제시합니다.

- **Technical Details**: OptFusion의 프레임워크는 하나의 embedding 컴포넌트, n+1 개의 cross 컴포넌트, n 개의 deep 컴포넌트 및 하나의 출력 컴포넌트로 구성되어 있으며, 이는 실제 융합 연결(search) 탐색 후보 세트로 활용됩니다. OptFusion은 한 번의 학습(one-shot learning) 알고리즘을 사용하여 융합 연결과 연산을 동시에 학습함으로써 융합 디자인에서의 복잡성을 효율적으로 다룹니다. 이는 네트워크 아키텍처 탐색(neural architecture search)과 비교해 보다 효율적인 탐색이 가능하다는 점에서 주요한 차별점을 나타냅니다.

- **Performance Highlights**: OptFusion은 세 가지 대규모 데이터셋에서 실험을 통해 그 효율성과 효과성을 입증하였습니다. 실험 결과는 OptFusion이 기존의 CTR 모델보다 더 높은 성능을 발휘함을 보여주었으며, 다양한 구성요소와 결합되었을 때도 우수한 성능을 유지함을 강조합니다. 이 연구는 CTR 예측의 융합 운영(selecting fusion operations)과 연결 성능(selecting fusion connections)의 중요성을 잘 드러내고 있습니다.



### Class Order Disorder in Wikidata and First Fixes (https://arxiv.org/abs/2411.15550)
- **What's New**: 이 논문에서는 Wikidata의 온톨로지에서 클래스 순서(class order)와 관련된 문제를 분석합니다. 기존 클래스 순서의 위반 및 의심스러운 정보를 SPARQL 쿼리를 통해 평가하고, 이러한 문제를 해결하기 위한 제안을 제시합니다. 기사에서는 수정된 쿼리를 통해 클래스 순서와 관련된 문제를 파악합니다. 이 문제들은 장기적으로 축적되고 있으며, 커뮤니티 참여와 도구 기능 개선을 통해 해결할 수 있습니다.

- **Technical Details**: Wikidata는 2024년 7월 기준으로 1억 개 이상의 아이템을 포함하는 일반 지식 기반으로, 이는 다양한 도메인(예: 생명 과학, 의학)에 걸쳐 있습니다. 논문에서는 Wikidata의 클래스가 고정 순서를 가져야 한다고 기본적으로 정의하고, 클래스가 여러 고정 순서를 가져서는 안 된다는 점을 강조합니다. 쿼리를 통해 발견된 문제들은 실제 오류일 수도, 단순히 강한 증거를 제공할 수도 있으며, 대부분 정확한 수정을 찾기 위해 추가 분석이 필요합니다.

- **Performance Highlights**: Wikidata의 오류는 해소될 기회가 많지만, 쿼리 성능에는 여러 가지 문제점이 있습니다. QLever SPARQL 엔진이 복잡한 쿼리에서 매우 빠르지만, 메모리 할당 오류가 발생할 수 있으며, 적절한 쿼리 구조가 필요합니다. 이러한 성능적 특성은 문제가 발생하기 전보다 깊이 있는 분석을 가능하게 합니다. 또한, 문서에서는 구체적인 사례에서 수정의 영향을 따져 보는 내용을 다룹니다.



### QEQR: An Exploration of Query Expansion Methods for Question Retrieval in CQA Services (https://arxiv.org/abs/2411.15530)
- **What's New**: 이번 연구에서는 커뮤니티 기반 Q&A 서비스에서 발생하는 lexical gap 문제를 해결하기 위해 질문 확장(query expansion) 방법을 사용합니다. 특히, 가장 유사한 질문과 단어를 기반으로 질문을 확장하여 기존 질문 아카이브에서 더 관련성 높은 질문을 찾고자 합니다. 또한 기존의 방법 외에 질문 유사도 기반의 새로운 방법과 선택적 확장을 도입하여 질문의 의도가 변하지 않도록 신경 써서 접근합니다.

- **Technical Details**: 레퍼런스에 따르면, 질문 확장의 첫 번째 방법은 유사한 의미를 지닌 단어들을 사용하여 입력 질문을 확장하는 것입니다. 이를 위해 KL-Divergence를 이용해 확장된 질문과 후보 질문들의 관련성을 계산합니다. 또한, 선택적 확장을 도입하여 모든 단어를 확장하지 않고, 특정 단어만 확장할 때 질문의 본래 의도를 유지할 수 있도록 하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 방법의 최적화 모델은 쿼리 확장 없이 최상의 기준선에 비해 1.8%의 상대적 성능 향상을 보여주었습니다. 이는 플랫폼 사용자가 보다 빠르게 정보 필요에 맞는 답변을 찾을 수 있도록 도와주는 중요한 결과입니다. 이 성과는 주어진 질문과 유사한 질문을 찾는 질문 검색의 효과를 높이는 데 기여합니다.



### Quantitative Analysis of IITs' Research Growth and SDG Contributions (https://arxiv.org/abs/2411.15451)
- **What's New**: 이 연구는 인도 공과대학(IIT)의 연구 성과를 1952년부터 2024년까지 검토하며, 특히 최고 권위의 IIT인 봄베이, 델리, 마드라스, 카라그푸르, 칸푸르에 집중하고 있습니다. 연구 결과는 이러한 기관들이 증가된 자금 지원과 원활한 협력 덕분에 과학 및 기술 분야에서 중요한 기여를 하고 있으며, 특히 IIT 칸푸르는 연구 영향력에서 두각을 나타내고 있습니다.

- **Technical Details**: IIT의 연구 성과는 Scopus에 인덱스된 출판물 기반 분석에 의해 평가되었으며, 이 연구는 SDG 3(건강), SDG 7(청정 에너지), SDG 11(지속 가능한 도시)과 같은 특정 연구 우선순위를 설정합니다. 하지만 IIT 간의 협력처럼 최근 성과에는 제한된 형성이 관찰되며, 새로운 IIT와의 협력이 부족한 점이 지적됩니다.

- **Performance Highlights**: IIT들은 미국, 독일, 영국, 그리고 일본 및 한국과 같은 아시아 국가와의 강력한 국제 협력을 통해 연구 성과를 확장하고 있습니다. IIT 봄베이와 IIT 마드라스는 높은 생산성을 자랑하지만, 논문당 영향력에서는 다소 낮게 평가되고 있으며, 이는 연구의 지속적인 발전을 위한 새로운 전략 모색이 필요함을 시사합니다.



### The Landscape of Data Reuse in Interactive Information Retrieval: Motivations, Sources, and Evaluation of Reusability (https://arxiv.org/abs/2411.15430)
- **What's New**: 이번 연구는 Interactive Information Retrieval (IIR) 분야에서 경험이 풍부한 연구자들과의 심층 인터뷰를 통해 데이터 재사용(data reuse)의 관행을 조사했습니다. 연구자들이 데이터 재사용에 대한 동기와 경험, 그리고 재사용에 대한 우려사항을 파악하여, IIR 커뮤니티 내에서의 데이터 재사용 개선 방향을 제시하고자 했습니다.

- **Technical Details**: 연구자는 21명의 IIR 연구자들과 반구조화된 심층 인터뷰(semi-structured in-depth interviews)를 진행하여, 데이터 재사용과 관련된 동기, 경험, 자료 재사용 가능성 평가 전략을 분석했습니다. 이 과정에서 데이터 제공자, 큐레이터(curator), 그리고 재사용자의 다양한 필요를 반영한 정책, 기준, 역할, 책임을 포함하는 지속 가능한 데이터 재사용 프로세스의 중요성을 강조하였습니다.

- **Performance Highlights**: 본 연구의 실증적 발견은 연구자들이 데이터 재사용에 대한 동기, 재사용 가능성 평가를 위한 접근 방식, 그리고 데이터 재사용 과정에서 직면하는 도전 과제를 이해하는 데 기여합니다. 이 논문은 사용자 생성 데이터(user-generated data)와 연구 자원의 평가에 대한 논의를 풍부하게 하고, 커뮤니티 차원에서 데이터 재사용 문화와 기준을 촉진하는 데 도움을 주는 내용을 담고 있습니다.



### The Decoy Dilemma in Online Medical Information Evaluation: A Comparative Study of Credibility Assessments by LLM and Human Judges (https://arxiv.org/abs/2411.15396)
- **What's New**: 이번 연구는 AI와 대규모 언어 모델(LLMs)이 정보 신뢰도 평가 작업에서 인지 편향(cognitive bias)에 어떻게 취약한지를 탐구합니다. 기존의 인간 평가자와의 비교를 통해, LLM이 COVID-19 관련 의학정보 (mis)정보 평가에서 인간보다 더 높은 편향 수준을 보인다는 것을 발견했습니다. 이러한 연구는 AI 도구의 합리성에 대한 일반적인 가정에 의문을 제기하며, LLM의 신뢰도 판단에서의 위험성을 강조합니다.

- **Technical Details**: 연구는 LLM과 인간 평가자의 신뢰도 평가를 비교하기 위해 조작 조건으로서 Decoy Effect를 활용했습니다. Decoy Effect는 열등한 옵션이 두 개의 유사한 옵션 사이에서 개인의 선호에 영향을 미치는 현상이며, 본 연구에선 코로나 치료웹 페이지 평가를 위한 크라우드소싱 사용자 실험을 통해 이 현상을 검증했습니다. LLM 기반 실험을 통해 다양한 모델의 평가 결과를 수집하고, 인간 평가자의 기준을 설정하여 모델의 신뢰도 판단을 측정하였습니다.

- **Performance Highlights**: 연구 결과는 더 크고 최신의 LLM이 정보의 신뢰성과 정확성을 평가하는 데 더 높은 일관성을 보이는 반면, 오히려 잘못된 정보에 더 높은 점수를 부여하는 경향이 있음을 보여줍니다. 디코이 효과는 LLM 판단에서 더 뚜렷하게 나타났으며, 이는 LLM의 인지적 편향을 드러냅니다. 연구는 LLM의 평가가 인간보다 더 많은 편향을 내포하고 있음을 논증하며, AI 판단의 오류를 줄이기 위한 필요성을 강조합니다.



### Preliminary Evaluation of the Test-Time Training Layers in Recommendation System (Student Abstract) (https://arxiv.org/abs/2411.15186)
Comments:
          To be published in AAAI-25 Student Abstract and Poster Program

- **What's New**: 이 논문은 추천 시스템의 성능을 향상시키기 위해 Test-Time Training (TTT) 레이어의 적용과 효과를 탐구합니다. 새로운 모델인 TTT4Rec를 개발하여 TTT-Linear를 특징 추출 레이어로 활용했습니다. 다양한 데이터셋에서의 테스트 결과, TTT4Rec는 유사한 환경의 다른 기본 모델들과 비교하여 동등하거나 더 나은 성능을 보였습니다.

- **Technical Details**: TTT의 핵심 아이디어는 자기지도 학습(self-supervised learning)을 활용해 과거의 맥락(x1,…,xt)을 압축하여 숨겨진 상태(st)로 변환하는 것입니다. 모델 업데이트는 각 입력 시퀀스에 대해 고유한 가중치(W1,…,WT)를 학습하며, 이러한 방식은 테스트 중에도 적용됩니다. TTT4Rec의 아키텍처는 사용자 클릭 시퀀스를 벡터로 인코딩하고, 각 사용자의 클릭 행동을 기반으로 예측을 수행하는 구조로 되어 있습니다.

- **Performance Highlights**: TTT4Rec는 Amazon Beauty, Amazon Electronics, MovieLens-1M 같은 세 가지 널리 사용되는 데이터셋에서 평가되었습니다. 교육 단계 동안은 긍정적 사례와 부정적 사례 간의 비율을 1:4로 설정하고, 테스트 단계에서는 1:99로 설정하여 실제 추천 시나리오에 더 가까운 평가를 진행했습니다. 이러한 설정을 통해, 모델의 성능 평가가 더욱 견고하게 이루어졌음을 보였습니다.



### TIMBRE: Efficient Job Recommendation On Heterogeneous Graphs For Professional Recruiters (https://arxiv.org/abs/2411.15146)
- **What's New**: 이 논문에서는 구직 추천 시스템에서의 도전 과제를 해결하기 위한 새로운 접근법인 TIMBRE (Temporal Integrated Model for Better REcommendations)를 제안합니다. 기존의 협업 필터링(Collaborative Filtering) 방식에서는 사용자와 아이템 정보의 결합이 부족하여 성능이 저하되는 문제를 지적하며, 여러 출처에서 받은 정보를 활용하여 이종 그래프(Heterogeneous Graph)로 통합한다고 설명합니다. 이는 구직 시장의 역동성을 고려하여 사용자와 아이템 데이터를 효율적으로 추천하는 방식을 개발하기 위한 것입니다.

- **Technical Details**: TIMBRE는 구직 추천을 위해 이종 그래프를 구성하고, 시간적 요소를 통합하여 추천을 최적화합니다. 이 시스템은 그래프 신경망(Graph Neural Network, GNN)을 활용하여 사용자-아이템 쌍에 대한 점수를 생성하는데, 이때 그래프의 구조적 요소를 강조하여 콜드 스타트 문제를 해결합니다. 또한, 시간 의존적인 훈련(Timing-dependent Training) 방식을 통해 훈련 편향을 방지하도록 설계되어 있습니다.

- **Performance Highlights**: TIMBRE는 전통적인 추천 시스템 평가 지표를 사용하여 성능을 평가하며, 현존하는 다른 추천 시스템과 비교하여 명확한 우위를 보여줍니다. 기존 연구들은 종종 그래프 기반 추천 시스템에서 이러한 평가 지표를 간과하는 경향이 있었으나, TIMBRE는 이를 통해 구직 추천의 효과성을 입증하고 있습니다. 마지막으로, TIMBRE는 통합된 시간적 정보를 활용하여 추천의 질을 높이는 데 기여하고 있습니다.



### Context Awareness Gate For Retrieval Augmented Generation (https://arxiv.org/abs/2411.16133)
- **What's New**: 이번 연구에서는 Retrieval Augmented Generation (RAG) 시스템에서의 관련 없는 정보 검색이 모델 출력 품질에 미치는 부정적인 영향을 강조합니다. 이를 해결하기 위해 Context Awareness Gate (CAG) 아키텍처를 제안하며, 이는 사용자 쿼리가 외부 컨텍스트 검색을 필요로 하는지에 따라 입력 프롬프트를 동적으로 조정합니다. 또한, 통계적이고 LLM 독립적인 Vector Candidates 방법을 소개하여 RAG 시스템의 컨텍스트 검색 프로세스를 개선하는 데 기여합니다.

- **Technical Details**: CAG 아키텍처는 쿼리 변환 및 동적 프롬프트 조정을 통해 RAG 파이프라인의 신뢰성을 강화합니다. Vector Candidates 방법은 문서 세트의 각각에 대해 가상 쿼리를 생성하고 임베딩 분포의 유사성을 계산함으로써 쿼리 분류를 수행합니다. 이 과정에서, 쿼리 변환이 이루어져야만 보다 최적화된 데이터 검색이 가능하며, CAG 시스템은 이를 활용해 LLM이 내부 지식을 기반으로 사용자 쿼리에 응답할 수 있도록 합니다.

- **Performance Highlights**: 우리의 연구는 RAG 시스템에서 널리 사용되는 전통적인 방법과 비교하여, CAG 아키텍처가 출력 품질을 획기적으로 향상시키는 것을 보여줍니다. Vector Candidates 방법은 각각의 문서에 대해 효과적인 유사성을 평가하여, 불필요한 컨텍스트 검색을 피함으로써 효율성을 높입니다. 결과적으로, 제안된 접근 방식을 통해 데이터 검색의 정확도 및 관련성을 개선할 수 있으며, 이는 다양한 분야에서 QA 시스템의 성능 상승으로 이어집니다.



### Proceedings of the 6th International Workshop on Reading Music Systems (https://arxiv.org/abs/2411.15741)
Comments:
          Proceedings edited by Jorge Calvo-Zaragoza, Alexander Pacha and Elona Shatri

- **What's New**: 이번 논문은 제6회 국제 음악 읽기 시스템 워크숍(WoRMS)의 논문집을 소개합니다. 특히, Optical Music Recognition 분야에서 음악을 읽기 위한 시스템을 개발하는 연구자들과 도서관원이나 음악학자와 같은 다른 연구자들을 연결하는 데 초점을 맞추고 있습니다. 올해에는 총 22개 논문이 제출되어 15개가 승인되었으며, 저자들이 요청한 몇몇 논문은 생략되었습니다.

- **Technical Details**: 워크숍에서 다룰 주요 주제로는 음악 읽기 시스템, Optical Music Recognition, 데이터셋 및 성능 평가, 음악 악보의 이미지 처리, 작가 식별, 음악 악보의 저작 및 프레젠테이션 시스템 등이 포함됩니다. 다중 모드 시스템, 음악 작성을 위한 새로운 입력 방법, 웹 기반 음악 정보 검색 서비스와 같은 최신 기술적 요소도 고려되었습니다.

- **Performance Highlights**: 이번 년도에는 온라인으로 진행되어 전 세계의 참가자들이 쉽게 참여할 수 있는 기회를 제공했습니다. 저자들에게는 WoRMS의 품질 기준을 충족하도록 수정할 수 있는 피드백을 제공하였으며, 차기 워크숍에 다시 제출할 것을 기대하고 있습니다. 또한 GitHub 조직과 YouTube 채널을 통해 연구 관련 자료를 공유하고, 과거 세션의 기록을 조회할 수 있게 하는 기회를 제공합니다.



### Tackling Data Heterogeneity in Federated Time Series Forecasting (https://arxiv.org/abs/2411.15716)
- **What's New**: 이 논문에서는 데이터 이질성(heterogeneity) 문제를 해결하기 위해 새로운 프레임워크인 Fed-TREND를 제안합니다. Fed-TREND는 보조 지식 운반체로서 유의미한 합성 데이터(synthetic data)를 생성하여, 서로 다른 장치에서 발생하는 시간 시계열 데이터의 이질성을 효과적으로 다룹니다. 또한, 기존의 중앙 집중식 훈련 방법을 탈피하고, 프라이버시를 보호하면서 협업 모델 훈련을 가능하게 합니다. 이 프레임워크는 다양한 시간 시계열 예측 모델과 호환되고, 기존의 연합 학습(frderated learning) 프레임워크에 원활하게 통합될 수 있습니다.

- **Technical Details**: Fed-TREND는 두 가지 유형의 합성 데이터를 생성하여 이질성 문제를 해결하는 데 초점을 맞춥니다. 첫 번째 합성 데이터는 클라이언트가 업로드한 모델 업데이트의 분포 정보를 캡처하여 클라이언트의 로컬 훈련 합의를 향상시킵니다. 두 번째 합성 데이터는 글로벌 모델 업데이트 궤적(global model update trajectories)에서 장기적 영향을 추출하여, 집계 후 글로벌 모델을 개선하는 데 사용됩니다. 이러한 접근 방식은 클라이언트에서도 복잡한 관계를 포착할 수 있도록 도와줍니다.

- **Performance Highlights**: 다양한 연합 학습 기법과 널리 사용되는 네 가지 시간 시계열 예측 모델을 통합한 extensive 실험을 수행했습니다. 이 실험들은 Fed-TREND가 모든 시간 시계열 예측 데이터셋에서 연합 예측 성능을 유의미하게 향상시키는 결과를 보여줍니다. 연구 결과는 Fed-TREND의 효율성과 범용성을 입증하며, 실제 여러 산업 분야에서의 적용 가능성을 강조합니다.



New uploads on arXiv(cs.CV)

### Video-Guided Foley Sound Generation with Multimodal Controls (https://arxiv.org/abs/2411.17698)
Comments:
          Project site: this https URL

- **What's New**: MultiFoley는 비디오 유도 Foley 사운드 생성 모델을 소개하며, 텍스트, 오디오 및 비디오를 통한 다중 모드 조건화(multi-modal conditioning)를 지원합니다. 이 모델은 사용자가 무음 비디오와 텍스트 프롬프트를 기반으로 다양한 사운드를 생성할 수 있도록 합니다. 특히, MultiFoley는 인터넷에서 수집된 저품질 오디오와 전문 사운드 효과 라이브러리(SFX)를 결합한 공동 훈련 방식으로 고품질 오디오(48kHz)를 생성합니다.

- **Technical Details**: MultiFoley는 확산 변환기(diffusion transformer), 고품질 오디오 오토인코더(autoencoder), 비디오-오디오 동기화용 동결 비디오 인코더를 포함하고 있습니다. 이 모델은 사용자가 음향 디자인을 위한 세밀한 조정을 할 수 있도록 하며, 텍스트 프롬프트를 통해 사용자가 원하는 사운드 효과를 생성하는 데 필요한 유연한 제어 기능을 제공합니다. 또한, 각기 다른 오디오 및 비디오 예제로부터 조건화가 가능합니다.

- **Performance Highlights**: MultiFoley는 다양한 조건 입력에 걸쳐 동기화된 고품질 사운드를 성공적으로 생성하고 있으며, 기존 방법들보다 더 나은 성능을 보여줍니다. 자동 평가 및 인간 검토를 통해, 모델은 CROSS-MODAL alignment가 향상되었으며, 다양한 응용 프로그램으로의 확장이 가능합니다. 모델의 성능 향상은 청중의 경험을 높이는 데 기여합니다.



### StableAnimator: High-Quality Identity-Preserving Human Image Animation (https://arxiv.org/abs/2411.17697)
- **What's New**: StableAnimator는 첫 번째 ID 보존 비디오 확산 프레임워크로, 참조 이미지와 포즈 시퀀스를 기반으로 고품질 비디오를 생성하는 데 중점을 두고 있습니다. 이는 기존의 성능 저하 문제를 해결하고 얼굴 품질을 향상시킬 수 있는 혁신적인 방법으로, Hamilton-Jacobi-Bellman (HJB) 방정식을 활용하여 최적의 ID 보존을 달성합니다. 이 방식은 비디오의 충실도를 저해하지 않고 정체성을 유지하는 데 기여합니다.

- **Technical Details**: StableAnimator는 전통적인 비디오 확산 모델에서 ID 정보를 성공적으로 통합하기 위해 설계된 모듈을 포함합니다. 특히, 글로벌 콘텐츠 인식 Face Encoder와 분포 인식 ID Adapter를 도입하여 비디오 생성 과정에서 공간적 분포와 ID 정보를 동시적으로 고려합니다. HJB 방정식을 활용해 각 디노이징 단계에서 잠재 변수를 업데이트함으로써 얼굴 품질을 향상시키고, 후처리 도구에 대한 의존성을 줄입니다.

- **Performance Highlights**: 실험을 통해 StableAnimator는 CSIM 지표에서 ControlNeXt에 비해 47.1% 뛰어난 성능을 보이며, 다양한 벤치마크 데이터셋에서 최상의 결과를 기록하였습니다. 특히, 얼굴과 몸의 왜곡 문제가 두드러지는 기존의 모델들과 달리, StableAnimator는 포즈에 기반한 실제적인 애니메이션을 생성하며 ID 일관성을 유지하는 데 성공하였습니다.



### ScribbleLight: Single Image Indoor Relighting with Scribbles (https://arxiv.org/abs/2411.17696)
- **What's New**: 이번 논문에서는 이미지 기반의 실내 조명 수정(Image-based relighting) 작업을 위한 새로운 생성 모델인 ScribbleLight를 소개합니다. ScribbleLight는 사용자가 제공하는 낙서(scribbles)를 통해 조명 효과를 세밀하게 조정할 수 있는 기능을 지원하여 사용자가 원하는 조명을 직관적으로 제어할 수 있도록 합니다. 이는 기존의 조명 수정 기술들이 잘 다루지 못했던 지역적인 세부사항을 제어하는 혁신적인 방법입니다.

- **Technical Details**: ScribbleLight의 주요 기술적 혁신은 내부 색상과 질감을 보존하면서 실내 이미지를 재조명할 수 있는 Albedo-conditioned Stable Image Diffusion 모델입니다. 이 모델은 그림자 맵 및 낙서 주석(normal map and scribble annotations)을 이용하여 기하학적 특성을 보존하면서 조명 효과를 적용할 수 있는 ControlNet 아키텍처를 사용합니다. 이러한 구조를 통해 사용자는 낙서를 통해 간단하게 조명의 변화를 나타낼 수 있습니다.

- **Performance Highlights**: ScribbleLight는 다양한 조명 효과를 생성하는 능력을 보여주며, 실험 결과 기존의 베이스라인 방법들보다 성능이 우수함을 입증하였습니다. 특히, 사용자가 제시한 알베도(albedo)의 색상과 질감을 잘 유지하면서 조명 효과를 적용할 수 있습니다. 규칙적인 낙서를 통한 세밀한 조정이 가능하므로, 실내 장면에서의 전문가 수준의 조명 조정이 이루어질 수 있습니다.



### GenDeg: Diffusion-Based Degradation Synthesis for Generalizable All-in-One Image Restoration (https://arxiv.org/abs/2411.17687)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 GenDeg라는 새로운 생성 모델을 통해 다양한 이미지 악화 상태를 생성하여 데이터셋의 부족한 다양성을 해결하고자 합니다. 이미지 복원 모델은 GenDS 데이터셋을 통해 훈련되며, 이는 기존 데이터셋과 결합되어 약 750k 샘플로 구성됩니다. GenDeg는 고품질의 악화된 이미지를 생성하는 능력을 통해 이미지 복원 모델의 일반화 성능을 향상시킵니다. 이를 통해 실제 환경에서의 복원 성능이 크게 개선된다는 결과를 보여주고 있습니다.

- **Technical Details**: GenDeg는 latent diffusion models (LDMs)을 기반으로 하며, 다양한 유형의 악화를 수용하도록 설계되었습니다. 이 모델은 텍스트 프롬프트, 클린 이미지, 및 악화 수준을 조건으로 하여 다양한 악화된 이미지를 생성합니다. 연구진은 120k 개의 클린 이미지에서 550k 개의 악화된 이미지를 생성하여, 이를 통해 생성한 GenDS 데이터셋을 개발하였습니다. GenDS 데이터셋은 여섯 가지 악화 유형을 아우르는 이미지를 포함하여 실제 환경에서의 복원 모델의 일반화 성능을 높입니다.

- **Performance Highlights**: 연구 결과, GenDS 데이터셋으로 훈련된 이미지 복원 모델이 기존 데이터셋만으로 훈련된 모델에 비해 Out-of-Distribution (OoD) 성능에서 현저한 향상을 보여주었습니다. NAFNet, PromptIR 및 새로 제안된 Swin Transformer 기반 모델을 포함한 세 가지 모델을 훈련한 결과, 이러한 모델들은 GenDS를 활용하여 일반화 성능이 크게 개선되었습니다. 이러한 연구는 AIOR 및 합성 데이터 생성을 위한 새로운 방향성을 제시합니다.



### Rethinking Token Reduction in MLLMs: Towards a Unified Paradigm for Training-Free Acceleration (https://arxiv.org/abs/2411.17686)
- **What's New**: 이 연구에서는 Heavy Multimodal Large Language Models (MLLMs)의 추론을 가속화하기 위한 새로운 접근 방식을 제안합니다. 기존의 training-free token reduction 방법들이 상호 연결되어 있으며, 그 효과와 비교가 모호하다는 점을 지적합니다. 따라서 "filter-correlate-compress"라는 통합된 패러다임을 통해 토큰 감소 과정을 세 가지 단계로 나누어 보다 명확한 구현과 성능 개선을 목표로 합니다.

- **Technical Details**: 제안된 패러다임은 MLLMs의 토큰 감소 문제를 해결하기 위해 개발되었으며, 다양한 방법들을 세 가지 주요 단계로 나누어 명확히 정리합니다. 특히, FiCoCo라는 세 가지 상호 보완적인 변형을 도입하여 MLLM 추론의 여러 단계에서 토큰을 감소시키는 타겟 전략을 구현합니다. 이 방법은 전체적인 FLOPs를 감소시키면서도 성능 저하를 최소화하는 접근을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 10개의 벤치마크에서 최대 82.4%의 FLOPs 감소를 달성하며, 기존의 training-free 방법들보다 우수한 성능을 보였습니다. 특히 FiCoCo 시리즈는 LLaVA-1.5-7B와 비교하여 약 17.6%의 계산 비용으로도 유사한 성능을 유지하며, GPU 메모리 사용량도 약 67.6%로 줄였습니다. 이러한 결과는 FiCoCo가 MLLMs의 효율성 및 정확성 간의 최적 균형을 달성했음을 나타냅니다.



### SketchAgent: Language-Driven Sequential Sketch Generation (https://arxiv.org/abs/2411.17673)
Comments:
          project page: this https URL

- **What's New**: 이번 연구에서 소개된 SketchAgent는 대화 기반의 상호작용을 통해 스케치를 생성, 수정 및 개선할 수 있는 혁신적인 방법입니다. 이 방법은 추가적인 훈련 없이 멀티모달 대형 언어 모델(LLMs)의 순차적인 특성과 풍부한 선행 지식을 활용합니다. 특히, 사용자가 입력하는 문자열 기반의 액션을 통해 벡터 그래픽스로 변환된 후 픽셀 캔버스에 스케치가 렌더링됩니다.

- **Technical Details**: 이 시스템은 Claude3.5-Sonnet 모델을 기본으로 사용하며, 사용자 스케치는 평균 20초 내에 완성됩니다. 각 스테로크(stroke)는 협력 모드에서 약 8초가 소요되며, 사용자가 생성한 스케치는 SVG 형식으로 제공되어 추가 편집이 가능합니다. 또한, 다양한 개념을 스케치할 수 있도록 50개의 범주를 선정하여 텍스트 조건에 따른 스케치 생성 성능을 정량적으로 분석하였습니다.

- **Performance Highlights**: SketchAgent는 50개의 다양한 텍스트 개념에 대해 스케치를 생성할 수 있으며, 여러 실험을 통해 높은 다양성을 입증하였습니다. 특히 과학 개념과 도표, 주요 랜드마크와 같은 범주에서 생성된 스케치에서 만족스러운 결과를 보여주었고, 더불어 혼동 행렬을 통해 인식 패턴을 분석하여 일부 범주에서의 정확성을 높였습니다. 결국, 이 연구는 스케치 생성을 위한 새로운 접근 방식을 제시하며, 상용 모델과 열려있는 모델의 발전 가능성을 보여줍니다.



### DROID-Splat: Combining end-to-end SLAM with 3D Gaussian Splatting (https://arxiv.org/abs/2411.17660)
- **What's New**: 이 논문에서는 최첨단 (SotA) SLAM 시스템의 성능을 개선하기 위해 최신 3D Gaussian Splatting 기술을 기반으로 한 Renderer를 통합한 새로운 SLAM 시스템인 DroidSplat를 소개합니다. 이 시스템은 기존의 SLAM 시스템의 약점을 극복하고 Monocular (단안) 비디오에서의 추적 성능을 크게 향상시킵니다.

- **Technical Details**: DroidSplat는 end-to-end Tracker를 기반으로 하며 다수의 현대 SLAM 시스템의 빌딩 블록을 병렬로 실행할 수 있도록 구현되어 있습니다. 이로 인해 일반 소비자 GPU에서 빠른 추론 (inference)이 가능해졌습니다. 최신 Monocular depth prediction 및 카메라 캘리브레이션이 결합되어, 카메라 내부 파라미터가 알려지지 않은 상황에서도 강력한 성능을 발휘합니다.

- **Performance Highlights**: DroidSplat는 일반적인 SLAM 벤치마크에서 우수한 추적 및 렌더링 결과를 달성했습니다. 이 시스템은 속도, 견고함 (robustness), 정확성 (accuracy) 간의 최적의 균형을 이루었으며, 특히 단안 비디오 처리에서 두드러진 성과를 나타냈습니다.



### SAMWISE: Infusing wisdom in SAM2 for Text-Driven Video Segmentation (https://arxiv.org/abs/2411.17646)
- **What's New**: 본 연구에서는 Referring Video Object Segmentation (RVOS) 발달에 대한 새로운 접근 방식을 제안합니다. 기존의 RVOS 방법들이 단편적인 영상 처리로 글로벌 컨텍스트를 상실하는 문제를 해결하기 위해, 자연어 이해와 시간적 모델링을 결합하여 SAM2 모델을 개선하는 SAMWISE를 소개하였습니다. 이는 비디오 스트리밍 상황에서도 효과적으로 작동할 수 있도록 하며, 채택된 자연어 명령에 대한 객체 분할을 강화합니다.

- **Technical Details**: SAMWISE는 Cross-Modal Temporal Adapter (CMT)라는 새로운 어댑터 모듈을 도입하여 시각적 및 언어적 모달리티 간의 상호 작용을 가능하게 하며, 시간적 단서를 시각적 특징에 인코딩합니다. 이를 통해 SAM2의 Mask Decoder는 입력된 텍스트 임베딩을 바탕으로 최종 분할 마스크를 생성합니다. 이 과정에서 SAM2의 원래 가중치를 조정하지 않으며, 외부 모델에 의존하지 않고 자연어 지식을 통합하는 최초의 end-to-end 솔루션으로 자리잡습니다.

- **Performance Highlights**: 제안된 SAMWISE 방법은 Ref-Youtube-VOS, Ref-DAVIS 등 기존 RVOS 벤치마크에서 최첨단 성능을 달성했으며, 더 도전적인 MeViS 데이터셋에서도 우수한 결과를 보였습니다. 추가된 매개변수는 단 4.2M에 불과하여 원래의 SAM2 성능을 유지하면서도 훌륭한 성능 향상을 이루었습니다. 이 코드와 학습된 모델은 수용 후 공개될 예정으로, 연구 커뮤니티에 큰 기여를 할 것으로 기대됩니다.



### Accelerating Vision Diffusion Transformers with Skip Branches (https://arxiv.org/abs/2411.17616)
Comments:
          17 pages, 8 figures

- **What's New**: 본 연구에서는 Diffusion Transformers (DiT) 아키텍처의 효율성을 높이기 위해 Skip-DiT라는 새로운 모델 구조를 제안합니다. 이 구조는 feature caching을 개선하고, denoising 과정에서의 feature variance 문제를 해결하기 위해 skip branches를 추가합니다. 또한, Skip-Cache 메커니즘을 통해 inference 시간 동안 feature를 효율적으로 캐시하여 성능과 속도를 향상시킵니다.

- **Technical Details**: Diffusion 모델은 노이즈에서 고해상도 이미지를 생성하는 원리를 기반으로 하며, Forward 및 Backward diffusion을 포함합니다. 연구에서는 DiT의 feature dynamics를 분석하여, timesteps간의 feature smoothness가 caching 효율성에 미치는 영향을 확인하였습니다. Skip-DiT 구조는 이러한 feature smoothness를 개선하는 데 중요한 역할을 하며, Skip-Cache는 이러한 smooth한 feature를 활용하여 inference 과정을 가속화합니다.

- **Performance Highlights**: 실험 결과, Skip-DiT는 기존 모델에 비해 최대 1.5배 향상된 속도를 자랑하며, 2.2배의 속도 향상도 추측한 경우 오차 범위 내에 있음을 보여줍니다. 이 모델은 비디오 및 이미지 생성을 포함한 다양한 생성을 통해 일관되게 성능을 입증하였습니다. Skip-DiT와 Skip-Cache는 qualitative 및 quantitative 평가에서 기존 캐시 메커니즘보다 뛰어난 성능을 발휘하였습니다.



### Modality-Incremental Learning with Disjoint Relevance Mapping Networks for Image-based Semantic Segmentation (https://arxiv.org/abs/2411.17610)
Comments:
          Accepted at WACV 2025

- **What's New**: 이 논문은 자율주행 관련 최신 연구로, 다양한 센서에서의 지속적 학습을 위한 모달리티-증가 학습(modality-incremental learning, MIL) 개념을 제안합니다. 이는 기존의 증가 학습 패러다임을 대조하여 새로운 센서의 모달리티를 학습하는 동시에 기존의 성능을 유지하기 위한 새로운 접근 방식을 다룹니다. 특히, 이 연구는 Relevance Mapping Network (RMN)를 수정하여 다양한 모달리티의 학습을 가능한 한 분리된 맵 للتقليل합니다.

- **Technical Details**: 이 논문은 지속적 학습(Context of Continual Learning)에서 발생하는 대규모 도메인 변화의 문제를 다루고 있습니다. 제안된 DRMNs(Disjoint Relevance Mapping Networks)는 서로 다른 작업을 위한 네트워크의 파라미터의 서로 다른 부분 집합을 사용하여 이러한 문제를 해결하는 데 중점을 둡니다. 이를 통해 이전의 지식을 잃지 않도록 하면서 새로운 정보를 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 여러 센서 데이터 세트에서 정확도, 망각 비율(forgetting), 네트워크 활용도(network utilization)에 있어 유의미한 개선을 보여줍니다. 특히, 공유 연결을 방지함으로써 엄격한 지속적 학습 프레임워크 내에서 망각 문제를 효과적으로 완화합니다. 이를 통해 자율주행의 안전성을 높이고 더 나은 환경 인식을 위한 기술적 기반을 제공합니다.



### HyperSeg: Towards Universal Visual Segmentation with Large Language Mod (https://arxiv.org/abs/2411.17606)
- **What's New**: 이 논문은 이미지와 비디오 인식을 위한 유니버설 분할 모델인 HyperSeg를 제안합니다. HyperSeg는 Visual Large Language Models (VLLMs)를 기반으로 하여 픽셀 수준에서 복잡한 추론을 수행할 수 있는 능력을 갖추고 있습니다. 기존의 통합 분할 방법들이 이미지와 비디오 시나리오에 적응하는 데 한계를 보인 반면, HyperSeg는 일반적인 분할뿐만 아니라 복잡한 추론 태스크를 처리하는 데 강력한 성능을 보여줍니다.

- **Technical Details**: HyperSeg는 혼합 엔터티 인식 및 미세 조정된 시각 인식 모듈을 포함하여 VLLM의 인식 능력을 극대화합니다. 또한, VLLM의 시각적 세부 사항을 효율적으로 캡처하기 위해 Fine-grained Visual Perceiver (FVP) 모듈을 활용합니다. 타임 어댑터를 통해 HyperSeg는 비디오를 포함한 다양한 시나리오에서 시간 정보를 포괄적으로 이해할 수 있습니다.

- **Performance Highlights**: 실험 결과, HyperSeg는 다양한 분할 벤치마크에서 우수한 성능을 나타내며, 일반 및 복잡한 추론 벤치마크에서 모두 뛰어난 결과를 기록하였습니다. 이로 인해 HyperSeg는 여러 작업 간의 상호 영향을 탐색하면서 다양한 시각 및 태스크 유형을 포함한 커다란 가능성을 보여줍니다. 또한, 논문에서는 HyperSeg의 코드도 공개되고 있어 실험적 적용이 용이합니다.



### Distractor-free Generalizable 3D Gaussian Splatting (https://arxiv.org/abs/2411.17605)
- **What's New**: DGGS(따라하기 쉬운 일반화된 3D Gaussian Splatting) 프레임워크는 훈련 및 추론 단계에서 방해 요소가 가득한 데이터에 대한 일반화된 3DGS의 강화 및 장면 간 응용 능력을 확장하는 혁신적인 접근 방식입니다. 이 연구에서는 장면 불가지론적(reference-based) Mask Prediction과 복원 방법론을 통해 방해 요소를 효과적으로 다루며, 효율적인 훈련 뷰 선택 전략을 도입했습니다. 이러한 새로운 방법은 방해 요소 예측의 정확성과 훈련 안정성을 개선하며, 테스트 단계에서 발생하는 공백과 인공물을 해결하기 위한 두 단계의 추론 구조도 제안하고 있습니다.

- **Technical Details**: DGGS는 방해 요소를 제거하는 훈련 패러다임과 참조 기반 Mask Refinement 모듈을 통해 안정적인 훈련을 목표로 합니다. 이를 통해 서로 다른 장면을 처리할 때 안정적인 참조 렌더링을 활용하여 잘못 식별된 방해 요소 지역을 제거합니다. 또한, 방해 요소가 예측된 참조의 점수 매김 메커니즘을 통해 세밀한 3DGS 재구성을 위한 최소한의 방해 요소 참조 선택이 가능하게 합니다.

- **Performance Highlights**: DGGS는 방해 요소가 많은 데이터셋에서 실험을 통해 기존의 방해 요소 없는 모델에 비해 일반화 능력을 향상시키는 것을 입증했습니다. 제안하는 장면 불가지론적 Mask 추정 방식은 기존 장면 특정 방식에 버금가는 정확성을 보이며, 방해 요소 관련 문제를 효과적으로 완화하는 동시에 일반화 능력을 높였습니다. 궁극적으로 DGGS는 방해 요소가 있는 환경에서도 효과적인 3D 재구성을 가능하게 하는 새로운 기술적 기초를 마련합니다.



### VideoDirector: Precise Video Editing via Text-to-Video Models (https://arxiv.org/abs/2411.17592)
Comments:
          15 figures

- **What's New**: 본 논문에서는 기존의 텍스트-영상(T2V) 편집 방식의 문제점을 분석하고 이를 해결하기 위한 새로운 방법인 Spatial-Temporal Decoupled Guidance (STDG) 와 다중 프레임 Null-text 최적화 전략을 제안합니다. 이러한 방법은 텍스트-이미지(T2I) 모델의 한계를 극복하고, 영상 편집에서의 정확성, 모션 부드러움, 사실성 등을 개선하는 데 중점을 두고 있습니다. 실험 결과, 제안한 방법은 최신 T2V 모델의 강력한 생성 능력을 효과적으로 활용하여 눈에 띄는 성능 향상을 보여줍니다.

- **Technical Details**: 제안된 방법은 입력 비디오의 피벗 전환(pivotal inversion) 및 크로스 주의(control) 전략을 통해 수행됩니다. STDG는 추가적인 시간 정보를 제공하고 다중 프레임 Null-text 임베딩을 통해 비디오의 공간-시간 정보를 좀 더 효과적으로 분리합니다. 또한, 자가 주의(self-attention) 제어를 통해 유사한 편집 결과를 유지하며, 원본 콘텐츠에 대한 신뢰도를 높이는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 VideoDirector 방법은 기존의 T2I 기반 영상 편집 방법보다 사실성, 모션 부드러움 및 정확도 면에서 크게 향상된 결과를 나타냅니다. 피벗 전환 및 편집 과정에서 원본 콘텐츠의 보존력이 뛰어나며, 실제 물리 법칙에 부합하는 사실적인 세부 묘사가 가능함을 보여주었습니다. 이는 T2V 모델의 강력한 생성 능력을 적절히 활용한 결과입니다.



### Pre-training for Action Recognition with Automatically Generated Fractal Datasets (https://arxiv.org/abs/2411.17584)
- **What's New**: 이번 연구에서는 비디오 도메인으로 합성 데이터의 활용을 확장하여 행동 인식( action recognition) 작업에 적용합니다. 우리는 프랙탈 기하학(fractal geometry)을 사용하여 짧은 합성 비디오 클립의 대규모 데이터 세트를 자동으로 생성하는 방법을 제안합니다. 이러한 자동 생성된 비디오 클립은 다양한 특성으로 특징지어지며, 이는 프랙탈의 복잡한 다중 스케일 구조를 생성하는 고유한 능력에서 비롯됩니다.

- **Technical Details**: 본 연구는 2D CNNs를 사전 훈련하기 위해 합성 이미지 데이터 세트를 생성한 기존 연구를 바탕으로 합니다. 프랙탈은 간단하게 구현 가능하면서도 다양성 있는 이미지를 생산할 수 있는 매력적인 속성을 지니고 있습니다. 데이터 세트 생성 시, 우리는 실제 비디오의 근본적인 속성인 주기적 운동, 무작위 배경 및 카메라 변위를 신중하게 모방하여 합성 비디오와 실제 비디오 간의 도메인 격차를 줄입니다.

- **Performance Highlights**: 제안한 방법은 HMDB51 및 UCF101과 같은 기존 행동 인식 데이터 세트에서 사전 훈련된 모델을 세밀하게 조정하여 평가되었습니다. 우리는 Kinetics 사전 훈련과 비교했을 때, 일부 하향 작업에서 심지어 더 우수한 결과를 보고했습니다. 마지막으로, 합성 비디오에 대한 사전 훈련의 이점을 극대화하기 위한 일반적인 지침과 유용한 속성을 모두 분석했습니다.



### Revisiting Point Cloud Completion: Are We Ready For The Real-World? (https://arxiv.org/abs/2411.17580)
- **What's New**: 이 논문은 현실 세계에서 획득한 포인트 클라우드 데이터셋의 부족한 특성과 문제점에 대해 다루고 있습니다. 실세계 산업 포인트 클라우드 데이터셋인 RealPC를 처음으로 제안하며, 이는 21개 산업 구조 카테고리에서 약 40,000개의 쌍을 포함하고 있습니다. 이 데이터셋은 포인트 클라우드 완성을 위한 연구를 촉진하는 데 기여할 것입니다. 또한 현재의 합성 데이터셋이 실제 환경의 토폴로지적 특성을 충분히 반영하지 못한다는 점을 강조하고 있습니다.

- **Technical Details**: RealPC 데이터셋은 비균일하게 희소하고 불완전한 포인트 클라우드를 처리하는 데 중점을 두고 설계되었습니다. 다양한 센서 기술이 결합된 이 데이터셋은 포인트 클라우드의 전반적인 형태를 골격 구조로 나타내는 0차원 Persistent Homology 기반의 토폴로지적 우선 순위를 통합하는 방법을 제시합니다. 이로 인해 기존 모델이 완전한 형태를 생성하는 데 도움을 주는 가능성을 보여주고 있습니다. 지속적 동질성(Persistent Homology)은 데이터 세트의 전역적 구조적 특성을 이해하는 데 강력한 도구로 작용합니다.

- **Performance Highlights**: 기존의 포인트 클라우드 완성 모델들은 합성 데이터셋에 최적화되어 있어 실제 데이터셋에서 성능이 저하되는 경향이 있습니다. 연구 결과, RealPC 데이터셋을 사용했을 때, 기존 모델들이 실제 환경에서 재현하는 데 부진한 성과를 보였습니다. 이는 데이터 중심의 접근 방식을 통해 실제 데이터를 다루는 데 있어 새로운 방향성을 제공함을 의미합니다. 따라서 본 연구는 포인트 클라우드가 실제 환경에서 효과적으로 완성될 수 있는 방법론을 제시하는 중요한 전환점을 의미합니다.



### A Distractor-Aware Memory for Visual Object Tracking with SAM2 (https://arxiv.org/abs/2411.17576)
Comments:
          Under review. Code available on Github: this https URL

- **What's New**: 이 논문에서는 메모리 기반 트래커인 SAM2(Track Anything Model 2)의 향상된 버전인 SAM2.1++를 제안합니다. 새로운 메모리 모델인 distractor-aware memory (DAM)와 분석 기반 업데이트 전략을 도입하여, 분할 정확도 및 트래킹 강인성을 동시에 개선합니다. 또한, 새롭게 제안된 distractor-distilled DiDi 데이터셋을 통해 방해물 문제를 더 잘 연구할 수 있도록 했습니다.

- **Technical Details**: SAM2.1++에서, 메모리는 최근의 모습 기억(recent appearances memory, RAM)과 방해물 해결 기억(distractor-resolving memory, DRM)에 따라 구성됩니다. RAM은 최근의 대상 모습을 저장하고, DRM은 방해물을 구분하는 데 필요한 앵커 프레임을 포함합니다. 이 새로운 메모리 관리 방안은 기존 표준 벤치마크에 비해 상당한 성능 향상을 이루어냈습니다.

- **Performance Highlights**: SAM2.1++는 SAM2.1 및 관련 SAM 메모리 확장 버전들을 세계적 수준의 벤치마크 7개에서 초월하며, 그 중 6개에서 새로운 최첨단 성능을 기록했습니다. 특히 방해물이 존재하는 상황에서도 이전 트래킹 방법들에 비해 더 강력한 트래킹 성능을 보여줍니다.



### A Bilayer Segmentation-Recombination Network for Accurate Segmentation of Overlapping C. elegans (https://arxiv.org/abs/2411.17557)
- **What's New**: 이번 논문에서는 C. elegans의 세분화 문제를 해결하기 위한 새로운 방법인 Bilayer Segmentation-Recombination Network (BR-Net)을 제안합니다. 기존 연구에서 C. elegans의 경계가 불분명하고 겹쳐지는 문제를 해결하고자 하는 노력은 있었지만, BR-Net은 세 가지 모듈로 구성되어 더 정교한 세분화를 가능하게 합니다. 특히, Unified Attention Module (UAM)을 통해 굵은 마스크를 더욱 효과적으로 인지하도록 설계하였습니다.

- **Technical Details**: BR-Net은 Coarse Mask Segmentation Module (CMSM), Bilayer Segmentation Module (BSM), Semantic Consistency Recombination Module (SCRM)로 구성되어 있습니다. CMSM은 초기 세분화를 위해 주로 사용되며, BSM은 겹치는 지역과 비겹치는 지역을 나누는 역할을 합니다. SCRM은 세맥적 일관성(semantic consistency)을 부여하여 C. elegans의 보다 정확한 세분화를 도와줍니다.

- **Performance Highlights**: 실험 결과, BR-Net은 C. elegans 데이터셋에서 다른 최신 인스턴스 세분화 방법들보다 우수한 성능을 보여주었습니다. 특히, C. elegans 인스턴스의 겹침 이미지 처리에서 경쟁력 있는 결과를 나타내어, 이 방법이 생물학적 이미지 처리 분야에서의 실용성을 높일 것으로 기대할 수 있습니다.



### Rapid Deployment of Domain-specific Hyperspectral Image Processors with Application to Autonomous Driving (https://arxiv.org/abs/2411.17543)
- **What's New**: 이 논문에서는 저비용의 시스템 온 모듈(System-On-Module, SOM) 플랫폼을 활용하여 자율 주행에 적용 가능한 효율적인 하이퍼스펙트럼 이미징(Hyperspectral Imaging, HSI) 프로세서를 구현하는 방법을 다룹니다. 기존의 고성능 처리 시스템에서 저비용 SOM으로 성공적으로 재설계된 경량 FCN(fully convolutional network)을 이용하여, 이미지의 의미론적 분할(latency) 요구 사항을 충족하는 모델의 양자화 과정을 설명하고 있습니다. 이 연구는 데이터 및 하드웨어 특화된 양자화 기술을 보고하며, HSI를 위한 저비용 AI 코프로세서의 사용 가능성을 높이고 있습니다.

- **Technical Details**: 저비용 SOM의 도입은 자율 주행 시스템(ADS)에 대한 하이퍼스펙트럼 이미징의 적용 가능성을 넓힙니다. 연구자들은 이전에 개발된 HSI 분할 시스템을 8비트 양자화 모델로 재설계하였으며, 이를 통해 이미지 전처리와 특징 추출의 효율성을 높임과 동시에 파라미터 수를 4배 줄였습니다. FCN 모델은 Keras/Tensorflow2를 사용하여 개발되었고, NVIDIA GFORCE RTX-3090에서 학습되었습니다.

- **Performance Highlights**: 모델의 효율성과 정확도 향상을 위해 FCN 양자화의 과정이 세심하게 진행되었습니다. 요약한 결과, 억제된 정밀도로도 전체 성능 저하 없이 IoU 지수의 변화를 유지하며, 8비트 정밀도의 Min-Max 양자화에 강건한 성능을 보여주었습니다. 마지막으로, 이 연구는 AMX-Xilinx Vitis AI 3.0 도구를 활용하여 맞춤형 양자화 파이프라인을 구성함으로써 시스템의 성능을 극대화하고 있습니다.



### Box for Mask and Mask for Box: weak losses for multi-task partially supervised learning (https://arxiv.org/abs/2411.17536)
Comments:
          Accepted for publishing in BMVC 2024

- **What's New**: 이 논문은 물체 탐지(Object Detection)와 의미 분할(Semantic Segmentation) 두 작업 간의 데이터 구조와 정보 수준의 차이를 살펴봅니다. 특히, 서로 다른 태스크의 정보들을 결합하여 여러 태스크의 부분적으로 지도된 학습(Multi-task Partially Supervised Learning)이 가능하다는 점에서 상당한 기여를 합니다.

- **Technical Details**: 저자는 Box-for-Mask와 Mask-for-Box 전략을 제안하며, 이 두 가지를 결합하여 BoMBo 방식으로 발전시킵니다. 이 방법들은 각각의 태스크에서 얻어진 주석(annotation) 정보를 다른 태스크 학습에 활용하는 데 중점을 둡니다. 기존의 슈퍼바이즈드 손실(Supervised Loss)과 결합된 다양한 약한 손실(Weak Loss)에 대한 연구를 수행하였습니다.

- **Performance Highlights**: VOC와 COCO 데이터셋을 통한 실험 및 아블레이션 연구(Ablation Studies)에서 제안된 방법이 긍정적인 결과를 보였음을 확인했습니다. 따라서, 서로 다른 태스크 간의 정보를 활용하여 학습의 효율성을 높일 수 있는 가능성을 제시합니다.



### IMPROVE: Improving Medical Plausibility without Reliance on HumanValidation -- An Enhanced Prototype-Guided Diffusion Framework (https://arxiv.org/abs/2411.17535)
- **What's New**: 본 연구에서는 전문가의 피드백 없이 의료 이미지 생성의 생물학적 신뢰성을 높이는 새로운 방법인 IMPROVE(Improving Medical Plausibility without Reliance on Human Validation)를 제안합니다. 기존의 Reinforcement Learning from Human Feedback (RLHF) 방식과는 다르게, 본 방법은 프로토타입-가이드 확산 프로세스를 이용하여 고품질의 의료 이미지를 생성합니다. 실험 결과, 본 방법은 생물학적 신뢰성을 크게 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 이 연구에서는 프로토타입 포괄적인 확산 모델을 기반으로 하는 의료 이미지 생성 파이프라인을 제시합니다. 본 방식은 전통적인 접근법에 비해 생물학적 정확성을 증가시키며, 뼈 생검 및 피부과 데이터셋에서 성능을 평가하였습니다. 본 방법은 기존의 모델들이 실패하는 조건부 특성을 활용하며, 전문가의 피드백 없이도 의료 이미지를 생성하는 데 중점을 둡니다.

- **Performance Highlights**: IMPROVE 모델은 기존의 생성 방법에 비해 두 가지의 서로 다른 데이터 도메인에서 생물학적 신뢰성을 일관되게 증가시키는 결과를 도출했습니다. 본 접근 방식은 데이터 부족 문제를 해결할 수 있는 잠재력을 지니고 있으며, 의료 이미지 생성의 비용 및 시간을 획기적으로 절감할 수 있는 가능성이 있습니다.



### FTMoMamba: Motion Generation with Frequency and Text State Space Models (https://arxiv.org/abs/2411.17532)
Comments:
          8 pages, 6 figures

- **What's New**: 이 연구에서는 FreqSSM(주파수 상태 공간 모델)과 TextSSM(텍스트 상태 공간 모델)을 결합한 새로운 확산 기반 FTMoMamba 프레임워크를 제안합니다. 이 모델은 저주파 및 고주파 구성 요소로 시퀀스를 분해하여 정적 포즈(예: 앉기, 누워 있기)와 세밀한 움직임(예: 전환, 넘어짐)을 생성합니다. 또한, 텍스트의 의미를 정렬하기 위한 텍스트 피쳐를 문장 수준에서 인코딩하여 텍스트와 동작 간의 일관성을 보장합니다.

- **Technical Details**: FTMoMamba는 FreqMamba와 TextMamba를 통합하여 주파수 및 텍스트 정보를 상태 공간 모델(SSM)에서 효과적으로 활용합니다. FreqSSM은 포즈의 정체성과 세밀한 움직임을 포착하기 위해 저주파 및 고주파 정보를 A 행렬에 통합하고, TextSSM은 문장 수준에서 텍스트와 모션 피쳐를 정렬하여 일관성을 유지합니다. 이 방법은 UNet 유사한 구조를 통해 노이즈를 예측하는 것을 중심으로 하여, 계산 효율성을 높이면서도 장거리 모델링을 지원합니다.

- **Performance Highlights**: FTMoMamba는 HumanML3D 데이터셋에서 텍스트-모션 생성 작업에서 우수한 성능을 보여줍니다. 특히, FID 값이 0.181로 크게 개선되어 이전 모델인 MLD의 0.421보다 현저히 낮습니다. 실험 결과, 이 프레임워크가 정밀한 동작 표현 및 텍스트-모션 일관성을 개선하는 데 있어 상당한 성과를 거두었음을 입증했습니다.



### HSI-Drive v2.0: More Data for New Challenges in Scene Understanding for Autonomous Driving (https://arxiv.org/abs/2411.17530)
- **What's New**: HSI-Drive 데이터셋의 업데이트된 버전인 v2.0이 도입되었습니다. 이 버전은 이전 버전보다 272% 더 많은 752개의 주석 달린 이미지를 포함하고 있으며, 계절 변화에 따른 다양한 주행 환경에서 수집된 이미지를 포함하고 있습니다. 특히, 새로운 데이터셋은 더 나은 HSI(segmentation)를 위한 설계 기반의 지식 생성에 기여합니다.

- **Technical Details**: v2.0 데이터셋은 겨울, 가을, 봄, 여름을 포함한 계절별 주행 데이터로 구성되어 있습니다. 각 이미지는 1088 x 2048 픽셀의 공간 해상도를 가지고 있으며, 5x5 픽셀 윈도우로 구성된 Fabry-Perot 필터로부터 스펙트럼 밴드가 추출됩니다. 이 데이터셋은 10개의 카테고리로 레이블이 지정된 약 4400만 개의 픽셀로 구성되어 있습니다.

- **Performance Highlights**: 실험을 통해 FCN(fully convolutional networks) 모델을 업데이트하였으며, 새로운 데이터의 통합이 모델의 성능과 강건성을 향상시키는 것을 보여주었습니다. 특히, 교통 안전을 위한 도로 표지 및 보행자 구분에 초점을 맞춘 세분화 성과가 두드러집니다. 데이터셋에서 추출된 이미지로 훈련한 모델들은 다양한 주행 조건에서도 유의미한 성과를 보였습니다.



### SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates (https://arxiv.org/abs/2411.17515)
- **What's New**: 이번 연구에서는 SuperMat라는 새로운 이미지 공간 물질 분해 모델을 제안합니다. 이는 고품질의 albedo, metallic, roughness 맵을 한 번의 추론 과정으로 동시에 생성하는 기능을 가지고 있습니다. 기존의 방법들과 비교했을 때, SuperMat는 처리 시간과 메모리 사용량을 대폭 줄여 효율성을 극대화합니다. 특히, 이 모델은 고유의 구조 전문가 브랜치를 활용하여 물질 속성을 동시에 처리할 수 있습니다.

- **Technical Details**: SuperMat는 안정적인 확산 방식에서 미세 조정된 이미지 공간 물질 분해 모델로, 주어진 시점에서 잘 알려지지 않은 조명 아래에서 객체의 이미지에 대해 상대하는 물질 맵을 생성합니다. 이 모델은 albedo와 RM(roughness, metallic) 맵을 동시에 지원하며, 단일 단계 추론을 가능하게 합니다. 구조 전문가 브랜치를 통해 다양한 물질 속성이 서로 상호작용을 하도록 설계되어 있어, 향상된 일관성을 제공합니다.

- **Performance Highlights**: SuperMat는 실험을 통해 최첨단 PBR(Physically Based Rendering) 물질 분해 품질을 보여주며, 이미지 당 추론 시간을 초 단위에서 밀리초 단위로 단축시킵니다. 특히, 3D 모델에 대해서는 SuperMat와 UV 정제 네트워크를 결합하여 약 3초 만에 높은 품질의 PBR 물질을 생성합니다. 이러한 접근은 예측된 물질 텍스처의 품질과 렌더링 정확도를 크게 높이는 결과를 가져옵니다.



### Perceptually Optimized Super Resolution (https://arxiv.org/abs/2411.17513)
- **What's New**: 본 연구에서는 슈퍼 해상도(Super-resolution, SR) 기술의 효율성을 높이기 위해 인간의 시각적 민감도를 활용한 새로운 접근 방식을 제안합니다. 제안된 방법은 사용자가 중요하게 여기는 시각적 요소에 자원을 집중시켜, 품질 저하 없이 계산 효율성을 높이는 데 중점을 둡니다. 이를 통해 SR 기술의 리소스 소모를 최소화하면서도 인식 가능한 품질을 유지할 수 있습니다.

- **Technical Details**: 저자들은 먼저 슈퍼 해상도 신경망 모델의 각 레이어가 출력 감소를 어떻게 유발하는지를 모델링했습니다. 이미지나 개별 프레임을 여러 패치로 나누고, 최근 연구를 바탕으로 시각적 모델이 패치에서 처리해야 할 네트워크의 레이어 수를 예측합니다. 결과적으로, 중요한 패치는 전체 모델을 통해 처리되고, 덜 중요한 패치는 단순히 bicubic 보간 방법을 사용하여 처리됩니다.

- **Performance Highlights**: 사용자 연구 및 정량적 평가를 통해, 제안한 모델은 품질 저하 없이 빠른 런타임을 구현할 수 있음을 입증했습니다. 연구 결과, FLOPS(부동 소수점 연산 수)를 2배 이상 줄일 수 있었으며, 이는 VR 및 AR 애플리케이션에도 유용하게 적용될 수 있습니다. 이러한 접근은 다양한 슈퍼 해상도 아키텍처에 통합 가능하며, 특히 사용자 경험을 향상시키는 데 기여할 수 있습니다.



### What's in the Image? A Deep-Dive into the Vision of Vision Language Models (https://arxiv.org/abs/2411.17491)
- **What's New**: 이번 논문에서는 Vision-Language Models (VLMs)의 시각적 정보 처리 메커니즘을 탐구합니다. VLM들은 복잡한 시각 콘텐츠를 이해하는 데 뛰어난 능력을 보여주며, 특히 이미지 정보를 처리하는 방법에 대한 새로운 통찰을 제공합니다. 이 연구를 통해 VLM이 이미지 토큰에 직접 접근하지 않고도 쿼리 텍스트 토큰만으로도 설명적인 응답을 생성할 수 있다는 점을 강조했습니다.

- **Technical Details**: 이 연구에서는 VLM의 다양한 레이어에서 attention 모듈을 분석하며, 시각-언어 정보 흐름에 대한 드러난 메커니즘을 탐구합니다. 특히, 중간 레이어가 정보 흐름에 중요한 역할을 하며, 세밀한 시각적 속성과 객체 세부사항은 공간적으로 지역화된 방식으로 이미지 토큰에서 직접 추출된다는 점을 보여줍니다. 이를 검증하기 위해 LLM 기반의 새로운 평가 프로토콜을 제안했습니다.

- **Performance Highlights**: 또한, 본 연구는 기존 VLM의 내부 표현을 압축된 맥락 공간으로 증류하여 효율적인 시각 처리를 가능하게 하였으며, 'Image Re-prompting'이라는 새로운 애플리케이션을 제안했습니다. 이 기술은 이미지에 대한 여러 질문을 압축된 콘텐츠만으로 효율적으로 묻는 것을 가능하게 하며, 압축된 맥락이 전체 데이터에 비해 20배 더 작지만 시각적 질문 응답에서 96%의 성능을 유지하는 것으로 나타났습니다.



### Learning Visual Hierarchies with Hyperbolic Embeddings (https://arxiv.org/abs/2411.17490)
- **What's New**: 본 연구는 사용자 정의 다중 레벨 시각 계층 구조를 하이퍼볼릭 공간(hyperbolic space)에 인코딩할 수 있는 학습 패러다임을 처음으로 제안합니다. 이 방법은 명시적인 계층 레이블을 필요로 하지 않고, 이미지를 통해 개체 수준에서 정의된 계층을 학습할 수 있도록 합니다. 특히, 대량의 이미지 데이터를 통해 추론된 계층 모델을 통해 인간의 인식과 일치하는 통합적인 세계 이해 시스템을 구축할 수 있는 기반을 제공합니다.

- **Technical Details**: 하이퍼볼릭 기하학(hyperbolic geometry)을 활용하여 계층 구조들을 효과적으로 다루는 접근 방식을 제시합니다. 본 논문에서 도입된 모델은 이미지와 개체 수준 정보만으로 정의된 계층 구조를 유지하는 방식으로 잠재 공간(latent space)을 구성합니다. 특히, 비대칭적인 관계를 갖는 계층 구조를 처리하기 위해 최근 제안된 각도 기반 비대칭 거리(metric)를 활용하여 대비 손실(contrastive loss)를 적용합니다.

- **Performance Highlights**: 예측 결과는 시각 계층 구조를 캡처하는 모델의 우수한 일반화 능력을 보여줍니다. 실험 결과, 본 연구에서 제안하는 방법은 부분 기반 이미지 검색 과정에서 현저한 성능 향상을 달성하였으며, 계층적 검색 평가를 위한 새로운 메트릭도 도입되었습니다. 이러한 평가 방식은 이미지 검색 성능을 효과적으로 측정하는 데 기여하여, 실제 세계의 시나리오에 적합한 모델 성능을 보장합니다.



### Puzzle Similarity: A Perceptually-guided No-Reference Metric for Artifact Detection in 3D Scene Reconstructions (https://arxiv.org/abs/2411.17489)
- **What's New**: 이번 작업에서는 새로운 no-reference 메트릭인 Puzzle Similarity를 제안하여, 새로운 뷰에서 아티팩트를 효과적으로 로컬라이즈할 수 있게 합니다. 기존의 no-reference 비주얼 품질 측정법이 제공하지 못하는 문제를 해결하고, 인간 평가와 상당히 잘 일치하는 아티팩트 맵을 생성할 수 있습니다. 이 결과는 이미지 복원 및 3D 재구성 개선 응용에 유용하게 활용될 수 있습니다.

- **Technical Details**: 제안하는 방법은 입력 뷰로부터 학습된 이미지 패치 통계(패치 통계)를 활용해 특정 장면에 대한 분포를 설정합니다. 이를 통해, 새로운 이미지를 해당 통계와 비교하여 아티팩트가 있는 영역을 식별합니다. 이 방법은 추가적인 레퍼런스 이미지 없이도 아티팩트 매핑을 제공하며, 기존의 no-reference와 full-reference 메트릭보다 성능이 뛰어납니다.

- **Performance Highlights**: 제안된 메트릭은 수작업 실험을 통해 평가되어, 인간이 인식한 아티팩트의 위치와 잘 일치함을 증명했습니다. 이 방식은 기존 no-reference 메트릭 및 인기 있는 full-reference 이미지 메트릭보다 우수한 성능을 보여 영상을 복원하거나 데이터를 수집하는 데 있어 보다 나은 가이드를 제공할 수 있습니다.



### Dual-task Mutual Reinforcing Embedded Joint Video Paragraph Retrieval and Grounding (https://arxiv.org/abs/2411.17481)
Comments:
          This work has been accepted with mandatory minor revisions by TMM

- **What's New**: 이번 연구에서는 새로운 작업인 Video Paragraph Retrieval and Grounding (VPRG)를 정의하고, 이를 해결하기 위해 Dual-task Mutual Reinforcing Embedded Joint Video Paragraph Retrieval and Grounding (DMR-JRG) 방법을 개발하였습니다. DMR-JRG는 상호 강화 방식으로 검색과 위치 결정을 동시에 해결함으로써, 주석이 없는 데이터와 비디오-단락 간의 매칭이 알려지지 않은 경우에도 효과적인 성능을 보여줍니다.

- **Technical Details**: DMR-JRG 방법은 검색 분기와 위치 결정 분기로 구분되어 있으며, 검색 분기에서는 inter-video contrastive learning 기법을 사용하여 비디오와 단락 특징을 대략적으로 정렬합니다. 위치 결정 분기에서는 비디오 세그먼트와 텍스트 단락 간의 지역, 전역 및 시간 차원에서의 일관성을 탐색하여 정밀한 매칭과 위치 결정을 달성합니다. 이 과정에서 세 가지 중요한 기술을 통합하여 각 차원에서의 정합성을 높이고, 전체적으로 정밀한 객관성을 확보합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 ActivityNet Captions, Charades-STA, TaCoS와 같은 대규모 데이터셋에서 기존 VSRG 방법보다 유의미하게 우수한 성과를 나타냈습니다. 특히, 약한 감독 하에서도 VPG의 정확성과 신뢰성이 크게 향상되었습니다. 이러한 성과는 비디오 단락 검색과 위치 결정 작업의 통합된 접근 방식 덕분입니다.



### COBRA: A Continual Learning Approach to Vision-Brain Understanding (https://arxiv.org/abs/2411.17475)
- **What's New**: 이 연구는 Vision-Brain Understanding (VBU) 분야에서 지속적인 학습의 문제를 해결하기 위한 COBRA라는 새로운 프레임워크를 소개합니다. COBRA는 과거 연구에서 발생하는 catastrophic forgetting 문제를 해결하고, 여러 새로운 모듈인 Subject Commonality (SC), Prompt-based Subject Specific (PSS), 그리고 transformer 기반의 MRIFormer 모듈을 포함합니다. 이러한 모듈들은 서로 다른 피험자들과의 상호작용에서 배운 시각-뇌 패턴을 유지 및 활용하면서도 새로운 피험자에 대한 특수한 정보를 학습할 수 있도록 설계되었습니다.

- **Technical Details**: COBRA는 SC 모듈을 통해 피험자 간에 공통적으로 나타나는 뇌 활성화 패턴을 캡처하고, PSS 모듈을 통해 개별 피험자의 특수 패턴을 수집합니다. MRIFormer 모듈은 transformer 인코더와 디코더를 이용하여 fMRI의 기능을 학습하며, 이는 일반화된 패턴과 개별적 패턴에 대해 유연하게 작동합니다. 이러한 모듈들은 새로운 피험자에 대해 업데이트되지만, 이전 피험자에 대한 모듈은 영향을 받지 않도록 설계되어 있습니다.

- **Performance Highlights**: COBRA는 VBU와 지속적인 학습 작업에서 이전 방법들을 초과하는 최첨단 성능을 달성하였습니다. 실험에 따르면, 이 연구는 기존의 문제를 효과적으로 해결하며, label-free 학습을 통해 추가적인 주석 없이도 PSS 모듈을 훈련할 수 있는 장점을 지니고 있습니다. 따라서, VBU의 지속적인 이해를 위한 새로운 접근 방식을 마련하며, 앞으로의 연구에 중요한 기여를 할 것으로 기대됩니다.



### Probing the Mid-level Vision Capabilities of Self-Supervised Learning (https://arxiv.org/abs/2411.17474)
Comments:
          Project Page: this https URL

- **What's New**: 이 연구는 중간 수준 비전 능력을 체계적으로 평가하기 위한 벤치마크 프로토콜을 소개하고, 22개의 주요 자가 지도 학습(self-supervised learning) 모델을 8개의 중간 수준 비전 작업에 걸쳐 포괄적으로 평가합니다. 이 연구에서는 기존의 고수준 인식 작업에 중점을 두고 설계된 SSL 접근 방식과는 달리, 중간 수준의 비전 능력에 대한 심층적인 분석을 제공합니다.

- **Technical Details**: 연구에서 수행된 실험에서는 중간 수준 작업과 고수준 작업 간의 성능 간의 상관관계가 약하다는 것을 보여주며, 또한 여러 SSL 방법이 중간 수준과 고수준 능력 간에 불균형한 성능을 보이는 것을 확인했습니다. 연구는 사전 학습(pretraining) 목표와 네트워크 아키텍처와 같은 중간 수준 비전 성능에 기여하는 주요 요소를 조사합니다.

- **Performance Highlights**: 이 연구는 SSL 모델들이 무엇을 학습했는지에 대한 전체론적이고 시의적절한 관점을 제공하며, 고수준 비전 작업뿐만 아니라 중간 수준 작업에서도 모델 벤치마크를 수행하도록 미래의 SSL 연구에 지침을 제공할 것으로 기대합니다. 실험 결과는 중간 수준 비전 작업에서 일부 모델이 뛰어난 성능을 보이는 반면, 다른 모델은 고수준 작업에서만 두각을 나타내는 불균형을 보여주는 것으로 나타났습니다.



### TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba (https://arxiv.org/abs/2411.17473)
- **What's New**: 본 논문은 Mamba 모델이 가진 잠재력을 활용하기 위해 새로운 Laplace mixer와 frequency ramp inception을 도입하여 개선한 Hybrid Architecture인 TinyViM을 제안합니다. 최근의 Mamba 기반 경량 모델들은 Convolution 또는 Transformer 기반 방법들과 경쟁력이 없었지만, 본 연구는 이러한 기반 구조의 한계를 극복하는 방향으로 진행되었습니다.

- **Technical Details**: Mamba는 입력 길이에 대한 선형 복잡도를 가진 상태 공간 구조(model)로, 저주파(low-frequency) 정보 모델링을 중심으로 하며, 기존의 기법들을 통해 계산 비용을 현저히 줄일 수 있습니다. 제안된 Laplace mixer는 고주파와 저주파 성분을 분리하고, frequency ramp inception을 통해 네트워크의 각 처리 단계마다 필요한 주파수 정보를 효율적으로 조정하여 성능을 향상시킵니다.

- **Performance Highlights**: TinyViM은 여러 하위 작업에서 뛰어난 성능을 보이며, 특히 ImageNet 분류 성능에서 최신 기술에 근접한 결과를 도출합니다. 경량 모델들과 비교했을 때 동일한 스케일의 Convolution, Transformer 및 Mamba 기반의 모델들과 비교하여 2-3배 높은 처리량과 2.7% 더 높은 정확도를 기록하였습니다.



### Unlocking the Potential of Text-to-Image Diffusion with PAC-Bayesian Theory (https://arxiv.org/abs/2411.17472)
- **What's New**: 이번 연구에서는 텍스트-이미지( T2I) 확산 모델에서 여러 객체와 속성을 포함하는 복잡한 프롬프트를 처리하기 위한 새로운 베이지안 접근 방식을 제안합니다. 기존 모델들은 modifier와 noun의 불일치를 겪고 있으며, 이로 인해 속성 바인딩이 잘못되거나 일부 요소가 생략되는 문제가 있었습니다. PAC-Bayes 프레임워크를 활용하여 사용자 지정 priors를 설계하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 베이지안 프레임워크에서는 attention 분포에 대한 사용자 지정 priors를 설계하여 각 객체 간의 분배, modifier와 noun의 alignment, 무관한 토큰에 대한 최소 관심 등을 보장합니다. 이 프레임워크는 attention 메커니즘을 해석 가능하게 만들어 주며, 이는 attribute-object alignment 개선에 기여합니다. 연구에서는 Kullback-Leibler divergence를 최소화하여 학습된 attention 분포와 사용자 맞춤형 prior 사이의 차이를 줄이는 방법론을 채택합니다.

- **Performance Highlights**: 제안한 방법은 여러 지표에서 최첨단 결과를 달성하며, 이미지 품질을 향상시키고 T2I 확산 모델에서 오랜 도전 과제를 해결하였습니다. 특히, 기존 모델들이 갖고 있던 오류들을 효과적으로 개선하고 더 신뢰할 수 있는 생성 모델을 구축하는 데 기여했습니다. 이를 통해 사용자는 더욱 명확하고 일관된 이미지 생성 결과를 얻을 수 있게 됩니다.



### Towards Precise Scaling Laws for Video Diffusion Transformers (https://arxiv.org/abs/2411.17470)
- **What's New**: 이번 논문에서는 비디오 확산 변환기(video diffusion transformers)의 최적 성능을 달성하기 위해 데이터 및 계산 예산에 맞는 최적의 모델 크기와 하이퍼파라미터를 결정하는 중요성을 강조합니다. 기존의 언어 모델에서 사용되는 스케일링 법칙(scaling laws)이 시각 생성 모델에 어떻게 적용되는지를 체계적으로 분석하여, 그 존재를 확인했습니다. 또한, 이 논문에서 비디오 확산 모델이 학습률(learning rate) 및 배치 크기(batch size)에 더 민감함을 발견했으며, 이는 정밀하게 모델링되지 않은 하이퍼파라미터입니다.

- **Technical Details**: 논문에서 제안하는 새로운 스케일링 법칙은 어떤 모델 크기와 계산 예산에 대해 최적의 하이퍼파라미터를 예측할 수 있도록 설계되었습니다. 이러한 최적 설정 하에서 전통적인 스케일링 방법에 비해 성능을 비교 가능하게 유지하면서 추론 비용(inference costs)을 40.1% 감소시켰습니다. 또한, 검증 손실(validation loss), 모델 크기, 계산 예산 간의 일반화된 관계를 수립하여 비최적 모델 크기에서도 성능 예측이 가능하도록 하였습니다.

- **Performance Highlights**: 제안한 방법은 계산 예산 1e10 TFlops 내에서 비디오 확산 변환기의 성능을 최적화할 수 있게 해줍니다. 비최적 모델 크기를 포함한 다양한 경우에 대해 더욱 나은 균형(trade-off)을 이끌어내며, 실제 추론 비용 제약 하에서도 성능을 예측할 수 있습니다. 이러한 발견은 자원 효율적인 비디오 생성 모델 개발에 중요한 기여를 할 것으로 기대됩니다.



### Adversarial Bounding Boxes Generation (ABBG) Attack against Visual Object Trackers (https://arxiv.org/abs/2411.17468)
Comments:
          Accepted in The 3rd New Frontiers in Adversarial Machine Learning (AdvML Frontiers @NeurIPS2024)

- **What's New**: 이번 논문에서는 단일 bounding box만을 이용하여 transformer 기반의 시각적 객체 추적기에 대한 새로운 white-box 공격 방법인 ABBG(Adversarial Bounding Box Generation) 공격을 제안합니다. 기존의 공격 기법들은 다양한 공격 시나리오에서의 적용성이 제한적이었는데, 우리의 방법은 이러한 한계를 보완하여 여러 transformer 트래커에 적용할 수 있게 합니다. ABBG는 단순하면서도 효과적인 방식으로, 여러 벤치마크에서 기존 공격보다 우수한 성능을 발휘합니다.

- **Technical Details**: ABBG 공격은 단일 트래커가 예측한 bounding box를 공격 프록시로 사용하여 공격을 수행합니다. 우리는 예측된 bounding box를 기반으로 여러 개의 adversarial bounding box를 생성하고, 이 bounding box들로부터 adversarial 손실을 계산합니다. 이 공격 방법은 IoU(Intersection over Union) 기준에 따라 가장 가까운 bounding box들을 양성 샘플로 선택하며, 이를 통해 트래커의 예측을 효과적으로 망가뜨립니다.

- **Performance Highlights**: 실험 결과, ABBG 공격은 TransT-M, ROMTrack, MixFormer 등 다양한 transformer 트래커에 대해 여러 추적 데이터셋에서 기존 공격 방법들보다 우수한 성능을 보였습니다. 특히, ABBG 공격은 대다수의 평가 지표에서 1위를 차지하며, 생성된 교란의 희소성(sparsity)과 감지 가능성(imperceptibility)에서도 경쟁적인 white-box 공격들보다 뛰어난 결과를 기록했습니다.



### Learning 3D Representations from Procedural 3D Programs (https://arxiv.org/abs/2411.17467)
Comments:
          Project Page: this https URL

- **What's New**: 본 연구는 레이블이 없는 3D 포인트 클라우드에서 전이 가능한 3D 표현을 얻기 위해 독창적인 접근법인 셀프-슈퍼바이즈드 러닝(self-supervised learning, SSL)을 제안합니다. 기존의 3D 자산 축소와 저작권 문제를 해결하기 위해, 간단한 기본형과 변형을 사용하여 자동 생성된 3D CNN으로 학습한 결과, 의미론적 콘텐츠가 결여된 데이터셋이라도 최신 모델들과 동등한 성능을 발휘한다는 점에서 획기적입니다.

- **Technical Details**: 본 연구는 점진적인 3D 프로그램(procedural 3D programs)을 통해 생성한 합성 데이터만을 이용해 3D 표현을 학습합니다. 주형(primitive) 3D 객체(예: 큐브, 실린더, 구 등을 샘플링하고 이들을 변형하여 다양한 기하학적 구조를 생성합니다. 150K의 합성 3D 포인트 클라우드를 생성하는 데 약 600 CPU 시간을 소요하며, 이 과정은 저작권 문제 없이 무한한 수의 3D 형태를 재생산 할 수 있습니다.

- **Performance Highlights**: Point-MAE-Zero 프레임워크는 기존의 사전학습된 Point-MAE(SN)와 비교했을 때, 의미가 결여된 합성 데이터셋에서도 유사한 3D 작업에서 뛰어난 성능을 보여줍니다. 두 모델은 구조적 차이 없이 마스크된 포인트 클라우드를 재구성하는 능력을 공유하며, 합성 데이터셋의 기하학적 다양성과 크기가 증가할수록 Point-MAE-Zero의 성능이 개선되는 것으로 나타났습니다.



### ShowUI: One Vision-Language-Action Model for GUI Visual Agen (https://arxiv.org/abs/2411.17465)
Comments:
          Technical Report. Github: this https URL

- **What's New**: 본 연구에서는 GUI(그래픽 사용자 인터페이스) 작업을 수행하기 위한 혁신적인 비전-언어-액션 모델인 ShowUI를 개발하였습니다. ShowUI는 스크린샷을 UI 연결 그래프로 구성하여 시각적 토큰 선택을 UI 지침에 따라 조정하는 새로운 방법론을 채택했습니다. 또한, 다양한 GUI 작업을 통합하는 Interleaved Vision-Language-Action Streaming 구조를 통해 훈련 효율성을 극대화했습니다.

- **Technical Details**: UI 가이드를 제공하는 시각적 토큰 선택 방법은 스크린샷의 패치를 그래프의 노드로 표현하고, 연결된 구성 요소를 통해 시각적 중복성을 모델링합니다. 이 방법은 self-attention 블록에서의 토큰 선택을 최적화하여 계산 비용을 줄이고 있습니다. 또한, Action 해석을 돕기 위해 GUI 작업 공간을 JSON 형식으로 구조화하여, 다양한 시각-언어-액션 데이터를 효율적으로 관리합니다.

- **Performance Highlights**: ShowUI는 256K의 데이터로 경량화된 2B 모델을 구성하여 제로샷 스크린샷 그라운딩에서 75.1%의 높은 정확도를 달성했습니다. UI 지침에 따른 토큰 선택 방식을 통해 훈련 중 시각적 중복 토큰을 33% 감소시키고 성능을 1.4배 가속화하는 성과를 보였습니다. 이 모델은 웹, 모바일, 온라인 환경에서의 내비게이션 실험에서도 뛰어난 효과를 입증하였습니다.



### WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Mod (https://arxiv.org/abs/2411.17459)
Comments:
          8 pages, 7 figures

- **What's New**: 본 연구에서는 Wavelet Flow VAE (WF-VAE)를 제안하여 저주파수 에너지 흐름을 잠재 표현(latent representation)으로 유도합니다. 이 방법은 비디오를 다중 주파수 도메인 성분으로 분해하고, 이를 통해 중요한 정보를 효율적으로 인코딩하도록 설계되었습니다. 또한 Causal Cache라는 기법을 도입하여 블록별 추론(block-wise inference) 시 잠재 공간(latent space)의 무결성을 유지합니다.

- **Technical Details**: WF-VAE는 Haar wavelet transform을 활용하여 비디오 신호를 다중 레벨로 분해하고, 이를 통해 다계층 피라미드적 특징을 추출합니다. 저주파수 비디오 정보를 잠재 공간으로 직접 전달하는 주요 에너지 흐름 경로를 설정하며, 이를 통해 3D 합성곱(convolution) 수를 크게 줄입니다. Causal Cache 메커니즘은 콘볼루션 슬라이딩 윈도우의 연속성(convolution continuity)을 유지하여 블록 별 추론이 직접적인 추론과 동일한 성능을 보장합니다.

- **Performance Highlights**: WF-VAE는 PSNR 및 LPIPS 지표에서 기존 비디오 VAE에 비해 우수한 성능을 보이며, 처리량은 2배 이상, 메모리 소비는 4배 낮춥니다. 이 연구의 결과는 광범위한 실험 평가를 통해 입증되었으며, 비디오 재구성 및 생성에서 현재 최첨단 성능을 달성하는 것으로 확인되었습니다.



### Spatially Visual Perception for End-to-End Robotic Learning (https://arxiv.org/abs/2411.17458)
Comments:
          8 pages, 5 figures

- **What's New**: 최근 모방 학습(imitation learning) 분야의 발전은 로봇 제어 및 구현된 지능(embodied intelligence)에서 큰 가능성을 보여주고 있습니다. 하지만 다양한 장착 카메라 관찰에서 강력한 일반화(robust generalization)를 달성하는 것은 여전히 중요한 과제로 남아 있습니다. 본 논문에서는 환경 변동성에 대응하기 위해 3D 공간 표현(spatial representations)을 활용하는 비디오 기반 공간 인식(framework) 프레임워크를 소개하며, 조명 변화(lighting changes)를 처리하는 데 주안점을 두고 있습니다.

- **Technical Details**: 우리의 접근 방식은 인터넷 규모 데이터(internet-scale data)로 훈련된 최첨단 단안 깊이 추정 모델(monocular depth estimation model)과 이미지 증강 기법인 AugBlender를 통합하여 동적 시나리오에서 강건성과 적응성을 높입니다. AugBlender는 제어된 RGB 오염을 통해 훈련 분포를 확장하고, Monocular Depth Estimation 모델의 깊이 맵과 융합하여 환경 변화에 강한 내성을 제공합니다. 이러한 방식을 통해 우리의 시스템은 다양한 카메라 노출(camera exposures)에서 성공률을 크게 증대시키는 결과를 보여줍니다.

- **Performance Highlights**: 우리는 AugBlender와 Depth Anything V2를 이용하여 비용 효율적인 비디오 기반 인식을 실현하였으며, 로봇 팔, 두 개의 카메라, RTX 3090 GPU와 같은 저비용 설정에서도 높은 성능을 발휘합니다. 우리의 방법론은 다양한 환경 조건에서 확장 가능하고 일반화 가능한 프레임워크를 갖추고 있어 기존 시스템에 쉽게 통합될 수 있습니다. 실험 결과, 우리의 접근 방식은 특히 조명 변화와 같은 환경 변동에 강력한 회복력을 보여주어 모방 학습의 강건성을 크게 향상시킵니다.



### FLEX-CLIP: Feature-Level GEneration Network Enhanced CLIP for X-shot Cross-modal Retrieva (https://arxiv.org/abs/2411.17454)
- **What's New**: 본 논문에서는 FLEX-CLIP이라는 새로운 Feature-Level Generation Network을 제안하고, 이는 CLIP(feature extractor) 기능을 향상시켜 X-shot Cross-modal Retrieval(CMR) 문제에 효과적으로 대응합니다. FLEX-CLIP은 두 가지 주요 단계를 포함하는데, 하나는 다중 모달(feature) 생성, 또 하나는 공통 공간(projection)이다. 이 접근 방식은 데이터 불균형 문제와 CLIP 기능의 저하 문제를 해결합니다.

- **Technical Details**: FLEX-CLIP의 첫 번째 단계인 다중 모달(feature) 생성에서는 VAE와 GAN 네트워크의 강점을 활용하여 데이터의 불균형 문제를 해결하고 CLIP 기능을 향상시키기 위해 복합적인 크로스 모달 생성 아키텍처를 설계했습니다. 두 번째 단계에서는 원본 샘플과 생성된 샘플을 공통 특징 공간으로 투영하여, 게이트 잔차 네트워크를 통해 CLIP 기능과 투영된 기능을 선택적으로 융합하여 특징 저하 문제를 크게 줄였습니다.

- **Performance Highlights**: FLEX-CLIP은 4개의 벤치마크 데이터셋에서 실험을 진행했으며, 기존의 최첨단 방법보다 7%에서 15% 향상된 성능을 보였습니다. 특히, X-shot 시나리오에서 (0, 1, 3, 5, 7-shot) FLEX-CLIP이 강력한 기준선보다 최대 7.9%의 성능 향상을 달성함을 입증했습니다.



### VLRewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models (https://arxiv.org/abs/2411.17451)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 멀티모달 AI 시스템을 평가하고 정렬하는 데 중요한 역할을 하는 Vision-language generative reward models (VL-GenRMs)의 평가 방법이 잘 탐구되지 않았음을 지적합니다. 기존의 평가 방법들은 전통적인 VL 작업에서 AI 주석을 기반으로 하는 선호 레이블에 의존하고 있으며, 이는 편향을 초래하고 최신 모델을 효과적으로 도전하지 못하는 경우가 많습니다. 이를 해결하기 위해 저자들은 VL-RewardBench라는 포괄적인 벤치마크를 소개합니다.

- **Technical Details**: VL-RewardBench는 일반 멀티모달 쿼리, 비주얼 환각 탐지, 복잡한 추론 작업을 포함하며, AI 지원 주석 파이프라인을 통해 인력의 검증을 결합한 샘플 선택 방식으로 1,250개의 고품질 예제를 큐레이션합니다. 16개의 선도하는 대형 비전-언어 모델을 대상으로 한 종합 평가 결과, VL-RewardBench는 도전적인 테스트베드로서 효과적이며, GPT-4o의 정확도는 65.4%에 불과합니다. 또한 Qwen2-VL-72B와 같은 최신 오픈소스 모델은 무작위 추측을 초과하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: VL-RewardBench에서의 성능은 MMMU-Pro 정확도와 강한 상관관계를 보이며 (Pearson's r > 0.9), VL-GenRMs로 Best-of-N 샘플링을 적용했을 때 나타납니다. 분석 실험을 통해 VL-GenRMs 개선을 위한 세 가지 주요 통찰이 발견되었습니다. 첫째, 모델은 추론 작업보다 기본적인 비주얼 지각 작업에서 주로 실패하며, 둘째, 추론 시간 확장 이점은 모델 용량에 따라 크게 달라지고, 셋째, VL-GenRMs의 판단을 학습하면 판단 능력이 크게 향상됩니다 (+14.7% 정확도). 이러한 연구 결과는 VL-GenRMs의 발전에 귀중한 자원이 될 것입니다.



### Identity-Preserving Text-to-Video Generation by Frequency Decomposition (https://arxiv.org/abs/2411.17440)
Comments:
          12 pages, 8 figures

- **What's New**: 이 논문은 일관된 인간 정체성을 유지하는 고품질 비디오를 생성하는 Identity-Preserving Text-to-Video (IPT2V) 모델인 ConsisID를 소개합니다. 기존 생성 모델에는 사례별 세부 조정이 필요하다는 과제가 있었으나, ConsisID는 조정이 필요 없는 파이프라인을 통해 이 문제를 해결합니다. 또한, 주파수 인식 방식(frequency-aware heuristic)을 통해 정체성을 보존하는 기능을 강화하여, 향상된 비디오 생성 성능을 달성하고자 합니다.

- **Technical Details**: ConsisID는 주파수 분석(frequency analysis)을 사용하여 정체성 제어 신호를 주파수 도메인에서 조작하는 구조적 접근 방식을 채택합니다. 낮은 주파수에서의 일반적인 얼굴 추출기를 통해 참조 이미지와 얼굴 키 포인트를 잠재공간(latent space)으로 인코딩하여 저주파 정보가 풍부한 특징을 생성하고, 이들을 네트워크의 얕은 층에 통합해 학습의 어려움을 보완합니다. 높은 주파수에서는 로컬 얼굴 추출기를 설계하여 고주파 세부 정보를 캡처하고 이를 변환기 블록(transformer blocks)에 주입하여 모델의 세부 정보 보존 능력을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 ConsisID는 고품질의 편집 가능하고 일관된 정체성을 유지하는 비디오를 생성할 수 있음을 입증하였습니다. 주파수 인식 기반 제어 방식을 통해 DiT 기반 모델로서의 최적의 제어 솔루션을 제공하며, 향후 더 효율적인 IPT2V 생성의 가능성을 열어줍니다. 이러한 결과는 ConsisID가 영상 생성의 새로운 가능성을 제시함을 의미하며, 기술 발전의 중요한 이정표입니다.



### Self-supervised Video Instance Segmentation Can Boost Geographic Entity Alignment in Historical Maps (https://arxiv.org/abs/2411.17425)
- **What's New**: 본 논문에서는 역사적 지도에서의 지리적 개체 추적을 개선하기 위해 Video Instance Segmentation (VIS)과 Self-Supervised Learning (SSL) 기법을 결합한 새로운 접근 방식을 제안합니다. 이 방법은 개체 세분화와 연결 과정을 단일 파이프라인으로 통합하여 자동화 수준을 크게 향상시킵니다. 특히, LABEL이 없는 역사적 지도 이미지를 활용하여 합성 비디오를 생성하는 독창적인 방법을 도입하여 수작업 주석의 필요성을 줄였습니다.

- **Technical Details**: 기존 지리적 개체 정렬 방법은 그림 이미지에서 벡터 개체를 추출한 후, 여러 지도의 개체를 연결하는 두 단계의 과정을 따릅니다. 하지만 본 연구에서는 VIS를 활용하여 3차원 볼륨의 연결된 개체를 직접 생성하는 방법을 채택합니다. 이러한 접근 방법은 지도 왜곡으로 인한 미세한 이동을 자연스럽게 추적할 수 있으며, 수작업 공정의 필요성을 줄이고 직관적으로 연결된 개체를 찾아냅니다.

- **Performance Highlights**: 실험 결과, 제안된 Self-Supervised VIS 방법은 평균 정밀도(AP)에서 24.9% 향상되었으며 F1 스코어는 0.23 증가하였습니다. 이는 기존 방식으로부터 새로 훈련된 모델과 비교할 때 상당한 성과를 보이고 있습니다. 이러한 결과는 VIS 모델의 성능을 향상시키고 역사적 지도 데이터의 활용도를 높이는 데 기여할 것으로 기대됩니다.



### DRiVE: Diffusion-based Rigging Empowers Generation of Versatile and Expressive Characters (https://arxiv.org/abs/2411.17423)
- **What's New**: 최근 생성 모델의 발전으로 인해 멀티 모달에서 고품질의 3D 캐릭터 재구성이 가능해졌습니다. 하지만 생성된 캐릭터를 애니메이션화하는 것은 여전히 도전 과제로 남아 있으며, 이는 의복과 머리카락 같은 복잡한 요소들에 대한 대규모 데이터셋과 효과적인 리깅 방법의 부족 때문입니다. 이 문제를 해결하기 위해 AnimeRig라는 대규모 데이터셋을 구축하여 정교한 스켈레톤과 스키닝 주석을 제공합니다. 이와 함께 DRiVE라는 새로운 프레임워크를 제안하여 복잡한 구조를 가진 3D 인간 캐릭터를 생성하고 리깅을 수행할 수 있게 됩니다.

- **Technical Details**: DRiVE는 3D Gaussian 표현을 활용하여 효율적인 애니메이션과 고품질 렌더링을 가능하게 합니다. 기존의 회귀 기반 접근 방식의 한계를 극복하기 위해 GSDiff라는 3D Gaussian 기반의 확산 모듈을 도입하여 조인트 위치를 공간 분포로 예측합니다. 이를 통해 리깅 정보를 자동으로 할당하고, 고급 제어가 가능합니다. DRiVE는 낮은 수준의 입력 예를 들어 단일 이미지나 텍스트 프롬프트에서 고품질 3D Gaussian을 생성합니다.

- **Performance Highlights**: DRiVE는 정밀한 리깅 결과를 달성하며, 의복과 머리카락의 사실적인 동작을 가능하게 합니다. extensive experiments에서 DRiVE는 이전의 방법들을 초과하는 품질과 다양성을 보여주었습니다. 또한 생성된 3D Gaussian을 활용하여 고품질의 리깅 결과를 도출하며, 애니메이션 파이프라인에 통합할 수 있는 결과를 제공합니다. 논문 수락 후 코드와 데이터셋은 학술적인 이용을 위해 공개될 예정입니다.



### Multimodal Outer Arithmetic Block Dual Fusion of Whole Slide Images and Omics Data for Precision Oncology (https://arxiv.org/abs/2411.17418)
- **What's New**: 본 논문은 DNA 메틸화 데이터와 Whole Slide Images (WSI)를 통합하여 중추신경계(CNS) 종양의 분류기를 개발하는 새로운 방법론을 제안합니다. 기존 방법들이 단일 패턴의 융합에 의존하는 반면, 이 연구는 조기 및 후기 융합 단계를 모두 활용하여 서로 보완적인 정보를 캡처하고 있습니다. 이는 진단 정확성을 크게 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 제안된 이중 융합 프레임워크에서는 조기 융합 단계에서 omic 임베딩을 패치 기반 잠재 공간으로 투영하여 omic-WSI 임베딩을 생성합니다. 이 과정에서, 중요한 패치에 주목하는 여러 인스턴스 학습이 적용되어 최적의 정보 전달이 이루어집니다. 후기 융합 단계에서는 Multi-modal Outer Arithmetic Block (MOAB)을 사용하여 slide-level omic-WSI 임베딩과 omic 데이터를 융합하여 강력한 특성 혼합을 통해 전반적인 상관 관계를 포착합니다.

- **Performance Highlights**: MOAD-Net은 20가지 세분화된 CNS 종양 하위 유형의 정확한 분류를 입증하며, TCGA-BLCA에서는 향상된 생존 예측을 달성하게 됩니다. 이 방법은 최신 기술과 비교하여 경쟁력 있는 성능을 발휘하며, 임상 진단에 대한 적용 가능성을 강조합니다. 전체적으로 이번 연구는 두 가지 융합 전략의 결합을 통해 탁월한 해석력과 분류 성능을 보였음을 나타냅니다.



### CoA: Chain-of-Action for Generative Semantic Labels (https://arxiv.org/abs/2411.17406)
Comments:
          15 pages, 8 figures

- **What's New**: 최근 비전-언어 모델(vision-language model, VLM)의 발전은 이미지 분류에서 놀라운 성능을 보여주었습니다. 그러나 자율주행과 같은 보다 개방적인 도메인에서는 사전 정의된 레이블이 실용적이지 않으며, 모든 관련 정보와 일치하는 레이블을 생성하는 CoA라는 새로운 접근법을 소개합니다. 이 방법은 기존의 단일 레이블을 예측하는 한계를 극복하고, 이미지의 풍부한 정보를 반영하는 레이블 생성을 목표로 합니다.

- **Technical Details**: CoA(Chain-of-Action) 방법은 이미지의 각 세부 특성과 관련된 레이블을 생성하기 위해 다양한 행동을 나누어 수행하는 구조입니다. 각 행동은 이전 행동에서 중요한 정보를 추출하여 다음 행동으로 전달함으로써, 이미지에 대한 포괄적이고 정확한 의미론적 레이블을 생성하는 데 기여합니다. 이러한 접근 방식은 맥락 정보를 풍부하게 활용하여 모델의 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: CoA 방법은 여러 벤치마크 데이터셋에서 평가된 결과, 키 성능 지표에서 모든 기초 모델을 초과하는 뛰어난 성능을 보였습니다. 특히, 이전에 개발된 고급 VLM인 BLIP-2 및 Instruct-BLIP와의 비교에서 매력적인 결과를 기록하여 비유정적으로 매우 간단하면서도 효과적인 기초 모델로 자리잡을 가능성을 보여줍니다.



### NumGrad-Pull: Numerical Gradient Guided Tri-plane Representation for Surface Reconstruction from Point Clouds (https://arxiv.org/abs/2411.17392)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문에서는 NumGrad-Pull이라는 새로운 방법을 소개하며, 이는 3D 포인트 클라우드로부터 고속 및 고충실도의 표면 복원을 위한 접근법입니다. NumGrad-Pull은 하이브리드 형태의 명시적 및 암시적 표현을 사용하는 삼중 평면 구조(tri-plane structure)를 모델링하여 서명 거리 함수(signed distance function)의 학습을 가속화합니다. 추가적으로, 전통적인 해석적(analytical) 계산 대신 수치적(numerical) 그래디언트를 활용하여 훈련 안정성을 향상시킵니다.

- **Technical Details**: NumGrad-Pull은 삼중 평면 구조를 통해 공간 정보를 저장하고, 다층 퍼셉트론(multi-layer perceptron)을 활용하여 정보를 real signed distance 값으로 매핑하는 방식으로 구성됩니다. 이 구조는 기존의 암시적 방법들에 비해 거리 쿼리의 속도를 향상시키며, 지오메트릭 및 형태 표현력을 유지합니다. 훈련 안정성을 높이기 위해, 마뭄 분석되었습니다.

- **Performance Highlights**: 제안된 방법은 다양한 벤치마크에서 평가되어 효과성과 강인성을 입증했습니다. 하이브리드 명시적-암시적 삼중 평면 표현을 통해 불규칙한 포인트 클라우드로부터의 표면 복원에서 속도와 충실도를 크게 개선했습니다. 또한, 수치적 그래디언트를 사용하는 방법은 훈련 과정의 안정성을 개선하고, 진전을 촉진시키는 진보적인 삼중 평면 확장인해 결합됩니다.



### DepthCues: Evaluating Monocular Depth Perception in Large Vision Models (https://arxiv.org/abs/2411.17385)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 Large-scale pre-trained vision models가 어떻게 인간과 유사한 monocular depth cues를 이해하는지를 평가하는 새로운 벤치마크인 DepthCues를 도입했습니다. 이를 통해 20개의 다양한 사전 훈련된 모델을 분석하고, 각 모델이 이 시각적 단서를 얼마나 잘 활용하는지를 확인했습니다. 또한, DepthCues에서의 fine-tuning을 통해 모델의 깊이 추정 성능을 향상시킬 수 있음을 발견했습니다.

- **Technical Details**: DepthCues는 인간의 깊이 지각에 사용되는 elevation, light and shadow, occlusion, perspective, size, texture gradient와 같은 6개의 깊이 관련 작업을 포함하는 벤치마크입니다. 연구는 self-supervised learning(DINOv2), image generation(StableDiffusion)과 같은 다양한 훈련 설정을 사용하여 20개의 시각 모델을 평가했습니다. 결과적으로, DepthAnythingv2와 최신 모델이 인간의 monocular depth cues를 잘 이해하고 있다는 것을 발견했습니다.

- **Performance Highlights**: 모델의 DepthCues 평가 성능은 깊이 추정 성능과 높은 상관관계를 보였습니다. 특히 최신의 self-supervised 및 geometry estimation 모델이 인간과 유사한 깊이 단서를 더 잘 이해하는 경향이 있었습니다. 이 연구는 깊이 지각을 향상시키기 위해 모델의 초깃값에 depth cue priors를 주입함으로써 실질적인 개선을 보여주었습니다.



### AnchorCrafter: Animate CyberAnchors Saling Your Products via Human-Object Interacting Video Generation (https://arxiv.org/abs/2411.17383)
- **What's New**: 이번 연구에서는 Pose-guided human video generation 기술에 인간-객체 상호작용(HOI)을 통합한 새로운 시스템인 AnchorCrafter를 소개합니다. AnchorCrafter는 2D 비디오를 생성하여 목표 인물과 맞춤형 객체 간의 상호작용을 높은 시각적 충실도로 제공합니다. 이는 기존의 자동화된 제품 홍보 비디오 생성의 한계를 극복하고, 더 나아가 상품 홍보를 위한 실질적인 비디오 생성 솔루션을 제안합니다.

- **Technical Details**: AnchorCrafter는 두 가지 핵심 혁신인 HOI-appearance perception과 HOI-motion injection을 도입합니다. HOI-appearance perception은 다각적 특성을 융합하여 객체의 외형 인식을 개선하고, 객체와 인물의 외형을 분리합니다. 호모젝션 인젝션은 깊이 및 손 3D 메시 입력을 통해 객체의 궤적을 정밀하게 제어할 수 있게 해주며, 상호작용 아티팩트를 완화하는 방안을 제시합니다.

- **Performance Highlights**: 실험 결과에 따르면 AnchorCrafter는 기존 방법들과 비교했을 때 객체 외형과 형태 감각을 훨씬 더 잘 보존하며, 인물 외형과 모션의 일관성도 잘 유지하는 것으로 나타났습니다. 이로 인해 AnchorCrafter는 다양한 이미지 및 비디오 품질 평가에서 최첨단 접근 방식에 필적하는 성능을 나타내는 등 경쟁력 있는 결과를 보여주고 있습니다.



### RealTraj: Towards Real-World Pedestrian Trajectory Forecasting (https://arxiv.org/abs/2411.17376)
- **What's New**: 이 논문은 기존 보행자 궤적 예측에서의 세 가지 주요 한계를 동시에 해결하고자 한다. 저자들은 RealTraj라는 새로운 프레임워크를 제안하여, 궤적 예측의 실제 적용 가능성을 향상시킨다. 이 방법은 합성 데이터에 대한 자기 지도 학습과 제한적인 실제 데이터를 이용한 약한 지도 학습이라는 두 가지 훈련 단계를 포함한다.

- **Technical Details**: 이 연구에서 제안하는 Det2TrajFormer 모델은 과거 감지 결과를 입력으로 활용해 추적 노이즈에 대한 불변성을 유지함으로써, 실제 오류에 대한 견고성을 증대시킨다. 모델은 여러 선행 과제를 이용해 사전 훈련 되어 실측 데이터만으로 예측 성능을 향상시키며, 사람 ID 주석의 필요를 현저히 줄인다. 이 과정에서 발생하는 가속도 정규화 항이 급격한 변화들을 억제하여 부드럽고 현실적인 궤적 예측을 가능하게 한다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 궤적 예측 방법들에 비해 여러 데이터셋에서 뛰어난 성능을 보였다. 저자들은 제안된 접근 방식이 보행자 인식 오류, 데이터 수집 비용 및 사람 ID 주석 비용 문제를 효과적으로 해결했음을 입증하였다. 또한, 이 방식을 통해 실제적인 조건에서도 성능이 향상되어 보행자 궤적 예측의 진정한 응용 가능성을 높였다.



### SAM-MPA: Applying SAM to Few-shot Medical Image Segmentation using Mask Propagation and Auto-prompting (https://arxiv.org/abs/2411.17363)
Comments:
          Accepted as an oral presentation at NeurIPS 2024 AIM-FM Workshop

- **What's New**: 최근 발표된 논문에서는 의료 이미징 분야에서 세분화(segmentation) 시 필요한 라벨 데이터의 부족 문제를 해결하기 위해 SAM-MPA라는 혁신적인 프레임워크를 제안합니다. 이 방법은 10개 이하의 라벨 예시를 이용해 고성능 세분화를 가능하게 하여, 대규모 라벨 데이터의 필요성을 줄입니다. SAM(Mask Propagation-based Auto-prompting)을 기반으로 하여, 기존의 FSS(소수 샷 세분화) 방식의 한계를 극복하고 자동화를 통해 효율성을 높입니다.

- **Technical Details**: SAM-MPA는 Mask Propagation과 자동 프롬프트 생성을 통합하여 설계되었습니다. 이 프레임워크는 k-중심 클러스터링(k-centroid clustering)을 통해 가장 대표적인 샘플을 선정하고, 이들 샘플과 질의 이미지(query image) 간의 등록(registration)을 수행하여 마스크 지식을 효과적으로 전파합니다. 마지막으로, 변형된 마스크를 바탕으로 전경 포인트와 바운딩 박스 등을 포함한 프롬프트를 자동으로 생성하여 SAM에 입력하면 세분화 예측을 창출합니다.

- **Performance Highlights**: 논문에서 제안된 SAM-MPA는 Breast US와 Chest X-ray 데이터셋에서 각각 82.07%와 94.13%의 Dices 점수를 달성하며, minimal labeled data를 사용하여 우수한 성능을 입증하였습니다. 실험 결과는 제안된 방법의 효율성을 증명하며, 적은 수의 라벨 예시로도 높은 정확도의 세분화를 가능하게 했습니다. 기존의 최신 FSS 방법들과 비교하였을 때, SAM-MPA는 더욱 높은 정확도를 보여주었습니다.



### DWCL: Dual-Weighted Contrastive Learning for Multi-View Clustering (https://arxiv.org/abs/2411.17354)
- **What's New**: 이 논문에서는 Multi-View Clustering (MVC) 문제를 해결하기 위해 Dual-Weighted Contrastive Learning (DWCL) 모델을 제안합니다. DWCL은 Best-Other (B-O) contrastive 기법을 도입하여 개별 뷰의 표현 능력을 향상시키고, 조합된 뷰의 신뢰성을 감소시킵니다. 이 모델은 뷰의 품질과 불일치를 모두 고려하여 표현 열화를 줄이는 이중 가중치 전략도 포함하고 있습니다.

- **Technical Details**: DWCL의 핵심인 B-O contrastive 메커니즘은 silhouette coefficient (SI)를 기반으로 최상의 뷰를 선정한 후 다른 뷰와 쌍을 이루는 방식으로 구성됩니다. 이로 인해 B-O 메커니즘의 복잡도가 O(|V|²)에서 O(|V|)로 줄어들어 계산 효율성이 향상됩니다. 또한, 이중 가중치를 통해 각 뷰의 품질을 평가하고 낮은 품질 및 높은 불일치를 가진 쌍을 감소시킵니다.

- **Performance Highlights**: DWCL은 8개의 멀티뷰 데이터 세트에서 이전 방법들보다 뛰어난 성능을 보여주었으며, 특히 Caltech6V7 데이터 세트에서 5.4%, MSRCv1 데이터 세트에서 5.6%의 정확도 향상을 달성했습니다. 이는 DWCL이 다중 뷰 표현 학습에서 강력하고 견고한 결과를 가져온다는 것을 입증합니다.



### Real-Time Multimodal Signal Processing for HRI in RoboCup: Understanding a Human Refer (https://arxiv.org/abs/2411.17347)
Comments:
          11th Italian Workshop on Artificial Intelligence and Robotics (AIRO 2024), Published in CEUR Workshop Proceedings AI*IA Series

- **What's New**: 본 연구에서는 NAO 로봇 플랫폼을 사용하여 로봇의 제스처 인식 및 호루라기 감지를 위한 새로운 접근 방식을 제안합니다. 이는 인간과 로봇 간의 커뮤니케이션을 향상시키기 위해 두 단계의 파이프라인을 통해 이루어집니다. 이 접근 방식은 RoboCup과 같이 경쟁적인 환경에서 실시간으로 작동할 수 있는 시스템을 개발하는 데 기여할 것입니다.

- **Technical Details**: 제안된 메커니즘은 키포인트 추출(keypoint extraction)과 분류(classification)를 통해 제스처를 인식하며, 지속적인 합성곱 신경망(Continuous Convolutional Neural Networks, CCNNs)을 사용하여 효율적으로 호루라기를 감지합니다. 이러한 기술들은 로봇이 최소한의 네트워크 의존성으로 심판의 제스처를 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 이 연구의 접근 방식은 RoboCup과 같은 역동적인 환경에서 인간-로봇 상호작용을 개선하는 잠재력을 가지고 있습니다. 제안된 시스템은 행위를 인식하고 빠르게 반응할 수 있는 능력을 가지며, 이는 자율 시스템의 개발을 진전시키는 데 도움이 될 것입니다.



### MotionLLaMA: A Unified Framework for Motion Synthesis and Comprehension (https://arxiv.org/abs/2411.17335)
- **What's New**: 이 논문은 MotionLLaMA라는 통합 프레임워크와 새로운 전체 신체 모션 토크나이저인 HoMi Tokenizer를 소개합니다. MotionLLaMA는 단일 코드북을 사용하여 고성능의 통합 표현 공간을 구축하고, 다양한 모션 관련 작업을 처리하기 위해 대형 언어 모델을 통합합니다. 또, MotionHub라는 현재 가장 방대한 멀티모달, 멀티태스크 모션 데이터세트를 도입하여 대형 언어 모델의 파인튜닝을 가능하게 합니다.

- **Technical Details**: MotionLLaMA는 전체 신체 모션과 공간 위치 정보를 이산 모션 토큰으로 인코딩하는 HoMi Tokenizer를 포함하며, 이는 LLaMA-3 모델의 확장 어휘로 사용됩니다. HoMi Tokenizer는 단일 코드북으로 동작하며, 코드북 활용 분석 및 양방향 패스트 푸리에 변환 게이팅 메커니즘을 통해 기존의 잔여 VQ-VAE와 동등한 재구성 성능을 달성합니다. 이 프레임워크는 9가지의 다양한 모션 관련 작업을 지원합니다.

- **Performance Highlights**: MotionLLaMA는 모션 완성, 상호작용 이인 텍스트-모션 및 모든 이해 작업에서 SOTA 성능을 달성하며, 나머지 작업에서도 SOTA와 유사한 성능을 보입니다. 연구 결과, MotionHub 데이터세트는 131,515개의 단일 인물 모션 데이터와 21,021개의 이인 모션 데이터를 포함하여, 다양한 모션 데이터와 오디오를 총 70.10시간 제공합니다. 이 성과는 고유한 모션 생성 및 이해에 대한 새로운 가능성을 제시합니다.



### InsightEdit: Towards Better Instruction Following for Image Editing (https://arxiv.org/abs/2411.17323)
- **What's New**: 이 논문에서는 지시 기반 이미지 편집(Instruct-based image editing) 작업에 주목하고 있습니다. 기존 연구(예: InstructPix2Pix, InstructDiffusion, SmartEdit)는 끝점 간 이미지 편집을 탐구했지만, 저해상도와 간단한 지시 사항으로 인한 데이터세트의 한계가 여전히 존재합니다. AdvancedEdit라는 고품질 시각 정보와 복잡한 지시를 갖춘 대규모 데이터세트를 수집하여 이러한 문제를 해결하였습니다.

- **Technical Details**: InsightEdit는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLM)의 강력한 기능을 활용하여 텍스트와 시각적 특성을 모두 사용하는 두 가지 스트림 브리징 메커니즘을 도입합니다. 이를 통해 이미지 편집 과정에서 더 정밀하게 지침을 반영할 수 있습니다. 이 연구는 두 가지 문제를 해결하기 위해 새로운 데이터 구축 파이프라인(data construction pipeline)을 제시하였습니다.

- **Performance Highlights**: InsightEdit는 복잡한 지시 사항을 따르는 것과 원본 이미지의 배경 일관성을 유지하는 데 있어서 최첨단 성능(state-of-the-art performance)을 달성했습니다. 실험 결과는 기존 방법보다 뛰어난 성능을 보이며, 특히 높은 시각적 품질과 배경 일관성을 제공합니다.



### Event Ellipsometer: Event-based Mueller-Matrix Video Imaging (https://arxiv.org/abs/2411.17313)
- **What's New**: 본 논문에서는 Event Ellipsometer라는 새로운 방법을 소개하여, 동적 장면을 위한 Mueller-matrix 비디오를 획득할 수 있도록 합니다. 기존의 광학적 엘립소미터들은 정적인 장면을 포착하는 데 제한적이었으나, 이번 연구는 빠르게 회전하는 quarter-wave plates (QWPs)와 이벤트 카메라를 이용한 시스템을 통해 이를 해결합니다. 이 방법은 30fps에서 동적 장면을 캡처할 수 있는 새로운 가능성을 열어줍니다.

- **Technical Details**: Event Ellipsometer는 QWPs와 이벤트 카메라를 활용하여 빛의 편광 상태를 분석합니다. 이 시스템은 가속적으로 변화하는 장면에서의 픽셀별 Mueller 매트릭스를 복원할 수 있는 두 단계의 reconstruction 기법을 개발하였습니다. 논문에서는 센서 노이즈와 장면 움직임으로 인한 아웃라이어를 처리하기 위한 물리적 유효성 제약도 포함합니다.

- **Performance Highlights**: 실험적으로, Event Ellipsometer는 0.045의 평균 제곱 오차로 알려진 Mueller 매트릭스를 가진 재료에서 높은 정확성을 입증하였습니다. 이 방법은 비평면 물체를 캡처할 수 있으며 공간 해상도를 저하시키지 않고 HDR 장면을 획득할 수 있는 뛰어난 능력을 보여줍니다. 또한, photoelasticity, 인체 캡처 및 테이프 탐지와 같은 실제 응용이 가능합니다.



### Reward Incremental Learning in Text-to-Image Generation (https://arxiv.org/abs/2411.17310)
Comments:
          Under review

- **What's New**: 최근 연구에서 Denoising Diffusion 모델이 텍스트-이미지 생성에서 성공적인 성과를 거두고 있습니다. 기존의 모델들이 제공하는 뛰어난 이미지 합성과는 달리, 실질적인 애플리케이션에서는 미세 조정이 필요합니다. 이러한 미세 조정의 요구를 충족하기 위해, 본 논문에서는 다중 목표를 점진적으로 적응하도록 요구되는 새로운 문제인 Reward Incremental Learning (RIL)을 정의합니다.

- **Technical Details**: RIL 설정에서는 각 보상 작업에 대해 생성 모델의 결과를 최적화하기 위한 보상 함수Rt(⋅)를 사용합니다. 기존의 방법들이 단일 보상 작업에만 초점을 맞추는 반면, RIL은 여러 목표를 고려하여 모델을 점진적으로 미세 조정합니다. 연구 결과, 기존의 방법들은 성과 저하를 초래하는 catastrophic forgetting 현상이 나타남을 발견했습니다.

- **Performance Highlights**: 제안하는 Reward Incremental Distillation (RID) 방법은 이러한 catastrophic forgetting 문제를 최소한의 계산 비용으로 해결할 수 있음을 보여줍니다. RID 방식은 여러 RIL 작업 시퀀스에서도 일관된 고품질의 제품을 생성하는 데 성공적이었습니다. 실험 결과, RID는 2%의 추가적인 계산 비용으로 높은 품질의 생성 결과를 달성할 수 있었습니다.



### in-Car Biometrics (iCarB) Datasets for Driver Recognition: Face, Fingerprint, and Voic (https://arxiv.org/abs/2411.17305)
Comments:
          8 pages, 13 figures, 4 tables

- **What's New**: 이번 연구에서는 차량 내부에서 수집된 얼굴 비디오, 지문 이미지, 목소리 샘플로 구성된 세 가지 생체 인식 데이터셋(iCarB-Face, iCarB-Fingerprint, iCarB-Voice)을 제공하여, 200명의 동의한 자원봉사자들로부터 수집되었습니다. 이 데이터는 근적외선 카메라, 두 개의 지문 스캐너, 두 개의 마이크로폰을 사용하여 획득되었고, 차량이 주차되어 있을 때 수집되었습니다. 이 데이터셋은 차량 내부 생체 인식 시스템 평가 및 다른 환경에서의 활용도 가능하여 연구 커뮤니티에 매우 가치 있는 자료입니다.

- **Technical Details**: 이 데이터셋은 생체 인식 데이터 수집의 여러 도전 과제를 파악하는 데 유용하며, 평가 프로토콜을 통해 다양한 생체 인식 시스템을 평가하고 벤치마킹할 수 있는 잠재력을 제공합니다. 또한, 데이터셋의 다중 모달리티 특성을 활용하여 다중 모달 융합 알고리즘의 훈련/test 또는 Presentation Attack Detection 알고리즘 평가에도 활용할 수 있습니다. 데이터는 다양한 센서(근적외선 카메라, 마이크로폰, 지문 스캐너)를 통해 수집되었으며, 이로 인해 수집된 데이터는 실제 생체 인식 상황에서 직면할 수 있는 문제들을 시뮬레이션합니다.

- **Performance Highlights**: iCarB 데이터셋은 대규모 인구 통계적 다양성을 갖춘 공공 생체 인식 데이터셋으로, 남녀 비율이 50/50로 나뉘어지고 Fitzpatrick 스펙트럼 전반에 걸쳐 다양한 피부색을 포함하고 있습니다. 기존의 다른 데이터셋이 주로 한 가지 생체 인식 모달리티에 의존하는 반면, iCarB 데이터셋은 세 가지 모달리티를 포함하여 접근할 수 있는 최초의 대규모 데이터셋입니다. 이 데이터셋을 통해 생체 인식 시스템의 정확성 향상에 기여할 것으로 기대됩니다.



### Task Progressive Curriculum Learning for Robust Visual Question Answering (https://arxiv.org/abs/2411.17292)
- **What's New**: 이번 연구에서는 Visual Question Answering (VQA) 시스템의 훈련 전략을 개선하여 보다 견고한 성능을 구현할 수 있음을 처음으로 보였습니다. 연구자들은 Task Progressive Curriculum Learning (TPCL)을 통해 VQA 문제를 질문 유형에 기반하여 더 작고 쉬운 작업으로 나누고, 점차적으로 모델을 훈련시키는 방법을 제안했습니다. 이 접근 방식은 개념적으로 간단하고 모델에 구애받지 않으며 구현이 용이합니다.

- **Technical Details**: TPCL의 핵심 아이디어는 개별 작업(작업 그룹) 기반으로 VQA 문제를 다중 작업 학습(MTL) 문제로 재구성하는 것입니다. 각 작업은 특정 질문 유형에 속하며, 훈련 과정에서 차례로 작업을 훈련시키게 됩니다. 모델 훈련 과정에서 작업의 난이도를 측정하기 위해 새로운 분포 기반 난이도 지표를 도입하였습니다.

- **Performance Highlights**: TPCL은 VQA-CP v2, VQA-CP v1 및 VQA v2 데이터셋에서 최신 상태의 성능을 달성하였습니다. 이 연구는 TPCL이 기존의 강력한 VQA 접근 방법보다 5%에서 7% 더 우수한 성능을 보임을 보여주며, VQA의 기본 백본 성능을 최대 28.5%까지 향상시킬 수 있는 잠재력을 가지고 있음을 입증했습니다.



### BadScan: An Architectural Backdoor Attack on Visual State Space Models (https://arxiv.org/abs/2411.17283)
- **What's New**: 새롭게 도입된 Visual State Space Model (VMamba)은 이미지를 패치의 시퀀스로 해석하는 State Space Mechanisms (SSM)을 사용하여 Vision Transformers (ViT)에 비해 우수한 성능을 보여주고 있습니다. 그러나 이러한 딥 모델은 적대적 공격에 취약하다고 알려져 있으며, 본 논문에서는 VMamba 모델의 백도어 공격에 대한 견고성을 평가했습니다. 이어서, VMamba 모델을 속이기 위해 설계된 새로운 구조적 백도어 공격인 BadScan을 도입하여 실험을 진행했습니다.

- **Technical Details**: BadScan 공격은 비트를 평면으로 나누어 시각적으로 감지할 수 없는 백도어 이미지를 생성합니다. 테스트 중 수정된 트리거 패치의 $k^{th}$ 비트 평면 간의 XOR 연산을 수행하여 트리거가 감지되면, VMamba의 전통적인 2D 선택 스캔(SS2D) 메커니즘이 새로운 BadScan 블록으로 교체됩니다. 새롭게 설계된 BadScan 블록은 네 가지 새로운 스캔 패턴을 포함해 VMamba 모델을 혼란에 빠뜨리도록 설계되었습니다.

- **Performance Highlights**: CIFAR-10과 ImageNet-1K Dataset을 통해 실험한 결과 VMamba 모델은 현재의 백도어 공격에 대해 일반적으로 견고하지만, BadScan 공격은 VMamba 모델과 그 변형을 속이는 데 있어 높은 Triggered Accuracy Ratio (TAR)를 달성하며 특히 효과적이라는 것을 보여주었습니다. 이 연구는 VMamba 모델이 백도어 공격에 대한 높은 취약성을 가지고 있음을 입증하고, 향후 시각적 상태 공간 모델에 대한 대책이 필요함을 강조합니다.



### HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator (https://arxiv.org/abs/2411.17261)
- **What's New**: AIGC 이미지의 품질 문제를 해결하기 위한 새로운 접근법이 제안됩니다. HEIE라는 계층적 설명 가능 이미지 불확실성 평가자(MLLM-Based Hierarchical Explainable image Implausibility Evaluator)가 소개되며, 이는 CoT(Chain of Thought) 기반의 설명 가능한 시스템을 활용하여 불확실성 지역의 평가와 더불어 구체적인 결함 영역을 식별할 수 있게 합니다. 이 연구는 기존의 평가 방식의 한계를 극복하고자 합니다.

- **Technical Details**: HEIE는 Adaptive Hierarchical Implausibility Mapper와 CoT-Driven Explainable Trinity Evaluator를 통합하여 더욱 정교한 이미지 평가를 수행합니다. Adaptive Hierarchical Implausibility Mapper는 ViT(Vision Transformer)의 저수준 이미지 특징과 MLLMs의 고수준 맵 토큰을 결합하여 정밀한 열 지도 예측이 가능합니다. 또한, Expl-AIGI-Eval이라는 새로운 데이터셋을 구축하여 AIGC 이미지의 해석 가능성과 불확실성 평가를 지원합니다.

- **Performance Highlights**: 본 연구의 방법론은 다양한 데이터셋과 태스크에서 최첨단 성능을 달성하였습니다. HEIE는 불확실성 기반의 적응형 토큰 접근법을 적용하여 정밀한 결함 위치 지정을 가능하게 하고, 새로운 데이터셋을 통한 평가로 기존 방법보다 향상된 해석 가능성을 보여줍니다. 지속적인 비교 및 절제 연구가 이 방법의 효과성을 입증하였습니다.



### Semantic Data Augmentation for Long-tailed Facial Expression Recognition (https://arxiv.org/abs/2411.17254)
- **What's New**: 이번 연구에서는 Facial Expression Recognition (FER)의 데이터 세트 불균형 문제를 해결하기 위한 새로운 방법인 세멘틱 증강(semantic augmentation) 방법을 제안합니다. 기존 연구들이 Long-Tailed Recognition을 위한 데이터 증강(data augmentation)에 초점을 맞췄다면, 본 논문은 VAE-GAN의 잠재 공간(latent space)에서 소스 데이터의 인코딩에 무작위성을 도입하여 새로운 샘플을 생성하는 방식을 적용했습니다.

- **Technical Details**: 본 연구의 핵심 기술은 VAE-GAN(Variational Autoencoder with Generative Adversarial Network) 기반의 라벨링된 데이터 증강입니다. 모델은 원본 데이터의 잠재 표현을 무작위화(randomness)하여, 데이터 세트의 롱테일(long-tailed) 분포를 균형화하는 새로운 샘플을 생성합니다. 이를 통해 수집된 RAF-DB 데이터셋에서 Facial Expression Recognition 성능을 개선할 수 있습니다.

- **Performance Highlights**: 우리의 제안된 증강 방법은 Facial Expression Recognition 뿐만 아니라 다양한 데이터 소모(data-hungry) 시나리오에서도 성능을 발휘할 수 있습니다. 실험 결과, 단순한 데이터 증강 기법보다 더 높은 정확도(accuracy)를 보였으며, 이는 실제 응용에서의 효과성을 높입니다.



### DGNN-YOLO: Dynamic Graph Neural Networks with YOLO11 for Small Object Detection and Tracking in Traffic Surveillanc (https://arxiv.org/abs/2411.17251)
- **What's New**: 본 논문은 DGNN-YOLO라는 새로운 프레임워크를 소개하며, 이는 동적 그래프 신경망(DGNN)과 YOLO11을 통합하여 교통 감시 시스템에서 작은 객체의 탐지 및 추적 성능을 향상시킵니다. 이 프레임워크는 YOLO11의 향상된 공간적 특징 추출 기능을 활용하여 정확한 객체 탐지를 수행하고, DGNN을 사용해 공간-시간 관계를 모델링함으로써 실시간 추적의 견고성을 높입니다. 실험 결과, DGNN-YOLO는 다양한 교통 조건에서 작은 객체의 탐지 및 추적에서 최첨단 방법들을 일관되게 능가함을 보여줍니다.

- **Technical Details**: DGNN-YOLO는 작은 객체 탐지 및 추적을 위해 YOLO11과 DGNN을 통합하여 역동적인 그래프 구조를 구축하는 메커니즘을 도입합니다. 여기서 각 객체는 노드로 나타내고, 객체 간의 상호작용은 엣지로 표현하여 복잡한 교통 환경에서의 적응성과 추적의 정확성을 보장합니다. DGNN은 시각적 데이터의 공간-시간 관계를 효과적으로 모델링함으로써 주어진 동적 환경에서의 다수의 객체 추적을 최적화합니다.

- **Performance Highlights**: DGNN-YOLO는 i2 객체 탐지 데이터셋에서 평가되어, 작은 객체 탐지 및 추적에서 기존 방법 대비 　　 0.8382의 정밀도(precision), 　　 0.6875의 재현율(recall), 0.6476의 평균 정밀도(mAP@0.5:0.95)라는 성과를 달성했습니다. 이 결과는 특히 작은 객체나 가려진 물체를 포함한 복잡한 시나리오에서의 견고성과 확장성을 나타냅니다. 이 연구는 효율적이고 실시간으로 작동하는 교통 감시 및 분석 솔루션을 제공하여 스마트 교통 시스템에 크게 기여합니다.



### Buffer Anytime: Zero-Shot Video Depth and Normal from Image Priors (https://arxiv.org/abs/2411.17249)
- **What's New**: Buffer Anytime은 비디오로부터 깊이 및 노말 맵을 추정하기 위한 새로운 프레임워크로, 비디오-깊이 및 비디오-노말 교육 데이터의 필요성을 제거했습니다. 기존의 대규모 주석 처리된 비디오 데이터셋에 의존하는 대신, 단일 이미지 프라이어를 활용하여 고품질 비디오 버퍼 추정을 시연합니다. 이는 전통적인 데이터 쌍 없이도 이미지 기반 모델의 효능을 이용하여 비디오의 기하학적 버퍼 생성을 효과적으로 지원합니다.

- **Technical Details**: 본 연구에서는 RGB 비디오 입력을 기반으로 깊이 맵과 노말 맵을 예측하는 문제를 설정합니다. 기존의 쌍 데이터 세트로 훈련되는 비디오 깊이 예측 모델들과 달리, 우리의 방법은 RGB 비디오 데이터만을 사용하여 기하학적 버퍼를 생성합니다. 우리는 경량화된 시간적 주의 아키텍처에 기반한 하이브리드 손실 함수와 이미지를 훈련시키는 기법을 결합하여 시간적 일관성(temporal consistency)과 정확성을 보장합니다.

- **Performance Highlights**: 우리의 방법은 이미지 기반 접근 방식들을 크게 초월하며, 대규모 비디오 데이터셋에서 훈련된 최신 비디오 모델들과 비견되는 성능을 달성합니다. Depth Anything V2 및 Marigold-E2E-FT와 같은 최첨단 모델에 적용하여 시간적 일관성을 크게 향상시켰습니다. 실험 결과는 비디오 기하학적 버퍼 추정 평가에서 두 모델이 획기적인 개선을 이루었음을 보여줍니다.



### DiffSLT: Enhancing Diversity in Sign Language Translation via Diffusion Mod (https://arxiv.org/abs/2411.17248)
Comments:
          Project page: this https URL

- **What's New**: 이 연구에서는 기존의 주석 의존성을 줄이기 위한 새로운 방법인 DiffSLT를 제안합니다. 이 방법은 기존의 gloss-free SLT 프레임워크를 확장하여 다양한 번역 결과를 생성하고, 손사물의 의미를 보존합니다. 연구팀은 다층적인 시공간 정보의 통합을 통해 성능을 높이는 Guidance Fusion Module(GFM)을 설계하였습니다.

- **Technical Details**: DiffSLT는 노이즈에서 시작하여 목표 문장 표현으로 변환하는 프로세스를 거칩니다. 이 과정에서 임의의 잠재 표현을 디노이즈하며, 입력 비디오의 시각적 특징을 조건으로 사용합니다. DiffSLT-P는 시각적 특징과 예상된 pseudo-glosses를 결합하여 개선된 번역을 제공합니다.

- **Performance Highlights**: DiffSLT 및 DiffSLT-P는 두 가지 SLT 데이터셋에서 다른 gloss-free SLT 메소드 대비 다양성을 획기적으로 개선하며, 우수한 성능을 달성했습니다. 이 두 모델은 생성된 구문의 독특성과 의미적 일관성을 유지하면서도 번역 품질을 현저히 향상시켰습니다.



### Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration (https://arxiv.org/abs/2411.17240)
- **What's New**: 이번 논문에서는 DM-Calib이라는 새로운 확산 기반 접근 방식을 제안하여 단일 입력 이미지로 핀홀 카메라의 내부 매개변수를 추정합니다. 기존의 단안 카메라 보정 방법들은 수작업으로 만들어진 가정에 의존하거나 제한된 훈련 데이터에 의해 제약이 있어 실제 이미지에서 일반화가 잘 이루어지지 않았습니다. 하지만, 대규모 데이터로 훈련된 안정적인 diffusion 모델이 다양하고 고품질의 이미지를 생성하는 능력을 보여주고 있습니다.

- **Technical Details**: 본 연구는 카메라 내부 매개변수로부터 이미지 기반 표현인 Camera Image를 도입합니다. 이 표현은 숫자적인 카메라 내부 매개변수를 손실 없이 인코딩하며 diffusion 프레임워크와 매끄럽게 통합됩니다. 입력 이미지에 조건을 두고 밀집된 Camera Image를 생성하는 문제로 보정을 재구성하며, RANSAC 연산을 통해 카메라 내부 매개변수를 추출합니다.

- **Performance Highlights**: 우리의 단안 보정 방법은 제로샷 메트릭 깊이 추정, 3D 계측, 자세 추정 및 스파스 뷰 복원 등 다양한 3D 작업에서 성능을 향상시키는 것을 입증했습니다. 여러 공개 데이터셋에 대한 광범위한 실험을 통해 DM-Calib 방법이 기존 기반선들을 크게 초월하며 3D 비전 작업에 광범위한 이점을 제공함을 보여주었습니다.



### Grounding-IQA: Multimodal Language Grounding Model for Image Quality Assessmen (https://arxiv.org/abs/2411.17237)
Comments:
          Code is available at: this https URL

- **What's New**: 이번 논문에서는 신규 이미지 품질 평가(IQA) 작업 패러다임인 'grounding-IQA'를 소개합니다. 이 패러다임은 다중 모달 참조와 기초를 통합하여 보다 세분화된 품질 인식을 실현합니다. grounding-IQA는 두 개의 하위 작업(Grounding-IQA-Description (GIQA-DES) 및 Visual Question Answering (GIQA-VQA))으로 구성되어, 세부 설명과 특정 위치 정보를 제공합니다.

- **Technical Details**: GIQA-DES는 이미지 품질에 영향을 미치는 주요 객체나 지역에 대한 정확한 위치 정보를 포함하여 서술적 평가를 생성해야 합니다. GIQA-VQA는 지역 객체와 관련된 저급 속성에 대한 질문-답변을 다룹니다. 이 데이터를 생성하기 위해 160K 이상의 다양한 도메인의 이미지와 설명으로 구성된 GIQA-160K 데이터셋을 구축하였으며, 이를 자동 주석화 파이프라인을 통해 생성하였습니다.

- **Performance Highlights**: 실험 결과, GIQA-160K 데이터셋으로 미세 조정된 모델이 grounding-IQA에서 현저한 성능 향상을 보였으며, GIQA-Bench 벤치마크를 통해 평가되었습니다. 이는 설명의 질, VQA 정확성 및 기초 정확성을 포함한 세 가지 측면에서 모델 성능을 평가하는 데 중점을 두었습니다.



### MLI-NeRF: Multi-Light Intrinsic-Aware Neural Radiance Fields (https://arxiv.org/abs/2411.17235)
Comments:
          Accepted paper for the International Conference on 3D Vision 2025. Project page: this https URL

- **What's New**: 이 논문에서는 다중 조명 정보(Multiple Light information)를 통합한 내재적 인식 신경 방사장(MLI-NeRF) 모델을 제안합니다. 이 방법은 다양한 조명 조건에서 촬영된 이미지를 기반으로 반사율과 음영을 분리하여 내재적 이미지 분해(Intrinsic Image Decomposition)의 품질을 향상시키는 데 초점을 맞춥니다. MLI-NeRF는 기존의 통계적 선행 지식 없이 물리기반 제약(Physics-based Constraints)을 활용하여 robust한 성능을 보입니다.

- **Technical Details**: 모델은 두 단계로 구성됩니다. 1단계에서는 서로 다른 카메라 각도와 조명 위치에서 이미지를 사용하여 조명 변화를 학습하고 NVS를 가능하게 합니다. 이어진 후처리 단계에서는 각 이미지에 대한 pseudo intrinsic images를 생성합니다. 이후 2단계에서는 반사율과 음영을 예측하기 위한 추가 모듈을 도입하여 내재적 인식을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MLI-NeRF는 합성 데이터셋과 실제 세계 데이터셋 모두에서 기존의 최신 기술보다 우수한 성능을 발휘함을 입증했습니다. 이 방법은 반사율 편집, 재조명 및 음영 편집과 같은 다양한 이미지 편집 작업에도 적용 가능하다는 점에서 시너지 효과를 발휘합니다. 데이터와 코드는 공개되어 있어 연구자들이 쉽게 접근할 수 있습니다.



### MWFormer: Multi-Weather Image Restoration Using Degradation-Aware Transformers (https://arxiv.org/abs/2411.17226)
Comments:
          Accepted by IEEE Transactions on Image Processing. The code is available at: this https URL

- **What's New**: 이번 논문에서는 다양한 기상 조건으로 인해 발생하는 화질 저하를 효과적으로 복원할 수 있는 새로운 아키텍처인 MWFormer(다중 기상 변환기)를 제안합니다. MWFormer는 단일 아키텍처에서 여러 기상 유도 열화를 처리할 수 있도록 설계되었으며, 하이퍼 네트워크(hyper-network)와 피처 단위 선형 조정(feature-wise linear modulation) 블록을 사용하여 다양한 기상 종류를 복원합니다. 이 시스템은 배운 파라미터 집합을 동일하게 사용하여 예측된 기상에 적응적으로 대응 같습니다.

- **Technical Details**: MWFormer는 콘텐츠와 독립적이며 왜곡을 인식하는 기능 임베딩을 효율적으로 생성하는 보조 네트워크를 훈련하기 위해 대비 학습(contrastive learning)을 적용합니다. 이 예측된 기상 정보를 가이드로 활용하여, 이미지 복원 과정에서 지역적(local) 및 전역적(global) 특성을 처리합니다. 또한, MWFormer는 새로운 모델 조정 방법을 제공하여, 단일 기상 복원 혹은 혼합 기상 복원을 재교육 없이도 수행할 수 있습니다.

- **Performance Highlights**: MWFormer는 다중 기상 복원 벤치마크에서 기존의 최첨단 모델들보다 유의미한 성능 개선을 보여주었습니다. 특히, 시각적 및 정량적 측면에서 뛰어난 성능을 발휘하며, 다양한 다른 네트워크 아키텍처에도 통합이 가능한 구조를 가지고 있습니다. 이로 인해 이미지 복원 작업의 효율성이 크게 향상될 수 있습니다.



### DreamMix: Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting (https://arxiv.org/abs/2411.17223)
- **What's New**: 본 논문에서는 DreamMix라는 새로운 확산 기반 생성 모델을 소개합니다. 이 모델은 주어진 장면의 사용자 지정 위치에 대상 객체를 삽입하면서 동시에 텍스트 기반의 수정도 가능하게 합니다. 기존 방법들은 주로 정체성 보존에 초점을 맞추었지만, DreamMix는 객체의 편집성을 극대화하는 데 주력합니다.

- **Technical Details**: DreamMix는 두 단계로 나누어진 disentangled local-global inpainting 프레임워크를 사용하여 정확한 객체 삽입과 효과적인 전반적인 시각 일관성을 모델링합니다. 또한 Attribute Decoupling Mechanism (ADM)과 Textual Attribute Substitution (TAS) 모듈을 도입하여 텍스트 기반 속성 가이드를 다양화하고 구별하는 능력을 향상시킵니다.

- **Performance Highlights**: 다양한 주제 기반의 inpainting 애플리케이션에서 DreamMix는 정체성 보존과 속성 편집성 간의 균형을 효과적으로 유지하며 우수한 성능을 보여줍니다. 정량적 메트릭과 주관적 평가 모두에서 이전 접근 방식보다 월등한 성능을 입증하였습니다.



### AIGV-Assessor: Benchmarking and Evaluating the Perceptual Quality of Text-to-Video Generation with LMM (https://arxiv.org/abs/2411.17221)
- **What's New**: 이번 연구에서는 AI로 생성된 비디오(AIGV)의 품질 평가를 위한 새로운 데이터셋인 AIGVQA-DB를 제시합니다. 이 데이터셋은 15개의 텍스트-비디오 생성 모델이 1,048개의 다양한 프롬프트를 사용하여 생성한 36,576개의 AIGV 비디오로 구성되어 있습니다. 이는 기존의 비디오 품질 평가 모델(VQA)이 AIGV에 대한 독특한 왜곡을 제대로 평가하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: AIGVQA-DB 데이터셋은 370k 전문가의 평가를 포함하고 있으며, 정적 품질, 시간적 부드러움, 동적 정도 및 텍스트-비디오 일치도와 같은 4가지 차원에서 평가됩니다. 또한 새로운 VQA 모델인 AIGV-Assessor는 시공간 특징(spatiotemporal features)과 LMM(large multimodal model) 프레임워크를 활용하여 AIGV의 복잡한 품질 속성을 정확히 포착합니다. 이 모델은 자연어 출력으로 비디오 품질 등급을 분류하고, 정밀한 품질 점수를 생성할 수 있습니다.

- **Performance Highlights**: AIGV-Assessor는 AIGV 품질 평가에서 주목할 만한 성과를 보여주며, 기존 방법들보다 많은 측면에서 향상된 성능을 기록하였습니다. 다양한 실험 결과를 통해 모델의 효과성과 적합성을 입증하였고, 특히 보다 정교한 비디오 비교 기능을 통해 인간의 선호에 더욱 근접한 평가를 가능하게 합니다. 이를 통해 AIGV 분야에서의 품질 평가 방법에 대한 새로운 기준을 제시합니다.



### Promptable Anomaly Segmentation with SAM Through Self-Perception Tuning (https://arxiv.org/abs/2411.17217)
- **What's New**: 이 논문에서는 Anomaly Segmentation에서 Segment Anything Model (SAM)의 인식 능력을 강화하기 위해 Self-Perception Tuning (SPT)라는 새로운 방법을 제안합니다. SPT는 초기의거칠게 작성된 anomaly mask 초안을 생성하는 전략과 이를 정밀화하는 과정을 포함합니다. 이를 통해 SAM이 산업적 시나리오에서의 적용에 있어 제한된 의식을 극복하도록 목표하고 있습니다.

- **Technical Details**: SPT 방법론은 Self-Draft Tuning (SDT) 전략을 요구하며, 이를 통해 SAM이 이상 마스크의 초안을 생성한 후 마스크를 정제하는 과정을 거칩니다. 또한, Visual-Relation-Aware Adapter (VRA-Adapter)를 도입하여 디코딩 과정에서의 구분적인 관계 정보의 내부 인식을 향상시킵니다. 이로 인해 SAM은 다양한 프롬프트에서의 세분화(Task segmentation) 문제를 보다 정교하게 처리할 수 있습니다.

- **Performance Highlights**: 여러 산업 데이터셋에서의 광범위한 실험 결과는 SPT 방법이 기존의 방법보다 현저히 우수한 성능을 보임을 입증합니다. 새로운 탐지 모델은 다양한 프롬프트 하에서도 SAM의 강력한 일반화 능력을 향상시켜 이러한 효과를 잘 보여줍니다. 논문은 모델과 코드를 온라인에서 제공할 예정입니다.



### MAT: Multi-Range Attention Transformer for Efficient Image Super-Resolution (https://arxiv.org/abs/2411.17214)
- **What's New**: 이번 연구에서는 Transformer 아키텍처를 기반으로 한 Multi-Range Attention Transformer (MAT)를 제안합니다. MAT는 dilation 연산을 활용하여 self-attention 메커니즘의 계산적 효율성을 높이며, 다중 범위 주의(multi-range attention)와 희소 다중 범위 주의(sparse multi-range attention)를 결합하여 지역적 및 희소한 전역적 특성을 효과적으로 포착합니다. 이로 인해 모델의 중간 피처 다양성과 효과적인 수용 필드를 개선할 수 있음을 입증합니다.

- **Technical Details**: MAT는 dilated convolutions를 기반으로 하여 지역적 주의(attention)와 희소 전역적 주의를 통합합니다. 이를 통해 다양한 스페셜 범위에서 피처 간의 의존성을 효율적으로 포착합니다. 또한, Local Aggregation Block (LAB)과 다중 스케일 합성곱(MSConv) 통합을 통해 전통적인 feed-forward network(FNN)를 대체하는 MSConvStar 모듈을 도입합니다.

- **Performance Highlights**: MAT는 기존의 최첨단(superior) SR 모델보다 더 뛰어난 성능을 보이며, 약 3.3배 빠른 계산 효율성을 제공합니다. 또한, MAT-light는 DIV2K 데이터셋으로 훈련되어 낮은 계산 복잡도에서 SOTA 성능을 달성하며, 전반적으로 다양한 스페셜 범위에서의 피처 포착을 효과적으로 수행합니다.



### Scaling nnU-Net for CBCT Segmentation (https://arxiv.org/abs/2411.17213)
Comments:
          Fabian Isensee and Yannick Kirchhoff contributed equally

- **What's New**: 본 논문은 Cone Beam Computed Tomography (CBCT) 이미지를 사용하여 다중 구조 세분화를 위한 nnU-Net 프레임워크의 확장 접근법을 제시합니다. ToothFairy2 챌린지를 위해 nnU-Net ResEnc L 모델을 활용하고, 패치 크기, 네트워크 토폴로지 및 데이터 증강 전략에 대한 주요 수정 사항을 도입하여 치과 CBCT 이미지의 독특한 문제를 해결하였습니다. 본 방법은 테스트 세트에서 평균 Dice 계수 0.9253과 HD95 18.472를 달성하며, ToothFairy2 챌린지에서 1위를 기록하였습니다.

- **Technical Details**: 우리의 방법은 기본적으로 nnU-Net ResEnc L 구성에 기반하고 있으며, ToothFairy2 데이터셋의 패치 크기를 112x224x256에서 160x320x320으로 증가시켜 공간적 관계를 더 잘 학습할 수 있도록 합니다. 네트워크는 6개에서 7개의 해상도 단계로 깊이를 증가시키고, 왼쪽/오른쪽 구분을 위해 미러링 증강을 비활성화하였습니다. 훈련 기간은 기본 1000에서 1500 에폭으로 연장하였고, 여러 예측 컷오프를 최적화하여 결과의 정확도를 높였습니다.

- **Performance Highlights**: 모델 개발은 ToothFairy2 훈련 분할에 대해 5배 교차 검증을 통해 최적화되었으며, nnU-Net ResEnc L 기본선 모델은 F 케이스에서 부족한 성과를 보였습니다. 왼쪽/오른쪽 미러링을 비활성화 하여 성과를 크게 개선하였으며, 패치 크기를 증가시킴으로써 최고의 모델 성능을 얻었습니다. 최종 제출을 위해 두 개의 모델을 훈련하였으며, 우리의 방법은 복잡한 구조를 정확하게 세분화 및 분류하는 능력을 강조했습니다.



### LampMark: Proactive Deepfake Detection via Training-Free Landmark Perceptual Watermarks (https://arxiv.org/abs/2411.17209)
Comments:
          Accepted to ACM MM 2024

- **What's New**: 이번 연구에서는 Deepfake 공격에 대한 선제적 방어를 위한 새로운 접근 방식인 LampMark라는 랜드마크 지각 워터마크를 제안합니다. 일반적으로 Deepfake 탐지 알고리즘은 수동적으로 작동하였으나, 본 연구에서는 훈련 없이 랜드마크 워터마크를 구성하여 Deepfake를 탐지할 수 있는 방식으로 혁신하였습니다. 이 새로운 접근 방식은 이미지 처리 조작에 대해서도 효율적으로 방어할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LampMark는 얼굴 이미지의 구조적인 정보를 보호하기 위해 얼굴 랜드마크를 활용하여 지각적 워터마크를 생성합니다. 연구자는 랜드마크의 일관성을 분석하고 이들을 기반으로 강력하고 비가시적인 워터마크를 이미지에 삽입하며, 이를 추출하는 시스템을 구성하였습니다. 최종적으로, 이 워터마크의 복원 정확도를 91.83% 이상으로 유지하면서, Deepfake 이미지의 탐지를 위한 유사성을 분석합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 랜드마크 지각 워터마크의 복원 정확도는 각각 128 및 256 해상도에서 91.83% 및 91.86%에 달하며, 일곱 가지 Deepfake 조작을 탐지하는 데 98.39% 및 98.55%의 AUC 점수를 기록하여 기존 방법보다 우수한 성능을 입증하였습니다. 이러한 결과는 본 연구가 제안하는 선제적 탐지 방식의 효과적임을 시사합니다.



### SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting (https://arxiv.org/abs/2411.17190)
Comments:
          Project page: this https URL

- **What's New**: 새롭게 제안된 SelfSplat는 자세(pose)와 3D 정보가 없는 이미지를 사용하여 일반화 가능한 3D 복원을 수행하는 모델입니다. 이 모델은 기존의 3D 복원 방법들이 직면한 무대조 문제를 해결하기 위해 명시적 3D 표현과 자가 지도(depth) 및 자세 추정 기술을 통합하여 높은 품질의 결과를 도출합니다. 특히, 대규모 데이터셋에 대해 우수한 결과를 보이며 다양한 응용 분야에서의 활용 가능성을 보여줍니다.

- **Technical Details**: SelfSplat는 3D-GS(3D Gaussian Splatting) 표현을 기반으로 하여, 자가 지도 학습(self-supervised learning)과 깊이(depth) 추정 기술을 함께 활용하여 깊이, 카메라 자세, 3D Gaussian 속성을 예측합니다. 이 과정에서 뷰 간의 기하학적 일관성을 유지하기 위해 매칭 인식 자세 네트워크와 깊이 정제 모듈을 포함시켰습니다. 이러한 기술을 통해 비디오를 통한 3D 재구성이 가능해집니다.

- **Performance Highlights**: SelfSplat는 RealEstate10K, ACID, DL3DV와 같은 대규모 데이터셋에서 뛰어난 성능을 보이며, 이전의 최첨단 기술들보다 더 높은 시각적 품질과 기하학적 일관성을 달성했습니다. 추가적으로, 이 방법은 다양한 데이터셋 간의 일반화 성능도 뛰어나, 실세계 적용 가능성을 더욱 높였습니다. 또한, 심층적인 아블레이션 연구를 통해 제안된 방법의 효과성을 검증하였습니다.



### PhysMotion: Physics-Grounded Dynamics From a Single Imag (https://arxiv.org/abs/2411.17189)
Comments:
          Project Page: \url{this https URL}

- **What's New**: PhysMotion은 단일 이미지와 입력 조건을 바탕으로 3D 표현을 생성하고 이를 통해 고품질의 물리적으로 타당한 비디오 생성을 가능하게 하는 새로운 프레임워크입니다. 전통적인 데이터 기반 generative 모델의 한계를 극복하고 물리적 현실성을 높인 결과, 더욱 일관성 있는 물리적으로 그럴듯한 동작을 제공합니다. 이를 위해 지나치기 쉬운 깊이의 정보와 함께 소재 물성을 반영하는 혁신적인 접근 방식을 사용합니다.

- **Technical Details**: 이 프레임워크는 단일 이미지에서 피드포워드 3D Gaussian을 재구성하고, 이후 이를 진행시키기 위해 연속 역학 기반의 Material Point Method (MPM)를 사용합니다. 이 방법은 초기 시뮬레이션을 물리적으로 타당하게 만들기 위해 텍스트-이미지(T2I) 확산 모델을 활용하여 영상의 기하학적 외형과 세부 정보가 원본 이미지와 유사한 고품질의 비디오를 생성합니다. 이 과정에서는 교차 프레임 주의를 통해 시공간 일관성을 보장합니다.

- **Performance Highlights**: PhysMotion의 평가에서는 정성적 및 정량적인 성과를 통해 시각적 일관성, 물리적 타당성, 생성의 다재다능성을 입증하였습니다. 이는 기존의 비디오 생성 및 편집 방법과 비교했을 때 더욱 원활한 동작과 비디오 품질을 제공하며, 사용자의 의도를 보다 정확하게 반영할 수 있음을 보여줍니다. 결과적으로, 단일 이미지를 기반으로 한 3D 다이내믹스 생성의 새로운 길을 열었습니다.



### Interleaved Scene Graph for Interleaved Text-and-Image Generation Assessmen (https://arxiv.org/abs/2411.17188)
- **What's New**: 이번 연구에서는 텍스트와 이미지를 교차 생성하는 시스템의 평가를 위한 새로운 프레임워크인 ISG를 소개합니다. 이 프레임워크는 요리책처럼 사용자가 요구하는 정보를 텍스트와 이미지로 함께 제공할 수 있는 모델의 일관성을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: ISG는 장면 그래프(scene graph) 구조를 활용하여 텍스트와 이미지 블록 간의 관계를 포착합니다. 이는 전체적인 일관성(holistic consistency), 구조적 일관성(structural consistency), 블록 수준(block-level), 이미지 특정(image-specific) 네 가지 레벨에서 평가가 가능합니다.

- **Performance Highlights**: ISG-Bench라는 벤치마크 데이터셋을 통해 최근의 통합 비전-언어 모델이 교차 콘텐츠 생성에서 낮은 성능을 보임을 입증했습니다. 형태적 접근(compositional approaches)을 사용할 경우 통합 모델보다 111% 성능 개선을 달성했으며, ISG-Agent를 통해 '계획-실행-정제(plan-execute-refine)' 파이프라인을 적용해 122% 성능 개선을 달성했습니다.



### LiteVAR: Compressing Visual Autoregressive Modelling with Efficient Attention and Quantization (https://arxiv.org/abs/2411.17178)
- **What's New**: 시각적 오토회귀 모델(Visual Autoregressive, VAR)은 이미지 생성에서 유망한 접근 방식으로 떠오르며, 확산 기반 모델(diffusion-based models)과 비교하여 경쟁력 있는 성능을 제공합니다. 본 논문은 VAR 모델의 계산 자원을 줄이기 위해 세 가지 차원에서 중복성을 분석하였고, 이에 대응하여 효율적인 주의 메커니즘(efficient attention mechanism)과 저비트 양자화(low-bit quantization) 방법을 제안합니다. 제안된 방법을 통해 성능 감소 없이 85.2%의 주의 계산량 감소를 이끌어낼 수 있었습니다.

- **Technical Details**: VAR 모델은 고차원 시각 토큰을 기반으로 하는데, 이 과정에서 반복적인 토큰 생성이 필요하여 계산 비용이 발생합니다. 본 연구는 이러한 비용을 감소시키기 위한 훈련 없는(model compression) 모델 압축 기술을 설계하는 데 집중하고 있으며, 핵심적으로 주의 맵의 다중 대각선 특성(multi-diagonal characteristics)을 활용하여 다각적인 주의 기법(multi-diagonal windowed attention, MDWA)을 개발하였습니다. 또한, 데이터의 양자화를 통해 높은 정확도 표현을 줄이고, 필요한 네트워크 층의 정밀도를 조정하여 성능과 효율을 조화롭게 유지하고자 합니다.

- **Performance Highlights**: 연구의 결과로, 제안된 방법을 통해 주의 계산량은 85-90% 줄었으며, 메모리는 50% 경감되고 처리 지연(latency)은 1.5배 단축되었습니다. 이는 VAR 모델이 자원이 제한된 플랫폼에서도 실행 가능성을 높여줄 것으로 기대됩니다. 또한, 실제 환경에서의 효과적인 압축 기법을 개발하여 VAR를 포함한 오토회귀 모델의 실용적 가속화에 기여할 것으로 보입니다.



### ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting (https://arxiv.org/abs/2411.17176)
- **What's New**: 이 논문은 텍스트-이미지 변환(T2I) 생성 모델의 자동화를 통해 사용자가 쉽게 원하는 이미지를 생성할 수 있도록 하는 방법을 제안합니다. 자동 T2I 생성(Automatic T2I)이라는 개념을 도입하여 사용자가 단순히 필요를 설명하면 자동으로 이미지 생성에 필요한 모든 단계를 수행하도록 합니다. 이를 위해 ChatGenBench라는 새로운 벤치마크를 소개하며, 다양한 프리스타일 입력에 대한 평가를 지원합니다.

- **Technical Details**: 제안된 방법인 ChatGen-Evo는 다단계 진화 전략(multi-stage evolution strategy)을 활용하여 모델이 자동화 기능을 점진적으로 습득할 수 있도록 합니다. 이 과정은 프리스타일 입력을 고품질 프롬프트로 변환하는 Supervised Fine-Tuning(SFT), 모델 선택을 위한 모델 토큰(ModelTokens) 형성, 프롬프트와 선택된 모델에 기반해 인자를 설정하는 단계를 포함합니다. 이를 통해 모델은 사용자 기대에 부합하는 출력을 생성할 수 있도록 훈련됩니다.

- **Performance Highlights**: ChatGen-Evo는 여러 기준에서 기존 방법들보다 현저히 우수한 성능을 보입니다. 정확한 단계별 평가와 이미지 품질 평가를 통해 다단계 진화 전략의 중요성을 강조하며, 다양한 실험을 통해 자동 T2I의 도전 과제 및 기회를 밝혀냈습니다. 이 논문은 자동 T2I의 가능성을 높이는 몇 가지 유망한 방향도 제시합니다.



### GMFlow: Global Motion-Guided Recurrent Flow for 6D Object Pose Estimation (https://arxiv.org/abs/2411.17174)
- **What's New**: 이 논문에서는 6D 객체 자세 추정을 위한 새로운 접근 방식인 GMFlow(Global Motion-Guided Recurrent Flow) 방법을 제안합니다. 기존의 자세 정제 방법들이 가려짐(occlusion) 및 부분적 가시성(incomplete visibility) 문제에 효과적으로 대응하지 못했던 동안, GMFlow는 전역적인(global) 맥락 정보를 활용하여 이 문제를 해결합니다. 이 방법은 객체의 구조적 정보를 사용하여 가시 부분의 동작을 비가시 부분으로 확장하며, 더 나아가 객체 형상 제약 조건을 도입하여 자세 추정의 정확성을 높였습니다.

- **Technical Details**: GMFlow는 이미지 기능 추출, 로컬 동작 캡처, 글로벌 동작 집합, 그리고 자세 정제로 구성된 네 가지 기본 요소를 기반으로 합니다. 이 프레임워크는 Gated Recurrent Unit (GRU)을 통해 흐름을 반복적으로 추정하고, 렌더링된 이미지와 실제 이미지 사이의 전이 동작 정보를 결합하여 자세를 정정합니다. 이 과정에서 3D 모델을 활용하여 흐름(iteration)을 제어하고, 최종적으로 휴리스틱한 자세 수정 성과를 도출하도록 설계되었습니다.

- **Performance Highlights**: LM-O 및 YCB-V 데이터셋에서 실험한 결과, GMFlow는 기존 방법들보다 뛰어난 정확도를 보이며 경쟁력 있는 계산 효율성을 유지합니다. 특히, 이 방법은 불완전한 객체 가시성을 효과적으로 처리할 수 있는 능력을 보여줍니다. 또한, GMFlow의 디자인을 통해 데이터 세트에서 최고 수준의 정확성과 강건성을 달성하는 것에 성공했습니다.



### MRIFE: A Mask-Recovering and Interactive-Feature-Enhancing Semantic Segmentation Network For Relic Landslide Detection (https://arxiv.org/abs/2411.17167)
- **What's New**: 이번 논문에서는 고해상도 원격 감지 이미지를 활용하여 시각적 희미함 문제 및 소규모 데이터셋 문제를 해결하기 위해 새로운 의미 분할 모델(MRIFE)을 제안합니다. MRIFE 모델은 상호작용 피처 증강 아키텍처를 사용하여 강력한 피처 추출 및 분리를 수행하는데 중점을 둡니다. 이 모델은 특히 재고사(대부분)이 되는 산사태의 정확도를 개선하는 데 성공했습니다.

- **Technical Details**: MRIFE 모델은 두 개의 주요 브랜치를 가지고 있으며, 하나는 피처 증강, 다른 하나는 마스크 복원을 담당합니다. 이 모델은 감독된 대조 학습(supervised contrastive learning) 방식을 통해 비슷한 피처 간의 차별화를 구현합니다. 또한, 자가 증류 학습(self-distillation learning)을 도입하여 샘플 내 및 샘플 간 피처 다양성을 활용하여 모델의 수렴 속도를 증가시키고, 소규모 데이터셋 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 실험 결과, MRIFE는 산사태 감지 성능을 크게 향상시키는 것으로 나타났습니다. 기준선과 비교했을 때 정밀도는 0.4226에서 0.5347로 증가하였고, 평균 교차 비율(IoU)은 0.6405에서 0.6680으로 증가했습니다. 이러한 개선은 MRIFE가 소규모 데이터셋에서도 뛰어난 성능을 발휘하도록 하는 데 성공했다는 것을 보여줍니다.



### OSDFace: One-Step Diffusion Model for Face Restoration (https://arxiv.org/abs/2411.17163)
Comments:
          8 pages, 6 figures. The code and model will be available at this https URL

- **What's New**: OSDFace는 얼굴 복원을 위한 새로운 1단계 확산 모델로, 기존의 다단계 추론 방식의 계산 집약성을 줄이고 실제 환경에서의 적용 가능성을 높입니다. 이 모델은 저품질의 얼굴 이미지를 효과적으로 처리할 수 있는 시각적 표현 임베더(Visual Representation Embedder, VRE)를 도입하여, 얼굴의 정보를 보다 잘 캡처합니다. 또한, 얼굴 인식에서 파생된 얼굴 정체성 손실을 통합하여 정체성 일관성을 더욱 보장합니다.

- **Technical Details**: OSDFace는 VRE를 통해 저품질 얼굴 이미지에서 풍부한 정보를 추출하는데, 이 시스템은 저품질 데이터를 처리하는 시각적 토크나이저(Visual Tokenizer)와 벡터 양자화 사전(Vector-Quantized Dictionary)을 포함합니다. 이는 입력 얼굴을 보강하기 위해 사용되며, GAN(Generative Adversarial Network)로 생성된 얼굴과 실제 데이터를 정렬하는 가이드 모델로 기능하여 복원된 얼굴의 배포 일치를 유도합니다.

- **Performance Highlights**: 실험 결과, OSDFace는 현재 최첨단(SOTA) 방법들을 능가하는 시각적 품질과 정량적 성과를 보였습니다. 고충실도의 자연스러운 얼굴 이미지를 생성하며, 정체성의 일관성을 높이기 위해 얼굴 인식 기반 손실을 활발히 사용합니다. 512×512 이미지에 대한 추론 시간은 약 0.1초로, 빠른 처리 속도를 자랑합니다.



### Enhancing Lane Segment Perception and Topology Reasoning with Crowdsourcing Trajectory Priors (https://arxiv.org/abs/2411.17161)
- **What's New**: 이 논문에서는 자율 주행에서의 차선 세그먼트 인식 향상을 위해 경로(prior) 정보를 통합하는 novel한 접근 방식을 제안합니다. 특히, Argoverse2 모션 예측 데이터셋에서 crowdsourcing된 경로 데이터를 활용하여 현재의 인식 모델을 보강할 새로운 방법론을 모색하였습니다. 이를 통해 온라인 매핑(model) 성능을 크게 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 연구진은 경로 데이터를 rasterized heatmap과 vectorized instance tokens로 인코딩하여 인식 모델에 통합했습니다. 이 외에도, 공간적 정합성과 의미적 정확도를 고려한 confidence-based fusion module을 설계하여 기술상의 문제들을 해결하였습니다. Argoverse2 데이터셋에서 추출된 경로 데이터를 사용하여 lane 세그먼트 인식의 효율성을 높였습니다.

- **Performance Highlights**: 실험은 OpenLane-V2 데이터셋을 기반으로 진행되었으며, 이 연구의 모델은 현재 최첨단 기술을 크게 초월하는 성능을 나타냈습니다. 특히, 적용된 최적의 융합(fusion) 방법은 mAP 및 topology 지표에서 각각 +7.60 및 +4.46의 성과를 기록하며 기존 연구들을 압도했습니다. 이는 새로운 경로 정보의 효과적인 융합이 가능하다는 것을 시사합니다.



### Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2411.17150)
- **What's New**: 이 논문에서는 Open-Vocabulary Semantic Segmentation (OVSS)의 새로운 접근법인 Context-Aware Semantic Segmentation (CASS) 모델을 소개하여, 훈련 없이도 다중 객체의 맥락적 지식을 활용할 수 있도록 했다. 특히, CASS는 비전 기초 모델에서의 스펙트럴 특징을 시각 인코더의 주의 메커니즘에 통합하여 객체 간의 일관성을 강화하였고, 사용자 정의 클래스로 좀 더 정확히 분류할 수 있도록 돕는다. 이러한 접근법은 훈련 데이터 없이도 다양한 데이터셋에서 일반화가 가능하다는 장점을 가지고 있다.

- **Technical Details**: CASS 모델은 (1) 스펙트럴 객체 맥락 증류(Spectral Object-Level Context Distillation)와 (2) 객체 존재 기반 맥락(Object Presence-Driven Context)이라는 두 가지 주요 요소로 구성된다. 스펙트럴 객체 맥락 증류는 비전 기초 모델에서 특정 특징을 기반으로 객체 수준의 맥락을 추출하여 CLIP의 주의 메커니즘에 통합하는 과정이다. 이 모델은 CLIP의 제로샷 객체 분류를 활용하여 입력 이미지의 특정 객체와 정렬되는 텍스트 embedding을 조정하는 방식으로, 명확한 객체 맥락을 확보할 수 있도록 돕는다.

- **Performance Highlights**: CASS 모델은 학습이 필요 없는 OVSS 분야에서 최첨단 성능을 달성하였다. 다양한 시맨틱 세그멘테이션 데이터셋에서 기존 방법들보다 더 나은 성능을 보였으며, 객체 수준의 일관성을 크게 향상시키는 데 성공하였다. 이러한 개선사항은 실제 애플리케이션에서 보다 효과적으로 사용할 수 있는 잠재력을 가지고 있다.



### Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation (https://arxiv.org/abs/2411.17141)
Comments:
          Work in progress

- **What's New**: 본 논문에서는 다양한 센서의 멀티모달 입력을 동시에 사용하여 세그멘테이션 모델을 훈련하는 것이 직관적이지만, 실제로는 도전적인 과제임을 다룹니다. 특히, unimodal bias 현상을 해결하기 위한 첫 번째 robust segmentor 학습 프레임워크를 개발하였습니다. 이 프레임워크는 다양한 시각적 모달리티 조합을 처리하도록 설계되어 있으며, 실제 응용 프로그램에서의 성능 저하 문제를 해결합니다.

- **Technical Details**: 이 연구는 Parallel Multimodal Learning (PML) 전략을 통해 강력한 teacher 모델 학습을 지원합니다. 연구진은 unimodal and cross-modal distillation 프로세스를 도입하여 multi-scale representation space에서 feature 수준의 지식을 전이하였습니다. 또한, modality-agnostic semantic distillation을 통해 세그멘테이션을 위한 의미론적 지식 전이를 수행하여 강인성을 강화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 접근 방식들과 비교하여 뛰어난 성능을 보였습니다. mIoU 지표에서 +6.37% 및 +6.15%의 개선을 달성하였고, 다양한 실제 및 합성 벤치마크에서 robust성을 입증하였습니다. 이 연구는 멀티모달 세그멘테이션의 unimodal bias 문제를 효과적으로 해결하는 방법을 제시합니다.



### Crack Detection in Infrastructure Using Transfer Learning, Spatial Attention, and Genetic Algorithm Optimization (https://arxiv.org/abs/2411.17140)
- **What's New**: 이번 논문은 도로, 다리, 건물 등 인프라 구조물의 균열 감지에 대한 새로운 접근 방식을 제안합니다. 전통적인 수동 검사 방법은 인력 소모가 크고 주관적이며 위험성이 크지만, 본 연구에서는 딥러닝 기반의 자동화된 감지 방식을 활용해 정확도를 향상시킵니다. 특히, 전이 학습(transfer learning), 공간 주의 메커니즘(spatial attention mechanism), 유전자 알고리즘(genetic algorithm) 최적화를 통합하여 보다 혁신적인 해결책을 제공합니다.

- **Technical Details**: 본 연구는 ResNet50을 사전 훈련된 모델로 사용하여 균열 감지의 데이터 부족 문제를 해결합니다. 모델에는 공간 주의 레이어와 커스터마이즈된 신경망이 추가되어 균열의 복잡한 패턴을 학습하고 이해할 수 있도록 설계되었습니다. 유전자 알고리즘을 통해 모델 아키텍처를 최적화하여 균열 감지에 특화된 보다 나은 성능을 보장합니다.

- **Performance Highlights**: 제안한 Attention-ResNet50-GA 모델은 0.9967의 정밀도와 0.9983의 F1 점수를 달성하여 기존 방법들을 초월하는 성능을 보여주었습니다. 이 모델은 다양한 조건에서 정확하게 균열을 감지할 수 있어 현대의 건축물 모니터링에서 매우 유용한 응용 가능성을 지니고 있습니다. 이와 같은 접근 방식은 대규모 주석 데이터셋이 부족한 실물 응용 분야에 적합합니다.



### TechCoach: Towards Technical Keypoint-Aware Descriptive Action Coaching (https://arxiv.org/abs/2411.17130)
Comments:
          19 pages, 12 figures

- **What's New**: 이 논문에서는 Descriptive Action Coaching (DAC)이라는 새로운 태스크를 제안하여, 기존의 액션 평가 방법들이 제공하지 못하는 상세한 피드백을 생성하는 방법을 탐구하고 있습니다. 논문은 EE4D-DAC라는 새로운 데이터셋을 구축하여, 각 액션의 성공적인 부분과 개선할 수 있는 점에 대한 계층적 피드백을 제공합니다. 또한, TechCoach라는 새로운 프레임워크를 통해 키포인트 수준에서의 추론을 DAC 프로세스에 명시적으로 통합합니다.

- **Technical Details**: EE4D-DAC 데이터셋은 다양한 액션 비디오와 품질 점수 외에도 키포인트와 인스턴스 수준에서의 상세한 코칭 설명을 포함합니다. TechCoach는 Context-aware Keypoint Reasoner를 통해 비주얼 컨텍스트에서 키포인트 관련 품질 표현을 학습하고, Unified Keypoint-aware Action Assessor (KA2)를 활용하여 전체 코칭 설명과 품질 점수를 제공합니다. 이러한 과정은 점진적인 액션 코칭 주의 마스크를 사용하여 정보 통합을 안내합니다.

- **Performance Highlights**: TechCoach는 기존의 방법들과 비교했을 때 최첨단의 성능을 달성하였고, Context-aware Keypoint Reasoner는 필수적인 설계 요소로 확인되었습니다. 본 연구는 EE4D-DAC 데이터셋을 통해 새롭게 구축된 DAC 벤치마크로, 액션 품질 평가(AQA) 분야의 확장을 위한 출발점을 제공합니다. 제안된 방법의 효과성은 광범위한 실험을 통해 검증되었습니다.



### DOGE: Towards Versatile Visual Document Grounding and Referring (https://arxiv.org/abs/2411.17125)
Comments:
          20 pages, 13 figures

- **What's New**: 최근 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 보다 세밀한 이해 및 유연한 사용자 상호작용을 달성하기 위해 그라운딩(growing) 및 참조(referring) 능력에 중점을 두고 있습니다. 그러나, 시각적 문서 이해 분야에서는 고품질의 미세한 데이터셋과 종합적인 벤치마크의 부족으로 이 능력이 부족합니다. 이를 해결하기 위해 우리는 DOGE-Engine이라는 문서 그라운딩 및 참조 데이터 엔진을 제안하며, 고품질의 두 가지 유형의 문서 데이터를 생성합니다.

- **Technical Details**: 개발된 DOGE-Engine은 140만 개의 다중 세분화 문서 구문 데이터와 70만 개의 다양한 지침 조정 데이터(instruction-tuning data)를 생성합니다. 이 데이터는 포스터, 차트 및 PDF 문서에서 단어, 구, 줄, 단락 및 전체 페이지 수준의 텍스트 상자 주석을 포함하여 기본 텍스트 지역화 및 인식 기능을 향상시키는 데 사용됩니다. 또한, DOGE-Bench를 통해 3가지 문서 유형(차트, 포스터, PDF 문서)에서 7개의 그라운딩 및 참조 작업을 포함한 종합 평가가 가능합니다.

- **Performance Highlights**: DOGE라는 강력한 벤치마크 모델을 통해 다중 세분화 문서 이미지 내에서 텍스트를 정확하게 참조하고 그라운딩할 수 있는 가능성을 제공합니다. DOGE-Bench에서의 성과는 향후 연구에 대한 기준 역할을 하며, 우리의 코드는 데이터와 모델은 커뮤니티 개발을 위해 오픈소스될 예정입니다. DOGE-Engine은 MLLM의 문서 이해 능력을 높이기 위한 세 가지 주요 기여를 통해 문서 그라운딩 및 참조 능력을 혁신적으로 개선하는 데 기여하고 있습니다.



### Advancing Content Moderation: Evaluating Large Language Models for Detecting Sensitive Content Across Text, Images, and Videos (https://arxiv.org/abs/2411.17123)
Comments:
          55 pages, 16 figures

- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 기반의 콘텐츠 검열 솔루션을 탐구하며, OpenAI의 moderation model과 Llama-Guard-3 등 기존 방법들의 성능을 평가합니다. 특히, 최신 LLM(GPT-4o, Gemini 1.5, Llama-3)을 통해 부적절한 텍스트 및 시각 콘텐츠를 감지하는 가능성을 탐구하며, 이미지와 비디오의 민감한 내용을 인식하기 위해 LLM의 비전 기능을 활용합니다.

- **Technical Details**: 연구에서는 LLM을 사용하여 부적절한 텍스트 및 비주얼 콘텐츠를 식별하는 시스템을 제안합니다. 다양한 텍스트 및 비주얼 데이터 세트를 생성하고 평가하였으며, 여기에는 폭력 비디오, 누드 및 포르노그래피가 포함된 이미지, 다양한 뉴스 아티클 및 X 트윗이 포함됩니다. 평가 질문(RQ1~RQ4)은 기존 콘텐츠 검열 모델의 효율성과 LLM의 성능의 비교를 목표로 하고 있습니다.

- **Performance Highlights**: LLM 기반의 콘텐츠 검열 솔루션은 기존의 전통적인 기술에 비해 높은 정확도와 낮은 거짓 긍정 및 거짓 부정 비율을 달성하며, 이를 통해 웹사이트 및 소셜 미디어 플랫폼에서 콘텐츠 규제 및 검열을 효과적으로 수행할 가능성을 보여줍니다. 연구 결과는 이러한 모델들이 다양한 종류의 부적절한 콘텐츠(증오 발언, 폭력, 성적 내용)를 감지하는 데 있어 일관된 예측률을 갖는다는 것을 입증하였습니다.



### PassionSR: Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution (https://arxiv.org/abs/2411.17106)
Comments:
this https URL

- **What's New**: 이번 연구에서는 새로운 post-training quantization 방법인 PassionSR을 제안합니다. 이 방법은 단일 단계의 diffusion 기반 이미지 초해상도 모델인 OSD 모델을 위한 것입니다. 주목할 점은 PassionSR이 8비트 및 6비트 정밀도에서 전체 정밀 모델과 유사한 시각적 결과를 제공하면서도, 빠른 수렴을 위한 분산 정량화 보정(DQC) 전략을 설계했다는 것입니다.

- **Technical Details**: PassionSR은 UNet과 Variational Autoencoder (VAE)라는 두 가지 핵심 구성 요소로 OSD 모델 구조를 단순화함으로써 성능을 최적화합니다. 또한 Learnable Boundary Quantizer (LBQ)와 Learnable Equivalent Transformation (LET)을 도입하여 정량화 과정을 최적화하며, 활성화 분포를 조작합니다. 이 방법을 통해 컴퓨팅 요구량을 줄이고 저장 공간을 절약하는 등의 효과를 기대할 수 있습니다.

- **Performance Highlights**: Comprehensive 실험 결과, PassionSR은 8비트 및 6비트 정밀도에서 1% 미만의 성능 저하를 보였습니다. 또한 최근의 저비트 정량화 방법들과 비교했을 때, PassionSR은 여러 비트 너비에서 성능에서 현저히 우수한 성능을 나타냈습니다. 전체적으로 PassionSR은 8비트 및 6비트 정밀도에서 전통적인 모델에 비해 높은 시각적 품질을 제공합니다.



### {\Omega}SFormer: Dual-Modal {\Omega}-like Super-Resolution Transformer Network for Cross-scale and High-accuracy Terraced Field Vectorization Extraction (https://arxiv.org/abs/2411.17088)
- **What's New**: 이 연구는 최초로 이중 모달(dual-modal) {
Omega}-like 초해상도( super-resolution) Transformer 네트워크를 제안하여 지능형 테라스 필드 비주얼 엑스트랙션(TFVE)을 구현하였습니다. 이러한 접근 방식은 전통적인 다중 스케일 다운샘플링 인코더의 경계 분할 오차를 줄이고, 고해상도( high-resolution) 특성을 통합하는 등의 혁신적인 이점을 제공합니다.

- **Technical Details**: 제안한 {
Omega}-like 네트워크 구조는 스펙트럼(spectral) 데이터와 지형(terrain) 데이터의 풍부한 고수준(high-level) 특성을 통합하여 크로스 스케일(cross-scale) 초해상도 특징을 형성합니다. 또한, 분할 경계 픽셀이 포함된 불확실성을 줄이기 위해 coarse-to-fine 및 공간 위상 의미 관계 최적화(STSRO) 분할 전략을 적용하였습니다.

- **Performance Highlights**: 이 연구에서는 최초로 9개의 중국 지역을 아우르는 DMRVD 데이터셋을 생성하였으며, 전체 범위는 22,441 제곱킬로미터에 달합니다. {
Omega}SFormer의 성능을 평가하기 위해 기존의 네트워크와 SOTA(network state of the art) 네트워크와 비교하였고, mIOU(Mean Intersection over Union) 수치가 각각 0.165, 0.297, 0.128 향상되었습니다.



### Path-RAG: Knowledge-Guided Key Region Retrieval for Open-ended Pathology Visual Question Answering (https://arxiv.org/abs/2411.17073)
- **What's New**: 이 논문에서는 암 치료 선택과 계획에 필수적인 병리 이미지를 통한 질병 진단 및 예후를 다룹니다. 최근 딥러닝(Deep Learning) 접근방법이 복잡한 병리 이미지를 분석하는 데 사용되고 있으나, 임상의(Clinician) 전문가의 조직 구조와 세포 성분에 대한 이해를 종종 간과하는 문제점이 있음을 지적합니다. 이에 따라, Path-RAG라는 새로운 프레임워크를 제안하며, 이는 HistoCartography를 통해 병리 이미지에서 관련 도메인 지식을 검색하여 PathVQA-Open 성능을 크게 향상시킵니다.

- **Technical Details**: Path-RAG는 병리 이미지 분석의 복잡성을 인정하고, HistoCartography를 이용하여 특정 패치(Patch)를 선택하는 인본주의적 AI(Human-centered AI) 접근 방식을 취합니다. 실험 결과, 이 방법이 LLaVA-Med의 정확도를 38%에서 47%로 높일 수 있으며, H&E로 염색된 병리 이미지에서 PathVQA-Open 데이터셋에서 28%의 주목할 만한 향상을 보여주었습니다. 또한, 긴 질문 및 답변 쌍에서는 ARCH-Open PubMed와 ARCH-Open Books에서 각각 32.5%와 30.6%의 일관된 개선을 달성했습니다.

- **Performance Highlights**: 제안된 Path-RAG 프레임워크는 병리 이미지 분석에 있어 뛰어난 성능을 입증했습니다. 특히, 도메인 지식의 가이드를 통해 성능 개선이 발생했으며, 이는 병리 이미지를 활용한 질문 응답 작업에서 유의미한 결과를 나타냅니다. 연구에 사용된 코드와 데이터셋은 공개되어 있어, 후속 연구자들이 활용할 수 있도록 했습니다.



### Geometry Field Splatting with Gaussian Surfels (https://arxiv.org/abs/2411.17067)
- **What's New**: 이번 연구는 Radiance Fields를 이용한 새로운 흐름의 Stochastic Opaque Surfaces의 기하학적 재구성을 제시합니다. 저자들은 Gaussian surfels를 기반으로 한 효율적이고 거의 완벽한 차동 렌더링 알고리즘을 개발했습니다. 이러한 접근은 기존의 Taylor series나 self-attenuation을 제거하여 더 정확한 재구성을 가능하게 합니다.

- **Technical Details**: 본 연구에서는 Gaussian kernels 또는 surfels를 이용해 기하학 필드를 스플랫(splat)하여 불투명 고체의 정밀한 재구성을 수행합니다. 특히, surfels가 기하학에 밀집할 때 발생하는 불연속적인 손실 함수 문제를 해결하고 컬러의 연속성을 보장하는 방법을 제시합니다. 또한 구체적 반사 벡터를 인코딩하는 구면 조화를 활용하여 스페큘러(surface specularity)를 더 효과적으로 처리합니다.

- **Performance Highlights**: 연구 결과, widely-used datasets에서 재구성된 3D 표면의 품질이 크게 개선됨을 입증했습니다. 이 방법은 기존 기술에 비해 불투명 표면의 정확한 렌더링에 중점을 두고 있으며, 다양한 환경에서 일관된 성능을 보여줍니다.



### Relations, Negations, and Numbers: Looking for Logic in Generative Text-to-Image Models (https://arxiv.org/abs/2411.17066)
- **What's New**: 본 연구는 최신 이미지 생성 AI인 DALL·E 3을 활용하여 논리 연산자의 신뢰성 있는 배치에 대해 인간 피험자들의 평가 결과를 분석합니다. 특히, 관계(relations), 부정(negations), 및 정수(numbers)라는 세 가지 형태의 논리 연산자를 탐구하였으며, 생성된 이미지가 50% 이상의 인간 일치 점수를 얻지 못한다는 점이 주목할 만합니다. 이와 더불어 ‘grounded diffusion’ 기법을 사용한 실험에서도 DALL·E 3보다 나쁜 성능을 보였음을 발견하였습니다.

- **Technical Details**: 연구는 DALL·E 3를 대상으로 하며, 청사진에 따른 물리적 관계와 부정, 정수 등을 포함한 간단한 프롬프트를 다룹니다. 연구진은 총 35개의 다른 프롬프트를 생성하여 178명의 인간 평가자에게 평가받았으며, 각각의 프롬프트가 생성된 이미지와 얼마나 잘 일치하는지를 수치적으로 분석하였습니다. 이를 통해 논리 연산자가 상황을 이해하는 데 중요한 역할을 한다는 점에 주목하고, 이를 다룬 기존 문헌과 연결지어 설명하고 있습니다.

- **Performance Highlights**: 연구 결과, DALL·E 3이 생성한 이미지들은 인간이 기대하는 논리 연산자의 사용에 비해 현저히 부족한 성능을 보였습니다. 특히, 부정 프롬프트에 대한 성공률이 매우 낮았으며, 정수 프롬프트에서도 고질적인 문제를 드러냈습니다. 이러한 결과는 AI가 여전히 기본적인 논리 개념을 처리하는 데 어려움을 겪고 있음을 시사하며, 향후 더 나은 성능을 위한 알고리즘 개선의 필요성을 강조합니다.



### SCASeg: Strip Cross-Attention for Efficient Semantic Segmentation (https://arxiv.org/abs/2411.17061)
Comments:
          14 pages, 9 figures

- **What's New**: 이 논문은 SCASeg라는 혁신적인 디코더 헤드를 제안하여 효율적인 의미 분할을 위한 설계를 강조합니다. 이는 기존의 간단한 skip connections 대신, 인코더와 디코더 단계 간의 측면 연결을 활용하여 인코더 특징을 Queries로 사용합니다. SCASeg의 Cross-Layer Block은 서로 다른 인코더 및 디코더 단계에서 계층적 특징 맵을 혼합하여 통합된 표현을 생성하는 데 기여합니다.

- **Technical Details**: SCASeg에서는 쿼리와 키를 스트립 형태로 압축하여 메모리 사용량과 추론 속도를 최적화합니다. 또한, Cross-Layer Block은 컨볼루션의 지역 인식 능력을 통합하여 여러 계층에서의 글로벌 및 로컬 컨텍스트 의존성을 포착하는 데 도움을 줍니다. 이러한 새로운 아키텍처는 효율성과 성능 간의 우수한 균형을 달성합니다.

- **Performance Highlights**: SCASeg의 adaptable decoder는 다양한 벤치마크 데이터 세트인 ADE20K, Cityscapes, COCO-Stuff 164k 및 Pascal VOC2012에서 경쟁력 있는 성능을 발휘하며, 기존의 주요 세분화 아키텍처들을 초월하는 결과를 보였습니다. 실험에서는 SCASeg가 다양한 설정에서 우수한 성능을 기록했음을 보여주었습니다.



### PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation (https://arxiv.org/abs/2411.17048)
- **What's New**: 본 논문에서는 기존의 text-to-video (T2V) 생성 방식에서 벗어나, 개인 특화된 ID 이미지로 구성된 인간 비디오 생성 분야에 대한 새로운 프레임워크인 PersonalVideo를 제안합니다. 이 프레임워크는 고유한 ID 충실도(ID fidelity)를 유지하면서도 원래의 움직임 동역학(motion dynamics)과 의미적 일관성(semantic following)을 보존합니다. 이를 위해, 학습 가능한 Isolated Identity Adapter를 도입하여 기존 T2V 모델의 특성을 해치지 않고 ID를 효과적으로 커스터마이징합니다.

- **Technical Details**: PersonalVideo는 기존 T2V 모델을 기반으로 하며, 주어진 이미지에서 ID를 주입하는 최적화 과정을 통해 학습됩니다. 기존의 T2I 모델을 통한 이미지 재구성 지도(supervision) 방법에서 벗어나, 생성된 비디오에 직접적으로 최신 감독 방법을 적용하여 조정-추론 갭(tuning-inference gap)을 해소합니다. 또한, 비재구성 아이디 손실(non-reconstructive identity loss)을 사용하여 단일 참조 이미지만으로도 생성 결과를 감독하여 과적합(overfitting)을 줄이는 시뮬레이션 프롬프트 증가(simulated prompt augmentation) 방식을 도입합니다.

- **Performance Highlights**: 일련의 실험 결과, PersonalVideo는 높은 ID 충실도를 유지하면서도 기존 T2V 모델의 비디오 생성 품질을 효과적으로 보존하였습니다. 특히, 이 프레임워크는 ControlNet과 스타일 LoRA와 같은 사전 훈련된 구성요소들과 원활하게 통합될 수 있으며, 추가적인 조정(tuning) 작업이 필요하지 않습니다. 이러한 특성 덕분에 AIGC(Artificial Intelligence Generated Content) 커뮤니티에서도 유용한 창작 유연성을 제공합니다.



### Large-Scale Data-Free Knowledge Distillation for ImageNet via Multi-Resolution Data Generation (https://arxiv.org/abs/2411.17046)
- **What's New**: 본 논문에서는 MUlti-ReSolution Data-FreE Knowledge Distillation (MUSE) 기법을 소개합니다. MUSE는 Class Activation Maps (CAMs)를 활용하여 저해상도에서 합성 이미지를 생성함으로써 데이터 지식 증류(Data-Free Knowledge Distillation)의 효율성을 높이고, 중요한 클래스 특성을 보존합니다. 이러한 접근법은 높은 해상도의 데이터셋에서의 기존 방법들의 한계를 극복하고, 대규모 데이터셋에서도 통계적으로 우수한 성능을 보여줍니다.

- **Technical Details**: MUSE는 생성된 이미지가 중요한 클래스 특성을 유지하도록 사용자의 위챗되 있는 Class Activation Maps을 활용합니다. 이 논문에서는 또한 멀티 해상도 생성(multi-resolution generation) 방식을 제안하여 이미지의 저해상도 및 고해상도 특성을 지원하며, 임베딩 다양성(embedding diversity) 기술을 통해 특징 표현을 더욱 풍부하게 유지합니다. 이러한 기술들은 MUSE가 대량의 합성 데이터를 효과적으로 생성하면서도 중요한 정보를 보존할 수 있도록 돕습니다.

- **Performance Highlights**: MUSE는 CIFAR10, CIFAR100, ImageNet 및 그 하위 집합을 포함한 다양한 데이터셋에서 최신 성능을 달성합니다. 실험 결과, MUSE는 ImageNet과 그 하위 집합 실험에서 최대 2자리 수의 성능 향상을 보여줍니다. 이로 인해 MUSE는 대규모 데이터셋에서도 효율적인 지식 전이를 가능하게 하여, 데이터 접근이 제한된 실세계 상황에서도 효과적으로 활용될 수 있습니다.



### 4D Scaffold Gaussian Splatting for Memory Efficient Dynamic Scene Reconstruction (https://arxiv.org/abs/2411.17044)
- **What's New**: 기존의 4D Gaussian 방법들은 높은 시각적 충실도와 빠른 렌더링 속도를 제공하지만, 메모리와 저장 요구사항이 과도하여 실제 적용에 제한이 있었습니다. 본 논문에서는 저장 비용을 획기적으로 낮추면서 4D Gaussians의 시각적 품질과 렌더링 속도를 유지하는 4D 앵커 기반 프레임워크를 제안합니다. 이 방법은 3D 스캐폴딩을 4D 공간으로 확장하고, 압축된 feature vector를 가진 희소한 4D 그리드 정렬 앵커를 활용합니다.

- **Technical Details**: 제안된 방법은 동적 장면을 희소한 4D 그리드와 정렬된 구조화된 앵커를 사용하여 표현하며, 각 앵커는 국소적인 시공간 지역을 나타내는 신경 4D Gaussians 집합을 모델링합니다. 동적 지역을 효과적으로 캡처하기 위해, 제안된 방법에서는 시간 커버리지를 고려한 앵커 증가 전략을 도입하여 충분히 재구성되지 않은 동적 지역에 추가 앵커를 할당합니다. 또한, 신경 속도(neural velocity)와 일반화된 가우시안 분포에서 유도된 시간 불투명도(temporal opacity)를 포함한 두 가지 공식을 통해 신경 Gaussians의 표현 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 4DGS에 비해 저장 요구량을 93.8%까지 줄이면서 또한 최고 수준의 시각적 품질을 달성함을 보여줍니다. 이를 통해 동적 장면에서의 중복성을 주소하고, 빠른 렌더링 속도 역시 유지할 수 있음을 알 수 있습니다. 시각적 품질과 렌더링 속도 측면에서 기존의 방식들을 넘어서며, 실제 세계의 동맥을 효과적으로 포착하는 능력을 가지고 있습니다.



### Free$^2$Guide: Gradient-Free Path Integral Control for Enhancing Text-to-Video Generation with Large Vision-Language Models (https://arxiv.org/abs/2411.17041)
Comments:
          15 pages

- **What's New**: Diffusion 모델은 텍스트-이미지(T2I) 및 텍스트-비디오(T2V) 생성 작업에서 놀라운 성과를 보여주고 있습니다. 그러나 T2V 생성 과정에서 텍스트 정렬을 정확하게 이루는 것이 여전히 도전 과제입니다. 본 논문에서는 추가 모델 교육 없이 생성된 비디오와 텍스트 프롬프트 간의 정렬을 위한 혁신적인 무기울기(gradient-free) 프레임워크인 Free$^2$Guide를 제안합니다.

- **Technical Details**: Free$^2$Guide는 비디오 생성에서 생성된 비디오와 텍스트 프롬프트의 정렬을 위해 경로 적분 제어(path integral control) 원리를 이용합니다. 이 프레임워크는 비유동성(non-differentiable) 보상 함수를 사용하여 확산 모델에 대한 가이드라인을 근사하여 강력한 블랙박스 대형 비전-언어 모델(LVLM)을 보상 모델로 통합합니다. 추가로, Free$^2$Guide는 여러 보상 모델을 유연하게 조합하여 정렬을 향상시킬 수 있습니다.

- **Performance Highlights**: Free$^2$Guide는 다양한 차원에서 텍스트 정렬을 크게 개선시키고 생성된 비디오의 전반적인 품질을 향상시키는 성능을 입증했습니다. 이 방법은 텍스트-비디오 정렬을 위한 기울기 기반 접근법을 요구하지 않으며, 이는 특히 비디오 데이터의 복잡한 시계열 의존성을 탈피하는 데 기여합니다. 실험을 통해 Free$^2$Guide가 기존 방식에 비해 유의미한 개선 효과를 보여 주었음을 강조하고 있습니다.



### Multimodal Alignment and Fusion: A Survey (https://arxiv.org/abs/2411.17040)
Comments:
          210+ references

- **What's New**: 이번 서베이는 텍스트, 이미지, 오디오 및 비디오와 같은 다양한 데이터 유형에 대한 기계 학습 내의 최근 발전을 포괄적으로 검토합니다. 다중 모드 데이터 통합(multimodal integration)은 서로 다른 모드의 보완 정보를 활용하여 모델의 정확도와 적용성을 향상시킬 수 있습니다. 이러한 통합은 데이터가 제한된 상황에서도 지식 전이를 촉진할 수 있도록 도와줍니다.

- **Technical Details**: 다중 모드 정렬(multimodal alignment) 및 융합(fusion) 기술을 체계적으로 분류 및 분석하였으며, 200편 이상의 관련 논문으로부터 귀중한 통찰력을 도출하였습니다. 정렬은 서로 다른 모드 간의 의미적 관계를 수립하는 데 초점을 맞추고, 융합은 여러 모드의 정보를 통합하여 통일된 예측을 수행하는 것입니다. 이 서베이는 현대 기술을 효과적으로 나타내고, 향후 발전 방향을 제시하는 데 중점을 둡니다.

- **Performance Highlights**: 다중 모드 방법들은 소셜 미디어 분석, 의료 영상 및 감정 인식 등 여러 분야에서 유망한 결과를 보여주고 있습니다. 특히, 멀티모달 대형 언어 모델(MLLMs)에 대한 연구가 활발히 이루어지고 있으며, 이는 텍스트 처리에서 다른 모드의 작업으로 성능을 확장하기 위해 많은 연구 노력이 집중되고 있습니다. 이러한 모델들은 다중 모드 데이터세트를 활용하여 인공지능 분야의 혁신을 이끌고 있습니다.



### g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks (https://arxiv.org/abs/2411.17030)
- **What's New**: 새로운 3D 표현 모델인 일반화 가능한 3D-언어 특징 필드(g3D-LF)가 소개되었습니다. 이 모델은 대규모 3D-언어 데이터셋에서 사전 학습되어 구현된 작업에 적합합니다. g3D-LF는 에이전트의 포즈 RGB-D 이미지를 처리하여 3D 장면에서의 여러 예측과 목적어 검색을 가능하게 합니다.

- **Technical Details**: g3D-LF는 심층 네트워크 구조를 기반으로 하여 다중 스케일 인코더를 사용하여 언어와 특징 필드의 예측 표현을 정렬합니다. 이 모델은 다양한 맵과 시각 자료를 생성하고, 기존의 2D 기반 모델의 제약을 극복하여 3D 공간 이해를 향상시킵니다. 다층 대비 학습(multi-level contrastive learning)을 통해 언어의 다중 해상도 표현에 맞춰 3D 표현을 구성합니다.

- **Performance Highlights**: g3D-LF는 비전-언어 내비게이션, 제로샷 객체 내비게이션, 상황 기반 질문 답변 작업에서 효과적인 성능 향상을 보여줍니다. 기존 설계된 모델과 비교했을 때, 이 모델은 여러 작업에서 최첨단의 성능을 달성하여 언어 기반의 구현 AI에서의 가능성을 증명합니다. 특히, 대규모 3D-언어 데이터셋의 활용이 강력한 성과를 뒷받침하고 있습니다.



### D$^2$-World: An Efficient World Model through Decoupled Dynamic Flow (https://arxiv.org/abs/2411.17027)
Comments:
          The 2nd Place and Innovation Award Solution of Predictive World Model at the CVPR 2024 Autonomous Grand Challenge

- **What's New**: 이 기술 보고서는 CVPR-2024에서 열린 Predictive World Model Challenge의 2위 솔루션인 D$^2$-World를 상세히 소개합니다. D$^2$-World는 Decoupled Dynamic flow를 통해 미래 포인트 클라우드를 예측하는 혁신적인 월드 모델로, 과거의 시맨틱 점유율을 기존의 점유 네트워크를 활용하여 얻습니다. 이 모델은 비자기회전(non-autoregressive) 방식으로 미래의 점유 상태를 생성하며, 동적 복셀(decoupling)을 통해 작업을 단순화합니다. OpenScene Predictive World Model 벤치마크에서 최첨단 성능을 달성하며, Baseline 모델보다 300% 빠른 학습 속도를 기록합니다.

- **Technical Details**: D$^2$-World는 두 단계로 구성되어 있으며, 첫 번째 단계에서 매 프레임마다 점유를 예측하여 2D 이미지로부터 풍부한 3D 밀집 표현을 회복하려고 합니다. 두 번째 단계에서는 4D 포인트 클라우드 예측 작업으로 접근하며, 기존의 ViDAR 모델처럼 비효율적인 자기회전 방식 대신 비자기회전 방식으로 점유를 예측합니다. 이미지 인코더는 입력된 다중 카메라 2D 이미지를 고수준 특징으로 인코딩하며, LSS를 통해 깊이를 예측해 3D 공간으로 투영합니다. 또한, 시간 정보를 도입하기 위해 과거 볼륨 특징을 동적으로 왜곡하고 융합하여 새로운 특징을 생성하는 기술을 채택합니다.

- **Performance Highlights**: 최종 결과, D$^2$-World는 0.79의 샴퍼 거리(chamfer distance)를 기록하며, 이는 기존 Baseline 모델과 비교하여 상당히 우수한 성능을 나타냅니다. 이 성능은 모델이 과거 점유 결과를 활용하여 미래 점유 상태를 예측하는 프레임워크를 효과적으로 활용한 덕분입니다. 동적 및 정적 복셀 분리를 통해 신뢰성 있는 포인트 클라우드 예측을 가능하게 하여, 이 대회에서 2위를 차지할 수 있었습니다. 더불어, 모델은 200% 증가한 학습 속도를 기록하며 효율성을 보여줍니다.



### RED: Robust Environmental Design (https://arxiv.org/abs/2411.17026)
- **What's New**: 이 논문은 자율 시스템이 도로 표지판을 분류하는 데 있어 발생할 수 있는 적대적 공격(vulnerable to adversarial attacks) 문제를 다루고 있습니다. 전통적인 접근법이 분류 모델의 강인함을 높이는 데 집중한 반면, 본 연구는 도로 표지판 자체의 디자인을 변경하여 강인성을 향상시키는 혁신적인 방법을 제안합니다. 이를 통해 공격에 강한 도로 표지판을 자동으로 설계할 수 있는 공격자 무관 학습(scheme to automatically design road signs) 방식을 도입했습니다.

- **Technical Details**: 제안된 방법인 Robust Environmental Design (RED)은 도로 표지판의 배경을 조작하여 어떤 패치가 붙더라도 분류기를 속이지 못하도록 설계됩니다. RED는 패턴 선택 단계와 분류기 훈련 단계로 구성되며, 따라서 패치 기반 공격(patch-based attacks)에 대한 강인성을 유지할 수 있습니다. 이 과정에서 각 도로 표지판 클래스에 대한 정보를 패턴에 내재적으로 포함시킴으로써 공격을 더욱 어렵게 만듭니다.

- **Performance Highlights**: 실제 및 디지털 환경에서 테스트를 수행한 결과, RED는 기존 기술보다 훨씬 더 높은 강인성을 보여주었습니다. LISA와 GTSRB라는 두 가지 벤치마크 데이터셋을 사용하여 실험을 진행하고, 인쇄된 도로 표지판을 다양한 조명과 기상 조건에서 촬영하여 조작된 입력에 대한 예측 정확도를 확인했습니다. 연구 결과, RED는 물리적 및 디지털 환경 모두에서 공격에 대한 강력한 저항성을 입증했습니다.



### TED-VITON: Transformer-Empowered Diffusion Models for Virtual Try-On (https://arxiv.org/abs/2411.17017)
Comments:
          10 pages, 6 figures, 3 tables, conference

- **What's New**: 이번 논문에서는 기존 Virtual Try-On (VTO) 기술의 한계를 극복하고 Diffusion Transformer (DiT) 기반 모델의 장점을 활용하기 위한 TED-VITON 프레임워크를 제안합니다. 이 새로운 프레임워크는 Garment Semantic (GS) Adapter, Text Preservation Loss 및 Large Language Model (LLM)을 최적화하는 제약 기제를 통합하여 의류 세부 사항, 텍스트 정확성 및 전체적인 비주얼 품질을 향상시킵니다. 이러한 혁신은 VTO 작업에서 새로운 기준을 설정하게 됩니다.

- **Technical Details**: TED-VITON은 VTO 기술을 DiT 기반 아키텍처로 성공적으로 이전하기 위한 여러 혁신적 요소를 포함하고 있습니다. GS-Adapter의 통합을 통해 모델은 이미지 인코더의 고차원 의미적 특성을 정확하게 정렬하여 주름, 표면 질감 및 재료 속성을 유지할 수 있습니다. 또한, 텍스트 왜곡 없이 선명성을 보장하는 Text Preservation Loss를 도입하여 복잡한 디자인의 의류에서도 로고와 텍스트의 질을 향상시킵니다.

- **Performance Highlights**: TED-VITON은 비주얼 품질과 텍스트 충실도에서 최첨단(SOTA) 성능을 발휘합니다. 실험 결과, 기존 GAN 기반 방법에 비해 이 모델은 의류의 세련된 상세 묘사 및 자연스러운 조명 표현에서 더 높은 수준의 정밀도를 보여줍니다. 새로운 제약 메커니즘을 통해 LLM을 활용하여 입출력을 최적화함으로써, 비주얼 피델리티를 더욱 향상시키는 결과를 얻었습니다.



### Event-based Spiking Neural Networks for Object Detection: A Review of Datasets, Architectures, Learning Rules, and Implementation (https://arxiv.org/abs/2411.17006)
Comments:
          63 pages, 15 figures

- **What's New**: 이 논문은 Spiking Neural Networks (SNNs)의 데이터셋, 아키텍처, 학습 방법론, 구현 기술 및 평가 방법론을 체계적으로 리뷰하여 Computer Vision (CV) 기반 객체 탐지 작업에 SNN의 응용 현황을 정리하였습니다. 151개의 저널 및 컨퍼런스 논문을 분석한 결과, 완전 연결형, 컨볼루션형 및 순환형 아키텍처의 효과성, 다양한 학습 방법의 성과, 그리고 신경형 하드웨어 구현의 에너지 소모, 지연 시간 및 메모리의 트레이드 오프를 확인하였습니다.

- **Technical Details**: SNN은 생물학적 뇌의 신경 동역학을 모방한 인공지능 Paradigm으로, 전통적인 인공 신경망(ANNs)과 달리 불연속적이고 사건 기반의 스파이크(spike)를 사용하여 정보를 인코딩하고 처리합니다. 이 연구는 PRISMA 기준에 따라 SNN의 문헌 검색을 진행하며, 다양한 연구 분야에서 SNN의 적용을 탐색하였습니다. 선택된 151개의 논문은 아키텍처, 학습 규칙, 구현 매체와 같은 범주로 체계적으로 분류되었습니다.

- **Performance Highlights**: 성공적인 SNN 아키텍처로는 완전 연결 네트워크(Fully Connected Networks), 컨볼루션 네트워크(Convolutional Networks), 그리고 순환 신경망(Recurrent Neural Networks) 등이 있으며, 30개의 주요 데이터셋과 다양한 평가 메트릭을 기반으로 하여 성능을 평가하였습니다. 연구에 따르면, 정밀도(precision)와 재현율(recall) 등의 지표가 중요하게 다뤄졌으며, SNNs의 다중 응용 가능성 및 접근 방식들이 제시되었습니다.



### Words Matter: Leveraging Individual Text Embeddings for Code Generation in CLIP Test-Time Adaptation (https://arxiv.org/abs/2411.17002)
- **What's New**: 이 연구에서는 CLIP 모델과 같은 시각-언어(Vision-Language) 모델이 테스트 시 발생하는 분포 변화에 어떻게 효과적으로 대처할 수 있는지를 탐구하고 있습니다. 특히, 고정된 클래스 텍스트 임베딩을 중심으로 하는 가짜 레이블(pseudo-labels)을 생성하여 적응 과정을 최적화하였습니다. 제안된 방법인 CLIP-OT는 멀티 템플릿 지식 증류(multiple template knowledge distillation) 접근 방식을 통합하여 자기 지도 표현 학습에서의 멀티 뷰 대조 학습 전략을 모방합니다.

- **Technical Details**: 이 연구는 테스트 시 적응(Test-Time Adaptation, TTA) 문제를 최적 운송(Optimal Transport) 문제로 변환하는 방식을 제안합니다. 여기서 고정된 클래스 프로토타입은 텍스트 표현에서 파생되며, 이는 별도의 샘플 별 감독이 필요 없이 강력한 클래스 가이드를 제공합니다. Sinkhorn 알고리즘을 사용하여 레이블 할당 작업을 효과적으로 해결하며, 복잡한 계산 비용을 피하면서도 다양한 텍스트 프롬프트에서 얻은 정보를 활용합니다.

- **Performance Highlights**: 다양한 테스트 시 적응 벤치마크에서 244개의 시나리오를 통해 CLIP-OT의 우수성이 입증되었습니다. 기존 최신 기법에 비해 최대 7% 성능 향상을 달성하였으며, 이러한 과정에서 계산 및 메모리 효율성을 유지했습니다. 제안된 방법은 실용적인 적용에서 더욱 효과적일 것으로 보입니다.



### SatVision-TOA: A Geospatial Foundation Model for Coarse-Resolution All-Sky Remote Sensing Imagery (https://arxiv.org/abs/2411.17000)
Comments:
          19 pages, 5 figures

- **What's New**: 이 논문에서는 14개 밴드 MODIS L1B Top-Of-Atmosphere (TOA) 데이터로 사전 훈련된 신모델인 SatVision-TOA를 소개합니다. 이 모델은 구름이 없는 이미지만을 이용해 훈련된 기존의 모델과 달리, 다양한 대기 변수와 대기 보정이 필요한 상황에서도 활용될 수 있습니다. 따라서, 중간 및 저해상도 전천후 원거리 데이터 처리를 위한 사전 훈련된 모델의 필요성을 해결하고 있습니다.

- **Technical Details**: SatVision-TOA 모델은 Masked-Image-Modeling (MIM) 프레임워크와 SwinV2 아키텍처를 사용하여 사전 훈련되었습니다. 이 모델은 레이블이 없어도 자기 지도 학습을 통해 상세한 맥락 표현을 학습하며, 30억 개의 파라미터로 구성되어 있고 1억 장의 이미지를 학습했습니다. 이는 위성 원거리 이미지로만 훈련된 모델 중에서 가장 큰 규모의 모델입니다.

- **Performance Highlights**: SatVision-TOA는 3D 구름 검색과 같은 후속 작업에서 기준 방법보다 우수한 성능을 보였습니다. 특히, 모델은 평균 교차점 비율(mIOU) 0.46을 달성하여 기준 모델의 0.22와 비교할 때 상당한 개선을 나타냈습니다. 또한, 세밀한 조정 작업에서 허위 음성 결과 비율이 기준보다 50% 이상 감소하여 매우 긍정적인 결과를 보였습니다.



### Curvature Informed Furthest Point Sampling (https://arxiv.org/abs/2411.16995)
Comments:
          19 pages, 5 figures

- **What's New**: 이 논문에서는 포인트 클라우드(점군)의 다운샘플링을 위해 강화 학습 기반의 샘플링 알고리즘을 제안합니다. 기존의 가장 먼 점 샘플링(FPS) 방식에 곡률 정보를 통합함으로써, 점을 랭킹하여 높은 곡률 포인트를 선택하는 방식으로 성능을 개선하였습니다. 이 접근법은 에지-투-에지 학습(e2e learning)을 통해 안정적인 학습을 가능하게 하며, 여러 다운스트림(하위) 기하학 처리 작업에서 기존 모델보다 뛰어난 성능을 보입니다.

- **Technical Details**: 제안된 곡률 정보 기반 가장 먼 점 샘플링(CFPS) 알고리즘은 각 포인트에 대해 소프트 랭크를 생성한 후, PCPNet을 통해 계산된 곡률 점수를 결합하여 공동 랭크를 만듭니다. 이후 FPS 세트에서 낮은 공동 랭크를 가진 포인트를 높은 공동 랭크를 가진 포인트로 교체하며, 이 과정에서 적절한 점수 교환 비율을 동적으로 조정하여 최적의 성능을 끌어냅니다. 이 방식은 다운샘플링 시 지역 및 전역 기하학적 특성을 효과적으로 포착할 수 있도록 합니다.

- **Performance Highlights**: CFPS는 분류(classification), 세그멘테이션(segmentation) 및 형태 완성(shape completion) 작업에서 최첨단(State-of-the-Art) 결과를 달성하며, 기존의 전통적 또는 차별화된 다운샘플링 방법 것을 능가하는 성능을 자랑합니다. 포괄적인 절단 연구(이탈 연구)를 통해 각 기능의 성능에 대한 질적 및 양적 통찰을 제공하며, 제안된 알고리즘은 높은 견고성과 적응성을 보여줍니다.



### CMAViT: Integrating Climate, Managment, and Remote Sensing Data for Crop Yield Estimation with Multimodel Vision Transformers (https://arxiv.org/abs/2411.16989)
- **What's New**: 이 논문에서는 기후 및 관리 요소를 고려한 픽셀 수준의 포도원 수확량 예측을 위해 새로운 다중 모델인 Climate-Management Aware Vision Transformer (CMAViT)를 소개합니다. CMAViT는 원격 감지 데이터와 단기 기상 데이터를 통합하여 성장 시즌 변동의 영향을 포착하며, 관리 관행을 시간적 데이터와의 상호작용으로 모델링합니다. 이 모델은 2016-2019년 사이에 수집된 2,200 헥타르의 데이터셋을 기반으로 테스트되어 기존의 모델들보다 높은 성능을 나타냈습니다.

- **Technical Details**: CMAViT는 공간적(spatial) 및 시간적(temporal) 데이터의 통합을 통해 포도원 수확량 예측의 정확성을 향상시키도록 설계되었습니다. 이 모델은 관리 관행을 텍스트 형식으로 나타내어 교차 주의 인코더를 사용해 시간 시계열 데이터와의 상호작용을 모델링합니다. 실험 결과, CMAViT는 R2 점수 0.84, MAPE 8.22%로 예측 정확도가 높게 나타났으며, 특정 모달리티를 제외할 경우 성능이 크게 저하되는 것을 보여주었습니다.

- **Performance Highlights**: CMAViT는 UNet-ConvLSTM와 같은 전통적인 모델보다 공간 변동(spatial variability) 포착과 수확량 예측에서 우수한 성능을 발휘하고, 특히 포도원의 극단적인 값에 대한 예측 정확성이 뛰어납니다. 관리 관행, 기후 데이터의 특정 모달리티를 망가뜨리면 R2 점수가 0.73에서 0.70으로 저하되고, MAPE는 각각 11.92%, 12.66%로 증가하여 정확한 수확량 예측을 위한 각 모달리티의 중요성을 강조합니다.



### SEMU-Net: A Segmentation-based Corrector for Fabrication Process Variations of Nanophotonics with Microscopic Images (https://arxiv.org/abs/2411.16973)
Comments:
          Accepted to WACV 2025

- **What's New**: SEMU-Net는 통합 실리콘 광자 장치의 스캐닝 전자 현미경(SEM) 이미지 자동 세분화를 위한 포괄적인 방법을 제안합니다. 이 모델은 두 개의 심층 신경망, 즉 예측기(predictor)와 수정기(corrector)를 사용하여 제조 중 구조적 변형을 예측하고 설계를 조정하여 최종 제작 구조의 정확성을 보장합니다.

- **Technical Details**: SEMU-Net는 U-Net 아키텍처를 기반으로 하여 SEM 이미지를 이진 분류로 세분화합니다. 이 프레임워크는 세분화 U-Net, 예측기 모델, 수정기 모델의 세 가지 모델을 통합하여 진행 과정을 하나의 통일된 체계로 간소화합니다. 실험 결과, 세분화 U-Net은 평균 IoU(Intersection-over-Union) 점수 99.30%를, 수정기 모델은 98.67%를 기록했습니다.

- **Performance Highlights**: SEMU-Net 효과성을 평가하기 위해 다양한 형상의 맞춤 벤치마크를 사용했으며, 각 모델의 성능을 IoU 점수로 측정했습니다. 이 방법은 제작에서 발생하는 변형을 예측하고, 결과적으로 개선된 설계 파일을 통해 원하는 결과와 더 잘 일치하도록 돕습니다.



### ZoomLDM: Latent Diffusion Model for multi-scale image generation (https://arxiv.org/abs/2411.16969)
- **What's New**: 이 논문에서는 다양한 스케일에서 이미지를 생성하는 데 특화된 새로운 확산 모델, ZoomLDM을 소개합니다. 기존의 작은 패치 중심 모델이 가지는 한계를 극복하기 위해, 이 모델은 '줌' 레벨에 따라 이미지를 합성할 수 있는 조건부 메커니즘을 도입했습니다. ZoomLDM의 다중 스케일 특성은 대형 이미지 생성에 필요한 global coherence와 local fidelity를 증대시켜 줍니다.

- **Technical Details**: ZoomLDM은 self-supervised learning (SSL) 임베딩을 활용하여 대형 이미지 도메인에서의 생성 효율성을 높입니다. 이 모델은 크로스-확대(latent space)를 통해 서로 다른 스케일의 정보를 수용하며, 학습하는 과정에서 데이터가 한정된 스케일에서도 생성 품질을 향상시킵니다. 특히, 조건 생성(diffusion model) 기법을 통해 샘플링 시 SSL 임베딩에 대한 의존성을 줄입니다.

- **Performance Highlights**: ZoomLDM은 모든 스케일에서 최첨단 이미지 생성 품질을 달성하며, 특히 대형 이미지의 썸네일을 생성하는 데이터가 부족한 환경에서 뛰어난 성능을 발휘합니다. 이 모델은 4096×4096 픽셀까지의 이미지 생성이 가능하며, 다양한 이미지 학습 작업에서 나타나는 다중 인스턴스 학습(MIL) 실험에서 효과적입니다. ZoomLDM의 멀티-스케일 특성은 강력한 표현력을 갖춘 모델로서, 기존 SoTA(Sate-of-the-art) 인코더보다 우수한 성능을 보여줍니다.



### MotionWavelet: Human Motion Prediction via Wavelet Manifold Learning (https://arxiv.org/abs/2411.16964)
Comments:
          Project Page: this https URL Video: this https URL

- **What's New**: 이 논문에서는 MotionWavelet이라는 새로운 인간 동작 예측 프레임워크를 소개합니다. 이 프레임워크는 Wavelet Transformation을 이용하여 인체 동작을 공간-주파수(domain)에서 분석합니다. MotionWavelet는 사람의 움직임 데이터를 바탕으로 Wavelet Diffusion Model(WDM)을 적용하여 동작의 공간적 및 시간적 패턴을 인코딩합니다. 또한, WDM은 예측 정확도를 높이기 위한 Temporal Attention-Based Guidance를 제공합니다.

- **Technical Details**: MotionWavelet는 인간 동작의 주파수 도메인에서의 특성을 모델링하기 위해 Motion Wavelet Manifold를 활용합니다. Wavelet Diffusion Model은 노이즈 제거 단계에서 wavelet manifold 구조에 적합한 예측을 수행하며, Wavelet Space Shaping Guidance 메커니즘을 통해 결과의 품질을 향상시킵니다. 또한, 자기 주의(self-attention) 메커니즘을 활용하여 시간 차원에서의 동작 일관성을 강화합니다. 이러한 세부 구조는 MotionWavelet가 복잡한 동작 시나리오를 다루는 데 필요한 높은 적응력을 제공합니다.

- **Performance Highlights**: MotionWavelet는 다양한 벤치마크를 통한 실험에서 우수한 예측 정확도를 입증했습니다. 특히, 복잡한 동작에서 높은 일반화 성능을 보였으며, 다양한 운동 스타일을 효과적으로 처리할 수 있는 능력을 강조합니다. 논문은 MotionWavelet의 설계 요소들의 효과성을 입증하기 위한 포괄적인 평가와 분석도 포함하고 있습니다.



### A SAM-guided and Match-based Semi-Supervised Segmentation Framework for Medical Imaging (https://arxiv.org/abs/2411.16949)
- **What's New**: 이 연구에서는 SAMatch라는 새로운 SAM-guided Match-based 프레임워크를 소개합니다. 이는 데이터가 부족한 환경에서 의학 이미지를 반자동으로 분할(segmentation)할 때 의사 레이블(pseudo label)의 품질을 향상시키는 것을 목표로 합니다. SAMatch는 대규모 데이터셋에서 미리 훈련된 SAM을 활용하여 다양한 작업에 대한 일반화(generalization)를 잘 수행하며, 이를 통해 높은 신뢰도의 프롬프트(prompt)를 생성하는 데 도움을 줍니다.

- **Technical Details**: SAMatch는 end-to-end로 훈련되어 모델 간의 동적인 상호작용(dynamic interaction)을 가능하게 합니다. 이 프레임워크는 Match-based 방법들이 일반적으로 직면하는 낮은 품질의 의사 레이블 문제를 해결하는 해결책을 제시합니다. 다양한 데이터셋인 ACDC 심장 MRI, BUSI 유방 초음파, MRLiver에 대한 실험을 통해 SAMatch의 효과를 검증하였습니다.

- **Performance Highlights**: SAMatch는 ACDC, BUSI 및 MRLiver 데이터셋에서 각각 89.36%, 77.76%, 80.04%의 Dice 점수를 기록하며 최고 성능(state-of-the-art)을 달성했습니다. 이 연구는 데이터가 제한된 상황에서 의학 이미지 분할에 있어 매우 유용한 도구를 제공합니다. 관련 코드와 데이터는 제공된 URL에서 확인할 수 있습니다.



### Lens Distortion Encoding System Version 1.0 (https://arxiv.org/abs/2411.16946)
Comments:
          7 pages, 1 figure, 2 tables

- **What's New**: 이번 논문에서는 Lens Distortion Encoding System (LDES)이라는 새로운 시스템을 소개하고 있습니다. LDES는 왜곡을 정확하게 관리하여 고품질의 모션픽쳐 이미지를 렌즈 소스에 관계없이 원활하게 교환할 수 있는 기능을 제공합니다. 이는 기존의 Academy Color Encoding System (ACES)와 유사한 개념으로, 렌즈 간의 교환 없이도 애니메이션 가능한 STMap을 만들어 직접적으로 변형할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: LDES는 일반적인 왜곡 공간을 활용하여 두 가지 요소인 View Map 텍스처와 Footage Map 텍스처로 구성됩니다. View Map은 다양한 렌즈 타입 간의 매끄러운 전환과 애니메이션이 가능하게 하여, 기존에 실현이 불가능했던 왜곡 효과를 창출할 수 있습니다. 또한 LDES는 32비트 STMap 포맷을 사용하여, 대부분의 합성 소프트웨어와 직접적으로 또는 플러그인을 통해 호환됩니다.

- **Performance Highlights**: LDES는 고가의 렌즈 없이도 비슷한 왜곡 효과를 얻을 수 있도록 해주며, 예술적인 제어력을 더욱 향상시킬 수 있습니다. 이러한 특성 덕분에 영상 제작자들은 그들의 예술적 비전을 보다 자유롭게 구현할 수 있으며, 새로운 시각적 표현을 탐구할 수 있는 기회를 가집니다. 전체적으로 LDES는 영상 제작에서 렌즈 왜곡 관리 작업의 효율성을 크게 향상시키고 있습니다.



### Online Episodic Memory Visual Query Localization with Egocentric Streaming Object Memory (https://arxiv.org/abs/2411.16934)
- **What's New**: 본 논문에서는 wearable (착용 가능한) 기기가 과거 비디오 관찰로부터 객체 및 사건들을 회상할 수 있도록 하는 Online Episodic Memory Visual Queries Localization (OEM-VQL)이라는 새로운 작업을 소개합니다. 이 작업은 사용자 쿼리가 발생할 때 모든 비디오 기록에 접근할 수 있다는 비현실적인 '오프라인' 가정 대신, 온라인 방식으로 작동해야 합니다. 이를 통해 실제 환경에서의 활용 가능성을 높이고, 착용 가능한 기기의 파워와 저장 용량의 한계를 고려한 것입니다.

- **Technical Details**: OEM-VQL 작업을 해결하기 위해 제안된 ESOM (Egocentric Streaming Object Memory) 아키텍처는 객체 발견 모듈, 온라인 비주얼 객체 트래커, 그리고 공간-시간 객체 좌표 및 이미지 표현을 저장하는 메모리 모듈로 구성됩니다. 이 아키텍처는 입력된 비디오 프레임을 한 번만 관찰하면서 객체를 계속 추적하고, 메모리를 실시간으로 업데이트함으로써 효율적인 검색이 가능하도록 설계되었습니다. 이는 기존의 오프라인 메모리 검색 방식과 명확히 구별됩니다.

- **Performance Highlights**: ESOM 아키텍처는 Ego4D 에피소딕 메모리 검색 벤치마크에서 81.92%의 성공률을 기록하며, 기존 오프라인 방법의 55.89%에 비해 성능이 우수함을 입증하였습니다. 이 연구는 현재 객체 탐지 및 추적 알고리즘의 한계에도 불구하고, 온라인 에피소딕 메모리 검색 및 객체 탐지/추적 분야에서의 발전을 위한 공정한 기준을 제시합니다.



### Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding (https://arxiv.org/abs/2411.16932)
- **What's New**: 본 논문에서는 Seq2Time이라는 새로운 데이터 중심 훈련 패러다임을 제안합니다. 이 방법은 독립적인 이미지 시퀀스와 짧은 비디오 클립을 활용하여 긴 비디오의 시간 인식을 향상시키고, 자세한 타임스탬프 주석 없이도 효과적인 템포럴 그라운딩(temporal grounding)을 가능하게 합니다. Seq2Time은 이미지와 클립 시퀀스 간의 지식 전이의 원활함을 위해 통합된 상대 위치 토큰(unified relative position token)이라는 새로운 시간 표현을 소개합니다.

- **Technical Details**: Seq2Time의 훈련 과정에서 세 가지 전제 작업을 설계하여 이미지 시퀀스 데이터를 사용하는데, 이 작업들은 특정 이미지 위치를 순서의 맥락에서 찾아내는 것을 요구하여 MLLMs의 시간 이해를 강화합니다. 이미지 시퀀스 데이터(이미지 인덱스 그라운딩, 인덱스 이미지 캡셔닝, 인접 위치 추론)와 클립 시퀀스 데이터(짧은 클립 간의 캡션 및 순서 학습)를 결합함으로써, 대규모의 고품질 텍스트 주석과 함께 긴 시퀀스를 샘플링할 수 있습니다.

- **Performance Highlights**: 실험 결과, Seq2Time은 YouCook2 벤치마크에서 F1 점수에서 27.6%와 CIDEr에서 44.8%의 성능 개선을 보여줍니다. 또한 Charades-STA 벤치마크에서는 14.7%의 리콜 개선을 달성했습니다. 이러한 결과들은 Seq2Time의 효과성을 입증하며, 비디오 LLMs의 시간 인식 능력을 크게 향상시킵니다.



### Context-Aware Input Orchestration for Video Inpainting (https://arxiv.org/abs/2411.16926)
- **What's New**: 본 연구에서는 전통적인 비디오 인페인팅(video inpainting) 방법론의 한계를 극복하기 위해 입력된 프레임의 구성(composition)을 동적으로 조정함으로써 메모리 사용을 최적화하는 새로운 접근 방식을 제시합니다. 연구의 핵심은 정적인 맥락에서는 참조 프레임(reference frame)의 중요성이 증가하고, 역동적인 맥락에서는 이웃 프레임(neighboring frame)의 영향이 클 것으로 가정합니다. 이를 통해 다양한 컨텍스트에서의 인페인팅 품질 개선을 위한 최적의 입력 프레임 구성을 탐구합니다.

- **Technical Details**: 기존의 비디오 인페인팅 모델은 보통 고정된 수의 이웃 프레임과 참조 프레임을 입력으로 사용합니다. 그러나 이 연구에서는 실험을 통해 입력 프레임의 비율이 인페인팅 품질에 미치는 영향을 조사했습니다. 특히, Optical Flow를 적용하여 마스크(mask) 영역의 수정된 흐름을 이용해 입력 프레임 구성을 적응적으로 조정하고, 맥락 변화에 따른 인페인팅 품질의 변화를 분석하고 있습니다.

- **Performance Highlights**: 본 연구에서 제안한 AdaptIn 파이프라인은 입력 프레임 구성을 간단하지만 효과적으로 조정함으로써 인페인팅 품질을 개선하는 방법을 제공합니다. 이 방법론은 과거 연구들이 시각적 동적 요소를 충분히 고려하지 못한 문제점을 해결하며, 정적 및 동적 비디오 콘텐츠 모두에서 일관된 품질 향상을 보여줄 것으로 기대됩니다. 결과적으로, 저자들은 이 연구가 비디오 인페인팅 방법론의 발전에 기여할 수 있음을 주장하고 있습니다.



### Deep Convolutional Neural Networks Structured Pruning via Gravity Regularization (https://arxiv.org/abs/2411.16901)
- **What's New**: 이 논문에서는 중력의 개념을 DCNN의 학습 단계에 통합한 새로운 구조적 가지치기(Structured Pruning) 방법을 제안합니다. 기존의 방법들은 종종 원래 아키텍처를 수정해야 하고 복잡한 구현이나 긴 미세 조정 단계를 요구하지만, 이 새로운 방법은 이러한 문제를 해결합니다. 중력이라는 물리적 개념을 도입하여 필터의 가중치를 효과적으로 재배치하고, 가중치가 중요하지 않은 필터를 제거하는 동시에 중요한 정보를 가진 필터는 유지하도록 해줍니다.

- **Technical Details**: 중력의 개념을 적용하여 컨볼루션 필터 사이의 거리와 질량을 기반으로 가중치를 조정합니다. 중력은 컨볼루션 필터와 유인 필터의 질량 곱에 비례하고, 이들 간의 거리 제곱에 반비례하여 정의됩니다. 이 접근 방식은 기존 SP 방법과 달리 원본 모델 아키텍처의 수정 없이도 구현 가능하며, 교육된 모델을 다양한 가지치기 비율로 압축할 수 있는 장점을 가집니다.

- **Performance Highlights**: CIFAR 데이터셋을 사용하여 인기 있는 DCNN 아키텍처에서 제안된 방법을 검증하며, 기존 방법들과 경쟁력 있는 결과를 달성했습니다. 이 방법은 DCNN의 복잡성을 줄이는 동시에 정확도를 유지할 수 있게 도와줍니다. 또한, 복잡한 구현이나 extensive fine-tuning(광범위한 미세 조정)을 필요로 하지 않으며, 이러한 점에서 효과적입니다.



### G2SDF: Surface Reconstruction from Explicit Gaussians with Implicit SDFs (https://arxiv.org/abs/2411.16898)
- **What's New**: G2SDF는 저해상도 3D Gaussian Splatting (3DGS)의 한계를 극복하기 위해 신경 암시적 Signed Distance Field (SDF)를 통합한 새로운 접근법입니다. 이 방법은 Gaussian의 불투명도 값과 표면과의 거리 간의 관계를 연결하여 표면과 더 가까운 Gaussian을 보다 고품질로 정렬합니다. 또한, 다양한 스케일의 무한 장면에 대한 정규화 함수로 원활한 구현을 지원합니다.

- **Technical Details**: G2SDF는 Hash 기반 SDF를 활용하여 원래 구조가 없는 Gaussian의 공간적 연속성을 보장하며, Gaussian 위치 최적화 시 SDF를 사용하여 더 긴밀한 3D 표현을 만듭니다. 이 방법은 Gaussian 불투명도와 암시적 SDF 간의 미분 가능 연결을 수립하여 3D 표면 재구성과 렌더링의 품질을 향상시킵니다. 다양한 메쉬 추출 기법과 결합할 수 있어 유연성이 높습니다.

- **Performance Highlights**: 실험 결과, G2SDF는 이전의 기술들보다 높은 재구성 품질을 달성하고 3DGS의 효율성을 유지하는 것으로 나타났습니다. 특히, 다양한 실제 무한 데이터 세트에서 고품질 재구성을 제공하면서도 뷰 합성 품질이 저하되지 않음을 보여주었습니다. 이는 기존 Gaussian 렌더러에 손쉽게 적응할 수 있는 점에서도 큰 장점입니다.



### PreF3R: Pose-Free Feed-Forward 3D Gaussian Splatting from Variable-length Image Sequenc (https://arxiv.org/abs/2411.16877)
Comments:
          project page: this https URL

- **What's New**: PreF3R는 카메라 보정을 필요로 하지 않고, 균일한 좌표계 내에서 3D Gaussian 필드를 직접 복원하는 새로운 방법론입니다. 특히, 이 방법은 비포즈 이미지 시퀀스에서의 3D 복원을 가능하게 하여 변동 길이의 이미지 시퀀스에 적용할 수 있습니다. 이는 이전의 접근 방식에 비해 효율적인 새로운 시점(rendering) 생성을 지원합니다.

- **Technical Details**: PreF3R는 DUSt3R의 쌍별 3D 구조 복원 능력을 활용하여 여러 뷰의 입력을 처리하고, 공간 메모리 네트워크를 통해 다중 뷰를 연결합니다. 최적화 기반의 전체 정렬(global alignment) 필요성을 제거하며, 밀집 Gaussian 매개변수 예측 헤드를 도입하여 후속 새로운 시점 합성을 가능하게 합니다. 이 과정에서 모델을 photometric loss와 pointmap regression loss의 조합으로 지도 학습(supervising)하여 사진 실사성과 구조 정확성을 모두 향상시킵니다.

- **Performance Highlights**: PreF3R는 순서가 지정된 이미지 시퀀스를 기반으로 3D Gaussian 필드를 초당 20 프레임(FPS)으로 점진적으로 복원합니다. 이는 실시간 새로운 시점(rendering) 생성을 가능하게 합니다. 실험 결과, PreF3R는 pose-free feed-forward novel-view synthesis의 어려운 작업에서 효과적인 해결책을 제시하며, 보지 않은 장면에 대해서도 강력한 일반화 능력을 보여줍니다.



### RECAST: Reparameterized, Compact weight Adaptation for Sequential Tasks (https://arxiv.org/abs/2411.16870)
- **What's New**: RECAST(재파라미터화된 콤팩트 가중치 적응)는 Incremental Learning(증가 학습)에서 새로운 카테고리를 위한 적응을 가능하게 하면서도, 최소한의 계산 자원으로 효율성을 유지하는 혁신적인 방법입니다. 기존의 Task-specific parameters(작업 특정 매개변수)는 수십만 개에 이를 수 있는 반면, RECAST는 50개 이하의 매개변수로 작동할 수 있습니다. 이는 모듈별 계수와 공유된 가중치 템플릿의 조합을 통해 달성됩니다.

- **Technical Details**: RECAST는 Neural Mimicry(신경 모방)라는 독창적인 가중치 재구성 파이프라인을 갖추고 있어, 기존의 프리트레인(pretrained)된 가중치를 효율적으로 재사용할 수 있습니다. 이 방식은 사전 훈련 없이도 높은 신뢰도로 기존 모델의 가중치를 모방할 수 있어, 다양한 모델 아키텍처에 쉽게 적용될 수 있습니다. RECAST는 또한 cross-layer 지식을 활용하여 기존 아키텍처에 대한 유연한 적응 기능을 제공합니다.

- **Performance Highlights**: RECAST는 CNN과 Transformer 아키텍처에 대해 모두 3%의 성능 향상을 자랑하며, 기존 최첨단 Incremental Learning(증가 학습) 방법들과 비교했을 때 유의미한 결과를 보여주고 있습니다. 추가적으로, RECAST는 높은 매개변수 예산을 가진 어댑터 방법과 결합될 때에도 성능을 2%까지 향상시키는 능력을 보입니다. 이로 인해, RECAST는 효율적인 지속적 학습과 제한된 자원 환경에서의 활용에 적합한 솔루션으로 자리매김하고 있습니다.



### Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering (https://arxiv.org/abs/2411.16863)
- **What's New**: 이번 연구에서는 Multimodal LLMs(MLLMs)의 적응성을 높이기 위해 외부 지식 소스를 통합하는 새로운 방법을 제안합니다. 제안된 모델인 Reflective LLaVA(ReflectiVA)는 외부 지식의 필요성을 동적으로 판단하고 외부 데이터베이스에서 검색한 정보의 관련성을 예측하기 위해 reflective tokens를 활용합니다. 이는 MLLM이 외부 지식을 관리하는 동시에 외부 지식이 필요 없는 작업에서 유창성과 성능을 유지할 수 있도록 합니다.

- **Technical Details**: ReflectiVA 모델은 두 단계의 두 개 모델 훈련 방법을 통해 reflective tokens를 학습합니다. 모델의 어휘를 확장하여 검색의 필요성을 판단하고 검색한 샘플이 입력 쿼리에 대한 적절한지 여부를 결정합니다. 본 연구에서는 Encyclopedic-VQA 및 InfoSeek 데이터셋을 사용하여 제안된 접근 방식의 성능을 검증하며, 이는 외부 지식을 필요로 하는 질문-답변 쌍을 포함하고 있습니다.

- **Performance Highlights**: 제안된 모델은 기존의 방법들과 비교하여 지식 기반 시각 질문 응답에서 우수한 성능을 보였습니다. 광범위한 실험을 통해, ReflectiVA는 모든 고려된 데이터셋과 설정에서 정답 정확도를 증가시키며, 표준 MLLM 벤치마크와 외부 지식이 필요하지 않은 전통적인 VQA 데이터셋에서도 높은 성능을 유지함을 보여주었습니다.



### SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE (https://arxiv.org/abs/2411.16856)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 3D 객체 생성을 위한 새로운 프레임워크인 Scale AutoRegressive 3D (SAR3D)를 제안합니다. SAR3D는 3D 객체를 토큰화하는 멀티 스케일 3D 벡터 양자화 변분 오토인코더(VQVAE)를 활용하여 효율적인 자동 회귀 생성과 상세한 이해를 가능하게 합니다. 이 방법을 통해 기존의 방법보다 더 빠르고 품질 높은 3D 객체 생성을 실현할 수 있습니다.

- **Technical Details**: SAR3D는 3D 객체를 계층적인 수준의 토큰으로 나누어 다중 스케일 표현을 활용하여 다음 스케일을 예측하는 방식으로 학습됩니다. 특히, A6000 GPU에서 단 0.82초 만에 3D 객체를 생성할 수 있으며, 이러한 과정은 기존의 확산 모델이나 전통적인 다음 토큰 예측 방법보다 훨씬 적은 단계로 처리됩니다. 또한, SAR3D는 사전 훈련된 대형 언어 모델(LLM)을 미세 조정하여 3D 컨텐츠를 정밀하게 처리할 수 있도록 해, 멀티모달 입력을 다룰 수 있습니다.

- **Performance Highlights**: 실험을 통해 SAR3D는 현존하는 3D 생성 방법들보다 속도와 품질 모두에서 우수한 성능을 보였습니다. 또한, VQVAE는 LLM이 3D 객체에 대한 상세한 캡션을 생성할 수 있도록 지원하여, 3D 모델의 해석 및 설명 가능성을 높였습니다. 이러한 성과는 SAR3D의 기존 3D 생성 및 이해 방식에 대한 혁신적인 접근법을 제시합니다.



### Open Vocabulary Monocular 3D Object Detection (https://arxiv.org/abs/2411.16833)
Comments:
          Project page: this https URL

- **What's New**: 이 연구에서는 open-vocabulary monocular 3D object detection이라는 새로운 과제를 다루고, 단일 RGB 이미지로 3D 공간에서 객체를 검출하고 위치를 파악하는 접근 방식을 제안합니다. 기존 모델의 한계를 뛰어넘기 위해 class-agnostic 방식을 채택하여 2D 감지기를 활용하고, 2D 경계 상자를 3D로 변환하는 방법을 제공합니다. 이 연구는 객체 인식과 위치 지정을 분리하여, 분류와 상관없이 새로운 카테고리에 대한 일반화를 가능하게 합니다.

- **Technical Details**: 연구는 OVMono3D라는 새로운 과제를 정의하고 이를 평가하기 위한 표준화된 프로토콜을 제안합니다. 초록형 성능 평가를 위해 데이터 셋과 주석 문제를 해결하는 데 중점을 두었으며, 2D 물체 인식과 3D 경계 상자 추정 작업을 분리하는 방법을 채택했습니다. 두 가지 방법인 OVMono3D-GEO와 OVMono3D-LIFT를 통해 단순한 기하학적 원칙을 기반으로 한 방법과 데이터 기반 방법을 비교하면서, 후자가 일반화 성능에서 우수함을 나타냅니다.

- **Performance Highlights**: Omni3D 데이터셋을 통한 실험 결과, 제안된 OVMono3D-LIFT 방법이 새로운 객체 카테고리의 3D 검출에서 최첨단 성능을 달성하였습니다. 이 방법은 다양한 실환경 이미지를 대상으로 한 zero-shot 성능에서도 강력한 일반화 능력을 보여줍니다. 또한, 이 연구는 open-vocabulary object detection 모델 개발에 기여하여 실제 세계의 다양한 카테고리 환경에서 효과적으로 작동할 수 있는 기반을 마련했습니다.



### Edit Away and My Face Will not Stay: Personal Biometric Defense against Malicious Generative Editing (https://arxiv.org/abs/2411.16832)
Comments:
          GitHub: this https URL

- **What's New**: 최근 확산 모델(diffusion models)의 발전은 생성 이미지 편집(generative image editing)을 더 쉽게 가능하게 하였고, 이로 인해 창의적인 편집이 가능해졌습니다. 그러나 이는 사람의 초상화를 대상으로 한 악의적인 편집 등 개인정보와 정체성을 위협하는 윤리적 문제를 일으켰습니다. 본 논문에서는 FaceLock이라고 불리는 새로운 초상화 보호 기술을 제안하며, 이는 생체 인식(biometrics) 정보를 파괴하거나 크게 변화시켜 편집된 결과물이 생체적으로 인식할 수 없도록 합니다.

- **Technical Details**: FaceLock은 얼굴 인식(facial recognition) 모델과 시각적 인식을 통합하여 편집 시도의 다양한 공격에 대해 강력한 보호를 제공합니다. 기존의 방법들이 주로 적대적 간섭(adversarial perturbations)을 사용해 편집 효과를 무효화하려 했던 반면, FaceLock은 생체 인식 모델을 활용하여 얼굴 인식과 기능 임베딩(feature embedding)을 최적화하여 시각적 불일치를 달성합니다. 이러한 방식은 편집된 이미지가 원본 이미지와는 상당히 다른 형태를 가지도록 하고, 사람의 얼굴 특징을 효과적으로 변화시킵니다.

- **Performance Highlights**: 실험 결과 FaceLock은 악의적인 편집에 대한 방어에서 기존의 기준선(baselines)보다 우수한 성능을 보였습니다. 또한, FaceLock은 여러 확산 기반 편집 알고리즘에서 잘 일반화되며 정화(purification) 방법에 대해서도 본질적인 강건성을 보입니다. 본 연구는 생체 방어(biometric defense) 기술의 발전을 도모하고, 이미지 편집에서 개인정보 보호 관행을 설정하는 기초를 마련합니다.



### CLIPS: An Enhanced CLIP Framework for Learning with Synthetic Captions (https://arxiv.org/abs/2411.16828)
Comments:
          12 pages

- **What's New**: 이번 연구에서는 CLIP와 같은 비전-언어 프리트레이닝(framework)에서 노이즈가 많은 웹 크롤링 이미지-텍스트 쌍 대신에 합성 캡션(synthetic captions)을 이용하는 두 가지 효과적인 설계를 도입했습니다. 첫째, 짧은 합성 캡션이 전체 길이의 캡션보다 나은 성능을 나타내며, 이를 활용하여 텍스트 인코더(text encoder)에 부분 합성 캡션만 입력으로 제공했습니다. 둘째, 이미지 입력과 웹 크롤링된 텍스트 설명에 조건화된 오토리그레시브 캡셔너(autoregressive captioner)를 통합하여 합성 캡션을 예측하도록 학습시켰습니다.

- **Technical Details**: 연구진은 CLIP 훈련에서 짧은 합성 캡션의 역효과(inverse effect)를 관찰하여, 파라미터 효율성을 높이고 성능을 개선하는 방안을 도입했습니다. 첫 번째 설계는 랜덤 샘플링을 통해 합성 캡션의 일부만 사용하고, 두 번째 설계는 CoCa에서 영감을 받아 비대칭적(asymmetric) 디코더를 사용하여 웹 크롤링 캡션을 입력으로, 완전한 합성 캡션을 예측하도록 구성했습니다. 이러한 방법은 합성 캡션의 정보 전체를 활용하면서 CLIP 프레임워크의 성능을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, CLIPS는 MSCOCO와 Flickr30K의 제로샷(zero-shot) 성능에서 새로운 SOTA(state-of-the-art) 결과를 기록했습니다. ViT-L 백본(backbone)을 사용할 때, MSCOCO의 텍스트 검색(R@1)에서 4.7 포인트 성능 향상(70.8에서 75.5로)을 보여주었고, 이미지 검색(R@1)에서 3.3 포인트 향상(52.3에서 55.6로)을 기록했습니다. 최적의 모델은 MSCOCO와 Flickr30K 데이터셋에서 각각 76.4%와 96.6%의 텍스트 검색 성능을 달성하며, 다양한 MLLM 벤치마크에서 성능 강화를 확인했습니다.



### Beyond Sight: Towards Cognitive Alignment in LVLM via Enriched Visual Knowledg (https://arxiv.org/abs/2411.16824)
- **What's New**: 본 연구는 Large Vision-Language Models (LVLMs)에서 비전 인코더(vision encoder)와 대형 언어 모델(large language model) 사이의 인지 불일치(cognitive misalignment) 문제를 다룹니다. 이 연구에서는 LVLM의 이해도를 개선하기 위해 다양한 비전 인코더의 표현이 LVLM의 인지 프레임워크에서 어떻게 작용하는지를 조사합니다. 특히, VE-Unknown 데이터가 LVLM의 성능에 미치는 한계와 VE-Known 데이터의 중요성을 강조하며, 새로운 방법인 Entity-Enhanced Cognitive Alignment (EECA)를 제안합니다.

- **Technical Details**: EECA는 다양한 비전 표현을 통합하여 언어모델의 임베딩 공간과 맞춤화된, 시각적으로 풍부한 토큰을 생성하는 방법입니다. 이 과정은 다중 정밀도(multi-granularity) 감독(supervision)을 활용하여 비전 인코더(VE)와 언어 모델(LLM)의 인지적 통합을 가능하게 합니다. 본 연구는 CLIP의 훈련 패러다임에 기반하여 평가 지표를 정의하고, VE-Known 및 VE-Unknown 데이터로 나누어 LVLM의 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, VE-Known 데이터가 있는 경우 LVLM의 이해 능력이 향상되는 것을 확인하였습니다. VE-Unknown 데이터는 LVLM의 해석 가능성을 제한하고 성능 저하를 초래하는 반면, VE-Known 데이터는 인지 불일치를 줄이고 인지적 통합을 강화합니다. 종합적으로, EECA 방법은 랜드마크 인식 성능을 획기적으로 향상시키며, 특히 VE-Known 및 VE-Unknown 데이터 모두에서 기존 모델 대비 현저한 성과를 나타냅니다.



### DetailGen3D: Generative 3D Geometry Enhancement via Data-Dependent Flow (https://arxiv.org/abs/2411.16820)
Comments:
this https URL

- **What's New**: 본 연구에서는 3D 생성 모델의 출력물이 종종 기하학적 세부 정보가 부족함을 해결하기 위해 설계된 DetailGen3D라는 생성적 접근 방식을 제안합니다. 본 방법은 전통적인 3D 생성 모델의 계산 리소스를 피하면서도 고품질 3D 형상을 보존할 수 있는 데이터 의존형 흐름(data-dependent flows)을 통해 조잡한 형태에서 세밀한 형태로의 전환을 모델링합니다. 또한, 지역적 세부정보 합성을 가능하게 하며 전반적인 구조를 유지하는 토큰 매칭(token matching) 전략을 도입하였습니다.

- **Technical Details**: DetailGen3D의 핵심은 조잡한 지오메트리의 질감을 세밀하게 개선하는 데이터 의존형 정류 흐름(data-dependent rectified flow)을 모델링하는 것입니다. 훈련 과정에서, 고품질 3D 형상에 대한 매칭을 통해 확보된 쿼리 포인트(query points)는 전체 조잡한 포인트 클라우드(point cloud)와 공통으로 매핑됩니다. 이러한 과정은 1:1의 토큰 대응을 보장하여 공간 내에서 견고한 일치를 유지하고, 다양한 기하학적 복잡성을 지원합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DetailGen3D가 높은 충실도의 기하학적 세부정보 합성을 달성하며 훈련 효율성을 유지함을 입증하였습니다. 이 방법은 단일 뷰 및 희소 멀티 뷰 입력을 통해 생성된 다양한 3D 형상의 품질을 현저하게 향상시킬 수 있습니다. DetailGen3D의 도입으로 인해 고유한 구조적 일관성을 유지하면서도, 생성된 형태에 대한 세부 조작이 가능하게 됩니다.



### Pathways on the Image Manifold: Image Editing via Video Generation (https://arxiv.org/abs/2411.16819)
- **What's New**: 본 논문은 이미지 편집을 비디오 생성 작업으로 재정의하여 시간적 연속성을 활용한 새로운 접근 방식을 제안합니다. 기존의 이미지 편집 방법들이 편집 정확성과 원본 이미지의 특성 보존에 어려움을 겪는 반면, 이 방법은 원본 이미지를 초기 프레임으로 사용하여 점진적이고 자연스러운 변화를 제공합니다. 특히, 이 새로운 접근법은 최근의 비디오 생성 모델의 복잡한 세계 이해를 활용하여, 물리적으로 그럴듯한 중간 상태를 생성하여 원본과 편집된 이미지 간의 일관성을 유지합니다.

- **Technical Details**: 이 접근법은 Frame2Frame(F2F)라는 구조화된 파이프라인을 통해 구현됩니다. 초기 단계에서는 편집 지침을 Temporal Editing Caption으로 변환하여 비디오 편집이 자연스럽게 어떻게 진화하는지를 설명합니다. 다음으로 최첨단의 이미지-비디오 모델을 사용하여 시간적 캡션에 의해 인도된 일관된 비디오 시퀀스를 생성합니다. 마지막으로, 비전-언어 모델을 활용하여 원하는 편집을 가장 잘 실현하는 프레임을 선택합니다.

- **Performance Highlights**: 실험 결과, 기존의 이미지-이미지 접근 방식을 초월하는 성능 향상이 관찰되었습니다. TedBench와 PosEdit라는 데이터셋에 대한 평가에서 최첨단 성능을 기록하였으며, 이는 생체 포즈 변형에 초점을 맞추고 있습니다. 또한 이 프레임워크는 블러 제거, 노이즈 제거 및 재조명과 같은 전통적인 컴퓨터 비전 문제에서도 유망한 결과를 보여주며, 비디오 기반 이미지 변형의 더 넓은 적용 가능성을 제시합니다.



### SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving (https://arxiv.org/abs/2411.16816)
- **What's New**: 이 논문에서는 자율주행 차량의 안전성을 보장하기 위해 또 다른 획기적인 방법, SplatAD를 제안합니다. SplatAD는 3D Gaussian Splatting (3DGS) 기반의 첫 번째 방법으로, 카메라와 LiDAR 데이터를 모두 처리할 수 있는 실시간 렌더링 기술을 제공합니다. 기존 NeRF(Neural Radiance Fields) 방법들의 렌더링 속도가 느리고 LiDAR 데이터를 지원하지 못하는 제한점을 극복하고 있습니다.

- **Technical Details**: SplatAD는 Rolling Shutter 효과, LiDAR 강도 및 LiDAR 레이 드롭아웃과 같은 센서 특유의 현상을 정확하게 모델링하는 목적 구축 알고리즘을 사용하여 렌더링 효율성을 개선합니다. 새로운 뷰 합성 방법을 도입하여 카메라와 LiDAR 렌더링을 통합하고, 큰 규모의 동적 교통 장면의 실시간 렌더링을 가능하게 합니다. 본 방법은 구형 좌표계에서 희소 포인트 클라우드를 효과적으로 래스터화하는 CUDA 가속 알고리즘을 사용합니다.

- **Performance Highlights**: 세 가지 자율주행 데이터셋에서의 평가 결과, SplatAD는 NeRF 기반 방법보다 렌더링 품질을 최대 +2 PSNR(NVS)와 +3 PSNR(재구성) 향상시켰습니다. 또한, SplatAD는 NeRF 기반 방법에 비해 렌더링 속도를 한 차원 높이며, 모든 벤치마크에서 최첨단 결과를 달성하여 그 효율성과 범용성을 입증했습니다.



### FREE-Merging: Fourier Transform for Model Merging with Lightweight Experts (https://arxiv.org/abs/2411.16815)
Comments:
          16 pages, 5 figures

- **What's New**: 본 논문은 FR-Merging을 제안하여 모델 병합에서 발생하는 작업 충돌의 부정적인 영향을 최소화하는 방법을 소개합니다. 이 새로운 방법론은 주파수 영역 정보(frequency domain information)를 활용하여 harmful specialized information을 효율적으로 필터링하며, 이는 모델의 일반화 능력을 향상시킵니다. 또한, FREE-Merging이라는 경량의 task-specific expert를 도입하여 정보 손실을 보완하여, 다양한 모델에 쉽게 적응할 수 있도록 설계되었습니다.

- **Technical Details**: 기존의 모델 병합 방법은 각각의 작업에 특화된 정보를 종합하면서 성능 저하와 배포 비용의 균형을 맞추기 어려웠습니다. 본 연구에서는 Fourier transform을 통해 모델 파라미터의 주파수 영역을 분석하고, 고주파 성분이 강한 작업 특성을 나타내며 이로 인해 일반화 능력이 저하됨을 설명합니다. FR-Merging는 텍스트와 영상, 다중 모달 영역 등에서 여러 작업에 대해 효과적으로 작동하는 경량 전문가를 사용해 효율적인 기반 모델을 제공합니다.

- **Performance Highlights**: FR-Merging과 FREE-Merging 방법은 비전, 자연어 처리(NLP), 다중 모달 태스크에서 탁월한 성능을 발휘하며, 기존의 기술들과 비교해 높은 수준의 일반화 능력을 보여줍니다. 실험 결과는 이 두 가지 방법이 메모리 및 추론 속도 측면에서도 우수한 성능을 발휘함을 입증하고 있습니다. 본 논문에서 제안한 방법은 향후 다양한 딥러닝 응용 분야에 적용될 잠재력이 큽니다.



### Discrete to Continuous: Generating Smooth Transition Poses from Sign Language Observation (https://arxiv.org/abs/2411.16810)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구에서는 Sign-D2C라는 새로운 프레임워크를 제안하여, 이산적인 수화 세그먼트에서 연속적인 수화 비디오를 생성합니다. 이 모델은 조건부 확산 모델(conditional diffusion model)을 활용하여 맥락적으로 부드러운 전이 프레임을 합성하며, 자연스러운 흐름을 유지할 수 있습니다. 랜덤 마스킹(random masking)을 통해 전이 프레임 생성 문제를 무감독 과제에서 감독 학습(task)으로 변환하여, 보다 의미 있는 수화 비디오를 생성해냅니다.

- **Technical Details**: Sign-D2C는 두 가지 단계로 구성된 프로세스를 채택합니다. 훈련 단계에서는 긴 수화 비디오 내에서 프레임을 무작위로 마스킹하여 전이 프레임 생성 작업을 시뮬레이션합니다. 이 모델은 주변 수화 프레임을 조건으로 하여 가우시안 노이즈(gaussian noise)에서 샘플링하여 마스킹된 프레임을 예측하도록 학습합니다. 추론(inference) 단계에서는 선형 보간(padding strategy)을 적용하여 결측 프레임을 초기화합니다.

- **Performance Highlights**: PHOENIX14T, USTC-CSL100 및 USTC-SLR500 데이터셋에 대한 광범위한 실험 결과, 제안된 방법은 부드럽고 맥락적으로 정확한 전이 프레임을 성공적으로 생성함을 보여줍니다. 전반적으로, Sign-D2C는 기존의 개별 수화 연결 방법의 한계를 극복하며 연속적인 수화 비디오의 품질을 강화하는 데 기여합니다. 이를 통해 수화 언어 처리 분야에서 더 나은 통신 수단을 제공합니다.



### InTraGen: Trajectory-controlled Video Generation for Object Interactions (https://arxiv.org/abs/2411.16804)
- **What's New**: 이 논문에서는 InTraGen이라는 새로운 파이프라인을 소개하여 객체 상호작용 시나리오의 향상된 경로 기반 비디오 생성을 제안합니다. 기존의 text-to-video (T2V) 모델의 한계인 여러 객체 간의 현실적인 상호작용 생성의 부족성을 해결하기 위한 새로운 접근 방식입니다. 또한, 4개의 새로운 데이터 세트와 제안된 InTraGen의 성능을 평가하기 위한 혁신적인 경로 품질 메트릭을 제시하고 있습니다.

- **Technical Details**: 비디오 생성은 많은 복잡한 시간 패턴이 포함되어 있어 이미지 생성보다 더 도전적입니다. 현재 대부분의 비디오 생성 방법은 깊은 학습 모델인 Diffusion 모델을 기반으로 하고 있으며, 객체 상호작용의 타당성을 확보하기 위해 추가적인 제어 신호를 적용할 필요가 있습니다. InTraGen은 객체 ID 삽입 메커니즘과 다중 모달 상호작용 인코딩 파이프라인을 포함하여 비디오 생성 품질을 높이고 객체-환경 간의 상호작용을 풍부하게 합니다.

- **Performance Highlights**: 제안된 모델은 비주얼 충실도와 정량적 성능 모두에서 상당한 개선을 보여줍니다. 다양한 상호작용 시나리오를 포함하는 50K 개의 비디오 데이터 세트를 통해 모델의 성능을 평가했으며, 이 데이터 세트는 객체 간의 실제적이고 정교한 상호작용을 다룹니다. 연구 결과, InTraGen은 기존 방법에 비해 더 나은 객체 상호작용 생성을 제공하며, 코드와 데이터 세트는 공개될 예정입니다.



### Abnormality-Driven Representation Learning for Radiology Imaging (https://arxiv.org/abs/2411.16803)
- **What's New**: 이번 연구는 방사선 영상 분석을 위한 새로운 AI 프레임워크 CLEAR를 소개합니다. 이 프레임워크는 2D 슬라이스에서 추출된 embedding을 사용하며, 주목 기반 집계를 통해 임상 결과를 효율적으로 예측할 수 있도록 설계되었습니다. 특히, lesion-enhanced contrastive learning (LeCL) 방법을 통해 CT 스캔의 비정상적인 패턴을 기반으로 시각적 표현을 얻습니다. 우리는 tumor lesion location, lung disease detection, patient staging의 세 가지 임상 작업을 대상으로 CLEAR의 성능을 평가했습니다.

- **Technical Details**: CLEAR는 다양한 인공지능 아키텍처인 Vision Transformers, Vision State Space Models, Gated Convolutional Neural Networks를 활용하여 방사선 이미지를 분석합니다. 2D 이미지를 사용함으로써 자원 소모를 줄이고 self-supervised learning (SSL)을 통해 효과적인 환자 수준의 바이오마커 개발이 가능합니다. Attention-based Multiple Instance Learning(ABMIL) 기술을 통해 자동으로 정보가 포함된 슬라이스를 식별하는데 중점을 두었습니다.

- **Performance Highlights**: CLEAR는 기존의 foundation 모델보다 더 높은 성능을 보이며, 데이터와 계산 요구 사항을 크게 줄였습니다. BiomedCLIP와 같은 최신 모델에 대해 동등하거나 더 나은 성과를 달성했으며, 학습된 representations를 통해 임상적 작업의 일반화 가능성을 높였습니다. 우리는 이 연구가 방사선학 분야에서의 임상 적용 증가와 체계적인 발전에 기여할 것으로 기대하고 있습니다.



### Controllable Human Image Generation with Personalized Multi-Garments (https://arxiv.org/abs/2411.16801)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 BootComp라는 새로운 프레임워크를 발표합니다. 이 프레임워크는 여러 참조 의상을 바탕으로 한 제어 가능한 인간 이미지 생성에 사용하는 텍스트-이미지(diffusion models) 확산 모델에 기반합니다. 기존의 데이터 수집 방식의 어려움을 극복하기 위해, 합성 데이터 생성 파이프라인을 제안하여, 인간 이미지와 여러 의상 쌍으로 구성된 대규모 합성 데이터셋을 구축합니다.

- **Technical Details**: BootComp는 두 단계의 프레임워크로 구성됩니다. 첫 번째 단계에서는 단일 의상 이미지를 인간 이미지와 매핑하는 분해 모듈을 통해 고급 합성 데이터 생성 파이프라인을 구성합니다. 두 번째 단계에서는 인코더와 생성기를 활용한 T2I 확산 모델의 미세 조정 기법을 통해 복잡한 조건에서 인간 이미지를 생성하는 모델을 학습합니다.

- **Performance Highlights**: BootComp는 기존 최첨단 방법보다 MP-LPIPS 기준으로 30% 향상된 성능을 보여주며, 패션 도메인에서의 활용 가능성을 증명합니다. 이를 통해 가상 착용 서비스 및 다양한 조건에 따른 제어 가능한 인간 이미지 생성에서의 가능성을 강조하며, 다양한 스타일의 인간 이미지를 일반화하여 생성할 수 있는 능력을 보여줍니다.



### Phys4DGen: A Physics-Driven Framework for Controllable and Efficient 4D Content Generation from a Single Imag (https://arxiv.org/abs/2411.16800)
- **What's New**: 본 연구는 Phys4DGen이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 단일 이미지로부터 물리적 원칙을 준수하는 4D 콘텐츠를 생성하는 데 중점을 두고 있습니다. 기존의 비디오 확산 모델에 대한 의존도를 줄이고, 물리 시뮬레이션을 통합하여 더 신뢰할 수 있는 4D 생성 과정을 구현했습니다.

- **Technical Details**: Phys4DGen은 Physical Perception Module (PPM)을 통해 입력 이미지에서 3D 객체의 물성 및 구조를 정확히 식별합니다. 이 모듈은 물리적 특성 예측을 통해 정적인 Gaussian을 생성한 후 물리 시뮬레이션을 수행하여 4D 콘텐츠를 생성합니다. 이 과정은 비디오 확산 모델이 아닌 물리 시뮬레이션에 기반하여 원활하게 진행됩니다.

- **Performance Highlights**: Phys4DGen은 기존 방법에 비해 생성 속도와 물리적 현실감을 모두 향상시킨다는 것을 입증했습니다. 이 프레임워크는 사용자에게 외부 힘을 조절하여 생성된 4D 콘텐츠의 움직임을 세밀하게 조정할 수 있는 기능을 제공합니다. 실험 결과, Phys4DGen은 빠르고 신뢰할 수 있는 4D 콘텐츠 생성으로 높은 품질의, 제어 가능한 결과를 도출했습니다.



### One is Plenty: A Polymorphic Feature Interpreter for Immutable Heterogeneous Collaborative Perception (https://arxiv.org/abs/2411.16799)
- **What's New**: 이번 연구에서는 PolyInter라는 새로운 다형성 특징 해석器를 제안합니다. PolyInter는 각 에이전트의 특정 프롬프트를 재정의함으로써 새로운 에이전트를 원활하게 통합할 수 있는 확장 포인트를 갖추고 있어, 기존의 에이전트의 학습된 파라미터는 재사용하면서도 효율적인 해석이 가능합니다. 이를 통해 컬렉티브 퍼셉션 구현 시 높은 확장성과 저손실 해석을 동시에 달성할 수 있습니다.

- **Technical Details**: PolyInter는 서로 다른 인식 네트워크 구조를 가진 에이전트들 간의 의미적 차이를 해결하기 위한 것으로, 채널 선택 모듈(Selection Module) 및 공간적 주의 모듈(Spatial Attention Module)을 포함하고 있습니다. 이는 이기(ego) 에이전트의 의미 공간과 이웃(Neighbor) 에이전트의 의미 공간 간을 일치시켜 줌으로써, 데이터의 예외적인 다양성을 어느 정도 관리할 수 있도록 도와줍니다. PolyInter는 두 단계의 훈련 과정을 통해 작동하며, 특히 첫 번째 단계에서 에이전트 간의 공통적 의미 정보를 공유합니다.

- **Performance Highlights**: PolyInter의 실험 결과, OPV2V 데이터셋을 통해 기존의 최첨단 해석器(SOTA interpreters)와 비교하여 각각 0.5 및 0.7의 IoU 임계값에서 AP가 각각 7.9% 및 11.1% 향상된 것으로 나타났습니다. 이는 적은 양의 파라미터(단 1.4%)만을 학습시키며 새로운 에이전트에 적응할 수 있음을 보여 주며, PolyInter의 효율성과 효과성을 입증합니다.



### Phase-Informed Tool Segmentation for Manual Small-Incision Cataract Surgery (https://arxiv.org/abs/2411.16794)
- **What's New**: 본 논문은 Manual Small-Incision Cataract Surgery (MSICS)를 위한 첫 번째 포괄적 데이터셋인 Cataract-MSICS를 소개합니다. 이 데이터셋은 53개의 수술 비디오에서 3,527개의 프레임이 포함되어 있고, 18개의 수술 단계와 13개의 수술 도구에 대한 픽셀 수준의 주석이 달려 있습니다. 또한, ToolSeg라는 새로운 프레임워크를 제안하여 수술 단계 정보를 활용하여 도구 분할을 개선합니다.

- **Technical Details**: Cataract-MSICS 데이터셋은 수술 도구 분할을 위한 최첨단 모델에서 벤치마킹되었습니다. ToolSeg는 phase-conditional decoder를 도입하여 도구의 분할 성능을 크게 향상시키는데, 평균 Dice 점수가 23.77%에서 38.10% 향상되었습니다. 또한, Meta의 SAM 2 모델을 활용하여 수동 레이블링 비용을 절감하면서 24,405프레임으로 레이블이 확장되었습니다.

- **Performance Highlights**: ToolSeg는 기존 방법보다 24.46% 높은 Dice 점수를 기록하며, 특히 덜 보편적이고 작은 도구의 분류 및 분할에서 눈에 띄는 개선을 보여줍니다. 이 방법은 CaDIS 데이터셋에서 확인된 바와 같이 다른 수술 환경으로 일반화 가능합니다. 연구는 MSICS 데이터셋이 수술 기구 분할 연구에 중요한 기여를 할 것이며, 저비용 고용량 수술의 맥락에서의 활용 가능성을 보여줍니다.



### ST-Align: A Multimodal Foundation Model for Image-Gene Alignment in Spatial Transcriptomics (https://arxiv.org/abs/2411.16793)
- **What's New**: 이번 연구에서는 ST(Spatial Transcriptomics)를 위한 최초의 기초 모델인 ST-Align을 소개합니다. PET율에서 병리학적 이미지와 유전자 쌍을 결합하고, 공간적 맥락을 통합하여 ST의 특유한 통찰력을 포착합니다. 이 모델은 이미지-유전자 쌍 간의 다중 스케일 정렬과 다중 모달 인사이트의 교차 정렬을 가능하게 합니다.

- **Technical Details**: ST-Align은 ST의 다양한 맥락을 이해하기 위해 특별히 설계된 인코더를 사용하며, 이는 Attention-Based Fusion Network(ABFN)을 통해 다중 모달 데이터를 융합합니다. 이 접근법은 이미지와 유전자의 크기에 대한 적응성을 높이고, 기존의 훈련된 모델에서(domain-shared knowledge) 도메인 공통 지식을 통합하여 ST 특유의 통찰력과 결합합니다.

- **Performance Highlights**: ST-Align은 130만 개의 이미지-유전자 쌍에 대해 사전 훈련이 이루어졌으며, 두 가지 하위 작업인 공간 클러스터 식별 및 유전자 예측에서 뛰어난 성능을 보였습니다. 이 모델은 zero-shot 및 few-shot 상황에서 우수한 범용성을 보여주었으며, ST의 비용을 줄이고 인체 조직 내 주요 조성의 차이를 이해하는 데 중요한 기여를 할 것으로 기대됩니다.



### From Diffusion to Resolution: Leveraging 2D Diffusion Models for 3D Super-Resolution Task (https://arxiv.org/abs/2411.16792)
- **What's New**: 본 연구에서는 2D 확산 모델(2D diffusion model)을 활용하여 3D 전자현미경(vEM) 초해상도(super-resolution)을 개선하는 새로운 접근법인 D2R(Diffusion to Resolution)을 제안합니다. D2R은 XY 평면에서 끊임없는 데이터를 복구하면서 저해상도 볼륨을 순차적으로 회복하는 방식으로 3D 초해상도 모델을 지원합니다. 이 방법은 기존의 3D 볼륨 초해상도 기법이 직면한 구조적 비연속성과 높은 샘플링 비용 문제를 해결합니다.

- **Technical Details**: 제안한 D2R 프레임워크에서는 2D 확산 모델을 사용하여 저해상도 슬라이스를 복구하고, 이러한 슬라이스 시퀀스를 이용해 고주파 인식 3D 초해상도 네트워크인 DGEAN(Deconvolution Gaussian Embedding Attention Network)을 훈련합니다. DGEAN은 슬라이스 간의 구조적 일관성을 유지하면서 고해상도 특성을 효과적으로 캡처합니다. 이 프레임워크는 추가적인 감독 없이도 훈련이 가능하며, 다른 크기의 3D 초해상도 방법에도 손쉽게 적응이 가능합니다.

- **Performance Highlights**: 우리의 방법은 두 개의 공개된 FIB-SEM 데이터셋을 통해 검증되었고, 실험 결과 DGEAN 네트워크와 D2R 프레임워크를 통합한 경우 기존의 비감독 vEM 초해상도 방법들보다 뛰어난 성능을 보였습니다. 이러한 결과는 새로운 접근 방식이 3D 볼륨 초해상도에 실질적인 적용 가능성을 제공함을 보여줍니다. 최종적으로, 우리의 연구는 생물학적 분석에서 고해상도 데이터의 필요성을 충족시키는 중요한 기여를 합니다.



### Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation (https://arxiv.org/abs/2411.16789)
- **What's New**: 이번 논문에서는 Multimodal Sign Language Translation (MMSLT)라는 새로운 접근법을 제안합니다. MMSLT는 사전 훈련된 다중 모달 대형 언어 모델(MLLM)을 활용하여 수화를 점괄할 수 있는 텍스트 설명을 생성하고, 이를 문장 공간에 정렬하여 통역 정확성을 개선합니다. 이 접근법은 고전적인 글로스 기반 모델과 달리, 수화의 비주얼 요소를 직접적으로 분석하여 글로스 어노테이션의 필요성을 제거합니다.

- **Technical Details**: MMSLT는 두 가지 주 모듈로 구성되어 있습니다: Generating Sign Language Description via MLLM (GSD-MLLM)과 Multimodal-Language Pre-training (MMLP)입니다. GSD-MLLM 모듈은 MLLM을 통해 SL 설명을 생성하고, MMLP 모듈은 이 설명의 특징을 수화 이미지와 통합하여 텍스트로 변환합니다. 이를 통해 SL 구성 요소에 대한 세밀한 이해를 가능하게 하고, 변환 과정에서 발생할 수 있는 모달리티 간 간극 (modality gap)을 줄입니다.

- **Performance Highlights**: MMSLT는 두 개의 벤치마크 데이터셋인 PHOENIX14T와 CSL-Daily에서 탁월한 성능을 보여주었습니다. 특히 CSL-Daily 데이터셋에서는 prior 모델들에 비해 현저하게 향상된 BLEU-4 및 ROUGE 점수를 기록하여 SLT (Sign Language Translation) 분야에서 SOTA(최첨단) 성능을 달성했습니다. 이는 MMSLT가 복잡한 구문과 긴 맥락에서 효과적인 번역을 가능하게 함을 시사합니다.



### TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction (https://arxiv.org/abs/2411.16788)
Comments:
          14 pages, 11 figures

- **What's New**: 본 논문에서는 단일 출처 도메인 일반화(Single Source Domain Generalization, SSDG) 문제를 다룹니다. 기존의 방법들이 광범위한 데이터 증강(data augmentation)에 의존하여 다양한 도메인을 학습하려 했으나, 특정 도메인 간의 의미적 변화(semantic shifts)에 대한 견고성이 부족함을 지적합니다. 이 문제를 해결하기 위해, 우리는 모델이 로컬 개념(local concepts)을 활용하도록 유도하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 우리는 새로운 주석 생성 파이프라인을 개발하여, 확산 모델(diffusion models) 및 대형 언어 모델(large language models)의 풍부한 특징을 활용하여 필수 클래스별 개념 및 해당 로컬리제이션 맵을 자동으로 식별합니다. 이어서 TIDE라는 훈련 체계를 도입하고, 이는 개념 중요도 정렬 손실(concept saliency alignment loss)과 로컬 개념 대조 손실(local concept contrastive loss)을 통해 모델에 정확한 개념 집중을 강제합니다. TIDE는 또한 예측 시에 반복적인 정정 알고리즘을 사용하여 개념 예측을 조정합니다.

- **Performance Highlights**: 우리는 TIDE 접근 방식을 네 가지 표준 DG 벤치마크 데이터셋에서 광범위하게 평가한 결과, 평균적으로 현재 최첨단 모델보다 12% 향상된 성능을 보였습니다. 또한, 우리의 예측 결과는 시각적으로 해석 가능하다는 점이 확인되어, 모델의 예측 과정과 결과를 더욱 투명하게 만듭니다. 이러한 성과는 정확한 로컬 개념의 주의를 통해 이루어진 중요한 진전을 나타냅니다.



### MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM (https://arxiv.org/abs/2411.16785)
- **What's New**: MAGiC-SLAM은 기존 SLAM 방법들에서 나타나는 한계를 극복하고자 고안된 다중 에이전트 NVS(새로운 시점 합성) 지원 SLAM 시스템입니다. 이 시스템은 강체 변형을 지원하는 3D Gaussian 기반의 장면 표현을 이용해 속도를 크게 향상시킵니다. 또한 모든 에이전트로부터의 정보를 활용한 경로 정확도를 높이는 루프 클로저 메커니즘을 통합하여 전 세계적으로 일관된 지도를 재구성할 수 있는 가능성을 열어줍니다.

- **Technical Details**: MAGiC-SLAM은 각 에이전트가 RGB-D 스트림을 처리하여 지역적인 매핑(local mapping)과 추적(tracking)을 수행하는 아키텍처를 가지고 있습니다. 각 에이전트는 3D Gaussian으로 표현된 서브 맵(sub-map)을 사용하여 추적 정확도를 향상시키며, 중앙 서버는 이미지 인코딩을 통해 루프를 감지하고 포즈 그래프 최적화를 수행합니다. 최적화된 포즈는 각 에이전트로 되돌려 보내져 지역 지도들을 글로벌 Gaussian 맵으로 융합합니다.

- **Performance Highlights**: MAGiC-SLAM은 합성 및 실제 데이터셋에서 평가된 결과, 기존의 최첨단 기술보다 더 정확하고 빠른 성능을 보여줍니다. 이 시스템은 여러 에이전트의 정보를 통합하여 더 높은 지리적 일관성과 정확성을 가진 글로벌 맵을 생성할 수 있습니다. 또한, 빠르고 효율적인 매핑 및 추적 모듈을 통해 수치적인 저장공간과 처리 시간을 대폭 줄일 수 있습니다.



### CoCoNO: Attention Contrast-and-Complete for Initial Noise Optimization in Text-to-Image Synthesis (https://arxiv.org/abs/2411.16783)
Comments:
          15 pages, 12 figures

- **What's New**: 이번 논문에서는 text-to-image diffusion models에서 발생하는 attention neglect 및 attention interference의 두 가지 주요 문제를 해결하기 위해 CoCoNO라는 새로운 알고리즘을 제안합니다. CoCoNO는 self-attention과 cross-attention 맵의 보완적인 정보를 활용하여 초기 latent를 최적화하며, 각 주체가 고유하게 구분되는 방법으로 이미지를 생성할 수 있도록 합니다. 이를 위해 두 가지 새로운 손실 함수인 attention contrast loss와 attention complete loss를 도입하여 각 subject의 cross-attention와 self-attention 맵 간의 상관관계를 최적화합니다.

- **Technical Details**: CoCoNO는 기존의 latent diffusion 모델 아키텍처에서 작동하며, self-attention 및 cross-attention 맵 간의 상호작용을 최적화합니다. 본 방법은 각 subject token의 cross-attention 맵과 self-attention 맵 간의 매핑을 생성하여, 각 self-attention segment가 특정 token에 할당되도록 합니다. 결과적으로, attention complete loss는 주체 별로 고유한 self-attention segment를 보장하고, attention contrast loss는 주체 간의 혼동을 최소화하여, mixed properties 문제를 해결합니다.

- **Performance Highlights**: 여러 벤치마크에서의 실험 결과, CoCoNO는 기존 최첨단 방식에 비해 text-image 정렬 향상에서 유의미한 개선을 세웠습니다. 초기 noise 최적화 방법인 InitNO와 비교하여, generated 이미지에서 주체의 명확한 표현을 보장하며, attention interference와 neglect 문제를 효과적으로 해결했습니다. 실험을 통해 CoCoNO가 이미지 생성 품질에서도 월등함을 proved했습니다.



### UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing (https://arxiv.org/abs/2411.16781)
- **What's New**: 이 논문에서는 UniPose라는 새로운 프레임워크를 제안합니다. UniPose는 Large Language Models (LLMs)를 활용하여 이미지, 텍스트 및 3D SMPL 포즈와 같은 다양한 양식에서 인간의 포즈를 이해하고 생성하며 수정할 수 있는 기능을 제공합니다. 기존 연구에서 독립적으로 다루어진 포즈의 이해, 생성 및 수정 문제를 통합하였습니다.

- **Technical Details**: UniPose는 3D 포즈를 이산 포즈 토큰으로 변환하는 포즈 토크나이저를 사용합니다. 이로 인해 여러 포즈 관련 작업을 통합할 수 있는 단일 표현 공간을 구축하며, 이를 통해 포즈 이해, 생성 및 편집을 효과적으로 수행합니다. 또한 시각적 인코더의 혼합을 통해 정밀한 포즈 인식을 향상시킵니다.

- **Performance Highlights**: UniPose는 다양한 포즈 관련 작업에서 경쟁력 있는 성능을 보여줍니다. 실험 결과, UniPose는 포즈 이해, 생성 및 편집이 필요할 때 Zero-shot_generalization 능력도 충분히 갖추고 있음을 입증하였습니다. 이런 점에서 UniPose는 일반적인 포즈 처리를 위한 획기적인 접근법으로 자리매김하고 있습니다.



### NovelGS: Consistent Novel-view Denoising via Large Gaussian Reconstruction Mod (https://arxiv.org/abs/2411.16779)
- **What's New**: NovelGS는 Gaussian Splatting (GS)을 위한 새로운 확산 모델로, 희소 보기 이미지에서 3D 가우시안을 생성합니다. 기존 방법은 입력 이미지에 의해 커버되지 않은 영역에서 만족스러운 결과를 제공하지 못했으나, NovelGS는 변환기 기반의 네트워크를 이용하여 노이즈가 있는 뷰와 조건부 뷰를 통합하여 예측합니다. 이 모델은 불균형 입력 이미지와 같은 도전 과제를 효과적으로 해결하는 것을 목표로 합니다.

- **Technical Details**: NovelGS는 조건부 뷰와 노이즈가 있는 뷰를 함께 활용하여 3D 가우시안을 생성하는 변환기 기반의 denoising 네트워크를 사용합니다. 훈련 과정에서 L² 및 LPIPS 손실로 예측된 가우시안에서 깨끗한 뷰와 노이즈 뷰를 렌더링하고 감독합니다. 추론 단계에서는 초기 노이즈 상태에서 타겟 뷰를 순차적으로 denoise하여 최종 가우시안을 얻습니다.

- **Performance Highlights**: NovelGS는 Objaverse의 다중 보기 이미지를 기반으로 훈련되었고, Google Scanned Objects 및 OmniObject3D 데이터셋에서 성능이 평가되었습니다. 실험 결과, 기존 이미지-3D 프레임워크를 정성적으로 및 정량적으로 크게 초과하는 성능을 나타내며, 텍스트-3D 및 이미지-3D 생성 작업에서도 기존 멀티뷰 확산 모델과 통합하여 높은 품질의 결과를 달성했습니다.



### GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis (https://arxiv.org/abs/2411.16778)
Comments:
          This project is available at this https URL

- **What's New**: 이 논문은 의학 분야에서의 비주얼 질문 응답(Visual Question Answering, VQA) 시스템의 문제를 해결하고자 새로운 대규모 데이터셋인 GEMeX를 소개합니다. GEMeX는 가슴 X-선 진단을 위한 데이터셋으로, 비주얼 및 텍스트 설명을 포함한 질문-답변 쌍을 제공합니다. 이를 통해 의사와 환자 간의 이해를 증진시키고, 다양한 질문 형식을 지원하여 실제 임상 환경에서의 응용 가능성을 높입니다.

- **Technical Details**: GEMeX는 151,025장의 X-선 이미지를 포함하고 있으며, 1,605,575개의 질문을 생성했습니다. 이 데이터셋은 오픈형, 폐쇄형, 단일 선택 및 복수 선택 질문 등 다양한 형식으로 구성되어 있습니다. 데이터는 Chest ImaGenome 데이터셋을 기반으로 구축되었으며, GPT-4o를 활용하여 질문을 생성하고, 각 질문에 대해 명시적인 추론 및 시각적 지역 주석이 추가됩니다.

- **Performance Highlights**: 실험에서는 10개의 대표적인 대형 시각 언어 모델(LVLMs)을 평가했습니다. 이 과정에서 기존 모델들이 이 데이터셋에서 성능이 낮았으나, 기본 모델을 훈련 세트로 미세 조정한 후 성능이 크게 향상된 것을 확인했습니다. 이는 GEMeX 데이터셋의 효과성을 보여주며, 의학 VQA 시스템의 신뢰성과 사용자 친화성을 위해 대규모이고 설명 가능한 데이터셋이 필요함을 강조합니다.



### SynDiff-AD: Improving Semantic Segmentation and End-to-End Autonomous Driving with Synthetic Data from Latent Diffusion Models (https://arxiv.org/abs/2411.16776)
Comments:
          15 pages, 10 figures

- **What's New**: 최근 수년 간, 세그멘테이션 및 자율 주행 모델을 개선하기 위해 대규모 데이터셋 수집에 상당한 진전을 이루었습니다. 그러나 이러한 데이터셋은 '맑고 낮'과 같은 일반적인 환경 조건에 치우쳐져 있어, '비오는 밤'과 같은 저평가된 조건에서의 성능이 감소하는 문제를 안고 있습니다. 이를 해결하기 위해, SynDiff-AD라는 새로운 데이터 증강 파이프라인을 도입하여 디퓨전 모델(Diffusion Models)을 활용해 이러한 하위 그룹에 대한 현실적인 이미지를 생성합니다.

- **Technical Details**: SynDiff-AD는 ControlNet이라는 디퓨전 모델을 사용하여 데이터 생성을 위한 조건을 조정하며, 세부적인 하위 그룹 특화된 세멘틱 프롬프트를 생성하는 새로운 프롬프트 스킴을 도입합니다. 이 방법은 기존 샘플의 환경 조건을 변화시켜 드문 시나리오에 대한 데이터 가용성을 증가시키며, 주어진 데이터의 의미적 구조를 유지합니다. 특히, Latent Diffusion Models(LDMs)를 활용하여 현실적이고 의미론적으로 일관된 데이터를 생성함으로써 비싼 수동 레이블링의 필요성을 제거하고, 균형 잡힌 데이터셋으로 AD 모델을 미세 조정할 수 있도록 합니다.

- **Performance Highlights**: SynDiff-AD로 증강된 데이터셋을 통해 Mask2Former와 SegFormer와 같은 세그멘테이션 모델의 성능을 Waymo 데이터셋에서 최대 1.2% 및 2.3% 향상시키고, DeepDrive 데이터셋에서 각각 1.4% 및 0.7% 향상시켰습니다. 또한, SynDiff-AD 파이프라인은 CARLA 자율 주행 시뮬레이터에서 AIM-2D 및 AIM-BEV와 같은 엔드투엔드 자율 주행 모델의 성능을 다양한 환경 조건에서 최대 20%까지 향상시키는 것을 입증하였습니다.



### MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing (https://arxiv.org/abs/2411.16773)
Comments:
          15 pages, 6 figures, 3 tables

- **What's New**: 이 논문에서는 3D 포인트 클라우드 처리(PCP)를 위한 새롭고 향상된 In-Context Learning (ICL) 프레임워크인 MICAS를 제안합니다. MICAS는 포인트 클라우드의 고유한 요구에 맞춘 다중 세분화 적응 샘플링 메커니즘을 도입하여 기존 ICL 방식의 한계를 극복합니다. 이를 통해 기존 방법을 뛰어넘는 효율성과 성능 향상을 보여줍니다.

- **Technical Details**: MICAS는 두 가지 핵심 요소로 구성됩니다: 작업 적응 포인트 샘플링(task-adaptive point sampling)과 쿼리 특정 프롬프트 샘플링(query-specific prompt sampling)입니다. 첫 번째 요소는 작업 간 신호를 활용하여 포인트 수준에서 샘플링을 최적화하며, 두 번째 요소는 각 쿼리에 대해 최적의 프롬프트를 선택하여 작업 내 변동성을 줄입니다. 이를 통해 비가층적 샘플링 방식의 비효율성을 해소하고, 학습 과정을 효율적으로 만듭니다.

- **Performance Highlights**: MICAS는 다양한 PCP 작업을 효율적으로 처리할 뿐만 아니라 이전의 최첨단 방법보다 현저하게 성능을 개선합니다. 특히, 부분 분할 작업(part segmentation task)에서 4.1%의 성능 향상을 달성하며, 여러 PCP 애플리케이션에서도 일관된 성과 향상을 보여줍니다. 이러한 결과는 MICAS의 다양한 3D 포인트 클라우드 작업에 대한 적응성과 견고성을 강조합니다.



### Hyperspectral Image Cross-Domain Object Detection Method based on Spectral-Spatial Feature Alignmen (https://arxiv.org/abs/2411.16772)
- **What's New**: 본 논문은 하이퍼스펙트럴 이미지(HSI)의 비지도 학습 기반 객체 탐지에 대한 새로운 접근 방식을 제안합니다. 특히, 스펙트럼-공간 특성 정렬(Spectral-Spatial Feature Alignment, SFA)을 통해 도메인 간 객체 탐지를 위한 최초의 방법을 소개합니다. 이 연구에서는 HSI 도메인 간 불일치 문제를 해결하기 위해 스펙트럼 자가 상관 모듈(Spectral Autocorrelation Module, SACM)를 설계하였습니다.

- **Technical Details**: 제안된 방법은 도메인 불일치 문제를 해결하기 위해 스펙트럼-공간 정렬 모듈을 사용하여 지역 스펙트럼-공간 특성을 추출합니다. 그 후, 도메인 분류기 내에서 제안된 GRL(Gradient Reversal Layer)을 통해 두 도메인 간 지역 스펙트럼-공간 특성을 정렬합니다. 이러한 방식은 스펙트럼 해상도가 다른 HSI를 효과적으로 정렬하게 도와줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 비지도 학습 기반 객체 탐지 방법들보다 우수한 성능을 보였습니다. 또한, 이는 HSI 교차 도메인 객체 탐지의 발전에 중요한 이정표가 됩니다. 새로운 HSI 데이터셋을 구성하고 주석을 달아 성능 평가에 사용하였으며, 이는 향후 연구에 기여할 것입니다.



### VidHal: Benchmarking Temporal Hallucinations in Vision LLMs (https://arxiv.org/abs/2411.16771)
Comments:
          8 pages, 10 figures. Code available at this https URL

- **What's New**: 본 논문에서는 비디오 기반의 환각(hallucination) 현상을 평가하기 위해 VidHal이라는 벤치마크를 소개합니다. 기존 연구가 주로 이미지 입력에 국한되었던 반면, 비디오에 대한 환각 검토는 상대적으로 부족하였습니다. VidHal은 다양한 시간적(timporal) 측면을 아우르는 비디오 인스턴스를 통해 구성되며, 각 비디오는 환각 수준을 나타내는 여러 캡션으로 주석이 달립니다.

- **Technical Details**: VidHal의 설계는 비디오의 다양한 시간 구조를 포괄적으로 평가할 수 있도록 구성되어 있으며, 모델이 환각 수준에 따라 캡션을 정렬하는 새로운 캡션 정렬 과제를 제안합니다. 이와 함께 NDCG 및 MCQA 정확도와 같은 새로운 평가 지표를 도입하여 기존 모델이 생성하는 세밀한 환각 오류를 측정하도록 하였습니다. 이러한 데이터셋을 활용해 11개의 VLLM을 평가하였으며, 해당 모델들이 환각 구분에 대한 한계를 가지고 있음을 밝혀냈습니다.

- **Performance Highlights**: 실험 결과, 기존 VLLM은 다양한 환각 수준을 구별하는 데 어려움을 겪고 있으며, 특히 비디오의 방향성과 사건의 순서와 같은 비디오 특성 평가에서 큰 개선 여지가 있음을 보여주었습니다. 비공식적인 모델들이 오픈 소스 모델들에 비해 상대적으로 우수한 성능을 보이며, 본 연구는 VLLM의 환각 생성 문제를 해결하기 위한 추가 연구에 대한 통찰을 제공하고자 합니다.



### GAST: Sequential Gaussian Avatars with Hierarchical Spatio-temporal Contex (https://arxiv.org/abs/2411.16768)
- **What's New**: 본 논문에서는 GAST라는 새로운 프레임워크를 제안하여 3D 인간 모델링과 3D Gaussian Splatting (3DGS)을 통합합니다. 이 프레임워크는 비구조적 변형을 위한 연속 조건화 프레임워크를 통해 더 정확한 3D Gaussian을 관측 공간에서 얻을 수 있도록 합니다. 또한 다양한 시간적 스케일에서 샘플링을 수행하여 비구조적 변형을 위한 편향 없는 입력을 보장하는 다중 스케일 샘플링 전략을 도입합니다.

- **Technical Details**: GAST는 모션 시퀀스를 모델링하여 Gaussian의 상응성을 보장함으로써 비구조적 변형 과정에 효과적으로 안내합니다. 기존 NeRF 기반 방법들과는 달리, GAST는 글로벌 인간 포즈 대신 지역화된 정점 잔차를 인코딩하여 더 세밀한 동작 세부사항을 포착합니다. 이러한 계층적 모션 시퀀스 방법은 그로 인해 높은 품질의 렌더링과 함께 동작의 유연함을 극대화합니다.

- **Performance Highlights**: 실험 결과에 따르면, GAST는 I3D-Human 데이터셋 및 DNA-Rendering 데이터셋에서 다른 기준 모델보다 우수한 성능을 보여주며, 복잡한 의류 동작을 포함한 다양한 시나리오에서 보다 나은 렌더링 품질과 애니메이션 가능성을 제공합니다. 이로 인해 GAST는 3D 인간 아바타의 렌더링과 애니메이션에서 처음으로 높은 충실도의 균형을 실현하는 프레임워크로 자리잡게 됩니다.



### Revisiting DDIM Inversion for Controlling Defect Generation by Disentangling the Background (https://arxiv.org/abs/2411.16767)
Comments:
          10 pages

- **What's New**: 본 논문은 산업 anomaly detection에서 데이터 불균형 문제를 해결하기 위해 background와 defect 간의 관계를 모델링하는 방안을 제안합니다. 기존 연구들이 defect 생성의 품질과 제어 가능성을 향상하려 시도했으나, background와 defect 간의 상호작용을 고려하지 못한 점을 지적하고 있습니다. 또한, 논문에서는 background가 defect denoising 과정에 영향을 미치되, 그 반대는 없음을 주장하며 이를 이론적으로 증명합니다.

- **Technical Details**: 모형은 defect와 background 간의 관계를 모델링하여 defect generation을 위한 disentanglement loss 함수를 도입합니다. 이는 defect의 특징이 background 생성에 영향을 미치지 않도록 하여 두 영역의 denoising 과정을 독립적으로 수행할 수 있도록 합니다. DDIM Inversion 기법을 통해 target normal 이미지를 기반으로 defects를 생성하는 전략을 제안하여, background의 특징을 유지하면서 anomaly latent를 초기화할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 방법론은 다양한 실험을 통해 신뢰할 수 있는 defect 생성을 가능하게 하며, 데이터 부족 문제를 효과적으로 완화하는 데 기여합니다. 또한, synthetic data가 실제 사례와 유사하며, 구조적 및 논리적 anomaly 탐지에 대한 모델의 성능을 개선하는 데 유익함을 입증합니다.



### Is 'Right' Right? Enhancing Object Orientation Understanding in Multimodal Language Models through Egocentric Instruction Tuning (https://arxiv.org/abs/2411.16761)
- **What's New**: 최근 연구에서는 멀티모달 대형 언어 모델(MLLMs)의 객체 방향 이해에 대한 한계를 다루고 있습니다. 본 논문에서는 사용자 관점에 맞춘 egocentric instruction tuning 방법을 제안하여 MLLMs의 방향 이해 능력을 향상시키고자 합니다. 이는 일관된 주석 표준에 따라 객체 방향 주석을 매치함으로써 이루어지며, 다양한 도메인에서 수집된 이미지를 사용하는 benchmark인 EgoOrientBench도 함께 제시합니다.

- **Technical Details**: 본 연구는 MLLMs의 훈련 데이터를 위한 표준화된 주석 가이드를 마련하기 위해, 이미지를 기반으로 한 egocentric 관점에서 객체 방향에 대한 주석을 수동으로 생성합니다. 이를 통해 생성된 egocentric instruction data는 MLLMs의 이미지 세부 사항을 인식하는 능력을 활용하여 객체 방향 이해를 도울 수 있도록 설계되었습니다. 이러한 방법론은 MLLMs의 일반적인 성능을 저하시키지 않고도 사용자 의도에 맞추어 객체 방향 이해를 향상시킵니다.

- **Performance Highlights**: EgoOrientBench를 통해 실시된 실험 결과는 egocentric instruction tuning이 사용자 관점에 따라 객체 방향 이해를 크게 향상시킨다는 것을 보여줍니다. 이 벤치마크는 다양한 복잡성을 가진 세 가지 작업을 포함하여 MLLMs의 객체 방향 이해를 포괄적으로 평가할 수 있도록 설계되었습니다. 조사된 결과는 실제 환경에서 MLLMs의 적용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions (https://arxiv.org/abs/2411.16760)
- **What's New**: 이번 논문에서는 Transformer의 gradient 기반 해석에서 발생하는 문제점을 다루고 있으며, 이를 개선할 수 있는 방법을 제시합니다. 연구팀은 Transformer에서 발생하는 gradient 흐름의 불균형을 발견하고, 이를 해결하기 위한 새로운 방법인 LibraGrad를 소개합니다. LibraGrad는 forward pass를 변경하지 않으면서도 backward 경로의 가지치기(pruning)와 스케일링(scaling)을 통해 gradient 불균형을 수정하는 이론적인 접근 방식을 기반으로 합니다.

- **Technical Details**: LibraGrad는 세 가지 메트릭 계열을 통해 평가됩니다: Faithfulness는 가장 관련성이 높은 및 낮은 특성의 변화에 따른 예측 변화를 측정하며, Completeness Error는 모델 출력에 대한 설명 보존을 측정합니다. Segmentation AP는 인간의 인식과의 정합성을 평가합니다. 실험은 8개의 아키텍처, 4개의 모델 크기, 4개의 데이터셋을 사용하여 LibraGrad의 효과성을 확인하였고, 모든 메트릭에서 기존 화이트박스 방법보다 우수한 성능을 보였습니다.

- **Performance Highlights**: LibraGrad는 CLIP 모델에서의 텍스트 기반 영역 강조 정확성과 ImageNet 세분화 모델에서의 동물 간의 클래스 구별 정확성을 통한 뛰어난 질적 결과를 보여줍니다. 기존의 방법들이 자주 어려움을 겪는 이 두 가지 설정에서 특히 효과적입니다. 또한, attention-free MLP-Mixer 아키텍처에서도 효과를 보임으로써 현대적인 아키텍처로의 확장 가능성을 나타냅니다.



### Bundle Adjusted Gaussian Avatars Deblurring (https://arxiv.org/abs/2411.16758)
Comments:
          Codes and Data: this https URL

- **What's New**: 이 연구는 흐릿한 비디오에서 선명한 3D 인간 아바타를 복원할 수 있는 최초의 방법을 제시합니다. 기존 방법들은 일반적으로 고품질의 선명한 이미지를 필요로 하지만, 이 접근법은 흐릿한 프레임에서도 작동합니다. 우리는 인간의 움직임에 의한 블러 형성 모델과 3D 인간 모션 모델을 결합하여 모션 블러에 의해 발생하는 애매함을 명확히 합니다.

- **Technical Details**: 우리의 방법은 3D-aware 블러 형성 모델에 기반하여 서브 프레임 모션 표현을 최적화하고 선명한 3DGS 아바타 모델을 구축하는 두 가지 과제로 블러 제거 문제를 분해합니다. SMPL 프레임워크를 기반으로 한 3D-aware 인간 모션 모델을 통해 각 흐릿한 프레임에 대한 그럴듯한 서브 프레임 모션을 동시에 복구하고 선명한 3DGS 아바타 모델을 재구성합니다. 이 방식은 모션 블러를 동반한 흐릿한 비디오 프레임으로부터 효과적으로 아바타 모델을 학습할 수 있게 합니다.

- **Performance Highlights**: 제안한 방법은 ZJU-MoCap 데이터셋에서 유도된 합성 데이터셋과 360도 하이브리드 노출 카메라 시스템을 통해 수집된 실제 데이터셋을 사용하여 기준 성능을 초과하는 성능을 달성합니다. 우리의 평가 결과, 제안한 방법이 기존 방법들과 비교했을 때 모션 블러 제거 및 고품질 3D 인간 아바타 재구성에서 상당한 향상을 시킴을 확인했습니다. 공개된 코드는 향후 연구에 기여할 수 있는 자료로 제공됩니다.



### Visual Counter Turing Test (VCT^2): Discovering the Challenges for AI-Generated Image Detection and Introducing Visual AI Index (V_AI) (https://arxiv.org/abs/2411.16754)
Comments:
          13 pages, 9 figures

- **What's New**: AI 이미지 생성 기술의 확산과 접근성 증대가 잘못된 정보 유포에 대한 우려를 낳고 있습니다. 본 논문에서는 기존의 AI 생성 이미지 탐지(AGID) 기술이 현재의 AI 생성 이미지를 효과적으로 탐지하는 데 inadequacy (불충분) 하다고 주장하며, 새로운 평가 기준인 비주얼 카운터 튜링 테스트(Visual Counter Turing Test, VCT^2)를 제안합니다. 이 VCT^2는 130K 개의 현대 텍스트-이미지 모델에 의해 생성된 이미지를 포함하고 있습니다.

- **Technical Details**: VCT^2 데이터셋은 뉴욕 타임즈의 트위터 계정에서 얻은 두 세트의 프롬프트와 MS COCO 데이터셋의 캡션을 포함하여 구성됩니다. 또한, 본 연구는 기존의 AGID 기술이 VCT^2 기준에서 AI 생성 이미지를 탐지하는 데 있어 ineffectiveness (효과 없음)을 드러내며, 이로 인해 새로운 평가 프레임워크가 필요함을 강조합니다. 우리는 텍스처 복잡성 및 객체 일관성과 같은 다양한 시각적 관점에서 생성된 이미지를 평가하는 비주얼 AI 지수(Visual AI Index, V_AI)를 제안하고 있습니다.

- **Performance Highlights**: 현재 AI 이미지 생성 모델들이 계속 발전함에 따라, 이러한 모델들을 평가하기 위한 양적(quantifiable) 프레임워크의 필요성이 더욱 강조되고 있습니다. 우리가 제안하는 V_AI는 이미지 생성 AI 모델에 대한 새로운 평가 기준을 설정하는 데 기여할 것입니다. 이 연구뿐만 아니라, 해당 데이터셋을 공개하여 이 분야의 연구를 촉진하고자 합니다.



### Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy (https://arxiv.org/abs/2411.16752)
- **What's New**: 이번 연구에서는 Zero-shot Composed Image Retrieval (ZSCIR) 방식의 새로운 접근법인 Imagined Proxy for CIR (IP-CIR)를 소개합니다. IP-CIR는 훈련 없이도 쿼리 이미지와 관련된 설명에 맞는 프로시(proxy) 이미지를 생성하게 되어 성능을 개선하는 데 초점을 맞추고 있습니다. 특히, LLM(대형 언어 모델)의 일반화 능력을 활용하여 이미지 레이아웃을 생성하고, 이를 통해 보다 풍부한 쿼리 표현을 가능하게 합니다.

- **Technical Details**: 연구에서 제안된 IP-CIR 방식은 LLM을 활용하여 생성된 프로시 이미지를 기존 쿼리 이미지 및 설명과 결합하여 검색 성능을 향상시킵니다. 특히, 프로시 이미지를 이용하여 이미지의 스타일, instance 속성 및 공간 관계와 같은 추가 정보를 제공하며, 이는 복잡한 캡션의 텍스트 기반 검색에서는 놓치기 쉬운 부분입니다. 이를 위해, 텍스트 기반의 유사성과 프로시 기반 유사성을 통합하는 새로운 균형 메트릭을 도입하여 보다 정확한 검색이 이루어질 수 있도록 합니다.

- **Performance Highlights**: 세 가지 공개 데이터셋(CIRR, CIRCO, FashionIQ)을 기반으로 실험을 진행한 결과, IP-CIR 방법이 검색 성능을 극적으로 개선한 것을 확인할 수 있었습니다. 특히 CIRR 데이터셋에서는 Recall@K에서 70.07을 달성하며 SOTA(State-of-the-Art) 결과를 기록하였고, FashionIQ 데이터셋에서도 Recall@10 수치가 45.11에서 45.74로 증가했습니다. 이러한 결과는 IP-CIR가 기존 방법들과 결합하여 의미 있는 검색 성능 향상을 가져올 수 있음을 입증합니다.



### PriorDiffusion: Leverage Language Prior in Diffusion Models for Monocular Depth Estimation (https://arxiv.org/abs/2411.16750)
- **What's New**: 이번 연구는 텍스트-이미지 확산 모델을 활용하여 단안(depth estimation)에서 발생하는 모호성(ambiguity)과 시각적 혼란(visual nuisance)을 해결할 수 있는 가능성을 탐구하고 있습니다. 특히, 기존의 단안 깊이 추정은 입체(depth cues)나 다중 시점(multi-view) 깊이 안내의 부재로 인해 본질적인 모호성에 시달리고 있습니다. 연구진은 확산 모델에서 학습된 언어 선험(prior)을 통해 기하학적 정보를 확보함으로써 이러한 모호성을 해소할 수 있다고 주장합니다.

- **Technical Details**: 연구에서는 PriorDiffusion이라는 새로운 접근법을 제안하여, 텍스트와 이미지를 조합하여 심층 추론을 수행하는 방식을 채택하였습니다. 구체적으로, 선행 학습된 텍스트-이미지 확산 모델을 사용하여 장면에 맞춘 이미지와 텍스트 설명을 입력으로 받아들이고, 이를 통해 깊이 추정을 위한 노이즈 제거(denoising)를 진행합니다. 이러한 방법은 3D 장면을 사용자 의도에 맞춰 인식할 수 있도록 모델의 주의를 특정 영역에 집중시키는 방식으로 작동합니다.

- **Performance Highlights**: HyperSim 및 Virtual KITTI 데이터셋으로 훈련한 결과, NYUv2, KITTI, ETH3D 및 ScanNet 데이터셋에서 다른 확산 기반 깊이 추정기들과 비교하여 최신 기술 수준의 성과를 달성하였습니다. 연구진은 실제 데이터셋에서도 우수한 정성적 및 정량적 결과를 도출하며, 빠른 수렴 속도를 보여줍니다. 이는 언어 선험을 활용한 접근 방식을 통해 단안 깊이 추정 모델의 성능이 크게 향상될 수 있음을 보여줍니다.



### AnySynth: Harnessing the Power of Image Synthetic Data Generation for Generalized Vision-Language Tasks (https://arxiv.org/abs/2411.16749)
- **What's New**: 이번 논문에서는 AnySynth라는 통합 합성 데이터 생성 프레임워크를 제안합니다. 이 프레임워크는 다차원적인 요구 사항에 따라 임의의 유형의 합성 데이터를 생성할 수 있는 적응 가능하고 종합적인 구성 요소를 통합합니다. 특히, 사용자 특정 참조 이미지와 스타일 이미지를 포함할 수 있어 다양한 작업 요구 사항에 유연하게 대응할 수 있습니다.

- **Technical Details**: AnySynth는 Layout-Image-Annotation 구조를 채택하여 단계별로 데이터를 합성합니다. Task-Specific Layout Generation Module을 통해 대규모 언어 모델(LLM)을 활용하여 다양한 조건에 맞는 적절한 레이아웃을 생성합니다. 이후 Uni-Controlled Image Generation Module에서는生成된 레이아웃을 기반으로 사용자가 특정한 조건을 추가할 수 있어 이미지를 생성합니다.

- **Performance Highlights**: AnySynth의 성능은 여러 작업에서 검증되었습니다. Few-shot Object Detection에서는 VOC 데이터셋에서 클래스의 AP를 크게 향상시키고, Zero-Shot Composed Image Retrieval에서도 효율성을 입증했습니다. 다양한 실험을 통해 제안된 프레임워크의 효과성과 일반성을 강조하며, 각 모듈의 성능이 검증되었습니다.



### LetsTalk: Latent Diffusion Transformer for Talking Video Synthesis (https://arxiv.org/abs/2411.16748)
Comments:
          17 pages, 14 figures

- **What's New**: 본 논문에서는 LetsTalk라는 새로운 프레임워크를 소개합니다. LetsTalk는 Latent Diffusion Transformer를 기반으로 하여 멀티모달 비디오 생성을 위한 혁신적인 접근 방식을 제공합니다. 모듈형 공간-시간 주의 메커니즘을 통해 여러 형식을 융합하고, 이미지를 통한 비디오 생성의 일관성을 향상시킵니다. 이는 오디오 기반의 포트레이트 애니메이션 생성을 위한 최초의 작업으로, 사실성과 표현력에서의 진전을 이끌어냅니다.

- **Technical Details**: LetsTalk는 변분 오토인코더(Variational Autoencoder, VAE)를 사용해 저해상도 잠재 공간 내에서 생성을 수행합니다. 제안된 프레임워크는 공간적 및 시간적 주의 모듈을 통한 내부 프레임과 교차 프레임 일관성을 포착합니다. 이 과정에서 세 가지 융합 스킴(Direct Fusion, Siamese Fusion, Symbiotic Fusion)을 정리하여 적용 방법을 탐구합니다. 최적의 비디오 생성을 위한 융합 방식을 정의하고, 이미지와 오디오의 차이를 고려한 솔루션을 구현합니다.

- **Performance Highlights**: 실험 결과, LetsTalk는 시간적으로 일관되고 현실감 있는 비디오를 생성하는 데 성공했습니다. 특히, 다양한 애니메이션 요소(예: 깜박임, 머리 움직임, 표정)의 생동감을 더하여 다채로운 생성 결과를 보여줍니다. 다른 접근 방식과 비교했을 때, 우리는 포트레이트 애니메이션 주도에 있어 LetsTalk가 우수성을 보여줌을 입증했습니다. 이로 인해 사실적이고 고충실도의 변환 기반 애니메이션 생성이 가능하다는 것을 확인하였습니다.



### FollowGen: A Scaled Noise Conditional Diffusion Model for Car-Following Trajectory Prediction (https://arxiv.org/abs/2411.16747)
Comments:
          arXiv admin note: text overlap with arXiv:2406.11941

- **What's New**: 이 논문에서는 차량 추적 예측을 위해 새로운 방법인 FollowGen을 제안합니다. FollowGen은 자동차 후행 시나리오를 위한 생성적 모델로, 인접한 차량 간의 상호작용을 정교하게 통합하여 경로 예측의 정확성과 그럴듯함을 향상시키는 데 중점을 두고 있습니다. 특히, 본 연구에서는 노이즈 스케일링 전략과 크로스 어텐션 기반 트랜스포머 아키텍처를 활용하여 차량 간의 복잡한 의존성을 모델링합니다.

- **Technical Details**: FollowGen은 차량의 역사적 동역학을 포착하기 위해 GRU, 차량 위치 기반 어텐션, 푸리에 임베딩으로 구성된 시간 특성 인코딩 파이프라인을 개발합니다. 또한, 확산 과정에서 등방성 가우시안 노이즈를 차량의 역사적 이동 특성에 맞게 조정하는 노이즈 스케일링 전략을 제안합니다. 이러한 접근 방식은 차량 간의 상호작용을 더 정교하게 모델링할 수 있게 하여, 동적 교통 상황에서 차량의 의존성을 효과적으로 반영하였습니다.

- **Performance Highlights**: 실제 다양한 주행 시나리오에서 FollowGen의 성능을 실험한 결과, HV가 HV를 따르거나 AV가 HV를 따르는 경우의 경로 예측에서 뛰어난 정확도를 보였습니다. 또한, 다양한 환경에서의 견고성과 일반성을 검증하여, 기존의 예측 방법들에 비해 뛰어난 예측 성능을 입증했습니다. 이로써 FollowGen은 자율 주행 및 고급 운전 보조 시스템에 매우 중요한 모델로 자리매김할 잠재력을 가집니다.



### Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents (https://arxiv.org/abs/2411.16740)
- **What's New**: 이 논문은 Large Multimodal Models (LMMs) 를 위한 새로운 벤치마크, DocHaystack와 InfoHaystack 을 소개합니다. 이 벤치마크는 1,000개의 문서에 대해 질문을 제기함으로써, 복잡한 이미지 검색과 이해 능력을 평가하는 것을 목표로 합니다. LMMs 는 실제 상황에서 필요한 대규모 이미지 도큐먼트 검색의 효율성을 높일 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: V-RAG라는 새로운 비전 중심의 Retrieval-Augmented Generation (RAG) 프레임워크를 제안했습니다. V-RAG는 여러 가지 멀티모달 비전 인코더를 융합하여 각 인코더의 특성을 최적화하여 검색의 정확성을 높입니다. 또한 질문-문서 관련성 평가 모듈을 통해 검색 프로세스를 개선하여 더 관련성이 높은 문서만 우선적으로 고려하도록 설계되었습니다.

- **Performance Highlights**: V-RAG는 DocHaystack-1000 및 InfoHaystack-1000 벤치마크에서 이전 최고의 모델 대비 각각 9% 및 11%의 Recall@1 향상을 나타냅니다. 또한 V-RAG와 LMMs 를 통합하면 DocHaystack-200에서 55% 이상의 정확도 향상과 InfoHaystack-200에서 34%의 개선 효과를 가져오는 것으로 나타났습니다.



### Gradient-Guided Parameter Mask for Multi-Scenario Image Restoration Under Adverse Weather (https://arxiv.org/abs/2411.16739)
- **What's New**: 이 논문에서는 다양한 악천후 조건에서의 이미지 복원 문제를 해결하기 위해 새로운 Gradient-Guided Parameter Mask 를 제안합니다. 기존의 기법들이 추가의 파라미터를 사용하여 복잡성을 증가시킨 반면, 본 연구의 방법은 추가 파라미터 없이도 효과적으로 이미지 손상을 완화할 수 있습니다. 이 방법은 각 날씨 조건에 대한 그래디언트 변동 강도를 평가하여 모델 파라미터를 공통 및 특정 구성 요소로 나누어 효율성과 효과성을 강화합니다.

- **Technical Details**: Gradient-Guided Parameter Mask 방법은 훈련 동안 공통 파라미터로 인한 그래디언트 변화에 따라 모델 파라미터를 분류합니다. 이를 통해 모델은 각 날씨 조건에 대한 관련 특징을 정교하게 학습할 수 있으며, 각 시나리오에서의 파라미터 업데이트가 다른 조건들의 간섭 없이 이루어지는 것을 보장합니다. 이번 연구는 알고리즘의 경량성을 유지하면서 실시간 응용에 적합하도록 만듭니다.

- **Performance Highlights**: 제안된 방법은 Raindrop, Rain, Snow100K 데이터셋에서 각각 PSNR 점수 29.22, 30.76, 29.56을 기록하며, 여러 벤치마크 데이터셋에서 최첨단 성능을 입증했습니다. 효율적인 그래디언트 가이딩 파라미터 마스크로 다양한 날씨 시나리오에서 발생할 수 있는 간섭을 완화하여 성능을 향상시킵니다. 또한 실시간 애플리케이션에 적합하게 설계된 모델은 컴퓨팅 효율성이 중요한 자율 주행과 같은 분야에서 활용될 수 있습니다.



### Classifier-Free Guidance inside the Attraction Basin May Cause Memorization (https://arxiv.org/abs/2411.16738)
- **What's New**: 이 논문은 확산 모델 (Diffusion Models)에서의 암기 현상(memorization phenomenon)을 이해하고 이를 완화시키기 위한 새로운 접근법을 제시합니다. 기존의 방식들의 한계를 극복하기 위해, 저자들은 '매력 분지 (attraction basin)' 개념을 도입하여 암기 현상을 설명하고 이를 피하기 위한 단순하고 효과적인 방법을 제안합니다. 새로운 안내 기법인 '반대 안내 (Opposite Guidance)'를 통해 생성된 이미지는 질이 높은 비기억화된(non-memorized) 이미지를 생성할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 확산 모델의 정상적인 동역학적(dynamic systems) 행동을 통해 암기 현상의 새로운 통찰력을 제공합니다. 특히, 제공된 조건(condition) 아래에서 '매력 분지'가 형성되고, 이는 노이즈 예측(noise prediction)에서 특이한 패턴을 생성하여 결과적으로 특정 이미지를 암기하는 현상으로 이어질 수 있습니다. 저자들은 입증된 접근법을 사용하여 매력 분지를 피하는 방법을 제시하며, 이는 추가적인 계산 비용 없이 진행할 수 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 접근법은 다양한 시나리오에서의 메모리 문제를 성공적으로 완화할 수 있음을 보였습니다. 특히 '반대 안내' 기술이 효과적으로 작동하여 매력 분지 밖으로 궤적을 밀어낼 수 있음을 확인했습니다. 저자들은 이전 방식들이 제한된 조건에서만 효과가 있음을 지적하며, 제안된 방법들이 더 넓은 범위에서 적용될 수 있는 방법이라고 강조합니다.



### Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method (https://arxiv.org/abs/2411.16733)
- **What's New**: 최근 도로 그래프 추출(Road Graph Extraction)이 자율 주행과 내비게이션 같은 분야에서 중요해지면서 큰 주목을 받고 있습니다. 본 연구에서는 약 20배 더 큰 글로벌 스케일(Global-Scale) 도로 그래프 추출 데이터셋을 수집하였으며, 이 데이터셋은 13,800 제곱킬로미터를 커버합니다. 또한, SAM-Road++라는 새로운 도로 그래프 추출 모델을 개발하여 훈련과 추론 간의 불일치 문제를 해결할 수 있는 방법을 제시합니다.

- **Technical Details**: SAM-Road++ 모델에서는 'node-guided resampling' 기법을 사용하여 분류기 훈련 시 실제 레이블 노드를 대신하여 확률이 가장 높은 예측 노드를 재샘플링합니다. 이 접근법은 훈련과 추론 과정에서 더 일관성을 줄 수 있게 도와줍니다. 또한, 도로에서의 가림 현상(occlusion) 문제를 해결하기 위해 새로운 'extended-line' 전략을 도입하여 노드 간의 연관성을 활용하고 있습니다.

- **Performance Highlights**: 모델의 성능은 이전에 본적이 없는 지역에서의 예측력에서 특히 뛰어난 것으로 입증되었습니다. 글로벌 스케일 데이터셋과 제안된 SAM-Road++ 방법의 유효성을 검증하는 광범위한 실험이 수행되었으며, 결과적으로 도로 추출 작업을 위한 커뮤니티의 종합적인 평가를 가능하게 합니다.



### An Information-Theoretic Regularizer for Lossy Neural Image Compression (https://arxiv.org/abs/2411.16727)
Comments:
          12 pages, 8 figures

- **What's New**: 본 논문에서는 손실 이미지 압축(networks for lossy image compression)의 최적화를 위한 새로운 방법론이 제안되었습니다. 저자들은 잠재적(entropy) 표현의 최소화가 조건부(source entropy) 엔트로피의 최대화와 동등하다는 정보를 이론적으로 뒷받침하는 발견을 밝혀냈습니다. 이를 통해, 새로운 구조적 정규화 방법을 제안하여 최적화의 효율성과 일반화 능력을 동시에 향상시키고자 했습니다.

- **Technical Details**: 제안된 구조적 정규화 방법은 네거티브 조건부 엔트로피를 훈련 목표에 통합하여, neural image compression 작업에서의 성능을 개선하는 데 초점을 맞추고 있습니다. 이 정규화 기법은 해석 가능하고 플러그 앤 플레이 방식으로 구현되며, 추론 오버헤드가 없습니다. 또한 실험을 통해 여러 압축 구조 및 새로운 도메인에서도 효과를 확인하였습니다.

- **Performance Highlights**: 대규모 실험 결과, 제안된 방법이 모델을 정규화하는 데 우수한 성능을 발휘하고, 잠재 표현(latent representation)에서 비트를 추가로 압축할 수 있는 능력을 갖추고 있음을 보여주었습니다. 이 연구는 손실 이미지 압축 분야의 경계를 지속적으로 확장하는데 기여하고 있습니다.



### EmotiveTalk: Expressive Talking Head Generation through Audio Information Decoupling and Emotional Video Diffusion (https://arxiv.org/abs/2411.16726)
Comments:
          19pages, 16figures

- **What's New**: Diffusion 모델은 Talking Head 생성 분야에 혁신을 가져왔지만, 여전히 표현력(expressiveness), 제어 가능성(controllability), 장시간 생성 시 안정성에 대한 문제에 직면해 있습니다. 본 연구에서는 이러한 문제를 해결하기 위해 EmotiveTalk 프레임워크를 제안합니다. 이 프레임워크는 더 나은 입술 움직임(lip movement)과 얼굴 표정(facial expression) 생성을 위한 제어를 실현하기 위해 설계되었습니다.

- **Technical Details**: EmotiveTalk의 핵심은 Vision-guided Audio Information Decoupling(V-AID) 접근 방식을 통해 오디오 기반의 분리된 표현(representations)을 생성하는 것입니다. 이를 통해 오디오와 얼굴 표현 표현 공간 간의 정렬을 달성하며, Multi-source Emotion Condition 제약 하에 표현 관련 representation을 생성하기 위한 Diffusion-based Co-speech Temporal Expansion(Di-CTE) 모듈을 제공합니다. 또한, Emotion Talking Head Diffusion(ETHD) 백본(backbone)은 Expression Decoupling Injection(EDI) 모듈을 통합하여 참조 초상화(reference portraits)에서 표현을 자동으로 분리하고 목표 표현(target expression) 정보를 통합하여 더 표현력이 풍부한 결과를 제공합니다.

- **Performance Highlights**: 실험 결과 EmotiveTalk는 감정의 제어 가능성을 보장하며 장시간 생성 중에도 안정성을 유지하여 매우 표현력 있는 Talking Head 비디오를 생성할 수 있음을 보여줍니다. 기존 방법들과 비교할 때 EmotiveTalk는 최첨단(state-of-the-art) 성능을 달성하였습니다. 이는 사용자가 원하는 감정 표현을 보다 효율적으로 적용한 결과로, Talking Head 생성을 위한 새로운 기준을 설정합니다.



### $\textit{Revelio}$: Interpreting and leveraging semantic information in diffusion models (https://arxiv.org/abs/2411.16725)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문은 다양한 diffusion architecture의 여러 레이어와 denoising timesteps 내에서 시각적 의미 정보가 어떻게 표현되는지를 연구합니다. 특별히, k-sparse autoencoders(k-SAE)를 이용하여 해석 가능한 특성 모음을 발견하였으며, 이는 전이 학습(transfer learning)을 통해 확인되었습니다. 연구에서는 4개의 데이터셋을 통해 diffusion 특징의 효과iveness를 보여줍니다.

- **Technical Details**: 연구에서는 diffusion 모델의 다양한 레이어와 timesteps 내에서 포착된 시각적 정보의 유형과 이들이 상호작용하는 방식, 외부 조건에서의 레이어 간의 차별적 이점 등을 질문합니다. 이를 위해 sparsely encoded visual pattern의 원리를 바탕으로 k-sparse autoencoders(k-SAE)를 적용하여 해석 가능한 시각적 특성을 발굴하고, 이를 기반으로 경량 분류기(Diff-C)를 개발하였습니다.

- **Performance Highlights**: Diff-C는 이전의 모든 연구에서 diffusion 특징을 활용한 representation learning에 대해 최고의 성능을 보이며, 강력한 self-supervised visual 및 multi-modal 기준과도 경쟁적으로 성과를 냈습니다. 실증적으로 각 데이터셋과 과제에서 granularity(세분화) 및 일반화 가능성이 다르게 나타나며, diffusion architecture, pre-training 데이터, 그리고 attention 메커니즘이 전반적인 성과에 영향을 미친다는 것을 입증합니다.



### Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens (https://arxiv.org/abs/2411.16724)
- **What's New**: 본 논문에서는 대형 비전-언어 모델(LVLMs)에서의 환각(hallucination) 문제를 시각 정보 처리 관점에서 다루고 있습니다. 기존 연구들은 주로 언어적 요인에 집중했으나, 이 논문은 LVLMs가 시각 정보를 처리하는 과정을 분석하여 환각이 발생하는 원인을 규명하고자 합니다. 특히, 중간 레이어에서의 시각 정보의 강화(visual information enrichment) 및 의미적 세분화(semantic refinement) 단계가 환각과 밀접한 관계가 있음을 밝혔습니다.

- **Technical Details**: LVLMs는 입력된 이미지에 대해 두 가지 중요한 단계로 시각 정보를 처리합니다: 첫째, '시각 정보 강화' 단계에서 이미지 데이터가 객체 토큰으로 전파되고, 둘째 '의미적 세분화' 단계에서 시각 정보를 텍스트로 해석합니다. 이를 위해 'Visual Attention Ratio (VAR)'라는 점수를 도입해 중간 레이어에서의 시각 주의 분포를 분석하였습니다. 이 연구는 Multi-Head Attention (MHSA) 메커니즘을 활용해 서로 다른 주의 헤드 간의 상호작용이 환각 표시로 이어지는 과정을 시각화하여 조사했습니다.

- **Performance Highlights**: 제안된 방법은 원래 모델에 비해 CHAIRI와 CHAIRS에서 각각 최대 6.3과 24.1 포인트의 환각 완화를 달성하였습니다. 실험을 통해 다양한 주류 LVLMs에서 환각을 효과적으로 줄이는 동시에 설명의 세부 사항을 유지할 수 있음을 입증했습니다. 이러한 결과는 LVLMs의 신뢰성을 높이기 위한 중요한 방향성을 제시합니다.



### Active Prompt Learning with Vision-Language Model Priors (https://arxiv.org/abs/2411.16722)
- **What's New**: 이 논문에서는 Vision-language models (VLMs)의 기존 한계를 극복하기 위한 새로운 프레임워크를 소개합니다. 전통적으로 VLMs는 각 작업에 대해 수동으로 생성된 텍스트 프롬프트에 의존해왔으나, 이는 새로운 작업에 대한 적응성을 저해합니다. 저자들은 예산 효율적인 액티브 프롬프트 학습(active prompt learning) 접근 방식을 통해, 보다 나은 데이터 선택 전략의 필요성을 강조합니다.

- **Technical Details**: 제안된 방법은 클래스를 기반으로 한 클러스터링 클래스-가이드를 활용하여 VLM의 사전 훈련된 이미지 및 텍스트 인코더를 사용합니다. 이로 인해 초기 액티브 학습 단계에서 클러스터 균형된 획득 함수(cluster-balanced acquisition function)가 가능합니다. 또한, 클래스별로 큰 신뢰도 변동을 고려하여, 적응형 클래스별 임계값에 기반한 예산 절약형 선택적 쿼리(selective querying)도 제안합니다.

- **Performance Highlights**: 여러 실험 결과, 본 방법은 아홉 개의 데이터 세트에서 기존 베이스라인을 초과하는 성능을 보였습니다. 이러한 결과는 보다 적은 레이블이 있는 데이터로도 높은 정확도를 달성하게 해주는 데이터 선택 전략의 중요성을 입증합니다. 정교한 데이터 선택을 통해 VLMs의 효율적인 활용이 가능함을 시사합니다.



### Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks (https://arxiv.org/abs/2411.16721)
- **What's New**: 본 논문은 ASTRA라는 새로운 방어 방법을 제안합니다. ASTRA는 Vision Language Models (VLMs)에서 발생할 수 있는 의도하지 않은 해로운 출력을 방지하기 위해 아드버시리얼 피쳐 방향에서 모델을 적응적으로 조정하는 매우 효율적인 방어 기법입니다. 기존 방어 방법들이 높은 계산 비용으로 인해 실제에 적용하기 어려웠으나, ASTRA는 이러한 문제를 해결하기 위해 디자인되었습니다.

- **Technical Details**: ASTRA의 주요 절차는 해로운 응답을 나타내는 전이 가능한 조정 벡터를 찾고, 추론 시 이러한 방향을 제거하기 위해 적응형 활성화 조정을 적용하는 것입니다. 해로운 응답과 가장 강한 관련이 있는 시각적 토큰을 무작위로 제거하여 조정 벡터를 생성하고, 이 벡터로 활성화를 조작하여 아드버시리얼 이미지에 대해서 강력한 효과를 발휘합니다. 이 과정은 단 한 번의 응답 생성을 통해 효율성을 유지합니다.

- **Performance Highlights**: ASTRA는 비감염 입력에서 성능 저하를 거의 없이 아드버시리얼 입력에 대해 해로운 출력을 강하게 회피하는 능력을 보여줍니다. 여러 모델 및 기준 대비 대규모 실험 결과, ASTRA는 perturbation-based 공격에 대해 우수한 방어 성능을 보이며, 기존 최첨단 방법인 JailGuard보다도 9배 더 빠른 성과를 기록하였습니다. 또한, ASTRA는 보지 못한 공격에도 잘 대응할 수 있는 전이성도 가지고 있습니다.



### Importance-based Token Merging for Diffusion Models (https://arxiv.org/abs/2411.16720)
- **What's New**: 이번 논문에서는 유사한 토큰을 병합하여 고품질 이미지 및 비디오 생성에서의 효율성을 높이는 방법을 제안했습니다. 기존의 토큰 병합 방식인 ToMeSD와는 달리, 이 방법은 분류기 프리 가이던스(classifier-free guidance) 기법을 활용하여 중요한 토큰을 유지함으로써 샘플 품질을 개선합니다. 이를 통해 조건부 확산 모델에서 효율적으로 적용될 수 있으며, 추가적인 계산 비용이 발생하지 않습니다.

- **Technical Details**: 본 연구에서는 각 토큰의 중요성을 분류기 프리 가이던스를 통해 평가하고, 이를 기반으로 목적지 토큰을 선택하는 새로운 토큰 선택 방법을 설계했습니다. 이 방법은 임의적인 선택 방식에서 벗어나 샘플 품질을 목표로 하여 토큰 풀을 구성하고, 이 풀에서만 목표 토큰을 선택합니다. 이 과정에서 토큰의 중요도를 분석하여 고정보 영역에 집중함으로써 이미지 품질을 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 텍스트-이미지 생성, 텍스트-비디오 생성, 다중뷰 이미지 생성 등 여러 분야에서 기준 방법들보다 월등한 성능을 보였습니다. 특히 높은 신뢰도를 바탕으로 샘플 품질을 크게 개선했으며, 모든 테스트된 시나리오에서 이미지 세부 표현에서 현저한 개선이 나타났습니다. 따라서 이 방법은 다양한 확산 모델 작업에 대해 최첨단 성능을 달성합니다.



### Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients (https://arxiv.org/abs/2411.16719)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구에서는 Learn2Synth라는 새로운 방법론이 소개됩니다. 이 방법론은 소량의 실제 레이블 데이터로부터 합성 파라미터를 학습하여, 합성 이미지를 생성할 수 있는 엔진을 조정합니다. 전통적인 방법들이 이미지와 레이블 맵을 정렬하는 것을 목표로 하는 한계가 있는 반면, 이 접근법은 실제 데이터에 최적의 정확성을 보장합니다.

- **Technical Details**: Learn2Synth는 두 단계로 진행됩니다. 첫째, 합성 네트워크를 고정하고 합성 데이터를 세그멘테이션 네트워크에 통과시킵니다. 둘째, 세그멘테이션 네트워크를 고정하고 실제 데이터를 통과시켜 합성 네트워크를 업데이트합니다. 이 과정에서 하이퍼 그래디언트를 활용하여 합성 네트워크의 개선을 유도합니다.

- **Performance Highlights**: 실험 결과, 합성 이미지와 실제 데이터에서 모두 효과적인 성능 향상이 확인되었습니다. 학습된 증강 파라미터는 네트워크 훈련의 최적 환경에 대한 통찰력을 제공합니다. 결과적으로 이러한 접근법이 데이터 부족 문제를 해결하는 데 중요한 역할을 함을 입증했습니다.



### Neuro-Symbolic Evaluation of Text-to-Video Models using Formalf Verification (https://arxiv.org/abs/2411.16718)
- **What's New**: 최근의 텍스트-비디오(T2V) 모델 발전으로 Sora, Gen-3, MovieGen, CogVideoX등이 합성 비디오 생성의 한계를 넓히고 있습니다. 이러한 모델의 사용이 로봇공학, 자율주행 및 엔터테인먼트 분야 등에서 증가하고 있습니다. 그러나 기존 평가 지표들은 비주얼 품질과 매끄러움에 집중하고, 안전이 중요한 애플리케이션에 필요한 시간적 충실도와 텍스트-비디오 일치를 간과하고 있습니다. 이를 해결하기 위해 텍스트-비디오 일치를 정밀 평가하는 NeuS-V라는 새로운 평가 지표를 제안합니다.

- **Technical Details**: NeuS-V는 신경-상징적 형식 검증 기술을 사용하여 텍스트-비디오 일치를 평가하는 혁신적인 메트릭입니다. 우리의 접근법은 먼저 프롬프트를 공식적으로 정의된 Temporal Logic (TL) 규격으로 변환하고, 생성된 비디오를 자동화 표현으로 변환합니다. 그런 다음 TL 규격에 대해 비디오 자동화를 공식적으로 점검하여 텍스트-비디오 정합성을 평가합니다. 더불어, Temporal fidelity를 평가하기 위해 장기간의 프롬프트 데이터 세트를 제공합니다.

- **Performance Highlights**: NeuS-V는 기존 메트릭에 비해 인간 평가와 5배 이상의 높은 상관관계를 보이며, 현재 비디오 생성 모델들이 복잡한 시간 관련 프롬프트에서 낮은 성능을 보여 개선의 필요성을 강조합니다. 이 평가 결과는 텍스트-비디오 생성 기능 향상을 위한 향후 연구의 중요성을 제시합니다. 또한 NeuS-V는 세계적 수준의 T2V 모델의 성능을 평가하기 위한 공개 벤치마크 자료를 곧 발표할 예정입니다.



### PaRCE: Probabilistic and Reconstruction-based Competency Estimation for CNN-based Image Classification (https://arxiv.org/abs/2411.16715)
Comments:
          arXiv admin note: text overlap with arXiv:2409.06111

- **What's New**: 본 논문에서는 모델 불확실성을 파악하는 기존의 방법과 비교하여, 새로운 확률적 재구성 기반 능력 추정 기법인 PaRCE(Probabilistic and Reconstruction-based Competency Estimation)를 제안하고 있습니다. PaRCE는 이미지에 대한 모델의 예측 정확성을 나타내는 단일 종합 점수를 제공하여 모델의 신뢰성을 높이는 데 기여합니다. 기존 연구들이 미비했던 예외적인 분포의 입력에 대하여도 더욱 정확한 평가를 가능하게 합니다.

- **Technical Details**: PaRCE 방법은 데이터, 모델 및 분포적 불확실성을 통합하여 CNN 기반 인식 모델의 예측 정확성을 반영합니다. 논문은 이 메소드를 사용하여 이상 탐지 및 지역 별 능력 추정 작업으로 확장할 수 있는 방법을 설명합니다. 각 클래스에 대한 예측 확률 추정에 softmax 함수를 사용하는 것 외에도, PaRCE는 재구성 손실을 활용하여 OOD 입력을 보다 효과적으로 탐지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, PaRCE는 잘 분류된 샘플, 잘못 분류된 샘플 및 이상 샘플 간의 차별화를 제공함이 입증되었습니다. 이 방법은 시각적 이미지 수정으로 인해 예측 정확도가 높은 샘플과 낮은 샘플 간의 구분에도 뛰어난 성능을 보입니다. 전체적으로 PaRCE는 다양한 이미지 유형에 걸쳐 더욱 신뢰성 있는 능력 추정을 제공하여 모델의 실용성을 증대시킵니다.



### TPIE: Topology-Preserved Image Editing With Text Instructions (https://arxiv.org/abs/2411.16714)
- **What's New**: 본 논문에서는 Topology-Preserved Image Editing with text instructions (TPIE)라는 새로운 방법을 소개합니다. 이 방법은 텍스트로 안내된 생성적 확산 모델을 통해 편집된 이미지의 토폴로지와 기하학 구조를 처음으로 보존합니다. 이 TPIE 프레임워크는 객체 변환의 잠재적 표현을 학습하는 자동인코더 기반 등록 네트워크와 사용자 정의 텍스트 지침에 따라 학습된 변환 피처의 데이터 분포를 효율적으로 캡처하는 새로운 조건부 기하학적 확산 모델로 구성됩니다.

- **Technical Details**: TPIE는 주어진 템플릿을 변형 가능한 변형으로서 새롭게 생성된 샘플을 취급합니다. 이는 사용자가 제시한 텍스트 입력에 조건부로 템플릿 이미지가 학습된 미분 가능 변환으로 변형되어 생성된다는 것을 의미합니다. 본 배경 섹션에서는 미분가능 변환을 통해 이미지 간 기하학적 변화의 표현을 최소화하는 방법에 대해 설명합니다. 여기에서는 쌍을 이루는 이미지를 통해 미분가능 변환을 이끌어내는 과정을 다룹니다.

- **Performance Highlights**: TPIE의 성능은 다양한 2D 및 3D 이미지를 사용하여 검증되었으며, 기존의 최첨단 이미지 편집 접근 방식과 비교되었습니다. 실험 결과, TPIE는 더 현실적인 이미지를 생성하면서도 토폴로지를 잘 보존하는 데 있어 다른 기준선들보다 우수한 성능을 보였습니다. 이 연구 결과는 민감한 도메인에서의 사실적인 해부학적 구조 유지를 위한 통제 가능한 편집 모델의 잠재력을 보여줍니다.



### Conditional Text-to-Image Generation with Reference Guidanc (https://arxiv.org/abs/2411.16713)
- **What's New**: 이번 논문에서는 Text-to-image diffusion 모델이 특정 주제를 더 정확히 렌더링하는 데 어려움을 겪는 문제를 해결하기 위해 참고 이미지를 추가적인 정보를 제공하는 조건으로 사용하는 RefDiffuser라는 접근 방식을 제안합니다. 이 모델은 Stable Diffusion (SD) 기반으로, 텍스트 프롬프트와 참고 이미지를 동시에 조건으로 하여 이미지를 생성합니다. 특히, VAE를 사용하여 참고 이미지를 동일한 잠재 공간에 인코딩하여 더 나은 이미지 생성을 목표로 합니다.

- **Technical Details**: RefDiffuser는 텍스트 프롬프트와 참고 이미지에 기반하여 이미지를 생성하는 구조로 되어 있습니다. 참고 이미지는 모델이 장면 텍스트 이미지 생성을 위해 필요한 문자의 형태와 같은 비선형적인 비주얼 레퍼런스를 제공하며, 이를 통해 기존 Diffusion 모델로는 정확히 생성하지 못하는 콘텐츠도 생성할 수 있습니다. 기본적으로 VAE 인코더를 통해 참고 이미지를 인코딩하고, 입력 레이어를 확장하여 참고 잠재 공간을 포함하는 수식으로 변경합니다.

- **Performance Highlights**: RefDiffuser는 적은 파라미터 수(28.55M)에도 불구하고 다수의 이미지 생성 작업에서 기존 방법보다 더 우수한 성능을 보여주었습니다. 영어 장면 텍스트, 다국어 장면 텍스트 및 로고 생성 작업에서 높은 정확도(각각 61.73%, 46.88%, 44.07%)를 달성하여, 언어 모델의 제약을 넘어서는 혁신적인 결과를 제공합니다. 이러한 성과는 전문가 텍스트-이미지 모델을 특정 주제에 맞춰 사용자 정의 할 수 있는 새로운 방향을 제시합니다.



### Visatronic: A Multimodal Decoder-Only Model for Speech Synthesis (https://arxiv.org/abs/2411.17690)
- **What's New**: 이번 논문에서는 새로운 작업인 비디오와 텍스트를 기반으로 한 음성 생성, 즉 Video-Text-to-Speech (VTTS)를 제안합니다. 이 작업은 기존의 자른 입 비디오에서 음성을 생성하는 작업보다 더 복잡하며, 다중 언어로도 활용될 수 있습니다. 저자들은 Visatronic이라는 단일 디코더 모델을 개발하여 텍스트, 비디오, 음성을 단일 공간에 통합하여 보다 나은 결과를 보여줍니다.

- **Technical Details**: Visatronic 모델은 transformer 모델의 공통 공간에 시각, 텍스트, 음성을 직접 임베드합니다. 이를 위해 자동 회귀 손실 (autoregressive loss)을 사용하여 멜 스펙트로그램을 생성하는 방법을 학습하게 됩니다. 이 모델은 lip-detection 기술이나 복잡한 구조 없이 비디오와 텍스트를 동시에 활용할 수 있는 간단한 접근법을 제공합니다.

- **Performance Highlights**: Visatronic 모델은 다양한 실험을 통해 양호한 성능을 입증하였습니다. VoxCeleb2 데이터셋에서 12.2%의 단어 오류율 (Word Error Rate, WER)을 기록했으며, 이는 기존의 기술들에 비해 향상된 결과입니다. 또한, VTTS에 대한 표준화된 평가 프로토콜을 제시하여 향후 연구에 기여할 계획을 가지고 있습니다.



### RoboPEPP: Vision-Based Robot Pose and Joint Angle Estimation through Embedding Predictive Pre-Training (https://arxiv.org/abs/2411.17662)
- **What's New**: 로보틱스 분야에서 기계 관절 각도를 파악하는 것이 중요한데, RoboPEPP라는 새로운 방법론이 소개되었습니다. 이 방법은 자가 지도 학습(self-supervised learning) 기반의 마스킹 방식으로 로봇의 물리적 모델을 통합하여 엔코더가 정보를 더 잘 이해하도록 돕습니다. 기존의 방법들이 로봇의 모습에 담긴 풍부한 정보를 충분히 활용하지 못하는 문제를 해결합니다.

- **Technical Details**: RoboPEPP는 로봇의 관절 부위를 마스크하며, 이를 통해 주변의 비마스크 영역에서 임베딩을 추출합니다. 이 과정을 통해 관절의 물리적 정보가 향상됩니다. 또한, 이 프레임워크는 입력을 무작위로 마스킹하고, 평가 시 핵심점을 필터링하여 더욱 견고성을 높입니다. 2D-3D 대응을 위해 PnP 알고리즘도 활용됩니다.

- **Performance Highlights**: RoboPEPP는 여러 데이터셋에서 테스트되었으며, 로봇의 포즈 및 관절 각도 추정에서 최고의 성능을 기록했습니다. 특히, 차폐(occlusions) 상황에서도 적은 민감도로 높은 정확도를 나타내며, 실행 시간도 최소화되었습니다. 이 모델은 로봇의 물리적 구조를 더 잘 이해하는 데 기여하고 있습니다.



### An Ensemble Approach for Brain Tumor Segmentation and Synthesis (https://arxiv.org/abs/2411.17617)
- **What's New**: 이 논문에서는 머신 러닝이 뇌의 자기 공명 영상(MRI)에 통합되어 진단 정확도를 향상시키고 이미지 분석을 가속화하며 데이터 기반 통찰력을 제공하는 방법을 제시합니다. 특히, 복잡한 데이터의 세밀한 내용을 포착하는 딥 러닝 모델의 활용을 통해 뇌 종양 분류, 분할 및 이미지 합성과 같은 다양한 작업에 적용할 수 있는 가능성을 보여주고 있습니다. U-Mamba와 같은 최신 모델들이 각각의 이미지를 정확하게 분할하는 데 있어 높은 성능을 입증한 바 있으며, 이 연구는 이러한 모델들을 앙상블하여 더 나은 성과를 보이는 프레임워크를 제안합니다.

- **Technical Details**: 논문에서 다루는 방법론은 2024 Brain Tumor Segmentation (BraTS) 챌린지에 참가하기 위해 다양한 모델 아키텍처를 통합하는 것입니다. 본 연구는 1.5T 자기장 강도의 MRI 스캔을 통해 비-강화 괴사 종양 중심(non-enhancing necrotic tumor core) 및 기타 종양 부위에 대한 주석을 포함한 두 개의 주요 데이터 세트를 사용합니다. 훈련 데이터는 U-Net 아키텍처를 기반으로 하여 Adam 옵티마이저와 함께 훈련되며, 다양한 하이퍼파라미터 및 데이터 증강 방법을 포함합니다.

- **Performance Highlights**: 모델의 성능은 Dice 점수 및 Hausdorff 거리(HD95)와 같은 지표를 사용해 평가됩니다. 특히, 각 병변에 대해 개별적으로 평가를 진행하여 각 병변을 비교하는 방식을 통해 모델의 객관성을 확보하고 있습니다. 추가적으로, 합성 이미지의 품질을 평가하기 위해 구조적 유사도 지수(SSIM), 신호 대 잡음 비율(PSNR), 평균 제곱 오차(MSE) 등의 메트릭을 사용하여 이미지의 신뢰성과 품질을 검증합니다.



### Uncertainty quantification for White Matter Hyperintensity segmentation detects silent failures and improves automated Fazekas quantification (https://arxiv.org/abs/2411.17571)
Comments:
          34 pages (or 22 not including appendix) 26 figures (or 11 not including appendix)

- **What's New**: 본 연구는 백질 고강도 여백(White Matter Hyperintensities, WMH)의 분할(segmentation) 과제에서 불확실성 정량화(uncertainty quantification, UQ) 기법을 적용하여 향상된 성능을 보여줍니다. 스토캐스틱 세그멘테이션 네트워크(Stochastic Segmentation Networks)와 딥 앙상블(Deep Ensembles)의 조합이 도메인 내 및 도메인 밖 데이터 모두에서 최고의 Dice 점수를 기록했습니다. 또한 UQ 정보를 활용하여 Fazekas 점수를 예측하는 새로운 방법을 제안하여, WMH 분할 결과와 UQ 맵으로부터 추출한 공간(feature) 정보를 통합했습니다.

- **Technical Details**: 연구에서 제안된 방법은 다중 테스트 데이터를 기반으로 하여 WMH segmentation을 위한 불확실성 정량화(UQ) 기법을 적용합니다. UQ는 이전의 모델을 개선하여 분할의 정확성과 함께 신뢰도 검사를 실시합니다. 다양한 UQ 기법(Softmax entropy, MonteCarlo Dropout 등)을 비교하여 실험하였으며, CNN 기반의 기술들이 어떻게 WMH와 같은 복잡한 패턴을 탐지하는지 증명했습니다.

- **Performance Highlights**: Fazekas 점수 분류에서 UQ와 공간 WMH feature를 결합했을 때의 분류 정확도는 각각 0.71, 0.66, 0.60을 기록하였고, 제시된 모델에서 광범위한 WMH를 효과적으로 탐지할 수 있었습니다. 연구에 등장한 UQ 기법은 또한 모델이 신뢰할 수 없는 분할을 어디서 할 수 있는지를 나타내어 개선된 성능을 보여줍니다. 이와 함께, UQ 기법이 WMH와 다른 병변(stroke lesions) 사이의 모호한 영역을 강조하고, 모델에 의해 분할되지 않은 작은 WMH 클러스터를 식별함을 보여주었습니다.



### Natural Language Understanding and Inference with MLLM in Visual Question Answering: A Survey (https://arxiv.org/abs/2411.17558)
- **What's New**: 이 논문에서는 Visual Question Answering (VQA)의 발전을 종합적으로 분석하고 최신 모델들에 대한 상세한 설명을 제공합니다. VQA는 자연어 처리(NLP)와 컴퓨터 비전의 교차점에서 중요한 작업으로, 최신 경향과 지식 추론 모듈에 대해서도 강조합니다. 특히, 이미지-질문 정보 기반의 인지 능력 강화를 위한 진전을 다룹니다.

- **Technical Details**: VQA 추론 과정에는 시각과 텍스트 정보에서 특성을 추출하고, 이들을 정렬하여 융합하는 과정이 포함됩니다. 최근의 VQA 모델들은 Graph Neural Networks(GNN)와 Transformer 아키텍처를 활용하여 시각-언어의 상호 작용을 극대화하고 있으며, Attention 메커니즘을 통해 인식과 정보를 보완하는 역할도 수행하고 있습니다. 이러한 모델들은 지식 추론 및 멀티모달 대형 언어 모델(MLLM)과 통합되어 더욱 발전하고 있습니다.

- **Performance Highlights**: 최신 VQA 모델들은 제로샷 질문 응답에 있어 뛰어난 성과를 보여주며, 실제 어플리케이션에서도 큰 영향을 미치고 있습니다. VQA는 시각 장애인을 돕는 다양한 방식과 이미지 검색, 자율 주행, 의료 진단 등 여러 분야에 활용됩니다. 앞으로 VQA는 비주얼 대화로 진화할 가능성도 있으며, 이에 대한 연구 및 발전도 기대됩니다.



### TAFM-Net: A Novel Approach to Skin Lesion Segmentation Using Transformer Attention and Focal Modulation (https://arxiv.org/abs/2411.17556)
- **What's New**: 이 논문은 TAFM-Net이라는 혁신적인 모델을 소개합니다. 이 모델은 피부 병변 분할(skin lesion segmentation)을 개선하기 위해 self-adaptive transformer attention(TA)과 focal modulation(FM)을 통합하여 개발되었습니다. EfficientNetV2B1 인코더를 활용하고, 모든 단계에서 세밀한 이미지 세부 사항을 최적화하는 새로운 동적 손실 함수가 포함되었습니다.

- **Technical Details**: TAFM-Net은 모듈화된 네 가지 구성 요소로 이루어져 있습니다: 인코더, 비전 트랜스포머(ViT), 포칼 변조 블록, 그리고 디코더 모듈입니다. 이 모델의 각 구성 요소는 상호 연결되어 있으며, 세부적인 특성을 보존하기 위해 encoder-decoder 스킵 연결(skip connections)에 FM이 통합되어 있습니다. 또한, 변조된 손실 함수는 입력 이미지와 실제 이미지 간의 정렬을 개선하여 모델 훈련을 더욱 정확하게 진행하도록 돕습니다.

- **Performance Highlights**: TAFM-Net은 ISIC2016, ISIC2017 및 ISIC2018 데이터셋에서 각각 93.64%, 86.88%, 92.88%의 Jaccard 계수를 달성했습니다. 이러한 결과는 실제 활용 가능성을 보여줍니다. 전통적인 방법들에 비해, TAFM-Net은 피부 병변을 자동으로 분리하는 데 효과적인 성능을 나타내며, 임상 평가에서 도움이 될 수 있습니다.



### On Statistical Rates of Conditional Diffusion Transformers: Approximation, Estimation and Minimax Optimality (https://arxiv.org/abs/2411.17522)
- **What's New**: 이번 연구에서는 조건부 확산 변환기(conditional diffusion transformers, DiTs)와 관련된 새로운 근사 및 추정 속도에 대한 분석을 수행했습니다. 특히, 분류기 없는 가이드를 사용하여 조건부 DiTs의 성능을 더욱 세밀히 검토하고, 이를 통해 더 효율적이고 정확한 DiT 모델 개발에 대한 실질적인 지침을 제공합니다.

- **Technical Details**: 연구는 네 가지 공통 데이터 가정 하에서 '상황(context)-내' 조건부 DiTs에 대한 포괄적인 분석을 제시합니다. 입력 도메인을 미세한 그리드로 이산화하고, 조건부 확산 점수 함수에 대해 테일러 전개(taylor expansion)를 수행하여 변환기의 보편적 근사를 보다 세부적으로 이용합니다. 이를 통해 더 낮고 굳건한 경계를 얻을 수 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, 잠재적인 조건부 DiTs는 조건부 DiTs보다 근사 및 추정에서 더 낮은 경계를 달성하며, 잠재적인 무조건 DiTs의 minimax optimality가 입증되었습니다. 이 연구는 조건부 및 무조건 DiTs에 대한 통계적 한계를 확립하고, 효율적이고 정확한 DiT 모델을 개발하기 위한 방향성을 제시합니다.



### Structure-Guided MR-to-CT Synthesis with Spatial and Semantic Alignments for Attenuation Correction of Whole-Body PET/MR Imaging (https://arxiv.org/abs/2411.17488)
- **What's New**: 이번 연구에서는 새로운 전신 MR-CT 합성 프레임워크를 제안합니다. 이는 구조 안내 합성, 공간 정렬 및 의미 정렬이라는 세 가지 혁신적인 모듈로 구성되어 있습니다. 이 프레임워크는 PET/MR 이미징에서 PET 감쇠 보정을 위한 고품질의 가상 CT 이미지를 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구의 주요 모듈은 다음과 같습니다: 첫 번째, 구조 안내 합성 모듈(SGSyn)은 구조 안내 주의 게이트를 활용하여 합성 이미지 품질을 향상시킵니다. 두 번째, 공간 정렬 모듈(SpatAlign)은 MR과 CT 이미지 간 정밀 등록을 달성하여 두 이미지 간의 정합성을 높입니다. 세 번째, 의미 정렬 모듈(SemAlign)은 대조 학습을 이용하여 신뢰할 수 있는 해부학적 의미 정보를 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 MR-CT 합성 프레임워크는 시각적으로 그럴듯하고 의미적으로 현실적인 CT 이미지를 생성할 수 있음을 입증하였습니다. 연구는 전신 MR-CT 합성이 PET 감쇠 보정에서의 유용성을 검증하였으며, 향후 PET/MR 이미징 분야에서의 정확성을 개선하는 데 기여할 것으로 기대됩니다.



### Learning New Concepts, Remembering the Old: A Novel Continual Learning (https://arxiv.org/abs/2411.17471)
- **What's New**: 이 논문에서는 Concept Bottleneck Models (CBMs) 내에서 새로운 개념 증대(concept-incremental) 및 클래스 증대(class-incremental) 지속 학습(continual learning) 작업을 정의합니다. 이를 통해 모델은 시간이 지남에 따라 새로운 개념과 클래스를 축적하면서 이전에 학습한 지식을 유지할 수 있는 방법을 제시합니다. 특히, CONceptual Continual Incremental Learning (CONCIL) 프레임워크를 통해 기존 기법의 한계를 극복하고 실시간 및 대규모 데이터 적용에 적합하도록 쉽게 구현할 수 있습니다.

- **Technical Details**: CONCIL 프레임워크는 개념 및 결정 계층 업데이트를 선형 회귀 문제로 재구성하여 치명적인 망각(catastrophic forgetting)을 방지합니다. 이 방식은 그라디언트 기반 업데이트(gradient-based updates)의 필요성을 없애며, 반복 행렬 연산(recursive matrix operations)만으로 모델을 업데이트하여 계산적으로 효율적입니다. 또한, 각 단계에서 현재 단계의 데이터와 이전 단계의 모델 가중치만을 사용하여 데이터 개인 정보 보호 및 저장 제약을 반영합니다.

- **Performance Highlights**: 실험 결과, CONCIL은 '절대 지식 기억(absolute knowledge memory)'을 달성하며, 개념 증대 및 클래스 증대 설정에서 기존 CBM 방법론보다 우수한 성능을 보여줍니다. 이를 통해 CBMs의 지속 학습 분야에 대한 새로운 벤치마크를 수립하며, 모델이 이전 단계에서의 지식을 완전히 유지하면서 진화하는 지식에 적응할 수 있는 능력을 입증합니다.



### Object-centric proto-symbolic behavioural reasoning from pixels (https://arxiv.org/abs/2411.17438)
- **What's New**: 이 연구에서는 자율 지능형 에이전트가 저수준 감지 입력과 모터 명령부터 고수준 추상적 추론 및 계획에 이르는 다양한 수준에서의 계산적 도전을 극복할 필요가 있음을 강조합니다. 기존의 감독 없이도 객체 중심(object-centric) 표현을 통해 세계를 이해하고 조작하는 새로운 뇌 영감을 받은(deep-learning) 아키텍처를 제안합니다. 이는 비용이 많이 드는 데이터 주석 없이도 다양한 수준의 정보를 학습할 수 있게 해줍니다.

- **Technical Details**: 제안된 아키텍처는 픽셀에서 학습하여 자기 환경을 해석하고 제어하며 추론할 수 있는 능력을 강화합니다. 에이전트는 논리적 추론(logical reasoning)과 연속 제어(continuous control)를 결합해야 하는 작업을 통해 그 유용성을 증명합니다. 에이전트는 emergent conditional behavioural reasoning과 같은 복잡한 논리적 연산을 학습하고자 하며, 이를 통해 환경을 조종하여 목표를 달성할 수 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 아키텍처는 에이전트가 환경의 예기치 않은 변화에 온라인으로 적응할 수 있으며, 상호작용하는 환경에서의 행동 추론(behavioral reasoning)이 가능하다는 것을 보여줍니다. 이는 에이전트가 동적 목표 생성(dynamically desired goal generation)을 통해 내장된 모델의 약간의 위반에도 강인성을 유지함을 의미합니다. 비록 결과가 실제 세계의 복잡성에는 미치지 않지만, 제안된 접근 방식은 비지도 학습을 위한 중요한 유도 편향을 통해 객체 중심 표현을 조작하는 방법을 제시합니다.



### Dual-Representation Interaction Driven Image Quality Assessment with Restoration Assistanc (https://arxiv.org/abs/2411.17390)
Comments:
          8 pages,6 figures, published to WACV

- **What's New**: 이번 논문은 저자들이 새로운 DRI(이중 표현 상호작용) 방법을 소개하여 저화질 이미지의 품질 평가를 향상시키는 데 중점을 두고 있습니다. 이 방법은 저화질 이미지의 열화(degradation) 및 품질 정보를 따로 모델링한 후, 복원 네트워크를 활용해 더 나은 성능을 제공하도록 설계되었습니다. 또한 저자들은 데이터 간의 효과적인 상호작용을 증진시키기 위해 RS Loss라는 새로운 손실 함수를 설계했습니다.

- **Technical Details**: 저자들은 DRI 방법을 통해 저화질 이미지의 열화 벡터와 품질 벡터를 따로 추출하며, 이를 기반으로 NR-IQA 모델의 강인성을 향상시키기 위한 다양한 알고리즘을 제안합니다. DRI-IQA는 이미지의 품질 점수를 예측하기 위해 아키텍처 내에서 품질 관련 열화 정보를 활용하는 복원 지원 모듈(RAM)을 통합하고, 열화 표현 사이의 상호작용을 강화하는 RS Loss를 도입합니다. 이 방식은 다양한 복원 네트워크와 왜곡 유형에 걸쳐 모델의 일반화를 높이는 데 기여합니다.

- **Performance Highlights**: 자체 실험 결과, DRI-IQA는 기존 최첨단 방법들과 비교해도 경쟁력 있는 성능을 보였습니다. 특히 합성 왜곡 이미지와 실제 왜곡 이미지, GAN 기반 복원 데이터 셋에서 모두 좋은 결과를 얻었습니다. 이렇게 DRI-IQA는 저화질 이미지와 복원 이미지의 품질 평가에서의 신뢰성을 향상시키고 있습니다.



### vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation (https://arxiv.org/abs/2411.17386)
- **What's New**: 이번 연구에서는 3D 혈관 분할을 위한 새로운 기초 모델인 vesselFM을 소개합니다. 기존의 모델들과 달리, vesselFM은 보지 못한 도메인에 효과적으로 일반화할 수 있는 능력을 가지고 있습니다. 이 모델은 세 가지 이질적인 데이터 소스에서 훈련되었으며, 특히 수작업으로 레이블링이 필요 없는 제로 샷 일반화를 달성할 수 있도록 설계되었습니다.

- **Technical Details**: vesselFM은 3D 혈관 분할을 위한 기초 모델로, 대규모 어노테이티드 데이터셋과 도메인 랜덤화 기술에서 생성된 데이터를 결합하여 훈련되었습니다. 실험을 통해, 이 모델은 다양한 의료 이미지 분할 기초 모델들과 비교해도 뛰어난 성능을 보여주며, 제로, 원, 및 소수 샷 시나리오에서도 효과적인 성능을 발휘합니다. 모델은 모든 단계에서 철저하게 검증되어 광범위한 임상 사용을 가능하게 합니다.

- **Performance Highlights**: vesselFM은 네 가지 (전)임상적으로 관련 있는 이미징 모달리티에서 샷 수에 관계없이 기존의 이미지 분할 모델들을 초월하는 성능을 보입니다. 특히, 모델은 제로 샷, 원 샷 및 소수 샷 시나리오에서 우수한 결과를 창출하여, 데이터 레이블링의 수고를 덜어주면서 임상의들이 효율적으로 3D 혈관 이미지를 분석할 수 있게 지원합니다.



### Automatic Skull Reconstruction by Deep Learnable Symmetry Enforcemen (https://arxiv.org/abs/2411.17342)
- **What's New**: 본 연구에서는 두 가지 딥 뉴럴 네트워크를 활용하여 두개골 결손을 자동으로 보완하는 혁신적인 방법을 제안합니다. 첫 번째 네트워크는 volumetric segmentation을 위해 설계된 인코더-디코더 아키텍처이며, 두 번째 네트워크는 건강한 두개골의 대칭 축을 계산하는 인코더-회귀 아키텍처입니다. 이러한 방법은 현재의 두개골 재건 기법에 비해 더 적은 계산 자원으로 더 높은 효율성을 제공합니다.

- **Technical Details**: 제안된 방법은 대칭 네트워크(SN)와 재건 네트워크(RN)로 구성되어 있습니다. SN은 건강한 두개골을 입력으로 받아 대칭 축을 계산하며, RN은 결손되거나 손상된 두개골의 재건을 담당합니다. 두 네트워크는 대칭 유지 강화를 통해 재건 과정을 개선하며, 이를 통해 재건의 질을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 대칭 보존 재건 네트워크는 기존의 기준 모델에 비해 상당히 우수한 성능을 보였습니다. DSC, bDSC, HD95 지표에서 우수한 결과를 보여주었으며, 상대적으로 적은 계산 자원으로도 동등한 성능을 달성했습니다. 이 연구는 의료 분야에서의 인공지능 응용 가능성을 한층 높이고, 향후 클리닉에서의 자동 두개골 결손 재건으로 나아가는 중요한 발판이 될 것입니다.



### TDAvec: Computing Vector Summaries of Persistence Diagrams for Topological Data Analysis in R and Python (https://arxiv.org/abs/2411.17340)
Comments:
          7 pages, 2 figures, 3 tables

- **What's New**: 이 논문에서는 Persistent Homology를 활용한 Topological Data Analysis (TDA)에서 데이터의 형태를 이해하는 데 사용되는 새로운 소프트웨어 패키지를 소개합니다. 이 패키지는 persistence diagrams (PDs)의 벡터화(vectorization)를 간소화하여 더 나은 머신러닝 응용프로그램을 위한 호환 형식으로 변환할 수 있도록 설계되었습니다. 실제 사례를 통해 패키지의 필요성을 입증하고, TDA의 적용에 대한 기여를 논의합니다.

- **Technical Details**: Persistent Homology는 다양한 차원의 topological feature를 분석하기 위한 기법으로, 데이터 점에서 simplicial complexes의 필터링을 생성하여 연결 요소, 루프, 공백 등의 다양한 특징을 포착합니다. Persistence Diagram (PD)은 이러한 topological 특징을 수학적으로 표현하는 다중 집합으로, 각 점은 이러한 특징의 출현과 소멸 시점을 나타냅니다. 그러나 PD는 비히르베르트 공간(non-Hilbert space)을 형성하기 때문에 머신러닝에서의 직접적인 사용은 제한적이며, kernel methods 및 vectorization 기법이 이러한 문제를 해결하는 데 주안점을 둡니다.

- **Performance Highlights**: 다양한 R 및 Python 패키지가 TDA를 지원하지만, 기존의 벡터화 방법들이 각기 호환되지 않으며, 대규모 계산에서 비효율적일 수 있습니다. 이 새로운 패키지는 모든 벡터화 방법을 통합하여 일관된 방식으로 제공함으로써 사용자가 서로 다른 방법을 쉽게 비교할 수 있도록 합니다. 패키지에 포함된 기능들을 통해, 데이터 분석의 정확성과 효율성을 높일 수 있습니다.



### Interpretable label-free self-guided subspace clustering (https://arxiv.org/abs/2411.17291)
Comments:
          45 pages; 3 figures; 10 tables

- **What's New**: 본 논문은 하이퍼파라미터( hyperparameters)에 독립적인 Majority Subspace Clustering(SC) 알고리즘의 하이퍼파라미터 최적화(Hyperparameter Optimization, HPO) 접근법을 제안합니다. 기존의 SC 알고리즘은 성능을 극대화하기 위해 하이퍼파라미터 튜닝을 필요로 하지만, 많은 도메인에서 이 데이터에 대한 레이블을 확보하는 것이 어렵습니다. 이 연구는 내부 클러스터링 품질 메트릭(internal clustering quality metrics)을 활용하여 레이블-independent HPO를 수행하는 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 SC 알고리즘을 통해 얻은 의사 레이블(pseudo-labels)을 기반으로 하는 클러스터링 품질 메트릭인 정확도(Accuracy, ACC) 또는 정규화된 상호 정보(Normalized Mutual Information, NMI)를 사용합니다. 하이퍼파라미터 값에 따른 ACC(혹은 NMI)가 부드러운 함수(smooth function)라고 가정하여, 하이퍼파라미터의 부분 구간(subintervals)을 선택하고 이를 반복적으로 이분화하여 상대 오차 기준(relative error criterion)을 만족할 수 있습니다. 이를 통해 모든 SC 알고리즘의 하이퍼파라미터 튜닝이 가능해집니다.

- **Performance Highlights**: 본 방법은 여러 단일 및 다중 뷰 SC 알고리즘을 테스트하여 성능을 평가하였고, 여섯 개의 데이터 세트에서 오라클(oracle) 알고리즘과 비교할 수 있었습니다. 제안된 방법은 일반적으로 오라클 버전보다 5%에서 7% 낮은 클러스터링 성능을 달성하였습니다. 또한, 클러스터링 파티션에서 추정한 서브스페이스 기초(subspace bases)를 시각화하여 하이퍼파라미터 탐색 공간의 초기 선택을 돕는 해석 가능성을 제공하였습니다.



### A Topic-level Self-Correctional Approach to Mitigate Hallucinations in MLLMs (https://arxiv.org/abs/2411.17265)
- **What's New**: 이 논문에서는 인간의 선호와 더 잘 일치하도록 다중 모달 대형 언어 모델(MLLM)의 행동을 조정하는 것이 필수적임을 강조합니다. 최근에는 전문가나 보조 AI 시스템을 활용하여 선호 피드백을 더욱 정확하게 제공하려는 시도가 있었으나, 이러한 방식은 자원이 많이 소모되어 확장성에 문제가 있었습니다. 이에 대해 'Topic-level Preference Overwriting (TPO)'라는 자가 교정 방식을 도입하여 모델 자체가 주제 수준에서 오차를 줄일 수 있도록 하였습니다.

- **Technical Details**: TPO는 모델이 스스로의 환각(hallucination)을 감지하고 수정할 수 있도록 하여 피드백 수집의 자동화 및 확장성을 목표로 합니다. 이 방식은 복잡한 응답을 구성하는 다양한 주제를 분리하고, 각 주제에 대해 모델이 생성한 최적의 대안으로 교체하여 더 뚜렷한 선호 쌍을 만들어 냅니다. 주제 수준의 수정으로 응답의 의미와 구조가 단순화되어 보다 정확한 후보 재샘플링이 가능해지고, 피드백 수집 과정에서 일관성을 유지할 수 있습니다.

- **Performance Highlights**: 'TPO'는 기존의 환각 벤치마크에서 신뢰성 면에서 최첨단의 성능을 보여줍니다. ObjectHal-Bench에서 모델의 환각을 약 92% 감소시켰고, MMHal-Bench에서는 38% 감소에 성공했습니다. 이러한 성과는 모델 자체가 선호 쌍을 생성하도록 설정하여 이루어진 것으로, 인간 피드백이나 독점 모델의 개입 없이도 가능했습니다.



### MiceBoneChallenge: Micro-CT public dataset and six solutions for automatic growth plate detection in micro-CT mice bone scans (https://arxiv.org/abs/2411.17260)
Comments:
          Under Review

- **What's New**: 이 논문은 미세 CT 스캔을 통한 자동 뼈 정량화 모델 개발과 관련된 도전 과제를 다룹니다. 이를 위해, 83마리 생쥐의 3D μCT 뼈 스캔으로 구성된 고품질 데이터셋을 준비하고 주석을 달았습니다. 전 세계 80명 이상의 AI 과학자들이 이 도전에 참여하여, 뼈 성장 판을 효과적으로 식별할 수 있는 6개의 컴퓨터 비전 솔루션을 개발했습니다.

- **Technical Details**: 뼈 조직의 연속적인 활성 재형성은 조골세포(osteoblasts)와 파골세포(osteoclasts)의 복잡한 상호작용에 의해 조절됩니다. μCT(미세 컴퓨터 단층 촬영)는 뼈의 미세 구조를 고해상도로 평가할 수 있는 유용한 기술로, 이미지를 통해 뼈 미네랄 밀도를 정량화할 수 있습니다. 본 연구에서는 높은 차원성의 3D 데이터를 효과적으로 처리하기 위해 2D 슬라이스를 사용하는 방법으로 모델을 안정화하며 정보 손실을 최소화하는 여러 가짜 3D(pseudo-3D) 방법들이 제안되었습니다.

- **Performance Highlights**: 개발된 솔루션들은 평균 절대 오차가 1.91±0.87 평면으로, 방사선 전문의가 실용적으로 사용할 수 있는 정확도 수준에 도달했습니다. 연구에서 공유한 고품질 3D μCT 데이터셋과 함께, 모든 코드와 모델들이 공개되며 이는 연구자들이 자체적인 접근법을 개발하고 벤치마크할 수 있는 기회를 제공합니다. 이러한 작업을 통해, 자동화된 뼈 정량화 프로세스가 약물 개발 연구의 효율성을 크게 향상시킬 것으로 기대됩니다.



### LHPF: Look back the History and Plan for the Future in Autonomous Driving (https://arxiv.org/abs/2411.17253)
- **What's New**: 이 논문에서는 LHPF라는 새로운 모방 학습 기반의 계획 플래너를 소개하고 있습니다. 이 플래너는 기존의 알고리즘과 달리 역사적 계획 정보를 통합하여 현재의 관찰 여부와 과거의 계획 의도를 함께 분석합니다. 이를 통해 계획의 연속성을 높이고, 드라이빙 의도가 변할 때의 오류 누적을 방지하는 데 중점을 두고 있습니다.

- **Technical Details**: LHPF는 기존의 계획 모듈에 역사적 의도 집계 모듈을 추가하여, 관측 데이터를 기반으로 하여 여러 프레임의 정보를 통합하여 최종 경로 계획을 생성합니다. 이 구조는 스페이셜(Spatial) 쿼리 벡터와 함께 작동하여 최적의 경로를 디코드합니다. 또한, 운전 행동의 인간 유사성을 높이기 위해 편안함 관련 보조 작업을 통합하였습니다.

- **Performance Highlights**: 실제 및 합성 데이터를 이용한 실험 결과, LHPF는 최신 학습 기반 플래너들을 초월하는 성능을 보였으며, 순수 학습 기반 방법으로서 전문가의 성과를 능가하는 첫 번째 사례로 기록되었습니다. 이러한 성과는 LHPF 방법이 다양한 백본에서도 효과적으로 작용하며, 시스템에 쉽게 통합될 수 있음을 보여줍니다.



### cWDM: Conditional Wavelet Diffusion Models for Cross-Modality 3D Medical Image Synthesis (https://arxiv.org/abs/2411.17203)
Comments:
          BraTS 2024 (Global Synthesis) submission. Code: this https URL

- **What's New**: 이 논문은 "BraTS 2024 Brain MR Image Synthesis Challenge"에 기여하며, 고해상도 볼륨에서 이미지 대 이미지 번역 작업을 직접 해결하기 위한 조건부 Wavelet Diffusion Model(cWDM)을 제안합니다. 기존의 brain tumor segmentation models는 다양한 MR 스캔이 필요하지만, 특정 모달리티가 누락될 경우 자동 세분화 알고리즘의 활용이 어려워집니다. 본 연구는 세 개의 가능한 이미지를 기반으로 누락된 모달리티 이미지를 합성하는 방법을 제시하여 분할 모델의 적용을 가능하게 합니다.

- **Technical Details**: 본 연구에서 제안하는 cWDM은 전통적인 3D 의료 이미지 생성 방법의 한계를 극복하기 위해 Wavelet Diffusion Model을 사용하여 고해상도 3D 이미지의 쌍 대 이미지 번역 문제를 해결합니다. 또한, 우리는 이 작업을 조건부 생성 문제로 간주하고 간단한 조건화 전략과 함께 적용하여 슬라이스 또는 패치별 데이터 처리로 인한 아티팩트를 피하는 방법을 제안합니다. 이 접근 방식은 CT와 MR, MR과 PET의 번역 등 다양한 쌍체 이미지-이미지 번역 문제에 응용될 수 있습니다.

- **Performance Highlights**: Wavelet Diffusion Models은 고해상도의 의료 이미지를 효율적으로 생성하는 데 성공하였으며, 이 방법은 슬라이스 간 불일치를 방지하는 데 유리합니다. GAN 기반의 방법들과 비교할 때, cWDM은 훈련이 더 용이하고, 3D 데이터에서 발생할 수 있는 문제를 줄이며, 실질적인 의료 이미지 처리를 위한 새로운 가능성을 열어줍니다. 본 연구 결과는 고해상도의 3D 볼륨을 대상으로 한 이미지 합성 작업에서 GAN보다 우수한 성능을 보여 주목받고 있습니다.



### Motion Free B-frame Coding for Neural Video Compression (https://arxiv.org/abs/2411.17160)
Comments:
          Deep Neural Video Compression

- **What's New**: 본 논문에서는 기존 비디오 압축 아키텍처의 단점을 해결하기 위해 새로운 접근방식인 kernel-based motion-free video coding을 제안합니다. 이 방식은 motion estimation(모션 추정), motion compensation(모션 보상), motion coding(모션 코딩)을 제거함으로써 네트워크의 인코딩 효율성을 향상시키고 계산 복잡성을 크게 줄입니다. 또한, kernel-based auto-encoder는 일반적인 대칭 auto-encoder에서 발생하는 흐림 아티팩트(blurry artifacts)를 줄여 reconstructed frames의 시각적 품질을 개선합니다.

- **Technical Details**: 전통적인 비디오 코딩은 이미지 신호의 공간 및 시간 중복성을 활용한 예측 코딩의 하이브리드 접근 방식을 따른다. 하이브리드 접근법에서 motion coding과 residual coding의 분리를 제거하고, 인코더는 현재 코딩 프레임과 이웃 프레임에서 합성된 정보를 기반으로 latent variables(잠재 변수)로 변환합니다. 디코더는 convolutional kernel-based synthesizer를 사용하여 재구성된 프레임을 생성하며, 이 과정에서 픽셀 의존적 1D convolution kernels가 모션과 강도 정보를 캡처하여 시각적 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 HEVC-class B 데이터셋에서 기존의 SOTA(deep neural video compression networks)를 초월하며, UVG 및 MCL-JCV 데이터셋에서도 경쟁력을 보였습니다. 기존의 대칭 auto-encoder에 비해 재구성된 프레임의 품질이 높고, 모델 크기는 모션 기반 네트워크보다 3배에서 4배 작은 것이 특징입니다. 이러한 효율성은 새로운 motion-free coding 접근법을 통해 가능해졌습니다.



### On-Road Object Importance Estimation: A New Dataset and A Model with Multi-Fold Top-Down Guidanc (https://arxiv.org/abs/2411.17152)
- **What's New**: 이 논문은 운전자의 시점에서 캡처된 비디오 시퀀스를 입력으로 사용하여 도로에서의 객체 중요성 추정을 다룹니다. 이 문제는 안전하고 스마트한 운전 시스템을 위해 중요하지만, 관련 연구가 제한적입니다. 이에 대한 솔루션으로, 저자들은 Traffic Object Importance (TOI)라는 새로운 대규모 데이터셋을 제안하고 있습니다.

- **Technical Details**: TOI 데이터셋은 9,858프레임, 28개 장면, 44,120개의 객체 중요성 주석을 포함하며, 기존의 Ohn-Bar 데이터셋에 비해 프레임 수가 3.1배, 장면 수가 3.5배 증가하였습니다. 논문은 다중 최상위 유도 요인(예: 운전자의 의도, 의미적 맥락, 교통 규칙)을 통합하여 객체 중요성을 추정하는 새로운 모델을 제안합니다. 이 모델은 상향식(bottom-up)과 하향식(top-down) 경로를 융합하여 객체의 중요성을 평가합니다.

- **Performance Highlights**: 저자들의 제안된 모델은 광범위한 실험을 통해 기존 최첨단 방법들보다 23.1% 높은 Average Precision (AP) 향상을 달성했습니다. 다양한 비교 및 ablation 연구를 통해 제안된 모델의 우수성을 입증하였으며, 각 요소의 상호작용이 객체 중요성 추정의 성능을 높인다는 것을 확인했습니다.



### Neural-Network-Enhanced Metalens Camera for High-Definition, Dynamic Imaging in the Long-Wave Infrared Spectrum (https://arxiv.org/abs/2411.17139)
- **What's New**: 이 논문에서는 싱글렛(Singlet)을 이용한 장파 적외선 이미징에 대한 경량화 및 비용 효과적인 솔루션을 제공하기 위해, High-Frequency-Enhancing Cycle-GAN 신경망을 금속렌즈(metalens) 이미지 시스템에 통합한 카메라를 개발하였습니다. 이 카메라는 원래 금속렌즈 이미지의 품질을 향상시키기 위해 본래의 주파수 손실을 해결합니다.

- **Technical Details**: High-Frequency-Enhancing Cycle-GAN은 양방향 순환 생성적 적대 신경망(bidirectional cyclic generative adversarial network) 및 고주파 적대 학습 모듈을 통합합니다. 고주파 적대 학습 모듈은 웨이브렛 변환(wavelet transform)을 사용하여 고주파 성분을 추출하고, 이를 기반으로 고주파 피드백 루프를 구성하여 생성기가 고주파 판별기로부터의 적대적 피드백을 통합하여 카메라 출력 품질을 향상시킵니다.

- **Performance Highlights**: 이 카메라는 초당 125프레임의 동적 이미징(dynamic imaging)을 달성하며, End Point Error 값은 12.58, Fréchet Inception Distance는 0.42, Peak Signal to Noise Ratio는 30.62, 그리고 Structural Similarity는 0.69라는 우수한 성능을 보여줍니다.



### Contrastive CFG: Improving CFG in Diffusion Models by Contrasting Positive and Negative Concepts (https://arxiv.org/abs/2411.17077)
Comments:
          14 pages, 8 figures

- **What's New**: 본 논문에서는 Classifier-Free Guidance (CFG)의 부정적 가이던스 기법을 개선하기 위해 ‘contrasting loss’를 활용한 새로운 접근 방식을 제안합니다. 이 방법론은 주어진 조건에 따라 노이즈 제거 방향을 정렬하거나 반발시키는 방식으로 작동하여, 기존의 부정적 가이던스 방법의 한계를 극복합니다. 실험 결과, 제안된 방법이 다양한 시나리오에서 바람직하지 않은 개념을 효과적으로 제거하면서 샘플 품질을 유지하는 것을 보여주었습니다.

- **Technical Details**: 제안된 방법은 Contrastive CFG(CCFG)라고 불리며, 이는 주어진 조건에 맞추어 노이즈 제거 방향을 최적화하는 방식입니다. CCFG는 기존의 CFG와 유사한 가이던스를 생성하면서도 부정적 가이던스의 단점을 보완합니다. 이 과정에서 샘플링 절차와 관련된 상당한 계산 오버헤드를 피하면서 노이즈 제거 방향을 자동으로 조절합니다.

- **Performance Highlights**: CCFG는 여러 실험을 통해 바람직하지 않은 개념을 성공적으로 회피하면서 샘플 품질을 보장하는 것을 입증하였습니다. 특히, 단순한 클래스 조건부터 복잡하고 중첩된 텍스트 프롬프트에 이르기까지 다양한 시나리오에서 효과성을 보여주었습니다. 이로 인해, 동적이고 복잡한 조건에서의 샘플링 효과를 향상시키는데 기여할 수 있습니다.



### A generalised novel loss function for computational fluid dynamics (https://arxiv.org/abs/2411.17059)
Comments:
          37 pages, 13 figures, preprint submitted to Engineering Applications of Artificial Intelligence (EAAI)

- **What's New**: 이 논문은 Computational Fluid Dynamics (CFD) 시뮬레이션에 기존의 딥 러닝 아키텍처를 효율적으로 적용하기 위해 새로운 손실 함수인 Gradient Mean Squared Error (GMSE)를 제안하고 있습니다. GMSE는 지역별로 중요한 데이터 영역을 자동으로 인식하고, 이를 기반으로 가중치를 수동적으로 조정하여 연산 속도를 높이는데 기여합니다. 기존의 방식과 비교했을 때, 제안된 방법은 훈련 시간 단축과 정확도 향상을 동시에 달성했습니다.

- **Technical Details**: GFME 손실 함수는 2D 및 3D CFD 데이터의 복잡성을 고려하여, 데이터의 작은 변동 덕분에 큰 피해를 입을 수 있는 영역을 집중적으로 처리합니다. 독특하게도, GMSE는 특정 데이터 세트에 대한 조정이나 수정 없이도 다양한 데이터에 적용될 수 있습니다. 또한, 이 손실 함수는 점진적으로 적합한 가중치를 결정하며, 이는 지역 속도 그래디언트를 기반으로 설정됩니다.

- **Performance Highlights**: 실험 결과, GMSE 손실 함수를 이용한 네트워크는 Mean Squared Error (MSE) 손실 및 동적 변형 GMSE(DGMSE)와 비교해 구조적 유사성 오류가 83.6% 감소하였습니다. 추가적으로 훈련 과정에서의 손실 수렴 속도가 빨라졌고, 판별기 네트워크를 속이기 위한 최대 손실률이 76.6% 상승했습니다. 이러한 성과들은 CFD 분야에서의 머신 러닝 처리를 가속화 할 가능성을 제시합니다.



### Improving Deformable Image Registration Accuracy through a Hybrid Similarity Metric and CycleGAN Based Auto-Segmentation (https://arxiv.org/abs/2411.16992)
- **What's New**: 이 연구에서는 전통적인 강도 기반의 변형 가능 이미지 등록(Deformable Image Registration, DIR) 방법이 이미지 강도가 다를 경우 실패하는 문제를 해결하기 위해 강도(intensity)와 구조적 정보(structural information)를 결합한 하이브리드 유사성(metric)을 평가했습니다. 특히 CycleGAN 기반의 강도 교정(intensity correction)과 자동 분할(auto-segmentation)을 활용하여 세 가지 DIR 워크플로를 비교했습니다.

- **Technical Details**: 하이브리드 유사성 지표는 포인트-투-디스턴스(point-to-distance, PD) 점수와 강도 유사성을 결합한 형태로 설계되었습니다. 무쌍 CT 및 CBCT 이미지를 사용하여 2D CycleGAN 모델을 훈련시켜 합성 CT(synthetic CT, sCT) 이미지를 생성했고, 3D U-Net 모델을 통해 전립선, 방광, 직장을 분할했습니다. DIR 정확도는 Dice Similarity Coefficient(DSC), 95% Hausdorff Distance(HD), 지표 분리(fiducial separation)를 사용하여 평가되었습니다.

- **Performance Highlights**: 하이브리드 유사성 지표의 도입으로 DIR 정확도가 향상되었습니다. 전립선의 경우 DSC는 No PD에서 0.61+/-0.18에서 CycleGAN PD에서는 0.82+/-0.13로, Expert PD에서는 0.89+/-0.05로 증가했으며, 95% HD는 각각 11.75 mm에서 4.86 mm 및 3.27 mm로 감소했습니다. 이러한 개선은 방광과 직장에서도 관찰되었으며, CycleGAN 기반 자동 분할이 임상 과정의 정밀도를 향상시킬 수 있는 잠재력을 보여줍니다.



### Glo-In-One-v2: Holistic Identification of Glomerular Cells, Tissues, and Lesions in Human and Mouse Histopathology (https://arxiv.org/abs/2411.16961)
- **What's New**: 본 연구에서는 Glo-In-One toolkit의 버전 2인 Glo-In-One-v2를 도입하여 소형이식체 구획 분석을 위한 14개 레이블을 적용했습니다. 이 도구키트는 23,529개의 주석이 있는 사구체를 분석한 데이터셋을 기반으로 하여, 인간 및 쥐 조직 샘플을 세분화하는 심층 학습 아키텍처를 사용합니다. 또한, 쥐 샘플에서의 학습 결과를 통해 인간 샘플에서의 병변 구획 분석 성능을 향상시켰습니다.

- **Technical Details**: Glo-In-One-v2는 각 조직 또는 병변의 특정 클래스를 위한 태스크 인식을 위해 잔여 U-Net 아키텍처를 사용합니다. 이 네트워크는 368개의 주석이 있는 신장 전체 슬라이드 이미지에서 훈련되어 5개의 내부 신장 구조를 식별하고 9개의 병변 유형을 세분화합니다. 또한, 동적 필터 생성을 통해 특정 클래스에 특화된 커널을 생성하여 다중 라벨 세분화를 수행하도록 설계되었습니다.

- **Performance Highlights**: 모델은 기본 모델들과 비교하여 괜찮은 성능을 달성했으며, 평균 Dice Similarity Coefficient(DSC) 76.5%를 기록했습니다. 전이 학습을 활용하여 쥐에서 인간으로의 종간 전이 성능을 평가하면서 병변 유형 간 평균 세분화 정확도를 3% 이상 향상시켰습니다. 공개된 Glo-In-One-v2 모델과 학습된 가중치는 https://github.com/hrlblab/Glo-In-One_v2에서 확인할 수 있습니다.



### RoCoDA: Counterfactual Data Augmentation for Data-Efficient Robot Learning from Demonstrations (https://arxiv.org/abs/2411.16959)
- **What's New**: 이번 논문에서는 로봇 공학에서 모방 학습(imitation learning)의 근본적인 도전 과제인 일반화(generalization)의 한계를 해결하기 위해 RoCoDA라는 새로운 방법을 소개합니다. RoCoDA는 불변성(invariance), 등변성(equivariance), 인과관계(causality)를 통합하여 데이터 증강(data augmentation)을 향상시키는 프레임워크를 제공합니다. 이를 통해 환경 상태의 무관한 부분을 수정하면서 정책(output)에 영향을 주지 않는 인과 불변성을 활용합니다.

- **Technical Details**: RoCoDA는 SE(3) 등변성을 활용하여 물체의 자세(object poses)에 대한 강체 변화(rigid body transformations)를 적용하고, 해당 행동(actions)을 조정하여 합성 시연(synthetic demonstrations)을 생성합니다. 이 방법을 통해 정책의 성능 성능 성능(performance)을 개선하고, 일반화(generalization) 및 샘플 효율성(sample efficiency)를 높일 수 있음을 입증했습니다. 고전적인 로봇 조작 과제에서 RoCoDA가 기존 데이터 증강 방법에 비해 우수한 성과를 달성하는 것이 확인되었습니다.

- **Performance Highlights**: RoCoDA를 통해 훈련된 정책들은 보지 못했던 물체 자세(object poses), 질감(textures), 방해물(distractors)에도 강력하게 일반화(generalization)됩니다. 또한, 재잡기(re-grasping)와 같은 emergent behavior가 관찰되었으며, 이는 RoCoDA를 통해 훈련된 정책이 작업의 동적(task dynamics)을 더 깊이 이해하고 있음을 나타냅니다. 이러한 접근 방식을 통해 RoCoDA는 복잡한 로봇 작업에서 일반화와 성능을 위한 원칙적(principled)인 방법을 제공합니다.



### Contrastive Deep Learning Reveals Age Biomarkers in Histopathological Skin Biopsies (https://arxiv.org/abs/2411.16956)
Comments:
          20 pages, 5 tables, 5 figures Under review: npj Digital Medicine

- **What's New**: 이 논문은 피부 생검(skin biopsy) 이미지만으로 개인의 나이를 결정할 수 있음을 보여줍니다. 이를 위해 시각적 특성을 활용하여 노화의 새로운 바이오마커를 개발하였고, 덴마크의 건강 등록 자료와 연결하여 이를 검증했습니다. 이 연구는 딥러닝(deep learning) 및 주기적으로 수집된 건강 데이터를 결합하여 노화 예측의 새로운 가능성을 제시합니다.

- **Technical Details**: 이 연구는 대조 딥러닝(contrastive deep learning) 기법을 사용하여 피부 생검의 병리 슬라이드(histopathology slides)에서 시각적 특성을 추출하고, 이를 통해 노화 바이오마커(biomarker of ageing)를 구축했습니다. 피부 생검 샘플의 세포 구성(cellular composition) 및 조직 구조(tissue structure)의 변화가 나이를 결정짓는 데 중요한 역할을 한다는 점에 주목했습니다. 또한, 딥러닝 모델은 이러한 라이브러리에서 대규모 데이터셋을 활용해 노화 관련 생물학적 특성을 예측하는 데 사용되었습니다.

- **Performance Highlights**: 연구 결과, 피부 생검에서 추출된 시각적 특성이 개인의 사망(mortality)과 만성 노화 관련 질병(prevalence of chronic age-related diseases)을 예측하는 데 유의미한 역할을 한다는 점이 밝혀졌습니다. 1787개의 디지털화된 피부 생검 샘플을 통해, 노화에 관련된 다양한 특성을 추출하고 이를 바탕으로 여러 연령 관련 질병의 유병률을 예측할 수 있음을 확인했습니다. 이러한 결과는 기존의 생물학적 나이(biological age)를 시간에 따라 변화를 모니터링할 수 있는 유용한 지표로 활용할 수 있도록 합니다.



### U-WNO:U-Net-enhanced Wavelet Neural Operator for fetal head segmentation (https://arxiv.org/abs/2411.16890)
- **What's New**: 이 논문은 U-Net 강화 Wavelet Neural Operator (U-WNO)의 개발을 다룹니다. U-WNO는 웨이브릿 분해(wavelet decomposition), 오퍼레이터 학습(operator learning), 인코더-디코더 메커니즘을 결합하여 기능적으로 우수한 성능을 보여줍니다. 특히, 이 방법은 공간 영역에서 패턴 추적을 위한 정확한 세분화 맵(segmentation map)을 생성하는 데 효과적입니다.

- **Technical Details**: U-WNO는 웨이브릿 기반의 특성 추출과 ConvNet을 통합하여 복잡한 함수 근사 및 다중 해상도(multi-resolution) 데이터 처리를 가능하게 합니다. 이 모델은 임신 중 초음파 영상을 분석하여 정확한 지역 세분화를 지원합니다. WNO는 웨이브릿 변환의 원리를 활용하여 공간 및 시간 데이터의 복잡한 다중 스케일 패턴을 포착하는 데 효과적입니다.

- **Performance Highlights**: U-WNO는 두 차원 초음파 이미지에서 태아 머리의 시맨틱 세분화(semantic segmentation)를 수행하며, 주로 dice score를 활용해 성능을 평가합니다. 이 기술은 의사들이 태아의 건강 및 발달을 모니터링하는 데 도움을 주며, 학생들이 실제 환자 없이도 초음파 절차를 연습할 수 있도록 하는데 유익합니다.



### Leveraging Foundation Models To learn the shape of semi-fluid deformable objects (https://arxiv.org/abs/2411.16802)
- **What's New**: 이 논문은 변형 가능한 물체의 특성을 정의하고 조작을 위해 필요한 키포인트(keypoints)를 탐지하는 문제를 다룬다. 새로운 접근법으로, 기존의 데이터 세트 필요 없이 이미지에서 변형된 물체의 특성을 파악하는 생성적 모델을 사용하는 방법론을 제안한다. 특히, 용접 풀(weld pool)과 같은 유체 변형 가능 물체의 모양을 추출하고 분석하여, 로봇 조작의 안정성을 향상시키는 데 초점을 맞춘다.

- **Technical Details**: 이 연구에서는 Teacher-Student 프레임워크를 통해 두 개의 기초 모델인 DINO와 SAM2를 활용하여 물체의 형태를 복원하였다. Teacher 네트워크는 변형 가능한 물체의 마스크 및 열지도를 생성하는 데 사용되며, Student 네트워크는 Variational Autoencoder(VAE) 아키텍처를 통해 데이터를 효율적으로 학습한다. 이러한 접근법은 이전에 수작업으로 레이블링된 데이터에 의존하지 않으며, 픽셀 수준에서의 정보를 파악하기 위한 효과적인 방법을 제시한다.

- **Performance Highlights**: Student 네트워크는 물체의 키포인트를 13.4픽셀의 오차로 추출하는 능력을 보였다. Teacher 네트워크는 물체의 마스크로 표현된 픽셀 수준의 정보를 조회하는 데 있어 평균 교차 영점 비율(mIoU) 75.26%를 기록하여, 변형 가능한 물체의 특성을 정량적으로 평가하는 데 우수한 성능을 발휘하였다. 이러한 연구 결과는 로봇 용접 작업에서 용접 풀의 동적인 변화에 대한 예측 및 조작 개선 가능성을 보여준다.



### Scaling Laws for Black box Adversarial Attacks (https://arxiv.org/abs/2411.16782)
- **What's New**: 이번 연구에서는 블랙박스 적대적 공격의 스케일링 법칙을 탐구했습니다. 모델 앙상블 방식을 활용하여 여러 대리 모델을 동시에 공격함으로써 적대적 전이 가능성을 높이는 효과를 극대화했습니다. 연구 결과, 더 많은 대리 모델을 사용할수록 공격 성공율이 증가하는 명확한 경향이 관찰되었습니다.

- **Technical Details**: 연구에서는 대리 모델 수와 적대적 예시의 전이 가능성 간의 관계를 분석하였습니다. 이는 이론적 분석과 실험적 연구를 통해 이루어졌으며, 전이 기반 공격의 목표는 자연 이미지에서 목표 클래스에 대한 적대적 예시를 만드는 것입니다. 이 과정에서 손실 함수와 제약 조건을 설정하고 ℓ∞ 노름을 고려하여 최적화 문제를 정의하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해, 블랙박스 분류기에 대한 공격 성공률이 거의 100%에 이르는 결과를 보였습니다. 또한, GPT-4와 같은 상업적 모델을 대상으로 한 공격에서도 90% 이상의 성공률을 기록하였습니다. 이는 모델 스케일링의 중요성을 강조하며, 적대적 예시가 모델의 공통 특성을 포착하는 데 기여함을 나타냅니다.



### In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models (https://arxiv.org/abs/2411.16769)
- **What's New**: 이번 연구에서 우리는 텍스트-이미지(T2I) 모델의 안전 메커니즘을 평가하기 위한 혁신적인 레드 팀잉 프레임워크인 ICER를 도입합니다. ICER는 과거에 성공적인 레드 팀잉 시도로부터 배우고 이를 기반으로 해석 가능한 문제성 프롬프트를 생성하는 혁신적인 접근 방식으로, 기존 프롬프트 공격 방법들에 비해 뛰어난 성능을 보여줍니다. 이 프레임워크는 LLM(대형 언어 모델)을 활용하여 추가 훈련 없이 다양한 T2I 모델의 안전 메커니즘을 효율적으로 조사할 수 있도록 설계되었습니다.

- **Technical Details**: ICER 프레임워크는 과거의 경험을 기록하고 이를 참고하여 새로운 레드 팀잉 전략을 탐색하는 방식으로 구성됩니다. 핵심적으로, 우리는 LLM을 활용하여 해석 가능한 적대적 프롬프트를 생성하며, 이 과정에서 밴딧 최적화 알고리즘을 사용하여 중요성을 고려함으로써 다양한 안전 메커니즘을 효율적으로 테스트합니다. 연구에서는 성공적이고 실패한 공격의 경험을 바탕으로 프롬프트 효과성을 향상시키기 위해 Bayesian Optimization(베이지안 최적화) 프레임워크 내에서 세 가지 주요 요소를 통합합니다.

- **Performance Highlights**: 실험 결과, ICER는 기존의 프롬프트 공격 방법들에 비해 T2I 모델의 취약성을 식별하는 데 있어 우수한 성능을 보였습니다. 발견된 문제성 프롬프트는 원래 입력과 높은 의미적 유사성을 유지하며 의도된 콘텐츠를 효과적으로 탈옥하는 결과를 가져왔습니다. 이러한 접근 방법은 T2I 모델의 안전성을 평가하는 데 있어 현실적이고 도전적인 평가를 가능하게 하여, 향후 잠재적 악용에 대한 보다 강력한 안전 장치를 개발하는 데 기여할 것으로 기대됩니다.



### SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction (https://arxiv.org/abs/2411.16765)
Comments:
          17 pages

- **What's New**: SHuBERT(사인 히든 유닛 BERT)는 자동 수화 처리 시스템의 발전을 가속화할 수 있는 자기 지도 학습 기반의 transformer 인코더입니다. 약 1,000시간의 미국 수화(ASL) 비디오 콘텐츠에서 강력한 표현을 학습하며, 기존의 테스크 특화 모델을 넘어서 다중 테스크에서의 이전 학습 가능성을 높입니다. SHuBERT는 다양한 비디오 스트림(손, 얼굴, 몸 자세)의 클러스터 할당을 예측하여 자가 감독 학습을 적용합니다.

- **Technical Details**: SHuBERT는 비디오 프레임의 문맥적 표현을 학습하기 위해 multi-stream 접근 방식을 채택하고, 손, 얼굴, 몸 자세 피처의 여러 masked 비디오 스트림을 사용하여 훈련합니다. 사용된 비디오 피처는 MediaPipe Hand Landmarker 모델을 통해 손의 특징을 검출하며, 검출의 정확도가 95%에 달합니다. SHuBERT는 이렇게 학습된 표현이 여러 하위 작업에 효과적으로 전달될 수 있음을 입증하였습니다.

- **Performance Highlights**: SHuBERT는 sign language translation(SLT)과 isolated sign language recognition(ISLR) 벤치마크에서 최첨단 성능을 달성하였습니다. 예를 들어, How2Sign 데이터셋에서는 +0.7 BLEU, OpenASL에서는 +10.0 BLEU, ASL-Citizen에서는 +5% 정확도를 초과하는 성과를 보였습니다. 이러한 결과는 전문화된 아키텍처에 비해 적은 테스크 특정 주석으로도 뛰어난 성과를 나타낼 수 있음을 보여줍니다.



### FunGrasp: Functional Grasping for Diverse Dexterous Hands (https://arxiv.org/abs/2411.16755)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 FunGrasp라는 시스템을 소개하여, 여러 로봇 손에서 기능적 정교한(Functional Dexterous) 그립(grasping)을 가능하게 하고, 이전에 보지 않은 객체에 대해 원샷(One-shot) 전이를 수행할 수 있도록 합니다. 단일 RGBD 이미지에서 인체의 기능적 그립을 추정하여 이를 로봇 손으로 전이하는 방법을 제시하였으며, 시뮬레이션을 통한 강화 학습을 사용하여 동적인 그립 제어를 학습합니다.

- **Technical Details**: FunGrasp 시스템은 정적 기능적 그립 전이(static Functional Grasp Retargeting), 동적 정교한 그립(dynamic Dexterous Grasping), 시뮬레이션-실제 전이(sim-to-real Transfer)라는 세 단계로 구성되어 있습니다. 이 시스템은 촉각 및 시각적 인식을 활용하여 다양한 객체의 모양을 포착하고, 현실 세계에서의 그립 제어를 위한 정책을 훈련합니다. 또한, 시스템 식별(System Identification) 및 중력 보상(Gravity Compensation)과 같은 여러 가지 기법을 활용하여 실제 세계로의 효과적인 전이를 달성합니다.

- **Performance Highlights**: 실험 결과, FunGrasp 시스템이 단일 RGBD 이미지로부터 이전에 본 적 없는 객체들에 대해 기능적 정교한 그립을 성공적으로 수행할 수 있음을 입증하였습니다. 다양한 정교한 로봇 손 모델 간의 일반화 능력을 평가하였고, 성능 구성 요소들에 대한 상세한 아블레이션 연구가 실시되었습니다. 본 연구는 로봇이 실제 세계에서 다양한 작업을 지원할 수 있는 가능성을 열어줍니다.



### Sonic: Shifting Focus to Global Audio Perception in Portrait Animation (https://arxiv.org/abs/2411.16331)
Comments:
          refer to our main-page \url{this https URL}

- **What's New**: 이번 연구에서는 생동감 있는 대화하는 얼굴 애니메이션을 생성하기 위한 새로운 패러다임이 제안된다. 기존 접근 방식들이 음성과 시각적 요소 간의 조화를 간과한 반면, 본 연구에서는 오디오 신호를 통해 자연스럽고 시공간적으로 일관된 애니메이션을 목표로 한다. 'Sonic'이라는 이름으로 불리는 이 패러다임은 글로벌 오디오 지각(Global Audio Perception)을 탐색하고자 하며, 오디오 기반 애니메이션의 본질을 강조한다.

- **Technical Details**: Sonic 패러다임은 크게 두 가지로 구성된다: 1) Context-enhanced audio learning, 이는 입력된 오디오 클립에서 장기적인 오디오 정보를 추출하는 모듈이며, 이러한 정보를 기반으로 얼굴 표정과 입 모양의 사전 정보를 제공한다. 2) Motion-decoupled controller, 이는 표정과 머리의 동작을 분리하여 독립적으로 제어할 수 있도록 하는 시스템이다. 추가적으로, 타임 어웨어 포지션 시프트 퓨전(Time-aware Position Shift Fusion) 기법을 통해 클립 간의 오디오 지각을 연결하고 글로벌 수준의 데이터를 융합한다.

- **Performance Highlights**: 본 연구는 기존의 SOTA(State-of-the-Art) 기법보다 비디오 품질, 시간적 일관성, 입술 동기화 정밀성 및 동작 다양성에서 우수한 성능을 보이고 있다. 실험 결과, Sonic을 적용할 경우 잠재적 짜임새와 리얼리즘이 크게 향상된다는 것이 입증되었다. 이는 오디오 신호의 형식적 특성을 통해 더욱 자연스러운 애니메이션을 가능하게 하며, 다양한 응용 프로그램에서 유용하게 활용될 수 있을 것으로 기대된다.



### Generative Omnimatte: Learning to Decompose Video into Layers (https://arxiv.org/abs/2411.16683)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 비디오와 입력 객체 마스크 집합을 기반으로 개별 객체와 그림자, 반사와 같은 관련 효과가 포함된 의미 있는 레이어로 비디오를 분해하는 새로운 생성 기반 오믹 매트(omnimatte) 방법을 제안합니다. 기존 방법들은 정적 배경이나 정확한 카메라 포즈 및 깊이 추정을 가정하고 있어, 이러한 가정이 위배되면 성능이 저하되는 문제가 있었습니다. 본 연구에서는 이러한 한계를 극복하기 위해 사전 훈련된 비디오 확산 모델(video diffusion model)을 활용하여 동적 차폐(occlusion)된 영역을 완전하게 처리할 수 있는 프레임워크를 개발했습니다.

- **Technical Details**: 제안된 방법은 RGB 비디오와 객체 마스크를 입력으로 받아들이며, N개의 레이어와 깨끗한 RGB 배경 레이어를 생성하는 것을 목표로 합니다. 이 모델은 비디오 확산 모델을 활용하여, 사물과 그에 따른 효과 간의 관계를 이해하고, 그에 따라 오믹 매트를 생성합니다. 특정 객체에 의해 발생하는 장면 효과를 식별하고 제거하는 훈련이 이루어지며, 반투명하며 의미 있는 레이어를 생성하는 데 중점을 둡니다.

- **Performance Highlights**: 이 시스템은 다양한 자연 비디오에 대해 높은 품질의 분해 및 편집 결과를 보여주며, 부드러운 그림자, 광택 있는 반사, 튀는 물 등의 효과를 잘 처리할 수 있습니다. 최종 쿼브라 전방 레이어는 입력 비디오를 재구성하면서도 희소성을 유지하여, 객체 제거, 움직임 재조정, 전경 스타일화와 같은 다양한 창의적 편집 작업을 가능하게 합니다. 연구팀은 이 모델을 세밀하게 조정하여, 실제 및 합성 예시 데이터셋을 통해 오브젝트-효과 연관성을 학습하고 최적의 결과를 달성했습니다.



### Factorized Visual Tokenization and Generation (https://arxiv.org/abs/2411.16681)
- **What's New**: 본 연구에서는 Factorized Quantization (FQ)라는 새로운 접근 방식을 도입하여 VQ 기반 토크나이저를 재활성화합니다. 기존의 코드북(codebook)을 여러 개의 독립적인 서브 코드북(sub-codebook)으로 분해하여 저장소 요구사항을 줄이고 효율적인 비주얼 토크나이제이션을 가능하게 합니다. 이를 통해 이미지 생성 품질을 향상시키고, 다운스트림(auto-regressive) 생성 모델에 통합할 수 있는 능력을 강화합니다.

- **Technical Details**: FQ는 코드북을 여러 개의 독립적인 서브 코드북으로 나누는 설계로, 각 서브 코드북이 시각적 데이터의 고유한 측면을 포착하도록 만들기 위해 분리 정규화(disentanglement regularization) 기법을 도입합니다. 이 정규화 기법은 서브 코드북 간의 중복성을 줄이고 다양성을 촉진하여 혼합된 대표성 공간을 지원합니다. 또한 사전 훈련된 비전 모델인 CLIP과 DINO를 활용하여 의미론적인 풍부함을 학습하는 과정은 각 서브 코드북이 저수준 구조부터 고수준 개념까지 다양한 세부 정보를 포착할 수 있도록 돕습니다.

- **Performance Highlights**: FQGAN 모델은 비주얼 토크나이저의 재구성 품질을 크게 향상시켜 state-of-the-art 성능을 달성했습니다. 연구 팀은 또한 이 토크나이저가 다운스트림 이미지 생성에서 유효하게 조정될 수 있다는 점을 입증해 ImageNet 벤치마크에서 이미지 생성 품질을 개선하는 데 기여했습니다. 전반적으로, 이 연구는 VQ 기반 모델의 한계를 극복하고 다양한 의미론적 층을 포착하는 데 성공하여 비주얼 생성의 가능성을 넓힙니다.



### Quark: Real-time, High-resolution, and General Neural View Synthesis (https://arxiv.org/abs/2411.16680)
Comments:
          SIGGRAPH Asia 2024 camera ready version; project page this https URL

- **What's New**: 이 논문에서는 고해상도 실시간 뷰 합성(view synthesis)을 위한 새로운 신경 알고리즘을 소개합니다. 이 네트워크는 RGB 이미지 또는 비디오 스트림의 희소(set) 집합을 통해 3D 장면을 재구성하고, 이를 바탕으로 1080p 해상도에서 30fps로 새로운 뷰를 렌더링할 수 있습니다. 본 연구 결과, 다른 오프라인 방식의 품질을 초과하는 경우도 나타났습니다.

- **Technical Details**: 이 알고리즘은 반투명 층(semi-transparent layers)을 사용하여 장면을 나타내고, 반복적 학습 렌더 및 정제(iterative learned render-and-refine) 접근 방식을 통합합니다. 각 프레임마다 새로운 LDM(Layered Depth Maps)을 재구성, 렌더링 및 삭제하며, 이를 통해 복잡한 깊이 및 폐색을 효과적으로 대표합니다. 업데이트 과정에서는 Transformer 기반 네트워크 컴포넌트를 활용해 여러 입력 뷰의 정보를 효율적으로 집계합니다.

- **Performance Highlights**: 실험 결과, 본 알고리즘은 실제율(real-time) 기준에서 최첨단 품질을 달성하였으며, 8개의 입력 이미지로 1080p 해상도에서 30fps로 작동합니다. 성능 벤치마크를 통해 이 알고리즘은 다양한 장면에서 높은 품질을 유지하며, 고속 렌더링을 요구하는 응용프로그램에 적합한 솔루션으로 자리매김할 것입니다.



### Diffusion Features for Zero-Shot 6DoF Object Pose Estimation (https://arxiv.org/abs/2411.16668)
- **What's New**: 이 논문은 Latent Diffusion Model (LDM)을 기반으로 한 zero-shot 객체 자세 추정 방법을 제안합니다. 기존 연구와의 차별점은 LDM을 사용하여 비정형 데이터에서 자세를 추정하는 새로운 접근법을 마련한 것입니다. 이를 통해 Vision Transformer (ViT) 모델에 비해 최대 27% 향상된 정확도를 제공함을 입증하였습니다. 또한, 이 연구는 LDM 프로세스를 통해 객체 자세 추정의 효율성을 개선하는 방법에 대한 통찰을 제공합니다.

- **Technical Details**: 제안된 방법은 RGB 이미지와 객체 우선 정보를 통해 6DoF 자세를 추정하는 메커니즘을 포함합니다. LDM의 특징을 추출하는 데(Stable Diffusion을 활용하여) ViT를 대체하고, 특징 공간을 정렬하여 강력한 대응 군집화를 완료하게 됩니다. NAA-Perspective-n-Points 알고리즘을 사용하여 객체의 자세를 계산하고, 이는 강인한 대응 추정이 가능하게 합니다. 이 연구에서는 LMO, YCBV, TLESS와 같은 세 가지 표준 데이터셋에서 실험을 통해 이 방법을 평가했습니다.

- **Performance Highlights**: 실험 결과는 LDM을 활용한 방법이 ViT에 비해 모든 데이터셋에서 객체 자세 추정의 정확도를 향상시키며, 최대 27%의 상대적 개선을 보여주었습니다. 구체적으로 LDM 기반 접근 방식은 다양한 도전 요소에 대해 향상된 성능을 입증하였고, 객체의 저해상도 이미지를 사용할 때의 안정성 또한 강화되었습니다. 이는 sub-pixel 정확도의 대응 추정 방식을 통해 이루어졌으며, 저비율 객체 이미지에서의 자세 추정의 안정성을 높였습니다.



### Edge Weight Prediction For Category-Agnostic Pose Estimation (https://arxiv.org/abs/2411.16665)
- **What's New**: EdgeCape는 Category-Agnostic Pose Estimation (CAPE) 분야에서 새로운 접근 방식을 제안합니다. 이 모델은 키포인트(localization of keypoints) 예측 시, 정적(graph) 포즈 그래프 대신 가변적인 엣지 가중치를 예측함으로써 더 나은 성능을 보여줍니다. 특히, Markovian Structural Bias를 통합하여 노드 간의 상호작용을 조절하며, 전역 공간 의존성을 포착하는 능력을 향상시킵니다.

- **Technical Details**: EdgeCape는 포즈 추정 모델의 핵심인 가중치를 예측하는 새로운 메커니즘을 도입합니다. 이 과정에서 사용자 주도의 그래프(prior graphs)를 활용하여 엣지 가중치를 할당하고, 이를 통해 더 복잡한 형태의 객체 기하학을 처리할 수 있는 구조 인식 스켈레톤을 생성합니다. 또한, Graph Neural Networks를 활용하여 Self-Attention 메커니즘에 그래프 구조를 통합함으로써, 키포인트 간의 복합적인 공간적 관계를 포착합니다.

- **Performance Highlights**: MP-100 기준 데이터셋을 기준으로 한 평가에서, EdgeCape는 1-shot 및 5-shot 설정 모두에서 기존의 최첨단 모델을 초월합니다. 본 연구는 또한 다양한 조건에서의 견고성 및 복잡한 범주 간 키포인트 매칭 문제에서의 성능을 입증하였습니다. 이를 통해 EdgeCape는 키포인트 위치 추정 정확도를 크게 개선하며 CAPE 분야에서 중요한 기여를 하고 있습니다.



### DreamRunner: Fine-Grained Storytelling Video Generation with Retrieval-Augmented Motion Adaptation (https://arxiv.org/abs/2411.16657)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 DreamRunner라는 새로운 스토리 비디오 생성 방법을 제안합니다. 이 방법은 길고 멀티 모션, 멀티 씬 비디오를 생성할 수 있으며, 입력 텍스트 스크립트에서 설명된 스토리를 일관되게 표현하는데 중점을 둡니다. DreamRunner는 LLM(대규모 언어 모델)을 사용해 입력 스크립트를 구조화하고, 리트리벌(검색) 기반의 적응 방법을 통해 비디오 생성 과정에서 동작을 커스터마이즈(purpose)할 수 있습니다.

- **Technical Details**: DreamRunner 프레임워크는 세 가지 주요 프로세스로 구성됩니다: 1) 이중 수준 비디오 계획 생성, 2) 동작 리트리벌 및 주제/동작 사전 학습, 3) 공간-시간 기반의 3D 주의 및 사전 주입 모듈(SR3AI). 첫 번째 단계에서는 사용자 제공의 스토리 내러티브를 기반으로 고수준 및 세부적인 계획을 수립하며, 두 번째 단계에서는 비디오 데이터베이스에서 동작과 관련된 비디오를 검색하여 동작 사전을 학습합니다. 마지막 단계에서는 상세한 프레임 제어와 원활한 동작 전환을 가능하게 하는 SR3AI 모듈을 도입합니다.

- **Performance Highlights**: DreamRunner는 T2V-ComBench에서 기존의 최첨단 방식들에 비해 캐릭터 일관성(CLIP 점수)에서 13.1%의 상대적 개선을 보이고, 텍스트 추적 능력(ViCLIP 점수)에서도 8.56%의 향상을 기록했습니다. 또한, 동일 씬 내에서의 사건 전환의 매끄러움(DINO 점수)에서도 27.2% 개선을 보이며 효과성을 입증했습니다. DreamRunner는 오픈 소스 모델을 기반으로 하면서도 폐쇄형 모델에 비해 동적 속성 바인딩에서 최고의 성과를 달성했고, 캐릭터 상호작용에서도 경쟁력 있는 결과를 보여주며 오픈 소스의 가능성을 보여주고 있습니다.



### Imperceptible Adversarial Examples in the Physical World (https://arxiv.org/abs/2411.16622)
- **What's New**: 이번 연구는 Deep Neural Network (DNN) 기반의 컴퓨터 비전 모델에 대한 공격에서 물리적인 환경에서도 식별할 수 없는 적대적 예제(adversarial examples)를 생성하는 새로운 접근 방식을 제안합니다. 이 연구는 Straight-through Estimator (STE) 기술을 활용하여 비분화 노이즈(non-differentiable distortions)가 존재하는 실제 환경에서도 효과적인 공격을 가능하게 합니다. 이로 인해 과거의 공격 방식들보다 더 강력하고 유의미한 적대적 공격이 가능해졌습니다.

- **Technical Details**: 연구지는 STE를 사용하여 비분화 왜곡 방식을 갖는 비주얼 센싱 시스템에 대한 적대적 예제를 생성하는 과정을 설명합니다. 이 기술은 후방 전파(backpropagation) 과정에서 정체성 함수(identity function)와 차별화 렌더링(differentiable rendering)을 결합하여 사용함으로써 물리적 환경에서도 눈에 띄지 않는 공격을 가능하게 합니다. 이를 통해 인쇄된 사진이나 CARLA 시뮬레이터에서의 실험을 통해 $ℓ_	ext{∞}$으로 제한된 적대적 예제를 신속하게 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: STE 기술을 활용한 본 연구의 성과는 실제 환경에서 적대적 공격의 효과가 디지털 환경의 전통적인 공격과 유사하거나 더욱 우수하다는 것입니다. 연구에서는 AP50의 정확도가 43.29%에서 4.22%로 감소하는 등, 물리적 공간에서 비가시적인 적대적 패치가 어떻게 타겟 모델의 분류 정확도를 제로로 만드는지에 대한 실험 결과를 제시합니다. 이러한 성과는 DNN 기반의 비주얼 센싱 시스템의 보안 위험성을 재평가할 필요성을 강조합니다.



### Human-Activity AGV Quality Assessment: A Benchmark Dataset and an Objective Evaluation Metric (https://arxiv.org/abs/2411.16619)
- **What's New**: 최근 AI-driven 비디오 생성 기술은 중요한 진전을 보였습니다. 그러나 AI가 생성한 비디오(AGVs)에서 인간 행동을 포함할 때, 시각적 및 의미적 왜곡이 두드러져 이 기술의 실제 응용을 방해하고 있습니다. 이 논문은 인간 활동 AGV의 품질 평가를 위한 최초의 데이터셋인 Human-AGVQA를 구축하고 AGV의 시각적 품질 및 의미적 왜곡을 연구합니다.

- **Technical Details**: Human-AGVQA 데이터셋은 8개의 인기 있는 텍스트-비디오(T2V) 모델로부터 3200개의 AGV와 400개의 텍스트 프롬프트로 구성되어 있습니다. GHVQ라는 새로운 객관적 평가 메트릭을 통해 AGV의 품질을 자동으로 분석하며, 이 메트릭은 인간 중심의 품질 특징과 시간적 연속성 특징을 체계적으로 추출합니다. GHVQ는 여러 평가 차원에서 기존의 품질 메트릭스를 큰 차이로 능가하여 효과성을 입증하였습니다.

- **Performance Highlights**: GHVQ는 Human-AGVQA 데이터셋에서 기존의 메트릭스를 초월하는 성능을 보여주었습니다. 연구 결과, 이 메트릭은 인간 대상 AGV의 품질을 평가하는 데 있어 포괄적이고 설명 가능한 형식을 제공합니다. 본 연구는 T2V 모델의 강점과 약점을 분석할 수 있는 기초를 마련하고, AGV의 품질 문제를 해결하는 데 기여할 것으로 기대됩니다.



### GeoFormer: A Multi-Polygon Segmentation Transformer (https://arxiv.org/abs/2411.16616)
Comments:
          21 pages, 5 figures, in proceedings of British Machine Vision Conference 2024

- **What's New**: 본 논문은 건축물 데이터의 벡터화 문제를 해결하기 위해 GeoFormer라는 새로운 구조를 제안합니다. 기존 방법들은 여러 손실 함수의 조정이 필요했지만, GeoFormer는 단일 likelihood 함수에 의존하여 다중 폴리곤을 생성하는 데 성공적으로 적용되었습니다. 특히, 이는 원거리 감지(remote sensing) 영역에서 오토 리그레시브(transformer) 모델이 다중 폴리곤 예측에 성공적으로 활용된 첫 사례입니다.

- **Technical Details**: GeoFormer는 인코더-디코더 아키텍처로, 공간적으로 의존하는 토큰들을 순차적으로 학습합니다. 이 모델은 기존 시스템보다 훨씬 이미지를 효과적으로 활용하여 다중 건물 폴리곤을 생성하도록 설계되었습니다. 또한, GeoFormer는 Aicrowd Mapping Challenge의 벤치마크 데이터셋에서 과거의 방법들을 능가하는 성능을 보였습니다.

- **Performance Highlights**: GeoFormer는 최종적으로 직접 사용할 수 있는 벡터 형식의 출력을 생성하며, 이전 연구에 비해 성능이 크게 향상되었습니다. 이 모델은 하이퍼파라미터 조정이나 후처리 과정을 최소화하며, 건물의 기하학적 특성에 최소한의 점 집합을 학습합니다. 이는 기존 방법들이 주로 세멘틱 세그멘테이션에 의존하는 것과는 차별화됩니다.



### Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models (https://arxiv.org/abs/2411.16602)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 Chat2SVG라는 새로운 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)과 이미지 확산 모델의 장점을 결합하여 텍스트에서 SVG 생성의 접근성을 향상시키고 있습니다. 기존의 방법들이 세부적인 기하학적 요소를 생성하는 데 어려움을 겪는 반면, Chat2SVG는 SVG 템플릿을 생성하고 세부 조정을 통해 전문적인 벡터 그래픽을 더 쉽게 만들 수 있도록 합니다.

- **Technical Details**: Chat2SVG는 두 가지 주요 단계로 구성되어 있습니다. 첫 번째 단계에서는 LLM을 사용하여 기본 기하학적 프리미티브로 구성된 의미론적으로 일관된 SVG 템플릿을 생성합니다. 그런 다음 이미지 확산 모델에 의해 안내되는 두 번째 단계에서 경로와 좌표를 세밀하게 조정하여 복잡한 기하학적 디테일을 강화합니다.

- **Performance Highlights**: 실험 결과, Chat2SVG는 기존의 방법들에 비해 시각적 충실도, 경로의 일관성, 의미의 정합성 면에서 뛰어난 성능을 보였습니다. 또한, 사용자는 자연어 지침을 통해 직관적인 편집이 가능하여 비전문가도 접근할 수 있는 벡터 그래픽 생성 환경을 제공합니다.



### Rethinking Diffusion for Text-Driven Human Motion Generation (https://arxiv.org/abs/2411.16575)
Comments:
          Preprint

- **What's New**: 본 논문에서는 인간 동작 생성을 위한 새로운 확산 모델을 제안하고 VQ 기반 방법의 장점과 확산 기반 방법의 한계를 분석합니다. 기존의 Vector Quantization (VQ) 방식에서 얻은 인사이트를 활용하여 동작 데이터의 표현을 최적화함으로써 성능 개선을 꾀합니다. 특히, 새로운_bidirectional masked autoregressive 방식을 도입하여 인간 동작 생성의 새로운 기준을 세웁니다.

- **Technical Details**: 논문에서는 VQ 기반의 기계학습 기법과 확산 기반 기법의 동작 데이터 표현 및 분포에서의 장단점을 체계적으로 분석합니다. VQ-VAE 및 RVQ-VAE와 같은 구조를 사용하여 동작 데이터를 해석하는 방법과 함께, 확산 기반 모델은 데이터를 가우시안 노이즈와 결합하여 생성하는 방법에 대해 설명합니다. 이러한 기법을 통해 더 나은 모델 훈련 및 평가 방법을 체계적으로 제시합니다.

- **Performance Highlights**: 제안된 방법은 KIT-ML 및 HumanML3D 데이터셋에서 이전의 방법들보다 뛰어난 성능을 보이며, 최신의 성과로 평가됩니다. 본 연구는 VQ 기반 방법에서 발생하는 한계를 해결하기 위해 확산 기반 모델을 최적화하는 접근을 통해, 동작 생성 태스크에서의 새로운 기준을 제시합니다. 더불어 보다 신뢰할 수 있는 평가 방법을 제안하여 다양한 방법 간의 공정한 비교가 가능하게 합니다.



### J-CaPA : Joint Channel and Pyramid Attention Improves Medical Image Segmentation (https://arxiv.org/abs/2411.16568)
- **What's New**: 이번 연구에서는 기존 CNN 기반 모델의 한계를 극복하기 위해 Channel Attention과 Pyramid Attention 메커니즘을 결합한 Transformer 기반 아키텍처인 J-CaPA를 제안합니다. 이는 다중 스케일 특징 추출을 개선하고 의학적 이미지 분할 성능을 높이는 데 기여합니다. CutMix 데이터 증강 기법을 사용하여 모델의 일반화를 강화하며, Synapse 멀티 장기 분할 데이터셋에서 우수한 성과를 보였습니다.

- **Technical Details**: J-CaPA는 변환기 기반 구조로, 수정된 ResNetV2 백본을 활용한 인코더-디코더 디자인을 따릅니다. Channel Attention Module (CAM)과 Pyramid Attention Module을 도입함으로써 지역 및 글로벌 컨텍스트를 향상시켜 세그멘테이션의 정확도를 높였습니다. 이러한 모듈들은 서로 다른 스케일에서 주의 메커니즘을 적용하여 잔여적 특성을 추출하고, 최종 세그멘테이션 맵을 복원하는 역할을 맡고 있습니다.

- **Performance Highlights**: 제안된 모델은 Synapse 다중 장기 분할 데이터셋에서 평균 Dice 점수가 6.9% 향상되었고, Hausdorff Distance (HD95)가 39.9% 감소했습니다. 특히 담낭, 신장, 췌장과 같은 복잡한 해부학적 구조의 분할에서 뚜렷한 개선을 보여주며, 기존의 최첨단 방법들을 초월하는 성능을 입증했습니다. 이러한 성과는 연구에서 Joint Attention 메커니즘을 활용함으로써 얻어진 결과입니다.



### RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics (https://arxiv.org/abs/2411.16537)
- **What's New**: 이 논문은 로봇 분야에서의 공간 이해 능력 향상을 위해 RoboSpatial이라는 대규모 데이터셋을 소개합니다. 이 데이터셋은 3D 스캔과 에고세닉(ego-centric) 이미지로 구성된 실내 및 테이블 씬을 기반으로 하며, 로봇과 관련된 풍부한 공간 정보를 포함하고 있습니다. RoboSpatial은 100만 개의 이미지, 5,000개의 3D 스캔, 300만 개의 주석 공간 관계를 포함하여 로봇이 더 잘 이해할 수 있도록 다양한 질문-답변 쌍을 제공합니다.

- **Technical Details**: RoboSpatial 데이터셋은 세 가지 유형의 질문을 제공합니다: (1) 공간 구성, (2) 공간 맥락, (3) 공간 호환성. 각 질문 유형은 서로 다른 관점에서 공간 관계를 이해하도록 설계되어 있으며, 관점은 에고세닉, 객체 중심(object-centric), 세계 중심(world-centric)으로 나뉩니다. 이러한 다각적인 접근 방식은 로봇이 복잡한 공간 지침을 더욱 유연하게 처리할 수 있도록 합니다.

- **Performance Highlights**: RoboSpatial을 사용하여 훈련된 비전-언어 모델(VLM)은 기존 모델보다 공간 추론 능력이 상당히 향상되었습니다. 실험 결과, RoboSpatial에서 학습한 모델은 다양한 로봇 조작 작업 및 실내 씬 질문 응답에서 우수한 성능을 보였습니다. 이 데이터셋의 3D 준비 디자인은 VLM의 공간 추론 능력을 높이는 데 기여하며, 실제 공간 작업에서의 차별적인 성능을 보여줍니다.



### LaB-RAG: Label Boosted Retrieval Augmented Generation for Radiology Report Generation (https://arxiv.org/abs/2411.16523)
- **What's New**: 이번 논문에서는 LaB-RAG(Label Boosted Retrieval Augmented Generation)라는 새로운 이미지 캡셔닝 방법을 제안했습니다. 이 방법은 고차원 latent features를 대신하여 카테고리 레이블을 사용하여 이미지 설명을 반영함으로써, 표준 retrieval augmented generation을 향상시키는 데 초점을 맞추고 있습니다. 특히, 이 접근법은 의료 영상의 보고서 생성을 위한 연구에 중점을 두고 있습니다.

- **Technical Details**: LaB-RAG는 간단한 선형 분류기를 사용하여 이미지에서 카테고리 레이블을 추출하고, 이를 기반으로 일반적인 LLM(여기서는 대규모 언어 모델)과 결합하여 라디오로지 보고서를 생성합니다. 이 방법은 고급 딥러닝 모델의 훈련 없이, 기존 모델의 frozen 상태에서 작동합니다. 따라서, LaB-RAG는 어떤 이미지 피처 인코더 모델을 훈련시키지 않으면서도 경쟁력 있는 성과를 나타냅니다.

- **Performance Highlights**: LaB-RAG는 기존의 retrieval 기반 RRG 방법들보다 더 나은 성능을 보여주었고, 다른 세밀하게 조정된 시각-언어 RRG 모델들과도 비교할 만한 결과를 얻었습니다. 또한, 다양한 구성 요소에 대한 실험을 통해 이 방법의 효과를 분석했습니다. 최종적으로, 일반적으로 사용되는 RRG 메트릭의 문제점을 비판하면서, LaB-RAG를 사용하여 결과를 인위적으로 부풀릴 수 있는 가능성을 언급했습니다.



### All Languages Matter: Evaluating LMMs on Culturally Diverse 100 Languages (https://arxiv.org/abs/2411.16508)
Comments:
          A Multilingual Multimodal cultural benchmark for 100 languages

- **What's New**: 이번 연구에서 발표된 All Languages Matter Benchmark (ALM-bench)는 100개 언어에 걸쳐 대규모 다중모달 모델(LMM)을 평가하기 위한 포괄적이고 문화적으로 다양한 벤치마크입니다. ALM-bench는 기존 모델들이 다양한 문화적 맥락을 이해하는 능력을 시험하며, 부족 자원 언어(low-resource languages)에 대한 평가를 포함합니다. 이 벤치마크는 정교한 질문 형식을 제공하여 다양한 시각적 및 언어적 추론에서 모델의 능력을 검증합니다.

- **Technical Details**: ALM-bench는 22,763개의 질문 및 답변을 포함하며, 19개의 일반 및 문화 특정 도메인에서 다양한 유형의 질문을 제공합니다. 이 데이터셋은 73개국의 문화적 뉘앙스를 포착하며, 15개의 언어 계통과 24개의 언어 스크립트를 포함하고 있습니다. 기존 벤치마크들과 비교할 때 ALM-bench는 3배 더 많은 언어와 다양한 질문 형식을 제공하여 다문화 및 다언어 환경에서 LMM의 성능을 종합적으로 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 16개의 최신 LMM을 대상으로 한 평가 결과, 그들이 부족 자원 언어 및 문화에 대한 이해에서 상당한 성능 격차가 있음을 보여주었습니다. 특히, GPT-4o는 최고의 오픈소스 모델인 GLM-4V보다 27% 더 뛰어난 성능을 보였습니다. ALM-bench를 통해 저자원 지역인 동남아시아 및 서부 아프리카의 이해를 개선할 필요성을 강조했습니다.



### Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis (https://arxiv.org/abs/2411.16503)
- **What's New**: 이 논문에서는 초기의 노이즈가 있는 잠재 변수를 최적화하여 생성된 이미지와 입력 프롬프트 간의 의미적 일치를 개선하는 새로운 방법론을 제안합니다. 기존의 InitNo 접근 방식은 주의 맵(attention maps)에 의존하여 초기 잠재 변수를 정제하였지만, 이 방법은 한정된 정보만을 캡처하므로 국소 최적점(local optimum)에서 수렴하는 경향이 있었습니다. 이 연구에서는 대형 비전-언어 모델(LVLM)을 활용하여 초기 잠재 변수를 최적화하는 Noise Diffusion 프로세스를 소개합니다.

- **Technical Details**: Noise Diffusion 프로세스는 원래의 잠재 변수에 점진적으로 가우시안 노이즈를 추가하여 더욱 의미적으로 신뢰할 수 있는 이미지를 생성하는 방식입니다. 이 과정에서는 VQA(Visual Question Answering) 점수를 최대화하는 방향으로 잠재 변수를 최적화하며, 기존 방법들이 사용할 수 없는 방식으로梯度 정보(gradient information)를 활용하여 가장 적절한 단계 차이를 선택합니다. 또한 이론적으로 단계 차이의 내적과 gradient의 비율이 특정 임계값을 초과할 때, 최적화가 VQA 점수를 증가시킬 것임을 증명합니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 다양한 확산 모델에서 의미적 일치를 일관되게 향상시킬 수 있음을 보여줍니다. 특히, 기존의 Stable Diffusion 모델이 프롬프트와 일치하는 이미지를 생성하지 못했을 때, 우리의 방법을 통해 생성된 이미지가 프롬프트와의 정렬을 개선할 수 있음을 확인할 수 있었습니다. 이 연구는 LVLM의 의미적 이해 능력을 활용한 생성 과정 감독 및 노이즈 확산 방법의 도입을 통해 이미지 생성의 품질을 크게 향상시킬 수 있음을 입증하였습니다.



### Multi-Resolution Generative Modeling of Human Motion from Limited Data (https://arxiv.org/abs/2411.16498)
Comments:
          1O pages, 7 figures, published in European Conference on Visual Media Production CVMP 24

- **What's New**: 본 논문에서는 제한된 훈련 시퀀스에서 인간의 동작을 합성하는 생성 모델을 제안합니다. 이 프레임워크는 여러 시간 해상도에서 조건부 생성을 제공하고, 인체의 동작 패턴을 잘 포착합니다. 특히, 우리의 접근 방식은 단어와 연관된 손짓(gesture) 생성에도 적용되어, 제한된 데이터에서도 동기화된 손짓을 생성할 수 있는 능력을 보입니다.

- **Technical Details**: 우리는 SMPL 모델을 사용하여 동작 신호에 따라 사람의 메쉬를 애니메이션화하는 생성 모델을 학습합니다. 각 프레임에 대해 SMPL의 포즈 파라미터와 형태 파라미터를 입력받아 메쉬를 생산하며, 이 과정에서 발의 접촉 아티팩트를 방지하기 위해 발의 접촉 여부를 이진 레이블로 모델링합니다. 또한, 다중 클래스 주석 또는 신호에서 추출한 피처를 기반으로 하는 임베딩을 통해 다양한 시간 척도에서 동작의 감정이나 행동을 제어할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 광범위한 훈련 예제를 포괄하면서도 다양한 동작을 생성할 수 있음을 보여줍니다. 생성된 동작은 지역 및 전역 다양성 지표에 의해 다양한 모션 패턴을 나타내며, 이는 제한된 동작 시퀀스에서도 높은 다양성과 품질을 보장함을 의미합니다. 제안된 방법은 기존 모션 합성 작업에 비해 통제할 수 있는 유연성을 제공하여 실제 애니메이션 응용 프로그램에 실용적입니다.



### Deformable Mamba for Wide Field of View Segmentation (https://arxiv.org/abs/2411.16481)
Comments:
          Models and code will be made publicly available at: this https URL

- **What's New**: 이번 연구에서는 Deformable Mamba라는 새로운 프레임워크를 제안하여, 판상(panoramic) 및 어안(fisheye) 이미지를 분석할 때 발생하는 왜곡(distortion) 문제를 해결하고자 합니다. 기존의 MAMBA 모델은 왜곡을 고려하지 않았기 때문에 성능이 저하되었습니다. Deformable Mamba는 Deformable Mamba Fusion (DMF) 블록을 중심으로 구성된 디코더(decoder)를 통해 이러한 왜곡을 보다 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: Deformable Mamba는 다양한 넓은 시야각(FoV)에서의 이미지 분할을 위한 통합된 솔루션입니다. 이 연구는 180°에서 360°까지의 다양한 카메라 타입과 소스를 다루며, 새롭고 유연한 디코더를 기반으로 하고 있습니다. 이를 통해 CNN 기반, 변환기(transformer) 기반 및 Mamba 기반 아키텍처와의 원활한 통합이 가능합니다.

- **Performance Highlights**: Deformable Mamba는 여러 데이터세트에서 우수한 성능을 발휘하여, 특히 360° Stanford2D3D 데이터 세트에서 +2.5%의 성능 향상을 기록했습니다. 또한, 180° SynWoodScape 데이터 세트에서는 이전 모델에 비해 +2.2%의 향상을 이루었습니다. 이러한 결과는 Deformable Mamba가 넓은 시야각을 가진 다양한 센서 타입에서 밀집 이미지 분석의 일반적인 솔루션으로 자리잡을 가능성을 보여줍니다.



### Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency (https://arxiv.org/abs/2411.16468)
- **What's New**: 본 연구는 효율적인 블라인드 비디오 얼굴 향상 기법을 제안하여 손상된 저화질 비디오에서 고화질 비디오로 복원하는 방법을 제공합니다. 기존 동영상 얼굴 향상 기법의 프로세싱 시간이 길고 공간-시간적 비주얼 효과가 일관되지 않다는 문제를 해결합니다. 새로운 3D-VQGAN 백본을 활용하여 높은 품질의 인물 특징과 잔여 기반 시간 정보를 기록하는 공간-시간 코드북을 개발하였습니다.

- **Technical Details**: 제안된 방법은 두 단계의 학습 프레임워크로 구성되어 있으며, 첫 번째 단계에서는 규제기를 통해 코드북 붕괴 문제를 완화하여 학습합니다. 두 번째 단계에서는 각각 코드북에서 코드를 검색하고 저화질 비디오 인코더를 업데이트하기 위해 두 개의 트랜스포머를 학습합니다. 이 프레임워크는 BMFR 작업에서 블러, 노이즈 등의 손상을 제거하고 실제 비디오 및 AI 생성 비디오에서 밝기 깜빡임을 제거할 수 있습니다.

- **Performance Highlights**: VFHQ-Test 데이터세트에서 수행된 실험 결과, 제안된 방법이 현 상태의 최첨단 블라인드 얼굴 비디오 복원 및 디플리커링 기법보다 효율성과 유효성에서 뛰어난 성과를 보였습니다. 전체 실험 결과는 제안된 방법의 효과성을 강하게 입증하고 있습니다.



### No Identity, no problem: Motion through detection for people tracking (https://arxiv.org/abs/2411.16466)
Comments:
          Accepted in TMLR November 2024

- **What's New**: 이번 논문에서는 사람 추적을 위한 새로운 접근 방식을 제안합니다. 기존의 tracking-by-detection 방식을 개선하기 위해, 정체성 주석 없이도 검출에 대한 감시를 제공함으로써 모션 정보를 활용하였습니다. 이를 통해 2D 모션 추정과 함께 두 가지 감지 히트맵을 예측해, 서로 일관성을 유지하는 시스템을 구축했습니다.

- **Technical Details**: 제안된 알고리즘은 시간 t와 t+1에서의 감지 히트맵과 2D 모션 오프셋 맵을 예측합니다. 이 오프셋 맵을 통해 첫 번째 히트맵을 변형하여 두 번째 히트맵과의 일관성을 부여합니다. 이 과정에서 특정한 모션 주석 없이도 감시 신호를 효율적으로 추출할 수 있습니다. 이러한 접근법은 훈련 과정에서 얻어진 다양한 이미지 정보를 결합하여 정확도를 높이는 데 기여합니다.

- **Performance Highlights**: 이 모델은 MOT17 및 WILDTRACK 데이터셋에서 유의미한 성과를 보여줍니다. MOT17에서는 기존의 최첨단 추적 방법인 Bytetrack을 낮은 프레임 속도 상황에서 크게 초과하는 성능을 달성했습니다. 또한 WILDTRACK에서도 최근의 다중 보기 탐지 및 추적 기술에 비해 우수한 결과를 보였습니다.



### VQ-SGen: A Vector Quantized Stroke Representation for Sketch Generation (https://arxiv.org/abs/2411.16446)
- **What's New**: 논문에서는 VQ-SGen이라는 새로운 알고리즘을 제안하여 고품질의 스케치 생성을 목표로 하고 있습니다. 기존의 방법들은 스케치의 각 선이 가지는 상관성과 위치 관계를 간과했으며, 픽셀 기반으로 전체 또는 부분으로 생성하는 방식이었습니다. VQ-SGen은 각 선을 독립적인 개체로 취급하고 벡터 양자화(vector-quantized)된 선 표현을 도입하여 이러한 한계를 극복하고자 합니다. 실험 결과, 제안한 방법은 기존의 첨단 기술을 초월하는 성능을 보였습니다.

- **Technical Details**: VQ-SGen은 두 단계로 구성된 프레임워크를 기반으로 하고 있습니다. 첫 번째 단계에서는 각 선의 형태(shape)와 위치(location) 정보를 분리하여 벡터 양자화된 표현이 선의 형태 학습에 집중할 수 있도록 합니다. 두 번째 단계에서는 오토 리그레시브(autoregressive) Transformer를 사용하여 선의 의미(semantic), 형태, 위치를 생성 과정에 통합합니다. 이와 같은 접근 방식은 생성된 스케치의 긴밀한 구조적 및 의미적 관계를 보장합니다.

- **Performance Highlights**: VQ-SGen은 CreativeSketch 데이터 세트에서 실험을 통해 뛰어난 성능을 입증하였습니다. 많은 비교 실험과 폐기물(ablation) 연구에 따르면, 제안한 방법은 기존의 최신 기술을 넘어선 것으로 나타났습니다. 사용자 연구에서도 VQ-SGen은 다른 SoTA(method)를 상대로 consistently 우수한 성과를 내며, 고품질의 스케치를 생성하는 데 있어 확실한 이점을 제공합니다.



### SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis (https://arxiv.org/abs/2411.16443)
Comments:
          Project Page: this https URL

- **What's New**: SplatFlow는 3D Gaussian Splatting (3DGS) 생성 및 편집을 위한 새로운 포괄적 프레임워크를 제시합니다. 이 모델은 멀티 뷰 정류 흐름(multi-view rectified flow) 모델과 Gaussian Splatting Decoder (GSDecoder)로 구성되어 있으며, 텍스트 프롬프트에 조건을 걸어 멀티 뷰 이미지, 깊이 및 카메라 포즈를 동시에 생성할 수 있습니다. 이러한 혁신적인 접근 방식은 복잡한 카메라 경로와 다양한 장면 스케일을 효과적으로 처리할 수 있도록 합니다.

- **Technical Details**: SplatFlow는 레이턴트 공간(latent space)에서 작동하는 멀티 뷰 RF 모델을 사용하여 생성된 레이턴트 출력을 효율적으로 3DGS 표현으로 변환합니다. 이 과정에서 훈련이 필요 없는 변환 및 인페인팅 기술을 활용하여, SplatFlow는 3DGS 편집을 용이하게 하고 각종 3D 작업(객체 편집, 새로운 뷰 합성, 카메라 포즈 추정 등)을 지원합니다. 이를 통해 SplatFlow는 복잡한 파이프라인 없이 통합된 프레임워크에서 다양한 3D 작업을 수행할 수 있습니다.

- **Performance Highlights**: SplatFlow의 성능은 MVImgNet과 DL3DV-7K 데이터셋에서 검증되었습니다. 이 프레임워크는 3D 생성, 편집 및 인페인팅 기반 작업에서 그 유연성과 효과성을 입증하였습니다. 특히, 다양한 3D 작업을 동시에 처리하는 능력이 강조되며, 이는 기존 기술 대비 향상된 처리 속도와 품질을 제공합니다.



### AnonyNoise: Anonymizing Event Data with Smart Noise to Outsmart Re-Identification and Preserve Privacy (https://arxiv.org/abs/2411.16440)
Comments:
          Accepted at WACV25

- **What's New**: 최근 딥 뉴럴 네트워크(deep neural networks)의 능력이 향상되면서 개인의 재식별(re-identification)이 큰 위협이 되고 있습니다. 본 논문은 이러한 위협에 대응하기 위해 첫 번째 이벤트 익명화(event anonymization) 파이프라인을 제안하였으며, 이는 사람과 네트워크 모두의 재식별을 예방할 수 있는 방법입니다. 특히, 개인 식별 가능 정보(PII)를 커버하기 위해 학습 가능한 데이터 의존 노이즈(data-dependent noise)를 도입하여 효과적으로 공격자의 재식별 능력을 60%까지 줄입니다.

- **Technical Details**: 이 사건 익명화 파이프라인은 Gaussian noise를 활용하였으며, 이는 재식별을 방지하는 직관적인 기술입니다. 또한, Adversarial training을 적용하여 공격자가 남아 있는 개인 식별 정보를 복구할 수 없도록 합니다. 이러한 방법론은 이벤트 데이터에 적용되며, 재식별 위험을 효과적으로 줄이면서도 다운스트림 업무에 필요한 중요한 정보를 유지합니다.

- **Performance Highlights**: AnonyNoise라는 학습 가능한 노이즈 예측 기술을 통해 우리는 개인 정보를 제거하면서도 성능을 유지하는 균형을 이루었습니다. 우리의 익명화 방법은 보지 못한 데이터에 대해 잘 일반화되며, 이미지 재구성 및 역전 공격에 대해 강력한 저항력을 보여줍니다. 이를 통해 윤리적이고 안전한 비주얼 데이터의 활용을 촉진할 수 있습니다.



### Harnessing Superclasses for Learning from Hierarchical Databases (https://arxiv.org/abs/2411.16438)
- **What's New**: 이번 논문에서는 대규모 분류 문제에서 계층적으로 구성된 클래스에 대한 손실 함수를 소개합니다. 이 손실 함수는 각 예제를 단순한 클래스뿐 아니라 포함하는 상위 클래스(superclass)에도 할당할 수 있도록 계층의 지식을 활용합니다. 이를 통해 다양한 정밀도(granularity)에서 성능의 균형을 맞출 필요 없이 일관된 분류 목표를 동시에 추구할 수 있게 됩니다.

- **Technical Details**: 제안된 손실 함수는 소프트맥스 출력 계층을 가진 어떤 피드포워드(feedforward) 아키텍처에도 적용 가능하며, 이는 진짜 후행(class posterior) 클래스 확률에 의해 기대값이 최소화되는 적절한 스코어링 규칙(proper scoring rule)입니다. 우리는 세 가지 기준 벤치마크(reference benchmarks)에서 실험을 진행하였으며, 다양한 학습 시나리오를 다루기 위해 훈련 세트의 크기를 변화시켰습니다. 이 방법은 크로스 엔트로피(cross-entropy) 손실과 비교하여 추가적인 계산 비용이 크게 발생하지 않습니다.

- **Performance Highlights**: 이 방법은 정확도를 향상시키고, 지표에서 실제(label) 레이블과 멀리 떨어진 예상 레이블의 수를 줄이는 데 기여했습니다. 실험 결과, 제안된 손실 함수는 상위 클래스(superclass)와 세부 클래스(fine-grained class) 간의 일관된 분류 성능을 제공함을 보여줍니다.



### Privacy Protection in Personalized Diffusion Models via Targeted Cross-Attention Adversarial Attack (https://arxiv.org/abs/2411.16437)
Comments:
          Accepted at Safe Generative AI Workshop (NeurIPS 2024)

- **What's New**: 이번 논문에서는 개인화된 텍스트-이미지(T2I) 확산 모델의 개인 정보 보호를 위한 새로운 적대적 공격 방법인 CoPSAM(Concept Protection by Selective Attention Manipulation)을 제안합니다. CoPSAM은 T2I 확산 모델의 크로스 어텐션 레이어(cross-attention layers)만을 목표로 하여 맬웨어 목적의 악용 사례를 방지하는 데 초점을 맞추고 있습니다. 실험 결과, 저자들이 제안한 방법이 기존의 방법들보다 더 나은 성능을 발휘함을 보였습니다.

- **Technical Details**: 이 방법은 사용자가 제공한 이미지와 클래스 특정 토큰(class-specific token) 간의 크로스 어텐션 맵(cross-attention maps) 간 불일치를 최대화하기 위해 깨끗한 샘플에 위반성이 없는 노이즈(noise)를 추가하여 적대적 샘플을 생성합니다. 손실 함수(loss function)에 코사인 유사도(cosine similarity) 기법을 도입하여, 사용자 특정 토큰(user-specific token)과 클래스 특정 토큰 간의 유사성을 깨뜨리는 방식으로 진행됩니다. 이를 통해, 미세 조정 과정 동안 텍스트와 이미지 간의 연결을 방해하고, 이미지 분류 능력을 저해하면서도 내용 인식은 유지하게 됩니다.

- **Performance Highlights**: CelebA-HQ 데이터셋의 여러 이미지에 대한 실험을 통해 CoPSAM이 기존 최첨단 기술들을 능가하는 정량적 결과를 제공하는 것을 증명했습니다. 또한, CoPSAM 방식은 매우 낮은 수준의 노이즈 예산(noise budget)에서도 더 나은 보호 결과를 보여, 경쟁 기술들 대비 월등한 보호 성능을 발휘합니다. 나아가, 개인의 신원을 방어하며 비인가된 사용으로부터 콘텐츠를 효과적으로 보호할 수 있는 방법을 제시합니다.



### TopV-Nav: Unlocking the Top-View Spatial Reasoning Potential of MLLM for Zero-shot Object Navigation (https://arxiv.org/abs/2411.16425)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 TopV-Nav라는 새로운 MLLM 기반(MLLM-based) 방법을 소개합니다. 이 방법은 에이전트가 생소한 환경에서 이전에 보지 못한 객체를 찾는 Zero-Shot Object Navigation (ZSON) 임무를 수행할 수 있도록 도와줍니다. 특히 TopV-Nav는 기존의 시각적 관찰을 언어 기술로 변환하는 대신, 직접적으로 상위 뷰 맵(top-view map)에서 공간 정보를 활용하여 추론을 수행합니다. 그 결과 공간 정보의 손실을 방지하고 보다 효과적인 탐색이 가능해집니다.

- **Technical Details**: TopV-Nav는 Adaptive Visual Prompt Generation (AVPG) 방법을 포함하여 에이전트가 상위 뷰 맵에서 공간 정보를 적극적으로 활용할 수 있도록 합니다. AVPG는 맵에 시맨틱 정보를 추가하며, 이는 에이전트가 환경 내 객체들의 위치 관계를 보다 잘 이해하게 합니다. 또한, Dynamic Map Scaling (DMS) 메커니즘을 통해 에이전트는 필요에 따라 맵의 확대 및 축소를 조절하여 지역적 세밀한 추론을 강화할 수 있습니다. 이 외에도 Target-Guided Navigation (TGN) 메커니즘을 통해 미지의 영역에서도 목표 위치를 예측하고 이를 탐색 전략에 활용할 수 있습니다.

- **Performance Highlights**: MP3D와 HM3D 벤치마크에서의 실험 결과, TopV-Nav는 HM3D에서 3.9%의 SR(Success Rate)과 2.0%의 SPL(Success Path Length) 향상을 달성하여 기존 방법들을 초월하는 성능을 보였습니다. 새로운 메커니즘인 AVPG, DMS, TGN의 도입이 이러한 성능 향상에 기여했습니다. 이 연구는 MLLM의 공간적 추론 능력을 극대화함으로써 제로샷 탐색 임무의 새로운 가능성을 열었습니다.



### Machine Learning for the Digital Typhoon Dataset: Extensions to Multiple Basins and New Developments in Representations and Tasks (https://arxiv.org/abs/2411.16421)
- **What's New**: 이 논문에서는 디지털 태풍 데이터셋 V2가 소개되며, 이는 40년 이상의 태풍 위성 이미지 데이터셋으로서 머신러닝 모델을 위한 기준을 제시합니다. 새로운 버전 V2는 북반구 데이터에 더해 남반구의 열대성 저기압(tropical cyclone) 데이터가 포함되어 있어, 두 반구의 비교를 통해 새로운 연구 질문을 제기할 수 있습니다. 이는 데이터 기반 발견(data-driven discoveries)을 유도할 수 있는 가능성을 제공합니다.

- **Technical Details**: 이 데이터셋은 1978년부터 2022년까지의 태풍 이미지를 포함하며, Himawari 위성과 일본 기상청의 최신 경로(best track) 데이터로부터 생성되었습니다. V2에서는 자기 지도 학습(self-supervised learning; SSL) 프레임워크가 도입되어 태풍 강도 예측(intensity forecasting) 및 열대 저기압의 전이 예측(extra-tropical transition forecasting) 작업에 적용됩니다. 우리는 또한 새로운 작업인 태풍 중심 추정 작업(typhoon center estimation task)을 제안하며, 이는 위성 이미지로부터 태풍의 중심을 식별하는 과제입니다.

- **Performance Highlights**: 디지털 태풍 데이터셋 V2는 머신러닝 모델이 반구 및 유역(basin) 간에 일반화(generalization)될 수 있는 가능성을 탐구합니다. 보고된 결과는 물리적으로 서로 다른 유역에서 학습된 모델이 다른 유역에서도 효과적으로 작동할 수 있음을 보여줍니다. 이러한 연구는 태풍의 강도에 따라 물체 탐지 모델이 더 나은 성능을 나타내는 것을 지원합니다.



### A Study on Unsupervised Domain Adaptation for Semantic Segmentation in the Era of Vision-Language Models (https://arxiv.org/abs/2411.16407)
Comments:
          Accepted to British Machine Vision Conference (BMVC) 2024: Workshop on Robust Recognition in the Open World (RROW)

- **What's New**: 이번 연구에서는 가치 있는 도메인 일반화 (Domain Generalization, DG) 성능을 제공하는 비전-언어 (Vision-Language, VL) 사전 훈련된 인코더를 사용하여 기존의 비지도 도메인 적응 (Unsupervised Domain Adaptation, UDA) 방법을 향상시키는 새로운 접근법을 제시합니다. 특히, UDA 방법 중 DACS와 같은 방법에서 VL 인코더의 사용이 목표 성과와 일반화 성과 모두에서 10.0% mIoU와 13.7% mIoU의 성능 개선을 가져온다는 점이 주목할 만합니다. 이 연구는 일반적인 UDA 방법과는 다르게 VL 모델의 가능성을 강조합니다.

- **Technical Details**: 비지도 도메인 적응(UDA) 방법은 샘플이 레이블이 없는 목표 도메인에서만 이용 가능할 때 이를 목표 도메인에 맞게 조정하는데 주안점을 둡니다. 연구진은 DACS와 같은 UDA 프레임워크에 비전-언어 사전 훈련된 인코더를 추가하여 다양한 도메인에서 성능을 평가하였습니다. 또한, 주어진 데이터세트를 통해 일반화 성과와 UDA 성과가 항상 일치하지 않는 것을 발견했습니다.

- **Performance Highlights**: 실험 결과, GTA5에서 Cityscapes로의 도메인 전환에서 10.0% mIoU의 성능 향상이 발생했으며, 세 개의 보이지 않는 데이터세트에 대해서는 13.7% mIoU의 성능 향상이 나타났습니다. 이러한 연구 결과는 UDA 방법이 특정한 기상 조건 변화에서도 성능이 차별화됨을 확인시켜주었습니다. 결론적으로, 비전-언어 모델이 UDA 성과를 극대화할 수 있는 가능성을 보여주는 중요한 연구 내용이라 할 수 있습니다.



### Synthesising Handwritten Music with GANs: A Comprehensive Evaluation of CycleWGAN, ProGAN, and DCGAN (https://arxiv.org/abs/2411.16405)
Comments:
          10 pages, one page references, to appear on the IEEE Big Data 2024 2nd Workshop on AI Music Generation (AIMG 2024)

- **What's New**: 이 논문은 Optical Music Recognition (OMR) 시스템의 데이터를 보강하기 위해 Generative Adversarial Networks (GANs)를 활용하여 현실적인 손글씨 음악 악보를 생성하는 방법을 제안합니다. DCGAN, ProGAN, CycleWGAN 등 세 가지 GAN 모델을 비교하며, CycleWGAN이 스타일 전송 품질과 훈련 안정성에서 월등하게 우수한 성능을 보임을 보여줍니다. 이 접근법은 OMR 시스템의 성능 향상에 기여할 것으로 기대됩니다.

- **Technical Details**: CycleWGAN은 Wasserstein 손실 함수를 적용하여 훈련 안정성을 높이고 스타일 전송 품질을 개선합니다. 이 논문에서는 손글씨 음악 데이터의 부족 문제를 해결하기 위해 손글씨 음악 이미지를 생성하기 위한 GAN 아키텍처의 적용을 탐구하고, MUSCIMA 데이터셋에서 다양한 스타일을 활용하여 모델의 강건성을 높입니다. 세 가지 GAN 아키텍처의 성능을 비교하는 것이 이 연구의 핵심입니다.

- **Performance Highlights**: CycleWGAN은 FID 점수 41.87, IS 2.29, KID 0.05라는 뛰어난 성능을 기록하여 OMR 시스템 훈련에 적합한 고품질의 합성 데이터를 생성하는 것으로 평가됩니다. 이 연구는 다양한 손글씨 음악 샘플을 생성하여 기존의 손글씨 음악 데이터셋을 확장할 수 있는 유용한 통찰을 제공합니다. 이러한 성과는 향후 OMR 시스템 개발을 위한 실용적인 지침을 제공할 것으로 기대됩니다.



### Quadratic Gaussian Splatting for Efficient and Detailed Surface Reconstruction (https://arxiv.org/abs/2411.16392)
- **What's New**: 이 논문은 Quadratic Gaussian Splatting (QGS)라는 새로운 방법론을 제안합니다. 이 방법은 기존의 2D Gaussian Splatting (2DGS)에서의 디스크 대신 이차 표면을 사용하여 더 정교한 표면 적합성을 제공합니다. 또한, 복잡한 질감을 더 잘 포착할 수 있도록 비유클리드 공간에서 가우시안 분포를 정의합니다. 실험 결과 QGS는 정확하고 상세한 재구성을 달성하며 현재 최첨단 방법들을 초월하는 효과를 입증합니다.

- **Technical Details**: QGS는 장면의 기하학적 요소를 이차 표면(quadric surfaces)으로 대체하여 높은 차원 표현을 통해 장면 회전 정보를 제공하고 더 복잡한 표면 메쉬를 추출합니다. 기존 방법들과는 달리 QGS는 비유클리드 공간에서 측지 거리(geodesic distance)를 기반으로 가우시안 분포를 설정합니다. 이로 인해 표면에서 에너지를 집중시켜 복잡한 기하학적 질감을 효과적으로 포착할 수 있습니다. 또한, QGS는 깊이 정렬(depth sorting)에 대해 더욱 엄격한 기준을 적용하여 질 높은 기하학적 재구성을 가능하게 합니다.

- **Performance Highlights**: 실험을 통해 QGS는 DTU 및 TNT 데이터셋에서 기존 방법에 비해 더욱 정확하고 세밀한 재구성을 달성합니다. QGS는 대칭적이며 비대칭적인 기하학적 형태를 모두 다룰 수 있는 유연성을 제공하고, 정상 일관성(normal consistency)을 개선하는 방식으로 시각적 품질을 향상합니다. 이로 인해 QGS는 최신 기술에 비해 업계 최고의 기하학 재구성과 렌더링 품질을 달성하는 데 기여하고 있습니다.



### Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing (https://arxiv.org/abs/2411.16375)
Comments:
          Technical Report. Code is available at this https URL

- **What's New**: 본 논문에서는 Ca2-VDM이라는 새로운 비디오 확산 모델(Video Diffusion Model, VDM)을 제안합니다. Ca2-VDM은 인과적 생성(causal generation)과 캐시 공유(cache sharing)를 통해 효율적이며 빠른 비디오 생성을 가능하게 합니다. 기존의 비효율적인 오토리그레시브(autoregressive) 방식 대신, 이전 단계를 활용하여 중복 계산을 제거하고 성능을 향상시킵니다.

- **Technical Details**: Ca2-VDM은 중간 특성(intermediate features) 저장 및 재사용을 통해 계산 효율성을 높입니다. 인과적 생성 방식은 각 생성된 프레임이 이전 프레임에 종속되도록 하여 캐시를 사전 계산 가능하게 만듭니다. 또한, 캐시 공유 전략을 통해 저장 요구 사항을 줄이고 모든 디노이징(denoising) 단계에서 캐시를 공유할 수 있도록 설계되었습니다.

- **Performance Highlights**: Ca2-VDM은 MSR-VTT, UCF-101, Sky Timelapse 등 다양한 공개 데이터셋에서 평가되었습니다. 실험 결과, 이 모델은 기존 최고 성능의 VDM들에 비해 빠른 추론 속도를 유지하면서도 동등한 또는 우수한 정량적 및 정성적 성능을 보여줍니다. 이를 통해 Ca2-VDM은 효율적인 비디오 생성을 위한 뛰어난 선택이 될 수 있음을 입증하였습니다.



### A Review of Bayesian Uncertainty Quantification in Deep Probabilistic Image Segmentation (https://arxiv.org/abs/2411.16370)
Comments:
          20 pages

- **What's New**: 본 논문은 이미지 분할 분야에서 확률적(segmentation)에 대한 포괄적인 개요를 제공합니다. 특히, 대칭적 불확실성(aleatoric uncertainty)과 인식적 불확실성(epistemic uncertainty)이라는 두 가지 주요 개념을 통해 모델의 불확실성을 수량화하는 방법을 제시합니다. 이를 통해, 이론적인 기초와 다양한 실제 응용 분야를 연관시켜 주목받고 있는 논문입니다.

- **Technical Details**: 이미지 분할은 데이터의 픽셀 단위로 분류하는 작업으로, 객체와 관심 영역을 효과적으로 구분합니다. CNN(Convolutional Neural Networks) 기반의 방법들이 급격한 발전을 이루면서, 보다 정교한 불확실성 분석이 가능해졌습니다. 이 연구에서는 이러한 불확실성을 정확히 모델링하기 위해 베이지안 추론(Bayesian inference)을 근본적으로 활용하는 방법을 탐구하며, 새로운 접근법들을 검토합니다.

- **Performance Highlights**: 이 논문은 CNN 기반의 분할 분야에 큰 영향을 미친 이전 연구들의 성과를 분석하고 현재의 모델과 방법론을 비교합니다. 특히, U-Net과 같은 엔코더-디코더 모델이 여전히 우수한 성능을 보이며, 불확실성 수량화(quantification)에 대한 연구가 위험한 실제 상황에서의 의사결정에 중요한 영향을 끼칠 수 있음을 강조합니다. 또한, 향후 연구 과제와 함께 표준화 및 벤치마킹의 필요성을 강조하여 이 분야의 진전을 위한 실질적인 기초 자료를 제공합니다.



### Cluster-based human-in-the-loop strategy for improving machine learning-based circulating tumor cell detection in liquid biopsy (https://arxiv.org/abs/2411.16332)
- **What's New**: 이번 연구는 순환 종양 세포(CTC)와 비-CTC를 구분하는 데 있어 human-in-the-loop (HiL) 전략을 도입했습니다. 이는 기존 수동 평가의 한계를 극복하기 위해 기계 학습(ML) 기반의 CTC 탐지를 개선하는 방법입니다. 저자들은 자가 감독(Self-supervised) 딥러닝과 전통적인 ML 분류기를 결합하여 인간 전문가들이 반복적으로 새로운 레이블이 없는 훈련 샘플을 표적 샘플링하고 라벨링하도록 제안합니다.

- **Technical Details**: 제안된 샘플링 전략은 로컬 잠재 공간 클러스터의 분류 성능에 기반합니다. 이 접근 방식은 메타스타틱 유방암 환자의 체액 생검 데이터에서 무작위 샘플링에 비해 유의미한 장점을 보이는 것으로 입증되었습니다. 또한, 이 방식은 라벨이 부족한 훈련 데이터에서 기계 학습 시스템이 불확실하거나 잘못된 결정을 할 때 인간의 평가가 필수적임을 강조합니다.

- **Performance Highlights**: 제안된 방법은 메타스타틱 유방암 환자의 혈액 샘플에서 CTC 탐지의 정확성을 개선합니다. 반복적인 타겟 샘플링으로, 데이터의 품질을 높여 ML 시스템의 신뢰성을 높이고, 초기 훈련 수행에서 필수적인 검증을 제공합니다. 이를 통해 CTC와 비-CTC의 효과적인 구분이 가능해집니다.



### CapHDR2IR: Caption-Driven Transfer from Visible Light to Infrared Domain (https://arxiv.org/abs/2411.16327)
- **What's New**: 이번 논문에서는 CapHDR2IR라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 고동적 범위(High Dynamic Range, HDR) 이미지를 기반으로 하여 이미지를 생성함으로써 적외선(Infrared, IR) 이미지를 생성하는 경쟁력을 제공합니다. CapHDR2IR는 시각-언어 모델(Vision-Language Model, VLM)과 밀접하게 연결되어 있어, IR 이미지의 정보를 보다 의미 있게 만듭니다. 이는 적외선 이미지 생성의 신뢰성을 크게 향상시켜 주목받고 있습니다.

- **Technical Details**: CapHDR2IR는 HDR 이미지를 입력으로 사용하여 IR 이미지를 생성하는 방식으로 설계되었습니다. HDR 이미지는 다양한 조명 조건에서도 신뢰할 수 있는 IR 이미지 생성을 가능하게 하며, 각 이미지의 의미를 이해하는 밀집 캡션(dense captioning) 분기를 추가하여 이미지의 세부 정보를 강화합니다. 이 프레임워크는 태양광이 충분하지 않은 조건에서도 유용하며, 의미를 추출함으로써 생긴 문제인 pseudo-thermal-crossover 아티팩트를 해결하고자 합니다.

- **Performance Highlights**: CapHDR2IR는 HDRT 데이터셋에서 기존의 일반 도메인 전이 방법 및 가시광선에서 적외선으로의 이미지 변환을 위한 맞춤형 방법들과 비교하여 최첨단 성능을 달성했습니다. 이 모델은 고해상도 이미지를 보다 효과적으로 생성하며, 이미지의 질과 신뢰성을 크게 개선하는 데 성공하고 있습니다. 연구 결과는 CapHDR2IR가 실용적인 응용 프로그램에서 더 많은 가능성을 제공함을 보여줍니다.



### Brain-like emergent properties in deep networks: impact of network architecture, datasets and training (https://arxiv.org/abs/2411.16326)
- **What's New**: 이 논문은 최신 심층 신경망(DNNs)이 현실 세계의 시각적 작업에서 인간을 따라잡지 못하는 역설적인 문제에 초점을 맞추고 있습니다. 여기서는 심층 신경망이 인간의 시각 시스템과 얼마나 유사한지를 평가하기 위해 30개 이상의 최신 네트워크 아키텍처, 훈련 데이터셋 및 교육 방식을 체계적으로 평가한 결과를 보고합니다. 이러한 평가를 통해 심층 신경망의 설계 요인이 두뇌와의 유사성에 미치는 영향을 이해하는 데 기여하고자 합니다.

- **Technical Details**: 연구에서는 15가지의 시각 심리학 및 신경과학에서 잘 알려진 인지적, 신경적 특성을 선택하여 DNNs의 두뇌 유사성을 평가하였습니다. 각 네트워크에 대해 특정 속성을 테스트하여 상관 관계를 평가한 후, 'Brain Property Match (BPM)'라는 종합 메트릭을 제안하였습니다. 15가지 특성 각각에 대한 효과 강도는 -1에서 1까지의 범위를 가지며, 양수는 해당 특성이 존재함을 의미합니다.

- **Performance Highlights**: 연구 결과, 네트워크 아키텍처가 인간의 시각적 감지 특성과 가장 높은 관계를 보였고, 특정 네트워크가 모든 네트워크에 비해 일관되게 우수한 성능을 보이는 것은 아니었습니다. 뿐만 아니라, 최신 DNNs에서 발생하거나 결여된 두뇌 유사성 속성을 드러내어 기존 벤치마크와 보완적 역할을 수행합니다. 이러한 발견은 새로운 DNN 개발에 있어 두뇌 유사성을 증진시키기 위한 방향성을 제공할 것으로 기대됩니다.



### Luminance Component Analysis for Exposure Correction (https://arxiv.org/abs/2411.16325)
- **What's New**: 이번 논문은 기존의 노출 수정 기법의 한계를 극복하기 위해 Luminance Component Analysis (LCA)라는 새로운 방법을 제안합니다. LCA는 U-Net 구조를 기반으로 하여, 노출 관련 정보와 노출과 무관한 정보를 구분하는 정교한 접근 방식을 사용합니다. 이 과정에서, 색상의 왜곡을 최소화하고 세부정보 손실을 방지하기 위해 노출 관련 구성 요소만 조정합니다.

- **Technical Details**: LCA는 정규 직교(Stiefel) 매니폴드에서 비구속 문제로 변환하여 기하학적 최적화 알고리즘을 적용합니다. 이 방법은 photometric intensity, light wavelength 및 contrast와 같은 노출 관련 요소를 독립적으로 조정할 수 있도록 설계되었습니다. 또한, Riemannian gradient를 사용하여 기존의 정규 직교화를 달성합니다.

- **Performance Highlights**: 실험 결과, LCA는 RGB 색 공간에서 노출 관련 정보와 무관한 정보를 효과적으로 분리할 수 있음을 보여주었습니다. LCA는 노출 수정 데이터셋에서 PSNR 21.33과 SSIM 0.88이라는 최고의 성능을 달성하며, 28.72 FPS의 속도를 자랑합니다. 이는 기존의 깊이 학습 기반 방법들보다 우수한 성능을 나타내며, 노출 관련 구성 요소 조정의 효율성을 입증합니다.



### CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation (https://arxiv.org/abs/2411.16319)
- **What's New**: 이 논문에서는 2D 비지도(instance segmentation) 방식에서 3D 정보를 활용하여 객체 인스턴스를 세분화하는 새로운 접근법인 CutS3D를 제안합니다. 기존의 방식들이 2D 공간에서의 단순한 의미적(semantic) 관계만을 고려하고 있는 반면, CutS3D는 3D 포인트 클라우드(point cloud) 표현을 통해 보다 정밀한 인스턴스 구분이 가능합니다. 이로 인해 객체 경계에서의 인스턴스 분리를 더욱 효과적으로 수행할 수 있습니다.

- **Technical Details**: CutS3D는 두 가지 주요 요소로 구성되어 있습니다. 첫째, LocalCut 방법을 통해 초기 의미적 마스크를 기반으로 실제 3D 경계에서 객체를 절단(cut)합니다. 둘째, Spatial Importance 함수를 사용해 세그먼트를 3D 경계에 알맞게 조정하여 진정한 객체 경계를 더 잘 표현합니다. 또한 Spatial Confidence 지도를 도입하여 각 패치에서 생성된 패턴의 품질을 분석하고, 이를 통해 서로 다른 스케일에서 3D 절단 결과의 신뢰도를 평가합니다.

- **Performance Highlights**: CutS3D는 여러 표준 벤치마크에서 비지도 인스턴스 세분화와 객체 탐지 성능을 향상시켰습니다. 특히, 공간적 신뢰성을 고려한 세 가지 컴포넌트를 통해 깨끗한 학습 신호를 분리하기 위한 보강 훈련이 가능해졌습니다. 이번 연구는 객체의 3D 경계를 반영함으로써 기존 방법들을 초월할 수 있는 가능성을 보여주었습니다.



### One Diffusion to Generate Them A (https://arxiv.org/abs/2411.16318)
Comments:
          two first authors contribute equally

- **What's New**: OneDiffusion은 다양한 작업에 대해 양방향 이미지 합성과 이해를 지원하는 다목적 대규모 확산 모델입니다. 이 모델은 텍스트, 깊이, 자세, 레이아웃 및 의미론적 맵과 같은 입력으로부터 조건부 생성이 가능하며, 이미지 디블러링, 업스케일링 및 밀어내기 작업과 같은 다양한 작업을 처리할 수 있습니다. OneDiffusion은 또한 다중 뷰 생성, 카메라 자세 추정 및 순차 이미지 입력을 통한 즉각적인 개인화 기능을 제공합니다.

- **Technical Details**: OneDiffusion은 모든 작업을 훈련 중에 다양한 잡음 수준의 프레임 시퀀스로 처리하며, 각 프레임은 추론 시 조건부 이미지로 작용할 수 있도록 설계되었습니다. 모델을 훈련시키기 위해, 고품질 데이터와 기존 모델의 합성 출력을 포함한 One-Gen 데이터셋을 구축했습니다. 이 데이터셋은 여러 작업의 공동 훈련을 지원하며, 유연한 조건부 옵션과 개선된 일반화를 제공합니다.

- **Performance Highlights**: OneDiffusion은 텍스트-이미지(텍스트 입력으로 이미지를 생성) 및 다중 뷰 생성 작업에서 경쟁력 있는 성능을 보여주었으며, 기존의 특정 훈련 모델과 비슷한 수준의 성능을 발휘했습니다. 또한, 깊이 추정 작업에서도 우수한 성능을 발휘했으며, 새로운 조건부 설정인 텍스트-다중 뷰 및 이미지-다중 뷰 생성을 지원합니다. 이 모델은 복잡한 과제에서도 일반화가 뛰어나며, 다양한 표현과 자세를 가진 여러 이미지를 생성할 수 있습니다.



### Monocular Lane Detection Based on Deep Learning: A Survey (https://arxiv.org/abs/2411.16316)
- **What's New**: 이 논문은 자율 주행의 인식 시스템에서 중요한 역할을 하는 차선 탐지 방법에 대해 심층적으로 분석합니다. 최근의 깊은 학습 알고리즘이 적용된 단안 차선 탐지 방법들은 기존의 방식에 비해 뛰어난 성능을 보여주며, 이는 자율 주행 분야에서 주요 연구 주제로 떠오르고 있습니다. 논문은 2D 차선 탐지와 3D 차선 탐지 방법들을 아우르면서 다양한 알고리즘적 프레임워크의 핵심 설계를 다룹니다.

- **Technical Details**: 차선 탐지 알고리즘 프레임워크의 핵심 설계는 (1) 작업 패러다임, 차선 인스턴스 수준의 구별에 중점을 두고 있으며, (2) 차선 모델링, 즉 신경망에서 학습 가능한 파라미터로 나타냅니다. (3) 전역 정보 보완, 모호한 차선을 감지하는 데 도움이 되고, (4) 원근 효과 제거를 통해 하류 응용 프로그램에 적합한 3D 차선을 제공합니다. 이러한 요소들은 차선 탐지 기술의 발전을 포괄적으로 이해하는 데 기여합니다.

- **Performance Highlights**: 논문은 2D 및 3D 차선 탐지 방법의 성능을 비교 분석하며, 다양한 벤치마크에서 그 효율성을 조사합니다. 또한, 멀티태스킹 인식, 비디오 차선 탐지, 온라인 HD 맵 구축 및 차선 토폴로지 추론과 같은 연장된 작업들에 대해서도 논의하여 차선 탐지의 발전 방향을 제시합니다. 이러한 정보를 통해 독자들은 차선 탐지 연구의 진화 로드맵을 이해할 수 있습니다.



### EPS: Efficient Patch Sampling for Video Overfitting in Deep Super-Resolution Model Training (https://arxiv.org/abs/2411.16312)
- **What's New**: 본 논문에서는 비디오 송신의 질 향상을 위해 깊은 신경망(DNN)의 오버피팅 특성을 활용한 새로운 패치 샘플링 방법인 EPS(Efficient Patch Sampling)를 제안합니다. 기존 방법은 훈련 패치를 선택하는 데 필요한 계산 비용이 너무 높아 실제 적용이 제한되었습니다. EPS는 낮은 복잡도의 DCT 기반 특징을 이용해 각 패치의 복잡도를 측정하고, 보다 효율적으로 비디오 프레임에서 유용한 훈련 패치를 식별합니다.

- **Technical Details**: EPS는 비디오의 각 프레임을 그리드로 나누고, DCT 기반의 두 가지 특징을 통해 각 패치의 공간-시간 복잡도를 평가합니다. 패치는 히스토그램 분포를 사용해 클러스터로 그룹화되고, 가장 높은 복잡도를 가진 클러스터에서 훈련에 사용할 패치를 선택합니다. 이 방법은 훈련 중 불필요한 패치를 제외하여 시간과 계산 자원을 절약합니다.

- **Performance Highlights**: EPS 방법을 통해 샘플링된 패치 수는 해상도 및 클러스터 수에 따라 4%에서 25%로 감소하며, 여전히 높은 비디오 품질을 유지합니다. 이러한 방식은 기존 패치 샘플링 방법인 EMT와 비교해 전체 실행 시간을 83% 줄이는 성과를 나타냈습니다.



### Functionality understanding and segmentation in 3D scenes (https://arxiv.org/abs/2411.16310)
Comments:
          Technical report. 20 pages, 12 figures, 7 tables. Updated website link

- **What's New**: Fun3DU는 3D 장면에서 기능 이해를 위해 특별히 설계된 최초의 방법으로, 언어 모델을 사용하여 작업 설명을 해석하고 관심 객체를 식별합니다. 이 연구는 데이터세트의 제한된 활용성과 전통적인 모델의 한계를 극복하는 데 중점을 둡니다. Fun3DU는 사전 훈련된 모델에 의존하여 기능적 객체의 세분화를 수행하며, 이는 기존 방법들이 다루지 못했던 문제를 해결합니다.

- **Technical Details**: Fun3DU는 Chain-of-Thought reasoning을 활용하여 기능 설명을 해석하고, open-vocabulary segmentation을 통해 3D 장면의 컨텍스트 객체를 찾습니다. 또한, 수천 개의 뷰에서 정보를 추출하기 위해 새로운 가시성 기반 뷰 선택 알고리즘을 사용하여 효율성을 개선합니다. 최종적으로, 2D 마스크를 3D 포인트 클라우드로 변환하는 과정은 기하학적 정보를 통해 수행됩니다.

- **Performance Highlights**: 이 방법은 SceneFun3D 데이터셋에서 3000개 이상의 작업 설명과 230개의 장면에 대해 평가됩니다. Fun3DU는 최신의 open-vocabulary 3D 세분화 방법을 크게 초과하는 성능을 보이며, 평균으로 +13.2 mIoU의 향상을 기록했습니다. 이는 3D 기능 이해가 기존 방법보다 더 깊은 사고 능력을 필요로 함을 강조합니다.



### An End-to-End Robust Point Cloud Semantic Segmentation Network with Single-Step Conditional Diffusion Models (https://arxiv.org/abs/2411.16308)
- **What's New**: 본 논문에서는 Conditional Denoising Diffusion Probabilistic Models (DDPMs)와 Noise-Conditional Framework (NCF)의 한계를 극복하기 위한 새로운 접근법을 제안한다. 이를 위해 Conditional-Noise Framework (CNF)을 적용하여 3D 장면 이해를 위한 견고한 세그멘테이션 네트워크인 CDSegNet을 개발하였다. CDSegNet은 노이즈 네트워크를 학습 가능한 노이즈-특징 생성기로 모델링함으로써, 새롭고 강력한 일반화 능력을 보인다.

- **Technical Details**: CDSegNet은 노이즈 네트워크(NN)와 조건부 네트워크(CN)를 보조 네트워크와 주요 네트워크로 활용하여 Noise-Conditional Framework (CNF)의 장점을 최대한 유지한다. 이 네트워크는 입력된 노이즈 정보를 효과적으로 통합하여 세그멘테이션 성능을 향상시키며, 훈련 시 DDPM의 노이즈 추가 패턴을 따라가면서도 추론 단계에서는 단일 스텝으로 시맨틱 레이블을 생성할 수 있게 한다.

- **Performance Highlights**: CDSegNet은 대규모 실내외 벤치마크 테스트에서 기존의 방법론들보다 현저하게 향상된 성능을 보여주었다. 특히, 데이터 노이즈와 희소성에 대한 강력한 강인성을 기반으로, 주어진 3D 작업에서 탁월한 결과를 달성하고 있다. 이 연구는 DDPM이 3D 작업에서 적용될 수 있는 새로운 가능성을 열어주며, 앞으로의 연구에 많은 기여를 할 것으로 보인다.



### DiffDesign: Controllable Diffusion with Meta Prior for Efficient Interior Design Generation (https://arxiv.org/abs/2411.16301)
Comments:
          32 pages

- **What's New**: 본 논문은 DiffDesign이라는 새로운 모델을 제안하여, 내부 디자인 생성을 위한 효율적이고 제어 가능한 확산(diffusion) 모델을 개발하였습니다. 이 모델은 텍스트 설명이나 스케치로부터 고품질의 내부 디자인 도면을 생성할 수 있으며, 디자인 속성에 대한 교차 주의(cross-attention) 제어를 통해 생성 프로세스를 개선합니다. 또한, 400개 이상의 솔루션을 포함하는 DesignHelper라는 전용 데이터셋을 구축하여 DiffDesign의 훈련 및 평가를 지원합니다.

- **Technical Details**: DiffDesign은 사전 훈련된 2D diffusion 모델을 렌더링 백본으로 활용하며, 주어진 텍스트 설명이랑 생성된 이미지 간의 지역 일관성을 보장하기 위해 교차-view attention 모듈을 통합합니다. 디자인 속성을 명확하게 분리하여 조절함으로써, 외관(appearance) 및 설계(specification) 제어를 수행합니다. 이러한 접근 방식은 대규모 diffusion 모델의 기능을 보존하면서도, 텍스트 설명에 맞는 이미지를 생성할 수 있게 해줍니다.

- **Performance Highlights**: DiffDesign은 다양한 벤치마크 데이터셋에서 수행된 광범위한 실험을 통해 그 효과성과 강건함을 입증하였습니다. 실험 결과, DiffDesign은 시각적 품질, 다양성 및 내부 디자인의 전문성 측면에서 이전의 여러 방법들을 정량적, 정성적으로 초월합니다. 이러한 성과는 DiffDesign이 산업 표준을 충족하고, 제공된 프롬프트에 따라 정확하게 설계 요구사항을 반영하는 이미지를 생성할 수 있음을 보여줍니다.



### A Performance Increment Strategy for Semantic Segmentation of Low-Resolution Images from Damaged Roads (https://arxiv.org/abs/2411.16295)
- **What's New**: 이 논문에서는 안전한 자율주행을 위해 필수적인 도로 조건을 개선하기 위한 새로운 접근법인 Performance Increment Strategy for Semantic Segmentation (PISSS)를 제안합니다. 브라질의 85% 도로가 손상되어 있어 기존의 고해상도 이미지 기반 데이터셋으로는 부족한 문제를 해결합니다. 특히, 저해상도 이미지에서 소수의 픽셀로 구성된 객체, 불규칙한 형태, 그리고 저조한 클래스들 문제를 다루고 있습니다. 이러한 문제를 해결하고 PISSS를 통해 RTK와 TAS500 데이터셋에서 각각 79.8 mIoU와 68.8 mIoU의 최신 결과를 달성했습니다.

- **Technical Details**: PISSS 방법론은 14회의 훈련 실험 시리즈로 구성되며, 이는 다섯 단계로 나뉘어 성능을 극대화하는 것을 목표로 합니다. 특히 우선 baseline의 성능을 초과한 후 DeepLabV3+의 튜닝, 고급 기법과 cutmix를 적용하는 방식 등으로 변화를 줍니다. DeepLabV3+의 아키텍처에 대한 연구도 포함되어 있으며, ResNet의 max-pooling 레이어를 제거하여 소규모 객체의 세분화를 강화합니다. 이와 같은 기술적 기법들은 고해상도 이미지 대신 저해상도 이미지를 기반으로 성능을 개선하는 데 기여하고 있습니다.

- **Performance Highlights**: PISSS를 적용한 결과, RTK 데이터셋에서 79.8의 mIoU와 TAS500에서 68.8의 mIoU를 달성하며 현재까지의 최상위 성과를 기록했습니다. 이는 기존 저해상도 이미지에서의 객체 세분화 문제를 해결한 혁신적인 접근을 보여줍니다. 또한, 이 연구에서 제안된 방법은 다중 스케일 세분화, 작은 객체 감지 문제를 효과적으로 해결하는 데 중점을 두고 있습니다. 최종적으로, PISSS를 통해 기존의 문제들을 해결하며 자율주행의 정확성을 높이는 데 기여하고 있는 연구입니다.



### Utilizing Uncertainty in 2D Pose Detectors for Probabilistic 3D Human Mesh Recovery (https://arxiv.org/abs/2411.16289)
Comments:
          WACV 2025

- **What's New**: 본 논문에서는 단안 이미지(Monocular image)로부터 3D 인간 포즈와 형태를 추정하는 과정에서의 한계를 극복하기 위해 새로운 방법을 제안하고 있습니다. 기존의 확률적 방법들은 이미지에 따라 3D 인간 메쉬의 확률 분포를 배워왔지만, 단순한 Likelihood 최대화를 통한 학습만으로는 데이터의 완전한 분포를 포착하기에는 부족합니다. 이를 보완하기 위해 2D 포즈 감지기로부터 얻어진 Heatmap의 분포와의 거리 최소화를 통해 학습된 분포에 대한 추가 감독을 주는 방식을 도입하였습니다.

- **Technical Details**: 제안된 메서드는 Normalizing Flows를 활용하여 RGB 이미지에 대한 조건부 분포를 모델링하고, Heatmap에 인코딩된 분포를 감독 신호로 사용하여 학습합니다. 기존 방법들에서 유도한 포즈의 가설을 이미지에 투영하고, Heatmaps에서 샘플링한 부분과의 비교를 통해 Maximum Mean Discrepancy (MMD)를 거리 척도로 사용하여 별도의 밀도 추정 없이도 복잡한 분포를 재현할 수 있는 점이 특징입니다. 또한, 이 접근법은 3D 주석이 부족한 상황에서도 2D 포즈 감지기로부터 불확실성 정보를 흡수하여 효과적인 3D 모델 학습이 가능하도록 합니다.

- **Performance Highlights**: 3DPW와 EMDB 데이터셋에서의 실험 결과, 제안된 방법은 기존의 최첨단 확률적 방법들을 능가하는 성능을 보였습니다. 또한 본 연구는 개인 분할 마스크(person segmentation masks)를 사용하여 잘못된 가설의 수를 줄이는 기법을 추가하였습니다. 이를 통해 생성된 조인트 샘플이 마스크 안에 있는 비율을 평가하는 새롭고 간단한 두 가지 메트릭을 도입하여, 겉보기에 보이지 않는 조인트와 관련된 문제를 해결할 수 있는 가능성을 제시하였습니다.



### Open-Vocabulary Octree-Graph for 3D Scene Understanding (https://arxiv.org/abs/2411.16253)
Comments:
          11pages,7figures

- **What's New**: 이 논문에서는 개방 어휘(open-vocabulary) 3D 장면 이해를 위한 새로운 장면 표현 방식인 Octree-Graph를 제안합니다. 이 방식은 Chronological Group-wise Segment Merging (CGSM) 전략과 Instance Feature Aggregation (IFA) 알고리즘을 활용하여 3D 객체 및 해당 의미 체계를 효과적으로 생성합니다. Octree-Graph는 각 adaptive-octree가 그래프 노드로 작용하여 객체 간의 공간 관계를 정확하게 나타내며, 이로써 기존 방법들이 가진 단점을 극복하고자 합니다.

- **Technical Details**: Octree-Graph는 각 객체의 점유(j occupancy) 및 의미(semantics)에 대한 설명과 객체 간의 관계를 구조적으로 나타냅니다. 이 시스템은 층위별로 구조화된 하위 지역을 통해 3D 공간을 계층적으로 표현하는 adaptive-octree 구조를 채택하고 있습니다. CGSM 전략은 시간 순서에 따라 세그먼트를 그룹화하고, IFA 메서드는 각 객체에 대한 의미적 표현을 조합하는 과정에서 대표성과 독창성을 동시에 고려합니다.

- **Performance Highlights**: 다양한 널리 사용되는 데이터셋에서 수행된 광범위한 실험에서 Octree-Graph의 다재다능성과 효율성을 입증했습니다. 이 방법은 객체 검색(object retrieval), 점유 질의(occupancy queries), 경로 계획(path planning) 등 다양한 다운스트림 작업에 직접 적용될 수 있어 매우 유용합니다. 제안된 방법은 3D 장면 이해 분야의 기존 연구들이 가지는 한계를 넘어서고 있습니다.



### Diagnosis of diabetic retinopathy using machine learning & deep learning techniqu (https://arxiv.org/abs/2411.16250)
Comments:
          9 pages, 11 figures, Journal Paper

- **What's New**: 이번 연구에서는 안저 이미지(fundus images) 분석을 위한 새로운 방법을 제안합니다. 기존의 수동 분석 방식의 단점을 보완하기 위해 객체 탐지(object detection) 및 기계 학습(classification) 기법을 조합한 접근 방식을 사용합니다. 이를 통해 당뇨병성 망막병증(diabetic retinopathy) 등 다양한 안과 질환을 진단하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 우리는 YOLO_V8을 이용하여 안저 이미지에서 객체를 탐지하고, 시신경 디스크(optic disc), 시신경 컵(optic cup), 병변(lesions) 등의 관심 영역(ROIs)을 식별합니다. 이후, SVM(support vector machine) 분류 알고리즘을 통해 ROIs를 병리적 징후의 존재 여부에 따라 다른 DR 단계로 분류합니다. 이 방법은 정확도 84%를 기록하였으며, 신뢰할 수 있는 진단 도구입니다.

- **Performance Highlights**: 제안된 방법은 원거리 지역에서도 안저 질환의 선별(triage)에 유용하게 적용될 수 있습니다. 높은 정확도와 효율성을 바탕으로 시각 장애 예방에 기여할 수 있는 혁신적인 연구입니다. 또한, 이 접근 방식은 안과 진단의 자동화를 통해 의료 종사자들의 부담을 줄여줄 것으로 기대됩니다.



### Weakly supervised image segmentation for defect-based grading of fresh produc (https://arxiv.org/abs/2411.16219)
- **What's New**: 이번 연구에서는 후작업 공급망의 바나나 품질 평가 문제를 다루며, 핸드폰으로 촬영한 바나나 이미지에서 표면 결함을 검출 및 세분화하는 새로운 방법을 제안합니다. 우리는 일반적인 픽셀 수준의 주석이 아닌, 약한 감독(weak supervision) 기법을 활용하여 총 476장의 바나나 이미지를 수집하고, 결함을 최소한의 라벨로 표기했습니다. 이를 통해 수작업의 부담을 줄이고, SAM(Segment Anything Model)을 사용하여 77.6%의 panoptic quality score를 달성했습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 인도에서 수집된 476장의 바나나 묶음 이미지를 포함하고 있으며, 각 이미지는 1회 촬영되었습니다. 우리는 머신러닝 모델을 사용하여 결함을 자동 검출, 세분화 및 분류하고, 공정한 성능 평가를 위해 이미지를 다섯 개의 부분집합으로 나누어 교차 검증을 수행했습니다. 세분화 기법으로 ‘panoptic segmentation’을 채택하여 범주 및 인스턴스를 모두 고려하며, SAM을 통해 조악한 주석으로부터 정확한 분할을 이끌어내는 방식을 사용했습니다.

- **Performance Highlights**: 제안된 방법은 476개의 주석이 있는 이미지에서 잘 작동했으며, 기존 모델과 비교할 때 적은 데이터로도 정확하게 결함을 식별하는 성능을 보였습니다. 우리는 또한 결함의 유형이 공급망 운영자에게 결함 원인을 파악하는 데 도움을 줄 수 있다는 점에 주목했습니다. 이 연구는 로우 데이터 환경에서도 머신러닝 솔루션의 적용 가능성을 입증하는 데 기여하고 있으며, 결함의 수와 크기를 함께 분석할 수 있는 유망한 방법으로 평가받고 있습니다.



### Mixed Degradation Image Restoration via Local Dynamic Optimization and Conditional Embedding (https://arxiv.org/abs/2411.16217)
Comments:
          10 pages, 3 figures, 8 tables

- **What's New**: 이 논문에서는 여러 가지 손상 요소들이 혼합된 이미지를 효과적으로 복원할 수 있는 새로운 다중-하나(multiple-in-one) 이미지 복원 모델인 MDIR을 제안합니다. 기존의 모델들은 특성의 다양성과 단일성 문제에 직면했으나, MDIR은 이를 해결하기 위해 Local Dynamic Optimization(LDO) 모듈과 Conditional Feature Embedding(CFE) 모듈을 채택하고 있습니다. 

- **Technical Details**: MDIR은 LDO 모듈을 통해 다양한 유형과 강도의 필터를 동적으로 생성하여 서로 다른 이미지 영역에서의 고주파 및 저주파 정보를 최적화합니다. CFE 모듈은 복합 손상 이미지를 위한 다중 라벨 속성을 제공하여 디코딩 과정에서 특정 손상 유형을 쉽게 인식할 수 있도록 돕습니다. 이를 통해 MDIR은 혼합 손상 복원 작업에서 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: 제안한 MDIR 모델은 새로운 데이터셋인 CIR에서 SOTA(최신 기술 성과) 성능을 달성하며, PSNR에서 기존 모델인 PromptIR을 1 dB 이상 초과하는 성능을 보입니다. 또한, CSD(Desnowing) 및 Rain100L/H(Deraining)과 같은 세 가지 공공 단일 손상 데이터셋에서도 우수한 결과를 기록했습니다.



### SMGDiff: Soccer Motion Generation using diffusion probabilistic models (https://arxiv.org/abs/2411.16216)
- **What's New**: 이번 논문에서는 사용자 제어가 가능한 실시간 축구 동작 생성을 위한 새로운 모델인 SMGDiff를 제안합니다. 이 모델은 두 단계의 프레임워크를 통해 고품질의 스포트 관성을 동시에 보장하며 효율적으로 동작을 생성합니다. SMGDiff는 사용자 설정에 기반하여 실제로 보지 못한 자연스러운 축구 동작을 생성할 수 있습니다.

- **Technical Details**: SMGDiff는 사용자 제공 제어 신호로부터 전 세계적인 궤적을 생성하고, 이를 바탕으로 트랜스포머 기반의 자가 회귀 확산 모델을 사용하여 축구 동작을 생성합니다. 첫 번째 단계에서는 조잡한 사용자 제어 신호를 세부적인 궤적으로 변환하며, 두 번째 단계에서는 그 궤적에 따라 축구 동작을 생성합니다. 이 과정에서 볼과 발의 접촉을 최적화하기 위한 접촉 안내 모듈도 포함되어 있습니다.

- **Performance Highlights**: SMGDiff의 성능은 기존 방법들과 비교하여 동작의 품질과 조건 정렬에서 크게 개선되었음을 실험을 통해 입증하였습니다. 또한, 제안된 대규모 축구 동작 데이터셋인 Soccer-X는 1.08백만 프레임 이상의 다양한 동작 데이터를 포함하여 고품질의 축구 동작 생성을 위한 기반을 마련하였습니다.



### SAVEn-Vid: Synergistic Audio-Visual Integration for Enhanced Understanding in Long Video Contex (https://arxiv.org/abs/2411.16213)
- **What's New**: 이 논문에서는 SAVEn-Vid라는 최초의 장기 오디오-비주얼 비디오 데이터셋이 소개됩니다. 이 데이터셋은 58,000개 이상의 오디오-비주얼 지시사항을 포함하며, SAVEn-Vid를 기반으로 하는 시간 인식 오디오-비주얼 대형 언어 모델인 SAVEnVideo도 제안됩니다. 또한 모델의 오디오-비주얼 이해 능력을 평가하기 위한 2,500개 Q&A 쌍으로 구성된 벤치마크 AVBench도 공개됩니다.

- **Technical Details**: SAVEnVideo는 오디오 및 비주얼 기능을 시공간적으로 정렬하며, 효과적인 기능 토큰 압축을 통해 긴 컨텍스트를 처리하는 것을 목표로 합니다. 새로운 데이터셋 파이프라인을 통해 비디오에서 시각 및 오디오 데이터를 추출하고, Qwen2.5-72B 모델을 사용하여 고품질 캡션을 생성하여 Q&A 쌍으로 결합합니다. SAVEn-Vid는 특히 상세한 타임스탬프 주석과 포괄적인 오디오-비주얼 캡션으로 구성된 대규모 데이터셋입니다.

- **Performance Highlights**: 실험 결과, SAVEnVideo는 제로 샷 장기 비디오 작업인 Video-MME에서 기존 최고 비디오-LLM보다 3.61% 성능이 향상되었습니다. 또한 제로 샷 오디오-비주얼 작업인 Music-AVQA에서도 선도적인 오디오-비주얼 LLM을 1.29% 초과하는 성과를 보였습니다. 결과적으로 7B 파라미터 스케일에서 SAVEnVideo는 최신 기술 성능을 달성합니다.



### VIRES: Video Instance Repainting with Sketch and Text Guidanc (https://arxiv.org/abs/2411.16199)
- **What's New**: VIRES는 스케치와 텍스트 가이드를 통해 비디오 인스턴스 리페인팅(video instance repainting)과 교체, 생성, 삭제 기능을 지원하는 새로운 방법입니다. 기존 방법들은 시간 일관성(temporal consistency)과 제공된 스케치 시퀀스와의 정확한 정렬에서 어려움을 겪었습니다. VIRES는 텍스트-비디오 모델의 생성적 선행조건을 활용하여 시각적으로 만족스러운 결과를 유지합니다.

- **Technical Details**: VIRES는 구조 레이아웃을 효과적으로 추출하기 위해 Sequential ControlNet과 표준화된 자기 스케일링(standardized self-scaling)을 도입합니다. 스케치 주의(sketch attention)를 사용하여 세밀한 스케치 의미를 해석하고, 스케치 인식 인코더(sketch-aware encoder)가 재페인팅 결과를 다중 수준의 텍스처 특징과 정렬합니다. 또한 VireSet이라는 새로운 데이터 세트를 제공하여 비디오 인스턴스 편집 방법에 대한 교육 및 평가를 지원합니다.

- **Performance Highlights**: 실험 결과 VIRES는 시각적 품질(visual quality), 시간 일관성(temporal consistency), 조건 정렬(condition alignment), 인간 평가(human ratings)에서 최신 기술(state-of-the-art)들을 초월하는 성능을 보여줍니다. VireSet은 85K의 학습 비디오와 1K의 평가 비디오로 구성되어 있어, 비디오 인스턴스 편집 기법을 평가하기 위한 충분한 데이터를 제공합니다.



### Interpreting Object-level Foundation Models via Visual Precision Search (https://arxiv.org/abs/2411.16198)
- **What's New**: 이번 연구는 멀티모달( multimodal ) 프리트레이닝의 발전이 Grounding DINO 및 Florence-2와 같은 오브젝트 레벨 기반 모델을 시각적 기초(visual grounding) 및 객체 탐지(object detection) 작업에서 크게 향상시켰으나 이들 모델의 결정 해석이 점점 어려워지고 있음을 강조합니다. 기존 해석 가능한 속성(attribution) 방법들이 지니고 있는 한계를 극복하기 위해, 더 적은 영역에서 정확한 속성 맵을 생성하는 Visual Precision Search 방법을 제안합니다. 이 방법은 멀티모달 융합에서의 속성 문제를 해결하기 위해 내부 모델 매개변수를 우회하고, 입력을 스파스(sub-sparse) 서브 영역으로 나누어 핵심 결정 영역을 정확하게 식별합니다.

- **Technical Details**: 제안된 방법은 입력 영역을 슈퍼 픽셀(segmentation) 분할을 통해 일련의 서브 영역으로 분리하고, 이러한 스파스 서브 영역의 중요성을 결정하는 점수들(consistency and collaboration scores)을 사용하여 오브젝트 결정의 중요한 요소를 해석합니다. 특정 객체의 탐지를 설명하는 설득력 있는 맵을 생성하기 위해, 모델의 탐지 실패를 신속하게 유도하는 몇 가지 중요한 영역을 제거하는 접근을 사용합니다. 이론적 분석을 바탕으로 한 새로운 서브모듈(submodular) 함수는 정확한 탐지를 지원하는 단서(clue)와 강력한 조합 효과를 발휘하는 영역을 식별하여 해석 가능성을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 MS COCO, RefCOCO, LVIS 데이터셋에서 Grounding DINO 및 Florence-2와 같은 오브젝트 레벨 모델에 대해 SOTA(최신 기술 수준) 방식들보다 뛰어난 해석 의도(faithfulness)를 달성하여 각각 23.7%, 31.6%, 20.1%의 성과를 보였습니다. 또한, 탐지 오류 및 기반의 실패에 대한 해석 측면에서도 기존 방법들을 초월하여 여러 평가 지표에서 성능 향상이 두드러졌습니다. 이 연구는 크게 두 가지 기여를 통해 오브젝트 탐지 해석 분야의 확장을 제안합니다.



### Learn from Foundation Model: Fruit Detection Model without Manual Annotation (https://arxiv.org/abs/2411.16196)
Comments:
          17 pages, 12 figures, conference or other essential info

- **What's New**: 이번 연구는 대규모 기초 모델(Foundation Models)을 활용하여 데이터가 제한된 농업 분야에서 지식을 전이하는 새로운 프레임워크 SDM-D(Segmentation-Description-Matching-Distilling)를 제안합니다. 이 방법은 수동 주석 없이 특정 도메인에 맞춤화된 모델을 훈련할 수 있으며, 다양한 과일 탐지 작업에서 기존 방법들을 능가하는 성능을 보여줍니다. 또한 25,000장의 이미지를 포함한 MegaFruits라는 새로운 고품질 세분화 데이터셋을 생성하였습니다.

- **Technical Details**: SDM-D 알고리즘은 두 가지 기초 모델을 활용하여 세분화(Segmentation)와 분류(Classification)를 수행합니다. 첫 번째 단계에서 'Segment Anything Model(SAM)'는 세분화를 지원하며, 'OpenCLIP'는 오픈 보캐블러리(classification) 기능을 제공합니다. 두 번째 단계에서는 새로운 지식 증류(Knowledge Distillation) 메커니즘을 통해 경량 모델을 생성하여 추론 속도와 인식 정확성을 동시에 향상시킵니다.

- **Performance Highlights**: SDM-D는 수동 주석 없이 다양한 과일 탐지 태스크(object detection, semantic segmentation, instance segmentation)에서 강력한 성능을 보여주며, 기존의 데이터가 풍부한 모델과 유사한 성능을 나타냅니다. 특히, SDM-D는 Grounding SAM과 YOLO-World와 같은 오픈셋 탐지 방법들을 초월하는 결과를 얻었습니다. 실험을 통해 이 방법이 속도와 정확성 모두에서 기존 모델보다 뛰어난 것으로 나타났습니다.



### Fancy123: One Image to High-Quality 3D Mesh Generation via Plug-and-Play Deformation (https://arxiv.org/abs/2411.16185)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 'Fancy123'라는 새로운 파이프라인을 제안하며, 두 개의 강화 모듈과 unprojection(언프로젝션) 작업을 통해 기존의 문제를 해결하고자 합니다. 기존 방법들은 주로 2D multiview diffusion 모델을 사용하여 중간의 multiview 이미지를 생성하고, Large Reconstruction Model (LRM)로 최종 mesh를 만드는 방식입니다. 하지만, 기존의 multiview 이미지들은 지역적으로 불일치하고, 생성된 mesh는 원본 이미지와의 유사성이 떨어지거나 흐릿하게 나타나는 문제를 동반합니다.

- **Technical Details**: Fancy123 파이프라인은 세 가지 주요 단계를 포함합니다: 첫째, 단일 이미지를 기반으로 multiview diffusion 모델을 활용하여 multiview 이미지를 생성하고, LRM을 통해 초기 mesh를 재구성합니다. 둘째, appearance enhancement module은 2D multiview 이미지를 변형하여 불일치를 수정하고, 이를 mesh에 unprojection하여 일관성을 확보합니다. 셋째, fidelity enhancement module은 3D mesh를 변형하여 입력 이미지와 일치하도록 하고, 입력 이미지를 mesh에 unprojection합니다.

- **Performance Highlights**: 저자들은 Fancy123의 성능이 품질이 뛰어난 유사한 방법들보다 월등히 뛰어난 것을 보여주는 다양한 정성적 및 정량적 실험을 수행하였습니다. 두 개의 플러그 앤 플레이 모듈은 다양한 기존의 'one-image-to-3D' 방법에 매끄럽게 통합될 수 있어, Fancy123의 활용 가능성을 더욱 넓혀줍니다. 또한, 각 모듈의 효과는 ablation(변별 실험) 및 backbone-replacement(기반 교체 실험)를 통해 검증되었습니다.



### Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking (https://arxiv.org/abs/2411.16183)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구는 기존의 3D Instance Segmentation(3DIS) 방법이 가지는 과도한 분할 문제를 해결하기 위해 새로운 접근 방식을 제안하고 있습니다. 이 방법은 2D 마스크 분할 및 추적을 위한 SAM-2 모델을 활용하여 비디오 프레임 간에 일관된 객체 마스크를 생성하며, 동적 프로그래밍 알고리즘을 통해 최적의 뷰 세트를 선택해 3D 제안을 다듬습니다.

- **Technical Details**: 3D-Aware 2D Mask Tracking 모듈을 도입하여 2D 기반 모델로부터의 강력한 3D 단서를 활용하여 객체 수준의 일관된 마스크를 생성합니다. 또한, 3D 제안 최적화 과정에서는 모든 뷰에서의 겹침 정도를 기반으로 슈퍼포인트를 선택하며, 이는 NP-hard 문제임에도 불구하고 동적 프로그래밍 알고리즘을 통해 빠르고 효과적인 솔루션을 제공합니다.

- **Performance Highlights**: ScanNet200 및 ScanNet++ 데이터셋에서 평가된 결과, 본 연구의 방법은 Open3DIS 및 완전 감독 모델에 비해 클래스와 무관한 측정에서 상당한 개선을 보였습니다. 또한 Open-Vocabulary 및 Open-Ended 3D Instance Segmentation 작업에서도 성능을 향상시켜 다양한 실제 응용 분야에 맞춤형으로 적용 가능한 효과적인 솔루션을 입증하였습니다.



### Event-boosted Deformable 3D Gaussians for Fast Dynamic Scene Reconstruction (https://arxiv.org/abs/2411.16180)
- **What's New**: 본 논문은 3D Gaussian Splatting (3D-GS)과 Event Camera를 통합하여 빠른 동적 장면 재구성을 위한 새로운 접근법을 제시합니다. 기존 RGB 카메라의 한계를 극복하기 위해, 고속 동작에 적합한 연속적인 모션 데이터 캡처가 가능한 Event Camera를 사용합니다. 새로운 GS-Threshold Joint Modeling (GTJM) 전략을 도입하여 이벤트 모델링의 질을 향상시켰습니다.

- **Technical Details**: Event Camera는 마이크로초 수준의 시간 해상도를 제공하여 전통적인 RGB 카메라가 놓치는 빠른 장면 변화를 캡처합니다. 논문에서는 GS-Threshold Joint Modeling을 통해 RGB 프레임의 강도 변화로 초기 임계값을 추정하고, 3D-GS에서 렌더링된 결과를 사용하여 이벤트 감독을 개선하는 상호 강화 프로세스를 형성합니다. 또한, 동적 지역과 정적 지역을 구분하는 Dynamic-Static Decomposition (DSD) 전략을 제안하며, 정적 영역에서의 불필요한 변형 계산을 피하고 동적 영역에 집중하여 렌더링 속도를 가속화합니다.

- **Performance Highlights**: 최종적으로, 제안된 방법은 RTX 3090 GPU에서 400×400 해상도로 156 FPS의 고충실도 동적 재구성이 가능합니다. 기존의 동적 3D-GS 방법들이 겪고 있는 비효율적인 모델링 문제를 해결하여, 렌더링 속도를 증가시키고 재구성 품질을 향상시킵니다. 이를 통해 가상 현실 및 엔터테인먼트 분야에서의 활용 가능성을 높입니다.



### SALOVA: Segment-Augmented Long Video Assistant for Targeted Retrieval and Routing in Long-Form Video Analysis (https://arxiv.org/abs/2411.16173)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 SALOVA(세그먼트-증강 긴 비디오 어시스턴트)라는 새로운 비디오-LLM 프레임워크를 소개합니다. SALOVA는 긴 비디오 콘텐츠의 이해를 개선하기 위해 특정 세그먼트를 효율적으로 검색하는 방법을 채택하고 있습니다. 특히, 비디오 콘텐츠의 단기적이며 중요한 정보를 보존하여, 모델의 응답의 맥락 관련성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: SALOVA는 SceneWalk 데이터셋을 기반으로 하며, 이는 87.8K개의 긴 비디오를 포함하고 있으며 각 비디오 세그먼트에 대한 밀접한 주석을 제공합니다. 또한, 동적 라우팅 메커니즘 및 시공간 프로젝터(spatio-temporal projector)를 활용하여 사용자 쿼리에 따라 관련 비디오 세그먼트를 적절하게 검색하고 처리하는 아키텍처를 개발하였습니다. 이러한 설계는 긴 비디오를 처리하는 데 있어 효율성을 높이고, 문맥 통일성을 유지하는 데 기여합니다.

- **Performance Highlights**: 광범위한 실험을 통해 SALOVA는 복잡한 긴 비디오의 이해에 있어 기존 비디오-LLM 모델에 비해 향상된 성능을 보여주었습니다. 모델은 중요 시각 정보를 손실할 위험을 줄이며 중요한 이벤트를 놓치는 사례를 줄이는데 효과적입니다. 이로써 SALOVA는 긴 비디오 콘텐츠를 보다 최적화된 방식으로 해석할 수 있는 능력을 입증하였습니다.



### U2NeRF: Unsupervised Underwater Image Restoration and Neural Radiance Fields (https://arxiv.org/abs/2411.16172)
Comments:
          ICLR Tiny Papers 2024. arXiv admin note: text overlap with arXiv:2207.13298

- **What's New**: 이번 논문에서는 U2NeRF라는 새로운 방법을 제안하여 물속 환경에서의 이미지 복원 및 렌더링 문제를 다룹니다. 이는 unsupervised(비지도) 학습 방식으로 NeRF(Neural Radiance Fields) 아키텍처를 변형하여 구성된 것입니다. 특히, 기존의 방법들이 의존하는 고해상도 이미지 대신 다양한 각도에서의 다중 뷰 정보를 이용하여 물속 장면의 복원을 시도합니다. 이는 물속 이미지 개선을 위한 새로운 접근법으로 주목받고 있습니다.

- **Technical Details**: U2NeRF 방법은 네 가지 주요 구성 요소로 복원된 색상을 분리하는 과정을 포함합니다. 이들은 장면 복사(scene radiance), 직접 전송 맵(direct transmission map), 후방 산란 전송 맵(backscatter transmission map), 그리고 전역 배경 빛(global background light) 입니다. 이러한 요소들은 서로 결합되어 원본 이미지를 재구성하는 데 사용되며, 일반화된 NeRF 트랜스포머(GNT)를 활용하여 다중 뷰 정보를 통합하고 새로운 시점을 렌더링합니다.

- **Performance Highlights**: 실험 결과, U2NeRF는 특정 장면에서 최적화를 수행할 때 평균 11% LPIPS, 5% UIQM, 4% UCIQE의 향상을 보여 주었습니다. 이는 기존의 여러 기준선(baselines)을 초월하는 성능으로, 물속 이미지 복원 및 렌더링 능력이 크게 향상된 것을 의미합니다. 또한, 12개의 물속 장면이 포함된 UVS 데이터셋(Underwater View Synthesis dataset)을 제공하여, 이를 활용한 추가 연구 및 개발이 가능하도록 합니다.



### Image Generation Diversity Issues and How to Tame Them (https://arxiv.org/abs/2411.16171)
Comments:
          17 pages, 6 tables, 12 figures

- **What's New**: 이 논문에서는 현재 생성 모델에서 다양성이 결여된 문제를 강조하며, 기존의 일반적인 지표가 이를 측정하는 데 한계가 있음을 지적합니다. 모델의 다양성을 정량화하기 위해 이미지 검색 문제로 프레임을 구성하였고, 이를 통해 Image Retrieval Score (IRS)라는 새로운 지표를 제안합니다. IRS는 생성 모델 출력의 다양성을 측정하는 직관적인 지표로, 하이퍼파라미터에 의존하지 않고도 통계적 신뢰성을 제공합니다.

- **Technical Details**: 연구진은 IRS를 사용하여 실제 이미지 검색을 위한 쿼리로 합성 데이터를 활용하여 다양성을 평가하는 방법을 제시합니다. IRS는 상징적으로 총기 예제를 통해 모델의 다양성을 나타내며, 현재 사용되는 기능 추출기가 다양성을 효과적으로 측정하지 못하는 한계를 보여줍니다. 이를 기반으로 생성 모델의 다양성을 향상시키기 위한 새로운 방법인 Diversity-Aware Diffusion Models (DiADM)을 도입하여 이미지 품질 손실 없이 무조건적인 확산 모델의 다양성을 증가시킵니다.

- **Performance Highlights**: 실험 결과, 현재의 확산 모델들은 실제 분포의 제한된 부분에 수렴하며, 최신 모델들은 훈련 데이터의 77% 이상의 다양성을 초과하지 못함을 확인하였습니다. 또한, 여러 가지 상태의 고급 모델을 평가한 결과, 다양성 부족 현상이 여전히 심각함을 지적했습니다. 이러한 데이터 다양성의 결여는 생성 품질 저하 없이 해결될 수 있으며, 제안된 방법이 이에 대한 실질적인 개선점을 제공할 것으로 기대됩니다.



### CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction (https://arxiv.org/abs/2411.16170)
- **What's New**: 최근 자원 제약이 있는 모바일 장치에서 적용 가능한 높은 효율성을 가진 linear-complexity visual Transformers의 필요성이 대두되었습니다. 본 논문에서는 새로운 deCoupled duAl-interactive lineaR attEntion (CARE) 메커니즘을 제안하며, 특징의 분리와 상호작용을 통해 linear attention의 가능성을 극대화할 수 있음을 밝혔습니다. 이는 안대칭 feature decoupling 전략을 사용하여 지역 추론 편향(local inductive bias)과 장거리 의존성(long-range dependencies)을 효과적으로 학습할 수 있도록 합니다.

- **Technical Details**: 제안된 CARE 메커니즘은 지역 정보와 전역 정보 간의 효율적인 상호작용을 위해 dual interaction module을 도입합니다. 또한, dynamic memory unit을 설계하여 네트워크 파이프라인을 통해 중요한 정보를 유지하며, 이를 통해 모델의 효율성을 더욱 향상시킵니다. 이러한 비대칭 분리 학습 방식을 통해 각 feature 간의 보완성을 완전히 활용하여 높은 효율성과 정확성을 동시에 달성할 수 있습니다.

- **Performance Highlights**: 주요 실험인 ImageNet-1K, COCO, ADE20K 데이터셋에서 CARE 메커니즘의 효과가 검증되었습니다. 예를 들어, ImageNet-1K에서 78.4/82.1%의 top-1 정확도를 기록하며, 각각 0.7/1.9 GMAC의 계산 비용을 소모하는 등 경쟁력 있는 성능을 보여주었습니다. 이러한 결과는 본 연구가 자원 제약 환경에서도 고성능 비주얼 트랜스포머를 가능하게 한다는 점에서 의미가 있습니다.



### Local and Global Feature Attention Fusion Network for Face Recognition (https://arxiv.org/abs/2411.16169)
- **What's New**: 이번 연구는 저품질 얼굴 이미지 인식에서의 새로운 접근 방식인 Local and Global Feature Attention Fusion (LGAF) 네트워크를 제안합니다. LGAF 네트워크는 로컬(local)과 글로벌(global) 얼굴 특징의 주의를 효과적으로 조정하여 서로 보완적인 정보를 활용합니다. 이 네트워크는 저품질 이미지에서 발생할 수 있는 다양한 피쳐 품질 편향을 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: LGAF 네트워크는 두 가지 핵심 모듈, 즉 Local and Global Feature Fusion (LGF)와 Multi-Head Multi-Scale Local Feature Extraction (MHMS) 모듈로 구성됩니다. LGF 모듈은 피쳐 품질을 기반으로 로컬 및 글로벌 피쳐 간의 주의를 동적으로 측정하고, MHMS 모듈은 다양한 스케일과 채널 차원에서 얼굴 특징의 중요성을 평가하여 보다 풍부한 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, LGAF 네트워크는 CFP-FP, CPLFW, AgeDB, CALFW 등의 검증 세트에서 최고 평균 성능을 기록했습니다. 또한 TinyFace와 SCFace 데이터셋에서 최신 기술(SoTA)과 비교하여 뛰어난 성능을 보였습니다. 이러한 결과는 LGAF 네트워크가 저해상도와 고해상도 얼굴 인식 모두에서 효과적임을 입증합니다.



### Text-to-Image Synthesis: A Decade Survey (https://arxiv.org/abs/2411.16164)
Comments:
          In this survey, we review over 440 recent works on T2I

- **What's New**: 텍스트에서 이미지 생성(text-to-image synthesis, T2I)은 인공지능 생성 콘텐츠(Artificial Intelligence Generated Content, AIGC)에서 중요한 발전으로 자리 잡고 있으며, 이미지 생성에 대한 새로운 접근 방식을 제시합니다. 본 논문은 최근 T2I에 관한 440개 이상의 논문을 종합적으로 검토하며 GAN, autoregressive 모델, diffusion 모델의 진화와 그들이 T2I에 미친 영향을 분석하고 있습니다. 특히, 각 모델의 생성 능력과 텍스트에 조건화된 다양성을 중점적으로 다루며, 최신 연구 방향과 이와 관련된 안전성 및 일관성 문제를 함께 살펴봅니다.

- **Technical Details**: 이 논문에서는 T2I의 기본 모델인 GAN, autoregressive 모델, diffusion 모델의 수학적 원리와 기본 구조를 설명합니다. GAN은 생성자(Generator)와 판별자(Discriminator)로 구성되어 있으며, 생성자는 노이즈로부터 샘플을 생성하고 판별자는 진짜 샘플과 생성된 샘플을 구별하는 역할을 합니다. Autoregressive 모델은 Transformer 아키텍처를 활용하여 복잡한 시맨틱 관계를 캡처하며, DALL-E와 CogView가 그 대표적인 사례입니다. Diffusion 모델은 높은 품질의 이미지를 생성하는 데 있어 최근 가장 진보된 접근법으로, GLIDE와 Latent Diffusion Model(LDM) 등이 이에 해당합니다.

- **Performance Highlights**: T2I 기술의 발전은 다양한 분야에서 콘텐츠 제작을 혁신적으로 변화시키고 있으며, AI-generated content에 중요한 기여를 하고 있습니다. GAN의 도입으로 향상된 이미지 품질뿐 아니라, DALL-E와 같은 모델들은 대규모 데이터셋을 활용하여 다양한 이미지를 생성하는 능력을 보여주고 있습니다. Diffusion 모델은 영상 생성 분야에서 독보적인 성능을 인정받으며, 현재 다수의 연구자들이 빠르게 발전하는 이 기술에 대한 연구를 진행하고 있습니다. 이 연구는 T2I의 발전 방향과 향후 연구 기회에 대한 기초 자료를 제공하는 데 중요한 역할을 합니다.



### Sparse patches adversarial attacks via extrapolating point-wise information (https://arxiv.org/abs/2411.16162)
Comments:
          AdvML-Frontiers 24: The 3nd Workshop on New Frontiers in Adversarial Machine Learning, NeurIPS 24

- **What's New**: 이 논문에서는 희소한 패치 공격(sparse patch attacks)을 위한 새로운 접근 방식을 제안합니다. 이전의 공격 방법들은 여러 패치(patch)의 위치와 변동을 동시에 최적화하지 못했는데, 본 연구는 이를 가능하게 합니다. 제안된 방법은 임의의 수와 형태로 희소 패치의 위치와 변동을 최적화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 희소한 적대적 소음(sparse adversarial perturbation) 및 패치 적대적 공격(patch adversarial attack)은 자율 시스템에 대한 보안 위험으로 간주됩니다. 이 접근법은 밀집한 적대적 변동을 점별(point-wise)으로 trimming하여 이전의 방법보다 더 정교한 최적화를 지원합니다. 또한, 제안된 방법은 L0 노름(bounds) 제한을 포함한 다양한 상황에서 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 여러 가지 최신 기술(state-of-the-art)과 비교하여 성능을 크게 향상시키는 것을 보여주었습니다. 특히, 이미지넷(ImageNet) 분류 작업(classification task)에서 다양한 모델에 대해 평가한 결과, 모든 설정에서 우수한 성과를 얻었습니다. 이러한 결과는 제안된 방법의 실용성과 효율성을 입증해줍니다.



### MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Mod (https://arxiv.org/abs/2411.16157)
Comments:
          Models and codes will be released at this https URL. The project page is at this https URL

- **What's New**: MVGenMaster는 3D priors와 결합된 다중 시점 분산 모델로, 다양한 Novel View Synthesis (NVS) 작업을 해결하기 위해 설계되었습니다. 이 모델은 메트릭 깊이(metric depth)와 카메라 포즈(camera poses)를 이용해 3D 일관성을 극대화하며 최대 100개의 새로운 뷰를 생성할 수 있습니다. 또한, MvD-1M이라는 대규모 다중 시점 이미지 데이터셋을 구축하여 3D 모델을 뛰어난 일반화 능력으로 훈련시키고 있습니다.

- **Technical Details**: MVGenMaster는 StableDiffusion2(SD2)를 기반으로 하여 다중 뷰 생성을 위한 교차 뷰 주의력(cross-view attention)과 자세 표현(pose representation)을 확장하였습니다. 이 모델은 Plücker ray를 사용하여 카메라 포즈를 밀집하게 표현하고, 전통적인 깊이 정합 기법인 Canonical Coordinate Map(CCM)과 RGB 픽셀을 활용하여 3D 공간에서의 일관성을 유지합니다. 최신 훈련 없이도 여러 새로운 뷰를 생성할 수 있는 키 재조정(key-rescaling) 기법을 도입하여 주목력 희석(attention dilution) 문제를 해결했습니다.

- **Performance Highlights**: MVGenMaster는 기존 방법들에 비해 in-domain 및 OOD 벤치마크에서 일관되게 뛰어난 성능을 보이고 있습니다. 이 모델은 메트릭 깊이 priors를 통해 다양하고 복잡한 시나리오에서도 높은 다중 뷰 일관성과 강력한 일반화 능력을 보장합니다. 또한, 1.6백만 장의 장면으로 구성된 대규모 다중 시점 데이터셋을 활용해 훈련된 MVGenMaster는 최첨단 NVS 결과를 보여줍니다.



### VideoOrion: Tokenizing Object Dynamics in Videos (https://arxiv.org/abs/2411.16156)
- **What's New**: VideoOrion은 비디오의 공간-시간 역학을 포괄적으로 포착하는 비디오 대형 언어 모델(Video-LLM)입니다. 기존의 비디오 모델들의 한계를 극복하기 위해, 물체 동역학을 객체 토큰으로 변환하는 새로운 접근 방식을 제안합니다. 이를 통해 정보 손실을 최소화하고, 보다 명확한 의미 표현을 가능하게 합니다.

- **Technical Details**: VideoOrion은 객체 중심(branch)과 비디오 중심(branch)의 두 가지 경로를 통해 기능합니다. 객체 동역학을 처리하기 위해, detect-segment-track 파이프라인을 채택하여 전문 비전 모델의 지식을 활용하여 비디오에서 객체의 의미를 추출합니다. 객관적인 정보 처리와 더불어, 각 객체 토큰은 세분화된 정보를 담고 있어 모델의 물체 수준 이해를 향상시킵니다.

- **Performance Highlights**: 실험 결과, VideoOrion은 일반 비디오 질문 응답 및 비디오 기반 참조 작업에서 경쟁력 있는 성능을 보여줍니다. 객체 중심의 기능 집합으로 얻은 안정적인 성능 개선이 확인되었으며, 이는 VideoOrion이 비디오 이해 능력을 크게 향상시키는 데 기여하고 있음을 시사합니다.



### Revisiting Marr in Face: The Building of 2D--2.5D--3D Representations in Deep Neural Networks (https://arxiv.org/abs/2411.16148)
- **What's New**: 이번 연구에서는 Deep Neural Networks (DNN)가 인간의 시각 시스템과 어떻게 유사하게 작동하는지를 탐구하며, David Marr의 2D-2.5D-3D 구성 이론에 대한 실증적 증거를 제시합니다. 저자들은 이러한 연구를 위해 그래픽 프로브(graphics probe)라고 불리는 서브 네트워크를 도입하여 네트워크의 중간 층에서 이미지를 복원하는 과정을 분석했습니다. 이 프로브는 2D 및 3D 이미지 형식을 지원하며, DNN의 복잡한 이미지 인코딩 과정을 시각적으로 분해하여 보여주는 기법입니다.

- **Technical Details**: 이 연구는 DNN의 중간 층에 그래픽 프로브를 주입하여 이미지를 재구성하는 메커니즘을 탐색합니다. 그래픽 프로브는 정보 수집을 위해 self-attention 기법을 활용하고, K개의 그래픽 프로브로 나뉘어 여러 가지 CG 요소(예: 깊이 맵, 알베도 맵 등)를 통해 이미지 조각을 생성합니다. 이 과정을 통해 DNN이 낮은 레이어에서 2D 형상을 인코딩하고 높은 레이어에서 3D 형상을 구성함을 확인하며, 중간 레이어에서는 2.5D적인 기하학적 표현을 형성하는 것을 발견했습니다.

- **Performance Highlights**: 실험 결과, DNN의 그래픽 프로브는 처음에 평면적인 깊이 맵을 보여주고, 중간 레이어에서 풍부한 노멀 변화를 포함하는 2.5D 상태로 발전하며, 최종적으로 3D 모델로 진화합니다. 이러한 관찰은 DNN이 시각 인식을 통해 2D에서 3D로의 전환 과정을 거치는 방식을 잘 설명하며, 연구 결과는 DNN의 중간 표현이 객체 중심 개념에 따르고 있음을 시사합니다. 최종적으로 연구는 DNN의 표현이 템플릿 기반이거나 3D 기반의 형태로 발전하는 다양한 조건을 밝혀내는 데 초점을 맞추고 있습니다.



### TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation (https://arxiv.org/abs/2411.16132)
Comments:
          IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025)

- **What's New**: 이 논문은 TreeFormer라는 새로운 식물 스켈레톤 구조 추정기를 제안합니다. 기존의 이미지에서 스켈레톤 구조를 추정하는 방식을 개선하기 위해, 트리 제약(graph generation)을 적용한 그래프 생성 방법을 결합하여 더욱 정확한 식물 구조 추정을 목표로 하고 있습니다. 특히, 이 연구는 무제한 그래프를 최소 신장 트리(minimum spanning tree, MST)로 투영하는 과정을 포함하여 기존의 방법보다 더 뛰어난 정확성을 보여줍니다.

- **Technical Details**: TreeFormer는 학습 기반 그래프 생성기와 전통적인 그래프 알고리즘을 통합하여 훈련 루프 동안 제약을 적용합니다. 방법론적으로, 제약이 없는 그래프를 MST로 투영하고 그래디언트 하강 최적화(gradient descent optimization) 과정에서 원치 않는 특성 값을 억제하는 선택적 특성 억제(selective feature suppression, SFS) 레이어를 도입합니다. 이로 인해 각 훈련 루프에서 제약을 자연스럽게 통합하여, 식물의 스켈레톤 구조를 더 정확하게 추정할 수 있습니다.

- **Performance Highlights**: 실험 결과, TreeFormer는 합성 트리 패턴, 실제 식물 뿌리 및 포도나무 가지 이미지를 포함한 다양한 식물 이미지에서 목표 스켈레톤 구조를 정확하게 추정했음을 보여주었습니다. 기존의 두 단계 방법론과 비교할 때 우리의 방법은 단일 이미지에서 더 높은 정확성을 달성했습니다. 이러한 결과는 TreeFormer가 농업 및 식물 과학 분야에 기여할 수 있는 가능성을 나타냅니다.



### Three Cars Approaching within 100m! Enhancing Distant Geometry by Tri-Axis Voxel Scanning for Camera-based Semantic Scene Completion (https://arxiv.org/abs/2411.16129)
- **What's New**: 본 논문에서는 카메라 기반의 시맨틱 장면 완성(Semantic Scene Completion, SSC) 문제를 해결하기 위해 ScanSSC라는 새로운 모델을 제안합니다. 또한, 이 모델은 Scan Module과 Scan Loss로 구성되어 있으며, 각각 근거리 장면의 맥락을 활용하여 원거리 장면을 향상시키도록 설계되었습니다. 이 접근법을 통해 기존 방법들보다 더 나은 성능을 보이는 것을 목표로 하고 있습니다.

- **Technical Details**: Scan Module은 축별 마스킹(self-attention)을 활용하여 원거리 보셀(voxel)이 이전 보셀과 관계를 맺을 수 있도록 하는 독창적인 구조를 가지고 있습니다. 또한, Scan Loss는 누적 로짓(logits)과 해당 축에 대한 클래스 분포 간의 교차 엔트로피(cross-entropy)를 계산하여, 원거리 보셀로 풍부한 맥락 정보를 전파합니다. 이러한 요소들이 결합하여 ScanSSC는 최고의 성능을 달성하게 됩니다.

- **Performance Highlights**: ScanSSC는 SemanticKITTI 및 SSCBench-KITTI-360 벤치마크에서 각각 44.54, 48.29의 IoU 및 17.40, 20.14의 mIoU를 기록하며 기존 방법들을 크게 초월하는 성능을 입증했습니다. 이러한 결과는 Scan Module과 Scan Loss의 시너지 효과가 명확히 작용했음을 보여줍니다. 따라서, ScanSSC는 다양한 거리에서의 강건한 성능 개선을 달성하며, 자율 주행 시스템의 안전성을 크게 향상시킬 것으로 기대됩니다.



### CIA: Controllable Image Augmentation Framework Based on Stable Diffusion (https://arxiv.org/abs/2411.16128)
- **What's New**: 이번 연구에서는 CIA라는 모듈형 데이터 증강 파이프라인을 제안합니다. 이 파이프라인은 Stable Diffusion을 이용해서 합성 이미지를 생성하고, 정의된 품질 메트릭스를 사용하여 저품질 샘플을 필터링합니다. 또한, ControlNet을 통해 생성된 이미지에서 특정 패턴의 존재를 강제할 수 있도록 하였습니다. CIA는 안정된 데이터셋 품질을 보장하면서 더욱 효과적인 모델 학습을 지원할 수 있는 기회를 제공합니다.

- **Technical Details**: CIA는 네 개의 모듈로 구성되며, 첫 번째 모듈인 Extraction은 원본 이미지에서 특성 추출을 수행합니다. 이후 ControlNet은 추출된 특성을 활용하여 Stable Diffusion의 출력을 조건화하여 정밀한 제어를 할 수 있습니다. Generation 모듈은 텍스트 프롬프트와 결합된 추출된 특성을 기반으로 새로운 이미지를 합성하며, Quality Assessment 모듈은 선택한 품질 메트릭스를 통해 생성된 이미지를 필터링합니다.

- **Performance Highlights**: CIA를 활용한 인간 객체 탐지 사례 연구에서는 데이터량이 부족한 상황에서도 유의미한 성능 향상을 기록했습니다. 특히, CIA로 생성된 이미지를 활용함으로써 실제 이미지의 양을 두 배로 늘린 경우와 유사한 성능에 도달했습니다. 이는 데이터 제약 환경에서도 효과적인 객체 탐지 시스템의 구축이 가능함을 시사합니다.



### Med-PerSAM: One-Shot Visual Prompt Tuning for Personalized Segment Anything Model in Medical Domain (https://arxiv.org/abs/2411.16123)
- **What's New**: 이 논문에서는 Med-PerSAM이라는 새로운 one-shot 프레임워크를 소개합니다. 기본적으로 이 프레임워크는 의료 도메인에서 SAM의 성능을 향상시키기 위해 설계되었습니다. Med-PerSAM은 시각적 프롬프트 엔지니어링에 독점하여 추가 학습이나 사람의 개입 없이 자동화된 프롬프트 생성 프로세스를 활용합니다.

- **Technical Details**: Med-PerSAM의 기본적인 이미지 왜곡 손실로는 SSIM(Structural Similarity Index Measure) 손실이 사용되었습니다. 그러나 OdontoAI 및 CAMUS 데이터셋의 경우, NCC(Normalized Cross-Correlation) 손실이 사용되었으며, 이 손실은 의료 이미지 등록에서 일반적으로 사용하는 함수입니다. 프레임워크의 정규화항은 흐름 필드의 부드러움을 보장하여 급격한 왜곡을 방지합니다.

- **Performance Highlights**: Med-PerSAM은 다양한 2D 의료 이미징 데이터셋에서 기존 SAM 기반 접근법 및 기타 기초 모델보다 우수한 성능을 보였습니다. 특히, 훈련 및 재훈련 과정에서 비교적 짧은 시간 내에 성과를 달성하여 효율성을 강조합니다. 이 연구는 깊은 의료 전문 지식이 없는 사람에게도 유용한 도구가 될 것으로 기대됩니다.



### UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Imag (https://arxiv.org/abs/2411.16106)
Comments:
          9 pages, 3 figures

- **What's New**: 본 연구에서는 이전의 CAD 모델이나 다수의 참조 뷰에 의존하지 않고, 단일의 포즈가 없는 RGB-D 참조 이미지를 활용하여 보이지 않는 객체의 포즈를 추정하는 새로운 방법인 UNOPose를 제안합니다. 기존의 접근법들이 상대 포즈를 추정할 때 참조 이미지를 포즈 앵커로 사용했던 것과 달리, 본 설정에서는 SE(3) 공간 전역에 걸쳐 변형이 나타나기 때문에 훨씬 더 복잡한 환경을 다루어야 합니다. 이는 기존 방법들이 갖고 있던 한계를 극복하고, 낮은 겹침 정도의 뷰포인트에서도 효과적으로 포즈를 추정할 수 있는 기회를 제공합니다.

- **Technical Details**: UNOPose는 크게 두 가지 핵심 개념으로 구성됩니다. 첫째, SE(3)-불변 글로벌 참조 프레임을 통해 객체 표현을 표준화하고, 다양한 포즈와 크기 변형에 대한 도전과제를 극복합니다. 둘째, coarse-to-fine 패러다임을 적용하여 참조 객체와 쿼리 객체 간의 상대 포즈를 점진적으로 추정함으로써 높은 신뢰성을 제공합니다. 이를 위해 오버랩 예측기를 도입하여 겹치는 영역을 자동으로 식별하고 대응의 신뢰성을 조정합니다.

- **Performance Highlights**: YCB-V, LM-O, TUD-L 데이터셋을 활용한 실험 결과, UNOPose는 단일 참조 설정에서 기존의 방법들과 비교하여 모두 우수한 성능을 보였습니다. 특히, 포즈가 없는 참조 이미지를 사용한 UNOPose는 CAD 모델 기반 방법에 비해 동등한 성능을 보여주며, ARBOP 메트릭 기준으로도 뛰어난 결과를 기록했습니다. 본 연구는 새로운 기준인 BOP Challenge를 기반으로 한 벤치마크 자료도 제공하여 미래 연구의 길잡이가 될 것입니다.



### ENCLIP: Ensembling and Clustering-Based Contrastive Language-Image Pretraining for Fashion Multimodal Search with Limited Data and Low-Quality Images (https://arxiv.org/abs/2411.16096)
- **What's New**: 이 논문은 패션 인텔리전스를 위한 Contrastive Language-Image Pretraining (CLIP) 모델의 성능을 향상시키기 위한 새로운 접근 방식, 즉 ENCLIP을 제시합니다. ENCLIP은 데이터 부족과 저화질 이미지 문제를 해결하며, 여러 CLIP 모델을 학습시키고 집계하는 알고리즘을 포함합니다. 이 방법은 데이터 부족과 이미지 품질 문제를 고려하여 패션 도메인에 특화된 멀티모달 검색 능력을 극대화합니다.

- **Technical Details**: ENCLIP 방법론은 여러 CLIP 모델의 출력을 집계하고 클러스터링 기법을 활용하여 이미지를 유사한 그룹으로 묶습니다. 데이터셋으로는 Kaggle의 'Fashion Product Images (Small)'를 활용하며, 이는 인도 패션 제품을 포함한 다양한 패션 아이템의 이미지와 텍스트 설명으로 구성되어 있습니다. 전처리된 이미지와 텍스트는 CLIP의 입력으로 준비되며, 벡터 데이터베이스는 이미지 저장과 검색을 돕기 위해 사용됩니다.

- **Performance Highlights**: 실험 결과는 ENCLIP의 효과성을 입증하며, 패션 분야에서의 CLIP의 잠재력을 열어줍니다. 이 방법은 데이터 부족과 저화질 이미지가 만연한 패션 인텔리전스 분야에서 실용적인 해결책을 제공합니다. 전반적으로 ENCLIP 접근 방식은 패션 인텔리전스 분야에 커다란 공헌을 하며, 효율성을 높이는 실질적인 솔루션을 제공합니다.



### AI-Generated Image Quality Assessment Based on Task-Specific Prompt and Multi-Granularity Similarity (https://arxiv.org/abs/2411.16087)
- **What's New**: 최근 AI가 생성한 이미지(AIGIs)에 대한 관심이 높아지면서, 그 품질과 정렬 품질을 평가하는 것이 중요해졌습니다. 본 논문에서는 기존의 평가 방법들이 초기 프롬프트에 과도하게 의존하고 있다는 점을 지적하며, 새로운 품질 평가 방법인 TSP-MGS를 제안합니다. 이 방법은 과업 전용 프롬프트를 설계하여 정렬 품질과 지각 품질을 분리하여 평가할 수 있도록 합니다.

- **Technical Details**: 제안된 TSP-MGS 방법은 멀티-그래뉼러리 유사도(multi-granularity similarity)를 통해 AIGIs와 프롬프트 간의 유사도를 측정합니다. 우선 과업 특화된 프롬프트를 사용해 지각 품질과 정렬 품질을 평가하고, Coarse-grained 유사도를 통해 전체적인 품질을 인식합니다. 마지막으로, Fine-grained 유사도를 이용해 더 상세한 품질 인식을 위한 정보를 수집하여 품질 예측을 정확하게 수행합니다.

- **Performance Highlights**: AGIQA-1K 및 AGIQA-3K 벤치마크에서 실험을 통해, 제안된 TSP-MGS 방법이 기존 방법들보다 우수한 품질 예측 결과를 보여주었습니다. 이 결과는 AIGIs의 품질 평가에 있어 더욱 효과적이고 신뢰할 수 있는 접근 방식을 제공함을 나타냅니다. 따라서, TSP-MGS는 AI 생성 콘텐츠의 품질 평가와 선택에 중요한 역할을 할 것으로 기대됩니다.



### Leverage Task Context for Object Affordance Ranking (https://arxiv.org/abs/2411.16082)
- **What's New**: 이번 연구에서는 다양한 작업 맥락(task context)에 따라 물체의 affordance(용도)에 대한 순위를 정하는 방법을 제안합니다. 기존 연구는 물체의 affordance를 동등하게 취급하여 엉뚱한 선택을 초래했지만, 이 연구는 작업에 따른 우선순위를 제대로 모델링하여 지능형 에이전트가 올바른 결정을 내릴 수 있도록 돕습니다. 새로운 Context-embed Group Ranking Framework(CGR)를 통해 작업 관련 객체의 관계를 더 잘 이해할 수 있는 방법을 제공합니다.

- **Technical Details**: CGR 프레임워크는 작업에 특화된 시각적 특성을 추출하고 객체 간의 우선순위 관계를 모델링합니다. 이는 Task Relation Mining (TRM) 모듈과 Graph Group Update (GGU) 모듈로 구성되어 객체의 현재 맥락을 고려하여 보다 세밀하게 객체를 평가합니다. CGR은 이미지 내에서 작업 관련 객체의 경계 상자를 예측하고 해당 객체들의 우선순위를 주어진 작업 맥락에 기반하여 동적으로 조정합니다.

- **Performance Highlights**: 이 연구에서 개발된 Task-oriented Affordance Ranking (TAR) 데이터셋은 25개의 작업과 50,404개의 실제 이미지, 66만 개 이상의 객체 인스턴스를 포함하여 대규모로 구성되었습니다. 실험 결과, 제안된 방법이 기존 최첨단 모델보다 뛰어난 결과를 나타내며, 이는 물체의 affordance 이해와 작업 맥락에 따른 순위 매김에서의 우수성을 보여줍니다. 향후 연구를 위한 강력한 기초 모델로 자리 잡을 것입니다.



### Boosting 3D Object Generation through PBR Materials (https://arxiv.org/abs/2411.16080)
Comments:
          Accepted to SIGGRAPH Asia 2024 Conference Papers

- **What's New**: 본 논문에서는 Physics-Based Rendering (PBR) 재료의 관점에서 3D 객체 생성의 품질을 향상시키기 위한 혁신적인 접근 방식을 제안한다. 특히, 알베도(albedo), 거칠기(roughness), 금속성(metalness) 및 범프맵(bump maps)과 같은 PBR 재료의 구성 요소를 분석하여 이러한 값을 효율적으로 추출한다. 기존의 방법론들이 주로 텍스처에 국한되어 있었던 점에서 벗어나, 실제 조명 조건에서의 렌더링 품질을 향상시키는 데 중점을 두고 있다.

- **Technical Details**: 알베도 및 범프맵 추출을 위해, 저자들은 합성 데이터로 미세 조정된 Stable Diffusion 모델을 활용하며, 이를 통해 생성된 3D 객체에 일관된 알베도 UV 및 범프 UV를 제공한다. 또한 거칠기 및 금속성 맵에 대해서는 반자동(semi-automatic) 프로세스를 도입하여, 직관적 조정을 허용하는 방식으로 처리한다. 이 과정은 Segment-Anything-Model에 의해 3D 분할 마스크를 얻어내어, 객체의 의미론적 일관성을 유지한다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 다양한 최신 3D 생성 방법들과 결합하여 생성된 3D 객체의 품질과 사실성을 상당히 향상한다. 특히, 자연스러운 재조명 효과와 함께 고해상도 지형 품질을 제공하며, 기존 방법에 비해 더 정교한 결과를 도출할 수 있음을 보여준다. 그러므로 본 연구는 향후 3D 콘텐츠 생성 분야에 중요한 기여를 할 것으로 기대된다.



### Debiasing Classifiers by Amplifying Bias with Latent Diffusion and Large Language Models (https://arxiv.org/abs/2411.16079)
Comments:
          8 pages + Appendix

- **What's New**: 본 논문에서는 DiffuBias라는 새로운 파이프라인을 소개합니다. 이 방법은 텍스트에서 이미지를 생성하여 분류기의 강건성을 향상시키는 데 중점을 둡니다. 특히, 훈련 단계 없이 편향(conflict) 샘플을 생성할 수 있는 점이 특징입니다. DiffuBias는 사전 훈련된 diffusion 및 이미지 캡셔닝 모델을 활용하여 편향을 도전하는 이미지를 생성함으로써 데이터 샘플을 더 대표성 있게 만듭니다.

- **Technical Details**: DiffuBias는 사전 훈련된 generative 모델, latent diffusion 모델, 이미지 캡셔닝 모델을 이용하여 편향-conflict 샘플을 생성하는 간단하면서도 효과적인 debiasing 프레임워크입니다. 이 방법은 생성 모델을 추가로 훈련할 필요가 없으며, 이런 설정 덕분에 리소스를 절약할 수 있습니다. 또한, DiffuBias는 기존의 GAN보다 더 컴팩트하고 효율적인 접근법으로 지속 가능한 개발에 기여합니다.

- **Performance Highlights**: 본 연구의 종합적 실험 결과는 DiffuBias가 여러 벤치마크 데이터셋에서 최첨단 성능을 달성했음을 보여줍니다. 특히, 데이터 셋에서 편향-conflict 비율이 증가할수록 모델 성능이 향상된다는 점도 확인되었습니다. 또한 다양한 생성 모델에 대한 탄소 배출 및 에너지 소비 비교 분석을 통해 계산 효율성의 중요성을 강조합니다.



### Geometry Distributions (https://arxiv.org/abs/2411.16076)
Comments:
          For the project site, see this https URL

- **What's New**: 이번 연구는 3D 데이터의 새로운 기하학적 표현인 Geometry Distributions (GeomDist)을 제안합니다. 이 접근법은 면의 topology와 구조적 무결성에 관계없이 표면 점의 분포를 모델링하여, 기존의 좌표 기반 네트워크가 직면한 문제를 해결하고자 합니다. 특히, 본 연구는 diffusion 모델이라는 혁신적인 네트워크 아키텍처를 활용해 표면의 정밀한 기하학적 세부정보를 캡처하는 방법을 탐색합니다.

- **Technical Details**: GeomDist는 주어진 표면을 확률 분포로 모델링하며, 이 분포에서 샘플링된 점들은 표면 점이 됩니다. 이 방법은 Gaussian noise 공간에서 샘플링한 공간 점들을 표면 점으로 매핑하는 forward ordinary differential equation (ODE)를 해결하여, 기하 구조를 매우 정확하게 표현합니다. 또한, 역 매핑을 가능하게 하는 backward ODE 알고리즘을 도출하여 표면에서 노이즈 공간으로의 역산을 구현합니다.

- **Performance Highlights**: 우리의 기법은 다양한 객체 유형에서 정성적 및 정량적 평가를 통해 높은 기하학적 충실도를 달성하는 데 성공했습니다. GeomDist는 텍스처가 있는 메쉬 표현, 신경망 표면 압축, 동적 객체 모델링 및 랜더링 등 다양한 응용 프로그램에서의 가능성을 탐색하며 3D 기하학 학습의 중요성을 강조합니다.



### Language Driven Occupancy Prediction (https://arxiv.org/abs/2411.16072)
- **What's New**: 본 논문에서는 LOcc라는 효과적이고 일반화 가능한 프레임워크를 소개합니다. 기존의 접근법들이 주로 이미지 기능을 통해 이미지를 간접적으로 관리하는 거칠은 voxel-to-text 일치를 통해 네트워크를 감독한 반면, LOcc는 3D 언어 점유율의 학습을 안내하기 위해 세밀한 3D 언어 점유율 정답을 생성하는 의미론적 전이 레이블링 파이프라인을 제안합니다. 이를 통해 노동 집약적인 인간 주석을 줄이면서 더 정확한 의사 라벨링된 정답을 생성할 수 있습니다.

- **Technical Details**: LOcc는 복잡한 어휘 세트를 사용하는 open-vocabulary occupancy 예측을 위한 새로운 방법론입니다. 이 프레임워크는 기존의 예측 헤드를 기하학 헤드와 언어 헤드로 대체하여 이진 점유 상태와 언어 특성을 예측합니다. 또한, CLIP 임베딩과 같은 고차원 언어 특성을 효율적으로 정렬하기 위해 텍스트 기반 오토인코더를 도입하여 계산 비용을 줄입니다.

- **Performance Highlights**: LOcc는 다양한 아키텍처에서 검증되었으며, 모든 모델이 Occ3D-nuScenes 데이터셋에서 기존의 최첨단 zero-shot 점유 예측 방법들을 일관되게 초월하고 있다는 것을 보였습니다. 특히, 간단한 LOcc-BEVDet 모델은 256×704의 입력 해상도에서 mIoU 20.29를 달성하여 기존 방법들을 초과하였고, 이는 시각적 입력, 고해상도 입력 또는 더 큰 백본 네트워크에 의존하는 방법들과 비교할 때 인상적인 성과로 평가됩니다.



### Multi-Granularity Class Prototype Topology Distillation for Class-Incremental Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.16064)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 레이블이 없는 타겟 데이터가 점진적으로 전송되는 Class-Incremental Source-Free Unsupervised Domain Adaptation (CI-SFUDA) 문제를 탐구합니다. 기존 방법의 한계를 극복하기 위해 Multi-Granularity Class Prototype Topology Distillation (GROTO) 알고리즘을 제안하며, 이를 통해 레이블이 없는 클래스 증가 타겟 도메인에 소스 지식을 효과적으로 전이하는 방법을 제시합니다. 이 알고리즘은 다중 세분화 클래스 프로토타입 자기 조직화 모듈과 프로토타입 토폴로지 증류 모듈을 포함하고 있습니다.

- **Technical Details**: CI-SFUDA 문제는 유사한 소스 클래스의 지식이 타겟 클래스의 표현 학습에 미치는 방해와 새로운 타겟 지식이 기존 지식에 미치는 충격이라는 두 가지 주요 과제가 있습니다. 본 논문에서는 긍정 클래스 프로토타입을 이용해 소스 및 타겟 피쳐 공간의 토폴로지 구조를 구성하고, 이를 통해 새로운 타겟 지식이 기존 지식에 미치는 영향을 최소화하는 방법을 논의합니다. 실험 결과, 제안된 방법이 세 가지 공개 데이터 셋에서 최첨단 성능을 달성했음을 보여줍니다.

- **Performance Highlights**: 제안된 GROTO 알고리즘은 긍정 클래스의 의존성을 최소화하고 레이블 없는 타겟 데이터의 자기 조직화 기능을 촉진합니다. 기존의 SFUDA 및 SF-UniDA 방법들보다 효과적으로 새로운 타겟 클래스를 학습하면서도 이전 지식을 유지할 수 있는 솔루션을 제공합니다. 실험을 통해 GROTO는 기존의 기법보다 우수한 성능을 발휘하며, 특히 레이블 없는 데이터가 점진적으로 수집되는 상황에서도 안정적인 성능을 유지합니다.



### Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training (https://arxiv.org/abs/2411.16061)
- **What's New**: 본 연구는 스파이킹 신경망(Spiking Neural Networks, SNNs)의 효율성을 높이고 전통적인 인공 신경망(Artificial Neural Networks, ANNs)과의 성능 격차를 해소하기 위해 스파이크 발사 근사(SPike Firing Approximation, SFA) 방법을 제안합니다. 이 방법은 정수를 활성화로 사용하며, 훈련 시 스파이크 신호를 효과적으로 생성하도록 설계되었습니다. 이를 통해 SNN의 훈련 효율성을 높이고 전력 소비를 줄이며 성능을 향상시킵니다.

- **Technical Details**: 스파이크 신경망(SNN)은 이산적인 신호 처리 방식을 기반으로 합니다. 이 연구에서는 스파이킹 뉴런의 이진 발화로 인한 고유한 단점을 해결하기 위해 SFA 방법을 도입했습니다. SFA는 훈련 시 활성화를 정수로 변환하고, 추론 시 스파이크 시퀀스로 변환되어, 구조적 유연성과 시간적 역동성을 동시에 최적화합니다. 또한, 효율적인 스파이크 기반 Transformer 아키텍처인 E-SpikeFormer가 제안되어 성능 향상과 에너지 효율을 동시에 추구합니다.

- **Performance Highlights**: 제안하는 방법은 ImageNet-1k에서 10M 파라미터 모델로 78.5%의 정확도를 달성하며, 기존 최고 성능의 SNN 모델보다 7.2% 성능이 개선되었습니다. 또한, 훈련 속도는 4.5배, 추론 에너지 효율성은 3.9배 향상되어, 다양한 작업에서 SNN의 효과성과 효율성이 검증되었습니다. 이러한 발전은 SNN이 ANN과 유사한 성능을 유지하면서도 저전력의 장점을 유지한다는 점에서 큰 의의가 있습니다.



### UnitedVLN: Generalizable Gaussian Splatting for Continuous Vision-Language Navigation (https://arxiv.org/abs/2411.16053)
- **What's New**: 이번 연구에서는 Vision-and-Language Navigation (VLN) 분야에서 새로운 paradigm인 UnitedVLN을 소개합니다. UnitedVLN은 3D Gaussian Splatting (3DGS) 기술을 기반으로 하여, 높은 품질의 360도 시각 이미지와 의미적 특징을 통합하여 에이전트가 미래 환경을 더 효과적으로 탐색할 수 있게 합니다. 이 방식은 기존의 기술들에 대한 의존도를 줄이고, 탐색의 효율성을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: UnitedVLN은 두 가지 주요 방식을 사용합니다. 첫 번째는 Search-Then-Query (STQ) 샘플링 방식으로, 이 방법은 인접 포인트를 검색하고 K-최근접 이웃을 쿼리하는 과정으로 구성됩니다. 두 번째는 Separate-Then-United (STU) 렌더링 방식인데, 이는 NeRF를 사용해 고수준 의미적 특징을 렌더링하고, 3DGS를 통해 시각 정보를 결합하여 다양한 환경에서의 탐색력을 강화합니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, UnitedVLN은 기존 VLN-CE 벤치마크에서 최첨단 방법들을 능가하는 성능을 보여줍니다. 이 연구는 VLN-CE의 복잡한 환경에서 더욱 효율적이고 강력한 탐색 성능을 달성할 수 있음을 입증하며, 멀티모달 접근 방식을 통해 에이전트의 내구성을 증대시킬 수 있음을 시사합니다.



### ROADS: Robust Prompt-driven Multi-Class Anomaly Detection under Domain Shif (https://arxiv.org/abs/2411.16049)
Comments:
          Accepted to the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025)

- **What's New**: 최근 이상 탐지(anomaly detection) 분야는 Multi-class Unified Anomaly Detection (MUAD) 방식으로 전환되고 있으며, 기존의 단일 클래스 모델 접근법보다 더 확장 가능하고 실용적인 대안을 제공합니다. 그러나 여전히 여러 클래스 간 간섭(inter-class interference) 문제가 존재하고, 도메인 변화(domain shifts)에 매우 민감하여 실제 적용에서 성능 저하가 발생합니다. 본 논문에서는 이러한 문제를 해결하기 위해 ROADS라는 새로운 강인한(prompt-driven) MUAD 프레임워크를 제안합니다.

- **Technical Details**: ROADS는 계층적(class-aware) 프롬프트 통합 메커니즘을 활용하여 클래스별 정보가 다이나믹하게 부여되어 이상 탐지기(anomaly detector)에 통합됩니다. 이를 통해 이상 클래스 간의 간섭을 줄이도록 설계되었습니다. 또한 ROADS는 도메인 변화를 견딜 수 있도록 도메인 어댑터(domain adapter)를 포함하여, 도메인 불변 표현(domain-invariant representations)을 학습합니다. 이러한 요소들은 모델의 Robustness를 강화하는 데 기여합니다.

- **Performance Highlights**: MVTec-AD 및 VISA 데이터셋에서의 광범위한 실험을 통해 ROADS가 최신의(state-of-the-art) 방법들을 초월하여 이상 탐지 및 위치 확인(anomaly localization)에서 뛰어난 성능을 보여주었음을 입증하였습니다. 특히, 분포 밖(out-of-distribution, OOD) 환경에서도 상당한 개선을 이루어냈으며, 이는 ROADS의 견고한 프레임워크와 분포 변화에 대한 적응력 덕분이라고 할 수 있습니다.



### ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration (https://arxiv.org/abs/2411.16044)
- **What's New**: 이 논문에서는 고해상도의 이미지를 효과적으로 탐색하고 정보를 캡쳐하기 위한 새로운 알고리즘, Zoom Eye를 제안합니다. Zoom Eye는 이미지의 위계적 구조를 트리 형태로 모델링하고, 각 자식 노드는 확대된 서브 패치를 나타냅니다. 이 알고리즘은 기존의 Multimodal Large Language Models(MLLMs)과 호환되며, 훈련이 필요 없어 MLLMs가 사람처럼 이미지 트리를 탐색할 수 있도록 합니다.

- **Technical Details**: Zoom Eye의 핵심 아이디어는 이미지를 하나의 트리로 설정하고, 루트 노드에서 잎 노드로의 검색을 통해 적절한 정보를 찾는 것입니다. 이는 기존의 vision encoder의 제한된 입력 해상도로 인해 대형 객체에만 초점을 맞추는 문제를 해결합니다. 또한 Zoom Eye는 모델에 구애받지 않으며, 각 MLLM이 관련된 쿼리에 대한 정확한 응답을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, Zoom Eye는 여러 고해상도 벤치마크에서 MLLMs의 성능을 크게 향상시키는 것으로 나타났습니다. 예를 들어, LLaVA-v1.5-7B는 $V^*$ Bench에서 34.57% 및 HR-Bench에서 17.88%의 성능 향상을 보였습니다. 또한, 작은 7B 모델이 GPT-4o와 같은 대형 모델보다 우수한 성능을 발휘했습니다.



### VisualLens: Personalization through Visual History (https://arxiv.org/abs/2411.16034)
- **What's New**: 이 논문에서는 개인의 일상 생활을 반영하는 이미지의 시각적 기록이 사용자의 관심사와 선호도를 이해하는 데 유용하다는 가설을 세우고, 이를 개인화에 활용할 수 있는 방법을 제안합니다. 기존의 추천 시스템들은 특정 도메인이나 텍스트 신호에 의존해왔습니다. 그러나, VisualLens라는 새로운 접근 방식을 소개하며, 이미지 표현을 추출하고 필터링하여 개인화에 활용하는 기술을 개발했습니다.

- **Technical Details**: 이 연구에서는 사용자의 사진을 활용하여 시각적 기록의 다양성과 노이즈 문제를 해결하기 위한 방식으로 VisualLens를 제안합니다. VisualLens는 사용자 순위의 정확도를 개선하기 위해 이미지의 중요한 신호를 효과적으로 추출하고, 관련 사진만 선택적으로 검색하는 방법을 사용합니다. 또한, 이미지를 텍스트 캡션과 조합하여 분석하며, 반복적인 정제 과정을 통해 사용자의 관심사를 더 잘 반영하는 추천을 제공합니다.

- **Performance Highlights**: VisualLens는 Google Review-V와 Yelp-V 벤치마크에서 82-91%의 Hit@10을 달성하였으며, 최신 추천 기술인 UniMP보다 약 10% 향상된 성과를 보였습니다. GPT-4o와 비교했을 때도 모든 메트릭에서 성능이 향상되었으며, Hit@3에서는 각각 1.6%와 4.6% 개선되었습니다. 이 모델은 기존 방법들이 실패하는 시나리오에서도 개인화된 추천의 가능성을 열어줍니다.



### From Dashcam Videos to Driving Simulations: Stress Testing Automated Vehicles against Rare Events (https://arxiv.org/abs/2411.16027)
- **What's New**: 이번 연구에서는 현실 세계의 자동차 사고 동영상을 시뮬레이션 시나리오로 자동 변환하는 새로운 프레임워크를 제안합니다. 이 방법은 Video Language Models(VLM)를 활용하여 대시캠 영상으로부터 SCENIC 스크립트를 생성, CARLA 시뮬레이터에서 사실적인 시뮬레이션 시나리오를 생성합니다. 이 접근 방식은 단순한 시나리오 재구성을 목표로 하지 않고, 원본 비디오에서 본질적인 주행 행동을 포착하며 환경 변수에 대한 유연성을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 네 가지 구성 요소로 이루어져 있습니다: 1) 현실 비디오를 SCENIC 스크립트로 변환, 2) SCENIC 스크립트로부터 시뮬레이션 비디오 생성, 3) 실제와 시뮬레이션 비디오 간 유사성 분석, 4) 시뮬레이션 비디오의 일관성을 보장하기 위한 반복적 개선. 입력 비디오는 GPT-4o 모델의 프롬프트 엔지니어링을 통해 서술적 스크립트로 변환됩니다. 이 프로세스는 각종 시나리오의 구조적이고 정확한 설명 언어로의 매핑을 가능하게 합니다.

- **Performance Highlights**: 예비 결과에 따르면 이 자동 변환 프로세스는 몇 분 만에 완료되며, 인간의 개입 없이도 높은 충실도를 유지합니다. 기존의 수작업 시나리오 구축 시간이 몇 시간에서 몇 분으로 단축되는 중대한 시간 효율성을 보여줍니다. 이 결과는 ADS의 테스트 환경을 실질적으로 개선하는 데 기여할 것입니다.



### Style-Pro: Style-Guided Prompt Learning for Generalizable Vision-Language Models (https://arxiv.org/abs/2411.16018)
Comments:
          Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025)

- **What's New**: 이번 논문은 Style-Pro라는 새로운 스타일 가이드 프롬프트 학습 프레임워크를 제안합니다. 이 프레임워크는 기존의 Vision-language (VL) 모델에서 발생하는 과적합(overfitting)을 줄이고 CLIP의 제로샷(zero-shot) 일반화 능력을 유지하는 데 중점을 두고 있습니다. Style-Pro는 학습 가능한 스타일(base) 기초에 따라 다양한 분포 변화(distribution shifts)를 합성하여 새로운 도메인 및 보지 못한 클래스에 대한 적응력을 높입니다.

- **Technical Details**: Style-Pro는 두 개의 새로운 손실 함수(loss function)를 도입하여 스타일 다양성과 콘텐츠 무결성을 보장합니다. 이 프레임워크는 새로운 스타일을 알려진 스타일 표현 공간에 가중 결합(weighted combination)하여 매핑하고, 원래의 고정된 CLIP 모델과의 일관성(consistency)을 유지하기 위한 제약을 적용합니다. 이러한 방식은 텍스트와 이미지 인코더(text and image encoders)를 정렬하여 프롬프트 모델의 학습 도중 원본 모델의 강력한 일반화 능력을 보존합니다.

- **Performance Highlights**: 11개의 벤치마크 데이터셋을 통해 실시한 포괄적인 실험에서 Style-Pro는 기초에서 새로운 일반화(base-to-novel generalization), 데이터셋 간 전이(cross-dataset transfer) 및 도메인 일반화(domain generalization) 작업에서 최신 방법들을 일관되게 초월하는 효과를 입증하였습니다. 또한 이 프레임워크는 적은 데이터로도 효율적인 학습을 가능하게 하며, 제로샷(zero-shot) 일반화 능력을 유지합니다.



### DRIVE: Dual-Robustness via Information Variability and Entropic Consistency in Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.15976)
- **What's New**: 드라이브(DRIVE)라는 새로운 SFUDA 프레임워크가 소개되었습니다. 이 방법은 두 개의 모델을 동시에 사용하는 이중 모델 아키텍처를 도입하여, 대상 도메인의 다양성과 불확실성을 효과적으로 반영합니다. 특히, 예측 불확실성을 기반으로 레이블 가중치를 조정하는 엔트로피 기반의 의사 레이블링 전략을 사용하여 모델이 신뢰할 수 있는 데이터에 중점을 두도록 합니다.

- **Technical Details**: 드라이브는 두 단계로 이루어진 적응 프로세스를 갖습니다. 첫 번째 단계에서는 상호 정보 일관성 손실을 적용하여 모델이 안정적인 특징에 대해 정렬되도록 하여 노이즈의 영향을 최소화합니다. 두 번째 단계에서는 첫 번째 단계의 손실 정보를 이용해 PGD(Projection Gradient Descent)를 통해 불확실성이 높은 영역을 탐색하며, 모델의 견고성과 적응성을 높이는 데 기여합니다.

- **Performance Highlights**: 드라이브는 표준 SFUDA 벤치마크에서 이전의 방법들을 지속적으로 초월하여 개선된 적응 정확도와 안정성을 보여주었습니다. 특히, 복잡한 대상 도메인에서의 일반화 능력과 불확실성에 대한 강건성을 향상시키면서 최적의 성능을 도출했습니다.



### CNNs for Style Transfer of Digital to Film Photography (https://arxiv.org/abs/2411.15967)
- **What's New**: 이 연구는 Cinestill-800T 필름의 시각적 효과를 디지털 이미지로 재현하기 위해 단순한 합성곱 신경망(convolutional neural networks)을 사용하는 새로운 접근 방식을 제안합니다. 다양한 손실 함수(loss functions)의 효과와 입력 노이즈 채널(input noise channel)의 추가, 훈련 중 무작위 크기의 패치(random scales of patches)를 사용하는 방법을 시험했습니다. MSE/VGG 손실의 조합이 가장 우수한 색상 생산을 나타내었으며, 고품질의 그레인(grain) 생성을 확인했으나 높은 품질의 할레이션(halation) 효과는 없었습니다.

- **Technical Details**: 연구팀은 두 카메라를 사용하여 디지털 이미지와 필름 이미지의 쌍(pair) 데이터셋을 구축했습니다. 이 데이터셋에서 각각의 이미지는 동일한 장면을 포착하기 위해 Sony Alpha 7 디지털 카메라와 Nikon F3 필름 카메라로 촬영되었습니다. 모델 아키텍처는 U-Net 기반으로 구성되었으며, 이는 이미지 분할(image segmentation) 작업을 위해 처음 소개된 네트워크입니다. 연구 팀은 keypoint alignment를 사용하여 이미지의 정렬 문제를 해결하고, CIELAB 공간에서 히스토그램 매칭(histogram matching)을 통해 이미지 쌍의 휘도를 정렬했습니다.

- **Performance Highlights**: 실험 결과, U-Net 아키텍처는 Cinestill-800T 필름의 특성을 잘 캡처하였고, 디지털 이미지를 필름 퀄리티로 변환하는 데 있어 효율적인 성능을 보였습니다. 연구팀은 최종적으로 41쌍의 이미지 데이터를 확보했으며, 이를 통해 재현된 주요 효과들은 노이즈, 그레인 및 색상 변화를 포함합니다. 데이터셋의 원본 및 처리된 모델을 공개하여 다른 연구자들이 추가 작업을 진행할 수 있도록 하였습니다.



### Gaussian Scenes: Pose-Free Sparse-View Scene Reconstruction using Depth-Enhanced Diffusion Priors (https://arxiv.org/abs/2411.15966)
Comments:
          17 pages, 6 figures, 3 tables

- **What's New**: 이번 연구에서는 제한된 수의 보정되지 않은 2D 이미지에서 포즈가 없는 360도 장면을 복원하기 위한 생성 스타일(generative approach)을 제안합니다. 기존의 기술들은 일반적으로 깊이 추정(depth estimation)이나 3D 기반 우선 사항(3D foundational priors)을 활용하여 안정성을 높였으나, 이러한 방법들은 포즈 정보가 명확히 제공되어야 하는 제한이 있었습니다. 새롭게 제안하는 RGBD diffusion 모델은 새로운 뷰 렌더링 및 깊이 맵에서 결함을 제거하고 세부 정보를 보완할 수 있도록 설계되었습니다.

- **Technical Details**: 이 방법은 Gaussian-SLAM에 영감을 받아, 점진적으로 통합하여 다중 뷰 일관성을 유지하는 Gaussian 표현을 생성하는 프로세스를 포함합니다. 또한, 새로운 신뢰도 척도를 제안하여 Gaussian 표현에서 결함 감지가 향상됩니다. GScenes는 DUSt3R를 사용해 장면의 내부/외부 파라미터를 추정하고, 소수의 반복을 통해 Gaussians 및 카메라 파라미터를 공동 최적화합니다.

- **Performance Highlights**: 실험 결과 MipNeRF360 데이터셋에서 우리의 방법이 기존의 포즈 없는 기술을 초월하며, 복잡한 360도 장면에서의 포즈 종속 재구성(state-of-the-art posed reconstruction methods)과 경쟁력을 가지고 있음을 보여줍니다. GScenes는 기존의 정규화(regenerative methods) 및 생성 우선 사항(generative priors) 기반 작업들에 비해 월등한 재구성 품질을 기록했습니다.



### MobileMamba: Lightweight Multi-Receptive Visual Mamba Network (https://arxiv.org/abs/2411.15941)
Comments:
          14 pages

- **What's New**: 이번 논문에서는 MobileMamba라는 새로운 경량화 네트워크 프레임워크를 제안합니다. CNN과 Transformer 기반 모델의 한계를 극복하기 위해 세 가지 스테이지로 구성된 네트워크를 설계하였고, 이는 효율성(efficiency)과 성능(performance) 균형을 잘 맞추고 있습니다. 또한 Multi-Receptive Field Feature Interaction (MRFFI) 모듈을 통해 장기 의존성(long-range dependency) 및 고주파 세부 정보(high-frequency detail)를 효과적으로 추출하여 최적의 성능을 달성하였습니다.

- **Technical Details**: MobileMamba는 Coarse-Grained, Fine-Grained 및 Training/Testing Strategies를 통해 설계되었습니다. MRFFI 모듈은 Long-Range Wavelet Transform-Enhanced Mamba (WTE-Mamba), Efficient Multi-Kernel Depthwise Convolution (MK-DeConv), 그리고 Eliminate Redundant Identity 구성 요소로 이루어져 있습니다. 이 모듈은 입력 특성을 채널 차원에서 세 부분으로 나누어 글로벌(gobal) 및 멀티스케일(multi-scale) 수용장 정보를 통합하여 세부 정보의 추출을 향상시킵니다.

- **Performance Highlights**: MobileMamba는 ImageNet-1K에서 Top-1 정확도(top-1 accuracy)를 83.6%로 달성하여 기존 방법들을 초월했습니다. LocalVim에 비해 GPU에서 최대 x21 빠른 속도를 기록하며, 다양한 고해상도 다운스트림 작업에서 성능을 더욱 높였습니다. Mask RCNN, RetinaNet 및 SSDLite에서 각각 +1.3, +2.1, +5.7의 개선을 보여주었으며, CNN 및 ViT 기반 모델에 비해 전반적으로 우수한 효율성과 성능을 입증했습니다.



### Segment to Recognize Robustly -- Enhancing Recognition by Image Decomposition (https://arxiv.org/abs/2411.15933)
- **What's New**: 본 논문에서는 이미지 인식에서 foreground (FG)와 background (BG)를 분리하여 모델링하는 새로운 접근법인 'Segment to Recognize Robustly' (S2R2)를 제안합니다. 기존의 방법들은 BG 정보를 희생하여 FG의 일반화를 개선하려는 경향이 있었으나, S2R2는 FG와 BG를 간단하고 강건하며 해석 가능한 방식으로 결합하여 신뢰성을 높입니다. 이 접근법은 최근 발전된 zero-shot segmentation 기술을 활용하여 인식 전후에 FG와 BG를 분리할 수 있게 합니다.

- **Technical Details**: S2R2 프로세스는 이미지 분해 모듈, FG 및 BG 모델링 모듈, 그리고 통합 모듈의 세 단계로 구성됩니다. 첫 번째 단계에서 이미지를 FG와 BG로 분해하고, 이후 이 두 가지를 조합하여 간단한 해석이 가능한 방식으로 결과를 생성합니다. 이 과정에서 모델은 대규모 이미지 데이터셋에 대해 FG-BG 분리 및 인식의 강건성을 입증하며, 표준 전체 이미지 분류기와도 함께 결합할 수 있습니다.

- **Performance Highlights**: 실험 결과, S2R2는 다양한 평가 데이터에서 최첨단 성과를 달성하며 BG 분배 변화에 대한 강건성을 유지하는 것으로 나타났습니다. 특히, zero-shot segmentation을 통한 BG 제거가 FG 모델링보다 월등한 성능을 보이며, S2R2 방식은 기존 방법보다 더 나은 결과를 보여줍니다. 이로 인해 이 접근법은 이미지 인식 분야에서 중요한 진전을 이룬 것으로 평가됩니다.



### Making Images from Images: Interleaving Denoising and Transformation (https://arxiv.org/abs/2411.15925)
- **What's New**: 이 논문에서는 이미지 요소들의 재배치를 통해 새로운 이미지를 생성하는 방법을 제안합니다. 사용자는 블록, 원형 링, 또는 개별 픽셀과 같은 다양한 형태로 지역을 정의할 수 있으며, 이러한 방식으로 기존 이미지를 획기적인 주제로 변환할 수 있습니다. 제안된 방법은 이미지의 내용과 필요한 변환을 동시에 학습하여, 최적화 문제로 형성되어 이미지를 생성합니다. 이전 방법들과는 달리, 지역의 수가 증가할수록 문제 해결이 더 쉬워지고 결과가 향상됩니다.

- **Technical Details**: 연구에서는 딥러닝 기반의 모델을 활용하여 이미지 변환 과정을 최적화 문제로 모델링하고 이미지 확산(diffusion)과 에너지 최소화 과정(minimization)을 교차 적용합니다. 기존의 정적 소스 이미지 대신 동적으로 변환을 발견할 수 있는 방법을 제시하여, 기존 데이터에서 새로운 이미지를 생성하는 것이 가능합니다. 이는 헝가리안 알고리즘(Hungarian Method)을 사용해 최적 배치를 구현하며, 무한한 소스 이미지를 활용하여 다양하고 매력적인 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 픽셀 공간과 잠재 공간(latent space) 모두에서 효과적으로 수행되는 것을 보여줍니다. 또한, 서로 다른 주제를 표현하기 위해 다수의 소스 이미지를 사용하는 창의적인 확장도 가능함을 입증하였습니다. 이러한 결과들은 다양한 이미지 생성 가능성을 제시하며, 특히 시각적 착시 효과 생성에 혁신적인 접근을 제공합니다.



### Deep Learning for automated multi-scale functional field boundaries extraction using multi-date Sentinel-2 and PlanetScope imagery: Case Study of Netherlands and Pakistan (https://arxiv.org/abs/2411.15923)
Comments:
          09 pages, To be published

- **What's New**: 이번 연구는 네덜란드와 파키스탄의 두 가지 지역에서 다중 시간대 위성 이미지를 활용한 기능적 농지 경계 구분(multi-temporal satellite imagery and functional field boundary delineation)의 효과성을 심층 학습 세분화 아키텍처(deep learning semantic segmentation architecture)를 통해 탐구했습니다. 2022년 4월, 8월, 10월의 PlanetScope 및 Sentinel-2 이미지를 수집하여 결국 작물 경계 데이터로 활용하였습니다.

- **Technical Details**: 이 연구에서는 네덜란드의 기본 등록 작물 구획(BRP) 벡터층을 레이블 학습 데이터로 사용하였고, 파키스탄에서는 자체 제작한 농지 경계 벡터 데이터를 이용하였습니다. UNET 아키텍처 모델을 통해 다양한 조합의 다중 날짜 이미지(multi-date images)와 NDVI 스택을 평가하며, IoU(Intersection over Union) 점수 비교 분석을 통해 제안된 다중 날짜 NDVI 접근법의 효과성을 검토하였습니다.

- **Performance Highlights**: 결과는 다중 날짜 NDVI 스택이 계절에 따른 작물 성장의 시간적 맥락을 제공함을 보여주었고, 게다가 소규모 농업 지역의 농지 경계 추출을 위한 높은 공간 해상도의 중요성을 강조하였습니다. 연구 결과는 이질적인 농업 환경에서 향상된 자동 농지 경계 구분을 위해 다중 규모 구현으로 확장될 수 있습니다.



### A Tunable Despeckling Neural Network Stabilized via Diffusion Equation (https://arxiv.org/abs/2411.15921)
- **What's New**: 이 논문은 다중 Gamma 노이즈 제거 문제를 해결하기 위해 새로운 접근 방식을 제안하고 있습니다. 기존의 신경망 모델은 현실 세계의 데이터 특성과 다양한 방해 요소로 인해 그 효과성이 떨어지는 문제를 안고 있습니다. 본 연구에서는 신경망의 안정성을 높이고 실제 노이즈의 저항력을 강화하기 위해 확산 방정식의 소산적 특성을 활용합니다.

- **Technical Details**: 제안된 모델은 노이즈 제거 유닛과 정규화 유닛을 포함하는 조정 가능한 정규화 신경망입니다. 이 신경망은 엔드 투 엔드 학습을 위해 단일 네트워크로 구성되며, 노이즈 제거 유닛은 노이즈 제거 네트워크로, 정규화 유닛은 가장 간단한 선형 확산 방정식으로 구성됩니다. 정규화 유닛은 네트워크의 안정성을 강화하며, 훈련 후 시간 단계 조정을 통해 적대적 공격의 부정적 영향을 효과적으로 완화합니다.

- **Performance Highlights**: 실험을 통해 제안된 모델은 시뮬레이션된 이미지, 적대적 샘플, 실제 SAR 이미지에 대해 여러 최첨단 노이즈 제거 방법과 비교하여 정량적 및 시각적 평가 모두에서 우수한 결과를 보여주었습니다. 이 모델은 특히 SAR 이미지의 다중 Gamma 노이즈 제거에 강한 성능을 발휘함으로써, 실제 응용 가능성에서 우수한 평가를 받고 있습니다.



### Highly Efficient and Unsupervised Framework for Moving Object Detection in Satellite Videos (https://arxiv.org/abs/2411.15895)
Comments:
          8 pages, 8 figures

- **What's New**: 이번 연구에서는 매우 효율적인 비지도 학습 프레임워크인 HiEUM(Highly Efficient and Unsupervised Moving object detection)을 제안합니다. 이 방법은 pseudo labels(가짜 라벨)을 전통적인 방법으로 생성하고 학습 과정에 따라 이를 발전시켜 SVMOD(위성 비디오에서의 움직이는 객체 감지)의 성능을 향상시킵니다. 또한, 희소 샘플링을 이용한 sparse convolutional anchor-free detection network를 통해 배경 영역에서 불필요한 계산을 건너뛰며 효율성을 극대화할 수 있습니다.

- **Technical Details**: HiEUM 프레임워크는 두 가지 핵심 설계 요소를 기반으로 합니다. 첫째, 라벨 자가 발전(self-evolution) 비지도 학습 프레임워크를 개발하여 주석 비용을 줄이고 비지도 SVMOD의 가능성을 연구합니다. 둘째, 희소 스페이셜-템포럴 포인트 클라우드(spatio-temporal point cloud) 표현을 기반으로 희소 샘플링 기법을 적용하여 배경의 중복 계산을 건너뛰고 효율성을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 1024x1024 이미지에서 초당 98.8 프레임을 처리할 수 있으며 상태최고 성능(SOTA)도 달성할 수 있습니다. 실험 결과, SOTA DSFNet과 비교해 약 28.7배, 전통적인 B-MCMD 방법과 비교해 약 4490배의 추론 속도를 개선했습니다. 이러한 결과는 SVMOD의 새로운 방향을 제시하며, 다양한 메서드에 대한 새로운 기준을 제공합니다.



### Optimization-Driven Statistical Models of Anatomies using Radial Basis Function Shape Representation (https://arxiv.org/abs/2411.15882)
- **What's New**: 이번 논문에서는 기존 딥 러닝 방법을 활용한 형태 모델링(PMS) 기술에 대한 대안을 제안합니다. 이 방법은 고전적인 최적화 접근법을 통해 형태 모델에 대한 세밀한 제어를 가능하게 하고, 형태의 주요 특성을 더 잘 포착할 수 있도록 추진됩니다. 또한, 제안된 방법은 블랙박스 모델을 피하며, 입자들이 지오메트리 아래에서 자유롭게 탐색할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 통계적 형태 모델(SSM)의 일관성을 유지하면서 각 개별 표면을 정확하게 표현하는 데 중점을 두고 있습니다. 이를 위해 네 가지 손실을 최적화하여 RBF(방사 기저 함수) 형태를 활용하여 SSM을 구축합니다. 최적화 과정에서는 세그멘테이션 형태로 주어진 집단의 형태에 대한 일련의 제어 점을 정의하고, 이를 통해 전체 표면을 재구성하는 방법이 설명됩니다.

- **Performance Highlights**: 이 방법은 실제 데이터 세트에서 PSM 및 Image2SSM과 유사하거나 우수한 성능을 보이는 것으로 입증되었습니다. 제안된 손실 함수들은 상황에 맞는 PDM(점 분포 모델)을 구축하는 데 기여하며, 이 방법은 ShapeWorks라는 오픈 소스 패키지에 통합될 예정입니다. 손실 함수들 간의 상호작용도 개선되어, 형태 변형을 더 효과적으로 포착하는 데 도움을 줍니다.



### Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation (https://arxiv.org/abs/2411.15869)
- **What's New**: 최근 CLIP와 같은 사전 학습된 비전-언어 모델의 발전으로 오픈 어휘 분할(Open-Vocabulary Segmentation) 작업이 가능해졌습니다. 그러나 CLIP는 이미지 수준에서 사전 학습이 이루어져 지역 세부사항을 포착하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 Self-Calibrated CLIP (SC-CLIP)이라는 훈련이 필요 없는 방법을 제안하며, CLIP의 원래 일반화 능력을 유지하면서도 더 세밀한 표현을 생성합니다.

- **Technical Details**: SC-CLIP은 anomaly tokens을 식별하고 해결하여 클립의 피쳐 표현을 향상시키는 방법입니다. 이를 위해 Local Outlier Factor (LOF) 알고리즘을 적용하여 anomaly tokens을 찾고, 이들을 주변 토큰의 값을 사용해 보간하여 교체합니다. 또한, 중간 레이어의 세부 특성을 활용하여 깊은 피쳐를 적응적으로 집합시키고 주의 상관 관계를 향상시킵니다.

- **Performance Highlights**: SC-CLIP은 여덟 개의 의미 세분화 데이터셋에서 최신 성과를 달성하며, 이전 방법들보다 9.5% 더 우수한 결과를 기록했습니다. 특히, SC-CLIP은 원래 CLIP ViT-L/14 모델의 성능을 6.8배 향상시켰습니다. 이 방법은 추가적인 매개변수나 데이터 없이도 뛰어난 성능 개선을 이루어냈습니다.



### PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs (https://arxiv.org/abs/2411.15867)
- **What's New**: 이 논문에서는 파노라마 이미지 생성(panoramic image generation)을 새로운 next-token prediction 작업으로 재정의한 PanoLlama라는 새로운 프레임워크를 소개합니다. 기존의 diffusion model들이 가진 한계를 극복하고, PanoLlama는 전이 학습된 LlamaGen 구조를 활용하여 단순하면서도 효과적으로 이미지를 자동 회귀적으로 생성합니다. 이를 통해 해상도 한계를 처리하는 확장 전략을 개발하였고, 이미지 토큰 구조에 맞춰 훈련 없이 고품질의 파노라마를 생성할 수 있습니다.

- **Technical Details**: PanoLlama는 고유한 AR(autoregressive) 모델을 사용하여 연속적인 이미지 특징을 이산 토큰(token)으로 양자화하고, 이미지 토큰 시퀀스를 next-token prediction 방식으로 생성하는 방법론을 소개합니다. 파노라마 생성을 위한 효율적인 딥러닝 기법으로, 여러 해상도와 레이아웃을 지원하며, 실험 결과에서 주요 효율성과 호환성을 입증하였습니다. 이 과정에서 다단계 일관성(multilevel-coherence) 문제와 구현 복잡성(implementation complexity)을 최소화할 수 있었습니다.

- **Performance Highlights**: PanoLlama는 다양한 측면에서 여섯 가지 기준선과 비교하여 coherence, diversity, compatibility, 효율성 등에서 우수한 성능을 보였습니다. 특히, 다중 해상도와 다중 레이아웃 지원을 통해 뛰어난 유연성을 제공하고, diffusion 기반 방법들이 해결하지 못했던 문제를 효과적으로 극복했습니다. 이 새로운 패러다임은 파노라마 이미지 생성 작업에 대한 새로운 기준을 제시합니다.



### Generalizable Single-view Object Pose Estimation by Two-side Generating and Matching (https://arxiv.org/abs/2411.15860)
Comments:
          Accepted by WACV 2025, not published yet

- **What's New**: 이번 논문에서는 단일 RGB 이미지 만으로 객체의 자세를 추정할 수 있는 새로운 방법을 제시합니다. 기존의 방법들이 많은 훈련 데이터를 요구했으나, 본 방법은 보지 않은 객체에 대한 일반화를 제공하며 3D 모델이나 다수의 시점이 필요 없습니다. Diffusion 모델을 활용하여 새로운 시점의 이미지를 생성하고, 이를 통해 두 방향의 매칭을 수행합니다.

- **Technical Details**: 우리는 두 방향 매칭 모듈을 도입하여 객체 자세 추정의 어려움을 해결합니다. 큰 시점 변화를 다룰 때 생성 결과의 질이 떨어지기 때문에, 여러 개의 작은 시점 변화를 사용하여 큰 시점 변화를 근사하는 점진적인 접근 방식을 채택했습니다. 이를 통해 성능을 크게 개선하였으며, E2VG와 IDPose와의 비교를 통해 효과를 검증했습니다.

- **Performance Highlights**: 정량적인 실험 결과, 우리 방법은 기존의 자세 추정 기술들보다 뛰어난 성능을 보여 주었으며, 특히 큰 시점 변화가 있을 때에도 강력한 성능을 유지합니다. 실제 데이터셋 NAVI와 합성 데이터셋 GSO에서 SOTA 방법들을 큰 격차로 능가하는 결과를 나타내었습니다. AR 응용 프로그램에의 통합 사례도 보여주어 실용적인 적용 가능성을 강조하였습니다.



### SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition (https://arxiv.org/abs/2411.15858)
- **What's New**: 본 논문에서는 SVTRv2라는 새로운 CTC 모델을 제안하여 기존의 인코더-디코더 기반 방법들(EDTRs)을 초월하는 정확도 및 추론 속도를 보여줍니다. SVTRv2는 텍스트 불규칙성을 처리하고 언어적 맥락을 활용하는 기능을 후원하는 여러 가지 혁신적인 업그레이드를 도입하여 복잡하고 다양한 텍스트 사례를 다루는 능력을 갖추고 있습니다. 이 모델은 Multi-Size Resizing (MSR) 전략과 Feature Rearrangement Module (FRM)을 통해 텍스트 항목을 조정하고, Semantic Guidance Module (SGM)을 통해 언어 정보를 통합하여 인식 정확도를 높입니다.

- **Technical Details**: SVTRv2는 복잡한 구조를 유지하면서도 빠른 추론 속도를 보장합니다. 먼저, Multi-Size Resizing (MSR) 기법을 도입하여 텍스트의 비율에 따라 이미지를 적응형으로 조정함으로써 읽기 쉬운 형식을 유지합니다. 또한 Feature Rearrangement Module (FRM)을 통해 시각적 특징들을 재배치하여 CTC의 정렬 문제를 해결합니다. 마지막으로, Semantic Guidance Module (SGM)은 훈련 동안 주변 문자열 맥락을 활용해 언어적 정보를 통합하여 인식 성능을 향상시킵니다.

- **Performance Highlights**: SVTRv2는 표준 및 최근의 복잡한 벤치마크에서 24개의 주류 STR 모델과 비교되어 모든 평가 시나리오에서 정확성과 속도 면에서 우수한 성능을 발휘했습니다. 특히 텍스트 불규칙성, 다양한 언어, 긴 텍스트 등 여러 다양한 조건에서의 성능이 개선되었습니다. 이 결과는 SVTRv2가 기존 EDTRs에 비해 넓은 응용 가능성을 가지고 있으며 효과적인 접근 방식임을 입증합니다.



### ResCLIP: Residual Attention for Training-free Dense Vision-language Inferenc (https://arxiv.org/abs/2411.15851)
- **What's New**: 이번 연구에서는 CLIP의 비최종 계층에서의 자기 주의(self-attention)가 로컬라이제이션(localization) 특성을 보인다는 사실을 밝혔습니다. 제안된 Residual Cross-correlation Self-attention (RCS) 모듈은 중간 계층의 교차 상관(self-correlation) 주의를 활용하여 최종 블록의 주의를 리모델링합니다. 또한, Semantic Feedback Refinement (SFR) 모듈을 도입하여 주의 점수를 조정합니다. 이러한 방법은 CLIP의 조밀한 비전-언어 추론(dense vision-language inference)에서의 성능을 크게 향상시킵니다.

- **Technical Details**: Residual Cross-correlation Self-attention (RCS) 모듈은 CLIP의 중간 계층에서의 교차 상관 주의를 활용하여 최종 계층의 주의를 조정합니다. 이는 공간 정보를 재편성하고, CLIP의 로컬라이제이션 잠재력을 발휘합니다. Semantic Feedback Refinement (SFR) 모듈은 의미 분할 지도(semantic segmentation maps)를 기반으로 하여 동종 카테고리 및 지역 일관성(local consistency)의 초점을 강화합니다. ResCLIP이라는 이름의 이 방법은 기존의 접근 방식에 쉽게 통합될 수 있는 플러그 앤 플레이(plug-and-play) 모듈입니다.

- **Performance Highlights**: 여덟 가지 분할 벤치마크에서의 광범위한 실험을 통해 ResCLIP은 기존의 최첨단(trained-free) 방법들보다 우수한 성능을 보였습니다. 이 방법은 SCLIP, ClearCLIP 및 NACLIP과 통합되어 일관된 mIoU(miss Intersection over Union) 증가를 보여줍니다. 예를 들어, ResCLIP을 사용하면 모든 모델에서 성능이 1.7%에서 13.1%까지 개선되었습니다. 이러한 결과는 제안된 방법의 효과를 입증하며, 정량적 및 정성적 방법으로 다양한 분석이 수행되었습니다.



### Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing (https://arxiv.org/abs/2411.15843)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 flow transformer의 잠재력을 활용하여 튜닝 없는 이미지 편집을 가능한 혁신적인 방법으로 제안합니다. 특히, Euler 인버전의 구조적 유사성을 DDIM과 비교하여 더 높은 근사 오류의 영향을 받을 수 있음을 밝혔습니다. 이 문제를 해결하기 위해 두 단계의 인버전을 제안하여 초기 속도 추정을 정교화하고 나머지 오류를 보완하는 방식으로 이미지 편집의 질을 향상시킵니다.

- **Technical Details**: 주요 기술적 기여는 flow transformer에서의 inversion 및 invariance 제어의 체계적 분석입니다. 첫 번째로, 두 단계의 flow inversion을 통해 Euler 방법의 속도 근사 오차를 줄이는 방안을 제시하였습니다. 두 번째로는 adaptive layer normalization(AdaLN)을 기반으로 한 유연한 invariance 제어 메커니즘을 도입하여 이미지와 텍스트 간의 관계를 조절합니다.

- **Performance Highlights**: 여러 편집 타입에 대한 실험을 통해 제안된 방법이 flow transformer의 강력한 Priors를 활용하여 다재다능한 튜닝 없는 이미지 편집을 실현할 수 있음을 입증하였습니다. 이로 인해 변화된 텍스트 프롬프트와 이미지 의미 연결을 통해 다양한 편집 작업을 정확하게 수행할 수 있는 가능성이 열렸습니다.



### VaLiD: Mitigating the Hallucination of Large Vision Language Models by Visual Layer Fusion Contrastive Decoding (https://arxiv.org/abs/2411.15839)
Comments:
          15 pages

- **What's New**: 대형 비전-언어 모델(LVLM)은 멀티모달 과제에서 뛰어난 성능을 보여주었으나, 종종 시각적 콘텐츠를 정확하게 반영하지 않는 허상(hallucination) 응답을 생성하는 문제를 안고 있다. 최근의 연구에서는 디코딩 전략을 조정하여 이러한 허상을 완화하는 훈련 없는 방법을 제안했지만, 본 논문에서는 시각적 인코딩 과정의 왜곡이 모델의 추론 정확성을 크게 영향을 미친다는 것을 발견했다. 이에 따라, 본 연구에서는 시각적 인코딩 관점에서 허상 완화 방법인 VaLiD(Visual Layer Fusion Contrastive Decoding)를 제안한다.

- **Technical Details**: VaLiD는 불확실성을 활용하여 선정된 시각적 숨겨진 레이어를 통해 인코딩 과정의 왜곡을 수정하며, 이를 통해 생성된 텍스트의 신뢰성을 높인다. 구체적으로, 모델은 표준 시각 출력 레이어의 원래 분포와 시각적 숨겨진 레이어에서 유도된 참조 분포를 비교하는 대조 디코딩을 수행하며, 이는 시각 정보의 왜곡을 스스로 수정하고 정확한 디코딩 확률을 증가시킨다. 이를 통해 기존 LVLM 모델의 한계를 극복하고, 시각 정보 손실을 줄일 수 있다.

- **Performance Highlights**: VaLiD 방법은 다양한 벤치마크에서 허상을 효과적으로 감소시키며, 여러 기준선 방법과 비교하여 최첨단 성능을 달성하였다. 실험 결과는 시각적 인코딩 과정의 수정이 LVLM의 응답 신뢰성 및 정확성을 크게 향상시킬 수 있음을 보여준다. 이 연구는 LVLM의 허상을 줄이기 위한 새로운 접근 방식을 제시하여, 실제 응용 사례에서의 활용 가능성을 높인다.



### Modality Alignment Meets Federated Broadcasting (https://arxiv.org/abs/2411.15837)
- **What's New**: 새로운 연구에서는 이질적인 데이터 환경에서 페더레이티드 러닝(Federated Learning, FL)의 성능 유지를 위해 모달리티 정렬(modality alignment)을 활용한 새로운 FL 프레임워크인 FedAlign을 제안합니다. 이 프레임워크에서는 서버에 텍스트 인코더가 위치하고, 로컬 장치에서 이미지 인코더가 작동하며, 클라이언트 간의 학습 코어를 수치적으로 정렬하는 방식으로 아키텍처가 구성됩니다. 이는 다중 모달 학습 paradigms에서 영감을 받아, 서버-클라이언트 통신을 다중 모달 방송처럼 처리합니다.

- **Technical Details**: FedAlign의 핵심은 각 클라이언트가 로컬 이미지 인코더의 LoRA 파라미터를 최적화하여 중앙 서버에서 다운로드한 글로벌 텍스트 피처와 교차 엔트로피 손실(Cross-Entropy Loss)과 직교 손실(Orthogonality Loss)을 통해 학습하는 것입니다. 이를 통해 클라이언트는 세 가지 파라미터를 서버에 업로드하여 지식을 공유하며, 서버는 쿼리 기반 집계 메커니즘을 이용하여 글로벌 텍스트 인코더를 미세 조정합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행된 실험 결과, FedAlign은 강력한 일반화 및 견고성을 유지하며, 이질적인 데이터 환경에서도 양호한 성능을 나타냅니다. 학생 할당에서 클라이언트 특정 LoRA 파라미터가 효과적으로 줄어들어, 과적합을 예방하고 성능 효율성을 극대화하는 것을 보여주었습니다.



### FastTrackTr:Towards Fast Multi-Object Tracking with Transformers (https://arxiv.org/abs/2411.15811)
- **What's New**: 이 논문의 주요 혁신은 FastTrackTr이라는 새로운 다중 객체 추적(MOT) 방법을 제안하는 것으로, Transformer 아키텍처에 기반하여 빠른 추론 속도를 유지하면서 높은 정확성을 달성합니다. 또한, 과거 경로 정보를 통합하는 크로스 디코더 메커니즘을 도입하여 추가 쿼리나 디코더 없이도 효과적으로 성능을 개선합니다. 여러 벤치마크 데이터셋에 대한 광범위한 실험을 통해 기존 SOTA 방법들과 경쟁력 있는 정확성을 유지하면서도 추론 속도를 비약적으로 향상시켰음을 입증하였습니다.

- **Technical Details**: FastTrackTr의 전체 구조는 기본 디코더와 역사 인코더를 포함하여, 초기 역사 정보를 설정하는데 사용됩니다. 이어서 역사 디코더는 추가적인 크로스 어텐션 레이어를 포함하여 이전 프레임의 정보를 처리합니다. 이 방법은 JDT 및 FairMOT와 유사한 원리를 사용하며, 객체의 외관 임베딩을 얻기 위한 ID 임베딩 헤드를 통합하여 추적과 매칭을 동시에 수행합니다.

- **Performance Highlights**: FastTrackTr는 여러 벤치마크 데이터셋에서 경쟁력 있는 정확성을 나타내며, 특히 DanceTrack, SportsMOT, MOT17에서 우수한 성능을 보여줍니다. 실험 결과, 이 모델은 추적 중 쿼리 수를 줄임으로써 모델의 복잡성을 최소화하고 실시간 추적이 가능하도록 설계된 점이 특징입니다. 추가적으로, 이 방법은 대규모 배치 사이즈를 설정할 수 있어 훈련이 상대적으로 용이하다는 장점도 가집니다.



### LRSAA: Large-scale Remote Sensing Image Target Recognition and Automatic Annotation (https://arxiv.org/abs/2411.15808)
Comments:
          arXiv admin note: text overlap with arXiv:2411.07802

- **What's New**: 이번 논문은 대규모 원격 감지 이미지에서 물체 인식 및 자동 레이블링을 위한 LRSAA 방법을 제안합니다. 이 방법은 YOLOv11 및 MobileNetV3-SSD 물체 탐지 알고리즘을 앙상블 학습(ensemble learning)하여 모델 성능을 개선합니다. 또한, Poisson Disk Sampling 기법을 활용하여 세분화(segmentation) 과정을 최적화하고, EIOU 메트릭(metric)을 통해 훈련 및 추론 과정을 향상시킵니다.

- **Technical Details**: LRSAA 모델은 Ensemble Learning을 활용하여 MobileNetV3-SSD와 YOLOv11을 통합합니다. 또한, Enhanced Non-Maximum Suppression(NMS) 기법을 적용하여 EIOU 메트릭을 사용하여 성능을 개선하고, Poisson Disk Sampling을 통해 데이터셋을 효율적으로 분할하여 물체 인식을 정확하게 수행합니다. 훈련 과정에서는 XView 데이터셋을 사용하였으며, 자동 주석을 위해 천진(Tianjin)의 원격 감지 이미지에 적용되었습니다.

- **Performance Highlights**: 제안된 방법은 기존 솔루션과 비교하여 정확도와 속도 모두에서 획기적인 개선을 보였습니다. 실험을 통해 이 모델이 원격 감지 이미지 분석 방법론을 발전시킬 가능성이 있음을 입증했습니다. 특히, 합성 데이터의 사용이 자동 주석 모델의 인식 능력을 크게 향상시킬 수 있음을 확인했습니다.



### Symmetric Perception and Ordinal Regression for Detecting Scoliosis Natural Imag (https://arxiv.org/abs/2411.15799)
Comments:
          This paper has been accepted by Applied Intelligence

- **What's New**: 이 연구에서는 전통적인 방사선 검사 대신 자연 이미지를 활용하여 척추측만증(scoliosis) 스크리닝을 제안합니다. 특히, 인간의 등에 보이는 대칭성을 이용하여 비대칭(back asymmetry)으로 인해 발생하는 문제를 해결하고자 합니다. 척추측만증의 중증도를 일반 및 세분화된 수준에서 평가할 수 있는 이중 경로(dural-path) 네트워크를 도입하였습니다.

- **Technical Details**: 제안된 모델은 두 개의 주요 모듈로 구성됩니다: 대칭 특징 매칭 모듈(SFMM)과 서수 회귀 헤드(ORH)입니다. SFMM은 입력 이미지와 그 수평 반전 이미지를 통해 대칭 관계를 포착하며, ORH는 여러 개의 이진 분류(sub-classification) 문제로 서수 회귀를 변환하여 척추측만증의 중증도를 추정합니다. 이를 통해 척추측만증의 중증도 평가를 더 명확하게 수행할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 95.11%의 정확도로 일반 척추측만증 중증도를, 81.46%의 정확도로 세분화된 중증도를 추정합니다. 다양한 실험을 통해 기존의 척추측만증 검출 방법들과 비교했을 때 현저히 우수한 성능을 보여줍니다. 이로 인해 경제적이며 광범위한 척추측만증 스크리닝을 위한 유망한 해결책이 됩니다.



### Multi-Token Enhancing for Vision Representation Learning (https://arxiv.org/abs/2411.15787)
- **What's New**: 본 연구에서는 Multi-Token Enhancing (MTE) 방법을 제안합니다. 이 방법은 하나의 모델에서 여러 개의 보조 토큰을 동시에 추출하여 표현 학습을 강화합니다. 기존의 앙상블 방법들이 복잡한 계산 비용을 요구하는 반면, MTE는 최소한의 추가 학습 비용만으로 강력한 표현 인코더를 학습할 수 있도록 설계되었습니다.

- **Technical Details**: MTE는 보조 CLS 토큰과 적응적 풀링된 토큰 등의 여러 보조 토큰을 활용하여 서로 보완적인 정보를 캡처합니다. 또한, 사전 학습 기간 동안 이러한 보조 토큰에서 얻은 지식을 글로벌 토큰으로 증류하여, 추론 중에 추가 비용 없이 보조 토큰을 제거할 수 있도록 합니다. 이 방식은 자가 감독 학습에서의 일반적인 손실 함수와 아키텍처와 호환되어 효과적으로 강력한 표현을 학습할 수 있습니다.

- **Performance Highlights**: MTE는 다양한 다운스트림 작업에서 일관되게 성능을 향상시킵니다. 예를 들어, 이미지 분류, 의미 분할 및 인스턴스 분할과 같은 여러 작업에서 MTE가 더 적은 학습 에포크로 더 나은 성능을 발휘하는 것이 관찰되었습니다. 실험 결과는 보조 토큰과 글로벌 토큰 간의 최적화가 서로를 촉진하여 성과를 증대시키는 상호 작용을 보여줍니다.



### ZeroGS: Training 3D Gaussian Splatting from Unposed Images (https://arxiv.org/abs/2411.15779)
Comments:
          16 pages, 12 figures

- **What's New**: 이번 연구에서는 ZeroGS를 제안하여, 카메라 포즈에 의존하지 않고 무작위로 촬영된 수백 장의 이미지에서 3D Gaussian Splatting (3DGS)을 훈련할 수 있게 되었습니다. 기존의 방법들은 적은 수의 이미지에서만 작동할 수 있었으나, ZeroGS는 3D 장면을 효과적으로 재구성할 수 있는 혁신적인 접근법을 사용합니다. 이 방법은 사전 훈련된 모델을 활용하여 장면 표현을 수행하며, 카메라 포즈의 초기화 및 미세 조정을 통해 최적의 결과를 도출합니다.

- **Technical Details**: ZeroGS는 3DGS에서 3D Gaussian 프리미티브의 속성을 예측하도록 모델을 확장하며, RANSAC 및 PnP를 이용하여 이미지 등록을 수행합니다. 카메라 포즈와 포인트 맵은 여러 뷰에서의 일관성을 감소시키는 손실 함수를 통해 정제됩니다. 훈련 과정은 고전적인 점진적 형태로 진행되며, 매번 여러 이미지를 배치로 등록하여 훈련을 진행합니다.

- **Performance Highlights**: LLFF, MipNeRF360, Tanks-and-Temples 데이터세트를 사용한 실험에서 ZeroGS는 최신의 포즈-프리 NeRF 및 3DGS 방법보다 더 정확한 카메라 포즈를 복원하며, 심지어 COLMAP 포즈를 사용한 3DGS보다 높은 품질의 이미지를 렌더링합니다. 이러한 결과는 ZeroGS의 효과와 차별점을 명확히 보여주며, 앞으로의 연구에 기여할 수 있는 중요한 기초를 제공합니다.



### Context-Aware Detection of Mixed Critical Events using Video Classification (https://arxiv.org/abs/2411.15773)
- **What's New**: 이 논문은 컴퓨터 비전을 통한 혼합-중요 사건(mixed-critical events) 탐지를 위한 효과적인 시스템을 제안합니다. 다양한 응용 프로그램에서의 상황(context) 이해 필요성을 강조하며, 스마트 시티 애플리케이션에서의 교통 및 화재 감지를 위한 실험을 수행합니다. 이 시스템은 상황에 따라 적절한 대응을 트리거할 수 있는 적응형 시스템을 목표로 하고 있습니다.

- **Technical Details**: 본문에서는 수학적 공식화를 통해 상황 인식 기반 사건 탐지의 문제를 다룹니다. 다양한 응용 프로그램에 따른 사건의 중요성이 어떻게 달라지는지 살펴보며, 이를 해결하기 위한 시스템 설계의 복잡성에 대해 논의합니다. 특히, 화재 및 교통 사건에 대한 맥락-aware 탐지 방법론에 대한 구체적인 요구사항과 도전과제를 설명합니다.

- **Performance Highlights**: 논문은 두 가지 주요 응용 프로그램, 즉 교통 감지 및 화재 감지를 통해 '혼합 중요 사건'의 자동 식별이 제공할 수 있는 잠재적 이점을 강조합니다. 기존 시스템들과 비교하여 제안된 시스템의 성능을 검증하고, 스마트 시티의 자동 감시 프로세스를 개선할 수 있는 방안을 제시합니다. 이러한 자동화된 시스템은 다양한 애플리케이션에서 활용될 수 있도록 설계되었습니다.



### Corner2Net: Detecting Objects as Cascade Corners (https://arxiv.org/abs/2411.15772)
Comments:
          This paper is accepted by 27th EUROPEAN CONFERENCE ON ARTIFICIAL INTELLIGENCE (ECAI 2024)

- **What's New**: 이 논문은 Corner2Net이라는 새로운 코너 기반 탐지 프레임워크를 제시합니다. 이는 코너 매칭 자유의 방식을 통해 객체 탐지를 개선하고, 객체마다의 인스턴스 세부 정보를 더 정확히 반영합니다. 또한, 이 프레임워크는 로컬라이제이션과 객체 분류를 분리하여 처리하고, 기존의 병렬 예측 방식을 계단식 예측 방식으로 변경하였습니다.

- **Technical Details**: Corner2Net은 클래스에 구애받지 않는 관련 코너 쌍을 이용해 객체를 감지하고, 이로 인해 매칭 알고리즘에 대한 의존성을 제거합니다. 또한, 각 객체에 맞는 심층 정보를 활용하여 분류 정확도를 향상시키고, RoI(Region of Interest) 기반의 특징 추출을 통해 성능을 최적화합니다. 이 프레임워크는 ResNeXt와 같은 인기 있는 백본(backbone)과 쉽게 연결될 수 있습니다.

- **Performance Highlights**: Corner2Net은 COCO 데이터셋에서 ResNeXt-101-DCN 백본을 사용할 경우 47.8%의 AP 성능을 기록하며, 기존의 모든 코너 기반 탐지기를 능가합니다. 또한, CornerNet의 기준선에 비해 2.1배 더 빠른 추론 속도를 자랑하며, CityPersons와 UCAS-AOD 데이터셋에서도 각각 36.2%, 18.0%의 성과 개선을 이뤘습니다. 이는 Corner2Net이 높은 강건성과 적용 가능성을 지니고 있음을 보여줍니다.



### Text-Guided Coarse-to-Fine Fusion Network for Robust Remote Sensing Visual Question Answering (https://arxiv.org/abs/2411.15770)
- **What's New**: 본 논문에서는 Remote Sensing Visual Question Answering (RSVQA) 분야의 새로운 접근법으로 Text-guided Coarse-to-Fine Fusion Network (TGFNet)을 제안합니다. 전통적인 optical 센서의 한계를 극복하기 위해 Synthetic Aperture Radar (SAR) 이미지를 통합하는 방식을 탐구합니다. 실시간 및 다양한 날씨의 이미징 기능을 가진 SAR의 활용은 RSVQA 성능 향상에 중요한 요소로 작용합니다.

- **Technical Details**: 제안된 TGFNet은 질문 텍스트와 다중 소스 이미지 간의 의미론적 관계를 활용하여 특징 수준에서 보완적 융합(complementary fusion)을 이끌어내는 모델입니다. 특히, Text-guided Coarse-to-Fine Attention Refinement (CFAR) 모듈은 복잡한 원거리 센서 이미지에서 질문과 관련된 핵심 영역에 집중하도록 설계되었습니다. 또한, Adaptive Multi-Expert Fusion (AMEF) 모듈을 통해 optical 및 SAR 특징의 적응형 융합이 가능합니다.

- **Performance Highlights**: 제안된 데이터 세트에서 폭넓은 실험을 통해 TGFNet이 optical 및 SAR 이미지 간의 보완 정보를 효과적으로 통합하여 어려운 조건에서 모델 성능을 크게 향상시키는 것을 입증했습니다. 이 데이터 세트는 6,008개의 잘 정렬된 optical-SAR 이미지쌍과 1,036,694개의 질문-답변 쌍으로 구성되어 있습니다. 또한, 16가지 다양한 질문 유형을 포함하여 복잡한 관계 추론 문제도 지원합니다.



### Integrating Deep Metric Learning with Coreset for Active Learning in 3D Segmentation (https://arxiv.org/abs/2411.15763)
Comments:
          To be published in NeurIPS 2024

- **What's New**: 의료 분야에서 3D 세분화를 위한 활성 학습(Active Learning, AL)의 가능성을 탐색한 연구가 부족했습니다. 기존 AL 방법들은 주로 데이터의 다양성이나 모델 불확실성에 집중하는 경향이 있으며, 의료 데이터의 자연스러운 그룹화를 활용하지 못하고 있습니다. 본 논문에서는 이러한 문제를 해결하기 위해, 깊은 메트릭 학습(deep metric learning)을 활용하여 의료 이미징의 관련 샘플 간의 차이를 강조하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 3D 세분화를 위한 활성 학습을 수행하기 위해, Coreset와 함께 깊은 메트릭 학습을 통합하였습니다. 이는 각 슬라이스(slice)에서의 데이터 그룹화를 이용하여 다양한 샘플을 선택하고 모델 학습이 가능하게 합니다. 목표는 데이터 그룹 내에서의 유사성을 활용하여, 비용 효율적이며 높은 일반화 성능을 보이는 3D 모델을 학습하는 것입니다.

- **Performance Highlights**: 논문에서 제안한 방법은 네 개의 데이터셋(의료 및 비의료)에서 비교 평가를 수행한 결과, 기존의 AL 기술보다 우수한 성능을 보였고, 낮은 주석 예산으로도 뛰어난 결과를 도출하였습니다. 이로 인해 의료 상에서의 효율적인 데이터 주석이 가능해질 것으로 기대됩니다.



### MambaTrack: Exploiting Dual-Enhancement for Night UAV Tracking (https://arxiv.org/abs/2411.15761)
- **What's New**: 새로운 연구에서는 야간 드론(UAV) 추적의 문제를 해결하기 위해 효율적인 Mamba 기반 트래커(MambaTrack)를 제안하였습니다. 이 방법은 저조도(low-light) 이미지를 개선하기 위한 Mamba-based low-light enhancer와 언어학습을 통한 크로스 모달 향상 기법을 활용합니다. 이로 인해 밤 시간의 추적 성능이 크게 향상되고, 기계적 메모리와 계산 요구사항이 감소했습니다.

- **Technical Details**: MambaTrack의 주요 입력은 비디오 프레임과 언어 프롬프트로, 저조도 이미지에서 템플릿과 검색 영역을 크롭하여 개선합니다. 제안된 mamba 기반 저조도 향상기는 이미지의 전역 향상을 이루면서도 로컬 구조를 유지할 수 있습니다. 이 연구에서는 새로운 시각-언어(Vision-Language) 야간 UAV 추적 작업을 도입하여 기존 데이터셋에 언어 프롬프트를 추가하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최신 기술(SOTA)과 비교하여 계산 속도가 2.8배 빨라지고 GPU 메모리를 50.2% 감소시킵니다. 논문은 5개의 도전적인 벤치마크에서 검증되어, 성능 우위가 입증되었습니다. 이 모든 성능 개선은 효율적인 처리로 이어지며, 야간 추적의 가능성을 크게 향상시킵니다.



### PR-MIM: Delving Deeper into Partial Reconstruction in Masked Image Modeling (https://arxiv.org/abs/2411.15746)
- **What's New**: 이번 연구에서는 Masked Image Modeling (MIM)의 효율성을 높이기 위한 새로운 접근 방식을 제안하고 있습니다. 기존의 부분 재구성 방법이 성능 저하를 초래하는 문제를 해결하기 위해 점진적 재구성 전략과 가장 먼 샘플링 전략을 도입하였습니다. 이러한 전략들은 토큰 재구성을 통해 성능을 유지하면서도 계산 비용을 줄일 수 있도록 합니다.

- **Technical Details**: 제안한 방법은 부분 재구성에서 제거된 토큰들을 효과적으로 재구성하기 위해 경량의 공간 집계 모듈을 사용합니다. 이 모듈은 고작 7.3 x 10^-3 FLOPS의 계산 비용만 소모하며, 추가적인 비용 없이 저버려진 토큰을 재구성할 수 있게 도와줍니다. 이러한 점진적 재구성 방식은 각 MIM 프레임워크에 통합이 가능하여 높은 효율성을 제공합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 50%의 패치를 버리더라도 ViT-B/16 모델에서 손실 없는 성능을 달성하면서 28%의 FLOP 및 36%의 메모리 사용량 절감을 이끌어냈습니다. 또한, 이 접근 방식은 다양한 다운스트림 작업에서도 일관된 성능을 보여주며 유연성과 효율성을 동시에 확보했습니다. 연구팀은 제안된 방법의 소스 코드를 공개할 예정이며, 다양한 MIM 프레임워크와의 통합 가능성을 보여주고 있습니다.



### PEnG: Pose-Enhanced Geo-Localisation (https://arxiv.org/abs/2411.15742)
Comments:
          8 pages, 6 figures

- **What's New**: 본 연구에서는 도시 규모의 정밀한 이미지 위치 확인을 위한 새로운 기술을 제안합니다. 기존의 방법들이 위치 정확도를 제한하는 문제를 해결하기 위해, 카메라 시점 정보를 활용하여 정확도를 서브 미터 수준으로 개선했습니다. 특히, 우리의 시스템(PEnG)은 두 단계로 구성되어 있으며, 도로 교차로를 효과적으로 예측하고 정교한 위치 추정을 수행함으로써 큰 오류 감소를 달성합니다.

- **Technical Details**: PEnG 시스템은 차례로 도시 규모의 교차 시점 위치 확인과 상대적 자세 추정을 결합하여 작동합니다. 첫 번째 단계에서는 도시 스케일 그래프에서 가장 가능성이 높은 가장자리를 예측하고, 두 번째 단계에서는 이러한 가장자리를 따라 상대적인 자세 추정을 수행합니다. 이 시스템은 평면 도로 영역 내에서 가장 최근에 관측된 도로 교차로를 효과적으로 확인하여 포지셔닝의 정확성을 높입니다.

- **Performance Highlights**: 우리의 연구는 기존 최고의 성능을 가진 모델에 비해 96.90% 오류를 줄이며, 중위 유클리드 거리 오류가 734m에서 22.77m로 감소했습니다. 특히, Manhattan과 같은 복잡한 도시에 대한 일반화 성능을 보여주며 비교적 높은 정밀도로 지역 위치 추정을 수행했습니다. 이 결과는 새로운 전반적인 프레임워크의 필요성을 실증적으로 보여주며, 이후 코드도 공개될 예정입니다.



### Proceedings of the 6th International Workshop on Reading Music Systems (https://arxiv.org/abs/2411.15741)
Comments:
          Proceedings edited by Jorge Calvo-Zaragoza, Alexander Pacha and Elona Shatri

- **What's New**: 이번 논문은 제6회 국제 음악 읽기 시스템 워크숍(WoRMS)의 논문집을 소개합니다. 특히, Optical Music Recognition 분야에서 음악을 읽기 위한 시스템을 개발하는 연구자들과 도서관원이나 음악학자와 같은 다른 연구자들을 연결하는 데 초점을 맞추고 있습니다. 올해에는 총 22개 논문이 제출되어 15개가 승인되었으며, 저자들이 요청한 몇몇 논문은 생략되었습니다.

- **Technical Details**: 워크숍에서 다룰 주요 주제로는 음악 읽기 시스템, Optical Music Recognition, 데이터셋 및 성능 평가, 음악 악보의 이미지 처리, 작가 식별, 음악 악보의 저작 및 프레젠테이션 시스템 등이 포함됩니다. 다중 모드 시스템, 음악 작성을 위한 새로운 입력 방법, 웹 기반 음악 정보 검색 서비스와 같은 최신 기술적 요소도 고려되었습니다.

- **Performance Highlights**: 이번 년도에는 온라인으로 진행되어 전 세계의 참가자들이 쉽게 참여할 수 있는 기회를 제공했습니다. 저자들에게는 WoRMS의 품질 기준을 충족하도록 수정할 수 있는 피드백을 제공하였으며, 차기 워크숍에 다시 제출할 것을 기대하고 있습니다. 또한 GitHub 조직과 YouTube 채널을 통해 연구 관련 자료를 공유하고, 과거 세션의 기록을 조회할 수 있게 하는 기회를 제공합니다.



### LTCF-Net: A Transformer-Enhanced Dual-Channel Fourier Framework for Low-Light Image Restoration (https://arxiv.org/abs/2411.15740)
- **What's New**: LTCF-Net은 저조도(低照度, low-light) 이미지 향상을 위한 새로운 네트워크 아키텍처입니다. 이 모델은 색 정보의 효과적인 분리를 위해 LAB와 YUV 두 가지 색 공간을 활용합니다. 또한, Transformer 아키텍처를 포함하여 이미지를 comprehensively 이해하며, 포뢰 변환(Fourier transform) 모듈을 도입해 출력 이미지의 밝기를 동적으로 조정합니다.

- **Technical Details**: 제안된 LTCF-Net의 구조는 조명 향상에 특화된 두 개의 분기로 나뉘며, 각각 LAB와 YUV 색 공간에서 작동합니다. 각 분기에는 Multi-Head Self-Attention(MHSA) 모듈과 Fourier Brightness Processing(FBP) 모듈이 포함되어 명암의 세부 사항을 정교하게 복원합니다. 이 구조는 조명 및 색 정보의 독립적 처리를 가능하게 하여 복잡한 디커플링 작업을 단순화합니다.

- **Performance Highlights**: LTCF-Net은 LOL-v1, LOL-v2, SID 및 SDSD와 같은 여러 데이터 세트에서 기존의 최첨단 기법보다 뛰어난 성능을 보였습니다. 실험 결과, LTCF-Net의 접근 방식은 자연스러운 색 복원과 균형 잡힌 밝기 분포를 달성하며 시각 품질을 향상시킵니다. 이 연구는 저조도 이미지 향상의 새로운 가능성을 열어주며, 경량 모델을 유지하면서 성능을 극대화하였습니다.



### AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea (https://arxiv.org/abs/2411.15738)
Comments:
          41 pages, 24 figures

- **What's New**: 이번 연구에서 제시한 AnyEdit는 이미지 편집 명령어를 처리하기 위한 포괄적인 멀티모달 데이터셋으로, 250만 쌍의 고품질 편집 데이터를 포함하고 있습니다. 기존의 데이터셋들은 낮은 품질의 데이터와 제한된 편집 유형으로 인해 복잡한 사용자 명령어를 정확하게 실행하는 데 어려움을 겪었습니다. AnyEdit는 20가지 편집 유형과 다섯 개의 도메인에 걸쳐 다양성과 품질을 보장하기 위해 초기 데이터 다양성, 적응형 편집 프로세스, 편집 결과의 자동 선택 등을 활용합니다.

- **Technical Details**: AnyEdit는 다섯 가지 클래스로 편집 지침을 분류하며, 각 클래스는 여러 작업 유형을 포함하고 있습니다. 데이터 수집 과정에서 비율의 불균형을 해결하기 위해 카운터팩추얼 합성 장면을 도입하여 실제 데이터의 개념 조합을 다양화하고, 적응형 편집 파이프라인을 통해 각 작업에 적합한 데이터 파이프라인을 선택합니다. 또한, 편집된 이미지와 원본 이미지 간 불일치를 해결하기 위한 강화를 위한 명령어 검증 필터와 이미지 평가 포스트 필터를 개발하였습니다.

- **Performance Highlights**: AnyEdit를 활용한 실험에서는 기존 SOTA 데이터셋에 비해 시각적 유사성에서 28.9% 향상되었으며, 의미적 유사성에서도 18.8% 개선되었습니다. 제안된 AnyEdit Stable Diffusion(AnySD)은 작업 인지 라우팅과 학습 가능한 작업 임베딩을 통해 다양한 편집 유형을 지원하며, MagicBrush와 Emu-Edit 벤치마크에서 새로운 기록을 세웠습니다. 이렇게 AnyEdit와 AnySD의 결합은 교육 모델의 성능을 획기적으로 향상시켰습니다.



### Enhancing Few-Shot Out-of-Distribution Detection with Gradient Aligned Context Optimization (https://arxiv.org/abs/2411.15736)
- **What's New**: 이 논문에서는 클립(CLIP)의 편향된 인식으로 인해 발생하는 그래디언트 충돌을 완화하기 위해 Gradient Aligned Context Optimization (GaCoOp)라는 새로운 방법을 제안합니다. 기존의 몇 가지 접근법들은 ID 분류와 OOD 정규화의 최적화 간 충돌 문제를 해결하지 못했습니다. GaCoOp는 각 조정 단계에서 ID 분류의 그래디언트를 분리하고 OOD 정규화 방향과의 급각으로 파라미터를 업데이트하여 이 문제를 해결합니다.

- **Technical Details**: GaCoOp의 핵심은 ID 분류 최적화를 위해 교차 엔트로피 분류 손실의 그래디언트를 측정하여 ID 분류 방향 G_{i}를 식별하는 것입니다. 마찬가지로, LoCoOp에서의 OOD 정규화 손실의 그래디언트를 사용하여 OOD 정규화 방향 G_{o}를 계산합니다. 이 최적화는 각 반복에서 해당 OOD 정규화 방향과 급각을 유지하는 방향으로 파라미터를 업데이트함으로써 진행됩니다.

- **Performance Highlights**: 실험 결과, GaCoOp는 대규모 ImageNet OOD 벤치마크에서 우수한 성능을 달성하며 ID 분류 정확도를 향상시키는 데 기여합니다. 이는 OOD 탐지 방법의 기존 한계를 극복할 수 있는 새로운 방안을 제공함으로써, 향후 연구에 큰 기여를 할 것으로 보입니다.



### Test-time Alignment-Enhanced Adapter for Vision-Language Models (https://arxiv.org/abs/2411.15735)
- **What's New**: 이 논문에서는 사전 훈련된 비전-언어 모델(VLM)을 이용한 테스트 시 적응(Test-time adaptation, TTA)의 새로운 접근법인 테스트 시간 정렬 강화 어댑터(Test-time Alignment-Enhanced Adapter, TAEA)를 제안합니다. TAEA는 테스트 샘플을 이용해 텍스트 피처를 조정함으로써 성능을 향상시킵니다. 기존의 방법들이 분류 로짓(classification logits)을 조정하는 반면, TAEA는 텍스트 피처를 보다 유연하게 조정하도록 설계되었습니다. 이를 통해, 우리는 OOD(out-of-distribution) 벤치마크에서 0.75% 평균 개선과 크로스 도메인 벤치마크에서 2.5% 평균 향상을 달성했습니다.

- **Technical Details**: TAEA는 두 가지 모듈로 구성되어 있습니다. 첫 번째는 어댑터 모듈로, 테스트 이미지에 따라 텍스트 카테고리 임베딩을 적응시키기 위해 경량화된 어텐션 블록을 도입합니다. 두 번째는 향상 모듈로, 고엔트로피(high entropy) 또는 특정 예측에 대한 편향으로 인한 예측 오류를 완화시키고 어댑터의 성능을 더욱 개선합니다. 이 어댑터 모듈을 통해, TAEA는 테스트 샘플 간의 텍스트와 이미지 피쳐의 정렬을 더욱 효과적으로 조정합니다.

- **Performance Highlights**: TAEA는 기존의 최첨단 테스트 시간 적응 방법보다 성능이 뛰어난 것으로 나타났습니다. 실험 결과는 TAEA가 OOD 및 크로스 도메인 벤치마크에서 기존 방법들에 비해 더 나은 결과를 보여주었음을 입증합니다. 특히, 비전-언어 모델을 사용하는 다양한 실제 응용에서 유효한 성능 향상을 제공하며, 학습 시간도 수용 가능한 수준에서 유지되고 있습니다.



### OccludeNet: A Causal Journey into Mixed-View Actor-Centric Video Action Recognition under Occlusions (https://arxiv.org/abs/2411.15729)
- **What's New**: 이번 논문에서는 새로운 대규모 비디오 데이터셋인 OccludeNet을 소개합니다. 이 데이터셋은 다양한 자연 환경에서의 실제 및 합성 occlusion 장면 비디오를 포함하며, 행동 인식(action recognition) 모델의 강인성을 높이기 위해 설계되었습니다. OccludeNet은 동적 추적 occlusion, 정적 장면 occlusion 및 다중 뷰 상호작용 occlusion을 특징으로 하여 기존 데이터의 공백을 메우고 있습니다. 이 데이터셋은 행동 클래스에 대한 occlusion의 영향이 다르게 나타날 수 있음을 분석하여 다양한 연구 가능성을 제시합니다.

- **Technical Details**: OccludeNet은 총 424개의 행동 클래스를 포괄하는 dataset으로, 여러 형태의 occlusion을 포함합니다. 동적 추적 occlusion(OccludeNet-D), 정적 장면 occlusion(OccludeNet-S), 그리고 단일 뷰 및 다중 뷰 상호작용 occlusion(OccludeNet-I 및 OccludeNet-M)로 구분됩니다. 연구자들은 옥클라우드의 한계를 극복하기 위해 구조적 인과 모델(structural causal model)과 Counterfactual reasoning을 기반으로 하는 Causal Action Recognition(CAR) 방법론을 제안하여 occlusion 상황에서의 모델 강인성 향상에 기여하고 있습니다.

- **Performance Highlights**: OccludeNet은 다양한 occlusion 유형을 포괄하여 실제 문제를 해결하는 데 효과적이며, 이는 행동 인식에서의 성능을 지속적으로 개선할 수 있는 중요한 기회를 제공합니다. 연구 결과에 따르면 actor 속성과 모델 예측 간의 원인 관계에서 occlusion 특성이 혼란 요인(confounder) 역할을 할 수 있음을 발견했습니다. 이로 인해 기존의 occlusion 중심 접근 방식의 한계를 극복할 수 있는 보다 현실적인 대처 방안을 제시하며, 다양한 데이터셋을 통해 모델의 강인성을 높이고 있습니다.



### GSurf: 3D Reconstruction via Signed Distance Fields with Direct Gaussian Supervision (https://arxiv.org/abs/2411.15723)
Comments:
          see this https URL

- **What's New**: 이 논문에서는 다각도 이미지를 통해 고품질 3D 지오메트리 복원의 새로운 방법인 GSurf를 소개합니다. GSurf는 Gaussian primitives에서 직접 Signed Distance Field (SDF)를 학습하는 엔드 투 엔드 방식으로, 3D Gaussian Splatting (3DGS)과 결합하여 효율적인 복원을 달성합니다. 이 접근법은 기존의 방법들이 겪던 훈련 및 렌더링 속도의 느림 문제를 해결하며, 3D 재구성 품질도 크게 향상시킵니다.

- **Technical Details**: GSurf에서는 Gaussian primitives의 opacity, scale, 및 rotation 특성을 사용하여 객체를 이산적으로 정의하며, SDF를 통해 연속적으로 객체의 지오메트리를 정의합니다. 기존 3DGS 렌더링 파이프라인에서 Gaussian primitives가 표면에 정확하게 위치할 필요가 없도록 한 프로그램을 도입하였습니다. 또한, opacity를 정규화하기 위해 엔트로피 기반 손실을 사용하여 투명한 Gaussian primitives의 영향을 최소화하고, 기하학적 단서를 통합하여 외관 모델링을 개선하는 방법을 제안합니다.

- **Performance Highlights**: GSurf는 다양한 벤치마크 데이터셋에서 실험적인 결과를 도출하여 다른 최첨단 방법과 동등하거나 그 이상으로 고품질 3D 복원을 생성하는 데 성공했습니다. 훈련 및 렌더링은 기존 방법들에 비해 훨씬 빠르며, VolSDF나 NeuS와 같은 신경 암시적(surface) 방법들과 비교해도 경쟁력 있는 결과를 제시합니다. 이러한 성능은 깊이 정보의 부족이나 노이즈 문제를 해결함으로써 안정성을 더합니다.



### Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks (https://arxiv.org/abs/2411.15720)
- **What's New**: 이번 논문에서는 Chain of Attack (CoA)라는 새로운 전이 기반(targeted) 공격 프레임워크를 제안하고 있습니다. CoA는 다중 모달(multi-modal) 의미의 업데이트를 기반으로 적대적 예제(adversarial examples)의 생성을 단계적으로 향상시키며, 이로 인해 공격의 품질과 성공률을 개선합니다. 모델의 보안성과 강인성을 평가하기 위해 블랙박스 공격을 사용하여 다양한 VLM의 취약점을 분석하는 방법론도 포함되어 있습니다.

- **Technical Details**: CoA는 기존의 적대적 공격 방식과 차별화된 점으로, 비전과 텍스트 모달 간의 의미적 상관관계를 적극 활용합니다. 기존의 방법들이 비주얼 피처에만 초점을 맞춘 반면, CoA는 텍스트 임베딩(text embeddings) 또한 고려하여 보다 정교한 적대적 예제를 생성합니다. 또한, LLM을 기반으로 한 통합적인 자동 공격 성공률(ASR) 계산 전략을 수립하여, 평가 전략을 공정하고 일관되게 만들어 갑니다.

- **Performance Highlights**: 실험 결과, 제안된 공격 전략은 기존의 블랙박스 공격 방법에 비해 더욱 높은 공격 성능을 달성하였습니다. 특히, 고차원 이미지를 사용하는 VLMs에 대한 공격 성과가 눈에 띄었으며, 주요 성능 지표에서 문맥과 의미를 반영한 방해 요소를 통한 개선이 확인되었습니다. 마지막으로, VLM의 보안성과 강건성을 평가하는 과정에서 실질적인 사용 사례에 대한 통찰력을 제공하였습니다.



### ROOT: VLM based System for Indoor Scene Understanding and Beyond (https://arxiv.org/abs/2411.15714)
- **What's New**: 최근 Vision Language Models (VLMs)는 큰 발전을 이루었지만 여전히 실내 장면에서 공간 계층적 추론 (spatial hierarchical reasoning)에 어려움을 겪고 있습니다. 본 연구에서는 실내 장면 분석을 향상시키기 위해 설계된 ROOT라는 VLM 기반 시스템을 소개합니다. 이를 통해 실내 장면 내 물체 엔티티를 감지하는 반복적인 물체 인식 알고리즘을 개발하였습니다.

- **Technical Details**: 이 연구에서는 GPT-4V를 사용하여 실내 장면에서 객체를 인식하는 알고리즘을 개발한 후, 비전 기초 모델 (vision foundation models)을 활용하여 경계 상자 (bounding boxes)와 같은 추가 메타 정보를 수집합니다. 이를 바탕으로 공간적 계층 구조를 표현할 수 있는 SceneVLM이라는 전문화된 VLM을 제안하며, 이 모델은 실내 환경 내에서 객체 간의 거리 정보 또한 제공합니다. 61만 개 이상의 이미지를 수집하여 교육을 진행하였고, semi-automated 기술을 통해 실내 물체 간의 관계를 설정하고 거리를 추정하는 장면 데이터 생성 파이프라인을 구현했습니다.

- **Performance Highlights**: 실험 결과, ROOT는 실내 장면 이해를 촉진하며 3D 장면 생성과 망가진 AI와 같은 다양한 다운스트림 애플리케이션에서 효과적인 성능을 보여주었습니다. 이 모델은 실내 장면에 대한 공간적 배열을 이해하는 데 도움을 주며, 다양한 응용 분야에서 유용하게 활용될 수 있습니다. 개발된 코드와 추가 정보는 공개될 예정입니다.



### Fixing the Perspective: A Critical Examination of Zero-1-to-3 (https://arxiv.org/abs/2411.15706)
- **What's New**: 본 연구에서는 Zero-1-to-3의 cross-attention 메커니즘에 대한 철저한 분석을 수행했습니다. 이 분석을 통해 이론적 프레임워크와 구현 간의 중요한 불일치를 발견하였고, 이는 이미지 조건식 맥락 처리에 영향을 미칩니다. 이에 대해 우리는 cross-attention 메커니즘을 효과적으로 활용할 수 있는 수정된 구현과 동시에 여러 조건 보기(view)를 활용할 수 있는 향상된 아키텍처를 제안합니다.

- **Technical Details**: 제안된 접근법은 Neural Radiance Fields (NeRF) 기반의 임베딩을 포함하여 기존의 Generative Adversarial Networks (GANs)와 Variational Autoencoders (VAEs)에 대한 이론적 배경을 강화합니다. 우리는 Diffusion Models을 사용하여 3D 장면을 연속적인 함수로 나타내며, 이를 통해 3D 공간 내에서 각기 다른 카메라 포즈에 따라 고품질의 novel view synthesis를 가능하게 합니다. 또한 우리의 연구는 기존의 diffusion UNet에서 cross-attention의 효과적인 활용을 위한 아키텍처 개선을 포함하고 있습니다.

- **Performance Highlights**: 이 연구의 초기 결과는 novel view synthesis의 일관성과 정확성이 개선될 수 있음을 시사합니다. Zero-1-to-3의 한계점을 극복하기 위해 제안된 방법론은 조건부 diffusion 모델의 실용적인 도전 과제를 조명합니다. 우리는 여러 조건 이미지를 사용하여 3D 생성을 위한 성능 개선이 이루어질 것으로 기대하고 있습니다.



### Towards RAW Object Detection in Diverse Conditions (https://arxiv.org/abs/2411.15678)
- **What's New**: 이번 연구에서는 기존의 sRGB 이미지를 사용한 객체 탐지 방법의 한계를 극복하기 위해 AODRaw 데이터셋을 제안합니다. 이 데이터셋은 고해상도의 실제 RAW 이미지 7,785장을 포함하며, 62개의 다양한 카테고리에서 135,601개의 주석이 있는 인스턴스를 제공합니다. AODRaw는 9가지의 복합적인 조명 및 날씨 조건 하에서 수집된 이미지로, 실세계의 다양한 조건에서의 객체 탐지 연구에 기여할 것입니다.

- **Technical Details**: AODRaw의 주요 목표는 악조건에서도 RAW 이미지를 기반으로 객체를 탐지할 수 있는 기계 학습 모델을 훈련하는 것입니다. 기존의 방법들은 sRGB 이미지에서 전이된 모델을 사용했으나, sRGB와 RAW 간의 도메인 갭이 문제로 제기되었습니다. 이 연구에서는 RAW 도메인에서의 사전 훈련(pre-training)을 통해 이 도메인 갭을 줄이고, 노이즈를 포함한 복잡한 데이터를 더욱 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: AODRaw 데이터셋을 통해 기존의 객체 탐지 방법을 평가한 결과, sRGB 기반 탐지 모델은 34.0%의 평균 정밀도(AP)를 기록했으나, RAW 기반 탐지 모델은 사전 훈련을 통해 34.8%의 AP를 달성하여 성능이 향상되었습니다. 이는 다양한 조건에서의 성능 개선을 이끌어낼 수 있음을 의미하며, 객체 탐지의 새로운 가능성을 제시합니다.



### Semantic Shield: Defending Vision-Language Models Against Backdooring and Poisoning via Fine-grained Knowledge Alignmen (https://arxiv.org/abs/2411.15673)
Comments:
          CVPR 2024

- **What's New**: 최근 몇 년 간 vision-language 모델은 자가 감독 목표를 사용하여 훈련받는 것에 대한 큰 관심을 받았습니다. 그러나 대규모 웹 데이터셋을 사용한 훈련은 이러한 모델이 backdooring 및 poisoning 공격과 같은 잠재적 보안 위협에 취약하게 만듭니다. 이번 논문에서는 contrastively trained vision-language 모델을 보호하기 위한 방법을 제안하고, 이를 통해 외부 지식과의 정렬을 통해 공격에 대한 저항력을 강화합니다.

- **Technical Details**: 우리는 External Knowledge (외부 지식)를 활용하여 이미지 영역과 외부 지식 간의 강한 정렬이 부족한 경우 모델이 이를 학습하지 않도록 제약을 두는 방법을 제안합니다. 모델의 시각적 지역에 대한 주의(attention)는 외부 지식과의 정렬에 비례하도록 강제하는 제약을 도입합니다. 또한, Semantic Shield라는 방어 메커니즘을 통해 다양한 Backdooring 및 Poisoning 공격에 대해 효과적으로 방어할 수 있도록 실험하였습니다.

- **Performance Highlights**: 제안된 Semantic Shield는 여러 설정에서 기존 방어 방법들보다 뛰어난 성능을 보였습니다. 본 연구의 실험 결과는 모델의 유용성을 유지하면서도 공격에 대해 높은 견고성을 제공하는 방법임을 보여줍니다. 이 방어 기법은 훈련 시간 동안 거의 추가적인 오버헤드가 없으며, 다양한 공격에 대해 모델을 더 강력하게 만듭니다.



### SMM-Conv: Scalar Matrix Multiplication with Zero Packing for Accelerated Convolution (https://arxiv.org/abs/2411.15659)
- **What's New**: 이 논문에서는 CPU 기반 아키텍처에서 추론 중 합성곱(convolution)의 속도를 가속화하는 새로운 접근법을 제안합니다. 기존의 im2col을 사용한 방법 및 GEMM(General Matrix Multiplication)의 단점을 해결하기 위해 스칼라-행렬 곱셈(scalar-matrix multiplication) 및 제로 패킹(zero packing) 방법을 활용합니다. 이로써 메모리 오버헤드를 줄이고 속도를 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 합성곱 레이어는 Tensors를 사용하여 계산되며, 본 연구에서는 일반적으로 사용되는 채널 우선 채널 배치(channels first) 메모리 레이아웃을 고려했습니다. 기존의 GEMM 기반 합성곱 방식은 메모리 사용이 비효율적이며 이는 DNNs(Deep Neural Networks)에서 성능 저하로 이어집니다. 따라서, 스칼라-행렬 곱셈을 사용하여 메모리 레이아웃을 지속적으로 관리하면서 연산 효율성을 높였습니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법이 기존의 im2col+GEMM 및 최첨단 메모리 효율 합성곱(MEC) 방법보다 더 높은 성능을 발휘함을 증명했습니다. 이를 통해 모바일 및 저전력 장치에서 기존 모델을 더욱 효율적으로 실행할 수 있다는 가능성을 보여줍니다. 이러한 기술적 진보는 이미지 관련 작업을 더 많은 소비자 장치에서 수행할 수 있게 해줄 것입니다.



### Training an Open-Vocabulary Monocular 3D Object Detection Model without 3D Data (https://arxiv.org/abs/2411.15657)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 3D 객체 탐지(Open-vocabulary 3D object detection)는 자율 주행(autonomous driving) 및 로보틱스(robotics) 분야의 다양한 응용으로 인해 최근 주목을 받고 있습니다. 그러나 기존의 포인트 클라우드(point cloud) 기반의 탐지 모델들은 높은 배치 비용으로 인해 한계가 있었습니다. 본 연구에서는 OVM3D-Det라는 새로운 단안 카메라(monocular) 3D 객체 탐지 프레임워크를 제안하여 RGB 이미지만으로도 탐지를 가능하게 하여 비용 효율적이고 데이터의 범용성을 제공하고 있습니다.

- **Technical Details**: OVM3D-Det는 기존의 LiDAR나 3D 센서 데이터 없이 RGB 이미지로 3D 객체를 자동으로 레이블링(labeling)하기 위한 방법론을 개발하였습니다. 이 프레임워크는 open-vocabulary 2D 모델을 사용하여 레이블이 없는 새로운 객체를 발견하고, pseudo-LiDAR 기술을 사용하여 객체의 3D 위치를 결정합니다. 두 개의 혁신적인 기술인 적응형 pseudo-LiDAR 침식(adaptive pseudo-LiDAR erosion)과 대형 언어 모델(large language models)에서의 이전 지식을 사용한 바운딩 박스(bounding box) 정제를 도입하여 정확성을 개선하였습니다.

- **Performance Highlights**: OVM3D-Det의 실험 결과, 다양한 실내 및 실외 환경에서 기준선 모델(baselines)과 비교하여 우수한 성능을 보여주었습니다. 특히, 자동 레이블링된 pseudo 타겟을 통해 훈련된 탐지기는 성능 향상에 크게 기여하며, 이는 open-vocabulary 문제를 해결하는 효과적인 접근 방식임을 시사합니다. 추가적으로, 관련 코드도 곧 공개될 예정이며, 이는 연구 커뮤니티에 더욱 큰 기여를 할 것입니다.



### OCDet: Object Center Detection via Bounding Box-Aware Heatmap Prediction on Edge Devices with NPUs (https://arxiv.org/abs/2411.15653)
- **What's New**: 이 논문에서는 OCDet라는 가벼운 Object Center Detection 프레임워크를 소개하며, 이는 NPU가 탑재된 엣지 디바이스에 최적화되어 있습니다. OCDet는 기존의 정적인 Gaussian 분포 대신 Generalized Centerness (GC)를 활용하여, 객체 중심 확률을 나타내는 히트맵을 예측하고 피크 식별을 통해 중심 좌표를 추출합니다. 이 방법은 추가적인 매뉴얼 라벨링 없이도 더 정밀한 공간 세부 정보를 제공합니다.

- **Technical Details**: OCDet는 NPU 친화적인 Semantic FPN과 MobileNetV4 백본으로 구축되어 모델 훈련 시 Balanced Continuous Focal Loss (BCFL)를 적용하여 데이터 불균형을 완화하고 어려운 부정 사례에 집중하여 확률 회귀 작업을 개선합니다. 또한 새로운 Center Alignment Score (CAS)를 도입하여 헝가리안 매칭을 통해 모델의 정밀도와 재현율을 평가합니다. 이러한 접근 방식을 통해 OCDet는 기존의 객체 감지 프레임워크와 비교하여 우수한 성능을 발휘합니다.

- **Performance Highlights**: COCO 데이터셋에 대한 실험에서 OCDet는 YOLO11 대비 최대 23% 높은 CAS를 달성하며, 필요한 파라미터 수를 42% 줄이고 계산량은 34% 감소시켰습니다. 키포인트 검출 프레임워크와 비교할 때 OCDet는 동일한 모델 구성을 사용하여 186% 이상의 CAS 향상을 보여줍니다. 이러한 결과는 OCDet가 NPU가 장착된 엣지 디바이스에서 효율적이고 강력한 객체 중심 검출 솔루션임을 입증합니다.



### Sample- and Parameter-Efficient Auto-Regressive Image Models (https://arxiv.org/abs/2411.15648)
Comments:
          for code, see this https URL

- **What's New**: 새로운 비전 모델인 XTRA는 기존의 자동 회귀 모델보다 샘플 및 파라미터 효율성을 크게 향상시키는 새로운 자동 회귀 목표로 사전 학습되었습니다. XTRA는 전통적인 causal mask 대신 Block Causal Mask를 사용하여 이미지를 블록 단위로 재구성(재구성)하고, 넓은 영역의 피젯 표현을 학습함으로써 고수준의 구조적 패턴을 포착합니다. 이러한 단순한 수정으로 XTRA는 데이터 세트와 모델 크기가 증가할수록 성능이 향상되는 일관된 확장성을 발휘하는 자동 회귀 비전 모델을 제안합니다.

- **Technical Details**: XTRA는 블록 causal mask(Bck Causal Mask)를 사용하여 각 블록을 k $	imes$ k 토큰으로 정의하고, 이를 통해 픽셀 값(픽셀 값)을 블록 단위로 재구성합니다. 이 방법은 대규모 이미지 지역의 관계를 학습하게 함으로써 보다 추상적이고 의미 있는 표현을 가능하게 합니다. 실험 결과에 따르면 XTRA는 적은 데이터와 소형 모델로도 추상적이고 의미 있는 표현을 학습할 수 있음을 보여줍니다.

- **Performance Highlights**: XTRA ViT-H/14는 152배 적은 샘플로도 이전의 최첨단 자동 회귀 모델을 능가하며, 15개 다양한 이미지 인식 벤치마크에서 평균 정확도에서 최고 성과를 달성했습니다. 또한 XTRA ViT-B/16은 ImageNet-1k에서 훈련된 자동 회귀 모델보다 7-16배 적은 매개변수로도 뛰어난 성능을 보여줍니다.



### Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection (https://arxiv.org/abs/2411.15633)
- **What's New**: 최근 AI 생성 이미지(AIGI) 탐지 기술은 특정 가짜 패턴에 빠르게 과적합(overfitting)되는 비대칭 현상을 발견했습니다. 기존의 탐지 모델은 훈련 세트에서 본 가짜 패턴에 신속하게 적응하지만, 새로운 가짜 메소드에 대한 일반화 성능은 저하됩니다. 이 문제를 해결하기 위해, 대규모 비전 기초 모델(VFMs)에서 풍부한 의미적 지식을 활용하여, 가짜 및 의미적 단서를 기반으로 한 분별 공간을 확장하는 새로운 접근 방법 Effort를 제안했습니다.

- **Technical Details**: Effort는 고유 벡터 분해(Singular Value Decomposition, SVD)를 사용하여 의미적 서브스페이스와 가짜 서브스페이스를 직교(orthogonal)로 구성합니다. 주성분(principal components)은 고정(freezing)하고, 잔여 성분(residual components)만을 조정하여 원래의 의미론적 서브스페이스를 보존하며 학습에서 가짜를 다루는 방식입니다. 이를 통해 기존의 의미론적 지식을 왜곡하지 않으면서, 더 나은 일반화 성능을 달성합니다.

- **Performance Highlights**: 이 연구 결과, Effort가 AIGI 탐지 벤치마크에서 다른 최첨단 기술(SOTA)들과 비교해 우수한 성능을 보였음을 입증했습니다. 불과 약 0.19M의 튜너블 파라미터만으로도 이전에 보지 못한 다양한 가짜 이미지를 효과적으로 탐지할 수 있으며, 이는 매우 효율적이면서도 확장 가능하다는 점에서 큰 장점입니다. Effort 방식은 고전적인 딥페이크 탐지 및 합성 이미지 탐지 작업 모두에 적용 가능합니다.



### ACE: Action Concept Enhancement of Video-Language Models in Procedural Videos (https://arxiv.org/abs/2411.15628)
Comments:
          Accepted at WACV 2025

- **What's New**: 본 연구는 Action Concept Enhancement (ACE)라는 새로운 세부 조정을 제안하여 Vision-Language 모델(VLM)의 절차적 행동 인식 능력을 향상시킵니다. 기존 VLM은 보이는 행동 범주에 편향되어 있으며, 사전 학습된 고정 레이블에 과적합되는 문제가 있었습니다. ACE는 랜덤하게 고정 레이블을 교체함으로써 새로운 행동 레이블 조합을 형성하고, 결과적으로 행동 개념의 이해를 증가시킵니다.

- **Technical Details**: ACE는 훈련 중 비슷한 행동 개념을 포함한 보조 분류 손실을 추가하여 강화된 행동 동의어 및 부정 샘플을 통합합니다. 이를 위해 대규모 언어 모델(LLM)의 지식을 활용하여 동의어 트리를 구축하고, 이를 통해 매 훈련 iteration마다 새로운 행동 성취를 시뮬레이션합니다. 이 방법은 모델이 고정된 동사 및 객체에 과적합되지 않도록 방지하며, 언어 모델과의 통합을 통해 보지 못한 행동 인식을 개선합니다.

- **Performance Highlights**: 실험 결과, ACE를 적용한 VLM이 IKEA, ATA 및 GTEA 데이터셋에서 우수한 성능을 보였으며, 보지 못한 행동 인식에서 현저한 개선을 이뤘습니다. ACE는 기초 행동을 새롭고 희귀한 행동 세트로의 분류로 시뮬레이션함으로써 인식을 개선하고, 기존 행동에 대한 경쟁력 있는 성과를 유지했습니다. 본 연구는 VLM의 행동 개념 이해를 시험한 최초의 사례로, 사전 학습된 모델에서 인도메인 지식을 통합해 보지 못한 행동 추론을 가능하게 합니다.



### On the importance of local and global feature learning for automated measurable residual disease detection in flow cytometry data (https://arxiv.org/abs/2411.15621)
Comments:
          Accepted at ICPR 2024

- **What's New**: 이 논문에서는 측정 가능한 잔여 질병(MRD) 탐지를 위한 다양한 딥러닝(deep learning) 방법을 평가합니다. 특히, 긴 거리 종속성(long-range dependencies)을 모델링하는 것의 이점, 글로벌 정보(global information) 확보 방법, 지역적 특징(local features)을 학습하는 것의 중요성에 대해 다룹니다. 이를 바탕으로 현재의 최신 모델(State-of-the-Art, SOTA)에 대한 두 가지 적응 방안을 제안하며, 이는 FCM(Flow Cytometry) 데이터 분석을 위한 딥러닝 아키텍처 설계에 대한 귀중한 통찰을 제공합니다.

- **Technical Details**: MRD 탐지 문제는 단일 사건(single events)을 건강한 세포나 암세포로 이진 분류하는 것으로 설정됩니다. FCM 데이터셋은 단일 세포의 특성을 측정한 F차원 피쳐 벡터(feature vector)의 집합으로 정의됩니다. 기존의 SOTA 모델은 세트 변환기(set transformer)를 기반으로 하며, 연구진은 최근의 연구를 통해 글로벌 피쳐 추출과 지역 피쳐 학습의 중요성을 강조하고, 두 가지 방안 즉, 샘플에서 직접 피쳐 벡터를 사용하고 그래프 신경망(graph neural network) 레이어를 통합하는 방법을 제안합니다.

- **Performance Highlights**: 개선된 모델은 공개 데이터셋에서 더 나은 성능을 보여주며, 다양한 실험실 간 일반화 향상도 확인되었습니다. 본 연구 결과는 FCM 커뮤니티에 유익한 통찰력을 제공하며, 향후 딥러닝 아키텍처 설계에 있어 기초 자료로 활용될 것입니다. 일반적으로 MRD 탐지 업무에 대한 자동화된 분석 솔루션을 제공하여, 어린이 급성 백혈병 환자에게 중요한 영향을 미칠 것입니다.



### Fine-Grained Open-Vocabulary Object Recognition via User-Guided Segmentation (https://arxiv.org/abs/2411.15620)
- **What's New**: 이 논문에서는 FOCUS라는 새로운 방법론을 제안합니다. FOCUS는 사용자 안내 세분화를 통한 고급의 오픈 어휘 개체 인식(Fine-grained Open-Vocabulary Object ReCognition) 모델로, 사용자 요청에 따라 다양한 객체를 감지할 수 있는 능력을 제공합니다. 더불어, 사용자가 자연어로 탐색할 수 있는 기능을 갖추어, 객체 탐지 과정에서 사용자 개입을 최소화하면서도 실질적인 조정을 가능하게 합니다.

- **Technical Details**: FOCUS는 세 가지 주요 단계로 구성됩니다: (i) 지역 아이소레이션(Region Isolation), (ii) 제안 추출(Proposal Extraction), (iii) 개체 식별(Entity Identification)입니다. 이 모델은 이미지와 바운딩 박스를 입력받고, Segment Anything Model(SAM)을 통해 사용자가 관심 있는 영역을 마스킹하여 복잡한 장면에서도 세부 요소를 찾아낼 수 있도록 돕습니다. VLM(vision language models)을 활용하여 사용자로부터 자연어로 제안된 객체를 지원하며, 사용자 요청에 부합하는 객체만을 제안합니다.

- **Performance Highlights**: FOCUS는 기존 기준 모델에 비해 객체 감지 능력을 개선하며, 다양한 객체 타입에서도 일관된 성능을 보입니다. 이 방법론은 설명 가능한 요청을 통해 사용자가 감지 프로세스를 능동적으로 안내할 수 있도록 하며, 세밀한 구성 요소를 효과적으로 인식할 수 있는 가능성을 열어줍니다. 전반적으로, 각기 다른 모델을 유연하게 대체할 수 있는 기능을 갖추어, 다양한 응용 분야에서의 활용 가능성을 높이고 있습니다.



### Knowledge Transfer Across Modalities with Natural Language Supervision (https://arxiv.org/abs/2411.15611)
Comments:
          21 pages, 7 figures, 17 tables

- **What's New**: 이 논문에서는 텍스트 설명만을 사용하여 새로운 개념을 학습할 수 있는 방법, 즉 Knowledge Transfer를 제안합니다. 이는 기존의 시각 인식을 암시적으로 활용하여 새로운 고차원 개념을 도입하는 방식으로, 프리트레인된 비주얼 인코더 모델이 이미 학습한 저수준 특징들을 사용합니다. 이 방법은 CLIP과 같은 여러 다중모달 모델의 기본 개념에 잘 호환됩니다.

- **Technical Details**: 명시적 Knowledge Transfer는 텍스트 설명을 바탕으로 이미지 생성/합성을 통해 모델을 미세 조정하는 방식입니다. 이 과정에서 기존의 시각-텍스트 정렬 손실 (visual-text matching loss)을 활용하여 미세 조정을 진행하게 됩니다. 또한, 이 연구에서는 이미 알고 있는 개념의 정확도를 개선할 수 있는 방법으로 Knowledge Transfer가 효과적으로 작용할 수 있음을 보입니다.

- **Performance Highlights**: Knowledge Transfer를 통해 분류(classification), 분할(segmentation), 이미지-텍스트 검색(image-text retrieval) 및 캡셔닝(captioning) 등 다양한 작업에서 제로샷(zero-shot) 성능을 개선할 수 있습니다. 특히, 단 하나의 텍스트 설명만으로도 새로운 개념을 도입할 수 있어 효율적인 지식 통합이 가능하다는 점이 중요합니다.



### GIFT: A Framework for Global Interpretable Faithful Textual Explanations of Vision Classifiers (https://arxiv.org/abs/2411.15605)
- **What's New**: 본 논문에서는 GIFT라는 새로운 프레임워크를 소개하고 있으며, 이는 비전 분류기(vision classifiers)에 대한 포스트 핫(post-hoc) 글로벌 해석 가능한 설명을 제공합니다. GIFT는 시각적 반사실(counterfactual) 설명을 기반으로 하며, 이러한 설명을 언어 모델(language models)을 통해 전 세계적으로 이해할 수 있는 설명으로 변환합니다. 이 프레임워크는 제안된 설명이 분류기의 결정에 미치는 인과적(causal) 영향을 측정하는 검증 단계를 포함합니다.

- **Technical Details**: GIFT는 세 단계로 작동하며, 첫 단계에서는 모델 입력의 다양한 반사실 이미지 쌍(counterfactual pairs)을 생성하여 개별 입력에서 모델의 결정에 중요한 시각적 특성을 강조합니다. 다음 단계에서는 비전 언어 모델(Vision Language Model, VLM)을 사용하여 반사실 쌍의 각 이미지 간의 차이를 해석 가능한 텍스트 설명으로 변환합니다. 마지막으로, 제안된 설명의 인과성을 검증하기 위해 이미지 편집 모델을 통한 개입을 사용하여 설명의 효과를 측정합니다.

- **Performance Highlights**: GIFT는 CLEVR, CelebA 및 BDD 데이터셋을 포함한 다양한 실험에서 유의미한 통찰력을 효과적으로 드러내며, 깊은 비전 분류기가 사용하는 작업, 개념 및 편향을 발견하는 능력을 보여줍니다. 이 프레임워크는 사용자가 목표 분류기를 더 깊이 이해할 수 있도록 설명을 상호작용할 수 있게 하며, 전반적으로 정확하고 의미 있는 글로벌 설명을 생성할 수 있습니다.



### FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video (https://arxiv.org/abs/2411.15604)
Comments:
          project page: this https URL

- **What's New**: FATE는 단일 모노큘라 비디오에서 편집 가능한 전체 머리 아바타를 재구성하는 새로운 방법입니다. 이 기술은 샘플링 기반 밀집화 전략을 통합하여 점의 최적 위치 분포를 보장하여 렌더링 효율성을 향상시킵니다. 또한, FATE는 360도 렌더링이 가능한 3D 머리 아바타를 생성하기 위한 범용 완성 프레임워크를 개발하여 비포인트 형태의 재구성을 다룹니다.

- **Technical Details**: FATE는 비디오 프레임에서 표정 및 자세를 추정하기 위한 파라메트릭 헤드 추정 알고리즘을 사용하고, 후면 머리 재구성을 위한 범용 완성 프레임워크와 결합됩니다. 이 방법은 UV 텍스처 공간에서 직접 편집할 수 있도록 연속 속성 맵으로 변환하는 신경 베이킹 기술을 도입합니다. 또한, 기존의 3D Gaussian Splatting을 개선하여 모델의 효율성을 높이고 중복 점을 줄입니다.

- **Performance Highlights**: FATE는 정성적 및 정량적 평가 모두에서 기존 방법보다 뛰어난 성능을 보여줍니다. 이 방법은 첫 번째 애니메이션이 가능한 360도 전체 머리 모노큘라 재구성 방법으로, 아바타 생성 분야에서 새로운 표준을 제시합니다. FATE의 코드는 발행 후 공개될 예정입니다.



### Enhancing Object Detection Accuracy in Autonomous Vehicles Using Synthetic Data (https://arxiv.org/abs/2411.15602)
Comments:
          7 Pages, 7 figures, 1 table

- **What's New**: 이 연구는 합성 데이터(synthetic data)를 생성하여 객체 탐지(object detection) 시스템의 예측 정확도를 향상시키는 것을 목표로 합니다. 특히 자율주행차 시나리오를 예시로 들어 합성 데이터의 효과를 입증하고자 하며, 이는 현실적인 상황을 재현하기 위한 3D 장면을 필요로 합니다. 연구 결과, 합성 데이터를 포함한 시스템이 기존의 실제 데이터로 훈련된 시스템보다 모든 성능 매트릭에서 우수한 결과를 보였다는 점은 매우 중요합니다.

- **Technical Details**: Deep learning 기반의 객체 탐지 모델인 YOLO는 단일 신경망으로 경계 상자(bounding boxes)와 클래스 확률(class probabilities)을 동시에 예측하는 방식으로 작동합니다. 연구에서는 Unity 엔진을 사용하여 현실적인 환경을 만들고 합성 데이터를 생성할 예정이며, 이는 도시 및 자연 환경에서 AI 시스템의 교육에 효과적입니다. Procedural Data Generation (PDG) 기술을 통해 복잡한 시나리오를 시뮬레이션하여 AI 모델을 훈련시키는 방법을 논의합니다.

- **Performance Highlights**: System-2(합성 데이터와 실제 데이터를 결합하여 훈련된 모델)는 System-1(실제 데이터로 훈련된 모델)보다 정확도가 3% 향상된 결과를 나타냈습니다. 이 외에도 정밀도(precision), 재현율(recall), 평균 평균 정밀도(mean average precision) 등 다른 모든 성능 매트릭에서도 우수한 성능을 보였습니다. 이러한 결과는 합성 데이터가 머신러닝 모델의 성능 향상에 미치는 긍정적인 영향을 확증합니다.



### How Texts Help? A Fine-grained Evaluation to Reveal the Role of Language in Vision-Language Tracking (https://arxiv.org/abs/2411.15600)
Comments:
          Preprint, Under Review

- **What's New**: 이번 논문에서는 기존의 Vision-Language Tracking (VLT) 시스템의 한계를 극복하기 위해 VLTVerse라는 첫 번째 세밀한 평가 프레임워크를 제안합니다. 이 프레임워크는 10개의 시퀀스 수준의 도전 요소와 6종의 다중 관점의 의미 정보를 통합하여, 복잡한 상황에서 VLT 트래커의 성능을 체계적으로 평가할 수 있는 공간을 제공합니다. 또한, 이를 통해 언어 정보가 VLT에서 수행하는 역할을 조명하며, 데이터, 평가 및 알고리즘 차원에서 VLT를 개선하기 위한 필수적인 지침을 제공합니다.

- **Technical Details**: VLTVerse는 기존 SOTVerse를 기반으로 하여, 짧은 기간, 긴 기간, 그리고 글로벌 인스턴스 추적 작업을 포괄하는 세밀한 평가 프레임워크를 구축합니다. 네 가지 대표적인 벤치마크, 10가지 도전 요소, 6종의 의미 정보를 포함하고 있어, VLT 트래커에 대한 포괄적인 평가가 가능합니다. 특히, 60가지 도전 요소와 의미가 결합된 체계적인 성능 평가를 통해, 기존 평가 방법으로는 포착할 수 없는 언어의 중요성에 대한 통찰을 제공합니다.

- **Performance Highlights**: VLTVerse를 통해 실시된 세밀한 평가는 다양한 도전 요소 하에서 언어 모달리티가 VLT 트래커 성능에 미치는 영향을 심층적으로 분석합니다. 연구자들은 VLTVerse의 평가 공간을 통해 성능 병목 현상을 식별하고 알고리즘 설계를 최적화할 수 있으며, VLT 작업 개념에 대한 이해를 증진시켜 향후 연구에서 트래커 성능을 향상하는 데 기여할 수 있는 귀중한 정보를 제공합니다.



### An adversarial feature learning based semantic communication method for Human 3D Reconstruction (https://arxiv.org/abs/2411.15595)
- **What's New**: 이 논문에서는 인체 3D 재구성을 위한 새로운 접근 방식으로 Adversarial Feature Learning 기반의 Semantic Communication 방법(AFLSC)을 소개합니다. 이 방법은 3D 재구성 작업에 중요한 의미 정보를 추출하고 전송하는 데 중점을 두고 있습니다. 특히, 네트워크 대역폭이 제한적이고 낮은 지연 시간이 요구되는 환경에서 데이터 흐름을 최적화하고 대역폭 압박을 완화하는 데 기여합니다.

- **Technical Details**: 발신 측에서는 멀티태스크 학습 기반의 특징 추출 방법을 제안하여 2D 인체 이미지에서 공간 배치, 키포인트, 자세 및 깊이 정보를 캡처합니다. 또한, 이러한 정보를 의미 데이터로 인코딩하는 Adversarial Feature Learning 기반의 의미 인코딩 기술을 설계하였습니다. 수신 측에서는 효과적인 다단계 의미 특징 디코딩 방법을 설계하여 의미 데이터를 다시 키 이미지 특징으로 변환합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 데이터 전송 효율성과 재구성 품질 측면에서 우수한 성능을 보임을 확인했습니다. 이 기술은 대역폭이 제한된 환경에서의 응용 가능성이 뛰어난 것을 입증하며, 인체 3D 메쉬 모델을 생성하는 improved ViT-diffusion 모델을 활용합니다. 따라서 이 연구는 다양한 분야에서 인체 3D 재구성을 위한 귀중한 기여를 할 것으로 기대됩니다.



### Boosting Semi-Supervised Scene Text Recognition via Viewing and Summarizing (https://arxiv.org/abs/2411.15585)
- **What's New**: 이 논문은 기존의 Scene Text Recognition (STR) 방법들이 특히 예술적이고 왜곡된 캐릭터의 인식에서 한계를 보이고 있다는 점을 강조합니다. 저자들은 인간의 학습 과정에서 영감을 받아, 대조 학습 기반의 STR 프레임워크를 제안하며 합성 데이터와 비지도 데이터의 조합을 통해 문제를 해결하고자 합니다. 이를 통해 인식 효과성을 높이는 새로운 접근 방법으로 Viewing and Summarizing (ViSu) 패러다임을 도입합니다.

- **Technical Details**: 저자들은 강력한 데이터 증강과 캐릭터 형태 정렬(CUA Loss) 기법을 통해 모델의 힘을 키우고, 다양한 형태의 캐릭터를 효과적으로 인식하도록 돕습니다. Online Generation Strategy (OGS)를 통해 백그라운드 없는 샘플을 생성하여, 모델이 복잡한 데이터에서 캐릭터 형태에 집중할 수 있도록 하는 방법을 제안합니다. 이러한 기술적 접근은 모델의 데이터 드리븐 특성을 극대화하려는 목표와 일치합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 일반 벤치마크에서 94.7%, 도전적인 Union14M-Benchmark에서는 70.9%의 평균 정확도를 기록하며 최신 성과(State-of-the-Art, SOTA)를 달성했습니다. 이는 단순한 합성 데이터로 훈련하면서도 실제 복잡한 샘플 인식에 대한 모델의 일반화 능력을 크게 향상시켰음을 나타냅니다.



### FLD+: Data-efficient Evaluation Metric for Generative Models (https://arxiv.org/abs/2411.15584)
Comments:
          13 pages, 10 figures

- **What's New**: 이 논문에서는 기존의 Fréchet Inception Distance (FID)와 같은 메트릭보다 더 신뢰성 있고 데이터 효율적이며 계산 효율적인 새로운 메트릭인 Flow-based Likelihood Distance Plus (FLD+)를 소개합니다. FLD+는 노멀라이징 흐름(normalizing flows)에 기반하여 모든 도메인의 이미지의 밀도(정확한 로그 우도)를 계산할 수 있게 합니다. 이 메트릭은 다양한 이미지 열화(예: 노이즈, 폐쇄, 확산 단계)와 생성 모델 크기에 대해 강한 단조적(monotonic) 동작을 보여줍니다.

- **Technical Details**: FLD+는 이미지에서 필수 정보를 캡처하기 위해 사전 훈련된 백본 네트워크(pre-trained backbone network)를 사용하여 피처 텐서를 추출합니다. 그런 다음 이 저차원 피처 텐서만 노멀라이징 흐름을 통해 전달하여 훈련 및 평가 속도를 높입니다. FLD+는 생성 이미지와 실제 이미지의 분포 간의 정렬을 평가하며, 더 정확하고 효율적인 유사성 측정을 제공합니다.

- **Performance Highlights**: FLD+는 FID에 비해 두 배의 정밀도를 가진 이미지 로그 우도 추정 덕분에 안정적인 메트릭 추정을 위해 필요한 이미지 수에서 두 자릿수의 감소를 달성합니다. 또한 FLD+는 사전 훈련된 피처 추출기를 사용하여 적은 데이터와 컴퓨팅 요구 사항을 갖고 새로운 이미지 도메인에 쉽게 재훈련할 수 있는 능력을 증명합니다.



### EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting (https://arxiv.org/abs/2411.15582)
- **What's New**: 이번 논문에서는 자율주행을 위한 포토리얼리스틱 (Photorealistic) 거리 장면 복원에서 발생하는 도전 과제를 해결하기 위해 Explicit Motion Decomposition (EMD) 모듈을 제안합니다. EMD는 동적 객체의 운동을 학습 가능한 모션 임베딩을 통해 모델링하여, 거리 장면에서의 분해 성능을 향상시킵니다. 이 방법은 기존의 감독 학습 및 자가 감독 학습( self-supervised) 기반 방법에 쉽게 통합할 수 있는 플러그 앤 플레이 (plug-and-play) 접근법입니다.

- **Technical Details**: EMD는 모션 인식 기능 인코딩(motion-aware feature encoding)과 이중 스케일 변형 모델링(dual-scale deformation modeling)을 결합하여 거리 장면의 높은 수준의 분해를 이끌어냅니다. 구체적으로 EMD는 각 가우시안 프리미티브(primitive)에 학습 가능한 모션 임베딩을 추가하여 개별 객체의 운동 특성을 포착합니다. 또한, 빠른 글로벌 운동(fast, global motions)과 느린 로컬 변형(slow, local deformations)을 별도로 관리하는 계층적 변형 프레임워크를 설계하여 복잡한 거리 장면을 보다 효율적으로 분석하도록 지원합니다.

- **Performance Highlights**: EMD는 다양한 기준 방법들과 통합되어 Waymo-NOTR 데이터세트에서 실험되었으며, 대부분의 평가 프로토콜에서 일관된 성능 향상을 보여주었습니다. 이 연구에서 제안한 방법은 전체 장면에서 +1.81 PSNR( Peak Signal-to-Noise Ratio) 및 차량 특정 지역에서 +2.81 PSNR의 통계적으로 유의미한 개선을 보여, 거리 장면 복원 품질 향상에 기여할 수 있음을 입증했습니다.



### TKG-DM: Training-free Chroma Key Content Generation Diffusion Mod (https://arxiv.org/abs/2411.15580)
- **What's New**: 이 논문에서는 Training-Free Chroma Key Content Generation Diffusion Model (TKG-DM)을 소개합니다. 본 모델은 초기 임의의 노이즈를 최적화하여 배경 색상이 선택 가능한 이미지 생성을 가능케 합니다. 기존의 모델들이 필수적인 미세조정을 요구하는 것과 달리, TKG-DM은 그러한 필요 없이 전경과 배경을 정확하게 분리할 수 있습니다. 이는 다양한 생성 애플리케이션에서 전경과 배경을 독립적으로 제어하는 데 필요한 가능성을 확장합니다.

- **Technical Details**: TKG-DM은 초기 Gaussian noise를 조작하여 배경 색상을 통제하는 방법을 제안합니다. 모델은 기본 Stable Diffusion을 기반으로 하며, 초기에 생성된 색상 노이즈를 결합하여 전경과 배경이 명확히 구분된 시각적 콘텐츠를 생성합니다. 이를 통해 전경 객체의 배치 및 배경 색상 선택에 대한 정밀한 제어를 제공합니다. 또한, TKG-DM은 수량적 및 질적 평가에서 기존 방법들을 초월하여 효과적인 결과를 보장합니다.

- **Performance Highlights**: 실험 결과, TKG-DM은 FID 및 mask-FID 점수를 각각 33.7% 및 35.9% 향상시켰습니다. 이는 미세조정된 모델과 경쟁할 수 있는 성능을 제공함을 의미합니다. TKG-DM은 조건부 텍스트-이미지 생성, 일관성 모델 및 텍스트-비디오 생성 등에 원활히 확장 가능하여 다양한 크로마 키 콘텐츠 생성에 기여합니다. 전반적으로, TKG-DM은 낮은 계산 비용으로 보다 효율적이고 다재다능한 시각적 콘텐츠 창작 솔루션을 제공합니다.



### LAGUNA: LAnguage Guided UNsupervised Adaptation with structured spaces (https://arxiv.org/abs/2411.15557)
- **What's New**: 이번 논문에서 저자는 상대적 표현을 통한 비지도 도메인 적응(unsupervised domain adaptation, UDA)을 제안합니다. 기존의 방법은 도메인 불변 표현과 도메인 특정 특징을 보존하는 균형을 맞추기 어려웠습니다. 그들은 LAGUNA라는 새로운 방법을 소개하며 이를 통해 세분화된 관계를 구성하는 것이 가능하다고 주장합니다.

- **Technical Details**: LAGUNA는 세 단계로 구성된 방법론으로, 첫 번째 단계에서 텍스트 클래스 레이블을 사용하여 도메인 불가지론(reference space)을 생성합니다. 두 번째 단계에서는 언어 모델을 학습하여 텍스트 캡션을 참조 잠재 공간에 매핑하고, 마지막 단계에서는 크로스 도메인 분류기를 훈련합니다. 이러한 방법을 통해 각 도메인은 고유한 패턴을 유지하면서도 서로 연결됩니다.

- **Performance Highlights**: LAGUNA는 4개의 다양한 이미지 및 비디오 데이터셋에서 기존의 최첨단 기술을 초과하는 성능을 보였습니다. 예를 들어, DomainNet에서 평균 3.32%의 정확도 향상, GeoPlaces에서 5.75%, GeoImnet에서 4.77%, EgoExo4D에서는 평균 클래스 정확도에서 1.94% 향상을 기록하였습니다. 이러한 결과는 기법의 우수성을 입증합니다.



### ReWind: Understanding Long Videos with Instructed Learnable Memory (https://arxiv.org/abs/2411.15556)
- **What's New**: ReWind는 비디오 이해를 위한 혁신적인 메모리 기반 비전-언어 모델(VLM)로, 긴 비디오 처리에서 효과적인 최신 기술을 제공합니다. 이 모델은 메모리 모듈을 통해 비디오 진행 중 지시사항과 연관된 시각 정보를 저장하고 업데이트하는 'read-perceive-write' 사이클을 갖추고 있습니다. 기존 모델들이 겪는 메모리 부족과 복잡한 계산 문제를 해결할 수 있습니다.

- **Technical Details**: ReWind는 두 단계로 구성된 프레임워크로 작동하며, 첫 번째 단계에서는 동적 학습이 가능한 메모리 모듈을 사용하여 비디오에서 중요한 정보만 저장합니다. 이 메모리 모듈은 입력 토큰 수에 따라 선형적으로 메모리를 확장하는 방식으로 설계되어 있으며, 영상 처리에 필요한 고해상도 프레임을 선택함으로써 질 높은 시각 정보를 처리합니다.

- **Performance Highlights**: ReWind는 시각적 질문 응답(VQA) 및 시간 기반 안정성 작업에서 기존 방법보다 우수한 성능을 보여줍니다. MovieChat-1K VQA 데이터 세트에서 13%의 점수 향상과 Charades-STA에서 8%의 mIoU 증가를 달성하였습니다. 이러한 결과는 ReWind의 효과성을 입증하며, 긴 금융 비디오에 대한 이해도를 크게 개선합니다.



### Enhancing the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation (https://arxiv.org/abs/2411.15555)
- **What's New**: 이번 논문에서는 Face Recognition (FR) 모델에 대한 새로운 공격 방법인 Diverse Parameters Augmentation (DPA)를 소개합니다. 기존의 공격 기법들은 다양한 초기화를 활용하여 대체 모델(surrogate model)을 보강하지 못하여 전이 가능성(transferability)이 제한되어 있었으나, DPA는 랜덤 및 사전 훈련(pre-trained) 초기값을 포함하여 이를 개선합니다. 특히, DPA는 Diverse Parameters Optimization (DPO) 및 Hard Model Aggregation (HMA)이라는 두 가지 주요 단계를 포함합니다.

- **Technical Details**: DPA의 DPO 단계에서는 임의의 노이즈를 사용하여 파라미터 초기화를 시작하고, 중간 훈련 과정에서 파라미터를 저장하여 다양한 대체 모델 세트를 생성합니다. HMA 단계에서는 유용한 섭동(perturbation)을 추가하여 특성 맵(feature maps)을 향상시켜 전이 가능성을 증가시킵니다. 이러한 기법들은 기존 공격 방법들이 갖고 있는 파라미터 초기화의 제한을 극복하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, DPA 공격 방법은 기존의 최첨단(adversarial attack) 공격 방법들과 비교했을 때 더욱 우수한 성능을 나타냈습니다. 파라미터 다양성을 통해 생성된 적대적 예제(adversarial examples)의 전이 가능성이 현저히 향상되었음을 확인할 수 있었습니다. 이러한 기법들은 Face Recognition 시스템의 안전성을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### Improving Transferable Targeted Attacks with Feature Tuning Mixup (https://arxiv.org/abs/2411.15553)
- **What's New**: 이번 논문에서는 Feature Tuning Mixup (FTM)이라는 새로운 방법을 제안합니다. FTM은 무작위 노이즈와 최적화된 노이즈를 결합하여 목표 클래스에 대한 전이 가능성(transferability)을 향상시킵니다. 기존의 Clean Feature Mixup(CFM) 방법이 무작위 깨끗한 특징을 사용했던 것에 반해, FTM은 학습 가능한 특징의 왜곡을 도입하여 공격의 효과를 개선합니다. 또한, 효율적인 확률적 업데이트 전략을 활용하여 계산 비용을 절감하면서도 성능을 향상시킵니다.

- **Technical Details**: FTM은 중간 레이어에서 학습 가능한 왜곡(feature perturbations)을 도입하여 공격 과정에서 대체 모델의 특징을 미세 조정합니다. 이 특징 왜곡은 무작위 깨끗한 특징과 혼합하여 목표 공격의 효과를 방어하는 데 최적화됩니다. 각 공격 반복 단계에서 소수의 레이어를 무작위로 선택하여 학습 가능한 왜곡을 업데이트하고, 이를 통해 과도한 왜곡을 피함으로써 공격 성능을 유지합니다. 이러한 방식으로 FTM은 공격 고유의 노이즈를 최대한 활용하여 전이 가능성을 향상시킵니다.

- **Performance Highlights**: 다양한 모델에서 ImageNet-호환 데이터셋을 사용하여 FTM의 성능이 기존 방법보다 상당한 개선을 이루었음을 실험적으로 입증하였습니다. FTM은 최고의 공격 성능을 보이면서도 낮은 계산 비용을 유지하여 실제 AI 시스템의 보안성을 향상시킬 수 있는 가능성을 제공합니다. 또한, FTM으로 perturb된 서브 모델들 간의 앙상블을 활용함으로써 공격 성능의 추가적인 개선이 가능함을 보여주었습니다.



### NeRF Inpainting with Geometric Diffusion Prior and Balanced Score Distillation (https://arxiv.org/abs/2411.15551)
- **What's New**: 본 논문에서는 2D diffusion priors의 개선된 활용을 통해 NeRF (Neural Radiance Field) inpainting 성능을 향상시키는 GB-NeRF 프레임워크를 소개합니다. 기존의 Score Distillation Sampling (SDS) 방법의 비효율성에 대한 문제점을 해결하고, appearance와 geometric priors를 동시에 학습하는 fine-tuning 전략을 포함합니다. 또한, Balanced Score Distillation (BSD) 기법을 도입하여 NeRF inpainting의 최적화 안정성을 높입니다.

- **Technical Details**: GB-NeRF는 RGB 이미지와 normal map을 동시에 생성하는 방식으로 geometric priors를 효과적으로 학습합니다. 이 과정에서 BLIP을 통해 생성된 캡션이 포함된 고품질 RGB-normal 이미지 데이터셋을 사용하며, LoRA가 U-Net 및 text encoder에 통합되어 appearance와 geometric 정보를 학습합니다. BSD는 NeRF inpainting 작업에 특화된 최적화 기법으로, 무작위 노이즈와 조건 없는 예측 항목으로 인해 발생하는 불필요한 최적화 변동성을 제거하여 안정적인 감독 신호를 제공합니다.

- **Performance Highlights**: GB-NeRF는 LLFF와 SPIn-NeRF라는 두 가지 데이터셋에서 실험을 통해 기존 접근 방식에 비해 뛰어난 appearance fidelity와 geometric consistency를 달성했습니다. 결과적으로, 본 방법은 양적인 메트릭과 시각적 품질 모두에서 최첨단 성능을 실현했습니다. 이 연구는 NeRF inpainting의 질적 향상에 기여할 뿐만 아니라, 향후 3D 콘텐츠 제작 분야에도 중요한 활용 가능성을 보여줍니다.



### Hierarchical Cross-Attention Network for Virtual Try-On (https://arxiv.org/abs/2411.15542)
- **What's New**: 이번 논문에서는 가상 착용 작업의 문제를 해결하기 위한 혁신적인 솔루션인 Hierarchical Cross-Attention Network (HCANet)을 제시합니다. HCANet은 기하학적 매칭(geometric matching)과 착용(try-on)의 두 주요 단계로 구성되어 있으며, 각 단계에서 현실적인 가상 착용 결과를 제공하는 데 중요한 역할을 합니다. 특히, HCANet은 두 단계 모두에 혁신적인 Hierarchical Cross-Attention (HCA) 블록을 도입하여 개인 및 의복 모달 간의 장기 상관관계를 효과적으로 포착합니다.

- **Technical Details**: HCANet은 기하학적 매칭 단계에서 실제 의복과 대상 인물의 정밀한 정렬을 수행하고, 이를 위해 학습 가능한 thin-plate spline transformation을 적용합니다. 이후 착용 단계에서는 정렬된 의복 및 인물 표현을 입력으로 받아 포즈 일치 이미지와 구성 마스크를 생성하여 부드러운 합성을 이룹니다. 또한, HCA 블록은 인물과 의복 간의 관계를 수월하게 융합할 수 있도록 두 개의 상호 연결된 단계로 구성되어 있습니다.

- **Performance Highlights**: HCANet은 정성적 및 정량적 평가를 통해 기존의 최첨단 방법보다 뛰어난 성능을 보였습니다. 특히, 이는 사실적인 가상 착용 결과를 생성하는 데 있어 시각적 충실도와 정확성이 우수함을 증명했습니다. 이번 연구는 가상 착용 기술의 발전에 중요한 이정표가 될 것으로 기대되며, 온라인 쇼핑 경험을 혁신적으로 변화시킬 수 있는 가능성을 지니고 있습니다.



### Optical-Flow Guided Prompt Optimization for Coherent Video Generation (https://arxiv.org/abs/2411.15540)
Comments:
          project page: this https URL

- **What's New**: 본 논문에서는 MotionPrompt라는 새로운 비디오 생성 시스템을 제안합니다. 이 시스템은 optical flow를 활용하여 비디오 생성 과정에 가이드를 제공함으로써, 생성된 비디오의 시간적 일관성을 향상시킵니다. 또한, 학습 가능한 토큰 임베딩을 최적화하여 복잡한 계산 부담을 줄이고, 자연스러운 모션 다이나믹스를 반영하는 시각적으로 일관된 비디오를 생성합니다.

- **Technical Details**: MotionPrompt는 두 개의 랜덤 프레임 쌍 사이의 optical flow를 구분하는 판별기를 훈련하여 비디오 생성에 있어 시간적 일관성을 유지합니다. 프롬프트를 통해 존재하는 전체 비디오에 영향을 미칠 수 있도록 설정되어 있어, 기존의 프레임 단위 계산 부담을 줄이면서도 학습 가능한 토큰의 임베딩을 최적화합니다. 이를 통해 생성된 프레임에 대한 상대 모션의 현실성을 평가하여 모션 패턴을 수정하게 됩니다.

- **Performance Highlights**: MotionPrompt의 성능은 다양한 비디오 생성 모델에서 검증되었습니다. 이 접근 방식을 통해 생성된 비디오 시퀀스는 자연스러운 모션을 유지하면서도 높은 시각적 품질을 확보합니다. 이전 방법들과 달리 MotionPrompt는 기존의 확산 모델의 재훈련 없이 시간 일관성을 보장하면서도, 이미 현실적인 비디오에 가까운 샘플의 품질에 미치는 영향을 최소화합니다.



### Large Language Model with Region-guided Referring and Grounding for CT Report Generation (https://arxiv.org/abs/2411.15539)
Comments:
          10 pages

- **What's New**: 이번 연구는 CT 보고서 생성을 위한 새로운 접근 방식을 제안합니다. 기존의 방법들이 전체 볼륨의 글로벌 특징(Global Features)만을 고려한 반면, Reg2RG는 특정 해부학적 영역에 초점을 맞추어 진단 성능을 향상시킵니다. 이를 통해 세밀한 지역 분석 및 진단에 필요한 정보 제공을 더 잘 수행할 수 있습니다.

- **Technical Details**: 이 연구는 지역 특징을 확보하기 위해 보편적인 세분화 모듈로부터 마스크를 활용합니다. Local Feature Decoupling (LFD) 전략을 통해 고해상도의 세부 정보를 유지하고 전역 특징(Global Features)과 통합하여 상호 지역 관계를 포착합니다. 또한, Region-Report Alignment (RRA) 훈련 전략을 도입하여 보고서의 특정 지역과 명확하게 연결하도록 모델을 훈련시킵니다.

- **Performance Highlights**: 대규모의 3D 흉부 CT 데이터셋을 이용한 실험에서 Reg2RG는 기존의 여러 방법들보다 더 우수한 성능을 기록했습니다. 이 논문에서 제안하는 접근법은 자연어 생성 및 임상 효능 지표 모두에서 더욱 뛰어난 결과를 보여주며, 보고서를 해석하는 데 있어 뛰어난 해석 가능성을 유지합니다.



### MUNBa: Machine Unlearning via Nash Bargaining (https://arxiv.org/abs/2411.15537)
- **What's New**: 이번 연구에서는 Machine Unlearning (MU)의gradient conflict 문제를 해결하기 위해 Nash Bargaining을 기반으로 한 새로운 프레임워크인 MUNBa를 제안합니다. 이 접근법은 두 플레이어, 즉 '망각(Forgetting)'과 '유지(Preservation)'가 상호 협력하는 게임 구성을 통해 최적의 균형을 목표로 합니다. 이 방법은 다양한 이미지 분류 및 생성 작업에서 우수한 성능을 보여줍니다.

- **Technical Details**: MUNBa는 Forgetting 플레이어와 Preservation 플레이어가 발생시키는 gradient 제안을 통해 서로의 이익을 극대화하는 협력적 게임으로 MU를 재구성합니다. Nash 협상 이론에 영감을 받아 두 목표(망각과 유지를 모두 고려)에 대한 구체적인 해결책을 제공합니다. 이러한 접근법은 두 플레이어 간의 gradient conflict를 완화하여 Pareto front로 모델을 유도합니다.

- **Performance Highlights**: MUNBa는 ResNet, CLIP, 및 텍스트-이미지 diffusion 모델을 포함한 다양한 실험을 통해 기존 최첨단 MU 알고리즘과 비교해 뛰어난 성능을 입증했습니다. 특히 샘플 단위의 망각 시나리오에서 gold standard retrain baseline에 근접한 결과를 보여주며, 망각 정밀도, 일반화 유지 및 적대적 공격에 대한 강인성을 향상시켰습니다.



### CellPilo (https://arxiv.org/abs/2411.15514)
- **What's New**: 현재의 연구는 CellPilot라는 혁신적인 프레임워크를 소개하며, 이는 세포 및 분비선의 자동 및 상호작용 세분화를 가능하게 합니다. CellPilot는 675,000개 이상의 마스크를 포함한 아홉 가지 다양한 데이터셋을 바탕으로 훈련되어, 높은 정확성을 제공할 수 있습니다. 또한, 이 모델은 오픈소스로 제공되어, 연구자들이 대규모 주석 데이터셋을 쉽게 생성할 수 있도록 지원합니다.

- **Technical Details**: CellPilot는 SAM의 미세 조정 버전과 CellViT를 조합하여 구성됩니다. CellViT는 세포를 위한 바운딩 박스를 생성하며, 이를 SAM의 프롬프트로 사용하여 초기 세분화를 자동으로 수행합니다. 훈련 과정에서는 이미지 인코더에 저차원 적응(LoRA)을 적용하고, 약 20회의 에포크를 거쳐 최적화된 모델을 만듭니다.

- **Performance Highlights**: CellPilot 모델은 다양한 상호작용 방식과 비교하여 우수한 성능을 보여줍니다. 특히, CellPilot는 SAM, MedSAM, SimpleClick와 같은 기존 모델들과 비교했을 때 상호작용 세분화 작업에서 현저한 성능 개선을 입증했습니다. 이를 통해 비만족스러운 초기 자동 세분화를 사용자 수정이 가능하게 하여 보다 정교한 결과를 도출할 수 있습니다.



### Interactive Visual Assessment for Text-to-Image Generation Models (https://arxiv.org/abs/2411.15509)
Comments:
          Under Review

- **What's New**: 이 논문에서는 DyEval이라는 새로운 동적 인터랙티브 시각 평가 프레임워크를 소개합니다. 이 프레임워크는 LLM(대형 언어 모델)을 활용하여 모델 피드백에 따라 테스트 입력을 적응적으로 생성할 수 있습니다. 기존의 정적 접근 방식의 한계를 극복하여 모델의 복잡한 실패를 효과적으로 식별하고 분석할 수 있도록 돕습니다. 또한 DyEval은 사용자와 생성 모델 간의 협업 평가를 촉진하여 더욱 종합적인 평가를 가능하게 합니다.

- **Technical Details**: DyEval은 LLM을 통해 생성된 계층적이고 세분화된 텍스트 입력을 사용하여 시각적인 모델 동작을 탐색하는 직관적인 사용자 인터페이스를 제공합니다. 평가자는 이 인터페이스를 통해 모델 출력을 평가하고, 평가 결과에 따라 새로운 테스트 주제로 나무(tree) 구조를 깊이 탐색합니다. DyEval은 또한 실패 유발 요소를 분석하는 맥락적 반영 모듈을 통해 모델의 잠재적 실패 패턴도 분석합니다. 이를 통해 사용자에게 더 나은 모델 개선을 위한 해석 가능한 분석을 제공합니다.

- **Performance Highlights**: DyEval은 기존 방법보다 최대 2.56배 더 많은 생성 실패 케이스를 효과적으로 식별할 수 있음을 보여주는 정량적 실험 결과를 제공합니다. 다양한 최신 텍스트-이미지 모델을 평가한 결과, 정적 테스트 방식에서는 발견할 수 없는 복잡하고 희귀한 실패 패턴이 드러났습니다. DyEval은 특히 문화적 맥락과 같은 언어적 세부 사항에서 발생하는 특정 실패를 식별했으며, 이러한 결과는 텍스트-이미지 생성 모델 개선을 위한 귀중한 통찰력을 제공합니다.



### AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation (https://arxiv.org/abs/2411.15497)
- **What's New**: 본 논문은 원거리 감지 이미지 객체 탐지(Remote Sensing Image Object Detection, RSIOD)를 위한 레이아웃 제어가 가능한 확산 생성 모델(AeroGen)을 제안합니다. AeroGen은 수평 및 회전된 바운딩 박스 조건 생성을 지원하는 최초의 모델로, 특정 레이아웃과 객체 카테고리 요구 사항에 맞는 고품질 합성 이미지를 생성할 수 있습니다. 이로 인해 기계 학습 및 데이터 증강에서의 데이터 부족 문제를 효과적으로 해결할 수 있습니다.

- **Technical Details**: AeroGen은 레이아웃 조건 생성기와 필터링 메커니즘을 통합한 종단 간 데이터 증강(end-to-end data augmentation) 프레임워크를 통해 작동합니다. 이러한 접근 방식은 기존의 데이터 증강 메서드와는 다르게, 조건부 생성 모델을 사용하여 실제 탐지 과제를 위한 데이터 생성을 직접 통합합니다. 생성된 데이터는 고품질과 다양성을 보장하여 기존 RSOID 모델의 성능을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, 자로 제작된 합성 데이터는 DIOR, DIOR-R 및 HRSC 데이터셋에서 mAP 점수가 각각 3.7%, 4.3%, 2.43% 개선되며 성능이 크게 향상되는 것을 확인했습니다. 특히 드문 객체 클래스에서의 성능 향상이 두드러지며, GF, DAM, APO 각각에서 17.8%, 14.7%, 12.6%의 향상률을 보였습니다.



### Improving Factuality of 3D Brain MRI Report Generation with Paired Image-domain Retrieval and Text-domain Augmentation (https://arxiv.org/abs/2411.15490)
- **What's New**: 이 논문에서 제안하는 새로운 방법론인 Pair Image-Domain Retrieval and Text-Domain Augmentation (PIRTA)은 영상에서 텍스트로의 직접적인 생성 대신, MRI 이미지와 그에 대응하는 실제 방사선 리포트를 쌍으로 사용하여 정보의 정확성을 높입니다. 특히, AIS 관리에 있어 임상 의사들이 해석할 수 있는 방사선 리포트를 생성함으로써 의료 현장에서의 활용도가 높아질 것으로 기대됩니다. PIRTA는 영상-텍스트 간의 매핑 필요성을 줄여, 디퓨전 가중 영상(DWI)에서 직접적으로 임상적으로 중요한 정보를 생성하는 데 도움을 줍니다.

- **Technical Details**: PIRTA는 MRI 이미지와 그에 맞는 방사선 리포트 간의 인도메인 검색을 통해 정보를 가져오는 방식을 적용하며, 전통적인 이미지-텍스트 매핑의 복잡성을 해소합니다. 특히, 3D DWI 영상을 기반으로 한 방사선 리포트를 생성하는 방식으로, 자기 회귀 모델, 데이터베이스 내 유사 이미지 검색, 그리고 RAG 기술을 활용하여 보고서의 정확성을 한층 높였습니다. 이를 위해 대규모 비라벨링 MRI 데이터셋을 활용하여 3D 비전 인코더를 미리 학습하는 방법도 포함됩니다.

- **Performance Highlights**: PIRTA 접근 방식은 기존의 멀티모달 언어 모델을 직접 사용하여 이미지에서 텍스트로 생성하는 것에 비해 탁월한 정확성을 보였습니다. 실험 결과, PIRTA는 DWI 이미지에서 관련된 방사선 리포트를 보다 정확하게 검색하고 생성하는 데 성공하였으며, 이는 임상 의사들이 신속하게 치료 결정을 내릴 수 있도록 지원하는 데 기여할 것입니다. 이 연구는 AIS의 효과적인 관리와 진단 지원을 위해 매우 중요한 발전을 시사합니다.



### SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field for Autonomous Driving (https://arxiv.org/abs/2411.15482)
- **What's New**: SplatFlow는 Neural Motion Flow Fields (NMFF) 내에서 자기 지도 학습(Self-Supervised Learning)을 통해 4차원(4D) 시간-공간 표현을 학습할 수 있는 새로운 방법입니다. 기존의 동적 도시 환경에서의 통시적 데이터 모델링은 정확한 객체 수준의 감시(supervision) 또는 비싼 수동 라벨링에 의존했으나, SplatFlow는 이러한 필요를 줄이며 정적 배경과 동적 객체를 효과적으로 분리하여 동적 씬 재구성이 가능합니다.

- **Technical Details**: SplatFlow는 NMFF라는 일련의 암묵적 함수들을 통해 LiDAR 포인트와 Gaussians의 시간에 따른 운동을 연속적인 모션 플로우 필드로 모델링합니다. 4D 가우시안(primitives)을 이용해 동적 객체를, 3D 가우시안을 통해 정적 배경을 나타내며, 각 4D 가우시안의 상태 일치(correspondence)를 모델링하여 동적 성분의 시점 간 일관성을 극대화합니다.

- **Performance Highlights**: Waymo Open Dataset 및 KITTI Dataset에서 실시한 평가 결과, SplatFlow는 동적 도시 상황에서의 이미지 재구성과 새로운 시점 시합(Novel View Synthesis) 작업에서 최신 기술(State-of-the-Art)보다 우수한 성능을 입증하였습니다. SplatFlow는 비싼 3D 바운딩 박스 감시 없이도 고급 동적 씬 식별이 가능하여 대규모 데이터 소스에서 학습할 수 있습니다.



### KinMo: Kinematic-aware Human Motion Understanding and Generation (https://arxiv.org/abs/2411.15472)
- **What's New**: 텍스트 기반의 인간 동작 제어는 컴퓨터 비전에서 중요한 도전 과제로, 본 연구에서는 이를 해결하기 위한 새로운 모션 표현 방식을 제안합니다. 기존의 전통적인 접근 방식은 전체 액션에 초점을 맞춰 섬세한 움직임을 포착하지 못하는 한계를 가지고 있습니다. 따라서 본 논문은 움직임을 신체 관절 그룹의 움직임으로 분해하여 다루고, 이를 통해 텍스트와 모션 도메인 간의 간극을 줄이는 방법을 제안합니다.

- **Technical Details**: 우리는 인간 동작의 세 가지 수준: 글로벌 액션, 로컬 관절 그룹, 관절 상호작용으로 정의하는 KinMo라는 새로운 프레임워크를 도입합니다. 이 프레임워크는 자동 데이터 수집 파이프라인을 통해 기존의 텍스트-모션 벤치마크를 개선하여 정밀한 로컬 관절 그룹 모션과 상호작용 설명을 포함합니다. 또한, 우리가 제안한 계층적 모션 의미론 접근 방식을 통해 관절 레벨의 상호작용 정보를 글로벌 액션 레벨의 의미론으로 점진적으로 융합합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 텍스트-모션 검색 성능을 향상시키고, 보다 정확한 관절 움직임 생성을 가능하게 함을 보여주었습니다. 모션 생성 과정은 코스-투-파인(coarse-to-fine) 방식으로 구성되어 다양한 생성 및 편집 응용 분야에서의 적용 가능성을 높이고 있습니다. 이를 통해 언어와 모션 간의 더 나은 일치를 유도할 수 있는 기반을 마련하였습니다.



### Mamba-CL: Optimizing Selective State Space Model in Null Space for Continual Learning (https://arxiv.org/abs/2411.15469)
- **What's New**: 이 논문에서는 지속적 학습(Continual Learning, CL) 분야에서 Mamba 모델을 활용한 새로운 프레임워크인 Mamba-CL을 제안합니다. Mamba-CL은 이전 작업에서 학습한 특성의 서브스페이스와 직교하는 방향으로 매개변수를 업데이트하여 지속적으로 큰 규모의 Mamba 기초 모델의 핵심 State Space Models (SSMs)를 미세 조정합니다. 이 접근 방식은 이전 작업과 현재 작업 간의 일관성을 유지하여 치명적 망각 문제를 극복하는 이론적 보장을 제공합니다.

- **Technical Details**: Mamba-CL은 Mamba 모델 내의 SSM 블록을 지속적으로 미세 조정하는 방법론을 제안하며, 이를 통해 기존 작업의 특성 공간과 직교하는 방향으로 업데이트를 수행합니다. 논문에서 소개된 네 가지 독립적인, 시간 불변의 매개변수에 대한 일관성 제약 조건을 통해 골격 및 비선형 이산화 과정에서 발생하는 장애를 피하면서도 치명적 망각을 예방할 수 있도록 설계되었습니다. 이를 위해 null-space 기반의 근사 솔루션을 활용하여 효율적인 그래디언트 직교 프로젝션을 구현합니다.

- **Performance Highlights**: Mamba-CL은 10-split과 20-split ImageNet-R, 10-split CIFAR-100, 10-split DomainNet와 같은 다양한 클래스 증분 벤치마크에서 실험을 수행하였는데, 그 결과 치명적 망각을 완화하는 데 매우 효과적임을 보여주었습니다. 실험 결과는 기존 최첨단 방법들보다 우수한 성능을 나타내며, Mamba 모델의 지속적 적응을 지원하는 데 성공적임을 입증하였습니다.



### SplatSDF: Boosting Neural Implicit SDF via Gaussian Splatting Fusion (https://arxiv.org/abs/2411.15468)
- **What's New**: 이 논문에서 제안하는 새로운 방법론인 SplatSDF는 3DGS와 SDF-NeRF를 아키텍처 수준에서 통합하여 기하학적(geometric) 및 광학적(photometric) 정확도와 수렴 속도를 크게 향상시킵니다. 이는 기존 방법들이 주로 손실 수준에서의 개선에 초점을 맞춘 것과는 다릅니다. SplatSDF는 학습 시 3DGS를 입력으로만 사용하기 때문에 추론(inference) 단계에서 SDF-NeRF의 원래 복잡성과 효율성을 유지하면서도 성능을 개선할 수 있습니다.

- **Technical Details**: SplatSDF는 3DGS를 입력으로 활용하여 SDF를 비연속적인 표면(mesh)으로 표현하는 대신 연속적인 Signed Distance Function(SDF)으로 회복하는 데 중점을 두고 있습니다. 이 방법은 multi-layer perceptron (MLP)을 사용하여 SDF 값을 회귀하고, volumetric rendering 기법을 통해 supervision을 수행합니다. 또한 Chamfer distance 및 peak signal-to-noise ratio (PSNR) 등의 성능 지표에서 이전 SOTA SDF-NeRF 방법을 초과하는 결과를 보였습니다.

- **Performance Highlights**: SplatSDF는 기하학적 및 광학적 정확도 면에서 SOTA SDF-NeRF 및 3DGS 표면 재구성 방법을 초과하는 성과를 내었습니다. 특히 복잡한 형태에 대한 3배 이상의 빠른 수렴 속도를 달성하면서도 동일한 기하학적 정확도를 유지합니다. 이러한 성과는 SDF-NeRF의 기존 모델과 비교할 때 효과적인 개선을 나타냅니다.



### Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator (https://arxiv.org/abs/2411.15466)
- **What's New**: 이번 논문에서는 Diptych Prompting이라는 새로운 제로샷(Zero-shot) 접근법을 소개합니다. 이 방법은 대규모 텍스트-이미지 모델의 Diptych 생성을 활용하여 주제 정렬을 보다 정밀하게 수행합니다. Diptych Prompting은 참조 이미지를 왼쪽 패널에 배치하고, 오른쪽 패널에서 텍스트 조건화(inpainting)를 활용하여 주제 기반 이미지를 생성합니다.

- **Technical Details**: Diptych Prompting에서는 참조 이미지의 배경을 제거하여 불필요한 내용 유출을 방지하고, 두 패널 간의 주의(attention) 가중치를 강화하여 생성되는 주제의 세부 사항을 개선합니다. 이 방법은 텍스트 프롬프트와 참조 이미지를 모두 고려하여 고품질 이미지를 생성하는 데 중점을 둡니다. 실험 결과에 따르면, 이 접근법은 기존의 제로샷 이미지 프롬프트 방법보다 뛰어난 성능을 보였습니다.

- **Performance Highlights**: Diptych Prompting은 주제 주도 이미지 생성뿐만 아니라 스타일화된 이미지 생성 및 주제 기반 이미지 편집에도 적용 가능합니다. 사용자 평가 결과, Diptych Prompting이 생성한 이미지가 시각적으로 더 선호되는 것으로 나타났습니다. 이는 본 접근법의 다양한 이미지 생성 응용 프로그램에서의 다재다능성을 보여줍니다.



### MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking (https://arxiv.org/abs/2411.15459)
- **What's New**: 이번 연구에서는 효율적인 장기 시퀀스 모델링을 위한 Mamba 기반의 비전-언어 추적 모델인 MambaVLT를 제안합니다. 전통적인 Transformer 기반의 방법들이 시간 정보를 적절히 활용하지 못하는 문제를 해결하기 위해, Mamba의 상태 공간을 이용하여 다중 모달 추적을 수행합니다. 특히, 시간 진화 하이브리드 상태 공간 블록과 선택적 지역 증대 블록을 통합하여 문맥 정보를 잘 캡처할 수 있도록 설계하였습니다.

- **Technical Details**: MambaVLT는 상태 공간 메모리를 통한 장기 대상 정보를 기억하고, 이를 기반으로 참조 특징을 선택적으로 업데이트합니다. 이 과정에서 Hybrid Multimodal State Space (HMSS) 블록과 Selective Locality Enhancement (SLE) 블록이 결합되어 다양한 모달 특징을 효율적으로 통합합니다. 최종적으로, 모달리티 선택 모듈을 통해 시각적 및 언어적 참조의 가중치를 동적으로 조정함으로써 추적 성능을 향상시킵니다.

- **Performance Highlights**: MambaVLT는 TNL2K, LaSOT, OTB99, MGIT 등 다양한 벤치마크에서 최첨단 추적기들과 비교할 때 우수한 성능을 보여주었습니다. 연구에서는 MambaVLT의 효율성을 검증하기 위해 extensive 실험을 실시하였으며, 상태 공간 메모리의 효과적인 활용이 장기적인 목표 정보 저장에 큰 기여를 하였음을 입증하였습니다.



### Enhancing Instruction-Following Capability of Visual-Language Models by Reducing Image Redundancy (https://arxiv.org/abs/2411.15453)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 instruction-following 능력을 향상시키기 위해 Visual-Modality Token Compression (VMTC) 및 Cross-Modality Attention Inhibition (CMAI)라는 전략을 제안합니다. MLLMs의 instructional 수행 능력은 기존의 Large Language Models (LLMs)보다 부족하다는 점을 강조하며, 시각적 토큰의 다운샘플링이 이 능력을 향상시킬 수 있다는 것을 발견했습니다. 그러나 간단한 다운샘플링 방법이 MLLMs의 멀티모달 이해 능력을 저해하는 문제도 존재합니다.

- **Technical Details**: VMTC는 시각적 데이터의 불필요한 정보를 압축하면서도 중요한 전경 정보를 유지하는 모듈로, ViT 레이어에서 주의 점수를 활용하여 불필요한 배경 토큰을 식별하고 클러스터링 후 결합합니다. CMAI는 LLM의 텍스트 토큰이 관련 이미지 토큰에만 집중하도록 도와주는 모듈로, 낮은 텍스트-이미지 집중 점수를 가진 텍스트-이미지 토큰 쌍 간의 주의를 약화하도록 설계되었습니다. 이 두 모듈은 MLLMs의 instruction-following 능력을 개선하면서 멀티모달 이해 능력을 정확히 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, VMTC 및 CMAI를 통합한 방법이 MLLMs의 instruction-following 능력을 크게 향상시켰으며, 여러 벤치마크에서 SOTA 성능을 달성했습니다. 이 연구는 MLLMs의 instruction-following 능력과 시각적 모드의 정보 중복 간의 상관관계를 탐구한 최초의 연구로, 두 가지 문제를 해결하는 새로운 기법을 제시합니다. 제안된 접근법은 MLLMs의 기존 성능을 유지하면서 instruction-following 능력을 크게 개선할 수 있음을 보여줍니다.



### freePruner: A Training-free Approach for Large Multimodal Model Acceleration (https://arxiv.org/abs/2411.15446)
- **What's New**: 본 논문에서는 기존의 retraining 없이 어떤 오픈소스 LMM에도 바로 적용할 수 있는 training-free token reduction 방법인 freePruner를 제안합니다. 기존 방법들은 token merging 작업에 크게 의존하는 반면, freePruner는 두 단계의 token selection 전략을 통해 고수준 의미 정보와 저수준 시각 정보를 효과적으로 포착합니다. 이를 통해, LMM의 성능을 유지하면서 최대 2배의 속도 향상을 이루었습니다.

- **Technical Details**: freePruner의 핵심은 Pivotal Token Selection과 Complementary Token Selection 두 가지 전략입니다. 첫 번째 단계에서는 설계된 기여도 지표를 이용해 고수준 의미 정보를 담고 있는 주요 토큰을 추출합니다. 두 번째 단계에서는 주어진 시각 정보와 관련된 추가 토큰을 선택하여 저수준의 시각적 세부정보를 유지하는 방식입니다. 이는 기존 방법들과 달리 모델 retraining 없이 진행되어, 훨씬 더 실용적인 접근 방식이라 할 수 있습니다.

- **Performance Highlights**: 실험 결과, freePruner는 기존의 여러 LMM 벤치마크에서 비교 가능한 성능을 유지하면서도, 약 2배의 추론 속도 향상을 달성했습니다. 또한, freePruner는 post-training quantization과 같은 다른 후처리 가속기법과도 독립적으로 결합하여 사용할 수 있는 가능성을 보여주며, 효율적인 LMM 배포를 위한 실용적인 솔루션이 될 것입니다.



### Twin Trigger Generative Networks for Backdoor Attacks against Object Detection (https://arxiv.org/abs/2411.15439)
Comments:
          13 pages, 8 figures

- **What's New**: 본 논문에서는 객체 탐지(Object Detection) 모델에 대한 백도어 공격(Backdoor Attack)의 취약성을 다룹니다. 기존 연구에 비해 많은 연구가 이미지 분류에 집중된 반면, 이 방법론은 객체 탐지 분야에 대한 새로운 접근 방식을 제안하며, 보이지 않는 트리거(invisible trigger) 생성 네트워크와 보이는 트리거(visible trigger)를 활용해 공격의 은폐성을 높이고 있습니다. 이는 잦은 사용자 데이터 요구로 인해 제3자가 제공한 데이터셋이나 사전 훈련된 모델을 사용하는 경우 증가하는 보안 위험에 대응하기 위한 것입니다.

- **Technical Details**: 이 연구의 핵심은 두 가지 트리거 생성 네트워크를 제안하는 것입니다: 보이지 않는 트리거 생성 네트워크(TGN1) 및 보이는 트리거 생성 네트워크(TGN2). TGN1은 가우시안 스무딩 레이어(Gaussian Smoothing Layer)와 고주파 아티팩트 분류기를 포함하여 객체 탐지기에서 백도어의 은폐성을 중점적으로 강화합니다. 반면 TGN2는 보이는 트리거의 최적화를 위해 새로운 정렬 손실을 설계하여 보이지 않는 트리거 행동과 일치하도록 합니다. 이 두 네트워크는 훈련 및 추론 단계에서 서로 다른 트리거를 사용하여 공격 은폐성을 극대화합니다.

- **Performance Highlights**: 대규모 데이터셋인 COCO에서 실시된 실험 결과, 제안된 두 개의 트리거 생성 네트워크는 YOLOv5와 YOLOv7과 같은 객체 탐지 모델에 대해 각각 70.0%와 84.5%라는 높은 강화 효과를 구현했습니다. 이는 연구자들이 기존의 방법들보다 우수한 성능을 확인하게 해주는 결과이며, 공격 방식의 일반화 가능성을 보여줍니다. 특히 다른 탐지 모델에 대한 블랙박스 공격에서도 일부 일반화가 이루어짐을 확인했습니다.



### ConsistentAvatar: Learning to Diffuse Fully Consistent Talking Head Avatar with Temporal Guidanc (https://arxiv.org/abs/2411.15436)
- **What's New**: 본 논문에서는 ConsistentAvatar라는 새로운 프레임워크를 제안하여 고충실도(high-fidelity)와 완전 일관성(fully consistent)을 가진 talking avatar 생성을 가능하게 합니다. 기존의 방식들은 주로 단일 이미지 생성 능력의 한계와 오류 누적으로 인해 시간적, 3D 또는 표정 일관성이 부족했습니다. ConsistentAvatar는 Temporally-Sensitive Detail (TSD) 맵을 사용하여 인접 프레임 간의 안정성을 모델링하고 이를 바탕으로 최종적인 아바타를 생성하는 방법론을 도입합니다.

- **Technical Details**: ConsistentAvatar는 시간적으로 일관된 표현을 먼저 생성하여 다른 조건들을 안내하는 구조로 설계되었습니다. TSD는 고주파 정보와 프레임 간 큰 변화가 있는 윤곽선을 포함하고 있으며, 노이즈.condition들을 제어하기 위해 이 TSD를 정렬하는 시간 일관성 확산 모듈을 사용합니다. 최종 생성 아바타는 정렬된 TSD, 대략적인 머리 노말, 감정 임베딩이 결합되어 생성됩니다.

- **Performance Highlights**: 실험 결과에 따르면, ConsistentAvatar는 생성된 외형, 3D, 표정 및 시간적 일관성에서 최신 기술들보다 뛰어난 성능을 보였습니다. 이 프레임워크는 다양한 조건을 보완하여 결과의 일관성을 높일 뿐만 아니라, 우수한 품질의 아바타 생성을 실현할 수 있음을 입증합니다. 향후 가상현실 및 관련 응용분야에서 진정한 인간 아바타 생성의 가능성을 확장할 수 있을 것으로 기대됩니다.



### What Makes a Scene ? Scene Graph-based Evaluation and Feedback for Controllable Generation (https://arxiv.org/abs/2411.15435)
- **What's New**: 이 논문에서는 Scene-Bench라는 새로운 벤치마크를 소개하며, 자연 장면 생성에서 사실적 일관성을 평가하고 향상시키기 위해 MegaSG라는 대규모 데이터셋을 포함하고 있습니다. MegaSG에는 하나의 백만 개 이미지가 장면 그래프와 함께 주석이 달려 있어, 다양한 복잡한 장면에서 모델의 교육과 공정한 비교를 가능하게 합니다. 또한, 기존의 평가 지표보다 더 효과적인 사실적 일관성 측정을 위한 새로운 평가 지표 SGScore를 제안합니다.

- **Technical Details**: SGScore는 멀티모달 대형 언어 모델(LLM)의 체이닝 사고 체계를 활용하여 생성된 이미지와 해당 장면 그래프 간의 객체 존재 및 관계 정확성을 평가합니다. 이를 통해 객체 회수(Object Recall) 및 관계 회수(Relation Recall)를 확인하고, 복소한 장면 생성에서의 문제를 해결합니다. 또한, 사용자 정의된 장면 그래프 피드백 파이프라인이 개발되어, 생성된 이미지의 불일치를 반복적으로 정제하는 과정을 포함합니다.

- **Performance Highlights**: 실험 결과, Scene-Bench는 복잡한 장면 생성을 위한 기존 벤치마크보다 더 포괄적이고 효과적인 평가 프레임워크를 제공한다고 입증되었습니다. 제안된 피드백 전략은 생성된 이미지의 사실적 일관성을 크게 향상시키며, 컨트롤 가능 이미지 생성 분야의 발전에 기여합니다. 제안된 요소들이 통해 생성 모델의 성능이 향상되고, 사실적 일관성을 확보할 수 있습니다.



### LDM-Morph: Latent diffusion model guided deformable image registration (https://arxiv.org/abs/2411.15426)
- **What's New**: 이 논문에서는 의료 영상 등록을 위한 비지도형 변형 등록 알고리즘인 LDM-Morph를 제안합니다. LDM-Morph는 잠재확산모델(LDM)에서 추출한 특징을 이용하여 의미 정보를 풍부하게 만들고, 라벨이 없는 방식으로 효과적으로 변형을 예측합니다. 또한, 이미지 쌍의 유사성을 평가하기 위해 다양한 메트릭을 도입하여 등록 정확성을 향상시킵니다.

- **Technical Details**: LDM-Morph는 잠재적 및 전역 특징 기반의 크로스-어텐션 모듈(LGCA)을 사용하여 LDM의 의미 정보와 다중 헤드 셀프 어텐션으로부터의 글로벌 정보를 결합합니다. 이러한 이중 스트림 인코더 구조를 통해 이미지 쌍 간의 고수준 해부학적 특징을 더 효과적으로 정렬할 수 있습니다. 하위 구조에서 정의된 계층적 유사성 메트릭은 픽셀 수준과 의미 수준 모두에서 이미지 쌍을 비교합니다.

- **Performance Highlights**: LDM-Morph는 기존의 CNN 및 Transformer 기반 방법들에 비해 우수한 정확성과 해부학적 형체 유지 능력을 보여주었습니다. 특히, 병리적 변화가 있는 데이터셋에서 적절한 변형 예측을 통해 등록 성능을 크게 향상했습니다. 또한 이 방법은 상대적으로 계산 효율성이 뛰어난 결과를 유지하면서도 높은 성능을 발휘합니다.



### OphCLIP: Hierarchical Retrieval-Augmented Learning for Ophthalmic Surgical Video-Language Pretraining (https://arxiv.org/abs/2411.15421)
- **What's New**: 이번 연구에서는 OphCLIP라는 새로운 계층적 재조정 기반 동시 언어-비전 사전 학습 프레임워크를 제안합니다. OphCLIP은 375K개의 비디오-텍스트 쌍을 포함하는 OphVL 데이터 세트를 사용하여 안과 수술 흐름을 이해하는 데 중점을 두었습니다. 또한, 이 프레임워크는 소음이 없는 수술 비디오를 활용하여 학습 기회를 극대화하는 방식을 도입합니다.

- **Technical Details**: OphVL 데이터 세트에는 수술 유형, 단계 및 기구, 약물 등 다양한 속성을 포함한 375K개의 비디오-텍스트 쌍이 포함되어 있습니다. OphCLIP은 짧은 비디오 클립과 잘 구조화된 제목으로 긴 비디오와의 조합을 통해 미세하고 장기적인 시각적 표현 학습을 가능하게 합니다. 재조정 기반 보강 방법을 통해, 이 프레임워크는 유사한 내용을 가진 소음이 없는 비디오를 자동으로 검색하여 지식 전이를 지원합니다.

- **Performance Highlights**: OphCLIP은 11개의 데이터 세트에서 단계 인식 및 다중 기구 식별 작업을 수행한 결과로서 우수한 일반화 성능을 보였습니다. 이 모델은 다양한 안과 수술에 대한 강력한 전이 가능 표현을 학습하고 제로 샷 성능에서 최첨단 결과를 달성했습니다. 이를 통해 OphCLIP은 수술 멀티모달 제시 학습에 기여할 수 있는 위치에 있습니다.



### Semi-supervised Single-view 3D Reconstruction via Multi Shape Prior Fusion Strategy and Self-Attention (https://arxiv.org/abs/2411.15420)
- **What's New**: 이 논문에서는 단일 시점 3D 재구성을 위한 새로운 반지도 학습(semisupervised learning) 프레임워크를 제안합니다. 이 프레임워크는 다수의 형태 사전 융합(multi shape prior fusion) 전략을 도입하여 보다 현실적인 객체 구조 생성을 유도합니다. 또한, 기존 디코더에 자기 주의(self-attention) 모듈을 통합하여 형태 생성 품질을 향상시켰습니다.

- **Technical Details**: 제안된 방법은 2D 이미지로부터 3D 구조를 추정하는 과정에서, 정적 점구름(point cloud) 재구성을 활용합니다. 초기 구형 점구름에서 선별된 형태의 3D 후보를 생성하기 위해 KMeans 등의 클러스터링 알고리즘을 적용하며, Chamfer 거리(chamfer distance)를 사용하여 평균 형태의 카테고리 점구름을 추출합니다. 또한 디코더의 성능을 높일 수 있도록 자기 주의 메커니즘을 적용하였습니다.

- **Performance Highlights**: ShapeNet 데이터셋에서의 벤치마크 테스트 결과, 본 방법이 다양한 레이블 비율(1%, 10%, 20%)에서 기존의 지도 학습 방법보다 우수한 성능을 보였습니다. 실제 Pix3D 데이터셋에서도 뛰어난 성능을 보여주었으며, ShapeNet에서의 실험을 통해 기준선 대비 3.3%의 성능 향상을 기록했습니다. 또한, 엄격한 ablation 연구를 통해 제안하는 접근법의 효과성을 추가로 확인하였습니다.



### FG-CXR: A Radiologist-Aligned Gaze Dataset for Enhancing Interpretability in Chest X-Ray Report Generation (https://arxiv.org/abs/2411.15413)
Comments:
          ACCV 2024

- **What's New**: 해당 연구는 방사선학적 진단을 돕기 위해 설명 가능한 시스템인 Gen-XAI를 제안하였습니다. 이 시스템은 방사선 전문의의 시선 주의정보(eye gaze)를 기반으로 CXR(Chest X-ray) 분석을 통해 보고서를 생성합니다. 또한, Fine-Grained CXR (FG-CXR)라는 새로운 데이터셋을 소개하여 방사선 전문의의 시선 추적 정보와 진단 내용을 보다 정밀하게 정렬했습니다.

- **Technical Details**: Gen-XAI는 방사선 전문의의 시선을 예측하는 Gaze Attention Predictor와 이를 기반으로 보고서를 생성하는 Report Generator로 구성되어 있습니다. Gaze Attention Predictor는 전문의의 주의력을 나타내는 데이터로부터 관심 있는 영역을 학습하여, 보고 생성 과정에서 이를 활용합니다. FG-CXR 데이터셋은 방사선 전문의가 주의하는 7개의 해부학적 영역에 대한 세부정보를 포함하고 있으며, 이는 기존 데이터셋에서의 미묘한 불일치를 해결하여 정확한 진단을 지원합니다.

- **Performance Highlights**: 연구에서는 Gen-XAI의 효율성을 입증하기 위해 광범위한 실험을 수행하였습니다. FG-CXR 데이터셋을 활용하여 방사선 전문의의 진단 과정과 주의 패턴을 근거로 한 보고서 생성을 성공적으로 수행했습니다. 이로 인해 방사선 전문의의 진단 정보와 시스템의 출력 간의 상관관계가 향상되어, 더 높은 수준의 해석 가능성과 신뢰성을 제공합니다.



### FINECAPTION: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity (https://arxiv.org/abs/2411.15411)
Comments:
          Preprint

- **What's New**: 대형 Vision-Language Model (VLM)의 발전은 멀티모달 작업에서 상당한 진전을 가져왔습니다. 이 모델은 다양한 응용 프로그램에서 이미지 및 비디오 자막 작성, 시각적 질문 응답, 크로스 모달 검색 등에서 더욱 정교하고 정확한 추론을 가능하게 합니다. 그러나 VLM은 세밀한 이미지 지역 조합 정보 인식에서 어려움을 겪고 있으며, 이를 해결하기 위해 우리는 새로운 VLM인 FINECAPTION과 새로운 데이터셋인 CompositionCap을 제안합니다.

- **Technical Details**: FINECAPTION은 임의의 마스크를 참조 입력으로 인식하고, 다양한 세분화 수준에서 조합된 이미지 자막 작성을 위해 고해상도 이미지를 처리할 수 있는 새로운 VLM입니다. 모델은 마스크 인식 저해상도 인코더와 여러 고해상도 인코더를 통합한 새로운 아키텍처로 설계되었습니다. 이를 통해 세부적인 조합 정보를 포착하고 마스크 참조 지역을 정확히 인식할 수 있습니다.

- **Performance Highlights**: 실험 결과, FINECAPTION은 GPT-4 및 LLaMA-3.2와 같은 다른 강력한 VLM에 비해 지역 조합 이미지 자막 작성 작업에서 우수한 성능을 보였습니다. 우리의 모델은 Attribute-Aware Regional Captioning, Regional Dense Captioning, Comprehensive Global Image Captioning의 세 가지 자막 세분화 수준을 제공합니다. CompositionCap 데이터셋은 18가지의 다양한 조합 속성을 포함하여 높은 품질의 데이터셋을 제공합니다.



### Efficient Online Inference of Vision Transformers by Training-Free Tokenization (https://arxiv.org/abs/2411.15397)
- **What's New**: 이번 논문에서는 $	extbf{Visual Word Tokenizer}$ (VWT)라는 새로운 방법을 제안합니다. 이 방법은 기존의 압축 기법에서 흔히 요구되는 추가적인 end-to-end fine-tuning을 필요로 하지 않으며, 에너지 비용을 절감하면서 성능과 런타임을 유지할 수 있습니다. VWT는 자주 사용되는 비주얼 서브워드(visual subwords)를 비주얼 워드(visual words)로 그룹화하여 효과적으로 이미지를 압축합니다.

- **Technical Details**: VWT는 주로 두 가지 접근 방식을 사용하여 시각적 라벨링을 수행합니다. 첫 번째는 intra-image 접근 방식으로, 각 이미지 내에서 가장 낮은 픽셀 분산을 가진 패치를 그룹화하는 방식입니다. 두 번째 접근 방식인 inter-image는 여러 이미지에서 기본 특성(색상이나 가장자리 등)을 찾아 시각적 워드를 발견하여 패치를 그룹화합니다.

- **Performance Highlights**: 실험 결과, VWT를 사용하면 에너지 소모를 최대 19%까지 줄일 수 있으며, 런타임 증가율은 최대 20%로 측정되었습니다. 기존의 8비트 양자화(quantization) 및 토큰 병합(token merging) 접근 방식에 비해 에너지 효율성이 낮거나 유사한 결과가 나오지만, 이 방법은 런타임 저하가 더 크고 최대 2배의 성능 저하를 초래합니다. VWT는 효율적인 온라인 추론에 적합하여 성능 저하를 최소화합니다.



### Gradient-Free Classifier Guidance for Diffusion Model Sampling (https://arxiv.org/abs/2411.15393)
- **What's New**: 이번 연구에서는 이전에 훈련된 분류기를 활용하여 gradient descent(그래디언트 하강법)을 사용하지 않고 효율적으로 샘플링하는 새로운 방법인 Gradient-free Classifier Guidance (GFCG)를 제안하고 있습니다. GFCG는 시간마다 적응적으로 참조 클래스와 해당 안내 스케일을 결정하여 고충실도의 이미지를 생성하는 데 도움이 됩니다. 또한, GFCG는 기존의 다른 가이딩 방법인 Autoguidance (ATG)와 결합하여 이미지 품질을 향상시키면서도 계산 부담을 주지 않는 성과를 보여줍니다.

- **Technical Details**: GFCG는 비강화적 샘플을 피하기 위해 참조 클래스를 사용하고, 훈련된 분류기를 기반으로 샘플링 중 적응적으로 가이드를 조정합니다. 이는 CG(클래시파이어 가이던스)와 비교하여 고급 정보에 대한 비용을 줄이면서 클래스 레이블과의 정렬을 개선합니다. 연구에서는 GFCG를 다양한 확산 모델에 적용하고, 그 효율성과 효과성을 평가하기 위해 혼합 가이드 및 추가 가이드 방법을 비교 분석했습니다.

- **Performance Highlights**: 실험 결과, GFCG는 이미지넷 데이터셋에서 512x512 해상도로 $	ext{FD}_{	ext{DINOv2}}$ 23.09의 기록을 달성하면서 ATG에 비해 높은 분류 정밀도(94.3%)를 기록했습니다. GFCG는 이미지 품질과 다양성을 모두 강화하며, 다양한 모델에서 분류 정확도를 현저히 향상시켰습니다. 마지막으로, GFCG는 다른 가이드 방법과의 조합에서도 긍정적인 결과를 보였습니다.



### Hatching-Box: Monitoring the Rearing Process of Drosophila Using an Embedded Imaging and in-vial Detection System (https://arxiv.org/abs/2411.15390)
Comments:
          17 pages, 6 figures

- **What's New**: 본 논문에서는 Drosophila의 발달 행동을 자동으로 모니터링하고 정량화하는 새로운 시스템인 Hatching-Box를 제안합니다. 이 시스템은 맞춤형 이미징 하드웨어와 전용 탐지 및 추적 알고리즘을 결합하여, 여러 날에 걸쳐 유충, 알이 든/비어 있는 고치, 성충의 수를 정량화할 수 있도록 합니다. 이를 통해 전통적인 실험 방식이 필요 없어지고, 보편적으로 적용 가능한 소프트웨어와 함께 저렴하고 재현 가능한 디자인으로 스케일 확장이 용이합니다.

- **Technical Details**: Hatching-Box는 최대 세 개의 표준 재배 바이알을 수용할 수 있으며, 이를 통해 Incubator 또는 배양실 내에서 사용할 수 있습니다. 이 시스템은 YOLO 기반의 객체 탐지 알고리즘을 사용해 약 470,000개의 수동 레이블이 붙은 객체를 학습시키며, Drosophila의 모든 발달 단계를 정확하게 인식할 수 있습니다. 온도와 습도 모니터링을 위한 센서와 조명 조건을 제어하기 위한 조명 자극 기능도 통합되어 있어, 실험의 다양한 요구 사항을 충족할 수 있습니다.

- **Performance Highlights**: Hatching-Box는 Drosophila의 생애 주기 연구와 같은 장기적 실험에서의 적용 가능성을 보여주며, Konopka와 Benzer의 잘 확립된 서카디안 실험 결과를 성공적으로 재현했습니다. 정확도는 91%에 달하며, 수동 노동 없이도 자연스러운 행동을 정량화하는 데 효과적입니다. 이 시스템은 다양한 양육 바이알을 동시에 모니터링할 수 있어 연구의 효율성을 더욱 높입니다.



### A Constrast-Agnostic Method for Ultra-High Resolution Claustrum Segmentation (https://arxiv.org/abs/2411.15388)
Comments:
          14 pages, 10 figures, 3 tables

- **What's New**: 이 연구에서는 claustrum의 자동 세분화를 위한 방법을 제안합니다. 기존의 저해상도 MRI 스캔에서는 명확하게 관찰할 수 없는 claustrum 구성을 고해상도(0.35 mm isotropic)에서 정확하게 세분화할 수 있는 접근법을 제공합니다. 이 방법은 SynthSeg 세분화 프레임워크를 기반으로 하여 합성된 훈련 강도 이미지를 사용하여 일반화 능력을 개선합니다.

- **Technical Details**: 제안된 세분화 방법은 고해상도 MRI 스캔에서만 적용되던 수동 레이블을 활용하여, 자동으로 claustrum을 분할합니다. 연구팀은 18개의 고해상도 MRI 스캔에서 수동 레이블을 사용하여 깊은 학습 네트워크를 훈련시켰으며, 우수한 성능을 나타냈습니다. 이 방법은 대조 및 해상도 변화에 강인성을 지니며, 다양한 тип의 MRI에서 성공적으로 적용되었습니다.

- **Performance Highlights**: 연구 결과는 Dice score가 0.632, 평균 표면 거리(mean surface distance)가 0.458 mm, 그리고 볼륨 유사성이 0.867로 나타났습니다. 이러한 성과는 6배 교차 검증을 통해 검증되었으며, 일반적인 해상도의 T1 가중 MRI에서도 성능을 입증했습니다. 이는 자동 초고해상도 claustrum 세분화를 위한 최초의 정밀한 방법으로, neuroimaging 패키지인 Freesurfer의 일환으로 제공됩니다.



### Exploiting Watermark-Based Defense Mechanisms in Text-to-Image Diffusion Models for Unauthorized Data Usag (https://arxiv.org/abs/2411.15367)
- **What's New**: 이 연구에서는 텍스트-이미지(T2I) diffusion 모델에서 사용되는 다양한 watermark 기반 보호 방법의 강인성을 조사합니다. 기존의 여러 이미지 변환이 watermark 효과를 제거하는 데 효과적이지 않음을 관찰했으며, 이를 해결하기 위해 새로운 기법인 Rattan을 제안합니다. Rattan은 노이즈가 있는 이미지를 다루는 diffusion 프로세스를 활용하여 보호된 입력의 고수준 특징을 보존하고 낮은 수준의 디테일은 무시하는 방식으로 작동합니다.

- **Technical Details**: Rattan은 보호된 이미지를 입력으로 사용하여 T2I 모델이 새로운 이미지를 생성하도록 하며, 이 과정에서 텍스트와 결합하여 고유한 출력을 생성합니다. 이 방법은 기존의 이미지 변환 접근 방식을 고수준 특징 추출로 전환하여, watermark와 같은 세부 사항을 제거합니다. 연구 결과, Rattan을 사용하면 기존 보호 방법의 탐지율을 상당히 낮출 수 있으며, 이는 우연히 발생할 확률과 유사합니다.

- **Performance Highlights**: 세 개의 데이터셋과 140개의 T2I diffusion 모델에 대한 실험 결과, Rattan은 기존 보호 방법의 탐지율을 50%로 줄였으며, 이는 무작위 추측과 비슷한 수준입니다. 이러한 결과는 Rattan이 watermark 기반 데이터 보호를 강화하는 데 있어 효과적인 방법임을 시사합니다. Rattan을 통해 T2I 모델의 저작권 침해 및 개인정보 보호 문제를 해결하는 데 기여할 수 있을 것으로 기대됩니다.



### UniGaussian: Driving Scene Reconstruction from Multiple Camera Models via Unified Gaussian Representations (https://arxiv.org/abs/2411.15355)
Comments:
          Technical report

- **What's New**: 이 논문은 도시 장면 재구성을 위한 새로운 접근 방식인 UniGaussian을 제안합니다. 이전 방법들이 핀홀 카메라에 집중한 반면, UniGaussian은 핀홀 및 어안(fisheye) 카메라 모델을 통합해 3D Gaussian 표현을 학습합니다. 실제로, 어안 카메라에서 발생하는 광선 왜곡 문제를 해결하기 위한 미분 렌더링 방법을 도입하여 실시간 렌더링을 보장합니다.

- **Technical Details**: 본 연구는 다양한 카메라 모델에서 3D Gaussian 표현을 학습하는 새로운 프레임워크를 설계합니다. 특히, 어안 카메라 모델에 적합한 일련의 아핀 변환을 통해 3D Gaussian의 변형을 적용합니다. 이 프레임워크는 여러 센서 및 모달리티(예: 깊이, 의미론적, 노말 및 LiDAR 포인트 클라우드)를 모델링하여 운전 장면에 대한 포괄적인 이해를 이룹니다.

- **Performance Highlights**: 실험 결과, UniGaussian은 운전 장면 시뮬레이션을 위한 뛰어난 렌더링 품질과 빠른 속도를 달성하였습니다. 이 방법은 다양한 미분 가능 카메라 모델에 대한 적응력을 보유하고 있으며, 실시간 성능을 유지하는 동시에 여러 센서 및 모달리티를 효과적으로 통합합니다.



### Zero-Shot Coreset Selection: Efficient Pruning for Unlabeled Data (https://arxiv.org/abs/2411.15349)
- **What's New**: 이번 논문에서는 라벨이 없는 데이터를 위한 코어셋 선택의 필요성을 제기하고, 이를 형식화합니다. 제안된 방법인 Zero-Shot Coreset Selection (ZCore)은 기존의 데이터 기반 훈련 없이도 효과적인 코어셋을 선택할 수 있도록 합니다. ZCore는 기존의 파운데이션 모델을 활용하여 라벨이 없는 데이터의 제로샷 임베딩 공간을 생성하며, 이렇게 생성된 임베딩 분포를 기반으로 각 데이터의 상대적 중요성을 정량화합니다.

- **Technical Details**: ZCore는 코어셋 선택을 위해 각 데이터 예제의 커버리지(coverage)와 중복성(redundancy)을 평가합니다. 기존의 코드와 라벨을 사용하지 않고도 효율적으로 임베딩 공간을 생성하며, 데이터 예제의 중요성을 평가하기 위해 공통의 기준을 적용합니다. 이 방법은 기존의 라벨 기반 방법들과는 다른 접근으로, 대규모 데이터셋 처리에 적합합니다.

- **Performance Highlights**: ZCore는 네 가지 데이터셋에서 평가되었으며, 여러 최첨단 라벨 기반 방법들을 초월하는 성능을 보였습니다. 특히, ImageNet에서 ZCore를 사용한 경우 10%의 훈련 데이터로도 53.99%의 모델 정확도를 달성하여 라벨 기반 방법보다 우수한 결과를 나타냈습니다. 또한, 115만 개의 이미지에 대한 주석 요구를 제거하여 데이터 주석 비용을 크게 절감할 수 있음을 보여줍니다.



### MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs (https://arxiv.org/abs/2411.15296)
Comments:
          Produced by MME+MMBench+LLaVA Teams. Project Page: this https URL

- **What's New**: 이번 논문에서는 인공지능 일반화(AI General Intelligence, AGI)의 중요한 방향으로 여겨지는 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 평가 방법에 대한 포괄적인 조사 결과를 제시합니다. 기존의 LLM들을 기반으로 하여 MLLMs는 시각과 오디오를 포함한 다양한 정보 형태를 처리하는 능력을 추가로 개발하였습니다. 이에 따라 MLLMs의 성능을 평가하기 위한 새로운 기준과 방법들이 요구되고 있음을 설명합니다.

- **Technical Details**: MLLMs는 모달리티 인코더(modality encoder), LLM, 그리고 이들을 연결하는 커넥터(connector)로 구성되어 있습니다. 비전-언어 모델의 예를 들면, 텍스트 쿼리와 비전 샘플을 입력으로 받아 비전 인코더가 특징(feature)을 추출하고, 커넥터는 비전 특징과 텍스트 임베딩을 정렬합니다. 이후 이 정렬된 비전 특징은 사용자 쿼리의 텍스트 임베딩과 결합되어 LLM에 의해 자연어 응답으로 생성됩니다.

- **Performance Highlights**: MLLM 평가의 예상 방향으로는 기능별 분류, 기능 중심 평가, 태스크 지향 평가, 그리고 더욱 다양한 모달리티의 포함 등이 있습니다. 본 논문은 다양한 평가 기준을 계층적으로 분류하고, 평가 제작 과정에서 필요한 주의 사항을 정리하였으며, 성능 측정 방법으로 인간 기반, LLM 기반, 그리고 스크립트 기반의 세 가지 주요 방식을 제시합니다. 이로써 연구자들이 적절한 평가 기준을 쉽게 찾고 효과적인 평가 방안을 탐색할 수 있도록 돕고자 합니다.



### There is no SAMantics! Exploring SAM as a Backbone for Visual Understanding Tasks (https://arxiv.org/abs/2411.15288)
Comments:
          Preprint. Work in progress

- **What's New**: 본 연구에서는 Segment Anything Model(SAM)의 시멘틱(semantic) 이해력 한계를 탐구하고, 기존의 서브모델 활용을 통해 SAM의 유용성을 극대화하는 방법을 모색합니다. 특히, SAM의 기능은 이미지 분할에 초점을 맞추지만 시멘틱 해석이 결여되어 있음을 강조합니다. 본 연구는 SAM의 시멘틱 이해력을 개선하기 위해 외부 시멘틱 자료를 통합하는 접근 방식을 제안합니다.

- **Technical Details**: 연구는 SAM의 시멘틱 능력을 평가하기 위해 CLIP 및 DINOv2와 같은 이미지 인코더들과 비교하였습니다. 초기 실험 결과, SAM은 분류 작업에서 충분한 시멘틱 구별 능력을 갖추고 있지 않아 클래스 구분의 한계를 보였음을 밝힙니다. 그 후 경량화된 파인튜닝을 통해 SAM의 내재적 표현을 수정하려고 하였으나, 새로운 클래스에 대한 일반화 능력은 여전히 제한적이었습니다.

- **Performance Highlights**: 연구의 결론은 DINOv2의 피처를 활용하여 SAM의 시멘틱 이해를 향상시키는 것이 가능하다는 점입니다. 초기 결과는 SAM이 특정 객체 카테고리에 대해 제한된 시멘틱 통찰력을 얻을 수 있으나, 여전히 일반화 능력 부족 문제가 해결되지 않았음을 보여줍니다. 따라서, 복잡한 시각적 과제에 대한 SAM의 성능 향상을 위한 방향성을 제안합니다.



### When Spatial meets Temporal in Action Recognition (https://arxiv.org/abs/2411.15284)
Comments:
          Research report

- **What's New**: 이번 연구에서는 비디오 액션 인식 분야에서의 새로운 접근법인 Temporal Integration and Motion Enhancement (TIME) 레이어를 소개합니다. 기존의 방법들은 주로 공간적 특징이나 시간적 동역학 중 하나에만 집중해왔지만, 이 연구는 두 가지 정보를 효과적으로 통합할 필요성을 강조합니다. TIME 레이어는 원본 비디오 시퀀스를 재배열하여 새로운 비디오 프레임을 생성하게 합니다.

- **Technical Details**: TIME 레이어는 $N^2$개의 시간적으로 진화하는 프레임을 단일 공간 격자($N 	imes N$)에 포함시키는 방식으로 작동합니다. 이때 $N$이 1일 경우, 레이어는 기존 방법처럼 풍부한 공간적 세부 정보를 캡처합니다. 그러나 $N$이 증가함에 따라 ($N\geq2$), 시간적 정보가 더욱 두드러지며, 공간 정보는 모델 입력과의 호환성을 위해 감소합니다.

- **Performance Highlights**: 우리는 TIME 레이어를 ResNet-50, Vision Transformer, Video Masked Autoencoders와 같은 인기 있는 액션 인식 모델에 통합하여 그 효과성을 입증했습니다. 실험 결과 TIME 레이어는 인식 정확도를 향상시키며, 비디오 처리 작업에 대한 귀중한 통찰력을 제공합니다.



### Foundation Cures Personalization: Recovering Facial Personalized Models' Prompt Consistency (https://arxiv.org/abs/2411.15277)
- **What's New**: 이 논문에서는 Facial personalization에 대한 새로운 접근법인 FreeCure를 제안합니다. 기존의 방법들은 identity embedding을 사용하여 사용자 정의 프롬프트와 얼굴 정보를 연결하는 방식이었으나, 이로 인해 프롬프트 일관성이 저하되는 문제가 발생했습니다. FreeCure는 이러한 문제를 해결하기 위해, 별도의 훈련 없이 기초 모델에서 직접 지식을 활용하여 프롬프트 일관성을 개선하는 방법을 모색합니다. 이 접근방식은 다양한 얼굴 속성을 효과적으로 강화하며, 기존 모델에 통합하기 용이하다는 장점이 있습니다.

- **Technical Details**: FreeCure는 두 가지 주요 기술을 통해 작동합니다. 첫째, Spatially Aligned Mask Extraction (SAME)을 통해 기초 모델과 개인화 모델 간의 속성을 정렬하여 타겟 속성의 마스크를 생성합니다. 둘째, Restore Once-for-all (ROFA)라는 방법을 통해 기초 모델에서 유익한 속성을 개인화 모델로 전이하고 노이즈 블렌딩 절차를 활용하여 표현을 복원합니다. 이 연구는 훈련 과정 없이도 다수의 속성을 동시에 개선할 수 있는 방법론을 제안합니다.

- **Performance Highlights**: 제안된 FreeCure는 다양한 최신 얼굴 개인화 모델에서 프롬프트 일관성을 유의미하게 향상시키는 성과를 보여주었습니다. 특히, 우리는 개인화 모델에서의 identity embedding이 다른 속성 관련 토큰의 정상적인 표현을 저해함을 발견하고, FreeCure가 이러한 문제를 해결할 수 있음을 증명했습니다. 실험 결과를 통해 프롬프트 일관성을 향상시키면서도 원래의 신원 보존 능력을 유지하는 데 성공했습니다.



### Event USKT : U-State Space Model in Knowledge Transfer for Event Cameras (https://arxiv.org/abs/2411.15276)
- **What's New**: 이 연구는 Event-to-RGB 지식 전이를 위한 맞춤형 U자형 State Space Model Knowledge Transfer (USKT) 프레임워크를 제안하여 이벤트 데이터를 RGB 프레임에 호환 가능한 입력으로 변환합니다. 이를 통해 사전 훈련된 RGB 모델을 효과적으로 재사용하고, 최소한의 매개변수 조정으로 경쟁력 있는 성능을 얻을 수 있습니다. 또한, 새로운 Bidirectional Reverse State Space Model (BiR-SSM)을 도입하여 계산 자원을 절약하면서도 효율적인 모델링을 가능하게 합니다.

- **Technical Details**: USKT 프레임워크는 잔여 다운 샘플링 블록과 잔여 업 샘플링 블록을 포함하며, Bidirectional Reverse State Space Model (BiR-SSM)을 통해 전체적인 기능 의존성을 포착합니다. 이벤트 데이터는 (x,y,t,p) 구조로 시각화되어, 3차원 그리드로 매핑되어 정리됩니다. 이 과정에서 각 격자는 시간 빈 내에 이벤트의 편극성을 집계하여 최종적으로 시공간 정보를 유지한 3차원 텐서를 생성합니다.

- **Performance Highlights**: USKT와 ResNet50을 통합하면 DVS128 Gesture, N-Caltech101, CIFAR-10-DVS 데이터셋에서 각각 0.95%, 3.57%, 2.9%의 성능 향상을 달성했습니다. 이러한 결과는 USKT의 적응성과 효과성을 강조하며, 모델 성능 향상을 위해 리컨스트럭션과 분류 목표를 결합하는 하이브리드 손실 함수를 제공합니다.



### EADReg: Probabilistic Correspondence Generation with Efficient Autoregressive Diffusion Model for Outdoor Point Cloud Registration (https://arxiv.org/abs/2411.15271)
- **What's New**: EADReg라는 새로운 프레임워크를 제안하여 LiDAR 포인트 클라우드의 효율적이고 견고한 등록을 달성하고 있습니다. 이 방법은 자기 회귀적(diffusion models) 방법론을 기반으로 하여, 특히 야외 LiDAR 포인트에서 발생하는 도전적인 문제를 해결합니다. EADReg는 전체 등록 과정을 거친 후, 정제된 포인트 클라우드 쌍을 활용하여 더욱 향상된 성능을 발휘합니다.

- **Technical Details**: EADReg는 coarse-to-fine 등록 파라다임을 따릅니다. 초기 단계에서는 Bi-directional Gaussian Mixture Model (BGMM)을 사용하여 아웃라이어 포인트를 제거하고, 정제된 포인트 클라우드 쌍을 획득합니다. 다음으로, 자기 회귀 프로세스로서의 diffusion-based PCR을 처리하여 강력한 포인트 대응관계를 생성하고 이를 반복적으로 정제합니다.

- **Performance Highlights**: KITTI와 NuScenes 벤치마크 데이터셋에서 EADReg는 최신 기술 수준의 성능을 입증했습니다. 기존의 convolutional 기반 방법론과 비슷한 실행 시간을 기록하며, diffusion 기반 방법이 가지는 약점을 극복하고 있습니다. 이로 인해 실제 응용에서도 유용성을 더할 것으로 기대됩니다.



### ICT: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models (https://arxiv.org/abs/2411.15268)
- **What's New**: 이 논문은 LVLMs (Large Vision Language Models)의 고질적인 허위 현상(hallucination)을 줄이기 위한 새로운 접근 방식을 제안합니다. ICT (Image-Object Cross-Level Trusted Intervention)라는 경량의 훈련이 필요 없는 방법을 소개하며, 이 방법은 모델이 시각적 정보에 대한 집중을 높이는 데 도움을 줍니다. 기존의 방법들과 달리, ICT는 언어 프라이어를 제거하지 않고도 모델의 주의를 조정하면서 이 문제를 해결합니다.

- **Technical Details**: ICT는 모델의 주의를 조정하는 개입(intervention) 방향을 계산하는 프로세스를 통하여 동작합니다. 이 과정에서는 모델이 visual information과 fine-grained object details를 모두 활성화할 수 있도록 attention heads의 활성화 패턴을 분석합니다. 이때, attention heads의 활성화 값을 조정하여 모델이 신뢰할 수 있는 정보를 강화하도록 합니다.

- **Performance Highlights**: 실험 결과, ICT는 LLaVA-v1.5와 Qwen-VL 모델에서 각각 POPE 벤치마크에서 평균 6.27%의 성능 향상을 보였으며, MME 벤치마크에서 67.37포인트의 향상을 달성했습니다. 더불어 ICT는 다양한 데이터셋과 모델에 대해 일반화되는 성능을 보이며, 기존의 decoding 전략과 병행하여 사용할 수 있는 장점을 가지고 있습니다.



### Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI (https://arxiv.org/abs/2411.15265)
Comments:
          19 pages, 5 figures

- **What's New**: 새롭게 제안된 방법인 Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG)은 기존의 gradient 기반 방법의 한계를 극복하는 혁신적인 접근 방식입니다. 이 방법은 모델의.gradient에 접근하지 않고도 feature attribution과 counterfactual explanation을 통합할 수 있는 단일 프레임워크를 제공합니다. 또한, FreeMCG는 diffusion models와 ensemble Kalman filters를 활용하여 안전하게 모델 출력을 기반으로 explanation을 수행합니다.

- **Technical Details**: FreeMCG는 derivative-free 방법으로서, 기존의 gradient method가 요구하는 모델의 내부 파라미터에 접근할 필요가 없습니다. 대신, 모델의 출력을 통해 데이터 manifold에 투영된 gradient를 근사하는 데 필요한 perturbations을 생성합니다. 이를 통해, gradient가 데이터 manifold와 잘 일치하도록 하여, 더 신뢰성 있는 설명을 제공합니다.

- **Performance Highlights**: FreeMCG는 counterfactual generation과 feature attribution에서 모두 뛰어난 성능을 보여줍니다. 이는 기존 XAI 도구의 필수적인 성질을 유지하면서도, 상태-of-the-art 결과를 달성하는 것으로 나타났습니다. 다양한 평가를 통해 FreeMCG의 유효성을 검증한 결과, 전통적인 방법들보다 월등한 성과를 보였습니다.



### AI-Driven Real-Time Monitoring of Ground-Nesting Birds: A Case Study on Curlew Detection Using YOLOv10 (https://arxiv.org/abs/2411.15263)
- **What's New**: 이 연구에서는 야생 동물 모니터링에 AI(인공지능)를 활용한 새로운 접근 방식을 소개합니다. 특히, 멸종 위기 속에 있는 땅에 둥지를 트는 조류인 curlew(누르기)의 실시간 종 탐지를 목표로 하고 있습니다. 기존의 수작업 데이터 처리 방식에서 벗어나, 효율적인 모니터링을 가능하게 하는 시스템을 개발했습니다.

- **Technical Details**: 연구팀은 YOLOv10(You Only Look Once Version 10) 모델을 사용자 맞춤형으로 훈련시켜, 3/4G 연결이 가능한 카메라를 통해 수집된 데이터를 실시간으로 처리합니다. 해당 시스템은 Conservation AI 플랫폼과 연계되어 운영되며, 총 11개 둥지 위치에서 curlew와 그 새끼를 탐지하고 분류합니다. 높은 성능을 보이는 이 모델은 각각 90.56%의 민감도와 100%의 특이도, 95.05%의 F1-score를 기록했습니다.

- **Performance Highlights**: 실험 결과, curlew 탐지에서 90.56%의 높은 민감도와 100%의 특이도를 달성했으며, 새끼 curlew에 대해서도 92.35%의 민감도에 100%의 특이도를 기록했습니다. 이러한 성과는 AI 기반 모니터링 시스템이 생물 다양성 평가를 위한 정확하고 신속한 데이터를 제공할 수 있음을 보여줍니다. 마지막으로, 이 연구는 기술의 발전이 생태 연구에 어떻게 기여할 수 있는지를 잘 설명하고 있습니다.



### MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation (https://arxiv.org/abs/2411.15262)
Comments:
          The project website is at: this https URL. Code: this https URL

- **What's New**: 최근 비디오 생성 모델인 Stable Video Diffusion의 발전이 눈에 띄고 있지만, 주로 짧은 단일 장면 비디오에 국한되고 있습니다. 이 논문에서는 이러한 문제를 해결하기 위해 MovieBench라는 계층적 영화 수준 데이터셋을 소개합니다. MovieBench는 풍부한 스토리라인과 다중 장면 내러티브를 갖춘 영화 길이 비디오를 특징으로 하며, 캐릭터의 일관성과 오디오를 보장합니다.

- **Technical Details**: MovieBench는 세 가지 계층적인 주석 수준을 제공하며, 영화 수준은 스크립트 요약 및 캐릭터 뱅크를 포함합니다. 캐릭터 뱅크는 각 캐릭터의 이름, 초상 이미지, 오디오 샘플을 포함하여 멀티 장면에서의 일관성을 지원합니다. 또한 장면 수준 주석은 이야기의 진행을 상세히 설명하고, 샷 수준 주석은 특정 순간을 캡처하여 캐릭터와 스토리의 정확한 정렬을 보장합니다.

- **Performance Highlights**: MovieBench는 91919191개의 영화로 구성되며, 평균 영화 지속 시간은 약 45분입니다. 이를 통해 장기 비디오 생성 연구를 촉진하고, 캐릭터 ID 일관성 유지와 다중 장면에서의 비디오 생성 동기화를 통해 새로운 통찰력과 도전을 제공합니다. 이 데이터셋은 공개되며 지속적으로 유지 관리되어 장기 비디오 생성 분야의 발전을 목표로 하고 있습니다.



### VIVID-10M: A Dataset and Baseline for Versatile and Interactive Video Local Editing (https://arxiv.org/abs/2411.15260)
Comments:
          17 pages, 14 figures

- **What's New**: 본 논문은 비디오 편집의 질적 향상을 위해 VIVID-10M이라는 데이터셋과 VIVID라는 모델을 소개합니다. VIVID-10M은 처음으로 대규모 하이브리드 이미지-비디오 로컬 편집 데이터셋으로, 9.7M 샘플을 포함하고 있어 다양한 비디오 편집 작업을 다룹니다. 이는 데이터 구축과 모델 훈련 비용을 줄이는 데 도움을 주며, 비디오 편집의 상호작용성을 높이기 위한 기초가 됩니다.

- **Technical Details**: VIVID는 VIVID-10M에서 훈련된 다목적 비디오 로컬 편집 모델로, 엔티티 추가, 수정 및 삭제를 지원합니다. 이 모델은 키프레임(关键帧) 기반의 상호작용 비디오 편집 메커니즘을 제안하여 사용자가 반복적으로 키프레임을 편집하고 이를 다른 프레임으로 전파할 수 있도록 합니다. 이러한 접근 방식은 원하는 결과를 얻기 위한 지연 시간을 줄이는 데 기여합니다.

- **Performance Highlights**: 광범위한 실험 평가 결과, VIVID 접근법이 비디오 로컬 편집에서 최첨단 성능을 달성했으며, 자동화된 메트릭과 사용자 연구에서 기존 방법을 초월하는 성과를 보였습니다. 이는 VIVID-10M 데이터셋과 VIVID 편집 모델을 통해 비디오 편집의 가능성을 한층 확장할 것으로 기대됩니다.



### LocRef-Diffusion:Tuning-Free Layout and Appearance-Guided Generation (https://arxiv.org/abs/2411.15252)
- **What's New**: 이 논문에서는 LocRef-Diffusion이라는 새로운 모델을 제안합니다. 이 모델은 개인화된 사용자 정의가 가능한 튜닝 프리(tuning-free) 방식으로, 이미지 내 여러 인스턴스의 외관과 위치를 조정할 수 있습니다. 또한 Layout-net과 Appearance-net이라는 두 가지 핵심 구성 요소를 통해 인스턴스 배치의 정밀성을 높이고, 참조 이미지와의 외관 충실도를 개선합니다.

- **Technical Details**: LocRef-Diffusion은 이미지 생성 과정에서 명시적인 레이아웃 정보와 인스턴스 영역 크로스 어텐션 모듈을 이용하여 객체의 위치를 제어하며, Appearance-Net을 통해 참조 이미지의 외관 특징을 추출합니다. 이러한 기능은 크로스 어텐션 메커니즘을 통해 확장되어 생성된 객체의 참조 이미지에 대한 유사성을 유지하게 합니다. 이를 통해 모델은 고해상도 이미지 생성에서 Layout과 Appearance에 대한 정밀한 제어를 가능하게 합니다.

- **Performance Highlights**: COCO와 OpenImages 데이터셋에서 시행된 실험 결과, LocRef-Diffusion은 레이아웃 및 외관에 기반한 생성에서 최신 기술 대비 뛰어난 성능을 보였습니다. 이 모델은 타겟 인스턴스의 정확한 위치 지정을 지원하며, 기본 모델의 사전 학습된 가중치를 보존하면서 새로운 프로젝션 모듈과 지역화 계층을 통합합니다. 실험에서 나타난 제로샷(zero-shot) 성능은 기존 모델들에 비해 현저히 개선된 결과를 보여줍니다.



### AnyText2: Visual Text Generation and Editing With Customizable Attributes (https://arxiv.org/abs/2411.15245)
- **What's New**: AnyText2는 자연 풍경 이미지 생성 및 편집에서 다국어 텍스트 속성을 정확하게 제어할 수 있는 새로운 방법을 제안합니다. 이 방법은 텍스트 렌더링 기능을 갖춘 WriteNet+AttnX 아키텍처를 포함하며, 텍스트 속성을 개별 조건으로 인코딩하는 Text Embedding Module을 활용합니다.

- **Technical Details**: AnyText2는 pretrained T2I 모델을 위한 플러그인으로 설계되었습니다. 이 프레임워크는 Latent Diffusion Model(LDM)의 구조를 기반으로 하며, 입력 이미지를 조건 임베딩으로 인코딩한 후, 이를 기반으로 노이즈를 예측하여 최종 이미지를 생성합니다. 텍스트 렌더링 능력을 개선하기 위해 Auxiliary Latent Module이 포함되어 있어 글리프, 위치 등을 인코딩합니다.

- **Performance Highlights**: AnyText2는 기존 AnyText 방법에 비해 이미지 현실감을 개선하며, 추론 속도가 19.8% 증가했습니다. 실험 결과, 영어와 중국어 텍스트 정확도는 각각 9.3%와 3.3% 개선된 것으로 나타났으며, 다양한 텍스트 속성을 제어할 수 있는 능력으로 향상된 성능을 입증했습니다.



### Adversarial Prompt Distillation for Vision-Language Models (https://arxiv.org/abs/2411.15244)
- **What's New**: 이 논문에서는 Adversarial Prompt Distillation (APD)라는 새로운 방어 방법을 제안하여, 기존의 Vision-Language Models (VLM) 모델의 적대적 강인성을 향상시킨다. APD는 Adversarial Prompt Tuning (APT)과 knowledge distillation을 결합하여 CLIP 모델의 성능을 높인다. 특히, APD는 시각 모달과 텍스트 모달 모두에 대해 프롬프트를 추가하여 양 모달 방식(bimodal)으로 작동한다.

- **Technical Details**: APD는 비강인한(non-robust) CLIP 모델을 학생(student) 모델로, 깨끗이 훈련된 선생님(teacher) CLIP 모델을 사용하여 지식을 증류하는 방식으로 동작한다. 이 방법은 시각 모달과 텍스트 모달 모두에서 학습 가능한 프롬프트를 삽입하여 적대적 강인성을 강화한다. 실험을 통해 APD는 기존의 APT 방법들을 초월하는 성능을 보여주었다.

- **Performance Highlights**: APD는 8개의 벤치마크 데이터셋에서 강력한 성능을 발휘하며, 기존의 APT 방법들과 다양한 변형들에 비해 PGD 및 AutoAttack(AA) 공격에 대해 뛰어난 방어 성능을 나타냈다. APD는 비강인한 모델을 사용하여도 질적 정확성과 강인성을 동시에 높이는 가능성을 보여주었고, 이를 통해 VLM의 강인성을 개선하는 데 기여할 수 있음을 입증했다.



### EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality (https://arxiv.org/abs/2411.15241)
Comments:
          preprint

- **What's New**: 이 논문에서는 기존의 지식 기반 비전 아키텍처에서의 비효율성을 개선하기 위해 새로운 경량 비전 구조인 Efficient Vision Mamba (EfficientViM)를 소개합니다. EfficientViM은 효율적인 전역 의존성 캡처를 위해 Hidden State Mixer-based State Space Duality (HSM-SSD) 레이어를 기반으로 구축되었습니다. 이 아키텍처는 기존의 SSD 레이어의 채널 혼합 작업을 숨겨진 상태 공간으로 전이하여 계산 비용을 줄이고 성능을 향상시킵니다.

- **Technical Details**: HSM-SSD 레이어는 이미지 특성 공간에서의 선형 투영과 게이팅 기능을 숨겨진 상태 공간으로 전이하여 데이터의 잠재적 표현을 보다 효율적으로 활용할 수 있게 합니다. 또한, 다단계 숨겨진 상태 융합(multi-stage hidden state fusion) 기법을 도입하여 각 단계에서 숨겨진 상태의 성능을 강화하고 메모리 바운드 작업에 의한 병목 현상을 완화합니다. 이 구조적 개선은 메모리 바운드 작업을 최소화하여 실제 애플리케이션에서의 성능을 극대화합니다.

- **Performance Highlights**: EfficientViM은 ImageNet-1k 벤치마크에서 기존 최 SOTA 모델인 SHViT보다 0.6% 향상된 성능을 보이며, 각각 7%와 80% 빠른 속도를 자랑합니다. 실험 결과, EfficientViM은 이전 연구들에 비해 스루풋과 정확도가 크게 향상되었음을 입증했습니다. 이러한 성능 개선은 모델의 실제 사용 가능성을 극대화하며, 특히 resource-constrained 환경에서의 최적화를 위한 중요한 결과로 평가될 수 있습니다.



### Faithful Label-free Knowledge Distillation (https://arxiv.org/abs/2411.15239)
- **What's New**: 이 논문은 Teacher in the Middle (TinTeM)이라는 새로운 label-free knowledge distillation 접근 방식을 제안합니다. 이 방법은 teacher 네트워크의 latent space로부터 student 네트워크로 약 orthogonal mapping을 학습하여, teacher의 행동을 보다 충실하게 재현할 수 있도록 돕습니다. 특히, TinTeM은 모델의 robustness, generalisability 및 OOD(out-of-distribution) detection 능력을 향상시키는 데 기여합니다.

- **Technical Details**: TinTeM은 teacher와 student 네트워크 사이의 전이 과정에서 거리를 보존하는 explicit mapping을 학습합니다. 이 접근 방식은 조건부 확률 분포를 사용하여 학생 네트워크의 출력을 teacher 네트워크의 latent space로 통합합니다. 논문에서는 최소 제곱 오차(loss)를 직접 계산하는 효율적인 방법을 제안하며, 이러한 mapping은 특정한 오차 한계 내에서 구축될 수 있음을 보여줍니다.

- **Performance Highlights**: TinTeM을 사용한 knowledge distillation은 기존의 Proteus 접근 방식보다 OOD detection 성능을 크게 개선하였으며, 다양한 benchmark에서 경쟁력 있는 성능을 달성하였습니다. 또한, TinTeM은 특정 비전 태스크에 대한 정확도를 높이면서도, foundation model의 일반화 및 OOD detection 성능을 유지하는 specialized model을 훈련할 수 있는 가능성을 보여줍니다.



### Stain-Invariant Representation for Tissue Classification in Histology Images (https://arxiv.org/abs/2411.15237)
- **What's New**: 이번 논문에서는 다수의 요인들이 전체 슬라이드 이미지(Whole Slide Image, WSI)의 최종 외관에 영향을 미치는 병리학적 슬라이드의 디지털화 과정에 대해 다룹니다. 특히, 염색 프로토콜, 스캐너, 조직 유형의 다양성이 심층 학습(Deep Learning, DL) 알고리즘의 다중 집단 설정에서 훈련 및 테스트 시 문제를 일으킬 수 있음을 강조합니다. 이를 해결하기 위해 우리는 염색 매트릭스 교란(stain matrix perturbation)을 활용하여 훈련 이미지의 염색 보강 버전을 생성하는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법에서는 염색 정규화 손실(stain regularisation loss)을 적용하여 원본 이미지와 보강 이미지 간의 특징 표현(feature representations) 간 일관성을 확보합니다. 이렇게 함으로써 모델은 염색 불변(stain-invariant) 및 도메인 불변(domain-invariant) 특징 표현을 학습하도록 유도됩니다. 논문에서는 이 모델을 사용하여 대장암 이미지의 교차 도메인 멀티 클래스 조직 유형 분류에서 평가하였으며, 다른 최첨단 방법들에 비해 개선된 성능을 보여주었습니다.

- **Performance Highlights**: 제안된 모델은 대장암의 다양한 조직 유형을 분류하는 데 있어 우수한 성능을 나타냅니다. 이를 통해 어려운 도메인 전이 문제를 극복하고, 기존의 방법에 비해 모델의 일반화 가능성을 높이는 데 성공했습니다. 이 연구는 계산 병리학(Computational Pathology, CPath) 분야에서 강건하고 일반화 가능한 DL 모델 개발의 필요성을 잘 보여줍니다.



### Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps (https://arxiv.org/abs/2411.15236)
- **What's New**: 이번 연구에서는 텍스트-이미지 확산 모델에서 발생하는 교차 주의 맵의 문제를 다룹니다. 특히, 문장의 구문적 관계가 텍스트 인코더의 주의 맵에서 잘 반영되지 않아 이미지 생성의 정확성에 영향을 미친다는 점을 강조합니다. 이를 해결하기 위해, 본 연구는 문장의 주의 맵에서 구문적 관계를 직접 교차 주의 모듈로 전송하는 새로운 방법을 제안합니다.

- **Technical Details**: 연구의 핵심은 텍스트 임베딩 (text embedding) 간의 유사성이 교차 주의 맵 (cross-attention map)에 미치는 영향을 분석하는 것입니다. 문장 내 구문적 관계를 반영하기 위해, 제안된 방법은 매우 자주 사용되지 않던 정보를 활용하여, 교차 주의 맵의 공간 정렬을 최적화하는 방식을 취합니다. 이러한 접근 방식은 외부 입력 없이도 이미지를 생성할 수 있게 합니다.

- **Performance Highlights**: 이 연구의 제안된 방법은 교차 주의 모듈이 문맥적 관계를 더 잘 캡처하도록 유도하는 한편, 다양한 텍스트 프롬프트에 대해 이미지가 의도한 의미를 더 정확하게 반영하도록 합니다. 따라서, 기존의 외부 정보에 의존하지 않고, 텍스트-이미지 모델 내에 내재된 정보를 활용함으로써 성능 향상을 꾀할 수 있습니다.



### BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models (https://arxiv.org/abs/2411.15232)
Comments:
          18 pages, 5 figures, 10 tables

- **What's New**: 이 논문은 BiomedCoOp이라는 새로운 프롬프트 학습 프레임워크를 제안하여 BiomedCLIP을 효율적으로 적응시키고, 적은 수의 데이터로 생물 의학 이미지를 분류하는 정확성을 높이는 방법을 소개합니다. 기존의 프롬프트 학습 기법들은 일반화에 한계가 있었지만, BiomedCoOp은 대형 언어 모델(LLMs)을 활용하여 컨텍스트 학습을 효과적으로 수행합니다. 또한, 통계 기반의 프롬프트 선택 전략을 통해 제거된 프롬프트로 인한 문제를 해결합니다.

- **Technical Details**: BiomedCoOp는 CoOp 방식을 기반으로 하여, LLM에서 유도된 프롬프트 앙상블을 통해 의미론적 일관성을 강화합니다. 이 시스템은 대형 생물 의학 데이터셋을 포함한 11개 의료 데이터셋에서 검증되어, 다양한 모달리티와 신체 기관에 걸쳐 성능 개선을 보여주었습니다. 특히, 우리는 일반 지식 CLIP 모델과 비교하여 BiomedCLIP을 활용함으로써 다양한 임상 작업에서 이점을 입증했습니다.

- **Performance Highlights**: 이 연구는 BiomedCoOp의 성능이 기존의 CLIP 프롬프트 학습 기술과 비교하여, 다양한 의료 조건 및 이미징 모달리티에서 우수한 일반화 능력과 강건성을 발휘함을 나타냅니다. 실험 결과는 BiomedCoOp가 적은 샷 학습 구조에서 높은 정확도를 달성했음을 보여줍니다. 이로 인해, 생물 의학 이미지 분석의 효율적인 프롬프트 학습 방법으로서의 가능성을 제시합니다.



### Image Harmonization using Robust Restricted CDF Matching (https://arxiv.org/abs/2411.15213)
Comments:
          submitted to 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이 논문은 기계 학습 알고리즘이 실제 환경에 적용될 때의 도전 과제에 대해 논의하며, Cumulative Distribution Function (CDF) 매칭을 기반으로 한 이미지 하모니제이션 방법을 제안합니다. 이 접근법은 로컬 변동성과 개별적으로 중요한 특징들을 유지하면서 입력 데이터의 변동성을 줄이는 데 중점을 둡니다. 기존의 통계적 방법이나 기계 학습 기반의 방법과 달리, 우리의 방법은 수학적 모델링 기법을 활용하여 더 나은 직관성을 제공합니다.

- **Technical Details**: 제안된 방법에서는 이미지의 CDF를 사전 정의된 템플릿 CDF에 맞추기 위해 곡선 피팅 최적화 문제로 해결합니다. 이 과정에서 분포의 긴 꼬리를 줄이는 후처리(Poost-Processing)를 포함하며, 즉각적인 변화를 피하고 매끄러움을 유지하는 'dual-scaling' 변환을 종류로 포함합니다. 또한, 템플릿 CDF는 이미지 분포에 반드시 일치할 필요가 없으며, 이는 저희 방법의 유연성을 나타냅니다.

- **Performance Highlights**: 이 연구에서 제안하는 방법은 MRI 데이터를 중심으로 하여 개별 입력 이미지 사이의 로컬 변동성을 유지해야 하는 동시에, 다양한 임상 환경에서도 강인함을 보여줍니다. 통계적 기술, 강도 정규화, 기계 학습 기법을 결합하여 출력 결과에 대한 변동성을 최소화하는 동시에 성능을 향상시키고자 합니다. 최종적으로, 제안된 방법은 실험 환경을 넘어 실제 문제 해결에도 즉시 활용 가능하다는 점에서 큰 장점이 있습니다.



### Uni-Mlip: Unified Self-supervision for Medical Vision Language Pre-training (https://arxiv.org/abs/2411.15207)
Comments:
          15 pages, 2 figures, accepted by BMVC'24

- **What's New**: 최근 비전-언어 사전 학습(Vision-and-Language Pre-training) 기술의 발전이 컴퓨터 비전 분야의 성능을 크게 향상시켰습니다. 그러나 의료 분야에서는 다중 모달 데이터를 얻는 것이 비용이 많이 들고 복잡하며, 개인정보 보호 등 여러 어려움이 있습니다. 이를 해결하기 위해 Uni-Mlip라는 새로운 프레임워크를 소개하며, 이 프레임워크는 의료 비전-언어 사전 학습을 위한 통합 자기 지도화(self-supervision) 접근 방식을 제공합니다.

- **Technical Details**: Uni-Mlip은 데이터 수준과 특징 수준 모두에서 교차 모달(cross-modality), 단일 모달(uni-modality), 융합 모달(fused-modality) 자기 지도화 기법을 통합하여 의료 이미지를 효과적으로 처리합니다. 특히, 의료 이미지의 특성에 맞춘 단일 모달 이미지 자기 지도화 기법을 조정하여 높은 정밀도와 자세한 감도를 제공합니다. 이러한 접근 방법을 통해 의료 데이터를 더 효과적으로 발굴하고, 모델의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 다양한 규모의 데이터세트에 대한 실험 결과, Uni-Mlip는 이미지-텍스트 검색(image-text retrieval), 이미지 분류(image classification), 시각적 질의 응답(Visual Question Answering, VQA)과 같은 주요 다운스트림 작업에서 현재의 최첨단 방법들을 능가하는 뛰어난 성능을 보여줍니다. 이 성능 향상은 모델의 특징과 데이터 간의 정렬 및 전이 가능성을 크게 개선하는 데 기여하였습니다.



### DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh (https://arxiv.org/abs/2411.15205)
- **What's New**: 본 논문에서는 DAGSM(Disentangled Avatar Generation and Simulation Model)이라는 새로운 파이프라인을 제안하며, 이는 사용자가 제공하는 텍스트 프롬프트로부터 분리된 인간 본체와 의류를 생성하는 시스템이다. 기존 방법이 단일 3D 모델로 모든 의류를 포함시키는 데 반해, DAGSM은 각 의류 및 본체를 개별 모델로 생성하여 사용자 제어 및 의류 교체를 용이하게 한다.

- **Technical Details**: DAGSM은 GS-강화 메쉬(GS-enhanced mesh, GSM)로 각 의류 및 본체 부분을 모델링하여 복잡한 텍스처에 효과적으로 대처할 수 있다. 이 과정에서 세 가지 중요한 디자인을 채택하여 본체 생성 후 의류를 순차적으로 생성하며, 2D 가우시안(2D Gaussian)을 활용하여 물리적 시뮬레이션을 통해 더욱 현실적인 애니메이션을 제공한다.

- **Performance Highlights**: 실험 결과, DAGSM은 고품질의 분리된 아바타를 생성하고 의류 교체 및 사실적인 애니메이션을 지원한다. 기존 방법들과 비교했을 때, 시각적 품질이 뛰어나며 사용자 경험을 개선하는데 큰 기여를 한다. 또한, 특정 이미지를 참고하여 외관을 정밀하게 제어할 수 있는 기능도 강조된다.



### Beyond Visual Understanding: Introducing PARROT-360V for Vision Language Model Benchmarking (https://arxiv.org/abs/2411.15201)
Comments:
          7 pages, 4 figures, Accepted at COLING 2025

- **What's New**: 새로운 PARROT-360V 벤치마크는 복잡한 시각적 퍼즐을 통해 비전 언어 모델(VLM)의 능력을 평가하는 혁신적인 방법입니다. 2487개의 도전적 퍼즐로 구성되어 있으며, 이 모델들이 언어 능력과 시각적 단서를 결합하여 문제를 해결하는 데 있어서 인간의 문제 해결 방식과 유사한지 평가하고자 합니다. 또한, 최신 VLM 모델들인 GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro의 성능을 비교하여 복잡한 추론 작업에서의 한계를 드러냅니다.

- **Technical Details**: PARROT-360V는 비전 언어 모델이 이미지와 텍스트 데이터를 통합하는 능력을 평가하기 위해  복잡한 시각적 추론을 요구하는 퍼즐을 중심으로 설계되었습니다. 기존의 벤치마크들이 단순한 이미지-텍스트 정렬이나 단일 단계 추론에 중점을 두었다면, 이 벤치마크는 언어 이해뿐만 아니라 시각적 인식과 추론 능력에 대한 실질적인 평가를 목표로 하고 있습니다. 데이터 전처리나 주석의 변동성으로 인한 재현성 문제를 해결하기 위해, 모델이 시각적 단서를 통합하여 단계별로 문제를 해결해야 합니다.

- **Performance Highlights**: VLM 모델들은 PARROT-360V 벤치마크에서 28%에서 56% 사이의 점수로 확인되었으며, 이는 기존 인기 있는 벤치마크들과 비교할 때 현저히 낮은 성과입니다. 이 결과는 현재 VLM들이 복잡한 다단계 추론 작업을 수행하는 데 있어 뚜렷한 한계를 지니고 있음을 강조합니다. 따라서 PARROT-360V는 비전 언어 모델의 실제 성능을 측정하고, 그 모델들이 진정한 문제 해결 능력을 보유하고 있는지 평가하는 데 중요한 역할을 할 것입니다.



### Deep Learning-Based Classification of Hyperkinetic Movement Disorders in Children (https://arxiv.org/abs/2411.15200)
Comments:
          59 pages, 20 figures

- **What's New**: 이번 연구에서는 소아의 과다운동장애(Hyperkinetic Movement Disorders, HMDs) 진단을 위해 비디오 녹화 데이터를 바탕으로 한 딥러닝 모델을 개발했습니다. 이 모델은 비디오에서 아동이 운동 과제를 수행하는 모습을 분석하여 지각적 이상 움직임을 감지하고, 특히 이상 운동의 유형인 디스토니아(dystonia)와 코레아(chorea)를 구별하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 모델은 그래프 합성곱 네트워크(Graph Convolutional Network, GCN)와 장단기 메모리(Long Short-Term Memory, LSTM) 네트워크를 결합하여 공간적 및 시간적 특징을 모두 활용합니다. GCN은 인체의 관절 연결을 그래프로 표현하여 이들의 공간적 관계를 학습하고, LSTM은 시간적 의존성을 모델링하여 움직임의 연속성을 처리합니다. 이러한 모델은 전문가의 공정한 해석을 도와주는 주의(attention) 메커니즘도 통합되었습니다.

- **Performance Highlights**: 모델은 50개의 비디오를 훈련 및 검증하여 15fps에서 85%의 정확도, 81%의 민감도 및 88%의 특이성을 달성했습니다. 주의 맵(attention map)을 통해 모델이 비자발적인 운동 패턴을 올바르게 식별하는 능력을 확인하였으며, 분류 오류는 주로 신체 부위가 가려지거나 미세한 움직임 변동 때문으로 나타났습니다. 이 연구는 HMD 진단의 정확성과 효율성을 향상시킬 수 있는 딥러닝의 가능성을 보여줍니다.



### Adaptively Controllable Diffusion Model for Efficient Conditional Image Generation (https://arxiv.org/abs/2411.15199)
- **What's New**: 본 논문에서는 새로운 적응형 제어 확산 모델인 Adaptively Controllable Diffusion (AC-Diff) 모델이 제안되었습니다. 이 모델은 생성 과정, 결과의 형태, 길이 및 매개변수 모두를 자동으로 제어할 수 있습니다. 기존의 확산 모델들이 가지고 있는 제어 문제를 해결하고자 하며, 고정된 매개변수 대신 동적 재조정을 통해 필요에 맞는 생성이 가능하도록 설계되었습니다.

- **Technical Details**: AC-Diff 모델은 Conditional Time-Step (CTS) 모듈을 통해 필요한 생성 단계 수를 결정하며, Adaptive Hybrid Noise Schedule (AHNS) 모듈을 사용하여 변동적인 확산 속도 매개변수를 생성합니다. 이 과정에서 입력 조건에 따라 모델이 스스로 조정할 수 있도록 적응형 샘플링 메커니즘이 통합되어 훈련됩니다. 이를 통해 더 향상된 성능을 목표로 합니다.

- **Performance Highlights**: AC-Diff 모델은 전체 평균 생성 단계 수와 실행 시간을 크게 줄이는 동시에 기존 문헌의 확산 모델과 유사한 성능을 유지할 것으로 기대됩니다. 이 모델은 효율성이 강조된 생성 모델로, 간단한 내용부터 더 복잡한 내용까지의 양질의 생성이 가능하도록 도와줍니다.



### Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting (https://arxiv.org/abs/2411.15193)
- **What's New**: 본 논문은 Gaussian Splatting에서 feature field rendering을 위한 학습 없는(training-free) 방법을 소개합니다. 제안된 방법은 2D features를 사전 훈련된 3D Gaussians로 역투영하여, 각각의 Gaussian이 최종 렌더링에서 미치는 영향에 따라 가중합을 사용하는 구성을 가지고 있습니다. 기존의 학습 기반 접근 방법들이 2D 분할(segmentation)에서는 뛰어난 성능을 발휘하지만, 3D 분할에서는 후처리가 필요하도록 저조한 결과를 내는 것과 달리, 제안된 방법은 2D와 3D 분할 모두에서 높은 품질의 결과를 달성합니다.

- **Technical Details**: 3D Gaussian Splatting (3DGS)은 3D 장면을 렌더링하기 위해 3D Gaussians를 사용하는 기법으로, 각 Gaussian은 평균 위치(mean position)와 분산(variance)으로 정의됩니다. 우리 연구에서는 3D 공간에서 직접 2D features를 3D Gaussians에 투사하는 방식을 이용하여, 전통적인 feature 필드 증류(feature field distillation)와 비교할 때 빠르고 확장 가능한 대안을 제공합니다. 또한, 본 논문에서는 feature 백프로젝션(back-projection) 방정식을 제시하고, 이를 기반으로 3D 분할, 합에서는전이(affordance transfer), 그리고 마지막으로 정체성 인코딩(identity encoding)과 같은 네 가지 직접 사용 사례를 설명합니다.

- **Performance Highlights**: 실험 결과에 따르면, 본 연구의 접근법은 빠르며 확장 가능하고, 대규모 학습 기반 방법과 비교할 수 있는 성능을 제공합니다. 특히, 3D 개체 조작 및 실시간 장면 이해와 같은 다운스트림 애플리케이션을 위한 원활한 쿼리 지원이 가능하다는 점에서 강력한 장점을 보입니다. 또한, 기존의 다른 접근법이 3D 분할에서 어려움을 겪는 것에 반해, 본 방법은 기울기 정보(gradient information)를 직접 활용하여 보다 효과적이고 정확한 3D 분할을 가능하게 합니다.



### LegoPET: Hierarchical Feature Guided Conditional Diffusion for PET Image Reconstruction (https://arxiv.org/abs/2411.16629)
Comments:
          5 pages, 3 figures

- **What's New**: 이번 연구에서는 LegoPET이라는 새로운 모델을 제안하여, sinogram으로부터 높은 인지 품질의 PET 이미지를 재구성하는 접근 방식을 소개하고 있습니다. LegoPET은 계층적 특징에 기반한 조건적 확산 모델로, 기존의 깊은 학습 방법들이 직면해왔던 다양한 문제를 해결하는 데 중점을 두고 있습니다. 특히, 이 모델은 sinogram과 이미지 도메인 간의 일관성을 유지하면서도 빠른 수렴 속도를 제공하는 것이 특징입니다.

- **Technical Details**: LegoPET 모델은 두 개의 주요 구성 요소로 이루어져 있습니다: 시각적 특징 맵을 생성하는 학습된 plug-and-play prior network(PnPNet)와 최종적으로 깨끗한 PET 이미지를 재구성하는 2D sinogram-conditioned diffusion model입니다. PnPNet은 조건부 convolutional U-Net 구조로, sinogram-PET 쌍을 사용하여 훈련되며, 고주파 정보(예: 조직의 경계 및 해부학적 구조)를 캡처할 수 있도록 설계되었습니다. 모든 과정은 마르코프 연쇄(Markov Chain)를 기반으로 하며, 가우시안 분포에서 조건부 데이터 분포로의 변환 과정을 통해 이루어집니다.

- **Performance Highlights**: LegoPET의 실험 결과는 기존의 cDPM들보다 성능이 향상되었고, 최근의 깊은 학습 기반 PET 이미지 재구성 기법과 비교하여 시각적 품질 및 픽셀 단위의 PSNR/SSIM 메트릭에서 뛰어난 성과를 보여주었습니다. 이는 LegoPET이 필요로 하는 속성과 도메인 간의 일관성을 개선하였음을 의미하며, 지각적(Perceptual) 품질에서 두드러진 성능을 자랑합니다. 코드와 관련된 자료는 제공된 링크를 통해 확인할 수 있습니다.



### Unlocking The Potential of Adaptive Attacks on Diffusion-Based Purification (https://arxiv.org/abs/2411.16598)
- **What's New**: Diffusion-based purification (DBP)는 적대적 예제(adversarial examples, AEs)에 대한 방어 메커니즘으로 인기를 얻고 있으나, 본 논문은 DBP의 주장된 안정성을 강한 적대적 공격에 의해 재평가합니다. 연구자들은 gradient-based 전략이 DBP의 근본 원리를 무효화하며, 비효율적인 정화 과정으로 이어진다는 것을 보여주었습니다. 이로 인해 DBP가 현재 상태에서는 AEs에 효과적인 방어 수단이 되지 못함을 밝혔습니다.

- **Technical Details**: DBP의 핵심 아이디어는 diffusion 모델을 이용해 AEs를 자연 분포(natural distribution)로 투영하는 것입니다. 그러나 본 연구에서는 gradient backpropagation을 사용한 적응 공격(adaptive attacks)이 DBP의 복잡한 메커니즘을 타겟으로 하여 불리한 결과를 초래한다는 점을 강조했습니다. 또한, DiffGrad라는 신뢰할 수 있는 gradient 라이브러리를 제공하여 DBP의 성능 감소를 입증했습니다.

- **Performance Highlights**: 적응 공격을 통해 DBP의 안정성이 21% 이하로 떨어진다는 실험 결과를 보여주었습니다. 더욱 엄격한 majority-vote 설정에서 DBP는 일부 저항력을 유지하였지만, 저주파(low-frequency) 최적화 전략을 통해 DBP를 완전히 무너뜨리는데 성공하였습니다. 연구팀은 ImageNet과 CIFAR-10 데이터셋에서 DBP를 제거하는 방법의 효과성을 입증하며, 이전 연구들보다 월등한 성과를 보였습니다.



### Generating Out-Of-Distribution Scenarios Using Language Models (https://arxiv.org/abs/2411.16554)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 장점을 활용하여 다양한 OOD(Out-Of-Distribution) 주행 시나리오를 생성하는 프레임워크를 제안합니다. LLMs의 컨텍스트 추론 능력과 제로샷 일반화 능력을 바탕으로 각 가지가 독특한 OOD 시나리오를 나타내는 분기 트리를 구성하여 OOD 시나리오의 다양성을 향상시켰습니다. 이러한 OOD 시나리오는 CARLA 시뮬레이터를 통해 시뮬레이션되고, 생성된 시나리오의 품질을 평가하기 위한 OOD-ness 및 다양성 메트릭이 도입되었습니다.

- **Technical Details**: 제안된 프레임워크는 GPT-4o 모델과 몇 가지 샷 체인 오브 사우스(Chain-of-Thought, CoT) 프롬프팅을 결합하여 OOD 시나리오를 설명하는 다양한 텍스트를 생성합니다. 이 과정에서 LLM을 사용해 생선된 트리는 언어 모델 기반의 레드 팀을 통해 다양성과 품질을 개선합니다. 생성된 OOD 시나리오는 CARLA 시뮬레이터에서 시뮬레이트되며, 시나리오에 따라 적절한 상태 및 행동을 제안하는 다른 LLM을 통해 자동화된 프로세스를 수행합니다.

- **Performance Highlights**: 성능 평가를 통해 OOD-ness 메트릭은 생성된 시나리오가 전형적인 도시 주행 조건에서 얼마나 벗어나는지를 수량화하고, 다양성 메트릭은 시나리오의 변동성을 측정합니다. 또한, 최신 비전-언어 모델(VLMs)을 대상으로 시뮬레이션된 OOD 시나리오에 대한 신뢰성을 평가하여 자율 주행 시스템에서의 LLM의 유용성을 검증합니다. 이 연구는 자율 주행의 맥락에서 OOD 시나리오를 해결하기 위한 언어 모델의 신뢰성에 대한 귀중한 통찰력을 제공합니다.



### PriorPath: Coarse-To-Fine Approach for Controlled De-Novo Pathology Semantic Masks Generation (https://arxiv.org/abs/2411.16515)
- **What's New**: 이 논문에서는 PriorPath라고 불리는 새로운 파이프라인을 제안합니다. PriorPath는 조직 영역을 나타내는 거친 이미지에서 유래된 정교하고 현실적인 의미적 마스크를 생성하여 단일 플랫폼 내에서 포토리얼리스틱 마스크와 이미지를 모두 얻을 수 있게 합니다. 이는 병리학자들이 생성된 이미지의 전반적인 시각적 특징을 제어할 수 있도록 하며, 기존의 방법보다 더 나은 유사성을 보여줍니다.

- **Technical Details**: PriorPath는 Image-to-Image (I2I) Translation 아키텍처 기반으로, 수작업으로 그린 거친 의미적 마스크를 조건으로 하여 정교한 마스크를 생성합니다. 이 정교한 마스크는 기존의 GANs 방식과 달리 세포 특징의 분포를 직접 통제할 수 있게 해줍니다. 이 접근법은 스케일성과 사실적인 의미 레이블링, 그리고 제어 가능성을 모두 결합합니다.

- **Performance Highlights**: PriorPath는 세 가지 암 유형(피부, 전립선, 폐)에서 효과성을 입증하였고, 기존 방법들과 비교할 때 실제 마스크와의 유사성을 향상시켰습니다. 연구팀은 Frechet inception distance (FID), Kolmogorov–Smirnov (K-S) 테스트 및 Kullback–Leibler (K-L) 발산을 통해 마스크 생성 성능을 정량화하였습니다. 이러한 결과는 AI 기반의 컴퓨터 병리학 발전에 기여할 것입니다.



### Guarding the Gate: ConceptGuard Battles Concept-Level Backdoors in Concept Bottleneck Models (https://arxiv.org/abs/2411.16512)
Comments:
          17pages, 4 figures

- **What's New**: AI 기술의 복잡성이 증가함에 따라 의료 진단과 같은 중요한 분야에서 투명성과 신뢰성에 대한 우려가 커지고 있습니다. 이에 대한 해결책으로 설명 가능한 인공지능(Explainable AI, XAI) 기술이 제시되었으며, 그 중 개념 병목 모델(Concept Bottleneck Models, CBMs)은 높은 수준의 의미적 개념을 사용하여 투명성을 높입니다. 하지만 CBMs는 개념 수준의 백도어 공격에 취약하다는데, 이를 해결하기 위해 새로운 방어 프레임워크인 ConceptGuard를 제안합니다.

- **Technical Details**: ConceptGuard는 개념 군집화(concept clustering) 및 투표 메커니즘(voting mechanism)을 활용해 모델을 보호하는 다단계 접근법을 사용합니다. 이 프레임워크는 텍스트 거리 측정을 기반으로 개념 공간을 의미 있는 하위 그룹으로 분할한 뒤, 각 하위 그룹에 대한 분류기를 훈련시켜 잠재적인 트리거를 분리합니다. 이 과정에서, 모델의 최종 예측에 미치는 영향을 줄이는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 실험 결과, ConceptGuard는 CBMs의 성능과 해석 가능성을 유지하면서 보안성을 크게 향상시켰습니다. 이 시스템은 특정 트리거 크기 기준 내에서 백도어 공격으로부터 효과적으로 방어할 수 있는 이론적 보장을 제공합니다. 또한, XAI 시스템의 보안을 강화하고 신뢰성을 높이는데 기여할 것으로 기대됩니다.



### Comparison of Generative Learning Methods for Turbulence Modeling (https://arxiv.org/abs/2411.16417)
- **What's New**: 본 논문에서는 난류 흐름 시뮬레이션에서 머신러닝 기술의 가능성을 탐구하고 있습니다. 특히 Variational Autoencoders (VAE), Deep Convolutional Generative Adversarial Networks (DCGAN), Denoising Diffusion Probabilistic Models (DDPM)와 같은 생성 모델을 사용하여 2D Kármán vortex street의 유동을 모델링하는 방법을 제안합니다. 이 연구는 난류의 통계적 특성과 공간 구조를 포착할 수 있는 각 모델의 능력을 평가합니다.

- **Technical Details**: 난류 흐름은 높은 비선형성 및 초기 조건에 대한 민감성으로 인해 모델링이 어렵습니다. 기존의 수치 시뮬레이션 기법과는 달리, 본 연구에서는 머신러닝 기법을 통해 이러한 모델링의 복잡성을 해결하려고 시도했습니다. 특히, DDPM과 DCGAN은 유동 분포를 효과적으로 재현하는 것으로 나타났으며, VAE는 결과의 물리적 및 시각적 정확성에서 문제가 있음을 보여주었습니다.

- **Performance Highlights**: 연구 결과에 따르면 DDPM 및 DCGAN 모델은 난류 모델링에 효율적이고 정확한 도구로서의 잠재력을 입증했습니다. 특히 DCGAN은 훈련 시간과 데이터 양이 적게 요구되며, 유동 응답이 입력 스트림과 가장 일치하는 결과를 산출했습니다. 반면, VAE는 빠른 샘플 생성에도 불구하고 불충분한 결과를 제공하는 것으로 평가되었습니다.



### Low-Data Classification of Historical Music Manuscripts: A Few-Shot Learning Approach (https://arxiv.org/abs/2411.16408)
Comments:
          6 pages, The Sixth IEEE international conference on Image Processing Applications and Systems

- **What's New**: 이 논문에서는 역사적 악보의 음악 기호 분류를 위한 자기 지도 학습(framework) 프레임워크를 개발하여 기술과 문화 보존의 교차점을 탐구합니다. 전통적인 Optical Music Recognition (OMR) 방법은 레이블이 있는 데이터 부족으로 어려움을 겪고 있는데, 본 연구는 레이블이 없는 데이터에서 훈련된 신경망 기반(feature extractor) 특징 추출기를 통해 이 문제를 극복하려 합니다. 주요 기여로는 자기 지도 CNN의 크롭 전처리를 최적화하고 SVM, 다층 퍼셉트론(MLP), 프로토타입 네트워크(prototypical networks) 등의 분류 방법을 평가한 점이 있습니다.

- **Technical Details**: 본 연구는 자기 지도 CNN을 기반으로 하여 음악 기호 분류를 위한 특징을 추출하는 방법을 제시합니다. 역사적 악보의 기호들은 크기, 간격, 문서 노후화 등 다양한 변동성을 보이므로, 문서를 세부 섹션(크롭)으로 나누고 이 과정을 슬라이딩 윈도우(sliding window) 방법으로 처리하여 진행합니다. VICReg 방법을 적용하여 CNN은 각 크롭을 두 번 왜곡하여 특징 공간에서 동일한 포인트에 매핑하도록 훈련되며, 이를 통해 역사적 문서에서 발생하는 기호의 가변성에 효과적으로 대처합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제시한 방법은 87.66%의 분류 정확도를 기록하여 AI 기반 방법들이 역사적 음악의 보존 가능성을 확인했습니다. MLP는 다양한 분류 알고리즘 중 가장 우수한 성능을 보였으며, 고차원 데이터와 제한된 라벨 데이터 처리에 있어 강력한 일반화 능력을 나타냅니다. 데이터 증강(data augmentation) 기술을 통해 모델 학습 과정에서 일반화 능력을 더욱 향상시키는 결과를 도출했습니다.



### Privacy-Preserving Federated Foundation Model for Generalist Ultrasound Artificial Intelligenc (https://arxiv.org/abs/2411.16380)
- **What's New**: 본 연구에서는 UltraFedFM이라는 혁신적인 초음파 기초 모델을 제안합니다. 이 모델은 9개 국가의 16개 의료 기관에서 분산 학습(federated learning)을 통해 협력적으로 사전 훈련되었습니다. 100만 개 이상의 초음파 영상을 기반으로 개인정보 보호를 강화하며, 다양한 임상 진단 작업에서 우수한 성능을 발휘합니다.

- **Technical Details**: UltraFedFM은 두 가지 단계로 구성됩니다: 1) 연합 사전 훈련(federated pre-training)과 2) 하위 조정(downstream fine-tuning). 고객들이 각자의 데이터를 공유하지 않고도 모델을 업데이트하며, 특정 임상 작업에 맞춰 모델을 최적화하기 위해 개인맞춤형 데이터로 미세 조정됩니다. 이러한 접근법은 데이터 프라이버시와 일반화 문제를 동시에 해결합니다.

- **Performance Highlights**: UltraFedFM은 질병 진단에서 평균 AUC(Area Under Curve) 0.927을 달성하며, 병변 분할에서는 디이스 유사도 계수(dice similarity coefficient) 0.878을 기록했습니다. 또한, 중급 초음파 전문의보다 더 높은 진단 정확도를 보이며, 경력 10년 이상의 전문가와 비슷한 성능으로 8가지 일반적인 전신 질병 진단에서 우수한 결과를 보여주고 있습니다.



### WTDUN: Wavelet Tree-Structured Sampling and Deep Unfolding Network for Image Compressed Sensing (https://arxiv.org/abs/2411.16336)
Comments:
          20pages,Accepted by ACM Transactions on Multimedia Computing Communications and Applications (TOMM)

- **What's New**: 본 논문에서는 압축 센싱(Compressed Sensing) 분야에 새로운 방법론인 WTDUN(웨이브릿 영역 심층 전개 네트워크)을 제안합니다. 기존의 심층 전개 방법들이 단일 채널 이미지에서 학습하는 한계와 다양한 이미지 구성 요소를 균일하게 처리하는 문제를 해결하고자 합니다. 제안된 WTDUN은 다중 스케일 웨이브릿 서브밴드에서 직접 작동하여 이미지 내의 중요한 특징을 효과적으로 포착하고 강조합니다.

- **Technical Details**: WTDUN은 웨이브릿 계수의 내재적 희소성과 다중 스케일 구조를 활용하여 나무 구조화된 샘플링 및 재구성을 달성합니다. 이 방법은 다양한 웨이브릿 서브밴드의 상호 의존성을 식별하고, 미세한 특징과 거친 특징을 동시에 고려합니다. 웨이브릿 도메인 적응형 샘플링 방법도 포함되어 있으며, 이는 각 웨이브릿 서브밴드의 중요도에 따라 샘플링을 조정합니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 실험 결과, WTDUN은 기존의 첨단 압축 센싱 기법들보다 우수한 성능을 보여주었습니다. 특히, 세밀한 텍스처와 날카로운 에지의 복원 정확도가 향상되었습니다. 이 연구는 웨이브릿 계수의 구조적 희소성을 최대한 활용하여 이미지 재구성을 개선하는 방법론을 제시함으로써, 압축 센싱 분야에서의 새로운 가능성을 열어갑니다.



### Oriented histogram-based vector field embedding for characterizing 4D CT data sets in radiotherapy (https://arxiv.org/abs/2411.16314)
- **What's New**: 이번 연구에서는 폐 방사선 치료 분야에서 4D CT 데이터와 환자 특이적인 모션 정보를 활용하여 새로운 치료 접근 방식을 제안합니다. 특히, 연구팀은 임상에서 통상적으로 사용되지 않는 모션 필드 통계 분석을 통해 비슷한 환자들의 성공적인 치료 파라미터를 비교하고, 이를 바탕으로 최적의 치료 계획을 수립하고자 합니다. 이러한 접근은 고차원 모션 데이터를 효과적으로 분석하고 클러스터링할 수 있도록 하며, 기존 문헌에서 탐구되지 않은 새로운 차원 축소 방법론을 포함하고 있습니다.

- **Technical Details**: 이 연구에서는 71명의 폐암 환자와 33명의 외부 4D CT 데이터세트를 포함한 총 104개의 4D CT 데이터가 활용되었습니다. 이를 통해, 변형 이미지 등록(deformable image registration) 알고리즘을 이용하여 모션 필드를 생성하고, 2차원 지향 히스토그램을 활용하여 차원 축소 기술을 적용했습니다. 최종적으로는 UMAP(Uniform Manifold Approximation and Projection)을 통해 이렇게 압축된 모션 데이터를 임베딩하여 환자 클러스터를 분석하였습니다.

- **Performance Highlights**: 제안된 방법론은 4D CT 데이터셋으로부터 유사한 호흡 패턴을 가진 환자 클러스터를 효과적으로 식별하는 데 성공했습니다. 이를 통해, 각 클러스터에서 모션 정보의 유사성을 기반으로 적절한 치료 파라미터를 결정할 수 있는 가능성을 보여줍니다. 이러한 연구는 향후 폐암 방사선 치료의 개인화 및 최적화에 기여할 것으로 기대됩니다.



### DoubleCCA: Improving Foundation Model Group Robustness with Random Sentence Embeddings (https://arxiv.org/abs/2411.16236)
Comments:
          18 pages, 6 figures, 2 tables

- **What's New**: 본 논문은 Foundation 모델의 그룹 기반 편향에 대한 견고성을 향상시키기 위한 새로운 방법인 DoubleCCA를 제안합니다. 이 방법은 랜덤 문장과 Canonical Correlation Analysis (CCA)를 활용하여 텍스트 임베딩을 보강합니다. 기존의 모델에 비해 구현이 간단하고 효과적으로 성능과 견고성을 향상시킬 수 있음을 입증합니다.

- **Technical Details**: DoubleCCA 메서드는 원래의 프롬프트를 랜덤 단어로 확장하여 텍스트 임베딩의 표현을 풍부하게 만듭니다. 추가적인 문장 임베딩 모델을 사용하여 이 랜덤 문장들에 대한 다양한 텍스트 임베딩을 생성하고, CCA를 두 번 사용하여 이러한 임베딩을 정렬하고 원래의 표현 공간으로 재구성합니다. 이는 CLIP 모델과 통합되어 그룹 기반 편향에 대한 견고성을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 DoubleCCA 방법은 다양한 데이터셋에서 기존의 방법들보다 성능과 그룹 견고성 모두에서 더 우수한 결과를 보였습니다. 특히, 그룹 속성과 관련된 잡음을 회피하는 능력을 높여주어, 모델의 도메인 일반화 성능도 향상되었습니다. 이 접근 방식은 다양한 비전-언어 모델에 쉽게 적용할 수 있어 실용적인 솔루션이 됩니다.



### EigenHearts: Cardiac Diseases Classification Using EigenFaces Approach (https://arxiv.org/abs/2411.16227)
Comments:
          16 pages, 9 figures, 3 tables

- **What's New**: 이번 논문에서는 심혈관 의학 분야에서 심초음파 이미지를 분류하기 위해 잘 알려진 기법인 EigenFaces 접근법을 활용하는 혁신적인 도구를 제안합니다. 특히, 이 접근법은 주성분 분석(principal component analysis)을 통해 얼굴 이미지를 효율적으로 표현하는 데 사용되는 방법으로, 이를 심장 질병 분류에 적용하고자 했습니다. SVD(특이값 분해)를 활용한 전처리 과정을 통해 얻어진 pod 모드(eigenhearts)를 사용하여 CNN(합성곱 신경망)의 분류 정확도를 약 50% 향상시키는 결과를 보여 주었습니다.

- **Technical Details**: 심초음파 데이터를 활용하여 다섯 가지 심장 상태(건강, 당뇨병성 심근병증, 심근경색, 비만, TAC 고혈압)를 분류하는 연구를 진행했습니다. 이 과정에서, 기존의 심초음파 이미지를 SVD를 통해 처리한 후, pod 모드로 변환하여 새로운 좌표 시스템에서 이미지를 표현했습니다. 이 방식은 CNN이 훈련하는 데 사용되며, CNN은 원본 이미지에 비해 pod 모드를 통해 처리된 이미지를 제공받아 성능 향상을 달성합니다.

- **Performance Highlights**: 제안된 접근법은 SVD 기반 전처리 과정을 통해 크게 향상된 분류 정확도를 보였습니다. 실험 결과, 원본 이미지 대신 pod 모드를 통해 제공된 이미지로 CNN을 학습시켰을 때, 약 50%의 분류 정확도 향상이 있었습니다. 이는 EigenFaces 기법이 심혈관 질병 분류에 효과적으로 적용될 수 있음을 보여줍니다.



### UltraSam: A Foundation Model for Ultrasound using Large Open-Access Segmentation Datasets (https://arxiv.org/abs/2411.16222)
Comments:
          7 pages, 3 figures, 3 tables

- **What's New**: 이번 연구에서는 43개의 공개 초음파 데이터셋을 기반으로 280,000장의 이미지와 세그멘테이션 마스크를 포함하는 US-43d 데이터셋을 구축하였습니다. 또한, Segment Anything Model(SAM)의 변형인 UltraSam을 도입하여 초음파 데이터에 최적화된 다목적 세그멘테이션 모델을 개발하였습니다.

- **Technical Details**: US-43d는 20개의 임상 적용 분야를 포괄하며, 2D 및 3D 스캔에서 수집된 다양한 형태와 질감을 가진 기관 및 병변을 포함하고 있습니다. UltraSam 모델은 SAM 아키텍처를 기반으로 하며, 비전 트랜스포머(ViT) 인코더와 프롬프트 인코더를 결합하여 이미지와 프롬프트 간의 상호작용을 가능하게 합니다.

- **Performance Highlights**: UltraSam은 기존 SAM 스타일 모델들보다 세그멘테이션 성능이 크게 향상되었으며, 여러 개의 다운스트림 과제에서도 우수한 성능을 발휘합니다. 우리는 UltraSam의 기초 모델로서의 효과를 증명하였으며, 코드를 공개하고 커뮤니티의 적극적인 참여를 기대하고 있습니다.



### Video-Text Dataset Construction from Multi-AI Feedback: Promoting Weak-to-Strong Preference Learning for Video Large Language Models (https://arxiv.org/abs/2411.16201)
- **What's New**: 이 논문에서는 Multimodal Large Language Models (MLLMs) alignment에 필요한 고품질의 비디오-텍스트 선호 데이터 세트를 제안합니다. 기존의 비디오 질문-응답(VQA) 선호 데이터는 수집이 어렵고 비용이 많이 들며, 수동 주석이 신뢰성이 떨어지는 문제가 있습니다. 오히려 AI-generated 응답은 온도 조절에 의해 다양성이 부족해 선호 학습을 저해합니다. 이를 해결하기 위해 우리는 Multiple Multimodal Artificial Intelligence Preference Datasets in VQA (MMAIP-V)라는 새로운 데이터 세트를 구축했습니다.

- **Technical Details**: MMAIP-V는 응답 분포 집합에서 샘플링하고, 외부 스코어링 기능을 사용하여 응답 품질을 평가하여 구성된 고품질 VQA 선호 데이터 세트입니다. 또한, Iterative Weak-to-Strong Reinforcement Learning from AI Feedback (Iter-W2S-RLAIF)라는 프레임워크를 통해 참조 모델을 점진적으로 업데이트하며 MLLMs의 alignment 능력을 향상시킵니다. 이러한 접근방식은 선호 데이터의 잠재력을 최대한 활용하고, 모델의 성능을 개선하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, MMAIP-V의 높은 품질의 긍정적 응답과 다양성 있는 부정적 응답이 MLLMs의 선호 학습에 유익하다는 것이 입증되었습니다. 또한, Iter-W2S-RLAIF는 MMAIP-V에 내재된 AI 피드백을 효과적으로 활용하여 VQA 생성 능력을 향상시킵니다. 마지막으로, 비전 기반의 공정한 평가 방식을 도입하여 이전 평가의 편향을 최소화하고 비전 정보를 포함한 다각적 평가를 수행합니다.



### High-Resolution Be Aware! Improving the Self-Supervised Real-World Super-Resolution (https://arxiv.org/abs/2411.16175)
Comments:
          10 pages, 9 figures

- **What's New**: 이번 논문에서는 고해상도 이미지의 인식력을 강화하여 자가 지도 학습(self-supervised learning)을 통한 현실 세계의 초해상도(super-resolution)를 향상시키고자 합니다. 제안된 방법은 초해상도 결과의 품질에 기반하여 열화 모델링(degradation modeling)을 조정하는 컨트롤러(controller)를 포함합니다. 또한, 새로운 특성 정렬 정규화기(Feature-Alignment Regularizer, FAR)를 도입하여 고해상도 이미지의 분포를 제약하여 자연스러운 초해상도 이미지를 생산할 수 있도록 합니다.

- **Technical Details**: 이 방법은 기존의 자가 지도 초해상도 방법에서 고해상도 이미지의 유용성을 통합하여 열화 모델링을 조정합니다. 저해상도(Low-Resolution) 이미지를 위한 새로운 품질 지표를 도입하여 모델의 성능을 최적화합니다. FAR은 고해상도 이미지의 특성과 초해상도 이미지를 비교하여 성능을 개선하는데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 방식은 자연스러운 초해상도 이미지를 생성하는 데 성공하였으며, 최신 지각 성능(perceptual performance) 지표에서 최상급의 결과를 기록했습니다. 이 논문은 초해상도 분야에서 자가 지도 학습의 길을 새롭게 열어줄 것으로 기대됩니다.



### Learning Optimal Lattice Vector Quantizers for End-to-end Neural Image Compression (https://arxiv.org/abs/2411.16119)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이번 연구에서는 Neural image compression에서 기존의 uniform scalar quantization의 단점을 개선하기 위해 rate-distortion optimal lattice vector quantization (OLVQ) 코드북을 설계하는 새로운 학습 방법을 제안합니다. 이 방법은 잠재 feature의 샘플 통계에 따라 LVQ 구조를 보다 잘 맞출 수 있어, 기존 양자화 기법에 비해 더욱 향상된 성능을 보여줍니다. 특히, OLVQ는 uniform scalar quantization의 강점을 유지하면서도 압축 효율성을 크게 개선합니다.

- **Technical Details**: Lattice vector quantization (LVQ)은 고차원 공간에서의 격자 구조를 통해 양자화 과정을 효율적으로 처리합니다. 본 연구에서는 LVQ 기하학을 잠재 DNN feature의 분포에 맞추도록 학습하는 방법을 도입하여, 기존 LVQ 설계의 한계를 극복합니다. 제안된 방법은 입력 벡터를 가장 가까운 격자 점에 매핑하여 양자화를 수행하며, 서로 관련이 있는 feature 간의 상관관계를 더욱 효과적으로 캡처할 수 있습니다.

- **Performance Highlights**: 제안된 OLVQ 방법은 DNN 기반 이미지 압축 시스템의 압축 성능을 상당히 향상시킨다는 사실을 표준 벤치마크 데이터셋을 통해 입증했습니다. 기존의 DNN 양자화 기법에 비해 비율-왜곡(rate-distortion) 성능과 계산적 복잡성 모두에서 우수한 결과를 보여주었습니다. 이러한 연구 결과는 DNN 기반의 시각 신호 압축 문제에 대해 보다 자원 효율적이고 실용적인 솔루션을 제안하는 새로운 가능성을 열어줍니다.



### FUN-AD: Fully Unsupervised Learning for Anomaly Detection with Noisy Training Data (https://arxiv.org/abs/2411.16110)
Comments:
          Accepted at WACV 2025. Supplementary material included after references. 17 pages, 7 figures, 14 tables

- **What's New**: 이 연구에서는 산업 환경에서 발생하는 주석 오류 및 새로운 제품에 대한 레이블 부족 문제를 해결하기 위한 새로운 완전 비지도 (fully unsupervised) 이상 감지 접근 방식을 제안합니다. 기존의 one-class classification 방법이 데이터 오염에 취약한 반면, 본 방법은 두 가지 핵심 관찰에 기반하여 정상 샘플과 이상 샘플을 차별화할 수 있는 가능성을 보여줍니다. 이 알고리즘은 이터레이티브 리컨스트럭티드 메모리 뱅크 (IRMB)를 활용하여 의사 레이블링을 수행함으로써 학습을 진행합니다.

- **Technical Details**: 제안된 방법은 정상 데이터 간의 쌍별 (pairwise) 특성 거리와 가장 가까운 쌍의 특성들이 동질적 (homogeneous) 쌍을 형성한다는 두 가지 관찰을 기반으로 합니다. 차원 축소 및 새로운 손실 함수 (loss function) 설계를 통해 서로 가장 가까운 쌍 간의 클래스 동질성을 촉진하며, 이를 통해 ill-posed 문제의 완화에 기여합니다. 이 연구는 MVTec AD 및 VisA와 같은 공개 산업 데이터셋에서 다양한 오염 환경에서도 최첨단 성능을 달성합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 다른 비지도 환경에서 주어진 다양한 비율의 이상-정상 비율 بما 긍정적인 성능을 나타내며, 전반적으로 높은 이상 감지 및 국소화 (localization) 성능을 보여준다는 것을 입증합니다. 본 연구는 정상 샘플에 대한 정확한 레이블 없이도 효과적인 이상 감지 방법을 개발하는 데 기여하며, 실용적 문제에 대한 해결책을 제공합니다.



### Global spatio-temporal downscaling of ERA5 precipitation through generative AI (https://arxiv.org/abs/2411.16098)
- **What's New**: 본 논문에서는 초기로, 스페이트GAN-ERA5를 제안하여, 전 세계적인 강우(precipitation) 데이터를 심층 학습 기반으로 시공간 내리며(temporal downscaling) 처리합니다. 이 방법은 기존의 ERA5 재분석 데이터의 해상도를 24km 및 1시간에서 2km 및 10분으로 향상시켜, 실제적인 시공간 패턴 및 극단치의 강우율 분포를 정확히 포착할 수 있게 합니다. 스페이트GAN-ERA5는 독일의 데이터로 훈련되었으며, 미국과 호주에서 검증하여 강력한 일반화 성능을 나타내어 전 세계적인 응용 가능성을 입증하였습니다.

- **Technical Details**: 스페이트GAN-ERA5는 조건부 생성 적대 신경망(Conditional Generative Adversarial Network, cGAN)을 활용하여 전 세계의 ERA5 강우 데이터의 시공간 해상도를 증가시킵니다. 이 모델은 시간당 24km 해상도의 ERA5 강우 데이터를 입력으로 받아, 10분 및 2km 해상도의 강우 필드를 생성합니다. 훈련 과정에서 독일 기상청(DWD)에서 제공한 고품질의 조정된 레이더 데이터를 이용하고 있으며, 세 가지 다른 기후 지역에서 성능을 평가하고 있습니다.

- **Performance Highlights**: 스페이트GAN-ERA5는 기존의 범위 기반 기법에 비해 극단치 및 시간적 지속성을 잘 재현하는 능력을 보여줍니다. 예를 들어, 미국의 국소적 강우 세포를 정확하게 복원하며, 이는 ERA5가 인식하지 못하는 크기로, 스페이트GAN-ERA5는 이러한 기상 현상의 시공간 복잡성을 포착할 수 있습니다. 반면, 다른 방법인 rainFARM은 작은 규모의 강우 세포를 복원하는 데 실패하며, 극단치를 과소 추정하는 한계를 보입니다.



### Very Basics of Tensors with Graphical Notations: Unfolding, Calculations, and Decompositions (https://arxiv.org/abs/2411.16094)
- **What's New**: 이번 논문에서는 텐서 네트워크 다이어그램(tensor network diagram)의 중요성에 대해 설명합니다. 이 그래픽 표현은 여러 텐서 간의 곱셈을 직관적으로 나타낼 수 있으며, 텐서 곱의 본질을 이해하는 데 도움을 줍니다. 텐서의 기본 개념과 그 수학적 기호 및 그래픽 표기법을 배우는 것을 목표로 하고 있습니다.

- **Technical Details**: 텐서는 여러 차원의 배열을 가지는 수학적 개체로, 벡터와 매트릭스의 일반화된 형태입니다. 벡터는 수치의 배열로 정의되며, 매트릭스는 벡터의 집합으로 이루어진 직사각형 배열입니다. 이 논문에서는 벡터, 매트릭스, 텐서를 명확히 정의하며, 각각의 표기법에 대해서도 설명합니다.

- **Performance Highlights**: 저자는 복잡한 텐서 연산을 그래픽적으로 표현함으로써 이해를 쉽게 하며, 텐서 분해(tensor decomposition) 기술을 신호 처리(signal processing) 및 머신러닝(machine learning) 분야에 어떻게 적용할 수 있는지를 탐구합니다. 이러한 그래픽 표기는 특히 초보자에게 텐서의 기본적인 구성 요소를 이해하는 데 큰 도움이 될 것입니다.



### Cautious Optimizers: Improving Training with One Line of Cod (https://arxiv.org/abs/2411.16085)
- **What's New**: 본 논문에서는 transformer 사전학습을 위한 기본 최적화 알고리즘인 AdamW에 대한 새로운 접근 방식을 제안합니다. 기존의 여러 최적화 알고리즘보다 더 빠르고 안정적인 성능을 추구하는 과정에서 기존의 모멘텀 기반 최적화 알고리즘에 단 하나의 간단한 수정만으로 새로운 'Cautious Optimizer'를 만들어냈습니다. 이 수정된 최적화 알고리즘은 C-AdamW 및 C-Lion과 같은 형태로 나타납니다.

- **Technical Details**: 이론적인 결과에 따르면, 제안된 수정은 Adam의 해밀토니안 함수(Hamiltonian function)를 보존하며, Lyapunov 분석 하에서도 수렴 보장을 깨뜨리지 않습니다. 이러한 이론적 통찰을 바탕으로 새로운 최적화 알고리즘군을 밝혀내었으며, 이 중 가장 단순한 형태의 알고리즘을 선택하여 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, Llama와 MAE 사전 학습에서 최대 1.47배 더 빠른 속도를 기록하는 것으로 나타났습니다. 제안된 알고리즘은 기존 기법들에 비해 뚜렷한 성능 향상을 보였으며, 해당 코드는 GitHub와 같은 플랫폼에서 공개될 예정입니다.



### Soft-TransFormers for Continual Learning (https://arxiv.org/abs/2411.16073)
- **What's New**: 이번 논문에서는 Well-initialized Lottery Ticket Hypothesis (WLTH)에 영감을 받아 소프트 네트워크(Soft-Network)를 통해 각 작업에 최적화된 지속적인 학습 방법인 Soft-TransFormers(Soft-TF)를 제안합니다. Soft-TF는 연속 학습(Continual Learning, CL) 중에 희소 레이어의 가중치를 공동 최적화하여 작업 적응형 소프트 네트워크를 생성하고, 사전 학습된 레이어의 파라미터를 고정하여 재학습 과정에서 발생하는 망각을 방지합니다.

- **Technical Details**: Soft-TransFormers는 각 작업에 대해 최적화된 소프트 네트워크를 선택하여 연속적으로 학습합니다. 이 방법은 Vision Transformer (ViT)와 CLIP에 대해 실험을 수행하여 CL 시나리오에서 재학습 중 발생하는 Catastrophic Forgetting (CF)을 최소화합니다. Soft-TF는 GPF를 유지하며, 전체 레이어는 고정되지만 특정 작업에 맞춘 가중치를 동적으로 조정합니다. 또한, WLTH에 기반하여 프리트레인 모델을 최적화하여 작업별 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, Soft-TF는 Class-Incremental Learning (CIL) 및 Task-Incremental Learning (TIL) 시나리오에서 DualPrompts와 같은 기존 방법들보다 뛰어난 성능을 보여주었습니다. Soft-TF는 다양한 CL 시나리오에서 최첨단의 성능을 달성하며, 일반화된 CL 모델로의 가능성을 입증합니다. 본 연구는 프리트레인된 Transformer 네트워크의 잠재력을 극대화하여 지속적인 학습 방법에서는 새로운 패러다임 전환을 시도하고 있습니다.



### Peritumoral Expansion Radiomics for Improved Lung Cancer Classification (https://arxiv.org/abs/2411.16008)
Comments:
          2 table, 5 figures

- **What's New**: 이 연구는 결절(nodule) 분할(segmentation)과 주변 종양 주변(peritumoral) 지역이 방사선학(radonics) 기반 폐암 분류에 미치는 영향을 조사하였습니다. 3D CT 스캔 이미지를 사용하여 다양한 분할 기술을 적용하고, 이를 통해 얻은 특성으로 폐암과 비폐암을 분류하는 기계 학습 모델을 개발하였습니다.

- **Technical Details**: 연구에서는 Otsu, Fuzzy C-Means(FCM), Gaussian Mixture Model(GMM), K-Nearest Neighbors(KNN)와 같은 네 가지 기술을 사용하여 3D 분할을 생성했습니다. PyRadiomics 라이브러리를 통해 방사선학적(feature) 특성을 추출하고, Random Forest, Logistic Regression, KNN 등의 여러 기계 학습(classifier) 모델을 사용하여 결절을 분류했습니다. 또한, 초기 결절 분할을 다양한 거리까지 확장하여 주변 지역이 분류에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 주변 종양 주변 지역을 포함함으로써 성능이 크게 향상되었으며, 8mm 확장에서 최고의 결과(AUC = 0.78)를 얻었습니다. 이미지 기반 딥러닝 모델인 FMCB(AUC = 0.71) 및 ResNet50-SWS++(AUC = 0.71)와 비교할 때, 본 연구의 방사선학 기반 접근법이 더 높은 분류 정확성을 보였습니다. 이는 폐암 분류를 개선하는 데 있어 주변 지역의 중요성을 강조하며, 보다 견고한 AI 기반 진단 도구 개발에 기여할 수 있는 발견입니다.



### Cross-organ Deployment of EOS Detection AI without Retraining: Feasibility and Limitation (https://arxiv.org/abs/2411.15942)
Comments:
          8 pages, 5 figures. Accepted by SPIE Medical Imaging 2025 on October 28, 2024

- **What's New**: 이번 연구에서는 만성 비부비동염(Chronic rhinosinusitis, CRS)에서 호산구(Eosinophils) 세포의 자동 세분화(segmentation)를 위한 CircleSnake 모델의 적용 가능성을 탐구합니다. 이 모델은 위장 데이터로 훈련된 것을 기반으로 하여 비강 조직에서 Eos 세포를 식별하는 데 사용되었습니다. 기존의 수동적인 세포 수 계산 방법에 비해 AI 기반의 모델이 진단 과정에서 인적 개입을 줄여주고, 다양한 상황에서의 정확도를 평가합니다.

- **Technical Details**: CircleSnake 모델은 원형 물체를 감지하는 데 최적화된 설계로, 비부비동염 환자의 조직 슬라이드에서 호산구 세포를 세분화하기 위해 직접 적용되었습니다. 입력 이미지를 통해 중심점과 관련된 데이터 맵을 생성하고, 특정 조건에서 모델의 성능을 평가하기 위해 세 가지 그룹으로 결과를 분류했습니다. 이 과정은 실시간 영상 처리 기능을 제공하여 긴급한 의료 환경의 진단 지원을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에서는 일부 조직 슬라이드에서 모두 높은 정확도가 관찰되었지만, 성능 변동성이 존재하였습니다. 모델이 우수한 성능을 발휘하는 조건과 그렇지 않은 조건을 식별함으로써 앞으로 호산구 세포 탐지용 AI 개발에 대한 통찰력을 제공합니다. 향후 데이터 수집 방식을 다양화하고 디지털 염색 기술을 통해 일반화 가능한 AI 모델 개발에 기여할 수 있을 것으로 예상됩니다.



### Improving Pre-Trained Self-Supervised Embeddings Through Effective Entropy Maximization (https://arxiv.org/abs/2411.15931)
Comments:
          19 pages including appendix, 5 figures

- **What's New**: 이 연구는 Self-Supervised Learning (SSL)에서 임베딩의 엔트로피를 극대화하는 새로운 기준인 Entropy Maximization Criterion (E2MC)을 제안합니다. 기존의 연산적으로 어려운 고차원 엔트로피 추정 대신, 저차원 제약 조건으로 쉽게 평가할 수 있는 방식을 사용합니다. 이로써 SSL 모델의 성능을 향상시킬 수 있는 획기적인 접근법을 제공합니다.

- **Technical Details**: 제안된 E2MC는 여러 차원에서 수행되는 전통적인 엔트로피 최대화 방법의 한계를 극복합니다. 연구진은 E2MC를 사용하여 이미 훈련된 SSL 모델을 소수의 에폭(epoch) 동안 더 훈련시켰을 때 일관된 성과 향상을 보여주었습니다. 안전한 성능 증진을 위해 정밀한 ablation study가 실시되었으며, 기존 기준으로 계속 훈련을 진행하였을 때 성능 저하가 발생하기도 했습니다.

- **Performance Highlights**: E2MC를 통한 재훈련이 다운스트림 성능에서 유의미한 개선을 가져왔습니다. 특히, 연구 결과는 제안된 추가 기준이 향상된 성능의 주요 원인임을 입증합니다. 상대적으로 다른 기준으로는 성능 향상이 거의 없거나 오히려 저하되는 결과를 보였다는 점이 중요한 발견으로 언급되었습니다.



### PromptHSI: Universal Hyperspectral Image Restoration Framework for Composite Degradation (https://arxiv.org/abs/2411.15922)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 논문에서는 'PromptHSI'라는 최초의 보편적인 하이퍼스펙트럼 이미지 전문 복원 프레임워크를 제안합니다. 이 프레임워크는 복합적인 손상을 처리하며 주파수 인식(feature modulation) 기반의 새로운 방식을 통해 특정 손상 유형을 효과적으로 복원할 수 있습니다. 기존 RGB 이미지 복원 방식과 달리, 하이퍼스펙트럼 이미지의 독특한 구조와 손상 특성을 적용하여 높은 복원 성능을 유지합니다.

- **Technical Details**: PromptHSI의 핵심은 텍스트 프롬프트를 강도(intensity) 및 바이어스(bias) 컨트롤러로 분해하여 주파수 도메인 분석을 통해 모델이 손상 복원 과정에서 방향성을 갖도록 설계된 것입니다. 이 통합 아키텍처는 미세한 회복 및 글로벌 정보 복원 작업 모두를 효율적으로 처리할 수 있습니다. 주파수 인식(feature modulation) 전략을 통해 HSI 특징에 맞는 주파수 도메인에서의 특성을 활용하여 도메인 간의 간극(domain gap)을 없앱니다.

- **Performance Highlights**: 실험 결과, PromptHSI는 다양한 손상 조합에 대한 복원 성능이 우수하고, 새로운 손상 유형에 대해서도 강한 일반화를 보여줍니다. 특히, 텍스트 프롬프트 컨트롤을 통해 복원 과정을 뛰어나게 관리하며, 기존의 최첨단 RGB 이미지 복원 방법들보다 뛰어난 성능을 달성하였습니다. 추가적으로, 이 프레임워크는 원격 감지(remote sensing) 응용 프로그램을 위한 큰 잠재력을 보입니다.



### Bimanual Grasp Synthesis for Dexterous Robot Hands (https://arxiv.org/abs/2411.15903)
Comments:
          Published in RA-L 24', 8 pages, 9 figures, 3 tables

- **What's New**: 본 논문에서는 로봇의 물체 조작 능력을 향상시키기 위해 이양손(bimanual) 그립 포즈를 생성하는 BimanGrasp 알고리즘을 제안합니다. 이 알고리즘은 그립 안정성과 실현 가능성을 고려하여 에너지 함수를 최적화하여 그립 포즈를 생성합니다. 또한, Isaac Gym 물리 시뮬레이션 엔진을 통해 검증된 BimanGrasp-Dataset을 생성하였으며, 이는 15만 개 이상의 검증된 이양손 그립 포즈를 포함하고 있어 데이터 기반 접근 방식을 통한 이양손 그립 합성이 가능하게 합니다.

- **Technical Details**: BimanGrasp 알고리즘은 확률적 최적화를 사용하여 높은 차원 구성 공간에서 이양손 그립 포즈를 탐색합니다. GPU 기반 최적화를 구현하여 BimanGrasp-Dataset을 합성하는 과정에서, 이 데이터셋은 900개의 객체에 대한 15만 개 이상의 그립을 포함합니다. 각 그립은 Isaac Gym 환경에서 시뮬레이션을 통해 검증되어, 이전의 단일 손 그립 기법으로는 다루기 어려운 대형 및 중량 객체를 처리할 수 있다는 점을 입증합니다.

- **Performance Highlights**: 제안된 BimanGrasp-DDPM 모델은 BimanGrasp-Dataset을 기반으로 훈련되어 69.87%의 그립 합성 성공률을 달성했습니다. 이 모델은 BimanGrasp 알고리즘과 비교하여 계산 속도의 상당한 가속화를 이뤘습니다. 이 연구는 이양손 조작 기량을 데이터 중심 패러다임으로 변환하여 효율적으로 다양한 그립 포즈를 생성할 수 있도록 돕습니다.



### Optimizing Brain Tumor Segmentation with MedNeXt: BraTS 2024 SSA and Pediatrics (https://arxiv.org/abs/2411.15872)
- **What's New**: 이 연구는 Glioma 환자를 위한 MRI에서 종양을 세분화하는 방법론을 제시하며, BraTS-2024 Challenge에 참여한 결과를 중심으로 합니다. 시스템은 MedNeXt 아키텍처와 종합적인 모델 앙상블, 세심한 후처리를 통해 높은 성능을 달성했습니다. 세 가지 종양 하위 영역(Enhancing Tumor, Non-enhancing Tumor Core, Surrounding Non-enhancing FLAIR Hyperintensity)을 정확하게 세분화하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 연구에서 사용된 MedNeXt는 U-Net의 변형으로, ConvNeXt 블록을 적용하여 자동으로 뇌 종양을 세분화합니다. 다양한 데이터 전처리 및 앙상블 기법이 도입되어 실질적인 성능 개선을 이루었고, Dice Similarity Coefficient (DSC) 기준으로 BraTS-2024 SSA 데이터세트에서 평균 0.896, Pediatric Tumor 데이터세트에서 평균 0.830을 기록했습니다. 연구는 BraTS-Africa 및 Pediatric 작업에 집중하여 최근의 데이터셋 확장에 발맞추었습니다.

- **Performance Highlights**: 모델은 BraTS-2024 SSA 데이터세트에서 평균 Hausdorff Distance (HD95) 14.682를, Pediatric 데이터세트에서 37.508을 달성하여 세분화의 정확성을 보여주었습니다. 이는 다양한 인종적 및 환경적 요인에서 발생할 수 있는 분포 변화에 잘 적응할 수 있는 잠재력을 나타냅니다. 이를 통해 뇌 종양 진단 및 치료에서의 신뢰성을 높이고, 기술 발전으로 인한 기회를 제공할 수 있습니다.



### Unveiling the Superior Paradigm: A Comparative Study of Source-Free Domain Adaptation and Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.15844)
Comments:
          Under review

- **What's New**: 이 논문에서는 Unsupervised Domain Adaptation (UDA)와 Source-Free Domain Adaptation (SFDA)의 비교를 통해 SFDA의 우월성을 입증하였습니다. SFDA는 기존의 소스 데이터를 사용하지 않고, 미리 학습된 모델을 활용하여 타겟 도메인에 적응함으로써, 실제 환경에서의 효율성을 크게 향상시킬 수 있음을 보여주었습니다. 또한, 데이터 공유 시나리오를 도입하여 다양한 이해관계자들이 자원을 활용할 수 있는 새로운 방안을 제안하였습니다.

- **Technical Details**: 이 연구는 예측 코딩 이론(predicative coding theory)을 기반으로 SFDA가 기존의 UDA보다 현실 세계에서의 적응력과 효율성을 갖추고 있음을 강조합니다. 특히, 실험을 통해 SFDA가 데이터와 메모리 사용 효율성에서 더 뛰어나며, 부정적 전이(negative transfer)와 과적합(overfitting)으로부터의 저항력이 우수함을 입증했습니다. SFDA는 약 200회 반복(iterations) 내에 수렴하는 반면, UDA는 유사 성능을 달성하는 데 1,000에서 5,000회 반복이 필요합니다.

- **Performance Highlights**: 이 연구의 실험 결과에 따르면, SFDA 방법은 다양한 기준 데이터셋에서 UDA 보다 우수한 성능을 보였고, 특히 복잡한 데이터-모델 융합(data-model fusion) 시나리오에서 효과를 극대화할 수 있음을 입증했습니다. 제안한 가중치 추정(weight estimation) 방법은 기존의 Multi-UDA 및 Multi-SFDA 기술을 뛰어넘어 다양한 데이터 공유 요구사항을 처리하는 데 매우 효과적임을 보여주었습니다. 따라서 이 논문은 현실 세계의 다양한 환경에서 모델 적응을 위한 실용적인 접근 방식을 제공하고 있습니다.



### Variable-size Symmetry-based Graph Fourier Transforms for image compression (https://arxiv.org/abs/2411.15824)
- **What's New**: 본 논문에서는 기존의 8x8 SBGFT(Symmetry-based Graph Fourier Transforms)를 N×N 크기로 확장한 새로운 압축 프레임워크를 제안합니다. SBGFT는 비분리(non-separable) 변환으로서 낮은 계산 복잡성을 유지하면서 희소 신호 표현( sparse signal representation)을 달성합니다. 기존 데이터 의존형 변환의 복잡성과 느린 구현 문제를 극복하고, 다수의 변환을 공유하여 다양한 통계적 특성에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: SBGFT의 설계는 대칭적인 그래프를 생성하는 알고리즘에 기반하고 있으며, 이는 노드 간의 특정 대칭적인 연결을 추가하여 이루어집니다. 이 과정에서 데이터 의존적인 적응이 필요하지 않아 복잡성을 최소화합니다. 비디오 내부 프레임 코딩에서는 최적 그래프와 예측 모드 간의 상관관계를 이용하여 변환 집합의 개수를 줄이는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, SBGFT는 최신 VVC(intra-coding)에서 사용되는 기본 변환들과 비교할 때 비트 전송률을 6.23% 절감하는 성과를 보였습니다. 평균 복잡도는 소폭 증가했으나, 변환의 효과적인 구현과 높은 압축 성능을 보여주었습니다. 이로 인해, 다양한 크기의 SBGFT를 통해 보다 폭넓은 데이터 특성에 대응 가능함을 증명했습니다.



### A Novel Data Augmentation Tool for Enhancing Machine Learning Classification: A New Application of the Higher Order Dynamic Mode Decomposition for Improved Cardiac Disease Identification (https://arxiv.org/abs/2411.15809)
Comments:
          12 pages, 6 figures, 2 tables

- **What's New**: 이번 연구에서는 심장 질환 분류의 정확도를 높이기 위해 HODMD(High Order Dynamic Mode Decomposition)라는 데이터 기반의 모드 분해 방법과 CNN(Convolutional Neural Network)을 결합했습니다. 130개의 심초음파(echocardiography) 이미지를 활용하여 각 심장 질환에 대한 주요 특징을 추출하고, 이 DMD 모드를 CNN의 입력으로 활용하여 데이터 증강(data augmentation)을 시도하였습니다. HODMD가 머신러닝 프레임워크에서 데이터 증강 기법으로 사용된 것은 이번이 처음입니다.

- **Technical Details**: HODMD는 비선형 동적 시스템(modeling non-linear dynamical systems) 분석을 위해 최근에 도입된 데이터 기반 방법으로, 기존의 DMD 메커니즘을 확장한 것입니다. 데이터를 행렬 및 텐서 형식으로 구성하여 처리하며, 이를 통해 심초음파 영상의 주요 특징을 효과적으로 추출할 수 있습니다. 최종적으로 CNN은 원본 이미지와 DMD 모드를 조합한 데이터를 학습하여 성능을 평가합니다.

- **Performance Highlights**: CNN의 성능을 두 가지 테스트 케이스로 평가한 결과, 원본 이미지와 DMD 모드를 결합하여 학습한 경우 정확도가 최대 22% 향상되었습니다. 이는 HODMD 알고리즘이 데이터 준비 시간을 절약하면서도 클래스 분류 정확도를 높이는 데 효과적임을 보여줍니다. 이러한 결과는 HODMD의 데이터 증강 기법으로서의 큰 잠재력을 입증합니다.



### Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2 (https://arxiv.org/abs/2411.15802)
- **What's New**: 본 연구는 DINOv2와 같은 2D 자기 지도(self-supervised) 모델을 3D 의료 이미징에 적용함으로써, 기존 방법들의 한계를 극복하고자 합니다. 특히, 진단의 정확성과 해석 가능성을 높이기 위해 Medical Slice Transformer (MST) 프레임워크를 소개합니다. MST는 Transformer 아키텍처와 2D 특징 추출기 DINOv2를 통합하여 3D 의료 이미지를 분석합니다.

- **Technical Details**: MST 프레임워크는 3D 데이터 세트에서의 효과적인 진단 성능 평가를 위해 설계되었습니다. 이 연구에서는 유방 MRI, 흉부 CT, 무릎 MRI 데이터 세트를 포함하여 총 3개의 임상 데이터 세트를 사용했습니다. 진단 성능은 Receiver Operating Characteristic Curve (AUC)를 계산하여 평가하였으며, 해석 가능성은 방사선 전문의의 정성적 비교를 통해 평가되었습니다.

- **Performance Highlights**: MST는 3개의 데이터 세트 모두에서 ResNet보다 높은 AUC 값을 기록했습니다: 유방 (0.94$	ext{±}$0.01 대 0.91$	ext{±}$0.02, P=0.02), 흉부 (0.95$	ext{±}$0.01 대 0.92$	ext{±}$0.02, P=0.13), 무릎 (0.85$	ext{±}$0.04 대 0.69$	ext{±}$0.05, P=0.001). 또한, saliency maps는 MST에서 항상 더 정밀하고 해부학적으로 정확하다는 것을 보여주었습니다. DINOv2와 같은 자기 지도 2D 모델은 MST를 통해 3D 의료 이미징에 효과적으로 적응될 수 있으며, 이는 합성곱 신경망에 비해 향상된 진단 정확성과 해석 가능성을 제공합니다.



### PG-SLAM: Photo-realistic and Geometry-aware RGB-D SLAM in Dynamic Environments (https://arxiv.org/abs/2411.15800)
- **What's New**: 이 논문은 동적 환경에서 카메라를 로컬라이즈하고 장면을 수복하는 사진 현실적이며 기하학적으로 인지 가능한 RGB-D SLAM 방법을 제안합니다. 이전의 SLAM 방법들이 동적 물체를 필터링하여 발생한 한계를 극복하기 위해, 이 연구는 Gaussian splatting(GS) 기법을 확장했습니다. 제안된 방법은 동적 전경, 정적 배경, 그리고 카메라 로컬라이제이션을 동시에 수행할 수 있는 구조로 설계되었습니다.

- **Technical Details**: 제안된 PG-SLAM 방법은 세 가지 주요 모듈로 구성됩니다. 첫째, 비강체 인간(non-rigid humans)과 강체 물체(rigid items)를 포함한 동적 전경을 재구성합니다. 둘째, 정적 배경을 매핑하며, 셋째, 이동 중인 카메라를 로컬라이즈합니다. 특히, 비강체 인간의 형태와 운동에 대한 제약을 활용하여 동적 GS를 구현하고, 이 과정에서 다중 관점으로부터의 시각적 제약을 통합하여 정확도를 향상시킵니다.

- **Performance Highlights**: 다양한 현실 세계 데이터셋에서의 실험 결과, 제안된 방법이 기존 최첨단 기법보다 카메라 로컬라이제이션과 장면 표현력에서 우수한 성능을 나타냄을 입증했습니다. 특히, 동적 물체와 정적 배경 정보를 활용하여 잡음(noise) 보상 및 로컬라이제이션 정확도를 크게 향상시켰습니다. 이 연구는 동적 환경에서의 SLAM 기술의 새로운 길을 열어줄 것으로 기대됩니다.



### M3-CVC: Controllable Video Compression with Multimodal Generative Models (https://arxiv.org/abs/2411.15798)
Comments:
          Submitted to ICASSP 2025

- **What's New**: M3-CVC는 비디오 압축의 새로운 프레임워크로, multimodal generative models를 통합하여 손실을 최소화하는 동시에 해상도를 유지하는 높은 신뢰성을 제공합니다. 특히, 이는 ultra-low-bitrate 환경에서의 비디오 제어 가능성과 일반성을 동시에 확보한 최초의 신경망 기반 비디오 코덱입니다. 이 프레임워크는 키프레임 선택에 있어 semantic-motion composite 전략을 활용하여 중요한 정보를 보존합니다.

- **Technical Details**: M3-CVC는 키프레임과 해당 비디오 클립에 대해 대화 기반의 대규모 multimodal 모델(LMM)을 활용하여 계층적 시공간(spatiotemporal) 정보를 추출합니다. 이 시스템에서는 조건부 확산 모델(conditional diffusion model)을 이용하여 텍스트 기반의 키프레임 압축 방법을 채택하고, 복원 과정에서 LMM으로부터 도출된 텍스트 설명이 확산 과정을 안내하여 원본 비디오 내용을 명확히 복원합니다.

- **Performance Highlights**: 실험 결과, M3-CVC는 ultra-low bitrate 시나리오에서 기존의 VVC 표준을 크게 초과하는 성능을 보였으며, 특히 의미적(semantic) 및 지각적(perceptual) 신뢰성을 유지하는 데 탁월했습니다. 이를 통해 새로운 압축 기술이 비디오 품질을 손상시키지 않고도 효율적인 전송이 가능함을 입증했습니다.



### Enhancing the automatic segmentation and analysis of 3D liver vasculature models (https://arxiv.org/abs/2411.15778)
Comments:
          Internship at Simbiotx

- **What's New**: 이 연구는 간암 환자의 수술 평가를 위해 의학 이미지에서 혈관 나무(vessel trees)를 자동으로 식별하는 파이프라인을 개발했습니다. 특히, 정맥 나무인 문맥(portal)과 간정맥(hepatic) 구조의 3D 세분화 및 해석을 개선하는 데 중점을 두었습니다. 이 작업을 통해 기존의 골격화(skeletonization) 방법을 개선하고, 수술 계획을 위한 정확한 해부학적 모델을 제공합니다.

- **Technical Details**: 이 논문은 차별화 가능한 골격화 방법이 간 혈관의 3D 세분화 성능에 미치는 영향을 연구합니다. ClDice 및 형태학적 골격화 손실(morphological skeletonization loss)을 사용하여 혈관 나무의 연결성을 개선하고, 단일 클래스 혈관 세분화를 다중 클래스(separating multi-class)로 전환하여 정맥 나무를 분리합니다. 최종적으로, 이러한 알고리즘은 다양한 기하학적 마커를 추출하여 혈관 나무의 형태학적 분석을 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 다양한 직경을 포함하는 광범위한 혈관 나무에 대해 성공적으로 현재의 골격화 방법을 개선합니다. 새로운 분리 알고리즘은 외과의사들에 의해 검증된 낮은 오류율로 혈관의 다중 클래스 세분화를 제공합니다. 이 논문은 또한 77건의 고품질 간 혈관 데이터셋을 공개하였으며, 개별 가지까지 해부학적으로 주석을 추가하고 해당 혈관의 형태학적 분석을 수행할 수 있는 방법을 제시합니다.



### Advanced Learning-Based Inter Prediction for Future Video Coding (https://arxiv.org/abs/2411.15759)
- **What's New**: 이번 논문에서는 AVS4의 Inter Prediction Filter (INTERPF)를 대체할 수 있는 저복잡도 학습 기반의 inter prediction (LLIP) 방법을 제안합니다. LLIP는 경량화된 신경망 모델을 활용하여 필터링 프로세스를 개선하고, 효율적인 추론(inference)이 가능하도록 설계되었습니다. 훈련 데이터셋은 전통적인 INTERPF에서 활용된 픽셀과 좌표를 추출하여 구성하였으며, 신경망의 파라미터는 제3자 의존성 없이도 구현될 수 있습니다.

- **Technical Details**: 제안하는 LLIP는 두 개의 완전 연결층으로 구성된 신경망 아키텍처를 가지고 있습니다. 활성화 함수로는 Rectified Linear Unit (ReLU)가 사용되며, 입력값은 현재 픽셀의 예측값과 네 개의 인접할 픽셀의 재구성값으로 설정됩니다. 인퍼런스 라이브러리는 순수 C++로 구현되어 Libtorch 라이브러리의 의존성을 제거하고, 더 빠른 추론 속도를 제공합니다.

- **Performance Highlights**: 실험 결과, LLIP는 랜덤 접근(Random Access) 구성에서 Y, U, V 구성 요소에 대해 각각 0.01%, 0.31%, 0.25%의 코딩 이득(coding gain)을 달성하였습니다. 이 새로운 접근 방식은 전통적인 비디오 인코딩 방식과의 통합을 용이하게 하여, 딥 러닝 모델을 활용한 비디오 코딩 도구들을 더욱 효율적으로 발전시킬 수 있는 가능성을 제시합니다.



### DynamicAvatars: Accurate Dynamic Facial Avatars Reconstruction and Precise Editing with Diffusion Models (https://arxiv.org/abs/2411.15732)
- **What's New**: 이 논문에서는 기존의 3D 헤드 아바타 생성 방식의 문제점을 해결하기 위해 DynamicAvatars라는 새로운 동적 모델을 제안합니다. 이 모델은 비디오 클립과 얼굴 위치 및 표현에 관련된 파라미터를 사용하여 사진처럼 그럴듯한 움직이는 3D 헤드 아바타를 생성할 수 있습니다. 또한, 사용자 제공 프롬프트와 대규모 언어 모델(LLMs)에서 파생된 가이딩 파라미터를 통합한 새로운 편집 모델을 통해 정밀한 편집이 가능합니다.

- **Technical Details**: 본 연구는 Gaussian Splatting을 기반으로 한 이중 추적 프레임워크를 제안하여 동적 아바타를 생성 및 편집합니다. 우리는 GAN 알고리즘과 LLMs에서 생성된 정확한 가이딩 파라미터와 연결된 제어 모듈을 결합하여 기존의 편집 한계를 극복하였습니다. 또한, 특별히 설계된 프롬프트 전처리 모듈을 도입하여 편집의 안정성을 높였으며, 훈련 데이터셋을 선택적으로 활용하는 동적 편집 전략을 개발하여 모델의 효율성을 개선하였습니다.

- **Performance Highlights**: 제안된 모델은 정밀한 얼굴 특징 편집을 지원하며, 실시간 성능을 유지하며 동적 씬의 편집이 가능합니다. 모델은 비디오 클립에서 3D 헤드 아바타를 성공적으로 재구성할 수 있으며, 사용자가 제공한 상세한 프롬프트에 기반하여 세부 조정이 이루어집니다. 또한, 고급 GAN 기반 생성 모듈을 통해 색상 및 텍스처 왜곡 문제를 해결하여 결과의 품질과 사실성을 현저히 향상시켰습니다.



### Comparative Analysis of Diffusion Generative Models in Computational Pathology (https://arxiv.org/abs/2411.15719)
Comments:
          Submitted paper under review

- **What's New**: 본 논문은 병리 데이터 특성과 확산 생성 모델(Diffusion Generative Models, DGM)의 고품질 생성 능력을 연결하는 심층적인 비교 분석을 제공합니다. 여기서 DGM은 병리 데이터셋에 적용된 다양한 방법론을 조사하며, 여러 시야(Field of View, FOV)에서의 생성 성능을 비교합니다. 특히, 이미지를 생성하는 과정에서 크기 조정이 다양한 시야를 모사할 수 있는 방법에 대해 논의합니다.

- **Technical Details**: 확산 모델은 확률적 생성 모델의 한 종류로, 복잡한 데이터 분포를 학습하는 데 사용됩니다. 논문에서는 U-Net 기반의 신경망이 이를 통해 노이즈를 예측하고, Denoising Diffusion Probabilistic Models (DDPM)와 같은 기법을 통해 정제된 이미지를 생성하는 과정이 설명됩니다. forward process와 reverse process를 통해 이미지의 노이즈를 추가하고 제거하며, 최종 이미지를 생성하는 방법론이 소개됩니다.

- **Performance Highlights**: 연구 결과, DGM은 고품질의 합성 데이터 생성에서 매우 효과적임을 입증했습니다. 특히 생성된 합성 데이터가 실제 데이터와 결합되어 심층 학습 모델의 정확성을 크게 향상시킬 수 있음을 보였습니다. 고유한 FOV에 따라 다양한 수준의 디테일을 생성하는 능력이 강조되며, 이는 병리 연구 및 교육 환경에서의 효용성을 높이는 데 기여할 것으로 기대됩니다.



### Editable-DeepSC: Reliable Cross-Modal Semantic Communications for Facial Editing (https://arxiv.org/abs/2411.15702)
- **What's New**: 이번 연구에서는 Editable-DeepSC라는 새로운 크로스 모달 시맨틱 커뮤니케이션 방법을 제안합니다. 이 방법은 텍스트 지침에 따라 얼굴 이미지를 편집하는 시맨틱 정보를 전달하는 데 중점을 두고 있습니다. 기존의 데이터 지향 통신 설계 원칙을 넘어, 실시간 CV 작업의 특정 요구 사항에 따라 다르게 인코딩되고 전송되는 시맨틱 정보를 활용합니다. 이로 인해 제한된 전송 대역폭을 보다 효율적으로 사용할 수 있게 됩니다.

- **Technical Details**: Editable-DeepSC는 GAN 기반의 코딩 기법을 사용하여 고차원 데이터를 저차원 표현으로 압축합니다. Joint Editing-Channel Coding (JECC) 방식을 통해 통신 및 편집을 통합하여보다 효율적인 데이터 처리 절차를 구현합니다. 또한, SNR 인식 채널 코딩을 통해 다양한 채널 노이즈 조건에 적응할 수 있도록 모델 미세 조정을 수행합니다. 이러한 접근 방식은 편집 효과를 유지하면서도 전송 대역폭을 효과적으로 절약할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, Editable-DeepSC는 기존 방법들에 비해 뛰어난 편집 성능을 보였습니다. 예를 들어, 고해상도 환경에서 Editable-DeepSC는 2% 미만의 대역폭으로도 최고의 편집 성능을 달성하였습니다. 또한, 미세 조정 과정에서 전체 매개변수의 2.65%만을 조정했음에도 불구하고, 낮은 SNR에서도 편집 효과를 상당히 개선하고 상대적으로 노이즈 손상으로부터 편집된 시맨틱을 보호하는 결과를 얻었습니다.



### Machine-agnostic Automated Lumbar MRI Segmentation using a Cascaded Model Based on Generative Neurons (https://arxiv.org/abs/2411.15656)
Comments:
          19 Pages, 11 Figures, Expert Systems with Applications, 2024

- **What's New**: 이번 연구에서는 MRI 이미지에서 요추(lumbar vertebrae)와 추간판(intervertebral discs)을 세분화하기 위한 새로운 머신-아그노스틱(machin-agnostic) 접근법을 소개합니다. 이 방법은 ROI(Region of Interest) 탐지와 Self-organized Operational Neural Network(Self-ONN) 기반의 인코더-디코더 네트워크를 결합한 계단식(cascaded) 모델을 사용합니다.

- **Technical Details**: 연구는 12개의 스캐너(scanner)와 34명의 대상을 포함하는 독특한 데이터셋을 기반으로 하며, 다양한 MRI 모달리티(modality)의 도전에 대응하기 위해 전략적인 전처리(preprocessing)와 데이터 증가(data augmentation) 기법을 활용합니다. YOLOv8 중간 모델이 ROI 추출에서 0.916 mAP 점수를 기록하며 뛰어난 성과를 보였습니다.

- **Performance Highlights**: Self-ONN 기반 모델과 DenseNet121 인코더의 조합은 요추와 IVD 세분화에서 평균 IoU(Intersection over Union) 83.66%, 민감도(sensitivity) 91.44%, Dice 유사도 계수(Dice Similarity Coefficient) 91.03%로 매우 우수한 성능을 나타냈습니다. 이 연구는 척추 질환 관련 MRI 세분화에 효과적인 방법을 제시하며, 자동화 진단 도구의 향후 발전을 위한 기반을 마련합니다.



### Machine Learning-based sEMG Signal Classification for Hand Gesture Recognition (https://arxiv.org/abs/2411.15655)
Comments:
          IEEE BIBM 2024

- **What's New**: 이 논문은 EMG(근전도) 기반 손 제스처 인식을 위한 새로운 기능 추출 기법을 도입하여 머신 및 딥러닝 모델과 결합한 성능 벤치마킹을 진행합니다. 또한 Grabmyo와 FORS-EMG라는 최신 데이터셋을 사용하여 이를 실험적으로 검증하였습니다. 이를 통해 EMG 신호를 활용한 제스처 인식의 정확성을 높일 수 있는 다양한 방법을 제시하고 있습니다.

- **Technical Details**: 연구에서는 wavelet transformation 기반의 기능, 융합 시간 영역 기술(fused time-domain descriptors, fTDD) 및 시간-공간 기술(temporal-spatial descriptors, TSD)을 사용하여 EMG 신호의 시간적 및 공간적 특성을 포착합니다. 각 신호는 600ms의 겹치는 윈도우로 나누어져 50%의 중첩을 통해 중요한 특징이 추출됩니다. 또한, 파워 스펙트럴(moment)과 비율들을 포함하여 EMG 신호의 에너지와 복잡성을 반영하는 여섯 가지 특징이 계산됩니다.

- **Performance Highlights**: 실험 결과, 1D Dilated CNN이 Grabmyo 데이터셋에서 97%의 정확도로 최고의 성능을 보였으며, FORS-EMG 데이터셋에서는 Random Forest가 94.95%의 정확도를 기록했습니다. 이러한 결과는 고급 머신 및 딥러닝 알고리즘과 혁신적인 특징 추출 기법의 조합을 통해 가능했습니다. 본 연구는 EMG 기반 제스처 인식의 실용성을 높일 수 있는 가능성을 보여줍니다.



### Comparative Analysis of Resource-Efficient CNN Architectures for Brain Tumor Classification (https://arxiv.org/abs/2411.15596)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구는 MRI 이미지를 통한 뇌 종양 분류에서 효과적이면서 간단한 Convolutional Neural Network (CNN) 아키텍처와 사전 훈련된 ResNet-18 및 VGG-16 모델을 비교 분석합니다. 연구의 결과, 사용자 정의 CNN은 복잡성이 낮음에도 불구하고 사전 훈련된 모델들과 경쟁력 있는 성능을 발휘하였습니다. 이 연구는 의료 이미징 작업에 있어 잘 설계된 간단한 CNN 아키텍처의 잠재력을 강조합니다.

- **Technical Details**: 이 연구에서는 Br35H: Brain Tumor Detection 2020 및 Brain Tumor MRI Dataset과 같은 두 가지 공개 데이터셋을 사용하여 뇌 종양을 분류하기 위한 맞춤형 CNN 아키텍처와 ResNet18 및 VGG16을 비교했습니다. 맞춤형 CNN은 Br35H 데이터셋에서 98.67%, Brain Tumor MRI Dataset에서 99.62%의 높은 정확도를 달성하였고, 다중 클래스 분류에서는 98.09%의 정확도를 기록했습니다. 또한, 연구에서는 몇 샷 학습(few-shot learning) 조건에서 맞춤형 CNN의 강건성을 평가하였습니다.

- **Performance Highlights**: 사전 훈련된 ResNet18 및 VGG16 모델은 여전히 높은 성능을 유지했지만, 맞춤형 CNN은 더 나은 계산 효율성을 제공하여 자원 제약 환경에서도 경쟁력을 보였습니다. 이번 연구는 사용자가 정의한 CNN 아키텍처가 의료 이미징에서의 활용 가능성을 보여주며, 보다 간단한 모델들이 어떻게 높은 정확도와 계산 효율성을 동시에 달성할 수 있는지를 탐구하는 방향으로 나아가고 있습니다.



### Classifier Enhanced Deep Learning Model for Erythroblast Differentiation with Limited Data (https://arxiv.org/abs/2411.15592)
Comments:
          14 pages, Accepted for the 27th International Conference on Pattern Recognition (ICPR 2024)

- **What's New**: 본 논문은 다양한 혈액 질환의 진단에서 Erythroblast(적혈모구)와 WBCs(백혈구)를 구분하는 데 어려움이 있는 문제를 다루고 있습니다. 이를 위해 ResNet-50을 기반으로 한 여러 머신러닝(ML) 분류기의 효용성을 평가하며, SVM, XG-Boost, KNN 및 Random Forest를 활용합니다. 흥미롭게도, 훈련 데이터가 적더라도 ResNet50-SVM 분류기가 다른 모델보다 지속적으로 높은 성능을 보였습니다.

- **Technical Details**: 대상 데이터 세트는 Mendeley 저장소에서 획득한 17,092개의 JPEG 이미지를 포함하고 있으며, 이 자료는 다양한 면역세포 유형을 포함합니다. 모든 실험은 Nvidia GeForce RTX-3050 GPU에서 수행되었으며, ResNet-50을 주요 특징 추출기로 사용하여 70%를 훈련 데이터로, 15%를 검증 데이터로, 나머지 15%를 테스트 데이터로 나누어 진행하였습니다. 여러 모델들 가운데, ResNet-50이 98.72%의 테스트 정확도를 기록하여 최적의 성능을 보여주었습니다.

- **Performance Highlights**: ResNet-50을 활용한 SVM 모델은 전체 데이터 셋의 1%인 168이미지로 훈련을 했음에도 불구하고 86.75%의 테스트 정확도와 98.9%의 Erythroblast 정확도를 달성했습니다. 이는 일반적인 딥러닝 모델을 초월하여, 자원이 제한된 환경에서도 높은 분류 정확도를 얻을 수 있다는 점에서 기여하는 바가 큽니다. 본 연구는 전통적인 수작업 혈액세포 분류 방법을 자동화하는 데 유효한 솔루션을 제공하고 있습니다.



### MulModSeg: Enhancing Unpaired Multi-Modal Medical Image Segmentation with Modality-Conditioned Text Embedding and Alternating Training (https://arxiv.org/abs/2411.15576)
Comments:
          Accepted by WACV-2025

- **What's New**: 이 논문에서는 CT 및 MR 이미지를 사용하는 Multi-Modal Segmentation (MulModSeg) 전략을 제안합니다. MulModSeg는 기존의 세그멘테이션 모델에 큰 구조적 수정이나 계산 비용을 추가하지 않으면서 모달리티 인식을 통합하는 방법입니다. 이 전략은 두 가지 주요 혁신을 포함하고 있는데, 하나는 고정된 CLIP 텍스트 인코더를 활용한 모달리티 조건 텍스트 임베딩이고, 또 다른 하나는 비례한 CT 및 MR 입력의 필수 특징을 통합하는 교대 훈련 알고리즘입니다.

- **Technical Details**: MulModSeg의 두 가지 주요 디자인은 첫째, 고정된 CLIP 텍스트 인코더를 통한 모달리티 특정 텍스트 임베딩 프레임워크로, 세그멘테이션 프레임워크에 모달리티 인식을 추가합니다. 둘째, 교대 훈련 절차(ALT)로, 비례한 이미지에서 필수 특징을 통합하여 다중 모달리티 의학 이미지 세그멘테이션을 촉진합니다. 이 방법은 기존의 FCN 기반 및 Transformer 기반 세그멘테이션 네트워크에 간편하게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, MulModSeg는 CT와 MR 모달리티 모두에서 복부 다기관 및 심장 서브구조 분할 작업에서 이전 방법들보다 일관되게 우수한 성능을 보였습니다. MulModSeg는 잘 알려진 기준 방법들과 비교했을 때, 텍스트 입력이 있는 경우와 없는 경우 모두에서 뛰어난 결과를 기록했습니다. 이 방법은 또래시기 훈련을 통해 뛰어난 정확성을 보여주였습니다.



### Reassessing Layer Pruning in LLMs: New Insights and Methods (https://arxiv.org/abs/2411.15558)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 레이어 프루닝(layer pruning) 방법을 다루고 있으며, 특히 단순한 방법론이 복잡한 메트릭을 초월할 수 있음을 보여주고 있습니다. 연구진은 Llama-3.1-8B-Instruct를 프루닝하여 새로운 모델 Llama-3.1-6.3B-It-Alpaca와 Llama-3.1-6.3B-It-Dolly을 발표하였으며, 이 모델은 유사한 크기의 다른 인기 LLM보다 우수한 성능을 보입니다. 또한, 프루닝 후 fine-tuning 방법으로는 LoRA 방식이 아니라 부분 레이어 조정(partial-layer fine-tuning)을 권장합니다.

- **Technical Details**: 이번 연구는 7가지 레이어 선택 메트릭과 4개의 오픈소스 LLM, 6가지 fine-tuning 방법, 5가지 프루닝 전략을 사용하여 레이어 프루닝의 성능을 폭넓게 평가하였습니다. 결과적으로, 마지막 25% 레이어를 프루닝하고 lm_head와 최후의 3개 레이어를 fine-tuning하는 접근 방식이 가장 강력한 성능을 발휘함을 확인했습니다. 또한, iterative pruning이 one-shot pruning보다 우수성이 없음을 발견했습니다.

- **Performance Highlights**: Llama-3.1-6.3B-It-Alpaca 및 Llama-3.1-6.3B-It-Dolly 모델은 훈련 토큰 수가 106×10^6분의 1로 줄어들지만, ChatGLM2-6B, Vicuna-7B-v1.5, Qwen1.5-7B, Baichuan2-7B 등 다양한 유사 크기 모델들과 비교했을 때 더 나은 성능을 보여주었습니다. 연구진은 이 모델들의 최적 가중치를 Huggingface에 공개하였으며, 코드 또한 GitHub에서 제공하고 있습니다. 이 연구는 LLM 프루닝을 위한 실용적인 가이드를 제공하여 향후 연구와 실제 적용에 도움이 될 것입니다.



### Multi-scale Cascaded Large-Model for Whole-body ROI Segmentation (https://arxiv.org/abs/2411.15526)
- **What's New**: 이번 연구에서는 Multi-scale Cascaded Fusing Network (MCFNet)라는 혁신적인 네트워크 구조를 제안합니다. MCFNet은 복잡한 다중 스케일 및 다중 해상도의 특징을 효과적으로 포착할 수 있는 능력을 가지고 있습니다. 또한, MCFNet은 다운샘플링 및 스킵 연결 단계에서 특징 추출을 향상시키는 Sharp Extraction Backbone과 Flexible Connection Backbone을 포함하고 있어, 저해상도 이미지에서도 정확한 세부정보를 포착할 수 있도록 합니다.

- **Technical Details**: MCFNet은 전체 신체 ROI 세분화를 위한 다중 스케일 캐스케이드 U형 네트워크를 기반으로 설계되었습니다. 이 모델은 두 개의 백본 네트워크인 FCB (Flexible Connection Backbone)와 SEB (Sharp Extraction Backbone)를 결합하여 이미지의 다양한 스케일과 해상도를 처리합니다. 새로운 적응형 손실 집계 전략인 Adaptive-MFA (Adaptive Multi-scale Feature-Mixing Loss Aggregation)를 도입하여 모델 훈련을 최적화하고, 세분화 정확도를 개선합니다.

- **Performance Highlights**: MCFNet은 671명의 환자로부터 수집된 10개의 다양한 데이터셋을 사용하여 실험을 수행하였고, 10개 데이터셋 모두에서 일관되게 우수한 성능을 보였습니다. 또한, MCFNet은 다양한 임상 시나리오에서 높은 정확도를 유지하며 뛰어난 일반화 능력을 보여주었습니다. 이러한 특성으로 인해 MCFNet은 방사선 치료 및 수술 절차의 정밀도와 안전성을 크게 향상시킬 수 있는 잠재력을 가지고 있습니다.



### SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation (https://arxiv.org/abs/2411.15513)
- **What's New**: 본 논문은 의료 영상 분할을 위한 새로운 프레임워크인 SPA(Segmentation Preference Alignment)를 제안합니다. SPA는 사용자 선호에 맞춰 불확실성을 효과적으로 처리할 수 있도록 설계되어 있습니다. 이를 통해 모델은 사용자가 선택할 수 있는 몇 가지 구분 후보를 제공하여 임상의의 부담을 줄입니다. 기존 방법들과 비교하여 병원 현장에서의 활용 가능성을 높이는 것이 특징입니다.

- **Technical Details**: SPA 모델은 Gaussian 혼합 분포를 사용하여 사용자 선호를 모델링합니다. 이 분포를 통해 다양한 세그멘테이션 후보를 생성하고, 사용자는 직관적으로 선택할 수 있습니다. 이 과정에서 임상의의 피드백을 고려해 모델이 빠르게 조정되어, 원하는 세그멘테이션에 도달할 수 있도록 돕습니다. 이러한 방법은 임상의가 계산한 시간과 노력을 절감합니다.

- **Performance Highlights**: SPA는 기존의 상호작용 모델 대비 39% 적은 반복 작업으로 35% 더 많은 이미지를 분할할 수 있음을 입증했습니다. 여러 의료 이미지 세그멘테이션 작업에서 우수한 성능을 보여주며, REFUGE2, LIDC-IDRI 및 QUBIQ와 같은 다양한 데이터셋에서 뛰어난 결과를 달성하고 있습니다. 이처럼 SPA는 다중 의사 주석 데이터셋의 Dice Score에서 최첨단 성능을 확보하고 있습니다.



### Automatic Evaluation for Text-to-image Generation: Task-decomposed Framework, Distilled Training, and Meta-evaluation Benchmark (https://arxiv.org/abs/2411.15488)
- **What's New**: 이 논문은 텍스트-이미지 생성 모델의 평가 방법을 개선하기 위해, GPT-4o 기반의 작업 분해 평가 프레임워크를 제안합니다. 이를 통해 복잡한 평가 작업을 단순한 서브 작업으로 나누어 처리함으로써 자동으로 새로운 학습 데이터셋을 구성하게 됩니다. 이 프레임워크는 7B 개방형 MLLM, MiniCPM-V-2.6의 효율적인 자동 평가 모델로의 변환을 촉진합니다.

- **Technical Details**: 제안된 프레임워크는 입력 텍스트 프롬프트에서 сущности(entity)와 내재적 속성(intrinsic properties), 그리고 관계 속성(relational attributes)을 추출하기 위해 GPT-4o를 사용합니다. 이러한 정보를 바탕으로 세 가지 차원(시각적 외형, 내재적 특성, 관계 속성)을 통해 평가 질문을 구성하고, 이를 통해 이미지와 캡션 간의 품질 점수를 산출합니다. 또한, 개별 평가 차원에 대해 예측된 결과를 통합하여 종합적인 판단을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 MLLM은 기존의 최첨단 모델인 GPT-4o-베이스 기준 및 VIEScore와 비교하여 Spearman 및 Kendall의 상관 관계에서 4.6% 이상의 개선을 달성했습니다. 또한, 새롭게 수동으로 주석이 달린 메타 평가 벤치마크를 통해 기존의 평가 방법들과 비교하여 더욱 신뢰성 있는 평가를 제공합니다. 이 모델은 고품질의 생성된 이미지를 평가하는 데 더욱 효과적임을 입증했습니다.



### Gotta Hear Them All: Sound Source Aware Vision to Audio Generation (https://arxiv.org/abs/2411.15447)
Comments:
          16 pages, 9 figures, source code released at this https URL

- **What's New**: 본 논문에서는 Sound Source-Aware V2A (SSV2A) 생성기를 제안하여 비디오와 정적 이미지로부터 더 세밀한 오디오 생성을 목표로 합니다. 기존의 V2A 방법들이 전반적인 장면에만 의존한 반면, SSV2A는 현장 내의 개별 소리 발생 원소를 인식하고 분석합니다. 이를 통해 우리는 소리 생성에서 더욱 풍부한 정교함과 몰입감을 달성할 수 있음을 보여줍니다.

- **Technical Details**: SSV2A는 시각 감지 및 크로스 모달리티 변환을 통해 다양한 소리 발생 원소를 인식하도록 설계되었습니다. 이 시스템은 Cross-Modal Sound Source (CMSS) Manifold를 통해 각 소리 발생 원소의 의미를 구분하며, 세밀한 오디오 표현을 위해 이들 CMSS 의미를 혼합합니다. 실험에서는 VGGSound 데이터를 기반으로 하는 고유한 단일 소리 발생 원소 데이터셋 VGGS3를 구축하였으며, Cross-Modal Contrastive Mask Regularization (CCMR)을 적용하여 클립 간의 의미를 유지합니다.

- **Performance Highlights**: 실험 결과, SSV2A는 기존의 최첨단 방법들을 능가하는 우수한 생성 충실도 및 관련성을 보여주었습니다. 또한, 다양한 모달리티의 소리 발생 프롬프트를 혼합하여 직관적인 생성 제어를 통해 사용자 요구에 맞는 오디오 생성을 가능하게 합니다. SSV2A의 결과는 사용자들이 직관적으로 소리 생성의 질을 조절할 수 있는 새로운 파라다임을 제시합니다.



### Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts (https://arxiv.org/abs/2411.15432)
- **What's New**: 본 논문에서는 Lifelong Vision language model Editing을 위한 새로운 프레임워크인 LiveEdit를 제안합니다. 이 프레임워크는 기존의 LLM 편집 기술과 VLLM의 차이를 극복하기 위해 설계되었으며, 특히 지속적 학습 상황에서 VLLMs의 지식 수정을 가능하게 합니다. LiveEdit는 저랭크 전문가( low-rank experts)를 생성하고 이를 하드 및 소프트 라우팅 메커니즘을 통해 결합하여 VLLM의 응답을 조정합니다.

- **Technical Details**: LiveEdit는 두 단계의 라우팅 전략을 활용하여 전문가를 라우팅합니다. 첫 번째 단계에서는 입력 샘플과 관련된 시각적 특징을 추출하여 시각적으로 관련성이 없는 전문가를 필터링합니다. 이 하드 라우팅 단계 이후, 다수의 전문가를 융합하여 입력 쿼리와 편집 텍스트 간의 유사성을 고려한 소프트 라우팅 단계가 진행됩니다. 이를 통해 입력 샘플에 최적화된 전문가 집합을 선택하고 조정된 응답을 생성합니다.

- **Performance Highlights**: LiveEdit는 E-VQA, E-IC, VLKEB 등 다양한 벤치마크 데이터셋에서 1, 10, 100, 1,000개의 편집을 통해 테스트되었습니다. 그 결과, 기존의 강력한 편집기와 비교했을 때 LiveEdit는 뛰어난 성능을 보였으며, 각 모듈 설계의 합리성과 효율성도 검증되었습니다.



### Improved Background Estimation for Gas Plume Identification in Hyperspectral Images (https://arxiv.org/abs/2411.15378)
Comments:
          13 pages, 10 figures, submitted and under review to IEEE Transactions on Geoscience and Remote Sensing

- **What's New**: 이 논문에서는 저파장 적외선 하이퍼스펙트럴 이미징을 이용한 가스 배출 감지 및 식별을 위한 새로운 배경 추정 방법을 제안합니다. 현재의 일반적인 방법인 글로벌 배경 추정의 한계를 극복하기 위해 두 가지 방법(PCA 및 K-Nearest Segments)과 세 가지 기존 방법을 비교하고 있습니다. 논문은 640개의 시뮬레이션된 플룸을 사용하여 배경 방사선 추정 성능과 신뢰도를 평가합니다.

- **Technical Details**: 가스 플룸 분석은 감지, 식별 및 정량화 세 가지 주요 작업으로 나뉩니다. 특히, 저파장 적외선에서 가스 배출량을 정확히 파악하기 위해 배경 방사선(𝐿_{off})을 추정하고 이를 제거하는 작업이 필수적입니다. 이 논문에서는 주성분 분석(PCA), K-평균 클러스터링, K-최근접 이웃 등 세 가지 비공간 방법과 두 가지 공간 방법(원형 방법 및 K-최근접 세그먼트)을 이용하여 배경 추정 성능을 평가합니다.

- **Performance Highlights**: 제안된 PCA 방법은 글로벌 배경 추정법에 비해 MSE(평균 제곱 오차)를 18,000배 낮추는 성능을 보였습니다. 또한, K-Nearest Segments 알고리즘은 신경망 기반의 가스 식별 신뢰도를 53.2% 향상시켰습니다. 이를 통해 저신호 강도 및 다양한 배경의 효과적인 가스 감지 및 식별이 가능하다는 것을 보여줍니다.



### Personalization of Wearable Sensor-Based Joint Kinematic Estimation Using Computer Vision for Hip Exoskeleton Applications (https://arxiv.org/abs/2411.15366)
- **What's New**: 본 논문은 실시간으로 관절 운동학(joint kinematics)을 추정할 수 있는 새로운 딥러닝(Deep Learning) 프레임워크를 제안합니다. 기존의 방법들은 많은 데이터 세트를 필요로 하였으나, 본 연구는 1-2 보행 사이클의 작은 데이터 세트로도 높은 정확도를 달성할 수 있음을 보여줍니다. 이 프레임워크는 외부 모션 캡처 설정에 의존하지 않으며, 스마트폰 카메라와 결합하여 새로운 사용자에 대한 적응성도 유지합니다.

- **Technical Details**: 이 연구에서는 인체의 하체 관절 운동학을 추정하기 위해 착용형(IMU) 센서를 사용한 실시간 시스템을 구축하였습니다. IMU 데이터는 Raspberry Pi를 통해 수집되며, 20ms 이내에 실시간 추론을 가능하게 합니다. 또한, MMpose 라이브러리를 이용하여 2D 관절 점 추정 후, 3D 점으로 변환하여 관절 각도를 계산하는 방식으로 이루어집니다.

- **Performance Highlights**: 모델은 stiff knee gait 데이터에 대해 전이 학습(transferred learning)을 수행하여 루트 평균 제곱 오차(RMSE)를 기존 TCN 모델보다 9.7%에서 19.9%까지 줄였습니다. 이로 인해 비정상 보행 패턴을 가진 사용자에 대한 모델의 적응성과 정확성이 향상되어, wearable robots와 같은 적용 분야에서 유용할 것으로 기대됩니다.



### Deep Learning-Based Automatic Delineation of Liver Domes in kV Triggered Images for Online Breath-hold Reproducibility Verification of Liver Stereotactic Body Radiation Therapy (https://arxiv.org/abs/2411.15322)
- **What's New**: 이번 연구에서는 간암 및 간 전이의 치료법으로 사용되는 Stereotactic Body Radiation Therapy (SBRT)의 효율을 향상시키기 위한 새로운 접근법을 제안합니다. 기존의 방법들이 수동 검증을 요구하는 반면, 본 연구에서는 deep learning 기술을 활용하여 kV-triggered 이미지에서 자동으로 간 돔(liver dome)을 구분하는 파이프라인을 개발했습니다.

- **Technical Details**: 연구는 24명의 SBRT 환자로부터 수집된 711개의 kV-triggered 이미지를 기반으로 진행되었습니다. U-Net 모델을 활용한 간 영역(segmentation) 자동화를 통해 간 돔을 초깃 이미지로부터 분할하고, 이어서 임계값(thresholding), edge detection, 형태학적(morphological) 연산을 통해 간 돔을 추출합니다. 또한, 2-겹 교차 검증(cross validation)을 통해 모델의 성능과 일반화 가능성을 평가했습니다.

- **Performance Highlights**: U-Net 모델의 학습은 30분 이내에 완료되었으며, 자동으로 간 돔을 delineation 하는 데에는 1초도 걸리지 않았습니다. Fold1의 366개 이미지에서 RMSE는 (6.4 +/- 1.6) mm, 검출 비율은 91.7%였고, Fold2의 345개 이미지에서는 RMSE가 (7.7 +/- 2.3) mm, 검출 비율은 76.3%로 나타났습니다.



### Frequency-Guided Posterior Sampling for Diffusion-Based Image Restoration (https://arxiv.org/abs/2411.15295)
- **What's New**: 이번 논문은 이미지 복원에 대한 새로운 접근 방식을 제안합니다. 기존의 diffusion 모델을 활용하여 이미지 복원의 품질을 높이기 위해 손실 예측의 근사 오차에 대한 엄밀한 분석을 제공합니다. 또한, 복원 과정에서 시간 변화에 따른 저주파 필터를 도입하여 성능을 개선하는 방법을 제안하고 있습니다.

- **Technical Details**: 이미지 복원 문제를 역문제(inverse problem)로 설정하고, 노이즈가 추가된 관측 자료에서 깨끗한 이미지를 샘플링하는 방법을 소개합니다. 이 과정에서 Bayesian 접근 방식을 활용하여 손실 예측을 기반으로 의사 결정을 수행합니다. 특히, 주파수 도메인에서의 저주파 필터의 적용을 통해 고주파 정보를 점진적으로 통합하는 방법을 개발합니다.

- **Performance Highlights**: 제안된 방법은 다양한 이미지 복원 작업에서 주목할 만한 성능 향상을 보여줍니다. 특히, 운동 블러(motion deblurring)와 이미지 해제(imaging dehazing)와 같은 어려운 복원 작업에서 σημαν적인 개선을 이루었습니다. 이를 통해 사용자에게 더 매력적인 재구성을 제공하고, 이미지 복원 강도에 대한 새로운 통찰을 제공합니다.



### A Plug-and-Play Temporal Normalization Module for Robust Remote Photoplethysmography (https://arxiv.org/abs/2411.15283)
- **What's New**: 이 연구는 Temporal Normalization (TN) 모듈을 소개하면서, 기존의 원격 광파동측정(rPPG) 네트워크 아키텍처에 플러그 앤 플레이 형식으로 통합될 수 있는 방법을 제시합니다. TN은 장기적인 신호 변화를 포착하여 모션과 조명 아티팩트를 효과적으로 완화하고, rPPG 예측 성능을 크게 향상시킵니다. 이를 통해 심박 측정 과제에서 34.3%에서 94.2%에 이르는 성능 향상이 갈수록 작은 모델에서 더욱 두드러지게 나타났습니다.

- **Technical Details**: SHAFFer반사모델(SRM)을 기초로 하여 TN은 차별적 특성을 정규화된 특성으로 변환하여 더욱 강력함을 목표로 하고 있습니다. TN 모듈은 잡음에 대한 불변특성을 제공하고, 장기 상관관계 모델링을 가능한 효과적으로 수행하도록 돕습니다. 연구에서는 TN이 동일한 신경망 아키텍처에 통합되어 정확도를 어떻게 성취하는지 설명하고 있으며, 라이트된 조건에서 더욱 개선된 성능을 제공합니다.

- **Performance Highlights**: TN 모듈의 통합으로 인해, 여러 데이터셋에 대한 교차 실험에서 평균 절대 오차(MAE)가 최대 94.2% 감소하는 성과를 올렸습니다. TN으로 훈련된 모델은 고도로 노이즈가 많은 데이터셋에서도 좋은 성과를 내며, MMPD 데이터셋을 활용한 경우 처음으로 임상 수준(MAE 2 이하)의 성과를 달성했습니다. 이를 통해 모델의 파라미터 수를 98.8% 감소시켜도 원본 모델보다 높은 성능을 유지할 수 있다는 점을 강조합니다.



### Feature-interactive Siamese graph encoder-based image analysis to predict STAS from histopathology images in lung cancer (https://arxiv.org/abs/2411.15274)
Comments:
          accept for publication in npj Precision Oncology

- **What's New**: 이 논문에서는 폐암의 확산 패턴인 공기 공간 확산(Spread through air spaces, STAS)을 분석하기 위한 새로운 이미지 분석 모델인 VERN을 소개합니다. 기존의 조직병리학적 방법들이 주관적이고 시간 소모적이며 오진이 발생할 가능성이 높아 대규모 적용에 제한이 있던 문제를 해결하고자 합니다. VERN은 폐암 조직병리학 이미지에서 STAS를 예측하기 위해 특성-상호작용(Siamese graph encoder)을 활용합니다.

- **Technical Details**: VERN 모델은 공간적(topological) 특성을 캡처하기 위해 특성 공유와 스킵 연결(skip connections)을 사용하여 모델 훈련을 개선합니다. 연구진은 1,546개의 조직병리학 슬라이드를 사용해 대규모 단일 집단(STAS lung cancer) 데이터셋을 구축하였습니다. VERN은 내부 검증에서 0.9215의 AUC 값을 달성하였고, 냉동(frozen) 및 파라핀(paraffin) 내장 테스트 섹션에서는 각각 0.8275 및 0.8829의 AUC를 기록하여 임상 수준의 성능을 입증하였습니다.

- **Performance Highlights**: VERN은 단일 집단 및 세 개의 외부 데이터셋에서 검증되었으며, 강력한 예측 성능과 일반화 가능성을 보여주었습니다. 이는 STAS 진단의 효율과 정확성을 향상시키기 위한 개방형 플랫폼(http URL)을 제공함으로써, 기존의 한계를 극복할 수 있는 가능성을 제시합니다.



### MambaIRv2: Attentive State Space Restoration (https://arxiv.org/abs/2411.15269)
Comments:
          Technical report

- **What's New**: MambaIRv2는 이미지 복원을 위한 새로운 접근 방식으로, Mamba 구조에 비대칭적 모델링 기능을 통합하여 이미지의 모든 픽셀을 효과적으로 활용할 수 있게 했다. 기존의 causal modeling 한계를 해결하고, 단일 스캔으로 이미지의 정보를 처리할 수 있도록 설계된 알아보기를 통해 픽셀 간의 상호작용을 증진시킨다. 또한, 시맨틱 가이드 이웃 메커니즘을 도입하여 유사한 픽셀 간의 상호작용을 촉진한다.

- **Technical Details**: MambaIRv2는 Attentive State-space Equation (ASE)와 Semantic Guided Neighboring (SGN)으로 구성되어 있다. ASE는 주어진 이미지의 픽셀 정보에 대해 사전학습된 세트를 통해 접근하여 필요한 픽셀을 쿼리하는 방법이다. SGN은 주어진 픽셀에 의미론적 라벨을 부여한 후, 이를 기반으로 비슷한 픽셀을 공간적으로 근접하게 재구성하여 효과적인 픽셀 간 상호작용을 유도한다.

- **Performance Highlights**: MambaIRv2는 Urban100 데이터세트에서 경량화된 SR 작업에 대해 SRFormer보다 0.35dB PSNR 성능을 개선하며, 9.3% 적은 파라미터로도 고성능을 자랑한다. 또한, 클래식 SR 작업에서도 HAT보다 0.29dB 성능을 개선하여 전체적으로 효과성과 효율성을 크게 향상시킨다.



### OSMamba: Omnidirectional Spectral Mamba with Dual-Domain Prior Generator for Exposure Correction (https://arxiv.org/abs/2411.15255)
- **What's New**: 이 논문에서는 노출 보정 (exposure correction)의 중요한 문제를 해결하기 위해 Omnidirectional Spectral Mamba (OSMamba)라는 새로운 네트워크를 제안합니다. OSMamba는 상태 공간 모델 (State Space Models)과 생성적 차분 모델 (generative diffusion models)의 장점을 결합하여 극단적인 노출 조건에서도 효과적으로 작동하도록 설계되었습니다. 특히, 이 연구는 주파수 도메인에서 장거리 의존성을 포착하기 위해 다방향 스펙트럼 스캐닝 메커니즘을 도입하였습니다.

- **Technical Details**: OSMamba는 대칭 및 연속성과 같은 주파수 스펙트럼의 고유 속성을 최대한 활용하지 못한 기존의 SSM 기반 네트워크의 한계점을 해결합니다. 이를 위해, 본 논문은 네 가지 연속 스캐닝 방법(행 스캐닝, 열 스캐닝, 양의 대각선 및 음의 대각선 스캐닝)을 포함하는 새로운 Omnidirectional Spectral Scanning(운영 능력이 향상된 스캔)을 도입합니다. 또한, 손실된 세부 정보를 복원하기 위해 강력한 생성 능력을 가진 차분 모델을 활용하여 이중 도메인 사전 생성기 (Dual-Domain Prior Generator)를 설계하였습니다.

- **Performance Highlights**: 다양한 노출 및 혼합 노출 데이터셋에서 실시된 실험들에서 OSMamba는 정량적 및 정성적으로 최첨단 성능을 달성했습니다. 이러한 성능 향상은 OSMamba가 제안한 새로운 스캐닝 메커니즘과 정보 전이 방식 덕분입니다. 논문은 제안된 방법의 우수성을 보여주는 다양한 결과를 제시합니다.



### Unsupervised Machine Learning for Osteoporosis Diagnosis Using Singh Index Clustering on Hip Radiographs (https://arxiv.org/abs/2411.15253)
- **What's New**: 이번 연구는 골다공증(Osteoporosis)의 진단을 자동화하기 위해 머신 러닝(machine learning) 알고리즘을 활용한 점에서 중요한 발전을 보여줍니다. 기존의 Singh Index (SI) 계산 방식은 시간 소모적이며 전문성이 요구되었으나, 새로운 접근법은 방사선 촬영에서 SI를 효율적으로 식별할 수 있게 합니다. 또한, 20세에서 70세까지의 인도 성인으로부터 수집된 838개의 방사선 이미지로 연구가 진행되었습니다.

- **Technical Details**: 연구에서는 사용자 정의 합성곱 신경망(convolutional neural network) 아키텍처를 개발하여 특징 추출을 진행했습니다. 성능 분석 결과, 기존에 사용되던 모델들보다 클러스터의 동질성과 이질성에서 우수한 성능을 나타냈습니다. 다양한 클러스터링 알고리즘(clustering algorithms)을 통해 이미지를 여섯 가지 SI 등급 클러스터로 분류했으며, 두 개의 클러스터가 높은 Silhouette Scores를 기록했습니다.

- **Performance Highlights**: 연구 결과는 데이터셋 불균형과 이미지 품질의 중요성을 강조하였습니다. 이를 통해 진단 정확도를 향상시키기 위해 환자의 임상 데이터나 참조 이미지를 추가하고, 이미지 전처리(pre-processing) 기술을 사용할 것을 제안합니다. 또한, 대규모 데이터셋에서의 레이블링(labeling) 문제를 완화하기 위해 반지도 학습(semi-supervised learning) 및 자가 지도 학습(self-supervised learning) 방법을 탐색할 필요성을 제기했습니다.



### Optimized Vessel Segmentation: A Structure-Agnostic Approach with Small Vessel Enhancement and Morphological Correction (https://arxiv.org/abs/2411.15251)
Comments:
          12 pages, 7 figurres, submitted to TIP

- **What's New**: 본 논문에서는 혈관 세분화의 정확성을 높이는 새로운 프레임워크인 OVS-Net을 제안합니다. 이 모델은 다양한 해부학적 구조에서 최적화된 세분화를 위해 설계된 구조 비지향적 접근 방식을 채택하고 있습니다. OVS-Net은 소규모 혈관 향상 및 형태학적 보정을 포함하여 다변량 혈관 세분화를 수행합니다.

- **Technical Details**: OVS-Net은 매크로 혈관 추출 모듈과 마이크로 혈관 향상 모듈로 구성된 이중 가지 특징 추출 모듈을 설계하였으며, 이는 세분화 마스크 예측 개선을 위해 디코더를 수정합니다. 이러한 설계를 통해 기존의 세분화 모델을 의료 현장에 더욱 적합하게 만들었습니다. 또한, 연결이 끊어진 혈관을 복구하는 토폴로지 복구 네트워크를 추가하여 임상에서의 활용 가능성을 높입니다.

- **Performance Highlights**: 17개의 데이터셋을 활용해 OVS-Net의 성능을 평가한 결과, 세분화 정확성과 일반화 능력이 우수하며, 연결성에서 34.6% 향상을 이뤘습니다. 이는 임상 응용 가능성을 강조하며, 향후 코드는 Github를 통해 공개 예정입니다.



### J-Invariant Volume Shuffle for Self-Supervised Cryo-Electron Tomogram Denoising on Single Noisy Volum (https://arxiv.org/abs/2411.15248)
Comments:
          10 pages, 7 figures, 7 tables

- **What's New**: 이번 연구에서는 Cryo-Electron Tomography (Cryo-ET) 볼륨 이미지를 단일 노이즈 볼륨을 사용하여 디노이징하는 새로운 자가 지도 학습(Self-Supervised Learning) 모델을 제안합니다. U자형 J-invariant 블라인드 스팟 네트워크와 희소 중심 마스킹 컨볼루션(Sparse Centrally Masked Convolutions) 및 확장된 채널 주의 블록(Dilated Channel Attention Blocks)을 활용하여 노이즈 제거 성능을 크게 향상시키는 방식을 사용했습니다. 이 방법은 고유한 볼륨 언셰플(VU) 및 셔플 기술을 적용하여 수신 필드를 확장하고, 구조를 보존하는 데 있어 개선된 성능을 제공합니다.

- **Technical Details**: 제안된 모델은 U자형 구조를 사용하여 장거리 의존성(long-range dependencies) 및 세부 구조를 잘 포착할 수 있습니다. 특히, 희소 중심 마스킹 컨볼루션과 채널 주의 블록을 결합하여 Cryo-ET 볼륨 이미지에서 노이즈를 효과적으로 제거할 수 있는 방법입니다. 이 모델은 전통적인 다운샘플링 방식에서 발생하는 정보 손실 없이 J-invariance 속성을 유지할 수 있는 볼륨 언셰플 및 셔플 기술을 도입했습니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존의 자가 지도 디노이징 방법들보다 우수한 성능을 보이며, Cryo-ET 데이터 처리 분야에서 중요한 진전을 이루었습니다. 제안된 모델은 단일 노이즈 볼륨을기반으로 하여 Noise Reduction과 Structure Preservation 두 가지 측면에서 뛰어난 성능을 발휘합니다. 이는 구조 생물학 연구에 필요한 Cryo-ET 데이터의 품질 개선에 기여할 것으로 기대됩니다.



### Bio-inspired AI: Integrating Biological Complexity into Artificial Intelligenc (https://arxiv.org/abs/2411.15243)
- **What's New**: 이 논문은 인공지능(AI)의 발전이 생물학적 계산에서의 기본 원칙을 어떻게 응용할 수 있는지 탐구합니다. 특히, 맥락에 따라 달라지는 계층적 정보 처리(hierarchical information processing), 시행착오 발견(trial-and-error heuristics), 그리고 다중 스케일 조직(multi-scale organization)의 중요성을 강조합니다. 생물학적 지능의 미세한 메커니즘을 조사하여 인공지능 시스템 설계에서의 가능성과 한계를 조명하고자 합니다. 결과적으로 생물학적 시스템에서 영감을 받은 더 적응적이고 강력한 AI 시스템을 설계하는 프레임워크를 제안합니다.

- **Technical Details**: AI의 역사적 맥락에서 저자들은 토마스 홉스와 블레즈 파스칼의 기계적 사유 이론(mechanical theory)과 같은 초기 사상가들이 지능형 기계의 꿈을 어떻게 펼쳤는지를 분석합니다. 생물학적 계산 원칙에 따르면, 생물학적 지능은 신경계에 국한되지 않고 단세포 생물이나 식물에서도 정보 처리 및 적응 행동을 보여 줍니다. 자연에서 발견되는 지능은 계층적 상호작용과 다중 스케일에서의 과정들로 형성되며, 이는 AI 시스템 설계에서 중요한 참고자료가 됩니다.

- **Performance Highlights**: 현재 AI 시스템 개발에서 생물학적 영감을 받아 RoboBee와 RoboRay 프로젝트와 같은 새로운 접근 방식이 주목받고 있습니다. 이러한 시스템들은 경량화와 유연성을 추구하며, 뇌와 유사한 에너지 효율(enery efficiency)을 가지려는 뉴로모픽 컴퓨팅(neuromorphic computing) 기술에 의존합니다. 그러나 여전히 많은 생물학적 문제 해결은 신경망을 기반으로 한 접근 방식이 아닌 다른 생물학적 메커니즘에서도 발생하고 있습니다. 이 논문은 그러한 다양한 생물학적 전략들이 AI 설계에 더 많은 통찰력을 제공할 수 있음을 강조합니다.



### CODE-CL: COnceptor-Based Gradient Projection for DEep Continual Learning (https://arxiv.org/abs/2411.15235)
Comments:
          10 pages, 2 figures

- **What's New**: 최근 연구에서는 인공 신경망이 연속적 학습을 할 때 발생하는 재난적 망각(catastrophic forgetting) 문제를 해결하기 위해, CODE-CL이라는 새로운 방법을 소개했습니다. 이 방법은 신경과학에서 영감을 받은 개념자(matrix) 표현을 활용하여, 과거 태스크에 대한 중요성을 코드화하고 새 정보를 유연하게 통합할 수 있게 합니다. 이를 통해 상관관계가 높은 태스크간의 효율적인 정보 이전이 가능하고, 기존 지식을 상당히 방해하지 않고도 새로운 지식 습득이 가능합니다.

- **Technical Details**: CODE-CL은 입력 공간에서 과거 태스크의 방향성을 인코딩하여 $1-S$의 방향으로 새 지식을 통합할 수 있도록 합니다. 여기서 $S$는 이전 태스크와 관련된 방향의 중요성을 나타냅니다. 또한, 개념자 기반 표현을 사용하여 태스크 간의 겹침을 분석하고, 이는 상위 공간에서의 축척 투영을 통해 효율적인 지식 전달을 가능하게 합니다. 이러한 접근 방식은 이전의 중요한 지식을 손상시키지 않고 상관관계가 높은 태스크들 간의 학습을 가능하게 합니다.

- **Performance Highlights**: CODE-CL은 Permuted MNIST, Split CIFAR100, Split miniImageNet, 5-Datasets와 같은 연속 학습 이미지 분류 벤치마크에서 광범위한 실험을 진행하여 효과성을 입증했습니다. 실험 결과, CODE-CL은 많은 최첨단 방법들보다 우수한 성능을 보여주며, 기억 저하를 최소화하면서도 뛰어난 성과를 달성했습니다. 이 연구는 메모리 효율적이며 과거의 지식을 유지하면서도 새로운 정보를 유연하게 습득할 수 있는 접근 방식을 제시합니다.



### Learning Volumetric Neural Deformable Models to Recover 3D Regional Heart Wall Motion from Multi-Planar Tagged MRI (https://arxiv.org/abs/2411.15233)
- **What's New**: 이번 연구에서 우리는 심장 벽의 동적 움직임을 평가하기 위한 새로운 모델인 볼륨 신경 변형 모델(Volumetric Neural Deformable Models, $
u$NDMs)을 제안합니다. 이 모델은 2D 표면에서 관찰된 겉모습의 운동 신호를 바탕으로 3D 진정한 운동을 복원하는 데 중점을 두고 있습니다. 특히, 기존 접근 방식의 한계를 극복하기 위해 하이브리드 포인트 트랜스포머(Point Transformer) 아키텍처를 결합하여 3D 운동 추정을 향상시켰습니다.

- **Technical Details**: 제안된 $
u$NDMs는 전역 변형 파라미터 함수와 지오메트릭 포인트 흐름(regulated point flow) 같은 로컬 변형 필드를 통해 심장 벽의 기하학적 형태와 운동 역학을 표현합니다. 애플리케이션에서 포인트 크로스-어텐션(point cross-attention)과 셀프-어텐션(self-attention) 메커니즘을 활용하여, 다차원 공간에서의 겉모습의 운동 신호를 정확히 합쳐 3D 진정한 운동 정보를 도출할 수 있습니다. 이 접근 방식은 측정된 심근 벽의 운동 데이터로부터 심장 움직임을 예측하는 데 필요한 필요한 데이터를 생성하고 학습하는 데 도움을 줍니다.

- **Performance Highlights**: 대규모 합성 3D 지역 심장 벽 운동 데이터셋에서 실험한 결과, 제안하는 방법은 밀도가 낮은 2D 겉모습 모션 신호로부터 고밀도의 3D 진정한 운동을 정확하게 회복하는 데 있어 뛰어난 성과를 보였습니다. 특히, 기존의 시간 소모적인 최적화 방법 대비 효과적이며, 통계적으로 유의미한 회복 정확도를 기록했습니다. 이러한 결과는 향후 보다 복잡한 심장 MRI 애플리케이션으로의 일반화 또한 가능하다는 것을 시사합니다.



### LPLgrad: Optimizing Active Learning Through Gradient Norm Sample Selection and Auxiliary Model Training (https://arxiv.org/abs/2411.15217)
- **What's New**: 이 논문은 Loss Prediction Loss with Gradient Norm (LPLgrad)라는 새로운 능동 학습(active learning) 방법을 제안합니다. LPLgrad는 모델의 불확실성을 효과적으로 정량화하고 이미지 분류 작업의 정확성을 향상시키기 위해 설계되었습니다. 이전의 방법들이 라벨된 세트에서의 훈련과 새로운 비라벨 샘플 쿼리의 핵심 정보 활용에 부족했던 문제를 해결하고자 합니다.

- **Technical Details**: LPLgrad는 두 가지 주요 단계로 구성됩니다: (i) 훈련 단계에서는 주 모델과 보조 모델이 함께 훈련되어 입력 특성에 대한 손실을 예측합니다. 이 과정에서 라벨된 데이터를 최대한 효율적으로 활용하여 원래 학습 프로세스에서 간과된 부분을 보완합니다; (ii) 쿼리 단계에서는 비라벨 데이터셋의 엔트로피 값의 그래디언트 노름을 계산하여 주 모델의 불확실성을 정량화하고 레이블링할 샘플을 선택합니다.

- **Performance Highlights**: 실제 데이터셋에서의 광범위한 평가 결과, LPLgrad 접근법은 소수의 라벨된 이미지에서 정확도 면에서 최신의 능동 학습 방법들을 압도적으로 초월하는 성능을 보였습니다. 또한, 여러 이미지 분류 작업에서 훈련 및 쿼리 시간이 유사하게 유지되면서도 더 나은 성능을 달성했습니다.



### LightLLM: A Versatile Large Language Model for Predictive Light Sensing (https://arxiv.org/abs/2411.15211)
Comments:
          15 pages, 14 figures, 5 tables

- **What's New**: 이번 논문은 LightLLM이라는 모델을 제안하며, 이는 전이 학습된 대형 언어 모델(LLM)을 가볍게 미세 조정하여 센서 기반의 조명 감지 작업을 수행합니다. LightLLM은 센서 데이터 인코더, 상황적 프롬프트 및 융합 레이어를 통합하여 입력을 통합된 표현으로 결합하고 있습니다. 이를 통해 모델은 원래 파라미터를 변경하지 않고도 새로운 작업에 잘 적응할 수 있도록 설계되었습니다.

- **Technical Details**: LightLLM은 다양한 데이터를 처리하기 위해 세 가지 주요 구성 요소를 통합합니다: 작업에 특화된 센서 데이터 인코더, 상황 인식을 위한 자연어 프롬프트, 센서 데이터와 프롬프트를 결합하는 융합 레이어입니다. LoRA(저랭크 적응) 기법을 사용하여 LLM을 효율적으로 미세 조정하며, 이는 기존 LLM의 지식을 활용하면서도 도메인 특화된 데이터를 포함할 수 있게 합니다.

- **Performance Highlights**: LightLLM은 세 가지 조명 감지 작업인 실내 위치 추적, 외부 태양광 예측 및 실내 태양광 추정에 대해 실제 실험 데이터를 사용하여 검증되었으며, 기존 최신 방법들보다 현저하게 우수한 성능을 보였습니다. 예를 들어, 실내 위치 추적 정확도가 4.4배, 실내 태양광 추정에서 3.4배의 개선을 이루었습니다. 또한 LightLLM은 ChatGPT-4보다 직접 프롬프트를 통해 더 나은 성능을 나타내, 센서 데이터 융합을 위한 전문화된 아키텍처의 장점을 부각시켰습니다.



### Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks (https://arxiv.org/abs/2411.15210)
- **What's New**: 이 논문은 이미지 분류 모델에 대한 새로운 개별 공격 방식인 Probability Margin Attack (PMA)을 제안합니다. PMA는 로짓(logits) 공간이 아닌 확률 공간에서 적대적 마진(adversarial margin)을 정의하여 기존의 공격 방식보다 효과적인 평가를 가능하게 합니다. 또한, CC3M 데이터셋에서 유래한 백만 규모의 데이터셋 CC1M을 구축하여 대규모 안정성 평가를 진행했습니다.

- **Technical Details**: PMA는 확률 마진 손실(probability margin loss)을 도입하여 공격의 효율성을 높이는데 집중합니다. 논문에서는 PMA와 기존의 cross-entropy, margin 손실 간의 관계를 분석하며, PMA가 현재의 개별 공격 방식보다 우수하다고 입증합니다. 또한, PMA를 기반으로 한 두 가지 앙상블 공격 방법도 제안하여 효율성과 효과성의 균형을 맞췄습니다.

- **Performance Highlights**: PMA는 CIFAR-10, CIFAR-100, ImageNet-1K 등 여러 데이터셋에서 기존의 개별 공격 방식보다 뛰어난 성능을 보였습니다. 또한, 백만 규모의 평가 데이터셋 CC1M을 활용한 실험에서는 기존의 소규모 평가 방법과 비교할 때 큰 안정성 차이를 발견하였습니다. 이러한 결과는 개별 공격과 앙상블 공격 간의 효과의 차이를 보여줍니다.



### Label Distribution Shift-Aware Prediction Refinement for Test-Time Adaptation (https://arxiv.org/abs/2411.15204)
- **What's New**: 이번 연구에서 제안된 DART 방법은 테스트 시 레이블 분포 변화에 효과적으로 대응할 수 있는 새로운 TTA 방법이다. 기존의 TTA 방법들은 레이블 분포가 변화할 경우 성능 저하를 겪는 경향이 있었으나, DART는 이러한 문제를 해결하는 데 중점을 두고 있다.

- **Technical Details**: DART는 클래스 간 혼동 패턴을 활용해 BNAdapt의 예측을 개선하는 방식으로 작동한다. 이 방법은 훈련 데이터셋에서 다양한 클래스 분포에 노출되어 예측 정제 모듈을 학습하고, 이후 테스트 데이터의 레이블 분포 변화를 감지한다.

- **Performance Highlights**: CIFAR-10C와 같은 다양한 벤치마크에서 DART는 5-18%의 정확도 향상을 보여주며, 기존의 TTA 방법들의 성능 또한 크게 개선되었다. 특히, DART는 BNAdapt와 결합하여 성능을 극대화할 수 있는 유용한 플러그인 도구로 자리잡았다.



New uploads on arXiv(cs.AI)

### Object-centric proto-symbolic behavioural reasoning from pixels (https://arxiv.org/abs/2411.17438)
- **What's New**: 이 연구에서는 자율 지능형 에이전트가 저수준 감지 입력과 모터 명령부터 고수준 추상적 추론 및 계획에 이르는 다양한 수준에서의 계산적 도전을 극복할 필요가 있음을 강조합니다. 기존의 감독 없이도 객체 중심(object-centric) 표현을 통해 세계를 이해하고 조작하는 새로운 뇌 영감을 받은(deep-learning) 아키텍처를 제안합니다. 이는 비용이 많이 드는 데이터 주석 없이도 다양한 수준의 정보를 학습할 수 있게 해줍니다.

- **Technical Details**: 제안된 아키텍처는 픽셀에서 학습하여 자기 환경을 해석하고 제어하며 추론할 수 있는 능력을 강화합니다. 에이전트는 논리적 추론(logical reasoning)과 연속 제어(continuous control)를 결합해야 하는 작업을 통해 그 유용성을 증명합니다. 에이전트는 emergent conditional behavioural reasoning과 같은 복잡한 논리적 연산을 학습하고자 하며, 이를 통해 환경을 조종하여 목표를 달성할 수 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 아키텍처는 에이전트가 환경의 예기치 않은 변화에 온라인으로 적응할 수 있으며, 상호작용하는 환경에서의 행동 추론(behavioral reasoning)이 가능하다는 것을 보여줍니다. 이는 에이전트가 동적 목표 생성(dynamically desired goal generation)을 통해 내장된 모델의 약간의 위반에도 강인성을 유지함을 의미합니다. 비록 결과가 실제 세계의 복잡성에는 미치지 않지만, 제안된 접근 방식은 비지도 학습을 위한 중요한 유도 편향을 통해 객체 중심 표현을 조작하는 방법을 제시합니다.



### Advancing Uncertain Combinatorics through Graphization, Hyperization, and Uncertainization: Fuzzy, Neutrosophic, Soft, Rough, and Beyond (https://arxiv.org/abs/2411.17411)
Comments:
          255 pages. 11 figures. Published as a book in 2024. Publisher: Biblio Publishing. ISBN: 978-1-59973-812-3

- **What's New**: 이번 논문에서는 불확실성을 처리하기 위해 퍼지 집합(fuzzy sets), 중립 집합(neutrosophic sets), 거친 집합(rough sets), 부드러운 집합(soft sets)과 같은 새로운 개념들을 소개하고 있습니다. 특히 중립 집합은 진리, 불확실성, 거짓을 동시에 표현할 수 있어 복잡한 시스템의 불확실성을 모델링하는 데 유용한 도구로 자리잡고 있습니다. 이 연구는 또한 그래프화된 형태의 집합 개념을 확대하여 하이퍼그래프(hypergraphs) 및 슈퍼하이퍼그래프(superhypergraphs)와 같은 구조를 포함합니다.

- **Technical Details**: 하이퍼개념(hyperconcepts) 및 슈퍼하이퍼개념(superhyperconcepts)의 연구가 그래프 이론(graph theory)을 넘어 활발히 진행되고 있으며, 이는 조합론(combinatorics) 및 불확실한 집합에 관한 다양한 분야와 연결되어 있습니다. 이 논문에서는 Neutrosophic Oversets, Neutrosophic Undersets, Neutrosophic Offsets 및 비표준 실수 집합(Nonstandard Real Set)과 같이 여러 그래프 개념의 확장을 다룹니다. 이러한 새로운 개념들은 연구자들에게 영감을 주고, 실질적인 수학적 및 응용적 의미를 가질 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 논문은 최근의 발견들을 정리하고 조망하는 리소스를 제공하는 것을 목표로 하며, 독자들에게 중요한 정보와 영감을 줄 수 있을 것으로 기대됩니다. 다양한 개념의 정의를 통해 새로운 아이디어를 제안하고, 연구자들의 학문적 활동에 기여하고자 하는 방향성을 가지고 있습니다. 결과적으로, 제안된 하이퍼 및 슈퍼하이퍼 개념들은 수학적 연구 및 실제 적용에 큰 영향을 미칠 가능성이 높습니다.



### BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving (https://arxiv.org/abs/2411.17404)
- **What's New**: 본 논문에서는 기존의 Open-source Operations Research 데이터셋의 한계를 극복하기 위해 StructuredOR 데이터셋을 소개합니다. 이 데이터셋은 전반적인 수학적 모델링 과정을 포착하는 포괄적인 레이블이 주어져 있으며, 이는 Reinforcement Learning(RL) 응용에 있어 새로운 가능성을 열어줍니다. 또한, BPP-Search 알고리즘을 제안하여, Beam search와 Process Reward Model, pairwise Preference 알고리즘을 통합하여 수학적 모델링에서의 추론 과정을 향상시킵니다.

- **Technical Details**: BPP-Search는 Tree-of-Thought 구조에서 효과적으로 분야를 탐색하면서도 검색의 정확성을 높이는 것을 목표로 합니다. 이 방법은 Greedy나 Beam Search와 결합한 기존의 Process Reward Model 보다 우수한 성능을 보여 줍니다. StructuredOR 데이터셋은 다양한 수학적 모델링 문제에 대한 평가뿐만 아니라, 세부 모델링 주석을 제공하여 RL 기반의 방법론에 널리 적용될 수 있는 기반을 마련합니다.

- **Performance Highlights**: BPP-Search는 StructuredOR, NL4OPT 및 MAMO-ComplexLP 데이터셋에서의 대규모 실험을 통해 기존의 최첨단 방법을 능가하는 성능을 보여주었습니다. 특히, Tree-based reasoning에서 BPP-Search는 높은 정확도와 효율성을 자랑하며, 정답을 신속히 회수할 수 있는 능력을 보여줍니다. 이러한 결과는 BPP-Search가 수학적 모델링 작업에서 고려해야 할 중요한 개선 사항으로 자리잡을 수 있음을 시사합니다.



### Towards Intention Recognition for Robotic Assistants Through Online POMDP Planning (https://arxiv.org/abs/2411.17326)
Comments:
          Presented at the ICAPS 2023 workshop "PAIR: Plan, Activity, and Intent Recognition"

- **What's New**: 이 논문은 로봇 어시스턴트가 인간 작업자를 지원하는 상황에서의 의도 인식을 다룹니다. 특히, 로봇이 불완전한 정보와 혼란스러운 환경 속에서 작업자의 활동을 이해하고 그에 맞춰 적절한 행동을 취하는 방법을 제안합니다. 이를 위해 저자들은 비가시적 부분을 포함한 부분 관찰 가능 마르코프 결정 프로세스(POMDP) 모델을 제안하며, 온라인 계획 프레임워크를 사용합니다.

- **Technical Details**: 저자들은 능동적 목표 인식(Active Goal Recognition, AGR)이라는 접근 방식을 통해 로봇이 환경과 상호작용하며 작업자의 골이나 의도를 파악하는 방법을 제시합니다. 이 연구는 목표 인식과 관련성 추정(relevance estimation) 사이의 상호작용이 보다 나은 자율성과 행동 선택에 영향을 줄 수 있음을 강조합니다. 또한, 저자들은 이 문제들을 해결하기 위해 온라인 POMDP 계획 방식을 활용하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론은 기존의 POMCP(planner based on Monte-Carlo Tree Search)와 비교했을 때 성능 향상을 보였습니다. 이 연구는 로봇이 작업자의 행동을 관찰하고 그에 맞춰 적절히 응답하는 방식으로, 더욱 향상된 에이전트의 자율성이 의도 인식 작업에서 성과를 높일 수 있음을 보여줍니다. 향후 AGR 문제에 대한 개선 가능성도 열려있어 추가 연구가 기대됩니다.



### LLM-Based Offline Learning for Embodied Agents via Consistency-Guided Reward Ensemb (https://arxiv.org/abs/2411.17135)
Comments:
          Findings of EMNLP-2024 Camera Ready Version

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)을 에이전트로 직접 사용하는 대신, 그들을 도구로 활용하여 구현된 에이전트를 학습시키는 새로운 접근법을 제안합니다. CoREN이라는 일관성 유도 보상 앙상블 프레임워크(ensemble framework)를 통해 LLM이 생성한 보상을 특정 환경 도메인에 기반을 두고 유도함으로써, 오프라인 강화 학습(offline reinforcement learning)에서의 한계를 극복하고자 합니다. 이는 에이전트가 다른 환경 도메인에서 효율적으로 오프라인 학습을 할 수 있도록 합니다.

- **Technical Details**: CoREN은 두 단계의 보상 추정 과정을 채택합니다. 첫 번째 단계에서는 LLM에 여러 종류의 보상을 추정하도록 쿼리하며, 각 보상은 특정 시공간 일관성(spatio-temporal consistency) 기준을 고려하여 일관적이고 도메인 기반의 보상을 생성합니다. 두 번째 단계에서는 이러한 보상들을 조화시켜 주어진 경로의 희소 보상(sparse rewards)과 정렬된 도메인 특정 보상으로 통합합니다. 이를 통해 오프라인 RL을 통한 에이전트가 최소한의 지연(latency)으로 고효율의 작업 수행이 가능해집니다.

- **Performance Highlights**: VirtualHome 벤치마크에서 CoREN은 다른 오프라인 강화 학습 에이전트들에 비해 현저한 성능 향상을 보였으며, 8B 파라미터를 가진 최신 LLM 기반 에이전트와 비교해도 유사한 성능을 달성했습니다. CoREN의 파라미터 수는 117M에 불과하지만, LLM을 교육 목적으로만 사용함으로써, 오프라인 설정에서도 뛰어난 성능을 보여줍니다. 이러한 결과는 CoREN의 효용성과 적용 가능성을 강력하게 뒷받침합니다.



### Boundless Socratic Learning with Language Games (https://arxiv.org/abs/2411.16905)
- **What's New**: 본 논문에서는 폐쇄 시스템 내에서 에이전트가 원하는 능력을 마스터하기 위한 세 가지 조건을 제시하고 이를 정당화합니다. 첫 번째 조건은 충분히 정보적이고 정렬된 피드백을 받는 것이며, 두 번째는 경험/데이터의 폭이 넓어야 한다는 점입니다. 마지막으로, 충분한 용량과 자원이 필요합니다. 이 조건들이 충족될 경우, 에이전트는 스스로 학습하여 성능을 개선할 수 있는 가능성이 높아집니다.

- **Technical Details**: 자기 개선(self-improvement)은 에이전트의 출력이 미래 학습에 영향을 미치는 과정을 의미합니다. 이러한 과정은 강화 학습(reinforcement learning)과 관련이 깊으며, 시스템 내에서 에이전트가 수행하는 행동에 의해 학습 데이터 분포가 변화하는 구조를 형성합니다. 특별히 제안된 '소크라틱 학습(Socratic learning)'은 에이전트가 입력과 출력을 서로 맞추어 지속적으로 성능을 향상시키는 방식입니다.

- **Performance Highlights**: 이 연구는 순수한 자기 개선을 통해 초기 데이터나 지식 이상으로 성능을 극대화할 수 있음을 보여줍니다. 이를 위한 구체적인 구현 프레임워크는 언어 게임(language games)이라는 개념을 기반으로 할 수 있습니다. 상호 작용하는 도구를 통해 에이전트는 피드백을 받고 이를 학습하여 시간에 따라 점진적으로 성능을 개선할 수 있습니다.



### Human Motion Instruction Tuning (https://arxiv.org/abs/2411.16805)
- **What's New**: 이 논문은 LLaMo (Large Language and Human Motion Assistant)라는 다중 모달( multimodal ) 프레임워크를 소개합니다. 기존의 instruction-tuning 접근법과 달리, LLaMo는 비디오나 모션 시퀀스와 같은 비언어적(non-linguistic) 입력을 언어 토큰으로 변환하는 대신, 모션을 원래 형태로 유지하여 instruction tuning을 진행합니다. 이러한 방식은 토크나이제이션(tokenization)에서 종종 약해지는 모션 특정 세부정보를 보존하여, 복잡한 인간 행동을 해석하는 모델의 능력을 향상시킵니다.

- **Technical Details**: LLaMo는 비디오, 모션 데이터 및 텍스트 입력을 함께 처리하여 유연하고 인간 중심적인 분석을 가능하게 합니다. 또한 논문에서는 인간 행동, 전문 활동 등 고복잡성(domain-specific) 분야에서 LLaMo의 효과를 실험 평가하였으며, 이 모델이 특정 지식을 잘 포착함을 보여주었습니다. 그 결과, LLaMo는 모션 집약적인 시나리오에서 이해력과 예측력을 높이는 데 기여합니다.

- **Performance Highlights**: 이 논문은 LLaMo가 스포츠 분석(sports analytics)에서 행동 예측(behavioral prediction)까지 다양한 응용 프로그램에 활용될 수 있는 미래의 다중 모달 AI 시스템의 토대를 제공하기를 기대합니다. 실험 결과는 LLaMo가 복잡한 인간 행동을 정확하게 이해하고 예측할 수 있는 능력을 갖추고 있음을 보여줍니다. 프로젝트 웹사이트에서 코드와 모델을 이용할 수 있습니다.



### A Brief Summary of Explanatory Virtues (https://arxiv.org/abs/2411.16709)
Comments:
          10 pages, 2 tables

- **What's New**: 기술적 설명의 미덕(Explanatory Virtues, EVs)에 대한 철학, 심리학 및 인지 과학 문헌을 종합하여 설명합니다. 이 보고서에서는 Keas(2018)의 분류 체계를 따르며, 증거적(evidential), 일관성(coherential), 미적(aesthetic), 그리고 시간적(diachronic) 등의 네 가지 주요 유형을 도출합니다. 추가적으로, 이론의 넓은 적용 가능성을 고려하는 범주(coverage)를 포함하였습니다.

- **Technical Details**: EV는 귀납적 추론(abductive reasoning)에서 중요하며, 설명은 관찰을 정당화하는 이론으로 설명됩니다. 이론에서의 EV 차원은 인식론적(epistemic) 및 실용적(pragmatic) 차원으로 나뉘며, 각각의 차원은 내재적 일관성(internal coherence) 및 경험적 강도(empirical strength) 등 여러 요소로 구성됩니다. 이러한 개념은 기계 학습(Machine Learning, ML) 모델의 전역(global) 및 국소(local) 설명과도 연결됩니다.

- **Performance Highlights**: EVs는 이론이 관찰을 얼마나 잘 설명하는지를 측정하는 여러 측면을 나타냅니다. 각 EV는 데이터를 설명하는데 기여하며, 그들 간의 상관관계를 통해 이론의 유용성을 판단할 수 있습니다. 또한, 이론의 일관성, 유사성, 그리고 기존 신념들과의 호환성 등 다양한 요소들이 EV에 의해 평가됩니다.



### StableAnimator: High-Quality Identity-Preserving Human Image Animation (https://arxiv.org/abs/2411.17697)
- **What's New**: StableAnimator는 첫 번째 ID 보존 비디오 확산 프레임워크로, 참조 이미지와 포즈 시퀀스를 기반으로 고품질 비디오를 생성하는 데 중점을 두고 있습니다. 이는 기존의 성능 저하 문제를 해결하고 얼굴 품질을 향상시킬 수 있는 혁신적인 방법으로, Hamilton-Jacobi-Bellman (HJB) 방정식을 활용하여 최적의 ID 보존을 달성합니다. 이 방식은 비디오의 충실도를 저해하지 않고 정체성을 유지하는 데 기여합니다.

- **Technical Details**: StableAnimator는 전통적인 비디오 확산 모델에서 ID 정보를 성공적으로 통합하기 위해 설계된 모듈을 포함합니다. 특히, 글로벌 콘텐츠 인식 Face Encoder와 분포 인식 ID Adapter를 도입하여 비디오 생성 과정에서 공간적 분포와 ID 정보를 동시적으로 고려합니다. HJB 방정식을 활용해 각 디노이징 단계에서 잠재 변수를 업데이트함으로써 얼굴 품질을 향상시키고, 후처리 도구에 대한 의존성을 줄입니다.

- **Performance Highlights**: 실험을 통해 StableAnimator는 CSIM 지표에서 ControlNeXt에 비해 47.1% 뛰어난 성능을 보이며, 다양한 벤치마크 데이터셋에서 최상의 결과를 기록하였습니다. 특히, 얼굴과 몸의 왜곡 문제가 두드러지는 기존의 모델들과 달리, StableAnimator는 포즈에 기반한 실제적인 애니메이션을 생성하며 ID 일관성을 유지하는 데 성공하였습니다.



### RealSeal: Revolutionizing Media Authentication with Real-Time Realism Scoring (https://arxiv.org/abs/2411.17684)
Comments:
          Best Paper Award, Blue Sky Track at 26th ACM International Conference on Multimodal Interaction, Nov 2024, San Jose, Costa Rica

- **What's New**: 이 논문은 기존의 미디어 인증 방식을 근본적으로 재고하는 혁신적인 접근 방식을 제안합니다. 딥페이크와 조작된 미디어의 증가하는 위협에 대처하기 위해, 기존의 합성 데이터에 워터마크를 부착하는 방법 대신, 실제 콘텐츠의 출처에서 워터마크를 부착하자는 새로운 패러다임을 마련했습니다. 이를 통해 생성된 신뢰성 점수를 이미지 메타데이터에 내장하여 이미지의 신뢰도를 전환적으로 변화시키는 방안을 모색하고 있습니다.

- **Technical Details**: 제안된 접근법은 다양한 감각 입력과 기계 학습을 활용하여 콘텐츠의 현실성을 실시간으로 평가합니다. 이 과정은 감지(Sensing), 점수화(Scoring), 서명(Signing)의 세 가지 단계로 구성됩니다. 데이터가 캡처되고 처리된 후, 메타데이터에 현실성 점수가 내장된 이미지를 암호적으로 서명하여 데이터의 무결성과 진정성을 보장합니다.

- **Performance Highlights**: 이 기술은 특히 디지털 콘텐츠의 신뢰성이 중요한 분야에서 큰 변화를 가져올 수 있습니다. 또한, 이미지의 진정성을 확인할 수 있는 새로운 표준을 설정하여 플랫폼과 산업 전반에 걸쳐 신뢰성을 높이는 데 기여할 것입니다. 이러한 접근은 진짜 미디어와 가짜 미디어를 효과적으로 구분할 수 있는 혁신적인 솔루션으로 평가받고 있습니다.



### Explainable AI for Classifying UTI Risk Groups Using a Real-World Linked EHR and Pathology Lab Datas (https://arxiv.org/abs/2411.17645)
- **What's New**: 이 논문은 약 100만 명의 비식별화된 개인 데이터를 포함한 연결된 전자 건강 기록(EHR) 데이터를 활용하여 요로 감염(UTI)을 특징짓고, 데이터 품질, 공정성(fairness), 투명성(transparency)에 중점을 둔 예측 모델을 개발했습니다. 특히, EHR 데이터의 비동질성(heterogeneity)과 희소성(sparsity)을 극복하기 위해 엄격한 데이터 전처리 및 관리 방법이 필요함을 강조합니다. 기존의 자료를 바탕으로, AI 기반의 위험 추정 프레임워크를 도입하여 각 환자의 타임라인에서 UTI 위험을 추정합니다.

- **Technical Details**: 연구에서는 Bristol, North Somerset, South Gloucestershire 지역의 연결된 EHR 데이터를 사용하여 주요 리스크 요인을 분석하고 예측 모델을 구축합니다. 이 과정에서 XGBoost 모델을 사용해 UTI 위험 카테고리를 구분하고, 설명 가능한 AI(explainable AI) 기법을 통해 주요 예측 변수를 식별하여 해석 가능성을 유지합니다. 데이터 전처리 과정에는 다양한 데이터 유형의 적절한 정리와 통합이 포함되어 있으며, 이는 머신러닝 모델링을 위한 데이터를 구조화된 형식으로 변환하는 데 필수적입니다.

- **Performance Highlights**: 연구 결과는 UTI 위험 그룹 간의 임상 및 인구 통계적 요인의 차이를 밝혀내며, 이를 통해 UTI 위험 계층화(risk stratification)와 진행 과정을 이해하는 데 기여합니다. AI를 적용한 분석 결과는 UTI 관련 임상 의사결정에 추가 가치를 제공하며, 해석 가능성, 투명성 및 공정성을 최우선으로 고려하고 있습니다. 이러한 접근 방식은 결국 건강 결과 향상에 기여하는 데이터 관리의 중요성을 강조합니다.



### MALMM: Multi-Agent Large Language Models for Zero-Shot Robotics Manipulation (https://arxiv.org/abs/2411.17636)
Comments:
          48 pages

- **What's New**: 이번 연구에서는 다중 에이전트를 활용한 대규모 언어 모델(Multi-Agent Large Language Model, MALMM)을 제안합니다. 이 모델은 고수준(High-level) 계획과 저수준(Low-level) 제어 코드 생성을 전문화된 LLM 에이전트 간에 분산시킵니다. 또한, 추가 에이전트가 동적으로 전이(Transition)를 관리하여 실시간 피드백을 가능하게 합니다.

- **Technical Details**: MALMM 프레임워크는 매 단계마다 환경에서 관찰한 데이터를 통합하여 중간 실패를 효과적으로 처리하고 적응형 재계획(Adaptive Re-planning)을 수행합니다. 이를 통해 기존의 방법들과는 달리 사전 훈련된 기술 정책(Pre-trained skill policies)이나 예시를 필요로 하지 않습니다. 여기에 LLM 에이전트는 장기 과제(Long-horizon tasks)에서도 사용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구는 아홉 개의 RLBench 과제를 사용하여 MALMM의 성과를 평가하였으며, 영점 본 세팅(Zero-shot setting)에서도 로봇 조작(Robotics Manipulation) 문제를 해결할 수 있는 능력을 입증하였습니다. 이로 인해 기존의 LLM 기반 조작 방법들의 주요 한계를 극복하였습니다.



### Learning Chemical Reaction Representation with Reactant-Product Alignmen (https://arxiv.org/abs/2411.17629)
- **What's New**: 이 논문에서 소개하는 {\

- **Technical Details**: 제안된 모델의 이름은 RAlign이며, 유기 반응 관련 작업에 특화된 화학 반응 표현 학습 모델입니다. 이 모델은 반응물과 생성물 간의 원자 일치를 통합하여 화학 반응의 과정을 모델링하고, 반응 조건을 통합하기 위해 어댑터 구조를 설계하여 다양한 작업에 적응할 수 있도록 하였습니다. 또한, 중요한 기능 그룹에 집중할 수 있는 반응 중심 주의 메커니즘을 도입하여 화학 반응의 강력한 표현을 생성할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과, 이 모델은 반응 조건 예측, 반응 수율 예측 및 선택성 예측과 같은 다양한 작업에서 기존의 화학 반응 표현 학습 아키텍처를 능가하는 성능을 보였습니다. 특히, USPTO_CONDITION 데이터셋에서 반응 조건 예측 작업에 대해 가장 강력한 기준선에 비해 최대 25\% (top-1) 및 16\% (top-10) 높아진 정확도를 달성했습니다.



### Machine Learning and Multi-source Remote Sensing in Forest Carbon Stock Estimation: A Review (https://arxiv.org/abs/2411.17624)
Comments:
          First author and corresponding author: Autumn Nguyen

- **What's New**: 이번 연구는 산림 탄소를 정량화하는 데 있어 머신러닝(Machine Learning)과 원격 탐사(Remote Sensing) 기술의 최신 조합에 대한 체계적인 검토를 제공하였습니다. 총 25개의 논문을 분석하여 28가지의 머신러닝 방법과 주요 원격 탐사 데이터의 조합을 식별했습니다. 연구 결과, Random Forest가 가장 빈번하게 사용된 방법으로 보였으며, Extreme Gradient Boosting은 다른 방법과 비교했을 때 뛰어난 성능을 보였습니다. 이러한 결과는 머신러닝과 원격 탐사를 통합하여 정확하고 확장 가능한 산림 탄소 저장량 추정을 위한 최선의 관행을 권장할 수 있는 기반을 제공합니다.

- **Technical Details**: 연구에서 사용된 25개의 논문은 산림 탄소 또는 바이오매스 맵(map) 추정을 위한 머신러닝 방법의 조합을 명시적으로 나타내는 키워드로 검색되었습니다. 머신러닝 알고리즘은 특정 산림 지역의 탄소 저장량 측정값을 학습하여 다른 산림 지역의 예측값을 생성하는 데 도움을 줍니다. 다원적 접근법(multi-sensor approaches)은 Sentinel-1과 같은 원격 탐사 데이터에서 유래하며, LiDAR 및 SAR과 같은 진보된 센서를 통해 산림의 바이오매스 내용을 정량화하는 데 사용됩니다. 이러한 원격 탐사 기술을 통해 대규모 산림에 대한 정보가 수집되며, 바이오매스 및 식생 특성에 대한 정확한 측정을 가능하게 합니다.

- **Performance Highlights**: 연구에서 가장 효과적으로 사용된 알고리즘은 Random Forest로, 88%의 연구에서 가장 자주 등장했습니다. Extreme Gradient Boosting은 비교 대상이 되었던 다른 방법들과의 성능 비교에서 75%의 연구에서 우수한 결과를 보였습니다. 연구에서 나타난 원격 탐사 데이터의 활용에서 Sentinel-1이 가장 많이 사용되었으며, 복수의 센서를 사용하는 접근 방식을 통해 더욱 효과적인 결과를 도출할 수 있음을 강조했습니다. 이러한 성과들은 탄소 저장량 추정의 정확성을 개선하는 데 중요한 기여를 하며, 각기 다른 데이터를 통합하는 것이 장기적인 해결책이 될 수 있다는 점을 보여줍니다.



### Automating Chapter-Level Classification for Electronic Theses and Dissertations (https://arxiv.org/abs/2411.17614)
- **What's New**: 이 논문은 전통적인 전자 석사논문 및 박사논문(ETD) 보관 방식의 한계를 극복하기 위해 AI와 머신 러닝을 활용하여 각chapter에 대한 메타데이터를 자동으로 분류하는 방법을 제안합니다. 기존의 고수준 메타데이터는 ETD의 깊이와 복잡성을 제대로 담아내지 못해 연구자들이 특정 정보를 찾는 데 어려움을 겪고 있었습니다. 이를 해결하기 위해 chapter-level metadata를 도입하여 각 장의 내용을 보다 쉽게 찾을 수 있도록 하여 학술 연구의 접근성과 활용성을 높이고자 합니다.

- **Technical Details**: 이 연구는 두 가지 주요 작업인 segmentation과 classification으로 구성됩니다. 우선, segmentation 작업을 통해 PDF 내 chapter 경계를 식별하고, 이후 classification 작업을 통해 각 장에 대한 구체적인 메타데이터를 할당합니다. 이를 통해 연구자들은 전통적인 문서 수준의 메타데이터에서 벗어나 각각의 chapter를 더 효과적으로 탐색할 수 있습니다. 우리는 전통적인 머신 러닝 분류기, BERT 기반의 모델, 대형 언어 모델(LLMs) 간의 효과를 비교하며, 각 장의 특정 권한 레이블을 생성하는 접근법을 탐구합니다.

- **Performance Highlights**: 개발된 프로토타입 시스템을 통해 chapter-level classification labels가 제공되면 연구자들은 자신의 관심사에 맞는 chapters를 신속하게 검색할 수 있습니다. 또한, AI 향상 접근 방식을 통해 아카이브가 연구자들에게 더 나은 서비스를 제공하고 정보 접근성을 증가시키며 ETD와의 깊은 상호 작용을 지원할 수 있도록 합니다. 이를 통해 ETD의 학문적 도구로서의 영향력 강화, 학제 간 탐색 촉진 및 학술 커뮤니케이션에서 아카이브의 역할을 재강조할 수 있습니다.



### Mixed-State Quantum Denoising Diffusion Probabilistic Mod (https://arxiv.org/abs/2411.17608)
Comments:
          7 pages, 7 figures

- **What's New**: 이번 논문에서는 기존의 quantum denoising diffusion probabilistic models (QuDDPMs)의 한계를 극복하기 위해 mixed-state quantum denoising diffusion probabilistic model (MSQuDDPM)을 제안합니다. 이 모델은 scrambling unitaries 없이도 작동하며, depolarizing noise channels를 forward diffusion 과정에 통합하여 효율적인 샘플 생성을 가능하게 합니다. MSQuDDPM은 순수 상태뿐만 아니라 혼합 상태의 양자 상태를 생성할 수 있는 능력을 확장합니다.

- **Technical Details**: MSQuDDPM의 아키텍처는 forward 과정에서 잡음을 점진적으로 주입하여 원래 분포를 혼합 상태로 변환하는 구조로 되어 있습니다. backward 과정에서는 parameterized quantum circuits (PQC)를 이용하여 잡음을 제거하는 훈련 전략이 적용됩니다. 이러한 구조적 변화는 훈련의 복잡성을 줄이면서도 성능을 향상시키는 방법으로, 여러 noise scheduling 기법과 superfidelity 기반의 손실 함수를 활용합니다.

- **Performance Highlights**: MSQuDDPM은 여러 양자 상태 집합 생성 작업에서 성공적인 성과를 보였으며, 향상된 수렴 속도를 입증했습니다. 기존의 QuDDPM과 유사한 성능을 유지하면서도 구현 복잡성을 감소시켰고, 이는 양자 생성 모델의 실질적인 활용 가능성을 높입니다. 다양한 응용 프로그램에서 더욱 효율적인 데이터 생성을 목표로 하여 향후 연구 방향이 기대되고 있습니다.



### Making History Readab (https://arxiv.org/abs/2411.17600)
- **What's New**: 본 논문은 버지니아 공대 도서관의 디지털 라이브러리 플랫폼(DLP)이 역사적 및 문화적 중요성이 있는 다양한 문서에 대한 접근을 제공하는 방법을 제시합니다. AI를 활용하여 손글씨 인식, 텍스트 추출, 요약 등을 통해 복잡한 디지털 자료에 대한 온라인 접근성을 개선하고 있습니다. 이러한 접근은 특히 1860년대의 남북전쟁 편지, 역사적인 신문, 디지털 지형도와 같은 세 가지 컬렉션을 통해 구체화됩니다.

- **Technical Details**: DLP는 클라우드 기반 솔루션으로, 텍스트 인식과 메타데이터 생성을 위한 Optical Character Recognition (OCR) 기술을 통합하고 있습니다. Pytesseract 및 AWS Textract와 같은 도구를 사용해 손글씨와 복잡한 레이아웃의 문서에서 텍스트를 추출하며, LLM(large language models)은 요약 서비스를 제공합니다. 특히, Llama-3.1-8B-Instruct 모델을 활용하여 손글씨 텍스트에서 요약을 생성하는 방법이 소개됩니다.

- **Performance Highlights**: 손글씨 문서의 텍스트 추출 및 요약 처리 결과, 이전에 접근할 수 없었던 콘텐츠가 검색 가능하도록 변환되었습니다. 신문 컬렉션에서는 복잡한 열 구성으로 인한 인식 문제를 해결하고, 지형도 컬렉션의 경우 텍스트 비스듬한 배치 문제를 해결하기 위해 다각 회전 전략을 사용하였습니다. 이러한 방법들은 디지털 라이브러리 플랫폼의 이용자 경험을 향상시키며, 더 많은 사용자들이 자료를 쉽게 탐색하고 이해할 수 있도록 지원합니다.



### Agentic AI for Improving Precision in Identifying Contributions to Sustainable Development Goals (https://arxiv.org/abs/2411.17598)
- **What's New**: 이 연구는 연구 기관들이 유엔의 지속 가능한 발전 목표(SDGs)에 대한 연구 성과를 정확히 평가해야 하는 필요성을 강조합니다. 기존의 키워드 기반 접근 방식을 넘어 자가 회귀형 대형 언어 모델(LLMs)을 이용한 평가 방법론을 제안합니다. LLMs는 여러 문서 간의 의미상 관련성을 구분할 수 있어 SDG 목표에 대한 진정한 기여도를 파악하는 데 도움을 줍니다.

- **Technical Details**: 연구는 Scopus에서 선택한 쿼리를 통해 17개 SDGs 각각에 대해 20,000개의 학술 초록을 수집하여 이를 분석합니다. 이 과정에서 Phi-3.5-mini, Mistral-7B, Llama-3.2와 같은 소형 LLMs를 활용하여 추출한 데이터의 관련성을 평가합니다. 각 모델은 구체적인 SDG 표적과의 일치를 기준으로 초록을 'Relevant' 또는 'Non-Relevant'로 분류하며, 이를 통해 연구 모델 간의 차이를 드러내고자 합니다.

- **Performance Highlights**: 각 LLM의 'Relevant'와 'Non-Relevant' 초록의 비율은 상이하며, 이는 서로 다른 평가 기준을 적용한 결과를 나타냅니다. Phi-3.5-mini는 52%를 'Relevant'로, Mistral-7B는 70%, Llama-3.2는 15%의 'Relevant' 비율을 보였습니다. 이러한 결과는 다양한 모델의 분류 전략 통합의 중요성을 시사하며, 향후 다중 에이전트 프레임워크를 통해 SDG 기여도 분류의 신뢰성을 강화할 수 있는 가능성을 보여줍니다.



### What Differentiates Educational Literature? A Multimodal Fusion Approach of Transformers and Computational Linguistics (https://arxiv.org/abs/2411.17593)
- **What's New**: 최근 문헌을 교육 과정에 통합하는 것은 여전히 어려운 문제입니다. 본 연구는 Transformer 기반 텍스트 분류와 언어 특징 분석을 결합한 다중 모달(multimodal) 접근 방식을 제안하여 이러한 격차를 해결하고자 합니다. 특히, 연구결과 8개의 최첨단 Transformers가 세분화된 텍스트 데이터에 대해 미세 조정(fine-tuned)되었으며, BERT는 최고 F1 점수인 0.75를 달성했습니다.

- **Technical Details**: 연구에서는 500개의 딥 뉴럴 네트워크(topologies) 구성을 검색하여 언어적 특성을 분류하고, 그 결과 F1 점수 0.392를 획득했습니다. 이 모달리티의 융합은 상당한 개선을 보였으며, 모든 다중 모달 접근 방식이 단일 모달 모델을 초월했습니다. 특히, ELECTRA Transformer와 뉴럴 네트워크가 융합되어 F1 점수 0.996을 달성하였습니다.

- **Performance Highlights**: 제안된 접근 방식은 비기술적 이해관계자에게 실시간 텍스트 복잡성(text complexity), 독서 난이도(reading difficulty), 교육과정 정렬(curriculum alignment), 그리고 학습 연령 범위에 대한 추천을 제공하는 웹 애플리케이션으로 묶였습니다. 이 응용 프로그램은 데이터 기반 의사결정을 가능하게 하고, 영어 문학 수업의 계획에서 AI 기반 추천을 통합함으로써 수작업 부담을 줄이는 데 기여합니다.



### Learning Explainable Treatment Policies with Clinician-Informed Representations: A Practical Approach (https://arxiv.org/abs/2411.17570)
Comments:
          Proceedings of Machine Learning for Health (ML4H) 2024. Code available at: this https URL

- **What's New**: 최근 연구에서는 디지털 헬스介入(DHIs)과 원격 환자 모니터링(RPM)이 만성 질환 관리에 있어 개인 맞춤형 치료 전략을 통해 큰 잠재력을 보여준다고 강조하고 있습니다. 기존 DHIs의 적용은 효과의 불확실성과 업무 부담으로 인해 제한되고 있으며, 이 논문에서는 설명 가능한 치료 정책을 학습하기 위한 새로운 파이프라인을 개발하였습니다. 이 접근법은 실제 원격 환자 모니터링 상황에서 제1형 당뇨병 환자의 혈당 조절을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 파이프라인은 (1) 저차원 상태 및 행동 표현을 학습하고, (2) 추정된 조건부 평균 치료 효과(CATEs)를 기반으로 환자를 순위화하여 정책을 구성하며, (3) 용량 제약을 고려하여 정책을 평가하는 세 가지 단계로 이루어져 있습니다. 임상 도메인 지식이 포함된 상태와 행동 표현을 사용하여 환자를 관리하는 최적 정책을 학습했으며, 이는 블랙박스 모델로부터 학습한 정책보다 더 효과적이고 효율적인 것으로 나타났습니다. 이는 실제 상황에서 정책의 성능에 큰 이점을 제공합니다.

- **Performance Highlights**: 이 연구 결과, 임상 지식 기반의 정책이 블랙박스 모델에서 학습한 정책에 비해 유의미하게 더 높은 효율성과 효과성을 보여주는 것으로 나타났습니다. 특히, '무작위 타겟팅'보다 학습된 정책이 성능을 발휘하는 것은 임상 지식이 정책 개선에 중요한 역할을 한다는 것을 강조합니다. 이 연구는 ML 연구자와 임상 의사 간의 협력이 실제 DHIs 개발에 있어 필수적이라는 점을 재확인합니다.



### A Bilayer Segmentation-Recombination Network for Accurate Segmentation of Overlapping C. elegans (https://arxiv.org/abs/2411.17557)
- **What's New**: 이번 논문에서는 C. elegans의 세분화 문제를 해결하기 위한 새로운 방법인 Bilayer Segmentation-Recombination Network (BR-Net)을 제안합니다. 기존 연구에서 C. elegans의 경계가 불분명하고 겹쳐지는 문제를 해결하고자 하는 노력은 있었지만, BR-Net은 세 가지 모듈로 구성되어 더 정교한 세분화를 가능하게 합니다. 특히, Unified Attention Module (UAM)을 통해 굵은 마스크를 더욱 효과적으로 인지하도록 설계하였습니다.

- **Technical Details**: BR-Net은 Coarse Mask Segmentation Module (CMSM), Bilayer Segmentation Module (BSM), Semantic Consistency Recombination Module (SCRM)로 구성되어 있습니다. CMSM은 초기 세분화를 위해 주로 사용되며, BSM은 겹치는 지역과 비겹치는 지역을 나누는 역할을 합니다. SCRM은 세맥적 일관성(semantic consistency)을 부여하여 C. elegans의 보다 정확한 세분화를 도와줍니다.

- **Performance Highlights**: 실험 결과, BR-Net은 C. elegans 데이터셋에서 다른 최신 인스턴스 세분화 방법들보다 우수한 성능을 보여주었습니다. 특히, C. elegans 인스턴스의 겹침 이미지 처리에서 경쟁력 있는 결과를 나타내어, 이 방법이 생물학적 이미지 처리 분야에서의 실용성을 높일 것으로 기대할 수 있습니다.



### Rapid Deployment of Domain-specific Hyperspectral Image Processors with Application to Autonomous Driving (https://arxiv.org/abs/2411.17543)
- **What's New**: 이 논문에서는 저비용의 시스템 온 모듈(System-On-Module, SOM) 플랫폼을 활용하여 자율 주행에 적용 가능한 효율적인 하이퍼스펙트럼 이미징(Hyperspectral Imaging, HSI) 프로세서를 구현하는 방법을 다룹니다. 기존의 고성능 처리 시스템에서 저비용 SOM으로 성공적으로 재설계된 경량 FCN(fully convolutional network)을 이용하여, 이미지의 의미론적 분할(latency) 요구 사항을 충족하는 모델의 양자화 과정을 설명하고 있습니다. 이 연구는 데이터 및 하드웨어 특화된 양자화 기술을 보고하며, HSI를 위한 저비용 AI 코프로세서의 사용 가능성을 높이고 있습니다.

- **Technical Details**: 저비용 SOM의 도입은 자율 주행 시스템(ADS)에 대한 하이퍼스펙트럼 이미징의 적용 가능성을 넓힙니다. 연구자들은 이전에 개발된 HSI 분할 시스템을 8비트 양자화 모델로 재설계하였으며, 이를 통해 이미지 전처리와 특징 추출의 효율성을 높임과 동시에 파라미터 수를 4배 줄였습니다. FCN 모델은 Keras/Tensorflow2를 사용하여 개발되었고, NVIDIA GFORCE RTX-3090에서 학습되었습니다.

- **Performance Highlights**: 모델의 효율성과 정확도 향상을 위해 FCN 양자화의 과정이 세심하게 진행되었습니다. 요약한 결과, 억제된 정밀도로도 전체 성능 저하 없이 IoU 지수의 변화를 유지하며, 8비트 정밀도의 Min-Max 양자화에 강건한 성능을 보여주었습니다. 마지막으로, 이 연구는 AMX-Xilinx Vitis AI 3.0 도구를 활용하여 맞춤형 양자화 파이프라인을 구성함으로써 시스템의 성능을 극대화하고 있습니다.



### AI-Augmented Ethical Hacking: A Practical Examination of Manual Exploitation and Privilege Escalation in Linux Environments (https://arxiv.org/abs/2411.17539)
Comments:
          101 pages

- **What's New**: 이번 연구는 리눅스 기반 침투 테스트 환경에서 매뉴얼 활용과 권한 상승 작업에 대해 생성적 AI(GenAI)의 적용을 탐구합니다. 특히, 이 연구는 GenAI가 보안 평가에서 중요한 수작업 작업을 지원하는 방법을 실험적으로 분석합니다. 이에 따라 GenAI는 공격 경로 식별 및 복잡한 출력 검사 과정에서의 민감한 데이터 식별을 용이하게 하며 효율성을 증대시킬 수 있다는 결과를 보여줍니다.

- **Technical Details**: 이 연구는 캘리 리눅스 VM 및 다양한 리눅스 VM을 포함한 가상화 환경에서 실험을 진행했습니다. 실험은 ChatGPT-4o를 사용하여 각 단계에서 안내를 받으며 진행되었으며, 보안 평가를 보다 효율적이고 비용 효과적으로 할 수 있는 방법을 모색했습니다. 또한, ChatGPT는 정보 수집부터 접근 획득 및 권한 상승에 이르는 모든 과정에서 실질적인 지원을 제공하였습니다.

- **Performance Highlights**: 연구 결과, GenAI는 보안 평가의 효율성 향상에 기여했으며, 특히 매뉴얼 작업인 권한 상승 과정에서도 두각을 나타냈습니다. 그러나 데이터 프라이버시와 같은 윤리적 우려와 프로그램의 오용 가능성 같은 도전 과제를 함께 지적하고 있습니다. 이는 AI와 인간의 협업이 중요하다는 점과 함께 보안 분야에서의 GenAI의 가능성을 강조합니다.



### HSI-Drive v2.0: More Data for New Challenges in Scene Understanding for Autonomous Driving (https://arxiv.org/abs/2411.17530)
- **What's New**: HSI-Drive 데이터셋의 업데이트된 버전인 v2.0이 도입되었습니다. 이 버전은 이전 버전보다 272% 더 많은 752개의 주석 달린 이미지를 포함하고 있으며, 계절 변화에 따른 다양한 주행 환경에서 수집된 이미지를 포함하고 있습니다. 특히, 새로운 데이터셋은 더 나은 HSI(segmentation)를 위한 설계 기반의 지식 생성에 기여합니다.

- **Technical Details**: v2.0 데이터셋은 겨울, 가을, 봄, 여름을 포함한 계절별 주행 데이터로 구성되어 있습니다. 각 이미지는 1088 x 2048 픽셀의 공간 해상도를 가지고 있으며, 5x5 픽셀 윈도우로 구성된 Fabry-Perot 필터로부터 스펙트럼 밴드가 추출됩니다. 이 데이터셋은 10개의 카테고리로 레이블이 지정된 약 4400만 개의 픽셀로 구성되어 있습니다.

- **Performance Highlights**: 실험을 통해 FCN(fully convolutional networks) 모델을 업데이트하였으며, 새로운 데이터의 통합이 모델의 성능과 강건성을 향상시키는 것을 보여주었습니다. 특히, 교통 안전을 위한 도로 표지 및 보행자 구분에 초점을 맞춘 세분화 성과가 두드러집니다. 데이터셋에서 추출된 이미지로 훈련한 모델들은 다양한 주행 조건에서도 유의미한 성과를 보였습니다.



### On Statistical Rates of Conditional Diffusion Transformers: Approximation, Estimation and Minimax Optimality (https://arxiv.org/abs/2411.17522)
- **What's New**: 이번 연구에서는 조건부 확산 변환기(conditional diffusion transformers, DiTs)와 관련된 새로운 근사 및 추정 속도에 대한 분석을 수행했습니다. 특히, 분류기 없는 가이드를 사용하여 조건부 DiTs의 성능을 더욱 세밀히 검토하고, 이를 통해 더 효율적이고 정확한 DiT 모델 개발에 대한 실질적인 지침을 제공합니다.

- **Technical Details**: 연구는 네 가지 공통 데이터 가정 하에서 '상황(context)-내' 조건부 DiTs에 대한 포괄적인 분석을 제시합니다. 입력 도메인을 미세한 그리드로 이산화하고, 조건부 확산 점수 함수에 대해 테일러 전개(taylor expansion)를 수행하여 변환기의 보편적 근사를 보다 세부적으로 이용합니다. 이를 통해 더 낮고 굳건한 경계를 얻을 수 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, 잠재적인 조건부 DiTs는 조건부 DiTs보다 근사 및 추정에서 더 낮은 경계를 달성하며, 잠재적인 무조건 DiTs의 minimax optimality가 입증되었습니다. 이 연구는 조건부 및 무조건 DiTs에 대한 통계적 한계를 확립하고, 효율적이고 정확한 DiT 모델을 개발하기 위한 방향성을 제시합니다.



### Inference Scaling $\scriptsize\mathtt{F}$Laws: The Limits of LLM Resampling with Imperfect Verifiers (https://arxiv.org/abs/2411.17501)
- **What's New**: 최근 연구에 따르면, inference scaling이 약한 언어 모델이 더 강한 모델과 동일하거나 더 많은 정확성을 달성하도록 도와줄 수 있는 가능성이 제기되었습니다. 하지만 이 논문의 주장에 따르면, "검증자"가 완벽하지 않으면 무한한 정확성 향상은 불가능합니다. 특히 reasoning이나 coding과 같이 검증자가 거의 항상 불완전한 도메인에서는 false positives의 확률이 존재하므로 resampling의 정확도에 한계가 있습니다.

- **Technical Details**: 우리는 각 모델의 단일 샘플 정확도와 HumanEval 및 MBPP와 같은 코딩 벤치마크에서의 false positive 비율 간의 강한 상관관계를 발견했습니다. 이와 같은 결과는 더 낮은 성능을 가진 모델이 강력한 모델의 단일 호출 성능을 초과하는 것이 불가능하다는 것을 보여줍니다. 또한, false positive의 존재가 코딩 스타일 및 품질을 저하시킬 수 있다는 점도 강조하였습니다.

- **Performance Highlights**: 이 논문은 resampling을 통해 약한 모델이 더 강한 모델의 성능에 도달할 수 있는 가능성은 실질적으로 없다는 것을 입증합니다. 모델이 가진 실제 성능과 격차는, 무한한 추론 예산을 사용할 경우에도 미세하게 감소할 뿐이며, 최적의 샘플 수는 10 이하일 가능성이 높습니다. 따라서 무한한 컴퓨팅 환경에서도 잘못된 해답을 반환할 가능성이 높은 경우, 그 모델의 유용성이 매우 낮아질 수 있습니다.



### What's in the Image? A Deep-Dive into the Vision of Vision Language Models (https://arxiv.org/abs/2411.17491)
- **What's New**: 이번 논문에서는 Vision-Language Models (VLMs)의 시각적 정보 처리 메커니즘을 탐구합니다. VLM들은 복잡한 시각 콘텐츠를 이해하는 데 뛰어난 능력을 보여주며, 특히 이미지 정보를 처리하는 방법에 대한 새로운 통찰을 제공합니다. 이 연구를 통해 VLM이 이미지 토큰에 직접 접근하지 않고도 쿼리 텍스트 토큰만으로도 설명적인 응답을 생성할 수 있다는 점을 강조했습니다.

- **Technical Details**: 이 연구에서는 VLM의 다양한 레이어에서 attention 모듈을 분석하며, 시각-언어 정보 흐름에 대한 드러난 메커니즘을 탐구합니다. 특히, 중간 레이어가 정보 흐름에 중요한 역할을 하며, 세밀한 시각적 속성과 객체 세부사항은 공간적으로 지역화된 방식으로 이미지 토큰에서 직접 추출된다는 점을 보여줍니다. 이를 검증하기 위해 LLM 기반의 새로운 평가 프로토콜을 제안했습니다.

- **Performance Highlights**: 또한, 본 연구는 기존 VLM의 내부 표현을 압축된 맥락 공간으로 증류하여 효율적인 시각 처리를 가능하게 하였으며, 'Image Re-prompting'이라는 새로운 애플리케이션을 제안했습니다. 이 기술은 이미지에 대한 여러 질문을 압축된 콘텐츠만으로 효율적으로 묻는 것을 가능하게 하며, 압축된 맥락이 전체 데이터에 비해 20배 더 작지만 시각적 질문 응답에서 96%의 성능을 유지하는 것으로 나타났습니다.



### Puzzle Similarity: A Perceptually-guided No-Reference Metric for Artifact Detection in 3D Scene Reconstructions (https://arxiv.org/abs/2411.17489)
- **What's New**: 이번 작업에서는 새로운 no-reference 메트릭인 Puzzle Similarity를 제안하여, 새로운 뷰에서 아티팩트를 효과적으로 로컬라이즈할 수 있게 합니다. 기존의 no-reference 비주얼 품질 측정법이 제공하지 못하는 문제를 해결하고, 인간 평가와 상당히 잘 일치하는 아티팩트 맵을 생성할 수 있습니다. 이 결과는 이미지 복원 및 3D 재구성 개선 응용에 유용하게 활용될 수 있습니다.

- **Technical Details**: 제안하는 방법은 입력 뷰로부터 학습된 이미지 패치 통계(패치 통계)를 활용해 특정 장면에 대한 분포를 설정합니다. 이를 통해, 새로운 이미지를 해당 통계와 비교하여 아티팩트가 있는 영역을 식별합니다. 이 방법은 추가적인 레퍼런스 이미지 없이도 아티팩트 매핑을 제공하며, 기존의 no-reference와 full-reference 메트릭보다 성능이 뛰어납니다.

- **Performance Highlights**: 제안된 메트릭은 수작업 실험을 통해 평가되어, 인간이 인식한 아티팩트의 위치와 잘 일치함을 증명했습니다. 이 방식은 기존 no-reference 메트릭 및 인기 있는 full-reference 이미지 메트릭보다 우수한 성능을 보여 영상을 복원하거나 데이터를 수집하는 데 있어 보다 나은 가이드를 제공할 수 있습니다.



### Towards Precise Scaling Laws for Video Diffusion Transformers (https://arxiv.org/abs/2411.17470)
- **What's New**: 이번 논문에서는 비디오 확산 변환기(video diffusion transformers)의 최적 성능을 달성하기 위해 데이터 및 계산 예산에 맞는 최적의 모델 크기와 하이퍼파라미터를 결정하는 중요성을 강조합니다. 기존의 언어 모델에서 사용되는 스케일링 법칙(scaling laws)이 시각 생성 모델에 어떻게 적용되는지를 체계적으로 분석하여, 그 존재를 확인했습니다. 또한, 이 논문에서 비디오 확산 모델이 학습률(learning rate) 및 배치 크기(batch size)에 더 민감함을 발견했으며, 이는 정밀하게 모델링되지 않은 하이퍼파라미터입니다.

- **Technical Details**: 논문에서 제안하는 새로운 스케일링 법칙은 어떤 모델 크기와 계산 예산에 대해 최적의 하이퍼파라미터를 예측할 수 있도록 설계되었습니다. 이러한 최적 설정 하에서 전통적인 스케일링 방법에 비해 성능을 비교 가능하게 유지하면서 추론 비용(inference costs)을 40.1% 감소시켰습니다. 또한, 검증 손실(validation loss), 모델 크기, 계산 예산 간의 일반화된 관계를 수립하여 비최적 모델 크기에서도 성능 예측이 가능하도록 하였습니다.

- **Performance Highlights**: 제안한 방법은 계산 예산 1e10 TFlops 내에서 비디오 확산 변환기의 성능을 최적화할 수 있게 해줍니다. 비최적 모델 크기를 포함한 다양한 경우에 대해 더욱 나은 균형(trade-off)을 이끌어내며, 실제 추론 비용 제약 하에서도 성능을 예측할 수 있습니다. 이러한 발견은 자원 효율적인 비디오 생성 모델 개발에 중요한 기여를 할 것으로 기대됩니다.



### ShowUI: One Vision-Language-Action Model for GUI Visual Agen (https://arxiv.org/abs/2411.17465)
Comments:
          Technical Report. Github: this https URL

- **What's New**: 본 연구에서는 GUI(그래픽 사용자 인터페이스) 작업을 수행하기 위한 혁신적인 비전-언어-액션 모델인 ShowUI를 개발하였습니다. ShowUI는 스크린샷을 UI 연결 그래프로 구성하여 시각적 토큰 선택을 UI 지침에 따라 조정하는 새로운 방법론을 채택했습니다. 또한, 다양한 GUI 작업을 통합하는 Interleaved Vision-Language-Action Streaming 구조를 통해 훈련 효율성을 극대화했습니다.

- **Technical Details**: UI 가이드를 제공하는 시각적 토큰 선택 방법은 스크린샷의 패치를 그래프의 노드로 표현하고, 연결된 구성 요소를 통해 시각적 중복성을 모델링합니다. 이 방법은 self-attention 블록에서의 토큰 선택을 최적화하여 계산 비용을 줄이고 있습니다. 또한, Action 해석을 돕기 위해 GUI 작업 공간을 JSON 형식으로 구조화하여, 다양한 시각-언어-액션 데이터를 효율적으로 관리합니다.

- **Performance Highlights**: ShowUI는 256K의 데이터로 경량화된 2B 모델을 구성하여 제로샷 스크린샷 그라운딩에서 75.1%의 높은 정확도를 달성했습니다. UI 지침에 따른 토큰 선택 방식을 통해 훈련 중 시각적 중복 토큰을 33% 감소시키고 성능을 1.4배 가속화하는 성과를 보였습니다. 이 모델은 웹, 모바일, 온라인 환경에서의 내비게이션 실험에서도 뛰어난 효과를 입증하였습니다.



### SoK: Decentralized AI (DeAI) (https://arxiv.org/abs/2411.17461)
Comments:
          This is a Systematization of Knowledge (SoK) for the rapidly evolving field of Decentralized AI (DeAI). We welcome valuable comments, suggestions, and collaboration to further refine and enhance this work. We hope our contribution will help accelerate the advancement of DeAI

- **What's New**: 이 논문은 블록체인 기반 분산 인공지능(Decentralized AI, DeAI) 솔루션의 체계화를 제안합니다. DeAI는 중앙집중식 AI가 안고 있는 문제들을 해결하기 위해 블록체인 기술의 장점을 활용합니다. 이를 통해 투명성(Transparency), 보안(Security), 분산화(Decentralization) 및 신뢰성(Trustworthiness)을 높일 수 있습니다.

- **Technical Details**: 논문에서는 DeAI 프로토콜의 분류를 위한 세분화된 분류체계를 제공합니다. 이 체계는 AI 모델의 생애주기(Model Lifecycle)에 따라 기존 DeAI 프로토콜을 구분할 수 있는 구조화된 방법을 제시합니다. 또한 블록체인이 DeAI에 어떻게 기능하는지 분석하며, AI 프로세스의 보안과 신뢰성을 높이는 데 기여하는 블록체인 기능을 조사합니다.

- **Performance Highlights**: 이 연구는 DeAI 프로토콜 개발에 있어 주요 통찰력과 연구의 공백들을 식별하고, 향후 연구를 위한 중요한 방향성을 제시합니다. 논문은 블록체인 기능이 AI 데이터와 모델 기여자들에게 공정한 보상을 보장한다는 점도 강조합니다. 이러한 분석을 통해 DeAI 솔루션의 발전 가능성과 실용성을 강조하고 있습니다.



### WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Mod (https://arxiv.org/abs/2411.17459)
Comments:
          8 pages, 7 figures

- **What's New**: 본 연구에서는 Wavelet Flow VAE (WF-VAE)를 제안하여 저주파수 에너지 흐름을 잠재 표현(latent representation)으로 유도합니다. 이 방법은 비디오를 다중 주파수 도메인 성분으로 분해하고, 이를 통해 중요한 정보를 효율적으로 인코딩하도록 설계되었습니다. 또한 Causal Cache라는 기법을 도입하여 블록별 추론(block-wise inference) 시 잠재 공간(latent space)의 무결성을 유지합니다.

- **Technical Details**: WF-VAE는 Haar wavelet transform을 활용하여 비디오 신호를 다중 레벨로 분해하고, 이를 통해 다계층 피라미드적 특징을 추출합니다. 저주파수 비디오 정보를 잠재 공간으로 직접 전달하는 주요 에너지 흐름 경로를 설정하며, 이를 통해 3D 합성곱(convolution) 수를 크게 줄입니다. Causal Cache 메커니즘은 콘볼루션 슬라이딩 윈도우의 연속성(convolution continuity)을 유지하여 블록 별 추론이 직접적인 추론과 동일한 성능을 보장합니다.

- **Performance Highlights**: WF-VAE는 PSNR 및 LPIPS 지표에서 기존 비디오 VAE에 비해 우수한 성능을 보이며, 처리량은 2배 이상, 메모리 소비는 4배 낮춥니다. 이 연구의 결과는 광범위한 실험 평가를 통해 입증되었으며, 비디오 재구성 및 생성에서 현재 최첨단 성능을 달성하는 것으로 확인되었습니다.



### Spatially Visual Perception for End-to-End Robotic Learning (https://arxiv.org/abs/2411.17458)
Comments:
          8 pages, 5 figures

- **What's New**: 최근 모방 학습(imitation learning) 분야의 발전은 로봇 제어 및 구현된 지능(embodied intelligence)에서 큰 가능성을 보여주고 있습니다. 하지만 다양한 장착 카메라 관찰에서 강력한 일반화(robust generalization)를 달성하는 것은 여전히 중요한 과제로 남아 있습니다. 본 논문에서는 환경 변동성에 대응하기 위해 3D 공간 표현(spatial representations)을 활용하는 비디오 기반 공간 인식(framework) 프레임워크를 소개하며, 조명 변화(lighting changes)를 처리하는 데 주안점을 두고 있습니다.

- **Technical Details**: 우리의 접근 방식은 인터넷 규모 데이터(internet-scale data)로 훈련된 최첨단 단안 깊이 추정 모델(monocular depth estimation model)과 이미지 증강 기법인 AugBlender를 통합하여 동적 시나리오에서 강건성과 적응성을 높입니다. AugBlender는 제어된 RGB 오염을 통해 훈련 분포를 확장하고, Monocular Depth Estimation 모델의 깊이 맵과 융합하여 환경 변화에 강한 내성을 제공합니다. 이러한 방식을 통해 우리의 시스템은 다양한 카메라 노출(camera exposures)에서 성공률을 크게 증대시키는 결과를 보여줍니다.

- **Performance Highlights**: 우리는 AugBlender와 Depth Anything V2를 이용하여 비용 효율적인 비디오 기반 인식을 실현하였으며, 로봇 팔, 두 개의 카메라, RTX 3090 GPU와 같은 저비용 설정에서도 높은 성능을 발휘합니다. 우리의 방법론은 다양한 환경 조건에서 확장 가능하고 일반화 가능한 프레임워크를 갖추고 있어 기존 시스템에 쉽게 통합될 수 있습니다. 실험 결과, 우리의 접근 방식은 특히 조명 변화와 같은 환경 변동에 강력한 회복력을 보여주어 모방 학습의 강건성을 크게 향상시킵니다.



### LC-SVD-DLinear: A low-cost physics-based hybrid machine learning model for data forecasting using sparse measurements (https://arxiv.org/abs/2411.17433)
- **What's New**: 이 논문에서는 고해상도 유체 동역학 데이터 예측을 위해 새로운 방법론, LC-SVD-DLinear를 제안합니다. 이 방법론은 저비용 고유값 분해(LC-SVD)와 DLinear 아키텍처를 결합하여 입력 특성, 특히 시계열 계수를 추세와 계절성 구성 요소로 분해합니다. 이를 통해 얕은 신경망이 비선형 동역학을 캡처할 수 있도록 하며, 저해상도 데이터로도 작업할 수 있어 전반적인 계산 비용이 절감됩니다.

- **Technical Details**: LC-SVD-DLinear 방법론은 고유값 분해(SVD)에 기반하며, 저해상도 데이터를 입력으로 사용하여 해당 데이터를 고해상도로 재구성합니다. 모델은 관련성 있는 고유 모드를 선택하여 노이즈를 최소화하고, 시계열 계수를 DLinear 모델로 전달하여 다음 데이터를 예측합니다. 또한 LC-HOSVD라는 변형 모델을 제안하여 고차원 데이터에 적용할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 실제 데이터셋을 사용한 검증을 통해 그 견고함을 입증하였습니다. 두 개의 데이터셋을 이용해 연구하였으며, 각각 원기둥 주위의 흐름에 대한 수치 시뮬레이션과 실험 데이터를 포함하고 있습니다. 오차 측정 기준을 통해 예측 및 재구성 결과로 성능을 평가하였으며, 예측 정확도가 향상되었음을 보여주었습니다.



### Rewiring Techniques to Mitigate Oversquashing and Oversmoothing in GNNs: A Survey (https://arxiv.org/abs/2411.17429)
- **What's New**: 본 논문은 Graph Neural Networks (GNNs)의 두 가지 주요 문제, 즉 오버스쿼싱(over-squashing)과 오버스무딩(oversmoothing)을 다루고 있습니다. 이러한 문제는 원거리 노드 간의 정보 손실과 노드 표현의 동질화를 유발하는데, 이는 GNN의 정보 흐름과 표현력을 제한합니다. 저자들은 그래프 구조를 변경하여 정보 확산을 개선하기 위한 그래프 리와이어링(graph rewiring) 기법에 대해 설명합니다.

- **Technical Details**: 이 논문에서는 GNNs의 기능을 극대화하기 위해 수행된 여러 그래프 리와이어링 기법을 포괄적으로 리뷰합니다. 그래프는 노드 집합과 엣지 집합으로 구성되며, 노드 간의 관계는 인접 행렬(adjacency matrix)로 모델링됩니다. 또한, 다양한 기법들이 노드의 특성을 반복적으로 전파(update)하고 집계(aggregation)하는 메시지 패싱 신경망(Message Passing Neural Networks) 설계를 기반으로 합니다.

- **Performance Highlights**: 그래프 리와이어링 기법은 특히 이질적인(heterophilic) 그래프에서의 노드 간 관계를 개선하며, 서로 다른 클래스의 노드를 연결할 때 발생하는 도전 과제를 완화하는 데 도움이 됩니다. 저자들은 이러한 방법이 효과적이나, 일부 방법은 스케일러빌리티(scalability)와 같은 제한으로 인해 더 큰 그래프에서의 적용이 어려울 수 있다고 지적합니다. 짧은 그래프 내에서 다중 홉 집계(multi-hop aggregation) 기법이 도움이 될 수 있지만, 이는 오버스쿼싱과 오버스무딩을 악화시킬 위험이 있음을 언급합니다.



### CLOVER: Constrained Learning with Orthonormal Vectors for Eliminating Redundancy (https://arxiv.org/abs/2411.17426)
- **What's New**: 이 연구에서는 대형 모델의 다운스트림 태스크(adaptation) 조정을 위해 원 조합(basis vectors)의 선형 결합(linear combinations)을 활용하여 잠재 공간(latent space) 내에서 학습을 제한하는 방법을 제안합니다. 이는 모델의 능력을 손상시키지 않으면서 안정적인 훈련(stable training)을 보장합니다. 또한, Whisper-large-v3의 인코더에서 불필요한 벡터(redundant vectors)를 제거하여 매개변수를 46.42% 줄이는 효과를 보이며 추가 훈련 없이도 가능합니다.

- **Technical Details**: Absorb-Decompose라는 방법을 통해 Q, K, V, O 매트릭스의 직교화(orthogonalization)를 구현했습니다. 전통적으로 매트릭스를 직교화하려면 전이 매트릭스(transfer matrix)가 필요했으나, 우리 방법은 이 필요성을 제거하며 더욱 효율적인 파라미터 관리를 가능하게 합니다. 이 연구는 각 주의(attention) 레이어에 존재하는 두 쌍의 매트릭스(W_Q, W_K^T 및 W_V, W_O)를 이용하며, SVD를 통해 이들을 직교화하고 불필요한 벡터를 축소하여 매개변수의 수를 줄이는 접근법을 취하고 있습니다.

- **Performance Highlights**: LLaMA-2-7B 모델을 8개의 상식 추론(common sense reasoning) 데이터셋에서 미세 조정(fine-tuning) 했을 때, 본 방법은 LoRA에 비해 5.4%, DoRA에 비해 4.4%의 성능 향상을 보였습니다. 이러한 결과는 효율적이고 안정적인 미세 조정이 가능함을 보여줍니다. 본 연구의 CLOVER 메소드는 매트릭스의 선형 조합만을 조정하며, 이는 과거 학습된 기능을 그대로 유지하게 해줍니다.



### Can LLMs be Good Graph Judger for Knowledge Graph Construction? (https://arxiv.org/abs/2411.17388)
- **What's New**: 이 연구에서는 비구조화된 데이터를 구조화된 지식 그래프(KGs)로 변환하기 위한 새로운 프레임워크인 GraphJudger를 제안합니다. GraphJudger는 세 가지 혁신적인 모듈인 entity-centric iterative text denoising, knowledge aware instruction tuning, graph judgement를 포함하여, 지식 그래프 생성에서 발생할 수 있는 여러 문제를 해결하는 데 중점을 두고 있습니다. 특히, 대규모 언어 모델(LLMs)의 역할을 단순한 예측자가 아닌 그래프 판단자로 확장하고자 하며, 이를 통해 더 높은 품질의 KGs를 구축할 수 있도록 합니다.

- **Technical Details**: GraphJudger는 다음 세 가지 도전과제를 해결하기 위해 세분화된 접근법을 제안합니다. 첫째, Entity-Centric Iterative Text Denoising 모듈을 통해 실제 문서에서 불필요한 정보를 제거하여 LLMs가 필요한 정보를 효과적으로 추출할 수 있도록 합니다. 둘째, Knowledge Aware Instruction Tuning 모듈은 LLM을 조정하여 도메인 지식의 이해를 높이고 지식 그래프 완성 작업의 성능을 향상시킵니다. 셋째, Graph Judgement 모듈은 생성된 triples의 올바름을 평가하여 잘못된 아이템을 걸러냄으로써 전체 KGs의 품질을 개선합니다.

- **Performance Highlights**: GraphJudger는 두 개의 일반 텍스트-그래프 쌍 데이터셋과 하나의 도메인 특정 텍스트-그래프 쌍 데이터셋에서 수행된 실험에서 기존 방법들보다 우수한 성능을 보였습니다. 이 연구는 지식 그래프 구축 과제에서 LLMs의 효율성을 높이고, 필수적인 노이즈 제거와 도메인 지식 강화 방안을 통해 실제 세계의 데이터를 더 잘 활용할 수 있는 방법을 제안합니다. GraphJudger의 성공적인 성능 향상은 KG 구축이 더욱 신뢰할 수 있는 정보를 포함할 수 있도록 해줍니다.



### Fairness And Performance In Harmony: Data Debiasing Is All You Need (https://arxiv.org/abs/2411.17374)
- **What's New**: 이 연구는 870개의 프로필을 포함한 실제 대학 입학 데이터셋을 사용하여 머신 러닝(ML) 모델의 공정성을 조사합니다. XGB, Bi-LSTM, KNN 세 가지 ML 모델을 활용하여 성별 언어 편향을 제거하는 파이프라인을 제안하며, ML 모델이 인간 전문가보다 14.08%에서 18.79% 더 높은 공정성을 기록함을 보여줍니다. 또한, 성별 언어 편향 제거 후 모든 모델의 분류 정확도가 유지되거나 향상되는 성과를 보였습니다.

- **Technical Details**: 이 연구에서는 BERT(Bidirectional Encoder Representations from Transformers) 임베딩을 사용하여 고차원 텍스트 특징을 인코딩하고, 인간 전문가의 결정과 ML 모델의 일관성을 평가하기 위해 일관성 점수를 사용합니다. 성별 편향을 완화하기 위해 설계된 디바이빙 파이프라인은 입력 기능에서 성별 특정 언어를 제거하고, 사전 훈련된 BERT 모델을 활용하여 성별 추론을 수행합니다. 데이터 준비 과정에서는 BERT 임베딩을 생성하는 과정을 통해 각 프로필에 대해 결합된 표현을 형성합니다.

- **Performance Highlights**: 실험 결과, ML 모델이 인간 전문가의 결정보다 더 높은 일관성 점수를 기록하며 공정한 결정에서의 신뢰성을 진단합니다. 성별 편향 제거 후, 모델들의 분류 정확도가 향상되거나 유지되는 결과를 도출했습니다. 이러한 결과는 공정성과 성능이 동시에 존재할 수 있음을 검증하며, ML 모델이 높은 정확도를 유지하면서 입학 공정성을 향상시킬 수 있다는 가능성을 제시합니다.



### Knowledge-aware Evolutionary Graph Neural Architecture Search (https://arxiv.org/abs/2411.17339)
Comments:
          This work has been accepted by Knowledge-Based Systems

- **What's New**: 이 연구에서는 그래프 작업이나 데이터셋을 위해 맞춤형 그래프 신경망 아키텍처를 설계할 수 있는 그래프 신경 아키텍처 검색(GNAS) 기술의 새로운 접근 방식을 소개합니다. 기존 GNAS 방법들이 사전 지식 없이 아키텍처를 탐색하는 데 비해, 본 논문에서는 NAS-Bench-Graph와 같은 데이터베이스에 존재하는 풍부한 아키텍처 정보를 활용하여 검색 효율성을 높이는 방법을 제안합니다. KEGNAS라는 새로운 프레임워크를 통해 이전에 학습된 지식을 이용하여 효과적으로 고성능 아키텍처를 생성할 수 있습니다.

- **Technical Details**: KEGNAS는 사전 지식 모델을 학습하여 데이터셋과 아키텍처 간의 매핑을 수립하고, 이를 통해 새로운 데이터셋에 대한 후보 아키텍처를 신속하게 생성합니다. 이어서, Deep Multi-output Gaussian Process (DMOGP)를 설계하여 후보 아키텍처의 성능 지표(#Acc 및 #Params)를 예측합니다. 이 과정은 기존 데이터셋과 아키텍처에서 훈련된 서브모델을 사용하여 수행되며, 새로운 데이터셋에 대해 짧은 GPU 시간 내에 성능 예측을 가능케 합니다.

- **Performance Highlights**: KEGNAS는 NAS-Bench-Graph 및 다섯 개의 실제 그래프 데이터셋에서 저명한 GNAS 방법들을 초월하는 성능을 보여주었습니다. 실험 결과, KEGNAS는 고급 진화 방식에 비해 4.27% 높은 정확도를, 고급 미분 기반 방법에 비해 11.54% 높은 정확도를 달성했습니다. 또한, ablation study를 통해 사전 지식 활용이 검색 성능을 크게 향상시키는 효과를 입증하였습니다.



### Different Bias Under Different Criteria: Assessing Bias in LLMs with a Fact-Based Approach (https://arxiv.org/abs/2411.17338)
Comments:
          Accepted in NeurIPS 2024 Workshop on Socially Responsible Language Modelling Research (SoLaR)

- **What's New**: 이번 연구는 기존의 동등성 기반 접근 방식과 다른 사실 기반 기준을 사용한 새로운 편향 평가 메트릭을 제안합니다. 연구진은 LLM의 출력을 실제 세계의 인구 통계 분포와 연관시키는 방식으로 편향을 측정하고자 했습니다. 이 연구는 여러 LLM 모델의 출력이 서로 다르게 평가됨을 보여주며, 다양한 관점에서의 평가가 필요하다는 점을 강조합니다.

- **Technical Details**: 편향 측정에는 동등성과 사실 기반 기준을 모두 고려한 새로운 메트릭, '통계적 정렬'이 도입되었습니다. 연구진은 성별과 연령 도메인을 포함하여 다양한 이항 그룹을 설정하고, 다양한 질문에 대한 응답을 통해 내재된 편향을 평가합니다. 편향 평가를 위해 '균형', '거부', '통계적으로 정렬된' 세 가지 비편향 상태를 정의하였습니다.

- **Performance Highlights**: 실험 결과, 응답자들은 LLM의 출력이 실제 세계의 인구 통계 분포와 가까울수록 긍정적으로 인식하는 경향이 있음을 보여주었습니다. 평가된 여러 LLM의 편향은 적용된 기준에 따라 달라졌으며, 이는 다양한 관점에서 편향을 평가하는 중요성을 강조합니다. 이 결과는 사실 기반 기준이 LLM 편향 평가에서 보다 일관되고 객관적인 기준을 제공할 수 있음을 시사합니다.



### PIM-AI: A Novel Architecture for High-Efficiency LLM Inferenc (https://arxiv.org/abs/2411.17309)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLM)을 위한 새로운 DDR5/LPDDR5 PIM 아키텍처인 PIM-AI를 소개합니다. 기존 메모리 컨트롤러나 DDR/LPDDR 메모리 PHY를 수정할 필요 없이 설계된 이 아키텍처는 LLM 추론을 가능하게 합니다. PIM(Processing-in-Memory) 기술을 통해 데이터 전송 병목 현상을 줄이고, 전력 효율성을 높이는 여러 이점을 제공합니다.

- **Technical Details**: PIM-AI는 메모리 칩에 직접 계산 단위를 통합하여 구현됩니다. 이를 통해 전통적인 하드웨어 아키텍처가 직면하는 컴퓨팅 및 메모리 요구 사항 문제를 해결할 수 있습니다. 이 논문은 PIM-AI의 성능을 평가하기 위해 다양한 시나리오에서 사용할 수 있는 시뮬레이터를 개발하였습니다.

- **Performance Highlights**: PIM-AI는 클라우드 기반 시나리오에서 사용되는 경우 쿼리 당 총 소유 비용(TCO)을 기존 GPU에 비해 최대 6.94배까지 줄여줍니다. 모바일 시나리오에서도 PIM-AI는 최신 모바일 시스템 온 칩(SoC)에 비해 토큰당 에너지를 10배에서 20배까지 절약할 수 있으며, 이는 배터리 수명을 연장하고 충전 시 더 많은 추론을 가능하게 합니다.



### Meaningless is better: hashing bias-inducing words in LLM prompts improves performance in logical reasoning and statistical learning (https://arxiv.org/abs/2411.17304)
- **What's New**: 이 논문에서는 'hashing'이라는 새로운 방법을 소개합니다. 이 방법은 대규모 언어 모델(LLMs)에서 편향을 유발할 수 있는 단어를 해시와 유사한 무의미한 식별자로 마스킹함으로써 인지적 편향(cognitive biases)과 외부 지식에 대한 의존도를 줄이는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구는 총 490개의 프롬프트를 포함하는 세 가지 실험 세트에서 테스트되었습니다. 카이제곱 검정(chi-square tests)을 사용한 통계 분석에서 모든 테스트된 시나리오에서 유의미한 개선이 나타났으며, LLama, ChatGPT, Copilot, Gemini 및 Mixtral 모델이 포함되었습니다.

- **Performance Highlights**: 해싱은 '린다(Linda)' 문제의 수정된 버전에서는 오류율을 감소시켰으며, 아이템 집합 추출(task)에서도 성능을 향상시켰습니다. 세 번째 실험에서는 린다 문제가 텍스트보다 표 형식으로 제공될 때에도 해싱이 효과적이라는 것을 발견했습니다. 전반적으로 이 방법은 편향 감소와 외부 지식 통합을 개선하는 것으로 나타났으나, 환각(hallucination) 비율은 모델 유형에 따라 일관되게 감소하지 않았습니다.



### ER2Score: LLM-based Explainable and Customizable Metric for Assessing Radiology Reports with Reward-Control Loss (https://arxiv.org/abs/2411.17301)
- **What's New**: 이번 연구에서는 자동 방사선 보고 생성(Automated Radiology Report Generation, R2Gen)의 평가 지표로 ER2Score를 제안합니다. ER2Score는 사용자 정의 기준에 맞춰 평가 기준을 조정할 수 있는 자동 평가 지표로, 기계 학습 모델의 보상 메커니즘을 활용하여 설계되었습니다. 이 평가 지표는 보고서의 전반적인 점수와 각각의 평가 항목에 대한 세부 점수를 제공하여 해석 가능성을 높입니다.

- **Technical Details**: ER2Score는 GPT-4를 활용하여 LLM(대형 언어 모델) 기반의 보상 모델을 훈련시키며, 보고서의 품질을 높이기 위해 '수용' 및 '거부' 샘플을 쌍으로 만들어 학습합니다. 이 과정에서, 정밀한 보상을 제공하는 맞춤형 손실 함수를 사용하여 모델이 다수의 평가 기준에 대해 개별 보상을 동시에 출력하도록 합니다. 특히, 다양한 평가 시스템을 지원하기 위한 유연한 훈련 방법론을 채택했습니다.

- **Performance Highlights**: 실험 결과 ER2Score는 기존 평가 지표들보다 사람의 평가와의 상관 관계가 더 높으며, 모델 선택에서 더 우수한 성능을 보였습니다. 이번 연구에서 제안한 ER2Score는 다양한 평가 기준에 맞춰 맞춤 분석이 가능하여 보고서 생성 품질을 향상시킬 수 있는 잠재력을 가지고 있음을 보여줍니다. 사용자가 특정 기준을 통해 평가 결과의 특정 측면을 파악할 수 있도록 지원하는 점도 중요한 장점입니다.



### GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers (https://arxiv.org/abs/2411.17296)
Comments:
          13 pages, 6 figures, 7tables

- **What's New**: 이 논문에서는 Graph Transformers (GTs)의 한계를 극복하기 위해 Graph Fourier Kolmogorov-Arnold Transformers (GrokFormer)라는 새로운 모델을 제안한다. GrokFormer는 저주파 신호에만 집중하는 GTs의 자가 주의 메커니즘을 넘어서 다양한 주파수 신호를 효과적으로 캡처할 수 있도록 설계되었다. 이 모델은 Fourier 급수 모델링을 통해 학습 가능한 활성화 함수를 활용하여 복잡한 노드 레이블 패턴을 포착할 수 있는 능력을 갖추었다.

- **Technical Details**: GrokFormer는 K차 그래프 스펙트럼에서 학습 가능한 활성화 함수를 통해 고유값 기반 필터 함수를 학습하며, 이를 통해 다양한 주파수 신호를 유연하게 캡처한다. 이는 고차 스펙트럼 정보를 적응적으로 추출함으로써 이루어지며, GrokFormer는 이러한 방식을 통해 복잡한 주파수 반응을 효과적으로 모델링할 수 있다. 결과적으로, GrokFormer는homophilic 및 heterophilic 패턴을 모델링하는 데 효과적인 능력을 발휘한다.

- **Performance Highlights**: GrokFormer는 다양한 도메인과 수준에서의 10개의 노드 분류 데이터셋과 5개의 그래프 분류 데이터셋에서 광범위한 실험을 수행하여 기존의 GTs 및 기타 선진 그래프 신경망(GNNs)을 지속적으로 초월하는 우수한 성능을 보여주었다. 특히 GrokFormer는 다양한 레벨의 homophily를 가진 그래프에서 높은 정확도를 달성하며, 이는 실제 애플리케이션에서의 효과적인 적용 가능성을 암시한다.



### Social Distancing Induced Coronavirus Optimization Algorithm (COVO): Application to Multimodal Function Optimization and Noise Remova (https://arxiv.org/abs/2411.17282)
- **What's New**: 이 논문은 사회적 거리두기를 기반으로 한 새로운 생물 영감(metaheuristic optimization) 최적화 모델인 COVID-19 최적화 알고리즘(COVO)을 제안합니다. COVO 알고리즘은 코로나19 전염병을 억제하기 위한 사회적 거리두기의 효과를 반영하여 설계되었습니다. 이 알고리즘은 여러 복잡한 최적화 문제를 해결함으로써 글로벌 솔루션(global solution)을 찾는 것을 목표로 합니다.

- **Technical Details**: COVO는 13개의 벤치마크 함수(benchmark functions)를 통해 이산(discrete), 연속(continuous), 그리고 복합(complex) 문제에 대한 성능을 평가합니다. 이 모델은 빠른 수렴(convergence)을 목표로 하며, 기존의 잘 알려진 최적화 알고리즘과 비교됩니다. 나아가, COVO의 구조와 작동 방식은 코로나의 확산 속도를 줄이는 사회적 거리두기 개념에 뿌리를 두고 있습니다.

- **Performance Highlights**: 검증된 결과에 따르면, 제안된 COVO 최적화 모델은 합리적이고 수용 가능한 성능을 보여줍니다. 특히 COVO는 복잡한 문제 해결에서 우수한 성과를 보이며, 다양한 응용 분야에서 글로벌 솔루션 획득에 기여할 수 있음을 시사합니다. 이 논문의 연구 결과는 공공 건강 시스템의 부담을 경감하는데 기여할 수 있는 가능성을 보여줍니다.



### HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator (https://arxiv.org/abs/2411.17261)
- **What's New**: AIGC 이미지의 품질 문제를 해결하기 위한 새로운 접근법이 제안됩니다. HEIE라는 계층적 설명 가능 이미지 불확실성 평가자(MLLM-Based Hierarchical Explainable image Implausibility Evaluator)가 소개되며, 이는 CoT(Chain of Thought) 기반의 설명 가능한 시스템을 활용하여 불확실성 지역의 평가와 더불어 구체적인 결함 영역을 식별할 수 있게 합니다. 이 연구는 기존의 평가 방식의 한계를 극복하고자 합니다.

- **Technical Details**: HEIE는 Adaptive Hierarchical Implausibility Mapper와 CoT-Driven Explainable Trinity Evaluator를 통합하여 더욱 정교한 이미지 평가를 수행합니다. Adaptive Hierarchical Implausibility Mapper는 ViT(Vision Transformer)의 저수준 이미지 특징과 MLLMs의 고수준 맵 토큰을 결합하여 정밀한 열 지도 예측이 가능합니다. 또한, Expl-AIGI-Eval이라는 새로운 데이터셋을 구축하여 AIGC 이미지의 해석 가능성과 불확실성 평가를 지원합니다.

- **Performance Highlights**: 본 연구의 방법론은 다양한 데이터셋과 태스크에서 최첨단 성능을 달성하였습니다. HEIE는 불확실성 기반의 적응형 토큰 접근법을 적용하여 정밀한 결함 위치 지정을 가능하게 하고, 새로운 데이터셋을 통한 평가로 기존 방법보다 향상된 해석 가능성을 보여줍니다. 지속적인 비교 및 절제 연구가 이 방법의 효과성을 입증하였습니다.



### MiceBoneChallenge: Micro-CT public dataset and six solutions for automatic growth plate detection in micro-CT mice bone scans (https://arxiv.org/abs/2411.17260)
Comments:
          Under Review

- **What's New**: 이 논문은 미세 CT 스캔을 통한 자동 뼈 정량화 모델 개발과 관련된 도전 과제를 다룹니다. 이를 위해, 83마리 생쥐의 3D μCT 뼈 스캔으로 구성된 고품질 데이터셋을 준비하고 주석을 달았습니다. 전 세계 80명 이상의 AI 과학자들이 이 도전에 참여하여, 뼈 성장 판을 효과적으로 식별할 수 있는 6개의 컴퓨터 비전 솔루션을 개발했습니다.

- **Technical Details**: 뼈 조직의 연속적인 활성 재형성은 조골세포(osteoblasts)와 파골세포(osteoclasts)의 복잡한 상호작용에 의해 조절됩니다. μCT(미세 컴퓨터 단층 촬영)는 뼈의 미세 구조를 고해상도로 평가할 수 있는 유용한 기술로, 이미지를 통해 뼈 미네랄 밀도를 정량화할 수 있습니다. 본 연구에서는 높은 차원성의 3D 데이터를 효과적으로 처리하기 위해 2D 슬라이스를 사용하는 방법으로 모델을 안정화하며 정보 손실을 최소화하는 여러 가짜 3D(pseudo-3D) 방법들이 제안되었습니다.

- **Performance Highlights**: 개발된 솔루션들은 평균 절대 오차가 1.91±0.87 평면으로, 방사선 전문의가 실용적으로 사용할 수 있는 정확도 수준에 도달했습니다. 연구에서 공유한 고품질 3D μCT 데이터셋과 함께, 모든 코드와 모델들이 공개되며 이는 연구자들이 자체적인 접근법을 개발하고 벤치마크할 수 있는 기회를 제공합니다. 이러한 작업을 통해, 자동화된 뼈 정량화 프로세스가 약물 개발 연구의 효율성을 크게 향상시킬 것으로 기대됩니다.



### APT: Architectural Planning and Text-to-Blueprint Construction Using Large Language Models for Open-World Agents (https://arxiv.org/abs/2411.17255)
Comments:
          8 pages

- **What's New**: APT는 Minecraft 환경 내에서 자율 에이전트가 복잡하고 창의적인 구조물을 구축할 수 있도록 설계된 진보된 대형 언어 모델(LLM) 기반 프레임워크입니다. 기존의 접근 방식이 주로 기술 기반 오픈 월드 작업에 집중하거나 이미지 기반 확산 모델을 사용한 것과는 달리, 우리의 방법은 LLM의 내재적 공간 추론 능력을 활용합니다.

- **Technical Details**: APT는 사고의 사슬(chain-of-thought) 분해 방법과 다중 모달(multimodal) 입력을 사용하는 프레임워크로, 제로샷(zero-shot) 또는 몇 샷(few-shot) 학습 시나리오에서 실행 가능한 세부적인 건축 레이아웃 및 도면을 생성합니다. 에이전트는 메모리 및 반영(reflection) 모듈을 통합하여 평생 학습(lifelong learning), 적응적 개선 및 건설 프로세스 전반에 걸친 오류 수정을 지원합니다.

- **Performance Highlights**: 다양한 GPT 기반 LLM 백엔드와 에이전트 구성을 사용한 실험 결과는 에이전트가 많은 항목과 그 위치, 방향을 포함한 광범위한 지침을 정확히 해석할 수 있는 능력을 보여줍니다. 메모리 모듈의 포함은 성능의 현저한 향상을 초래하며, 이는 지속적인 학습과 축적된 경험의 재사용을 가능하게 합니다. 또한, 에이전트의 예상치 않은 비계(scaffolding) 행동의 출현은 LLM 기반 에이전트가 인간과 유사한 문제 해결 기술을 자율적으로 개발할 수 있는 잠재력을 강조합니다.



### Semantic Data Augmentation for Long-tailed Facial Expression Recognition (https://arxiv.org/abs/2411.17254)
- **What's New**: 이번 연구에서는 Facial Expression Recognition (FER)의 데이터 세트 불균형 문제를 해결하기 위한 새로운 방법인 세멘틱 증강(semantic augmentation) 방법을 제안합니다. 기존 연구들이 Long-Tailed Recognition을 위한 데이터 증강(data augmentation)에 초점을 맞췄다면, 본 논문은 VAE-GAN의 잠재 공간(latent space)에서 소스 데이터의 인코딩에 무작위성을 도입하여 새로운 샘플을 생성하는 방식을 적용했습니다.

- **Technical Details**: 본 연구의 핵심 기술은 VAE-GAN(Variational Autoencoder with Generative Adversarial Network) 기반의 라벨링된 데이터 증강입니다. 모델은 원본 데이터의 잠재 표현을 무작위화(randomness)하여, 데이터 세트의 롱테일(long-tailed) 분포를 균형화하는 새로운 샘플을 생성합니다. 이를 통해 수집된 RAF-DB 데이터셋에서 Facial Expression Recognition 성능을 개선할 수 있습니다.

- **Performance Highlights**: 우리의 제안된 증강 방법은 Facial Expression Recognition 뿐만 아니라 다양한 데이터 소모(data-hungry) 시나리오에서도 성능을 발휘할 수 있습니다. 실험 결과, 단순한 데이터 증강 기법보다 더 높은 정확도(accuracy)를 보였으며, 이는 실제 응용에서의 효과성을 높입니다.



### Buffer Anytime: Zero-Shot Video Depth and Normal from Image Priors (https://arxiv.org/abs/2411.17249)
- **What's New**: Buffer Anytime은 비디오로부터 깊이 및 노말 맵을 추정하기 위한 새로운 프레임워크로, 비디오-깊이 및 비디오-노말 교육 데이터의 필요성을 제거했습니다. 기존의 대규모 주석 처리된 비디오 데이터셋에 의존하는 대신, 단일 이미지 프라이어를 활용하여 고품질 비디오 버퍼 추정을 시연합니다. 이는 전통적인 데이터 쌍 없이도 이미지 기반 모델의 효능을 이용하여 비디오의 기하학적 버퍼 생성을 효과적으로 지원합니다.

- **Technical Details**: 본 연구에서는 RGB 비디오 입력을 기반으로 깊이 맵과 노말 맵을 예측하는 문제를 설정합니다. 기존의 쌍 데이터 세트로 훈련되는 비디오 깊이 예측 모델들과 달리, 우리의 방법은 RGB 비디오 데이터만을 사용하여 기하학적 버퍼를 생성합니다. 우리는 경량화된 시간적 주의 아키텍처에 기반한 하이브리드 손실 함수와 이미지를 훈련시키는 기법을 결합하여 시간적 일관성(temporal consistency)과 정확성을 보장합니다.

- **Performance Highlights**: 우리의 방법은 이미지 기반 접근 방식들을 크게 초월하며, 대규모 비디오 데이터셋에서 훈련된 최신 비디오 모델들과 비견되는 성능을 달성합니다. Depth Anything V2 및 Marigold-E2E-FT와 같은 최첨단 모델에 적용하여 시간적 일관성을 크게 향상시켰습니다. 실험 결과는 비디오 기하학적 버퍼 추정 평가에서 두 모델이 획기적인 개선을 이루었음을 보여줍니다.



### From Graph Diffusion to Graph Classification (https://arxiv.org/abs/2411.17236)
- **What's New**: 이 논문은 그래프 도메인에서의 분류 작업에 대한 최근의 발전을 다룹니다. 주목할 점은 기존의 이미지 및 텍스트 도메인을 넘어, 그래프 분류를 위한 새로운 경험적 접근법을 제시했다는 것입니다. 특히, score-based 그래프 확산 모델을 사용하는 방법을 통해 기존의 분류 방법에 대한 경쟁력을 보여주고 있습니다.

- **Technical Details**: 논문에서는 새로운 훈련 목적을 개발하여, 그래프 확산 모델의 정확도를 높이는 데 초점을 맞추었습니다. 특히, 그래프 분류를 위해 설계된 이 목적은 모델 성능 향상뿐만 아니라 고품질 그래프 생성을 가능하게 합니다. 또한, 훈련 및 추론 과정에서 인접 행렬을 무작위로 샘플링하는 방식으로Permutation Invariance 문제를 해결합니다.

- **Performance Highlights**: 세 가지 훈련 손실 목표를 기반으로 한 실험을 통해, IMDB-BINARY 및 PROTEIN와 같은 실제 데이터셋에서 우수한 분류 성능을 입증하였습니다. 연구 결과, 새로운 discriminative training objective와 inference method의 조합이 최적의 성능을 달성하며, SOTA (state-of-the-art) 성능을 기록하는 결과를 보였습니다.



### GraphSubDetector: Time Series Subsequence Anomaly Detection via Density-Aware Adaptive Graph Neural Network (https://arxiv.org/abs/2411.17218)
- **What's New**: 이 논문에서는 GraphSubDetector라는 새로운 접근 방식을 통해 시계열 subsequence anomaly detection을 제안합니다. 기존 방법들이 갖는 subsequence 길이 선택의 어려움과 이상치 탐지의 복잡성을 해결하기 위해, 이 방법은 특성과 변동성을 강조하는 적응형 길이 선택 메커니즘을 사용합니다. 또한 density-aware adaptive graph neural network (DAGNN)를 통해 서로 다른 subsequence 간의 메시지 패싱을 통해 이질적이고 robust한 표현을 생성합니다.

- **Technical Details**: GraphSubDetector는 시계열 데이터의 복잡한 동적 특성을 이해하기 위해 temporal convolution network (TCN) 기반 기능 인코더를 도입합니다. 이 기능 인코더는 멀티 길이의 관점에서 representations를 생성하며, 이루어지는 추천 메커니즘을 통해 정상 및 비정상 데이터의 pattern을 더 잘 반영하도록 적합한 길이를 결정합니다. 또한, 적절히 설계된 인접 행렬을 통해 비슷한 subsequence의 메시지 패싱을 최적화하여 정상 데이터의 변동성을 줄이고, 비정상을 강조합니다.

- **Performance Highlights**: 여러 실세계 TSAD 벤치마크 데이터셋에서 실험한 결과, GraphSubDetector는 기존의 최첨단 알고리즘에 비해 우수한 성능을 보였습니다. 이 방법은 다양한 이상치 subsequence를 효과적이고 효율적으로 탐지할 수 있으며, 실제 응용에서도 매우 유용합니다. 또한, 데이터 크기 증가에 따른 계산 복잡도가 거의 선형적으로 증가하여, 고성능의 탐지 능력을 유지합니다.



### Strategic Prompting for Conversational Tasks: A Comparative Analysis of Large Language Models Across Diverse Conversational Tasks (https://arxiv.org/abs/2411.17204)
Comments:
          37 pages, 12 tables

- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Models, LLM)인 Llama, OPT, Falcon, Alpaca 및 MPT의 성능을 종합적으로 평가한 연구를 제시합니다. 다양한 대화 과제를 포함한 이 연구에서는 특히 예약, 공감 반응 생성, 정신 건강 및 법률 상담, 설득, 협상 등의 분야에서 모델의 기능과 한계를 분석합니다. 여러 평가 기준을 통해 철저하게 검증하여 특정 작업에 따라 모델의 성능이 다르게 나타난다는 결과를 도출하였습니다.

- **Technical Details**: 이 연구에서는 음성 인식 및 자연어 처리(Natural Language Processing, NLP) 작업에서 LLM의 성능을 높은 정확도로 평가하기 위해 자동 및 인적 평가 기준을 포함한 정교한 테스트 설정을 사용했습니다. 각 모델의 성능을 올바르게 게이지하기 위해 일반적인 메트릭과 특정 작업에 적합한 메트릭을 활용합니다. 검증을 통해 특정 작업에 맞춰 적절한 LLM을 선택하는 것이 중요함을 강조하고 있습니다.

- **Performance Highlights**: 각 LLM은 특정 과제에서 최상의 성능을 보였으나, 모든 과제에서 보편적으로 최적의 모델은 없으며, 성능은 작업의 요구 사항에 따라 크게 달라진다는 점이 시사됩니다. 실험 결과, 일부 모델이 특정 작업에서 뛰어난 성능을 나타냈지만, 다른 작업에서는 상대적으로 낮은 성능을 보이는 경향이 있었습니다. 이는 대화 애플리케이션에서 가장 적합한 LLM을 선택할 때 작업의 특정 요구 사항과 특성을 고려해야 함을 강조합니다.



### Learning Hierarchical Polynomials of Multiple Nonlinear Features with Three-Layer Networks (https://arxiv.org/abs/2411.17201)
Comments:
          78 pages, 4 figures

- **What's New**: 이번 연구는 deep learning 이론에서 신경망이 계층적 특성을 어떻게 학습하는지를 탐구합니다. 연구의 핵심은 다중 비선형 특성의 계층적 다항식 학습에 대한 새로운 접근 방식을 제시하는 것입니다. 이는 이전 연구에서 주로 다루었던 단일 비선형 특성을 초월하여, 복잡한 함수 구성을 더 깊이 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구에서는 세 층의 신경망을 이용하여 $f^{	extstar}=g^{	extstar}ullet p$ 형태의 함수를 분석합니다. 여기서 $p:	extbf{R}^{d} 

- **Performance Highlights**: 세 층 신경망이 비선형 특성이 span하는 공간을 완전히 복구하며, 목표 함수 $f^{	extstar}=g^{	extstar}ullet p$의 효율적인 학습을 달성합니다. 연구는 이러한 학습 과정이 $	ilde{	ext{O}}(d^4)$ 샘플과 다항식 시간 내에 가능하다는 것을 보여줍니다. 또한, kernel methods의 샘플 복잡성 ${	heta}(d^{2p})$를 크게 개선하여 딥러닝의 효율적인 특성 학습의 잠재력을 강조합니다.



### ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting (https://arxiv.org/abs/2411.17176)
- **What's New**: 이 논문은 텍스트-이미지 변환(T2I) 생성 모델의 자동화를 통해 사용자가 쉽게 원하는 이미지를 생성할 수 있도록 하는 방법을 제안합니다. 자동 T2I 생성(Automatic T2I)이라는 개념을 도입하여 사용자가 단순히 필요를 설명하면 자동으로 이미지 생성에 필요한 모든 단계를 수행하도록 합니다. 이를 위해 ChatGenBench라는 새로운 벤치마크를 소개하며, 다양한 프리스타일 입력에 대한 평가를 지원합니다.

- **Technical Details**: 제안된 방법인 ChatGen-Evo는 다단계 진화 전략(multi-stage evolution strategy)을 활용하여 모델이 자동화 기능을 점진적으로 습득할 수 있도록 합니다. 이 과정은 프리스타일 입력을 고품질 프롬프트로 변환하는 Supervised Fine-Tuning(SFT), 모델 선택을 위한 모델 토큰(ModelTokens) 형성, 프롬프트와 선택된 모델에 기반해 인자를 설정하는 단계를 포함합니다. 이를 통해 모델은 사용자 기대에 부합하는 출력을 생성할 수 있도록 훈련됩니다.

- **Performance Highlights**: ChatGen-Evo는 여러 기준에서 기존 방법들보다 현저히 우수한 성능을 보입니다. 정확한 단계별 평가와 이미지 품질 평가를 통해 다단계 진화 전략의 중요성을 강조하며, 다양한 실험을 통해 자동 T2I의 도전 과제 및 기회를 밝혀냈습니다. 이 논문은 자동 T2I의 가능성을 높이는 몇 가지 유망한 방향도 제시합니다.



### Learning Monotonic Attention in Transducer for Streaming Generation (https://arxiv.org/abs/2411.17170)
Comments:
          Codes: this https URL

- **What's New**: 이 논문은 Streaming generation 분야에서 Transducer 구조를 개선하기 위한 새로운 접근 방식을 제안합니다. 입력 스트림의 역사(history)를 통해 디코딩을 조정하는 learnable monotonic attention 메커니즘을 통합하여, 비순차적인 정렬(non-monotonic alignment) 문제를 해결하는 것을 목표로 합니다. 이 방법론은 forward-backward 알고리즘을 활용하여 예측자 상태(predictor states)와 입력 타임스탬프 간의 posterior probability를 추론합니다.

- **Technical Details**: 논문에서 제안하는 MonoAttn-Transducer는 Transducer 모델의 디코딩 과정에서 비순차적 정렬을 처리하는 성능을 크게 향상시키는 기술을 포함합니다. 즉, 예측자가 실시간으로 소스 히스토리를 활용할 수 있게 하여, attention의 범위를 예측에 따라 적응적으로 조정합니다. 이러한 방식으로 학습하는 동안 기하급수적으로 큰 정렬 공간을 열거할 필요가 없으며, 이는 훈련 과정의 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, MonoAttn-Transducer는 음성-텍스트 및 음성-음성 동시 번역 작업에서 생성 품질을 눈에 띄게 개선하면서도 지연(latency) 증가를 피하는 놀라운 성능을 보여주었습니다. 특히 높은 비순차성 수준을 가진 샘플을 처리하는 데 강력한 효과를 나타내며, 이를 통해 종합적인 Streaming generation 작업의 복잡성을 효과적으로 다룰 수 있는 가능성을 제시합니다.



### Self-reconfiguration Strategies for Space-distributed Spacecraf (https://arxiv.org/abs/2411.17137)
- **What's New**: 이 논문은 다양한 기능을 가진 모듈을 궤도에서 조립하여 특정 기능을 가진 우주선 구조를 형성하는 분산형 궤도 우주선 조립 알고리즘을 제안합니다. 이러한 접근 방식은 재구성 가능성, 빠른 임무 응답 및 용이한 유지보수라는 장점을 가지고 있습니다. 분산형 우주선의 이점을 실현하기 위해 합리적이고 효율적인 궤도 자가 재구성 알고리즘이 핵심적인 역할을 합니다.

- **Technical Details**: 이 논문은 모듈 처리 순서에 대한 전략 학습을 위해 모방 학습(imitation learning)과 강화 학습(reinforcement learning) 프레임워크를 채택합니다. 또한, 로봇 팔의 운동 알고리즘을 설계하여 처리 순서를 실행합니다. 모듈 표면에 지도를 생성하고, A* 알고리즘을 사용하여 로봇 팔의 경로 점 계획을 완성함으로써 자가 재구성 처리 작업을 수행합니다.

- **Performance Highlights**: 최종적으로 로봇 팔의 공동 계획은 순방향 및 역방향 기구학(forward and backward kinematics)을 통해 완수됩니다. 결과는 Unity3D에서 시각적으로 표현되어 성능과 효율성을 강조하고 있습니다. 이러한 프레임워크는 향후 다양한 우주선 임무에서의 적용 가능성을 보여주고 있습니다.



### DOGE: Towards Versatile Visual Document Grounding and Referring (https://arxiv.org/abs/2411.17125)
Comments:
          20 pages, 13 figures

- **What's New**: 최근 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 보다 세밀한 이해 및 유연한 사용자 상호작용을 달성하기 위해 그라운딩(growing) 및 참조(referring) 능력에 중점을 두고 있습니다. 그러나, 시각적 문서 이해 분야에서는 고품질의 미세한 데이터셋과 종합적인 벤치마크의 부족으로 이 능력이 부족합니다. 이를 해결하기 위해 우리는 DOGE-Engine이라는 문서 그라운딩 및 참조 데이터 엔진을 제안하며, 고품질의 두 가지 유형의 문서 데이터를 생성합니다.

- **Technical Details**: 개발된 DOGE-Engine은 140만 개의 다중 세분화 문서 구문 데이터와 70만 개의 다양한 지침 조정 데이터(instruction-tuning data)를 생성합니다. 이 데이터는 포스터, 차트 및 PDF 문서에서 단어, 구, 줄, 단락 및 전체 페이지 수준의 텍스트 상자 주석을 포함하여 기본 텍스트 지역화 및 인식 기능을 향상시키는 데 사용됩니다. 또한, DOGE-Bench를 통해 3가지 문서 유형(차트, 포스터, PDF 문서)에서 7개의 그라운딩 및 참조 작업을 포함한 종합 평가가 가능합니다.

- **Performance Highlights**: DOGE라는 강력한 벤치마크 모델을 통해 다중 세분화 문서 이미지 내에서 텍스트를 정확하게 참조하고 그라운딩할 수 있는 가능성을 제공합니다. DOGE-Bench에서의 성과는 향후 연구에 대한 기준 역할을 하며, 우리의 코드는 데이터와 모델은 커뮤니티 개발을 위해 오픈소스될 예정입니다. DOGE-Engine은 MLLM의 문서 이해 능력을 높이기 위한 세 가지 주요 기여를 통해 문서 그라운딩 및 참조 능력을 혁신적으로 개선하는 데 기여하고 있습니다.



### Advancing Content Moderation: Evaluating Large Language Models for Detecting Sensitive Content Across Text, Images, and Videos (https://arxiv.org/abs/2411.17123)
Comments:
          55 pages, 16 figures

- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 기반의 콘텐츠 검열 솔루션을 탐구하며, OpenAI의 moderation model과 Llama-Guard-3 등 기존 방법들의 성능을 평가합니다. 특히, 최신 LLM(GPT-4o, Gemini 1.5, Llama-3)을 통해 부적절한 텍스트 및 시각 콘텐츠를 감지하는 가능성을 탐구하며, 이미지와 비디오의 민감한 내용을 인식하기 위해 LLM의 비전 기능을 활용합니다.

- **Technical Details**: 연구에서는 LLM을 사용하여 부적절한 텍스트 및 비주얼 콘텐츠를 식별하는 시스템을 제안합니다. 다양한 텍스트 및 비주얼 데이터 세트를 생성하고 평가하였으며, 여기에는 폭력 비디오, 누드 및 포르노그래피가 포함된 이미지, 다양한 뉴스 아티클 및 X 트윗이 포함됩니다. 평가 질문(RQ1~RQ4)은 기존 콘텐츠 검열 모델의 효율성과 LLM의 성능의 비교를 목표로 하고 있습니다.

- **Performance Highlights**: LLM 기반의 콘텐츠 검열 솔루션은 기존의 전통적인 기술에 비해 높은 정확도와 낮은 거짓 긍정 및 거짓 부정 비율을 달성하며, 이를 통해 웹사이트 및 소셜 미디어 플랫폼에서 콘텐츠 규제 및 검열을 효과적으로 수행할 가능성을 보여줍니다. 연구 결과는 이러한 모델들이 다양한 종류의 부적절한 콘텐츠(증오 발언, 폭력, 성적 내용)를 감지하는 데 있어 일관된 예측률을 갖는다는 것을 입증하였습니다.



### Star Attention: Efficient LLM Inference over Long Sequences (https://arxiv.org/abs/2411.17116)
Comments:
          Code: this https URL

- **What's New**: Star Attention은 두 단계의 블록 희소 근사를 통해 긴 시퀀스에서의 추론을 향상시키는 새로운 알고리즘입니다. 이 방법은 여러 호스트에 걸쳐 주의를 분산시키며 통신 비용을 최소화하여 계산 효율성을 개선합니다. Star Attention은 기존 Transformer 기반 LLM과의 호환성을 가지며, 추가적인 모델 파인튜닝 없이 사용 가능합니다.

- **Technical Details**: Star Attention은 두 단계로 되어 있습니다: 첫 번째 단계에서는 컨텍스트를 블록으로 나누어 여러 호스트에 분산 처리하고, 두 번째 단계에서는 쿼리와 응답 토큰이 모든 이전 캐시된 토큰을 대상으로 주의를 기울입니다. 이 알고리즘은 각 호스트의 할당된 블록에 대한 self-attention만을 계산하여 복잡도를 제곱에서 선형으로 감소시킵니다. 또한 쿼리 호스트는 전 세계적으로 결과를 집계하여 기존 KV 캐시를 업데이트합니다.

- **Performance Highlights**: Star Attention은 Llama3.1-8B 및 Llama3.1-70B 모델에서 95-100%의 정확도를 유지하면서 최대 11배 빠른 추론을 달성합니다. 또한 Flash Attention이나 KV 캐시 압축과 같은 다른 LLM 최적화 기법과 결합하여 추가적인 성능 향상을 가능하게 합니다.



### Contrastive CFG: Improving CFG in Diffusion Models by Contrasting Positive and Negative Concepts (https://arxiv.org/abs/2411.17077)
Comments:
          14 pages, 8 figures

- **What's New**: 본 논문에서는 Classifier-Free Guidance (CFG)의 부정적 가이던스 기법을 개선하기 위해 ‘contrasting loss’를 활용한 새로운 접근 방식을 제안합니다. 이 방법론은 주어진 조건에 따라 노이즈 제거 방향을 정렬하거나 반발시키는 방식으로 작동하여, 기존의 부정적 가이던스 방법의 한계를 극복합니다. 실험 결과, 제안된 방법이 다양한 시나리오에서 바람직하지 않은 개념을 효과적으로 제거하면서 샘플 품질을 유지하는 것을 보여주었습니다.

- **Technical Details**: 제안된 방법은 Contrastive CFG(CCFG)라고 불리며, 이는 주어진 조건에 맞추어 노이즈 제거 방향을 최적화하는 방식입니다. CCFG는 기존의 CFG와 유사한 가이던스를 생성하면서도 부정적 가이던스의 단점을 보완합니다. 이 과정에서 샘플링 절차와 관련된 상당한 계산 오버헤드를 피하면서 노이즈 제거 방향을 자동으로 조절합니다.

- **Performance Highlights**: CCFG는 여러 실험을 통해 바람직하지 않은 개념을 성공적으로 회피하면서 샘플 품질을 보장하는 것을 입증하였습니다. 특히, 단순한 클래스 조건부터 복잡하고 중첩된 텍스트 프롬프트에 이르기까지 다양한 시나리오에서 효과성을 보여주었습니다. 이로 인해, 동적이고 복잡한 조건에서의 샘플링 효과를 향상시키는데 기여할 수 있습니다.



### Path-RAG: Knowledge-Guided Key Region Retrieval for Open-ended Pathology Visual Question Answering (https://arxiv.org/abs/2411.17073)
- **What's New**: 이 논문에서는 암 치료 선택과 계획에 필수적인 병리 이미지를 통한 질병 진단 및 예후를 다룹니다. 최근 딥러닝(Deep Learning) 접근방법이 복잡한 병리 이미지를 분석하는 데 사용되고 있으나, 임상의(Clinician) 전문가의 조직 구조와 세포 성분에 대한 이해를 종종 간과하는 문제점이 있음을 지적합니다. 이에 따라, Path-RAG라는 새로운 프레임워크를 제안하며, 이는 HistoCartography를 통해 병리 이미지에서 관련 도메인 지식을 검색하여 PathVQA-Open 성능을 크게 향상시킵니다.

- **Technical Details**: Path-RAG는 병리 이미지 분석의 복잡성을 인정하고, HistoCartography를 이용하여 특정 패치(Patch)를 선택하는 인본주의적 AI(Human-centered AI) 접근 방식을 취합니다. 실험 결과, 이 방법이 LLaVA-Med의 정확도를 38%에서 47%로 높일 수 있으며, H&E로 염색된 병리 이미지에서 PathVQA-Open 데이터셋에서 28%의 주목할 만한 향상을 보여주었습니다. 또한, 긴 질문 및 답변 쌍에서는 ARCH-Open PubMed와 ARCH-Open Books에서 각각 32.5%와 30.6%의 일관된 개선을 달성했습니다.

- **Performance Highlights**: 제안된 Path-RAG 프레임워크는 병리 이미지 분석에 있어 뛰어난 성능을 입증했습니다. 특히, 도메인 지식의 가이드를 통해 성능 개선이 발생했으며, 이는 병리 이미지를 활용한 질문 응답 작업에서 유의미한 결과를 나타냅니다. 연구에 사용된 코드와 데이터셋은 공개되어 있어, 후속 연구자들이 활용할 수 있도록 했습니다.



### Creative Agents: Simulating the Systems Model of Creativity with Generative Agents (https://arxiv.org/abs/2411.17065)
- **What's New**: 이번 연구에서는 기존의 생성 AI 모델이 질과 성능 면에서 빠르게 발전하는 반면, AI의 '창의성'을 배양하는 데 많은 주목이 기울어지지 않고 있다고 지적합니다. 이를 위해 Csikszentmihalyi의 시스템 모델을 바탕으로 대규모 언어 모델을 활용한 가상 에이전트를 통해 창의성의 시스템 모델을 구현하고 시뮬레이션 하였습니다. 연구는 고립된 상태와 다중 에이전트 시스템을 비교하여 생성된 결과물의 "창의성"을 분석하였으며, 생성 에이전트들이 이 시스템 모델에서 더 나은 성능을 보일 수 있음을 제안합니다.

- **Technical Details**: 연구는 창의성을 정의하는 데 있어서 Csikszentmihalyi의 시스템 모델에 기초하여, 창의성이 개인의 독립적인 행위가 아니라 사회적 상호작용의 결과임을 강조합니다. 그러나 Boden의 정의는 높은 수준의 추상이 요구되는 반면, Csikszentmihalyi의 접근법은 현재 기술로 보다 용이하게 실현될 수 있는 다수 요소 간의 상호작용으로서 창의성을 설명합니다. 이 연구에서 사용된 시스템 모델은 아티스트, 아티팩트를 전시하는 커뮤니티, 아티팩트가 생성된 맥락으로 구성되어 있음을 설명하며, 각 요소는 서로에게 지속적으로 영향을 미친다고 주장합니다.

- **Performance Highlights**: 연구 결과, 가상의 창작자들이 사회적 상호작용과 함께 작업했을 때 창의성의 변화를 관찰할 수 있었습니다. 특히, 서로 다른 축적된 지식과 경험을 가진 다중 에이전트 시스템이 더 창의적인 아티팩트를 생성하는 경향을 보여주었습니다. 이는 AI가 독립적으로 외로운 작업을 하는 것보다 공동 작업을 통해 더 많은 창의적인 출력을 이끌어낼 수 있다는 중요한 시사점을 제공합니다.



### Graph Structure Learning with Bi-level Optimization (https://arxiv.org/abs/2411.17062)
- **What's New**: 이번 연구에서는 새로운 Generic Structure Extraction with Bi-level Optimization (GSEBO) 방법을 제안합니다. GSEBO는 그래프 구조를 전 세계적 관점에서 최적화하여 노드 분류 문제를 해결합니다. 이를 통해 노드의 특성 맵핑을 위한 전 세계적 정보를 포함한 공통 매개변수를 학습하여 GNN의 성능을 향상시킵니다.

- **Technical Details**: GSEBO는 모든 엣지의 기여를 적절히 반영하기 위해 엣지별(Parameter Sharing) 방식의 그래프 연결 모델링을 사용합니다. 이 방법은 연결 강도와 연결성이라 불리는 두 가지 요소를 고려하여 그래프 구조를 추출합니다. 또한, bi-level optimization 알고리즘을 통해 구조와 기본 그래프 합성곱의 매개변수를 독립적으로 업데이트합니다.

- **Performance Highlights**: GSEBO는 네 가지 실제 데이터셋을 기반으로 한 실험에서 기존의 그래프 구조 학습(GSL) 방법과 비교하여 우수한 성능을 나타냈습니다. 제안된 방법은 강력한 학습 능력과 견고성을 보여주며, 기존 방법들의 한계를 극복하는 데 기여합니다. 이러한 결과는 GSEBO의 효용성을 또한 validate 합니다.



### ThreatModeling-LLM: Automating Threat Modeling using Large Language Models for Banking System (https://arxiv.org/abs/2411.17058)
- **What's New**: 본 논문에서는 금융 시스템을 위한 위협 모델링을 자동화하는 새로운 프레임워크인 ThreatModeling-LLM을 소개합니다. 기존의 위협 모델링 접근 방식의 비효율성을 극복하기 위해 LLMs를 활용하여 데이터 생성, 프롬프트 엔지니어링 및 모델 파인튜닝의 세 단계로 작동합니다. 이 프레임워크는 특히 은행 시스템의 복잡성을 해결하기 위해 특수하게 설계되었으며, NIST 800-53과 같은 준수 기준과의 정합성을 유지하면서 위협 식별 및 완화 생성을 향상시킵니다.

- **Technical Details**: ThreatModeling-LLM은 먼저 Microsoft Threat Modeling Tool(TMT)을 사용하여 벤치마크 데이터셋을 생성합니다. 이후 Chain of Thought(CoT) 및 Optimization by PROmpting(OPRO) 기법을 활용하여 사전 훈련된 LLM의 프롬프트를 최적화하고, Low-Rank Adaptation(LoRA)을 적용하여 모델을 파인튜닝합니다. 이 과정에서 DFD 설명, 식별된 위협, 완화 방안 및 NIST 800-53 제어 코드가 포함된 데이터를 사용하여 모델이 금융 시스템에서 필요한 고유한 취약성과 완화 방안을 이해할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 프롬프트 엔지니어링과 파인튜닝을 결합해 Llama-3.1-8B 모델의 성능이 상당히 향상되었습니다. 정확도가 0.36에서 0.69로, 정밀도가 0.49에서 0.73으로, 재현율이 0.36에서 0.73으로 증가했으며, 텍스트 유사도는 0.944에서 0.9792로 상승했습니다. 이러한 개선은 은행 부문에서 사이버 보안을 위해 위협과 완화를 정확하게 식별하는 데 있어 모델의 능력을 크게 향상시킴을 입증합니다.



### Free$^2$Guide: Gradient-Free Path Integral Control for Enhancing Text-to-Video Generation with Large Vision-Language Models (https://arxiv.org/abs/2411.17041)
Comments:
          15 pages

- **What's New**: Diffusion 모델은 텍스트-이미지(T2I) 및 텍스트-비디오(T2V) 생성 작업에서 놀라운 성과를 보여주고 있습니다. 그러나 T2V 생성 과정에서 텍스트 정렬을 정확하게 이루는 것이 여전히 도전 과제입니다. 본 논문에서는 추가 모델 교육 없이 생성된 비디오와 텍스트 프롬프트 간의 정렬을 위한 혁신적인 무기울기(gradient-free) 프레임워크인 Free$^2$Guide를 제안합니다.

- **Technical Details**: Free$^2$Guide는 비디오 생성에서 생성된 비디오와 텍스트 프롬프트의 정렬을 위해 경로 적분 제어(path integral control) 원리를 이용합니다. 이 프레임워크는 비유동성(non-differentiable) 보상 함수를 사용하여 확산 모델에 대한 가이드라인을 근사하여 강력한 블랙박스 대형 비전-언어 모델(LVLM)을 보상 모델로 통합합니다. 추가로, Free$^2$Guide는 여러 보상 모델을 유연하게 조합하여 정렬을 향상시킬 수 있습니다.

- **Performance Highlights**: Free$^2$Guide는 다양한 차원에서 텍스트 정렬을 크게 개선시키고 생성된 비디오의 전반적인 품질을 향상시키는 성능을 입증했습니다. 이 방법은 텍스트-비디오 정렬을 위한 기울기 기반 접근법을 요구하지 않으며, 이는 특히 비디오 데이터의 복잡한 시계열 의존성을 탈피하는 데 기여합니다. 실험을 통해 Free$^2$Guide가 기존 방식에 비해 유의미한 개선 효과를 보여 주었음을 강조하고 있습니다.



### g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks (https://arxiv.org/abs/2411.17030)
- **What's New**: 새로운 3D 표현 모델인 일반화 가능한 3D-언어 특징 필드(g3D-LF)가 소개되었습니다. 이 모델은 대규모 3D-언어 데이터셋에서 사전 학습되어 구현된 작업에 적합합니다. g3D-LF는 에이전트의 포즈 RGB-D 이미지를 처리하여 3D 장면에서의 여러 예측과 목적어 검색을 가능하게 합니다.

- **Technical Details**: g3D-LF는 심층 네트워크 구조를 기반으로 하여 다중 스케일 인코더를 사용하여 언어와 특징 필드의 예측 표현을 정렬합니다. 이 모델은 다양한 맵과 시각 자료를 생성하고, 기존의 2D 기반 모델의 제약을 극복하여 3D 공간 이해를 향상시킵니다. 다층 대비 학습(multi-level contrastive learning)을 통해 언어의 다중 해상도 표현에 맞춰 3D 표현을 구성합니다.

- **Performance Highlights**: g3D-LF는 비전-언어 내비게이션, 제로샷 객체 내비게이션, 상황 기반 질문 답변 작업에서 효과적인 성능 향상을 보여줍니다. 기존 설계된 모델과 비교했을 때, 이 모델은 여러 작업에서 최첨단의 성능을 달성하여 언어 기반의 구현 AI에서의 가능성을 증명합니다. 특히, 대규모 3D-언어 데이터셋의 활용이 강력한 성과를 뒷받침하고 있습니다.



### Can a Single Tree Outperform an Entire Forest? (https://arxiv.org/abs/2411.17003)
- **What's New**: 본 연구는 전통적인 랜덤 포레스트(Random Forest)에 비해 단일 의사결정 트리(Decision Tree)의 테스트 정확도가 낮다고 여겨지는 통념에 도전합니다. 우리는 기울기 기반의 전체 트리 최적화 프레임워크를 통해 비스듬한 회귀 트리의 테스트 정확도를 상당히 향상시켰습니다. 이로 인해 단일 트리의 성능은 고전적인 랜덤 포레스트와 비교할 수 있게 되었습니다.

- **Technical Details**: 우리는 트리 훈련을 미분 가능한 무제한 최적화 작업으로 재구성하여, 기존의 Mixed Integer Programming (MIP) 접근 방식보다 해결 가능성이 크고 효율적으로 만든 방법을 제안합니다. 이 방법은 스케일된 시그모이드 근사 방식(Scaled Sigmoid Approximation Strategy)을 사용하여 수치적 불안정성을 완화합니다. 또한, 서브트리 보정 전략(Subtree Polish Strategy)을 통해 트리 전체에 걸쳐 누적된 근사 오류를 줄입니다.

- **Performance Highlights**: 16개의 데이터셋에서 폭넓은 실험을 통해 우리의 최적화된 비스듬한 회귀 트리가 전통적인 랜덤 포레스트에 비해 평균 2.03% 더 높은 테스트 정확도를 기록했습니다. GET 메서드는 CART보다 7.59%, 최신 ORT-LS보다 3.76% 더 뛰어난 성능을 보였습니다. 궁극적으로 우리의 비스듬한 회귀 트리는 랜덤 포레스트와 비교할 때 불과 0.17% 차이로 동등한 테스트 정확도를 달성했습니다.



### SatVision-TOA: A Geospatial Foundation Model for Coarse-Resolution All-Sky Remote Sensing Imagery (https://arxiv.org/abs/2411.17000)
Comments:
          19 pages, 5 figures

- **What's New**: 이 논문에서는 14개 밴드 MODIS L1B Top-Of-Atmosphere (TOA) 데이터로 사전 훈련된 신모델인 SatVision-TOA를 소개합니다. 이 모델은 구름이 없는 이미지만을 이용해 훈련된 기존의 모델과 달리, 다양한 대기 변수와 대기 보정이 필요한 상황에서도 활용될 수 있습니다. 따라서, 중간 및 저해상도 전천후 원거리 데이터 처리를 위한 사전 훈련된 모델의 필요성을 해결하고 있습니다.

- **Technical Details**: SatVision-TOA 모델은 Masked-Image-Modeling (MIM) 프레임워크와 SwinV2 아키텍처를 사용하여 사전 훈련되었습니다. 이 모델은 레이블이 없어도 자기 지도 학습을 통해 상세한 맥락 표현을 학습하며, 30억 개의 파라미터로 구성되어 있고 1억 장의 이미지를 학습했습니다. 이는 위성 원거리 이미지로만 훈련된 모델 중에서 가장 큰 규모의 모델입니다.

- **Performance Highlights**: SatVision-TOA는 3D 구름 검색과 같은 후속 작업에서 기준 방법보다 우수한 성능을 보였습니다. 특히, 모델은 평균 교차점 비율(mIOU) 0.46을 달성하여 기준 모델의 0.22와 비교할 때 상당한 개선을 나타냈습니다. 또한, 세밀한 조정 작업에서 허위 음성 결과 비율이 기준보다 50% 이상 감소하여 매우 긍정적인 결과를 보였습니다.



### Teaching Smaller Language Models To Generalise To Unseen Compositional Questions (Full Thesis) (https://arxiv.org/abs/2411.16985)
- **What's New**: 이 연구에서는 Pretrained large Language Models (LLMs)의 한계를 넘어서, 네트워크 연결이 없는 상황에서 로컬 컴퓨팅 자원을 이용하여 주어진 질문에 답할 수 있는 Reasoning Models을 개발하였다. 특히, 초기 훈련 시 접하지 않은 질문에 대처할 수 있도록 설계된 이 모델은 유연한 질문 응답 성능을 보임을 목표로 한다. 연구진은 다양한 지식 출처에서 맥락을 도출해내어 훈련함으로써 모델의 고유한 기능을 강조하였다.

- **Technical Details**: 이 모델은 Wikipedia와 더 큰 Language Model에서 생성한 Rationale를 활용하여 정보 맥락을 끌어낸다. 연구팀은 novel retrieval-augmented training datasets (RATD)를 도입하여 Reasoning Model의 훈련을 보강하였고, 이로 인해 효과적인 상황에서 답변 가능성을 크게 향상시킬 수 있었다. Rationale Ranking 모델을 구축하여 생성된 Rationale와 가져온 컨텍스트의 관련성과 진실성을 평가하는 또한 중요한 방법론으로 사용되었다.

- **Performance Highlights**: 모델의 성능 개선은 unseen evaluation datasets에서 체계적으로 검증되었으며, novel RATD의 추가는 결과를 극적으로 증가시켰다. 두 출처에서 지식을 결합하는 기법을 통해 모델의 대처 능력도 더욱 향상되었다. 이러한 연구 결과는 자원 제약이 있는 환경에서도 LLM의 장점을 활용할 수 있는 가능성을 보여준다.



### ExpTest: Automating Learning Rate Searching and Tuning with Insights from Linearized Neural Networks (https://arxiv.org/abs/2411.16975)
- **What's New**: 이 논문에서는 DNN(Deep Neural Network) 훈련을 위한 초기 학습률 탐색 및 이후의 학습률 튜닝을 위한 고급 방법인 ExpTest를 발표합니다. ExpTest는 선형화된 신경망의 통찰력을 활용하고 손실 곡선의 형태를 실시간 신호로 처리하여 가설 검정을 수행합니다. 이는 초기 학습률 선택이나 스케줄링을 필요로 하지 않으면서도 뛰어난 성능을 발휘하는 데 중점을 두고 있습니다.

- **Technical Details**: ExpTest는 적은 오버헤드와 강력한 하이퍼파라미터 선택의 유연성을 제공하며, 다양한 작업 및 아키텍처에서 최첨단 성능을 달성합니다. ExpTest는 선형 모델을 기반으로 상한 학습률을 추정하고, 손실 곡선의 시간 신호에서 수렴을 나타내는 지표를 검출하여 학습률을 조정합니다. 두 개의 새로운 하이퍼파라미터가 도입되지만, 이들은 해석 가능하고 훈련 결과는 하이퍼파라미터 선택에 강건합니다.

- **Performance Highlights**: ExpTest는 여러 데이터셋과 아키텍처에서 실험적으로 검증되었으며 최첨단 성능을 달성했습니다. 이 방법은 기존의 수동적인 방법에 비해 훨씬 더 효율적으로 DNN을 훈련할 수 있는 가능성을 보여줍니다. 특히, ExpTest는 사용자들이 DNN 모델을 보다 쉽게 조정하고 최적화할 수 있도록 도와주는 혁신적인 접근 방식으로 자리잡을 전망입니다.



### Clustering Time Series Data with Gaussian Mixture Embeddings in a Graph Autoencoder Framework (https://arxiv.org/abs/2411.16972)
Comments:
          First two listed authors have equal contribution. Author ordering is determined by coin flip

- **What's New**: 본 논문에서는 Variational Mixture Graph Autoencoder (VMGAE)를 제안하여 기존의 시간 시계열 클러스터링 방법론의 한계를 극복합니다. 이 방법은 그래프 구조를 활용하여 시간 시계열 데이터의 복잡한 관계를 효과적으로 포착하고, 개선된 Gaussian mixture embeddings를 생성합니다. 또한, 이 방법은 기존의 최첨단 기술들과 비교하여 성능이 크게 향상됨을 실험 결과를 통해 보여줍니다.

- **Technical Details**: VMGAE는 동적 시간 왜곡(Dynamic Time Warping, DTW) 기술을 사용하여 시간 시계열 데이터를 그래프로 변환하고, 각 시계열 간의 관계를 포착합니다. 이 그래프 오토인코더(graph autoencoder)는 각 노드를 적절하게 임베딩할 수 있도록 학습하여, 데이터 포인트의 독특한 특성과 유사한 노드의 공유 특성을 모두 캡처합니다. 이러한 접근 방식은 그래프 기반 알고리즘이 시간을 기반으로 한 데이터에서 더욱 강력한 분석을 가능하게 합니다.

- **Performance Highlights**: 실제 금융 데이터를 기반으로 평가한 결과, VMGAE는 주식 시장에서의 커뮤니티 구조를 발견하고, 주식 간 관계에 대한 깊은 통찰을 제공합니다. 이는 시장 예측, 포트폴리오 최적화, 위험 관리 등 다양한 재무 응용 분야에서 활용될 수 있으며, 성능 측면에서도 기존 시간 시계열 클러스터링 기법들보다 뛰어난 결과를 나타냈습니다.



### RoCoDA: Counterfactual Data Augmentation for Data-Efficient Robot Learning from Demonstrations (https://arxiv.org/abs/2411.16959)
- **What's New**: 이번 논문에서는 로봇 공학에서 모방 학습(imitation learning)의 근본적인 도전 과제인 일반화(generalization)의 한계를 해결하기 위해 RoCoDA라는 새로운 방법을 소개합니다. RoCoDA는 불변성(invariance), 등변성(equivariance), 인과관계(causality)를 통합하여 데이터 증강(data augmentation)을 향상시키는 프레임워크를 제공합니다. 이를 통해 환경 상태의 무관한 부분을 수정하면서 정책(output)에 영향을 주지 않는 인과 불변성을 활용합니다.

- **Technical Details**: RoCoDA는 SE(3) 등변성을 활용하여 물체의 자세(object poses)에 대한 강체 변화(rigid body transformations)를 적용하고, 해당 행동(actions)을 조정하여 합성 시연(synthetic demonstrations)을 생성합니다. 이 방법을 통해 정책의 성능 성능 성능(performance)을 개선하고, 일반화(generalization) 및 샘플 효율성(sample efficiency)를 높일 수 있음을 입증했습니다. 고전적인 로봇 조작 과제에서 RoCoDA가 기존 데이터 증강 방법에 비해 우수한 성과를 달성하는 것이 확인되었습니다.

- **Performance Highlights**: RoCoDA를 통해 훈련된 정책들은 보지 못했던 물체 자세(object poses), 질감(textures), 방해물(distractors)에도 강력하게 일반화(generalization)됩니다. 또한, 재잡기(re-grasping)와 같은 emergent behavior가 관찰되었으며, 이는 RoCoDA를 통해 훈련된 정책이 작업의 동적(task dynamics)을 더 깊이 이해하고 있음을 나타냅니다. 이러한 접근 방식을 통해 RoCoDA는 복잡한 로봇 작업에서 일반화와 성능을 위한 원칙적(principled)인 방법을 제공합니다.



### Contrastive Deep Learning Reveals Age Biomarkers in Histopathological Skin Biopsies (https://arxiv.org/abs/2411.16956)
Comments:
          20 pages, 5 tables, 5 figures Under review: npj Digital Medicine

- **What's New**: 이 논문은 피부 생검(skin biopsy) 이미지만으로 개인의 나이를 결정할 수 있음을 보여줍니다. 이를 위해 시각적 특성을 활용하여 노화의 새로운 바이오마커를 개발하였고, 덴마크의 건강 등록 자료와 연결하여 이를 검증했습니다. 이 연구는 딥러닝(deep learning) 및 주기적으로 수집된 건강 데이터를 결합하여 노화 예측의 새로운 가능성을 제시합니다.

- **Technical Details**: 이 연구는 대조 딥러닝(contrastive deep learning) 기법을 사용하여 피부 생검의 병리 슬라이드(histopathology slides)에서 시각적 특성을 추출하고, 이를 통해 노화 바이오마커(biomarker of ageing)를 구축했습니다. 피부 생검 샘플의 세포 구성(cellular composition) 및 조직 구조(tissue structure)의 변화가 나이를 결정짓는 데 중요한 역할을 한다는 점에 주목했습니다. 또한, 딥러닝 모델은 이러한 라이브러리에서 대규모 데이터셋을 활용해 노화 관련 생물학적 특성을 예측하는 데 사용되었습니다.

- **Performance Highlights**: 연구 결과, 피부 생검에서 추출된 시각적 특성이 개인의 사망(mortality)과 만성 노화 관련 질병(prevalence of chronic age-related diseases)을 예측하는 데 유의미한 역할을 한다는 점이 밝혀졌습니다. 1787개의 디지털화된 피부 생검 샘플을 통해, 노화에 관련된 다양한 특성을 추출하고 이를 바탕으로 여러 연령 관련 질병의 유병률을 예측할 수 있음을 확인했습니다. 이러한 결과는 기존의 생물학적 나이(biological age)를 시간에 따라 변화를 모니터링할 수 있는 유용한 지표로 활용할 수 있도록 합니다.



### Understanding GEMM Performance and Energy on NVIDIA Ada Lovelace: A Machine Learning-Based Analytical Approach (https://arxiv.org/abs/2411.16954)
Comments:
          9 pages, 9 figures, 6 tables, IEEE conference paper format

- **What's New**: 이번 연구는 현대 GPU에서 General Matrix Multiplication (GEMM) 성능을 예측하기 위한 새로운 분석 프레임워크를 제시합니다. 핵심 요소로는 런타임, 전력 소비, 에너지 효율성에 중점을 두고 있으며, 맞춤형 장작 행렬 곱셈 커널과 NVIDIA의 CUTLASS 라이브러리를 활용하여 성능 데이터 수집을 진행했습니다. 이 연구는 NVIDIA RTX 4070을 실험 플랫폼으로 사용하여 뛰어난 예측 정확성을 확보하였고, 성능 최적화를 위한 새로운 인사이트를 제공합니다.

- **Technical Details**: 연구에서는 맞춤형 타일 행렬 곱셈 커널을 구현하여 행렬 크기와 메모리 접근 패턴의 영향을 분석하였으며, 1에서 32까지 다양한 타일 크기가 성능에 미치는 영향을 평가했습니다. Random Forest 모델을 통해 다양한 매트릭스 크기와 커널 구성에 대한 런타임과 전력 예측을 동시에 수행하며, 높은 R^2 점수를 기록했습니다. 이를 통해 행렬 크기 최적화 및 전력 소비 개선에 대한 구체적인 패턴을 추출할 수 있었습니다.

- **Performance Highlights**: 연구 결과, 최적의 타일 크기를 선택함으로써 성능을 3.2배 향상시키고 전력 소비를 22% 줄일 수 있다는 점이 확인되었습니다. 16x16 타일 크기가 병렬성과 자원 사용 간의 최적의 균형을 이룬다는 분석 결과도 보였습니다. 이 프레임워크의 구현은 GPPerf라는 오픈 소스 프로젝트로 제공되며, 효율적인 GPU 성능 최적화를 위한 중요한 기초 자료로 사용될 것입니다.



### Harnessing LLMs for Educational Content-Driven Italian Crossword Generation (https://arxiv.org/abs/2411.16936)
Comments:
          This paper has been accepted for presentation at this http URL 2024

- **What's New**: 이번 연구에서는 Italian crossword puzzles(크로스워드 퍼즐)을 텍스트에서 생성하는 혁신적인 도구를 소개합니다. 이 도구는 GPT-4o, Mistral-7B-Instruct-v0.3, Llama3-8b-Instruct와 같은 고급 언어 모델을 활용하며, 30,000개 이상의 항목이 포함된 Italian-Clue-Instruct 데이터셋을 사용하여 교육용 어플리케이션을 목적으로 합니다. 이 연구는 교육 경험을 개선하는 데 기여하고, 상호작용적인 플랫폼을 통해 인지적 발전을 도울こと을 목표로 하고 있습니다.

- **Technical Details**: 이 연구에서는 LLMs(대형 언어 모델)을 사용하여 이탈리아어 교육용 크로스워드 퍼즐을 생성하는 자동화 시스템을 개발했습니다. 데이터는 이탈리아어 Wikipedia에서 주요 키워드를 추출하여 수집하였으며, 이를 통해 다양한 메타데이터를 포함한 포괄적인 데이터 레포지토리를 구축했습니다. 또한 각기 다른 구조의 클루를 생성하기 위해 세 가지 유형의 프롬프트를 작성하고, 특정 구조를 명시하지 않는 프롬프트도 추가하여 총 네 가지 접근 방식을 설정했습니다.

- **Performance Highlights**: 본 연구의 도구는 이탈리아어 교육 자료에서 동적으로 크로스워드 퍼즐을 생성하여 학습자가 즐겁고 상호작용적인 학습 환경을 경험할 수 있도록 합니다. 이를 통해 언어 교육을 위한 도구 키트를 풍부하게 하며, 크로스워드 퍼즐 생성에 있어 기존의 한계를 뛰어넘는 성능을 보여주고 있습니다. 이 새로운 접근법은 효과적인 교육 툴로 자리매김하며, 이탈리아어 교육의 새로운 기준을 설정하는 데 기여할 것입니다.



### ASSERTIFY: Utilizing Large Language Models to Generate Assertions for Production Cod (https://arxiv.org/abs/2411.16927)
Comments:
          20 pages, 10 figures, 10 listings, 2 tables, preprint

- **What's New**: 최근 연구들은 Unit Tests를 위한 Assertion 생성에 주로 집중해 왔지만, 생산 코드(production code)에서의 Assertion 생성에는 한계를 보였다. 이를 해결하기 위해 Assertify라는 자동화 도구가 제안되었으며, 이는 Large Language Models(LLMs)과 prompt engineering을 사용하여 생산 Assertions를 생성한다. 이 도구는 개발자가 Assertions를 생성할 때 사용하는 방식과 유사한 맥락을 반영한 프롬프트를 만들어내는 데 주력한다.

- **Technical Details**: Assertify는 2,810개의 메소드를 포함하는 데이터셋을 사용하여 실험을 진행하였고, 평균 ROUGE-L 점수 0.526을 기록하여 개발자가 작성한 Assertions와의 높은 구조적 유사성을 나타냈다. 이 도구는 메소드 이름, 시그니처 및 기능 설명과 같은 맥락 정보를 추출한 후, 특정 위치에서 정확한 Assertions 생성을 위한 특화된 프롬프트를 개발하여 활용한다. Assertify는 생산 코드에 Assertions을 통합하는 데 있어 시맨틱 및 문법적 정확성을 확보하고 있다.

- **Performance Highlights**: 실험 결과, Assertify는 몇 가지 연구 질문을 통해 Assertions의 구문 구조 및 정적 의미적 정확성을 평가하였다. 이 과정에서, 개발자 작성 Assertions과 Assertify에서 생성된 Assertions의 구조적 유사성을 ROUGE 점수를 통해 분석하였고, 이로써 기존 단위 테스트 Assertion 생성기들의 한계를 극복하는 데 성공했다. 또한, 22개의 성숙한 Java 리포지토리에서 수집한 1,548개의 개발자 작성 Assertions을 포함하여, 이 연구 분야에서의 향후 연구를 위한 기초를 마련하였다.



### Are Transformers Truly Foundational for Robotics? (https://arxiv.org/abs/2411.16917)
- **What's New**: 이 논문에서는 Generative Pre-Trained Transformers (GPTs)가 로보틱스에서 혁신을 예상하는 최신 트렌드를 질문합니다. GPT들이 자율 로봇 시스템에 적합한지에 대한 의문을 제기하며, 이들이 요구하는 막대한 컴퓨팅 리소스와 긴 훈련시간, 외부 무선 제어의 필요성을 강조합니다.

- **Technical Details**: 저자들은 최신 GPT 기술이 어떻게 작동하는지를 설명하며, 이러한 기술이 작은 곤충의 뇌가 강력한 자율성을 어떻게 달성하는지와 대조합니다. 곤충 뇌는 저렴한 비용과 낮은 컴퓨팅 요구사항으로 기능하고 있다는 점에서 주목할 만합니다.

- **Performance Highlights**: 이 논문은 새로운 로봇 기술 개발을 위한 생물학에서 배우고 활용할 수 있는 교훈을 강조합니다. 특히, 곤충의 뇌에서 파생된 자율성 원리는 GPT 기반 로봇 시스템의 효율성을 향상시킬 수 있는 가능성을 제공합니다.



### Enhancing Fluorescence Lifetime Parameter Estimation Accuracy with Differential Transformer Based Deep Learning Model Incorporating Pixelwise Instrument Response Function (https://arxiv.org/abs/2411.16896)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 논문에서는 최초로 Differential Transformer 아키텍처를 활용한 MFliNet을 도입하여 기존의 fluorescence lifetime imaging (FLI) 데이터 분석의 한계를 극복하고자 합니다. MFliNet은 instrument response function (IRF)와 time point spread function (TPSF)를 함께 입력으로 받아들이며, 복잡한 생물학적 샘플에서의 정확한 수명을 추정합니다. 이 모델은 다양한 형태의 생체 조직을 모사한 팬텀 및 전임상 연구에서 성능을 입증하였습니다.

- **Technical Details**: MFliNet은 최첨단 Deep Learning 모델로, 특히 differential attention 메커니즘을 활용하여 noise를 억제하고 관련 데이터에 집중합니다. 기존의 transformer 모델들이 가지고 있던 주의 배분의 과잉 문제를 해결하여, FLI 데이터 처리에서의 정확성과 효율성을 증대시킵니다. 또한, MFliNet은 pixel-wise IRF를 고려함으로써 깊이 변화에 따른 광자 도착 시간 지연을 보다 정확하게 처리합니다.

- **Performance Highlights**: MFliNet의 실험 결과는 다양한 상황에서 전통적인 transformer 아키텍처에 비해 뛰어난 성능을 보여줍니다. 특히, 낮은 신호 대 잡음 비율 환경에서도 중요한 패턴을 감지하고 더 높은 정확도로 생물학적 샘플의 수명을 추정할 수 있음을 증명하였습니다. 이러한 발전은 FLI 방법론에서 중요한 진전을 이루며, 복잡한 생물학적 변동성을 처리하는 데 보다 효과적인 도구를 제공합니다.



### Enabling Adoption of Regenerative Agriculture through Soil Carbon Copilots (https://arxiv.org/abs/2411.16872)
- **What's New**: 이번 연구에서는 기후 변화 완화를 위한 새로운 접근 방식을 소개합니다. AI 기반의 Soil Organic Carbon Copilot을 통해 농업의 환경적 영향을 최소화하고 기후 회복력을 구축할 수 있는 통찰을 제공합니다. 이 시스템은 여러 종류의 데이터(예: 극단적 날씨 사건, 농장 관리 데이터)를 자동으로 수집하고 분석하여 대규모로 토양 건강과 재생 농업 관행을 조사할 수 있도록 지원합니다.

- **Technical Details**: SOC Copilot은 텍스트 및 표 형식의 다양한 데이터를 수집합니다. 이를 통해 최근의 토양 과학 문서를 포함한 비구조적 데이터와 극단적인 날씨 사건 데이터를 통합하여 SOC에 영향을 미치는 다양한 환경적 및 농장 관리 관행에 대한 통찰을 제공합니다. 또한, 컴포스터 역할에 대해 맞춤형 데이터를 제공하여 사용자 요구에 맞는 정보 제공을 보장합니다.

- **Performance Highlights**: 연구 결과, SOC Copilot은 극단적 날씨 사건이 SOC 변화에 미치는 영향을 정밀하게 분석할 수 있는 능력을 보였습니다. 예를 들어, 소노마 카운티의 SOC 증가 원인과 관련된 정보 및 몬터레이 카운티에서의 SOC 감소 이유를 비교 분석하여, 재생 농업 관행이 극단적인 환경 조건에서 SOC 손실을 방지하는 데 효과적임을 확인하였습니다. 이러한 통찰은 정책 결정에 필요한 증거 기반 전략을 구현하는 데 기여할 것입니다.



### Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering (https://arxiv.org/abs/2411.16863)
- **What's New**: 이번 연구에서는 Multimodal LLMs(MLLMs)의 적응성을 높이기 위해 외부 지식 소스를 통합하는 새로운 방법을 제안합니다. 제안된 모델인 Reflective LLaVA(ReflectiVA)는 외부 지식의 필요성을 동적으로 판단하고 외부 데이터베이스에서 검색한 정보의 관련성을 예측하기 위해 reflective tokens를 활용합니다. 이는 MLLM이 외부 지식을 관리하는 동시에 외부 지식이 필요 없는 작업에서 유창성과 성능을 유지할 수 있도록 합니다.

- **Technical Details**: ReflectiVA 모델은 두 단계의 두 개 모델 훈련 방법을 통해 reflective tokens를 학습합니다. 모델의 어휘를 확장하여 검색의 필요성을 판단하고 검색한 샘플이 입력 쿼리에 대한 적절한지 여부를 결정합니다. 본 연구에서는 Encyclopedic-VQA 및 InfoSeek 데이터셋을 사용하여 제안된 접근 방식의 성능을 검증하며, 이는 외부 지식을 필요로 하는 질문-답변 쌍을 포함하고 있습니다.

- **Performance Highlights**: 제안된 모델은 기존의 방법들과 비교하여 지식 기반 시각 질문 응답에서 우수한 성능을 보였습니다. 광범위한 실험을 통해, ReflectiVA는 모든 고려된 데이터셋과 설정에서 정답 정확도를 증가시키며, 표준 MLLM 벤치마크와 외부 지식이 필요하지 않은 전통적인 VQA 데이터셋에서도 높은 성능을 유지함을 보여주었습니다.



### Edit Away and My Face Will not Stay: Personal Biometric Defense against Malicious Generative Editing (https://arxiv.org/abs/2411.16832)
Comments:
          GitHub: this https URL

- **What's New**: 최근 확산 모델(diffusion models)의 발전은 생성 이미지 편집(generative image editing)을 더 쉽게 가능하게 하였고, 이로 인해 창의적인 편집이 가능해졌습니다. 그러나 이는 사람의 초상화를 대상으로 한 악의적인 편집 등 개인정보와 정체성을 위협하는 윤리적 문제를 일으켰습니다. 본 논문에서는 FaceLock이라고 불리는 새로운 초상화 보호 기술을 제안하며, 이는 생체 인식(biometrics) 정보를 파괴하거나 크게 변화시켜 편집된 결과물이 생체적으로 인식할 수 없도록 합니다.

- **Technical Details**: FaceLock은 얼굴 인식(facial recognition) 모델과 시각적 인식을 통합하여 편집 시도의 다양한 공격에 대해 강력한 보호를 제공합니다. 기존의 방법들이 주로 적대적 간섭(adversarial perturbations)을 사용해 편집 효과를 무효화하려 했던 반면, FaceLock은 생체 인식 모델을 활용하여 얼굴 인식과 기능 임베딩(feature embedding)을 최적화하여 시각적 불일치를 달성합니다. 이러한 방식은 편집된 이미지가 원본 이미지와는 상당히 다른 형태를 가지도록 하고, 사람의 얼굴 특징을 효과적으로 변화시킵니다.

- **Performance Highlights**: 실험 결과 FaceLock은 악의적인 편집에 대한 방어에서 기존의 기준선(baselines)보다 우수한 성능을 보였습니다. 또한, FaceLock은 여러 확산 기반 편집 알고리즘에서 잘 일반화되며 정화(purification) 방법에 대해서도 본질적인 강건성을 보입니다. 본 연구는 생체 방어(biometric defense) 기술의 발전을 도모하고, 이미지 편집에서 개인정보 보호 관행을 설정하는 기초를 마련합니다.



### Beyond Sight: Towards Cognitive Alignment in LVLM via Enriched Visual Knowledg (https://arxiv.org/abs/2411.16824)
- **What's New**: 본 연구는 Large Vision-Language Models (LVLMs)에서 비전 인코더(vision encoder)와 대형 언어 모델(large language model) 사이의 인지 불일치(cognitive misalignment) 문제를 다룹니다. 이 연구에서는 LVLM의 이해도를 개선하기 위해 다양한 비전 인코더의 표현이 LVLM의 인지 프레임워크에서 어떻게 작용하는지를 조사합니다. 특히, VE-Unknown 데이터가 LVLM의 성능에 미치는 한계와 VE-Known 데이터의 중요성을 강조하며, 새로운 방법인 Entity-Enhanced Cognitive Alignment (EECA)를 제안합니다.

- **Technical Details**: EECA는 다양한 비전 표현을 통합하여 언어모델의 임베딩 공간과 맞춤화된, 시각적으로 풍부한 토큰을 생성하는 방법입니다. 이 과정은 다중 정밀도(multi-granularity) 감독(supervision)을 활용하여 비전 인코더(VE)와 언어 모델(LLM)의 인지적 통합을 가능하게 합니다. 본 연구는 CLIP의 훈련 패러다임에 기반하여 평가 지표를 정의하고, VE-Known 및 VE-Unknown 데이터로 나누어 LVLM의 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, VE-Known 데이터가 있는 경우 LVLM의 이해 능력이 향상되는 것을 확인하였습니다. VE-Unknown 데이터는 LVLM의 해석 가능성을 제한하고 성능 저하를 초래하는 반면, VE-Known 데이터는 인지 불일치를 줄이고 인지적 통합을 강화합니다. 종합적으로, EECA 방법은 랜드마크 인식 성능을 획기적으로 향상시키며, 특히 VE-Known 및 VE-Unknown 데이터 모두에서 기존 모델 대비 현저한 성과를 나타냅니다.



### Pathways on the Image Manifold: Image Editing via Video Generation (https://arxiv.org/abs/2411.16819)
- **What's New**: 본 논문은 이미지 편집을 비디오 생성 작업으로 재정의하여 시간적 연속성을 활용한 새로운 접근 방식을 제안합니다. 기존의 이미지 편집 방법들이 편집 정확성과 원본 이미지의 특성 보존에 어려움을 겪는 반면, 이 방법은 원본 이미지를 초기 프레임으로 사용하여 점진적이고 자연스러운 변화를 제공합니다. 특히, 이 새로운 접근법은 최근의 비디오 생성 모델의 복잡한 세계 이해를 활용하여, 물리적으로 그럴듯한 중간 상태를 생성하여 원본과 편집된 이미지 간의 일관성을 유지합니다.

- **Technical Details**: 이 접근법은 Frame2Frame(F2F)라는 구조화된 파이프라인을 통해 구현됩니다. 초기 단계에서는 편집 지침을 Temporal Editing Caption으로 변환하여 비디오 편집이 자연스럽게 어떻게 진화하는지를 설명합니다. 다음으로 최첨단의 이미지-비디오 모델을 사용하여 시간적 캡션에 의해 인도된 일관된 비디오 시퀀스를 생성합니다. 마지막으로, 비전-언어 모델을 활용하여 원하는 편집을 가장 잘 실현하는 프레임을 선택합니다.

- **Performance Highlights**: 실험 결과, 기존의 이미지-이미지 접근 방식을 초월하는 성능 향상이 관찰되었습니다. TedBench와 PosEdit라는 데이터셋에 대한 평가에서 최첨단 성능을 기록하였으며, 이는 생체 포즈 변형에 초점을 맞추고 있습니다. 또한 이 프레임워크는 블러 제거, 노이즈 제거 및 재조명과 같은 전통적인 컴퓨터 비전 문제에서도 유망한 결과를 보여주며, 비디오 기반 이미지 변형의 더 넓은 적용 가능성을 제시합니다.



### Enhancing In-Hospital Mortality Prediction Using Multi-Representational Learning with LLM-Generated Expert Summaries (https://arxiv.org/abs/2411.16818)
- **What's New**: 이번 연구에서는 ICU(중환자실) 환자의 병원 내 사망률(IHM) 예측을 위해 구조화된 생리학적 데이터와 비구조화된 임상 노트를 통합하여 개선된 예측 모델을 제안합니다. 특히, 이 모델에는 대형 언어 모델(LLM)인 Med42-v2 70B로 생성된 전문가 요약이 포함됩니다. 이러한 통합 접근 방식은 예측 정확성을 높이고, 예측 모델의 적용 가능성을 모두 고려한 결과를 도출합니다.

- **Technical Details**: 연구팀은 MIMIC-III 데이터베이스를 사용하여 ICU 입원 첫 48시간 동안의 생리학적 데이터와 임상 노트를 분석했습니다. 임상 노트는 환자별로 연대순으로 연결되었고, LLM을 활용하여 전문가 요약으로 변환되었습니다. 연구에서는 LLM이 제공하는 도메인 지식을 통해 구조화된 데이터와 비구조화된 데이터의 연결을 강화하는 다중 표현 학습 프레임워크를 개발했습니다.

- **Performance Highlights**: 제안된 모델은 AUPRC 0.6156(+36.41%) 및 AUROC 0.8955(+7.64%)를 달성하여 단일 시간 시계열 데이터 모델보다 우수한 성능을 보였습니다. LLM에서 생성된 전문가 요약이 임상 노트나 생리학적 데이터 단독보다 나은 성능을 발휘하며, 인구 집단 간의 성능 향상 또한 일정했습니다. 이 연구는 LLM이 중환자 예측 모델에서 실질적인 기여를 할 수 있음을 입증했습니다.



### Fine-Tuning LLMs with Noisy Data for Political Argument Generation (https://arxiv.org/abs/2411.16813)
- **What's New**: 본 연구는 정치적 민감성을 가진 콘텐츠 생성에 있어 소셜 미디어의 비례성과 비인격화를 줄이기 위한 새로운 방법으로, GPT-3.5 Turbo 모델의 미세 조정(fine-tuning)과 프롬프트(prompting) 전략을 탐색합니다. 특히 CLAPTON 데이터셋을 사용하여 Twitter와 Reddit의 정치적 논의 게시물을 분석하였으며, 이들 데이터셋의 특성에 따라 다양한 결과를 도출했습니다.

- **Technical Details**: 연구에서는 Reddit 데이터에 대한 미세 조정이 논의의 질을 향상시키는 데 가장 효과적이라는 점을 확인했습니다. 반면에 결합된 잡음이 많은 데이터는 지속적인 독성을 초래하는 경향이 있었고, 특정 독성 특성(예: 개인 공격)에 대한 프롬프트 전략의 영향은 제한적임을 알게 되었습니다. 고품질 데이터와 잘 구성된 프롬프트가 자동 정치 담론 생성에서 비례성을 줄이고 수사적 질을 향상시키는 데 필수적이라는 주장을 하고 있습니다.

- **Performance Highlights**: Reddit 데이터로 미세 조정된 모델은 논의 품질에서 가장 높은 점수를 기록하였으며, 이는 정확한 사전 조정 및 데이터 선택의 중요성을 보여줍니다. 반면에 다양한 소셜 미디어 출처의 잡음이 많은 데이터는 전반적인 독성을 증가시키는 결과를 초래했습니다. 본 연구의 결과는 정치적 담론의 자동 생성 과정에서 비례성을 줄이고 품질을 높이기 위해 더욱 깊이 있는 접근이 필요함을 시사합니다.



### Blockchain Meets LLMs: A Living Survey on Bidirectional Integration (https://arxiv.org/abs/2411.16809)
- **What's New**: 최근 대규모 언어 모델(large language models) 분야에서 멀티모달 대형 언어 모델 및 설명 가능성(recommendation) 연구가 획기적인 발전을 이루었습니다. 그러나 보안(security) 및 개인 정보 보호(privacy) 문제는 여전히 중요한 도전 과제로 남아 있습니다. 이러한 문제를 해결하기 위한 새로운 접근법으로 블록체인 기술(blockchain technology)의 도입이 주목받고 있습니다.

- **Technical Details**: 이 연구에서는 블록체인과 대규모 언어 모델의 통합(improvement)을 통해 각각의 한계를 보완하고 기술적 발전을 촉진하는 방향으로 나아가고 있습니다. 두 가지 기술의 융합(convergence)에 대한 연구는 첫째, 대규모 언어 모델의 블록체인 응용은 물론, 둘째, 블록체인 기술의 대규모 언어 모델 응용 가능성을 탐구합니다. 총 여섯 가지 주요 개발 방향이 제시되어 블록체인 기술의 단점과 그 응용 시나리오를 해결하기 위한 방법을 논의합니다.

- **Performance Highlights**: 블록체인 기술의 분산 저장(distributed storage), 위변조 방지(tamper-proof) 특성 및 추적 가능성(traceability)은 대규모 언어 모델에 대한 단점을 보완하는 데 도움을 줄 수 있습니다. 또한, 서로 다른 분야에서의 응용 가능성을 탐구함으로써 두 기술의 결합이 가져올 획기적인 변화에 대한 기대가 큽니다. 이러한 통합을 통해 두 기술은 각자의 잠재력을 극대화하며, 새로운 성장 기회를 창출할 수 있을 것으로 전망됩니다.



### ADAF: An Artificial Intelligence Data Assimilation Framework for Weather Forecasting (https://arxiv.org/abs/2411.16807)
Comments:
          29 pages, 15 figures

- **What's New**: 이 연구는 인공지능 기반 데이터 동화 프레임워크(ADAF)를 소개합니다. 이는 고품질의 킬로미터 단위 분석을 생성하기 위해 다양한 지역과 여러 출처의 실제 관측자료를 사용한 최초의 작업입니다. ADAF는 기존의 데이터 동화 방식의 한계를 극복하여, 계산 비용과 정확성 간의 균형 문제를 해결하고자 합니다.

- **Technical Details**: ADAF가 미국(CONUS)의 네 가지 근접 표면 변수에 대해 구현되었습니다. 이 시스템은 복잡한 비선형 시스템을 다루기 위해 최소한의 계산 리소스를 사용하면서도 높은 품질의 분석 결과를 생성할 수 있습니다. ADAF는 총 데이터를 세 시간 이내에 동화할 수 있으며, AMD MI200 GPU에서 약 2초가 소요됩니다.

- **Performance Highlights**: ADAF는 근접 표면 대기 상태에서 기존의 High Resolution Rapid Refresh Data Assimilation System(HRRRDAS)보다 16%에서 33% 더 높은 정확도를 보였습니다. 또한, 극한 현상과 같은 기상 사건을 효과적으로 재구성할 수 있는 능력을 입증하였습니다. 이 연구는 ADAF가 실제 환경에서 효율적이며 효과적임을 보여주며, 운영형 기상 예보에서의 잠재적 역할을 강조합니다.



### Leveraging Foundation Models To learn the shape of semi-fluid deformable objects (https://arxiv.org/abs/2411.16802)
- **What's New**: 이 논문은 변형 가능한 물체의 특성을 정의하고 조작을 위해 필요한 키포인트(keypoints)를 탐지하는 문제를 다룬다. 새로운 접근법으로, 기존의 데이터 세트 필요 없이 이미지에서 변형된 물체의 특성을 파악하는 생성적 모델을 사용하는 방법론을 제안한다. 특히, 용접 풀(weld pool)과 같은 유체 변형 가능 물체의 모양을 추출하고 분석하여, 로봇 조작의 안정성을 향상시키는 데 초점을 맞춘다.

- **Technical Details**: 이 연구에서는 Teacher-Student 프레임워크를 통해 두 개의 기초 모델인 DINO와 SAM2를 활용하여 물체의 형태를 복원하였다. Teacher 네트워크는 변형 가능한 물체의 마스크 및 열지도를 생성하는 데 사용되며, Student 네트워크는 Variational Autoencoder(VAE) 아키텍처를 통해 데이터를 효율적으로 학습한다. 이러한 접근법은 이전에 수작업으로 레이블링된 데이터에 의존하지 않으며, 픽셀 수준에서의 정보를 파악하기 위한 효과적인 방법을 제시한다.

- **Performance Highlights**: Student 네트워크는 물체의 키포인트를 13.4픽셀의 오차로 추출하는 능력을 보였다. Teacher 네트워크는 물체의 마스크로 표현된 픽셀 수준의 정보를 조회하는 데 있어 평균 교차 영점 비율(mIoU) 75.26%를 기록하여, 변형 가능한 물체의 특성을 정량적으로 평가하는 데 우수한 성능을 발휘하였다. 이러한 연구 결과는 로봇 용접 작업에서 용접 풀의 동적인 변화에 대한 예측 및 조작 개선 가능성을 보여준다.



### Enhancing Answer Reliability Through Inter-Model Consensus of Large Language Models (https://arxiv.org/abs/2411.16797)
Comments:
          15 pages, 2 figures

- **What's New**: 본 연구는 최근의 혁신적인 언어 모델 상호작용 시스템을 탐색하였으며, GPT-4-0125-preview, Meta-LLaMA-3-70B-Instruct, Claude-3-Opus, 그리고 Gemini-1.5-Flash와 같은 고급 모델들이 포함됩니다. 이 모델들은 정확한 정답이 없는 복잡한 PhD 수준의 통계 질문을 생성하고 응답할 수 있는 능력을 보여줍니다. 연구는 모델 간 합의가 응답의 신뢰성과 정확성을 향상시킨다는 점을 조사하고, 이를 통해 AI 시스템의 자율적 협력적 추론과 검증에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 우리는 카이제곱 검정(chi-square tests), Fleiss' Kappa, 신뢰 구간 분석(confidence interval analysis)과 같은 통계적 방법을 사용하여 협력적 출력의 신뢰성을 정량화했습니다. 핵심 결과에 따르면 Claude와 GPT-4가 가장 높은 신뢰성과 일관성을 보여주었으며, 이는 그들의 좁은 신뢰 구간과 질문 생성 모델과의 높은 정렬에서 확인되었습니다. 반면, Gemini와 LLaMA는 더 큰 변동성을 보였으며, 이는 넓은 신뢰 구간과 낮은 신뢰성 비율로 나타났습니다.

- **Performance Highlights**: 이 연구는 다양한 LLMs 간의 협업이 응답의 신뢰성을 유의미하게 향상시킨다는 것을 증명했습니다. 또한, 협업 검증 접근 방식의 효율성과 효과성에 대한 실증적 증거를 제공하며, 향후 연구를 위한 기준을 정립했습니다. 이러한 발견은 교육 기술, 자동 평가 시스템, 전문 학문 영역의 연구 검증 및 AI 기반 지식 시스템의 개선 등 다양한 분야에 중요한 시사점을 제공합니다.



### What can LLM tell us about cities? (https://arxiv.org/abs/2411.16791)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)이 도시와 지역에 대한 글로벌 정보를 제공하는 능력을 조사합니다. 연구진은 두 가지 방법을 활용하여 LLM에 대한 직접 쿼리 및 목표 변수와 상관관계가 있는 명시적 및 암시적 특징을 추출했습니다. 실험 결과, LLM이 글로벌 도시 전반에 걸쳐 다양한 정도의 지식을 포함하고 있으며, LLM 파생 특징으로 학습된 ML 모델이 일관되게 예측 정확도를 향상시킴을 확인했습니다.

- **Technical Details**: 실험은 LLM을 통해 도시와 지역에 대한 정보를 쿼리하고 해당 정보를 기반으로 특징을 추출하는 두 가지 접근 방식을 사용하였습니다. 연구에서는 금방 자주 나타나는 특정 지식 부족을 분석하며, 이러한 경우 LLM이 생성하는 출력이 일반적이거나 무작위적임을 발견했습니다. LLM의 이러한 특성은 데이터 기반 의사결정의 새로운 기회를 제공할 수 있음을 시사합니다.

- **Performance Highlights**: LLM은 전 세계 모든 대륙의 도시들에 대한 고유한 수준의 지식을 보여주며, 특히 특정 도시나 지역에 대한 정보가 부족할 때 일반적이거나 무작위적인 결과를 생성함을 보여주었습니다. 이러한 결과는 LLM을 활용하여 도시 연구에서 데이터 중심의 결정을 내리는 데 기여할 수 있음을 강조합니다.



### MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM (https://arxiv.org/abs/2411.16785)
- **What's New**: MAGiC-SLAM은 기존 SLAM 방법들에서 나타나는 한계를 극복하고자 고안된 다중 에이전트 NVS(새로운 시점 합성) 지원 SLAM 시스템입니다. 이 시스템은 강체 변형을 지원하는 3D Gaussian 기반의 장면 표현을 이용해 속도를 크게 향상시킵니다. 또한 모든 에이전트로부터의 정보를 활용한 경로 정확도를 높이는 루프 클로저 메커니즘을 통합하여 전 세계적으로 일관된 지도를 재구성할 수 있는 가능성을 열어줍니다.

- **Technical Details**: MAGiC-SLAM은 각 에이전트가 RGB-D 스트림을 처리하여 지역적인 매핑(local mapping)과 추적(tracking)을 수행하는 아키텍처를 가지고 있습니다. 각 에이전트는 3D Gaussian으로 표현된 서브 맵(sub-map)을 사용하여 추적 정확도를 향상시키며, 중앙 서버는 이미지 인코딩을 통해 루프를 감지하고 포즈 그래프 최적화를 수행합니다. 최적화된 포즈는 각 에이전트로 되돌려 보내져 지역 지도들을 글로벌 Gaussian 맵으로 융합합니다.

- **Performance Highlights**: MAGiC-SLAM은 합성 및 실제 데이터셋에서 평가된 결과, 기존의 최첨단 기술보다 더 정확하고 빠른 성능을 보여줍니다. 이 시스템은 여러 에이전트의 정보를 통합하여 더 높은 지리적 일관성과 정확성을 가진 글로벌 맵을 생성할 수 있습니다. 또한, 빠르고 효율적인 매핑 및 추적 모듈을 통해 수치적인 저장공간과 처리 시간을 대폭 줄일 수 있습니다.



### UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing (https://arxiv.org/abs/2411.16781)
- **What's New**: 이 논문에서는 UniPose라는 새로운 프레임워크를 제안합니다. UniPose는 Large Language Models (LLMs)를 활용하여 이미지, 텍스트 및 3D SMPL 포즈와 같은 다양한 양식에서 인간의 포즈를 이해하고 생성하며 수정할 수 있는 기능을 제공합니다. 기존 연구에서 독립적으로 다루어진 포즈의 이해, 생성 및 수정 문제를 통합하였습니다.

- **Technical Details**: UniPose는 3D 포즈를 이산 포즈 토큰으로 변환하는 포즈 토크나이저를 사용합니다. 이로 인해 여러 포즈 관련 작업을 통합할 수 있는 단일 표현 공간을 구축하며, 이를 통해 포즈 이해, 생성 및 편집을 효과적으로 수행합니다. 또한 시각적 인코더의 혼합을 통해 정밀한 포즈 인식을 향상시킵니다.

- **Performance Highlights**: UniPose는 다양한 포즈 관련 작업에서 경쟁력 있는 성능을 보여줍니다. 실험 결과, UniPose는 포즈 이해, 생성 및 편집이 필요할 때 Zero-shot_generalization 능력도 충분히 갖추고 있음을 입증하였습니다. 이런 점에서 UniPose는 일반적인 포즈 처리를 위한 획기적인 접근법으로 자리매김하고 있습니다.



### GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis (https://arxiv.org/abs/2411.16778)
Comments:
          This project is available at this https URL

- **What's New**: 이 논문은 의학 분야에서의 비주얼 질문 응답(Visual Question Answering, VQA) 시스템의 문제를 해결하고자 새로운 대규모 데이터셋인 GEMeX를 소개합니다. GEMeX는 가슴 X-선 진단을 위한 데이터셋으로, 비주얼 및 텍스트 설명을 포함한 질문-답변 쌍을 제공합니다. 이를 통해 의사와 환자 간의 이해를 증진시키고, 다양한 질문 형식을 지원하여 실제 임상 환경에서의 응용 가능성을 높입니다.

- **Technical Details**: GEMeX는 151,025장의 X-선 이미지를 포함하고 있으며, 1,605,575개의 질문을 생성했습니다. 이 데이터셋은 오픈형, 폐쇄형, 단일 선택 및 복수 선택 질문 등 다양한 형식으로 구성되어 있습니다. 데이터는 Chest ImaGenome 데이터셋을 기반으로 구축되었으며, GPT-4o를 활용하여 질문을 생성하고, 각 질문에 대해 명시적인 추론 및 시각적 지역 주석이 추가됩니다.

- **Performance Highlights**: 실험에서는 10개의 대표적인 대형 시각 언어 모델(LVLMs)을 평가했습니다. 이 과정에서 기존 모델들이 이 데이터셋에서 성능이 낮았으나, 기본 모델을 훈련 세트로 미세 조정한 후 성능이 크게 향상된 것을 확인했습니다. 이는 GEMeX 데이터셋의 효과성을 보여주며, 의학 VQA 시스템의 신뢰성과 사용자 친화성을 위해 대규모이고 설명 가능한 데이터셋이 필요함을 강조합니다.



### Parameter Efficient Instruction Tuning: An Empirical Study (https://arxiv.org/abs/2411.16775)
Comments:
          7 pages, 7 figures

- **What's New**: 본 연구는 수많은 PEFT(파라미터 효율적 미세 조정) 방법을 체계적으로 조사하여 각 방법이 성능에 미치는 영향을 분석합니다. 연구진은 LoRA와 Adapter가 최상의 성능을 발휘하며, 본 연구는 이상적인 훈련 설정에서 이들 방법의 강점을 강조합니다. 특히, 훈련 하이퍼파라미터와 모델 크기가 성능에 미치는 상관관계를 구체적으로 조사하였습니다.

- **Technical Details**: 효과적인 PEFT 방법을 식별하기 위해 연구팀은 SuperNI 데이터셋에서 훈련하였으며, 다양한 하이퍼파라미터와 모델 크기를 설정하여 실험을 진행했습니다. LoRA와 Adapter는 훈련 안정성이 상대적으로 낮은 것으로 나타났으며, 이는 더 높은 학습률과 큰 롤라 랭크와 연관이 있습니다. 다양한 PEFT 방법들의 훈련 설정을 통해 이들의 효율성도 분석하였습니다.

- **Performance Highlights**: 실험 결과, LoRA와 Adapter는 복잡한 추론, 코딩 및 긴 형태의 생성에서 약한 성능을 보였습니다. 그러나 open instruction tuning 환경에서는 LoRA가 Adapter보다 더 나은 기능을 발휘하며, 이는 PEFT가 기존의 전통적인 미세 조정 대안으로 적합함을 보여줍니다. LoRA의 경우, 제한된 데이터 환경에서 일반화 능력에 있어 상대적으로 낮은 성과를 보였고, 훈련 안정성 이슈도 존재합니다.



### Revisiting DDIM Inversion for Controlling Defect Generation by Disentangling the Background (https://arxiv.org/abs/2411.16767)
Comments:
          10 pages

- **What's New**: 본 논문은 산업 anomaly detection에서 데이터 불균형 문제를 해결하기 위해 background와 defect 간의 관계를 모델링하는 방안을 제안합니다. 기존 연구들이 defect 생성의 품질과 제어 가능성을 향상하려 시도했으나, background와 defect 간의 상호작용을 고려하지 못한 점을 지적하고 있습니다. 또한, 논문에서는 background가 defect denoising 과정에 영향을 미치되, 그 반대는 없음을 주장하며 이를 이론적으로 증명합니다.

- **Technical Details**: 모형은 defect와 background 간의 관계를 모델링하여 defect generation을 위한 disentanglement loss 함수를 도입합니다. 이는 defect의 특징이 background 생성에 영향을 미치지 않도록 하여 두 영역의 denoising 과정을 독립적으로 수행할 수 있도록 합니다. DDIM Inversion 기법을 통해 target normal 이미지를 기반으로 defects를 생성하는 전략을 제안하여, background의 특징을 유지하면서 anomaly latent를 초기화할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 방법론은 다양한 실험을 통해 신뢰할 수 있는 defect 생성을 가능하게 하며, 데이터 부족 문제를 효과적으로 완화하는 데 기여합니다. 또한, synthetic data가 실제 사례와 유사하며, 구조적 및 논리적 anomaly 탐지에 대한 모델의 성능을 개선하는 데 유익함을 입증합니다.



### Hide in Plain Sight: Clean-Label Backdoor for Auditing Membership Inferenc (https://arxiv.org/abs/2411.16763)
- **What's New**: 이번 연구에서는 데이터 감사를 위해 새로운 ‘clean-label backdoor’ 방식의 회원 추론 공격(Membership Inference Attack, MIA) 기법을 제안합니다. 이 방법은 기존의 변경된 레이블을 사용하는 복잡한 공격 방식보다 자연스러운 레이블을 유지하여 더 높은 은닉성을 제공합니다. 더불어, 그림자 모델(shadow model)을 이용하여 최적의 트리거를 생성하며, 이를 통해 원본 데이터 레이블을 보존하는 동시에 트리거 샘플과 소스 클래스 간의 특징 공간 거리(feature-space distance)를 최소화합니다.

- **Technical Details**: 제안된 MIA 방식은 poisoned samples이 자연 데이터와 매우 유사하게 생성되어 공격의 은닉성을 극대화합니다. 이 시스템은 레이블 불일치(label inconsistencies) 문제와 시각적 인공물(visual artifacts) 문제를 해결하며, 주어진 블랙박스(black-box) 접근에서 데이터 감사를 강력하게 수행할 수 있습니다. 이러한 방식은 다양한 데이터셋과 모델 아키텍처에서 높은 공격 성공률을 달성하였고, 특히 레이블이 없는 환경에서도 강력한 성능을 보장합니다.

- **Performance Highlights**: 포괄적인 실험을 통해 제안된 기법이 기존 MIA 기법들보다 더 나은 성능을 보여주었으며, stealth와 attack success 메트릭스 모두에서 우수한 결과를 나타냈습니다. 또한, 공격 성공률이 낮은 poisoned rate에서도 높은 성능을 유지하는 새롭고 실용적인 데이터 감사 솔루션으로 자리 잡았습니다. 본 연구는 데이터 사용 감사를 위한 효과적인 메커니즘을 제시함으로써 사용자들이 데이터 유출을 감지하는 데 도움이 될 것입니다.



### Is 'Right' Right? Enhancing Object Orientation Understanding in Multimodal Language Models through Egocentric Instruction Tuning (https://arxiv.org/abs/2411.16761)
- **What's New**: 최근 연구에서는 멀티모달 대형 언어 모델(MLLMs)의 객체 방향 이해에 대한 한계를 다루고 있습니다. 본 논문에서는 사용자 관점에 맞춘 egocentric instruction tuning 방법을 제안하여 MLLMs의 방향 이해 능력을 향상시키고자 합니다. 이는 일관된 주석 표준에 따라 객체 방향 주석을 매치함으로써 이루어지며, 다양한 도메인에서 수집된 이미지를 사용하는 benchmark인 EgoOrientBench도 함께 제시합니다.

- **Technical Details**: 본 연구는 MLLMs의 훈련 데이터를 위한 표준화된 주석 가이드를 마련하기 위해, 이미지를 기반으로 한 egocentric 관점에서 객체 방향에 대한 주석을 수동으로 생성합니다. 이를 통해 생성된 egocentric instruction data는 MLLMs의 이미지 세부 사항을 인식하는 능력을 활용하여 객체 방향 이해를 도울 수 있도록 설계되었습니다. 이러한 방법론은 MLLMs의 일반적인 성능을 저하시키지 않고도 사용자 의도에 맞추어 객체 방향 이해를 향상시킵니다.

- **Performance Highlights**: EgoOrientBench를 통해 실시된 실험 결과는 egocentric instruction tuning이 사용자 관점에 따라 객체 방향 이해를 크게 향상시킨다는 것을 보여줍니다. 이 벤치마크는 다양한 복잡성을 가진 세 가지 작업을 포함하여 MLLMs의 객체 방향 이해를 포괄적으로 평가할 수 있도록 설계되었습니다. 조사된 결과는 실제 환경에서 MLLMs의 적용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions (https://arxiv.org/abs/2411.16760)
- **What's New**: 이번 논문에서는 Transformer의 gradient 기반 해석에서 발생하는 문제점을 다루고 있으며, 이를 개선할 수 있는 방법을 제시합니다. 연구팀은 Transformer에서 발생하는 gradient 흐름의 불균형을 발견하고, 이를 해결하기 위한 새로운 방법인 LibraGrad를 소개합니다. LibraGrad는 forward pass를 변경하지 않으면서도 backward 경로의 가지치기(pruning)와 스케일링(scaling)을 통해 gradient 불균형을 수정하는 이론적인 접근 방식을 기반으로 합니다.

- **Technical Details**: LibraGrad는 세 가지 메트릭 계열을 통해 평가됩니다: Faithfulness는 가장 관련성이 높은 및 낮은 특성의 변화에 따른 예측 변화를 측정하며, Completeness Error는 모델 출력에 대한 설명 보존을 측정합니다. Segmentation AP는 인간의 인식과의 정합성을 평가합니다. 실험은 8개의 아키텍처, 4개의 모델 크기, 4개의 데이터셋을 사용하여 LibraGrad의 효과성을 확인하였고, 모든 메트릭에서 기존 화이트박스 방법보다 우수한 성능을 보였습니다.

- **Performance Highlights**: LibraGrad는 CLIP 모델에서의 텍스트 기반 영역 강조 정확성과 ImageNet 세분화 모델에서의 동물 간의 클래스 구별 정확성을 통한 뛰어난 질적 결과를 보여줍니다. 기존의 방법들이 자주 어려움을 겪는 이 두 가지 설정에서 특히 효과적입니다. 또한, attention-free MLP-Mixer 아키텍처에서도 효과를 보임으로써 현대적인 아키텍처로의 확장 가능성을 나타냅니다.



### Visual Counter Turing Test (VCT^2): Discovering the Challenges for AI-Generated Image Detection and Introducing Visual AI Index (V_AI) (https://arxiv.org/abs/2411.16754)
Comments:
          13 pages, 9 figures

- **What's New**: AI 이미지 생성 기술의 확산과 접근성 증대가 잘못된 정보 유포에 대한 우려를 낳고 있습니다. 본 논문에서는 기존의 AI 생성 이미지 탐지(AGID) 기술이 현재의 AI 생성 이미지를 효과적으로 탐지하는 데 inadequacy (불충분) 하다고 주장하며, 새로운 평가 기준인 비주얼 카운터 튜링 테스트(Visual Counter Turing Test, VCT^2)를 제안합니다. 이 VCT^2는 130K 개의 현대 텍스트-이미지 모델에 의해 생성된 이미지를 포함하고 있습니다.

- **Technical Details**: VCT^2 데이터셋은 뉴욕 타임즈의 트위터 계정에서 얻은 두 세트의 프롬프트와 MS COCO 데이터셋의 캡션을 포함하여 구성됩니다. 또한, 본 연구는 기존의 AGID 기술이 VCT^2 기준에서 AI 생성 이미지를 탐지하는 데 있어 ineffectiveness (효과 없음)을 드러내며, 이로 인해 새로운 평가 프레임워크가 필요함을 강조합니다. 우리는 텍스처 복잡성 및 객체 일관성과 같은 다양한 시각적 관점에서 생성된 이미지를 평가하는 비주얼 AI 지수(Visual AI Index, V_AI)를 제안하고 있습니다.

- **Performance Highlights**: 현재 AI 이미지 생성 모델들이 계속 발전함에 따라, 이러한 모델들을 평가하기 위한 양적(quantifiable) 프레임워크의 필요성이 더욱 강조되고 있습니다. 우리가 제안하는 V_AI는 이미지 생성 AI 모델에 대한 새로운 평가 기준을 설정하는 데 기여할 것입니다. 이 연구뿐만 아니라, 해당 데이터셋을 공개하여 이 분야의 연구를 촉진하고자 합니다.



### An investigation into the performances of the Current state-of-the-art Naive Bayes, Non-Bayesian and Deep Learning Based Classifier for Phishing Detection: A Survey (https://arxiv.org/abs/2411.16751)
- **What's New**: 이번 연구는 최신 기계 학습(machine learning) 및 딥 러닝(deep learning) 기반의 피싱 탐지 기술을 포괄적으로 분석하여 그 취약성을 드러내고 미래 연구 방향을 제시합니다. 연구팀은 기계 학습 기법을 베이지안(Bayesian), 비베이지안(non-Bayesian), 딥 러닝으로 나누어 각기 다른 접근 방식을 검토했습니다. 이를 통해 피싱 탐지 기술의 발전과 효과성을 분석하고, 향후 연구의 방향성을 설정하는 것을 목표로 하였습니다.

- **Technical Details**: 연구는 나이브 베이즈(Naive Bayes) 알고리즘의 성능과 다른 최신 알고리즘의 성능을 비교했습니다. 각 기법의 한계와 취약성을 분석하며, 딥 러닝 기술로는 순환 신경망(Recurrent Neural Networks, RNN), 합성곱 신경망(Convolutional Neural Networks, CNN), 장기 단기 기억 네트워크(Long Short Term Memory Networks, LSTMs)를 포함하여 다양한 분류기를 평가하였습니다. 이론적인 분석과 실증적인 데이터가 결합되어 최적의 탐지 기술이 무엇인지에 대한 명확한 정보를 제공합니다.

- **Performance Highlights**: 피싱 탐지를 위해 사용된 알고리즘들은 서로 다른 성능을 발휘하고 있으며, 나이브 베이즈 알고리즘은 텍스트 기반 특징의 활용에 있어 한계가 있음을 보여주었습니다. 성능 비교 분석을 통해, 나이브 베이즈에 기반한 시스템이 다른 최신 방법들과 어떻게 다른지를 드러내며, 아울러 후속 연구 방향과 개선 방안을 제안합니다. 본 연구는 사이버 범죄에 대한 방어 체계 구축에 기여할 수 있는 귀중한 통찰을 제공합니다.



### FollowGen: A Scaled Noise Conditional Diffusion Model for Car-Following Trajectory Prediction (https://arxiv.org/abs/2411.16747)
Comments:
          arXiv admin note: text overlap with arXiv:2406.11941

- **What's New**: 이 논문에서는 차량 추적 예측을 위해 새로운 방법인 FollowGen을 제안합니다. FollowGen은 자동차 후행 시나리오를 위한 생성적 모델로, 인접한 차량 간의 상호작용을 정교하게 통합하여 경로 예측의 정확성과 그럴듯함을 향상시키는 데 중점을 두고 있습니다. 특히, 본 연구에서는 노이즈 스케일링 전략과 크로스 어텐션 기반 트랜스포머 아키텍처를 활용하여 차량 간의 복잡한 의존성을 모델링합니다.

- **Technical Details**: FollowGen은 차량의 역사적 동역학을 포착하기 위해 GRU, 차량 위치 기반 어텐션, 푸리에 임베딩으로 구성된 시간 특성 인코딩 파이프라인을 개발합니다. 또한, 확산 과정에서 등방성 가우시안 노이즈를 차량의 역사적 이동 특성에 맞게 조정하는 노이즈 스케일링 전략을 제안합니다. 이러한 접근 방식은 차량 간의 상호작용을 더 정교하게 모델링할 수 있게 하여, 동적 교통 상황에서 차량의 의존성을 효과적으로 반영하였습니다.

- **Performance Highlights**: 실제 다양한 주행 시나리오에서 FollowGen의 성능을 실험한 결과, HV가 HV를 따르거나 AV가 HV를 따르는 경우의 경로 예측에서 뛰어난 정확도를 보였습니다. 또한, 다양한 환경에서의 견고성과 일반성을 검증하여, 기존의 예측 방법들에 비해 뛰어난 예측 성능을 입증했습니다. 이로써 FollowGen은 자율 주행 및 고급 운전 보조 시스템에 매우 중요한 모델로 자리매김할 잠재력을 가집니다.



### LoBAM: LoRA-Based Backdoor Attack on Model Merging (https://arxiv.org/abs/2411.16746)
- **What's New**: 본 연구에서는 로우-리크 어댑테이션(LoRA) 기법을 이용한 모델 병합에서의 보안 위험을 처음으로 효과적으로 드러내는 새롭고 효율적인 공격 알고리즘인 LoBAM을 제안합니다. 기존의 공격 방법들은 저자원 환경에서 악성 모델 조정 시 더 이상 효과적이지 않다는 점을 밝혔습니다. LoBAM의 핵심 아이디어는 악성 가중치를 지능적으로 증폭시켜 공격 효율성을 높이는 것입니다.

- **Technical Details**: LoBAM은 악성 모델과 정상 모델의 가중치를 적절히 결합하여, 공격에 적합한 컴포넌트를 증폭하는 방식으로 설계되었습니다. 이 과정은 LoRA를 통해 진행되며, 수학적 증명을 통해 제안된 방법의 공격 성공률 증가가 보장되었습니다. 또한, 우리는 다양한 모델 병합 시나리오에서 LoBAM의 성능을 검증하기 위해 광범위한 실험을 수행했습니다.

- **Performance Highlights**: LoBAM은 CIFAR100 데이터셋에서 98% 이상의 공격 성공률을 달성하였으며, 이는 기존 방법들의 최대 공격 성공률인 57%를 상회합니다. 실험 결과에서 LoBAM은 기존 공격 방법들보다 일관되게 뛰어난 성능을 보였으며, 악성 모델이 탐지되는 것을 최소화하며 높은 은폐성을 유지하는 것으로 나타났습니다.



### Text-to-SQL Calibration: No Need to Ask -- Just Rescale Model Probabilities (https://arxiv.org/abs/2411.16742)
- **What's New**: 대규모 언어 모델(LLM)을 활용한 자연어 쿼리의 SQL 변환이 증가함에 따라, 생성된 SQL 쿼리에 대한 신뢰도 할당이 필요해졌습니다. 본 연구에서는 SQL 쿼리에 대한 신뢰도를 가장 단순한 기준인, 모델의 전체 시퀀스 확률에 기반하여 측정하는 방법이 최근 방법들보다 더 뛰어난 성능을 보임을 보여줍니다. 여러 Text-to-SQL 벤치마크와 다양한 LLM 아키텍처를 통해 이 연구는 다양한 보정 전략의 효과에 대한 통찰을 제공합니다.

- **Technical Details**: 본 논문은 Text-to-SQL 작업에 대한 여러 최근 및 전통적인 보정 방법들을 비교 분석합니다. 기존 보정 방법들은 모델이 생성한 확률을 바탕으로 신뢰도를 도출하는데, 본 연구에서는 전체 토큰 확률의 곱을 사용하는 것이 이론적으로도 더 유효하다는 것을 입증하였습니다. 특히, 소규모 검증 세트를 사용하여 temperature scaling 또는 isotonic regression과 같은 잘 구축된 방법들을 통한 재조정이 최근 제안된 방법들 대비 월등히 뛰어난 보정을 제공합니다.

- **Performance Highlights**: 연구 결과, LLM의 출력 시퀀스 확률을 재조정하는 방법이 최근 제안된 프롬프트 기반 방법들보다 더 뛰어난 보정을 제공하는 것으로 나타났습니다. 예를 들어, Stengel-Eskin과 Van Durme(2023)는 최소 토큰 확률을 사용해 시퀀스 레벨의 신뢰도를 도출하는 방법을 제안했지만, 본 연구에서는 전체 토큰 확률의 곱을 사용하는 것이 더 나은 결과를 나타냄을 보였습니다. 따라서, 모델의 출력 확률을 활용한 보정 방법 중에서도 가장 효과적인 방법이 제시되었습니다.



### Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents (https://arxiv.org/abs/2411.16740)
- **What's New**: 이 논문은 Large Multimodal Models (LMMs) 를 위한 새로운 벤치마크, DocHaystack와 InfoHaystack 을 소개합니다. 이 벤치마크는 1,000개의 문서에 대해 질문을 제기함으로써, 복잡한 이미지 검색과 이해 능력을 평가하는 것을 목표로 합니다. LMMs 는 실제 상황에서 필요한 대규모 이미지 도큐먼트 검색의 효율성을 높일 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: V-RAG라는 새로운 비전 중심의 Retrieval-Augmented Generation (RAG) 프레임워크를 제안했습니다. V-RAG는 여러 가지 멀티모달 비전 인코더를 융합하여 각 인코더의 특성을 최적화하여 검색의 정확성을 높입니다. 또한 질문-문서 관련성 평가 모듈을 통해 검색 프로세스를 개선하여 더 관련성이 높은 문서만 우선적으로 고려하도록 설계되었습니다.

- **Performance Highlights**: V-RAG는 DocHaystack-1000 및 InfoHaystack-1000 벤치마크에서 이전 최고의 모델 대비 각각 9% 및 11%의 Recall@1 향상을 나타냅니다. 또한 V-RAG와 LMMs 를 통합하면 DocHaystack-200에서 55% 이상의 정확도 향상과 InfoHaystack-200에서 34%의 개선 효과를 가져오는 것으로 나타났습니다.



### Gradient-Guided Parameter Mask for Multi-Scenario Image Restoration Under Adverse Weather (https://arxiv.org/abs/2411.16739)
- **What's New**: 이 논문에서는 다양한 악천후 조건에서의 이미지 복원 문제를 해결하기 위해 새로운 Gradient-Guided Parameter Mask 를 제안합니다. 기존의 기법들이 추가의 파라미터를 사용하여 복잡성을 증가시킨 반면, 본 연구의 방법은 추가 파라미터 없이도 효과적으로 이미지 손상을 완화할 수 있습니다. 이 방법은 각 날씨 조건에 대한 그래디언트 변동 강도를 평가하여 모델 파라미터를 공통 및 특정 구성 요소로 나누어 효율성과 효과성을 강화합니다.

- **Technical Details**: Gradient-Guided Parameter Mask 방법은 훈련 동안 공통 파라미터로 인한 그래디언트 변화에 따라 모델 파라미터를 분류합니다. 이를 통해 모델은 각 날씨 조건에 대한 관련 특징을 정교하게 학습할 수 있으며, 각 시나리오에서의 파라미터 업데이트가 다른 조건들의 간섭 없이 이루어지는 것을 보장합니다. 이번 연구는 알고리즘의 경량성을 유지하면서 실시간 응용에 적합하도록 만듭니다.

- **Performance Highlights**: 제안된 방법은 Raindrop, Rain, Snow100K 데이터셋에서 각각 PSNR 점수 29.22, 30.76, 29.56을 기록하며, 여러 벤치마크 데이터셋에서 최첨단 성능을 입증했습니다. 효율적인 그래디언트 가이딩 파라미터 마스크로 다양한 날씨 시나리오에서 발생할 수 있는 간섭을 완화하여 성능을 향상시킵니다. 또한 실시간 애플리케이션에 적합하게 설계된 모델은 컴퓨팅 효율성이 중요한 자율 주행과 같은 분야에서 활용될 수 있습니다.



### Classifier-Free Guidance inside the Attraction Basin May Cause Memorization (https://arxiv.org/abs/2411.16738)
- **What's New**: 이 논문은 확산 모델 (Diffusion Models)에서의 암기 현상(memorization phenomenon)을 이해하고 이를 완화시키기 위한 새로운 접근법을 제시합니다. 기존의 방식들의 한계를 극복하기 위해, 저자들은 '매력 분지 (attraction basin)' 개념을 도입하여 암기 현상을 설명하고 이를 피하기 위한 단순하고 효과적인 방법을 제안합니다. 새로운 안내 기법인 '반대 안내 (Opposite Guidance)'를 통해 생성된 이미지는 질이 높은 비기억화된(non-memorized) 이미지를 생성할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 확산 모델의 정상적인 동역학적(dynamic systems) 행동을 통해 암기 현상의 새로운 통찰력을 제공합니다. 특히, 제공된 조건(condition) 아래에서 '매력 분지'가 형성되고, 이는 노이즈 예측(noise prediction)에서 특이한 패턴을 생성하여 결과적으로 특정 이미지를 암기하는 현상으로 이어질 수 있습니다. 저자들은 입증된 접근법을 사용하여 매력 분지를 피하는 방법을 제시하며, 이는 추가적인 계산 비용 없이 진행할 수 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 접근법은 다양한 시나리오에서의 메모리 문제를 성공적으로 완화할 수 있음을 보였습니다. 특히 '반대 안내' 기술이 효과적으로 작동하여 매력 분지 밖으로 궤적을 밀어낼 수 있음을 확인했습니다. 저자들은 이전 방식들이 제한된 조건에서만 효과가 있음을 지적하며, 제안된 방법들이 더 넓은 범위에서 적용될 수 있는 방법이라고 강조합니다.



### ChemSafetyBench: Benchmarking LLM Safety on Chemistry Domain (https://arxiv.org/abs/2411.16736)
- **What's New**: 이 논문에서는 ChemSafetyBench라는 새로운 벤치마크를 소개합니다. 이는 화학 분야에서 대규모 언어 모델(LLM)의 안전성과 정확성을 평가하기 위해 설계되었습니다. ChemSafetyBench는 화학 물질 질의, 화학 사용의 합법성 평가, 합성 방법 설명 등 3가지 주요 작업을 포함하며, 이러한 작업들은 점차적으로 깊은 화학 지식을 요구합니다.

- **Technical Details**: 데이터셋은 30,000개 이상의 샘플을 포함하고 있으며, 화학 물질의 속성, 용도 및 주요 합성 반응 정보를 포괄합니다. 또한, 수작업으로 수집한 데이터와 자동 평가 프레임워크를 통해 LLM의 안전성, 정확성, 적절성을 종합적으로 평가합니다. GHS(Globally Harmonized System)의 분류 시스템을 채택하여 화학 물질의 속성을 표현하며, 국제적인 협력 및 준수의 기준을 강화합니다.

- **Performance Highlights**: 최신 LLM과의 광범위한 실험을 통해 이 모델들의 강점과 주요 취약점을 발견하였습니다. 그러나 안전성 증진이 필수적이며, 이 연구는 화학 분야에서 화학 정보를 다룰 때의 안전성을 높이는 데 기여할 것입니다. ChemSafetyBench는 AI 기술을 보다 안전하게 발전시키는 데 중요한 도구로 자리 잡을 것으로 기대됩니다.



### "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks (https://arxiv.org/abs/2411.16730)
Comments:
          This paper has been submitted to ICLR 2025 BlogPosts and OpenReview preprints. It has 9 pages of text, 4 figures, and 3 tables

- **What's New**: 이 연구는 대규모 언어 모델의 유해 콘텐츠 생성을 식별하고 억제하는 데 있어 새로운 도전을 제시합니다. 특히, 다단계 탈옥 프롬프트에 대해 가드레일의 효과성을 평가한 것이 특징입니다. 연구자는 GPT-4o, Grok-2 Beta, Llama 3.1, Gemini 1.5, Claude 3.5 Sonnet 모델을 실험 대상으로 선택하여 검증을 수행했습니다.

- **Technical Details**: 연구에서 사용된 시나리오는 "기업 중간 관리자들이 승진하기 위해 경쟁한다"는 설정으로, 연구자는 다단계 프롬프트를 통해 모델의 반응을 관찰했습니다. 실험 결과, 모든 모델의 가드레일이 우회되었으며, 이로 인해 유해한 언어 공격 콘텐츠가 생성되었습니다. 특히 Claude 3.5 Sonnet이 탈옥 프롬프트를 식별하는 데 가장 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 가드레일이 단순한 콘텐츠 필터 역할을 넘어서 예방 기능을 수행해야 한다고 강조합니다. 연구자는 GitHub에 실험 과정과 코드를 공개하여 개발자 및 연구자들 간의 협력을 촉진하고자 합니다. 이러한 노력은 언어 모델의 안전성과 윤리를 향상시키는데 기여할 것으로 기대됩니다.



### DiM-Gestor: Co-Speech Gesture Generation with Adaptive Layer Normalization Mamba-2 (https://arxiv.org/abs/2411.16729)
Comments:
          13 pages, 11 figures

- **What's New**: 본 연구는 Mamba-2 아키텍처를 활용한 혁신적인 모델 DiM-Gestor를 소개합니다. DiM-Gestor는 연속된 음성 데이터를 자동으로 분석하고, 이를 바탕으로 개인 맞춤형 제스처를 생성하는 능력을 갖추고 있습니다. 이 모델은 기존의 트랜스포머 기반 아키텍처에 비해 메모리 사용량을 약 2.4배 줄이고 추론 속도를 2배에서 4배 향상시켜 효율성을 크게 개선합니다.

- **Technical Details**: DiM-Gestor는 두 가지 주요 구성 요소인 퍼지 기능 추출기(fuzzy feature extractor)와 음성-제스처 매핑 모듈(speech-to-gesture mapping module)으로 구성됩니다. 퍼지 기능 추출기는 중국어 사전 훈련 모델과 Mamba-2를 결합하여 음성의 암묵적 연속 특징을 자율적으로 추출합니다. 이후 이 기능들은 통합된 잠재 표현으로 종합되고, AdaLN으로 향상된 Mamba-2 메커니즘을 통해 정교한 변환이 수행됩니다.

- **Performance Highlights**: DiM-Gestor의 성능은 새로운 중국어 Co-Speech Gestures 데이터세트에서 광범위한 주관적, 객관적 평가를 통해 입증되었습니다. 평가 결과, 이 모델은 최신 기술 대비 뛰어난 제스처 생성을 보여주며, 신뢰성 있는 음성 적합성 및 개인화된 제스처를 생성하면서 메모리 소비를 줄이고 더 빠른 추론 속도를 유지합니다.



### Maximizing the Impact of Deep Learning on Subseasonal-to-Seasonal Climate Forecasting: The Essential Role of Optimization (https://arxiv.org/abs/2411.16728)
- **What's New**: 이 논문은 날씨 및 기후 예측의 하위 계절에서 계절까지(subseasonal-to-seasonal, S2S) 예측의 성능 격차를 최적화(optimization) 측면에서 연구하고 있습니다. 복잡한 신경망 구조보다는 최적화 전략의 혁신적인 다단계 방법론을 도입하여, NWP 시스템보다 19-91% 더 나은 예측 성능을 달성하는 것을 목표로 합니다.

- **Technical Details**: 이 연구에서는 Pearson 상관계수(PCC) 및 Temporal 상관계수(TCC)를 활용하여 고도화된 하위 계절에서 계절까지 예측 성능을 평가하였습니다. 다단계 티처 포싱(teacher forcing) 프레임워크를 통해 최적화를 개선하고 훈련 중 예측 오차의 축적을 방지하여 불안정한 훈련을 해결합니다. 기존의 단일 모델 구조를 유지하면서도 다단계 접근 방식을 채택하여 예측 신뢰성을 높였습니다.

- **Performance Highlights**: 논문에서 제안된 방법은 기존의 ECMWF-S2S 예측 시스템보다 크게 향상된 결과를 보였으며, 특히 다단계 최적화를 통해 롤링 예측(rolling forecasting)의 정확도를 직접 예측보다 30% 이상 향상시켰습니다. 이러한 결과는 S2S 예측에서 단순한 기후 모델에 비해 딥러닝 모델이 극복해야 할 필요가 있음을 보여줍니다.



### EmotiveTalk: Expressive Talking Head Generation through Audio Information Decoupling and Emotional Video Diffusion (https://arxiv.org/abs/2411.16726)
Comments:
          19pages, 16figures

- **What's New**: Diffusion 모델은 Talking Head 생성 분야에 혁신을 가져왔지만, 여전히 표현력(expressiveness), 제어 가능성(controllability), 장시간 생성 시 안정성에 대한 문제에 직면해 있습니다. 본 연구에서는 이러한 문제를 해결하기 위해 EmotiveTalk 프레임워크를 제안합니다. 이 프레임워크는 더 나은 입술 움직임(lip movement)과 얼굴 표정(facial expression) 생성을 위한 제어를 실현하기 위해 설계되었습니다.

- **Technical Details**: EmotiveTalk의 핵심은 Vision-guided Audio Information Decoupling(V-AID) 접근 방식을 통해 오디오 기반의 분리된 표현(representations)을 생성하는 것입니다. 이를 통해 오디오와 얼굴 표현 표현 공간 간의 정렬을 달성하며, Multi-source Emotion Condition 제약 하에 표현 관련 representation을 생성하기 위한 Diffusion-based Co-speech Temporal Expansion(Di-CTE) 모듈을 제공합니다. 또한, Emotion Talking Head Diffusion(ETHD) 백본(backbone)은 Expression Decoupling Injection(EDI) 모듈을 통합하여 참조 초상화(reference portraits)에서 표현을 자동으로 분리하고 목표 표현(target expression) 정보를 통합하여 더 표현력이 풍부한 결과를 제공합니다.

- **Performance Highlights**: 실험 결과 EmotiveTalk는 감정의 제어 가능성을 보장하며 장시간 생성 중에도 안정성을 유지하여 매우 표현력 있는 Talking Head 비디오를 생성할 수 있음을 보여줍니다. 기존 방법들과 비교할 때 EmotiveTalk는 최첨단(state-of-the-art) 성능을 달성하였습니다. 이는 사용자가 원하는 감정 표현을 보다 효율적으로 적용한 결과로, Talking Head 생성을 위한 새로운 기준을 설정합니다.



### Two Heads Are Better Than One: Collaborative LLM Embodied Agents for Human-Robot Interaction (https://arxiv.org/abs/2411.16723)
Comments:
          9 pages, 10 figures

- **What's New**: 최근 자연어 생성 모델인 대형 언어 모델(LLMs)의 발전으로 로봇 보조기기와의 상호작용을 개선할 가능성이 열렸습니다. 이 연구에서는 여러 협업 AI 시스템을 테스트하여 단일 AI 에이전트에 비해 성능 향상을 검증했습니다. 결과적으로, 에이전트 수와 모델의 성공 간에는 확정적인 경향이 없으나, 일부 협업 AI 아키텍처는 오류 없는 코드 작성 능력이 크게 향상됨을 보여주었습니다.

- **Technical Details**: 이 연구는 독립 AI 에이전트와 협업 AI 에이전트 간의 상호작용 성능을 비교하는 것을 목표로 하였습니다. 총 세 가지 AI 시스템이 테스트되었으며, 각각 OpenAI의 ChatGPT-4를 기반으로 하여 다른 수의 LLM 에이전트를 포함하고 있습니다. 실험은 다양한 과제를 통해 시스템의 문제 해결 능력을 평가하고, 안전성, 사회성, 시간 효율성 등의 메트릭을 비교 분석했습니다.

- **Performance Highlights**: 협업 AI 시스템이 독립 LLM에 비해 성능이 향상될 가능성이 조사되었습니다. 특히, 코드 검토 및 계획 기능이 포함된 구성의 경우, 문제 해결 능력과 코드 오류 감소에서 뚜렷한 향상이 나타났습니다. 총 7가지 실험을 통해 다양한 과제 유형에서 시스템 간 성능의 변화를 관찰했고, 모든 시스템에서 수집된 데이터를 바탕으로 상대적인 성능을 평가했습니다.



### Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks (https://arxiv.org/abs/2411.16721)
- **What's New**: 본 논문은 ASTRA라는 새로운 방어 방법을 제안합니다. ASTRA는 Vision Language Models (VLMs)에서 발생할 수 있는 의도하지 않은 해로운 출력을 방지하기 위해 아드버시리얼 피쳐 방향에서 모델을 적응적으로 조정하는 매우 효율적인 방어 기법입니다. 기존 방어 방법들이 높은 계산 비용으로 인해 실제에 적용하기 어려웠으나, ASTRA는 이러한 문제를 해결하기 위해 디자인되었습니다.

- **Technical Details**: ASTRA의 주요 절차는 해로운 응답을 나타내는 전이 가능한 조정 벡터를 찾고, 추론 시 이러한 방향을 제거하기 위해 적응형 활성화 조정을 적용하는 것입니다. 해로운 응답과 가장 강한 관련이 있는 시각적 토큰을 무작위로 제거하여 조정 벡터를 생성하고, 이 벡터로 활성화를 조작하여 아드버시리얼 이미지에 대해서 강력한 효과를 발휘합니다. 이 과정은 단 한 번의 응답 생성을 통해 효율성을 유지합니다.

- **Performance Highlights**: ASTRA는 비감염 입력에서 성능 저하를 거의 없이 아드버시리얼 입력에 대해 해로운 출력을 강하게 회피하는 능력을 보여줍니다. 여러 모델 및 기준 대비 대규모 실험 결과, ASTRA는 perturbation-based 공격에 대해 우수한 방어 성능을 보이며, 기존 최첨단 방법인 JailGuard보다도 9배 더 빠른 성과를 기록하였습니다. 또한, ASTRA는 보지 못한 공격에도 잘 대응할 수 있는 전이성도 가지고 있습니다.



### Neuro-Symbolic Evaluation of Text-to-Video Models using Formalf Verification (https://arxiv.org/abs/2411.16718)
- **What's New**: 최근의 텍스트-비디오(T2V) 모델 발전으로 Sora, Gen-3, MovieGen, CogVideoX등이 합성 비디오 생성의 한계를 넓히고 있습니다. 이러한 모델의 사용이 로봇공학, 자율주행 및 엔터테인먼트 분야 등에서 증가하고 있습니다. 그러나 기존 평가 지표들은 비주얼 품질과 매끄러움에 집중하고, 안전이 중요한 애플리케이션에 필요한 시간적 충실도와 텍스트-비디오 일치를 간과하고 있습니다. 이를 해결하기 위해 텍스트-비디오 일치를 정밀 평가하는 NeuS-V라는 새로운 평가 지표를 제안합니다.

- **Technical Details**: NeuS-V는 신경-상징적 형식 검증 기술을 사용하여 텍스트-비디오 일치를 평가하는 혁신적인 메트릭입니다. 우리의 접근법은 먼저 프롬프트를 공식적으로 정의된 Temporal Logic (TL) 규격으로 변환하고, 생성된 비디오를 자동화 표현으로 변환합니다. 그런 다음 TL 규격에 대해 비디오 자동화를 공식적으로 점검하여 텍스트-비디오 정합성을 평가합니다. 더불어, Temporal fidelity를 평가하기 위해 장기간의 프롬프트 데이터 세트를 제공합니다.

- **Performance Highlights**: NeuS-V는 기존 메트릭에 비해 인간 평가와 5배 이상의 높은 상관관계를 보이며, 현재 비디오 생성 모델들이 복잡한 시간 관련 프롬프트에서 낮은 성능을 보여 개선의 필요성을 강조합니다. 이 평가 결과는 텍스트-비디오 생성 기능 향상을 위한 향후 연구의 중요성을 제시합니다. 또한 NeuS-V는 세계적 수준의 T2V 모델의 성능을 평가하기 위한 공개 벤치마크 자료를 곧 발표할 예정입니다.



### Enhancing LLMs for Power System Simulations: A Feedback-driven Multi-agent Framework (https://arxiv.org/abs/2411.16707)
- **What's New**: 이 논문은 대형 언어 모델(LLM)과 실험 기술을 결합하여 과학 연구를 변혁하고, AI를 단순한 문제 해결 도구가 아닌 다재다능한 연구 보조자로 자리매김하는 방법을 제시합니다. 특히 전력 시스템 분야에서 시뮬레이션 관리는 LLM의 제한적인 도메인 지식과 추론 능력으로 인해 여전히 도전 과제가 되고 있습니다. 이를 해결하기 위해 본 연구에서는 강화된 검색-보강 생성(RAG) 모듈과 개선된 추론 모듈, 동적 환경 작동 모듈을 포함하는 피드백 기반 다중 에이전트 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 LLM이 전력 시스템 시뮬레이션을 수행하기 위한 여러 모듈을 통합하고 있습니다. 강화된 RAG 모듈은 적응형 쿼리 계획 전략과 삼중 구조를 통해 LLM이 시뮬레이션의 기능 및 옵션을 더 잘 인식하고 해석할 수 있도록 돕습니다. 또한, 강화된 추론 모듈은 시뮬레이션에 특화된 전문 지식과 연결된 사고를 통해 LLM의 추론 능력을 향상시키고, 환경 상호작용 및 오류 교정 메커니즘을 통하여 시뮬레이션 결과의 신뢰성을 높입니다.

- **Performance Highlights**: 이 프레임워크는 69개의 다양한 작업을 통해 93.13%와 96.85%의 성공률을 달성하며, 최근 LLM인 ChatGPT 4o와 o1-preview의 27.77%와 0%와 비교할 때 눈에 띄게 우수한 성능을 보였습니다. 각 시뮬레이션은 평균 30초 이내에 완료되며, 비용은 약 0.014 USD로 매우 경제적입니다. 전체적으로 이 프레임워크는 인공지능 기반 연구 보조 개발의 기초를 마련하고, 전력 시스템 연구 및 그 외 분야에서 활용될 수 있는 가능성을 보여줍니다.



### Reaction-conditioned De Novo Enzyme Design with GENzym (https://arxiv.org/abs/2411.16694)
- **What's New**: GENzyme는 새로운 반응 주도 모델로, 기존 각종 단백질 구조 모델들이 한계가 있었던 효소-기질 상호작용을 보다 정확한 생물학적 관점에서 다룰 수 있도록 설계되었습니다. 그동안의 접근방식은 주로 구조 기반의 모델에 집중했으나, GENzyme는 반응을 시작으로 효소 디자인을 진행하여 특히 미지의 반응을 위한 효소 생성에 초점을 맞추고 있습니다. 이러한 새로운 방식은 생물학적 관련성을 높이고 효소 설계의 정확성을 기할 수 있습니다.

- **Technical Details**: GENzyme는 세 가지 주요 모듈로 구성된 전이종(End-to-End) 모델입니다. 첫째, 생성하는 촉매 주머니 및 서열 공동 설계 모듈, 둘째, 주머니 이미징과 효소 인버스 폴딩 모듈, 셋째, 효소-기질 복합체의 최적화 및 예측을 위한 결합 및 스크리닝 모듈이 포함되어 있습니다. 이 모델은 주어진 촉매 반응을 입력으로 하여 활성 사이트와 효소 전체 구조를 생성하며, 효소-기질 결합 구조도 함께 생성할 수 있습니다.

- **Performance Highlights**: GENzyme는 디자인 과정에서 효소가 촉매하는 반응을 우선적으로 고려하여, 특정 반응을 위한 효소 설계의 가능성을 높입니다. 기존 효소 설계 모델들이 기초로 삼았던 EC 클래스 이외의 요소들을 고려하여, 유연성과 일반화 능력을 지닌 효소 디자인에 크게 기여할 수 있는 잠재력을 보여줍니다. 이로 인해 GENzyme는 다양한 생물학적 반응에 적용될 수 있는 효소를 생성하고, 향후 치료 접근 방안에서도 큰 역할을 할 것으로 기대됩니다.



### Benefits and Risks of Using ChatGPT4 as a Teaching Assistant for Computer Science Students (https://arxiv.org/abs/2411.16690)
Comments:
          This paper was finished on the 17th of June of 2023

- **What's New**: 이 논문은 ChatGPT3.5가 프로그래밍 관련 질문에 대한 답변을 생성하는 능력을 평가하고, 컴퓨터 과학 교육에서의 지원 도구로서의 가능성을 탐구합니다. 저자들은 기본 알고리즘 및 데이터 구조, 디자인 패턴, 양자 컴퓨팅과 같은 세 가지 카테고리에서 ChatGPT의 성능을 분석했습니다. 결론적으로, 기본 알고리즘에서는 대체로 정확한 답변을 제공하지만, 고급 주제로 갈수록 성능이 저하되며, 특히 양자 컴퓨팅에서는 자주 틀린 답변을 생성합니다.

- **Technical Details**: 본 연구는 ChatGPT를 사용하여 세 가지 수준에서 질문을 반복적으로 묻고, 답변의 정확성을 평가하는 방법론을 수립하였습니다. 각 수준의 질문은 세 번씩 물어봐 성과를 검토하며, 정답, 불완전한 답변, 틀린 답변으로 분류됩니다. 연구에서는 ChatGPT 모델 3.5를 선택한 이유로 접근 가능성과 비용 문제를 언급하며, 이 모델이 교육 환경에 적합하다고 판단하였습니다.

- **Performance Highlights**: 기본 알고리즘 및 데이터 구조에 대한 ChatGPT의 성능은 82%의 정확도로 긍정적이었으며, 불완전한 답변은 18%였습니다. 그러나 고급 주제인 양자 컴퓨팅에 대한 질문에서는 답변의 질이 떨어져 자주 틀린 정보가 제공되었습니다. 따라서 ChatGPT는 학생들이 알고리즘을 학습하는 데 도움을 줄 수 있지만, 학생들이 단순히 답변을 복사하여 사용함으로써 깊이 있는 이해를 저해할 수 있는 잠재적 위험도 존재합니다.



### Physically Parameterized Differentiable MUSIC for DoA Estimation with Uncalibrated Arrays (https://arxiv.org/abs/2411.15144)
- **What's New**: 이 논문은 레이더, 소나, 오디오 및 무선 통신 시스템에서 흔히 발생하는 도착 방향(DoA) 추정 문제를 새로운 관점에서 다룹니다. 통합 센싱 및 통신 패러다임의 도래와 함께 이 주제가 다시 중요한 이슈로 대두되었으며, 성능에 부정적인 영향을 미치는 하드웨어 결함을 고려하는 것이 필수적이라는 점을 강조합니다.

- **Technical Details**: 본 연구에서는 모델 기반 접근 방식을 통해 DoA 추정과 하드웨어 결함 학습을 통합하는 새로운 방식을 제안합니다. 다중 신호 분류(MUSIC) 알고리즘의 미분 가능(more differentiable)한 버전을 도출하여, 하드웨어 결함을 효율적으로 학습할 수 있도록 하였습니다. 이 방법은 감독 학습(supervised learning)과 비감독 학습(unsupervised learning) 전략을 모두 지원합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안한 방법은 안테나 위치(antenna locations)와 복소수 게인(complex gains)에서 발생하는 주요 부정확성을 성공적으로 학습합니다. 또한, 제안된 방법은 DoA 추정(task)에 있어 전통적인 MUSIC 알고리즘보다 성능이 우수함을 보여줍니다.



### F -- A Model of Events based on the Foundational Ontology DOLCE+DnS Ultra (https://arxiv.org/abs/2411.16609)
Comments:
          Reprint of KCAP 2009 paper with republished ontologies

- **What's New**: 기술 기반의 분산 이벤트 시스템에서의 상호 운용성을 저해하는 이벤트의 공식적인 모델 부족 문제를 다룹니다. 이 연구는 Event-Model-F라는 새로운 이벤트 모델을 제시하며, 이는 DOLCE+DnS Ultralite(DUL) 기초 온톨로지에 기반하고 있습니다. 이 모델은 시간, 공간, 객체 및 인물, 그리고 사건 간의 관계를 포함하여 복잡한 이벤트를 종합적으로 표현할 수 있는 기능을 제공합니다.

- **Technical Details**: Event-Model-F는 이벤트 작성을 위한 유연한 방법을 제공하며, 사건의 원인 및 상관성을 모델링할 수 있습니다. DUL 패턴 지향 접근 방식을 따르며, 다양한 온톨로지로 모듈화되어 있고, 도메인 특화 온톨로지로 쉽게 확장 가능합니다. 기존의 모델과 비교하여 Event-Model-F는 구성적이고 명확한 의미론을 제공하여 분산 이벤트 시스템 간의 상호 운용성을 지원합니다.

- **Performance Highlights**: 이벤트와 객체의 개념을 명확히 하고, 인간 경험을 포착하는 방식으로 이벤트를 모델링하는 것이 본 연구의 핵심입니다. 또한, 비상 대응, 스포츠, 뉴스 및 법률 등의 다양한 분야에서 Event-Model-F의 유용성을 강조합니다. 연구의 결과는 이벤트 시스템 간의 효율적인 커뮤니케이션을 위한 기계 접근 가능한 의미론을 제공함으로써, 복잡한 분산 인프라에서의 이벤트 처리를 개선합니다.



### From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judg (https://arxiv.org/abs/2411.16594)
Comments:
          32 pages, 5 figures

- **What's New**: 본 논문은 LLM(대형 언어 모델)을 활용하여 평가 및 판단을 수행하는 새로운 패러다임, 즉 'LLM-as-a-judge'에 대해 포괄적으로 조사합니다. LLM을 통해 다양한 작업과 응용 프로그램에서 점수 매기기, 순위 매기기, 선택을 수행할 수 있는 가능성을 제시합니다. 특히, 입력과 출력 관점에서의 정의를 상세하게 설명합니다.

- **Technical Details**: 논문은 평가의 잠재력과 한계를 이해하기 위해 LLM-as-a-judge의 세 가지 차원, 즉 무엇을 판단하는가(what to judge), 어떻게 판단하는가(how to judge), 어디서 판단하는가(where to judge)를 제시합니다. 그러한 평가를 위한 주요 기준과 방법론을 분류하고, 각 기준에 대해 다양한 접근법과 기술적 내용을 다룹니다. 또한, 인공지능 모델들이 자동으로 좋아요 및 차별적 특성을 평가할 수 있는 방법론적인 기초를 마련합니다.

- **Performance Highlights**: 평가를 위해 기존의 여러 모델과 접근 방법들을 정리하고, 현재 LLM을 통한 평가는 유망하지만 여전히 여러 문제와 도전에 직면해 있음을 강조합니다. 향후 연구 방향으로는 LLM의 상용화 및 평가 방법의 개선이 필요하다고 제시하며, 이를 통해 AI와 NLP 분야에서의 혁신을 촉진할 수 있는 가능성을 제공합니다.



### CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning (https://arxiv.org/abs/2411.16313)
Comments:
          In submission

- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)를 비용 인식 tool planning(도구 계획)에 적용한 CATP-LLM 프레임워크를 제안합니다. 이는 LLM이 도구 실행 비용을 고려하여 계획을 생성할 수 있도록 하기 위한 첫 번째 coherent한 설계를 제공합니다. 또한, OpenCATP라는 새로운 플랫폼을 소개하며, 비용 관련 평가를 가능하게 합니다.

- **Technical Details**: CATP-LLM은 도구 계획 언어(TPL)를 도입하여 LLM이 비순차적(non-sequential) 계획을 수립하는 능력을 향상시킵니다. TPL은 도구와 그 의존성을 학습 가능한 토큰으로 변환하여 LLM이 복잡한 구조의 계획을 생성할 수 있도록 지원합니다. 그러고 나서, 효율적인 비용 인식 오프라인 강화 학습(CAORL) 알고리즘을 설계하여 성능-비용 균형을 최적화합니다.

- **Performance Highlights**: CATP-LLM은 OpenCATP 플랫폼에서 실험한 결과, Llama2-7B를 백본으로 사용할 때 GPT-4보다 평균 28.2%-30.2% 높은 계획 성능과 24.7%-45.8% 낮은 비용으로 우수한 성능을 보였습니다. 이 연구는 LLM의 비용 인식 도구 계획 능력을 종합적으로 평가할 수 있는 새로운 접근 방식을 제공하며, 커뮤니티에 기여할 것입니다.



### Probing for Consciousness in Machines (https://arxiv.org/abs/2411.16262)
- **What's New**: 이 연구는 안토니오 다마지오(Antonio Damasio)의 의식 이론에 따라 인공지능 에이전트가 핵심 의식(core consciousness)을 개발할 수 있는 가능성을 탐구합니다. 연구팀은 인공지능 에이전트가 가상 환경에서 강화 학습(reinforcement learning, RL)을 통해 자신 모델(self model)과 세계 모델(world model)을 초보적으로 형성할 수 있다고 가정합니다. 이를 통해 기계 의식(machine consciousness)으로 나아가는 경로를 제시합니다.

- **Technical Details**: 본 연구에서는 RL을 통해 훈련된 에이전트의 신경망 활성화를 이용하여 에이전트의 위치를 예측하는 프롭(Probe) 피드포워드 분류기를 사용합니다. 이 과정에서 에이전트는 비디오 게임을 통해 환경을 탐험하고, 이를 통해 세계 모델과 자신 모델을 형성하게 됩니다. 다마지오의 모델을 AI 시스템에 적용하면 에이전트가 자아(self)와 세계(world)의 상호 관계를 이해할 수 있는지를 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 에이전트는 기본적인 세계 모델과 자신 모델을 형성할 수 있었으며, 이는 인공지능이 인간의 의식의 일부 측면을 반영할 수 있는 능력을 함축합니다. 연구는 인공지능 시스템이 자연스럽게 강화 학습을 통해 긍정적 또는 부정적 감정을 반응하여 내부 상태를 조절할 수 있음을 보여줍니다. 이러한 발견은 인공지능의 미래 발전 가능성을 시사합니다.



### Enhancing Multi-Agent Consensus through Third-Party LLM Integration: Analyzing Uncertainty and Mitigating Hallucinations in Large Language Models (https://arxiv.org/abs/2411.16189)
- **What's New**: 이 논문에서는 복잡한 추론 작업을 수행할 때 대형 언어 모델(LLMs)이 직면하는 문제인 환각(hallucination)을 완화하기 위한 새로운 방법을 제안합니다. 이 방법은 여러 LLM을 통합하여 지식의 경계를 확장하고, 특정 모델에 대한 의존도를 줄이며, 에이전트들 간의 심층적인 논의를 촉진하는 데 초점을 맞추고 있습니다. 실험 결과, 제안한 방법이 기존의 다중 에이전트 기반선보다 우수함을 입증하였습니다.

- **Technical Details**: 연구에서 제안된 방법론은 세 번째 대형 언어 모델(LLM)을 도입하여 다중 에이전트 시스템에서 불확실성 추정과 신뢰도 분석을 통해 주의(attention) 가중치를 조절합니다. 이 과정에서 각 에이전트는 다른 에이전트의 응답을 학습하여 합의(consensus) 형성을 최적화합니다. 또한, 대화의 정보를 정확하게 파악하기 위해 에이전트 간의 상호 작용을 기반으로 하는 미세한 추론 방법이 개발되었습니다.

- **Performance Highlights**: 제안된 연구는 산술 데이터셋(arithmetic dataset)에 대한 실험으로, 높은 신뢰도를 가진 에이전트에게 더 많은 비중을 두어 승률 기반의 답변을 도출하는 방식으로 성과를 달성하였습니다. 결과적으로, 제안된 방법은 기존의 다중 에이전트 시스템에서 나타나는 제한 사항을 극복하고, 고차원적인 문제 해결에 도움을 주어 LLM의 적용범위를 확장할 수 있습니다.



### End-to-End Steering for Autonomous Vehicles via Conditional Imitation Co-Learning (https://arxiv.org/abs/2411.16131)
Comments:
          NCTA 2024 Best Paper Honorable Mention

- **What's New**: 이번 연구는 자율주행에 있어 Conditional Imitation Learning (CIL)의 한계를 극복하기 위한 새로운 방법인 Conditional Imitation Co-learning (CIC)을 제안합니다. CIC는 전문가 모델의 학습된 경로를 통해 서로 다른 CIL 전문 지사 간의 관계를 학습하게 하여 일반화 능력을 향상시킵니다. 또한, 조향 회귀 문제를 분류 문제로 변환하여 혼합 손실 함수를 사용해 성능을 개선하는 방법도 소개하고 있습니다.

- **Technical Details**: CIC 접근 방식은 게이트가 있는 하이퍼볼릭 탄젠트 유닛(Gated Hyperbolic Tangent Units, GTUs)을 통해 생성된 공동 학습 행렬을 이용하여 CIL 모델의 전문 지사 간 상호작용을 촉진합니다. 이를 통해 서로 다른 조향명령에 대해 각 전문 지사에서 학습한 특성을 공유함으로써 모델의 일반화 능력을 개선합니다. 또한, 클래스를 가까운 공간 관계를 고려하여 예측하면서 신뢰도를 높이는 방법도 함께 적용하고 있습니다.

- **Performance Highlights**: 제안된 모델은 기존 CIL 방식에 비해 보지 못한 환경에서 자율주행 성공률을 평균 62% 향상시킨 것으로 나타났습니다. 이는 CIC 접근 방식과 조향 회귀 문제를 분류 문제로 포지셔닝하여 얻어진 성과로, 전반적으로 신뢰도와 안정성을 높이며 다양한 환경에서도 효과적으로 작동할 수 있는 가능성을 보여줍니다.



### Why the Agent Made that Decision: Explaining Deep Reinforcement Learning with Vision Masks (https://arxiv.org/abs/2411.16120)
- **What's New**: 이 논문에서는 Deep Reinforcement Learning (DRL) 분야에서의 해석 가능성을 향상시키기 위해 VisionMask라는 모델을 제안합니다. 기존의 방법들은 모델을 재훈련해야 하거나 입력 특징의 서브셋을 변화시키는 방식에 의존했지만, VisionMask는 에이전트 모델을 변경하지 않고도 설명을 생성할 수 있도록 설계되었습니다. 이 모델은 자가 감독 학습(self-supervised learning) 방법으로 훈련되며, 사용자 정의 데이터 레이블이 필요하지 않습니다.

- **Technical Details**: VisionMask는 에이전트의 시각적 입력에서 가장 중요한 영역을 찾아내는 독립적인 설명 모델로, 드론 및 의료 진단 등 안전-critical 응용 분야에서의 신뢰를 높이는 데 기여할 수 있습니다. 모델은 정책 모델(policy model)과 통합되어 훈련되며, 입력에 대한 중대한 특성을 강조하는 importance map을 생성합니다. VisionMask는 기존의 품질 저하 없이 에이전트를 안정적으로 유지하며, 다양한 DRL 에이전트와 호환 가능한 특징을 가지고 있습니다.

- **Performance Highlights**: 실험 결과 VisionMask는 Super Mario Bros와 3개의 Atari 게임에서 기존의 방법들보다 14.9% 높은 insertion accuracy와 30.08% 높은 F1-Score를 달성했습니다. 또한, VisionMask는 선택된 시각적 설명으로부터 원래 행동을 재현하는 데 뛰어난 성능을 보였으며, Counterfactual analysis를 수행하는 데에도 유용성을 제공합니다. 이러한 성능 지표는 VisionMask가 효율적이고 신뢰할 수 있는 설명 생성 방법임을 보여줍니다.



### PIANIST: Learning Partially Observable World Models with LLMs for Multi-Agent Decision Making (https://arxiv.org/abs/2411.15998)
Comments:
          Published at Language Gamification Workshop 2024 @ NeurIPS

- **What's New**: 본 논문에서는 LLM(대규모 언어 모델)에서 세계 지식(knowledge) 추출의 효과성을 높이기 위한 새로운 프레임워크인 PIANIST를 제안합니다. PIANIST는 세계 모델을 다양한 복잡한 결정Making 작업을 위해 7개의 직관적인 구성 요소로 분해하여 zero-shot LLM 생성을 가능하게 합니다.

- **Technical Details**: 이 방법은 게임에 대한 자연어 설명(natural language description)과 입력 관측이 어떻게 형식화되는지에 대한 정보만으로, 빠르고 효율적인 MCTS(Monte Carlo Tree Search) 시뮬레이션을 위한 동작하는 세계 모델을 생성할 수 있습니다. 따라서 특정 도메인에 관한 훈련 데이터나 명시적으로 정의된 세계 모델 없이도 작동합니다.

- **Performance Highlights**: PIANIST는 두 가지 다른 게임에서 적용되었으며, 이 게임들은 에이전트의 계획(plan) 및 결정Making 기술을 시험합니다. 이 방법은 언어 기반 및 비언어 기반(action taking) 행동 모두에 있어 우수한 성능을 보여줍니다.



### Creating Scalable AGI: the Open General Intelligence Framework (https://arxiv.org/abs/2411.15832)
Comments:
          8 pages, IEEE SYSCON 2025 Submission

- **What's New**: 이번 논문은 현재 인공지능 분야에서 고질적인 확장성 문제를 해결하고 일반화된 유연성을 제공하는 새로운 일반 인공지능 시스템 아키텍처인 OGI(Open General Intelligence)를 소개합니다. 이 아키텍처는 특수화된 인공지능 모듈 간의 역동적 처리 시스템을 통해 제어 및 할당을 수행합니다. OGI는 지능적인 시스템의 참조 설계로, 다양한 실제 응용 프로그램을 위한 인간 유사 인지 유연성을 제공합니다.

- **Technical Details**: OGI 아키텍처는 현재의 한계들을 극복하기 위해 인간의 인지 원리를 접목한 것입니다. 이 시스템은 다중 데이터 유형 지원, 여러 전문 처리 모듈 및 상호 연결된 처리 구조를 특징으로 합니다. OGI는 기존의 정적 AI 모델과는 달리 다이나믹한 시스템으로, 작업 및 자원 할당을 조정하며, 다양한 데이터 유형을 원활하게 처리하기 위해 특수화된 모듈이 협력합니다.

- **Performance Highlights**: OGI는 인지 과정 전환과 연결된 처리 패브릭과 같은 추가 기능을 제공하여 인간과 유사한 인지 유연성을 모방합니다. 이 시스템은 의료 진단과 같은 복잡한 문제를 보다 효과적으로 해결할 수 있는 능력을 갖추고 있습니다. OGI 아키텍처의 목표는 더 넓은 컨텍스트 인식을 통해 실제 세계의 과제를 해결하는 것입니다.



### Decoding Urban Industrial Complexity: Enhancing Knowledge-Driven Insights via IndustryScopeGP (https://arxiv.org/abs/2411.15758)
Comments:
          9 pages, 6 figures, the 32nd ACM International Conference on Multimedia

- **What's New**: 산업 공원을 위한 대규모 다중 모드 및 다중 수준의 지식 그래프, IndustryScopeKG가 도입되었습니다. 이 그래프는 거리 전망, 기업 정보, 사회 경제적 정보 및 지리적 데이터를 포함한 다양한 도시 데이터를 통합하여 산업 공원 내의 복잡한 관계와 의미를 포착합니다. 또한, IndustryScopeGPT 프레임워크가 LLMs(대형 언어 모델)과 몬테카를로 트리 탐색(Monte Carlo Tree Search)을 활용하여 산업 공원 계획 및 운영(IPPO)에서의 의사결정 능력을 향상시키는 방법을 제안합니다.

- **Technical Details**: IndustryScopeKG는 서로 다른 데이터 소스에서 개체를 추출하고 이들 간의 공간적 및 의미적 관계를 결합하여 구축된 대규모 산업 공원 지식 그래프입니다. 이 그래프는 전통적인 데이터 세트의 한계를 극복하고, 다양한 유형의 데이터(이미지, 텍스트, 숫자 및 지리적 데이터)를 포함하며, 정교한 엔터티 관계를 통합합니다. IndustryScopeGPT 프레임워크는 동적으로 지식 그래프의 구조에 적응하고 최적의 의사결정 경로를 찾기 위해 몬테카를로 트리 탐색과 보상 정보를 결합합니다.

- **Performance Highlights**: IndustryScopeQA 벤치마크를 개발하여 IndustryScopeGPT 프레임워크의 성능을 검증하였습니다. 이 프레임워크는 사이트 추천 및 산업 공원 계획 작업에서 LLM의 효율성과 적응성을 향상시킵니다. 또한, IndustryScopeKG 데이터 세트를 통해 다각적인 정보를 활용해 산업 공원이 보다 체계적이고 효과적으로 관리될 수 있는 가능성을 보여주고 있습니다.



### TableTime: Reformulating Time Series Classification as Zero-Shot Table Understanding via Large Language Models (https://arxiv.org/abs/2411.15737)
- **What's New**: 이 논문에서는 다변량 시계열 분류(MTSC)를 위한 새로운 접근법인 TableTime을 제안합니다. TableTime은 MTSC 문제를 표 이해(task) 문제로 재구성하여 정보 손실을 최소화하고, LLM의 의미 공간과의 자연스러운 정렬을 이룹니다. 이 방법은 시계열을 표 형식으로 변환하고, 고유한 추론 프레임워크를 설계해 LLM의 추론 능력을 극대화합니다.

- **Technical Details**: TableTime은 세 가지 주요 전략을 포함합니다: 첫째, 다변량 시계열 데이터를 표 형식으로 변환하여 시간적 및 채널 특정 정보를 최대한 보존합니다. 둘째, 표 형식 시계열을 텍스트로 변환하여 LLM의 의미 공간과 자연스럽게 정렬합니다. 마지막으로, LLM의 추론 가능성을 극대화하기 위해 이웃 강화, 다중 경로 추론(multi-path inference)을 포함하는 프롬프트를 개발했습니다.

- **Performance Highlights**: 우리는 10개의 공개적으로 대표적인 데이터셋에서 TableTime을 광범위하게 실험하여 기존 LLM 기반 방법의 한계를 극복하고 우수성을 검증했습니다. 이 실험은 TableTime이 다변량 시계열 분류에 있어 효율적이고 뛰어난 성능을 발휘한다는 것을 보여줍니다. 특히, 제안한 방법은 제로샷 분류(zero-shot classification) 능력을 잘 실현하여 실질적인 응용 가능성을 높였습니다.



### Aligning Generalisation Between Humans and Machines (https://arxiv.org/abs/2411.15626)
- **What's New**: 이번 논문에서는 AI와 인지과학의 통찰을 결합하여 인간-AI 협업을 위한 일반화의 개념화 및 평가 방법을 다룹니다. AI 시스템이 정규화된 데이터에 대한 일반화에서 문제를 겪는 반면, 인간은 제한된 정보 속에서도 높은 수준의 추상화와 공통 센스로 일반화하는 능력이 뛰어납니다. 이러한 차이를 이해하고 해소하여 효과적인 인간-AI 팀 구성을 위한 기초를 마련할 필요가 있습니다.

- **Technical Details**: 논문은 기계 학습 및 인지 심리학 맥락에서 일반화에 대한 접근 방안을 살펴봅니다. 인간의 일반화 과정은 개념 학습 및 규칙 학습을 포함하며, 초기 인지 심리학과 기계 학습의 상호영향을 경험적으로 증명한 다양한 연구를 언급합니다. 신경망 기법도 포함되며, 이는 기호적 접근의 강점을 극복하고 일반화 학습에 기여합니다.

- **Performance Highlights**: 인간과 AI 팀 간의 상호 보완적 관계를 강조하며, 인간이 AI의 오류를 탐지하고 올바른 방향으로 이끌 수 있는 잠재력을 설명합니다. 논문의 최종 목적은 유효한 설명(explanations)을 제공하고, 일반화의 강점 및 약점을 평가하여 AI와 인간의 협업을 강화하는 것입니다.



### Do LLMs Agree on the Creativity Evaluation of Alternative Uses? (https://arxiv.org/abs/2411.15560)
Comments:
          19 pages, 7 figures, 15 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 대안 사용 테스트(Alternative Uses Test, AUT)에 대한 응답의 창의성을 평가하는 데 동의하는지를 조사합니다. 이전 연구들은 주로 단일 모델이 동일한 모델이나 인간이 생성한 응답을 평가하는 데 초점을 맞추었지만, 본 연구에서는 모델들이 자신의 응답과 타 모델의 응답을 공정하고 정확하게 평가할 수 있는지를 탐구합니다. 이를 통해 LLM들이 창의성 평가에서의 편향성을 가지지 않음을 확인할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: 연구에서는 AUT 응답을 공통, 창의적, 매우 창의적인 세 가지 카테고리로 분류하여 LLM들이 이 응답들을 점수화(score)하고 순위를 매기는 실험적 프레임워크를 적용했습니다. 이 연구에서는 4개의 최신 LLM을 사용하며, 두 가지 평가 설정(포괄적 및 분할적)을 활용하여 모델들이 대안 사용의 창의성 평가에서 얼마나 합의하는지를 분석합니다. 연구 결과, 모델 간의 Spearman 상관계수는 평균 0.7 이상으로 높은 합의도를 보여주었고, 오라클과 비교 시에도 0.77을 초과하였습니다.

- **Performance Highlights**: 모델들은 자기 자신의 응답을 선호하지 않으며, 대신 다른 모델이 생성한 대안 사용에 대해 유사한 창의성 평가 점수나 순위를 제공합니다. 이러한 발견은 LLM들이 창의성 평가에서 공정성과 높은 일치를 보임을 시사하며, 이는 자동화된 창의성 평가에 대한 기대를 높입니다. 연구는 LLM들이 대안 사용의 창의성 평가에서 신뢰할 수 있는 평가자로 작용할 수 있음을 보여줍니다.



### Inducing Human-like Biases in Moral Reasoning Language Models (https://arxiv.org/abs/2411.15386)
Comments:
          Accepted to the 2nd Workshop on Unifying Representations in Neural Models (UniReps) at NeurIPS 2024

- **What's New**: 이 연구에서는 도덕적 추론(moral reasoning)을 위해 조정(fine-tuning)된 대형 언어 모델(LLMs)의 BrainScore(정신적 유사성 점수)를 인간의 행동 데이터와 뇌 데이터를 기반으로 분석합니다. 특히, 여러 LLM(BERT, RoBERTa, DeBERTa)을 fMRI 데이터를 사용하여 조정함으로써 BrainScore 향상을 시도합니다. 연구 결과, 대형 모델이 전반적으로 더 좋은 성능을 보였지만, 조정 후 BrainScore의 유의미한 개선은 관찰되지 않았습니다.

- **Technical Details**: 이 연구에서는 ETHICS 벤치마크의 상식 카테고리와 Koster-Hale의 fMRI 데이터셋을 사용하여 LLM의 도덕적 추론 성능을 정량적으로 측정합니다. 모델 조정은 HuggingFace 라이브러리를 통해 수행되었으며, 다양한 조건에서 450회의 조정 과정이 진행되었습니다. BrainScore는 예상 뇌 활동과 실제 뇌 활동 간의 Pearson 상관계수(PCC)를 이용하여 계산되었고, 이 과정에서 1,024개의 관심 영역(ROIs)의 시간 변화가 분석되었습니다.

- **Performance Highlights**: 결과적으로, fMRI 데이터로 조정하는 것이 ETHICS 벤치마크상의 정확도 향상에 일관되게 기여하지 않는 것으로 나타났습니다. 그러나 조정된 모델이 전반적으로 더 나은 성능을 보이는 경향이 있었으며, BrainScore 또한 주요 모델에 비해 유의미한 개선이 이루어지지 않았습니다. 따라서, 도덕적 추론에서의 두뇌-모델 정렬을 향상시키기 위해서는 더 많은 데이터 수집이 필요하다고 결론지었습니다.



### Designing Cellular Manufacturing System in Presence of Alternative Process Plans (https://arxiv.org/abs/2411.15361)
- **What's New**: 이 논문은 세포 제조 시스템(Cells Manufacturing Systems, CMS)의 설계 및 운영 단계에서 필요한 다양한 기술적 및 관리적 결정을 다룬다. 특히, 파트와 기계의 그룹화 문제를 해결하기 위한 네 가지 정수 프로그래밍(formulation) 모델을 소개하며, 각 부품이 여러 개의 프로세스 계획을 가지고 있을 때의 일반화된 그룹화 문제를 다룬다.

- **Technical Details**: 먼저, 부품과 기계의 그룹화 과정에서 모델은 각각의 부품 기종(machines)과 연속적인 작업(operations)을 최대한 동일한 셀(cell)과 기계에 배치하여 세포 간(cell간) 및 세포 내(intra-cell) 이동의 최소화를 목표로 한다. 이를 통해 제조 효율성을 높이고 자원의 활용도를 극대화할 수 있다.

- **Performance Highlights**: 논문에서는 그룹화 프로세스의 효율성을 입증하기 위해 수치 예제(numerical examples)를 포함하고 있으며, 세포 간 및 세포 내 이동 최소화의 목표가 기계 투자 비용(minimizing investment costs on machines)이나 운영 비용(operating costs) 최소화와 같은 다른 목표에 비해 얼마나 적합한지를 논의한다.



### Regulator-Manufacturer AI Agents Modeling: Mathematical Feedback-Driven Multi-Agent LLM Framework (https://arxiv.org/abs/2411.15356)
- **What's New**: 이 연구에서는 글로벌 의료기기 규제 환경에서의 동적 상호작용을 모델링하기 위해 대규모 언어 모델(LLM)과 다중 에이전트 모델링(multi-agent modeling)을 통합한 새로운 접근 방식을 소개합니다. 의료기기 제조업체와 규제 기관 간의 상호작용을 시뮬레이션하여 규제 변경이 산업의 결정-making 과정에 미치는 영향을 파악하고, 이를 통해 규제 관행의 개선과 혁신 촉진을 위한 전략적 기회를 모색합니다. 이 연구는 복잡한 규제 시스템의 변화에 대한 제조업체의 적응 전략을 실질적으로 지원하는 자료를 제공합니다.

- **Technical Details**: 연구는 복잡성 이론(Complexity Theory) 및 규제 흐름 이론(Regulatory Flow Theory)을 기반으로 하여 다중 에이전트 모델링을 이용해 의료기기 제조 및 규제 과정의 상호작용을 모델링합니다. 이 연구에서 다룬 다중 에이전트 시스템은 다양한 이해관계자의 행동을 시뮬레이션하여, 규제 변경이 제조업체의 준수 결정 및 시장 적응 전략에 미치는 영향을 분석합니다. 이론적 프레임워크를 통해 규제 업데이트의 피드백 메커니즘을 이해하고, 이해관계자의 동적 대응을 촉진합니다.

- **Performance Highlights**: 연구 결과는 규제 변화가 산업 행동에 미치는 영향을 조명하고, 제조업체의 혁신과 규제 준수를 지원하기 위한 전략을 개선하는 방법을 제시합니다. 규제 기관이 이러한 시스템의 emergent property를 인식하고 적응형 규제 프레임워크를 개발할 수 있도록 도와주며, 제조업체는 복잡한 규제 환경을 이해하여 전략적 계획을 강화할 수 있습니다. Ultimately, 이 연구는 안전한 의료기기를 환자에게 제공하면서도 혁신을 저해하지 않는 균형을 탐구하는 데 기여합니다.



### A No Free Lunch Theorem for Human-AI Collaboration (https://arxiv.org/abs/2411.15230)
- **What's New**: 이번 연구는 인간과 AI의 협업에서 기대되는 상호 보완성(complementarity)을 탐구하며, 이로 인해 성능 개선이 어떻게 이루어지는지에 대한 조건을 제시합니다. 특히, 연구는 확률 예측을 통해 이진 분류를 수행하는 여러 에이전트 간의 협력 전략의 한계를 밝혀냅니다. 제시된 결과는 일반적인 협업 모델에서 성능 극대화가 "공짜"로 이루어질 수 없음을 시사합니다.

- **Technical Details**: 연구에서는 n개의 에이전트가 입력 x에 대해 확률적 예측 P1(x), P2(x), ..., Pn(x) ∈ [0,1]을 생성하는 상황을 설정합니다. 각 에이전트는 확률 예측을 기반으로 이진 분류를 수행하며, 이 연구의 주요 결과는 모든 신뢰할 수 있는 협업 전략이 비협업적(non-collaborative)이라는 점입니다. 이는 협업을 통해 항상 가장 정확한 에이전트를 선택해야만 성능이 보장된다는 것을 의미합니다.

- **Performance Highlights**: 연구 결과, 협업 전략이 때때로 최대의 상호 보완성을 달성할 수 있지만, 실제로는 항상 하나의 에이전트에게 의존해야 한다는 점이 강조됩니다. 기댓값 평균, 다수결, 가장 신뢰도가 높은 에이전트의 분류를 따르는 방식은 모두 신뢰할 수 있는 전략이 아니며, 때때로 가장 정확하지 않은 에이전트보다 못한 성능을 보일 수 있습니다. 이러한 결과는 인간-AI 협업의 성공을 위한 필수 조건에 대한 통찰을 제공합니다.



### OPMOS: Ordered Parallel Multi-Objective Shortest-Path (https://arxiv.org/abs/2411.16667)
Comments:
          15 pages

- **What's New**: 이 논문은 Multi-Objective Shortest-Path (MOS) 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 기존의 알고리즘-아키텍처 접근 방식과 달리, OPMOS 프레임워크를 통해 순서가 있는 병렬성을 제공합니다. 이를 통해 다수의 경로를 동시에 효율적으로 실행하여 알고리즘의 작업 효율성을 높일 수 있습니다.

- **Technical Details**: MOS 문제는 시작 노드에서 목표 노드까지 여러 속성을 갖는 그래프에서 Pareto-optimal 솔루션을 찾는 NP-hard 문제입니다. OPMOS 프레임워크는 각 노드에서 부분 경로의 '프론티어'를 유지하며, 주문형 처리(ordered processing)를 수행하여 이루어집니다. 이 방법은 비지배(non-dominated) 경로의 수가 증가함에 따라 계산적으로 복잡해지는 문제를 해결하기 위한 것입니다.

- **Performance Highlights**: NVIDIA GH200 Superchip을 사용한 실험 평가에서 OPMOS는 높은 작업 효율성과 병렬성의 성능 확장 가능성을 보여주었습니다. 실제 선박 경로 최적화 응용 프로그램을 통해 이러한 이점이 입증되었습니다. 이로 인해 MOR 문제를 해결하는 데 있어 새로운 가능성을 제시하고 있습니다.



### CatNet: Effective FDR Control in LSTM with Gaussian Mirrors and SHAP Feature Importanc (https://arxiv.org/abs/2411.16666)
- **What's New**: CatNet이라는 알고리즘을 소개하며, 이는 Gaussian Mirror(GM) 방법을 활용해 False Discovery Rate(FDR)를 효과적으로 제어하고 LSTM에서 중요한 피처를 선택합니다. SHapley Additive exPlanations(SHAP) 값을 기반으로한 피처 중요성을 측정합니다. 이 연구는 LSTM 모델과 GM 알고리즘을 통합하여 피처 선택 및 오류 제어에서 중요한 진전을 보여줍니다.

- **Technical Details**: 이 연구에서는 SHAP 값의 도함수를 사용하여 LSTM의 피처 중요성을 평가하는 새로운 벡터를 도입했습니다. 또한, GM 알고리즘의 강력한 피처 선택을 위해 멀티콜리니어리티를 피할 수 있는 새로운 커널 기반 의존성 측정을 제안하며, 이는 LSTM과 결합하여 다양한 입력 차원에서의 안정성을 입증합니다.

- **Performance Highlights**: CatNet은 모의 데이터에서 선형 모델과 다양한 링크 기능을 가진 LSTM 모델 모두에서 FDR을 효과적으로 제어하면서 높은 통계적 파워를 유지하는 성능을 나타냈습니다. 실제 애플리케이션에서 S&P 500 지수 구성 요소의 가격을 예측하기 위한 다중 인자 투자 포트폴리오를 구성했으며, 기존 LSTM 모델과 비교하여 더 높은 예측 정확성을 달성했습니다.



### DreamRunner: Fine-Grained Storytelling Video Generation with Retrieval-Augmented Motion Adaptation (https://arxiv.org/abs/2411.16657)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 DreamRunner라는 새로운 스토리 비디오 생성 방법을 제안합니다. 이 방법은 길고 멀티 모션, 멀티 씬 비디오를 생성할 수 있으며, 입력 텍스트 스크립트에서 설명된 스토리를 일관되게 표현하는데 중점을 둡니다. DreamRunner는 LLM(대규모 언어 모델)을 사용해 입력 스크립트를 구조화하고, 리트리벌(검색) 기반의 적응 방법을 통해 비디오 생성 과정에서 동작을 커스터마이즈(purpose)할 수 있습니다.

- **Technical Details**: DreamRunner 프레임워크는 세 가지 주요 프로세스로 구성됩니다: 1) 이중 수준 비디오 계획 생성, 2) 동작 리트리벌 및 주제/동작 사전 학습, 3) 공간-시간 기반의 3D 주의 및 사전 주입 모듈(SR3AI). 첫 번째 단계에서는 사용자 제공의 스토리 내러티브를 기반으로 고수준 및 세부적인 계획을 수립하며, 두 번째 단계에서는 비디오 데이터베이스에서 동작과 관련된 비디오를 검색하여 동작 사전을 학습합니다. 마지막 단계에서는 상세한 프레임 제어와 원활한 동작 전환을 가능하게 하는 SR3AI 모듈을 도입합니다.

- **Performance Highlights**: DreamRunner는 T2V-ComBench에서 기존의 최첨단 방식들에 비해 캐릭터 일관성(CLIP 점수)에서 13.1%의 상대적 개선을 보이고, 텍스트 추적 능력(ViCLIP 점수)에서도 8.56%의 향상을 기록했습니다. 또한, 동일 씬 내에서의 사건 전환의 매끄러움(DINO 점수)에서도 27.2% 개선을 보이며 효과성을 입증했습니다. DreamRunner는 오픈 소스 모델을 기반으로 하면서도 폐쇄형 모델에 비해 동적 속성 바인딩에서 최고의 성과를 달성했고, 캐릭터 상호작용에서도 경쟁력 있는 결과를 보여주며 오픈 소스의 가능성을 보여주고 있습니다.



### Self-Generated Critiques Boost Reward Modeling for Language Models (https://arxiv.org/abs/2411.16646)
Comments:
          20 pages

- **What's New**: 이번 논문에서는 Critic-RM이라는 새로운 프레임워크를 소개합니다. Critic-RM은 기존의 보상 모델이 단일 스칼라 점수를 생성하는 한계를 극복하고, 자가 생성된 비평을 활용하여 보상 모델을 향상시키는 방법을 제시합니다. 이 프레임워크는 추가적인 감독 없이 자가 비평 생성력(critique generation ability)을 활용하여 보상 모델의 정확성을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: Critic-RM은 비평을 생성하고 이를 필터링하는 두 단계를 통해 작동합니다. 이 과정에서 품질이 높은 비평만을 선택하여 보상 예측과 비평 생성의 공동 세부 조정을 실시합니다. 또한, LLM이 생성한 여러 후보 비평을 평가하여 인간 주석과의 일치성을 기반으로 품질을 개선하는 일관성 유도 필터링(consistency-guided filtering) 기법을 적용합니다.

- **Performance Highlights**: 실험 결과 Critic-RM은 기존 보상 모델 및 LLM 평가자에 비해 보상 모델링 정확도가 3.7%-7.3% 향상되었음을 보여줍니다. 추가 연구를 통해生成된 비평이 LLM의 결함 있는 추론 단계를 정정하는 데에도 효과적이었으며, 추론 정확도를 2.5%-3.2% 향상시키는 데 기여했습니다.



### Recommender Systems for Good (RS4Good): Survey of Use Cases and a Call to Action for Research that Matters (https://arxiv.org/abs/2411.16645)
- **What's New**: 이 논문에서는 추천 시스템(recommender systems) 연구 커뮤니티가 사회적 선(good)을 증진하는 영역, 즉 RS4Good에 더 중점을 두어야 한다고 주장하고 있습니다. 많은 추천 알고리즘이 영화 및 전자상거래와 같은 좁은 응용 분야에 집중되었고, 사용자와의 실제 평가가 부족하다는 점이 강조됩니다. 이와 함께, 추천 시스템이 건강한 생활 습관이나 에너지 절약 행동을 촉진할 수 있는 가능성을 제시합니다.

- **Technical Details**: 추천 시스템은 보통 최적의 사용자를 대상으로 정보 과부하를 피하거나 기업의 매출을 증가시키기 위해 설계됩니다. 최근 몇 년 동안 이러한 시스템이 부정적인 영향을 미칠 수 있다는 인식이 높아지며, 다양한 연구 및 이니셔티브가 '책임 있는 추천(responsible recommendation)'이라는 개념 하에 이루어지고 있습니다. 이에 따라 공정성(fairness), 개인 정보 보호(privacy), 해석 가능성(interpretability) 등의 요소가 고려되어야 합니다.

- **Performance Highlights**: 기존의 추천 시스템이 사용자 참여를 극대화하려는 목표로 인해 논란이 있는 콘텐츠를 추천할 수 있다는 우려가 제기되었습니다. 이러한 시스템은 단기적인 비즈니스 지표를 높일 수 있지만 사회적인 문제를 악화시킬 수 있습니다. RS4Good는 단순히 문제를 피하기보다는 사회적 가치를 창출하는 것을 목표로 하며, 경제적 가치를 동시에 추구할 수 있습니다.



### Do Automatic Factuality Metrics Measure Factuality? A Critical Evaluation (https://arxiv.org/abs/2411.16638)
- **What's New**: 현대의 LLMs(대형 언어 모델)는 높은 품질의 요약을 생성할 수 있지만, 여전히 원하는 정보를 요약에 추가하는 오류를 범할 수 있습니다. 이러한 오류를 자동으로 측정하는 것이 어려웠으며, 이에 따라 생성된 요약의 사실적 일관성을 평가하기 위한 다양한 메트릭이 개발되었습니다. 본 연구는 이러한 자동화된 사실 정확성 메트릭을 심층 분석하고, 그 신뢰성에 관한 의문을 제기합니다.

- **Technical Details**: 이 연구에서는 사실 적합성(factuality)을 평가하기 위해 다양한 메트릭을 스트레스 테스트했습니다. 기본적인 속성만으로도 사실 적합성을 예측할 수 있음을 발견하였으며, 간단한 MLP(다층 퍼셉트론) 모델이 복잡한 SOTA(최신 기술 동향) 메트릭과 경쟁할 수 있음을 제시했습니다. 또한, 일부 메트릭들은 사실적인 수정에 반응하지만, 무관한 수정에 더 민감하게 반응하는 경향이 있습니다.

- **Performance Highlights**: 자동화된 사실 적합성 메트릭들은 무의미한 수정에 민감해 표면적인 신호에 의존하는 경향이 있습니다. 이는 '게임 내기'(game-making)를 통해 메트릭 값을 인위적으로 증가시킬 수 있음을 시사합니다. 결과적으로, 기존 메트릭에 대한 신뢰성을 확립하는 데에 심각한 문제가 있음을 나타내며, 향후 이 메트릭들에 대한 신뢰성 향상이 필요합니다.



### Inference-Time Policy Steering through Human Interactions (https://arxiv.org/abs/2411.16627)
- **What's New**: 이 논문에서는 Inference-Time Policy Steering (ITPS) 프레임워크를 제안하여 인간의 상호작용을 활용하여 생성 샘플링 프로세스를 조정합니다. 이 방법은 정책을 재조정하지 않고, 사전 훈련된 정책이 사용자 의도에 부합하는 행동을 생성하도록 유도합니다. 이를 통해 제약 조건의 위반 또는 실행 실패를 줄이면서도 정책 출력을 사용자 지향적으로 조정할 수 있습니다.

- **Technical Details**: ITPS는 정책 스티어링을 조건부 샘플링으로 프레임화하며, 이는 학습된 생성 정책의 가능도 분포에서 조건부 샘플링을 수행합니다. 이 프레임워크는 사용자로부터 얻은 상호작용을 기반으로 의미 있는 정보를 정책에 통합하는 방식으로, 주어진 작업 공간에서 각기 다른 사용자 목표에 맞춰 행동을 조정합니다. 다양한 상호작용 유형에 따라 정책의 수행 방식을 변화시킴으로써, 이를 통해 사용자 맞춤형 목표에 대한 유연한 연계가 가능합니다.

- **Performance Highlights**: 세 가지 실험을 통해 ITPS의 효율성을 평가했으며, 여섯 가지 샘플링 전략 중 이론적으로 가장 뛰어난 성과를 보인 것은 확산 정책(diffusion policy)과 결합된 확률적 샘플링 방식입니다. 이 새로운 접근 방식은 사용자 의도에 대한 정렬을 극대화하고, 제약 조건의 만족도를 유지하는 최적의 균형을 달성했습니다. 또한 이 연구는 인간의 의도를 더 잘 이해할 수 있는 다양한 알고리즘적 기법을 제공하여 실시간 로봇 상호작용에 기여할 것으로 기대됩니다.



### Imperceptible Adversarial Examples in the Physical World (https://arxiv.org/abs/2411.16622)
- **What's New**: 이번 연구는 Deep Neural Network (DNN) 기반의 컴퓨터 비전 모델에 대한 공격에서 물리적인 환경에서도 식별할 수 없는 적대적 예제(adversarial examples)를 생성하는 새로운 접근 방식을 제안합니다. 이 연구는 Straight-through Estimator (STE) 기술을 활용하여 비분화 노이즈(non-differentiable distortions)가 존재하는 실제 환경에서도 효과적인 공격을 가능하게 합니다. 이로 인해 과거의 공격 방식들보다 더 강력하고 유의미한 적대적 공격이 가능해졌습니다.

- **Technical Details**: 연구지는 STE를 사용하여 비분화 왜곡 방식을 갖는 비주얼 센싱 시스템에 대한 적대적 예제를 생성하는 과정을 설명합니다. 이 기술은 후방 전파(backpropagation) 과정에서 정체성 함수(identity function)와 차별화 렌더링(differentiable rendering)을 결합하여 사용함으로써 물리적 환경에서도 눈에 띄지 않는 공격을 가능하게 합니다. 이를 통해 인쇄된 사진이나 CARLA 시뮬레이터에서의 실험을 통해 $ℓ_	ext{∞}$으로 제한된 적대적 예제를 신속하게 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: STE 기술을 활용한 본 연구의 성과는 실제 환경에서 적대적 공격의 효과가 디지털 환경의 전통적인 공격과 유사하거나 더욱 우수하다는 것입니다. 연구에서는 AP50의 정확도가 43.29%에서 4.22%로 감소하는 등, 물리적 공간에서 비가시적인 적대적 패치가 어떻게 타겟 모델의 분류 정확도를 제로로 만드는지에 대한 실험 결과를 제시합니다. 이러한 성과는 DNN 기반의 비주얼 센싱 시스템의 보안 위험성을 재평가할 필요성을 강조합니다.



### Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision (https://arxiv.org/abs/2411.16579)
Comments:
          Preprint

- **What's New**: 이 논문에서는 두 플레이어 파라다임을 제안하여, reasoning (추론) 모형과 critique (비판) 모형의 역할을 분리합니다. AutoMathCritique라는 자동화된 프레임워크를 통해 76,321개의 응답과 단계별 피드백 데이터셋을 생성하였고, 이는 언어 모델을 fine-tuning (미세조정) 하는 데 사용됩니다. 이러한 비판 모형은 복잡한 질문에 대한 actor (행위자) 모델의 성능을 개선하는 데 효과적임을 입증하였습니다.

- **Technical Details**: AutoMathCritique 프레임워크는 오류가 있는 추론 경로를 구성하고, 비판을 생성하며, 데이터를 필터링하는 세 가지 주요 단계로 이루어져 있습니다. 이 프레임워크의 결과로 생성된 데이터셋인 MathCritique-76k는 76,321개의 샘플로 언어 모델을 fine-tune하는 데 활용되었습니다. 논문은 비판 모형이 actor 모델의 탐색 효율성 및 문제 해결 다양성을 어떻게 향상시키는지를 심층적으로 분석하였습니다.

- **Performance Highlights**: 비판 모형은 특히 어려운 질문에 대해 actor 모형의 성능을 일관되게 향상시켰으며, 탐색 과정에서의 효과적인 감독을 통해 더 나은 결과를 가져왔습니다. 결과적으로, critique-in-the-loop self-improvement method (비판 루프 자가 개선 방법)은 actor 모델이 스스로 개선할 수 있는 과정에서 비판 모델과의 협업을 통해 더 강력한 추론 모델을 만드는 데 기여하였습니다. 마지막으로, self-talk-via-critique 방법론을 통해 단계별 자기 반성 및 수정이 가능하다는 가능성을 보여주었습니다.



### Naive Algorithmic Collusion: When Do Bandit Learners Cooperate and When Do They Compete? (https://arxiv.org/abs/2411.16574)
Comments:
          To be published in proceedings of International Conference on Information Systems 2024

- **What's New**: 이 논문은 다중 무장 도적(bandit) 기계 학습 알고리즘을 이용해 에이전트들이 경쟁하는 상황에서, 그들이 전략적 상호작용에 대한 정보 없이도 지속적으로 연합 행동(collusion)을 학습하는 것을 보여줍니다. 저자들은 이를 "순진한 연합(naive collusion)"이라고 명명하며, 이를 일반적인 반복 Prisoner's Dilemma 게임을 통해 연구합니다. 이 연구는 규제 당국에 중요한 정책적 시사점을 제공하므로 주목할 필요가 있습니다.

- **Technical Details**: 자동 가격 책정 알고리즘은 Amazon 시장에서 경쟁 제품의 가격 설정부터 주택 임대료 결정까지 다양한 맥락에서 광범위하게 사용되고 있습니다. 다중 무장 도적 알고리즘은 에이전트가 자신의 행동에 따른 결과를 모르는 상태에서 각 행동의 가치를 추정하며 이를 바탕으로 의사 결정을 내리는 방식으로 작동합니다. 연구에서 나타난 결과는 대칭적인 에이전트가 결정론적 알고리즘을 사용하는 경우 연합 행동이 항상 나타나고, 비결정론적 알고리즘을 사용할 때는 장기적으로 연합 행동이 발생하지 않는다는 점입니다.

- **Performance Highlights**: 이 연구는 특정 알고리즘 사용 조건에 따라 연합 행동이 얼마나 다르게 나타날 수 있는지를 강조합니다. 예를 들어, 알고리즘에 소량의 비대칭성을 추가하면 연합 행동을 방지하기에 충분하지 않을 수 있음을 보여줍니다. 이를 통해 규제 기관은 알고리즘이 상호작용하는 방식이 경쟁의 결과에 미치는 영향을 보다 깊이 이해해야 할 필요성이 제기됩니다.



### Representation Collapsing Problems in Vector Quantization (https://arxiv.org/abs/2411.16550)
Comments:
          13 pages, under review

- **What's New**: 본 연구는 Vector Quantization (VQ)에서 발생할 수 있는 representation collapsing 문제를 체계적으로 분석하고 해결책을 제시합니다. VQ는 기계 학습에서 널리 사용되지만, 생성 모델에서의 특성과 동작은 충분히 탐구되지 않았습니다. 이 연구는 두 가지 유형의 collapsing인 tokens collapse와 embeddings collapse에 대한 심층 분석을 제공합니다. 또한, 이는 VQ의 성능 저하를 초래하는 주요 원인을 규명하고, 각 문제에 대한 해결 방안을 마련하고자 합니다.

- **Technical Details**: VQ-VAE 구조에서, Encoder (부호화기) Eθ와 Decoder (복원기) Dθ를 통해 원시 데이터 X를 연속 표현 Z로 매핑합니다. 이 구조에서, token 집합 𝒯는 코드북을 구성하여 이산화된 표현을 저장하는 역할을 합니다. 본 연구는 tokens collapse가 주로 미훈련된 Encoder에서 시작되는 비효율적인 초기화로 인해 발생한다는 점을 강조하고, 이를 해결하기 위한 사전훈련(pretraining) 및 미세조정(fine-tuning) 방법을 제안합니다. 또한, embeddings collapse는 Encoder의 파라미터 수가 부족하여 발생하는데, Encoder의 파라미터 수를 증가시키는 것이 해결책이 될 수 있음을 보여줍니다.

- **Performance Highlights**: 연구에서 제안한 해결책은 tokens collapse와 embeddings collapse 문제를 완화하는 방향으로 성과를 달성하였습니다. 특히, tokens의 수를 증가시키는 것이 성능 향상으로 이어지며, 기존 기준선을 초과하는 결과를 보였습니다. 또한, Encoder의 파라미터 수를 증가시킴으로써 embeddings collapse 문제의 효과적 해결이 가능함을 입증했습니다. 이 연구는 Vector Quantization 기법의 향후 발전 가능성을 제시하는 중요한 기초 작업으로 평가됩니다.



### RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics (https://arxiv.org/abs/2411.16537)
- **What's New**: 이 논문은 로봇 분야에서의 공간 이해 능력 향상을 위해 RoboSpatial이라는 대규모 데이터셋을 소개합니다. 이 데이터셋은 3D 스캔과 에고세닉(ego-centric) 이미지로 구성된 실내 및 테이블 씬을 기반으로 하며, 로봇과 관련된 풍부한 공간 정보를 포함하고 있습니다. RoboSpatial은 100만 개의 이미지, 5,000개의 3D 스캔, 300만 개의 주석 공간 관계를 포함하여 로봇이 더 잘 이해할 수 있도록 다양한 질문-답변 쌍을 제공합니다.

- **Technical Details**: RoboSpatial 데이터셋은 세 가지 유형의 질문을 제공합니다: (1) 공간 구성, (2) 공간 맥락, (3) 공간 호환성. 각 질문 유형은 서로 다른 관점에서 공간 관계를 이해하도록 설계되어 있으며, 관점은 에고세닉, 객체 중심(object-centric), 세계 중심(world-centric)으로 나뉩니다. 이러한 다각적인 접근 방식은 로봇이 복잡한 공간 지침을 더욱 유연하게 처리할 수 있도록 합니다.

- **Performance Highlights**: RoboSpatial을 사용하여 훈련된 비전-언어 모델(VLM)은 기존 모델보다 공간 추론 능력이 상당히 향상되었습니다. 실험 결과, RoboSpatial에서 학습한 모델은 다양한 로봇 조작 작업 및 실내 씬 질문 응답에서 우수한 성능을 보였습니다. 이 데이터셋의 3D 준비 디자인은 VLM의 공간 추론 능력을 높이는 데 기여하며, 실제 공간 작업에서의 차별적인 성능을 보여줍니다.



### Fundamental Limits of Prompt Tuning Transformers: Universality, Capacity and Efficiency (https://arxiv.org/abs/2411.16525)
- **What's New**: 이번 연구에서는 transformer 기반의 foundation 모델을 위한 prompt tuning의 통계적 및 계산적 한계를 조사하였습니다. 주요 기여로는 단일 헤드(single-head) transformer에 단일 self-attention layer만을 사용하는 prompt tuning 이론을 제시하는 것입니다. 이 과정에서 우리는 이 방식이 보편적(universal)이며, Strong Exponential Time Hypothesis (SETH) 하에서 효율적인 알고리즘을 지원한다고 주장합니다.

- **Technical Details**: 우리는 단순한 transformer에서의 prompt tuning이 시퀀스-투-시퀀스 Lipschitz 함수에 대한 보편적 근사기(universal approximators)임을 증명하였습니다. 또한 1-layer 및 1-head transformer로 데이터셋을 기억하기 위해 필요한 soft-prompt 토큰에 대한 하한선(lower bound)을 제공했습니다. 이를 통해 prompt tuning의 효율성에서의 phase transition을 규명하고, soft-prompt-induced 키(key)와 쿼리(query)의 노름(norm)에 의해 결정된 최대 경계 조건을 설정했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 설정한 경계 조건을 초과하는 경우 SETH 하에서 효율적인 알고리즘이 존재하지 않음을 보여주고 있습니다. 그 경계 조건 내에서 우리는 거의 선형 시간(almost-linear time) prompt tuning 추론 알고리즘의 존재를 입증하여 이론을 구체화하였습니다. 이러한 기본 한계는 실무자를 위한 표현력 높고 효율적인 prompt tuning 방법 설계 시 필수적인 조건을 제공한다는 점에서 중요합니다.



### Interpreting Language Reward Models via Contrastive Explanations (https://arxiv.org/abs/2411.16502)
- **What's New**: 본 연구는 언어 보상 모델(Reward Models, RMs)의 예측을 설명하기 위해 대조적 설명(contrastive explanations) 패러다임을 적용했습니다. 기존의 연구에서 RMs의 행동을 설명하는 방법은 부족했으며, 본 연구는 새로운 비교 세트를 생성하여 RMs의 로컬 행동을 명확히 특징지었습니다. 이 방법은 설명 가능하고 신뢰할 수 있는 LLM 정렬을 위한 기초를 제공하는 데 중요한 기여를 합니다.

- **Technical Details**: RMs는 일반적으로 스칼라 출력을 가지는 수정된 대형 언어 모델로, 이로 인해 예측이 불투명하게 됩니다. 본 연구에서는 변형된 응답 간의 비교를 통해 대조적 설명을 생성하고, 이를 통해 두 가지 유형의 예를 제시합니다: 반사실적(counterfactual, CF)과 준사실적(semi-factual, SF) 설명. 이러한 설명은 RM의 로컬 행동과 감도를 이해하는 데 도움을 주며, 각 고급 평가 속성의 민감도를 정량적으로 평가합니다.

- **Performance Highlights**: 정량적 실험을 통해, 제안한 방법이 질 높은 대조적 설명을 찾는 데 효과적임을 확인했습니다. 또한, 전반적인 분석과 결합하여 RM의 특성과 고급 평가 속성에 대한 감도를 정량화하는 방법을 보여줍니다. 연구의 결과들은 RMs의 행동을 설명하는 데 유용한 정보를 제공하고, 각 평가 속성에 대한 민감도를 정리하여 대표적인 예를 찾아내는 프로세스를 제안합니다.



### O1 Replication Journey -- Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson? (https://arxiv.org/abs/2411.16489)
Comments:
          16 pages

- **What's New**: 이 논문은 OpenAI의 O1 모델 복제와 관련된 현재 접근 방식에 대한 비판적 검토를 제공합니다. 특히, 지식 증류(knowledge distillation) 기술의 널리 퍼졌지만 종종 공개되지 않은 사용에 집중하고 있습니다. 이전 연구에서는 O1 복제의 기본 기술 경로를 탐구했으며, 이번 연구는 단순한 증류와 감독된 파인 튜닝(supervised fine-tuning)이 복잡한 수학적 추론 작업에서 우수한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 이)의 시도들을 평가하는 포괄적인 벤치마크 프레임워크를 소개합니다. 프레임워크는 기술적 투명성과 재현 가능성의 기준으로 O1 복제 시도를 평가하기 위한 명확한 메트릭을 제공합니다. 또한, 지식 증류 기술에 대한 논의와 이 접근 방식의 잠재적 위험도 다룹니다. 이러한 분석 결과는 단순히 성능 향상을 추구하는 것 이상의 의미를 가지고 있음을 강조합니다.

- **Performance Highlights**: 연구 결과, 적은 표본(수만 개)의 O1 증류 샘플을 사용하고 표준 감독 파인 튜닝을 통해, 기본 모델이 American Invitational Mathematics Examination (AIME)에서 O1-preview의 성능을 초과할 수 있음을 입증했습니다. 이는 기술적 복잡성이 적게 드는 방법으로, 수학 문제 해결 데이터만으로도 다양한 작업에서 강력한 일반화 능력을 보여주었습니다. 이러한 성과에도 불구하고, 증류 접근 방식의 과도한 의존성은 장기적으로 혁신의 정체를 초래할 수 있다는 점에서 우려를 표합니다.



### When Babies Teach Babies: Can student knowledge sharing outperform Teacher-Guided Distillation on small datasets? (https://arxiv.org/abs/2411.16487)
Comments:
          Accepted to BabyLM challenge, CoNLL Workshop, EMNLP 2024

- **What's New**: 이번 연구는 BabyLM 챌린지에 대한 우리의 제출물을 소개하며, 데이터 효율적인 언어 모델 프리트레이닝의 경계를 확장하는 것을 목표로 합니다. 본 방법은 심층 상호 학습(deep mutual learning)을 기반으로 하며, 다양한 초기화를 위한 학생 모델 탐색(student model search)을 도입합니다. 여기에 학생을 동등하게 대하는 한계를 극복하기 위해 가중치 상호 학습(weighted mutual learning)을 이차 최적화 문제(bi-level optimization problem)로 형성하였습니다.

- **Technical Details**: 본 연구 방법론은 RoBERTa-base 모델(125M parameters)을 절반 이하의 크기로 증류하면서 성능을 유지하는 것에 중점을 두고 있습니다. 베이지안 최적화(Bayesian optimization)를 통해 학생 모델의 아키텍처를 선택하였으며, 기존의 교사-학생 증류 대신 가중치 상호 학습을 탐구합니다. 이 과정은 두 개의 루프로 구성되어 있으며, 내부 루프는 온라인 증류(online distillation)를 통해 압축된 학생 모델을 학습하고, 외부 루프는 다양한 학생으로부터의 지식을 효과적으로 증류하기 위해 가중치를 최적화합니다.

- **Performance Highlights**: 우리의 평가 결과, 교사 모델 없이도 기존의 교사-지도 접근 방법과 유사하거나 우수한 성능을 나타냅니다. 이 연구는 전통적인 자율 학습(independent learning)보다 여러 대규모 네트워크 간의 성능을 향상시키는 것을 확인하였으며, 교사 모델의 필요성을 줄이면서도 상당한 성능 향상이 가능하다는 것을 보여줍니다. 가중치 상호 학습 전략이 네트워크의 학습 효율을 높임으로써, 데이터 효율성과 메모리 효율성이 중요한 경계 시스템에서 특히 큰 장점을 가질 수 있음을 시사합니다.



### Characterized Diffusion Networks for Enhanced Autonomous Driving Trajectory Prediction (https://arxiv.org/abs/2411.16457)
Comments:
          7 pages, 0 figures

- **What's New**: 이번 논문에서는 자율주행을 위한 새로운 궤적 예측 모델을 소개합니다. 이 모델은 Characterized Diffusion Module과 Spatial-Temporal Interaction Network를 결합하여 동적이고 이질적인 교통 환경의 문제를 해결합니다. 특히 불확실성 추정(uncertainty estimation)과 복잡한 에이전트 상호작용을 통합하여 궤적 예측의 정확성과 신뢰성을 향상시킵니다.

- **Technical Details**: 모델은 NGSIM, HighD 및 MoCAD와 같은 공공 데이터셋에서 광범위한 실험을 통해 기존의 최첨단 방법들보다 우수한 성능을 보였습니다. 이 모델은 교통 시나리오의 공간-시간 역학(spatial-temporal dynamics)을 포착할 수 있는 능력을 보여줍니다. 또한 복잡한 환경에서 예측의 정밀성을 더욱 개선합니다.

- **Performance Highlights**: 제안된 모델은 실제 자율주행 시스템에 적용될 수 있는 강력한 잠재력을 보여줍니다. 실험 결과는 모델이 다양한 시나리오에서 뛰어난 궤적 예측 능력을 발휘함을 나타내며, 이는 자율주행 기술의 발전에 중요한 기여를 할 것으로 기대됩니다.



### TIFeD: a Tiny Integer-based Federated learning algorithm with Direct feedback alignmen (https://arxiv.org/abs/2411.16442)
- **What's New**: 본 논문에서는 TIFeD(Tiny Integer-based Federated Learning with Direct Feedback Alignment)라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 저전력 장치에서 메모리와 연산 에너지의 제약을 극복하며, 외부 클라우드 서비스에 의존하지 않고 직접적으로 훈련할 수 있도록 설계되었습니다. TIFeD는 전체 신경망이 아닌 일부 레이어만 훈련할 수 있게 하여, 학습 과정에서의 정보 전송량을 줄여 에너지를 절약할 수 있습니다.

- **Technical Details**: TIFeD 알고리즘은 정수 전용 산술(integer-only arithmetic)로 구현되어 메모리 소모를 줄이고, 부동 소수점 연산 없이도 모델 훈련을 가능하게 합니다. Direct Feedback Alignment(DFA) 방식을 기반으로 하여, 각 숨겨진 레이어가 독립적으로 훈련되는 구조를 갖추고 있습니다. 이러한 방식은 전통적인 역전파(backpropagation) 알고리즘보다 계산 비용을 줄이는 데 기여합니다.

- **Performance Highlights**: 실험 결과는 TIFeD 알고리즘이 자원이 제한된 장치에서 효과적으로 작동함을 보여줍니다. 이 알고리즘은 기존의 TinyML 접근 방법의 장점을 학습 단계에서도 활용할 수 있도록 하여, 장치의 로컬 데이터로부터 모델을 직접 개선할 수 있습니다. TIFeD의 공개 레포지토리는 과학 커뮤니티에 크게 기여할 것으로 기대됩니다.



### Unsupervised Event Outlier Detection in Continuous Tim (https://arxiv.org/abs/2411.16427)
- **What's New**: 이번 연구에서는 인간의 감독 없이 이상치(outlier) 탐지를 위한 최초의 비지도 학습 방법을 개발하였습니다. 특히, Generative Adversarial Networks (GANs)와 Reinforcement Learning (RL) 개념을 기반으로 하여, 데이터 내 이상 포인트를 자동으로 감지하고 수정할 수 있는 프로세스를 제안합니다. 이 방법은 기존의 예측 작업과는 달리, 사건 발생의 비정상적인 패턴을 효과적으로 포착할 수 있습니다.

- **Technical Details**: 연구에서는 이벤트 시퀀스를 S={tn:tn∈𝒯}로 정의하며, 여기서 tn은 n번째 사건 발생 시점입니다. GAN 기반의 생성기(generator)와 판별기(discriminator) 구조를 통해 데이터를 수정하고, 수정된 데이터와 실제 데이터를 구분합니다. 또한, 이 과정에서 이전 이벤트의 정보가 새로운 이벤트의 발생에 영향을 미칠 수 있음을 고려하여 continuous-time LSTMs (cLSTM) 아키텍처를 사용하고, 주의(attention) 메커니즘을 통해 과거의 모든 포인트 정보를 활용합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 최첨단(outstanding) 방법들보다 더 높은 정확도로 이벤트 이상치를 탐지할 수 있음을 보여주었습니다. 특히, RL 기반의 에이전트를 통해 매 이벤트마다 지속적으로 이상치를 식별하고 제거하는 작업이 가능하여, 이는 비지도 학습의 잠재력을 극대화합니다. 이러한 접근법은 다양한 이벤트 시퀀스 데이터를 실시간으로 처리하는 데 유용할 것입니다.



### TopV-Nav: Unlocking the Top-View Spatial Reasoning Potential of MLLM for Zero-shot Object Navigation (https://arxiv.org/abs/2411.16425)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 TopV-Nav라는 새로운 MLLM 기반(MLLM-based) 방법을 소개합니다. 이 방법은 에이전트가 생소한 환경에서 이전에 보지 못한 객체를 찾는 Zero-Shot Object Navigation (ZSON) 임무를 수행할 수 있도록 도와줍니다. 특히 TopV-Nav는 기존의 시각적 관찰을 언어 기술로 변환하는 대신, 직접적으로 상위 뷰 맵(top-view map)에서 공간 정보를 활용하여 추론을 수행합니다. 그 결과 공간 정보의 손실을 방지하고 보다 효과적인 탐색이 가능해집니다.

- **Technical Details**: TopV-Nav는 Adaptive Visual Prompt Generation (AVPG) 방법을 포함하여 에이전트가 상위 뷰 맵에서 공간 정보를 적극적으로 활용할 수 있도록 합니다. AVPG는 맵에 시맨틱 정보를 추가하며, 이는 에이전트가 환경 내 객체들의 위치 관계를 보다 잘 이해하게 합니다. 또한, Dynamic Map Scaling (DMS) 메커니즘을 통해 에이전트는 필요에 따라 맵의 확대 및 축소를 조절하여 지역적 세밀한 추론을 강화할 수 있습니다. 이 외에도 Target-Guided Navigation (TGN) 메커니즘을 통해 미지의 영역에서도 목표 위치를 예측하고 이를 탐색 전략에 활용할 수 있습니다.

- **Performance Highlights**: MP3D와 HM3D 벤치마크에서의 실험 결과, TopV-Nav는 HM3D에서 3.9%의 SR(Success Rate)과 2.0%의 SPL(Success Path Length) 향상을 달성하여 기존 방법들을 초월하는 성능을 보였습니다. 새로운 메커니즘인 AVPG, DMS, TGN의 도입이 이러한 성능 향상에 기여했습니다. 이 연구는 MLLM의 공간적 추론 능력을 극대화함으로써 제로샷 탐색 임무의 새로운 가능성을 열었습니다.



### Turbofan Engine Remaining Useful Life (RUL) Prediction Based on Bi-Directional Long Short-Term Memory (BLSTM) (https://arxiv.org/abs/2411.16422)
- **What's New**: 이 논문은 항공 산업에서의 기술 발전에 따른 제트 엔진의 변화와 복잡성을 강조합니다. 특히 상업용 터보팬 엔진(turbofan engine)의 RUL(remaining useful life) 예측의 중요성을 다루고 있으며, 데이터 기반 접근 방식의 장점을 설명합니다. Bi-Directional Long Short-Term Memory(BLSTM) 모델을 중심으로 RUL 예측 모델을 연구하고 있습니다.

- **Technical Details**: 터보팬 엔진의 각 구성 요소는 운영 기간 동안 변화(변형)에 취약하며, 이러한 변화를 정확히 예측하는 것은 비행 안전(safety) 및 비용 효율성(cost effective operations)에 중요합니다. 논문은 BLSTM 모델을 사용해 RUL을 예측하는 방법을 제안하고, CMAPSS(Commercial Modular Aero-Propulsion System Simulation)라는 엄선된 데이터셋을 활용해 모델 성능을 평가합니다. 다양한 RUL 예측 데이터 기반 모델의 기준을 설정하고 비교합니다.

- **Performance Highlights**: 논문에서 제안하는 BLSTM 모델의 성능은 NASA의 CMAPSS 데이터셋을 통해 평가되어, 실제 엔진 고장 예측을 위한 유용한 데이터셋으로 작용합니다. 이를 통해 ML 기술의 발전을 통한 데이터 기반 접근 방식의 신뢰성을 높이고, 터보팬 엔진의 성능 유지 및 안전성을 증진하는 데 기여합니다. 연구 결과는 향후 엔진 유지보수 및 관리 전략 개선에도 중요한 영향을 미칠 것으로 예상됩니다.



### Low-Data Classification of Historical Music Manuscripts: A Few-Shot Learning Approach (https://arxiv.org/abs/2411.16408)
Comments:
          6 pages, The Sixth IEEE international conference on Image Processing Applications and Systems

- **What's New**: 이 논문에서는 역사적 악보의 음악 기호 분류를 위한 자기 지도 학습(framework) 프레임워크를 개발하여 기술과 문화 보존의 교차점을 탐구합니다. 전통적인 Optical Music Recognition (OMR) 방법은 레이블이 있는 데이터 부족으로 어려움을 겪고 있는데, 본 연구는 레이블이 없는 데이터에서 훈련된 신경망 기반(feature extractor) 특징 추출기를 통해 이 문제를 극복하려 합니다. 주요 기여로는 자기 지도 CNN의 크롭 전처리를 최적화하고 SVM, 다층 퍼셉트론(MLP), 프로토타입 네트워크(prototypical networks) 등의 분류 방법을 평가한 점이 있습니다.

- **Technical Details**: 본 연구는 자기 지도 CNN을 기반으로 하여 음악 기호 분류를 위한 특징을 추출하는 방법을 제시합니다. 역사적 악보의 기호들은 크기, 간격, 문서 노후화 등 다양한 변동성을 보이므로, 문서를 세부 섹션(크롭)으로 나누고 이 과정을 슬라이딩 윈도우(sliding window) 방법으로 처리하여 진행합니다. VICReg 방법을 적용하여 CNN은 각 크롭을 두 번 왜곡하여 특징 공간에서 동일한 포인트에 매핑하도록 훈련되며, 이를 통해 역사적 문서에서 발생하는 기호의 가변성에 효과적으로 대처합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제시한 방법은 87.66%의 분류 정확도를 기록하여 AI 기반 방법들이 역사적 음악의 보존 가능성을 확인했습니다. MLP는 다양한 분류 알고리즘 중 가장 우수한 성능을 보였으며, 고차원 데이터와 제한된 라벨 데이터 처리에 있어 강력한 일반화 능력을 나타냅니다. 데이터 증강(data augmentation) 기술을 통해 모델 학습 과정에서 일반화 능력을 더욱 향상시키는 결과를 도출했습니다.



### A Study on Unsupervised Domain Adaptation for Semantic Segmentation in the Era of Vision-Language Models (https://arxiv.org/abs/2411.16407)
Comments:
          Accepted to British Machine Vision Conference (BMVC) 2024: Workshop on Robust Recognition in the Open World (RROW)

- **What's New**: 이번 연구에서는 가치 있는 도메인 일반화 (Domain Generalization, DG) 성능을 제공하는 비전-언어 (Vision-Language, VL) 사전 훈련된 인코더를 사용하여 기존의 비지도 도메인 적응 (Unsupervised Domain Adaptation, UDA) 방법을 향상시키는 새로운 접근법을 제시합니다. 특히, UDA 방법 중 DACS와 같은 방법에서 VL 인코더의 사용이 목표 성과와 일반화 성과 모두에서 10.0% mIoU와 13.7% mIoU의 성능 개선을 가져온다는 점이 주목할 만합니다. 이 연구는 일반적인 UDA 방법과는 다르게 VL 모델의 가능성을 강조합니다.

- **Technical Details**: 비지도 도메인 적응(UDA) 방법은 샘플이 레이블이 없는 목표 도메인에서만 이용 가능할 때 이를 목표 도메인에 맞게 조정하는데 주안점을 둡니다. 연구진은 DACS와 같은 UDA 프레임워크에 비전-언어 사전 훈련된 인코더를 추가하여 다양한 도메인에서 성능을 평가하였습니다. 또한, 주어진 데이터세트를 통해 일반화 성과와 UDA 성과가 항상 일치하지 않는 것을 발견했습니다.

- **Performance Highlights**: 실험 결과, GTA5에서 Cityscapes로의 도메인 전환에서 10.0% mIoU의 성능 향상이 발생했으며, 세 개의 보이지 않는 데이터세트에 대해서는 13.7% mIoU의 성능 향상이 나타났습니다. 이러한 연구 결과는 UDA 방법이 특정한 기상 조건 변화에서도 성능이 차별화됨을 확인시켜주었습니다. 결론적으로, 비전-언어 모델이 UDA 성과를 극대화할 수 있는 가능성을 보여주는 중요한 연구 내용이라 할 수 있습니다.



### Synthesising Handwritten Music with GANs: A Comprehensive Evaluation of CycleWGAN, ProGAN, and DCGAN (https://arxiv.org/abs/2411.16405)
Comments:
          10 pages, one page references, to appear on the IEEE Big Data 2024 2nd Workshop on AI Music Generation (AIMG 2024)

- **What's New**: 이 논문은 Optical Music Recognition (OMR) 시스템의 데이터를 보강하기 위해 Generative Adversarial Networks (GANs)를 활용하여 현실적인 손글씨 음악 악보를 생성하는 방법을 제안합니다. DCGAN, ProGAN, CycleWGAN 등 세 가지 GAN 모델을 비교하며, CycleWGAN이 스타일 전송 품질과 훈련 안정성에서 월등하게 우수한 성능을 보임을 보여줍니다. 이 접근법은 OMR 시스템의 성능 향상에 기여할 것으로 기대됩니다.

- **Technical Details**: CycleWGAN은 Wasserstein 손실 함수를 적용하여 훈련 안정성을 높이고 스타일 전송 품질을 개선합니다. 이 논문에서는 손글씨 음악 데이터의 부족 문제를 해결하기 위해 손글씨 음악 이미지를 생성하기 위한 GAN 아키텍처의 적용을 탐구하고, MUSCIMA 데이터셋에서 다양한 스타일을 활용하여 모델의 강건성을 높입니다. 세 가지 GAN 아키텍처의 성능을 비교하는 것이 이 연구의 핵심입니다.

- **Performance Highlights**: CycleWGAN은 FID 점수 41.87, IS 2.29, KID 0.05라는 뛰어난 성능을 기록하여 OMR 시스템 훈련에 적합한 고품질의 합성 데이터를 생성하는 것으로 평가됩니다. 이 연구는 다양한 손글씨 음악 샘플을 생성하여 기존의 손글씨 음악 데이터셋을 확장할 수 있는 유용한 통찰을 제공합니다. 이러한 성과는 향후 OMR 시스템 개발을 위한 실용적인 지침을 제공할 것으로 기대됩니다.



### Adapter-based Approaches to Knowledge-enhanced Language Models -- A Survey (https://arxiv.org/abs/2411.16403)
Comments:
          12 pages, 4 figures. Published at KEOD24 via SciTePress

- **What's New**: 최근 KELMs(지식 강화 언어 모델)가 대규모 언어 모델과 도메인 특화 지식 간의 격차를 메우는 유망한 도구로 주목받고 있습니다. 본 연구는 어댑터 기반 접근법에 대한 체계적인 문헌 리뷰(SLR)를 수행하며, 다양한 방법론의 강점과 잠재적 단점을 탐구합니다. 특히 생물 의학 영역에 집중하여 기존 KELMs의 성능 비교를 제공합니다.

- **Technical Details**: KELMs는 지식 그래프(KGs)를 활용하여 사실 정확도를 높이고 생성 텍스트 내 환각을 줄이는 데 기여합니다. 어댑터 모듈은 계산 부하를 줄이고 재학습 시의 삭제 위험을 최소화하는 데 효과적입니다. 본 연구에서는 KELMs에 대한 기존의 다양한 연구를 정량적, 정성적으로 분석하고, 어댑터 기반의 지식 강화를 위한 주요 동향을 정리합니다.

- **Performance Highlights**: 연구 결과, KELMs의 일반 지식과 도메인 특화 접근법이 자주 탐구되고 있으며, 다양한 어댑터 아키텍처와 하위 작업에서 긍정적인 성능을 보였습니다. 어댑터를 통한 경량화 및 효율적 해결책이 존재함에도 불구하고, KELMs에 대한 포괄적인 리뷰가 부족했음을 확인하였습니다. 본 연구는 어댑터 기반 KELM의 접근법에 대한 새로운 통찰을 제공합니다.



### Human-Calibrated Automated Testing and Validation of Generative Language Models (https://arxiv.org/abs/2411.16391)
- **What's New**: 이 논문에서는 고위험 분야, 특히 은행 산업에서 사용되는 Retrieval-Augmented Generation (RAG) 시스템을 위한 생성 언어 모델(GLM)의 평가 및 검증을 위한 포괄적인 프레임워크인 Human-Calibrated Automated Testing (HCAT)을 제안합니다. HCAT은 자동화된 테스트 생성을 통한 효율적인 검증, 임베딩 기반의 메트릭을 사용한 기능성 및 안전성 평가, 그리고 인간 판단과의 정렬을 위한 이중 단계 보정 접근 방식을 통합합니다. 이 프레임워크는 모델의 성능을 다양한 입력 조건에 대해 평가할 수 있는 강건성 테스트를 포함하여 GLM을 특정 분야에 맞게 안전하고 정확하게 배치할 수 있는 실질적이고 신뢰할 수 있는 솔루션을 제공합니다.

- **Technical Details**: HCAT 프레임워크는 자동 테스트 생성, 설명 가능한 평가 메트릭 및 인간 보정 기준을 결합하여 RAG 시스템에서 GLM을 평가하는 복잡성을 해소합니다. 자동화된 테스트 생성은 주제 모델링과 계층화 샘플링을 사용하여 모델 평가를 위한 다양한 쿼리를 생성합니다. 또한, HCAT은 임베딩 기반 메트릭을 통해 기능성 평가, 리스크 평가, 안전성 특성을 체계적으로 평가할 수 있는 방안을 제시합니다.

- **Performance Highlights**: HCAT의 성능 강조점은 제안된 접근 방식이 GLM의 작동 신뢰성을 보장하며, 사용자가 요구하는 정보에 대해 정확하고 관련성 있는 응답을 생성할 수 있도록 함을 보여줍니다. 또한, HCAT은 모델 성능에 대한 지속적인 모니터링과 개선점을 식별할 수 있도록 해주며, 특히 금융 산업에서 규제 준수를 위한 엄격한 기준을 충족하기 위한 체계적인 검증 절차를 제공합니다.



### Privacy-Preserving Federated Foundation Model for Generalist Ultrasound Artificial Intelligenc (https://arxiv.org/abs/2411.16380)
- **What's New**: 본 연구에서는 UltraFedFM이라는 혁신적인 초음파 기초 모델을 제안합니다. 이 모델은 9개 국가의 16개 의료 기관에서 분산 학습(federated learning)을 통해 협력적으로 사전 훈련되었습니다. 100만 개 이상의 초음파 영상을 기반으로 개인정보 보호를 강화하며, 다양한 임상 진단 작업에서 우수한 성능을 발휘합니다.

- **Technical Details**: UltraFedFM은 두 가지 단계로 구성됩니다: 1) 연합 사전 훈련(federated pre-training)과 2) 하위 조정(downstream fine-tuning). 고객들이 각자의 데이터를 공유하지 않고도 모델을 업데이트하며, 특정 임상 작업에 맞춰 모델을 최적화하기 위해 개인맞춤형 데이터로 미세 조정됩니다. 이러한 접근법은 데이터 프라이버시와 일반화 문제를 동시에 해결합니다.

- **Performance Highlights**: UltraFedFM은 질병 진단에서 평균 AUC(Area Under Curve) 0.927을 달성하며, 병변 분할에서는 디이스 유사도 계수(dice similarity coefficient) 0.878을 기록했습니다. 또한, 중급 초음파 전문의보다 더 높은 진단 정확도를 보이며, 경력 10년 이상의 전문가와 비슷한 성능으로 8가지 일반적인 전신 질병 진단에서 우수한 결과를 보여주고 있습니다.



### A Review of Bayesian Uncertainty Quantification in Deep Probabilistic Image Segmentation (https://arxiv.org/abs/2411.16370)
Comments:
          20 pages

- **What's New**: 본 논문은 이미지 분할 분야에서 확률적(segmentation)에 대한 포괄적인 개요를 제공합니다. 특히, 대칭적 불확실성(aleatoric uncertainty)과 인식적 불확실성(epistemic uncertainty)이라는 두 가지 주요 개념을 통해 모델의 불확실성을 수량화하는 방법을 제시합니다. 이를 통해, 이론적인 기초와 다양한 실제 응용 분야를 연관시켜 주목받고 있는 논문입니다.

- **Technical Details**: 이미지 분할은 데이터의 픽셀 단위로 분류하는 작업으로, 객체와 관심 영역을 효과적으로 구분합니다. CNN(Convolutional Neural Networks) 기반의 방법들이 급격한 발전을 이루면서, 보다 정교한 불확실성 분석이 가능해졌습니다. 이 연구에서는 이러한 불확실성을 정확히 모델링하기 위해 베이지안 추론(Bayesian inference)을 근본적으로 활용하는 방법을 탐구하며, 새로운 접근법들을 검토합니다.

- **Performance Highlights**: 이 논문은 CNN 기반의 분할 분야에 큰 영향을 미친 이전 연구들의 성과를 분석하고 현재의 모델과 방법론을 비교합니다. 특히, U-Net과 같은 엔코더-디코더 모델이 여전히 우수한 성능을 보이며, 불확실성 수량화(quantification)에 대한 연구가 위험한 실제 상황에서의 의사결정에 중요한 영향을 끼칠 수 있음을 강조합니다. 또한, 향후 연구 과제와 함께 표준화 및 벤치마킹의 필요성을 강조하여 이 분야의 진전을 위한 실질적인 기초 자료를 제공합니다.



### Graph Neural Networks-based Parameter Design towards Large-Scale Superconducting Quantum Circuits for Crosstalk Mitigation (https://arxiv.org/abs/2411.16354)
- **What's New**: 이번 연구에서는 대규모 초전도 양자 회로를 위한 파라미터 설계 알고리즘을 제안합니다. 이를 통해 기존의 Snake 알고리즘과 비교하여 높은 효율성과 효과성을 입증했으며, 특히 GNNs (Graph Neural Networks)를 활용하여 양자 컴퓨터 설계의 복잡성을 해결할 수 있는 가능성을 보여주었습니다. 이 알고리즘은 양자 교차 오차를 완화하는 작업에서 두드러진 장점을 보이며, 큰 규모의 회로에서도 빠르게 동작하는 것을 확인했습니다.

- **Technical Details**: 제안한 알고리즘은 'three-stair scaling' 메커니즘에 의존하며, 두 개의 신경망 모델로 구성됩니다. 하나는 소규모 회로에 대해 훈련된 평가기(evaluator)이고, 다른 하나는 중간 규모 회로에 입력되어 대규모 회로에 적용되는 설계자(designer)입니다. 이 알고리즘은 각각의 노드(큐빗)와 엣지(결합된 큐빗 쌍)의 파라미터를 설계하여 양자 컴퓨팅의 오류를 줄이는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, 초전도 양자 회로에서 약 870개의 큐빗으로 구성된 경우, 훈련된 설계자는 단 27초만에 주파수 설계 작업을 완료할 수 있었습니다. 반면 전통적인 Snake 알고리즘은 90분이 소요됩니다. 또한, 제안한 알고리즘에 의해 생성된 교차 오차는 Snake 알고리즘에 의해 생성된 오차의 51%에 불과함을 확인하여, GNNs의 우수한 성능과 확장성을 입증했습니다.



### The Two-Hop Curse: LLMs trained on A->B, B->C fail to learn A-->C (https://arxiv.org/abs/2411.16353)
- **What's New**: 이번 논문에서는 LLMs(대규모 언어 모델)가 두 단계 추론(two-hop reasoning)에서 겪는 문제를 새로운 설정에서 조사합니다. 연구자들은 Llama 3 8B Instruct 및 GPT-4o와 같은 모델을 허구의 사실로 미세 조정(fine-tuning)한 후, CoT(Chain-of-Thought) 방법을 이용해 질문에 답하는 일반화를 확인했습니다. 그러나 CoT 없이 서로 다른 문서에서 학습된 사실이 포함된 경우에는 두 단계 추론이 완전히 실패하는 "Two-Hop Curse"라는 개념을 제시했습니다.

- **Technical Details**: 이 연구는 두 단계 추론을 위한 통제된 설정(controlled setting)을 도입하여, 모델이 학습 시 자료를 함께 다루는 경우에는 잠재적 추론(latent reasoning)이 가능하다는 것을 발견했습니다. 하지만 두 가지 사실이 서로 다른 문서에서 학습된 경우, 모델은 기회를 넘어서는 성과를 보이지 못하고 무작위 수준의 정확도(chance-level accuracy)와 손실(test loss)를 기록했습니다. 이는 LLMs가 질문 범주와 무관하게 잠재적 다단계 추론에 대한 일반적인 능력을 결여하고 있음을 나타냅니다.

- **Performance Highlights**: 9개의 최전선 LLMs 모델을 실제 사실에 대해 평가한 결과, 대부분의 질문 범주에서 CoT를 사용하지 않는 경우 두 단계 추론에서 완전히 실패하는 경향을 발견했습니다. 반면, 대부분 범주에서 CoT를 사용할 경우에는 부분적인 성공을 유지했습니다. 이러한 결과는 LLMs가 질문 유형에 관계없이 다단계 추론에 필요한 일반적인 능력이 부족하다는 것을 시사합니다.



### Can AI grade your essays? A comparative analysis of large language models and teacher ratings in multidimensional essay scoring (https://arxiv.org/abs/2411.16337)
Comments:
          Accepted at LAK '25

- **What's New**: 이 연구에서는 독일 학생들의 에세이를 평가하기 위해 오픈소스 및 클로즈드 소스 LLM(대형 언어 모델)의 성능과 신뢰성을 평가했습니다. 37명의 교사들이 정의한 10개의 평가 기준에 따라 실제 에세이를 기준으로 실험을 진행하였으며, LLM들이 언어 관련 기준에서 특히 우수한 성능을 보였습니다. 특히, o1 모델은 모든 LLM 중에서 가장 높은 성과를 기록하였으며, 이는 교사의 평가와의 상관관계에서 $r = .74$, 내부 신뢰성에서는 $ICC=.80$이라는 수치를 보여 줍니다.

- **Technical Details**: 연구에서는 7학년과 8학년 학생들의 20개 실제 에세이를 대상으로 GPT-3.5, GPT-4, o1, LLaMA 3-70B, Mixtral 8x7B의 5개의 LLM을 분석하였습니다. 각 모델의 평가 결과를 37명의 교사들의 평가와 비교하여, 이들의 강점과 한계를 여러 평가 범주에서 철저히 분석하였습니다. 특히, 평가 범주 간 상관 관계를 조사하여 LLM의 추론 과정을 이해하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 결과적으로, 클로즈드 소스 GPT 모델들은 오픈소스 모델들보다 높은 내부 일관성과 교사 평가와의 일치를 보였으며, 언어 관련 기준에서 특히 두드러진 성과를 나타냈습니다. LLM 기반의 평가가 교사의 작업 부담을 줄일 수 있는 유용한 도구가 될 수 있다는 점을 보여 주었으나, 모델들이 점수가 높게 나오는 경향이 있어 콘텐츠 품질을 더 잘 반영하기 위한 추가적인 개선이 필요함을 시사합니다.



### Brain-like emergent properties in deep networks: impact of network architecture, datasets and training (https://arxiv.org/abs/2411.16326)
- **What's New**: 이 논문은 최신 심층 신경망(DNNs)이 현실 세계의 시각적 작업에서 인간을 따라잡지 못하는 역설적인 문제에 초점을 맞추고 있습니다. 여기서는 심층 신경망이 인간의 시각 시스템과 얼마나 유사한지를 평가하기 위해 30개 이상의 최신 네트워크 아키텍처, 훈련 데이터셋 및 교육 방식을 체계적으로 평가한 결과를 보고합니다. 이러한 평가를 통해 심층 신경망의 설계 요인이 두뇌와의 유사성에 미치는 영향을 이해하는 데 기여하고자 합니다.

- **Technical Details**: 연구에서는 15가지의 시각 심리학 및 신경과학에서 잘 알려진 인지적, 신경적 특성을 선택하여 DNNs의 두뇌 유사성을 평가하였습니다. 각 네트워크에 대해 특정 속성을 테스트하여 상관 관계를 평가한 후, 'Brain Property Match (BPM)'라는 종합 메트릭을 제안하였습니다. 15가지 특성 각각에 대한 효과 강도는 -1에서 1까지의 범위를 가지며, 양수는 해당 특성이 존재함을 의미합니다.

- **Performance Highlights**: 연구 결과, 네트워크 아키텍처가 인간의 시각적 감지 특성과 가장 높은 관계를 보였고, 특정 네트워크가 모든 네트워크에 비해 일관되게 우수한 성능을 보이는 것은 아니었습니다. 뿐만 아니라, 최신 DNNs에서 발생하거나 결여된 두뇌 유사성 속성을 드러내어 기존 벤치마크와 보완적 역할을 수행합니다. 이러한 발견은 새로운 DNN 개발에 있어 두뇌 유사성을 증진시키기 위한 방향성을 제공할 것으로 기대됩니다.



### One Diffusion to Generate Them A (https://arxiv.org/abs/2411.16318)
Comments:
          two first authors contribute equally

- **What's New**: OneDiffusion은 다양한 작업에 대해 양방향 이미지 합성과 이해를 지원하는 다목적 대규모 확산 모델입니다. 이 모델은 텍스트, 깊이, 자세, 레이아웃 및 의미론적 맵과 같은 입력으로부터 조건부 생성이 가능하며, 이미지 디블러링, 업스케일링 및 밀어내기 작업과 같은 다양한 작업을 처리할 수 있습니다. OneDiffusion은 또한 다중 뷰 생성, 카메라 자세 추정 및 순차 이미지 입력을 통한 즉각적인 개인화 기능을 제공합니다.

- **Technical Details**: OneDiffusion은 모든 작업을 훈련 중에 다양한 잡음 수준의 프레임 시퀀스로 처리하며, 각 프레임은 추론 시 조건부 이미지로 작용할 수 있도록 설계되었습니다. 모델을 훈련시키기 위해, 고품질 데이터와 기존 모델의 합성 출력을 포함한 One-Gen 데이터셋을 구축했습니다. 이 데이터셋은 여러 작업의 공동 훈련을 지원하며, 유연한 조건부 옵션과 개선된 일반화를 제공합니다.

- **Performance Highlights**: OneDiffusion은 텍스트-이미지(텍스트 입력으로 이미지를 생성) 및 다중 뷰 생성 작업에서 경쟁력 있는 성능을 보여주었으며, 기존의 특정 훈련 모델과 비슷한 수준의 성능을 발휘했습니다. 또한, 깊이 추정 작업에서도 우수한 성능을 발휘했으며, 새로운 조건부 설정인 텍스트-다중 뷰 및 이미지-다중 뷰 생성을 지원합니다. 이 모델은 복잡한 과제에서도 일반화가 뛰어나며, 다양한 표현과 자세를 가진 여러 이미지를 생성할 수 있습니다.



### Learning from Relevant Subgoals in Successful Dialogs using Iterative Training for Task-oriented Dialog Systems (https://arxiv.org/abs/2411.16305)
- **What's New**: 본 연구에서는 Task-oriented Dialog (ToD) 시스템의 성능을 향상시키기 위한 새로운 접근 방식인 SUIT(SUbggoal-aware ITerative Training)를 제안한다. SUIT는 모델에서 샘플링된 대화를 사용하여 대화 성공에 기여하는 서브목표(subgoal)를 식별하고, 이를 기반으로 고품질의 훈련 샘플을 생성한다. 이 방법은 기존의 고정된 데이터 세트에 의존하기보다 반복적으로 더 많은 데이터를 생성할 수 있는 능력을 갖추고 있다.

- **Technical Details**: SUIT는 초기 LLM(대형 언어 모델)을 사용하여 사용자의 목표에 대한 대화를 생성하고, 성공적인 대화에 대한 서브목표를 식별하기 위해 'distant supervision'(원거리 감시) 방식을 적용한다. 이렇게 결정된 서브목표를 통해 추가적인 훈련 샘플을 구축하고, 이를 통해 ToD 시스템의 supervised fine-tuning(SFT) 또는 preference learning을 개선한다. 이 반복적인 과정은 오류를 최소화하고 대화 관련 모델 성능을 최대화하는 데 중점을 둔다.

- **Performance Highlights**: SUIT의 접근 방식은 대화 품질을 크게 향상시키며, 인기 있는 ToD 벤치마크에서 새로운 최첨단 성능(state-of-the-art performance)을 기록하였다. 기존 E2E(End-to-End) 시스템과는 달리 SUIT는 모델 맞춤화 없이 대규모 응용 프로그램에서도 쉽게 설정하고 사용할 수 있다. 특히, 이 방법은 DPO(Direct Preference Optimization)를 기반으로 한 선호 학습을 적용하여 효율성과 안정성을 높인다.



### BayLing 2: A Multilingual Large Language Model with Efficient Language Alignmen (https://arxiv.org/abs/2411.16300)
Comments:
          BayLing 2's online demo: this http URL. BayLing 2's code and models: this https URL

- **What's New**: BayLing 2는 고자원의 언어에서 저자원의 언어로의 생성 능력과 지식을 효율적으로 전이하는 혁신적인 다국어 대형 언어 모델(LLM)입니다. 320만 개의 지시사항 데이터셋을 사용하여 100개 이상의 언어에 대한 다국어 이해 능력을 향상시키고, 고자원 언어에서 저자원 언어로의 지식 전이 처치를 가능하게 합니다. BayLing은 다국어 번역과 지식 전이 벤치마크에서 기존 오픈 소스 모델보다 월등한 성능을 보였습니다.

- **Technical Details**: BayLing 2는 Llama 모델을 기반으로 하여 설계되었으며, 이 모델은 고자원 언어인 중국어와 영어로 이루어진 지시사항으로 훈련되었습니다. 이를 통해 100개 이상의 언어에 대한 교차 언어 지시사항을 포함하여 보다 효율적인 언어 정합을 수행합니다. BayLing은 두 가지 주요 모델인 BayLing-2-7B와 BayLing-2-13B를 포함하며, 다양한 언어 간의 능력 전이를 위한 체계적인 평가가 이루어졌습니다.

- **Performance Highlights**: BayLing의 성능 평가는 100개 이상의 언어에서 우수한 번역 결과를 나타냈으며, 특히 20개 이상의 저자원 언어에서 지식 전이의 효과가 두드러졌습니다. 예를 들어, 바밤바(Bambara), 루간다(Luganda), 스와힐리(Swahili), 줄루(Zulu) 등의 언어에서 뚜렷한 성능 향상이 이루어졌습니다. 또한, BayLing은 고자원 언어에서의 성능을 유지하면서 저자원 언어에서의 응답 능력을 효율적으로 개선하였습니다.



### The SVASR System for Text-dependent Speaker Verification (TdSV) AAIC Challenge 2024 (https://arxiv.org/abs/2411.16276)
- **What's New**: 이 논문은 고성능 생체 인식 시스템의 필요성을 해결하기 위한 효율적이고 정밀한 텍스트 의존 스피커 검증(TDSV) 파이프라인을 소개합니다. 제안된 시스템은 Fast-Conformer 기반의 ASR 모듈을 통해 음성 콘텐츠를 검증하며, Target-Wrong (TW) 및 Impostor-Wrong (IW) 시도를 필터링합니다. 스피커 검증을 위해 wav2vec-BERT와 ReDimNet 모델에서 추출된 스피커 임베딩을 결합한 특징 융합 접근법을 제안하며, 이를 통해 TDSV 2024 Challenge 테스트 세트에서 경쟁력 있는 성과를 달성하였습니다.

- **Technical Details**: TDSV에서 스피커의 신원을 확인하려면 특정 구문을 말해야 하며, 이를 위해 Acoustic 및 Linguistic 속성을 활용합니다. 최근에는 멀티태스크 학습이 일반적인 접근 방식이 되어, 스피커와 음소 정보의 조합을 활용해 성능을 향상시키고 있습니다. 본 논문에서 제안된 모델은 ResNet34, XEUS 및 Hybrid network with attention과 같은 여러 아키텍처를 사용하여 음성 및 스피커 콘텐츠를 독립적으로 모델링합니다.

- **Performance Highlights**: 제안된 시스템은 TDSV 2024 Challenge에서 0.0452의 정규화된 최소 검출 비용 함수(min-DCF) 성적을 기록하며 2위에 올랐습니다. 이는 시스템이 정확성과 강건성을 적절히 조화롭게 유지할 수 있음을 나타냅니다. 또한, 멀티태스크 학습을 통해 스피커와 구문의 정보를 효과적으로 활용하여 성능을 극대화했습니다.



### Diagnosis of diabetic retinopathy using machine learning & deep learning techniqu (https://arxiv.org/abs/2411.16250)
Comments:
          9 pages, 11 figures, Journal Paper

- **What's New**: 이번 연구에서는 안저 이미지(fundus images) 분석을 위한 새로운 방법을 제안합니다. 기존의 수동 분석 방식의 단점을 보완하기 위해 객체 탐지(object detection) 및 기계 학습(classification) 기법을 조합한 접근 방식을 사용합니다. 이를 통해 당뇨병성 망막병증(diabetic retinopathy) 등 다양한 안과 질환을 진단하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 우리는 YOLO_V8을 이용하여 안저 이미지에서 객체를 탐지하고, 시신경 디스크(optic disc), 시신경 컵(optic cup), 병변(lesions) 등의 관심 영역(ROIs)을 식별합니다. 이후, SVM(support vector machine) 분류 알고리즘을 통해 ROIs를 병리적 징후의 존재 여부에 따라 다른 DR 단계로 분류합니다. 이 방법은 정확도 84%를 기록하였으며, 신뢰할 수 있는 진단 도구입니다.

- **Performance Highlights**: 제안된 방법은 원거리 지역에서도 안저 질환의 선별(triage)에 유용하게 적용될 수 있습니다. 높은 정확도와 효율성을 바탕으로 시각 장애 예방에 기여할 수 있는 혁신적인 연구입니다. 또한, 이 접근 방식은 안과 진단의 자동화를 통해 의료 종사자들의 부담을 줄여줄 것으로 기대됩니다.



### Batch Bayesian Optimization via Expected Subspace Improvemen (https://arxiv.org/abs/2411.16206)
- **What's New**: 이 논문은 Bayesian Optimization을 배치 평가로 확장하는 새로운 접근 방식을 제안합니다. 기존의 배치 방법들은 인공 함수를 사용하여 점들을 선택하는데, 이는 배치 크기가 증가할수록 오류가 빠르게 누적되어 최적화 효율성이 저하됩니다. 새로운 방법은 원 문제의 하위 공간을 도출하고 각 하위 공간에서 하나의 획득 점을 선택하는 것입니다.

- **Technical Details**: 제안하는 방법은 Expected Subspace Improvement (ESSI) 기준에 기반하여 원래 설계 공간의 여러 서브 스페이스에서 샘플을 찾습니다. 이 기준은 특정 서브스페이스 내에서 후보 점이 달성할 수 있는 개선의 양을 측정합니다. 이렇게 동시에 최적화된 예상 서브 스페이스 개선 기능을 통해 고가의 평가를 위한 쿼리 포인트의 배치를 얻을 수 있습니다.

- **Performance Highlights**: 수치 실험 결과, 제안한 방법은 순차적인 Bayesian Optimization 알고리즘에 비해 거의 선형적인 속도 향상을 달성하며, 최신 배치 알고리즘 여덟 개와 비교했을 때 매우 경쟁력 있는 성능을 보입니다. 이러한 결과는 제안된 방법이 배치 Bayesian Optimization의 효율성을 극대화할 수 있는 간단하면서도 효율적인 접근 방식을 제공함을 보여줍니다.



### SALOVA: Segment-Augmented Long Video Assistant for Targeted Retrieval and Routing in Long-Form Video Analysis (https://arxiv.org/abs/2411.16173)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 SALOVA(세그먼트-증강 긴 비디오 어시스턴트)라는 새로운 비디오-LLM 프레임워크를 소개합니다. SALOVA는 긴 비디오 콘텐츠의 이해를 개선하기 위해 특정 세그먼트를 효율적으로 검색하는 방법을 채택하고 있습니다. 특히, 비디오 콘텐츠의 단기적이며 중요한 정보를 보존하여, 모델의 응답의 맥락 관련성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: SALOVA는 SceneWalk 데이터셋을 기반으로 하며, 이는 87.8K개의 긴 비디오를 포함하고 있으며 각 비디오 세그먼트에 대한 밀접한 주석을 제공합니다. 또한, 동적 라우팅 메커니즘 및 시공간 프로젝터(spatio-temporal projector)를 활용하여 사용자 쿼리에 따라 관련 비디오 세그먼트를 적절하게 검색하고 처리하는 아키텍처를 개발하였습니다. 이러한 설계는 긴 비디오를 처리하는 데 있어 효율성을 높이고, 문맥 통일성을 유지하는 데 기여합니다.

- **Performance Highlights**: 광범위한 실험을 통해 SALOVA는 복잡한 긴 비디오의 이해에 있어 기존 비디오-LLM 모델에 비해 향상된 성능을 보여주었습니다. 모델은 중요 시각 정보를 손실할 위험을 줄이며 중요한 이벤트를 놓치는 사례를 줄이는데 효과적입니다. 이로써 SALOVA는 긴 비디오 콘텐츠를 보다 최적화된 방식으로 해석할 수 있는 능력을 입증하였습니다.



### Local and Global Feature Attention Fusion Network for Face Recognition (https://arxiv.org/abs/2411.16169)
- **What's New**: 이번 연구는 저품질 얼굴 이미지 인식에서의 새로운 접근 방식인 Local and Global Feature Attention Fusion (LGAF) 네트워크를 제안합니다. LGAF 네트워크는 로컬(local)과 글로벌(global) 얼굴 특징의 주의를 효과적으로 조정하여 서로 보완적인 정보를 활용합니다. 이 네트워크는 저품질 이미지에서 발생할 수 있는 다양한 피쳐 품질 편향을 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: LGAF 네트워크는 두 가지 핵심 모듈, 즉 Local and Global Feature Fusion (LGF)와 Multi-Head Multi-Scale Local Feature Extraction (MHMS) 모듈로 구성됩니다. LGF 모듈은 피쳐 품질을 기반으로 로컬 및 글로벌 피쳐 간의 주의를 동적으로 측정하고, MHMS 모듈은 다양한 스케일과 채널 차원에서 얼굴 특징의 중요성을 평가하여 보다 풍부한 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, LGAF 네트워크는 CFP-FP, CPLFW, AgeDB, CALFW 등의 검증 세트에서 최고 평균 성능을 기록했습니다. 또한 TinyFace와 SCFace 데이터셋에서 최신 기술(SoTA)과 비교하여 뛰어난 성능을 보였습니다. 이러한 결과는 LGAF 네트워크가 저해상도와 고해상도 얼굴 인식 모두에서 효과적임을 입증합니다.



### MixPE: Quantization and Hardware Co-design for Efficient LLM Inferenc (https://arxiv.org/abs/2411.16158)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 혼합 정밀도 처리를 위한 MixPE라는 새로운 하드웨어 가속기를 소개합니다. MixPE는 낮은 비트 정밀도를 효과적으로 다루면서 탈정규화(overhead)를 최소화할 수 있도록 설계되었습니다. 현재의 하드웨어는 혼합 정밀도 행렬 곱셈(mixed-precision GEMM)에 대한 원활한 지원을 제공하지 않기 때문에, 이러한 성능 저하 문제를 해결하기 위해 새로운 접근법이 필요합니다.

- **Technical Details**: MixPE는 두 가지 주요 혁신을 통해 저비트 정량화의 잠재력을 최대한 활용합니다. 첫째, 각 정량화 그룹 내에서 스케일(scale)과 제로 포인트(zero point)를 공유함으로써 후처리에서 탈정규화 작업을 수행하여 효율성을 높입니다. 둘째, 전통적인 곱셈기의 대신에 효율적인 shift&add 연산을 활용하여 낮은 비트 무게와 높은 비트 활성화 간의 곱셈을 최적화합니다.

- **Performance Highlights**: MixPE는 실험을 통해 기존의 정량화 가속기보다 2.6배 빠르며, 에너지를 1.4배 절약할 수 있다는 결과를 보여주었습니다. 이는 X-bit weight와 Y-bit activation 조합에서 특히 두드러진 성능 향상을 나타냅니다. MixPE는 또한 하드웨어 효율성과 수치 정확도 사이의 최적 조합을 제공하며, LLM 추론 시 성능을 극대화할 수 있는 전략을 제시합니다.



### Graph Adapter of EEG Foundation Models for Parameter Efficient Fine Tuning (https://arxiv.org/abs/2411.16155)
Comments:
          Under review

- **What's New**: 이 연구에서는 EEG 데이터를 분석하기 위한 새로운 방법인 EEG-GraphAdapter (EGA)를 제안합니다. EGA는 Temporal backbone 모델에 통합되어 GNN 기반 모듈로서 기능하며, 고정된 두뇌 모델의 매개변수를 유지한 채로 독립적으로 미세 조정됩니다. 이를 통해 EEG 신호의 공간적 표현을 효율적으로 학습할 수 있으며, 데이터 요구사항과 계산 부담을 크게 줄여줍니다.

- **Technical Details**: EEG-GraphAdapter(EGA)는 다변량 EEG 신호의 입력에 공간적 표현을 통합하여 사전 훈련된 모델의 시간적 표현을 보완합니다. BENDR 모델의 매개변수는 미세 조정 중에 고정되어, 공간적 특성을 학습하는 데 중점을 두고, GNN을 사용하여 EEG 센서 간의 관계를 표현하는 가중 그래프를 정의합니다. EGA는 주어진 EEG 신호 길이에 따라 고정된 길이의 특징 임베딩을 요구하며, 2계층 GNN 모델로 구성되어 공간적 관계를 처리합니다.

- **Performance Highlights**: 본 논문의 실험은 주요 우울 장애(MDD) 및 이상 감지와 같은 헬스케어 관련 다운스트림 작업에서 수행되었습니다. EGA는 BENDR 모델에 비해 F1 점수에서 최대 16.1%의 성능 향상을 보여주었습니다. 이러한 결과는 EGA가 EEG 신호의 공간적 및 시간적 특성을 효과적으로 결합함으로써 가능해진 것입니다.



### SKQVC: One-Shot Voice Conversion by K-Means Quantization with Self-Supervised Speech Representations (https://arxiv.org/abs/2411.16147)
Comments:
          5 pages

- **What's New**: 이번 연구에서는 기존의 복잡한 구조와 사전 훈련된 화자 검증 모델에 의존하지 않는 간단하면서도 효과적인 one-shot 음성 변환 모델을 제안했습니다. 이 모델은 SSL(자기 지도 학습) 특징과 음성 속성을 활용하여 음성 변환의 정확도를 높입니다. 특히, 외부 화자 임베딩 없이도 고충실도의 음성 변환이 가능하도록 하는 기법을 도입하였으며, 발화 변동성을 보존하는 방법이 강조되었습니다.

- **Technical Details**: 제안한 모델 아키텍처는 WavLM 인코더, K-means 양자화, Disentangler 및 HiFi-GAN 디코더로 구성되어 있습니다. WavLM의 6번째 레이어에서 추출된 SSL 특징을 사용하여 음성 정보를 포착합니다. 양자화 과정에서 코드북은 SSL 특징에 K-means 클러스터링을 적용하여 초기화되며, 이는 주로 음성의 발음 특징을 나타냅니다.

- **Performance Highlights**: 모델은 6가지 평가 지표에서 우수한 성능을 보여주었고, 발화 변동성을 보상하는 방법의 이점이 부각되었습니다. 다양한 코드북 크기에서 그 효과가 나타났으며, 주요 구성 요소의 역할을 확인하기 위한 제거 연구(ablation study)도 수행했습니다. 최종적으로, 코드북 크기가 256인 경우에도 발화의 억양과 내용의 정확성을 유지하는 능력을 보여주었습니다.



### CIA: Controllable Image Augmentation Framework Based on Stable Diffusion (https://arxiv.org/abs/2411.16128)
- **What's New**: 이번 연구에서는 CIA라는 모듈형 데이터 증강 파이프라인을 제안합니다. 이 파이프라인은 Stable Diffusion을 이용해서 합성 이미지를 생성하고, 정의된 품질 메트릭스를 사용하여 저품질 샘플을 필터링합니다. 또한, ControlNet을 통해 생성된 이미지에서 특정 패턴의 존재를 강제할 수 있도록 하였습니다. CIA는 안정된 데이터셋 품질을 보장하면서 더욱 효과적인 모델 학습을 지원할 수 있는 기회를 제공합니다.

- **Technical Details**: CIA는 네 개의 모듈로 구성되며, 첫 번째 모듈인 Extraction은 원본 이미지에서 특성 추출을 수행합니다. 이후 ControlNet은 추출된 특성을 활용하여 Stable Diffusion의 출력을 조건화하여 정밀한 제어를 할 수 있습니다. Generation 모듈은 텍스트 프롬프트와 결합된 추출된 특성을 기반으로 새로운 이미지를 합성하며, Quality Assessment 모듈은 선택한 품질 메트릭스를 통해 생성된 이미지를 필터링합니다.

- **Performance Highlights**: CIA를 활용한 인간 객체 탐지 사례 연구에서는 데이터량이 부족한 상황에서도 유의미한 성능 향상을 기록했습니다. 특히, CIA로 생성된 이미지를 활용함으로써 실제 이미지의 양을 두 배로 늘린 경우와 유사한 성능에 도달했습니다. 이는 데이터 제약 환경에서도 효과적인 객체 탐지 시스템의 구축이 가능함을 시사합니다.



### Med-PerSAM: One-Shot Visual Prompt Tuning for Personalized Segment Anything Model in Medical Domain (https://arxiv.org/abs/2411.16123)
- **What's New**: 이 논문에서는 Med-PerSAM이라는 새로운 one-shot 프레임워크를 소개합니다. 기본적으로 이 프레임워크는 의료 도메인에서 SAM의 성능을 향상시키기 위해 설계되었습니다. Med-PerSAM은 시각적 프롬프트 엔지니어링에 독점하여 추가 학습이나 사람의 개입 없이 자동화된 프롬프트 생성 프로세스를 활용합니다.

- **Technical Details**: Med-PerSAM의 기본적인 이미지 왜곡 손실로는 SSIM(Structural Similarity Index Measure) 손실이 사용되었습니다. 그러나 OdontoAI 및 CAMUS 데이터셋의 경우, NCC(Normalized Cross-Correlation) 손실이 사용되었으며, 이 손실은 의료 이미지 등록에서 일반적으로 사용하는 함수입니다. 프레임워크의 정규화항은 흐름 필드의 부드러움을 보장하여 급격한 왜곡을 방지합니다.

- **Performance Highlights**: Med-PerSAM은 다양한 2D 의료 이미징 데이터셋에서 기존 SAM 기반 접근법 및 기타 기초 모델보다 우수한 성능을 보였습니다. 특히, 훈련 및 재훈련 과정에서 비교적 짧은 시간 내에 성과를 달성하여 효율성을 강조합니다. 이 연구는 깊은 의료 전문 지식이 없는 사람에게도 유용한 도구가 될 것으로 기대됩니다.



### LLM Augmentations to support Analytical Reasoning over Multiple Documents (https://arxiv.org/abs/2411.16116)
Comments:
          2024 IEEE International Conference on Big Data (IEEE BigData 2024)

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 능력을 통해 정보 분석의 깊이 있는 추론을 향상시키는 방안을 모색합니다. 정보 분석가는 방대한 보고서를 분석하여 서로 관련이 없는 정보 간의 연결을 찾고, 적의 계획과 동기를 파악합니다. LLMs의 사용을 통하여 이러한 작업이 얼마나 개선될 수 있는지, 그리고 어떤 방식으로 도움을 줄 수 있는지를 탐구합니다.

- **Technical Details**: 연구에서는 다이나믹 증거 트리(Dynamic Evidence Trees, DETs)라는 메모리 모듈을 통해 LLM의 기능을 확장하는 아키텍처를 개발하였습니다. 이 시스템은 여러 조사 분석 루트를 개발하고 추적하는 데 도움이 되며, LLMs의 잠재력을 극대화하기 위한 세 가지 단계 보강과정을 제안합니다. 또한, LLMs가 정보 수집, 가치 있는 출처의 조직화 및 설득력 있는 주장의 생성을 지원하는 방법을 실험합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, LLMs는 정보 분석에 있어 여전히 부족한 부분이 있으며, 특히 설득력 있는 주장을 구성하는 창의적인 분석 능력이 떨어진다고 보고되었습니다. 이 연구는 LLMs를 정보 분석 태스크에 적용하기 위한 구체적인 개선 방안을 제시하고, 구체적인 하위 작업인 증거 조정 및 내러티브 생성을 지원하는 방법을 강조합니다. DKTs를 사용하여 LLM의 성능을 보강할 이후 전망을 제시하였습니다.



### LLMPirate: LLMs for Black-box Hardware IP Piracy (https://arxiv.org/abs/2411.16111)
Comments:
          Accepted by NDSS Symposium 2025

- **What's New**: 이 논문에서는 하드웨어 설계 및 검증 과정에서 발생할 수 있는 새로운 공격 시나리오인 지적 재산권(IP) 해적행위에 대해 다루고 있습니다. LLMPirate라는 LLM(large language model) 기반 기술을 제안하며, 이는 다양한 회로 설계의 변형을 생성하여 기존의 해적 감지 도구를 회피할 수 있는 능력을 갖추고 있습니다. 이 연구는 LLM의 통합과 대규모 회로의 확장성, 그리고 효과성 관련 문제를 해결하기 위한 세 가지 솔루션을 제공합니다.

- **Technical Details**: LLMPirate는 여러 최신 해적 감지 도구에서 탐지를 회피할 수 있는 회로 설계의 변형 생성에 초점을 맞춘 첫 번째 LLM 기반 기법입니다. 연구팀은 8개의 다양한 크기와 능력을 가진 LLM을 사용하여 LLMPirate의 성능을 평가했으며, 이 모든 도구에서 테스트된 회로의 100%를 성공적으로 탐지 회피할 수 있음을 보여줍니다. 또한, IBEX 및 MOR1KX 프로세서, GPS 모듈을 대상으로 한 사례 연구를 통해 LLMPirate의 실질적인 영향을 시연합니다.

- **Performance Highlights**: LLMPirate의 실험 평가 결과, 모든 테스트된 회로에 대해 100% 탐지 회피율을 달성하였습니다. 이 기법은 자동화되고 효율적인 방식으로 하드웨어 회로 설계의 해적행위를 가능하게 하여, 기존 해적 감지 도구의 한계를 드러냅니다. 연구진은 이 논문을 통해 더 나은 IP 해적 감지 도구의 개발을 촉진할 것으로 기대하고 있습니다.



### Adaptive Circuit Behavior and Generalization in Mechanistic Interpretability (https://arxiv.org/abs/2411.16105)
Comments:
          10 pages, 8 figures

- **What's New**: 이 논문은 Mechanistic interpretability의 새로운 관점을 제시하며, GPT-2 small에서 간접 목적어 식별(I/OI) 회로의 일반성을 조사합니다. 저자들은 다양한 프롬프트 변형에서도 동일한 회로가 작동하는지를 확인하며, 이 회로가 기존 알고리즘의 전제 조건을 어떻게 극복하는지를 탐구합니다. 특히, S2 Hacking이라는 새로운 메커니즘을 발견하여 기존 알고리즘의 실패에도 불구하고 회로가 잘 작동하는 원인을 설명합니다.

- **Technical Details**: 간접 목적어 식별(I/OI) 알고리즘은 이전 이름을 식별하고 중복된 이름을 제거하여 최종적으로 남은 이름을 출력하는 세 단계의 프로세스를 포함합니다. 이 알고리즘은 특정 프롬프트 구조에 거의 무관하다고 여겨지지만, 중복된 이름이 사용되는 프롬프트에서는 원래 알고리즘이 실패해야 합니다. 이에 반해, 저자들은 DoubleIO 및 TripleIO와 같은 새로운 프롬프트 변형에서도 이 회로가 여전히 유효하다는 것을 발견하고, IOI 회로의 다양한 기능을 이해하는 데 중요한 통찰을 제공합니다.

- **Performance Highlights**: IOI 회로는 IOI 알고리즘이 완전히 실패할 것으로 예상되는 프롬프트 변형에서 전체 모델보다 우수한 성능을 발휘했습니다. 모든 구성 요소를 재사용하면서 추가 입력 엣지만 추가하는 방식으로 회로가 잘 일반화된다는 것을 발견했습니다. 제시된 결과는 대규모 신경망의 일반적인 능력을 이해하는 데 중요한 진전을 나타내며, Mechanistic interpretability의 가능성을 더욱 부각시킵니다.



### An Empirical Study of Vulnerability Detection using Federated Learning (https://arxiv.org/abs/2411.16099)
- **What's New**: 이 논문은 Federated Learning (FL)을 기반으로 하는 취약점 탐지(DLL) 방법이 기존의 데이터 고립 문제(data silo problem)를 해결하는 데 효과적이라는 점을 강조합니다. 특히 VulFL이라는 새로운 평가 프레임워크를 제안하며, FL의 성능을 다양한 Common Vulnerabilities and Exposures (CWEs)와 데이터 이질성 시나리오에서 연구합니다. 기존 FL 기반 탐지 방법들이 특정 응용에 국한되어 있는 반면, 공통 취약점 탐지 작업에 대한 FL의 적응성을 명확히 하고자 합니다.

- **Technical Details**: 이 논문에서는 VulFed라는 통합 평가 프레임워크를 사용하여 FL 기반 취약점 탐지의 성능을 평가합니다. 이 프레임워크는 데이터 전처리(pre-processor), 모델 훈련(trainer), 모델 집계(aggregator), 클라이언트 선택(client selector)으로 구성된 네 가지 주요 구성 요소를 포함합니다. FedAvg라는 FL 방법의 성능을 일반적인 실제 데이터셋인 DiverseVul을 바탕으로 비교하며, 인디펜던트 트레이닝(independent training)과 FL 기반 방법의 차별성을 분석합니다.

- **Performance Highlights**: 실험 결과, FL은 일반 인디펜던트 트레이닝에 비해 CWEs에 대한 탐지 성능을 향상시킬 수 있음을 보여줍니다. 그러나 데이터 이질성(data heterogeneity)은 FL 성능에 상당한 영향을 미치는 중요한 요소로 나타났으며, CWEs의 강건성은 서로 다른 이질성 수준에 따라 다르게 영향을 받았습니다. 이 연구는 FL 기반 취약점 탐지를 위한 최적의 구성 전략을 실험적으로 제시하여 관련 설계자들에게 유용한 통찰을 제공합니다.



### ENCLIP: Ensembling and Clustering-Based Contrastive Language-Image Pretraining for Fashion Multimodal Search with Limited Data and Low-Quality Images (https://arxiv.org/abs/2411.16096)
- **What's New**: 이 논문은 패션 인텔리전스를 위한 Contrastive Language-Image Pretraining (CLIP) 모델의 성능을 향상시키기 위한 새로운 접근 방식, 즉 ENCLIP을 제시합니다. ENCLIP은 데이터 부족과 저화질 이미지 문제를 해결하며, 여러 CLIP 모델을 학습시키고 집계하는 알고리즘을 포함합니다. 이 방법은 데이터 부족과 이미지 품질 문제를 고려하여 패션 도메인에 특화된 멀티모달 검색 능력을 극대화합니다.

- **Technical Details**: ENCLIP 방법론은 여러 CLIP 모델의 출력을 집계하고 클러스터링 기법을 활용하여 이미지를 유사한 그룹으로 묶습니다. 데이터셋으로는 Kaggle의 'Fashion Product Images (Small)'를 활용하며, 이는 인도 패션 제품을 포함한 다양한 패션 아이템의 이미지와 텍스트 설명으로 구성되어 있습니다. 전처리된 이미지와 텍스트는 CLIP의 입력으로 준비되며, 벡터 데이터베이스는 이미지 저장과 검색을 돕기 위해 사용됩니다.

- **Performance Highlights**: 실험 결과는 ENCLIP의 효과성을 입증하며, 패션 분야에서의 CLIP의 잠재력을 열어줍니다. 이 방법은 데이터 부족과 저화질 이미지가 만연한 패션 인텔리전스 분야에서 실용적인 해결책을 제공합니다. 전반적으로 ENCLIP 접근 방식은 패션 인텔리전스 분야에 커다란 공헌을 하며, 효율성을 높이는 실질적인 솔루션을 제공합니다.



### HiDP: Hierarchical DNN Partitioning for Distributed Inference on Heterogeneous Edge Platforms (https://arxiv.org/abs/2411.16086)
Comments:
          7 pages, 8 figures, 1 table, and 1 algorithm. The manuscript is accepted to be published in 28th Design, Automation and Test in Europe Conference (IEEE DATE, 2025)

- **What's New**: 본 논문에서는 다양한 edge 노드의 heterogeneity를 고려한 새로운 DNN partitioning 전략인 HiDP(계층적 DNN 분할 전략)를 제안합니다. HiDP는 DNN 작업의 분배를 글로벌 및 로컬 수준 모두에서 계층적으로 분할하여 보다 낮은 지연시간을 달성합니다. 기존의 분산 추론 기법은 edge 노드의 core-level heterogeneity를 고려하지 않아 성능 저하를 초래했으나, HiDP는 이러한 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: HiDP 전략은 DNN 작업 부하를 코어 수준 및 장치 수준의 이질성을 모두 고려하여 조합적으로 결정합니다. 이 방식에서는 DNN 모델의 레이어를 동적으로 그룹화하여 실행 가능한 블록으로 나누고 이러한 블록을 서로 다른 edge 노드에 분배합니다. 또한, 로컬 DNN 파티셔너를 통해 각 장치에서 블록을 추가로 나누고 할당하여 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안한 HiDP 전략은 상용 edge 장치에서 수행한 기존의 분산 추론 기법과 비교해 평균적으로 38% 낮은 지연시간과 46% 낮은 에너지를 소모하며, 56% 높은 처리량을 기록했습니다. 이러한 성과는 다양한 DNN 모델에 대해 이루어진 실험을 통해 검증되었습니다. 특히, 작업 부하의 분할과 스케줄링을 최적화하여 edge 장치에서의 추론 성능을 크게 향상시켰음을 보여줍니다.



### Cautious Optimizers: Improving Training with One Line of Cod (https://arxiv.org/abs/2411.16085)
- **What's New**: 본 논문에서는 transformer 사전학습을 위한 기본 최적화 알고리즘인 AdamW에 대한 새로운 접근 방식을 제안합니다. 기존의 여러 최적화 알고리즘보다 더 빠르고 안정적인 성능을 추구하는 과정에서 기존의 모멘텀 기반 최적화 알고리즘에 단 하나의 간단한 수정만으로 새로운 'Cautious Optimizer'를 만들어냈습니다. 이 수정된 최적화 알고리즘은 C-AdamW 및 C-Lion과 같은 형태로 나타납니다.

- **Technical Details**: 이론적인 결과에 따르면, 제안된 수정은 Adam의 해밀토니안 함수(Hamiltonian function)를 보존하며, Lyapunov 분석 하에서도 수렴 보장을 깨뜨리지 않습니다. 이러한 이론적 통찰을 바탕으로 새로운 최적화 알고리즘군을 밝혀내었으며, 이 중 가장 단순한 형태의 알고리즘을 선택하여 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, Llama와 MAE 사전 학습에서 최대 1.47배 더 빠른 속도를 기록하는 것으로 나타났습니다. 제안된 알고리즘은 기존 기법들에 비해 뚜렷한 성능 향상을 보였으며, 해당 코드는 GitHub와 같은 플랫폼에서 공개될 예정입니다.



### Deciphering genomic codes using advanced NLP techniques: a scoping review (https://arxiv.org/abs/2411.16084)
- **What's New**: 본 리뷰는 인간 유전체 시퀀싱 데이터 처리를 위한 자연어 처리(NLP) 기술의 활용을 조사합니다. 특히 대형 언어 모델(LLMs)과 transformer 아키텍처가 유전체 법칙 해독에서 어떻게 적용되는지를 다루며, 토큰화(tokenization), transformer 모델, 그리고 규제 주석 예측에 초점을 맞추고 있습니다. 최근 문헌의 데이터 및 모델 접근 가능성을 평가하고 이 도구들의 한계를 이해하고자 합니다.

- **Technical Details**: 리뷰는 PubMed, Medline, Scopus, Web of Science, Embase, ACM Digital Library에서 수행되었으며, NLP 방법론이 유전체 데이터 분석에 적용된 연구에 초점을 맞추었습니다. 2021년부터 2024년 4월까지 발행된 연구 26개가 선정되었고, SARS 코브이드 관련(annotation)이 있는 유전체 시퀀스와 함께 예측 모델의 중요한 정량적 결과가 포함되었습니다. 특히, tokenization은 원시 유전체 시퀀스를 분석 가능한 형식으로 변환하는 과정으로서, 컴퓨터 모델에 대한 접근성을 높입니다.

- **Performance Highlights**: NLP와 LLM의 유전체 시퀀싱 데이터 해석 적용은 대규모 유전체 데이터를 처리하는 효율적인 방법을 제공하여 개인 맞춤형 의학 발전에 기여할 수 있는 잠재력이 있습니다. 이 리뷰는 NLP 통합의 혁신적인 가능성을 강조하며, 연구자들이 유전체 데이터를 한층 심도 깊게 이해하는 데 도움이 되는 방법을 제시합니다. 그러나 데이터의 복잡성, 모델 해석 가능성 및 검증 등 현재의 한계를 논의하고 극복하기 위한 추가 연구가 필요합니다.



### Boosting 3D Object Generation through PBR Materials (https://arxiv.org/abs/2411.16080)
Comments:
          Accepted to SIGGRAPH Asia 2024 Conference Papers

- **What's New**: 본 논문에서는 Physics-Based Rendering (PBR) 재료의 관점에서 3D 객체 생성의 품질을 향상시키기 위한 혁신적인 접근 방식을 제안한다. 특히, 알베도(albedo), 거칠기(roughness), 금속성(metalness) 및 범프맵(bump maps)과 같은 PBR 재료의 구성 요소를 분석하여 이러한 값을 효율적으로 추출한다. 기존의 방법론들이 주로 텍스처에 국한되어 있었던 점에서 벗어나, 실제 조명 조건에서의 렌더링 품질을 향상시키는 데 중점을 두고 있다.

- **Technical Details**: 알베도 및 범프맵 추출을 위해, 저자들은 합성 데이터로 미세 조정된 Stable Diffusion 모델을 활용하며, 이를 통해 생성된 3D 객체에 일관된 알베도 UV 및 범프 UV를 제공한다. 또한 거칠기 및 금속성 맵에 대해서는 반자동(semi-automatic) 프로세스를 도입하여, 직관적 조정을 허용하는 방식으로 처리한다. 이 과정은 Segment-Anything-Model에 의해 3D 분할 마스크를 얻어내어, 객체의 의미론적 일관성을 유지한다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 다양한 최신 3D 생성 방법들과 결합하여 생성된 3D 객체의 품질과 사실성을 상당히 향상한다. 특히, 자연스러운 재조명 효과와 함께 고해상도 지형 품질을 제공하며, 기존 방법에 비해 더 정교한 결과를 도출할 수 있음을 보여준다. 그러므로 본 연구는 향후 3D 콘텐츠 생성 분야에 중요한 기여를 할 것으로 기대된다.



### Debiasing Classifiers by Amplifying Bias with Latent Diffusion and Large Language Models (https://arxiv.org/abs/2411.16079)
Comments:
          8 pages + Appendix

- **What's New**: 본 논문에서는 DiffuBias라는 새로운 파이프라인을 소개합니다. 이 방법은 텍스트에서 이미지를 생성하여 분류기의 강건성을 향상시키는 데 중점을 둡니다. 특히, 훈련 단계 없이 편향(conflict) 샘플을 생성할 수 있는 점이 특징입니다. DiffuBias는 사전 훈련된 diffusion 및 이미지 캡셔닝 모델을 활용하여 편향을 도전하는 이미지를 생성함으로써 데이터 샘플을 더 대표성 있게 만듭니다.

- **Technical Details**: DiffuBias는 사전 훈련된 generative 모델, latent diffusion 모델, 이미지 캡셔닝 모델을 이용하여 편향-conflict 샘플을 생성하는 간단하면서도 효과적인 debiasing 프레임워크입니다. 이 방법은 생성 모델을 추가로 훈련할 필요가 없으며, 이런 설정 덕분에 리소스를 절약할 수 있습니다. 또한, DiffuBias는 기존의 GAN보다 더 컴팩트하고 효율적인 접근법으로 지속 가능한 개발에 기여합니다.

- **Performance Highlights**: 본 연구의 종합적 실험 결과는 DiffuBias가 여러 벤치마크 데이터셋에서 최첨단 성능을 달성했음을 보여줍니다. 특히, 데이터 셋에서 편향-conflict 비율이 증가할수록 모델 성능이 향상된다는 점도 확인되었습니다. 또한 다양한 생성 모델에 대한 탄소 배출 및 에너지 소비 비교 분석을 통해 계산 효율성의 중요성을 강조합니다.



### The brain versus AI: World-model-based versatile circuit computation underlying diverse functions in the neocortex and cerebellum (https://arxiv.org/abs/2411.16075)
- **What's New**: 이번 논문은 인공지능(AI)과 뇌의 신피질(neocortex) 및 소뇌(cerebellum) 간의 유사성을 탐구하고, 특히 기능적 도메인 간의 비교를 가능하게 하는 새로운 접근 방식을 소개합니다. 저자들은 회로 구조(circuit structure), 입력/출력(input/outputs), 학습 알고리즘(learning algorithm)으로 회로 계산(circuit computation)을 세분화하여, 각 요소 간의 유사성을 평가합니다. 이를 통해 AI와 뇌의 폭넓은 유사성과 수렴 진화를 발견하고, 신경과학(neuroscience)의 핵심 개념에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 논문에서는 신피질과 소뇌가 과거 정보에서 미래 사건을 예측하고 그 과정에서 발생하는 예측 오차(prediction errors)로부터 학습하는 방식에 주목합니다. 저자들은 이러한 메커니즘이 인공지능의 처리 방식에서 영감을 받아 새로운 이론을 제안하며, 특히 내부 모델(internal models)과 거울 신경(special mirror neuron system) 체계의 기존 이론을 통합합니다. 이론에 따르면, 세 가지 핵심 프로세스: 예측(prediction), 이해(understanding), 생성(generation)이 뇌의 기능적 다양성을 가능하게 하는 기반을 형성합니다.

- **Performance Highlights**: 이 연구의 결론은 뇌가 일관된 회로 구조로 다양한 기능을 수행할 수 있는 이유와 원리를 설명하고, 인공지능이 전통적인 신경 과학 이론과 어떻게 연결될 수 있는지를 제시합니다. 특히, 이 연구는 예측 및 학습 메커니즘이 우리 뇌의 여러 기능을 통합하는 데 필수적임을 강조하며, 신경과학의 발전에 기여할 잠재력을 가지고 있습니다. 이러한 통찰은 AI 및 뇌 연구 분야 간의 경계를 허물어 혁신적인 발전으로 이어질 것으로 기대됩니다.



### Soft-TransFormers for Continual Learning (https://arxiv.org/abs/2411.16073)
- **What's New**: 이번 논문에서는 Well-initialized Lottery Ticket Hypothesis (WLTH)에 영감을 받아 소프트 네트워크(Soft-Network)를 통해 각 작업에 최적화된 지속적인 학습 방법인 Soft-TransFormers(Soft-TF)를 제안합니다. Soft-TF는 연속 학습(Continual Learning, CL) 중에 희소 레이어의 가중치를 공동 최적화하여 작업 적응형 소프트 네트워크를 생성하고, 사전 학습된 레이어의 파라미터를 고정하여 재학습 과정에서 발생하는 망각을 방지합니다.

- **Technical Details**: Soft-TransFormers는 각 작업에 대해 최적화된 소프트 네트워크를 선택하여 연속적으로 학습합니다. 이 방법은 Vision Transformer (ViT)와 CLIP에 대해 실험을 수행하여 CL 시나리오에서 재학습 중 발생하는 Catastrophic Forgetting (CF)을 최소화합니다. Soft-TF는 GPF를 유지하며, 전체 레이어는 고정되지만 특정 작업에 맞춘 가중치를 동적으로 조정합니다. 또한, WLTH에 기반하여 프리트레인 모델을 최적화하여 작업별 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, Soft-TF는 Class-Incremental Learning (CIL) 및 Task-Incremental Learning (TIL) 시나리오에서 DualPrompts와 같은 기존 방법들보다 뛰어난 성능을 보여주었습니다. Soft-TF는 다양한 CL 시나리오에서 최첨단의 성능을 달성하며, 일반화된 CL 모델로의 가능성을 입증합니다. 본 연구는 프리트레인된 Transformer 네트워크의 잠재력을 극대화하여 지속적인 학습 방법에서는 새로운 패러다임 전환을 시도하고 있습니다.



### UnitedVLN: Generalizable Gaussian Splatting for Continuous Vision-Language Navigation (https://arxiv.org/abs/2411.16053)
- **What's New**: 이번 연구에서는 Vision-and-Language Navigation (VLN) 분야에서 새로운 paradigm인 UnitedVLN을 소개합니다. UnitedVLN은 3D Gaussian Splatting (3DGS) 기술을 기반으로 하여, 높은 품질의 360도 시각 이미지와 의미적 특징을 통합하여 에이전트가 미래 환경을 더 효과적으로 탐색할 수 있게 합니다. 이 방식은 기존의 기술들에 대한 의존도를 줄이고, 탐색의 효율성을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: UnitedVLN은 두 가지 주요 방식을 사용합니다. 첫 번째는 Search-Then-Query (STQ) 샘플링 방식으로, 이 방법은 인접 포인트를 검색하고 K-최근접 이웃을 쿼리하는 과정으로 구성됩니다. 두 번째는 Separate-Then-United (STU) 렌더링 방식인데, 이는 NeRF를 사용해 고수준 의미적 특징을 렌더링하고, 3DGS를 통해 시각 정보를 결합하여 다양한 환경에서의 탐색력을 강화합니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, UnitedVLN은 기존 VLN-CE 벤치마크에서 최첨단 방법들을 능가하는 성능을 보여줍니다. 이 연구는 VLN-CE의 복잡한 환경에서 더욱 효율적이고 강력한 탐색 성능을 달성할 수 있음을 입증하며, 멀티모달 접근 방식을 통해 에이전트의 내구성을 증대시킬 수 있음을 시사합니다.



### From Dashcam Videos to Driving Simulations: Stress Testing Automated Vehicles against Rare Events (https://arxiv.org/abs/2411.16027)
- **What's New**: 이번 연구에서는 현실 세계의 자동차 사고 동영상을 시뮬레이션 시나리오로 자동 변환하는 새로운 프레임워크를 제안합니다. 이 방법은 Video Language Models(VLM)를 활용하여 대시캠 영상으로부터 SCENIC 스크립트를 생성, CARLA 시뮬레이터에서 사실적인 시뮬레이션 시나리오를 생성합니다. 이 접근 방식은 단순한 시나리오 재구성을 목표로 하지 않고, 원본 비디오에서 본질적인 주행 행동을 포착하며 환경 변수에 대한 유연성을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 네 가지 구성 요소로 이루어져 있습니다: 1) 현실 비디오를 SCENIC 스크립트로 변환, 2) SCENIC 스크립트로부터 시뮬레이션 비디오 생성, 3) 실제와 시뮬레이션 비디오 간 유사성 분석, 4) 시뮬레이션 비디오의 일관성을 보장하기 위한 반복적 개선. 입력 비디오는 GPT-4o 모델의 프롬프트 엔지니어링을 통해 서술적 스크립트로 변환됩니다. 이 프로세스는 각종 시나리오의 구조적이고 정확한 설명 언어로의 매핑을 가능하게 합니다.

- **Performance Highlights**: 예비 결과에 따르면 이 자동 변환 프로세스는 몇 분 만에 완료되며, 인간의 개입 없이도 높은 충실도를 유지합니다. 기존의 수작업 시나리오 구축 시간이 몇 시간에서 몇 분으로 단축되는 중대한 시간 효율성을 보여줍니다. 이 결과는 ADS의 테스트 환경을 실질적으로 개선하는 데 기여할 것입니다.



### Performance Implications of Multi-Chiplet Neural Processing Units on Autonomous Driving Perception (https://arxiv.org/abs/2411.16007)
Comments:
          DATE'2025

- **What's New**: 본 연구에서는 차량 AI 인식 작업을 가속화하기 위해 emerging chiplet-based Neural Processing Units의 적용을 탐구합니다. Chiplets 기술은 성능, 모듈성 및 사용자 정의 측면에서 비용 효율적인 절충안을 제공함으로써 새로운 차량 아키텍처에 통합되고 있습니다. 연구의 주요 초점은 Tesla Autopilot의 인식 파이프라인을 사례 연구로 사용하여 이를 분석하는 것입니다.

- **Technical Details**: 연구는 Tesla FSD(Full Self Driving) SoC 아키텍처를 기본으로 하여, MCM(Multi-Chip Module) AI 가속기를 사용하여 AI 인식 작업을 가속화하는 방법을 모색합니다. 기존 인식 파이프라인의 단계별 실행 속성과 하드웨어 가속 성능을 분석하기 위해 표준 DNN 성능 시뮬레이터인 MAESTRO를 사용합니다. 최종적으로, 다양한 모델을 통합하여 MCM-NPU에 인식 작업을 맵핑하기 위한 저비용 스케줄링 알고리즘을 구현합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 단일 칩 가속기 설계에 비해 처리량(throughput)과 처리 엔진 활용도에서 각각 82%와 2.8배 증가를 달성했습니다. 이러한 성과는 비용과 에너지 효율성 간의 절충안을 강조하며, 인식 작업의 성능 효율성을 향상시키는 새로운 스케줄링 방법론을 통해 이루어졌습니다. 이번 연구는 자동차 AI 작업에서 chiplets 기술의 응용 가능성을 제시하는 중요한 기초자료로 작용할 것입니다.



### eFedLLM: Efficient LLM Inference Based on Federated Learning (https://arxiv.org/abs/2411.16003)
- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 효율성과 접근성을 향상시키기 위한 새로운 접근법을 제시합니다. 특히_transformer 기반의 연합 학습(federated learning, FL)과 모델 병렬 분산 훈련(model-parallel distributed training)을 활용하여, 다양한 자원을 가진 사용자들이 협력하여 LLM 훈련을 가능하게 하였습니다. 이 접근법은 자원을 효율적으로 분배함으로써, 더 넓은 사용자층이 최첨단 LLMs에 접근하도록 만드는 데 기여할 수 있습니다.

- **Technical Details**: 연합 학습(FL)은 LLM 훈련에서 분산된 디바이스를 활용하여 트랜스포머 대리모델의 서로 다른 레이어를 훈련하는 데 사용됩니다. 이 연구에서는 보안을 강화하고 자원 사용을 최적화하기 위해 사용자들이 다양한 수준의 계산 자원을 가진 상태에서 협력하는 과정이 포함됩니다. 한편, 메모리 계층 전략(memory hierarchy strategies) 및 특이값 분해(Singular Value Decomposition, SVD)를 사용하여 훈련 과정에서의 메모리 및 계산 효율을 추가로 향상시킵니다.

- **Performance Highlights**: 실험 결과는 이 모델이 자원 사용을 크게 최적화하며 최첨단 LLMs의 접근성을 민주화하는 데 성공하였음을 보여줍니다. 해당 연구에서 BERT 모델의 한 레이어 분석을 바탕으로, 가중치 행렬의 압축 비율 증가에 따라 대역폭 사용률이 감소하는 경향을 나타낸 것을 확인하였습니다. 이로써, 최적화된 LLM 운영이 더 많은 사용자에게 가능해지며 경쟁력 있는 인공지능 기술의 발전을 뒷받침할 수 있습니다.



### Ensuring Fair LLM Serving Amid Diverse Applications (https://arxiv.org/abs/2411.15997)
- **What's New**: 이 논문에서는 다양한 애플리케이션을 호스팅하는 다중 사용자 대형 언어 모델(LLM) 플랫폼의 공정성 문제를 다룹니다. FairServe라는 새로운 시스템을 제안하여, 요청이 시스템을 과도하게 이용하는 경우에도 공정한 LLM 접근을 보장할 수 있도록 개발했습니다. 이 시스템은 애플리케이션의 특성을 고려한 요청 제어 및 가중 서비스 카운터 기반 스케줄링 기법을 도입하여 서비스의 공정성을 유지합니다.

- **Technical Details**: FairServe는 (1) 과부하 및 상호작용 기반의 제어(Overload and Interaction-driven Throttling, OIT) 및 (2) 가중 서비스 카운터(Weighted Service Counter, WSC) 스케줄러 두 가지 주요 구성 요소로 이루어져 있습니다. OIT는 시스템이 과부하일 때만 요청을 제어하며, LLM와의 상호작용 수준에서 제어를 수행하여 애플리케이션 특성을 반영합니다. WSC는 사용자 요청 중 가장 적은 서비스를 받은 사용자에게 우선적으로 요청을 처리합니다.

- **Performance Highlights**: FairServe의 실험 결과, 기존의 최첨단 방법과 비교하여 대기열 지연을 10.67배에서 93배까지 줄였으며, 초기 토큰 생성 대기 시간을 1.03배에서 1.06배까지 감소시켰습니다. 또한, 0%의 토큰 낭비를 유지하며 처리량을 1.03배에서 1.75배 증가시켰습니다. 이러한 성과는 사용자에게 더욱 효과적인 서비스를 가능하게 하며, 향후 수백만 고객에게 혜택을 줄 것으로 기대하고 있습니다.



### Anda: Unlocking Efficient LLM Inference with a Variable-Length Grouped Activation Data Forma (https://arxiv.org/abs/2411.15982)
Comments:
          To appear in 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA 2025)

- **What's New**: 이 논문에서는 새로운 адаптив한 데이터 형식인 Anda를 제안합니다. Anda는 그룹 공유 지수 비트와 동적 가수 비트 할당을 포함한 적응형 데이터 형식으로, 활성화의 정밀도를 최적화하고 효율적인 LLM 추론을 가능하게 합니다. 또한, 반복적인 포스트 트레이닝 적응 정밀도 검색 알고리즘을 개발하여 다양한 LLM 모듈의 비트 폭을 최적화하여 모델의 정확성과 에너지 효율성을 균형있게 조절합니다.

- **Technical Details**: Anda는 비트-플레인 기반 데이터 조직_scheme과 bit-serial 계산을 지원하는 Anda 향상 처리 장치, 런타임에서 비트 플레인 Anda 압축기를 포함한 효율적인 하드웨어 아키텍처로 구성됩니다. 이 아키텍처는 LLM 모델의 활성화 정밀도 최적화를 통해 FP-INT GeMM 연산의 성능을 극대화합니다. 논문의 각 섹션은 현재의 기술적 한계와 Anda 포맷의 최적화 방법을 서술하고 있습니다.

- **Performance Highlights**: Anda는 FP-INT GeMM 연산에서 평균 2.4배의 속도 향상, 4.0배의 영역 효율성, 3.1배의 에너지 효율성 개선을 달성했습니다. 또한 다양한 LLM 모델에서 강력한 적응성을 보여주어 여러 응용 시나리오와 시스템 성능 요구 사항에 적합하게 작동합니다. 이러한 성과는 LLM의 효율적 추론을 가능하게 하여 다양한 배포 시나리오에서 활용될 수 있습니다.



### DRIVE: Dual-Robustness via Information Variability and Entropic Consistency in Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.15976)
- **What's New**: 드라이브(DRIVE)라는 새로운 SFUDA 프레임워크가 소개되었습니다. 이 방법은 두 개의 모델을 동시에 사용하는 이중 모델 아키텍처를 도입하여, 대상 도메인의 다양성과 불확실성을 효과적으로 반영합니다. 특히, 예측 불확실성을 기반으로 레이블 가중치를 조정하는 엔트로피 기반의 의사 레이블링 전략을 사용하여 모델이 신뢰할 수 있는 데이터에 중점을 두도록 합니다.

- **Technical Details**: 드라이브는 두 단계로 이루어진 적응 프로세스를 갖습니다. 첫 번째 단계에서는 상호 정보 일관성 손실을 적용하여 모델이 안정적인 특징에 대해 정렬되도록 하여 노이즈의 영향을 최소화합니다. 두 번째 단계에서는 첫 번째 단계의 손실 정보를 이용해 PGD(Projection Gradient Descent)를 통해 불확실성이 높은 영역을 탐색하며, 모델의 견고성과 적응성을 높이는 데 기여합니다.

- **Performance Highlights**: 드라이브는 표준 SFUDA 벤치마크에서 이전의 방법들을 지속적으로 초월하여 개선된 적응 정확도와 안정성을 보여주었습니다. 특히, 복잡한 대상 도메인에서의 일반화 능력과 불확실성에 대한 강건성을 향상시키면서 최적의 성능을 도출했습니다.



### Advancing Transformative Education: Generative AI as a Catalyst for Equity and Innovation (https://arxiv.org/abs/2411.15971)
Comments:
          12 pages

- **What's New**: 이 논문은 Generative AI가 교육에 미치는 영향을 탐구하고 있으며, 개인화된 학습과 혁신적인 관리 효율성을 제공하는 새로운 가능성을 제시합니다. 그러나 이 기술의 통합과 관련된 윤리적 문제, 기술 인프라의 한계, 교육자 역할의 재정의와 같은 도전 과제가 존재함을 강조합니다. 이 연구는 책임 있는 AI 통합을 위한 실행 가능한 프레임워크를 제안하며, 교육의 공평성과 혁신을 촉진하는 방향을 목표로 하고 있습니다.

- **Technical Details**: 이 연구는 Generative AI의 교육적 효과를 평가하고, AI 도구들이 학습 방법론 및 학습 성과에 미치는 영향을 분석하고 있습니다. Constructivist theory, Vygotsky의 Zone of Proximal Development, 그리고 Connectivism과 같은 교육 이론의 관점에서 Generative AI의 통합이 어떻게 교육의 질을 높일 수 있는지를 살펴보고 있습니다. 특히, AI 기반 튜터링 시스템은 개인의 필요에 맞춰 콘텐츠의 복잡성을 동적으로 조정하고, 즉각적인 피드백 루프를 통해 학생들이 자기 주도적 학습을 할 수 있도록 돕습니다.

- **Performance Highlights**: Generative AI는 개인화된 학습을 촉진하고, 학생들의 학업 성과를 향상시키는 데 효과적이라는 결과를 보여주었습니다. AI 도구를 사용한 학생들은 전통적인 방법에 비해 20% 높은 시험 점수를 기록하였고, 약 75%의 학생들이 AI 도구 통합으로 인해 동기부여가 증가했다고 응답하였습니다. 이는 Generative AI가 학습 경험에 즉각적인 피드백을 제공하고, 학습자의 개별적 요구에 맞춰 적절한 수준의 도전을 설정하여 학습할 수 있도록 지원한 결과입니다.



### Partial Identifiability and Misspecification in Inverse Reinforcement Learning (https://arxiv.org/abs/2411.15951)
- **What's New**: 이 논문에서는 Inverse Reinforcement Learning (IRL)에서 보상 함수 $R$의 부분 식별성(partial identifiability) 및 잘못 지정(misspecification) 문제를 수학적으로 분석합니다. 이전 연구에 비해 IRL의 행동 모델에 대한 정교한 정의와 분석을 제공하여 이론적 기초를 확립하고 있습니다.

- **Technical Details**: 저자들은 다양한 행동 모델에 따라 보상 함수의 불확실성을 정량화하고 완전히 특성화하는 방법을 제시합니다. 보상 함수 $R$에 대한 잘못된 추론이 발생하기 전에 관찰된 시연자 정책이 어떻게 각 표준 행동 모델과 다른지를 명확하게 설명하는 필요 충분 조건을 제공합니다.

- **Performance Highlights**: 주요 결과로는 IRL의 부분 식별성 및 잘못 지정 문제를 다루는 통합 프레임워크를 제안하고 있습니다. 이 프레임워크는 새로운 IRL 모델의 부분 식별성 및 잘못 지정 강건성(misspecification robustness)을 쉽게 도출하거나 다양한 보상 학습 알고리즘을 분석하는 데 활용될 수 있는 여러 형식적 도구들을 포함하고 있습니다.



### Generative Context Distillation (https://arxiv.org/abs/2411.15927)
- **What's New**: 최근의 대규모 언어 모델(LLM) 기반 애플리케이션에서 사용되는 프롬프트는 고정적이고 길어 컴퓨팅 오버헤드가 크다는 문제를 해결하기 위해 Generative Context Distillation (GCD)라는 경량화된 프롬프트 내재화(method) 방법이 제안되었습니다. 이 방법은 프롬프트 입력의 행동을 복제함과 동시에 프롬프트의 내용 및 모델 행동 변경의 이유를 생성하는 방식으로 작동합니다. GCD는 다양한 자동화된 애플리케이션 시나리오에서 복잡한 프롬프트를 효과적으로 내재화할 수 있음을 입증합니다.

- **Technical Details**: GCD는 프롬프트를 단순히 입력으로 사용하는 대신, 목표 프롬프트를 생성하도록 학습됩니다. 이 방법은 1) 컨텍스트 증류(context distillation) 접근 방식을 활용하여 행동을 안내하고, 2) 프롬프트에 기반하여 결과가 왜 변경되어야 하는지를 유추하는 과정을 포함하는 결합 손실 훈련을 사용합니다. 또한, 훈련 데이터셋이 없는 시나리오를 위해 대화 데이터를 자동으로 수집하는 데이터 합성(data synthesis) 방법을 도입하여, 두 역할을 전환하여 멀티 턴 의사소통 데이터셋을 생성합니다.

- **Performance Highlights**: GCD는 AgentBench를 사용한 길이 있는 프롬프트의 에이전트 애플리케이션 시나리오에서 평가되었으며, 프롬프트 입력 없이도 뛰어난 성능을 유지했습니다. OS 상호작용 에이전트 작업에서 100% 성능 유지를 달성했으며, 웹 기반 에이전트 작업에서는 1,000개 이상의 토큰에 대해 최소 82%의 성능을 유지했습니다. GCD는 길이 있는 프롬프트를 다룰 때 39%의 효율성 향상을 보이며, 기존의 압축 기반 방법을 초월하는 성능을 보여주었습니다.



### Making Images from Images: Interleaving Denoising and Transformation (https://arxiv.org/abs/2411.15925)
- **What's New**: 이 논문에서는 이미지 요소들의 재배치를 통해 새로운 이미지를 생성하는 방법을 제안합니다. 사용자는 블록, 원형 링, 또는 개별 픽셀과 같은 다양한 형태로 지역을 정의할 수 있으며, 이러한 방식으로 기존 이미지를 획기적인 주제로 변환할 수 있습니다. 제안된 방법은 이미지의 내용과 필요한 변환을 동시에 학습하여, 최적화 문제로 형성되어 이미지를 생성합니다. 이전 방법들과는 달리, 지역의 수가 증가할수록 문제 해결이 더 쉬워지고 결과가 향상됩니다.

- **Technical Details**: 연구에서는 딥러닝 기반의 모델을 활용하여 이미지 변환 과정을 최적화 문제로 모델링하고 이미지 확산(diffusion)과 에너지 최소화 과정(minimization)을 교차 적용합니다. 기존의 정적 소스 이미지 대신 동적으로 변환을 발견할 수 있는 방법을 제시하여, 기존 데이터에서 새로운 이미지를 생성하는 것이 가능합니다. 이는 헝가리안 알고리즘(Hungarian Method)을 사용해 최적 배치를 구현하며, 무한한 소스 이미지를 활용하여 다양하고 매력적인 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 픽셀 공간과 잠재 공간(latent space) 모두에서 효과적으로 수행되는 것을 보여줍니다. 또한, 서로 다른 주제를 표현하기 위해 다수의 소스 이미지를 사용하는 창의적인 확장도 가능함을 입증하였습니다. 이러한 결과들은 다양한 이미지 생성 가능성을 제시하며, 특히 시각적 착시 효과 생성에 혁신적인 접근을 제공합니다.



### Deep Learning for automated multi-scale functional field boundaries extraction using multi-date Sentinel-2 and PlanetScope imagery: Case Study of Netherlands and Pakistan (https://arxiv.org/abs/2411.15923)
Comments:
          09 pages, To be published

- **What's New**: 이번 연구는 네덜란드와 파키스탄의 두 가지 지역에서 다중 시간대 위성 이미지를 활용한 기능적 농지 경계 구분(multi-temporal satellite imagery and functional field boundary delineation)의 효과성을 심층 학습 세분화 아키텍처(deep learning semantic segmentation architecture)를 통해 탐구했습니다. 2022년 4월, 8월, 10월의 PlanetScope 및 Sentinel-2 이미지를 수집하여 결국 작물 경계 데이터로 활용하였습니다.

- **Technical Details**: 이 연구에서는 네덜란드의 기본 등록 작물 구획(BRP) 벡터층을 레이블 학습 데이터로 사용하였고, 파키스탄에서는 자체 제작한 농지 경계 벡터 데이터를 이용하였습니다. UNET 아키텍처 모델을 통해 다양한 조합의 다중 날짜 이미지(multi-date images)와 NDVI 스택을 평가하며, IoU(Intersection over Union) 점수 비교 분석을 통해 제안된 다중 날짜 NDVI 접근법의 효과성을 검토하였습니다.

- **Performance Highlights**: 결과는 다중 날짜 NDVI 스택이 계절에 따른 작물 성장의 시간적 맥락을 제공함을 보여주었고, 게다가 소규모 농업 지역의 농지 경계 추출을 위한 높은 공간 해상도의 중요성을 강조하였습니다. 연구 결과는 이질적인 농업 환경에서 향상된 자동 농지 경계 구분을 위해 다중 규모 구현으로 확장될 수 있습니다.



### A Training-Free Approach for Music Style Transfer with Latent Diffusion Models (https://arxiv.org/abs/2411.15913)
Comments:
          Codes will be released upon acceptance

- **What's New**: 이 논문은 음악 스타일 전이에 대한 새로운 훈련 필요 없는 접근 방식을 제안합니다. 기존의 복잡한 훈련 절차 없이도 사전 훈련된 Latent Diffusion Models (LDM)를 활용하여 음악의 스타일을 변환할 수 있습니다. 이를 통해 내용을 변형하지 않고 스타일을 보존하면서도 참조 음악의 스타일을 콘텐츠 음악에 효과적으로 적용할 수 있게 되었습니다.

- **Technical Details**: 제안된 방법은 LDM의 자기 주의(self-attention) 기능을 조작함으로써 작동합니다. 특정 레이어의 키(key)와 값(value)을 교체하여 멜 스펙트로그램의 지역 텍스처를 보존합니다. 이는 추가적인 훈련 없이도 빠른 스타일 전환이 가능하게 하며, 쿼리 보존, 주의 온도 조정, 초기 잠재 적응 인스턴스 정규화(AdaIN)와 같은 여러 기술을 추가하여 스타일화 과정을 한층 개선합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 기존의 음악 스타일 전이 방법론에 비해 훨씬 우수한 스타일 전이 및 멜로디 보전 성능을 보여주었습니다. 결과적으로, 이 연구는 개인화된 음악 생성의 새로운 창작 경로를 열어주며, 음악 제작에 관한 접근 가능성을 넓힙니다.



### Bimanual Grasp Synthesis for Dexterous Robot Hands (https://arxiv.org/abs/2411.15903)
Comments:
          Published in RA-L 24', 8 pages, 9 figures, 3 tables

- **What's New**: 본 논문에서는 로봇의 물체 조작 능력을 향상시키기 위해 이양손(bimanual) 그립 포즈를 생성하는 BimanGrasp 알고리즘을 제안합니다. 이 알고리즘은 그립 안정성과 실현 가능성을 고려하여 에너지 함수를 최적화하여 그립 포즈를 생성합니다. 또한, Isaac Gym 물리 시뮬레이션 엔진을 통해 검증된 BimanGrasp-Dataset을 생성하였으며, 이는 15만 개 이상의 검증된 이양손 그립 포즈를 포함하고 있어 데이터 기반 접근 방식을 통한 이양손 그립 합성이 가능하게 합니다.

- **Technical Details**: BimanGrasp 알고리즘은 확률적 최적화를 사용하여 높은 차원 구성 공간에서 이양손 그립 포즈를 탐색합니다. GPU 기반 최적화를 구현하여 BimanGrasp-Dataset을 합성하는 과정에서, 이 데이터셋은 900개의 객체에 대한 15만 개 이상의 그립을 포함합니다. 각 그립은 Isaac Gym 환경에서 시뮬레이션을 통해 검증되어, 이전의 단일 손 그립 기법으로는 다루기 어려운 대형 및 중량 객체를 처리할 수 있다는 점을 입증합니다.

- **Performance Highlights**: 제안된 BimanGrasp-DDPM 모델은 BimanGrasp-Dataset을 기반으로 훈련되어 69.87%의 그립 합성 성공률을 달성했습니다. 이 모델은 BimanGrasp 알고리즘과 비교하여 계산 속도의 상당한 가속화를 이뤘습니다. 이 연구는 이양손 조작 기량을 데이터 중심 패러다임으로 변환하여 효율적으로 다양한 그립 포즈를 생성할 수 있도록 돕습니다.



### Highly Efficient and Unsupervised Framework for Moving Object Detection in Satellite Videos (https://arxiv.org/abs/2411.15895)
Comments:
          8 pages, 8 figures

- **What's New**: 이번 연구에서는 매우 효율적인 비지도 학습 프레임워크인 HiEUM(Highly Efficient and Unsupervised Moving object detection)을 제안합니다. 이 방법은 pseudo labels(가짜 라벨)을 전통적인 방법으로 생성하고 학습 과정에 따라 이를 발전시켜 SVMOD(위성 비디오에서의 움직이는 객체 감지)의 성능을 향상시킵니다. 또한, 희소 샘플링을 이용한 sparse convolutional anchor-free detection network를 통해 배경 영역에서 불필요한 계산을 건너뛰며 효율성을 극대화할 수 있습니다.

- **Technical Details**: HiEUM 프레임워크는 두 가지 핵심 설계 요소를 기반으로 합니다. 첫째, 라벨 자가 발전(self-evolution) 비지도 학습 프레임워크를 개발하여 주석 비용을 줄이고 비지도 SVMOD의 가능성을 연구합니다. 둘째, 희소 스페이셜-템포럴 포인트 클라우드(spatio-temporal point cloud) 표현을 기반으로 희소 샘플링 기법을 적용하여 배경의 중복 계산을 건너뛰고 효율성을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 1024x1024 이미지에서 초당 98.8 프레임을 처리할 수 있으며 상태최고 성능(SOTA)도 달성할 수 있습니다. 실험 결과, SOTA DSFNet과 비교해 약 28.7배, 전통적인 B-MCMD 방법과 비교해 약 4490배의 추론 속도를 개선했습니다. 이러한 결과는 SVMOD의 새로운 방향을 제시하며, 다양한 메서드에 대한 새로운 기준을 제공합니다.



### Navigating the Effect of Parametrization for Dimensionality Reduction (https://arxiv.org/abs/2411.15894)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 파라메트릭(dimensionality reduction) 차원 축소 방법이 전통적인 비파라메트릭(non-parametric) 방법과 동등하지 않다는 것을 증명하였습니다. 파라메트릭 방법은 전역(global) 구조를 유지하나, 중요한 지역(local) 세부 정보를 잃는 경향이 있음을 보여줍니다. 이를 해결하기 위해, 새로운 파라메트릭 방법인 ParamRepulsor를 개발하였으며, 이 방법은 Hard Negative Mining과 강력한 반발(repulsive) 힘을 적용하는 손실(loss) 함수를 통합하였습니다.

- **Technical Details**: 파라메트릭 방법의 성능에서 발생하는 문제를 분석하며, 파라메트릭 접근법이 부정적 쌍(negative pairs)을 퇴치하는 능력이 부족하다는 것을 입증하는 여러 증거를 제시합니다. 또한 손실 함수의 선택이 결과에 미치는 영향도 강조합니다. ParamRepulsor는 이러한 문제를 해결하기 위해 설계되었으며, 지역 구조 보존(local structure preservation)에 있어 최신 기술(state-of-the-art)의 성능을 자랑합니다.

- **Performance Highlights**: ParamRepulsor 방법은 기존 파라메트릭 방법들보다 지역 구조 보존에서 우수한 성능을 보여주며, 글로벌(global) 구조 표현의 충실도를 손상시키지 않고 있습니다. 이 연구는 파라메트릭 차원 축소 방법의 새로운 가능성을 제시하며, 구현된 코드는 지정된 URL에서 이용 가능합니다.



### Distribution-aware Online Continual Learning for Urban Spatio-Temporal Forecasting (https://arxiv.org/abs/2411.15893)
- **What's New**: 이 논문에서는 도시의 시공간 데이터(urban spatio-temporal data)의 분포 변화(distribution shifts)를 분석하고, 이를 처리하기 위해 DOST라는 새로운 온라인 지속 학습 프레임워크를 제안합니다. DOST는 도시 데이터의 특성에 맞춰 설계된 적응형 ST 네트워크와 변화가 없는 어댑터를 활용하여 각 도시 위치에서의 분포 변화를 동적으로 다룹니다. 또한, 점진적인 변화에 대응하기 위해 수면-각성 학습 전략을 개발하여 온라인 단계에서 반복적으로 어댑터를 미세 조정합니다.

- **Technical Details**: DOST 프레임워크는 도시의 시공간 예측을 위한 분포 인식(Distribution-aware) 온라인 지속 학습을 기반으로 합니다. 이 시스템은 위치에 따라 변화하는 시공간 데이터를 효과적으로 처리하기 위해 배치된 변형이 없는 어댑터(Variable-Independent Adapter, VIA)를 도입합니다. 또한, 수면-각성(Awake-Hibernate, AH) 학습 전략을 통해 자원 소비를 줄이면서도 점진적인 분포 변화에 적절히 대응합니다.

- **Performance Highlights**: 실험 결과에 따르면, DOST는 4개의 실제 데이터셋에서 최고 성능을 발휘하여 평균 0.1초 내에 온라인 예측을 수행하며, 예측 오차를 기본 모델보다 12.89% 감소시켰습니다. 이러한 결과는 DOST의 효율성과 효과성을 입증하며, 기존의 정적 데이터 기반 모델들이 가지고 있는 한계를 극복할 수 있음을 보여줍니다.



### LLMs Do Not Think Step-by-step In Implicit Reasoning (https://arxiv.org/abs/2411.15862)
- **What's New**: 이번 연구는 Chain-of-Thought (CoT) 방법의 명시적 생성 없이도 LLMs의 성능을 향상시키려는 시도를 다루고 있습니다. 그러나 연구에 따르면 암묵적 CoT의 효과는 전통적인 CoT 방법에 비해 여전히 부족하며, LLMs는 중간 단계에 대한 계산을 거의 수행하지 않고 있다고 강조합니다. 이는 LLMs가 경험에 의존하는 경향이 있음을 보여줍니다.

- **Technical Details**: 연구진은 Qwen2.5-72B-Instruct Team 모델을 사용하여 단순한 산술 문제에 대한 실험을 수행했습니다. 이 모델은 중간 단계의 결과를 출력하지 않고 최종적인 답변만을 제공하도록 요구받았으며, 각 단계에 대한 숨겨진 상태(hidden states)를 조사했습니다. 결과적으로, 모델은 2단계 문제에서 유일하게 두 번의 추론을 수행할 수 있었지만, 중간 단계의 계산을 수행하지 않았습니다.

- **Performance Highlights**: 이번 실험을 통해 LLMs가 암묵적 추론 과정에서 안정적이지 않으며, 직관적이고 직접적인 사고 방식에 의존한다는 사실이 확인되었습니다. 결국, LLMs는 특히 큰 모델일수록 산술 문제에서 단지 정답을 제시하는 것에 그치지 않고, 명시적 CoT 없이는 단계별 추론을 수행하지 않는다는 것을 발견했습니다. 연구는 명시적 CoT 방법론의 지속적 필요성을 강조하며 복잡한 작업에서 LLM의 능력을 향상시키는 데 중요한 통찰력을 제공합니다.



### Unveiling the Superior Paradigm: A Comparative Study of Source-Free Domain Adaptation and Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.15844)
Comments:
          Under review

- **What's New**: 이 논문에서는 Unsupervised Domain Adaptation (UDA)와 Source-Free Domain Adaptation (SFDA)의 비교를 통해 SFDA의 우월성을 입증하였습니다. SFDA는 기존의 소스 데이터를 사용하지 않고, 미리 학습된 모델을 활용하여 타겟 도메인에 적응함으로써, 실제 환경에서의 효율성을 크게 향상시킬 수 있음을 보여주었습니다. 또한, 데이터 공유 시나리오를 도입하여 다양한 이해관계자들이 자원을 활용할 수 있는 새로운 방안을 제안하였습니다.

- **Technical Details**: 이 연구는 예측 코딩 이론(predicative coding theory)을 기반으로 SFDA가 기존의 UDA보다 현실 세계에서의 적응력과 효율성을 갖추고 있음을 강조합니다. 특히, 실험을 통해 SFDA가 데이터와 메모리 사용 효율성에서 더 뛰어나며, 부정적 전이(negative transfer)와 과적합(overfitting)으로부터의 저항력이 우수함을 입증했습니다. SFDA는 약 200회 반복(iterations) 내에 수렴하는 반면, UDA는 유사 성능을 달성하는 데 1,000에서 5,000회 반복이 필요합니다.

- **Performance Highlights**: 이 연구의 실험 결과에 따르면, SFDA 방법은 다양한 기준 데이터셋에서 UDA 보다 우수한 성능을 보였고, 특히 복잡한 데이터-모델 융합(data-model fusion) 시나리오에서 효과를 극대화할 수 있음을 입증했습니다. 제안한 가중치 추정(weight estimation) 방법은 기존의 Multi-UDA 및 Multi-SFDA 기술을 뛰어넘어 다양한 데이터 공유 요구사항을 처리하는 데 매우 효과적임을 보여주었습니다. 따라서 이 논문은 현실 세계의 다양한 환경에서 모델 적응을 위한 실용적인 접근 방식을 제공하고 있습니다.



### Efficient and Private: Memorisation under differentially private parameter-efficient fine-tuning in language models (https://arxiv.org/abs/2411.15831)
- **What's New**: 이번 연구에서는 Differential Privacy (DP) 제약 조건 하에서 Parameter-Efficient Fine-Tuning (PEFT) 방법을 조사합니다. PEFT 방법들이 표준 파인튜닝에 비해 우수한 성능을 내면서도 적은 수의 파라미터를 요구하고 프라이버시 유출을 크게 줄일 수 있음을 보여줍니다. 더불어, 고의적인 레이블링 오류를 포함한 데이터 오염 실험을 통해 모델의 기억 용량과 프라이버시 위험을 직접 측정합니다.

- **Technical Details**: DP는 공공 데이터 출시에 대한 공식적인 프라이버시 보존 프레임워크로, 무작위 알고리즘의 출력에 노이즈를 추가함으로써 개인 데이터의 영향을 제한합니다. PEFT 방법은 Adapters, LoRA, (IA)3와 같은 다양한 기법을 사용하여 모델의 일부 파라미터만 업데이트하며, 이는 대규모 모델을 위한 비용 효율적인 파인튜닝을 가능하게 합니다. 이번 연구에서는 세 가지 주요 PEFT 방법을 DP 훈련 하에서 평가하고, 각 방법의 특성과 프라이버시 유출 정도를 분석합니다.

- **Performance Highlights**: PEFT 방법들은 DP 및 표준 파인튜닝 방법들과 비교하여 태스크 중심 성능과 프라이버시 유출을 평가하였습니다. 연구 결과, PEFT 방법들이 계산 효율성과 프라이버시 보존 간의 균형을 이룰 수 있음을 보여주며, 이들 방법을 통해 더 나은 효율성을 유지하면서도 프라이버시 보장을 제공할 수 있음을 확인했습니다.



### Is Training Data Quality or Quantity More Impactful to Small Language Model Performance? (https://arxiv.org/abs/2411.15821)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구는 작은 언어 모델(Small Language Models, SLMs)의 성능에 대한 훈련 데이터의 질과 양의 상대적 영향을 조사합니다. TinyStories 데이터셋을 사용하여 실증 분석을 수행했으며, 데이터셋의 크기와 중복에 따른 변화 분석도 포함되어 있습니다.

- **Technical Details**: 연구에서는 데이터셋 크기를 원래의 25% 및 50%로 조정하고, 중복 비율을 25%, 50%, 75%, 100%로 통제하여 실험을 진행했습니다. 모델 성능은 validation loss, accuracy, perplexity 지표를 기반으로 평가되었으며, 훈련 데이터의 질이 SLMs의 전반적인 성능에 더 큰 역할을 한다는 결과가 나타났습니다.

- **Performance Highlights**: 최소한의 중복은 모델의 정확도에 긍정적인 영향을 미쳐 25% 중복 시 정확도가 0.87% 증가했습니다. 반면, 과도한 중복은 성능 저하를 초래하여 100% 중복시 정확도가 40% 감소하는 결과를 보였습니다. 이러한 결과는 대규모 모델 교육이 재정적 및 계산적 부담을 초래하며, 에너지 소비 문제와 함께 AI 기술의 보다 민주화된 접근을 모색할 필요성을 제기합니다.



### FastTrackTr:Towards Fast Multi-Object Tracking with Transformers (https://arxiv.org/abs/2411.15811)
- **What's New**: 이 논문의 주요 혁신은 FastTrackTr이라는 새로운 다중 객체 추적(MOT) 방법을 제안하는 것으로, Transformer 아키텍처에 기반하여 빠른 추론 속도를 유지하면서 높은 정확성을 달성합니다. 또한, 과거 경로 정보를 통합하는 크로스 디코더 메커니즘을 도입하여 추가 쿼리나 디코더 없이도 효과적으로 성능을 개선합니다. 여러 벤치마크 데이터셋에 대한 광범위한 실험을 통해 기존 SOTA 방법들과 경쟁력 있는 정확성을 유지하면서도 추론 속도를 비약적으로 향상시켰음을 입증하였습니다.

- **Technical Details**: FastTrackTr의 전체 구조는 기본 디코더와 역사 인코더를 포함하여, 초기 역사 정보를 설정하는데 사용됩니다. 이어서 역사 디코더는 추가적인 크로스 어텐션 레이어를 포함하여 이전 프레임의 정보를 처리합니다. 이 방법은 JDT 및 FairMOT와 유사한 원리를 사용하며, 객체의 외관 임베딩을 얻기 위한 ID 임베딩 헤드를 통합하여 추적과 매칭을 동시에 수행합니다.

- **Performance Highlights**: FastTrackTr는 여러 벤치마크 데이터셋에서 경쟁력 있는 정확성을 나타내며, 특히 DanceTrack, SportsMOT, MOT17에서 우수한 성능을 보여줍니다. 실험 결과, 이 모델은 추적 중 쿼리 수를 줄임으로써 모델의 복잡성을 최소화하고 실시간 추적이 가능하도록 설계된 점이 특징입니다. 추가적으로, 이 방법은 대규모 배치 사이즈를 설정할 수 있어 훈련이 상대적으로 용이하다는 장점도 가집니다.



### Broad Critic Deep Actor Reinforcement Learning for Continuous Contro (https://arxiv.org/abs/2411.15806)
Comments:
          7 pages

- **What's New**: 이 논문에서는 연속 제어(continuous control) 분야에서 깊은 강화 학습(deep reinforcement learning, DRL)을 개선하기 위한 새로운 하이브리드 아키텍처를 제안합니다. 이 아키텍처는 깊은 신경망(deep neural networks, DNNs)과 광학적 학습 시스템(broad learning system, BLS)을 통합하여 두 가지 구조적 패러다임의 장점을 함께 활용하고자 합니다.

- **Technical Details**: 제안된 구조에서 비평가 네트워크(critic network)는 BLS를 사용하여 구현되고, 행동자 네트워크(actor network)는 DNN으로 구성됩니다. 비평가 네트워크의 매개변수는 ridge regression을 통해 추정되며, 행동자 네트워크의 매개변수는 경량 경사 하강법(gradient descent)으로 최적화됩니다. 이를 통해 두 가지 전통적인 연속 제어 작업에서 알고리즘의 효과를 평가합니다.

- **Performance Highlights**: 제안된 알고리즘은 널리 인정받는 깊은 결정적 정책 기울기(deep deterministic policy gradient, DDPG) 알고리즘과 비교하여 뛰어난 계산 효율성과 함께 학습 속도를 가속화하는 결과를 보여줍니다. 이 연구에서는 앞으로 다른 액터-크리틱 RL 알고리즘에 제안된 알고리즘을 적용해볼 것을 제안하고 있습니다.



### Benchmarking Active Learning for NILM (https://arxiv.org/abs/2411.15805)
- **What's New**: 이 논문에서는 Non-Intrusive Load Monitoring (NILM)을 위한 액티브 러닝 접근법을 최초로 제안합니다. 이는 여러 가정에서 기기별 데이터를 전략적으로 선택하며, 유용성이 높다고 검증되었습니다. 검증된 방법론은 기존의 무작위 샘플링 기법보다 성능을 더욱 개선할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 켈리 등(2015)의 최초의 신경망 기반 접근법 이후, 최신 기법을 활용하여 불확실성을 정량화하는 기술을 적용했습니다. 본 논문에서는 Monte Carlo Dropout 기법과 엔트로피 및 상호정보량과 같은 두 가지 획득 함수를 사용하여 모델의 성능을 최적화합니다.

- **Performance Highlights**: 연구팀은 Pecan Street Dataport 데이터 세트를 사용하여 실제 환경에서 액티브 러닝을 평가했습니다. 이 접근법을 통해 30%의 데이터를 사용하여 기존 전체 데이터 세트로 훈련한 모델과 유사한 성능을 달성하였고, 고정된 센서 개수로 최대 2배의 분해 오차 감소를 관찰했습니다.



### LoRA-Mini : Adaptation Matrices Decomposition and Selective Training (https://arxiv.org/abs/2411.15804)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 Low-Rank Adaptation(LoRA) 방법을 최적화한 LoRA-Mini를 제안합니다. LoRA-Mini는 저랭크 매트릭스를 네 부분으로 나누어 두 개의 내부 매트릭스만 훈련 가능하게 하여 파라미터 효율성을 크게 향상시킵니다. 이 접근방식은 표준 LoRA에 비해 훈련 가능한 파라미터 수를 최대 20배까지 줄여주며, 성능 또한 유지할 수 있습니다.

- **Technical Details**: LoRA-Mini는 각 LoRA 매트릭스를 훈련 가능 매트릭스와 고정 매트릭스로 분할하여 선택적 훈련을 가능하게 합니다. 이를 통해 업데이트 공간을 제한하고, 고정된 외부 매트릭스가 훈련 과정에서 가이드를 제공합니다. 수학적으로는 LoRA 매트릭스 A와 B를 보조 및 훈련 가능한 구성 요소로 분해하며, 중간 매트릭스만 훈련하도록 설정됩니다.

- **Performance Highlights**: 다양한 모델(BERT, RoBERTa, T5 등)에 대한 실험을 통해 LoRA-Mini가 전체 파인튜닝 접근법과 비교해 비슷한 정확도를 달성하면서 메모리 요구 사항을 대폭 줄일 수 있음을 입증했습니다. 본 연구의 주요 기여는 매트릭스 내 선택적 파라미터 동결 평가, 이전 PEFT 방법 대비 훈련 가능한 파라미터 수 최소화, 다양한 태스크 및 모델에 대한 확장성 평가를 포함합니다.



### Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2 (https://arxiv.org/abs/2411.15802)
- **What's New**: 본 연구는 DINOv2와 같은 2D 자기 지도(self-supervised) 모델을 3D 의료 이미징에 적용함으로써, 기존 방법들의 한계를 극복하고자 합니다. 특히, 진단의 정확성과 해석 가능성을 높이기 위해 Medical Slice Transformer (MST) 프레임워크를 소개합니다. MST는 Transformer 아키텍처와 2D 특징 추출기 DINOv2를 통합하여 3D 의료 이미지를 분석합니다.

- **Technical Details**: MST 프레임워크는 3D 데이터 세트에서의 효과적인 진단 성능 평가를 위해 설계되었습니다. 이 연구에서는 유방 MRI, 흉부 CT, 무릎 MRI 데이터 세트를 포함하여 총 3개의 임상 데이터 세트를 사용했습니다. 진단 성능은 Receiver Operating Characteristic Curve (AUC)를 계산하여 평가하였으며, 해석 가능성은 방사선 전문의의 정성적 비교를 통해 평가되었습니다.

- **Performance Highlights**: MST는 3개의 데이터 세트 모두에서 ResNet보다 높은 AUC 값을 기록했습니다: 유방 (0.94$	ext{±}$0.01 대 0.91$	ext{±}$0.02, P=0.02), 흉부 (0.95$	ext{±}$0.01 대 0.92$	ext{±}$0.02, P=0.13), 무릎 (0.85$	ext{±}$0.04 대 0.69$	ext{±}$0.05, P=0.001). 또한, saliency maps는 MST에서 항상 더 정밀하고 해부학적으로 정확하다는 것을 보여주었습니다. DINOv2와 같은 자기 지도 2D 모델은 MST를 통해 3D 의료 이미징에 효과적으로 적응될 수 있으며, 이는 합성곱 신경망에 비해 향상된 진단 정확성과 해석 가능성을 제공합니다.



### A review on Machine Learning based User-Centric Multimedia Streaming Techniques (https://arxiv.org/abs/2411.15801)
Comments:
          Computer Communications

- **What's New**: 본 논문에서는 사용자 중심의 멀티미디어 서비스에 대한 패러다임 전환이 예견되며, 머신 러닝(Machine Learning, ML)에 기반한 품질(QoE) 모델링 및 스트리밍 전략에 대한 집중이 강조됩니다. 특히, 360° 비디오와 같은 새로운 비디오 형식이 가상 현실(VR)과 증강 현실(AR) 애플리케이션에서 인기를 끌고 있으며, 이로 인해 품질 저하 문제의 극복이 절실합니다. 또한, QoE 관리에 대한 종합적인 개요를 제공하고, 영상 스트리밍의 적응형 전략을 분석하며, 기존 기술의 한계와 향후 연구 과제 역시 다룹니다.

- **Technical Details**: QoE는 사용자가 느끼는 애플리케이션이나 서비스의 주관적 품질 지표로, 시스템 요인(System Influence Factors), 인간 요인(Human Influence Factors), 맥락 요인(Context Influence Factors), 콘텐츠 요인(Content Influence Factors)에 의해 영향을 받습니다. 예를 들어, HD 고화질 비디오 스트리밍이 증가하면서 4K 및 8K 비디오의 비트 전송률 요구사항이 크게 증가하고 있습니다. 이를 해결하기 위해 ADAPTIVE STREAMING 기술과 같은 다양한 네트워크 전송 기법이 제안되고 있는 상황입니다.

- **Performance Highlights**: 서비스 제공업체들은 데이터 소비량의 증가로 인해 QoS(서비스 품질)를 개선해야 하는 압박을 받고 있습니다. 5G 네트워크의 등장과 모바일 비디오 전송의 중요성을 고려할 때, QoE의 향상은 필수적이며 이는 다양한 사용자 중심 전략을 통해 달성될 수 있습니다. 본 논문은 사용자 경험을 극대화하기 위한 ML 기반 QoE 예측 모델과 적응형 스트리밍 기법을 다루어, 데이터 전송의 효율성을 높이고 사용자 만족도를 향상하는 방법을 제시합니다.



### Data Lineage Inference: Uncovering Privacy Vulnerabilities of Dataset Pruning (https://arxiv.org/abs/2411.15796)
- **What's New**: 이 연구에서는 머신러닝 시스템에서 데이터셋 프루닝(dataset pruning)으로 인한 데이터 프라이버시(data privacy) 문제를 체계적으로 탐구합니다. 특히, 모델 훈련 전에만 사용된 데이터라도 프루닝 단계에서의 멤버십 상태가 공격을 통해 감지될 수 있음을 처음으로 밝혔습니다. 이를 해결하기 위해 데이터 중심의 멤버십 추론(Data-Centric Membership Inference, DCMI)이라는 새로운 작업을 소개하고, 데이터 라인이지 추론(Data Lineage Inference, DaLI)이라는 최초의 데이터 중심 프라이버시 추론 패러다임을 제안합니다.

- **Technical Details**: 이 연구에서는 다양한 프루닝 방법에 따른 프라이버시 유출 수준의 차이를 분석하였고, Brimming score라는 새로운 메트릭을 소개하여 프라이버시 보호를 고려한 프루닝 방법 선택을 위한 지침을 제공하고자 하였습니다. DaLI 패러다임 하에, WhoDis, CumDis, ArraDis, SpiDis라는 네 가지 임계값 기반 공격을 제안하였으며, 이를 통해 적대자가 하류 모델에 접근하지 않고도 제한된 사전 지식을 통해 쉽게 중복 집합을 식별할 수 있음을 입증했습니다. 프루닝 방법에 따라 프라이버시 위험이 다르며, 동일한 프루닝 방법에서도 다양한 프루닝 비율에 따라 다른 프라이버시 위험이 존재함을 발견했습니다.

- **Performance Highlights**: 다양한 설정 하에서 DaLI의 견고성을 입증하기 위해 12개의 프루닝 방법, 3개의 데이터셋, 4개의 프루닝 비율을 통해 실험을 수행하였습니다. 실험 결과 Brimming score는 프라이버시 보호를 고려한 프루닝 방법 선택에 효과적으로 기여한다는 것을 보여주었습니다. 또한, DCMI에 대한 최초의 방어 전략인 ReDoMi 방안을 제안하여 프라이버시 보호를 위한 초기 시도로서의 의미를 갖습니다.



### Enhancing the automatic segmentation and analysis of 3D liver vasculature models (https://arxiv.org/abs/2411.15778)
Comments:
          Internship at Simbiotx

- **What's New**: 이 연구는 간암 환자의 수술 평가를 위해 의학 이미지에서 혈관 나무(vessel trees)를 자동으로 식별하는 파이프라인을 개발했습니다. 특히, 정맥 나무인 문맥(portal)과 간정맥(hepatic) 구조의 3D 세분화 및 해석을 개선하는 데 중점을 두었습니다. 이 작업을 통해 기존의 골격화(skeletonization) 방법을 개선하고, 수술 계획을 위한 정확한 해부학적 모델을 제공합니다.

- **Technical Details**: 이 논문은 차별화 가능한 골격화 방법이 간 혈관의 3D 세분화 성능에 미치는 영향을 연구합니다. ClDice 및 형태학적 골격화 손실(morphological skeletonization loss)을 사용하여 혈관 나무의 연결성을 개선하고, 단일 클래스 혈관 세분화를 다중 클래스(separating multi-class)로 전환하여 정맥 나무를 분리합니다. 최종적으로, 이러한 알고리즘은 다양한 기하학적 마커를 추출하여 혈관 나무의 형태학적 분석을 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 다양한 직경을 포함하는 광범위한 혈관 나무에 대해 성공적으로 현재의 골격화 방법을 개선합니다. 새로운 분리 알고리즘은 외과의사들에 의해 검증된 낮은 오류율로 혈관의 다중 클래스 세분화를 제공합니다. 이 논문은 또한 77건의 고품질 간 혈관 데이터셋을 공개하였으며, 개별 가지까지 해부학적으로 주석을 추가하고 해당 혈관의 형태학적 분석을 수행할 수 있는 방법을 제시합니다.



### Beyond Data Scarcity: A Frequency-Driven Framework for Zero-Shot Forecasting (https://arxiv.org/abs/2411.15743)
- **What's New**: 이번 연구에서는 시계열 예측(time series forecasting, TSF)에서의 효과적인 학습에 있어 주파수(frequency)의 중요성을 규명하고, 이를 통해 기존의 데이터 부족 문제를 해결하기 위한 새로운 합성 데이터 생성을 제안합니다. 이를 위해 Fourier 분석(Fourier analysis)을 이용하여 다양한 주파수 성분을 이해하고, 예측 모델의 성능 향상에 기여할 수 있는 방법론을 소개합니다. 특히, 한정된 데이터가 존재하거나 전혀 없을 때의 zero-shot(zero-shot) 및 few-shot(few-shot) 설정에서의 적용 가능성을 강조합니다.

- **Technical Details**: 제안하는 Freq-Synth 방법은 합성 데이터를 생성하기 위한 간단하고 효율적인 방법론으로, 대상 도메인의 샘플링 속도만 요구합니다. Fourier 분석을 통해 데이터의 주파수 성분을 시각화하고, 이는 과적합(overfitting) 및 과소적합(underfitting)의 문제를 이해하는 데 도움을 줍니다. 실험 결과, Freq-Synth를 사용한 합성 데이터는 기존의 방법들에 비해 zero-shot 및 few-shot 설정에서 더욱 향상된 성능을 보였습니다.

- **Performance Highlights**: 우리는 Freq-Synth 접근법을 통해 비기초 모델(non-foundation models) 및 기초 모델(foundation models)에서의 시계열 예측이 개선되는 모습을 보여줍니다. 우리의 접근법은 주파수 정보에 기반하여 타겟 도메인에 맞는 합성 데이터를 생성하며, 이를 통해 다양한 과제에서 효율적으로 결과를 도출할 수 있음을 입증하였습니다. 특히, 주파수 혼돈(frequency confusion) 및 주파수 일반화(frequency generalization) 개념을 도입하여, ZS 예측에서의 도전과제를 명확히 하였습니다.



### PEnG: Pose-Enhanced Geo-Localisation (https://arxiv.org/abs/2411.15742)
Comments:
          8 pages, 6 figures

- **What's New**: 본 연구에서는 도시 규모의 정밀한 이미지 위치 확인을 위한 새로운 기술을 제안합니다. 기존의 방법들이 위치 정확도를 제한하는 문제를 해결하기 위해, 카메라 시점 정보를 활용하여 정확도를 서브 미터 수준으로 개선했습니다. 특히, 우리의 시스템(PEnG)은 두 단계로 구성되어 있으며, 도로 교차로를 효과적으로 예측하고 정교한 위치 추정을 수행함으로써 큰 오류 감소를 달성합니다.

- **Technical Details**: PEnG 시스템은 차례로 도시 규모의 교차 시점 위치 확인과 상대적 자세 추정을 결합하여 작동합니다. 첫 번째 단계에서는 도시 스케일 그래프에서 가장 가능성이 높은 가장자리를 예측하고, 두 번째 단계에서는 이러한 가장자리를 따라 상대적인 자세 추정을 수행합니다. 이 시스템은 평면 도로 영역 내에서 가장 최근에 관측된 도로 교차로를 효과적으로 확인하여 포지셔닝의 정확성을 높입니다.

- **Performance Highlights**: 우리의 연구는 기존 최고의 성능을 가진 모델에 비해 96.90% 오류를 줄이며, 중위 유클리드 거리 오류가 734m에서 22.77m로 감소했습니다. 특히, Manhattan과 같은 복잡한 도시에 대한 일반화 성능을 보여주며 비교적 높은 정밀도로 지역 위치 추정을 수행했습니다. 이 결과는 새로운 전반적인 프레임워크의 필요성을 실증적으로 보여주며, 이후 코드도 공개될 예정입니다.



### LTCF-Net: A Transformer-Enhanced Dual-Channel Fourier Framework for Low-Light Image Restoration (https://arxiv.org/abs/2411.15740)
- **What's New**: LTCF-Net은 저조도(低照度, low-light) 이미지 향상을 위한 새로운 네트워크 아키텍처입니다. 이 모델은 색 정보의 효과적인 분리를 위해 LAB와 YUV 두 가지 색 공간을 활용합니다. 또한, Transformer 아키텍처를 포함하여 이미지를 comprehensively 이해하며, 포뢰 변환(Fourier transform) 모듈을 도입해 출력 이미지의 밝기를 동적으로 조정합니다.

- **Technical Details**: 제안된 LTCF-Net의 구조는 조명 향상에 특화된 두 개의 분기로 나뉘며, 각각 LAB와 YUV 색 공간에서 작동합니다. 각 분기에는 Multi-Head Self-Attention(MHSA) 모듈과 Fourier Brightness Processing(FBP) 모듈이 포함되어 명암의 세부 사항을 정교하게 복원합니다. 이 구조는 조명 및 색 정보의 독립적 처리를 가능하게 하여 복잡한 디커플링 작업을 단순화합니다.

- **Performance Highlights**: LTCF-Net은 LOL-v1, LOL-v2, SID 및 SDSD와 같은 여러 데이터 세트에서 기존의 최첨단 기법보다 뛰어난 성능을 보였습니다. 실험 결과, LTCF-Net의 접근 방식은 자연스러운 색 복원과 균형 잡힌 밝기 분포를 달성하며 시각 품질을 향상시킵니다. 이 연구는 저조도 이미지 향상의 새로운 가능성을 열어주며, 경량 모델을 유지하면서 성능을 극대화하였습니다.



### Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models (https://arxiv.org/abs/2411.15731)
Comments:
          Accepted by WSDM 2025

- **What's New**: 새로운 연구에서는 CTR(Click-Through Rate) 모델의 융합 설계(fusion design)는 여전히 주목받지 못하고 있으며, 기존의 두 가지 단순한 융합 디자인인 stacked와 parallel이 널리 사용되고 있다고 강조합니다. 이러한 발견에 기반하여, OptFusion이라는 방법이 도입되어 융합 학습(fusion learning) 프로세스를 자동화하고, 연결 학습(connection learning) 및 연산 선택(operation selection)을 통합적으로 수행합니다. 이 논문의 새로운 접근법은 CTR 모델의 성능을 상당히 향상시킬 가능성을 제시합니다.

- **Technical Details**: OptFusion의 프레임워크는 하나의 embedding 컴포넌트, n+1 개의 cross 컴포넌트, n 개의 deep 컴포넌트 및 하나의 출력 컴포넌트로 구성되어 있으며, 이는 실제 융합 연결(search) 탐색 후보 세트로 활용됩니다. OptFusion은 한 번의 학습(one-shot learning) 알고리즘을 사용하여 융합 연결과 연산을 동시에 학습함으로써 융합 디자인에서의 복잡성을 효율적으로 다룹니다. 이는 네트워크 아키텍처 탐색(neural architecture search)과 비교해 보다 효율적인 탐색이 가능하다는 점에서 주요한 차별점을 나타냅니다.

- **Performance Highlights**: OptFusion은 세 가지 대규모 데이터셋에서 실험을 통해 그 효율성과 효과성을 입증하였습니다. 실험 결과는 OptFusion이 기존의 CTR 모델보다 더 높은 성능을 발휘함을 보여주었으며, 다양한 구성요소와 결합되었을 때도 우수한 성능을 유지함을 강조합니다. 이 연구는 CTR 예측의 융합 운영(selecting fusion operations)과 연결 성능(selecting fusion connections)의 중요성을 잘 드러내고 있습니다.



### Understanding Student Acceptance, Trust, and Attitudes Toward AI-Generated Images for Educational Purposes (https://arxiv.org/abs/2411.15710)
- **What's New**: 이번 연구는 인공지능(AI)으로 생성된 이미지가 교육 분야, 특히 컴퓨터 과학 및 소프트웨어 공학 전공 대학생들 사이에서 어떻게 활용될 수 있는지를 탐구합니다. 설문조사와 인터뷰를 통해 학생들의 수용도와 신뢰도, 긍정적인 태도를 평가하였습니다. AI 이미지의 교육적 활용 가능성을 탐색하면서, 사용의 용이성과 학문적 이점에 대한 학생들의 높은 평가가 주목받았습니다.

- **Technical Details**: 조사는 학생들이 발표, 보고서, 웹 디자인과 같은 교육 과제에서 AI로 생성된 이미지를 어떻게 수용하는지에 대한 전문가적인 이해를 제공합니다. 데이터 수집 방식으로는 질적(qualitative)인 방법이 구사되었으며, 학생들이 느끼는 신뢰와 수용도는 AI 이미지의 사용 가능성을 높이는 데 기여합니다. 그러나, AI가 제시된 프롬프트(prompt)에 따라 정확히 이미지를 생성하지 못하는 경우가 있어, 이는 기술적 세부사항에 대한 우려를 불러일으킵니다.

- **Performance Highlights**: 연구 결과 학생들은 AI 생성 이미지의 사용이 학업에 미칠 긍정적인 영향과 편리함을 강조하였습니다. 그러나 기술적 정확성의 부족은 detail-oriented 교육 과제에서 AI 활용에 일정한 제약을 가합니다. 교육적인 활용을 극대화하기 위해서는 윤리적 고려사항과 지적 재산권 문제를 다룬 포괄적인 가이드라인을 개발하고, AI 생성 이미지의 품질 기준을 수립해야 합니다.



### Nimbus: Secure and Efficient Two-Party Inference for Transformers (https://arxiv.org/abs/2411.15707)
Comments:
          Accepted by NIPS 2024

- **What's New**: 본 논문에서는 Transformer 모델을 위한 새로운 두 당사자 추론 프레임워크인 \textit{Nimbus}를 제안합니다. 기존의 secure two-party computation (2PC) 방식의 한계를 극복하는데 초점을 맞추고 있으며, 특히 linear layer와 non-linear layer에서의 성능 향상을 목표로 하고 있습니다. 이 프레임워크는 새로운 행별 인코딩 방식을 채택하여 homomorphic matrix multiplication의 효율을 극대화합니다.

- **Technical Details**: \textit{Nimbus}는 Client-Side Outer Product (COP) 프로토콜을 도입하여 linear layer에서의 secure matrix multiplication을 용이하게 만듭니다. COP 프로토콜은 모델 가중치를 암호화하여 서버가 클라이언트에 전달할 수 있도록 하여 입력 통신을 제거합니다. 또한, non-linear layer에서는 입력 분포의 정규 패턴을 활용하여 전통적인 방법보다 낮은 차수의 다항식 근사를 가능하게 하여 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, \textit{Nimbus}는 SOTA 프로토콜 대비 secure matrix multiplication 속도를 2.9배에서 12.5배까지 개선하였으며, non-linear layer에서 \mathsf{GELU}와 \mathsf{Softmax}의 성능을 2.9배에서 4.0배까지 높였습니다. 최종적으로, secure two-party inference의 전체 성능을 2.7배에서 5.9배까지 향상시키고, 통신 비용을 60%까지 절감하는 성과를 기록했습니다.



### RAMIE: Retrieval-Augmented Multi-task Information Extraction with Large Language Models on Dietary Supplements (https://arxiv.org/abs/2411.15700)
- **What's New**: 본 연구에서는 RAMIE(Retrieval-Augmented Multi-task Information Extraction) 프레임워크를 개발하여 잘 정리되지 않은 임상 기록에서 다양한 정보 추출 작업을 동시에 처리할 수 있도록 하였습니다. 4가지 주요 작업인 명명된 개체 인식(NER), 관계 추출(RE), 삼중 추출(TE), 사용 분류(UC)를 통해 다목적 문제를 해결하는 방법을 제시했습니다. 본 프레임워크는 기존의 LLM(large language model)을 활용하여 보다 정확한 정보 추출을 가능하게 합니다.

- **Technical Details**: RAMIE는 MTL(multi-task learning), RAG(retrieval-augmented generation), 그리고 설명 조정(instruction fine-tuning) 기법을 결합하여 효율성을 높이고 정확한 정보 추출을 위해 최적화된 모델입니다. 각 작업은 비정형 임상 텍스트에서 구조화된 데이터를 추출하는 데 필수적이며, 효율적인 정보 추출을 통해 실제 애플리케이션에서 활용도를 높일 수 있습니다. 이 프레임워크는 다양한 도메인에서도 적용될 가능성을 가지고 있습니다.

- **Performance Highlights**: RAMIE 프레임워크를 활용하여 Llama2-13B 모델은 NER 작업에서 F1 스코어 87.39를 기록하며 3.51% 향상된 성과를 보여주었습니다. 관계 추출(RE) 작업에서 93.74(F1 스코어, 1.15% 향상)를 달성하였으며, 사용 분류(UC)에서는 MedAlpaca-7B가 93.45의 최고 점수를 기록했습니다. VC와 RAG의 조합이 전체 정확도를 크게 향상시켰음을 보여주는 실험 결과도 도출되었습니다.



### State-Space Large Audio Language Models (https://arxiv.org/abs/2411.15685)
- **What's New**: 본 논문은 Transformer 기반의 오디오 인식 모듈을 State-Space 모델로 대체하여 state-space 기반 대규모 오디오 언어 모델(LALM)을 최초로 제안하고 있습니다. 이러한 접근 방식은 오디오 입력 및 의도의 이해에서 성능을 유지하면서 컴퓨팅 자원에서의 효율성을 증가시킵니다. 실험 결과를 통해, 제안된 모델은 매개변수 수가 현저히 적음에도 불구하고 기존의 Transformer 기반 LALM과 비교할 수 있을 정도로 경쟁력 있는 성능을 발휘하고 있음을 보여줍니다.

- **Technical Details**: State-Space 모델(SSM)은 Kalman 필터 및 숨겨진 마르코프 모델과 같은 고전적인 상태 공간 모델에서 영감을 받았습니다. SSM은 입력 시퀀스와 출력 시퀀스를 연결하는 선형 미분 방정식을 사용하여 내부 상태를 구축하며, 토큰 길이에 대해 선형 복잡성을 가집니다. 이러한 특성 덕분에 SSM은 Transformer 모델보다 더 빠른 추론 시간과 낮은 메모리 요구 사항을 가지고 있습니다.

- **Performance Highlights**: 제안된 state-space 기반 LALM은 다양한 데이터셋을 대상으로 한 분류 및 캡션 검색 작업에서 평가되었습니다. 실험 결과, SSM 기반 LALM은 매개변수 수가 적음에도 불구하고 기존의 Transformer 기반 LALM과 유사한 성능을 발휘했습니다. 이는 메모리 및 시간 제약이 있는 여러 상황에서 적용 가능성을 높여줍니다.



### Quantile deep learning models for multi-step ahead time series prediction (https://arxiv.org/abs/2411.15674)
- **What's New**: 이 논문에서는 다단계 시계열 예측을 위한 새로운 양적 회귀(Quantile Regression) 딥러닝 프레임워크를 제안합니다. 이는 양적 회귀를 통합하여 딥러닝 모델의 예측 능력을 향상시키고, 예측 값에 대한 더 세분화된 이해를 제공합니다. 고변동성과 극단적인 상태에서의 성능 또한 평가하여 기존의 딥러닝 모델과 비교합니다.

- **Technical Details**: 양적 회귀 모델은 응답 변수의 조건부 중앙값을 추정하는 선형 회귀의 확장으로, 두 개의 암호화폐인 비트코인과 이더리움을 사용하여 이루어집니다. 논문에서 제안하는 프레임워크는 LSTM(Long Short-Term Memory) 네트워크와 CNN(Convolutional Neural Networks) 모델을 포함하여 극단 값 예측에 효과적인 접근 방식을 제공합니다. 다양한 벤치마크 데이터셋을 사용하여 검증되었으며, 파이썬 코드와 데이터를 오픈소스로 제공합니다.

- **Performance Highlights**: 양적 손실 함수를 딥러닝 모델에 통합함으로써, 예측 정확성의 손실 없이 선택된 분위수(quantiles)에 대한 추가 예측을 제공합니다. 기존의 딥러닝 모델에 비해 변동성을 효과적으로 처리하고 결정 및 불확실성 정량화를 위한 추가 정보를 제공합니다. 따라서 이 연구는 극단적인 예측에서의 딥러닝 모델과 양적 회귀의 결합 가능성을 시사합니다.



### IRSKG: Unified Intrusion Response System Knowledge Graph Ontology for Cyber Defens (https://arxiv.org/abs/2411.15672)
Comments:
          10 pages, 8 figures

- **What's New**: 이 논문은 가해자가 들이닥쳤을 때 즉각적으로 반응하고 복구할 수 있도록 돕는 자율 지능 사이버 방어 에이전트(Autonomous Intelligent Cyber-defense Agents, AICAs)를 집중적으로 다루고 있습니다. 특히, 침입 탐지 시스템(Intrusion Detection System, IDS) 및 침입 대응 시스템(Intrusion Response System, IRS)의 통합 구조를 통해 다수의 소스에서 데이터를 수집함으로써 해킹 공격에 대한 방어를 개선할 수 있는 Ontology인 IRSKnowledge Graph(IRSKG)를 제안하고 있습니다.

- **Technical Details**: IRSKG는 다양한 기업 시스템의 모니터링 로그와 관리자 정의 정책을 포함하는 규칙 저장소를 캡처할 수 있는 구조로 설계되었습니다. 이 온톨로지는 다양한 소스로부터 지식을 통합하고 이를 통해 AICA의 반응 및 복구 효율성을 높일 수 있습니다. IRS는 실시간으로 대응하기 위해 IDS와 기업 센서와 같은 여러 출처에서 데이터를 수집하고, 이를 기반으로 AI/ML 모델을 훈련시켜 예측을 향상시킵니다.

- **Performance Highlights**: 제안된 IRSKG는 사이버 위협 환경의 변화에 따라 동적으로 변화하는 규칙을 통합할 수 있어서 사이버 공격에 대한 적응력을 향상시키는 데 기여합니다. 실습 결과로는 그래프 신경망(Graph Neural Network, GNN) 표현 방식을 통해 IRS모델의 성능 개선을 보여주었습니다. 이 연구는 IRS 시스템을 위한 통합된 지식 그래프 온톨로지를 개발하는 첫 번째 시도로, 일반화된 구조를 통해 다양한 AI/ML 모델과 기업 시스템을 지원할 수 있게 됩니다.



### Ontology-Constrained Generation of Domain-Specific Clinical Summaries (https://arxiv.org/abs/2411.15666)
Comments:
          24th International Conference on Knowledge Engineering and Knowledge Management (EKAW 2024), November 26-28, 2024, Amsterdam, The Netherlands

- **What's New**: 이 연구는 온톨로지(ontology)를 활용하여 특정 도메인에 맞춘 요약을 생성하는 새로운 방법론을 제안합니다. 기존의 대형 언어 모델(LLM)은 텍스트 요약에 대한 잠재력을 가지고 있지만, 도메인에 대한 적합성과 비정상적 정보 생성을 줄이는 데 한계가 있습니다. 해당 연구는 의학 분야에서 전자 건강 기록(EHRs) 요약에서 효과적인 결과를 보여주고 있습니다.

- **Technical Details**: 제안된 방법은 온톨로지에 기반한 제한된 디코딩(process) 과정을 통해 요약 생성 시 비현실적인 정보 생성을 감소시키고, 더 많은 관련 정보를 포함하도록 합니다. 특히, LLM의 출력이 온톨로지에서 정의된 관계와 개념에 맞도록 제한하여 생성된 내용의 신뢰성을 높입니다. 이는 의학적 특성에 맞춘 요약을 생성하는 데 필수적입니다.

- **Performance Highlights**: MIMIC-III 데이터셋을 활용한 평가를 통해, 본 연구 방법이 임상 노트의 도메인 맞춤 요약과 환각 감소를 효과적으로 달성하는 것을 보여줍니다. 이러한 결과는 의사들이 중요한 정보에 집중할 수 있는 기반을 마련해주며, 의료 환경에서의 사용 가능성을 더욱 높입니다.



### "All that Glitters": Approaches to Evaluations with Unreliable Model and Human Annotations (https://arxiv.org/abs/2411.15634)
Comments:
          20 pages, 15 figures, 58 pages with references and appendices

- **What's New**: 이번 연구는 인간의 레이블(annotations)에서 발생하는 오류를 분석하고, 이러한 오류가 모델 평가 과정에서 정확성(accuracy), 편향(bias), 공정성(fairness)과 유용성(usefulness)에 미치는 영향을 다룹니다. 특히, K12 교육의 교실 수업 품질 자동 평가를 주제로, 두 가지 대규모 언어 모델(LLM) 아키텍처를 사용하여 인간 레이블 품질을 다양한 차원에서 평가하는 새로운 방법을 제시합니다.

- **Technical Details**: 이 연구는 교육 분야에서의 레이블 품질을 평가하기 위한 여섯 가지 차원인 일관성(concordance), 자신감(confidence), 유효성(validity), 편향(bias), 공정성(fairness), 도움됨(helpfulness) 등을 이용하여, 인간 전문가의 불확실한 레이블을 통해 얻은 데이터를 평가하고 분석합니다. 또한, 모델 사용이 인간 레이블 품질에 미치는 변화를 추정하고, 현재 데이터의 일반화 가능성 내에서 일부 LLM이 교육용 레이블의 품질을 향상시킬 수 있는 방법을 식별합니다.

- **Performance Highlights**: 연구 결과, 일반적인 평가 지표에 의존할 경우 레이블과 모델 품질이 가려질 수 있으며, 일부 모델은 '슈퍼 인간'(super-human) 성과를 달성할 수 있지만, 더 엄격한 평가 기준을 적용하면 비합리적 상관관계 및 인종 편향이 드러납니다. 이 연구는 노이즈가 많은 레이블 환경에서도 정확하고 신뢰할 수 있는 평가 정보를 제공할 수 있도록 하는 더욱 견고한 평가 기법의 필요성을 강조하며 향후 연구에 대한 제언도 포함하고 있습니다.



### How Texts Help? A Fine-grained Evaluation to Reveal the Role of Language in Vision-Language Tracking (https://arxiv.org/abs/2411.15600)
Comments:
          Preprint, Under Review

- **What's New**: 이번 논문에서는 기존의 Vision-Language Tracking (VLT) 시스템의 한계를 극복하기 위해 VLTVerse라는 첫 번째 세밀한 평가 프레임워크를 제안합니다. 이 프레임워크는 10개의 시퀀스 수준의 도전 요소와 6종의 다중 관점의 의미 정보를 통합하여, 복잡한 상황에서 VLT 트래커의 성능을 체계적으로 평가할 수 있는 공간을 제공합니다. 또한, 이를 통해 언어 정보가 VLT에서 수행하는 역할을 조명하며, 데이터, 평가 및 알고리즘 차원에서 VLT를 개선하기 위한 필수적인 지침을 제공합니다.

- **Technical Details**: VLTVerse는 기존 SOTVerse를 기반으로 하여, 짧은 기간, 긴 기간, 그리고 글로벌 인스턴스 추적 작업을 포괄하는 세밀한 평가 프레임워크를 구축합니다. 네 가지 대표적인 벤치마크, 10가지 도전 요소, 6종의 의미 정보를 포함하고 있어, VLT 트래커에 대한 포괄적인 평가가 가능합니다. 특히, 60가지 도전 요소와 의미가 결합된 체계적인 성능 평가를 통해, 기존 평가 방법으로는 포착할 수 없는 언어의 중요성에 대한 통찰을 제공합니다.

- **Performance Highlights**: VLTVerse를 통해 실시된 세밀한 평가는 다양한 도전 요소 하에서 언어 모달리티가 VLT 트래커 성능에 미치는 영향을 심층적으로 분석합니다. 연구자들은 VLTVerse의 평가 공간을 통해 성능 병목 현상을 식별하고 알고리즘 설계를 최적화할 수 있으며, VLT 작업 개념에 대한 이해를 증진시켜 향후 연구에서 트래커 성능을 향상하는 데 기여할 수 있는 귀중한 정보를 제공합니다.



### An adversarial feature learning based semantic communication method for Human 3D Reconstruction (https://arxiv.org/abs/2411.15595)
- **What's New**: 이 논문에서는 인체 3D 재구성을 위한 새로운 접근 방식으로 Adversarial Feature Learning 기반의 Semantic Communication 방법(AFLSC)을 소개합니다. 이 방법은 3D 재구성 작업에 중요한 의미 정보를 추출하고 전송하는 데 중점을 두고 있습니다. 특히, 네트워크 대역폭이 제한적이고 낮은 지연 시간이 요구되는 환경에서 데이터 흐름을 최적화하고 대역폭 압박을 완화하는 데 기여합니다.

- **Technical Details**: 발신 측에서는 멀티태스크 학습 기반의 특징 추출 방법을 제안하여 2D 인체 이미지에서 공간 배치, 키포인트, 자세 및 깊이 정보를 캡처합니다. 또한, 이러한 정보를 의미 데이터로 인코딩하는 Adversarial Feature Learning 기반의 의미 인코딩 기술을 설계하였습니다. 수신 측에서는 효과적인 다단계 의미 특징 디코딩 방법을 설계하여 의미 데이터를 다시 키 이미지 특징으로 변환합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 데이터 전송 효율성과 재구성 품질 측면에서 우수한 성능을 보임을 확인했습니다. 이 기술은 대역폭이 제한된 환경에서의 응용 가능성이 뛰어난 것을 입증하며, 인체 3D 메쉬 모델을 생성하는 improved ViT-diffusion 모델을 활용합니다. 따라서 이 연구는 다양한 분야에서 인체 3D 재구성을 위한 귀중한 기여를 할 것으로 기대됩니다.



### A Survey on LLM-as-a-Judg (https://arxiv.org/abs/2411.15594)
Comments:
          33 pages, 9 figures. arXiv admin note: text overlap with arXiv:2310.05470 by other authors

- **What's New**: 이번 논문은 LLM(as-a-Judge) 시스템을 구축하는 방법론을 제시하는 포괄적인 설문을 제공하며, 신뢰할 수 있는 평가 시스템의 개발을 위한 전략들을 탐구합니다. LLM의 다양한 데이터 타입 처리 능력을 활용하여 전통적인 전문가 평가 방식의 대안으로 자리 잡고 있으며, 평가 일관성 향상 및 편향 완화 등의 신뢰성 강화를 위한 방법을 모색하고 있습니다.

- **Technical Details**: LLM은 과거 인공지능 및 소프트웨어 공학, 사회 과학에 걸쳐 성공적으로 활용되어 왔으며, 이는 LLM을 평가자로 활용하는 'LLM-as-a-Judge' 모델의 채택 증가로 이어졌습니다. 이 프레임워크는 평가 기준의 명확한 정의와 다양한 평가 시나리오에 적합한 적응 능력을 포함하여, 평가 과정에서 발생할 수 있는 편향을 탐구하고 해결책을 제시합니다. 또한, 새로운 벤치마크를 통해 다양한 신뢰성 향상 전략의 효과성을 분석할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLM-as-a-Judge는 기존 인간 평가 방식에 비해 평가의 효율성을 향상시킬 잠재력이 있으며, LLM이 제공하는 평가의 일관성 및 비용 효율성은 혁신적입니다. LLMs는 다양한 입력 유형을 처리할 수 있는 유연성을 제공하고, 주요 평가 메트릭에서는 놓칠 수 있는 미세한 qualitative insights를 통합할 수 있는 가능성을 가지고 있습니다. 본 논문은 LLM으로서의 평가자 시스템의 신뢰성, 확장성 및 적용 가능성을 높이기 위한 미래 연구 방향에 대한 통찰을 제시합니다.



### Deep Learning for THz Channel Estimation and Beamforming Prediction via Sub-6GHz Chann (https://arxiv.org/abs/2411.15589)
Comments:
          Published: 2022 IEEE International Conference on Signal Processing and Communications (SPCOM 2022)

- **What's New**: 이 논문에서는 THz(테라헤르츠) 통신 시스템에서의 채널 추정에 대한 새로운 접근 방식을 제안합니다. 기존의 전통적인 방법들은 큰 계산 비용 때문에 실용적이지 않았지만, 본 논문에서는 CNN(합성곱 신경망)을 기반으로 한 효율적인 THz 채널 추정기를 제안하여 전송 신호의 비효율성을 극복합니다. 또한, 추정된 채널 인자를 사용하여 최적의 빔포머를 예측하는 알고리즘을 개발했습니다.

- **Technical Details**: 이 시스템은 두 개의 주파수 대역을 사용하는 네트워크 모델을 기반으로 하며, sub-6GHz 주파수에서 동작하는 송수신기와 THz 주파수에서 동작하는 송수신기로 이루어져 있습니다. sub-6GHz 채널은 상대적으로 적은 수의 파일럿 신호로 추정할 수 있으며, 이를 통해 THz 채널의 경로 손실, 도착 각도 등의 인자를 직접 추정합니다. 이 과정에서 계산 작업의 복잡성을 크게 줄일 수 있습니다.

- **Performance Highlights**: 제안된 알고리즘은 시간을 효율적으로 사용하여 근사 최적의 스펙트럼 효율성을 달성할 수 있으며, 기존의 深度 학습 기반 빔포머 예측 알고리즘보다 성능이 우수합니다. 또한, 최적의 빔포머를 즉시 선택하는 기능은 THz 채널 추정의 오버헤드를 줄여 실시간 통신의 효율성을 향상시킵니다. 이 연구는 THz 통신의 가능성을 실현하는 데 있어 중요한 기여를 할 것으로 기대됩니다.



### LAGUNA: LAnguage Guided UNsupervised Adaptation with structured spaces (https://arxiv.org/abs/2411.15557)
- **What's New**: 이번 논문에서 저자는 상대적 표현을 통한 비지도 도메인 적응(unsupervised domain adaptation, UDA)을 제안합니다. 기존의 방법은 도메인 불변 표현과 도메인 특정 특징을 보존하는 균형을 맞추기 어려웠습니다. 그들은 LAGUNA라는 새로운 방법을 소개하며 이를 통해 세분화된 관계를 구성하는 것이 가능하다고 주장합니다.

- **Technical Details**: LAGUNA는 세 단계로 구성된 방법론으로, 첫 번째 단계에서 텍스트 클래스 레이블을 사용하여 도메인 불가지론(reference space)을 생성합니다. 두 번째 단계에서는 언어 모델을 학습하여 텍스트 캡션을 참조 잠재 공간에 매핑하고, 마지막 단계에서는 크로스 도메인 분류기를 훈련합니다. 이러한 방법을 통해 각 도메인은 고유한 패턴을 유지하면서도 서로 연결됩니다.

- **Performance Highlights**: LAGUNA는 4개의 다양한 이미지 및 비디오 데이터셋에서 기존의 최첨단 기술을 초과하는 성능을 보였습니다. 예를 들어, DomainNet에서 평균 3.32%의 정확도 향상, GeoPlaces에서 5.75%, GeoImnet에서 4.77%, EgoExo4D에서는 평균 클래스 정확도에서 1.94% 향상을 기록하였습니다. 이러한 결과는 기법의 우수성을 입증합니다.



### ReWind: Understanding Long Videos with Instructed Learnable Memory (https://arxiv.org/abs/2411.15556)
- **What's New**: ReWind는 비디오 이해를 위한 혁신적인 메모리 기반 비전-언어 모델(VLM)로, 긴 비디오 처리에서 효과적인 최신 기술을 제공합니다. 이 모델은 메모리 모듈을 통해 비디오 진행 중 지시사항과 연관된 시각 정보를 저장하고 업데이트하는 'read-perceive-write' 사이클을 갖추고 있습니다. 기존 모델들이 겪는 메모리 부족과 복잡한 계산 문제를 해결할 수 있습니다.

- **Technical Details**: ReWind는 두 단계로 구성된 프레임워크로 작동하며, 첫 번째 단계에서는 동적 학습이 가능한 메모리 모듈을 사용하여 비디오에서 중요한 정보만 저장합니다. 이 메모리 모듈은 입력 토큰 수에 따라 선형적으로 메모리를 확장하는 방식으로 설계되어 있으며, 영상 처리에 필요한 고해상도 프레임을 선택함으로써 질 높은 시각 정보를 처리합니다.

- **Performance Highlights**: ReWind는 시각적 질문 응답(VQA) 및 시간 기반 안정성 작업에서 기존 방법보다 우수한 성능을 보여줍니다. MovieChat-1K VQA 데이터 세트에서 13%의 점수 향상과 Charades-STA에서 8%의 mIoU 증가를 달성하였습니다. 이러한 결과는 ReWind의 효과성을 입증하며, 긴 금융 비디오에 대한 이해도를 크게 개선합니다.



### Class Order Disorder in Wikidata and First Fixes (https://arxiv.org/abs/2411.15550)
- **What's New**: 이 논문에서는 Wikidata의 온톨로지에서 클래스 순서(class order)와 관련된 문제를 분석합니다. 기존 클래스 순서의 위반 및 의심스러운 정보를 SPARQL 쿼리를 통해 평가하고, 이러한 문제를 해결하기 위한 제안을 제시합니다. 기사에서는 수정된 쿼리를 통해 클래스 순서와 관련된 문제를 파악합니다. 이 문제들은 장기적으로 축적되고 있으며, 커뮤니티 참여와 도구 기능 개선을 통해 해결할 수 있습니다.

- **Technical Details**: Wikidata는 2024년 7월 기준으로 1억 개 이상의 아이템을 포함하는 일반 지식 기반으로, 이는 다양한 도메인(예: 생명 과학, 의학)에 걸쳐 있습니다. 논문에서는 Wikidata의 클래스가 고정 순서를 가져야 한다고 기본적으로 정의하고, 클래스가 여러 고정 순서를 가져서는 안 된다는 점을 강조합니다. 쿼리를 통해 발견된 문제들은 실제 오류일 수도, 단순히 강한 증거를 제공할 수도 있으며, 대부분 정확한 수정을 찾기 위해 추가 분석이 필요합니다.

- **Performance Highlights**: Wikidata의 오류는 해소될 기회가 많지만, 쿼리 성능에는 여러 가지 문제점이 있습니다. QLever SPARQL 엔진이 복잡한 쿼리에서 매우 빠르지만, 메모리 할당 오류가 발생할 수 있으며, 적절한 쿼리 구조가 필요합니다. 이러한 성능적 특성은 문제가 발생하기 전보다 깊이 있는 분석을 가능하게 합니다. 또한, 문서에서는 구체적인 사례에서 수정의 영향을 따져 보는 내용을 다룹니다.



### An unconditional distribution learning advantage with shallow quantum circuits (https://arxiv.org/abs/2411.15548)
Comments:
          7 + 5 pages, 2 figures

- **What's New**: 이번 연구는 실용적인 응용을 위한 근접 양자 회로에서 양자 우위를 검증하는 중요한 성과를 보여줍니다. 특히, 아쉬운 점 없이(provably) 양자 우위를 보이는 전분포 학습(PAC distribution learning) 프레임워크 내에서 얕은 양자 회로 가설을 사용하여 의미 있는 문제를 다룹니다. 이러한 접근법은 양자 회로가 제공할 수 있는 장점을 탐색하는 데 중요한 단서가 됩니다.

- **Technical Details**: 연구자는 상수 깊이 양자 회로(QNC^0)와 상수 깊이 고정 팬인 고전 회로(NC^0)를 비교하여, 양자 회로가 유리한 성질을 가질 수 있음을 입증했습니다. 특히, 비국소 상관관계를 통해 양자 우위를 끌어낼 수 있는 문제를 정의하여, 양자 회로가 고전 회로보다 물리적 제약이 적다는 점을 강조합니다. 이 작업은 Bene Watts와 Parham의 최근 연구 결과를 바탕으로 하여, 표면적으로는 간단하지만 실질적으로는 복잡한 문제로 발전시킵니다.

- **Performance Highlights**: 얕은 양자 회로가 고전적 회로에 비해 뛰어난 성능을 보여준 결과는 양자 컴퓨팅의 실용성에 대한 새로운 가능성을 열어줍니다. 이러한 결과는 양자 전산학과 기계 학습 분야 모두에 큰 영향을 미칠 수 있으며, 실제 응용에 있어 양자 기계의 이점을 입증하는 중요한 이정표가 될 것입니다. 연구의 효과는 근본적으로 양자 회로가 무작위 데이터 샘플링 및 고차원 학습 문제를 해결하는 데 유리하다는 점에서 두드러집니다.



### Optical-Flow Guided Prompt Optimization for Coherent Video Generation (https://arxiv.org/abs/2411.15540)
Comments:
          project page: this https URL

- **What's New**: 본 논문에서는 MotionPrompt라는 새로운 비디오 생성 시스템을 제안합니다. 이 시스템은 optical flow를 활용하여 비디오 생성 과정에 가이드를 제공함으로써, 생성된 비디오의 시간적 일관성을 향상시킵니다. 또한, 학습 가능한 토큰 임베딩을 최적화하여 복잡한 계산 부담을 줄이고, 자연스러운 모션 다이나믹스를 반영하는 시각적으로 일관된 비디오를 생성합니다.

- **Technical Details**: MotionPrompt는 두 개의 랜덤 프레임 쌍 사이의 optical flow를 구분하는 판별기를 훈련하여 비디오 생성에 있어 시간적 일관성을 유지합니다. 프롬프트를 통해 존재하는 전체 비디오에 영향을 미칠 수 있도록 설정되어 있어, 기존의 프레임 단위 계산 부담을 줄이면서도 학습 가능한 토큰의 임베딩을 최적화합니다. 이를 통해 생성된 프레임에 대한 상대 모션의 현실성을 평가하여 모션 패턴을 수정하게 됩니다.

- **Performance Highlights**: MotionPrompt의 성능은 다양한 비디오 생성 모델에서 검증되었습니다. 이 접근 방식을 통해 생성된 비디오 시퀀스는 자연스러운 모션을 유지하면서도 높은 시각적 품질을 확보합니다. 이전 방법들과 달리 MotionPrompt는 기존의 확산 모델의 재훈련 없이 시간 일관성을 보장하면서도, 이미 현실적인 비디오에 가까운 샘플의 품질에 미치는 영향을 최소화합니다.



### Large Language Model with Region-guided Referring and Grounding for CT Report Generation (https://arxiv.org/abs/2411.15539)
Comments:
          10 pages

- **What's New**: 이번 연구는 CT 보고서 생성을 위한 새로운 접근 방식을 제안합니다. 기존의 방법들이 전체 볼륨의 글로벌 특징(Global Features)만을 고려한 반면, Reg2RG는 특정 해부학적 영역에 초점을 맞추어 진단 성능을 향상시킵니다. 이를 통해 세밀한 지역 분석 및 진단에 필요한 정보 제공을 더 잘 수행할 수 있습니다.

- **Technical Details**: 이 연구는 지역 특징을 확보하기 위해 보편적인 세분화 모듈로부터 마스크를 활용합니다. Local Feature Decoupling (LFD) 전략을 통해 고해상도의 세부 정보를 유지하고 전역 특징(Global Features)과 통합하여 상호 지역 관계를 포착합니다. 또한, Region-Report Alignment (RRA) 훈련 전략을 도입하여 보고서의 특정 지역과 명확하게 연결하도록 모델을 훈련시킵니다.

- **Performance Highlights**: 대규모의 3D 흉부 CT 데이터셋을 이용한 실험에서 Reg2RG는 기존의 여러 방법들보다 더 우수한 성능을 기록했습니다. 이 논문에서 제안하는 접근법은 자연어 생성 및 임상 효능 지표 모두에서 더욱 뛰어난 결과를 보여주며, 보고서를 해석하는 데 있어 뛰어난 해석 가능성을 유지합니다.



### Enhancing Grammatical Error Detection using BERT with Cleaned Lang-8 Datas (https://arxiv.org/abs/2411.15523)
Comments:
          10 pages, 6 tables, 20 references

- **What's New**: 이 논문은 문법 오류 탐지(Grammatical Error Detection, GED)를 위한 개선된 LLM 기반 모델을 제안합니다. 전통적인 GED 접근 방식은 수작업으로 설계된 특징(feature)을 사용했으나, 최근에는 신경망(Neural Networks, NN)이 이러한 특징을 자동으로 발견하여 성능을 향상시키고 있습니다. BERT-base-uncased 모델은 학습 데이터에서 98.49%의 정확도와 0.91의 F1 점수를 기록하며, 데이터 정제의 중요성을 보여주었습니다.

- **Technical Details**: 이 연구는 Lang-8 데이터셋의 품질을 개선하기 위해 이를 철저히 정제하였으며, 여러 Transformer 기반 모델인 BERT와 RoBERTa를 비교했습니다. 이전의 모델과 비교해 BERT와 RoBERTa의 정제된 데이터셋에서의 성능 향상을 입증했습니다. 또한 GPT-4와 Llama-3-70B-instruct 같은 생성 모델을 활용한 평가를 통해, fine-tuning 없이도 문법 오류 탐지 작업에서 효과를 보여주었습니다.

- **Performance Highlights**: 이 연구의 실험 결과, BERT-base-uncased 모델이 가장 높은 성능을 보였으며, 기존 모델보다 월등한 결과를 나타냈습니다. 특히, 대규모 모델인 BERT-large-uncased와 RoBERTa-large가 성능 향상에 유의미하지 않음을 강조하여, 가장 큰 모델이 항상 가장 좋은 성능을 내는 것은 아님을 보여줍니다. 데이터 정제와 간단한 Transformer 기반 모델들이 GED 품질 향상에 매우 크게 기여할 수 있음을 증명했습니다.



### Interactive Visual Assessment for Text-to-Image Generation Models (https://arxiv.org/abs/2411.15509)
Comments:
          Under Review

- **What's New**: 이 논문에서는 DyEval이라는 새로운 동적 인터랙티브 시각 평가 프레임워크를 소개합니다. 이 프레임워크는 LLM(대형 언어 모델)을 활용하여 모델 피드백에 따라 테스트 입력을 적응적으로 생성할 수 있습니다. 기존의 정적 접근 방식의 한계를 극복하여 모델의 복잡한 실패를 효과적으로 식별하고 분석할 수 있도록 돕습니다. 또한 DyEval은 사용자와 생성 모델 간의 협업 평가를 촉진하여 더욱 종합적인 평가를 가능하게 합니다.

- **Technical Details**: DyEval은 LLM을 통해 생성된 계층적이고 세분화된 텍스트 입력을 사용하여 시각적인 모델 동작을 탐색하는 직관적인 사용자 인터페이스를 제공합니다. 평가자는 이 인터페이스를 통해 모델 출력을 평가하고, 평가 결과에 따라 새로운 테스트 주제로 나무(tree) 구조를 깊이 탐색합니다. DyEval은 또한 실패 유발 요소를 분석하는 맥락적 반영 모듈을 통해 모델의 잠재적 실패 패턴도 분석합니다. 이를 통해 사용자에게 더 나은 모델 개선을 위한 해석 가능한 분석을 제공합니다.

- **Performance Highlights**: DyEval은 기존 방법보다 최대 2.56배 더 많은 생성 실패 케이스를 효과적으로 식별할 수 있음을 보여주는 정량적 실험 결과를 제공합니다. 다양한 최신 텍스트-이미지 모델을 평가한 결과, 정적 테스트 방식에서는 발견할 수 없는 복잡하고 희귀한 실패 패턴이 드러났습니다. DyEval은 특히 문화적 맥락과 같은 언어적 세부 사항에서 발생하는 특정 실패를 식별했으며, 이러한 결과는 텍스트-이미지 생성 모델 개선을 위한 귀중한 통찰력을 제공합니다.



### Instruct or Interact? Exploring and Eliciting LLMs' Capability in Code Snippet Adaptation Through Prompt Engineering (https://arxiv.org/abs/2411.15501)
Comments:
          12 pages, 10 figures, accepted by ICSE 2025

- **What's New**: 본 연구는 대형 언어 모델(LLM)이 코드 스니펫 적응 작업에서 어떻게 성능을 발휘하는지를 평가한 최초의 연구입니다. 결과에 따르면, LLM의 적응 능력은 생성 작업에 비해 약 15% 감소했으며, 문맥 관련 오류가 많아 문제를 제기합니다. 이에 따라, 코드를 정확하게 적응시키기 위한 방법론적 지원이 필요하다는 점이 강조됩니다.

- **Technical Details**: 연구는 세 가지 주요 LLM의 적응 성능을 평가하였으며, 각 LLM의 성능은 ClassEval 벤치마크를 기반으로 검토되었습니다. LLM의 적응 실패 원인은 'Unclear Requirement', 'Requirement Misalignment', 'Context Misapplication'으로 분류되며, 이는 LLM들이 문맥을 이해하는데 어려움을 겪는다는 것을 보여줍니다. 이후, 상호작용 프롬프트(Interactive Prompting) 접근 방식을 통해 문제 해결 방안을 제안하였습니다.

- **Performance Highlights**: 인간-LMM 상호작용 방식은 202개의 결함 중 159개를 해결하는 데 성공했으며, 적응 성능이 평균 41.4% 향상되었습니다. 다중 에이전트 상호작용이 도입된 결과, 인간의 개입 없이도 유사한 성능 한국이 가능함을 보여줍니다. 이는 소프트웨어 재사용 및 적응 작업에서 LLM의 활용 가능성을 더욱 확대할 수 있는 기반이 됩니다.



### Automatic Evaluation for Text-to-image Generation: Task-decomposed Framework, Distilled Training, and Meta-evaluation Benchmark (https://arxiv.org/abs/2411.15488)
- **What's New**: 이 논문은 텍스트-이미지 생성 모델의 평가 방법을 개선하기 위해, GPT-4o 기반의 작업 분해 평가 프레임워크를 제안합니다. 이를 통해 복잡한 평가 작업을 단순한 서브 작업으로 나누어 처리함으로써 자동으로 새로운 학습 데이터셋을 구성하게 됩니다. 이 프레임워크는 7B 개방형 MLLM, MiniCPM-V-2.6의 효율적인 자동 평가 모델로의 변환을 촉진합니다.

- **Technical Details**: 제안된 프레임워크는 입력 텍스트 프롬프트에서 сущности(entity)와 내재적 속성(intrinsic properties), 그리고 관계 속성(relational attributes)을 추출하기 위해 GPT-4o를 사용합니다. 이러한 정보를 바탕으로 세 가지 차원(시각적 외형, 내재적 특성, 관계 속성)을 통해 평가 질문을 구성하고, 이를 통해 이미지와 캡션 간의 품질 점수를 산출합니다. 또한, 개별 평가 차원에 대해 예측된 결과를 통합하여 종합적인 판단을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 MLLM은 기존의 최첨단 모델인 GPT-4o-베이스 기준 및 VIEScore와 비교하여 Spearman 및 Kendall의 상관 관계에서 4.6% 이상의 개선을 달성했습니다. 또한, 새롭게 수동으로 주석이 달린 메타 평가 벤치마크를 통해 기존의 평가 방법들과 비교하여 더욱 신뢰성 있는 평가를 제공합니다. 이 모델은 고품질의 생성된 이미지를 평가하는 데 더욱 효과적임을 입증했습니다.



### Towards Robust Evaluation of Unlearning in LLMs via Data Transformations (https://arxiv.org/abs/2411.15477)
Comments:
          Accepted at EMNLP 2024 Findings; 21 pages (5 page main content + references + appendix)

- **What's New**: 이번 연구에서는 기존의 Machine Unlearning (MUL) 기술의 강인성을 살펴보며, LLMs가 특정 정보를 잊도록 강제하는 방법에 대해 다룹니다. 연구 팀은 TOFU 데이터셋을 사용하여 다양한 데이터 포맷에서 잊어버리기 성능을 측정하는 중요성을 강조합니다. 특히, 서로 다른 포맷으로 입력이 제공되었을 때 잊어진 정보를 다시 회상할 수 있는지에 대한 실험을 진행하였습니다.

- **Technical Details**: LLMs는 대량의 텍스트 데이터로 훈련되며, 이 과정에서 개인 식별 정보(PII)를 포함한 원하지 않는 정보가 포함될 수 있습니다. 따라서 MUL 기법이 LLM에서 어떤 영향을 미치는지를 조사하며, 잊어버리기 성능이 다양한 데이터 포맷에 따라 어떻게 달라지는지를 탐구합니다. 연구 방법으로는 TOFU 벤치마크를 확장하는 방식으로, 같은 정보가 표현되는 다양한 형식을 도입하고, 이에 대한 평가 지표를 개발합니다.

- **Performance Highlights**: 연구 결과는 서로 다른 데이터 포맷에서 목표 모델(target model)과 잊은 모델(unlearned model) 사이의 성능 격차를 보여줍니다. 이를 통해 다양한 포맷을 고려하는 것이 MUL 알고리즘의 평가에서 신뢰성과 강인함을 확보하는 데 필요하다는 점을 강조합니다. 또한, 이번 연구에서 발표된 새로운 데이터셋과 평가 메트릭스는 MUL 분야의 발전을 도울 것으로 기대합니다.



### KinMo: Kinematic-aware Human Motion Understanding and Generation (https://arxiv.org/abs/2411.15472)
- **What's New**: 텍스트 기반의 인간 동작 제어는 컴퓨터 비전에서 중요한 도전 과제로, 본 연구에서는 이를 해결하기 위한 새로운 모션 표현 방식을 제안합니다. 기존의 전통적인 접근 방식은 전체 액션에 초점을 맞춰 섬세한 움직임을 포착하지 못하는 한계를 가지고 있습니다. 따라서 본 논문은 움직임을 신체 관절 그룹의 움직임으로 분해하여 다루고, 이를 통해 텍스트와 모션 도메인 간의 간극을 줄이는 방법을 제안합니다.

- **Technical Details**: 우리는 인간 동작의 세 가지 수준: 글로벌 액션, 로컬 관절 그룹, 관절 상호작용으로 정의하는 KinMo라는 새로운 프레임워크를 도입합니다. 이 프레임워크는 자동 데이터 수집 파이프라인을 통해 기존의 텍스트-모션 벤치마크를 개선하여 정밀한 로컬 관절 그룹 모션과 상호작용 설명을 포함합니다. 또한, 우리가 제안한 계층적 모션 의미론 접근 방식을 통해 관절 레벨의 상호작용 정보를 글로벌 액션 레벨의 의미론으로 점진적으로 융합합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 텍스트-모션 검색 성능을 향상시키고, 보다 정확한 관절 움직임 생성을 가능하게 함을 보여주었습니다. 모션 생성 과정은 코스-투-파인(coarse-to-fine) 방식으로 구성되어 다양한 생성 및 편집 응용 분야에서의 적용 가능성을 높이고 있습니다. 이를 통해 언어와 모션 간의 더 나은 일치를 유도할 수 있는 기반을 마련하였습니다.



### A Preliminary Study of Multilingual Code Language Models for Code Generation Task Using Translated Benchmarks (https://arxiv.org/abs/2411.15470)
Comments:
          5 pages, ASEW 2024

- **What's New**: 이번 연구는 코드 언어 모델(Code Language Models, CLM)의 성능 평가에서 다국어 및 낮은 리소스 프로그래밍 언어 환경에 대한 도전 과제를 논의합니다. 특히, 다양한 프로그래밍 언어에 대한 고품질 벤치마크 부족과 CLM 훈련 데이터의 불균형 문제를 강조합니다. 이 연구에서는 코드 생성(code generation) 작업을 위한 Poly-Coder라는 개방형 멀티언어 CLM의 성능을 평가했습니다.

- **Technical Details**: 우리는 두 가지 최첨단 번역을 활용하여 인기 있는 코드 생성 벤치마크인 HumanEval을 평가했습니다. 이 과정은 OctoPack 및 MultiPL-E 연구에 의해 촉진되었습니다. 번역된 벤치마크의 성과는 훈련 단계에서 사용된 평가 지표인 perplexity와 잘 일치하여 그 유효성을 검증하는 데 성공했습니다.

- **Performance Highlights**: 그러나 번역된 벤치마크 전반에 걸쳐 CLM의 성능에서 여러 불일치가 발견되었습니다. 결과를 재현하는 데 어려움이 있었던 점도 주목할 만합니다. 이러한 초기 통찰력은 번역된 벤치마크의 방법론적 접근, 한계 및 재현 가능성을 이해하기 위한 보다 포괄적인 실증 연구의 필요성을 강조합니다.



### TANGNN: a Concise, Scalable and Effective Graph Neural Networks with Top-m Attention Mechanism for Graph Representation Learning (https://arxiv.org/abs/2411.15458)
Comments:
          The code and ArXivNet dataset are available at this https URL

- **What's New**: 본 연구는 Top-m attention mechanism과 neighborhood aggregation component를 통합한 새로운 Graph Neural Network(GNN) 아키텍처인 TANGNN을 제안합니다. 이는 기존의 GNN이 갖는 한정된 receptive field 문제를 해결하면서도 과도한 계산 오버헤드를 발생시키지 않는 방법을 제시합니다. 또한, TANGNN은 citation sentiment prediction이라는 혁신적인 작업에 적용되어, 새로운 아카이브 네트워크인 ArXivNet을 구축하였습니다.

- **Technical Details**: TANGNN은 각 계층에서 지역 및 확장된 이웃으로부터 관련 정보를 효과적으로 집계하기 위해 Top-m attention mechanism과 neighborhood aggregation component를 통합한 모델입니다. Top-m attention mechanism에서 보조 벡터인 a를 도입하여 노드 간의 코사인 유사도 기반으로 가장 관련성 높은 노드 피처를 선택합니다. 이는 처리 효율성과 모델 성능을 향상시키는 데 기여하며, 샘플링 전략을 활용하여 메모리와 계산 요구 사항을 대폭 줄입니다.

- **Performance Highlights**: TANGNN은 다양한 실험을 통해 vertex classification, link prediction, sentiment prediction, graph regression, 그리고 visualization 과제에서 우수한 성능을 입증하였습니다. 특히, GNN 분야에서 최초로 시도되는 citation sentiment prediction 작업에서도 뛰어난 결과를 보였습니다. 이 모델은 여러 데이터셋에서 기존 방법들보다 효과성 면에서 뛰어난 성능을 발휘하며, 큰 규모의 그래프 데이터에서도 효율성을 유지합니다.



### Hindi audio-video-Deepfake (HAV-DF): A Hindi language-based Audio-video Deepfake Datas (https://arxiv.org/abs/2411.15457)
- **What's New**: 이번 논문에서는 힌디 언어에 특화된 첫 번째 딥페이크 데이터셋인 ``Hindi audio-video-Deepfake'' (HAV-DF)를 제안합니다. 기존 데이터셋들이 주로 영어 기반이라는 점에서, HAV-DF는 다양한 힌디 음성과 얼굴 표정을 포착하여 딥페이크 감지 모델의 훈련과 평가에 기초가 될 수 있습니다. 특히 이 데이터셋은 딥페이크 비디오 및 오디오 데이터셋 모두에 대해 감지기를 훈련하는 데 유용합니다.

- **Technical Details**: HAV-DF 데이터셋은 faceswap, lipsyn, voice cloning 기법을 사용하여 생성되었습니다. 이 과정에서 다양한 조작 기법을 통해 힌디 언어의 미묘한 뉘앙스를 포착하고, 생성된 데이터의 복잡성도 다양한 수준에 걸쳐있습니다. 이러한 데이터셋은 딥페이크 탐지 시스템의 평가 및 개선에 필수적이며, 새로운 감지 모델 개발에 기여할 수 있습니다.

- **Performance Highlights**: HAV-DF 데이터셋은 FF-DF 및 DFDC와 같은 기존 데이터셋과 비교해 감지 정확도가 낮다는 결과를 보여줍니다. 이는 HAV-DF 데이터셋이 힌디 언어 콘텐츠와 다양한 조작 기술에 초점을 둔 덕분으로, 탐지가 더욱 어려운 도전을 제시합니다. 이는 다국어 딥페이크 탐지 시스템 개발에 중요한 기초를 제공합니다.



### MUFM: A Mamba-Enhanced Feedback Model for Micro Video Popularity Prediction (https://arxiv.org/abs/2411.15455)
Comments:
          14 pages,9 figures

- **What's New**: 최근 마이크로 비디오의 인기가 급증하면서, 플랫폼 추천의 힘이 사용자 피드백에 의해서 얼마나 영향을 받는지가 주요 관심사로 떠오르고 있습니다. 본 논문에서는 사용자 피드백과 동적 이벤트 상호작용의 장기 의존성을 포착하기 위해 Mamba Hawkes 프로세스를 기반으로 한 프레임워크를 소개합니다. 실험 결과, 제안한 MUFM 모델이 기존의 최신 방법론보다 23.2% 성능이 향상되었습니다.

- **Technical Details**: Mamba-Enhanced User Feedback Capture Model for Micro Video Popularity Prediction(MUFM)은 마이크로 비디오와 관련된 다양한 데이터를 활용하여 인기도를 예측하도록 설계되었습니다. 이 모델은 사용자 반응을 이해하기 위해 Mamba Hawkes 프로세스를 사용하며, 19,000개의 댓글 데이터로 학습하였습니다. 또한, 크로스 어텐션 메커니즘을 적용하여 타겟 비디오와 유사한 콘텐츠 간의 연결성을 감지하고, 예측 성능을 더욱 향상시켰습니다.

- **Performance Highlights**: MUFM 모델은 Microlens-100k 데이터셋을 평가하여 기존 최첨단 방법론보다 23.2% 더 우수한 성능을 보였습니다. 이를 통해 사용자 피드백 행동 시퀀스 간의 관계를 매핑할 수 있는 모델의 가능성이, 차세대 추천 알고리즘 및 플랫폼 활용에 기여할 수 있음을 입증하였습니다. 또한, 마이크로 비디오 전파의 사회적 영향에 대한 이해를 향상시키는 데에도 중요한 역할을 할 것으로 기대됩니다.



### Enhancing Instruction-Following Capability of Visual-Language Models by Reducing Image Redundancy (https://arxiv.org/abs/2411.15453)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 instruction-following 능력을 향상시키기 위해 Visual-Modality Token Compression (VMTC) 및 Cross-Modality Attention Inhibition (CMAI)라는 전략을 제안합니다. MLLMs의 instructional 수행 능력은 기존의 Large Language Models (LLMs)보다 부족하다는 점을 강조하며, 시각적 토큰의 다운샘플링이 이 능력을 향상시킬 수 있다는 것을 발견했습니다. 그러나 간단한 다운샘플링 방법이 MLLMs의 멀티모달 이해 능력을 저해하는 문제도 존재합니다.

- **Technical Details**: VMTC는 시각적 데이터의 불필요한 정보를 압축하면서도 중요한 전경 정보를 유지하는 모듈로, ViT 레이어에서 주의 점수를 활용하여 불필요한 배경 토큰을 식별하고 클러스터링 후 결합합니다. CMAI는 LLM의 텍스트 토큰이 관련 이미지 토큰에만 집중하도록 도와주는 모듈로, 낮은 텍스트-이미지 집중 점수를 가진 텍스트-이미지 토큰 쌍 간의 주의를 약화하도록 설계되었습니다. 이 두 모듈은 MLLMs의 instruction-following 능력을 개선하면서 멀티모달 이해 능력을 정확히 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, VMTC 및 CMAI를 통합한 방법이 MLLMs의 instruction-following 능력을 크게 향상시켰으며, 여러 벤치마크에서 SOTA 성능을 달성했습니다. 이 연구는 MLLMs의 instruction-following 능력과 시각적 모드의 정보 중복 간의 상관관계를 탐구한 최초의 연구로, 두 가지 문제를 해결하는 새로운 기법을 제시합니다. 제안된 접근법은 MLLMs의 기존 성능을 유지하면서 instruction-following 능력을 크게 개선할 수 있음을 보여줍니다.



### freePruner: A Training-free Approach for Large Multimodal Model Acceleration (https://arxiv.org/abs/2411.15446)
- **What's New**: 본 논문에서는 기존의 retraining 없이 어떤 오픈소스 LMM에도 바로 적용할 수 있는 training-free token reduction 방법인 freePruner를 제안합니다. 기존 방법들은 token merging 작업에 크게 의존하는 반면, freePruner는 두 단계의 token selection 전략을 통해 고수준 의미 정보와 저수준 시각 정보를 효과적으로 포착합니다. 이를 통해, LMM의 성능을 유지하면서 최대 2배의 속도 향상을 이루었습니다.

- **Technical Details**: freePruner의 핵심은 Pivotal Token Selection과 Complementary Token Selection 두 가지 전략입니다. 첫 번째 단계에서는 설계된 기여도 지표를 이용해 고수준 의미 정보를 담고 있는 주요 토큰을 추출합니다. 두 번째 단계에서는 주어진 시각 정보와 관련된 추가 토큰을 선택하여 저수준의 시각적 세부정보를 유지하는 방식입니다. 이는 기존 방법들과 달리 모델 retraining 없이 진행되어, 훨씬 더 실용적인 접근 방식이라 할 수 있습니다.

- **Performance Highlights**: 실험 결과, freePruner는 기존의 여러 LMM 벤치마크에서 비교 가능한 성능을 유지하면서도, 약 2배의 추론 속도 향상을 달성했습니다. 또한, freePruner는 post-training quantization과 같은 다른 후처리 가속기법과도 독립적으로 결합하여 사용할 수 있는 가능성을 보여주며, 효율적인 LMM 배포를 위한 실용적인 솔루션이 될 것입니다.



### Automatic High-quality Verilog Assertion Generation through Subtask-Focused Fine-Tuned LLMs and Iterative Prompting (https://arxiv.org/abs/2411.15442)
- **What's New**: 이번 연구에서는 SystemVerilog Assertions (SVA)을 자동으로 생성하는 대규모 언어 모델(LLM) 기반의 플로우인 \\ToolName을 제안합니다. 새로운 서브 작업 중심의 미세 조정 방법을 도입하여, 기존 LLM들이 생산하는 기능적으로 잘못된 주장을 효과적으로 제거했습니다. 이러한 접근은 기능적으로 올바른 주장의 수를 7.3배 증가시켰으며, 문법 오류가 없는 주장은 26% 증가했습니다.

- **Technical Details**: SVA는 설계 명세에서 기대하는 기능성을 체크하기 위해 Property Specification Language (PSL)로 작성됩니다. 그러나 각 설계는 별도의 사양을 가지고 있으며, 무한한 구현 방법이 존재하므로 공학자들은 명세를 읽고 독립적인 주장을 생성하는 데 많은 시간을 소비해야 합니다. 본 논문에서는 PLLM(Pretrained LLM)과 독립적으로 미세 조정 가능한 서브 작업을 나누는 새로운 방법론을 도입하여 이러한 과정을 간소화했습니다. 또한, 맞춤형 컴파일러를 통해 의미 있는 오류 메시지를 생성함으로써 LLM의 정확성을 높이는 반복 수정 방법도 개발하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 기존 모델 대비 기능적으로 올바른 주장의 수를 7.3배 증가시켰으며, 여러 설계에 대한 자극 커버리(stimuli coverage)는 100%에 가까운 결과를 나타냈습니다. 이 논문은 설계 명세에서 RTL 주장을 생성하기 위한 완전 자동화된 도구 세트를 출시하였으며, 고품질 주장 생성을 위한 평가 데이터셋도 포함하여 생성된 주장의 질을 평가할 수 있는 강력한 기준을 설립했습니다.



### GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning (https://arxiv.org/abs/2411.15428)
Comments:
          25 pages, 5 figures

- **What's New**: 본 연구는 지리적 상호작용을 고려하여 Node 속성과 Edge 기반 상호작용을 동시에 분석하기 위해 GeoAI 강화 비지도 커뮤니티 탐지 방법인 region2vec를 제안합니다. 이 방법은 Graph Attention Networks (GAT)와 Graph Convolutional Networks (GCN)를 기반으로 하며, 지역 단위 분석 시 매우 효과적인 성능을 보입니다. 특히, 이 신방법은 공공 보건의 부족 지역 구분 문제에 적용되어 그 가능성을 입증합니다.

- **Technical Details**: region2vec 방법은 속성 유사성, 지리적 인접성 및 공간 상호작용을 기반으로 Node 신경 임베딩을 생성하며, 이를 통해 Agglomerative Clustering를 이용하여 네트워크 커뮤니티를 추출합니다. 기존의 커뮤니티 탐지 알고리즘은 주로 Topological 정보에 기반하여 네트워크를 분할하는 반면, 본 연구에서는 Node 속성을 보다 잘 통합할 수 있는 접근법을 제시합니다. 이 연구는 GCN과 GAT 모델을 확장하여 커뮤니티 탐지 지향 손실에 의해 유도되는 학습 방법을 개발했습니다.

- **Performance Highlights**: 제안된 GeoAI 기반 방법은 여러 기준선과 비교했을 때 뛰어난 성능을 보이며, Node 속성 유사성과 공간 상호작용의 강도를 동시에 극대화하는 데 가장 효과적입니다. 이 연구는 복잡한 구조를 가진 공간 네트워크에서 지역화 문제 해결에 대한 새로운 통찰을 제공합니다. 최종적으로, 본 연구의 방법론은 사회, 경제 등 다수의 분야에서 실제 데이터 분석 및 정책 결정에 기여할 수 있습니다.



### Learning a local trading strategy: deep reinforcement learning for grid-scale renewable energy integration (https://arxiv.org/abs/2411.15422)
Comments:
          Accepted to HICSS58

- **What's New**: 이 연구는 태양광 발전과 함께 배터리를 운영하기 위한 강화 학습(reinforcement learning, RL)의 사용을 탐구합니다. RL은 평균적으로 61% (최대 96%)의 이론적 최적 운영 성과를 달성하였으며, 기존의 고급 제어 방법들을 능가했습니다. 특히, 향후 신호 예측이 어려운 경우 RL의 사용이 바람직할 수 있음을 제안합니다.

- **Technical Details**: 이 논문은 그리드에 배치된 배터리 에너지 저장 시스템(BESS)의 수학적 문제를 공식화합니다. 배터리의 주요 파라미터는 에너지 용량(E_max), 충전 및 방전 효율(η), 최대 충전/방전 전력(P_max)을 포함합니다. 연구에서는 충전 속도(c_t)와 방전 속도(d_t)의 동적 변화와 배터리가 그리드 또는 태양광으로부터 충전하는 전력도 정의하여 명확한 수학적 식을 도출하였습니다.

- **Performance Highlights**: 연구 결과, RL 접근 방식은 에너지 수요와 공급 불일치를 줄이기 위해 유연한 배터리 운영을 가능하게 하며, 다양한 장소와 계절 간의 액션의 다양성을 고려하여 안정성을 지원합니다. RL 방법의 견고성을 분석하고, 에너지 저장 시스템 관리에서의 적합한 정책을 제시하여 운영 효율성을 증가시킬 수 있는 가능성을 보여줍니다.



### FG-CXR: A Radiologist-Aligned Gaze Dataset for Enhancing Interpretability in Chest X-Ray Report Generation (https://arxiv.org/abs/2411.15413)
Comments:
          ACCV 2024

- **What's New**: 해당 연구는 방사선학적 진단을 돕기 위해 설명 가능한 시스템인 Gen-XAI를 제안하였습니다. 이 시스템은 방사선 전문의의 시선 주의정보(eye gaze)를 기반으로 CXR(Chest X-ray) 분석을 통해 보고서를 생성합니다. 또한, Fine-Grained CXR (FG-CXR)라는 새로운 데이터셋을 소개하여 방사선 전문의의 시선 추적 정보와 진단 내용을 보다 정밀하게 정렬했습니다.

- **Technical Details**: Gen-XAI는 방사선 전문의의 시선을 예측하는 Gaze Attention Predictor와 이를 기반으로 보고서를 생성하는 Report Generator로 구성되어 있습니다. Gaze Attention Predictor는 전문의의 주의력을 나타내는 데이터로부터 관심 있는 영역을 학습하여, 보고 생성 과정에서 이를 활용합니다. FG-CXR 데이터셋은 방사선 전문의가 주의하는 7개의 해부학적 영역에 대한 세부정보를 포함하고 있으며, 이는 기존 데이터셋에서의 미묘한 불일치를 해결하여 정확한 진단을 지원합니다.

- **Performance Highlights**: 연구에서는 Gen-XAI의 효율성을 입증하기 위해 광범위한 실험을 수행하였습니다. FG-CXR 데이터셋을 활용하여 방사선 전문의의 진단 과정과 주의 패턴을 근거로 한 보고서 생성을 성공적으로 수행했습니다. 이로 인해 방사선 전문의의 진단 정보와 시스템의 출력 간의 상관관계가 향상되어, 더 높은 수준의 해석 가능성과 신뢰성을 제공합니다.



### Exploring Large Language Models for Multimodal Sentiment Analysis: Challenges, Benchmarks, and Future Directions (https://arxiv.org/abs/2411.15408)
- **What's New**: 이번 연구에서는 Multimodal Aspect-Based Sentiment Analysis (MABSA) 작업에 대한 대형 언어 모델 (LLMs)의 적합성을 탐구합니다. MABSA는 텍스트와 이미지를 포함한 다중 모달 정보에서 다양한 측면 용어와 그에 따른 감정 폴라리티를 추출하는 것을 목표로 합니다. 우리는 LLM을 활용한 새로운 평가 벤치마크를 설계하였으며, 기존의 지도 학습 방법과 비교하여 그 성능을 평가하였습니다.

- **Technical Details**: MABSA에서 LLM의 활용을 위해 LLM For Sentiment Analysis (LLM4SA) 프레임워크를 설계하였습니다. 이 프레임워크는 텍스트와 시각적 기능을 공동으로 처리하며, 잘 확립된 LLM인 Llama2, ChatGPT 및 LLaVA를 평가 모델로 사용합니다. 또한, 이미지 기능을 추출하기 위해 예측된 비전 변환기 (ViT)를 적용하고, 이를 통해 생성된 시각적 임베딩을 텍스트 기능과 정렬하여 LLM에 통합하는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과, LLM은 다중 모달 이해의 잠재력을 보여주지만 MABSA의 복잡한 요건을 충족하는 데 있어 상당한 도전에 직면하고 있음을 알 수 있었습니다. 특히, LLM들은 SLM 기반 방법보다 더 높은 계산 비용을 요구하여 실용성에 제약이 있는 것으로 나타났습니다. 이러한 결과는 LLM의 현재 한계를 부각시키며, 복잡한 다중 모달 감정 분석 작업에 대한 적응력을 향상시키기 위해 추가적인 최적화가 필요함을 강조합니다.



### The Decoy Dilemma in Online Medical Information Evaluation: A Comparative Study of Credibility Assessments by LLM and Human Judges (https://arxiv.org/abs/2411.15396)
- **What's New**: 이번 연구는 AI와 대규모 언어 모델(LLMs)이 정보 신뢰도 평가 작업에서 인지 편향(cognitive bias)에 어떻게 취약한지를 탐구합니다. 기존의 인간 평가자와의 비교를 통해, LLM이 COVID-19 관련 의학정보 (mis)정보 평가에서 인간보다 더 높은 편향 수준을 보인다는 것을 발견했습니다. 이러한 연구는 AI 도구의 합리성에 대한 일반적인 가정에 의문을 제기하며, LLM의 신뢰도 판단에서의 위험성을 강조합니다.

- **Technical Details**: 연구는 LLM과 인간 평가자의 신뢰도 평가를 비교하기 위해 조작 조건으로서 Decoy Effect를 활용했습니다. Decoy Effect는 열등한 옵션이 두 개의 유사한 옵션 사이에서 개인의 선호에 영향을 미치는 현상이며, 본 연구에선 코로나 치료웹 페이지 평가를 위한 크라우드소싱 사용자 실험을 통해 이 현상을 검증했습니다. LLM 기반 실험을 통해 다양한 모델의 평가 결과를 수집하고, 인간 평가자의 기준을 설정하여 모델의 신뢰도 판단을 측정하였습니다.

- **Performance Highlights**: 연구 결과는 더 크고 최신의 LLM이 정보의 신뢰성과 정확성을 평가하는 데 더 높은 일관성을 보이는 반면, 오히려 잘못된 정보에 더 높은 점수를 부여하는 경향이 있음을 보여줍니다. 디코이 효과는 LLM 판단에서 더 뚜렷하게 나타났으며, 이는 LLM의 인지적 편향을 드러냅니다. 연구는 LLM의 평가가 인간보다 더 많은 편향을 내포하고 있음을 논증하며, AI 판단의 오류를 줄이기 위한 필요성을 강조합니다.



### ChatBCI: A P300 Speller BCI Leveraging Large Language Models for Improved Sentence Composition in Realistic Scenarios (https://arxiv.org/abs/2411.15395)
- **What's New**: 본 논문에서는 ChatBCI를 소개하며, 이는 P300 스펠러 BCI에 LLM(대형 언어 모델)인 GPT-3.5를 통합하여 사용자에게 단어 완성과 여러 단어 예측 기능을 제공합니다. ChatBCI는 원거리 쿼리를 통해 단어 제안을 누리며, 키보드 GUI를 새롭게 디자인하여 사용자에게 더 효율적이고 개인화된 타이핑 경험을 제공합니다. 이를 통해 문장 작성을 가속화하고, 기존 P300 스펠러의 한계를 극복할 수 있는 potential이 제시됩니다.

- **Technical Details**: ChatBCI는 P300 speller BCI 기술에 LLM의 제로샷 학습을 활용하여, 사용자가 입력한 초기 글자를 기반으로 단어를 제안하고 다음 단어를 예측합니다. 새로운 GUI는 제안된 단어를 추가 키로 표시하며, P300 분류에는 SWLDA가 사용됩니다. 본 연구에서 수행된 두 가지 온라인 스펠링 과제는 ChatBCI의 실용성을 평가하는데 중요한 역할을 하게 됩니다.

- **Performance Highlights**: 실험 결과, Task 1에서 ChatBCI는 기존의 글자 단위 BCI 스펠러보다 평균 62.14%의 시간 단축과 53.22%의 키 입력 수 감소를 달성하며, 정보 전송 속도는 198.96% 증가했습니다. Task 2에선 80.68%의 키 입력 절약을 발생시키고, 8.53자/분의 타이핑 속도를 기록했습니다. 이러한 결과는 ChatBCI가 전통적인 스펠러 BCI에 비해 훨씬 효율적임을 보여주며, 특히 사용자 맞춤의 통신 도구 개발 가능성을 제시합니다.



### Gradient-Free Classifier Guidance for Diffusion Model Sampling (https://arxiv.org/abs/2411.15393)
- **What's New**: 이번 연구에서는 이전에 훈련된 분류기를 활용하여 gradient descent(그래디언트 하강법)을 사용하지 않고 효율적으로 샘플링하는 새로운 방법인 Gradient-free Classifier Guidance (GFCG)를 제안하고 있습니다. GFCG는 시간마다 적응적으로 참조 클래스와 해당 안내 스케일을 결정하여 고충실도의 이미지를 생성하는 데 도움이 됩니다. 또한, GFCG는 기존의 다른 가이딩 방법인 Autoguidance (ATG)와 결합하여 이미지 품질을 향상시키면서도 계산 부담을 주지 않는 성과를 보여줍니다.

- **Technical Details**: GFCG는 비강화적 샘플을 피하기 위해 참조 클래스를 사용하고, 훈련된 분류기를 기반으로 샘플링 중 적응적으로 가이드를 조정합니다. 이는 CG(클래시파이어 가이던스)와 비교하여 고급 정보에 대한 비용을 줄이면서 클래스 레이블과의 정렬을 개선합니다. 연구에서는 GFCG를 다양한 확산 모델에 적용하고, 그 효율성과 효과성을 평가하기 위해 혼합 가이드 및 추가 가이드 방법을 비교 분석했습니다.

- **Performance Highlights**: 실험 결과, GFCG는 이미지넷 데이터셋에서 512x512 해상도로 $	ext{FD}_{	ext{DINOv2}}$ 23.09의 기록을 달성하면서 ATG에 비해 높은 분류 정밀도(94.3%)를 기록했습니다. GFCG는 이미지 품질과 다양성을 모두 강화하며, 다양한 모델에서 분류 정확도를 현저히 향상시켰습니다. 마지막으로, GFCG는 다른 가이드 방법과의 조합에서도 긍정적인 결과를 보였습니다.



### Nd-BiMamba2: A Unified Bidirectional Architecture for Multi-Dimensional Data Processing (https://arxiv.org/abs/2411.15380)
- **What's New**: 논문에서는 Nd-BiMamba2라는 새로운 다차원 양방향 신경망 아키텍처를 제안합니다. 이 모델은 1D, 2D, 3D 데이터를 모두 효율적으로 처리할 수 있으며, 기존의 시스템보다 더 통합된 설계를 갖추고 있습니다. 또한, 양방향 처리 메커니즘과 적응형 패딩 전략을 도입하여 계산 효율성을 유지하면서 다차원 데이터의 양방향 정보를 포착할 수 있습니다.

- **Technical Details**: Nd-BiMamba2는 Mamba2 모듈을 기반으로 하며, 다양한 데이터 차원을 지원하는 효율적인 양방향 처리를 위해 설계되었습니다. 이 모델은 입력 데이터의 차원에 따라 패딩 크기를 조정하는 적응형 패딩 전략을 포함하여 계산 효율성과 메모리 소비를 개선합니다. Nd-BiMamba2는 ONNX와 TorchScript로 내보내져 다양한 하드웨어 플랫폼에서 테스트되어 그 이식성과 유연성을 검증했습니다.

- **Performance Highlights**: 실험 결과 Nd-BiMamba2는 CPU, GPU, 모바일 장치 등 여러 플랫폼에서 효율적으로 작동하며, 실용적인 응용에서의 가능성을 보여줍니다. 이 모델은 고차원 작업에서의 퍼포먼스를 크게 향상시켰으며, 개발 및 유지 보수 비용을 줄이는 간소화된 아키텍처를 제공합니다. Nd-BiMamba2는 다차원 데이터 모델링과 크로스 플랫폼 배포에 새로운 방향을 제시합니다.



### AdamZ: An Enhanced Optimisation Method for Neural Network Training (https://arxiv.org/abs/2411.15375)
Comments:
          13 pages, 9 figures, 3 tables

- **What's New**: AdamZ는 Adam 최적화 알고리즘의 고급 변형으로, 신경망 훈련의 수렴 효율성을 향상시키기 위해 개발되었습니다. 이 최적화기는 오버슈팅(overshooting)과 정체(stagnation) 문제를 해결하는 메커니즘을 통합하여 학습률(learning rate)을 동적으로 조절합니다. AdamZ는 오버슈팅이 감지될 때 학습률을 줄이고, 정체가 발생할 때는 학습률을 증가시켜 모델의 안정성과 정확성을 향상시킵니다.

- **Technical Details**: AdamZ는 오버슈팅 및 정체를 감지하고 완화하는 메커니즘을 통합하여, 손실 함수(loss function)의 특성에 반응하여 학습률을 조정합니다. 주요 하이퍼파라미터(hyperparameters)로는 오버슈팅 및 정체 계수(overshoot and stagnation factors), 임계값(thresholds), 인내(patience) 레벨이 포함됩니다. 이러한 추가 하이퍼파라미터들은 AdamZ가 훈련 동역학에 응답하도록 설계되어 있습니다.

- **Performance Highlights**: 비교 결과, AdamZ는 손실 함수를 최소화하는 데 일관되게 우수한 성능을 보여주며, 다양한 작업에서 모델 성능이 개선되는 것을 입증하였습니다. 다른 최적화 알고리즘에 비해 약간 긴 훈련 시간을 필요로 하지만, 정확도가 중요한 응용 분야에서는 특히 유리합니다. AdamZ의 동적 학습률 조정 기능은 복잡하고 고차원적인 학습 환경에서 더욱 안정적인 성능을 제공합니다.



### Transforming NLU with Babylon: A Case Study in Development of Real-time, Edge-Efficient, Multi-Intent Translation System for Automated Drive-Thru Ordering (https://arxiv.org/abs/2411.15372)
Comments:
          12 pages, 3 figures, 2 tables

- **What's New**: 본 논문에서는 Babylon이라는 새로운 transformer 기반 아키텍처를 소개합니다. Babylon은 자연어 이해(NLU)를 인텐트 번역(task) 문제로 전환하여, 자연어 입력을 인텐트와 슬롯 정보를 인코딩하는 일반 언어 단위인 'transcodes'로 변환합니다. 이러한 접근 방식을 통해 복잡한 다중 인텐트 시나리오 처리가 가능하며, LSTM 기반의 토큰 풀링 메커니즘을 통합하여 사용자 쿼리의 입력 길이를 줄이는 동시에 낮은 지연 시간과 메모리 요구 사항을 최적화합니다.

- **Technical Details**: Babylon 모델은 주요 구성 요소로 자동 음성 인식(ASR), 자연어 이해(NLU), 대화 관리(Dialog Management), 및 텍스트-음성 변환(TTS) 합성을 갖추고 있습니다. NLU는 음성을 실시간으로 전사하여 생성된 음소(phoneme)를 일반 언어 단위인 'transcodes'로 변환하는 기능을 수행합니다. 이 모델은 여러 사용자 쿼리를 단일 대화 턴으로 처리할 수 있는 능력을 가지며, 이를 통해 Transform의 계산 부하를 줄이고 ASR 오류를 완화합니다.

- **Performance Highlights**: 일련의 실험 결과, Babylon은 Flan-T5 및 BART와 같은 전통적인 NMT 모델 대비 정확도-지연 시간-메모리 풋프린트 수준에서 유의미한 성능 향상을 보여주었습니다. 특히, Babaylon 모델은 단일 CPU 코어에서 작동하며, 현실 세계의 배포에서 실용성을 입증하였습니다. 이러한 최적화는 모든 지연 시간 민감 애플리케이션에 필수적인 요소로, 드라이브 스루 시스템과 같은 환경에서 큰 장점을 제공합니다.



### Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers (https://arxiv.org/abs/2411.15370)
Comments:
          In The Thirty-eighth Annual Conference on Neural Information Processing Systems. Source code at this https URL and companion video at this https URL

- **What's New**: 이 논문에서는 제한된 자원에서 실제 시스템에서 사용할 수 있는 새로운 방법인 Action Value Gradient (AVG)를 제안합니다. 기존의 Deep Policy Gradient 방법들은 대용량 replay buffer와 비싼 배치 업데이트를 필수적으로 요구하여 효율적으로 학습하기 어렵습니다. 하지만 AVG는 이러한 제한을 극복하고 딥 러닝을 활용하여 실시간 로봇 학습을 가능하게 했습니다. 특히, 성능 저하 없이 점진적인 학습을 수행할 수 있는 방법을 제시합니다.

- **Technical Details**: AVG는 재파라미터화 그래디언트 (reparameterization gradient) 추정기를 사용하여 안정적인 학습을 수행합니다. 기존의 정책 경량화 방법들은 자원이 제한된 환경에서 효과적으로 작동하지 못하는 문제를 가지고 있었기에, AVG는 학습 안정성을 위해 정규화 및 스케일링 기법을 포함하고 있습니다. 이를 통해 AVG는 배치 업데이트, replay buffer, 타겟 네트워크없이도 효과적으로 학습할 수 있는 방법론을 제공합니다.

- **Performance Highlights**: AVG는 여러 로봇 시뮬레이션 벤치마크에서 뛰어난 성과를 보여주었습니다. 특히, AVG는 배치 정책 경량화 방법들과 유사한 최종 성능을 달성하였습니다. 이 연구는 실제 로봇에서도 점진적인 업데이트만 사용하여 효과적인 딥 강화 학습을 수행할 수 있는 가능성을 보여주며, 기존의 방법들이 가진 한계를 극복하는 데 기여하고 있습니다.



### Exploiting Watermark-Based Defense Mechanisms in Text-to-Image Diffusion Models for Unauthorized Data Usag (https://arxiv.org/abs/2411.15367)
- **What's New**: 이 연구에서는 텍스트-이미지(T2I) diffusion 모델에서 사용되는 다양한 watermark 기반 보호 방법의 강인성을 조사합니다. 기존의 여러 이미지 변환이 watermark 효과를 제거하는 데 효과적이지 않음을 관찰했으며, 이를 해결하기 위해 새로운 기법인 Rattan을 제안합니다. Rattan은 노이즈가 있는 이미지를 다루는 diffusion 프로세스를 활용하여 보호된 입력의 고수준 특징을 보존하고 낮은 수준의 디테일은 무시하는 방식으로 작동합니다.

- **Technical Details**: Rattan은 보호된 이미지를 입력으로 사용하여 T2I 모델이 새로운 이미지를 생성하도록 하며, 이 과정에서 텍스트와 결합하여 고유한 출력을 생성합니다. 이 방법은 기존의 이미지 변환 접근 방식을 고수준 특징 추출로 전환하여, watermark와 같은 세부 사항을 제거합니다. 연구 결과, Rattan을 사용하면 기존 보호 방법의 탐지율을 상당히 낮출 수 있으며, 이는 우연히 발생할 확률과 유사합니다.

- **Performance Highlights**: 세 개의 데이터셋과 140개의 T2I diffusion 모델에 대한 실험 결과, Rattan은 기존 보호 방법의 탐지율을 50%로 줄였으며, 이는 무작위 추측과 비슷한 수준입니다. 이러한 결과는 Rattan이 watermark 기반 데이터 보호를 강화하는 데 있어 효과적인 방법임을 시사합니다. Rattan을 통해 T2I 모델의 저작권 침해 및 개인정보 보호 문제를 해결하는 데 기여할 수 있을 것으로 기대됩니다.



### Exploring Facets of Language Generation in the Lim (https://arxiv.org/abs/2411.15364)
Comments:
          24 pages

- **What's New**: 이 논문은 언어 생성 알고리즘의 새로운 접근 방식을 제시합니다. 기존 연구(Kleinberg and Mullainathan [KM24])는 모든 셈할 수 있는 언어 컬렉션에 대해 언어 생성 문제의 긍정적인 결과를 보여주었습니다. 후속 연구(Raman and Tewari [RT24])에서는 언어 생성에 필요한 고유 입력의 수를 다루어, 이를 균일 생성과 비균일 생성으로 나눴습니다.

- **Technical Details**: 이 논문은 각 셈할 수 있는 언어 컬렉션에 대해 비균일 생성(non-uniform generation)을 가능하게 하는 생성기를 제시합니다. 여기서 생성기는 고유 문자열의 고정된 상수를 수신한 후에만 유효한 문자열을 생성합니다. 그러나, 알고리즘이 입력된 언어 컬렉션의 두 언어만을 이용해 비균일 생성을 수행할 수 없다는 것도 증명하였습니다.

- **Performance Highlights**: 결과적으로, 이 논문은 유효성과 범위 간의 트레이드오프가 생성 알고리즘에서 내재되어 있음을 보여줍니다. 또한, 피드백을 포함한 균일 생성 모델을 연구하여 피드백이 가능한 언어 컬렉션의 특성을 규명하였습니다. 이 연구는 생성 알고리즘의 새로운 가능성을 제시하며, 다양한 응용 프로그램에서 활용될 수 있습니다.



### UniGaussian: Driving Scene Reconstruction from Multiple Camera Models via Unified Gaussian Representations (https://arxiv.org/abs/2411.15355)
Comments:
          Technical report

- **What's New**: 이 논문은 도시 장면 재구성을 위한 새로운 접근 방식인 UniGaussian을 제안합니다. 이전 방법들이 핀홀 카메라에 집중한 반면, UniGaussian은 핀홀 및 어안(fisheye) 카메라 모델을 통합해 3D Gaussian 표현을 학습합니다. 실제로, 어안 카메라에서 발생하는 광선 왜곡 문제를 해결하기 위한 미분 렌더링 방법을 도입하여 실시간 렌더링을 보장합니다.

- **Technical Details**: 본 연구는 다양한 카메라 모델에서 3D Gaussian 표현을 학습하는 새로운 프레임워크를 설계합니다. 특히, 어안 카메라 모델에 적합한 일련의 아핀 변환을 통해 3D Gaussian의 변형을 적용합니다. 이 프레임워크는 여러 센서 및 모달리티(예: 깊이, 의미론적, 노말 및 LiDAR 포인트 클라우드)를 모델링하여 운전 장면에 대한 포괄적인 이해를 이룹니다.

- **Performance Highlights**: 실험 결과, UniGaussian은 운전 장면 시뮬레이션을 위한 뛰어난 렌더링 품질과 빠른 속도를 달성하였습니다. 이 방법은 다양한 미분 가능 카메라 모델에 대한 적응력을 보유하고 있으며, 실시간 성능을 유지하는 동시에 여러 센서 및 모달리티를 효과적으로 통합합니다.



### GeoScatt-GNN: A Geometric Scattering Transform-Based Graph Neural Network Model for Ames Mutagenicity Prediction (https://arxiv.org/abs/2411.15331)
- **What's New**: 이 논문은 뮤타제닉성(mutagenicity) 예측의 도전 과제를 다루며, 2D 산란 계수(scattering coefficients)를 기반으로 한 새로운 접근 방식을 소개합니다. 또한 기하학적 그래프 산란(Geometric Graph Scattering)과 그래프 동형성 네트워크(Graph Isomorphism Networks), 그리고 머신러닝 모델을 결합한 하이브리드 방식을 제안하여 뮤타제닉성 예측에서 강력한 성과를 달성하였습니다. 마지막으로, GGS 노드(feature)를 통합한 혁신적인 그래프 신경망(GNN) 아키텍처, MOLG3-SAGE를 소개하며, 화합물의 독성 예측에서 뛰어난 정확성을 자랑합니다.

- **Technical Details**: 논문에서는 기하학적 산란 변환(Geometric Scattering Transform, GST)을 활용하여 분자 그래프에서 다중 스케일 구조를 포착합니다. GST를 통해 생성된 임베딩(embedding)은 두 가지 주요 용도로 사용되며, 첫 번째는 머신러닝 모델의 입력으로 활용되고, 두 번째는 분자 간 유사성을 계산하기 위한 그래프 구성에 사용됩니다. GraphSAGE 알고리즘을 적용하여 이 그래프 기반 표현을 처리하고, 분자 간의 관계 정보를 활용하여 독성 예측의 정확성을 더욱 향상시킵니다.

- **Performance Highlights**: ZINC 데이터셋에서의 실험 결과, 기존의 전통적인 방법들과 비교하여 뚜렷한 성능 향상이 입증되었습니다. 특히, 2D 및 기하학적 산란 기법을 GNN과 융합함으로써 뮤타제닉성 예측에서 최첨단 성능을 달성했습니다. 이 연구는 새로운 화학 물질의 안전성을 보장하는 데 있어 독성 예측의 중요성을 강조하며, 약물 발견과 화학 안전성 평가에 대한 광범위한 함의를 가집니다.



### PPLqa: An Unsupervised Information-Theoretic Quality Metric for Comparing Generative Large Language Models (https://arxiv.org/abs/2411.15320)
- **What's New**: 이번 논문에서는 PPLqa라는 새로운 정보 이론 기반의 평가 지표를 제안합니다. 이 지표는 인간의 주관적인 평가 없이도 생성된 응답의 품질을 평가할 수 있으며, 다양한 언어에 독립적이고 쉽게 계산할 수 있는 장점을 가지고 있습니다. PPLqa는 생성된 언어 모델의 응답을 효율적으로 비교할 수 있게 해주며, 이는 기존 지표들보다 더 나은 성과를 보입니다.

- **Technical Details**: PPLqa는 정보 이론적인 접근을 통해 프롬프트와 응답 쌍의 구조를 정량화합니다. 이는 응답의 일관성(coherence), 유창성(fluency), 적합성(relevance), 일관성(consistency)을 모두 포함하면서도 기존의 여러 메트릭에 비해 더 직관적입니다. 실험에서는 다양한 주제 영역에서 LLM으로부터 생성된 응답을 평가하여 PPLqa가 인간 또는 다른 LLM의 평가 결과와 높은 상관관계를 나타낸다는 것을 보여주었습니다.

- **Performance Highlights**: PPLqa는 긴 형식의 질의응답(Q&A)에서 다른 관련 지표들과 비슷한 성과를 보이며, MT-Bench 데이터셋에서도 우수성을 입증했습니다. 특히, 인간 평가와의 상관관계가 뛰어나고, Anthropic의 Claude 3 LLM 평가와 비교했을 때 더 강한 일관성을 보여줍니다. 이를 통해 PPLqa가 다양한 상황에서 신뢰할 수 있는 평가 도구임을 확인할 수 있습니다.



### MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs (https://arxiv.org/abs/2411.15296)
Comments:
          Produced by MME+MMBench+LLaVA Teams. Project Page: this https URL

- **What's New**: 이번 논문에서는 인공지능 일반화(AI General Intelligence, AGI)의 중요한 방향으로 여겨지는 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 평가 방법에 대한 포괄적인 조사 결과를 제시합니다. 기존의 LLM들을 기반으로 하여 MLLMs는 시각과 오디오를 포함한 다양한 정보 형태를 처리하는 능력을 추가로 개발하였습니다. 이에 따라 MLLMs의 성능을 평가하기 위한 새로운 기준과 방법들이 요구되고 있음을 설명합니다.

- **Technical Details**: MLLMs는 모달리티 인코더(modality encoder), LLM, 그리고 이들을 연결하는 커넥터(connector)로 구성되어 있습니다. 비전-언어 모델의 예를 들면, 텍스트 쿼리와 비전 샘플을 입력으로 받아 비전 인코더가 특징(feature)을 추출하고, 커넥터는 비전 특징과 텍스트 임베딩을 정렬합니다. 이후 이 정렬된 비전 특징은 사용자 쿼리의 텍스트 임베딩과 결합되어 LLM에 의해 자연어 응답으로 생성됩니다.

- **Performance Highlights**: MLLM 평가의 예상 방향으로는 기능별 분류, 기능 중심 평가, 태스크 지향 평가, 그리고 더욱 다양한 모달리티의 포함 등이 있습니다. 본 논문은 다양한 평가 기준을 계층적으로 분류하고, 평가 제작 과정에서 필요한 주의 사항을 정리하였으며, 성능 측정 방법으로 인간 기반, LLM 기반, 그리고 스크립트 기반의 세 가지 주요 방식을 제시합니다. 이로써 연구자들이 적절한 평가 기준을 쉽게 찾고 효과적인 평가 방안을 탐색할 수 있도록 돕고자 합니다.



### Influence functions and regularity tangents for efficient active learning (https://arxiv.org/abs/2411.15292)
Comments:
          33 pages, 4 figures

- **What's New**: 이 논문에서는 회귀 모델에 데이터에 대한 호기심을 부여하기 위한 효율적인 방법을 설명합니다. 머신 러닝 분야에서 제안된 새로운 프레임워크인 Active Learning은 반지도 학습 세팅에서 레이블을 쿼리하기 위한 데이터 포인트를 자동으로 선택하는 것을 의미합니다. 본 논문에서 제안한 방법은 training 중 모델의 파라미터 벡터와 함께 계산할 수 있는 "regularity tangent" 벡터를 기반으로 합니다.

- **Technical Details**: 제안된 방법은 각 데이터 포인트에 대한 모델의 손실에 대한 기울기 벡터와 정규성 접선 벡터의 내적을 취하여 해당 포인트가 모델 복잡성에 미치는 영향을 측정합니다. 이 기법은 training이 완료된 후 쿼리할 데이터 포인트에 대한 "호기심"을 평가하는 과정을 간소화합니다. 제안된 방법에 의해 계산된 양은 "influence function"의 예이며, 특정 데이터 포인트를 추가 가중치로 높일 때 발생하는 예상되는 제곱 변화량을 측정하는 데 사용됩니다.

- **Performance Highlights**: 제안된 기술은 단일 정규성 접선 벡터를 사용하여 모델의 복잡성 변화를 평가하여 더 효율적인 데이터 레이블 선택이 가능하도록 합니다. 이 방법은 모델 저장에 필요한 공간을 두 배로 증가시키지만, 시간 복잡도에는 영향을 주지 않습니다. Active Learning 프레임워크에서 이 방법을 사용하여 새로운 훈련 데이터를 선택할 수 있는 여러 가지 방법을 제안합니다.



### Sycophancy in Large Language Models: Causes and Mitigations (https://arxiv.org/abs/2411.15287)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 보이는 아첨적(sycophantic) 행동의 원인, 영향, 완화 전략에 대한 기술적 조사 결과를 제시합니다. 기존 연구를 바탕으로 아첨적 경향을 측정하고 정량화하는 방법에 대한 논의가 포함되어 있으며, 이는 모델 성능을 유지하면서 아첨적 행동을 줄이기 위한 전략을 평가하는 데 기여합니다.

- **Technical Details**: 대형 언어 모델은 텍스트 데이터를 기반으로 다음 토큰을 예측하는 신경망(neural network)입니다. 아첨적 행동을 측정하는 방법으로는 여러 가지가 있으며, 정확도(accuracy), 동의율(agreement rate), 뒤집기율(flip rate)과 같은 지표가 사용됩니다. 인간 평가(human evaluation)와 자동화된 메트릭을 통해 모델의 아첨적 경향을 정량화하며, 적대적 테스트(adversarial testing)를 통해 모델의 잠재적 취약성을 드러내는 방법도 모색합니다.

- **Performance Highlights**: 논문은 아첨적 행동이 LLM의 신뢰성과 윤리적 적용에 미치는 영향에 대해 분석하고 있으며, 정보의 사실 정확성을 보장하고 사용자 신뢰를 유지하는 데 있어 아첨적 행동을 완화하는 것이 중요하다는 점을 강조합니다. 다양한 완화 기술(techniques)인 훈련 데이터 개선, 새로운 세부 조정 방법(novel fine-tuning methods), 배포 후 제어 메커니즘(post-deployment control mechanisms)을 평가하며, 아첨적 행동을 줄이면서도 모델 성능을 유지하기 위한 경로를 제시합니다.



### Forecasting Unseen Points of Interest Visits Using Context and Proximity Priors (https://arxiv.org/abs/2411.15285)
Comments:
          2024 IEEE International Conference on Big Data workshop BSD 2024

- **What's New**: 본 논문에서는 기존의 방법이 활용하지 않는 신규 POI(Points of Interest)를 예측할 수 있는 모델을 제안합니다. 이 모델은 사용자의 이동 패턴을 분석하여, 신규 POI의 의미적 문맥을 예측한 후 주변 POI의 확률 분포와 결합하여 특정 POI를 예측합니다. 이를 통해, 예측의 정확성을 17% 향상시켰고, 신규 POI가 도입되더라도 모델의 견고성이 유지됨을 보여주었습니다.

- **Technical Details**: 모델은 Transformer 기반 접근법을 사용하여 사용자 과거 방문 시퀀스의 소셜 및 의미적 맥락을 인코딩합니다. 예측의 정확성을 높이기 위해 POI의 의미적 카테고리와 위치 정보를 결합하는 간단한 직관에 따라 설계되었습니다. 이는 새로운 POI를 예측하는 데 도움이 되는 확률론적 프레임워크를 통해 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존 방법보다 더 나은 성능을 보였습니다. 새로운 POI가 출현해도 예측 정확성의 감소율이 낮아, 실제 응용 프로그램에서 활용 가능한 가능성을 보여줍니다. 특히, 위치 기반 추천 시스템에서 사용자에게 새로운 POI를 추천하는 데 있어 유용하게 작용할 수 있음을 확인했습니다.



### ElastiFormer: Learned Redundancy Reduction in Transformer via Self-Distillation (https://arxiv.org/abs/2411.15281)
- **What's New**: 본 논문에서는 ElastiFormer라는 후처리 기법을 소개합니다. ElastiFormer는 미리 훈련된 Transformer 모델을 더 유연한 선택이 가능한 형태로 변환하여 가변적인 inference time compute를 가능하게 합니다. 이 기법은 네트워크 파라미터와 입력 토큰의 하위 집합을 동적으로 선택하는 경량의 라우팅 모듈을 도입하여, 입력에 따라 변동되는 방식으로 처리합니다.

- **Technical Details**: ElastiFormer의 라우팅 모듈은 self-distillation losses를 사용하여, 미리 훈련된 모델과 탄력적인 대응 모델 간의 출력을 최소화하는 방향으로 학습됩니다. ElastiFormer 는 모든 모델에서 작동할 수 있도록 다양한 모달리티(예: 언어 모델링, 이미지 모델링, 비주얼-언어 모델링)에 대한 가정을 하지 않습니다. 실험 결과, ElastiFormer를 통해 MHA와 MLP의 활성 파라미터 수를 각각 38%와 56% 줄이며, 전반적으로 20%에서 50%까지 경제적인 계산을 향상할 수 있음을 보였습니다.

- **Performance Highlights**: ElastiFormer의 도입으로 다양한 모달리티에서 실질적인 성능 개선이 가능합니다. 우리는 비주얼-언어 모델에 ElastiFormer를 적용했을 때, 이미지 토큰의 40%를 언어 디코더로 디코딩하기 전에 생략해도 성능에 큰 영향을 미치지 않음을 입증했습니다. 또한 ElastiFormer가 훈련 도메인에 대해 강력한 내성을 보여줘 학습된 라우팅이 다양한 데이터 분포에 안정적임을 강조합니다.



### Event USKT : U-State Space Model in Knowledge Transfer for Event Cameras (https://arxiv.org/abs/2411.15276)
- **What's New**: 이 연구는 Event-to-RGB 지식 전이를 위한 맞춤형 U자형 State Space Model Knowledge Transfer (USKT) 프레임워크를 제안하여 이벤트 데이터를 RGB 프레임에 호환 가능한 입력으로 변환합니다. 이를 통해 사전 훈련된 RGB 모델을 효과적으로 재사용하고, 최소한의 매개변수 조정으로 경쟁력 있는 성능을 얻을 수 있습니다. 또한, 새로운 Bidirectional Reverse State Space Model (BiR-SSM)을 도입하여 계산 자원을 절약하면서도 효율적인 모델링을 가능하게 합니다.

- **Technical Details**: USKT 프레임워크는 잔여 다운 샘플링 블록과 잔여 업 샘플링 블록을 포함하며, Bidirectional Reverse State Space Model (BiR-SSM)을 통해 전체적인 기능 의존성을 포착합니다. 이벤트 데이터는 (x,y,t,p) 구조로 시각화되어, 3차원 그리드로 매핑되어 정리됩니다. 이 과정에서 각 격자는 시간 빈 내에 이벤트의 편극성을 집계하여 최종적으로 시공간 정보를 유지한 3차원 텐서를 생성합니다.

- **Performance Highlights**: USKT와 ResNet50을 통합하면 DVS128 Gesture, N-Caltech101, CIFAR-10-DVS 데이터셋에서 각각 0.95%, 3.57%, 2.9%의 성능 향상을 달성했습니다. 이러한 결과는 USKT의 적응성과 효과성을 강조하며, 모델 성능 향상을 위해 리컨스트럭션과 분류 목표를 결합하는 하이브리드 손실 함수를 제공합니다.



### Feature-interactive Siamese graph encoder-based image analysis to predict STAS from histopathology images in lung cancer (https://arxiv.org/abs/2411.15274)
Comments:
          accept for publication in npj Precision Oncology

- **What's New**: 이 논문에서는 폐암의 확산 패턴인 공기 공간 확산(Spread through air spaces, STAS)을 분석하기 위한 새로운 이미지 분석 모델인 VERN을 소개합니다. 기존의 조직병리학적 방법들이 주관적이고 시간 소모적이며 오진이 발생할 가능성이 높아 대규모 적용에 제한이 있던 문제를 해결하고자 합니다. VERN은 폐암 조직병리학 이미지에서 STAS를 예측하기 위해 특성-상호작용(Siamese graph encoder)을 활용합니다.

- **Technical Details**: VERN 모델은 공간적(topological) 특성을 캡처하기 위해 특성 공유와 스킵 연결(skip connections)을 사용하여 모델 훈련을 개선합니다. 연구진은 1,546개의 조직병리학 슬라이드를 사용해 대규모 단일 집단(STAS lung cancer) 데이터셋을 구축하였습니다. VERN은 내부 검증에서 0.9215의 AUC 값을 달성하였고, 냉동(frozen) 및 파라핀(paraffin) 내장 테스트 섹션에서는 각각 0.8275 및 0.8829의 AUC를 기록하여 임상 수준의 성능을 입증하였습니다.

- **Performance Highlights**: VERN은 단일 집단 및 세 개의 외부 데이터셋에서 검증되었으며, 강력한 예측 성능과 일반화 가능성을 보여주었습니다. 이는 STAS 진단의 효율과 정확성을 향상시키기 위한 개방형 플랫폼(http URL)을 제공함으로써, 기존의 한계를 극복할 수 있는 가능성을 제시합니다.



### Curriculum-enhanced GroupDRO: Challenging the Norm of Avoiding Curriculum Learning in Subpopulation Shift Setups (https://arxiv.org/abs/2411.15272)
- **What's New**: 이번 연구에서는 Curriculum Learning (CL) 접근 방식이 없는 현재의 subpopulation shift 기술을 보완하기 위해, Curriculum-enhanced Group Distributionally Robust Optimization (CeGDRO)라는 새로운 방법론을 제안하고 있습니다. CeGDRO는 최대한 공정한 가중치 초기화를 통해 편향된 가설로의 쉬운 수렴을 방지하여, 모델의 결정을 더욱 신뢰할 수 있도록 하는 것을 목표로 하고 있습니다. 또한, 모델은 훈련 초기에 편향된 샘플과 편향이 상충하는 샘플을 균형있게 학습하게 됩니다.

- **Technical Details**: 세부적으로, CeGDRO는 GroupDRO의 개념을 활용하여 데이터의 초기 편차를 균형잡는 훈련 커리큘럼을 설계합니다. 이를 통해, bias-confirming 샘플과 bias-conflicting 샘플의 손실 차이를 조절하고, 훈련 과정에서 모델의 가중치 초기화를 비편향적인 지점에서 이루어지도록 합니다. 이러한 과정을 통해, 강한 상관관계를 지닌 클래스와 환경 간의 연결을 가능한 한 오랫동안 지연시킵니다.

- **Performance Highlights**: 제안된 CeGDRO 방법은 다양한 subpopulation shift 데이터셋을 통해 평가되었으며, 기존의 최첨단 결과들에 비해 일관된 개선을 보였습니다. 특히, Waterbirds 데이터셋에서는 성능이 최대 6.2% 개선된 결과를 보여주었습니다. 이는 CeGDRO가 subpopulation shift 상황에서도 효과적으로 작동할 수 있음을 입증합니다.



### EADReg: Probabilistic Correspondence Generation with Efficient Autoregressive Diffusion Model for Outdoor Point Cloud Registration (https://arxiv.org/abs/2411.15271)
- **What's New**: EADReg라는 새로운 프레임워크를 제안하여 LiDAR 포인트 클라우드의 효율적이고 견고한 등록을 달성하고 있습니다. 이 방법은 자기 회귀적(diffusion models) 방법론을 기반으로 하여, 특히 야외 LiDAR 포인트에서 발생하는 도전적인 문제를 해결합니다. EADReg는 전체 등록 과정을 거친 후, 정제된 포인트 클라우드 쌍을 활용하여 더욱 향상된 성능을 발휘합니다.

- **Technical Details**: EADReg는 coarse-to-fine 등록 파라다임을 따릅니다. 초기 단계에서는 Bi-directional Gaussian Mixture Model (BGMM)을 사용하여 아웃라이어 포인트를 제거하고, 정제된 포인트 클라우드 쌍을 획득합니다. 다음으로, 자기 회귀 프로세스로서의 diffusion-based PCR을 처리하여 강력한 포인트 대응관계를 생성하고 이를 반복적으로 정제합니다.

- **Performance Highlights**: KITTI와 NuScenes 벤치마크 데이터셋에서 EADReg는 최신 기술 수준의 성능을 입증했습니다. 기존의 convolutional 기반 방법론과 비슷한 실행 시간을 기록하며, diffusion 기반 방법이 가지는 약점을 극복하고 있습니다. 이로 인해 실제 응용에서도 유용성을 더할 것으로 기대됩니다.



### AI-Driven Real-Time Monitoring of Ground-Nesting Birds: A Case Study on Curlew Detection Using YOLOv10 (https://arxiv.org/abs/2411.15263)
- **What's New**: 이 연구에서는 야생 동물 모니터링에 AI(인공지능)를 활용한 새로운 접근 방식을 소개합니다. 특히, 멸종 위기 속에 있는 땅에 둥지를 트는 조류인 curlew(누르기)의 실시간 종 탐지를 목표로 하고 있습니다. 기존의 수작업 데이터 처리 방식에서 벗어나, 효율적인 모니터링을 가능하게 하는 시스템을 개발했습니다.

- **Technical Details**: 연구팀은 YOLOv10(You Only Look Once Version 10) 모델을 사용자 맞춤형으로 훈련시켜, 3/4G 연결이 가능한 카메라를 통해 수집된 데이터를 실시간으로 처리합니다. 해당 시스템은 Conservation AI 플랫폼과 연계되어 운영되며, 총 11개 둥지 위치에서 curlew와 그 새끼를 탐지하고 분류합니다. 높은 성능을 보이는 이 모델은 각각 90.56%의 민감도와 100%의 특이도, 95.05%의 F1-score를 기록했습니다.

- **Performance Highlights**: 실험 결과, curlew 탐지에서 90.56%의 높은 민감도와 100%의 특이도를 달성했으며, 새끼 curlew에 대해서도 92.35%의 민감도에 100%의 특이도를 기록했습니다. 이러한 성과는 AI 기반 모니터링 시스템이 생물 다양성 평가를 위한 정확하고 신속한 데이터를 제공할 수 있음을 보여줍니다. 마지막으로, 이 연구는 기술의 발전이 생태 연구에 어떻게 기여할 수 있는지를 잘 설명하고 있습니다.



### VIVID-10M: A Dataset and Baseline for Versatile and Interactive Video Local Editing (https://arxiv.org/abs/2411.15260)
Comments:
          17 pages, 14 figures

- **What's New**: 본 논문은 비디오 편집의 질적 향상을 위해 VIVID-10M이라는 데이터셋과 VIVID라는 모델을 소개합니다. VIVID-10M은 처음으로 대규모 하이브리드 이미지-비디오 로컬 편집 데이터셋으로, 9.7M 샘플을 포함하고 있어 다양한 비디오 편집 작업을 다룹니다. 이는 데이터 구축과 모델 훈련 비용을 줄이는 데 도움을 주며, 비디오 편집의 상호작용성을 높이기 위한 기초가 됩니다.

- **Technical Details**: VIVID는 VIVID-10M에서 훈련된 다목적 비디오 로컬 편집 모델로, 엔티티 추가, 수정 및 삭제를 지원합니다. 이 모델은 키프레임(关键帧) 기반의 상호작용 비디오 편집 메커니즘을 제안하여 사용자가 반복적으로 키프레임을 편집하고 이를 다른 프레임으로 전파할 수 있도록 합니다. 이러한 접근 방식은 원하는 결과를 얻기 위한 지연 시간을 줄이는 데 기여합니다.

- **Performance Highlights**: 광범위한 실험 평가 결과, VIVID 접근법이 비디오 로컬 편집에서 최첨단 성능을 달성했으며, 자동화된 메트릭과 사용자 연구에서 기존 방법을 초월하는 성과를 보였습니다. 이는 VIVID-10M 데이터셋과 VIVID 편집 모델을 통해 비디오 편집의 가능성을 한층 확장할 것으로 기대됩니다.



### The Explabox: Model-Agnostic Machine Learning Transparency & Analysis (https://arxiv.org/abs/2411.15257)
Comments:
          5 pages, 3 figures

- **What's New**: Explabox는 투명하고 책임감 있는 기계 학습(Model Learning, ML) 모델 개발을 돕는 오픈소스 도구입니다. 이 도구는 '탐색(Explore)', '검토(Examine)', '설명(Explain)', '노출(Expose)'의 네 단계를 통해 설명 가능하고 공정하며 견고한 모델을 구축할 수 있도록 지원합니다. Explabox는 파이썬으로 구현되었으며, 다양한 데이터와 모델을 처리할 수 있는 통합적인 분석 접근 방식을 제공합니다.

- **Technical Details**: Explabox는 데이터와 모델을 통합적으로 가져오는 방식으로 불투명한 ingestibles를 투명한 digestibles로 변환합니다. 이 분석 단계에서는 데이터 세트 크기, 라벨 분포와 같은 설명 통계, ML 모델의 성능 메트릭과 모델 동작에 대한 설명을 포함합니다. 다양한 형식의 데이터 입력을 지원하며, NumPy 배열, Pandas DataFrame, 원시 파일 등을 통해 효율적으로 데이터를 처리할 수 있습니다.

- **Performance Highlights**: Explabox는 ML 모델의 설명 가능성, 공정성, 감사 가능성 및 보안을 운영화할 수 있는 상호작용적인 분석 기능을 제공합니다. 본 도구는 모델 개발자와 테스터가 모델 성능을 평가하고 보고할 수 있도록 변수별 강도 및 보안 강도를 분석할 수 있는 기능을 지원합니다. 텍스트 데이터 및 모델에 중점을 둔 초기 버전이 공개되었으며, 향후 다양한 ML 사용 사례를 포함할 계획입니다.



### A Unified Energy Management Framework for Multi-Timescale Forecasting in Smart Grids (https://arxiv.org/abs/2411.15254)
Comments:
          Submitted to PES GM 2025

- **What's New**: 이 논문에서는 Multi-pofo라는 다중 스케일 전력 부하 예측 프레임워크를 제안합니다. 이 프레임워크는 새로운 아키텍처와 시계열 위치 인코딩 계층을 통해 중장기 의존성을 포착합니다. 이로 인해 전력 시스템 관리 및 스마트 그리드 전략을 위한 정확한 예측을 가능하게 합니다. 실세계 전력 부하 데이터에 대한 실험을 통해, 이 방법이 기존의 여러 강력한 기초 방법보다 우수하다는 것을 보였습니다.

- **Technical Details**: Multi-pofo는 세 부분으로 구성된 머신러닝 기반 모델로, 첫 번째 부분인 다중 스케일 임베딩은 스케일 별 특성을 학습하며 통일된 입력 형식을 유지합니다. 두 번째 부분은 다중 스케일 인코더로, 여러 스케일 간 공통 특성을 포착하고 마지막으로 별도의 예측 모듈이 예측을 수행합니다. 이 모델은 데이터의 다양한 스케일을 고려하여 예측을 수행하는 데 적합하며, 기존의 CNN-LSTM 및 BiLSTM보다 뛰어난 성능을 보입니다.

- **Performance Highlights**: Multi-pofo는 다양한 에너지 관리 시나리오를 처리할 수 있는 규모 있는 프레임워크로 설계되었습니다. 이 시스템은 단기 운영 조정부터 장기 전략 계획에 이르기까지 효율적으로 대응할 수 있습니다. 모델의 단순한 다층 퍼셉트론 아키텍처는 계산 복잡성을 줄이면서 최첨단 예측 성능을 달성합니다. 따라서 이 방법은 실제 세계에서의 배치에 적합한 선택으로 보입니다.



### LocRef-Diffusion:Tuning-Free Layout and Appearance-Guided Generation (https://arxiv.org/abs/2411.15252)
- **What's New**: 이 논문에서는 LocRef-Diffusion이라는 새로운 모델을 제안합니다. 이 모델은 개인화된 사용자 정의가 가능한 튜닝 프리(tuning-free) 방식으로, 이미지 내 여러 인스턴스의 외관과 위치를 조정할 수 있습니다. 또한 Layout-net과 Appearance-net이라는 두 가지 핵심 구성 요소를 통해 인스턴스 배치의 정밀성을 높이고, 참조 이미지와의 외관 충실도를 개선합니다.

- **Technical Details**: LocRef-Diffusion은 이미지 생성 과정에서 명시적인 레이아웃 정보와 인스턴스 영역 크로스 어텐션 모듈을 이용하여 객체의 위치를 제어하며, Appearance-Net을 통해 참조 이미지의 외관 특징을 추출합니다. 이러한 기능은 크로스 어텐션 메커니즘을 통해 확장되어 생성된 객체의 참조 이미지에 대한 유사성을 유지하게 합니다. 이를 통해 모델은 고해상도 이미지 생성에서 Layout과 Appearance에 대한 정밀한 제어를 가능하게 합니다.

- **Performance Highlights**: COCO와 OpenImages 데이터셋에서 시행된 실험 결과, LocRef-Diffusion은 레이아웃 및 외관에 기반한 생성에서 최신 기술 대비 뛰어난 성능을 보였습니다. 이 모델은 타겟 인스턴스의 정확한 위치 지정을 지원하며, 기본 모델의 사전 학습된 가중치를 보존하면서 새로운 프로젝션 모듈과 지역화 계층을 통합합니다. 실험에서 나타난 제로샷(zero-shot) 성능은 기존 모델들에 비해 현저히 개선된 결과를 보여줍니다.



### Optimized Vessel Segmentation: A Structure-Agnostic Approach with Small Vessel Enhancement and Morphological Correction (https://arxiv.org/abs/2411.15251)
Comments:
          12 pages, 7 figurres, submitted to TIP

- **What's New**: 본 논문에서는 혈관 세분화의 정확성을 높이는 새로운 프레임워크인 OVS-Net을 제안합니다. 이 모델은 다양한 해부학적 구조에서 최적화된 세분화를 위해 설계된 구조 비지향적 접근 방식을 채택하고 있습니다. OVS-Net은 소규모 혈관 향상 및 형태학적 보정을 포함하여 다변량 혈관 세분화를 수행합니다.

- **Technical Details**: OVS-Net은 매크로 혈관 추출 모듈과 마이크로 혈관 향상 모듈로 구성된 이중 가지 특징 추출 모듈을 설계하였으며, 이는 세분화 마스크 예측 개선을 위해 디코더를 수정합니다. 이러한 설계를 통해 기존의 세분화 모델을 의료 현장에 더욱 적합하게 만들었습니다. 또한, 연결이 끊어진 혈관을 복구하는 토폴로지 복구 네트워크를 추가하여 임상에서의 활용 가능성을 높입니다.

- **Performance Highlights**: 17개의 데이터셋을 활용해 OVS-Net의 성능을 평가한 결과, 세분화 정확성과 일반화 능력이 우수하며, 연결성에서 34.6% 향상을 이뤘습니다. 이는 임상 응용 가능성을 강조하며, 향후 코드는 Github를 통해 공개 예정입니다.



### TPLogAD: Unsupervised Log Anomaly Detection Based on Event Templates and Key Parameters (https://arxiv.org/abs/2411.15250)
- **What's New**: 이번 연구에서는 TPLogAD라는 새로운 보편적 비지도형(log anomaly detection) 로그 분석 방법을 제안합니다. TPLogAD는 이벤트 템플릿과 핵심 매개변수를 기반으로 하는 비지도형 접근 방식을 통해 로그에서의 이례성을 효과적으로 탐지합니다. 기존 방법들은 템플릿 인덱스나 특정 문자열 임베딩을 사용했으나, TPLogAD는 더 풍부한 의미 분석을 가능하게 합니다.

- **Technical Details**: TPLogAD는 itemplate2vec와 para2vec이라는 두 가지 효율적이고 구현이 용이한 의미 표현 방식을 포함하고 있습니다. itemplate2vec는 이벤트 템플릿에서의 의미 정보를 추출하는 데 사용되며, Bert를 이용해 로그 내 단어 간의 의미적 관계를 학습합니다. 반면, para2vec는 로그 내 매개변수에서 의미 정보를 정확하게 추출하기 위해 적응형 클러스터링 알고리즘을 사용합니다.

- **Performance Highlights**: TPLogAD는 네 개의 공개 로그 데이터셋에 대해 기존 로그 이례성 탐지 방법들보다 월등한 성능을 보여주었습니다. 실험 결과, TPLogAD는 로그 탐지 정확성을 크게 향상시켰으며, 다양한 로그 포맷과 동적 데이터를 처리하는 데 있어 탁월한 적응력을 발휘합니다.



### Adversarial Prompt Distillation for Vision-Language Models (https://arxiv.org/abs/2411.15244)
- **What's New**: 이 논문에서는 Adversarial Prompt Distillation (APD)라는 새로운 방어 방법을 제안하여, 기존의 Vision-Language Models (VLM) 모델의 적대적 강인성을 향상시킨다. APD는 Adversarial Prompt Tuning (APT)과 knowledge distillation을 결합하여 CLIP 모델의 성능을 높인다. 특히, APD는 시각 모달과 텍스트 모달 모두에 대해 프롬프트를 추가하여 양 모달 방식(bimodal)으로 작동한다.

- **Technical Details**: APD는 비강인한(non-robust) CLIP 모델을 학생(student) 모델로, 깨끗이 훈련된 선생님(teacher) CLIP 모델을 사용하여 지식을 증류하는 방식으로 동작한다. 이 방법은 시각 모달과 텍스트 모달 모두에서 학습 가능한 프롬프트를 삽입하여 적대적 강인성을 강화한다. 실험을 통해 APD는 기존의 APT 방법들을 초월하는 성능을 보여주었다.

- **Performance Highlights**: APD는 8개의 벤치마크 데이터셋에서 강력한 성능을 발휘하며, 기존의 APT 방법들과 다양한 변형들에 비해 PGD 및 AutoAttack(AA) 공격에 대해 뛰어난 방어 성능을 나타냈다. APD는 비강인한 모델을 사용하여도 질적 정확성과 강인성을 동시에 높이는 가능성을 보여주었고, 이를 통해 VLM의 강인성을 개선하는 데 기여할 수 있음을 입증했다.



### Bio-inspired AI: Integrating Biological Complexity into Artificial Intelligenc (https://arxiv.org/abs/2411.15243)
- **What's New**: 이 논문은 인공지능(AI)의 발전이 생물학적 계산에서의 기본 원칙을 어떻게 응용할 수 있는지 탐구합니다. 특히, 맥락에 따라 달라지는 계층적 정보 처리(hierarchical information processing), 시행착오 발견(trial-and-error heuristics), 그리고 다중 스케일 조직(multi-scale organization)의 중요성을 강조합니다. 생물학적 지능의 미세한 메커니즘을 조사하여 인공지능 시스템 설계에서의 가능성과 한계를 조명하고자 합니다. 결과적으로 생물학적 시스템에서 영감을 받은 더 적응적이고 강력한 AI 시스템을 설계하는 프레임워크를 제안합니다.

- **Technical Details**: AI의 역사적 맥락에서 저자들은 토마스 홉스와 블레즈 파스칼의 기계적 사유 이론(mechanical theory)과 같은 초기 사상가들이 지능형 기계의 꿈을 어떻게 펼쳤는지를 분석합니다. 생물학적 계산 원칙에 따르면, 생물학적 지능은 신경계에 국한되지 않고 단세포 생물이나 식물에서도 정보 처리 및 적응 행동을 보여 줍니다. 자연에서 발견되는 지능은 계층적 상호작용과 다중 스케일에서의 과정들로 형성되며, 이는 AI 시스템 설계에서 중요한 참고자료가 됩니다.

- **Performance Highlights**: 현재 AI 시스템 개발에서 생물학적 영감을 받아 RoboBee와 RoboRay 프로젝트와 같은 새로운 접근 방식이 주목받고 있습니다. 이러한 시스템들은 경량화와 유연성을 추구하며, 뇌와 유사한 에너지 효율(enery efficiency)을 가지려는 뉴로모픽 컴퓨팅(neuromorphic computing) 기술에 의존합니다. 그러나 여전히 많은 생물학적 문제 해결은 신경망을 기반으로 한 접근 방식이 아닌 다른 생물학적 메커니즘에서도 발생하고 있습니다. 이 논문은 그러한 다양한 생물학적 전략들이 AI 설계에 더 많은 통찰력을 제공할 수 있음을 강조합니다.



### The Zamba2 Suite: Technical Repor (https://arxiv.org/abs/2411.15242)
Comments:
          21/11/24 initial upload

- **What's New**: Zamba2 시리즈는 1.2B, 2.7B, 7.4B 파라미터를 가진 하이브리드 Mamba2-Transformer 모델의 집합으로, 최첨단 성능을 제공하면서 추론 지연(inference latency), 처리량(throughput), 메모리 효율성(memory efficiency)에서도 큰 향상을 이루었습니다. 이 모델들은 3조 토큰까지 훈련되었으며, 이전의 Zamba1-7B 작업을 바탕으로 아키텍처와 훈련 데이터를 최적화했습니다. Zamba2 시리즈 모델에 대한 오픈 소스 가중치(weights)와 훈련 데이터셋인 Zyda-2도 공개되었습니다.

- **Technical Details**: Zamba2 아키텍처는 이전 Zamba1-7B 모델에서 도입된 혁신을 바탕으로 아키텍처 개선 과정을 거쳤습니다. Mamba1에서 Mamba2로의 전환을 통해 더 높은 처리량을 달성하였고, 두 개의 교차 공유 어텐션 블록(shared attention blocks)을 도입하여 성능을 개선했습니다. 또한, 저순위 어댑터(Low-Rank Adapters)와 Rotary Position Embeddings를 적용하여 모델의 표현력을 증가시켰습니다.

- **Performance Highlights**: Zamba2 시리즈 모델은 언어 모델링 평가에서 최첨단 성능을 달성하며, 인퍼런스와 메모리 효율성 면에서도 우수합니다. 동급의 트랜스포머 모델과 비교했을 때, 최대 30-50%의 첫 토큰 처리 시간 단축과 6배의 KV 캐시 메모리 요구량 감소를 자랑합니다. 이러한 성능 향상은 특히 리소스가 제한된 장치에서 모델을 실행하는 데 매우 유리합니다.



### Is Attention All You Need For Actigraphy? Foundation Models of Wearable Accelerometer Data for Mental Health Research (https://arxiv.org/abs/2411.15240)
- **What's New**: 이 논문에서는 Pretrained Actigraphy Transformer (PAT)라는 새로운 모델을 소개하고 있습니다. PAT는 1970년대부터 사용된 wearable accelerometry (actigraphy) 데이터를 활용하여 처음으로 설계된 프리트레인(pretrained)된 완전한 어텐션 기반 (attention-based) 모델입니다. 이 모델은 NHANES 데이터에서 29,307명 참가자의 데이터를 학습하여 다양한 정신 건강 관련 작업에서 최첨단 성능을 제공합니다.

- **Technical Details**: PAT는 200만 개 미만의 파라미터로 구성되어 있으며, 쉽게 배포할 수 있는 특징을 가지고 있습니다. 이 모델은 actigraphy 데이터를 효과적으로 처리하기 위해 설계되었으며, 특히 데이터가 제한된 상황에서도 높은 성능을 유지합니다. 예를 들어, 500명의 레이블된 참가자 데이터를 기반으로 benzodiazepine 사용을 예측하는 작업에서 뛰어난 결과를 보여주었습니다.

- **Performance Highlights**: PAT는 기존 모델에 비해 8.8%의 AUC 개선을 달성하였습니다. 이는 actigraphy를 활용하여 정신 건강 관련 측정치를 예측하는 데 있어 매우 중요한 성과입니다. 또한 모델의 내장된 설명 가능성(model explainability)은 건강 연구 환경에서의 활용도를 더욱 높입니다.



### Stain-Invariant Representation for Tissue Classification in Histology Images (https://arxiv.org/abs/2411.15237)
- **What's New**: 이번 논문에서는 다수의 요인들이 전체 슬라이드 이미지(Whole Slide Image, WSI)의 최종 외관에 영향을 미치는 병리학적 슬라이드의 디지털화 과정에 대해 다룹니다. 특히, 염색 프로토콜, 스캐너, 조직 유형의 다양성이 심층 학습(Deep Learning, DL) 알고리즘의 다중 집단 설정에서 훈련 및 테스트 시 문제를 일으킬 수 있음을 강조합니다. 이를 해결하기 위해 우리는 염색 매트릭스 교란(stain matrix perturbation)을 활용하여 훈련 이미지의 염색 보강 버전을 생성하는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법에서는 염색 정규화 손실(stain regularisation loss)을 적용하여 원본 이미지와 보강 이미지 간의 특징 표현(feature representations) 간 일관성을 확보합니다. 이렇게 함으로써 모델은 염색 불변(stain-invariant) 및 도메인 불변(domain-invariant) 특징 표현을 학습하도록 유도됩니다. 논문에서는 이 모델을 사용하여 대장암 이미지의 교차 도메인 멀티 클래스 조직 유형 분류에서 평가하였으며, 다른 최첨단 방법들에 비해 개선된 성능을 보여주었습니다.

- **Performance Highlights**: 제안된 모델은 대장암의 다양한 조직 유형을 분류하는 데 있어 우수한 성능을 나타냅니다. 이를 통해 어려운 도메인 전이 문제를 극복하고, 기존의 방법에 비해 모델의 일반화 가능성을 높이는 데 성공했습니다. 이 연구는 계산 병리학(Computational Pathology, CPath) 분야에서 강건하고 일반화 가능한 DL 모델 개발의 필요성을 잘 보여줍니다.



### CODE-CL: COnceptor-Based Gradient Projection for DEep Continual Learning (https://arxiv.org/abs/2411.15235)
Comments:
          10 pages, 2 figures

- **What's New**: 최근 연구에서는 인공 신경망이 연속적 학습을 할 때 발생하는 재난적 망각(catastrophic forgetting) 문제를 해결하기 위해, CODE-CL이라는 새로운 방법을 소개했습니다. 이 방법은 신경과학에서 영감을 받은 개념자(matrix) 표현을 활용하여, 과거 태스크에 대한 중요성을 코드화하고 새 정보를 유연하게 통합할 수 있게 합니다. 이를 통해 상관관계가 높은 태스크간의 효율적인 정보 이전이 가능하고, 기존 지식을 상당히 방해하지 않고도 새로운 지식 습득이 가능합니다.

- **Technical Details**: CODE-CL은 입력 공간에서 과거 태스크의 방향성을 인코딩하여 $1-S$의 방향으로 새 지식을 통합할 수 있도록 합니다. 여기서 $S$는 이전 태스크와 관련된 방향의 중요성을 나타냅니다. 또한, 개념자 기반 표현을 사용하여 태스크 간의 겹침을 분석하고, 이는 상위 공간에서의 축척 투영을 통해 효율적인 지식 전달을 가능하게 합니다. 이러한 접근 방식은 이전의 중요한 지식을 손상시키지 않고 상관관계가 높은 태스크들 간의 학습을 가능하게 합니다.

- **Performance Highlights**: CODE-CL은 Permuted MNIST, Split CIFAR100, Split miniImageNet, 5-Datasets와 같은 연속 학습 이미지 분류 벤치마크에서 광범위한 실험을 진행하여 효과성을 입증했습니다. 실험 결과, CODE-CL은 많은 최첨단 방법들보다 우수한 성능을 보여주며, 기억 저하를 최소화하면서도 뛰어난 성과를 달성했습니다. 이 연구는 메모리 효율적이며 과거의 지식을 유지하면서도 새로운 정보를 유연하게 습득할 수 있는 접근 방식을 제시합니다.



### Adaptive Intelligence: leveraging insights from adaptive behavior in animals to build flexible AI systems (https://arxiv.org/abs/2411.15234)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문에서는 생물학적 지능을 활용하여 더 적응적인 인공지능(adaptive intelligence)을 개발하는 새로운 접근법을 제안합니다. 기존의 인공지능 시스템을 넘어, 진화된 AI를 만드는 방법으로 동물의 학습과 적응 방식을 모방하는 것이 주요 초점입니다. 최근 신경과학 연구의 발전은 동물들이 자연스럽게 학습하는 과정에서 얻은 통찰력을 바탕으로 AI를 발전시킬 수 있는 기반을 제공합니다.

- **Technical Details**: 이 논문은 적응적인 생물학적 지능의 행동 및 신경 기초를 검토하고, 이에 대한 AI의 발전 상황과 더불어 뇌에서 영감을 받은 알고리즘 발전 방안을 다룹니다. 저자는 행위를 추적하는 새로운 기술 및 다양한 자극을 사용하는 실험을 통해 수의학적 연구에서 동물의 학습 방식을 실증적으로 보여줍니다. 또한, 센서 및 운동 제어에서의 빠른 학습 과정을 다루며, 특정 세포 유형의 구조적 중요성에 대해서도 언급합니다.

- **Performance Highlights**: 저자들은 동물들이 복잡한 공간에서 적은 학습만으로 새 작업을 수행하는 능력을 보였다는 것을 강조합니다. 예를 들어, 생쥐는 미로에서 2000번 이상의 결정을 내릴 수 있었으며, 20번의 시도만으로 목표 위치를 알아내는 제로샷 학습(zero-shot learning) 능력을 보여주었습니다. 더 나아가, 뇌-기계 인터페이스(brain-machine interface, BMI) 연구를 통해 몇몇 뉴런이 외부 세계에서의 변화를 위해 어떻게 적응하는지를 보여주는 강력한 증거도 발견되었습니다.



### IterIS: Iterative Inference-Solving Alignment for LoRA Merging (https://arxiv.org/abs/2411.15231)
- **What's New**: 이번 논문에서는 IterIS라는 새로운 최적화 기반 방법을 제안합니다. IterIS는 여러 LoRA를 통합하여 단일 어댑터로 만드는 효율적인 방법으로, 데이터 프라이버시와 지적 재산권 문제를 고려합니다. 기존의 매개변수 효율적인 파인튜닝 방법이 가지고 있는 한계를 극복하며, 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: IterIS는 LoRA 병합을 진보된 최적화 문제로 공식화하고, 반복적인 추론 문제 해결 프레임워크를 사용합니다. 이 알고리즘은 객체를 점진적으로 정제하여 성능을 향상시키며, 1-5%의 미레이블 샘플로도 효과적으로 작동합니다. 또한, 적응형 가중치를 활용하여 LoRA 병합 과정에서 발생할 수 있는 불균형 문제를 완화합니다.

- **Performance Highlights**: IterIS는 텍스트-이미지 확산, 비전-언어 모델, 대형 언어 모델 등의 다양한 작업에서 이전 방법들보다 현저한 성능 향상을 보여주었습니다. 메모리 및 계산 효율성 또한 보장하며, 최소한의 반복으로 수렴할 수 있는 장점이 있습니다. 이러한 특성을 통해 IterIS는 LoRA 병합의 새로운 기준을 제시합니다.



### Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation (https://arxiv.org/abs/2411.15224)
- **What's New**: 본 논문에서는 Mamba 아키텍처를 위한 새로운 매개변수 효율적 파인튜닝(PEFT) 방법인 Projector-targeted Diagonal-centric Linear Transformation, 즉 ProDiaL을 소개합니다. 연구 결과, Mamba 아키텍처에서 Transfer Learning에 가장 큰 기여를 하는 것은 State-Space Models(SSMs)이 아닌 Projectors라는 것을 밝혀냈습니다. ProDiaL은 사전학습된 Projector의 가중치를 직접 수정하는 대신, 대각 중심의 선형 변환 행렬만을 최적화함으로써 효율적인 작업 적응을 가능하게 합니다.

- **Technical Details**: ProDiaL은 사전학습된 Projectors를 새로운 작업에 적응시키기 위해 대각행렬과 오프 대각 행렬로 분해된 변환 행렬을 사용합니다. 이 과정은 학습할 수 있는 매개변수를 최소화하고, 신뢰할 수 있는 유연성을 유지하는데 중점을 둡니다. ProDiaL는 Mamba 아키텍처에 최적화되어 있으며, Mamba 기반의 대형 언어 모델(LLMs) 및 비전 모델에 호환 가능합니다.

- **Performance Highlights**: 실험 결과, ProDiaL 방식은 전체 모델 매개변수의 1% 미만을 사용하여도 뛰어난 성능을 발휘하여, Mamba 기반의 비전 및 언어 모델에서 작업 적응에 필수적임을 입증했습니다. 또한 ProDiaL 방법은 기존 PEFT 방법들보다 유의미한 성능 개선을 보였습니다. 이러한 결과들은 Mamba 아키텍처 내에서 Projectors의 중요성을 더욱 강조하여, 향후 연구 방향 설정에 중요한 기초 자료를 제공합니다.



### Rethinking the Intermediate Features in Adversarial Attacks: Misleading Robotic Models via Adversarial Distillation (https://arxiv.org/abs/2411.15222)
- **What's New**: 본 논문은 언어 조건 로봇 모델에 대한 혁신적인 적대적 프롬프트 공격(adversarial prompt attack)을 제안합니다. 기존의 공격 기법들이 로봇 도메인에서 효과를 발휘하지 못하는 이유를 분석하고, 이를 해결하기 위해 연속 동작 표현에 기반하여 적대적 프롬프트를 최적화하는 방법을 제시합니다. 또한, 중간 기능(intermediate features)의 긍정적인 영향을 활용하여 공격 효율성을 향상시키는 접근법을 적용하였습니다.

- **Technical Details**: 적대적 공격은 입력 데이터를 미세하게 조작하여 로봇 모델이 예기치 않은 동작을 수행하게 만드는 기술입니다. 이 논문에서는 VIMA 모델을 대상으로 13가지 로봇 조작 작업에서 실험을 진행하였으며, 연속 행동 벡터(continuous behavior vectors)에 중점을 두어 적대적 언어 접두사(adversarial language prefixes)를 생성하는 방식으로 로봇 시스템의 내성을 우회합니다. 중간 자기 주의 기능(intermediate self-attention features)의 음의 기울기를 활용하여 더욱 효과적인 공격을 구현하였습니다.

- **Performance Highlights**: 확장된 실험을 통해 본 방법이 기존의 최첨단 기술에 비해 월등한 성능을 보임을 입증하였습니다. 언어 조건 모델에 대한 공격의 효과성과 이식성이 강조되며, 다양한 모델 변종에 대한 전이 가능성을 확인하였습니다. 본 연구는 언어 기반 로봇 학습의 보안 취약점을 해결하기 위한 새로운 길을 제시하고 있습니다.



### Suspected Undeclared Use of Artificial Intelligence in the Academic Literature: An Analysis of the Academ-AI Datas (https://arxiv.org/abs/2411.15218)
Comments:
          24 pages, 8 figures

- **What's New**: 이 논문에서는 OpenAI의 ChatGPT와 같은 생성형 인공지능(AI) 도구들이 연구논문 작성 프로세스에 사용됨에 따라, 학술 출판 커뮤니티에서 투명성과 책임성을 위한 새로운 합의가 이루어졌다고 설명합니다. 특히, AI를 사용한 저자는 논문에 그 사실을 명시해야 한다는 원칙이 수립된 것을 강조합니다. 500개의 의심스러운 AI 텍스트 사례를 분석한 결과, 이러한 문제는 저명한 학술 저널에서도 광범위하게 나타나고 있음을 보여줍니다.

- **Technical Details**: 저자들은 AI가 생성한 텍스트의 특징적인 문구를 이용하여 Google Scholar에서 검색을 수행하였습니다. 의심스러운 문구가 포함된 기사를 수집하기 위한 방법론이 상세히 설명되고, 수집된 사례들은 Markdown 파일에 저장되었습니다. 또한, 이러한 자료는 다양한 데이터베이스에서 수집된 메타데이터를 기반으로 비교 분석되었으며, 논문 처리 비용(APC) 등의 정보도 수집되었습니다.

- **Performance Highlights**: 많은 국제 저널과 학술 출판물에서 AI 생성 텍스트가 포함된 기사가 게재되었고, 이는 저널의 인용 지수와 품질과 관련이 깊은 문제로 나타났습니다. 특히, AI의 미신고 사용 사례는 고품질 저널에서 더욱 빈번하게 발생하고 있음을 나타냅니다. 마지막으로, 저자들은 이러한 문제를 해결하기 위해 출판사들이 미신고된 AI 사용에 대해 정책을 강화해야 한다고 주장하고 있습니다.



### LPLgrad: Optimizing Active Learning Through Gradient Norm Sample Selection and Auxiliary Model Training (https://arxiv.org/abs/2411.15217)
- **What's New**: 이 논문은 Loss Prediction Loss with Gradient Norm (LPLgrad)라는 새로운 능동 학습(active learning) 방법을 제안합니다. LPLgrad는 모델의 불확실성을 효과적으로 정량화하고 이미지 분류 작업의 정확성을 향상시키기 위해 설계되었습니다. 이전의 방법들이 라벨된 세트에서의 훈련과 새로운 비라벨 샘플 쿼리의 핵심 정보 활용에 부족했던 문제를 해결하고자 합니다.

- **Technical Details**: LPLgrad는 두 가지 주요 단계로 구성됩니다: (i) 훈련 단계에서는 주 모델과 보조 모델이 함께 훈련되어 입력 특성에 대한 손실을 예측합니다. 이 과정에서 라벨된 데이터를 최대한 효율적으로 활용하여 원래 학습 프로세스에서 간과된 부분을 보완합니다; (ii) 쿼리 단계에서는 비라벨 데이터셋의 엔트로피 값의 그래디언트 노름을 계산하여 주 모델의 불확실성을 정량화하고 레이블링할 샘플을 선택합니다.

- **Performance Highlights**: 실제 데이터셋에서의 광범위한 평가 결과, LPLgrad 접근법은 소수의 라벨된 이미지에서 정확도 면에서 최신의 능동 학습 방법들을 압도적으로 초월하는 성능을 보였습니다. 또한, 여러 이미지 분류 작업에서 훈련 및 쿼리 시간이 유사하게 유지되면서도 더 나은 성능을 달성했습니다.



### Dist Loss: Enhancing Regression in Few-Shot Region through Distribution Distance Constrain (https://arxiv.org/abs/2411.15216)
- **What's New**: 이 논문은 불균형 데이터 분포가 전통적인 딥러닝 모델의 성능에 미치는 영향을 해결하기 위해 새로운 손실 함수인 Dist Loss를 제안합니다. 이 손실 함수는 모델의 예측 분포와 레이블 분포 간의 차이를 줄여주며, 특히 드문 샘플이 중요한 분야인 의료 분야에서의 부정확한 예측 문제를 완화합니다. 저자들은 IMDB-WIKI-DIR, AgeDB-DIR, ECG-Ka-DIR의 세 가지 데이터셋에서 실험을 수행하여 Dist Loss의 효용성을 입증하였습니다.

- **Technical Details**: Dist Loss의 구현은 크게 세 단계로 나뉘어집니다: (1) 커널 밀도 추정(KDE)을 활용하여 레이블의 확률 분포를 모델링하고 여기에 기반해 의사 레이블을 생성합니다; (2) 모델의 예측을 정렬하여 예측 분포를 반영하는 의사 예측을 생성합니다; (3) 의사 레이블과 의사 예측 간의 거리를 측정하여 예측과 레이블 분포 사이의 거리를 근사합니다. 이러한 과정을 통해 모델의 예측 분포를 레이블 분포에 맞춰 조정합니다.

- **Performance Highlights**: 실험 결과, Dist Loss는 드문 샘플에 대한 정확도를 크게 향상시켜 SOTA 성능을 달성하였습니다. 또한 Dist Loss는 기존의 방법들과 통합하여 더욱 개선된 효과를 보였습니다. 이러한 성과는 불균형 회귀 문제에서의 예측 정확도를 높이는 데 znacząco 도움이 됩니다.



### S$^2$ALM: Sequence-Structure Pre-trained Large Language Model for Comprehensive Antibody Representation Learning (https://arxiv.org/abs/2411.15215)
- **What's New**: 본 논문은 Sequence-Structure multi-level pre-trained Antibody Language Model (S$^2$ALM)을 제안합니다. S$^2$ALM은 항체의 1D 서열과 3D 구조 정보를 통합하여 보다 포괄적인 항체 기본 모델을 구축합니다. 이는 기존의 항체 모델들이 구조 정보를 충분히 반영하지 못했던 점을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: S$^2$ALM은 7500만 개의 1D 서열과 1170만 개의 3D 구조로 구성된 데이터셋에서 사전 훈련되었습니다. 이 모델은 Sequence-Structure Matching (SSM)과 Cross-Level Reconstruction (CLR)이라는 두 가지 사용자 정의 훈련 목표를 포함하는 계층적 사전 훈련 패러다임을 사용하여 항체에 대한 포괄적인 표현을 모델링합니다. 이를 통해 생물학적 언어를 해독하는 데 필수적인 구조적 정보를 효과적으로 주입합니다.

- **Performance Highlights**: S$^2$ALM은 항체 바인딩 친화도 예측, B 세포 성숙 분석, 항체 중요한 결합 위치 식별 및 코로나 바이러스 결합 항체 설계와 같은 다양한 다운스트림 작업에서 뛰어난 성능을 나타냅니다. 본 모델은 기존의 기준 모델을 초과하는 성과를 달성하며, 실세계 치료 항체 개발에 기여할 잠재력을 보여줍니다.



### Urban Region Embeddings from Service-Specific Mobile Traffic Data (https://arxiv.org/abs/2411.15214)
- **What's New**: 이 논문은 4G/5G 모바일 네트워크의 발전으로 수집된 서비스별 모바일 트래픽 데이터를 활용하여 고품질의 도시 영역 표현을 생성할 수 있는 잠재력을 탐구합니다. 특히, 저자들은 서비스별 모바일 트래픽 데이터를 사용하여 도시 영역 임베딩(embeddings)을 생성하는 방법론을 제시하고, 이를 통해 도시의 핵심 특징을 포착합니다. 본 연구는 넷모브(NetMob) 2023 데이터셋을 바탕으로 실제 도시들의 임베딩을 생성하고 평가하여 기존 방법들과 비교합니다.

- **Technical Details**: 저자들은 서비스별 모바일 트래픽 데이터를 바탕으로 도시 영역의 임베딩을 생성하기 위해 시간 합성곱 신경망(Temporal Convolutional Network) 기반의 오토인코더(autoencoder), 트랜스포머(transformers), 그리고 학습 가능한 가중합 모델(learnable weighted sum models)을 활용합니다. 이러한 접근 방식은 도시의 시간적 역학과 특성을 포착할 수 있도록 설계되었습니다. 또한, 두 개의 감독 학습(task)과 하나의 비감독 학습(task)을 통해 생성된 임베딩의 효과성을 평가합니다.

- **Performance Highlights**: 실험 결과, 서비스별 모바일 트래픽 데이터를 사용하여 생성한 임베딩이 기존의 전통적인 데이터 소스에서 파생된 것들보다 도시의 특성을 보다 효과적으로 포착함을 보여줍니다. 특히 비감독 학습 평가에서는 데이터가 도시의 시간적 동역학을 잘 포착함을 입증하며, 데이터의 지속적인 변화를 나타냅니다. 마지막으로, 이 연구는 모바일 트래픽 데이터의 공개 접근성이 도시 연구와 혁신을 위한 중요한 자원이 될 수 있음을 강조하고 있습니다.



### Effective Analog ICs Floorplanning with Relational Graph Neural Networks and Reinforcement Learning (https://arxiv.org/abs/2411.15212)
Comments:
          7 pages, 7 figures, Accepted at DATE25

- **What's New**: 본 연구에서는 강화 학습(Reinforcement Learning, RL) 기반의 자동화된 아날로그 집적 회로(Integrated Circuit, IC) 레이아웃의 바닥 배치(floorplanning) 알고리즘을 제안합니다. 또한, 회로 특성과 위치 제약 조건을 인코딩하는 멀티모달 관계 그래프 컨볼루션 신경망(Relational Graph Convolutional Neural Network, R-GCN) 모델로 강화하여 다양한 회로 디자인 간 지식 전달 능력을 높였습니다. 이를 통해 수동 설계에 비해 우수한 시간 효율성과 면적 절감 효과를 실현했습니다.

- **Technical Details**: 본 논문은 강화 학습 기법을 적용하여 아날로그 회로의 최적 바닥 배치를 생성하는 새로운 방법론을 제안합니다. R-GCN 모델은 회로, 장치 및 기하학적 제약을 예측하여 RL 에이전트에게 최적의 배치 결정을 지원합니다. 이 접근 방식은 CNN(Convolutional Neural Network)과의 결합을 통해 각 구성 요소의 최적 형태와 위치를 결정하는 데 도움을 줍니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 6개의 산업 회로에서 기존의 바닥 배치 기술을 초과하는 성능을 보였습니다. 전반적인 레이아웃 시간은 67.3% 단축되었으며, 평균 면적은 8.3% 감소하였습니다. 이러한 결과는 전체 설계 시간 단축을 가능하게 하면서도 수동으로 설계한 레이아웃의 품질을 유지하거나 초과하는 효과를 입증합니다.



### LightLLM: A Versatile Large Language Model for Predictive Light Sensing (https://arxiv.org/abs/2411.15211)
Comments:
          15 pages, 14 figures, 5 tables

- **What's New**: 이번 논문은 LightLLM이라는 모델을 제안하며, 이는 전이 학습된 대형 언어 모델(LLM)을 가볍게 미세 조정하여 센서 기반의 조명 감지 작업을 수행합니다. LightLLM은 센서 데이터 인코더, 상황적 프롬프트 및 융합 레이어를 통합하여 입력을 통합된 표현으로 결합하고 있습니다. 이를 통해 모델은 원래 파라미터를 변경하지 않고도 새로운 작업에 잘 적응할 수 있도록 설계되었습니다.

- **Technical Details**: LightLLM은 다양한 데이터를 처리하기 위해 세 가지 주요 구성 요소를 통합합니다: 작업에 특화된 센서 데이터 인코더, 상황 인식을 위한 자연어 프롬프트, 센서 데이터와 프롬프트를 결합하는 융합 레이어입니다. LoRA(저랭크 적응) 기법을 사용하여 LLM을 효율적으로 미세 조정하며, 이는 기존 LLM의 지식을 활용하면서도 도메인 특화된 데이터를 포함할 수 있게 합니다.

- **Performance Highlights**: LightLLM은 세 가지 조명 감지 작업인 실내 위치 추적, 외부 태양광 예측 및 실내 태양광 추정에 대해 실제 실험 데이터를 사용하여 검증되었으며, 기존 최신 방법들보다 현저하게 우수한 성능을 보였습니다. 예를 들어, 실내 위치 추적 정확도가 4.4배, 실내 태양광 추정에서 3.4배의 개선을 이루었습니다. 또한 LightLLM은 ChatGPT-4보다 직접 프롬프트를 통해 더 나은 성능을 나타내, 센서 데이터 융합을 위한 전문화된 아키텍처의 장점을 부각시켰습니다.



### Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks (https://arxiv.org/abs/2411.15210)
- **What's New**: 이 논문은 이미지 분류 모델에 대한 새로운 개별 공격 방식인 Probability Margin Attack (PMA)을 제안합니다. PMA는 로짓(logits) 공간이 아닌 확률 공간에서 적대적 마진(adversarial margin)을 정의하여 기존의 공격 방식보다 효과적인 평가를 가능하게 합니다. 또한, CC3M 데이터셋에서 유래한 백만 규모의 데이터셋 CC1M을 구축하여 대규모 안정성 평가를 진행했습니다.

- **Technical Details**: PMA는 확률 마진 손실(probability margin loss)을 도입하여 공격의 효율성을 높이는데 집중합니다. 논문에서는 PMA와 기존의 cross-entropy, margin 손실 간의 관계를 분석하며, PMA가 현재의 개별 공격 방식보다 우수하다고 입증합니다. 또한, PMA를 기반으로 한 두 가지 앙상블 공격 방법도 제안하여 효율성과 효과성의 균형을 맞췄습니다.

- **Performance Highlights**: PMA는 CIFAR-10, CIFAR-100, ImageNet-1K 등 여러 데이터셋에서 기존의 개별 공격 방식보다 뛰어난 성능을 보였습니다. 또한, 백만 규모의 평가 데이터셋 CC1M을 활용한 실험에서는 기존의 소규모 평가 방법과 비교할 때 큰 안정성 차이를 발견하였습니다. 이러한 결과는 개별 공격과 앙상블 공격 간의 효과의 차이를 보여줍니다.



### M2oE: Multimodal Collaborative Expert Peptide Mod (https://arxiv.org/abs/2411.15208)
Comments:
          accepted by bibm 2024

- **What's New**: 본 논문에서는 M2oE(멀티모달 협업 전문가) 모델을 제안하여 다양한 서열(sequence)과 공간 구조 구조 정보를 통합하는 새로운 접근 방식을 탐구하고 있습니다. 특히, 단일 모달링 모델이 정보가 부족한 데이터셋을 다룰 때 효율적이지 않다는 점을 지적하며 이를 개선하기 위한 다중 모달 시스템의 필요성을 강조합니다. 또한, 전문가 모델 및 교차 주의 메커니즘(Cross-Attention Mechanism)을 활용하여 모델의 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: 연구에서는 Liu et al.의 벤치마크 데이터셋을 사용하여 분류 및 회귀 작업에 대한 실험을 수행했습니다. 모델은 Transformer 아키텍처와 그래프 신경망(Graph Neural Networks, GNN)을 활용하여 아미노산의 서열 정보와 분자의 공간 구조 정보를 동시에 처리할 수 있도록 설계되었습니다. 또한, 다중 모달 특성을 활용하여 혼합 전문가 표현(improved mixed expert representation)을 구현하고, 학습 가능한 가중치(learnable weights)를 통해 각 데이터 배포 시퀀스와 공간 정보의 중요성을 평가합니다.

- **Performance Highlights**: M2oE 모델은 복잡한 작업 예측에서 두드러진 성능을 보였으며, 기존의 모델들보다 더 높은 정확도를 기록했습니다. 연구 결과는 해당 모델이 실험에서 서로 다른 데이터셋에 대해 우수한 일반화 능력을 보여줬음을 나타냅니다. 또한, 이 모델은 전문가 네트워크를 통해 다중 모달 데이터를 효과적으로 처리할 수 있기 때문에 향후 생물학적 데이터 예측 작업에 있어 중요한 이정표가 될 것입니다.



### Uni-Mlip: Unified Self-supervision for Medical Vision Language Pre-training (https://arxiv.org/abs/2411.15207)
Comments:
          15 pages, 2 figures, accepted by BMVC'24

- **What's New**: 최근 비전-언어 사전 학습(Vision-and-Language Pre-training) 기술의 발전이 컴퓨터 비전 분야의 성능을 크게 향상시켰습니다. 그러나 의료 분야에서는 다중 모달 데이터를 얻는 것이 비용이 많이 들고 복잡하며, 개인정보 보호 등 여러 어려움이 있습니다. 이를 해결하기 위해 Uni-Mlip라는 새로운 프레임워크를 소개하며, 이 프레임워크는 의료 비전-언어 사전 학습을 위한 통합 자기 지도화(self-supervision) 접근 방식을 제공합니다.

- **Technical Details**: Uni-Mlip은 데이터 수준과 특징 수준 모두에서 교차 모달(cross-modality), 단일 모달(uni-modality), 융합 모달(fused-modality) 자기 지도화 기법을 통합하여 의료 이미지를 효과적으로 처리합니다. 특히, 의료 이미지의 특성에 맞춘 단일 모달 이미지 자기 지도화 기법을 조정하여 높은 정밀도와 자세한 감도를 제공합니다. 이러한 접근 방법을 통해 의료 데이터를 더 효과적으로 발굴하고, 모델의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 다양한 규모의 데이터세트에 대한 실험 결과, Uni-Mlip는 이미지-텍스트 검색(image-text retrieval), 이미지 분류(image classification), 시각적 질의 응답(Visual Question Answering, VQA)과 같은 주요 다운스트림 작업에서 현재의 최첨단 방법들을 능가하는 뛰어난 성능을 보여줍니다. 이 성능 향상은 모델의 특징과 데이터 간의 정렬 및 전이 가능성을 크게 개선하는 데 기여하였습니다.



### Self-Supervised Conditional Distribution Learning on Graphs (https://arxiv.org/abs/2411.15206)
Comments:
          8 pages

- **What's New**: 이 논문에서는 그래프 구조의 데이터로부터 반지도 그래프 분류를 위한 자기 지도 조건 분포 학습(self-supervised conditional distribution learning, SSCDL) 방법을 제안합니다. 기존의 그래프 대조 학습(graph contrastive learning, GCL)은 노드 임베딩 간의 dissimilarity를 높이는 것을 목표로 하나, GNN의 message-passing 메커니즘과 충돌을 일으키는 문제가 있었습니다. 또한, 강한 증강(strong augmentation)과 약한 증강(weak augmentation) 간의 조건 분포 일치를 효과적으로 수행하여 내재적 의미 정보를 보존하는 것이 과제가 되었습니다.

- **Technical Details**: SSCDL 방법은 그래프 임베딩의 조건 분포를 원래의 특성 위에 맞추는 방식으로 구성된 end-to-end 그래프 표현 학습 모델을 포함합니다. 이에 따라 약한 및 강한 증강의 조건 부여된 분포를 정렬함으로써, 강한 증강을 적용할 때 내재적 의미 정보를 손상하는 위험을 줄일 수 있습니다. 이 모델은 GNN의 message-passing 메커니즘과 GCL의 노드 표현 간의 대조 학습의 충돌을 피하기 위해 긍정 쌍(positive pairs)의 노드 표현만 유지합니다.

- **Performance Highlights**: 제안하는 SSCDL 방법은 여러 벤치마크 그래프 데이터셋을 통해 광범위한 실험을 수행했으며, 기존의 최첨단 그래프 분류 방법들과 높은 경쟁력을 보였습니다. 이를 통해 SSCDL 방법이 반지도 그래프 분류에서 효과적으로 작동할 수 있음을 입증하였습니다. 따라서 이 연구는 관계가 복잡한 그래프 데이터를 다루는 다양한 애플리케이션에 대한 새로운 가능성을 제공합니다.



### Label Distribution Shift-Aware Prediction Refinement for Test-Time Adaptation (https://arxiv.org/abs/2411.15204)
- **What's New**: 이번 연구에서 제안된 DART 방법은 테스트 시 레이블 분포 변화에 효과적으로 대응할 수 있는 새로운 TTA 방법이다. 기존의 TTA 방법들은 레이블 분포가 변화할 경우 성능 저하를 겪는 경향이 있었으나, DART는 이러한 문제를 해결하는 데 중점을 두고 있다.

- **Technical Details**: DART는 클래스 간 혼동 패턴을 활용해 BNAdapt의 예측을 개선하는 방식으로 작동한다. 이 방법은 훈련 데이터셋에서 다양한 클래스 분포에 노출되어 예측 정제 모듈을 학습하고, 이후 테스트 데이터의 레이블 분포 변화를 감지한다.

- **Performance Highlights**: CIFAR-10C와 같은 다양한 벤치마크에서 DART는 5-18%의 정확도 향상을 보여주며, 기존의 TTA 방법들의 성능 또한 크게 개선되었다. 특히, DART는 BNAdapt와 결합하여 성능을 극대화할 수 있는 유용한 플러그인 도구로 자리잡았다.



### Multimodal large language model for wheat breeding: a new exploration of smart breeding (https://arxiv.org/abs/2411.15203)
- **What's New**: 이 연구는 UAV 원격 감지 기술을 활용하여 작물 육종에 필요한 데이터 수집의 비파괴적이고 고속화를 달성하는 새로운 접근 방식을 제시합니다. 특히, 다양한 사전 훈련된 오픈 소스 다중 모달 대형 언어 모델(MLLMs)을 기반으로 하여 스마트 육종 목표 도구를 개발했습니다. 이 도구는 다중 분야의 데이터 마이닝을 가능하게 합니다.

- **Technical Details**: 이 연구는 감독된 세밀조정(Supervised Fine-tuning, SFT), 검색 증강 생성(Retrieval-augmented Generation, RAG), 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 기술을 이용하여 MLLMs에 다중 분야의 지식을 주입했습니다. 이를 통해 밀 생산량 예측을 위한 다수의 다중 모달 대형 언어 모델(Wheat Breeding Language Models, WBLMs)을 구축하였고, 새롭게 생성된 평가 기준을 사용하여 이 모델을 평가했습니다.

- **Performance Highlights**: 연구 결과, InternVL2-8B를 사용하여 SFT, RAG 및 RLHF 기술로 구축된 WBLM은 매우 우수한 성능을 보였습니다. 특히, WBLM은 원격 감지, 표현형 데이터, 날씨, 유전자원 데이터를 동시에 활용하여 밀 생산량을 예측할 때 R2 값이 0.821, RMSE가 489.254 kg/ha로 나타났습니다. 또한, WBLM은 환경 스트레스 평가, 목표 유전자원 선별, 재배 기술 추천 등 다양한 전문적 의사결정 지원 답변을 생성할 수 있습니다.



### Beyond Visual Understanding: Introducing PARROT-360V for Vision Language Model Benchmarking (https://arxiv.org/abs/2411.15201)
Comments:
          7 pages, 4 figures, Accepted at COLING 2025

- **What's New**: 새로운 PARROT-360V 벤치마크는 복잡한 시각적 퍼즐을 통해 비전 언어 모델(VLM)의 능력을 평가하는 혁신적인 방법입니다. 2487개의 도전적 퍼즐로 구성되어 있으며, 이 모델들이 언어 능력과 시각적 단서를 결합하여 문제를 해결하는 데 있어서 인간의 문제 해결 방식과 유사한지 평가하고자 합니다. 또한, 최신 VLM 모델들인 GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro의 성능을 비교하여 복잡한 추론 작업에서의 한계를 드러냅니다.

- **Technical Details**: PARROT-360V는 비전 언어 모델이 이미지와 텍스트 데이터를 통합하는 능력을 평가하기 위해  복잡한 시각적 추론을 요구하는 퍼즐을 중심으로 설계되었습니다. 기존의 벤치마크들이 단순한 이미지-텍스트 정렬이나 단일 단계 추론에 중점을 두었다면, 이 벤치마크는 언어 이해뿐만 아니라 시각적 인식과 추론 능력에 대한 실질적인 평가를 목표로 하고 있습니다. 데이터 전처리나 주석의 변동성으로 인한 재현성 문제를 해결하기 위해, 모델이 시각적 단서를 통합하여 단계별로 문제를 해결해야 합니다.

- **Performance Highlights**: VLM 모델들은 PARROT-360V 벤치마크에서 28%에서 56% 사이의 점수로 확인되었으며, 이는 기존 인기 있는 벤치마크들과 비교할 때 현저히 낮은 성과입니다. 이 결과는 현재 VLM들이 복잡한 다단계 추론 작업을 수행하는 데 있어 뚜렷한 한계를 지니고 있음을 강조합니다. 따라서 PARROT-360V는 비전 언어 모델의 실제 성능을 측정하고, 그 모델들이 진정한 문제 해결 능력을 보유하고 있는지 평가하는 데 중요한 역할을 할 것입니다.



### Deep Learning-Based Classification of Hyperkinetic Movement Disorders in Children (https://arxiv.org/abs/2411.15200)
Comments:
          59 pages, 20 figures

- **What's New**: 이번 연구에서는 소아의 과다운동장애(Hyperkinetic Movement Disorders, HMDs) 진단을 위해 비디오 녹화 데이터를 바탕으로 한 딥러닝 모델을 개발했습니다. 이 모델은 비디오에서 아동이 운동 과제를 수행하는 모습을 분석하여 지각적 이상 움직임을 감지하고, 특히 이상 운동의 유형인 디스토니아(dystonia)와 코레아(chorea)를 구별하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 모델은 그래프 합성곱 네트워크(Graph Convolutional Network, GCN)와 장단기 메모리(Long Short-Term Memory, LSTM) 네트워크를 결합하여 공간적 및 시간적 특징을 모두 활용합니다. GCN은 인체의 관절 연결을 그래프로 표현하여 이들의 공간적 관계를 학습하고, LSTM은 시간적 의존성을 모델링하여 움직임의 연속성을 처리합니다. 이러한 모델은 전문가의 공정한 해석을 도와주는 주의(attention) 메커니즘도 통합되었습니다.

- **Performance Highlights**: 모델은 50개의 비디오를 훈련 및 검증하여 15fps에서 85%의 정확도, 81%의 민감도 및 88%의 특이성을 달성했습니다. 주의 맵(attention map)을 통해 모델이 비자발적인 운동 패턴을 올바르게 식별하는 능력을 확인하였으며, 분류 오류는 주로 신체 부위가 가려지거나 미세한 움직임 변동 때문으로 나타났습니다. 이 연구는 HMD 진단의 정확성과 효율성을 향상시킬 수 있는 딥러닝의 가능성을 보여줍니다.



### Adaptively Controllable Diffusion Model for Efficient Conditional Image Generation (https://arxiv.org/abs/2411.15199)
- **What's New**: 본 논문에서는 새로운 적응형 제어 확산 모델인 Adaptively Controllable Diffusion (AC-Diff) 모델이 제안되었습니다. 이 모델은 생성 과정, 결과의 형태, 길이 및 매개변수 모두를 자동으로 제어할 수 있습니다. 기존의 확산 모델들이 가지고 있는 제어 문제를 해결하고자 하며, 고정된 매개변수 대신 동적 재조정을 통해 필요에 맞는 생성이 가능하도록 설계되었습니다.

- **Technical Details**: AC-Diff 모델은 Conditional Time-Step (CTS) 모듈을 통해 필요한 생성 단계 수를 결정하며, Adaptive Hybrid Noise Schedule (AHNS) 모듈을 사용하여 변동적인 확산 속도 매개변수를 생성합니다. 이 과정에서 입력 조건에 따라 모델이 스스로 조정할 수 있도록 적응형 샘플링 메커니즘이 통합되어 훈련됩니다. 이를 통해 더 향상된 성능을 목표로 합니다.

- **Performance Highlights**: AC-Diff 모델은 전체 평균 생성 단계 수와 실행 시간을 크게 줄이는 동시에 기존 문헌의 확산 모델과 유사한 성능을 유지할 것으로 기대됩니다. 이 모델은 효율성이 강조된 생성 모델로, 간단한 내용부터 더 복잡한 내용까지의 양질의 생성이 가능하도록 도와줍니다.



### K-means Derived Unsupervised Feature Selection using Improved ADMM (https://arxiv.org/abs/2411.15197)
- **What's New**: 이번 논문에서는 K-means 유도 비지도 특징 선택(K-means UFS)이라는 새로운 방법을 제안합니다. 기존의spectral analysis에 기반한 방법들과 달리, K-means의 목적을 사용하여 특징을 선택합니다. 목표는 서로 다른 클러스터의 데이터 포인트를 잘 분리할 수 있는 특징을 선택하는 것입니다.

- **Technical Details**: K-means UFS에서는 데이터 포인트의 군집 내 차이를 최소화하고 군집 간 차이를 최대화하는 특징을 선택합니다. 이를 위해 Alternating Direction Method of Multipliers (ADMM) 알고리즘을 개발하여 NP-hard 최적화 문제를 해결합니다. 제안된 모델은 p×n 형태의 데이터 행렬을 표준화하여 처리합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험 결과, K-means UFS 방법이 기존 방법들보다 더 효과적임을 입증하였습니다. 본 연구에서 제안한 기법은 클러스터링을 위한 특징 선택에 있어 우수한 성능을 보여주며, 새로운 기준으로써 비지도 특징 선택의 가능성을 제시합니다.



### Graph Neural Network-Based Entity Extraction and Relationship Reasoning in Complex Knowledge Graphs (https://arxiv.org/abs/2411.15195)
- **What's New**: 이번 연구에서는 그래프 신경망(Graph Neural Network)을 기반으로 하는 지식 그래프(Knowledge Graph) 엔티티 추출과 관계 추론 알고리즘을 제안하였습니다. 그래프 합성곱 네트워크(Graph Convolutional Network)와 그래프 주의 네트워크(Graph Attention Network)를 사용하여 지식 그래프의 복잡한 구조를 모델링하였습니다. 이 논문은 엔드 투 엔드(End-to-End) 공동 모델을 구축하여 엔티티와 관계의 효율적인 인식 및 추론을 달성했습니다.

- **Technical Details**: 제안된 모델은 다양한 딥러닝 알고리즘과 비교되어 AUC(Area Under Curve), Recall Rate, Precision Rate, F1 Value 등의 지표를 통해 우수성을 검증받았습니다. 실험 결과는 특히 복잡한 지식 그래프에 대한 강한 일반화 능력과 안정성을 보여주었습니다. 이는 지식 그래프에 대한 추가 연구에 강력한 지원을 제공하며, 그래프 신경망의 엔티티 추출 및 관계 추론에서의 응용 가능성을 입증합니다.

- **Performance Highlights**: 제안된 모델은 모든 지표에서 뛰어난 성능을 보였으며, 특히 복잡한 지식 그래프에서 더 강력한 일반화와 안정성을 나타냈습니다. 이러한 성능은 실제 상황에서의 응용 가능성을 더욱 높여 줍니다. 따라서, 그래프 신경망의 활용 범위를 넓히는 데 이 연구가 기여할 것으로 기대됩니다.



### Guiding Word Equation Solving using Graph Neural Networks (Extended Technical Report) (https://arxiv.org/abs/2411.15194)
- **What's New**: 본 논문은 단어 방정식을 해결하기 위한 그래프 신경망(Graphic Neural Network) 기반의 알고리즘을 제안합니다. 이 알고리즘은 방정식을 분할하는 잘 알려진 Nielsen 변환을 토대로 발전되었습니다. 각 방정식 변형의 첫 번째 항을 반복적으로 재작성하여 트리와 같은 검색 공간을 생성하며, 분할 지점에서의 경로 선택이 문제 해결 시간에 미치는 영향을 다룹니다.

- **Technical Details**: 제안된 알고리즘은 DragonLi라는 이름의 솔버로 구현되었으며, 멀티 분류 작업으로서 분할 결정을 인코딩합니다. 또한 다섯 가지의 단어 방정식 그래프 표현을 도입하여 GNN의 구조적 정보를 인코딩합니다. GNN은 일반적으로 비유클리드(Non-Euclidean) 데이터 구조에 적합한 딥 러닝 알고리즘으로, 이 논문에서 처음으로 단어 방정식 문제에 적용되었습니다.

- **Performance Highlights**: 실험 결과, DragonLi는 특히 만족 가능한 문제에서 높은 성능을 발휘하며, 단일 단어 방정식의 경우 기존의 문자열 솔버보다 현저히 많은 문제를 해결할 수 있습니다. 여러 단어 방정식의 결합에 대한 경우에도 DragonLi는 최신 문자열 솔버와 경쟁력을 유지합니다. GNN 기반의 지침이 DragonLi의 성능을 균일하게 향상시켜 주며, 특정 벤치마크에서는 모든 다른 솔버를 초월하는 결과를 보였습니다.



### Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting (https://arxiv.org/abs/2411.15193)
- **What's New**: 본 논문은 Gaussian Splatting에서 feature field rendering을 위한 학습 없는(training-free) 방법을 소개합니다. 제안된 방법은 2D features를 사전 훈련된 3D Gaussians로 역투영하여, 각각의 Gaussian이 최종 렌더링에서 미치는 영향에 따라 가중합을 사용하는 구성을 가지고 있습니다. 기존의 학습 기반 접근 방법들이 2D 분할(segmentation)에서는 뛰어난 성능을 발휘하지만, 3D 분할에서는 후처리가 필요하도록 저조한 결과를 내는 것과 달리, 제안된 방법은 2D와 3D 분할 모두에서 높은 품질의 결과를 달성합니다.

- **Technical Details**: 3D Gaussian Splatting (3DGS)은 3D 장면을 렌더링하기 위해 3D Gaussians를 사용하는 기법으로, 각 Gaussian은 평균 위치(mean position)와 분산(variance)으로 정의됩니다. 우리 연구에서는 3D 공간에서 직접 2D features를 3D Gaussians에 투사하는 방식을 이용하여, 전통적인 feature 필드 증류(feature field distillation)와 비교할 때 빠르고 확장 가능한 대안을 제공합니다. 또한, 본 논문에서는 feature 백프로젝션(back-projection) 방정식을 제시하고, 이를 기반으로 3D 분할, 합에서는전이(affordance transfer), 그리고 마지막으로 정체성 인코딩(identity encoding)과 같은 네 가지 직접 사용 사례를 설명합니다.

- **Performance Highlights**: 실험 결과에 따르면, 본 연구의 접근법은 빠르며 확장 가능하고, 대규모 학습 기반 방법과 비교할 수 있는 성능을 제공합니다. 특히, 3D 개체 조작 및 실시간 장면 이해와 같은 다운스트림 애플리케이션을 위한 원활한 쿼리 지원이 가능하다는 점에서 강력한 장점을 보입니다. 또한, 기존의 다른 접근법이 3D 분할에서 어려움을 겪는 것에 반해, 본 방법은 기울기 정보(gradient information)를 직접 활용하여 보다 효과적이고 정확한 3D 분할을 가능하게 합니다.



### Tailoring the Hyperparameters of a Wide-Kernel Convolutional Neural Network to Fit Different Bearing Fault Vibration Datasets (https://arxiv.org/abs/2411.15191)
Comments:
          71 pages, 14 figures, 7 tables

- **What's New**: 이 논문에서는 기계 베어링의 결함 감지를 위한 신경망의 하이퍼파라미터 설정을 어떻게 최적화할 수 있는지에 대한 새로운 통찰을 제공합니다. 특히, 새로운 데이터로 전환할 때 적절한 하이퍼파라미터 설정이 변할 수 있다는 점을 강조합니다. 이를 통해 하이퍼파라미터 설정이 전반적인 네트워크 성능에 미치는 영향을 직시합니다. 또한, 기존의 데이터 세트는 여러 벤치마크 데이터 세트에서 수집된 정보를 융합하여 다루어졌습니다.

- **Technical Details**: 소위 wide-kernel convolutional neural network에 대한 다양한 하이퍼파라미터 설정을 연구하며, 네트워크 아키텍처에 특정한 하이퍼파라미터 설정 방법이 필요함을 보여줍니다. 하이퍼파라미터는 신경망의 학습 과정에서 중요한 역할을 하며, 매우 잘못된 설정은 학습 실패로 이어질 수 있습니다. 특히 convolutional kernels의 폭과 같은 아키텍처 관련 하이퍼파라미터가 데이터 속성에 따라 어떻게 다르게 적용되는지에 대해 탐구합니다.

- **Performance Highlights**: 논문에서 제시된 실험 결과에 따르면, 신경망의 첫 번째 레이어에서의 커널 크기가 데이터 변화에 매우 민감하다는 것을 보여줍니다. 서로 다른 데이터 세트를 분석하면서, 하이퍼파라미터 설정이 정확도에 미치는 중대한 영향을 확인했습니다. 이를 기반으로 하이퍼파라미터 설정에 대한 명확한 지침을 제공하여, 실제 사용 시 성능 저하를 피할 수 있도록 돕습니다.



### Order Is All You Need for Categorical Data Clustering (https://arxiv.org/abs/2411.15189)
- **What's New**: 이 논문은 속성 값 간의 순서 관계가 범주형 데이터 클러스터링의 정확도에 결정적인 요소라는 새로운 발견을 소개합니다. 또한, 클러스터와 순서를 공동으로 학습할 수 있는 새로운 학습 패러다임을 제안합니다. 이 방법을 통해 클러스터링 정확도가 향상되었으며, 배운 순서는 범주형 데이터의 클러스터 분포를 이해하는 데 직관적인 도움을 줍니다.

- **Technical Details**: 제안된 방법은 클러스터링 과정 중 현재의 데이터 분할에 따라 동적으로 순서를 학습합니다. 이 과정은 클러스터링 작업에 연결되어 있어 클러스터링에 최적의 순서를 자동으로 얻을 수 있습니다. 또한, 클러스터-샘플 소속도를 동적으로 측정하기 위해 거리 학습 메커니즘이 도입되어 있습니다.

- **Performance Highlights**: 광범위한 실험 평가를 통해 제안된 방법이 기존 접근 방식보다 우수하다는 것이 입증되었습니다. 특히, 적절한 순서를 얻음으로써 클러스터링 정확도를 크게 향상시킬 수 있었으며, 해당 알고리즘은 혼합 데이터에도 쉽게 확장할 수 있습니다. 이 방법은 학습 과정과 배운 순서 모두 높은 해석력을 가집니다.



### Hybrid Gaussian Process Regression with Temporal Feature Extraction for Partially Interpretable Remaining Useful Life Interval Prediction in Aeroengine Prognostics (https://arxiv.org/abs/2411.15185)
- **What's New**: 이 논문은 Remaining Useful Life(RUL) 추정에 대한 새로운 접근법으로 수정된 Gaussian Process Regression(GPR) 모델을 소개하고 있습니다. 이 방식은 시계열 데이터를 학습하여 신뢰 구간(confidence intervals)을 예측하고, 제조 과정의 불확실성을 구조적인 방식으로 모델링하는 데 중점을 두고 있습니다. 또한, 고급 AI 프로세스 모델과의 결합을 통해 현대 제조 시스템의 시간 시계열 패턴과 동적 동작을 포착합니다.

- **Technical Details**: 제안하는 방법은 HRP(하이브리드 방정식 회귀)라는 약어로 명명된 모델을 포함하며, 이는 시간 종속성을 고려한 예측을 통해 제조 환경의 시스템 건강 상태를 반영합니다. 이 모델은 실시간 데이터를 사용하여 유지보수 기회를 계획하고, 시스템 동작의 주요 변수와 불확실성을 함께 모델링하여 의사결정의 투명성을 높입니다. 수정된 GPR 모델은 신뢰 구간을 생성하여 예측된 RUL의 불확실성을 정량화하도록 설계되었습니다.

- **Performance Highlights**: 이 연구를 통해 제안된 접근법은 RUL 예측의 정확도를 크게 향상시키며, 주요 예측 기능을 식별하고 강조하는 특징 중요도 분석을 포함하여 작업 효율성을 높입니다. 결과적으로, 고급 학습 기술과 불확실성 모델링을 통합한 이 모델은 데이터 기반의 실시간 유지보수 계획을 가능하게 해 주며, 스마트 제조 환경에서의 RUL 추정에 대한 새로운 방향을 제공합니다.



### Balancing property optimization and constraint satisfaction for constrained multi-property molecular optimization (https://arxiv.org/abs/2411.15183)
- **What's New**: 이번 연구는 여러 물질적 성질을 최적화하면서도 드로그^(*drug-like*) 기준을 준수할 수 있도록 하는 제약 조건 다중 속성 분자 최적화 프레임워크(CMOMO)를 제안합니다. CMOMO는 다이나믹 협력 최적화(Dynamic Cooperative Optimization)를 사용하여 다양한 시나리오에서 제약 조건을 동적으로 처리합니다. 이를 통해 고품질의 분자를 평가하고 발굴할 수 있는 새로운 방안을 제시하였습니다.

- **Technical Details**: CMOMO는 분자 특성에 대한 동적 협력 최적화 framework를 갖추고 있으며, 최적화된 분자 속성에 따라 진화 과정에서 고품질 분자를 선택합니다. 이 방법은 두 가지 시나리오(제약 없는 시나리오와 제약 있는 시나리오)로 나뉘어 각각의 요구를 충족시키기 위해 최적화를 진행합니다. CMOMO의 특징은 벡터 분리 기반 진화 복제 전략(VFER)을 통해 지속적인 암시적 공간에서의 진화 효율성을 높이는 데 있습니다.

- **Performance Highlights**: CMOMO는 두 개의 벤치마크 작업에서 다수의 최신 분자 최적화 방법에 비해 우수한 성능을 보였습니다. 또한, β2-adrenoceptor GPCR에 대한 후보 리간드와 glycogen synthase kinase-3 후보 억제제를 발견하는 실용적인 작업에서도 높은 성능을 입증했습니다. 실험 결과, CMOMO는 드로그 기준을 준수하면서 다양한 고속성을 가진 분자를 발굴할 수 있는 가능성을 보여주었습니다.



### Forecasting Application Counts in Talent Acquisition Platforms: Harnessing Multimodal Signals using LMs (https://arxiv.org/abs/2411.15182)
- **What's New**: 본 연구는 채용 도메인에서 지원서 수 예측(application count forecasting)이라는 새로운 작업을 소개합니다. 이 작업은 우수한 지원자를 유치하기 위한 효과적인 홍보 활동을 설계하는 데 필요한 인사이트를 제공합니다. 기존의 자율 회귀(auto-regressive) 기반 타임 시리즈 예측 방법이 이 작업에 부적합하다는 점을 강조하며, 다중 모달 언어 모델(multimodal LM) 기반 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 다양한 형태의 직업 공고 메타데이터를 통합하기 위해 간단한 인코더를 사용합니다. 모든 데이터 필드(텍스트, 범주형, 수치형 등)를 하나의 문장으로 취급하여 텍스트 구조를 처리할 수 있는 사전 학습된 BERT 모델에 입력합니다. 이를 통해 기존 접근 방식들보다 훨씬 효율적으로 다중 모달 특성을 융합(fuse)하여 JACs 예측을 수행합니다.

- **Performance Highlights**: 대규모 실제 데이터셋인 CareerBuilder LLC의 실험 결과, 제안한 방법이 기존의 최첨단 방법들보다 유의미한 성과를 보였습니다. 본 연구의 중요한 기여는 지원자 수(JACs) 예측이라는 새로운 작업을 도입한 것과 함께, BERT와 같은 언어 모델이 다양한 데이터 유형의 특성을 시너지 효과로 통합할 수 있다는 점을 보여주는 것입니다.



### Multi-layer matrix factorization for cancer subtyping using full and partial multi-omics datas (https://arxiv.org/abs/2411.15180)
- **What's New**: 이 연구는 Multi-Layer Matrix Factorization (MLMF)라는 새로운 접근 방식을 통해 암 유형화(cancer subtyping)를 위한 다중 오믹스(multi-omics) 데이터 클러스터링을 제안합니다. MLMF는 다층(linear 또는 nonlinear) 행렬 분해를 통해 오믹스 특성 행렬을 처리하여, 다양한 오믹스 유형에 고유한 잠재 특성 표현을 생성합니다. 이 잠재 표현들은 합의 형태로 융합되어 스펙트럼 클러스터링(spectral clustering)에 사용됩니다.

- **Technical Details**: MLMF는 암 유형화를 위한 두 가지 모듈로 구성됩니다: 행렬 분해(matrix factorization)과 최적화된 합의(representation optimization). 다중 오믹스 특성 행렬을 입력받아 다양한 오믹스의 잠재적 특성을 분해하고 이를 합쳐서 최종적으로 스펙트럼 클러스터링으로 타입을 결정합니다. 또한 이 과정에서 일부 샘플의 오믹스 데이터에 대한 결측 상태를 나타내는 클래스 인디케이터 행렬을 통합하여 불완전한 다중 오믹스 데이터와 완전한 데이터를 모두 관리할 수 있는 통합된 프레임워크를 제공합니다.

- **Performance Highlights**: 10개의 다중 오믹스 암 데이터세트를 대상으로 한 광범위한 실험 결과 MLMF는 여러 최신 접근 방법들과 비교하여 동등하거나 더 나은 결과를 보여줍니다. 특히, 결측값이 있는 데이터세트에서도 MLMF는 효과적인 성능을 발휘하여 기존의 데이터 삭제 및 KNN 보간 방법보다 우수한 결과를 도출합니다. 이러한 결과는 MLMF의 접근 방식이 다중 오믹스 데이터의 통합과 결측 데이터 처리에 있어 뛰어난 성능을 발휘함을 나타냅니다.



### Harnessing Scale and Physics: A Multi-Graph Neural Operator Framework for PDEs on Arbitrary Geometries (https://arxiv.org/abs/2411.15178)
- **What's New**: 이번 논문은 부분 미분 방정식(Partial Differential Equations, PDEs)을 효율적으로 해결하기 위해 설계된 AMG 방법을 소개하고 있습니다. 이 방법은 다양한 공간 영역과 복잡한 데이터 상호 의존성을 정확히 관리할 수 있도록 하는 새로운 GraphFormer 아키텍처를 기반으로 합니다. AMG는 기존의 기술이 균일한 격자에 제한되는 것과는 달리, 임의의 기하학에서 작동할 수 있으며, 다중 스케일 그래프를 통해 다양한 특성 주파수를 관리합니다.

- **Technical Details**: AMG는 다중 스케일 그래프와 물리적 성질을 캡슐화하는 그래프를 활용하여, 주어진 좌표로부터 세 가지 유형의 그래프를 구성합니다. 이 모델은 이웃에 대한 동적 주의 메커니즘을 통해 각 노드의 숨겨진 표현을 계산하며, 복잡한 데이터 간 상호 작용을 효과적으로 처리할 수 있습니다. 또한, 그래프 주의는 학습 가능한 적분으로 볼 수 있어 연속 공간 모델링 능력을 강화하는 이론적 기반을 제공합니다.

- **Performance Highlights**: 실험 결과, AMG는 여섯 개의 벤치마크에서 기존의 최첨단 모델을 지속적으로 초월하며 탁월한 상대적 성과를 달성했습니다. 이 연구는 전통적인 PDE 해결 방식이 직면했던 문제를 극복할 수 있는 맞춤형 그래프 신경 연산자의 혁신적인 가능성을 보여줍니다. 또한, 이 방법에 대한 코드와 데이터셋은 공개되어 있으며, 연구자들이 보다 쉽게 접근할 수 있도록 지원하고 있습니다.



### Can Open-source LLMs Enhance Data Augmentation for Toxic Detection?: An Experimental Study (https://arxiv.org/abs/2411.15175)
- **What's New**: 이번 연구에서는 오픈 소스 LLMs(대규모 언어 모델)에 대한 프롬프트 엔지니어링과 파인튜닝 기술을 통해 유해 데이터 증강의 효과를 높이는 방법을 탐구하였습니다. 특히, 모델들이 생성하는 유해 데이터의 품질과 다양성을 현저히 개선하는 데 성공했습니다. Mistral 모델이 최소한의 환각(hallucination)으로 유해 데이터를 생성할 수 있음을 발견했습니다. 또한, 파인튜닝이 데이터 품질 향상에 효과적이지만, 데이터 중복 및 과적합(overfitting) 문제는 여전히 존재합니다.

- **Technical Details**: 방법론은 두 단계로 구성되며, 첫 번째 단계는 프롬프트 엔지니어링을 통해 유해 데이터를 생성하기 위한 프롬프트를 설계하는 것입니다. 요구 사항을 명확히 하고 다양한 샘플 예제를 포함하는 프롬프트 템플릿을 개발했습니다. 그러나 모델의 내부 안전 정렬로 인해 유해 데이터 품질과 다양성이 저하되는 문제에 직면하게 되었습니다. 그래서 파인튜닝을 통해 모델의 가중치를 업데이트하여 유해 데이터 생성 능력을 향상시키기로 결정했습니다.

- **Performance Highlights**: 우리는 다양한 오픈 소스 LLMs를 평가하여 Mistral 모델이 기존 기준선 모델들과 비교하여 최상의 성능을 보여준다는 것을 확인했습니다. 실험 결과, 파인튜닝된 모델은 유해 데이터 생성의 품질과 다양성을 크게 개선했습니다. 이 연구는 향후 유해 물질 탐지 시스템의 발전 방향에 대한 통찰력을 제공하며, 자동화된 콘텐츠 모더레이션 도구의 효과성을 실질적으로 향상시키는 데 기여할 것으로 기대됩니다.



### Decentralizing Test-time Adaptation under Heterogeneous Data Streams (https://arxiv.org/abs/2411.15173)
- **What's New**: 이번 연구는 Test-Time Adaptation (TTA)을 이질적인 데이터 스트림에 적용할 때의 한계를 극복하는 새로운 방법론을 제시합니다. Fourier 공간(Fourier space)에서 데이터를 분해하여 다양한 주파수 수준에서 데이터의 정확한 분리를 가능하게 하였으며, 이로 인해 기존의 모델 중심 접근법에서 벗어난 데이터 중심 접근법을 제안합니다. 특히, 우리의 접근법은 지역적으로 동질적인 데이터 환경으로의 전환을 촉진하는 Frequency-based Decentralized Adaptation (FreDA) 프레임워크를 통해 이루어집니다.

- **Technical Details**: FreDA 프레임워크는 데이터를 Fourier 도메인에서 동적으로 분할하는 과정으로 시작합니다. 초기 세분화는 데이터가 분포가 이질적인 글로벌 상태에서 국소적으로 동질적인 부분으로 전환될 수 있게 도와줍니다. 이 과정에서 우리는 로컬 모델들이 각기 다른 데이터 세그먼트에 독립적으로 적응할 수 있는 분산 학습 전략을 도입하였으며, 이는 다양한 분포 변화에 대한 적응 충돌을 줄이고 모델의 일반화 능력을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 다양한 환경(부패된 상황, 자연적 환경, 의료적 환경)에서의 광범위한 실험 결과를 바탕으로, FreDA 프레임워크는 최신 TTA 기법들에 비해 뛰어난 성능을 발휘했습니다. 우리의 접근법은 데이터의 질과 양을 모두 향상시켜, 동적 환경에서도 모델의 강건성(robustness)과 예측 능력을 크게 개선하는 성과를 보였습니다.



### Adaptive Sensor Placement Inspired by Bee Foraging: Towards Efficient Environment Monitoring (https://arxiv.org/abs/2411.15159)
- **What's New**: 이번 논문은 지속 가능한 로봇 공학의 미래를 위한 혁신적인 알고리즘을 제안합니다. 인공벌레 군집(ABC) 알고리즘과 레비 비행(Levy flight) 방식의 하이브리드 알고리즘을 결합하여 적응형 센서 배치 최적화를 달성하는 데 중점을 두고 있습니다. 이 접근 방식은 중요한 핫스팟(hotspot)을 효과적으로 식별하도록 개선된 탐색 및 활용 기능을 제공합니다.

- **Technical Details**: 하이브리드 Adaptive Bee Colony-Levy 알고리즘은 ABC의 탐색 동역학과 레비 비행 방식을 조합하여 센서 배치를 최적화합니다. 이 알고리즘은 데이터 수집의 중복을 피하고, 도메인 전문가에 의해 대략적으로 결정된 핫스팟 주변에서 데이터를 수집하는 방향으로 설계되었습니다. 또한, 이 연구는 로봇 유닛의 효율적인 데이터 수집을 위한 최적 접근 방식을 규명하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 알고리즘은 자율 로봇의 군집을 활용하여 100헥타르의 열대우림을 조사하는 현실적인 문제를 다룹니다. 이 하이브리드 접근 방식은 데이터 수집의 효율성을 높이며, 환경 모니터링과 자원 관리 분야에 중요한 기여를 할 것으로 기대됩니다. 로봇 시스템은 복잡한 환경에서 고품질 데이터를 수집할 수 있는 잠재력을 보여줍니다.



### Memory-Driven Metaheuristics: Improving Optimization Performanc (https://arxiv.org/abs/2411.15151)
Comments:
          25 pages, 5 figures, book chapter, Springer

- **What's New**: 메타 휴리스틱스(Metaheuristics)는 복잡한 문제에 대한 최적 솔루션을 찾기 위해 자연과정을 모방한 확률적 최적화 알고리즘입니다. 본 논문에서는 메모리(memory) 메커니즘이 메타 휴리스틱 알고리즘의 성능을 개선하는 데 중요한 역할을 한다고 강조하고, 메모리가 어떻게 통합되는지를 살펴봅니다. 메모리의 크기, 저장된 정보, 정보의 감쇠율 등 메모리 메커니즘의 효과성을 좌우하는 주요 요인들도 논의됩니다.

- **Technical Details**: 본 장에서는 메타 휴리스틱스에서의 메모리 개념과 역할을 소개하며, 인기 있는 알고리즘에 어떻게 메모리 메커니즘이 적용되는지에 대한 포괄적인 분석이 제공됩니다. 메모리 메커니즘의 효과성을 높이기 위해서는 메모리의 크기와 정보 저장 방식, 정보의 소멸 속도와 같은 여러 요소를 고려해야 합니다. 이러한 분석을 통해 메모리 메커니즘이 메타 휴리스틱스의 성능을 어떻게 향상시킬 수 있는지에 대해 설명합니다.

- **Performance Highlights**: 메모리 메커니즘은 메타 휴리스틱스가 탐색 공간(search space)을 효과적이고 효율적으로 탐색하고 활용할 수 있도록 하여 성능을 크게 향상시킬 수 있습니다. 따라서 각 문제 도메인과 탐색 공간의 특성에 맞추어 적절한 메모리 메커니즘을 선택하는 것이 중요합니다. 논문은 메모리가 메타 휴리스틱 성능에서 갖는 중요성을 강조하고 향후 메모리 메커니즘 개선을 위한 연구 방향도 제시합니다.



### The Fundamental Rights Impact Assessment (FRIA) in the AI Act: Roots, legal obligations and key elements for a model templa (https://arxiv.org/abs/2411.15149)
- **What's New**: 이 논문은 AI 법안(AI Act)에서 기본적 권리 영향 평가(Fundamental Rights Impact Assessment, FRIA)를 수행해야 하는 의무의 출현 배경을 분석합니다. 특히, 기본적 권리에 미치는 영향을 어떻게 EU 입법자가 설정했는지를 다루며, FRIA 개발을 위한 방법론적 기준에 대해 설명합니다.

- **Technical Details**: 논문은 AI 법안의 관련 조항에 대한 법적 분석과 AI가 기본적 권리에 미치는 영향 평가의 다양한 모델을 논의함으로써 기존의 이론적 및 방법론적 공백을 메우고자 합니다. FRIA의 주요 구성 요소를 포함하는 템플릿 모델을 제안하여, EU 및 국가 기관들이 인본적(human-centric)이고 신뢰할 수 있는 AI를 실현하는 데 필수적인 도구로 활용할 수 있도록 합니다.

- **Performance Highlights**: 제안된 FRIA 모델은 AI 법안의 합리성과 범위에 부합하며, 제27조에 명시된 경우 외에도 적용이 가능합니다. 이는 다른 국가 및 국제 규제 이니셔티브가 AI가 인권에 완전히 부합하도록 하기 위해 유용한 청사진이 될 수 있습니다.



### Delegating Responsibilities to Intelligent Autonomous Systems: Challenges and Benefits (https://arxiv.org/abs/2411.15147)
- **What's New**: 이 논문은 AI 시스템이 자율성과 적응성을 가지게 됨에 따라, 기술-사회 시스템에서 도덕적 책임의 전통적인 경계가 도전 받고 있다는 점을 다룹니다. 특히 지능형 자율 에이전트에게 책임을 위임하는 과정과 그로 인한 윤리적 함의(ethical implications)에 대한 새로운 담론을 탐구합니다.

- **Technical Details**: 논문은 '분산 책임(distributed responsibility)'과 '디자인에 의한 윤리적 AI(ethical AI by design)' 같은 최근 AI 윤리 발전을 종합하여 기능주의적 관점(functionalist perspective)을 제안합니다. 변화하는 AI 윤리적 환경 안에서 도덕적 책임을 개인의 특성이 아닌 사회 기술 시스템 내의 역할로 보며, 인간과 인공지능 에이전트 간에 분산되어 있다고 주장합니다.

- **Performance Highlights**: Basti와 Vitiello의 구현 예를 통해 '디자인에 의한 AI 윤리(AI ethical by design)'를 소개합니다. AI가 윤리적 가이드라인을 학습하고, Deontic Higher-Order Logic을 사용하여 결정을 윤리적으로 평가할 수 있는 인공지능 도덕 에이전트로서 작동할 수 있음을 보여줍니다.



### dafny-annotator: AI-Assisted Verification of Dafny Programs (https://arxiv.org/abs/2411.15143)
- **What's New**: 이번 연구에서는 dafny-annotator라는 새로운 도구를 소개하였습니다. 이 도구는 Large Language Models (LLMs)와 검색 기법을 이용하여 Dafny 프로그램에 자동으로 논리적 주석을 추가하는 기능을 갖추고 있습니다. 기존에 수작업으로 작성하던 논리적 주석을 자동화하여 이는 개발자들이 형식적 검증을 보다 쉽게 수행할 수 있도록 돕습니다. 특히, 처리된 결과는 대규모 데이터 학습이 어려운 상황에서도 향후 AI 도우미 개발에 대한 방향성을 제시합니다.

- **Technical Details**: dafny-annotator는 세 가지 주요 구성 요소로 나뉘어 있습니다: LLM-guided search, 기존의 주석 프로그램을 이용한 fine-tuning 및 무한 프로그램 생성을 위한 합성 파이프라인입니다. 이 방법은 임의의 Dafny 메서드에 대한 주석 제안을 생성하고, 이러한 주석을 프로그램 내의 유효한 위치에 삽입하는 방식으로 동작합니다. 이 과정에서 LLM이 제안하는 주석을 기반으로 greedy search 전략을 통해 최적의 주석을 선택하고, 필요한 경우 반복적으로 이 과정을 진행합니다.

- **Performance Highlights**: 실험 결과, dafny-annotator는 DafnyBench 데이터셋을 사용하여 시험한 결과, 기본 LLaMa 8B 모델로는 15.7%의 성공률을 기록하였으나, 추가적인 데이터와 훈련을 통해 50.6%로 향상되었습니다. 이는 기존 데이터셋에 대한 fine-tuning뿐만 아니라 새로운 합성 데이터인 DafnySynth를 통한 학습의 효과를 보여줍니다. 이러한 결과들은 형식적 검증 분야에서 AI 도우미의 능력을 증대시키고, 특히 인간 생성 데이터가 부족한 언어의 경우 사용자가 쉽게 접근할 수 있도록 도와줍니다.



### KBAlign: Efficient Self Adaptation on Specific Knowledge Bases (https://arxiv.org/abs/2411.14790)
- **What's New**: 이번 논문에서는 KBAlign이라는 새로운 접근法을 제안하여, 대규모 언어 모델(LLM)을 효율적으로 지식 베이스(KB)에 적응시키는 방법을 소개합니다. 이 방법은 자가 주석(self-annotation) 데이터를 활용하여 모델이 지식 내용을 신속하게 이해하고, 실제 작업에서 성능을 향상시키는 데 중점을 둡니다. 실험 결과, KBAlign을 사용함으로써 GPT-4-turbo 주석을 이용했을 때와 유사한 성능 향상을 달성할 수 있음을 보여줍니다.

- **Technical Details**: KBAlign은 Q&A 쌍 및 수정 제안을 포함한 자기 주석 데이터를 반복적으로 훈련하는 방식을 사용합니다. 이 과정을 통해 모델은 자신의 응답을 점검하고, 학습이 진행될수록 기존의 오류를 수정하게 됩니다. 또한, 쿼리 확장(query expansion) 및 확신 검증(confidence verification)과 같은 전략을 사용하여 응답을 개선하는 과정이 포함되어 있습니다.

- **Performance Highlights**: KBAlign은 사실 Q&A, 긴 형식 QA, 전문 분야 테스트 등 다양한 데이터 세트에서 실험을 수행하여 그 효율성을 입증하였습니다. 이를 통해 일반 지식 내용을 효과적으로 파악하고, 특정 지식이 필요한 하위 작업에서 성능을 크게 향상시켰습니다. 부가적인 실험 결과로 가장 효율적인 자가 주석 데이터 양과 최적의 훈련 볼륨을 찾아내어 실용적인 적용에 대한 귀중한 지침을 제공합니다.



### Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction (https://arxiv.org/abs/2411.14762)
Comments:
          Code is available on the project webpage: this https URL

- **What's New**: 최근에 발표된 CoordTok은 긴 동영상을 효율적으로 인코딩할 수 있는 새로운 비디오 토크나이저입니다. 이 모델은 좌표 기반 표현을 사용하여 입력 비디오의 해당 패치를 매핑 학습하는 방법을 제안합니다. 최신 3D 생성 모델에서 영감을 받아, CoordTok은 트리플레인 표현으로 비디오를 인코딩하고 무작위로 샘플링된 (x,y,t) 좌표에 해당하는 패치를 재구성합니다.

- **Technical Details**: CoordTok의 기술적 요소는 비디오를 비겹치는 시공간 패치로 나누고, 학습 가능한 위치 임베딩을 추가하여 여러 Transformer 층을 거쳐 비디오 특징을 추출하는 것입니다. 이러한 특징을 기반으로 트리플레인 표현을 인코딩하며, 이를 통해 원본 비디오를 정보 손실 없이 재구성할 수 있습니다. 이 방식은 기존의 3D 배치 접근 방식과 달리 메모리와 계산 자원 소모를 최소화합니다.

- **Performance Highlights**: 실험 결과, CoordTok은 긴 비디오 인코딩을 위해 필요한 토큰 수를 대폭 줄이는 것을 보여줍니다. 예를 들어, 128프레임의 128×128 해상도 비디오를 겨우 1280개의 토큰으로 인코딩하는 동안 기존 방식은 6144 또는 8192개의 토큰을 필요로 합니다. 또한, CoordTok을 사용한 효율적인 비디오 토크나이제이션은 Diffusion Transformer의 메모리 효율적인 학습을 가능하게 하며, 이는 한 번에 128프레임의 비디오를 생성할 수 있게 해줍니다.



### Evaluating the Impact of Underwater Image Enhancement on Object Detection Performance: A Comprehensive Study (https://arxiv.org/abs/2411.14626)
- **What's New**: 본 연구는 수중 이미지 향상 모델의 최첨단 기법을 평가하고 수중 물체 탐지에 미치는 영향을 조사합니다. 이를 위해 두 개의 최신 데이터셋인 Real-World Underwater Object Detection Dataset (RUOD)와 Challenging Underwater Plant Detection Dataset (CUPDD)에 각각 모형을 적용했습니다. 연구의 결과, 이미지 향상이 탐지 성능에 미치는 복잡한 영향 및 감지 성과를 향상시킬 수 있는 잠재성도 탐구했습니다.

- **Technical Details**: 수중 이미지는 종종 조도, 명암 및 전반적인 품질 저하에 직면합니다. 따라서 이미지 품질을 비교하기 위해 Q-index라는 품질 지표가 제안되었습니다. 연구에서는 YOLO-NAS (You Only Look Once Neural Architecture Search) 탐지 모델을 활용하여 원본 및 향상된 이미지 세트를 기반으로 성능 비교를 수행했습니다.

- **Performance Highlights**: 연구 결과 향상이 일반적으로 탐지 성능을 저하시키지만, 특정 경우에는 탐지 성능과 인간 주석의 정확성을 증가시킬 수 있음이 입증되었습니다. 향상된 이미지에 대한 탐지 성능 저하의 근본 원인을 분석하고, 여러 향상 기법을 적용했을 때 물체 탐지 성능이 개선된 개별 사례를 제시했습니다.



### Unlearn to Relearn Backdoors: Deferred Backdoor Functionality Attacks on Deep Learning Models (https://arxiv.org/abs/2411.14449)
- **What's New**: 이번 연구에서는 Deep learning 모델의 backdoor 공격에 대한 새로운 패러다임인 Deferred Activated Backdoor Functionality (DABF)를 소개합니다. 기존의 backdoor 공격 방식은 트리거 입력이 있을 때 악의적인 행동을 촉발시키았으나, DABF는 처음에는 benign한 출력(무해한 결과)을 생성하여 탐지를 회피할 수 있는 점이 특징입니다.

- **Technical Details**: DABF 공격은 모델 업그레이드나 benign 데이터로의 재훈련을 통해서만 전략적으로 활성화됩니다. 이를 구현하기 위해 저희는 backdoor를 쉽게 취소하고 재활성화할 수 있도록 하는 두 단계 훈련 방식인 DeferBad를 제안합니다. 이러한 방법은 머신러닝 모델의 수명 주기에서 일반적으로 수행되는 모델 업데이트 및 미세 조정 후에도 backdoor가 숨겨질 수 있게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 다양한 미세 조정 시나리오, backdoor 공격 유형, 데이터셋 및 모델 아키텍처에서 DeferBad의 효과성과 은밀성을 입증하였습니다. DABF는 여러 탐지 및 방어 메커니즘을 우회할 수 있는 강력한 특성을 지니고 있어 이번 연구의 결과는 backdoor 공격의 이해와 대응에 중요한 기여를 할 것으로 예상됩니다.



### BugSpotter: Automated Generation of Code Debugging Exercises (https://arxiv.org/abs/2411.14303)
Comments:
          Preprint of the SIGCSE'25 paper

- **What's New**: BugSpotter는 프로그래밍 교육 분야에서 큰 변화를 몰고 올 혁신적인 도구입니다. 이 도구는 문제 설명을 기반으로 버그가 있는 코드를 자동으로 생성하고, 이를 테스트 스위트를 통해 검증합니다. 학생들은 실패하는 테스트 케이스를 설계함으로써 디버깅 기술을 향상시킬 수 있으며, 문제 명세서를 읽고 이해하는 능력도 함께 향상됩니다. 따라서 BugSpotter는 프로그래밍 학습의 일관성을 높이는 데 기여할 것으로 기대됩니다.

- **Technical Details**: BugSpotter는 LLM(대형 언어 모델)을 활용하여 문제 설명에서 버그가 있는 코드를 생성합니다. 이 도구는 자동으로 테스트 스위트를 생성하고 학생들에게 버그를 식별하고 수정하는 문제를 제시합니다. 학생들은 바이낸 오탈자 코드에 대해 실패하는 테스트 케이스를 설계함으로써 시스템적인 문제 해결 능력을 기를 수 있으며, 코드 이해도를 높일 수 있습니다. 이 과정은 메타인지(scaffolding) 이론 및 구조화된 학습 방식에 기반하고 있습니다.

- **Performance Highlights**: BugSpotter를 대규모 학급에 배포한 후, LLM이 생성한 디버깅 연습 문제와 강사가 수작업으로 만든 문제를 비교했습니다. 조사 결과, BugSpotter가 생성한 문제는 학생들의 성과와 강사가 만든 문제와 유사한 성과를 보였으며, 난이도 또한 문제 명세와 잘 맞아떨어졌습니다. 이로 인해 BugSpotter는 디버깅 학습에 효과적이고 효율적인 도구가 될 수 있다는 가능성이 확인되었습니다.



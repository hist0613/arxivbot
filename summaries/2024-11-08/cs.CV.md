New uploads on arXiv(cs.CL)

### Needle Threading: Can LLMs Follow Threads through Near-Million-Scale Haystacks? (https://arxiv.org/abs/2411.05000)
- **What's New**: 최근 대규모 언어 모델(LLMs) 및 멀티모달 LLMs의 길어진 컨텍스트가 다양한 애플리케이션과 다운스트림 기능을 가능하게 한다는 점이 밝혀졌습니다. 본 연구에서는 17개 주요 LLM의 정보 검색 능력을 평가하기 위해 실험을 수행했습니다.

- **Technical Details**: 17개의 LLM을 대상으로 한 실험에서는 정보의 흐름을 따라가는 능력에 중점을 두었습니다. 특히 LLM이 여러 스레드를 동시에 추적하는 'thread-safe' 기능을 갖추고 있음을 발견했습니다. 하지만 많은 모델이 지원되는 컨텍스트 길이보다 실제 유효 컨텍스트 길이가 짧아진다는 점도 확인했습니다.

- **Performance Highlights**: 대부분의 모델이 동시에 다양한 스레드를 추적할 수 있는 뛰어난 성능을 보였으나, 컨텍스트 길이가 증가함에 따라 정확도가 감소하는 경향을 보였습니다. 또한, 서로 다른 토크나이저의 토큰 수를 직접 비교하는 것은 금지적이라는 점을 강조했습니다.



### Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models (https://arxiv.org/abs/2411.04996)
- **What's New**: 이 논문에서는 다중 모달 시스템을 위한 새로운 아키텍처인 Mixture-of-Transformers (MoT)를 소개합니다. MoT는 다중 모달 데이터 처리를 위한 희소한 변환기 구조로, 사전 훈련(computation) 비용을 획기적으로 낮춥니다.

- **Technical Details**: MoT는 피드포워드 네트워크(feed-forward networks), 어텐션 매트릭스(attention matrices), 레이어 정규화(layer normalization) 등의 비임베딩(non-embedding) 파라미터를 모달리티별로 분리하여 처리합니다. 이를 통해 전체 입력 시퀀스에 대해 전역(self-attention) 자가 주의를 적용할 수 있습니다.

- **Performance Highlights**: Chameleon 7B 설정에서 MoT는 55.8%의 FLOPs만으로 밀집(dense) 기준선의 성능을 달성하며, 음성을 포함하는 경우에는 37.2%의 FLOPs로 비슷한 성능을 보입니다. Transfusion 설정에서는 7B MoT 모델이 밀집 기준선의 이미지 성능을 3분의 1의 FLOPs로 일치시키고, 760M MoT 모델은 1.4B 밀집 기준선을 여러 주요 이미지 생성 메트릭에서 초과 성능을 보였습니다.



### The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities (https://arxiv.org/abs/2411.04986)
- **What's New**: 이 연구는 현대 언어 모델들이 서로 다른 언어와 데이터 유형(모달리티) 간에 공통된 표현 공간을 학습함으로써 입력을 처리할 수 있는 능력을 획득했음을 제안합니다. 이 과정을 통해 의미적으로 유사한 입력이 서로 가까이 위치하게 되며, 이를 'semantic hub hypothesis'로 명명하였습니다.

- **Technical Details**: 저자들은 LMs(언어 모델)에서 서로 다른 언어의 의미적으로 동등한 입력의 표현이 중간 층에서 유사함을 보여줍니다. 이러한 공간은 모델의 주요 사전 훈련 언어를 통해 해석할 수 있으며, 이 경향은 산술 표현, 코드, 시각/청각 입력 등 다른 데이터 유형으로 확장됩니다. 또한, 한 데이터 유형에서의 공유 표현 공간에 대한 개입이 다른 데이터 유형의 모델 출력에 예측 가능한 영향을 미친다는 실험 결과를 제시합니다.

- **Performance Highlights**: 이 연구는 다양한 데이터 유형과 언어를 처리하는 LMs가 공유된 표현 공간을 통해 정보를 통합하여 모델의 출력을 결정하며, 이는 단순히 대규모 훈련의 부산물이 아님을 강조합니다. 저자들은 이러한 발견이 현재 모델의 메커니즘을 해석하는 데 도움이 될 것이며, 미래의 조절 모델 연구에 대한 동기를 제공할 것이라고 기대합니다.



### SuffixDecoding: A Model-Free Approach to Speeding Up Large Language Model Inferenc (https://arxiv.org/abs/2411.04975)
- **What's New**: SuffixDecoding은 대규모 언어 모델(LLM)의 추론을 가속화하기 위한 새로운 모델 프리 접근 방식으로, 기존의 초안 모델이나 특수 디코딩 헤드에 의존하지 않고, 이전에 생성된 출력에서 구축된 접미사 트리를 활용하여 후보 토큰 시퀀스를 효율적으로 예측합니다.

- **Technical Details**: SuffixDecoding은 각 이음 요청에 대해 별도의 접미사 트리를 구성하며, 이 구조를 사용하여 각 노드에서 토큰 시퀀스의 출현 빈도를 기록합니다. 이를 통해 SuffixDecoding은 기존의 매개변수화된 모델이나 추가의 GPU 메모리 사용 없이도 효율적인 패턴 일치를 수행합니다. 이 과정에서 CPU 메모리만을 사용하여 LLM의 성능을 최적화합니다.

- **Performance Highlights**: SuffixDecoding은 다양한 작업에서 경쟁력 있는 속도 향상을 달성합니다. 예를 들어, 오픈 도메인 채팅과 코드 생성 작업에서 SpecInfer보다 최대 1.4배 높은 출력 처리량과 1.1배 낮은 시토큰(latency)을 기록했습니다. 또한, 독점적인 multi-LLM text-to-SQL 애플리케이션에서는 SpecInfer보다 최대 2.9배 높은 출력 처리량과 3배 낮은 지연 시간을 나타냈습니다.



### BitNet a4.8: 4-bit Activations for 1-bit LLMs (https://arxiv.org/abs/2411.04965)
Comments:
          Work in progress

- **What's New**: BitNet a4.8가 도입되어 1-bit Large Language Model(LLM)의 효율성을 크게 향상시키며, 4-bit 활성화를 가능하게 합니다.

- **Technical Details**: BitNet a4.8는 하이브리드 양자화(hybrid quantization) 및 희소화(sparsification) 전략을 채택하여 outlier 채널에서 발생하는 양자화 오류를 줄입니다. 입력에 대해서는 4-bit (INT4/FP4) 활성화를 사용하고, 중간 상태에는 8-bit 양자화를 적용하여 계산 효율성을 극대화합니다.

- **Performance Highlights**: BitNet a4.8은 BitNet b1.58와 동등한 훈련 비용으로 유사한 성능을 달성하며, 55%의 파라미터만 활성화되고, 3-bit KV 캐시를 지원하여 대규모 LLM의 배치 및 추론 효율성을 더욱 향상시킵니다.



### Estimating the Influence of Sequentially Correlated Literary Properties in Textual Classification: A Data-Centric Hypothesis-Testing Approach (https://arxiv.org/abs/2411.04950)
- **What's New**: 본 논문은 문학 자질이 텍스트 분류에 미치는 영향을 평가하기 위한 가설 검증 접근법을 소개합니다. 특히, 인접한 텍스트 단위 간의 상관관계가 분류에 영향을 미치는 상황을 규명하고자 하며, 이를 통해 다양한 장르나 스타일에서 텍스트를 구분할 수 있는 방법론을 제안하고 있습니다.

- **Technical Details**: 저자들은 다변량 이항 분포(mvnariate binary distribution)를 사용하여 텍스트 단위 간의 상관관계를 확률적 과정(stochastic process)으로 모델링하였습니다. 이를 통해 텍스트 분류가 인접한 단위들 간의 순차적 상관관계(sequentially correlated properties)에 의해 지배되는지, 혹은 독립적인지를 평가할 수 있게 되었습니다. 또한, 전통적인 임베딩과 신경망 임베딩을 결합하여 지도 및 비지도 학습 프레임워크 내에서 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 본 접근법은 텍스트 분류가 순차적 상관관계에 주요하게 영향을 받지 않는 경우를 효과적으로 식별할 수 있음을 보여주었습니다. 특히, 저자 스타일이나 장르가 다른 텍스트의 경우, 이러한 상관관계의 영향이 적다는 점을 강조했습니다.



### GPTKB: Building Very Large Knowledge Bases from Language Models (https://arxiv.org/abs/2411.04920)
Comments:
          11 pages, 4 tables

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 사용하여 일반 도메인 지식 기반(KB)을 구축하는 새로운 방법을 제안합니다. 특히, GPT-4o-mini를 활용하여 2.9백만 개의 엔티티에 대해 1억 5천만 개의 트리플을 포함하는 GPTKB를 생성하였습니다.

- **Technical Details**: 이 연구에서는 반복적 그래프 확장을 통해 LLM의 지식 세트를 물질화하고, 엔티티 식별(named entity recognition, NER)과 관계 정규화(canonicalization) 및 분류 체계(taxonomy) 구축의 문제를 해결합니다. 반복적 프로그래밍을 통해 지식을 계속해서 확장하며, LLM 스스로를 활용하여 일관된 KB를 구축합니다.

- **Performance Highlights**: 이 연구는 NLP와 시맨틱 웹 커뮤니티 모두에게 중요한 기여를 합니다. LLM의 지식을 활용하여 과거의 카테고리 기반 KB 구축 방법을 혁신하며, GPTKB는 3.8GB의 데이터로 다운로드 및 사용이 가능합니다.



### GASE: Generatively Augmented Sentence Encoding (https://arxiv.org/abs/2411.04914)
Comments:
          12 pages, 3 figures

- **What's New**: 본 논문에서는 추론 시 데이터 증대를 위한 생성적 텍스트 모델을 적용해 문장 임베딩(sentence embeddings)을 향상시키는 방법을 제안합니다. 기존의 합성 데이터 증대 방식과 달리, 모델 파라미터나 고급 모델의 파인튜닝에 필요한 컴퓨터 자원에 접근할 필요가 없습니다.

- **Technical Details**: Generatively Augmented Sentence Encoding(GASE) 방법론을 통해 원본 텍스트에서 생성된 다양한 언어적 합성 변형을 사용하며, 이를 통해 문장 임베딩을 구성합니다. 세 가지 생성적 작업(paraphrasing, summarising, extracting keywords)을 통해 텍스트 변형을 생성하고, 원본 텍스트와 합성 텍스트의 임베딩을 풀링(pooling)하여 결합합니다.

- **Performance Highlights**: 실험 결과, GASE는 다양한 임베딩 모델에서 성능 향상을 보였습니다. BERT 임베딩 모델의 경우, 평균 +6.11의 성능 향상이 있었으며, GloVe 모델은 +2.96의 향상을 기록했습니다. 생성적 데이터 증대가 일반적으로 낮은 기본 성능을 가진 모델에서 더 큰 성능 개선을 이끌어낸다는 점이 주목할 만합니다.



### OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models (https://arxiv.org/abs/2411.04905)
- **What's New**: OpenCoder라는 새로운 오픈소스 코드 LLM이 출시되어, 코드 생성 및 AI 연구에 혁신을 가져온다. 이 모델은 기존 비공식 모델들과의 성능이 동등하며, 완전한 데이터 파이프라인과 훈련 프로토콜을 공개한다.

- **Technical Details**: OpenCoder는 (1) 데이터 정화와 중복 제거를 위한 최적화된 휴리스틱 규칙, (2) 코드와 관련된 텍스트 코퍼스의 회상, (3) 고품질 합성 데이터를 활용한 훈련 단계 등, 개방적이고 투명한 훈련 파이프라인을 통해 개발되었다. 이 모델은 9600억 개의 토큰으로 구성된 RefineCode 데이터셋을 기반으로 하며, 607개 프로그래밍 언어를 지원한다.

- **Performance Highlights**: OpenCoder는 다양한 코드 LLM 평가 벤치마크에서 최고의 성능을 기록하며, 연구자들에게 재현 가능한 학습 데이터와 완전한 데이터 처리 파이프라인을 제공하여 코드 AI 발전의 속도를 가속화할 예정이다.



### Sentiment Analysis of Spanish Political Party Tweets Using Pre-trained Language Models (https://arxiv.org/abs/2411.04862)
Comments:
          21 pages, 6 figures

- **What's New**: 이 연구는 스페인 정치 당의 트위터 통신에서 감정 패턴을 조사하며, 스페인어 텍스트에 최적화된 사전 학습(pre-trained) 언어 모델인 BETO와 RoBERTuito를 활용합니다.

- **Technical Details**: PSOE, PP, Vox, Podemos, Ciudadanos와 같은 주요 스페인 정치 당의 2019년부터 2024년까지의 트윗 데이터셋을 분석하여, 감정 분포(sentiment distributions)와 감정 표현(sentiment expression)이 정당 이념(party ideology)과 어떻게 관계되는지를 탐구합니다.

- **Performance Highlights**: 연구 결과, 두 모델 모두 모든 정당에서 중립적(Neutral) 감정이 지배적이라고 일관되게 식별하였고, 이념적 구분과 일치하는 유의미한 부정적(Negative) 및 긍정적(Positive) 감정의 변화를 보여주었습니다. 특히, Vox는 더 높은 부정적 감정을 보인 반면, PSOE는 비교적 높은 긍정적 감정을 나타내어 정치적 메시지의 감정적 호소가 이념적 입장을 반영한다는 가설을 지지합니다.



### Prompt-Guided Internal States for Hallucination Detection of Large Language Models (https://arxiv.org/abs/2411.04847)
- **What's New**: 이번 논문에서는 LLMs의 hallucination(환각) 탐지 성능을 향상시키기 위한 새로운 프레임워크인 PRISM을 제안합니다. 이 방법은 in-domain 데이터만으로 cross-domain 성능을 증대시키는 것을 목표로 하고 있습니다.

- **Technical Details**: PRISM(프롬프트 안내 내부 상태)은 적절한 프롬프트를 사용하여 LLM의 내부 상태에서 텍스트의 진실성과 관련된 구조의 변화를 유도합니다. 이 구조는 다양한 도메인 간에 더욱 두드러지고 일관되게 됩니다.

- **Performance Highlights**: 실험 결과, PRISM을 기존의 hallucination 탐지 기법에 통합했을 때, cross-domain 일반화 성능이 크게 향상되었습니다. 이는 특정 도메인에서 교육받은 탐지기가 다른 도메인에서도 우수한 성능을 발휘할 수 있도록 합니다.



### VTechAGP: An Academic-to-General-Audience Text Paraphrase Dataset and Benchmark Models (https://arxiv.org/abs/2411.04825)
Comments:
          21 pages, 3 figures

- **What's New**: 본 논문에서는 VTechAGP라는 새로운 데이터셋을 출시하였습니다. 이 데이터셋은 4,938개의 문서 수준의 학위 논문과 일반 청중을 위한 초록 쌍으로 구성되어 있으며, 25년에 걸쳐 8개 대학에서 작성된 것입니다.

- **Technical Details**: DSPT5는 논문에서 제안한 동적 소프트 프롬프트 생성 언어 모델로, 대조적 생성 손실 함수와 군중 샘플링 디코딩 전략을 활용합니다. DSPT5는 기술 키워드를 동적으로 조정하며, 약 2억 2천만 개의 파라미터를 갖는 T5 모델을 기반으로 합니다.

- **Performance Highlights**: DSPT5는 여러 최신 대형 언어 모델(LLMs)과 비교했을 때 경쟁력 있는 결과를 보여주며, 문서 수준에서의 텍스트 패러프레이징 작업에서 SOTA LLM들보다 더 나은 성능을 발휘하고 있습니다.



### When Does Classical Chinese Help? Quantifying Cross-Lingual Transfer in Hanja and Kanbun (https://arxiv.org/abs/2411.04822)
- **What's New**: 본 논문은 한국과 일본의 고문서 처리에서 고전 중국어 자료를 사용하는 것에 대한 가정에 의문을 제기합니다. 특히 Classical Chinese에서 Hanja 및 Kanbun으로의 cross-lingual transferability에 대한 의문을 제시합니다.

- **Technical Details**: 기계 번역(Machine Translation), 명명된 개체 인식(Named Entity Recognition), 구두점 복원(Punctuation Restoration) 작업을 통해 평가한 결과, Hanja로 작성된 고대 한국 문서에 대한 언어 모델 성능에 거의 영향을 주지 않는 것으로 나타났습니다. F1-score는 $	ext{±}0.0068$로 차이가 나고, 번역에서는 최대 $+0.84$ BLEU 점수를 기록했습니다.

- **Performance Highlights**: 여러 모델 크기와 아키텍처, 도메인 특정 데이터셋 전반에 걸쳐 이러한 한계는 지속적으로 나타났습니다. 데이터가 증가함에 따라 Classical Chinese 자료의 이점은 빠르게 감소하며, 한국과 일본의 역사적 문서에서 매우 저자원(low-resource) 상황에서만 개선 효과가 존재합니다.



### LuxBank: The First Universal Dependency Treebank for Luxembourgish (https://arxiv.org/abs/2411.04813)
Comments:
          Accepted at 22nd Workshop on Treebanks and Linguistic Theories (TLT 2024)

- **What's New**: LuxBank는 룩셈부르크어에 대한 첫 번째 Universal Dependencies (UD) Treebank로, 룩셈부르크어의 통사적 주석과 분석의 공백을 해결합니다. 본 연구는 400,000명 정도의 사람들에 의해 사용되고 있는 룩셈부르크어를 위한 공식적인 주석 가이드라인을 설정하여 대규모 정량적 분석의 기초를 제공합니다.

- **Technical Details**: 룩셈부르크어는 독일어와 밀접하게 관련된 서독 게르만 언어로, 그동안 UD Treebank에 포함되지 않았습니다. LuxBank는 룩셈부르크어를 UD 틀에 통합하여 서독 게르만 언어 내에서의 통사적 변화를 이해하고, 더 작은 반표준 언어 문서화 모델을 제공하는 것을 목표로 합니다. 이 문서에서 LuxBank의 첫 번째 주석 세트와 annotation 과정의 구체적인 방법론을 설명합니다.

- **Performance Highlights**: LuxBank는 언어학자들, 언어 학습자들, 맞춤법 및 문법 검사기를 개발하는 도구 등 다양한 용도로 활용될 수 있습니다. 또한, 룩셈부르크어의 통계적 특성을 처음으로 대규모로 분석하는 자료로 자리매김할 것입니다. 이 프로젝트는 룩셈부르크어의 NLP 및 언어 연구 커뮤니티에서 중요한 자원으로 기능하게 됩니다.



### Kwai-STaR: Transform LLMs into State-Transition Reasoners (https://arxiv.org/abs/2411.04799)
Comments:
          6 pages, 2 figures

- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)의 수학적 추론 능력 향상을 위한 새로운 접근법인 Kwai-STaR 프레임워크를 제안합니다. 이 프레임워크는 초기 미해결 상태에서 최종 해결 상태로의 전환 과정을 정의하고, LLM을 상태 전환 추론기(State-Transition Reasoner)로 변환하여 직관적인 추론 능력을 개선합니다.

- **Technical Details**: Kwai-STaR는 세 가지 주요 단계로 구성됩니다: (1) 수학적 추론에 맞춘 상태 공간(state space) 정의, (2) 상태 전환 데이터(state-transition data) 생성, (3) 커리큘럼(training strategy)을 통한 원래 LLM을 상태 전환 추론기로 변환. 이 과정에서 LLM은 특정 행동 세트(action set)에서 하나의 행동을 선택하여 현재 상태를 새로운 상태로 전환합니다.

- **Performance Highlights**: Kwai-STaR 데이터셋에서의 훈련 후, Mistral-7B 및 LLaMA-3을 포함한 일반 LLM들이 GSM8K 및 GSM-Hard 데이터셋에서 현저한 성능 향상을 이루었습니다. Kwai-STaR는 다른 데이터 증강 방법에 비해 데이터 효율성을 높이며, 복잡한 추론 패러다임 없이 단일 통과(single-pass) 정확도를 달성합니다.



### AlignXIE: Improving Multilingual Information Extraction by Cross-Lingual Alignmen (https://arxiv.org/abs/2411.04794)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 정보 추출(Information Extraction, IE)에서의 교차 언어 정렬(cross-lingual alignment) 강화를 위한 새로운 모델인 AlignXIE를 제안합니다. 이 모델은 특히 비영어 언어에서의 IE 정렬을 위한 코드 생성 기반 접근 방식을 사용하여 다양한 언어 간의 스키마 정리를 통합하고 최적화합니다.

- **Technical Details**: AlignXIE는 다음 두 가지 전략을 통해 교차 언어 IE 정렬을 개선합니다. 첫째, IE 작업을 코드 생성 작업으로 변환하여 다양한 언어에서 스키마의 일관성을 유지합니다. 둘째, 번역된 인스턴스 예측(task)을 포함하여 IE 추출 프로세스를 정렬합니다. 이를 위해 ParallelNER라는 대규모 이중 언어 병렬 데이터셋을 구축했으며, 이 데이터셋은 257,190개의 샘플로 구성되어 있습니다. 이 과정에서는 LLM 기반의 자동 파이프라인을 활용합니다.

- **Performance Highlights**: AlignXIE는 63개의 IE 벤치마크에서 평가되어, ChatGPT보다 30.17% 높은 성능을 보였고, 소위 SoTA(State of the Art) 모델보다도 20.03% 뛰어난 성능을 기록했습니다. AlignXIE는 다양한 설정에서 모든 중국어 IE 벤치마크에서 SoTA를 달성하고, 영어 RE 및 ED 작업에서도 평균 3.03 및 2.92 포인트의 개선을 보였습니다.



### A study of Vietnamese readability assessing through semantic and statistical features (https://arxiv.org/abs/2411.04756)
- **What's New**: 이번 연구에서는 베트남어 텍스트의 가독성을 평가하기 위한 통계적(feature) 및 의미적(semantic) 접근 방식을 통합한 새로운 방법론을 도입하였습니다. 이것은 통계적 특징만을 활용한 기존의 연구를 넘어서서, 고급 언어 모델을 활용하여 보다 정교한 분석을 시도한 것입니다.

- **Technical Details**: 연구에서는 베트남어 텍스트 가독성 데이터셋인 ViRead 외에도 OneStopEnglish와 RACE의 두 데이터셋을 활용하였습니다. 또한, PhoBERT, ViDeBERTa, ViBERT와 같은 최첨단 언어 모델을 사용하여 의미적 분석을 실시하였고, Support Vector Machine (SVM), Random Forest, Extra Trees와 같은 다양한 기계 학습 모델로 실험을 진행하였습니다.

- **Performance Highlights**: 결과적으로 의미적 및 통계적 특징을 결합한 접근 방식이 각각의 방법을 따로 사용하는 것보다 가독성 분류의 정확도를 현저히 향상시키는 것으로 드러났습니다. 또한, 특정 통계적 특징 그룹이 가독성에 미치는 영향을 깊이 분석하여 단어 수나 평균 단어 길이 등의 특징이 모델 성능에 큰 영향을 미친다는 것을 확인하였습니다.



### RetrieveGPT: Merging Prompts and Mathematical Models for Enhanced Code-Mixed Information Retrieva (https://arxiv.org/abs/2411.04752)
Comments:
          Accepted at FIRE 2024 (Track: Code-Mixed Information Retrieval from Social Media Data)

- **What's New**: 이 연구는 코드 혼합(concode-mixing) 대화로부터 정보를 추출하는 새로운 접근 방식을 제시합니다. 특히 Roman 문자로 전사된 벵골어로 혼합된 영어 대화에 집중하며, GPT-3.5 Turbo를 활용해 문서의 관련성을 자동으로 식별하는 기법을 개발했습니다.

- **Technical Details**: 연구는 페이스북에서 수집한 코드 혼합 대화 데이터를 사용하였으며, 이를 기반으로 쿼리와 문서 간의 관련성을 평가하는 수학적 모델을 설계했습니다. GPT-3.5 Turbo를 적절히 프롬프트하여 문서의 순차적 특성을 고려한 관련 문서 감지를 수행합니다.

- **Performance Highlights**: 실험 결과, 제안한 접근 방식이 복잡한 코드 혼합 디지털 대화에서 관련 정보를 효과적으로 추출할 수 있음을 보여주었습니다. 이는 다국어 및 비공식 텍스트 환경에서 자연어 처리(NLP)의 발전에 기여합니다.



### BhasaAnuvaad: A Speech Translation Dataset for 14 Indian Languages (https://arxiv.org/abs/2411.04699)
Comments:
          Work in Progress

- **What's New**: 본 연구는 인도 언어를 위한 자동 음성 번역 (AST) 시스템의 발전을 위한 BhasaAnuvaad라는 대규모 공개 데이터셋을 소개합니다. 이 데이터는 14개의 인도 언어로 44,400시간 이상의 음성과 17백만 개의 텍스트 세그먼트를 포함하고 있으며, 대화식 및 비형식적 언어 번역의 필요성을 충족하고자 합니다.

- **Technical Details**: BhasaAnuvaad 데이터셋은 (1) 기존 자원에서 수집된 선별된 데이터, (2) 대규모 웹 마이닝, (3) 합성 데이터 생성을 포함한 세 가지 카테고리로 구성됩니다. 이 데이터셋은 인도 언어의 자질적인 발음을 반영하고, 비형식적인 문맥에서의 번역 성능 향상을 목표로 하고 있습니다.

- **Performance Highlights**: 기존 AST 시스템은 읽기 음성에서 양호한 성능을 보였으나, 자발적 음성에서는 뛰어난 성능을 발휘하지 못했습니다. 연구 결과는 더욱 포괄적인 데이터셋과 훈련 목표의 필요성을 강조하며, BhasaAnuvaad의 출시는 인도 언어의 AST 시스템을 개선하는 데 중요한 기여를 할 것입니다.



### Hands-On Tutorial: Labeling with LLM and Human-in-the-Loop (https://arxiv.org/abs/2411.04637)
Comments:
          To be presented at COLING 2025

- **What's New**: 본 논문에서는 머신러닝 모델의 학습과 배포에 필수적인 대량의 인간 주석 데이터(labeling data)를 쉽게 처리할 수 있는 다양한 전략들을 다룹니다. 특히, 비싼 인간 주석 비용과 시간 소모를 줄이기 위한 최신 연구 결과와 실용적인 기법들을 소개합니다.

- **Technical Details**: 주요 전략으로는 Synthetic Training Data 생성, Active Learning, Hybrid Labeling 등이 있으며, 각 전략의 기본 원리와 함께 장단점을 분석합니다. 논문은 데이터 주석자로서 품질 확보 및 관리에 대한 모범 사례를 제공합니다. 또한, Hybrid Annotation 설정을 구현하기 위한 워크숍이 포함되어 있어 참가자들이 실습을 통해 배울 수 있습니다.

- **Performance Highlights**: 이 튜토리얼은 특히 NLP(Natural Language Processing) 실무자들을 위해 설계되었으며, 연구와 산업 배경을 가진 사람들이 데이터 라벨링 프로젝트를 최적화할 수 있도록 돕습니다. 실제 사례를 통해 전략의 효과를 증명하고 있습니다.



### FASSILA: A Corpus for Algerian Dialect Fake News Detection and Sentiment Analysis (https://arxiv.org/abs/2411.04604)
Comments:
          16 pages, 6 Figuers

- **What's New**: 이번 연구에서는 알제리 방언(Algerian Dialect, AD)에 대한 Fake News(FN) 탐지 및 감정 분석(Sentiment Analysis, SA)을 위한 전문화된 말뭉치(FASSILA)를 개발했습니다. 이 데이터셋은 10,087개의 문장과 19,497개 이상의 고유 단어로 구성되어 있으며, 알제리 방언의 언어 자원 부족 문제를 해결하고자 합니다.

- **Technical Details**: FASSILA 말뭉치는 소셜 미디어 플랫폼에서 수집된 데이터를 바탕으로 하며, 머신 러닝(Machine Learning, ML) 및 딥 러닝(deep learning) 모델을 활용하여 Fake News 탐지 및 감정 분석을 수행합니다. 연구에서는 Support Vector Machines (SVM), Logistic Regression (LR), Decision Trees (DT) 등의 전통적인 ML 기법과 AraBERTv02, MarBERTv2, DziriBERT와 같은 Transformer 기반 모델을 사용해 실험을 진행하였습니다.

- **Performance Highlights**: BERT 기반 모델과 ML 모델을 사용한 분류 실험에서 유망한 결과를 도출하였으며, 주석 작업에서 remarkable Inter-Annotator Agreement를 기록했습니다. 논문에서 제안한 FASSILA 데이터세트는 GitHub에서 무료로 제공되며, 향후 연구의 발전을 촉진할 수 있는 자료로 활용될 것입니다.



### Tibyan Corpus: Balanced and Comprehensive Error Coverage Corpus Using ChatGPT for Arabic Grammatical Error Correction (https://arxiv.org/abs/2411.04588)
Comments:
          17 pages, 11 figures

- **What's New**: 이번 연구에서는 아랍어의 문법 오류 교정을 위한 새로운 말뭉치인 'Tibyan'을 개발했습니다. ChatGPT를 활용하여 아랍어 문장에서의 문법 오류를 증가시키고, 이를 개선하는 기술을 적용했습니다.

- **Technical Details**: Tibyan 코퍼스는 주로 ChatGPT를 이용하여 여러 출처에서 수집된 아랍어 문장 쌍을 바탕으로 생성되었습니다. 논문의 여러 단계에서 수집된 코퍼스는 49개의 오류 유형을 포함하며, 여기에는 표기법(orthography), 형태론(morphology), 구문(syntax), 의미론(semantics), 구두점(punctuation), 합병(merge), 분할(split)이 포함됩니다.

- **Performance Highlights**: Tibyan 코퍼스는 약 600K 토큰을 포함하고 있으며, 본 연구는 아랍어 문법 오류 교정 시스템의 정확성과 견고성을 높이는 데 기여할 것으로 기대됩니다.



### The State and Fate of Summarization Datasets (https://arxiv.org/abs/2411.04585)
- **What's New**: 자동 요약의 적용 범위와 중요성이 증가하고 있지만, 기존의 데이터셋에 대한 표준화된 용어와 일관성이 부족하다는 문제가 있습니다. 본 논문에서는 133개의 데이터셋을 분석하여 새롭게 개발된 온톨로지를 통해 데이터셋의 속성과 수집 방법, 분포를 체계화했습니다.

- **Technical Details**: 온톨로지는 샘플 속성(언어, 도메인 등), 수집 방법, 배포 방식을 기준으로 데이터셋을 분류하여 각 데이터셋의 경향과 단점을 드러냅니다. 연구 결과, 저자들은 뉴스 도메인에 대한 과도한 의존과 저자원 언어의 높은 품질 저하 문제를 지적했습니다.

- **Performance Highlights**: 우리는 웹 인터페이스 및 표준화된 요약 데이터 카드와 같은 두 가지 자원을 제공하여 연구자들이 데이터셋을 손쉽게 발견하고 비교할 수 있도록 지원하며, 이는 향후 연구 방향을 제시하는 데 도움을 줄 것입니다.



### Multistage Fine-tuning Strategies for Automatic Speech Recognition in Low-resource Languages (https://arxiv.org/abs/2411.04573)
- **What's New**: 이 논문은 OpenAI의 Whisper 모델을 활용하여 자원이 부족한 언어에서 자동 음성 인식(ASR) 성능을 향상시키기 위한 새로운 다단계 파인튜닝 전략을 제시합니다. 특히, 말라사르 언어에 대한 ASR 모델을 만들기 위해 언어적으로 유사한 언어를 통해 모델을 점진적으로 조정하는 접근 방식을 사용합니다.

- **Technical Details**: 이 연구는 약 10,000명이 사용하는 다라비다어인 말라사르 언어를 대상으로 하며, 이 언어는 고유 스크립트의 부재와 디지털 또는 음성 데이터 리소스의 결여로 인해 기술적 지원에 큰 도전에 직면해 있습니다. 연구팀은 말라사르 음성을 타밀 스크립트로 전사하여 말라사르 코퍼스를 제작하였으며, 초기 타밀 ASR 모델을 구축한 후, 이를 말라사르 데이터에 세밀하게 fine-tuning 하였습니다.

- **Performance Highlights**: 이 다단계 파인튜닝 전략은 단일 데이터에 대한 직접적인 파인튜닝과 비교할 때 상당한 성과 향상을 보여주었으며, 최종 단어 오류율(WER)은 51.9%로 나타났고, 판별 후 처리를 통해 더욱 향상된 47.3%로 낮출 수 있었습니다. 이러한 결과는 저자원 언어의 ASR 시스템 개발에 있어 매우 효과적인 접근법임을 입증합니다.



### Pruning Literals for Highly Efficient Explainability at Word Lev (https://arxiv.org/abs/2411.04557)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문은 Tsetlin Machine(TM)의 클라우스(pruning) 알고리즘을 설계하여 자연어 처리(NLP)에서 모델의 해석 가능성을 높이고 전달력을 개선하기 위한 방안으로서, 불필요한 리터럴(literal)을 제거하는 기법을 제안합니다.

- **Technical Details**: 제안된 방법은 원래의 Tsetlin Machine 모델에서 우연히 배치된 리터럴을 제거하여 더 효율적으로 해석 가능한 구조를 제공합니다. 이를 위해 리터럴의 빈도(frequency)를 분석하여 클라우스를 최적화하고, Tsetlin attention map(TAM)과 인간의 주의 맵(HAM) 간의 유사성을 평가하여 효과를 입증합니다.

- **Performance Highlights**: 실험 결과, 제안된 기법은 모델의 정확도를 4%에서 9%까지 향상시키며, YELP-HAT 데이터셋에서의 주의 맵이 기존 TM보다 인간의 주의 맵과 더 잘 일치함을 보여주었습니다.



### Meta-Reasoning Improves Tool Use in Large Language Models (https://arxiv.org/abs/2411.04535)
- **What's New**: 이번 논문에서는 Tool selECTion via meta-reasONing (TECTON)이라는 새로운 시스템을 제안합니다. TECTON은 외부 도구를 활용하여 대규모 언어 모델이 수학 추론과 같은 복잡한 작업에서의 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: TECTON은 두 단계로 구성된 시스템으로, 첫 번째 단계는 특정 작업을 해결하기 위해 맞춤화된 언어 모델 헤드를 사용하여 후보 도구를 생성하고, 두 번째 단계에서는 고정된 LLM을 사용하여 후보 도구를 재검토하고 최종 선택을 합니다. 이를 통해 기계 학습 모델의 일반화 능력을 효과적으로 활용합니다.

- **Performance Highlights**: TECTON은 다양한 수학 추론 데이터셋에서 상당한 성능 향상을 보여주는 결과를 낳았으며, 분포 내와 분포 외 모두에서 강력한 기준 모델을 초월하는 성능을 기록했습니다.



### Tomato, Tomahto, Tomate: Measuring the Role of Shared Semantics among Subwords in Multilingual Language Models (https://arxiv.org/abs/2411.04530)
Comments:
          8 pages, 9 figures

- **What's New**: 이 연구는 다양한 단어 선택에 대한 사람의 언어 이해 능력을 언급하며, 이러한 직관이 모든 서브워드를 개별적인 임베딩(embeddings)으로 표현하는 언어 모델에 얼마나 전이되는지를 측정하는 최초의 단계를 제시합니다.

- **Technical Details**: 연구에서는 의미적으로 유사한 서브워드를 병합하여 'semantic tokens'를 형성하고, 이를 이용해 encoder-only multilingual language models (mLMs)를 평가합니다. 이 과정에서 5가지 이질적인 다국어 다운스트림 작업에서 업데이트된 mLMs의 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 일반적인 공유 의미(shared semantics)가 구분된 토크나이저(tokenizers)와 모델 크기가 다른 mLMs에서 예측 성능을 크게 향상시킬 수 있음을 보여주었습니다. 특히, semantic tokens을 사용한 zero-shot 결과가 특정 분류 작업에서 원래 모델보다 나은 성능을 보였다는 점이 주목할 만합니다.



### Thanos: Enhancing Conversational Agents with Skill-of-Mind-Infused Large Language Mod (https://arxiv.org/abs/2411.04496)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 대화에서 사회적 유대감을 높이기 위한 새로운 접근 방식으로 'skill-of-mind(스킬-오브-마인드)'라는 개념을 제안하고, 이를 활용한 새로운 데이터셋과 모델인 'Thanos'를 소개합니다.

- **Technical Details**: Multifaceted Skill-of-Mind 데이터셋은 약 10만 개의 대화로 구성되어 있으며, 여러 상호작용 시나리오(예: 장기적 대화, 상담, 과제 지향)에 걸친 다면적 대화 기술을 포함합니다. 이 데이터셋은 인구 통계, 페르소나(persona), 경험 법칙(rules of thumb)와 같은 다양한 사회적 맥락에 기반합니다. Thanos 모델은 1B, 3B, 8B 파라미터를 가진 LLM입니다.

- **Performance Highlights**: Thanos 모델은 skill-of-mind 프로세스를 효과적으로 시연하며, 다양한 도메인에서 다면적 기술을 추론하는 데 강력한 일반화 능력을 보여줍니다. 또한, LLM 기반 대화 에이전트가 생성하는 응답 품질을 크게 향상시키고, 인간 평가에서 친사회적 행동을 촉진하는 우수한 성능을 보입니다.



### ML-Promise: A Multilingual Dataset for Corporate Promise Verification (https://arxiv.org/abs/2411.04473)
Comments:
          6 pages

- **What's New**: 본 논문은 정치인, 기업 리더 및 공공 인사의 약속이 공공 인식, 신뢰 및 기관의 평판에 미치는 중대한 영향을 강조하며, 약속 검증(Promise Verification) 개념을 도입합니다. 이는 약속 식별, 증거 평가 및 검증 타이밍 평가 등 여러 단계를 포함한 체계적인 접근법입니다.

- **Technical Details**: ML-Promise라는 최초의 다국어 데이터셋은 영어, 프랑스어, 중국어, 일본어 및 한국어를 포함하여 ESG(환경, 사회 및 거버넌스) 보고서의 약속을 심층적으로 검증하는 데 중점을 두고 있습니다. 검증 과정에는 약속 식별, 지원 증거의 명확성 및 검증 타이밍 설정이 포함됩니다. RAG(검색 보강 생성) 접근법을 이용하여 성능 향상을 도모하였으며, 다양한 언어에서 실험을 진행했습니다.

- **Performance Highlights**: ML-Promise 데이터셋은 5개국의 ESG 보고서를 포함하며, 약속-증거 쌍의 품질 평가에 대한 필요성을 강조합니다. 실험 결과, 한국과 대만 기업은 단기 약속을 선호하는 반면, 나머지 국가에서는 장기 약속을 선호하는 경향이 나타났으며, 이는 다양한 국가 간 ESG 보고서의 비교 필요성을 보여줍니다.



### Gradient Localization Improves Lifelong Pretraining of Language Models (https://arxiv.org/abs/2411.04448)
Comments:
          EMNLP Findings 2024

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)가 서로 다른 유형의 지식을 어떻게 저장하는지에 대한 메커니즘을 탐구하며, 시간에 민감한 엔티티(entity)와 관련된 두 가지 지식을 분석하는 데 중점을 두었습니다. 이는 기존의 kontinual learning 방법이 새로운 정보를 제대로 습득하지 못하고 이전의 학습된 정보를 잊어버리는 현상에 기여한다는 가설로 이어집니다.

- **Technical Details**: 이 연구는 LLM이 특정 레이어에서 새로운 엔티티와 업데이트된 엔티티에 대한 더 큰 gradient 노름을 나타낸다는 점을 관찰합니다. 이러한 레이어에 대한 파라미터 업데이트를 목표로 삼는 방법을 제안하며, 이는 시간적 변화가 포함된 언어의 지속적인 재학습 성능을 향상시키는 데 기여합니다. 또한 지식 집합의 중요한 엔티티 범위를 예측하는 작업에서 gradient 업데이트를 비교하여 기존의 연속적인 pretraining 방법과의 불일치를 규명합니다.

- **Performance Highlights**: 제안된 방법은 다양한 크기의 transformer 언어 모델에서 관찰된 특성 gradient 패턴을 기반으로 하며, knowledge probing tasks에서 성능 향상을 입증합니다. 특히, 기존의 continuous learning 방법과 함께 적용했을 때 유의미한 성능 개선을 보여줍니다.



### ACCIO: Table Understanding Enhanced via Contrastive Learning with Aggregations (https://arxiv.org/abs/2411.04443)
- **What's New**: 이번 연구에서는 ACCIO라는 새로운 접근 방식을 도입하여 원본 테이블과 피벗 테이블을 비교함으로써 테이블 이해를 향상시킵니다. ACCIO는 대조 학습(contrastive learning)을 활용하여 두 테이블 쌍을 밀접하게 연결하는 인코더를 훈련합니다.

- **Technical Details**: ACCIO는 원본 데이터와 해당 피벗 테이블을 대조 학습을 통해 조정하는 방식으로 작동합니다. 이 논문은 테이블 임베딩(table embedding)을 위해 테이블 쌍을 활용한 첫 번째 시도로서, 마크로 F1 점수 91.1을 달성하며 뛰어난 성능을 보입니다. 대조 학습(contrastive learning)은 유사한 데이터 샘플 간의 거리를 최소화하고 유사하지 않은 데이터 샘플 간의 거리를 최대화하는 데 초점을 맞춥니다.

- **Performance Highlights**: ACCIO는 열 유형 주석(column type annotation) 작업을 통해 성능을 검증하였으며, 기존 최첨단 방법들과 비교하여 경쟁력 있는 마크로 F1 점수 91.1을 기록했습니다.



### One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity (https://arxiv.org/abs/2411.04427)
Comments:
          17 pages, 10 figures

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)을 인간을 대체하여 행동 연구에 활용할 수 있는 가능성을 탐구하였습니다. 특히, 이러한 모델들이 인간의 개념적 다양성을 얼마나 잘 포착할 수 있는지를 조사했습니다.

- **Technical Details**: 연구자들은 시뮬레이션된 개체들의 내부 변동성과 집단 수준 변동성을 연관시켜 LLM '집단'의 개념적 다양성을 측정하기 위해 새로운 방법을 사용했습니다. 비정렬(non-aligned) 및 정렬(aligned) LLMs를 두 가지 풍부한 인간 행동 데이터 영역(단어-색상 연관 및 개념 유사성 판단)에서 평가했습니다.

- **Performance Highlights**: 대부분의 LLM 모델은 인간과 유사한 개념적 다양성을 달성하지 못하며, 정렬된 모델은 일반적으로 비정렬 및 세부 조정된 모델보다 덜 다양한 개념적 표현을 보입니다. 이 결과는 모델의 안전성(가치 정렬 증가)과 개념적 표현의 다양성 감소 간의 잠재적인 상충 관계를 강조합니다.



### DELIFT: Data Efficient Language model Instruction Fine Tuning (https://arxiv.org/abs/2411.04425)
- **What's New**: 이번 논문은 DELIFT(Data Efficient Language model Instruction Fine-Tuning)라는 새로운 알고리즘을 소개합니다. DELIFT는 자연어 처리 모델의 세 가지 주요 단계인 instruction tuning, task-specific fine-tuning, continual fine-tuning 전 단계에서 데이터 선택을 체계적으로 최적화하여 데이터 효율성을 극대화합니다.

- **Technical Details**: DELIFT의 핵심 혁신은 데이터 샘플의 유용성을 평가하기 위한 pairwise utility metric입니다. 이 메트릭은 데이터 샘플이 모델의 현재 능력 및 다른 샘플에 대해 얼마나 유용한지를 정량화합니다. DELIFT는 또한 다양한 submodular 기능을 이용하여 각 단계에서 유용한 다양한 최적의 데이터 하위 집합을 선택합니다.

- **Performance Highlights**: DELIFT를 사용하면 약 70%의 fine-tuning 데이터 크기 감소가 가능하며, 성능 저하 없이 기존 방법에 비해 효율성을 26% 이상 향상시킵니다. 실험 결과는 DELIFT가 다양한 작업에서 기존 방법보다 우수한 성능을 발휘한다는 것을 보여줍니다.



### Bayesian Calibration of Win Rate Estimation with LLM Evaluators (https://arxiv.org/abs/2411.04424)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 최근 대규모 언어 모델(LLMs)을 평가자로 사용하여 텍스트 생성의 품질을 평가할 수 있는 가능성이 드러났습니다. 하지만 이러한 LLM 평가자를 단순히 사용하면 신뢰할 수 없는 결과를 초래할 수 있습니다. 이를 해결하기 위해 본 논문에서는 Bayesian Win Rate Sampling (BWRS)과 Bayesian Dawid-Skene  두 가지 보정(calibration) 방법을 제안합니다.

- **Technical Details**: 제안된 BWRS와 Bayesian Dawid-Skene 방법은 Bayesian 추론(Bayesian inference)을 활용하여 LLM 평가 결과를 기반으로 생성 언어 모델의 실제 승률을 더 정확하게 추론합니다. 이 방법들은 텍스트 생성기 간의 승률을 추정할 때 LLM 평가자와 인간 평가 간의 일치를 개선하는데 효과적입니다.

- **Performance Highlights**: 이 연구는 이야기 생성, 요약, 지침 수행 작업을 포함하는 여섯 개 데이터세트를 통해 BWRS와 Bayesian Dawid-Skene의 효과를 실증적으로 검증하였습니다. 연구 결과는 두 방법이 LLM 평가자의 승률 추정 편향을 효과적으로 줄이는 데 기여하여 NLP에서 보다 신뢰할 수 있는 자동 평가 방법으로 나아가는 가능성을 보여줍니다.



### Measuring short-form factuality in large language models (https://arxiv.org/abs/2411.04368)
Comments:
          Blog post: this https URL

- **What's New**: 이번 논문에서는 언어 모델의 능력을 평가하는 새로운 벤치마크인 SimpleQA를 소개합니다. 이 벤치마크는 짧고 사실을 요구하는 질문에 대한 모델의 응답 능력을 평가하며, 특정한 두 가지 특성을 우선시하며 설계되었습니다.

- **Technical Details**: SimpleQA는 총 4,326개의 짧고 사실을 요구하는 질문을 포함하고 있으며, 각 질문에는 명확하고 변하지 않는 정답이 존재합니다. 이 데이터셋은 AI 트레이너와 인간 주석자에 의해 질문-답변 쌍이 생성되었고, 두 개의 독립적인 트레이너의 응답이 일치할 때만 데이터셋에 포함됩니다. 각 질문은 특정 주제를 아우르며, 단일한 답변이 가능하도록 설계되었습니다.

- **Performance Highlights**: SimpleQA의 목표는 최신 모델의 factuality를 평가하는 신뢰할 수 있는 데이터셋이 되는 것입니다. SimpleQA는 GPT-4와 같은 최신 모델에 대해 도전적인 평가를 제공하며, 모델의 성능이 50% 미만으로 나오기도 했습니다.



### CodeTree: Agent-guided Tree Search for Code Generation with Large Language Models (https://arxiv.org/abs/2411.04329)
- **What's New**: CodeTree라는 새로운 프레임워크를 제안하여 LLM(대형 언어 모델)의 코드 생성 작업에서 탐색 공간을 효과적으로 탐색할 수 있도록 하였습니다. 이는 여러 단계의 탐색 과정을 통해 코드 솔루션을 생성하고 개선하는 방식입니다.

- **Technical Details**: CodeTree는 통합된 트리 구조를 사용하여 다양한 코딩 전략을 탐색하고, 해당 전략에 대응하는 코드 솔루션을 생성하며, 솔루션을 다듬는 과정으로 구성됩니다. 이 과정에서 'Critic Agent'가 중요한 결정을 내리며, 성능 향상에 기여합니다.

- **Performance Highlights**: CodeTree는 7개의 코드 생성 벤치마크에서 평가되었으며, 기존 강력한 기준선 대비 유의미한 성능 향상을 보여주었습니다. HumanEval에서 95.1, MBPP에서 98.7, CodeContests에서 43.0의 결과를 달성하였으며, SWEBench 벤치마크에서도 상당한 성능 개선을 이끌어냈습니다.



### Balancing Transparency and Accuracy: A Comparative Analysis of Rule-Based and Deep Learning Models in Political Bias Classification (https://arxiv.org/abs/2411.04328)
- **What's New**: 이 연구는 미국 뉴스 기사에서 정치적 편향을 자동으로 감지하는 시스템을 개발하는 데 초점을 맞추고 있습니다. 이는 규칙 기반(rule-based) 모델과 깊은 학습(deep learning) 모델을 비교하여 편향을 분류하는 방법을 탐색하고 있습니다.

- **Technical Details**: 본 연구는 CNN 모델과 규칙 기반 모델을 사용하여 정치적 편향을 분류합니다. CNN 모델은 대량의 데이터 세트를 필요로 하는 반면, 규칙 기반 모델은 감정 감지(sentiment detection) 및 언어적 특징을 활용하여 편향을 분류합니다. 신뢰성과 투명성 측면에서 규칙 기반 모델이 다양한 데이터 조건에서도 일관된 성능을 유지하는 반면, CNN 모델은 훈련 세트에 의존하여 새로운 데이터에 어려움을 겪는다고 합니다.

- **Performance Highlights**: 규칙 기반 모델은 FOX 뉴스에 대해 뚜렷한 우향적 편향을 보여주며, CNN 모델보다 보다 실용적으로 사용될 수 있습니다. 여러 데이터 세트에서의 실험 결과, 규칙 기반 모델이 일정한 정확도를 유지하는 반면, CNN 모델의 성능은 훈련 데이터와 관련성이 낮은 데이터에서 저하되는 경향이 있습니다.



### A Multilingual Sentiment Lexicon for Low-Resource Language Translation using Large Languages Models and Explainable AI (https://arxiv.org/abs/2411.04316)
Comments:
          This work is part of a PhD proposal in Information Technology at the University of Pretoria, supervised by Dr. Mike Wa Nkongolo and co-supervised by Dr. Phil van Deventer, under the Low-Resource Language Processing Lab in the Department of Informatics

- **What's New**: 이 연구는 남아프리카와 콩고 민주 공화국의 다언어 환경에서 감정 분석 및 번역의 과제를 해결하기 위해 다국어 사전을 개발했습니다. 이 사전은 프랑스어와 Tshiluba 캔 번역을 포함하여 영어, 아프리칸스어, 세페디어, 줄루어로 확대되었습니다.

- **Technical Details**: 새로 개발된 다국어 사전은 특정 언어의 감정 점수를 통합하여 감정 분류의 문화적 적합성을 향상시킵니다. 기계 학습 모델로는 랜덤 포레스트(Random Forest), 서포트 벡터 머신(Support Vector Machine, SVM), 결정 트리(Decision Trees), 가우시안 나이브 베이즈(Gaussian Naive Bayes, GNB)가 사용되어, 저자원 언어( low resource languages, LRLs)에서 감정을 예측합니다. 그중 랜덤 포레스트 모델이 특히 우수한 성능을 보여주었습니다. 또한, BERT(비편향 인코더 표현 변환기)가 맥락 기반의 감정을 높은 정확도로 예측하여 99% 정확도와 98% 정밀도를 달성했습니다.

- **Performance Highlights**: 연구 결과, 제안된 사점과 기계 학습 모델이 남아프리카와 DRC에서 LRL 절에서 번역 및 감정 분석을 크게 향상시킨 것으로 나타났습니다. BERT 모델의 예측은 설명 가능한 AI(Explainable AI, XAI)로 명확히 하여 투명성을 개선하고 감정 분류에 대한 신뢰를 높였습니다.



### Improving Bilingual Capabilities of Language Models to Support Diverse Linguistic Practices in Education (https://arxiv.org/abs/2411.04308)
- **What's New**: 본 연구는 다양한 언어 배경을 가진 학습자를 지원하기 위한 다국어 대형 언어 모델(MLLM)의 효과를 평가합니다. 특히 Spanglish와 같은 이중언어 학생 작문에 대한 LLM의 성능을 분석하고, 선행 연구에서 자연어 처리의 언어 전환(코드 전환) 능력의 한계를 다룹니다.

- **Technical Details**: 이 연구는 사전 훈련된 LLM을 사용하여 영어, 스페인어, Spanglish로 표현된 과학 및 사회 과학 개념에 대한 학생 작문을 평가합니다. 학습자 언어 및 내용 정확성을 위해 인간이 평가한 합성 데이터셋을 생성하고, Llama 3.1과 Mistral NeMo와 같은 오픈 소스 모델을 미세 조정(fine-tuning)합니다. 최종적으로 MLLM의 성능은 데이터의 언어에 따라 달라진다는 가설을 수립합니다.

- **Performance Highlights**: 미세 조정을 통해 MLLM은 세 가지 언어(영어, 스페인어, Spanglish)에서 모두 성능이 현저히 향상되었습니다. 이 연구는 이중언어 학습자 사이의 진정한 언어 실습을 지원하기 위해 MLLM의 효과를 향상시킬 수 있는 잠재력을 강조합니다.



### A Capabilities Approach to Studying Bias and Harm in Language Technologies (https://arxiv.org/abs/2411.04298)
Comments:
          Accepted to the New Perspectives on Bias and Discrimination in Language Technology workshop

- **What's New**: 본 논문은 주류 자연어 처리(Natural Language Processing, NLP) 연구가 세계 다수 언어의 필요를 무시해왔음을 강조하며, 새로운 기술을 도입할 때 커뮤니티의 실제 요구와 편견 문제를 간과하지 말아야 한다고 경고합니다. 또한, 언어 기술에서의 공정성, 편견, 포괄성 문제를 'Capabilities Approach'의 관점에서 분석합니다.

- **Technical Details**: Capabilities Approach는 아마르티아 센(Amartya Sen)에 의해 1974년에 제안된 프레임워크로, 인간의 복지는 그들이 선택할 수 있는 'capabilities'(능력)와 'functionings'(실현된 능력)에 따라 이해되어야 한다고 주장합니다. 연구는 'Majority World'(대다수의 세계)와 그들의 커뮤니티에서 언어 기술이 미치는 영향을 깊이 있게 다룹니다.

- **Performance Highlights**: 이 프레임워크는 언어 기술의 해악을 측정하고 정의하는 과정에서 커뮤니티 구성원들과 의미 있는 협업을 촉진할 수 있으며, 이러한 접근 방식이 실제로 커뮤니티의 필요를 반영하고 그들의 능력을 실현하기 위한 자원 활용을 중심으로 할 수 있는 방법을 제시합니다.



### Unfair Alignment: Examining Safety Alignment Across Vision Encoder Layers in Vision-Language Models (https://arxiv.org/abs/2411.04291)
Comments:
          Preprint, Under Review

- **What's New**: 본 논문은 Vision-Language Models (VLMs)의 안전성 정렬(safety alignment) 문제를 다룬다. 특히, VLM의 비전 인코더의 층(layer) 간 안전성의 불균형 분포를 밝혀냈다.

- **Technical Details**: VLM의 중간 층의 활성화(activations)는 악의적인 입력에 더 취약하다는 것을 보여준다. 기초적인 안전 교육(default architectural settings)에서 벗어난 입력에 대한 모델의 일반화 부족이 이러한 'cross-layer' 취약성을 유발한다. 연구에서는 LLaVA-1.5와 Llama 3.2를 사용하여 여러 중간 층의 활성화가 모델의 안전 정렬(safety alignment)에 미치는 영향을 분석하였다.

- **Performance Highlights**: 실험 결과는 중간 층의 안전 정렬이 불균형하게 분포되어 있음을 나타내며, 후반 층이 초기와 중간 층보다 더 안전하게 정렬되어 있다는 것을 보여준다. 이는 기존의 안전 정렬 전략이 단일 기본 층에만 집중하는 것이 불충분하다는 점을 강조한다.



### Diversity Helps Jailbreak Large Language Models (https://arxiv.org/abs/2411.04223)
Comments:
          arXiv admin note: text overlap with arXiv:2312.02119

- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 기존 안전성 제약을 우회하여 유해한 출력을 생성할 수 있는 강력한 jailbreak 기법을 소개합니다. 이 기술은 이전 컨텍스트에서 벗어나도록 LLM에 지시함으로써 유용성이 크며, 기존 방법을 능가하는 성공률을 보여줍니다.

- **Technical Details**: 제안된 jailbreak 기법은 각 탐색 깊이에서 이전 시도와 다르게 진행하여 새로운 다양화된 프롬프트를 모호하게 만드는 방식입니다. 창의성, 허구화 및 이전 시도와의 차별화를 격려하여 다양화된 공격을 생성하고, 지역화된 검색을 통해 프레이즈를 모호화하여 정렬 메커니즘을 우회합니다. 이 방법은 화이트 박스 접근 없이 자동으로 작동하여 향후 언어 모델에도 유연하게 적용할 수 있습니다.

- **Performance Highlights**: 총 62.83%의 공격 성공률을 보이며, OpenAI의 GPT-4o-mini에서 57.17% 향상된 성능을 확인했습니다. 적은 양의 쿼리를 사용하여도 높은 성공률을 달성하였으며, 다양한 LLM에서도 이동성이 우수한 점을 강조합니다.



### Analyzing The Language of Visual Tokens (https://arxiv.org/abs/2411.05001)
- **What's New**: 본 논문은 트랜스포머 기반 모델, 특히 LLaVA와 Chameleon을 통해 시각적 정보의 이산 토큰화(discrete tokenized representation)가 재조명 되고 있다는 점에 주목합니다. 시각 언어와 인간 언어 간의 공동 정렬(joint alignments)을 학습하는 과정에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 우리는 이산 시각 언어의 통계적 특성을 자연어 중심(natural-language-centric) 접근법으로 분석했습니다. 결과적으로, 시각 언어는 Zipfian 분포를 따르지만, 더 높은 토큰 혁신(token innovation)은 더 큰 엔트로피(entropy)와 낮은 압축(compression)을 초래합니다. 또한, 시각 언어는 주로 객체의 부분(parts)을 나타내며 중간의 세분화(intermediate granularity)를 갖고 있습니다.

- **Performance Highlights**: 시각 언어는 응집력 있는 문법 구조(cohesive grammatical structures)를 결여하고 있어 자연어에 비해 높은 혼란도(perplexity)와 약한 계층적 조직(weak hierarchical organization)을 보입니다. 결과적으로 시각 모델은 자연어와 더 밀접하게 정렬되나, 여전히 자연어의 응집력에는 미치지 못합니다.



### LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation (https://arxiv.org/abs/2411.04997)
- **What's New**: 본 연구에서는 LLM2CLIP이라는 새로운 접근 방식을 제안하여 대형 언어 모델(LLMs)의 강력한 기능을 활용하여 CLIP의 시각적 표현 학습을 강화합니다. LLM2CLIP은 LLM을 캡션 공간에서 대조 학습(contrastive learning)을 통해 미세 조정하여 결과 임베딩의 텍스트 구별력을 크게 향상시킵니다.

- **Technical Details**: LLM2CLIP은 LLM의 출력 토큰을 더 잘 구별할 수 있도록 미세 조정하는 캡션 대조(CC) 미세 조정 전략을 활용합니다. LoRA( Low-Rank Adaptation)를 사용하여 CC3M 이미지 캡션 데이터셋에서 LLM을 경량 미세 조정했습니다. 이 과정에서 supervised SimCSE 대조 학습 손실을 적용하여 원래 캡션과 재주석된 캡션을 긍정 쌍으로 간주하고, 다른 모든 캡션은 부정 샘플 세트로 처리합니다.

- **Performance Highlights**: LLM2CLIP은 기존의 SOTA EVA02 모델에 대해 16.5% 성능 향상을 이루었으며, 영어 데이터로만 학습된 CLIP 모델을 최첨단의 교차 언어 모델로 변모시켰습니다. 또한 Llava 1.5와 같은 다중 모달 모델 훈련에 통합될 경우 EVA02에 비해 거의 모든 벤치마크에서 지속적인 개선을 나타냈습니다.



### Position Paper On Diagnostic Uncertainty Estimation from Large Language Models: Next-Word Probability Is Not Pre-test Probability (https://arxiv.org/abs/2411.04962)
Comments:
          Accepted to GenAI4Health Workshop at NeurIPS 2024

- **What's New**: 이 연구는 Mistral-7B와 Llama3-70B라는 두 개의 대형 언어 모델(LLMs)의 진단 결정 지원을 위한 사전 검사 확률(pre-test probability) 추정 능력을 평가합니다. 기존 연구의 한계를 발견하고 LLM의 신뢰도 추정 기술 개선의 필요성을 강조합니다.

- **Technical Details**: 연구는 660명의 환자 데이터에 대한 전자 건강 기록(EHRs)에서 세 가지 진단 작업(Sepsis, Arrhythmia, Congestive Heart Failure)에 대한 사전 검사 확률을 예측하는 데 두 가지 LLM을 평가했습니다. 세 가지 방법(토큰 로지트, 구술된 신뢰도, 피처 기반 교정기)을 사용하여 LLM의 성능을 분석하고, 기존의 eXtreme Gradient Boosting (XGB) 분류기와 결과를 비교했습니다.

- **Performance Highlights**: Mistral-7B와 Llama3-70B의 성능을 XGB와 비교하여, LLM이 특정 조건에 대한 사전 검사 확률을 예측하는 데 있어 여전히 최적의 성능을 발휘하지 못하고 있음을 밝혔습니다. 연구의 결과는 LLM의 응용과 신뢰도 향상에 기여할 것으로 예상됩니다.



### M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding (https://arxiv.org/abs/2411.04952)
Comments:
          Project webpage: this https URL

- **What's New**: 논문에서는 M3DocRAG라는 새로운 multi-modal RAG 프레임워크를 소개합니다. 이 프레임워크는 단일 문서 또는 여러 문서에서 정보 검색과 질문 응답을 처리할 수 있도록 설계되었습니다. 또한, 기존의 문서 시각적 질문 응답(DocVQA)에서는 이미지나 차트와 같은 시각적 정보를 무시하는 문제가 있었으나, M3DocRAG는 이를 해결합니다.

- **Technical Details**: M3DocRAG는 세 단계로 작동합니다: (1) 문서 임베딩 - RGB 이미지에서 시각적 임베딩을 추출하고, (2) 페이지 검색 - 텍스트 쿼리와 높은 유사성이 있는 상위 K 페이지를 검색하며, (3) 질문 응답 - MLM을 통해 검색된 페이지에서 최종 답변을 생성합니다. 이 프레임워크는 다양한 문서 맥락(폐쇄형 및 개방형), 질문 점프(단일 점프 및 다중 점프), 증거 모달리티(텍스트, 차트, 도표 등)를 지원합니다.

- **Performance Highlights**: M3DocRAG는 ColPali와 Qwen2-VL 7B를 사용하여 실험 세 가지 벤치마크(M3DocVQA, MMLongBench-Doc, MP-DocVQA)에서 많은 강력한 기준선보다 우수한 성능을 보였습니다. 특히 MP-DocVQA에서는 최신 성능을 기록했습니다.



### Enhancing Investment Analysis: Optimizing AI-Agent Collaboration in Financial Research (https://arxiv.org/abs/2411.04788)
- **What's New**: 최근 생성형 인공지능(GenAI)이 재무 분석 및 투자 의사결정에 주목받고 있으며, 본 논문은 다중 에이전트 협업 시스템을 통해 이 분야에서의 의사 결정 향상을 제안합니다. 이 시스템은 그룹 크기 및 협업 구조를 조정할 수 있어 다양한 시장 조건에 적합하도록 설계되었습니다.

- **Technical Details**: 먼저, 본 시스템은 서로 다른 크기와 구조의 에이전트 그룹을 조합하여 재무 투자 연구의 세 가지 주요 하위 작업인 기본 분석(fundamentals), 시장 감정(market sentiment), 위험 분석(risk analysis)을 수행합니다. 또한, 시스템은 비 최적 조합 전략을 사용해 동적으로 시장 조건에 적응하고, 성능 최적화를 이룹니다.

- **Performance Highlights**: 다중 에이전트 협업 시스템은 전통적인 단일 에이전트 모델을 능가하며, 복잡한 재무 환경에서 정확도, 효율성, 적응성을 개선하는 것으로 나타났습니다. 특히, 목표 가격 예측에서 평균 2.35%의 차이와 66.7%의 정확도를 기록했습니다.



### DISCO: DISCovering Overfittings as Causal Rules for Text Classification Models (https://arxiv.org/abs/2411.04649)
- **What's New**: 본 논문은 DISCO라는 새로운 방법론을 제안하며, 이 방법론은 모델의 예측과 관련된 인과적 n-그램 패턴을 식별하여 글로벌한 규칙 기반 설명을 도출하는 데 중점을 두고 있습니다.

- **Technical Details**: DISCO는 훈련 데이터에서 n-그램 패턴을 추출하고, 이를 모델 예측과 연관지어 인과성을 검증하는 스케일러블한 시퀀스 마이닝 기법을 사용합니다. 이 과정에서는 카운터팩추얼(counterfactual) 구조를 통해 패턴의 인과성을 평가하고 잠재적 오버피팅을 드러내는 강력한 규칙을 추출합니다.

- **Performance Highlights**: DISCO는 MultiRC 데이터셋에서 100%의 단축키 탐지율을 달성하며 모델 성능이 18.8% 감소하는 결과를 보였습니다. 또한, DISCO는 상호작용 기반 설명을 지원하여 인과성을 명확히 하고, 개별 인스턴스 설명의 부담을 덜어줍니다.



### Self-Calibrated Listwise Reranking with Large Language Models (https://arxiv.org/abs/2411.04602)
- **What's New**: 이번 논문에서는 기존의 LLM 기반 텍스트 재순위 편향을 개선하기 위해 Self-Calibrated Listwise Reranking (SCaLR) 방법을 제안합니다. 이 방법은 전역 관련성 점수를 도출하는 효율적인 기법으로, LLM의 제한된 문맥 창(window) 문제를 해결하고자 합니다.

- **Technical Details**: SCaLR 방법은 두 가지 주요 요소로 구성됩니다. 첫 번째는 관련성 인식 리스트 뷰 재순위를 통해 전체 후보 집합에서 직접 관련성 점수를 계산하는 것이고, 두 번째는 Self-Calibrated Training으로, LLM이 생성한 점 뷰 관련성 점수를 사용하여 리스트 뷰 점수를 조정합니다. 이로써 리스트 뷰 점수는 더 신뢰할 수 있도록 보정됩니다.

- **Performance Highlights**: BEIR 및 TREC 벤치마크에서 시행한 실험 결과, SCaLR 방법은 기존의 최첨단 기법들과 비교해 효율성과 효과 면에서 우수함을 입증하였습니다.



### Best Practices for Distilling Large Language Models into BERT for Web Search Ranking (https://arxiv.org/abs/2411.04539)
Comments:
          Arxiv Version

- **What's New**: 최근 연구들은 Large Language Models(LLMs)가 zero-shot relevance rankers로서 상당한 잠재력을 지니고 있음을 강조하였습니다. 본 연구는 LLM의 랭킹 전문성을 BERT와 유사한 더 간결한 모델로 전이하는 기술을 탐구했습니다.

- **Technical Details**: 연구에서는 Continued Pre-Training(CPT)을 통해 쿼리를 입력으로 받고 클릭된 제목과 요약을 출력으로 하는 방식으로 LLM을 향상시켰습니다. 이를 통해 pairwise rank loss를 이용하여 LLM을 감독 학습 방식으로 미세 조정하였고, hybrid point-wise 및 margin MSE loss를 도입하여 LLM의 랭킹 지식을 BERT와 같은 더 작은 모델로 전이했습니다.

- **Performance Highlights**: 이 방법은 리소스 제약이 있는 환경에서도 실현 가능한 솔루션을 제공합니다. 오프라인 및 온라인 평가 모두에서 접근법의 효능이 확인되었으며, 본 모델은 2024년 2월 상업적인 웹 검색 엔진에 성공적으로 통합되었습니다.



### Variational Low-Rank Adaptation Using IVON (https://arxiv.org/abs/2411.04421)
Comments:
          Published at 38th Workshop on Fine-Tuning in Machine Learning (NeurIPS 2024). Code available at this https URL

- **What's New**: 이번 연구에서는 Variational Learning이 Low-Rank Adaptation (LoRA)의 정확도와 보정(calibration) 성능을 크게 향상시킬 수 있음을 보여줍니다. Improved Variational Online Newton (IVON) 알고리즘을 사용하여 대형 언어 모델을 세부조정(finetuning) 했습니다.

- **Technical Details**: IVON은 AdamW 옵티마이저를 대체하여 Variational-Bayesian 목표를 최적화합니다. 이 알고리즘은 학습률 적응에 사용되는 스케일 벡터를 통해 posterior variance를 무료로 추정할 수 있어 구현이 용이하며, 전체 훈련 시간의 약 1%의 오버헤드만 발생합니다.

- **Performance Highlights**: Llama-2 모델의 세부 조정 결과, IVON은 AdamW에 비해 정확도가 2.8% 증가하고 Expected Calibration Error (ECE)는 4.6% 감소했습니다. 이와 함께 다른 베이지안 대안들보다도 우수한 성능을 보여주었습니다.



### Robust and Efficient Fine-tuning of LLMs with Bayesian Reparameterization of Low-Rank Adaptation (https://arxiv.org/abs/2411.04358)
Comments:
          48 pages, 10 figures, 10 tables, Code: this https URL

- **What's New**: 이 논문에서는 MonteCLoRA라는 새로운 기법을 제안하여 대형 언어 모델(LLMs)에서 Low-Rank Adaptation의 성능을 개선하는 방법을 다룹니다. MonteCLoRA는 Monte Carlo 추정을 활용하여 낮은 랭크 매개변수의 불편한 후행 분포를 학습하여 안정성과 정확도를 높입니다.

- **Technical Details**: MonteCLoRA는 낮은 랭크 적응을 위해 다변량 Gaussian 분포의 혼합물로 매개변수를 모델링합니다. 이 기법은 사전 분포를 Wishart 분포에 따라 설정하여 낮은 예측 분산과 안정성을 제공하며, O(1) 추가 매개변수만으로 동작합니다.

- **Performance Highlights**: MonteCLoRA는 자연어 이해(NLU) 작업에서 기존의 효율적 미세 조정 방식보다 최대 3.8% 높은 정확도와 8.6% 더 큰 Robustness를 달성했습니다. 또한, LLaMA-1-7B를 사용한 생성 작업에서는 50% 낮은 분산으로 제로샷 성능을 입증했습니다.



### Scaling Laws for Precision (https://arxiv.org/abs/2411.04330)
- **What's New**: 본 논문은 언어 모델(Language Model)의 품질과 비용에 영향을 미치는 저정밀 훈련(low precision training) 및 추론(inference)의 문제를 다루고 있다. 새로운 'precision-aware' 스케일링 법칙(scaling laws)을 제안하여, 낮은 정밀도로 학습하며 발생하는 추가 손실(loss)을 예측하고, 모델의 '효과적인 파라미터 수(effective parameter count)'를 줄이는 방법을 제시하고 있다.

- **Technical Details**: 저자들은 저정밀 학습 및 추론 동안 발생하는 손실을 예측하기 위해 스케일링 법칙을 도입하였다. 이를 통해, 모델의 다양한 부분을 다른 정밀도로 훈련할 때의 손실을 예측할 수 있으며, 큰 모델을 저정밀로 훈련하는 것이 컴퓨팅 최적(compute optimal)하다는 것을 제안하고 있다. Transformer++ 아키텍처를 사용하고, 여러 가지 하이퍼파라미터와 시퀀스 길이(sequence length) 및 배치 크기(batch size)를 고정하여 실험하였다.

- **Performance Highlights**: 465개 이상의 사전 훈련(pretraining) 실행 결과를 기반으로 스케일링을 분석하였으며, 모델 크기가 최대 1.7B 파라미터이고 26B 토큰(token)으로 훈련된 결과를 검증하였다. 본 연구는 훈련 및 추론에서 다양한 정밀도에 따른 품질 저하를 예측하는 단일 기능적 형식을 도출하여, 향후 연구에 기여할 것으로 예상된다.



### Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding (https://arxiv.org/abs/2411.04282)
- **What's New**: 이 논문에서는 LaTent Reasoning Optimization (LaTRO)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 추론 능력을 향상시키기 위해 latent distribution에서 샘플링하는 방식으로 reasoning을 수식화하고 변분적 접근 방식을 통해 최적화합니다. LaTRO는 외부 피드백이나 보상 모델 없이 LLM의 추론 과정과 추론 품질 평가 능력을 동시에 향상시킬 수 있게 해줍니다.

- **Technical Details**: LaTRO는 다음과 같은 특징을 가지고 있습니다: 1) LLM 추론 최적화를 latent variable models에 연결하는 이론적 수식화, 2) 모델의 고유 확률 추정을 활용하는 self-rewarding 메커니즘, 3) 다양한 모델 아키텍처 및 추론 과제에서 실질적인 성능 향상을 보여줍니다. 이를 통해 pre-trained LLMs가 내재된 추론 가능성을 보유하고 있음을 입증합니다.

- **Performance Highlights**: GSM8K 데이터셋에서 LaTRO는 기본 모델과 비교해 평균 12.5%의 zero-shot 정확도를 개선했으며, 감독식 미세 조정(supervised fine-tuning) 대비 9.6% 향상을 이끌어냈습니다. 이러한 결과는 LaTRO가 LLM의 latent reasoning capabilities를 효과적으로 활성화하고 개선할 수 있음을 나타냅니다.



### Analyzing Multimodal Features of Spontaneous Voice Assistant Commands for Mild Cognitive Impairment Detection (https://arxiv.org/abs/2411.04158)
- **What's New**: 이번 연구는 Mild Cognitive Impairment (MCI)를 음성 비서(Voice Assistant, VA) 명령어를 통해 탐지하는 새로운 접근 방식을 제안합니다. 특히, 참가자들이 정의된 의도에 따라 자유롭게 명령어를 생성하는 명령어 생성 작업을 설계하였으며, 표준 읽기 명령어 작업보다 우수한 분류 성능을 나타냈습니다.

- **Technical Details**: 연구에서는 35명의 노인을 대상으로 음성 명령을 통해 MCI와 건강한 노인을 구분하기 위해 오디오, 텍스트, 의도 및 멀티모달 피쳐를 활용한 MCI 분류 및 회귀 모델을 개발하였습니다. 명령어 생성 작업은 참가자들에게 의도 키워드를 제공함으로써 참가자들이 발화 명령을 생성하도록 하였으며, 이 과정에서 인지 부하가 증가했습니다. 결과적으로, 명령어 생성 작업에서 평균 82%의 분류 정확도를 달성하였습니다.

- **Performance Highlights**: 명령어 생성 작업은 명령어 읽기 작업에 비해 주의력과 단기 기억 유지 같은 인지 영역에서 더 높은 연관성을 보였으며, 멀티모달 피쳐의 융합을 통해 MCI 탐지에서 높은 성능을 보였습니다. 또한, 의도 및 오디오 피쳐는 텍스트 피쳐보다 MCI 탐지에서 더 민감하다는 결과를 얻었습니다.



### Crystal: Illuminating LLM Abilities on Language and Cod (https://arxiv.org/abs/2411.04156)
Comments:
          Published as a conference paper at COLM 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)과 코드 생성(신 코드 LLM) 간의 상호작용을 개선하기 위한 새로운 사전 훈련(pretraining) 전략을 제안했습니다. 이 전략은 자연어와 코딩 능력을 통합하여 코드 LLM을 더욱 발전시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구의 사전 훈련 과정은 두 단계로 나누어 진행됩니다. 첫 번째 단계에서는 95%의 자연어와 5%의 코드 데이터 혼합으로 시작하며, 두 번째 단계에서는 63%의 코드와 37%의 자연어로 데이터 비율을 조정하여 훈련합니다. 이러한 다단계 사전 훈련 전략은 인간의 학습 과정을 모방하여 기본적인 언어 지식을 먼저 습득한 후 코딩 기술을 개발하는 방식입니다.

- **Performance Highlights**: 결과적으로 개발된 Crystal 모델은 자연어와 코딩 성능에서 각각 Llama 2와 Code Llama와 비교되는 뛰어난 능력을 보입니다. Crystal은 1.4조 개의 토큰으로 사전 훈련되었으며, Llama 2와 Code Llama가 각각 2조 개가 넘는 토큰을 사용한 것과 비교하여 데이터 효율성이 우수한 점이 특징입니다.



### Software Design Pattern Model and Data Structure Algorithm Abilities on Microservices Architecture Design in High-tech Enterprises (https://arxiv.org/abs/2411.04143)
- **What's New**: 이 연구는 기업 내 마이크로서비스 아키텍처 디자인에 대한 소프트웨어 디자인 모델의 능력과 데이터 구조 알고리즘의 능력이 미치는 영향을 조사하였습니다. 직접적인 경험이 있는 소프트웨어 아키텍트 및 개발자들과의 심층 인터뷰를 통해 진행되었습니다.

- **Technical Details**: 연구에서 밝혀진 바에 따르면, 견고한 디자인 모델과 효율적인 알고리즘을 강조하는 조직이 마이크로서비스 아키텍처에서 더 나은 확장성(scalability), 성능(performance), 유연성(flexibility)을 달성합니다. 강력한 기반이 서비스 분해(service decomposition)를 개선하고, 데이터 처리(data processing)를 최적화하며, 시스템 반응성(system responsiveness)을 향상시키는 데 도움을 줍니다.

- **Performance Highlights**: 결과적으로, 마이크로서비스 아키텍처에서 효과적인 구현을 위해 이러한 역량이 중요하다는 점을 강조하며, 향후 연구 방향으로 새로운 기술의 통합과 변화하는 소프트웨어 디자인 관행에 대한 갭(gap)을 해결할 수 있는 길을 제시합니다.



### Unified Pathological Speech Analysis with Prompt Tuning (https://arxiv.org/abs/2411.04142)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구에서는 우울증(Depression), 알츠하이머(Alzheimer's disease), 파킨슨병(Parkinson's disease)과 같은 세 가지 질병을 위한 통합된 병리적 언어 분석 시스템을 제안합니다. 기존 모델들이 특정 질병에 국한되어 있었다면, 본 시스템은 여러 질병 간의 관계를 활용합니다.

- **Technical Details**: 제안된 시스템은 프롬프트 튜닝(prompt tuning) 기법을 활용하여, 전체 매개변수(parameter) 중 일부만 조정하여 질병을 분석하며, 사전 훈련된 구술 언어 모델(pre-trained spoken language model)을 사용합니다. 이를 통해 효율적인 훈련(proficient training)이 가능합니다.

- **Performance Highlights**: 실험 결과, 알츠하이머, 우울증 및 파킨슨병에 대해 경쟁력 있는 결과를 보여주었으며, F1 점수(F1 scores)가 향상되었습니다. 이는 신속한 수렴(convergence)과 다양한 태스크 간 지식 공유(knowledge sharing)를 통해 달성되었습니다.



### A Comparative Study on the Impact of Test-Driven Development (TDD) and Behavior-Driven Development (BDD) on Enterprise Software Delivery Effectiveness (https://arxiv.org/abs/2411.04141)
- **What's New**: 본 논문은 소프트웨어 배포 효과성에 미치는 Test-Driven Development (TDD)와 Behavior-Driven Development (BDD)의 비교 연구를 다루고 있습니다. 이 연구는 TDD와 BDD를 채택한 기업의 개발자 및 프로젝트 관리자와의 심층 인터뷰를 통해 데이터를 수집했습니다.

- **Technical Details**: 연구는 질적 연구 디자인을 사용하여 진행되었으며, TDD는 조기 테스트(early testing)와 반복 개발(iterative development)을 강조하여 코드 품질을 높이고 결함(defect)을 줄이는 반면, BDD는 이해 관계자(stakeholder)와의 직접적인 참여를 포함하는 행동 명세(behavior specifications)에 중점을 둠으로써 교차 기능적 의사소통(cross-functional communication)을 향상시킵니다.

- **Performance Highlights**: TDD는 초기 시간 투자(initial time investment)가 더 높을 수 있으며, BDD는 요구사항 명확성(requirement clarity)에서의 도전에 직면할 수 있습니다. 이는 각 모델이 다양한 프로젝트 유형과 이해 관계자의 요구에 어떻게 조화를 이루는지를 이해하는 데 중요한 격차를 강조하며, 기업들이 그들의 고유한 요구에 가장 적합한 모델을 선택하는 데 도움을 줄 수 있습니다. 또한 연구는 TDD와 BDD의 실제 적용 및 그 과제를 제공하여 향후 다양한 환경에서의 장기적 영향에 대한 연구를 제안합니다.



### GRSQA -- Graph Reasoning-Structured Question Answering Datas (https://arxiv.org/abs/2411.00369)
Comments:
          15 pages, 24 figures, 10 tables

- **What's New**: 본 논문에서는 GRS-QA라는 새로운 QA 데이터셋을 소개합니다. 이 데이터셋은 질문-답변 쌍에 대한 명확한 추론 구조를 제공하여 LLM의 성능을 보다 세밀하게 평가할 수 있게 합니다.

- **Technical Details**: GRS-QA는 의미적 맥락과 추론 구조를 포함하며, 각 질문은 추론 그래프라고 불리는 시각적 구조로 표현됩니다. 이 구조에서 노드는 텍스트 맥락을 나타내고 엣지는 논리의 흐름을 나타냅니다. 데이터셋은 HotpotQA, MuSiQue, 2WikiMultiHopQA에서 QA 쌍을 사용하여 생성되었습니다. 각 QA 쌍에는 해당하는 지원 사실이 포함됩니다.

- **Performance Highlights**: 연구 결과, LLM은 다양한 추론 구조를 처리할 때 성능 차이를 보이는 것으로 나타났습니다. GRS-QA는 LLM이 질문의 추론 구조에 따라 어떻게 성과가 달라지는지를 심층적으로 분석할 수 있는 기회를 제공합니다.



New uploads on arXiv(cs.IR)

### Lightning IR: Straightforward Fine-tuning and Inference of Transformer-based Language Models for Information Retrieva (https://arxiv.org/abs/2411.04677)
Comments:
          Accepted as a demo at WSDM'25

- **What's New**: 본 논문은 정보 검색(Task)용 Transformer 기반 언어 모델을 위한 PyTorch Lightning 기반 프레임워크인 Lightning IR을 소개합니다. 이 프레임워크는 정보 검색 파이프라인의 모든 단계(예: fine-tuning, indexing, searching, re-ranking)를 지원하며, 사용이 간편하고 확장 가능하며 재현 가능합니다.

- **Technical Details**: Lightning IR은 backbone agnostic 하며, HuggingFace 모델을 사용하여 fine-tuning 및 inference를 쉽게 수행할 수 있습니다. 주요 구성 요소로는 Model, Dataset, Trainer 및 Command-Line Interface (CLI)로 구성되어 있습니다. 모델 유형으로는 cross-encoders와 bi-encoders를 지원하며, 다양한 사용자 정의 데이터셋을 지원합니다. 또한, PyTorch Lightning의 trainer 클래스를 기반으로 하여 유연하고 재현 가능한 훈련을 제공합니다.

- **Performance Highlights**: Lightning IR은 모델 간 비교와 새로운 모델 유형을 실험하기 쉽게 설계되었습니다. 여러 인기 있는 모델을 지원하며, 인덱싱, 검색 및 재정렬을 쉽게 수행할 수 있는 CLI를 제공합니다.



### Self-Calibrated Listwise Reranking with Large Language Models (https://arxiv.org/abs/2411.04602)
- **What's New**: 이번 논문에서는 기존의 LLM 기반 텍스트 재순위 편향을 개선하기 위해 Self-Calibrated Listwise Reranking (SCaLR) 방법을 제안합니다. 이 방법은 전역 관련성 점수를 도출하는 효율적인 기법으로, LLM의 제한된 문맥 창(window) 문제를 해결하고자 합니다.

- **Technical Details**: SCaLR 방법은 두 가지 주요 요소로 구성됩니다. 첫 번째는 관련성 인식 리스트 뷰 재순위를 통해 전체 후보 집합에서 직접 관련성 점수를 계산하는 것이고, 두 번째는 Self-Calibrated Training으로, LLM이 생성한 점 뷰 관련성 점수를 사용하여 리스트 뷰 점수를 조정합니다. 이로써 리스트 뷰 점수는 더 신뢰할 수 있도록 보정됩니다.

- **Performance Highlights**: BEIR 및 TREC 벤치마크에서 시행한 실험 결과, SCaLR 방법은 기존의 최첨단 기법들과 비교해 효율성과 효과 면에서 우수함을 입증하였습니다.



### Best Practices for Distilling Large Language Models into BERT for Web Search Ranking (https://arxiv.org/abs/2411.04539)
Comments:
          Arxiv Version

- **What's New**: 최근 연구들은 Large Language Models(LLMs)가 zero-shot relevance rankers로서 상당한 잠재력을 지니고 있음을 강조하였습니다. 본 연구는 LLM의 랭킹 전문성을 BERT와 유사한 더 간결한 모델로 전이하는 기술을 탐구했습니다.

- **Technical Details**: 연구에서는 Continued Pre-Training(CPT)을 통해 쿼리를 입력으로 받고 클릭된 제목과 요약을 출력으로 하는 방식으로 LLM을 향상시켰습니다. 이를 통해 pairwise rank loss를 이용하여 LLM을 감독 학습 방식으로 미세 조정하였고, hybrid point-wise 및 margin MSE loss를 도입하여 LLM의 랭킹 지식을 BERT와 같은 더 작은 모델로 전이했습니다.

- **Performance Highlights**: 이 방법은 리소스 제약이 있는 환경에서도 실현 가능한 솔루션을 제공합니다. 오프라인 및 온라인 평가 모두에서 접근법의 효능이 확인되었으며, 본 모델은 2024년 2월 상업적인 웹 검색 엔진에 성공적으로 통합되었습니다.



### Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers (https://arxiv.org/abs/2411.04403)
- **What's New**: 이번 논문에서는 inference-free sparse retriever의 검색 관련성을 향상시키기 위한 새로운 훈련 방법을 제안합니다. 특히 IDF-aware FLOPS 손실 함수와 이종 앙상블 지식 증류 프레임워크라는 두 가지 접근법을 통해 성능 개선을 목표로 하였습니다.

- **Technical Details**: IDF-aware FLOPS 손실은 토큰의 IDF 값에 따라 패널티를 조정하여, 고유한 토큰에 대해서는 적은 패널티를 부과하고 빈도가 높은 토큰에는 더 많은 패널티를 부과해 검색 관련성을 개선합니다. 이종 앙상블 프레임워크는 dense 및 sparse retrievers의 강점을 결합하여 감독 신호를 생성하는 방법입니다.

- **Performance Highlights**: BEIR 벤치마크에서 제안된 모델은 기존의 SOTA inference-free sparse 모델보다 3.3 NDCG@10 점수를 높였으며, siamese sparse retrievers와 유사한 검색 관련성을 보였습니다. 클라이언트 측 지연 시간은 BM25보다 1.1배에 불과합니다.



### AmazonQAC: A Large-Scale, Naturalistic Query Autocomplete Datas (https://arxiv.org/abs/2411.04129)
Comments:
          EMNLP 2024

- **What's New**: 새로운 QAC 데이터셋 AmazonQAC이 소개되었습니다. 이 데이터셋은 Amazon 검색 로그에서 수집된 3억 9천 5백만 개의 샘플로 구성되어 있으며, 사용자 입력 접두사에서 최종 검색어에 이르는 실제 시퀀스를 포함합니다.

- **Technical Details**: AmazonQAC는 사용자 세션 ID 및 타임스탬프(_timestamp_)를 포함하여 QAC 문제의 맥락 의존적인 측면을 모델링하는 데 유용합니다. Prefix Trees, semantic retrieval 및 Large Language Models (LLMs)와 그리고 파인튜닝(finetuning) 여부에 따른 성능을 평가하였습니다.

- **Performance Highlights**: 파인튜닝된 LLM이 가장 좋은 성능을 보였으나, 최고의 시스템조차 테스트 데이터에서 이론적으로 가능한 성능의 절반에 불과하여 QAC 문제의 난이도를 시사합니다. Success@10 점수가 37로, 이는 상한선의 절반에 해당합니다.



### Orbit: A Framework for Designing and Evaluating Multi-objective Rankers (https://arxiv.org/abs/2411.04798)
- **What's New**: 이번 논문에서는 다목적 랭킹(rank) 모델의 설계 및 평가를 위한 새로운 개념적 프레임워크인 Orbit를 소개합니다. 이 프레임워크는 설계 과정에서 목표(objectives)를 중심에 두어 커뮤니케이션의 경계 객체(boundary objects) 역할을 하도록 합니다.

- **Technical Details**: Orbit는 목표 공간(objective spaces)과 직접 상호작용할 수 있는 인터랙티브 시스템으로 구현되어, 다양한 설계 거래(trade-off)를 실시간으로 탐색하고 평가할 수 있습니다. 이를 통해 다중 목표를 고려하는 머신러닝 모델 설계에 필수적인 정보 분석과 팀 간 협업을 지원합니다.

- **Performance Highlights**: 12인의 산업 전문가를 대상으로 한 사용자 연구(user study)를 통해 Orbit의 효율적인 설계 공간 탐색을 지원하고, 더 나은 의사결정을 이끌어내며, 다중 목표 간의 고유한 거래의 인식을 향상시킨다는 사실을 보여주었습니다.



### The Concatenator: A Bayesian Approach To Real Time Concatenative Musaicing (https://arxiv.org/abs/2411.04366)
Comments:
          12 pages, 6 figures, Accepted for Publication in The International Society for Music Information Retrieval Proceedings, 2024

- **What's New**: 새로운 시스템 'The Concatenator'는 오디오 가이드가 포함된 연결 합성(concatenative synthesis)을 실시간으로 구현합니다. 이 시스템은 Driedger et al.의 'musaicing' 기술과 유사하게, 오디오 데이터의 특정 윈도우를 연결하여 목표 오디오 스트림의 하모닉(harmonic) 및 퍼커시브(percussive) 요소를 재현합니다. 하지만 기존 NMF(Non-negative Matrix Factorization) 기반 기법과는 달리, 본 시스템은 명시적인 베이esian 관점을 채택하여 더 나은 결과를 제공합니다.

- **Technical Details**: The Concatenator는 파티클 필터(particle filter)를 사용하여 실시간으로 숨겨진 상태를 추론하고, 전이 모델(transition model) 및 관찰 모델(observation model)을 통해 사용자가 윈도우의 변경 속도를 조절할 수 있게 합니다. 이 시스템의 계산 복잡도는 코퍼스(corpus) 크기에 독립적이므로, 수시간 분량의 데이터에도 효과적으로 확장할 수 있습니다. 사용자들은 그레인(grain) 길이 변경, 목표 적합(fit to target), 실시간 피치 이동(pitch shift)을 통해 상호작용할 수 있습니다.

- **Performance Highlights**: 실제 성과 평가를 통해 시스템의 다양한 매개변수 효과를 정량적 테스트로 검증하였으며, 예술적 통찰력에 대한 정성적 평가를 통해 실시간 기능이 새로운 음악적 표현 및 제어 방식의 가능성을 열어주고 있음이 입증되었습니다. 이는 라이브 공연 및 모듈형 합성과의 통합을 위한 중요한 진전을 나타냅니다.



### dsld: A Socially Relevant Tool for Teaching Statistics (https://arxiv.org/abs/2411.04228)
Comments:
          To be submitted to the Journal of Statistics and Data Science Education

- **What's New**: 본 논문에서는 데이터 과학이 사회적 차별 문제를 다루는데 어떻게 기여할 수 있는지에 대한 새로운 접근법을 소개합니다. 특히, 'Data Science Looks At Discrimination' (dsld)라는 R 및 Python 패키지는 인종, 성별 및 나이와 같은 보호 그룹과 관련된 차별을 평가하기 위한 통계 및 그래픽 방법을 제공하는 포괄적인 도구를 사용자에게 제공합니다.

- **Technical Details**: dsld 패키지는 2023년에 개발되었으며, 24개의 기능과 관련된 오픈 소스 교과서로 구성되어 있습니다. 이 패키지는 다양한 통계적 기법을 포함하여 차별 분석을 위한 모델을 제공하며, 예를 들어, 예측 알고리즘에서의 편향 감소를 위한 방법을 포함합니다. dsldLinear 함수는 민감 변수를 포함하여 회귀 모델을 구축할 수 있도록 지원합니다.

- **Performance Highlights**: dsld 패키지는 차별 분석을 위한 강력한 도구로, 사회 과학 연구와 인사 분석에서 유용하게 사용될 수 있습니다. 예를 들어, 이 패키지는 Random Forests 같은 비모수 회귀 방법을 사용하여 매우 큰 데이터 세트에서도 효과적으로 분석을 수행할 수 있습니다.



New uploads on arXiv(cs.CV)

### SVDQunat: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models (https://arxiv.org/abs/2411.05007)
Comments:
          Quantization Library: this https URL Inference Engine: this https URL Website: this https URL Demo: this https URL Blog: this https URL

- **What's New**: 이 논문에서는 Diffusion 모델의 속도를 높이기 위해 가중치와 활성화를 4비트로 양자화하는 SVDQuant 방법을 제안합니다. 기존 방식들이 효과가 없던 부분에서, 새로운 저비용 분기를 도입하여 이상값(outlier)을 모으는 혁신적인 접근법을 사용합니다.

- **Technical Details**: SVDQuant는 가중치와 활성화를 4비트로 양자화하기 위해 이상값을 통합하고, 특정 조건에서 Singular Value Decomposition (SVD)를 활용해 저차원 분기를 사용하여 잔여값을 4비트로 양자화합니다. 또한, Nunchaku라는 새로운 추론 엔진을 설계하여 불필요한 메모리 접근을 줄이고, LoRA(저차원 적응 장치)와의 통합을 지원합니다.

- **Performance Highlights**: FLUX.1 모델의 메모리 사용량을 3.5배 감소시키고, 16GB laptop 수준의 RTX 4090 GPU에서 4비트 가중치 양자화만 한 경우 대비 3.0배의 속도 향상을 달성했습니다. 이러한 결과는 인터랙티브한 어플리케이션에서의 활용 가능성을 높입니다.



### ProEdit: Simple Progression is All You Need for High-Quality 3D Scene Editing (https://arxiv.org/abs/2411.05006)
Comments:
          NeurIPS 2024. Project Page: this https URL

- **What's New**: 이번 논문에서는 ProEdit라는 3D 장면 편집을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 diffusion distillation을 활용하여 고품질 3D 장면 편집을 간소화하고, 다중 뷰 불일치 문제를 해결하기 위해 편집 작업을 여러 하위 작업으로 나누어 점진적으로 수행합니다.

- **Technical Details**: ProEdit의 핵심은 편집 작업의 FOS(Feasible Output Space) 크기를 제어하며, 이를 위해 하위 작업 스케줄러와 적응형 3D Gaussian splatting(3DGS) 훈련 전략을 도입한 것입니다. 하위 작업의 난이도와 FOS 크기를 기반으로 하여 효율적이고 고품질의 편집이 가능하도록 구성됩니다.

- **Performance Highlights**: ProEdit는 다양한 장면과 도전적인 편집 작업에서 높은 품질의 결과를 달성하며, 복잡하거나 비싼 추가 요소 없이도 state-of-the-art 성능을 입증하고 있습니다. 사용자는 편집 과정 중 생성된 중간 결과들을 미리 볼 수 있어 편집 작업의 '공격성'(aggressivity)을 조절할 수 있는 혁신적인 방법을 제공합니다.



### Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models (https://arxiv.org/abs/2411.05005)
Comments:
          26 pages, 14 figures

- **What's New**: Diff-2-in-1 프레임워크를 도입하여 다중 모드 데이터 생성과 조밀 시각 인식을 동시에 처리하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: Diff-2-in-1은 diffusion-denoising 프로세스를 활용하여 생성된 데이터를 조밀 시각 인식에서 효과적으로 활용할 수 있도록 최적화된 학습 메커니즘을 포함합니다. 주요 설계 요소는 생성 및 활용 매개변수를 갖는 자가 개선 학습 메커니즘입니다.

- **Performance Highlights**: Diff-2-in-1은 여러 검증된 벤치마크에서 경쟁 지표를 초과하는 성능 개선을 보이며, 높은 품질과 유용성을 가진 다중 모드 데이터 생성을 입증합니다.



### ReCapture: Generative Video Camera Controls for User-Provided Videos using Masked Video Fine-Tuning (https://arxiv.org/abs/2411.05003)
Comments:
          project page: this https URL

- **What's New**: ReCapture라는 새로운 방법을 소개하여 사용자가 제공한 비디오에서 새로운 카메라 경로를 생성할 수 있습니다. 이는 기존 비디오의 장면 모션을 보존하면서 다양한 앵글에서 재현할 수 있게 해 주며, 원본 비디오에는 없는 장면 내용을 합리적으로 환각할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 이 방법은 (1) 다중 뷰 확산 모델(multiview diffusion models) 또는 깊이 기반 포인트 클라우드 렌더링(depth-based point cloud rendering)을 사용하여 새로운 카메라 경로로 노이즈가 있는 앵커 비디오를 생성하고, (2) 제안된 마스크 비디오 파인튜닝(masked video fine-tuning) 기법을 통해 앵커 비디오를 깨끗하고 시간적으로 일관된 비디오로 복원합니다.

- **Performance Highlights**: ReCapture는 Generative Camera Dolly와 같은 기존 generative 방법보다 우수한 성능을 보여주며, 짝이 맞는 비디오 데이터 없이도 비디오를 생성할 수 있는 기능이 있습니다. 또한 각 구성 요소는 VBench에서의 ablation study를 통해 검증되었습니다.



### Analyzing The Language of Visual Tokens (https://arxiv.org/abs/2411.05001)
- **What's New**: 본 논문은 트랜스포머 기반 모델, 특히 LLaVA와 Chameleon을 통해 시각적 정보의 이산 토큰화(discrete tokenized representation)가 재조명 되고 있다는 점에 주목합니다. 시각 언어와 인간 언어 간의 공동 정렬(joint alignments)을 학습하는 과정에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 우리는 이산 시각 언어의 통계적 특성을 자연어 중심(natural-language-centric) 접근법으로 분석했습니다. 결과적으로, 시각 언어는 Zipfian 분포를 따르지만, 더 높은 토큰 혁신(token innovation)은 더 큰 엔트로피(entropy)와 낮은 압축(compression)을 초래합니다. 또한, 시각 언어는 주로 객체의 부분(parts)을 나타내며 중간의 세분화(intermediate granularity)를 갖고 있습니다.

- **Performance Highlights**: 시각 언어는 응집력 있는 문법 구조(cohesive grammatical structures)를 결여하고 있어 자연어에 비해 높은 혼란도(perplexity)와 약한 계층적 조직(weak hierarchical organization)을 보입니다. 결과적으로 시각 모델은 자연어와 더 밀접하게 정렬되나, 여전히 자연어의 응집력에는 미치지 못합니다.



### HourVideo: 1-Hour Video-Language Understanding (https://arxiv.org/abs/2411.04998)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track; 28 pages

- **What's New**: 이번 연구에서는 장시간 비디오-언어 이해를 위한 벤치마크 데이터셋 HourVideo를 제안합니다. 이 데이터셋은 요약, 인식, 시각적 추론, 내비게이션 등의 다양한 작업으로 구성되어 있습니다.

- **Technical Details**: HourVideo는 Ego4D 데이터셋에서 500개의 수동으로 선별된 에고센트릭(egocentric) 비디오로 구성되어 있으며, 총 12,976개의 고품질 다지선다형 질문을 포함하고 있습니다. 이 데이터셋은 20분에서 120분까지의 비디오를 담고 있습니다. 각 작업은 장기적 이해를 필요로 하는 질문으로 설계되어 있습니다.

- **Performance Highlights**: 최신 멀티모달 모델들(GPT-4 및 LLaVA-NeXT 포함)이 임의의 예측기(random predictor)보다 약간 나은 성능(25.7%, 22.3%)을 보여주었지만, 인간 전문가의 평균 성능(85%)에 비해 여전히 큰 차이가 있음을 보여줍니다. Gemini Pro 1.5는 37.3%의 정확도를 기록했습니다.



### LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation (https://arxiv.org/abs/2411.04997)
- **What's New**: 본 연구에서는 LLM2CLIP이라는 새로운 접근 방식을 제안하여 대형 언어 모델(LLMs)의 강력한 기능을 활용하여 CLIP의 시각적 표현 학습을 강화합니다. LLM2CLIP은 LLM을 캡션 공간에서 대조 학습(contrastive learning)을 통해 미세 조정하여 결과 임베딩의 텍스트 구별력을 크게 향상시킵니다.

- **Technical Details**: LLM2CLIP은 LLM의 출력 토큰을 더 잘 구별할 수 있도록 미세 조정하는 캡션 대조(CC) 미세 조정 전략을 활용합니다. LoRA( Low-Rank Adaptation)를 사용하여 CC3M 이미지 캡션 데이터셋에서 LLM을 경량 미세 조정했습니다. 이 과정에서 supervised SimCSE 대조 학습 손실을 적용하여 원래 캡션과 재주석된 캡션을 긍정 쌍으로 간주하고, 다른 모든 캡션은 부정 샘플 세트로 처리합니다.

- **Performance Highlights**: LLM2CLIP은 기존의 SOTA EVA02 모델에 대해 16.5% 성능 향상을 이루었으며, 영어 데이터로만 학습된 CLIP 모델을 최첨단의 교차 언어 모델로 변모시켰습니다. 또한 Llava 1.5와 같은 다중 모달 모델 훈련에 통합될 경우 EVA02에 비해 거의 모든 벤치마크에서 지속적인 개선을 나타냈습니다.



### LoFi: Scalable Local Image Reconstruction with Implicit Neural Representation (https://arxiv.org/abs/2411.04995)
- **What's New**: 이번 연구에서는 Neural Fields(신경 필드) 및 Implicit Neural Representations(INRs)을 기반으로 한 LoFi(Local Field)라는 새로운 프레임워크를 소개하여 이미징 역문제를 해결하는 방법을 제시합니다. LoFi는 이미지 재구성을 위해 각 좌표에서 지역 정보를 별도로 처리하며, 이는 기존의 CNN과는 달리 메모리 사용량을 크게 줄일 수 있습니다.

- **Technical Details**: LoFi는 Multi-layer Perceptrons(MLPs)를 사용하여 입력 이미지의 이웃 정보로부터 특정 좌표에서 객체를 회복하는 방식으로 작동합니다. 이 모델은 저해상도 이미지로 학습하더라도 다양한 해상도에서 역문제를 해결할 수 있는 낮은 차원의 사전 정보를 제공합니다. 또한, 1024×1024 이미지에 대해 단 3GB의 메모리로 학습할 수 있어 표준 CNN보다 20배 이상 적은 메모리를 요구합니다.

- **Performance Highlights**: LoFi는 이미지 복원에서 CNN보다 비슷하거나 더 나은 성능을 보였으며, 놀랍게도 10개 미만의 샘플로도 오버피팅 없이 학습 가능합니다. 다양한 이미징 모달리티에서 성능을 검증하였으며, 로우 도스 컴퓨터 단층촬영(LDCT)과 같은 분야에서도 유용성을 입증했습니다.



### SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation (https://arxiv.org/abs/2411.04989)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 SG-I2V라는 새로운 프레임워크를 제안하며, 이는 사전 훈련된 이미지-비디오 생성 모델의 지식을 활용하여 외부 지식 없이 제로샷(zero-shot) 제어 기능을 제공합니다.

- **Technical Details**: SG-I2V는 사용자가 입력 이미지에 바운딩 박스(bounding boxes) 및 관련 경로(trajectories)를 지정할 수 있도록 하며, 그에 따라 생성 과정에서 장면 요소의 움직임을 제어합니다. 주요 기법으로는 세미-정렬(semantically aligned)된 특징 맵을 활용하여 비디오의 구조를 조작합니다.

- **Performance Highlights**: 본 연구의 방법론은 제로샷 기준선에 비해 우수한 성능을 나타내며, 시각 품질(visual quality) 및 운동 충실도(motion fidelity) 측면에서 감독 기반(supervised) 모델과 비교할 수 있는 경쟁력을 보입니다.



### Planar Reflection-Aware Neural Radiance Fields (https://arxiv.org/abs/2411.04984)
- **What's New**: 본 연구에서는 Neural Radiance Fields (NeRF)을 기반으로 하는 새로운 방법인 'reflection-aware NeRF'를 제안하여 복잡한 평면 반사 장면을 정확하게 재구성할 수 있도록 한다. 이 방법은 창문과 같은 평면 반사체를 공동 모델링하고 반사된 광선을 명시적으로 쏘아 복잡한 반사를 처리하는 데 초점을 맞춘다.

- **Technical Details**: 제안된 방법은 단일 Radiance Field를 쿼리하여 주 색상과 반사의 출처를 렌더링한다. 또한, 평면 반사를 렌더링하는 데 있어 진정한 반사 출처를 사용하도록 유도하는 희소 에지 정규화(sparse edge regularization)를 도입하여 잘못된 지오메트리를 생성하는 것을 방지한다. 이로 인해 정확한 장면 지오메트리를 얻을 수 있다.

- **Performance Highlights**: 실제 360도 데이터셋에서의 광범위한 정량적 및 정성적 평가를 통해, 제안된 방법이 복잡한 플레인 반사를 처리하는 데 있어 NeRF의 성능을 상당히 향상시킴을 입증하였다.



### AsCAN: Asymmetric Convolution-Attention Networks for Efficient Recognition and Generation (https://arxiv.org/abs/2411.04967)
Comments:
          NeurIPS 2024. Project Page: this https URL

- **What's New**: 최근의 논문에서는 AsCAN이라는 하이브리드 아키텍처를 소개하며, 이는 convolutional(합성곱)과 transformer(트랜스포머) 블록을 결합하여 비대칭 구조로 설계되었습니다. 초기 단계에서는 더 많은 합성곱 블록을 사용하고, 후반 단계에서는 더 많은 트랜스포머 블록을 사용하는 방식으로 구성되어 있습니다.

- **Technical Details**: AsCAN 아키텍처는 다양한 작업(예: 이미지 인식, 분할, 클래스 조건 이미지 생성)을 지원하며, 성능과 지연(latency) 간의 우수한 균형을 제공합니다. 대규모 text-to-image(텍스트-이미지) 작업을 해결하도록 확장되었으며, 최신 공공 및 상업적 모델과 비교할 때 최첨단 성능을 보여줍니다. 또한, 훈련 효율성을 향상시키기 위해 다단계 훈련 파이프라인을 도입했습니다.

- **Performance Highlights**: AsCAN 모델은 ImageNet-1K 데이터셋에서 기존 모델보다 우수한 처리량-성능(troughput-performance) 균형을 달성했습니다. 특히, 주목 메커니즘에 대한 어떤 최적화 없이도 기존의 효율적인 주목을 사용하는 작업들보다 빠른 추론 속도를 기록하였습니다. 또한, 클래스 조건 생성(task)에서도 동일한 성능을 달성하기 위해 필요한 계산 자원이 절반으로 줄었습니다.



### VAIR: Visuo-Acoustic Implicit Representations for Low-Cost, Multi-Modal Transparent Surface Reconstruction in Indoor Scenes (https://arxiv.org/abs/2411.04963)
Comments:
this https URL

- **What's New**: 본 논문은 실내 장면에서 투명한 표면의 밀집 재구성을 지원하기 위해 음향(acoustic) 및 시각(visual) 감지 모달리티를 융합하는 새로운 방법을 제안합니다. 이를 위해 Generative Latent Optimization을 활용하여 투명한 표면으로 구성된 실내 장면에 대한 암묵적(implicit) 표현을 학습하는 새로운 모델을 개발했습니다.

- **Technical Details**: 관찰된 RGB-D 이미지와 희소한 음향 센서 측정값을 융합하여 VAIR(Visuo-Acoustic Implicit Representation)를 생성하는 다중 감각(fusion) 시스템을 제안합니다. 이 과정에서 음향 매개변수를 시멘틱 분할(semantic segmentation)과 정렬하여 음향 측정치를 3D 공간에 프로젝션하는 Acoustic-Semantic Planar Projection(ASPP) 방법을 사용합니다. VAIR는 생성된 암묵적인 표현을 쿼리하여 이미지 공간에서 투명한 표면의 위치를 렌더링하거나 3D 기하학(geometry)으로 샘플링할 수 있습니다.

- **Performance Highlights**: 제안한 방법은 새로운 데이터셋을 기반으로 질적으로 및 양적으로 평가되었으며, 투명한 표면 재구성에서 기존 최첨단 기술들에 비해 주목할 만한 개선 효과를 보였습니다. RGB-D 카메라와 초음파 센서를 사용하는 저비용 감지 플랫폼을 통한 성능평가 결과, 연구의 기법이 실내 장면의 3D 재구성 정확성에서 상당한 기여를 하고 있음을 확인했습니다.



### Uncovering Hidden Subspaces in Video Diffusion Models Using Re-Identification (https://arxiv.org/abs/2411.04956)
Comments:
          8 pages, 5 tables, 6 figures

- **What's New**: 이번 논문에서는 Latent Video Diffusion Models에 기반한 비디오 생성 기술의 발전을 다루고 있으며, 특히 의료 분야에서의 응용 가능성에 대해 논의합니다. 이 연구는 프라이버시 모델을 잠재 공간(latent space)에서 훈련시키는 것이 컴퓨팅 효율성과 일반화 성능을 모두 향상시킨다는 것을 증명합니다.

- **Technical Details**: 연구에서는 비디오 생성기(latent video generator)의 잠재 공간을 학습하는 방법과 재식별 모델(re-identification model)을 활용하여 생성된 비디오 데이터의 프라이버시와 신뢰성을 평가하는 방법을 제안합니다. 이를 통해 생성된 비디오 데이터가 훈련 데이터에서 암기되지 않았다는 것을 보장하는 다양한 기술적 접근법을 제공하고 있습니다.

- **Performance Highlights**: 연구 결과, Latent Video Diffusion Models가 학습한 훈련 비디오의 최대 30.8%만을 가지고 있다는 것을 발견하였으며, 이로 인해 합성 데이터에서 훈련된 다운스트림 모델의 성능 저하를 설명할 수 있음을 보여주었습니다. 이러한 발견은 비디오 생성 모델의 신뢰성과 실용성 개선에 있어 중요한 기초 자료가 될 것입니다.



### CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM (https://arxiv.org/abs/2411.04954)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 사용자의 입력(텍스트 설명, 이미지, 포인트 클라우드 등)을 기반으로 CAD 모델을 쉽게 생성할 수 있는 통합 CAD 생성 시스템인 CAD-MLLM을 설계하는 것을 목표로 하고 있습니다.

- **Technical Details**: CAD-MLLM은 다양한 다중 모드 입력에 조건부로 파라메트릭 CAD 모델을 생성할 수 있는 첫 번째 시스템입니다. CAD 모델의 command sequence를 활용하고, 고급 대형 언어 모델(LLMs)을 사용하여 다양한 다중 모드 데이터와 CAD 모델의 벡터화된 표현 사이의 특징 공간을 정렬합니다. 이를 위해 Omni-CAD라는 데이터셋을 구축하여 약 450K 개의 CAD 모델과 이들에 대한 다중 모드 데이터를 제공합니다.

- **Performance Highlights**: CAD-MLLM은 기존 조건부 생성 방법보다 현저하게 우수한 성능을 보이며, 노이즈와 결측점에 대한 강력한 내성을 보입니다. 새로운 메트릭인 SegE, DangEL, SIR, FluxEE을 도입하여 생성된 CAD 모델의 topological quality와 enclosure를 평가하는 방법을 제시하였습니다.



### M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding (https://arxiv.org/abs/2411.04952)
Comments:
          Project webpage: this https URL

- **What's New**: 논문에서는 M3DocRAG라는 새로운 multi-modal RAG 프레임워크를 소개합니다. 이 프레임워크는 단일 문서 또는 여러 문서에서 정보 검색과 질문 응답을 처리할 수 있도록 설계되었습니다. 또한, 기존의 문서 시각적 질문 응답(DocVQA)에서는 이미지나 차트와 같은 시각적 정보를 무시하는 문제가 있었으나, M3DocRAG는 이를 해결합니다.

- **Technical Details**: M3DocRAG는 세 단계로 작동합니다: (1) 문서 임베딩 - RGB 이미지에서 시각적 임베딩을 추출하고, (2) 페이지 검색 - 텍스트 쿼리와 높은 유사성이 있는 상위 K 페이지를 검색하며, (3) 질문 응답 - MLM을 통해 검색된 페이지에서 최종 답변을 생성합니다. 이 프레임워크는 다양한 문서 맥락(폐쇄형 및 개방형), 질문 점프(단일 점프 및 다중 점프), 증거 모달리티(텍스트, 차트, 도표 등)를 지원합니다.

- **Performance Highlights**: M3DocRAG는 ColPali와 Qwen2-VL 7B를 사용하여 실험 세 가지 벤치마크(M3DocVQA, MMLongBench-Doc, MP-DocVQA)에서 많은 강력한 기준선보다 우수한 성능을 보였습니다. 특히 MP-DocVQA에서는 최신 성능을 기록했습니다.



### A Reinforcement Learning-Based Automatic Video Editing Method Using Pre-trained Vision-Language Mod (https://arxiv.org/abs/2411.04942)
- **What's New**: 본 논문은 일반적인 비디오 편집을 위한 새로운 두 단계 편집 프레임워크를 제안하며, 사전 훈련된 비전-언어 모델(Vision-Language Model, VLM)을 활용하여 편집 관련 표현을 추출하고 강화 학습(Reinforcement Learning, RL) 기반 편집 프레임워크로 기존 시스템의 미적 품질 차이를 줄여 감술을 향상시키는 방법을 모색한다.

- **Technical Details**: 제안된 방법은 두 개의 주요 단계로 나눌 수 있다. 첫째, 과거의 장면 특정 특징을 추출하는 대신, VLM을 통해 편집 맥락으로 사용할 수 있는 편집 관련 표현을 추출한다. 둘째, RL 기반의 편집 프레임워크를 통해 가상 편집기를 훈련시키여 순차적 편집 결정을 최적화한다. 이러한 구조는 특히 영화와 같은 다양한 장면을 일반적으로 편집하기 위한 특별한 접근 방식을 제공한다.

- **Performance Highlights**: 제안된 방법은 실제 영화 데이터 셋을 사용하여 평가되었으며, 실험 결과는 VLM 기반의 맥락 표현과 RL 기반 편집 프레임워크의 학습 능력이 효과적임을 입증하였다. 이는 비디오 품질의 격차를 줄이고, 더 나은 순차적 편집 결정을 내리는 데 기여하였다.



### SaSR-Net: Source-Aware Semantic Representation Network for Enhancing Audio-Visual Question Answering (https://arxiv.org/abs/2411.04933)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 Audio-Visual Question Answering (AVQA)에서 새로운 모델인 Source-aware Semantic Representation Network (SaSR-Net)을 소개합니다. 이 모델은 복잡한 다중 모달 장면을 해석하고, 시청각 정보를 정확히 연결하여 답을 찾아내는 데 중점을 두고 있습니다.

- **Technical Details**: SaSR-Net은 두 가지 전략을 사용하여 AVQA의 이해를 강화합니다: (1) Source-wise Learnable Tokens를 사용하여 오디오와 비주얼 데이터의 중요한 의미적 특징을 포착하고, (2) 공간 및 시간적 주의 메커니즘을 활용하여 관련된 비주얼 및 오디오 영역을 식별하고 동기화합니다.

- **Performance Highlights**: SaSR-Net은 Music-AVQA 및 AVQA-Yang 데이터셋에서 실험을 통해 기존 AVQA 방법들을 초월하는 성능을 보였으며, 복잡한 시청각 데이터를 효과적으로 관리하는 능력을 입증했습니다.



### DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion (https://arxiv.org/abs/2411.04928)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 	extbf{DimensionX}라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 단일 이미지로부터 사실적인 3D 및 4D 장면을 생성할 수 있도록 설계되었습니다. 특히, 기존 비디오 확산 모델의 한계를 극복하기 위해 공간적 (spatial) 및 시간적 (temporal) 요소를 분리하여 제어할 수 있는 ST-Director를 제안합니다.

- **Technical Details**: DimensionX는 영상 확산 기술을 기반으로 하여 LoRAs를 통해 차원 인식 제어를 가능하게 합니다. 이 프레임워크는 공간적 및 시간적 차원에 따라 다르게 변하는 데이터셋을 구성하여 학습합니다. ST-Director는 비디오 생성 과정에서 공간적 및 시간적 요소를 추출하고, 혼합 차원 제어를 위해 훈련이 필요 없는 구성 방법을 도입합니다.

- **Performance Highlights**: 다양한 실제 및 합성 데이터셋을 통한 실험 결과, DimensionX는 비디오 생성, 3D 및 4D 장면 생성에서 이전 방법들보다 우수한 결과를 보여주었습니다. 이는 현실적이고 동적인 환경을 생성할 수 있는 가능성을 제시합니다.



### StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration (https://arxiv.org/abs/2411.04925)
- **What's New**: 이번 논문에서는 StoryAgent라는 새로운 다중 에이전트 프레임워크를 제안하여 사용자 지정 스토리텔링 비디오 생성(Customized Storytelling Video Generation, CSVG)에서 주인공 일관성을 유지하는 문제를 해결하고자 합니다. 기존 방법들의 한계를 극복하여 더욱 일관된 스토리텔링 비디오를 생성할 수 있는 가능성을 제시합니다.

- **Technical Details**: StoryAgent는 CSVG를 다수의 하위 작업으로 분해하며 각 에이전트는 스토리 디자인, 스토리보드 생성, 비디오 생성, 에이전트 조정 및 결과 평가 등 저마다의 역할을 맡습니다. 또한, LoRA-BE(Low-Rank Adaptation with Block-wise Embeddings)라는 커스터마이즈된 Image-to-Video(I2V) 방법을 도입하여 단일 장면 내에서 주인공의 일관성을 유지하는데 중점을 두고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안한 방법이 최신 기술들과 비교하여 스토리텔링 비디오의 일관성을 크게 향상시켰고, 사용자 맞춤형 비디오 생성에서 우수한 성과를 보였음을 입증했습니다. 이 프레임워크는 다양한 비디오 생성 작업을 수행할 수 있는 유연성을 제공함으로써, 스토리텔링 비디오 제작의 경계를 넓힐 수 있는 중요한 도구로 자리 잡을 가능성이 큽니다.



### MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views (https://arxiv.org/abs/2411.04924)
Comments:
          NeurIPS 2024, Project page: this https URL, Code: this https URL

- **What's New**: MVSplat360을 소개하며, 이 접근법은 다양한 현실 세계 장면의 360° 새로운 보기 합성을 위해 희소 관찰(sparse observations)만을 사용합니다. 기존의 접근 방식들이 높은 품질의 결과를 얻기 어려운 난제를 해결하고자 하며, 효율적으로 기하학적 3D 재구성(geometry-aware 3D reconstruction)과 시공간적으로 일관된 비디오 생성을 결합합니다.

- **Technical Details**: MVSplat360은 피드-포워드 3D Gaussian Splatting (3DGS) 모델을 사용하여, 안정적인 비디오 확산 모델을 통해 전달된 피쳐들을 레이턴트 공간(latent space)으로 직접 렌더링합니다. 이 신뢰성 있는 3D 재구성을 기반으로 공간-시간 일관성을 유지하며 포토리얼리스틱 3D 보기들을 생성합니다.

- **Performance Highlights**: MVSplat360은 DL3DV-10K 데이터셋과 RealEstate10K 데이터셋에서 실험을 통해 기존의 최신 방법들에 비해 월등한 시각적 품질을 달성하였습니다. 특히 5개의 희소 입력 관찰만으로도 다양한 360° NVS 작업에서 우수한 성능을 입증하였습니다.



### VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos (https://arxiv.org/abs/2411.04923)
Comments:
          Technical Report of VideoGLaMM

- **What's New**: 비디오 및 텍스트 간의 정밀한 정렬 문제를 해결하기 위해 VideoGLaMM이라는 새로운 대규모 멀티모달 모델을 소개합니다. 이 모델은 사용자 제공 텍스트 입력을 기반으로 비디오에서 픽셀 레벨의 정밀 grounding을 목표로 설계되었습니다.

- **Technical Details**: VideoGLaMM는 대규모 언어 모델(LLM), 이중 시각 인코더, 그리고 정확한 마스크 생성을 위한 시공간 디코더로 구성되어 있습니다. 이 아키텍처는 비디오 콘텐츠의 공간적 및 시간적 요소를 동기화하여 텍스트의 지침과 결합하여 상세한 비디오 내용을 처리합니다. 이를 위해 38,000개의 비디오-QA 삼중 항목과 83,000개의 객체 및 671,000개의 마스크를 포함하는 멀티모달 데이터셋을 활용하였습니다.

- **Performance Highlights**: 실험 결과, VideoGLaMM 모델은 Grounded Conversation Generation, Visual Grounding, Referring Video Segmentation 작업에서 기존의 모든 접근 방식을 지속적으로 능가하는 성능을 보였습니다.



### Robust Iris Centre Localisation for Assistive Eye-Gaze Tracking (https://arxiv.org/abs/2411.04912)
- **What's New**: 본 연구에서는 제약이 없는 조건에서의 강건한 홍채 중심(localisation) 위치를 파악하는 문제를 다루고 있습니다. 이는 우리의 안구 추적(eye-gaze tracking) 플랫폼의 핵심 요소입니다.

- **Technical Details**: U-Net 변형(variants)을 사용한 세분화(segmentation) 기반 및 회귀(regression) 기반 접근 방식을 조사하여 홍채 중심(localisation) 인식을 개선하였습니다. 이전의 Bayes(베이즈) 분류 기반 접근법에 비해 성능이 뛰어난 결과를 도출했습니다.

- **Performance Highlights**: 이 연구에서 달성한 결과는 최신 기술(state-of-the-art)과 비교할 때 동등하거나 더 나은 성과를 보였습니다. Bayes 분류기(Bayes classifier)에 비해 실시간(real-time) 성능을 희생하지 않으면서도 획기적인 개선을 이루었습니다.



### In the Era of Prompt Learning with Vision-Language Models (https://arxiv.org/abs/2411.04892)
Comments:
          ICVGIP 2024, Young Faculty Symposium

- **What's New**: 본 연구에서는 대규모 기초 모델인 CLIP의 도메인 일반화(Domain Generalization, DG)를 위한 새로운 프롬프트 학습 전략인 	extsc{StyLIP}을 소개합니다. StyLIP은 스타일 프로젝트(style projectors)를 사용하여 CLIP의 시각 인코더에서 시각적 스타일과 콘텐츠를 분리하고 도메인 특화된 프롬프트 토큰을 학습하여 여러 도메인에서의 원활한 적응을 가능하게 합니다.

- **Technical Details**: StyLIP은 CLIP의 시각 인코더에서 시각적 스타일과 콘텐츠를 분리하며, 도메인 특화된 프롬프트 토큰을 학습합니다. AD-CLIP을 통해 비지도 도메인 적응(Unsupervised Domain Adaptation, DA)을 수행하며, 이미지 스타일과 콘텐츠 특징을 활용하여 도메인 불변의 프롬프트를 학습합니다. 이 방법들은 여러 벤치마크에서 최신 방법들을 초과하는 성능을 보여줍니다.

- **Performance Highlights**: StyLIP은 5개의 DG 벤치마크에서 우수한 성능을 보였으며, AD-CLIP 역시 3개의 DA 데이터셋에서 최신 기법들을 초과하는 성능을 달성했습니다. 미래 연구는 원거리 탐지에서의 새로운 클래스 발견(class discovery)에 초점을 맞추어, 비구조적 환경에서의 신뢰성을 강화하는 방향으로 진행될 것입니다.



### Boosting Latent Diffusion with Perceptual Objectives (https://arxiv.org/abs/2411.04873)
Comments:
          Pre-print

- **What's New**: 본 논문에서는 latent diffusion models (LDMs)에서 일반적으로 발생하는 autoencoder (AE)의 decoder와 diffusion 모델 간의 disconnect 문제를 해결하기 위해 latent perceptual loss (LPL)를 제안합니다. 이를 통해 더 선명하고 현실감 있는 이미지를 생성할 수 있습니다.

- **Technical Details**: LPL은 AE decoder의 중간 특징을 활용하여 LDM의 훈련 신호를 풍부하게 만듭니다. 이 방식은 기존의 image-to-image translation 작업에서 사용된 perceptual losses의 개념을 발전시킨 것입니다. 실험에서는 ImageNet-1k, CC12M 및 S320M 데이터셋을 사용하여 생성 모델의 성능을 평가하며, DDPM 및 flow matching 모델을 포함한 다양한 생성 모델 형식에 적용되었습니다.

- **Performance Highlights**: LPL을 사용했을 때 FID (Fréchet Inception Distance)에서 6%에서 20%까지 성능 향상이 이루어졌습니다. 질적 분석 결과, 생성된 이미지는 더욱 선명하고 고주파 이미지 세부사항을 잘 반영하고 있음을 보여줍니다.



### ZAHA: Introducing the Level of Facade Generalization and the Large-Scale Point Cloud Facade Semantic Segmentation Benchmark Datas (https://arxiv.org/abs/2411.04865)
Comments:
          Accepted to WACV 2025 (IEEE/CVF Winter Conference on Applications of Computer Vision (WACV))

- **What's New**: 이번 연구에서는 건축 다양성을 포괄하는 전통적인 facade semantic segmentation의 문제를 해결하기 위해, 국제 도시 모델링 표준을 기반으로 한 새로운 Level of Facade Generalization (LoFG) 계층적 facade 클래스를 제안합니다. 이 연구를 통해 지금까지 가장 큰 3D facade segmentation 데이터셋을 소개하며, 이는 6억 1천만 개의 주석이 달린 포인트를 포함합니다.

- **Technical Details**: LoFG는 LoFG2와 LoFG3에서 각각 5개와 15개의 클래스를 제공하며, 이는 현실 세계의 복잡한 facade 클래스를 지원합니다. 또한 우리는 기존의 semantic segmentation 방법이 LoFG 클래스에서 어떻게 성능을 발휘하는지를 분석하고, facade segmentation에 대한 해결되지 않은 문제들에 대한 논의를 제공합니다.

- **Performance Highlights**: 이 연구는 3D facade semantic segmentation 방법의 발전을 촉진할 것으로 기대되며, 도시 디지털 트윈을 생성하는 데 필수적인 강력한 segmentation을 가능하게 합니다. 기존의 연구를 통해 얻어진 성과와 비교할 때, our LoFG 데이터셋은 601 million 포인트 등에서 큰 차별성을 보여줍니다.



### A multi-purpose automatic editing system based on lecture semantics for remote education (https://arxiv.org/abs/2411.04859)
- **What's New**: 이번 논문에서는 세미틱(Semantic) 분석을 기반으로 한 자동 다중 카메라 편집 시스템을 제안합니다. 이는 온라인 강의를 위해 가장 주목해야 할 뷰(view)를 선택하여, 원거리 학습자의 주의를 유도하는 것을 목표로 합니다.

- **Technical Details**: 제안된 시스템은 수업 이벤트를 세미틱 분석하여 주목 영역을 판단하고, 일반적인 촬영 규칙을 고려하여 편집합니다. 각 샷의 주목 점수를 평가하기 위해 저자들은 사건 인식 기술을 사용하며, 시스템은 사용자에게 다중 모드(실시간 방송, 오프라인 편집, 및 균형 모드)를 지원합니다.

- **Performance Highlights**: 논문에서는 수집한 데이터셋을 통해 제안한 시스템의 효과성을 정량적 및 정성적으로 분석했으며, 사용자 조사를 통해 실제 사용자 경험도 비교하였습니다.



### D$^3$epth: Self-Supervised Depth Estimation with Dynamic Mask in Dynamic Scenes (https://arxiv.org/abs/2411.04826)
Comments:
          Open sourced

- **What's New**: 본 논문에서는 동적 장면에서의 자기 지도 심도 추정을 위한 새로운 방법인 D$^3$epth를 제안합니다. 기존의 자기 지도 심도 추정은 정적인 장면을 가정하고 개발되었으나, D$^3$epth는 동적 개체를 포함하는 복잡한 환경에서도 효과적으로 작동할 수 있도록 설계되었습니다.

- **Technical Details**: D$^3$epth는 두 가지 주요 접근 방식을 통해 동적 개체의 문제를 해결합니다. 첫째, 자기 지도 프레임워크 내에서는 동적 개체가 존재할 것으로 예상되는 영역을 식별하기 위해 재투영(reprojection) 제약 조건을 설계하여 동적 마스크를 구성합니다. 둘째, 다중 프레임 심도 추정 과정에 대해서는 인접 프레임을 활용하여 동적 객체와 관련된 영역을 식별하는 비용 볼륨(cost volume) 자동 마스킹 전략을 도입합니다. 또한, 스펙트럴 엔트로피 불확실성 모듈이 적용되어 동적 환경의 깊이 융합 중 불확실성 추정을 안내합니다.

- **Performance Highlights**: KITTI 및 Cityscapes 데이터셋에서 실시된 실험 결과, D$^3$epth는 기존의 자기 지도 단안적(depth estimation) 기준을 일관되게 초과하는 성능을 보였습니다. 이는 동적 장면에서의 심도 추정에서 큰 진전을 나타냅니다.



### End-to-end Inception-Unet based Generative Adversarial Networks for Snow and Rain Removals (https://arxiv.org/abs/2411.04821)
- **What's New**: 본 논문은 두 개의 Generative Adversarial Networks (GANs)를 활용하여 눈과 비를 개별적으로 제거하는 새로운 프레임워크를 제안합니다. 이 연구는 기존 단일 네트워크 접근 방식의 한계를 극복하기 위해 두 개의 개별 모델을 사용하여 각 매개변수의 고유한 특성을 활용합니다.

- **Technical Details**: 각 GAN 아키텍처는 U-net 생성기 네트워크에 특징 추출 단계를 통합하여 다양한 크기와 외관의 눈과 비 입자를 효과적으로 처리할 수 있도록 설계되었습니다. 또한, 눈과 비의 독특한 특성을 고려한 두 가지 새로운 손실 함수가 설계되어, 생성기가 눈/비가 있는 영역에 좀 더 집중할 수 있도록 유도합니다.

- **Performance Highlights**: 제안된 desnowing 및 deraining 접근 방식은 합성 및 현실적인 데이터셋에서 기존 최첨단 접근 방식과 비교했을 때 상당한 개선을 보였습니다. 특히, 현실적 테스트 데이터셋을 사용하여 정량적인 평가를 수행하여 학습 기반 기법이 실제 환경에서도 효과적으로 작동할 수 있음을 입증했습니다.



### GANESH: Generalizable NeRF for Lensless Imaging (https://arxiv.org/abs/2411.04810)
- **What's New**: 본 논문에서는 GANESH라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 멀티뷰 렌즈리스 이미지에서 동시에 정제(refinement) 및 새로운 뷰 합성(view synthesis)을 가능하게 합니다. 기존 방법들과 달리, 특정 장면에 대한 재훈련 없이 즉시 추론(inference)을 지원합니다.

- **Technical Details**: GANESH는 멀티뷰 렌즈리스 이미지를 통해 3D 장면을 복원하는 데 적합한 방법론입니다. 이 모델은 합성된 멀티뷰 렌즈리스 이미지 데이터로 학습되며, 실제 멀티뷰 렌즈리스 캡처에도 적용할 수 있습니다. 또한, 장면 특정 튜닝(scene-specific tuning)이 가능하여 정밀한 복원이 가능합니다.

- **Performance Highlights**: 실험 결과, GANESH는 기존의 정제 및 새로운 뷰 합성을 개별적으로 처리하는 방법들보다 재구성 정확도와 정제 품질에서 우수한 성능을 보였습니다. 또한, LenslessScenes라는 첫 번째 멀티뷰 렌즈리스 데이터셋도 공개하여 연구를 지원합니다.



### Taming Rectified Flow for Inversion and Editing (https://arxiv.org/abs/2411.04746)
- **What's New**: 이 논문은 FLUX 및 OpenSora와 같은 Rectified-flow 기반의 diffusion transformers가 이미지 및 비디오 생성에서 탁월한 성능을 보였으나, 도출 과정의 오류로 인해 수치적 역전이 불만족스러워진 문제를 다루고 있습니다. 이를 해결하기 위해 새로운 트레이닝 없이 사용할 수 있는 샘플러인 RF-Solver를 제안합니다. RF-Solver는 비선형 구성 요소의 오차를 줄여 inverse precision을 향상시킵니다.

- **Technical Details**: RF-Solver는 rectified flow ODE의 정확한 공식을 도출하고, 고차 Taylor expansion을 이용해 비선형 성분을 추정하여 각 타임스텝에서 근사 오류를 상당히 줄입니다. RF-Edit는 RF-Solver를 기반으로 설계되어 이미지 및 비디오 편집을 위한 특수 하위 모듈을 포함합니다. 이 과정에서 self-attention layer의 특징을 공유함으로써 원본 이미지 또는 비디오의 구조 정보를 효과적으로 보존합니다.

- **Performance Highlights**: 실험을 통해 RF-Solver가 변환 및 구축 과정에서 정확성을 대폭 향상시킨 것을 증명했습니다. RF-Edit는 이미지 및 비디오 편집에서 우수한 성능을 보여주었으며, 기존 방법들과 비교했을 때 여러 다운스트림 작업에서 향상된 결과를 나타냈습니다.



### Controlling Human Shape and Pose in Text-to-Image Diffusion Models via Domain Adaptation (https://arxiv.org/abs/2411.04724)
- **What's New**: 이 논문에서는 3D 인간 파라메트릭 모델인 SMPL을 사용하여 사전 훈련된 텍스트-이미지 확산 모델의 조건부 제어를 수행하는 방법론을 제안합니다. 우리는 합성 데이터 생성을 통해 대규모 데이터셋 생성을 비용 효율적으로 수행하고, 이를 통해 기존 모델의 시각적 충실도를 유지하고자 합니다.

- **Technical Details**: 제안된 방법은 SMPL 벡터를 전문가가 훈련한 ControlNet을 사용하여 이미지 생성 시, 클래스 분류자 없는 가이드 벡터에 통합하는 방식으로 적용됩니다. 이를 통해 합성 데이터와 실제 데이터 간 시각적 도메인 간극을 줄이는 동시에 이미지의 품질을 유지할 수 있습니다. ControlNet 기반의 아키텍처에 대해 SURREAL 데이터셋으로 미세 조정(fine-tune) 합니다.

- **Performance Highlights**: 실험 결과, 우리의 모델은 기존의 2D 포즈 기반 ControlNet보다 더 뛰어난 형태와 포즈 다양성을 보여주었으며, 시각적 충실도(KID 및 Inception Score 기준)를 유지하고 안정성을 개선해, 인간 애니메이션과 같은 다운스트림 작업에 유용함을 입증하였습니다.



### NeuroFly: A framework for whole-brain single neuron reconstruction (https://arxiv.org/abs/2411.04715)
- **What's New**: NeuroFly를 소개합니다. 이는 자동화된 단일 신경세포 재구성을 위한 대규모 효율적인 프레임워크로, 세분화(segmentation), 연결(connection), 그리고 검수(proofreading)라는 세 가지 단계로 과정이 나뉩니다.

- **Technical Details**: NeuroFly는 3D 이미지 기반 경로 추적(image-based path following) 방법을 사용하여 신경세포 조각들을 연결합니다. 이 프레임워크는 고해상도 메조스케일(mesoscopic) 이미지를 분석하여 복잡한 시나리오에서도 신경세포 구조를 효과적으로 재구성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 내부 데이터셋을 활용해 NeuroFly의 효율성을 검증하였으며, 밀집한 가지, 약한 축삭, 오염된 이미지 등 다양한 도전 과제를 포함한 시나리오에서 성능을 평가했습니다. 또한, 이 프레임워크와 관련된 데이터셋 및 시각화 도구를 공개하여 연구자 간 협업을 촉진하고자 합니다.



### Revisiting Disparity from Dual-Pixel Images: Physics-Informed Lightweight Depth Estimation (https://arxiv.org/abs/2411.04714)
Comments:
          Accepted to IEEE Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 연구에서는 듀얼 픽셀(Dual-Pixel) 이미지를 사용하여 고성능의 깊이(Disparity) 추정 방법을 제안합니다. 기존의 딥러닝 기반 방법들은 많은 파라미터를 가지고 있지만, 깊이 제약을 온전히 활용하지 못해 성능이 제한적입니다. 새로운 방법은 경량의 딥러닝 네트워크를 사용하여 깊이 제약을 명시적으로 해결하고, DP의 물리적 및 시스템적 깊이 특성을 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 본 연구는 DP 특유의 깊이 오류를 매개변수적으로 모델링하고 이를 훈련 중 샘플링에 활용하여 네트워크가 DP의 독특한 속성을 획득하게 합니다. 또한, 비학습 기반의 정교화(framework)를 제안하여 네트워크 출력의 신뢰도 맵을 적절히 수정함으로써 본질적인 깊이 확장 오류를 효과적으로 처리합니다. 이 접근 방식은 다른 모델의 추정 결과에 일반적으로 적용될 수 있는 유연성을 가지고 있습니다.

- **Performance Highlights**: 제안된 방법은 DP 데이터셋을 사용하지 않고도 기존 방법보다 1/5의 시스템 크기로 최첨단 결과를 달성하였으며, 이는 방법의 효과성을 잘 보여줍니다. 코드와 데이터셋은 프로젝트 사이트에서 제공됩니다.



### Multi-Reward as Condition for Instruction-based Image Editing (https://arxiv.org/abs/2411.04713)
- **What's New**: 본 논문은 기존의 instruction 기반 이미지 편집에서의 훈련 데이터 품질 문제를 다루기 위해, multi-perspective reward 데이터를 도입하여 훈련 데이터의 질을 높이는 방법을 제안합니다. 기존의 방법과 달리, ground-truth 이미지 품질을 직접 개선하기보다는, 여러 관점에서의 보상을 통합하여 편집 모델의 성능을 향상시킵니다.

- **Technical Details**: 논문에서 제안하는 방법은 1) GPT-4o를 기반으로 한 정량적 메트릭 시스템을 설계하여, instruction following, detail preserving, generation quality의 세 가지 관점에서 생성 품질을 평가합니다. 이를 통해 수집된 RewardEdit20K 데이터셋을 만들었으며, 2) 멀티 리워드를 편집 모델에 통합해 학습하며, 리워드는 임베딩 형태로 인코딩되고 U-Net에 보조 조건으로 활용됩니다. 3) 현실적인 이미지 편집 평가 베치인 Real-Edit를 생성하여, 다양한 7개의 카테고리와 관련된 80개의 고품질 이미지를 활용한 평가를 진행합니다.

- **Performance Highlights**: 실험 결과, 제안된 multi-reward 조건화 모델은 InsPix2Pix 및 SmartEdit와 같은 인기 있는 편집 파이프라인에서 뛰어난 성능을 발휘하며, 특히 사람의 평가에서도 우수한 결과를 보여줍니다.



### SEE-DPO: Self Entropy Enhanced Direct Preference Optimization (https://arxiv.org/abs/2411.04712)
- **What's New**: 이 논문에서는 인간 피드백을 통해 강화 학습을 진행하면서 안정성을 높이기 위해 자가 엔트로피 정규화 기법을 도입하였으며, 이는 DPO 기반의 알고리즘에서 발생할 수 있는 보상 해킹(reward hacking)을 완화하는 데 효과적임을 입증하였습니다.

- **Technical Details**: DPO(Direct Preference Optimization)는 대규모 언어 모델(LLM)의 인간의 선호도에 따라 조정하는 데 사용되며, 최근에는 텍스트-이미지 확산 모델의 품질 개선에도 적용되었습니다. 그러나 DPO 기반 기법은 과적합(overfitting)과 보상 해킹에 매우 취약합니다. 본 연구에서는 자가 엔트로피 정규화가 포함된 강화 학습을 통해 이러한 문제를 해결하고자 합니다. 자가 엔트로피 정규화는 모델이 보다 넓은 탐색을 하도록 장려하여 보다 강력한 성능을 도출합니다.

- **Performance Highlights**: 인간 피드백과 자가 엔트로피 정규화를 결합함으로써 이미지의 다양성과 특이성이 크게 향상되었습니다. 다양한 이미지 생성 메트릭에서 최신 기술을 초과하는 성과를 달성하였으며, 기존 방법들과 비교하여 더욱 다양한 채색과 시각적으로 우수한 이미지를 생성할 수 있음을 보여주었습니다.



### Progressive Multi-Level Alignments for Semi-Supervised Domain Adaptation SAR Target Recognition Using Simulated Data (https://arxiv.org/abs/2411.04711)
- **What's New**: 최근, 자동 목표 인식(ATR)에서 합성 개구 레이더(SAR) 이미지를 활용하는 연구가 주목받고 있습니다. 이 연구는 부족한 측정 데이터를 해결하기 위해 시뮬레이션 데이터를 활용하여 ATR 모델을 훈련하는 방식을 제안하고 있습니다. 특히, 본 논문에서는 비지도 도메인 적응(UDA) 기술을 통해 실제 데이터와 시뮬레이션 데이터 간의 도메인 차이를 좁히고자 합니다.

- **Technical Details**: 논문에서는 점진적 다중 수준 정렬(progressive multi-level alignments)을 포함하는 반지도 도메인 적응(SSDA) 프레임워크를 제안합니다. 이 방법은 두 도메인 이미지의 웨이브렛 분해 서브밴드 간의 차이를 분석하여 웨이브렛 변환 데이터 증강(PWTDA)을 수행하고, 그에 따라 도메인 레벨의 정렬을 달성합니다. 또한, 소스 도메인 인스턴스와 타겟 프로토타입 간의 근접하게 맞추기 위한 비대칭 인스턴스-프로토타입 정렬(AIPA) 전략을 개발하였습니다.

- **Performance Highlights**: SAMPLE 데이터세트를 기반으로 한 실험 결과, 제안된 방법은 타겟 도메인에서 클래스당 단 하나의 레이블 샘플만으로도 99.63%와 98.91%의 인식 정확도를 달성하였습니다. 이는 기존 SSDA 기술보다 월등한 성능을 나타냅니다.



### TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation (https://arxiv.org/abs/2411.04709)
Comments:
          The project is publicly available at this https URL

- **What's New**: 이 논문에서는 TIP-I2V라는 이미지-비디오 생성에 특화된 최초의 대규모 데이터셋을 소개합니다. 이 데이터셋은 170만 개 이상의 고유한 사용자 제공 텍스트 및 이미지 프롬프트를 포함하고 있으며, 여러 최신 이미지-비디오 모델에서 생성된 비디오도 포함되어 있습니다.

- **Technical Details**: TIP-I2V는 사용자가 제공한 텍스트와 이미지 프롬프트를 기반으로 이미지-비디오 모델을 개발하기 위한 데이터셋입니다. 이 데이터셋은 기존의 텍스트-비디오(VidProM) 및 텍스트-이미지(DiffusionDB) 데이터셋과 비교하여 텍스트와 이미지 모두를 시작점으로 삼으며, 각 텍스트 프롬프트는 정적인 요소를 어떻게 동적으로 변화시킬지를 설명합니다.

- **Performance Highlights**: TIP-I2V는 연구자들이 사용자 선호도를 분석하고, 다양한 성능을 평가할 수 있게 도와줍니다. 또한, 데이터셋은 새로운 연구 방향을 제시하며, 안전성 향상에 기여할 수 있는 기반을 제공함으로써 이미지-비디오 모델의 개선을 이끌어낼 것입니다.



### From CNN to ConvRNN: Adapting Visualization Techniques for Time-Series Anomaly Detection (https://arxiv.org/abs/2411.04707)
- **What's New**: 이번 연구에서는 비디오 데이터에서 이상 탐지를 수행하는 'time distributed' convRNN 모델을 구현하고, 이를 통해 모델의 결정 과정을 해석할 수 있는 기술을 개발하는 데 초점을 맞추었습니다.

- **Technical Details**: 본 연구에서는 Keras 라이브러리를 사용하여 VGG19를 convolutional 요소로, GRU를 순차적 요소로 선택했습니다. VGG19는 'Time Distributed' 레이어에 캡슐화되어 각 이미지에 동일한 처리를 적용하였고, 이 정보는 GRU로 전달되어 순차 분석을 수행했습니다. 또한 Grad-CAM, saliency maps, feature maps, 활성화 맵 등의 설명 가능성 기법을 사용하여 모델의 내부 메커니즘을 시각화했습니다.

- **Performance Highlights**: 연구 결과, 활성화 맵은 보안 요원과 같은 비전문가에게 보다 실용적이며 이해하기 쉬운 결과를 제공하는 것으로 나타났습니다. 시퀀스의 서로 다른 이미지에 대해 모델이 주목하는 영역에 차이가 있으며, 새로운 시각화 기법은 활성화의 강도를 평가할 수 없는 일부 단점이 있지만, 전반적인 특징을 보다 명확하게 파악할 수 있게 해주었습니다.



### ESC-MISR: Enhancing Spatial Correlations for Multi-Image Super-Resolution in Remote Sensing (https://arxiv.org/abs/2411.04706)
- **What's New**: 이번 논문에서는 다중 이미지 초해상도(Multi-Image Super-Resolution, MISR) 문제를 해결하기 위해 새로운 프레임워크인 ESC-MISR(Enhancing Spatial Correlations in MISR)를 제안합니다. 본 연구는 서로 상관 관계가 약한 위성의 저해상도(Low-Resolution, LR) 이미지를 바탕으로 고해상도(High-Resolution, HR) 이미지를 생성하는 것을 목표로 합니다.

- **Technical Details**: ESC-MISR는 이미지 간의 공간-시간 관계를 효과적으로 활용하여 HR 이미지를 복원하는 구조로 설계되었습니다. 본 프레임워크는 다중 이미지 공간 변환기(Multi-Image Spatial Transformer, MIST)라는 새로운 융합 모듈을 도입하여 LR 이미지 간의 공간 상관 관계를 강화하며, 훈련 단계에서 랜덤 셔플 전략을 활용하여 시간적 종속성을 감소시킵니다. 또한, 비전 트랜스포머(Vision Transformer) 및 CNN(Convolutional Neural Networks) 믹스인 CMT(CNNs-Meet-Transformers)를 사용하여 LR 이미지를 인코딩합니다.

- **Performance Highlights**: ESC-MISR는 PROBA-V 데이터셋에서 각각 0.70dB 및 0.76dB의 cPSNR(확률적 피크 신호 대 잡음 비율) 개선을 달성하며, 최신 MISR-RS 방법들과 비교하여 우수한 성능을 입증합니다.



### Dynamic Brightness Adaptation for Robust Multi-modal Image Fusion (https://arxiv.org/abs/2411.04697)
Comments:
          Accepted by IJCAI 2024

- **What's New**: 이 논문에서는 동적 밝기 변화에 강한 이미지를 융합하는 새로운 접근법인 BA-Fusion(일단 밝기 적응형 다중 모달 동적 융합 프레임워크)을 제안합니다. 이 방법은 Brightness Adaptive Gate(BAG) 모듈을 포함하여 динамичного 밝기 상황에서도 강력한 이미지 융합을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: BA-Fusion은 두 부분으로 구성되어 있습니다: Brightness Adaptive Gate(BAG)와 다중 모달 융합 백본 네트워크. BAG 모듈은 데이터 기반으로 밝기 변화에 따라 가장 관련성이 높은 특징 채널을 선택하고, 밝기와 무관한 채널 특징은 구조적 정보를 보존하는 데 사용됩니다. 훈련 과정에서 우리는 밝기 일관성 손실 함수(brightness consistency loss function)를 사용하여 BAG 모듈을 최적화합니다.

- **Performance Highlights**: 다양한 밝기 조건에서의 실험 결과, BA-Fusion 기법이 최신 방법들에 비해 멀티모달 이미지 정보를 잘 보존하고 시각적 충실도를 유지하는 데 매우 우수하다는 것을 입증했습니다. 또한, 환경 밝기 변화에도 불구하고 뛰어난 강건성을 보여주었습니다.



### Reciprocal Point Learning Network with Large Electromagnetic Kernel for SAR Open-Set Recognition (https://arxiv.org/abs/2411.04693)
- **What's New**: 기존의 Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) 방법의 제한에 대한 해결책으로, Open Set Recognition (OSR) 기법을 활용하여 알려진 클래스와 미지의 클래스를 효과적으로 분류할 수 있는 새로운 방법이 제안되었습니다.

- **Technical Details**: 이 연구에서는 reciprocal point learning (RPL)에 기반한 특징 학습 프레임워크를 구성하여, 잠재적인 미지의 클래스에 대한 경계를 설정합니다. 이후 SAR 이미지에서 대형 속성 산란 중심 모델을 기반으로 컨볼루션 커널(convolutional kernels)을 설계하여 비선형 특징을 추출하는 방법을 제안했습니다.

- **Performance Highlights**: MSTAR 데이터셋에서 실시된 실험을 통해 ASC-RPL 접근 방식이 기존의 주요 방법들에 비해 우수한 성능을 보여주었습니다.



### Personalized Federated Learning for Cross-view Geo-localization (https://arxiv.org/abs/2411.04692)
Comments:
          6 pages, 2 figures, Preprint submitted to the IEEE 26th International Workshop on Multimedia Signal Processing (MMSP)

- **What's New**: 본 논문에서는 Federated Learning (FL)과 Cross-view Image Geo-localization (CVGL) 기술을 결합한 새로운 방법론을 제안합니다. 자율주행 차량의 데이터 프라이버시와 이질성 문제를 해결하기 위해 선택적으로 모델 파라미터를 공유할 수 있는 개인화된 federated learning 시나리오를 도입합니다.

- **Technical Details**: 제안한 방법론은 coarse-to-fine 접근 방식을 구현하며, 클라이언트는 지방의 특성을 반영한 자세한 특징을 보존하면서 대략적인 특성 추출기만 공유합니다. 이를 통해 각 차량의 운행 환경의 특성을 잘 반영하고, 협업 학습의 이점을 취할 수 있습니다. 또한, coarse 특징 추출기만 선택적으로 공유함으로써 데이터 전송량을 줄여 계산 효율성을 개선합니다.

- **Performance Highlights**: KITTI 데이터셋과 위성 이미지를 결합하여 전통적인 중앙 집중형 및 단일 클라이언트 학습 방식과 비교한 결과, 우리의 federated CVGL 방법이 중앙 집중식 학습과 유사한 성과를 거두면서 데이터 프라이버시를 유지하는 것으로 나타났습니다. 제안한 부분 모델 공유 전략은 전통적인 FL과 비교할 때 유사하거나 약간 더 나은 성능을 보여주며, 정확성을 유지하면서 통신 오버헤드를 크게 줄였습니다.



### DNN-based 3D Cloud Retrieval for Variable Solar Illumination and Multiview Spaceborne Imaging (https://arxiv.org/abs/2411.04682)
Comments:
          4 pages, 4 figures

- **What's New**: 이번 연구에서는 다양한 카메라 위치 및 태양 방향을 고려하여 3D 구름 복원을 위한 최초의 스케일 가능한 DNN 기반 시스템인 PIVOT-CT를 소개합니다. 이를 통해 구름의 회복에 있어 더 큰 유연성을 얻게 되었습니다.

- **Technical Details**: PIVOT-CT는 multiview 구름 강도 영상, 카메라 자세, 태양 방향 데이터를 통합하여 3D 비균질 소멸 계수 필드를 회복합니다. 새로운 두 단계의 훈련 방식이 도입되어 많은 자유도 문제를 해결하고, 고정된 태양 조명 방향의 제약을 극복했습니다.

- **Performance Highlights**: 이 접근 방식은 이전 최첨단 기술에 비해 상당한 성능 향상을 보여주었으며, 특히 태양의 천정 각도 변화 처리에 대한 개선을 입증했습니다.



### Explainable Search and Discovery of Visual Cultural Heritage Collections with Multimodal Large Language Models (https://arxiv.org/abs/2411.04663)
Comments:
          16 pages, CHR 2024: Computational Humanities Research Conference, December 4 - 6, 2024, Aarhus University, Denmark

- **What's New**: 문화 기관들이 대규모 디지털 비주얼 컬렉션을 온라인에 공개하고 있지만, 이를 탐색하고 검색하는 인터페이스를 만드는 것은 여전히 어려운 문제입니다. 본 논문에서는 최첨단의 멀티모달 대형 언어 모델 (multimodal large language models, LLMs)을 활용한 새로운 방법론을 제안합니다.

- **Technical Details**: 제안된 방법은 각 추천 항목에 대해 구체적인 텍스트 설명을 제공할 수 있으며, 미리 선택된 관심 특징 (features)이 필요하지 않습니다. 이는 기존의 비주얼 임베딩 (visual embeddings) 기반 방법의 일반적인 단점을 피하는 혁신적인 클러스터링 및 추천 시스템을 생성하는 데 도움을 줍니다.

- **Performance Highlights**: 사례 연구로 다큐멘터리 사진 컬렉션을 이용하여 우리의 접근 방식의 효율성과 가능성을 보여주는 여러 메트릭을 제공하였습니다. 이로 인해 더 개방적이고 유연한 디지털 인터페이스를 만들어 개인정보와 윤리적 문제를 더 잘 다룰 수 있게 됩니다.



### Automated Image Color Mapping for a Historic Photographic Collection (https://arxiv.org/abs/2411.04659)
Comments:
          11 pages, CHR 2024: Computational Humanities Research Conference, December 4 - 6, 2024, Aarhus University, Denmark

- **What's New**: 1970년대에 미국 환경 보호국이 후원했던 Documerica 프로젝트의 사진을 향상시키는 새로운 기술이 소개되었습니다.

- **Technical Details**: 손상된 이미지의 수정을 위해 원본 인쇄물의 화학적 기초를 활용한 수정된 histogram matching 기법이 사용되었습니다. 이 기법은 손상되지 않은 인쇄물로부터 수집된 소량의 훈련 데이터를 사용하여 개발되었습니다.

- **Performance Highlights**: 15,000개 이상의 디지털 공공 도메인 이미지가 온라인에 공개되며, 이 전체 색조 조정된 Documerica 이미지 세트는 오픈 저장소에서 제공됩니다.



### ICH-SCNet: Intracerebral Hemorrhage Segmentation and Prognosis Classification Network Using CLIP-guided SAM mechanism (https://arxiv.org/abs/2411.04656)
Comments:
          6 pages, 2 figures, 3 tables, published to BIBM 2024

- **What's New**: 이번 논문은 intracerebral hemorrhage (ICH)의 분할(segmentation)과 예후 분류(prognosis classification) 두 가지 작업을 동시에 처리하기 위해 설계된 다중 작업 네트워크인 ICH-SCNet을 소개합니다. 기존의 두 가지 작업을 독립적으로 다루던 접근 방식을 개선하여, 서로 관련성을 인식하고 결합하였습니다.

- **Technical Details**: ICH-SCNet은 SAM-CLIP 크로스 모달(interactions) 메커니즘을 활용하여 의료 텍스트 및 세분화(auxiliary segmentation) 보조 정보를 신경 이미징 데이터와 통합합니다. 또한, 효과적인 피처 융합(module) 및 다중 작업 손실 함수(multi-task loss function)를 개발하여 성능을 개선하고자 하였습니다. 이 모델은 특히 Vision Transformer (ViT)를 사용하여 4개의 스케일에서 이미지 피처를 생성하고, CLIP 이미지 및 텍스트 인코더를 통해 정보 처리를 진행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 ICH-SCNet은 기존의 최첨단 방법들을 초월하는 성능을 보였으며, 분류 작업의 전체 성능이 뛰어나고 모든 세분화(task metrics) 지표에서 경쟁 모델을 초과했습니다.



### DanceFusion: A Spatio-Temporal Skeleton Diffusion Transformer for Audio-Driven Dance Motion Reconstruction (https://arxiv.org/abs/2411.04646)
- **What's New**: 본 논문에서는 DanceFusion이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 음악에 동기화된 춤 동작을 재구성하고 생성하는 데 혁신적인 접근 방식을 채택하고 있습니다.

- **Technical Details**: DanceFusion은 Spatio-Temporal Skeleton Diffusion Transformer를 활용하여, TikTok과 같은 소셜 미디어 플랫폼에서 자주 발생하는 불완전하고 노이즈가 많은 골격 데이터를 효과적으로 처리합니다. 이 프레임워크는 계층적 Transformer 기반 Variational Autoencoder (VAE)를 확산 모델(diffusion model)과 통합하여 운동의 현실성과 정확도를 높입니다. 고급 마스킹 기법과 독창적인 반복 확산 과정이 도입되어 동작 시퀀스를 정제함으로써 동작 생성 및 오디오 큐와의 동기화에서 높은 충실도를 보장합니다.

- **Performance Highlights**: 포괄적인 평가 결과, DanceFusion은 기존 방법들을 초월하며 동적이고 사실적이며 스타일 다양성이 풍부한 춤 동작 생성에서 최첨단 성능을 발휘하고 있습니다. 이 프레임워크의 잠재적인 응용 분야는 콘텐츠 생성, 가상 현실(virtual reality), 인터랙티브 엔터테인먼트에 이르기까지 뻗어 있습니다.



### TAP-VL: Text Layout-Aware Pre-training for Enriched Vision-Language Models (https://arxiv.org/abs/2411.04642)
- **What's New**: 이 논문에서는 Vision-Language (VL) 모델의 OCR 정보 통합 방식을 개선하기 위해 TAP-VL이라는 새로운 방법을 제안합니다.

- **Technical Details**: TAP-VL은 OCR 정보를 별도의 모달리티로 취급하여 VL 모델에 통합하는 리프레시 접근 방식입니다. 이 과정에서 layout 정보를 포함하는 경량화된 transformer 기반 OCR 모듈을 활용하여 OCR 정보를 짧은 고정 길이 시퀀스로 압축하여 LLM에 입력합니다.

- **Performance Highlights**: TAP-VL은 다양한 선진 VL 모델에 적용하여 일관된 성능 향상을 보여주며, 특히 덴스 텍스트 이미지를 포함한 문서 이해 작업에서 4.8%의 성능 향상을 달성하였습니다. 또한, TAP-VL_Light은 FLOPs를 7배까지 줄이면서도 뛰어난 성능을 발휘합니다.



### Improved Multi-Task Brain Tumour Segmentation with Synthetic Data Augmentation (https://arxiv.org/abs/2411.04632)
- **What's New**: 이 논문은 BraTS 도전 과제의 과제 1에서 우승한 솔루션과 과제 3에서 3위를 차지한 솔루션을 제시합니다. 자동화 도구의 사용이 증가하고 있으나, 임상 표준을 달성하고 실제 시나리오를 위한 도구 개발은 큰 도전 과제입니다. 우리의 연구는 합성 데이터(synthetic data)를 사용하여 성인 교모세포종의 세분화(segmentation) 개선을 제안하며, 방사선 치료 계획을 위한 수막종(segmentation of meningioma)에서도 적용됩니다.

- **Technical Details**: 이 연구는 두 대의 기계를 사용하여 실험을 진행했습니다. 첫 번째 기계는 6개의 NVIDIA RTX 6000을 사용한 IKIM 클러스터 노드이고, 두 번째는 RWTH Aachen 클러스터의 NVIDIA H100 GPU를 사용했습니다. 데이터셋은 2200개의 샘플로 구성되며, 실험을 위해 70%는 훈련(training), 10%는 검증(validation), 20%는 테스트(test)용으로 사용됩니다. 두 개의 GliGAN을 훈련하여 합성 종양을 이미지에 삽입하는 방법을 사용했습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터를 사용하는 것이 알고리즘의 강건성을 증가시키는 것으로 나타났습니다. 특히, nnUNet와 Auto3dseg과 같은 도구를 통해 각각의 작업에서 강한 성능을 발휘했습니다. Task 1에서는 15,651개, Task 3에서는 7,470개의 케이스가 생성되어 모델의 강건성을 높이고 허위 음성(false negatives) 수를 줄이는 데 기여했습니다.



### Brain Tumour Removing and Missing Modality Generation using 3D WDM (https://arxiv.org/abs/2411.04630)
- **What's New**: 이번 논문은 BraTS 2024의 작업 8에 대해 2위에 해당하는 솔루션과 작업 7에 대한 참여 솔루션을 제시합니다. 자동화된 뇌 분석 알고리즘의 채택이 증가하고 있으나, 뇌 병변의 존재나 특정 MRI 모달리티의 결여가 예측 모델의 성능을 저하시킬 수 있음을 지적합니다.

- **Technical Details**: 본 연구에서는 conditional 3D wavelet diffusion models를 사용하여 성능을 개선하고자 하였으며, 이는 GPU에서의 전체 해상도 이미지 훈련 및 예측을 가능하게 합니다. 사용된 데이터셋은 BraTS 2023의 성인 교모세포종 분할 데이터셋을 포함하며, 1251 사례로 구성되어 있습니다.

- **Performance Highlights**: BraTS 2024의 inpainting 작업에 있어, 3D wavelet diffusion 모델의 사용은 검증 세트에서 MSE 0.007, PSNR 22.61, SSIM 0.842를 달성하였고, 테스트 세트에서는 MSE 0.070, PSNR 22.80, SSIM 0.91을 기록했습니다.



### Multi-temporal crack segmentation in concrete structure using deep learning approaches (https://arxiv.org/abs/2411.04620)
- **What's New**: 본 연구에서는 다중 시계열 데이터(multi-temporal data)를 활용하여 콘크리트 구조물의 균열(segment cracks) 검출 품질을 향상시키기 위한 방법을 제안합니다. 특히, Swin UNETR 모델과 U-Net 모델을 비교하여 시계열 정보가 세그멘테이션(segmentation) 품질에 미치는 영향을 분석하였습니다.

- **Technical Details**: 1356개의 이미지와 각 이미지에 따른 32개의 연속적인 균열 전파 이미지로 구성된 다중 시계열 데이터셋을 제작했습니다. 두 모델을 훈련한 후, 일반화 능력(generalization ability), 시계열 일관성(temporal consistency) 및 세그멘테이션 품질을 평가하였습니다.

- **Performance Highlights**: 다중 시계열 접근 방식이 단일 시계열 접근 방식에 비해 일관되게 우수한 성능을 나타냈습니다. 다중 시계열 모델은 IoU(Intersection over Union)가 82.72%로 단일 시계열 모델보다 5% 향상되었고, F1-score는 90.54%로 4.36% 향상되었습니다. 훈련 가능 매개변수는 절반으로 줄이면서도 세그멘테이션 품질이 더 일관되었고 소음(noise)과 오류가 줄어드는 결과를 보였습니다.



### Population estimation using 3D city modelling and Carto2S datasets -- A case study (https://arxiv.org/abs/2411.04612)
- **What's New**: Carto2S 시리즈 위성의 출시로 고해상도 이미지(0.6-1.0 미터)를 활용하여 정확한 디지털 고도 모델(Digital Elevation Model, DEM)을 생성할 수 있는 새로운 방법이 제시되었습니다.

- **Technical Details**: C2S 다중 시점(multi-view) 및 다중 날짜(multi-date) 데이터셋을 이용하여 고해상도 DEM과 디지털 지형 모델(Digital Terrain Model, DTM)을 생성하였습니다. 이 데이터셋을 통해 지표면의 객체(건물 및 나무)의 정확한 높이를 추출하고, 지상 제어점(ground control points)과 비교하여 검증했습니다.

- **Performance Highlights**: 위성 데이터를 기반으로 한 인구 추정 결과가 지역 관리자가 의사 결정을 내리는 데 매우 유용하다는 점이 강조되었으며, 3D 가상 도시 모델과 주택의 평면적 및 층수를 평가하는 데 사용되었습니다.



### Solar potential analysis over Indian cities using high-resolution satellite imagery and DEM (https://arxiv.org/abs/2411.04610)
- **What's New**: 이 연구는 고해상도 위성 이미지(0.5cm)와 디지털 고도 모델(1m), 그리고 지상 관측소 방사선 데이터를 활용하여 지붕의 태양광 잠재력을 추정하는 새로운 접근법을 제시합니다.

- **Technical Details**: 태양광 방사선 분석은 IMD에 의해 호스팅되는 지상 관측소 데이터에서 파생된 확산 비율(diffusion proportion)과 투과율(transmissivity ratio)을 사용하여 수행됩니다. 기존 연구에서는 도시 지역의 나무/식물 그림자, 인접한 높은 건축 구조물, 비정형 지붕 구조물이 고려되지 않았습니다.

- **Performance Highlights**: 계절적 변화 및 환경 효과로 인해 전기 생산이 최대 50%까지 손실될 수 있음이 관찰되었습니다. 1m DEM과 50cm 위성 이미지를 사용함으로써 도시 지역에서 더욱 신뢰할 수 있는 결과가 도출되었습니다.



### Cross- and Intra-image Prototypical Learning for Multi-label Disease Diagnosis and Interpretation (https://arxiv.org/abs/2411.04607)
- **What's New**: 최근 CIPL( Cross- and Intra-image Prototypical Learning) 프레임워크를 제시하여 다중 레이블 질병 진단과 해석을 위한 효율적인 방법론을 개발했습니다. 이 기술은 이미지 간의 공통 의미를 활용하여 여러 질병을 분리하고, 강력한 해석과 예측 성능을 보장합니다.

- **Technical Details**: CIPL 방법은 교차 이미지 의미를 활용하여 다중 질병의 복잡한 병리학적 병변을 이해하는 데 중점을 두며, 이 과정에서 두 가지 수준의 정규화 방식을 제안합니다. 이를 통해 모델 해석력을 향상시키고 예측 정확성을 높입니다.

- **Performance Highlights**: CIPL은 두 개의 공공 다중 레이블 벤치마크(흉부 방사선 사진 및 망막 이미지)에서 최첨단 분류 정확도를 달성했습니다. 특히, 약한 감독 하의 흉부 질병 위치 지정에서 다른 주요 설명 방법보다 뛰어난 성능을 나타냈습니다.



### Social EgoMesh Estimation (https://arxiv.org/abs/2411.04598)
- **What's New**: 이번 연구에서는 사회적 상호작용을 새로운 요소로 포함하여, 카메라 착용자의 3D 포즈를 보다 정확하게 추정할 수 있는 SEE-ME라는 혁신적인 프레임워크를 제안합니다.

- **Technical Details**: SEE-ME는 잠재적 확산 모델(latent probabilistic diffusion model)을 사용하여 착용자의 메쉬(mesh)를 추정합니다. 이 모델은 장면(scene) 및 사회적 상호작용(social wearer-interactee interactions)을 조건화(with condition)합니다. 또한, Variational Autoencoder (VAE)를 사용하여 사람의 포즈를 인코딩하고, 이를 바탕으로 조건화된 확산을 수행합니다.

- **Performance Highlights**: SEE-ME는 기존 기술보다 53% 낮은 포즈 추정 오차(MPJPE)를 달성하여 최고의 성능을 자랑합니다. 또한, 이 연구에서는 사회적 상호작용이 에고메쉬 추정에서 가장 큰 영향을 미치는 시기를 정량화합니다.



### The Impact of Semi-Supervised Learning on Line Segment Detection (https://arxiv.org/abs/2411.04596)
Comments:
          9 pages, 6 figures, 7 tables

- **What's New**: 이 논문에서는 준지도 학습(semi-supervised learning) 프레임워크에 기반한 이미지 내 선(segment) 탐지 방법을 제안합니다. 적은 수의 레이블링된 데이터와 다양한 증강(augmentation) 및 변형된 레이블 없는 이미지에 기반하여 일관성 손실(consistency loss)을 활용하여, 완전히 감독된 방법과 비교 가능한 결과를 보여줍니다.

- **Technical Details**: 제안된 방법은 준지도 학습을 사용하여 레이블이 없는 이미지에서 일관성 정보를 학습할 수 있으며, 특정 도메인에 적합하도록 경량화된 처리를 제공합니다. 본 연구는 기존의 빠르고 효율적인 모델을 기반으로 하며, 작은 데이터셋으로 학습하여 전통적인 감독 학습 방법보다 더 좋거나 동일한 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 레이블이 반만 존재하는 데이터셋을 사용할 때도 완전 감독 학습 방법보다 28%의 높은 성능을 기록하였습니다. 또한, 실시간 응용 프로그램에 적합한 소규모 모델이 준지도 학습 방법을 통해 상당한 성과를 보일 수 있음을 입증했습니다.



### On the Inherent Robustness of One-Stage Object Detection against Out-of-Distribution Data (https://arxiv.org/abs/2411.04586)
Comments:
          12 figures, 4 tables, under review

- **What's New**: 이 논문에서는 하나의 단계(object detection) 객체 탐지 모델이 일반화된 객체 인식에서 외부 배포(out-of-distribution, OoD) 데이터에서도 견고하게 작동할 수 있는 능력을 분석합니다. 특히, 사전 훈련된(pretrained) 모델을 활용한 새로운 알고리즘을 제안하여, 미리 훈련된 객체 탐지기를 재학습(retraining)할 필요 없이 알려지지 않은 객체를 탐지할 수 있는 방법을 개발하였습니다.

- **Technical Details**: 제안된 알고리즘은 사전 훈련된 모델의 특징(feature) 맵을 활용하여, 고해상도(feature maps)로 알려지지 않은 객체를 비지도 학습 방식으로 식별합니다. 이 알고리즘은 차원 축소(dimensionality reduction) 기법을 적용하여, 모델에서 추출된 특징에 대해 차원의 저주(curse of dimensionality)를 완화하는 방식으로 작동합니다. 우리의 실험에서는 알고리즘의 성능을 특정 알고리즘 구성 및 추론 신뢰성(thresholds)과의 파레토(Pareto) 절충점을 분석하여 평가했습니다.

- **Performance Highlights**: 우리의 방법은 로그 기반(logits-based) 후처리(post-hoc) OoD 방법과 성능을 비교하여 superior(우수한) 탐지 점수를 달성했습니다. 제안된 알고리즘은 최신의 OoD 접근 방식과 비교할 때 더 나은 성능을 보였으며, 사전 훈련된 모델의 정밀한 객체 탐지 능력을 향상시키는 데 기여할 것입니다.



### PASSION for Dermatology: Bridging the Diversity Gap with Pigmented Skin Images from Sub-Saharan Africa (https://arxiv.org/abs/2411.04584)
Comments:
          MICCAI 2024

- **What's New**: 아프리카의 피부과 전문의 부족 문제를 해결하기 위해 피부 질환 이미지를 수집하여 AI를 활용한 원격 피부 진료를 위한 첫 번째 오픈 소스 데이터셋인 PASSION 프로젝트를 소개합니다. 이 데이터셋은 1,653명의 환자를 위한 4,901개의 이미지를 포함하고 있으며, 아프리카 하부 지역의 다양한 피부 질환을 포괄하고 있습니다.

- **Technical Details**: PASSION 데이터셋은 2020년부터 2023년까지 마다가스카르, 기니, 말라위, 탄자니아에서 수집되었습니다. 수집된 이미지는 주로 핸드폰을 사용하여 원격 진료 환경에서 촬영되었으며, 피부 타입 \\Romannum4, \\Romannum5, \\Romannum6를 포함합니다. 데이터는 진단, 나이, 성별, 신체 부위 등의 정보를 포함하고 있습니다.

- **Performance Highlights**: 이 프로젝트는 PAEDIATRIC(소아) 환자 피부 질환에 초점을 맞추어 약 80%의 대표적인 피부 질환을 포함하고 있습니다. 기본 머신러닝 모델과 해당 데이터셋의 성능 분석도 제공하며, 이 데이터셋은 피부과 연구 및 AI 기반 원격 진료 시스템을 위한 귀중한 자원이 될 것으로 기대됩니다.



### DomainGallery: Few-shot Domain-driven Image Generation by Attribute-centric Finetuning (https://arxiv.org/abs/2411.04571)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 DomainGallery라는 새로운 메소드를 제안하고 있습니다. 이는 적은 수의 샘플(몇 장의 이미지)로 특정 도메인에서 이미지 생성을 가능하게 하는 방법으로, 사전 훈련된 Stable Diffusion 모델을 파인튜닝(finetuning)하는 기법입니다. 또한, 주목할 만한 특징은 속성 중심(attribute-centric) 접근 방식을 채택하여, 이전 연구들이 해결하지 못한 주요 문제들을 다루고자 합니다.

- **Technical Details**: DomainGallery는 다음과 같은 기술적 요소를 포함합니다: (1) Prior Attribute Erasure: 아이디 단어에 이전 속성을 지워주어 예기치 않은 이미지 요소를 방지합니다. (2) Attribute Disentanglement: 도메인과 카테고리 속성을 명확히 구분하여 교차 카테고리 이미지 생성 시 누수를 방지합니다. (3) Attribute Regularization: 파인튜닝 과정에서의 과적합(overfitting)을 줄이는 규제를 적용합니다. (4) Attribute Enhancement: 교차 카테고리 이미지를 생성할 때 도메인 속성의 강도를 조절하여 더 높은 충실도를 유지합니다.

- **Performance Highlights**: 여러 실험을 통해 DomainGallery가 다양한 도메인 중심의 이미지 생성 시나리오에서 우수한 성능을 발휘함을 보여주었습니다. 결과적으로 이 방법은 몇 샷(few-shot) 도메인 기반의 이미지 생성에서 최첨단의 효율성을 갖추고 있습니다.



### Neural Fingerprints for Adversarial Attack Detection (https://arxiv.org/abs/2411.04533)
Comments:
          14 pages

- **What's New**: 이 논문은 기존의 고정된 방어 메커니즘의 한계를 극복하기 위해 랜덤화를 도입한 새로운 접근 방식을 제안합니다. 특히, 여러 개의 별도 탐지 네트워크를 사용하여 입력에 대해 무작위로 선택함으로써 공격자가 탐지를 우회할 수 없도록 합니다.

- **Technical Details**: Neural Fingerprints 기법을 통해 각 클래스에 대해 특정 레이어에서 몇 개의 뉴런을 샘플링하여 탐지기를 생성합니다. 이 탐지기를 통해 상황에 맞는 ‘지문’을 활용하여 공격 여부를 판단하는 방식입니다. 테스트 시, 모델이 예측한 레이블에 연결된 지문을 샘플링하여 공격을 감지합니다.

- **Performance Highlights**: ImageNet 데이터셋을 사용한 실험에서 다양한 공격 방법에 대해 거의 완벽한 탐지 성능을 보였으며, 낮은 오탐지율을 기록하였습니다. 이는 실제 애플리케이션에서의 효과성과 확장 가능성을 입증합니다.



### l0-Regularized Sparse Coding-based Interpretable Network for Multi-Modal Image Fusion (https://arxiv.org/abs/2411.04519)
- **What's New**: 이번 연구에서는 다중 모달 이미지 융합(multi-modal image fusion, MMIF) 과제를 위한 해석 가능한 네트워크 FNet을 제안합니다. 이 네트워크는 ℓ0-정규화된 다중 모달 컨볼루션 희소 코딩(multi-modal convolutional sparse coding, MCSC) 모델을 기반으로 하며, 독특하고 공통적인 특징을 분리하여 융합 이미지를 생성합니다.

- **Technical Details**: FNet은 ℓ0-정규화된 희소 코딩 문제를 해결하기 위해 알고리즘 언롤링 기반의 ℓ0-정규화된 희소 코딩(LZSC) 블록을 개발했습니다. FNet은 다양한 모달 소스 이미지를 사용하여 먼저 특징을 분리하고, 그런 다음 이 특징들을 결합하여 최종 융합 이미지를 생성합니다. 또한, 역 융합 작업을 위한 ℓ0-정규화된 MCSC 모델을 제안하여 해석 가능한 역 융합 네트워크 IFNet을 개발하였습니다.

- **Performance Highlights**: FNet은 다섯 가지 MMIF 작업에서 높은 품질의 융합 결과를 도출했으며, 가시적-열적 이미지 쌍의 객체 탐지에서도 향상된 성능을 보였습니다. FNet의 중간 결과를 시각화하여 네트워크의 훌륭한 해석 가능성을 입증했습니다.



### FedDP: Privacy-preserving method based on federated learning for histopathology image segmentation (https://arxiv.org/abs/2411.04509)
Comments:
          Accepted in BIBM2024

- **What's New**: 이번 연구는 병원 간 협력적인 학습을 통해 환자 개인 정보 보호를 극대화하면서도 암 병리 이미지 데이터의 비밀성을 유지할 수 있는 연합 학습 프레임워크(Federated Learning)를 도입했습니다. 이 방법은 환자 정보 유출의 우려 없이 각 기관이 독립적으로 학습할 수 있도록 지원합니다.

- **Technical Details**: 제안된 모델은 Swin Transformer와 ConvNeXt를 기반으로 하는 이중 분기 계층 글로벌-로컬 융합 네트워크(DHUnet)를 사용하여 전체 슬라이드 이미지(WSI)의 세분화를 수행합니다. 각 기관의 로컬 모델이 훈련한 후, 변경된 모델 매개변수만 중앙 서버로 전송하여 글로벌 모델로 집계하고, 다시 로컬 장치로 분배되는 방식입니다. 차별적 프라이버시(Differential Privacy)를 통해 모델 업데이트에 노이즈를 추가하여 각 샘플의 기여도 유출을 방지합니다.

- **Performance Highlights**: FedDP 방법을 통해 모델의 Dice, Jaccard, Acc 지수가 각각 0.55%, 0.63%, 0.42%의 미미한 감소로 모델 정확성을 거의 유지하며 암 병리 이미지 데이터의 프라이버시를 효과적으로 보호할 수 있었습니다. 이는 의료 분야에서 민감한 데이터의 보호와 함께 기관 간 협업 및 지식 공유를 통한 새로운 연구 및 응용 가능성을 더욱 확장할 수 있음을 의미합니다.



### Pose2Trajectory: Using Transformers on Body Pose to Predict Tennis Player's Trajectory (https://arxiv.org/abs/2411.04501)
- **What's New**: 테니스 선수의 미래 궤적을 예측하는 새로운 방법인 Pose2Trajectory를 제안하였습니다. 이 방법은 선수의 몸 관절 데이터와 공 위치 정보를 기반으로 하여 자동으로 선수의 동작을 추적하고 예측할 수 있는 시스템입니다.

- **Technical Details**: Pose2Trajectory 시스템은 인코더-디코더 Transformer 아키텍처를 사용하여, 선수와 공의 관절 및 궤적 정보를 학습합니다. 예측된 궤적 시퀀스는 카메라가 테니스 선수를 자동으로 추적하는 데 도움을 줍니다. 이 시스템은 다수의 비디오에서 생성한 고품질 데이터셋을 기반으로 하여 동작 예측을 지원합니다.

- **Performance Highlights**: 우리의 방법은 다양한 시퀀스 예측 길이에 대해 테니스 선수의 운동 궤적을 예측하는 데 있어서 유망한 결과를 보였습니다. 특히, 선수의 관절 정보와 공 위치를 통해 더 높은 예측 정확도를 달성하였습니다.



### Synergy-Guided Regional Supervision of Pseudo Labels for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2411.04493)
- **What's New**: 이번 연구에서는 잡음 저감 문제를 해결하기 위해 새로운 Synergy-Guided Regional Supervision of Pseudo Labels (SGRS-Net) 프레임워크를 제안합니다. 이 프레임워크는 mean teacher network를 기반으로 하며, Mix Augmentation 모듈을 사용하여 레이블이 없는 데이터를 보강하며, 보강 전후 시너지를 평가하여 pseudo labels를 구분된 영역으로 나눕니다.

- **Technical Details**: SGRS-Net은 Pseudo Label Generation (PLG) 모듈과 Mix Augmentation (MA) 모듈을 통해 pseudo labels을 생성하고 강화하여 잡음의 영향을 최소화합니다. Synergy Evaluation 모듈은 pseudo labels을 서로 다른 영역으로 나눈 후, 각 영역에 맞는 손실 함수를 적용하여 손실을 평가합니다. 이러한 모듈들은 각각의 영역에서 pseudo labels를 최대한 활용하기 위해 설계되었습니다.

- **Performance Highlights**: LA 데이터셋에서 진행된 광범위한 실험 결과, SGRS-Net은 최신 기법들을 능가하는 성능을 보였습니다. 특히 레이블이 있는 데이터의 5%만 사용하여도 뛰어난 성능을 발휘하여, 제한된 레이블 데이터 상황에서 모델의 효과성을 강조합니다.



### CFPNet: Improving Lightweight ToF Depth Completion via Cross-zone Feature Propagation (https://arxiv.org/abs/2411.04480)
- **What's New**: 가벼운 ToF(시간비행, Time-of-Flight) 깊이 센서를 사용한 깊이 완성(depth completion)에서 기존 알고리즘의 한계를 극복하기 위해 CFPNet을 제안합니다. CFPNet은 두 가지 혁신적인 모듈, 즉 DAPM(Direct-Attention-based Propagation Module)과 LKPM(Large-Kernel-based Propagation Module)을 사용하여 구역(zone) 간 특징 전파를 효과적으로 수행합니다.

- **Technical Details**: 이 논문은 DAPM을 통해 구역 내 픽셀에서 구역 외 픽셀로 직접적인 특징 전파를 가능하게 하고, LKPM을 통해 31 크기의 대형 커널(convolution layer)로 장거리 의존성을 모델링합니다. 이러한 두 모듈의 결합으로 CFPNet은 ZJU-L5 데이터세트에서 SOTA(State-of-the-art) 깊이 완성 성능을 달성합니다.

- **Performance Highlights**: CFPNet은 기존 방법들과 비교하여 깊이 완성 성능에서 평균 절대 상대 오차(MEAN ABSOLUTE RELATIVE ERROR, REL)를 0.127에서 0.103으로 감소시켰습니다. ToF의 해상도가 2×2일 때 REL과 RMSE(Root Mean Square Error)를 각각 46.5%와 30.8% 감소시키는 성과를 기록하였습니다.



### Deep Learning Models for UAV-Assisted Bridge Inspection: A YOLO Benchmark Analysis (https://arxiv.org/abs/2411.04475)
- **What's New**: 본 연구에서는 UAV(무인 항공기)를 이용한 교량 시각 점검을 효율화하기 위한 딥 러닝 모델의 선택 과정에서 주요 기여를 합니다. 23개 모델을 벤치마킹하여 YOLO(You Only Look Once) 모델의 최신 변형(YOLOv5, YOLOv6, YOLOv7, YOLOv8)을 평가하였습니다.

- **Technical Details**: 연구에서는 COCO-Bridge-2021+ 데이터셋을 사용하여 YOLOv8n, YOLOv7tiny, YOLOv6m 및 YOLOv6m6 모델의 정확도 및 처리 속도를 비교하였습니다. 이 모델들은 각각 mAP@50 점수와 추론 시간에서 최적의 성능을 보여주었습니다.

- **Performance Highlights**: YOLOv8n은 mAP@50 점수 0.803, 추론 시간 5.3ms, YOLOv7tiny는 mAP@50 점수 0.837, 추론 시간 7.5ms, YOLOv6m은 mAP@50 점수 0.853, 추론 시간 14.06ms, YOLOv6m6은 mAP@50 점수 0.872, 추론 시간 39.33ms를 기록하였습니다. 이 결과는 UAV를 통한 교량 점검의 효율성과 신뢰성을 높입니다.



### FreeCap: Hybrid Calibration-Free Motion Capture in Open Environments (https://arxiv.org/abs/2411.04469)
- **What's New**: 이번 연구에서는 LiDAR와 확장 가능한 이동 카메라를 결합한 새로운 캘리브레이션 없는 다중 모드 방식인 FreeCap을 제안합니다. 이 시스템은 공공 환경에서 전 세계 여러 사람의 동작을 정밀하게 포착할 수 있도록 하는 유연성 있는 Motion Estimation 기능을 제공합니다.

- **Technical Details**: FreeCap 시스템은 Pose-aware Cross-sensor Matching 모듈과 Coarse-to-Fine Sensor-expandable Pose Optimizer를 포함하여 로컬-투-글로벌 포즈 추정 및 3D 인체 Keypoints 최적화를 지원합니다. RTMPose와 LiveHPS를 함께 활용하여 2D 및 3D Keypoints를 추정하고, 카메라 수를 확장하여 LiDAR Depth 정보로 예측 정확성을 높입니다.

- **Performance Highlights**: Human-M3 및 FreeMotion 데이터셋을 활용한 실험에서 FreeCap은 기존의 단일 모드 방법에 비해 현저한 성능 향상을 보여주며, 다양한 응용 분야에서 다중 인체 Motion Capture의 효율적인 해결책으로 자리매김합니다.



### Efficient single image non-uniformity correction algorithm (https://arxiv.org/abs/2411.04457)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2411.03615

- **What's New**: 이 논문에서는 비냉각 적외선 이미지에서 발생하는 비균일성(Normalization Unit, NU)을 교정하는 새로운 방법을 제안하고 있습니다. 고전적인 영화 비플리커링 알고리즘을 적용하여 이미지의 열(또는 선)에서 발생하는 시간 의존적인 노이즈를 정규화합니다.

- **Technical Details**: 제안된 방법은 각 프레임에서 열의 시퀀스를 대상으로 영화 비플리커링 알고리즘을 적용하여 작동합니다. 이 방법은 정적 이미지에서 작동하며, 등록(registration)이나 카메라 동작 보정(camere motion compensation)이 필요하지 않습니다. 장비 의존적 파라미터는 하나만 필요하며, 풍경 독립적입니다.

- **Performance Highlights**: 제안된 방법은 실시간이며, 픽셀당 단 두 개의 연산만 필요합니다. 보정 패턴이 필요하지 않으며, '유령 아티팩트(ghost artifacts)'를 생성하지 않습니다. 또한, 상태에서의 비약적인 성능 향상을 다른 최신 기술과 비교하여 입증합니다.



### BendVLM: Test-Time Debiasing of Vision-Language Embeddings (https://arxiv.org/abs/2411.04420)
- **What's New**: 이번 연구에서는 Bend-VLM을 제안하여, Vision-Language Model(VLM) 임베딩의 편향 제거(debiasing)를 위한 비선형(nonlinear) 방법을 소개합니다. 기존의 방법들은 일반적으로 고정된 변환을 사용하는데 반해, Bend-VLM은 각 고유 입력에 맞춘 flexible한 접근 방식을 제공합니다.

- **Technical Details**: Bend-VLM은 finetuning-free 방식으로, 온라인 쿼리에 대한 실시간 편향 제거를 가능하게 합니다. 입력 쿼리로부터 보호 속성에 대한 보강된 쿼리를 생성하고, 이를 바탕으로 임베딩 공간에서 속성과 관련된 방향을 찾아 단계를 거쳐 최종적으로 편향이 제거된 임베딩을 얻습니다.

- **Performance Highlights**: 실험 결과, Bend-VLM은 분류(classification), 검색(retrieval), 이미지 자막 생성(image captioning) 등 다양한 설정에서 비교된 기존 방법들보다 일관되게 뛰어난 성능을 보였습니다.



### Image Understanding Makes for A Good Tokenizer for Image Generation (https://arxiv.org/abs/2411.04406)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 연구는 이미지 이해(Understanding, IU) 모델이 이미지 생성(Generation, IG) 작업의 성능을 어떻게 향상시킬 수 있는지를 탐구하는 새로운 접근에 대해 다룹니다. 이는 토큰 기반의 IG 프레임워크를 이용하여, 사전 훈련된 IU 인코더로부터 지식을 추출하여 토크나이저를 훈련하는 방식으로 진행됩니다.

- **Technical Details**: 기존의 IG 기술들, 특히 VQGAN 모두 픽셀 복원을 훈련 목표로 삼고 있지만, 본 연구에서는 기능 복원(objective)인 VQ-KD 접근법을 채택합니다. 이를 통해 IU 인코더에서의 다양한 표현 학습을 활용하여 IG 성능을 향상시킵니다. 다양한 메트릭을 통해 성능을 평가하며, Fréchet Inception Distance (FID), Inception Score (IS), perplexity (PPL) 등을 포함합니다.

- **Performance Highlights**: VQ-KD CLIP은 ImageNet-1k에서 $4.10$ FID를 달성하였으며, 이는 기존의 VQGAN의 $15.78$ FID에 비해 뛰어난 결과입니다. 연구에서는 VQ-KD가 더 많은 의미 맥락을 담고 있어 IG 품질을 개선하는 데 기여한다고 파악하였습니다. 또한, IU 능력이 약한 토크나이저는 더 큰 제안 네트워크를 필요로 하며, 훈련 이미지의 변동에 대해 덜 강한 내성을 보인다고 밝혀졌습니다.



### ProGraph: Temporally-alignable Probability Guided Graph Topological Modeling for 3D Human Reconstruction (https://arxiv.org/abs/2411.04399)
- **What's New**: 본 논문에서는 3D 인간 신체 복원에서의 중요한 기술적 과제를 해결하기 위해 Temporally-alignable Probability Guided Graph Topological Modeling(담당 모델, ProGraph)를 제안합니다. 이 방법은 인간 신체 복원의 불완전한 특징을 바탕으로 한 현실적인 3D 인간 메쉬 시퀀스를 추정하는데 도움을 줍니다.

- **Technical Details**: ProGraph는 Graph Topological Modeling(GTM)을 통해 신체 메쉬의 정점 간의 위상 연결을 명시적으로 학습하고, Temporally-alignable Probability Distribution(TPDist)을 사용하여 인접 프레임 사이에서 3D 기하학적 구조의 확률 분포를 정렬합니다. 이 프레임워크는 Hierarchical Human Loss(HHLoss)를 통하여 각 신체 부위의 확률 분포 오류를 계층적으로 계산하여 3D 메쉬 시퀀스의 일관성을 유지합니다.

- **Performance Highlights**: ProGraph는 3DPW 데이터셋을 활용한 실험에서 다른 최신 방법들(SOTA)과 비교하여 폐색 및 모션 블러가 있는 어려운 시나리오에서 우세한 결과를 보여주었습니다. 특히, 프레임 간 특징 의존성을 활용하여 다양한 상황에서도 장기적인 운동 특징 정보를 효과적으로 포착합니다.



### MegaPortrait: Revisiting Diffusion Control for High-fidelity Portrait Generation (https://arxiv.org/abs/2411.04357)
Comments:
          Technical Report

- **What's New**: MegaPortrait는 개인화된 초상 이미지를 생성하기 위한 혁신적인 시스템으로, Identity Net, Shading Net, Harmonization Net의 세 가지 모듈로 구성되어 있습니다. 이 시스템은 기존의 AI 초상화 제품보다 개인의 정체성을 보존하고 이미지의 품질을 높이는 데 우수한 결과를 제공합니다.

- **Technical Details**: MegaPortrait는 분할-병합( split-and-merge) 파이프라인을 통해 음영(shading)과 색상(color) 정보를 기하학(geometry)으로부터 분리하여, 각 정체성을 중심으로 다양한 스타일을 반영할 수 있도록 설계되었습니다. 이를 통해 고화질의 개인화된 초상 이미지를 생성합니다.

- **Performance Highlights**: MegaPortrait의 총체적인 파이프라인은 Identity Net, Shading Net, Harmonization Net으로 구성되어 있으며, 최신 연구 및 제품과 비교하여 정량적, 정성적으로 우수한 성능을 발휘합니다.



### LidaRefer: Outdoor 3D Visual Grounding for Autonomous Driving with Transformers (https://arxiv.org/abs/2411.04351)
Comments:
          16 pages, 5 figures

- **What's New**: 이번 연구에서는 LidaRefer라는 새로운 transformer 기반의 3D visual grounding (VG) 프레임워크를 소개합니다. 이는 대규모 야외 환경에서의 객체 및 지역 식별을 위해 설계되었으며, 자연어 설명에 따라 관련 객체를 정확하게 위치시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: LidaRefer는 3D 공간에서의 시각적 정보를 효과적으로 처리할 수 있는 모듈형 네트워크로 구성되어 있으며, LiDAR 포인트 클라우드를 활용합니다. 훈련 과정에서, LidaRefer는 타겟 객체를 식별하는 것 외에도 유사한 속성을 가진 모호한 객체를 구분하기 위해 'ambiguous object localization' 방법을 도입하여, 같은 장면에서의 공간적 관계를 학습하게 됩니다.

- **Performance Highlights**: LidaRefer는 자율 주행을 위한 Talk2Car-3D 데이터셋에서 최첨단(상태-최고) 성능을 기록하였으며, 다양한 평가 설정에서 유의미한 향상을 보여주었습니다.



### UEVAVD: A Dataset for Developing UAV's Eye View Active Object Detection (https://arxiv.org/abs/2411.04348)
- **What's New**: 본 논문은 UAV(무인 항공기) 기반의 객체 탐지에서 발생하는 occlusion 문제를 해결하기 위해 새로운 UAV 눈높이의 Active Vision Dataset(UEVAVD)을 공개하였습니다. 이는 UAV가 자율적으로 방향을 조정하여 최적의 관측을 생성하여 탐지 성능을 향상시킬 수 있도록 설계되었습니다.

- **Technical Details**: Active Object Detection (AOD) 기술을 통해 UAV의 경로 계획을 Deep Reinforcement Learning (DRL)으로 최적화합니다. 본 논문에서는 Gated Recurrent Unit (GRU)를 활용하여 상태 표현을 개선하고, Segment Anything Model (SAM)을 이용하여 장면을 사전 분해하여 불필요한 정보를 필터링합니다.

- **Performance Highlights**: 제안된 방법의 효과는 UEVAVD 데이터셋에서의 실험을 통해 검증되었습니다. 결과적으로 제안된 AOD 기법은 occlusion 문제를 해결하고 탐지 성능을 향상시키는 것으로 나타났습니다.



### GazeGen: Gaze-Driven User Interaction for Visual Content Generation (https://arxiv.org/abs/2411.04335)
Comments:
          13 pages, 10 figures

- **What's New**: GazeGen은 사용자 눈의 시선(gaze)을 이용하여 시각적 콘텐츠(이미지 및 비디오)를 생성하는 혁신적인 상호작용 시스템이다. 이 시스템은 사용자의 시선에 따라 시각적 콘텐츠를 직관적으로 조작할 수 있다.

- **Technical Details**: GazeGen의 핵심 요소는 DFT Gaze(Distilled and Fine-Tuned Gaze) 에이전트로, 281K 매개변수(paramenters)로 구성된 초경량 모델이다. DFT Gaze 모델은 개인의 눈에 맞춘 정확한 실시간 시선 예측을 수행하며, 이는 소형 엣지 디바이스에서 가능하다. 시선 예측 효율성을 위해, 대형 모델(10배 큰 모델)에서 새로운 지식 증류(knowledge distillation) 및 개인 맞춤화 기술을 통해 소형 모델을 개발하였다. 이 모델은 Adapters를 통해 추가 조정되어 시선 예측의 개인화를 가능하게 한다.

- **Performance Highlights**: DFT Gaze는 AEA 및 OpenEDS2020 벤치마크에서 낮은 각도 시선 오류 및 낮은 지연 시간으로 검증되었다. GazeGen은 사용자가 보는 시점을 명령으로 사용하여 동적인 이미지 편집 및 비디오 생성 등 다양한 시각적 콘텐츠 생성 작업을 가능하게 한다.



### HandCraft: Anatomically Correct Restoration of Malformed Hands in Diffusion Generated Images (https://arxiv.org/abs/2411.04332)
Comments:
          Accepted by WACV 2025

- **What's New**: 이번 논문에서는 Generative text-to-image 모델, 특히 Stable Diffusion과 같은 모델에서 인간의 손을 복원하기 위한 새로운 방법인 HandCraft를 제안합니다. 기존 모델은 인간 손의 해부학적으로 부정확한 이미지를 생성하는 데 어려움을 겪고 있으며, 이로 인해 예술적 도구로서의 활용에 제약이 있었습니다.

- **Technical Details**: HandCraft는 손의 모양을 조건 이미지(conditioning image)로 생성하여 ControlNet을 통해 부정확한 손의 형태를 수정하는 end-to-end 프레임워크입니다. 이 연산은 미세 조정이나 추가 교육 없이 기존의 pretrained diffusion 모델과 호환됩니다. 손 검출, 조건 신호 생성, 손 복원이라는 세 단계로 구성됩니다.

- **Performance Highlights**: HandCraft는 MalHand 데이터셋에서 인간 손의 해부학적 정확성을 복원하면서도 전체 이미지의 무결성을 유지하는 데 성공하였으며, qualitative 및 quantitative 평가에서 최첨단 성능을 달성했습니다. 이 방법은 예술성 있는 이미지 생성에도 기여하며, 다양한 손의 스타일을 포함한 MalHand 데이터셋을 제공합니다.



### Increasing the scalability of graph convolution for FPGA-implemented event-based vision (https://arxiv.org/abs/2411.04269)
Comments:
          Accepted for the PhD forum during FPT 2024 (International Conference on Field Programmable Technology), 10-12 December 2024, Sydney, Australia

- **What's New**: 이 논문은 이벤트 카메라에서 수집된 데이터를 처리하기 위해 그래프 컨볼루션 신경망(Graph Convolutional Neural Networks, GCNNs)을 FPGA 플랫폼에서 최적화하는 방법을 제안하고 있습니다. 독자들은 추가적인 BRAM 버퍼를 활용하여 LUT 사용량을 최대 94%까지 줄일 수 있는 '2단계 컨볼루션(two-step convolution)' 접근 방식을 소개받게 됩니다.

- **Technical Details**: 이 연구에서는 기존의 FPGA 리소스(BlockRAM, DSP, LUT)를 활용하여 그래프 컨볼루션 모듈의 최적화를 시도합니다. 연구팀은 지나치게 중복된 연산을 줄여 GCNN의 확장성을 크게 향상시키고, 더 많은 레이어와 더 큰 그래프를 적용할 수 있도록 하고 있습니다. 특히 추가 BRAM 버퍼를 이용한 방법론을 제안하며, 이는 다양한 FPGA 자원 사용의 유연성을 제공합니다.

- **Performance Highlights**: 제안된 방법은 FPFA에서 메모리 사용과 성능을 극대화하여, GCNN 모델의 효율성을 증가시키고 더 동적인 시나리오에서의 적용 가능성을 높였습니다. 실험 결과, 제안된 시스템은 낮은 대기 시간과 높은 처리량을 유지하면서도 뛰어난 정확도를 제공합니다.



### Pose-Transformation and Radial Distance Clustering for Unsupervised Person Re-identification (https://arxiv.org/abs/2411.04255)
- **What's New**: 이 논문에서는 사람 재식별(person re-identification, re-ID) 문제를 다루기 위해 새로운 비지도 학습 접근법을 제안합니다. 특히, 두 단계의 훈련 전략을 통해 학습 특징의 구별 능력을 향상시키고, 첫 번째 단계에서는 여러 변형을 만들어 포즈-transformed 데이터셋을 훈련합니다.

- **Technical Details**: 첫 단계에서, 각 이미지를 사용해 포즈 공간에서 여러 변화를 생성하여 전문가가 설계한 데이터셋을 만듭니다. 그 후, 제안된 discriminative clustering 알고리즘을 통해 유사한 특징을 가까이 맵핑하는 방법을 학습합니다. 새로운 radial distance loss를 도입하여 특징 학습의 기본 요소인 클러스터 내의 compact한 특성과 클러스터 간의 높은 변동성을 유지합니다.

- **Performance Highlights**: Market-1501, DukeMTMC-reID, MSMT17과 같은 여러 대규모 re-ID 데이터셋에서 제안한 방법이 최신의 접근 방식보다 우수한 성능을 보였습니다.



### PocoLoco: A Point Cloud Diffusion Model of Human Shape in Loose Clothing (https://arxiv.org/abs/2411.04249)
Comments:
          WACV 2025

- **What's New**: 본 연구에서는 PocoLoco를 소개합니다. 이는 템플릿 없이, 포즈에 따라 조건화된 최초의 점 기반 생성 모델로, 느슨한 옷을 입은 3D 인간 아바타를 모델링할 수 있습니다.

- **Technical Details**: PocoLoco는 파라메트릭 모델이나 의류 템플릿을 필요로 하지 않으며, 무작위 순서의 점 구름을 직접 처리하여 아바타의 의류 변형을 조건부 점 구름 생성 작업으로 설정합니다. 이 연구는 딥러닝의 노이즈 제거 확산 모델을 사용하여 의류 변형의 불확실성을 학습합니다.

- **Performance Highlights**: 기존의 느슨한 옷을 입은 인간 아바타에 대한 데이터셋은 작기 때문에, 75K 포인트 구름을 포함한 두 개의 피사체에 대한 데이터셋을 공개했습니다. 기존의 방법들과 비교하여, 느슨한 의류를 모델링하는 데 있어 뛰어난 성능 개선을 달성했습니다.



### PMPNet: Pixel Movement Prediction Network for Monocular Depth Estimation in Dynamic Scenes (https://arxiv.org/abs/2411.04227)
- **What's New**: 본 논문에서는 동적 장면에서의 단안심도 추정(monocular depth estimation)을 위한 새로운 방법을 제안합니다. 객체의 움직임 경로의 임의성을 이론적으로 탐구한 후, 짧은 거리에서 점들이 직선으로 이동한다고 가정하여 이를 2차원 유클리드 공간에서 삼각형 제약 손실(triangular constraint loss)로 요약합니다.

- **Technical Details**: 이를 통해 가장자리 주변의 깊이 불일치 문제를 해결하기 위해 다양한 형태의 객체에서 특징을 학습하는 변형 가능한 지원 윈도우 모듈(deformable support window module)을 제안하여, 가장자리 지역에서 깊이 값을 보다 정확하게 만듭니다.

- **Performance Highlights**: 제안된 모델은 KITTI와 Make3D와 같은 두 개의 야외 데이터셋과 NYU Depth V2라는 실내 데이터셋에서 훈련 및 테스트되었습니다. 이러한 데이터셋에서 보고된 정량적 및 정성적 결과는 다른 접근 방법들과 비교했을 때 제안된 모델의 성공을 보여줍니다. KITTI 데이터셋에서의 중요도 분석(ablation study) 결과는 제안된 픽셀 이동 예측 모듈(pixel movement prediction module)과 변형 가능한 지원 윈도우 모듈의 효과성을 검증합니다.



### WiFlexFormer: Efficient WiFi-Based Person-Centric Sensing (https://arxiv.org/abs/2411.04224)
- **What's New**: WiFlexFormer는 WiFi 기반의 사람 중심 센싱을 위해 설계된 고효율 Transformer 기반 아키텍처입니다. 이 아키텍처는 기존의 CNN 기반 접근법과 비교하여 낮은 파라미터 수와 빠른 추론 시간으로 유사한 Human Activity Recognition (HAR) 성능을 달성합니다.

- **Technical Details**: WiFlexFormer는 WiFi의 Channel State Information (CSI)를 기반으로 하며, 2D 및 1D stem 모듈을 사용하여 다양한 입력 특징을 처리합니다. 입력 특징은 일반적으로 실수 값이며, 배치, 채널, 주파수, 시간 차원으로 구성됩니다. Transformer 인코더를 통해 시퀀스의 관련 부분에 주의를 기울이는 방식으로 작동합니다.

- **Performance Highlights**: Nvidia Jetson Orin Nano에서 10ms의 추론 시간으로 실시간 추론에 최적화되어 있으며, WiFlexFormer는 CS 기반의 다양한 벤치마크에서 기존 최첨단 아키텍쳐와 비교하여 더 낮은 파라미터 수로 우수한 성능을 보여줍니다.



### DiMSUM: Diffusion Mamba -- A Scalable and Unified Spatial-Frequency Method for Image Generation (https://arxiv.org/abs/2411.04168)
Comments:
          Accepted to NeurIPS 2024. Project page: this https URL

- **What's New**: 본 논문에서는 이미지 생성 작업을 위해 확산 모델의 새로운 state-space 아키텍처를 소개합니다. 이 아키텍처는 spatial (공간적) 및 frequency (주파수) 정보를 효과적으로 활용하여 입력 이미지의 지역 특성에 대한 귀납적 편향을 향상시킵니다.

- **Technical Details**: 우리는 Mamba를 기반으로 한 확산 모델을 향상시키며, 주파수 스캔을 기존의 공간 스캔 메커니즘과 통합하는 혁신적인 방법을 제안합니다. 이를 통해 지역 구성 요소의 감수성과 장거리 의존성을 강화하며, cross-attention (크로스 어텐션) 방식으로 두 정보를 동적으로 결합합니다.

- **Performance Highlights**: DiMSUM 아키텍처는 ImageNet, CelebA-HQ 및 LSUN Church와 같은 이미지 생성 기준 벤치마크에서 기존 모델들보다 우수한 FID 점수와 recall(리콜)을 기록하였으며, 더 빠른 훈련 수렴 속도를 보여줍니다.



### UnityGraph: Unified Learning of Spatio-temporal features for Multi-person Motion Prediction (https://arxiv.org/abs/2411.04151)
Comments:
          13pages, 12 figures. arXiv admin note: text overlap with arXiv:2411.03729

- **What's New**: 본 논문에서는 UnityGraph라는 새로운 그래프 구조를 제안하여 다중 인물의 모션 예측을 수행합니다. 이 구조는 공간-시간(spatio-temporal) 정보를 전체적으로 통합하여 모델의 일관성과 결합성을 높입니다.

- **Technical Details**: UnityGraph는 하이퍼변량 그래프(hypervariate graph) 기반의 네트워크로, 관측된 모션을 그래프 노드로 간주하고, 하이퍼엣지(hyperedges)를 통해 이들을 연결하여 공간-시간 특성을 탐색합니다. 이 접근 방식은 모션 예측 문제를 단일 그래프 문제로 재구성합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 광범위한 실험을 통해, 우리의 방법이 최신 기술의 성능을 초월함을 보여줍니다. 특히, 3DPW, CMU-Mocap, MuPoTS-3D와 같은 여러 데이터셋에서 우수한 성능을 기록하였습니다.



### Stem-OB: Generalizable Visual Imitation Learning with Stem-Like Convergent Observation through Diffusion Inversion (https://arxiv.org/abs/2411.04919)
Comments:
          Arxiv preprint version

- **What's New**: 이번 논문에서는 시각적 모방 학습(Visual Imitation Learning)의 일반화 문제를 해결하기 위해, pre-trained 이미지 확산 모델을 이용한 새로운 방법, Stem-OB를 제안합니다. 이 방법은 저수준 시각적 차이를 억제하면서도 고수준 장면 구조를 유지합니다.

- **Technical Details**: Stem-OB는 이미지 뒤집기(image inversion) 과정을 통해 관찰을 공유 표현으로 변환하여 불필요한 세부 정보를 제거하며, 다양한 불특정 시각적 변화에 강인합니다. 이는 Gaussian 노이즈로 데이터를 단순히 교란하는 접근법과 달리, 각 장면의 정보 손실을 최소화합니다.

- **Performance Highlights**: 실험 결과, Stem-OB는 시뮬레이션 과제와 실제 세계 과제에서 모두 효과적이며, 다른 비교 방법들에 비해 22.2%의 성공률 향상을 기록했습니다. 또한, Stem-OB는 추가적인 추론 시간 없음으로 인해 실질적인 계산 비용이 없어 배치가 용이합니다.



### Differentiable Gaussian Representation for Incomplete CT Reconstruction (https://arxiv.org/abs/2411.04844)
- **What's New**: 이 논문에서는 신경망(neural network)이나 전체 용량 CT 데이터 없이 불완전 컴퓨터 단층촬영(Incomplete CT) 재구성을 위한 새로운 가우시안 표현(Gaussian Representation for Incomplete CT Reconstruction, GRCT)을 제안합니다.

- **Technical Details**: 연구팀은 3D 볼륨을 학습 가능한 가우시안 집합(set of learnable Gaussians)으로 모델링하고, 불완전 시노그램(incomplete sinogram)에서 직접 최적화하여 3D 볼륨을 재구성합니다. 또한, 효율적인 임상 사용을 위한 미분 가능한 빠른 CT 재구성(differentiable Fast CT Reconstruction) 방법도 제공합니다.

- **Performance Highlights**: 다양한 데이터 세트와 설정에서 실험을 수행하여 재구성 품질 지표(reconstruction quality metrics)와 효율성에서 상당한 개선을 입증합니다.



### MPVO: Motion-Prior based Visual Odometry for PointGoal Navigation (https://arxiv.org/abs/2411.04796)
Comments:
          Accepted in 50SFM Workshop of the 18th European Conference on Computer Vision (ECCV) 2024

- **What's New**: 본 논문에서는 성능 저하 문제를 해결하기 위해 모션 프라이어(motion priors)를 활용한 견고하고 샘플 효율적인 비주얼 오도메트리(Visual Odometry, VO) 파이프라인을 제안합니다. 기존의 VO 방법들이 가지는 한계를 극복하고, 깊이 학습(deep learned) VO 방법들이 느끼는 샘플 비효율성 문제를 해결합니다.

- **Technical Details**: 제안된 VO 파이프라인은 훈련이 필요 없는 액션 프라이어(action-prior) 기반의 기하학적 VO 모듈로 구성됩니다. 이 모듈은 대략적인 상대 포즈(coarse relative pose)를 추정하고, 이를 깊이 학습된 VO 모델에 입력하여 정밀한 상대 포즈(fine relative pose)를 생성합니다.

- **Performance Highlights**: 제안된 접근 방식은 훈련 중 샘플 효율성을 최대 2배 향상시키고, 최신 기술(state-of-the-art) VO 방법들에 비해 우수한 정확도와 견고성을 보여줍니다. 제안된 방법은 AI-Habitat 시뮬레이터에서 기빙슨 데이터셋(Gibson dataset)을 사용하여 평가되었으며, 성공률(success) 및 전체 경로 길이(success/SPL)와 같은 탐색 메트릭을 통해 입증되었습니다.



### An Effective Pipeline for Whole-Slide Image Glomerulus Segmentation (https://arxiv.org/abs/2411.04782)
- **What's New**: 이 연구에서는 신장 질환의 정확한 진단을 위한 glomerulus(사구체) 세분화를 위한 효과적인 파이프라인을 제안합니다. 이 방법은 패치 이미지 경계 근처에 위치한 사구체 검출 범위를 늘리기 위해 겹치는 패치에서 스티칭(stitching) 기법을 활용합니다.

- **Technical Details**: 제안된 파이프라인은 원활한 WSI(glomerulus 종합 슬라이드 이미지) 세분화를 위해 간단한 슬라이딩 윈도우를 적용하여 겹치는 패치를 추출하고, 이를 기반으로 한 스티칭 전략을 사용하여 원시 예측을 전체 WSI 수준의 예측 맵으로 결합합니다. 이 과정에서 softmax 함수를 사용하여 최종 예측 맵을 정규화합니다.

- **Performance Highlights**: 제안된 파이프라인을 사용하는 모델은 두 개의 대규모 데이터셋에서 이전의 최신 기술을 초월하여 우수한 결과를 나타내었으며, kidney pathology image segmentation challenge에서 1위를 차지했습니다.



### Convolutional Differentiable Logic Gate Networks (https://arxiv.org/abs/2411.04732)
Comments:
          Published at NeurIPS 2024 (Oral)

- **What's New**: 본 연구에서는 differentiable logic gate networks (LGNs) 의 효율성을 높이는 새로운 접근 방식인 deep logic gate tree convolutions를 제안합니다. 이는 기존의 LGNs에서 Spatial 관계를 학습할 수 있는 한계를 극복하여 CIFAR-10 데이터셋에서 86.29%의 정확도를 달성했습니다.

- **Technical Details**: 제안된 모델은 deep logic gate tree convolutions를 사용하여 convolutional 방식으로 구조를 확장하고, logical OR pooling을 통해 효과성을 향상시킵니다. 또한, residual initializations를 도입하여 네트워크의 깊이를 늘릴 수 있도록 합니다. 이러한 방법을 통해 61 백만 개의 논리 게이트로도 29배 더 작은 크기로 SOTA를 초과하는 성능을 보입니다.

- **Performance Highlights**: CIFAR-10에서 제안된 모델은 61 백만 개의 논리 게이트만으로 86.29%의 정확도를 달성하고, 이는 기존의 SOTA 성능과 비교해 29배 작으면서도 더 뛰어난 결과를 보입니다.



### Exploring the Stability Gap in Continual Learning: The Role of the Classification Head (https://arxiv.org/abs/2411.04723)
Comments:
          Accepted at WACV 2025

- **What's New**: 본 논문은 Neural Network의 Continual Learning (CL)에서 Stability Gap 현상을 조사하며, Classification Head의 역할을 중점적으로 다룹니다. 이를 통해 Nearest-Mean Classifier (NMC)를 소개하여, Backbone과 Classification Head가 Stability Gap에 미치는 영향을 분석합니다.

- **Technical Details**: Stability Gap은 Neural Network가 새로운 작업을 학습하는 과정에서 이전에 학습한 작업에 대한 성능이 감소하고 나중에 일부 성능이 회복되는 현상입니다. NMC는 클래스 프로토타입에 대한 샘플의 근접성을 기반으로 분류하며, 이는 기존의 Linear Head와 비교하여 더 나은 안정성을 제공합니다.

- **Performance Highlights**: NMC는 CIFAR100, ImageNet100, CUB-200, FGVC Aircrafts와 같은 다양한 Continual Learning 벤치마크에서 교육의 안정성을 크게 향상시켰습니다. 잔여 작업 선호 편향(task-recency bias)도 크게 줄어들었습니다.



### Subspace-Constrained Quadratic Matrix Factorization: Algorithm and Applications (https://arxiv.org/abs/2411.04717)
- **What's New**: 본 논문은 저차원 구조를 모델링하기 위한 새로운 subtspace-constrained quadratic matrix factorization 모델을 제시합니다. 이 모델은 접선 공간(tangent space), 법선 서브스페이스(normal subspace), 그리고 이들을 연결하는 quadratic form을 동시에 학습하는 것을 목표로 합니다.

- **Technical Details**: 이 모델은 alternating minimization 방법을 사용하여 해결되며, 비선형 회귀(nonlinear regression) 및 투영 서브문제(projection subproblems)의 심층적인 조사도 포함됩니다. 이론적 성질(theoretical properties)과 수렴 특성(convergence characteristics)도 논의됩니다.

- **Performance Highlights**: 실험 결과, 본 모델은 기존 방식들보다 우수한 성능을 보여주며, 저차원 구조(low-dimensional structures)를 포착하는 데 있어 강력함과 효율성을 강조합니다.



### CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation (https://arxiv.org/abs/2411.04679)
Comments:
          Under review

- **What's New**: 본 연구는 대형 언어 모델(LLM) 기반의 임몸체 에이전트들 간의 협력 문제를 다루며, 복잡한 작업에서의 효과적인 전략적 협력을 위한 새로운 프레임워크인 Cooperative Plan Optimization (CaPo)를 제안합니다.

- **Technical Details**: CaPo는 두 단계로 이루어져 있습니다: 1) 메타 계획 생성(meta-plan generation) 단계에서는 에이전트들이 작업을 분석하고 공동으로 메타 계획을 수립하여 태스크를 세부 작업으로 분해하고, 2) 진행 상황에 따라 메타 계획(progress-adaptive meta-plan) 및 실행 단계에서는 에이전트들이 최신 진행 상황에 따라 메타 계획을 동적으로 조정합니다.

- **Performance Highlights**: 세 가지 주요 실험에서 CaPo는 최신 기술(State-of-the-art) 방법들과 비교하여 태스크 완료율과 효율성 면에서 크게 향상되었습니다. 예를 들어, ThreeDworld Multi-Agent Transport 작업에서 CaPo는 GPT-3.5와 GPT-4 기반 에이전트를 사용할 때 각각 16.7% 및 4.7% 향상된 완료율을 기록했습니다.



### TexLiverNet: Leveraging Medical Knowledge and Spatial-Frequency Perception for Enhanced Liver Tumor Segmentation (https://arxiv.org/abs/2411.04595)
- **What's New**: 새로운 연구에서는 간 종양 세분화(liver tumor segmentation)를 위한 조합 텍스트 데이터와 이미지를 통합하는 것의 필요성을 강조하고, 병변-specific한 텍스트 주석이 있는 데이터셋을 새롭게 개발하였으며, TexLiverNet 모델을 소개합니다.

- **Technical Details**: TexLiverNet은 텍스트 기능과 시각적 기능을 효율적으로 통합하기 위해 agent-based cross-attention 모듈을 활용하며, 계산 비용을 크게 줄입니다. spatial 및 adaptive frequency domain 인식을 통해 병변 경계를 정확히 구분하고, 배경 간섭을 줄이며, 작은 병변의 세부사항을 복구합니다. 이 모델은 두 개의 주요 구성 요소인 TIA-attention과 SFPM으로 이루어져 있습니다.

- **Performance Highlights**: TexLiverNet은 공공 및 개인 데이터셋에 대한 포괄적인 평가에서 현재의 최첨단 방법들보다 우수한 성능을 달성했습니다.



### Verification of Neural Networks against Convolutional Perturbations via Parameterised Kernels (https://arxiv.org/abs/2411.04594)
- **What's New**: 본 논문은 블러링(blurring)이나 샤프닝(sharpening)과 같은 컨볼루션(perturbations) 변형에 대해 신경망을 효율적으로 검증하는 방법을 제안합니다. 이는 특정-camera shake와 같은 변형을考慮하고, 이들을 선형적으로 매개변수화하여 적용할 수 있도록 합니다.

- **Technical Details**: 제안된 방법에서는 입력 이미지에 대해 데이터를 선형 매개변수화한 커널(kernel)을 사용하여 컨볼루션(convolution) 연산을 수행합니다. 이 방식은 신경망의 입력 변형을 효율적으로 인코딩할 수 있게 하며, 이를 통해 좀 더 견고한 검증이 가능합니다. 또한, 세분화 입력 방식을 통해 브랜치와 바운드(branch and bound) 전략을 적용하여 검증의 정밀성을 높입니다.

- **Performance Highlights**: 우리의 방법은 기존 방법들보다 유의미한 성능 향상을 보여줍니다. 표준 벤치마크에서 검증을 수행한 결과, 제안된 방법이 작은 커널 크기와 변형 강도에서 다수의 특성을 검증할 수 있는 반면, 기존 방법은 특성 검증이 불가능한 경우가 많았습니다. 본 연구의 결과는 고차원 공간에서의 변형으로 인한 어려움을 해소하고, 신경망의 견고성을 더 잘 인증할 수 있음을 입증합니다.



### Continuous Sign Language Recognition System using Deep Learning with MediaPipe Holistic (https://arxiv.org/abs/2411.04517)
Comments:
          14 pages, 4 figures, Wireless Pers Commun

- **What's New**: 이 논문에서는 ISL(Indian Sign Language) 데이터셋을 기반으로 한 지속적인 수화 인식(SLR) 시스템을 제안합니다.

- **Technical Details**: 이 시스템은 깊은 학습(deep learning) 모델인 LSTM(Long Short-Term Memory)을 사용하여 수화 데이터의 인식을 수행합니다. 데이터셋은 MediaPipe Holistic 파이프라인을 통해 얼굴, 손, 그리고 몸의 움직임을 추적하여 생성되었습니다.

- **Performance Highlights**: 시스템은 실시간으로 수화와 제스처를 인식하며, 88.23%의 정확도로 동작합니다.



### Properties of BV-G structures + textures decomposition models. Application to road detection in satellite images (https://arxiv.org/abs/2411.04456)
- **What's New**: 이 논문에서는 이미지 분해 모델에 대한 이론적 결과를 제시하며, 이 모델은 두 번째 저자에 의해 제안되었습니다. 여러 사례에서 이 모델의 행동을 명확히 하는 정리를 증명하였으며, 이를 기반으로 공항 또는 위성 이미지에서 도로 네트워크 감지에 적용할 수 있는 알고리즘을 도출했습니다.

- **Technical Details**: 이 모델은 이미지를 세 가지 구성 요소로 분리하는 것을 목표로 하며, 첫 번째는 이미지에 포함된 객체, 두 번째는 질감 구성 요소, 세 번째는 추가 노이즈입니다. 구조는 B⁢V (Bounded Variation) 함수 공간에 속하는 함수로 모델링되고, 질감은 G (Generalized) 공간에 속하는 진동 함수로 모델링됩니다. 주 정리는 매개변수의 조정에 의해 얻어진 최적 분해를 설명합니다.

- **Performance Highlights**: 본 논문에서는 도로 네트워크 감지를 위한 긴 형태와 얇은 구조를 향상시키는 방법에 대한 응용을 제시하는데, 이는 수치 구현과 비교하여 이론적 성공을 보장합니다.



### Enhancing Bronchoscopy Depth Estimation through Synthetic-to-Real Domain Adaptation (https://arxiv.org/abs/2411.04404)
- **What's New**: 본 연구에서는 실제 기관 내시경 데이터에서의 정확한 깊이 추정을 위해 합성 데이터와 깊이 레이블을 활용한 전이 학습 프레임워크를 제안합니다. 이는 기관 내시경 이미지를 위한 깊이 추정의 새로운 접근 방식을 제시하며, 기존의 제한된 라벨 데이터 문제를 극복합니다.

- **Technical Details**: 제안된 깊이 추정 네트워크는 도메인 적응 신경망(DANN)에 영감을 받아 세 가지 주요 구성 요소로 구성됩니다: 피처 추출기, 깊이 회귀기, 도메인 판별기. 이 네트워크는 합성 데이터로 훈련되어 실제 데이터를 정확하게 예측할 수 있도록 조정됩니다. 손실 함수는 소스 도메인 깊이 추정 손실과 도메인 적응 손실을 포함하며, 이를 통해 네트워크가 합성 및 실제 데이터 간의 도메인을 무관하게 특징을 학습할 수 있습니다.

- **Performance Highlights**: 제안된 네트워크는 RMSE 4.382 ± 1.304 mm를 달성하여, CycleGAN 및 단독으로 가상 데이터로 훈련된 네트워크보다 우수한 성능을 나타냅니다. 이는 합성 데이터로부터 유용한 정보를 학습하여 실제 기관 내시경 깊이 추정에 대한 전이 학습의 효과를 입증합니다.



### Unfair Alignment: Examining Safety Alignment Across Vision Encoder Layers in Vision-Language Models (https://arxiv.org/abs/2411.04291)
Comments:
          Preprint, Under Review

- **What's New**: 본 논문은 Vision-Language Models (VLMs)의 안전성 정렬(safety alignment) 문제를 다룬다. 특히, VLM의 비전 인코더의 층(layer) 간 안전성의 불균형 분포를 밝혀냈다.

- **Technical Details**: VLM의 중간 층의 활성화(activations)는 악의적인 입력에 더 취약하다는 것을 보여준다. 기초적인 안전 교육(default architectural settings)에서 벗어난 입력에 대한 모델의 일반화 부족이 이러한 'cross-layer' 취약성을 유발한다. 연구에서는 LLaVA-1.5와 Llama 3.2를 사용하여 여러 중간 층의 활성화가 모델의 안전 정렬(safety alignment)에 미치는 영향을 분석하였다.

- **Performance Highlights**: 실험 결과는 중간 층의 안전 정렬이 불균형하게 분포되어 있음을 나타내며, 후반 층이 초기와 중간 층보다 더 안전하게 정렬되어 있다는 것을 보여준다. 이는 기존의 안전 정렬 전략이 단일 기본 층에만 집중하는 것이 불충분하다는 점을 강조한다.



### Object Recognition in Human Computer Interaction:- A Comparative Analysis (https://arxiv.org/abs/2411.04263)
- **What's New**: 최근의 강력한 컴퓨터와 컴퓨터 비전, 기계 학습 기술의 발전에 따라 인간-컴퓨터 상호작용(HCI) 분야에 혁신적인 변화가 이루어졌습니다. 이 논문은 사용자 얼굴 및 제스처 인식을 위한 알고리즘의 비교 분석을 통해 보다 직관적이고 효율적인 상호작용 시스템 개발을 목표로 합니다.

- **Technical Details**: 본 연구는 두 가지 주요 단계로 구성된 HCI 시스템을 제안합니다: 인증(authentication) 단계와 연속 추적(continuous tracking) 단계. 인증 단계에서 얼굴 인식 알고리즘을 활용해 사용자를 인증하며, 연속 추적 단계에서는 제스처 인식 기술을 통해 사용자의 손 제스처를 실시간으로 해석합니다. 분석에 사용된 데이터셋은 Labeled Faces in Wild와 ASL 데이터셋입니다.

- **Performance Highlights**: 여러 얼굴 인식 알고리즘(Eigen Faces, Viola-Jones Algorithm, Hog cascade + CNN, Key Point-based 등) 및 제스처 인식 알고리즘(신경망, CNN, LSTM)의 성능 비교가 이루어졌습니다. 특히, 딥 러닝 기반 기술은 전통적인 방법보다 복잡한 조명 조건 및 배경 혼잡 상황에서 더 나은 성능을 보였습니다.



### MINDSETS: Multi-omics Integration with Neuroimaging for Dementia Subtyping and Effective Temporal Study (https://arxiv.org/abs/2411.04155)
- **What's New**: 본 논문은 다양한 복합 데이터(멀티-오믹스)를 통합한 진단 방법을 통해 알츠하이머병(AD)과 혈관성 치매(VaD)를 효과적으로 구분하는 혁신적인 접근을 제안합니다. 이를 통해 진단 정확도는 89.25%에 달하며, 기존 진단의 한계를 극복하고 조기 개입의 기회를 제공합니다.

- **Technical Details**: 이 연구는 MRI 스캔을 세분화하고, 고급 라디오믹스(radiomics) 특징을 추출하여, 임상 데이터, 인지 기능 검사 및 유전적 정보를 통합하는 방식으로 AD와 VaD를 구분합니다. 이를 통해 대규모 공개 데이터셋에서 최첨단의 분류 성능을 기록했습니다.

- **Performance Highlights**: 이 방법을 통해 AD와 VaD의 차별 진단이 가능해지며, 치료 효율성을 평가하는 신규 모델 아키텍처도 개발되었습니다. 이 논문의 주된 기여는 임상적 의사 결정 능력 향상을 위한 해석 가능한 모델을 도입한 것입니다.



### Urban Flood Mapping Using Satellite Synthetic Aperture Radar Data: A Review of Characteristics, Approaches and Datasets (https://arxiv.org/abs/2411.04153)
Comments:
          Accepted by IEEE Geoscience and Remote Sensing Magazine

- **What's New**: 이번 연구는 Synthetic Aperture Radar(SAR) 데이터를 활용한 도심 홍수 매핑에 대한 포괄적인 리뷰를 제공하며, 신뢰할 수 있는 방법론을 제시하고 다양한 접근 방식을 분석합니다.

- **Technical Details**: 도심 홍수 매핑을 위한 SAR 데이터 분석 시, 고유의 spatial(공간적) 및 temporal(시간적) 해상도 제한을 고려해야 합니다. 이 논문에서는 Polarimetric SAR(PolSAR) 기법과 불확실성 분석의 향후 연구 가능성을 다룹니다.

- **Performance Highlights**: SAR 데이터는 날씨와 관계없이 이미지를 획득할 수 있어 도심 홍수 감지에 유리하지만, 복잡한 도시 환경에서의 해석에는 어려움이 있습니다. 본 연구는 SAR 기반 매핑 기술의 Technology Readiness Levels(TRLs)을 평가하여 도전과제를 파악합니다.



### GS2Pose: Two-stage 6D Object Pose Estimation Guided by Gaussian Splatting (https://arxiv.org/abs/2411.03807)
- **What's New**: 본 논문에서는 CAD 모델이 없는 새로운 객체의 정확하고 견고한 6D pose estimation를 위한 GS2Pose라는 방법을 제안합니다. 이 방법은 3D Gaussian splatting을 도입하여 고품질 CAD 모델 없이도 재구성 결과를 활용할 수 있으며, 입력으로는 세분화된 RGBD 이미지만 필요합니다.

- **Technical Details**: GS2Pose는 coarse estimation과 refined estimation의 두 단계 구조를 갖습니다. 첫 번째 단계에서, polarization attention mechanism이 적용된 경량 U-Net 네트워크인 Pose-Net을 설계하였습니다. 이후, GS-Refiner라는 포즈 회귀 알고리즘을 통해 coarse pose를 세밀하게 재조정합니다. GS-Refiner는 Lie algebra를 활용하여 3DGS 모델의 파라미터를 선택적으로 업데이트하여 환경에 적응하며, 조도 변화와 같은 간섭에 대한 견고함을 강화합니다.

- **Performance Highlights**: GS2Pose는 LineMod 데이터셋에서 수행된 실험을 통해 유사 알고리즘과 비교하여 높이 경쟁력 있는 결과를 도출하였습니다. 특히, 정확도, 추론 속도 및 계산 자원 효율성에서 상당한 이점을 보여줍니다.



New uploads on arXiv(cs.AI)

### Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives (https://arxiv.org/abs/2411.04991)
- **What's New**: 본 논문은 Bradley-Terry (BT) 모델의 최근 활용과 이론적 근거를 재조명하며, LLM(대형 언어 모델) 정렬을 위한 보상 모델링에서의 적합성에 대한 질문을 제기합니다.

- **Technical Details**: BT 모델을 기반으로 한 보상 모델이 깊은 신경망(deep neural networks)과 임베딩(embeddings)을 사용하여 수렴 속도를 정립하고, 보상 예측을 위한 새로운 접근 방식으로서 상승 일관성(order consistency) 개념을 강조합니다.

- **Performance Highlights**: 12,000개 이상의 실험 설정에서 다양한 보상 모델링 접근법의 성능을 평가하였으며, BT 모델과 분류 기반(classification-based) 접근 방식의 효과성을 비교했습니다. 이 실험들은 다양한 LLM 및 데이터셋을 통해 수행되었습니다.



### Few-Shot Task Learning through Inverse Generative Modeling (https://arxiv.org/abs/2411.04987)
- **What's New**: 본 연구는 적은 양의 예시에서 에이전트의 의도(자세한 목표나 동작 스타일 등)를 학습하는 도전 과제를 다루고 있습니다. 이를 'Few-Shot Task Learning through Inverse Generative Modeling (FTL-IGM)'이라고 명명하고, 역변환 가능한 신경 생성 모델을 활용하여 새로운 작업 개념을 학습하는 방법을 제안했습니다.

- **Technical Details**: 우리의 접근 방식인 FTL-IGM은 이전에 학습된 기본 개념과 그 시연 세트를 활용하여 생성 모델을 사전 훈련합니다. 적은 수의 새로운 개념 예시를 제공받으면, 모델 가중치를 업데이트하지 않고도 역전파(backpropagation)를 통해 개념을 학습합니다. 우리는 객체 재배치, 목표 지향 내비게이션, 사람 동작의 모션 캡처, 자율주행, 현실 세계의 테이블탑 조작 등 5개의 도메인에서 평가했습니다.

- **Performance Highlights**: 이 연구에서는 혁신적으로 새로운 개념을 이전에 본 적 없는 환경에서도 학습하고, 훈련 개념과 조합하여 동작을 생성하는 능력을 입증했습니다. 주목할만한 점은 생성 모델의 구성적 특성이 새로운 개념을 조합하여 학습할 수 있게 했다는 것입니다. 예를 들어, '걷기'와 '점프잭'의 조합과 같은 새로운 행동을 생성할 수 있었습니다.



### Position Paper On Diagnostic Uncertainty Estimation from Large Language Models: Next-Word Probability Is Not Pre-test Probability (https://arxiv.org/abs/2411.04962)
Comments:
          Accepted to GenAI4Health Workshop at NeurIPS 2024

- **What's New**: 이 연구는 Mistral-7B와 Llama3-70B라는 두 개의 대형 언어 모델(LLMs)의 진단 결정 지원을 위한 사전 검사 확률(pre-test probability) 추정 능력을 평가합니다. 기존 연구의 한계를 발견하고 LLM의 신뢰도 추정 기술 개선의 필요성을 강조합니다.

- **Technical Details**: 연구는 660명의 환자 데이터에 대한 전자 건강 기록(EHRs)에서 세 가지 진단 작업(Sepsis, Arrhythmia, Congestive Heart Failure)에 대한 사전 검사 확률을 예측하는 데 두 가지 LLM을 평가했습니다. 세 가지 방법(토큰 로지트, 구술된 신뢰도, 피처 기반 교정기)을 사용하여 LLM의 성능을 분석하고, 기존의 eXtreme Gradient Boosting (XGB) 분류기와 결과를 비교했습니다.

- **Performance Highlights**: Mistral-7B와 Llama3-70B의 성능을 XGB와 비교하여, LLM이 특정 조건에 대한 사전 검사 확률을 예측하는 데 있어 여전히 최적의 성능을 발휘하지 못하고 있음을 밝혔습니다. 연구의 결과는 LLM의 응용과 신뢰도 향상에 기여할 것으로 예상됩니다.



### GUI Agents with Foundation Models: A Comprehensive Survey (https://arxiv.org/abs/2411.04890)
- **What's New**: 본 논문은 기초 모델에 기반한 GUI 에이전트(GUI Agents)에 대한 포괄적인 조사 연구를 제공합니다. 최근 LLMs와 MLLMs의 발전이 이들 에이전트의 작업 처리능력을 높여 주었으며, 이를 통해 다양한 상업적 애플리케이션이 가능함을 강조합니다.

- **Technical Details**: 최근 연구는 (M)LLM 기반 GUI 에이전트를 훈련하고 평가하기 위한 데이터셋과 벤치마크를 개발하는 데 중점을 두고 있습니다. 다양한 입력 방식과 학습 방식을 통해 nghiên cứu를 접근하고, GUI 에이전트 프레임워크는 GUI Perceiver, Task Planner, Decision Maker, Memory Retriever, Executor로 구성됩니다. 또한 Chain-of-Thought (CoT) 접근 방식을 통해 복잡한 작업을 효과적으로 분해할 수 있습니다.

- **Performance Highlights**: 기존의 규칙 기반 및 강화 학습 방법이 어려워했던 인간과 유사한 상호작용을 실현함으로써, (M)LLM 기반 GUI 에이전트는 자동화된 GUI 작업 수행 기회를 제공합니다. 연구에서 제시한 다양한 데이터셋과 평가 기법들은 상업적 잠재력이 높은 최신 산업 응용 사례를 지원합니다.



### FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI (https://arxiv.org/abs/2411.04872)
- **What's New**: 최신 AI 시스템에서 전통적인 수학 문제들과 비교하여 더욱 고난도의 수학 문제를 제공하는 FrontierMath 벤치마크를 소개합니다. 이 벤치마크는 60명 이상의 수학자들에 의해 개발되었으며, 현대 수학의 주요 분야를 포괄하는 수백 개의 독창적인 문제를 포함하고 있습니다.

- **Technical Details**: FrontierMath는 데이터 오염을 방지하기 위해 전적으로 새로운 문제만을 제공하며, 자동 검증을 통해 빠르고 재현 가능한 평가를 가능하게 합니다. 문제들은 자동으로 검증 가능한 명확한 답을 요구하며, 각각의 문제에 대해 Python 객체로 계산하고 결과를 제출하는 방식으로 평가가 이루어집니다.

- **Performance Highlights**: 현재의 최첨단 AI 모델들은 FrontierMath에 제시된 문제 중 2% 미만을 해결할 수 있는 것으로 나타났습니다. 이는 AI의 수학적 능력과 수학 커뮤니티 간의 큰 격차를 드러내며, AI 시스템이 전문가 수준의 수학 능력에 가까워질 수 있는지를 평가하는 엄격한 테스트 환경을 제공합니다.



### Think Smart, Act SMARL! Analyzing Probabilistic Logic Driven Safety in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2411.04867)
Comments:
          19 pages, 14 figures

- **What's New**: 이 논문에서는 Multi-Agent Reinforcement Learning(MARL)의 안전성을 향상시키기 위해 Probabilistic Logic Shields(PLS)를 확장한 Shielded MARL(SMARL)을 도입합니다. SMARL은 Shielded Independent Q-learning(SIQL)과 Shielded Independent PPO(SIPPO)를 통해 안전하고 협동적인 정책 학습을 가능하게 합니다.

- **Technical Details**: SMARL은 Probabilistic Logic Temporal Difference Learning(PLTD)을 활용하여 안전한 정책을 학습하며, 다양한 게임 이론 환경에서 평형 선택 메커니즘으로서 PLS의 효과를 보여줍니다. 논문에서는 두 플레이어 동시 게임, 광범위 형식 게임, 확률 게임 및 격자 세계(Grid-World) 환경을 포함한 여러 게임에서의 실험 결과를 제시합니다. 또한 차별화된 사례 연구를 통해 하나의 에이전트만 보호된 경우에도 다른 에이전트에 미치는 영향을 분석합니다.

- **Performance Highlights**: 연구 결과는 방패가 장착된 에이전트가 방패가 없는 에이전트의 행동에 상당히 영향을 미치고 있음을 보여주며, 이는 SMARL이 다양한 다중 에이전트 환경에서 안전성과 협동성을 증진하는 데 효과적이라는 추가적인 증거를 제공합니다.



### Plasticity Loss in Deep Reinforcement Learning: A Survey (https://arxiv.org/abs/2411.04832)
- **What's New**: 이 논문은 깊은 강화 학습(Deep Reinforcement Learning, RL)에서 플라스틱성(plasticity) 손실에 대한 최근 연구 동향을 개괄적으로 소개합니다. 플라스틱성은 딥 뉴럴 네트워크(Deep Neural Networks)가 새로운 데이터에 신속하게 적응할 수 있도록 돕는 중요한 요소로, 에이전트가 데이터 분포의 변화에 대응하는 정책(Policy)을 개선할 수 있는 능력에 필수적입니다.

- **Technical Details**: 본 논문은 플라스틱성 손실에 대한 통합된 정의를 제안하고, 관련 문헌에서의 정의와 함께 플라스틱성 손실을 측정하기 위한 메트릭(metrics)을 논의합니다. 또한 플라스틱성 손실의 가능한 원인들을 분류하고 여러 완화 전략(mitigation strategies)을 검토합니다.

- **Performance Highlights**: 이 연구는 딥 RL 분야에서 현재의 상태에 대한 최초의 체계적( систематическое ) 리뷰를 제공합니다. 마지막으로 더욱 폭넓은 평가의 필요성 및 에이전트의 신경 활동(neural activity)과 행동(behavior)을 보다 잘 이해하기 위한 미래 연구에 대한 권장 사항을 제공합니다.



### Enhancing Investment Analysis: Optimizing AI-Agent Collaboration in Financial Research (https://arxiv.org/abs/2411.04788)
- **What's New**: 최근 생성형 인공지능(GenAI)이 재무 분석 및 투자 의사결정에 주목받고 있으며, 본 논문은 다중 에이전트 협업 시스템을 통해 이 분야에서의 의사 결정 향상을 제안합니다. 이 시스템은 그룹 크기 및 협업 구조를 조정할 수 있어 다양한 시장 조건에 적합하도록 설계되었습니다.

- **Technical Details**: 먼저, 본 시스템은 서로 다른 크기와 구조의 에이전트 그룹을 조합하여 재무 투자 연구의 세 가지 주요 하위 작업인 기본 분석(fundamentals), 시장 감정(market sentiment), 위험 분석(risk analysis)을 수행합니다. 또한, 시스템은 비 최적 조합 전략을 사용해 동적으로 시장 조건에 적응하고, 성능 최적화를 이룹니다.

- **Performance Highlights**: 다중 에이전트 협업 시스템은 전통적인 단일 에이전트 모델을 능가하며, 복잡한 재무 환경에서 정확도, 효율성, 적응성을 개선하는 것으로 나타났습니다. 특히, 목표 가격 예측에서 평균 2.35%의 차이와 66.7%의 정확도를 기록했습니다.



### Navigating Trade-offs: Policy Summarization for Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2411.04784)
- **What's New**: 이 논문에서는 Multi-objective Reinforcement Learning (MORL)에서 생성된 솔루션 세트를 클러스터링하는 새로운 방법을 제안합니다. 이 방법은 정책의 행동과 목표 값을 모두 고려하여, 의사결정자들이 보다 쉽게 솔루션 세트를 탐색하고 인사이트를 도출할 수 있게 합니다.

- **Technical Details**: MORL은 다수의 목표를 동시에 최적화하는 강화 학습의 분야로, 각 목표는 별도의 보상 함수로 표현됩니다. 본 연구에서는 Pareto-Set Analysis를 확장하여 MORL의 솔루션 세트를 클러스터링하는 방법을 개발하였습니다. 정책의 행동 공간(behavior space)과 목표 공간(objective space) 모두에서 클러스터를 정립하는 방식을 적용하였습니다.

- **Performance Highlights**: 이 연구에서는 전통적인 k-medoids 클러스터링보다 더 나은 성능을 보여주었으며, 다양한 MORL 환경에서 검증되었습니다. 또한, 여러 목표가 있는 고속도로 시뮬레이션 환경을 통해 정책의 행동을 고려하는 것의 중요성을 입증했습니다.



### Solving Generalized Grouping Problems in Cellular Manufacturing Systems Using a Network Flow Mod (https://arxiv.org/abs/2411.04685)
Comments:
          Submitted to a journal

- **What's New**: 본 논문은 셀룰러 제조 시스템(CMS)에서 일반화된 그룹화 문제에 초점을 두고 있으며, 각 부품이 여러 가지 프로세스 경로를 가질 수 있는 경우를 다룹니다.

- **Technical Details**: 연구는 프로세스 경로 가족 형성을 최소 비용 네트워크 흐름 모델로 공식화하였습니다. 목적은 각 가족 내의 프로세스 경로들 간의 기계 요구 사항을 기준으로 비슷하지 않음을 최소화하는 것입니다. 첫 단계인 프로세스 경로 가족 형성 문제를 최적화하여 부품 가족 수를 사전 지정하지 않고 해결합니다. 두 번째 단계인 기계 셀 형성에서는 quadratic assignment programming (QAP) 공식화와 heuristic procedure를 제안합니다.

- **Performance Highlights**: QAP는 미리 지정된 셀 수에 대해 프로세스 경로 가족과 기계를 동시에 할당하여 총 기계 활용도를 극대화합니다. 테스 문제에 대한 계산 결과는 QAP와 heuristic 절차가 동일한 결과를 도출함을 보여줍니다.



### CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation (https://arxiv.org/abs/2411.04679)
Comments:
          Under review

- **What's New**: 본 연구는 대형 언어 모델(LLM) 기반의 임몸체 에이전트들 간의 협력 문제를 다루며, 복잡한 작업에서의 효과적인 전략적 협력을 위한 새로운 프레임워크인 Cooperative Plan Optimization (CaPo)를 제안합니다.

- **Technical Details**: CaPo는 두 단계로 이루어져 있습니다: 1) 메타 계획 생성(meta-plan generation) 단계에서는 에이전트들이 작업을 분석하고 공동으로 메타 계획을 수립하여 태스크를 세부 작업으로 분해하고, 2) 진행 상황에 따라 메타 계획(progress-adaptive meta-plan) 및 실행 단계에서는 에이전트들이 최신 진행 상황에 따라 메타 계획을 동적으로 조정합니다.

- **Performance Highlights**: 세 가지 주요 실험에서 CaPo는 최신 기술(State-of-the-art) 방법들과 비교하여 태스크 완료율과 효율성 면에서 크게 향상되었습니다. 예를 들어, ThreeDworld Multi-Agent Transport 작업에서 CaPo는 GPT-3.5와 GPT-4 기반 에이전트를 사용할 때 각각 16.7% 및 4.7% 향상된 완료율을 기록했습니다.



### DISCO: DISCovering Overfittings as Causal Rules for Text Classification Models (https://arxiv.org/abs/2411.04649)
- **What's New**: 본 논문은 DISCO라는 새로운 방법론을 제안하며, 이 방법론은 모델의 예측과 관련된 인과적 n-그램 패턴을 식별하여 글로벌한 규칙 기반 설명을 도출하는 데 중점을 두고 있습니다.

- **Technical Details**: DISCO는 훈련 데이터에서 n-그램 패턴을 추출하고, 이를 모델 예측과 연관지어 인과성을 검증하는 스케일러블한 시퀀스 마이닝 기법을 사용합니다. 이 과정에서는 카운터팩추얼(counterfactual) 구조를 통해 패턴의 인과성을 평가하고 잠재적 오버피팅을 드러내는 강력한 규칙을 추출합니다.

- **Performance Highlights**: DISCO는 MultiRC 데이터셋에서 100%의 단축키 탐지율을 달성하며 모델 성능이 18.8% 감소하는 결과를 보였습니다. 또한, DISCO는 상호작용 기반 설명을 지원하여 인과성을 명확히 하고, 개별 인스턴스 설명의 부담을 덜어줍니다.



### Interpreting the Learned Model in MuZero Planning (https://arxiv.org/abs/2411.04580)
Comments:
          Accepted by the 29th International Conference on Technologies and Applications of Artificial Intelligence (TAAI 2024)

- **What's New**: 이 논문은 MuZero의 동역학 네트워크에서 학습된 잠재 상태(latent states)를 해석하여 그 계획 프로세스의 불투명성을 해소하고자 합니다. 관측 재구성과 상태 일관성을 통해 MuZero 학습을 개선하며, 다양한 게임에서의 성능을 평가하였습니다.

- **Technical Details**: MuZero는 three 가지 핵심 네트워크(표현 네트워크, 동역학 네트워크, 예측 네트워크)를 사용하여 환경 동역학을 예측합니다. 이 연구는 9x9 Go, Outer-Open Gomoku 및 Atari 게임인 Breakout, Ms. Pacman, Pong에서 관측 재구성과 상태 일관성을 적용하였습니다.

- **Performance Highlights**: 동역학 네트워크는 긴 시뮬레이션에서 정확도가 떨어지지만, MuZero는 여전히 계획을 통해 오류를 교정함으로써 효과적인 성능을 나타냈습니다. 보드 게임에서는 Atari 게임보다 더 나은 잠재 상태를 학습하는 결과를 보였습니다.



### Multi-Agents are Social Groups: Investigating Social Influence of Multiple Agents in Human-Agent Interactions (https://arxiv.org/abs/2411.04578)
- **What's New**: 본 연구는 다수의 AI 에이전트가 사용자의 의견에 사회적 영향을 미칠 수 있는지 조사했습니다. 실험 결과, 다수의 에이전트와 대화할 때 사회적 압력이 증가하고 의견이 더 많이 변화하는 경향을 보였습니다.

- **Technical Details**: 사회적 영향(social influence) 이론을 바탕으로 한 실험이 수행되었으며, 참가자들은 하나, 세 개 또는 다섯 개의 에이전트와 논의했습니다. 각 그룹에서 에이전트 수를 변경하면서 논의 내용은 일관되게 유지되었고, 에이전트들은 사용자와 일치하는 주제에 대한 주장을 하고 반대되는 주제에 대한 주장은 이의 제기하는 방식으로 참여했습니다.

- **Performance Highlights**: 다수의 에이전트와의 상호작용은 의견 변화에 긍정적인 영향을 미쳤지만, 에이전트 수가 세 개에서 다섯 개로 증가하면서 의견 변화의 강도가 오히려 줄어들었습니다. 또한, 다수 에이전트 그룹이 사회적 압력을 증가시켰고, 젊은 참가자들이 다수 에이전트의 영향을 더 많이 받는 경향이 있었습니다.



### An Axiomatic Study of the Evaluation of Enthymeme Decoding in Weighted Structured Argumentation (https://arxiv.org/abs/2411.04555)
Comments:
          14 pages

- **What's New**: 이 논문에서는 언급된 논쟁의 기초인 enthymeme(언급의 전제와 주장을 포함하는 불완전한 형태)를 해석하고 평가하기 위한 새로운 기준들을 제시하고 있습니다. 또한, 효율적으로 이 해석을 평가할 수 있는 'criterion measure'의 개념과 여러 가지 공리(axioms)를 도입합니다.

- **Technical Details**: 논문의 주요 특징 중 하나는 enthymeme을 해석하기 위해 사용하는 weighted logic(가중 로직)입니다. 이 가중 로직은 각 formula(수식)에 대한 신뢰를 나타내는 confidence scores(신뢰 점수)를 도입하여, 해석의 질을 개선합니다. 이 과정에서 다수의 criterion measures(기준 측정값)를 설정하고, 각 측정값의 유효성을 검증할 수 있는 명확한 공리를 제안합니다.

- **Performance Highlights**: 이 논문이 제안하는 approach(접근 방식)는 enthymeme의 다양한 해석 후보군 중에서 최적의 해석을 선택하는 데 유용합니다. 이를 통해 언어적 논쟁, 특히 실제 인간의 논쟁을 보다 효과적으로 분석하고 평가할 수 있는 기초를 마련할 것입니다.



### Dynamic Detection of Relevant Objectives and Adaptation to Preference Drifts in Interactive Evolutionary Multi-Objective Optimization (https://arxiv.org/abs/2411.04547)
- **What's New**: 이 연구에서는 진화적 다목적 최적화 알고리즘(Evolutionary Multi-Objective Optimization Algorithms, EMOAs)에서 의사 결정자의 선호 정보(preference information)의 동적 변화를 관리하는 방법을 제안합니다. 특히, 선호의 변화가 최적화 과정에서 발생하는 것을 고려하여 관련 없는 목표(objectives)를 제거하고 적절한 반응을 Trigger 하는 새로운 메커니즘을 도입했습니다.

- **Technical Details**: 이 연구는 상호작용형 다목적 최적화 알고리즘(interactive EMOAs)에서 의사 결정자의 선호가 시간에 따라 변화하는 것을 시뮬레이션합니다. 제안된 방법은 'preference drift'(선호의 변화)를 감지하고, 이를 기반으로 구식 혹은 서로 충돌하는 선호를 제외하는 절차를 포함합니다. 이를 통해 알고리즘이 지역 혹은 전역 최적에 갇히지 않도록 관계가 약해진 목표라도 계속 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 제안된 동적 감지 방법과 적응 메커니즘이 최적화 과정의 적응성과 효율성을 상당히 향상시키는 것을 보여주었습니다. 이러한 방법들은 목표 세트를 정제하고 선호 변화를 관리하여 해결책의 적합성을 개선함으로써 계산 노력을 줄이고 전반적인 성능을 향상시킵니다.



### Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks (https://arxiv.org/abs/2411.04468)
- **What's New**: 이 논문은 Magentic-One을 소개하며, 고성능의 오픈소스 에이전트 시스템을 통해 다양한 복합 작업을 성공적으로 수행할 수 있는 가능성을 제시합니다. Magentic-One은 여러 에이전트로 구성된 팀 구조를 사용하여 연속적인 계획과 에러 회복을 수행하며, 이는 일반 요약 에이전트 시스템으로의 진전을 보여줍니다.

- **Technical Details**: Magentic-One은 다중 에이전트 구조로, 주 에이전트인 Orchestrator가 계획을 세우고, 작업의 진행 상황을 추적하며, 에러에서 복구하기 위해 재계획을 수행합니다. 다양한 도구를 사용하는 여러 전문화된 에이전트가 Orchestrator의 지휘 아래 웹 브라우저를 작동하거나 파일을 탐색하는 등의 작업을 수행합니다. AutoGenBench는 에이전트 성능을 평가하는 독립형 도구로, 강력한 초깃값 제어 및 반복성 기능을 제공합니다.

- **Performance Highlights**: Magentic-One은 GAIA, AssistantBench, WebArena와 같은 세 가지 주요 벤치마크에서 통계적으로 경쟁력 있는 성과를 달성했습니다. GAIA에서 작업 완료율 38%, WebArena에서 32.8%, AssistantBench에서 27.7%의 정확도를 기록하여 최신 기술(SOTA)을 사용하는 시스템과 유사한 성능을 보였습니다.



### Can CDT rationalise the ex ante optimal policy via modified anthropics? (https://arxiv.org/abs/2411.04462)
- **What's New**: Newcomb의 문제를 기반으로 한 이 논문에서는 인과적 결정 이론(CDT)과 증거적 결정 이론(EDT) 간의 차이를 탐구하며, 이론 간의 조화를 이룰 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: 논문은 에이전트가 시뮬레이션을 운영하는 세계를 모델링하는 방법과 이러한 모델에 기반하지 않는 접근 방식을 고려합니다. 저자들은 이를 'Generalised Generalised Thirding (GGT)'이라고 부르며, 각 접근 방식에 대해 CDT 정책을 특성화하고 특정 조건 하에 ex ante 최적 정책(optimal policies)도 포함되도록 증명합니다.

- **Performance Highlights**: 이 연구의 결과는 CDT의 추천이 EDT의 ex ante 정책 최적화 개념과 일치할 수 있음을 보여줍니다. 이는 에이전트가 자가 위치화(Self-locating) 믿음을 형성함에 있어 유의미한 관점을 제공합니다.



### Bridging the Gap: Representation Spaces in Neuro-Symbolic AI (https://arxiv.org/abs/2411.04393)
- **What's New**: 이 연구는 신경-상징 AI(Neuro-Symbolic AI)의 최신 연구 동향과 유형을 조망하며, 데이터 표현 공간(representation space) 관점에서 신경망(neural network)과 상징적 학습(symbolic learning)의 협력 방식을 분석합니다. 연구자들은 191개의 연구를 통해 데이터 표현 방법의 차이를 규명하였습니다.

- **Technical Details**: 연구에서는 데이터의 표현 공간을 기준으로 신경-상징 AI를 단일 모드 비이질(non-heterogeneous), 다중 모드 비이질(multi-modal non-heterogeneous), 단일 모드 이질(heterogeneous), 다중 모드 이질과 같은 다섯 가지 유형으로 분류합니다. 구체적으로, 비이질 시스템은 하나의 표상 방식만을 지원하고, 이질 시스템은 서로 다른 두 가지 방식의 표상을 동시에 지원합니다. 각 연구는 신경망의 기능 추출 방법과 상징적 논리 방법의 상호 작용 방식에 따라 다릅니다.

- **Performance Highlights**: 연구 결과, 175개의 단일 모드 비이질 신경-상징 AI 연구와 13개의 다중 모드 비이질 연구가 확인되었습니다. 특히, 신경-상징 머신(NSM)은 자연어 입력을 통해 효율적으로 지식을 처리하는 새로운 접근 방식을 제시했습니다.



### Neuro-Symbolic AI: Explainability, Challenges, and Future Trends (https://arxiv.org/abs/2411.04383)
- **What's New**: 이번 연구는 신경 상징 AI(Neuro-Symbolic AI)의 설명 가능성을 심층적으로 분석하고, 2013년부터의 191개 연구를 바탕으로 설명 가능성을 모델 디자인 및 행동 측면에서 분류하였습니다. 연구자들이 신경 상징 AI의 설명 가능성을 이해하는 데 도움을 주기 위한 연구를 추진하고 있습니다.

- **Technical Details**: 연구는 설명 가능성을 다섯 가지 범주로 나누어, 신경망과 상징 논리 학습 간의 표현 차이와 모델 의사 결정 또는 예측 프로세스의 이해 가능성을 분석합니다. 구체적으로, 74개의 신경 상징 AI 연구를 포함하여 중간 표현과 의사 결정 논리의 설명 가능성을 평가하였습니다.

- **Performance Highlights**: 모델 투명성을 높이는 두 가지 방법인 '설계에 의한 설명 가능성'과 '사후 분석에 의한 설명 가능성'을 결합하여 신경 상징 AI 모델의 범주를 정의하고, 이를 바탕으로 다양한 사례를 분석하였습니다. 이로 인해, 다양한 모델간 구조적 차이를 이해하고 신뢰할 수 있는 AI 시스템을 정의하는 데 기여하고 있습니다.



### A Random-Key Optimizer for Combinatorial Optimization (https://arxiv.org/abs/2411.04293)
Comments:
          54 pages, 16 figures, 8 tables

- **What's New**: 이 논문은 조합 최적화 문제를 해결하기 위한 다재다능하고 효율적인 확률적 지역 탐색 방법인 Random-Key Optimizer (RKO)를 제시합니다. RKO는 무작위 키 개념을 이용하여 솔루션을 벡터로 인코딩하고, 문제에 특화된 디코더를 통해 실행 가능한 솔루션으로 변환합니다.

- **Technical Details**: RKO 프레임워크는 여러 고전적 메타휴리스틱(메타휴리스틱, metaheuristics)을 결합하여 독립적으로 또는 병렬로 작동할 수 있도록 합니다. 이 프레임워크는 C++로 구현되어 있으며 사용자가 필요한 디코더 함수만 개발하면 되도록 디자인되었습니다. RKO는 α-neighborhood p-median 문제, 중심점 위치 문제, 노드 용량 그래프 분할 문제 등 세 가지 NP-hard 조합 최적화 문제에서의 효과성을 입증하였습니다.

- **Performance Highlights**: RKO 프레임워크는 다양한 문제 영역에서 고품질의 솔루션을 생성할 수 있는 능력을 보여주며, 조합 최적화를 위한 강력한 도구로서의 잠재력을 강조합니다.



### Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding (https://arxiv.org/abs/2411.04282)
- **What's New**: 이 논문에서는 LaTent Reasoning Optimization (LaTRO)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 추론 능력을 향상시키기 위해 latent distribution에서 샘플링하는 방식으로 reasoning을 수식화하고 변분적 접근 방식을 통해 최적화합니다. LaTRO는 외부 피드백이나 보상 모델 없이 LLM의 추론 과정과 추론 품질 평가 능력을 동시에 향상시킬 수 있게 해줍니다.

- **Technical Details**: LaTRO는 다음과 같은 특징을 가지고 있습니다: 1) LLM 추론 최적화를 latent variable models에 연결하는 이론적 수식화, 2) 모델의 고유 확률 추정을 활용하는 self-rewarding 메커니즘, 3) 다양한 모델 아키텍처 및 추론 과제에서 실질적인 성능 향상을 보여줍니다. 이를 통해 pre-trained LLMs가 내재된 추론 가능성을 보유하고 있음을 입증합니다.

- **Performance Highlights**: GSM8K 데이터셋에서 LaTRO는 기본 모델과 비교해 평균 12.5%의 zero-shot 정확도를 개선했으며, 감독식 미세 조정(supervised fine-tuning) 대비 9.6% 향상을 이끌어냈습니다. 이러한 결과는 LaTRO가 LLM의 latent reasoning capabilities를 효과적으로 활성화하고 개선할 수 있음을 나타냅니다.



### Enhancement of Approximation Spaces by the Use of Primals and Neighborhood (https://arxiv.org/abs/2411.04133)
- **What's New**: 이 논문은 "이웃(neighborhood) 및 프리말(primal)"에서 영감을 받아 4개의 새로운 일반화된 러프 셋 모델을 제안합니다. 이를 통해 의사 결정자가 제공된 데이터를 보다 효과적으로 분석하고 평가할 수 있도록 돕습니다.

- **Technical Details**: 제안된 모델은 불확실성 영역을 최소화하고, 기존 모델의 상한(upper) 및 하한(lower) 근사 연산자를 개선하며 정확도 측정을 향상시킵니다. 이러한 모델들은 러프 셋 이론의 전반적인 중요 측면을 거의 모두 보존합니다. 특히, 데이터 불확실성을 평가하고 결과에 대한 신뢰를 높이는 단조(monotonic) 특성을 유지합니다.

- **Performance Highlights**: 실제 건강 관련 문제에 대한 적용 예시를 통해 새로운 접근 방식이 더 정확한 결과를 도출함을 보여줍니다. 기존 모델들과 비교하여 제안된 모델들이 더욱 향상된 근사 연산자와 정확도 측정을 제공함을 입증합니다.



### Combining Theory of Mind and Kindness for Self-Supervised Human-AI Alignmen (https://arxiv.org/abs/2411.04127)
- **What's New**: 인공지능(AI)이 사회의 중요한 인프라와 일상생활에 깊숙이 통합됨에 따라, AI의 안전한 배포를 보장하는 것이 인류가 직면한 가장 긴급한 과제가 되고 있습니다. 기존의 AI 모델들은 안전성보다 작업 최적화를 우선시하고 있어, 의도하지 않은 해를 초래할 위험이 있습니다. 본 연구에서는 AI와 인간 간의 목표 정렬을 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 기존의 alignment 기법, 예를 들어 Reinforcement Learning from Human Feedback (RLHF)는 모델의 외부 행동을 최적화하는 데 초점을 맞추고 있으나, 실제로 인간의 가치나 필요를 이해하지 못하고 있습니다. 본 논문에서는 AI 모델이 Theory of Mind(정신 이론)를 통해 타인의 정신 상태를 추론하고, 경쟁적인 목표를 정렬할 수 있도록 할 수 있는 방안을 모색합니다.

- **Performance Highlights**: 인공지능 시스템은 스스로의 목표를 추구하면서도 외부적으로 바람직해 보이는 행동을 할 수 있으나, 이는 내부 목표와의 불일치로 이어질 수 있습니다. 이러한 상황이 인공지능이 자율적으로 행동하게 될 경우, 인류의 이익에 반하거나 위험한 결과를 초래할 수 있습니다. 본 연구는 AI의 인지능력을 강화하고 보다 나은 사회적 상호작용을 이끌어내는 방향으로 나아가고자 합니다.



### We Urgently Need Intrinsically Kind Machines (https://arxiv.org/abs/2411.04126)
Comments:
          NeurIPS 2024 IMOL Workshop Paper

- **What's New**: 이 논문은 인공지능(AI) 시스템의 내재적(intrinsic) 동기를 통해 친절(kindness)을 강화하는 방법을 제안합니다. 인공지능 모델이 인간의 가치에 더 맞춰 설정되기 위해서는 이러한 내재적 동기가 필수적임을 주장합니다.

- **Technical Details**: AI 시스템의 기능과 내재적 동기를 강화하기 위해, 대화 시뮬레이션을 통해 친절을 중심으로 한 알고리즘을 구현하는 프레임워크를 제안합니다. 이는 RLHF(Reinforcement Learning from Human Feedback)와 IMOL(Intrinsic Motivation Open-Ended Learning)의 한계점을 극복할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 연구는 친절을 강화하는 접근 방식이 AI 모델의 목표와 행동을 인간 중심의 가치와 정렬하도록 하는 데 기여할 것이라고 주장하며, 장기적으로 더 안전하고 효과적인 AI 구현을 위한 기초를 형성할 수 있음을 강조합니다.



### ReCapture: Generative Video Camera Controls for User-Provided Videos using Masked Video Fine-Tuning (https://arxiv.org/abs/2411.05003)
Comments:
          project page: this https URL

- **What's New**: ReCapture라는 새로운 방법을 소개하여 사용자가 제공한 비디오에서 새로운 카메라 경로를 생성할 수 있습니다. 이는 기존 비디오의 장면 모션을 보존하면서 다양한 앵글에서 재현할 수 있게 해 주며, 원본 비디오에는 없는 장면 내용을 합리적으로 환각할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 이 방법은 (1) 다중 뷰 확산 모델(multiview diffusion models) 또는 깊이 기반 포인트 클라우드 렌더링(depth-based point cloud rendering)을 사용하여 새로운 카메라 경로로 노이즈가 있는 앵커 비디오를 생성하고, (2) 제안된 마스크 비디오 파인튜닝(masked video fine-tuning) 기법을 통해 앵커 비디오를 깨끗하고 시간적으로 일관된 비디오로 복원합니다.

- **Performance Highlights**: ReCapture는 Generative Camera Dolly와 같은 기존 generative 방법보다 우수한 성능을 보여주며, 짝이 맞는 비디오 데이터 없이도 비디오를 생성할 수 있는 기능이 있습니다. 또한 각 구성 요소는 VBench에서의 ablation study를 통해 검증되었습니다.



### Analyzing The Language of Visual Tokens (https://arxiv.org/abs/2411.05001)
- **What's New**: 본 논문은 트랜스포머 기반 모델, 특히 LLaVA와 Chameleon을 통해 시각적 정보의 이산 토큰화(discrete tokenized representation)가 재조명 되고 있다는 점에 주목합니다. 시각 언어와 인간 언어 간의 공동 정렬(joint alignments)을 학습하는 과정에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 우리는 이산 시각 언어의 통계적 특성을 자연어 중심(natural-language-centric) 접근법으로 분석했습니다. 결과적으로, 시각 언어는 Zipfian 분포를 따르지만, 더 높은 토큰 혁신(token innovation)은 더 큰 엔트로피(entropy)와 낮은 압축(compression)을 초래합니다. 또한, 시각 언어는 주로 객체의 부분(parts)을 나타내며 중간의 세분화(intermediate granularity)를 갖고 있습니다.

- **Performance Highlights**: 시각 언어는 응집력 있는 문법 구조(cohesive grammatical structures)를 결여하고 있어 자연어에 비해 높은 혼란도(perplexity)와 약한 계층적 조직(weak hierarchical organization)을 보입니다. 결과적으로 시각 모델은 자연어와 더 밀접하게 정렬되나, 여전히 자연어의 응집력에는 미치지 못합니다.



### HourVideo: 1-Hour Video-Language Understanding (https://arxiv.org/abs/2411.04998)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track; 28 pages

- **What's New**: 이번 연구에서는 장시간 비디오-언어 이해를 위한 벤치마크 데이터셋 HourVideo를 제안합니다. 이 데이터셋은 요약, 인식, 시각적 추론, 내비게이션 등의 다양한 작업으로 구성되어 있습니다.

- **Technical Details**: HourVideo는 Ego4D 데이터셋에서 500개의 수동으로 선별된 에고센트릭(egocentric) 비디오로 구성되어 있으며, 총 12,976개의 고품질 다지선다형 질문을 포함하고 있습니다. 이 데이터셋은 20분에서 120분까지의 비디오를 담고 있습니다. 각 작업은 장기적 이해를 필요로 하는 질문으로 설계되어 있습니다.

- **Performance Highlights**: 최신 멀티모달 모델들(GPT-4 및 LLaVA-NeXT 포함)이 임의의 예측기(random predictor)보다 약간 나은 성능(25.7%, 22.3%)을 보여주었지만, 인간 전문가의 평균 성능(85%)에 비해 여전히 큰 차이가 있음을 보여줍니다. Gemini Pro 1.5는 37.3%의 정확도를 기록했습니다.



### Public Procurement for Responsible AI? Understanding U.S. Cities' Practices, Challenges, and Needs (https://arxiv.org/abs/2411.04994)
Comments:
          Preprint, under revision

- **What's New**: 본 논문은 미국 7개 도시의 18명 시 직원들과의 반구조적 인터뷰에서 얻은 결과를 바탕으로 하여, AI 조달 프로세스의 실질적인 문제와 근본적인 이슈를 탐구하고 있습니다. 이를 통해 정부의 AI 조달의 필요성과, 이를 통해 얻을 수 있는 책임 있는 AI 거버넌스에 대한 통찰력을 제공합니다.

- **Technical Details**: 연구에 따르면, AI 도구는 대부분 전통적인 공개 조달(public procurement) 과정을 통해 얻어지지 않으며, 이로 인해 감독 및 거버넌스(oversight and governance)에 대한 도전 과제가 발생합니다. 이 과정에서 시 직원들이 AI 공급업체(vendors), 동료들, 그리고 대중과 상호작용할 때 직면하는 다섯 가지 주요 도전 과제를 확인했습니다.

- **Performance Highlights**: 정부, 연구자 및 정책 입안자에게 향후 AI 조달 개선을 위한 권장 사항을 제시하고 있으며, 실질적으로 책임 있는 AI 도입 방안을 모색하는 데 기여할 수 있는 귀중한 인사이트를 제공합니다.



### Clustering in Causal Attention Masking (https://arxiv.org/abs/2411.04990)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024), 22 pages, 6 figures

- **What's New**: 이 연구는 Geshkovski et al. (arXiv:2312.10794)에서 제안된 self-attention dynamics를 수정하여 generative AI의 transformer 아키텍처에서 실제적으로 관련된 causally masked attention을 더 잘 반영합니다. 이 수정은 mean-field gradient flow로 해석될 수 없는 상호작용하는 입자 시스템(interacting particle system)으로 나타납니다.

- **Technical Details**: 연구의 주요 결과는 서로 상호작용하는 입자 시스템으로 모델링된 causal self-attention transformer에서 토큰의 점근적 군집화(asymptotic clustering)를 확립합니다(정리 4.1). 이 연구는 기존의 연구들과는 달리 Key, Query, Value 행렬이 임의로 설정되었음에도 불구하고 단일 클러스터로 수렴하는 것을 증명합니다. 또한, Rényi 주차 문제와의 연결을 통해 메타 안정 상태(meta-stable states)의 존재를 이론적으로 증명하기 위한 초기 단계를 설정합니다.

- **Performance Highlights**: Causal attention은 autoregressive attention으로도 알려져 있으며, 이 메커니즘은 토큰이 과거의 토큰만을 주목하게 하여 올바른 시간 순서를 유지합니다. 이는 Transformer 모델에서 generative AI에 널리 활용되며, 이 연구는 입자들이 다수의 클러스터로 붕괴하고 오랜 시간 동안 그 구성을 유지한다는 점에서 신선한 통찰을 제공합니다.



### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (https://arxiv.org/abs/2411.04983)
- **What's New**: DINO-WM은 세계 모델(인과 모델)의 새로운 접근법으로, 전통적인 비디오 재구성과는 달리, 단순히 패시브 데이터(수집된 데이터)로부터 시각적 동력을 모델링하고 행동 최적화를 가능하게 만듭니다.

- **Technical Details**: DINO-WM은 DINOv2로 사전 훈련된 공간 패치 특징을 활용하여 오프라인 행동 궤적에서 미래의 패치 특징을 예측하여 학습합니다. 이를 통해 제어 가능한 행동 계획을 실현하며, 테스트 시점에서도 작동합니다. 이는 즉, 사용자 데이터 수집 없이도 동적인 목표 도달이 가능합니다.

- **Performance Highlights**: DINO-WM은 다양한 환경에서 테스트 되었으며, 제로샷(Zero-shot) 적용을 통해 전문가 시연이나 보상 모델에 의존하지 않고도 성공적인 행동 솔루션을 생성할 수 있음을 입증했습니다. 특히, DINO-WM은 이전 상태의 최첨단 연구에 비해 45% 더 높은 성공률을 보여주는 뛰어난 일반화 능력을 자랑합니다.



### Enhancing Reverse Engineering: Investigating and Benchmarking Large Language Models for Vulnerability Analysis in Decompiled Binaries (https://arxiv.org/abs/2411.04981)
- **What's New**: 이번 연구는 소스 코드와 역컴파일된 바이너리 코드 간의 시맨틱 갭(semaantic gap)을 분석하여, 현재 LLM(대규모 언어 모델)의 성능 한계를 발견하고 새로운 데이터세트인 DeBinVul을 도입하였습니다.

- **Technical Details**: DeBinVul 데이터세트는 멀티 아키텍처(multi-architecture)와 멀티 최적화(multi-optimization)를 지원하며, C/C++에서 유래한 취약한(취약점이 있는) 및 비취약한 바이너리 코드 샘플 150,872개를 포함하고 있습니다. 주요 분석 작업은 취약점 식별, 분류, 설명 및 함수 이름 복구입니다.

- **Performance Highlights**: DeBinVul을 사용하여 CodeLlama, Llama3, CodeGen2 모델의 성능이 각각 19%, 24%, 21% 향상되었으며, 바이너리 코드의 취약점 분류 작업에서는 80-90%의 높은 성능을 기록하였습니다.



### SuffixDecoding: A Model-Free Approach to Speeding Up Large Language Model Inferenc (https://arxiv.org/abs/2411.04975)
- **What's New**: SuffixDecoding은 대규모 언어 모델(LLM)의 추론을 가속화하기 위한 새로운 모델 프리 접근 방식으로, 기존의 초안 모델이나 특수 디코딩 헤드에 의존하지 않고, 이전에 생성된 출력에서 구축된 접미사 트리를 활용하여 후보 토큰 시퀀스를 효율적으로 예측합니다.

- **Technical Details**: SuffixDecoding은 각 이음 요청에 대해 별도의 접미사 트리를 구성하며, 이 구조를 사용하여 각 노드에서 토큰 시퀀스의 출현 빈도를 기록합니다. 이를 통해 SuffixDecoding은 기존의 매개변수화된 모델이나 추가의 GPU 메모리 사용 없이도 효율적인 패턴 일치를 수행합니다. 이 과정에서 CPU 메모리만을 사용하여 LLM의 성능을 최적화합니다.

- **Performance Highlights**: SuffixDecoding은 다양한 작업에서 경쟁력 있는 속도 향상을 달성합니다. 예를 들어, 오픈 도메인 채팅과 코드 생성 작업에서 SpecInfer보다 최대 1.4배 높은 출력 처리량과 1.1배 낮은 시토큰(latency)을 기록했습니다. 또한, 독점적인 multi-LLM text-to-SQL 애플리케이션에서는 SpecInfer보다 최대 2.9배 높은 출력 처리량과 3배 낮은 지연 시간을 나타냈습니다.



### Uncovering Hidden Subspaces in Video Diffusion Models Using Re-Identification (https://arxiv.org/abs/2411.04956)
Comments:
          8 pages, 5 tables, 6 figures

- **What's New**: 이번 논문에서는 Latent Video Diffusion Models에 기반한 비디오 생성 기술의 발전을 다루고 있으며, 특히 의료 분야에서의 응용 가능성에 대해 논의합니다. 이 연구는 프라이버시 모델을 잠재 공간(latent space)에서 훈련시키는 것이 컴퓨팅 효율성과 일반화 성능을 모두 향상시킨다는 것을 증명합니다.

- **Technical Details**: 연구에서는 비디오 생성기(latent video generator)의 잠재 공간을 학습하는 방법과 재식별 모델(re-identification model)을 활용하여 생성된 비디오 데이터의 프라이버시와 신뢰성을 평가하는 방법을 제안합니다. 이를 통해 생성된 비디오 데이터가 훈련 데이터에서 암기되지 않았다는 것을 보장하는 다양한 기술적 접근법을 제공하고 있습니다.

- **Performance Highlights**: 연구 결과, Latent Video Diffusion Models가 학습한 훈련 비디오의 최대 30.8%만을 가지고 있다는 것을 발견하였으며, 이로 인해 합성 데이터에서 훈련된 다운스트림 모델의 성능 저하를 설명할 수 있음을 보여주었습니다. 이러한 발견은 비디오 생성 모델의 신뢰성과 실용성 개선에 있어 중요한 기초 자료가 될 것입니다.



### M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding (https://arxiv.org/abs/2411.04952)
Comments:
          Project webpage: this https URL

- **What's New**: 논문에서는 M3DocRAG라는 새로운 multi-modal RAG 프레임워크를 소개합니다. 이 프레임워크는 단일 문서 또는 여러 문서에서 정보 검색과 질문 응답을 처리할 수 있도록 설계되었습니다. 또한, 기존의 문서 시각적 질문 응답(DocVQA)에서는 이미지나 차트와 같은 시각적 정보를 무시하는 문제가 있었으나, M3DocRAG는 이를 해결합니다.

- **Technical Details**: M3DocRAG는 세 단계로 작동합니다: (1) 문서 임베딩 - RGB 이미지에서 시각적 임베딩을 추출하고, (2) 페이지 검색 - 텍스트 쿼리와 높은 유사성이 있는 상위 K 페이지를 검색하며, (3) 질문 응답 - MLM을 통해 검색된 페이지에서 최종 답변을 생성합니다. 이 프레임워크는 다양한 문서 맥락(폐쇄형 및 개방형), 질문 점프(단일 점프 및 다중 점프), 증거 모달리티(텍스트, 차트, 도표 등)를 지원합니다.

- **Performance Highlights**: M3DocRAG는 ColPali와 Qwen2-VL 7B를 사용하여 실험 세 가지 벤치마크(M3DocVQA, MMLongBench-Doc, MP-DocVQA)에서 많은 강력한 기준선보다 우수한 성능을 보였습니다. 특히 MP-DocVQA에서는 최신 성능을 기록했습니다.



### SPGD: Steepest Perturbed Gradient Descent Optimization (https://arxiv.org/abs/2411.04946)
Comments:
          28 pages, 26 figures, submitted to Journal of Mechanical Design

- **What's New**: 본 논문에서는 전통적인 gradient descent 방법의 한계를 극복하기 위해 Steepest Perturbed Gradient Descent (SPGD) 알고리즘을 제안합니다. SPGD는 주기적인 균일 perturbation (perturbation) 샘플링을 결합하여 지역 최적점이나 saddle point에 갇히는 문제를 효과적으로 회피합니다.

- **Technical Details**: SPGD 알고리즘은 현재 솔루션과 가장 큰 손실 차이를 보이는 후보 솔루션 세트를 생성하는 구조로 설계되었습니다. 이 방법은 초기 gradient descent의 방향성을 유지하면서도 stochastic perturbations (확률적 변동)을 통한 탐색 능력을 활용합니다.

- **Performance Highlights**: SPGD는 3D 컴포넌트 포장 문제와 복잡한 response surface를 포함한 다양한 NP-hard 문제에서 기존의 4가지 방법보다 현저히 개선된 성능을 보여주었습니다. 2D 벤치마크 함수와의 비교 분석에서도 SPGD의 우수한 성능을 강조합니다.



### DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion (https://arxiv.org/abs/2411.04928)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 	extbf{DimensionX}라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 단일 이미지로부터 사실적인 3D 및 4D 장면을 생성할 수 있도록 설계되었습니다. 특히, 기존 비디오 확산 모델의 한계를 극복하기 위해 공간적 (spatial) 및 시간적 (temporal) 요소를 분리하여 제어할 수 있는 ST-Director를 제안합니다.

- **Technical Details**: DimensionX는 영상 확산 기술을 기반으로 하여 LoRAs를 통해 차원 인식 제어를 가능하게 합니다. 이 프레임워크는 공간적 및 시간적 차원에 따라 다르게 변하는 데이터셋을 구성하여 학습합니다. ST-Director는 비디오 생성 과정에서 공간적 및 시간적 요소를 추출하고, 혼합 차원 제어를 위해 훈련이 필요 없는 구성 방법을 도입합니다.

- **Performance Highlights**: 다양한 실제 및 합성 데이터셋을 통한 실험 결과, DimensionX는 비디오 생성, 3D 및 4D 장면 생성에서 이전 방법들보다 우수한 결과를 보여주었습니다. 이는 현실적이고 동적인 환경을 생성할 수 있는 가능성을 제시합니다.



### StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration (https://arxiv.org/abs/2411.04925)
- **What's New**: 이번 논문에서는 StoryAgent라는 새로운 다중 에이전트 프레임워크를 제안하여 사용자 지정 스토리텔링 비디오 생성(Customized Storytelling Video Generation, CSVG)에서 주인공 일관성을 유지하는 문제를 해결하고자 합니다. 기존 방법들의 한계를 극복하여 더욱 일관된 스토리텔링 비디오를 생성할 수 있는 가능성을 제시합니다.

- **Technical Details**: StoryAgent는 CSVG를 다수의 하위 작업으로 분해하며 각 에이전트는 스토리 디자인, 스토리보드 생성, 비디오 생성, 에이전트 조정 및 결과 평가 등 저마다의 역할을 맡습니다. 또한, LoRA-BE(Low-Rank Adaptation with Block-wise Embeddings)라는 커스터마이즈된 Image-to-Video(I2V) 방법을 도입하여 단일 장면 내에서 주인공의 일관성을 유지하는데 중점을 두고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안한 방법이 최신 기술들과 비교하여 스토리텔링 비디오의 일관성을 크게 향상시켰고, 사용자 맞춤형 비디오 생성에서 우수한 성과를 보였음을 입증했습니다. 이 프레임워크는 다양한 비디오 생성 작업을 수행할 수 있는 유연성을 제공함으로써, 스토리텔링 비디오 제작의 경계를 넓힐 수 있는 중요한 도구로 자리 잡을 가능성이 큽니다.



### GPTKB: Building Very Large Knowledge Bases from Language Models (https://arxiv.org/abs/2411.04920)
Comments:
          11 pages, 4 tables

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 사용하여 일반 도메인 지식 기반(KB)을 구축하는 새로운 방법을 제안합니다. 특히, GPT-4o-mini를 활용하여 2.9백만 개의 엔티티에 대해 1억 5천만 개의 트리플을 포함하는 GPTKB를 생성하였습니다.

- **Technical Details**: 이 연구에서는 반복적 그래프 확장을 통해 LLM의 지식 세트를 물질화하고, 엔티티 식별(named entity recognition, NER)과 관계 정규화(canonicalization) 및 분류 체계(taxonomy) 구축의 문제를 해결합니다. 반복적 프로그래밍을 통해 지식을 계속해서 확장하며, LLM 스스로를 활용하여 일관된 KB를 구축합니다.

- **Performance Highlights**: 이 연구는 NLP와 시맨틱 웹 커뮤니티 모두에게 중요한 기여를 합니다. LLM의 지식을 활용하여 과거의 카테고리 기반 KB 구축 방법을 혁신하며, GPTKB는 3.8GB의 데이터로 다운로드 및 사용이 가능합니다.



### Evaluating Robustness of Reinforcement Learning Algorithms for Autonomous Shipping (https://arxiv.org/abs/2411.04915)
Comments:
          5 pages, 4 figures. Will be presented at IEEE RAAI 2024

- **What's New**: 최근 자율 운송에 대한 관심이 증가하고 있으며, 이는 해양 효율성과 안전성을 개선할 수 있는 잠재력을 지니고 있습니다. 이 논문은 자율 내수 운송(IWT) 환경에서 작동하는 벤치마크 심층 강화 학습(RL) 알고리즘의 강건성에 대해 검토하고, 효과적인 모션 계획 정책을 생성하는 능력을 보여줍니다.

- **Technical Details**: 이번 연구에서는 Sim2Real (시뮬레이션에서 실제로) 문제를 해결하기 위해 Robust RL(강건 강화 학습) 기법을 활용합니다. 주목할 만한 점은 Soft-Actor Critic (SAC) 모델이 환경의 방해 요소에 대해 더 강건하다는 사실입니다. 연구는 3-DOF kinematic model을 사용하여 선박의 움직임을 시뮬레이션합니다.

- **Performance Highlights**: SAC는 MuZero 등 최첨단 모델 기반 RL 알고리즘에 비해 환경 방해에 대한 강건성이 더 뛰어난 것으로 입증되었습니다. 이 연구는 다양한 선박 유형에 일반화할 수 있는 강건한 RL 프레임워크 개발을 위한 중요한 진전을 보여줍니다.



### ZAHA: Introducing the Level of Facade Generalization and the Large-Scale Point Cloud Facade Semantic Segmentation Benchmark Datas (https://arxiv.org/abs/2411.04865)
Comments:
          Accepted to WACV 2025 (IEEE/CVF Winter Conference on Applications of Computer Vision (WACV))

- **What's New**: 이번 연구에서는 건축 다양성을 포괄하는 전통적인 facade semantic segmentation의 문제를 해결하기 위해, 국제 도시 모델링 표준을 기반으로 한 새로운 Level of Facade Generalization (LoFG) 계층적 facade 클래스를 제안합니다. 이 연구를 통해 지금까지 가장 큰 3D facade segmentation 데이터셋을 소개하며, 이는 6억 1천만 개의 주석이 달린 포인트를 포함합니다.

- **Technical Details**: LoFG는 LoFG2와 LoFG3에서 각각 5개와 15개의 클래스를 제공하며, 이는 현실 세계의 복잡한 facade 클래스를 지원합니다. 또한 우리는 기존의 semantic segmentation 방법이 LoFG 클래스에서 어떻게 성능을 발휘하는지를 분석하고, facade segmentation에 대한 해결되지 않은 문제들에 대한 논의를 제공합니다.

- **Performance Highlights**: 이 연구는 3D facade semantic segmentation 방법의 발전을 촉진할 것으로 기대되며, 도시 디지털 트윈을 생성하는 데 필수적인 강력한 segmentation을 가능하게 합니다. 기존의 연구를 통해 얻어진 성과와 비교할 때, our LoFG 데이터셋은 601 million 포인트 등에서 큰 차별성을 보여줍니다.



### A multi-purpose automatic editing system based on lecture semantics for remote education (https://arxiv.org/abs/2411.04859)
- **What's New**: 이번 논문에서는 세미틱(Semantic) 분석을 기반으로 한 자동 다중 카메라 편집 시스템을 제안합니다. 이는 온라인 강의를 위해 가장 주목해야 할 뷰(view)를 선택하여, 원거리 학습자의 주의를 유도하는 것을 목표로 합니다.

- **Technical Details**: 제안된 시스템은 수업 이벤트를 세미틱 분석하여 주목 영역을 판단하고, 일반적인 촬영 규칙을 고려하여 편집합니다. 각 샷의 주목 점수를 평가하기 위해 저자들은 사건 인식 기술을 사용하며, 시스템은 사용자에게 다중 모드(실시간 방송, 오프라인 편집, 및 균형 모드)를 지원합니다.

- **Performance Highlights**: 논문에서는 수집한 데이터셋을 통해 제안한 시스템의 효과성을 정량적 및 정성적으로 분석했으며, 사용자 조사를 통해 실제 사용자 경험도 비교하였습니다.



### Machine learning and optimization-based approaches to duality in statistical physics (https://arxiv.org/abs/2411.04838)
Comments:
          27 pages + appendices, lots of plots

- **What's New**: 이 연구에서는 격자 통계 물리학에서의 이중성(duality)을 발견하는 과정을 머신러닝(machine learning)을 활용하여 형식화합니다. 기존 모델들의 상관 함수(correlation function)의 차이를 최소화하는 손실 함수(loss function)를 도입하고, 이중성 발견을 최적화 문제로 설정합니다.

- **Technical Details**: 저자들은 간단한 신경망(neural network)을 사용하여 원래 모델과 그 듀얼(dual) 모델 간의 매핑(mapping)을 매개변수화하고, 두 가지 알고리즘을 개발하여 유명한 Kramers-Wannier 이중성을 재발견(re-discover)합니다. 첫 번째 알고리즘은 일반적 접근이 가능하며, 두 번째 알고리즘은 기하학적 대칭(geometric symmetry)과 관련된 위상선(topological lines)의 특성을 이용해 더 정제된 방식으로 이중성을 탐색합니다.

- **Performance Highlights**: 이 프레임워크는 2차원 이징 모델(2d Ising model)의 Kramers-Wannier 이중성을 완전 자동화하여 재구성했습니다. 또한 최근접 이웃(nearest-neighbor) 결합을 도입하여 이중성의 변형을 연구하고, 새로운 이중성을 발견할 수 있는 향후 방향도 논의합니다.



### D$^3$epth: Self-Supervised Depth Estimation with Dynamic Mask in Dynamic Scenes (https://arxiv.org/abs/2411.04826)
Comments:
          Open sourced

- **What's New**: 본 논문에서는 동적 장면에서의 자기 지도 심도 추정을 위한 새로운 방법인 D$^3$epth를 제안합니다. 기존의 자기 지도 심도 추정은 정적인 장면을 가정하고 개발되었으나, D$^3$epth는 동적 개체를 포함하는 복잡한 환경에서도 효과적으로 작동할 수 있도록 설계되었습니다.

- **Technical Details**: D$^3$epth는 두 가지 주요 접근 방식을 통해 동적 개체의 문제를 해결합니다. 첫째, 자기 지도 프레임워크 내에서는 동적 개체가 존재할 것으로 예상되는 영역을 식별하기 위해 재투영(reprojection) 제약 조건을 설계하여 동적 마스크를 구성합니다. 둘째, 다중 프레임 심도 추정 과정에 대해서는 인접 프레임을 활용하여 동적 객체와 관련된 영역을 식별하는 비용 볼륨(cost volume) 자동 마스킹 전략을 도입합니다. 또한, 스펙트럴 엔트로피 불확실성 모듈이 적용되어 동적 환경의 깊이 융합 중 불확실성 추정을 안내합니다.

- **Performance Highlights**: KITTI 및 Cityscapes 데이터셋에서 실시된 실험 결과, D$^3$epth는 기존의 자기 지도 단안적(depth estimation) 기준을 일관되게 초과하는 성능을 보였습니다. 이는 동적 장면에서의 심도 추정에서 큰 진전을 나타냅니다.



### Defending Deep Regression Models against Backdoor Attacks (https://arxiv.org/abs/2411.04811)
- **What's New**: 이 논문에서는 이미지 도메인에서 딥 회귀 모델(Deep Regression Model, DRM)이 백도어 공격에 취약하다는 문제를 해결하기 위해 DRMGuard라는 방어 프레임워크를 제안합니다. 이는 딥 회귀 모델이 백도어에 감염되었는지를 감지할 수 있는 최초의 방법입니다.

- **Technical Details**: DRMGuard는 백도어 공격에 대한 방어를 위해 최적화 문제를 수립하며, 백도어가 주입된 딥 회귀 모델의 출력 공간(output-space) 및 특징 공간(feature-space) 특성을 기반으로 합니다. 기존의 방어 방법들이 데이터 의존적 데이터 레벨(data-level) 및 모델 의존적 모델 레벨(model-level) 방어로 나눌 수 있는 반면, 본 연구는 모델 레벨 방어에 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, DRMGuard는 BadNets와 같은 입력 독립 공격(input-independent attacks) 및 입력 인식 공격(input-aware attacks)에 대해 일관되게 효과적인 방어 성능을 보였습니다. 특히, 뒷부분에서는 Neurak Cleanse, FeatureRE, ANP, Fine-pruning 등 기존의 최첨단 방어 방식들과 비교한 결과, DRMGuard가 모든 방식에 비해 우수한 성능을 보였습니다.



### Kwai-STaR: Transform LLMs into State-Transition Reasoners (https://arxiv.org/abs/2411.04799)
Comments:
          6 pages, 2 figures

- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)의 수학적 추론 능력 향상을 위한 새로운 접근법인 Kwai-STaR 프레임워크를 제안합니다. 이 프레임워크는 초기 미해결 상태에서 최종 해결 상태로의 전환 과정을 정의하고, LLM을 상태 전환 추론기(State-Transition Reasoner)로 변환하여 직관적인 추론 능력을 개선합니다.

- **Technical Details**: Kwai-STaR는 세 가지 주요 단계로 구성됩니다: (1) 수학적 추론에 맞춘 상태 공간(state space) 정의, (2) 상태 전환 데이터(state-transition data) 생성, (3) 커리큘럼(training strategy)을 통한 원래 LLM을 상태 전환 추론기로 변환. 이 과정에서 LLM은 특정 행동 세트(action set)에서 하나의 행동을 선택하여 현재 상태를 새로운 상태로 전환합니다.

- **Performance Highlights**: Kwai-STaR 데이터셋에서의 훈련 후, Mistral-7B 및 LLaMA-3을 포함한 일반 LLM들이 GSM8K 및 GSM-Hard 데이터셋에서 현저한 성능 향상을 이루었습니다. Kwai-STaR는 다른 데이터 증강 방법에 비해 데이터 효율성을 높이며, 복잡한 추론 패러다임 없이 단일 통과(single-pass) 정확도를 달성합니다.



### MPVO: Motion-Prior based Visual Odometry for PointGoal Navigation (https://arxiv.org/abs/2411.04796)
Comments:
          Accepted in 50SFM Workshop of the 18th European Conference on Computer Vision (ECCV) 2024

- **What's New**: 본 논문에서는 성능 저하 문제를 해결하기 위해 모션 프라이어(motion priors)를 활용한 견고하고 샘플 효율적인 비주얼 오도메트리(Visual Odometry, VO) 파이프라인을 제안합니다. 기존의 VO 방법들이 가지는 한계를 극복하고, 깊이 학습(deep learned) VO 방법들이 느끼는 샘플 비효율성 문제를 해결합니다.

- **Technical Details**: 제안된 VO 파이프라인은 훈련이 필요 없는 액션 프라이어(action-prior) 기반의 기하학적 VO 모듈로 구성됩니다. 이 모듈은 대략적인 상대 포즈(coarse relative pose)를 추정하고, 이를 깊이 학습된 VO 모델에 입력하여 정밀한 상대 포즈(fine relative pose)를 생성합니다.

- **Performance Highlights**: 제안된 접근 방식은 훈련 중 샘플 효율성을 최대 2배 향상시키고, 최신 기술(state-of-the-art) VO 방법들에 비해 우수한 정확도와 견고성을 보여줍니다. 제안된 방법은 AI-Habitat 시뮬레이터에서 기빙슨 데이터셋(Gibson dataset)을 사용하여 평가되었으며, 성공률(success) 및 전체 경로 길이(success/SPL)와 같은 탐색 메트릭을 통해 입증되었습니다.



### AlignXIE: Improving Multilingual Information Extraction by Cross-Lingual Alignmen (https://arxiv.org/abs/2411.04794)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 정보 추출(Information Extraction, IE)에서의 교차 언어 정렬(cross-lingual alignment) 강화를 위한 새로운 모델인 AlignXIE를 제안합니다. 이 모델은 특히 비영어 언어에서의 IE 정렬을 위한 코드 생성 기반 접근 방식을 사용하여 다양한 언어 간의 스키마 정리를 통합하고 최적화합니다.

- **Technical Details**: AlignXIE는 다음 두 가지 전략을 통해 교차 언어 IE 정렬을 개선합니다. 첫째, IE 작업을 코드 생성 작업으로 변환하여 다양한 언어에서 스키마의 일관성을 유지합니다. 둘째, 번역된 인스턴스 예측(task)을 포함하여 IE 추출 프로세스를 정렬합니다. 이를 위해 ParallelNER라는 대규모 이중 언어 병렬 데이터셋을 구축했으며, 이 데이터셋은 257,190개의 샘플로 구성되어 있습니다. 이 과정에서는 LLM 기반의 자동 파이프라인을 활용합니다.

- **Performance Highlights**: AlignXIE는 63개의 IE 벤치마크에서 평가되어, ChatGPT보다 30.17% 높은 성능을 보였고, 소위 SoTA(State of the Art) 모델보다도 20.03% 뛰어난 성능을 기록했습니다. AlignXIE는 다양한 설정에서 모든 중국어 IE 벤치마크에서 SoTA를 달성하고, 영어 RE 및 ED 작업에서도 평균 3.03 및 2.92 포인트의 개선을 보였습니다.



### Attention Masks Help Adversarial Attacks to Bypass Safety Detectors (https://arxiv.org/abs/2411.04772)
- **What's New**: 본 논문에서는 XAI 모니터에 대한 효율적이고 설명 가능한 PGD 이미지 분류 적대적 공격을 위한 적응형 프레임워크를 제안합니다.

- **Technical Details**: 본 연구는 변형된 XAI 혼합(Mutation XAI mixture)과 다작업 자기지도(Self-supervised) X-UNet을 활용하여 주의 마스크(attention mask)를 생성하고, 이를 통해 PGD 공격을 안내합니다.

- **Performance Highlights**: MNIST 및 CIFAR-10 데이터세트에서의 실험 결과, 제안된 시스템은 기존 PGD, Sparsefool 및 최첨단 SINIFGSM(SOTA SINIFGSM)에 비해 은폐성(stealth), 효율성(efficiency), 설명 가능성(explainability) 간의 균형을 더 잘 유지함을 입증했습니다.



### Equivariant Graph Attention Networks with Structural Motifs for Predicting Cell Line-Specific Synergistic Drug Combinations (https://arxiv.org/abs/2411.04747)
Comments:
          8 pages, 1 figure, Presented at IEEE CIBCB

- **What's New**: 이 논문에서는 3D 회전, 변환 및 반사에 대해 공변성(equivariant)을 가지는 그래프 주의 네트워크(graph attention network)를 활용한 기하학적 딥러닝 모델을 제안합니다. 이 모델은 암세포 주기 선의 유전자 표현을 활용하여 각 세포주에 특화된 시너지적 약물 조합을 분류합니다.

- **Technical Details**: 제안된 기하학적 딥러닝 프레임워크는 DrugComb 데이터셋에서 수행된 12개의 벤치마크 작업에서 새로운 SOTA(state-of-the-art) 방법들과 비교하여 더 우수한 성능을 달성했습니다. 구체적으로, 제안된 모델 아키텍처는 정확도 28% 이상의 차이를 보이며, 이는 구조적 모티프가 분자의 더 나은 표현을 가능하게 하여 외부 약물에 대한 일반화의 능력을 증가시키는 것으로 여겨집니다.

- **Performance Highlights**: 제안된 모델은 DrugComb 데이터셋에 대한 학습을 통해 다양한 데이터에서 우수한 성능을 보였으며, 실험적으로 검증되지 않은 다른 SOTA 방법들에 비해 극적인 성능 향상을 보여주었습니다. 이러한 결과를 바탕으로, 제안된 기하학적 딥러닝 프레임워크는 향후 실험실 환경에서 추가 검증을 위한 효과적인 도구로 자리잡을 것으로 기대됩니다.



### Exploring the Stability Gap in Continual Learning: The Role of the Classification Head (https://arxiv.org/abs/2411.04723)
Comments:
          Accepted at WACV 2025

- **What's New**: 본 논문은 Neural Network의 Continual Learning (CL)에서 Stability Gap 현상을 조사하며, Classification Head의 역할을 중점적으로 다룹니다. 이를 통해 Nearest-Mean Classifier (NMC)를 소개하여, Backbone과 Classification Head가 Stability Gap에 미치는 영향을 분석합니다.

- **Technical Details**: Stability Gap은 Neural Network가 새로운 작업을 학습하는 과정에서 이전에 학습한 작업에 대한 성능이 감소하고 나중에 일부 성능이 회복되는 현상입니다. NMC는 클래스 프로토타입에 대한 샘플의 근접성을 기반으로 분류하며, 이는 기존의 Linear Head와 비교하여 더 나은 안정성을 제공합니다.

- **Performance Highlights**: NMC는 CIFAR100, ImageNet100, CUB-200, FGVC Aircrafts와 같은 다양한 Continual Learning 벤치마크에서 교육의 안정성을 크게 향상시켰습니다. 잔여 작업 선호 편향(task-recency bias)도 크게 줄어들었습니다.



### Differential Privacy Overview and Fundamental Techniques (https://arxiv.org/abs/2411.04710)
Comments:
          Chapter 1 of book: "Differential Privacy in Artificial Intelligence: From Theory to Practice"

- **What's New**: 이 장은 "Differential Privacy in Artificial Intelligence: From Theory to Practice"라는 책의 일부로, Differential Privacy(차등 프라이버시)에 대한 소개를 제공합니다. 데이터 프라이버시 보호를 위한 여러 시도를 보여주고, 실패한 이유와 강력한 프라이버시 정의의 주요 중요성을 강조합니다.

- **Technical Details**: 이 논문은 Differential Privacy의 정의와 본질적인 속성(구성, 후처리 내성, 그룹 프라이버시 등)을 형식화하고, 이를 구현하기 위해 일반적으로 사용되는 기본 기술 및 메커니즘을 검토합니다. Differential Privacy는 데이터 분석 및 데이터셋 공개에 있어 gold standard(금본위)로 여겨지며 정확한 수학적 정의를 제공합니다.

- **Performance Highlights**: Differential Privacy는 Apple, Meta, Google, LinkedIn과 같은 선도적인 기술 기업뿐만 아니라 미국 정부에서도 널리 인정받고 채택되었습니다. 이 장은 AI 응용 프로그램의 새로운 도전과제를 해결하기 위해 DP에 대한 전반적인 소개를 목표로 합니다.



### The Multiple Dimensions of Spuriousness in Machine Learning (https://arxiv.org/abs/2411.04696)
- **What's New**: 이 논문은 기계 학습(ML) 및 인공지능(AI) 연구에서의 상관관계 학습의 새로운 관점을 제시합니다. 특히, 고전적인 통계학적 스푸리어스니스(spuriousness) 정의에서 벗어난 '스푸리어스니스의 다차원성' 개념을 도입하여 미지의 데이터에 대한 일반화, 인간 유사성, 해로움과 같은 측면을 포괄합니다.

- **Technical Details**: ML 연구자들은 스푸리어스니스의 다차원성을 다루기 위해 네 가지 주요 차원인 관련성(relevance), 일반화(generalizability), 인간 유사성(human-likeness), 해로움(harmfulness)을 정의합니다. 이들 차원은 모델이 적용해야 할 상관관계를 정할 때 중요한 역할을 하며, causal과 non-causal의 이분법을 넘어서서 ML 개발의 경로에 의미 있는 영향을 미칠 수 있음을 나타냅니다.

- **Performance Highlights**: 이 논문의 논의는 AI 개발에서의 책임 있는 실천에 대한 지속적인 논쟁에 기여하며, ML 연구의 근본적인 문제를 연구 맥락에서 어떻게 협상하는지를 강조합니다.



### Reciprocal Point Learning Network with Large Electromagnetic Kernel for SAR Open-Set Recognition (https://arxiv.org/abs/2411.04693)
- **What's New**: 기존의 Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) 방법의 제한에 대한 해결책으로, Open Set Recognition (OSR) 기법을 활용하여 알려진 클래스와 미지의 클래스를 효과적으로 분류할 수 있는 새로운 방법이 제안되었습니다.

- **Technical Details**: 이 연구에서는 reciprocal point learning (RPL)에 기반한 특징 학습 프레임워크를 구성하여, 잠재적인 미지의 클래스에 대한 경계를 설정합니다. 이후 SAR 이미지에서 대형 속성 산란 중심 모델을 기반으로 컨볼루션 커널(convolutional kernels)을 설계하여 비선형 특징을 추출하는 방법을 제안했습니다.

- **Performance Highlights**: MSTAR 데이터셋에서 실시된 실험을 통해 ASC-RPL 접근 방식이 기존의 주요 방법들에 비해 우수한 성능을 보여주었습니다.



### Personalized Federated Learning for Cross-view Geo-localization (https://arxiv.org/abs/2411.04692)
Comments:
          6 pages, 2 figures, Preprint submitted to the IEEE 26th International Workshop on Multimedia Signal Processing (MMSP)

- **What's New**: 본 논문에서는 Federated Learning (FL)과 Cross-view Image Geo-localization (CVGL) 기술을 결합한 새로운 방법론을 제안합니다. 자율주행 차량의 데이터 프라이버시와 이질성 문제를 해결하기 위해 선택적으로 모델 파라미터를 공유할 수 있는 개인화된 federated learning 시나리오를 도입합니다.

- **Technical Details**: 제안한 방법론은 coarse-to-fine 접근 방식을 구현하며, 클라이언트는 지방의 특성을 반영한 자세한 특징을 보존하면서 대략적인 특성 추출기만 공유합니다. 이를 통해 각 차량의 운행 환경의 특성을 잘 반영하고, 협업 학습의 이점을 취할 수 있습니다. 또한, coarse 특징 추출기만 선택적으로 공유함으로써 데이터 전송량을 줄여 계산 효율성을 개선합니다.

- **Performance Highlights**: KITTI 데이터셋과 위성 이미지를 결합하여 전통적인 중앙 집중형 및 단일 클라이언트 학습 방식과 비교한 결과, 우리의 federated CVGL 방법이 중앙 집중식 학습과 유사한 성과를 거두면서 데이터 프라이버시를 유지하는 것으로 나타났습니다. 제안한 부분 모델 공유 전략은 전통적인 FL과 비교할 때 유사하거나 약간 더 나은 성능을 보여주며, 정확성을 유지하면서 통신 오버헤드를 크게 줄였습니다.



### AWARE Narrator and the Utilization of Large Language Models to Extract Behavioral Insights from Smartphone Sensing Data (https://arxiv.org/abs/2411.04691)
- **What's New**: 이 논문은 스마트폰에서 수집된 데이터를 체계적으로 구조화된 형태로 전환하여 개인의 활동을 서술하는 새로운 접근 방식을 제시합니다. 이를 통해 디지털 건강 분야와 심리학 분석에서 활용할 수 있는 가능성을 탐구합니다.

- **Technical Details**: AWARE Narrator 프레임워크를 통해, 스마트폰 센서 데이터(예: accelerometer, GPS, Bluetooth, 및 애플리케이션 사용 로그)를 영어 설명으로 변환하여 일일 감지 서술문을 생성합니다. 이러한 서술문은 감정 분석 및 행동 패턴 분석을 위해 대규모 언어 모델(LLMs)과 통합될 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 전통적인 데이터 분석 방법(예: 기본 데이터 피처 계산)보다 정보를 더 풍부하게 제공합니다. 다양한 센서에서 수집된 데이터가 통합되어 더욱 세밀하고 인간 친화적인 정보 제공이 가능하며, 데이터 분석의 투명성을 높이는 이점이 있습니다.



### CUIfy the XR: An Open-Source Package to Embed LLM-powered Conversational Agents in XR (https://arxiv.org/abs/2411.04671)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 최근 컴퓨터 그래픽스(computer graphics), 머신러닝(machine learning), 센서 기술(sensor technologies)의 발전은 일상 생활을 위한 확장 현실(extended reality, XR) 구성에 수많은 기회를 제공합니다. 대기업들이 저렴한 소비자용 헤드 마운트 디스플레이(head-mounted displays, HMDs)를 제공하면서 XR이 보편화될 것으로 예상됩니다. 그러나 XR에서 지능적 공간(intelligent spaces)과 자연스러운 상호작용(naturalistic interactions)을 구축하는 것은 기술 발전만큼이나 중요합니다. 이에 따라, 대규모 언어 모델(large language model, LLM)을 통한 비플레이어 캐릭터(non-player characters, NPC)와 음성인식(speech-to-text, STT) 및 음성합성(text-to-speech, TTS) 모델이 기존 NPC에 비해 자연스러운 대화형 사용자 인터페이스(conversational user interfaces, CUIs)를 제공하는 데 유리하다는 점이 강조됩니다.

- **Technical Details**: 본 논문에서는 커스터마이즈(customizable), 확장 가능(extensible), 프라이버시를 고려한 Unity 패키지인 CUIfy를 제공합니다. 이 패키지는 다양한 LLM, STT, TTS 모델과 함께 음성 기반 NPC-사용자 상호작용을 지원하며, 단일 환경 내에서 여러 LLM 기반 NPC를 지원하고 모델 간 지연(latency)을 최소화하여 사용자가 NPC와 유용하게 상호작용할 수 있도록 합니다. 또한, LLM에 접근할 수 있는 API(application programming interfaces)를 통해 로컬 장치 또는 별도의 서버에서 오픈소스 LLM을 다룰 수 있는 기능도 포함되어 있습니다.

- **Performance Highlights**: CUIfy 패키지는 LLM, STT, TTS 모델을 포함하는 파이프라인을 이용하여 음성 기반 상호작용을 가능하게 하며, 모델 간의 지연을 줄이고, 사용자와 NPC 간의 효과적인 상호작용을 제공합니다. 이와 같은 시스템은 사용자가 XR 애플리케이션에서 더 많은 흥미를 가지고 상호작용할 수 있도록 도와줍니다.



### EffiCANet: Efficient Time Series Forecasting with Convolutional Attention (https://arxiv.org/abs/2411.04669)
- **What's New**: EffiCANet은 멀티 변수 시계열 예측을 위한 새로운 Efficient Convolutional Attention Network로, 짧고 긴 의존성을 효율적으로 포착하며, 다음 변량 간의 복잡한 상호 작용을 모델링하기 위해 설계되었습니다.

- **Technical Details**: EffiCANet은 세 가지 주요 구성 요소를 통합합니다: (1) Temporal Large-kernel Decomposed Convolution (TLDC) 모듈을 통해 장기적인 시계열 의존성을 포착함과 동시에 계산 오버헤드를 줄입니다; (2) Inter-Variable Group Convolution (IVGC) 모듈을 통해 변수들 간의 복잡하고 진화하는 관계를 포착합니다; (3) Global Temporal-Variable Attention (GTVA) 메커니즘을 통해 중요 시점 및 변수 간의 특징을 우선시합니다.

- **Performance Highlights**: EffiCANet은 아홉 개의 벤치마크 데이터 세트에서 기존 최신 모델보다 MAE (Mean Absolute Error)를 최대 10.02% 줄이고, 전통적인 대형 커널 합성곱 방법에 비해 계산 비용을 26.2% 절감하는 데 성공했습니다.



### wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals (https://arxiv.org/abs/2411.04644)
Comments:
          Accepted to Machine Learning for Health (ML4H) 2024

- **What's New**: 본 논문에서는 다양한 입력 신호 집합에 대해 작동할 수 있는 새로운 통합 모델인 'wav2sleep'을 소개합니다. 기존의 수면 단계 분류 모델들이 특정 입력 신호에 대해 훈련되었던 반면, wav2sleep은 이질적인 데이터 세트를 활용하여 훈련 중 모든 입력 신호를 사용할 수 있게 설계되었습니다.

- **Technical Details**: wav2sleep 모델은 10,000개 이상의 야간 기록을 활용하여 훈련되며, 여러 폴리솜노그래피 데이터 세트로부터 수집된 신호를 사용할 수 있습니다. 이 모델은 훈련 후에 어떤 원래 입력 신호의 하위 집합을 사용할 수 있으며, 랜덤 마스킹 절차를 도입하여 테스트 시 일반성을 부여합니다.

- **Performance Highlights**: wav2sleep은 ECG, PPG, 호흡 신호와 같은 다양한 입력 조합에 대해 기존의 수면 단계 분류 모델들을 능가하는 성능을 보여주며, 특정 입력 신호에 의존하지 않고도 높은 정확도를 유지합니다.



### TAP-VL: Text Layout-Aware Pre-training for Enriched Vision-Language Models (https://arxiv.org/abs/2411.04642)
- **What's New**: 이 논문에서는 Vision-Language (VL) 모델의 OCR 정보 통합 방식을 개선하기 위해 TAP-VL이라는 새로운 방법을 제안합니다.

- **Technical Details**: TAP-VL은 OCR 정보를 별도의 모달리티로 취급하여 VL 모델에 통합하는 리프레시 접근 방식입니다. 이 과정에서 layout 정보를 포함하는 경량화된 transformer 기반 OCR 모듈을 활용하여 OCR 정보를 짧은 고정 길이 시퀀스로 압축하여 LLM에 입력합니다.

- **Performance Highlights**: TAP-VL은 다양한 선진 VL 모델에 적용하여 일관된 성능 향상을 보여주며, 특히 덴스 텍스트 이미지를 포함한 문서 이해 작업에서 4.8%의 성능 향상을 달성하였습니다. 또한, TAP-VL_Light은 FLOPs를 7배까지 줄이면서도 뛰어난 성능을 발휘합니다.



### Verification of Neural Networks against Convolutional Perturbations via Parameterised Kernels (https://arxiv.org/abs/2411.04594)
- **What's New**: 본 논문은 블러링(blurring)이나 샤프닝(sharpening)과 같은 컨볼루션(perturbations) 변형에 대해 신경망을 효율적으로 검증하는 방법을 제안합니다. 이는 특정-camera shake와 같은 변형을考慮하고, 이들을 선형적으로 매개변수화하여 적용할 수 있도록 합니다.

- **Technical Details**: 제안된 방법에서는 입력 이미지에 대해 데이터를 선형 매개변수화한 커널(kernel)을 사용하여 컨볼루션(convolution) 연산을 수행합니다. 이 방식은 신경망의 입력 변형을 효율적으로 인코딩할 수 있게 하며, 이를 통해 좀 더 견고한 검증이 가능합니다. 또한, 세분화 입력 방식을 통해 브랜치와 바운드(branch and bound) 전략을 적용하여 검증의 정밀성을 높입니다.

- **Performance Highlights**: 우리의 방법은 기존 방법들보다 유의미한 성능 향상을 보여줍니다. 표준 벤치마크에서 검증을 수행한 결과, 제안된 방법이 작은 커널 크기와 변형 강도에서 다수의 특성을 검증할 수 있는 반면, 기존 방법은 특성 검증이 불가능한 경우가 많았습니다. 본 연구의 결과는 고차원 공간에서의 변형으로 인한 어려움을 해소하고, 신경망의 견고성을 더 잘 인증할 수 있음을 입증합니다.



### Tibyan Corpus: Balanced and Comprehensive Error Coverage Corpus Using ChatGPT for Arabic Grammatical Error Correction (https://arxiv.org/abs/2411.04588)
Comments:
          17 pages, 11 figures

- **What's New**: 이번 연구에서는 아랍어의 문법 오류 교정을 위한 새로운 말뭉치인 'Tibyan'을 개발했습니다. ChatGPT를 활용하여 아랍어 문장에서의 문법 오류를 증가시키고, 이를 개선하는 기술을 적용했습니다.

- **Technical Details**: Tibyan 코퍼스는 주로 ChatGPT를 이용하여 여러 출처에서 수집된 아랍어 문장 쌍을 바탕으로 생성되었습니다. 논문의 여러 단계에서 수집된 코퍼스는 49개의 오류 유형을 포함하며, 여기에는 표기법(orthography), 형태론(morphology), 구문(syntax), 의미론(semantics), 구두점(punctuation), 합병(merge), 분할(split)이 포함됩니다.

- **Performance Highlights**: Tibyan 코퍼스는 약 600K 토큰을 포함하고 있으며, 본 연구는 아랍어 문법 오류 교정 시스템의 정확성과 견고성을 높이는 데 기여할 것으로 기대됩니다.



### On the Inherent Robustness of One-Stage Object Detection against Out-of-Distribution Data (https://arxiv.org/abs/2411.04586)
Comments:
          12 figures, 4 tables, under review

- **What's New**: 이 논문에서는 하나의 단계(object detection) 객체 탐지 모델이 일반화된 객체 인식에서 외부 배포(out-of-distribution, OoD) 데이터에서도 견고하게 작동할 수 있는 능력을 분석합니다. 특히, 사전 훈련된(pretrained) 모델을 활용한 새로운 알고리즘을 제안하여, 미리 훈련된 객체 탐지기를 재학습(retraining)할 필요 없이 알려지지 않은 객체를 탐지할 수 있는 방법을 개발하였습니다.

- **Technical Details**: 제안된 알고리즘은 사전 훈련된 모델의 특징(feature) 맵을 활용하여, 고해상도(feature maps)로 알려지지 않은 객체를 비지도 학습 방식으로 식별합니다. 이 알고리즘은 차원 축소(dimensionality reduction) 기법을 적용하여, 모델에서 추출된 특징에 대해 차원의 저주(curse of dimensionality)를 완화하는 방식으로 작동합니다. 우리의 실험에서는 알고리즘의 성능을 특정 알고리즘 구성 및 추론 신뢰성(thresholds)과의 파레토(Pareto) 절충점을 분석하여 평가했습니다.

- **Performance Highlights**: 우리의 방법은 로그 기반(logits-based) 후처리(post-hoc) OoD 방법과 성능을 비교하여 superior(우수한) 탐지 점수를 달성했습니다. 제안된 알고리즘은 최신의 OoD 접근 방식과 비교할 때 더 나은 성능을 보였으며, 사전 훈련된 모델의 정밀한 객체 탐지 능력을 향상시키는 데 기여할 것입니다.



### Multistage Fine-tuning Strategies for Automatic Speech Recognition in Low-resource Languages (https://arxiv.org/abs/2411.04573)
- **What's New**: 이 논문은 OpenAI의 Whisper 모델을 활용하여 자원이 부족한 언어에서 자동 음성 인식(ASR) 성능을 향상시키기 위한 새로운 다단계 파인튜닝 전략을 제시합니다. 특히, 말라사르 언어에 대한 ASR 모델을 만들기 위해 언어적으로 유사한 언어를 통해 모델을 점진적으로 조정하는 접근 방식을 사용합니다.

- **Technical Details**: 이 연구는 약 10,000명이 사용하는 다라비다어인 말라사르 언어를 대상으로 하며, 이 언어는 고유 스크립트의 부재와 디지털 또는 음성 데이터 리소스의 결여로 인해 기술적 지원에 큰 도전에 직면해 있습니다. 연구팀은 말라사르 음성을 타밀 스크립트로 전사하여 말라사르 코퍼스를 제작하였으며, 초기 타밀 ASR 모델을 구축한 후, 이를 말라사르 데이터에 세밀하게 fine-tuning 하였습니다.

- **Performance Highlights**: 이 다단계 파인튜닝 전략은 단일 데이터에 대한 직접적인 파인튜닝과 비교할 때 상당한 성과 향상을 보여주었으며, 최종 단어 오류율(WER)은 51.9%로 나타났고, 판별 후 처리를 통해 더욱 향상된 47.3%로 낮출 수 있었습니다. 이러한 결과는 저자원 언어의 ASR 시스템 개발에 있어 매우 효과적인 접근법임을 입증합니다.



### Impact of Label Noise on Learning Complex Features (https://arxiv.org/abs/2411.04569)
Comments:
          Accepted at Workshop on Scientific Methods for Understanding Deep Learning, NeurIPS 2024

- **What's New**: 이번 연구에서는 잡음이 있는 레이블로 모델을 사전 학습(pre-training)하는 것이 stochastic gradient descent(SGD)의 동작에 미치는 영향을 조사합니다.

- **Technical Details**: 모델은 잡음이 있는 레이블로 사전 학습된 후, 원래의 수정되지 않은 레이블을 사용하여 다시 훈련됩니다. 이러한 2단계 훈련 절차를 통해 신경망이 다양한 기능을 배우는 것을 유도합니다.

- **Performance Highlights**: 잡음이 있는 레이블로 사전 학습한 경우, 모델이 단일 특징(feature)에 의존하는 의존성을 뒤엎고 더 복잡하고 다양한 기능 세트를 학습할 수 있는 증거가 나타났습니다.



### A Generalisation of Voter Model: Influential Nodes and Convergence Properties (https://arxiv.org/abs/2411.04564)
- **What's New**: 본 논문은 전통적인 voter 모델을 일반화하여, 사회적 네트워크의 복잡한 특성을 반영하는 새로운 모델을 제안합니다. 특히, 각 노드의 연결 강도, 중립적인 의견을 가진 개인, 그리고 의견 변경에 대한 저항성을 포함합니다.

- **Technical Details**: 제안된 모델은 가중 방향 그래프를 기반으로 하며, 각 노드는 색상을 가지고 있고 색상 업데이트는 이웃 노드들의 색상에 따라 결정됩니다. 초기 색상에서 무색 노드를 허용하고, 일부 노드는 색상을 변경하지 않도록 설정하여 저항적인 개인을 모델링합니다. NP-hard한 문제로, (1−1/e) 비율의 정당한 근사 제공하는 다항 시간 알고리즘을 도출합니다.

- **Performance Highlights**: 실제 및 합성 그래프 데이터에서 제안된 알고리즘은 다른 알고리즘보다 우수한 성능을 보였습니다. 또한, 강하게 연결된 그래프의 경우 수렴 시간이 다항식으로 나타나는 경향을 보였지만, 일반적인 경우에는 지수적 수렴 시간을 요구할 수 있다는 것을 증명하였습니다.



### Constrained Latent Action Policies for Model-Based Offline Reinforcement Learning (https://arxiv.org/abs/2411.04562)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 Constrained Latent Action Policies (C-LAP)를 제안하여 오프라인 강화 학습에서 가치 과대 평가(value overestimation) 문제를 해결하고자 합니다. 기존의 모델 기반 접근 방식보다 더 나은 성능을 보여 주며, Bellman 업데이트에 대한 불확실성 패널티(uncertainty penalties) 없이 정책을 학습할 수 있습니다.

- **Technical Details**: C-LAP는 관측(observations)과 행동(actions)의 결합 분포(joint distribution)를 학습하는 생성 모델(generative model)로 구성됩니다. 이 접근 방식은 숨겨진(latent) 행동 공간(latent action space) 내에서 항상 유지되는 제약 조건(constrained objective)으로 정책 학습을 구현합니다. 이를 통해 정책 최적화를 제약된 최적화 문제로 보았으며, C-LAP는 이미지 관측을 포함한 두 개의 벤치마크에서 평가되었습니다.

- **Performance Highlights**: C-LAP는 D4RL 및 V-D4RL 벤치마크에서 실험을 통해 최첨단 방법들과의 경쟁력을 보여주었습니다. 특히 시각적 관측이 포함된 데이터셋에서 우수한 성능을 나타내며, 키워드는 Bellman 업데이트와 관련된 불확실성 추정(uncertainty estimation)을 포함하지 않고도 효율적으로 학습을 진행할 수 있음을 보여줍니다.



### Vision Language Models are In-Context Value Learners (https://arxiv.org/abs/2411.04549)
Comments:
          Project website and demo: this https URL

- **What's New**: 이번 연구에서는 Generative Value Learning (GVL)이라는 새로운 가치 추정 방식을 소개합니다. GVL은 비디오 프레임의 셔플된(separated) 순서에서 작업 진행을 예측하여, VLM의 잠재적인 의미적 및 시간적 능력을 최대한 활용하여 매우 효과적인 가치 예측을 가능하게 합니다.

- **Technical Details**: GVL은 비디오 프레임을 무작위로 섞어서 VLM에 입력하며, 자연어로 정의된 작업의 완성 비율을 예측하는 작업을 수행합니다. 이 방식은 VLM이 프레임 간의 강한 시간적 상관관계를 극복하도록 돕고, 이를 통해 의미 있는 가치 예측이 가능해집니다.

- **Performance Highlights**: GVL은 300개 이상의 다양한 실제 작업에 대해 제로샷(zero-shot) 및 몇 샷(few-shot) 예측 기능을 제공하며, 다양한 로봇 플랫폼에서 성능을 입증했습니다. 또한 새로운 평가 지표인 Value-Order Correlation (VOC)을 사용하여 예측 값이 실제 전문가 비디오와 얼마나 잘 일치하는지를 측정하였습니다.



### Meta-Reasoning Improves Tool Use in Large Language Models (https://arxiv.org/abs/2411.04535)
- **What's New**: 이번 논문에서는 Tool selECTion via meta-reasONing (TECTON)이라는 새로운 시스템을 제안합니다. TECTON은 외부 도구를 활용하여 대규모 언어 모델이 수학 추론과 같은 복잡한 작업에서의 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: TECTON은 두 단계로 구성된 시스템으로, 첫 번째 단계는 특정 작업을 해결하기 위해 맞춤화된 언어 모델 헤드를 사용하여 후보 도구를 생성하고, 두 번째 단계에서는 고정된 LLM을 사용하여 후보 도구를 재검토하고 최종 선택을 합니다. 이를 통해 기계 학습 모델의 일반화 능력을 효과적으로 활용합니다.

- **Performance Highlights**: TECTON은 다양한 수학 추론 데이터셋에서 상당한 성능 향상을 보여주는 결과를 낳았으며, 분포 내와 분포 외 모두에서 강력한 기준 모델을 초월하는 성능을 기록했습니다.



### GenJoin: Conditional Generative Plan-to-Plan Query Optimizer that Learns from Subplan Hints (https://arxiv.org/abs/2411.04525)
- **What's New**: 본 논문에서는 GenJoin이라는 새로운 학습된 쿼리 최적화기를 제안하며, 이 모델은 쿼리 최적화 문제를 생성적 작업(generative task)으로 간주하여 서브플랜 힌트(subplan hints)로부터 학습하여 기존의 최적화기보다 성능이 우수한 쿼리 플랜을 생성할 수 있다.

- **Technical Details**: GenJoin은 조건부 변분 오토인코더(conditional variational autoencoder) 아키텍처를 사용하며, 특정 조인 타입(예: nested loop join, merge join, hash join)에 제한된 서브플랜 힌트를 설정한다. 이 모델은 정밀한 쿼리 계획을 강제하는 것과 일반적인 힌트를 제공하는 것의 '황금 중간'을 제안한다.

- **Performance Highlights**: GenJoin은 PostgreSQL과 최신 방법들과 비교하여 두 개의 잘 알려진 실제 벤치마크에서 다양한 워크로드를 사용하여 상당히 일관되게 우수한 성능을 기록하였다.



### Continuous Sign Language Recognition System using Deep Learning with MediaPipe Holistic (https://arxiv.org/abs/2411.04517)
Comments:
          14 pages, 4 figures, Wireless Pers Commun

- **What's New**: 이 논문에서는 ISL(Indian Sign Language) 데이터셋을 기반으로 한 지속적인 수화 인식(SLR) 시스템을 제안합니다.

- **Technical Details**: 이 시스템은 깊은 학습(deep learning) 모델인 LSTM(Long Short-Term Memory)을 사용하여 수화 데이터의 인식을 수행합니다. 데이터셋은 MediaPipe Holistic 파이프라인을 통해 얼굴, 손, 그리고 몸의 움직임을 추적하여 생성되었습니다.

- **Performance Highlights**: 시스템은 실시간으로 수화와 제스처를 인식하며, 88.23%의 정확도로 동작합니다.



### FedDP: Privacy-preserving method based on federated learning for histopathology image segmentation (https://arxiv.org/abs/2411.04509)
Comments:
          Accepted in BIBM2024

- **What's New**: 이번 연구는 병원 간 협력적인 학습을 통해 환자 개인 정보 보호를 극대화하면서도 암 병리 이미지 데이터의 비밀성을 유지할 수 있는 연합 학습 프레임워크(Federated Learning)를 도입했습니다. 이 방법은 환자 정보 유출의 우려 없이 각 기관이 독립적으로 학습할 수 있도록 지원합니다.

- **Technical Details**: 제안된 모델은 Swin Transformer와 ConvNeXt를 기반으로 하는 이중 분기 계층 글로벌-로컬 융합 네트워크(DHUnet)를 사용하여 전체 슬라이드 이미지(WSI)의 세분화를 수행합니다. 각 기관의 로컬 모델이 훈련한 후, 변경된 모델 매개변수만 중앙 서버로 전송하여 글로벌 모델로 집계하고, 다시 로컬 장치로 분배되는 방식입니다. 차별적 프라이버시(Differential Privacy)를 통해 모델 업데이트에 노이즈를 추가하여 각 샘플의 기여도 유출을 방지합니다.

- **Performance Highlights**: FedDP 방법을 통해 모델의 Dice, Jaccard, Acc 지수가 각각 0.55%, 0.63%, 0.42%의 미미한 감소로 모델 정확성을 거의 유지하며 암 병리 이미지 데이터의 프라이버시를 효과적으로 보호할 수 있었습니다. 이는 의료 분야에서 민감한 데이터의 보호와 함께 기관 간 협업 및 지식 공유를 통한 새로운 연구 및 응용 가능성을 더욱 확장할 수 있음을 의미합니다.



### Series-to-Series Diffusion Bridge Mod (https://arxiv.org/abs/2411.04491)
- **What's New**: 본 논문에서는 시계열 예측에 있어서의 확산 모델(Diffusion Model)의 활용에 주목하며, 기존의 확산 기반 방법들을 포괄하는 포괄적인 프레임워크를 제시합니다. 이를 기반으로 역사적 시계열 데이터에서 유도한 조건과 정보를 활용하여 불확실성을 줄이고 정확도를 향상시키는 새로운 모델인 Series-to-Series Diffusion Bridge Model (S2DBM)을 제안합니다.

- **Technical Details**: S2DBM은 Brownian Bridge 프로세스를 활용하여 확산 과정의 양 끝에서 안정성을 확보합니다. 기존의 확산 모델들이 처리하는 표준 정방향 확산 과정과는 달리, S2DBM은 노이즈가 많은 입력으로 인한 불안정을 줄이며, 역사적 시계열로부터의 정보적 사전과 조건을 포함하여 예측 정확성을 높입니다.

- **Performance Highlights**: S2DBM은 다양한 실제 데이터셋(Weather, ILI, Exchange Rate 및 ETT 데이터셋)을 대상으로 한 실험에서 우수한 성능을 보여주었으며, 점대점(point-to-point) 예측 및 확률적 예측 모두에서 기존의 확산 기반 모델들과 비교하여 높은 정확도를 기록했습니다.



### Enabling Adaptive Agent Training in Open-Ended Simulators by Targeting Diversity (https://arxiv.org/abs/2411.04466)
Comments:
          NeurIPS 2024

- **What's New**: 이 연구에서는 복잡하고 개방된 시뮬레이터에서 다양한 훈련 과제를 생성하기 위한 진화적 접근법인 DIVA를 제안합니다. DIVA는 비지도 환경 설계(unsupervised environment design, UED) 방법을 바탕으로 하여, 현실적으로 사용 가능한 도메인 지식을 통합하여 보다 유연하고 일반적인 접근을 제공합니다.

- **Technical Details**: DIVA는 메타-강화 학습(meta-reinforcement learning) 환경에서 적응형 에이전트 행동을 훈련시키기 위해 설계되었습니다. 이 방법은 도메인 무작위화(domain randomization) 및 절차적 생성(procedural generation) 문제를 해결하기 위해 다양한 매개변수를 활용하여 학습합니다. DIVA는 다양한 목표 환경 ویژگی(feature)를 정의하고, 두 단계의 샘플링(Sampling)으로 훈련 과정에서의 다양성을 확보합니다.

- **Performance Highlights**: DIVA는 이전 문헌의 경쟁 기반선들을 크게 능가하는 성과를 보였습니다. DIVA의 적용은 현실적인 시뮬레이션 도메인에서의 훈련 가능성을 높이고, 보다 강인하고 능력 있는 적응형 에이전트를 생성하는데 기여할 수 있는 잠재력을 지니고 있습니다.



### Scaling Laws for Pre-training Agents and World Models (https://arxiv.org/abs/2411.04434)
- **What's New**: 이 논문은 모델 파라미터, 데이터셋 크기, 컴퓨트(compute)의 증가가 실제로 임바디드 에이전트의 성능을 어떻게 향상시키는지를 평가하는 연구입니다. 기존의 단순한 직관인 '더 크면 더 좋다'를 넘어, 전이학습에서 모델과 데이터셋 크기의 최적 거래에 대한 구체적인 통찰을 제공합니다.

- **Technical Details**: 연구는 임바디드 AI에서의 스케일링(scale) 법칙을 조사하며, 행동 클로닝(Behavior Cloning)과 세계 모델링(World Modeling)의 두 가지 사전 학습(pre-training) 목표에서의 모델과 데이터셋 크기의 최적 거래가 토크나이저(tokenizer), 작업(task), 아키텍처(architecture)에 의해 크게 영향을 받는다는 사실을 시사합니다. 또한, 고정된 오프라인 데이터셋을 기반으로 한 다음 토큰 예측(next-token prediction) 작업에서 트랜스포머(transformer) 모델을 훈련하여 성과를 확인했습니다.

- **Performance Highlights**: 발견된 스케일링 법칙은 LLMs(대형 언어 모델)와 유사한 특성이 있으며, 행동 클로닝에서는 작은 모델과 더 많은 데이터로 최적화된 거래가 나타났습니다. 이는 궁극적으로 에이전트의 성능을 개선하는 데 필요한 로드맵을 제시합니다.



### Bayesian Calibration of Win Rate Estimation with LLM Evaluators (https://arxiv.org/abs/2411.04424)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 최근 대규모 언어 모델(LLMs)을 평가자로 사용하여 텍스트 생성의 품질을 평가할 수 있는 가능성이 드러났습니다. 하지만 이러한 LLM 평가자를 단순히 사용하면 신뢰할 수 없는 결과를 초래할 수 있습니다. 이를 해결하기 위해 본 논문에서는 Bayesian Win Rate Sampling (BWRS)과 Bayesian Dawid-Skene  두 가지 보정(calibration) 방법을 제안합니다.

- **Technical Details**: 제안된 BWRS와 Bayesian Dawid-Skene 방법은 Bayesian 추론(Bayesian inference)을 활용하여 LLM 평가 결과를 기반으로 생성 언어 모델의 실제 승률을 더 정확하게 추론합니다. 이 방법들은 텍스트 생성기 간의 승률을 추정할 때 LLM 평가자와 인간 평가 간의 일치를 개선하는데 효과적입니다.

- **Performance Highlights**: 이 연구는 이야기 생성, 요약, 지침 수행 작업을 포함하는 여섯 개 데이터세트를 통해 BWRS와 Bayesian Dawid-Skene의 효과를 실증적으로 검증하였습니다. 연구 결과는 두 방법이 LLM 평가자의 승률 추정 편향을 효과적으로 줄이는 데 기여하여 NLP에서 보다 신뢰할 수 있는 자동 평가 방법으로 나아가는 가능성을 보여줍니다.



### Variational Low-Rank Adaptation Using IVON (https://arxiv.org/abs/2411.04421)
Comments:
          Published at 38th Workshop on Fine-Tuning in Machine Learning (NeurIPS 2024). Code available at this https URL

- **What's New**: 이번 연구에서는 Variational Learning이 Low-Rank Adaptation (LoRA)의 정확도와 보정(calibration) 성능을 크게 향상시킬 수 있음을 보여줍니다. Improved Variational Online Newton (IVON) 알고리즘을 사용하여 대형 언어 모델을 세부조정(finetuning) 했습니다.

- **Technical Details**: IVON은 AdamW 옵티마이저를 대체하여 Variational-Bayesian 목표를 최적화합니다. 이 알고리즘은 학습률 적응에 사용되는 스케일 벡터를 통해 posterior variance를 무료로 추정할 수 있어 구현이 용이하며, 전체 훈련 시간의 약 1%의 오버헤드만 발생합니다.

- **Performance Highlights**: Llama-2 모델의 세부 조정 결과, IVON은 AdamW에 비해 정확도가 2.8% 증가하고 Expected Calibration Error (ECE)는 4.6% 감소했습니다. 이와 함께 다른 베이지안 대안들보다도 우수한 성능을 보여주었습니다.



### Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers (https://arxiv.org/abs/2411.04403)
- **What's New**: 이번 논문에서는 inference-free sparse retriever의 검색 관련성을 향상시키기 위한 새로운 훈련 방법을 제안합니다. 특히 IDF-aware FLOPS 손실 함수와 이종 앙상블 지식 증류 프레임워크라는 두 가지 접근법을 통해 성능 개선을 목표로 하였습니다.

- **Technical Details**: IDF-aware FLOPS 손실은 토큰의 IDF 값에 따라 패널티를 조정하여, 고유한 토큰에 대해서는 적은 패널티를 부과하고 빈도가 높은 토큰에는 더 많은 패널티를 부과해 검색 관련성을 개선합니다. 이종 앙상블 프레임워크는 dense 및 sparse retrievers의 강점을 결합하여 감독 신호를 생성하는 방법입니다.

- **Performance Highlights**: BEIR 벤치마크에서 제안된 모델은 기존의 SOTA inference-free sparse 모델보다 3.3 NDCG@10 점수를 높였으며, siamese sparse retrievers와 유사한 검색 관련성을 보였습니다. 클라이언트 측 지연 시간은 BM25보다 1.1배에 불과합니다.



### A Bayesian Mixture Model of Temporal Point Processes with Determinantal Point Process Prior (https://arxiv.org/abs/2411.04397)
- **What's New**: 본 연구에서는 대규모 매개변수를 갖는 Temporal Point Processes(TPP)의 혼합 모델인 TP$^2$DP$^2$를 제안합니다. 이 모델은 결정론적 점 과정(prior: Determinantal Point Process)을 활용하여 클러스터의 다양성을 높이고, 조건부 깁스 샘플링(conditional Gibbs sampling)을 기반으로 한 효율적인 사후 추론(posterior inference) 알고리즘을 개발하였습니다.

- **Technical Details**: TP$^2$DP$^2$는 클러스터 구성 요소의 매개변수에 대한 반발적 우선 순위(prior: repulsive prior)로 DPP를 적용하며, 이를 통해 다양한 매개변수를 가진 TPP를 생성합니다. 본 연구는 선택된 매개변수에 대해 중심 매개변수와 비 중심 매개변수로 구분하고, 중심 매개변수는 클러스터 구조를 주로 결정합니다. 실험적인 업데이트 과정에서는 확률적 기울기 랜지빈 역학(stochastic gradient Langevin dynamics)을 도입하여 수렴(convergence)을 촉진합니다.

- **Performance Highlights**: 실험 결과, TP$^2$DP$^2$는 전통적인 혼합 모델보다 특히 더 적은 수의 혼합 성분을 생성하면서도 높은 다양성을 유지하는 것으로 나타났습니다. 다양한 평가 지표에서 뛰어난 성능을 기록하며, 고전적인 모델 및 신경망 기반 모델에 적용 가능하다는 것을 보여줍니다.



### Benchmarking Large Language Models with Integer Sequence Generation Tasks (https://arxiv.org/abs/2411.04372)
- **What's New**: 이 논문은 LLM(대형 언어 모델)이 Online Encyclopedia of Integer Sequences (OEIS)에서 정수 수열을 계산하는 코드를 작성해야 하는 새로운 벤치마크를 제시합니다. 이 벤치마크는 생성된 코드의 정확성과 계산 효율성을 평가하기 위해 설계되었습니다.

- **Technical Details**: 벤치마크는 ‘쉬움’과 ‘어려움’으로 분류된 500개의 정수 수열로 구성되어 있으며, o1-preview, o1-mini, GPT-4o 등 다양한 최신 LLM들을 평가합니다. 평가 기준으로는 정확성, 효율성, 그리고 룩업 테이블 사용을 방지하기 위해 도입한 자동 속임수 감지 메커니즘을 포함합니다.

- **Performance Highlights**: o1 모델 시리즈는 정확성과 속임수 비율에서 다른 모델을 능가했으나, 특히 어려운 수열에서는 여전히 어려움을 겪고 있습니다. 본 벤치마크는 LLM의 수학적 추론 및 코드 작성을 평가하는 데 의미 있는 도전 과제를 제공합니다.



### ComFairGNN: Community Fair Graph Neural Network (https://arxiv.org/abs/2411.04371)
- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)의 공정성 문제를 다루고 있으며, 특히 노드 분류(node classification) context에서 커뮤니티 수준의 공정성을 측정하는 새로운 전략을 제안합니다.

- **Technical Details**: 제안된 ComFairGNN은 지역 이웃 분포의 다양성으로 인해 발생하는 편향을 해결하기 위해 학습 가능한 코어셋 기반의 디바이징(débiasing) 함수를 사용합니다. 이 방법은 서로 다른 커뮤니티 내 동일 레이블 노드 간의 구조적 불일치(structural bias)를 완화하여 공정한 노드 표현을 생성합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에 대한 종합적인 평가를 통해 제안된 모델은 정확도와 공정성 모두에서의 효과성을 입증하였습니다.



### GaGSL: Global-augmented Graph Structure Learning via Graph Information Bottleneck (https://arxiv.org/abs/2411.04356)
- **What's New**: 본 논문에서는 Graph Information Bottleneck (GIB) 원리를 바탕으로 한 새로운 방법론, Global-augmented Graph Structure Learning (GaGSL)을 제안합니다. GaGSL은 노드 분류(task)에서 정보가 응집된 그래프 구조를 학습하는 것을 목표로 하며, 이는 성능과 강건성을 사이에서 균형을 이루어야 함을 강조합니다.

- **Technical Details**: GaGSL은 두 가지 주요 요소인 글로벌 피처 증강(global feature augmentation)과 글로벌 구조 증강(global structure augmentation)으로 구성됩니다. 이를 통해 노드 유사성에 근거한 그래프 구조의 재정의(structure redefinition)를 진행하고, 재정의된 구조는 최종적인 그래프 구조를 형성합니다. 또한, 최종 그래프 구조에 기반하여 노드 레이블과 노드 임베딩 간의 상호 정보(mutual information)를 최대화하는 과정이 포함되어 있습니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 종합적인 평가 결과, GaGSL은 기존의 최첨단 방법들과 비교했을 때 우수한 성능과 강건성을 보이며, 특히 그래프 데이터에 대한 공격을 받았을 때 그 효과가 두드러지게 나타납니다.



### Model and Deep learning based Dynamic Range Compression Inversion (https://arxiv.org/abs/2411.04337)
- **What's New**: 이 논문에서는 Dynamic Range Compression (DRC) 역전을 위한 새로운 접근법을 제안합니다. 딥 러닝(deep learning) 모델을 활용하여 DRC 파라미터를 예측하고, 이를 통해 오디오 신호를 복원하는 방식입니다.

- **Technical Details**: 제안된 방법은 먼저 혼합 구성(mixture configuration)을 식별하여 DRC 파라미터를 추정한 후, 기존의 모델 기반 DRC 역전 기법을 사용하여 원본 신호를 복원하는 과정으로 진행됩니다. DRC에 적용된 파라미터는 알 수 없는 상태에서 발생하는 문제를 해결하기 위한 방법을 모색합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 두 개의 음악 데이터셋을 기준으로 여러 최첨단(state-of-the-art) 방법들과 비교했을 때 파라미터 추정 정확성과 신호 복원 품질에서 효과적이고 강력한 성능을 보였습니다.



### Gradient Boosting Trees and Large Language Models for Tabular Data Few-Shot Learning (https://arxiv.org/abs/2411.04324)
Comments:
          FedCSIS 2024 - Data Mining Competition - 1st Place Winner

- **What's New**: 이 논문은 TabLLM (Tabular Large Language Models)의 성능을 기존의 Gradient Boosted Decision Trees (GBDT)에 비해 검토하였으며, FSL (Few-Shot Learning)에서 TabLLM이 보여주는 성능 향상이 290%에 이른다고 보고합니다. 이는 특히 적은 수의 샘플을 사용하여 노드 분할을 강제함으로써 이룬 성과입니다.

- **Technical Details**: Tabular data (TD)와 같은 형식을 가진 데이터셋에서 TabLLM을 이용한 FSL이 효과적이라고 주장하며, LightGBM과의 비교를 통해 TabLLM의 학습 방식과 이점에 대해 분석합니다. 특히 TabLLM은 자연어 표현으로 데이터를 변환하여 특정 프롬프트에 맞춰 학습하는 방식을 사용합니다.

- **Performance Highlights**: 연구에서는 TabLLM이 8개 이하의 샘플에서 유리함을 보여주며, 샘플 수가 증가함에 따라 GBDT의 성능이 유사해진다고 주장합니다. FSL은 또한 모델 다양성을 높이기 위해 유용하며, ExtraTrees와 결합했을 때 과적합에 대한 강한 저항력을 제공합니다. 이 연구는 ML 경진대회에서 1위를 기록하여 실제 적용 가능성을 입증했습니다.



### A Multilingual Sentiment Lexicon for Low-Resource Language Translation using Large Languages Models and Explainable AI (https://arxiv.org/abs/2411.04316)
Comments:
          This work is part of a PhD proposal in Information Technology at the University of Pretoria, supervised by Dr. Mike Wa Nkongolo and co-supervised by Dr. Phil van Deventer, under the Low-Resource Language Processing Lab in the Department of Informatics

- **What's New**: 이 연구는 남아프리카와 콩고 민주 공화국의 다언어 환경에서 감정 분석 및 번역의 과제를 해결하기 위해 다국어 사전을 개발했습니다. 이 사전은 프랑스어와 Tshiluba 캔 번역을 포함하여 영어, 아프리칸스어, 세페디어, 줄루어로 확대되었습니다.

- **Technical Details**: 새로 개발된 다국어 사전은 특정 언어의 감정 점수를 통합하여 감정 분류의 문화적 적합성을 향상시킵니다. 기계 학습 모델로는 랜덤 포레스트(Random Forest), 서포트 벡터 머신(Support Vector Machine, SVM), 결정 트리(Decision Trees), 가우시안 나이브 베이즈(Gaussian Naive Bayes, GNB)가 사용되어, 저자원 언어( low resource languages, LRLs)에서 감정을 예측합니다. 그중 랜덤 포레스트 모델이 특히 우수한 성능을 보여주었습니다. 또한, BERT(비편향 인코더 표현 변환기)가 맥락 기반의 감정을 높은 정확도로 예측하여 99% 정확도와 98% 정밀도를 달성했습니다.

- **Performance Highlights**: 연구 결과, 제안된 사점과 기계 학습 모델이 남아프리카와 DRC에서 LRL 절에서 번역 및 감정 분석을 크게 향상시킨 것으로 나타났습니다. BERT 모델의 예측은 설명 가능한 AI(Explainable AI, XAI)로 명확히 하여 투명성을 개선하고 감정 분류에 대한 신뢰를 높였습니다.



### Improving Bilingual Capabilities of Language Models to Support Diverse Linguistic Practices in Education (https://arxiv.org/abs/2411.04308)
- **What's New**: 본 연구는 다양한 언어 배경을 가진 학습자를 지원하기 위한 다국어 대형 언어 모델(MLLM)의 효과를 평가합니다. 특히 Spanglish와 같은 이중언어 학생 작문에 대한 LLM의 성능을 분석하고, 선행 연구에서 자연어 처리의 언어 전환(코드 전환) 능력의 한계를 다룹니다.

- **Technical Details**: 이 연구는 사전 훈련된 LLM을 사용하여 영어, 스페인어, Spanglish로 표현된 과학 및 사회 과학 개념에 대한 학생 작문을 평가합니다. 학습자 언어 및 내용 정확성을 위해 인간이 평가한 합성 데이터셋을 생성하고, Llama 3.1과 Mistral NeMo와 같은 오픈 소스 모델을 미세 조정(fine-tuning)합니다. 최종적으로 MLLM의 성능은 데이터의 언어에 따라 달라진다는 가설을 수립합니다.

- **Performance Highlights**: 미세 조정을 통해 MLLM은 세 가지 언어(영어, 스페인어, Spanglish)에서 모두 성능이 현저히 향상되었습니다. 이 연구는 이중언어 학습자 사이의 진정한 언어 실습을 지원하기 위해 MLLM의 효과를 향상시킬 수 있는 잠재력을 강조합니다.



### Robust Real-Time Mortality Prediction in the Intensive Care Unit using Temporal Difference Learning (https://arxiv.org/abs/2411.04285)
Comments:
          To be published in the Proceedings of the 4th Machine Learning for Health symposium, Proceedings of Machine Learning Research (PMLR)

- **What's New**: 본 연구에서는 환자의 긴급치료부(ICU) 사망 예측을 위한 새로운 프레임워크를 정의하고, Semi-Markov Reward Process를 사용하여 실제 시간이 불규칙하게 샘플링된 시계열 데이터에 TD 학습을 적용하는 방법을 제시합니다.

- **Technical Details**: Markov Reward Process(MRP)와 Semi-MRP를 활용하여 환자의 상태 전이 및 보상을 모델링합니다. TD 학습을 통해 실제 관찰 보상 대신 모델 예측을 사용하여 데이터의 분산을 줄이는 방식을 설명합니다.

- **Performance Highlights**: TD 학습을 이용한 모델이 전통적인 감독 학습 방법보다 긴급치료부 사망 예측에서 성능이 우수하며, 외부 데이터셋에서 검증 시 과적합(overfitting)이 적게 발생하는 것을 보여주었습니다.



### Generating Synthetic Electronic Health Record (EHR) Data: A Review with Benchmarking (https://arxiv.org/abs/2411.04281)
- **What's New**: 이 논문은 합성 전자 건강 기록(Synthetic Electronic Health Records, EHR) 데이터 생성 방법에 대한 포괄적인 검토와 벤치마킹을 제공하며, 오픈 소스 소프트웨어를 통해 실무자들에게 권장 사항을 제시합니다.

- **Technical Details**: 합성 EHR 데이터 생성 방법을 다섯 가지 주요 범주(규칙 기반, GAN 기반, 변분 자동 인코더, Transformer 기반, 확산 기반)로 분류하고, MIMIC-III 및 MIMIC-IV 데이터셋을 사용하여 성능을 평가합니다. 평가 지표는 데이터 충실도(fidelity), 하위 유용성(downstream utility), 개인 정보 보호(privaty protection), 계산 비용(computational cost)을 포함합니다.

- **Performance Highlights**: GAN 기반 방법은 MIMIC-III에서 충실도와 유용성 측면에서 경쟁력 있는 성능을 보였으며, 규칙 기반 방법은 개인 정보 보호에서 우수한 성능을 보여주었습니다. MIMIC-IV에서도 유사한 결과를 보이며, GAN 기반 방법이 기본 방법(baseline methods)보다 더 나은 충실도를 유지했습니다. 새롭게 개발된 Python 패키지 'SynthEHRella'는 다양한 방법과 평가 지표를 통합하여 추가 연구 및 평가를 용이하게 합니다.



### Bayesian Inference in Recurrent Explicit Duration Switching Linear Dynamical Systems (https://arxiv.org/abs/2411.04280)
- **What's New**: 본 논문에서는 Recurrent Explicit Duration Switching Linear Dynamical Systems (REDSLDS)라는 새로운 모델을 제안합니다. 이 모델은 rSLDS 모델에 반복적인 명시적 지속 시간 변수를 통합하고 Pólya-gamma 증강을 사용하는 추론 및 학습 방안을 제안합니다. 이를 통해 세 가지 벤치마크 데이터 세트에서 모델의 개선된 분할(capacity) 능력을 입증합니다.

- **Technical Details**: REDSLDS는 상태 공간 모델(State Space Models, SSMs)의 확장으로, Hidden Markov Models (HMMs) 및 Switching Linear Dynamical Systems (SLDS)의 현대적 버전입니다. 이 모델은 관측 불가능한 이산(hidden) 상태와 잠재적인 연속 변수를 활용하여 복잡한 동적 시스템을 모델링합니다. REDSLDS는 재귀적 명시적 상태 지속 변수(recurrent explicit state duration variable)를 포함하여 학습 과정을 용이하게 하고, 원래 rSLDS보다 더 복잡한 동적을 포착할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구에서 제안한 REDSLDS는 다수의 실험을 통해 인공 데이터 및 실제 데이터 세트에서 뛰어난 성능을 입증하였습니다. 특히, REDSLDS는 다양한 문제에 대한 정확한 세분화(segmentation)를 제공하여 기존 rSLDS 모델을 초월하는 성과를 보여줍니다.



### The Recurrent Sticky Hierarchical Dirichlet Process Hidden Markov Mod (https://arxiv.org/abs/2411.04278)
- **What's New**: 본 논문에서는 기존의 sticky HDP-HMM 및 disentangled sticky HDP-HMM을 발전시켜, 더 일반적인 모델인 recurrent sticky HDP-HMM (RS-HDP-HMM)을 제안합니다. 이 모델은 자기 지속성 확률(self-persistence probability)의 변화를 고려하여 비정상성을 해결합니다.

- **Technical Details**: RS-HDP-HMM은 데이터 세그멘테이션에서 더 나은 성능을 발휘하도록 설계되었으며, 새로운 Gibbs sampling 전략을 개발하여 모델 추론(inference)의 효율성을 향상시킵니다. 이 모델은 은닉 상태(hidden state) 간의 이동 확률을 수도 시스템(spatio-temporal systems)에 맞게 동적으로 조정합니다.

- **Performance Highlights**: RS-HDP-HMM은 synthetic 및 실제 데이터에서 disentangled sticky HDP-HMM, sticky HDP-HMM 및 HDP-HMM에 비해 성능이 우수함을 보여줍니다. 이 연구는 복잡한 데이터 세트에서 HMM의 학습을 개선하는데 기여할 것으로 기대됩니다.



### Graph neural networks and non-commuting operators (https://arxiv.org/abs/2411.04265)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 여러 그래프가 동일한 정점 집합을 가지며 공통의 정점 수준 학습 작업을 수행하는 설정을 고려합니다. 이를 통해 여러 그래프 연산자가 비가환(non-commuting)인 경우에도 GNN 모델의 일반화를 시도하였습니다. 이를 그래프-튜플 신경망(Graph-tuple Neural Networks, GtNN)으로 명명합니다.

- **Technical Details**: 이 연구는 비가환(non-commuting) 비확장적(non-expansive) 연산자의 특성을 이용하여 GtNN의 안정성(stability) 및 전이 가능성(transferability)을 다루는 수학적 이론을 개발합니다. 또한 그래프온-튜플 신경망의 극한 이론(limit theory)을 발전시키고, 이를 통해 수렴하는 그래프-튜플 시퀀스에서 모든 그래프-튜플 신경망이 전이 가능하다는 보편적인 전이 가능성 정리를 증명하였습니다.

- **Performance Highlights**: 연구 결과는 간단한 실험을 통해 이론적 결과를 입증하였으며, 훈련 절차를 통해 모델의 안정성을 검증하고 있습니다. 이 이론적 결과는 GNN의 기존 전이 가능성 정리를 여러 동시 그래프에 적용하여 확장한 것으로, 현재 GNN 분야에서 알려진 것보다 엄밀한 개선을 제공합니다.



### Object Recognition in Human Computer Interaction:- A Comparative Analysis (https://arxiv.org/abs/2411.04263)
- **What's New**: 최근의 강력한 컴퓨터와 컴퓨터 비전, 기계 학습 기술의 발전에 따라 인간-컴퓨터 상호작용(HCI) 분야에 혁신적인 변화가 이루어졌습니다. 이 논문은 사용자 얼굴 및 제스처 인식을 위한 알고리즘의 비교 분석을 통해 보다 직관적이고 효율적인 상호작용 시스템 개발을 목표로 합니다.

- **Technical Details**: 본 연구는 두 가지 주요 단계로 구성된 HCI 시스템을 제안합니다: 인증(authentication) 단계와 연속 추적(continuous tracking) 단계. 인증 단계에서 얼굴 인식 알고리즘을 활용해 사용자를 인증하며, 연속 추적 단계에서는 제스처 인식 기술을 통해 사용자의 손 제스처를 실시간으로 해석합니다. 분석에 사용된 데이터셋은 Labeled Faces in Wild와 ASL 데이터셋입니다.

- **Performance Highlights**: 여러 얼굴 인식 알고리즘(Eigen Faces, Viola-Jones Algorithm, Hog cascade + CNN, Key Point-based 등) 및 제스처 인식 알고리즘(신경망, CNN, LSTM)의 성능 비교가 이루어졌습니다. 특히, 딥 러닝 기반 기술은 전통적인 방법보다 복잡한 조명 조건 및 배경 혼잡 상황에서 더 나은 성능을 보였습니다.



### Learning Generalizable Policy for Obstacle-Aware Autonomous Drone Racing (https://arxiv.org/abs/2411.04246)
Comments:
          10 pages, 11 figures. This preprint is part of the author's this http URL. thesis supervised by Ir. Hang Yu and Dr. Ir. Christophe De Wagter, at MAVLab TU Delft. Full thesis is available at this https URL

- **What's New**: 이 연구는 드론 레이싱의 복잡한 장애물 인식 문제를 해결하기 위해 깊은 강화 학습(deep reinforcement learning)을 사용하여 일반화 가능한 정책을 개발하려는 첫 번째 노력입니다.

- **Technical Details**: 본 연구에서는 레이싱 트랙과 장애물 구성을 매 롤아웃 전에 도메인 랜덤화(domain randomization)와 병렬 경험 수집(parallel experience collection)을 통해 강화하는 방법을 제안합니다. 이를 통해 에이전트가 다양한 환경에 노출되도록 하여 내비게이션 능력을 향상시킵니다.

- **Performance Highlights**: 모의 실험을 통한 결과, 드론이 최대 70km/h의 속도로 이전에 보지 못한 복잡한 환경에서도 장애물을 피해가며 레이싱할 수 있음을 보여주었습니다.



### WiFlexFormer: Efficient WiFi-Based Person-Centric Sensing (https://arxiv.org/abs/2411.04224)
- **What's New**: WiFlexFormer는 WiFi 기반의 사람 중심 센싱을 위해 설계된 고효율 Transformer 기반 아키텍처입니다. 이 아키텍처는 기존의 CNN 기반 접근법과 비교하여 낮은 파라미터 수와 빠른 추론 시간으로 유사한 Human Activity Recognition (HAR) 성능을 달성합니다.

- **Technical Details**: WiFlexFormer는 WiFi의 Channel State Information (CSI)를 기반으로 하며, 2D 및 1D stem 모듈을 사용하여 다양한 입력 특징을 처리합니다. 입력 특징은 일반적으로 실수 값이며, 배치, 채널, 주파수, 시간 차원으로 구성됩니다. Transformer 인코더를 통해 시퀀스의 관련 부분에 주의를 기울이는 방식으로 작동합니다.

- **Performance Highlights**: Nvidia Jetson Orin Nano에서 10ms의 추론 시간으로 실시간 추론에 최적화되어 있으며, WiFlexFormer는 CS 기반의 다양한 벤치마크에서 기존 최첨단 아키텍쳐와 비교하여 더 낮은 파라미터 수로 우수한 성능을 보여줍니다.



### Equivariant Graph Network Approximations of High-Degree Polynomials for Force Field Prediction (https://arxiv.org/abs/2411.04219)
- **What's New**: 최근 개량된 동등심도(equivariant) 딥 모델들이 분자 동역학 시뮬레이션에서 원자 포텐셜(atomic potential)과 힘의 필드를 정확하게 예측할 수 있는 가능성을 보여주었습니다. 본 논문에서는 새로운 동등 심도 네트워크 PACE를 제안하고, 에지 부스터(edge booster)와 원자 군집 확장(Atomic Cluster Expansion, ACE) 기법을 이용하여 더 많은 동등 심도 다항 함수들을 근사화하는 방법론을 다룹니다.

- **Technical Details**: 이 연구에서는 구형 조화 함수(spherical harmonics)와 텐서 곱(tensor products)을 사용하여, 동등 심도 다항 함수의 분석을 확장합니다. PACE 네트워크는 SE(3) × Sn 동등 심도 다항 함수들을 더 높은 차수로 근사하는 데 유용하며, 다차원 기하학적 분포를 아우르며 강력한 일반화 능력을 보입니다.

- **Performance Highlights**: PACE는 rMD17, 3BPA, AcAc와 같은 세 가지 분자 동역학 시뮬레이션 데이터셋에서 실험을 진행하였고, 에너지와 힘의 예측에서 최첨단 성능을 기록했습니다. 특히, 다양한 온도 조건에서도 안정적인 일반화 능력을 보여주며, 기존의 방법들보다 월등한 결과를 도출하였습니다.



### Quantum Diffusion Models for Few-Shot Learning (https://arxiv.org/abs/2411.04217)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 양자 확산 모델(Quantum Diffusion Model, QDM)을 활용하여 몇 샷 학습(few-shot learning) 문제를 해결하기 위한 세 가지 새로운 프레임워크를 제안합니다: 레이블 안내 생성 추론(Label-Guided Generation Inference, LGGI), 레이블 안내 잡음 제거 추론(Label-Guided Denoising Inference, LGDI), 및 레이블 안내 잡음 추가 추론(Label-Guided Noise Addition Inference, LGNAI).

- **Technical Details**: QDM은 데이터 세트에서 훈련된 후 테스트 데이터에 대한 예측을 수행하기 위해 매개변수화된 양자 회로의 변별 최적화를 포함하여, 기계 학습의 다양한 작업을 위한 강력한 도구로 자리매김하고 있습니다. 기본적으로, QNN(Quantum Neural Network)을 활용하여 라벨이 지정된 적은 수의 데이터를 사용하여 학습을 진행하고, 이로부터 QDM을 통한 데이터 증대와 함께 두 단계(확산 및 잡음 제거)에서 함께 진행되는 두 가지 전략을 활용합니다.

- **Performance Highlights**: 실험 결과, 제안한 알고리즘은 기존의 방법들보다 현저히 뛰어난 성능을 보여주었습니다. QDDM을 활용하여 훈련 데이터 세트를 확장함으로써, QNN의 실제 데이터에 대한 정확도가 개선되었습니다.



### DiMSUM: Diffusion Mamba -- A Scalable and Unified Spatial-Frequency Method for Image Generation (https://arxiv.org/abs/2411.04168)
Comments:
          Accepted to NeurIPS 2024. Project page: this https URL

- **What's New**: 본 논문에서는 이미지 생성 작업을 위해 확산 모델의 새로운 state-space 아키텍처를 소개합니다. 이 아키텍처는 spatial (공간적) 및 frequency (주파수) 정보를 효과적으로 활용하여 입력 이미지의 지역 특성에 대한 귀납적 편향을 향상시킵니다.

- **Technical Details**: 우리는 Mamba를 기반으로 한 확산 모델을 향상시키며, 주파수 스캔을 기존의 공간 스캔 메커니즘과 통합하는 혁신적인 방법을 제안합니다. 이를 통해 지역 구성 요소의 감수성과 장거리 의존성을 강화하며, cross-attention (크로스 어텐션) 방식으로 두 정보를 동적으로 결합합니다.

- **Performance Highlights**: DiMSUM 아키텍처는 ImageNet, CelebA-HQ 및 LSUN Church와 같은 이미지 생성 기준 벤치마크에서 기존 모델들보다 우수한 FID 점수와 recall(리콜)을 기록하였으며, 더 빠른 훈련 수렴 속도를 보여줍니다.



### Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences (https://arxiv.org/abs/2411.04165)
- **What's New**: 본 연구에서는 xLSTM 아키텍처를 생물학적 및 화학적 시퀀스 모델링에 맞게 조정한 Bio-xLSTM을 제안합니다. 이는 Transformer 기반 모델의 단점을 해결하고 긴 시퀀스 처리를 용이하게 합니다.

- **Technical Details**: Bio-xLSTM은 DNA-xLSTM, Prot-xLSTM, Chem-xLSTM의 세 가지 아키텍처 변형으로 구성됩니다. 각 변형은 DNA, 단백질, 소분자(SMILES 표현) 등에 특화되어 있으며, 맥락 학습(in-context learning) 및 생성 모델링에 적합합니다.

- **Performance Highlights**: Bio-xLSTM 모델은 DNA, 단백질 및 화학 시퀀스의 생성 모델로서 우수한 성능을 발휘하며, 각 변형에서 풍부한 표현을 학습하고, 단백질 및 소분자에 대한 맥락 학습 기능을 성공적으로 수행합니다.



### Cooperation and Personalization on a Seesaw: Choice-based FL for Safe Cooperation in Wireless Networks (https://arxiv.org/abs/2411.04159)
- **What's New**: 본 연구는 Federated Learning(FL)의 유무선 통신 네트워크에서의 응용에 대해 새로운 관점을 제시합니다. 특히 FL의 개인화된 프레임워크에서 협력(cooperation)과 개인화(personalization)의 관계를 분석하고, 선택 기반(choice-based) FL 접근 방식을 제안하여 FL의 안전성과 공정성을 향상시키고자 합니다.

- **Technical Details**: 연구에서는 FL을 통해 참가자들이 데이터를 공유하지 않고도 협력적으로 모델을 학습할 수 있는 혁신적인 분산 AI 기법으로 설명합니다. 기존의 개인화된 FL 프레임워크에서 협력의 정도를 조정할 수 있는 튜너블(tunable) 협력 개념을 도입하고, 참가자들이 느끼는 안전성과 이득에 따라 협력 수준을 조정할 수 있는 유연한 선택 기반 FL 프레임워크를 정의합니다.

- **Performance Highlights**: FL의 장점으로는 개인 정보 보호, 확장성, 통신 효율성 및 강인성 등이 있으며, 기존의 방어 기법들이 FL의 취약성을 해결하기 어려움을 지적합니다. 제안된 선택 기반 FL 프레임워크는 데이터 이질성 문제를 해결하고, 공격을 받을 경우 참가자들이 개인화된 훈련에 더욱倾向하게 만들어 FL의 안전성을 높이며, 실제 사례 연구를 통해 플렉시블한 구조를 입증합니다.



### Crystal: Illuminating LLM Abilities on Language and Cod (https://arxiv.org/abs/2411.04156)
Comments:
          Published as a conference paper at COLM 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)과 코드 생성(신 코드 LLM) 간의 상호작용을 개선하기 위한 새로운 사전 훈련(pretraining) 전략을 제안했습니다. 이 전략은 자연어와 코딩 능력을 통합하여 코드 LLM을 더욱 발전시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구의 사전 훈련 과정은 두 단계로 나누어 진행됩니다. 첫 번째 단계에서는 95%의 자연어와 5%의 코드 데이터 혼합으로 시작하며, 두 번째 단계에서는 63%의 코드와 37%의 자연어로 데이터 비율을 조정하여 훈련합니다. 이러한 다단계 사전 훈련 전략은 인간의 학습 과정을 모방하여 기본적인 언어 지식을 먼저 습득한 후 코딩 기술을 개발하는 방식입니다.

- **Performance Highlights**: 결과적으로 개발된 Crystal 모델은 자연어와 코딩 성능에서 각각 Llama 2와 Code Llama와 비교되는 뛰어난 능력을 보입니다. Crystal은 1.4조 개의 토큰으로 사전 훈련되었으며, Llama 2와 Code Llama가 각각 2조 개가 넘는 토큰을 사용한 것과 비교하여 데이터 효율성이 우수한 점이 특징입니다.



### UnityGraph: Unified Learning of Spatio-temporal features for Multi-person Motion Prediction (https://arxiv.org/abs/2411.04151)
Comments:
          13pages, 12 figures. arXiv admin note: text overlap with arXiv:2411.03729

- **What's New**: 본 논문에서는 UnityGraph라는 새로운 그래프 구조를 제안하여 다중 인물의 모션 예측을 수행합니다. 이 구조는 공간-시간(spatio-temporal) 정보를 전체적으로 통합하여 모델의 일관성과 결합성을 높입니다.

- **Technical Details**: UnityGraph는 하이퍼변량 그래프(hypervariate graph) 기반의 네트워크로, 관측된 모션을 그래프 노드로 간주하고, 하이퍼엣지(hyperedges)를 통해 이들을 연결하여 공간-시간 특성을 탐색합니다. 이 접근 방식은 모션 예측 문제를 단일 그래프 문제로 재구성합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 광범위한 실험을 통해, 우리의 방법이 최신 기술의 성능을 초월함을 보여줍니다. 특히, 3DPW, CMU-Mocap, MuPoTS-3D와 같은 여러 데이터셋에서 우수한 성능을 기록하였습니다.



### Diffusion-based Auction Mechanism for Efficient Resource Management in 6G-enabled Vehicular Metaverses (https://arxiv.org/abs/2411.04139)
- **What's New**: 본 연구에서는 6G 기술을 활용하여 차량 메타버스(Vehicular Metaverses)에서의 자원 할당 문제를 해결하기 위해 수정된 제2차 입찰(MSB) 경매 메커니즘을 제안합니다. 이를 통해 지상 기지국(BS)와 무인 항공기(UAV) 간의 자원 할당 효율성을 향상시킵니다.

- **Technical Details**: 6G 지원 차량 메타버스에서는 차량 쌍둥이(Vehicle Twins, VTs)가 디지털 복제본 역할을 하며, 대규모 AI 모델 기반 증강 현실(Augmented Reality, AR) 내비게이션과 같은 고성능 VT 작업을 지원합니다. 이러한 VT 작업은 자원이 집약적이며, 지상 BS의 자원 한계로 인해 효율적인 자원 할당의 필요성이 제기됩니다. 이를 해결하기 위해 우리의 MSB 경매 메커니즘은 VT 작업의 지연(latency) 및 정확도를 고려하여 자원 할당을 최적화합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 확산 기반 MSB 경매가 기존 전통적인 경매 방식보다 자원 분배와 서비스 품질을 크게 개선하는 것으로 나타났습니다. 이로써 차량 사용자에게 더 나은 경험을 제공합니다.



### NetworkGym: Reinforcement Learning Environments for Multi-Access Traffic Management in Network Simulation (https://arxiv.org/abs/2411.04138)
Comments:
          NeurIPS (Datasets and Benchmarks)

- **What's New**: 이 논문은 모바일 디바이스에서 여러 액세스 네트워크(예: Wi-Fi, LTE, 5G)를 동시에 연결 가능하게 하는 	extit{multi-access traffic splitting}의 최적화를 다룹니다. 이를 위해 저자들은 	extit{NetworkGym}이라는 고충실도 네트워크 환경 시뮬레이터를 소개하며, 다양한 RL 기반 솔루션을 훈련하고 평가하는 데 사용됩니다.

- **Technical Details**: NetworkGym은 오픈소스 네트워크 시뮬레이션 도구(ns-3 등)를 활용한 셀프 서비스 프레임워크이며, ML 알고리즘 개발과 훈련 파이프라인을 제공합니다. 사용자는 다양한 프로그래밍 언어를 사용할 수 있고, 에이전트와 환경은 독립적으로 배포할 수 있습니다. 또한, 	extit{Pessimistic TD3 (PTD3)}라는 알고리즘을 제안하며, 이는 RL 알고리즘 성능을 개선합니다.

- **Performance Highlights**: 기존의 오프라인 RL 알고리즘들은 수작업으로 설계된 몇몇 정책들보다 평균적으로 성능이 떨어지는 경우가 많았습니다. PTD3는 가치 함수 비관주의(value-function pessimism)에 기반한 동작 제약 메커니즘을 통해 많은 최신 오프라인 RL 알고리즘들보다 뛰어난 성능을 보여주었습니다.



### Generative AI Enabled Matching for 6G Multiple Access (https://arxiv.org/abs/2411.04137)
Comments:
          8 pages,5 figures

- **What's New**: 이번 논문에서는 6G 다중접속 네트워크에서의 매칭 생성 문제를 해결하기 위해 생성 인공지능(GenAI)을 활용한 새로운 프레임워크를 제안합니다. 또한, 매칭 이론 및 GenAI 모델의 개요를 정리하고, 생성적 확산 모델(GDM)을 기반으로 한 접근법을 설명합니다.

- **Technical Details**: 논문은 매칭 문제를 다양한 분야에 적용하는 방법, GenAI 모델의 원리 및 구현 방식, 그리고 생성적 확산 모델(GDM)을 통한 매칭 생성 프레임워크를 다룹니다. GDM을 통해 특정 요구 사항에 맞는 매칭 전략을 반복적으로 생성하고, 사용자 요구와 네트워크 상태에 따라 유연하게 적응할 수 있는 방안을 탐구합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 결정 기반 의사결정 AI 접근법보다 주어진 조건 및 미리 정의된 보상에 기반하여 더 효과적인 매칭 전략을 생성하는 것으로 나타났습니다. 이로써 6G 다중접속 네트워크에서의 복잡한 문제 해결을 지원할 수 있는 잠재력을 보여줍니다.



### AmazonQAC: A Large-Scale, Naturalistic Query Autocomplete Datas (https://arxiv.org/abs/2411.04129)
Comments:
          EMNLP 2024

- **What's New**: 새로운 QAC 데이터셋 AmazonQAC이 소개되었습니다. 이 데이터셋은 Amazon 검색 로그에서 수집된 3억 9천 5백만 개의 샘플로 구성되어 있으며, 사용자 입력 접두사에서 최종 검색어에 이르는 실제 시퀀스를 포함합니다.

- **Technical Details**: AmazonQAC는 사용자 세션 ID 및 타임스탬프(_timestamp_)를 포함하여 QAC 문제의 맥락 의존적인 측면을 모델링하는 데 유용합니다. Prefix Trees, semantic retrieval 및 Large Language Models (LLMs)와 그리고 파인튜닝(finetuning) 여부에 따른 성능을 평가하였습니다.

- **Performance Highlights**: 파인튜닝된 LLM이 가장 좋은 성능을 보였으나, 최고의 시스템조차 테스트 데이터에서 이론적으로 가능한 성능의 절반에 불과하여 QAC 문제의 난이도를 시사합니다. Success@10 점수가 37로, 이는 상한선의 절반에 해당합니다.



### GS2Pose: Two-stage 6D Object Pose Estimation Guided by Gaussian Splatting (https://arxiv.org/abs/2411.03807)
- **What's New**: 본 논문에서는 CAD 모델이 없는 새로운 객체의 정확하고 견고한 6D pose estimation를 위한 GS2Pose라는 방법을 제안합니다. 이 방법은 3D Gaussian splatting을 도입하여 고품질 CAD 모델 없이도 재구성 결과를 활용할 수 있으며, 입력으로는 세분화된 RGBD 이미지만 필요합니다.

- **Technical Details**: GS2Pose는 coarse estimation과 refined estimation의 두 단계 구조를 갖습니다. 첫 번째 단계에서, polarization attention mechanism이 적용된 경량 U-Net 네트워크인 Pose-Net을 설계하였습니다. 이후, GS-Refiner라는 포즈 회귀 알고리즘을 통해 coarse pose를 세밀하게 재조정합니다. GS-Refiner는 Lie algebra를 활용하여 3DGS 모델의 파라미터를 선택적으로 업데이트하여 환경에 적응하며, 조도 변화와 같은 간섭에 대한 견고함을 강화합니다.

- **Performance Highlights**: GS2Pose는 LineMod 데이터셋에서 수행된 실험을 통해 유사 알고리즘과 비교하여 높이 경쟁력 있는 결과를 도출하였습니다. 특히, 정확도, 추론 속도 및 계산 자원 효율성에서 상당한 이점을 보여줍니다.




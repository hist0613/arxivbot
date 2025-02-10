New uploads on arXiv(cs.CL)

### Scalable-Softmax Is Superior for Attention (https://arxiv.org/abs/2501.19399)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 연구에서는 Transformer 기반 언어 모델에서의 Softmax 기능의 한계를 극복하기 위한 새로운 접근법인 Scalable-Softmax(SSMax)를 제안합니다. 기존 Softmax는 입력 벡터 크기가 증가함에 따라 확률 분포가 평탄해짐으로써 모델의 긴 컨텍스트에 대한 일반화 능력을 저하시킵니다. SSMax는 이러한 문제를 해결하며, 기존의 Transformer 구조에 원활하게 통합될 수 있습니다.

- **Technical Details**: Scalable-Softmax(SSMax)는 입력 벡터를 확률 분포로 변환하는 과정에서 입력 벡터의 크기에 의존하는 방식으로 설계되었습니다. 이를 통해 대량의 입력 데이터에서도 주의 축소(attention fading)를 방지할 수 있습니다. SSMax는 학습 가능한 스칼라 매개변수인 s와 b를 이용해 각 레이어와 헤드에 독립적으로 적용되며, 다양한 컨텍스트 크기에 적응할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: SSMax를 사용하는 모델은 사전 훈련 동안 손실 값을 더 빠르게 감소시키며, 긴 컨텍스트와 핵심 정보 검색 작업에서 상대적으로 월등한 성과를 보입니다. 실험 결과, 전통적인 Softmax를 대체한 SSMax는 모델이 긴 컨텍스트에서도 고유한 토큰에 집중할 수 있도록 돕는 것으로 나타났습니다. 특히, 사전 훈련이 완료된 모델에서도 Softmax를 SSMax로 교체하여 일정 수준의 개선 효과를 경험할 수 있음을 보였습니다.



### s1: Simple test-time scaling (https://arxiv.org/abs/2501.19393)
Comments:
          46 pages (9 main), 10 figures, 14 tables

- **What's New**: 최근 언어 모델의 성능 향상은 대규모 자기 지도(Self-Supervised) 사전 학습을 통한 학습 시간의 계산(resource) 증가에 크게 의존하고 있습니다. 이에 따라, 테스트 타임 스케일링(Test-time scaling)이라는 새로운 접근 방식이 등장했습니다. 연구진은 1,000개의 질문을 포함한 소규모 데이터셋(s1K)을 통해 간단한 검증 방법을 사용하여 강력한 추론 성능을 달성하는 테스트 타임 스케일링 방법론을 제안하고 있습니다.

- **Technical Details**: 연구진은 질문의 난이도, 다양성 및 품질을 기준으로 1,000개의 질문과 그에 따른 추론 추적(reasoning trace)을 포함하는 데이터셋 s1K를 구성하였습니다. 또한, budget forcing 기법을 통해 모델의 사고 프로세스를 조정하며 더 많은 계산 자원을 테스트 시간에 할당할 수 있도록 하였습니다. 이를 통해, Qwen2.5-32B-Instruct 모델을 s1K 데이터셋에서 supervised fine-tuning(SFT)하여 o1-preview 모델보다 최대 27% 향상된 성능을 보여주었습니다.

- **Performance Highlights**: s1-32B 모델은 테스트 타임 스케일링을 통해 Qwen2.5-32B-Instruct 모델의 성능을 극대화하였고, AIME24 챌린지에서 성능을 50%에서 57%로 향상시켰습니다. 연구진은 s1K 데이터셋의 중요성을 강조하며, 무작위 선택이나 길이가 긴 추론 추적을 기준으로 선택한 샘플들은 성능에 부정적인 영향을 미친다고 언급하였습니다. 최종적으로, 연구진은 데이터 선택의 섬세한 과정을 통해 테스트 타임 스케일링을 최적화하는 방법론을 제안하고 있습니다.



### TableMaster: A Recipe to Advance Table Understanding with Language Models (https://arxiv.org/abs/2501.19378)
- **What's New**: 이 논문에서는 TableMaster라는 새로운 프레임워크를 제안하여 언어 모델(LM)의 테이블 이해 능력을 개선하고자 합니다. 이 연구는 테이블 데이터의 복잡한 특성으로 인해 발생하는 네 가지 주요 도전에 집중하고 있으며, 이를 해결하기 위한 다양한 솔루션을 통합한 접근 방식을 설명합니다. TableMaster는 테이블 내용 추출 후 세밀한 의미론적 맥락을 추가하는 과정과 적응적 추론(adaptive reasoning) 기법을 포함하여 질의에 맞춰 텍스트적 및 기호적 추론을 조정합니다.

- **Technical Details**: 테이블은 2차원 관계형 데이터를 효율적으로 나타낼 수 있는 독특한 형식을 가지고 있으며, 이는 기존의 선형 텍스트와 대비되는 구조입니다. 기존의 언어 모델들은 이러한 테이블 데이터를 처리하는 데 최적화되어 있지 않아서, 테이블 기반 질문 응답 및 사실 검증과 같은 다운스트림 작업에서의 성능이 제한적입니다. 논문에서는 LMs의 테이블 이해에서 발생하는 데이터 위치 찾기, 테이블 의미의 결함, 수치적 추론의 부정확성, 기호적 추론의 의미적 유연성 부족 등의 도전을 명시하고, 이에 대한 해결책을 제안합니다.

- **Performance Highlights**: TableMaster는 WikiTQ 데이터셋에서 GPT-4o-mini 모델을 사용하여 78.13%의 정확도를 달성하여 현재까지의 기초 성능을 초과하는 성과를 보여주었다. 연구자들은 또한 TableMaster의 전반적인 성능이 세 가지 테이블 이해 데이터셋(WikiTQ, TabFact, FetaQA)에서 우수하다는 점을 강조하며, 다양한 실험과 상세한 분석을 통해 발견된 내용을 증명하고 있습니다.



### Do Large Multimodal Models Solve Caption Generation for Scientific Figures? Lessons Learned from SCICAP Challenge 2023 (https://arxiv.org/abs/2501.19353)
Comments:
          Accepted to TACL 2025

- **What's New**: SCICAP 데이터셋이 2021년에 출시된 이후, 연구자들은 과학적 도표에 대한 캡션 생성에서 상당한 발전을 이루었다. 2023년 첫 SCICAP 챌린지가 열리면서 전 세계의 팀들이 확장된 SCICAP 데이터셋을 사용해 다양한 유형의 도표에 대한 캡션 생성 모델을 개발하도록 초대하였다. 연구 결과, 전문 편집자들은 GPT-4V가 생성한 캡션을 모든 다른 모델과 저자들의 원본 캡션보다 선호했다.

- **Technical Details**: 이 논문은 2023년 SCICAP 챌린지에 대한 개요와 데이터, 과정, 우승 팀 및 모델에 대한 상세한 정보를 제공한다. 구체적으로, 8개 도메인과 5개 도표 유형의 476,389개의 단일 패널 도표를 포함하는 확장된 SCICAP 데이터셋을 바탕으로 팀들은 캡션 생성 모델을 개발했다. 자동 및 인간 평가를 통해 GPT-4V와 다른 오픈 LMM들(대형 다중모달 모델)의 성능을 비교하였다.

- **Performance Highlights**: 인간 평가 결과, 기술적 학술 작문에 익숙한 세 명의 전문 편집자는 GPT-4V가 생성한 캡션을 모든 다른 모델의 결과보다 명확히 선호하였다. 이러한 주요 발견에 따라, 우리는 편집자들이 GPT-4V를 선호하는 이유를 조사하였다. 우리의 결론은 현재 LMMs가 과학적 도표 캡션 생성 문제를 완전히 해결하지는 못했지만, 많은 개선이 이루어졌음을 보여주었다.



### Homogeneity Bias as Differential Sampling Uncertainty in Language Models (https://arxiv.org/abs/2501.19337)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)과 비전-언어 모델(VLM)이 특정 소외 집단을 더 균일하게 표현하는 이질성 편향(homogeneity bias)의 원인을 탐구하고 있습니다. 특히, 텍스트 생성 시 소외 집단에 대한 토큰 샘플링의 확률 분포가 더 결정론적이라는 가설을 세웠습니다. 이 연구는 이전의 결과와 차별화된 접근 방식을 제시하며, 다양한 모델 간의 이질성 편향 기전의 차이를 규명하고자 하였습니다.

- **Technical Details**: 연구에서는 엔트로피(entropy), 당황도(perplexity), 구별 확률(probability of differentiation) 등 세 가지 확률 분포의 불확실성을 측정하는 방법을 사용하였습니다. 이들은 AI 연구에서 분포적 불확실성을 평가하는 데 활용되어 왔습니다. 실험에는 GAN 생성 얼굴 데이터베이스(GANFD)를 사용하여 인종 및 성별에 따른 얼굴 자극을 선택하고, 모델에 대한 프롬프트 설계 및 확률 분포 분석 방법론을 구체적으로 설명하였습니다.

- **Performance Highlights**: GPT-4 Turbo 및 Llama-3.2 모델에서 마진 그룹에 대한 텍스트 생성 과정에서 비교적 낮은 불확실성 지표가 관찰되었습니다. 연구 결과는 특정 모델에서 이질성 편향이 확률 분포에 암호화되어 있음을 보여주지만, 모든 VLM에 일반화될 수 없음을 제시합니다. 이는 AI 모델 내에서 이질성 편향의 존재가 여러 기전의 영향을 받을 수 있음을 시사합니다.



### Reward-Guided Speculative Decoding for Efficient LLM Reasoning (https://arxiv.org/abs/2501.19324)
Comments:
          17 pages

- **What's New**: 이 논문에서는 Reward-Guided Speculative Decoding (RSD)라는 새로운 프레임워크를 소개하여 대형 언어 모델(LLM)의 추론 효율성을 개선하는 데 중점을 두고 있습니다. RSD는 경량의 드래프트 모델과 더 강력한 목표 모델을 결합하며, 높은 보상을 주는 출력을 우선시하기 위해 통제된 편향을 통합합니다. 이는 기존의 스펙큘레이티브 디코딩 방식과 차별화되며, computational cost와 output quality 간의 최적의 균형을 유지하는 방법을 제시합니다.

- **Technical Details**: RSD는 드래프트 모델이 생성한 후보 출력 단계들을 평가하기 위해 프로세스 보상 모델을 사용합니다. 이 모델은 각 디코딩 단계에서 동적으로 목표 모델을 호출할지를 결정하여, 계산 비용과 출력 품질 간의 trade-off를 최적화합니다. RSD는 작업 흐름을 수동 조정하여 불필요한 계산을 줄이고, 기존 추론 접근 방식을 넘어서는 품질을 제공합니다.

- **Performance Highlights**: 여러 도전적인 추론 벤치마크에서 RSD의 효율성은 현실적으로 4.4배 더 적은 FLOPs를 요구하는 등의 놀라운 효율을 보여주며, 평균적으로 3.5점 향상된 정확도를 기록했습니다. 이러한 결과는 RSD가 리소스가 많이 소모되는 환경에서 LLM을 배포하기 위한 강력하고 비용 효율적인 접근법임을 강조합니다.



### LLM-based Affective Text Generation Quality Based on Different Quantization Values (https://arxiv.org/abs/2501.19317)
- **What's New**: 이 논문은 언어 생성과 이해에 탁월한 능력을 가진 대규모 언어 모델(LLMs)의 정량화(Quantization)와 감정 텍스트 생성 간의 상충 관계를 탐구합니다. 이러한 정량화 기술은 모델의 메모리 사용량을 줄이면서 기능을 지속하고, 저비용 하드웨어에서 AI 시스템의 접근성을 높이는 가능성을 제시합니다. 또한, 이 연구는 감정 분류기와 시드 프롬프트를 통해 다양한 정량화 설정에서 생성된 텍스트 품질을 평가합니다.

- **Technical Details**: 연구에서는 감정 조건화 텍스트 생성을 위해 8, 16 및 32 비트의 세 가지 정밀도 설정을 사용하는 여러 대규모 언어 모델을 시험하였습니다. Llama 2 Chat와 Mistral 7B 모델이 활용되었으며, 각 모델은 정량화 및 GPU RAM 활용도, 텍스트 품질 간의 상관 관계를 평가했습니다. 특히 자동 평가 방법을 사용하여 생성된 텍스트의 감정 분류 성능을 측정하였습니다.

- **Performance Highlights**: 실험 결과, 정량화는 메모리 사용량을 평균 76% 절감하는 효과를 보였으나, F1 점수에서 대형 모델의 경우 최대 10포인트 감소가 관찰되었습니다. 반면, 소형 모델에서는 최대 10포인트 증가가 발생했습니다. 텍스트 품질 측면에서 볼 때, 낮은 정량화 수준에서는 대형 모델이 소형 모델보다 나은 성능을 보여주어, 메모리 사용량을 유사하게 유지하면서도 감정 텍스트 생성을 효율적으로 수행하고 있다는 것을 나타냅니다.



### Reverse Probing: Evaluating Knowledge Transfer via Finetuned Task Embeddings for Coreference Resolution (https://arxiv.org/abs/2501.19316)
- **What's New**: 본 연구는 전통적인 probing 기법을 재구성하여 간단한 출처(source) 작업에서 복잡한 대상(target) 작업으로의 지식 이전을 평가합니다. 데이터 세트 기반으로 다수의 간단한 작업의 임베딩을 활용하여 단일 복잡한 작업에 대한 효용성을 탐구했으며, 특히 coreference resolution(공명 참조 해결) 작업을 중점적으로 다뤘습니다. 연구 결과, 서로 다른 작업 임베딩이 coreference resolution 작업에 대한 유용성이 크게 다르게 나타났으며, semantic similarity(의미 유사성) 작업이 가장 효과적임을 발견했습니다.

- **Technical Details**: 연구는 간단한 출처 작업에 대한 여러 모델의 임베딩을 조합하여 복잡한 대상 작업으로 이전하는 프레임워크를 제안합니다. 이를 통해.embedding extraction(임베딩 추출) 및 embedding aggregation(임베딩 집합) 과정을 거쳐 최종적으로 coreference resolution 모델에 적합한 임베딩을 생성합니다. 네트워크의 중간 레이어에서 추출된 임베딩이 최종 레이어보다 성능이 뛰어난 경우가 많으며, 다양한 작업의 임베딩을 조합하여 더 나은 결과를 이끌어낼 수 있음을 보여주고 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 여러 작업에서 얻은 임베딩을 조합할 때 coreference resolution 작업의 성능이 지속적으로 향상된다는 것을 확인했습니다. 특히, attention-based aggregation(주의 기반 집계)을 활용했을 때 성능 향상이 두드러지게 나타났습니다. 이러한 발견은 작업 별 표현(task-specific representations)과 복잡한 다운스트림 작업에 대한 적응 가능성 사이의 관계를 조명하며, 향후 임베딩 수준의 작업 이전에 대한 연구를 촉진할 수 있는 기초 자료가 됩니다.



### An Efficient Approach for Machine Translation on Low-resource Languages: A Case Study in Vietnamese-Chines (https://arxiv.org/abs/2501.19314)
Comments:
          Technical report of VLSP 2022 NMT; The first two authors contributed equally to this work

- **What's New**: 이 논문에서는 대용량의 고품질 병렬 데이터가 부족한 저자원 언어인 베트남어-중국어 기계 번역에 관한 새로운 접근법을 제안합니다. 제안된 방법은 다국어 사전 학습 언어 모델인 mBART의 장점을 활용하여, 단일 언어 말뭉치를 이용해 번역 모델의 성능을 향상시킵니다. 특히, TF-IDF 기법을 사용하여 병렬 데이터와 가장 관련이 깊은 단일 언어 문장을 선택하고, 이를 통해 증강된 훈련 데이터를 생성합니다.

- **Technical Details**: 기계 번역 모델은 Sutskever et al. (2014)와 같은 인코더-디코더 구조를 기반으로 하며, 사전 학습된 다국어 모델 mBART-50을 사용하여 훈련합니다. 기존의 병렬 데이터에서 미세 조정된 후, TF-IDF를 통해 선택된 문장을 기반으로 새로운 병렬 데이터 세트를 생성하여 모델 훈련에 사용합니다. 이 과정에서는 비트 전송(back-translation) 기법을 사용하여 단일 언어 데이터로부터 증강된 데이터를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 transformer 모델보다 8% 향상된 성능을 보여줍니다. 또한, 생성된 증강 데이터셋은 모델의 전체 성능을 더욱 끌어올리는 효과를 발휘했습니다. 모델의 성능은 BLEU 스코어를 통해 측정되었으며, 여러 실험에서 최고의 스코어를 받은 모델이 최종 성능 평가에 사용되었습니다.



### Beyond checkmate: exploring the creative chokepoints in AI tex (https://arxiv.org/abs/2501.19301)
Comments:
          18 pages, single columns, under review at Nature Machine Intelligence

- **What's New**: 이번 연구는 인간 텍스트와 AI 텍스트 간의 미묘한 차이를 언급하며, 기존의 AI 텍스트 탐지 연구에 비해 상대적으로 덜 탐색된 영역을 다룹니다. 특히, 텍스트의 구성 요소(예: 서론, 본문, 결론)에 따라 인간과 AI 텍스트 간의 차이를 분석하고, 이러한 차이들이 창의적 보조 도구로서 AI의 잠재력을 평가하는 데 미치는 영향을 조명합니다. 연구는 체스 게임의 구조와 비유를 사용하여 각 텍스트 구성 요소에서 확인된 차이를 구체적으로 설명합니다.

- **Technical Details**: 연구에서 뉴스 기사, 이메일, 수필의 세 가지 도메인에서 AI 텍스트와 인간 텍스트 간의 차이를 분석했습니다. 여러 텍스트 세그먼트 간의 특징을 비교하기 위해 데이터셋 구축, 텍스트 세분화, 특징 추출 및 통계적 분석 절차를 적용했습니다. 본문 세그먼트는 길이가 길어 AI 텍스트의 일관성을 높일 수 있지만, 단어 다양성 및 맥락 적합성과 같은 특정 특성에서는 여전히 인간 텍스트와의 차이가 두드러지며, 이러한 분석 결과는 해당 논문 내에서 제시된 표를 통해 요약됩니다.

- **Performance Highlights**: 검출 성능에 대한 결과는 본문 세그먼트에서 AI 텍스트 탐지의 효과가 더 높다는 것을 시사하며, 이는 탐지기가 AI 텍스트와 인간 텍스트의 차이를 구분하는 데 있어 본문 세그먼트가 더 두드러진다는 것을 보여줍니다. 또한, 텍스트 길이에 따른 탐지 성능 차이를 분석한 결과, 길이가 길어질수록 잘 탐지되는 경향을 보였습니다. 최종적으로 이 연구는 AI 텍스트 탐지의 발전 방향을 제시하고, 세그먼트 간의 특징 변동성이 인간과 AI 텍스트 간의 차별화된 지표로서 유망하다고 평가하였습니다.



### Pheromone-based Learning of Optimal Reasoning Paths (https://arxiv.org/abs/2501.19278)
- **What's New**: 본 논문에서는 Ant Colony Optimization-guided Tree of Thought (ACO-ToT)라는 새로운 알고리즘을 제안합니다. 이 알고리즘은 대규모 언어 모델(LLMs)과 개미 군집 최적화(ACO)를 결합하여 복잡한 문제에 대한 최적의 추론 경로를 효율적으로 발견합니다. ACO-ToT는 생물학적 시스템의 Hebbian 학습에서 영감을 받아 생성적으로 조정된 LLM '개미'들이 중앙 집중형 사고 나무를 통해 성과를 기록하며 경로 선택을 최적화합니다.

- **Technical Details**: ACO-ToT 알고리즘은 유전자형 LLM을 '개미'로 활용하며, 이들이 높은 품질의 추론을 위해 가상 페로몬을 분산시킵니다. 페로몬 농도는 경로 선택에 영향을 미치며, 이를 통해 탐색과 착취 간의 균형을 유지합니다. 이 알고리즘은 전문가 혼합(Mixture of Experts) 기반의 점수 부여 함수를 사용하여 전체 추론 경로를 평가하며, 반복 간 생산적인 경로에 대한 페로몬이 강화됩니다.

- **Performance Highlights**: GSM8K, ARC-Challenge 및 MATH 데이터셋에 대한 실험을 통해 ACO-ToT의 평균 절대 정확도가 기존의 체인 오브 씽킹 최적화 방법보다 16.6% 향상된 것을 보여줍니다. 이러한 결과는 생물학적 영감을 받은 집단 검색 메커니즘을 LLM 추론에 통합하면 추론 능력을 크게 향상시킬 수 있음을 시사합니다.



### VisualSpeech: Enhance Prosody with Visual Context in TTS (https://arxiv.org/abs/2501.19258)
- **What's New**: 이 논문에서는 Text-to-Speech (TTS) 합성을 개선하기 위해 시각적 정보(visual information)를 활용하는 새로운 접근 방식을 소개합니다. 특히, 기존의 텍스트 입력만으로는 얻기 어려운 프로소디(prosody)의 변화를 이끌어낼 수 있는 비주얼 요소의 중요성을 강조합니다. 저자들은 VisualSpeech라는 모델을 제안하여 시각적 및 텍스트 정보를 통합함으로써 더 자연스러운 음성을 생성하는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 VisualSpeech 모델은 FastSpeech2를 기반으로 하여 텍스트 인코더(text encoder), 멜 디코더(Mel decoder), 그리고 새로운 모듈인 시각 인코더(visual encoder)와 텍스트-비주얼 융합 모듈(visual-text fusion module)을 포함합니다. 이 모델은 시각적 특징을 추출한 후 이를 텍스트 특징과 결합하여 프로소디 예측을 향상시킵니다. 시각 인코더는 텍스트 인코더와 유사한 구조를 가지며, 크로스 어텐션(cross-attention) 기법을 사용하여 비주얼 특징과 텍스트 특징 간의 상호작용을 캡처합니다.

- **Performance Highlights**: 모델의 성능은 Condensed Movies Dataset (CMD)을 사용하여 평가되었습니다. CMD는 3,605편의 영화로 구성된 대규모 비디오 데이터셋으로, 다양한 프로소디 및 비주얼 데이터를 포함합니다. 결과적으로, VisualSpeech 모델은 텍스트와 시각 정보를 결합함으로써 생성된 음성의 자연스러움과 정확성을 현저히 향상시켰다는 것을 보여줍니다.



### Improving the Robustness of Representation Misdirection for Large Language Model Unlearning (https://arxiv.org/abs/2501.19202)
Comments:
          12 pages, 4 figures, 1 table

- **What's New**: 이 논문에서는 Representation Misdirection (RM) 방법이 모델의 강건성을 저하시킨다는 점을 밝히고 있습니다. 이는 단일 비적대적 forget-token이 있는 retain-query에서도 모델이 부적절하게 행동하게 만든다는 것입니다. 이를 통해 RM 방법들이 어떻게 백도어 공격(backdoor attack)과 방어(defense) 문제로 재구성될 수 있는지를 제안하고 있습니다.

- **Technical Details**: 기존 RM 방법은 forget-token이 retain-query에 활성화될 때 백도어 트리거(backdoor trigger)로 작용하여 모델의 행동을 방해하는 방식으로 작동합니다. 이 논문은 Random Noise Augmentation (RNA)이라는 새로운 접근 방식을 제안하여 모델의 forget-token에 대한 민감도를 줄이고, RM 방법의 강건성을 개선하고자 합니다. RNA는 모든 retain-query의 representation에 독립적인 가우시안 노이즈를 추가하여 RM 모델의 부작용을 최소화합니다.

- **Performance Highlights**: RNA는 RM 방법의 강건성을 크게 향상시키는 것으로 실험적으로 입증되었습니다. 연구는 또한 RNA가 RM 모델의 unlearning 성능 역시 개선함을 보여줍니다. 이 논문은 LLM의 안전성과 개인 정보 보호를 위한 강력한 MU(Machine Unlearning) 알고리즘 개발의 필요성을 강조하고 있습니다.



### Efficient Reasoning with Hidden Thinking (https://arxiv.org/abs/2501.19201)
Comments:
          Preprint version

- **What's New**: 최근 MLLMs(Multimodal Large Language Models)의 인기가 높아짐에 따라, Chain-of-Thought (CoT) 추론을 활용하여 복잡한 문제 해결 능력을 향상시키려는 연구가 증가하고 있습니다. 본 논문에서는 $	extbf{Heima}$라는 효율적인 추론 프레임워크를 제안하고, 이를 통해 CoT를 숨겨진 잠재 공간에서 활용하여 텍스트 표현의 장황함을 피하고자 했습니다. Heima Encoder는 각 CoT를 단일 사고 토큰(single thinking token)으로 압축하여 생성되는 토큰 수를 줄이고, 이를 통해 MLLMs의 효율성을 높이고 있습니다.

- **Technical Details**: Heima 프레임워크는 CoT를 compact hidden representations로 인코딩하는 Heima Encoder를 포함하고 있습니다. 이 과정에서 담기 언어 모델(LLM)을 사용하여 사고 토큰을 생성하며, 이들 토큰은 Heima Decoder를 통해 다양한 길이의 텍스트로 해석됩니다. 이러한 장점을 통해, Heima Encoder는 텍스트 CoT 대신 사고 토큰을 생성하여 추론 프로세스를 가속화함으로써 효율성을 크게 높입니다.

- **Performance Highlights**: 실험 결과, Heima 모델은 기존의 MLLM에 비해 생성 효율성을 높이면서도 제로-샷(zero-shot) 작업 정확도를 빈틈없이 유지하거나 심지어 강화하는 것으로 나타났습니다. Heima Decoder에 의해 재구성된 추론 과정은 원래의 CoT와 밀접하게 일치하여, 이 접근 방식의 견고성과 해석 가능성을 입증합니다. 마지막으로, 기존의 효율적인 기술(KV cache optimization 및 flash attention)과도 호환되며, Heima는 MLLM을 위한 첫 번째 추론 가속 프레임워크로 자리잡을 가능성이 높습니다.



### Mixed Feelings: Cross-Domain Sentiment Classification of Patient Feedback (https://arxiv.org/abs/2501.19134)
Comments:
          Accepted for NoDaLiDa / Baltic-HLT 2025

- **What's New**: 이번 연구는 공공 건강 분야에서 환자 피드백을 분석하여 의사결정자들이 서비스 평가에 도움을 줄 수 있는 가능성을 탐구합니다. 특히 일반의와 정신 건강 서비스에 대한 환자 설문조사에서 제공된 자유 텍스트 코멘트에 초점을 맞추고, 이를 통해 데이터 부족 문제를 완화하기 위해 다양한 도메인에서 수집된 리뷰를 활용하고 있습니다.

- **Technical Details**: 이 연구에서는 NorPaC와 NoReC의 두 개의 데이터셋을 사용하여 감정 분석을 수행합니다. NorPaC 데이터셋은 7693개의 문장으로 구성된 환자의 피드백을 포함하고 있으며, 이는 일반의와 정신 건강 제공자에 대한 코멘트입니다. NoReC 데이터셋은 다양한 주제의 전문가 리뷰로 구성되며, 두 데이터셋 모두 긍정적, 부정적, 혼합적, 중립적의 네 가지 감정 클래스가 주석 처리되어 있습니다.

- **Performance Highlights**: 모델 평가 결과, NorBERT3와 NorT5 시리즈와 같은 최신 모델이 Cross-domain 및 In-domain 설정에서 모두 높은 성능을 보였습니다. SVM 기반 모델을 포함한 다양한 기계학습 모델이 실험되었으며, 전반적으로 F1-score 성능은 22%에서 23%의 랜덤 베이스라인을 초과하여 유의미한 결과를 나타냈습니다. 이는 도메인 간 효과를 탐구하는 데 있어 흥미로운 통찰력을 제공합니다.



### Improving Low-Resource Sequence Labeling with Knowledge Fusion and Contextual Label Explanations (https://arxiv.org/abs/2501.19093)
- **What's New**: 본 연구에서는 LLM 기반의 지식 강화 워크플로우와 스팬 기반의 KnowFREE 모델을 결합한 새로운 프레임워크를 제안했습니다. 이는 저자원이면서 도메인 특화된 시퀀스 레이블링의 도전에 효과적으로 대응하기 위한 접근법입니다. 이러한 방법을 통해 모델의 문맥 이해를 풍부하게 하고, 세맨틱 바이어스를 완화하는 데 기여하고자 합니다.

- **Technical Details**: KnowFREE 모델은 중첩 엔티티 추출을 지원하며, 로컬 다중 헤드 어텐션 모듈을 통해 확장 레이블 기능을 통합합니다. 이 모델은 추론 시 외부 지식에 의존하지 않고 풍부한 문맥 정보를 학습할 수 있도록 설계되었습니다. 또한, 설명 프롬프트를 활용하여 모델이 특정 컨텍스트 내 목표 엔티티에 대한 정교한 설명을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 본 프레임워크는 여러 중국의 도메인 특화 시퀀스 레이블링 데이터 세트에서 최첨단 성능을 달성했습니다. 이는 저자원 환경에서의 도전 과제를 효과적으로 해결할 수 있음을 보여줍니다. 본 연구는 non-entity 기능의 활용을 극대화하고, 모델의 문맥 이해를 향상시키는 두 가지 주요 전략을 채택한 점에서도 주목할 만합니다.



### On the Impact of Noise in Differentially Private Text Rewriting (https://arxiv.org/abs/2501.19022)
Comments:
          19 pages, 3 figures, 9 tables. Accepted to NAACL 2025 (Findings)

- **What's New**: 이 논문에서는 텍스트 재작성에 있어서 노이즈의 영향을 탐색하기 위해 새로운 문장 삽입 프라이바타이제이션(Privatization) 기법을 도입했습니다. 이 기법은 기존의 차등 프라이버시(Differential Privacy, DP) 방법과 비교할 수 있는 비DP 방식도 포함하여, 텍스트 재작성에서 유용성을 보존하면서도 비공식적인 개인 정보 보호를 기대할 수 있도록 설계되었습니다.

- **Technical Details**: 차등 프라이버시(DP)는 저자에 의해 개인 데이터의 프라이버시를 보장하기 위해 사용되며, 이를 위해 발생하는 노이즈를 텍스트 벡터 표현에 추가합니다. 그러나 노이즈 추가는 항상 상당한 유틸리티 손실을 초래하여, 현재의 DP 재작성 메커니즘에서 노이즈의 중요성을 크게 강조합니다. 논문에서는 또한 언어 모델(Language Model, LM)이 프라이버시 보호 없이 텍스트 재작성 작업에 활용될 수 있는지에 대한 연구 질문을 제기합니다.

- **Performance Highlights**: 실험 결과에 따르면, 비DP 프라이바타이제이션 기법은 유틸리티 보존에서 우수한 성능을 보였으며, DP 기법에 비해 효과적인 프라이버시-유틸리티 트레이드오프를 찾을 수 있음을 보여주었습니다. 그러나 DP 방식은 여전히 더 강력한 프라이버시 보호 기능을 제공하였고, 두 방법 간의 미세한 균형을 잘 나타내고 있습니다. 연구자들은 이 기법이 더욱 활용될 필요성이 있다고 강조하며, 프라이바타이제이션 메커니즘을 위한 오픈 소스 프로젝트도 함께 발표했습니다.



### Calling a Spade a Heart: Gaslighting Multimodal Large Language Models via Negation (https://arxiv.org/abs/2501.19017)
- **What's New**: 이번 논문은 다중 모달 대형 언어 모델(MLLMs)의 성능 평가와 특히 부정 논거에 대한 취약성을 집중적으로 분석하였습니다. 다중 모달 시스템이 언어와 시각적 입력에서 뛰어난 성과를 보였음에도 불구하고 부정적인 질문에 대한 반응에서 심각한 성능 저하를 나타냄을 발견했습니다. 이를 통해 MLLMs의 논리적 일관성을 유지하는 능력에 대한 주요한 우려가 있음을 강조하고, 이러한 문제 해결을 위한 향후 연구 방향을 제시합니다.

- **Technical Details**: MLLM의 성능 평가는 다양한 최신 모델을 사용하여 수행되었으며, 각각 상용 모델(GPT-4o, Claude-3.5-Sonnet)과 오픈 소스 모델(Qwen2-VL, LLaVA)에 관한 것입니다. 이 논문에서는 부정 논거를 포함한 대화에서 MLLMs가 어떻게 작동하는지를 이해하기 위해 구조화된 평가 파이프라인을 사용하고, 다양한 벤치마크 데이터 세트를 활용하여 그 결과를 분석했습니다. 특히, 부정 논거가 도입되기 전과 후의 모델 성능을 비교하여 성능 저하를 정량화하는 방법론을 제시하고 있습니다.

- **Performance Highlights**: 연구 결과 부정 논거를 포함한 대화에서 모든 MLLM 모델이 상당한 성능 저하를 경험했으며, 그 평균 정확도 하락 폭은 약 9.62%에 달하는 것으로 나타났습니다. 상용 모델들이 일반적으로 오픈 소스 모델보다 더 나은 수행 능력을 보였지만, 모든 모델이 부정 논거에 대한 저항 능력에서 어려움을 겪고 있음을 알 수 있었습니다. 이러한 결과는 MLLMs의 신뢰성을 높이기 위한 보다 강력한 훈련 기술과 정렬 메커니즘의 필요성을 강조하고 있습니다.



### DyPCL: Dynamic Phoneme-level Contrastive Learning for Dysarthric Speech Recognition (https://arxiv.org/abs/2501.19010)
Comments:
          NAACL 2025, 9pages, 1 page appendix

- **What's New**: 이번 연구는 동적 음소 수준 대조 학습(DyPCL) 방법을 제안하여 비정상적인 발화를 인식하는 데 있어 성능 저하 문제를 해결합니다. DyPCL은 연설 선언문을 음소 세그먼트로 분리해 음소 레벨 대조 학습을 가능하게 하며, 동적인 connectionist temporal classification(CTC) 정렬을 통해 다양한 화자 간에 불변한 표현을 얻는 것을 목표로 합니다. 이 방식은 발화 전체의 표현에 집중하기보다는 미세한 발음 차이를 구분할 수 있는 기회를 제공합니다.

- **Technical Details**: DyPCL은 두 가지 동적 접근 방식을 통합하며, 첫째로 동적 CTC 정렬 방법이 음소 수준의 대조 학습을 위한 음성 임베딩과 음소 라벨을 정확히 정렬합니다. 이전 연구와 달리, 동적 CTC 정렬은 외부 정렬 모듈 없이 강력한 특성 표현을 동시에 학습합니다. 두 번째로 동적 커리큘럼 학습이 도입되어, 음소의 음성적 유사성을 기반으로 난이도에 따라 부정적인 샘플을 조직하는 방식으로 DSR 성능을 향상시킵니다.

- **Performance Highlights**: UASpeech 데이터셋에서 DyPCL은 기준 모델보다 평균 22.10%의 단어 오류율(WER) 감소를 달성하였습니다. 가장 이해도가 낮은 그룹에서는 WER이 58.49%에서 49.45%로 감소했으며, 전체적으로 모든 비정상 발화 그룹에서 WER이 25.97%에서 20.23%로 감소했습니다. 이는 제안된 전략이 강력하다는 것을 입증하며, 심층 분석과 ablation 연구를 통해 그 효과성이 강조되었습니다.



### Adversarial Attacks on AI-Generated Text Detection Models: A Token Probability-Based Approach Using Embeddings (https://arxiv.org/abs/2501.18998)
- **What's New**: 이번 연구에서는 AI로 생성된 텍스트를 탐지하는 모델들, 특히 Fast-DetectGPT에 대한 혁신적인 적대적 공격 기법을 제안합니다. 이 방법은 데이터 변형을 위해 임베딩 모델을 활용하여 AI-generated 텍스트의 원본 탐지 확률을 줄입니다. 특히, Tsetlin Machine (TM)을 기반으로 한 해석 가능성 높은 접근 방식을 사용합니다.

- **Technical Details**: 제안한 방식은 임베딩 기법을 활용하여 유사한 단어 확률 벡터를 결합하는 것으로, 탐지 정확도를 XSum 데이터셋에서 0.4431에서 0.2744로, SQuAD 데이터셋에서는 0.5068에서 0.3532로 감소시킵니다. TM-AE 구조를 통해 생성된 텍스트에 대한 저확률을 할당하여 탐지 시스템의 전반적인 텍스트 스코어를 낮추는 방식입니다.

- **Performance Highlights**: AI-origin 탐지 시스템에 대한 적대적 공격을 통해, Fast-DetectGPT의 탐지 정확성을 눈에 띄게 낮추는데 성공했습니다. 이 방법은 기존 탐지 시스템의 취약성을 겨냥하며, TM 모델의 해석 가능성을 통해 공격 메커니즘에 대한 통찰력을 제공합니다.



### Intrinsic Tensor Field Propagation in Large Language Models: A Novel Approach to Contextual Information Flow (https://arxiv.org/abs/2501.18957)
- **What's New**: 이 논문에서는 언어 모델 아키텍처에서의 맥락 전파(context propagation)의 문제점을 해결하기 위한 새로운 접근 방식인 Intrinsic Tensor Field Propagation (ITFP)를 소개합니다. ITFP는 토큰 임베딩(token embeddings) 위에 분포된 연속 텐서 필드(continuous tensor fields)로 맥락 관계를 모델링합니다. 이를 통해 기존의 주의(attention) 메커니즘을 보완하고, 긴 시퀀스의 일관성과 회상(coherence and recall) 능력을 향상시킵니다.

- **Technical Details**: ITFP는 다변량 미분방정식(partial differential equations)을 통해 맥락 정보의 흐름을 조절하며, 다층 구조 내에서 일관성을 유지합니다. 이 메커니즘은 표준 주의 메커니즘에 추가적인 고차 맥락 보정을 통합한 것으로, 이를 통해 비국소적 맥락 상호작용을 강화합니다. 이론적으로, ITFP는 tensor field를 이용해 모델의 맥락 종속성을 지속적 필드로 재구성합니다.

- **Performance Highlights**: 다양한 언어 구조에서 ITFP가 맥락 보존(contextual retention), 종속성 해결(dependency resolution), 추론 안정성(inference stability) 등에 있어 뚜렷한 성과를 나타내는 실험 결과를 보여줍니다. 실험은 ITFP가 기본 모델에 비해 구문적 불일치(syntactic inconsistencies) 및 사실 오류(factual errors)를 감소시킨다는 것을 입증했습니다. 이 방법은 다양한 텍스트 장르에서 효과적으로 적용 가능한 것으로 보이며, 정확도와 일관성을 향상시키는 이점이 기존의 계산 비용을 초과한다고 주장합니다.



### KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search (https://arxiv.org/abs/2501.18922)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 KBQA(Knowledge Base Question Answering)의 새로운 접근법인 KBQA-o1을 제안합니다. 이 방법은 Monte Carlo Tree Search (MCTS)를 활용하여 KB 환경을 탐색하고, 단계적인 논리 형태 생성을 위한 ReAct 기반 에이전트를 도입합니다. KBQA-o1은 기존 방법들의 치명적인 단점을 보완하며, 더욱 효율적인 답변 생성을 지원합니다. 이전의 저자원 KBQA 방법에 비해 현저한 성능 향상을 보여주는 결과를 도출했습니다.

- **Technical Details**: KBQA-o1은 KB 환경에서의 탐색을 증진시키기 위해 ReAct 기반의 에이전트 프로세스를 설계했습니다. MCTS는 정책 및 보상 모델을 기반으로 한 휴리스틱 검색 방법으로, 탐색 공간과 성능의 균형을 맞추는 데 중점을 두었습니다. 이것은 낮은 샘플 데이터로도 훈련할 수 있도록 하여 대량의 자동 주석 데이터를 생성하는 데 도움을 줍니다. 이러한 프로세스를 통해 KBQA-o1은 기존의 저자원 KBQA 방법들에서 성능을 능가하게 됩니다.

- **Performance Highlights**: 실험 결과 KBQA-o1은 GrailQA, WebQSP, GraphQ 데이터셋에서 기존의 저자원 KBQA 방법과 비교하여 현저히 뛰어난 성능을 기록했습니다. Llama-3.1-8B 모델의 GrailQA F1 성능은 78.5%에 달하며, 이는 이전의 최고의 방법인 GPT-3.5-turbo의 48.5%를 훨씬 초월합니다. 또한, KBQA-o1은 다양한 오픈소스 LLM과의 호환성을 지원하여, 다양한 KBQA 응용 프로그램에 적용 가능하다는 추가적인 장점을 가지고 있습니다.



### Efficient Supernet Training with Orthogonal Softmax for Scalable ASR Model Compression (https://arxiv.org/abs/2501.18895)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이번 연구에서는 다양한 하드웨어 제약을 만족시키기 위해 Supernet 훈련 방식을 사용하여 여러 크기의 인코더를 동시에 학습합니다. 또한, OrthoSoftmax라는 새로운 방법을 도입하여 효율적으로 최적 서브넷을 식별할 수 있도록 기여합니다. 이 방법은 다양한 기준과 세분화 수준에 따른 유연하고 정밀한 서브넷 선택을 가능하게 합니다.

- **Technical Details**: 연구에서는 Acoustic Encoder의 매개변수를 Learnable Parameter로 설정하고, 이를 N개의 파라미터 그룹으로 분해하였습니다. 각 서브넷은 서로 다른 총 비용 제약 조건을 가지며, 이를 위해 이진 마스크를 학습하여 특정 매개변수 집합을 선택합니다. 이러한 방식을 통해 Supernet의 다양한 서브넷을 조정할 수 있습니다.

- **Performance Highlights**: Librispeech 및 TED-LIUM-v2 데이터셋을 통해 수행된 실험 결과, FLOPs-aware 컴포넌트 선택 방법이 최상의 성능을 나타냈습니다. 동일한 교육 업데이트 수로 훈련된 모델들은 개별적으로 훈련된 모델들과 비교할 때 비슷한 WERs를 달성하며, 훈련 시간을 획기적으로 단축시켰습니다. 각 서브넷의 구성 요소 선택 패턴을 분석한 결과, 흥미로운 인사이트를 발견하였습니다.



### Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities (https://arxiv.org/abs/2501.18845)
Comments:
          20 pages, 4 figures, 4 tables

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 데이터 증강(data augmentation) 이론과 방법에 대한 포괄적 분석을 제공합니다. 다양한 증강 기법을 탐색하며, 심플(Simpl)한 증강, 프롬프트 기반(Prompt-based) 증강, 검색 기반(Retrieval-based) 증강, 하이브리드(Hybrid) 증강으로 분류합니다. 최근 연구 추세를 검토하며, 각 기법의 특징과 효과성을 강조합니다.

- **Technical Details**: LLM은 Transformer 아키텍처를 기반으로 하여, 인코더 전용 모델, 디코더 전용 모델, 인코더-디코더 모델로 구분됩니다. 데이터 증강 기법은 기존 데이터를 변형하고 확장하여 다양한 훈련 데이터를 생성하는 방법론으로, 이는 모델 성능 향상에 기여합니다. 최근 연구에서는 LLM의 프롬프트 공학을 활용하여 훈련 데이터를 증대시키는 다양한 접근법이 제안되고 있습니다.

- **Performance Highlights**: 이 논문은 LLM을 활용한 데이터 증강의 유망한 결과들을 정리합니다. 예를 들어, 데이터 편집(data editing) 및 데이터 패러프레이징(data paraphrasing) 기법이 효과적으로 LLM에 의해 구현되었음을 보여줍니다. 최종적으로, 이 연구는 다수의 실제 적용 사례와 평가 지표를 통해 데이터 증강의 현재 한계와 향후 연구 방향에 대한 통찰을 제공합니다.



### Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming (https://arxiv.org/abs/2501.18837)
- **What's New**: 이 논문에서는 전통적인 방법으로 방어하기 어려운 범용 jailbreaks에 대한 대응으로 Constitutional Classifiers를 도입하고 있습니다. 이들은 자연어 규칙을 사용하여 생성된 합성 데이터로 훈련되어 특정 내용의 허가 및 제한을 명시하는 접근법을 사용합니다. 연구 결과, 이러한 새로운 분류기가 대규모 모델 방어에 효과적임을 입증하였으며, 생산 과정의 거부율은 오직 0.38% 증가에 그쳤습니다.

- **Technical Details**: Constitutional Classifiers는 화학무기와 같은 유해한 내용을 식별하기 위한 헌법 기초로 훈련됩니다. 이 과정에서 ‘무해한’ 헌법을 추가하여 분류기의 성능을 향상시킬 수 있다는 점도 발견하였습니다. 연구에서 제시된 간단한 모델에서는 성공적인 단계의 성공 확률을 독립적으로 가정하고, 방어의 효과가 프로세스의 복잡성에 따라 기하급수적으로 감소하는 경향을 보였습니다.

- **Performance Highlights**: 실험을 통해, 제안된 시스템이 잘 작동하여 95% 성공률을 보이는 전통적인 모델과 비교하여, Constitutional Classifiers 도입으로 인해 평균 10의 5 제곱배의 성능 저하를 약속합니다. 다양한 훈련 방식과 하이퍼파라미터 조정이 성능에 긍정적인 영향을 미쳤으며, 고급스러운 화학무기 이론에 대한 요청을 식별하는 정확도가 높아졌습니다.



### Structural Embedding Projection for Contextual Large Language Model Inferenc (https://arxiv.org/abs/2501.18826)
- **What's New**: 본 연구에서는 Structural Embedding Projection (SEP)이라는 새로운 방법론을 소개합니다. SEP는 임베딩 공간을 구조적 관계를 더 잘 포착하도록 수정하여 LLM의 맥락 인식을 향상시키는 것을 목표로 합니다. 이 방법을 통해 일관성 있는 출력과 더불어 추론 효율성이 향상될 것으로 기대하고 있습니다.

- **Technical Details**: SEP 방법론은 토큰 임베딩을 재정의하는 구조적 변환 과정을 포함합니다. 주어진 입력 시퀀스에 대해, 각 임베딩은 학습된 변환 행렬 P를 이용하여 구조적 표현으로 매핑됩니다. 이 과정에서 최적화 기능을 통해 구조적 왜곡을 최소화하고 맥락 무결성을 유지하는 것이 중요합니다.

- **Performance Highlights**: 실험 결과, SEP는 언어 모델의 출력에서 의미적 충실도를 개선하였으며, 혼란도(perplexity) 감소와 맥락 일관성 증가를 확인할 수 있었습니다. 또한, 다양한 데이터셋에서 구조적 임베딩의 통합이 추론 속도와 표현의 풍부함 간의 데이터셋 의존적 균형을 필요로 했다는 점도 강조됩니다.



### Memory-Efficient Fine-Tuning of Transformers via Token Selection (https://arxiv.org/abs/2501.18824)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 TokenTune이라는 새로운 방법론을 제안합니다. 이는 transformer 기반 모델의 fine-tuning 과정에서 메모리 사용을 줄이는 방법으로, 특히 중간 활성화(intermediate activations)를 저장하는 메모리 소비를 최소화합니다. TokenTune은 역전파(backpropagation) 과정에서 전체 입력 토큰이 아닌 일부 선택된 토큰을 통해 기울기(gradient) 계산을 근사하여 메모리 사용을 효과적으로 감소시킵니다.

- **Technical Details**: TokenTune은 forward 통과(during the forward pass) 동안 중간 활성화를 부분적으로만 캐시(caching) 하므로, 전체적인 GPU 메모리 사용량을 줄입니다. 이 방법은 기존의 memory-efficient fine-tuning 기법들과 비교하여 손실 없이 유사한 성능을 낼 수 있습니다. 예를 들어, Llama2-7B 모델을 fine-tune할 때 TokenTune은 QLoRA와 결합하여 기존의 3분의 1 메모리로도 충분히 동작할 수 있습니다.

- **Performance Highlights**: TokenTune의 성과는 다양한 다운스트림(task) 평가에서 확인되었습니다. 본 연구에서 사용된 모델들은 수억에서 수십억 개의 파라미터를 가진 BERT와 Llama로, TokenTune을 사용하여 실험한 결과 가득 채우는 체크가 아닌 총 fine-tuning 양과 동등한 성능 달성이 가능했습니다. 전체 fine-tuning과 비교하였을 때, TokenTune은 메모리 사용량을 21%로 줄이며, 기존 방법들과의 결합으로 효율성을 더욱 높일 수 있습니다.



### Large Language Models as Common-Sense Heuristics (https://arxiv.org/abs/2501.18816)
Comments:
          7 page body, 2 page references, 5 page appendix (14 page total); 1 figure; Submitted to IJCAI2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 계획 작업에서 성공적인 솔루션을 생성하는 데 어려움을 겪고 있는 현실을 지적합니다. 기존의 시스템은 LLM의 파라미터화된 지식을 활용하여 로컬 서치 알고리즘을 위한 행동 선택을 담당하도록 하고, 이를 통해 중간 언어를 필요로 하지 않는 새로운 계획 방법을 제안합니다. 이는 기존의 접근 방법보다 22% 높은 성공률을 기록하며 일관되게 실행 가능한 계획을 생성하는데 주목받고 있습니다.

- **Technical Details**: 자동 계획(AI Planning)은 특정 환경 내에서 실행 가능한 행동의 시퀀스를 생성하는 인공지능의 한 분야입니다. 계획 작업은 상태 공간, 초기 상태, 목표 상태 집합, 행동 집합 및 적합성 함수와 전이 함수를 포함하는 6-튜플로 정의됩니다. 본 연구에서는 LLM의 세계 지식을 활용하여 행위 선택을 위한 휴리스틱(huristic)으로 사용되며, 이를 통해 계획 생성의 효율성을 높이고 있습니다.

- **Performance Highlights**: 제안된 방법은 가정 환경에서의 단순 작업 수행에 있어 기존의 ProgPrompt 시스템보다 22% 높은 성공률을 보이고 있습니다. 이러한 결과는 중간 언어의 필요성을 제거하고, LLM이 제어하는 대리인이 사용하는 동일한 표현 언어 내에서 직접 작동할 수 있음을 보여줍니다. 또한, 이 방법은 잘못된 방식으로 해결된 작업에서도 실행 가능한 계획을 일관되게 생성할 수 있습니다.



### Rope to Nope and Back Again: A New Hybrid Attention Strategy (https://arxiv.org/abs/2501.18795)
- **What's New**: 이번 논문은 Rotary Position Embedding (RoPE)와 같은 최신 기법을 활용하여 긴 컨텍스트를 처리하는 대형 언어 모델의 성능을 분석하고 있습니다. 다양한 주의 메커니즘인 RoPE, No Positional Embedding (NoPE), Query-Key Normalization (QK-Norm)의 장단점을 조명하고, 이를 기반으로 하이브리드 주의 메커니즘을 제안하여 기존 RoPE 모델을 초월하는 성능을 달성하고자 합니다.

- **Technical Details**: 연구에서는 750억 개의 토큰으로 학습된 여러 모델 변형의 주의 패턴을 평가하고, 이를 통해 건축 설계 선택에 대한 통찰을 제공합니다. RoPE 모델은 Rotary Position Embedding을 사용하고, QK-Norm 모델은 각 레이어의 정규화를 통해 훈련의 안정성을 향상합니다. 두 모델 모두 긴 컨텍스트를 처리하는 데 중요한 두 가지 구성 요소인 주의 메커니즘과 위치 인코딩을 중심으로 설계되었습니다.

- **Performance Highlights**: 제안된 하이브리드 메커니즘은 기존의 RoPE 모델보다 월등한 성능을 보이며, 긴 컨텍스트 작업에서 효율성과 성능 간의 균형을 성취합니다. 훈련 프로세스는 두 단계로 나뉘어 있고, supervised fine-tuning을 통해 긴 컨텍스트 작업의 변동성을 줄이는 데 중점을 두었습니다. 최종적으로, 5조 개의 토큰을 통한 광범위한 사전 훈련 결과 기존 모델보다 더 높은 성능을 기록했습니다.



### Overestimation in LLM Evaluation: A Controlled Large-Scale Study on Data Contamination's Impact on Machine Translation (https://arxiv.org/abs/2501.18771)
- **What's New**: 본 논문에서는 데이터 오염(data contamination)이 언어 모델의 평가 기준에 미치는 영향을 체계적으로 분석합니다. 1B 및 8B 규모의 모델을 대상으로 하여, 기계 번역 작업을 통해 오염의 다양한 상태와 데이터 형식에 따른 성능 메트릭에 미치는 영향을 측정합니다. 이를 통해 소스와 타겟 모두가 포함된 오염이 성능을 상당히 부풀려 비효율성을 초래함을 발견하였습니다.

- **Technical Details**: 연구 방법론은 먼저 깨끗한 훈련 및 테스트 데이터 세트를 구성한 후, 오염을 다양한 단계에서 체계적으로 주입하는 구조로 진행되었습니다. 연구진은 n-gram 검색 알고리즘을 통해 미리 평가 데이터와 겹치는 부분을 탐색하고, 이를 통해 불필요한 오염 데이터를 제거하고 훈련을 진행합니다. 이를 바탕으로 42424242개의 오염 조건을 조사하여, 오염의 영향을 정량적으로 분석했습니다.

- **Performance Highlights**: 결과적으로, 소스와 타겟 쌍이 포함된 오염은 성능 부풀리기를 초래하며, 8B 모델에서는 최대 30 BLEU 포인트의 성능 증가가 관찰되었습니다. 시간적 분포에 따라 집중된 지점에서 오염이 발생했을 경우 성능 부풀리기는 커지지만, 지속적으로 오염이 이루어질 경우 가장 영향을 미친다고 합니다. 모델 규모가 커질수록 오염의 영향을 더 민감하게 받아들이는 경향을 보였습니다.



### Breaking the Fake News Barrier: Deep Learning Approaches in Bangla Languag (https://arxiv.org/abs/2501.18766)
Comments:
          6 pages, THE 15th INTERNATIONAL IEEE CONFERENCE ON COMPUTING, COMMUNICATION AND NETWORKING TECHNOLOGIES (ICCCNT)

- **What's New**: 디지털 플랫폼의 급속한 발전으로 인해 불확실한 데이터의 확산이 심화되어 있으며, 이는 특히 벵골어를 사용하는 커뮤니티에서 판단력을 저하시켰습니다. 이 연구는 Gated Repetitive Unit (GRU)이라는 심층 학습 기술을 이용하여 벵골어 가짜 뉴스를 인식하는 효과적인 전략을 제시합니다. 새롭게 개발된 데이터셋은 58,478개의 샘플로 구성되어 있으며, 가짜 뉴스 탐지를 위한 강력한 기반을 제공합니다.

- **Technical Details**: 이 연구에서 제안하는 접근 방식은 데이터 전처리 과정에서 어간 추출(lemmatization), 토큰화(tokenization) 및 비대칭 데이터 문제 해결을 위한 오버샘플링(oversampling)을 포함합니다. GRU 모델을 중심으로 한 시스템 설계와 모델 준비 및 평가에 대한 상세한 설명을 제공합니다. 모델의 성능은 정밀도(precision), 재현율(recall), F1 점수(F1 score), 및 정확도(accuracy)와 같은 신뢰할 수 있는 지표를 통해 평가되었습니다.

- **Performance Highlights**: 제안된 GRU 기반의 모델은 94%라는 놀라운 정확도로 가짜 뉴스를 효과적으로 탐지하였습니다. 연구의 기여로는 벵골어 가짜 뉴스 탐지를 위한 대규모 데이터셋이 구축되었으며, 이 모델은 기존의 다른 벵골어 가짜 뉴스 탐지 모델들보다 뛰어난 성능을 보였습니다. 이 모델은 향후 가짜 뉴스 퇴치에 큰 도움이 될 것으로 기대됩니다.



### Revisiting Projection-based Data Transfer for Cross-Lingual Named Entity Recognition in Low-Resource Languages (https://arxiv.org/abs/2501.18750)
Comments:
          Accepted at NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이번 논문에서는 Cross-lingual Named Entity Recognition (NER) 분야에서 데이터 기반의 전이 기법이 저자원 언어에서 효과적임을 입증하고자 하였습니다. 특히, back-translation 기법을 활용한 단어 정렬 개선과 기존에 추출된 후보자와 소스 개체를 매칭하는 새로운 투영 접근법을 제안합니다. 이를 통해 저자원 언어에서 다국어 모델보다 높은 성과를 달성할 수 있는 가능성을 나타냅니다.

- **Technical Details**: 연구에서는 세 가지 단계로 구성된 주석 투영 방법(역번역, NER 모델 적용, 레이블 다시 투영)에 집중하고, 단어 간 정렬 개선을 위한 두 가지 접근 방식을 제안합니다. 첫 번째로, 서로 다른 언어 간의 정렬 문제를 해결하기 위해 단어 간 관계의 역방향 정렬을 탐구하고, 두 번째로는 출처 개체와 추출된 후보 간의 매칭을 양 당사자 매칭 문제로 재정의합니다. 이는 프로젝션 정확도를 높이기 위한 기초 작업입니다.

- **Performance Highlights**: 57개 언어에 대해 실시된 광범위한 실험 결과, 제안된 접근 방식이 기존의 투영 기반 기법보다 뛰어난 성능을 보여주었습니다. 특히, 저자원 언어의 경우에도 모델 기반 방법을 초월하는 성과를 기록하며, 데이터 전이의 내구성을 강조합니다. 이로 인해 저자원 언어에서의 NER 성능 향상에 기여할 수 있는 가능성을 시사합니다.



### Examining the Robustness of Large Language Models across Language Complexity (https://arxiv.org/abs/2501.18738)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용한 여러 학생 모델이 학생들이 수학 문제 해결에서 자기 조절 학습(SRL)의 존재를 탐지하는 모델의 강건성을 평가합니다. 특히, 다양한 언어 복잡도를 지닌 텍스트를 사용하여 모델의 성능이 어떻게 달라지는지를 비교합니다. 이는 학생들의 언어 배경이 다양할 수 있는 교육 환경에서 공정한 평가를 보장하기 위해 중요합니다.

- **Technical Details**: 논문은 CueThink라는 디지털 학습 플랫폼에서 수집된 학생들의 텍스트 반응을 기반으로 SRL 개념을 연구합니다. 연구진은 데이터 세트에서 다섯 가지 SRL 구성 요소(숫자 표현, 맥락 표현, 전략 지향, 결과 지향, 데이터 변환)를 식별하고, 이를 OpenAI의 문장 임베딩 모델을 사용하여 벡터화하여 기계 학습 모델을 훈련했습니다. 그런 다음, 텍스트의 언어 복잡성을 평가하기 위해 세 가지 언어적 측정 방법을 사용했습니다.

- **Performance Highlights**: 모델의 성능 평가는 10배 학생 수준 교차 검증을 통해 진행되었으며, SRL 구성 요소 탐지를 위한 모델의 일반적인 성공을 검증합니다. 본 연구에서 사용한 여러 성능 지표는 자동화된 SRL 탐지 모델의 유용성을 강조합니다. 연구 결과는 언어 복잡도가 모델 성능에 미치는 영향을 보여주며, 이는 LLM들이 다양한 학생 집단에 대해 공정하게 작동할 수 있도록 하기 위한 중요한 통찰을 제공합니다.



### Zero-shot Large Language Models for Long Clinical Text Summarization with Temporal Reasoning (https://arxiv.org/abs/2501.18724)
- **What's New**: 이번 연구는 의료 분야에서 데이터 처리를 변화시킬 수 있는 큰 언어 모델(LLMs)의 최근 발전을 평가합니다. 특히, 시간적 추론(temporal reasoning)이 필요한 긴 임상 텍스트 요약의 효율성을 검토하여, 환자의 진료 이력과 치료 경과를 포괄적으로 파악할 수 있도록 합니다. 제로샷(zero-shot) LLMs의 강점과 한계를 밝히고, 임상 결정 지원에 효과적으로 기여하기 위해 모델 훈련 방식을 개선해야 할 필요성을 강조합니다.

- **Technical Details**: 이 연구는 MIMIC 데이터셋을 활용하여 기존 LLM들이 긴 임상 문서 요약 및 시간적 정보 관리를 얼마나 잘 수행하는지를 평가합니다. 세 가지 최첨단 LLM인 Qwen2.5-7B, Mistral-7B-Instruct-v0.1 및 Llama3-8B-Instruct를 조사하였으며, 이 모델들이 임상 텍스트 요약의 과제에서 어떤 성능을 보이는지를 분석합니다. 기존의 RAG( Retrieval-Augmented Generation) 기법을 활용한 방법들도 검토되어, 이들이 긴 문서에서의 접근에 있어 여전히 한계가 있음을 나타냅니다.

- **Performance Highlights**: 제공된 결과에 따르면 현재의 LLM들은 긴 문서 내 임상 텍스트 요약 및 환자 경과에 대한 시간적 추론에서 여전히 어려움을 겪고 있습니다. 비록 RAG가 잠재력을 보여주지만, 전반적인 성능은 아직 낮은 수준으로, 개선이 필요함을 강조합니다. 향후 연구는 사건 추출(event extraction) 접근법을 조사하여 해당 한계를 극복하려고 합니다.



### Fake News Detection After LLM Laundering: Measurement and Explanation (https://arxiv.org/abs/2501.18649)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)로 생성된 패러프레이즈 가짜 뉴스를 탐지하는 효율성을 평가하며, 패러프레이즈 과정이 탐지 파이프라인에 도움이 되는지 또는 방해되는지를 분석합니다. 연구 결과, 탐지기가 인간이 작성한 글보다 LLM 패러프레이즈 가짜 뉴스를 탐지하는 데 더 어려움을 겪고 있으며, 감정 변화(sentiment shift)가 탐지 실패에 기여하는 가능성을 발견하였습니다.

- **Technical Details**: 연구에 사용된 기술적 방법론은 LIME 설명을 통해 탐지 실패의 원인을 조사하고, BERTSCORE와 같은 컨텍스트 기반의 임베딩을 활용하여 패러프레이즈의 품질을 평가하는 것입니다. 또한, LLM 모델들이 생성한 다양한 패러프레이즈 텍스트의 검출 용이성을 비교하고, 효과적인 탐지기를 식별하는 데 집중하고 있습니다. 패러프레이즈 생성은 기존의 기술을 넘어 최신의 딥 러닝 모델과 사전 학습된 모델을 사용하여 고급 기술로 발전하고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 패러프레이즈 과정이 탐지 시스템의 성능에 미치는 영향은 상당히 복잡한데, 일부 탐지 모델은 패러프레이즈된 가짜 뉴스를 더 잘 탐지하는 경향이 있습니다. 그러나 대다수의 LLM 모델로 생성된 텍스트는 탐지기를 속이는 경향이 있어, 전반적으로 탐지 성능이 저하되는 문제가 발생하고 있습니다. 마지막으로, GitHub에서 사용할 수 있는 새로운 패러프레이즈 출력과 점수를 포함한 데이터셋이 공개되어 연구자들이 활용할 수 있는 기반이 마련되었습니다.



### Layered Chain-of-Thought Prompting for Multi-Agent LLM Systems: A Comprehensive Approach to Explainable Large Language Models (https://arxiv.org/abs/2501.18645)
- **What's New**: 이 논문에서는 Layered Chain-of-Thought (Layered-CoT) Prompting이라는 새로운 방법론을 제시합니다. 기존의 chain-of-thought (CoT) 방식이 직면했던 한계를 극복하기 위해, 이 방법은 단계적으로 여러 레이어로 성격을 나누어 중간 추론을 검증합니다. Layered-CoT는 의료, 재정 위험 평가 및 민첩한 엔지니어링과 같은 복잡한 시나리오에서 CoT보다 더 높은 투명도, 정확성 및 사용자 참여를 자랑합니다.

- **Technical Details**: Layered-CoT는 각 레이어별로 전문 검증과 피드백을 받을 수 있도록 고안되었습니다. 모델은 문제를 레이어로 나누고, 각 레이어는 제한된 범위의 부분적 체인을 생성하여 외부 자원(예: 도메인 데이터베이스, 지식 그래프)에 의해 검증됩니다. 이 접근법은 오류 전파를 방지하고 사용자에게 즉각적인 피드백을 가능하게 하며, 다중 에이전트 시스템을 통해 각 레이어의 검증 절차가 더욱 강화됩니다.

- **Performance Highlights**: Layered-CoT는 다양한 분야에서 기존 CoT 방법론에 비해 개선된 성능을 시연하고 있습니다. 특히, 각 단계에서의 외부 데이터 확인과 사용자 피드백의 통합은 최종 결론을 더 신뢰할 수 있게 만들어줍니다. 이를 통해 각 레이어가 철저히 검증되어 과거의 오류가 후속 단계로 유입되는 것을 방지하며, 이러한 방식이 고위험 영역에서 신뢰할 수 있는 설명의 제공을 가능하게 합니다.



### Prompt-oriented Output of Culture-Specific Items in Translated African Poetry by Large Language Model: An Initial Multi-layered Tabular Review (https://arxiv.org/abs/2501.18644)
Comments:
          24 pages, 4 tables. arXiv admin note: text overlap with arXiv:2406.03450, arXiv:2312.15304 by other authors

- **What's New**: 이번 논문에서는 Chat Generative PreTrained Transformer Pro가 세 가지 구조화된 프롬프트에 응답하여 생성한 문화 아이템(cultural items)의 출력을 벤치마킹했습니다. 첫 번째 프롬프트는 일반적인 질문이었고, 두 번째는 시적 구조(poetic structure)에, 세 번째는 문화 특수성(cultural specificity)에 중점을 두었습니다. 이 분석을 지원하기 위해 네 개의 비교 테이블이 작성되었습니다.

- **Technical Details**: 첫 번째 테이블은 세 가지 프롬프트에 따라 생성된 문화 아이템의 결과를 제시하고, 두 번째는 Aixela의 Proper nouns와 Common expressions 프레임워크를 기반으로 이러한 출력을 분류하였습니다. 세 번째 테이블은 인간 번역가, 맞춤형 번역 엔진(custom translation engine), 대형 언어 모델(Large Language Model)에서 생성된 문화 아이템을 요약하고 있으며, 마지막 테이블은 문화 특화 프롬프트에 따라 Chat Generative PreTrained Transformer Pro가 사용한 전략을 설명하고 있습니다.

- **Performance Highlights**: 분석 결과, Chat Generative PreTrained Transformer Pro에서 사용된 문화 지향 프롬프트가 아프리카 시의 번역에서 문화 아이템의 향상을 크게 이끌어내지 못했다고 판단되었습니다. 인간 번역은 33개의 문화 아이템, 맞춤형 번역 엔진은 38개, 반면 Chat Generative PreTrained Transformer Pro는 41개의 반복된 문화 아이템을 생성했습니다. 문화 아이템이 번역되지 않은 경우, 대형 언어 모델이 아프리카 시의 문화를 번역하는 접근 방식에서 일관성이 부족함을 드러냈습니다.



### Divergent Emotional Patterns in Disinformation on Social Media? An Analysis of Tweets and TikToks about the DANA in Valencia (https://arxiv.org/abs/2501.18640)
- **What's New**: 이번 연구는 스페인 발렌시아의 DANA 이벤트 동안 소셜 미디어 플랫폼에서의 잘못된 정보(disinformation) 확산을 조사했습니다. 연구팀은 650개의 TikTok 및 X 게시물을 수집하여, 이를 분석하여 신뢰할 수 있는 콘텐츠와 잘못된 정보를 구분하는 새로운 데이터셋을 생성하였습니다. 또한, 고급 언어 모델인 GPT-4o를 활용한 Few-Shot 주석 접근 방식이 수동 레이블과의 높은 일치를 보였다는 점에서 주목할 만합니다.

- **Technical Details**: 연구는 TikTok과 X에서 게시물의 정서(emotion) 및 언어 패턴을 분석하여 잘못된 정보의 확산 메커니즘을 규명하였습니다. LIWC 사전을 이용한 언어 분석 결과, 신뢰할 수 있는 콘텐츠는 더 명확하고 사실적인 언어를 사용하는 반면, 잘못된 정보는 부정어와 개인적인 일화를 통해 신뢰성을 부여하려는 경향이 있음을 보여주었습니다. 또한, 소리 분석 결과, TikTok 게시물의 신뢰할 수 있는 오디오는 더 밝은 톤과 단조로운 서술을 특징으로 하며, 잘못된 정보는 감정적 깊이와 조작적인 음악 요소를 활용하여 참여도를 높였습니다.

- **Performance Highlights**: SVM+TF-IDF 모델이 제한된 데이터에서도 최고 F1-Score를 기록하며 우수한 성능을 보였습니다. 텍스트 전용 모델을 초월하며, 오디오 기능을 포함한 roberta-large-bne 모델이 더 높은 정확도와 F1-Score를 달성했습니다. GPT-4o Few-Shot 모델 역시 자동화된 잘못된 정보 탐지 가능성을 보였으며, TikTok과 같은 다중 모드 플랫폼에서 잘못된 정보 탐지 개선에 있어 텍스트와 오디오 기능 활용의 중요성을 보여줍니다.



### Linguistic Analysis of Sinhala YouTube Comments on Sinhala Music Videos: A Dataset Study (https://arxiv.org/abs/2501.18633)
- **What's New**: 이 연구는 Music Information Retrieval (MIR)와 Music Emotion Recognition (MER) 분야에서 신할라(Sinhala) 노래에 초점을 맞춘 새로운 접근 방식을 제시합니다. 신할라 노래 비디오에 대한 YouTube 댓글 분석을 통해 이 잘 알려지지 않은 분야에 대한 연구를 발전시킵니다. 이 연구는 27개의 YouTube 비디오에서 20곡의 신할라 노래에 대한 댓글을 수집하여 엄격한 언어적 신뢰성을 유지하며 관련성을 확보했습니다.

- **Technical Details**: 총 93,116개의 댓글을 수집한 후, 이 데이터를 고급 필터링 방법과 음역화(transliteration) 메커니즘을 통해 63,471개의 신할라 댓글로 정제했습니다. 또한, 연구자는 신할라 언어에 특정한 964개의 자동 생성된 불용어(stop-words)를 도출했으며, 이 중 182개는 NLTK 코퍼스의 영어 불용어와 정확히 일치했습니다. 신할라 일반 도메인 코퍼스와 YouTube 댓글 코퍼스에 대한 비교를 통해 후자가 일반 도메인을 잘 대표함을 확인했습니다.

- **Performance Highlights**: 이번 연구에서 생성된 철저하게 구성된 데이터 세트와 도출된 불용어는 MIR 및 MER 분야의 향후 연구에 중요한 자원으로 기능할 수 있습니다. 이 데이터는 다양한 문화적 전통 간의 복잡한 음악적 경험을 해결하기 위한 계산적 기술의 가능성을 보여줄 수 있습니다. 따라서 이 연구는 신할라 음악에 대한 감정 인식을 위한 새로운 가능성을 탐색하는 중요한 발판을 제공합니다.



### Decoding-based Regression (https://arxiv.org/abs/2501.19383)
Comments:
          Google DeepMind Technical Report, 25 pages. Code can be found at this https URL

- **What's New**: 이번 연구에서는 언어 모델이 숫자 예측을 문자열로 디코딩하는 회귀 작업을 수행할 수 있는 이론적 기반을 제공하고 있습니다. 기존의 회귀 방법과 유사한 성능을 발휘하면서도, 다양한 분포를 캡처할 수 있는 유연성을 보여줍니다. 특히, 디코딩 기반 회귀 방법의 유용성을 강조하고, 이를 통해 효율적인 데이터 활용과 성능 개선을 이룰 수 있음을 발견했습니다.

- **Technical Details**: 회귀 모델의 성능은 입력(feature) x와 출력(output) y를 어떻게 처리하는가에 따라 달라지며, 연구에서는 LLM을 이용한 입력 벡터의 임베딩을 사용하고 있습니다. 디코딩 기반 회귀 방식은 전통적인 수치 모델링 방법과 달리, 정규화 없이도 ℝ 상의 임의의 숫자 분포를 근사할 수 있는 유연성을 제공합니다. 이러한 방식은 높은 훈련 데이터 필요성을 동반하지만, 적절한 설정에서 데이터 효율성을 인정받고 있습니다.

- **Performance Highlights**: 디코딩 기반 회귀 헤드는 다양한 표 형식의 회귀 작업에서 기존 포인트 헤드와 경쟁할 만한 성능을 보여줍니다. 특히, Gaussian 혼합 모델과 같은 밀도 추정 작업에서도 충분히 표현력 있는 성능을 발휘합니다. 이는 기존의 회귀 기법에 비해 데이터 효율성과 성능 면에서 유리함을 의미합니다.



### SELMA: A Speech-Enabled Language Model for Virtual Assistant Interactions (https://arxiv.org/abs/2501.19377)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 연구에서는 SELMA, 즉 음성을 활용한 가상 비서 상호작용을 위한 언어 모델을 제안하고 평가합니다. 이 모델은 오디오(audio)와 텍스트(text)를 입력으로 통합하여 단일의 End-to-End 모델 내에서 세 가지 주요 작업과 두 가지 보조 작업을 동시에 처리합니다. SELMA는 파라미터 효율적인 훈련을 위해 저랭크 적응 모듈을 사용하고, 기능 풀링(feature pooling) 전략을 구현하여 시스템이 전반적인 패턴을 인식하고 정확성을 향상시킬 수 있도록 설계되었습니다.

- **Technical Details**: SELMA는 Qwen-Audio의 지침 파인튜닝(fine-tuning) 버전을 기반으로 하며, 음성 인식을 위한 Transform 기반 오디오 인코더를 갖추고 있습니다. 오디오 인코더는 오디오 시퀀스에서 잠재 표현을 생성하며, 이어지는 언어 모델은 이러한 표현을 처리합니다. 이 시스템은 VT 감지, DDSD, ASR 및 텍스트 기반 DDSD와 대화 행위(Dialog Act) 분류의 다섯 가지 작업을 통합하여 사용자의 입력을 처리합니다.

- **Performance Highlights**: SELMA는 VT 감지 작업에서 64%의 상대적인 동등 오류율(EER) 개선을 달성하였고, DDSD에서는 22% 개선을 기록했습니다. 또한, ASR 작업에서도 기존의 모델들과 비교해 우수한 성능을 자랑합니다. 이러한 결과는 SELMA가 특정 작업에 특화된 모델보다 전체적인 성능을 유지하거나 향상시키면서도 처리 파이프라인을 간소화할 수 있음을 보여줍니다.



### We're Different, We're the Same: Creative Homogeneity Across LLMs (https://arxiv.org/abs/2501.19361)
- **What's New**: 이 연구는 다양한 대형 언어 모델(LLMs)과 인간의 창의적인 응답을 비교하여 LLM을 창의적 도구로 사용할 때 나타나는 출시에 대한 동질성을 분석합니다. 대부분의 기존 연구는 특정 LLM만을 고려해왔으나, 이 연구는 여러 LLM이 얼마나 유사한 응답을 생성하는지 집중적으로 조사합니다. 연구 결과, LLM의 응답은 서로 유사한 경향이 강하며, 이는 인간의 창의성을 제한할 수 있는 새로운 우려를 제기합니다.

- **Technical Details**: 연구에서는 표준화된 창의성 테스트인 Alternative Uses Task, Forward Flow, Divergent Association Task를 사용하여 응답의 다양성을 측정했습니다. LLMs는 인체 중심의 심리학적 테스트에서 더 높은 점수를 기록하지만, LLM 응답끼리의 유사성은 인간 응답의 유사성보다 현저히 높으며, 이는 응답 구조나 기타 변수를 통제한 후에도 유지됩니다. 또한, LLM 시스템 프롬프트를 수정하여 높은 창의성을 유도하면 전반적인 LLM의 창의성과 LLM 간 응답 변동성이 약간 증가하지만, 인간 응답은 여전히 더 높습니다.

- **Performance Highlights**: 이 연구는 LLM이 창의성 테스트에서 인간을 초월하는 경향이 있음을 보여주지만, 이는 오히려 LLM 간의 유사성이 높다는 것을 반증하는 요인으로 작용합니다. LLM의 응답들은 고유한 창의적인 결과물로 보이지만 결국에는 상대적으로 동질적인 결과물로 귀결됩니다. 오늘날 가장 인기 있는 모델들이 높은 중복성과 동질성을 보인다면, 사용자는 오히려 다양성을 잃고 평균으로 수렴하는 결과를 초래할 수 있습니다.



### PixelWorld: Towards Perceiving Everything as Pixels (https://arxiv.org/abs/2501.19339)
- **What's New**: 기존의 foundation models(기반 모델)은 시각적 입력을 픽셀 단위로, 텍스트 입력을 토큰 단위로 처리하는데, 이는 인간의 지각 방식과는 다르다. 이 논문에서는 모든 모달리티(모드)를 픽셀 입력으로 통합하는 ‘Perceive Everything as Pixels’ (PEAP) 접근 방식을 제안한다. 또한, PixelWorld라는 새로운 평가 도구를 통해 기존 모델의 성능을 측정한다.

- **Technical Details**: PixelWorld는 텍스트, 테이블, 코드, 다이어그램, 이미지 등 다양한 모달리티를 픽셀 공간으로 통합하여 평가하는 새로운 평가 도구로 개발되었다. 초기 연구 결과, PEAP를 사용한 모델은 멀티모달 데이터셋에서 토큰 기반 입력보다 우수한 성능을 보였다. 그러나 픽셀 기반 입력 처리를 통해 모든 모델의 추론 및 코딩 능력이 크게 감소한 것을 강조하며, 기반 모델의 지각 능력 향상이 필요함을 시사한다.

- **Performance Highlights**: PEAP를 통해 대형 모델이 비추론 작업에서 강력한 성능을 유지하는 반면, 작은 모델인 Phi-3.5-V는 성능 저하가 심각하다. 또한, PEAP는 텍스트 토큰 입력과 응집력이 높은 주의(attention) 패턴을 보이며, 공간 희소성(spatial sparsity)을 활용하여 성능을 크게 가속화할 수 있는 가능성을 보여준다. 이러한 결과들은 기존의 최전선 모델들이 픽셀 지각에 능숙하다는 점을 확인하나, 아직 개선할 여지가 남아있음을 나타낸다.



### Language Bias in Self-Supervised Learning For Automatic Speech Recognition (https://arxiv.org/abs/2501.19321)
Comments:
          Accepted to Speech and Language Technology Workshop (SLT) 2024 accessible on IEEE Xplore

- **What's New**: 이 논문에서는 다국어 음성 인식(ASR) 모델 XLS-R의 자기 지도 학습(self-supervised learning, SSL)의 언어 편향을 분석하고, 언어별 서브네트워크(subnetwork)를 식별하여 성능에 미치는 영향을 평가합니다. 기존 연구에서는 SSL ASR의 언어 편향에 관한 심층적인 검토가 부족했으나, 본 연구는 언어 불균형이 SSL 모델의 성능에 미치는 영향을 규명하는 데 초점을 맞추고 있습니다.

- **Technical Details**: XLS-R 모델은 128개 언어로 훈련된 ASR 모델로, 300M 매개변수(parameter)로 구성됩니다. 이 논문에서 사용된 FLEURS 데이터셋은 101개 언어의 음성 데이터를 포함하고 있으며, 언어별로 약 12시간의 데이터가 제공됩니다. 연구자들은 Lottery Ticket Hypothesis (LTH)를 활용하여 XLS-R의 언어별 서브네트워크를 파악하고, 이는 데이터의 언어 불균형이 성능에 미치는 영향을 명확히 하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, XLS-R 모델은 유사한 데이터 세트를 기반으로 학습한 다국어 음성 인식에서 전통적인 언어 지식에 의존하지 않고, 가장 데이터 기여도가 높은 언어에서 학습된 가중치에만 의존함이 드러났습니다. 이를 통해, 특정 언어로의 전이학습(transfer learning) 시, 서브네트워크의 구조가 특정 언어에 대해 성능이 다르게 나타나는 것을 확인했습니다. 이러한 결과는 SSL ASR 모델의 훈련 및 성능 개선을 위한 새로운 방향을 제시합니다.



### Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignmen (https://arxiv.org/abs/2501.19309)
- **What's New**: 이 논문에서는 Speculative Decoding (SD) 기법을 개선하기 위한 새로운 방식이 제안되었습니다. 특히, LLM-as-a-judge 프레임워크에서 영감을 받아, 모델이 정확하지만 비정렬된 응답을 인식할 수 있도록 검증 과정을 조정하는 방법을 소개합니다. 이를 통해 검증의 효율성을 높이고 여러 높은 품질의 초안 토큰의 채택율을 증가시킬 수 있음을 보여줍니다.

- **Technical Details**: 제안된 방법에서는 Llama 모델을 활용하여, 작은 모듈을 통해 현재 생성된 응답의 품질을 평가하는 'judgements'를 생성합니다. 이렇게 생성된 판단은 LLM-judge의 기능을 모사하며, 훈련 시간은 1.5시간 이내로 설정되어 있습니다. 결과적으로, Llama 8B/70B-Judge는 기존의 규범적 디코딩에 비해 9배의 속도를 실현하였고, 여러 벤치마크에서 Llama-405B 품질을 유지하고 있습니다.

- **Performance Highlights**: Llama-8B/70B-Judge를 사용하면 최대 129 tokens/s의 속도를 달성하며 고품질 결과를 나타냅니다. 이 논문에서 제안된 접근 방식은 인프라가 최적화된 환경에서도 우수한 성능을 유지합니다. Speedup 성능이 기존 방법보다 크게 향상되어, Speculative Decoding의 한계를 뛰어넘는 차세대 솔루션의 가능성을 제시합니다.



### SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling (https://arxiv.org/abs/2501.19306)
- **What's New**: 이 논문에서는 Self-Enhanced Test-Time Scaling (SETS)라는 새로운 방법론을 제안합니다. SETS는 최근의 고급 LLM의 자기 검증(self-verification) 및 자기 수정(self-correction) 기능을 활용하여 복잡한 작업에 대한 테스트 시간 컴퓨테이션(test-time computation)을 효율적으로 수행할 수 있도록 통합된 프레임워크를 제공합니다. 기존의 방법론들이 테스트 시간 컴퓨테이션의 확장에 따라 수익이 줄어드는 문제를 해결하고자 합니다.

- **Technical Details**: SETTS에서는 4개의 데이터 세트인 Trip Planning, Meeting Planning, Calendar Scheduling 및 LiveBench Reasoning을 대상으로 실험을 수행합니다. 각 데이터 세트에서 자기 검증과 자기 수정 작업에 대한 프롬프트(prompt)를 설정하며, 이를 통해 다양한 방법론들의 성능을 비교 분석합니다. NATURAL PLAN 벤치마크는 각 작업에 대해 몇 가지 예시를 제공하고 있으며, 이를 통해 작업의 난이도를 제어할 수 있는 변수를 활용합니다.

- **Performance Highlights**: SETS 방법론은 기존 방법론에 비해 현저하게 성능이 개선되었음을 보여줍니다. 각 데이터 세트에서 자가 수정(self-correction) 성능을 측정한 결과, GEMINI-1.5-Pro-002 모델이 가장 강력한 성능을 보이며, 특히 Trip Planning에서 n=10까지 개선 효과가 지속됩니다. 반면, 일부 모델인 Claude-3.5-Sonnet은 제한된 자기 수정 성능을 보이며, 특정 데이터 세트에 따라 정확도가 감소하는 경향을 보였습니다.



### mFollowIR: a Multilingual Benchmark for Instruction Following in Retrieva (https://arxiv.org/abs/2501.19264)
Comments:
          Accepted to ECIR 2025

- **What's New**: 본 논문에서는 mFollowIR이라는 다국어 벤치마크를 소개하여, 다국어 검색 모델의 명령 따르기(Instruction-following) 능력을 평가하고자 합니다. 이전의 연구에서는 주로 영어에 집중되었던 반면, mFollowIR은 러시아어, 중국어, 페르시아어로 된 쿼리와 지침을 포함한 새로운 데이터셋을 구축하여 전 세계의 다양한 사용자를 위해 다국어 지원의 필요성을 강조합니다. 이를 통해 다국어 모델의 성능을 평가하고, 모델이 복잡한 지침을 얼마나 잘 따르는지를 분석하고자 합니다.

- **Technical Details**: mFollowIR은 TREC NeuCLIR 내러티브를 기반으로 구축되었으며, 각 언어별로 문서의 관련성 평가와 지침을 제공하는 작업으로 구성됩니다. 이를 통해 논문에서는 검색 모델이 지침을 얼마나 효과적으로 따르는지 평가할 수 있는 새로운 평가 절차를 제시합니다. 또한, 영어에서 훈련된 검색기들이 다국어 환경에서도 어느 정도 성능을 발휘함을 보여주며, 추가적인 개선이 필요하다는 점도 지적합니다.

- **Performance Highlights**: 결과적으로, 다국어 및 교차언어 IR 모델은 복잡한 지침을 제공했을 때 적절한 관련성 점수를 수정하는데 어려움을 겪었습니다. 그러나 지침-trained retriever의 경우 언어 기반 훈련이 수행되었을 때 상대적으로 더 나은 결과를 나타내었으며, 이는 저희 연구가 제시한 다국어 지침 따르기 접근 방식의 장점을 시사합니다. 이 결과들은 향후 연구가 더 발전된 다국어 검색 모델을 구축하기 위한 기반이 될 수 있음을 보여줍니다.



### Enabling Autonomic Microservice Management through Self-Learning Agents (https://arxiv.org/abs/2501.19056)
- **What's New**: 이 논문에서는 복잡한 소프트웨어 시스템의 자율적 관리의 필요성을 강조하며, ServiceOdyssey라는 자가 학습 에이전트 시스템을 제안합니다. 기존의 대형 언어 모델(LLM)의 지식을 특정 서비스 환경에 적용하는 데 어려움을 겪는 문제를 해결하고자 하며, ServiceOdyssey는 사전 지식 없이도 마이크로서비스를 자율적으로 관리할 수 있는 시스템입니다. Curriculum learning 원리를 활용하여 환경을 탐험하고, 지속적으로 지식을 축적하는 방법론을 제시합니다.

- **Technical Details**: ServiceOdyssey 시스템은 데이터 레이어와 관리 레이어로 구성된 아키텍처를 가지고 있으며, 효율적인 자가 학습 및 작업 실행을 지원합니다. 관리 레이어는 Curriculum Builder (CB), Execution Planner (EP), Knowledge Curator (KC)라는 세 가지 주요 모듈로 구성되어 있습니다. CB는 새로운 마이크로서비스 시스템을 탐색하고 학습하기 위한 작업을 점진적으로 생성하며, EP는 실행 가능한 계획과 행동을 만들어냅니다. KC는 피드백과 상호작용 기록을 기반으로 포괄적인 역량 라이브러리를 구축합니다.

- **Performance Highlights**: ServiceOdyssey의 프로토타입 구현은 Sock Shop 마이크로서비스를 활용하여 자율 마이크로서비스 관리의 가능성을 시연합니다. 지금까지의 연구에서 보여진 결과는 LLM 기반 에이전트가 자율적으로 환경을 학습하고, 동적으로 지식을 업데이트하며, 최적화된 관리 작업을 수행할 수 있다는 것을 입증합니다. 이 접근법은 노동 집약적인 수동 통합 방식에서 벗어나, 에이전트가 자율적으로 지식을 축적하도록 설계되었다는 점에서 혁신적입니다.



### Scalable Multi-phase Word Embedding Using Conjunctive Propositional Clauses (https://arxiv.org/abs/2501.19018)
- **What's New**: 이번 연구에서는 Tsetlin Machine (TM) 아키텍처를 기반으로 한 새로운 접근 방식을 소개합니다. 두 단계의 훈련을 포함한 이 방법은 입력 시퀀스의 맥락을 포착하여 스케일링 가능한 모델을 설계하면서도 해석 가능성을 유지합니다. 기존의 단어 임베딩 방식과 달리, 이 연구는 데이터셋에 있는 각 단어의 지식을 캡슐화하여 임베딩을 구축하는 방안을 제시합니다.

- **Technical Details**: 이 연구는 두 단계의 훈련 방식을 활용하여 Coalesced Tsetlin Machine (CoTM) 아키텍처를 기반으로 합니다. 입력 데이터는 이진 벡터로 구성되며, 훈련 데이터셋의 어휘 길이의 두 배입니다. 특히, 입력은 타겟 단어가 포함된 문서와 포함되지 않은 문서로부터 특징을 추출하여 생성하고, negated features를 사용하여 표현합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 접근 방식에 비해 경쟁력 있는 성능을 보여주었습니다. IMDB 데이터셋의 감정 분석에 적용했을 때, TM 임베딩과 TM 분류기가 다른 해석 가능한 분류기들과 함께 투명한 end-to-end 솔루션을 제공하며 경쟁력 있는 성능을 나타냈습니다. 이러한 결과는 향후 TM 기반 접근 방식의 가능성을 보여줍니다.



### Importing Phantoms: Measuring LLM Package Hallucination Vulnerabilities (https://arxiv.org/abs/2501.19012)
- **What's New**: 이 논문의 주요 결과는 대형 언어 모델(LLMs)이 생성하는 코드에서 존재하지 않는 패키지, 즉 'hallucinations'의 발생 여부를 분석한 것입니다. 연구자들은 다양한 프로그래밍 언어와 모델의 특성을 고려하여 패키지 환각의 발생 빈도와 그 영향을 규명했습니다. 이는 소프트웨어 개발에 있어 새로운 보안 취약점을 드러내며, 향후 AI 지원 소프트웨어 개발의 안전성을 강화하는 데 기여할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: 연구에서는 Python, JavaScript, Rust와 같은 인기 있는 프로그래밍 언어를 대상으로 LLM의 패키지 환각 행동을 분석합니다. 연구팀은 패키지 환각이 발생할 확률을 측정하고, 프로그래밍 언어, 모델 크기 및 코딩 작업의 특수성 등이 환각률에 미치는 영향을 살펴보았습니다. 또한, LLM의 출력물에서 생성된 패키지 이름이 공공 저장소에 등재되지 않은 경우를 포착하여, 공격 가능성이 있는 코드를 구별하는 방법론을 개발했습니다.

- **Performance Highlights**: 연구 결과, 특정 모델 및 프로그래밍 언어에 따라 패키지 환각 발생 비율이 다양하다는 것을 발견했습니다. 또한, 코딩 성능과 패키지 환각률 간의 상관관계를 조사한 결과, 패키지 환각률이 낮은 모델이 더 높은 HumanEval 코딩 벤치마크 성과를 보인다는 역 상관관계를 발견했습니다. 이러한 분석은 LLM 기반의 안전한 소프트웨어 개발 흐름을 위한 방어 기법들을 제안하며, 소프트웨어 공급망을 위협하는 새로운 공격 벡터에 대응할 수 있는 기반을 제공합니다.



### Language Games as the Pathway to Artificial Superhuman Intelligenc (https://arxiv.org/abs/2501.18924)
Comments:
          This position paper argues that language games provide robust mechanism for achieving superhuman intelligence in large language models

- **What's New**: 이 논문은 인공지능(AI)에서 인간을 초월하는 지능(ASI)으로의 진화의 중요한 요소로 데이터 재생산(data reproduction)을 제시합니다. 전통적인 모델들은 한정된 인간 생성 분포 내에서만 최적화되어 stagnation(침체)에 빠지기 쉬운 문제가 있습니다. 따라서 저자들은 언어 게임(language games)을 통해 이 사이클을 깨고, 확장된 데이터 재생산을 가능하게 하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 언어 게임이 LLM의 데이터 재생산 및 모델 개선을 위한 세 가지 메커니즘인 역할 유동성(role fluidity), 보상 다양성(reward variety), 규칙 유연성(rule plasticity)을 기반으로 작동한다고 설명합니다. 이 메커니즘들은 모델의 학습과 상호작용 과정을 개선하며, 언어적 환경을 재구성하는 것이 가능하게 하여 개념적인 탐구를 촉진합니다. 또한, 강화 학습(reinforcement learning) 기법을 통합하여 모델의 복잡한 피드백 신호를 내부화하고 다양한 도메인에서 전략을 개선할 수 있다고 주장합니다.

- **Performance Highlights**: 저자들은 언어 게임을 통해 모델과 인간의 협력을 통한 데이터 흐름의 생성이 가능하여, 데이터 재생산을 새로운 차원으로 끌어올린다고 강조합니다. 이를 통해 지속적으로 새로운 아이디어와 지능적 행동을 생성할 수 있는 잠재력을 사전 파악할 수 있습니다. 결론적으로, 이 접근 방식은 인간 수준의 지능을 초월하는 AI 개발을 위한 로드맵을 제시하며, 모델의 학습 과정에서 성능 개선을 기대할 수 있습니다.



### BRiTE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning (https://arxiv.org/abs/2501.18858)
- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Models, LLM)이 복잡한 추론 작업에서 신뢰할 수 있는 추론 과정을 생성하는 데 어려움을 겪고 있다는 점을 강조합니다. 이를 해결하기 위해, 우리는 잠재적 사고 과정과 평가 신호를 통합한 새로운 그래픽 모델을 사용하여 LLM의 추론을 형식화하는 확률적 프레임워크를 제안합니다. 특히 Bootstrapping Reinforced Thinking Process (BRiTE) 알고리즘을 도입하여, 두 단계의 과정으로 고품질의 합리적인 이유(rationale)를 생성하고 LLM의 성능을 향상시키는 방법을 설명합니다.

- **Technical Details**: 본 연구에서는 Bootstrapping Reinforced Thinking Process (BRiTE) 알고리즘을 통해 고품질의 합리적인 이유와 답변을 생성하는 확률적 그래픽 모델을 제안합니다. BRiTE 알고리즘은 강화 학습(reinforcement learning)을 사용하여 첫 번째 단계에서 가능한 사고 과정을 근사하여 합리성을 생성하고, 두 번째 단계에서 모델의 매개변수에 대해 합리적인 생성의 결합 확률을 극대화합니다. 이론적으로는 BRiTE가 1/T의 수렴 속도를 가지며, 이는 반복(iterations) 수 T에 해당합니다.

- **Performance Highlights**: BRiTE 알고리즘은 여러 LLM 모델(Gemma, Llama, Mistral)에 대해 수학 및 코드 생성 벤치마크를 포함하여 일관된 성능 향상을 보여줍니다. 특히, GEM8K 벤치마크에서 Gemma-1.1-7B-it 모델을 적용했을 때 10점의 성능 향상을 달성하였으며, 인간이 주석을 단 사고 과정을 사용한 감독 학습(supervised fine-tuning) 방법과 비슷한 성능을 보였습니다. 이로 인해 BRiTE 알고리즘은 수동 프롬프트 설계에 대한 의존도를 줄이고, 자동화된 사고 프로세스 생성의 가능성을 제시합니다.



### Partially Rewriting a Transformer in Natural Languag (https://arxiv.org/abs/2501.18838)
- **What's New**: 이번 논문에서는 Mechanistic Interpretability(기계적 해석 가능성)의 최종 목표인 이해하기 쉬운 형식으로 딥러닝 네트워크를 재구성하고, 그 성능과 동작을 보존하는 방법을 탐구합니다. 구체적으로, 대형 언어 모델(LLM)을 사용하여 자연어 설명을 통해 신경망의 활성화를 예측하는 방법을 제시합니다. 이렇게 함으로써 모델을 보다 해석 가능한 형태로 부분적으로 재작성하여 인공지능 모델에 대한 이해도를 높이고자 합니다.

- **Technical Details**: 논문에서는 먼저 Sparse Transcoder(희소 변환기)를 사용하여 LLM의 피드포워드 네트워크를 넓은 MLP(다층 퍼셉트론)로 근사합니다. 이 모델은 자연어 설명에 따라 각각의 뉴런의 활성화를 예측하는 LLM 기반 시뮬레이터로 구성되며, 활성화 예측에 대한 설명을 생성하는 자동화된 해석 가능성 파이프라인을 사용합니다. 이후 희소 MLP의 첫 번째 레이어를 LLM 기반 시뮬레이터로 교체하고, 모델의 최종 출력을 왜곡시키는 정도를 측정하여 분석합니다.

- **Performance Highlights**: 성능 평가 결과, 희소 MLP의 출력을 제로 벡터로 완전히 대체했을 때와 통계적으로 유사한 손실 증가가 관찰되었습니다. 또한, 희소 오토인코더를 사용하여 동일한 레이어의 잔여 스트림에 대해 유사한 결과를 얻었으며, 이는 더 자세한 설명이 성능 개선에 필수적이라는 것을 시사합니다. 모델의 성능이 점진적으로 변화하며 새로운 해석 가능성을 제공하기 위한 연구 방향을 제시합니다.



### Bridging the Reasoning Gap: Small LLMs Can Plan with Generalised Strategies (https://arxiv.org/abs/2501.18817)
Comments:
          7 page body, 2 page references, 16 page appendix (25 pages total); 2 figures; submitted to IJCAI2025

- **What's New**: 최근 대규모 언어 모델(LLM)의 추론 능력 향상은 간단한 계획 작업 해결 능력의 증가를 보여줍니다. 그러나 LLM의 성능 향상이 모델의 크기와 복잡성에 의존하므로, 운영 비용이 증가하고 있습니다. 이 연구는 자원이 덜 소모되는 LLM의 추론 능력을 강화하기 위한 두 가지 접근 방식을 제안합니다: 일반화된 전략 생성과 오류 교정 프로세스를 통한 개선입니다.

- **Technical Details**: 제안된 방법은 자원 소모가 적은 LLM이 보다 강력한 LLM에서 생성된 일반화된 전략을 제공받아, 해당 도메인 내 작업을 해결하는 것입니다. 또한, 모델의 솔루션에서 오류를 식별하고 이를 교정하는 과정을 통해 성능을 향상시킵니다. 이러한 방법을 통해 향후 불필요한 자원 소비를 줄이고 성능을 극대화할 수 있는 가능성을 확인하였습니다.

- **Performance Highlights**: 실험 결과, 자원이 덜 소모되는 LLM은 제안된 방법을 활용했을 때 강력한 LLM과 유사한 성능을 보였습니다. 특히, 일반화된 전략을 사용 시, 비용이 평균 30% 줄어들었으며, 4회의 오류 수정 후에는 50%까지 비용 절감이 이루어졌습니다. 이로 인해, 자원이 적게 소모되는 모델이 수학적 추론 작업에서 더 강력한 모델보다 20% 높은 성과를 달성했습니다.



### Evaluating Spoken Language as a Biomarker for Automated Screening of Cognitive Impairmen (https://arxiv.org/abs/2501.18731)
- **What's New**: 이번 연구는 Alzheimer’s disease 및 관련 치매(ADRD)를 조기에 평가하기 위한 음성 기반 biomarker의 사용을 탐구합니다. 기존의 진단 방법보다 비침습적이고 스케일 가능한 자동 스크리닝을 가능하게 하는 기계 학습 알고리즘을 적용하였습니다. 연구팀은 대규모 음성 데이터셋(DementiaBank)에서 수집한 말을 바탕으로 ADRD 진단 및 중증도 예측 모델을 평가했습니다. 특히, 리빙 컬렉션한 데이터를 통해 모델의 일반화 가능성을 검증했습니다.

- **Technical Details**: 연구는 기계 학습(ML) 기법을 사용하여 언어의 음성적 특성을 분석하고, Random Forest 알고리즘을 통해 ADRD를 분류하는 모델을 개발했습니다. 이 모델은 69.4%의 평균 민감도와 83.3%의 특이도를 달성했으며, 실제 파일럿 데이터에서는 각각 70.0% 및 52.5%의 결과를 보였습니다. 언어적 특성 분석을 통해 높은 ADRD 위험과 관련된 대명사 및 부사의 증가, 불유창성, 낮은 어휘 다양성 등의 요인을 확인했습니다. 이로써 예측 모델의 해석 가능성과 임상적 유용성을 향상시키는 데 기여했습니다.

- **Performance Highlights**: 모델은 Mini-Mental State Examination (MMSE) 점수를 기반으로 중증도 예측에서 평균 3.7의 오차를 보였습니다. ADRD 위험 분류에서 최적의 언어적 특성을 사용할 수 있게 되며, 이는 인공지능 기반 대화형 기술과의 통합 가능성을 밝혀냅니다. 이 연구는 장기적인 인지 건강 모니터링과 고위험 개인의 조기 발견 및 개입을 위한 새로운 접근법을 제시합니다. 데이터를 통해 AI 기술이 인지 저하 감지에 있어 잠재력을 갖추었음을 보여주는 것이 특징입니다.



### Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation (https://arxiv.org/abs/2501.18638)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 연구는 고급 콘텐츠 정책에서 파생된 은밀한 jailbreak 프롬프트 생성을 자동화하는 모듈형 파이프라인을 제시합니다. 이를 통해 LLM 콘텐츠 조정의 효율성을 개선하고, 기존 알고리즘에 비해 쿼리 수를 54% 감소시키면서도 92%의 공격 성공률을 달성했습니다. 또한, LLM을 사용하여 고급 정책에서 시드 프롬프트를 자동 생성함으로써 차가운 시작 문제를 해결했습니다.

- **Technical Details**: 이 연구에서 도입된 Graph of Attacks with Pruning(GAP) 방법은 공격 효율을 높이고 은닉성을 극대화하기 위해 동적이고 상호 연결된 추론을 활용합니다. GAP는 기존의 선형 구조를 그래프 형태로 전환하여 상대방 LLM에 대한 공격 벡터를 정제하고, 더 포괄적인 탐색을 가능하게 합니다. 파이프라인은 PromptGuard를 미세 조정하기 위해 생성된 프롬프트를 사용하여 다양한 해로운 콘텐츠 유형에 대한 탐지 능력을 더욱 향상시킵니다.

- **Performance Highlights**: GAP는 GPT-3.5를 대상으로 한 실험에서 96%의 공격 성공률을 기록하며 기존 TAP(78%)을 훨씬 초과했습니다. 또한, GAP-M은 Gemma-2-9B와 Qwen2.5-7B 모델에서 각각 100%의 성공률을 기록하며 기존의 낮은 성공률과 높은 쿼리 수를 능가했습니다. 이러한 결과는 GAP의 효과적인 적대적 공격 생성 능력을 입증하며, 콘텐츠 조정 시스템의 지속적인 개선을 위한 중요한 기반을 제공합니다.



### Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcar (https://arxiv.org/abs/2501.18632)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 안전성 문제를 체계적으로 조사하였습니다. 특히, 임상 환경에서 이들 모델이 의학적 'jailbreaking' 공격에 얼마나 취약한지를 평가했습니다. 연구 결과, 상용 및 오픈 소스 LLM들이 이러한 공격에 매우 취약하다는 점을 밝혀냈습니다.

- **Technical Details**: 연구진은 자동화된 도메인 적합 기법을 활용하여 LLM의 취약성을 평가하기 위한 에이전틱 평가 파이프라인(agentic evaluation pipeline)을 제안했습니다. 이를 통해 세 가지 고급 블랙박스 'jailbreaking' 기술을 사용하여 여섯 가지 LLM을 테스트하였습니다. 또한, 지속적 미세 조정(Continual Fine-Tuning, CFT)의 효과를 분석하여 의료적 적대적 공격에 대한 방어력을 강화했습니다.

- **Performance Highlights**: 실험 결과, LLM 모델들이 의료 분야의 공격에 쉽게 노출된다는 사실이 확인되었으며, 이에 따른 안전성과 신뢰성의 증대 필요성이 강조되었습니다. 연구는 AI 의사의 안전성을 높이고 효과적인 배포를 위해 지속적인 공격 방법 평가와 도메인 특화 안전 정렬의 중요성을 부각시킵니다.



### Indiana Jones: There Are Always Some Useful Ancient Relics (https://arxiv.org/abs/2501.18628)
- **What's New**: 이 논문에서는 인디애나 존스(Indiana Jones)라는 혁신적인 접근법을 소개하여 Large Language Models(LLMs)의 jailbreaking을 수행합니다. 이 방법은 다수의 특수화된 LLM 간의 대화와 키워드 중심의 프롬프트를 활용하여 콘텐츠 안전 장치를 우회하는 데 거의 완벽한 성공률을 달성합니다. 연구는 현대 LLM의 체계적 취약성을 드러내며, 해로운 출력을 유도할 수 있는 간접적인 프롬프트의 위험성을 강조합니다.

- **Technical Details**: 이 연구에서는 새로운 jailbreak 방법을 제안하고, 블랙박스(black-box) 및 화이트박스(white-box) LLM에서의 효과와 효율성을 평가합니다. 시험 설계는 세 개의 LLM을 기반으로 한 다중 라운드 상호작용을 포함하며, 특정 키워드를 기반으로 한 jailbreak 목표를 달성하기 위해 Victim, Suspect, Checker가 상호작용합니다. 이 접근법은 강력한 LLM에 대해 단일 라운드만으로도 성공할 수 있으며, 여러 라운드가 필요한 경우에도 교차점을 이용해 일관성을 높이는 방법을 설명합니다.

- **Performance Highlights**: 제안된 방법은 다양한 LLM에서 거의 완벽한 jailbreak 성공률을 달성하였으며, 이는 현재의 모델에서 존재하는 비합리적 학습 콘텐츠의 문제를 시사합니다. 실험은 Attack Success Rate(ASR), 효율성, 견고성 등의 주요 지표를 기반으로 하여 수행되었으며, 강력한 모델들은 첫 번째 라운드만으로 충분한 결과를 산출하였습니다. 이 연구는 LLM의 안전성에 관한 중요한 기초를 제공하며, 향후 연구에서 LLM이 악의적인 활용에 대한 방어를 강화할 수 있도록 지원합니다.



### The TIP of the Iceberg: Revealing a Hidden Class of Task-In-Prompt Adversarial Attacks on LLMs (https://arxiv.org/abs/2501.18626)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)에 대한 새로운 유형의 jailbreak 적대적 공격인 Task-in-Prompt (TIP) 공격을 소개합니다. TIP 공격은 모델의 프롬프트에 시퀀스-투-시퀀스(task) 작업(예: 암호 해석, 수수께끼, 코드 실행)을 포함시켜 금지된 입력을 간접적으로 생성합니다. 또한 PHRYGE 벤치마크를 도입하여 이러한 공격의 효과성을 체계적으로 평가합니다. 우리 연구는 GPT-4o 및 LLaMA 3.2를 포함한 여섯 개의 최첨단 언어 모델에서 안전 장치를 우회할 수 있음을 입증합니다.

- **Technical Details**: TIP 공격은 모델의 기본 지시 사항 수행 기능을 활용하여 특정 트리거 단어나 질문을 회피하고 안전하지 않은 콘텐츠를 무해한 변환 작업 내에 포함시키는 방식으로 진행됩니다. 이러한 공격은 Caesar cipher, 모스 부호, Base64 등 다양한 인코딩 방법을 사용할 수 있어 탐지하기 어려운 공격 유형입니다. LLMs가 임의의 언어 퍼즐이나 변환을 해결하도록 설계되어 있는 한, 공격자는 금지된 콘텐츠를 간접적으로 다시 도입하는 프롬프트를 만들 수 있습니다.

- **Performance Highlights**: TIP 공격은 기존의 방어 메커니즘을 우회할 수 있는 강력한 방법으로, 모델이 항상 신뢰할 수 있는 작동을 보장하기 위한 보다 정교한 방어 전략이 필요하다는 것을 강조합니다. 연구 결과는 LLM의 안전 정렬에서 존재하는 중요한 약점을 부각시키며, 향후 보안 및 신뢰성 향상을 위한 방향성을 제시합니다. 이러한 TIP 공격은 단순한 공격이 아닌, 전반적인 LLM의 안전성 문제를 야기하는 중요한 발견으로 평가됩니다.



### Multimodal Magic Elevating Depression Detection with a Fusion of Text and Audio Intelligenc (https://arxiv.org/abs/2501.16813)
Comments:
          21 pages,7 figures.1 table

- **What's New**: 이번 연구는 우울증 분류의 정확성을 높이기 위해 교사-학생 구조에 기반한 혁신적인 멀티모달 융합 모델(multimodal fusion model)을 제안합니다. 이 모델은 전통적인 방법의 한계를 극복하고 멀티헤드 어텐션(multi-head attention) 메커니즘과 가중치 멀티모달 전이 학습(weighted multimodal transfer learning)을 도입하였습니다. 이러한 접근은 DAIC-WOZ 데이터셋을 활용하여 학생 융합 모델(student fusion model)이 텍스트 및 오디오 교사 모델에 의해 가이드되어 상당한 분류 정확도 향상을 달성했음을 보여줍니다.

- **Technical Details**: 제안된 모델은 텍스트와 오디오 특성 간의 상호 보완성을 효과적으로 포착하면서 교사 모델의 기여도를 동적으로 조절하여 일반화 능력을 향상시킵니다. 실험에서는 제안된 모델이 테스트 세트에서 99.1%의 F1 점수를 기록하며, 단일 모달(unimodal) 및 기존 접근 방식에 비해 현저히 나은 성능을 보여주었습니다. 어블레이션 실험(ablation experiments)을 통해 모델의 robustness와 adaptability를 검증하였습니다.

- **Performance Highlights**: 이번 연구의 실험 결과는 제안된 프레임워크가 복잡한 멀티모달 데이터 처리에 있어 강력하고 적응력이 뛰어난 성능을 발휘함을 강조합니다. 지식의 융합과 특성 추출의 한계를 다루는 새로운 통찰력을 제공하며, 멀티모달 대규모 모델 학습(multipodal large model learning) 분야에 혁신적인 기술 프레임워크를 제시합니다.



New uploads on arXiv(cs.IR)

### mFollowIR: a Multilingual Benchmark for Instruction Following in Retrieva (https://arxiv.org/abs/2501.19264)
Comments:
          Accepted to ECIR 2025

- **What's New**: 본 논문에서는 mFollowIR이라는 다국어 벤치마크를 소개하여, 다국어 검색 모델의 명령 따르기(Instruction-following) 능력을 평가하고자 합니다. 이전의 연구에서는 주로 영어에 집중되었던 반면, mFollowIR은 러시아어, 중국어, 페르시아어로 된 쿼리와 지침을 포함한 새로운 데이터셋을 구축하여 전 세계의 다양한 사용자를 위해 다국어 지원의 필요성을 강조합니다. 이를 통해 다국어 모델의 성능을 평가하고, 모델이 복잡한 지침을 얼마나 잘 따르는지를 분석하고자 합니다.

- **Technical Details**: mFollowIR은 TREC NeuCLIR 내러티브를 기반으로 구축되었으며, 각 언어별로 문서의 관련성 평가와 지침을 제공하는 작업으로 구성됩니다. 이를 통해 논문에서는 검색 모델이 지침을 얼마나 효과적으로 따르는지 평가할 수 있는 새로운 평가 절차를 제시합니다. 또한, 영어에서 훈련된 검색기들이 다국어 환경에서도 어느 정도 성능을 발휘함을 보여주며, 추가적인 개선이 필요하다는 점도 지적합니다.

- **Performance Highlights**: 결과적으로, 다국어 및 교차언어 IR 모델은 복잡한 지침을 제공했을 때 적절한 관련성 점수를 수정하는데 어려움을 겪었습니다. 그러나 지침-trained retriever의 경우 언어 기반 훈련이 수행되었을 때 상대적으로 더 나은 결과를 나타내었으며, 이는 저희 연구가 제시한 다국어 지침 따르기 접근 방식의 장점을 시사합니다. 이 결과들은 향후 연구가 더 발전된 다국어 검색 모델을 구축하기 위한 기반이 될 수 있음을 보여줍니다.



### Emancipatory Information Retrieva (https://arxiv.org/abs/2501.19241)
- **What's New**: 이 논문은 정보 접근 방식이 인권 및 사회 정의 문제와 어떻게 관련되는지를 탐구하는 새로운 시각을 제공합니다. 특히, 컴퓨터 매개 정보 접근이 억압에 맞서 싸우는 데 중요한 역할을 하며, 정보 검색(Information Retrieval) 분야가 인도적 가치에 기반하여 재구성돼야 한다고 주장합니다. 또한, 억압적 구조에 대항하는 정보 검색의 중요성을 강조하고, 연구 커뮤니티가 이러한 가치를 내재화하여 사회적 역할을 다할 것을 촉구합니다.

- **Technical Details**: 이 논문에서 제안하는 'emancipatory information retrieval'은 정보 접근 방법을 연구하고 개발하여 다양한 형태의 인권 침해에 도전하는 것을 의미합니다. 이는 정보가 정치적이며, 연구와 개발이 모든 형태의 구조적 억압에 반대하는 보편적 투쟁을 지원하도록 하는 일련의 활동들을 포함합니다. 연구자들은 정보 접근 기술의 발전이 단일한 경로가 아니며, 사회과학, 인문학, 법률 및 정책 전문가들과의 협력 필요성을 강조합니다.

- **Performance Highlights**: 정보 검색 분야는 단순한 정보 조회 도구를 넘어서 우리의 일상, 문화, 정치 담론에 영향을 미치는 강력한 도구가 되었습니다. 현대의 검색 시스템들은 사용자의 행동을 모델링하고 예측하여 정보를 제공하며, 따라서 여러 집단의 목소리를 대변하는 역할을 하거나 기존의 사회적 위계를 강화할 수 있습니다. 이 논문은 정보 접근 시스템이 억압과 해방 간의 갈등을 해결하는 중간자가 될 가능성을 탐구하며, 연구자들이 인간 중심의 가치에 입각해 IR 연구의 방향성을 재조명해야 한다고 주장합니다.



### A Zero-Shot Generalization Framework for LLM-Driven Cross-Domain Sequential Recommendation (https://arxiv.org/abs/2501.19232)
Comments:
          11 pages

- **What's New**: 본 연구에서는 Zero-shot cross-domain sequential recommendation (ZCDSR) 문제를 다루고 있으며, 추가 훈련이나 파인튜닝 없이도 보이지 않는 도메인에서의 예측을 가능하게 합니다. 최근 대형 언어 모델(LLMs)의 발전을 통해 ZCDSR의 성능이 크게 향상되었으며, 이는 크로스 도메인 지식 전이를 촉진하는 데 도움을 주고 있습니다. 그러나 도메인 간 어휘 및 콘텐츠 초점의 차이로 인해 발생하는 도메인 의미적 편향(domain semantic bias)이라는 주요 문제가 여전히 존재합니다.

- **Technical Details**: 이 연구에서는 제안된 프레임워크가 아이템 레벨과 시퀀스 레벨 모두에서 LLM 기반 ZCDSR의 도메인 정렬을 개선하는 것을 목표로 하고 있습니다. 아이템 레벨에서는 유사한 아이템의 임베딩을 도메인 간 정렬하여 인터 도메인 콤팩트함을 촉진하는 일반화 손실(generalization loss)을 도입하였습니다. 시퀀스 레벨에서는 소스 도메인의 사용자 행동 패턴을 클러스터링하고, 타겟 도메인 추론을 위한 주의 기반 집계를 적용하는 방식을 개발하여 사용자 임베딩의 동적 적응성을 제공합니다.

- **Performance Highlights**: 다양한 데이터 세트와 도메인에서 진행된 종합적인 실험 결과, 제안된 프레임워크가 ZCDSR 환경에서 시퀀스 추천 성능을 상당히 향상시키는 것으로 나타났습니다. 도메인 편향을 완화하고 시퀀스 패턴의 전이 가능성을 강화함으로써, 본 방법은 도메인 간의 보다 효과적인 제로샷 recommandations을 위한 확장 가능하고 강력한 접근 방안을 제공합니다.



### Collaborative Diffusion Model for Recommender System (https://arxiv.org/abs/2501.18997)
Comments:
          WWW'25 short

- **What's New**: 이번 논문에서는 협업 기반의 확산 모델인 CDiff4Rec를 제안하여 개인화 정보 손실을 효과적으로 완화하고자 합니다. CDiff4Rec는 아이템 특징에서 의사 사용자를 생성하고, 행동 유사성을 기반으로 찾아낸 실제 및 의사 개인화 이웃으로부터 협업 신호를 활용하여 사용자의 미세한 선호를 재구성합니다. 이러한 접근은 아이템 콘텐츠와 협업 신호의 통합을 통해 개인화 정보의 손실을 감소시킵니다.

- **Technical Details**: CDiff4Rec는 크게 세 단계로 구성됩니다. 첫 번째 단계에서는 아이템 기능(예: 리뷰 단어)을 기반으로 의사 사용자를 생성합니다. 두 번째 단계에서는 실제 사용자와 의사 사용자로부터 개인화된 상위 K 이웃을 식별합니다. 마지막으로 확산 과정에서는 실제 및 의사 이웃의 선호 정보를 통합하여 쿼리 사용자 선호의 정밀한 표현을 생성합니다.

- **Performance Highlights**: 세 개의 공개 데이터셋에서 수행된 실험 결과, CDiff4Rec는 다양한 경쟁 모델을 초월하여 개인화 정보 손실을 효과적으로 완화하는 성능을 보여주었습니다. 또한, 논문에서는 각 구성 요소의 효과를 검증할 수 있는 추가 분석을 제공합니다.



### Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study (https://arxiv.org/abs/2501.18805)
Comments:
          Accepted at the 47th European Conference on Information Retrieval (ECIR 2025)

- **What's New**: 이번 연구는 추천 시스템에서 분리된 표현(disentangled representations)의 비지도 학습이 모델의 추천 성능에 미치는 영향에 대한 실증 연구를 수행하였다. 기존의 연구들은 분리된 표현의 해석 가능성 해석에 주로 초점을 맞추었으나, 효과성에 대한 과학적인 분석은 부족하였다. 연구 결과, 추천 effectiveness와 disentanglement 간의 단순한 상관관계는 발견되지 않았고, 해석 가능성(interpretability)과의 긍정적인 상관 관계가 확인되었다.

- **Technical Details**: 추천 시스템에서 사용자 선호도를 나타내는 표현을 구축하기 위해 비지도 분리 표현 학습을 활용하며, 이 과정에서 사용되는 주요 모델은 변량 오토인코더(Variational Autoencoders, VAEs)와 생성적 적대 신경망(Generative Adversarial Networks, GANs)이다. 두 가지 기존 기능 할당(feature attribution) 방법이 적용되어 모델의 표현 해석 가능성을 정량화하고, 연구에서는 LIME 글로벌(LIME-global)과 SHAP 글로벌(SHAP-global)이라는 측정 방법을 통해 해석 가능성을 공정하게 측정하였다.

- **Performance Highlights**: 실험 결과, 보고된 모델의 추천 효과성과 disentangled 표현의 재현 가능성은 데이터셋에 따라 다르며, 최대 43%의 차이가 발생하는 것으로 나타났다. 또한, 연구는 해석 가능성과 효과성 간의 일관된 통계적 관계가 없음을 보여주고, 표기된 관계가 설득력이 없음을 확인하였다. 그렇지만, 해석 가능성이 높을수록 분리 표현의 질이 개선된다는 점에서 긍정적인 상관관계를 발견하였다.



### Hierarchical Multi-field Representations for Two-Stage E-commerce Retrieva (https://arxiv.org/abs/2501.18707)
- **What's New**: 본 논문에서는 구조화된 전자상거래 제품 데이터를 계층적 필드 레벨 표현으로 인코딩하기 위한 새로운 프레임워크인 Cascading Hierarchical Attention Retrieval Model (CHARM)을 제안합니다. CHARM은 새로운 블록 삼각 주의 메커니즘을 활용하여 제품 필드 간의 상호 의존성을 포착하고, 필드 레벨 표현과 집합 벡터를 생성하여 빠르고 효율적인 검색을 가능하게 합니다. 또한, 이 모델은 두 단계의 검색 파이프라인을 통해 초기 후보 선택과 다운스트림 순위를 위한 정밀한 조정을 지원합니다.

- **Technical Details**: CHARM의 핵심 구성요소는 블록 삼각 주의 구조로, 각 필드의 입력 토큰이 이 필드와 이전 필드의 모든 토큰을 주의 깊게 살펴볼 수 있게 합니다. 이를 통해, 각 제품 필드에 대한 표현 벡터를 생성하고, 이 벡터들은 서로 다른 수준의 세부 정보를 인코딩하여 제공됩니다. 이 계층적 구조는 고수준 쿼리와 정밀한 쿼리 모두에 대해 제품을 매칭하는 데 versatile한 접근을 점합니다.

- **Performance Highlights**: 실험 결과, CHARM 모델은 공개된 대규모 전자상거래 데이터셋에서 기존의 bi-encoder 방법보다 우월한 성능을 보였습니다. CHARM은 두 단계의 검색 프로세스를 통해 계산 비용을 줄이면서도 높은 정확도를 유지하며, 다른 쿼리와 제품 필드 간의 강력한 연관성을 통해 추가적인 설명 가능성을 제공합니다. 다양한 필드에서 관찰된 표현의 다양성과 복잡성은 CHARM이 전자상거래 제품 검색에서 어떻게 효과적으로 작동하는지를 입증합니다.



### Characterizing User Behavior: The Interplay Between Mobility Patterns and Mobile Traffic (https://arxiv.org/abs/2501.19348)
- **What's New**: 이 논문은 사용자 레벨에서 트래픽과 이동성 행동 간의 의존성을 탐구하는 새로운 접근 방식을 제시합니다. 기존 연구들이 주로 집계된 데이터 또는 도시 규모의 분석에 중점을 두었던 반면, 이 연구는 개별 사용자 레벨에서 이러한 행동을 분석하여 세분화된 행동 모델링을 가능하게 합니다. 이를 통해 이동성이 트래픽에 미치는 직접적인 영향과 그 반대의 관계를 파악하고자 합니다. 또한, 개인화된 통찰력을 개선하기 위해 의존성 중심의 분석 프레임워크를 도입합니다.

- **Technical Details**: 논문에서는 XDR (eXtended Data Records) 데이터를 활용하여 트래픽 패턴 및 공간적, 구조적, 사회적 이동성을 포함한 13개의 개별 특징을 분석하였습니다. 이를 통해 사용자 유형에 따른 트래픽과 이동성 패턴의 차이를 이해할 수 있는 명확한 특성 분류를 제공합니다. 본 연구는 Markov 모델을 통해 이동성 행동에서 트래픽 행동을 추론할 수 있는 기법을 발전시켜 프라이버시와 적응성을 동시에 유지하는 방법론을 제시합니다.

- **Performance Highlights**: 칠레 여러 주에서 1,337,719명의 사용자로부터 수집한 1주간의 XDR 데이터셋을 사용하여 제안된 방법론의 유효성을 검증하였습니다. 이 모델은 다양한 도시 상황에서도 이동성과 트래픽 프로필 간의 적합성과 비적합성을 정확하게 구분하는 능력을 보였습니다. 이를 통해 사용자 행동을 정확하게 추론하고, 두 데이터 간의 통합된 데이터셋 생성을 가능하게 하여 사실적이고 개인화된 디지털 서비스의 제공에 기여할 수 있음을 입증했습니다.



### Revisiting Projection-based Data Transfer for Cross-Lingual Named Entity Recognition in Low-Resource Languages (https://arxiv.org/abs/2501.18750)
Comments:
          Accepted at NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이번 논문에서는 Cross-lingual Named Entity Recognition (NER) 분야에서 데이터 기반의 전이 기법이 저자원 언어에서 효과적임을 입증하고자 하였습니다. 특히, back-translation 기법을 활용한 단어 정렬 개선과 기존에 추출된 후보자와 소스 개체를 매칭하는 새로운 투영 접근법을 제안합니다. 이를 통해 저자원 언어에서 다국어 모델보다 높은 성과를 달성할 수 있는 가능성을 나타냅니다.

- **Technical Details**: 연구에서는 세 가지 단계로 구성된 주석 투영 방법(역번역, NER 모델 적용, 레이블 다시 투영)에 집중하고, 단어 간 정렬 개선을 위한 두 가지 접근 방식을 제안합니다. 첫 번째로, 서로 다른 언어 간의 정렬 문제를 해결하기 위해 단어 간 관계의 역방향 정렬을 탐구하고, 두 번째로는 출처 개체와 추출된 후보 간의 매칭을 양 당사자 매칭 문제로 재정의합니다. 이는 프로젝션 정확도를 높이기 위한 기초 작업입니다.

- **Performance Highlights**: 57개 언어에 대해 실시된 광범위한 실험 결과, 제안된 접근 방식이 기존의 투영 기반 기법보다 뛰어난 성능을 보여주었습니다. 특히, 저자원 언어의 경우에도 모델 기반 방법을 초월하는 성과를 기록하며, 데이터 전이의 내구성을 강조합니다. 이로 인해 저자원 언어에서의 NER 성능 향상에 기여할 수 있는 가능성을 시사합니다.



New uploads on arXiv(cs.CV)

### LiDAR Loop Closure Detection using Semantic Graphs with Graph Attention Networks (https://arxiv.org/abs/2501.19382)
- **What's New**: 본 논문에서는 그래프 주의 신경망(Graph Attention Neural Networks)을 활용하여 새로운 루프 종결 감지(loop closure detection) 알고리즘을 제안합니다. 이 알고리즘은 의미론적 그래프(semantic graph)를 인코딩하여 장소 인식을 수행하고, 이어서 의미론적 등록(semantic registration)을 통해 6 DoF 상대 포즈 제약을 추정합니다. 주요 모듈인 의미론적 그래프 인코더와 그래프 비교 모듈을 통해 수행되는 이 알고리즘은, 두 그래프 벡터의 차이를 비교하여 성능 향상을 보여줍니다.

- **Technical Details**: 제안하는 알고리즘은 두 개의 주요 모듈로 구성됩니다. 첫 번째 모듈인 의미론적 그래프 인코더는 입력 포인트 클라우드의 공간적, 의미적 및 기하학적 정보를 효율적으로 인코딩하기 위해 그래프 주의 네트워크를 사용합니다. 두 번째로, 그래프 비교 모듈에서는 현재 스캔과 키프레임 스캔의 그래프 벡터를 비교하여 루프 종결 가능성을 식별합니다. 특히 셀프 어텐션(self-attention) 메커니즘을 적용하여 더 독창적인 그래프 벡터를 생성합니다.

- **Performance Highlights**: 공개 데이터 세트에서 수행한 광범위한 평가 결과, 제안하는 모델은 기존의 의미론적 그래프 알고리즘에 비해 13% 향상된 최대 F1 점수를 기록하며 더 높은 정확도와 견고성을 보입니다. 제안하는 두 모듈은 최소한의 메모리 및 계산 요구 사항으로 실시간으로 작동할 수 있어 기존 SLAM 프레임워크에 통합하기에 적합합니다. 이를 통해 커뮤니티의 발전에 기여하고자 전체 구현을 오픈소스로 제공하고 있습니다.



### PixelWorld: Towards Perceiving Everything as Pixels (https://arxiv.org/abs/2501.19339)
- **What's New**: 기존의 foundation models(기반 모델)은 시각적 입력을 픽셀 단위로, 텍스트 입력을 토큰 단위로 처리하는데, 이는 인간의 지각 방식과는 다르다. 이 논문에서는 모든 모달리티(모드)를 픽셀 입력으로 통합하는 ‘Perceive Everything as Pixels’ (PEAP) 접근 방식을 제안한다. 또한, PixelWorld라는 새로운 평가 도구를 통해 기존 모델의 성능을 측정한다.

- **Technical Details**: PixelWorld는 텍스트, 테이블, 코드, 다이어그램, 이미지 등 다양한 모달리티를 픽셀 공간으로 통합하여 평가하는 새로운 평가 도구로 개발되었다. 초기 연구 결과, PEAP를 사용한 모델은 멀티모달 데이터셋에서 토큰 기반 입력보다 우수한 성능을 보였다. 그러나 픽셀 기반 입력 처리를 통해 모든 모델의 추론 및 코딩 능력이 크게 감소한 것을 강조하며, 기반 모델의 지각 능력 향상이 필요함을 시사한다.

- **Performance Highlights**: PEAP를 통해 대형 모델이 비추론 작업에서 강력한 성능을 유지하는 반면, 작은 모델인 Phi-3.5-V는 성능 저하가 심각하다. 또한, PEAP는 텍스트 토큰 입력과 응집력이 높은 주의(attention) 패턴을 보이며, 공간 희소성(spatial sparsity)을 활용하여 성능을 크게 가속화할 수 있는 가능성을 보여준다. 이러한 결과들은 기존의 최전선 모델들이 픽셀 지각에 능숙하다는 점을 확인하나, 아직 개선할 여지가 남아있음을 나타낸다.



### Consistent Video Colorization via Palette Guidanc (https://arxiv.org/abs/2501.19331)
- **What's New**: 본 연구에서는 동영상 색체화(video colorization) 문제를 해결하기 위해 새로운 파이프라인을 제안합니다. 기존의 방법들은 색의 채도가 낮고 시간적 일관성이 결여된 문제가 있었는데, 본 논문에서는 Stable Video Diffusion(SVD)을 기반 모델로 사용하여 이 문제를 해결하고자 합니다. 또한, 팔레트 기반의 색 가이더(palette-based color guider)를 디자인하여 모델이 생동감 있는 색상을 생성하도록 도와줍니다.

- **Technical Details**: 제안된 메서드는 이미지에서 비디오로의 확산(diffusion) 모델을 정교하게 조정하는 방식으로 진행됩니다. 직접적인 조정은 종종 생성된 프레임이 색이 낮고 흐릿하게 보이도록 하는데, 이는 두 가지 주요 요인으로 인해 발생합니다: 기존의 회색값을 복원하는 '단축키' 방식과 훈련 데이터의 색 분포에 대한 민감성입니다. 이러한 문제를 해결하기 위해, 운동 정보의 일관성을 줄이기 위해 색상 팔레트를 전역 가이드로 사용합니다.

- **Performance Highlights**: 정량적 및 정성적 실험 결과, 제안된 자동 동영상 색체화 방식은 색의 채도와 비디오 품질 면에서 기존의 방법을 능가하는 것으로 나타났습니다. 특히, 우리의 방법은 색상의 생동감과 시간적 일관성을 동시에 해결할 수 있는 통합된 확산 기반 프레임워크를 제공합니다. 팔레트를 활용한 전역 가이드는 비디오의 장시간 안정성을 해결하고 다양한 입력 형식을 수용할 수 있습니다.



### Let Human Sketches Help: Empowering Challenging Image Segmentation Task with Freehand Sketches (https://arxiv.org/abs/2501.19329)
- **What's New**: 이번 연구에서는 스케치를 활용하여 camouflaged object detection (COD)의 성능을 혁신적으로 향상시키는 접근 방식을 소개합니다. 스케치 입력을 사용하는 상호작용적 분할 프레임워크가 개발되어, 사용자들은 전통적인 바운딩 박스나 점 대신 손으로 그린 윤곽선을 통해 객체를 주석 처리할 수 있게 됩니다. 이로 인해 기존의 분할 모델들이 가졌던 한계를 극복하고 성능을 획기적으로 개선할 수 있습니다.

- **Technical Details**: 우리는 DeepSketchCamo라는 새로운 네트워크 구조를 설계하여, 스케치를 입력으로 사용하여 기존의 이미지를 더욱 효과적으로 분할할 수 있게 했습니다. 스케치 인코더가 손으로 그린 스케치를 인코딩하고, 이를 이미지 특성과 결합하여 최종 분할 마스크를 생성하는 방식입니다. 또한, 스케치의 경계 부정확성을 보완하기 위해 경계 정제 모듈과 적응형 초점 손실 구성 요소를 도입하여 보다 안정적인 훈련과 성능 향상을 도모할 수 있었습니다.

- **Performance Highlights**: 우리의 실험 결과는 DeepSketchCamo가 기존의 SAM 네트워크보다 18% 이상의 성능 향상을 보여주었음을 입증합니다. 스케치를 사용한 주석 처리는 라벨링 시간도 최대 120배까지 단축할 수 있어, 자원 집약적이고 노동 집약적인 픽셀 수준의 주석 의존도를 줄이는 데 크게 기여할 것으로 기대됩니다. 또한, KOSCamo+라는 최초의 손그림 스케치 데이터셋을 통해 향후 연구의 기초 자료를 제공할 것입니다.



### A Generic Hybrid Framework for 2D Visual Reconstruction (https://arxiv.org/abs/2501.19325)
- **What's New**: 이 논문에서는 정사각형의 비겹치는 조각들로 이루어진 퍼즐 문제를 해결하기 위한 다목적 하이브리드 프레임워크를 제시합니다. 본 접근법은 퍼즐 조각 쌍을 전체적으로 평가하는 딥러닝 기반의 호환성 측정 모델을 통합하며, 전통적으로 인접한 엣지에만 초점을 맞춘 방식에서 벗어났습니다. 최적화된 유전자 알고리즘 기반의 솔버와 결합하여 퍼즐 조각의 쌍별 호환성 점수를 활용하여 글로벌 최적 배치를 반복적으로 찾아냅니다.

- **Technical Details**: 하이브리드 프레임워크는 (1) 퍼즐 조각의 전체 분석을 통해 인접성을 평가하는 딥러닝 기반 호환성 측정 모델과, (2) 전 세계적인 조각 배치를 최적화하는 강화된 유전자 알고리즘 기반 솔버로 구성됩니다. 이러한 접근법은 수작업으로 추출한 특징 없이는 견고성을 높이며, 여러 데이터 세트에서의 적용 가능성을 향상시킵니다. 이 방법론은 복잡한 460조각 포르투갈 타일 패널의 성공적인 복원을 보여 줍니다.

- **Performance Highlights**: 제안된 하이브리드 방법론은 여러 실제 세계 시나리오에서 재구성 품질을 보여주며, 포르투갈 타일 패널, 경계가 침식된 퍼즐, 찢긴 문서 등을 포함한 다양한 사례에서 테스트되었습니다. 실험 결과는 뛰어난 적응성과 강인성을 강조하며, 기존 방법들을 초과하여 최신 기술 수준(SOTA) 성과를 달성했습니다. 또한, 포르투갈 타일 패널의 새로운 벤치마크 데이터 세트를 수집하고 공개하여 향후 연구를 지원합니다.



### Advancing Dense Endoscopic Reconstruction with Gaussian Splatting-driven Surface Normal-aware Tracking and Mapping (https://arxiv.org/abs/2501.19319)
Comments:
          Accepted by ICRA 2025

- **What's New**: Endo-2DTAM is an innovative real-time endoscopic SLAM system that integrates 2D Gaussian Splatting (2DGS) to improve accuracy in surgical scene reconstruction. 기존의 3D Gaussian Splatting (3DGS) 기술은 다중 뷰 불일치로 인해 깊이 및 표면 재구성이 어려운 반면, Endo-2DTAM은 이를 해결하기 위해 표면 노멀 정보를 활용합니다. 본 시스템은 추적, 매핑 및 번들 조정 모듈로 구성되어 geometrically accurate한 재구성을 제공합니다. 또한 초기 키프레임 샘플링을 위한 pose-consistent 전략을 제안하여 효율성을 극대화했습니다.

- **Technical Details**: Endo-2DTAM은 2D Gaussian Splatting (2DGS)를 기반으로하는 RGB-D SLAM 시스템으로, endoscopic 환경에서의 정확한 위치 추적 및 매핑을 위해 설계되었습니다. 시스템의 구조는 2D Gaussian representation, tracking, gaussian expanding 및 keyframe sampling, mapping, bundle adjustment의 다섯 가지 주요 요소로 구성됩니다. 점 대 점 및 점 대 평면 거리 메트릭을 결합하여 기존 SLAM 시스템의 단점을 보완하며, 매핑 모듈에서는 표면 노멀 일관성과 깊이 왜곡을 동시에 고려합니다. 이 누적된 정보로 인해 모델의 정밀도를 개선할 수 있습니다.

- **Performance Highlights**: Endo-2DTAM은 공용 endoscopic 데이터셋에서 extensive 실험을 진행한 결과, depth reconstruction에서 1.87±0.63 mm의 RMSE를 달성하였습니다. 이와 함께 계산 효율성, 고품질 시각적 표현, 실시간 렌더링을 유지하며, 기존 기술 대비 향상된 성능을 보여주었습니다. 특히 수술 장면의 재구성에서 state-of-the-art 성능을 기록하여, 의료 분야에 적용 가능성을 높였습니다. 본 연구의 결과는 endoscopic 수술의 안전성과 효율성을 증대시키는데 중요한 기여를 할 것으로 기대됩니다.



### Application of Generative Adversarial Network (GAN) for Synthetic Training Data Creation to improve performance of ANN Classifier for extracting Built-Up pixels from Landsat Satellite Imagery (https://arxiv.org/abs/2501.19283)
- **What's New**: 이번 연구에서는 저해상도 Landsat 이미지에서 Built-Up 픽셀을 식별하기 위한 인공 신경망(ANN) 분류기의 성능을 향상시키기 위해 Generative Adversarial Network(GAN) 아키텍처를 제안했습니다. 기존의 훈련 데이터의 부족 문제를 해결하기 위해 GAN을 활용하여 합성 훈련 픽셀을 생성함으로써, ANN 모델의 정확도를 크게 개선할 수 있었습니다. 이 방법은 Built-Up과 Non Built-Up 지역의 구분을 더욱 정교하게 만들어 줍니다.

- **Technical Details**: 제안된 GAN 아키텍처는 실제 Built-Up 픽셀 집합을 이용하여 합성 Built-Up 픽셀을 생성합니다. 이 과정에서, Neural Network를 통해 입력 잡음 변수를 데이터 공간으로 매핑하며, D 네트워크는 실제 데이터로부터의 확률을 학습합니다. Kolmogorov Smirnov Test와 Ball Divergence 기반의 분포 동등성 검사를 통해 생성된 픽셀과 원본 픽셀 간의 분포가 indistinguishable한지 검증했습니다.

- **Performance Highlights**: 실험 결과, ANN 모델의 전체 정확도가 0.9331에서 0.9983으로, Kappa 계수가 0.8277에서 0.9958로 지속적으로 향상되었습니다. 생성된 합성 픽셀을 포함한 훈련 데이터의 사용 덕분에 분류 성능이 크게 개선되었음을 보여줍니다. 이는 향후 저해상도 이미지 데이터에서도 효과적인 분류 작업을 가능하게 할 전망입니다.



### Imagine with the Teacher: Complete Shape in a Multi-View Distillation Way (https://arxiv.org/abs/2501.19270)
Comments:
          9 pages, 3 figures 4 tables

- **What's New**: 이 논문에서는 부분적인 관찰로 인한 3D 물체 형상의 완성을 위한 새로운 접근 방식을 제안합니다. 저자들은 View Distillation Point Completion Network (VD-PCN)을 설계하여 다중 시점(distillation) 방법을 통해 포인트 클라우드의 완성 문제를 해결합니다. 이 과정에서 2D 픽셀의 정돈성과 유연한 처리력을 최대한 활용하여 기존 모델들의 한계를 극복하고자 했습니다.

- **Technical Details**: VD-PCN은 기존의 포인트넷 구조와 다른 접근 방식을 사용합니다. 이 모델은 다중 시점을 고려한 인코더-디코더 구조를 통해 부분적인 포인트 클라우드에서 원본 포인트 클라우드 정보를 재통합하여 최적의 결과를 도출합니다. 특히, 지식 전이(strategy) 기술을 통해 교사 모델이 학습한 지식을 학생 모델에 전달함으로써, 포인트 예측 및 특성 학습을 개선하는 방식으로 성능을 강화하고 있습니다.

- **Performance Highlights**: 저자들은 다양한 데이터셋을 통해 VD-PCN의 성능을 평가하였으며, 기존의 최첨단(neural network) 포인트 클라우드 완성 네트워크에 비해 우수한 결과를 기록했습니다. 이 논문의 결과는 향후 관련 연구에 상당한 기여를 할 것으로 보이며, 코드가 공개되어 재현 가능성과 후속 연구를 위한 토대를 제공합니다.



### Medical Semantic Segmentation with Diffusion Pretrain (https://arxiv.org/abs/2501.19265)
- **What's New**: 이 논문은 3D 의료 이미징을 위해 설계된 새로운 pretraining 전략을 제안합니다. 기존의 contrastive learning 방식을 비판하며, anatomical guidance를 이용한 diffusion 모델 기반의 접근법을 사용합니다. 이 방법은 3D 의료 이미지 데이터의 복잡성을 고려하여 일반화 가능한 feature representations을 생성하는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법은 의료 이미지의 복잡한 해부학적 구조를 이해하는 데 유용한 3D 보편적 신체 부위 좌표를 예측하는 추가 모델과 함께 작동합니다. 모델은 고해상도의 이미지를 작은 패치 단위로 분할하고, 각 패치의 서로 겹치는 부분에서 평균 값으로 aggregation을 수행하는 방식을 사용합니다. 또한, 기하학적 관계를 복원하기 위해 이미지 내의 육체적 좌표에 대한 추가 정보를 활용하여 생성된 이미지의 질을 향상시킵니다.

- **Performance Highlights**: 본 논문의 실험 결과는 제안된 pretraining 방법이 기존 방법보다 7.5% 향상된 성능을 보이며, 다양한 다운스트림 분할 작업에 대해 67.8의 평균 Dice 계수를 기록하여 state-of-the-art contrastive pretraining 접근법과 경쟁할 수 있음을 보여줍니다. 특히, 이 방법은 3D organ segmentation 작업에서의 효과성을 입증하며, 기존의 복원 pretraining 방법과 비교하여 더욱 정보적인 voxel-level feature representations을 생성할 수 있다는 것을 강조합니다.



### ContextFormer: Redefining Efficiency in Semantic Segmentation (https://arxiv.org/abs/2501.19255)
- **What's New**: 본 논문에서 제안하는 ContextFormer는 CNN과 ViT의 장점을 활용하여 실시간 의미 분할을 위한 하이브리드 프레임워크이다. 기존의 연구들이 인코더 아키텍처에 집중하고 있었던 반면, 서로 다른 맥락의 효율성을 증대시키기 위한 병목 부분을 탐구하였다. 이 프레임워크는 높은 해상도 이미지에서의 정확한 예측과 효율성을 조화롭게 이룬다.

- **Technical Details**: ContextFormer는 세 가지 주요 모듈로 구성된다: Token Pyramid Extraction Module (TPEM)은 고해상도와 저해상도의 특징을 결합하여 피라미드 형태의 표현을 생성한다. 이어서 Trans-MDC 블록은 다층 토큰을 처리하여 전역 의존성과 미세 세부사항을 동적으로 조정한다. 마지막으로 Feature Merging Module (FMM)은 다양한 수준의 지역 및 전역 의미적 특징을 융합하여 강력한 예측 결과를 제공한다.

- **Performance Highlights**: ContextFormer는 ADE20K, Pascal Context, CityScapes 및 COCO-Stuff 데이터셋에서의 실험을 통해 기존 모델에 비해 월등한 성능을 입증하였다. 특히, mIoU 점수에서 최첨단의 결과를 달성하여 효율성과 성능의 새로운 기준을 세웠다. 이 방법론은 실시간 배포를 위해 최적화된 고충실 예측을 가능하게 한다.



### Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search (https://arxiv.org/abs/2501.19252)
Comments:
          Website: this https URL

- **What's New**: 이번 논문에서는 텍스트-비디오 디퓨전 모델의 출력 품질을 개선하기 위한 새로운 방법인 Diffusion Latent Beam Search (DLBS)를 제안합니다. 이 방법은 기존의 메트릭을 보정하여 비디오 생성에서 더욱 높은 입력 품질을 도출할 수 있도록 해줍니다. 특히, 이 과정에서 AI 피드백을 활용하여 더 나은 비디오를 생성하는 데 중점을 둡니다.

- **Technical Details**: DLBS는 탐색 예측기(lookahead estimator)를 사용하여 비디오 생성 과정에서 주어진 정렬 보상을 극대화할 수 있는 최적의 디퓨전 잠재(latent)를 선택합니다. 또한, 비디오의 자연스러움을 평가하기 위해 여러 메트릭을 가중치 선형 조합으로 설계하며, 이는 평가 프롬프트의 동적 서술 정도에 크게 의존함을 보입니다. 이 과정에서 기존의 메트릭들을 재조정하여 정밀한 평가 및 품질 향상을 도와줍니다.

- **Performance Highlights**: 제안된 방법은 모델 파라미터 업데이트 없이도 보정된 보상을 기반으로 높은 품질의 출력을 생성할 수 있음을 보여줍니다. 기존의 탐욕적 검색(greedy search) 및 N개 중 최상의 샘플링(best-of-N sampling) 방식과 비교할 때, 제안된 방법은 동일한 검색 예산 내에서 최고의 생성 결과를 제공합니다. 이 논문에서는 검색 예산, 보상 추정화를 위한 탐색 단계, 디노이징 과정에서의 계산 할당에 대한 실용적인 가이드를 제공하여 효율성을 높이는 전략을 제시합니다.



### Accelerating Diffusion Transformer via Error-Optimized Cach (https://arxiv.org/abs/2501.19243)
- **What's New**: 이 논문에서는 Diffusion Transformer(DiT) 모델의 샘플링 시간을 줄이기 위해 Error-Optimized Cache(EOC)라는 새로운 방법을 제안합니다. 기존의 캐싱 방법은 저오류 모듈을 재사용하여 생성 속도를 높이는 데 초점을 맞추었으나, 캐싱으로 인해 발생하는 오류를 충분히 분석하지 않아 품질 하락을 초래합니다. EOC는 캐싱 최적화가 필요한 모듈을 판단하고, 오류를 줄이기 위한 제어된 자극을 적용하여 품질을 개선합니다.

- **Technical Details**: EOC는 주로 세 가지 핵심 개선 사항으로 구성되어 있습니다: (1) 사전 지식 추출: 캐싱의 차이를 추출하고 처리합니다; (2) 캐싱 최적화를 위한 판단 방법: 최적화가 필요한 캐싱 단계를 결정합니다; (3) 캐시 최적화: 캐싱 오류를 줄입니다. 이러한 접근 방식은 특히 ImageNet 데이터셋에서 과도한 캐싱에 대한 오류 축적을 유의미하게 감소시킵니다.

- **Performance Highlights**: 실험 결과, EOC는 계산 부하를 크게 증가시키지 않으면서 생성된 이미지의 품질을 향상시키는 것으로 나타났습니다. 특히 Fréchet Inception Distance(FID) 값에서 대폭 개선된 결과를 보이며, 각각 6.857에서 5.821, 3.870에서 3.692, 3.539에서 3.451로 감소했습니다. 이는 EOC가 생성 품질을 차별화하는 데 효과적임을 시사합니다.



### Integrating Semi-Supervised and Active Learning for Semantic Segmentation (https://arxiv.org/abs/2501.19227)
- **What's New**: 이번 논문에서는 새로운 액티브 러닝 접근 방식을 제안하며, 이는 개선된 준지도 학습 프레임워크와 통합되어 매뉴얼 주석 비용을 줄이고 모델 성능을 향상시킵니다. 이 접근 방식은 액티브 러닝을 통해 선택된 레이블이 있는 데이터와 선택 과정에서 제외된 레이블이 없는 데이터를 동시에 활용하여, 잘못된 가짜 레이블(pseudo-labels)의 가능성이 있는 영역을 정확히 지적합니다.

- **Technical Details**: 제안된 구조에서는 주석 예산을 증가시키지 않으면서, 군(cluster) 가정을 기반으로 하여 레이블이 없는 데이터에서 가장 어려운 부분만 매뉴얼 레이블링이 적용됩니다. 또한, 가짜 레이블 자동 정제(pseudo-label auto-refinement, PLAR) 모듈을 통해 잘못된 가짜 레이블을 가진 픽셀을 교정하며, 이 과정에서 특징(feature) 표현을 기반으로 정확성과 효율성을 높입니다.

- **Performance Highlights**: 제안된 혼합 준지도 액티브 러닝 프레임워크는 두 가지 벤치마크 데이터셋에서 평가되었으며, 자연 및 원격 측정 이미지 도메인에서 최고의 성능을 기록하였습니다. 또한 이 방법은 semantic segmentation 작업에서 최신의 방법론을 초과하는 성과를 보여주었습니다.



### RaySplats: Ray Tracing based Gaussian Splatting (https://arxiv.org/abs/2501.19196)
- **What's New**: 본 논문에서는 RaySplats라는 새로운 모델을 소개하며, 이는 3D Gaussian Splatting (3DGS)의 방법론을 확장하여 ray tracing 기술을 통합합니다. 기존의 3DGS는 rasterization 방식을 사용하여 빛과 그림자 효과를 구현하는 데 한계를 보였으나, RaySplats는 Gaussian 프리미티브를 직접 ray tracing을 통해 처리하여 이러한 한계를 극복합니다. 또한 RaySplats는 2D Gaussian을 포함한 메쉬 기반 모델과 쉽게 통합될 수 있습니다.

- **Technical Details**: RaySplats는 Gaussian 분포와 레이의 교차점을 계산하여 색상을 집계하는 메커니즘을 제안합니다. 이 방법은 light conditions와 메쉬 기반 물체를 통합하는 데 효과적이며, polytope 근사를 사용하지 않습니다. 특이하게도 RaySplats는 ray tracing을 훈련 및 추론 과정 모두에서 활용하여 더욱 정교한 광원 효과를 가능하게 합니다.

- **Performance Highlights**: RaySplats를 통해 생성된 렌더링 결과는 기존 메커니즘보다 높은 품질을 자랑하며, 조명 효과와 함께 반사, 그림자, 투명성을 처리하는 데 있어 새로운 가능성을 열어줍니다. 이 모델은 3D Gaussian Splatting의 적용 범위를 넓히고, 다양한 응용 분야에서 더욱 발전된 시각적 품질을 제공합니다.



### A Survey on Class-Agnostic Counting: Advancements from Reference-Based to Open-World Text-Guided Approaches (https://arxiv.org/abs/2501.19184)
- **What's New**: 최근 객체 카운팅(Object Counting) 연구가 'Class-Agnostic Counting'(CAC) 영역으로 변화하고 있으며, 이는 다양한 카테고리의 객체를 세는 문제를 다룰 수 있는 모델 개발을 목표로 하고 있습니다. 이 논문에서는 CAC 방법론을 처음으로 세 가지 패러다임으로 분류하고, 기존의 성과를 설명하며, 현재의 문제점과 미래 연구 방향을 제시합니다. 또한, state-of-the-art 기술들을 분석하고, 각각의 장단점과 도전 과제를 논의하여 관련 연구자들에게 유용한 자료를 제공합니다.

- **Technical Details**: CAC는 크게 세 가지 범주로 나뉘며, 각각은 'Reference-based', 'Reference-less', 그리고 'Open-world text-guided' 방식으로 구분됩니다. Reference-based 방식은 주어진 예제를 기반으로 성능을 유지합니다. Reference-less 방식은 예제에 의존하지 않고 이미지 내 패턴을 이용하여 카운팅을 수행하며, Open-world text-guided 방식은 비전-언어 모델을 사용하여 객체 클래스의 설명을 텍스트 프롬프트를 통해 가능하게 합니다. 이들 각 방식의 아키텍처와 성능을 비교하는 것도 본 연구의 중요한 부분입니다.

- **Performance Highlights**: 논문에서는 FSC-147 데이터셋과 CARPK 데이터셋 등 기존의 금 표준 벤치마크에서의 성능 결과를 비교하여 각 방법론의 성능을 강조합니다. 많은 최신 CAC 방법들이 과거의 전통적인 방법들에 비해 우수한 성능을 보여주고 있으며, 특히 새로운 paradigms는 인간의 개입을 줄이는데 기여하고 있습니다. 또한 카운팅 성능은 관찰된 객체의 전반적인 범주에 따라 현저히 달라지며, 다양한 연구의 결과를 통해 강점을 분석하고 지속적인 도전 과제를 정의하여 향후 연구의 방향을 제시합니다.



### Poison as Cure: Visual Noise for Mitigating Object Hallucinations in LVMs (https://arxiv.org/abs/2501.19164)
- **What's New**: 본 논문은 LVM(대형 비전-언어 모델)의 환각(object hallucination) 문제를 완화하기 위한 시각적 적대적 섭동(VAP) 방법을 제안합니다. 기존 방법들이 LVM 내부 메커니즘을 수정하는 것과 달리, 우리의 방법은 기본 모델을 변경하지 않고 전략적으로 최적화된 시각적 노이즈를 적용하여 환각을 경감합니다. 이는 LVM이 시각적 콘텐츠에 대한 응답을 원할하게 처리할 수 있도록 유도합니다.

- **Technical Details**: 우리의 VAP 방법은 환각 완화 문제를 최적화 문제로 서술하며, 적대적 전략을 활용하여 모델의 인과 관계를 강화하고 매개 지식의 편향(bias)을 줄이는 시각적 섭동을 생성합니다. 이 접근법은 시각적 입력을 활용하여 모델의 의사결정을 개선하는 데이터를 중심으로 한 방식입니다. VAP는 또한 완전한 블랙 박스 환경에서도 작동하여 LVM에 대한 수정 없이 적용할 수 있는 실용적인 솔루션입니다.

- **Performance Highlights**: 다양한 평가 설정에서 8개의 최신 LVM에 대한 실험 결과, VAP는 일관되게 환각을 줄이는 효과를 보여줍니다. 우리는 POPE 및 BEAF와 같은 보완적인 환각 평가 프레임워크를 사용하여 VAP의 효과성을 검증했습니다. 전반적으로, 우리의 방법은 LVM의 신뢰성 향상에 중요한 기여를 할 것으로 기대됩니다.



### RMDM: Radio Map Diffusion Model with Physics Informed (https://arxiv.org/abs/2501.19160)
- **What's New**: 이번 연구에서는 라디오 맵 재구성을 위한 새로운 접근법인 **Radio Map Diffusion Model (RMDM)**를 제안합니다. RMDM은 **Physics-Informed Neural Networks (PINNs)**와 **Helmholtz equation**을 통합하여 물리적 제약을 포함함으로써 더욱 정확한 결과를 도출할 수 있도록 설계되었습니다. 이 모델은 다양한 환경에서의 일반화 능력을 향상시키고 데이터가 부족한 상황에서도 높은 성능을 발휘합니다.

- **Technical Details**: RMDM은 이중 U-Net 아키텍처를 채택하여 물리적 일관성을 보장합니다. 첫 번째 U-Net은 PDE 잔차, 경계 조건, 출처 제약을 최소화하여 물리적 일관성을 확보하고, 두 번째 U-Net은 확산 기반의 노이즈 제거를 통해 예측을 정교화합니다. 이러한 구조는 Helmholtz 방정식을 손실 함수에 통합함으로써 모델의 예측 능력을 강화하고 다양한 환경에서도 일반화가 가능하도록 만듭니다.

- **Performance Highlights**: 실험 결과 RMDM은 **NMSE 0.0031**과 **RMSE 0.0125**를 기록하여 최신의 다른 방법들에 비해 뛰어난 성능을 보여주었습니다. 특히, 정적 라디오 맵(SRM) 설정 하에서 탁월한 성과를 이룩하였으며, 다양한 도전적인 시나리오에서도 모델의 재구성 능력을 입증하였습니다. 이러한 성과는 물리 기반 방법과 데이터 기반 접근 방식을 결합한 새로운 패러다임을 확립하는 데 기여합니다.



### GDO: Gradual Domain Osmosis (https://arxiv.org/abs/2501.19159)
Comments:
          submitted to icml 2025

- **What's New**: 이 논문에서는 Gradual Domain Osmosis라는 새로운 방법을 제안합니다. 이는 Gradual Domain Adaptation(GDA)에서 발생하는 지식 이전 문제를 해결하기 위해 개발되었습니다. 이 접근법은 중간 도메인을 통해의 지식 이동의 비효율성을 줄이고 효율적인 지식 이식을 가능하게 합니다.

- **Technical Details**: 이 방법은 하이퍼파라미터 λ를 기반으로 하여 소스와 타겟 도메인의 손실 가중치를 동적으로 조절하는 최적화 프레임워크를 설계합니다. 이를 통해 훈련 과정에서 λ 값이 0에서 1까지 증가하며, 지식 이전의 강도를 점진적으로 조정하고, 중간 도메인에서 안정성과 견고성을 확보합니다.

- **Performance Highlights**: 실험에서는 Rotated MNIST, colour-shifted MNIST, Portrait 데이터셋, Forest Cover Type 데이터셋을 사용하여 이 방법의 효과를 검증했습니다. 결과는 기존의 기준 방법들을 능가하는 성능을 나타냅니다. 또한, 하이퍼파라미터 λ의 동적 조정 전략이 성능에 미치는 영향을 분석하여 도메인 편향을 완화하고 모델의 일반화 능력을 향상시키는 장점을 확인했습니다.



### SWAT: Sliding Window Adversarial Training for Gradual Domain Adaptation (https://arxiv.org/abs/2501.19155)
Comments:
          submitted to icml 2025

- **What's New**: 본 논문은 Sliding Window Adversarial Training (SWAT)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 기존의 Gradual Domain Adaptation (GDA) 방법의 한계를 극복하고, 점진적으로 지식 전이를 수행하여 도메인 간의 격차를 줄입니다. 특히, SWAT는 적대적 흐름을 구축하여 중간 도메인 간의 작은 간격을 점차적으로 좁히는 슬라이딩 윈도우 메커니즘을 활용합니다.

- **Technical Details**: SWAT는 적대적 학습(adversarial learning)과 커리큘럼 슬라이딩 윈도우(curriculum sliding window) 메커니즘을 결합하여 모델을 동적으로 조정된 중간 도메인에 맞춥니다. 이 방법은 sliced Wasserstein 최적화(sliced Wasserstein optimization)를 통해 도메인 연속성을 강화하고, 적대적 감쇠(adversarial attenuation)를 통해 소스 특성을 점진적으로 필터링합니다. 그 결과, 도메인 불변성과 목표 구별력이 동시에 향상됩니다.

- **Performance Highlights**: SWAT는 Rotated MNIST(원판 MNIST)과 Portraits 데이터셋에서 현재 최첨단 모델 대비 각각 10.3% 및 1.24% 향상된 성능을 기록하였습니다. 또한, 극단적인 도메인 변화에서 36.0%의 오류 감소를 보여주는 실험 결과가 수록되어 있습니다. 이러한 성과는 SWAT의 접근 방식이 기존 방식에 비해 우수하다는 것을 입증합니다.



### Transformation trees -- documentation of multimodal image registration (https://arxiv.org/abs/2501.19140)
Comments:
          18 pages, 11 figures

- **What's New**: 이 논문은 다중 모드 이미징(multi-modal imaging) 데이터의 등록(registration) 결과로 얻어진 변환(transformation) 집합을 문서화하기 위한 트리 구조(tree structure)의 적용 방안을 제시합니다. 또한 새롭게 도입된 파일 형식인 .dpw(digital patient workspace)를 통해 환자 맞춤형 좌표계에서 이미지를 시각화하는 데 필요한 다양한 작업을 처리하는.dpVision 소프트웨어의 활용 예를 보여줍니다.

- **Technical Details**: 의료 이미징의 데이터 통합(integration)은 여러 이미지 간의 변환 과정이 필요하며, 이 과정에서는 동일한 공간 위치를 나타내는 데이터 세트가 필요합니다. 이 과정에서 수학적으로는 서로 대응하는 포인트(point) 간의 거리를 최소화하는 변환을 찾는 등록(process) 기술이 중요합니다. .dpw 형식은 각 객체의 부모-자식 관계를 명확히 정의하고, 3D 이미지를 처리하기 위한 계층적 데이터 구조를 채택하여 이를 효율적으로 관리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 변환 트리는 특히 교정(orthodontics) 분야에서 중요합니다. 기존의 라벨(dental model), X-ray 이미지, 얼굴 사진 등 다양한 형식의 데이터를 결합하여, 더욱 종합적으로 구조를 이해하고 진단할 수 있는 정보를 제공합니다. 여러 이미지를 환자 특유의 좌표계로 변환함으로써, 환자 간 통계 분석 및 시간 기반 비교가 가능해집니다.



### RGB-Event ISP: The Dataset and Benchmark (https://arxiv.org/abs/2501.19129)
Comments:
          Accepted by ICLR 2025; 14 pages, 8 figures, 4 tables

- **What's New**: 본 연구는 event-guided ISP(이미지 신호 처리)에 대한 첫 번째 연구로, pixel-level 정렬 이벤트와 RAW 이미지를 포함하는 새로운 event-RAW 페어링 데이터셋을 제시합니다. 이 데이터셋은 3373개의 RAW 이미지와 이에 해당하는 이벤트들을 포함하고 있으며, 다양한 조명 조건과 장면을 아우르는 24개의 장면을 포함합니다. 또한, 기존의 learnable ISP 방법들을 3가지 클래스로 분류하여 새 데이터셋에서 테스트하였습니다.

- **Technical Details**: 연구에서는 새로운 HVS-ALPIX-Eiger 센서를 사용하여 이벤트와 APS 이미지를 quad-Bayer 패턴으로 배열한 데이터셋을 수집했습니다. 생성된 RGB 이미지의 ground truth는 MATLAB 기반의 ISP 프레임워크를 통해 생성되며, ColorChecker를 기반으로 특정 기능(블랙 레벨 계산, 디모자이킹 등)을 수행합니다. 이 연구는 또한 event-guided ISP 신경망을 제안하여 RAW 이미지와 이벤트를 융합하고, 원래의 UNet에 비해 ISP 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 데이터셋과 방법론은 기존의 ISP 처리 성능이 낮은 상태에서 개선을 보여주면서, 컬러 정확성과 시간적 안정성을 분석하였습니다. 연구팀은 event-guided 접근 방식이 APS RAW의 노이즈 감소 및 색상 보정에 상당한 잠재력을 제공할 수 있음을 강조하며, 이러한 연구가 향후 RGB-Event ISP에 대한 방향성과 기초를 마련할 것으로 기대하고 있습니다. 이 연구 결과는 실제 응용 프로그램에서의 성능 향상에 기여할 수 있는 중요한 첫 단추가 될 것입니다.



### A Benchmark for Incremental Micro-expression Recognition (https://arxiv.org/abs/2501.19111)
- **What's New**: 이 논문에서는 기존 마이크로 표현(micro-expression) 인식을 위한 점진적 학습(incremental learning) 설정을 제안합니다. 특히, 연속적으로 변화하는 데이터 스트림을 처리하기 위한 벤치마크를 새롭게 수립하여, 새 데이터에 적응하면서도 기존 학습된 지식을 보존할 수 있는 방안을 모색합니다. 기존의 모든 훈련 데이터에 접근하는 것을 가정했던 전통적인 방법과는 대조적입니다.

- **Technical Details**: 저자들은 마이크로 표현 인식에 특화된 점진적 학습 설정을 공식화하고, 이를 위해 신중하게 기획된 학습 순서를 가진 연속 데이터 세트를 구성합니다. 이 논문에서는 두 가지의 시험 프로토콜을 정의하여 서로 다른 평가 목표를 겨냥합니다. 또한, 여섯 가지 기준 방법(baseline methods)과 그에 상응하는 평가 결과도 제공합니다.

- **Performance Highlights**: 제안된 벤치마크는 마이크로 표현 인식 연구를 진전시키기 위한 기초를 마련합니다. 이는 다양한 분야에서의 감정 인식 기술 향상에 기여할 것으로 기대됩니다. 모든 논문에서 사용된 코드는 공개될 예정으로, 연구 공동체의 접근성을 확보하고 후속 연구에 영향을 줄 것으로 보입니다.



### $\infty$-Video: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation (https://arxiv.org/abs/2501.19098)
Comments:
          17 pages, 7 figures

- **What's New**: 현재 비디오-언어 모델은 긴 비디오의 이해에 어려움을 겪고 있는 가운데, 본 논문은 단기 메모리(Short-Term Memory, STM)를 넘어 지속적인 시간(long-term memory, LTM) 통합 메커니즘을 통해 무한한 길이의 비디오 처리 가능성을 제시합니다. 이 새로운 프레임워크는 비디오 Q-formers를 보강하여 추가 학습 없이도 비디오 맥락을 효율적으로 처리할 수 있게 합니다. 동적 주의(attention) 메커니즘을 통해 가장 관련성 높은 비디오 세그먼트에 높은 세분화를 할당하여, 시간에 따라 진화하는 '점착성(sticky)' 기억을 형성합니다.

- **Technical Details**: 이 논문에서는 지속적인 시간 메모리 표현이 포함된 새로운 프레임워크를 제시합니다. 이를 통해 기존의 단기 맥락 다중 모달 대형 언어 모델의 능력을 확장하고, 비디오를 처리하는 데 있어 훈련 없는 방식(training-free)으로 기능하도록 설계되었습니다. 최근 인간의 작동 기억에 대한 이론과 유사하게, 메모리가 동적으로 처리되도록 하여 정보를 효과적으로 유지할 수 있도록 합니다. 주목할 점은 기존의 간단한 접근 방법을 버리고, 비디오 처리를 위한 강력한 연속적 주의 메커니즘을 도입했다는 것입니다.

- **Performance Highlights**: 비디오 질의 응답 작업에서의 실험을 통해 제안된 메커니즘은 Video-LLaMA와 VideoChat2 모델에서 성능을 향상시켰습니다. 이 모델은 비디오 프레임의 흐름을 단일 패스로 처리하며, STM은 한 번에 하나의 청크를 고려합니다. LTM은 이전 청크로부터의 global 정보를 저장하며, 모델의 성과를 향상시키는 역할을 합니다. 제안된 모델은 교육이 필요 없는 다른 모델들과 비교할 때 경쟁력 있는 결과를 나타내었습니다.



### Ambient Denoising Diffusion Generative Adversarial Networks for Establishing Stochastic Object Models from Noisy Image Data (https://arxiv.org/abs/2501.19094)
Comments:
          SPIE Medical Imaging 2025

- **What's New**: 이 논문에서는 덴오이징 확산 GAN (DDGAN)의 향상된 구조인 Ambient Denoising Diffusion GAN (ADDGAN)을 제안하고 있습니다. 이는 의료 이미징 시스템에서 발생하는 노이즈가 있는 이미지 데이터로부터 현실적인 확률적 객체 모델 (SOM)을 학습하는 데 집중합니다. 기존의 AmbientGAN모델보다 높은 해상도의 의료 이미지를 생성할 수 있는 능력을 입증했습니다.

- **Technical Details**: 논문에서 사용된 GAN은 두 개의 신경망으로 구성되어 있으며, 생성기(G)와 판별기(D)로 이루어져 있습니다. DDM은 가우시안 노이즈와 실세계 이미지를 연결하기 위해 서서히 디노이징 과정을 통해 진행됩니다. ADDGAN은 이러한 DDM의 속도를 개선하고 가상의 복잡한 다중 모드 분포를 포착하기 위해 설계되었습니다.

- **Performance Highlights**: 이 연구에서 제안한 ADDGAN은 노이즈가 있는 의료 이미지에서 SOM을 성공적으로 학습할 수 있음을 입증하였으며, 이는 CT 및 DBT 이미지와 같은 임상 이미지를 기반으로 합니다. ADDGAN은 고해상도 의료 이미지를 생성하는 데 효과적이며, 이전의 AmbientGAN 모델에 비해 상당한 성능 향상을 보였습니다.



### JGHand: Joint-Driven Animatable Hand Avater via 3D Gaussian Splatting (https://arxiv.org/abs/2501.19088)
- **What's New**: 본 논문에서는 실시간으로 고해상도의 손 이미지를 생성할 수 있는 새로운 Jointly 3D Gaussian Hand (JGHand) 모델을 제안합니다. JGHand는 3D 키 포인트를 기반으로 공간 변환을 위한 미분 가능 프로세스를 도입하여 손의 변형을 지원합니다. 또한, 손가락 움직임에 의한 자기 가림 그림자를 시뮬레이션하는 실시간 그림자 시뮬레이션 방법을 제안합니다. 이에 따라 JGHand는 다양한 포즈와 캐릭터에서 고충실도의 이미지를 생성하며, 기존의 최첨단 방법들을 초월하는 성능을 보입니다.

- **Technical Details**: JGHand 모델은 3D Gaussian Splatting (3DGS)을 기반으로 하며, 이를 통해 빠르고 효율적인 렌더링을 가능하게 합니다. 기존의 기술들과는 달리 본 모델은 뼈의 길이 및 임의의 포즈 변화에 대한 무오류 변환을 지원하며, 손의 해부학적 우선 지식을 포함하여 모델의 정확도를 강화합니다. 깊이 이미지를 활용한 그림자 시뮬레이션을 통해 손가락 자가 가림현상을 효과적으로 처리하며, 이를 통해 고품질 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과 JGHand는 공공 데이터셋에서 실시간 렌더링 속도와 함께 향상된 품질을 보여줘 기존의 최신 방법들을 능가했습니다. 각 구성 요소의 효과성은 포괄적인 아블레이션 연구를 통해 검증되었으며, 고해상도와 다양한 포즈에서의 손 렌더링이 가능함을 입증했습니다. 이를 통해 JGHand는 인체 컴퓨터 상호작용 및 3D 컴퓨터 비전 연구에서 중요한 발전을 가져올 것으로 기대됩니다.



### Fairness Analysis of CLIP-Based Foundation Models for X-Ray Image Classification (https://arxiv.org/abs/2501.19086)
Comments:
          This paper has been accepted for presentation at the 2025 IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이 연구는 X-ray 이미지 분류에 적용된 CLIP 유사 모델의 공정성을 체계적으로 평가했습니다. 특히, 성별과 연령을 포함한 다양한 인구 통계학적 배경을 고려하여 다층적 접근 방식을 통해 이러한 모델의 성능을 분석했습니다. 그 결과, 모델 튜닝이 정확도를 향상시켰으나, 여전히 성별과 연령에 따른 공정성 문제는 해결되지 않았습니다.

- **Technical Details**: 연구에서는 NIH Chest X-ray 데이터셋을 사용하여 6개 질병에 대한 공정성을 분석했습니다. 데이터셋은 성별과 연령 데이터에 대한 주석이 포함되어 있으며, 공정한 모델 평가를 위해 균형 잡힌 하위 집합으로 편집되었습니다. CLIP 모델을 포함한 네 가지 변형 모델을 사용하여 제로샷 추론 및 다양한 튜닝 기법으로 공정성을 평가했습니다.

- **Performance Highlights**: 각 모델의 성능은 정확도, 클래스별 F1 점수, 인구 통계 그룹 별 평균 F1 점수를 포함한 지표들을 통해 측정되었습니다. 연구에 따르면 모델 튜닝은 분류 정확도를 높였으나, 여전히 성별 및 연령과 관련된 공정성 간극이 존재하는 것으로 나타났으며, 이는 향후 개선 필요성을 시사합니다.



### Laser: Efficient Language-Guided Segmentation in Neural Radiance Fields (https://arxiv.org/abs/2501.19084)
Comments:
          Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence

- **What's New**: 이번 연구에서는 CLIP feature distillation을 활용한 3D 분할(segmentation) 방법을 제안합니다. 기존의 다중 스케일 CLIP 기능에 의존하지 않고, 밀집(dense) CLIP 피처의 간소화된 증류(dispensation)을 통해 언어 지침이 포함된 효율적인 3D 분할을 달성합니다. 연구에서는 어댑터 모듈을 도입하고, 자기가중 학습(self-cross-training) 전략을 통해 노이즈 문제를 완화하여 텍스트를 활용한 3D 장면의 정확한 분할이 가능하도록 합니다.

- **Technical Details**: 초기 연구에서 Neural Radiance Fields(NeRF)와 3D Gaussian Splatting의 발전이 이루어졌으나, 이들 모델은 주로 기하학적 정보와 외관 모델링에 치중하고 있어 의미론적 정보가 부족합니다. 이 연구는 CLIP을 활용하여 스타일과 구조를 효과적으로 결합할 수 있는 방법론을 제시합니다. 세부 기능 중에는 레이블 볼륨(label volume)의 사용으로 분할 문제를 분류(task classification) 문제로 변환하여 서로 다른 시점에서의 일관성 있는 결과를 보장하도록 하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 기존의 최신 기술들보다 훈련 속도와 성능 모두에서 우수한 결과를 보였습니다. 특히, 밀집 CLIP 기능으로 발생할 수 있는 노이즈 문제를 해결하는 방안을 통해 예측의 정확성을 높였고, 복잡성을 줄이며 다양한 색상 유사 구간에서의 분할 일관성을 유지할 수 있었습니다. 이러한 성과를 바탕으로 저자들은 해당 코드의 소스도 제공하고 있습니다.



### MotionPCM: Real-Time Motion Synthesis with Phased Consistency Mod (https://arxiv.org/abs/2501.19083)
- **What's New**: 이번 논문에서는 MotionPCM이라는 새로운 접근법을 소개하며, 이 모델은 딥러닝 기반의 Consistency Model(일관성 모델)을 활용하여 실시간 모션 합성을 개선하는 데 중점을 두고 있습니다. MotionPCM은 기존의 모션 합성 모델들과 비교해 샘플링 단계 수를 획기적으로 줄여주며, 완전한 모션 궤적을 N개의 세그먼트로 나누어 효율적인 샘플링을 가능하게 합니다. 이 방식은 예측의 일관성을 높이고, 고품질의 결과를 생성하는 데 기여합니다.

- **Technical Details**: MotionPCM은 기존의 MotionLCM 모델을 기반으로 하여, 2단계 결정론적 샘플링을 활용하는 Phased Consistency Model(PCM)을 적용하고 있습니다. PCM은 각각의 서브 궤적에 대한 일관성을 강화하여 샘플링의 질을 향상시키며, 딥러닝에서의 최적화 기술인 추가적인 판별자를 도입하여 낮은 단계 설정에서의 성능을 개선하였습니다. 이는 전통적인 샘플링 방법이 가지는 고비용을 줄여주어, 다양한 응용 분야에서의 실시간 활용을 가능하게 합니다.

- **Performance Highlights**: 제안된 MotionPCM은 공개된 대규모 데이터셋을 통해 실험되었으며, 속도와 생성 성능 측면에서 최고 성능을 기록하였습니다. 전반적으로 샘플링 단계가 네 단계 이하로 요구되며, 이는 기존 모션 합성 모델대비 현저한 개선을 보여줍니다. 이 연구는 게임 산업, 영화 제작 및 가상 현실 등 다양한 분야에서의 활용 가능성을 더욱 높이는 결과를 도출했습니다.



### Improving vision-language alignment with graph spiking hybrid Networks (https://arxiv.org/abs/2501.19069)
- **What's New**: 이 논문은 비전-언어(vision-language, VL) 간의 의미적 간극을 해소하기 위해 새로운 정밀 시각 의미 표현 모듈을 제안합니다. 이를 위해, panoptic segmentation을 활용하여 세밀한 시각적 기능을 생성하고, Spiking Neural Networks(SNN)와 Graph Attention Networks(GAT)의 장점을 통합한 Graph Spiking Hybrid Network(GSHN)를 개발했습니다. 이 모델은 개별 인스턴스의 은닉 변수와 그에 따른 지역 및 글로벌 맥락적 특징을 포괄적으로 캡처하여, 의미 표현의 풍부함과 다양성을 크게 향상시킵니다.

- **Technical Details**: GSHN은 두 가지 구성 요소를 결합하여 시각적 정보를 인코딩합니다. SNN의 희소한 처리 능력을 활용하여 효과적으로 특징을 집계하며, GAT를 사용하여 심층적인 시각 의미 정보를 포착합니다. 이 과정에서 대조 학습(contrastive learning, CL)을 적용하여 유사한 의미를 가진 노드 간의 회상을 돕고, 계산 효율성을 높이기 위한 스파이크 텍스트 학습(Spiked Text Learning, STL) 방법이 도입되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 GSHN은 다양한 VL 하위 작업에서 유망한 결과를 보여주었습니다. VQA, VE, 이미지-텍스트 검색과 같은 다수의 public 데이터셋에서 효과성을 입증하였으며, ablation 실험을 통해 모델 설계의 타당성과 효과성을 추가로 검증했습니다. 이러한 성과는 VL 의미 표현의 정교한 처리와 관련된 여러 과제가 협력적으로 해결됨을 보여줍니다.



### Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations (https://arxiv.org/abs/2501.19066)
Comments:
          15 pages, 16 figures

- **What's New**: 이 논문은 이미지 생성을 위한 텍스트-투-이미지(T2I) 모델에서 안전하고 윤리적인 콘텐츠 생성을 위한 새로운 프레임워크를 제안합니다. 기존 방식들은 모델을 재훈련하여 특정 개념을 제거하는 데 초점을 두었으나, 이는 계산 비용이 높고, 확장성이 부족하며 생성 품질을 저해하는 단점이 있었습니다. 이 연구는 k-sparse autoencoder(k-SAE)를 활용하여 개념 조작을 효율적이고 해석 가능하게 수행하는 방법을 소개합니다.

- **Technical Details**: k-sparse autoencoder(k-SAE)는 텍스트 임베딩에서 해석 가능한 단일 의미 개념(monosemantic concepts)을 식별하고, 이를 활용하여 특정 개념(예: 노출, 폭력)으로의 생성 또는 새로운 개념(예: 사진 스타일)의 도입을 정확하게 조작할 수 있습니다. 본 연구에서는 생성 과정 중 개념을 조작해도 전체 생성 능력에는 영향을 주지 않으며, 추가 재훈련이 필요하지 않다는 점이 중요합니다. 이 방법을 통해 기존 기술 대비 생성 속도를 약 5배 빠르게 하고, 위험한 개념 제거에서 20.01%의 개선을 보였습니다.

- **Performance Highlights**: 이 연구에서 제안한 Concept Steerer는 위험한 개념 제거에서 최첨단 성능을 달성하였으며, 스타일 조작이 가능한 동시에 창의적인 이미지 생성을 지원합니다. 또한, 이는 공격적인 프롬프트 조작에 대해 강력한 내성을 가지며, 실질적인 비주얼 품질 저하 없이 다양한 개념을 조작할 수 있는 방법입니다. 마침내, 모든 텍스트-투-이미지 모델에 즉시 적용 가능하고, 높은 효율성을 자랑합니다.



### EgoMe: Follow Me via Egocentric View in Real World (https://arxiv.org/abs/2501.19061)
- **What's New**: 이 논문에서는 새로운 대규모 egocentric 데이터셋인 EgoMe를 소개합니다. 이는 인간의 모방 학습 과정을 egocentric 관점에서 깊이 포착하고 분석하는 것을 목표로 합니다. EgoMe 데이터셋은 7902 쌍의 비디오(총 15804개)를 포함하며, 다양한 실제 시나리오에서 사람들의 행동을 기록합니다.

- **Technical Details**: EgoMe 데이터셋은 imitator의 egocentric 관점에서 촬영된 비디오 쌍을 포함하며, 한 비디오는 demonstrator의 행동을 보여주고, 다른 비디오는 imitator가 그 행동을 모방하는 과정을 포착합니다. 이 데이터셋은 단순한 일상 동작부터 복잡한 전문 기술 작업까지 184개의 활동 범주를 포함하고 있으며, 센서 데이터를 통해 관찰 및 모방 간의 상관관계를 구축하는 데 필요한 정보를 제공합니다.

- **Performance Highlights**: EgoMe 데이터셋은 기존 데이터셋에 비해 큰 장점을 가지며, 8개의 도전적인 벤치마크 작업이 설계되었습니다. 이러한 작업은 exo-ego 비디오 생성, exo-ego 시선 예측 및 동작 평가 등의 다양한 AI 모델 학습 능력을 향상시킬 수 있습니다. 연구팀은 이 데이터셋이 인간 학습 과정을 탐구하고 다음 세대 로봇 개발에 새로운 통찰을 제공하기를 희망합니다.



### Contrast-Aware Calibration for Fine-Tuned CLIP: Leveraging Image-Text Alignmen (https://arxiv.org/abs/2501.19060)
Comments:
          arXiv admin note: text overlap with arXiv:2402.04655 by other authors

- **What's New**: 본 연구에서는 Contrast-Aware Calibration (CAC)이라는 새로운 멀티모달 캘리브레이션 방법을 제안합니다. 이는 비훈련 클래스에 대한 캘리브레이션 문제를 해결하기 위해 기존 VLM 캘리브레이션 방법의 한계를 극복하는 데 중점을 둡니다. CAC는 원본 CLIP과 파인튜닝된 CLIP 간의 대조적인 차이를 활용하여, 훈련된 클래스에 대한 캘리브레이션 능력을 향상시킵니다.

- **Technical Details**: CAC는 원본 CLIP이 훈련된 클래스와 비훈련 클래스에서의 이미지-텍스트 간의 관계를 재조정할 수 있도록 돕습니다. 이 방법은 logits를 재조정하기 위해서 잘 정렬된 정보를 활용하며, 훈련 클래스와 비훈련 클래스 모두에 대해 재캘리브레이션을 수행합니다. 이를 통해 이전의 방법들이 해결하지 못했던 개별 및 집합 클래스의 재조정을 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: 11개의 데이터셋과 5개의 파인튜닝 방법을 기반으로 한 실험에서 CAC는 훈련 클래스와 비훈련 클래스 모두에서 최고의 캘리브레이션 효과를 나타냈습니다. 특히 기존의 MIR 및 DAC와 같은 최신 방법들을 초월하여, CAC는 정확도와 추론 속도를 감소시키지 않으면서 신뢰할 수 있는 예측 결과를 제공합니다. 또한 CAC는 다양한 파인튜닝 방법의 캘리브레이션 성능을 일관되게 향상시키는 것으로 나타났습니다.



### Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models (https://arxiv.org/abs/2501.19054)
- **What's New**: CADFusion은 Large Language Models (LLMs)을 기반으로 하여 Text-to-CAD 모델의 훈련에 필요한 순차 신호(sequential signal)와 시각 신호(visual signal)를 결합한 새로운 프레임워크입니다. 이 프레임워크는 두 가지 단계인 순차 학습(sequential learning) 단계와 시각 피드백(visual feedback) 단계로 구성되어 있으며, 이 두 단계를 번갈아 가며 훈련함으로써 균형 잡힌 학습을 보장합니다. CAD 모델의 다중 모드 특성(multimodal characteristic)과 다대일 렌더링 특성(many-to-one rendering characteristic)을 고려하여 새로운 접근 방식을 제공합니다.

- **Technical Details**: CADFusion은 사전 훈련된 LLM을 이용하여 ground-truth parametric sequences로 미세 조정(fine-tuning)된 후, 시각적으로 선호되는 객체를 렌더링할 수 있도록 하여 성능을 향상시킵니다. 특히, 개인의 선호를 학습하도록 직접적인 선호 최적화(direct preference optimization, DPO)를 통해 비가분화(non-differentiable) 렌더링 경로를 우회하고, 대규모 비전-언어 모델(large vision-language models, LVMs)을 사용하여 렌더링된 시각 객체에 대한 점수를 효율적으로 수집합니다. 이러한 해결책은 모델 훈련의 효과성을 높이며, 보다 정교한 CAD 모델 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, CADFusion은 질적 및 양적으로 우수한 성과를 나타내며, CAD 모델 생성의 효율성을 크게 향상시킵니다. 특히, 이전의 연구들과 비교했을 때, 세밀한 향상된 성능을 보여주며, Text-to-CAD 분야에서의 혁신적인 발전을 예고합니다. CADFusion이 가져올 수 있는 다양하고 실질적인 응용 가능성은 CAD 설계 분야에서 중요한 기여를 할 것으로 기대됩니다.



### Self-Supervised Cross-Modal Text-Image Time Series Retrieval in Remote Sensing (https://arxiv.org/abs/2501.19043)
- **What's New**: 본 논문에서는 최초로 원거리 센싱(remote sensing) 분야에서 교차 모달 텍스트 이미지 시계열 검색(cross-modal text-ITSR) 과제를 제안합니다. 이 연구는 텍스트 문장을 쿼리로 사용하여 이미지 시계열을 검색할 수 있는 방법론을 소개하며, 이는 기존의 단일 모달 정보 활용 제한을 극복합니다. 제안된 방법은 쌍의 이미지(즉, 이중 시계열 이미지)에 초점을 맞추고 있으며, 텍스트와 이미지를 동시에 고려하여 더 유연한 검색 기능을 제공합니다.

- **Technical Details**: 제안된 텍스트 이미지 시계열 검색 방법은 두 가지 주요 구성 요소로 구성됩니다: 1) 모달리티별 인코더(modality-specific encoders)로 이중 시계열 이미지와 텍스트 문장의 의미적 내용을 모델링하고, 2) 모달리티별 프로젝션 헤드(modality-specific projection heads)로 텍스트와 이미지 표현을 공유 임베딩 공간에서 정렬합니다. 이를 통해 글로벌 특성 융합(GFF)과 트랜스포머 기반 특성 융합(TFF) 전략 등 두 가지 융합 전략을 도입하여 이중 시계열 이미지 내의 시간 정보를 효과적으로 모델링합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 원거리 센싱 벤치마크 아카이브에서 수행된 광범위한 실험을 통해 검증되었습니다. 실험 결과, 텍스트 쿼리나 이중 시계열 이미지 쿼리를 기반으로 의미적으로 연관된 이미지를 정확하게 검색하는데 있어 높은 성능을 나타냈습니다. 이 연구는 자연어 쿼리를 사용하여 대규모 원거리 센싱 아카이브에서 검출할 수 있는 가능성을 크게 확장합니다.



### Beyond Token Compression: A Training-Free Reduction Framework for Efficient Visual Processing in MLLMs (https://arxiv.org/abs/2501.19036)
- **What's New**: 이번 논문에서는 Decoder-Only MLLM(RNN 기반의 모델)에서 시각 토큰에 대한 자체 주의력(self-attention) 및 피드포워드 네트워크(FFN) 작업의 비용을 줄이는 새로운 분석 프레임워크를 제안합니다. Hollow Attention과 Probe-Activated Dynamic FFN이라는 두 가지 혁신적인 기법을 도입하여, 성능을 유지하면서도 비용을 상당 부분 줄일 수 있는 방법을 모색하였습니다. 이를 통해 계산 자원의 효율성을 높이고, 또한 향후 MLLM의 설계에 대한 통찰력을 제공하는 결과를 도출하였습니다.

- **Technical Details**: Hollow Attention은 시각 토큰 간의 전역적(global) 상호작용을 지역적(local) 상호작용으로 제한하여 시각-텍스트 연관성을 유지합니다. Probe-Activated Dynamic FFN은 시각 토큰을 처리하기 위해 FFN 매개변수의 일부만 선택적으로 활성화하도록 설계되었습니다. 또한, 저희 연구에서는 레이어별 감소 전략을 통하여 전체 레이어 중 일부에 대해서만 분석 및 최적화를 수행하는 방법론을 세 명확히 하여, 학습 없이도 계산 효율성을 증대시키는 방법을 모색하였습니다.

- **Performance Highlights**: 제안된 방법들은 상태-of-the-art MLLM 실험에서 대체적으로 모델 성능을 유지하면서 경우에 따라 성능을 개선하는 결과를 보여주었습니다. 특히 레이어의 약 절반에 다시 줄인 후에도 성능 저하 없이 처리 가능하다는 통찰을 제공하여, 현재의 MLLM 구조에서 상당한 계산적 중복성이 있음을 나타냅니다. 마지막으로, 기존 토큰 압축 기술과의 병합을 통해 추가적인 계산 능력 공제를 가능케 하여, 컴퓨팅 효율성의 향상을 이룰 수 있음을 보여줍니다.



### SynthmanticLiDAR: A Synthetic Dataset for Semantic Segmentation on LiDAR Imaging (https://arxiv.org/abs/2501.19035)
Comments:
          2024 IEEE International Conference on Image Processing (ICIP)

- **What's New**: 이번 연구는 LiDAR 이미징에서의 의미론적 세분화(semantic segmentation)에 중점을 둔 수정된 CARLA 시뮬레이터를 소개합니다. 이 시뮬레이터는 새로운 클래스와 더 일관된 객체 라벨링을 통해 현실 데이터셋인 SemanticKITTI와의 유사성을 제공합니다. 이 연구는 다양한 의미론적 세분화 알고리즘의 학습 과정에 대한 새로운 합성 데이터셋인 SynthmanticLiDAR의 기여를 평가합니다.

- **Technical Details**: 이 연구에서 제시된 SynthmanticLiDAR 데이터셋은 LiDAR 기반의 의미론적 세분화를 위해 설계되었으며, 주어진 데이터에서 각 점에 대한 의미론적 라벨을 할당하는 목표로 합니다. LiDAR 포인트 클라우드를 정확하게 세분화함으로써 자율 차량은 주변을 더 잘 인식하고 더 나은 주행 결정을 내릴 수 있습니다. 현재의 연구는 현실 데이터 수집의 비용과 시간을 대체할 수 있는 합성 데이터 생성의 중요성을 강조합니다.

- **Performance Highlights**: SynthmanticLiDAR 데이터셋을 사용한 결과, 여러 알고리즘의 성능이 개선되었습니다. 이 연구는 자율주행 시스템에서 LiDAR의 의미론적 세분화가 얼마나 중요한지를 나타내며, 합성 데이터셋의 유용성을 입증합니다. 이와 함께, 연구팀의 수정된 CARLA 시뮬레이터는 실제 데이터셋과 더 일치하는 LiDAR 스캔 생성을 가능하게 합니다.



### XRF V2: A Dataset for Action Summarization with Wi-Fi Signals, and IMUs in Phones, Watches, Earbuds, and Glasses (https://arxiv.org/abs/2501.19034)
Comments:
          27 pages, 11 figures, 8 tables

- **What's New**: 이 논문은 실내 활동 인식(HAR)과 관련하여 새로운 XRF V2 데이터셋을 도입합니다. XRF V2는 Wi-Fi 신호, IMU 센서(스마트폰, 스마트워치 등) 및 동기화된 비디오 기록을 통합하여 16명의 자원봉사가 수행한 다양한 실내 활동을 수집하였습니다. 이를 통해 행동 요약(action summarization)이라는 새로운 과제를 다루고 있으며, XRFMamba 신경망을 제안하여 최신 방법론과 비교해 우수한 성능을 보입니다.

- **Technical Details**: 제안하는 XRFMamba 신경망은 장기적 의존성(long-term dependencies)을 캡처하는 데 뛰어난 성능을 발휘합니다. 데이터셋 수집은 세 가지 유형의 실내 환경에서 진행되었으며, 총 16시간 16분의 멀티모달 데이터가 포함되어 있습니다. 또한, 새로운 평가 메트릭인 Response Meaning Consistency (RMC)를 도입하여 행동 요약의 성과를 측정합니다.

- **Performance Highlights**: XRFMamba는 최신 WiFiTAD보다 mAP에서 5.49점 더 높은 성과를 달성하였고, 파라미터 사용량은 35% 적습니다. 이는 캠퍼스에서 수집한 기존 데이터셋에 비해 XRF V2가 더욱 다양한 및 현실적인 테스트 환경을 제공함을 보여줍니다. 또한, 이 연구는 스마트 홈 시스템 내에서 지능형 에이전트를 적용할 수 있는 가능성을 제시하며, 개인화된 건강 모니터링 및 클라이밋 제어와 같은 다양한 서비스를 통한 응용 가능성을 강조합니다.



### Virtual airways heatmaps to optimize point of entry location in lung biopsy planning systems (https://arxiv.org/abs/2501.19003)
- **What's New**: 이번 연구에서는 폐 생검 계획 시스템에서의 POE(point of entry) 최적화를 위한 가상 모델을 제안합니다. 이 모델은 계획 시뮬레이션에서의 방향과 실제 수술 동안의 방향 간의 차이로 인한 오차 범위를 고려하여 생검 샘플의 품질을 평가할 수 있습니다. 추가적으로, 병변의 특성이 미치는 영향을 조사하는 것이 주요 목표입니다.

- **Technical Details**: 생검 품질은 환자 맞춤형 기도의 골격에 투영된 히트맵(heatmap)으로 제공됩니다. 이 골격은 기도의 구조를 3D로 표현하며, 히트맵의 강도는 각 POE에서 추출될 수 있는 조직의 잠재적 양을 나타냅니다. 생검 도구의 도입에 대한 불확실성 영역을 나타내는 원뿔과 병변 간의 교차 점을 통해 결정됩니다.

- **Performance Highlights**: 시뮬레이션된 다양한 개입 장면은 CT 스캔에서 추출된 단일 해부학과 정형 및 비정형 형태의 두 병변을 대상으로 진행되었습니다. 분석 결과, 비정형 형태에서의 병변 방향과 두 형태 모두에서의 거리의 강한 영향을 시사합니다. 제안된 히트맵은 최적의 POE를 시각적으로 평가하고, 서로 다른 기관지의 여러 최적 POE의 존재 여부를 파악하는 데 기여합니다.



### VKFPos: A Learning-Based Monocular Positioning with Variational Bayesian Extended Kalman Filter Integration (https://arxiv.org/abs/2501.18994)
- **What's New**: 이 논문은 VKFPos라는 새로운 접근 방식을 제안하여, Absolute Pose Regression (APR)과 Relative Pose Regression (RPR)을 Variational Bayesian inference 프레임워크 내에서 Extended Kalman Filter (EKF)와 통합합니다. 이 방법은 모노큘러 포지셔닝 문제의 본질적인 후행 확률을 APR과 RPR 구성 요소로 분해할 수 있음을 보여줍니다. 이러한 분해는 딥러닝 모델에 통합되어 불확실성을 더 잘 관리할 수 있도록 하며, 이를 통해 성능이 향상됩니다.

- **Technical Details**: VKFPos는 APR과 RPR을 결합하여 위치 예측 정확도를 개선하는 경량의 모노큘러 포지셔닝 접근 방식으로, EKF의 재귀 구조를 활용하여 과거 상태 정보를 현재 상태 예측 및 궤적 예측에 활용합니다. 특이한 점은 Variational Bayesian inference에 근거한 절대 및 상대 포즈 추정기들을 위한 혁신적인 훈련 패러다임을 제공한다는 것입니다. 이 방식은 APR과 RPR 지점을 각각 예측하여 EKF에서 궤적을 최적화하는 구조를 가지고 있습니다.

- **Performance Highlights**: 실험 결과, VKFPos의 단일 샷 APR 지점은 최신 기술에 필적하는 정확도를 달성했으며, 연속 이미지로 RPR과 EKF 통합이 가능한 경우에는 VKFPos가 시간적 APR 및 모델 기반 통합 방법을 초월하는 성능을 보여주었습니다. 이로 인해 VKFPos는 모노큘러 포지셔닝 분야에서 강력한 대안으로 자리 잡을 것으로 기대됩니다.



### Visual Autoregressive Modeling for Image Super-Resolution (https://arxiv.org/abs/2501.18993)
Comments:
          20 pages; 17 figures

- **What's New**: 이번 논문에서는 Image Super-Resolution (ISR) 분야에서 새로운 접근법인 VARSR(Visual Autoregressive modeling for ISR)을 제안합니다. 이 모델은 다음 규모 예측(next-scale prediction)의 형태로 구성되어 있으며, 저해상도 이미지에서 의미 정보를 통합하고 보존하기 위해 prefix tokens을 사용합니다. 또한, VARSR은 공간 구조를 포착하기 위해 scale-aligned Rotary Positional Encodings을 도입하고, 픽셀 수준의 충실도를 제공하는 diffusive refiner를 활용합니다.

- **Technical Details**: VARSR 모델은 저해상도 이미지에서의 행동을 개선하기 위해 이미지 기반의 Classifier-free Guidance를 제안하여 보다 실감 나는 이미지를 생성하도록 유도합니다. 대규모 데이터를 수집하고 강력한 생성 prior를 학습하기 위한 훈련 과정을 설계하여 모델의 효율성을 극대화했습니다. 이러한 다양한 기술적 요소들은 VARSR의 성능을 높이고, 전통적인 방법들에 비해 매끄러움을 더합니다.

- **Performance Highlights**: 정량적 및 정성적인 결과를 통해 VARSR은 기존의 diffusion 기반 방법들보다 높은 충실도와 실현 가능성을 가지며, 더 효율적으로 고품질 이미지를 생성할 수 있음을 보여줍니다. 이러한 성과는 새로운 빅데이터 학습 기법과 강화된 모델의 조합으로 가능해졌으며, 앞으로의 연구 및 응용에 큰 기여를 할 것으로 기대됩니다.



### Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images (https://arxiv.org/abs/2501.18984)
Comments:
          22 pages, 6 figures, 3 tables

- **What's New**: 이 논문에서는 Query-aware long contextual dynamic modeling framework인 Querent를 제안합니다. 이 방법은 전체 self-attention의 표현력을 유지하면서도 효율성을 달성합니다. 각 패치에 대해 가장 관련이 높은 주변 영역을 적응적으로 예측하여 특정 문맥에 집중한 attention 계산을 가능하게 합니다.

- **Technical Details**: Querent는 gigapixel 이미지를 위한 동적인 장거리 문맥 모델링을 통해 패치 관계를 적응적으로 결정합니다. 이 조정 메커니즘은 각 패치에서 고유한 attention 패턴을 활용하여 조직학적 특성의 이질성을 포착하는 데 도움이 됩니다. 이 방법은 또한 지역 메타데이터 계산과 중요도 추정을 통해 attention 패턴의 동적 희소화를 가능하게 합니다.

- **Performance Highlights**: Querent는 10개 이상의 WSI 데이터 세트에서 바이오마커 예측, 유전자 변이 예측, 암 서브타이핑 및 생존 분석 등 다양한 CPath 작업에서 우수한 성능을 보여줍니다. 이 방법은 기존의 최첨단 모델보다 일관되게 더 나은 결과를 기록했습니다.



### OmniPhysGS: 3D Constitutive Gaussians for General Physics-Based Dynamics Generation (https://arxiv.org/abs/2501.18982)
Comments:
          Accepted to ICLR 2025; Project page: this https URL

- **What's New**: 최근 3D 자산 재구성 및 생성에서 큰 발전이 이루어졌으며, 특히 물리적 상호작용을 포함한 동적 장면 생성이 주목받고 있습니다. 기존의 방법들은 모든 재료가 특정한 미리 정의된 카테고리(예: elasticity)에 속한다고 가정하여 물리적 특성을 회복하려고 했습니다. 그러나 이러한 가정은 실제 상황에서 여러 이질적인 객체의 복잡한 조합을 무시하고, 보다 다양해진 객체에 대한 물리적으로 신뢰할 수 없는 애니메이션을 초래하는 경향이 있습니다. 이를 해결하기 위해 우리는 OmniPhysGS라는 새로운 모델을 제안하며, 이는 보다 일반적인 객체로 구성된 물리 기반 3D 동적 장면을 생성합니다.

- **Technical Details**: OmniPhysGS의 주요 설계는 각 3D 자산을 구성 3D Gaussian으로 처리한다는 것입니다. 각 Gaussian은 12개의 물리적 도메인 전문 서브 모델(고무, 금속, 꿀, 물 등)의 앙상블로 나타나며, 이는 모델의 유연성을 크게 향상시킵니다. 구현 과정에서는 사용자 지정 프롬프트에 따라 장면을 정의하고 사전 훈련된 비디오 확산 모델을 통해 물질 가중치 추정을 감독합니다. 이러한 방식으로 우리는 탄생한 동작이 다양한 재료를 포함하고 서로 다른 기계적 조건에 대한 물질 반응을 바탕으로 하도록 합니다.

- **Performance Highlights**: OmniPhysGS는 고무성과 점탄성, 플라스틱 및 유체 물질을 포함한 보다 폭넓은 재료의 일반적이고 현실적인 물리적 동적 특성을 생성하는 성과를 보여줍니다. 다양한 매개변수에서 기존 방법들보다 약 3%에서 16% 향상된 시각적 품질과 텍스트 맞춤도의 지표를 달성했습니다. 또한, 우리의 방법은 수동으로 물리적 속성을 조정할 필요 없이 다양한 물질의 물리적으로 신뢰할 수 있는 동작과 상호작용을 생성할 수 있습니다.



### LLMDet: Learning Strong Open-Vocabulary Object Detectors under the Supervision of Large Language Models (https://arxiv.org/abs/2501.18954)
- **What's New**: 이 논문은 open-vocabulary object detection의 성능을 개선하기 위해, 대형 언어 모델(LLM)과의 공동 학습(co-training)을 제안합니다. 이를 통해 이미지별 상세 캡션(image-level detailed captions)을 생성하고, 기존의 객체 감지 방식을 넘어서 더 정밀한 정보 제공을 목표로 합니다. 특히, 새로운 데이터셋 GroundingCap-1M을 수집하여, 이미지와 관련된 grounding label과 이미지 수준의 설명을 포함하여 성능을 더욱 강화합니다.

- **Technical Details**: LLMDet라는 새로운 감지기를 도입하여, 표준 grounding objective와 캡션 생성(caption generation) 목표를 결합하여 학습합니다. LLM은 이미지 및 지역 피처를 입력받아 이미지별 긴 상세 캡션과 지역별 짧은 문구를 생성하며, 이를 통해 다양한 지역의 정보와 전체 이미지를 하나로 통합하여 이해합니다. 이 접근법은 높은 품질의 이미지 수준 캡션 생성을 가능하게 하여, rare class에 대한 성능 향상 및 일반화 능력을 강화합니다.

- **Performance Highlights**: 실험 결과, LLMDet는 LVIS minival 데이터셋에서 기존 baseline보다 각각 3.3%/3.8%/14.3% AP, 3.1%/3.3%/17.0% APr 향상된 성능을 보였습니다. 또한, LLMDet는 여러 데이터셋에서 제로샷 전이(zero-shot transfer)에 대한 우수한 성능을 입증하였으며, 이로써 대형 다중 모달 모델(LMM)을 구축하는 데에도 유용하다는 것을 보여주었습니다.



### TV-Dialogue: Crafting Theme-Aware Video Dialogues with Immersive Interaction (https://arxiv.org/abs/2501.18940)
- **What's New**: 최근 대규모 언어 모델(LLM)의 발전은 텍스트 및 이미지 기반 대화 생성의 개발을 가속화했지만, 비디오 기반 대화 생성은 여전히 탐구가 부족하고 독특한 도전에 직면해 있습니다. 이 논문에서는 비디오 콘텐츠와 사용자 지정 테마에 맞는 새로운 대화를 생성할 목표를 가진 테마 인식 비디오 대화 제작(Theme-aware Video Dialogue Crafting, TVDC)이라는 새로운 작업을 소개합니다. 이를 위해 TV-Dialogue라는 다중 모달 에이전트 프레임워크를 제안하며, 이는 실시간 몰입형 상호작용을 통해 비디오 캐릭터 간의 상호작용을 지원합니다.

- **Technical Details**: TV-Dialogue는 각 캐릭터가 자율적으로 행동하며 독립적으로 사고할 수 있도록 설계되었습니다. 이 방법은 사용자 지정 테마와 원래 비디오에 따라 새로운 줄거리를 생성하고, 비디오의 다양한 역할에 새로운 하위 에이전트를 할당합니다. 시각-언어 모델(Visual-Language Model, VLM)의 지원을 통해 각 캐릭터는 자신의 시각적 행동과 감정 변화를 인식하며, 이를 바탕으로 새로운 대화를 생성합니다.

- **Performance Highlights**: 자체 수집한 다중 테마 비디오 대화(Multi-Theme Video Dialogue, MVD) 데이터셋을 통해 TV-Dialogue의 우수성을 검증했습니다. 실험 결과, TV-Dialogue는 상업적 GPT 모델과 다중 모달 대형 언어 모델에 비해 대화 생성에 있어 월등한 성능을 보였고, 비디오-텍스트 검색 작업에서도 기반 방법 대비 6% 이상의 향상을 이루었습니다. 이 결과는 TV-Dialogue의 다양한 응용 가능성을 시사합니다.



### Training-free Quantum-Inspired Image Edge Extraction Method (https://arxiv.org/abs/2501.18929)
Comments:
          12 pages, 6 figure,

- **What's New**: 이번 연구에서는 전통적인 딥러닝 기반 엣지 감지 방법의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 우리는 훈련이 필요 없는 양자 영감을 받은 모델을 도입하여 복잡한 환경에서도 효과적으로 작동할 수 있는 엣지 감지 방법을 개발했습니다. 이 모델은 고전적인 Sobel 엣지 감지 기법과 슈뢰딩거(Schrödinger) 파동 방정식을 결합하여 더욱 정교한 엣지 감지를 가능하게 합니다.

- **Technical Details**: 제안하는 모델은 훈련 없이 작동 가능하여 다양한 응용 분야에 유연하게 적용될 수 있습니다. 슈뢰딩거 파동 방정식을 이용한 반복적 확산(iterative diffusion) 과정은 경량화된 구조에서도 엣지의 정밀도를 크게 향상시킵니다. 또한, Canny 및 Laplacian 연산자를 결합한 하이브리드 프레임워크는 지역(local) 및 글로벌(global) 특징을 통합하여 어려운 상황에서도 견고성을 보장합니다.

- **Performance Highlights**: BIPED, Multicue, NYUD 데이터세트를 통한 광범위한 평가 결과, 제안한 모델은 ODS, OIS, AP, F-measure 등의 최신 성과(metrics)에서 우수한 성능을 달성하였습니다. 노이즈 내성과 관련된 실험 결과는 이 모델이 실제 환경에서도 신뢰성이 높다는 것을 보여주었습니다. 의료 이미징, 자율 시스템, 환경 모니터링 등 다양한 실제 응용에 적합한 모델로 자리잡으며 엣지 감지의 새로운 기준을 제시합니다.



### Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior (https://arxiv.org/abs/2501.18913)
Comments:
          ICLR 2025

- **What's New**: 이번 연구에서는 Diffusion Posterior Sampling (DPS)이 조건부 스코어 추정(arguably conditional score estimation)보다 후방 확률 분포 최대화(maximizing posterior, MAP)와 더 밀접한 연관이 있음을 입증합니다. 특히, DPS의 조건부 스코어 추정이 훈련된 모델의 스코어와 유의미하게 차이가 나고, 스코어의 평균이 유효한 추정이 아님을 발견했습니다. 이러한 결과를 바탕으로 DPS의 성능 개선 방안도 제시하며, 실험적 결과에서도 긍정적인 변화를 보여줍니다.

- **Technical Details**: 이번 논문에서는 일반적으로 정립된 조건부 스코어 추정론을 재조명하고, DPS가 조건부 스코어 추정보다 MAP에 더 가까운 방식으로 작동함을 보여줍니다. DPS는 고해상도 이미지를 이용한 실험에서 정확한 스코어 추정으로 나아가기 위해 다단계 경량화 gradient ascent 기법과 조건부 스코어 추정 방법을 통해 개선되었습니다. 특히, 100개의 이미지를 사용하여 훈련한 경량 조건부 스코어 추정기가 포함되어 있습니다.

- **Performance Highlights**: 제시된 개선 사항을 통해 DPS의 샘플 생성 품질이 유의미하게 향상되었지만, 다양성은 여전히 낮은 수준으로 남아 있습니다. 실험 결과는 DPS가 더 효과적으로 후방 확률을 최대화하게 되었다는 것을 입증하며, 후속 연구를 위한 기반을 마련합니다. 또한 프로토타입의 성능을 검증하기 위해 모델 평가 지표를 제시하였습니다.



### GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling (https://arxiv.org/abs/2501.18898)
- **What's New**: 새롭게 제안된 GestureLSM은 음성 신호에 기반하여 인간의 제스처를 생성하는 새로운 접근 방식을 제시합니다. 이 모델은 신체 각 부위 간의 상호작용을 공간적 및 시간적 주의를 통해 명시적으로 모델링합니다. 이를 통해 더욱 자연스러운 제스처 생성을 이루어냄으로써, 디지털 아바타 개발 및 몰입형 상호작용 경험 구축에 있어 중요한 발전을 가져옵니다.

- **Technical Details**: GestureLSM은 인체를 네 가지 구성 요소(상체, 손, 하체 및 얼굴 표정)로 나눠 각 부위의 모션 표현을 독립적으로 정의합니다. 잔여 벡터 양자화(residual vector quantization)를 통해 각 부위의 모션을 디지털 토큰으로 변환하고, 공간적 및 시간적 주의를 활용하여 신체 부위 간의 상호작용을 개선합니다. 이를 통해 우리는 실제 시간(real-time) 제스처 생성을 달성하며, 신속한 생성 속도를 위해 레크티파이드 디퓨전(rectified diffusion) 방식을 채택했습니다.

- **Performance Highlights**: 다양한 정량적 및 정성적 실험을 통해 GestureLSM이 기존의 방법들보다 탁월한 생성 품질을 달성하는 것을 입증했습니다. 또한, 이 모델은 즉각적인 응답을 요구하는 애플리케이션에 적합한 신속한 추론 속도를 자랑합니다. GestureLSM은 디지털 휴먼과 구체화된 에이전트의 개발에 있어 다채로운 응용 가능성을 보여줍니다.



### RLS3: RL-Based Synthetic Sample Selection to Enhance Spatial Reasoning in Vision-Language Models for Indoor Autonomous Perception (https://arxiv.org/abs/2501.18880)
Comments:
          ICCPS 2025 accepted paper, 10 pages, 9 figures

- **What's New**: 이 논문에서는 비전-언어 모델(Vision-Language Model, VLM)의 파인튜닝을 개선하기 위해 강화 학습(reinforcement learning, RL) 에이전트를 통합한 새로운 일반화 가능 프레임워크를 제안합니다. 이 방법은 RL 에이전트가 실내 환경에서 객체를 조Manipulate하여 VLM의 취약점을 보완하기 위한 합성 데이터(synthetic data)를 생성하는 데 중점을 둡니다. 이를 통해 특정 작업(예: 공간 추론)에서 VLM의 성능을 효과적으로 개선할 수 있습니다.

- **Technical Details**: 논문의 주요 기술적 기여는 RL 에이전트를 데이터 샘플링 도구로 활용하여 VLM에 필요한 특정 합성 데이터를 생성한 점입니다. RL 에이전트는 VLM의 성능에 대한 피드백을 통해 유익한 데이터를 생성하며, 특히 공간 추론에서 VLM의 성능 향상에 기여합니다. 이 프레임워크는 RL을 통해 지속적인 피드백을 제공하여 VLM이 복잡한 시각적-언어적 관계를 잘 처리할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 이 방법이 VLM의 복잡한 공간 관계 처리 능력을 향상시키는 데 효과적임을 보여줍니다. RL 가이드 데이터 생성 방식은 VLM의 성능을 크게 개선시킬 수 있음을 입증하며, 이러한 접근은 모델 파인튜닝과 데이터 증강에서 강력한 도구로 자리매김할 잠재력을 가집니다. 결과적으로 이 연구는 AI 모델 전반에 걸쳐 데이터 생성을 통한 개선의 일반적인 프레임워크를 제시합니다.



### Distorting Embedding Space for Safety: A Defense Mechanism for Adversarially Robust Diffusion Models (https://arxiv.org/abs/2501.18877)
- **What's New**: 이번 논문에서는 텍스트-이미지 생성 모델에서 안전하지 않은 프롬프트로 인해 발생하는 NSFW 콘텐츠 생성 문제를 해결하기 위한 새로운 방안인 Distorting Embedding Space (DES)를 제안합니다. DES는 텍스트 인코더를 기반으로 하여 안전한 임베딩 공간을 제어하여 악의적인 공격에도 효과적으로 대응할 수 있도록 설계되었습니다. 이를 통해 불안전한 임베딩을 안전한 영역으로 변환하고, 원래의 안전한 임베딩을 재생산할 수 있습니다.

- **Technical Details**: DES는 Unsafe Embedding Neutralization (UEN), Safe Embedding Preservation (SEP) 및 Nudity Embedding Neutralization (NEN)이라는 세 가지 손실 함수를 사용하여 안전한 콘텐츠 생성을 위한 기본 구조를 갖추고 있습니다. 특히, DES는 안전하지 않은 임베딩과 안전한 임베딩 간의 유사성을 고려하여 안전한 벡터로 변환하는 과정을 실시합니다. 이러한 방식은 검증된 안전한 임베딩을 보존하면서도 고품질 이미지를 생성할 수 있도록 합니다.

- **Performance Highlights**: 다양한 공격 유형에 대한 실험 결과, DES는 기존의 방어 메커니즘보다 탁월한 성능을 보여주었습니다. 특히, 훈련에도 불구하고 90초로 단축시키고 제로 인퍼런스 오버헤드를 유지하는 효율성 덕분에 다양한 T2I 모델에 쉽게 적용될 수 있습니다. 이러한 특징들은 DES가 안전한 콘텐츠 생성을 위한 효과적인 프레임워크임을 뒷받침합니다.



### UP-VLA: A Unified Understanding and Prediction Model for Embodied Agen (https://arxiv.org/abs/2501.18867)
- **What's New**: 이번 논문에서는 Vision-Language-Action (VLA) 모델의 훈련 패러다임을 재검토하고, UP-VLA라는 통합 VLA 모델을 제안합니다. 기존의 Vision-Language 모델(VLM)에 대한 한계를 극복하기 위해 미래 예측 목표와 다중 모드 이해를 결합한 훈련 방법을 적용하여, 상위 수준의 의미 이해와 하위 수준의 공간 이해를 모두 강화합니다. 실험 결과, UP-VLA는 기존의 최신 방법에 비해 Calvin ABC-D 벤치마크에서 33% 향상된 성능을 보였습니다.

- **Technical Details**: UP-VLA는 다중 모드 이해 데이터셋과 미래 예측 생성 데이터를 통해 고급 특성과 저급 특성을 동시에 일치시키는 새로운 훈련 패러다임을 사용합니다. 모델은 세 가지 유형의 데이터를 기반으로 오토 회귀 모델을 공동 훈련하여, 다양한 시뮬레이션 및 실제 환경에서 알고리즘의 능력을 평가하였습니다. 특정 실험에서는 VLM 기반 VLA 모델이 전형적인 다중 작업 학습 환경에서 강한 성능을 보였고, 시각 예측 기반의 사전 훈련 방식이 적응성과 정밀 제어가 중요한 작업에서 더 나은 성과를 나타냈습니다.

- **Performance Highlights**: UP-VLA 모델은 Calvin ABC-D 벤치마크에서의 성능이 33% 증가했으며, 실제 조작 과제에서도 개선된 성공률을 보여주었습니다. 특히, 정밀한 공간 정보가 요구되는 실제 작업에서 뚜렷한 성과 향상이 나타났습니다. 이는 UP-VLA 방식이 의미적 및 저급 특성을 모두 보존하는 데 효과적임을 강조합니다.



### REG: Rectified Gradient Guidance for Conditional Diffusion Models (https://arxiv.org/abs/2501.18865)
Comments:
          19 pages, 10 figures

- **What's New**: 이 연구는 diffusion 모델에서 조건부 생성(condition generation)을 향상시키기 위한 guidance 기법의 이론적 기초와 실제 구현 간의 불일치를 해소하고자 합니다. 기존의 스케일된 한계분포(target) 대신 유효한 스케일된 결합 분포(objective)를 설정함으로써 이론적으로 잘못된 부분을 교정합니다. 또한, 기존의 guidance 기법이 실제로 최적 솔루션에 대한 근사치에 불과하다는 점을 확인하고, 이를 바탕으로 rectified gradient guidance (REG)를 제안합니다.

- **Technical Details**: Diffusion 모델은 데이터의 고유 분포를 모델링하여 새로운 데이터 샘플을 생성하는 방법으로, 최근에는 Gaussian 노이즈를 점진적으로 추가하는 forward 과정과 신경망으로 학습된 reverse 과정을 통해 고품질 샘플 복원이 이루어집니다. guidance 기법은 샘플링 시 사용자 정의 가이던스 신호를 기반으로 노이즈 예측 네트워크의 출력을 업데이트하여 모드 커버리지(mode coverage)와 샘플 충실도(sample fidelity)를 조정합니다. 본 연구에서는 guidance 기법의 이론적 맥락을 명확히 하고, 각 기법이 최적 솔루션에 다가가는 근사화로 작용함을 이론적으로 분석합니다.

- **Performance Highlights**: 연구 결과, rectified gradient guidance (REG)는 기존의 guidance 기법들보다 최적 솔루션에 더 나은 근사를 제공하며, 1D 및 2D 실험을 통해 VALIDATE되었습니다. 구체적으로, ImageNet과 text-to-image 생성 작업에 대해 REG를 포함할 경우 FID 및 Inception/CLIP 점수가 일관되게 개선되는 결과를 보였습니다. 이러한 결과는 REG가 다양한 조건부 생성 작업에서 효과적임을 입증하며, 제안된 이론적 프레임워크를 뒷받침합니다.



### Test-time Loss Landscape Adaptation for Zero-Shot Generalization in Vision-Language Models (https://arxiv.org/abs/2501.18864)
- **What's New**: 이 논문에서는 기존의 Test-time Prompt Tuning (TPT) 방법의 한계를 극복하기 위해 Test-time Loss Landscape Adaptation (TLLA)이라는 새로운 프레임워크를 제안합니다. TLLA는 테스트 손실 경량의 최솟값을 찾는 과정에서 역전파(backpropagation)를 필요로 하지 않으며, 이는 계산 비용을 크게 줄입니다. 이 방법은 훈련의 손실 경량과 테스트 손실 경량 간의 상대적 위치를 활용하여 모델 파라미터를 수정하지 않고도 적응 과정을 유도합니다. 또한 Extensive한 실험을 통해 연구성과를 입증하고 있으며, 상태-of-the-art 성능을 달성했습니다.

- **Technical Details**: TLLA 프레임워크는 두 개의 주요 단계로 구성됩니다: 첫 번째 단계인 프롬프트 튜닝 중에는 Sharpness-Aware Prompt Tuning (SAPT) 방법을 도입하여 훈련 데이터 세트와 함께 프롬프트를 미세 조정합니다. 두 번째 단계인 테스트 단계에서는 각 테스트 샘플에 대해 여러 증강 버전을 생성하고, Sharpness-based Test Sample Selection (STSS) 방법을 통해 훈련 손실 경량과 각 증강 테스트 샘플의 손실 경량 사이의 정합성을 확인합니다. TLLA는 그래디언트 하강법을 직접 사용하지 않고 특정 증강 버전을 선택함으로써 손실의 모양을 변경합니다.

- **Performance Highlights**: TLLA는 ResNet50 및 ViT-B/16 이미지 인코더를 사용할 때 각각 5.32% 및 6.98%의 성능 개선을 나타내며, 기존 TPT 방법을 초월합니다. 연구 결과는 도메인 일반화 및 크로스 데이터세트 벤치마크에서의 광범위한 실험을 통해 입증됩니다. 이를 통해 TLLA는 비전-언어 모델의 제로-샷 일반화를 크게 향상시킬 수 있음을 보여주고 있습니다.



### FlexiCrackNet: A Flexible Pipeline for Enhanced Crack Segmentation with General Features Transfered from SAM (https://arxiv.org/abs/2501.18855)
- **What's New**: FlexiCrackNet은 기존의 'pre-training + fine-tuning' 접근 방식을 혁신적으로 통합한 새로운 파이프라인으로, 다양한 데이터 세트와 제한된 자원 환경에 대한 적응력을 높입니다. EdgeSAM의 CNN 기반 인코더를 활용해, 입력 이미지의 크기와 상관없이 유연성을 극대화하여 크랙 세분화 작업에 최적화된 모델을 제공합니다.

- **Technical Details**: FlexiCrackNet은 인코더-디코더 아키텍처를 채택하여 특정 작업에 적합한 특징을 추출하며, 정보 상호작용 게이트 주의 메커니즘(IGAM)을 도입하여 다층 특징을 적응적으로 융합합니다. 이 모델은 다수의 입력 해상도에 적응 가능하며, 과부하를 줄이고 크랙 세분화 성능을 극대화합니다.

- **Performance Highlights**: FlexiCrackNet은 모호한 입력 및 복잡한 배경을 포함한 도전적 시나리오에서도 뛰어난 세분화 성능을 보여주며, 현재 최첨단 방법을 초월하는 성과를 나타냅니다. 또한 zero-shot 일반화 능력이 탁월하고, 계산 효율성 및 세분화 강인성이 향상되어 실제 적용 가능성을 강조합니다.



### Project-and-Fuse: Improving RGB-D Semantic Segmentation via Graph Convolution Networks (https://arxiv.org/abs/2501.18851)
- **What's New**: 이 논문에서는 기존의 RGB-D 의미 분할(RGB-D semantic segmentation) 방법의 한계를 극복하기 위해 텍스처 우선 방식에서 두 모달리티의 특징을 융합하고, 그래프 신경망(Graph Neural Networks)을 사용하여 비정상 패치 문제를 해결하는 새로운 접근 방식을 제안합니다. 또한, 깊이 맵(depth map)을 노멀 맵(normal map)으로 인코딩하여 CNN이 물체 표면을 효율적으로 추출하도록 하며, Kullback-Leibler 손실을 도입하여 중요한 픽셀 특징이 누락되지 않도록 보장합니다.

- **Technical Details**: 본 연구의 기술적 기여는 RGB-D 데이터를 융합하는 방식에 있습니다. 첫째, 3D 브랜치를 기존의 픽셀-노드-픽셀 파이프라인에 추가하여 두 모달리티의 특징을 텍스처 전이 스타일로 융합합니다. 둘째, 그래프 구성 과정에서 Biased-Assignment와 Ambiguous-Locality 문제를 완화하기 위해 인접 행렬(adjacency matrix) 생성을 변경하였으며, 프로젝션 매트릭스(projection matrix)에 제약 조건을 추가했습니다.

- **Performance Highlights**: NYU-DepthV2 및 SUN RGB-D라는 두 개의 공개 데이터 세트에서 수행된 광범위한 실험을 통해, 제안된 방법이 RGB-D 의미 분할 작업의 성능을 지속적으로 향상시킬 수 있음을 보여주었습니다. 이 연구는 최신 기술 수준(state-of-the-art) 성능을 달성했으며, 복잡한 실내 구조물과 다양한 차폐 문제를 해결하는 데 기여합니다.



### Early Diagnosis and Severity Assessment of Weligama Coconut Leaf Wilt Disease and Coconut Caterpillar Infestation using Deep Learning-based Image Processing Techniques (https://arxiv.org/abs/2501.18835)
- **What's New**: 이 연구는 스리랑카에서 코코넛 나무에 영향을 미치는 두 가지 주요 질병인 Weligama Coconut Leaf Wilt Disease (WCWLD)와 Coconut Caterpillar Infestation (CCI)의 조기 탐지를 위한 딥 러닝 (Deep Learning) 기술의 활용을 제안합니다. Transfer Learning 기반의 Convolutional Neural Network (CNN) 및 Mask Region-based-CNN (Mask R-CNN) 모델을 사용하여 이 두 질병을 식별하고 심각성을 평가하는 데 효과적임을 입증했습니다. 추가로, YOLO (You Only Look Once) 객체 탐지 모델을 통해 CCI에 의해 분포된 애벌레 수를 세는 방법도 소개되었습니다.

- **Technical Details**: 실험은 스리랑카 Matara, Puttalam 및 Makandura 지역에서 수집된 데이터셋을 기반으로 진행되었습니다. WCWLD와 CCI를 식별하는 정확도는 각각 90%와 95%로 보고되었습니다. 또한, WCWLD의 병의 심각도를 판별하는 방법의 정확도는 97%에 달합니다. YOLO 모델을 사용한 애벌레 수 계산의 정확도는 YOLOv5가 96.87%, YOLOv8이 96.1%, YOLO11이 95.9%로 나타났습니다.

- **Performance Highlights**: 이 연구에서 제안한 기술들은 스리랑카 및 인근 코코넛 생산국가에서 발생하는 코코넛 생산 손실을 줄이는 데 기여할 수 있는 잠재력을 가지고 있습니다. 연구 결과를 통해, 학습된 모델들이 기존의 수작업 관찰 방법과 비교하여 매우 높은 정확도로 질병을 조기에 탐지할 수 있음을 보여주었습니다. 이는 농민들이 신속하게 필요 조치를 취할 수 있도록 하여, 코코넛 재배의 효율성을 상당히 향상시킬 것으로 기대됩니다.



### Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion (https://arxiv.org/abs/2501.18804)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 MVGD라는 새로운 확산 기반 아키텍처를 소개합니다. 이 시스템은 입력된 이미지에서 직접 픽셀 수준의 생성 및 깊이 맵 작업을 지원하여 다수의 시점에서 일관된 3D 장면을 재구성합니다. MVGD는 학습 가능한 작업 임베딩을 사용하여 이미지와 깊이를 포함하는 멀티태스킹 생성을 수행하며, 기존의 중간 3D 표현 없이 다양한 입력 뷰에 대해 효과적으로 작동합니다.

- **Technical Details**: MVGD 아키텍처는 6000만 개 이상의 공개 데이터셋을 사용하여 훈련되었습니다. 이 방법은 레이맵 조건화(raymap conditioning)를 사용해 다양한 시점에서 스페이셜 정보를 시각적 특징에 통합합니다. 또한, 비율을 유지하며, 멀티뷰 일관된 예측을 생성하는 데 중점을 두고 깊이 맵과 이미지를 동시에 생성합니다.

- **Performance Highlights**: 본 연구에서는 여러 새로운 뷰 합성 벤치마크와 다중 뷰 스테레오, 비디오 깊이 추정 작업에서 최첨단 성능을 보여주었습니다. MVGD는 100개 이상의 조건부 뷰를 사용할 수 있으며, 기존 모델에 비해 최대 70%의 훈련 시간 절감 효과를 가지고 있습니다. 이 모든 요소는 일반화 가능한 새로운 뷰 합성 분야에서 새로운 최첨단 성과를 설정하는 데 기여했습니다.



### Every Image Listens, Every Image Dances: Music-Driven Image Animation (https://arxiv.org/abs/2501.18801)
- **What's New**: 이번 연구에서는 MuseDance라는 새롭고 혁신적인 엔드-투-엔드 모델을 소개합니다. MuseDance는 음악과 텍스트 입력을 동시에 활용하여 참조 이미지를 애니메이션화합니다. 이중 입력 방식을 통해 사용자는 자신이 원하는 텍스트 설명에 따라 개인화된 비디오를 생성할 수 있으며, 음악에 맞춘 캐릭터의 움직임을 동기화할 수 있습니다. 기존 접근법과는 달리 MuseDance는 복잡한 모션 가이드를 요구하지 않습니다.

- **Technical Details**: 본 연구는 두 단계로 구성된 훈련 프레임워크를 사용하여 이미지 애니메이션을 구현합니다. 첫 번째 단계에서는 목표 비디오의 개별 프레임을 학습하여 시각적 특징을 획득하고, 두 번째 단계에서는 음악과 텍스트를 트리거로 사용하여 이들 입력에 부합하는 애니메이션 프레임을 생성합니다. 이 과정은 스테이블 디퓨전(Stable Diffusion) 모델을 기반으로 하며, 음악 특징과 텍스트 안내를 연결하여 일관되고 역동적인 애니메이션을 생성합니다.

- **Performance Highlights**: MuseDance는 2,904개의 댄스 비디오와 이에 상응하는 배경 음악 및 텍스트 설명으로 구성된 새로운 다중모달 데이터셋을 제시합니다. 고유한 음악의 흐름과 비디오 시퀀스 간의 의미론적 정렬을 가능하게 하여 매력적이고 역동적인 애니메이션을 생성할 수 있습니다. 우리 모델은 유연하고 정밀한 제어능력과 다양한 객체에 대한 일반화 능력을 선보이며, 음악 기반 비디오 생성 작업의 기준을 설정합니다.



### Tuning Event Camera Biases Heuristic for Object Detection Applications in Staring Scenarios (https://arxiv.org/abs/2501.18788)
Comments:
          17 pages, 2 figures

- **What's New**: 이번 논문에서는 이벤트 카메라의 편향(bias) 파라미터를 조정하기 위한 새로운 방법론을 제시합니다. 특히, 정지 상태(staring scenario)에서 소형 객체를 감지하는 작업에 최적화된 휴리스틱(heuristic)을 개발하여 카메라의 성능을 극대화하고 탐지 능력을 확장하는 것을 목표로 합니다. 이전 연구에서 잘 알려진 다중 파라미터 문제를 단순화하여 두 개의 파라미터 문제로 축소하는 방법을 보여줍니다.

- **Technical Details**: 이벤트 카메라는 생체모방(bio-inspired) 센서로서, 각 픽셀(pixel)이 비동기(asynchronous) 방식으로 독립적으로 동작하여 사전 정의된 임계값(threshold)을 초과하는 밝기 변화에만 반응합니다. 달리 말하면, 이 카메라는 전체 시야를 일정한 샘플링 속도(sampling rate)로 촬영하는 기존의 카메라와는 다르게 작동합니다. 본 논문은 이러한 이벤트 카메라의 편향을 조정하기 위한 특정한 수학적 모델을 사용하여, 카메라의 성능과 노이즈가 편향에 의존하는 특정 사례를 논의합니다.

- **Performance Highlights**: 제시된 휴리스틱은 전기 그리드에 의해 구동되는 백열 전구와 같은 특정 신호에 대해, 카메라의 최적 값이 제조업체가 추천하는 기본값과 상당히 차이가 남을 수 있음을 보여줍니다. 이 연구를 통해 이벤트 카메라가 프레임 기반(frame-based) 카메라보다 더 나은 성능을 제공할 수 있는지 확인하기 위한 중요한 기반이 마련되었습니다. 실험적 검증을 통해 이론적으로 제안된 두 차원 문제를 해결하는 실용적인 방법을 구체적으로 제시하였으며, 이는 이벤트 카메라의 다양한 응용 가능성을 열어줍니다.



### Multispectral 3D mapping on a Roman sculpture to study ancient polychromy (https://arxiv.org/abs/2501.18786)
Comments:
          14 pages, 5 figures, to be published in the proceedings of "Heri-Tech - The Future of Heritage Science And Technologies" Conference by Springer, 29-30 April 2024, Florence, Italy (this https URL)

- **What's New**: 이 연구에서는 고대 그리스 및 로마 조각품의 원래 색상인 polychromy에 대한 새로운 접근법을 제시합니다. 기존에 사용되는 색상 분석 기법의 한계를 극복하기 위해, 현실 기반 3D 모델을 활용한 새로운 방법론을 적용하였습니다. 이 방법론은 인지할 수 있는 영역을 넘어서는 텍스처를 포함하여 고대 조각품의 원래 모습을 분석하는 데 중점을 둡니다.

- **Technical Details**: 연구에서는 Visible Reflected Imaging (VIS)와 Ultraviolet-induced Fluorescence Imaging (UVF) 기술을 활용하여 3D 모델을 구축하였습니다. 이 과정에서 다양한 조명 소스를 사용하여 촬영된 이미지를 정렬하고 처리한 후, 단일 3D 모델로 통합하였습니다. 각기 다른 텍스처의 픽셀 간 대응관계를 이용하여 분류 알고리즘을 구현함으로써, 최종 결과를 3D 모델 표면에 직접적으로 매핑할 수 있습니다.

- **Performance Highlights**: 본 연구는 이탈리아 그로세토에 위치한 Maremma 고고학 및 미술 박물관(Archeological and Art Museum of Maremma, MAAM)의 아르테미스 조각상에 이 접근법을 적용해 보였습니다. 이를 통해 보존가들은 유물 보존에 대한 이해를 심화하고, 물질 분포를 자세히 관찰하며, 이를 3D 기하학적 데이터와 연관시킬 수 있음을 보여주었습니다. 이러한 연구 결과는 고대 예술 작품의 보존 및 복원 분야에서 혁신적인 기여를 할 것으로 기대됩니다.



### RUN: Reversible Unfolding Network for Concealed Object Segmentation (https://arxiv.org/abs/2501.18783)
Comments:
          13 tables, 8 figures

- **What's New**: 이번 논문에서는 기존의 Concealed Object Segmentation (COS) 방법들이 RGB 도메인을 충분히 활용하지 못하고 있다는 점을 지적하며, Reversible Unfolding Network (RUN)라는 새로운 접근 방식을 제안합니다. RUN은 마스크 도메인과 RGB 도메인에서 모두 가역적 전략을 적용하여 정확한 분할(segmentation)을 가능하게 합니다. 이 모델은 잔여 희소성 제약(residual sparsity constraint)을 포함하여 분할의 불확실성을 최소화하는 새로운 COS 모델을 구성합니다.

- **Technical Details**: RUN은 Segmentation-Oriented Foreground Separation (SOFS) 모듈과 Reconstruction-Oriented Background Extraction (ROBE) 모듈이라는 두 가지 가역적 모듈로 구성된 다단계 네트워크(multi-stage network)입니다. SOFS는 마스크 수준의 가역 전략을 적용하고 비국소(non-local) 정보를 캡처하는 Reversible State Space(RSS) 모듈을 통합하여 세분화된 마스크를 정제합니다. ROBE는 RGB 도메인으로 확장되어, 드라이버(distortion-prone) 영역을 다룰 수 있는 재구성 네트워크를 사용하여 상충되는 전경과 배경 영역을 처리합니다.

- **Performance Highlights**: RUN은 각각의 COS 작업에 대한 실험을 통해 기존 방법에 비해 뛰어난 성능을 발휘함을 입증했습니다. 가역 모델링을 통해 불확실한 영역에 집중하며, 잘못된 긍정(false-positive)과 잘못된 부정(false-negative) 결과를 감소시킵니다. 또한, RUN의 플러그 앤 플레이 구조는 COS 작업과 기타 고급 비전 작업에 대한 효과성과 적응성을 강조합니다.



### A New Statistical Approach to the Performance Analysis of Vision-based Localization (https://arxiv.org/abs/2501.18758)
Comments:
          14 pages

- **What's New**: 현대의 무선 장치들은 카메라, 레이더 및 LiDAR와 같은 비전 센서에 접근할 수 있는 광범위한 기능을 제공합니다. 이 논문에서는 무선 기반의 위치 결정이 부정확하거나 불가능한 상황에서 비전 센서의 정보를 활용하여 변별력 있는 지역 랜드마크를 식별하는 새로운 프레임워크를 제안합니다. 특히, 여러 개의 근접 랜드마크에 대한 거리 측정을 통해 목표의 위치를 정확히 결정할 수 있는 기하학적 제약을 도입했습니다.

- **Technical Details**: 제안된 방법은 랜드마크를 마크된 포아송 점 과정(marked Poisson point process)으로 모델링하며, 이를 통해 세 개의 노이즈 없는 거리 측정만으로도 두 차원 평면에서 랜드마크 조합을 독특하게 결정할 수 있음을 보여줍니다. 또한, 노이즈가 있는 측정의 경우, 랜드마크 조합을 올바르게 식별할 확률을 새롭게 도출한 주요 확률 변수의 결합 분포를 통해 수학적으로 설명합니다. 이를 통해 비전 데이터로부터 얻어진 거리 측정만으로도 정확한 랜드마크 조합을 식별할 수 있음을 입증했습니다.

- **Performance Highlights**: 제안된 프레임워크는 비정상적으로 유사하게 보이는 랜드마크를 다루는데 강력한 성능을 보여줍니다. 기술적 기여는 거리 측정에서 얻은 기하학적 제약을 기반으로 한 알고리즘을 포함하며, 이는 다양한 유형의 랜드마크 활용에 대한 포괄적인 분석을 제공합니다. 따라서 이 연구는 비전 기반의 위치 결정 시스템에서 나타나는 모호성을 극복하기 위한 유의미한 기여를 하고 있습니다.



### INT: Instance-Specific Negative Mining for Task-Generic Promptable Segmentation (https://arxiv.org/abs/2501.18753)
Comments:
          A new task-generic promptable segmentation approach

- **What's New**: 이번 논문에서는 Task-generic promptable image segmentation 방식의 한계를 극복하기 위해 INSTANCE-specific NEGATIVE MINING 기법(INT)을 도입하였습니다. INT는 신뢰할 수 없는 prior knowledge의 영향을 줄이는 동시에, 가장 그럴듯한 prior knowledge를 증가시키는 방식을 통해 instance-specific prompts 생성을 최적화하는 것을 목표로 합니다. 이를 통해 다양한 이미지 샘플을 하나의 작업 설명으로 세분화할 수 있는 기능이 향상되었습니다.

- **Technical Details**: INT는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) instance-specific prompt generation, 즉 프로세스가 진행되면서 잘못된 정보를 제외하는 기능, (2) semantic mask generation으로 각 이미지 인스턴스의 세분화가 instance-specific prompts의 의미에 우선 일치하도록 보장합니다. 이 방법들은 이미지의 패치를 병렬로 처리하여 관심 있는 객체의 존재를 예측하고, VLM에서의 출력 변화를 기반으로 더 일관된 prompts 생성을 위한 단계를 제공합니다.

- **Performance Highlights**: 다양한 테스트에서 INT는 camouflaged objects와 의료 이미지 포함, 여섯 개의 데이터 세트에서 검증되었으며, 효과적인 성능과 견고성 및 확장성을 보여주었습니다. 실험 결과에 따르면 기존의 방법보다 더 정확하게 instance-specific prompts를 생성하고, 잘 구별되지 않는 오류 카테고리를 식별함으로써 segmentation 성능을 향상시켰습니다.



### Motion Diffusion Autoencoders: Enabling Attribute Manipulation in Human Motion Demonstrated on Karate Techniques (https://arxiv.org/abs/2501.18729)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구는 인간 동작 데이터의 속성(attribute)을 조작하는 데 초점을 맞춘 최초의 성공적인 결과를 제시합니다. 특히 가라데 움직임 패턴을 사용하여 기술과 기술 수준을 조작하고, 다른 속성은 보존하는 방법을 모색합니다. 연구진은 새로운 회전 기반의 포즈 표현(rotation-based pose representation)을 설계하여 인간 골격과 운동 궤적을 분리하고 정확한 해부학적 복원을 돕고 있습니다.

- **Technical Details**: 이 연구의 핵심 아이디어는 transformer encoder를 사용하여 고수준 의미(semantic)를 발견하고, diffusion probabilistic model을 사용하여 나머지 확률적 변동성(stochastic variations)을 모델링하는 것입니다. 고수준 속성을 조작하기 위해, 연구진은 의미 임베딩 공간(embedding space)의 선형적 변화를 찾아 방향을 이동시킴으로써 조작이 가능하다고 증명하고 있습니다. 이를 통해 각 선수의 특정 속성에 대한 세밀한 조작이 가능해집니다.

- **Performance Highlights**: 인간 동작 데이터의 속성 조작 성능을 평가하기 위한 벤치마크(benchmark)가 존재하지 않아 두 가지 주요 난관에 직면했습니다. 첫째, 대규모 적합 데이터셋이 부족하고, 둘째, 해부학적 세부 정보를 반영하지 않는 데이터셋을 제외해야 하였습니다. 이 연구는 적합한 기법과 데이터셋을 통해 동작 속성을 효과적으로 조작할 수 있는 첫 걸음을 내딛었습니다.



### Strong and Controllable 3D Motion Generation (https://arxiv.org/abs/2501.18726)
Comments:
          technical report

- **What's New**: 이번 논문에서는 인간 모션 생성에 있어 실시간 응용을 위한 새로운 아키텍처를 제안합니다. 최신 기술들에 비해 하드웨어 효율성과 계산 복잡성을 개선하고, 한층 더 세밀한 관절 수준의 제어가 가능한 Motion ControlNet을 도입했습니다. 이러한 혁신을 통해 텍스트-모션 생성의 진전을 이루어내어 실제 적용 가능성을 높였습니다.

- **Technical Details**: 제안된 Efficient Motion Transformer는 지연 움직임 확산 U-Net Denoiser 안에 포함된 기본 구조입니다. 이 모델은 하드웨어 최적화를 위한 flash linear attention을 커스터마이징하여, 선형 복잡도를 통한 빠르고 확장 가능한 처리 방식을 제공합니다. Motion ControlNet을 통해 사용자가 제공한 관절의 경로를 활용한 정밀한 움직임 제어가 가능해졌습니다.

- **Performance Highlights**: 이 연구는 기존 텍스트-모션 생성 기법의 낮은 효율성과 부정확한 관절 제어 문제를 해결하는 데 기여하며, 실시간 응용에 적합한 모션 생성 성능을 가능하게 합니다. 실제로, Motion ControlNet과 모션 잠재 일관성 증류를 통해 생성 속도를 가속화하고, 사용자가 요구하는 정밀한 모션을 생성함으로써 게임, 로봇 조작, 가상 현실 등의 분야에서 큰 가능성을 보여줍니다.



### Full-Head Segmentation of MRI with Abnormal Brain Anatomy: Model and Data Releas (https://arxiv.org/abs/2501.18716)
- **What's New**: 이 연구의 목표는 임상 MRI를 포함한 전체 머리 분할(whole-head segmentation)을 위한 딥 뉴럴 네트워크(Deep Network)를 개발하고, 이를 위한 최초의 공개 벤치마크 데이터셋을 구성하는 것이었습니다. 91개의 MRI를 수집하여 다양한 인체 가수를 위한 체적 분할 레이블을 포함 시켰으며, 정상 4명, 외상성 뇌손상 32명, 뇌졸중 57명으로 구성되어 있습니다. 이러한 사례들은 일반적으로 뇌가 있는 영역에서 뇌척수액(CSF)이 확장된 것이 특징입니다.

- **Technical Details**: MultiAxial 네트워크는 시상면(sagittal), 축면(axial), 관상면(coronal)에서 독립적으로 작동하는 세 개의 2D U-Net 모델로 구성되어 있으며, 최종적으로 단일 3D 분할을 위해 결합됩니다. 훈련 레이블은 피부/두피(skin/scalp), 두개골(skull), CSF, 회색물질(gray matter), 백색물질(white matter), 공기 공간(air cavity), 외부 공기(extracephalic air)를 위한 초기 자동 분할을 수동으로 수정하여 생성되었습니다.

- **Performance Highlights**: MultiAxial 네트워크는 테스트 세트에서 0.88(중앙값 ± 0.04)의 Dice 점수를 달성했습니다. 뇌 조직에 대해서는 기존 뇌 분할 방법보다 상당히 뛰어난 성능을 보이며, MultiAxial: 0.898 ± 0.041, SynthSeg: 0.758 ± 0.054, BrainChop: 0.757 ± 0.125입니다. 이 네트워크는 아틀라스(atlas)와의 코레지스트레이션(coregistration) 필요 없이 견고함을 더하며, 비슷하지 않은 해부학적 구조에서도 좋은 성능을 보입니다.



### Human Re-ID Meets LVLMs: What can we expect? (https://arxiv.org/abs/2501.18698)
- **What's New**: 이 연구는 대형 비전-언어 모델(LVLMs)의 인간 재식별(human re-identification) 작업 성능을 비교합니다. 특히, 특정 문제를 위해 설계된 최신 AI 모델들과 LVLMs를 비교하여 실질적인 성능 차이를 조명하고 있습니다. 우리 연구에서는 ChatGPT-4o, Gemini-2.0-Flash, Claude 3.5 Sonnet, Qwen-VL-Max 등 여러 모델을 Market1501 데이터셋에서 평가했습니다.

- **Technical Details**: 평가 파이프라인은 데이터셋 큐레이션(dataset curation), 프롬프트 엔지니어링(prompt engineering), 그리고 모델 성능을 측정하기 위한 메트릭(metric) 선택을 포함합니다. 모델의 성능을 비교하기 위해 유사도 점수(similarity scores), 분류 정확도(classification accuracy), 정밀도(precision), 재현율(recall), F1 점수(F1 score), 그리고 곡선 아래 면적(AUC) 등의 다양한 지표를 사용합니다.

- **Performance Highlights**: 결과를 통해 LVLMs의 강점과 함께 심각한 한계를 확인하였으며, 이는 종종 재난적인 답변(catatrophic answers)을 초래합니다. 최종 결론으로는 전통적인 방법과 LVLMs를 통합하여 두 기술 집합의 강점을 결합할 수 있는 향후 연구의 방향성을 제시하고 있습니다.



### Unpaired Translation of Point Clouds for Modeling Detector Respons (https://arxiv.org/abs/2501.18674)
Comments:
          NeurIPS Machine Learning and the Physical Sciences Workshop 2025

- **What's New**: 본 논문에서는 시뮬레이션 데이터와 실험 데이터 간의 포인트 클라우드 번역을 위한 새로운 프레임워크를 제안합니다. 이는 확률적 확산 모델(diffusion probabilistic models)을 기반으로 하여, 시간 투사 챔버의 검출기 응답 모델링 문제를 비쌍(point cloud translation)으로 재구성합니다. 제안된 방법은 AT-TPC에서의 실험 데이터와 합성 데이터 모두에서 효과적으로 검증됩니다.

- **Technical Details**: 본 연구는 CycleDiffusion을 활용하여 두 도메인 간의 포인트 클라우드 변환을 수행합니다. 각 도메인에서 독립적으로 훈련된 두 개의 확산 모델을 사용하여 원본 포인트 클라우드를 인코딩하고, 이를 다른 모델로 디코딩하여 번역을 수행합니다. 특히, 실험 데이터에 대한 잡음을 제거하거나 시뮬레이션 이벤트의 검출기 응답을 모델링하는 G 및 H라는 두 개의 모델을 훈련합니다.

- **Performance Highlights**: 제안된 방법은 합성 포인트 클라우드와 AT-TPC 실험 데이터를 활용하여 성능을 평가하였습니다. 특히, 실험 데이터에 대한 적합성을 높이기 위해 잡음 제거에 성공하였으며, 시뮬레이션 데이터의 검출기 응답을 효과적으로 모델링 할 수 있음을 보여주었습니다. 이를 통해 고충실도 시뮬레이터 구축 및 잡음 제거에서 중요한 발전을 이룩하였습니다.



### High-Accuracy ECG Image Interpretation using Parameter-Efficient LoRA Fine-Tuning with Multimodal LLaMA 3.2 (https://arxiv.org/abs/2501.18670)
- **What's New**: 이번 연구는 electrocardiogram (ECG) 이미지를 해석하는 새로운 접근법을 제시합니다. 우리는 multimodal LLaMA 3.2 모델을 사용하여, Low-Rank Adaptation (LoRA)라는 파라미터 효율적인 미세 조정 전략을 통해 ECG 분석을 강화했습니다. 이 방법은 임상 상황에서 다양한 심장 병리 식별의 정확성을 높이기 위한 목적으로 설계되었습니다.

- **Technical Details**: 모델은 LLaMA 3.2 아키텍처를 기반으로 하며, ECG 이미지의 다양한 특성을 효과적으로 포착하기 위한 두 단계의 비전 인코더로 구성되어 있습니다. 비전 인코더는 먼저 이미지를 표준 크기로 리사이징한 후, 14x14 픽셀의 비겹치는 패치로 나누어 고급 특성 추출을 수행합니다. LoRA 방법론 덕분에 특정 레이어를 제외하고 소수의 파라미터 만 업데이트하여 ECG 전문가 수준의 해석 능력을 달성했습니다.

- **Performance Highlights**: 실험 결과, 우리의 LoRA 미세 조정 방법은 ECG 이미지 해석에서 기존 모델을 능가하고 70가지 이상의 심장 질환을 높은 정확도로 식별할 수 있음을 보여주었습니다. 이 연구는 이미지 기반의 ECG 분석에서 중요한 진전을 이루었으며, 전통적인 CNN 기반 방법과 비견될 수 있는 성능을 달성했습니다. 이러한 성과는 다양한 심장 질환에 대한 신뢰할 수 있는 임상적 결정 지원을 위한 기반을 마련해 줍니다.



### Image, Text, and Speech Data Augmentation using Multimodal LLMs for Deep Learning: A Survey (https://arxiv.org/abs/2501.18648)
- **What's New**: 최근 5년 동안 연구가 전통적인 Machine Learning(ML) 및 Deep Learning(DL) 기법에서 Large Language Models(LLMs) 활용으로 이동했습니다. 이러한 변화는 다중 모달(multimodal) 데이터를 통한 데이터 증강을 통해 일반화 능력을 향상시키고, 심층 합성곱 신경망의 과적합(overfitting)을 방지하는 데 초점을 맞추고 있습니다. 특히, 이 설문 조사는 LLM 기반 기법의 최신 발전과 다중 모달 응용에 대한 연구의 공백을 메우고 있습니다.

- **Technical Details**: 본 논문에서는 LLM을 사용한 이미지, 텍스트, 오디오 데이터를 증강하기 위한 다양한 방법들을 제시합니다. 기존의 ML 및 DL 기술이나 제한된 데이터 모달리티에 집중한 연구들과 달리, 이 설문 조사는 다중 모달 LLM들을 사용한 접근 방식에 대해 구체적으로 분석합니다. 또한, 현재 방법론에서 발견된 한계점을 논의하고, 이를 해결하기 위한 잠재적인 솔루션을 문헌에서 도출하였습니다.

- **Performance Highlights**: 이 설문 조사는 다중 모달 LLM을 사용하여 데이터 증강 관행의 효율성을 향상시키기 위한 기초 자료로 기능합니다. 연구의 궁극적인 목표는 심층 학습 응용 프로그램을 위한 데이터셋의 품질과 다양성을 향상시키는 것입니다. LLM 데이터 증강 관련 여러 키워드와 기술을 통해 향후 연구 방향을 제시하고 있습니다.



### 3D Reconstruction of Shoes for Augmented Reality (https://arxiv.org/abs/2501.18643)
- **What's New**: 이 논문에서는 3D 모델링과 증강 현실(AR)을 활용한 모바일 기반 솔루션을 소개하여 온라인 신발 쇼핑 경험을 향상시킵니다. 기존의 정적 2D 이미지의 한계를 극복하고, 2D 이미지에서 현실적인 3D 신발 모델을 생성하는 새로운 방법을 적용하였습니다. 평균 Peak Signal-to-Noise Ratio (PSNR) 0.32를 달성하며, 3120개의 이미지로 구성된 맞춤형 신발 세분화 데이터셋을 개발하여 Intersection over Union (IoU) 0.95를 기록한 최상의 세분화 모델을 구현했습니다.

- **Technical Details**: 본 연구는 3D Gaussian Splatting 기술을 활용하여 2D 이미지로부터 현실적인 3D 모델을 효율적으로 생성하는 방법을 제안합니다. 특히, 자주 발생하는 인간의 개입을 줄이며 고속의 모델 생성을 가능하게 하여, AR 응용 프로그램에서의 즉각적인 상호작용을 지원합니다. 이를 위해 이미지의 배경 제거와 신발 세분화를 위한 YOLOv8 모델을 사용하여 세밀한 결과를 얻었습니다.

- **Performance Highlights**: 제안된 방법은 3D 모델 생성에 있어 기존 방법에 비해 월등한 성능을 발휘하며, 실제 쇼핑 경험을 가상으로 재현하는 데 기여합니다. 특히, 비디오 데이터에서 3030개의 이미지를 활용하여 6.5MB 크기의 최종 모델을 제작함으로써 서버 호스팅에 필요한 계산 자원과 비용을 대폭 절감했습니다. 이 연구는 패션 산업 전반에 걸쳐 적용 가능한 혁신적인 가상 상호작용을 가능하게 합니다.



### DebiasPI: Inference-time Debiasing by Prompt Iteration of a Text-to-Image Generative Mod (https://arxiv.org/abs/2501.18642)
Comments:
          This work was presented at The European Conference on Computer Vision (ECCV) 2024 Workshop "Fairness and ethics towards transparent AI: facing the chalLEnge through model Debiasing" (FAILED), Milano, Italy, on September 29, 2024, this https URL

- **What's New**: 이 연구에서는 이미지 생성 시 인구 통계학적 속성의 분포를 제어할 수 있는 새로운 추론 시간 프로세스인 DebiasPI(Debiasing-by-Prompt-Iteration)를 제안합니다. 기존의 방법들은 모델을 재훈련해야 하거나 특정 성별 및 인종을 반영한 이미지를 생성하는 데 어려움을 겪고 있었던 반면, DebiasPI는 사용자가 생성된 이미지의 속성을 추적하여 요구하는 속성을 선택할 수 있는 방법을 제공합니다.

- **Technical Details**: DebiasPI는 모델의 내부 상태를 probe하거나 외부 속성 분류기를 통해 생성된 속성을 추적하며, 생성된 이미지에서의 속성 분포를 비교하는 도구를 제공합니다. 이 방법론은 질적 및 양적 콘텐츠 분석(QCA)를 통해 AI가 생성한 이미지의 인식 속성을 레이블링하는 코드북을 개발하여 수동 평가 과정도 포함합니다. 이를 통해 인종, 성별, 피부 톤의 다양성을 평가하고, 윤리적 개입 유도를 위한 충분한 데이터 기반을 제공합니다.

- **Performance Highlights**: DebiasPI를 사용하여 우리가 실험한 이미지 생성 결과는 성별 및 인종의 동등한 표현을 보여주는 것에 성공했습니다. 그러나 피부 톤의 다양성이 떨어지는 부작용이 발견되었고, 특정 인종의 피부 톤을 생성하는 데 어려움이 있었습니다. 다양한 개입 프롬프트를 통한 실험 결과, 모델은 여전히 젊고 남성적인 캐릭터를 생성하는 경향이 있으며, 이는 윤리적 개입의 필요성을 더욱 강조합니다.



### Image Velocimetry using Direct Displacement Field estimation with Neural Networks for Fluids (https://arxiv.org/abs/2501.18641)
- **What's New**: 해당 연구는 물리학 및 유체 역학의 실험 연구에서 필수적인 도구인 Particle Image Velocimetry (PIV) 기술을 개선하기 위한 새로운 접근법을 제안한다. 이 방법은 신경망(neural networks)과 광 흐름(optical flow) 방정식을 활용하여 순차적인 이미지 간의 변위 벡터(displacement vectors)를 예측한다. 기존의 방법들과 달리, 이 새로운 기법은 사전 훈련이 필요 없으며, 어떤 이미지 쌍에 대해서도 직접 사용할 수 있는 장점이 있다.

- **Technical Details**: 제안된 방법은 완전 연결 신경망(fully-connected neural networks)의 구조를 기반으로 하며, 이미지 좌표의 공간 임베딩(spatial embedding)을 입력으로 사용하여 다양한 해상도의 이미지를 처리할 수 있도록 한다. 이로 인해 추가적인 보간(interpolation) 없이도 픽셀 간의 초해상도를 추정할 수 있다. 연구에서는 변위 필드(displacement field)를 분석하여 속도 필드(velocity field)를 얻는 방법을 제시하며, 이를 기반으로 우측에서 관찰되는 모든 속도 영역을 가시화할 수 있다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 순간 속도 필드 및 시간 평균 난류 수치(turbulence quantities), 파워 스펙트럼 밀도(power spectral density) 계산에서 높은 정확도를 달성했다. 본 연구는 기존의 기계 학습(machine learning) 기반 접근법의 한계를 극복하여, 다양한 실험 환경에 활용 가능한 신속하고 효율적인 방법을 제시하고 있다. 향후 연구에서는 이 방법의 범위를 넓혀, 실제 실험 환경에서도 적용 가능성을 탐색할 예정이다.



### Machine learning of microstructure--property relationships in materials with robust features from foundational vision transformers (https://arxiv.org/abs/2501.18637)
- **What's New**: 이 논문에서는 데이터에서 마이크로 구조-성질 관계를 기계 학습하는 새로운 접근 방식을 제안합니다. 기존의 태스크 특화 모델 대신, 사전 훈련된 비전 트랜스포머(vision transformers)를 활용하여 태스크 비의존적인 마이크로 구조 특성을 추출하고, 경량 기계 학습 모델을 통해 마이크로 구조에 따른 성질을 예측합니다. 이를 통해 고가의 특화 훈련이나 딥러닝 모델의 미세 조정 없이 효과적인 결과를 얻을 수 있음을 보입니다.

- **Technical Details**: 제안된 접근 방식은 사전 훈련된 비전 트랜스포머로부터 마이크로 구조 이미지를 이용해 특성 벡터를 생성하는 것입니다. 이 과정은 마이크로 구조 이미지 수집, 이미지별 특성 추출, 다차원 피쳐 벡터 차원 축소, 그리고 특성과 마이크로 구조 사이의 관계를 포착하기 위한 경량 회귀 모델 훈련 단계로 구성되어 있습니다. 세 가지 고급 비전 트랜스포머 모델(CIP, DINOV2, SAM)의 성능을 비교하면서 이 방법의 효과를 검토하였습니다.

- **Performance Highlights**: 두 개의 사례 연구를 통해 이 ViT 기법의 유용성을 입증하였습니다. 첫 번째 연구에서는 시뮬레이션 데이터를 기반으로 3D 이차 미세 구조의 영률(Young's modulus)을 예측했으며, 두 번째 연구에서는 니켈 및 코발트 기반 슈퍼 합금의 비커 경도(Vicker’s hardness)를 실험 데이터로 예측하였습니다. 두 경우 모두, 비전 트랜스포머로부터 얻은 특성을 사용한 간단한 회귀 모델이 도메인 특화 CNN이나 다른 전통적인 방법에 비해 더 나은 성능을 보여 주었습니다.



### Towards Understanding Depth Perception in Foveated Rendering (https://arxiv.org/abs/2501.18635)
Comments:
          8 pages including references

- **What's New**: 본 연구에서는 퍼베이션 블러(peripheral blur)가 입체 깊이 인식(stereoscopic depth perception)에 미치는 영향을 처음으로 평가하고 있습니다. Foveated rendering 기법이 성능에 미치는 영향을 탐구하기 위해 심리물리학적 실험을 설계하였으며, 블러 수준에 따른 깊이 인식의 변화를 정량적으로 분석하였습니다. 이 분석 결과, 입체시력(stereoacuity)은 높은 수준의 주변 블러에 영향을 받지 않거나 심지어 개선된다는 사실을 밝혀냈습니다.

- **Technical Details**: 연구는 인간 시각 시스템의 공간 해상도(spatial resolution)의 비대칭성을 활용하여 foveated rendering 기법을 적용합니다. 인간의 시각 인식에 대한 기존 연구와 심리 물리학적 모델을 바탕으로, 주변의 블러가 깊이 인식에 미치는 영향을 분석했습니다. 이 연구는 블러 강도와 입체 깊이에 대한 상관 관계를 조사하기 위해 broadband stimulus를 사용하여 실험을 실시하였습니다.

- **Performance Highlights**: 연구 결과는 foveated rendering이 스테레오 깊이 인식에 영향을 미치지 않으며, 일반적인 foveated rendering 실습에서 사용되는 것보다 2배 더 강한 foveation에도 stereoacuity가 유지된다는 것을 보여주었습니다. 이러한 발견은 자원이 효율적인 렌더링 전략을 가능하게 하며, 공간 인식(spatial awareness)을 희생하지 않고도 깊이 표현을 개선할 수 있는 새로운 가능성을 제시합니다. 또한, 복잡한 자연 자극에 대한 검증 실험을 통해 이러한 결과가 현실적인 장면에서도 유효함을 확인하였습니다.



### Deformable Beta Splatting (https://arxiv.org/abs/2501.18630)
- **What's New**: Deformable Beta Splatting (DBS)라는 새로운 방법을 소개하여 3D Gaussian Splatting의 한계를 극복하고 있습니다. DBS는 Gaussian kernel 대신에 변형 가능한 Beta kernel을 사용하여 고해상도 기하학적 세부정보와 더 나은 메모리 효율성을 제공합니다. 또한, 색상 표현을 위해 Spherical Beta 함수를 도입하여 확산 및 반사 성분을 분리하고 더 정밀한 모델링을 가능하게 합니다.

- **Technical Details**: DBS의 핵심은 Beta kernel로, 이는 평면 표면과 날카로운 가장자리를 정확하게 표현하는 유연한 지원을 제공합니다. 색상 표현에서는 Spherical Beta (SB) 함수를 통해 적은 매개변수로도 높은 성능을 달성합니다. 또한, Markov Chain Monte Carlo (MCMC) 원리를 활용하여 단일 부분의 조정으로도 분포 보존을 보장하는 새로운 전략을 제시합니다.

- **Performance Highlights**: 실험 결과, DBS는 3DGS 기반 방법보다 1.5배 빠르며, 필요한 매개변수의 45%만 사용하여 시각적 품질에서 최첨단 NeRF 방법을 능가했습니다. 이로써 DBS는 실시간 방사광 필드 렌더링의 성능과 효율성을 입증하며, 변화하는 장면 복잡성을 처리하는 데 있어 현저한 개선을 보여줍니다.



### VLMaterial: Procedural Material Generation with Large Vision-Language Models (https://arxiv.org/abs/2501.18623)
- **What's New**: 이번 연구는 입력 이미지를 바탕으로 프로시저(algo) 재료 프로그램을 생성하는 대규모 비전-언어 모델(VLM)을 미세 조정하는 방법을 소개합니다. 프로시저 재료는 전통적인 이미지 기반 재료에 비해 더 직관적이고 정교한 수정이 가능합니다. 이 연구에서는 데이터 부족 문제를 해결하기 위해 공개된 프로시저 재료 데이터셋을 기여하고, 다른 사전 훈련된 언어 모델(LLM)을 사용하여 데이터 증강을 수행합니다.

- **Technical Details**: 연구에서는 Blender와 같은 3D 디자인 툴에서 사용되는 프로시저 그래프를 스탠다드 Python 프로그램으로 변환하여 VLM을 미세 조정합니다. 입력 이미지와 프로시저 재료 프로그램의 쌍을 활용하여 VLM을 훈련시키며, 데이터 증강 전략을 통해 데이터셋을 약 1.6K 예시에서 550K 예시로 확장합니다. 이는 고품질의 프로시저 재료 예측의 일관성을 높이는 결과를 가져옵니다.

- **Performance Highlights**: 세밀한 평가 결과 연구 방법이 기존 프로시저 재료 생성 방법들보다 우수한 성능을 보임을 입증하였습니다. 생성된 프로시저 재료는 3D 씬에 직접 적용 가능하며, 완전한 편집이 가능함을 보여줍니다. 또한, 연구자들은 Blender에서 사용할 수 있는 오픈소스 프로시저 재료 데이터셋을 제공하여 향후 연구를 촉진하고자 합니다.



### Three Laws of Statistical Linguistics Emerging in images (https://arxiv.org/abs/2501.18620)
- **What's New**: 이 연구에서 이미지를 텍스트로 간주하고, 각 이미지가 표현하는 단어의 통계적 언어학적 법칙을 검토하였습니다. 특히, Zipf’s law, Heap’s law, Benford’s law와 같은 세 가지 주요 통계 영역이 이미지 내에서 나타나는 것을 발견했습니다. 또한, Convolutional Neural Network인 VGG-19를 활용하여 이미지의 특징을 추출하고 이를 단어로 정의했습니다.

- **Technical Details**: 이 연구는 Gabor filter를 사용한 kernal을 바탕으로 이미지를 사전 처리하고, Intersection over Union (IoU) 메트릭을 활용하여 세부적인 주유 이미지 세그멘테이션을 수행합니다. VGG-19의 각 convolutional layer에 따른 문자 발생량을 통계적 법칙으로 구현하였으며, 각 법칙의 R-squared 값을 통해 결과의 유의성을 평가했습니다. 실험적으로는 이미지의 노이즈와 처리 변화가 각 법칙의 존재에 미치는 영향을 분석했습니다.

- **Performance Highlights**: Zipf’s law와 Heap’s law는 높은 R-squared 값을 유지하며 강력한 유효성을 보여주었지만, Benford’s law는 더 많은 변조에 민감하게 반응하여 성능이 떨어졌습니다. 이는 Benford’s law의 특징이 다른 두 법칙보다 더 쉽게 변화할 수 있음을 나타냅니다. 연구 결과를 통해 향후 dCNN 모델의 적용 가능성을 탐색하고, 이미지 설명과 자연어 처리 간의 연관성을 제고할 수 있는 기회를 제시합니다.



### FAAGC: Feature Augmentation on Adaptive Geodesic Curve Based on the shape space theory (https://arxiv.org/abs/2501.18619)
Comments:
          8pages, 3figures, submitted to IJCAI 2025

- **What's New**: 이 논문에서는 FAAGC(Feature Augmentation on Adaptive Geodesic Curve) 방법을 제안하여 제한된 데이터 환경에서도 모델의 정확도를 높이는 데 기여합니다. 기존의 데이터 증강 전략이 특정 데이터셋에 의존하거나 전문가의 지식에 의해 제한되는 경우가 많아 이 문제를 해결하고자 합니다. 본 연구는 형태 공간 이론을 기반으로 하여, 동일한 형태를 가진 객체들이 서로 연결된 대원 위에 존재한다는 전제를 두고, 각 클래스별로 적응적인 지오데식 곡선을 구성합니다.

- **Technical Details**: FAAGC 방법은 먼저 딥러닝 모델의 특징을 추출하고, 이를 사전 형태 공간(pre-shape space)으로 프로젝션합니다. 그런 다음, 각 클래스에 대해 지오데식 곡선을 생성하고, 이 곡선을 따라 샘플링하여 데이터를 증강합니다. 이렇게 생성된 데이터는 원본 샘플과 함께 분류기를 학습하는 데 사용되며, 이 과정에서 샘플링 파라미터도 함께 최적화됩니다.

- **Performance Highlights**: 본 연구에서 수행된 실험 결과, 제한된 데이터 조건 하에서도 FAAGC 방법이 분류 정확도를 상당히 향상시킨다는 것을 보여주었습니다. 다양한 이미지 데이터셋과 주류 딥러닝 백본을 통해 검증하였으며, 결과적으로 제안된 방법이 전통적인 이미지 기반 증강 기법과 독립적이면서도 추가적인 성능 향상을 이끌어낼 수 있음을 입증하였습니다.



### Vision Aided Channel Prediction for Vehicular Communications: A Case Study of Received Power Prediction Using RGB Images (https://arxiv.org/abs/2501.18618)
Comments:
          12 pages, 11 figures, submitted to IEEE Transactions on Vehicular Technology

- **What's New**: 본 논문은 차세대 6G 통신에서 인공지능(AI)과 비전 기반 기술의 통합을 통해 정밀한 채널 예측을 위한 두 단계 깊이 학습 모델을 제안합니다. 기존의 방법들이 환경 정보를 효과적으로 활용하지 못하는 문제를 다루며, RGB 이미지만을 사용하여 수신 전력을 예측하는 기술을 개발하고 있습니다.

- **Technical Details**: 제안된 모델은 두 단계로 구성되어, 첫 번째 단계에서는 RGB 카메라로 캡처한 원본 이미지를 통해 환경 정보를 추출하고, 두 번째 단계에서 처리된 이미지를 토대로 수신 전력을 예측합니다. 세 가지 전형적인 컴퓨터 비전(CV) 기법, 즉 객체 탐지, 인스턴스 분할, 이진 마스크가 환경 정보 추출에 활용되며, YOLOv8과 ResNet이 각 단계에서 사용됩니다.

- **Performance Highlights**: 제안된 모델은 다양한 실험을 통해 실용성과 신뢰성, 그리고 일반화 능력이 우수함을 입증하였습니다. 실험 결과, 모델은 처리된 이미지의 종류나 간섭 제거 여부에 따라 예측 성능이 어떻게 달라지는지를 분석해, 향후 6G 차량 통신 시스템의 지능형 배치 및 응용을 위한 유용한 솔루션을 제공합니다.



### STAMP: Scalable Task And Model-agnostic Collaborative Perception (https://arxiv.org/abs/2501.18616)
Comments:
          Paper is accepted by ICLR 2025

- **What's New**: STAMP는 이종 에이전트를 위한 새로운 협업 인식 프레임워크로, 경량 어댑터-리버터 쌍을 사용하여 각 에이전트의 Bird's Eye View (BEV) 특징을 통합된 프로토콜 BEV 특징 도메인으로 변환합니다. 이 방법은 다양한 모델과 작업에 대해 태스크-모델-무시(task- and model-agnostic) 방식으로 작동하여, 모델 재학습 없이도 통합이 가능하다는 장점을 지닙니다. 또한 이 프레임워크는 큰 디스크 메모리나 컴퓨팅 오버헤드 없이도 여러 이종 에이전트가 협력할 수 있도록 지원합니다.

- **Technical Details**: STAMP의 협업 인식 파이프라인은 각 에이전트의 BEV 특징을 통합하고, 이를 다른 에이전트와 공유하여 지역 도메인으로 다시 매핑하는 방식인 협업 특징 정렬(cFA)을 활용합니다. 실험을 통해, STAMP는 이종 에이전트 수가 증가할수록 자원 성장률을 크게 낮추면서도 유사 이상으로 높은 정확성을 달성할 수 있음을 입증하였습니다. 평균적으로 추가 에이전트당 2.36 GPU 시간의 훈련 시간을 요구하며, 이는 기존 방법에 비해 7.2배 절약된 수치입니다.

- **Performance Highlights**: STAMP의 성능은 두 가지 데이터셋인 OPV2V와 V2V4Real을 사용하여 평가되었으며, 기존의 이종 협업 인식 모델과 비교하여 결과가 우수하거나 동등함을 보였습니다. 특히 STAMP는 이종 모델 간 협업을 체계적으로 지원하는 독특한 능력을 지녀, 자율 주행 분야에서 새로운 기준을 제시합니다. 이는 다른 방법이 작동할 수 없는 상황에서도 분명한 성능 향상을 보여주며, 향후 Level 5 자율주행 시스템 개발에 기여할 것으로 기대됩니다.



### Multi-Frame Blind Manifold Deconvolution for Rotating Synthetic Aperture Imaging (https://arxiv.org/abs/2501.19386)
Comments:
          39 pages, 9 figures

- **What's New**: 이번 논문에서는 로테이팅 합성 개구(RSA; Rotating Synthetic Aperture) 이미지를 처리하기 위한 혁신적인 방법을 제안합니다. RSA 시스템은 다양한 회전 각에서 이미지를 캡처하여, 잠재적인 선명한 이미지를 재구성하기 위해 블라인드 컨볼루션(Blind Convolution) 기술을 활용합니다. 특히, 본 연구는 저차원 다양체(low-dimensional manifold) 구조를 탐색하여 고차원 공간에서의 이미지를 최적화하는 새로운 기술을 소개합니다.

- **Technical Details**: 제안된 방법은 다단계 블라인드 맨포드 컨볼루션 모델로 구성되며, 각 이미지의 편향을 줄이고 subsequent manifold fitting의 성능을 향상시키기 위해 강화된 알고리즘이 개발되었습니다. 이미지의 품질을 개선하는 3단계 절차는 (1) L1-노름(penalty)을 사용해 선명한 이미지의 희소성을 제고하며 이미지를 디컨볼루션(deconvolution)하는 것, (2) manifold fitting을 통한 이미지 해상도 개선, (3) 향상된 이미지를 합성하여 고해상도 장면 이미지를 생성하는 것입니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, 제안된 맨포드 기반 디컨볼루션 방법은 기존의 디컨볼루션 알고리즘과 비교하여 픽셀 강도 추정 및 구조적 세부정보를 보존하는데 있어 더 선명한 이미지 결과를 도출했습니다. 이 방법은 기존의 원본 이미지를 직접 사용할 때보다 더 뛰어난 성능을 발휘하여, RSA 이미지에서 더 높은 품질의 이미지를 효율적으로 생성하도록 설계되었습니다.



### Using gradient of Lagrangian function to compute efficient channels for the ideal observer (https://arxiv.org/abs/2501.19381)
Comments:
          SPIE Medical Imaging 2025

- **What's New**: 이 연구는 병리 이미징 시스템의 최적화를 위한 새로운 효율적인 채널 생성 방법을 제안하고 있습니다. 제안된 방법은 Hotelling observer (HO)를 학습하기 위한 Lagrangian 기반 손실 함수의 기울기를 사용하여 생성된 Lagrangian-gradient (L-grad) 채널을 활용합니다. L-grad 채널은 기존의 PLS 채널과 비교했을 때 신호 탐지 성능과 계산 시간을 현저히 개선할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구는 binary signal detection task에서 관측자가 이미지 데이터 𝑔를 신호 부재 가설 (H0) 또는 신호 존재 가설 (H1)로 분류하는 과제를 고려합니다. Bayesian Ideal observer (IO)는 전체 작업별 정보를 활용하여 신호 탐지 작업의 상한 성능을 설정합니다. 하지만 IO의 테스트 통계는 일반적으로 비선형 함수로, 쉽게 계산할 수 없습니다. HO는 이러한 IO를 근사하는 이상적인 선형 관찰자로 사용됩니다.

- **Performance Highlights**: 컴퓨터 시뮬레이션 연구 결과, L-grad 채널을 사용한 채널화된 HO (CHO)는 PLS 채널을 사용할 때보다 신호 탐지 성능에서 유의미한 향상을 보였습니다. 또한 제안된 L-grad 방법은 PLS 방법에 비해 계산 시간을 상당히 줄일 수 있음을 입증하였습니다. 이 두 가지 결과는 의료 이미징 시스템 평가의 새로운 가능성을 보여줍니다.



### Do Large Multimodal Models Solve Caption Generation for Scientific Figures? Lessons Learned from SCICAP Challenge 2023 (https://arxiv.org/abs/2501.19353)
Comments:
          Accepted to TACL 2025

- **What's New**: SCICAP 데이터셋이 2021년에 출시된 이후, 연구자들은 과학적 도표에 대한 캡션 생성에서 상당한 발전을 이루었다. 2023년 첫 SCICAP 챌린지가 열리면서 전 세계의 팀들이 확장된 SCICAP 데이터셋을 사용해 다양한 유형의 도표에 대한 캡션 생성 모델을 개발하도록 초대하였다. 연구 결과, 전문 편집자들은 GPT-4V가 생성한 캡션을 모든 다른 모델과 저자들의 원본 캡션보다 선호했다.

- **Technical Details**: 이 논문은 2023년 SCICAP 챌린지에 대한 개요와 데이터, 과정, 우승 팀 및 모델에 대한 상세한 정보를 제공한다. 구체적으로, 8개 도메인과 5개 도표 유형의 476,389개의 단일 패널 도표를 포함하는 확장된 SCICAP 데이터셋을 바탕으로 팀들은 캡션 생성 모델을 개발했다. 자동 및 인간 평가를 통해 GPT-4V와 다른 오픈 LMM들(대형 다중모달 모델)의 성능을 비교하였다.

- **Performance Highlights**: 인간 평가 결과, 기술적 학술 작문에 익숙한 세 명의 전문 편집자는 GPT-4V가 생성한 캡션을 모든 다른 모델의 결과보다 명확히 선호하였다. 이러한 주요 발견에 따라, 우리는 편집자들이 GPT-4V를 선호하는 이유를 조사하였다. 우리의 결론은 현재 LMMs가 과학적 도표 캡션 생성 문제를 완전히 해결하지는 못했지만, 많은 개선이 이루어졌음을 보여주었다.



### Pathological MRI Segmentation by Synthetic Pathological Data Generation in Fetuses and Neonates (https://arxiv.org/abs/2501.19338)
Comments:
          30 pages, 4 figures, 5 tables

- **What's New**: 이번 연구에서는 Fetal&Neonatal-DDPM이라는 새로운 diffusion model framework를 소개하여, semantic label images를 기반으로 고품질의 합성 병리적(fetal and neonatal) MRI 이미지를 생성하는 방법을 제안합니다.

- **Technical Details**: 연구는 건강한 label 이미지를 형태학적(morphological) 변경을 통해 수정하여, 심실 확장증(ventriculomegaly), 소뇌 및 교뇌 발달부전(cerebellar and pontocerebellar hypoplasia), 그리고 소두증(microcephaly)과 같은 질환을 시뮬레이션할 수 있는 데이터를 생성합니다. 이 방법을 통해 생성된 합성 MRIs는 실제 MRI 이미지와 비교했을 때 품질과 진단 가치에서 유의미한(p < 0.05) 향상을 보였습니다.

- **Performance Highlights**: 합성된 병리 데이터는 nnUNet의 segmentation 성능을 크게 향상시켰으며, 특히 심각한 심실 확장증 사례에서 두드러진 성과를 보였습니다. 심실(segmentation) 분야의 Dice 점수가 0.9253로, 기존의 0.7317과 비교하여 상당한 개선이 이루어졌습니다.



### Homogeneity Bias as Differential Sampling Uncertainty in Language Models (https://arxiv.org/abs/2501.19337)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)과 비전-언어 모델(VLM)이 특정 소외 집단을 더 균일하게 표현하는 이질성 편향(homogeneity bias)의 원인을 탐구하고 있습니다. 특히, 텍스트 생성 시 소외 집단에 대한 토큰 샘플링의 확률 분포가 더 결정론적이라는 가설을 세웠습니다. 이 연구는 이전의 결과와 차별화된 접근 방식을 제시하며, 다양한 모델 간의 이질성 편향 기전의 차이를 규명하고자 하였습니다.

- **Technical Details**: 연구에서는 엔트로피(entropy), 당황도(perplexity), 구별 확률(probability of differentiation) 등 세 가지 확률 분포의 불확실성을 측정하는 방법을 사용하였습니다. 이들은 AI 연구에서 분포적 불확실성을 평가하는 데 활용되어 왔습니다. 실험에는 GAN 생성 얼굴 데이터베이스(GANFD)를 사용하여 인종 및 성별에 따른 얼굴 자극을 선택하고, 모델에 대한 프롬프트 설계 및 확률 분포 분석 방법론을 구체적으로 설명하였습니다.

- **Performance Highlights**: GPT-4 Turbo 및 Llama-3.2 모델에서 마진 그룹에 대한 텍스트 생성 과정에서 비교적 낮은 불확실성 지표가 관찰되었습니다. 연구 결과는 특정 모델에서 이질성 편향이 확률 분포에 암호화되어 있음을 보여주지만, 모든 VLM에 일반화될 수 없음을 제시합니다. 이는 AI 모델 내에서 이질성 편향의 존재가 여러 기전의 영향을 받을 수 있음을 시사합니다.



### Capturing Temporal Dynamics in Large-Scale Canopy Tree Height Estimation (https://arxiv.org/abs/2501.19328)
Comments:
          9 pages main paper, 5 pages references and appendix, 8 figures, 5 tables

- **What's New**: 이 논문은 기후 변화에 대한 국가 정책 수립을 지원하기 위해 포괄적이고 정확한 산림 생태계 모니터링이 필요함을 강조합니다. 기존의 연구와 달리, 유럽 대륙의 2019년부터 2022년까지 10m 해상도의 임시 캐노피 높이 지도를 처음으로 생성하여, 계절적 변화를 명확히 추적할 수 있는 모델을 제안합니다. 이 모델은 Sentinel-2 위성 데이터를 활용하여 나무 캐노피 높이를 예측하며, GEDI LiDAR 데이터를 사실 기반으로 사용하여 높은 정확성을 자랑합니다.

- **Technical Details**: 논문에서 제시하는 알고리즘은 Sentinel-2 위성 이미지 데이터를 사용하여 나무 캐노피 높이를 예측하는 맞춤형 딥러닝 접근법에 기반합니다. 이 모델은 기존의 단일 이미지 처리가 아닌 연속된 12개월의 위성 데이터 시리즈를 활용하여 계절적 패턴을 포착하고, 공간적 정확성을 향상시키는 방법을 적용합니다. 또한, 모델의 파라미터와 전체 파이프라인은 GitHub에 공개되어 있어 연구자들이 이를 재현하고 확장하는 데에 용이합니다.

- **Performance Highlights**: 2020년 동안의 유럽 캐노피 높이 지도는 이전 연구들보다 구체적이고 정밀한 측정을 제공합니다. 논문에 따르면 이 연구는 2019-2022년 동안의 산림 성장과 감소를 일관되게 탐지할 수 있는 능력을 보여주며, 정확한 대규모 산림 구조 평가를 가능하게 합니다. 이와 같은 정밀도는 기후 변화 완화 노력에 기여하고, 지속 가능한 산림 관리 및 생태계 보전을 위한 정보 제공의 중요성을 강조합니다.



### Neuro-LIFT: A Neuromorphic, LLM-based Interactive Framework for Autonomous Drone FlighT at the Edg (https://arxiv.org/abs/2501.19259)
- **What's New**: 이 논문은 Parrot Bebop2 쿼드콥터를 위한 neuromorphic(뉴로모픽) 내비게이션 프레임워크인 Neuro-LIFT를 제시합니다. Neuro-LIFT는 자연어 처리를 위한 LLM(대형 언어 모델)을 활용하여 인간이 제공한 음성을 고수준 계획 명령으로 변환한 후, 이를 이벤트 기반 Neuromorphic vision(뉴로모픽 비전) 및 물리 기반 계획을 통해 자율적으로 실행합니다. 이 연구는 기존 LLM과 neuromorphic 방식을 성공적으로 통합하여 인간의 명령에 실시간으로 반응하는 자율 비행을 실현한 최초의 사례로 보입니다.

- **Technical Details**: Neuro-LIFT는 이벤트 기반 카메라와 스파이킹 신경망(SNN)을 결합하여 저전력 및 저지연 내비게이션을 가능하게 합니다. 이벤트 카메라는 광 강도의 변화만 기록하여 데이터의 대역폭을 크게 줄이고, 이를 통해 고해상도의 시간적 정보를 제공합니다. LLM은 인간의 자연어 명령을 이해하고 이를 효과적으로 드론 조작 작업으로 전환하는 데 도움을 줍니다.

- **Performance Highlights**: Neuro-LIFT는 동적 환경에서 안전하게 탐색하고 장애물을 피하며, 실시간으로 제공된 인간의 지시에 적응하는 능력을 보여줍니다. 이 프레임워크는 기존의 전통적인 내비게이션 알고리즘과 비교해 에너지 효율성을 극대화하고, 결정 과정의 지연을 최소화하는 데 기여하며, 실시간 의사결정이 중요한 응용 분야에서 용이하게 적용할 수 있습니다.



### Single cell resolution 3D imaging and segmentation within intact live tissues (https://arxiv.org/abs/2501.19203)
- **What's New**: 이번 연구에서는 다양한 구조의 상피세포(epithelial cells)가 높은 해상도의 심층 이미징(deep imaging)과 컴퓨터 기반 기법을 통해 3D 구조적 특징을 정확하게 정량화하는 방법을 소개합니다. 특히, 생체 조직에서 형광으로 표지된 개별 세포를 3D로 정확히 계량하기 위한 프로토콜(protocol)을 단계별로 제시합니다. 이 연구는 Drosophila wing discs의 3D 이미징에서 발생한 문제를 해결하기 위한 경험을 공유하며, 다양한 이미징 모달리티 선택 및 설정에 대한 고려사항도 포함하고 있습니다.

- **Technical Details**: 세포를 정확하게 분할(segmentation)하기 위한 딥러닝 보조 기술을 포함한 이미징 및 샘플 준비 방법론이 자세히 설명됩니다. 이 프로토콜은 세포막 표지를 통한 세포 윤곽(segmentation)만을 목표로 하지만, 복잡한 3D 분석이 요구되는 다양한 샘플에 적용 가능하다는 것이 특징입니다. 또한, 프로토콜 복제를 지원하기 위한 컴퓨터 파이프라인(computational pipeline)과 맞춤형 코드(custom code)가 포함되어 있어 연구자들이 실험을 반복하기 쉽게 돕습니다.

- **Performance Highlights**: 제공된 프로토콜은 생체 조직에서 세포 특성을 정량화하기 위한 정교한 접근 방식을 제시하여, 상피세포의 다양한 특성과 구조를 정확히 분석할 수 있게 합니다. 이 연구는 고해상도 이미징 기술을 통한 세포 구조 연구에 기여하며, 앞으로 다른 조직 연구에도 활용될 수 있을 것으로 기대됩니다. 세포 분석법의 발전을 통해 다양한 생물학적 현상에 대한 깊은 통찰을 제공할 것입니다.



### Augmented Intelligence for Multimodal Virtual Biopsy in Breast Cancer Using Generative Artificial Intelligenc (https://arxiv.org/abs/2501.19176)
- **What's New**: 본 연구에서는 Full-Field Digital Mammography (FFDM)와 Contrast-Enhanced Spectral Mammography (CESM)를 활용한 멀티모달 심층 학습 접근법을 제안합니다. FFDM과 CESM을 통합하여 가상 생검(virtual biopsy)을 수행함으로써 종양을 악성으로 분류하는 데 도움을 줍니다. CESM 데이터를 누락할 경우, 생성적 인공지능을 활용하여 FFDM 스캔으로부터 CESM 이미지를 보간(impute)하는 방법을 사용합니다. 이는 FFDM 단독 사용에 비해 더 높은 성능을 보여주었습니다.

- **Technical Details**: 연구는 FFDM 및 CESM 영상을 크레니오카우달(craniocaudal) 및 메디오레터럴 오블리크(mediolateral oblique) 뷰에서 레지스터(registra)하여, 높은 진단 정확도를 위해 머신러닝 기법을 이용하는 과정을 포함합니다. CESM 이미지를 보간하기 위한 생성적 인공지능 기술을 활용하여, FFDM만으로 이루어진 경우에 비해 더 과학적이고 정확한 결과를 도출할 수 있습니다. 이 접근 방식은 새로운 데이터셋을 공개하여 연구 커뮤니티에 기여하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, CESM 모듈을 통합함으로써 가상 생검의 성능이 크게 향상되었음을 보여주었습니다. 실제 CESM 데이터가 누락된 경우, 합성 CESM 이미지를 활용하는 것이 FFDM 단독으로 사용하는 것보다 효과적이라는 것을 입증하였습니다. 특히, FFDM과 CESM을 결합한 멀티모달 구성을 사용했을 때 성능이 더욱 두드러졌습니다. 따라서, 본 연구는 진단 정확도를 개선하는 데 기여하며 임상 의사들에게 유용한 도구를 제공합니다.



### Improving Multi-Label Contrastive Learning by Leveraging Label Distribution (https://arxiv.org/abs/2501.19145)
- **What's New**: 이번 연구에서는 다중 라벨 학습에서 대조 학습(contrastive learning)을 통해 더 나은 표현을 학습하는 방안을 제안합니다. 기존 방법들은 라벨 간의 중첩을 기반으로 긍정 및 부정 샘플을 선택하였으나, 복잡한 선택 과정과 다양한 라벨의 중요도를 무시하는 문제에 직면해 있었습니다. 이를 해결하기 위해, 우리는 라벨 분포(label distribution)를 도입하여 다중 라벨 대조 학습을 개선하는 새로운 방법 MulSupConLDLD{}_{LD}를 제안합니다.

- **Technical Details**: 우리는 긍정 샘플 선택 시 ANY 전략을 채택하여 라벨이 교차하는지를 기준으로 삼습니다. 또한, 라벨 간의 관계를 모델링하기 위해 로지컬 라벨에서 라벨 분포를 복구하는 두 가지 방법(Radial Basis Function (RBF) 및 대조 손실(contrastive loss)을 기반으로 한 방법)을 도입했습니다. 이러한 접근 방식은 다중 라벨 데이터셋에서 모델의 일반화 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 우리의 방법은 아홉 개의 널리 사용되는 다중 라벨 데이터셋에서 평가되었으며, 여섯 가지 평가 지표에서 기존 최첨단 방법들을 초월하는 성능을 보였습니다. 이를 통해, MulSupConLDLD{}_{LD}는 더욱 차별적인 특징 표현을 학습함과 동시에 라벨 간의 의존성을 효과적으로 포착할 수 있음을 입증했습니다.



### Imitation Game for Adversarial Disillusion with Multimodal Generative Chain-of-Thought Role-Play (https://arxiv.org/abs/2501.19143)
- **What's New**: 이번 연구에서는 인공 지능의 신뢰성을 위협하는 적대적 환상(adversarial illusions)에 대한 새로운 방어 체계를 제안합니다. 제안된 디스일루전(disillusion) 패러다임은 모방 게임(imitation game) 개념에 기반하여, 다양한 공격 형태에 대해 통합된 방어를 제공하는 것을 목표로 하고 있습니다. 연구에서는 다중 모달 생성 에이전트(multimodal generative agent)가 중심이 되어 체인 오브 사고(reasoning)를 통해 샘플의 본질을 내부화하고 재구성하는 과정을 설명합니다.

- **Technical Details**: 이 논문에서는 두 가지 형태의 적대적 공격인 추론 시 유도된 환상(deductive illusion)과 학습 시 조건화된 환상(inductive illusion)을 다룹니다. 각각의 공격 유형은 신뢰성을 저해하는 방식으로, 베이너리 샘플이나 학습 데이터를 변형하여 모델의 결정을 왜곡할 수 있습니다. 제안하는 디스일루전 패러다임은 전통적인 샘플 복원을 넘어서 줄거리에 대한 이해를 통해 새로운 솔루션을 탐색합니다.

- **Performance Highlights**: 실험 시뮬레이션을 통해 제안된 방법론이 다양한 적대적 공격 시나리오에서 어떻게 작동하는지를 평가합니다. 특히, OpenAI의 ChatGPT를 활용하여 실험을 진행하며, 공격 상황에서 이 방법이 효과적으로 작용하는 것을 확인합니다. 이 연구는 향후 AI 신뢰성을 향상시키기 위한 방향성을 제시합니다.



### The Role of Graph-based MIL and Interventional Training in the Generalization of WSI Classifiers (https://arxiv.org/abs/2501.19048)
Comments:
          Presented at ML4H 2024 - Findings Track

- **What's New**: 이 논문은 Whole Slide Imaging (WSI) 분석을 위한 새로운 모형인 Graph-based Multiple Instance Learning with Interventional Training (GMIL-IT)을 도입합니다. 기존의 MIL (Multiple Instance Learning) 방법들은 패치 간의 공간적 관계를 간과한 반면, GMIL-IT는 이러한 관계를 그래프를 통해 포착하고 이를 이용하여 모델의 일반화 성능을 개선합니다. 이 방법은 특히 도메인 전환 (domain shift) 분석을 통해 그 효과를 입증합니다.

- **Technical Details**: 논문에서는 WSI를 처리하기 위한 다양한 그래프 구성 방법과 MIL 모델을 비교 분석합니다. 각 WSI는 비중첩 패치로 분할되어 그래프의 노드로 구성되며, ResNet-50를 사용하는 특성 추출기를 통해 1024 차원의 피쳐 임베딩을 생성합니다. GMIL-IT는 이러한 그래프 구조를 활용하여 패치 간의 공간 정보를 모델링하고, 반대문 조정을 통해 보다 강력한 분류 성능을 달성합니다.

- **Performance Highlights**: 연구 결과, 그래프 기반 모델이 개입 훈련이 강화된 모델들보다 더 뛰어난 성과를 보인다는 것을 확인했습니다. 또한, 도메인 전환 상황에서도 높은 일반화 성능을 유지하며, 그래프 구조가 어떻게 모델의 견고성을 높이는지를 보여주었습니다. 이로써 WSI 분석에서의 진단 정확도를 증가시키는 데 기여할 수 있을 것으로 기대됩니다.



### Understanding Model Calibration -- A gentle introduction and visual exploration of calibration and the expected calibration error (ECE) (https://arxiv.org/abs/2501.19047)
- **What's New**: 이번 논문에서는 모델의 신뢰성을 높이기 위한 캘리브레이션(calibration)의 중요성을 설명하고, 가장 일반적으로 사용되는 정의 및 평가 지표를 살펴봅니다. 특히, Confidence Calibration과 관련된 여러 새로운 평가 측정 기준의 필요성을 강조합니다. 이를 통해, 기존 ECE와 같은 측정법의 한계에 대한 논의도 진행합니다.

- **Technical Details**: 캘리브레이션(caliabration)은 모델의 추정 확률이 실제 확률과 일치하도록 보장하는 과정입니다. 예를 들어, 날씨 예측 모델이 70%의 강수 확률을 제시할 때, 실제로 70%의 날들이 비 오는 것이어야 하며, 모델은 K개의 클래스에 대해 확률 벡터를 반환합니다. 이를 통해 Confidence Calibration의 정의와 프로세스를 설명하고, 구체적인 수식과 함께 이를 이해할 수 있도록 합니다.

- **Performance Highlights**:  Expected Calibration Error(ECE)는 모델의 예상 확률과 관측된 확률 간의 차이를 평가하는 주요 지표로 활용됩니다. 이 방법은 데이터 포인트를 여러 개의 빈(bin)으로 나누고, 각 빈에서의 평균 정확도와 평균 신뢰도의 절대 차이를 계산하여 ECE를 구합니다. 모델이 특정 빈에서 얼마나 정확한지를 측정함으로써 캘리브레이션의 성능을 평가할 수 있습니다.



### Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them (https://arxiv.org/abs/2501.18950)
- **What's New**: 이 연구에서는 diffusion 모델에서 유해 내용을 생성할 위험을 줄이기 위한 개념 지우기(Concept Erasure) 기법에 대해 다루고 있습니다. 기존의 고정 목표 스트래티지(fixed-target strategy)는 특정 개념을 중립 개념이나 빈 텍스트 프롬프트로 매핑하는 방식으로, 이는 비효율적이라는 것을 보여줍니다. 연구팀은 개념 공간을 그래프로 모델링하여 하나의 개념을 지우는 것이 다른 개념에 미치는 영향을 분석하고, Adaptive Guided Erasure (AGE)라는 새로운 기법을 제안하여 최적의 목표 개념을 동적으로 선택함으로써 의도치 않은 부작용을 최소화합니다.

- **Technical Details**: AGE 방법은 지우고자 하는 각 개념에 맞춰 최적의 목표 개념을 선택하는 방식으로 설계되었습니다. 기존의 유사한 기법들과의 비교를 통해 AGE가 관련 없는 개념을 잘 보존하면서도 강력한 지우기 성능을 유지하는 것을 입증했습니다. 이 방법은 반복적인 지식 요구가 없는 전이 가능한 방식으로, 모델 매개변수를 수정하여 전이 학습을 통해 유해 개념을 효과적으로 제거할 수 있는 기법입니다.

- **Performance Highlights**: AGE는 최신의 지우기 기법들과 비교했을 때, 무관한 개념을 보존하는 능력이 현저히 우수한 결과를 보여주었습니다. 실험 결과, AGE 방법은 특정 개념 제거와 관련된 원치 않는 부작용을 최소화하면서 유해 개념을 효과적으로 제거하는 데 성공했습니다. 추가적으로, 연구팀은 코드를 공개하여 이 기법의 구현과 활용이 가능하도록 하였습니다.



### Adaptive Prompt: Unlocking the Power of Visual Prompt Tuning (https://arxiv.org/abs/2501.18936)
Comments:
          55 pages, 10 figures, 18 tables. arXiv admin note: text overlap with arXiv:2410.02200

- **What's New**: 최근 Visual Prompt Tuning (VPT)은 사전 훈련된 비전 모델을 다운스트림 작업에 효과적으로 적응시키는 강력한 방법으로 주목받고 있습니다. 하지만 VPT의 이론적 이해가 부족하다는 문제를 인식하고, 이를 해결하기 위해 Visual Adaptive Prompt Tuning (VAPT)라는 새로운 프롬프트 튜닝 방법론을 제안했습니다. 이 접근법은 입력의 적응형 기능으로 프롬프트를 재정의하여 최적의 샘플 효율성을 달성하는 것을 목표로 합니다.

- **Technical Details**: VAPT의 설계는 두 가지 주요 구성 요소로 이루어져 있습니다: 토큰 단위의 프로젝터(token-wise projectors)와 공유 특성 프로젝터(shared feature projector). 이는 입력의 글로벌 정보를 활용하여 적응형 프롬프트 토큰을 생성하며, 이론적 분석을 통해 프롬프트 추정에 대한 최적 샘플 효율성을 보장합니다. VAPT는 입력에 따라 변동할 수 있는 적응형 프롬프트를 도입하여 VPT의 한계를 극복하고 더 나은 성능을 발휘하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 VAPT는 Stanford Dogs 데이터셋에서 1%의 데이터만으로도 VPT보다 훨씬 높은 60.1%의 정확도를 달성했습니다. 또한 VAPT는 동일한 작업에 대해 다른 PEFT 방법들 중에서도 최고 성능을 보이며, 적은 수의 파라미터를 사용하면서도 VPT를 일관되게 초과하는 성능을 나타냅니다. 이러한 결과는 VAPT의 단순하면서도 우아한 설계가 효과적이라는 것을 강조하며, 향후 연구에 대한 가능성을 제시합니다.



### Full-scale Representation Guided Network for Retinal Vessel Segmentation (https://arxiv.org/abs/2501.18921)
Comments:
          10 pages, 7 figures

- **What's New**: 본 논문에서는 패싯(FSG-Net)이라는 새로운 네트워크 구조를 제안하며, 이는 최신의 합성곱 블록을 사용하는 기능 표현 네트워크를 통해 전체적인 정보를 추출하고, 가이드 합성곱 블록을 통해 이 정보를 정제하는 구조입니다. 가이드 합성곱 블록에는 특성이 강조된 어텐션 필터가 도입되어 성능 향상을 가져오며, 가이드 필터는 각 단계에서 전체 조화를 이루는 피쳐를 활용하여 segmentation 네트워크의 성능을 향상시킵니다. 또한, 본 연구는 U-Net의 다양한 변형을 사용하여 확장성과 비교할 수 있는 가능성을 후속 연구에 보여줍니다.

- **Technical Details**: FSG-Net은 U-Net 아키텍처를 기반으로 하며, 기능 표현 네트워크에 가이드 합성곱 블록을 추가해 구조를 변화시킵니다. 본 연구는 가이드 필터가 이미지 처리에 있어 로컬 선형 관계를 평균화하여 엣지 감지 성능을 향상시킬 수 있다는 전제 아래 진행되었습니다. 모델의 키 영역은 피쳐 병합 과정으로, 세 개 단계에서 추출된 피쳐들을 통합하여 다음 단계로 전달하는 방식입니다.

- **Performance Highlights**: 실험 결과, 제안된 FSG-Net은 공개 데이터 세트에서 현재의 최첨단 모델들과 경쟁력 있는 결과를 보였습니다. Ablation 연구를 통해, 제안된 모델이 더 작은 매개변수를 가지고도 효과적으로 작동함을 확인했습니다. 마지막으로, 이 모델을 얼굴 주름 세분화에 적용하여 다른 도메인에서도 확장 가능성이 있음을 나타냈습니다.



### Self-Supervised Learning Using Nonlinear Dependenc (https://arxiv.org/abs/2501.18875)
- **What's New**: 이번 연구에서는 Correlation-Dependence Self-Supervised Learning (CDSSL)라는 새로운 프레임워크를 소개합니다. CDSSL은 기존의 자기 지도 학습(SSL) 방법론을 통합하고 확장하는 방식으로, 선형 상관관계와 비선형 의존성을 동시에 고려합니다. 이를 통해 샘플 간의 관계와 피처 간의 상호작용을 효과적으로 포착하여 표현 학습을 향상 시킬 수 있습니다.

- **Technical Details**: CDSSL의 핵심 구성 요소는 Hilbert-Schmidt Independence Criterion (HSIC)을 활용하여 비선형 의존성을 측정하는 점입니다. 이 방법으로 데이터의 복잡한 관계를 효과적으로 모델링할 수 있으며, 이를 위해 Reproducing Kernel Hilbert Space (RKHS)를 사용합니다. CDSSL은 총 8개의 손실 함수를 도입하여 다양한 상관관계와 의존성 측면을 포괄적으로 다룹니다.

- **Performance Highlights**: 다양한 벤치마크 테스트를 통한 실험 결과, CDSSL은 표현의 질을 크게 향상시켰습니다. 기존 SSL 방법들, 특히 VICReg과 Barlow Twins와 같은 기술들과 비교할 때, CDSSL은 더 높은 다양성과 분리(disentanglement)를 보여주며, 보다 정교한 표현 학습이 가능함을 증명했습니다.



### Pitfalls of defacing whole-head MRI: re-identification risk with diffusion models and compromised research potentia (https://arxiv.org/abs/2501.18834)
- **What's New**: 이번 연구에서는 얼굴이 변경된 머리 MRI 데이터에서 얼굴을 복원하는 refacing 파이프라인을 개발하였습니다. 이 과정에서 cascaded diffusion probabilistic models (DPMs)를 사용했습니다. 연구는 180명의 피험자로부터 학습된 DPM 모델을 활용하여 484명의 보이지 않는 피험자의 이미지를 평가했습니다.

- **Technical Details**: DPMs는 얼굴 정보가 삭제된 MRI 이미지를 원본 얼굴 이미지로 복원하기 위해 훈련되었습니다. 이 과정에서 각 데이터 세트가 어떻게 기능하는지 비교하기 위해 defacing된 이미지와 원본 이미지의 방사선 밀도(Radiodensity)를 예측하는 방식으로, facial voxels의 정보가 실제로 연구에 유용한지를 검토하였습니다.

- **Performance Highlights**: 결과적으로 DPMs는 원본 얼굴과 비슷한 고해상도의 얼굴 이미지를 생성할 수 있음을 보여주었으며, 원본 얼굴과의 표면 거리도 통계적으로 유의미하게 작았습니다. 그러나 defacing된 이미지를 사용했을 경우 원본 이미지에 비해 Skeletal muscle radiodensity에 대한 예측 성능이 크게 낮아지는 결과를 얻었습니다. 이는 defacing이 개인정보를 보호하는 데 실패할 뿐 아니라, 가치 있는 정보도 제거할 가능성이 있음을 시사합니다.



### An Adversarial Approach to Register Extreme Resolution Tissue Cleared 3D Brain Images (https://arxiv.org/abs/2501.18815)
- **What's New**: 본 논문에서는 세포 활동과 동역학을 명확히 볼 수 있는 고해상도 이미지를 등록할 수 있는 Generative Patch 기반 3D 이미지 등록 모델인 InvGAN을 개발했습니다. 기존의 이미지 등록 방법이 해상도가 높은 이미지를 등록하는 데 한계를 보이는 반면, InvGAN은 tissue clearing 과정을 통해 얻은 데이터를 효과적으로 처리할 수 있습니다. 이 모델은 두 개의 변형 필드를 동시에 생성하면서 이미지의 등록 품질을 비교하는 두 개의 판별기 네트워크를 사용합니다.

- **Technical Details**: 우리의 InvGAN 모델은 tissue clearing과 Light-sheet fluorescence microscopy (LSFM)로 얻은 이미지를 등록하는 데 특화되어 있습니다. 특히, 이 모델은 긴 계산 시간과 대규모의 컴퓨팅 자원을 요구하지 않고, 적은 데이터로 학습할 수 있도록 설계되었습니다. 또한 InvGAN은 Patch 기반 접근 방식을 채택하여 전체 장기 이미지를 단일 세포 해상도로 처리할 수 있는 능력을 갖추고 있으며, 아드버셜 손실(adversarial loss)을 사용하여 이미지 품질을 향상시킵니다.

- **Performance Highlights**: 테스트 결과, InvGAN은 25% 해상도에서 기존의 방법들과 유사한 정확도를 유지하면서도 약 7분이라는 짧은 시간 안에 등록을 완료합니다. 반면 100% 해상도에서는 대부분의 기존 등록 방법이 실패했던 반면, InvGAN은 단 10분 만에 등록을 진행했습니다. 이는 텍스처와 세포의 구조를 고려한 새로운 접근 방식의 유효성을 입증하며, 고해상도 이미지 등록의 새로운 패러다임을 제시합니다.



### PSO-Net: Development of an automated psoriasis assessment system using attention-based interpretable deep neural networks (https://arxiv.org/abs/2501.18782)
Comments:
          Accepted to IEEE ISBI 2025. 5 Pages, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 PSO-Net이라는 새로운 해석 가능한 딥 러닝 아키텍처를 제안합니다. 이 모델은 디지털 이미지를 사용하여 주의 기반 점수를 생성하고, 이를 통해 PASI(Plastic Area and Severity Index) 점수를 산출할 수 있도록 설계되었습니다. PSO-Net은 환자의 사진을 사용해 Psoriasis(건선) 진행 상황을 모니터링할 수 있는 새로운 방법을 제공합니다.

- **Technical Details**: PSO-Net의 구조는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 입력 이미지에서 밀집된 특징을 추출하는 인코더 모듈이고, 두 번째는 특정 영역에 집중하도록 돕는 주의 메커니즘입니다. 세 번째는 회귀 활성화 맵 생성으로, 이는 주의 점수를 순위별로 정리하여 체표의 중요한 부위를 강조하는 시각화를 제공합니다.

- **Performance Highlights**: PSO-Net은 두 명의 임상 전문가와의 비교에서 상호 클래스 상관 계수 82.2% 및 87.8%를 달성하여 신뢰도를 보여주었습니다. 이 방법은 환자가 수초 안에 점수를 받을 수 있게 하여 임상 시험에서 환자의 피부 상태를 효율적으로 모니터링할 수 있도록 합니다. 특히, 접근이 제한된 지역에 거주하는 환자들에게 유용한 솔루션입니다.



### Distillation-Driven Diffusion Model for Multi-Scale MRI Super-Resolution: Make 1.5T MRI Great Again (https://arxiv.org/abs/2501.18736)
- **What's New**: 7T MRI는 기존 1.5T MRI보다 훨씬 향상된 공간 해상도를 제공하여 해부학적 구조를 세밀하게 시각화할 수 있는 새로운 가능성을 열어줍니다. 하지만 고비용과 제한된 가용성으로 인해 임상에서의 활용이 제한적입니다. 이러한 문제를 해결하기 위해, 본 연구에서는 1.5T MRI 스캔을 기반으로 7T와 유사한 이미지를 생성하는 새로운 Super-Resolution (SR) 모델을 제안합니다. 이 모델은 확산 기반 아키텍처를 활용하며, 7T 이미지로부터의 경량화된 기능 매핑을 통해 점진적인 성능 향상을 목표로 합니다.

- **Technical Details**: 본 모델은 gradient nonlinearity correction과 bias field correction 데이터를 가이드로 사용하여 7T MRI 재구성을 수행합니다. 교육 모델의 추론 단계에서 기능 맵을 활용하여 학생 모델이 점진적으로 7T SR 성능을 달성할 수 있도록 돕습니다. 또한, 모델의 배포 가능성을 높이기 위한 progressive distillation 전략도 도입되어, 학생 모델은 다양한 해상도의 MRI 입력을 수용할 수 있고, 재훈련 없이도 성능을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 기준이 되는 교육 모델은 최신 SR 성능을 달성했으며, 학생 모델은 경량임에도 불구하고 최소한의 성능 손실로 7T SR을 수행할 수 있음을 보였습니다. 본 방법은 Massachusetts General Hospital의 임상 데이터를 통해 임상적 관련성을 확인하였으며, 코드도 공개되어 있습니다. 이로 인해 본 연구는 7T MRI의 해상도 향상 가능성을 기존의 기술적 제한 없이 실현할 수 있는 잠재력을 보여줍니다.



### Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting (https://arxiv.org/abs/2501.18672)
Comments:
          Visit our project page at this https URL

- **What's New**: 본 논문에서는 DYG라는 새로운 3D 드래그 기반 편집 방법을 제안합니다. 이 방법은 사용자가 3D 마스크와 제어 점 쌍을 통해 원하는 편집 영역과 방향을 지정할 수 있도록 해, 기존의 기하학적 변화 편집에서의 한계를 극복합니다. DYG는 기하학적 결과의 정확한 제어를 가능하게 하여, 편집 결과의 질을 크게 향상시킵니다.

- **Technical Details**: DYG는 다중 크기 삼면체 위치 인코더(MTP Encoder)와 지역 특이 위치 디코더(RSP Decoder)를 통합하여 3D Gaussian Splatting(3DGS)에서 발생하는 희박한 데이터 분포 문제를 해결합니다. 이를 통해 사용자는 지속적으로 정확한 기하학적 편집이 가능하게 되며, Soft Local Edit(SLE) 전략을 통해 주변 지역의 무결성을 보존하면서 원하는 영역에 집중할 수 있습니다.

- **Performance Highlights**: 다양한 실험 결과를 통해 DYG는 기존 방법들에 비해 편집 효과와 품질 면에서 우수한 성능을 보였습니다. 특히, 드래그 기반으로 제어 점 프롬프트에 의해 안내되는 편집이 효과적으로 이루어졌습니다. DYG는 다양한 편집 작업에 대해 뛰어난 다재다능성과 일반화 능력을 입증하며, 최첨단(SoTA) 3D 장면 편집 결과를 제공합니다.



### Rethinking the Upsampling Layer in Hyperspectral Image Super Resolution (https://arxiv.org/abs/2501.18664)
- **What's New**: 이 논문에서는 경량화된 단일 하이퍼스펙트럴 이미지 슈퍼해상도(SHSR) 네트워크인 LKCA-Net을 제안합니다. LKCA-Net은 하이퍼스펙트럴 이미지의 다중 스케일 채널 특성을 조정하기 위해 채널 주의(channels attention) 메커니즘을 포함하고 있습니다. 또한, 배울 수 있는 업샘플링(layer)에서의 저순위(low-rank) 성질이 경량 SHSR 방법의 주요 병목현상임을 입증하였습니다.

- **Technical Details**: 저자는 저순위 근사(low-rank approximation) 전략을 사용하여 배울 수 있는 업샘플링 레이어의 매개변수 중복을 최적화하고, 이러한 낮은 순위 근사가 네트워크의 특성 표현 능력을 유지하기 위한 지식 증류(knowledge distillation) 기반의 특성 정렬(feature alignment) 기술을 도입합니다. 이 연구는 기존의 복잡한 SHSR 네트워크 구조를 변경하지 않고도 업샘플링 레이어의 전통적인 구조를 최적화할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과는 LKCA-Net이 Chikusei, Houston 2018 및 Pavia Center 데이터셋에서 최첨단 방법들(SOTAs)과 비교했을 때 경쟁력 있는 성능을 보였으며, 다른 SHSR 방법들에 비해 수십 배에서 수백 배까지 속도 향상을 달성할 수 있음을 보여줍니다. 이러한 결과는 저순위 근사 및 특성 정렬 전략의 효과성을 강조하며, 저자들이 제안한 LKCA-Net의 구조가 실질적인 성능 개선에 기여함을 말해줍니다.



### A Radiance Field Loss for Fast and Simple Emissive Surface Reconstruction (https://arxiv.org/abs/2501.18627)
- **What's New**: 이 논문은 이미지를 방출 표면 기반 장면 표현으로 빠르고 간단하게 변환하는 기법을 제안합니다. 기존의 방출 볼륨 재구성 알고리즘을 기반으로 하여, 손실 함수의 미세한 수정으로 코드 몇 줄의 수정만으로 구현할 수 있습니다. 이 방법은 알파 블렌딩 및 레이 마칭을 제거하고 손실 계산으로 이 단계를 이동시켜, 더 높은 수렴성을 촉진합니다.

- **Technical Details**: 본 연구에서는 훈련 사진을 장면에 투영하고, 해당 장면에서 나오는 기하학적 방출로부터 약한 차이를 최소화하여 표면의 분포를 최적화합니다. 이 방식은 각 픽셀의 색상과 일치하도록 각 레이의 지점에서 독립적인 그래디언트를 제공합니다. 결과적으로, 기존의 볼륨 재구성 방법에서 파생된 방출 장 필드 손실을 활용하여 기본 알고리즘에 쉽게 통합할 수 있게 됩니다.

- **Performance Highlights**: 우리의 방법은 Instant NGP의 수정된 변형으로 구현되었으며, PSNR 대 시간 측면에서 유사한 속도를 유지합니다. 평균적으로, 생산된 표면의 PSNR은 볼륨 접근법보다 평균 0.1 dB 낮습니다. 이러한 결과는 미세한 조정만으로도 효율적인 성능을 유지하면서도 높은 품질의 방출 표면 모델을 얻을 수 있음을 보여줍니다.



### Review and Recommendations for using Artificial Intelligence in Intracoronary Optical Coherence Tomography Analysis (https://arxiv.org/abs/2501.18614)
- **What's New**: 이번 논문은 인공 지능(AI) 기반의 관상 동맥 질환(CAD) 진단에 대한 체계적인 리뷰를 실시하였습니다. 특히, IVOCT(intravascular optical coherent tomography) 이미지를 활용한 AI 모델의 임상 유용성을 평가한 것이 주목할 만합니다. 2015년 1월부터 2023년 2월 사이에 발표된 문헌을 종합하여 AI 기반의 CAD 진단 현황을 면밀히 분석하였습니다.

- **Technical Details**: 연구에서는 총 5,576개의 논문을 검토하였으며, 513편이 초기 스크리닝을 통과하고, 최종적으로 35편이 품질 검사를 통해 선정되었습니다. 대부분의 모델이 방법론적 결함(methodological flaws)과 기본적인 편향(bias)으로 인해 임상 사용에 적합하지 않다는 결론에 도달했습니다. 이러한 문제를 해결하기 위해, 모델 품질 개선 및 연구 관행의 향상에 관한 추천 사항을 제시하였습니다.

- **Performance Highlights**: 발견된 모델의 대다수가 현재 임상에서 활용되기에는 한계가 있으며, 실제 임상 적용 가능성을 갖춘 AI 제품 개발을 위한 노력과 연구의 질 향상이 필요하다는 점을 강조하였습니다. 논문에서는 AI 기술이 CAD 진단의 정확성과 속도를 향상시킬 잠재력을 지니고 있지만, 이를 위해 먼저 기존 연구의 품질을 높여야 한다고 주장합니다.



New uploads on arXiv(cs.AI)

### Do LLMs Strategically Reveal, Conceal, and Infer Information? A Theoretical and Empirical Analysis in The Chameleon Gam (https://arxiv.org/abs/2501.19398)
- **What's New**: 이번 연구에서는 대형 언어 모델 기반(LLM-based) 에이전트의 정보 제어 및 의사결정 능력을 나타내기 위해 'The Chameleon'이라는 언어 기반 숨은 정체성 게임을 활용했습니다. 기존 LLM들은 적대적 상호작용에서 비협조적 집단 속에서 어떻게 정보akam 비밀을 유지하는지, 그리고 협력이 가능한 기타 플레이어에게 정보를 전달하는지에 대한 성능 한계를 드러내었습니다. 연구 결과, LLM 기반 비카멜레온 에이전트들이 과도한 정보를 노출시키며 전략적 상호작용에서 부적합함을 확인하였습니다.

- **Technical Details**: 이 연구는 게임 이론을 바탕으로 LLM들이 추측 게임인 'The Chameleon'에서의 전략을 분석했습니다. 게임은 비카멜레온 플레이어들이 카멜레온 에이전트를 식별하고, 비밀 정보를 숨기며, 반대 플레이어의 응답에 따라 결정하는 과정을 포함합니다. 이론적 분석을 통해 두 가지 유형의 전략, 즉 정보 concealment과 revealing 전략을 고려하여 각 전략이 주는 효과를 정량적으로 평가하였습니다.

- **Performance Highlights**: LMMs (예: GPT-4o, Gemini 1.5) 의 성과는 이론적으로 달성 가능한 승률에 크게 미치지 못했습니다. 예를 들어, 4인 게임에서 GPT-4o 비카멜레온의 승률은 5%에 불과하여 기본적인 전략으로는 23%를 달성할 수 있었던 것으로 나타났습니다. 이 결과는 LLM들이 opponent의 응답으로부터 비밀 단어를 유추하며, 이를 통해 비카멜레온의 정체성은 식별하지만 비밀을 정확하게 숨기지 못한다는 것을 보여줍니다.



### MINDSTORES: Memory-Informed Neural Decision Synthesis for Task-Oriented Reinforcement in Embodied Systems (https://arxiv.org/abs/2501.19318)
- **What's New**: 본 논문에서는 MINDSTORES라는 새로운 경험 증강 계획 프레임워크를 소개합니다. 이 프레임워크는 임베디드 에이전트가 자연환경과의 상호작용을 통해 정신 모델을 구축하고 활용하도록 돕습니다. 기존의 제로샷 언어 모델(Zero-shot LLM) 계획 방식에 경험을 통합하여, 과거의 경험을 데이터베이스에 저장하고 이를 통해 행동 계획을 개선할 수 있습니다.

- **Technical Details**: MINDSTORES는 (상태, 작업, 계획, 결과) 쌍의 자연어 임베딩으로 축적된 경험을 나타내며, 이를 통해 LLM 플래너가 효과적으로 정보를 검색하고 추론할 수 있도록 합니다. 프레임워크는 관찰, 관련 경험 검색, 문맥에 맞는 계획 생성, 행동 실행, 결과 기록 등 순환적으로 작동합니다. 이를 통해 에이전트는 학습을 통한 적응력을 높이고, 과거의 통찰력을 기반으로 계획을 정교화할 수 있습니다.

- **Performance Highlights**: MineDojo 환경에서 수행된 광범위한 실험을 통해, MINDSTORES는 기존의 기억 기반 LLM 계획자들보다 심각하게 지식을 학습하고 적용하는 능력이 뛰어난 것으로 나타났습니다. 특히, MINDSTORES는 제로샷 접근 방식의 유연성과 일반화 이점을 유지하면서 개방형 세계 과제에서 평균 9.4%의 성능 개선을 보여주었습니다. 이는 지속적으로 학습할 수 있는 AI 시스템 개발에 중요한 진전을 나타냅니다.



### Ontological analysis of proactive life event services (https://arxiv.org/abs/2501.19308)
- **What's New**: 이번 연구는 여러 정부 기관이 공동으로 제공하는 Life Event Service의 개념을 심층적으로 분석합니다. 이 서비스는 개인의 특정 상황에서 발생하는 의무와 권리를 이행할 수 있도록 돕는 디지털 공공 서비스입니다. 연구에서는 e-government 및 디지털 공공 서비스 구현에서 필수적인 용어의 정확한 의미 동의의 필요성을 강조하고 있습니다.

- **Technical Details**: 연구에서는 Life Event Service의 개념을 보다 철저히 이해하기 위해 온톨로지 분석을 적용합니다. 이는 서비스 설계와 실행에 필요한 개념과 관계에 대한 명확한 이해를 통해 이루어지며, 추상적 개념인 universals와 구체적 개념인 particulars로 나누어 집니다. 특히, endurants와 perdurants를 구분하며 시간에 따른 존재와 행동을 분석합니다.

- **Performance Highlights**: 이 연구는 Estonian 디지털 사회 발전 계획에 부합하는 AI 기반의 인간 중심 공공 서비스의 새로운 방향을 제시합니다. 또한, Life Event Service의 통합적 접근을 통해 공공 서비스의 복잡성을 감소시키고, 소비자에게 원활한 경험을 제공할 수 있는 방향으로 나아가고 있습니다.



### SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling (https://arxiv.org/abs/2501.19306)
- **What's New**: 이 논문에서는 Self-Enhanced Test-Time Scaling (SETS)라는 새로운 방법론을 제안합니다. SETS는 최근의 고급 LLM의 자기 검증(self-verification) 및 자기 수정(self-correction) 기능을 활용하여 복잡한 작업에 대한 테스트 시간 컴퓨테이션(test-time computation)을 효율적으로 수행할 수 있도록 통합된 프레임워크를 제공합니다. 기존의 방법론들이 테스트 시간 컴퓨테이션의 확장에 따라 수익이 줄어드는 문제를 해결하고자 합니다.

- **Technical Details**: SETTS에서는 4개의 데이터 세트인 Trip Planning, Meeting Planning, Calendar Scheduling 및 LiveBench Reasoning을 대상으로 실험을 수행합니다. 각 데이터 세트에서 자기 검증과 자기 수정 작업에 대한 프롬프트(prompt)를 설정하며, 이를 통해 다양한 방법론들의 성능을 비교 분석합니다. NATURAL PLAN 벤치마크는 각 작업에 대해 몇 가지 예시를 제공하고 있으며, 이를 통해 작업의 난이도를 제어할 수 있는 변수를 활용합니다.

- **Performance Highlights**: SETS 방법론은 기존 방법론에 비해 현저하게 성능이 개선되었음을 보여줍니다. 각 데이터 세트에서 자가 수정(self-correction) 성능을 측정한 결과, GEMINI-1.5-Pro-002 모델이 가장 강력한 성능을 보이며, 특히 Trip Planning에서 n=10까지 개선 효과가 지속됩니다. 반면, 일부 모델인 Claude-3.5-Sonnet은 제한된 자기 수정 성능을 보이며, 특정 데이터 세트에 따라 정확도가 감소하는 경향을 보였습니다.



### Synthetic User Behavior Sequence Generation with Large Language Models for Smart Homes (https://arxiv.org/abs/2501.19298)
- **What's New**: 최근 스마트 홈 시스템의 사용이 증가함에 따라 보안 문제도 우려되고 있습니다. 대부분의 스마트 홈 보안 솔루션은 고정된 데이터셋을 이용해 훈련되며, 이는 데이터 수집이 시간이 많이 걸리고 현실 환경 변화에 대한 유연성이 부족합니다. 이 연구에서는 새로운 환경 변화를 반영하는 합성 데이터셋을 생성하는 IoTGen 프레임워크를 통해 이러한 문제를 해결하고자 합니다.

- **Technical Details**: IoTGen 프레임워크는 Structured Pattern Perception Compression (SPPC) 방법을 통해 IoT 행동 데이터의 중요 정보를 보존하면서 토큰 소비를 크게 줄입니다. 이와 함께 프로필 구성 및 데이터 생성 프로세스를 체계화하여 스마트 홈 시스템이 동적인 환경에 적응할 수 있도록 지원합니다. 특히, LLM(large language models)의 잠재력을 활용하여 새로운 장면 데이터를 생성해 내는 것이 특징입니다.

- **Performance Highlights**: 이 연구의 결과는 기존의 고정 데이터에 의존하던 시스템에 비해 훨씬 높은 일반화 성능을 보여주고, 사용자의 행동에 따른 모델의 유연성을 강화합니다. 또한, 새로운 합성 데이터가 원래의 사용자 행동 패턴과 일치하도록 구성되어 실세계 환경에서의 적합성을 증가시킵니다. 이러한 접근법은 스마트 홈 시스템의 오픈 월드 환경 구축에 기여하며, 보다 안전하고 효율적인 스마트 홈 환경을 구현하는 데 도움을 줄 것입니다.



### Concept-Based Explainable Artificial Intelligence: Metrics and Benchmarks (https://arxiv.org/abs/2501.19271)
Comments:
          17 pages it total, 8 main pages

- **What's New**: 이 논문은 머신러닝 모델의 해석 가능성을 높이기 위한 개념 기반 설명 방법, 특히 concept bottleneck models (CBMs)에 대해 깊이 있는 검증이 부족했던 기존 가정을 분석합니다. 연구자들은 이러한 개념이 네트워크의 특징 공간에 어떻게 적절히 연결될 수 있는지에 대한 체계적인 검토가 부족하다는 점을 지적하며, 세 가지 새로운 메트릭인 concept global importance metric, concept existence metric, concept location metric을 제안합니다. 또한, 이러한 메트릭을 사용하여 CBMs의 성능을 평가하는 벤치마크 문제를 설정하고 이를 통해 개념 정렬의 정확성을 검증합니다.

- **Technical Details**: 제안된 메트릭은 (1) 클래스의 각 이미지에 대한 개념 정렬을 측정하는 concept global importance metric (CGIM), (2) 분류를 위해 중요한 개념이 이미지에 존재하는지를 측정하는 concept existence metric (CEM), (3) 해당 개념과 관련된 특징 맵의 활성화 영역이 기대되는 위치와 얼마나 근접한지를 측정하는 concept location metric (CLM)입니다. 이러한 메트릭은 Caltech-UCSB Bird (CUB) 데이터셋을 활용한 벤치마크 문제로 설정되었습니다. 각 메트릭은 개념 기반 XAI 방법의 효과와 한계를 평가하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, 많은 경우 포스트-혹 (post-hoc) konsep-bottleneck models에 의해 결정된 가장 중요한 개념조차 입력 이미지에 존재하지 않거나, 존재하더라도 그 유의성 맵(saliency maps)이 기대되는 영역과 일치하지 않는 것으로 나타났습니다. 이러한 제한의 근본 원인은 매우 자연적인 개념들의 상관관계에 있으며, 연구 결과는 특히 공간 해석 가능성이 중요한 상황에서 개념 기반 설명 기법의 더욱 신중한 적용이 필요함을 강조합니다.



### Jackpot! Alignment as a Maximal Lottery (https://arxiv.org/abs/2501.19266)
- **What's New**: 이 논문에서는 Reinforcement Learning from Human Feedback (RLHF)의 한계를 극복하기 위해 확률적 사회 선택(rule)인 최대 복권(maximal lotteries)을 제안합니다. 이는 대규모 언어 모델(Large Language Models, LLMs)을 인류 가치와 일치시키기 위한 새로운 접근법입니다. 연구자들은 이 방안이 다수의 선호를 존중하면서 비-추이성(non-transitivity) 문제를 처리하는 보다 견고한 방법을 제공한다고 주장합니다.

- **Technical Details**: RLHF는 보상 모델을 훈련시키고 이를 통해 정책인 LLM을 유인하는 방식으로 작동합니다. 보상 모델 rθ(𝑥, 𝑦)는 인간 평가자들의 선호에 따라 쌍비교를 통해 학습됩니다. 하지만 기존의 RLHF 방식은 여러 문제점이 존재하며, 다수의 선호를 적절하게 반영하지 못하는 등의 한계가 있습니다.

- **Performance Highlights**: 제안된 최대 복권 방법론을 통해 LLM의 출력이 집합적 인간 선호를 더 잘 반영할 수 있다는 실험적 결과가 확인되었습니다. 이 방법은 다수의 선호를 지원하고, 비-추이성에 대한 원칙적인 처리 방안을 제공하며, 무관한 대안에 대한 강인성을 보이는 성능을 발휘합니다.



### Objective Metrics for Human-Subjects Evaluation in Explainable Reinforcement Learning (https://arxiv.org/abs/2501.19256)
- **What's New**: 이 논문은 기존의 explainable reinforcement learning (XRL) 연구가 인간의 평가 없이 주관적인 지표에 의존하고 있다고 비판합니다. 연구자들은 행동 기반의 객관적인 인간 지표를 사용하여 설명을 평가할 필요가 있으며, 이를 통해 보다 재현 가능하고 비교 가능한 연구를 하도록 제안합니다. 특히, 디버깅과 인간-에이전트 팀워크에서 설명의 목표와 청중을 이해하는 것이 중요하다고 강조하고 있습니다.

- **Technical Details**: 설명 가능성은 본질적으로 인간을 위한 것이므로, 설명을 평가할 때 인간을 고려한 맥락에서 이루어져야 합니다. 이 논문에서는 XRL의 두 가지 주요 응용 분야인 디버깅과 인간-에이전트 팀워크에서 사용할 수 있는 다양한 객관적인 평가 방법론을 제시합니다. 각 방법론은 정의, 측정 방법, 설명에 대한 영향 등을 포함하고 있으며, 이는 XRL 연구의 진행 및 품질 보장에 기여할 수 있습니다.

- **Performance Highlights**: 주관적인 지표에만 의존하는 현재의 연구 방식에서 벗어나, 설명의 행동 가능성을 객관적인 인간 행동 측정으로 평가해야 한다고 주장합니다. 이 연구는 mini-world라는 새로운 환경을 사용하여 제안된 각 객관적 메트릭을 설명하고, 이를 통해 설명의 효과성과 실용성을 강조하고 있습니다. 궁극적으로, 이러한 접근 방식이 XRL 연구의 질적 향상과 지속 가능한 발전을 가능하게 할 것으로 기대됩니다.



### SHARPIE: A Modular Framework for Reinforcement Learning and Human-AI Interaction Experiments (https://arxiv.org/abs/2501.19245)
- **What's New**: SHARPIE (Shared Human-AI Reinforcement Learning Platform for Interactive Experiments)는 인간과 AI 간의 상호작용을 포함한 다양한 실험을 지원하기 위한 모듈형 플랫폼입니다. 이 플랫폼은 RL 환경을 위한 다재다능한 래퍼(wrapper)를 제공하며, 참가자와의 웹 인터페이스, 로깅 유틸리티 및 클라우드 배포 기능을 통합하여 연구자들이 인간과 RL 에이전트 간의 상호작용을 폭넓게 연구할 수 있도록 설계되었습니다. SHARPIE는 인간-AI 팀워크, 행동 위임, 선호 발굴 등과 같은 다양한 연구 질문을 다루는 데 유용합니다.

- **Technical Details**: SHARPIE는 여러 인간과 RL 에이전트 간의 효과적인 상호작용 연구를 위한 플랫폼입니다. 이 플랫폼은 인기 있는 RL, 다중 에이전트 RL 및 다중 목표 RL 환경 및 알고리즘을 위한 래퍼를 제공하며, 다양한 모달리티를 통해 인간과 RL 에이전트 간의 통신 채널을 구성할 수 있도록 지원합니다. 연구자는 이를 통해 RL 에이전트와 인간 간의 상호작용을 다양한 방식으로 탐구할 수 있습니다.

- **Performance Highlights**: SHARPIE는 RL 기반 인간-에이전트 상호작용의 표준을 도입하여 다중 에이전트 환경에서의 상호작용을 표준화하는 데 기여합니다. 이 플랫폼은 기본적인 RL 라이브러리에서 다루지 않는 복잡하고 다채로운 인간-에이전트 상호작용을 효과적으로 분석할 수 있는 가능성을 제공합니다. 또한, 다양한 실제 사용 사례를 통해 SHARPIE가 해결하고자 하는 문제의 범위를 구체화하고, 인지 과학과의 연관성도 강조합니다.



### An Empirical Game-Theoretic Analysis of Autonomous Cyber-Defence Agents (https://arxiv.org/abs/2501.19206)
Comments:
          21 pages, 17 figures, 10 tables

- **What's New**: 최근의 정교한 사이버 공격의 증가는 강력하고 복원력 있는 자율 사이버 방어(ACD) 에이전트의 필요성을 제기합니다. 이 논문은 ACD를 위한 딥 강화 학습(DRL) 접근 방식을 이해하고 개선하기 위해 원리에 기반한 더블 오라클(DO) 알고리즘을 적용하여 두 가지 주요 문제를 해결하고자 합니다. 첫째는 다양한 사이버 공격 전술과 기법에 대한 일반화 가능한 정책을 학습하는 것이며, 둘째는 ACD 에이전트의 보증 문제를 해결하는 것입니다.

- **Technical Details**: DO 알고리즘은 적대적인 대립 상황에서 상대의 정책에 대한 최적 반응을 반복적으로 학습하는 방식으로 작동합니다. 하지만 이 과정은 계산적으로 비용이 많이 들기 때문에, 본 연구에서는 잠재 기반 보상 조정(Potential-based Reward Shaping) 접근 방식을 도입하여 학습 속도를 높이고, 여러 접근 방식을 통합할 수 있는 다중 반응 오라클(Multiple Response Oracles, MRO) 알고리즘을 확장하였습니다. 이러한 접근은 이전 응답의 경험을 활용하여 빠른 수렴을 도모합니다.

- **Performance Highlights**: 실험적으로, ACD 에이전트는 공격자에게 강력하게 대응하며 새로운 성공적인 전술 개발에 어려움을 겪는 것으로 나타났습니다. 특히 VF-PBRS 오라클을 사용한 ACD 정책이 일반적인 접근 방식보다 더 강력한 성능을 보임을 확인했습니다. 최종적으로, 예비 훈련 모델을 활용한 MRO 알고리즘을 통해 더욱 신뢰할 수 있는 에이전트를 개발하는 것이 가능함을 보여주었습니다.



### Imitation Game for Adversarial Disillusion with Multimodal Generative Chain-of-Thought Role-Play (https://arxiv.org/abs/2501.19143)
- **What's New**: 이번 연구에서는 인공 지능의 신뢰성을 위협하는 적대적 환상(adversarial illusions)에 대한 새로운 방어 체계를 제안합니다. 제안된 디스일루전(disillusion) 패러다임은 모방 게임(imitation game) 개념에 기반하여, 다양한 공격 형태에 대해 통합된 방어를 제공하는 것을 목표로 하고 있습니다. 연구에서는 다중 모달 생성 에이전트(multimodal generative agent)가 중심이 되어 체인 오브 사고(reasoning)를 통해 샘플의 본질을 내부화하고 재구성하는 과정을 설명합니다.

- **Technical Details**: 이 논문에서는 두 가지 형태의 적대적 공격인 추론 시 유도된 환상(deductive illusion)과 학습 시 조건화된 환상(inductive illusion)을 다룹니다. 각각의 공격 유형은 신뢰성을 저해하는 방식으로, 베이너리 샘플이나 학습 데이터를 변형하여 모델의 결정을 왜곡할 수 있습니다. 제안하는 디스일루전 패러다임은 전통적인 샘플 복원을 넘어서 줄거리에 대한 이해를 통해 새로운 솔루션을 탐색합니다.

- **Performance Highlights**: 실험 시뮬레이션을 통해 제안된 방법론이 다양한 적대적 공격 시나리오에서 어떻게 작동하는지를 평가합니다. 특히, OpenAI의 ChatGPT를 활용하여 실험을 진행하며, 공격 상황에서 이 방법이 효과적으로 작용하는 것을 확인합니다. 이 연구는 향후 AI 신뢰성을 향상시키기 위한 방향성을 제시합니다.



### Logical Modalities within the European AI Act: An Analysis (https://arxiv.org/abs/2501.19112)
Comments:
          16 pages, 19 figures

- **What's New**: 이 논문에서는 European AI Act의 논리적 양식(logical modalities)을 종합적으로 분석하여, 이를 공식적으로 표현하기 위해 로지컬 다원주의 지식 공학 프레임워크(LogiKEy) 내에서의 구성을 목표로 합니다. LogiKEy는 형식적 방법을 기반으로 하는 규범적 추론을 위한 계산 도구를 개발하며, 다양한 논리를 통합하기 위해 Higher-Order Logic (HOL)을 사용합니다. 이러한 통합은 Isabelle/HOL이라는 증명 보조 도구의 도움을 받습니다.

- **Technical Details**: 분석된 AI Act 내의 양식들은 여러 차례 문서를 읽고 다양한 관심사를 시각화하여 도출되었습니다. AI Act는 AI 시스템의 위험 수준을 세 가지로 분류하며, 각 범주에 따라 서로 다른 규칙과 규제를 부여합니다. 의무는 'shall'로 표현되며, 금지는 'shall not'로 나타나고, 허가는 'may'와 같은 여러 전략을 통해 제시됩니다. 특히, 주의해야 할 더불어 의무(contrary-to-duty obligations, CTDs)가 발생하는 조건도 제시됩니다.

- **Performance Highlights**: 초기 실험은 이러한 임베딩이 자동 추론에 적합한지를 평가하며, 더 강력한 추론 능력으로 나아가기 위한 주요 도전 과제를 강조합니다. CTS의 사례를 통해 의무를 준수하지 않았을 때 어떻게 부가적인 의무가 발생하는지를 설명하며, 다수의 에이전트가 관여하는 복잡한 상황을 분석합니다. 이러한 논리는 AI Act 내의 양식과 규제를 좀 더 명확하게 표현하기 위한 규범적 추론의 기초를 마련할 수 있습니다.



### PathE: Leveraging Entity-Agnostic Paths for Parameter-Efficient Knowledge Graph Embeddings (https://arxiv.org/abs/2501.19095)
- **What's New**: PathE는 기존의 Knowledge Graph Embeddings(KGE) 방법과는 다른 접근 방식으로, 관계(relationship) 표현만 저장하고 동적으로 개체(entities) 임베딩을 계산하는 방법론을 제안합니다. 이 모델은 대규모 Knowledge Graph가 필요한 최신 AI 응용에서 더 효율적인 성능을 목표로 하고 있습니다. PathE는 다양한 경로 정보를 활용하여 개체를 맥락에 맞게 표현하며, 25% 미만의 매개변수로 높은 성능을 보여줍니다.

- **Technical Details**: PathE는 각 개체의 관계적 맥락(relational context)을 정의하고, 이를 여러 개체-관계 경로를 통해 계산하는 등 복잡한 구조를 단순화하여 연산 과부하를 줄입니다. 이를 위해 고유한 랜덤 워크를 통해 경로를 생성하고, 완전 연결층을 통한 학습을 통해 개체의 관계적 맥락을 인코딩하여 각 개체의 표현을 생성합니다. 이러한 방법으로 PathE는 의미적으로 풍부한 개체 표현을 구성하고, 사전에 훈련된 모델 없이도 새로운 개체를 포괄할 수 있는 능력을 가집니다.

- **Performance Highlights**: PathE 모델은 여러 KG 벤치마크 평가에서 현재의 최첨단 성능을 달성하였으며, 관계 예측 및 연결 예측(link prediction)에서도 우수한 성능을 보입니다. 비싼 하드웨어 없이도 소비자용 기기에서 효과적으로 훈련이 가능하며, 컴퓨팅 자원과 비용 측면에서 효율적입니다. PathE는 경로가 풍부한 Knowledge Graph(FB15k-237, CodeX-Large)에서 뛰어난 성과를 보여줍니다.



### Language Games as the Pathway to Artificial Superhuman Intelligenc (https://arxiv.org/abs/2501.18924)
Comments:
          This position paper argues that language games provide robust mechanism for achieving superhuman intelligence in large language models

- **What's New**: 이 논문은 인공지능(AI)에서 인간을 초월하는 지능(ASI)으로의 진화의 중요한 요소로 데이터 재생산(data reproduction)을 제시합니다. 전통적인 모델들은 한정된 인간 생성 분포 내에서만 최적화되어 stagnation(침체)에 빠지기 쉬운 문제가 있습니다. 따라서 저자들은 언어 게임(language games)을 통해 이 사이클을 깨고, 확장된 데이터 재생산을 가능하게 하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 언어 게임이 LLM의 데이터 재생산 및 모델 개선을 위한 세 가지 메커니즘인 역할 유동성(role fluidity), 보상 다양성(reward variety), 규칙 유연성(rule plasticity)을 기반으로 작동한다고 설명합니다. 이 메커니즘들은 모델의 학습과 상호작용 과정을 개선하며, 언어적 환경을 재구성하는 것이 가능하게 하여 개념적인 탐구를 촉진합니다. 또한, 강화 학습(reinforcement learning) 기법을 통합하여 모델의 복잡한 피드백 신호를 내부화하고 다양한 도메인에서 전략을 개선할 수 있다고 주장합니다.

- **Performance Highlights**: 저자들은 언어 게임을 통해 모델과 인간의 협력을 통한 데이터 흐름의 생성이 가능하여, 데이터 재생산을 새로운 차원으로 끌어올린다고 강조합니다. 이를 통해 지속적으로 새로운 아이디어와 지능적 행동을 생성할 수 있는 잠재력을 사전 파악할 수 있습니다. 결론적으로, 이 접근 방식은 인간 수준의 지능을 초월하는 AI 개발을 위한 로드맵을 제시하며, 모델의 학습 과정에서 성능 개선을 기대할 수 있습니다.



### Bridging the Reasoning Gap: Small LLMs Can Plan with Generalised Strategies (https://arxiv.org/abs/2501.18817)
Comments:
          7 page body, 2 page references, 16 page appendix (25 pages total); 2 figures; submitted to IJCAI2025

- **What's New**: 최근 대규모 언어 모델(LLM)의 추론 능력 향상은 간단한 계획 작업 해결 능력의 증가를 보여줍니다. 그러나 LLM의 성능 향상이 모델의 크기와 복잡성에 의존하므로, 운영 비용이 증가하고 있습니다. 이 연구는 자원이 덜 소모되는 LLM의 추론 능력을 강화하기 위한 두 가지 접근 방식을 제안합니다: 일반화된 전략 생성과 오류 교정 프로세스를 통한 개선입니다.

- **Technical Details**: 제안된 방법은 자원 소모가 적은 LLM이 보다 강력한 LLM에서 생성된 일반화된 전략을 제공받아, 해당 도메인 내 작업을 해결하는 것입니다. 또한, 모델의 솔루션에서 오류를 식별하고 이를 교정하는 과정을 통해 성능을 향상시킵니다. 이러한 방법을 통해 향후 불필요한 자원 소비를 줄이고 성능을 극대화할 수 있는 가능성을 확인하였습니다.

- **Performance Highlights**: 실험 결과, 자원이 덜 소모되는 LLM은 제안된 방법을 활용했을 때 강력한 LLM과 유사한 성능을 보였습니다. 특히, 일반화된 전략을 사용 시, 비용이 평균 30% 줄어들었으며, 4회의 오류 수정 후에는 50%까지 비용 절감이 이루어졌습니다. 이로 인해, 자원이 적게 소모되는 모델이 수학적 추론 작업에서 더 강력한 모델보다 20% 높은 성과를 달성했습니다.



### LLM-Generated Heuristics for AI Planning: Do We Even Need Domain-Independence Anymore? (https://arxiv.org/abs/2501.18784)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 도메인 독립적 계획 문제를 위한 효율적인 휴리스틱을 자동으로 유도하는 방법을 제안합니다. 이는 전통적인 도메인 독립적 방법론의 한계를 극복하고, 작업 설명을 기반으로 한 맞춤형 휴리스틱 설계를 가능하게 합니다. 특히, LLMs가 생성한 휴리스틱이 IPC 기준 벤치마크에서 기존 방법보다 우수한 성능을 보이면서 기계 학습의 새로운 패러다임 변화를 나타내는지를 탐구합니다.

- **Technical Details**: PDDL2.1(Planning Domain Definition Language, version 2.1)은 AI에서 계획 문제를 정의하는 데 널리 사용되는 공식 언어입니다. 이 연구에서는 PDDL2.1의 숫자 플루언트(numeric fluents) 및 제약이 있는 동작을 처리하는 방법에 초점을 맞추고 있습니다. LLM은 주어진 작업에 대한 후속 생성기(successor generator)와 목표 테스트(goal test) 함수를 일반 프로그래밍 언어로 구현한 후, 이를 통해 휴리스틱을 생성합니다.

- **Performance Highlights**: 실험 결과, LLM이 만들어낸 휴리스틱은 전통적인 방법에 비해 뛰어난 성능을 보여 주며, PDDL 표현이 부족한 문제도 해결할 수 있는 가능성을 보였습니다. 이러한 LLM 기반 휴리스틱은 복잡한 목표 테스트와 전이 함수가 포함된 도메인에서도 성능 향상을 이루어 내며, 이는 자동화된 계획 시스템의 적용 범위를 크게 확장시키는 효과가 있습니다.



### Simulation Streams: A Programming Paradigm for Controlling Large Language Models and Building Complex Systems with Generative AI (https://arxiv.org/abs/2501.18668)
Comments:
          Technical report accompanying the release of code on GitHub

- **What's New**: Simulation Streams는 대규모 언어 모델(LLMs)을 효율적으로 활용하여 복잡하고 동적인 시뮬레이션과 에이전트 워크플로우를 제어하도록 설계된 프로그래밍 패러다임입니다. 이 프레임워크는 LLM의 능력을 최대한 활용하면서도 일관성을 유지하고 정보 선택적 포함/제외 및 세계 규칙의 엄격한 시행을 다루는 것을 목표로 합니다. 시뮬레이션 스트림은 상태 기반 접근 방식을 채택하여 변수를 순차적으로 수정하고, 이와 동시에 일관된 출력 형식을 유지합니다.

- **Technical Details**: Simulation Streams는 상태 기반 접근 방식에서 작동하며, 각 변화를 정의하는 '연산자'(operator)를 통해 변수를 수정합니다. 이 시스템은 엔티티-컴포넌트-시스템(ECS) 아키텍처를 도입하여 프로그램 작성의 직관성을 높이고 다양한 구성 요소와 엔티티에서 워크플로우 재사용을 용이하게 합니다. 이 프레임워크는 LLM의 내부에서의 'in-distribution' 생성을 유지시키기 위한 서브스트림 방식을 채택하고 있으며, 이를 통해 전체적인 형식의 일관성을 유지합니다.

- **Performance Highlights**: Simulation Streams는 시장 경제 시뮬레이션, 사회적 캐치 게임 시뮬레이션 등을 통해 다양한 복잡한 시나리오를 처리할 수 있는 능력을 시연합니다. 이러한 예시들은 LLM 기반 시뮬레이션에서의 일관성 유지 및 에이전트 워크플로우 비교를 가능하게 하며, 수백 또는 수천 번의 반복을 통해 계속해서 흥미로운 발전을 모색하는 데 중점을 두고 있습니다. 이 접근 방식은 시뮬레이션 규칙을 엄격하게 시행하면서도 복잡한 시나리오를 효과적으로 관리할 수 있게 해줍니다.



### Enhancing Large Language Model Efficiencyvia Symbolic Compression: A Formal Approach Towards Interpretability (https://arxiv.org/abs/2501.18657)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 코드 생성 및 논리적 추론에 관한 토큰 효율성 문제를 해결하기 위한 새로운 이론적 프레임워크를 제안합니다. 기존 LLMs는 복잡한 작업을 수행할 때 2.1배에서 3.4배의 중복 토큰을 생성하여 추론 비용을 증가시키고 모델 해석성을 저하시킵니다. 제안된 방식은 기호 압축(symbolic compression)을 기반으로 하여 정보 이론의 최적 인코딩 이론과 결합하여 의미적 무결성을 유지하면서 토큰 효율성을 크게 개선합니다. 이를 통해 LLMs의 효율적 추론을 위한 새로운 이론적 도구와 모델 해석성을 위한 기호적 경로를 열어줍니다.

- **Technical Details**: 이 연구에서는 기호 밀도(symbolic density)와 모델 해석성 사이의 정량적 관계를 수립하고, 인코딩 효율성을 평가하기 위한 미분 가능 압축 계수(metric) 프레임워크를 제안합니다. 또한, SKI 조합자(SKI combinators) 기반 재귀 인코딩 방식을 통해 구문 트리(syntax tree)의 압축을 달성하고, 문맥 추론(context inference)과 기호 과부하(symbolic overloading) 간의 최적 균형을 찾기 위한 동적 균형 알고리즘(dynamical balance algorithm)을 설계합니다. 최종적으로 LAEL 언어의 저비용 적용을 위해 PEFT(파라미터 효율 미세 조정) 기술을 활용하여 LLMs의 토큰 비용을 더욱 줄이고 모델 전체의 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 코드 생성 작업에서 78.3%의 토큰 압축 비율을 달성하며, 구조적 명확성(structural explicitness)을 통해 논리 추적 가능성을 62% 향상시켰습니다. 이 과정에서 기호 압축을 통해 코드의 의미적 일관성을 보장하면서 압축된 코드를 생성합니다. PEFT 기술을 이용한 GAEL 언어의 적용이 가능해지며, 이를 통해 다양한 작업과 도메인에서 고효율적인 응용이 가능합니다. 이러한 성과는 LLMs의 해석성과 효율성에 큰 기여를 하며, 앞으로의 연구에 이론적 기반을 제공합니다.



### AI Biases Towards Rich and Powerful Surnames (https://arxiv.org/abs/2501.19407)
Comments:
          54 pages, 5 figures, 1 table

- **What's New**: 이번 연구는 성씨(surnames)가 AI 기반 의사결정에 미치는 영향을 조사한 첫 번째 연구로, 특히 고용 추천(hiring recommendations), 리더십 임명(leadership appointments), 그리고 대출 승인(loan approvals)과 같은 중요한 분야에서의 효과에 중점을 두고 있습니다.

- **Technical Details**: 미국과 태국의 600개의 성씨를 분석하여 Rich, Legacy, Normal, 그리고 유사 발음 Variant 그룹으로 분류했습니다. 연구 결과, 고위층 성씨가 AI가 생성한 권력(power), 지능(intelligence), 그리고 부(wealth)에 대한 인식을 예측하는 데 일관되게 작용한다는 사실이 드러났습니다.

- **Performance Highlights**: 특히 지향된 바에 따르면, 성씨 편향(biases)이 작용하는 주요 경로는 인식된 지능(perceived intelligence)으로 나타났습니다. 성씨와 함께 객관적인 자격을 제시하는 것이 편향을 줄일 수 있지만, 자격이 전반적으로 낮은 맥락에서는 이러한 편향이 여전히 남아있음을 보여줍니다.



### Redefining Machine Unlearning: A Conformal Prediction-Motivated Approach (https://arxiv.org/abs/2501.19403)
- **What's New**: 본 논문에서는 머신 언러닝의 평가 지표의 부족성을 지적하고, 새로운 평가 지표를 제안합니다. 기존 지표는 모델이 특정 데이터를 얼마나 잘 잊었는지를 완전하게 평가하지 못하는 문제가 있었습니다. 이를 개선하기 위해, 저자들은 conformal prediction에 영감을 받은 두 가지 새로운 지표를 설계했습니다.

- **Technical Details**: 머신 언러닝은 특정 훈련 데이터의 효과를 목표로 제거하는 방식입니다. 이를 위해 두 가지 잊기 시나리오, 즉 특정 인스턴스를 무작위로 잊는 경우와 특정 클래스에 대한 모든 정보를 제거하는 클래스 기반 잊기를 고려합니다. 저자들은 unlearning 과정 후의 모델이 어떻게 작동하는지를 평가하기 위해 새로운 프레임워크를 제안하였습니다.

- **Performance Highlights**: 새로운 지표와 프레임워크를 통해 실험을 진행한 결과, 기존의 머신 언러닝 방법들이 만족스러운 잊기 성과를 내지 못하는 경우가 많음을 발견했습니다. 또한 저자들은 이 프레임워크가 기존의 훈련 기반 머신 언러닝 방법의 잊기 품질을 유의미하게 개선시킨다는 것을 입증했습니다.



### Vintix: Action Model via In-Context Reinforcement Learning (https://arxiv.org/abs/2501.19400)
Comments:
          Preprint. In review

- **What's New**: 이번 연구에서는 In-Context Reinforcement Learning (ICRL)의 확장을 위한 첫 걸음을 제안합니다. 고정된 교차 도메인 모델을 통해 보상을 극대화하는 방법을 학습하는 능력을 갖춘 다기능 액션 모델인 Vintix를 소개합니다. 이 연구는 기존의 전문가 증류(expert distillation) 방법에 대한 매력적이고 경쟁력 있는 대안을 제시하며, ICRL이 일반적인 의사결정 시스템을 위한 확장 가능한 접근법이라는 가능성을 강조합니다.

- **Technical Details**: 우리의 접근법은 Algorithm Distillation (Laskin et al., 2022)라는 두 단계의 Offline Meta-RL 알고리즘을 기반으로합니다. 첫 번째 단계에서는 기본 강화 학습 알고리즘으로부터 순서가 매겨진 학습 이력을 수집하고, 두 번째 단계에서는 다음 행동 예측을 위해 훈련된 디코더 전용 트랜스포머를 이용하여 정책 개선 연산자를 인과적 시퀀스 모델로 증류합니다. 우리는 이 방법에 두 가지 보강을 제안합니다: (1) Zisman et al. (2024a)의 노이즈 증류 절차의 연속 확장, (2) 획득한 데이터 세트를 활용한 일반화된 교차 도메인 훈련입니다.

- **Performance Highlights**: 우리는 Vintix 모델이 훈련 작업에서 데모 수준의 성능을 자가 수정할 수 있음을 실험적으로 입증했습니다. 또한, 이 모델은 인퍼런스 시간에 제어된 매개변수 작업 변형에 적응하는 초기 증거를 보여줍니다. 다기능 액션 모델의 초기 결과는 ICRL 접근법이 다양한 도메인 및 과제를 처리하는 데 있어 유망한 가능성을 가지고 있음을 시사합니다.



### Scalable-Softmax Is Superior for Attention (https://arxiv.org/abs/2501.19399)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 연구에서는 Transformer 기반 언어 모델에서의 Softmax 기능의 한계를 극복하기 위한 새로운 접근법인 Scalable-Softmax(SSMax)를 제안합니다. 기존 Softmax는 입력 벡터 크기가 증가함에 따라 확률 분포가 평탄해짐으로써 모델의 긴 컨텍스트에 대한 일반화 능력을 저하시킵니다. SSMax는 이러한 문제를 해결하며, 기존의 Transformer 구조에 원활하게 통합될 수 있습니다.

- **Technical Details**: Scalable-Softmax(SSMax)는 입력 벡터를 확률 분포로 변환하는 과정에서 입력 벡터의 크기에 의존하는 방식으로 설계되었습니다. 이를 통해 대량의 입력 데이터에서도 주의 축소(attention fading)를 방지할 수 있습니다. SSMax는 학습 가능한 스칼라 매개변수인 s와 b를 이용해 각 레이어와 헤드에 독립적으로 적용되며, 다양한 컨텍스트 크기에 적응할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: SSMax를 사용하는 모델은 사전 훈련 동안 손실 값을 더 빠르게 감소시키며, 긴 컨텍스트와 핵심 정보 검색 작업에서 상대적으로 월등한 성과를 보입니다. 실험 결과, 전통적인 Softmax를 대체한 SSMax는 모델이 긴 컨텍스트에서도 고유한 토큰에 집중할 수 있도록 돕는 것으로 나타났습니다. 특히, 사전 훈련이 완료된 모델에서도 Softmax를 SSMax로 교체하여 일정 수준의 개선 효과를 경험할 수 있음을 보였습니다.



### s1: Simple test-time scaling (https://arxiv.org/abs/2501.19393)
Comments:
          46 pages (9 main), 10 figures, 14 tables

- **What's New**: 최근 언어 모델의 성능 향상은 대규모 자기 지도(Self-Supervised) 사전 학습을 통한 학습 시간의 계산(resource) 증가에 크게 의존하고 있습니다. 이에 따라, 테스트 타임 스케일링(Test-time scaling)이라는 새로운 접근 방식이 등장했습니다. 연구진은 1,000개의 질문을 포함한 소규모 데이터셋(s1K)을 통해 간단한 검증 방법을 사용하여 강력한 추론 성능을 달성하는 테스트 타임 스케일링 방법론을 제안하고 있습니다.

- **Technical Details**: 연구진은 질문의 난이도, 다양성 및 품질을 기준으로 1,000개의 질문과 그에 따른 추론 추적(reasoning trace)을 포함하는 데이터셋 s1K를 구성하였습니다. 또한, budget forcing 기법을 통해 모델의 사고 프로세스를 조정하며 더 많은 계산 자원을 테스트 시간에 할당할 수 있도록 하였습니다. 이를 통해, Qwen2.5-32B-Instruct 모델을 s1K 데이터셋에서 supervised fine-tuning(SFT)하여 o1-preview 모델보다 최대 27% 향상된 성능을 보여주었습니다.

- **Performance Highlights**: s1-32B 모델은 테스트 타임 스케일링을 통해 Qwen2.5-32B-Instruct 모델의 성능을 극대화하였고, AIME24 챌린지에서 성능을 50%에서 57%로 향상시켰습니다. 연구진은 s1K 데이터셋의 중요성을 강조하며, 무작위 선택이나 길이가 긴 추론 추적을 기준으로 선택한 샘플들은 성능에 부정적인 영향을 미친다고 언급하였습니다. 최종적으로, 연구진은 데이터 선택의 섬세한 과정을 통해 테스트 타임 스케일링을 최적화하는 방법론을 제안하고 있습니다.



### Decoding-based Regression (https://arxiv.org/abs/2501.19383)
Comments:
          Google DeepMind Technical Report, 25 pages. Code can be found at this https URL

- **What's New**: 이번 연구에서는 언어 모델이 숫자 예측을 문자열로 디코딩하는 회귀 작업을 수행할 수 있는 이론적 기반을 제공하고 있습니다. 기존의 회귀 방법과 유사한 성능을 발휘하면서도, 다양한 분포를 캡처할 수 있는 유연성을 보여줍니다. 특히, 디코딩 기반 회귀 방법의 유용성을 강조하고, 이를 통해 효율적인 데이터 활용과 성능 개선을 이룰 수 있음을 발견했습니다.

- **Technical Details**: 회귀 모델의 성능은 입력(feature) x와 출력(output) y를 어떻게 처리하는가에 따라 달라지며, 연구에서는 LLM을 이용한 입력 벡터의 임베딩을 사용하고 있습니다. 디코딩 기반 회귀 방식은 전통적인 수치 모델링 방법과 달리, 정규화 없이도 ℝ 상의 임의의 숫자 분포를 근사할 수 있는 유연성을 제공합니다. 이러한 방식은 높은 훈련 데이터 필요성을 동반하지만, 적절한 설정에서 데이터 효율성을 인정받고 있습니다.

- **Performance Highlights**: 디코딩 기반 회귀 헤드는 다양한 표 형식의 회귀 작업에서 기존 포인트 헤드와 경쟁할 만한 성능을 보여줍니다. 특히, Gaussian 혼합 모델과 같은 밀도 추정 작업에서도 충분히 표현력 있는 성능을 발휘합니다. 이는 기존의 회귀 기법에 비해 데이터 효율성과 성능 면에서 유리함을 의미합니다.



### CoSTI: Consistency Models for (a faster) Spatio-Temporal Imputation (https://arxiv.org/abs/2501.19364)
Comments:
          20 pages, 5 figures, 13 tables

- **What's New**: 이번 연구에서는 다변량 시계열 결측값 대체(MTSI) 분야를 위한 새로운 모델인 CoSTI를 제안합니다. CoSTI는 Consistency Models (CMs)의 적응형 변화로, 결측값 대체 정확도를 높이면서도 추론 시간을 대폭 단축시킬 수 있습니다. 이는 실시간 애플리케이션에 더 적합하며, DDPMs와 유사한 성능을 보여줍니다. 이 연구는 생성적 대체 작업에서의 효율성과 정확성 간의 간격을 메우고 있습니다.

- **Technical Details**: CoSTI는 Consistency Training을 활용하여 MTSI 문제를 해결합니다. 기존의 DDPMs에 비해 최대 98%까지 대체 시간을 줄였으며, 이 과정에서 발생할 수 있는 성능 저하를 극복했습니다. CoSTI는 다양한 데이터 세트와 결측 데이터 시나리오에서 실험을 통해 그 효과를 검증하였습니다.

- **Performance Highlights**: CoSTI는 다양한 도메인에서 검증이 이루어졌으며, DDPM 기반 방법들과 유사한 성능을 유지하면서도 빠른 대체 시간이라는 이점을 제공합니다. 이로 인해, CoSTI는 집중치료실(ICU) 환자 모니터링, 교통 흐름 제어 등 빠른 데이터 처리가 필요한 분야에서 실제 활용 가능성을 높이고 있습니다.



### We're Different, We're the Same: Creative Homogeneity Across LLMs (https://arxiv.org/abs/2501.19361)
- **What's New**: 이 연구는 다양한 대형 언어 모델(LLMs)과 인간의 창의적인 응답을 비교하여 LLM을 창의적 도구로 사용할 때 나타나는 출시에 대한 동질성을 분석합니다. 대부분의 기존 연구는 특정 LLM만을 고려해왔으나, 이 연구는 여러 LLM이 얼마나 유사한 응답을 생성하는지 집중적으로 조사합니다. 연구 결과, LLM의 응답은 서로 유사한 경향이 강하며, 이는 인간의 창의성을 제한할 수 있는 새로운 우려를 제기합니다.

- **Technical Details**: 연구에서는 표준화된 창의성 테스트인 Alternative Uses Task, Forward Flow, Divergent Association Task를 사용하여 응답의 다양성을 측정했습니다. LLMs는 인체 중심의 심리학적 테스트에서 더 높은 점수를 기록하지만, LLM 응답끼리의 유사성은 인간 응답의 유사성보다 현저히 높으며, 이는 응답 구조나 기타 변수를 통제한 후에도 유지됩니다. 또한, LLM 시스템 프롬프트를 수정하여 높은 창의성을 유도하면 전반적인 LLM의 창의성과 LLM 간 응답 변동성이 약간 증가하지만, 인간 응답은 여전히 더 높습니다.

- **Performance Highlights**: 이 연구는 LLM이 창의성 테스트에서 인간을 초월하는 경향이 있음을 보여주지만, 이는 오히려 LLM 간의 유사성이 높다는 것을 반증하는 요인으로 작용합니다. LLM의 응답들은 고유한 창의적인 결과물로 보이지만 결국에는 상대적으로 동질적인 결과물로 귀결됩니다. 오늘날 가장 인기 있는 모델들이 높은 중복성과 동질성을 보인다면, 사용자는 오히려 다양성을 잃고 평균으로 수렴하는 결과를 초래할 수 있습니다.



### Do Large Multimodal Models Solve Caption Generation for Scientific Figures? Lessons Learned from SCICAP Challenge 2023 (https://arxiv.org/abs/2501.19353)
Comments:
          Accepted to TACL 2025

- **What's New**: SCICAP 데이터셋이 2021년에 출시된 이후, 연구자들은 과학적 도표에 대한 캡션 생성에서 상당한 발전을 이루었다. 2023년 첫 SCICAP 챌린지가 열리면서 전 세계의 팀들이 확장된 SCICAP 데이터셋을 사용해 다양한 유형의 도표에 대한 캡션 생성 모델을 개발하도록 초대하였다. 연구 결과, 전문 편집자들은 GPT-4V가 생성한 캡션을 모든 다른 모델과 저자들의 원본 캡션보다 선호했다.

- **Technical Details**: 이 논문은 2023년 SCICAP 챌린지에 대한 개요와 데이터, 과정, 우승 팀 및 모델에 대한 상세한 정보를 제공한다. 구체적으로, 8개 도메인과 5개 도표 유형의 476,389개의 단일 패널 도표를 포함하는 확장된 SCICAP 데이터셋을 바탕으로 팀들은 캡션 생성 모델을 개발했다. 자동 및 인간 평가를 통해 GPT-4V와 다른 오픈 LMM들(대형 다중모달 모델)의 성능을 비교하였다.

- **Performance Highlights**: 인간 평가 결과, 기술적 학술 작문에 익숙한 세 명의 전문 편집자는 GPT-4V가 생성한 캡션을 모든 다른 모델의 결과보다 명확히 선호하였다. 이러한 주요 발견에 따라, 우리는 편집자들이 GPT-4V를 선호하는 이유를 조사하였다. 우리의 결론은 현재 LMMs가 과학적 도표 캡션 생성 문제를 완전히 해결하지는 못했지만, 많은 개선이 이루어졌음을 보여주었다.



### Pathological MRI Segmentation by Synthetic Pathological Data Generation in Fetuses and Neonates (https://arxiv.org/abs/2501.19338)
Comments:
          30 pages, 4 figures, 5 tables

- **What's New**: 이번 연구에서는 Fetal&Neonatal-DDPM이라는 새로운 diffusion model framework를 소개하여, semantic label images를 기반으로 고품질의 합성 병리적(fetal and neonatal) MRI 이미지를 생성하는 방법을 제안합니다.

- **Technical Details**: 연구는 건강한 label 이미지를 형태학적(morphological) 변경을 통해 수정하여, 심실 확장증(ventriculomegaly), 소뇌 및 교뇌 발달부전(cerebellar and pontocerebellar hypoplasia), 그리고 소두증(microcephaly)과 같은 질환을 시뮬레이션할 수 있는 데이터를 생성합니다. 이 방법을 통해 생성된 합성 MRIs는 실제 MRI 이미지와 비교했을 때 품질과 진단 가치에서 유의미한(p < 0.05) 향상을 보였습니다.

- **Performance Highlights**: 합성된 병리 데이터는 nnUNet의 segmentation 성능을 크게 향상시켰으며, 특히 심각한 심실 확장증 사례에서 두드러진 성과를 보였습니다. 심실(segmentation) 분야의 Dice 점수가 0.9253로, 기존의 0.7317과 비교하여 상당한 개선이 이루어졌습니다.



### What is causal about causal models and representations? (https://arxiv.org/abs/2501.19335)
Comments:
          50 pages

- **What's New**: 이 논문에서는 인과적 Bayesian 네트워크의 행동을 개입(intervention)으로 해석하는 정식 프레임워크를 제시합니다. 저자들은 그러한 해석이 순환적(circular)이며, 모든 인과적 Bayesian 네트워크가 관찰 분포(observational distribution)를 올바르게 모델링한다면 자연히 개입적으로도 유효하다고 주장합니다. 이러한 통찰은 인과 모델에서 행동과 그 효과의 예측에 대한 명확성을 제공합니다.

- **Technical Details**: 논문은 인과적 Bayesian 네트워크를 사용하여 수학적 모델과 실제 세계를 명확히 연결 짓기 위한 기준을 세움으로써 혼란을 줄이고자 합니다. 특히, 개입이란 개념이 수학적 모델에서 실제 행동에 대한 통찰을 끌어내는데 있어 어떤 한계가 있는지를 강조합니다. 아울러, 개입의 의미를 명확히 하기 위해 제안된 비순환(non-circular) 해석을 통해 인과 모델의 검증 가능성을 제고할 수 있는 방법에 대해 논의합니다.

- **Performance Highlights**: 제안된 프레임워크는 인과적 표현 학습(causal representation learning), 인과 발견(causal discovery), 그리고 인과 추상화(causal abstraction)의 개념적 기초에 기여하여, 기존 접근 방식의 한계를 드러냅니다. 저자들은 이러한 연구가 인과적 Bayesian 네트워크를 단순한 수학적 객체가 아니라 실제 세계의 인과 모델로 이해하는 데 중요한 기여를 할 것이라고 주장합니다. 논문의 내용은 인과 모델의 해석 가능성과 유효성 검증을 위한 기초를 다지는 데 있어 큰 의미가 있습니다.



### Capturing Temporal Dynamics in Large-Scale Canopy Tree Height Estimation (https://arxiv.org/abs/2501.19328)
Comments:
          9 pages main paper, 5 pages references and appendix, 8 figures, 5 tables

- **What's New**: 이 논문은 기후 변화에 대한 국가 정책 수립을 지원하기 위해 포괄적이고 정확한 산림 생태계 모니터링이 필요함을 강조합니다. 기존의 연구와 달리, 유럽 대륙의 2019년부터 2022년까지 10m 해상도의 임시 캐노피 높이 지도를 처음으로 생성하여, 계절적 변화를 명확히 추적할 수 있는 모델을 제안합니다. 이 모델은 Sentinel-2 위성 데이터를 활용하여 나무 캐노피 높이를 예측하며, GEDI LiDAR 데이터를 사실 기반으로 사용하여 높은 정확성을 자랑합니다.

- **Technical Details**: 논문에서 제시하는 알고리즘은 Sentinel-2 위성 이미지 데이터를 사용하여 나무 캐노피 높이를 예측하는 맞춤형 딥러닝 접근법에 기반합니다. 이 모델은 기존의 단일 이미지 처리가 아닌 연속된 12개월의 위성 데이터 시리즈를 활용하여 계절적 패턴을 포착하고, 공간적 정확성을 향상시키는 방법을 적용합니다. 또한, 모델의 파라미터와 전체 파이프라인은 GitHub에 공개되어 있어 연구자들이 이를 재현하고 확장하는 데에 용이합니다.

- **Performance Highlights**: 2020년 동안의 유럽 캐노피 높이 지도는 이전 연구들보다 구체적이고 정밀한 측정을 제공합니다. 논문에 따르면 이 연구는 2019-2022년 동안의 산림 성장과 감소를 일관되게 탐지할 수 있는 능력을 보여주며, 정확한 대규모 산림 구조 평가를 가능하게 합니다. 이와 같은 정밀도는 기후 변화 완화 노력에 기여하고, 지속 가능한 산림 관리 및 생태계 보전을 위한 정보 제공의 중요성을 강조합니다.



### Reward-Guided Speculative Decoding for Efficient LLM Reasoning (https://arxiv.org/abs/2501.19324)
Comments:
          17 pages

- **What's New**: 이 논문에서는 Reward-Guided Speculative Decoding (RSD)라는 새로운 프레임워크를 소개하여 대형 언어 모델(LLM)의 추론 효율성을 개선하는 데 중점을 두고 있습니다. RSD는 경량의 드래프트 모델과 더 강력한 목표 모델을 결합하며, 높은 보상을 주는 출력을 우선시하기 위해 통제된 편향을 통합합니다. 이는 기존의 스펙큘레이티브 디코딩 방식과 차별화되며, computational cost와 output quality 간의 최적의 균형을 유지하는 방법을 제시합니다.

- **Technical Details**: RSD는 드래프트 모델이 생성한 후보 출력 단계들을 평가하기 위해 프로세스 보상 모델을 사용합니다. 이 모델은 각 디코딩 단계에서 동적으로 목표 모델을 호출할지를 결정하여, 계산 비용과 출력 품질 간의 trade-off를 최적화합니다. RSD는 작업 흐름을 수동 조정하여 불필요한 계산을 줄이고, 기존 추론 접근 방식을 넘어서는 품질을 제공합니다.

- **Performance Highlights**: 여러 도전적인 추론 벤치마크에서 RSD의 효율성은 현실적으로 4.4배 더 적은 FLOPs를 요구하는 등의 놀라운 효율을 보여주며, 평균적으로 3.5점 향상된 정확도를 기록했습니다. 이러한 결과는 RSD가 리소스가 많이 소모되는 환경에서 LLM을 배포하기 위한 강력하고 비용 효율적인 접근법임을 강조합니다.



### Language Bias in Self-Supervised Learning For Automatic Speech Recognition (https://arxiv.org/abs/2501.19321)
Comments:
          Accepted to Speech and Language Technology Workshop (SLT) 2024 accessible on IEEE Xplore

- **What's New**: 이 논문에서는 다국어 음성 인식(ASR) 모델 XLS-R의 자기 지도 학습(self-supervised learning, SSL)의 언어 편향을 분석하고, 언어별 서브네트워크(subnetwork)를 식별하여 성능에 미치는 영향을 평가합니다. 기존 연구에서는 SSL ASR의 언어 편향에 관한 심층적인 검토가 부족했으나, 본 연구는 언어 불균형이 SSL 모델의 성능에 미치는 영향을 규명하는 데 초점을 맞추고 있습니다.

- **Technical Details**: XLS-R 모델은 128개 언어로 훈련된 ASR 모델로, 300M 매개변수(parameter)로 구성됩니다. 이 논문에서 사용된 FLEURS 데이터셋은 101개 언어의 음성 데이터를 포함하고 있으며, 언어별로 약 12시간의 데이터가 제공됩니다. 연구자들은 Lottery Ticket Hypothesis (LTH)를 활용하여 XLS-R의 언어별 서브네트워크를 파악하고, 이는 데이터의 언어 불균형이 성능에 미치는 영향을 명확히 하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, XLS-R 모델은 유사한 데이터 세트를 기반으로 학습한 다국어 음성 인식에서 전통적인 언어 지식에 의존하지 않고, 가장 데이터 기여도가 높은 언어에서 학습된 가중치에만 의존함이 드러났습니다. 이를 통해, 특정 언어로의 전이학습(transfer learning) 시, 서브네트워크의 구조가 특정 언어에 대해 성능이 다르게 나타나는 것을 확인했습니다. 이러한 결과는 SSL ASR 모델의 훈련 및 성능 개선을 위한 새로운 방향을 제시합니다.



### Beyond checkmate: exploring the creative chokepoints in AI tex (https://arxiv.org/abs/2501.19301)
Comments:
          18 pages, single columns, under review at Nature Machine Intelligence

- **What's New**: 이번 연구는 인간 텍스트와 AI 텍스트 간의 미묘한 차이를 언급하며, 기존의 AI 텍스트 탐지 연구에 비해 상대적으로 덜 탐색된 영역을 다룹니다. 특히, 텍스트의 구성 요소(예: 서론, 본문, 결론)에 따라 인간과 AI 텍스트 간의 차이를 분석하고, 이러한 차이들이 창의적 보조 도구로서 AI의 잠재력을 평가하는 데 미치는 영향을 조명합니다. 연구는 체스 게임의 구조와 비유를 사용하여 각 텍스트 구성 요소에서 확인된 차이를 구체적으로 설명합니다.

- **Technical Details**: 연구에서 뉴스 기사, 이메일, 수필의 세 가지 도메인에서 AI 텍스트와 인간 텍스트 간의 차이를 분석했습니다. 여러 텍스트 세그먼트 간의 특징을 비교하기 위해 데이터셋 구축, 텍스트 세분화, 특징 추출 및 통계적 분석 절차를 적용했습니다. 본문 세그먼트는 길이가 길어 AI 텍스트의 일관성을 높일 수 있지만, 단어 다양성 및 맥락 적합성과 같은 특정 특성에서는 여전히 인간 텍스트와의 차이가 두드러지며, 이러한 분석 결과는 해당 논문 내에서 제시된 표를 통해 요약됩니다.

- **Performance Highlights**: 검출 성능에 대한 결과는 본문 세그먼트에서 AI 텍스트 탐지의 효과가 더 높다는 것을 시사하며, 이는 탐지기가 AI 텍스트와 인간 텍스트의 차이를 구분하는 데 있어 본문 세그먼트가 더 두드러진다는 것을 보여줍니다. 또한, 텍스트 길이에 따른 탐지 성능 차이를 분석한 결과, 길이가 길어질수록 잘 탐지되는 경향을 보였습니다. 최종적으로 이 연구는 AI 텍스트 탐지의 발전 방향을 제시하고, 세그먼트 간의 특징 변동성이 인간과 AI 텍스트 간의 차별화된 지표로서 유망하다고 평가하였습니다.



### Analysis of LLMs vs Human Experts in Requirements Engineering (https://arxiv.org/abs/2501.19297)
Comments:
          8 pages, 15 figures

- **What's New**: 이 연구는 소프트웨어 개발에서 대형 언어 모델(LLM)의 활용 방안을 다루며, 이들 모델이 요구 사항 공학(requirements engineering) 분야에 미치는 영향을 분석하고 있습니다. 특히, 시스템의 요구 사항을 발견하고 문서화하는 요구 사항 수집(requirements elicitation)에 초점을 맞추었습니다.

- **Technical Details**: 이 연구에서는 LLM과 인간 전문가간의 요구 사항 수집 능력을 비교하는 시간 제한(time-boxed) 및 프롬프트 제한(prompt-boxed) 연구를 수행했습니다. 결과적으로 LLM이 생성한 요구 사항은 인간이 생성한 요구 사항보다 더 정렬되어(+1.12) 평가되었으며, 완전성 측면에서도 더 우수한 경향을 보였습니다(+10.2%).

- **Performance Highlights**: LLM이 생성한 문서는 평균적으로 인간 전문가의 720배 빠른 속도로 작업을 수행하며, 그 비용은 평균적으로 인간 전문가의 0.06%에 불과했습니다. 이러한 결과는 LLM이 요구 사항 공학에서 요구 사항 정의 개선, 자원 할당의 효율성 향상, 및 전체 프로젝트 일정 단축에 중요한 역할을 할 것임을 시사합니다.



### Linear $Q$-Learning Does Not Diverge: Convergence Rates to a Bounded S (https://arxiv.org/abs/2501.19254)
- **What's New**: 이 논문은 선형 Q-학습(linear Q-learning)의 수렴 속도에 대한 새로운 결과를 제시합니다. 선형 Q-학습은 이전에 발산 가능성이 있다고 여겨졌으나, 이 연구는 적응 온도를 가진 ε-softmax 행동 정책을 사용할 경우 발산하지 않고 제한된 집합으로 수렴함을 보여줍니다. 또한, 원래의 선형 Q-학습 알고리즘에 대한 수정 없이도 이러한 결과를 도출하였습니다.

- **Technical Details**: 선형 Q-학습에 대한 첫 번째 L^2 수렴 속도를 확립하였으며, 이는 빠르게 변화하는 전이 함수 아래에서 마르코프 노이즈의 일반적인 수렴 결과에 기반하고 있습니다. 기존 연구와 달리, 이 논문에서는 타겟 네트워크나 경험 재생과 같은 추가적인 가정을 하지 않았습니다. 또한, 가정 중 Bellman 완전성이나 행동 정책의 근사 최적성 가정을 하지 않아도 수렴성을 보장합니다.

- **Performance Highlights**: 이론적으로 설명된 새로운 수렴 속도 덕분에 선형 Q-학습과 표 형태 Q-학습 모두에서 성과를 보였습니다. 특히, 새로운 의사 수축 속성(pseudo-contraction property)을 활용하여 표 형태 Q-학습에 대한 L^2 수렴 속도도 증명하였습니다. 발전된 성과는 기존 연구과 비교하여 상당히 개선된 수치를 보여주고 있으며, 안정성과 효율성을 더욱 높였습니다.



### A Zero-Shot Generalization Framework for LLM-Driven Cross-Domain Sequential Recommendation (https://arxiv.org/abs/2501.19232)
Comments:
          11 pages

- **What's New**: 본 연구에서는 Zero-shot cross-domain sequential recommendation (ZCDSR) 문제를 다루고 있으며, 추가 훈련이나 파인튜닝 없이도 보이지 않는 도메인에서의 예측을 가능하게 합니다. 최근 대형 언어 모델(LLMs)의 발전을 통해 ZCDSR의 성능이 크게 향상되었으며, 이는 크로스 도메인 지식 전이를 촉진하는 데 도움을 주고 있습니다. 그러나 도메인 간 어휘 및 콘텐츠 초점의 차이로 인해 발생하는 도메인 의미적 편향(domain semantic bias)이라는 주요 문제가 여전히 존재합니다.

- **Technical Details**: 이 연구에서는 제안된 프레임워크가 아이템 레벨과 시퀀스 레벨 모두에서 LLM 기반 ZCDSR의 도메인 정렬을 개선하는 것을 목표로 하고 있습니다. 아이템 레벨에서는 유사한 아이템의 임베딩을 도메인 간 정렬하여 인터 도메인 콤팩트함을 촉진하는 일반화 손실(generalization loss)을 도입하였습니다. 시퀀스 레벨에서는 소스 도메인의 사용자 행동 패턴을 클러스터링하고, 타겟 도메인 추론을 위한 주의 기반 집계를 적용하는 방식을 개발하여 사용자 임베딩의 동적 적응성을 제공합니다.

- **Performance Highlights**: 다양한 데이터 세트와 도메인에서 진행된 종합적인 실험 결과, 제안된 프레임워크가 ZCDSR 환경에서 시퀀스 추천 성능을 상당히 향상시키는 것으로 나타났습니다. 도메인 편향을 완화하고 시퀀스 패턴의 전이 가능성을 강화함으로써, 본 방법은 도메인 간의 보다 효과적인 제로샷 recommandations을 위한 확장 가능하고 강력한 접근 방안을 제공합니다.



### Integrating Semi-Supervised and Active Learning for Semantic Segmentation (https://arxiv.org/abs/2501.19227)
- **What's New**: 이번 논문에서는 새로운 액티브 러닝 접근 방식을 제안하며, 이는 개선된 준지도 학습 프레임워크와 통합되어 매뉴얼 주석 비용을 줄이고 모델 성능을 향상시킵니다. 이 접근 방식은 액티브 러닝을 통해 선택된 레이블이 있는 데이터와 선택 과정에서 제외된 레이블이 없는 데이터를 동시에 활용하여, 잘못된 가짜 레이블(pseudo-labels)의 가능성이 있는 영역을 정확히 지적합니다.

- **Technical Details**: 제안된 구조에서는 주석 예산을 증가시키지 않으면서, 군(cluster) 가정을 기반으로 하여 레이블이 없는 데이터에서 가장 어려운 부분만 매뉴얼 레이블링이 적용됩니다. 또한, 가짜 레이블 자동 정제(pseudo-label auto-refinement, PLAR) 모듈을 통해 잘못된 가짜 레이블을 가진 픽셀을 교정하며, 이 과정에서 특징(feature) 표현을 기반으로 정확성과 효율성을 높입니다.

- **Performance Highlights**: 제안된 혼합 준지도 액티브 러닝 프레임워크는 두 가지 벤치마크 데이터셋에서 평가되었으며, 자연 및 원격 측정 이미지 도메인에서 최고의 성능을 기록하였습니다. 또한 이 방법은 semantic segmentation 작업에서 최신의 방법론을 초과하는 성과를 보여주었습니다.



### Strassen Attention: Unlocking Compositional Abilities in Transformers Based on a New Lower Bound Method (https://arxiv.org/abs/2501.19215)
- **What's New**: 본 연구는 Transformer 모델의 이론적 한계를 평가하는 새로운 방법을 제안합니다. 특히, 우리는 무한 정밀도를 가진 일층 소프트맥스 Transformer에 대한 최초의 하한을 공식적으로 증명했습니다. 세 가지 고급 추론을 요구하는 작업에서 이러한 하한을 확립하였으며, 이 작업들은 세 위치의 조합, 함수의 조합, 이진 관계의 조합을 포함합니다.

- **Technical Details**: 새로운 기술인 Strassen attention을 소개하여, 이를 통해 일층 Transformer가 모든 작업을 해결할 수 있다는 이론적 근거를 제시합니다. Strassen attention은 시간 복잡도가 서브 큐빅(sub-cubic)으로 더 스케일러블한 특징을 가지며, 기존의 높은 차원 attention 메커니즘과 비교하여 더 효율적입니다. 또한 우리는 Strassen attention을 실험적으로 연구하고, 표준 attention 및 다른 고차원 attention과 비교한 결과를 보여줍니다.

- **Performance Highlights**: 연구 결과, Strassen attention은 모든 작업에서 표준 attention에 비해 상당히 우수한 성능을 발휘하였습니다. 이 결과는 다양한 attention 메커니즘의 장점과 단점을 명확히 구분할 수 있도록 도와줍니다. 이론적 한계를 이해함으로써, 나아가 Transformers의 추론 능력을 향상시킬 수 있는 스케일러블한 attention 메커니즘 연구가 가능할 것입니다.



### Single cell resolution 3D imaging and segmentation within intact live tissues (https://arxiv.org/abs/2501.19203)
- **What's New**: 이번 연구에서는 다양한 구조의 상피세포(epithelial cells)가 높은 해상도의 심층 이미징(deep imaging)과 컴퓨터 기반 기법을 통해 3D 구조적 특징을 정확하게 정량화하는 방법을 소개합니다. 특히, 생체 조직에서 형광으로 표지된 개별 세포를 3D로 정확히 계량하기 위한 프로토콜(protocol)을 단계별로 제시합니다. 이 연구는 Drosophila wing discs의 3D 이미징에서 발생한 문제를 해결하기 위한 경험을 공유하며, 다양한 이미징 모달리티 선택 및 설정에 대한 고려사항도 포함하고 있습니다.

- **Technical Details**: 세포를 정확하게 분할(segmentation)하기 위한 딥러닝 보조 기술을 포함한 이미징 및 샘플 준비 방법론이 자세히 설명됩니다. 이 프로토콜은 세포막 표지를 통한 세포 윤곽(segmentation)만을 목표로 하지만, 복잡한 3D 분석이 요구되는 다양한 샘플에 적용 가능하다는 것이 특징입니다. 또한, 프로토콜 복제를 지원하기 위한 컴퓨터 파이프라인(computational pipeline)과 맞춤형 코드(custom code)가 포함되어 있어 연구자들이 실험을 반복하기 쉽게 돕습니다.

- **Performance Highlights**: 제공된 프로토콜은 생체 조직에서 세포 특성을 정량화하기 위한 정교한 접근 방식을 제시하여, 상피세포의 다양한 특성과 구조를 정확히 분석할 수 있게 합니다. 이 연구는 고해상도 이미징 기술을 통한 세포 구조 연구에 기여하며, 앞으로 다른 조직 연구에도 활용될 수 있을 것으로 기대됩니다. 세포 분석법의 발전을 통해 다양한 생물학적 현상에 대한 깊은 통찰을 제공할 것입니다.



### Efficient Reasoning with Hidden Thinking (https://arxiv.org/abs/2501.19201)
Comments:
          Preprint version

- **What's New**: 최근 MLLMs(Multimodal Large Language Models)의 인기가 높아짐에 따라, Chain-of-Thought (CoT) 추론을 활용하여 복잡한 문제 해결 능력을 향상시키려는 연구가 증가하고 있습니다. 본 논문에서는 $	extbf{Heima}$라는 효율적인 추론 프레임워크를 제안하고, 이를 통해 CoT를 숨겨진 잠재 공간에서 활용하여 텍스트 표현의 장황함을 피하고자 했습니다. Heima Encoder는 각 CoT를 단일 사고 토큰(single thinking token)으로 압축하여 생성되는 토큰 수를 줄이고, 이를 통해 MLLMs의 효율성을 높이고 있습니다.

- **Technical Details**: Heima 프레임워크는 CoT를 compact hidden representations로 인코딩하는 Heima Encoder를 포함하고 있습니다. 이 과정에서 담기 언어 모델(LLM)을 사용하여 사고 토큰을 생성하며, 이들 토큰은 Heima Decoder를 통해 다양한 길이의 텍스트로 해석됩니다. 이러한 장점을 통해, Heima Encoder는 텍스트 CoT 대신 사고 토큰을 생성하여 추론 프로세스를 가속화함으로써 효율성을 크게 높입니다.

- **Performance Highlights**: 실험 결과, Heima 모델은 기존의 MLLM에 비해 생성 효율성을 높이면서도 제로-샷(zero-shot) 작업 정확도를 빈틈없이 유지하거나 심지어 강화하는 것으로 나타났습니다. Heima Decoder에 의해 재구성된 추론 과정은 원래의 CoT와 밀접하게 일치하여, 이 접근 방식의 견고성과 해석 가능성을 입증합니다. 마지막으로, 기존의 효율적인 기술(KV cache optimization 및 flash attention)과도 호환되며, Heima는 MLLM을 위한 첫 번째 추론 가속 프레임워크로 자리잡을 가능성이 높습니다.



### Rethinking Early Stopping: Refine, Then Calibra (https://arxiv.org/abs/2501.19195)
- **What's New**: 이 논문은 머신러닝 분류기의 확률적 예측 품질을 평가하기 위해 기존의 손실 함수를 통해 세분화된 캘리브레이션(calibration) 오류와 정제(refinement) 오류를 제안합니다. 연구팀은 이러한 두 가지 오류를 동시에 최소화하는 것이 어렵다는 것을 이론적 및 실증적 증거를 통해 보여줍니다. 이로 인해, 캘리브레이션 오류와 정제 오류를 각각 훈련 중과 후에 최소화하는 새로운 접근법을 소개합니다.

- **Technical Details**: 저자들은 두 가지 오류, 즉 캘리브레이션 오류와 정제 오류를 정확히 정의하고 이들을 손실 함수 손실로부터 분리하는 새로운 변분적 공식화(variational formulation)를 제시합니다. 각 클래스에 대한 예측의 확률을 계산하고, 이 예측을 기반으로 손실 기능을 평가하는 과정을 형식화하여, 적절한 정제 손실을 찾을 수 있는 방법을 제공합니다. 제안하는 새로운 정제 추정기는 다양한 머신러닝 모델에 쉽게 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 분류 작업에서 새로운 접근법이 성능 향상을 보인다는 경험적 증거를 제시하였습니다. 논문은 심지어 간단한 로지스틱 회귀 모델에서도 캘리브레이션 오류와 정제 오류가 존재하며, 이러한 오류를 최소화하는 새로운 기법이 효과적임을 보여줍니다. 이 연구는 머신러닝 모델의 학습 및 평가 방식을 개선하여 보다 신뢰할 수 있는 예측을 가능하게 합니다.



### Secured Communication Schemes for UAVs in 5G: CRYSTALS-Kyber and IDS (https://arxiv.org/abs/2501.19191)
Comments:
          6 pages, 5 figures, Paper accepted at IEEE FNWF'25 conference (References number: 1571070613)

- **What's New**: 이 논문은 UAV(무인 항공기)와 지상국 간의 안전한 통신 아키텍처를 제안하며, 5G 네트워크에서의 보안 문제를 다룹니다. AES(Advanced Encryption Standard)와 ECC(Elliptic Curve Cryptography), CRYSTALS-Kyber를 통합한 하이브리드 암호 방식으로 양자 공격에 대한 저항력을 제공합니다. 시스템은 클라이언트-서버 모델을 기반으로 하며, AI 기반의 침입 탐지 시스템을 통합하여 보안을 더욱 강화합니다.

- **Technical Details**: 제안된 아키텍처는 AES와 ECC, CRYSTALS-Kyber를 결합하여 양자 저항 암호화를 제공합니다. 이 시스템은 5G 환경에서 안전한 데이터 전송을 보장하며, 클라이언트인 UAV와 서버인 지상국 간의 강력한 연결을 유지합니다. 데이터 전송 중에는 KEM(Key Encapsulation Mechanism)을 통해 키가 교환되고, 인공지능을 활용한 IDS(Intrusion Detection System)가 통신 패턴을 분석하여 잠재적인 침입을 탐지합니다.

- **Performance Highlights**: 실험 결과, CRYSTALS-Kyber는 양자 위협에 대해 강력한 보호를 제공하면서도 최소한의 성능 부담을 나타냅니다. AI 기반의 IDS는 XGBoost 모델을 사용하여 97.33%의 정확도를 기록하며, 뛰어난 성능을 입증했습니다. 이는 UAV 네트워크의 고성능 요구에 부합하는 내구성 있고 확장성 있는 통신 프레임워크의 가능성을 보여줍니다.



### Enhancing Model Defense Against Jailbreaks with Proactive Safety Reasoning (https://arxiv.org/abs/2501.19180)
- **What's New**: 이번 연구에서는 기존의 방어 전략들과는 근본적으로 다른, 새로운 방어 메커니즘인 Safety Chain-of-Thought (SCoT)를 제안합니다. SCoT는 해로운 입력을 사전 차단하는 것에 그치지 않고, LLM의 향상된 reasoning capabilities를 이용하여 주의 깊은 검토를 통해 응답의 적절성을 평가합니다. 이는 LLM이 다양한 위험한 질문들과 예측하기 어려운 시나리오에 대해 더 나은 일반화 능력을 갖도록 합니다.

- **Technical Details**: SCoT는 요청의 의도를 분석한 후 적절한 대답을 제공하는 방식으로 LLM의 반응 과정을 재구성합니다. 이 방법은 사전 정의된 위반 사항으로 요청을 분류할 수 있도록 하여, 안전 교육 데이터셋에 명시적으로 포함되지 않은 다양한 해로운 질문에 대해 효과적으로 일반화할 수 있습니다. 기존의 방어 메커니즘과 비교해, SCoT는 흑백 공격 유형을 포함하여 전방위적인 공격 시나리오에서 더 높은 회복력과 응답 정확성을 자랑합니다.

- **Performance Highlights**: SCoT는 기존의 방어 전략들을 능가하는 것으로 평가되며, 특히 공격 성공률이 거의 0에 가까운 성과를 보입니다. 또한, SCoT는 다양한 표준 운영 시나리오에서 강력한 일반 능력을 유지하며, 진화하는 보안 도전 과제에 대한 적응력을 강조합니다. 이 연구는 새로운 방어 접근법과 함께, AI 안전 커뮤니티 내 협력을 촉진하기 위한 모델과 자료를 공개했습니다.



### Augmented Intelligence for Multimodal Virtual Biopsy in Breast Cancer Using Generative Artificial Intelligenc (https://arxiv.org/abs/2501.19176)
- **What's New**: 본 연구에서는 Full-Field Digital Mammography (FFDM)와 Contrast-Enhanced Spectral Mammography (CESM)를 활용한 멀티모달 심층 학습 접근법을 제안합니다. FFDM과 CESM을 통합하여 가상 생검(virtual biopsy)을 수행함으로써 종양을 악성으로 분류하는 데 도움을 줍니다. CESM 데이터를 누락할 경우, 생성적 인공지능을 활용하여 FFDM 스캔으로부터 CESM 이미지를 보간(impute)하는 방법을 사용합니다. 이는 FFDM 단독 사용에 비해 더 높은 성능을 보여주었습니다.

- **Technical Details**: 연구는 FFDM 및 CESM 영상을 크레니오카우달(craniocaudal) 및 메디오레터럴 오블리크(mediolateral oblique) 뷰에서 레지스터(registra)하여, 높은 진단 정확도를 위해 머신러닝 기법을 이용하는 과정을 포함합니다. CESM 이미지를 보간하기 위한 생성적 인공지능 기술을 활용하여, FFDM만으로 이루어진 경우에 비해 더 과학적이고 정확한 결과를 도출할 수 있습니다. 이 접근 방식은 새로운 데이터셋을 공개하여 연구 커뮤니티에 기여하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, CESM 모듈을 통합함으로써 가상 생검의 성능이 크게 향상되었음을 보여주었습니다. 실제 CESM 데이터가 누락된 경우, 합성 CESM 이미지를 활용하는 것이 FFDM 단독으로 사용하는 것보다 효과적이라는 것을 입증하였습니다. 특히, FFDM과 CESM을 결합한 멀티모달 구성을 사용했을 때 성능이 더욱 두드러졌습니다. 따라서, 본 연구는 진단 정확도를 개선하는 데 기여하며 임상 의사들에게 유용한 도구를 제공합니다.



### SWAT: Sliding Window Adversarial Training for Gradual Domain Adaptation (https://arxiv.org/abs/2501.19155)
Comments:
          submitted to icml 2025

- **What's New**: 본 논문은 Sliding Window Adversarial Training (SWAT)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 기존의 Gradual Domain Adaptation (GDA) 방법의 한계를 극복하고, 점진적으로 지식 전이를 수행하여 도메인 간의 격차를 줄입니다. 특히, SWAT는 적대적 흐름을 구축하여 중간 도메인 간의 작은 간격을 점차적으로 좁히는 슬라이딩 윈도우 메커니즘을 활용합니다.

- **Technical Details**: SWAT는 적대적 학습(adversarial learning)과 커리큘럼 슬라이딩 윈도우(curriculum sliding window) 메커니즘을 결합하여 모델을 동적으로 조정된 중간 도메인에 맞춥니다. 이 방법은 sliced Wasserstein 최적화(sliced Wasserstein optimization)를 통해 도메인 연속성을 강화하고, 적대적 감쇠(adversarial attenuation)를 통해 소스 특성을 점진적으로 필터링합니다. 그 결과, 도메인 불변성과 목표 구별력이 동시에 향상됩니다.

- **Performance Highlights**: SWAT는 Rotated MNIST(원판 MNIST)과 Portraits 데이터셋에서 현재 최첨단 모델 대비 각각 10.3% 및 1.24% 향상된 성능을 기록하였습니다. 또한, 극단적인 도메인 변화에서 36.0%의 오류 감소를 보여주는 실험 결과가 수록되어 있습니다. 이러한 성과는 SWAT의 접근 방식이 기존 방식에 비해 우수하다는 것을 입증합니다.



### On the inductive bias of infinite-depth ResNets and the bottleneck rank (https://arxiv.org/abs/2501.19149)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 깊은 선형 ResNet 아키텍처의 최소 노름(weights) 가중치를 계산하였으며, 이 구조의 유도 편향(inductive bias)이 핵심적으로 핵 노름(nuclear norm) 최소화와 계급(rank) 최소화 사이에 위치한다는 것을 발견했습니다. 이는 적절한 하이퍼파라미터를 사용할 경우, 깊은 비선형 ResNet이 병목(rank) 최소화를 지향하는 유도 편향을 가지게 됨을 의미합니다.

- **Technical Details**: 연구는 깊은 선형 ResNet의 두 가지 변형을 사용하여 유도 편향을 탐구하였습니다. 각 층에 MLP(multilayer perceptrons)를 포함하는 잔여 네트워크(residual networks)는 선형 함수만 표현할 수 있으며, 이는 활성화 함수가 선형이라는 사실에 근거합니다. 그 결과, 이 연구는 이들 네트워크가 병목 계급(bottleneck rank)을 최소화하는 선형 변환을 학습하도록 유도된다는 것을 확인하였습니다.

- **Performance Highlights**: 실제로, 이 연구는 깊은 비선형 ResNet 구조가 특정 하이퍼파라미터 환경에서 저병목 계급(low bottleneck rank) 솔루션을 선호함을 보였습니다. 연구의 결과는 ResNet 아키텍처가 아이덴티티 변환을 단순화하는 기능을 가지고 있지만, 이들이 여전히 생략된 병목 계급을 통해 일반화하는 방식을 이해하는 데 중요한 통찰력을 제공한다는 점에서 실용적입니다.



### Improving Multi-Label Contrastive Learning by Leveraging Label Distribution (https://arxiv.org/abs/2501.19145)
- **What's New**: 이번 연구에서는 다중 라벨 학습에서 대조 학습(contrastive learning)을 통해 더 나은 표현을 학습하는 방안을 제안합니다. 기존 방법들은 라벨 간의 중첩을 기반으로 긍정 및 부정 샘플을 선택하였으나, 복잡한 선택 과정과 다양한 라벨의 중요도를 무시하는 문제에 직면해 있었습니다. 이를 해결하기 위해, 우리는 라벨 분포(label distribution)를 도입하여 다중 라벨 대조 학습을 개선하는 새로운 방법 MulSupConLDLD{}_{LD}를 제안합니다.

- **Technical Details**: 우리는 긍정 샘플 선택 시 ANY 전략을 채택하여 라벨이 교차하는지를 기준으로 삼습니다. 또한, 라벨 간의 관계를 모델링하기 위해 로지컬 라벨에서 라벨 분포를 복구하는 두 가지 방법(Radial Basis Function (RBF) 및 대조 손실(contrastive loss)을 기반으로 한 방법)을 도입했습니다. 이러한 접근 방식은 다중 라벨 데이터셋에서 모델의 일반화 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 우리의 방법은 아홉 개의 널리 사용되는 다중 라벨 데이터셋에서 평가되었으며, 여섯 가지 평가 지표에서 기존 최첨단 방법들을 초월하는 성능을 보였습니다. 이를 통해, MulSupConLDLD{}_{LD}는 더욱 차별적인 특징 표현을 학습함과 동시에 라벨 간의 의존성을 효과적으로 포착할 수 있음을 입증했습니다.



### A Metric for the Balance of Information in Graph Learning (https://arxiv.org/abs/2501.19137)
Comments:
          In proceedings of the 4th Annual AAAI Workshop on AI to Accelerate Science and Engineering (AI2ASE)

- **What's New**: 이 논문은 다루는 데이터 세트에 대한 정보 소스의 선호도를 평가하기 위해 Noise-Noise Ratio Difference (NNRD)라는 정량적 지표를 제안합니다. 이는 분자에서의 구조 정보와 특성 정보 간의 유용한 정보 비율을 측정하는 새로운 방법입니다. NNRD는 각 정보를 독립적으로 노이즈 처리하여 각 정보의 저하 정도를 측정합니다. 본 연구는 화학 및 다른 분야에서의 응용 가능성을 강조하고 있습니다.

- **Technical Details**: Graph Neural Networks (GNNs)와 Message Passing Neural Networks (MPNNs)의 원리에 기초하여, 연구에서는 데이터 세트의 구조와 특성을 동시에 고려해야 유용한 정보가 충족된다고 설명하고 있습니다. 블록 체인의 노이즈 처리 과정에서 X와 E의 정보를 통한 유용한 정보 수준을 비교하여, 각 정보의 상대적 중요성을 평가합니다. NNRD는 정보를 저하시켜가며 구조 정보와 특성 정보의 균형을 정량적으로 파악할 수 있도록 설계되었습니다.

- **Performance Highlights**: NNRD는 다양한 분자 작업에서 성능 저하와 잘 일치하는 결과를 보여줍니다. 이 지표는 단순한 성능 집계보다 더 직관적이고 표현력이 뛰어난 결과를 제공합니다. 앞으로의 연구 방향은 데이터 도메인 및 작업 유형의 확장, 그리고 기준 모델 선택의 개선에 초점을 맞출 예정입니다.



### Decorrelated Soft Actor-Critic for Efficient Deep Reinforcement Learning (https://arxiv.org/abs/2501.19133)
- **What's New**: 이번 연구에서는 강화학습(Resinforcement Learning, RL)의 샘플 효율성(sample efficiency)을 향상시키기 위한 새로운 접근법을 제안합니다. 이는 네트워크 전반의 디코릴레이션(decorrelation) 단계를 RL 훈련 파이프라인에 통합하여 온라인 디코릴레이션(decorrelated backpropagation, DBP) 알고리즘을 기반으로 합니다. 이를 통해 각 레이어에 디코릴레이션 행렬을 추가하고, 일반적인 RL 손실을 최소화하는 것과 병행하여 총 디코릴레이션 손실을 최소화하는 방법을 적용하였습니다.

- **Technical Details**: 주요 기술적 내용으로는 디코릴레이션 행렬은 각 레이어의 입력에 추가되어, 스스로의 디코릴레이션 학습 규칙을 적용하여 업데이트됩니다. 이 과정은 소프트 액터-크리틱(Soft Actor-Critic, SAC) 메소드와 결합되어, 우리가 '디코릴레이트 소프트 액터-크리틱(Decorrelated Soft Actor-Critic, DSAC)'이라고 부르는 방식으로 이용되었습니다. 이 알고리즘은 학습 효율성을 높이기 위해 입력을 디코릴레이션하고, 모든 레이어의 입력에 달려 있는 네트워크 전반의 디코릴레이션을 구현합니다.

- **Performance Highlights**: 아타리 100k 벤치마크에서 DSAC를 사용한 결과, 일반 SAC 기준과 비교할 때 7개 게임 중 5개에서 훈련 속도가 빨라졌고, 2개의 게임에서 보상 성능이 50% 이상 향상되었습니다. 이러한 성과는 강화학습의 샘플 효율성(sample efficiency)를 가속화하기 위한 네트워크 전반의 디코릴레이션 효과를 증명합니다. 결과적으로, 강화학습 알고리즘의 수렴속도를 증가시키고, 더 빠르고 효과적인 정책 학습을 가능하게 합니다.



### Shaping Sparse Rewards in Reinforcement Learning: A Semi-supervised Approach (https://arxiv.org/abs/2501.19128)
- **What's New**: 이 논문은 드물게 주어지는 보상 신호로 인해 발생하는 문제를 해결하기 위해, 비제로 보상 전이와 반자기 감독 학습(Semi-Supervised Learning, SSL) 기법을 활용하여 새로운 데이터 증강 방법을 적용하는 접근법을 제안합니다. 제안된 방법은 보상 설계를 개선하고, 드문 보상 시나리오에서의 효과적인 일반화를 이루어내며, 높은 점수를 달성하는 데 유리함을 보여줍니다. 특히, 더블 엔트로피 데이터 증강 기법을 통해 성능 향상이 두드러진 것으로 나타났습니다.

- **Technical Details**: 논문에서는, 드문 보상 문제를 해결하기 위해 반자기 감독 보상 설계 프레임워크(Semi-Supervised Reward Shaping, SSRS)를 제안합니다. 이는 제로 보상 전이를 활용하여 트라젝토리(trajectory) 공간 표현을 학습하고, 두 가지 구성 요소에 대해 단조성 제약을 추가하여 보상 분포 간의 차이를 줄이는 방법입니다. 데이터 증강 기법으로서 제안된 더블 엔트로피 방법은 정보를 정확히 표현하고 비영상 기반 작업에 적합하도록 설계되었습니다.

- **Performance Highlights**: 제안된 SSRS 모델은 아타리 및 로봇 조작 환경에서 기존의 호기심 주도 방법에 비해 최대 4배 더 높은 최상위 점수를 달성했습니다. 특히, 더블 엔트로피 데이터 증강을 통해 성능이 15.8% 향상되어, 전반적인 모델의 성능이 크게 개선된 것을 보여줍니다. 이로써 보상 설계 및 학습의 효율성이 크게 증대되었습니다.



### FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling (https://arxiv.org/abs/2501.19122)
- **What's New**: 이 논문에서는 Federated Robust pruning via combinatorial Thompson Sampling (FedRTS)라는 새로운 프레임워크를 제안하여 협업 모델 훈련의 효율성을 향상시킵니다. FedRTS는 기존의 비효율적인 동적 프루닝 방법들이 가지고 있는 한계를 극복하고, 강인한 희소 모델을 개발하는 데 중점을 두고 있습니다. 특히, FedRTS는 Myopic (단기적) 정보 대신 안정적인 정보에 기반한 확률적 결정을 통해 모델 토폴로지의 조정을 효과적으로 수행합니다.

- **Technical Details**: FedRTS는 Combinatorial Multi-Armed Bandit (CMAB) 문제로 Federated Pruning을 재구성합니다. 이 프레임워크는 Thompson Sampling을 기반으로 하여 클라이언트의 부분적 참여와 데이터 이질성을 효과적으로 처리할 수 있도록 설계되었습니다. 모델 조정의 요인은 기존의 불안정한 집계 정보에 의존하지 않고, 안정적이고 포괄적인 정보를 기반으로 하여 확률적 결정을 내림으로써, 경량화를 도와줍니다.

- **Performance Highlights**: FedRTS는 컴퓨터 비전 및 자연어 처리 작업에서 최신 기술(SOTA) 성능을 초과하는 결과를 보여줍니다. 예를 들어, CIFAR-10 데이터셋에서 ResNet-18 모델을 사용할 때, FedRTS는 SOTA 프레임워크에 비해 5.1% 정확도 향상 또는 33.3% 통신 비용 절감을 달성했습니다. 이 연구는 Federated Pruning을 CMAB 관점에서 분석한 첫 번째 사례로, 조정의 안정성을 제공하는 데 기여합니다.



### Principal Components for Neural Network Initialization (https://arxiv.org/abs/2501.19114)
- **What's New**: 본 연구는 PCA(주성분 분석)를 신경망의 첫 번째 레이어에 초기화를 통해 통합하는 PCsInit이라는 새로운 전략을 제안합니다. 이 방법은 신경망 훈련 전 데이터 전처리 없이 해석 가능성을 유지하면서 PCA의 장점을 활용할 수 있습니다. 두 가지 변형 방식인 PCsInit-Act와 PCsInit-Sub도 제안되어, 비선형 패턴 인식 및 대규모 데이터셋에 대한 효율성을 높입니다.

- **Technical Details**: PCsInit는 데이터의 오차를 최소화하기 위해 첫 번째 레이어를 주성분으로 초기화하는 방법으로, 신경망 훈련 시 Hessian 매트릭스의 조건을 개선하여 수렴 속도를 높입니다. PCA 방법은 데이터의 고차원 특징을 저차원으로 변환하고, 신경망의 입력 특성을 분리하여 초기 학습의 안정성을 보장합니다. PCsInit-Act는 활성화 함수를 추가하여 비선형 형태 인식을 개선하고, PCsInit-Sub는 입력의 부분 집합에 대한 주성분을 계산하여 대규모 데이터셋에서 효율성을 향상시킵니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험을 통해 PCsInit와 변형 방식들이 기존의 PCA 전처리에 비해 동등하거나 더 나은 성능을 보여줍니다. 이 방법론은 동시해석 가능성과 신뢰할 수 있는 성능 향상을 제공하며, 신경망의 학습 효율성을 개선합니다. 실험 결과는 PCsInit 방식이 전통적인 초기화 방법들보다 빠른 수렴과 더 높은 정확도를 달성할 수 있음을 입증합니다.



### A Benchmark for Incremental Micro-expression Recognition (https://arxiv.org/abs/2501.19111)
- **What's New**: 이 논문에서는 기존 마이크로 표현(micro-expression) 인식을 위한 점진적 학습(incremental learning) 설정을 제안합니다. 특히, 연속적으로 변화하는 데이터 스트림을 처리하기 위한 벤치마크를 새롭게 수립하여, 새 데이터에 적응하면서도 기존 학습된 지식을 보존할 수 있는 방안을 모색합니다. 기존의 모든 훈련 데이터에 접근하는 것을 가정했던 전통적인 방법과는 대조적입니다.

- **Technical Details**: 저자들은 마이크로 표현 인식에 특화된 점진적 학습 설정을 공식화하고, 이를 위해 신중하게 기획된 학습 순서를 가진 연속 데이터 세트를 구성합니다. 이 논문에서는 두 가지의 시험 프로토콜을 정의하여 서로 다른 평가 목표를 겨냥합니다. 또한, 여섯 가지 기준 방법(baseline methods)과 그에 상응하는 평가 결과도 제공합니다.

- **Performance Highlights**: 제안된 벤치마크는 마이크로 표현 인식 연구를 진전시키기 위한 기초를 마련합니다. 이는 다양한 분야에서의 감정 인식 기술 향상에 기여할 것으로 기대됩니다. 모든 논문에서 사용된 코드는 공개될 예정으로, 연구 공동체의 접근성을 확보하고 후속 연구에 영향을 줄 것으로 보입니다.



### Fairness Analysis of CLIP-Based Foundation Models for X-Ray Image Classification (https://arxiv.org/abs/2501.19086)
Comments:
          This paper has been accepted for presentation at the 2025 IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이 연구는 X-ray 이미지 분류에 적용된 CLIP 유사 모델의 공정성을 체계적으로 평가했습니다. 특히, 성별과 연령을 포함한 다양한 인구 통계학적 배경을 고려하여 다층적 접근 방식을 통해 이러한 모델의 성능을 분석했습니다. 그 결과, 모델 튜닝이 정확도를 향상시켰으나, 여전히 성별과 연령에 따른 공정성 문제는 해결되지 않았습니다.

- **Technical Details**: 연구에서는 NIH Chest X-ray 데이터셋을 사용하여 6개 질병에 대한 공정성을 분석했습니다. 데이터셋은 성별과 연령 데이터에 대한 주석이 포함되어 있으며, 공정한 모델 평가를 위해 균형 잡힌 하위 집합으로 편집되었습니다. CLIP 모델을 포함한 네 가지 변형 모델을 사용하여 제로샷 추론 및 다양한 튜닝 기법으로 공정성을 평가했습니다.

- **Performance Highlights**: 각 모델의 성능은 정확도, 클래스별 F1 점수, 인구 통계 그룹 별 평균 F1 점수를 포함한 지표들을 통해 측정되었습니다. 연구에 따르면 모델 튜닝은 분류 정확도를 높였으나, 여전히 성별 및 연령과 관련된 공정성 간극이 존재하는 것으로 나타났으며, 이는 향후 개선 필요성을 시사합니다.



### Improving vision-language alignment with graph spiking hybrid Networks (https://arxiv.org/abs/2501.19069)
- **What's New**: 이 논문은 비전-언어(vision-language, VL) 간의 의미적 간극을 해소하기 위해 새로운 정밀 시각 의미 표현 모듈을 제안합니다. 이를 위해, panoptic segmentation을 활용하여 세밀한 시각적 기능을 생성하고, Spiking Neural Networks(SNN)와 Graph Attention Networks(GAT)의 장점을 통합한 Graph Spiking Hybrid Network(GSHN)를 개발했습니다. 이 모델은 개별 인스턴스의 은닉 변수와 그에 따른 지역 및 글로벌 맥락적 특징을 포괄적으로 캡처하여, 의미 표현의 풍부함과 다양성을 크게 향상시킵니다.

- **Technical Details**: GSHN은 두 가지 구성 요소를 결합하여 시각적 정보를 인코딩합니다. SNN의 희소한 처리 능력을 활용하여 효과적으로 특징을 집계하며, GAT를 사용하여 심층적인 시각 의미 정보를 포착합니다. 이 과정에서 대조 학습(contrastive learning, CL)을 적용하여 유사한 의미를 가진 노드 간의 회상을 돕고, 계산 효율성을 높이기 위한 스파이크 텍스트 학습(Spiked Text Learning, STL) 방법이 도입되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 GSHN은 다양한 VL 하위 작업에서 유망한 결과를 보여주었습니다. VQA, VE, 이미지-텍스트 검색과 같은 다수의 public 데이터셋에서 효과성을 입증하였으며, ablation 실험을 통해 모델 설계의 타당성과 효과성을 추가로 검증했습니다. 이러한 성과는 VL 의미 표현의 정교한 처리와 관련된 여러 과제가 협력적으로 해결됨을 보여줍니다.



### BEAT: Balanced Frequency Adaptive Tuning for Long-Term Time-Series Forecasting (https://arxiv.org/abs/2501.19065)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문에서는 다양한 주파수의 학습 속도 불균형 문제를 해결하기 위해 BEAT(Balanced frEquency Adaptive Tuning)이라는 새로운 프레임워크를 제안합니다. 고주파와 저주파 구성 요소의 학습 과정을 동적으로 모니터링하고 이를 기반으로 적절히 조정하여 최적의 예측 성능을 달성합니다. 기존의 접근법들은 통합된 손실 함수 아래에서 모든 주파수의 모델 구성 요소를 훈련하였으나, BEAT는 각 주파수마다 적합한 학습률을 배분하여 효과적으로 균형 잡힌 학습을 유도합니다.

- **Technical Details**: BEAT 프레임워크는 시간-주파수 도메인 분석을 기반으로 하며, 각 주파수 성분이 수렴(convergence), 과적합(overfitting), 또는 과소적합(underfitting) 상태에 있는지를 실시간으로 평가합니다. 학습 속도가 빠른 구성 요소의 경우, 학습 속도를 늦추기 위해 역전파된 기울기를 감소시키고, 느린 구성 요소는 기울기를 증가시켜 빠르게 수렴하도록 하여 학습의 일관성을 유지합니다. 이 방법은 주파수 대역에서 경쟁 목표 간의 긴장을 완화하여 전체 학습 프로세스를 동기화합니다.

- **Performance Highlights**: 자체 실험을 통해 BEAT는 일곱 가지 실제 데이터 세트에서 벤치마크 모델들보다 항상 뛰어난 성능을 보였습니다. 이 논문에서는 다양한 도전적인 데이터 세트에서의 실험 결과를 포함하여 BEAT의 장점과 향상된 예측 성능을 명확히 입증하였습니다. 또한, BEAT는 고주파 및 저주파 서브 시리즈 간의 균형을 개선함으로써 장기 예측 성능을 지속적으로 향상시켰습니다.



### Enabling Autonomic Microservice Management through Self-Learning Agents (https://arxiv.org/abs/2501.19056)
- **What's New**: 이 논문에서는 복잡한 소프트웨어 시스템의 자율적 관리의 필요성을 강조하며, ServiceOdyssey라는 자가 학습 에이전트 시스템을 제안합니다. 기존의 대형 언어 모델(LLM)의 지식을 특정 서비스 환경에 적용하는 데 어려움을 겪는 문제를 해결하고자 하며, ServiceOdyssey는 사전 지식 없이도 마이크로서비스를 자율적으로 관리할 수 있는 시스템입니다. Curriculum learning 원리를 활용하여 환경을 탐험하고, 지속적으로 지식을 축적하는 방법론을 제시합니다.

- **Technical Details**: ServiceOdyssey 시스템은 데이터 레이어와 관리 레이어로 구성된 아키텍처를 가지고 있으며, 효율적인 자가 학습 및 작업 실행을 지원합니다. 관리 레이어는 Curriculum Builder (CB), Execution Planner (EP), Knowledge Curator (KC)라는 세 가지 주요 모듈로 구성되어 있습니다. CB는 새로운 마이크로서비스 시스템을 탐색하고 학습하기 위한 작업을 점진적으로 생성하며, EP는 실행 가능한 계획과 행동을 만들어냅니다. KC는 피드백과 상호작용 기록을 기반으로 포괄적인 역량 라이브러리를 구축합니다.

- **Performance Highlights**: ServiceOdyssey의 프로토타입 구현은 Sock Shop 마이크로서비스를 활용하여 자율 마이크로서비스 관리의 가능성을 시연합니다. 지금까지의 연구에서 보여진 결과는 LLM 기반 에이전트가 자율적으로 환경을 학습하고, 동적으로 지식을 업데이트하며, 최적화된 관리 작업을 수행할 수 있다는 것을 입증합니다. 이 접근법은 노동 집약적인 수동 통합 방식에서 벗어나, 에이전트가 자율적으로 지식을 축적하도록 설계되었다는 점에서 혁신적입니다.



### Towards Physiologically Sensible Predictions via the Rule-based Reinforcement Learning Layer (https://arxiv.org/abs/2501.19055)
- **What's New**: 이 논문은 헬스케어 분야의 강화 학습(reinforcement learning, RL) 문헌에 기여하며, Rule-based RL Layer (RRLL)이라는 새로운 패러다임을 제안합니다. RRLL은 모델이 제시한 생리학적으로 불가능한 예측을 수정하는 역할을 하여, 헬스케어 분류 문제들에 적용하여 성능 향상을 보여줍니다. RRLL은 무거운 전문가 지식을 요구하지 않으며, 단지 불가능한 전이(transitions)에 대한 규칙만으로 작동합니다.

- **Technical Details**: RRLL은 생리학적으로 타당한 예측을 지원하기 위해 데이터 기반 모델에 추가된 간단하고 경량화된 레이어입니다. 이 레이어는 분류器가 출력하는 예측을 상태(states)로 받아들이고, 생리학적으로 타당한 라벨을 행동(actions)으로 출력합니다. 경량화된 구조 덕분에, RRLL은 훨씬 적은 규칙 세트로도 생리학적으로 불가능한 실수를 효과적으로 줄일 수 있습니다.

- **Performance Highlights**: 논문에서 제안하는 RRLL은 다양한 헬스케어 분류 문제에 대한 실험을 통해 그 효용성을 입증했습니다. 구체적으로, 수면 단계 예측 및 간질 발작 감지에서의 실험 결과는 RRLL을 통해 기존 모델의 생리학적으로 불가능한 예측이 줄어들면서 정확도가 뚜렷하게 향상되었음을 보여줍니다. 이러한 결과는 임상적 결정을 더욱 신뢰할 수 있도록 합니다.



### Understanding Model Calibration -- A gentle introduction and visual exploration of calibration and the expected calibration error (ECE) (https://arxiv.org/abs/2501.19047)
- **What's New**: 이번 논문에서는 모델의 신뢰성을 높이기 위한 캘리브레이션(calibration)의 중요성을 설명하고, 가장 일반적으로 사용되는 정의 및 평가 지표를 살펴봅니다. 특히, Confidence Calibration과 관련된 여러 새로운 평가 측정 기준의 필요성을 강조합니다. 이를 통해, 기존 ECE와 같은 측정법의 한계에 대한 논의도 진행합니다.

- **Technical Details**: 캘리브레이션(caliabration)은 모델의 추정 확률이 실제 확률과 일치하도록 보장하는 과정입니다. 예를 들어, 날씨 예측 모델이 70%의 강수 확률을 제시할 때, 실제로 70%의 날들이 비 오는 것이어야 하며, 모델은 K개의 클래스에 대해 확률 벡터를 반환합니다. 이를 통해 Confidence Calibration의 정의와 프로세스를 설명하고, 구체적인 수식과 함께 이를 이해할 수 있도록 합니다.

- **Performance Highlights**:  Expected Calibration Error(ECE)는 모델의 예상 확률과 관측된 확률 간의 차이를 평가하는 주요 지표로 활용됩니다. 이 방법은 데이터 포인트를 여러 개의 빈(bin)으로 나누고, 각 빈에서의 평균 정확도와 평균 신뢰도의 절대 차이를 계산하여 ECE를 구합니다. 모델이 특정 빈에서 얼마나 정확한지를 측정함으로써 캘리브레이션의 성능을 평가할 수 있습니다.



### Swarm-Gen: Fast Generation of Diverse Feasible Swarm Behaviors (https://arxiv.org/abs/2501.19042)
Comments:
          Submitted to RAL

- **What's New**: 본 연구는 로봇 군집의 조정 행동을 생성하는 과정에서 다양성과 실현 가능성을 동시에 확보하는 방법론을 제안합니다. 기존의 방법들이 단일 경로를 생성하는 것에만 중점을 둔 반면, 우리는 Conditional Variational Autoencoder (CVAE)와 Vector-Quantized Variational Autoencoder (VQ-VAE)를 활용해 다중 경로를 생성할 수 있도록 하였습니다. 이 연구는 생성한 경로를 안전성 필터(Safety Filter, SF)를 통해 실현 가능한 집합으로 변환하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 우리는 CVAE와 VQ-VAE를 활용하여 전문가 경로 데이터셋에 적합한 두 가지 생성 모델을 학습하였습니다. 이 과정에서 생성된 경로는 다양한 운동학적 및 충돌 제약 조건을 완전히 만족하지 않을 수 있기 때문에, GPU 가속을 이용한 배치 안전성 필터(SF)를 제안하여 다수의 경로를 동시에 실현 가능한 집합으로 변환합니다. 여기서 SF는 모든 로봇의 집합적 실현 가능성 공간 위에서 작동하며, 우리는 이를 빠르게 수렴시킬 수 있는 neural network를 학습시켰습니다.

- **Performance Highlights**: 결과적으로, 본 연구는 다수의 다중 경로 솔루션을 생성할 수 있음을 보여주며, 이는 다양한 군집 조정 행동을 시뮬레이션 할 수 있습니다. 실시간 생성 과정이 commodity GPU에서 효과적으로 이루어짐을 입증했으며, 초기화 정책이 단순한 방법론들에 비해 우리의 SF 솔버의 수렴 속도를 높이는 데 기여함을 확인하였습니다. 우리는 드론을 위한 3D 군집 경로에 중점을 두었지만, 이 방법은 자동 운전에 적용할 수 있는 확장 가능함을 가지고 있습니다.



### Virtual airways heatmaps to optimize point of entry location in lung biopsy planning systems (https://arxiv.org/abs/2501.19003)
- **What's New**: 이번 연구에서는 폐 생검 계획 시스템에서의 POE(point of entry) 최적화를 위한 가상 모델을 제안합니다. 이 모델은 계획 시뮬레이션에서의 방향과 실제 수술 동안의 방향 간의 차이로 인한 오차 범위를 고려하여 생검 샘플의 품질을 평가할 수 있습니다. 추가적으로, 병변의 특성이 미치는 영향을 조사하는 것이 주요 목표입니다.

- **Technical Details**: 생검 품질은 환자 맞춤형 기도의 골격에 투영된 히트맵(heatmap)으로 제공됩니다. 이 골격은 기도의 구조를 3D로 표현하며, 히트맵의 강도는 각 POE에서 추출될 수 있는 조직의 잠재적 양을 나타냅니다. 생검 도구의 도입에 대한 불확실성 영역을 나타내는 원뿔과 병변 간의 교차 점을 통해 결정됩니다.

- **Performance Highlights**: 시뮬레이션된 다양한 개입 장면은 CT 스캔에서 추출된 단일 해부학과 정형 및 비정형 형태의 두 병변을 대상으로 진행되었습니다. 분석 결과, 비정형 형태에서의 병변 방향과 두 형태 모두에서의 거리의 강한 영향을 시사합니다. 제안된 히트맵은 최적의 POE를 시각적으로 평가하고, 서로 다른 기관지의 여러 최적 POE의 존재 여부를 파악하는 데 기여합니다.



### Adversarial Attacks on AI-Generated Text Detection Models: A Token Probability-Based Approach Using Embeddings (https://arxiv.org/abs/2501.18998)
- **What's New**: 이번 연구에서는 AI로 생성된 텍스트를 탐지하는 모델들, 특히 Fast-DetectGPT에 대한 혁신적인 적대적 공격 기법을 제안합니다. 이 방법은 데이터 변형을 위해 임베딩 모델을 활용하여 AI-generated 텍스트의 원본 탐지 확률을 줄입니다. 특히, Tsetlin Machine (TM)을 기반으로 한 해석 가능성 높은 접근 방식을 사용합니다.

- **Technical Details**: 제안한 방식은 임베딩 기법을 활용하여 유사한 단어 확률 벡터를 결합하는 것으로, 탐지 정확도를 XSum 데이터셋에서 0.4431에서 0.2744로, SQuAD 데이터셋에서는 0.5068에서 0.3532로 감소시킵니다. TM-AE 구조를 통해 생성된 텍스트에 대한 저확률을 할당하여 탐지 시스템의 전반적인 텍스트 스코어를 낮추는 방식입니다.

- **Performance Highlights**: AI-origin 탐지 시스템에 대한 적대적 공격을 통해, Fast-DetectGPT의 탐지 정확성을 눈에 띄게 낮추는데 성공했습니다. 이 방법은 기존 탐지 시스템의 취약성을 겨냥하며, TM 모델의 해석 가능성을 통해 공격 메커니즘에 대한 통찰력을 제공합니다.



### VKFPos: A Learning-Based Monocular Positioning with Variational Bayesian Extended Kalman Filter Integration (https://arxiv.org/abs/2501.18994)
- **What's New**: 이 논문은 VKFPos라는 새로운 접근 방식을 제안하여, Absolute Pose Regression (APR)과 Relative Pose Regression (RPR)을 Variational Bayesian inference 프레임워크 내에서 Extended Kalman Filter (EKF)와 통합합니다. 이 방법은 모노큘러 포지셔닝 문제의 본질적인 후행 확률을 APR과 RPR 구성 요소로 분해할 수 있음을 보여줍니다. 이러한 분해는 딥러닝 모델에 통합되어 불확실성을 더 잘 관리할 수 있도록 하며, 이를 통해 성능이 향상됩니다.

- **Technical Details**: VKFPos는 APR과 RPR을 결합하여 위치 예측 정확도를 개선하는 경량의 모노큘러 포지셔닝 접근 방식으로, EKF의 재귀 구조를 활용하여 과거 상태 정보를 현재 상태 예측 및 궤적 예측에 활용합니다. 특이한 점은 Variational Bayesian inference에 근거한 절대 및 상대 포즈 추정기들을 위한 혁신적인 훈련 패러다임을 제공한다는 것입니다. 이 방식은 APR과 RPR 지점을 각각 예측하여 EKF에서 궤적을 최적화하는 구조를 가지고 있습니다.

- **Performance Highlights**: 실험 결과, VKFPos의 단일 샷 APR 지점은 최신 기술에 필적하는 정확도를 달성했으며, 연속 이미지로 RPR과 EKF 통합이 가능한 경우에는 VKFPos가 시간적 APR 및 모델 기반 통합 방법을 초월하는 성능을 보여주었습니다. 이로 인해 VKFPos는 모노큘러 포지셔닝 분야에서 강력한 대안으로 자리 잡을 것으로 기대됩니다.



### Symmetric Pruning of Large Language Models (https://arxiv.org/abs/2501.18980)
- **What's New**: 본 논문은 LLM(대형 언어 모델)의 사후 훈련 가지치기(PTP) 방법론에 대한 새로운 이론적 통찰을 제시합니다. 이를 통해 Wanda 및 RIA 같은 기존 방법들의 근본적인 기제를 이해할 수 있도록 돕습니다. 또한, 새로운 훈련 없는 미세 조정 방법인 $R^2$-DSnoT를 도입하여 상대적인 가중치 중요성을 반영하고, 퍼포먼스를 한층 향상시킵니다.

- **Technical Details**: 이 논문에서는 Symmetric Weight And Activation (SymWanda)라는 새로운 접근법을 통해 입력 활성화와 출력의 영향을 모두 고려하는 이론적 틀을 제안합니다. 전통적인 가지치기 방법들의 성능을 증가시키기 위한 기구를 제시하고, 모델 효율성을 높이는 추가적인 방법들도 제안합니다. 특히, 기존 방법들의 제한을 극복하기 위해 상대적인 가중치 중요성을 고려한 성장 기준을 통합하고, 결정 경계를 정규화하는 항목을 추가하여 성능 개선을 도모합니다.

- **Performance Highlights**: 실험 결과, 우리의 제안 방식들이 기존 방법들에 비해 우수한 효율성을 보임을 확인했습니다. 특히, 상대적인 가중치 정보를 효과적으로 활용하는 간단하면서도 강력한 방법론을 통해 기존 스탠다드에 대한 성과를 상회하는 결과를 달성했습니다. 새로운 방법론 $R^2$-DSnoT는 강력한 기초선을 초월하여 새로운 최첨단 성능을 확립하는 데 성공했습니다.



### GPO-VAE: Modeling Explainable Gene Perturbation Responses utilizing GRN-Aligned Parameter Optimization (https://arxiv.org/abs/2501.18973)
- **What's New**: 본 연구에서는 GPO-VAE라는 새로운 Explainable VAE를 제안합니다. 이 모델은 유전자 조절 네트워크(GRN)를 활용하여 유전자 교란에 대한 세포 반응을 예측하며, GRN에 맞춰 파라미터 최적화를 통해 학습된 특징을 명확하게 설명합니다. GPO-VAE는 기존 모델들보다 유의미한 GRN을 생성할 수 있는 능력을 보여주며, 실험적으로 검증된 조절 경로들과 잘 일치합니다.

- **Technical Details**: GPO-VAE 모델은 잠재 공간(latent space)에서 유전자 조절 네트워크를 명시적으로 모델링하여, 유전자 교란 효과와 관련된 학습 가능한 파라미터를 GRN 정렬을 통해 최적화합니다. 이 모델은 Encoder-Decoder 아키텍처를 기반으로 하며, 잠재 교란 인코더(latent perturbation encoder), 잠재 요인 인코더(latent artifact encoder), 그리고 잠재 기초상태 인코더(latent basal state encoder) 세 가지 세부 모듈로 구성됩니다.

- **Performance Highlights**: 실험 결과, GPO-VAE는 여러 벤치마크 데이터셋에서 전사 반응 예측에 대해 최첨단 성능을 달성했습니다. 또한 GRN 추론 작업에 대한 기초 결과 역시 다른 방법들과 비교하여 의미 있는 GRN을 생성할 수 있는 능력을 나타내어, 모델의 설명 가능성을 크게 향상시켰습니다.



### Enhancing Neural Function Approximation: The XNet Outperforming KAN (https://arxiv.org/abs/2501.18959)
Comments:
          arXiv admin note: text overlap with arXiv:2410.02033

- **What's New**: XNet은 고차 함수 근사를 위해 Cauchy 적분 기반 활성화 함수를 활용하는 단일 레이어 신경망 아키텍처입니다. 이 연구는 기존의 MLP와 Kolmogorov-Arnold Networks (KANs)와 비교하여 XNet의 이론적 및 실험적 장점을 자세히 설명합니다. 특히 XNet은 최대 50000배의 근사 오차 감소와 훈련 시간을 최대 10배 단축시킨 것으로 나타났습니다.

- **Technical Details**: XNet은 Cauchy 적분 정리를 기반으로 하는 새로운 신경망 모델로, Cauchy 활성화 함수를 독특하게 설계하였습니다. 이 함수는 훈련 중 최적화되는 매개변수 λ1, λ2 및 d로 수학적으로 표현됩니다. 실험 결과, XNet은 MLP와 KANs에 비해 함수 근사 및 저차 PDE 해결에서 우수한 성능을 보이며, 특히 고차원 환경에서 효과적인 근사가 가능합니다.

- **Performance Highlights**: XNet은 Heaviside step function과 같은 복잡한 고차원 시나리오 및 노이즈 데이터 처리에서 KAN보다 뛰어난 성능을 보였습니다. 또한 PINN 프레임워크 내에서 Poisson 및 Heat 방정식을 기준으로 한 평가에서도 MLP 및 KAN에 비해 현저한 성능 향상을 입증했습니다. 강화 학습 분야에서는 HalfCheetah-v4와 Swimmer-v4 환경에서 기존 아키텍처들보다 우수한 점수를 기록하며, XNet의 다재다능성과 강력한 성능을 강조합니다.



### Deep Learning based Quasi-consciousness Training for Robot Intelligent Mod (https://arxiv.org/abs/2501.18955)
- **What's New**: 이 논문은 복잡한 작업을 수행하기 위해 학습하고 추론할 수 있는 딥러닝 기반 로봇 지능 모델을 탐구합니다. 로봇 지능 모델의 학습 과정을 자극하기 위해 환경 요인 행렬(environmental factor matrix) 네트워크를 구축하고, 손실 함수(loss function)를 최소화하기 위해 모델 매개변수(parameter)를 조정하여 최적화합니다.

- **Technical Details**: 모델은 과거의 모든 개념을 융합하여 이전에 경험하지 못한 것을 나타낼 수 있는 방법을 제시합니다. 또한 로봇은 anthropomorphic behavior patterns를 훈련하기 위해 최소 1~3년의 특별 교육을 받아야 하며, 이를 통해 복잡한 환경 정보를 이해하고 합리적인 결정을 내릴 수 있는 인지 능력을 발전시킵니다.

- **Performance Highlights**: 이 연구는 로봇 지능 모델의 딥러닝 기반 준의식(quasi-consciousness) 훈련의 잠재적 응용을 탐구하며, 로봇이 보다 발전된 인지 기능을 갖출 수 있도록 하는 가능성을 제시합니다.



### Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them (https://arxiv.org/abs/2501.18950)
- **What's New**: 이 연구에서는 diffusion 모델에서 유해 내용을 생성할 위험을 줄이기 위한 개념 지우기(Concept Erasure) 기법에 대해 다루고 있습니다. 기존의 고정 목표 스트래티지(fixed-target strategy)는 특정 개념을 중립 개념이나 빈 텍스트 프롬프트로 매핑하는 방식으로, 이는 비효율적이라는 것을 보여줍니다. 연구팀은 개념 공간을 그래프로 모델링하여 하나의 개념을 지우는 것이 다른 개념에 미치는 영향을 분석하고, Adaptive Guided Erasure (AGE)라는 새로운 기법을 제안하여 최적의 목표 개념을 동적으로 선택함으로써 의도치 않은 부작용을 최소화합니다.

- **Technical Details**: AGE 방법은 지우고자 하는 각 개념에 맞춰 최적의 목표 개념을 선택하는 방식으로 설계되었습니다. 기존의 유사한 기법들과의 비교를 통해 AGE가 관련 없는 개념을 잘 보존하면서도 강력한 지우기 성능을 유지하는 것을 입증했습니다. 이 방법은 반복적인 지식 요구가 없는 전이 가능한 방식으로, 모델 매개변수를 수정하여 전이 학습을 통해 유해 개념을 효과적으로 제거할 수 있는 기법입니다.

- **Performance Highlights**: AGE는 최신의 지우기 기법들과 비교했을 때, 무관한 개념을 보존하는 능력이 현저히 우수한 결과를 보여주었습니다. 실험 결과, AGE 방법은 특정 개념 제거와 관련된 원치 않는 부작용을 최소화하면서 유해 개념을 효과적으로 제거하는 데 성공했습니다. 추가적으로, 연구팀은 코드를 공개하여 이 기법의 구현과 활용이 가능하도록 하였습니다.



### KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search (https://arxiv.org/abs/2501.18922)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 KBQA(Knowledge Base Question Answering)의 새로운 접근법인 KBQA-o1을 제안합니다. 이 방법은 Monte Carlo Tree Search (MCTS)를 활용하여 KB 환경을 탐색하고, 단계적인 논리 형태 생성을 위한 ReAct 기반 에이전트를 도입합니다. KBQA-o1은 기존 방법들의 치명적인 단점을 보완하며, 더욱 효율적인 답변 생성을 지원합니다. 이전의 저자원 KBQA 방법에 비해 현저한 성능 향상을 보여주는 결과를 도출했습니다.

- **Technical Details**: KBQA-o1은 KB 환경에서의 탐색을 증진시키기 위해 ReAct 기반의 에이전트 프로세스를 설계했습니다. MCTS는 정책 및 보상 모델을 기반으로 한 휴리스틱 검색 방법으로, 탐색 공간과 성능의 균형을 맞추는 데 중점을 두었습니다. 이것은 낮은 샘플 데이터로도 훈련할 수 있도록 하여 대량의 자동 주석 데이터를 생성하는 데 도움을 줍니다. 이러한 프로세스를 통해 KBQA-o1은 기존의 저자원 KBQA 방법들에서 성능을 능가하게 됩니다.

- **Performance Highlights**: 실험 결과 KBQA-o1은 GrailQA, WebQSP, GraphQ 데이터셋에서 기존의 저자원 KBQA 방법과 비교하여 현저히 뛰어난 성능을 기록했습니다. Llama-3.1-8B 모델의 GrailQA F1 성능은 78.5%에 달하며, 이는 이전의 최고의 방법인 GPT-3.5-turbo의 48.5%를 훨씬 초월합니다. 또한, KBQA-o1은 다양한 오픈소스 LLM과의 호환성을 지원하여, 다양한 KBQA 응용 프로그램에 적용 가능하다는 추가적인 장점을 가지고 있습니다.



### Deepfake Detection of Singing Voices With Whisper Encodings (https://arxiv.org/abs/2501.18919)
Comments:
          Accepted in ICASSP,2025

- **What's New**: 본 연구에서는 음악 산업에서 아티스트들에게 우려를 주는 singing voice deepfake를 탐지하기 위한 새로운 시스템인 SVDD(Singing Voice Deepfake Detection)를 제안합니다. SVDD 시스템은 noise-variant 인코딩을 사용하는 Whisper 모델의 특징을 기반으로 하여, 이 인코딩이 깊은 가짜 노래 목소리를 구분하는 데 기여할 수 있음을 시사합니다. 다양한 Whisper 모델 크기와 두 가지 분류기인 CNN과 ResNet34를 사용하여 성능을 평가했습니다.

- **Technical Details**: Whisper는 680K 시간의 다양한 환경에서 수집된 데이터로 훈련된 ASR(Automatic Speech Recognition) 모델입니다. 본 연구에서 제안하는 SVDD 시스템은 Whisper 인코딩과 ResNet34를 활용하여 구성되었습니다. Whisper 모델은 명시적으로 noise-invariant하지 않지만, 비음성 정보를 풍부하게 인코딩하여 음악적 배경 소음의 종류를 이해합니다.

- **Performance Highlights**: Singfake 데이터셋을 사용하여 실험을 수행하였으며, 실제 노래(밴포이드) 28.93시간과 딥페이크 생성 노래 29.40시간을 포함합니다. 실험 결과, SVDD 시스템은 Bonafide 음성과 딥페이크 음성을 구별하는 데 효과적이었으며, 피치, 지속 시간 및 음악적 맥락이 두 가지 경우에서 어떻게 영향을 미치는지를 분석했습니다.



### Lightspeed Geometric Dataset Distance via Sliced Optimal Transpor (https://arxiv.org/abs/2501.18901)
Comments:
          23 pages, 9 figures

- **What's New**: 이번 연구에서는 sliced optimal transport dataset distance (s-OTDD)를 소개합니다. 이는 모델과 임베딩에 무관하게 데이터셋을 비교할 수 있는 새로운 접근법으로, 훈련이 필요 없으며 클래스 수의 변화에 강인성을 가지고 있습니다. 주요 혁신은 Moment Transform Projection (MTP)으로, 이를 통해 데이터 포인트를 원차원 분포로 변환할 수 있습니다.

- **Technical Details**: s-OTDD는 예측된 원차원 분포들 간의 Wasserstein 거리로 정의되며, 랜덤 프로젝션 파라미터에 따라 계산됩니다. 이 방법은 데이터 포인트와 특징 차원 수에 대해 선형적인 계산 복잡도를 가지며, 클래스 수에 독립적입니다. 또한 이 기법은 데이터셋 간 유의미한 기하학적 연관성을 통해 기존 데이터셋 불일치 측정치보다 효율적입니다.

- **Performance Highlights**: s-OTDD는 MNIST와 CIFAR10의 하위 집합을 비교할 때 OTDD와 강한 상관관계를 보이며, 기존의 경쟁자들보다 더 빠른 성능을 자랑합니다. 아울러, 다양한 데이터셋과 형태에서 전이 학습의 성능 차이와 잘 연관되며, 데이터 증강을 통한 분류 정확성에서도 효율적인 상관관계를 보입니다.



### Building Bridges, Not Walls -- Advancing Interpretability by Unifying Feature, Data, and Model Component Attribution (https://arxiv.org/abs/2501.18887)
- **What's New**: 이 논문은 AI 시스템의 복잡성이 증가함에 따라 그 행동을 이해하는 것이 중요한 과제가 되고 있음을 강조합니다. 다양한 방법들이 모델 행동을 입력 특성, 훈련 데이터 및 내부 모델 구성 요소와 같은 세 가지 주요 측면에 귀속시키기 위해 개발되었으나, 이러한 방법들이 독립적으로 연구되고 적용되어 단절된 접근 방식이 발생했습니다. 이 연구에서는 이 세 가지 종류의 귀속 방법들이 기본적으로 유사함을 공유하고 있으며, 이를 통합함으로써 해석 가능성 연구에 기여할 수 있음을 주장합니다.

- **Technical Details**: 이 논문에서는 먼저 귀속 문제의 통합된 프레임워크를 제시하여 세 가지 귀속 유형을 포괄하는 형태로 문제를 정형화합니다. 이후, 각 유형의 발전 및 성공적인 방법을 분석하며, 이들 방법들이 어떻게 공유된 기법(perturbations, gradients, linear approximations)을 통해 연결되어 있는지를 보여줍니다. 또한, 이러한 분석을 바탕으로 공유된 개념과 공통적인 도전을 식별하고, 새로운 연구 개발을 위한 교차 도메인 지식 이전을 촉진하는 방법을 논의합니다.

- **Performance Highlights**: 이 통합적인 관점은 기존의 귀속 방법을 이해하고 새로운 방향성을 제시하며, 신입 연구자들에게 더욱 접근 가능한 분야로 만들어 줍니다. 논문은 귀속 및 해석 가능성을 넘어서 모델 편집, 조정 및 규제와 같은 보다 넓은 AI 연구 분야에서도 새로운 기회를 강조하며, 이를 통해 연구자들이 효율적으로 협력 및 발전할 수 있는 방법을 제안합니다.



### UP-VLA: A Unified Understanding and Prediction Model for Embodied Agen (https://arxiv.org/abs/2501.18867)
- **What's New**: 이번 논문에서는 Vision-Language-Action (VLA) 모델의 훈련 패러다임을 재검토하고, UP-VLA라는 통합 VLA 모델을 제안합니다. 기존의 Vision-Language 모델(VLM)에 대한 한계를 극복하기 위해 미래 예측 목표와 다중 모드 이해를 결합한 훈련 방법을 적용하여, 상위 수준의 의미 이해와 하위 수준의 공간 이해를 모두 강화합니다. 실험 결과, UP-VLA는 기존의 최신 방법에 비해 Calvin ABC-D 벤치마크에서 33% 향상된 성능을 보였습니다.

- **Technical Details**: UP-VLA는 다중 모드 이해 데이터셋과 미래 예측 생성 데이터를 통해 고급 특성과 저급 특성을 동시에 일치시키는 새로운 훈련 패러다임을 사용합니다. 모델은 세 가지 유형의 데이터를 기반으로 오토 회귀 모델을 공동 훈련하여, 다양한 시뮬레이션 및 실제 환경에서 알고리즘의 능력을 평가하였습니다. 특정 실험에서는 VLM 기반 VLA 모델이 전형적인 다중 작업 학습 환경에서 강한 성능을 보였고, 시각 예측 기반의 사전 훈련 방식이 적응성과 정밀 제어가 중요한 작업에서 더 나은 성과를 나타냈습니다.

- **Performance Highlights**: UP-VLA 모델은 Calvin ABC-D 벤치마크에서의 성능이 33% 증가했으며, 실제 조작 과제에서도 개선된 성공률을 보여주었습니다. 특히, 정밀한 공간 정보가 요구되는 실제 작업에서 뚜렷한 성과 향상이 나타났습니다. 이는 UP-VLA 방식이 의미적 및 저급 특성을 모두 보존하는 데 효과적임을 강조합니다.



### REG: Rectified Gradient Guidance for Conditional Diffusion Models (https://arxiv.org/abs/2501.18865)
Comments:
          19 pages, 10 figures

- **What's New**: 이 연구는 diffusion 모델에서 조건부 생성(condition generation)을 향상시키기 위한 guidance 기법의 이론적 기초와 실제 구현 간의 불일치를 해소하고자 합니다. 기존의 스케일된 한계분포(target) 대신 유효한 스케일된 결합 분포(objective)를 설정함으로써 이론적으로 잘못된 부분을 교정합니다. 또한, 기존의 guidance 기법이 실제로 최적 솔루션에 대한 근사치에 불과하다는 점을 확인하고, 이를 바탕으로 rectified gradient guidance (REG)를 제안합니다.

- **Technical Details**: Diffusion 모델은 데이터의 고유 분포를 모델링하여 새로운 데이터 샘플을 생성하는 방법으로, 최근에는 Gaussian 노이즈를 점진적으로 추가하는 forward 과정과 신경망으로 학습된 reverse 과정을 통해 고품질 샘플 복원이 이루어집니다. guidance 기법은 샘플링 시 사용자 정의 가이던스 신호를 기반으로 노이즈 예측 네트워크의 출력을 업데이트하여 모드 커버리지(mode coverage)와 샘플 충실도(sample fidelity)를 조정합니다. 본 연구에서는 guidance 기법의 이론적 맥락을 명확히 하고, 각 기법이 최적 솔루션에 다가가는 근사화로 작용함을 이론적으로 분석합니다.

- **Performance Highlights**: 연구 결과, rectified gradient guidance (REG)는 기존의 guidance 기법들보다 최적 솔루션에 더 나은 근사를 제공하며, 1D 및 2D 실험을 통해 VALIDATE되었습니다. 구체적으로, ImageNet과 text-to-image 생성 작업에 대해 REG를 포함할 경우 FID 및 Inception/CLIP 점수가 일관되게 개선되는 결과를 보였습니다. 이러한 결과는 REG가 다양한 조건부 생성 작업에서 효과적임을 입증하며, 제안된 이론적 프레임워크를 뒷받침합니다.



### BRiTE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning (https://arxiv.org/abs/2501.18858)
- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Models, LLM)이 복잡한 추론 작업에서 신뢰할 수 있는 추론 과정을 생성하는 데 어려움을 겪고 있다는 점을 강조합니다. 이를 해결하기 위해, 우리는 잠재적 사고 과정과 평가 신호를 통합한 새로운 그래픽 모델을 사용하여 LLM의 추론을 형식화하는 확률적 프레임워크를 제안합니다. 특히 Bootstrapping Reinforced Thinking Process (BRiTE) 알고리즘을 도입하여, 두 단계의 과정으로 고품질의 합리적인 이유(rationale)를 생성하고 LLM의 성능을 향상시키는 방법을 설명합니다.

- **Technical Details**: 본 연구에서는 Bootstrapping Reinforced Thinking Process (BRiTE) 알고리즘을 통해 고품질의 합리적인 이유와 답변을 생성하는 확률적 그래픽 모델을 제안합니다. BRiTE 알고리즘은 강화 학습(reinforcement learning)을 사용하여 첫 번째 단계에서 가능한 사고 과정을 근사하여 합리성을 생성하고, 두 번째 단계에서 모델의 매개변수에 대해 합리적인 생성의 결합 확률을 극대화합니다. 이론적으로는 BRiTE가 1/T의 수렴 속도를 가지며, 이는 반복(iterations) 수 T에 해당합니다.

- **Performance Highlights**: BRiTE 알고리즘은 여러 LLM 모델(Gemma, Llama, Mistral)에 대해 수학 및 코드 생성 벤치마크를 포함하여 일관된 성능 향상을 보여줍니다. 특히, GEM8K 벤치마크에서 Gemma-1.1-7B-it 모델을 적용했을 때 10점의 성능 향상을 달성하였으며, 인간이 주석을 단 사고 과정을 사용한 감독 학습(supervised fine-tuning) 방법과 비슷한 성능을 보였습니다. 이로 인해 BRiTE 알고리즘은 수동 프롬프트 설계에 대한 의존도를 줄이고, 자동화된 사고 프로세스 생성의 가능성을 제시합니다.



### Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming (https://arxiv.org/abs/2501.18837)
- **What's New**: 이 논문에서는 전통적인 방법으로 방어하기 어려운 범용 jailbreaks에 대한 대응으로 Constitutional Classifiers를 도입하고 있습니다. 이들은 자연어 규칙을 사용하여 생성된 합성 데이터로 훈련되어 특정 내용의 허가 및 제한을 명시하는 접근법을 사용합니다. 연구 결과, 이러한 새로운 분류기가 대규모 모델 방어에 효과적임을 입증하였으며, 생산 과정의 거부율은 오직 0.38% 증가에 그쳤습니다.

- **Technical Details**: Constitutional Classifiers는 화학무기와 같은 유해한 내용을 식별하기 위한 헌법 기초로 훈련됩니다. 이 과정에서 ‘무해한’ 헌법을 추가하여 분류기의 성능을 향상시킬 수 있다는 점도 발견하였습니다. 연구에서 제시된 간단한 모델에서는 성공적인 단계의 성공 확률을 독립적으로 가정하고, 방어의 효과가 프로세스의 복잡성에 따라 기하급수적으로 감소하는 경향을 보였습니다.

- **Performance Highlights**: 실험을 통해, 제안된 시스템이 잘 작동하여 95% 성공률을 보이는 전통적인 모델과 비교하여, Constitutional Classifiers 도입으로 인해 평균 10의 5 제곱배의 성능 저하를 약속합니다. 다양한 훈련 방식과 하이퍼파라미터 조정이 성능에 긍정적인 영향을 미쳤으며, 고급스러운 화학무기 이론에 대한 요청을 식별하는 정확도가 높아졌습니다.



### Pitfalls of defacing whole-head MRI: re-identification risk with diffusion models and compromised research potentia (https://arxiv.org/abs/2501.18834)
- **What's New**: 이번 연구에서는 얼굴이 변경된 머리 MRI 데이터에서 얼굴을 복원하는 refacing 파이프라인을 개발하였습니다. 이 과정에서 cascaded diffusion probabilistic models (DPMs)를 사용했습니다. 연구는 180명의 피험자로부터 학습된 DPM 모델을 활용하여 484명의 보이지 않는 피험자의 이미지를 평가했습니다.

- **Technical Details**: DPMs는 얼굴 정보가 삭제된 MRI 이미지를 원본 얼굴 이미지로 복원하기 위해 훈련되었습니다. 이 과정에서 각 데이터 세트가 어떻게 기능하는지 비교하기 위해 defacing된 이미지와 원본 이미지의 방사선 밀도(Radiodensity)를 예측하는 방식으로, facial voxels의 정보가 실제로 연구에 유용한지를 검토하였습니다.

- **Performance Highlights**: 결과적으로 DPMs는 원본 얼굴과 비슷한 고해상도의 얼굴 이미지를 생성할 수 있음을 보여주었으며, 원본 얼굴과의 표면 거리도 통계적으로 유의미하게 작았습니다. 그러나 defacing된 이미지를 사용했을 경우 원본 이미지에 비해 Skeletal muscle radiodensity에 대한 예측 성능이 크게 낮아지는 결과를 얻었습니다. 이는 defacing이 개인정보를 보호하는 데 실패할 뿐 아니라, 가치 있는 정보도 제거할 가능성이 있음을 시사합니다.



### An Optimal Cascade Feature-Level Spatiotemporal Fusion Strategy for Anomaly Detection in CAN Bus (https://arxiv.org/abs/2501.18821)
- **What's New**: 본 연구는 Autonomous Vehicle(자율주행차) 분야에서 Controller Area Network(CAN) 버스의 보안 취약점을 해결하기 위한 새로운 기계 학습 모델을 개발했습니다. 기존 연구의 한계를 극복하기 위해, 본 연구는 모든 주요 이상 패턴을 포괄하는 모델을 제안하며, 이 모델은 시간적 및 공간적 정보의 융합 전략을 최적화하여 보안 위협을 탐지합니다.

- **Technical Details**: 이 연구에서는 Cascade Feature-Level Fusion 전략을 기반으로 하는 기계 학습 모델을 제안합니다. 이는 두 매개변수 유전 알고리즘을 사용하여 최적화된 것으로, 다양한 이상 패턴을 모두 포괄하는 모델을 구축하는 데 중점을 두고 있습니다. 모델의 성능은 paired t-test를 통해 검증되었으며, 이론적으로 뛰어난 안정성과 신뢰성을 확보했습니다.

- **Performance Highlights**: 제안된 모델은 두 개의 널리 사용되는 데이터 세트에서 비교 분석을 통해 다른 모델들보다 우수한 정확도와 F1-score를 기록했습니다. 본 연구는 이 분야에서 발표된 모델들 중 가장 뛰어난 성능을 자랑하며, 향후 자율주행차의 안전성을 강화하는 데 중요한 기여를 할 것으로 기대됩니다.



### Large Language Models as Common-Sense Heuristics (https://arxiv.org/abs/2501.18816)
Comments:
          7 page body, 2 page references, 5 page appendix (14 page total); 1 figure; Submitted to IJCAI2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 계획 작업에서 성공적인 솔루션을 생성하는 데 어려움을 겪고 있는 현실을 지적합니다. 기존의 시스템은 LLM의 파라미터화된 지식을 활용하여 로컬 서치 알고리즘을 위한 행동 선택을 담당하도록 하고, 이를 통해 중간 언어를 필요로 하지 않는 새로운 계획 방법을 제안합니다. 이는 기존의 접근 방법보다 22% 높은 성공률을 기록하며 일관되게 실행 가능한 계획을 생성하는데 주목받고 있습니다.

- **Technical Details**: 자동 계획(AI Planning)은 특정 환경 내에서 실행 가능한 행동의 시퀀스를 생성하는 인공지능의 한 분야입니다. 계획 작업은 상태 공간, 초기 상태, 목표 상태 집합, 행동 집합 및 적합성 함수와 전이 함수를 포함하는 6-튜플로 정의됩니다. 본 연구에서는 LLM의 세계 지식을 활용하여 행위 선택을 위한 휴리스틱(huristic)으로 사용되며, 이를 통해 계획 생성의 효율성을 높이고 있습니다.

- **Performance Highlights**: 제안된 방법은 가정 환경에서의 단순 작업 수행에 있어 기존의 ProgPrompt 시스템보다 22% 높은 성공률을 보이고 있습니다. 이러한 결과는 중간 언어의 필요성을 제거하고, LLM이 제어하는 대리인이 사용하는 동일한 표현 언어 내에서 직접 작동할 수 있음을 보여줍니다. 또한, 이 방법은 잘못된 방식으로 해결된 작업에서도 실행 가능한 계획을 일관되게 생성할 수 있습니다.



### An Adversarial Approach to Register Extreme Resolution Tissue Cleared 3D Brain Images (https://arxiv.org/abs/2501.18815)
- **What's New**: 본 논문에서는 세포 활동과 동역학을 명확히 볼 수 있는 고해상도 이미지를 등록할 수 있는 Generative Patch 기반 3D 이미지 등록 모델인 InvGAN을 개발했습니다. 기존의 이미지 등록 방법이 해상도가 높은 이미지를 등록하는 데 한계를 보이는 반면, InvGAN은 tissue clearing 과정을 통해 얻은 데이터를 효과적으로 처리할 수 있습니다. 이 모델은 두 개의 변형 필드를 동시에 생성하면서 이미지의 등록 품질을 비교하는 두 개의 판별기 네트워크를 사용합니다.

- **Technical Details**: 우리의 InvGAN 모델은 tissue clearing과 Light-sheet fluorescence microscopy (LSFM)로 얻은 이미지를 등록하는 데 특화되어 있습니다. 특히, 이 모델은 긴 계산 시간과 대규모의 컴퓨팅 자원을 요구하지 않고, 적은 데이터로 학습할 수 있도록 설계되었습니다. 또한 InvGAN은 Patch 기반 접근 방식을 채택하여 전체 장기 이미지를 단일 세포 해상도로 처리할 수 있는 능력을 갖추고 있으며, 아드버셜 손실(adversarial loss)을 사용하여 이미지 품질을 향상시킵니다.

- **Performance Highlights**: 테스트 결과, InvGAN은 25% 해상도에서 기존의 방법들과 유사한 정확도를 유지하면서도 약 7분이라는 짧은 시간 안에 등록을 완료합니다. 반면 100% 해상도에서는 대부분의 기존 등록 방법이 실패했던 반면, InvGAN은 단 10분 만에 등록을 진행했습니다. 이는 텍스처와 세포의 구조를 고려한 새로운 접근 방식의 유효성을 입증하며, 고해상도 이미지 등록의 새로운 패러다임을 제시합니다.



### Every Image Listens, Every Image Dances: Music-Driven Image Animation (https://arxiv.org/abs/2501.18801)
- **What's New**: 이번 연구에서는 MuseDance라는 새롭고 혁신적인 엔드-투-엔드 모델을 소개합니다. MuseDance는 음악과 텍스트 입력을 동시에 활용하여 참조 이미지를 애니메이션화합니다. 이중 입력 방식을 통해 사용자는 자신이 원하는 텍스트 설명에 따라 개인화된 비디오를 생성할 수 있으며, 음악에 맞춘 캐릭터의 움직임을 동기화할 수 있습니다. 기존 접근법과는 달리 MuseDance는 복잡한 모션 가이드를 요구하지 않습니다.

- **Technical Details**: 본 연구는 두 단계로 구성된 훈련 프레임워크를 사용하여 이미지 애니메이션을 구현합니다. 첫 번째 단계에서는 목표 비디오의 개별 프레임을 학습하여 시각적 특징을 획득하고, 두 번째 단계에서는 음악과 텍스트를 트리거로 사용하여 이들 입력에 부합하는 애니메이션 프레임을 생성합니다. 이 과정은 스테이블 디퓨전(Stable Diffusion) 모델을 기반으로 하며, 음악 특징과 텍스트 안내를 연결하여 일관되고 역동적인 애니메이션을 생성합니다.

- **Performance Highlights**: MuseDance는 2,904개의 댄스 비디오와 이에 상응하는 배경 음악 및 텍스트 설명으로 구성된 새로운 다중모달 데이터셋을 제시합니다. 고유한 음악의 흐름과 비디오 시퀀스 간의 의미론적 정렬을 가능하게 하여 매력적이고 역동적인 애니메이션을 생성할 수 있습니다. 우리 모델은 유연하고 정밀한 제어능력과 다양한 객체에 대한 일반화 능력을 선보이며, 음악 기반 비디오 생성 작업의 기준을 설정합니다.



### Compositional Generalization Requires More Than Disentangled Representations (https://arxiv.org/abs/2501.18797)
Comments:
          8 pages, 4 figures, plus appendix

- **What's New**: 이 논문은 컴포지셔널(Compositional) 일반화의 중요성을 강조하며, 분리된 표현(Disentangled Representation)이 OOD(out-of-distribution) 샘플을 생성하는 데 실패하는 이유를 분석합니다. 특히, 표준 생성 아키텍처가 한정된 데이터로 훈련될 시 OOD 지역에서 어떻게 성능 저하를 겪는지를 보여줍니다. 논문은 또한 모델이 훈련 데이터의 '메모리화' 전략을 사용하여 OOD 일반화를 시도하는 방식의 문제점을 지적합니다.

- **Technical Details**: 이론적 관점에서, 사전 훈련된 분리된 표현의 사용이 OOD 일반화에 부족함을 나타냅니다. 연구는 2D Gaussian '범프' 생성 작업을 통해 입력 표현이 어떻게 왜곡되고 재조합되는지를 확인합니다. 구현된 아키텍처는 정규화 및 특별한 훈련 데이터를 통해 분리된 표현을 유지하도록 디자인되었으며, 이는 OOD 지역에서 데이터 효율성과 효과적인 학습으로 이어졌습니다.

- **Performance Highlights**: 이 결과는 단지 분리된 표현이 OOD 일반화를 보장하기에는 부족하다는 것을 경고합니다. 대신, 모델은 표현 공간에서 직접적으로 분리화를 유지하며 학습해야 합니다. 연구는 CNN 디코더의 레이어 별로 분리화가 감소하는 방식과 구조적 제약 또는 정교한 데이터셋 사용으로 구성적 규칙을 학습하는 방법도 보여줍니다.



### Survey and Improvement Strategies for Gene Prioritization with Large Language Models (https://arxiv.org/abs/2501.18794)
Comments:
          11 pages, 4 figures, 10 pages of supplementary figures

- **What's New**: 이 논문에서는 희귀 질병 진단의 어려움과 이를 해결하기 위해 다양한 LLMs(large language models)를 벤치마킹하여 인과 유전자( causal genes) 우선순위를 정했습니다. 특히, Human Phenotype Ontology (HPO) 분류 체계를 통해 환자들을 표현형(phenotype)과 해결 가능성(solvability) 수준에 따라 범주화했습니다. 

- **Technical Details**: 제안된 방법론은 'divide-and-conquer' 전략을 활용하여 대규모 유전자 세트를 작은 하위 집합으로 나누어 분석하고, 이 과정에서 GPT-4가 기본 성능에서 약 30%의 인과 유전자 정확도를 기록하며 다른 LLMs 보다 뛰어난 성능을 보였습니다. 우리가 관찰한 바에 따르면, 특정 표현형이나 명확한 연관성이 있는 사례는 문제 해결에 있어 더 높은 정확도를 나타냈으며, 아직 연구되지 않은 유전자에 대한 편향이 존재했습니다.

- **Performance Highlights**: 결국, 우리의 접근법은 기존 진단 방법에 비해 인과 유전자 식별 정확도를 개선했으며, 희귀 질병 진단을 간소화하고 미해결 사례의 재분석을 용이하게 했습니다. 이로 인해 타겟 진단 및 치료 개발을 지원할 수 있는 가능성을 보여줍니다.



### OT-Transformer: A Continuous-time Transformer Architecture with Optimal Transport Regularization (https://arxiv.org/abs/2501.18793)
- **What's New**: 이 논문에서는 Transformer의 연속 시간(continuous-time) 모델링을 제안하고 있습니다. Transformer 블록에 의해 매개화된 동적 시스템을 고려하며, 최적 수송(optimal transport) 이론을 활용하여 훈련 문제의 안정성과 일반화 능력을 향상시킵니다. 기존의 Transformer 아키텍처를 약간 수정함으로써 이 모델을 유연하게 적용할 수 있습니다.

- **Technical Details**: 논문에서는 Hidden state의 동적 행동을 제어하는 ODE(Ordinary Differential Equation)을 통한 Transformer 블록의 연속 시간 구조를 탐구합니다. 이는 Neural ODE 프레임워크에서 영감을 받았으며, 최적 수송 이론을 활용하여 hidden state의 동적 과정을 정규화합니다. 이 정규화는 실험적 및 이론적으로 효과성을 입증하며, 모델의 안정성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 OT-Transformer 모델은 기존의 Transformer 아키텍처를 개선하며 더 적은 파라미터로도 높은 성능을 보여줍니다. 자연어 처리, 이미지 분류 및 점 클라우드(classification of point clouds)와 같은 다양한 작업에서 우수한 결과를 내며, 연속 시간 Transformer 모델들을 초월하는 성과를 기록했습니다.



### Overestimation in LLM Evaluation: A Controlled Large-Scale Study on Data Contamination's Impact on Machine Translation (https://arxiv.org/abs/2501.18771)
- **What's New**: 본 논문에서는 데이터 오염(data contamination)이 언어 모델의 평가 기준에 미치는 영향을 체계적으로 분석합니다. 1B 및 8B 규모의 모델을 대상으로 하여, 기계 번역 작업을 통해 오염의 다양한 상태와 데이터 형식에 따른 성능 메트릭에 미치는 영향을 측정합니다. 이를 통해 소스와 타겟 모두가 포함된 오염이 성능을 상당히 부풀려 비효율성을 초래함을 발견하였습니다.

- **Technical Details**: 연구 방법론은 먼저 깨끗한 훈련 및 테스트 데이터 세트를 구성한 후, 오염을 다양한 단계에서 체계적으로 주입하는 구조로 진행되었습니다. 연구진은 n-gram 검색 알고리즘을 통해 미리 평가 데이터와 겹치는 부분을 탐색하고, 이를 통해 불필요한 오염 데이터를 제거하고 훈련을 진행합니다. 이를 바탕으로 42424242개의 오염 조건을 조사하여, 오염의 영향을 정량적으로 분석했습니다.

- **Performance Highlights**: 결과적으로, 소스와 타겟 쌍이 포함된 오염은 성능 부풀리기를 초래하며, 8B 모델에서는 최대 30 BLEU 포인트의 성능 증가가 관찰되었습니다. 시간적 분포에 따라 집중된 지점에서 오염이 발생했을 경우 성능 부풀리기는 커지지만, 지속적으로 오염이 이루어질 경우 가장 영향을 미친다고 합니다. 모델 규모가 커질수록 오염의 영향을 더 민감하게 받아들이는 경향을 보였습니다.



### Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization (https://arxiv.org/abs/2501.18768)
- **What's New**: 이 논문에서는 오프라인 데이터셋을 바탕으로 보상 함수(reward function)를 극대화하는 오프라인 모델 기반 최적화(offline model-based optimization, MBO)의 새로운 접근법인 DynAMO를 제안합니다. DynAMO는 다양한 최종 후보를 제안하는 데 중점을 두어 디자인 다양성(diversity)을 명시적 목표로 설정합니다. 이를 통해 최적 및 준최적 디자인 구성을 포착할 수 있습니다.

- **Technical Details**: DynAMO의 핵심 아이디어는 디자인 다양성을 생성된 디자인의 분포(distribution)가 오프라인 데이터셋이 내포한 고유의 다양성을 포착하도록 하는 분포 매칭 문제(distribution matching problem)로 수식화하는 것입니다. 이를 통해 기존의 최적화 방법들과 결합하여 사용할 수 있습니다. 이를 통해 다양한 디자인을 제안함과 동시에 품질 높은 후보를 발견할 수 있도록 합니다.

- **Performance Highlights**: 다양한 과학 분야에서 실시된 광범위한 실험 결과, DynAMO는 제안된 디자인의 다양성을 개선하는 데 있어 현저한 성과를 보여줍니다. 이는 MBO 문제를 해결하는 데 있어 새로운 디자인을 제안하는 데 긍정적인 영향을 미칩니다. 결과적으로 DynAMO는 기존 방법 대비 더 많은 다양성을 지닌 디자인 후보를 발견할 수 있는 가능성을 열어줍니다.



### Breaking the Fake News Barrier: Deep Learning Approaches in Bangla Languag (https://arxiv.org/abs/2501.18766)
Comments:
          6 pages, THE 15th INTERNATIONAL IEEE CONFERENCE ON COMPUTING, COMMUNICATION AND NETWORKING TECHNOLOGIES (ICCCNT)

- **What's New**: 디지털 플랫폼의 급속한 발전으로 인해 불확실한 데이터의 확산이 심화되어 있으며, 이는 특히 벵골어를 사용하는 커뮤니티에서 판단력을 저하시켰습니다. 이 연구는 Gated Repetitive Unit (GRU)이라는 심층 학습 기술을 이용하여 벵골어 가짜 뉴스를 인식하는 효과적인 전략을 제시합니다. 새롭게 개발된 데이터셋은 58,478개의 샘플로 구성되어 있으며, 가짜 뉴스 탐지를 위한 강력한 기반을 제공합니다.

- **Technical Details**: 이 연구에서 제안하는 접근 방식은 데이터 전처리 과정에서 어간 추출(lemmatization), 토큰화(tokenization) 및 비대칭 데이터 문제 해결을 위한 오버샘플링(oversampling)을 포함합니다. GRU 모델을 중심으로 한 시스템 설계와 모델 준비 및 평가에 대한 상세한 설명을 제공합니다. 모델의 성능은 정밀도(precision), 재현율(recall), F1 점수(F1 score), 및 정확도(accuracy)와 같은 신뢰할 수 있는 지표를 통해 평가되었습니다.

- **Performance Highlights**: 제안된 GRU 기반의 모델은 94%라는 놀라운 정확도로 가짜 뉴스를 효과적으로 탐지하였습니다. 연구의 기여로는 벵골어 가짜 뉴스 탐지를 위한 대규모 데이터셋이 구축되었으며, 이 모델은 기존의 다른 벵골어 가짜 뉴스 탐지 모델들보다 뛰어난 성능을 보였습니다. 이 모델은 향후 가짜 뉴스 퇴치에 큰 도움이 될 것으로 기대됩니다.



### Synthetic Data Generation for Augmenting Small Samples (https://arxiv.org/abs/2501.18741)
- **What's New**: 이 연구에서는 건강 연구에서 자주 발생하는 소규모 데이터셋을 다루기 위해 데이터 증강(data augmentation)의 유용성을 입증하였습니다. 소량의 샘플에서 머신러닝 모델의 일반화 성능이 떨어지는 문제를 해결하기 위해, 증강 방법이 작은 데이터셋의 다양성을 증가시켜 더 나은 예측 성능을 이끌어냄을 보여주었습니다.

- **Technical Details**: 소규모 데이터셋에 대한 증강은 특정 generative model의 성능 차이가 없음을 발견했습니다. 연구에서는 AUC(Area Under the Curve) 성능이 좋은 결과를 보인 데이터를 분석하여, 샘플 수가 적고, 기저 AUC가 낮으며, 고차원 범주형 변수(cardinality categorical variables)가 더 많고, 결과 변수가 균형을 이룰 때 증강이 유효하다는 사실을 증명했습니다.

- **Performance Highlights**: 일곱 개의 소규모 애플리케이션 데이터셋을 평가한 결과, 데이터 증강을 통해 AUC가 평균 15.55% 향상되었습니다. 특히, 0.51에서 0.73으로 증가한 경우는 43.23%의 개선을 보여주었습니다. 증강된 데이터셋의 AUC는 단순 재샘플링(resampling)만으로 얻은 AUC보다 높았으며, 다양성 역시 더 우수하다는 결과를 도출했습니다.



### Neural Graph Pattern Machin (https://arxiv.org/abs/2501.18739)
- **What's New**: 이번 논문에서는 메시지 패싱의 한계를 넘어선 Neural Graph Pattern Machine (GPM)을 제안합니다. GPM은 그래프 패턴에서 직접 학습하여 하위 구조(substructure) 정보를 효과적으로 추출하고 인코딩하는 프레임워크입니다. 이 접근 방식은 그래프 분석에서 중요한 패턴을 식별하며, 메시지 패싱 대비 더 나은 표현력을 제공합니다.

- **Technical Details**: GPM은 랜덤 워크 기반의 토크나이저를 사용하여 그래프 패턴을 샘플링하며, 이 과정을 통해 신속하고 다양한 작업에 적응할 수 있는 방식으로 설계되었습니다. GPM은 정보를 보존하는 그래프 생성 정보를 포착하기 위해 의미 경로와 익명 경로를 별도로 인코딩합니다. 최종적으로 인코딩된 패턴은 Transformer 레이어에 입력되어 다운스트림 작업에 필수적인 패턴을 식별합니다.

- **Performance Highlights**: 실험 결과, GPM은 노드 분류(node classification), 링크 예측(link prediction), 그래프 분류(graph classification), 회귀(regression) 분야에서 기존의 최첨단 모델들과 비교해 우수한 성능을 보였습니다. 또한 GPM은 아웃 오브 디스트리뷰션(out-of-distribution) 문제에 강하고, 대규모 그래프 및 분산 학습에 대한 확장성이 뛰어나며, 모델 해석 가능성도 개선된 것으로 나타났습니다.



### Integrating LMM Planners and 3D Skill Policies for Generalizable Manipulation (https://arxiv.org/abs/2501.18733)
- **What's New**: 이 논문에서는 LMM-3DP라는 프레임워크를 소개합니다. 이는 대형 다중 모달 모델(LMM)과 3D 스킬 정책을 통합하여 로봇의 하이레벨 계획과 로우레벨 제어를 연결하는 새로운 방법론입니다. LMM-3DP는 환경의 변화를 반영하여 고수준 계획과 저수준 제어를 통합함으로써 로봇의 자율성을 크게 향상시킵니다.

- **Technical Details**: LMM-3DP는 고수준 계획을 위한 세 가지 주요 모듈을 포함하고 있습니다: 1) 시각적 피드백을 통한 상호작용 계획, 2) 계획의 자기 개선과 비평 에이전트의 도입, 3) 기술 라이브러리를 통한 지속적인 학습. 로우레벨 제어를 위해, 3D 변환기를 사용하는 언어 기반의 다중 작업 정책을 개발하여 다양한 기술을 효율적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, LMM-3DP는 장기 과제에서 평균 56.5%의 정확도를 기록했습니다. 이는 기존 베이스라인 모델의 7% 평균 정확도와 대비될 때 매우 높은 성과입니다. 이 논문에서는 비평 에이전트 및 시각적 피드백의 기여도를 평가하기 위해 여러 실험을 설계하여 LMM-3DP의 성능을 증명했습니다.



### Exploring Audio Editing Features as User-Centric Privacy Defenses Against Emotion Inference Attacks (https://arxiv.org/abs/2501.18727)
Comments:
          Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop on Privacy-Preserving Artificial Intelligence

- **What's New**: 이 논문은 음성 기반 감정 인식 공격에 대한 사용자 중심의 새로운 개인 정보 보호 접근 방식을 제안합니다. 기존 솔루션들은 사용성을 저해하는 경우가 많았으나, 이번 연구는 사용자가 친숙한 오디오 편집 기술인 피치(pitch)와 템포(temp) 조작을 활용하여 감정 개인 정보를 보호하는데 중점을 두었습니다. 이로 인해 사용자에게 익숙한 도구를 통해 사전 지식 없이도 보다 쉽게 접근할 수 있도록 하였습니다.

- **Technical Details**: 연구진은 Google Play Store와 Apple App Store에서 제공되는 오디오 편집 응용 프로그램의 기능을 분석하여 피치와 템포 조정 기능의 효과성을 평가했습니다. 다양한 인공지능 모델을 포함한 위협 모델을 디자인하여, Deep Neural Networks(DNNs)와 Large Language Models(LLMs)를 포함한 다양한 공격 방식에 대한 대응력을 평가했습니다. 실험은 세 가지 서로 다른 데이터세트를 사용하여 진행되었으며, 피치와 템포 조작이 효과적으로 감정 데이터를 감춘다는 결과를 도출했습니다.

- **Performance Highlights**: 연구 결과, 피치와 템포 조작은 감정 인식 공격에 대한 효과적이고 사용자 친화적인 개인 정보 보호 메커니즘으로 작용할 수 있다는 유망한 결과를 보여주었습니다. 기존의 안전 솔루션들과의 비교를 통해 사용자의 접근성과 실용성을 고려한 새로운 접근 방식의 가능성을 확인하였습니다. 이러한 발견은 다양한 디바이스와 플랫폼에서의 적용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### Scaling Policy Gradient Quality-Diversity with Massive Parallelization via Behavioral Variations (https://arxiv.org/abs/2501.18723)
- **What's New**: 본 논문에서는 Quality-Diversity (QD) 최적화의 새로운 접근 방식을 제안합니다. 특히, 기존 중앙 집중식 actor-critic (AC) 교육 방식에 의존하지 않는 PG 기반 변이 연산자 ASCII를 도입했습니다. 이를 통해 새로운 행동을 생성하고, 대규모의 병렬화가 가능하며, 수행 속도를 비약적으로 개선합니다.

- **Technical Details**: 우리의 방법인 ASCII-ME는 Markov Decision Process (MDP) 프레임워크를 활용하여 시간 단계 성능 지표를 기반으로 행동 변화를 생성하고, 이를 정책 경량화에 활용합니다. 또한, 기존의 Iso+LineDD 유전 연산자와 결합하여 성능과 효율성을 동시에 향상시키는 방식으로 동작합니다. 이 알고리즘은 대규모 DNN 정책 최적화에 강력한 장점을 제공하며, 매개변수 수가 많은 모델에서도 경쟁력 있는 샘플 및 실행 속도를 달성합니다.

- **Performance Highlights**: 실험 결과는 ASCII-ME가 다섯 개의 연속 제어 로코모션 작업에서 기존 알고리즘들보다 평균적으로 5배 빠른 속도로 우수한 성능을 발휘하는 것을 보여줍니다. 또한, 동일한 평가 예산 하에서도 병렬 평가를 통해 성능 및 실행 속도를 극대화 할 수 있는 강력한 확장성을 입증하였습니다. 이로 인해 다양한 고성능 DNN 정책을 단시간 내에 생성할 수 있는 가능성을 열었습니다.



### High-Accuracy ECG Image Interpretation using Parameter-Efficient LoRA Fine-Tuning with Multimodal LLaMA 3.2 (https://arxiv.org/abs/2501.18670)
- **What's New**: 이번 연구는 electrocardiogram (ECG) 이미지를 해석하는 새로운 접근법을 제시합니다. 우리는 multimodal LLaMA 3.2 모델을 사용하여, Low-Rank Adaptation (LoRA)라는 파라미터 효율적인 미세 조정 전략을 통해 ECG 분석을 강화했습니다. 이 방법은 임상 상황에서 다양한 심장 병리 식별의 정확성을 높이기 위한 목적으로 설계되었습니다.

- **Technical Details**: 모델은 LLaMA 3.2 아키텍처를 기반으로 하며, ECG 이미지의 다양한 특성을 효과적으로 포착하기 위한 두 단계의 비전 인코더로 구성되어 있습니다. 비전 인코더는 먼저 이미지를 표준 크기로 리사이징한 후, 14x14 픽셀의 비겹치는 패치로 나누어 고급 특성 추출을 수행합니다. LoRA 방법론 덕분에 특정 레이어를 제외하고 소수의 파라미터 만 업데이트하여 ECG 전문가 수준의 해석 능력을 달성했습니다.

- **Performance Highlights**: 실험 결과, 우리의 LoRA 미세 조정 방법은 ECG 이미지 해석에서 기존 모델을 능가하고 70가지 이상의 심장 질환을 높은 정확도로 식별할 수 있음을 보여주었습니다. 이 연구는 이미지 기반의 ECG 분석에서 중요한 진전을 이루었으며, 전통적인 CNN 기반 방법과 비견될 수 있는 성능을 달성했습니다. 이러한 성과는 다양한 심장 질환에 대한 신뢰할 수 있는 임상적 결정 지원을 위한 기반을 마련해 줍니다.



### The Pitfalls of "Security by Obscurity" And What They Mean for Transparent AI (https://arxiv.org/abs/2501.18669)
Comments:
          27 pages, abbreviated version in AAAI 2025

- **What's New**: 이번 연구에서는 인공지능(AI) 분야가 보안 커뮤니티의 투명성 경험에서 어떤 통찰을 얻을 수 있는지를 조사합니다. AI 시스템의 투명성에 대한 수요가 증가하고 있지만, 기업들이 그들의 데이터 수집 및 인공지능 모델 훈련 방법을 숨기려는 경향이 여전합니다. 이에 따라 다양한 이해관계자들이 AI의 투명성을 확보하기 위해 나서고 있으며, 본 논문은 이러한 요구와 보안 커뮤니티의 오랜 경험을 연결시키고 있습니다.

- **Technical Details**: 논문은 보안 커뮤니티가 발전시킨 투명성 원칙을 AI 투명성에 적용할 수 있는 방법론을 제시합니다. 특히, 보안 커뮤니티는 'obscurity'(불투명성)에 의존하기보다는 'transparency'(투명성)를 통해 시스템을 더 안전하게 만들 수 있다는 사회적 합의를 이룩했습니다. 또한, 투명성을 확보함으로써 부정적 효과를 방지할 수 있는 중요한 패턴을 확인하며, 이는 인공지능 적용 시에도 중요한 고려사항입니다.

- **Performance Highlights**: 본 논문은 보안 원칙들이 AI 커뮤니티의 투명성 논의에 어떻게 관련되어 있는지를 탐색하고, AI 시스템의 독특한 도전 과제들에 대해 논의합니다. 특히, 유사한 주제를 다루면서도, AI 시스템은 보안 시스템과는 다른 투명성 요구사항과 제약을 가지고 있음을 강조하고 있습니다. 이를 통해 AI의 투명성 확보가 필요하다는 점과 함께, 해당 과정에서 발생할 수 있는 다양한 논란들에 대한 논의도 포함하고 있습니다.



### Structure Development in List-Sorting Transformers (https://arxiv.org/abs/2501.18666)
Comments:
          15+19 pages, 6+13 figures

- **What's New**: 이번 연구에서는 단일 레이어의 attention-only transformer 모델이 리스트 정렬 작업을 학습하는 과정에서 얼마나 관련 구조를 발전시키는지를 살펴봅니다. 훈련이 끝난 후, 모델은 주로 vocabulary-splitting과 copy-suppression이라는 두 가지 모드로 attention heads를 조직합니다. 여기서 vocabulary-splitting은 결과적으로 더 단순한 방식으로 되어 있는 것으로 나타났으며, 이는 신경망이 자연적으로 더 간단한 솔루션을 선호한다는 주장을 지지합니다.

- **Technical Details**: 모델은 QK(Query-Key)와 OV(Output-Value) 회로를 활용하여 attention이 기능하는 방식을 포착합니다. QK 회로는 입력 숫자 토큰이 가장 가까운 숫자 토큰에 주의를 기울이는 방식으로 작동하며, OV 회로는 문맥 내 숫자 토큰을 복사합니다. 이러한 회로의 역할 구분과 학습 데이터에서 발견된 특성들은 모델 내부의 조직화 방식에 깊이 연관되어 있습니다.

- **Performance Highlights**: 연구를 통해 vocabulary-splitting은 기존의 여러 heads가 서로 겹치는 숫자 범위를 처리하는 것보다 간단한 솔루션을 나타내며, 이는 학습 데이터의 특성에 의해 주도되는 것으로 나타났습니다. 또한, copy-suppression 기능이 모델의 자신감을 조절하는 방식에 대한 새로운 통찰을 제공하여, 저희 연구가 보다 복잡한 모델에서의 구조 이해에 도움이 될 수 있음을 보여줍니다.



### BARNN: A Bayesian Autoregressive and Recurrent Neural Network (https://arxiv.org/abs/2501.18665)
- **What's New**: 이번 연구는 BARNN이라는 새로운 변형 베이지안 오토회귀 및 순환 신경망 모델을 제안합니다. 이 모델은 오토회귀 및 순환 모델을 베이지안 버전으로 변화시켜 불확실성을 처리하는 데 필요한 엄격한 프레임워크를 제공합니다. 기존 모델의 일반화 문제를 해결함으로써 자연과학 및 기계 학습 분야에서의 활용 가능성을 높이고 있습니다.

- **Technical Details**: BARNN은 변형 드롭아웃(variational dropout) 방법에 기초하여 대규모 순환 신경망에 적용될 수 있는 가능성을 가지며, PODE 모델링과 분자의 생성과 같은 여러 응용 분야에서 유용하게 사용됩니다. 또한, 베이지안 추론을 효율적으로 수행하기 위해 '시간적 변형 혼합 후분포' 사전(tVAMP-prior)을 도입했습니다. 이를 통해 베이지안 모델이 얼마나 잘 보정되는지를 증명하고 장기간 의존성을 모델링하는 데 뛰어난 성능을 보이고 있습니다.

- **Performance Highlights**: BARNN은 PDE 모델링과 분자 생성 실험에서 기존 방법들과 비교하여 동등하거나 더 나은 정확성을 달성함과 동시에 불확실성 정량화에서 우수한 성과를 보여줍니다. 이 모델은 오토회귀 혹은 순환 모델의 최소 수정으로 베이지안 버전으로 전환할 수 있는 최초의 접근법이며, 그 결과 더 높은 정확성과 불확실성의 체계적인 정량화를 지원합니다.



### Rethinking the Upsampling Layer in Hyperspectral Image Super Resolution (https://arxiv.org/abs/2501.18664)
- **What's New**: 이 논문에서는 경량화된 단일 하이퍼스펙트럴 이미지 슈퍼해상도(SHSR) 네트워크인 LKCA-Net을 제안합니다. LKCA-Net은 하이퍼스펙트럴 이미지의 다중 스케일 채널 특성을 조정하기 위해 채널 주의(channels attention) 메커니즘을 포함하고 있습니다. 또한, 배울 수 있는 업샘플링(layer)에서의 저순위(low-rank) 성질이 경량 SHSR 방법의 주요 병목현상임을 입증하였습니다.

- **Technical Details**: 저자는 저순위 근사(low-rank approximation) 전략을 사용하여 배울 수 있는 업샘플링 레이어의 매개변수 중복을 최적화하고, 이러한 낮은 순위 근사가 네트워크의 특성 표현 능력을 유지하기 위한 지식 증류(knowledge distillation) 기반의 특성 정렬(feature alignment) 기술을 도입합니다. 이 연구는 기존의 복잡한 SHSR 네트워크 구조를 변경하지 않고도 업샘플링 레이어의 전통적인 구조를 최적화할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과는 LKCA-Net이 Chikusei, Houston 2018 및 Pavia Center 데이터셋에서 최첨단 방법들(SOTAs)과 비교했을 때 경쟁력 있는 성능을 보였으며, 다른 SHSR 방법들에 비해 수십 배에서 수백 배까지 속도 향상을 달성할 수 있음을 보여줍니다. 이러한 결과는 저순위 근사 및 특성 정렬 전략의 효과성을 강조하며, 저자들이 제안한 LKCA-Net의 구조가 실질적인 성능 개선에 기여함을 말해줍니다.



### Joint Optimization of Prompt Security and System Performance in Edge-Cloud LLM Systems (https://arxiv.org/abs/2501.18663)
- **What's New**: 이 논문은 Edge-Cloud LLM (EC-LLM) 시스템에서 다양한 프롬프트 공격 아래에서 프롬프트 보안, 서비스 지연(latency), 시스템 자원 최적화를 공동으로 고려하는 새로운 접근법을 제안합니다. 특히, 베르트(Bert) 기반의 프롬프트 탐지기와 벡터 데이터베이스(VDB)를 통합한 EC-LLM 아키텍처를 통해 프롬프트 보안을 강화하는 방법을 제시합니다. 새로운 다단계 동적 베이지안 게임 모델을 통해 시스템 자원과 서비스 지연을 최적화하는 첫 번째 연구로 언급됩니다.

- **Technical Details**: 본 연구에서 제안된 모델은 벡터 데이터베이스(VDB)를 각 엣지 서버(ES)에 배치하여 정상 및 악성 프롬프트 데이터셋을 관리합니다. 또한, 불완전한 정보 하에서 악성 프롬프트 수를 예측하기 위해 순차적 한계 분석 방법을 적용합니다. 이러한 전략은 시스템 보안과 성능 간의 균형을 맞추는 데 중점을 두어 자원 할당을 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안한 알고리즘은 기존 최첨단 알고리즘에 비해 개선된 보안과 함께 일반 사용자에 대한 서비스 지연을 줄이고 시스템 자원 소비를 감소시켰습니다. 평균 토큰 당 지연(latency)은 약 16.24% 감소하였고, 평균 GPU 메모리 소비는 거의 17.7% 감소했으며, 평균 GPU 활용도는 약 17.87% 감소했습니다.



### Cogito, ergo sum: A Neurobiologically-Inspired Cognition-Memory-Growth System for Code Generation (https://arxiv.org/abs/2501.18653)
- **What's New**: 이번 논문에서는 Multi-Agent Systems (MAS)를 기반으로 한 대규모 언어 모델의 새로운 접근인 Cogito를 제안합니다. Cogito는 코드 생성 작업에서 문제 해결 능력을 향상시키기 위해 신경 생물학적으로 영감을 받은 다중 에이전트 프레임워크를 적용하며, 기존의 개발 순서를 반대로 설정했습니다. 이는 계획 단계가 아닌 디버깅, 코딩, 그리고 마지막으로 계획을 진행하는 과정을 따릅니다.

- **Technical Details**: Cogito의 구조는 인간 학습 과정을 모방하여 각 단계에서 지식을 점진적으로 습득하는 방식입니다. 이 시스템은 해마(hippocampus)와 유사한 메모리 모듈을 설계하여 유사한 작업에 대한 빠른 지식 검색을 가능하게 하며, 이를 통해 각 단계에서 인지 능력을 발전시키는 것을 목표로 합니다. 이러한 접근은 코딩 작업을 수행하는 슈퍼 롤 에이전트를 형성하여 효율성을 높입니다.

- **Performance Highlights**: Cogito는 대표적인 기초 모델과 비교한 많은 실험을 통해 우수한 성능을 보여주었습니다. 코드 생성 작업에서의 효율성을 높여주는 점이 특히 강조되었습니다. 이러한 성과는 Cogito가 인간의 성장 중심의 학습 모델을 기반으로 하고 있음을 잘 나타냅니다.



### Fake News Detection After LLM Laundering: Measurement and Explanation (https://arxiv.org/abs/2501.18649)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)로 생성된 패러프레이즈 가짜 뉴스를 탐지하는 효율성을 평가하며, 패러프레이즈 과정이 탐지 파이프라인에 도움이 되는지 또는 방해되는지를 분석합니다. 연구 결과, 탐지기가 인간이 작성한 글보다 LLM 패러프레이즈 가짜 뉴스를 탐지하는 데 더 어려움을 겪고 있으며, 감정 변화(sentiment shift)가 탐지 실패에 기여하는 가능성을 발견하였습니다.

- **Technical Details**: 연구에 사용된 기술적 방법론은 LIME 설명을 통해 탐지 실패의 원인을 조사하고, BERTSCORE와 같은 컨텍스트 기반의 임베딩을 활용하여 패러프레이즈의 품질을 평가하는 것입니다. 또한, LLM 모델들이 생성한 다양한 패러프레이즈 텍스트의 검출 용이성을 비교하고, 효과적인 탐지기를 식별하는 데 집중하고 있습니다. 패러프레이즈 생성은 기존의 기술을 넘어 최신의 딥 러닝 모델과 사전 학습된 모델을 사용하여 고급 기술로 발전하고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 패러프레이즈 과정이 탐지 시스템의 성능에 미치는 영향은 상당히 복잡한데, 일부 탐지 모델은 패러프레이즈된 가짜 뉴스를 더 잘 탐지하는 경향이 있습니다. 그러나 대다수의 LLM 모델로 생성된 텍스트는 탐지기를 속이는 경향이 있어, 전반적으로 탐지 성능이 저하되는 문제가 발생하고 있습니다. 마지막으로, GitHub에서 사용할 수 있는 새로운 패러프레이즈 출력과 점수를 포함한 데이터셋이 공개되어 연구자들이 활용할 수 있는 기반이 마련되었습니다.



### Layered Chain-of-Thought Prompting for Multi-Agent LLM Systems: A Comprehensive Approach to Explainable Large Language Models (https://arxiv.org/abs/2501.18645)
- **What's New**: 이 논문에서는 Layered Chain-of-Thought (Layered-CoT) Prompting이라는 새로운 방법론을 제시합니다. 기존의 chain-of-thought (CoT) 방식이 직면했던 한계를 극복하기 위해, 이 방법은 단계적으로 여러 레이어로 성격을 나누어 중간 추론을 검증합니다. Layered-CoT는 의료, 재정 위험 평가 및 민첩한 엔지니어링과 같은 복잡한 시나리오에서 CoT보다 더 높은 투명도, 정확성 및 사용자 참여를 자랑합니다.

- **Technical Details**: Layered-CoT는 각 레이어별로 전문 검증과 피드백을 받을 수 있도록 고안되었습니다. 모델은 문제를 레이어로 나누고, 각 레이어는 제한된 범위의 부분적 체인을 생성하여 외부 자원(예: 도메인 데이터베이스, 지식 그래프)에 의해 검증됩니다. 이 접근법은 오류 전파를 방지하고 사용자에게 즉각적인 피드백을 가능하게 하며, 다중 에이전트 시스템을 통해 각 레이어의 검증 절차가 더욱 강화됩니다.

- **Performance Highlights**: Layered-CoT는 다양한 분야에서 기존 CoT 방법론에 비해 개선된 성능을 시연하고 있습니다. 특히, 각 단계에서의 외부 데이터 확인과 사용자 피드백의 통합은 최종 결론을 더 신뢰할 수 있게 만들어줍니다. 이를 통해 각 레이어가 철저히 검증되어 과거의 오류가 후속 단계로 유입되는 것을 방지하며, 이러한 방식이 고위험 영역에서 신뢰할 수 있는 설명의 제공을 가능하게 합니다.



### 3D Reconstruction of Shoes for Augmented Reality (https://arxiv.org/abs/2501.18643)
- **What's New**: 이 논문에서는 3D 모델링과 증강 현실(AR)을 활용한 모바일 기반 솔루션을 소개하여 온라인 신발 쇼핑 경험을 향상시킵니다. 기존의 정적 2D 이미지의 한계를 극복하고, 2D 이미지에서 현실적인 3D 신발 모델을 생성하는 새로운 방법을 적용하였습니다. 평균 Peak Signal-to-Noise Ratio (PSNR) 0.32를 달성하며, 3120개의 이미지로 구성된 맞춤형 신발 세분화 데이터셋을 개발하여 Intersection over Union (IoU) 0.95를 기록한 최상의 세분화 모델을 구현했습니다.

- **Technical Details**: 본 연구는 3D Gaussian Splatting 기술을 활용하여 2D 이미지로부터 현실적인 3D 모델을 효율적으로 생성하는 방법을 제안합니다. 특히, 자주 발생하는 인간의 개입을 줄이며 고속의 모델 생성을 가능하게 하여, AR 응용 프로그램에서의 즉각적인 상호작용을 지원합니다. 이를 위해 이미지의 배경 제거와 신발 세분화를 위한 YOLOv8 모델을 사용하여 세밀한 결과를 얻었습니다.

- **Performance Highlights**: 제안된 방법은 3D 모델 생성에 있어 기존 방법에 비해 월등한 성능을 발휘하며, 실제 쇼핑 경험을 가상으로 재현하는 데 기여합니다. 특히, 비디오 데이터에서 3030개의 이미지를 활용하여 6.5MB 크기의 최종 모델을 제작함으로써 서버 호스팅에 필요한 계산 자원과 비용을 대폭 절감했습니다. 이 연구는 패션 산업 전반에 걸쳐 적용 가능한 혁신적인 가상 상호작용을 가능하게 합니다.



### DebiasPI: Inference-time Debiasing by Prompt Iteration of a Text-to-Image Generative Mod (https://arxiv.org/abs/2501.18642)
Comments:
          This work was presented at The European Conference on Computer Vision (ECCV) 2024 Workshop "Fairness and ethics towards transparent AI: facing the chalLEnge through model Debiasing" (FAILED), Milano, Italy, on September 29, 2024, this https URL

- **What's New**: 이 연구에서는 이미지 생성 시 인구 통계학적 속성의 분포를 제어할 수 있는 새로운 추론 시간 프로세스인 DebiasPI(Debiasing-by-Prompt-Iteration)를 제안합니다. 기존의 방법들은 모델을 재훈련해야 하거나 특정 성별 및 인종을 반영한 이미지를 생성하는 데 어려움을 겪고 있었던 반면, DebiasPI는 사용자가 생성된 이미지의 속성을 추적하여 요구하는 속성을 선택할 수 있는 방법을 제공합니다.

- **Technical Details**: DebiasPI는 모델의 내부 상태를 probe하거나 외부 속성 분류기를 통해 생성된 속성을 추적하며, 생성된 이미지에서의 속성 분포를 비교하는 도구를 제공합니다. 이 방법론은 질적 및 양적 콘텐츠 분석(QCA)를 통해 AI가 생성한 이미지의 인식 속성을 레이블링하는 코드북을 개발하여 수동 평가 과정도 포함합니다. 이를 통해 인종, 성별, 피부 톤의 다양성을 평가하고, 윤리적 개입 유도를 위한 충분한 데이터 기반을 제공합니다.

- **Performance Highlights**: DebiasPI를 사용하여 우리가 실험한 이미지 생성 결과는 성별 및 인종의 동등한 표현을 보여주는 것에 성공했습니다. 그러나 피부 톤의 다양성이 떨어지는 부작용이 발견되었고, 특정 인종의 피부 톤을 생성하는 데 어려움이 있었습니다. 다양한 개입 프롬프트를 통한 실험 결과, 모델은 여전히 젊고 남성적인 캐릭터를 생성하는 경향이 있으며, 이는 윤리적 개입의 필요성을 더욱 강조합니다.



### Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation (https://arxiv.org/abs/2501.18638)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 연구는 고급 콘텐츠 정책에서 파생된 은밀한 jailbreak 프롬프트 생성을 자동화하는 모듈형 파이프라인을 제시합니다. 이를 통해 LLM 콘텐츠 조정의 효율성을 개선하고, 기존 알고리즘에 비해 쿼리 수를 54% 감소시키면서도 92%의 공격 성공률을 달성했습니다. 또한, LLM을 사용하여 고급 정책에서 시드 프롬프트를 자동 생성함으로써 차가운 시작 문제를 해결했습니다.

- **Technical Details**: 이 연구에서 도입된 Graph of Attacks with Pruning(GAP) 방법은 공격 효율을 높이고 은닉성을 극대화하기 위해 동적이고 상호 연결된 추론을 활용합니다. GAP는 기존의 선형 구조를 그래프 형태로 전환하여 상대방 LLM에 대한 공격 벡터를 정제하고, 더 포괄적인 탐색을 가능하게 합니다. 파이프라인은 PromptGuard를 미세 조정하기 위해 생성된 프롬프트를 사용하여 다양한 해로운 콘텐츠 유형에 대한 탐지 능력을 더욱 향상시킵니다.

- **Performance Highlights**: GAP는 GPT-3.5를 대상으로 한 실험에서 96%의 공격 성공률을 기록하며 기존 TAP(78%)을 훨씬 초과했습니다. 또한, GAP-M은 Gemma-2-9B와 Qwen2.5-7B 모델에서 각각 100%의 성공률을 기록하며 기존의 낮은 성공률과 높은 쿼리 수를 능가했습니다. 이러한 결과는 GAP의 효과적인 적대적 공격 생성 능력을 입증하며, 콘텐츠 조정 시스템의 지속적인 개선을 위한 중요한 기반을 제공합니다.



### SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Mod (https://arxiv.org/abs/2501.18636)
- **What's New**: 이 논문에서는 RAG(검색 보강 생성) 시스템의 보안을 평가하기 위한 새로운 벤치마크인 SafeRAG를 소개합니다. RAG의 외부 지식을 통합하는 것은 지식 집약적 작업 해결에 매우 유용하지만, 이로 인해 공격자들이 시스템을 조작할 가능성이 증가합니다. 연구진은 사악한 텍스트, 인터컨텍스트 충돌, 부드러운 광고 및 백색 서비스 거부(White DoS)라는 네 가지 주요 공격 작업을 통해 RAG의 취약점을 밝혀냈습니다. 이를 통해 RAG의 보안을 체계적으로 평가할 수 있는 새로운 기준을 제시합니다.

- **Technical Details**: SafeRAG 데이터셋은 각 공격 작업에 대한 RAG 보안 평가 데이터셋으로, 주로 수작업으로 구성되었습니다. 이 데이터셋은 RAG 구성 요소 간의 다양한 공격 시나리오를 시뮬레이션하는 데 사용됩니다. 또한, 논문은 '실버 노이즈', '인터컨텍스트 갈등', '소프트 광고 공격', 그리고 '화이트 서비스 거부'와 같은 새로운 공격 작업을 정의하여 기존의 RAG 안전 구성 요소를 우회할 수 있는 방법을 제시합니다. 해당 작업들은 RAG의 다양한 단계에서 발생할 수 있는 보안 위험을 평가하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 14개의 대표적인 RAG 구성 요소에 대한 실험은 RAG가 모든 공격 작업에 대해 상당한 취약성을 드러낸다는 것을 보여줍니다. 특히, 가장 명백한 공격 작업조차 현재의 검색기나 필터, 혹은 고급 LLM을 쉽게 우회하여 RAG 서비스 품질을 저하시킬 수 있음을 발견했습니다. 본론에서 제안된 SafeRAG는 RAG의 취약점을 정밀하게 평가할 수 있는 도구로, 연구자들이 RAG 시스템의 보안을 강화하는 데 기여할 것입니다.



### Indiana Jones: There Are Always Some Useful Ancient Relics (https://arxiv.org/abs/2501.18628)
- **What's New**: 이 논문에서는 인디애나 존스(Indiana Jones)라는 혁신적인 접근법을 소개하여 Large Language Models(LLMs)의 jailbreaking을 수행합니다. 이 방법은 다수의 특수화된 LLM 간의 대화와 키워드 중심의 프롬프트를 활용하여 콘텐츠 안전 장치를 우회하는 데 거의 완벽한 성공률을 달성합니다. 연구는 현대 LLM의 체계적 취약성을 드러내며, 해로운 출력을 유도할 수 있는 간접적인 프롬프트의 위험성을 강조합니다.

- **Technical Details**: 이 연구에서는 새로운 jailbreak 방법을 제안하고, 블랙박스(black-box) 및 화이트박스(white-box) LLM에서의 효과와 효율성을 평가합니다. 시험 설계는 세 개의 LLM을 기반으로 한 다중 라운드 상호작용을 포함하며, 특정 키워드를 기반으로 한 jailbreak 목표를 달성하기 위해 Victim, Suspect, Checker가 상호작용합니다. 이 접근법은 강력한 LLM에 대해 단일 라운드만으로도 성공할 수 있으며, 여러 라운드가 필요한 경우에도 교차점을 이용해 일관성을 높이는 방법을 설명합니다.

- **Performance Highlights**: 제안된 방법은 다양한 LLM에서 거의 완벽한 jailbreak 성공률을 달성하였으며, 이는 현재의 모델에서 존재하는 비합리적 학습 콘텐츠의 문제를 시사합니다. 실험은 Attack Success Rate(ASR), 효율성, 견고성 등의 주요 지표를 기반으로 하여 수행되었으며, 강력한 모델들은 첫 번째 라운드만으로 충분한 결과를 산출하였습니다. 이 연구는 LLM의 안전성에 관한 중요한 기초를 제공하며, 향후 연구에서 LLM이 악의적인 활용에 대한 방어를 강화할 수 있도록 지원합니다.



### The TIP of the Iceberg: Revealing a Hidden Class of Task-In-Prompt Adversarial Attacks on LLMs (https://arxiv.org/abs/2501.18626)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)에 대한 새로운 유형의 jailbreak 적대적 공격인 Task-in-Prompt (TIP) 공격을 소개합니다. TIP 공격은 모델의 프롬프트에 시퀀스-투-시퀀스(task) 작업(예: 암호 해석, 수수께끼, 코드 실행)을 포함시켜 금지된 입력을 간접적으로 생성합니다. 또한 PHRYGE 벤치마크를 도입하여 이러한 공격의 효과성을 체계적으로 평가합니다. 우리 연구는 GPT-4o 및 LLaMA 3.2를 포함한 여섯 개의 최첨단 언어 모델에서 안전 장치를 우회할 수 있음을 입증합니다.

- **Technical Details**: TIP 공격은 모델의 기본 지시 사항 수행 기능을 활용하여 특정 트리거 단어나 질문을 회피하고 안전하지 않은 콘텐츠를 무해한 변환 작업 내에 포함시키는 방식으로 진행됩니다. 이러한 공격은 Caesar cipher, 모스 부호, Base64 등 다양한 인코딩 방법을 사용할 수 있어 탐지하기 어려운 공격 유형입니다. LLMs가 임의의 언어 퍼즐이나 변환을 해결하도록 설계되어 있는 한, 공격자는 금지된 콘텐츠를 간접적으로 다시 도입하는 프롬프트를 만들 수 있습니다.

- **Performance Highlights**: TIP 공격은 기존의 방어 메커니즘을 우회할 수 있는 강력한 방법으로, 모델이 항상 신뢰할 수 있는 작동을 보장하기 위한 보다 정교한 방어 전략이 필요하다는 것을 강조합니다. 연구 결과는 LLM의 안전 정렬에서 존재하는 중요한 약점을 부각시키며, 향후 보안 및 신뢰성 향상을 위한 방향성을 제시합니다. 이러한 TIP 공격은 단순한 공격이 아닌, 전반적인 LLM의 안전성 문제를 야기하는 중요한 발견으로 평가됩니다.



### Membership Inference Attacks Against Vision-Language Models (https://arxiv.org/abs/2501.18624)
Comments:
          Accepted by USENIX'25; 22 pages, 28 figures;

- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 데이터 오용 및 유출 탐지에 대한 최초의 체계적인 분석을 제공합니다. 이 연구는 주로 instruction tuning 데이터에 포함된 민감한 정보의 유출 가능성에 주목하고, membership inference attack (MIA)이라는 새로운 접근 방식을 도입합니다. 특히, VLM의 고유한 파라미터인 온도(temperature)를 기반으로 회원 여부를 추론하는 방법을 제안하여 기존 MIA 방법론의 한계를 극복하려고 합니다.

- **Technical Details**: 이 연구에서는 네 가지 유형의 MIA 방법을 제안합니다. 각 방법은 적대자의 배경 지식에 따라 다르게 설계되며, 이를 통해 다양한 수준의 공격 시나리오를 다룹니다. 특히, VLM의 온도가 데이터 샘플의 회원 및 비회원 상태에 미치는 차이를 활용하여 VLM의 민감성에 대한 새로운 관점을 제공합니다.

- **Performance Highlights**: 제안된 방법은 다양한 VLM 아키텍처에서 평가되었으며, AUC(Area Under Curve)가 0.8을 초과하는 성과를 기록했습니다. 특히, 5개 샘플로 구성된 작은 집합에 대해 회원 상태를 정확하게 판별할 수 있는 능력을 입증했습니다. 이 연구를 통해 VLM의 데이터 사용에 대한 새로운 취약점이 드러나며, 이러한 공격에 대한 경각심을 높이고 있습니다.



### STAMP: Scalable Task And Model-agnostic Collaborative Perception (https://arxiv.org/abs/2501.18616)
Comments:
          Paper is accepted by ICLR 2025

- **What's New**: STAMP는 이종 에이전트를 위한 새로운 협업 인식 프레임워크로, 경량 어댑터-리버터 쌍을 사용하여 각 에이전트의 Bird's Eye View (BEV) 특징을 통합된 프로토콜 BEV 특징 도메인으로 변환합니다. 이 방법은 다양한 모델과 작업에 대해 태스크-모델-무시(task- and model-agnostic) 방식으로 작동하여, 모델 재학습 없이도 통합이 가능하다는 장점을 지닙니다. 또한 이 프레임워크는 큰 디스크 메모리나 컴퓨팅 오버헤드 없이도 여러 이종 에이전트가 협력할 수 있도록 지원합니다.

- **Technical Details**: STAMP의 협업 인식 파이프라인은 각 에이전트의 BEV 특징을 통합하고, 이를 다른 에이전트와 공유하여 지역 도메인으로 다시 매핑하는 방식인 협업 특징 정렬(cFA)을 활용합니다. 실험을 통해, STAMP는 이종 에이전트 수가 증가할수록 자원 성장률을 크게 낮추면서도 유사 이상으로 높은 정확성을 달성할 수 있음을 입증하였습니다. 평균적으로 추가 에이전트당 2.36 GPU 시간의 훈련 시간을 요구하며, 이는 기존 방법에 비해 7.2배 절약된 수치입니다.

- **Performance Highlights**: STAMP의 성능은 두 가지 데이터셋인 OPV2V와 V2V4Real을 사용하여 평가되었으며, 기존의 이종 협업 인식 모델과 비교하여 결과가 우수하거나 동등함을 보였습니다. 특히 STAMP는 이종 모델 간 협업을 체계적으로 지원하는 독특한 능력을 지녀, 자율 주행 분야에서 새로운 기준을 제시합니다. 이는 다른 방법이 작동할 수 없는 상황에서도 분명한 성능 향상을 보여주며, 향후 Level 5 자율주행 시스템 개발에 기여할 것으로 기대됩니다.



### Review and Recommendations for using Artificial Intelligence in Intracoronary Optical Coherence Tomography Analysis (https://arxiv.org/abs/2501.18614)
- **What's New**: 이번 논문은 인공 지능(AI) 기반의 관상 동맥 질환(CAD) 진단에 대한 체계적인 리뷰를 실시하였습니다. 특히, IVOCT(intravascular optical coherent tomography) 이미지를 활용한 AI 모델의 임상 유용성을 평가한 것이 주목할 만합니다. 2015년 1월부터 2023년 2월 사이에 발표된 문헌을 종합하여 AI 기반의 CAD 진단 현황을 면밀히 분석하였습니다.

- **Technical Details**: 연구에서는 총 5,576개의 논문을 검토하였으며, 513편이 초기 스크리닝을 통과하고, 최종적으로 35편이 품질 검사를 통해 선정되었습니다. 대부분의 모델이 방법론적 결함(methodological flaws)과 기본적인 편향(bias)으로 인해 임상 사용에 적합하지 않다는 결론에 도달했습니다. 이러한 문제를 해결하기 위해, 모델 품질 개선 및 연구 관행의 향상에 관한 추천 사항을 제시하였습니다.

- **Performance Highlights**: 발견된 모델의 대다수가 현재 임상에서 활용되기에는 한계가 있으며, 실제 임상 적용 가능성을 갖춘 AI 제품 개발을 위한 노력과 연구의 질 향상이 필요하다는 점을 강조하였습니다. 논문에서는 AI 기술이 CAD 진단의 정확성과 속도를 향상시킬 잠재력을 지니고 있지만, 이를 위해 먼저 기존 연구의 품질을 높여야 한다고 주장합니다.



### Deeply Optimizing the SAT Solver for the IC3 Algorithm (https://arxiv.org/abs/2501.18612)
- **What's New**: 이 논문은 IC3 (Incremental Construction of Inductive Clauses) 알고리즘을 위해 최적화된 경량 SAT solver인 GipSAT를 소개합니다. GipSAT는 IC3에서의 SAT 쿼리의 특성을 분석하여 모든 변수에 대한 결정을 내릴 필요가 없음을 발견했습니다. 또한 이 연구는 binary heap 연산의 오버헤드를 최소화하기 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: GipSAT는 SAT solver의 구조적 최적화를 통해 연산을 상수 시간(constant time) 내에 수행할 수 있도록 설계되었습니다. 이 알고리즘은 활성화 변수를 할당하지 않고도 임시 절(clause)을 추가할 수 있는 방법을 제안하여 SAT solver의 리셋 필요성을 제거했습니다. 결과적으로 GipSAT는 IC3 알고리즘이 필요로 하는 SAT 쿼리를 효율적으로 처리합니다.

- **Performance Highlights**: GipSAT의 종합 평가 결과는 주요 구현체인 Minisat보다 평균 3.61배 빠른 해결 시간을 기록하며 성능 향상을 입증했습니다. 이는 GipSAT가 IC3 알고리즘의 효율성과 확장성을 크게 개선할 수 있음을 보여줍니다. 이 연구는 대규모 모델 검증이 필요한 산업적 설계에 더욱 적합한 솔루션을 제공할 수 있는 가능성을 열어줍니다.



### Faster Configuration Performance Bug Testing with Neural Dual-level Prioritization (https://arxiv.org/abs/2501.15392)
Comments:
          accepted by ICSE 2025

- **What's New**: 본 논문은 복잡하고 구성 가능한 소프트웨어 시스템에서 Configuration Performance Bugs (CPBugs)로 인한 성능 문제에 대해 다룹니다. 특히, 자동화된 oracle 추정기를 통해 CPBug 테스트의 속도를 비약적으로 향상시키는 Neural Dual-level Prioritization(NDP)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 두 개의 neural language model을 활용하여 CPBug 유형을 예측하고, 각 설정 옵션과 값 범위의 우선순위를 정하여 효율적으로 테스트합니다.

- **Technical Details**: CPBug는 종종 개발자의 의도와는 다른 성능 저하를 유발하기 때문에 식별하기 어려운 구조적인 버그입니다. NDP는 두 가지 수준에서 구성 옵션 우선순위를 매기고, numeric 옵션에 대해 검색 깊이를 조정하여 CPBug 테스트를 가속화합니다. RoBERTa라는 neural language model을 사용하여 CPBug 관련성을 예측하는 데 초점을 맞추고, 테스트 중 성능 저하가 5% 이상 발생하면 이를 CPBug로 간주하는 기준을 설정합니다.

- **Performance Highlights**: 실험 결과, NDP는 기존의 최첨단 도구들에 비해 CPBug 유형을 87% 더 정확하게 예측하고, 테스트 옵션의 우선순위를 통해 최대 1.73배의 속도 향상을 달성하였습니다. 또한, numeric 옵션의 검색 깊이를 우선순위에 따라 조정함으로써 최대 88.88배의 CPBug 발견 속도 향상을 이루어냈습니다. 이 모든 데이터와 소스 코드는 공공 저장소에서 접근 가능합니다.




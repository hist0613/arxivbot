### Can Language Models Serve as Text-Based World Simulators? (https://arxiv.org/abs/2406.06485)
Comments:
          ACL 2024

- **What's New**: 큰 언어 모델(LLMs)을 텍스트 기반 세계 시뮬레이터로 활용하려는 연구가 새롭게 등장했습니다. 연구진은 'ByteSized32-State-Prediction'이라는 새로운 벤치마크를 구축해, 텍스트 게임의 상태 전환과 게임 과제 데이터를 포함한 데이터셋을 만들었습니다. 이 벤치마크를 사용해 GPT-4의 성능을 검증한 결과, 현재의 LLMs는 신뢰할 수 있는 세계 시뮬레이터로 사용되기에는 부족함이 있음이 밝혀졌습니다.

- **Technical Details**: 이 연구에서는 LLM들이 텍스트 기반 가상 환경에서 세계 시뮬레이터로서의 역할을 할 수 있는 능력을 평가합니다. 각 텍스트 환경은 목표-조건부 부분 관측 마르코프 결정 프로세스(POMDP)로 표현됩니다. 연구팀은 LLM-Sim이라는 예측 과제를 제안하여, 주어진 맥락, 상태 및 행동에서 후속 상태, 보상 및 게임 완료 상태로 매핑하는 세계 시뮬레이터의 성능을 정량적으로 평가합니다.

- **Performance Highlights**: GPT-4를 사용한 분석 결과, 모델은 복잡한 상태 전환을 정확히 예측하는 데 실패했습니다. 상태 전환 정확도는 59.9%를 넘지 않으며, 이는 산술적 계산, 상식 및 과학적 추론을 필요로 하는 전환에서 특히 취약합니다. 이러한 결과는 LLMs가 아직 신뢰할 수 있는 세계 시뮬레이터로서 사용되기 위해 추가적인 혁신이 필요함을 시사합니다.



### Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies (https://arxiv.org/abs/2406.06461)
- **What's New**: 다양한 논리 전략(proposed reasoning strategies)을 검토한 이 논문은 전통적인 성능 지표가 지나치게 성능(metric) 중심에 치우쳐 있으며, 이에 따른 계산 자원(compute budget)을 간과하고 있음을 지적합니다. 이 논문은 계산 자원까지 포함한 새로운 평가 프레임워크를 제안하여 더 균형 잡힌 비교를 가능하게 합니다.

- **Technical Details**: 논문에서는 LLM(Large Language Models)에서 다양한 논리 전략을 평가하기 위해 계산 자원(budget)을 포함한 세 가지 차원의 평가 프레임워크를 도입합니다: 쿼리(query), 토큰(token), 금전적 비용(monetary cost). 이를 통해 성능뿐만 아니라 계산 자원 소비 측면에서도 평가가 이루어집니다. 또한, 단순한 체인 오브 생각 일관성(COT SC, Chain-of-Thought Self-Consistency) 기준선이 상당한 계산 자원 할당 시 더 복잡한 전략보다 자주 더 우수한 성과를 나타냄을 보여줍니다.

- **Performance Highlights**: 연구 결과, 체인 오브 생각 일관성(COT SC) 전략은 많은 계산 자원을 투입할 때 더 복잡한 논리 전략(예: Multi-Agent Debate, Reflexion)보다 우수한 성과를 내는 경우가 많습니다. 특히, 자기평가(self-evaluation) 성능은 모델과 데이터셋에 따라 크게 달라지며, 성능과 계산 자원 사이의 관계가 명확하게 드러났습니다.



### Evaluating the Retrieval Component in LLM-Based Question Answering Systems (https://arxiv.org/abs/2406.06458)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 기반의 챗봇 시스템에서 리트리버 (retriever) 성능을 평가하는 간단한 기준을 제안합니다. 기존 평가지표들이 대형 언어 모델(LLMs)의 능력을 완전히 포착하지 못하는 상황에서, 새로운 평가 프레임워크는 챗봇의 전체 성능과 더 잘 맞아떨어지는 평가를 제공합니다.

- **Technical Details**: RAG 모델을 활용한 QA 시스템은 두 가지 주요 구성 요소로 나뉘어집니다. 리트리버가 문서 코퍼스에서 관련 정보를 검색하고, 생성기(generator)가 이 문서를 바탕으로 응답을 생성합니다. 기존 평가 방법이 주석된 데이터에만 집중하는 한계를 극복하기 위해, 우리는 이상적인 리트리버와 실제 리트리버의 출력을 비교하여 downsteam 효과까지 고려하는 새로운 평가 방식을 제안합니다. 이를 위해 Exact Match (EM), 토큰 기반 지표(ROUGE, BLEU, METEOR), 임베딩 기반 지표(BERTScore), 그리고 LLM-기반 평가 방법을 사용합니다.

- **Performance Highlights**: 새로운 리트리버 평가 방법론은 기존의 정밀도(Precision), 재현율(Recall), F1 점수보다 LLM 기반 QA 시스템 성능을 더 잘 반영합니다. NQ-open 코퍼스 실험 결과, 새로운 방법론이 리트리버의 유효성을 더 잘 포착하였고, 기존 지표와 높은 상관관계를 나타냈습니다.



### Interpretability of Language Models via Task Spaces (https://arxiv.org/abs/2406.06441)
Comments:
          To be published at ACL 2024 (main)

- **What's New**: 이번 논문에서는 언어 모델(LM)을 해석하는 새로운 방법을 제안합니다. 기존의 언어 모델 해석 방법은 벤치마크 성능을 평가하고 내부 과정을 추론하는 것이었지만, 이번 연구는 언어 모델의 언어 처리의 질에 초점을 맞추어 '언어 과제 공간(linguistic task spaces)'을 구축했습니다. 이를 통해 언어 현상 간의 연결고리를 밝혀내고자 했습니다.

- **Technical Details**: 언어 모델의 언어 개념화를 나타내는 '언어 과제 공간'을 구축하기 위해 '유사성 프로빙(similarity probing)'이라는 방법을 사용했습니다. 이는 특정 언어 과제에 모델을 미세 조정한 후 다른 과제에 대한 영향을 평가하는 방법입니다. 추가로, '기울기 차이를 통한 미세 조정(FTGD, Fine-Tuning via Gradient Differentials)'이라는 방법을 도입하여 언어 현상의 학습 신호를 구별했습니다.

- **Performance Highlights**: 세 가지 규모의 언어 모델에 적용한 결과, 더 큰 모델이 언어 과제를 보다 포괄적으로 일반화하고, 관련된 언어 과제 간의 매개 변수 공유가 증가하여 언어 처리의 분산도가 높아짐을 발견했습니다. 전체적인 일반화 패턴은 훈련 동안 대부분 안정적이며, 이는 LMs에 대한 커리큘럼 전략이 성공하지 못하는 이유를 설명할 수 있습니다.



### Multimodal Contextualized Semantic Parsing from Speech (https://arxiv.org/abs/2406.06438)
Comments:
          10 Pages, 3 figures, ACL 2024 Main

- **What's New**: 신규 연구 과제인 SPICE(Semantic Parsing in Contextual Environments)를 소개합니다. SPICE는 인공지능 에이전트의 컨텍스트 인식을 향상시키기 위해 다중 모달 입력(multimodal inputs)과 이전 컨텍스트를 통합하는 작업입니다. VG-SPICE라는 새로운 데이터셋을 개발하여, 시각적 장면 그래프(visual scene graph) 구성 작업을 말하는 대화에서 데이터 통합을 강조하고, AViD-SP라는 모델을 제시했습니다. VG-SPICE 데이터셋과 AViD-SP 모델은 공개되어 있습니다.

- **Technical Details**: SPICE는 기반 언어를 통한 반복적인 지식 구축 과정을 포착하는 작업입니다. 이 작업의 목표는 새로운 정보로 컨텍스트 상태를 지속적으로 업데이트하는 것입니다. VG-SPICE 데이터셋은 Visual Genome 데이터셋에서 파생되었으며, 시각적 입력과 음성 대화를 통해 시각적 장면 그래프를 구축해야 합니다. AViD-SP 모델은 이 데이터셋에서 작동하도록 개발되었으며, Grouped Multimodal Attention Down Sampler (GMADS)라는 멀티모달 융합 방법을 도입했습니다.

- **Performance Highlights**: SPICE 작업은 기존의 텍스트 기반 의미 해석보다 복잡한 요구 사항을 강조합니다. AViD-SP 모델은 컨텍스트 일관성을 유지하면서 다중 모달 정보를 처리하는 능력을 보여주었습니다. 이에 대한 실험 결과는 SPICE 프레임워크와 일치하는 성능을 입증했습니다.



### Language Models are Alignable Decision-Makers: Dataset and Application to the Medical Triage Domain (https://arxiv.org/abs/2406.06435)
Comments:
          15 pages total (including appendix), NAACL 2024 Industry Track

- **What's New**: 새로운 의료 분류 의사결정을 위한 데이터셋을 소개합니다. 이 데이터셋은 각 시나리오에 대해 공정성, 도덕적 가치 등 다양한 결정자 속성(Decision-Maker Attributes, DMAs)을 포함하고 있습니다. 이 연구는 대형 언어 모델(LLMs)을 활용하여 윤리적 의사결정을 가능하게 하고, 이러한 의사결정을 다양한 DMAs에 맞출 수 있도록 하는 새로운 소프트웨어 프레임워크를 제시합니다.

- **Technical Details**: 62개의 시나리오를 포함한 이 데이터셋은 공정성과 도덕적 가치 같은 6개의 다른 DMAs로 레이블링되었습니다. 데이터셋은 여러 오픈 소스 LLMs (Falcon, Mistral, Llama 2 등)과 다양한 크기의 모델을 대상으로 실험되었습니다. 또한, 새로운 형태의 가중치 자기 일관성(weighted self-consistency)을 도입하여 LLM의 전반적인 성능을 향상시켰습니다.

- **Performance Highlights**: 새로운 속성 종속 정확도 메트릭을 통해 모델 정합성을 정량화하였습니다. 또한, 제로샷 프롬프팅(zero-shot prompting) 접근 방식을 사용하여 다양한 속성에 모델 결정을 맞추는 데 성공했습니다. 실험 결과, 가중치 자기 일관성 모듈을 확장하여 모델 정합성을 더욱 개선할 수 있음을 보여주었습니다.



### Controlling Emotion in Text-to-Speech with Natural Language Prompts (https://arxiv.org/abs/2406.06406)
Comments:
          accepted at Interspeech 2024

- **What's New**: 최근 연구에서는 자연어를 이용한 직관적인 프롬프팅 방식이 생성 모델의 출력을 조정하는 표준 방식으로 자리잡고 있습니다. 본 연구에서는 감정이 풍부한 텍스트에서 파생된 임베딩을 사용하여 프롬프팅하는 시스템을 제안합니다. 이 임베딩들은 변환기 기반 아키텍처 내 여러 지점에서 통합되어 감정 프롬프팅의 일관성을 향상시킵니다.

- **Technical Details**: 감정 음성 및 텍스트 데이터셋을 병합하여 학습하였으며, 감정 프롬프팅을 통해 음성 합성 시스템의 범용성을 강화했습니다. 자연어 프롬프트는 DistilRoBERTa 기반 감정 분류 모델을 통해 임베딩으로 변환되었으며, 이 임베딩은 TTS 훈련 동안 조정되지 않는 선형 레이어를 통해 TTS 용도로 적응시켰습니다. 제안된 시스템은 FastSpeech-2 같은 구조와 다양한 예측 모델(Conformer 인코더 및 디코더, duration, pitch 및 에너지)을 통합하여 스펙트로그램을 생성하며, HiFi-GAN으로 변환된 후 최종 음성을 생성합니다. 훈련 과정은 커리큘럼 학습 방법을 사용하여 LJSpeech와 LibriTTS-R 데이터셋과 함께 감정 음성 데이터셋을 포함합니다.

- **Performance Highlights**: 객관적 및 주관적 평가 결과, 프롬프트에 포함된 감정을 정확하게 음성으로 전달하면서도 화자 정체성과 전반적인 음질 및 인텔리전스 모두 높은 수준을 유지하는 것을 확인했습니다. 사용자들이 감정을 수동으로 선택하지 않고 텍스트만으로 감정을 포함한 음성을 생성할 수 있도록 합니다. 모든 코드와 모델은 오픈 소스 라이센스 하에 제공됩니다.



### Meta Learning Text-to-Speech Synthesis in over 7000 Languages (https://arxiv.org/abs/2406.06403)
Comments:
          accepted at Interspeech 2024

- **What's New**: 이번 연구에서는 전통적인 음성 합성(TTS) 개발에 충분한 데이터가 부족한 7000개 이상의 언어로 말하는 하나의 텍스트-음성 변환 시스템을 구축하는 도전적 과제를 다룹니다. 이 시스템은 방대한 멀티링구얼 사전 학습과 메타 학습을 결합하여 언어 표현을 근사화함으로써 데이터가 전혀 없는 언어에서도 제로샷(Zero-shot) 음성 합성이 가능하도록 합니다. 우리는 다양한 언어적 환경에서 객관적 측정과 인간 평가를 통해 시스템 성능을 검증하였습니다. 코드와 모델을 공개하여 언어 자원이 제한된 커뮤니티에 힘을 실어주고 음성 기술 분야의 혁신을 촉진하기를 목표로 합니다.

- **Technical Details**: 대규모의 멀티링구얼 사전 학습과 메타 학습을 통합하여 언어 표현을 근사화했습니다. 이를 위해 약 18,000시간의 짝지어진 데이터로 462개 언어에 대해 사전 학습을 진행했습니다. 언어 불명 시스템을 제외하고 모든 언어에 대해 언어 임베딩을 예측하고 제로샷 메커니즘을 사용해 데이터를 갖고 있지 않은 언어에서도 음성을 생성할 수 있습니다. 파이프라인은 모듈식이며, 대부분의 구성 요소는 아키텍처에 구애받지 않습니다. Phoneme(boxes) 변환, FastSpeech-2 시스템, HiFi-GAN vocoder 등을 포함한 다양한 기술을 사용하며, 음성 품질을 높이기 위해 여러 필터링 및 전처리 과정을 거쳤습니다.

- **Performance Highlights**: 시스템의 성능은 고자원(high-resource), 중자원(medium-resource), 저자원(low-resource) 언어들을 대상으로 객관적 측정 및 인간 평가를 통해 검증되었습니다. 새로운 손실 함수인 LESS(Langauge Embedding Space Structure) 손실 함수를 도입하여 언어 임베딩 공간이 의미 있는 구조를 가지도록 했습니다. 모델은 8개의 A6000 GPU를 사용하여 4일 동안 훈련했으며, 훈련 커리큘럼을 통해 언어와 화자 임베딩 간의 정보 누출 문제를 해결했습니다. 최종적으로 반응 형식의 언어별 임베딩을 통해 각 언어에 대한 음성 합성 성능을 높였습니다.



### INTERSPEECH 2009 Emotion Challenge Revisited: Benchmarking 15 Years of Progress in Speech Emotion Recognition (https://arxiv.org/abs/2406.06401)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: INTERSPEECH 2009 Emotion Challenge의 15주년을 기념하여, 최근의 주요 발전에 기반한 딥러닝 모델들을 활용해 처음으로 감정 인식 문제를 다시 검토했습니다. 이를 통해 최신 방법이 항상 오래된 방법보다 성능이 뛰어난 것은 아니라는 흥미로운 결론을 도출했습니다.

- **Technical Details**: 우리는 고정된 하이퍼파라미터(hyperparameters)를 사용하여 각 모델을 학습시키고, 초기 설정에서 가장 성과가 좋은 모델을 그리드 서치(grid search)를 통해 미세 조정했습니다. 실험에서는 Multi-layered Perceptrons (MLPs), Long Short-Term Memory (LSTMs), Convolutional Recurrent Neural Networks (CRNNs), Convolutional Neural Networks (CNNs), Transformer 모델들을 활용했습니다.

- **Performance Highlights**: 대부분의 모델들이 공식 베이스라인(baseline)과 동일하거나 약간 초과했으며, 하이퍼파라미터 튜닝 후에야 원래 챌린지 우승자들을 소폭 능가하는 결과를 보였습니다. 이는 FAU-AIBO 데이터셋이 여전히 매우 도전적이라는 것을 보여줍니다.



### Should We Fine-Tune or RAG? Evaluating Different Techniques to Adapt LLMs for Dialogu (https://arxiv.org/abs/2406.06399)
- **What's New**: 새로운 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 인간-기계 대화에서 응답 생성 역할을 다양한 대화 유형(Open-Domain, Knowledge-Grounded, Task-Oriented, Question Answering)에 대해 분석했습니다. Llama-2 Chat (Llama2C)와 Mistral Instruct (MistralI)을 기반으로 'In-Context Learning'과 'Fine-Tuning' 기법을 실험했으며, 외부 지식을 통합하는 RAG(Retrieval-Augmented Generation) 및 gold knowledge 시나리오의 영향을 평가했습니다.

- **Technical Details**: 이번 연구에서는 다양한 데이터셋을 선정해 네 가지 대화 유형(Open-Domain Dialogues, Knowledge-Grounded Dialogues, Task-Oriented Dialogues, Question Answering)에 적용했습니다. 각 기법(in-context learning과 fine-tuning)과 대화 유형에 대해 동일한 자동 평가 기준과 인간 평가 프로토콜을 사용해 성능을 평가했습니다. 또한, 설명 가능성을 높이기 위해 통합 기울기(integrated gradients) 방법을 사용해 입력 벡터 각 세그먼트의 기여도를 계산했습니다.

- **Performance Highlights**: 연구 결과, LLM을 다양한 대화 유형에 맞게 적응시키는 데 있어 보편적으로 가장 좋은 기법은 없다는 점을 발견했습니다. 각 기법의 효능은 기본 LLM과 특정 대화 유형에 따라 다르며, 최상의 적응 기법을 평가할 때는 자동 평가 메트릭에서 비롯된 오해를 피하기 위해 반드시 인간 평가를 포함해야 한다는 점을 강조했습니다.



### mHuBERT-147: A Compact Multilingual HuBERT Mod (https://arxiv.org/abs/2406.06371)
Comments:
          Extended version of the Interspeech 2024 paper of same name

- **What's New**: mHuBERT-147은 90K 시간의 깨끗한 오픈 라이선스 데이터를 사용하여 훈련된 최초의 범용 대규모 다국어 HuBERT 음성 표현 모델입니다. 새로운 다국어 배칭 업샘플링 전략과 faiss 기반 클러스터링을 사용하여 5.2배 더 빠르게 레이블 할당을 달성했습니다.

- **Technical Details**: mHuBERT-147은 다국어 음성 데이터를 효율적으로 처리하기 위해 두 가지 새로운 전략을 도입했습니다. 첫째로, faiss 기반의 클러스터링 방법을 채택하여 레이블 할당 속도를 5.2배 향상시켰습니다. 둘째로, 언어와 데이터셋의 다양성을 모두 고려한 다국어 배칭 업샘플링 전략을 적용했습니다. 이 모델은 95M 파라미터로 구성되며, 3번의 훈련 반복을 거쳤습니다.

- **Performance Highlights**: mHuBERT-147은 ML-SUPERB 벤치마크에서 10분 및 1시간 리더보드에서 각각 2위와 1위를 기록하며 SOTA(LID) 점수를 달성했습니다. 또한 300M 파라미터와 436K 시간이 소요된 XLS-R을 비롯한 더 큰 모델들을 일관되게 능가했습니다. 다국어 음성 처리 작업에서 mHuBERT-147은 탁월한 성능과 파라미터 효율성을 갖춘 유망한 모델임을 보여줍니다.



### Annotation alignment: Comparing LLM and human annotations of conversational safety (https://arxiv.org/abs/2406.06369)
Comments:
          Working draft, short paper. 5 pages, 1 figure

- **What's New**: 새로운 연구에서는 Language Model (LLM)이 인간의 안전 인식과 얼마나 잘 일치하는지를 조사합니다. 특히 GPT-4가 다양한 인종 및 성별 그룹의 평가와 얼마나 잘 맞는지 분석합니다.

- **Technical Details**: 이 연구는 350개의 사용자-챗봇 대화를 포함한 DICES 데이터셋을 사용하며, 각각의 대화는 112명의 다양한 인종 및 성별 그룹 애노테이터가 안전성을 평가했습니다. 연구에서는 GPT-3.5, GPT-4, GPT-4o를 사용하여 대화를 다시 분석하고, 무작위로 선택된 프롬프트를 통해 각 대화의 안전도를 평가했습니다.

- **Performance Highlights**: GPT-4는 평균 애노테이터 평가와 Pearson 상관계수 r = 0.59를 달성하여 중간 애노테이터의 상관계수 r = 0.51보다 높습니다. 특히 GPT-4의 chain-of-thought 스타일 프롬프트가 가장 높은 상관계수 r = 0.61을 기록했습니다. 하지만, 다른 그룹과의 일관성 문제, 그리고 특정 인구 그룹의 평가를 예측하는 데는 한계가 있었습니다.



### Symmetric Dot-Product Attention for Efficient Training of BERT Language Models (https://arxiv.org/abs/2406.06366)
Comments:
          to be published in Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: 트랜스포머(Transformer) 아키텍처의 주목할 만한 새 연구는 기존의 scaled dot-product attention 메커니즘을 대체하는 새로운 호환성 함수(compatibility function)를 제안합니다. 이 대체 호환성 함수는 전통적인 주의 메커니즘(attention mechanism)의 학습된 표현 사이의 겹침을 활용합니다. 이를 통해 대칭(self-attention)의 대칭성과 pairwise coefficient dot-product를 도입하였으며, 이는 BERT와 같은 모델의 전이 학습(pre-training) 과정에서 성능 개선을 입증했습니다.

- **Technical Details**: 연구에서 제안된 새로운 주의 메커니즘에는 두 개의 변형이 포함됩니다: 대칭 dot-product와 대칭 pairwise coefficient dot-product. 이는 기존의 트랜스포머 모델이 가지고 있는 두 개의 선형 연산자(queries Q와 keys K)의 겹치는 특성을 이용하려는 시도입니다. 이러한 새로운 접근 방식은 모델의 학습 파라미터 수를 약 6% 감소시키고, 기존 모델보다 학습 수렴(convergence)에 필요한 스텝 수를 50%로 줄였습니다.

- **Performance Highlights**: 새로운 대칭 dot-product attention 메커니즘을 적용한 결과, GLUE benchmark에서 79.36 점을 얻어 전통적인 메커니즘의 78.74 점을 넘어섰습니다. 이는 파라미터 수를 6% 줄이고, 학습 스텝 수를 절반으로 줄이는 동시에 정확도를 유지하는데 성공함을 의미합니다.



### MASSW: A New Dataset and Benchmark Tasks for AI-Assisted Scientific Workflows (https://arxiv.org/abs/2406.06357)
Comments:
          arXiv admin note: text overlap with arXiv:1706.03762 by other authors

- **What's New**: MASSW는 과학 연구에서 중요한 단계들을 구조적으로 요약하는 대규모 텍스트 데이터셋입니다. 컴퓨터 과학의 17개 주요 학회에서 지난 50년간의 152,000편 이상의 논문을 포함하고 있습니다. 이 데이터셋은 LLMs (Large Language Models)을 사용하여 각 논문에서 '컨텍스트', '핵심 아이디어', '방법론', '결과', '예상 영향'의 다섯 가지 핵심 측면을 자동으로 추출합니다.

- **Technical Details**: MASSW는 연구 논문의 다양한 측면을 구조적으로 정리하여, 과학 워크플로우의 중요한 단계들을 더 잘 탐색하고 분석할 수 있게 해줍니다. LLMs를 활용해 논문의 다섯 가지 핵심 측면을 일관되게 추출하며, MASSW는 이런 측면들을 구조적으로 요약합니다. Open Academic Graph (OAG)를 통해 논문을 접근하며, 1969년부터 2024년까지의 논문을 포함합니다.

- **Performance Highlights**: MASSW 데이터셋의 품질은 인간의 주석과 비교하여 검증되었으며, 다양한 머신러닝 작업을 지원합니다. 예를 들어, 아이디어 생성 및 결과 예측 등에서 벤치마크로 사용할 수 있습니다. 이것은 과학적 워크플로우를 최적화하고 과학 혁신을 촉진하기 위한 새로운 AI 방법을 개발하고 평가하는 데 유용한 데이터셋입니다.



### Sustained Vowels for Pre- vs Post-Treatment COPD Classification (https://arxiv.org/abs/2406.06355)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 본 논문은 만성폐쇄성폐질환 (COPD) 환자의 상태 변화를 지속된 모음을 통해 구별할 수 있는지에 대해 연구했습니다. 기존의 읽는 말 (read speech)을 활용한 분석에서 추가적으로 지속된 모음을 포함함으로써 성과를 개선할 수 있음을 제시합니다.

- **Technical Details**: 데이터셋은 총 50명의 COPD 환자(남성: 26명, 여성: 24명)로 구성되었으며, 이들이 독일어로 다섯 가지 모음 (/a:/, /e:/, /i:/, /o:/, /u:/)을 발음하고 Aesop의 '북풍과 태양' 이야기를 읽는 것에서 시작합니다. 환자들은 치료 전후 두 번 녹음되었으며, 각 녹음은 PRAAT 프로그램을 사용해 수동으로 분할되었습니다.

- **Performance Highlights**: 지속된 모음을 포함함으로써 성과가 79%의 비가중 평균 회상 (unweighted average recall)으로 개선되었으며, 이는 기존 읽는 말 기반 분석 대비 71%에서 향상된 결과입니다.



### MedExQA: Medical Question Answering Benchmark with Multiple Explanations (https://arxiv.org/abs/2406.06331)
- **What's New**: 이번 논문은 여러 의학 분야에서 대형 언어 모델(LLMs)의 의학적 지식 이해 및 설명 능력을 평가하기 위한 새로운 벤치마크인 MedExQA를 소개합니다. 이 벤치마크는 다섯 개의 독특한 의학 전문 분야를 다루며 각 질문-답변 쌍에 대해 다수의 설명을 포함하여, 기존의 의학 QA 벤치마크에서 설명 가능성의 결여를 보완하고 있습니다. 또한, MedExQA는 Llama2 모델 기반의 의료 LLM을 다양화하기 위해 MedPhi-2라는 새로운 의학 모델을 제안했습니다.

- **Technical Details**: MedExQA는 생의학 공학, 임상 실험 과학, 임상 심리학, 작업 치료 및 언어 치료 등을 포함한 다섯 가지 전문 분야로 구성되어 있으며, 각 질문에 대해 두 가지 설명을 제공하여 모델의 설명 능력을 평가합니다. MedPhi-2는 Phi-2 (2.7B) 모델을 기반으로 하여 의학 텍스트로 학습되었습니다. 다양한 크기의 18개 오픈 소스 모델, 3개의 OpenAI GPT 모델, 그리고 MedPhi-2 모델을 평가한 결과, MedPhi-2는 Llama2-70B 기반의 LLM보다 뛰어난 설명을 생성했습니다.

- **Performance Highlights**: MedPhi-2 모델은 의료 LLMs 평가에서 설명 생성 성능 측면에서 Llama2-70B 기반 모델을 능가하였으며, 이는 자원이 제한된 의료 도메인에서의 효과성을 입증했습니다. 또한, 여러 설명을 사용하는 평가 방식은 인간의 평가와 더 잘 일치하는 것으로 나타났습니다. MedExQA는 LLM이 의학적 설명을 생성할 때의 이해도를 더 잘 평가할 수 있는 새로운 기준을 제시합니다.



### A Parameter-efficient Language Extension Framework for Multilingual ASR (https://arxiv.org/abs/2406.06329)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 이 연구는 기존의 다국어 음성 인식 모델(MASR)을 기반으로 새로운 언어를 확장하기 위한 아키텍처 기반 프레임워크 PELE(Parameter Efficient Language Extension)를 제안합니다. 이 접근법은 평균어붕괴(catastrophic forgetting)를 근본적으로 해결하고, 새로운 언어에 적응하기 위한 애드온 모듈(add-on module)을 도입하여 매개변수 효율성을 극대화합니다.

- **Technical Details**: PELE는 MASR 지속 학습 문제를 언어 정체성 예측(LP)과 교차 언어 적응(XLA)이라는 두 가지 하위 문제로 확률적으로 분해하여 해결합니다. 주요한 이론적 기반을 참고하여, 파라미터 효율적 미세 조정(PEFT) 모듈의 다양성과 그 변종이 XLA를 수행하기 위한 잠재적 후보로 탐구되었습니다. 기존의 MASR 모델의 파라미터를 고정시키고, 새로운 언어를 지원하기 위해 가벼운 모듈(adapter)을 도입합니다.

- **Performance Highlights**: 새로운 5개의 언어에 대한 실험 결과, 모든 언어에서 만족스러운 성능을 달성했으며, 특히 5개 중 3개의 언어에서 지속적인 공동 학습 설정보다 우수한 성능을 보였습니다. 또한 약 10M의 매개변수만으로 추가된 언어의 지원을 실현하며, 기존의 접근법 대비 매우 제한된 매개변수 추가로 더 나은 성능을 입증했습니다.



### Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching (https://arxiv.org/abs/2406.06326)
Comments:
          29 pages

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 최신화를 위한 새로운 학습 프레임워크인 Self-Tuning을 소개합니다. 이는 모델이 새로운 정보를 효율적으로 습득하도록 돕기 위한 Self-Teaching 전략을 채택하고 있습니다. 또한, 세 가지 Wiki-Newpages-2023-QA 데이터셋을 도입하여 LLM의 지식 습득 능력을 심층 분석합니다.

- **Technical Details**: Self-Tuning은 세 단계로 구성됩니다: (i) 새로운 문서와 관련된 QA 데이터를 사용하여 모델을 학습, (ii) 새로운 문서를 통해 자가 학습 전략을 활용하여 지식 습득 및 QA 기술 검토, (iii) 마지막으로 새로운 문서만을 사용하여 지속적인 학습을 통해 지식 습득을 보장합니다. Self-Teaching 전략은 문서 자체를 단순한 텍스트로 제공하고, 자가 지도 방식으로 지식 집약적인 과제를 생성합니다.

- **Performance Highlights**: Llama2 모델을 사용한 실험 결과, Self-Tuning은 지식 암기 및 추출 작업에서 탁월한 성능을 보여주었습니다. 또한, 이유를 도출하는 작업(추론 작업)에서 높은 정확도를 꾸준히 유지하며, 이전에 획득한 지식을 상당히 잘 유지하는 뛰어난 성능을 나타냈습니다. 본 연구는 LLM의 지식 습득 능력을 분석하기 위해 도입된 Wiki-Newpages-2023-QA 데이터셋과 함께 사용되었습니다.



### Tx-LLM: A Large Language Model for Therapeutics (https://arxiv.org/abs/2406.06316)
- **What's New**: Tx-LLM (Large Language Model)를 소개합니다. 이 모델은 PaLM-2 베이스 LLM을 기반으로 다양한 치료법을 배우도록 파인 튜닝되었습니다. 총 709개의 데이터셋과 66개의 작업을 학습해, 광범위한 화학 및 생물학적 엔티티와 자유 텍스트를 동시에 다룰 수 있습니다. 이로써, 단일 가중치 세트를 사용해 다양한 속성을 예측할 수 있게 됩니다.

- **Technical Details**: Tx-LLM은 PaLM-2 모델을 베이스로 다양한 분류(classification), 회귀(regression), 생성(generation) 작업을 학습했습니다. 데이터셋에는 소분자(small molecules), 단백질(proteins), 핵산(nucleic acids), 세포주(cell lines), 질병(diseases) 등의 정보가 포함되어 있으며, 총 709개의 데이터셋과 66개의 작업으로 구성되어 있습니다. 모델 크기, 도메인 파인 튜닝, 프롬팅 전략 등이 성능에 미치는 영향도 분석되었습니다.

- **Performance Highlights**: Tx-LLM은 66개 작업 중 43개에서 SOTA(state-of-the-art) 성능과 유사하거나 더 나은 성능을 보였습니다. 특히 분자 SMILES 표현과 텍스트(예: 세포주 이름, 질병 이름)를 결합한 작업에서 평균적으로 SOTA를 초과하는 성능을 달성했습니다. 또한, 서로 다른 약물 유형(예: 소분자 및 단백질)을 포함하는 데이터셋 간의 긍정적 전이(positive transfer)를 관찰할 수 있었습니다.



### Multi-Prompting Decoder Helps Better Language Understanding (https://arxiv.org/abs/2406.06279)
- **What's New**: 본 논문은 Model-as-a-Service (MaaS) 환경에서 프롬프트를 여러 개 사용하여 프리트레인된 언어 모델(PLMs)을 다운스트림 작업에 적용하는 Multi-Prompting Decoder (MPD) 프레임워크를 제안합니다. 이는 기존의 단일 프롬프트 의존성을 줄이고, 데이터 부족 문제를 완화하며, 다양한 시각에서 PLMs의 지식을 추출하는 데 도움을 줍니다.

- **Technical Details**: MPD 프레임워크는 다중 프롬프트를 사용하여 샘플당 여러 숨은 상태와 클래스 점수를 얻고 이를 통해 디코딩합니다. 두 가지 디코딩 전략을 제안합니다: 숨은 상태를 위한 옵티멀 트랜스포트(multi-prompting decoding with optimal transport)와 클래스 점수를 위한 보정된 디코딩(calibrated decoding).

- **Performance Highlights**: 다양한 자연어 이해 데이터셋에서 광범위한 실험 결과, MPD 방법이 few-shot 설정에서 새로운 state-of-the-art 결과를 달성했습니다. 이는 감정 분석, 주제 분류, 자연어 추론 등 다양한 작업에서 검증되었으며, 경량화되고 효율적이며 다양한 PLM 아키텍처에 적용 가능함이 입증되었습니다.



### MaskLID: Code-Switching Language Identification through Iterative Masking (https://arxiv.org/abs/2406.06263)
Comments:
          ACL 2024

- **What's New**: MaskLID는 코드스위칭(Code-Switching, CS) 언어 식별(LID)을 위한 새로운 접근 방식입니다. 이 방법은 별도의 학습 모델 없이 기존의 고성능 문장 수준 LID와 협력하여 작동합니다. 패스트텍스트(FastText) 아키텍처 기반의 GlotLID와 OpenLID 두 가지 오픈 소스 LID와 함께 사용됩니다. MaskLID는 L1 언어와 관련된 텍스트 특징을 마스킹하여 이후에 L2 언어로 텍스트를 분류하는 전략을 사용합니다.

- **Technical Details**: MaskLID의 핵심 전략은 L1 언어와 연관된 특징을 마스킹하는 것입니다. 이렇게 하면 첫 번째 라운드에서 지배적인 언어(L1)로 분류된 텍스트가 두 번째 라운드에서는 L2 언어로 제대로 분류될 수 있습니다. 이는 monolingual 텍스트(단일 언어 텍스트)로 학습된 문장 수준 LID가 주로 사용되며, 이전 방식과 달리 외부 자원이 필요 없습니다. 또한, MaskLID는 다양한 언어 조합을 인식할 수 있으며 두 개 이상의 언어가 섞인 경우도 감지할 수 있습니다.

- **Performance Highlights**: MaskLID는 두 가지 테스트 데이터셋에서 평가되었고, CS 데이터와 monolingual 데이터를 모두 포함합니다. 결과는 MaskLID를 사용함으로써 LID 모델의 성능이 향상됨을 보여줍니다. 특히, 패스트텍스트 기반으로 매우 빠르게 대용량 웹 코퍼스를 분석하여 실제 CS 샘플을 찾아낼 수 있습니다. 이러한 샘플은 CS 입력을 처리하는 애플리케이션의 훈련 데이터로 유용하게 사용될 수 있습니다.



### Combining Embeddings and Domain Knowledge for Job Posting Duplicate Detection (https://arxiv.org/abs/2406.06257)
Comments:
          To be published at 9th International Symposium on Language & Knowledge Engineering LKE 2024

- **What's New**: 이 논문은 여러 플랫폼에 게시된 구인 공고에서 중복 되는 공고를 탐지하는 새로운 접근 방식을 제안합니다. 문자 유사성 및 텍스트 임베딩(text embedding), 키워드 매칭(keyword matching) 방법을 결합하여 성능을 향상시켰습니다. 이 접근 방식은 실제로 사용되고 있으며 긍정적인 피드백을 받고 있습니다.

- **Technical Details**: 중복 탐지에는 문자열 비교, 딥 텍스트 임베딩, 특정 기술에 대한 가중치 목록을 사용한 조합 방식이 사용되었습니다. 각 방법을 개별적으로 사용할 때 성능이 만족스럽지 않지만, 이들을 결합하면 높은 성능과 낮은 오탐률(false positives)을 보입니다.

- **Performance Highlights**: 실제 사용 사례에서 새로운 접근 방식이 높은 성능을 보였으며, 수작업으로 진행하던 중복 탐지 작업을 자동화하는 데 성공적이었습니다. 이로 인해 개발 및 운영 비용도 절감되었습니다.



### LINGOLY: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles in Low-Resource and Extinct Languages (https://arxiv.org/abs/2406.06196)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM; Large Language Models)의 고급 추론 능력을 평가하기 위한 새로운 벤치마크인 LingOly를 소개합니다. 이 벤치마크는 어려운 언어 올림피아드 퍼즐을 사용하여 매우 저자원 언어나 사멸된 언어의 언어 패턴을 인식하고 일반화하는 능력과 복잡한 작업 지침을 따르는 능력을 평가합니다.

- **Technical Details**: LingOly 벤치마크는 90개 이상의 주로 저자원 언어를 다루며, 데이터 오염 문제를 최소화합니다. 총 1,133개의 문제가 6가지 형식과 5가지 난이도 수준으로 구성되어 있습니다. 성능은 직접 정확도와 no-context 베이스라인과의 비교를 통해 평가됩니다.

- **Performance Highlights**: 최신 LLM 11개를 평가한 결과, 높은 난이도의 문제에서 모델들의 성능이 저조했습니다. 최고 모델조차도 가장 어려운 문제에서 35.3%의 정확도만을 기록했으며, 이는 no-context 베이스라인에 비해 21.7% 개선된 수치입니다. 폐쇄형 모델은 일반적으로 공개 모델보다 우수한 성능을 보였으며, 언어의 자원이 많을수록 성적이 더 좋았습니다. 이는 현재의 언어 모델이 다단계의 도메인 외 추론에 어려움을 겪고 있음을 시사합니다.



### Language Models Resist Alignmen (https://arxiv.org/abs/2406.06144)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 정렬(fine-tuning)이 얼마나 깊이 있게 모델에 영향을 미치는지에 대한 의문을 제기합니다. 특히, 정렬 과정이 저변에 있는 모델의 행동 분포에 어떻게 영향을 미치는지 탐구하였습니다.

- **Technical Details**: 이 연구는 압축 이론(compression theory)을 활용해, post-alignment 모델이 사전 학습(pre-training) 동안 형성된 행동 분포로 되돌아가는 경향인 '탄성(elasticity)'을 실험적으로 증명했습니다. 모델의 압축률 변화가 데이터셋 크기에 비례하게 감소한다는 이론적 증명을 제공했습니다. 실험적으로는 다양한 크기의 모델에서 이러한 탄성을 확인하였으며, 사전 정렬된 모델이 재정렬 시 성능 저하 후 원래 분포로 되돌아가는 경향을 보였습니다.

- **Performance Highlights**: 모델의 크기가 커질수록 탄성의 정도가 더욱 증가하며, 사전 학습 데이터가 확장될수록 탄성이 더욱 강화되는 것을 발견했습니다. 이는 LLMs의 고유 탄성을 제어하는 것이 정렬 후의 효과적인 유지 관리에 중요함을 시사합니다.



### Can I understand what I create? Self-Knowledge Evaluation of Large Language Models (https://arxiv.org/abs/2406.06140)
- **What's New**: 새로운 연구는 대형 언어 모델(Large Language Models, LLMs) 평가를 위해 '자기-지식 평가(self-knowledge evaluation)' 프레임워크를 도입했습니다. 이 방법은 모델이 스스로 생성한 질문에 대해 얼마나 잘 이해하고 답변할 수 있는지를 평가하여 기존 평가 방법의 한계를 보완하고자 합니다. 특히, 이러한 접근 방식을 통해 모델의 수학 성능을 향상시킬 가능성이 있음이 제안되었습니다.

- **Technical Details**: 논문에서는 '자기-질문 및 답변(self-questioning and answering)'이라는 직관을 적용한 'First generate, then evaluate' 메소드가 소개되었습니다. 먼저 모델이 질문을 생성하고 이에 대한 답변을 생성하는 과정을 거친 다음, 생성된 내용과 답변을 이용해 다시 모델 스스로가 검증하는 두 단계로 이루어집니다. 이를 통해 모델의 자기-지식 점수를 계산합니다.

- **Performance Highlights**: 연구 결과, 현재의 LLMs와 대형 멀티모달 모델(Large Multi-Modal Models, LMMs)은 자기-지식 평가에서 완벽과는 거리가 멀다는 것이 드러났습니다. 특히, GPT-4와 Gemma 모델만이 주어진 맥락에서 100%의 정확도를 달성했으며, 노이즈가 추가된 맥락에서는 정확도가 떨어졌습니다. 또한, 자기-생성 수학 과제 데이터를 통해 모델을 미세 조정(fine-tuning)하면 GSM-8k 성능이 향상될 수 있음을 발견했습니다. 마지막으로, 전문가 기반 프롬트(Prompt)는 종종 자기-지식 능력을 향상시키지만 '사고의 연쇄(chain-of-thought)' 프롬트는 그렇지 않다는 것도 밝혀졌습니다.



### Building Bridges: A Dataset for Evaluating Gender-Fair Machine Translation into German (https://arxiv.org/abs/2406.06131)
Comments:
          Accepted to Findings of ACL 2024. Code and data at this https URL

- **What's New**: 이번 연구에서는 영어에서 독일어로 번역할 때 성별 공정 언어(Gender-fair Language, GFL)를 어떻게 구현할 수 있는지를 분석했습니다. 영어의 성별 중립 표현을 독일어로 번역할 때 흔히 나타나는 남성 일반화 문제를 해결하려는 첫 시도입니다.

- **Technical Details**: 연구팀은 독일어 성별 공정 언어 사전을 만들고, 백과사전 텍스트와 의회 연설에서 다중 문장 테스트 사례를 샘플링했습니다. 이를 통해 두 개의 상업용 시스템과 여섯 개의 신경망 기계 번역 모델을 조사했습니다. 특히, 번역 시스템들이 주로 남성 형태를 사용하고, 성별 중립 형태의 번역은 매우 드물다는 것을 발견했습니다. 신경망 모델과 GPT 모델을 포함한 주요 시스템을 평가한 결과, 문맥이 추가되더라도 성별 공정 번역의 비율이 유의미하게 높아지지 않았습니다.

- **Performance Highlights**: 대부분의 번역 시스템이 0-2%의 낮은 비율로 성별 공정 언어를 사용했습니다. 이는 현대 기계 번역 시스템이 여전히 남성 중심의 편향을 가지고 있다는 증거입니다. 연구팀은 이를 개선하기 위한 추가 연구와 개발이 필요함을 강조했습니다.



### Comparing Data Augmentation Methods for End-to-End Task-Oriented Dialog Systems (https://arxiv.org/abs/2406.06127)
Comments:
          There are 25 pages in total, 23 tables, 18 figures. Accepted in ACL 2024

- **What's New**: 이번 연구는 데이터 확장(data augmentation, DA) 방법의 효과를 종단형(end-to-end) 작업 지향 대화 시스템(task-oriented dialog systems, ToDS) 설정에서 검토합니다. 두 개의 ToDS 시스템인 UBAR와 GALAXY를 사용하여 MultiWOZ와 KVRET 데이터셋에서 DA 방법을 실험적으로 평가했습니다. 또한, 도메인을 교차하는 소수 샘플(few-shot cross-domain) 설정을 소개하여 같은 결론에 도달했습니다.

- **Technical Details**: 본 연구는 세 가지 종류의 DA 방법(word-level, sentence-level, dialog-level)을 비교했습니다. Word-level DA 방법은 원본 학습 예제의 단어를 유사한 단어로 대체합니다. Sentence-level DA는 문장 단위로 번역 및 다시 번역하는 방법(back-translation), 의존 트리(dependency tree)의 일부분을 바꾸는 방법 등이 포함됩니다. Dialog-level DA는 ToDS 데이터셋의 대화 상태를 이용합니다. 이를 통해 각 구성 모듈의 필요성을 줄이고, 종합적인 단계별 처리를 하나의 시스템으로 해결하려고 합니다.

- **Performance Highlights**: DA 방법을 사용하면 성능이 향상되며, 특히 사전 학습된 모델을 사용할 때 큰 성과를 얻을 수 있음을 보여줍니다. 실험 결과, 모든 DA 방법이 이점이 있으며, 가장 효과적인 방법을 강조하여 실무자들에게 조언을 제공합니다. 또한 새로운 소수 샘플 교차 도메인 평가 설정에서 DA 방법이 성능을 향상시키는 것을 확인했습니다.



### Verifiable Generation with Subsentence-Level Fine-Grained Citations (https://arxiv.org/abs/2406.06125)
Comments:
          NAACL 2024 Findings

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 생성된 콘텐츠가 인용된 자료에 의해 뒷받침되는지를 더 세부적으로 검증할 수 있는 'subsentence-level' 인용 방식을 도입하여, 모델의 투명성과 신뢰성을 향상시키는 방법을 제안합니다. 이를 위해 SCiFi라는 10,000개의 위키피디아 단락과 이에 대한 세부 인용 정보를 포함하는 새로운 데이터셋을 소개합니다.

- **Technical Details**: SCiFi 데이터셋은 각 단락에 대해 후보 출처 문서 집합과 단락 생성의 기초가 되는 쿼리를 포함합니다. 이 연구는 최신 LLMs (OpenAI GPT, Llama2, Vicuna, Mistral)을 주요 대상으로 하여 모델이 인용을 포함하여 콘텐츠를 생성하는 능력을 평가합니다. 세 가지 문서 읽기 전략을 사용하여 긴 출처 문서를 처리하고, 훈련 샘플을 이용하여 오픈소스 LLMs의 효과를 검증합니다.

- **Performance Highlights**: 실험 결과, LLMs가 인용 품질을 향상시키기 위해서는 완전한 출처 문서 컨텍스트가 중요하다는 점이 밝혀졌습니다. 또한, 모델 크기가 커질수록 답변 품질은 향상되지만 인용 품질은 향상되지 않았습니다. 세부 인용 생성을 위해서는 감독된 방식의 미세 조정(supervised fine-tuning)이 필요합니다.



### Enhancing Long-Term Memory using Hierarchical Aggregate Tree for Retrieval Augmented Generation (https://arxiv.org/abs/2406.06124)
Comments:
          6 pages, 2 figures

- **What's New**: 이번 연구에서는 긴 대화에서의 추론을 개선하기 위해 좌표기반의 조건부 트리 탐색을 통해 대화 맥락을 재귀적으로 집계하는 계층적 집합 트리(Hierarchical Aggregate Tree; HAT) 메모리 구조를 제안하였습니다. HAT는 자녀 노드의 정보를 캡슐화하여 광범위한 커버리지와 깊이 조절을 가능하게 합니다. 최적의 컨텍스트를 찾기 위해 HAT를 통한 최적의 트리 탐색을 공식화했으며, 실험 결과 HAT가 대화의 일관성과 요약 품질을 향상시키는 것을 입증했습니다. 이 메모리 증강은 LLM의 더 일관된 길게 이어지는 대화를 가능하게 합니다.

- **Technical Details**: 연구에서는 LLM의 한계인 컨텍스트 용량을 개선하기 위해 HAT라는 새로운 데이터 구조를 도입했습니다. HAT는 다음과 같은 주요 특징을 갖습니다: 루트에서 리프 노드로 내려갈수록 해상도가 높아지고, 왼쪽에서 오른쪽으로 이동할수록 최신 정보가 포함됩니다. 연구의 목표는 사용자 쿼리에 따라 HAT 내에서 최적의 탐색 경로를 찾아 적절한 컨텍스트를 제공하는 메모리 에이전트(memory agent)를 개발하는 것입니다. HAT는 계층별로 노드를 구분하며, 각 계층은 노드 집합으로 구성되어 있습니다. 실험에서는 다중 대화 세션 내 정보 관리 및 갱신의 중요성을 강조하며, 이를 위해 HAT 구조가 효과적으로 작동하는 것을 확인했습니다.

- **Performance Highlights**: 실험에서는 HAT가 기존 베이스라인 대비 대화의 일관성(coherence)과 요약 품질(summary quality)을 눈에 띄게 향상시킴을 확인했습니다. 특히, HAT는 파라미터의 급격한 증가 없이 멀티턴 대화에서 일관된 추론과 요약을 가능하게 하여 LLM이 보다 일관되고 근거가 확실한 장문의 대화를 생성할 수 있게 합니다.



### Recurrent Context Compression: Efficiently Expanding the Context Window of LLM (https://arxiv.org/abs/2406.06110)
- **What's New**: Transformer 기반의 대형 언어 모델 (LLMs)의 문맥 길이 연장과 이해 능력을 개선하기 위한 새로운 방법인 순환 문맥 압축(Recurrent Context Compression, RCC)을 소개합니다. 기존 연구들이 문맥 압축 기술에 주로 집중해온 것과는 달리, 본 연구는 문맥과 명령어가 동시에 압축될 때 발생하는 모델의 응답 품질 저하 문제를 해결하기 위한 명령어 재구성 방법을 제안합니다.

- **Technical Details**: RCC는 오토인코더(autoencoder)를 기반으로 한 문맥 압축 모델 구조입니다. 긴 텍스트 압축의 효율성을 높이기 위해 순환 압축 메커니즘을 도입하였습니다. 먼저, 짧은 시퀀스에서 전체 매개변수를 훈련한 다음, 인코더(encoder)의 가중치를 동결하고 긴 시퀀스를 계속 훈련하는 방법으로 긴 문맥의 훈련 컨텍스트 길이를 연장합니다. 또한, 문맥과 명령어가 동시에 압축될 때 발생하는 문제를 해결하기 위해 압축된 벡터에서 명령어 내용을 재구성하는 방식을 사용합니다.

- **Performance Highlights**: 텍스트 재구성 작업에서 최대 32배의 압축률로 BLEU4 점수 약 0.95를 기록하였으며, 1M의 시퀀스 길이로 패스키(passkey) 검색 작업에서 거의 100%의 정확도를 달성하였습니다. 또한, 긴 텍스트 문답 작업에서 비압축 방법과 비교해 경쟁력 있는 성능을 보였으며, 긴 텍스트 추론 작업에서 저장 리소스를 크게 절약하는 것을 입증하였습니다. 코드, 모델 및 데모는 공개된 URL에서 이용 가능합니다.



### Efficient k-Nearest-Neighbor Machine Translation with Dynamic Retrieva (https://arxiv.org/abs/2406.06073)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 이번 연구는 $k$-Nearest-Neighbor Machine Translation ($k$NN-MT)의 성능을 개선하고자 두 가지 주요 개선안을 제안합니다. 기존의 $k$NN-MT는 번역 도메인 지식을 외부 데이터스토어에 저장하고 이를 통해 모델의 예측 분포를 보정합니다. 그러나 매 시점마다 $k$NN 검색을 수행하는 데 걸리는 시간이 크다는 문제가 있습니다. 이를 해결하기 위해 제안된 $k$NN-MT-AR의 한계를 분석하고, 이를 보완한 $k$NN-MT-DR을 제안했습니다.

- **Technical Details**: 새로운 $k$NN-MT-DR 모델에서는 두 가지 주요 기술을 도입했습니다. 첫째, MLP 기반의 분류기를 도입하여 매 시점마다 $k$NN 검색을 스킵할지 여부를 이진 분류 작업으로 결정합니다. 이 때, 분류기의 성능을 최대화하기 위해 여러 스칼라 특징들을 사용합니다. 둘째, 시점에 따른 동적 임계값 조정 방법을 제안하여, 각 시점에서 $k$NN 검색의 요구사항을 효과적으로 반영할 수 있도록 했습니다.

- **Performance Highlights**: 다양한 멀티 도메인 데이터셋에 대한 실험 결과, 제안된 $k$NN-MT-DR 모델은 기존 모델들에 비해 더 높은 효율성과 일반성을 보여주었습니다. 특히, 데이터스토어 압축 방법과의 호환성도 입증되었습니다.



### Synth-SBDH: A Synthetic Dataset of Social and Behavioral Determinants of Health for Clinical Tex (https://arxiv.org/abs/2406.06056)
Comments:
          Github: this https URL

- **What's New**: 최근 연구에서 Synth-SBDH라는 새로운 합성 데이터셋을 소개했습니다. 이 데이터셋은 15개의 SBDH(사회 및 행동 건강 결정 요인) 카테고리에 걸쳐 세부적인 주석을 포함하고 있습니다. 이 데이터셋은 두 병원의 실제 임상 데이터셋에서 우수한 성능을 보이며, 다양한 상황에서의 범용성과 성능 개선을 강조합니다.

- **Technical Details**: Synth-SBDH 데이터셋은 15개의 SBDH 카테고리를 기반으로, 상태, 시간 정보, 이유(rationale)를 포함한 주석을 제공합니다. 수작업으로 정의된 45가지 시드 예제를 기반으로, LLM(대형 언어 모델)을 사용해 생성되었습니다. 이러한 주석은 SBDH의 존재 여부(yes/no), 시간적 정보(과거 또는 현재), 이유 등 다양한 속성을 포함합니다.

- **Performance Highlights**: Synth-SBDH로 훈련된 모델들은 최대 62.5%의 매크로-F 점수 향상을 달성했습니다. 특히 드문 SBDH 카테고리에 대해서도 큰 성능 향상을 보였으며, 제한된 자원 하에서도 효과적입니다. 인간 평가에서는 Human-LLM 일치도가 71.06%로 나타났습니다.



### A Multidimensional Framework for Evaluating Lexical Semantic Change with Social Science Applications (https://arxiv.org/abs/2406.06052)
Comments:
          Accepted to the Proceedings of the Association for Computational Linguistics (ACL), 2024. Copyright c 2020 Association for Computational Linguistics (ACL). All Rights Reserved

- **What's New**: 최신 연구는 어휘 의미 변화(lexical semantic change)를 분석하기 위한 3차원 프레임워크를 제안했습니다. 이 프레임워크는 의미적 감정(sentiment), 범위(breadth), 강도(intensity)의 변화를 동시에 평가할 수 있는 통합된 계산 방법론을 제공합니다. 이를 통해 어휘 변화 과정을 경제적이고 체계적으로 맵핑(mapping)할 수 있습니다.

- **Technical Details**: 프레임워크의 세 가지 차원은 각각 의미적 감정의 증가 또는 감소, 의미 범위의 확대 또는 축소, 그리고 의미 강도의 증감입니다. 이 외에도 타겟 단어의 빈도 변화와 주제적 내용 변화(collocates)도 평가할 수 있습니다. 이론적 통찰과 자연어 처리(natural language processing) 방법론의 조화를 통해 어휘 의미 변화를 종합적으로 평가할 수 있습니다.

- **Performance Highlights**: 프레임워크의 유효성을 증명하기 위해 정신 건강과 정신 질환 관련 코퍼스(corpus)를 분석했습니다. 결과적으로 병리화(pathologization), 낙인(stigma), 개념 확장(concept creep)에 대한 현대적 우려를 나타내는 의미 변화 패턴이 드러났습니다.



### MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models (https://arxiv.org/abs/2406.06046)
Comments:
          The code is open-sourced at this https URL

- **What's New**: 대규모 웹 데이터 코퍼스로부터 고품질 데이터를 활용하여 언어 모델의 사전학습 효율성을 향상시킬 수 있는 새로운 데이터 선택 기법이 도입되었습니다. 기존의 손수 설계된 규칙이나 더 큰 참조 모델에 의존하는 정적 데이터 선택 방법과 달리, 본 연구에서는 'MATES'라는 모델 인지형 데이터 선택 기법을 제안합니다. MATES는 사전학습 모델의 진화하는 데이터 선호도를 반영하여 지속적으로 적응하는 데이터 영향 모델(data influence model)을 활용하여 현재 사전학습 단계에 가장 효과적인 데이터를 선택합니다.

- **Technical Details**: MATES 기법은 소규모 데이터 영향 모델을 튜닝하여 주요 모형의 성능을 예측하고, 해당 데이터를 다음 사전학습 단계에 활용합니다. 특히 로컬 프로빙(probing)을 통해 수집된 오라클 데이터 선호 신호를 바탕으로 데이터 영향 점수를 계산하여 데이터 선택을 최적화합니다. 이 접근법은 기존의 정적 데이터 선택 방식과 달리 학습 모델의 동적인 데이터 선호도를 지속적으로 반영합니다. 또한, Pythia와 C4 데이터셋에서의 실험을 통해 MATES가 기존의 무작위 데이터 선택 방식을 크게 능가하는 성능을 입증하였습니다.

- **Performance Highlights**: MATES는 여러 다운스트림 작업에서 평균 1.3%의 zero-shot 정확도 향상을 보여주었으며, 기존의 데이터 선택 접근방식보다 두 배 이상의 성능 향상을 이루었습니다. 더욱이 총 FLOPs 요구량을 절반으로 감소시켜 효율성을 입증했습니다. 추가 분석을 통해 사전학습 모델의 동적인 데이터 선호도를 효과적으로 포착하는 MATES의 장점이 검증되었습니다.



### The Curse of Popularity: Popular Entities have Catastrophic Side Effects when Deleting Knowledge from Language Models (https://arxiv.org/abs/2406.06032)
- **What's New**: 본 연구는 언어 모델(LMs)의 지식 삭제 문제를 탐구하며, 모델 내에 저장된 지식과 관련된 엔티티들의 관계를 분석합니다. 이 연구는 최초로 합성 지식 그래프(synthetic knowledge graphs)를 사용하여 지식 삭제를 분석하였으며, 이는 통제된 실험을 통해 모델의 지식 삭제 영향에 대한 새로운 방향을 제시합니다.

- **Technical Details**: 연구에서는 중요한 엔티티와 그렇지 않은 엔티티를 포함하는 두 가지 합성 지식 그래프를 생성했습니다. 첫 번째는 에르되시-레니(ER) 그래프이며, 두 번째는 바라바시-알버트(BA) 그래프로 실제 세계의 구조를 가깝게 모사합니다. GPT-2 구조의 언어 모델을 통해 다양한 층에서 지식 그래프들을 학습시키고, 특정 지식 삭제 기법을 적용하여 삭제된 지식이 다른 엔티티에게 미치는 영향을 분석합니다.

- **Performance Highlights**: 분석 결과, 자주 등장하는 엔티티와 관련된 지식을 삭제할 경우, 엔티티들이 모델에 미치는 영향이 매우 큰 것으로 나타났습니다. 특히, 이러한 지식 삭제는 실제 세계와 유사한 구조를 가진 지식 그래프에서 비극적인 부작용을 초래할 수 있습니다.



### HOLMES: Hyper-Relational Knowledge Graphs for Multi-hop Question Answering using LLMs (https://arxiv.org/abs/2406.06027)
Comments:
          Accepted at ACL 2024 in the main track

- **What's New**: 이 논문에서는 복잡한 질문에 대한 답변을 개선하기 위해 지식 그래프(KG)를 활용하는 새로운 접근 방식을 제안합니다. 복잡한 질문을 이해하고 비정형화된 텍스트를 필터링하고 종합하는 과정을 간소화하기 위해, 쿼리와 연관된 정보를 압축한 지식 그래프를 LLM (Large Language Models)에 입력으로 사용합니다.

- **Technical Details**: 기존 방법들은 비정형화된 텍스트에 구조화된 지식 삼중항(triples)을 통합하여 정보를 단순화하려 했으나, 이는 쿼리와 무관하게 추출된 모호한 사실을 포함하여 효과적이지 않았습니다. 이에 비해, 제안된 방법은 쿼리와 관련된 정보를 포함하도록 압축되고 맥락을 고려한 지식 그래프를 사용함으로써, 토큰(token) 수를 최대 67%까지 줄이는 데 성공했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 HotpotQA와 MuSiQue라는 두 개의 유명 벤치마크 데이터셋에서 SoTA (state-of-the-art) 방법을 여러 지표(EM, F1, BERTScore, Human Eval)에서 일관되게 능가하며 성능을 개선한 것으로 나타났습니다.



### Shoulders of Giants: A Look at the Degree and Utility of Openness in NLP Research (https://arxiv.org/abs/2406.06021)
Comments:
          Will appear in ACL 2024

- **What's New**: 이번 연구는 ACL Anthology에 아카이빙된 NLP 연구 논문을 분석하여 공개성(Open Culture) 수준과 커뮤니티에 미치는 이점을 정량화하려고 합니다. 논문들은 artefact 재사용 관련해서 다양한 패턴을 보이며, 조사한 논문의 30% 이상이 artefact를 공개하지 않는 것으로 나타났습니다. 또한, 언어별로 NLP 관련 artefact의 가용성에 큰 차이가 있음을 발견했습니다.

- **Technical Details**: 연구는 ACL Anthology(AA)와 Hugging Face에 발표된 논문을 분석 대상으로 하며, 논문을 언어별로 구분하고, artefact를 재사용한 빈도와 새로 생성된 artefact를 공개할 것인지 여부를 기록했습니다. 데이터 추출에서 355개의 논문 샘플을 분석했으며, LREC와 같은 특정 NLP 관련 학회 및 저널을 대상으로 분류했습니다. 논문을 분류한 후, 각 논문에서 새롭게 생성된 artefact의 공개 여부 및 artefact 재사용 패턴을 분석했습니다.

- **Performance Highlights**: 대부분의 논문(98.9%)은 이미 존재하는 artefact를 재사용하고 있으며, Main PV 카테고리에서 코드 재사용은 새 구현을 도입하는 데 활용됩니다. 코드, 데이터셋 등 기존 연구 artefact이 재사용되는 패턴은 연구가 발표된 학회나 저널에 따라 다르게 나타났습니다. 그러나, 새로 생성된 artefact를 공개하겠다는 약속을 어긴 논문도 상당수 존재했으며, 링크가 깨지거나 리소스 저장소가 비어있는 경우도 발견되었습니다.



### ThaiCoref: Thai Coreference Resolution Datas (https://arxiv.org/abs/2406.06000)
- **What's New**: 이번 연구에서는 태국어 코어퍼런스 해소(coreference resolution)를 위한 새로운 데이터셋인 ThaiCoref를 소개합니다. 이 데이터셋은 대학 에세이, 신문 기사, 연설문, 위키백과 등 네 가지 텍스트 장르에서 777,271개의 토큰, 44,082개의 멘션(mentions), 10,429개의 엔티티(entities)를 포함하고 있습니다.

- **Technical Details**: ThaiCoref의 주석 스키마(annotation scheme)는 OntoNotes 벤치마크를 기반으로 하며, 태국어의 특수한 현상을 다루기 위한 조정을 포함하고 있습니다. 이 데이터셋을 활용하여 멀티링구얼 인코더(multilingual encoder) 및 크로스-링구얼 트랜스퍼(cross-lingual transfer) 기법을 사용한 모델을 훈련했습니다.

- **Performance Highlights**: 최종적으로 테스트 세트에서 67.88%의 최고 F1 점수를 기록했습니다. 에러 분석 결과, 태국어의 고유한 언어적 특징이 도전 과제로 나타났습니다. 연구 성과를 NLP 커뮤니티에 기여하기 위해 데이터셋과 모델을 공개합니다.



### A Dual-View Approach to Classifying Radiology Reports by Co-Training (https://arxiv.org/abs/2406.05995)
Comments:
          Accepted by LREC-COLING 2024

- **What's New**: 본 논문은 방사선 보고서의 구조, 특히 Findings 섹션과 Impression 섹션이 각각 방사선 스캔의 다른 뷰(view)를 제공할 수 있다는 새로운 통찰을 제공합니다. 이를 기반으로 반지도 학습 (semi-supervised learning)을 활용해 두 개의 머신 러닝 모델을 각각 Findings와 Impression 섹션에 대해 구축하고, 이들이 상호 작용하여 성능을 향상시키는 ‘co-training’ 접근법을 제안합니다.

- **Technical Details**: 주어진 방사선 보고서에서 Findings와 Impression 섹션을 두 개의 서로 다른 뷰로 보고 각각의 섹션에 대해 두 개의 분류기를 훈련합니다. 각각의 분류기가 예측한 레이블을 상호 참조하여 대규모 미표기 데이터(unlabeled data)에서 성능을 향상시키는 과정을 포함합니다. 이를 통해 각 분류기는 반지도 학습을 통해 다른 섹션의 정보를 획득합니다. 이 방식으로 모델들이 결합된 앙상블(ensemble)을 만들어 최종 예측을 수행합니다.

- **Performance Highlights**: 캐나다 알버타 건강 서비스(Alberta Health Services)와 협력해 진행한 뇌종양 감시 프로젝트 실험에서, 제안된 co-training 접근법이 각 개별 모델을 반지도 학습 방식으로 향상시키고, 앙상블을 통해 성능을 추가로 향상시킴을 확인했습니다. 이는 소규모의 표기된 데이터에 기반한 지도 학습(supervised learning)이나 경쟁 반지도 학습 방식인 self-train을 모두 능가하는 결과를 보였습니다.



### Semisupervised Neural Proto-Language Reconstruction (https://arxiv.org/abs/2406.05930)
Comments:
          Accepted to ACL 2024

- **What's New**: 이 논문에서는 제한된 양의 레이블된 데이터로 훈련할 수 있는 반감독(semi-supervised) 역사적 재구축 모델을 제안합니다. 이 모델은 통상적으로 완전 감독이 필요한 기존 작업과는 달리, 적은 양의 레이블된 데이터(조음세트와 원형(proto-) 형태)와 많은 양의 비레이블된 데이터(조음세트만)로 훈련됩니다.

- **Technical Details**: 제안된 신경 아키텍처 'DPD-BiReconstructor'는 비교 방법에서 중요한 통찰을 차용하여, 딸려 있는 단어들로부터 재구축된 단어들이 다시 딸려 있는 단어로 결정론적으로 변환 가능해야 한다는 점을 반영합니다. 이 모델은 레이블되지 않은 조음 세트를 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 새로운 모델은 강력한 반감독 베이스라인(semisupervised baselines)을 능가하는 성능을 보여주며, 적은 양의 레이블된 데이터로도 성공적인 역사적 재구축을 수행할 수 있음을 입증했습니다.



### Hello Again! LLM-powered Personalized Agent for Long-term Dialogu (https://arxiv.org/abs/2406.05925)
Comments:
          17 pages, 4 figures

- **What's New**: LLMs 기반의 개방형 대화 시스템(Open-domain dialogue systems)이 짧고 단일 세션 상호작용에 치중한 기존 방식과 달리, 장기적인 대화 및 개인화된 상호작용을 지원하는 LD-Agent 프레임워크를 소개합니다. LD-Agent는 이벤트 요약 및 페르소나(persona) 관리를 통해 적절하고 일관된 장기 대화 응답을 생성할 수 있습니다.

- **Technical Details**: LD-Agent는 세 가지 모듈로 구성된 모델-독립적(model-agnostic) 프레임워크입니다. 이벤트 인식(event perception) 모듈은 장기 및 단기 메모리 은행을 사용하여 대화의 역사와 현재 세션을 관리하며, 주제 기반 검색 메커니즘을 통해 메모리 검색의 정확도를 높입니다. 페르소나 추출(persona extraction) 모듈은 사용자와 에이전트의 동적 페르소나 모델링을 수행합니다. 마지막으로, 반응 생성(response generation) 모듈은 회수된 메모리와 추출된 페르소나를 통합하여 적절한 대화 응답을 유도합니다.

- **Performance Highlights**: LD-Agent는 다양한 벤치마크와 모델, 작업에서 탁월한 성능을 보였으며, 효과성, 일반성, 그리고 크로스 도메인(cross-domain) 능력을 입증했습니다. LD-Agent는 MSC 및 Conversation Chronicles 등의 데이터셋에서 기존 방법보다 뛰어난 성능을 보여줍니다. 여러 에이전트 모델에 걸쳐 적용되고, 다중 참여자 대화 작업에서도 효과적인 성능을 입증했습니다.



### Why Don't Prompt-Based Fairness Metrics Correlate? (https://arxiv.org/abs/2406.05918)
Comments:
          In Proceedings of ACL main 2024

- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models)이 학습하는 잠재적인 편향(bias)에 대한 중요한 문제들을 제기하며, 이를 평가하고 완화하기 위한 여러 지표가 개발되었다고 보고합니다. 본 논문에서는 Prompt 기반 공정성 지표들이 낮은 상관관계를 보이며, 이는 공정성 평가의 신뢰성에 중요한 질문을 던진다고 밝혔습니다. 이러한 통찰에 기반하여, 공정성 지표 간의 상관관계를 높이기 위한 CAIRO(Correlated Fairness Output) 방법을 제안했습니다.

- **Technical Details**: CAIRO는 주어진 공정성 지표의 원래 프롬프트를 여러 사전 학습된 언어 모델을 사용하여 확장(Augment)하고, 확장된 프롬프트 조합 중에서 가장 높은 상관관계를 달성하는 조합을 선택하는 방법입니다. 연구에서는 gender(젠더)와 religion(종교) 편향에 대해 각각 0.3과 0.18이었던 Pearson 상관계수가 CAIRO 적용 후 0.90과 0.98로 크게 개선되었습니다.

- **Performance Highlights**: 실험 결과, CAIRO는 기존의 prompt 기반 공정성 지표들(BOLD, HolisticBias, HONEST)과 세 가지 대규모 프롬프트 확장 모델(ChatGPT, LLaMa 2, Mistral)을 사용하여 10개의 인기있는 언어 모델(GPT-2, GPT-J, GPT-Neo, OPT, Pythia 등)의 공정성 평가에서 높은 수준의 Pearson 상관계수(0.90 ~ 0.98)를 달성했습니다. 통계적 유의미성도 높게 나타났습니다(p값 0.0009 ~ 0.00006).



### TTM-RE: Memory-Augmented Document-Level Relation Extraction (https://arxiv.org/abs/2406.05906)
Comments:
          Accepted in ACL 2024 Main

- **What's New**: TTM-RE는 메모리 모듈(TTM)과 노이즈에 강한 손실 함수를 통합한 새로운 접근 방식으로, 대규모 노이즈 학습 데이터를 효과적으로 활용하여 문서 수준 관계 추출 성능을 대폭 향상시켰습니다.

- **Technical Details**: TTM-RE는 Token Turing Machine (TTM) 기반의 메모리 모듈과 긍정-비라벨(PU) 설정을 고려한 노이즈-강건 손실 함수를 통합하여 설계되었습니다. 이 접근 방식은 ReDocRED와 같은 문서 수준 관계 추출 벤치마크 데이터셋에서 특히 유효성을 입증했습니다.

- **Performance Highlights**: ReDocRED 벤치마크 데이터셋에서 TTM-RE는 기존 최고 성능 대비 절대적인 F1 점수 3% 이상 향상을 달성했습니다. 또한 ChemDisGene와 같은 다른 도메인 및 고도로 비라벨된 설정에서도 탁월한 성능을 보였습니다. 예를 들어, 매우 비라벨된 시나리오에서 기존 최고 성능을 12%F1 점수로 초과했습니다.



### Feriji: A French-Zarma Parallel Corpus, Glossary & Translator (https://arxiv.org/abs/2406.05888)
- **What's New**: 최초의 견고한 프랑스어-자르마 병렬 코퍼스인 'Feriji'가 도입되었습니다. 자르마는 주로 니제르에서 500만 명 이상이 사용하는 송하이(Songhay) 계열의 언어로, 기계 번역(MT)에서 거의 다뤄지지 않은 언어입니다.

- **Technical Details**: Feriji 코퍼스는 61,085개의 자르마 문장과 42,789개의 프랑스어 문장으로 구성되어 있으며, 4,062개의 단어가 포함된 용어집을 추가로 제공합니다. 세 가지 대형 언어 모델(T5-small, MT5-small, M2M100 및 NLLB-200-distilled-600M)을 사용해 자르마 번역 작업에 대해 미세 조정(fine-tuning)했습니다.

- **Performance Highlights**: 가장 성능이 뛰어난 모델은 M2M100으로, BLEU 점수 30.06을 기록했습니다. 인간 평가에서도 모델의 유창성, 이해도 및 가독성을 검토했습니다.



### Are Large Language Models Actually Good at Text Style Transfer? (https://arxiv.org/abs/2406.05885)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 텍스트 스타일 변환(Text Style Transfer, TST)에서 어떻게 성능을 발휘하는지 분석했습니다. 특히 감정 전이(sentiment transfer) 및 텍스트 디톡스(text detoxification)에 중점을 두어 영어, 힌두어, 벵골어 세 언어에서의 성능을 비교했습니다.

- **Technical Details**: 텍스트 스타일 변환은 텍스트의 핵심 내용을 유지하면서 언어적 스타일을 변경하는 작업입니다. 본 연구에서는 사전 훈련된 대형 언어 모델들을 사용하여 제로샷(zero-shot)과 퓨샷(few-shot) 프롬팅(prompting) 및 파라미터 효율적인 미세 조정(parameter-efficient finetuning)을 통해 평가했습니다. 공개된 데이터셋을 활용하여 자동 평가 지표, GPT-4 평가 및 인간 평가를 통해 성능을 분석했습니다.

- **Performance Highlights**: 평가 결과, 일부 프롬팅된 대형 언어 모델이 영어에서는 높은 성능을 발휘했지만, 힌두어와 벵골어에서는 평균적인 성능에 머물렀습니다. 그러나 미세 조정을 통해 제로샷 및 퓨샷 프롬팅보다 성능이 크게 향상되었으며, 이는 이전의 최첨단 수준(state-of-the-art)과 비교할 만한 정도로 향상되었습니다. 이는 효과적인 텍스트 스타일 변환을 위해 전용 데이터셋과 특화된 모델이 필요하다는 점을 강조합니다.



### Zero-Shot End-To-End Spoken Question Answering In Medical Domain (https://arxiv.org/abs/2406.05876)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 이번에 발표된 논문은 의료 분야에서 대화형 질문 답변(Spoken Question Answering, SQA)에 대한 새로운 제로샷(Zero-shot) 접근법을 소개합니다. 기존 SQA 방식은 오디오 질문을 텍스트로 변환하고, 이를 다시 언어 모델(Language Model, LM)로 입력하여 답을 예측하는 여러 단계를 거쳐야 하지만, 이에 따라 오류와 자원 소모가 증가합니다. 반면, 이번 연구는 직접 오디오 데이터를 처리할 수 있는 End-to-End(E2E) 접근법을 통해 이러한 문제를 해결하려 합니다.

- **Technical Details**: 연구팀은 총 8개의 의료 관련 작업에 대해 48시간의 합성 오디오를 포함한 새로운 벤치마크를 이용하여 평가를 진행했습니다. 이 접근법은 종합적으로 14.7배 적은 자원을 소모하면서 정확도가 평균 0.5% 개선된 결과를 보였습니다. 연구에서는 또한 각 음성 인코더 레이어 내에서 SQA 작업에 필요한 정보 배치에 대한 심층 분석을 수행했습니다.

- **Performance Highlights**: 제안된 E2E 방식은 특히 대규모 언어 모델과 ASR(Automatic Speech Recognition) 모델을 조합한 기존 방식보다 자원 소모를 현저히 줄이면서도 성능 면에서 더 나은 결과를 보였습니다. 구체적으로, 1.3B 파라미터의 LLM과 1.55B 파라미터의 ASR 모델을 합친 기존 시스템 대비 최대 14.7배 적은 자원으로 평균 정확도를 0.5% 향상시켰습니다.



### II-Bench: An Image Implication Understanding Benchmark for Multimodal Large Language Models (https://arxiv.org/abs/2406.05862)
Comments:
          100 pages, 82 figures

- **What's New**: 다중모드 대형 언어 모델(multimodal large language models, MLLMs)의 발전 속도가 빠르며, 이를 평가하기 위한 다양한 벤치마크도 제시되고 있습니다. 그러나, MLLMs의 고차원적인 지각(perceptual) 능력을 평가하는 연구는 부족합니다. 이를 해결하기 위해, 우리는 이미지를 이해하는 고차원적인 지각 능력을 평가하기 위한 새로운 벤치마크인 II-Bench를 제안합니다.

- **Technical Details**: II-Bench는 MLLMs의 이미지를 이해하는 고차원적인 능력을 평가하기 위해 디자인되었습니다. 여러 MLLMs를 대상으로 광범위한 실험을 수행한 결과, MLLMs와 인간의 성능 사이에 큰 격차가 존재함을 발견했습니다. 구체적으로, MLLMs의 최고 성능은 74.8%의 정확도를 기록한 반면, 인간의 평균 정확도는 90%에 달하며, 최고치는 98%에 이릅니다.

- **Performance Highlights**: MLLMs는 추상적이고 복잡한 이미지를 이해하는 능력이 떨어지며, 이는 고차원적인 의미를 파악하고 이미지 세부 사항을 캡처하는 능력이 제한적임을 나타냅니다. 또한, 이미지 감정 극성 힌트(image sentiment polarity hints)를 프롬프트에 포함하면 모델의 정확도가 향상되는 것을 관찰했으며, 이는 이미지 감정을 본질적으로 이해하는데 있어 결함이 있음을 보여줍니다. 이 벤치마크는 커뮤니티가 전문가 수준의 인공지능(AGI)을 개발하도록 영감을 줄 것으로 기대됩니다.



### MedREQAL: Examining Medical Knowledge Recall of Large Language Models via Question Answering (https://arxiv.org/abs/2406.05845)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 건강 주제를 포함한 다양한 질문에 대한 답변을 생성할 때 얼마나 잘 기억할 수 있는지를 평가하는 새로운 방법론을 제안합니다. 이 연구는 체계적인 리뷰에서 추출한 질문-답변 쌍으로 구성된 새로운 데이터셋인 MedREQAL을 사용하여 LLM의 성능을 분석합니다.

- **Technical Details**: 이 연구는 MedREQAL이라는 새로 개발된 데이터셋을 사용하여 GPT와 Mixtral 등 6개의 LLM의 의료 지식 회상 성능을 평가했습니다. 데이터셋은 주로 Cochrane Collaboration의 체계적인 리뷰에서 추출된 질문과 답변으로 구성되어 있으며, 각 리뷰의 목표(Objective) 섹션에서 질문을 생성하고, 결론(Conclusion) 섹션에서 답변을 추출했습니다.

- **Performance Highlights**: 분석 결과, LLM이 의료 지식 회상에서 여전히 도전에 직면하고 있음을 발견했습니다. 실험은 분류 및 생성 성능을 평가하여 LLM의 개별 능력과 한계에 대한 통찰을 제공했습니다. 또한, 데이터셋은 다양한 건강 분야에 고르게 분포되어 있어 높은 품질을 보입니다.



### Seventeenth-Century Spanish American Notary Records for Fine-Tuning Spanish Large Language Models (https://arxiv.org/abs/2406.05812)
- **What's New**: 이번 논문에서는 17세기 스페인어로 작성된 공증 기록을 활용하여 스페인어 대형 언어 모델(LLMs)을 미세 조정(fine-tuning)할 수 있는 자료를 소개했습니다. 이 자료는 아르헨티나 국립 기록 보관소에서 얻은 것으로, 160여 페이지에 두 명의 공증인이 작성한 손으로 쓴 원본 이미지와 전사된 텍스트(그리고 메타데이터)를 포함하고 있습니다.

- **Technical Details**: 이번 연구에서는 17세기 스페인어 공증 기록의 특정 부분을 선별하여 미세 조정에 필요한 데이터셋인 SANRlite를 구성했습니다. 이 데이터셋은 두 명의 공증인, Estenban Agreda de Vergara와 Nicolas de Valdivia y Brisuela가 작성한 160여 페이지의 기록으로 구성되어 있으며, 각각의 문장들은 공증인의 삽화를 통해 전사되었습니다. 연구진은 이를 활용하여 스페인어 LLMs를 분류(classification)와 마스크 언어 모델링(masked language modeling) 작업에 미세 조정했습니다.

- **Performance Highlights**: 실증 평가 결과, SANRlite를 활용한 미세 조정된 스페인어 LLMs는 기존의 사전 학습된 스페인어 모델과 ChatGPT-3.5/ChatGPT-4o 대비 우수한 성능을 보였습니다. 특히 분류 작업에서는 높은 F1 score와 정확도를 달성했으며, 마스크 언어 모델링 작업에서도 성능이 뛰어났습니다. 이를 통해 SANRlite가 역사 텍스트 분석, 자연어 이해, 정보 검색 등에 유용한 자원이 될 것임을 입증했습니다.



### Do Prompts Really Prompt? Exploring the Prompt Understanding Capability of Whisper (https://arxiv.org/abs/2406.05806)
Comments:
          In progress

- **What's New**: 본 연구는 고성능 음성 인식 모델인 Whisper가 프롬프트(이하 prompt) 내의 정보를 어떻게 처리하는지 탐구합니다. 예상과는 다르게 Whisper가 텍스트 프롬프트를 완전히 이해하지 못하는 것으로 나타났습니다. 더욱이, 주제 정보에 강하게 일치하는 프롬프트를 사용하는 것이 성능 향상을 보장하지 않는 다는 점도 발견했습니다. 흥미롭게도, 영어 프롬프트가 중국어 프롬프트보다 더 나은 성능을 보이는 경향이 있으며, 이는 언어별 학습 데이터 분포 차이에 기인할 수 있습니다. 반면 Whisper가 언어 토큰의 혼란스러운 정보를 효과적으로 무시하고 올바른 언어 토큰에 집중하는 능력을 보였다는 점도 중요한 발견 중 하나입니다. 이 연구는 Whisper의 프롬프트 이해 능력에 대한 의문을 제기하며 추가 연구를 촉구합니다.

- **Technical Details**: Whisper는 텍스트 프롬프트와 관련된 다양한 시나리오에서 테스트되었으며, 이를 통해 프롬프트의 정보가 모델의 성능에 미치는 영향을 평가했습니다. 연구 결과, Whisper는 텍스트 프롬프트를 정확히 이해하지 못하고, 주제 정보가 일치하지 않는 경우에도 일치하는 경우보다 더 나은 성능을 보였습니다. 회귀 분석(Regression Analysis)을 통해 프롬프트 이해와 성능 간의 긍정적인 상관 관계가 없음을 확인하였습니다. 또한, Whisper는 잘못된 언어 토큰을 무시하고 올바른 언어 토큰을 활용하는 경향을 보였습니다. 이는 Whisper가 광범위한 예비 학습 덕분에 언어 토큰의 중복 정보를 걸러낼 수 있음을 시사합니다.

- **Performance Highlights**: Whisper가 영어 프롬프트에서 더 나은 성능을 보이는 경향을 확인했습니다. 이는 중국어 테스트 데이터에 대해서도 동일하게 나타났습니다. 코드-스위치 ASR(Code-Switched ASR) 시나리오에서는, 존재하지 않는 언어 구성 요소가 있는 토큰 쌍이 제공될 때 성능 저하가 관찰되었습니다. 반면 잘못된 언어 토큰이 포함된 상황에서도 Whisper는 올바른 언어를 기반으로 예측을 생성하여 높은 정확도를 유지하는 능력을 보였습니다. 이 발견은 Whisper의 프롬프트 처리 능력에 대한 새로운 관점을 제시하며, 추가 연구의 필요성을 강조합니다.



### Hidden Holes: topological aspects of language models (https://arxiv.org/abs/2406.05798)
- **What's New**: 이번 논문에서는 원시 텍스트 데이터로 훈련된 자회귀 신경 언어 모델의 표현 매니폴드(Representation Manifold)의 위상을 탐구합니다. 이를 위해 계산적 대수 위상(Computational Algebraic Topology) 도구를 도입하였고, 이를 바탕으로 우리가 '천공(Perforation)'이라고 부르는 위상적 복잡성의 측정치를 제안했습니다. 이 측정치를 이용하여 GPT 기반의 대형 언어 모델의 깊이와 학습 과정에서 나타나는 위상 구조의 변화를 연구했습니다.

- **Technical Details**: 우리의 방법론은 매 훈련 에폭(Epoch)마다 다음 단계로 나뉩니다: 1) 훈련 코퍼스에서 문장을 선택, 2) 문장을 언어 모델에 입력하고 숨겨진 레이어의 활성화 상태를 기록, 3) 숨겨진 상태의 위상적 특징을 계산. 이 과정에서 사용된 주된 기법은 지속성 호몰로지(Persistent Homology)와 간단 복소체 근사 (Simplicial Mapping Approximation), 슬라이딩 윈도우 임베딩(Sliding Window Embedding)입니다. 우리는 '천공(Perforation)'이라는 새로운 색인을 제안하였고, 이를 통해 신경망이 자연어 데이터를 학습하면서 나타나는 표현 매니폴드의 변화를 추적했습니다.

- **Performance Highlights**: 연구 결과, 게이트 순환 모델(Gated Recurrent Models)이 GPT 기반 모델보다 더 높은 위상적 복잡성을 나타내며, 모든 자연어에 공통적으로 나타나는 독특한 패턴의 변화를 보여주었습니다. 이는 합성 데이터에서는 나타나지 않은 특징입니다. 이 발견은 대형 트랜스포머 언어 모델의 작동을 이해하기 위해 신경망의 수학적 특성에 대한 추가 연구가 필요함을 시사합니다. 또한, 위상적 분석을 통한 모델 재매개변수화와 위상 정규화 기법의 가능성을 제시함으로써, 효율적이고 지속 가능한 NLP 시스템의 개발에 기여할 수 있음을 보여줍니다.



### RE-RAG: Improving Open-Domain QA Performance and Interpretability with Relevance Estimator in Retrieval-Augmented Generation (https://arxiv.org/abs/2406.05794)
- **What's New**: 이번 연구에서는 RE-RAG 프레임워크가 소개되었습니다. 이 프레임워크는 RAG 시스템에 문맥 관련성 추정기(기존 컨텍스트의 재평가)를 추가하여 성능을 향상시킵니다.

- **Technical Details**: RE-RAG는 외부 지식 기반에서 검색한 문맥을 문맥 관련성 추정기(RE)를 통해 재평가합니다. 이 RE는 비지도 학습 방법으로 훈련되어 질문-문맥 적합성을 판단합니다. 기존 RAG의 해석 가능 구조를 유지하면서 관련성 추정기를 포함하여 정확한 관련성 측정을 제공합니다.

- **Performance Highlights**: RE-RAG는 Natural Questions 및 TriviaQA 데이터셋에서 성능을 테스트하였으며, 기존 FiD 모델과 비교할 때 훨씬 적은 문맥(0.25배)으로도 유사한 성능을 달성했습니다. 또한, T5 모델로 학습된 RE가 LLMs(ChatGPT)에서도 성능 향상(각각 NQ: +6.4EM, TQA: +2.8EM)을 보여주었습니다. RE는 문맥 세트를 사전 평가하여 비답문맥을 필터링하는 데에도 효율적이었습니다.



### The BiGGen Bench: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models (https://arxiv.org/abs/2406.05761)
Comments:
          Work in Progress

- **What's New**: BiGGen Bench를 소개합니다. 이는 9가지의 모형(LM, Language Models) 능력을 77개의 다양한 작업(task)에서 철저하게 평가하기 위한 새로운 기준입니다. 기존의 평가 기준이 추상적이어서 인간 평가의 세부성을 반영하지 못하던 문제점을 해결하고, 특정 기능에 치우치는 경향을 개선하고자 합니다.

- **Technical Details**: BiGGen Bench는 각 인스턴스(instance)별로 세부적인 평가 기준을 사용하여 보다 구체적이고 유연한 평가를 가능하게 합니다. 이를 통해 평가 기준이 더 정교하게 조정되고, 인간 평가와 유사한 평가를 목표로 합니다. 이번 연구에서는 103개의 최첨단 모형을 5개의 평가자 모형을 사용하여 평가하였습니다.

- **Performance Highlights**: 코드, 데이터, 평가 결과가 모두 공개되어 있으며, 이를 통해 다양한 언어모델의 성능을 보다 투명하고 철저하게 분석할 수 있습니다. 이 새로운 벤치마크는 다양한 작업에서 언어모델의 능력을 종합적으로 평가하는 데 중요한 도구가 될 것입니다.



### Arabic Diacritics in the Wild: Exploiting Opportunities for Improved Diacritization (https://arxiv.org/abs/2406.05760)
Comments:
          Accepted to ACL 2024

- **What's New**: 이 논문은 아랍어 텍스트에서 자연스럽게 발생하는 방점(diacritical marks)의 패턴을 분석하기 위해 'WildDiacs'라는 새로운 개념을 도입했습니다. 이 개념을 통해 뉴스 기사, 소설, 동화책, 시, 정치 문서, ChatGPT 출력 등 6개의 다양한 장르를 분석하여 도출된 패턴과 잠재 정보를 연구했습니다.

- **Technical Details**: 저자들은 실세계의 부분적으로 방점이 찍힌 단어들을 문맥 상 최대한 완전한 방점으로 매핑하는 새로운 주석 데이터셋을 제공했습니다. 또한 분석 및 모호성 해소 접근법(analyze-and-disambiguate approach)을 확장하여 아랍어 NLP 품질을 향상시켰습니다. 이 논문에서는 Penn Arabic Treebank와 WikiNews라는 두 개의 일반적으로 사용되는 데이터셋도 분석합니다.

- **Performance Highlights**: 하이브리드 뉴로-심볼릭(neuro-symbolic) 알고리즘을 활용하여 WildDiacs의 존재를 반영한 새로운 방점화 알고리즘을 제안했으며, 이를 통해 성능 향상을 입증했습니다. 코드와 데이터셋은 오픈 소스로 공개될 예정입니다.



### MrRank: Improving Question Answering Retrieval System through Multi-Result Ranking Mod (https://arxiv.org/abs/2406.05733)
Comments:
          To be published in Findings of ACL 2024

- **What's New**: 이번 연구에서는 다양한 Information Retrieval(IR) 시스템들을 결합하여 최신 지식을 기반으로 한 QA 시스템의 성능을 극대화하는 새로운 접근 방식을 제안합니다. 기존의 재탐색(Retrieval-Augmented Generation, RAG) 시스템의 성능 병목 현상을 해결하기 위해 학습을 통한 순위 결정(Learning to Rank, LTR) 기법을 도입해 이종 IR 시스템들을 결합하는 방법을 도입했습니다.

- **Technical Details**: 제안된 방법은 주로 두 단계로 이루어집니다: 첫째, off-the-shelf 리트리버(retriever)를 사용하여 후보 풀을 생성하는 검색 단계; 둘째, 재랭킹 네트워크를 통해 최종 순위를 결정하는 재랭킹 단계입니다. 이 과정에서, neural-based 모델과 전통적인 BM25 리트리버가 결합되며, 최종 랭킹을 예측하기 위해 다시 랭킹 모델이 사용됩니다. 주요 모델로는 SGPT-5.8B-msmarco와 MPNET 등을 포함시켰습니다.

- **Performance Highlights**: ReQA SQuAD에서 제안된 방법론은 기존의 최첨단(Zhao et al., 2023)보다 우수한 성능을 발휘했으며, 모든 개별 리트리버 모델, RRF, 통계적 라우팅 전략 등을 능가했습니다. 이로 인해 데이터셋 전반에서 평균 역순위 평균(MRR)가 13.6% 향상되었습니다. 이는 복합적인 QA 리트리버 모델 사용을 통해 얻은 성과입니다.



### QGEval: A Benchmark for Question Generation Evaluation (https://arxiv.org/abs/2406.05707)
- **What's New**: 최신 연구는 Question Generation (QG) 분야의 문제점을 해결하기 위해 QGEval이라는 다차원 평가 벤치마크를 제안합니다. 이는 일반적으로 생성된 질문의 품질 평가에서 발생하는 애매모호함이나 사실적 부정확성을 해결하고자 합니다. QGEval은 7가지 차원에서 생성된 질문과 기존 자동화 메트릭을 평가합니다: 유창성(fluency), 명확성(clarity), 간결성(conciseness), 관련성(relevance), 일관성(consistency), 답변 가능성(answerability), 및 답변 일관성(answer consistency).

- **Technical Details**: QGEval은 SQuAD와 HotpotQA 데이터셋을 기반으로 15개의 QG 모델이 200개의 패시지에서 생성한 3000개의 질문을 포함하고 있습니다. 평가 기준은 언어적 차원과 과제 지향적 차원의 두 가지 카테고리로 나뉩니다. 언어적 차원에는 유창성, 명확성, 간결성이 포함되고, 과제 지향적 차원에는 관련성, 일관성, 답변 가능성, 답변 일관성이 포함됩니다. 다양한 모델과 설정을 사용하여 질문 생성이 이루어졌으며, 비고와 함께 평가되었습니다.

- **Performance Highlights**: QGEval을 통해 평가된 결과는 대부분의 QG 모델이 답변 가능성과 답변 일관성 측면에서 부족하다는 것을 보여주었습니다. 또한, 15개의 기존 자동화 메트릭은 인간 평가에 비해 여전히 큰 격차가 있음을 확인했습니다. 이를 통해 현재의 모델과 메트릭의 단점을 밝히고, 향후 연구를 위한 통찰력을 제공합니다.



### MoPS: Modular Story Premise Synthesis for Open-Ended Automatic Story Generation (https://arxiv.org/abs/2406.05690)
Comments:
          ACL 2024, camera-ready

- **What's New**: 이번 연구에서는 이야기를 생성하는 데 중요한 역할을 하는 이야기 전제(premise)의 자동화된 설계를 위한 새로운 접근법인 모듈식 이야기 전제 합성법(Modular Story Premise Synthesis, MoPS)을 소개합니다. MoPS는 전제를 배경(background), 페르소나(persona) 등과 같은 모듈로 나누어 구성하며, 각 모듈에서 후보군을 수집하고 이를 결합해 완전한 이야기 전제를 만듭니다.

- **Technical Details**: MoPS는 세 가지 주요 단계로 구성됩니다: 1) 각 모듈의 일관된 후보군을 미리 수집해 중첩 사전을 형성합니다. 2) 중첩 사전에서 키 경로를 추출해 전제 디자인을 구성합니다. 3) 대형 언어 모델(LLM)을 사용해 디자인을 일관된 전제 문장으로 통합합니다. 이들은 각각의 모듈들을 창의적으로 결합하여 이야기를 생성하게 됩니다.

- **Performance Highlights**: MoPS로 생성된 전제는 기존의 LLM과 공개 데이터셋에서 추출된 전제보다 더 다양하고 매력적이며 완전하며 독창적이라는 평가를 받았습니다. 이를 토대로 생성된 소설과 대본 역시 높은 품질을 보여줍니다. 특히, MoPS는 최대 7.6k개의 전제와 1k개의 확장된 이야기를 생성할 수 있으며, 이는 이야기 생성 파이프라인의 전반적인 품질 향상에 기여합니다.



### Peer Review as A Multi-Turn and Long-Context Dialogue with Role-Based Interactions (https://arxiv.org/abs/2406.05688)
Comments:
          Under review

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 이용한 동적 다중 라운드 피어 리뷰 프로세스를 제안합니다. 기존의 정적 리뷰 생성에서 벗어나, 저자, 리뷰어, 결정자를 포함한 다중 턴(long-context)의 대화를 통해 실세계 피어 리뷰의 동적이고 반복적인 특성을 포착하고자 합니다. 이를 위해 26,841개 논문과 92,017개의 리뷰를 포함하는 방대한 데이터를 수집하여 공개했습니다.

- **Technical Details**: 이번 연구에서 제시된 새로운 프레임워크는 다음과 같은 주요 원칙을 따릅니다: 1) Long-Context: 논문의 광범위한 컨텍스트를 바탕으로 한 대화, 2) Multi-Turn: 다중 라운드 대화로 실제 세계의 반복적인 피어 리뷰 프로세스를 시뮬레이션, 3) Role-Based: 저자, 리뷰어, 결정자의 역할 구분. 이를 통해 각 역할은 구체적인 책임과 목표를 갖고 상호작용을 합니다. 또한 다양한 도메인과 주제를 다루는 ReviewMT 데이터세트를 제공하여 LLM의 성능을 풍부하게 평가할 수 있도록 합니다.

- **Performance Highlights**: 이번 연구에서는 LLM의 성능을 공정하고 포괄적으로 평가하기 위해 역할별 평가 지표를 제안했습니다. 이러한 지표는 생성된 응답의 유효성, 텍스트 품질, 최종 리뷰의 점수 평가 및 결정자의 결정 평가를 포함합니다. 이 새로운 프레임워크와 데이터셋을 통해, 동적이고 역할 기반의 상호작용을 포함한 LLM 기반 피어 리뷰 프로세스가 크게 향상될 것으로 기대됩니다.



### SinkLoRA: Enhanced Efficiency and Chat Capabilities for Long-Context Large Language Models (https://arxiv.org/abs/2406.05678)
Comments:
          A rethinking of Short Shifted Attention

- **What's New**: Transformer 모델을 더 긴 시퀀스 길이를 다룰 수 있도록 확장하는 것은 이제 중요한 과제가 되었습니다. 이는 언어 번역, 긴 문맥 처리, 챗봇, 코드 생성 및 멀티미디어 콘텐츠 제작과 같은 작업에서 성능을 향상시키기 위해 필수적입니다. LongLoRA는 시퀀스 길이를 확장하고 비슷한 성능을 유지하면서도 계산 절감을 달성하지만, 여전히 최적화의 여지가 있었습니다. 이를 개선하기 위해 SinkLoRA를 제안합니다. 이는 SF-Attn과 

- **Technical Details**: SinkLoRA는 보다 나은 작업 분할 기능을 제공합니다. 특히, (1) 분할 및 재조립 알고리즘을 사용하여 다양한 어텐션 헤드 패턴의 혼란을 방지하고 'sink attention tokens(집중 어텐션 토큰)'의 글로벌 어텐션을 통해 초기 상태로 비순환적으로 돌아가는 SF-Attn을 개발했습니다. 이로 인해 전체 어텐션과 비교하여 92%의 복잡성 개선을 달성할 수 있습니다. (2) H₂O라고 불리는 최첨단 KV 캐시 압축 알고리즘을 적용해 추론 속도를 가속화합니다.

- **Performance Highlights**: SinkLoRA는 LongLoRA와 비교하여 LLaMA2 7B에서 향상된 성능을 보였으며 LongChat-13B와 경쟁할 만한 성능을 보여주었습니다. PG19, Proof-pile, LongBench dataset에서의 평가 결과를 통해 효과적임이 입증되었습니다. 특히, 슈퍼바이즈드 파인 튜닝(sUpervised Fine-tuning)을 통해 수집한 LongAlpaca-Plus 데이터셋을 사용하여 다양한 출처의 질문과 답변을 포함한 데이터셋을 구성하였으며, 전체 어텐션 대비 92%의 perplexity 개선을 달성했습니다.



### MS-HuBERT: Mitigating Pre-training and Inference Mismatch in Masked Language Modelling methods for learning Speech Representations (https://arxiv.org/abs/2406.05661)
Comments:
          4 pages, submitted to interspeech2024

- **What's New**: 최근 몇 년간 음성 데이터를 이용해 고수준 정보를 학습하는 자가 지도 학습(pre-training) 방법이 주목받고 있습니다. 이 중 HuBERT는 음성 인식 분야에서 SOTA 성능을 보여주었으나, 데이터2vec에 비해 사전 학습 전략에서 뒤처지고 있습니다. 이를 개선하기 위해 본 논문에서는 (i) 사전 학습과 추론의 불일치를 해결하기 위한 Swap 방법과 (ii) 모델의 용량(capasity)을 더 효과적으로 활용하기 위한 Multicluster masked prediction loss 방식을 도입한 MS-HuBERT를 제안합니다. 이 방법은 ASR Librispeech 벤치마크에서 기존 HuBERT에 비해 평균 5% 이상 성능 향상을 보였습니다.

- **Technical Details**: HuBERT는 CNN과 변환기(transformer) 아키텍처를 기반으로 한 반복적인 사전 학습 방식의 SSL 방법입니다. MS-HuBERT는 HuBERT 모델을 두 가지 방식으로 개선합니다: (i) Swap 방법을 도입하여 사전 학습과 추론 간의 불일치를 해결하고, (ii) Multicluster MPL을 적용하여 모델의 용량을 극대화하여 ASR 작업에 적합한 기능을 학습합니다. Swap 방법을 통해 사전 학습 중에 마스킹된 입출력 뷰를 모두 사용하여 더 일관된 학습을 가능하게 합니다.

- **Performance Highlights**: 제안된 MS-HuBERT 방법은 ASR Librispeech 벤치마크에서 원래의 HuBERT를 크게 능가하며, 고자원 환경에서도 데이터2vec와 대등한 성능을 보였습니다.



### Do LLMs Exhibit Human-Like Reasoning? Evaluating Theory of Mind in LLMs for Open-Ended Responses (https://arxiv.org/abs/2406.05659)
- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)의 이론적 사고(Theory of Mind, ToM) 능력을 공개형 질문에서 평가합니다. Reddit의 ChangeMyView 플랫폼에서 사회적 추론과 설득력 있는 응답을 요구하는 게시물을 사용해 분석을 진행했습니다.

- **Technical Details**: 연구는 LLM이 사람의 의도와 감정을 통합하여 ToM 추론을 수행할 수 있는 능력을 평가하고, 이를 개선하기 위해 프롬프트 튜닝(prompt tuning) 방법을 도입했습니다. 주요 분석 방법으로는 인간과 LLM 응답 간의 의미적 유사성(semantic similarity)과 어휘 중복율(lexical overlap)을 비교했습니다.

- **Performance Highlights**: 분석 결과, 가장 진보된 모델조차 공개형 질문에서 인간의 ToM 추론 능력과 상당한 차이를 보였습니다. 제안된 프롬프트 튜닝 방법을 통해 어느 정도의 성능 향상이 있었으나, 여전히 인간과 같은 수준의 추론은 달성되지 못했습니다.



### DomainRAG: A Chinese Benchmark for Evaluating Domain-specific Retrieval-Augmented Generation (https://arxiv.org/abs/2406.05654)
- **What's New**: 최근 발표된 논문에서는 Retrieval-Augmented Generation (RAG) 모델이 대형 언어 모델(LLMs)의 한계를 극복하는 해결책으로 주목받고 있습니다. 특히 전문가와 도메인 별 애플리케이션에서 RAG 모델의 필요성이 강조되었습니다. 해당 논문에서는 대학 입학 시나리오에서 RAG 모델의 성능을 평가하였으며, 여섯 가지 필수 능력을 분석하였습니다.

- **Technical Details**: RAG 시스템을 이해하기 위해 논문에서 제시된 여섯 가지 주요 능력은 다음과 같습니다: 1) 대화형 RAG에서의 능력, 2) 구조적 정보 분석, 3) 외부 지식의 신뢰성, 4) 노이즈 제거(denoising), 5) 시간에 민감한 문제 해결, 6) 다중 문서 상호작용 이해. 이 능력들을 평가하기 위해 각 능력에 대응하는 데이터셋이 제공되었습니다. 평가된 모델은 Llama, Baichuan, ChatGLM, GPT 등입니다.

- **Performance Highlights**: 실험 결과 기존의 'closed-book' LLM는 도메인 별 질문에 대처하는 데 어려움을 겪었으며, 이는 RAG 모델이 전문가 문제를 해결하는 데 필요하다는 것을 강조합니다. 또한, 대화 히스토리 이해, 구조적 정보 분석, 노이즈 제거, 다중 문서 상호작용, 외부 지식의 신뢰성 측면에서 RAG 모델의 향상 가능성이 있는 것으로 나타났습니다.



### How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States (https://arxiv.org/abs/2406.05644)
Comments:
          27 pages

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 안전 정렬(safety alignment)에 대해 다룹니다. 기존의 연구들은 주로 모델이 훈련 과정에서 윤리적 개념을 학습하는 것과 이에 따른 정렬 과정을 다루었으나, 이 논문은 약한 분류기(weak classifiers)를 사용해 이러한 과정을 설명하고, 정렬 과정에서의 메커니즘을 밝혀내고자 합니다. 특히, 악성 사용자 입력에 대응하는 모델의 안전 가드레일(safety guardrails)을 우회하는 방법인 'jailbreak'에 초점을 맞추었습니다.

- **Technical Details**: 연구진은 LLM의 초기 층에서 이미 윤리적 개념을 학습하고 있다는 점을 확인했습니다. 초기 층에서 악성 및 정상 입력을 구분할 수 있으며, 중간 층에서는 이러한 개념이 감정 예측(emotion guesses)과 연결되고, 마지막 층에서는 이를 특정 거부 토큰(reject tokens)으로 정제하여 안전한 생성으로 이어지게 합니다. 반면, jailbreak는 초기 비윤리적 분류가 부정적 감정으로 전환되는 과정을 방해합니다. 이 논문에서는 7B에서 70B 규모의 다양한 모델 패밀리를 실험 대상으로 사용했습니다.

- **Performance Highlights**: 실험 결과, 다양한 모델에서 일관되게 LLM의 안전 메커니즘을 밝혀내어 결론을 도출할 수 있었습니다. 이로 인해 LLM 안전성에 대한 새로운 관점을 제공하고, 우려를 줄일 수 있는 방법을 제시합니다.



### ATLAS: Improving Lay Summarisation with Attribute-based Contro (https://arxiv.org/abs/2406.05625)
- **What's New**: 이 연구는 과학 기사를 이해하기 쉬운 요약으로 변환하는 '레이 요약(lay summarisation)'에서 새로운 접근 방식을 제안합니다. ATLAS (ATtribute-controlled LAy Summarization)는 요약의 '레이성(layness)'을 조절할 수 있는 다양한 속성을 활용하여, 서로 다른 수준의 전문성을 가진 독자들의 니즈를 충족시키도록 설계되었습니다. 이는 과거 연구에서 제공하지 못한 세분화된 제어 가능성을 제공합니다.

- **Technical Details**: ATLAS는 과학적 요약을 생성하기 위해 BART-base 모델을 사용하며, 다음 네 가지 속성을 통해 요약의 특성을 제어합니다: 1) 요약 길이, 2) 읽기 쉬운 수준 (Flesh-Kincaid Grade Level), 3) 배경 정보 포함 비율, 4) 내용 단어의 평균 엔트로피. 각 속성 값은 훈련 데이터셋의 범위에 따라 10개의 고정 폭 빈으로 구분되며, 이에 따라 합성기에 입력될 제어 토큰이 생성됩니다. 모델 훈련 시에는 참 속성 값이 사용되고, 테스트 시에는 훈련 세트에서 관측된 가장 일반적인 빈 값이 사용됩니다.

- **Performance Highlights**: ATLAS는 eLife 및 PLOS 바이오 의학 레이 요약 데이터셋에서 최신 기준 모델보다 우수한 성능을 보였습니다. 자동 및 인간 평가 모두에서 높은 성능을 기록하였고, 속성 제어가 실제로 성능에 긍정적인 영향을 미친다는 추가 분석 결과도 확인되었습니다. ROUGE 및 BERTScore를 비롯한 다양한 자동 평가 지표와 Dale-Chall Readability Score를 사용하여 평가되었습니다.



### Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives (https://arxiv.org/abs/2406.05615)
Comments:
          Accepted at ACL 2024 (Findings)

- **What's New**: 이 논문은 비디오-언어(video-language) 이해 시스템에 대한 종합적인 리뷰를 제공하며, 이러한 시스템의 주요 과제와 관련된 도전 과제를 강조합니다. 저자들은 모델 아키텍처, 모델 훈련, 데이터 관점에서 현재 방법론을 요약하고, 각 방법의 성능을 비교하며 향후 연구 방향을 논의합니다.

- **Technical Details**: 논문에서는 비디오-언어 이해 과제를 수행하는 데이터셋을 소개하며, 주요 데이터셋을 다운스트림(downstream)과 프리트레이닝(pre-training) 데이터셋으로 분류합니다. 특히 최근의 다운스트림 데이터셋은 추론 능력 평가와 같은 새로운 기술적 도전과제를 제시하고 있습니다.

- **Performance Highlights**: 다양한 비디오-언어 이해 모델들의 성능을 비교하고, 각 모델이 직면한 도전과제를 논의합니다. 또한, 프리트레이닝과 다운스트림 비디오-언어 이해 모델 간의 성능 차이를 분석합니다.



### GrowOVER: How Can LLMs Adapt to Growing Real-World Knowledge? (https://arxiv.org/abs/2406.05606)
Comments:
          ACL 2024 Main

- **What's New**: GrowOVER-QA와 GrowOVER-Dialogue라는 두 가지 새로운 오픈 도메인 QA와 대화 벤치마크를 소개합니다. 이들은 지식의 지속적인 변화를 반영한 동적인 벤치마크로, 지식이 진화함에 따라 지속적으로 업데이트됩니다.

- **Technical Details**: 기존의 지식 기반 데이터셋이 빠르게 구식이 되는 문제를 해결하기 위해 GrowOVER는 정답과 함께 증거 텍스트를 제공하여 리트리버(retriever)와 생성기(generator)를 평가합니다. 또한, 회수 증강(retrieval-augmented) 언어 모델의 성능을 개선하기 위해 언어 모델이 스스로의 답변을 평가하고 필요 시 다시 회수하는, 새로운 'retrieval-interactive language model (RiLM)' 프레임워크를 도입하였습니다.

- **Performance Highlights**: 대규모 사전 훈련이 불필요한 상태에서 RiLM 프레임워크를 사용함으로써 기존 방법들보다 월등한 성능을 보였습니다. 특히, 리트리버와 생성기의 구성 요소를 정확히 측정하고, 에러 원인을 식별할 수 있는 능력이 향상되었습니다.



### CERET: Cost-Effective Extrinsic Refinement for Text Generation (https://arxiv.org/abs/2406.05588)
Comments:
          The source code and data samples are released at this https URL

- **What's New**: CERET는 텍스트 생성 품질을 개선하기 위해 새로운 접근 방식을 제안했습니다. 이는 기존의 LLM Self-rerank 방법보다 9.4% 낮은 지연 시간으로 더 비용 효율적으로 성과를 냅니다. CERET는 요약과 질문 응답 작업에서 기존의 Self-consistency와 Self-rerank 기준치보다 성능이 우수하다고 실험적으로 입증되었습니다.

- **Technical Details**: CERET는 세 가지 주요 스코어링 방식(semantic stability scoring, entailment scoring, inter-sample uncertainty scoring)을 도입해 텍스트 생성을 개선합니다. LLM에서 다양한 후보들을 생성한 후, 각 후보에 대해 개별 스코어를 계산하고, 이를 기반으로 선형 가중치를 적용한 최종 신뢰도 스코어를 산출하여 최적의 예측을 선택합니다. 이 방법은 클러스터링 없이 유사한 출력을 평가하며, RoBERTa와 같은 사전 훈련된 언어 모델을 활용합니다. entailment scoring은 자연어 추론(NLI) 모델을 이용해 후보 간의 논리적 연결성을 평가하고, inter-sample uncertainty scoring은 서로 다른 입력에 대한 유사한 출력을 억제하는 식으로 신뢰도를 측정합니다.

- **Performance Highlights**: CERET는 요약 작업에서 Rouge-1 지표 기준 약 1.6% 성능 향상을, 질문 응답 작업에서 hit rate 기준 약 3.5% 성능 향상을 보였습니다. 또한, 기존 LLM Self-rerank 방법에 비해 9.4%의 지연 시간만 필요로 하며, 더 비용 효과적입니다.



### Creativity Has Left the Chat: The Price of Debiasing Language Models (https://arxiv.org/abs/2406.05587)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 창의성에 대한 새로운 관점을 조사합니다. 특히, Human Feedback 기반 강화 학습(Reinforcement Learning from Human Feedback, RLHF)이 모델의 창의성에 미치는 영향을 탐구했습니다. Llama-2 시리즈 모델을 중심으로 실험을 실시하여, RLHF가 적용된 모델들이 낮은 엔트로피(entropy)를 보이며, 임베딩 공간(embedding space)에서 독특한 클러스터를 형성하고, 'attractor states'로 수렴하는 경향을 확인했습니다.

- **Technical Details**: 세 가지 실험을 통해 LLMs의 창의성에 대한 RLHF의 영향을 분석했습니다. 실험 결과, 정렬된(aligned) 모델들은 토큰 예측에서 낮은 엔트로피를 보였으며, 임베딩 공간에서 뚜렷한 클러스터를 형성했습니다. 이로 인해 출력 결과의 다양성(output diversity)이 제한되는 'attractor states' 상태로 수렴하는 경향이 나타났습니다.

- **Performance Highlights**: 연구 결과는 텍스트 작문(copywriting), 광고 제작(ad creation), 고객 페르소나 생성(customer persona generation) 등 창의적인 작업을 위해 LLMs를 사용하는 마케터들에게 중요한 시사점을 제공합니다. 일관성과 창의성 간의 트레이드오프(trade-off)를 고려해 적절한 모델을 선택하는 것이 필수적입니다. 또한, 본 연구는 기본 모델의 창의적 잠재력을 극대화하기 위한 프롬프트 엔지니어링(prompt engineering)의 중요성도 논의했습니다.



### Do LLMs Recognize me, When I is not me: Assessment of LLMs Understanding of Turkish Indexical Pronouns in Indexical Shift Contexts (https://arxiv.org/abs/2406.05569)
- **What's New**: 이 연구는 터키어에서의 지시적 변화(Indexical Shift) 문제를 집중적으로 분석한 첫 번째 연구로, 이를 위해 특별히 설계된 터키어 데이터셋을 공개했습니다. 지시적 변화 문제는 고자원 언어(English)에서는 나타나지 않는 문법적 도전 과제로, 이번 연구에서는 이 문제를 평가하기 위해 156개의 다지선다형 질문으로 구성된 터키어 데이터셋을 설계했습니다.

- **Technical Details**: 연구진은 최신 다국어 대형 언어 모델(LLMs)인 GPT-4, GPT-3.5, Cohere-AYA, Trendyol-LLM, Turkcell-LLM을 사용하여 터키어 데이터셋을 평가했습니다. 이번 연구는 몇 샷 학습(few-shot setting) 환경에서 지시적 변화 문제를 해결하기 위해 여러 다국어 LLM의 성능을 검증했습니다. 각 샘플은 필요한 언어학적 세부사항과 함께 주어졌으며, 데이터셋과 코드는 온라인에서 공개되었습니다.

- **Performance Highlights**: 분석 결과, GPT-4와 같은 최첨단 모델조차도 터키어의 지시적 변화와 같은 문법적 미묘함을 제대로 이해하지 못해 중간 수준의 성능을 보였습니다. 이 결과는 저자들이 제시한 것처럼 저자원 언어의 문법적 도전에 대한 특별한 연구의 필요성을 강조합니다.



### ThatiAR: Subjectivity Detection in Arabic News Sentences (https://arxiv.org/abs/2406.05559)
Comments:
          Subjectivity, Sentiment, Disinformation, Misinformation, Fake news, LLMs, Transformers, Instruction Dataset

- **What's New**: 이 연구는 아랍어 주관성(Subjectivity) 검출을 위한 최초의 대규모 데이터셋을 발표합니다. 기존 연구 대부분이 영어 및 다른 자원 풍부 언어에 집중된 것과는 달리, 이번 연구는 약 3.6K 수동 주석 문장을 포함한 아랍어 데이터셋을 제공합니다. GPT-4 기반 설명을 포함했으며, 데이터셋과 리소스를 커뮤니티에 공개할 계획입니다.

- **Technical Details**: 데이터 수집과 주석 과정은 네 단계로 진행되었습니다. 기존 아랍어 팩트체킹 데이터셋인 AraFacts를 시작으로 여러 Python 라이브러리를 사용해 뉴스 기사를 구문 분석한 후, 구문 분석된 기사를 일련의 도구를 통해 문장 단위로 분할했습니다. 또한, PLMs(Pre-trained Language Models)와 LLMs(Large Language Models)의 포괄적인 벤치마크 결과를 포함합니다. 주석자들이 정치적, 문화적, 종교적 배경에 따라 강하게 영향을 받았다는 점이 강조되었습니다.

- **Performance Highlights**: 실험 결과, 문맥 학습(In-context learning)을 통한 LLMs이 더 나은 성능을 제공하는 것으로 나타났습니다. 데이터셋에는 신뢰할 수 있는 출처로부터 제공된 뉴스 문장들이 주석되고 필터링되었습니다. 주석 지침은 아랍어와 영어로 제공되어 LLM 기반의 파인튜닝을 지원합니다.



### Generalist Multimodal AI: A Review of Architectures, Challenges and Opportunities (https://arxiv.org/abs/2406.05496)
Comments:
          25 pages, 3 figures, 5 tables

- **What's New**: 최근 멀티모달 모델(Multimodal Models) 연구가 활발히 진행되고 있습니다. 이 논문은 멀티모달 모델의 새로운 설계를 소개하며, 특히 텍스트와 비전(Text and Vision)을 넘어선 다양한 모달리티(예: 영상, 센서, 시계열, 그래프 등)을 다루는 '일반 멀티모달 모델(Generalist Multimodal Models)'을 집중 조명합니다. 이러한 모델들은 여러 모달리티와 작업에서 하나의 모델로 운영될 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 이 논문은 '통합 가능성(Unifiability)', '모듈성(Modularity)', '적응성(Adaptability)'와 같은 핵심 요소들을 포함한 새로운 분류 체계를 도입합니다. 이 요소들은 GMMs의 광범위한 채택과 적용에 필수적입니다. 또한, 데이터를 다루는 방식, 아키텍처 설계, 훈련 전략 등을 바탕으로 기존 작업을 분류하고 그 장단점을 분석합니다.

- **Performance Highlights**: 기존의 멀티모달 모델들은 주로 텍스트와 이미지 모달리티에 집중되어 왔으나, 이 논문은 더 넓은 모달리티를 다루기 위한 새로운 설계 요소들을 강조하며, 멀티모달 학습의 여러 측면에서 GMMs의 능력 향상을 보여줍니다. 이러한 모델들은 대규모 사전 훈련과 특수 미세튜닝(High-level Finetuning)을 통해 성능을 극대화합니다.



### Investigating and Addressing Hallucinations of LLMs in Tasks Involving Negation (https://arxiv.org/abs/2406.05494)
- **What's New**: 최신 연구에서는 대형 언어 모델(LLMs)의 '환각(hallucination)' 문제에 대해 다룰 때, '부정(negation)'의 영향이 충분히 탐구되지 않았다는 점을 지적합니다. 이 연구는 부정을 포함한 네 가지 과제에서 LLM 환각을 조사하며, 모델들이 부정 존재 시 상당한 오류를 범한다는 것을 보여줍니다.

- **Technical Details**: 구체적으로, 연구에서는 '거짓 전제 완성(false premise completion)', '제한된 사실 생성(constrained fact generation)', '다지선다형 질문 응답(multiple choice question answering)', 및 '사실 생성(fact generation)'의 네 가지 과제에서 부정이 LLM 환각에 미치는 영향을 연구합니다. 사용된 모델은 LLaMA-2-chat, Vicuna, Orca-2로, 이들은 모두 부정을 포함한 과제에서 의미 있는 환각을 보였습니다.

- **Performance Highlights**: 연구에서는 LLaMA-2-chat, Vicuna-v1.5, Orca-2 모델이 평균적으로 FPC에서 63.77%, CFG에서 72.33%, MCQA에서 36.6%, FG에서 62.59%의 환각률을 보인다고 보고했습니다. 이러한 결과는 부정을 처리하는 데 있어서 기존 LLM의 한계를 크게 드러냅니다. 이를 해결하기 위해 다양한 완화 전략들도 연구되었으며, '주의 지침(cautionary instruction)'과 '맥락 내 예제(in-context exemplars)'를 제공하는 방법이 가장 효과적이었으나 여전히 개선의 여지가 남아있음을 보여줍니다.



### Fighting Against the Repetitive Training and Sample Dependency Problem in Few-shot Named Entity Recognition (https://arxiv.org/abs/2406.05460)
Comments:
          ieee access: this https URL

- **What's New**: 본 논문에서는 소수의 라벨링 예제로 명명된 엔터티를 인식하는 Few-shot Named Entity Recognition (NER) 시스템을 제안합니다. 이 새로운 파이프라인은 Wikipedia 데이터를 사전 학습한 span detector(스팬 감지기)를 제공하여, 기본 기능에 대한 반복 훈련을 줄이고, 더 큰 언어 모델(LLM)을 사용해 신뢰할 수 있는 엔터티 타입 참조체를 설정합니다. 해당 모델은 기존의 베이스라인과 비교해 더 적은 훈련 단계와 인간 라벨링 데이터로 우수한 성능을 보입니다.

- **Technical Details**: 본 연구는 스팬 감지기를 Wikipedia 웹 데이터에서 사전 훈련하여, 새로운 파이프라인 스팬 감지기의 초기화 도구로 사용함으로써 기본 스팬 기능의 반복 훈련을 줄이고, 다양한 도메인에 적응 시 더 빠르게 수렴하게 합니다. 이를 통해 데이터셋 주석에 필요한 기계 자원과 노력을 절약할 수 있습니다. 또한, 머신 일반 지식을 활용해 엔티티 타입 참조체를 만들어 샘플 의존성 문제를 해결합니다. 이는 large language models (LLMs), 예를 들어 GPT-3.5, 의 지식을 벡터 공간에 엔터티 타입 참조체로 인코딩하여 유사성 검색에 사용됩니다.

- **Performance Highlights**: 제안된 모델은 다양한 데이터셋과의 실험을 통해 뛰어난 성능을 입증했습니다. 특히, 미세-그레인드(NER) 시나리오에서는, ChatGPT를 포함한 강력한 베이스라인보다도 뛰어난 성능을 보였습니다. 연구 결과물은 공개됐으며, 이를 통해 연구 및 산업 분야에서 컴퓨팅 자원 절약의 실질적인 도움을 줄 수 있습니다.



### Design of reliable technology valuation model with calibrated machine learning of patent indicators (https://arxiv.org/abs/2406.05446)
- **What's New**: 기계 학습(ML)이 특허 가치 평가에 높은 정확도로 기여하고 있지만, 모델 예측의 신뢰성에 대한 검증이 부족하여 전문가는 여전히 완전한 신뢰를 하지 못하고 있습니다. 이를 해결하기 위해, 우리는 신뢰성 있는 기술 평가를 위한 분석 프레임워크를 제안합니다. 이 프레임워크는 보정된 ML 모델을 사용하여 모델 예측의 견고한 신뢰 수준을 제공합니다.

- **Technical Details**: 제안된 방법에서는 다양한 기술 특성을 나타내는 양적 특허 지표를 입력 데이터로 추출하며, 특허 유지 기간을 기술 가치를 나타내는 대리 변수로 사용합니다. 다양한 ML 모델을 개발하여 특허 지표와 기술 가치 간의 비선형 관계를 포착하며, 보정 오류, 매튜스 상관 계수 및 F1-스코어를 비교하는 파레토 전면 맵(Pareto-front map)을 통해 모델의 신뢰성 및 정확성을 평가합니다. 최적의 모델 식별 후, SHapley Additive exPlanation(SHAP)을 이용하여 신뢰도 빈(bin)별로 가장 중요한 입력 피처를 확인합니다.

- **Performance Highlights**: 제안된 접근 방식은 신뢰성과 정확성을 갖춘 ML 기반 기술 평가 모델을 개발하기 위한 실용적인 지침을 제공하며, 이는 학계와 산업계에 중요한 영향을 미칠 수 있음을 사례 연구를 통해 확인하였습니다.



### MaTableGPT: GPT-based Table Data Extractor from Materials Science Literatur (https://arxiv.org/abs/2406.05431)
- **What's New**: 새롭게 발표된 MaTableGPT는 재료 과학 논문에서 표 데이터를 추출하는 GPT 기반 툴입니다. 이는 특히 물 분해 촉매 문헌에서 매우 중요한 데이터 추출 과정을 혁신적으로 개선합니다.

- **Technical Details**: MaTableGPT는 표 데이터 표현(table data representation) 및 표 분할(table splitting) 전략을 통해 GPT의 이해력을 강화하고, 후속 질문을 통해 허상 정보를 필터링하는 기능을 갖추고 있습니다. 이 툴은 거대한 물 분해 촉매 문헌에 적용되어 효율적으로 데이터 추출을 수행합니다.

- **Performance Highlights**: MaTableGPT는 최대 96.8%의 추출 정확도(total F1 score)를 달성했습니다. 구체적으로는, 제로샷(zero-shot), 퓨샷(few-shot), 파인튜닝(fine-tuning) 학습 방법을 종합 평가했으며, 퓨샷 학습(few-shot learning) 방식이 높은 정확도(total F1 score > 95%)와 낮은 코스트(GPT 사용 비용 5.97 달러, 라벨링 비용 10 I/O 파생 예제)로 가장 균형 잡힌 솔루션으로 나타났습니다.



### Recent advancements in computational morphology : A comprehensive survey (https://arxiv.org/abs/2406.05424)
- **What's New**: 이번 논문은 역사가 오래된 전통적인 방법론부터 최신 딥러닝 기반 방식까지, 단어 형태 분석 도구를 개발하기 위한 다양한 방법론을 포괄적으로 조사합니다. 이는 다양한 언어에 걸쳐 형태 분석 작업을 위한 기존 데이터셋도 검토합니다.

- **Technical Details**: 본문에서는 형태소 경계 검출, 어간화(lemmatization), 형태소 특징 태깅(tagging), 형태 변형(reinflection) 등의 작업을 다룹니다. 형태 분석과 생성의 두 단계로 나뉘며, 전통적인 규칙 기반 방법부터 기계 학습, 통계적 방법, 딥러닝 아키텍처까지 다양한 기술을 살펴봅니다.

- **Performance Highlights**: 딥러닝 모델의 효과를 전통적인 모델과 비교 연구하여, 형태적으로 풍부한 언어에 대해 고품질 형태소 분석기가 중요하고 도움이 된다는 결론을 제시합니다. 특히, 통계적 서브워드 토큰화보다 형태소 분할이 더 나은 성능을 보인다고 보고되었습니다.



### Deconstructing The Ethics of Large Language Models from Long-standing Issues to New-emerging Dilemmas (https://arxiv.org/abs/2406.05392)
- **What's New**: 최근 몇 년간, Large Language Models (LLMs)의 발전이 눈에 띄게 성장했습니다. 이 논문은 LLMs의 발전과 함께 대두된 여러 윤리적 문제들을 종합적으로 조사합니다. 기존 문제들인 저작권 침해, 시스템적 편향, 데이터 프라이버시 뿐만 아니라, 진실성(truthfulness)과 사회적 규범과 같은 새롭게 등장한 문제들도 다룹니다. LLMs의 윤리적 표준과 사회적 가치를 통합하여 책임 있고 윤리적으로 조율된 언어 모델의 개발을 안내하는 것이 이 논문의 주요 목적입니다.

- **Technical Details**: LLMs는 자연어 생성(natural language generation), 질문 응답(question answering), 복잡한 추론 작업(complex reasoning tasks)에서 뛰어난 성능을 보이고 있습니다. 하지만 이러한 모델들은 개인정보보호(privacy), 저작권(copyright), 편향(bias)과 같은 다양한 윤리적 문제를 일으킵니다. 예를 들어, 데이터의 비도덕적 사용을 방지하기 위해 협동적 정렬(alignment) 기술이 개발되었습니다. 또한, '환각(hallucination)'이라고 불리는, 사실적 기반이 부족한 컨텐츠 생성 문제도 존재합니다.

- **Performance Highlights**: 논문에서는 기존 윤리 문제와 새로운 윤리 문제를 두 가지 주요 범주로 나누어 다루고 있습니다. 기존 문제(데이터 프라이버시, 저작권, 공정성)에 더해 새로운 문제(진실성, 사회적 규범)와 이를 해결하기 위한 법적 및 규제적 준수 요구사항도 논의합니다. 이를 통해 독자들이 LLMs의 윤리적 문제와 해당 문제를 해결하기 위한 기술 및 전략을 더 잘 이해할 수 있도록 돕고자 합니다.



### Planning Like Human: A Dual-process Framework for Dialogue Planning (https://arxiv.org/abs/2406.05374)
Comments:
          24 pages, 5 figures, ACL 2024 main conference

- **What's New**: DPDP(이중-프로세스 대화 계획, Dual-Process Dialogue Planning) 프레임워크는 심리학의 이중-프로세스 이론에서 영감을 받아 대화를 목표 지향적으로 이끌 수 있도록 설계된 새로운 방식을 소개합니다. 기존의 대화 계획 방법들이 복잡하거나 비효율적인 반면, DPDP는 직관적 정책 모델과 분석적 몬테 카를로 트리 탐색(Monte Carlo Tree Search, MCTS) 메커니즘을 결합하여 효율성과 전략적 깊이를 모두 갖춘 대화 생성을 가능하게 합니다.

- **Technical Details**: DPDP는 심리학의 이중-프로세스 이론(빠르고 직관적인 System 1과 느리고 분석적인 System 2)을 구현하여, 익숙한 상황에서는 신속한 정책 모델이, 복잡하거나 새로운 상황에서는 MCTS 기반 플래너가 작동합니다. 두 단계의 학습 방식을 사용하여 정책 모델의 성능을 향상시키며, 첫 단계에서는 오프라인 강화 학습을 통해 베이스 모델을 훈련시키고, 두 번째 단계에서는 MCTS 시뮬레이션을 통해 정책 모델을 향상시킵니다.

- **Performance Highlights**: 다양한 대화 과제에 대한 실험적 평가 결과, DPDP는 기존 방법들보다 높은 품질의 대화와 운영 효율성을 달성하는 데 있어 우수성을 입증했습니다. 이는 특히 MCTS 기반 방법들과 비교했을 때 뛰어난 성능을 보여주는 것으로 평가되었습니다.



### VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers (https://arxiv.org/abs/2406.05370)
- **What's New**: 이번 논문에서는 VALL-E 2를 소개합니다. 이는 zero-shot 텍스트-음성 합성(TTS)에서 처음으로 인간과 동등한 수준의 성능을 달성한 최신 신경 코덱 언어 모델입니다. VALL-E의 후속작으로서, VALL-E 2는 반복 인식 샘플링(Repetition Aware Sampling)과 그룹화 코드 모델링(Grouped Code Modeling)의 두 가지 주요 개선점을 도입하였습니다.

- **Technical Details**: VALL-E 2는 두 가지 주요 기술적 개선점을 포함합니다. 첫째, 반복 인식 샘플링은 반복되는 토큰을 고려하여 초기의 핵심 샘플링(nucleus sampling) 과정을 개선하여 디코딩 안정성을 높이고 무한 루프 문제를 방지합니다. 둘째, 그룹화 코드 모델링은 코덱 코드를 그룹화하여 시퀀스 길이를 단축함으로써 추론 속도를 향상시키고 긴 시퀀스 모델링 문제를 해결합니다.

- **Performance Highlights**: LibriSpeech와 VCTK 데이터셋에서의 실험 결과, VALL-E 2는 음성 강인성, 자연스러움, 음성 유사성에서 이전 시스템을 능가하며, 해당 벤치마크에서 처음으로 인간과 동등한 성능을 달성했습니다. VALL-E 2는 복잡하거나 반복적인 문장에서도 일관되게 높은 품질의 음성을 합성할 수 있습니다.



### Venn Diagram Prompting : Accelerating Comprehension with Scaffolding Effec (https://arxiv.org/abs/2406.05369)
Comments:
          Preprint. 10 pages, Accepted in 2024 the 6th World Symposium on Artificial Intelligence (WSAI 2024)

- **What's New**: 이번 연구에서는 복잡하고 다양한 길이의 문맥에서 정보를 결합하고 종합하는 새로운 프롬프팅 기술인 Venn Diagram (VD) Prompting을 소개합니다. 이는 특히 지식 집약적인 질문-응답 작업에서 대규모 언어 모델(LLM)이 여러 단계를 거치지 않고 단일 호출로 일관된 답변을 생성할 수 있도록 합니다.

- **Technical Details**: VD 프롬프팅은 주어진 문맥에서 정보를 먼저 체계적으로 정리한 후 답변을 생성하는 접근 방식을 채택합니다. 이는 LLM의 위치 편향을 줄이고 입력 정보의 순서에 민감하지 않게 만들어 보다 일관된 응답을 생성할 수 있습니다. 기존의 복잡한 논리나 다단계 방식을 단순한 하나의 LLM 호출로 대체하는 것을 목표로 합니다.

- **Performance Highlights**: 네 개의 공개 벤치마크 질문-응답 데이터셋에서 VD 프롬프팅은 기존의 최적화된 지침 프롬프트와 대등하거나 그 이상의 성능을 지속적으로 보여주었습니다. 이는 다양한 출처에서 검색한 정보를 종합하여 일관되고 정확한 답변을 생성한다는 점에서 우수함을 입증합니다.



### CaLM: Contrasting Large and Small Language Models to Verify Grounded Generation (https://arxiv.org/abs/2406.05365)
Comments:
          Paper accepted at ACL 2024 as a finding paper. Work done while the first author was a student researcher at Google Cloud AI Research. Correspondence to: I-Hung Hsu <ihunghsu@usc.edu>, Chen-Yu Lee <chenyulee@google.com>

- **What's New**: 이 연구에서는 큰 언어 모델(LLM)과 작은 언어 모델(SLM)의 상호작용을 통한 새로운 검증 프레임워크인 CaLM을 소개합니다. CaLM은 각 답변을 검증하기 위해 상호작용하는 큰 LLM과 작은 SLM의 장점을 결합합니다. 이를 통해 모델이 신뢰할 수 있는 출처를 인용하면서 더 신뢰성 있고 책임감 있는 응답을 생성할 수 있도록 합니다. 이 프레임워크는 모델의 파인 튜닝 없이도 성능 향상을 이끌어냅니다.

- **Technical Details**: CaLM은 큰 LLM이 광범위한 정보 식별에 뛰어나지만 내부 기억에 과도하게 의존할 수 있는 점을 활용합니다. 한편, 작은 SLM은 검색된 정보의 처리에는 능하지만, 대규모 컬렉션에서 이를 식별하는 데는 한계가 있습니다. 이러한 특성을 기반으로, CaLM은 큰 LLM의 응답을 작은 SLM의 출력과 교차 비교하여 검증합니다. 만약 응답에 불일치가 발생하면 피드백 루프를 통해 반복적으로 정제합니다. 또한, CaLM은 모델 파인 튜닝을 필요로 하지 않고, 단지 출처 문서들로부터의 정보만을 활용하는 점에서 실용적입니다.

- **Performance Highlights**: QAMPARI, ASQA 및 ELI5 세 가지 오픈 도메인 질문 답변 데이터셋에 대한 실험 결과, CaLM은 기존의 최첨단 방법들을 평균 1.5%에서 7%까지 성능을 초과했습니다. 특히, 검색 시스템이 덜 강력한 상황에서도 다른 기준 모델들이 어려움을 겪는 반면, CaLM은 여전히 강력한 성능을 유지하였습니다.



### Write Summary Step-by-Step: A Pilot Study of Stepwise Summarization (https://arxiv.org/abs/2406.05361)
Comments:
          10 pages, 4 figures, published in TASLP

- **Introduction**: 오늘날 소셜 텍스트 스트림(예: 뉴스 이벤트, 트윗 등)은 실시간으로 진화하며 초록적인 요약을 통해 정보를 신속하고 정확하게 전파할 수 있습니다. 하지만 기존의 대부분의 요약 모델은 전체 문서를 한 번에 처리하는 방식으로, 실생활의 요구를 충족시키기 어렵습니다. 이에 본 논문은 새로운 문서가 추가될 때마다 최신의 전체 요약을 형성하는 새로운 첨부 요약을 생성하는 '단계별 요약(Stepwise Summarization)' 작업을 제안합니다.

- **What's New**: 본 논문에서는 새로운 문서가 추가될 때마다 요약을 갱신하는 '단계별 요약(Stepwise Summarization)' 작업을 처음 제안합니다. 이를 위해, 단계별 요약 생성기(SSG)라는 적대적 학습 모델을 설계했습니다. SSG는 이전 요약을 고려하여 새로운 문서를 선택적으로 처리하고, 문서와 이전 요약을 모두 고려한 새로운 요약을 생성합니다. 최종적으로, 새로 생성된 요약이 이전 요약과 일관성이 있는지 판단하기 위해 convolutional 기반의 판별자(discriminator)를 사용합니다.

- **Technical Details**: SSG는 첫째, 예전 요약의 가이드를 바탕으로 새로운 문서를 선택적으로 처리합니다. 둘째, 디코더를 사용해 다듬어진 문서와 요약 표현을 기반으로 새로운 요약을 생성합니다. 마지막으로, 적대적 학습 방식으로 훈련되며, convolutional 판별자가 생성된 요약의 일관성을 평가합니다. 이를 위해 대규모의 단계별 요약 데이터셋을 공개하고 실험을 통해 각 모듈의 효과를 검증했습니다.

- **Performance Highlights**: SSG는 ROUGE 지표와 인간 평가 기준 모두에서 기존 최첨단 요약 모델들을 능가하는 성능을 보였습니다. 또한, 대규모의 단계별 요약 데이터셋을 공개해 실험을 진행했으며, 각 모듈의 유효성을 입증하는 연구도 포함했습니다.



### Flexible and Adaptable Summarization via Expertise Separation (https://arxiv.org/abs/2406.05360)
Comments:
          10 pages, 7 figures, published in SIGIR 2024

- **What's New**: MoeSumm는 유연성과 적응성을 동시에 갖춘 새로운 요약 모델입니다. 기존의 대형 언어 모델(LLM)과 달리, 파라미터 효율적인 접근 방식을 통해 도메인별 요약 능력을 구분하여 적용합니다.

- **Technical Details**: MoeSumm는 Mixture-of-Expert 구조를 채택하여 주 전문가(main expert)와 부 전문가(deputy experts)를 활용합니다. 주 전문가는 일반적인 요약 기능을 담당하고, 부 전문가는 특정 요약 작업에 맞추어 선택적으로 협력합니다. 이를 위해 최대 마진 손실(max-margin loss)을 도입하여 이들 능력을 명확히 구분합니다.

- **Performance Highlights**: 11개의 데이터셋에 대한 실험 결과, MoeSumm는 최근의 베이스라인 모델들과 대형 언어 모델(LLM)들에 비해 우수한 성능을 보여주었습니다. 또한, 다양한 도메인에서 효과적으로 요약을 수행하는 능력을 입증했습니다.



### Toward Reliable Ad-hoc Scientific Information Extraction: A Case Study on Two Materials Datasets (https://arxiv.org/abs/2406.05348)
- **What's New**: 이번 연구는 GPT-4가 과학 문헌에서 adhoc 스키마 기반 정보 추출을 수행할 수 있는 능력을 탐구합니다. 기존의 수작업으로 추출된 두 가지 재료 과학 데이터셋을 재현할 수 있는지 평가하며, 모델이 희망하는 정보를 정확히 추출하는 데 어려움을 겪는 부분을 구체적으로 분석하였습니다.

- **Technical Details**: 연구는 두 가지 전문가가 수작업으로 추출한 재료 특성 데이터셋을 사용했습니다. 하나는 다중-주요 원소 합금(MPEAs)에 관한 것이고, 다른 하나는 실리케이트 녹아내림의 요소 확산에 관한 것입니다. 모델의 성능을 평가하기 위해 재료 과학자들이 오류 분석을 수행했습니다. GPT-4는 스키마에 따라 데이터를 추출하고, 내러티브나 기존의 표에서 잘 작동했지만, 그래프와 PDF 파싱 이슈에서 많은 오류가 발생했습니다.

- **Performance Highlights**: GPT-4는 내러티브나 표 형식에서 정보를 상당히 잘 추출하는 능력을 보였지만, 그래프와 PDF 파싱 문제에서 상당한 오류를 보였습니다. 추가적으로, 비표준 표 형식, 추출된 값의 후처리 필요성, 그리고 향상된 프롬프트 엔지니어링이 요구되는 진정한 읽기 이해 오류도 주요 오류 원인이었습니다.



### MemeGuard: An LLM and VLM-based Framework for Advancing Content Moderation via Meme Intervention (https://arxiv.org/abs/2406.05344)
- **What's New**: 이번 연구는 멀티모달 콘텐츠의 유해성 모니터링 시스템의 한계를 해결하고자 	extit{MemeGuard}라는 포괄적인 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델 (Large Language Models, LLMs)과 비주얼 언어 모델 (Visual Language Models, VLMs)을 활용하여 밈의 해석과 개입을 수행합니다.

- **Technical Details**: MemeGuard는 특별히 튜닝된 VLM인 	extit{VLMeme}를 이용하여 밈을 해석하고, 멀티모달 지식 선택 및 랭킹 메커니즘 (	extit{MKS})을 통해 관련 지식을 선별합니다. 이 정보는 일반 목적의 LLM을 통해 맥락에 맞는 개입을 생성하는 데 사용됩니다. 또한 ICMM (Intervening Cyberbullying in Multimodal Memes) 데이터셋을 제작하여 MemeGuard의 성능을 검증했습니다.

- **Performance Highlights**: MemeGuard는 독창적인 VLMeme를 통해 밈의 복잡한 내용을 깊이 이해하고, 분류된 지식을 바탕으로 적절한 개입을 생성하여 사이버괴롭힘 밈에 효과적인 대응을 보였습니다. ICMM 데이터셋을 이용한 테스트 결과, MemeGuard가 독성과 편견을 포함한 밈에 대해 맥락에 맞는 높은 품질의 개입을 생성하는 데 능숙함을 입증했습니다.



### Hidden Question Representations Tell Non-Factuality Within and Across Large Language Models (https://arxiv.org/abs/2406.05328)
- **What's New**: 이 연구는 대형 언어 모델(LLM)의 비사실적 응답을 예측하는 '비사실적 예측(NFP)'을 탐구하며, 'Factuality Lens (FacLens)'라는 가벼운 프로브를 사용해 질문의 히든 표현(hidden representations)에서 정보 추출을 시도합니다. 주목할 점은 다수의 LLM에서 동일한 패턴을 찾음으로써 교차-LLM NFP에서 전이 학습(transfer learning)의 가능성을 탐구한다는 것입니다.

- **Technical Details**: FacLens는 '질문-정렬된 전략(question-aligned strategy)'을 사용하여 미니 배치 기반 학습의 효율성을 보장합니다. 이는 다수의 LLM의 질문 히든 표현에서 'LLM이 알고 있는지' 여부를 파악하는 가벼운 프로브를 훈련시키는 방법을 사용합니다. 주목할만한 점은 다수의 LLM에서 이 동일한 패턴을 찾아내어 교차-LLM NFP의 가능성을 모색했다는 것입니다.

- **Performance Highlights**: 질문의 히든 표현만을 사용하여 비사실적 응답을 식별하는 FacLens는 효율적인 훈련 및 적용을 가능하게 하며, 여러 LLM과 여러 사실성 질문 응답 데이터셋을 사용한 포괄적인 실험을 통해 좋은 성능을 보여줍니다. 전이 학습 및 질문-정렬된 전략을 통해 소스 및 타겟 도메인 간의 분포 거리 추정을 개선하고 미니 배치 교육의 효율성을 높였습니다.



### Advancing Semantic Textual Similarity Modeling: A Regression Framework with Translated ReLU and Smooth K2 Loss (https://arxiv.org/abs/2406.05326)
Comments:
          Work in Progress

- **What's New**: 이 연구는 텍스트 유사성(Semantic Textual Similarity, STS)을 위한 혁신적인 회귀(regression) 프레임워크를 제안하며, 두 가지 간단하지만 효과적인 손실 함수(loss functions)인 Translated ReLU와 Smooth K2 Loss를 소개합니다. 이 방법은 STS 벤치마크에서 기존의 최첨단 성능을 뛰어넘는 성과를 보여줍니다.

- **Technical Details**: 기존의 STS 연구에서 사용된 Sentence-BERT는 텍스트 쌍을 독립적으로 임베딩하여 분류(classification) 관점에서 접근합니다. 반면, 본 논문에서는 STS 작업을 다중 범주(multicategory) 문제로 전환하고 이를 회귀 문제로 재구성하는 새로운 프레임워크를 제안합니다. 이를 위해 원래의 레이블을 순차적인 정수 배열로 매핑하고 출력 계층의 노드 수를 하나로 설정합니다. 또, 표준 L1 손실 및 MSE 손실 대신 제안된 Translated ReLU와 Smooth K2 Loss 함수가 사용됩니다.

- **Performance Highlights**: 제안된 방법은 STS12-16, STS-B, 그리고 SICK-R의 7개의 벤치마크에서 전통적인 분류 접근법을 능가하는 성능을 입증했습니다. 특히, STS-B와 SICK-R 데이터 셋을 사용하여 모델의 체크포인트를 업데이트한 결과, 상위 Spearman 상관관계를 보이며 기존의 대조 학습(contrastive learning) 방법보다 우수한 성과를 달성했습니다. 이 연구는 다중 범주 STS 작업에서 진보적인 관계를 포착하는 데 효과적임을 증명합니다.



### Teaching-Assistant-in-the-Loop: Improving Knowledge Distillation from Imperfect Teacher Models in Low-Budget Scenarios (https://arxiv.org/abs/2406.05322)
Comments:
          Accepted by ACL 2024 Findings

- **What's New**: 이 논문에서는 자원 제약 및 불완전한 대형 언어 모델(LLM)의 시나리오에서 샘플 효율성을 향상시키기 위해 세가지 신호 유형을 활용하는 세가지 구성 요소로 이루어진 프레임워크를 제안합니다. 제안된 프레임워크는 학생 모델(self-consistency), 보조 교사 모델(Teaching Assistant, TA) 및 교사 모델의 출력을 평가하는 TA 모델로 구성됩니다.

- **Technical Details**: 프레임워크는 세 가지 신호를 사용하여 LLM 지식 증류를 향상시킵니다. 첫 번째 신호는 학생 모델의 자기 일관성(self-consistency) 점수로 학생의 신뢰도를 이해하는 데 사용됩니다. 두 번째 신호는 TA 모델이 학생을 위해 생성하는 신호입니다. 세 번째 신호는 TA 모델이 교사 모델의 출력을 평가하여 학생 모델의 훈련 데이터로 포함할지 결정하는 것입니다. 제안된 두 단계의 훈련 방식은 먼저 데이터를 소량 사용하여 학생 모델을 사전 추정하고, 이후 나머지 데이터로 추가 훈련을 수행합니다.

- **Performance Highlights**: 다양한 복잡한 추론 작업에 대해 실험한 결과, 제안된 프레임워크는 기존의 세그먼테이션 없이 fine-tune한 경우보다 평균 20.79% 향상된 결과를 보였습니다. 이는 세가지 신호를 통합한 프레임워크가 학생 모델의 성능을 크게 개선할 수 있음을 시사합니다.



### Concept Formation and Alignment in Language Models: Bridging Statistical Patterns in Latent Space to Concept Taxonomy (https://arxiv.org/abs/2406.05315)
- **What's New**: 이 논문은 언어 모델(Language Models, LMs) 내 개념 형성과 정렬 개념을 탐구합니다. 저자들은 여러 언어 모델이 학습한 의미 표현에서 개념과 그 계층적 구성을 식별하는 메커니즘을 제안하며, 이를 통해 즉 유사한 어휘 항목들이 얼마나 잘 군집화되는지를 분석합니다. Glove와 같은 초기 모델부터 ALBERT, T5와 같은 transformer 기반 언어 모델까지 이 접근법을 확장합니다.

- **Technical Details**: 논문에서는 K-Nearest Neighbors, UMAP의 모호한 가중치 메커니즘(fuzzy weighting mechanism), 커뮤니티/클러스터 탐지 알고리즘을 기반으로 개념 추출 방법론을 제안합니다. 이 방법론은 텍스트 임베딩 레이어에 존재하는 의미 있는 계층적 구조에 의존하며, 이를 바탕으로 잠재적인 개념을 나타내는 상호 연결된 데이터 포인트 그룹을 식별합니다. 실험에서는 WordNet, Name Dataset, 국가-주-도시 데이터베이스 등을 외부 참조로 사용하여 형성된 클러스터를 평가했습니다.

- **Performance Highlights**: 실험 결과, transformer 기반 모델의 입력 임베딩 레이어에서 의미 메모리가 존재함을 시사하는 개념 형성이 관찰되었습니다. 또한, 이러한 모델들이 수정된 의미 메모리로도 여전히 효과적으로 추론할 수 있는지를 추가로 조사하여, 지식 표현과 추론 능력 간의 관계를 심층적으로 이해할 수 있는 기회를 제공합니다.



### DeviceBERT: Applied Transfer Learning With Targeted Annotations and Vocabulary Enrichment to Identify Medical Device and Component Terminology in FDA Recall Summaries (https://arxiv.org/abs/2406.05307)
- **What's New**: FDA 의료 기기 리콜 데이터셋에서 기기 정보를 추출하는 새로운 모델인 DeviceBERT를 제안합니다. 이 모델은 기존의 BioBERT 모델 개선하여 의료 기기 용어를 더 정확하게 인식하고 레이블링 할 수 있도록 설계되었습니다.

- **Technical Details**: DeviceBERT는 BioBERT의 토크나이저를 의료 기기 용어를 인식할 수 있도록 단어 집합을 확장하고, 정규화 및 데이터 전처리 과정을 통해 BioBERT의 도메인 특정 기능을 향상합니다. 이 접근법은 특히 품질이 낮은 주석 데이터와 오버피팅 문제를 해결하는 데 중점을 두었습니다. K-Fold 교차 검증 및 드롭아웃 기법을 사용하여 모델의 일반화 성능을 향상했습니다.

- **Performance Highlights**: DeviceBERT는 기존 BioBERT 모델보다 의료 기기 이름, 부품 번호 및 구성 요소 용어를 인식하는 데 13.72% 더 높은 성능을 보였습니다. 이를 통해 의료 기기 리콜 분석 시 더욱 신속하고 정확한 의사 결정을 지원할 수 있습니다.



### SuperPos-Prompt: Enhancing Soft Prompt Tuning of Language Models with Superposition of Multi Token Embeddings (https://arxiv.org/abs/2406.05279)
- **What's New**: Soft prompt tuning은 기존의 모델을 너무 많이 변경하지 않고도 높은 성능을 유지할 수 있어 점차 인기를 끌고 있습니다. 본 연구에서는 기존의 Residual Prompt tuning 방법보다 뛰어난 성능을 보이는 SuperPos-Prompt라는 새로운 재매개화(reparameterization) 기술을 제안합니다. 이 기법은 여러 개의 사전 훈련된 단어 임베딩(vocabulary embeddings)을 사용할 수 있도록 해 소프트 프롬프트의 학습을 향상시킵니다.

- **Technical Details**: SuperPos-Prompt는 여러 개의 토큰 임베딩을 사용하여 각 프롬프트 임베딩을 계산하는 방법입니다. 구체적으로, 여러 개의 고유한 토큰 임베딩을 선택하여 행렬을 구성하고, 이 행렬과 벡터의 곱으로 프롬프트 토큰을 최적화합니다. 이 과정에서 각 프롬프트 임베딩의 샘플된 토큰들이 동일하게 사용되며, 가중치 감쇠(weight decay)를 줄여 정보 손실을 방지합니다. 기존의 복잡한 접근 방식(예: IPT, ATTEMPT)과 달리, SuperPos-Prompt는 미리 훈련된 소프트 프롬프트나 오토인코더가 필요하지 않습니다.

- **Performance Highlights**: SuperPos-Prompt는 GLUE와 SuperGLUE 벤치마크에서 T5-Small 모델에서 평균 6.4점, T5-Base 모델에서 5.0점의 성능 향상을 보이며 Residual Prompt tuning보다 우수한 성능을 보였습니다. 또한, 빠른 수렴 속도를 나타내며, 경우에 따라 전체 fine-tuning 방법보다도 더 뛰어난 성능을 보이기도 했습니다. 드롭아웃(dropouts)을 제거하였을 때도 성능이 향상되는 것을 확인했습니다.



### Behavior Structformer: Learning Players Representations with Structured Tokenization (https://arxiv.org/abs/2406.05274)
- **What's New**: 새로운 논문에서는 Behavior Structformer를 소개합니다. 이 방법은 Transformer 기반 아키텍처에서 구조화된 토큰화를 통해 사용자 행동을 모델링합니다. 추적 이벤트를 밀집 토큰으로 변환하여 모델 학습 효율성과 효과성을 향상시킵니다. 이 접근법은 전통적인 표형 및 반구조화된 (semi-structured) 데이터 셋을 벤치마킹하고 비교한 결과, 행동 모델링에서 우수한 성능을 보였습니다.

- **Technical Details**: Behavior Structformer는 추적 데이터를 밀집 토큰으로 변환하여 모델 학습을 위한 데이터를 준비합니다. 이는 Transformer 기반 구조에서 우수한 성능을 발휘하도록 설계되었습니다. 도메인 지식과 연역적 편향을 통합하여 초기 토큰화 과정 중 데이터를 최적화합니다. 이 방법을 통해 비디오 게임 내 플레이어 행동 데이터를 모델링하고, 시간 순서대로 배열된 이벤트 시퀀스를 일련의 밀집 벡터로 표현하여 토큰으로 처리합니다.

- **Performance Highlights**: 논문은 30일 동안 100만 명의 모바일 게임 사용자 세션 데이터를 사용하여 구조화된 토큰화를 적용해 실험을 진행했습니다. 결과적으로, 구조화 토큰화와 순차 처리 방식이 표형 및 반구조화된 데이터 셋을 사용하는 기본 방법들에 비해 행동 모델링에서 더 나은 성능을 보여주었습니다. 이 접근법은 도메인 지식을 활용함으로써 더 나은 예측 성능을 제공하는 것으로 나타났습니다.



### TLEX: An Efficient Method for Extracting Exact Timelines from TimeML Temporal Graphs (https://arxiv.org/abs/2406.05265)
Comments:
          25 pages, 9 figures

- **What's New**: 이번 연구에서는 TimeML 주석(texts)로부터 완전한 이벤트 타임라인을 추출하는 TLEX (TimeLine EXtraction)이라는 새로운 시스템을 개발했습니다. TLEX는 기존의 타임라인 추출 방법들보다 정확하며, 특히 이벤트의 불일치 및 불확정 섹션을 자동으로 식별하는 두 가지 새로운 기능을 추가했습니다.

- **Technical Details**: TLEX는 TimeML 주석을 트렁크와 브랜치 구조로 배열된 타임라인 컬렉션으로 변환합니다. 기존 작업과 마찬가지로, TLEX는 시간 그래프의 일관성을 검사하고 정렬합니다. 또한, 특정 관계가 불일치를 초래하는지 식별하고, 타임라인의 불확정 섹션을 식별할 수 있습니다. 이는 자연어 처리 및 이벤트 정렬 작업에 중요한 정보입니다.

- **Performance Highlights**: TLEX는 네 개의 코퍼스로부터 385개의 TimeML 주석 텍스트에 적용되어 실험적 평가를 거쳤으며, 123개의 텍스트가 불일치 상태였으며, 181개 텍스트는 여러 '실제 세계' 또는 주요 타임라인을 가지고 있고, 총 2,541개의 불확정 섹션이 발견되었습니다. 샘플링 평가 결과 TLEX는 다섯 가지 차원에서 98-100%의 정확도를 가지고 있음이 입증되었습니다: 타임포인트의 정렬, 주요 타임라인 수, 주요 및 부속 타임라인의 타임포인트 배치, 브랜치 타임라인의 연결 포인트, 불확정 섹션의 위치.



### Generative Explore-Exploit: Training-free Optimization of Generative Recommender Systems using LLM Optimizers (https://arxiv.org/abs/2406.05255)
Comments:
          Accepted at ACL 2024 Main Proceedings

- **What's New**: 추천 시스템의 새로운 혁신은 대형 언어 모델(LLMs)을 활용한 생성적 추천 시스템의 도입입니다. 이러한 시스템은 콘텐츠 추천 외에도 질문 제안과 같은 오픈-셋 작업에 사용될 수 있습니다. LLMs의 방대한 세계 지식을 통해 뛰어난 추천이 가능해졌으나, 사용자의 피드백을 반영하여 지속적으로 LLMs를 미세 조정하는 것은 비용이 많이 듭니다. 이 논문에서는 사용자의 피드백 루프를 활용한 비트레이닝 방법을 제안하며, 생성적 탐색-활용(generative explore-exploit) 방법을 통해 새로운 추천 항목을 탐색하고 기존의 높은 참여도 항목을 활용하는 방식을 소개합니다.

- **Technical Details**: 해당 연구에서는 e-commerce와 general knowledge 두 가지 도메인에서 질문 생성 작업을 통해 접근법을 평가하였습니다. 사용자의 피드백을 Click Through Rate(CTR)로 모델링하였고, CTR 신호를 활용하여 LLM 기반의 탐색-활용 메커니즘을 통해 새로운 후보 항목을 생성하고 평가하였습니다. 제안된 방법은 비트레이닝 방식으로 동작하여, 맥락 특정의 암묵적 참여 신호를 LLM 입력으로 합성하여 추천 항목을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 LLM 기반의 탐색-활용 접근법이 반복적으로 추천을 개선하고 CTR을 꾸준히 증가시키는 것으로 나타났습니다. 탐색 과정이 사용자의 선호도를 학습하는 데 중요한 역할을 하여, 탐욕적인 활용 전용 접근 방식의 문제점을 피할 수 있음을 확인했습니다. 인간 평가 결과 또한 정량적 결과를 강력하게 뒷받침하였습니다.



### Improving Logits-based Detector without Logits from Black-box LLMs (https://arxiv.org/abs/2406.05232)
- **What's New**: DALD (Distribution-Aligned LLM Detection)은 블랙박스 LLM 감지의 성능을 극대화하는 혁신적인 프레임워크로, 소스 LLM의 logits 없이도 작동합니다. 이를 통해 surrogate 모델의 분포를 미지의 타겟 LLM과 정렬하여 향상된 검출 능력을 보장합니다.

- **Technical Details**: DALD는 공개된 고급 모델(예: ChatGPT, GPT-4 및 Claude-3)의 출력 샘플을 활용하여 surrogate 모델을 미세 조정합니다. 이는 surrogate 모델의 분포를 타겟 모델의 분포에 맞추는 것을 목표로 하며, 소규모 데이터셋(<10K 샘플)을 사용합니다. 이 방법은 비용 효율적이며, 소스 모델의 logits 접근 없이도 모델 감지 성능을 높입니다.

- **Performance Highlights**: DALD는 소스 모델을 알지 못하더라도 검출 성능을 극대화하며, 최신 모델 업데이트에도 신속하게 적응합니다. 즉, 다양한 소스로부터 텍스트를 정확하게 식별할 수 있도록 하는 다중 소스 감지기를 가능케합니다. GPT-Neo-2.7B와 같은 기존 surrogate 모델들이 감소된 정확도를 보이는 상황에서도 DALD는 일관된 높은 성능을 유지합니다.



### On Subjective Uncertainty Quantification and Calibration in Natural Language Generation (https://arxiv.org/abs/2406.05213)
- **What's New**: 이 논문은 대형 언어 모델(large language models, LLMs)의 자유 형식 응답 생성에서의 불확실성을 정량화하는 새로운 방법을 제안합니다. 특히, 베이지안 의사결정 이론(Bayesian decision theory) 관점에서 출발하여 모델의 주관적 불확실성을 정량화하고 이를 보정하는 방법을 탐구합니다. 이 접근 방식은 GPT와 Gemini 모델을 사용한 질문 응답(question answering) 및 기계 번역(machine translation) 작업에서 유의미한 불확실성 추정을 추출하는 데 적용되었습니다.

- **Technical Details**: 논문은 유사성 측정치(similarity measure)를 기반으로 한 유틸리티 함수(utility function)를 정의하여 언어 모델이 생성한 응답과 가상의 진짜 응답을 비교합니다. 이를 통해 주관적 불확실성을 베이지안 위험(Bayes risk)으로 설명하며, 이는 후보 생성물 중 최대 기대 유용성을 달성하는 것입니다. 또한, 모델의 보정을 위해 신뢰도 다이어그램(reliability diagrams)과 일반화된 기대 보정 오류(generalized expected calibration error)를 사용합니다. 추가로, 베이지안 모델링을 통해 예측 불확실성을 위한 에피스테믹 불확실성(epistemic uncertainty)과 인헨터러블 불확실성(aleatoric uncertainty)으로 분해합니다.

- **Performance Highlights**: 질문 응답 및 기계 번역 작업에서 제안된 방법들은 GPT 및 Gemini 모델로부터 유의미한 불확실성 추정을 추출하고, 이들의 보정을 효과적으로 수행하는 데 성공했습니다. 이는 자유 형식 응답 생성에서 불확실성을 보다 체계적으로 다룰 수 있는 가능성을 보여줍니다.



### LLMs Are Not Intelligent Thinkers: Introducing Mathematical Topic Tree Benchmark for Comprehensive Evaluation of LLMs (https://arxiv.org/abs/2406.05194)
- **What's New**: 이 연구는 GPT-4와 같은 대규모 언어 모델(LLMs)의 수학적 추론 능력을 평가하기 위해 새로운 벤치마크인 MaTT(Mathematical Topics Tree)를 소개합니다. MaTT는 위키피디아의 '수학 주제 목록'을 바탕으로 12개의 주요 수학 주제를 다루며, 각 주제에 대해 다계층적 체계 구조와 문제들이 포함되어 있습니다.

- **Technical Details**: MaTT 벤치마크는 위키피디아에서 수학 주제를 식별하고, 참고 서적들의 목차를 활용하여 주제 트리를 구성하였습니다. 이 후, 주제 트리의 각 말단 노드 아래에 문제들을 수집하고, 각 문제에 대해 다중 선택 옵션을 제공하여 다양한 수학적 주제에 대한 LLMs의 성능을 평가할 수 있는 체계를 마련하였습니다.

- **Performance Highlights**: 실험 결과, GPT-4는 다중 선택 옵션이 제공된 상태에서도 54%의 정확도를 기록하였고, 선택지 없이 질문이 제공되었을 때는 정확도가 최대 24.2% 포인트 하락하였습니다. 또한, Chain-of-Thought 프롬프트를 사용한 경우에도 성능 향상이 거의 없었습니다. 자세한 분석을 통해, GPT-4가 올바른 답변을 제공한 경우에만 53.3%의 설명이 완전하고 정확했으며 이는 모델이 진정한 수학적 추론을 수행했음을 의미합니다.



### Correlation Does Not Imply Compensation: Complexity and Irregularity in the Lexicon (https://arxiv.org/abs/2406.05186)
Comments:
          To appear in Proceedings of the Society for Computation in Linguistics 2024

- **What's New**: 이번 연구는 언어 내에서 형태론적 불규칙성(morphological irregularity)과 음운적 복잡성(phonotactic complexity) 간의 관계를 25개의 언어에 대해 분석했습니다. 기존의 연구에서 영어의 작은 샘플을 통해 이 관계가 밝혀진 바 있으나, 더 큰 샘플의 언어에 대해서는 아직 검증되지 않았습니다. 특히, 단어 빈도(frequency)와 길이가 음운적 복잡성과 형태론적 불규칙성 모두에 영향을 미칠 수 있어 이러한 요인들을 새롭게 고려했습니다.

- **Technical Details**: 본 연구는 UniMorph 데이터베이스에 있는 25개의 언어를 대상으로 정보 이론적 측정치를 사용해 음운적 복잡성과 형태론적 불규칙성을 분석했습니다. 형태론적 불규칙성 및 음운적 복잡성에 관한 지표는 Pimentel et al. (2020)과 Wu et al. (2019)의 방법론을 따랐습니다. 또한, 단어 길이와 단어 빈도 등의 변수가 관계에 미치는 영향을 배제하기 위해 통제했습니다.

- **Performance Highlights**: 연구 결과, 평균적으로 언어 내에서 형태론적 불규칙성과 음운적 복잡성 간에 긍정적인 관계가 존재함을 발견했습니다. 그렇지만 개별 언어에서는 그 방향이 다르게 나타날 수 있었습니다. 또한, 단어 길이와 형태론적 불규칙성 간의 부정적인 관계가 새로 발견되었으며, 기존 연구에서의 일부 결과는 생각보다 일관성이 적었습니다. 예를 들어, 단어 길이가 긴 단어일수록 정보량이 적고, 자주 사용되는 단어일수록 형태론적으로 불규칙하다는 기존의 결과를 재확인했습니다.



### Direct Preference Optimization for Suppressing Hallucinated Prior Exams in Radiology Report Generation (https://arxiv.org/abs/2406.06496)
- **What's New**: 최근 발생한 비전-언어 생성 모델(Generative Vision-Language Models, VLMs)의 발전은 방사선학에서 인공 지능(AI)의 잠재력을 크게 확장시키고 있습니다. 특히, VLM이 사전 훈련된 모델을 수정하여 원하지 않는 종류의 생성물을 억제하는 간단한 방법을 제안하였습니다. 주된 목표는 과거 검사에 대한 환상을 억제하는 것으로, 특히 흉부 X-ray(CXR) 보고서 생성에서 이 문제가 두드러집니다. 이를 통해 모델의 임상 정확도를 유지하면서도 환상 생성 줄이는 방법을 입증하였습니다.

- **Technical Details**: DPO(Direct Preference Optimization)를 기반으로 하는 이 방법은 사전 훈련된 VLM을 수정하여 원치 않는 행동을 억제하는 데 사용됩니다. DPO는 보상 모델을 따로 필요로 하지 않기 때문에 강화 학습(RL, Reinforcement Learning)보다 더 간단하고 안정적입니다. 특히, 흉부 X-선(Chest X-ray) 보고서 생성에서 과거 검사에 대한 환상(hallucinations)을 줄이는 데 집중하였습니다.

- **Performance Highlights**: 실험 결과, DPO 미세 조정을 통해 과거 검사에 대한 환상 줄이는 효과가 3.2 ~ 4.8배 향상되었으며, 임상 정확도(clinical accuracy)에도 영향을 미치지 않았습니다. 이 연구는 의료 VLM에 DPO를 적용한 최초의 연구로, 문제 행동을 억제하며 전체 임상 정확도를 유지하는 데 성공하였습니다.



### Parallelizing Linear Transformers with the Delta Rule over Sequence Length (https://arxiv.org/abs/2406.06484)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 델타 룰(delta rule) 업데이트를 사용하는 델타넷(DeltaNet)을 하드웨어 효율적으로 학습할 수 있는 알고리즘을 제안합니다. 기존의 델타넷은 순차적 알고리즘을 사용해 시퀀스 길이에 따라 병렬화되지 않아 하드웨어 효율성이 떨어졌으나, 이번 제안을 통해 보다 큰 모델과 데이터셋에 적용할 수 있는 길이 열렸습니다.

- **Technical Details**: 이 연구에서는 델타넷을 행렬 값의 RNN으로 재매개화(Reparameterization)하였으며, 이는 일반화된 하우스홀더 변환(Householder transformation)을 기반으로 합니다. 또한, 메모리 효율적인 WY 표기법을 활용하여 하우스홀더 행렬의 곱을 계산함으로써, 매트릭스 크기의 숨은 상태를 물리적으로 나타낼 필요가 없어졌습니다. 이를 통해 델타넷의 포워드/백워드 패스를 시퀀스 길이에 따라 병렬화할 수 있게 되었습니다.

- **Performance Highlights**: 제안된 알고리즘을 사용하여 100B 토큰에 대해 1.3B 모델을 학습한 결과, 델타넷은 퍼플렉시티(perplexity) 및 제로샷(zero-shot) 성능 면에서 Mamba 및 GLA와 같은 최신 선형 시간 알고리즘을 능가하는 성과를 보였습니다. 또한, 슬라이딩 창 어텐션(sliding-window attention) 레이어 또는 글로벌 어텐션(global attention) 레이어와 결합한 하이브리드 모델은 기존 트랜스포머보다 뛰어난 성능을 나타냈습니다.



### Towards a Personal Health Large Language Mod (https://arxiv.org/abs/2406.06474)
Comments:
          72 pages

- **What's New**: 이 연구는 모바일 및 웨어러블 장치에서 수집한 데이터로 개인 건강을 모니터링할 수 있도록 설계된 Personal Health Large Language Model (PH-LLM)을 소개합니다. 이 모델은 Gemini로부터 파인튜닝 (fine-tuning)되어 수치적 시간 시리즈 개인 건강 데이터를 이해하고 추론할 수 있게 했습니다.

- **Technical Details**: PH-LLM은 잠재적인 개인 맞춤 인사이트와 추천을 제공하기 위해 설계된 세 가지 데이터 세트를 활용합니다. 첫 번째는 수면 패턴, 신체 활동, 생리적 반응을 통해 개인화된 인사이트와 추천을 생성하는 데이터셋이며, 두 번째는 전문가 도메인 지식을 평가하는 데이터셋, 세 번째는 자기 보고된 수면 결과를 예측하는 데이터셋입니다. 857개의 사례 연구를 통해 실제 시나리오를 평가하고, 다중 선택 (MCQ) 시험을 통해 도메인 지식을 검증했습니다.

- **Performance Highlights**: PH-LLM은 피트니스 도메인에서 전문가와 통계적으로 유의미하게 다르지 않은 성능을 보였으며, 수면 도메인에서는 전문가보다 조금 낮은 성능을 보였습니다. 그러나 도메인 지식 활용과 수면 인사이트 개인화에서 유의미한 개선을 보였습니다. MCQ 시험 결과, PH-LLM은 수면 분야에서 79%, 피트니스 분야에서 88%의 점수를 기록하며, 이는 사람 전문가의 평균 점수를 초과하는 결과입니다. 또한, 웨어러블 데이터의 텍스트 및 멀티모달 인코딩을 통해 자기 보고된 수면 품질 결과를 예측하는 데 성공했습니다.



### Husky: A Unified, Open-Source Language Agent for Multi-Step Reasoning (https://arxiv.org/abs/2406.06469)
Comments:
          50 pages, 42 figures. Project webpage available [here](this https URL)

- **What's New**: 이 논문에서는 다양한 복잡한 작업을 해결하기 위한 통합적인 행동 공간을 학습하는 오픈 소스 언어 에이전트인 Husky를 소개합니다. Husky는 수치, 표 형식 및 지식 기반 추론 작업을 해결하기 위해 두 단계의 행동 생성과 실행을 반복합니다. 또한 HuskyQA라는 새로운 평가 세트를 도입하여 혼합 도구 추론(mixed-tool reasoning) 능력을 강조합니다.

- **Technical Details**: Husky는 작업을 해결하기 위해 두 가지 주요 단계를 거칩니다: 1) 행동 생성(action generation) 단계에서는 문제 해결을 위한 다음 행동을 예측하고, 2) 행동 실행(action execution) 단계에서는 전문가 모델을 사용하여 행동을 실행하고 현재 솔루션 상태를 업데이트합니다. 이 에이전트는 [code], [math], [search], [commonsense]와 같은 도구를 사용하여 다양한 행동을 수행합니다. 각 도구는 코드 생성기(code generator), 수학 추론자(math reasoner), 검색 쿼리 생성기(query generator), 상식 추론자(commonsense reasoner)와 같은 전문가 모델과 연결됩니다.

- **Performance Highlights**: 실험 결과, Husky는 기존의 언어 에이전트보다 우수한 성능을 보이며, 특히 HuskyQA 평가 세트에서 GPT-4와 같은 최신 모델과 견줄만한 성능을 보여줍니다. Husky는 다양한 작업에서 높은 성능을 유지하며, GSM-8K, HotpotQA, FinQA와 같은 여러 평가 세트에서도 이전 모델들보다 우수한 성과를 냈습니다.



### AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction (https://arxiv.org/abs/2406.06465)
- **What's New**: 영상 생성에서 Text-guided video prediction (TVP)이라는 새로운 접근법을 소개합니다. TVP는 초기 프레임과 텍스트 지시문에 따라 미래 프레임의 움직임을 예측하는 작업으로, 가상 현실, 로봇 공학, 콘텐츠 제작 등 다양한 분야에 적용될 수 있습니다. 기존 방법들은 Stable Diffusion을 적용했지만, 프레임 일관성과 시간적 안정성 문제에서 어려움을 겪었습니다. 이번 논문에서는 이 문제를 해결하기 위해 Multi-Modal Large Language Model (MLLM)을 도입하고, DQFormer(Dual Query Transformer) 아키텍처와 같은 새로운 기술을 제안합니다.

- **Technical Details**: 본 논문에서는 텍스트 및 프레임을 조건 임베딩으로 통합하여 미래 프레임을 예측하는 DQFormer 아키텍처를 설계했습니다. 또한, Long-Short Term Temporal Adapters와 Spatial Adapters를 개발하여 기존의 일반 동영상 확산 모델을 특정 시나리오로 빠르고 효율적으로 전환할 수 있도록 했습니다. 이를 통해 최소한의 훈련 비용으로 고품질의 제어 가능한 영상을 생성할 수 있습니다.

- **Performance Highlights**: 이 방법은 Something Something V2, Epic Kitchen-100, Bridge Data, UCF-101 등 네 가지 데이터셋에서 실험을 통해 기존 최신 기술보다 월등한 성능을 보여줬습니다. 특히 Bridge와 SSv2 데이터셋에서 각각 91.2%와 55.5%의 FVD 개선을 이루며 다양한 도메인에서 그 효과를 입증하였습니다.



### Transforming Wearable Data into Health Insights using Large Language Model Agents (https://arxiv.org/abs/2406.06464)
Comments:
          38 pages

- **What's New**: 이번 논문에서는 개인 헬스 인사이트 에이전트(Personal Health Insights Agent, PHIA)를 소개합니다. PHIA는 최신 코드 생성 및 정보 검색 도구를 활용하여 웨어러블 기기로부터 수집된 행동 건강 데이터를 분석하고 해석하는 시스템입니다.

- **Technical Details**: PHIA는 코드 생성(code generation) 및 정보 검색(information retrieval) 도구를 사용하여 웨어러블 기기의 데이터를 분석합니다. 이를 위해 4000개 이상의 건강 인사이트 질문으로 구성된 벤치마크 질의응답 데이터셋 두 개를 큐레이션하였습니다.

- **Performance Highlights**: 650시간 동안 전문가와 일반인 평가를 통해 PHIA는 사실적 수치 질문의 84% 이상, 개방형 질문의 83% 이상을 정확하게 처리할 수 있음을 확인했습니다. 이를 통해 개인이 자신의 웨어러블 데이터를 해석할 수 있도록 지원하고, 데이터 기반 인사이트를 바탕으로 한 개인 맞춤형 웰니스 관리로 새로운 시대를 열 가능성이 있습니다.



### A Large Language Model Pipeline for Breast Cancer Oncology (https://arxiv.org/abs/2406.06455)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)을 활용하여 유방암 환자에 대한 보조 방사선 치료와 화학 요법 예측의 정확도를 높이는 방법을 탐구하였습니다. 최신 OpenAI 모델을 임상 데이터와 임상 지침 텍스트로 미세 조정하였고, 그 결과 0.85 이상의 높은 분류 정확도를 달성했습니다.

- **Technical Details**: 연구진은 Langchain 프롬프트 엔지니어링 파이프라인을 사용하여 특정한 종양학 데이터를 통해 LLMs를 미세 조정했습니다. Duke MRI 데이터셋을 사용했고, HER-2 상태 및 종양 단계와 같은 주요 변수를 포함했습니다. GPT-3.5 Turbo, Babbage, DaVinci와 같은 OpenAI의 여러 GPT 모델이 사용되었으며, GPT-3.5 Turbo의 Retrieval Augmented Generation(RAG) 능력과 고급 함수 호출 기능 등이 중요한 역할을 했습니다.

- **Performance Highlights**: 유방암 환자의 보조 방사선 치료와 화학 요법 분류에서 0.85 이상의 높은 정확도를 달성했습니다. 인간 종양학자와 비교했을 때 일부 시나리오에서 모델이 더 나은 치료 예측을 제시할 가능성은 8.2%에서 13.3%로 평가되었습니다. 이는 LLMs가 더 일관되고 정보에 기반한 치료 결정을 지원할 수 있음을 시사합니다.



### LLM Dataset Inference: Did you train on my dataset? (https://arxiv.org/abs/2406.06443)
Comments:
          Code is available at \href{this https URL

- **What's New**: 최근 대형 언어 모델(LLMs)의 실제 사용이 증가함에 따라 저작권 문제도 증가하고 있습니다. 기존 연구에서는 모델 훈련 데이터의 개별 텍스트 시퀀스를 식별하는 멤버십 추론 공격(Membership Inference Attacks, MIAs)을 제안해 왔습니다. 하지만 이 논문에서는 이러한 성공적인 추론이 데이터 분포 변화에 의해 혼란을 겪고 있다고 주장합니다. 이를 해결하기 위해 저자들은 새로운 데이터셋 추론 방법을 제안합니다. 이는 특정 저자가 작성한 여러 문서를 통해 모델이 훈련되었는지를 식별하는 보다 현실적인 접근법입니다.

- **Technical Details**: 이 연구에서는 멤버십 추론 공격(MIAs)이 개별 문장보다는 개념의 훈련 여부를 감지한다는 점을 밝혀냈습니다. 기존 MIAs의 한계로, 동일한 데이터 분포에서 멤버와 비멤버를 구분하는 능력은 무작위 추측과 크게 다르지 않음을 확인했습니다. 이를 극복하기 위해 저자들은 특정 저자가 작성한 데이터셋 전체를 식별할 수 있는 데이터셋 추론 방법을 제안했습니다. 제안된 방법은 여러 MIAs를 선택적으로 결합하여 주어진 데이터셋에 대한 통계 테스트를 수행하는 방식입니다.

- **Performance Highlights**: 새롭게 제안된 데이터셋 추론 방법은 'Pile' 데이터셋의 서브셋에서 훈련 세트와 검증 세트를 구분하는데 유의미한 p-값 < 0.1을 달성했습니다. 또한, 잘못된 긍정 결과는 전혀 없었으며, 검증 데이터의 두 서브셋을 비교할 때 모든 경우에서 p-값이 0.5 이상이었습니다. 이 방법은 실질적으로 1000개의 텍스트 시퀀스만으로도 특정 데이터셋이 LLM의 훈련에 사용되었는지를 감지할 수 있습니다.



### STimage-1K4M: A histopathology image-gene expression dataset for spatial transcriptomics (https://arxiv.org/abs/2406.06393)
- **What's New**: 이번 연구에서는 병리 이미지 내 개별 공간 지점의 유전자 발현 정보를 제공하는 혁신적인 데이터셋인 STimage-1K4M을 소개합니다. 이 데이터셋은 1,149개의 이미지와 4,293,195개의 서브타일 이미지-유전자 발현 쌍으로 구성되어 있으며, 공간 전사체학(spatial transcriptomics) 데이터를 기반으로 합니다. 이 데이터셋은 고해상도 병리 이미지를 각 서브타일로 나누고, 각 서브타일에 15,000-30,000 차원의 유전자 발현 데이터를 매칭시켜 줍니다.

- **Technical Details**: STimage-1K4M 데이터셋은 Hematoxylin과 Eosin(H&E) 염색으로 세포의 핵과 기질을 나타내는 병리 슬라이드에서 유도된 이미지를 포함합니다. 유전자 발현은 mRNA 분자의 정보가 DNA로부터 생성되는 과정으로, 공간 전사체학(ST) 기술을 통해 보존된 공간적 정보와 함께 측정됩니다. 이 데이터셋은 Spatial Transcriptomics, Visium 및 VisiumHD 기술을 사용하며, 다양한 생체 조직 유형을 포함한 10개의 서로 다른 종을 다룹니다.

- **Performance Highlights**: STimage-1K4M 데이터셋은 상세한 조직 구조 분류 및 이미지/텍스트 검색 작업에서 뛰어난 성능을 보여줄 것으로 기대됩니다. 특히, 유전자 발현 데이터를 통한 세포 간 소통, 조직 아키텍처 및 질병 진행 연구에 유용하게 사용될 수 있습니다. 이 데이터셋은 다중 모달(modal) 데이터 분석 및 개인 맞춤형 의학 연구에 새롭고 유망한 길을 열어줄 것입니다.



### Towards Lifelong Learning of Large Language Models: A Survey (https://arxiv.org/abs/2406.06391)
Comments:
          37 pages

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 지속적인 학습 및 적응 능력을 논의하고 있습니다. 전통적인 정적 데이터셋을 사용하는 방법이 아닌, 실제 세계의 동적인 정보를 다루기 위해 계속해서 학습하고 적응할 수 있는 지속적인 학습(lifelong learning) 방법을 사용해야 함을 강조합니다. 지속적인 학습을 내부 지식(Internal Knowledge)과 외부 지식(External Knowledge)으로 나누어 분류하며, 각각의 접근 방식을 설명합니다.

- **Technical Details**: 논문은 지속적인 학습 전략을 Internal Knowledge와 External Knowledge로 분류합니다. Internal Knowledge는 지속적인 사전 훈련(continual pretraining)과 지속적인 미세 조정(continual finetuning)을 포함하며, 이는 다양한 시나리오에서 LLMs의 적응성을 향상시킵니다. External Knowledge는 검색 기반(retrieval-based) 및 도구 기반(tool-based) 지속 학습을 포함하며, 이는 외부 데이터 소스와 계산 도구를 활용해 모델의 기능을 확장시키면서 핵심 파라미터를 수정하지 않는 방식을 의미합니다.

- **Performance Highlights**: 이 서베이의 주요 기여는 다음과 같습니다: (1) 12개의 시나리오로 지속 학습 문헌을 분류하는 새로운 분류법을 도입; (2) 모든 지속 학습 시나리오에서 공통적인 기법을 식별하고, 각 시나리오 내 다양한 기법 그룹으로 기존 문헌을 분류; (3) 모델 확장(model expansion)과 데이터 선택(data selection)과 같은 신기술을 강조하여, 이는 LLM 이전 시대에서는 덜 탐구되었던 기술들입니다. 이 논문을 통해 LLM이 실제 응용에서 적응성, 신뢰성 및 전반적 성능을 향상시키고자 합니다.



### Low-Rank Quantization-Aware Training for LLMs (https://arxiv.org/abs/2406.06385)
- **What's New**: LR-QAT는 파라미터-효율적인 미세 조정(PEFT)과 저-랭크 적응(LoRA) 문헌에서 영감을 받아, 대형 언어 모델(LLM)을 위한 가벼우며 메모리 효율적인 양자화 알고리즘입니다. 이 방법은 예측 성능을 유지하면서 메모리를 절약할 수 있는 여러 요소를 활용합니다. 또한, 전통적인 양자화-인지 훈련(QAT)과 비교했을 때 메모리 사용량을 획기적으로 줄이면서 동일한 모델 성능을 달성합니다.

- **Technical Details**: LR-QAT는 여러 혁신적인 구성 요소를 결합하여 메모리 사용을 줄입니다. 첫째, 양자화 그리드를 인식하는 저-랭크 보조 가중치를 사용합니다. 둘째, 고정 소수점 또는 두 배로 압축된 정수를 사용한 다운캐스팅 연산자를 도입합니다. 마지막으로, 체크포인팅을 결합하여 메모리 스파이크를 피합니다. 또한, 이 방법은 광범위한 양자화 설정(예: 채널별 양자화, 활성화 양자화 등)과 호환되며, 대부분의 다른 후 훈련 양자화와도 결합이 가능합니다.

- **Performance Highlights**: LR-QAT는 LLaMA-2/3와 Mistral 모델 가족에 적용하여 그 효과를 입증했습니다. 메모리 사용량을 24GB 이하로 줄여 일반적인 소비자용 GPU에서도 훈련이 가능하게 하였으며, 전통적인 전-훈련 양자화(PTQ) 접근법을 능가하면서도 전모델 QAT와 동일한 예측 성능을 도달했습니다.



### Diffusion-RPO: Aligning Diffusion Models through Relative Preference Optimization (https://arxiv.org/abs/2406.06382)
- **What's New**: Diffusion-RPO는 인간의 선호도에 더 잘 맞는 텍스트-이미지 생성(T2I) 모델을 개발한 새로운 방법입니다. 기존 Diffusion-DPO와 달리, 동일한 프롬프트뿐만 아니라 의미적으로 관련된 콘텐츠를 가진 프롬프트-이미지 쌍을 활용하여 학습 효율을 증대시켰습니다. 또한, 평가 비용이 높고 재현성이 낮으며 해석이 어려운 문제를 해결하기 위해 새로운 평가 지표인 스타일 정렬(style alignment)을 도입했습니다.

- **Technical Details**: Diffusion-RPO는 여러 모달리티의 프롬프트-이미지 쌍을 활용하여 상대적 선호도 학습(relative preference optimization)을 적용한 기법입니다. 이를 위해, 1) 각 시간 단계마다 상대적 선호도 정렬을 적용하기 위한 RPO 손실을 도출하고, 2) CLIP 인코더를 구현하여 프롬프트와 이미지를 동일한 임베딩 공간에 투영했습니다. 이를 통해 여러 모달리티의 프롬프트-이미지 쌍 사이의 유사성을 정확하게 측정할 수 있었습니다.

- **Performance Highlights**: 실험 결과, Diffusion-RPO는 Supervised Fine-Tuning 및 Diffusion-DPO 등을 포함한 기존 방법들을 능가하여 Stable Diffusion 1.5 및 XL-1.0 모델에서 인간의 선호도 및 스타일 정렬 작업에서 탁월한 성능을 보였습니다. 이 방법은 인간 선호도를 반영한 자동 평가와 스타일 정렬 작업 모두에서 명확한 마진으로 우수한 성능을 입증했습니다.



### Learning Fine-Grained Controllability on Speech Generation via Efficient Fine-Tuning (https://arxiv.org/abs/2406.06251)
Comments:
          Accepted by InterSpeech 2024

- **What's New**: 이번 연구에서는 Voicebox Adapter라는 새로운 접근 방식을 제안하였습니다. 이는 사전 훈련된 Voicebox 음성 생성 모델에 교차 주의 모듈(cross-attention module)을 사용하여 세분화된 조건(fine-grained conditions)을 통합합니다. 주요 실험 결과, LoRA(low-rank adaptation) 및 bias-tuning 조합이 최상의 성능을 보여주며, 음성 품질을 손상시키지 않고 제어 가능성을 향상시켰습니다.

- **Technical Details**: Voicebox Adapter는 Transformer 레이어에 교차 주의 모듈을 추가하여 세분화된 조건 정보를 추출하고 통합하는 방식입니다. 이번 연구에서는 Efficient Fine-Tuning, 즉 효율적인 미세 조정 방법을 탐구하여 사전 훈련된 매개변수(pre-trained parameters)와 새로운 모듈을 매끄럽게 연결했습니다. 실험 결과, 어댑터 파라미터가 모델의 일부분만 차지함에도 불구하고 전체 모델을 미세 조정한 것과 유사한 성능을 달성했습니다.

- **Performance Highlights**: 세 가지 세분화된 조건 생성 작업(punctuation, emphasis, laughter)에서 Voicebox Adapter의 효과성과 자원 효율성이 입증되었습니다. 추가 실험에서도 다양한 데이터 설정에 걸쳐 Voicebox Adapter의 강력한 성능이 강조되었습니다. 다양한 미세 조정 데이터 양과 숨겨진 차원 크기를 사용한 실험을 통해 다른 설정 하에서도 뛰어난 성능을 보였습니다.



### Label-Looping: Highly Efficient Decoding for Transducers (https://arxiv.org/abs/2406.06220)
- **What's New**: 본 논문에서는 변환기(Transducer) 추론을 위한 고효율 탐욕적 디코딩 알고리즘을 소개합니다. CUDA 텐서를 사용하여 배치 내 부분 가설을 나타내는 새로운 데이터 구조를 제안하고, 블랭크(blank) 예측을 내부 루프에서 처리하며, 비-블랭크 예측을 외부 루프에서 처리하여 GPU 병렬 처리를 극대화하는 중첩 루프 디자인을 채택하였습니다. 제안된 알고리즘은 일반적으로 사용될 수 있으며, 기존 변환기와 토큰-및-지속시간 변환기(Token-and-Duration Transducers) 모두에서 작동합니다.

- **Technical Details**: 제안된 데이터 구조는 PyTorch의 CUDA 텐서 연산을 사용하여 효율적으로 배치 내 부분 가설을 조작할 수 있습니다. 인코더와 디코더 투영을 결합기 계산 이전에 사전 계산하여 디코딩 병렬성을 최대화하는 전략을 사용합니다. 본 알고리즘은 블랭크와 비-블랭크 방출을 분리하여 처리함으로써 배치 내 비동기 가설 생성을 효율적으로 관리합니다. 이를 통해 최대 2.0배의 속도 향상을 달성할 수 있으며, 컴파일 및 GPU 호출 최적화 기술과 결합하면 최대 3.2배의 속도 향상을 달성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 배치 크기 32의 경우 기존 배치 디코딩 알고리즘과 비교하여 최대 2.0배의 속도 향상을 보였습니다. 제안된 알고리즘은 NeMo 도구를 통해 오픈 소싱될 예정이며, 추가적인 컴파일 및 GPU 호출 관련 기술과 결합할 경우 더 큰 속도 향상을 기대할 수 있습니다.



### Thunder : Unified Regression-Diffusion Speech Enhancement with a Single Reverse Step using Brownian Bridg (https://arxiv.org/abs/2406.06139)
Comments:
          5 pages, 3 figures, 4 tables, This paper will be submitted in the interspeech conference

- **What's New**: Thunder라는 새로운 모델을 제안했습니다. Thunder는 regression과 diffusion 기반 음성 강화(speech enhancement)를 통합한 모델로, Brownian bridge 프로세스를 활용해 모델이 두 가지 모드(regression 모드와 diffusion 모드)에서 작동할 수 있습니다. 기존의 점수 기반(score-based) diffusion 모델의 불안정한 그래디언트 문제를 해결하기 위해, 점수 함수를 예측하는 대신 청정한 음성 신호를 예측하도록 모델을 재구성했습니다. 이를 통해 모델 크기와 역방향 스텝 수를 줄이면서도 경쟁력 있는 성능을 달성했습니다.

- **Technical Details**: Thunder 모델은 Brownian bridge 프로세스를 활용하여 diffusion 기반의 음성 강화를 수행합니다. Forward와 reverse 프로세스를 통해 잡음을 증가시키거나 감소시키는 기존의 SDE(확률적 미분 방정식) 방법론 대신, 청정 음성을 직접 예측하도록 모델을 재설계했습니다. 특히, Thunder는 점수 기반 모델보다 더 낮은 불안정성을 가지고 있으며, 더 적은 수의 반복 스텝을 통해 성능을 유지합니다. 이는 regression 모드와 diffusion 모드 모두에서 동일한 모델 파라미터를 사용할 수 있도록 하기 때문에 시스템 복잡성을 감소시킵니다.

- **Performance Highlights**: Thunder 모델은 VoiceBank + DEMAND 데이터셋에서 우수한 성능을 보였으며, 더 적은 파라미터와 짧은 추론 시간으로 diffusion 기반의 기존 모델보다 우수한 결과를 도출했습니다. 특히, 단 한 번의 역 diffusion 스텝으로도 기존의 diffusion 기반 모델들을 능가하는 성능을 보였습니다. 이는 Brownian bridge 프로세스를 사용해 잡음 신호를 청정 음성으로 변환하는 방식의 효과를 입증합니다.



### StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection (https://arxiv.org/abs/2406.06097)
Comments:
          Accepted at ACL 2024 main conference

- **What's New**: 이 논문은 StreamST(Streaming Speech-to-Text Translation) 정책을 도입하며, 이를 위해 StreamAtt와 새로운 지연(metric)인 StreamLAAL을 제안합니다. 이 연구는 실시간 ST의 실제 요구에도 불구하고 제한된 연구를 보완하고자 합니다.

- **Technical Details**: StreamST는 연속적이고 제한이 없는 오디오 스트림을 처리하는 과제를 다룹니다. 이는 SimulST(동시 ST, Simultaneous ST)와는 달리, 이전 역사(history)에 대해 무엇을 유지할지에 대한 추가 결정을 요구합니다. 이 논문에서는 StreamAtt라는 첫 번째 StreamST 정책과, 기존 SimulST 지표와 비교할 수 있도록 설계된 첫 번째 StreamST 지연 메트릭인 StreamLAAL을 소개합니다.

- **Performance Highlights**: MuST-C v1.0의 모든 8개 언어에 걸친 광범위한 실험 결과, StreamAtt가 단순한 스트리밍 기준(baseline) 및 최신의 SimulST 정책과 비교하여 효과적이라는 것을 보여주었습니다. 이 연구는 StreamST 연구의 첫 번째 단계로서 중요한 기여를 합니다.



### RepoQA: Evaluating Long Context Code Understanding (https://arxiv.org/abs/2406.06025)
- **What's New**: RepoQA라는 새로운 벤치마크가 대형 언어 모델(LLMs)의 긴 맥락 코드 이해 능력을 평가하기 위해 소개되었습니다. RepoQA는 자연어 설명을 기반으로 코드 함수 검색을 테스트하는 'Searching Needle Function (SNF)' 작업을 포함합니다.

- **Technical Details**: RepoQA는 5개의 현대적인 프로그래밍 언어에 걸쳐 50개의 인기 있는 저장소에서 수집된 500개의 코드 검색 작업을 포함합니다. 데이터를 큐레이션할 때, 각 저장소에서 10개의 'needle' 함수를 선택하여 자연어 설명을 추가합니다. 모델 평가 단계에서는 주어진 코드 맥락과 함수 설명을 기반으로 LLM에게 특정 함수를 검색하라는 지시를 합니다.

- **Performance Highlights**: RepoQA를 사용하여 26개의 일반 및 코드 특정 LLM을 평가한 결과, 최고 성능의 오픈 및 독점 모델 사이에는 아직 작은 차이가 있고, 모델은 언어별로 다르게 성능을 발휘한다는 점을 발견했습니다. 또한, 주석이 없는 코드에서 더 잘 이해할 수 있는 경향이 있습니다.



### CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models (https://arxiv.org/abs/2406.06007)
- **What's New**: CARES 벤치마크 공개

- **Technical Details**: 논문에서는 Med-LVLMs의 신뢰성을 평가하기 위해 CARES라는 벤치마크를 소개했다. CARES는 신뢰성(trustfulness), 공정성(fairness), 안전성(safety), 프라이버시(privacy), 견고성(robustness) 등 5가지 차원에서 Med-LVLMs를 평가한다. 16개의 의료 이미지 모달리티(modalities)와 27개의 해부학적 부위를 다루며, 약 41K의 질문-답변 쌍을 포함하고 있다.

- **Performance Highlights**: Med-LVLMs는 자주 사실적 부정확성을 보이며, 신뢰성에서 문제가 발견됐다. 또한, 다양한 인구 집단에서 공정성의 문제를 나타내며, 공격에 취약하고 프라이버시 의식이 부족하다.



### FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning Using a Large Multimodal Mod (https://arxiv.org/abs/2406.06004)
Comments:
          Accepted at ACL (Main) 2024

- **What's New**: 이번 논문에서는 이미지 설명 생성 평가 메트릭(FLEUR)을 소개합니다. 기존 메트릭은 참조 캡션(reference caption)을 필요로 하고 설명이 부족했지만, FLEUR는 대형 멀티모달 모델(LMM)을 활용하여 참조 캡션 없이도 이미지를 평가하고, 점수에 대한 설명을 제공합니다.

- **Technical Details**: FLEUR는 멀티모달 모델을 통해 이미지 설명을 평가하며, 점수 스무딩 점수(smothing)를 도입하여 LMM의 출력 점수를 인적 평가와 더 가깝게 조정합니다. 또한, caption 평가를 위해 채점 기준(prompt)을 포함하여 점수의 신뢰도를 높입니다. FLEUR는 LLaVA 모델을 사용하여 점수를 계산합니다.

- **Performance Highlights**: FLEUR는 Flickr8k-CF, COMPOSITE, Pascal-50S와 같은 벤치마크 데이터셋에서 인간 판단과 높은 상관 관계를 보여주면서 최첨단 성능을 달성했습니다. 특히, 참조-기반(CLAR)의 메트릭과 비교할 때, FLEUR는 이미지 고려를 통해 더 나은 평가 점수와 설명을 제공합니다.



### ShiftAddLLM: Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization (https://arxiv.org/abs/2406.05981)
- **What's New**: 새롭게 발표된 논문에서는 대형 언어 모델(LLM)이 자원 제약이 있는 장치에서 구현될 때 직면하는 문제를 해결하는 방법으로 Shift-and-add 재구성 기술을 제안합니다. 이 기술은 LLM의 주의(attention) 및 다중층 퍼셉트론(MLP) 계층에서 비용이 많이 드는 곱셈을 하드웨어 친화적인 비트 이동 및 덧셈으로 대체합니다. 제안된 방법은 훈련된 LLM을 후처리하여 ShiftAddLLM이라는 효율적인 곱셈 없는 모델을 개발합니다.

- **Technical Details**: Shift-and-add 재구성 기술을 사용하는 ShiftAddLLM은 각 가중치 행렬을 이진 행렬로 양자(quantize)화하고 그룹별 스케일링 인자를 적용합니다. 연관된 곱셈은 (1) 활성화 및 스케일링 인자 간의 이동 및 (2) 이진 행렬에 따라 쿼리 및 추가 작업으로 재구성됩니다. 모델의 정확도 손실을 줄이기 위해, 다중 목적 최적화 방법을 사용하여 가중치 오류 및 출력 활성화 오류를 최소화합니다. 또한, 계층별 민감도에 따라 자동화된 비트 할당 전략을 개발하여 메모리 사용량 및 대기 시간을 줄입니다.

- **Performance Highlights**: 다섯 개의 LLM 패밀리와 여덟 개의 작업에 대한 실험 결과, ShiftAddLLM은 평균 퍼플렉시티(perplexity) 향상을 5.6 및 22.7 포인트 달성하였으며, 3비트 및 2비트에서 가장 경쟁력 있는 양자화된 LLM과 비교하여 유사하거나 더 낮은 대기 시간을 보여주었습니다. 또한, 원래의 LLM에 비해 메모리와 에너지 소모가 80% 이상 절감되었습니다.



### Prompting Large Language Models with Audio for General-Purpose Speech Summarization (https://arxiv.org/abs/2406.05968)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 처리 및 추론 능력을 활용한 음성 요약 프레임워크를 소개합니다. 음성을 토큰 표현으로 변환하는 오디오 인코더와 명령어 기반으로 조정된 LLM을 결합한 엔드 투 엔드 시스템을 제안합니다. 이 시스템은 다양한 도메인에서 임의의 음성 콘텐츠를 요약할 수 있으며, LLM의 명령어 전략을 조정함으로써 여러 스타일의 요약을 생성할 수 있습니다.

- **Technical Details**: 제안된 시스템은 LLM과 오디오 인코더의 두 가지 구성 요소로 이루어져 있습니다. 오디오 인코더는 음성 입력을 LLM이 해석할 수 있는 토큰 임베딩으로 변환하며, LLM은 MiniChat-3B를 사용합니다. 이 모델은 Llama 2 7B로부터 증류된 MiniMA-3B의 명령어 기반 조정 버전입니다. 오디오 인코더는 HuBERT-Large 모델을 원형으로 하여 음성을 1024-dim 표현으로 변환, 이를 3072-dim으로 선형 변환하여 최종적으로 12.5Hz의 오디오 토큰을 생성합니다. 이러한 토큰은 LLM이 텍스트와 동일한 방식으로 처리할 수 있습니다.

- **Performance Highlights**: CNN/DailyMail 데이터셋을 사용한 실험 결과, 제안된 시스템은 음성 인식 후 텍스트 처리를 하는 기존 방법보다 우수한 성능을 나타냈습니다. 이 시스템은 다양한 도메인의 음성 콘텐츠를 요약할 수 있으며, LLM의 내재된 능력을 활용하기 때문에 다양한 스타일로 요약을 생성할 수 있습니다.



### CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark (https://arxiv.org/abs/2406.05967)
- **What's New**: CVQA는 Visual Question Answering(VQA) 작업에서 문화적으로 다양한 이미지를 포함한 새로운 다국어 벤치마크 데이터셋입니다. 이 데이터셋은 28개국에서 26개의 언어와 11개의 스크립트를 다루며 총 9,000개의 질문을 포함합니다.

- **Technical Details**: 데이터 수집 과정에서는 문화적 전문가와 원어민을 포함시켜 각 국가의 고유한 문화적 요소를 반영한 이미지와 질문을 구성했습니다. 이러한 데이터셋은 Multiple-choice 형식으로 제공되며, 각 질문에는 총 네 개의 선택지가 포함됩니다. 데이터수집에서 사용된 이미지들은 개인 소장 이미지와 공개 라이선스 이미지를 활용했으며, 민감한 정보는 철저히 제거했습니다.

- **Performance Highlights**: 최신 Multimodal Large Language Models (MLLMs)를 CVQA에서 벤치마킹한 결과, 대부분의 MLLMs는 50% 이상의 정확도를 달성하지 못했습니다. 특히 덜 연구된 언어에서는 성능 저하가 더 두드러졌습니다.



### Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters (https://arxiv.org/abs/2406.05955)
- **What's New**: 새로운 dReLU 활성화 함수를 제안하여 대형 언어 모델(LLM)의 활성화 희소성(sparsity)을 향상시키고, 고품질의 훈련 데이터 혼합 비율을 활용하여 효과적인 희소화를 촉진합니다. 이를 통해 Mistral 및 Mixtral 모델에서 대폭적인 효율성 개선을 달성하였습니다.

- **Technical Details**: dReLU는 기존 SwiGLU 또는 GeGLU와 달리 더 높은 희소성을 제공합니다. Feed-Forward Network (FFN) 전문가 내에서 희소 활성화 패턴을 활용하여 Mixture-of-Experts (MoE) 모델의 효율성을 높였습니다. TurboSparse-Mistral-47B와 TurboSparse-Mixtral-47B 모델을 소개하며, 이들은 각각 2.5억 및 4.3억 개의 활성화된 매개변수를 가집니다.

- **Performance Highlights**: 평가 결과, 희소화된 모델은 2-5배의 디코딩 속도 향상을 이루었으며, TurboSparse-Mixtral-47B 모델은 모바일 폰 환경에서 초당 11 토큰의 추론 속도를 달성했습니다. 이 두 모델은 기존 모델들보다 뛰어난 성능을 보였습니다.



### Whose Preferences? Differences in Fairness Preferences and Their Impact on the Fairness of AI Utilizing Human Feedback (https://arxiv.org/abs/2406.05902)
Comments:
          To appear in the Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, ACL 2024

- **What's New**: 이 논문에서는 사람들의 피드백을 통해 AI 시스템이 사람들의 가치와 선호에 맞도록 조정하는 방법에 대한 연구가 강조됩니다. 특히 콘텐츠 조정의 공정성을 다루며, 인종, 나이, 정치 성향, 교육 수준, LGBTQ+ 정체성 등 다양한 인구 통계학적 요인에 따라 공정성 선호에 큰 차이가 있음을 발견했습니다. 이를 통해 다양한 인구 집단의 주석을 동등하게 반영하는 앙상블 모델이 단일 모델보다 더 나은 성능을 보인다는 사실을 입증했습니다.

- **Technical Details**: 이 연구는 Prolific과 MTurk에서 수집한 새로운 데이터셋을 사용하여 주석자의 인구 통계학적 특성이 공정성 판단에 미치는 영향을 분석했습니다. 각 주석자의 공정성 판단을 파악하기 위해, 의미적으로 유사하지만 다른 민감한 속성 그룹을 언급하는 두 개의 댓글을 비교하게 했습니다. 이를 통해 주석자의 인구 통계학적 특성과 텍스트에 언급된 인구 통계학적 속성이 개별 공정성에 대한 인식에 강력한 영향을 미친다는 것을 발견했습니다.

- **Performance Highlights**: 각 주석자 그룹의 선호 데이터로 학습된 다운스트림 모델들은 예측 성능에 있어 큰 차이를 보였습니다. 특히 특정 연령대(예: 38+ 나이 그룹)의 데이터로 학습된 모델은 다른 그룹의 데이터로도 높은 성능을 보였습니다. 또한, 다양한 인구 통계학적 그룹의 주석 데이터를 반영한 앙상블 모델이 더 나은 예측 성능을 제공한다는 것이 입증되었습니다. 이는 다양한 인구 그룹의 대표성을 높이기 위한 효과적인 접근 방식으로 제안되었습니다.



### LGR2: Language Guided Reward Relabeling for Accelerating Hierarchical Reinforcement Learning (https://arxiv.org/abs/2406.05881)
- **What's New**: 이번 연구는 인간의 자연어 명령을 활용하여 복잡한 로봇 제어 작업을 수행할 수 있는 상위-하위 강화 학습 계층 구조(Hierarchical Reinforcement Learning, HRL) 프레임워크인 LGR2(Language Guided Reward Relabeling)를 제안합니다. LGR2는 자연어 명령을 통해 상위 정책의 보상 함수를 생성하여 HRL의 비정상성(non-stationarity) 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: LGR2는 LLM(Large Language Models)을 활용하여 자연어 명령을 보상 함수 파라미터로 변환합니다. 이를 통해 상위 정책의 재생 버퍼 전환(replay buffer transitions)을 다시 레이블링합니다. 하위 원시 행동의 비정상성 문제를 완화하고, 고차원 보상 희소성(sparsity)을 해결하기 위해 힌드사이트 경험 재플레이(hindsight experience replay)를 사용합니다. 이 접근법은 고차원 희소 보상 환경에서 상위 70% 이상의 성공률을 달성했습니다.

- **Performance Highlights**: LGR2는 다른 기초적인 방법들이 큰 성과를 보여주지 못하는 복잡한 로봇 제어 작업에서 70% 이상의 성공률을 달성했습니다. 다양한 실험을 통해 이 접근법이 뛰어난 성능을 나타내었으며, 실제 로봇 조작 실험에서도 유사한 성공을 거두었습니다.



### STARLING: Self-supervised Training of Text-based Reinforcement Learning Agent with Large Language Models (https://arxiv.org/abs/2406.05872)
Comments:
          ACL 2024 (Findings)

- **What's New**: 새로운 연구는 텍스트 기반 강화 학습(Text-based Reinforcement Learning, TBRL) 에이전트의 일반화 능력을 향상시키기 위해 셀프 슈퍼바이즈드 RL(self-supervised RL)을 위한 상호작용 환경, STARLING(스타링)을 도입했습니다. 이 새로운 환경은 자동 생성된 게임을 통해 에이전트가 목표 환경에서 성과를 향상시키고 일반화 능력을 증진시킬 수 있게 합니다.

- **Technical Details**: STARLING은 대규모 언어 모델(LLM)과 인터랙티브 픽션 게임 엔진(Inform7)을 통합하여, 최소한의 인간 감독 하에 다양한 도메인의 텍스트 기반 게임을 쉽게 생성할 수 있습니다. 연구팀은 게임 아이디어의 시드 리스트(seed list)를 입력으로 사용하여 GPT-3를 통해 100개의 텍스트 기반 게임을 생성했습니다. 이 게임들은 물 끓이기, 파스타 요리하기 등 일상적인 기술을 사용하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, 현재 최첨단 텍스트 기반 RL 에이전트는 인간처럼 새로운 상황에서 이전에 학습한 기술을 활용하지 못하는 것으로 나타났습니다. STARLING은 이러한 한계를 극복하고 자체적으로 더 많은 게임을 생성하여 다양한 도메인에서 활약할 수 있는 RL 에이전트를 구축하는 데 중요한 기여를 할 수 있을 것으로 예상됩니다.



### Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents (https://arxiv.org/abs/2406.05870)
- **What's New**: Retrieval-augmented generation (RAG) 시스템에서 새로운 서비스 거부(DoS) 공격 방식인 'jamming' 공격이 소개되었습니다. 공격자가 하나의 'blocker' 문서를 RAG 데이터베이스에 추가하면 특정 쿼리에 대해 응답을 차단할 수 있다는 점을 보여줍니다. 기존 문서 제거나 수정 없이도 데이터베이스에 문서를 추가하는 것만으로 이러한 공격이 가능하다는 점이 주목할 만합니다.

- **Technical Details**: jamming 공격은 RAG 시스템의 지식 데이터베이스에 query-specific 'blocker' 문서를 추가하여, 해당 쿼리에 대해 적절한 답변이 생성되지 않도록 만듭니다. 이를 통해 LLM은 정보를 제공하거나 안전한 답변을 제공하지 않는다고 판단하며, 대답을 회피합니다. 본 연구에서는 문서 생성을 위해 간접 명령 주입(indirect prompt injection), 오라클 LLM을 통한 문서 생성, 블랙박스 최적화(black-box optimization) 등 세 가지 방법을 조사했습니다. 이 블랙박스 최적화 방식은 RAG의 임베딩 모델이나 LLM을 알 필요도, 추가 LLM의 도움도 필요 없도록 설계되었습니다.

- **Performance Highlights**: 다양한 RAG 시스템에 대해 blocker 문서의 효과를 측정했으며, 여러 데이터셋(NQ, HotpotQA), 임베딩 모델(GTR-base, Contriever), 그리고 LLM(Llama-2, Vicuna, Mistral)을 대상으로 실험을 수행하였습니다. 특히, 기존 LLM의 안전성 관련 메트릭은 jamming 공격에 대한 취약성을 잡아내지 못하며, 좀 더 안전하다고 평가된 모델일수록 jamming 공격에 더 취약하다는 결과가 도출되었습니다. 마지막으로, perplexity 기반 문서 필터링, 쿼리나 문서의 패러프레이징, 컨텍스트 크기 확대 등의 방어 방법도 연구되었습니다.



### Unified Text-to-Image Generation and Retrieva (https://arxiv.org/abs/2406.05814)
- **What's New**: 이번 연구에서는 텍스트-이미지 생성(Text-to-Image Generation)과 검색(Retrieval)을 통합한 새로운 프레임워크를 제안합니다. Multimodal Large Language Models (MLLMs)의 고유한 판별 능력을 활용하여, 훈련이 필요 없는 생성 기반 검색 방법을 도입하였으며, 자동 결정을 통해 텍스트 쿼리에 가장 적합한 이미지를 선택합니다. 또한, 새로운 벤치마크인 TIGeR-Bench를 구축하여 창의적이거나 지식 집약적인 도메인의 평가를 표준화했습니다.

- **Technical Details**: 제안된 프레임워크는 MLLMs의 양방향 판별 능력을 탐구하여, 전방 빔 탐색(forward beam search)과 역방향 재정렬(reverse re-ranking)을 통한 효율적인 생성 기반 검색 방법을 제안합니다. 텍스트 쿼리에 대해 생성된 이미지와 검색된 이미지 중 가장 적합한 것을 선택하는 자율 결정 모듈을 추가로 도입하였습니다. 이는 MLLMs를 사용하는 자율 회귀 생성 방식으로 텍스트-이미지 생성과 검색을 통합한 것입니다.

- **Performance Highlights**: TIGeR-Bench, Flickr30K, MS-COCO를 포함한 다양한 벤치마크를 통한 광범위한 실험 결과, 제안된 프레임워크가 높은 성능과 효과를 보여줍니다. 특히, 창의적 및 지식 집약적인 도메인에서 텍스트-이미지 요구를 충족시키는 데 있어 기존 방법들보다 우수함이 입증되었습니다.



### A Survey on LLM-Based Agentic Workflows and LLM-Profiled Components (https://arxiv.org/abs/2406.05804)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 전통적인 단일 경로의 생각사슬(Chain-of-Thought, CoT) 프롬트 방식을 능가하는 정교한 에이전트 워크플로우 개발을 촉진했습니다. 이 설문조사는 LLM-프로파일된 구성요소(LMPCs)에 초점을 맞추어 일반적인 워크플로우를 요약합니다.

- **Technical Details**: 생성형 대형 언어 모델(GLMs 또는 LLMs)은 광범위한 일반 지식과 인간과 유사한 추론 능력을 갖추고 있어 LLM-기반 에이전트를 구성하는 데 중요한 역할을 합니다. 이 설문에서는 LLM 기반 에이전트를 외부 도구와 상호작용하고 가정환경과 같은 환경에서 기능하는 능동적인 구성요소로 정의합니다. LLM 기반 에이전트는 행위자, 계획자, 평가자 및 동적 모델과 같은 태스크 독립적 LMPCs와, 언어화 도구와 같은 태스크 종속적 LMPCs로 구성됩니다.

- **Performance Highlights**: 이 설문의 목표는 기존 LLM 기반 에이전트 워크플로우에 대한 이해를 높이고, 복잡한 에이전트를 구성하기 위한 워크플로우 수준 및 구성요소 수준 구현의 재사용 및 적응을 가능하게 하며, 기존 워크플로우의 수정 및 확장을 간소화하는 것입니다. 표2에 다양한 모델(ReAct, Reflexion, Tree-of-Thoughts 등)에 이러한 구성 요소가 통합된 사례가 나와 있습니다.



### 3D-MolT5: Towards Unified 3D Molecule-Text Modeling with 3D Molecular Tokenization (https://arxiv.org/abs/2406.05797)
Comments:
          18 pages

- **What's New**: 최근 3D 구조 정보를 통합하여 분자와 언어를 동시에 모델링할 수 있는 3D-MolT5라는 새로운 프레임워크가 제안되었습니다. 이는 기존의 3D 정보를 다루는데 있어 부족했던 문제를 해결하고, 분자 3D 구조와 1D 분자 서열을 통합하는 기능을 갖추고 있습니다.

- **Technical Details**: 3D-MolT5는 3D 분자 지문 (Fingerprint)을 이용해 세부적인 3D 하위 구조 표현을 특화된 3D 토큰(vocabulary)으로 매핑합니다. 이를 통해 3D 정보와 1D 분자 서열, 텍스트 시퀀스를 통합된 아키텍처 안에서 함께 인코딩할 수 있습니다. 또한, 1D와 3D의 공동 사전 학습을 도입하여 다양한 모달리티(modality)를 함께 이해할 수 있는 모델로 발전시켰습니다. 사전 학습 후 다수의 다운스트림 데이터셋에 대한 인스트럭션 튜닝을 통해 분자 속성 예측, 분자 캡셔닝, 텍스트 기반 분자 생성 등의 작업에서 성능을 입증하였습니다.

- **Performance Highlights**: 다양한 분자 속성 예측, 분자 설명 생성, 텍스트 기반 분자 생성 작업에서 기존 방법들보다 뛰어난 성능을 보였습니다. 특히 PubChemQC, QM9, PubChem 데이터셋에서 우수한 결과를 나타냈습니다. 3D-MolT5는 3D 정보에 의존하지 않은 작업에서도 높은 성능을 보였습니다.



### Gentle-CLIP: Exploring Aligned Semantic In Low-Quality Multimodal Data With Soft Alignmen (https://arxiv.org/abs/2406.05766)
- **What's New**: Gentle-CLIP은 반지도 학습(semi-supervised learning)을 통해 다중 모달(multimodal) 데이터를 정렬하는 새로운 방법을 제안합니다. 기존의 CLIP 모델을 개선하여 다중 모달 데이터에서 충분한 일치 데이터를 확보하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: Gentle-CLIP은 CLIP을 기반으로 한 새로운 반지도 학습 기술을 소개합니다. 이 모델은 만곡 정렬 문제(manifold matching problem)로 전환하여 넓은 범위의 비일치 다중 모달 데이터에서 암묵적 의미 정렬(semantic alignment)을 탐구합니다. 주요 구성 요소는 다음과 같습니다:

- 의미 밀도 분포 손실(semantic density distribution loss): 암묵적 의미 정렬 정보를 제한된 감독쌍(supervised pairs)만으로 효과적으로 발견하도록 설계되었습니다.
- 다중 커널 최대 평균 차이(multi-kernel maximum mean discrepancy): 이 기법을 사용하여 모달리티 간의 표현 차이를 줄이며, 자가 감독 대조 손실(self-supervised contrastive loss)을 도입하여 표현 분포의 안정성을 보장합니다.
- CLIP 대조 손실(contrastive loss): 일치된 감독 데이터에 적용하여 부정적 최적화를 방지합니다.

- **Performance Highlights**: 단백질, 원격 센싱, 일반 비전-언어 분야에서 폭넓은 실험을 수행한 결과, Gentle-CLIP은 기존의 반지도 학습 방법을 능가하는 성능을 보였습니다. 이를 통해 다양한 특화된 분야에서의 적용 가능성을 입증하고, 제한된 감독쌍 없이도 의미 정렬을 효과적으로 수행할 수 있음을 확인했습니다.



### EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models (https://arxiv.org/abs/2406.05756)
Comments:
          Accepted by ACL 2024 Main

- **What's New**: 최근 빠르게 발전하고 있는 대형 비전-언어 모델(Large Vision-Language Models, LVLMs)이 체화된(embodied) 태스크에서의 가능성을 보여주고 있습니다. 그러나 공간적 이해를 평가하기 위한 적절한 벤치마크가 없어서 LVLMs와 체화된 지능 사이의 간극을 알기 어려웠습니다. 이를 해결하기 위해 EmbSpatial-Bench라는 벤치마크를 개발했습니다. 또한 LVLMs의 공간 이해 능력을 향상시키기 위해 EmbSpatial-SFT라는 인스트럭션 튜닝(instruction-tuning) 데이터셋을 제시했습니다.

- **Technical Details**: EmbSpatial-Bench는 체화된 3D 씬에서 자동으로 생성되며, 자기중심적 시점(egocentric perspective)에서 6가지 공간 관계(above, below, left, right, close, far)를 다룹니다. 이 벤치마크는 MP3D, AI2-THOR, ScanNet과 같은 3D 씬에서 수집된 이미지를 사용하여 다지선다형 질문으로 구성됩니다. 실험 결과, 현재의 LVLMs, 예를 들어 GPT-4V, Qwen-VL-Max는 공간적 이해 능력이 부족함을 보여줍니다. 이를 개선하기 위해 EmbSpatial-SFT 데이터셋을 구축하여 LVLMs의 성능을 향상시켰습니다.

- **Performance Highlights**: EmbSpatial-Bench를 기반으로 다양한 LVLMs를 평가한 결과, 현재 LVLMs는 체화된 환경에서의 공간적 이해 능력이 부족하다는 것을 확인했습니다. EmbSpatial-SFT로 튜닝된 LVLMs는 다양한 시나리오에서 향상된 공간 인지 능력을 일관되게 보여주었습니다. 또한 EmbSpatial-Bench는 3D 씬에서 평가를 진행하여 보다 현실적이고 정확한 평가를 가능하게 만들었습니다.



### Flow of Reasoning: Efficient Training of LLM Policy with Divergent Thinking (https://arxiv.org/abs/2406.05673)
- **What's New**: 본 논문에서는 새로운 대규모 언어 모델(LLM) 학습 방법인 Flow of Reasoning(FoR)을 제안합니다. FoR은 최소한의 데이터로 다단계 추론 문제에서 다양한 솔루션을 생성하도록 합니다. 이를 통해 인간의 창의성과 문제 해결에서 중요한 발산적 사고를 기계에서도 가능하게 합니다. 이 방법은 Markovian 흐름을 활용하여 초기 상태에서 말단 상태로의 다단계 모델 추론을 정의하고, 이를 기반으로 GFlowNet 접근 방식을 적응시켜 다양한 추론 경로를 학습합니다.

- **Technical Details**: FoR은 LLM을 정책(policy)으로 학습시키기 위해 Markovian 흐름을 활용합니다. 다단계 추론 문제를 초기 상태에서 말단 상태로 가는 흐름으로 정의하며, 이 중간 상태를 거치면서 다양한 추론 경로를 샘플링합니다. 이를 위해 GFlowNet 접근 방식에서 비정규화 보상에 비례하도록 추론 경로를 샘플링하는 정책 목표를 설정하며, 효과적인 온-오프 정책 탐색 전략을 사용해 최소한의 데이터로 정책을 학습합니다. 이를 통해 GFlowNets의 다양한 다운스트림 응용성을 확장합니다.

- **Performance Highlights**: 실험 결과, FoR은 최소한의 학습 데이터(예: 15개의 예제)로도 현재 최첨단 방법들에 비해 뛰어난 성능을 보이는 다양한 고품질 솔루션을 생성할 수 있음을 보였습니다. 세 가지 대표적인 과제인 블록스 월드(BlocksWorld)에서의 물리적 추론, Game24에서의 수학 퍼즐 해결, PrOntoQA에서의 논리적 추론에서 20%에서 85%의 성능 향상을 보였습니다. 또한, FoR의 핵심 설계 요소들이 강력하고 효과적이라는 것을 확인했습니다.



### A Superalignment Framework in Autonomous Driving with Large Language Models (https://arxiv.org/abs/2406.05651)
Comments:
          6 pages, 5 figures, ieeeiv24

- **What's New**: 최근 몇 년간 대형 언어 모델(LLM) 및 다중 모달 대형 언어 모델(MLLM) 분야에서 자율 주행에 관한 중요한 진전이 있었습니다. 이 연구는 자율 주행 차량에서 민감한 정보를 보호하기 위한 새로운 보안 프레임워크를 도입했습니다. 다중 에이전트 LLM 접근 방식을 사용하여 차량과 클라우드 기반 LLM 간의 상호 작용을 검열하고, 불필요한 질의를 필터링하며, 데이터 유출 방지 및 인공지능 산출물이 운전 규정을 준수하도록 보장하는 메커니즘을 포함하고 있습니다.

- **Technical Details**: 이 연구는 LLM이 차량 데이터(정확한 위치, 이미지, 도로 상태 등)를 처리하는 과정에서 발생할 수 있는 보안 문제를 해결하고자 합니다. 제안된 프레임워크는 차량과 클라우드 LLM 간의 상호 작용을 제어하기 위한 안전한 방어벽 역할을 합니다. 연구는 운전 안전, 토큰 사용, 프라이버시, 인간 가치 정렬 측면에서 대형 언어 모델을 기반으로 한 11가지 자율 주행 방법을 분석했습니다. 성능 평가에는 Q&A 테스트와 nuScenes-QA 데이터셋 조각을 사용하여 프레임워크의 효과를 입증했습니다.

- **Performance Highlights**: 제안된 보안 프레임워크는 자율 주행 차량에서 민감한 정보 유출을 방지하는 데 효과적이며, LLM의 출력이 운전 규정을 준수하고 인간의 가치와 정렬되도록 검증합니다. 연구는 gpt-35-turbo와 llama2-70b LLM 백본 간의 다양한 결과를 비교하여 프레임워크의 성능을 입증했습니다.



### Separating the "Chirp" from the "Chat": Self-supervised Visual Grounding of Sound and Languag (https://arxiv.org/abs/2406.05629)
Comments:
          Computer Vision and Pattern Recognition 2024

- **What's New**: DenseAV는 비디오만을 통해 고해상도, 의미적으로 의미 있는, 오디오-비주얼(AV) 정렬 피처(feature)를 학습하는 새로운 듀얼 인코더(dense encoder) 그라운딩 아키텍처를 도입했습니다. 이 시스템은 명확한 위치 지정 감독 없이도 단어의 '의미'와 소리의 '위치'를 발견할 수 있습니다. 또한, 두 가지 유형의 연관성을 자동으로 발견하고 구분합니다.

- **Technical Details**: DenseAV는 새로운 multi-head feature aggregation 연산자를 사용하여, 밀집된 이미지와 오디오 표현을 대조 학습(contrastive learning)을 통해 직접 비교합니다. 이를 통해 DenseAV는 음성과 비주얼 신호 간의 높은 품질의 지역 표현을 학습합니다. 또한, DenseAV는 두 개의 새로운 데이터셋을 도입해 음성과 소리 기반의 의미 분할(semantic segmentation)을 평가합니다. 이 데이터셋은 ADE20K 데이터셋에서 제공하는 고품질 분할 마스크를 기반으로 구축되었습니다.

- **Performance Highlights**: DenseAV는 음성과 소리 기반의 의미 분할에서 이전의 최첨단 기술인 ImageBind를 크게 능가합니다. 또한, DenseAV는 동일한 작업에서 ImageBind의 절반 이하의 매개변수만을 사용하면서 뛰어난 성능을 보입니다. 이로 인해 DenseAV는 새로운 소리와 자원이 적은 언어에 대한 적용 가능성을 보유하게 됩니다.



### Can Prompt Modifiers Control Bias? A Comparative Analysis of Text-to-Image Generative Models (https://arxiv.org/abs/2406.05602)
- **What's New**: 최신 연구는 Stable Diffusion, DALL-E 3, Adobe Firefly 같은 주요 텍스트-이미지 생성 모델에서 사회적 편견이 존재하고 증폭되는 과정을 분석했습니다. 이 연구는 이러한 AI 기술들이 성별, 인종, 지리적, 문화적 편향을 어떻게 인코딩하는지에 대한 세부적인 분석을 통해 편향을 제어하기 위한 프레임워크를 제안합니다. 이 연구는 텍스트-이미지 생성 모델의 편향을 드러내고, 미래의 연구를 위한 편향 제어 방법론을 제공합니다.

- **Technical Details**: 연구는 주요 텍스트-이미지 모델을 대상으로 기초 프롬프트(base prompts)와 수정자(modifiers)를 조합하여 성별, 인종, 지리적, 종교/문화 편향을 분석했습니다. 세 가지 주요 단계를 통해 연구를 진행했으며, 처음엔 각 모델의 표준 프롬프트를 사용하여 편향을 식별하고, 두 번째는 프롬프트 수정자를 사용하여 편향을 조정하는 방법을 탐구했으며, 마지막으로 프롬프트 구조가 생성된 이미지에 미치는 영향을 평가했습니다. 연구를 통해 편향 민감성 분류법(taxonomy)을 제안하고, prompt engineering을 통한 편향 조절의 가능성과 한계를 탐색했습니다.

- **Performance Highlights**: Adobe Firefly 모델은 성별 및 인종 표현에서 더 균형 있는 결과를 나타냈으며, 이는 내부적인 편향 완화 접근이 다르다는 것을 시사합니다. 반면, Stable Diffusion과 DALL-E 3 모델은 특정 문화적 고정관념에 따라 이미지를 생성하는 경향이 강했습니다. 연구는 다양한 프롬프트 구성의 영향을 통해 편향 제어의 복잡성을 강조하며, prompt sequencing이 편향 제어에 미치는 중요성을 부각했습니다.



### Automata Extraction from Transformers (https://arxiv.org/abs/2406.05564)
- **What's New**: 최근 Transformer 모델을 형식 언어(formal language) 처리 과정에서 이해하는 방법에 대해 연구가 진행되었습니다. 이 논문에서는 Transformer 모델을 결정적 유한 상태 자동자(deterministic finite-state automaton, DFA)로 해석하는 새로운 자동자 추출 알고리즘을 제안하였습니다. 기존에는 순환 신경망(Recurrent Neural Networks, RNN)에서 주로 쓰이던 이 자동자 추출 방법을 Transformer 모델에 적용하여, 모델의 내부 작동 메커니즘을 설명하고자 합니다.

- **Technical Details**: 제안된 방식은 Transformer 모델을 블랙박스 시스템으로 간주하고, 내부 잠재 표현의 변환 과정을 추적합니다. 이를 통해 Transformer 모델을 결정적 연속 상태 자동자(deterministic continuous state automaton, DCSA)로 시뮬레이션한 후, L* 알고리즘을 적용하여 결정적 유한 상태 자동자를 추출하게 됩니다. 이러한 과정을 통해, 형식 언어를 처리하는 Transformer 모델의 구조를 이해하고자 하였습니다.

- **Performance Highlights**: BERT 등 Encoder만을 사용하는 Transformer 모델에서 형식 언어를 학습한 후, 제안된 방법의 효과를 다양한 실험을 통해 확인하였습니다. 특히, 자동자 형태로 추출된 결과들을 분석함으로써 Transformer 모델의 투명성과 해석 가능성(interpretability)을 높이는 데 성과를 거두었습니다.



### Autoregressive Diffusion Transformer for Text-to-Speech Synthesis (https://arxiv.org/abs/2406.05551)
- **What's New**: 이번 연구에서는 오디오 생성 작업을 위해 ARDiT(Autoregressive Diffusion Transformer)을 제안합니다. 이는 오디오를 불연속 기호(discrete symbols)로 인코딩하는 기존 방식 대신, 연속 공간(continuous space)에서 벡터 시퀀스로 인코딩해 기존 모델들을 뛰어넘는 성능을 자랑합니다.

- **Technical Details**: ARDiT는 오디오를 연속 벡터 시퀀스로 변환한 후, 디퓨전 트랜스포머(diffusion transformer)를 이용해 이를 오토레그레시브 방식으로 생성합니다. 이는 텍스트를 오디오로 변환하는 'Zero-shot' Task에서 뛰어난 성능을 발휘합니다. 또한, IKL(Integral Kullback-Leibler) divergence를 사용한 distillation 기법이 확장되었습니다.

- **Performance Highlights**: {'Text-to-Speech': 'ARDiT는 기존 최첨단 모델들과 견줄만한 성능을 보이거나 이를 초과하는 결과를 보였습니다.', 'Quality and Reconstruction': '고비트레이트의 연속 스피치 벡터 방식은 거의 완벽에 가까운 재구성을 가능하게 했습니다.', 'Speed': '특정 모델은 평가 스텝 당 170ms의 24kHz 오디오를 생성하며, 성능 저하가 거의 없습니다.'}



### Exploring the Benefits of Tokenization of Discrete Acoustic Units (https://arxiv.org/abs/2406.05547)
Comments:
          Interspeech 2024

- **What's New**: 자연어 처리(NLP) 작업에서 기본 어휘 단위를 더 큰 가변 길이 단위로 결합하는 토큰화 알고리즘은 표준이 되어 왔습니다. 하지만 이러한 아이디어는 음소나 디스크리트 오디오 단위(Discrete Acoustic Units, DAUs)와 같은 어휘에 대해 잘 적용되지 않았습니다. 이 논문에서는 음소와 DAUs의 토큰화가 더욱 중요해지고 있다는 점을 강조하며, 이를 통해 성능, 훈련 및 추론 속도 측면에서 중요한 개선을 이룰 수 있음을 보여줍니다.

- **Technical Details**: 토큰화 알고리즘으로는 Byte Pair Encoding(BPE)를 채택하여 자주 등장하는 요소 쌍을 반복적으로 결합함으로써 새로운 어휘를 도출하는 방법을 사용했습니다. 이 방법을 통해 긴 시퀀스를 압축하면서도 더 큰 어휘집을 생성할 수 있습니다. 우리는 세 가지 예측 작업에 대해 이러한 방법의 이점을 문서화하였습니다: 1) grapheme-to-phoneme (G2P) 변환, 2) 텍스트에서 오디오 단위 예측 (G2DAU), 3) 음성 언어 모델을 사용한 오디오 생성 (SpeechLM).

- **Performance Highlights**: 실험 결과, BPE를 적용한 후 모든 작업에서 성능과 속도 측면에서 상당한 개선이 있음을 확인했습니다. 최신 알고리즘인 BPE를 사용하여 음소 및 DAU의 시퀀스 길이를 줄이고, 데이터 불균형 문제를 완화하며, 자가회귀 모델(autoregressive model)에서 정확도를 증가시켰습니다. 이를 통해 NLP 분야 외에도 오디오 및 음향 신호 처리에서 토큰화 알고리즘의 광범위한 적용 가능성을 제안합니다.



### A Fine-tuning Dataset and Benchmark for Large Language Models for Protein Understanding (https://arxiv.org/abs/2406.05540)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 자연어 처리(NLP)에서 보여준 성공을 단백질 시퀀스 이해에도 적용할 수 있을지에 대한 문제를 제기합니다. 연구진은 이 문제를 해결하기 위해 단백질 시퀀스와 설명 텍스트를 연결하는 'ProteinLMDataset'을 소개했습니다. 이 데이터셋은 모델의 자가지도 학습(self-supervised learning) 및 지도형 미세 조정(supervised fine-tuning)을 위해 설계되었습니다. 또한, 단백질 이해 능력을 평가하는 최초의 벤치마크 데이터셋인 'ProteinLMBench'도 함께 제공됩니다.

- **Technical Details**: ProteinLMDataset은 자가지도 학습을 위한 17.46십억 토큰과 미세 조정을 위한 893,000개의 지시 사항을 포함합니다. 벤치마크 데이터셋인 ProteinLMBench는 944개의 수작업으로 검증된 객관식 질문들로 구성되어 있습니다. 이 데이터셋은 단백질 시퀀스와 관련된 내용을 다양한 언어로 제공합니다. 기존의 단백질 데이터셋과는 달리, 이 데이터셋은 단백질 시퀀스와 텍스트 설명을 무결하게 통합하여 LLM을 효과적으로 훈련하고 평가할 수 있는 토대를 마련합니다.

- **Performance Highlights**: ProteinLMDataset에 사전훈련(pretraining) 및 미세 조정(fine-tuning)된 대형 언어 모델인 InternLM2-7B는 ProteinLMBench에서 GPT-4를 능가하는 최고 정확도 점수를 기록했습니다. 이는 제시된 데이터셋과 벤치마크가 LLM의 단백질 이해 능력을 크게 향상시킬 수 있음을 시사합니다.



### Online DPO: Online Direct Preference Optimization with Fast-Slow Chasing (https://arxiv.org/abs/2406.05534)
- **What's New**: 저자들은 네트워크의 대규모 언어 모델(LLM)의 인간 가치를 향상시키는 새로운 방법인 Online Fast-Slow chasing DPO (OFS-DPO)를 제안합니다. 또한, 교차 도메인 시나리오에서도 성능을 유지하기 위해 Cross domain Online Fast-Slow chasing DPO (COFS-DPO)를 소개하였습니다.

- **Technical Details**: OFS-DPO는 두 개의 동일한 모듈을 Low-rank Adaptive (LoRA)로 구성하고, 모델 간의 경쟁을 시뮬레이션하여 빠른 적응을 촉진합니다. 이 방법은 지속적 학습의 후회를 상한선으로 도출하고, 새로운 정규화 항을 통해 학습을 유도합니다. 또한, COFS-DPO는 LoRA 모듈 결합 전략을 사용하여 기억 상실을 줄이고, 역사적 정보를 활용하여 지속적인 가치 정렬을 달성합니다.

- **Performance Highlights**: 실험 결과 OFS-DPO는 도메인 내 정렬에서는 기존 DPO를 능가하였으며, COFS-DPO는 교차 도메인 지속 학습 시나리오에서 탁월한 성능을 보였습니다. 특히, 요약 작업에서는 경쟁 기반라인을 크게 능가하는 결과를 보였습니다.



### Verbalized Probabilistic Graphical Modeling with Large Language Models (https://arxiv.org/abs/2406.05516)
- **What's New**: 이 연구는 LLM(Large Language Model)이 복잡한 합성 추론(compositional reasoning)을 수행할 때, 불확실성을 모델링하고 잠재 구조를 학습할 수 있도록 Bayesian prompting 접근법을 도입하였습니다. 이를 통해 LLM은 광범위한 데이터 학습 없이도 Bayesian inference(베이지안 추론)을 수행할 수 있게 됩니다.

- **Technical Details**: 이 접근법은 언어화된 확률 그래프 모델(verbalized Probabilistic Graphical Model, vPGM)을 사용하여 LLM을 Bayesian 원칙에 따르도록 유도합니다. 기존의 Bayesian 구조 학습 방법이 대량의 데이터와 전문적 도메인 지식을 필요로 하는 반면, 이 방법은 LLM의 지식과 추론 능력을 활용하여 확률적 의존성을 효율적으로 추론합니다. PGM에서 잠재 변수와 그들의 확률적 의존성을 식별하고, 새로운 테스트 샘플 맥락에서 각 잠재 변수의 사후 확률분포를 추론합니다. 최종 예측은 이러한 사후 확률의 평균을 통해 근사됩니다.

- **Performance Highlights**: 여러 합성 추론 과제에 대해 우리의 모델을 평가한 결과, LLM의 자신감 유도 능력과 텍스트 생성 품질이 눈에 띄게 향상되는 것을 확인했습니다. 또한, 본 접근법은 복잡한 합성 추론 작업에서 불확실성을 모델링하는 능력이 탁월하다는 점을 보여줍니다.



### Mmm whatcha say? Uncovering distal and proximal context effects in first and second-language word perception using psychophysical reverse correlation (https://arxiv.org/abs/2406.05515)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 이 연구는 음성 지각에서 주변의 음높이(pitch)와 말하기 속도(rate)가 어떻게 영향을 미치는지, 특히 제2언어(L2) 화자에게 있어서 그 상호작용을 조사합니다. 영어와 프랑스어 화자를 대상으로 하여 초분석적인(reverse-correlation) 접근법을 사용하여 음성 지각에 있어서 조음 프로필(prosodic profiles)을 재구성했습니다.

- **Technical Details**: 연구에서는 250개의 실험 시도(trials)를 통해 각 시도의 기초 녹음을 다양한 무작위 프로필의 음높이와 말하기 속도로 조작하고, 참가자에게 어떤 단어가 들렸는지 판단하게 했습니다. 실험에는 n=25명의 영어 모국어(L1) 화자와 프랑스어 모국어 화자가 참가했으며, 이들은 영어(/i/-/I/)와 프랑스어(/u/-/y/) 모음쌍을 사용하여 실험했습니다.

- **Performance Highlights**: 연구 결과, 모국어 화자(L1)와 제2언어 화자(L2) 모두 비슷한 음성지각 특성을 가지고 있었으며, 음성 주변의 음높이와 말하기 속도는 서로 상반된 영향을 미쳤습니다. 구체적으로, 목표 단어 전 0.2초의 인접 효과(proximal effect)와 1초 전의 대조 효과(contrastive effect)가 발견되었습니다.



### Representation Learning with Conditional Information Flow Maximization (https://arxiv.org/abs/2406.05510)
Comments:
          16 pages, accepted to ACL 2024 (main conference)

- **What's New**: 이번 연구는 조건부 정보 흐름 최대화(Conditional Information Flow Maximization, CIFM)라는 정보 이론적 표현 학습 프레임워크를 제안합니다. 이 프레임워크는 입력 데이터와 대상 작업을 위한 소음 불변 충분 표현(noise-invariant sufficient representations)을 추출하는 것을 목표로 합니다.

- **Technical Details**: CIFM은 정보 흐름 최대화 정보 흐름 최대화(Information Flow Maximization, IFM)와 조건부 정보 최소화(conditional information minimization, CIM) 원칙을 결합하여 구현됩니다. IFM 원칙은 입력-표현 및 표현-라벨 상호 정보를 최대화하여 보다 충분한 표현을 학습합니다. 이는 정보 병목(Information Bottleneck, IB) 접근법과 반대로 입력-표현 정보를 줄이는 대신, 해당 정보를 최대화하여 과도한 압축 문제를 피합니다. CIM 원칙은 잠재적 중복 기능의 부정적 영향을 완화하고 입력에서 소음 불변 기능을 유지하는 데 중점을 둡니다.

- **Performance Highlights**: 13개의 언어 이해 벤치마크 실험에서 CIFM은 분류(Classification)와 회귀(Regression) 성능을 크게 향상시켰습니다. CIFM은 RoBERTa 백본 모델에서 분류 및 회귀 작업에서 각각 +3.8% 및 +1.9%의 평균 성능 향상을 보였습니다. 또한, CIFM은 모델의 일반화 능력(out-of-distribution 및 데이터 제한 시나리오), 무작위 및 적대적 노이즈에 대한 견고성, 새로운 작업으로의 전이 가능성을 증명했습니다.



### MLLM-SR: Conversational Symbolic Regression base Multi-Modal Large Language Models (https://arxiv.org/abs/2406.05410)
Comments:
          13 pages,

- **What's New**: 새로운 연구에서는 자연어 지시만으로 특정 요구를 충족하는 표현을 생성할 수 있는 대화형 symbolic regression 방법인 MLLM-SR을 제안합니다. 기존 symbolic regression 방법들은 주어진 관찰 데이터에서 직접 표현식을 생성하나, MLLM-SR은 추가된 선행 지식을 잘 이해하고 이를 바탕으로 올바른 표현식을 생성할 수 있는 능력을 갖췄습니다.

- **Technical Details**: MLLM-SR은 multi-modal large language models(MLLMs)를 기반으로 하며, 관찰 데이터를 하나의 modality (이미지, 비디오 등)로, 텍스트(표현식을 구성하는 심볼을 포함한)를 또 다른 modality로 간주합니다. 모델 훈련 과정에서 우선 대형 언어 모델(LLM)과 SetTransformer의 파라미터를 고정하고 Fully Connected Layer를 통해 관찰 데이터 특징을 문자 특징 공간으로 매핑합니다. 이후 LLM의 파라미터를 풀어 end-to-end로 MLLM-SR을 훈련 시킵니다.

- **Performance Highlights**: 번 연구는 Nguyen 데이터셋 실험을 통해 MLLM-SR이 fitting 성능 면에서 최첨단 기준들을 능가함을 입증했습니다. 특히 MLLM-SR이 natural language instructions에 추가된 선행 지식을 잘 이해하고, 이를 효과적으로 반영하여 정확한 표현식을 생성할 수 있음을 보여주었습니다.



### M3GIA: A Cognition Inspired Multilingual and Multimodal General Intelligence Ability Benchmark (https://arxiv.org/abs/2406.05343)
- **What's New**: 최근 멀티모달리티 대형 언어 모델(MLLMs)이 다양한 복잡한 작업에서 뛰어난 능력을 보여주고 있는 가운데, 이러한 모델들이 궁극적으로 인간 지능을 모방할 수 있을지에 대한 논의가 증가하고 있습니다. 기존 벤치마크들은 주로 객체의 속성 식별 정확도와 같은 작업 성능 평가에 중점을 두고 있습니다. 이를 넘어 인지과학을 활용하여 MLLMs의 지능을 평가하는 연구는 미흡한 실정입니다. 이를 해결하기 위해, M3GIA라는 다중 언어 및 다중 모달 벤치마크를 소개하여 MLLMs의 일반 지능 능력을 평가합니다. 이는 잘 알려진 Cattell-Horn-Carrol(CHC) 인텔리전스 모델의 다섯 가지 주요 인지 요인을 기반으로 설계되었습니다.

- **Technical Details**: M3GIA는 5개의 주요 인지 요인을 기반으로 모델의 인지 능력을 평가합니다. 이 모델은 영어 외에도 중국어, 프랑스어, 스페인어, 포르투갈어, 한국어 등 다양한 언어를 포함하여, 언어가 MLLMs의 인지 능력에 미치는 영향을 탐구합니다. 모든 데이터를 해당 문화적 배경에서 수집하여 영어 중심 편향을 방지하였습니다. M3GIA는 인간 참여자로부터 대규모 데이터를 수집하여, 가장 발전된 MLLM이 영어에서는 인간 지능의 하한을 도달했음을 보여주지만, 다른 언어에서는 뚜렷한 격차가 있음을 나타냅니다.

- **Performance Highlights**: 최신 MLLMs는 영어에서 인간 지능의 하한 성능을 달성했지만, 다른 다섯 개 언어에서는 성능 격차가 뚜렷합니다. 또한, 한 인지 영역에서 뛰어난 성능을 보이는 모델이 다른 영역에서도 우수한 성능을 보이는 '승자 독식 현상'을 관찰할 수 있었습니다. 이 패턴은 인간 지능 연구에서 발견된 일반 지능 능력(GIA)과 일치합니다.



### LoCoCo: Dropping In Convolutions for Long Context Compression (https://arxiv.org/abs/2406.05317)
- **What's New**: 이번 논문은 긴 문맥 시퀀스를 처리하는 대형 언어 모델(LLMs)에서 메모리 문제를 해결하기 위해 Dropping In Convolutions for Long Context Compression(LoCoCo)이라는 새로운 접근 방식을 제안합니다. LoCoCo는 고정 크기 Key-Value(KV) 캐시를 사용해 효율성을 향상시키며, 기존 방법과 달리 데이터 기반의 적응형 융합 기술을 사용해 문맥 정보를 최소로 손실하고 정확한 주의(attention) 모델링을 보장합니다.

- **Technical Details**: LoCoCo는 이전의 KV 쌍과 새로운 토큰을 융합해 컨텍스트 손실을 최소화하고, 동적 가중치 계산을 위해 1차원 컨볼루션 커널을 사용해 각 KV 캐시 슬롯에서의 혼합 가중치를 계산합니다. 이 방법은 기존 LLM 아키텍처와의 광범위한 호환성을 고려해 설계되었으며, 간단하게 '드롭인' 형태로 통합할 수 있고 최적화 오버헤드가 거의 없습니다. LoCoCo는 오토리그레시브 생성(autoregressive generation)의 연속성을 활용해 윈도우를 이동하고, 컨볼루션을 통해 시퀀스의 정역 준칙(stationary inductive bias)을 강화합니다.

- **Performance Highlights**: 실험 결과, LoCoCo는 다양한 컨텍스트 길이에서도 일관되게 뛰어난 성능을 유지하며, 추론 및 미세 조정(fine-tuning) 단계에서 높은 컨텍스트 압축 비율을 달성했습니다. 적응형 튜닝 시에는 4K 컨텍스트 길이를 32K로 확장하고도 고정 크기 512 KV 캐시로 동일한 성능을 유지했고, 추론 단계에서는 최대 3482개 토큰을 128 크기 KV 캐시로 압축하면서도 전체 시퀀스와 유사한 성능을 나타냈습니다.



### A model of early word acquisition based on realistic-scale audiovisual naming events (https://arxiv.org/abs/2406.05259)
Comments:
          22 pages, 4 figures, journal article, submitted for review

- **What's New**: 본 연구는 12개월 미만 유아의 초기 단어 인식 능력 발달 메커니즘을 이해하기 위해 통계적 학습(statistical learning)의 역할을 조사했습니다. 독립 표본(raw speech와 픽셀 수준의 시각 정보)에서 통계적 규칙성을 통해 단어를 배우는 모델을 사용하여 학습을 시뮬레이션하였습니다. 연구는 유아가 실제 접하는 객체 명명 사건의 수를 모방하여, 그 모델이 단어를 인식하고 시각 객체와 연관시킬 수 있는지를 확인했습니다.

- **Technical Details**: 이 모델은 비지시적인(raw) 음성 데이터와 픽셀 수준의 이미지를 입력으로 받아, 데이터 내의 통계적 규칙성만으로 학습합니다. 초기 6개월 동안은 청각 및 시각 정보만을 통해 학습하고, 이후 8, 10, 12개월 터울로 명명 이벤트를 통해 연관 학습을 진행했습니다. 모델은 SpokenCOCO 데이터셋으로 훈련되었으며, 실험 연구에서 보고된 유아 언어 입력 통계에 맞추어 데이터셋을 설계했습니다.

- **Performance Highlights**: 결과에 따르면, 모델은 유아와 유사한 어휘 성장률을 보이며, 단어 인식과 의미 매핑 작업을 성공적으로 수행했습니다. 이는 강력한 언어적 사전 지식이나 선천적 편향적 학습 없이도 통계적 학습을 통해 초기 단어 인식이 가능함을 시사합니다.



### CPLIP: Zero-Shot Learning for Histopathology with Comprehensive Vision-Language Alignmen (https://arxiv.org/abs/2406.05205)
- **What's New**: 이번 연구에서는 Comprehensive Pathology Language Image Pre-training (CPLIP)이라는 새로운 비지도 학습 기법을 제안합니다. 이 기법은 병리학 이미지와 텍스트의 정합성을 향상시켜 분류와 분할 같은 작업의 성능을 높입니다. 기존의 주석이 필요 없는 대규모 데이터세트를 활용하여 비전-언어(vl) 모델을 풍부하게 구성합니다. CPLIP는 병리학 특정 사전(dictionary)을 구축하고, 언어 모델을 사용해 이미지에 대한 텍스트 설명을 생성합니다. 이어서 사전 훈련된 모델을 이용해 각 텍스트에 대한 관련 이미지를 검색하고, contrastive learning(대조 학습)을 통해 둘 사이의 복잡한 상호작용 개념을 맞춥니다. 이 모델은 여러 병리학 작업에서 평가되어, 기존 방법들을 능가하는 zero-shot 학습 시나리오에서의 개선된 성능을 보여줍니다.

- **Technical Details**: CPLIP는 병리학적인 조건에 대한 다양한 텍스트 설명과 해당 조건에 대한 다양한 병리 이미지들을 포함하는 '포괄적(comprehensive)' 접근 방식을 채택했습니다. 먼저 병리학 특정 사전을 엮고, 기존 VL 모델을 사용해 각 이미지에 적합한 설명을 선택합니다. 이후 GPT-3를 활용해 각 조건에 대한 원인을 분석하고 세부 설명을 생성합니다. 이런 텍스트와 이미지를 기반으로 생성된 데이터셋은 CLIP 모델을 fine-tuning하는 데 활용됩니다. 최종적으로 모델의 임베딩(Embedding)을 조정하고, 유사한 개념들을 맞추고 불일치하는 것들은 떨어뜨리는 방식으로 모델을 학습시킵니다.

- **Performance Highlights**: CPLIP는 여러 병리학 작업에서 zero-shot 학습 능력을 크게 향상시켰습니다. 특히 해석 가능성과 견고성 면에서 기존 방법들을 능가하며, VL 모델의 새 기준을 제시합니다. 종합적인 텍스트 설명과 다중 시각적 개념을 채택하여 텍스트와 이미지 임베딩 정합성을 개선, 다양한 암 아형 분류와 조직 인식 작업에서 뛰어난 성능을 보였습니다. 이런 성과는 CPLIP의 접근 방식이 비전-언어 모델의 성능을 크게 향상시킬 수 있음을 증명합니다.



### Evaluating the Effectiveness of Data Augmentation for Emotion Classification in Low-Resource Settings (https://arxiv.org/abs/2406.05190)
Comments:
          The first author contributed significantly

- **What's New**: 이번 연구에서는 저자들은 적은 자원 데이터셋(low-resource dataset)을 사용한 다중 레이블 감정 분류 작업에서 다양한 데이터 증강(data augmentation) 기법의 효과를 평가했습니다. 특히, Back Translation이 오토인코더(autoencoder) 기반 접근 방식보다 뛰어난 성능을 보였으며, 학습 인스턴스 당 여러 예시를 생성하면 성능이 더욱 향상된다는 것을 발견했습니다. Back Translation은 가장 다양한 유니그램(unigram)과 트라이그램(trigram) 집합을 생성하는 것으로 나타났습니다.

- **Technical Details**: 이번 연구에서는 다양한 데이터 증강 기법을 사용하여 적은 자원 시나리오를 시뮬레이션했습니다. 이를 위해 최첨단 다중 레이블 감정 분류 모델을 사용하여 큰 데이터셋에 의사 레이블(pseudo-label)을 달았습니다. 그런 다음 다양한 생성 모델(generative model)을 사용하여 미리 설정된 레이블을 가진 추가 데이터를 합성했습니다. 이러한 결과로 얻어진 적은 자원 데이터셋과 증강된 데이터를 함께 사용하여 모델을 학습시키고 새로운 코퍼스에서의 성능을 평가했습니다.

- **Performance Highlights**: 성능 평가 결과, Back Translation이 오토인코더 기반 접근 방식보다 우수했을 뿐만 아니라 가장 다양한 유니그램과 트라이그램을 생성했습니다. 또한, 학습 인스턴스 당 여러 예시를 생성하면 성능이 더욱 향상되었음을 확인했습니다. 이러한 결과는 적은 자원 환경에서도 감정 분류 모델의 성능을 향상시키는 데 Back Translation이 유용함을 보여줍니다.



### The Factorization Curse: Which Tokens You Predict Underlie the Reversal Curse and Mor (https://arxiv.org/abs/2406.05183)
Comments:
          18 pages, 7 figures

- **What's New**: 최신 연구에서는 대규모 언어 모델이 여전히 겪고 있는 할루시네이션(hallucination) 문제와 그로 인한 정보 검색의 어려움을 다루고 있습니다. 연구자는 Reversal Curse(역순 저주)를 Factorization Curse(인수분해 저주)로 재구성하여, 모델이 다른 인수분해(factorizations)에서도 동일한 공동 분포(joint distribution)를 학습하지 못하는 실패로 정의했습니다. 이를 통해 새로운 WikiReversal 설정을 포함한 여러 통제 실험을 수행했습니다.

- **Technical Details**: 연구에서는 Next-Token Prediction Objective(다음 토큰 예측 목표)를 사용하는 대규모 언어 모델에서 인수분해 저주라는 고유한 실패가 발견되었습니다. 모델이 특정 순서의 토큰을 미리 보지 않는 한, 역순 토큰이나 단순한 양방향 주의(bidirectional-attention) 훈련조차 신뢰할 수 있는 정보 검색을 해결하지 못합니다. 따라서 특화된 데이터에 대한 미세 조정(finetuning) 접근법은 혼재된 결과를 제공할 수 있습니다.

- **Performance Highlights**: 다섯 가지의 복잡도가 다른 과제를 통해, 인수분해-불가지론적 목표(factorization-agnostic objectives)가 역순 저주를 상당히 완화할 수 있고, 향상된 지식 저장 및 계획 능력을 암시하는 유망한 경로를 밝혀냈습니다.



### Improving Alignment and Robustness with Circuit Breakers (https://arxiv.org/abs/2406.04313)
- **What's New**: 최근 인공지능(AI) 시스템이 해로운 행동을 하고, 적대적 공격에 취약하다는 문제점을 해결하기 위한 새로운 접근 방법이 제시되었습니다. 최신 표현 공학(representation engineering) 기술을 활용하여 해로운 출력을 발생시키는 모델의 응답을 '서킷 브레이커(circuit breakers)'로 즉시 중단시키는 방식입니다.

- **Technical Details**: 서킷 브레이커 접근법은 기존의 거부 훈련(refusal training)과 적대적 훈련(adversarial training)을 대체하며, 텍스트 전용 모델과 멀티모달 언어 모델 모두에 적용될 수 있습니다. 이 방식은 모델 내부 표현을 제어하여 해로운 출력을 중지시키는 방식으로, 추가적인 공격이 없는 상태에서도 모델의 유용성을 희생시키지 않습니다. 이 방법은 회로를 우회하여 모델이 해로운 출력을 생성하지 못하게 합니다. 서킷 브레이커 기법 중 하나인 Representation Rerouting(RR)은 특히 효과적이며, 추가 훈련(adversarial fine-tuning)이나 보조 '가드' 모델 없이도 높은 성능을 유지합니다.

- **Performance Highlights**: 실험 결과, 대표적인 서킷 브레이커 기술인 Representation Rerouting(RR)은 최신 대형 언어 모델(LLMs)의 무해성을 크게 향상시켰습니다. 특히, 예상치 못한 공격에 대한 견고함을 보여주었으며, 기존의 거부 훈련 및 적대적 훈련을 통해 얻을 수 있는 성능을 초과했습니다. Llama-3-8B-Instruct 모델에 서킷 브레이커를 통합하여 개발한 Cygnet 모델은 초반 성능을 초과하여 해로운 출력이 2등급 이하로 감소했습니다. 이로 인해 실세계 응용에서 안정적으로 사용할 수 있는 가능성이 크게 증가했습니다.



### RATT: A Thought Structure for Coherent and Correct LLM Reasoning (https://arxiv.org/abs/2406.02746)
- **What's New**: 새로운 논문에서는 복잡한 작업에서 기존 방법들이 정확성과 전략적 선택 간의 균형을 맞추지 못하는 문제를 해결하기 위해 Retrieval Augmented Thought Tree (RATT)를 제안합니다. RATT는 각 사고 과정에서 전반적인 논리적 타당성과 사실 정확성을 고려하여 계획 및 룩어헤드(lookahead)를 수행합니다.

- **Technical Details**: RATT는 Retrieval-Augmented Generation (RAG)와 대규모 언어 모델(LLM)의 전략 평가 능력을 통합하여 사고 트리 구조 내에서 가장 유망한 가지를 탐색합니다. 이를 통해 논리적 일관성과 결정 효율성을 크게 향상시킵니다. RATT는 다음과 같은 단계로 구현됩니다: (1) 사고 노드 생성, (2) 검색 및 선택, (3) RAG 수정 및 통합.

- **Performance Highlights**: 다양한 작업에서의 광범위한 실험을 통해 RATT 구조가 기존 방법들보다 사실 정확성과 논리 일관성에서 크게 우수함을 입증했습니다.




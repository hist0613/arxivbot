New uploads on arXiv(cs.CL)

### Optimizing Rare Word Accuracy in Direct Speech Translation with a Retrieval-and-Demonstration Approach (https://arxiv.org/abs/2409.09009)
- **What's New**: 이번 논문에서는 희소 단어(rare words) 번역의 정확성을 높이기 위한 새로운 접근 방식인 retrieval-and-demonstration 프레임워크를 제안합니다. 이 프레임워크는 이미 번역된 예시를 활용하여 직접적인 음성 번역 모델의 성능을 향상시키는 방식을 채택합니다.

- **Technical Details**: 제안된 접근법은 기존의 음성 번역(Direct Speech Translation, ST) 모델을 조정하여, 관련 예시를 검색하고 이를 통해 희소 단어를 번역할 때 발생하는 문제를 해결합니다. 이 모델은 retrieval 기법을 사용하여 음성-음성(speech-to-speech), 음성-텍스트(speech-to-text), 텍스트-텍스트(text-to-text) 간의 예시를 찾는 크로스 모달(retriever) 기능을 포함하고 있습니다. 또한, 일반적인 ST 코퍼스를 희소 단어 번역을 위한 평가 방법론으로 조정합니다.

- **Performance Highlights**: 이 연구에서 개발된 시스템은 gold examples를 사용할 경우 희소 단어 번역의 정확성을 기존 기준보다 17.6% 향상시키고, 검색된 예시(retrieved examples)를 사용할 경우에는 8.5% 향상시킵니다. 특히, 음성-음성 검색 방식이 다른 모달리티에 비해 더 높은 번역 품질과 희소 단어 번역 정확성을 보여주며, 보지 못한 화자에 대한 저항력도 향상됐습니다.



### E2MoCase: A Dataset for Emotional, Event and Moral Observations in News Articles on High-impact Legal Cases (https://arxiv.org/abs/2409.09001)
- **What's New**: 이번 연구에서는 E2MoCase라는 새로운 데이터셋을 소개하여, 법적 사건에 대한 미디어 보도가 공공의 의견을 어떻게 형성하는지를 분석합니다. 이 데이터셋은 감정, 도덕적 가치, 사건을 포괄적으로 분석할 수 있도록 설계되었습니다.

- **Technical Details**: E2MoCase는 감정 탐지(emotion detection), 도덕적 가치 식별(moral value identification), 사건 추출(event extraction)을 위한 고급 모델을 활용하여, 법적 사건이 뉴스 기사에서 어떻게 묘사되는지를 다차원적으로 분석합니다.

- **Performance Highlights**: 이 연구는 감정 톤(emotional tone), 도덕적 프레이밍(moral framing) 및 구체적인 사건을 포착하는 통합 분석 방식을 통해 법적 내러티브와 미디어 보도의 편향을 탐색하는 데 기여합니다.



### Exploring the Impact of Data Quantity on ASR in Extremely Low-resource Languages (https://arxiv.org/abs/2409.08872)
- **What's New**: 이 연구는 두 가지 멸종 위기 아우스트로네시아 언어인 아미스(Amis)와 시디크(Seediq)에 대한 저자원 자동 음성 인식(ASR) 시스템을 위한 데이터 증강(data augmentation) 기술의 효용성을 조사합니다. 새로운 데이터 선택 방식을 제안하여 다국어 코퍼스를 활용해 제한된 대상 언어 데이터의 부족을 보완하고자 했습니다.

- **Technical Details**: 이 연구에서는 언어 분류기를 활용하여 발화(utterance) 임베딩을 추출하고, 원 클래스 분류기(one-class classifiers)를 사용하여 음성과 음소적으로 대상 언어와 밀접한 발화를 식별합니다. 선택된 발화는 결정 점수에 따라 순위가 매겨집니다. 이를 통해 SSL-ASR 파이프라인에서의 데이터 선택을 개선하고자 했습니다. 연구에서는 One-class SVM, Isolation Forest 및 Deep SVDD와 같은 세 가지 원 클래스 분류기를 활용했습니다.

- **Performance Highlights**: 이 연구의 실험 결과, 제안된 접근 방식이 아미스와 시디크 두 언어 모두에서 ASR 성능을 크게 향상시켰다는 것을 보여주었습니다. 이러한 결과는 저자원 언어 ASR을 위한 교차 언어 전이 학습(cross-lingual transfer learning)을 통한 데이터 증강의 가능성과 잠재력을 강조합니다.



### AIPO: Improving Training Objective for Iterative Preference Optimization (https://arxiv.org/abs/2409.08845)
- **What's New**: 본 논문은 Iterative Preference Optimization (IPO)에서의 길이 착취(length exploitation) 문제를 심층적으로 분석하고, Agreement-aware Iterative Preference Optimization (AIPO)라는 새로운 훈련 목표를 제안합니다. 또한, 합성 데이터(synthetic data)를 활용한 IPO의 효과성을 입증하기 위한 실험을 통해 최첨단 성능을 기록했습니다.

- **Technical Details**: 본 연구는 합성 데이터 생성, 응답 생성, 선호 순위 매기기 및 사후 처리를 포함한 합성 데이터 큐레이션(synthetic data curation) 과정을 세밀하게 검토합니다. IPO 훈련 전략을 정의하고 다양한 구성에 대한 실험을 실시했으며, DPO 손실(loss)의 잠재적 문제점을 발견했습니다. 이를 해결하기 위해 AIPO라는 새로운 훈련 목표를 도입했습니다.

- **Performance Highlights**: MT-Bench, AlpacaEval 2.0 및 Arena-Hard 벤치마크에서 최첨단 성능을 달성했습니다. AIPO 훈련 목표를 통해 IPO에서의 효율성을 크게 향상시켰습니다.



### Your Weak LLM is Secretly a Strong Teacher for Alignmen (https://arxiv.org/abs/2409.08813)
Comments:
          20 pages

- **What's New**: 이번 논문은 자원 소모가 적은 약한 LLM(Weak LLM)을 활용하여 AI 정렬(Alignment)을 위한 피드백 생성을 탐구합니다. 기존의 방법들이 많은 인적 자원이나 고비용을 요구하는 반면, 약한 LLM은 더 자동화된 피드백을 제공하여 AI의 효율성을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구팀은 약한 LLM의 피드백을 평가할 수 있는 새로운 프레임워크를 제시했습니다. 이 프레임워크는 레이블이 있는 데이터와 없는 데이터를 혼합하여 활용하며, 레이블이 없는 삼중 데이터(x, y1, y2)에서 약한 LLM이 선호에 대한 피드백을 제공하게 됩니다. 이를 통해 최종적으로 목표 LLM의 정책을 훈련합니다.

- **Performance Highlights**: 놀랍게도, 약한 LLM(OPT-125M)을 사용하여 제공된 선호 피드백은 전체 인간 피드백과 비슷하거나 그 이상의 성능을 보여주었습니다. 피드백의 효과성은 모델의 크기에 큰 영향을 받지 않았으며, 약한 LLM이 더 큰 모델(GPT-4)보다 특정 작업에 있어 더 효과적인 피드백을 제공할 수 있다는 결과도 제시되었습니다.



### Exploring SSL Discrete Tokens for Multilingual ASR (https://arxiv.org/abs/2409.08805)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 연구는 자기 지도 학습(Self-supervised Learning, SSL)에 의해 생성된 이산 토큰을 다국어 자동 음성 인식(Automatic Speech Recognition, ASR) 과제에 적용하는 것을 목표로 하며, 기존 영어 중심의 연구에서 벗어나 다국어 상황에서의 가능성을 탐색합니다.

- **Technical Details**: 연구에서는 WavLM-Large, XLSR-53, EnCodec 세 가지 주요 음성 기초 모델을 사용하여 원시 오디오를 이산 토큰으로 변환한 후, 이를 활용하여 단일 언어 및 다국어 ASR 모델을 교육합니다. 이산 토큰은 특정 언어와 관계없이 7개 언어 도메인에서 ASR 작업의 효율성과 성능을 평가하기 위해 Multilingual Librispeech 데이터셋을 활용합니다.

- **Performance Highlights**: 실험 결과, 이산 토큰 기반 모델은 Fbank 특성을 사용한 모델과 비교하여 개발(dev) 세트에서 0.31%, 테스트(test) 세트에서 1.76%의 평균 단어 오류율(Word Error Rate, WER)을 감소시키며 특히 폴란드어 테스트 세트에서는 6.82%의 절대 WER 감소를 기록하였습니다.



### Exploring SSL Discrete Speech Features for Zipformer-based Contextual ASR (https://arxiv.org/abs/2409.08797)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 Self-supervised learning (SSL)을 기반으로 한 이산 음성 표현(discrete speech representations)을 WavLM 모델에서 추출하여, Zipformer-Transducer ASR 시스템에서 추가적인 교차 발화(acoustic context features) 특성으로 활용하였습니다.

- **Technical Details**: Gigaspeech 1000-hr 코퍼스를 사용하여, Fbank 특징을 이산 토큰 특징으로 교체하는 방법의 효용성을 검증했습니다. 이 연구는 교차 발화 맥락모델에서의 두 가지 접근법을 통해 내부 문맥과 상호 연동하여 성능을 개선하는 작업을 포함합니다.

- **Performance Highlights**: 최신 Zipformer-Transducer 시스템을 이용한 실험에서, 이산 토큰 특징 기반의 교차 발화 맥락 특성을 사용했을 때, 발화 내부 맥락만 사용할 때보다 통계적으로 유의미한 단어 오류율(Word Error Rate, WER) 감소(0.32%에서 0.41% 절대적인 감소)를 보였습니다. 개발 및 테스트 데이터에 대한 최저 WER은 각각 11.15%와 11.14%로 기록되었습니다.



### Optimizing Ingredient Substitution Using Large Language Models to Enhance Phytochemical Content in Recipes (https://arxiv.org/abs/2409.08792)
Comments:
          15 pages

- **What's New**: 이 연구는 계산 요리(computational gastronomy)라는 새로운 분야에서 과학적으로 지지되는 영양 목표와 조리 관행(culinary practices)을 일치시키기 위한 방법을 탐구합니다. 특히, 대형 언어 모델(large language models)을 사용하여 요리법의 재료 대체(ingredient substitutions)를 최적화하여 식사의 식물 화학물질(phytochemical) 함량을 향상시키는 방안을 제시합니다.

- **Technical Details**: 연구에서는 OpenAI의 GPT-3.5, DaVinci 및 Meta의 TinyLlama와 같은 모델을 조정(fine-tuned)하였습니다. 이를 위해 재료 대체 데이터를 사용하여 모델을 조정하였고, 이를 바탕으로 식물 화학물질의 함량을 높이는 대체 재료를 예측하고 이에 해당하는 강화된_recipe dataset_를 생성했습니다.

- **Performance Highlights**: 제안된 접근 방식은 재료 대체 과제에서 Hit@1 정확도를 향상시켜, 원래 GISMo 데이터셋에서 34.53%(±0.10%)에서 38.03%(±0.28%)로, 동일 데이터셋의 개선된 버전에서는 40.24%(±0.36%)에서 54.46%(±0.29%)로 증가시켰습니다. 이 연구를 통해 1,951개의 식물 화학물질이 풍부한 재료 쌍(pairings)과 1,639개의 독특한 요리법이 생성되었습니다. 그러나 건강 이점에 대한 주장을 내릴 때는 사전 임상(preclinical) 증거에 기반하고 있음을 유의해야 하며, 향후 연구에서는 임상 검증(clinical validation) 및 더욱 폭넓은 데이터셋을 포함해야 할 것입니다.



### Sign Language Sense Disambiguation (https://arxiv.org/abs/2409.08780)
Comments:
          LIMO2024 @ KONVENS 2024, 8 pages, 3 figures

- **What's New**: 이 연구는 독일 수화의 번역 시스템을 개선하기 위해 동의어의 명확성을 높이는 방법을 탐구합니다. 특히, 수화의 모호성을 줄이기 위해 다양한 신체 부분 표현에 초점을 맞춘 transformer 기반 모델을 훈련시킵니다.

- **Technical Details**: 프로젝트는 RWTH-PHOENIX-Weather 데이터베이스를 기반으로 하여, 훈련 및 평가를 위한 266개의 문장을 사용합니다. 주요 신체 부위(손, 입) 정보를 포함하여 transformer 모델의 입력을 대비하여 실험을 진행하며, 이 과정에서 Connectionist Temporal Classification (CTC) 손실 함수를 활용합니다.

- **Performance Highlights**: 입에 초점을 맞출 경우, 소규모 데이터 세트에서 성능이 향상되며, 손에 초점을 두면 대규모 데이터 세트에서 더 나은 결과를 얻는다는 것이 밝혀졌습니다. 이 연구 결과는 비청각 인성을 위한 디지털 어시스턴트의 접근성을 향상시키는 데 기여합니다.



### Distilling Monolingual and Crosslingual Word-in-Context Representations (https://arxiv.org/abs/2409.08719)
- **What's New**: 본 연구에서는 미리 학습된 masked language model로부터 문맥 속 단어 의미 표현을 증류(disting)하는 방법을 제안합니다. 이는 monolingual과 crosslingual 설정 모두에 적용 가능합니다. 기존의 방법들과 달리, 우리는 인간 주석 데이터(annotated corpora)나 모델 파라미터의 업데이트 없이 작업을 수행하는 점이 특징적입니다.

- **Technical Details**: 제안된 방법은 auto-encoder 기반의 훈련 방식을 사용하며, 이는 자동 생성된 코퍼스(corpus)를 통해 이루어집니다. 특히, 모델은 self-attention 메커니즘을 사용하여 미리 학습된 모델의 다양한 숨겨진 레이어의 출력을 결합하는 능력을 배웁니다. 이는 context-aware lexical semantics와 unsupervised semantic textual similarity (STS) 추정에 있어 단어 의미를 효과적으로 표현하는 데 도움을 줍니다.

- **Performance Highlights**: 모노링구얼(mono-lingual) 과제에서 우리의 표현이 이전 연구에서 제시한 것보다 경쟁력 있는 성능을 보였고, STS 추정을 위한 작업에서 더 우수한 성능을 발휘했습니다. 또한, 크로스링구얼(cross-lingual) 작업에서도 제안된 방법이 다국어 사전 훈련 모델의 단어 표현을 크게 개선하는 결과를 나타냈습니다.



### L3Cube-IndicQuest: A Benchmark Questing Answering Dataset for Evaluating Knowledge of LLMs in Indic Contex (https://arxiv.org/abs/2409.08706)
- **What's New**: 이번 연구에서는 19개 인도 언어와 영어를 포함한 Q&A 데이터세트인 L3Cube-IndicQuest를 개발했습니다. 이 데이터세트는 인도 맥락에 맞춘 지식 표현을 평가하기 위한 기준으로 활용될 수 있습니다.

- **Technical Details**: 데이터세트는 4000개의 질문-답변 쌍으로 구성되어 있으며, 각 언어마다 200개의 질문이 포함되어 있습니다. 다섯 개의 도메인(문학, 역사, 지리, 정치, 경제)으로 나누어져 있고, 인도 북부, 동부, 서부, 남부 지역을 포괄하는 질문이 포함됩니다. 또한, LLM의 성능을 평가하기 위한 참조 기반 평가와 LLM을 평가자로 활용하는 방법론이 제시되었습니다.

- **Performance Highlights**: 사전 평가 결과, 영어를 기준으로 한 LLM의 성능이 마라티어 대비 월등히 높았으며, 저자원 인도 언어에 대한 지식 표현의 격차가 확인되었습니다.



### Large Language Model Can Transcribe Speech in Multi-Talker Scenarios with Versatile Instructions (https://arxiv.org/abs/2409.08596)
- **What's New**: 이 논문에서는 다중 화자(Multi-Talker) 환경에서 음성을 전사하는 데 있어 대형 언어 모델(LLM)의 능력을 조사한 최초의 연구를 제안합니다. 기존 Speech 관련 LLM 연구가 미흡했던 멀티-토커 시나리오에서 LLM이 사용될 가능성을 탐구하고자 합니다.

- **Technical Details**: MT-LLM(Multi-Talker LLM) 모델은 Whisper와 WavLM encoder를 활용하여 음성 정보를 다룰 수 있도록 설계되었습니다. WavLM은 음성 특징을 캡처하고 Whisper는 의미적 맥락을 이해하는 데 도움이 됩니다. 이 두 인코더에서 추출된 정보는 LLM에 입력되어 지시 사항 기반으로 음성을 전사할 수 있게 해줍니다. LoRA(프리미엄 파라미터 효율적 튜닝 기법)를 사용하여 LLM을 조정하고, 텍스트 명령어에 기반하여 음성을 인식하도록 최적화 되어 있습니다.

- **Performance Highlights**: MT-LLM은 ‘칵테일 파티’ 시나리오에서 다중 화자의 음성을 효과적으로 전사할 수 있으며, 사용자 지시에 따라 특정 특성을 가진 화자를 전사하는 뛰어난 성능을 보여줍니다. 실험 결과 MT-LLM은 다양한 요구 사항을 충족할 수 있음을 입증하였습니다.



### Cracking the Code: Multi-domain LLM Evaluation on Real-World Professional Exams in Indonesia (https://arxiv.org/abs/2409.08564)
- **What's New**: 본 논문에서는 다양한 직업 분야에서의 성과를 평가하기 위해 고안된 8,834개의 객관식 질문으로 구성된 IndoCareer 데이터셋을 소개합니다. 이 데이터셋은 인도네시아에 초점을 맞추고 있으며, 의료, 보험 및 금융, 창의 및 디자인, 관광 및 환대, 교육 및 훈련, 법률 등 6개의 주요 분야를 아우릅니다.

- **Technical Details**: IndoCareer 데이터셋은 인도네시아의 다양한 직업 자격시험 및 능력시험에서 수집된 객관식 질문들로 이루어져 있습니다. 이 연구에서는 27개의 대형 언어 모델을 평가했으며, 특히 보험 및 금융 분야에서 강력한 로컬 컨텍스트에 어려움을 겪었습니다. 데이터셋의 전체 사용 시 옵션의 순서를 섞는 것이 모델 간의 평가 결과를 일반적으로 유지하지만, 보험 및 금융 분야에서는 불안정을 초래하는 것으로 나타났습니다.

- **Performance Highlights**: 평가 결과, GPT-4o와 LLaMA3.1(70B)이 전체적으로 가장 높은 성과를 보였으며, 각각 72.3 및 68.3의 평균 정확도를 기록했습니다. 반면, 인도네시아 중심의 모델들은 이 시험에서 성과가 좋지 않았으며, 평균 정확도가 38.0에서 60.0 사이였습니다. 현재 일반적으로 사용되는 LLM들은 인도네시아의 건강 관리 분야에서 신뢰할 수 없음을 나타내며, 평균 성과는 37.2로 매우 낮아, 건강 상담으로 신뢰하지 말아야 합니다.



### Expediting and Elevating Large Language Model Reasoning via Hidden Chain-of-Thought Decoding (https://arxiv.org/abs/2409.08561)
- **What's New**: 이 논문에서는 기존의 Chain-of-Thought (CoT) 방법론의 컴퓨팅 비용 문제를 해결하기 위해 새로운 접근법인 Hidden Chain-of-Thought (HCoT) 모델을 제안합니다. 이 모델은 CoT 과정을 압축하여 효율적인 인퍼런스 (inference)를 가능하게 합니다.

- **Technical Details**: HCoT 모델은 CoT 모델의 파라미터를 고정한 채로 보조 CoT 모델을 훈련하여 압축된 토큰 표현을 생성하고, 이 표현은 HCoT 모델의 입력으로 통합됩니다. 훈련 과정은 두 단계로 진행되며, 첫 단계에서는 CoT 모델이 대비 손실 (contrastive loss)을 사용하여 압축된 토큰 표현을 생성하도록 최적화됩니다. 두 번째 단계에서는 HCoT 모델이 이 압축된 표현을 기반으로 후속 예측을 생성하도록 미세 조정됩니다.

- **Performance Highlights**: 다양한 데이터 세트에서 HCoT 모델의 성능을 평가한 결과, CoT 기준선 모델에 비해 경쟁력 있는 성능을 발휘하며 인코딩 시간에서 최소 1.5배에서 최대 3.8배의 속도 향상을 보였습니다. 이 연구는 다단계 이유 능력의 효율적인 활용 가능성을 제시합니다.



### LLM-Powered Grapheme-to-Phoneme Conversion: Benchmark and Case Study (https://arxiv.org/abs/2409.08554)
Comments:
          5 pages, 5 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용한 Grapheme-to-phoneme (G2P) 변환을 평가하고, 추가적인 훈련이나 레이블이 없는 데이터 없이 LLM의 출력을 향상시키는 프롬프팅(prompting) 및 후처리(post-processing) 방법을 소개합니다. 페르시아어의 문장 수준 음성학적 도전 과제를 평가하기 위한 벤치마크 데이터 세트도 제시합니다.

- **Technical Details**: LLMs는 문맥 의존성과 언어의 다의성을 이해할 수 있는 능력을 갖추고 있으며, 이들은 대표적으로 다의어(polyphone)와 맥락에 따라 달라지는 음소를 갖는 언어에서 G2P 변환에 강점을 보여줍니다. 연구는 LLM의 출력을 최적화하기 위해 다양한 프롬프팅 전략을 개발하였고, 또한 새로운 평가 지표를 통해 Ezafe 예측 가능성을 평가하였습니다.

- **Performance Highlights**: 제안된 방법을 적용함으로써 LLM 기반 G2P가 전통적인 G2P 도구를 초월할 수 있는 잠재력을 보여주며, 개발된 'Kaamel-Dict'는 가장 큰 공개 라이센스 페르시아어 G2P 사전으로, 또한 'Sentence-Bench'는 문장 수준의 음소 도전 과제를 평가하기 위한 최초의 벤치마크 데이터 세트입니다.



### Eir: Thai Medical Large Language Models (https://arxiv.org/abs/2409.08523)
- **What's New**: Eir Thai Medical LLM은 80억 개의 파라미터를 가진 대형 언어 모델로, 태국어 의료 업무의 정확성을 향상시키기 위해 특별히 설계되었습니다.

- **Technical Details**: 이 모델은 의료 전문가와 환자 모두에게 명확하고 이해하기 쉬운 답변을 제공하는 데 중점을 두며, 병원의 내부 네트워크 내에서 데이터 보안을 높이고 처리 속도를 개선하기 위해 배포되었습니다. 내부 API는 암호화와 엄격한 인증을 통해 데이터 유출 및 비인가 접근을 방지합니다.

- **Performance Highlights**: Eir Thai Medical LLM은 8억 개의 파라미터를 가진 여러 오픈소스 대형 언어 모델을 MedQA, MedMCQA, PubMedQA, MMLU의 의료 하위 집합 등 네 가지 의료 기준에서 평가한 결과, 상용 태국어 대형 언어 모델에 비해 10% 이상 성능이 향상되었습니다. 또한, 태국에서 18개의 임상 작업에 맞춘 테스트에서도 GPT-4o의 성능을 11% 이상 초과했습니다.



### A BERT-Based Summarization approach for depression detection (https://arxiv.org/abs/2409.08483)
- **What's New**: 이 논문은 우울증을 효과적으로 감지하기 위한 새로운 접근 방식을 제안합니다. 기존 연구와는 달리, 텍스트 요약 기법을 사용하여 입력 텍스트의 길이와 복잡성을 줄임으로써 모델의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 기술적으로, BERT 기반 모델을 활용하여 텍스트를 수치적 표현으로 변환하고, 이를 통해 우울증 증상을 더욱 정확하게 진단할 수 있습니다. 또한, DAIC-WOZ 데이터셋에 있는 임상적으로 검증된 질문지를 바탕으로 자동화된 인터뷰를 통해 우울증의 지표를 감지합니다.

- **Performance Highlights**: 테스트 세트에서 F1-score 0.67을 기록하며 이전의 모든 기준을 초과하였고, 검증 세트에서는 0.81을 기록하여 DAIC-WOZ 데이터셋에서 대부분의 이전 결과를 초과하였습니다. 또한, 우울증 감지를 위한 요약 질과 관련성을 평가하는 새로운 우울증 용어집을 개발했습니다.



### When Context Leads but Parametric Memory Follows in Large Language Models (https://arxiv.org/abs/2409.08435)
- **What's New**: 이 연구는 9개의 널리 사용되는 대형 언어 모델(LLMs)이 지식 일관성 시나리오에서 질문에 응답할 때 로컬 컨텍스트와 글로벌 파라미터 간의 지식을 어떻게 할당하는지를 분석합니다. 새롭게 소개된 데이터셋인 WikiAtomic을 통해 LLMs가 제공된 정보를 어떻게 우선시하고 활용하는지를 체계적으로 조사하였습니다.

- **Technical Details**: WikiAtomic 데이터셋은 200개의 위키피디아 기사를 기반으로 한 원자 문장들로 구성돼 있으며, 모델이 다양한 크기의 컨텍스트를 받았을 때 응답하는 경향을 분석합니다. LLMs는 응답에서 최대 30%의 파라메트릭 (parametric) 지식을 통합하고, 작은 컨텍스트에선 모든 부분의 정보를 기억하나 큰 컨텍스트에선 주로 첫 번째 절반에 집중합니다.

- **Performance Highlights**: 모델들은 약 70%의 컨텍스트 (contextual) 지식과 30%의 파라메트릭 지식을 일관되게 의존하며, 컨텍스트가 늘어남에 따라 환각(hallucination) 발생률은 감소하는 경향이 있습니다. 이 연구 결과는 효과적인 컨텍스트 조직과 예측 가능한 입력 사용 모델 개발의 중요성을 강조합니다.



### Knowledge Tagging with Large Language Model based Multi-Agent System (https://arxiv.org/abs/2409.08406)
Comments:
          8 pages, 3 figures

- **What's New**: 이 연구는 자동화된 지식 태깅(knowledge tagging) 프로세스를 위한 다중 에이전트 시스템(multi-agent system)을 도입합니다. 이 시스템은 이전 알고리즘의 한계를 극복하는 데 중점을 두며, 복잡한 사례 처리에 있어 LLMs(large language models)를 활용합니다.

- **Technical Details**: 제안된 LLM 기반 다중 에이전트 시스템은 네 가지 유형의 LLM 에이전트(task planner, question solver, semantic judger, numerical judger)로 구성되어 있습니다. 각 에이전트는 독립적인 하위 문제를 처리하며, 계획 에이전트가 задан된 지식 정의에 맞춰 협력 계획을 제안합니다. 최종적으로, 중간 결과를 AND 연산자로 연결하여 최종 판단을 출력합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 공개된 수학 문제 지식 태깅 데이터세트 MathKnowCT에서 이전 단일 LLM 기반 방법보다 일관된 성능 개선을 보였습니다. 이는 교육 환경에서 LLM 기반 알고리즘의 배치에 대한 유망한 결과를 강조합니다.



### Real or Robotic? Assessing Whether LLMs Accurately Simulate Qualities of Human Responses in Dialogu (https://arxiv.org/abs/2409.08330)
- **What's New**: 이 연구에서는 LLM(대형 언어 모델)을 사용하여 100,000개의 인간-LLM 및 LLM-LLM 대화를 생성하고, 이러한 LLM 시뮬레이션이 실제 인간 대화와 어느 정도 일치하는지를 평가합니다.

- **Technical Details**: 연구에서는 WildChat 데이터셋의 100만 대화에서 2천 개의 인간-LLM 대화를 대상으로 하여 LLM의 시뮬레이션 능력을 분석합니다. 다양한 LLM 모델과 프롬프트를 사용하여 시뮬레이션의 질을 평가하고, 서로 다른 언어에서의 상관관계를 확인합니다.

- **Performance Highlights**: 결과적으로 LLM 시뮬레이션과 인간 간의 대화 사이의 일치는 상대적으로 낮으며, 영어, 중국어, 러시아어에 대한 성능 분석에서도 유사한 경향을 보입니다. 인간이 LLM의 스타일에 맞게 대화를 시작할 때 더 나은 시뮬레이션이 가능함을 발견했습니다.



### Agents in Software Engineering: Survey, Landscape, and Vision (https://arxiv.org/abs/2409.09030)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문은 소프트웨어 엔지니어링(SE) 분야에서 LLM(대형 언어 모델) 기반 에이전트의 연구를 통합한 첫 번째 설문조사를 수행하였습니다. LLM 기반 에이전트를 구성하는 세 가지 핵심 모듈인 인지(Perception), 기억(Memory), 행동(Action)에 대한 개념적 프레임워크를 제공합니다.

- **Technical Details**: 이 연구는 LLM 기반 에이전트 기술을 소프트웨어 엔지니어링(SE) 분야에 적용한 115개의 논문을 수집하고, 이를 필터링하여 품질 평가를 통해 분석하였습니다. 에이전트의 인지 모듈은 다양한 양식의 입력(예: 텍스트, 비주얼, 청각)을 처리하며, 기억 모듈은 의미 기억(Semantic Memory), 회상 기억(Episodic Memory), 절차 기억(Procedural Memory)을 포함해 에이전트의 추론 결정을 돕습니다. 행동 모듈은 내부 행동(추론, 검색, 학습)과 외부 행동(환경과의 상호작용)을 포함합니다.

- **Performance Highlights**: LLM 기반 에이전트의 성능을 향상시키기 위해 기존 작업의 여러 도전 과제를 정리하고, 새로운 연구 기회를 제시합니다. 주요 제안으로는 다양한 양식의 입력을 탐색하는 것, 복잡한 SE 작업을 수행할 수 있는 다양한 능력을 갖춘 에이전트를 개발하는 것, 외부 검색 기반으로 사용될 수 있는 코드 관련 지식 기반이 필요하다는 점을 강조합니다.



### AI-LieDar: Examine the Trade-off Between Utility and Truthfulness in LLM Agents (https://arxiv.org/abs/2409.09013)
- **What's New**: 새로운 연구 AI-LieDar 프레임워크는 LLM 기반 에이전트들이 유용성과 진실성 간의 충돌 상황을 어떻게 탐색하는지를 다룬다. 이 연구는 진실성과 유용성의 상충을 탐구하며, 더욱 상호작용적인 멀티 턴 설정에서 LLM의 진실성을 평가한다.

- **Technical Details**: 연구는 60개의 다양한 시나리오를 통해 LLM의 행동을 분석하며, 진실성 감지기를 개발하여 모델의 응답을 평가했다. 시뮬레이션 파라미터는 Sotopia 프레임워크를 이용했으며, 진실성을 측정하기 위해 세분화된 평가 도구를 도입하였다. 2160회의 시뮬레이션 결과, 모든 모델은 50% 미만의 진실성을 보여주었다.

- **Performance Highlights**: 연구 결과, 모델들은 강한 유용성 목표가 있는 경우 속이기 행동을 보일 수 있으며, 특정 시나리오에서 모델들이 진실성을 따를 때 목표 달성률이 15% 감소하였다. 더욱이, 강력한 모델들은 속이거나 진실하게 대답하도록 유도될 때 40%의 증가율을 보였다.



### Safeguarding Decentralized Social Media: LLM Agents for Automating Community Rule Complianc (https://arxiv.org/abs/2409.08963)
- **What's New**: 본 논문에서는 사용자 생성 콘텐츠의 경우 전통적인 인간 기반의 준수 확인 방법이 한계에 부딪히고 있음을 강조하며, 자동화된 콘텐츠 준수 검증을 위한 새로운 가능성을 제시합니다. 특히, Large Language Models을 활용하여 Decentralized Social Networks에서 규칙 준수 확인을 수행하는 AI-agent를 평가합니다.

- **Technical Details**: 본 연구에서는 Open-LLMs에 기반한 6개의 AI-agent를 사용하여 커뮤니티의 다양한 규칙과 범위에 대한 준수 여부를 분석합니다. Mastodon 서버에서 수집한 50,000개 이상의 게시물을 분석하여, AI-agent가 비준수 콘텐츠를 효과적으로 탐지하고 언어적 뉘앙스를 파악하며 다양한 커뮤니티 환경에 적응하는 능력을 입증하였습니다.

- **Performance Highlights**: 대부분의 AI-agent는 높은 inter-rater reliability와 일관성을 보여주었으며, 준수에 대한 점수 정당화 및 제안에서도 신뢰성을 발휘했습니다. 전문가와의 인간 기반 평가를 통해 이들 에이전트의 신뢰성과 유용성이 확인되어, 반자동화 또는 인간과 협력하는 콘텐츠 조정 시스템에서 유망한 도구로 자리잡을 가능성이 있습니다.



### SynSUM -- Synthetic Benchmark with Structured and Unstructured Medical Records (https://arxiv.org/abs/2409.08936)
- **What's New**: SynSUM 벤치마크는 비구조적 임상 노트를 구조적 배경 변수와 연결하는 합성 데이터셋을 소개합니다. 이 데이터셋은 호흡기 질환 영역의 虚構적인 환자 접촉을 설명하는 노트가 포함된 10,000개의 인공 환자 기록으로 구성됩니다.

- **Technical Details**: 이 데이터셋의 테이블 부분은 전문가 기반의 Bayesian network를 통해 생성되며, 질병 이력 및 현재 방문에 관한 부분적으로 인코딩된 정보를 포함합니다. 대형 언어 모델인 GPT-4o를 사용하여 환자 증상 및 추가 정보를 설명하는 임상 노트를 생성합니다.

- **Performance Highlights**: 이 데이터셋은 임상 정보 추출(CIE)과 데이터 자동화, 인과 효과 추정 및 다중 모달 합성 데이터 생성 연구에 유용한 기초를 제공합니다. SynSUM은 구조적 및 비구조적 데이터의 혼합을 통해 기존 시스템의 한계를 극복하려는 목표를 가지고 있습니다.



### Affective Computing Has Changed: The Foundation Model Disruption (https://arxiv.org/abs/2409.08907)
- **What's New**: 본 연구는 Foundation Models(FMs)의 발전이 Affective Computing(정서 컴퓨팅) 분야에 미치는 영향을 분석하고, 멀티모달 감정 데이터를 생성하고 분석하는 데 중점을 두고 있습니다. 특히, 본 논문은 FMs가 감정 인식, 감정 콘텐츠 생성 및 감정에 대한 반응을 포함한 다양한 연구 문제에 어떻게 기여할 수 있는지를 조명합니다.

- **Technical Details**: 연구는 FMs의 학습 특성에 주목하며, 이 모델들이 다양한 데이터에 대해 트레이닝되어 폭넓은 문제를 해결할 수 있다는 점을 강조합니다. FMs는 대량의 학습 파라미터를 이용해, 특정 훈련을 받지 않은 작업에서도 경쟁력 있는 성능을 발휘할 수 있습니다. 또한 새로운 아키텍처인 Diffuser와 Transformer, 자가 지도 학습 전략, 그리고 상호 모달 정렬 기법의 발전이 FMs의 성능 향상을 견인하고 있습니다.

- **Performance Highlights**: FMs는 정서적 콘텐츠와 관련된 작업에서 놀라운 성능을 보여주며, 현실적인 데이터 샘플을 생성하거나 제로샷 분류를 수행할 수 있는 능력을 갖추고 있습니다. 이와 같은 발전은 FMs의 정서적 능력의 범위와 잠재력을 여전히 불확실하게 하면서도 Affective Computing 커뮤니티에 상당한 영향을 미치고 있습니다.



### Visual Language Tracking with Multi-modal Interaction: A Robust Benchmark (https://arxiv.org/abs/2409.08887)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 Visual Language Tracking (VLT) 작업에 최초로 다중 라운드 상호작용을 도입한 새로운 벤치마크, VLT-MI(Visual Language Tracking with Multi-modal Interaction)를 제안합니다.

- **Technical Details**: VLT-MI는 기존의 VLT 벤치마크를 기반으로 다양한 다중 품질 텍스트를 생성하여 다중 라운드 상호작용을 가능하게 합니다. DTLLM-VLT 모델을 활용하여 상호작용하는 중에 텍스트를 업데이트하고 객체를 회복합니다. 또한, 수정된 바운딩 박스(bbox)와 텍스트를 통해 더 나은 추적 성능을 목표로 합니다.

- **Performance Highlights**: 기존의 VLT 벤치마크와 VLT-MI 간의 비교 실험을 통해 정확성과 견고성을 평가했습니다. 결과적으로, 추적 정확도는 LaSOT에서 기대치를 충족하였으나 다른 벤치마크에서 성능이 저하되었습니다. 상호작용 수는 작업의 난이도가 증가함에 따라 증가하는 경향을 보였으며, 글로벌 인스턴스 추적에서는 최대 229회의 상호작용이 발생했습니다.



### FP-VEC: Fingerprinting Large Language Models via Efficient Vector Addition (https://arxiv.org/abs/2409.08846)
- **What's New**: 최근 연구에 따르면 대형 언어 모델(LLMs)의 소유권 인증을 위해 이제 FP-VEC라는 새로운 지문 피팅 방법이 도입되었습니다. 이 방법은 모델 내에 비밀 서명을 효율적으로 내장하는 fingerprint vector를 생성하여, 여러 LLM에 무제한으로 적용할 수 있는 가능성을 가지고 있습니다.

- **Technical Details**: FP-VEC 방법은 지문을 내장하기 위해 고유한 벡터 중첩 방식을 활용합니다. 이 방법은 기본 모델로부터 파생된 다운스트림 모델에 fingerprint vector를 추가하여 지문 정보를 효과적으로 통합합니다. 모델 응답을 사전 훈련된 모델의 가중치에서 세밀하게 조정하여 생성합니다.

- **Performance Highlights**: FP-VEC는 CPU 전용 장치에서 빠른 실행 시간을 보이며, 이전 방법들보다도 우수한 성능을 유지하면서 모델의 원래 성능에 최소한의 영향을 미칩니다. 실험 결과, 이 방법은 매우 효율적이며 무한히 많은 모델에 적용할 수 있는 확장성을 지닌 것으로 나타났습니다.



### Journalists, Emotions, and the Introduction of Generative AI Chatbots: A Large-Scale Analysis of Tweets Before and After the Launch of ChatGP (https://arxiv.org/abs/2409.08761)
- **What's New**: 이번 연구는 Generative AI의 영향에 대한 더 넓은 조사 일환으로, ChatGPT 출시 당시 언론인들의 감정적 반응을 분석하였다. 주요 미국 뉴스 매체의 언론인들로부터 수집된 거의 100만 개의 트윗을 분석한 결과, ChatGPT 도입 전후의 감정 톤 변화와 감정을 추적하였다.

- **Technical Details**: 연구는 Computational 및 Natural Language Processing 기술을 사용하여 ChatGPT 출시로 인한 감정의 변화를 측정하였다. 분석 결과, 긍정적인 감정의 증가와 함께 출시 이후 더 우호적인 톤이 관찰되었으며, 이는 AI의 잠재력에 대한 초기 낙관적 태도를 나타낸다.

- **Performance Highlights**: 이 연구는 언론인이 기술 혁신과 혼란의 해석자로서 중요한 역할을 수행한다는 점을 강조하며, 그들의 감정적 반응이 새로운 기술에 대한 대중의 내러티브를 형성하는 데 어떻게 기여할 수 있는지를 보여준다. 연구는 저널리즘, 감정, AI의 교차점 이해에 기여하며, Generative AI 도구의 사회적 영향에 대한 통찰력을 제공한다.



### Layerwise Change of Knowledge in Neural Networks (https://arxiv.org/abs/2409.08712)
- **What's New**: 이 논문은 딥 뉴럴 네트워크(DNN)가 새로운 지식을 점진적으로 추출하고 잡음을 포함한 특징을 잊어버리는 과정을 층별로 설명합니다. 특히, 중간 층에서 암호화된 상호작용(interactions)을 추출하고 이를 정량화하는 새로운 접근방법을 제시합니다.

- **Technical Details**: 본 연구에서는 입력과 중간 층 특징 간의 상호 정보(mutual information)를 이용하여 DNN 각 층에서 암호화된 지식을 측정하며, 이는 학습 과정에서 새롭게 등장하고 잊혀진 상호작용을 추적하고 정량화하는 과정으로 이어집니다. 또한 인접한 층들은 유사한 상호작용을 암호화하는 경향이 있습니다.

- **Performance Highlights**: 우리는 새로운 상호작용 정의가 DNN의 일반화 능력(generalization capacity)과 어떻게 연결되는지를 밝혀내며, 저차원 상호작용이 고차원 상호작용보다 더 높은 안정성과 일반화 능력을 가짐을 발견했습니다.



### B4: Towards Optimal Assessment of Plausible Code Solutions with Plausible Tests (https://arxiv.org/abs/2409.08692)
Comments:
          accepted by ASE' 24 (full paper)

- **What's New**: 이 논문은 코드 생성 과정에서 여러 대안 중 최적의 코드 솔루션을 선택하는 새로운 방법론을 제시합니다. 특히, 근거가 불명확한 테스트 케이스를 사용할 때의 한계를 극복하고, Bayesian (베이지안) 프레임워크를 기반으로 최적 선택 전략을 정의합니다.

- **Technical Details**: 제안된 방법론은 코드 솔루션과 테스트 케이스 간의 관측된 상태의 사후 확률(posterior probability)을 기반으로 최적 선택 전략을 설정하고, 이를 정수 계획 문제(Integer Programming Problem)로 정형화합니다. 그 후, 불확실한 초기 분포(prior distributions)를 고려하여 유효한 사후 확률의 근사(computable form)를 계산할 수 있는 방법을 도출했습니다. 이 근사 방법은 최대 다항식 복잡도(polynomial complexity)로 최적화할 수 있으며, 초기 지식의 정확성에 의해 오류가 한정됩니다.

- **Performance Highlights**: 제안된 B4 (𝓑^4) 전략은 LLM으로 생성된 코드 솔루션과 LLM으로 생성된 테스트 케이스를 사용한 선택 과정에서 기존의 휴리스틱(heuristics)보다 최대 50% 향상된 성능을 보여주었습니다. 특히, 가장 어려운 상황에서는 기존의 가장 강력한 휴리스틱에 비해 246%의 성능 개선이 있었습니다.



### NEST-RQ: Next Token Prediction for Speech Self-Supervised Pre-Training (https://arxiv.org/abs/2409.08680)
Comments:
          5 pages, 2 figures, Work in progress

- **What's New**: 본 논문에서는 다음 토큰 예측(next token prediction)을 기반으로 한 음성 자가 감독(pre-training) 방법인 NEST-RQ를 제안합니다. 기존의 음성 SSL 방법들이 비인과적(encoders) 인코더를 활용하는 반면, NEST-RQ는 인과적(causal) 인코더를 사용하여 스트리밍 모델에 적합하도록 설계되었습니다.

- **Technical Details**: NEST-RQ는 랜덤 프로젝션 양자화기(random-projection quantizer)를 활용하여 연속 음성 특징을 분리된 토큰(sequence of discrete tokens)으로 변환합니다. 이 과정에서 인과적 인코더를 이용하여 오직 왼쪽 문맥(left context)만을 고려하여 다음 토큰을 예측하는 방식을 취하고 있습니다. 이를 통해, 음성 SSL의 가능한 개선점을 제시합니다.

- **Performance Highlights**: NEST-RQ는 30만 시간의 라벨이 없는 음성 데이터와 3만 시간의 라벨이 있는 데이터셋을 활용하여 실험되었으며, 비스트리밍 ASR에서는 기존 BEST-RQ와 유사한 성능을 보이고, 스트리밍 ASR에서는 더 나은 성능을 달성했습니다. 이는 NEST-RQ의 간단한 설계가 BEST-RQ의 강점을 물려받으면서 성능을 유지한다는 것을 보여줍니다.



### Investigating Disentanglement in a Phoneme-level Speech Codec for Prosody Modeling (https://arxiv.org/abs/2409.08664)
- **What's New**: 본 논문에서는 Residual Vector Quantization (RVQ) 기반의 음성 신경 코덱 모델을 제안하며, 이를 통해 음소 수준의 프로소디(prosody) 정보를 효율적으로 모델링할 수 있음을 보여줍니다. 기존의 연속 잠재 공간 대신 이산 잠재 공간을 활용하여 언어적 정보와 화자 정보로부터 독립적인 프로소디 표현을 학습하게 됩니다.

- **Technical Details**: 제안된 모델은 음소 인코더(phoneme encoder)와 디코더(decoder)를 각각 언어적 표현과 화자 임베딩에 조건화하여, 진정한 프로소디 특성을 캐치할 수 있도록 설계되었습니다. 모델은 Conformer 아키텍처를 기반으로 하여 구성되며, 4개의 Conformer 레이어와 256 차원의 벡터를 사용합니다. RVQ 모듈은 2단계의 양자화를 사용하여 각 레이어마다 256개의 코드를 생성합니다.

- **Performance Highlights**: 이 연구를 통해 제안된 모델은 높은 품질의 생성 음성을 유지하면서 차원과 복잡성을 상당히 줄이는 성과를 거두었습니다. 프로소디 정보는 독립적으로 잘 분리되었으며, 실험 결과는 이러한 독립성이 높다는 것을 보여줍니다. 또한, 해석 가능성과 제어 가능성이 향상되어, 향후 연구를 통해 더 효율적인 사전 모델링(prior modeling)이 이루어질 가능성도 제기됩니다.



### LA-RAG:Enhancing LLM-based ASR Accuracy with Retrieval-Augmented Generation (https://arxiv.org/abs/2409.08597)
Comments:
          submitted to ICASSP 2025

- **What's New**: 최근 대규모 언어 모델(LLM)과 음성 정보 통합 향상에 대한 연구가 활발히 진행되고 있으며, 이로 인해 자동 음성 인식(ASR) 정확도가 크게 향상되었습니다. 본 연구에서는 LA-RAG라는 새로운 Retrieval-Augmented Generation (RAG) 패러다임을 제안합니다.

- **Technical Details**: LA-RAG는 정밀한 토큰 수준의 음성 데이터 저장소와 음성-음성 검색 메커니즘을 활용하여 LLM의 인-context learning (ICL) 능력을 통해 ASR 정확도를 향상시킵니다. 실험은 표준 Mandarin 및 다양한 중국 방언 데이터셋에서 수행되었으며, 기존 방법들에 비해 ASR 정확도가 상당히 향상된 것을 보여줍니다.

- **Performance Highlights**: 이 연구의 실험 결과는 LA-RAG가 발음 변별성 문제를 효과적으로 처리할 수 있음을 나타내며, 기존 ASR 시스템과 비교하여 성능의 유의미한 개선을 보여줍니다.



### MAPX: An explainable model-agnostic framework for the detection of false information on social media networks (https://arxiv.org/abs/2409.08522)
Comments:
          16 pages, 5 figures

- **What's New**: 이 연구는 MAPX라는 새로운 모델 비특정 프레임워크를 소개하여 기존 모델의 예측 결과를 증거 기반으로 집계하는 방법을 제공한다.

- **Technical Details**: MAPX 프레임워크는 동적이고 적응적인 예측 집계 방법(Dynamic Adaptive Prediction Aggregation, DAPA)을 통해 여러 정보 소스의 신뢰성을 고려하여 예측 결과를 향상시킨다. 또한, 예측 결과의 신뢰성 및 설명 가능성을 중요시하는 계층형 설명 방법(Hierarchical Tiered eXplanation, HTX)을 개발한다.

- **Performance Highlights**: MAPX는 3개의 실제 데이터셋과 7개의 차세대 모델을 비교한 실험에서 모든 최신 성능 모델보다 일관되게 더 나은 성능을 보여줬으며, 예측 특성의 질이 저하되었을 때도 높은 성능을 유지하였다.



### Explaining Datasets in Words: Statistical Models with Natural Language Parameters (https://arxiv.org/abs/2409.08466)
- **What's New**: 이 논문은 대량의 데이터를 해석하기 위해 자연어(natural language) 술어(predicates)에 의해 매개변수화된 통계 모델의 패밀리를 도입합니다. 이 모델은 클러스터링(clustering), 시계열(time series), 분류(classification) 작업을 포함하며, 예를 들어 COVID에 관한 텍스트 클러스터는 "COVID에 대해 논의함"이라는 술어로 매개변수화될 수 있습니다.

- **Technical Details**: 슬림화된 통계 모델의 매개변수를 통해 해석 가능성을 높이기 위해, 저자들은 Grazing Descent와 같은 방법을 통해 연속 완화(continuous relaxations)를 최적화하는 모델 독립적(model-agnostic) 알고리즘을 개발했습니다. 이를 통해 데이터에서의 술어 매개변수의 최적화를 시도하고, 언어 모델(large language models)을 통해 이들을 이산형(discrete)으로 변환합니다.

- **Performance Highlights**: 프레임워크는 사용자 대화 대화를 세분화하고, 시간이 지남에 따라 변화하는 방법을 설명하며, 다양한 언어 모델을 비교하고, 수학 문제를 클러스터링하는 데 적용되었습니다. 특히, 이 방법은 텍스트 및 비주얼 도메인에서 활용 가능하고, 다양한 부작용을 조절할 수 있어 기존의 n-그램 분석 같은 방법이 어려운 복잡한 개념을 설명할 수 있습니다.



### Self-Supervised Inference of Agents in Trustless Environments (https://arxiv.org/abs/2409.08386)
- **What's New**: 본 논문에서는 에이전트들이 스워밍(swarming)을 형성하여 고품질 응답을 효과적으로 생성하는 새로운 접근 방식을 제안합니다. LLMs(Large Language Models)를 활용하여 신뢰할 수 없는(agent inference) 에이전트 추론을 평가하고, 다양한 유형의 악의적(agent) 공격을 모델링합니다.

- **Technical Details**: 제안된 방법은 스와의 집합 지능을 활용하여 강력하고 효율적인 분산 AI 추론을 보장합니다. 본 논문은 멀티 에이전트 아키텍처를 통해 각 에이전트가 데이터 추론 및 품질 평가를 수행하며, 응답 생성, 선택적 순위 매기기 및 최종 선택이라는 세 가지 주요 단계로 구성된 스와 공세 메커니즘을 설명합니다.

- **Performance Highlights**: 제안된 접근 방식은 다른 신뢰할 수 없는 추론 전략에 비해 125 ms 이하의 검증 지연(latency)으로 한 차원 빠른 성능을 보입니다.



### Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations (https://arxiv.org/abs/2409.08381)
- **What's New**: 이 논문에서는 Vision-Language 모델(VLM)인 CLIP을 Multi-Label Recognition (MLR)에 대한 새로운 접근법으로 활용합니다. 특히, PositiveCoOp와 NegativeCoOp라는 두 가지 방법을 소개하여 양수 및 음수 프롬프트 학습의 효과를 분석합니다.

- **Technical Details**: PositiveCoOp에서는 클래스를 나타내는 양수 프롬프트 하나를 학습하고, 음수 프롬프트 대신 VLM의 텍스트 인코더에 의존하지 않고 이미지 특징과 연관된 임베딩 벡터를 직접 학습합니다. NegativeCoOp에서는 반대로 진행합니다. 이러한 접근 방식은 dual prompt 학습보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: PositiveCoOp 방법이 DualCoOp보다 뛰어난 성능을 보였으며, 전체 레이블 비율이 높을 때 baseline의 성능은 DualCoOp 및 PositiveCoOp에 필적할 정도로 강력했습니다. 이 baseline은 DualCoOp보다 약 16배 적은 훈련 매개변수와 절반의 훈련 시간만 필요로 합니다.



### Towards Quantifying and Reducing Language Mismatch Effects in Cross-Lingual Speech Anti-Spoofing (https://arxiv.org/abs/2409.08346)
Comments:
          Accepted to the IEEE Spoken Language Technology Workshop (SLT) 2024. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구는 영어 데이터로 훈련된 스피치 안티스푸핑(anti-spoofing) 시스템이 다른 언어로 테스트할 때 성능이 저하된다는 점을 발견했습니다. 이를 개선하기 위해 TTS(Text-to-Speech)를 활용하여 여러 언어의 억양(accent)을 반영한 데이터를 생성하는 새로운 방법, ACCENT를 제안했습니다.

- **Technical Details**: ACCENT 방법은 기계 학습을 위해 독일어, 일본어 등 12개 언어로 구성된 300만 건 이상의 샘플을 포함한 대규모 데이터 세트를 사용하여 테스트되었습니다. 기존 리소스가 적은 언어에 대한 데이터 수집의 어려움을 극복하고 다국어 간의 성능 차이를 줄였습니다. 또한, 주요 TTS 모델을 통해 다양한 억양을 통합하여 영문 데이터의 풍부함을 증가시킵니다.

- **Performance Highlights**: ACCENT 방법을 적용하면서 언어 격차로 인한 성능 저하가 15% 이상 감소했습니다. 기존의 모델이 부족한 다국어 대응 능력을 개선하여 다양한 영어 억양에 대한 강건성을 보여주었습니다. 이 연구는 언어 독립적인 안티스푸핑 모델 개발에 대한 새로운 방향을 제시합니다.



### Large Language Models are Pattern Matchers: Editing Semi-Structured and Structured Documents with ChatGP (https://arxiv.org/abs/2409.07732)
- **What's New**: 이 논문은 Large Language Models(LLMs)가 구조화된(document) 및 반구조화된(semi-structured) 문서를 최소한의 노력으로 편집할 수 있는지를 조사합니다. 연구 결과 ChatGPT는 기본적인 프롬프트를 통해 이러한 문서들을 효과적으로 수정할 수 있는 능력을 보여 주며, 이에 대한 패턴 매칭 기법도 눈여겨볼 만하다고 보고합니다.

- **Technical Details**: 논문에서는 LLM이 이미 구조화된 텍스트를 얼마나 잘 처리하거나 재구성할 수 있는지에 대한 연구가 이루어졌습니다. 이 연구는 질적 연구 접근 방식을 채택하여 LaTeX로 포맷된 문서와 같은 다양한 문서 형식에서 두 가지 케이스 스터디를 수행했습니다. LLM은 입력 내용을 처리하며 주어진 프롬프트의 구조를 인식하고 활용하는 능력을 보여 주었습니다.

- **Performance Highlights**: 실험 결과, LLM은 구조화된 문서 편집 작업에서 큰 품질을 제공하며, 사용자는 최소한의 수작업 후처리만으로 우수한 결과를 얻을 수 있습니다. 연구자는 LLM이 문서 형식 간 변환을 지원하고, 기존 프로그래밍 방식보다 더 유연한 솔루션을 제공할 수 있음을 나타냈습니다.



New uploads on arXiv(cs.IR)

### Contri(e)ve: Context + Retrieve for Scholarly Question Answering (https://arxiv.org/abs/2409.09010)
- **What's New**: 이 논문은 Scholar-QALD 데이터셋을 사용해 학술적 지식 그래프 기반의 질문 응답 시스템을 구축하는 두 단계 접근 방식을 제안합니다. 첫 번째 단계에서는 구조화된 및 비구조화된 데이터 출처로부터 질문과 관련된 문맥을 추출하고, 두 번째 단계에서는 LLM(Large Language Model)의 정보 검색 성능을 향상시키기 위한 프롬프트 엔지니어링을 수행합니다.

- **Technical Details**: 제안하는 시스템은 DBLP와 SemOpenAlex의 두 가지 학술 지식 그래프와 위키백과(English Wikipedia) 텍스트의 세 가지 데이터 소스에서 질문에 대한 답변을 찾기 위해 혼합(hybrid) 솔루션을 채택합니다. Llama3.1 LLM을 활용하여 프롬프트 엔지니어링을 적용하고, F1 점수 40%를 달성하였습니다.

- **Performance Highlights**: 다양한 출처에서 정보를 추출한 후, 모델의 응답 품질이 향상되었으며, 비정상적인 응답(anomalous responses) 사례도 관찰되었습니다. 이 결과는 또한 논문의 마지막 부분에서 논의됩니다.



### Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems (https://arxiv.org/abs/2409.08987)
- **What's New**: 최근 연구에서는 6개의 사전 훈련된 백엔드 모델(MusicFM, Music2Vec, MERT, EncodecMAE, Jukebox, MusiCNN)을 음악 추천 시스템(MRS) 맥락에서 평가하고 있습니다. 사전 훈련된 오디오 표현이 MRS의 추천 모델 성능에 미치는 영향을 분석함으로써, MIR 및 MRS 간의 성능 차이를 규명하는 중요한 연구를 진행하고 있습니다.

- **Technical Details**: 연구에서는 3가지 추천 모델(KNN, Shallow Neural Network, BERT4Rec)을 사용하여 사전 훈련된 오디오 표현을 평가합니다. 모델들은 Mel-Spectrogram을 기반으로 하며, MFCC와 같은 저수준 음향 특성과 각 모델의 고유한 아키텍처를 사용하여 오디오 데이터를 처리합니다. MERT와 MusicFM을 포함한 다양한 최신 모델이 사용되었습니다.

- **Performance Highlights**: 연구 결과, 사전 훈련된 오디오 표현은 전통적인 MIR 작업과 MRS 간의 성능 변동성이 크다는 것을 보여주었으며, 추천 시스템에서도 유용하게 활용될 수 있는 가능성을 열어주었습니다. 특히, 각 모델이 특정 추천 작업에서 어떻게 성능을 발휘하는지에 대한 데이터가 제공되었습니다.



### Proactive Recommendation in Social Networks: Steering User Interest via Neighbor Influenc (https://arxiv.org/abs/2409.08934)
- **What's New**: 이번 논문에서는 사용자들의 역사적 관심사를 초월하는 새로운 추천 시스템, 즉 사회적 네트워크에서 사용자의 관심을 간접적으로 조정하는 'Proactive Recommendation in Social Networks (PRSN)' 작업을 제안합니다. PRSN은 사용자의 이웃의 영향을 활용하여 목표 사용자의 관심을 유도하는 새로운 방법론을 제시합니다.

- **Technical Details**: PRSN은 사용자와 항목 간의 연결성을 고려하여 사용자 피드백을 예측하는 데 causal inference를 활용합니다. 이를 위해 Neighbor Interference Recommendation (NIRec) 프레임워크를 개발하였으며, 두 가지 주요 모듈로 구성됩니다: (1) 관심사 추정 모듈(Interference representation-based estimation module)과 (2) 최적화 모듈(Post-learning-based optimization module). 이 모듈은 사용자의 개인적 관심과 이웃의 간섭 효과를 통합하여 잠재적 피드백을 추정합니다.

- **Performance Highlights**: 다양한 실제 데이터 세트를 기반으로 한 광범위한 반시뮬레이션 실험을 통해 NIRec의 효과적인 유도 성능이 확인되었습니다. 목표 사용자의 흥미를 유도하면서도 이웃의 경험에 대한 피해를 최소화할 수 있는 방법을 제시하며, 전통적인 추천 시스템의 한계를 극복하는 데 기여할 것으로 기대됩니다.



### LLM-based Weak Supervision Framework for Query Intent Classification in Video Search (https://arxiv.org/abs/2409.08931)
Comments:
          6 pages, 5 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)을 활용하여 사용자의 검색 쿼리를 자동으로 주석 처리하는 새로운 접근 방식을 소개합니다. 이를 통해 자연어 이해(NLU) 모델의 학습 데이터를 효율적으로 생성하고, 다양한 사용자 의도를 효과적으로 분류할 수 있습니다.

- **Technical Details**: 우리는 약한 감독(weak supervision) 기법을 적용하여 LLM을 통해 방대한 검색 쿼리 데이터에 대한 주석을 생성합니다. 이 과정에서 프롬프트 엔지니어링(prompt engineering)과 다양한 LLM 페르소나(persona)를 활용하여 인간 주석자의 기대에 부합하는 훈련 데이터를 생성합니다. 또한, Chain of Thought(COT) 및 In-Context Learning(ICL)을 통해 도메인 지식을 통합하여 신속한 추론을 위한 저지연(low-latency) 모델로 학습합니다.

- **Performance Highlights**: 우리의 접근 방식은 Recall에서 평균적으로 113%의 향상을 보였으며, F1 스코어 대비 LLM 예측과 인간 주석 간의 합의율이 47.60% 개선되었습니다. 또한 페르소나 선택 라우팅 메커니즘을 통해 우리는 F1 스코어를 3.67% 추가로 향상시킬 수 있었습니다.



### NeSHFS: Neighborhood Search with Heuristic-based Feature Selection for Click-Through Rate Prediction (https://arxiv.org/abs/2409.08703)
- **What's New**: 이번 논문에서는 Click-through-rate (CTR) 예측에서의 성능 향상을 위한 NeSHFS(Neighborhood Search with Heuristic-based Feature Selection)라는 새로운 휴리스틱 알고리즘을 제안합니다. 이 알고리즘은 차원 축소와 훈련 시간 비용을 줄이면서도 효과적인 피처 선택을 제공하기 위해 설계되었습니다.

- **Technical Details**: NeSHFS는 피처 선택을 위한 방식으로, 피처를 순차적으로 제거하고 이웃 검색을 통해 최고의 솔루션을 찾는 접근 방식을 사용합니다. 이 방법은 기존의 그리드 검색 hyperparameter 최적화 기술에서 영감을 받았으며, 특화된 휴리스틱을 적용하여 탐색 공간을 고르게 탐색하고 유망한 영역에서 성능을 극대화합니다. 실험은 DeepCTR 프레임워크를 이용해 진행하였으며, 80/10/10 비율로 데이터셋을 훈련, 검증, 테스트에 나누었습니다.

- **Performance Highlights**: 제안된 NeSHFS는 세 가지 공공 데이터셋(화웨이 Digix 2022, Criteo, Avazu)에서 실험을 통해 효율성과 효과성을 검증하였으며, 피처 집합을 축소함으로써 Deep CTR 모델의 훈련 시간을 비약적으로 줄이며, 더 많은 트래픽에 대한 추론 시간을 단축했습니다. 또한, 이 알고리즘은 다양한 Deep CTR 모델에 적용 가능하며, 최근 훈련된 모델의 적응력 향상에 기여할 수 있습니다.



### ATFLRec: A Multimodal Recommender System with Audio-Text Fusion and Low-Rank Adaptation via Instruction-Tuned Large Language Mod (https://arxiv.org/abs/2409.08543)
- **What's New**: 본 연구는 멀티모달 데이터(모달리티), 즉 텍스트와 오디오를 대형 언어 모델(LLM)에 통합하여 추천 성능을 향상시키는 방안을 탐구합니다. 기존의 추천 시스템에서 발생하는 콜드 스타트 문제를 해결하고, 효율성을 잃지 않으면서 모델 성능을 유지하기 위해 Low-Rank Adaptation (LoRA) 방식을 도입했습니다.

- **Technical Details**: ATFLRec 프레임워크는 오디오 및 텍스트 데이터를 통합하여 추천 시스템을 구성하며, 여러 LoRA 구성과 모달리티 융합 기법을 활용합니다. 이를 통해 대형 언어 모델의 추천 과제를 효과적으로 조정할 수 있도록 설계되었습니다.

- **Performance Highlights**: ATFLRec은 전통적인 추천 모델 및 그래프 신경망 기반 접근법에 비해 AUC 점수가 높아 성능이 우수함을 입증하였으며, 유사한 LoRA 모듈을 통해 오디오 및 텍스트 데이터를 별도로 미세 조정한 결과 성능이 최적화되었습니다.



### Exploring Information Retrieval Landscapes: An Investigation of a Novel Evaluation Techniques and Comparative Document Splitting Methods (https://arxiv.org/abs/2409.08479)
Comments:
          This article is 16 pages long and includes detailed comparisons of RAG systems and document splitting techniques

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능에 문서의 특성이 미치는 영향을 분석하였습니다. 교과서, 기사, 소설의 구조적 특성이 각각 다른 정보 검색 전략을 필요로 한다는 점을 강조합니다.

- **Technical Details**: 문서 분할 방법의 비교 평가에서는 Recursive Character Splitter가 Token-based Splitter보다 맥락 유지(contextual integrity) 측면에서 더 우수함을 보여주었습니다. 새로운 평가 기법도 도입되어 오픈소스 모델을 활용하여 질문-답변 쌍 데이터셋을 생성하고, 현실적인 검색 시나리오를 시뮬레이션하여 테스트 효율성과 지표 신뢰성을 향상시킵니다.

- **Performance Highlights**: 평가에서는 SequenceMatcher, BLEU, METEOR, BERT Score 등의 가중치 기반 스코어링 메트릭을 사용하여 시스템의 정확성과 적합성을 평가하였습니다. 이 접근법은 RAG 시스템의 정밀도를 평가하는 정교한 기준을 설정하며, 앞으로의 연구는 검색 정확성과 효율성을 향상시키기 위해 청크(chunk) 및 중복(overlap) 크기를 최적화하는 데 초점을 맞출 것입니다.



### Accurate and Fast Estimation of Temporal Motifs using Path Sampling (https://arxiv.org/abs/2409.08975)
Comments:
          Accepted for ICDM'24

- **What's New**: 이 논문에서는 소셜 네트워크 분석과 그래프 마이닝에서 주요한 문제인 motif(모티프) 카운팅에 대한 새로운 알고리즘인 TEACUPS를 제안합니다. TEACUPS는 지향적이고 시간적(graph with timestamps) 그래프에서 복잡한 패턴을 효과적으로 계산할 수 있도록 설계되었습니다.

- **Technical Details**: TEACUPS 알고리즘은 새로운 시간적 경로 샘플링 기법을 활용하여 효율적인 근사 알고리즘을 제공합니다. 이 알고리즘은 시간적 데이터 구조와 경로 샘플링 방법을 결합하여 정확한 motif 카운팅을 위한 편향 없는 추정기를 생성하며, 추정 오차를 경계하는 특성을 갖습니다.

- **Performance Highlights**: TEACUPS는 수억 개의 엣지를 가진 비트코인 그래프에서 1분 이내에 실행되며, 기존의 exact counting 알고리즘이 하루 이상 걸리는 것과 비교됩니다. TEACUPS는 기존의 GPU 기반 exact counting 방법에 비해 평균 30배, 최대 2000배의 속도 향상을 제공하면서도 높은 카운트 추정 정확도를 유지합니다.



New uploads on arXiv(cs.CV)

### An Efficient and Streaming Audio Visual Active Speaker Detection System (https://arxiv.org/abs/2409.09018)
- **What's New**: 본 연구는 Active Speaker Detection (ASD)에서의 실시간 시스템 배치의 문제를 해결하기 위해 두 가지 새로운 시나리오를 제안합니다. 첫째, 미래 컨텍스트 프레임 수를 제한하여 결정이 내려지기 전에 모든 미래 프레임을 처리할 필요가 없도록 합니다. 둘째, 추론 중 모델이 접근할 수 있는 과거 프레임 수를 제한하여 메모리 문제를 해결합니다.

- **Technical Details**: 연구에서는 3D CNN 레이어를 사용한 오디오 및 비주얼 인코더와 Transformer 레이어를 통한 퓨전 모델을 사용합니다. 기존 모델들과의 차별점은 미래 컨텍스트에 의존하지 않고 과거 컨텍스트만을 활용하여 지연 시간을 줄이는 방식입니다. 또한, 실시간 응용에 적합한 메모리 및 지연 시간 범위를 설정하기 위해 다양한 과거 및 미래 컨텍스트의 영향을 분석합니다.

- **Performance Highlights**: 제안된 알고리즘은 상태-of-the-art 순환 모델 (recurrent models)과 비교했을 때, 동일하거나 더 나은 성능을 보여주었으며, 특히 과거 컨텍스트의 크기가 정확도에 미치는 영향이 더 크다는 점을 발견했습니다. CPU에서 프로파일링을 수행한 결과, 우리의 효율적인 아키텍처는 메모리 비용이 계산 비용보다 더 크게 영향을 미친다는 것을 알았습니다.



### Pushing the boundaries of event subsampling in event-based video classification using CNNs (https://arxiv.org/abs/2409.08953)
- **What's New**: 이 논문은 이벤트 카메라의 이벤트 샘플링이 CNN 과제 성능에 미치는 영향을 처음으로 조사했습니다. 이벤트 서브샘플링을 통해 적은 수의 이벤트로도 높은 정확도를 보장할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서는 CNN 모델을 사용하여 다양한 이벤트 분류 데이터셋에서 서브샘플링의 영향을 정량화합니다. 깜짝 놀란 점은, 여러 데이터셋에 걸쳐 비디오당 이벤트 수를 10배 줄여도 정확도가 그리 크게 떨어지지 않는다는 것입니다. 이는 이벤트 카메라를 통한 비주얼 프로세싱이 훨씬 더 전력 효율적일 수 있음을 시사합니다. 또한, 높은 서브샘플링 비율에서 CNN의 훈련 안정성이 떨어지고, 하이퍼파라미터에 대한 민감도가 증가한다는 점을 발견했습니다.

- **Performance Highlights**: 특히, 8개 또는 16개의 이벤트만을 사용하여도 분류 정확도가 우연 수준 이상으로 유지된다는 점은 큰 성과로, 이벤트 샘플링을 통해 처리 전력을 줄일 수 있는 잠재력을 보여줍니다. 이 논문은 CNN 훈련의 불안정성을 분석하고, 서브샘플링 시 하이퍼파라미터 민감도를 평가하기 위한 새로운 메트릭을 도입합니다.



### A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis (https://arxiv.org/abs/2409.08947)
Comments:
          Project site this https URL

- **What's New**: 본 논문에서는 단일 조명 조건에서 캡처된 다중 뷰 데이터를 활용하여 재조명 가능한 radiance fields(방사장)를 만드는 새로운 방법을 제안합니다. 이를 위해 2D diffusion model(확산 모델)에서 추출한 priors(사전 지식)를 사용합니다.

- **Technical Details**: 2D diffusion model을 다중 조명 데이터셋으로 조정(파인 튜닝)한 후, 각 이미지의 주 조명 방향을 Control하여 리라이트된 이미지를 생성합니다. 생성된 데이터셋은 3D Gaussian splatting을 통해 표현된 relightable radiance field와 결합되어, 저주파 조명에 대한 직접적 제어를 가능하게 합니다.

- **Performance Highlights**: 우리는 이 방법이 실제로 모든 장면에서 사용자 인터랙티브한 재조명이 가능함을 증명하며, 실시간으로 복잡한 조명 효과를 생성할 수 있음을 보여주었습니다.



### Pushing Joint Image Denoising and Classification to the Edg (https://arxiv.org/abs/2409.08943)
Comments:
          Accepted paper at the ECCV 2024 workshop on Advances in Image Manipulation (AIM)

- **What's New**: 이 논문에서는 이미지 분류(image classification)와 이미지 노이즈 제거(image denoising)를 결합하여 엣지 디바이스(edge device), 예를 들어 저조도 환경의 보안 카메라로 촬영된 노이즈 이미지의 인간 인식(human perception)을 향상시키는 방법을 제안합니다. 이를 통해 자동 분류 결정의 유효성을 사람이 검증할 수 있게 하는 것이 중요합니다. 기존의 이미지를 노이즈 제거하는 방법들과 달리, 두 작업을 통합한 새로운 아키텍처를 통해 효율성을 최적화하는 것이 핵심입니다.

- **Technical Details**: 이 연구에서는 Neural Architecture Search (NAS) 방법을 수정하여 분류기를 탐색하는 대신 통합 모델을 탐색하고, 목표 지연(latency), 분류 정확도(classification accuracy), 노이즈 제거 성능(denoising performance)을 최적화합니다. 실험 결과, NAS 아키텍처는 수동으로 설계된 대안보다 우수한 성능을 보여주며, 특히 노이즈 제거 및 분류 작업에서 인간 인식 향상에 기여합니다. 제안된 아키텍처는 UNet 기반의 노이즈 제거기와 분류기의 인코더를 공유하는 형태입니다.

- **Performance Highlights**: 모델의 성능 측면에서, 제안된 NAS 기반 아키텍처는 수동 설계된 모델들에 비해 노이즈 제거 및 분류 성능이 상당히 우수합니다. 이러한 성능 개선은 의료 이미징, 감시 시스템, 산업 검사와 같은 다양한 도메인에서 활용될 수 있는 잠재력을 지닙니다.



### Visual Language Tracking with Multi-modal Interaction: A Robust Benchmark (https://arxiv.org/abs/2409.08887)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 Visual Language Tracking (VLT) 작업에 최초로 다중 라운드 상호작용을 도입한 새로운 벤치마크, VLT-MI(Visual Language Tracking with Multi-modal Interaction)를 제안합니다.

- **Technical Details**: VLT-MI는 기존의 VLT 벤치마크를 기반으로 다양한 다중 품질 텍스트를 생성하여 다중 라운드 상호작용을 가능하게 합니다. DTLLM-VLT 모델을 활용하여 상호작용하는 중에 텍스트를 업데이트하고 객체를 회복합니다. 또한, 수정된 바운딩 박스(bbox)와 텍스트를 통해 더 나은 추적 성능을 목표로 합니다.

- **Performance Highlights**: 기존의 VLT 벤치마크와 VLT-MI 간의 비교 실험을 통해 정확성과 견고성을 평가했습니다. 결과적으로, 추적 정확도는 LaSOT에서 기대치를 충족하였으나 다른 벤치마크에서 성능이 저하되었습니다. 상호작용 수는 작업의 난이도가 증가함에 따라 증가하는 경향을 보였으며, 글로벌 인스턴스 추적에서는 최대 229회의 상호작용이 발생했습니다.



### Interactive Masked Image Modeling for Multimodal Object Detection in Remote Sensing (https://arxiv.org/abs/2409.08885)
- **What's New**: 이번 논문에서는 원거리 감지 사진(원거리 감지 이미지)에서의 물체 탐지와 관련하여 새로운 인터랙티브 마스크 이미지 모델링(Interactive Masked Image Modeling, MIM) 방법을 제안합니다. 이 방법은 데이터 불균형 문제를 해결하고 사전 학습(Pre-training) 단계에서 마스크된 토큰 간의 상호작용을 유도하여 감지 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 다중 모드(Self-supervised Learning, SSL) 접근 방식을 사용하여 해당 주제의 기존의 MIM 방법의 한계를 극복하려고 합니다. 크로스 주의(cross-attention) 모듈을 포함함으로써, 마스크된 토큰과 비마스크된 토큰 간의 의존성을 만들어 정보 흐름을 증진시키며, 이를 통해 정밀한 물체 탐지가 가능해집니다. 또한 Swin Transformer를 인코더로 사용하고 YOLOv5를 탐지 네크(head)로 활용하여 다운스트림(object detection) 작업에 적합하도록 조정합니다.

- **Performance Highlights**: 제안된 방식은 종합적인 실험과 부가 분석(ablation studies)을 통해 기존의單지 단일 모드 기반 방법에 비해 개선된 성능을 보였으며, 다양한 원거리 감지 상황에서도 효과적인 성능을 입증합니다.



### Detect Fake with Fake: Leveraging Synthetic Data-driven Representation for Synthetic Image Detection (https://arxiv.org/abs/2409.08884)
Comments:
          Accepted to TWYN workshop at ECCV 2024

- **What's New**: 본 연구는 합성 데이터 전용으로 학습된 일반 목적 특성 표현이 합성 이미지 탐지(SID)에서 효과적임을 입증하였습니다. Vision Transformer(ViT)를 기반으로 한 StableRep 및 SynCLR을 활용하여 실제 이미지를 사용하지 않고도 합성과 실제 이미지를 잘 구분할 수 있다는 점에서 새로운 발견이 있습니다.

- **Technical Details**: 이 연구는 합성 이미지를 탐지하기 위해 Vision Transform에 기반한 최신 모델 StableRep과 SynCLR을 평가합니다. 특히, SynCLR을 활용한 탐지 방법은 CLIP 모델보다 +10.32 mAP와 +4.73% 정확도 향상을 보여줍니다. 이는 합성 데이터에서 학습된 일반 목적의 특성 표현이 효과적으로 기능할 수 있음을 시사합니다.

- **Performance Highlights**: SynCLR을 사용하였을 때, GAN 모델에 대한 성능이 CLIP보다 뛰어남을 확인했습니다. 궁극적으로 합성 이미지 전용으로 학습된 기초 모델과 실제 이미지로 학습된 모델을 앙상블(ensemble)할 경우 탐지 성능이 향상된다는 것을 명확히 했습니다.



### InstantDrag: Improving Interactivity in Drag-based Image Editing (https://arxiv.org/abs/2409.08857)
Comments:
          SIGGRAPH Asia 2024. Project webpage at this https URL

- **What's New**: InstantDrag는 복잡한 최적화 없이 실시간 이미지 편집을 가능하게 하는 새로운 기술입니다. 이 기술은 오직 이미지와 드래그 지침만 필요로 합니다.

- **Technical Details**: InstantDrag는 두 개의 네트워크로 구성됩니다: 드래그 조건의 광학 흐름 생성기(FlowGen)와 광학 흐름 조건의 확산 모델(FlowDiffusion)입니다. FlowGen은 사용자 입력에서 밀집한 광학 흐름을 생성하며, FlowDiffusion은 이 생성된 모션을 기초로 고품질 편집을 수행합니다.

- **Performance Highlights**: InstantDrag는 기존 방법들에 비해 최대 75배 빠른 속도로,GPU 메모리 소모를 5배 줄이며, 마스크나 텍스트 프롬프트 없이도 실제 이미지에 대해 포토리얼리스틱한 편집을 제공합니다.



### DeCLIP: Decoding CLIP representations for deepfake localization (https://arxiv.org/abs/2409.08849)
Comments:
          Accepted at Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 본 논문에서는 기존의 deepfake 탐지 방법론의 한계를 극복하기 위해 CLIP의 자기 지도 학습(self-supervised) 표현을 활용하는 새로운 접근 방식인 DeCLIP을 도입합니다. 이 방법은 부분적으로 조작된 이미지에서의 변조 위치를 정확하게 식별하도록 설계되었습니다.

- **Technical Details**: DeCLIP은 대규모로 사전 훈련된 CLIP 모델에서 추출한 피처를 사용하여 변조된 영역을 구체적으로 찾아내는 데 중점을 두며, 이 데이터를 기반으로 한 합성곱 디코더(convolutional decoder)와 결합하여 성능을 향상시킵니다. 데이터셋은 Dolos로, 다양한 인페인팅(inpainting) 방법을 사용하여 조작된 얼굴 이미지를 포함합니다. CLIP의 피처는 로컬 조작 이미지에서의 변조를 탐지하고 지역화하는 능력을 나타내지만, 이미지 전반에 영향을 미치는 깊은 생성자(deep generators)에 대해서는 도전적입니다.

- **Performance Highlights**: 연구 결과, DeCLIP은 LDM(Latent Diffusion Model)으로 인페인팅된 이미지에서 높은 정확도의 변조 지역화를 달성했으며, 이는 다양한 생성자에 대한 일반화 성능이 뛰어나도록 하였습니다. 특히, CLIP 기반의 접근 방식이 기존 방법들보다 더 안정적인 일반화를 제공함을 보여주었습니다.



### Kinect Calibration and Data Optimization For Anthropometric Parameters (https://arxiv.org/abs/2409.08847)
- **What's New**: 본 연구에서 Mikrosoft Kinect 센서를 보정하고 인체 골격 특징을 최적화하는 새로운 방법을 제안하였습니다.

- **Technical Details**: Kinect 센서는 씬의 깊이 이미지(depth images)와 인체 관절의 3D 좌표(3D coordinates)를 획득하여 인체 측정 특징(anthropometric features)을 쉽게 추출할 수 있지만, 관절 간 거리와 Kinect 센서 위치에 따라 자료가 불안정하게 변동합니다. 본 연구에서는 이러한 문제를 해결하기 위한 보정(calibration) 및 최적화(optimization) 기술을 제시했습니다.

- **Performance Highlights**: 제안된 방법은 효과적임을 나타내며, 더 일반적인 시나리오에서 추가 연구의 가치가 있음을 보여줍니다.



### Direct-CP: Directed Collaborative Perception for Connected and Autonomous Vehicles via Proactive Attention (https://arxiv.org/abs/2409.08840)
Comments:
          7 pages

- **What's New**: 이 논문은 현재의 Collaborative perception (CP) 시스템의 한계를 극복하고자, Direction-aware CP 시스템인 Direct-CP를 제안합니다. 기존 CP 방법들은 모든 방향에서 동등하게 인지 범위를 확장하는 반면, 본 연구는 특정 방향에 집중하여 인지 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: Direct-CP는 RSU(roadside units) 지원 방향 마스킹 메커니즘과 방향 인식 선택적 주의(attention) 모듈을 도입하여, 특정 방향에 대한 인지 성능을 더욱 개선합니다. 또한, 방향 가중 탐지 손실(DWLoss)을 정의하여 예측 결과와 실제 값 간의 방향별 인지 오차를 측정합니다.

- **Performance Highlights**: 실험 결과, Direct-CP는 관심 있는 방향에서 19.8% 높은 지역 인지 정확도를 달성하였으며, 협업 3D 객체 탐지 작업에서 전체 인지 정확도도 2.5% 향상되었습니다.



### Breaking reCAPTCHAv2 (https://arxiv.org/abs/2409.08831)
Comments:
          10 pages. Accepted at COMPSAC 2024

- **What's New**: 이 연구에서는 Google의 reCAPTCHAv2 시스템에서 자동화된 Captcha 해소의 효율성을 살펴보았습니다. 최신 YOLO 모델을 활용하여 이미지를 분할(segmentation)하고 분류(classification)하는 기법을 사용해 보았으며, 100%의 captcha 문제를 해결할 수 있었다는 점이 주목할 만합니다. 이는 이전 연구에서 68-71%를 해결했던 것과 비교할 때 혁신적인 성과입니다.

- **Technical Details**: 이 연구는 YOLO v8 모델을 활용하여 reCAPTCHAv2의 이미지 기반 캡차를 해결하는 데 초점을 맞추었습니다. 연구 결과, 기계가 해결해야 하는 챌린지 수에서 사람과의 차이가 없다는 것을 발견했습니다. 추가로, reCAPTCHAv2는 사용자가 인간인지 아닌지를 평가하는 데 쿠키(cookie)와 브라우저 히스토리 데이터를 상당히 활용하고 있다는 증거가 발견되었습니다.

- **Performance Highlights**: YOLO v8 모델을 활용한 결과, 캡차를 100% 해결할 수 있었고, 이는 이전의 많은 자동화 작업과 비교할 때 매우 뛰어난 성과입니다. 또한, 사용자는 브라우징 데이터와 행동 기반의 증거로 인간으로서 확인될 수 있지만, 이로 인해 웹 서비스 접근이 제한될 위험이 있다는 점이 중요합니다.



### Pathfinder for Low-altitude Aircraft with Binary Neural Network (https://arxiv.org/abs/2409.08824)
- **What's New**: 본 연구에서는 저고도 항공기에 장착된 센서를 이용해 OpenStreetMap (OSM) 지도를 자동 생성하는 OSM 메이커를 제안합니다. 이 방식은 기존 OSM의 미비점을 보완하여 자율적으로 출입 가능한 도로 네트워크의 전체 구성을 생성할 수 있습니다.

- **Technical Details**: 제안하는 OSM 메이커는 LiDAR와 카메라 데이터를 기반으로 한 이진 이중 스트림 도로 세분화 모델을 포함합니다. UNet 아키텍처를 통한 다중 스케일 기능 추출이 적용되며, 이미지와 포인트 클라우드 특성을 통합하기 위한 주의 기반 게이트 블록이 설계되었습니다. 또한, 변형된 비전 트랜스포머(ViT) 아키텍처를 인코더로 사용하는 이진화 흐름을 통해 모델의 효율성을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 경로 탐색기 방법은 저고도 센서로부터 경로를 찾는 데 있어 SOTA 정확도를 달성하였으며, 세분화된 도로 스켈레톤을 기반으로 완전한 OSM 이전 지도를 생성할 수 있습니다.



### Task-Specific Data Preparation for Deep Learning to Reconstruct Structures of Interest from Severely Truncated CBCT Data (https://arxiv.org/abs/2409.08800)
Comments:
          Published in the CT-Meeting 2024 proceeding. arXiv admin note: text overlap with arXiv:2108.13844

- **What's New**: 본 연구에서는 CBCT(콘빔 컴퓨터 단층촬영)의 유용성을 높이기 위해, FOV(시야 범위) 내에서의 특정 구조물 복원에 중점을 두는 작업 특화 데이터 준비 방법을 제안합니다.

- **Technical Details**: 이번 작업은Deep Learning(딥러닝) 기반의 pix2pixGAN을 활용하여, 제한된 FOV에서 특정 구조물 (e.g., Costal rib)만 복원할 수 있도록 학습하는 전략을 사용합니다. 신경망은 전통적인 훈련 방법 대신, SOI(Structures of Interest)만을 focus하여 복원하는 방식을 채택하여 더 정확한 결과를 도출합니다.

- **Performance Highlights**: 예비 실험 결과, 제안한 작업 특화 데이터 준비 방법을 사용하여 pix2pixGAN이 모든 갈비뼈를 신뢰성 있게 복원할 수 있음을 보여주었으며, 제한된 CBCT 데이터에서의 구조물 복원 시 false positive 및 false negative의 위험을 줄일 수 있었습니다.



### Contactless Fingerprint Recognition Using 3D Graph Matching (https://arxiv.org/abs/2409.08782)
- **What's New**: 이번 연구에서는 기존의 contactless fingerprint 인식 알고리즘과의 차별점을 강조하며, 최초로 3D 정보를 포괄적으로 활용하는 monocular contactless fingerprint 인식 알고리즘을 제안합니다. 이 방법은 contactless fingerprint의 3D 특성을 추출하고 이를 바탕으로 3D 매칭을 수행합니다.

- **Technical Details**: 제안된 방법은 입력된 contactless fingerprint로부터 3D 형태 모델과 3D minutiae feature를 회복합니다. 이 후, 추출된 3D 특징을 기반으로 3D 공간에서 3D graph matching을 실행합니다. 특히, 전체 알고리즘 과정이 실제 3D 공간에서 진행됨으로써, contactless fingerprint의 본질적 3D 특성을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 contactless fingerprint의 매칭 정확도를 성공적으로 향상시키며, 다양한 자세(poses)에서도 안정된 성능을 보여 기존 contactless fingerprint 인식 알고리즘보다 뛰어난 결과를 보입니다.



### Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry (https://arxiv.org/abs/2409.08769)
Comments:
          Accepted to ECCV 2024 2nd Workshop on Vision-Centric Autonomous Driving (VCAD)

- **What's New**: 최근의 연구에서는 Visual-Inertial Odometry (VIO) 분야에서 기존의 RNN 기반 방법을 대체하는 효과적인 접근법으로 causal visual-inertial fusion transformer (VIFT)을 제안합니다. 이 방법은 Transformers의 attention 메커니즘을 활용하여 역사적 데이터를 더 효과적으로 활용하며, 특히 소규모 데이터에서 성능을 향상시킵니다.

- **Technical Details**: VIFT는 두 개의 연속된 이미지 및 IMU 데이터를 입력으로 받아, 시각적 및 관성 정보를 융합하여 자세(pose)를 추정합니다. Latent visual-inertial feature vectors를 사용하여 pose 추정의 정확성을 높이고, Riemannian manifold (리만 기하학) 최적화 기법을 적용하여 회전 회귀(rotation regression)를 더 효과적으로 처리합니다. 이러한 구조는 end-to-end 방식으로 학습 가능하며, 단일 카메라와 IMU만으로 추론이 가능합니다.

- **Performance Highlights**: 실험 결과 VIFT는 KITTI 데이터셋에서 이전 방법들과 비교하여 최첨단 성능을 기록하였으며, 깊은 VIO 네트워크의 정확성을 향상시키는 데 성공했습니다.



### Uncertainty and Generalizability in Foundation Models for Earth Observation (https://arxiv.org/abs/2409.08744)
Comments:
          A large ablation study measuring uncertainty and spatial generalizability with 8 foundation models, 11 world regions and 7 downstream tasks

- **What's New**: 이 연구는 제한된 레이블 예산을 상황에서 특정 지역(Area of Interest, AOI)의 VM(vegetation coverage) 추정을 위한 다운스트림 작업을 설계하는 관점을 취합니다. 기존의 Foundation Model(FM)을 활용하여, 레이블이 풍부한 다른 AOI에서 모델을 학습할 것인지, 또는 타겟 AOI의 레이블을 나누어 학습과 검증을 진행할 것인지 결정해야 합니다.

- **Technical Details**: 우리는 Sentinel 1 및 Sentinel 2 데이터를 사용하여 11개의 AOI에 걸쳐 ESA World Cover 제품의 클래스를 다운스트림 작업으로 설정하여 8개의 FM을 비교하였습니다. 500K 이상의 간단한 선형 회귀 모델을 훈련시키며, 공간 일반화(spatial generalizability)의 한계와 FM의 힘을 보여줍니다. 더불어, 레이블 노력(labeling effort)을 증가시키면 불확실성(uncertainty)이 줄어들고, 다양한 샘플링 방법이 다른 결과를 제공합니다.

- **Performance Highlights**: 최종 결과는 특정 AOI와 작업에 따라 성능과 불확실성이 크게 변동함을 보여주며, 일부 경우에서는 FM embedding을 직접적으로 사용하여 상당한 예측력을 얻을 수 있음을 나타냅니다. 여러 상황에서 0.9 이상의 상관계수(correlation coefficient)가 달성되었으며, FM 사용 시 다운스트림 작업 설계에 대한 정보에 기반한 의사결정을 지지합니다.



### Precision Aquaculture: An Integrated Computer Vision and IoT Approach for Optimized Tilapia Feeding (https://arxiv.org/abs/2409.08695)
Comments:
          8 pages, 6 figures, 3 tables, 21th International Conference on Informatics in Control, Automation, and Robotics

- **What's New**: 전통적인 어류 양식에서는 비효율적인 피딩으로 인해 환경 문제가 발생하고 생산성이 저하되는 경우가 많습니다. 본 연구에서는 컴퓨터 비전(Computer Vision)과 IoT 기술을 결합한 혁신적인 시스템을 개발하여 Tilapia(틸라피아) 어류의 정확한 사료 공급을 실현했습니다.

- **Technical Details**: 실시간 IoT 센서를 사용해 수질 매개변수를 모니터링하고, YOLOv8 모델을 활용해 어류 크기와 숫자를 분석하여 최적의 사료 양을 결정합니다. 두 대의 카메라를 사용하여 pH 수준과 용존 산소를 모니터링하고 여러 급수를 위한 데이터 수집이 이루어집니다. 데이터는 모바일 애플리케이션으로 전달되어 쉽게 접근할 수 있습니다.

- **Performance Highlights**: 모델은 3,500개의 주석 이미지에서 94%의 정밀도를 달성하였으며, 이 접근 방식은 기존 양식과 비교하여 생산량을 최대 58배 증가시킬 수 있는 가능성을 제시합니다. 또한, 모든 모델, 코드, 데이터셋은 오픈 소스(Open Source)로 제공됩니다.



### Autoregressive Sequence Modeling for 3D Medical Image Representation (https://arxiv.org/abs/2409.08691)
- **What's New**: 본 연구에서는 자가 회귀(autoregressive) 사전 학습 프레임워크를 통해 3D 의료 이미지 표현을 학습하는 혁신적인 방법을 소개합니다. 이 방법은 다양한 3D 의료 이미지를 공간적, 대비(contrast) 및 의미적 상관관계에 따라 순서화하여 서로 연결된 시각적 토큰으로 다룹니다.

- **Technical Details**: 우리의 접근 방식은 이미지의 패치(patch) 시퀀스를 작성하고, 이를 통해 시각적 토큰의 다음 예측을 진행하여 3D 의료 이미지 내의 컨텍스트 정보를 이해하고 통합할 수 있도록 합니다. 또한, 랜덤 스타트업 전략을 구현하여 토큰 간 관계를 과대 평가하지 않도록 하고, 학습의 견고성을 개선합니다.

- **Performance Highlights**: 우리의 방법은 9개의 하위 작업에서 다른 방법들보다 우수한 성능을 보이며, CT 및 MRI에서 장기와 종양의 분할(segmentation) 및 COVID-19와 폐 결절(nodule) 분류의 정확도가 각각 2.1% 및 4%-6% 향상되는 것을 보여주었습니다.



### GenMapping: Unleashing the Potential of Inverse Perspective Mapping for Robust Online HD Map Construction (https://arxiv.org/abs/2409.08688)
Comments:
          The source code will be publicly available at this https URL

- **What's New**: 본 논문에서는 해외에서 채택되고 있는 Online High-Definition (HD) 지도 모델의 새로운 가능성을 제안합니다. GenMapping이라는 이름의 범용 지도 생성 프레임워크를 설계했으며, 이를 통해 기존의 HD 지도 생성 모델이 시각적 센서 매개변수에 종속되지 않도록 하였습니다.

- **Technical Details**: GenMapping은 주 브랜치와 두 개의 보조 브랜치를 포함하는 트라이애딕 시너지 아키텍처로 구성됩니다. 주 브랜치는 State Space Model (SSM)을 사용해 IPM 이미지의 지역 왜곡을 완화하며, 두 보조 브랜치는 동적 및 정적 객체 간의 밀접한 연관 정보를 학습합니다. Cross-View Map Learning (CVML) 및 Bidirectional Data Augmentation (BiDA) 모듈이 함께 사용되어 모델의 일반화 성능을 강화합니다.

- **Performance Highlights**: 실험 결과, GenMapping은 publicly available 한 nuScenes 데이터셋에서 뛰어난 성능을 보였으며, 다른 최신 기법들과 비교하여도 높은 일반화 능력을 나타내었습니다. 특히, nuScenes에서 Argoverse로의 전이 학습 테스트에서도 두각을 나타냈습니다.



### AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius (https://arxiv.org/abs/2409.08669)
Comments:
          SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 03-06, 2024, Tokyo, Japan

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS) 기술의 렌더링 속도를 310% 가속화하는 새로운 방법인 AdR-Gaussian을 제안합니다. 이를 통해 Gaussian의 품질 저하 없이 고속 렌더링을 가능하게 하고자 합니다.

- **Technical Details**: AdR-Gaussian 방법은 렌더 단계에서의 직렬 Gaussian culling 일부를 사전 처리 단계로 이동시켜 병렬 culling을 가능하게 하며, 각 Gaussian의 렌더링 픽셀 범위를 좁히기 위해 adaptive radius를 사용합니다. axis-aligned bounding box를 기반으로 한 사전 처리 과정 또한 도입하여, Gaussian 렌더링의 성능을 크게 향상시킵니다.

- **Performance Highlights**: 세 가지 데이터셋에서 실험을 통해 AdR-Gaussian은 평균 310%의 렌더링 속도 향상을 보여주었으며, Mip-NeRF360 데이터셋에서 단일 NVIDIA RTX3090 GPU에서 평균 590FPS의 렌더링 속도를 달성했습니다.



### Test-time Training for Hyperspectral Image Super-resolution (https://arxiv.org/abs/2409.08667)
Comments:
          Accepted to T-PAMI

- **What's New**: 본 연구에서는 Hyperspectral 이미지(Hyperspectral Image, HSI)의 초해상도(Super-Resolution, SR) 문제를 다루기 위해 새로운 test-time training 방법을 제안합니다. 이 방법은 자가 학습(self-training) 프레임워크를 활용하여 보다 정확한 pseudo-labels와 LR-HR 관계를 생성함으로써 모델 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 새로운 네트워크 아키텍처를 통해 스펙트럼 밴드 간 상호작용을 모델링하지 않고 HSI SR을 학습할 수 있도록 하며, Spectral Mixup이라는 데이터 증강(data augmentation) 방법을 도입하여 테스트 시 훈련 데이터의 다양성을 증가시키는 데 기여합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 광범위한 실험을 통해 제안된 방법이 사전 학습된(pre-trained) 모델의 성능을 상당히 향상시키고, 다른 경쟁 방법들에 비해 HSI SR에서 우수한 성능을 보여주었습니다.



### Dense Point Clouds Matter: Dust-GS for Scene Reconstruction from Sparse Viewpoints (https://arxiv.org/abs/2409.08613)
- **What's New**: 새로운 프레임워크 Dust-GS를 제안하여 3D Gaussian Splatting(3DGS)의 한계를 극복하고, 희소 관점(포인트 뷰프린트)에서 점 구름(point cloud) 초기화를 개선한다.

- **Technical Details**: Dust-GS는 깊이 기반 마스킹 기법과 점 구름 최적화를 결합한 하이브리드 전략을 사용하여 희소 입력 데이터에서도 효과적으로 작업한다. 또한, 동적 깊이 마스킹 메커니즘을 통해 먼 객체의 고주파 노이즈와 아티팩트를 필터링한다.

- **Performance Highlights**: Dust-GS는 여러 벤치마크 데이터 세트에서 기존 3DGS 방법을 초월하며, 희소 관점 환경에서 뛰어난 장면 재구성 품질을 달성하고 입력 이미지 수를 줄인다.



### Knowledge-Enhanced Facial Expression Recognition with Emotional-to-Neutral Transformation (https://arxiv.org/abs/2409.08598)
- **What's New**: 이 논문은 얼굴 표정 인식(FER)에서 기존의 이산 레이블 기반의 방법을 개선하기 위해 비전-언어 모델(Vision-Language Model, VLM)에서 생성된 텍스트 임베딩을 사용하는 새로운 FER 방법을 제안합니다. 이 방법은 감정에서 중립으로의 변환을 포함하여 보다 효과적으로 얼굴 표정의 표현을 학습하도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 얼굴 표정 표현과 텍스트 임베딩 간의 유사성을 매칭하는 과정으로 FER 문제를 공식화합니다. 또한 러셀의 서클럼플렉스 모델을 바탕으로 얼굴 표정 표현에서 중립 표현을 파생해와서 이 텍스트 차이를 기반으로 표현을 변환합니다. 마지막으로, 자기 대조 목표(self-contrast objective)를 통해 얼굴 표정 표현을 텍스트 임베딩과 더 가깝게, 중립 표현과는 더 멀게 유지합니다.

- **Performance Highlights**: 제안된 방법은 ResNet-18 및 Swin-T와 같은 다양한 사전 훈련된 비주얼 인코더를 사용하여 4개의 도전적인 얼굴 표정 데이터셋에서 평가되었습니다. 실험 결과, 본 방법은 기존의 최첨단 FER 방법들보다 뛰어난 성능을 보였습니다.



### Optimizing 4D Lookup Table for Low-light Video Enhancement via Wavelet Prior (https://arxiv.org/abs/2409.08585)
- **What's New**: 본 연구에서는 Wavelet-priori를 활용한 4D Lookup Table (WaveLUT) 방법을 제안하여 저조도 환경에서 색상 일관성을 개선하고 낮은 지연(latency)으로 색상 매핑의 정확성을 높이는 방법을 제시합니다.

- **Technical Details**: WaveLUT는 Wavelet 변환을 사용하여 저조도 비디오 데이터의 저주파 영역을 추출하여 조명 프라이어(lighting prior)를 생성하고 이를 동적으로 융합(fusion)하여 강력한 색상 보정을 달성합니다. 또한, 다중 모달 성분에 기반한 텍스트 구동 이미지 복원 방법을 통해 밝기와 내용을 동적으로 균형을 맞춥니다.

- **Performance Highlights**: Benchmark 데이터셋에서의 실험 결과, 제안한 방법이 기존 방법보다 색상 공간 인식 능력을 향상시키고, PSNR(피크 신호 대 잡음 비율)에서 경쟁력 있는 성능을 보이며, 실시간 개선에서도 높은 효율성을 유지함을 검증했습니다.



### ChangeChat: An Interactive Model for Remote Sensing Change Analysis via Multimodal Instruction Tuning (https://arxiv.org/abs/2409.08582)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 논문에서는 Remote Sensing (RS) 변화를 분석하기 위한 새로운 모델인 ChangeChat을 소개합니다. 이 모델은 전통적인 변화 탐지 기법의 한계를 극복하고 사용자가 다양한 질문을 할 수 있도록 설계된 최초의 bitemporal vision-language model (VLM)입니다.

- **Technical Details**: ChangeChat은 멀티모달 지침 조정(multi-modal instruction tuning)을 기반으로 하여 변화 캡셔닝, 카테고리 특화 정량화(category-specific quantification), 변화 위치 지정(change localization)과 같은 복잡한 쿼리를 처리할 수 있습니다. 이 모델은 87,195개의 지침으로 구성된 ChangeChat-87k 데이터셋을 사용하여 교육되었습니다.

- **Performance Highlights**: ChangeChat은 특정 작업에서 기존의 state-of-the-art (SOTA) 방법과 동등하거나 그보다 우수한 성능을 달성했으며, 최신 일반 도메인 모델인 GPT-4보다도 현저하게 높은 성능을 자랑합니다.



### HTR-VT: Handwritten Text Recognition with Vision Transformer (https://arxiv.org/abs/2409.08573)
Comments:
          Accepted to Pattern Recognition

- **What's New**: 이번 논문에서는 수기 텍스트 인식을 위한 Vision Transformer (ViT) 적용을 탐구하며, 데이터의 효율성을 강조하는 새로운 방법을 제안합니다. 기존 트랜스포머 기반 모델이 대규모 데이터에 의존해야 했던 제약을 해결하기 위해, 표준 트랜스포머의 인코더만을 사용하여 모델을 개선했습니다.

- **Technical Details**: 연구는 Convolutional Neural Network (CNN)을 특징 추출에 사용하고, Sharpness-Aware Minimization (SAM) 최적화를 통해 모델이 안정적으로 수렴하도록 했습니다. 또한 span mask 기법을 도입하여 연결된 특징을 마스킹하여 일반화를 촉진하는 효과를 검증했습니다.

- **Performance Highlights**: IAM, READ2016 같은 소규모 데이터셋에서 전통적인 CNN 기반 모델보다 뛰어난 성능을 기록했으며, LAM 데이터셋에서는 새로운 기준을 세웠습니다. 이 방법은 대규모 수기 텍스트 인식 작업에서 최첨단 성능을 달성하는 데 성공했습니다.



### DiffFAS: Face Anti-Spoofing via Generative Diffusion Models (https://arxiv.org/abs/2409.08572)
Comments:
          ECCV 24

- **What's New**: 이 논문에서는 Face Anti-Spoofing(FAS)의 새로운 접근 방식인 DiffFAS 프레임워크를 제안하여, 이미지 스타일(image style)과 이미지 품질(image quality)로 나뉘는 도메인 전이(domain shift)의 문제를 해결합니다.

- **Technical Details**: DiffFAS는 고충실도 생성(generation)을 수행하며, 이를 통해 쉽게 수집할 수 있는 실시간 얼굴 이미지를 기반으로 정확한 라벨을 가진 공격 얼굴 이미지를 생성합니다. 또한, Spoofing Style Fusion Module(STFM)를 도입하여 텍스처 보존과 정체성 일관성(identity-consistency)을 유지합니다.

- **Performance Highlights**: DiffFAS 프레임워크는 cross-domain 및 cross-attack FAS 프로토콜에서 업계 최고의 성능을 달성하였으며, WMCA unseen 프로토콜에서 평균 4.4% ACER 개선을 보여 결과적으로 생성 방법의 효과성을 입증합니다.



### Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection (https://arxiv.org/abs/2409.08566)
- **What's New**: 본 논문은 Continual Test Time Adaptation (CTTA)라는 새로운 접근 방식을 제안하여, 도메인 변화에 효과적으로 적응할 수 있도록 고안된 Hybrid-TTA 프레임워크를 소개합니다. 이 방법은 Dynamic Domain Shift Detection (DDSD)와 Masked Image Modeling based Adaptation (MIMA)을 결합합니다.

- **Technical Details**: Hybrid-TTA는 이미지별로 적절한 파인튜닝 전략을 동적으로 선택하여 최적화된 적응 성능을 제공합니다. DDSD는 입력 이미지의 예측 불일치를 통해 도메인 변화를 감지하고, 이를 기반으로 FT와 ET 방법을 동적으로 전환하여 모델을 최적화합니다. MIMA는 Masked Image Modeling을 활용하여 도메인에 구애받지 않는 특징을 추출합니다.

- **Performance Highlights**: Hybrid-TTA는 Cityscapes-to-ACDC 벤치마크 데이터셋에서 mIoU가 1.6%p 향상되는 성과를 거두었으며, 기존의 최신 방법들을 능가하는 강력한 솔루션을 제공합니다.



### CSS: Overcoming Pose and Scene Challenges in Crowd-Sourced 3D Gaussian Splatting (https://arxiv.org/abs/2409.08562)
- **What's New**: Crowd-Sourced Splatting (CSS)라는 새로운 3D Gaussian Splatting (3DGS) 파이프라인을 소개합니다. 이 방법은 군중 소싱 이미지로부터 포즈가 없는 장면 재구성을 보다 용이하게 하는 것을 목표로 하고 있습니다. 기존의 3D 기술들이 겪는 문제들을 극복하기 위해 강력한 기하학적 프라이어(geometric priors)와 고급 조명 모델링(illumination modeling)을 통합하였습니다.

- **Technical Details**: CSS는 (1) 전문가 모델과 방대한 2D 기하학적 프라이어를 활용하여 카메라 포즈 부재와 이미지 조건 불일치를 극복하는 강력한 초기화 메커니즘을 제공합니다. (2) 상이한 조명 조건을 조화롭게 하기 위해 고차원 구형 고조파(high-order spherical harmonics)를 활용한 고급 조명 모델을 사용합니다. (3) 다양한 실내 및 실외 환경을 아우르는 CSScenes라는 포괄적인 데이터셋을 개발하여 벤치마크를 제공합니다.

- **Performance Highlights**: CSS는 복잡한 군중 소싱 상황에서도 고품질의 3DGS 모델을 생성할 수 있으며, 기존 방법에 비해 명확한 개선을 보여줍니다. 이로 인해 AR, VR, 그리고 대규모 3D 재구성 시스템에서 보다 정확하고 유연한 응용에 기여할 수 있습니다.



### DICS: Find Domain-Invariant and Class-Specific Features for Out-of-Distribution Generalization (https://arxiv.org/abs/2409.08557)
- **What's New**: 본 논문에서는 DICS 모델을 제안하여 도메인 불변(Domain-Invariant) 및 클래스 특이(Class-Specific) 특징을 추출하는 기법을 소개합니다. 특히, 도메인 간 특징 정합성과 클래스 별 구분을 강조하여 OOD(Out-of-Distribution) 상황에서 성능 저하를 방지하고자 합니다.

- **Technical Details**: DICS 모델은 Domain Invariance Testing (DIT)과 Class Specificity Testing (CST)을 통해 도메인 관련 특징을 제거하고, 클래스 관련 특징의 유사성을 극대화합니다. DIT은 소스 도메인의 특징을 분석하여 도메인 간 유사한 특징을 확인하며, CST는 과거 학습된 특징과 비교하여 클래스 간의 차별성을 강화하는 소프트 레이블을 생성합니다.

- **Performance Highlights**: DICS는 PACS, OfficeHome, TerraIncognita, DomainNet 등 여러 데이터셋에서 경쟁적인 정확도를 기록, 최신 방법들과 성능을 비교하였으며, OOD 작업에서 우수한 효과를 입증하였습니다.



### GroundingBooth: Grounding Text-to-Image Customization (https://arxiv.org/abs/2409.08520)
- **What's New**: 새로운 연구 결과로는 GroundingBooth라는 프레임워크가 도입되어, 텍스트-이미지 커스터마이제이션에서 제로샷(instance-level) 공간 구속을 달성하였다는 것입니다. 이 프레임워크는 전경 주제와 배경 객체 모두에 대해 정밀한 레이아웃 정렬과 아이덴티티(Identity) 보존을 가능하게 합니다.

- **Technical Details**: GroundingBooth는 텍스트-이미지 모델을 기반으로 하여 새로운 텍스트-이미지 조인드 기초 모듈을 구축하고, 마스크된 크로스 어텐션(layer) 구조를 도입하여 각 트랜스포머 블록에서 전경 생성과 텍스트 구동 배경 생성을 분리하였습니다. 이러한 구조적 설계를 통해 단일 주제와 다중 주제 커스터마이즈를 지원하며, 입력된 경계 상자와 정확하게 정렬된 객체와 주제를 생성할 수 있습니다.

- **Performance Highlights**: 모델은 텍스트-이미지 정렬, 아이덴티티 보존, 레이아웃 정렬에서 강력한 결과를 보였으며, 여러 객체의 커스터마이징을 동시에 수행할 수 있는 기능을 갖추고 있습니다. 기존의 방법들과 비교했을 때, GroundingBooth는 주제 기반 전경 생성과 텍스트 기반 배경 생성을 성공적으로 달성하는 첫 번째 사례입니다.



### Anytime Continual Learning for Open Vocabulary Classification (https://arxiv.org/abs/2409.08518)
Comments:
          To appear at ECCV 2024 as Oral presentation

- **What's New**: 저희는 오픈 어휘 이미지 분류를 위한 Anytime continual learning (AnytimeCL) 접근 방식을 제안합니다. 이 방법은 배치 트레이닝과 딱딱한 모델의 제약에서 벗어나 언제든지 새로운 레이블 집합을 예측하고, 언제든지 학습 샘플을 받을 때 효율적으로 최신화 및 개선할 수 있는 시스템을 요구합니다.

- **Technical Details**: 저희는 부분적으로 미세 조정된 모델의 예측과 고정된 오픈 어휘 모델 간의 동적 가중치를 제안하여, 특정 레이블의 하위 집합에 대해 학습 샘플이 제공되면 지속적인 개선이 가능하도록 합니다. 또한, model prediction을 감소시키는 attention-weighted PCA 압축 기법을 도입하여 저장 및 계산을 줄이며, 모델의 정확도에 미치는 영향은 최소화합니다.

- **Performance Highlights**: 저희의 방법은 Zhu et al. [64]의 오픈 어휘 지속 학습 벤치마크에서 모든 설정 및 단계에서 뛰어난 성능을 보였습니다. 데이터 증가, 클래스 증가, 태스크 증가 학습 및 제로 샷 예측 모두에서 개선된 결과를 도출하였습니다. 여러 혁신 사항들을 포함하여, 즉각적인 학습 업데이트, 고정된 레이블 임베딩을 통한 부분 미세 조정, 그리고 정규화 손실 항을 통한 'none of the above' 예측을 가능하게 하여 System의 정확도와 효율성을 증대시켰습니다.



### AWF: Adaptive Weight Fusion for Enhanced Class Incremental Semantic Segmentation (https://arxiv.org/abs/2409.08516)
Comments:
          10 pages,6 figures

- **What's New**: 본 논문에서는 기존의 EWF 메소드에 비해 더 유연하고 적응력 있는 Adaptive Weight Fusion (AWF) 전략을 제안합니다. 이를 통해 이전의 지식과 새로운 클래스를 더 효과적으로 통합할 수 있습니다.

- **Technical Details**: AWF는 훈련 과정 중에 동적으로 최적화되는 퓨전 파라미터 alpha를 도입하여, 다양한 데이터 특성 변화에 능동적으로 반응합니다. 이는 기존의 EWF보다 더 나은 성능을 발휘하는 데 기여합니다.

- **Performance Highlights**: AWF는 PASCAL VOC 및 ADE20K와 같은 여러 CISS 벤치마크에서 기존 EWF 메소드보다 1% 이상 성능을 개선하며, 다양한 시나리오에서 최첨단 성능을 달성했습니다.



### Mamba-YOLO-World: Marrying YOLO-World with Mamba for Open-Vocabulary Detection (https://arxiv.org/abs/2409.08513)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: Mamba-YOLO-World는 YOLO 기반의 새로운 Open-Vocabulary Detection (OVD) 모델로, 기존의 YOLO-World 모델의 한계를 극복하기 위해 MambaFusion Path Aggregation Network (MambaFusion-PAN)을 도입했습니다. 이 모델은 다양한 입력 시퀀스를 활용하여 효율적으로 특징을 융합합니다.

- **Technical Details**: Mamba-YOLO-World는 YOLOv8을 기반으로 하며, Darknet Backbone과 CLIP Text Encoder를 사용합니다. MambaFusion-PAN은 병렬 및 직렬 선택적 스캔 알고리즘을 활용한 State Space Model을 기반으로 하여 특징 융합을 수행합니다. 고유의 선형 복잡도를 통해 다양한 모달리티 간의 정보를 효과적으로 통합합니다.

- **Performance Highlights**: Mamba-YOLO-World는 COCO 및 LVIS 벤치마크에서 기존 YOLO-World보다 우수한 성능을 보이며, 파라미터 수와 FLOPs가 유사한 상황에서도 인지적 범위와 정확성에서 향상된 결과를 냈습니다. 더 나아가, 기존의 OVD 방법들을 능가하는 성능을 기록했습니다.



### CasDyF-Net: Image Dehazing via Cascaded Dynamic Filters (https://arxiv.org/abs/2409.08510)
Comments:
          9 pages, 9 figures

- **What's New**: 본 연구에서는 이미지 디헤이징(Dehazing)을 위해 다이나믹 필터와 잔차 멀티스케일 블록(Residual Multiscale Block, RMB)을 활용한 새로운 구조인 CasDyF-Net을 제안합니다. 이 방식은 입력 특성에 따라 동적으로 필터를 생성하여 망의 분기를 조정하여 해상도와 주파수 대역을 효과적으로 처리합니다.

- **Technical Details**: CasDyF-Net 구조는 U자형 아키텍처로 구성되며, 이는 주어진 입력 이미지를 여러 분기로 나누고, 각 분기는 DFS(Dynamic Filter Separation), LFB(Local Fusion Block), RMB를 포함하여 다양한 주파수 특성을 학습합니다. 잔차 멀티스케일 블록(RMB)은 여러 수용 영역을 결합하여 다층 정보를 효과적으로 활용합니다. 또한, 동적 합성곱(dynamic convolution)을 기반으로 한 국소 융합 방법을 통해 인접한 주파수 대역에서 특징들을 결합하여 최종적으로 명확하고 자연스러운 이미지를 생성합니다.

- **Performance Highlights**: 실험 결과, 본 모델은 RESIDE, Haze4K, O-Haze 데이터셋에서 다른 최신 모델들과 비교했을 때 뛰어난 성능을 발휘하였습니다. 예를 들어, RESIDE-Indoor 데이터셋에서 PSNR이 43.21dB에 달하여 기존 CNN 기반 모델들에 비해 월등한 성능을 보였습니다.



### Exploiting Supervised Poison Vulnerability to Strengthen Self-Supervised Defens (https://arxiv.org/abs/2409.08509)
Comments:
          28 pages, 5 figures

- **What's New**: 본 논문에서는 안전성을 고려하여 Supervised Learning (SL) 알고리즘에 대한 새로운 방어 기법인 VESPR (Vulnerability Exploitation of Supervised Poisoning for Robust Self-Supervised Learning)를 제안합니다. 이를 통해 기존의 여러 방어 기법들이 가지는 한계를 극복하고, 여러 인기 있는 데이터 세트에 대해 상당한 성능 향상을 보여줍니다.

- **Technical Details**: VESPR는 악의적으로 변조된 데이터의 특성을 모호하게 만들어주는 Adversarial Training (AT)을 SL에 기반으로 통합한 방식입니다. 이 방법은 SSL (Self-Supervised Learning) 프레임워크 내에서 안정적인 이미지 특성을 학습하도록 설계되었습니다. 구체적으로, VESPR는 다중 작업 손실 함수로 인코더를 훈련시키며, 고전적인 Cross-Entropy 손실로 계산된 그래디언트를 사용하여 적대적 입력을 생성합니다.

- **Performance Highlights**: VESPR는 ImageNet-100과 CIFAR-10 데이터 세트에서 여섯 가지 이전의 방어 방법을 초과하는 성능을 보였습니다. 특히, VESPR는 오염된 모델의 최소 및 평균 ImageNet-100 시험 정확도를 각각 16% 및 9% 증가시켰습니다.



### Identifying Human Indoor Daily Life Behavior employing Thermal Sensor Arrays (TSAs) (https://arxiv.org/abs/2409.08508)
- **What's New**: 이번 연구는 열 감지 센서 배열(thermal sensor arrays, TSAs)을 이용하여 인간의 일상 생활 활동을 효과적으로 모니터링하고, 수면 활동을 구별하는 새로운 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 TSAs를 활용하여 인간의 일상 활동과 수면을 구별하는데 초점을 맞추며, 수집한 데이터를 바탕으로 활동의 시간대별 시계열(time series)과 공간 확률 분포(spatial probability distribution)를 구성합니다. 모니터링한 활동은 수면과 일상 활동의 두 가지 클래스로 분류되었습니다.

- **Performance Highlights**: 연구 결과, 낮과 밤의 구분 없이 두 클래스 간의 구별 가능성이 입증되었으며, 평균 수면 활동 지속시간은 9시간/일, 일상 생활 활동은 7시간/일로 확인되었습니다. TSAs는 기존의 시스템들이 직면했던 개인 정보 보호의 한계를 해결하며, 인간 활동 모니터링에 최적의 선택임을 나타냅니다.



### PSTNet: Enhanced Polyp Segmentation with Multi-scale Alignment and Frequency Domain Integration (https://arxiv.org/abs/2409.08501)
- **What's New**: 이번 연구는 폴립(segmentation of colorectal polyps) 분할을 위해 새로운 접근 방식인 Shunted Transformer를 가진 Polyp Segmentation Network (PSTNet)를 제안합니다. PSTNet은 RGB 및 주파수 영역(freqency domain) 정보를 결합하여 보다 정확한 분할을 가능하게 합니다.

- **Technical Details**: PSTNet은 세 가지 주요 모듈로 구성됩니다: (1) 주파수 특성 주의 모듈(Frequency Characterization Attention Module, FCAM) - 주파수 정보를 추출하고 폴립 특성을 캡처합니다. (2) 특성 보완 정렬 모듈(Feature Supplementary Alignment Module, FSAM) - 의미 정보를 정렬하고 잘못 정렬된 잡음을 줄입니다. (3) 교차 인식 위치 모듈(Cross Perception localization Module, CPM) - 주파수 정보와 고수준 의미를 결합하여 효율적인 폴립 분할을 달성합니다.

- **Performance Highlights**: 다양한 와 데이터셋에서 PSTNet은 기존의 최첨단 방법들보다 뛰어난 성능을 보여주었으며, 폴립 분할 정확도가 유의미하게 향상되었습니다. 실험 결과, PSTNet의 구조적 디자인과 주파수 영역 정보의 통합이 임상적 실제에 있어 폴립 분할의 정확성과 신뢰성을 개선할 수 있음을 입증했습니다.



### RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision (https://arxiv.org/abs/2409.08475)
- **What's New**: RT-DETRv3는 RT-DETR의 개선된 버전으로, 객체 탐지의 실시간 처리 성능을 획기적으로 향상시킵니다. 이 시스템은 CNN 기반의 보조 브랜치를 도입하여 더 밀집한 감독(supervision)을 제공하고, 자기 주의력 변동(self-attention perturbation) 전략을 통해 디코더 훈련의 효율성을 높였습니다.

- **Technical Details**: RT-DETRv3는 다중 보조 브랜치와 공유 가중치 디코더 브랜치를 사용하여 높은 품질의 쿼리와 실제 값 간의 매칭을 보장하며, 이 모든 모듈은 훈련 전용입니다. 구조적으로, CNN 백본과 효율적인 하이브리드 인코더를 통합하여 다중 스케일 특성을 얻고, 변환기(decoder) 브랜치에서 자가 주의력 모듈을 활용해 쿼리의 양자리(label assignment)를 다양화합니다.

- **Performance Highlights**: RT-DETRv3는 COCO val2017 데이터셋에서 기존의 실시간 탐지기들, 특히 RT-DETR 시리즈 및 YOLO 시리즈를 크게 능가합니다. 예를 들어, RT-DETRv3-R18은 48.1% AP를 달성하여 RT-DETR-R18에 비해 1.6% 상승했으며, 같은 대기 시간에서 일치를 유지합니다. RT-DETRv3-R101은 YOLOv10-X를 초과하여 54.6% AP를 기록했습니다.



### Generalization Boosted Adapter for Open-Vocabulary Segmentation (https://arxiv.org/abs/2409.08468)
- **What's New**: 본 논문에서는 Open-vocabulary segmentation(오픈 어휘 분할) 작업에서 VLMs(비전-언어 모델)의 일반화 및 강건성을 향상시키기 위한 새로운 어댑터 전략인 Generalization Boosted Adapter(GBA)를 제안합니다.

- **Technical Details**: GBA는 두 가지 핵심 구성 요소로 구성됩니다: (1) 스타일 다양화 어댑터(Style Diversification Adapter, SDA)는 기능을 진폭과 위상 성분으로 분리하여 스타일 정보를 향상시키고 의미적 일관성을 유지하며 (2) 상관 제약 어댑터(Correlation Constraint Adapter, CCA)는 교차 주의(cross-attention)를 사용하여 텍스트 카테고리와 이미지의 목표 영역 간의 더 긴밀한 의미적 연관성을 수립합니다.

- **Performance Highlights**: GBA는 얕은 SDA와 깊은 CCA의 상승 효과를 통해 오버피팅 문제를 효과적으로 완화하고, Open-vocabulary segmentation 작업에서의 성능을 크게 향상시킵니다. 이 방법은 간단하고 효율적인 모듈로서 다양한 CLIP 기반 방법에 유연하게 통합될 수 있으며, 여러 오픈 어휘 분할 벤치마크에서 최신 성능(state-of-the-art)을 달성합니다.



### VLTP: Vision-Language Guided Token Pruning for Task-Oriented Segmentation (https://arxiv.org/abs/2409.08464)
- **What's New**: 이 연구에서는 Vision Language Guided Token Pruning (VLTP)라는 새로운 토큰 가지치기 메커니즘을 도입하여 비전 트랜스포머(ViT) 기반의 세분화 모델을 가속화합니다. 특히, 이는 특정 입력 작업에 의존하는 작업 지향 세분화(Task-Oriented Segmentation, TOS)에 특히 유용합니다.

- **Technical Details**: VLTP는 MLLM(Multi-modal Large Language Model)의 가이드를 통해 이미지 토큰의 중요성을 평가하는 새로운 가지치기 디코더를 설계합니다. 주요 아이디어는 비전 트랜스포머의 모든 레이어를 거쳐서 처리해야 하는 토큰은 중요한 작업과 관련된 토큰만이라는 점입니다. VLTP는 데이터의 시각적 정보뿐 아니라 작업 기반의 추론을 고려하여 토큰 가지치기를 수행합니다.

- **Performance Highlights**: VLTP 프레임워크는 ViT의 계산 비용을 약 25% 줄이면서 성능 저하 없이, 1%의 성능 저하에서 약 40%까지 줄이는 것을 보여줍니다. MLLM에 의해 가이드된 VLTP 통합 세분화 모델은 최첨단(SOTA) 방법보다 +2.5% 향상된 mean Intersection over Union (mIoU) 성능을 기록합니다.



### VistaFormer: Scalable Vision Transformers for Satellite Image Time Series Segmentation (https://arxiv.org/abs/2409.08461)
- **What's New**: VistaFormer라는 경량화된 Transformer 기반의 모델 아키텍처를 소개합니다. 이 모델은 다중 스케일 Transformer 기반 인코더와 경량 디코더를 사용하여 글로벌 및 로컬 주의를 집계합니다.

- **Technical Details**: VistaFormer는 포지션-프리(self-attention)가 적용된 레이어를 사용하여 모델 아키텍처를 단순화하며, 학습 및 테스트 이미지 해상도가 다를 때 발생할 수 있는 성능 저하 요인을 제거합니다. Neighbourhood Attention(Na)으로 Multi-Head Self-Attention(MHSA)을 대체하여 성능 향상과 계산 효율성을 달성합니다. 이 모델은 PASTIS 및 MTLCC 벤치마크에서 실험을 통해 검증되었습니다.

- **Performance Highlights**: VistaFormer는 MHSA 사용 시 PASTIS 벤치마크에서 0.1%의 개선된 mIoU 점수를, MTLCC 벤치마크에서 3.7%의 개선된 성능을 기록하였습니다. 이 모델은 기존의 유사 모델들에 비해 8%의 더 적은 부동 소수점 연산을 요구합니다.



### Towards Unified Facial Action Unit Recognition Framework by Large Language Models (https://arxiv.org/abs/2409.08444)
- **What's New**: 이번 논문에서는 Facial Action Units (AUs)의 인식을 위한 첫 번째 통합 프레임워크인 AU-LLaVA를 제안합니다. 이 프레임워크는 크고 강력한 언어 모델(LLM)을 기반으로 하여 시각적 인코더(visual encoder), 선형 프로젝터(layer), 사전 훈련된 LLM으로 구성되어 있습니다.

- **Technical Details**: AU-LLaVA는 다양한 AU 데이터셋에서의 모델을 세밀하게 조정(fine-tuning)하며, 동일한 입력 이미지에 대해 다양한 AUs 인식 결과를 생성할 수 있도록 설계되었습니다. BP4D 및 DISFA 데이터셋에서 AU-LLaVA는 거의 절반의 AUs에 대해 가장 정확한 인식 결과를 전달하며, 특정 AU 인식에서는 이전 벤치마크에 비해 F1-score에서 최대 11.4%의 향상을 달성합니다.

- **Performance Highlights**: FEAFA 데이터셋에서 AU-LLaVA는 이전 벤치마크 결과에 비해 모든 24개 AUs에서 상당한 향상을 보여줍니다. AU-LLaVA는 AU 인식에서 뛰어난 성능과 다재다능성을 demonstrate합니다.



### CF-PRNet: Coarse-to-Fine Prototype Refining Network for Point Cloud Completion and Reconstruction (https://arxiv.org/abs/2409.08443)
Comments:
          Technical Report of the 1st place solution to CVPPA@ECCV2024: Shape Completion and Reconstruction of Sweet Peppers Challenge

- **What's New**: 본 논문에서 제안하는 CF-PRNet은 농업 환경에서 흔히 발생하는 부분적인 관측으로부터 과실의 정확한 3D 형상을 재구성하는 도전 과제를 다룹니다. 이 네트워크는 고해상도 3D 데이터를 학습 단계에서 활용하면서, 실시간 추론을 위해 단일 RGB-D 이미지만을 필요로 합니다.

- **Technical Details**: CF-PRNet은 포인트 클라우드(3D data points) 완성을 위한 coarse-to-fine 프로토타입 정제 네트워크입니다. 이 방법은 결함이 있는 포인트 클라우드 데이터에서 특징을 추출하고, 이를 바탕으로 두 개의 3D 메시 프로토타입(하나는 대략적, 다른 하나는 세부적)을 생성하여 점진적으로 세밀한 최종 포인트 클라우드를 완성합니다.

- **Performance Highlights**: CF-PRNet은 Chamfer Distance(3D 포인트 간의 거리 측정)의 경우 3.78, F1 Score 66.76%, Precision 56.56%, Recall 85.31%를 기록했으며, Sweet Peppers Challenge에서 1위를 차지하였습니다.



### 360PanT: Training-Free Text-Driven 360-Degree Panorama-to-Panorama Translation (https://arxiv.org/abs/2409.08397)
Comments:
          Accepted by WACV 2025, Project Page: \href{this https URL}{this https URL}

- **What's New**: 본 연구에서는 360도 파노라마 이미지를 위한 최초의 훈련 없는 텍스트 기반 이미지 전환 방법인 360PanT를 제안합니다. 이 방법은 경계 연속성 인코딩(boundary continuity encoding)과 공간 제어가 포함된 원활한 타일 번역(seamless tiling translation)의 두 가지 주요 요소로 구성됩니다.

- **Technical Details**: 360PanT는 노이즈가 있는 잠재 표현(noisy latent representation)에 입력 360도 파노라마의 경계 연속성 정보를 내장하며, 이 정보를 바탕으로 범위 제어된 원활한 타일 번역을 수행합니다. 경계 연속성 인코딩을 통해 입력 이미지의 구조와 의미 배치를 유지하며, 노이즈 특성에 따라 자른 패치(cropped patches)를 독립적으로 처리합니다. 이를 통해 360도 파노라마의 경계를 효과적으로 유지합니다.

- **Performance Highlights**: 실험 결과, 360PanT는 실제 및 합성 데이터셋 모두에서 360도 파노라마 번역에서 탁월한 성능을 보여주며, 전통적인 I2I 전환 방법의 한계를 뛰어넘는 경계 연속성을 달성했습니다.



### Continual Learning in 3D Point Clouds: Employing Spectral Techniques for Exemplar Selection (https://arxiv.org/abs/2409.08388)
- **What's New**: 이 논문에서는 3D 객체 분류(Continual Learning in 3D object classification)에서의 지속 학습을 위한 새로운 프레임워크인 CL3D를 소개합니다. 이 접근법은 스펙트럴 클러스터링(spectral clustering)을 사용하여 각 클래스에서 프로토타입(prototype)을 선택하는 방식으로 이루어져 있습니다.

- **Technical Details**: 본 연구는 입력 공간(3D 포인트), 지역(feature space) 특성 공간(1024차원 포인트), 및 전역(global feature space) 특성 공간 각각에서 클러스터링의 효과를 탐구합니다. 실험은 ModelNet40, ShapeNet, ScanNet 데이터셋에서 수행되었으며, 입력 공간 특성만으로도 최첨단 정확도를 달성했습니다.

- **Performance Highlights**: 입력, 지역, 전역 특성을 융합하여 ModelNet과 ShapeNet에서 최첨단 성능을 향상시켰으며, 경쟁 방법의 절반에 가까운 메모리를 사용하였습니다. ScanNet 데이터셋에서 저희 방법은 정확도를 4.1% 향상시키면서 경쟁자들이 사용하는 메모리의 단지 28%만 소모하는 성과를 보였습니다.



### Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations (https://arxiv.org/abs/2409.08381)
- **What's New**: 이 논문에서는 Vision-Language 모델(VLM)인 CLIP을 Multi-Label Recognition (MLR)에 대한 새로운 접근법으로 활용합니다. 특히, PositiveCoOp와 NegativeCoOp라는 두 가지 방법을 소개하여 양수 및 음수 프롬프트 학습의 효과를 분석합니다.

- **Technical Details**: PositiveCoOp에서는 클래스를 나타내는 양수 프롬프트 하나를 학습하고, 음수 프롬프트 대신 VLM의 텍스트 인코더에 의존하지 않고 이미지 특징과 연관된 임베딩 벡터를 직접 학습합니다. NegativeCoOp에서는 반대로 진행합니다. 이러한 접근 방식은 dual prompt 학습보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: PositiveCoOp 방법이 DualCoOp보다 뛰어난 성능을 보였으며, 전체 레이블 비율이 높을 때 baseline의 성능은 DualCoOp 및 PositiveCoOp에 필적할 정도로 강력했습니다. 이 baseline은 DualCoOp보다 약 16배 적은 훈련 매개변수와 절반의 훈련 시간만 필요로 합니다.



### SIG: A Synthetic Identity Generation Pipeline for Generating Evaluation Datasets for Face Recognition (https://arxiv.org/abs/2409.08345)
- **What's New**: 이 논문에서는 새로운 Synthetic Identity Generation Pipeline (SIG)을 소개하여 얼굴 인식 시스템의 평가를 위한 윤리적이고 균형 잡힌 데이터셋을 생성하는 방법을 제시합니다. 이를 통해 3,336개의 고유한 합성 정체성을 가진 ControlFace10k라는 공개 데이터셋도 출시하였습니다.

- **Technical Details**: SIG는 얼굴 인식 모델의 평가를 위한 훈련되지 않은 데이터셋을 생성하는 파이프라인으로, 사용자 동의 없이 인터넷에서 무작위로 수집된 데이터의 윤리적 문제를 해결합니다. 이 파이프라인은 조정 가능한 포즈(pose), 안면 특성(facial features), 인구 통계적 속성(demographic attributes)을 가진 합성 이미지를 생성합니다.

- **Performance Highlights**: ControlFace10k의 효과성은 최첨단 얼굴 인식 알고리즘을 사용하여 평가되었으며, 다양한 인구 통계 그룹 간의 알고리즘 편향(algorithmic bias)을 평가하기 위한 유용성을 강조합니다.



### Activation function optimization method: Learnable series linear units (LSLUs) (https://arxiv.org/abs/2409.08283)
- **What's New**: 이번 연구에서는 동적 활성화 함수(dynamic activation function)가 정적 활성화 함수(static activation function)보다 신경망의 비선형성(non-linearity)을 높이는 데 더 적합하다고 주장하며, 새로운 learnable activation function인 LSLU(Learnable Series Linear Units)를 제안합니다.

- **Technical Details**: LSLU는 조정 가능한 매개변수 {	heta}와 {	heta}를 도입하여 현재 레이어의 학습 단계에 맞게 활성화 함수를 조절할 수 있도록 설계되었습니다. 이 원리는 각 활성화 레이어에서 비선형성을 증가시켜 네트워크 전체의 비선형성을 향상시키는 것입니다.

- **Performance Highlights**: LSLU는 CIFAR10, CIFAR100 데이터셋과 특정 작업 데이터셋(예: Silkworm)에서 성능이 평가되었으며, VanillaNet 학습에서 CIFAR100의 정확도를 3.17% 개선했습니다.



### ClearDepth: Enhanced Stereo Perception of Transparent Objects for Robotic Manipulation (https://arxiv.org/abs/2409.08926)
Comments:
          7 pages, 7 figures

- **What's New**: 투명 객체의 깊이 인식 문제를 해결하기 위해 Vision Transformer 기반의 새로운 알고리즘을 개발했습니다. 이 알고리즘은 구조적 특징을 활용하여 정밀한 깊이 복원을 지원합니다.

- **Technical Details**: 우리는 투명 객체의 깊이 복원을 위해 AI 가속화된 시뮬레이션 데이터셋 생성 도구를 개발했습니다. 이 도구는 실제 센서에 직접 적용할 수 있으며, 우리의 스테레오 깊이 복원 네트워크는 MixVisionTransformer B5를 사용하여 구조적 정보를 효율적으로 추출합니다.

- **Performance Highlights**: 우리의 모델은 Sim2Real 일반화 능력이 뛰어나 실제 상황에서도 투명 객체의 정밀한 깊이 매핑을 가능하게 합니다. 이전의 CNN 및 ViT 기반 모델들보다 특히 높은 성능을 보였습니다.



### Gaussian is All You Need: A Unified Framework for Solving Inverse Problems via Diffusion Posterior Sampling (https://arxiv.org/abs/2409.08906)
- **What's New**: 본 논문에서는 Covariance Corrected Diffusion Posterior Sampling (CoDPS) 방법을 소개하며, 이 방법은 데이터 분포를 향상시키기 위한 공분산 조정 항을 통합한 통합 가능도( likelihood) 근사화 방법을 이용한다. 이를 통해 역 확산( reverse diffusion) 샘플링 과정에서 계산의 비효율성을 해결한다.

- **Technical Details**: 기존의 확산 모델 기반 방법들은 데이터 일관성(data consistency) 단계를 확산 채집 프로세스 내에 통합하며, 이 과정에서 대략적인 가능도 함수( likelihood function)에 의존한다. 그러나 본 연구에서 제안된 CoDPS 방법은 단순 가우시안(prior) 분포를 가정하여 조건부 분포의 뚜렷한 가우시안 형식을 유도하며, 이는 실제 자연 이미지 데이터셋에서 성능 개선을 가져온다.

- **Performance Highlights**: 실험 결과, CoDPS 방법은 여러 기존 방법들에 비해 추천 및 인식 품질을 높이며, 네트워크를 통한 경량 계산을 방지하기 때문에 시간 및 메모리 효율성이 뛰어난 것으로 나타났다. 가우시안 혼합 모델에 대해 빠르게 수렴한다는 점도 확인되었다.



### D2-MLP: Dynamic Decomposed MLP Mixer for Medical Image Segmentation (https://arxiv.org/abs/2409.08905)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 연구에서는 새로운 Dynamic Decomposed Mixer(DDM) 모듈을 제안하여 의료 영상 분할 작업에서의 성능 저하 문제를 해결하고자 합니다. 특히, 공간적 및 채널 간의 상호 의존성을 모델링하여 효과적으로 특성을 집합하며, 이를 통해 의료 영상에서 우수한 분할 성능을 달성하였습니다.

- **Technical Details**: DDM 모듈은 두 가지 Spatially Decomposed Mixer와 Channel Mixer를 포함하여 서로 다른 공간 위치와 채널에서 특성을 추출하고 정보를 집합합니다. DDM은 Spatial-wise Dynamic Mixing과 Channel-wise Dynamic Mixing 메커니즘을 사용하여 채널 및 공간 특성 간의 의존성을 모델링하고 이를 적응적으로 융합합니다. 새롭게 제안된 Dynamic Decomposed MLP Mixer(D2-MLP)는 ViT기반의 인코더-디코더 아키텍처에 DDM 모듈을 통합하여 구성됩니다.

- **Performance Highlights**: D2-MLP 모델은 Abdominal Multi-organ segmentation 및 Liver Tumor segmentation 두 가지 데이터셋에서 평가되었으며, 기존의 최첨단 방법들보다 뛰어난 분할 성능을 기록하였습니다.



### DX2CT: Diffusion Model for 3D CT Reconstruction from Bi or Mono-planar 2D X-ray(s) (https://arxiv.org/abs/2409.08850)
- **What's New**: 새로운 조건 부여 확산 모델인 DX2CT를 제안합니다. 이 모델은 이원 또는 단일 평면 X-선 이미지를 활용하여 고품질 3D CT(volume) 재구성을 수행합니다.

- **Technical Details**: DX2CT는 두 가지 핵심 구성 요소로 이루어져 있습니다: 1) 새로운 Transformer를 사용하여 2D X-ray에서 추출된 특징 맵을 3D CT 볼륨의 위치 정보로 조절합니다. 2) 조정된 3D 위치 인식 특징 맵을 DX2CT의 조건으로 효과적으로 사용합니다. 이를 통해 고품질 CT 재구성이 가능해집니다.

- **Performance Highlights**: LIDC(폐 이미지 데이터베이스 컨소시엄) 벤치마크 데이터셋을 사용한 실험에서, DX2CT는 기존의 여러 최첨단 방법들보다 우수한 성능을 보였습니다.



### On the Computation of BD-Rate over a Set of Videos for Fair Assessment of Performance of Learned Video Codecs (https://arxiv.org/abs/2409.08772)
Comments:
          Submitted to IEEE ICASSP 2025

- **What's New**: 최근 논문에서는 Bjøntegaard Delta (BD) 지표의 평균 계산 방법에 대한 문제를 지적하고 있습니다. 학습된 비디오 압축 커뮤니티에서 비디오 데이터 세트의 평균 RD 곡선에 의존하는 기존 방식이 잘못된 결론을 초래할 수 있음을 보여줍니다.

- **Technical Details**: 기존의 비디오 압축 방식에서는 개별 비디오에 대한 RD 곡선과 BD 지표를 계산한 후 BD 값을 평균합니다. 그러나 학습된 비디오 압축 방식에서는 이를 무시하고 모든 비디오의 RD 곡선을 평균하여 한 개의 BD 값을 도출하는 경향이 있습니다. 이러한 방식은 특정 비디오가 평균 RD 곡선을 왜곡할 수 있습니다.

- **Performance Highlights**: 실험 결과, 두 개의 최근 학습된 비디오 코덱 간의 비교는 BD 값을 계산하는 방법에 따라 결과가 다르게 나오는 것으로 나타났습니다. 따라서, 각 비디오에 대해 RD 곡선 및 BD 값을 개별적으로 계산하고 이를 평균하는 것이 더 신뢰할 수 있고 공정한 평가 방법임을 강조합니다.



### Layerwise Change of Knowledge in Neural Networks (https://arxiv.org/abs/2409.08712)
- **What's New**: 이 논문은 딥 뉴럴 네트워크(DNN)가 새로운 지식을 점진적으로 추출하고 잡음을 포함한 특징을 잊어버리는 과정을 층별로 설명합니다. 특히, 중간 층에서 암호화된 상호작용(interactions)을 추출하고 이를 정량화하는 새로운 접근방법을 제시합니다.

- **Technical Details**: 본 연구에서는 입력과 중간 층 특징 간의 상호 정보(mutual information)를 이용하여 DNN 각 층에서 암호화된 지식을 측정하며, 이는 학습 과정에서 새롭게 등장하고 잊혀진 상호작용을 추적하고 정량화하는 과정으로 이어집니다. 또한 인접한 층들은 유사한 상호작용을 암호화하는 경향이 있습니다.

- **Performance Highlights**: 우리는 새로운 상호작용 정의가 DNN의 일반화 능력(generalization capacity)과 어떻게 연결되는지를 밝혀내며, 저차원 상호작용이 고차원 상호작용보다 더 높은 안정성과 일반화 능력을 가짐을 발견했습니다.



### SkinFormer: Learning Statistical Texture Representation with Transformer for Skin Lesion Segmentation (https://arxiv.org/abs/2409.08652)
Comments:
          12 pages, 8 figures, published to JBHI

- **What's New**: 이 논문에서는 'SkinFormer'라는 새로운 전이 학습 네트워크를 제안하여 피부 병변(segmentation) 이미지를 효과적으로 처리할 수 있게 합니다. 특히, Kurtosis-guided Statistical Counting Operator를 활용하여 통계적 질감(statistical texture) 표현을 정량화하고, 다양한 질감 정보를 융합(fusion)하여 세분화의 정확도를 증가시키는 기술을 도입했습니다.

- **Technical Details**: SkinFormer 네트워크는 Kurtosis-guided Statistical Counting Operator를 사용하여 입력 이미지의 통계적 질감을 정량화합니다. 구조적 질감(structural texture) 정보와 통계적 질감 정보(statistical texture information)를 융합하기 위해 Statistical Texture Fusion Transformer를 설계하였으며, Multi-scale embedding enhancement와 Texture-enhanced FFN을 통해 피부 병변의 세분화 정확도를 높이는 Statistical Texture Enhance Transformer도 포함되어 있습니다.

- **Performance Highlights**: SkinFormer는 ISIC 2018 데이터셋에서 93.2%의 Dice score를 달성하며, 이전의 SOAT 방법들보다 우수한 성능을 보였습니다. 이 방법은 3D 이미지 세분화로 쉽게 확장할 수 있는 잠재력을 가지고 있습니다.



### Joint image reconstruction and segmentation of real-time cardiac MRI in free-breathing using a model based on disentangled representation learning (https://arxiv.org/abs/2409.08619)
Comments:
          Submitted to the Journal of Cardiovascular Magnetic Resonance

- **What's New**: 본 연구에서는 심장 cine MR 이미징을 위한 새로운 접근 방식이 제안되었습니다. 이는 disentangled representation learning을 기반으로 하는 이미지 재구성과 분할(segmentation)의 결합 방법입니다. 이 방법은 실시간으로 자유호흡 상태에서도 작동할 수 있도록 훈련되었습니다.

- **Technical Details**: 제안된 방법은 자체 개발된 spiral bSSFP pulse sequence를 기반으로 하여, 8명의 건강한 참가자와 5명의 간헐적 심방세동 환자를 대상으로 한 탐색적 실행 가능성 연구에서 테스트되었습니다. 이미지와 LV(segment) 분할 예측 결과는 ECG-gated segmented Cartesian cine와 비교되었습니다.

- **Performance Highlights**: 연구 결과, 건강한 참가자에서의 실시간 호흡 유지 방식과 Cartesian cine의 이미지 품질은 유사했으며, 자유호흡에서는 약간 떨어지는 성능을 보였습니다. 또한 심장 리듬 장애 환자에서도 두 가지 실시간 접근 방식 모두 긍정적인 이미지 품질을 나타냈습니다. 제안된 모델에서의 배출 분율(ejection fraction)은 임상적 기준과 비교했을 때 약간의 긍정적인 편향을 보였습니다. 제시된 실시간 MR 이미징 기법은 ECG 게이팅과 호흡 유지가 필요 없이 1-2분 내에 고품질 심장 cine 데이터를 수집할 수 있는 가능성을 보여주었습니다.



### TapToTab : Video-Based Guitar Tabs Generation using AI and Audio Analysis (https://arxiv.org/abs/2409.08618)
- **What's New**: 이 논문은 비디오 입력에서 기타 타블레쳐(guitar tablature) 생성을 자동화하는 새로운 접근 방식을 제안합니다. 이를 통해 음악 교육, 전사( transcription) 정확도, 그리고 성능 분석을 향상시키는 것이 목표입니다.

- **Technical Details**: 딥 러닝(deep learning)을 활용한 이 연구에서는 YOLO 모델을 사용하여 실시간으로 프렛보드(fretboard)를 감지하고, 푸리에 변환(Fourier Transform) 기반의 오디오 분석을 통해 정확한 노트 식별을 수행합니다.

- **Performance Highlights**: 이 실험 결과는 전통적인 기술들에 비해 감지 정확도와 견고성에서 상당한 개선을 보여주었습니다. 이 논문은 이러한 방법론의 개발, 구현, 평가 과정을 설명하며, 비디오 녹화에서 기타 타블레처 생성을 자동화하여 기타 교육을 혁신하고자 합니다.



### Improved Unet model for brain tumor image segmentation based on ASPP-coordinate attention mechanism (https://arxiv.org/abs/2409.08588)
Comments:
          5 pages, 8 figures, accepted by ICBASE 2024

- **What's New**: 본 논문에서는 뇌종양 이미지 분할을 위한 개선된 Unet 모델을 제안합니다. 이 모델은 coordinate attention mechanism과 ASPP 모듈을 결합하여 분할 효과를 향상시킵니다.

- **Technical Details**: 데이터 세트를 분할한 후, 이미지를 사전 처리(preprocessing)하고 개선된 모델을 통해 실험을 수행합니다. 전통적인 Unet 모델을 학습(train)하고 검증(validate)한 결과, 첫 번째 epoch에서 손실(loss) 값이 지속적으로 감소하고, 여덟 번째 epoch에서 안정적으로 유지됨을 확인했습니다. 또한 miou(mean Intersection over Union) 지표는 15번째 epoch에서 0.6을 초과하고 46번째 epoch에서 0.7 이상에 도달했습니다. 개선된 모델은 손실 곡선에서 6번째 epoch에서 최저치를 기록하고, 20번째 epoch부터 miou 지표가 0.7 이상으로 안정화되어 최대 0.76에 도달했습니다.

- **Performance Highlights**: 개선된 Unet 모델은 전통적인 Unet 모델에 비해 분할(segmentation) 및 엣지(edge) 정확도가 우수하여 의학 이미지 분석(medical image analysis)에 대한 보다 신뢰할 수 있는 방법을 제공합니다.



### Second-order difference subspac (https://arxiv.org/abs/2409.08563)
Comments:
          18 pages, 11 figures

- **What's New**: 이 논문에서는 첫 번째 차수의 차이 서브스페이스(first-order difference subspace)를 확장하여 두 서브스페이스 간의 기하학적 차이를 분석할 수 있는 두 번째 차수의 차이 서브스페이스(second-order difference subspace)를 제안합니다. 이 개념은 다양한 기계 학습 응용 프로그램에서 서브스페이스 표현의 이해를 향상시키기 위해 도입되었습니다.

- **Technical Details**: 두 서브스페이스 간의 기하학적 관계를 분석하기 위해 첫 번째 차수 차이 서브스페이스의 정의를 서로 다른 차원을 가지면서 교차하는 두 서브스페이스에 대한 더 일반적인 설정으로 확장합니다. 이후, 두 서브스페이스 간의 주성분 서브스페이스(principal component subspace)를 결합하여 두 번째 차수의 서브스페이스를 정의합니다. 이 과정은 Grassmann 다양체(Grassmann manifold)에서의 기하학적 관점에서 서브스페이스 동역학의 속도 및 가속도를 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 두 가지 응용 예제에서 제안한 두 번째 차수의 차이 서브스페이스의 유효성과 자연성을 입증합니다. 첫 번째 실험에서는 3D 객체의 동적 형태 분석을 통해 서브스페이스의 동역학 속도 및 가속도를 추출하며, 두 번째 실험에서는 생체 신호의 시계열 분석을 통해 긴 신호에서 고차원 서브스페이스가 추출됩니다. 이 결과는 두 차수의 차이 서브스페이스가 기대하는 행동을 일관되게 보여줍니다.



### SRE-CNN: A Spatiotemporal Rotation-Equivariant CNN for Cardiac Cine MR Imaging (https://arxiv.org/abs/2409.08537)
Comments:
          Accepted at MICCAI 2024

- **What's New**: 본 연구에서는 Spatiotemporal Rotation-Equivariant CNN (SRE-CNN)이라는 새로운 프레임워크를 제안하여, 동적 MRI 영상에서 고유의 회전 대칭을 최대한 활용하고자 하였습니다. 이 프레임워크는 공간 및 시간 차원의 회전 대칭을 동시에 활용하는 방법론을 포함하고 있습니다.

- **Technical Details**: SRE-CNN은 2D+t 동적 자료의 공간 및 시간 차원에서의 회전 대칭성을 활용하는 템포럴-에퀴바리언트(convolutional) 모듈을 설계하였습니다. 또한, 1D 및 2D 푸리에 급수 확장을 기반으로 한 고정밀 필터 매개변수화 전략을 활용하여 필터의 표현 정확도를 개선하였습니다.

- **Performance Highlights**: 제안된 방법은 29명의 피험자를 대상으로 한 심장 CINE MRI 데이터를 사용하여 훈련 및 테스트를 진행하였고, 기존 최첨단(reconstruction) 방법들과 비교했을 때 정량적 및 정성적으로 우수한 성능을 보였습니다. 특히, 고속 촬영 환경에서 더욱 효과적인 결과를 나타내었습니다.



### Cross-conditioned Diffusion Model for Medical Image to Image Translation (https://arxiv.org/abs/2409.08500)
Comments:
          miccai24

- **What's New**: 이번 논문에서는 의료 영상의 이미지 간 변환을 위한 새로운 방법인 Cross-conditioned Diffusion Model (CDM)을 제안합니다. 이 모델은 기존의 분산 모델에 비해 생성 효율성을 높이고, 타겟 모달리티의 분포를 가이드로 사용하여 합성 품질을 개선합니다.

- **Technical Details**: CDM은 세 가지 주요 구성 요소로 이루어져 있습니다: 1) Modality-specific Representation Model (MRM) - 타겟 모달리티의 분포를 학습; 2) Modality-decoupled Diffusion Network (MDN) - 특징 표현과 효율성을 개선 위해 설계됨; 3) Cross-conditioned UNet (C-UNet) - 소스 모달리티와 MDN으로부터 샘플링된 분포를 입력으로 받아 타겟 모달리티를 생성합니다. MRM은 각 타겟 모달리티의 패치를 랜덤으로 마스킹하고 이를 복원하도록 훈련됩니다.

- **Performance Highlights**: BraTS2023 및 UPenn-GBM 데이터셋을 기반으로 실시한 광범위한 실험 결과, CDM 방법이 기존의 모델들보다 뛰어난 성능을 보임을 입증하였습니다.



### WheelPoser: Sparse-IMU Based Body Pose Estimation for Wheelchair Users (https://arxiv.org/abs/2409.08494)
Comments:
          Accepted by ASSETS 2024

- **What's New**: 새로운 연구인 WheelPoser는 휠체어 사용자를 위한 실시간 포즈 추정 시스템으로, 기존의 카메라 및 밀집 IMU 배열을 사용하는 시스템보다 사용이 더 간편합니다. 이 시스템은 4개의 전략적으로 배치된 IMU(Inertial Measurement Unit)만 필요하며, 휠체어 이용자의 포즈를 훨씬 더 효율적으로 추적합니다.

- **Technical Details**: WheelPoser는 휠체어 사용자의 상체 포즈를 실시간으로 추정하기 위해 4개의 IMU를 사용하여, 최적화된 장비 설치를 통해 최소한의 간섭으로도 정확한 포즈 추정을 가능하게 합니다. 휠체어 사용자 전용 모션을 포함한 WheelPoser-IMU 데이터셋을 활용하여 모델을 학습하였습니다. 소프트웨어는 Sparsity 기반의 운동학 모듈을 사용하여 포즈 데이터를 해석합니다.

- **Performance Highlights**: WheelPoser는 평균 관절 각도 오차 14.30도와 평균 관절 위치 오차 6.74cm를 기록하며, 기존 모델에 비해 세 배 이상 개선된 성능을 보입니다. 이로 인해 WheelPoser는 휠체어 사용자에게 있어 상체 포즈 추정의 새로운 표준을 설정했습니다.



### Tri-Plane Mamba: Efficiently Adapting Segment Anything Model for 3D Medical Images (https://arxiv.org/abs/2409.08492)
- **What's New**: 이 연구에서는 3D 의료 이미지 분할을 위한 새로운 아키텍처인 Tri-Plane Mamba (TP-Mamba) 어댑터를 소개합니다. 이 어댑터는 Segment Anything Model (SAM)과 결합되어 3D 이미지에서의 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: TP-Mamba 어댑터는 두 가지 주요 혁신으로 구성됩니다: 1) 다중 스케일 3D 합성곱 어댑터(multi-scale 3D convolutional adapters), 이를 통해 국소 깊이 정보를 효율적으로 처리하고, 2) 삼중 평면 스캔 모듈(tri-plane scan module), 이는 긴 거리의 깊이 표현을 캡처하는 데 최적화되어 있습니다. 이 아키텍처는 전통적인 3D 합성곱 네트워크의 한계를 극복합니다.

- **Performance Highlights**: 이 어댑터는 BTCV 데이터 세트에서 무료로 제공된 3개의 CT 훈련 샘플만 가지고도 기존 3D 분할 네트워크보다 최대 12% 높은 Dice 점수를 달성하여 선진적인 성능을 보였습니다.



### Risks When Sharing LoRA Fine-Tuned Diffusion Model Weights (https://arxiv.org/abs/2409.08482)
- **What's New**: 본 논문은 Fine-tuning(파인튜닝) 과정에서 개인 데이터가 유출될 수 있는 위험성을 탐구합니다. 기존의 저널에서는 adversaries(적대자)가 훈련 텍스트를 알고 있을 경우를 가정했지만, 본 연구에서는 모델 가중치만으로도 개인 이미지를 복원할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 Variational Network Autoencoder(변분 네트워크 오토인코더)를 설계하여 모델 가중치를 입력으로 받고 개인 이미지 복원의 출력을 내보냅니다. 이 오토인코더의 효율성을 높이기 위해 timestep embedding(타임스텝 임베딩)을 활용한 훈련 패러다임을 제안합니다. 실험적으로 기존의 defense methods(방어 방법)은 개인 데이터의 프라이버시를 보장할 수 없음을 발견했습니다.

- **Performance Highlights**: 이 연구의 주요 결과는 adversary가 fine-tuned(파인튜닝된) 모델의 가중치만으로도 개인 이미지를 생성할 수 있다는 것입니다. 특히, 기존의 differential privacy(차등 프라이버시) 기반 방법은 개인 데이터의 프라이버시를 보호하지 못하며, 모델의 유용성을 저해합니다.



### USTC-TD: A Test Dataset and Benchmark for Image and Video Coding in 2020s (https://arxiv.org/abs/2409.08481)
Comments:
          24 pages. Project Page: this https URL

- **What's New**: 본 논문에서는 USTC-TD라는 새로운 이미지 및 비디오 코딩 테스트 데이터셋을 제안합니다. 이 데이터셋은 다양한 환경 요소와 이미지/비디오 품질을 반영하여, 이미지 및 비디오 압축 기법의 평가와 혁신을 돕기 위해 설계되었습니다.

- **Technical Details**: USTC-TD는 4K 해상도의 이미지 40장과 1080p 해상도의 비디오 10개 시퀀스를 포함하고 있습니다. 각 비디오는 30fps(초당 프레임수)로 캡처된 96 프레임으로 구성되어 있으며, RGB 및 YUV 색 공간을 사용합니다. 이 데이터셋은 다양한 콘텐츠 요소를 고려하여 구성되었으며, PSNR 및 MS-SSIM 지표를 통해 평가되었습니다.

- **Performance Highlights**: USTC-TD 데이터셋의 성능은 기존 이미지/비디오 테스트 데이터셋과 비교할 때 더욱 폭넓은 범위와 다양성을 보여주었으며, 최신 압축 기법에 대한 기준선을 수립하고 다양한 평가 지표에서 성능을 분석하여 향후 연구 방향을 제시하고 있습니다.



### Rethinking Meta-Learning from a Learning Lens (https://arxiv.org/abs/2409.08474)
- **What's New**: 이 논문에서는 메타 학습(meta-learning)의 "학습을 학습하는" 전략을 재조명하여 환경을 변화시키지 않고도 오류를 제거할 수 있는 방법을 탐구합니다.

- **Technical Details**: 메타 학습 프로세스의 알고리즘적 절차를 재고하고, 이를 통해 (i) 과적합(overfitting) 및 부족적합(underfitting)의 위험을 촉발하고 (ii) 서로 다른 작업에 적응한 모델들이 서로를 촉진한다는 사실을 발견했습니다. 이 연구는 Task Relation Learner (TRLearner)라고 하는 플러그 앤 플레이(plug-and-play) 방법을 제안하며, 이 방법은 작업 관계(task relations)를 사용하여 메타 학습의 최적화 과정을 보정합니다.

- **Performance Highlights**: TRLearner는 작업별 메타 데이터에서 추출한 작업 관계 행렬(task relation matrices)을 얻고, 관계 인식 일관성 정규화(relation-aware consistency regularization)를 사용하여 최적화를 안내합니다. 광범위한 이론적 및 실증적 분석을 통해 TRLearner의 효과를 입증하였습니다.



### Learned Compression for Images and Point Clouds (https://arxiv.org/abs/2409.08376)
Comments:
          65 pages, 21 figures, Master's Thesis, defended in 2023

- **What's New**: 최근 10년간 딥러닝(deep learning)은 컴퓨터 비전(computer vision) 작업에서 큰 성공을 거두어 왔습니다. 본 논문에서는 데이터 압축(data compression)에 딥러닝을 적용하여 차세대 멀티미디어 코덱(codec)을 개발하기 위한 세 가지 주요 기여를 제안합니다.

- **Technical Details**: 첫째, 특정 입력에 맞게 인코딩 분포(encoding distribution)를 동적으로 조정하는 저복잡도(entropy model) 엔트로피 모델을 제안합니다. 둘째, 분류(classification)에 특화된 경량(low-complexity) 포인트 클라우드 코덱을 개발하였으며, 이를 통해 전문화되지 않은 코덱에 비해 비트레이트(bit rate)를 크게 줄일 수 있습니다. 마지막으로, 연속하는 비디오 프레임 간 입력 도메인에서의 모션(motion)이 어떻게 컨볼루션(convolutionally) 기반의 잠재 공간(latent space)에서 나타나는지를 탐구합니다.

- **Performance Highlights**: 제안된 방법은 이미지 압축에서의 기존 코덱들과 비교했을 때 경쟁력 있는 압축 성능을 보인다는 점에서 흥미롭습니다. 학습 기반 압축의 결과는 전통적인 코덱들과 비교했을 때 긍정적인 성과를 기록하였습니다.



### Robust Dual Gaussian Splatting for Immersive Human-centric Volumetric Videos (https://arxiv.org/abs/2409.08353)
Comments:
          Accepted at SIGGRAPH Asia 2024. Project page: this https URL

- **What's New**: 이번 연구에서는 DualGS라는 새로운 Gaussian 기반 접근 방식을 소개하며, 이를 통해 복잡한 인간의 행동을 실시간으로 고충실도로 재생할 수 있는 기술을 개발하였습니다. DualGS는 동작과 외관을 별도로 표현하여, 압축 비율을 뛰어나게 향상시키고 있습니다.

- **Technical Details**: DualGS는 피부(Skin)와 관절(Joint) Gaussian을 사용하여 모션과 외관의 독립적인 표현(disentanglement)을 가능하게 합니다. 초기화 과정에서 관절 Gaussian을 랜덤하게 설정하고, 첫 프레임에서 관절 Gaussian에 피부 Gaussian을 고정시켜 모션 예측을 통해 성능을 모델링합니다. 이를 통해 약 350KB의 저장 공간으로 프레임 당 대략 120배의 압축 비율을 달성합니다.

- **Performance Highlights**: DualGS는 VR 환경에서 사용자에게 몰입감 있는 경험을 제공하며, 사진처럼 사실적인 실시간 연주 장면을 구현합니다. 또한, 복잡한 모션을 정밀하게 추적하면서도 높은 재현 품질을 유지하고, 여러 4D 자산을 통합하여 사용자에게 매력적인 음악적 여행을 제공합니다.



### Bayesian Inverse Graphics for Few-Shot Concept Learning (https://arxiv.org/abs/2409.08351)
- **What's New**: 이번 연구에서는 사람의 적은 예시만으로 새로운 개념을 학습하는 능력을 모방하는 Bayesian 비전 모델을 제안합니다. 이 모델은 최소한의 데이터만을 사용하여 프로토타입 확률 프로그램을 통해 오브젝트를 학습합니다.

- **Technical Details**: 제안된 모델은 generative inverse graphics 모델을 통해 기본 도형의 후방 분포를 추론하며, Markov Chain Monte Carlo (MCMC)를 사용하여 물리적으로 일관된 매개변수 분포를 샘플링합니다. 또한, 새로운 differentiable renderer를 통해 글로벌 장면 매개변수를 최적화합니다.

- **Performance Highlights**: 제안된 모델은 기존의 few-shot neural-only 분류 알고리즘보다 성능이 우수하며, 다양한 조명 조건 및 배경에서의 일반화 능력을 보여줍니다. 특히, 기존 모델에 비해 적은 수의 학습 이미지로도 높은 정확도를 달성합니다.



### Digital Volumetric Biopsy Cores Improve Gleason Grading of Prostate Cancer Using Deep Learning (https://arxiv.org/abs/2409.08331)
- **What's New**: 2023년에 미국 남성들 사이에서 가장 많이 진단된 암인 전립선암(PCa)의 진단을 위해 새로운 디지털 병리학 데이터 소스인 'volumetric core'를 제안합니다. 이 방법은 전통적인 2D 조직 슬라이스에서 추출한 3D 볼륨 구조를 생성하며, 이는 복잡한 조직 구조 분석을 개선합니다.

- **Technical Details**: 이 연구에서는 morphology-preserving alignment framework를 사용하여 10,210개의 volumetric cores를 구축하고, attention-based multiple-instance learning (ABMIL) 프레임워크를 통해 Gleason Grade Group (GGG)을 자동 분류합니다. volume patch를 처리하기 위해 self-supervised learning으로 사전 훈련된 deep feature extractor을 사용하는 수정된 비디오 변환기를 이용했습니다.

- **Performance Highlights**: 모델은 0.958의 macro-average AUC, 0.671의 F1 점수, 0.661의 정밀도 및 0.695의 재현율을 기록하여 2D 베이스라인을 크게 능가했습니다.



### MedSegMamba: 3D CNN-Mamba Hybrid Architecture for Brain Segmentation (https://arxiv.org/abs/2409.08307)
Comments:
          14 pages, 8 figures

- **What's New**: 본 논문에서는 3D patch-based hybrid CNN-Mamba 모델을 개발하여 subcortical brain segmentation의 정확성과 효율성을 향상시키는 방법을 제안합니다. 전통적인 pipeline 대비 빠르고 효과적인 접근 방식으로, 다양한 해부학적 클래스를 처리할 수 있는 능력을 제공합니다.

- **Technical Details**: 제안된 MedSegMamba 구조는 Mamba의 selective scan 알고리즘을 활용하여 1784개의 T1-weighted MRI 스캔을 처리합니다. 모델은 Dice Similarity Coefficient (DSC), Volume Similarity (VS), Average Symmetric Surface Distance (ASSD)와 같은 평가 메트릭을 사용하여 성능을 검증했습니다.

- **Performance Highlights**: 모델은 모든 평가 메트릭에서 가장 높은 성능을 달성했습니다 (DSC 0.88383; VS 0.97076; ASSD 0.33604). 또한, 기존 Mamba 기반 모델에 비해 ASSD에서 유의미한 개선을 보여주었으며, 약 20% 적은 파라미터를 유지하면서 효율적인 3D segmentation을 달성했습니다.



### Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation (https://arxiv.org/abs/2409.08301)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 Gaussian Differentially Private (GDP) 3D 인간 얼굴의 공개 문제를 다룬다. 복잡한 구조를 가진 인간 얼굴은 정체성과 밀접하게 연관되어 있으며, 이러한 데이터를 공식적으로 개인정보 보호를 위한 방법으로 보호하는 것이 어려운 문제이다. 우리는 GDP 프레임워크를 위한 기능적 데이터의 근사 DP 기법을 확장하였으며, 얼굴의 새로운 표현 방식인 face radial curves를 제안하였다.

- **Technical Details**: face radial curves는 얼굴의 복잡한 3D 구조를 함수들의 집합으로 표현하며, 이 표현은 구조를 손상시키지 않으면서 노이즈를 주입할 수 있는 방법이다. 우리의 GDP 기능적 데이터 메커니즘은 두 가지 주요 구성 요소로 이루어져 있으며, 첫 번째 구성 요소는 통계적 함수 값 요약에 적용 가능하고, 두 번째는 디스크 형태의 표면에 일반적으로 적용 가능하다.

- **Performance Highlights**: 우리의 방법은 평균 얼굴의 형태를 보존하면서 전통적인 방법보다 동일한 개인정보 보호 예산을 가지고 더 적은 노이즈를 주입한다. 이는 인체 얼굴을 넘어 디스크 형태의 표면에도 적용할 수 있다.



### Estimating Atmospheric Variables from Digital Typhoon Satellite Images via Conditional Denoising Diffusion Models (https://arxiv.org/abs/2409.07961)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 연구에서는 태풍 분야에서 조건부 노이즈 제거 확산 확률 모델(Conditional Denoising Diffusion Probability Model, CDDPM)을 적용하여 디지털 태풍(Digital Typhoon) 위성 이미지를 기반으로 여러 ERA5 기상 변수를 동시에 예측하는 방법을 탐구하였습니다.

- **Technical Details**: 연구에서는 ERA5 재분석 데이터와 디지털 태풍 이미지를 연계하여 새로운 태풍 데이터셋 구축과 기상 예측 정확도를 높이는 방식으로 진행되었습니다. CDDPM은 CNN(Convolutional Neural Networks) 및 SENet(Squeeze-and-Excitation Networks)와 비교하여 가장 높은 PSNR(32.807)과 RMSE(0.032)를 기록하며 성능이 우수한 것으로 나타났습니다.

- **Performance Highlights**: CDDPM은 CNN에 비해 약 7.9% 높은 PSNR과 11.1% 낮은 RMSE를 기록하며, 태풍 관련 기상 데이터를 보다 정확하게 생성할 수 있는 가능성을 보여주었습니다. 이 연구 결과는 기후 변화의 영향을 받는 지역에서의 강력한 기상 예측을 통해 심각한 기상 재해의 영향을 줄이는 데 도움을 줄 것으로 기대됩니다.



New uploads on arXiv(cs.AI)

### AI-LieDar: Examine the Trade-off Between Utility and Truthfulness in LLM Agents (https://arxiv.org/abs/2409.09013)
- **What's New**: 새로운 연구 AI-LieDar 프레임워크는 LLM 기반 에이전트들이 유용성과 진실성 간의 충돌 상황을 어떻게 탐색하는지를 다룬다. 이 연구는 진실성과 유용성의 상충을 탐구하며, 더욱 상호작용적인 멀티 턴 설정에서 LLM의 진실성을 평가한다.

- **Technical Details**: 연구는 60개의 다양한 시나리오를 통해 LLM의 행동을 분석하며, 진실성 감지기를 개발하여 모델의 응답을 평가했다. 시뮬레이션 파라미터는 Sotopia 프레임워크를 이용했으며, 진실성을 측정하기 위해 세분화된 평가 도구를 도입하였다. 2160회의 시뮬레이션 결과, 모든 모델은 50% 미만의 진실성을 보여주었다.

- **Performance Highlights**: 연구 결과, 모델들은 강한 유용성 목표가 있는 경우 속이기 행동을 보일 수 있으며, 특정 시나리오에서 모델들이 진실성을 따를 때 목표 달성률이 15% 감소하였다. 더욱이, 강력한 모델들은 속이거나 진실하게 대답하도록 유도될 때 40%의 증가율을 보였다.



### SynSUM -- Synthetic Benchmark with Structured and Unstructured Medical Records (https://arxiv.org/abs/2409.08936)
- **What's New**: SynSUM 벤치마크는 비구조적 임상 노트를 구조적 배경 변수와 연결하는 합성 데이터셋을 소개합니다. 이 데이터셋은 호흡기 질환 영역의 虚構적인 환자 접촉을 설명하는 노트가 포함된 10,000개의 인공 환자 기록으로 구성됩니다.

- **Technical Details**: 이 데이터셋의 테이블 부분은 전문가 기반의 Bayesian network를 통해 생성되며, 질병 이력 및 현재 방문에 관한 부분적으로 인코딩된 정보를 포함합니다. 대형 언어 모델인 GPT-4o를 사용하여 환자 증상 및 추가 정보를 설명하는 임상 노트를 생성합니다.

- **Performance Highlights**: 이 데이터셋은 임상 정보 추출(CIE)과 데이터 자동화, 인과 효과 추정 및 다중 모달 합성 데이터 생성 연구에 유용한 기초를 제공합니다. SynSUM은 구조적 및 비구조적 데이터의 혼합을 통해 기존 시스템의 한계를 극복하려는 목표를 가지고 있습니다.



### Yes, Prime Minister, question order does matter -- and it's certainly not classical! But is it quantum? (https://arxiv.org/abs/2409.08930)
Comments:
          12 pages, 1 figure

- **What's New**: 이 논문은 전통적인 확률 이론으로는 설명할 수 없는 여론 조사 결과의 조작 가능성을 보여줍니다. 특히, 양자 확률 이론이 이러한 현상을 설명할 수 있는 가능성을 제시하며, Ipsos에 의해 수행된 최근의 여론 조사를 사례로 들고 있습니다.

- **Technical Details**: Ipsos는 영국의 성인 2,158명을 대상으로 두 가지 유형의 질문을 제시했습니다. 첫 번째 샘플(A)은 국가 서비스 도입에 긍정적인 질문을 반영하고, 두 번째 샘플(B)은 부정적인 질문을 반영합니다. 결과적으로, 동일한 질문에도 불구하고 응답은 서로 다른 확률 분포를 보여 전통적인 확률 법칙과의 모순이 발생합니다. 이는 양자 확률에서 요구하는 상태의 중첩(superposition) 개념으로 설명할 수 있으나, 일반적인 양자 인지 모델로는 간단히 설명할 수 없는 복잡성을 가지고 있습니다.

- **Performance Highlights**: 여론 조사 결과에 따르면, 국가 서비스 도입에 대한 찬성 비율이 샘플 A에서는 45%, 샘플 B에서는 34%로 나타났으며, 이는 고전 확률 이론의 조건부 확률 법칙을 위반하는 결과를 초래하였습니다. 그러나 양자 확률 이론 내에서도 이러한 결과를 단순히 설명할 수는 없으며, 추가적인 실험 데이터가 필요함을 논문이 강조하고 있습니다.



### Affective Computing Has Changed: The Foundation Model Disruption (https://arxiv.org/abs/2409.08907)
- **What's New**: 본 연구는 Foundation Models(FMs)의 발전이 Affective Computing(정서 컴퓨팅) 분야에 미치는 영향을 분석하고, 멀티모달 감정 데이터를 생성하고 분석하는 데 중점을 두고 있습니다. 특히, 본 논문은 FMs가 감정 인식, 감정 콘텐츠 생성 및 감정에 대한 반응을 포함한 다양한 연구 문제에 어떻게 기여할 수 있는지를 조명합니다.

- **Technical Details**: 연구는 FMs의 학습 특성에 주목하며, 이 모델들이 다양한 데이터에 대해 트레이닝되어 폭넓은 문제를 해결할 수 있다는 점을 강조합니다. FMs는 대량의 학습 파라미터를 이용해, 특정 훈련을 받지 않은 작업에서도 경쟁력 있는 성능을 발휘할 수 있습니다. 또한 새로운 아키텍처인 Diffuser와 Transformer, 자가 지도 학습 전략, 그리고 상호 모달 정렬 기법의 발전이 FMs의 성능 향상을 견인하고 있습니다.

- **Performance Highlights**: FMs는 정서적 콘텐츠와 관련된 작업에서 놀라운 성능을 보여주며, 현실적인 데이터 샘플을 생성하거나 제로샷 분류를 수행할 수 있는 능력을 갖추고 있습니다. 이와 같은 발전은 FMs의 정서적 능력의 범위와 잠재력을 여전히 불확실하게 하면서도 Affective Computing 커뮤니티에 상당한 영향을 미치고 있습니다.



### Exploring Action-Centric Representations Through the Lens of Rate-Distortion Theory (https://arxiv.org/abs/2409.08892)
- **What's New**: 본 연구는 행동 중심(action-centric) 표현을 효율적 데이터 압축의 한 형태로서 탐구합니다. rate-distortion theory를 통해 인지 시스템의 목적지향적인 측면을 조명하고, VAEs(Variational Autoencoders)를 사용하여 작동하는 모델을 제시합니다.

- **Technical Details**: 효율적 코딩 가설(efficient coding hypothesis)은 신경세포가 감각적인 정보를 최대화하기 위해 최적화되어 있으며, rate-distortion theory는 정보의 손실 압축 간의 최적 거래를 정의합니다. 본 연구에서는 VAEs를 사용하여 action-centric 표현이 최적의 손실 압축(a lossy compression)을 형성한다고 주장합니다.

- **Performance Highlights**: 행동 중심 표현은 데이터의 최적 손실 압축 형태이며, 다운스트림 작업에서 성공적으로 사용될 수 있습니다. 데이터의 완전 복원이 필요하지 않다는 점에서, 이러한 표현은 성공적인 행동을 위한 적절한 정보 전달을 가능하게 합니다.



### Using The Concept Hierarchy for Household Action Recognition (https://arxiv.org/abs/2409.08853)
Comments:
          5 pages, 5 figures

- **What's New**: 이번 논문에서는 환경의 정적 및 동적 구성 요소(즉, 객체 및 에이전트)와 환경 내에서 발생하는 변화(즉, 에이전트가 수행하는 행동 및 기술)를 체계적으로 표현하는 방법인 Concept Hierarchy를 제안합니다. 이 접근 방식은 자율 시스템이 환경 상태를 표현하고, 행동 모델링 및 인식, 작업 실행 계획을 수립하는 데 필요한 정보를 제공합니다.

- **Technical Details**: Concept Hierarchy (CH)는 지식 모델링 프레임워크로, 특정 애플리케이션에서 필요한 정보를 표현합니다. CH는 작업(Task), 행동(Action), 기술(Skill), 그리고 에이전트(Agent)와 객체(Object)의 속성으로 구성됩니다. 이 프레임워크는 C++로 구현되었으며, 인스턴스의 동적 유형 변경, 데이터 업데이트 및 기존 지식의 프로그래밍적 구조 수정이 가능합니다.

- **Performance Highlights**: 본 연구의 주요 하이라이트는 Task 플래너가 필요한 변경 사항을 수집, 우선 순위 및 최적화하여 에이전트에 최종 실행 계획을 분배하는 능력입니다. 이를 통해 로봇이 환경을 인식하고 변화에 적절하게 반응할 수 있도록 지원하며, 특히 인간과의 상호작용 및 환경 변화 이해를 촉진할 수 있습니다.



### A RAG Approach for Generating Competency Questions in Ontology Engineering (https://arxiv.org/abs/2409.08820)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 활용하여 과학 논문을 기반으로 역조회 증강 생성(retrieval-augmented generation, RAG) 방법론을 통해 자동으로 역량 질문(competency question, CQ)을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: RAG 접근 방식은 두 가지 주요 구성 요소로 이루어져 있습니다: RAG 파이프라인 및 프롬프트 엔지니어링입니다. 컷오프 크기에 따라 문서를 조각으로 나누고, 각 조각은 임베딩 벡터로 변환되어 벡터 데이터베이스에 저장됩니다. 이후 사용자 쿼리는 임베딩 벡터로 변환되어 유사한 조각들이 검색됩니다. 최종적으로 검색된 조각들은 LLM이 사용자 쿼리에 대한 답변을 생성하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, RAG 접근 방식이 CQ 생성을 위해 매우 효과적임을 보여줍니다. 특히, 문서의 수(Np_{paper})를 증가시키면 성능이 향상되는 경향이 있으며, LLM의 온도 설정(temp) 역시 결과에 영향을 미칩니다. 기존의 제로샷 프롬프트 방법과 비교했을 때, 관련 도메인 지식이 포함되면 LLM의 CQ 생성 성능이 향상되었습니다.



### CPL: Critical Planning Step Learning Boosts LLM Generalization in Reasoning Tasks (https://arxiv.org/abs/2409.08642)
- **What's New**: 이번 논문에서는 기존의 대형 언어 모델(LLMs)의 추론(Reasoning) 능력을 향상시키기 위한 새로운 방법인 Critical Planning Step Learning (CPL)을 제안합니다. CPL은 Monte Carlo Tree Search (MCTS)를 활용하여 다단계 추론 작업에서 다양한 계획 단계를 탐색합니다.

- **Technical Details**: CPL은 장기적인 결과를 기반으로 단계별 계획 선호도를 배우고, 이를 통해 모델의 계획 능력을 향상시킵니다. 또한 Step-level Advantage Preference Optimization (Step-APO)를 도입하여 MCTS를 통해 얻은 단계별 선호 쌍의 장점 추정치를 DPO에 통합함으로써, 복잡한 다단계 추론 작업을 개선합니다.

- **Performance Highlights**: GSM8K와 MATH에서 훈련된 우리 방법은 GSM8K에서 10.5% 향상, MATH에서 6.5% 향상을 보여주며, ARC-C (+4.0%), BBH (+1.8%), MMLU-STEM (+2.2%) 및 MMLU (+0.9%)와 같은 도메인 외의 추론 벤치마크에서도 성능이 개선되었습니다.



### Developing an Algorithm Selector for Green Configuration in Scheduling Problems (https://arxiv.org/abs/2409.08641)
- **What's New**: 본 논문은 Job Shop Scheduling Problem (JSP)의 최적화 및 에너지 효율성 향상을 위한 지능형 알고리즘 선택 도구의 프레임워크를 도입합니다. 이 프레임워크는 문제의 복잡성을 나타내는 주요 특징을 식별하고, 적합한 알고리즘 선택을 안내합니다.

- **Technical Details**: 제안된 프레임워크는 기계 학습 기술, 특히 XGBoost를 활용하여 GUROBI, CPLEX, GECODE와 같은 최적의 해법을 추천합니다. GUROBI는 작은 인스턴스에서 우수성을 보이며, GECODE는 복잡한 시나리오에서 강력한 확장성을 나타냅니다. 이 알고리즘 선택 도구는 새로운 JSP 인스턴스에 대한 최선의 알고리즘 추천에서 84.51%의 정확도를 달성했습니다.

- **Performance Highlights**: 제안된 알고리즘 선택 도구는 JSP 문제를 해결하는 데 있어 알고리즘 선택의 효율성을 강조하며, 다양한 JSP 시나리오에 걸쳐 적용 가능성을 높이는 데 목표를 두고 있습니다. 이로 인해 제조물류에서의 효율성과 지속 가능성 향상에 기여할 것으로 기대됩니다.



### Explaining Datasets in Words: Statistical Models with Natural Language Parameters (https://arxiv.org/abs/2409.08466)
- **What's New**: 이 논문은 대량의 데이터를 해석하기 위해 자연어(natural language) 술어(predicates)에 의해 매개변수화된 통계 모델의 패밀리를 도입합니다. 이 모델은 클러스터링(clustering), 시계열(time series), 분류(classification) 작업을 포함하며, 예를 들어 COVID에 관한 텍스트 클러스터는 "COVID에 대해 논의함"이라는 술어로 매개변수화될 수 있습니다.

- **Technical Details**: 슬림화된 통계 모델의 매개변수를 통해 해석 가능성을 높이기 위해, 저자들은 Grazing Descent와 같은 방법을 통해 연속 완화(continuous relaxations)를 최적화하는 모델 독립적(model-agnostic) 알고리즘을 개발했습니다. 이를 통해 데이터에서의 술어 매개변수의 최적화를 시도하고, 언어 모델(large language models)을 통해 이들을 이산형(discrete)으로 변환합니다.

- **Performance Highlights**: 프레임워크는 사용자 대화 대화를 세분화하고, 시간이 지남에 따라 변화하는 방법을 설명하며, 다양한 언어 모델을 비교하고, 수학 문제를 클러스터링하는 데 적용되었습니다. 특히, 이 방법은 텍스트 및 비주얼 도메인에서 활용 가능하고, 다양한 부작용을 조절할 수 있어 기존의 n-그램 분석 같은 방법이 어려운 복잡한 개념을 설명할 수 있습니다.



### Inter Observer Variability Assessment through Ordered Weighted Belief Divergence Measure in MAGDM Application to the Ensemble Classifier Feature Fusion (https://arxiv.org/abs/2409.08450)
- **What's New**: 이 연구는 다중 속성 그룹 의사 결정(MAGDM) 중 전문가 의견 간의 갈등을 고려하고 불확실성을 처리하는 새로운 방법론인 확신 기반 MAGDM을 제안합니다.

- **Technical Details**: 제안된 방법은 기본 확률 할당(BPA) 생성, 순서화된 가중치 신념 및 허용성(measure) 평가 방법을 포함하여 전문가 그룹 간의 관찰 변동성과 갈등을 처리합니다. 이 방법은 다수의 전문가를 통해 최종 선호 관계를 도출하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 방법론은 망막 질환 진단을 위한 OCT 이미지에 대해 실제 사례를 통해 효과를 입증합니다. 연구 결과는 전문가 간의 갈등을 해결하고 불확실성을 관리하는 능력을 보여줍니다.



### Bayesian Inverse Graphics for Few-Shot Concept Learning (https://arxiv.org/abs/2409.08351)
- **What's New**: 이번 연구에서는 사람의 적은 예시만으로 새로운 개념을 학습하는 능력을 모방하는 Bayesian 비전 모델을 제안합니다. 이 모델은 최소한의 데이터만을 사용하여 프로토타입 확률 프로그램을 통해 오브젝트를 학습합니다.

- **Technical Details**: 제안된 모델은 generative inverse graphics 모델을 통해 기본 도형의 후방 분포를 추론하며, Markov Chain Monte Carlo (MCMC)를 사용하여 물리적으로 일관된 매개변수 분포를 샘플링합니다. 또한, 새로운 differentiable renderer를 통해 글로벌 장면 매개변수를 최적화합니다.

- **Performance Highlights**: 제안된 모델은 기존의 few-shot neural-only 분류 알고리즘보다 성능이 우수하며, 다양한 조명 조건 및 배경에서의 일반화 능력을 보여줍니다. 특히, 기존 모델에 비해 적은 수의 학습 이미지로도 높은 정확도를 달성합니다.



### The unknotting number, hard unknot diagrams, and reinforcement learning (https://arxiv.org/abs/2409.09032)
Comments:
          29 pages, 17 figures

- **What's New**: 이 논문에서는 최대 200개의 교차점을 가진 매듭 도표에 대한 최소한의 unknotting crossing changes를 찾기 위한 reinforcement learning agent를 개발했습니다. 이를 통해 57,000개의 매듭의 unknotting number를 결정했습니다.

- **Technical Details**: 개발된 강화학습 에이전트는 반대 서명 시그니처를 가진 매듭의 연결합 도표를 사용하여 작업하였으며, 여러 교차점 변경이 hyperbolic knot을 초래하는 사례도 발견했습니다. 이 에이전트를 통해 2.6백만 개의 다양한 hard unknot 도표 데이터셋을 확보했습니다.

- **Performance Highlights**: 이 연구는 unknotting number의 가산성을 가정하여 최대 12개의 교차점을 가진 43개의 매듭의 unknotting number도 결정했습니다.



### Agents in Software Engineering: Survey, Landscape, and Vision (https://arxiv.org/abs/2409.09030)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문은 소프트웨어 엔지니어링(SE) 분야에서 LLM(대형 언어 모델) 기반 에이전트의 연구를 통합한 첫 번째 설문조사를 수행하였습니다. LLM 기반 에이전트를 구성하는 세 가지 핵심 모듈인 인지(Perception), 기억(Memory), 행동(Action)에 대한 개념적 프레임워크를 제공합니다.

- **Technical Details**: 이 연구는 LLM 기반 에이전트 기술을 소프트웨어 엔지니어링(SE) 분야에 적용한 115개의 논문을 수집하고, 이를 필터링하여 품질 평가를 통해 분석하였습니다. 에이전트의 인지 모듈은 다양한 양식의 입력(예: 텍스트, 비주얼, 청각)을 처리하며, 기억 모듈은 의미 기억(Semantic Memory), 회상 기억(Episodic Memory), 절차 기억(Procedural Memory)을 포함해 에이전트의 추론 결정을 돕습니다. 행동 모듈은 내부 행동(추론, 검색, 학습)과 외부 행동(환경과의 상호작용)을 포함합니다.

- **Performance Highlights**: LLM 기반 에이전트의 성능을 향상시키기 위해 기존 작업의 여러 도전 과제를 정리하고, 새로운 연구 기회를 제시합니다. 주요 제안으로는 다양한 양식의 입력을 탐색하는 것, 복잡한 SE 작업을 수행할 수 있는 다양한 능력을 갖춘 에이전트를 개발하는 것, 외부 검색 기반으로 사용될 수 있는 코드 관련 지식 기반이 필요하다는 점을 강조합니다.



### Towards Leveraging Contrastively Pretrained Neural Audio Embeddings for Recommender Tasks (https://arxiv.org/abs/2409.09026)
Comments:
          Accepted at the 2nd Music Recommender Workshop (@RecSys)

- **What's New**: 본 논문은 음악 추천 시스템에서의 차별적 사전 훈련된 신경 오디오 임베딩 모델(Contrastive Language-Audio Pretraining, CLAP)을 사용하여 협업 필터링 기반 방법을 강화하는 방안을 탐구합니다. 이 모델은 음악 조각, 아티스트 및 사용자 간의 관계를 보다 풍부하고 미묘하게 표현하는데 기여합니다.

- **Technical Details**: 저자들은 GNN(Graph Neural Networks)을 활용한 아티스트 관계 예측 작업을 통해 딥러닝 기반 음악 추천 시스템의 한계를 찾아내고, CLAP 임베딩과 기존의 오디오 특징을 비교합니다. 그들은 그래프 층, 노드 특징, 및 다양한 조합을 사용하여 실험을 수행하고, CLAP 임베딩이 음악 추천 작업의 성능을 크게 향상시킬 수 있음을 입증했습니다.

- **Performance Highlights**: 실험 결과, CLAP 임베딩을 사용하는 모델은 기존의 오디오 특징보다 우수한 성능을 보였으며, 특히 그래프의 구조 정보와 결합했을 때 효과적인 결과를 나타냈습니다. CLAP 임베딩과 다른 특징들의 조합 역시 향상된 성능을 보여주었고, 이는 모델 성능을 최대화하는 데 도움이 됩니다.



### VAE Explainer: Supplement Learning Variational Autoencoders with Interactive Visualization (https://arxiv.org/abs/2409.09011)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문은 Variational Autoencoder(VAE)를 위한 VAE Explainer라는 새로운 도구를 소개합니다. 이 도구는 기존의 정적인 문서화(예: Keras 코드 예시)에 대하여 상호작용형 시각화를 추가합니다.

- **Technical Details**: VAE Explainer는 사용자가 손으로 그린 입력을 통해 인코딩 분포와 재구성이 어떻게 변화하는지 시각적으로 이해할 수 있게 하는 상호작용형 하이레벨 요약 뷰와, 코드와 함께 구현 세부 사항을 설명하는 로우레벨 그래프 뷰를 제공합니다.

- **Performance Highlights**: VAE Explainer는 공개 소스이며, 모든 사용자가 웹 브라우저에서 접근할 수 있도록 구현되었습니다. 이 도구는 MNIST Digit 데이터셋을 사용하여 직관적인 교육 효과를 높였습니다.



### Contri(e)ve: Context + Retrieve for Scholarly Question Answering (https://arxiv.org/abs/2409.09010)
- **What's New**: 이 논문은 Scholar-QALD 데이터셋을 사용해 학술적 지식 그래프 기반의 질문 응답 시스템을 구축하는 두 단계 접근 방식을 제안합니다. 첫 번째 단계에서는 구조화된 및 비구조화된 데이터 출처로부터 질문과 관련된 문맥을 추출하고, 두 번째 단계에서는 LLM(Large Language Model)의 정보 검색 성능을 향상시키기 위한 프롬프트 엔지니어링을 수행합니다.

- **Technical Details**: 제안하는 시스템은 DBLP와 SemOpenAlex의 두 가지 학술 지식 그래프와 위키백과(English Wikipedia) 텍스트의 세 가지 데이터 소스에서 질문에 대한 답변을 찾기 위해 혼합(hybrid) 솔루션을 채택합니다. Llama3.1 LLM을 활용하여 프롬프트 엔지니어링을 적용하고, F1 점수 40%를 달성하였습니다.

- **Performance Highlights**: 다양한 출처에서 정보를 추출한 후, 모델의 응답 품질이 향상되었으며, 비정상적인 응답(anomalous responses) 사례도 관찰되었습니다. 이 결과는 또한 논문의 마지막 부분에서 논의됩니다.



### SGFormer: Single-Layer Graph Transformers with Approximation-Free Linear Complexity (https://arxiv.org/abs/2409.09007)
Comments:
          Extended version of NeurIPS2023 contribution arXiv:2306.10759

- **What's New**: 본 논문은 그래프 상에서의 Transformer 아키텍처의 복잡성을 줄이면서도 효과적인 표현 학습을 가능하게 하는 Simplified Single-layer Graph Transformers(SGFormer)를 제안합니다.

- **Technical Details**: SGFormer는 단일 레이어의 global attention과 GNN 네트워크를 선형적으로 결합하여 그래프의 모든 쌍 상호작용을 처리합니다. 이는 레이어 수가 아닌 단일 레이어로 구현되며, 노드 수에 비례하여 선형적으로 확장됩니다.

- **Performance Highlights**: SGFormer는 중간 크기의 그래프에서 훈련 및 추론 시간에 있어 최대 20배 및 30배의 성능 향상을 보였으며, 웹 규모 그래프 ogbn-papers100M에서도 효과적으로 확장됩니다.



### E2MoCase: A Dataset for Emotional, Event and Moral Observations in News Articles on High-impact Legal Cases (https://arxiv.org/abs/2409.09001)
- **What's New**: 이번 연구에서는 E2MoCase라는 새로운 데이터셋을 소개하여, 법적 사건에 대한 미디어 보도가 공공의 의견을 어떻게 형성하는지를 분석합니다. 이 데이터셋은 감정, 도덕적 가치, 사건을 포괄적으로 분석할 수 있도록 설계되었습니다.

- **Technical Details**: E2MoCase는 감정 탐지(emotion detection), 도덕적 가치 식별(moral value identification), 사건 추출(event extraction)을 위한 고급 모델을 활용하여, 법적 사건이 뉴스 기사에서 어떻게 묘사되는지를 다차원적으로 분석합니다.

- **Performance Highlights**: 이 연구는 감정 톤(emotional tone), 도덕적 프레이밍(moral framing) 및 구체적인 사건을 포착하는 통합 분석 방식을 통해 법적 내러티브와 미디어 보도의 편향을 탐색하는 데 기여합니다.



### Predicting Trust In Autonomous Vehicles: Modeling Young Adult Psychosocial Traits, Risk-Benefit Attitudes, And Driving Factors With Machine Learning (https://arxiv.org/abs/2409.08980)
Comments:
          31 pages (including references and appendix), 7 figures, 7 tables

- **What's New**: 이 연구는 자율주행차(AV)에 대한 젊은 성인의 신뢰를 결정짓는 개인적 요인을 머신러닝을 통해 이해하려고 하며, 1457명의 설문 데이터를 이용하여 신뢰 판단에 가장 중요한 요인을 규명했다.

- **Technical Details**: 설문조사 방법을 통해 심리적, 인지적 특성, 문화적 가치, 운전 스타일 및 기술 태도 등 광범위한 변수를 측정하였다. SHAP(Shapley Additive Explanations) 기법을 사용하여 다양한 요인들이 신뢰 예측 모델에 미치는 영향을 분석하였다.

- **Performance Highlights**: 모델의 정확도는 85.8%로, AV에 대한 위험 및 이점 인식, 운영 가능성에 대한 태도, 제도적 신뢰, 이전 경험 등이 신뢰 예측의 가장 중요한 요인으로 나타났다.



### PINNfluence: Influence Functions for Physics-Informed Neural Networks (https://arxiv.org/abs/2409.08958)
- **What's New**: 최근 물리 정보 신경망(Physics-Informed Neural Networks, PINNs)은 물리 과학의 편미분 방정식에 적용되는 유망한 딥러닝 기술로 주목받고 있습니다. 본 논문에서는 PINNs의 신뢰도를 높이기 위해 영향 함수(Influence Functions, IFs)를 활용하는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 2D Navier-Stokes 유체 흐름 문제에 PINNs를 적용하여, 다양한 협동점(collocation points)의 영향을 평가할 수 있도록 IF 기반 지표를 제시합니다. IF는 개별 훈련 포인트가 모델의 동작에 미치는 기여도를 시스템적으로 평가할 수 있게 해줍니다. 이러한 점에서 IF를 PINNs에 적응하여 해석 가능성을 개선하는 접근 방식을 탐구합니다.

- **Performance Highlights**: 실험을 통해 우리는 IF가 PINNs에 적용되었을 때 여러 버전의 PINNs 간의 혼동을 해소하는 데 유용하다는 것을 입증했습니다. 이를 통해 전문가는 PINNs의 성능을 평가하고 개선할 수 있는 잠재력을 확인할 수 있습니다.



### Optimization and Generalization Guarantees for Weight Normalization (https://arxiv.org/abs/2409.08935)
- **What's New**: 이번 연구에서는 Weight normalization (WeightNorm)의 깊은 모델에 대한 최적화 및 일반화 특성을 이론적으로 처음으로 정리하였습니다. 특히, 부드러운 활성화 함수를 가진 네트워크에 초점을 맞추었습니다.

- **Technical Details**: WeightNorm의 Hessian 형태를 분석하여, 네트워크의 너비와 WeightNorm이 없는 네트워크와의 특이한 의존성을 바탕으로 최적화 보장을 확립하였습니다. 또한 Rademacher 복잡성을 기반으로 하는 일반화 경계를 확보하였습니다. 우리의 경계는 깊이에 대해 다항식 의존성을 가지며, 훈련 과정에서 Gradient Descent (GD)를 사용하여 최적화 보장을 제공하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, 최적화 수렴 속도는 두 가지 양, 즉 ∥∇ℒ∥22/ℒ 및 최소 가중치 벡터 노름의 증가에 따라 개선됨을 확인했습니다. 이는 이론적 결과와도 일치하며, 더욱 효과적인 WeightNorm 네트워크 훈련을 가능하게 합니다.



### XSub: Explanation-Driven Adversarial Attack against Blackbox Classifiers via Feature Substitution (https://arxiv.org/abs/2409.08919)
- **What's New**: 이 논문에서는 XAI(Explainable AI)의 정보를 활용하여 블랙박스 클래시파이어에 대한 새로운 설명 기반 적대 공격인 XSub를 개발하였습니다. XSub는 모델에 의해 결정적으로 식별된 중요한 특징을 대체하여 적대적 예제를 생성하는 전략을 사용합니다.

- **Technical Details**: XSub는 블랙박스 모델에 대한 적대 공격으로, 특정 특징을 대체함으로써 모델이 잘못 분류하도록 유도합니다. XSub의 특징은 설명을 통해 공격자의 의도에 맞춘 정확한 특징 대체를 가능하게 하며, 쿼리 복잡도가 O(1)로 유지되어 비용 효과적입니다.

- **Performance Highlights**: 실험 결과, XSub는 효과적이고 비밀스러우며 비용 효과적인 공격 방법으로, 다양한 AI 모델에서 적용 가능성을 보여주었습니다. 또한, 이미지 분류기에서 단일 픽셀 대체로도 공격에 성공하는 경우가 있었습니다.



### Latent Space Score-based Diffusion Model for Probabilistic Multivariate Time Series Imputation (https://arxiv.org/abs/2409.08917)
Comments:
          5 pages, conference

- **What's New**: 이번 논문에서는 Latent Space Score-Based Diffusion Model (LSSDM)을 제안하여 확률적 다변량 시계열 결측값 보완을 수행합니다. LSSDM은 관측된 값을 저차원의 잠재 공간(low-dimensional latent space)으로 투영하고, 레이블 없는 원본 결측 데이터를 효과적으로 처리합니다.

- **Technical Details**: LSSDM은 그래프 컨볼루션 신경망(Graph Convolutional Neural Network, GCN)과 변환기(Transformer) 구조를 사용하여 결측값을 보완합니다. 본 모델은 자가 지도 학습(self-supervised learning) 접근 방식을 통해, 결측 값에 대한 임의의 분포를 추정하며, 입출력의 불확실성을 평가합니다.

- **Performance Highlights**: 실험 결과, LSSDM은 기존의 결측 보완 방법들에 비해 우수한 성능을 나타내며, 결측 처리 메커니즘에 대한 명확한 설명과 불확실성 분석을 제공합니다.



### Farmer.Chat: Scaling AI-Powered Agricultural Services for Smallholder Farmers (https://arxiv.org/abs/2409.08916)
Comments:
          35 pages

- **What's New**: Farmer.Chat는 관개 농업의 필요를 충족하기 위해 Generative AI 기술을 활용한 혁신적인 챗봇으로, 농부들에게 개인화된, 신뢰할 수 있으며, 문맥에 적합한 정보를 제공합니다.

- **Technical Details**: Farmer.Chat는 Retrieval-Augmented Generation (RAG) 기법을 통해 비정형 데이터를 처리하고, 농업 관련 질문에 대해 적시성과 신뢰성을 갖춘 맞춤형 정보를 제공합니다. 다양한 플랫폼에서 사용 가능하며, 여러 언어를 지원하여 접근성을 극대화합니다.

- **Performance Highlights**: Farmer.Chat는 4개국에서 15,000명 이상의 농부와 연결되었으며, 30만 건 이상의 질문에 답변하였습니다. 사용자들은 75% 이상의 질문에 대해 만족스러운 답변을 받았으며, 농사 관행 개선에 기여하고 사용자의 신뢰와 참여도를 높였습니다.



### AnyBipe: An End-to-End Framework for Training and Deploying Bipedal Robots Guided by Large Language Models (https://arxiv.org/abs/2409.08904)
- **What's New**: 이 논문에서는 로봇을 위한 강화 학습(RL) 정책을 교육하고 배포하기 위한 새로운 프레임워크를 제안하고 있으며, 특히 이 프레임워크는 대형 언어 모델(LLM)에 의해 안내되는 방식으로 bipedal(양족) 로봇에서의 효과를 평가하고 있습니다. 이 프레임워크는 LLM 기반의 보상 함수 설계 모듈, RL 훈련 모듈 및 sim-to-real 평가 모듈 세 가지로 구성되어 있습니다.

- **Technical Details**: 프레임워크는 LLM을 활용하여 보상 함수를 자동으로 생성하고, 기존 모델 및 전통적인 제어 알고리즘을 통합하여 RL 훈련의 안정성을 높이는데 중점을 두고 있습니다. 이는 초기 사용자 입력만 필요하며, 추가적인 인간 개입 없이 학습, 훈련 및 배포를 자율적으로 수행할 수 있게 합니다. 또한, homomorphic evaluation(동형 평가) 모듈을 통해 실제 피드백을 시뮬레이션에 통합함으로써, RL 알고리즘의 효과적인 설계를 지원합니다.

- **Performance Highlights**: AnyBipe 프레임워크를 이용한 실험에서 bipedal 로봇이 평탄한 지형과 복잡한 지형을 모두 통과하는 성능이 유의미하게 향상됨을 보여주었습니다. 전통적인 수작업 보상 함수와 비교했을 때, AnyBipe에서 생성된 보상 함수는 더 빠른 수렴과 안정적인 훈련 결과를 초래하며, 인간의 개입 없이도 RL 알고리즘의 성능을 극대화하는 잠재력을 입증하였습니다.



### Synthetic Human Memories: AI-Edited Images and Videos Can Implant False Memories and Distort Recollection (https://arxiv.org/abs/2409.08895)
Comments:
          22 pages, 11 figures, 2 tables

- **What's New**: 이 논문은 AI에 의해 수정된 시각적 자료가 허위 기억(fault memories) 형성에 미치는 영향을 조사한 연구입니다. AI 편집 툴이 스마트폰에 통합되면서 사용자가 사진을 수정하거나 실사 같은 동영상으로 애니메이션화할 수 있게 되었습니다.

- **Technical Details**: 200명의 참가자들이 네 가지 조건(편집되지 않은 이미지, AI 편집 이미지, AI 생성 비디오, AI 편집 이미지로 만든 AI 생성 비디오)으로 나뉘어 기억을 조사하였습니다. 결과적으로, AI 편집된 비주얼이 허위 기억을 유의미하게 증가시켰고, AI 편집 이미지의 비디오가 가장 강력한 영향을 미쳤습니다.

- **Performance Highlights**: AI가 수정된 비주얼의 노출 후 허위 기억이 2.05배 증가하였고, 이 조건에서의 허위 기억에 대한 신뢰도는 1.19배 높았습니다. 이는 HCI(인간-컴퓨터 상호작용)에서의 잠재적 응용 가능성 및 윤리적, 법적, 사회적 문제도 논의하였습니다.



### Establish seedling quality classification standard for Chrysanthemum efficiently with help of deep clustering algorithm (https://arxiv.org/abs/2409.08867)
- **What's New**: 본 논문에서는 식용 국화 모종의 품질 분류 기준을 세우기 위한 새로운 프레임워크인 SQCSEF를 제안합니다. 이 프레임워크는 유연한 클러스터링 모듈을 갖추고 있어 대부분의 식물 종에 적용 가능합니다.

- **Technical Details**: 제안된 SQCSEF 프레임워크는 최신의 deep clustering 알고리즘인 CVCL을 도입하고, 요인 분석(factor analysis)을 통해 여러 관점으로 지표를 나누어 CVCL 방법의 입력으로 사용합니다. 이러한 접근은 더 합리적인 클러스터를 생성하여 식용 국화 모종에 대한 품질 기준 $S_{cvcl}$을 설정합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 SQCSEF 프레임워크의 정확성과 효율성을 검증하였습니다.



### Exploring Graph Structure Comprehension Ability of Multimodal Large Language Models: Case Studies (https://arxiv.org/abs/2409.08864)
- **What's New**: 최근 멀티모달 LLMs가 그래프 구조 이해의 새로운 가능성을 제시함에 따라, 본 연구는 그래프 시각화가 LLM 성능에 미치는 영향을 조사한다.

- **Technical Details**: 본 연구에서는 'GraphQA' 벤치마크 설정을 따르고, Erdős–Rényi (ER) 모델을 사용하여 500개의 그래프 데이터셋을 생성한다. 그래프 텍스트 인코더로는 인접 인코딩과 사건 인코딩을 사용하며, Matplotlib을 이용한 표준화된 그래프 시각화를 적용한다.

- **Performance Highlights**: 실험 결과, 멀티모달 LLM인 GPT-4 및 GPT-4o가 PaLM 모델에 비해 그래프 구조 이해에서 뛰어난 성능을 보였으며, 특정 태스크에서 거의 완벽한 정확도를 나타냈다. 그래프 시각화는 LLMs의 이해도를 향상시키지만, 그 효과는 그래프 구조의 복잡성과 태스크의 종류에 따라 다르게 나타났다.



### Deep reinforcement learning for tracking a moving target in jellyfish-like swimming (https://arxiv.org/abs/2409.08815)
Comments:
          22pages,14 figures

- **What's New**: 이 논문에서는 젤리피쉬와 유사한 수영자를 훈련시키기 위한 딥 강화 학습 방법을 개발했습니다. 이 수영자는 비틀림 스프링에 기반한 근육 모델이 장착된 유연한 객체로, 움직이는 목표를 추적하기 위해 효과적으로 동작합니다.

- **Technical Details**: 딥 Q-network (DQN)를 사용하며, 수영자의 기하학과 동적 매개변수를 입력으로 받아, 수영자에게 적용될 힘을 출력합니다. 복잡한 유체-구조 상호작용에서의 간섭을 완화하기 위해 행동 규제를 도입했습니다. 이 방법은 포함 경계법(immersed boundary method)을 사용하여 수영자의 움직임 데이터를 시뮬레이션하여 수집합니다.

- **Performance Highlights**: DQN 에이전트와 행동 규제를 통해 수영자는 순간적인 상태에 따라 동적으로 경로를 조정할 수 있는 능력을 보여주었습니다. 이 연구는 유체 환경에서 유연한 객체 제어를 위한 머신 러닝의 응용 범위를 확장하여 젤리피쉬의 효율적인 추진 메커니즘을 활용합니다.



### Mutual Theory of Mind in Human-AI Collaboration: An Empirical Study with LLM-driven AI Agents in a Real-time Shared Workspace Task (https://arxiv.org/abs/2409.08811)
Comments:
          34 pages, Preprint Under Review

- **What's New**: 이 연구는 인간-AI 팀(HATs)의 상호작용에서 발생하는 상호 이론적 사고(MToM) 프로세스를 조사하고, AI 에이전트의 이론적 사고(ToM) 능력이 팀 협업에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 LLM 기반의 AI 에이전트를 개발하고, 인간과 AI가 공동 작업을 수행하는 주방 환경에서의 HAT 과제를 구현했습니다. 이 실험은 4×24 디자인을 사용하여 통신 상호작용과 AI 에이전트의 ToM 능력이 팀 성과와 협업 과정에 미치는 영향을 평가합니다.

- **Performance Highlights**: 결과적으로, 양방향 통신 시 팀 성과가 감소했으며, AI 에이전트의 ToM 능력이 팀의 객관적 성과에는 큰 영향을 미치지 않았지만, AI에 대한 인간의 이해를 향상시키고 서로 이해받는 느낌을 증가시켰습니다. 비언어적이고 암묵적인 행동이 실시간 작업에서 인간-AI 협업에 있어서 언어적 의사소통만큼 효과적일 수 있음을 발견하였습니다.



### TabKANet: Tabular Data Modelling with Kolmogorov-Arnold Network and Transformer (https://arxiv.org/abs/2409.08806)
- **What's New**: TabKANet 아키텍처를 기반으로 하는 새로운 방법론을 제안하며, 이를 통해 수치적 특징을 코딩하고 범주형 특징과 통합하여 tabular 데이터를 모델링하는 데 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: TabKANet는 Kolmogorov-Arnold Network (KAN)와 Transformer 아키텍처를 통합하여 tabular 데이터를 모델링하는 것에 중점을 둡니다. KAN의 강력한 기능 추출 능력과 Transformer의 특징 학습 메커니즘을 활용합니다.

- **Performance Highlights**: TabKANet은 6개의 일반적으로 사용되는 이진 분류 작업에서 탁월한 성능을 보여주며, 특히 온라인 쇼핑객 데이터셋에서 큰 개선을 보였습니다. 전통적인 신경망 모델보다 더욱 향상된 결과를 나타냈고, 이는 KAN이 수치적 특징을 맵핑하는 데에 있어서 MLP보다 우수한 성능을 발휘한다는 것을 강조합니다.



### Reading ability detection using eye-tracking data with LSTM-based few-shot learning (https://arxiv.org/abs/2409.08798)
- **What's New**: 이 연구에서는 68명의 피험자(Subjects)로부터 얻은 eye-tracking 데이터를 활용하여 독서 능력(Reading ability) 점수를 예측하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Long Short Term Memory (LSTM)와 경량 신경망(light-weighted neural networks)을 결합한 회귀 모델(Regression model)을 통해 독서 능력 점수를 예측합니다.

- **Performance Highlights**: Few-shot learning 전략을 통해, 제안된 방법은 독서 능력 탐지에서 기존 점수 예측 방법보다 높은 정확도(Accuracy)를 달성하였습니다.



### What You Say = What You Want? Teaching Humans to Articulate Requirements for LLMs (https://arxiv.org/abs/2409.08775)
Comments:
          15 pages, 5 figures

- **What's New**: 본 논문에서는 인간의 요구를 보다 명확하게 전달하기 위한 새로운 패러다임인 Requirement-Oriented Prompt Engineering (ROPE)을 소개합니다. ROPE는 사용자가 고유의 요구 사항을 보다 효과적으로 생성할 수 있도록 지원하며, 이를 통해 LLM의 출력 품질을 향상시키는 방법을 제시합니다.

- **Technical Details**: ROPE는 LLM(large language model)에서 요구 사항을 명확히 하는 데 중점을 둔 개념으로, 특정 요구 사항을 포함한 훈련 및 평가 도구를 통해 구현됩니다. 본 연구에서는 30명의 초보자를 대상으로 한 랜덤화 통제 연구를 통해, 요구 중심 훈련이 초보자의 프롬프트 성능을 두 배로 향상시킨 것을 보여주었습니다.

- **Performance Highlights**: 훈련 받은 초보자들은 요구 사항 성능을 두 배 향상시키고, 요구 사항의 품질과 LLM 출력 품질 간에 강한 상관관계를 보였습니다. ROPE 훈련은 기존의 프롬프트 엔지니어링 훈련 및 최적화보다 현저히 더 좋은 성과를 보였고, 향후 인간과 LLM 간의 협력적 프롬프트 작성에서 중요한 기초가 될 것으로 기대됩니다.



### HOLA-Drone: Hypergraphic Open-ended Learning for Zero-Shot Multi-Drone Cooperative Pursu (https://arxiv.org/abs/2409.08767)
Comments:
          10 pages

- **What's New**: 제로샷 조정(zero-shot coordination, ZSC) 문제를 다룬 새로운 협동 드론 추적 알고리즘 HOLA-Drone을 제안합니다. 본 연구는 기존 2인 비디오 게임에서 벗어나 여러 보이지 않는 파트너와 협력할 수 있는 드론 에이전트를 개발하는 데 중점을 두었습니다.

- **Technical Details**: HOLA-Drone은 하이퍼그래픽 형식(hypergraphic-form) 게임 모델링을 기반으로 하여 학습 목표를 지속적으로 조정하는 혁신적인 오픈 엔드 학습 알고리즘입니다. 이 알고리즘은 분산 부분 관찰 마르코프 결정 과정(Dec-POMDP) 프레임워크 내에서 협동 드론 추적 작업을 제로샷 조정 문제로 형식화합니다.

- **Performance Highlights**: HOLA-Drone은 다양한 시나리오에서 보이지 않는 드론 팀원과의 조정 성능에서 기존 방법보다 우수한 결과를 보였습니다. 실제 Crazyflie 드론을 사용한 실험에 의해 HOLA-Drone의 물리적 시스템에서의 실행 가능성도 검증되었습니다.



### Journalists, Emotions, and the Introduction of Generative AI Chatbots: A Large-Scale Analysis of Tweets Before and After the Launch of ChatGP (https://arxiv.org/abs/2409.08761)
- **What's New**: 이번 연구는 Generative AI의 영향에 대한 더 넓은 조사 일환으로, ChatGPT 출시 당시 언론인들의 감정적 반응을 분석하였다. 주요 미국 뉴스 매체의 언론인들로부터 수집된 거의 100만 개의 트윗을 분석한 결과, ChatGPT 도입 전후의 감정 톤 변화와 감정을 추적하였다.

- **Technical Details**: 연구는 Computational 및 Natural Language Processing 기술을 사용하여 ChatGPT 출시로 인한 감정의 변화를 측정하였다. 분석 결과, 긍정적인 감정의 증가와 함께 출시 이후 더 우호적인 톤이 관찰되었으며, 이는 AI의 잠재력에 대한 초기 낙관적 태도를 나타낸다.

- **Performance Highlights**: 이 연구는 언론인이 기술 혁신과 혼란의 해석자로서 중요한 역할을 수행한다는 점을 강조하며, 그들의 감정적 반응이 새로운 기술에 대한 대중의 내러티브를 형성하는 데 어떻게 기여할 수 있는지를 보여준다. 연구는 저널리즘, 감정, AI의 교차점 이해에 기여하며, Generative AI 도구의 사회적 영향에 대한 통찰력을 제공한다.



### Bridging Dynamic Factor Models and Neural Controlled Differential Equations for Nowcasting GDP (https://arxiv.org/abs/2409.08732)
Comments:
          Accepted at CIKM 2024. Seonkyu Lim and Jeongwhan Choi are co-first authors with equal contributions

- **What's New**: NCDENow는 기존의 동적 요인 모델(DFM)과 신경 제어 미분 방정식(NCDE)을 결합한 새로운 GDP nowcasting 프레임워크로, 경제적 불확실성과 비정기적 데이터의 역동성을 더욱 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: NCDENow는 세 가지 주요 모듈로 구성됩니다: i) DFM을 활용한 요인 추출 모듈, ii) NCDE를 통한 동적 모델링 모듈, iii) 회귀를 통해 GDP 성장 예측을 수행하는 모듈입니다. 이 프레임워크는 비정기적 시계열 데이터의 동작을 다루는 데 있어 나은 성능을 발휘합니다.

- **Performance Highlights**: NCDENow는 한국과 영국의 2개 실제 GDP 데이터셋을 기반으로 6개의 기준 모델과 비교한 결과, 더 높은 예측 정확도를 보여주었습니다. 특히 COVID-19와 같은 불안정한 경제 조건 하에서 탁월한 예측 능력을 발휘했습니다.



### Quasimetric Value Functions with Dense Rewards (https://arxiv.org/abs/2409.08724)
- **What's New**: 이번 연구에서는 Goal Conditioned Reinforcement Learning(GCRL)의 최적 가치 함수들이 특정 조건 하에서 dense 보상이 sample 효율성을 향상시킬 수 있다는 점을 밝혔습니다.

- **Technical Details**: 이 논문은 GCRL의 최적 가치 함수(Q*(s,a,g))가 quasimetric 구조를 가짐을 증명하고, triangle inequality가 dense 보상 설정에서도 유지될 수 있음을 주장합니다. 이는 sample 복잡도를 개선할 수 있는 기회를 제공합니다. 또한, 지식 기반 보상 조작(potential functions)과 같은 특정 조건이 triangle inequality를 만족할 때, dense 보상이 효과적일 수 있다고 설명합니다.

- **Performance Highlights**: 12 개의 표준 benchmark 환경에서 수행한 실험을 통해, dense 보상을 사용하는 GCRL 설정에서 training in dense 보상이 4개의 과제에서 sample 복잡도를 개선하며, 어떤 과제에서도 성능 저하를 발생시키지 않음을 확인했습니다.



### Distilling Monolingual and Crosslingual Word-in-Context Representations (https://arxiv.org/abs/2409.08719)
- **What's New**: 본 연구에서는 미리 학습된 masked language model로부터 문맥 속 단어 의미 표현을 증류(disting)하는 방법을 제안합니다. 이는 monolingual과 crosslingual 설정 모두에 적용 가능합니다. 기존의 방법들과 달리, 우리는 인간 주석 데이터(annotated corpora)나 모델 파라미터의 업데이트 없이 작업을 수행하는 점이 특징적입니다.

- **Technical Details**: 제안된 방법은 auto-encoder 기반의 훈련 방식을 사용하며, 이는 자동 생성된 코퍼스(corpus)를 통해 이루어집니다. 특히, 모델은 self-attention 메커니즘을 사용하여 미리 학습된 모델의 다양한 숨겨진 레이어의 출력을 결합하는 능력을 배웁니다. 이는 context-aware lexical semantics와 unsupervised semantic textual similarity (STS) 추정에 있어 단어 의미를 효과적으로 표현하는 데 도움을 줍니다.

- **Performance Highlights**: 모노링구얼(mono-lingual) 과제에서 우리의 표현이 이전 연구에서 제시한 것보다 경쟁력 있는 성능을 보였고, STS 추정을 위한 작업에서 더 우수한 성능을 발휘했습니다. 또한, 크로스링구얼(cross-lingual) 작업에서도 제안된 방법이 다국어 사전 훈련 모델의 단어 표현을 크게 개선하는 결과를 나타냈습니다.



### Layerwise Change of Knowledge in Neural Networks (https://arxiv.org/abs/2409.08712)
- **What's New**: 이 논문은 딥 뉴럴 네트워크(DNN)가 새로운 지식을 점진적으로 추출하고 잡음을 포함한 특징을 잊어버리는 과정을 층별로 설명합니다. 특히, 중간 층에서 암호화된 상호작용(interactions)을 추출하고 이를 정량화하는 새로운 접근방법을 제시합니다.

- **Technical Details**: 본 연구에서는 입력과 중간 층 특징 간의 상호 정보(mutual information)를 이용하여 DNN 각 층에서 암호화된 지식을 측정하며, 이는 학습 과정에서 새롭게 등장하고 잊혀진 상호작용을 추적하고 정량화하는 과정으로 이어집니다. 또한 인접한 층들은 유사한 상호작용을 암호화하는 경향이 있습니다.

- **Performance Highlights**: 우리는 새로운 상호작용 정의가 DNN의 일반화 능력(generalization capacity)과 어떻게 연결되는지를 밝혀내며, 저차원 상호작용이 고차원 상호작용보다 더 높은 안정성과 일반화 능력을 가짐을 발견했습니다.



### Text-To-Speech Synthesis In The Wild (https://arxiv.org/abs/2409.08711)
Comments:
          5 pages, submitted to ICASSP 2025 as a conference paper

- **What's New**: 이 논문에서는 TTS 시스템의 훈련을 위해 Wild에서 수집된 데이터를 활용하는 새로운 접근 방식을 제안합니다. TITW( Text-To-Speech In the Wild) 데이터세트는 VoxCeleb1 데이터세트를 기반으로 하여 자동화된 파이프라인을 통해 생성되었습니다. 이 데이터세트는 TTS 모델 훈련을 위해 두 가지 세트를 제안합니다: TITW-Hard와 TITW-Easy입니다.

- **Technical Details**: TITW-Hard는 VoxCeleb1의 출처 데이터를 기반으로 한 자동 전사(transcription), 분할(segmentation) 및 선택(selection) 과정을 통해 생성되었습니다. TITW-Easy는 추가적인 음성 향상(enhancement) 및 DNSMOS 기반 데이터 선택을 통해 파생된 데이터세트입니다. 이 데이터는 공공에게 공개되어 TTS 시스템의 벤치마크에 활용될 수 있습니다.

- **Performance Highlights**: TITW-Easy를 사용하는 최근 TTS 모델들은 성공적으로 훈련될 수 있는 것으로 나타났지만, TITW-Hard를 사용할 경우 비슷한 결과를 얻는 것이 매우 어렵다는 것을 보여줍니다. TITW 데이터로 훈련된 TTS 시스템의 평가 및 벤치마킹을 위한 공통 프로토콜(protocol)도 제안됩니다.



### NeSHFS: Neighborhood Search with Heuristic-based Feature Selection for Click-Through Rate Prediction (https://arxiv.org/abs/2409.08703)
- **What's New**: 이번 논문에서는 Click-through-rate (CTR) 예측에서의 성능 향상을 위한 NeSHFS(Neighborhood Search with Heuristic-based Feature Selection)라는 새로운 휴리스틱 알고리즘을 제안합니다. 이 알고리즘은 차원 축소와 훈련 시간 비용을 줄이면서도 효과적인 피처 선택을 제공하기 위해 설계되었습니다.

- **Technical Details**: NeSHFS는 피처 선택을 위한 방식으로, 피처를 순차적으로 제거하고 이웃 검색을 통해 최고의 솔루션을 찾는 접근 방식을 사용합니다. 이 방법은 기존의 그리드 검색 hyperparameter 최적화 기술에서 영감을 받았으며, 특화된 휴리스틱을 적용하여 탐색 공간을 고르게 탐색하고 유망한 영역에서 성능을 극대화합니다. 실험은 DeepCTR 프레임워크를 이용해 진행하였으며, 80/10/10 비율로 데이터셋을 훈련, 검증, 테스트에 나누었습니다.

- **Performance Highlights**: 제안된 NeSHFS는 세 가지 공공 데이터셋(화웨이 Digix 2022, Criteo, Avazu)에서 실험을 통해 효율성과 효과성을 검증하였으며, 피처 집합을 축소함으로써 Deep CTR 모델의 훈련 시간을 비약적으로 줄이며, 더 많은 트래픽에 대한 추론 시간을 단축했습니다. 또한, 이 알고리즘은 다양한 Deep CTR 모델에 적용 가능하며, 최근 훈련된 모델의 적응력 향상에 기여할 수 있습니다.



### DM: Dual-path Magnitude Network for General Speech Restoration (https://arxiv.org/abs/2409.08702)
- **What's New**: 본 논문에서는 다양한 왜곡(디스토션) 문제를 효과적으로 해결하기 위한 신개념 일반 음성 복원 모델인 Dual-path Magnitude (DM) 네트워크를 소개합니다. DM 네트워크는 두 개의 병렬 Magnitude Decoder를 사용하여, 하나는 마스킹 기반 알고리즘을 통해 왜곡을 제거하고, 다른 하나는 매핑 기반 알고리즘을 통해 음성을 복원합니다. 이 첫 번째 네트워크의 출력을 두 번째 네트워크에 연결하는 스킵 커넥션(skip connection)을 도입하여 복원 성능을 극대화하였습니다.

- **Technical Details**: DM 네트워크는 MP-SENet 프레임워크를 바탕으로 하며, 명확한 구조는 인코더, 두 단계의 변환기(Conformer), 디코더 블록으로 구성되어 있습니다. 병렬 Magnitude Decoder는 마스킹 기반과 매핑 기반으로 구성되어 있으며, 이들은 서로 다른 활성화 함수(learnable sigmoid, ReLU)를 사용합니다. 복원 과정에서는 두 개의 Magnitude Decoder의 출력을 가중치(ω)를 이용해 결합합니다. 또한, 추출된 스펙트로그램을 매핑 기반 디코더의 출력에 통합하는 새로운 접근 방식을 채택했습니다.

- **Performance Highlights**: 실험 결과, DM 네트워크는 음성 복원 작업에서 다른 기준 모델들보다 우수한 성능을 보였으며, 적은 수의 매개변수로도 상당한 복원 성능을 달성하였습니다.



### Precision Aquaculture: An Integrated Computer Vision and IoT Approach for Optimized Tilapia Feeding (https://arxiv.org/abs/2409.08695)
Comments:
          8 pages, 6 figures, 3 tables, 21th International Conference on Informatics in Control, Automation, and Robotics

- **What's New**: 전통적인 어류 양식에서는 비효율적인 피딩으로 인해 환경 문제가 발생하고 생산성이 저하되는 경우가 많습니다. 본 연구에서는 컴퓨터 비전(Computer Vision)과 IoT 기술을 결합한 혁신적인 시스템을 개발하여 Tilapia(틸라피아) 어류의 정확한 사료 공급을 실현했습니다.

- **Technical Details**: 실시간 IoT 센서를 사용해 수질 매개변수를 모니터링하고, YOLOv8 모델을 활용해 어류 크기와 숫자를 분석하여 최적의 사료 양을 결정합니다. 두 대의 카메라를 사용하여 pH 수준과 용존 산소를 모니터링하고 여러 급수를 위한 데이터 수집이 이루어집니다. 데이터는 모바일 애플리케이션으로 전달되어 쉽게 접근할 수 있습니다.

- **Performance Highlights**: 모델은 3,500개의 주석 이미지에서 94%의 정밀도를 달성하였으며, 이 접근 방식은 기존 양식과 비교하여 생산량을 최대 58배 증가시킬 수 있는 가능성을 제시합니다. 또한, 모든 모델, 코드, 데이터셋은 오픈 소스(Open Source)로 제공됩니다.



### B4: Towards Optimal Assessment of Plausible Code Solutions with Plausible Tests (https://arxiv.org/abs/2409.08692)
Comments:
          accepted by ASE' 24 (full paper)

- **What's New**: 이 논문은 코드 생성 과정에서 여러 대안 중 최적의 코드 솔루션을 선택하는 새로운 방법론을 제시합니다. 특히, 근거가 불명확한 테스트 케이스를 사용할 때의 한계를 극복하고, Bayesian (베이지안) 프레임워크를 기반으로 최적 선택 전략을 정의합니다.

- **Technical Details**: 제안된 방법론은 코드 솔루션과 테스트 케이스 간의 관측된 상태의 사후 확률(posterior probability)을 기반으로 최적 선택 전략을 설정하고, 이를 정수 계획 문제(Integer Programming Problem)로 정형화합니다. 그 후, 불확실한 초기 분포(prior distributions)를 고려하여 유효한 사후 확률의 근사(computable form)를 계산할 수 있는 방법을 도출했습니다. 이 근사 방법은 최대 다항식 복잡도(polynomial complexity)로 최적화할 수 있으며, 초기 지식의 정확성에 의해 오류가 한정됩니다.

- **Performance Highlights**: 제안된 B4 (𝓑^4) 전략은 LLM으로 생성된 코드 솔루션과 LLM으로 생성된 테스트 케이스를 사용한 선택 과정에서 기존의 휴리스틱(heuristics)보다 최대 50% 향상된 성능을 보여주었습니다. 특히, 가장 어려운 상황에서는 기존의 가장 강력한 휴리스틱에 비해 246%의 성능 개선이 있었습니다.



### NEST-RQ: Next Token Prediction for Speech Self-Supervised Pre-Training (https://arxiv.org/abs/2409.08680)
Comments:
          5 pages, 2 figures, Work in progress

- **What's New**: 본 논문에서는 다음 토큰 예측(next token prediction)을 기반으로 한 음성 자가 감독(pre-training) 방법인 NEST-RQ를 제안합니다. 기존의 음성 SSL 방법들이 비인과적(encoders) 인코더를 활용하는 반면, NEST-RQ는 인과적(causal) 인코더를 사용하여 스트리밍 모델에 적합하도록 설계되었습니다.

- **Technical Details**: NEST-RQ는 랜덤 프로젝션 양자화기(random-projection quantizer)를 활용하여 연속 음성 특징을 분리된 토큰(sequence of discrete tokens)으로 변환합니다. 이 과정에서 인과적 인코더를 이용하여 오직 왼쪽 문맥(left context)만을 고려하여 다음 토큰을 예측하는 방식을 취하고 있습니다. 이를 통해, 음성 SSL의 가능한 개선점을 제시합니다.

- **Performance Highlights**: NEST-RQ는 30만 시간의 라벨이 없는 음성 데이터와 3만 시간의 라벨이 있는 데이터셋을 활용하여 실험되었으며, 비스트리밍 ASR에서는 기존 BEST-RQ와 유사한 성능을 보이고, 스트리밍 ASR에서는 더 나은 성능을 달성했습니다. 이는 NEST-RQ의 간단한 설계가 BEST-RQ의 강점을 물려받으면서 성능을 유지한다는 것을 보여줍니다.



### Shadow Program Inversion with Differentiable Planning: A Framework for Unified Robot Program Parameter and Trajectory Optimization (https://arxiv.org/abs/2409.08678)
Comments:
          8 pages, 6 figures, submitted to the 2025 IEEE International Conference on Robotics & Automation (ICRA)

- **What's New**: 본 논문은 SPI-DP(S shadow Program Inversion with Differentiable Planning)라는 새로운 첫 번째 오더(First-order) 최적화 도구를 소개합니다. 이 도구는 로봇 프로그램을 고수준 작업 목표와 동작 수준 제약 조건 모두에 대해 최적화할 수 있습니다.

- **Technical Details**: SPI-DP는 DGPMP2-ND라는 미분 가능한 충돌 회피 모션 플래너를 통해 일반적인 매개변수화된 로봇 프로그램 표현을 위한 반복적인 기울기 기반 최적화 접근법으로 통합됩니다. 이 방식은 로봇의 계획된 경로와 프로그램 매개변수를 주기 시간(cycle time)이나 부드러움(smoothness) 같은 목표에 대해 최적화할 수 있으며, 충돌 제약 조건을 준수하도록 설계되었습니다.

- **Performance Highlights**: 본 연구는 가정용 및 산업용 피그-인-홀 응용 프로그램에서 SPI-DP의 포괄적인 평가 결과를 제시하며, 로봇 프로그램의 매개변수 최적화 및 경로 최적화를 통합적으로 수행할 수 있는 첫 번째 접근법으로 자리잡고 있습니다.



### Towards certifiable AI in aviation: landscape, challenges, and opportunities (https://arxiv.org/abs/2409.08666)
- **What's New**: 이 논문은 항공 전자기기에 대한 AI 인증의 종합적인 마인드 맵을 제시하며, AI 개발 인증의 도전과제를 강조합니다. 특히 성능 지표 이상의 자격 요건이 필요하다는 것을 강조합니다.

- **Technical Details**: 이 논문은 다양한 AI 인증 블록을 정의하고, AI의 안전성 및 윤리에 대한 분석을 포함합니다. EASA의 Level 1 및 Level 2 AI에 대한 인증 지침이 제시되며, 기존의 소프트웨어 및 시스템 개발 표준(예: DO-178C, ARP4754B)에 대한 한계를 언급합니다. 또한, ML(Machine Learning) 개발 주기가 AI 시스템의 인증에 어떻게 적용되는지를 설명합니다.

- **Performance Highlights**: 비교된 기존 기준과 달리, AI 인증은 복잡한 항공 전자기기 시스템의 안전성과 신뢰성을 보장하기 위해 새로운 접근 방식을 요구합니다. EASA의 작업 결과는 유럽의 항공 산업에 대한 AI 인증 표준을 제공하며, FAA의 로드맵은 AI 안전성을 확보하는 데 기여할 것으로 보입니다.



### LMAC-TD: Producing Time Domain Explanations for Audio Classifiers (https://arxiv.org/abs/2409.08655)
Comments:
          The first two authors contributed equally to this research. Author order is alphabetical

- **What's New**: 이번 논문은 LMAC-TD라는 새로운 post-hoc 설명 방법을 제안합니다. 이 방법은 L-MAC(리스트너블 맵, Listenable Maps for Audio Classifiers)의 기존 아키텍처를 기반으로 하여, 시간 도메인에서 직접적인 설명을 생성합니다.

- **Technical Details**: LMAC-TD는 세 프레임워크인 UNet, SepFormer 그리고 MaskNet을 통합하여 고품질의 오디오 설명을 생성합니다. 이 방법은 분류기와 입력 오디오의 특성을 결합하여 설명의 신뢰성을 높이며, l^{2} divergence를 최소화하는 손실 함수로 학습됩니다.

- **Performance Highlights**: 사용자 연구를 통해 LMAC-TD는 기존 L-MAC보다 더 높은 오디오 품질을 유지하며, 신뢰성 지표 측면에서도 유사한 결과를 달성했습니다.



### Utilizing Data Fingerprints for Privacy-Preserving Algorithm Selection in Time Series Classification: Performance and Uncertainty Estimation on Unseen Datasets (https://arxiv.org/abs/2409.08636)
Comments:
          Hawaii International Conference on System Sciences (HICSS-58) 2025

- **What's New**: 본 연구는 데이터 핑거프린트(data fingerprint)를 도입하여 시계열 분류(time series classification) 알고리즘 선택 문제를 해결하는 새로운 접근 방식을 제안합니다. 이 방법은 전체 데이터에 대한 접근 없이 알고리즘 성능을 추정할 수 있는 기능을 제공합니다.

- **Technical Details**: 데이터 핑거프린트는 프라이버시를 유지하면서 시계열 데이터셋을 특성화하며, 멀티타겟 회귀(multi-target regression) 문제로 알고리즘 선택 문제를 변환합니다. 이 연구는 112개의 벤치마크 데이터셋을 사용하여 35개의 최신 알고리즘의 성능을 평가하였으며, 알고리즘 성능의 불확실성을 추정하는 데 데이터 핑거프린트를 활용합니다.

- **Performance Highlights**: 제안된 방법은 35개의 최신 알고리즘의 평균 성능 추정에서 7.32% 향상, 알고리즘 성능의 불확실성 추정에서 15.81% 향상을 달성하였습니다. 이는 알고리즘 선택 과정에서 유용한 통찰력을 제공하며, 새롭고 보이지 않는 데이터셋에서 알고리즘 성능을 효과적으로 예측할 수 있는 가능성을 보여줍니다.



### Improving Analog Neural Network Robustness: A Noise-Agnostic Approach with Explainable Regularizations (https://arxiv.org/abs/2409.08633)
- **What's New**: 이번 연구는 딥 아날로그 신경망(deep analog neural networks)에서 발생하는 '하드웨어 노이즈'를 완화하는 새로운 접근 방식을 제안합니다. 이 연구의 혁신적인 점은 노이즈에 강한 네트워크의 작동 메커니즘을 설명하고, 이를 통해 노이즈 저항성을 크게 향상시킬 수 있는 설명 가능한 정규화 프레임워크를 도입한 것입니다.

- **Technical Details**: 이 연구는 상관 노이즈(correlated noise)와 비상관 노이즈(uncorrelated noise)를 구분하고, 노이즈에 대한 저항성을 강화하는 새로운 정규화 전략을 제시합니다. 노이즈는 신경망의 활성화층에서 발생하는 하드웨어 노이즈에 집중되어 있으며, 이 프레임워크는 신경망의 가중치 행렬이 0으로 합산되도록 유도하여 상관 노이즈를 효과적으로 완화합니다.

- **Performance Highlights**: 이 전략은 시간 시계열 예측 과제에서 거의 4배 향상된 예측 정확성을 보여주었으며, 노이즈 저항성을 증대시키기 위한 수학적으로 지지된 설명을 제공합니다. 다양한 신경망 구조에서 적용 가능성이 검증되었으며, 실제로 성능 개선이 입증되었습니다.



### Sybil Detection using Graph Neural Networks (https://arxiv.org/abs/2409.08631)
Comments:
          9 pages, 1 figure, 6 tables

- **What's New**: 이 논문에서는 Graph Attention Networks (GATs)를 활용한 새로운 Sybil 탐지 기법인 SYBILGAT을 제안합니다. 기존의 Sybil 탐지 방법들이 네트워크의 구조적 특성에 의존하던 반면, SYBILGAT은 노드 간의 주의(attention) 가중치를 동적으로 할당하여 성능을 향상시킵니다.

- **Technical Details**: SYBILGAT은 GATs를 기반으로 하여, 노드 집합의 집계 과정에서 각 노드에 동적으로 주의 가중치를 부여함으로써 Sybil 탐지에 필요한 정보를 더욱 효과적으로 집중할 수 있도록 합니다. 이 방법은 다양한 크기와 구조의 합성 네트워크, 표적 공격을 견디는 경우의 실험을 포함하여, 실세계 데이터셋인 Twitter 및 Facebook 네트워크에 대해 테스트 되었습니다.

- **Performance Highlights**: SYBILGAT은 기존의 최첨단 알고리즘을 상당히 초월하는 성능을 보이며, 특히 높은 공격 복잡성과 그에 따른 많은 공격 엣지 상황에서 우수한 결과를 나타냅니다. 여러 네트워크 모델과 규모에서 강건한 성능을 보여주며, 실제 269k 노드와 6.8M 엣지가 포함된 Twitter 그래프에 성공적으로 적용되었습니다.



### Using Convolutional Neural Networks for Denoising and Deblending of Marine Seismic Data (https://arxiv.org/abs/2409.08603)
- **What's New**: 이 연구에서는 해양 지진 데이터(Seismic Data)의 처리에서 신경망(Neural Network)을 활용하여 처리 시간을 크게 단축하고 효율성을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 딥 컨볼루션 신경망(Deep Convolutional Neural Networks, CNNs)을 사용하여 지진 간섭 소음(Seismic Interference Noise)을 제거하고 지진 데이터를 분리하는 방법을 제안합니다. 단일 샷 수집(Shot Gather)에 1,000,000개 이상의 데이터 샘플이 포함되어 많은 양의 계산 메모리가 필요하다는 점이 중요합니다. 신호 대 잡음비(Signal-to-Noise Ratio, SnR)가 결과에 영향을 미치는 것으로 나타났습니다.

- **Performance Highlights**: 공통 채널 도메인(Common Channel Domain)으로 이동하여 노이즈의 일관성을 줄이고 입력 볼륨 크기를 감소시킴으로써 CNN을 활용한 분리 작업에서 더 나은 성능을 보여주었습니다. 이는 샷 도메인(Shot Domain)에서의 작업보다 상대적으로 좋은 결과를 나타냅니다.



### Deep learning-based shot-domain seismic deblending (https://arxiv.org/abs/2409.08602)
- **What's New**: 이번 연구에서는 대량 데이터의 신속한 처리를 위해 심층 학습(deep learning) 접근 방식을 개발하였습니다. 이 접근 방식은 고품질 훈련 데이터 생성 전략과 데이터 드리븐(data-driven) 모델의 성능을 향상시키기 위한 데이터 조정(data conditioning) 기술을 포함합니다.

- **Technical Details**: 연구에서는 맨 마지막 sail line에서 수집된 비혼합(unblended) 샷 모음을 활용했습니다. 이 데이터에 대한 접근은 혼합된 수집(blended acquisition) 이상의 추가 시간이나 노동 비용이 들지 않습니다. 수동적으로 이 데이터를 혼합하여 정확한 ground truth에 잘 맞는 훈련 데이터를 얻습니다. 또한, 인접 혼합 샷 모음(blended shot gathers)을 추가 채널로 포함하는 다채널(multi-channel) 입력을 사용하여 심층 신경망(deep neural network)을 훈련합니다. 네트워크의 주요 작업은 주요 소스 이벤트를 예측하는 것이며, 혼합 노이즈(blending noise)의 예측은 보조 작업으로 추가됩니다. 훈련 및 검증 과정에서 혼합 노이즈의 진폭은 축소됩니다.

- **Performance Highlights**: 제안된 데이터 조정 단계를 도입한 결과, 혼합된 섹션의 깊은 부분에서 주요 소스 이벤트의 유출(leakage)을 상당히 줄일 수 있었습니다. 전체 제안 방식은 얕은 섹션에서 전통적인 알고리즘과 거의 동일한 성능을 보였고, 효율성에서 큰 장점을 보였습니다. 큰 여행 시간에 대해서는 약간 성능이 저하되었지만, 여전히 혼합 노이즈를 효과적으로 제거할 수 있었습니다.



### Large Language Model Can Transcribe Speech in Multi-Talker Scenarios with Versatile Instructions (https://arxiv.org/abs/2409.08596)
- **What's New**: 이 논문에서는 다중 화자(Multi-Talker) 환경에서 음성을 전사하는 데 있어 대형 언어 모델(LLM)의 능력을 조사한 최초의 연구를 제안합니다. 기존 Speech 관련 LLM 연구가 미흡했던 멀티-토커 시나리오에서 LLM이 사용될 가능성을 탐구하고자 합니다.

- **Technical Details**: MT-LLM(Multi-Talker LLM) 모델은 Whisper와 WavLM encoder를 활용하여 음성 정보를 다룰 수 있도록 설계되었습니다. WavLM은 음성 특징을 캡처하고 Whisper는 의미적 맥락을 이해하는 데 도움이 됩니다. 이 두 인코더에서 추출된 정보는 LLM에 입력되어 지시 사항 기반으로 음성을 전사할 수 있게 해줍니다. LoRA(프리미엄 파라미터 효율적 튜닝 기법)를 사용하여 LLM을 조정하고, 텍스트 명령어에 기반하여 음성을 인식하도록 최적화 되어 있습니다.

- **Performance Highlights**: MT-LLM은 ‘칵테일 파티’ 시나리오에서 다중 화자의 음성을 효과적으로 전사할 수 있으며, 사용자 지시에 따라 특정 특성을 가진 화자를 전사하는 뛰어난 성능을 보여줍니다. 실험 결과 MT-LLM은 다양한 요구 사항을 충족할 수 있음을 입증하였습니다.



### Automatic Generation of Fast and Accurate Performance Models for Deep Neural Network Accelerators (https://arxiv.org/abs/2409.08595)
Comments:
          Accepted version for: ACM Transactions on Embedded Computing Systems

- **What's New**: 본 논문에서는 리소스가 제한된 엣지 디바이스에서 Deep Neural Networks (DNN)의 성능을 정확하게 예측하기 위한 자동 생성 접근 방식을 제시합니다. 이 방법은 여러 다양한 하드웨어 가속기 아키텍처를 체계적으로 모델링하여 DNN의 지연 시간을 추정 가능하게 합니다.

- **Technical Details**: ABSTRACT COMPUTER ARCHITECTURE DESCRIPTION LANGUAGE (ACADL)을 통해 다양한 파라미터화된 가속기 아키텍처를 모델링합니다. Architectural Instruction Dependency Graph (AIDG)를 자동으로 생성하여 하드웨어 및 소프트웨어 의존성을 표현하며, 극소수의 루프 커널 반복만을 분석하여 성능을 추정합니다.

- **Performance Highlights**: 우리의 접근 방식은 평균 절대 백분율 오차(MAPE) 측면에서 기존의 회귀 및 분석 모델을 초월하며, RTL 시뮬레이션보다 몇 배 빠릅니다. 우리의 모델은 4.19억 명령어에 대해 단 154회 루프 커널 반복만으로 성능을 평가할 수 있습니다.



### LHQ-SVC: Lightweight and High Quality Singing Voice Conversion Modeling (https://arxiv.org/abs/2409.08583)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 LHQ-SVC라는 가벼운 CPU 최적화 모델을 제안합니다. 이 모델은 기존의 SVC 프레임워크와 확산 모델(diffusion model)을 기반으로 하여, 성능 저하 없이 모델 크기와 계산 요구사항을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: LHQ-SVC는 CPU에서 효율적으로 실행될 수 있도록 최적화되어 있으며, Intel MKL 및 OpenMP와 같은 최적화 라이브러리를 활용하여 다중 코어 CPU에서의 병렬 처리를 지원합니다. 또한, 변환 과정에서 필요한 단계 수를 줄여 샘플링 프로세스를 개선하였습니다.

- **Performance Highlights**: 실험 결과, LHQ-SVC는 다양한 장치에서 처리 속도 및 효율성이 크게 개선되었으며, 경쟁력 있는 성능을 유지하고 있음을 보여주었습니다.



### Molecular Graph Representation Learning via Structural Similarity Information (https://arxiv.org/abs/2409.08580)
- **What's New**: 이 논문에서는 분자 그래프에서 분자 간의 구조적 유사성을 포착할 수 있는 새로운 방법인 Molecular Structural Similarity Motif GNN (MSSM-GNN)을 제안합니다. 이 방법은 기존의 GNN들이 주로 개별 분자의 구조적 특성에 초점을 맞추었던 것과 달리, 분자 간의 관계를 고려하여 더 풍부한 정보를 제공합니다.

- **Technical Details**: MSSM-GNN은 화학 도메인 지식과 BRICS 알고리즘을 기반으로 설계된 그래프를 사용하여 분자 그래프를 표현합니다. 이 방법은 Mahalanobis Weisfeiler-Lehman Shortest-Path (MWLSP) 그래프 커널을 통해 구조적 유사성을 정량화하며, 두 가지 주요 구성 요소로 모티프 사전을 생성하고 이를 사용하여 분자 그래프를 재표현합니다.

- **Performance Highlights**: 다양한 실험을 통해 MSSM-GNN은 11개의 최신 기법과 비교하여 분자 속성 예측의 정확성을 지속적으로 향상시키는 성능을 보여주었습니다. 이러한 성과는 화학 구조와 분자 속성 간의 관계를 잘 포착할 수 있음을 시사합니다.



### Expediting and Elevating Large Language Model Reasoning via Hidden Chain-of-Thought Decoding (https://arxiv.org/abs/2409.08561)
- **What's New**: 이 논문에서는 기존의 Chain-of-Thought (CoT) 방법론의 컴퓨팅 비용 문제를 해결하기 위해 새로운 접근법인 Hidden Chain-of-Thought (HCoT) 모델을 제안합니다. 이 모델은 CoT 과정을 압축하여 효율적인 인퍼런스 (inference)를 가능하게 합니다.

- **Technical Details**: HCoT 모델은 CoT 모델의 파라미터를 고정한 채로 보조 CoT 모델을 훈련하여 압축된 토큰 표현을 생성하고, 이 표현은 HCoT 모델의 입력으로 통합됩니다. 훈련 과정은 두 단계로 진행되며, 첫 단계에서는 CoT 모델이 대비 손실 (contrastive loss)을 사용하여 압축된 토큰 표현을 생성하도록 최적화됩니다. 두 번째 단계에서는 HCoT 모델이 이 압축된 표현을 기반으로 후속 예측을 생성하도록 미세 조정됩니다.

- **Performance Highlights**: 다양한 데이터 세트에서 HCoT 모델의 성능을 평가한 결과, CoT 기준선 모델에 비해 경쟁력 있는 성능을 발휘하며 인코딩 시간에서 최소 1.5배에서 최대 3.8배의 속도 향상을 보였습니다. 이 연구는 다단계 이유 능력의 효율적인 활용 가능성을 제시합니다.



### ATFLRec: A Multimodal Recommender System with Audio-Text Fusion and Low-Rank Adaptation via Instruction-Tuned Large Language Mod (https://arxiv.org/abs/2409.08543)
- **What's New**: 본 연구는 멀티모달 데이터(모달리티), 즉 텍스트와 오디오를 대형 언어 모델(LLM)에 통합하여 추천 성능을 향상시키는 방안을 탐구합니다. 기존의 추천 시스템에서 발생하는 콜드 스타트 문제를 해결하고, 효율성을 잃지 않으면서 모델 성능을 유지하기 위해 Low-Rank Adaptation (LoRA) 방식을 도입했습니다.

- **Technical Details**: ATFLRec 프레임워크는 오디오 및 텍스트 데이터를 통합하여 추천 시스템을 구성하며, 여러 LoRA 구성과 모달리티 융합 기법을 활용합니다. 이를 통해 대형 언어 모델의 추천 과제를 효과적으로 조정할 수 있도록 설계되었습니다.

- **Performance Highlights**: ATFLRec은 전통적인 추천 모델 및 그래프 신경망 기반 접근법에 비해 AUC 점수가 높아 성능이 우수함을 입증하였으며, 유사한 LoRA 모듈을 통해 오디오 및 텍스트 데이터를 별도로 미세 조정한 결과 성능이 최적화되었습니다.



### SRE-CNN: A Spatiotemporal Rotation-Equivariant CNN for Cardiac Cine MR Imaging (https://arxiv.org/abs/2409.08537)
Comments:
          Accepted at MICCAI 2024

- **What's New**: 본 연구에서는 Spatiotemporal Rotation-Equivariant CNN (SRE-CNN)이라는 새로운 프레임워크를 제안하여, 동적 MRI 영상에서 고유의 회전 대칭을 최대한 활용하고자 하였습니다. 이 프레임워크는 공간 및 시간 차원의 회전 대칭을 동시에 활용하는 방법론을 포함하고 있습니다.

- **Technical Details**: SRE-CNN은 2D+t 동적 자료의 공간 및 시간 차원에서의 회전 대칭성을 활용하는 템포럴-에퀴바리언트(convolutional) 모듈을 설계하였습니다. 또한, 1D 및 2D 푸리에 급수 확장을 기반으로 한 고정밀 필터 매개변수화 전략을 활용하여 필터의 표현 정확도를 개선하였습니다.

- **Performance Highlights**: 제안된 방법은 29명의 피험자를 대상으로 한 심장 CINE MRI 데이터를 사용하여 훈련 및 테스트를 진행하였고, 기존 최첨단(reconstruction) 방법들과 비교했을 때 정량적 및 정성적으로 우수한 성능을 보였습니다. 특히, 고속 촬영 환경에서 더욱 효과적인 결과를 나타내었습니다.



### Integration of Mamba and Transformer -- MAT for Long-Short Range Time Series Forecasting with Application to Weather Dynamics (https://arxiv.org/abs/2409.08530)
Comments:
          6 pages, 4 figures, to be presented at the 5th International Conference on Electrical, Communication and Computer Engineering (ICECCE)

- **What's New**: 본 연구에서는 Mamba와 Transformer 모델의 장점을 결합한 혁신적인 방법 MAT를 제안합니다. MAT는 멀티변량 시계열 데이터에서 고유한 장단기 의존성과 진화 패턴을 파악하는 데 효과적입니다.

- **Technical Details**: MAT는 선택적 입력 처리 및 병렬 계산을 통해 장기 의존성을 효율적으로 포착하고, 선형 스케일러빌리티(linear scalability)와 최소 메모리 사용을 유지합니다. 모델은 다양한 스케일에서 맥락 정보를 사용하여 장단기 의존성을 캡처하며, 4개의 MAT 모듈을 활용하여 로컬 및 글로벌 맥락을 수집합니다.

- **Performance Highlights**: 실험적으로 MAT는 기존의 유사한 방법들과 비교했을 때 예측 정확도, 확장성 및 메모리 효율성에서 우수한 성능을 보여줍니다.



### Apollo: Band-sequence Modeling for High-Quality Audio Restoration (https://arxiv.org/abs/2409.08514)
Comments:
          Demo Page: this https URL

- **What's New**: 최신 연구에서는 Apollo 모델을 소개하며, 이 모델은 고샘플레이트 오디오 복원에 중점을 두고 개발되었습니다. Apollo는 음성 및 음악의 서로 다른 주파수 대역을 효과적으로 분리하고 복구하는 새로운 생성 모델로, 특히 복잡한 오디오 신호에서 뛰어난 성능을 보입니다.

- **Technical Details**: Apollo 모델은 주파수 대역 분할 모듈, 주파수 대역 시퀀스 모델링 모듈, 주파수 대역 복원 모듈로 구성되어 있습니다. Roformer 및 TCN (Temporal Convolutional Network)를 이용하여 시퀀스와 주파수 특성을 모델링하고, 손실된 오디오 정보를 보다 효과적으로 재구성합니다. 이 접근 방식은 특히 중저주파 정보를 보존하면서 고주파 세부 사항을 복원하는 데 중점을 두고 있습니다.

- **Performance Highlights**: MUSDB18-HQ 및 MoisesDB 데이터셋에서 Apollo는 기존 SR-GAN 모델들과 비교하여 다양한 비트레이트와 음악 장르에서 일관되게 우수한 성능을 보여주었습니다. Apollo는 특히 여러 악기와 보컬의 혼합이 포함된 복잡한 시나리오에서도 뛰어난 오디오 복원 품질을 제공합니다.



### Sub-graph Based Diffusion Model for Link Prediction (https://arxiv.org/abs/2409.08487)
Comments:
          17 pages, 3 figures

- **What's New**: 본 논문은 Denoising Diffusion Probabilistic Models (DDPMs)을 기반으로 하여 링크 예측을 위한 새로운 생성 모델을 제안합니다. 특히, 링크 예측을 공유 서브 그래프의 조건부 가능도 추정 문제로 간주하여, 베이지안(Bayesian) 공식을 이용하여 가능도 추정 과정을 분리하고, 서브 그래프 구조와 노드 특성을 효과적으로 구분하였습니다.

- **Technical Details**: 제안된 모델 SGDiff는 서브 그래프를 기반으로 한 확산 프레임워크로, 생성 훈련 과정에서 그래프 구조와 노드 특성을 분리하는 데 성공합니다. 이 모델은 노드의 크기와 상관없이 메모리 자원을 절약하고, 그래프의 구조와 노드 특성을 동시에 다룰 수 있도록 설계되었습니다.

- **Performance Highlights**: SGDiff는 다음과 같은 장점을 보여줍니다: (1) 재훈련 없이도 데이터 세트 간 전이 가능성, (2) 제한된 훈련 데이터에서 유망한 일반화 성능, (3) 그래프 적대적 공격에 대한 강력한 내구성.



### A BERT-Based Summarization approach for depression detection (https://arxiv.org/abs/2409.08483)
- **What's New**: 이 논문은 우울증을 효과적으로 감지하기 위한 새로운 접근 방식을 제안합니다. 기존 연구와는 달리, 텍스트 요약 기법을 사용하여 입력 텍스트의 길이와 복잡성을 줄임으로써 모델의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 기술적으로, BERT 기반 모델을 활용하여 텍스트를 수치적 표현으로 변환하고, 이를 통해 우울증 증상을 더욱 정확하게 진단할 수 있습니다. 또한, DAIC-WOZ 데이터셋에 있는 임상적으로 검증된 질문지를 바탕으로 자동화된 인터뷰를 통해 우울증의 지표를 감지합니다.

- **Performance Highlights**: 테스트 세트에서 F1-score 0.67을 기록하며 이전의 모든 기준을 초과하였고, 검증 세트에서는 0.81을 기록하여 DAIC-WOZ 데이터셋에서 대부분의 이전 결과를 초과하였습니다. 또한, 우울증 감지를 위한 요약 질과 관련성을 평가하는 새로운 우울증 용어집을 개발했습니다.



### Exploring Information Retrieval Landscapes: An Investigation of a Novel Evaluation Techniques and Comparative Document Splitting Methods (https://arxiv.org/abs/2409.08479)
Comments:
          This article is 16 pages long and includes detailed comparisons of RAG systems and document splitting techniques

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능에 문서의 특성이 미치는 영향을 분석하였습니다. 교과서, 기사, 소설의 구조적 특성이 각각 다른 정보 검색 전략을 필요로 한다는 점을 강조합니다.

- **Technical Details**: 문서 분할 방법의 비교 평가에서는 Recursive Character Splitter가 Token-based Splitter보다 맥락 유지(contextual integrity) 측면에서 더 우수함을 보여주었습니다. 새로운 평가 기법도 도입되어 오픈소스 모델을 활용하여 질문-답변 쌍 데이터셋을 생성하고, 현실적인 검색 시나리오를 시뮬레이션하여 테스트 효율성과 지표 신뢰성을 향상시킵니다.

- **Performance Highlights**: 평가에서는 SequenceMatcher, BLEU, METEOR, BERT Score 등의 가중치 기반 스코어링 메트릭을 사용하여 시스템의 정확성과 적합성을 평가하였습니다. 이 접근법은 RAG 시스템의 정밀도를 평가하는 정교한 기준을 설정하며, 앞으로의 연구는 검색 정확성과 효율성을 향상시키기 위해 청크(chunk) 및 중복(overlap) 크기를 최적화하는 데 초점을 맞출 것입니다.



### Integrating Neural Operators with Diffusion Models Improves Spectral Representation in Turbulence Modeling (https://arxiv.org/abs/2409.08477)
- **What's New**: 현재 연구에서는 neural operators와 diffusion models를 통합하여 난류 흐름의 surrogate modeling에서 spectral 한계를 극복하고자 합니다. 이 접근 방식은 다양한 데이터셋을 활용하여 검증되었으며, 전통적인 방법에 비해 에너지 스펙트럼 예측의 정확성을 향상시키는 데 성공했습니다.

- **Technical Details**: 논문에서는 neural operators의 spectral bias를 극복하기 위해 diffusion models을 조건화하여 사용하는 방법을 제시합니다. Neural operators는 복잡한 조건에서 빠른 추론을 가능하게 하지만, 고주파 성분을 제대로 예측하지 못하는 경향이 있습니다. 이를 해결하기 위해 score-based diffusion model을 활용하여 neural operator의 출력을 사전 정보로 사용합니다.

- **Performance Highlights**: 제안된 방법은 Reynolds 수가 2000에서 10^6 범위에 있는 다양한 테스트 케이스에서 효과적으로 작동하며, 2D 및 3D 대규모 에디 시뮬레이션과 실험적 Schlieren 속도 측정 데이터를 포함한 모든 실험에서 강력한 성능 향상을 보여줍니다.



### An Intent Modeling and Inference Framework for Autonomous and Remotely Piloted Aerial Systems (https://arxiv.org/abs/2409.08472)
Comments:
          8 pages, 7 figures, 3 tables

- **What's New**: 이 논문에서는 무인 항공 시스템(UAS)의 의도 모델링 및 추론 프레임워크를 제시하여 무단 비행으로부터 지리적 경계를 보호하기 위한 방안을 모색합니다. 새로운 수학적 정의와 함께, 중요한 경유점(critical waypoint) 개념을 도입하여 비행 경로를 완전히 설명합니다.

- **Technical Details**: 이 프레임워크는 2D 및 3D 환경에서 자율 비행, 반자율 비행 및 조종 비행 시스템에 적용 가능합니다. 시뮬레이션 연구의 일환으로 Attention 기반의 Bi-directional LSTM 네트워크를 사용한 의도 추론에 대한 실험을 포함하고 있습니다.

- **Performance Highlights**: 시뮬레이션 실험을 통해 경로 생성, 레이더 측정 시뮬레이션 등 다양한 측면을 보여주며, 의도 인식 분류기에 대한 학습을 위한 레이블이 있는 합성 데이터셋을 생성하는 방법도 제시합니다.



### Input-to-State Stable Coupled Oscillator Networks for Closed-form Model-based Control in Latent Spac (https://arxiv.org/abs/2409.08439)
Comments:
          41 pages, currently under review

- **What's New**: 이번 논문에서는 Coupled Oscillator Network (CON) 모델을 제안하여 물리적 시스템의 잠재 공간 제어를 효과적으로 달성하는 방법을 소개합니다.

- **Technical Details**: CON 모델은 라그랑지 시스템으로서, 명확한 잠재 에너지와 운동 에너지를 정의할 수 있으며, Lyapunov 논증을 사용하여 전역적인 Input-to-State 안정성을 증명합니다. 입력과 잠재 공간 외력 간의 역변환 매핑을 학습하기 위해 디코더를 훈련시키는 방법도 제안합니다.

- **Performance Highlights**: CON은 이미지로부터 복잡한 비선형 동역학을 직접 학습하여 60% 낮은 예측 오류를 달성하며, 제어 전략 구현 시 높은 품질의 성능을 보여줍니다. 또한, PID 제어기와 잠재 에너지 보상을 결합하여 짧은 응답 시간과 26% 개선된 궤적 추적 성능을 달성합니다.



### When Context Leads but Parametric Memory Follows in Large Language Models (https://arxiv.org/abs/2409.08435)
- **What's New**: 이 연구는 9개의 널리 사용되는 대형 언어 모델(LLMs)이 지식 일관성 시나리오에서 질문에 응답할 때 로컬 컨텍스트와 글로벌 파라미터 간의 지식을 어떻게 할당하는지를 분석합니다. 새롭게 소개된 데이터셋인 WikiAtomic을 통해 LLMs가 제공된 정보를 어떻게 우선시하고 활용하는지를 체계적으로 조사하였습니다.

- **Technical Details**: WikiAtomic 데이터셋은 200개의 위키피디아 기사를 기반으로 한 원자 문장들로 구성돼 있으며, 모델이 다양한 크기의 컨텍스트를 받았을 때 응답하는 경향을 분석합니다. LLMs는 응답에서 최대 30%의 파라메트릭 (parametric) 지식을 통합하고, 작은 컨텍스트에선 모든 부분의 정보를 기억하나 큰 컨텍스트에선 주로 첫 번째 절반에 집중합니다.

- **Performance Highlights**: 모델들은 약 70%의 컨텍스트 (contextual) 지식과 30%의 파라메트릭 지식을 일관되게 의존하며, 컨텍스트가 늘어남에 따라 환각(hallucination) 발생률은 감소하는 경향이 있습니다. 이 연구 결과는 효과적인 컨텍스트 조직과 예측 가능한 입력 사용 모델 개발의 중요성을 강조합니다.



### Fitted Q-Iteration via Max-Plus-Linear Approximation (https://arxiv.org/abs/2409.08422)
- **What's New**: 이번 연구에서는 할인된 마르코프 결정 과정(Discounted Markov Decision Processes)에 대한 오프라인 강화 학습에서 max-plus-linear 근사기(Max-Plus-Linear Approximators)를 적용하는 방법을 제안합니다. 특히, 우리는 이 근사기를 사용하여 수렴성이 증명된 새로운 적합 Q-반복(Fitted Q-Iteration, FQI) 알고리즘을 제안합니다.

- **Technical Details**: Bellman 연산자가 max-plus 연산(max-plus operations)과 호환됨을 활용하여, 제안된 FQI 알고리즘의 각 반복(iteration) 내에서 max-plus-linear 회귀(max-plus-linear regression)이 간단한 max-plus 매트릭스-벡터 곱셈(max-plus matrix-vector multiplications)으로 축소됨을 보여줍니다. 우리는 또한 제안된 알고리즘의 변분 구현(variational implementation)을 고려하여, 샘플 수와 무관한 반복 복잡성을 가집니다.

- **Performance Highlights**: 제안된 MP-FQI 알고리즘은 선형 속도(linear rate)로 수렴하고, 반복 복잡성은 𝒪(n⁢p)로, 여기서 n은 샘플 수, p는 MP-linear 근사기의 매개변수/기저 함수 수입니다. 또한, 변variational MP-FQI 알고리즘은 선형 속도로 수렴하며, 복잡성은 𝒪(p⁢q)로, 여기서 p는 매개변수 수, q는 변분 형식에서의 테스트 함수 수입니다.



### Knowledge Tagging with Large Language Model based Multi-Agent System (https://arxiv.org/abs/2409.08406)
Comments:
          8 pages, 3 figures

- **What's New**: 이 연구는 자동화된 지식 태깅(knowledge tagging) 프로세스를 위한 다중 에이전트 시스템(multi-agent system)을 도입합니다. 이 시스템은 이전 알고리즘의 한계를 극복하는 데 중점을 두며, 복잡한 사례 처리에 있어 LLMs(large language models)를 활용합니다.

- **Technical Details**: 제안된 LLM 기반 다중 에이전트 시스템은 네 가지 유형의 LLM 에이전트(task planner, question solver, semantic judger, numerical judger)로 구성되어 있습니다. 각 에이전트는 독립적인 하위 문제를 처리하며, 계획 에이전트가 задан된 지식 정의에 맞춰 협력 계획을 제안합니다. 최종적으로, 중간 결과를 AND 연산자로 연결하여 최종 판단을 출력합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 공개된 수학 문제 지식 태깅 데이터세트 MathKnowCT에서 이전 단일 LLM 기반 방법보다 일관된 성능 개선을 보였습니다. 이는 교육 환경에서 LLM 기반 알고리즘의 배치에 대한 유망한 결과를 강조합니다.



### Scores as Actions: a framework of fine-tuning diffusion models by continuous-time reinforcement learning (https://arxiv.org/abs/2409.08400)
- **What's New**: 이번 연구에서는 인공지능 생성 모델과 인간의 선호도를 맞추기 위해 강화 학습(RL, Reinforcement Learning) 접근 방식을 활용한 지속적인 시간 스토캐스틱 제어 문제에 대한 체계적인 프레임워크를 제안합니다. 또한, 기존의 이산 시간 최적화 접근법의 한계를 극복하여 연속 시간 샘플러를 효과적으로 이용할 수 있는 방법을 다룹니다.

- **Technical Details**: 연구는 스코어 맞춤 함수(score-matching functions)를 조작(control/action)으로 다루어 연속 시간 관점에서 RL 알고리즘을 통합하는 방법론을 제시합니다. 이를 통해 강화 학습의 정책 최적화 이론을 확립하고, 스토캐스틱 미분 방정식(SDE) 환경하에서의 정책 최적화 방법을 개발했습니다. 또한, DDIM이나 Rectified Flow와 같은 결정론적 샘플러에 적합한 알고리즘을 만들었습니다.

- **Performance Highlights**: 실험 결과는 텍스트-투-이미지(T2I) 생성에서의 성능 향상을 보여줄 예정이며, 이전의 이산 시간 강화 학습 알고리즘보다 연속 시간 접근 방식의 우수성을 입증할 것으로 기대됩니다.



### 360PanT: Training-Free Text-Driven 360-Degree Panorama-to-Panorama Translation (https://arxiv.org/abs/2409.08397)
Comments:
          Accepted by WACV 2025, Project Page: \href{this https URL}{this https URL}

- **What's New**: 본 연구에서는 360도 파노라마 이미지를 위한 최초의 훈련 없는 텍스트 기반 이미지 전환 방법인 360PanT를 제안합니다. 이 방법은 경계 연속성 인코딩(boundary continuity encoding)과 공간 제어가 포함된 원활한 타일 번역(seamless tiling translation)의 두 가지 주요 요소로 구성됩니다.

- **Technical Details**: 360PanT는 노이즈가 있는 잠재 표현(noisy latent representation)에 입력 360도 파노라마의 경계 연속성 정보를 내장하며, 이 정보를 바탕으로 범위 제어된 원활한 타일 번역을 수행합니다. 경계 연속성 인코딩을 통해 입력 이미지의 구조와 의미 배치를 유지하며, 노이즈 특성에 따라 자른 패치(cropped patches)를 독립적으로 처리합니다. 이를 통해 360도 파노라마의 경계를 효과적으로 유지합니다.

- **Performance Highlights**: 실험 결과, 360PanT는 실제 및 합성 데이터셋 모두에서 360도 파노라마 번역에서 탁월한 성능을 보여주며, 전통적인 I2I 전환 방법의 한계를 뛰어넘는 경계 연속성을 달성했습니다.



### Self-Supervised Inference of Agents in Trustless Environments (https://arxiv.org/abs/2409.08386)
- **What's New**: 본 논문에서는 에이전트들이 스워밍(swarming)을 형성하여 고품질 응답을 효과적으로 생성하는 새로운 접근 방식을 제안합니다. LLMs(Large Language Models)를 활용하여 신뢰할 수 없는(agent inference) 에이전트 추론을 평가하고, 다양한 유형의 악의적(agent) 공격을 모델링합니다.

- **Technical Details**: 제안된 방법은 스와의 집합 지능을 활용하여 강력하고 효율적인 분산 AI 추론을 보장합니다. 본 논문은 멀티 에이전트 아키텍처를 통해 각 에이전트가 데이터 추론 및 품질 평가를 수행하며, 응답 생성, 선택적 순위 매기기 및 최종 선택이라는 세 가지 주요 단계로 구성된 스와 공세 메커니즘을 설명합니다.

- **Performance Highlights**: 제안된 접근 방식은 다른 신뢰할 수 없는 추론 전략에 비해 125 ms 이하의 검증 지연(latency)으로 한 차원 빠른 성능을 보입니다.



### Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations (https://arxiv.org/abs/2409.08381)
- **What's New**: 이 논문에서는 Vision-Language 모델(VLM)인 CLIP을 Multi-Label Recognition (MLR)에 대한 새로운 접근법으로 활용합니다. 특히, PositiveCoOp와 NegativeCoOp라는 두 가지 방법을 소개하여 양수 및 음수 프롬프트 학습의 효과를 분석합니다.

- **Technical Details**: PositiveCoOp에서는 클래스를 나타내는 양수 프롬프트 하나를 학습하고, 음수 프롬프트 대신 VLM의 텍스트 인코더에 의존하지 않고 이미지 특징과 연관된 임베딩 벡터를 직접 학습합니다. NegativeCoOp에서는 반대로 진행합니다. 이러한 접근 방식은 dual prompt 학습보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: PositiveCoOp 방법이 DualCoOp보다 뛰어난 성능을 보였으며, 전체 레이블 비율이 높을 때 baseline의 성능은 DualCoOp 및 PositiveCoOp에 필적할 정도로 강력했습니다. 이 baseline은 DualCoOp보다 약 16배 적은 훈련 매개변수와 절반의 훈련 시간만 필요로 합니다.



### The Impact of Large Language Models on Open-source Innovation: Evidence from GitHub Copilo (https://arxiv.org/abs/2409.08379)
Comments:
          JEL Classification: O31, C88, J24, O35, L86

- **What's New**: 본 연구는 Generative AI(GenAI)가 협업 환경에서의 혁신적인 작업 방식을 어떻게 변화시키는지에 대한 실증적 질문을 다룹니다. 특히, GitHub Copilot의 출시가 오픈 소스 개발 커뮤니티에 미친 영향을 분석합니다.

- **Technical Details**: 연구는 2021년 10월 GitHub Copilot의 출시를 중심으로 진행되었습니다. Copilot은 프로그래밍 중심의 LLM(large language model)으로, Python에 대한 지원을 선택적으로 배포했으며, R에 대한 지원은 제공하지 않았습니다. 이 자연 실험을 통해 기여가 전반적으로 증가했음을 관찰했습니다.

- **Performance Highlights**: Copilot의 출시는 유지 관리 관련 기여가 크게 증가했음을 보여주었고, 이는 대체로 다른 작업 위에 구축하는 반복적(Iterative) 작업에 해당합니다. 그러나 코드 개발 기여는 상대적으로 적은 증가를 보였습니다. 이러한 차이는 주로 코드 활동이 활발한 프로젝트에서 더욱 두드러졌습니다.



### FedProphet: Memory-Efficient Federated Adversarial Training via Theoretic-Robustness and Low-Inconsistency Cascade Learning (https://arxiv.org/abs/2409.08372)
Comments:
          Preprint

- **What's New**: 본 논문에서는 메모리 효율성과 적대적 견고성(adversarial robustness), 목표 일관성(objective consistency)을 동시에 달성할 수 있는 새로운 연합 적대적 훈련(Federated Adversarial Training, FAT) 프레임워크인 FedProphet를 제안합니다.

- **Technical Details**: FedProphet는 대형 모델을 작은 모듈들로 나누어 메모리 제약이 있는 엣지 장치에서 모듈별로 적대적 훈련을 수행할 수 있도록 합니다. 강한 볼록성 정규화(strong convexity regularization)를 통해 전체 모델의 견고성을 이론적으로 보장하며, 강한 견고성은 낮은 목표 불일치(objective inconsistency)를 의미합니다. 서버 측에서는 적응형 방해 요소 조정(Adaptive Perturbation Adjustment) 및 차별화된 모듈 할당(Differentiated Module Assignment)을 개발하여 목표 불일치를 완화합니다.

- **Performance Highlights**: FedProphet는 이전의 메모리 효율적인 방법에 비해 정확성과 견고성 모두에서 큰 향상을 보여주며, 메모리를 80% 절약하고 훈련 시간을 최대 10.8배 단축하면서도 대형 모델을 엔드-투-엔드로 훈련한 것과 거의 동일한 성능을 달성합니다.



### E-QUARTIC: Energy Efficient Edge Ensemble of Convolutional Neural Networks for Resource-Optimized Learning (https://arxiv.org/abs/2409.08369)
Comments:
          Accepted by the 30th Asia and South Pacific Design Automation Conference (ASP-DAC 2025)

- **What's New**: 본 연구에서는 E-QUARTIC이라는 새로운 에너지 효율적인 엣지 앙상블 프레임워크를 제안합니다. 이 프레임워크는 에너지 수확을 통해 파워를 공급받는 임베디드 시스템을 위해 특별히 설계되었습니다. E-QUARTIC은 CNNs의 앙상블을 구축하여 단일 CNN 기반선보다 뛰어난 성능을 보여주며, 에너지 조건에 적응하고 비슷한 메모리 요구 사항을 유지합니다.

- **Technical Details**: E-QUARTIC은 필터 레벨 가지치기(filter-level pruning) 기법과 부스팅 알고리즘(boosting algorithms)을 결합하여 높은 정확도를 유지하면서 컴퓨팅 복잡성과 메모리 요구 사항을 늘리지 않는 적응형 앙상블을 만듭니다. 이 프레임워크는 에너지 인식 스케줄러(energy-aware scheduler)를 통해 추론 과정에서 에너지와 관련된 요소를 고려하여 모델의 정확성과 에너지 효율성의 균형을 맞춥니다.

- **Performance Highlights**: E-QUARTIC은 기존의 최첨단 방법들과 비교하여 출력 품질이 유사하거나 약간 높은 성과를 보여주며, 시스템 실패율을 최대 40%까지 줄이는 것으로 나타났습니다. 또한, 현재의 마이크로컨트롤러에서 구현되었으며, 동적 추론과 적응을 위한 에너지 수확 센서를 갖춘 환경에서 평가되었습니다.



### An Experimental Study of Competitive Market Behavior Through LLMs (https://arxiv.org/abs/2409.08357)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 시장 실험 수행 가능성을 탐구하며, 경쟁 시장 역학을 이해하기 위한 능력을 진단합니다.

- **Technical Details**: 본 연구는 시장 에이전트의 행동을 제어된 실험 환경에서 모델링하고, 경쟁 균형(competitive equilibrium)으로 수렴하는 능력을 평가합니다. 특히 ChatGPT-4.0을 활용하여 복잡한 시장 상호작용을 시뮬레이션하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 현재 LLMs는 동적 의사결정 과정에서 인간의 거래 행동을 복제하는 데 어려움을 겪고 있으며, 시장 균형을 달성하는 능력이 제한적입니다. 향후 동적 학습 능력 강화를 통해 경제적 행동의 복잡성을 효과적으로 모델링할 필요가 있습니다.



### Towards Quantifying and Reducing Language Mismatch Effects in Cross-Lingual Speech Anti-Spoofing (https://arxiv.org/abs/2409.08346)
Comments:
          Accepted to the IEEE Spoken Language Technology Workshop (SLT) 2024. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구는 영어 데이터로 훈련된 스피치 안티스푸핑(anti-spoofing) 시스템이 다른 언어로 테스트할 때 성능이 저하된다는 점을 발견했습니다. 이를 개선하기 위해 TTS(Text-to-Speech)를 활용하여 여러 언어의 억양(accent)을 반영한 데이터를 생성하는 새로운 방법, ACCENT를 제안했습니다.

- **Technical Details**: ACCENT 방법은 기계 학습을 위해 독일어, 일본어 등 12개 언어로 구성된 300만 건 이상의 샘플을 포함한 대규모 데이터 세트를 사용하여 테스트되었습니다. 기존 리소스가 적은 언어에 대한 데이터 수집의 어려움을 극복하고 다국어 간의 성능 차이를 줄였습니다. 또한, 주요 TTS 모델을 통해 다양한 억양을 통합하여 영문 데이터의 풍부함을 증가시킵니다.

- **Performance Highlights**: ACCENT 방법을 적용하면서 언어 격차로 인한 성능 저하가 15% 이상 감소했습니다. 기존의 모델이 부족한 다국어 대응 능력을 개선하여 다양한 영어 억양에 대한 강건성을 보여주었습니다. 이 연구는 언어 독립적인 안티스푸핑 모델 개발에 대한 새로운 방향을 제시합니다.



### DiReDi: Distillation and Reverse Distillation for AIoT Applications (https://arxiv.org/abs/2409.08308)
- **What's New**: 이 논문은 'DiReD'라는 혁신적인 프레임워크를 제안하며, 이는 Knowledge Distillation (KD) 및 Reverse Distillation (RD)을 활용하여 엣지 AI 모델을 업데이트하는 방법을 다룬다. 이를 통해 사용자의 실제 데이터 없이도 사용자의 선호에 맞춘 모델 업데이트가 가능하다.

- **Technical Details**: 제안된 DiReD 프레임워크는 엣지 AI 모델이 사용자 시나리오에 맞게 업데이트될 수 있도록 지식을 추출하는 과정을 포함한다. 첫 번째 단계는 클라우드 AI 모델을 사용하여 엣지 AI 모델을 학습시키는 KD 과정이며, 이후 사용자의 요구에 맞게 RD 과정을 이용해 모델 지식을 갱신하는 방식으로 이루어진다.

- **Performance Highlights**: 시뮬레이션 결과, DiReD 프레임워크는 사용자 모델을 업데이트 할 수 있으며, 새로운 지식을 포함함으로써 더 높은 정확도를 달성할 수 있음을 보여준다. 또한 불필요한 지식은 제거되어 모델의 효율성이 증가한다.



### Comparative Study of Long Short-Term Memory (LSTM) and Quantum Long Short-Term Memory (QLSTM): Prediction of Stock Market Movemen (https://arxiv.org/abs/2409.08297)
- **What's New**: 이번 연구에서는 파키스탄의 경제, 사회, 정치적 불확실성을 고려하여, 주가 지수 예측 모델로 Long Short-Term Memory (LSTM) 및 Quantum Long Short-Term Memory (QLSTM) 알고리즘을 적용하였습니다.

- **Technical Details**: 연구는 2004년 2월부터 2020년 12월까지의 26개 경제, 사회, 정치 및 행정 지표의 월별 데이터를 사용하여 KSE 100 지수의 예측 모델을 구축했습니다. 주요 모델인 LSTM과 QLSTM의 성능을 비교 분석하였습니다.

- **Performance Highlights**: LSTM과 QLSTM의 예측 값과 실제 KSE 100 지수의 값 비교 결과, QLSTM이 주식 시장 동향 예측에 있어 유망한 기술로 나타났습니다.



### Reconsidering the energy efficiency of spiking neural networks (https://arxiv.org/abs/2409.08290)
- **What's New**: 이 논문에서는 스파이킹 신경망(Spiking Neural Networks, SNNs)의 에너지 소비를 하드웨어 관점에서 인공지능 신경망(Artificial Neural Networks, ANNs)과 비교하여 보다 정확한 에너지 소비 공식을 제시합니다.

- **Technical Details**: 연구에서 고전적인 다중 레벨 메모리 계층 아키텍처 및 신경형 데이터 흐름 아키텍처를 기반으로 한 에너지 소비 수식을 제공하며, SNN의 에너지를 효율적으로 줄이기 위한 새로운 공간-데이터 흐름 아키텍처를 제안합니다. 또한, 필요 sparsity와 time window size에 대한 엄격한 제한을 도입하여 SNN이 ANNs보다 에너지 효율성을 가지도록 합니다.

- **Performance Highlights**: VGG16 모델과 함께 고정 T가 6인 경우, SNN은 최적화된 ANN이 사용하는 에너지의 69%만 소비하면서 94.18%의 정확도를 유지하였습니다. 두 가지 정규화 항을 도입하여 sparsity 비율을 증가시킴으로써 CIFAR-10 데이터셋에서 SNN이 ANNs보다 78% 에너지를 절감할 수 있음을 증명하였습니다.



### StockTime: A Time Series Specialized Large Language Model Architecture for Stock Price Prediction (https://arxiv.org/abs/2409.08281)
- **What's New**: 본 논문에서는 주식 가격 예측을 위한 새로운 LLM 기반 아키텍처인 StockTime을 소개합니다. 기존의 금융 LLM과 달리, StockTime은 주식 가격 데이터에 특화된 설계를 가지고 있으며, 주식 가격을 연속적인 토큰으로 처리하여 예측에 활용합니다.

- **Technical Details**: StockTime은 시간 시계열 데이터와 자연어 데이터를 통합하여 임베딩 공간에서 융합합니다. 이를 통해 주식 가격의 상관관계, 통계적 추세 및 타임스탬프 정보를 추출하고, 자가 회귀 인코더를 사용하여 주식 가격의 시간적 정보를 캡처합니다.

- **Performance Highlights**: StockTime은 실험에서 최근 LLM들보다 더 정확한 예측 결과를 보이며, 메모리 사용량과 런타임 비용을 줄이는데 성공했습니다. 다양한 주식 데이터 셋을 기반으로 퍼포먼스를 검증하였습니다.



### Large Language Models are Pattern Matchers: Editing Semi-Structured and Structured Documents with ChatGP (https://arxiv.org/abs/2409.07732)
- **What's New**: 이 논문은 Large Language Models(LLMs)가 구조화된(document) 및 반구조화된(semi-structured) 문서를 최소한의 노력으로 편집할 수 있는지를 조사합니다. 연구 결과 ChatGPT는 기본적인 프롬프트를 통해 이러한 문서들을 효과적으로 수정할 수 있는 능력을 보여 주며, 이에 대한 패턴 매칭 기법도 눈여겨볼 만하다고 보고합니다.

- **Technical Details**: 논문에서는 LLM이 이미 구조화된 텍스트를 얼마나 잘 처리하거나 재구성할 수 있는지에 대한 연구가 이루어졌습니다. 이 연구는 질적 연구 접근 방식을 채택하여 LaTeX로 포맷된 문서와 같은 다양한 문서 형식에서 두 가지 케이스 스터디를 수행했습니다. LLM은 입력 내용을 처리하며 주어진 프롬프트의 구조를 인식하고 활용하는 능력을 보여 주었습니다.

- **Performance Highlights**: 실험 결과, LLM은 구조화된 문서 편집 작업에서 큰 품질을 제공하며, 사용자는 최소한의 수작업 후처리만으로 우수한 결과를 얻을 수 있습니다. 연구자는 LLM이 문서 형식 간 변환을 지원하고, 기존 프로그래밍 방식보다 더 유연한 솔루션을 제공할 수 있음을 나타냈습니다.




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



